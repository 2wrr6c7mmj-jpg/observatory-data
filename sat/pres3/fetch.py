"""Step 1: fetch the Sentinel-2 L2A windows of the ports (Element84 Earth Search -> the public COGs on AWS, no login).

Bands blue, green, red, nir (10 m), swir16, scl (20 m), a window of the site's bbox + 0.004 deg, one .npz per scene in
<work>/data/<port>/. A scene already in the directory is not fetched again.

Two ways to choose the scenes:
  default            the scenes listed in sat/pres3/inputs/scenes.csv (the 2,517 scenes the published numbers were computed from)
  --start/--end      a search by date: the MGRS tile with the most scenes, the last scene of each day, cloud cover <= --max-cloud
                     (this is how the list was first made: 2024-10-01..2026-11-19 and 2023-06-01..2024-09-30, max cloud 85)

A refusal (HTTP 4xx, 403 and 429 included; also a proxy's refusal, which urllib gives as a URLError with the status in
its reason) and a certificate that does not verify are not sent again; only a transient failure (no connection, cut off,
timeout, 5xx) is retried, a fixed number of times (SEARCH_TRIES, WINDOW_TRIES, GDAL_MAX_RETRY below).

Contains modified Copernicus Sentinel data.
    python -m sat.pres3.fetch --work DIR [--ports QDH,XSN] [--start 2024-10-01 --end 2026-11-19 [--max-cloud 85]]
"""
from __future__ import annotations

import collections
import csv
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import common

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = {'blue': 10, 'green': 10, 'red': 10, 'nir': 10, 'swir16': 20, 'scl': 20}


# Retries. A refusal (any 4xx: 403, 429, ...) is never sent again: it is reported as it is (the project's rule: no way around
# a refusal or a bot check). Only a transient failure is retried: no connection, a connection cut in the middle, a timeout, a 5xx.
SEARCH_TRIES = 4                  # the STAC search: 1 request + at most 3 re-sends of a transient failure
SEARCH_DELAYS_S = (5, 10, 15)     #   seconds before the 1st, 2nd, 3rd re-send
WINDOW_TRIES = 3                  # the window of a scene (fetch.main): at most 2 more tries after a transient failure
WINDOW_DELAYS_S = (4, 8)
# GDAL (the COG reads): GDAL_HTTP_MAX_RETRY re-sends, the first after GDAL_HTTP_RETRY_DELAY s, then about x2..x2.5 each time.
# GDAL_HTTP_RETRY_CODES (GDAL >= 3.10) REPLACES GDAL's default list (429, 500, 502-504 and some curl errors): port/cpl_http.cpp,
# CPLHTTPGetNewRetryDelay. GDAL matches the code as a substring of this text (strstr), so this list must not contain the digits of
# any 4xx code; a failure without an HTTP status (no connection, cut off: code 0) is retried, as "0" is in it. Older GDAL ignores
# the option and would retry 429: there, no retry at all (gdal_env).
GDAL_MAX_RETRY = 3
GDAL_RETRY_DELAY_S = 3
GDAL_RETRY_CODES = '500,502,503,504'


def gdal_env(gdal_version=None):
    """The GDAL options of the COG reads. gdal_version: rasterio.__gdal_version__ (default: the installed one)."""
    if gdal_version is None:
        import rasterio
        gdal_version = rasterio.__gdal_version__
    codes_known = tuple(int(x) for x in re.findall(r'\d+', gdal_version)[:2]) >= (3, 10)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_TIMEOUT='60',
               GDAL_HTTP_MAX_RETRY=str(GDAL_MAX_RETRY if codes_known else 0), GDAL_HTTP_RETRY_DELAY=str(GDAL_RETRY_DELAY_S),
               GDAL_HTTP_RETRY_CODES=GDAL_RETRY_CODES)
    ca = os.environ.get('CURL_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')     # a proxy's CA bundle, when the machine has one
    if ca: env['CURL_CA_BUNDLE'] = ca
    return env


# a 4xx status in the text of an error without an HTTP status (a proxy's "Tunnel connection failed: 403 Forbidden"); not a
# port (":443") or a part of a number
STATUS_4XX = re.compile(r'(?<![:\d.\-])4\d\d(?![\d.])')
# a certificate that does not verify (urllib / ssl, and curl inside GDAL): sending again gives the same answer
CERT_FAILURE = re.compile(r'CERTIFICATE_VERIFY_FAILED|certificate verify failed|SSL certificate problem|certificate has expired|'
                          r'self[- ]signed certificate|unable to get local issuer certificate|hostname mismatch', re.I)


def transient(e):
    """True for a failure that may be retried: no connection, a connection cut in the middle, a timeout, an HTTP 5xx.
    False for a refusal (HTTP 4xx, also one a proxy gives as the reason of a URLError), a certificate that does not verify,
    and anything else (a bad answer, a bug)."""
    if isinstance(e, urllib.error.HTTPError): return e.code >= 500
    if isinstance(e, ssl.SSLCertVerificationError): return False
    if isinstance(e, urllib.error.URLError):                                  # no HTTP status: the connection failed, unless
        r = e.reason                                                          # the reason is a refusal or a certificate
        if isinstance(r, ssl.SSLCertVerificationError) or CERT_FAILURE.search(str(r)) or STATUS_4XX.search(str(r)): return False
        return True
    if isinstance(e, (TimeoutError, ConnectionError, http.client.IncompleteRead, http.client.RemoteDisconnected)): return True
    msg = str(e)                                                              # GDAL / rasterio: the HTTP status is in the text
    if CERT_FAILURE.search(msg): return False
    if re.search(r'\b4\d\d\b', msg): return False                            # a 4xx anywhere in it: taken as a refusal
    return bool(re.search(r'response code:?\s*5\d\d|HTTP error code:?\s*5\d\d|CURL error|timed out|Connection reset|'
                          r'Connection refused|Could not resolve|transfer closed', msg))


def with_retries(fn, tries, delays, sleep=time.sleep):
    """fn() with at most tries - 1 re-sends of a transient failure (delays[k] s before the (k+1)-th); a refusal is raised at once."""
    for k in range(tries):
        try: return fn()
        except Exception as e:
            if k == tries - 1 or not transient(e): raise
            sleep(delays[k])


def search(body, urlopen=None, sleep=time.sleep):
    out = []; body = dict(body, collections=["sentinel-2-l2a"], limit=100); urlopen = urlopen or urllib.request.urlopen
    while True:
        r = urllib.request.Request(SEARCH_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        j = with_retries(lambda: json.load(urlopen(r, timeout=90)), SEARCH_TRIES, SEARCH_DELAYS_S, sleep)
        out += j['features']
        nx = [l for l in j.get('links', []) if l['rel'] == 'next']
        if not nx: break
        body = nx[0]['body']; sleep(0.5)
    return out


def one(work, port, f, bb):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    fid = f['id']; dst = f'{work}/data/{port}/{fid}.npz'
    if os.path.exists(dst): return 'cached'
    pad = 0.004; b4 = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]
    arrs = {}; meta = {}
    with rasterio.Env(**gdal_env()):
        for name, res in BANDS.items():
            with rasterio.open(f['assets'][name]['href']) as ds:
                b = transform_bounds('EPSG:4326', ds.crs, *b4)
                # the window, snapped to the 20 m grid
                x0 = np.floor(b[0] / 20) * 20; y0 = np.floor(b[1] / 20) * 20; x1 = np.ceil(b[2] / 20) * 20; y1 = np.ceil(b[3] / 20) * 20
                w = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
                arrs[name] = ds.read(1, window=w, boundless=True, fill_value=0)
                if name == 'blue': meta = dict(crs=str(ds.crs), x0=float(x0), y1=float(y1))
    p = f['properties']
    meta.update(id=fid, datetime=p['datetime'], cloud=p.get('eo:cloud_cover'), baseline=p.get('s2:processing_baseline'),
                boa=p.get('earthsearch:boa_offset_applied'), platform=p.get('platform'))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst + '.tmp.npz', meta=json.dumps(meta), **arrs); os.replace(dst + '.tmp.npz', dst)
    return 'ok'


def pick_by_date(port, bb, t0, t1, maxcc):
    fs = search({"bbox": bb, "datetime": f"{t0}T00:00:00Z/{t1}T23:59:59Z"})
    tiles = collections.Counter(f['properties'].get('grid:code') for f in fs); tile = sorted(tiles, key=lambda t: (-tiles[t], t))[0]
    pick = {}
    for f in fs:
        if f['properties'].get('grid:code') != tile: continue
        d = f['properties']['datetime'][:10]
        if d not in pick or f['id'] > pick[d]['id']: pick[d] = f
    use = [f for f in pick.values() if (f['properties'].get('eo:cloud_cover') or 0) <= maxcc]
    print(port, 'tiles', dict(tiles), 'tile', tile, flush=True)
    print(port, 'found', len(fs), 'use', len(use), flush=True)
    return use


def manifest(port):
    with open(common.INPUTS / 'scenes.csv', newline='') as fh:
        return [r['scene_id'] for r in csv.DictReader(fh) if r['port'] == port]


def pick_by_manifest(work, port):
    ids = [i for i in manifest(port) if not os.path.exists(f'{work}/data/{port}/{i}.npz')]
    fs = []
    for k in range(0, len(ids), 100): fs += search({"ids": ids[k:k + 100]})
    got = {f['id'] for f in fs}
    print(port, 'listed', len(manifest(port)), 'to fetch', len(ids), 'found', len(got), 'not in the catalogue', sorted(set(ids) - got)[:5], flush=True)
    return fs


def fetch_one(work, port, f, bb, read=None, sleep=time.sleep):
    """one() with the retries of a transient failure (WINDOW_TRIES); 'ERR <id> <error>' when it fails or is refused."""
    read = read or one
    try: return with_retries(lambda: read(work, port, f, bb), WINDOW_TRIES, WINDOW_DELAYS_S, sleep)
    except Exception as e: return 'ERR ' + f['id'] + ' ' + repr(e)[:150]


def main(argv=None):
    ap = common.parser(__doc__.split('\n')[0])
    ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--max-cloud', type=float, default=85)
    a = ap.parse_args(argv); work = common.work_dir(a.work)
    if bool(a.start) != bool(a.end): ap.error('--start and --end go together')
    sites = {x['id']: x for x in json.load(open(common.REPO / 'sat' / 'sites.json'))['sites']}
    bad = 0
    for port in common.ports_of(a.ports):
        bb = sites[port]['bbox']
        use = pick_by_date(port, bb, a.start, a.end, a.max_cloud) if a.start else pick_by_manifest(work, port)

        with ThreadPoolExecutor(5) as ex:
            res = list(ex.map(lambda f: fetch_one(work, port, f, bb), use))
        errs = [r for r in res if r.startswith('ERR')]; bad += len(errs)
        print(port, dict(collections.Counter(r.split()[0] for r in res)), errs[:5], flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
