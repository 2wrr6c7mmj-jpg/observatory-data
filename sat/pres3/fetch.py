"""Step 1: fetch the Sentinel-2 L2A windows of the ports (Element84 Earth Search -> the public COGs on AWS, no login).

Bands blue, green, red, nir (10 m), swir16, scl (20 m), a window of the site's bbox + 0.004 deg, one .npz per scene in
<work>/data/<port>/. A scene already in the directory is not fetched again.

Two ways to choose the scenes:
  default            the scenes listed in sat/pres3/inputs/scenes.csv (the 2,517 scenes the published numbers were computed from)
  --start/--end      a search by date: the MGRS tile with the most scenes, the last scene of each day, cloud cover <= --max-cloud
                     (this is how the list was first made: 2024-10-01..2026-11-19 and 2023-06-01..2024-09-30, max cloud 85)

Contains modified Copernicus Sentinel data.
    python -m sat.pres3.fetch --work DIR [--ports QDH,XSN] [--start 2024-10-01 --end 2026-11-19 [--max-cloud 85]]
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import common

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = {'blue': 10, 'green': 10, 'red': 10, 'nir': 10, 'swir16': 20, 'scl': 20}


def gdal_env():
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='4',
               GDAL_HTTP_RETRY_DELAY='3', GDAL_HTTP_TIMEOUT='60')
    ca = os.environ.get('CURL_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')     # a proxy's CA bundle, when the machine has one
    if ca: env['CURL_CA_BUNDLE'] = ca
    return env


def search(body):
    out = []; body = dict(body, collections=["sentinel-2-l2a"], limit=100)
    while True:
        r = urllib.request.Request(SEARCH_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        for k in range(4):
            try: j = json.load(urllib.request.urlopen(r, timeout=90)); break
            except Exception:
                if k == 3: raise
                time.sleep(5 * (k + 1))
        out += j['features']
        nx = [l for l in j.get('links', []) if l['rel'] == 'next']
        if not nx: break
        body = nx[0]['body']; time.sleep(0.5)
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

        def run(f):
            err = ''
            for k in range(3):
                try: return one(work, port, f, bb)
                except Exception as e:
                    err = repr(e)[:150]; time.sleep(4 * (k + 1))
            return 'ERR ' + f['id'] + ' ' + err
        with ThreadPoolExecutor(5) as ex:
            res = list(ex.map(run, use))
        errs = [r for r in res if r.startswith('ERR')]; bad += len(errs)
        print(port, dict(collections.Counter(r.split()[0] for r in res)), errs[:5], flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
