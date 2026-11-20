"""PR-0 probe (confirm-1): can a GitHub Actions runner reach Element84 Earth Search and the public Sentinel-2 L2A COGs on AWS?

Read only. Run by .github/workflows/pres3-es-probe.yml on a push to probe/pres3-es-*; every result is a check-run annotation
(::notice / ::error). Nothing is written to the repository. No retries: a refusal (403 and the like) is reported as it is.

  1. one STAC search (one port, the last 10 days)
  2. the window of one scene of that search, 6 bands (as sat.pres3.fetch reads them)
  3. one band of a scene whose COG path is built from the scene id alone (no STAC): the last scene of the port in inputs/scenes.csv

Steps 2 and 3 run in a child process with libcurl's verbose log on, so the HTTP requests GDAL really made (count, status lines,
bytes of the ranged reads) can be counted. GDAL makes a HEAD and 2 ranged GETs per file, so the
probe makes about 22 HTTP requests in all (1 search + 7 files x 3), not the 10 of the design: 6 bands cannot be read in fewer. 2 s between the files.
    python -m sat.pres3.es_probe [--port QDH] [--days 10]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import common

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
COG_ROOT = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"
ASSET_FILE = {'blue': 'B02', 'green': 'B03', 'red': 'B04', 'nir': 'B08', 'swir16': 'B11', 'scl': 'SCL'}
PAUSE_S = 2.0


def cog_href(scene_id: str, band: str) -> str:
    """The COG of a band from the scene id alone: S2C_51STV_20261111_0_L2A ->
    <root>/51/S/TV/2026/11/S2C_51STV_20261111_0_L2A/B04.tif (the month is not zero padded)."""
    m = re.fullmatch(r"S2[A-D]_(\d{1,2})([A-Z])([A-Z]{2})_(\d{4})(\d{2})\d{2}_\d+_L2A", scene_id)
    if not m:
        raise ValueError(f"not a Sentinel-2 L2A scene id of Earth Search: {scene_id}")
    zone, lat, sq, year, month = m.groups()
    return f"{COG_ROOT}/{int(zone)}/{lat}/{sq}/{year}/{int(month)}/{scene_id}/{ASSET_FILE[band]}.tif"


def note(title, msg, level="notice"):
    print(f"::{level} title={title}::{msg}", flush=True)


def child(argv):
    """Read the windows of the given hrefs; print one JSON line. (Runs with CPL_CURL_VERBOSE=YES: curl's log goes to stderr.)"""
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    bb = json.loads(argv[0]); hrefs = argv[1:]; pad = 0.004
    b4 = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]; out = []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='0',
               GDAL_HTTP_TIMEOUT='60', CPL_CURL_VERBOSE='YES')
    with rasterio.Env(**env):
        for k, href in enumerate(hrefs):
            if k: time.sleep(PAUSE_S)
            t = time.time()
            try:
                with rasterio.open(href) as ds:
                    b = transform_bounds('EPSG:4326', ds.crs, *b4)
                    x0 = np.floor(b[0] / 20) * 20; y0 = np.floor(b[1] / 20) * 20; x1 = np.ceil(b[2] / 20) * 20; y1 = np.ceil(b[3] / 20) * 20
                    w = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
                    a = ds.read(1, window=w, boundless=True, fill_value=0)
                out.append(dict(file=href.rsplit('/', 1)[-1], ok=True, shape=list(a.shape), nonzero=round(float((a > 0).mean()), 3),
                                s=round(time.time() - t, 2)))
            except Exception as e:
                out.append(dict(file=href.rsplit('/', 1)[-1], ok=False, err=repr(e)[:200], s=round(time.time() - t, 2)))
    print("RESULT " + json.dumps(out), flush=True)
    return 0


def read_windows(bb, hrefs):
    """-> (per-file results, the HTTP requests GDAL made: count, status counts, body bytes, seconds)."""
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "sat.pres3.es_probe", "--child", json.dumps(bb), *hrefs], capture_output=True,
                       text=True, cwd=str(common.REPO), timeout=600)
    res = [json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    err = r.stderr
    status = {}; nreq = 0; nbytes = 0; method = ''; code = ''
    for line in err.splitlines():                     # curl's log: "> GET ...", "< HTTP/1.1 206 ...", "< Content-Length: n"
        m = re.match(r"> (GET|HEAD|CONNECT) ", line)
        if m:
            method = m.group(1); nreq += method != 'CONNECT'; continue
        m = re.match(r"< HTTP/[\d.]+ (\d{3})", line)
        if m:
            code = m.group(1)
            if method != 'CONNECT': status[code] = status.get(code, 0) + 1
            continue
        m = re.match(r"< content-length: (\d+)", line, flags=re.I)
        if m and method == 'GET' and code == '206': nbytes += int(m.group(1))     # the ranged reads (the HEAD carries no body)
    http = dict(requests=nreq, status=status, body_bytes=nbytes, s=round(time.time() - t, 2), rc=r.returncode)
    return (res[0] if res else [dict(ok=False, err=(r.stderr or r.stdout)[-300:])]), http


def main(argv=None):
    if argv is None: argv = sys.argv[1:]
    if argv[:1] == ["--child"]: return child(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', default='QDH'); ap.add_argument('--days', type=int, default=10)
    a = ap.parse_args(argv)
    bb = {x['id']: x for x in json.load(open(common.REPO / 'sat' / 'sites.json'))['sites']}[a.port]['bbox']
    bad = 0
    try:
        import rasterio
        note("pres3-es-probe rasterio", f"rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}, python {sys.version.split()[0]}")
    except Exception as e:
        note("pres3-es-probe rasterio", f"import failed: {e!r}"[:300], "error"); return 1

    # 1. one STAC search
    now = dt.datetime.now(dt.timezone.utc); t0 = now - dt.timedelta(days=a.days)
    body = dict(collections=["sentinel-2-l2a"], limit=100, bbox=bb,
                datetime=f"{t0:%Y-%m-%dT%H:%M:%SZ}/{now:%Y-%m-%dT%H:%M:%SZ}")
    feats = []; t = time.time()
    try:
        raw = urllib.request.urlopen(urllib.request.Request(SEARCH_URL, data=json.dumps(body).encode(),
                                                            headers={"Content-Type": "application/json"}), timeout=90).read()
        feats = json.loads(raw)['features']
        scenes = sorted(((f['properties']['datetime'], f['id'], f['properties'].get('created'), f['properties'].get('eo:cloud_cover'))
                         for f in feats), reverse=True)
        lag = ""
        if scenes and scenes[0][2]:
            d = dt.datetime.fromisoformat(scenes[0][2].replace('Z', '+00:00')) - dt.datetime.fromisoformat(scenes[0][0].replace('Z', '+00:00'))
            lag = f", newest scene: sensed {scenes[0][0]}, created in the catalogue {scenes[0][2]} ({d.total_seconds() / 3600:.1f} h later)"
        note("pres3-es-probe 1 STAC search", f"HTTP 200, {time.time() - t:.2f} s, {len(raw)} bytes, {a.port} last {a.days} days: "
             f"{len(feats)} scenes {[s[1] for s in scenes][:6]}{lag}")
    except urllib.error.HTTPError as e:
        bad += 1; note("pres3-es-probe 1 STAC search", f"HTTP {e.code} {e.reason} after {time.time() - t:.2f} s (not retried)", "error")
    except Exception as e:
        bad += 1; note("pres3-es-probe 1 STAC search", f"failed after {time.time() - t:.2f} s: {e!r}"[:300], "error")

    # 2. the window of one scene, 6 bands, the hrefs of the search
    if feats:
        f = max(feats, key=lambda f: f['properties']['datetime'])
        hrefs = [f['assets'][b]['href'] for b in ASSET_FILE]
        same = [h == cog_href(f['id'], b) for h, b in zip(hrefs, ASSET_FILE)]
        time.sleep(PAUSE_S)
        res, http = read_windows(bb, hrefs)
        ok = all(r.get('ok') for r in res); bad += not ok
        note("pres3-es-probe 2 COG window (6 bands)", f"{f['id']}: {'ok' if ok else 'FAILED'}, {http['s']} s in all (with {PAUSE_S:.0f} s pauses), "
             f"HTTP requests {http['requests']} {http['status']}, body bytes {http['body_bytes']}; files {json.dumps(res)}; "
             f"hrefs equal to the path built from the id: {sum(same)}/{len(same)}" + ("" if all(same) else f" (search: {hrefs[2]})"),
             "notice" if ok else "error")
    else:
        note("pres3-es-probe 2 COG window (6 bands)", "skipped: the search returned no scene", "warning")

    # 3. no STAC: the path from the scene id of the list in the repository
    with open(common.INPUTS / 'scenes.csv', newline='') as fh:
        sid = [r['scene_id'] for r in csv.DictReader(fh) if r['port'] == a.port][-1]
    time.sleep(PAUSE_S)
    res, http = read_windows(bb, [cog_href(sid, 'red')])
    ok = all(r.get('ok') for r in res); bad += not ok
    note("pres3-es-probe 3 COG by id (no STAC)", f"{cog_href(sid, 'red')}: {'ok' if ok else 'FAILED'}, {http['s']} s, HTTP requests "
         f"{http['requests']} {http['status']}, body bytes {http['body_bytes']}; {json.dumps(res)}", "notice" if ok else "error")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
