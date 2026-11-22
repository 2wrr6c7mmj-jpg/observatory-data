"""Step 2: from the fetched windows (<work>/data/<port>/*.npz) make the rectified patch of every section and day.

patch[N, 24 along the quay, 41 offsets from -80 m (land) to +120 m (water) every 5 m, 5 bands] in reflectance, and the cell list
(port, date, section, cloud / shadow / nodata shares from SCL, the owner's answer when there is one).
Writes <work>/patch_<port>.npz and <work>/cells_<port>.json.
    python -m sat.pres3.build --work DIR [--ports ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

from . import common, labels

OFFS = np.arange(-80, 125, 5.0); NA = labels.NA; CH = ['blue', 'green', 'red', 'nir', 'swir16']


def bil(a, r, c):
    h, w = a.shape[:2]; r = np.clip(r, 0, h - 1.001); c = np.clip(c, 0, w - 1.001)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int); fr = r - r0; fc = c - c0
    if a.ndim == 3: fr = fr[:, None]; fc = fc[:, None]
    return a[r0, c0] * (1 - fr) * (1 - fc) + a[r0 + 1, c0] * fr * (1 - fc) + a[r0, c0 + 1] * (1 - fr) * fc + a[r0 + 1, c0 + 1] * fr * fc


def sample_points(secs):
    """section id -> (lon, lat) [NA, len(OFFS)]"""
    geo = {}
    for sid, s in secs.items():
        near = np.asarray(s['near'], float); far = np.asarray(s['far'], float); u = (far - near) / 120.0
        lon = near[:, 0][:, None] + u[:, 0][:, None] * OFFS[None]; lat = near[:, 1][:, None] + u[:, 1][:, None] * OFFS[None]
        k = len(near); t = np.linspace(0, k - 1, NA); i0 = np.clip(np.floor(t).astype(int), 0, k - 2); f = (t - i0)[:, None]
        geo[sid] = (lon[i0] * (1 - f) + lon[i0 + 1] * f, lat[i0] * (1 - f) + lat[i0 + 1] * f)
    return geo


def day_cells(port, date, scenes, secs, geo, old, trial=()):
    """The cells and patches of one day from its fetched windows (scenes: [(file, meta)]). Several scenes on a day: the one
    with the least nodata; None when more than half of it is nodata. old: (date, section) -> the answer and the evaluation
    sheet's row (labels.lookup), {} for none.

    trial: the ids of the port's trial sections (sat.confirm_trial). A window read wider for them carries meta['base']
    (sat.pres3.fetch.one): the other sections are then cut from that part of the arrays, with its origin, and the nodata share
    that decides the day is that part's: the same arrays and the same arithmetic as a window of the site's bbox alone, so their
    patches are the same bytes. The trial sections are cut from the whole window."""
    from rasterio.warp import transform as rtransform
    best = None
    for fn, m in scenes:
        z = np.load(fn); a = np.stack([z[c].astype(np.float32) for c in CH[:4]], -1)
        sw = np.repeat(np.repeat(z['swir16'].astype(np.float32), 2, 0), 2, 1)[:a.shape[0], :a.shape[1]]
        scl = np.repeat(np.repeat(z['scl'], 2, 0), 2, 1)[:a.shape[0], :a.shape[1]]
        bw = m.get('base')
        cut = (slice(bw['row'], bw['row'] + bw['rows']), slice(bw['col'], bw['col'] + bw['cols'])) if bw else (slice(None), slice(None))
        nod = float((a[cut][..., 3] == 0).mean())
        if best is None or nod < best[0]: best = (nod, m, a, sw, scl)
    nod, m, a, sw, scl = best
    if nod > 0.5: return None
    img = np.concatenate([a, sw[..., None]], -1) / 10000.0; cells = []; P = []
    whole = (img, scl, m['x0'], m['y1']); bw = m.get('base')
    if bw:
        cut = (slice(bw['row'], bw['row'] + bw['rows']), slice(bw['col'], bw['col'] + bw['cols']))
        part = (img[cut], scl[cut], bw['x0'], bw['y1'])
    else:
        part = whole
    for sid, (lon, lat) in geo.items():
        im, sl, x0, y1 = whole if sid in trial else part
        x, y = rtransform('EPSG:4326', m['crs'], lon.ravel().tolist(), lat.ravel().tolist())
        c = (np.asarray(x) - x0) / 10.0 - 0.5; r = (y1 - np.asarray(y)) / 10.0 - 0.5
        p = bil(im, r, c).reshape(NA, len(OFFS), 5)
        ri = np.clip(np.round(r).astype(int), 0, sl.shape[0] - 1); ci = np.clip(np.round(c).astype(int), 0, sl.shape[1] - 1)
        sc = sl[ri, ci]; nd = bil((im[..., 3] == 0).astype(np.float32), r, c)
        o = old.get((date, sid)); s = secs[sid]
        cells.append(dict(port=port, date=date, section=sid, kind=s['kind'], sector=s['sector'], length_m=s['length_m'],
            cloud=float(np.isin(sc, [8, 9, 10]).mean()), shadow=float((sc == 3).mean()), cirrus=float((sc == 10).mean()), nodata=float(nd.mean()),
            tile_cloud=m['cloud'], scene=m['id'], time=m['datetime'],
            y=(o['y'] if o else None), eval=bool(o and o['eval']), y_eval=(o['y_eval'] if o else None), p_cur=(o['p_cur'] if o else None),
            p_pres1=(o.get('p_pres1') if o else None), dt_h=(o['dt_h'] if o else None)))
        P.append(p.astype(np.float16))
    return cells, P


def windows_by_day(work, port):
    """date -> [(file, meta)] of the fetched windows of a port."""
    byday = {}
    for fn in sorted(glob.glob(f'{work}/data/{port}/*.npz')):
        m = json.loads(str(np.load(fn)['meta'])); byday.setdefault(m['datetime'][:10], []).append((fn, m))
    return byday


def build_port(work, port):
    secs = labels.frames(port); old = labels.lookup(port); cells = []; P = []
    byday = windows_by_day(work, port)
    if not byday: raise SystemExit(f'no windows in {work}/data/{port} (run sat.pres3.fetch first)')
    geo = sample_points(secs)
    for date in sorted(byday):
        got = day_cells(port, date, byday[date], secs, geo, old)
        if got is None: continue
        cells += got[0]; P += got[1]
    np.savez_compressed(f'{work}/patch_{port}.npz', patch=np.stack(P), offs=OFFS)
    json.dump(cells, open(f'{work}/cells_{port}.json', 'w'), ensure_ascii=False)
    lab = [c for c in cells if c['y'] is not None]; want = {k for k, c in old.items() if c['y'] is not None and k[1] in secs}
    got = {(c['date'], c['section']) for c in lab}
    print(port, 'days', len(byday), 'secs', len(secs), 'cells', len(cells), 'labelled', len(lab), 'of', len(want),
          'missing days', sorted({k[0] for k in want - got}), 'lab cloud>0.2:', sum(c['cloud'] > 0.2 for c in lab), flush=True)


def main(argv=None):
    a = common.parser(__doc__.split('\n')[0]).parse_args(argv); work = common.work_dir(a.work)
    for port in common.ports_of(a.ports): build_port(work, port)


if __name__ == '__main__':
    sys.exit(main())
