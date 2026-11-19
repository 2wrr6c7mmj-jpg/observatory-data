"""Step 4: difference features. Gains and losses (m2) of blobs between the day's patch and
(a) the section's earlier answered days (the last one, the absent ones, the present ones) and
(b) the empty / full images made from the section's earlier clear days (10 / 50 / 90 % points).
Only days strictly before the cell's day are used, for the images and for the answers. Writes <work>/diff_<port>.npz.
    python -m sat.pres3.diffeat --work DIR [--ports ...]
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

from . import common, feats


def box(a): return (a[..., :-1, :-1] + a[..., 1:, :-1] + a[..., :-1, 1:] + a[..., 1:, 1:]) / 4


def maps(C, P, offs):
    X, W = feats.chans(P, offs, C); good = feats.good_day(C)
    core = (offs >= -55) & (offs <= 65)
    M = {'absd': box(X['absd'][:, :, core]), 'nir': box(X['nir'][:, :, core]), 'vis': box(X['vis'][:, :, core])}
    pix = np.array([c['length_m'] for c in C], np.float32) / P.shape[1] * 5.0   # area of one pixel, m2
    return M, good, pix


def cell_feats(C, P, offs):
    """C: the cells of one port. Returns F [N, 52] (NaN where there is no answer or no reference) and the names."""
    M, good, pix = maps(C, P, offs); N = len(C)
    g = {}
    for i, c in enumerate(C): g.setdefault(c['section'], []).append(i)
    names = []; F = np.full((N, 60), np.nan, np.float32)

    def gl(a, b, th):   # a: the day, b: the reference. Pixels gained / lost
        d = a - b; return float((d > th).sum()), float((d < -th).sum())
    for sid, v in g.items():
        v.sort(key=lambda i: C[i]['date']); va = np.array(v); gd = good[va]
        for j, i in enumerate(v):
            if C[i]['y'] is None: continue
            past = [t for t in v[:j] if C[t]['y'] is not None and C[t]['date'] < C[i]['date']]
            y1 = [t for t in past if C[t]['y'] == 1]; y0 = [t for t in past if C[t]['y'] == 0]
            r = va[:j][gd[:j]]
            k = 0; nm = []
            for ch, th in (('absd', 0.025), ('absd', 0.05), ('nir', 0.03), ('vis', 0.012)):
                a = M[ch][i]
                if past:
                    gp, lp = gl(a, M[ch][past[-1]], th); F[i, k] = gp * pix[i]; F[i, k + 1] = lp * pix[i]
                if y0:
                    q = [gl(a, M[ch][t], th) for t in y0]; F[i, k + 2] = min(x[0] for x in q) * pix[i]; F[i, k + 3] = min(x[0] + x[1] for x in q) * pix[i]; F[i, k + 4] = np.median([x[0] for x in q]) * pix[i]
                if y1:
                    q = [gl(a, M[ch][t], th) for t in y1]; F[i, k + 5] = min(x[1] for x in q) * pix[i]; F[i, k + 6] = min(x[0] + x[1] for x in q) * pix[i]; F[i, k + 7] = np.median([x[1] for x in q]) * pix[i]
                if len(r) >= 6:
                    lo, md, hi = np.percentile(M[ch][r], [10, 50, 90], axis=0)
                    F[i, k + 8] = gl(a, lo, th)[0] * pix[i]; F[i, k + 9] = gl(a, hi, th)[1] * pix[i]; gm, lm = gl(a, md, th); F[i, k + 10] = gm * pix[i]; F[i, k + 11] = lm * pix[i]
                    # the rank of the day's gain over the empty image among the earlier clear days' gains
                    hist = np.array([(M[ch][t] - lo > th).sum() for t in r]); F[i, k + 12] = (hist < (a - lo > th).sum()).mean()
                nm += [f'{ch}{th}_{s}' for s in ('gain_prev', 'loss_prev', 'gain_min0', 'dist_min0', 'gain_med0', 'loss_min1', 'dist_min1', 'loss_med1', 'gain_emp', 'loss_full', 'gain_med', 'loss_med', 'rank_emp')]
                k += 13
            names = nm
    return F[:, :len(names)], names


def load_port(work, port):
    C = json.load(open(f'{work}/cells_{port}.json')); z = np.load(f'{work}/patch_{port}.npz')
    return C, z['patch'].astype(np.float32), z['offs']


def load_feats(work, kind, port):
    """F of <work>/<kind>_<port>.npz (kind: diff or path), refused when it was made from another list of cells."""
    path = f'{work}/{kind}_{port}.npz'; z = np.load(path)
    if 'cells' not in z.files or str(z['cells']) != common.cells_key(work, [port]): raise common.stale(path, f'sat.pres3.{kind}feat' if kind == 'path' else 'sat.pres3.diffeat')
    return z['F']


def main(argv=None):
    warnings.filterwarnings('ignore')
    a = common.parser(__doc__.split('\n')[0]).parse_args(argv); work = common.work_dir(a.work)
    for p in common.ports_of(a.ports):
        F, names = cell_feats(*load_port(work, p)); np.savez_compressed(f'{work}/diff_{p}.npz', F=F, names=np.array(names), cells=common.cells_key(work, [p])); print(p, F.shape, flush=True)


if __name__ == '__main__':
    sys.exit(main())
