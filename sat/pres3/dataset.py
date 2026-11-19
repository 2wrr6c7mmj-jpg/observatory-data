"""Step 3: compute (and cache) the features of every cell of a port: <work>/feat_<port>_past.npz. The slow step.

Ports are independent, so several can run side by side:
    python -m sat.pres3.dataset --work DIR --ports ZJG &
    python -m sat.pres3.dataset --work DIR --ports SYL,SLP,QGZ,ZDH &   ...
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

from . import common, evalcv, feats


def load(work, ports, mode='past', cache=True):
    """All cells of the ports (in the order of ``ports``), their features [N, 545] and the feature names."""
    Cs = []; Xs = []; names = None
    for p in ports:
        cf = f'{work}/feat_{p}_{mode}.npz'; pf = f'{work}/patch_{p}.npz'
        C = json.load(open(f'{work}/cells_{p}.json'))
        if cache and os.path.exists(cf) and os.path.getmtime(cf) > os.path.getmtime(pf):
            z = np.load(cf, allow_pickle=False); X = z['X']; names = [str(n) for n in z['names']]; good = z['good']
        else:
            z = np.load(pf); X, names, good = feats.features(z['patch'].astype(np.float32), z['offs'], C, mode=mode)
            np.savez_compressed(cf + '.tmp.npz', X=X, names=np.array(names), good=good); os.replace(cf + '.tmp.npz', cf)
        for c, g in zip(C, good): c['good'] = bool(g)
        Cs += C; Xs.append(X)
    return Cs, np.concatenate(Xs), names


def report(C, idx, p, tag):
    idx = [i for i in idx if not np.isnan(p[i])]; y = np.array([C[i]['y'] for i in idx]); q = p[idx]
    m = evalcv.metrics(y, q); lo, hi = evalcv.boot_ci(C, idx, y, q, B=500)
    s = f"{tag:34s} n={m['n']:4d} acc={m['acc']:.3f} [{lo:.3f},{hi:.3f}] rec={m['rec']:.2f} pre={m['pre']:.2f} auc={m['auc']:.3f}"
    return s, m


def main(argv=None):
    warnings.filterwarnings('ignore')
    ap = common.parser(__doc__.split('\n')[0]); ap.add_argument('--recompute', action='store_true', help='ignore the cached features')
    a = ap.parse_args(argv); work = common.work_dir(a.work)
    for p in common.ports_of(a.ports):
        C, X, names = load(work, [p], cache=not a.recompute)
        print(p, 'cells', len(C), 'answered', sum(c['y'] is not None for c in C), 'features', X.shape[1], flush=True)


if __name__ == '__main__':
    sys.exit(main())
