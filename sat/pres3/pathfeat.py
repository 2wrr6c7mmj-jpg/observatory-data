"""Tried and NOT adopted (kept so that the table of tried configurations can be recomputed): change tracking.

From the section's previous answered day to the cell's day, the clear days in between are chained and the gains / losses of
blobs between neighbouring days are summed. Only the previous answer (the past) is used. Writes <work>/path_<port>.npz.
    python -m sat.pres3.pathfeat --work DIR [--ports ...]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from . import common, diffeat, evalcv

NAMES = [f'{ch}_{s}' for ch in ('absd', 'nir', 'vis') for s in ('pgain', 'ploss', 'pnet', 'pnet_big', 'step_max', 'step_min', 'noise', 'direct_net', 'last_net', 'last_big')] + ['n_between', 'prev_y']


def cell_feats(C, P, offs):
    M, good, pix = diffeat.maps(C, P, offs); N = len(C)
    TH = {'absd': 0.03, 'nir': 0.03, 'vis': 0.012}
    g = {}
    for i, c in enumerate(C): g.setdefault(c['section'], []).append(i)
    F = np.full((N, 3 * 10 + 2), np.nan, np.float32)
    for sid, v in g.items():
        v.sort(key=lambda i: C[i]['date']); labs = [i for i in v if C[i]['y'] is not None]
        for a, b in zip(labs, labs[1:]):
            ia = v.index(a); ib = v.index(b)
            chain = [a] + [t for t in v[ia + 1:ib] if good[t]] + [b]
            F[b, 30] = len(chain) - 2; F[b, 31] = C[a]['y']
            for k, ch in enumerate(('absd', 'nir', 'vis')):
                m = M[ch]; th = TH[ch]; G = []; Ls = []
                for p, q in zip(chain, chain[1:]):
                    d = m[q] - m[p]; G.append((d > th).sum() * pix[b]); Ls.append((d < -th).sum() * pix[b])
                G = np.array(G); Ls = np.array(Ls); net = G - Ls
                big = np.abs(net) >= 250
                d = m[b] - m[a]
                F[b, k * 10:(k + 1) * 10] = [G.sum(), Ls.sum(), net.sum(), net[big].sum(), net.max(), net.min(), np.median(np.minimum(G, Ls)),
                                             (d > th).sum() * pix[b] - (d < -th).sum() * pix[b], net[-1], (net[big][-1] if big.any() else 0.0)]
    return F


def main(argv=None):
    warnings.filterwarnings('ignore')
    a = common.parser(__doc__.split('\n')[0]).parse_args(argv); work = common.work_dir(a.work)
    Cs = []; Fs = []
    for p in common.ports_of(a.ports):
        C, P, offs = diffeat.load_port(work, p); F = cell_feats(C, P, offs)
        np.savez_compressed(f'{work}/path_{p}.npz', F=F, names=np.array(NAMES)); Cs += C; Fs.append(F)
    F = np.concatenate(Fs); C = Cs; lab = [i for i, c in enumerate(C) if c['y'] is not None and not np.isnan(F[i, 31])]
    y = np.array([C[i]['y'] for i in lab]); prev = F[lab, 31]
    print('n', len(lab), 'median clear days between', np.median(F[lab, 30]))
    for k, n in enumerate(NAMES[:30]):
        f = F[lab, k]
        a0 = evalcv.metrics(y[prev == 0], f[prev == 0])['auc']; a1 = evalcv.metrics(y[prev == 1], f[prev == 1])['auc']
        print(f'{n:18s} prev0 auc={a0:.3f}  prev1 auc={a1:.3f}')


if __name__ == '__main__':
    sys.exit(main())
