"""Step 5: the fusion and its out-of-sample evaluation (leave-one-port-out and forward in time).

For every fold: the generic scores (cached in <work>/gen_cache.pkl, the slow part), the exemplar / history features, then one
gradient-boosted classifier per tried configuration. Prints the table of tried configurations and writes <work>/fuse_out.pkl.
The adopted configuration is ('opt+diff+S1', 1) (final.py). The others, and the weight 3 on changed cells, were tried and are
not adopted.
    python -m sat.pres3.fuse --work DIR
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
import warnings

import numpy as np

from . import common, dataset, evalcv, model


def s1_block(C):
    """[the current S1 judgement p (the evaluation sheet), xh, dt_h]. xh (mean VH dB of the section's band in the paired S1
    scene) and dt_h (hours between the S1 scene and the S2 day) come from inputs/s1_cells.csv; NaN where the cell has no row."""
    s1 = {}
    with open(common.INPUTS / 's1_cells.csv', newline='') as fh:
        for r in csv.DictReader(fh): s1[(r['port'], r['date'], r['section_id'])] = r

    def f(v):
        try: return float(v)
        except ValueError: return np.nan
    S1 = np.array([[np.nan, f(s1[k]['xh']), f(s1[k]['dt_h'])] if (k := (c['port'], c['date'], c['section'])) in s1 else [np.nan] * 3 for c in C], np.float32)
    S1[:, 0] = np.array([c['p_cur'] if c['p_cur'] is not None else np.nan for c in C], np.float32)
    return S1


def load(work):
    C, X, names = dataset.load(work, common.PORTS)
    lab, y = model.labelled(C); ch = evalcv.changed_flags(C, lab); ev = [i for i in lab if C[i]['eval']]
    return C, X, names, lab, y, ch, ev


def main(argv=None):
    warnings.filterwarnings('ignore')
    a = common.parser(__doc__.split('\n')[0]).parse_args(argv); work = common.work_dir(a.work)
    C, X, names, lab, y, ch, ev = load(work); phys = model.phys_cols(names)
    D = np.concatenate([np.load(f'{work}/diff_{p}.npz')['F'] for p in common.PORTS]); PT = np.concatenate([np.load(f'{work}/path_{p}.npz')['F'] for p in common.PORTS])
    S1 = s1_block(C)
    allset = set(lab); cols = list(range(X.shape[1]))
    cache = f'{work}/gen_cache.pkl'; G = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    out = {}
    for fname, folds in (('LOPO', list(evalcv.folds_lopo(C, lab))), ('FWD', list(evalcv.folds_fwd(C, lab)))):
        P = {}
        for name, tr, te in folds:
            if (fname, name) not in G:
                G[(fname, name)] = model.gen_scores(C, X, y, tr, te, cols); pickle.dump(G, open(cache + '.tmp', 'wb')); os.replace(cache + '.tmp', cache)
                print('gen', fname, name, 'done', flush=True)
            s = G[(fname, name)]; A = model.adapt_feats(C, X, y, lab, phys, s, allset); trv = np.array([i for i in tr if not np.isnan(s[i])])
            chg = np.array([ch[i] == 1 for i in trv])
            sets = {'hist': A[:, :4], 'opt': A, 'opt+diff+S1': np.column_stack([A, D, S1]), 'opt+S1': np.column_stack([A, S1]),
                    'opt+diff+path+S1': np.column_stack([A, D, PT, S1]), 'opt+diff+path': np.column_stack([A, D, PT])}
            for nm, Z in sets.items():
                for w in (1, 3):
                    if w > 1 and nm in ('hist', 'opt'): continue
                    m = model.stack_model().fit(Z[trv], y[trv], sample_weight=np.where(chg, w, 1.0))
                    P.setdefault((nm, w), np.full(len(C), np.nan))[te] = m.predict_proba(Z[te])[:, 1]
        te = [i for i in ev if not np.isnan(P[('opt', 1)][i])]
        for k, p in P.items():
            r = dataset.report(C, te, p, f'{fname} {k[0]} w{k[1]}')
            print(r[0], '| chg %.3f same %.3f' % (dataset.report(C, [i for i in te if ch[i] == 1], p, '')[1]['acc'], dataset.report(C, [i for i in te if ch[i] == 0], p, '')[1]['acc']), flush=True)
        out[fname] = P
    pickle.dump(out, open(f'{work}/fuse_out.pkl', 'wb'))


if __name__ == '__main__':
    sys.exit(main())
