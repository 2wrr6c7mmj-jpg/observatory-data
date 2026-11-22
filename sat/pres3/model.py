"""The generic judgement (no answers of the port) and the exemplar / history features of a section.

gen_scores    a gradient-boosted classifier on the 545 answer-free features; coefficients from other ports
adapt_feats   for each answered cell, features made from the SAME section's answers on days strictly before the cell's day
              (and only from the cells in ``use``): counts, rate, the previous answer, and the distance of the day's
              generic score and physical features from the earlier present / absent days (the exemplars)
numpy only, except the two model constructors (scikit-learn, imported when called).
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np

PHYS_RE = r'ld_(nr|a04|a08|dk)_(run10|area|occ|ext75|new|gone)$'
PHYS_NAMES = ('blob_vis_min_L', 'blob_vis_min_M', 'blob_nir_max_L', 'blob_nir_max_M', 'blob_absd_max_M', 'dblob_vis_min_L', 'dblob_nir_max_M', 'dblob_absd_max_M')


def hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**kw)


def gen_model(n=200): return hgb(max_depth=3, max_iter=n, learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=15, random_state=0)


def stack_model(): return hgb(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=15, random_state=0)


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def labelled(C):
    lab = [i for i, c in enumerate(C) if c['y'] is not None]
    y = np.array([c['y'] if c['y'] is not None else -1 for c in C])
    return lab, y


def phys_cols(names):
    return [names.index(n) for n in names if re.match(PHYS_RE, n) or n in PHYS_NAMES]


def fill_empty(X):
    """X with every column that is missing in ALL its rows set to 0 (a copy when there is one; confirm-1, A4). scikit-learn 1.9 cannot bin an
    all-missing column when sample weights are given (the fit stops); a constant column is never split on, so the trees are
    the same as without the column, and a column with any value is left as it is (no change where nothing is empty)."""
    X = np.asarray(X); empty = np.isnan(X).all(0) if len(X) else np.zeros(X.shape[1], bool)
    if not empty.any(): return X
    X = X.copy(); X[:, empty] = 0.0
    return X


def gen_scores(C, X, y, tr, te, cols):
    """Logits. For the cells of tr: out-of-fold over the ports inside tr. For te: the model trained on all of tr."""
    s = np.full(len(C), np.nan); tr = np.array(tr)
    ports = sorted(set(C[i]['port'] for i in tr))
    for p in ports:
        a = [i for i in tr if C[i]['port'] != p]; b = [i for i in tr if C[i]['port'] == p]
        if len(set(y[a])) < 2: continue
        m = gen_model().fit(fill_empty(X[a][:, cols]), y[a]); s[b] = logit(m.predict_proba(X[b][:, cols])[:, 1])
    if len(te):
        m = gen_model().fit(fill_empty(X[tr][:, cols]), y[tr]); s[te] = logit(m.predict_proba(X[te][:, cols])[:, 1])
    return s


def adapt_feats(C, X, y, lab, phys, s, use):
    """Features from the section's earlier answers (date < the cell's date, and in ``use``). Columns:
    0 n present, 1 n absent, 2 (present+1)/(n+2), 3 the previous answer, 4 the generic score, 5 days since the previous answer,
    then Z - mean Z of the earlier present days, Z - mean Z of the earlier absent days, Z - Z of the previous answered day
    (Z = the generic score and the physical features)."""
    Z = np.column_stack([s] + [X[:, k] for k in phys]); nz = Z.shape[1]
    A = np.full((len(C), 6 + nz * 3), np.nan, np.float32)
    g = {}
    for i in lab: g.setdefault((C[i]['port'], C[i]['section']), []).append(i)
    for v in g.values():
        v.sort(key=lambda i: C[i]['date'])
        for j, i in enumerate(v):
            past = [t for t in v[:j] if t in use and C[t]['date'] < C[i]['date']]
            y1 = [t for t in past if y[t] == 1]; y0 = [t for t in past if y[t] == 0]
            A[i, 0] = len(y1); A[i, 1] = len(y0); A[i, 2] = (len(y1) + 1) / (len(past) + 2); A[i, 3] = y[past[-1]] if past else np.nan
            A[i, 4] = s[i]
            if past:
                A[i, 5] = (dt.date.fromisoformat(C[i]['date']) - dt.date.fromisoformat(C[past[-1]]['date'])).days
                A[i, 6 + 2 * nz:6 + 3 * nz] = Z[i] - Z[past[-1]]
            if y1: A[i, 6:6 + nz] = Z[i] - np.nanmean(Z[y1], 0)
            if y0: A[i, 6 + nz:6 + 2 * nz] = Z[i] - np.nanmean(Z[y0], 0)
    return A
