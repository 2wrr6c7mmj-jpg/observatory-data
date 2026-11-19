"""Out-of-sample evaluation: leave-one-port-out, forward in time, and a bootstrap over scenes (port x day). numpy only."""
import numpy as np
BOUNDS = ['2025-12-01', '2026-03-01', '2026-06-01', '2026-09-01']

def folds_lopo(C, lab):
    ports = sorted(set(c['port'] for c in C))
    for p in ports:
        tr = [i for i in lab if C[i]['port'] != p]; te = [i for i in lab if C[i]['port'] == p]
        yield p, tr, te

def folds_fwd(C, lab, bounds=BOUNDS):
    for k, b in enumerate(bounds):
        e = bounds[k + 1] if k + 1 < len(bounds) else '9999'
        tr = [i for i in lab if C[i]['date'] < b]; te = [i for i in lab if b <= C[i]['date'] < e]
        if tr and te: yield b, tr, te

def changed_flags(C, lab):
    """1: the answer differs from the section's previous answer, 0: the same, -1: no previous answer."""
    out = {}; g = {}
    for i in lab: g.setdefault((C[i]['port'], C[i]['section']), []).append(i)
    for v in g.values():
        v.sort(key=lambda i: C[i]['date']); prev = None
        for i in v:
            out[i] = -1 if prev is None else int(C[i]['y'] != prev); prev = C[i]['y']
    return out

def metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p); h = p >= thr
    tp = int((h & (y == 1)).sum()); fp = int((h & (y == 0)).sum()); fn = int((~h & (y == 1)).sum()); tn = int((~h & (y == 0)).sum())
    n = len(y); acc = (tp + tn) / n if n else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan'); pre = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan')
    auc = float('nan')
    if 0 < y.sum() < n:
        o = np.argsort(p, kind='stable'); r = np.empty(n); r[o] = np.arange(1, n + 1)
        for v in np.unique(p):
            m = p == v; r[m] = r[m].mean()
        auc = (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (n - y.sum()))
    return dict(n=n, acc=acc, rec=rec, pre=pre, f1=f1, brier=float(np.mean((p - y) ** 2)) if n else float('nan'), auc=float(auc), tp=tp, fp=fp, fn=fn, tn=tn)

def boot_ci(C, idx, y, p, stat='acc', B=2000, seed=0, thr=0.5):
    """Resamples scenes (port x day). idx indexes C; y and p are in the order of idx."""
    rng = np.random.default_rng(seed); keys = [C[i]['port'] + C[i]['date'] for i in idx]; uk = sorted(set(keys))
    pos = {k: [j for j, kk in enumerate(keys) if kk == k] for k in uk}; y = np.asarray(y); p = np.asarray(p); vals = []
    for _ in range(B):
        s = rng.integers(0, len(uk), len(uk)); j = np.concatenate([pos[uk[t]] for t in s])
        vals.append(metrics(y[j], p[j], thr)[stat])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
