"""Quay occupancy by a statistical change test against the per-orbit empty-quay baseline (occ-1, trial).

A second, separately versioned system beside the fixed-dB quay count (sat/quay.py, quay-v2 / est-2). It reads
the same inputs (the scene's VV and VH, the baseline layers of the scene's orbit, the quay sample geometry) and
uses no extra Sentinel Hub request. Nothing in the daily run (sat/run.py) calls it yet: it is evaluated against
the owner's eye labels first (sat/eval_occ.py).

Method (2部's survey, option 1; Conradsen et al. 2016 IEEE TGRS 54(5); Canty et al. 2020 Remote Sens. 12(1):46):

1. Speckle model. A 3x3-mean sigma0 of an unchanged pixel is Gamma distributed with mean mu and shape L (the
   equivalent number of looks, ENL; ``enl`` in the settings, or estimated from homogeneous water).
   Scene to scene the empty berth varies more than speckle (wind, small craft, incidence): with
   ``temporal_enl`` the shape per pixel is the one whose p90 / p10 ratio is the baseline's vv_p90 / vv_p10
   (capped at ``enl``), so the test is against the berth's own spread over the baseline's N scenes. Where the
   berth is busy p90 is a ship and the shape small: less sensitive there (a lower percentile, bl-2, helps).
2. Reference. The baseline stores per-pixel percentiles over N scenes, not the mean. For the empty berth the
   percentile q_p of Gamma(L, mu/L) is mu * G^-1(p; L) / L, so mu = q_p * L / G^-1(p; L). The percentile of N
   scenes is itself uncertain: var(q_p) ~ p (1 - p) / (N f(q_p)^2), turned into the reference's equivalent looks
   L_ref = mu^2 / var. Where the berth is busy (p10 is a ship, not the empty berth) a lower percentile (p0-p5 of
   a longer baseline, bl-2) is used when the baseline has it (``reference``).
3. Two-date test per pixel and band. Q = X / mu_ref follows F(2L, 2L_ref) when nothing changed; a ship raises
   the backscatter, so the one-sided p-value is P(F >= Q). VV and VH are combined by Fisher's method
   (-2 sum ln p ~ chi2 with 4 degrees of freedom).
4. Multi-temporal (omnibus) test of k scenes (``omnibus``): -2 rho ln Q ~ chi2((k-1) p^2) per band (p = 1 for an
   intensity), summed over VV and VH. Used for the baseline's own homogeneity (bl-2) and when a scene stack is at
   hand; the daily scene alone uses step 3.
5. Occupancy of a quay section. Along the line every 10 m, the smallest p of the depth pixels within
   ``reach_m`` of the water's edge (Bonferroni over those pixels) is a sample's p. Its log Bayes factor for
   "a ship" against "empty" is ln(a p^(a-1)) (a Beta(a, 1) alternative to the uniform p of an empty berth),
   damped by ``tau`` for the correlation of neighbouring samples (3x3 mean). The section's log-likelihood ratio
   is the largest sum over ``window_m`` of consecutive samples (a ship, not the whole section), and
   P(occupied) = sigmoid(logit(prior) + LLR).
6. Rows abreast (c). In the best window, how far off the edge the significant pixels reach without a break,
   from a first significant pixel within ``start_m`` of the edge (median over the window's samples that have one): c = 1 up to ``col_edges_m[0]``, 2 up to ``col_edges_m[1]``,
   3 beyond.

Only numpy (the incomplete beta and gamma functions are below). Trial: not calibrated; the evaluation is
sat/eval_occ.py.
"""
from __future__ import annotations

import math

import numpy as np

from .baseline import smooth_db

OCC_VERSION = "occ-1"
DEFAULTS = {
    "enl": 12.0,              # ENL of a 3x3 mean of S1 IW GRDH (single look ENL 4.4, neighbouring pixels correlated)
    "baseline_pct": 10.0,     # the percentile of the bl-1 layers (vv_p10, vh_p10)
    "baseline_scenes": 24,    # N of the baseline when its meta does not say
    "low_layers": (("p5", 5.0), ("p0", None)),   # bl-2: lower percentiles; None = the minimum (p = 1 / (N + 1))
    "busy_gap_db": 3.0,       # p10 - p5 above this: p10 is a ship, use the lower layer
    "reach_m": 40.0,          # depth pixels read for a sample's p (one ship's beam plus the 3x3 spread)
    "beta_a": 0.05,           # Beta(a, 1) alternative of the p-values under "a ship"
    "tau": 0.5,               # damping of the summed log Bayes factors (neighbouring samples correlated)
    "window_m": 40.0,         # shortest stretch of a ship along the line
    "prior": 0.5,             # prior P(occupied) of a section
    "alpha_px": 1e-3,         # a depth pixel is significant below this (for the reach)
    "start_m": 15.0,          # the significant run must start within this of the water's edge
    "col_edges_m": (35.0, 60.0),
    "temporal_enl": True,     # per-pixel ENL of the scene-to-scene spread (vv_p90 / vv_p10), capped at enl
    "enl_min": 1.0,
    "p_floor": 1e-6,          # a sample's p is not taken below this (one sample's evidence is capped; model misfit)
}


# ---------------------------------------------------------------- special functions (vectorised, numpy only)

def _betacf(a, b, x, iters: int = 300, eps: float = 1e-14):
    """Continued fraction of the incomplete beta function (Lentz), elementwise."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        de = d * c
        h *= de
        if np.all(np.abs(de - 1.0) < eps):
            break
    return h


def betainc(a: float, b: float, x) -> np.ndarray:
    """Regularised incomplete beta I_x(a, b), elementwise in x (a, b scalars > 0)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    out = np.empty_like(x)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        front = np.exp(lbeta + a * np.log(x) + b * np.log1p(-x))
    lo = x < (a + 1.0) / (a + b + 2.0)
    if lo.any():
        out[lo] = front[lo] * _betacf(a, b, x[lo]) / a
    hi = ~lo
    if hi.any():
        out[hi] = 1.0 - front[hi] * _betacf(b, a, 1.0 - x[hi]) / b
    out[x <= 0.0] = 0.0
    out[x >= 1.0] = 1.0
    return out


def f_sf(q, d1: float, d2: float) -> np.ndarray:
    """P(F >= q) of the F distribution with (d1, d2) degrees of freedom; NaN stays NaN."""
    q = np.asarray(q, float)
    ok = np.isfinite(q)
    qq = np.where(ok, np.maximum(q, 0.0), 1.0)
    out = betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * qq))
    return np.where(ok, out, np.nan)


def gammainc(a: float, x) -> np.ndarray:
    """Regularised lower incomplete gamma P(a, x), elementwise in x (series below a + 1, fraction above)."""
    x = np.maximum(np.asarray(x, float), 0.0)
    out = np.zeros_like(x)
    lga = math.lgamma(a)
    lo = (x > 0) & (x < a + 1.0)
    if lo.any():
        xl = x[lo]
        ap, s, d = a, np.full_like(xl, 1.0 / a), np.full_like(xl, 1.0 / a)
        for _ in range(500):
            ap += 1.0
            d = d * xl / ap
            s += d
            if np.all(np.abs(d) < np.abs(s) * 1e-15):
                break
        out[lo] = s * np.exp(-xl + a * np.log(xl) - lga)
    hi = x >= a + 1.0
    if hi.any():
        xh = x[hi]
        tiny = 1e-300
        b = xh + 1.0 - a
        c = np.full_like(xh, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        for i in range(1, 500):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            de = d * c
            h *= de
            if np.all(np.abs(de - 1.0) < 1e-15):
                break
        out[hi] = 1.0 - np.exp(-xh + a * np.log(xh) - lga) * h
    return np.clip(out, 0.0, 1.0)


def chi2_sf(x, k: float) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = 1.0 - gammainc(k / 2.0, np.where(ok, x, 0.0) / 2.0)
    return np.where(ok, out, np.nan)


def gamma_ppf(p: float, shape: float) -> float:
    """Quantile of Gamma(shape, scale 1) by bisection (scalar)."""
    lo, hi = 0.0, max(10.0, shape * 10.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gammainc(shape, np.array([mid]))[0] < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- the test

def to_lin(db):
    db = np.asarray(db, float)
    return np.where(np.isfinite(db), 10.0 ** (np.where(np.isfinite(db), db, 0.0) / 10.0), np.nan)


def enl_estimate(db: np.ndarray, region: np.ndarray, half: int = 2, min_px: int = 200) -> float | None:
    """ENL of a smoothed dB image from homogeneous pixels of ``region`` (mean^2 / var in local windows, median)."""
    from .grid import box_sum
    lin = to_lin(db)
    ok = np.isfinite(lin) & region
    if ok.sum() < min_px:
        return None
    z = np.where(ok, lin, 0.0)
    n = box_sum(ok.astype(float), half)
    m = box_sum(z, half) / np.maximum(n, 1)
    v = box_sum(z * z, half) / np.maximum(n, 1) - m * m
    full = (n >= (2 * half + 1) ** 2) & ok & (v > 0)
    if full.sum() < min_px:
        return None
    e = (m[full] ** 2) / v[full]
    return float(np.median(e))


def _ratio_table(enl_max: float):
    ls = np.exp(np.linspace(math.log(0.3), math.log(max(enl_max, 0.31)), 60))
    r = np.array([gamma_ppf(0.9, l) / gamma_ppf(0.1, l) for l in ls])
    return ls, r


_RT: dict = {}


def temporal_enl(vv_p10: np.ndarray, vv_p90: np.ndarray, enl_max: float, enl_min: float = 1.0) -> np.ndarray:
    """Per pixel the Gamma shape whose p90 / p10 ratio is the baseline's (dB layers), clipped to [enl_min, enl_max]."""
    key = round(float(enl_max), 3)
    if key not in _RT:
        _RT[key] = _ratio_table(enl_max)
    ls, r = _RT[key]
    with np.errstate(invalid="ignore"):
        ratio = 10.0 ** ((vv_p90 - vv_p10) / 10.0)
    # r decreases with L
    lt = np.interp(np.where(np.isfinite(ratio), ratio, r[-1]), r[::-1], ls[::-1])
    lt = np.clip(lt, enl_min, enl_max)
    return np.where(np.isfinite(ratio), lt, enl_max)


def ref_factor(pct: float, enl: float, n_scenes: int) -> tuple[float, float]:
    """(mu / q_p, L_ref): the factor from the p-th percentile of N scenes to the mean, and the equivalent looks of
    that reference (from the variance of a sample quantile)."""
    p = min(max(pct / 100.0, 1e-4), 1 - 1e-4)
    g = gamma_ppf(p, enl)                      # quantile of Gamma(L, 1); mean L
    # density of Gamma(L, 1) at g
    f = math.exp((enl - 1.0) * math.log(g) - g - math.lgamma(enl))
    relvar = p * (1.0 - p) / (max(n_scenes, 1) * f * f * g * g)
    return enl / g, 1.0 / max(relvar, 1e-6)


def _lclass(L: np.ndarray) -> np.ndarray:
    """Shapes rounded to 0.25 (the F and quantile computations are done per class)."""
    return np.round(np.asarray(L, float) * 4.0) / 4.0


def reference(base: dict, band: str, cfg: dict, n_scenes: int, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel (mu_ref linear, L_ref) of the empty berth for ``band`` ("vv" or "vh"), given the per-pixel shape L
    (classes of _lclass). bl-1 has only p10; bl-2 layers <band>_p5 / <band>_p0 (if present) replace p10 where
    p10 - p5 exceeds busy_gap_db (a busy berth)."""
    p10 = base[f"{band}_p10"]
    mu = np.full(p10.shape, np.nan)
    lref = np.full(p10.shape, np.nan)
    low = None
    for name, pct in cfg["low_layers"]:
        lay = base.get(f"{band}_{name}")
        if lay is not None:
            low = (lay, (100.0 / (n_scenes + 1)) if pct is None else pct)
            break
    with np.errstate(invalid="ignore"):
        busy = np.isfinite(low[0]) & ((p10 - low[0]) > cfg["busy_gap_db"]) if low else np.zeros(p10.shape, bool)
    for lc in np.unique(L[np.isfinite(L)]):
        sel = L == lc
        k10, l10 = ref_factor(cfg["baseline_pct"], float(lc), n_scenes)
        mu[sel] = to_lin(p10[sel]) * k10
        lref[sel] = l10
        if low is not None and (sel & busy).any():
            k, lr = ref_factor(low[1], float(lc), n_scenes)
            sb = sel & busy
            mu[sb] = to_lin(low[0][sb]) * k
            lref[sb] = lr
    lref[~np.isfinite(mu)] = np.nan
    return mu, lref


def pvalues(vv_db: np.ndarray, vh_db: np.ndarray, base: dict, cfg: dict | None = None, n_scenes: int | None = None,
            presmoothed: bool = False) -> dict:
    """Per-pixel one-sided p-values of an increase against the empty berth: {"vv", "vh", "p", "enl"} (p = Fisher's
    combination of VV and VH; where one band is missing, the other alone). NaN = no data."""
    cfg = {**DEFAULTS, **(cfg or {})}
    n = int(n_scenes or cfg["baseline_scenes"])
    svv, svh = (vv_db, vh_db) if presmoothed else (smooth_db(vv_db), smooth_db(vh_db))
    if cfg["temporal_enl"] and "vv_p90" in base:
        L = _lclass(temporal_enl(base["vv_p10"], base["vv_p90"], float(cfg["enl"]), float(cfg["enl_min"])))
    else:
        L = np.full(svv.shape, _lclass(float(cfg["enl"])))
    out = {"enl": L}
    for band, x in (("vv", svv), ("vh", svh)):
        mu, lref = reference(base, band, cfg, n, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = to_lin(x) / mu
        p = np.full(q.shape, np.nan)
        ok = np.isfinite(q) & np.isfinite(lref)
        for lc in np.unique(L[ok]):
            for lr in np.unique(lref[ok & (L == lc)]):
                sel = ok & (L == lc) & (lref == lr)
                p[sel] = f_sf(q[sel], 2.0 * float(lc), 2.0 * float(lr))
        out[band] = p
    pv, ph = out["vv"], out["vh"]
    tiny = 1e-300
    both = np.isfinite(pv) & np.isfinite(ph)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = -2.0 * (np.log(np.maximum(np.where(both, pv, 1.0), tiny)) + np.log(np.maximum(np.where(both, ph, 1.0), tiny)))
    comb = np.where(both, chi2_sf(stat, 4.0), np.where(np.isfinite(pv), pv, ph))
    out["p"] = comb
    return out


def omnibus(stack_vv, stack_vh=None, enl: float = DEFAULTS["enl"]) -> np.ndarray:
    """Omnibus test of equality of k scenes (Conradsen et al. 2016) for intensities (p = 1 per band), elementwise:
    -2 rho ln Q ~ chi2(k - 1) per band; bands summed (diagonal dual-pol). Returns the p-value (NaN where a scene is
    missing). k = 2 is the two-date likelihood-ratio test."""
    out_stat, dof = 0.0, 0.0
    for st in (stack_vv, stack_vh):
        if st is None:
            continue
        x = np.asarray(st, float)
        k = x.shape[0]
        n = float(enl)
        with np.errstate(divide="ignore", invalid="ignore"):
            lnq = n * (k * math.log(k) + np.log(x).sum(0) - k * np.log(x.sum(0)))
        rho = 1.0 - (k / n - 1.0 / (n * k)) / (6.0 * (k - 1))
        out_stat = out_stat + (-2.0 * rho * lnq)
        dof += k - 1
    return chi2_sf(out_stat, dof)


# ---------------------------------------------------------------- sections

def _lbf(p: np.ndarray, a: float) -> np.ndarray:
    return math.log(a) + (a - 1.0) * np.log(np.clip(p, 1e-300, 1.0))


def section_occupancy(geom, sections: list[dict], pmap: np.ndarray, cfg: dict | None = None) -> dict:
    """{section_id: {"p_occ", "llr", "c", "reach_m", "samples"}} from the per-pixel p-values (``pvalues()["p"]``).
    ``sections``: sat/s2_labels.sections() (profile index and sample indices)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    step = geom.cfg["step_m"]
    offs = geom.depth_offs
    use = offs <= cfg["reach_m"]
    m = int(use.sum())
    w = max(1, int(round(cfg["window_m"] / step)))
    prior = math.log(cfg["prior"] / (1.0 - cfg["prior"]))
    out = {}
    for sec in sections:
        pr = geom.profiles[sec["profile"]]
        idx = np.asarray(sec["idx"])
        rr, cc, ok = pr["drr"][idx], pr["dcc"][idx], pr["dvalid"][idx]
        pp = np.where(ok, pmap[rr, cc], np.nan)                 # (samples, depth offsets)
        near = pp[:, use]
        has = np.isfinite(near).any(1)
        with np.errstate(invalid="ignore"):
            pmin = np.where(has, np.nanmin(np.where(np.isfinite(near), near, np.inf), 1), np.nan)
        ps = np.clip(pmin * m, cfg["p_floor"], 1.0)
        lbf = np.where(np.isfinite(ps), _lbf(np.where(np.isfinite(ps), ps, 1.0), cfg["beta_a"]), 0.0) * cfg["tau"]
        if not has.any():
            out[sec["section_id"]] = {"p_occ": None, "llr": None, "c": None, "reach_m": None, "samples": 0}
            continue
        ww = min(w, len(lbf))
        sums = np.convolve(lbf, np.ones(ww), "valid")
        j = int(np.argmax(sums))
        llr = float(sums[j])
        z = max(min(prior + llr, 50.0), -50.0)
        p_occ = 1.0 / (1.0 + math.exp(-z))
        # reach of the significant pixels in the best window
        sig = (np.isfinite(pp) & (pp < cfg["alpha_px"]))[j:j + ww]
        reaches = []
        for row in sig:   # the run of significant pixels that starts within start_m of the edge, its far end
            st = np.flatnonzero(row & (offs <= cfg["start_m"]))
            if not len(st):
                continue
            k = int(st[0])
            while k + 1 < len(row) and row[k + 1]:
                k += 1
            reaches.append(float(offs[k]) + geom.cfg["depth_step_m"] / 2)
        reach = float(np.median(reaches)) if reaches else 0.0
        e1, e2 = cfg["col_edges_m"]
        c = 0 if reach <= 0 else (1 if reach <= e1 else (2 if reach <= e2 else 3))
        out[sec["section_id"]] = {"p_occ": round(p_occ, 4), "llr": round(llr, 2), "c": c, "reach_m": reach,
                                  "samples": int(has.sum())}
    return out


def settings() -> dict:
    return {"version": OCC_VERSION, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in DEFAULTS.items()}}
