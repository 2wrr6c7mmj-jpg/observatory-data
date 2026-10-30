"""cfar-v2 PR1 (experimental): log-domain standardised CFAR statistics with per-tile model selection (numpy only).

Experimental. Nothing in ``sat/run.py`` or ``sat/detect.py`` (cfar-v1, used for ZJG) calls this
module; detection after thresholding (hysteresis, shape, side lobes, ...) is PR2.

Pipeline (design ``cfar-v2`` sections 3.2-3.4):

1. ``y = ln sigma0`` taken straight from dB (``y = dB * ln10 / 10``; no linearisation).
2. Background over a square ring (Chebyshev guard < d <= outer) of water pixels, twice:
   pass 1 gives ``z1``; pixels with ``z1 > censor_z`` (2.5), dilated by 2 px, are removed from the
   ring and pass 2 gives the local mean ``m`` and std ``s``; ``z = (y - m) / s``.
   Guard 250 m and outer 1,000 m, rounded up to whole pixels (10 m: 25/100 px, 20 m: 13/50 px).
   A pixel needs >= 50% of the full ring to be water.
3. Per 2.56 km tile (128 px at 20 m, 256 px at 10 m; 10 m pixels thinned to every other row and
   column) five models of ``y`` are fitted by maximum likelihood to ``y - m`` (the residual, so a
   gentle trend inside the tile does not widen the fit):

   =============  =========================  =========================================
   model          y - location               parameters (``params`` in the tile table)
   =============  =========================  =========================================
   lognormal      normal                     mu, sigma
   weibull        Gumbel (minimum)           c (intensity Weibull shape), lnb
   loggamma       log of Gamma(L, theta)     L (ENL), lntheta
   logweibull3    a + Weibull(c, b)          a, c, b  (a on a 20-point grid below min)
   loggengamma    mu + s * ln Gamma(k, 1)    k, s, mu (log-cumulants; generalised gamma)
   =============  =========================  =========================================

   ``loggengamma`` is not in the design's list of four: it was added because none of the four
   follows the upper tail of K-distributed clutter (gamma speckle x gamma texture), whose log
   tail decays like exp(-C e^{y/2}); with it K clutter (texture shape 3, 10) stays near nominal.

   AIC picks the model; an upper partial Q-Q (90-99.9th percentile, RMSE in z units) checks the
   tail: if the AIC choice has ``tail_rmse > 0.25`` and a model with a heavier tail (larger z_t)
   is within 0.25, that one is used; if none is within 0.25 the largest z_t is used. The choice is
   smoothed by a 3x3 tile majority, counting only models the tile itself has parameters for (a
   model whose fit failed in that tile is never chosen for it). Tiles with < 30% usable water borrow
   from the nearest fitted tiles, at most ``borrow_max_tiles`` (3) tiles away (Chebyshev); farther
   tiles get no model and are not tested.
4. Thresholds ``z_t(PFA)`` per tile. ``z_threshold(..., censor_z=None)`` is the textbook value
   (``(t - mean) / sd`` of the model: lognormal 4.265 and Weibull 2.355 at 1e-5 for any
   parameters, log-gamma L=4.4 3.179). Because pass 2 drops the ring's own upper tail
   (``z1 > 2.5``), ``m`` and ``s`` are the model's moments *below* ``mean + 2.5 sd``; the applied
   threshold (``z_t`` in the tile table) uses those censored moments so the false-alarm rate stays
   at the nominal PFA.

Notes and known limits (1部 PR1 review):

* The tile fit leaves out pixels with ``z1 >= fit_exclude_z`` (5.0, the "first-pass detections",
  dilated by 2 px). This also cuts the clutter's own extreme upper tail out of the fit: the fitted
  tail is the tail of the clutter *below* that cut, which is what the upper partial Q-Q (90-99.9th
  percentile) checks. Effect is small for gamma / lognormal / Weibull / K(shape >= 3) clutter
  (synthetic test: 0.8-1.2x nominal), larger for very spiky clutter.
* Spiky K clutter (texture shape nu = 1) is outside the 1/2..2 goal: measured 1.56x nominal at 1e-3
  and 2.13x at 1e-4 (synthetic, 1024x1024 at 20 m). It stays within the design's acceptance range
  (1/3..3x) and is tested at that range only.
* The gamma (``loggamma``) model is fitted by maximum likelihood (ln L - psi(L) = ln mean(x) -
  mean(ln x), Newton), not by the design's moment equation psi1(L) = var(y); MLE is the more
  efficient estimate and the same in expectation.
"""
from __future__ import annotations

import math
import time
from statistics import NormalDist

import numpy as np

from .grid import dilate

DETECTOR_VERSION = "cfar-v2-pr1"
MODELS = ("lognormal", "weibull", "loggamma", "logweibull3", "loggengamma")
N_PARAMS = {"lognormal": 2, "weibull": 2, "loggamma": 2, "logweibull3": 3, "loggengamma": 3}
LN10_10 = math.log(10.0) / 10.0
EULER = 0.5772156649015329

DEFAULTS_V2 = {
    "guard_m": 250.0,            # 10 m -> 25 px (250 m), 20 m -> 13 px (260 m)
    "outer_m": 1000.0,           # 10 m -> 100 px, 20 m -> 50 px
    "min_background_fraction": 0.5,
    "censor_z": 2.5,             # pass 1: z1 above this is dropped from the ring ...
    "censor_dilate_px": 2,       # ... with this many pixels around it
    "fit_exclude_z": 5.0,        # "first-pass detections" kept out of the tile fit (with the same dilation)
    "tile_m": 2560.0,
    "thin_below_m": 15.0,        # pixels finer than this are thinned 2x2 for the fit
    "min_tile_water": 0.3,       # usable fraction of the tile needed to fit it
    "min_fit_samples": 1000,
    "borrow_max_tiles": 3,       # an unfitted tile borrows only from fitted tiles this close (Chebyshev, tiles)
    "tail_probs": (0.10, 0.001), # upper partial Q-Q: exceedance 10% .. 0.1% (90th..99.9th percentile)
    "tail_points": 16,
    "tail_rmse_max": 0.25,
    "lw3_grid": 20,
    "lw3_subsample": 4096,
    "pfa_seed": 1e-7,
    "pfa_grow": 1e-3,
    "pfa_eval": (1e-3, 1e-4),
    "sea_low_db": -25.0,
    "sea_high_db": -8.0,
}

_ND = NormalDist()


# ---------------------------------------------------------------- special functions (scalars)

def digamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def trigamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + 1 / x + f / 2 + (f / x) * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))


def tetragamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r -= 2.0 / (x * x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r - f - f / x - f * f * (0.5 - f * (1 / 6 - f * (1 / 6 - f * 3 / 10)))


def gammaincc(a: float, x) -> np.ndarray:
    """Regularised upper incomplete gamma Q(a, x), vectorised over ``x`` (series / Lentz CF)."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    lg = math.lgamma(a)
    lo = x < a + 1.0
    if lo.any():
        xs = x[lo]
        ap = np.full_like(xs, a)
        term = 1.0 / ap
        tot = term.copy()
        for _ in range(500):
            ap = ap + 1.0
            term = term * xs / ap
            tot += term
            if np.all(np.abs(term) < np.abs(tot) * 1e-16):
                break
        with np.errstate(divide="ignore"):
            out[lo] = 1.0 - tot * np.exp(-xs + a * np.log(xs) - lg)
    hi = ~lo
    if hi.any():
        xs = x[hi]
        tiny = 1e-300
        b = xs + 1.0 - a
        c = np.full_like(xs, 1.0 / tiny)
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
            h = h * de
            if np.all(np.abs(de - 1.0) < 1e-15):
                break
        out[hi] = np.exp(-xs + a * np.log(xs) - lg) * h
    return np.clip(out, 0.0, 1.0)


def _norm_isf(p) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, float))
    return np.array([-_ND.inv_cdf(float(q)) for q in p])


def _loggamma_isf_std(L: float, p) -> np.ndarray:
    """v = ln u with Q(L, u) = p (Gamma(L, 1)), Newton on ln Q in v."""
    p = np.atleast_1d(np.asarray(p, float))
    zq = _norm_isf(p)
    u0 = L * np.maximum(1 - 1 / (9 * L) + zq / (3 * math.sqrt(L)), 0.05) ** 3
    v = np.log(np.maximum(u0, 1e-12))
    lg = math.lgamma(L)
    target = np.log(p)
    for _ in range(60):
        u = np.exp(v)
        q = np.maximum(gammaincc(L, u), 1e-300)
        dlnq = -np.exp(L * v - u - lg) / q
        step = (np.log(q) - target) / dlnq
        step = np.clip(step, -1.0, 1.0)
        v = v - step
        if np.all(np.abs(step) < 1e-12):
            break
    return v


# ---------------------------------------------------------------- the four models of y

def isf(model: str, prm: tuple, p) -> np.ndarray:
    """Value t of y with P(Y > t) = p."""
    p = np.atleast_1d(np.asarray(p, float))
    if model == "lognormal":
        mu, sg = prm
        return mu + sg * _norm_isf(p)
    if model == "weibull":
        c, lnb = prm
        return lnb + np.log(-np.log(p)) / c
    if model == "loggamma":
        L, lnth = prm
        return lnth + _loggamma_isf_std(L, p)
    if model == "logweibull3":
        a, c, b = prm
        return a + b * (-np.log(p)) ** (1.0 / c)
    if model == "loggengamma":
        k, sc, mu = prm
        return mu + sc * _loggamma_isf_std(k, p)
    raise ValueError(model)


def logpdf(model: str, prm: tuple, y) -> np.ndarray:
    y = np.asarray(y, float)
    if model == "lognormal":
        mu, sg = prm
        return -0.5 * ((y - mu) / sg) ** 2 - math.log(sg) - 0.5 * math.log(2 * math.pi)
    if model == "weibull":
        c, lnb = prm
        t = c * (y - lnb)
        return math.log(c) + t - np.exp(t)
    if model == "loggamma":
        L, lnth = prm
        t = y - lnth
        return L * t - np.exp(t) - math.lgamma(L)
    if model == "logweibull3":
        a, c, b = prm
        u = y - a
        with np.errstate(divide="ignore", invalid="ignore"):
            out = math.log(c) - c * math.log(b) + (c - 1) * np.log(u) - (u / b) ** c
        return np.where(u > 0, out, -np.inf)
    if model == "loggengamma":
        k, sc, mu = prm
        t = (y - mu) / sc
        return k * t - np.exp(t) - math.log(sc) - math.lgamma(k)
    raise ValueError(model)


def moments(model: str, prm: tuple) -> tuple[float, float]:
    """Mean and standard deviation of y."""
    if model == "lognormal":
        return float(prm[0]), float(prm[1])
    if model == "weibull":
        c, lnb = prm
        return lnb - EULER / c, math.pi / (math.sqrt(6) * c)
    if model == "loggamma":
        L, lnth = prm
        return lnth + digamma(L), math.sqrt(trigamma(L))
    if model == "logweibull3":
        a, c, b = prm
        g1, g2 = math.gamma(1 + 1 / c), math.gamma(1 + 2 / c)
        return a + b * g1, b * math.sqrt(max(g2 - g1 * g1, 0.0))
    if model == "loggengamma":
        k, sc, mu = prm
        return mu + sc * digamma(k), sc * math.sqrt(trigamma(k))
    raise ValueError(model)


def censored_moments(model: str, prm: tuple, upper: float) -> tuple[float, float]:
    """Mean and std of y given y < ``upper`` (quadrature of the density)."""
    mu, sd = moments(model, prm)
    lo = prm[0] if model == "logweibull3" else mu - 40 * sd
    y = np.linspace(lo, upper, 40001)
    f = np.exp(logpdf(model, prm, y))
    f[~np.isfinite(f)] = 0.0
    w = np.trapezoid(f, y)
    m = np.trapezoid(f * y, y) / w
    v = np.trapezoid(f * (y - m) ** 2, y) / w
    return float(m), float(math.sqrt(v))


def z_threshold(model: str, prm: tuple | None, pfa, censor_z: float | None = None):
    """Standardised threshold z_t with P(Z >= z_t) = pfa.

    ``censor_z=None``: z = (y - mean) / sd (design table; lognormal and weibull do not depend on
    ``prm``, which may be None for them). ``censor_z=2.5``: z uses the moments of y below
    mean + 2.5 sd, matching the censored second pass of :func:`ring_stats`.
    """
    if prm is None:
        prm = {"lognormal": (0.0, 1.0), "weibull": (1.0, 0.0)}[model]
    mu, sd = moments(model, prm)
    if censor_z is not None:
        mu, sd = censored_moments(model, prm, mu + censor_z * sd)
    t = isf(model, prm, pfa)
    out = (t - mu) / sd
    return float(out[0]) if np.ndim(pfa) == 0 else out


# ---------------------------------------------------------------- maximum-likelihood fits on y

def _fit_lognormal(r):
    mu = float(r.mean())
    sg = float(r.std())
    n = r.size
    ll = -0.5 * n * (math.log(2 * math.pi * sg * sg) + 1)
    return (mu, sg), ll


def _weibull_shape(ell: np.ndarray, c0: float) -> float:
    """MLE of the Weibull shape c from ell = ln x (Newton on the profile equation)."""
    lm = float(ell.mean())
    emax = float(ell.max())
    d = ell - emax
    c = c0
    for _ in range(50):
        w = np.exp(c * d)
        A = w.sum()
        B = (w * ell).sum() / A
        C = (w * ell * ell).sum() / A
        g = B - 1.0 / c - lm
        gp = (C - B * B) + 1.0 / (c * c)
        step = g / gp
        cn = c - step
        if cn <= 0:
            cn = c / 2
        if abs(cn - c) < 1e-10 * c:
            c = cn
            break
        c = cn
    return c


def _fit_weibull(r):
    n = r.size
    c = _weibull_shape(r, math.pi / (math.sqrt(6) * max(float(r.std()), 1e-9)))
    rmax = float(r.max())
    lnb = rmax + math.log(np.exp(c * (r - rmax)).mean()) / c
    ll = n * math.log(c) + c * float((r - lnb).sum()) - n
    return (c, lnb), ll


def _fit_loggamma(r):
    n = r.size
    rmax = float(r.max())
    lmx = rmax + math.log(np.exp(r - rmax).mean())      # ln mean(x)
    s = lmx - float(r.mean())
    s = max(s, 1e-9)
    L = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(50):
        f = math.log(L) - digamma(L) - s
        fp = 1 / L - trigamma(L)
        Ln = L - f / fp
        if Ln <= 0:
            Ln = L / 2
        if abs(Ln - L) < 1e-12 * L:
            L = Ln
            break
        L = Ln
    lnth = lmx - math.log(L)
    ll = L * float((r - lnth).sum()) - n * L - n * math.lgamma(L)
    return (L, lnth), ll


def _lw3_ll(u, c, b):
    n = u.size
    return n * math.log(c) - n * c * math.log(b) + (c - 1) * float(np.log(u).sum()) - n


def _lw3_at(r, a, c0):
    u = r - a
    ell = np.log(u)
    c = _weibull_shape(ell, c0)
    lmax = float(ell.max())
    b = math.exp(lmax + math.log(np.exp(c * (ell - lmax)).mean()) / c)
    return c, b, _lw3_ll(u, c, b)


def _fit_logweibull3(r, grid: int, sub: int, rng):
    """a on a log-spaced grid below min(r) (profiled on a subsample), then c, b on all of r."""
    sd = max(float(r.std()), 1e-9)
    rmin = float(r.min())
    rs = r if r.size <= sub else rng.choice(r, sub, replace=False)
    best = None
    for delta in sd * np.geomspace(0.05, 50.0, grid):
        a = rmin - delta
        try:
            c, b, ll = _lw3_at(rs, a, 3.0)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if not (np.isfinite(ll) and c >= 1.0):
            continue
        if best is None or ll > best[0]:
            best = (ll, a)
    if best is None:
        return None, -math.inf
    a = best[1]
    c, b, ll = _lw3_at(r, a, 3.0)
    if not (np.isfinite(ll) and c >= 1.0):
        return None, -math.inf
    return (a, c, b), ll


def _fit_loggengamma(r, k_max: float = 1000.0):
    """y = mu + s * ln G_k (log of a generalised-gamma intensity), by log-cumulants.

    k from the skewness of y (psi2(k) / psi1(k)^1.5, monotone in k), s from the variance; mu is
    then the likelihood maximum given (k, s), and the log-likelihood is evaluated there (k and s
    are not refined further, which slightly favours the nested 2-parameter models under AIC).
    """
    n = r.size
    m = float(r.mean())
    v = float(r.var())
    skew = float(((r - m) ** 3).mean()) / v ** 1.5 if v > 0 else 0.0
    lo, hi = math.log(0.05), math.log(k_max)
    if skew >= tetragamma(k_max) / trigamma(k_max) ** 1.5:
        k = k_max
    elif skew <= tetragamma(0.05) / trigamma(0.05) ** 1.5:
        k = 0.05
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            kk = math.exp(mid)
            if tetragamma(kk) / trigamma(kk) ** 1.5 < skew:
                lo = mid
            else:
                hi = mid
        k = math.exp(0.5 * (lo + hi))
    sc = math.sqrt(v / trigamma(k))
    u = r / sc
    um = float(u.max())
    mu = sc * (um + math.log(np.exp(u - um).sum() / (n * k)))
    ll = k * float(((r - mu) / sc).sum()) - n * k - n * math.log(sc) - n * math.lgamma(k)
    return (k, sc, mu), ll


def fit_models(r: np.ndarray, cfg: dict | None = None, rng=None) -> dict:
    """Fit the four models to the 1-D sample ``r`` of y. Returns {model: {params, ll, aic, tail_rmse}}."""
    cfg = {**DEFAULTS_V2, **(cfg or {})}
    rng = rng if rng is not None else np.random.default_rng(0)
    r = np.asarray(r, float)
    fits = {
        "lognormal": _fit_lognormal(r),
        "weibull": _fit_weibull(r),
        "loggamma": _fit_loggamma(r),
        "logweibull3": _fit_logweibull3(r, cfg["lw3_grid"], cfg["lw3_subsample"], rng),
        "loggengamma": _fit_loggengamma(r),
    }
    pex = np.geomspace(cfg["tail_probs"][0], cfg["tail_probs"][1], cfg["tail_points"])
    emp = np.quantile(r, 1 - pex)
    out = {}
    for m in MODELS:
        prm, ll = fits[m]
        if prm is None or not np.isfinite(ll):
            out[m] = {"params": None, "ll": None, "aic": math.inf, "tail_rmse": math.inf}
            continue
        _mu, sd = moments(m, prm)
        q = isf(m, prm, pex)
        rmse = float(np.sqrt(np.mean(((emp - q) / sd) ** 2)))
        out[m] = {"params": tuple(float(v) for v in prm), "ll": float(ll),
                  "aic": float(2 * N_PARAMS[m] - 2 * ll), "tail_rmse": rmse}
    return out


def select_model(fits: dict, zt_ref: dict, rmse_max: float) -> str:
    """AIC choice, overridden by the upper-tail Q-Q rule (see module doc)."""
    ok_models = [m for m in MODELS if fits[m]["params"] is not None]
    best = min(ok_models, key=lambda m: fits[m]["aic"])
    if fits[best]["tail_rmse"] <= rmse_max:
        return best
    good = [m for m in ok_models if fits[m]["tail_rmse"] <= rmse_max]
    heavier = [m for m in good if zt_ref[m] > zt_ref[best]]
    if heavier:
        return min(heavier, key=lambda m: fits[m]["aic"])
    if not good:
        return max(ok_models, key=lambda m: zt_ref[m])
    return best


# ---------------------------------------------------------------- ring statistics

def _integral(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(a, axis=0, dtype=np.float64, out=ii[1:, 1:])
    np.cumsum(ii[1:, 1:], axis=1, out=ii[1:, 1:])
    return ii


def _window(ii: np.ndarray, half: int) -> np.ndarray:
    """Same as ``grid.box_sum`` but from a precomputed integral image."""
    h, w = ii.shape[0] - 1, ii.shape[1] - 1
    r = np.arange(h)
    c = np.arange(w)
    r0 = np.clip(r - half, 0, h)
    r1 = np.clip(r + half + 1, 0, h)
    c0 = np.clip(c - half, 0, w)
    c1 = np.clip(c + half + 1, 0, w)
    rows = ii[r1] - ii[r0]
    return rows[:, c1] - rows[:, c0]


def _ring(a: np.ndarray, guard: int, outer: int) -> np.ndarray:
    ii = _integral(a)
    return _window(ii, outer) - _window(ii, guard)


def ring_stats(y: np.ndarray, use: np.ndarray, guard: int, outer: int):
    """Mean, std and count of ``y`` over ``use`` pixels in the ring guard < d <= outer (Chebyshev)."""
    use = np.asarray(use, bool)
    yw = np.where(use, y, 0.0)
    n = _ring(use.astype(np.float64), guard, outer)
    s1 = _ring(yw, guard, outer)
    s2 = _ring(yw * yw, guard, outer)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu, np.sqrt(var), n


# ---------------------------------------------------------------- main entry

def window_px(pixel_m, cfg: dict | None = None) -> tuple[int, int, int, bool]:
    """(guard_px, outer_px, tile_px, thin) for a pixel size (m, or (row_m, col_m))."""
    cfg = {**DEFAULTS_V2, **(cfg or {})}
    px = float(np.mean(pixel_m))
    guard = int(math.ceil(cfg["guard_m"] / px - 1e-9))
    outer = int(math.ceil(cfg["outer_m"] / px - 1e-9))
    tile = int(round(cfg["tile_m"] / px))
    return guard, outer, tile, px < cfg["thin_below_m"]


def cfar2(db: np.ndarray, land: np.ndarray | None, pixel_m, cfg: dict | None = None, *,
          invalid: np.ndarray | None = None, seed: int = 0) -> dict:
    """Standardised log-domain CFAR statistics for one band (dB image, NaN = no data).

    Returns a dict:
      ``z``      float32 image of z = (y - m) / s (NaN where not tested)
      ``mean``, ``sd``  float32 local background of y = ln sigma0 (second pass)
      ``tiles``  list of per-tile dicts (model, z_t at the configured PFAs, tail_rmse, aic, params, ...)
      ``tile_px``, ``tile_shape``, ``info``
    Use :func:`threshold_image` for a per-pixel z_t at any PFA.
    """
    t_start = time.perf_counter()
    cfg = {**DEFAULTS_V2, **(cfg or {})}
    rng = np.random.default_rng(seed)
    db = np.asarray(db, float)
    h, w = db.shape
    guard, outer, tile, thin = window_px(pixel_m, cfg)
    water = np.isfinite(db)
    if land is not None:
        water &= ~np.asarray(land, bool)
    if invalid is not None:
        water &= ~np.asarray(invalid, bool)
    y = np.where(water, np.nan_to_num(db) * LN10_10, 0.0)
    y0 = float(np.median(y[water])) if water.any() else 0.0
    y = np.where(water, y - y0, 0.0)            # offset for numerically quiet sums; z unaffected
    ring_area = (2 * outer + 1) ** 2 - (2 * guard + 1) ** 2

    m1, s1, n1 = ring_stats(y, water, guard, outer)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = np.where(water, (y - m1) / s1, np.nan)
    enough1 = n1 >= cfg["min_background_fraction"] * ring_area
    z1[~enough1] = np.nan
    with np.errstate(invalid="ignore"):
        censor = dilate(z1 > cfg["censor_z"], cfg["censor_dilate_px"])
        fit_out = dilate(z1 >= cfg["fit_exclude_z"], cfg["censor_dilate_px"])
    use2 = water & ~censor
    m2, s2, n2 = ring_stats(y, use2, guard, outer)
    enough = enough1 & (n2 >= 0.5 * n1) & (s2 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(water & enough, (y - m2) / s2, np.nan)
    del z1, s1, n1, n2

    # ---- per-tile fits
    nty, ntx = -(-h // tile), -(-w // tile)
    tiles = []
    fitmask = np.isfinite(z) & ~fit_out
    resid = y - m2
    for ty in range(nty):
        for tx in range(ntx):
            r0, r1 = ty * tile, min(h, (ty + 1) * tile)
            c0, c1 = tx * tile, min(w, (tx + 1) * tile)
            fm = fitmask[r0:r1, c0:c1]
            rr = resid[r0:r1, c0:c1]
            area = fm.size
            usable = float(fm.sum()) / area
            if thin:
                fm = fm[::2, ::2]
                rr = rr[::2, ::2]
            sample = rr[fm]
            wt = water[r0:r1, c0:c1]
            rec = {"ty": ty, "tx": tx, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                   "usable_fraction": round(usable, 4), "n_fit": int(sample.size),
                   "borrowed": False, "own_model": None}
            if wt.any():
                vdb = db[r0:r1, c0:c1][wt]
                med = float(np.median(vdb))
                rec["vv_median_db"] = round(med, 2)
                rec["y_sd"] = round(float(np.std(vdb * LN10_10)), 4)
                rec["sea_state"] = ("low" if med < cfg["sea_low_db"] else
                                    "high" if med > cfg["sea_high_db"] else "normal")
            if usable >= cfg["min_tile_water"] and sample.size >= cfg["min_fit_samples"]:
                fits = fit_models(sample, cfg, rng)
                zt_ref = {}
                for m in MODELS:
                    p = fits[m]["params"]
                    zt_ref[m] = (z_threshold(m, p, cfg["pfa_seed"], cfg["censor_z"])
                                 if p is not None else -math.inf)
                rec["fits"] = fits
                rec["zt_ref"] = zt_ref
                rec["own_model"] = select_model(fits, zt_ref, cfg["tail_rmse_max"])
            tiles.append(rec)

    # ---- 3x3 majority smoothing, then borrowing for unfitted tiles
    own = np.full((nty, ntx), -1, dtype=int)
    for t in tiles:
        if t["own_model"] is not None:
            own[t["ty"], t["tx"]] = MODELS.index(t["own_model"])
    final = own.copy()
    grid_t = {(t["ty"], t["tx"]): t for t in tiles}
    for t in tiles:
        ty, tx = t["ty"], t["tx"]
        if own[ty, tx] < 0:
            continue
        nb = own[max(0, ty - 1):ty + 2, max(0, tx - 1):tx + 2].ravel()
        nb = nb[nb >= 0]
        cnt = np.bincount(nb, minlength=len(MODELS))
        # only models this tile has parameters for (a failed fit, e.g. logweibull3, has params None)
        cnt = np.where([t["fits"][m]["params"] is not None for m in MODELS], cnt, -1)
        top = np.flatnonzero(cnt == cnt.max())
        if own[ty, tx] in top:
            final[ty, tx] = own[ty, tx]
        else:
            final[ty, tx] = max(top, key=lambda k: t["zt_ref"][MODELS[k]])
    fitted = np.argwhere(own >= 0)
    for t in tiles:
        ty, tx = t["ty"], t["tx"]
        if own[ty, tx] >= 0:
            t["model"] = MODELS[final[ty, tx]]
            t["src"] = (ty, tx)
            continue
        t["borrowed"] = True
        if not len(fitted):
            t["model"] = None
            t["src"] = None
            continue
        dist = np.maximum(np.abs(fitted[:, 0] - ty), np.abs(fitted[:, 1] - tx))
        rad = dist.min()
        if rad > cfg["borrow_max_tiles"]:          # too far from any fitted tile: not tested
            t["model"] = None
            t["src"] = None
            continue
        near = fitted[dist == rad]
        labs = final[near[:, 0], near[:, 1]]
        cnt = np.bincount(labs, minlength=len(MODELS))
        top = np.flatnonzero(cnt == cnt.max())
        # final[src] always has params in src (smoothing above keeps to fitted models), so any
        # source picked here can supply the parameters.
        # tie: the safer (larger z_t) model, judged on the first neighbour that has it
        def zt_of(k):
            src = near[labs == k][0]
            return grid_t[(int(src[0]), int(src[1]))]["zt_ref"][MODELS[k]]
        k = max(top, key=zt_of)
        src = near[labs == k][0]
        t["model"] = MODELS[k]
        t["src"] = (int(src[0]), int(src[1]))

    pfas = sorted({cfg["pfa_seed"], cfg["pfa_grow"], *cfg["pfa_eval"]})
    for t in tiles:
        if t["model"] is None:
            t["params"] = None
            t["z_t"] = {}
            continue
        s = grid_t[t["src"]]
        f = s["fits"][t["model"]]
        t["params"] = f["params"]
        t["tail_rmse"] = round(f["tail_rmse"], 4)
        t["aic"] = round(f["aic"], 2) if not t["borrowed"] else None
        zs = z_threshold(t["model"], f["params"], pfas, cfg["censor_z"])
        t["z_t"] = {f"{p:.0e}": round(float(v), 4) for p, v in zip(pfas, zs)}
        t["z_t_nominal"] = {f"{p:.0e}": round(float(v), 4)
                            for p, v in zip(pfas, z_threshold(t["model"], f["params"], pfas))}

    res = {"z": z.astype(np.float32), "mean": (m2 + y0).astype(np.float32), "sd": s2.astype(np.float32),
           "tiles": tiles, "tile_px": tile, "tile_shape": (nty, ntx), "cfg": cfg}
    # ---- measured exceedance per tile and overall
    tested = np.isfinite(z)
    over = {}
    for p in cfg["pfa_eval"]:
        zt_img = threshold_image(res, p)
        with np.errstate(invalid="ignore"):
            hit = tested & (z >= zt_img)
        key = f"{p:.0e}"
        over[key] = round(float(hit.sum() / max(tested.sum(), 1)), 8)
        for t in tiles:
            sl = (slice(t["r0"], t["r1"]), slice(t["c0"], t["c1"]))
            nt = int(tested[sl].sum())
            t.setdefault("exceed", {})[key] = round(float(hit[sl].sum() / nt), 8) if nt else None
    counts = {m: sum(1 for t in tiles if t["model"] == m) for m in MODELS}
    zts = [t["z_t"].get(f"{cfg['pfa_seed']:.0e}") for t in tiles if t["z_t"]]
    res["info"] = {
        "detector_version": DETECTOR_VERSION,
        "shape": [h, w], "pixel_m": round(float(np.mean(pixel_m)), 3),
        "guard_px": guard, "outer_px": outer, "tile_px": tile, "thinned_fit": thin,
        "water_fraction": round(float(water.mean()), 4),
        "tested_fraction": round(float(tested.mean()), 4),
        "censored_fraction": round(float((censor & water).sum() / max(water.sum(), 1)), 4),
        "tiles": len(tiles), "tiles_borrowed": sum(t["borrowed"] for t in tiles),
        "model_counts": counts,
        "model_counts_unsmoothed": {m: sum(1 for t in tiles if t["own_model"] == m) for m in MODELS},
        "z_t_seed": ({"min": round(min(zts), 3), "median": round(float(np.median(zts)), 3),
                      "max": round(max(zts), 3)} if zts else None),
        "exceedance": over,
        "seconds": round(time.perf_counter() - t_start, 2),
    }
    return res


def threshold_image(res: dict, pfa: float) -> np.ndarray:
    """Per-pixel applied z_t at ``pfa`` (constant over each tile; NaN where a tile has no model)."""
    nty, ntx = res["tile_shape"]
    cz = res["cfg"]["censor_z"]
    key = f"{pfa:.0e}"
    zt = np.full((nty, ntx), np.nan)
    for t in res["tiles"]:
        if t["model"] is None:
            continue
        v = t["z_t"].get(key)
        if v is None:
            v = z_threshold(t["model"], t["params"], pfa, cz)
        zt[t["ty"], t["tx"]] = v
    tile = res["tile_px"]
    h, w = res["z"].shape
    return np.repeat(np.repeat(zt, tile, axis=0), tile, axis=1)[:h, :w]


def tile_table(res: dict) -> list[dict]:
    """Compact per-tile rows (no fit details) for output."""
    keys = ("ty", "tx", "model", "own_model", "borrowed", "z_t", "tail_rmse", "aic", "params",
            "usable_fraction", "n_fit", "vv_median_db", "y_sd", "sea_state", "exceed")
    return [{k: t.get(k) for k in keys} for t in res["tiles"]]
