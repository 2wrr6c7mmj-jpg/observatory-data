"""Ship-candidate detection on a Sentinel-1 sigma0 chip (VV CFAR + VH check).

Two-parameter (Gaussian) CFAR on VV linear sigma0, the background taken from
a square ring outside a guard band and from water pixels only. Detected
pixels are grouped (8-connected), groups under ``min_pixels`` are dropped,
and a group is kept only if its mean VH is ``vh_min_ratio`` times the VH
background. Every detection is a candidate (unconfirmed).
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .grid import box_sum, dilate, label

DETECTOR_VERSION = "cfar-v1"
DEFAULTS = {
    "pfa": 1e-5,             # false-alarm rate per pixel (Gaussian approximation)
    "guard_m": 70.0,         # guard band around the cell under test
    "background_m": 150.0,   # width of the background ring outside the guard band
    "min_background_fraction": 0.25,  # ring must be at least this much water
    "min_pixels": 3,
    "vh_min_ratio": 2.0,     # cluster mean VH / background VH (2.0 = +3 dB)
    "quay_band_m": 50.0,     # within this distance of land = quay band
}
LENGTH_CLASSES = (("lt50", 0, 50), ("50_100", 50, 100), ("100_200", 100, 200), ("ge200", 200, math.inf))


def length_class(length_m: float) -> str:
    for name, lo, hi in LENGTH_CLASSES:
        if lo <= length_m < hi:
            return name
    return LENGTH_CLASSES[-1][0]


def background(x: np.ndarray, water: np.ndarray, guard: int, outer: int):
    """Mean, std and count of ``x`` over water pixels in the ring guard < d <= outer (Chebyshev)."""
    xw = np.where(water, x, 0.0)
    w = water.astype(np.float64)
    n = box_sum(w, outer) - box_sum(w, guard)
    s1 = box_sum(xw, outer) - box_sum(xw, guard)
    s2 = box_sum(xw * xw, outer) - box_sum(xw * xw, guard)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu, np.sqrt(var), n


def detect(vv_db: np.ndarray, vh_db: np.ndarray, land: np.ndarray, pixel_m: tuple[float, float],
           cfg: dict | None = None):
    """Return (detections, info). ``land`` is True on land/structures; NaN dB = no data."""
    cfg = {**DEFAULTS, **(cfg or {})}
    ry, rx = pixel_m
    px = (ry + rx) / 2
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    water = valid & ~np.asarray(land, bool)
    vv = np.where(water, 10 ** (np.nan_to_num(vv_db) / 10), 0.0)
    vh = np.where(water, 10 ** (np.nan_to_num(vh_db) / 10), 0.0)
    guard = int(math.ceil(cfg["guard_m"] / px))
    outer = guard + int(math.ceil(cfg["background_m"] / px))
    ring_area = (2 * outer + 1) ** 2 - (2 * guard + 1) ** 2
    t = NormalDist().inv_cdf(1 - cfg["pfa"])
    mu, sd, n = background(vv, water, guard, outer)
    enough = n >= cfg["min_background_fraction"] * ring_area
    with np.errstate(invalid="ignore"):
        hit = water & enough & (vv > mu + t * sd)
    mu_vh, _sd_vh, _n = background(vh, water, guard, outer)

    quay_px = cfg["quay_band_m"] / px
    quay = dilate(np.asarray(land, bool), quay_px) & ~np.asarray(land, bool)
    labels, nlab = label(hit, connectivity=8)
    dets = []
    dropped_small = dropped_vh = 0
    if nlab:
        idx = np.flatnonzero(labels.ravel())
        lab_of = labels.ravel()[idx]
        order = np.argsort(lab_of, kind="stable")
        idx, lab_of = idx[order], lab_of[order]
        bounds = np.flatnonzero(np.diff(lab_of)) + 1
        for grp in np.split(idx, bounds):
            if len(grp) < cfg["min_pixels"]:
                dropped_small += 1
                continue
            rr, cc = np.divmod(grp, labels.shape[1])
            bg_vh = float(np.nanmean(mu_vh.ravel()[grp]))
            m_vh = float(vh.ravel()[grp].mean())
            ratio = m_vh / bg_vh if bg_vh > 0 else math.inf
            if not ratio >= cfg["vh_min_ratio"]:
                dropped_vh += 1
                continue
            y = rr * ry
            x = cc * rx
            length, width = _extent(x, y, px)
            r0, c0 = float(rr.mean()), float(cc.mean())
            zone = "quay" if quay[int(round(r0)), int(round(c0))] else "harbour"
            bg_vv = float(np.nanmean(mu.ravel()[grp]))
            dets.append({
                "row": r0, "col": c0,
                "rmin": int(rr.min()), "rmax": int(rr.max()), "cmin": int(cc.min()), "cmax": int(cc.max()),
                "pixels": int(len(grp)),
                "length_m": round(length), "width_m": round(width),
                "length_class": length_class(length),
                "zone": zone,
                "peak_vv_db": round(float(10 * np.log10(vv.ravel()[grp].max())), 1),
                "contrast_vv_db": round(float(10 * np.log10(vv.ravel()[grp].mean() / bg_vv)), 1) if bg_vv > 0 else None,
                "vh_ratio_db": round(10 * math.log10(ratio), 1) if math.isfinite(ratio) else None,
            })
    info = {
        "valid_fraction": round(float(valid.mean()), 4),
        "water_fraction": round(float(water.mean()), 4),
        "threshold_sigma": round(t, 3),
        "guard_px": guard, "outer_px": outer,
        "hit_pixels": int(hit.sum()), "clusters": int(nlab),
        "dropped_small": dropped_small, "dropped_vh": dropped_vh,
    }
    return dets, info


def _extent(x: np.ndarray, y: np.ndarray, px: float) -> tuple[float, float]:
    """Length and width (m) along the principal axes, including one pixel of footprint."""
    if len(x) < 2:
        return px, px
    pts = np.stack([x - x.mean(), y - y.mean()])
    cov = pts @ pts.T / len(x)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    minor = vecs[:, 0]
    pa = major @ pts
    pb = minor @ pts
    return float(pa.max() - pa.min() + px), float(pb.max() - pb.min() + px)
