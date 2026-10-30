"""Ship-candidate detection on a Sentinel-1 sigma0 cut-out (仕様書 v0 第3節, 試行版).

Trial limits (documented in docs/data/sat/CHANGES.md):
  - water mask comes from the image itself (smoothed VV darker than WATER_DB),
    not yet from the OSM coastline;
  - no fixed-object mask (needs ~20 scenes) and no azimuth-ambiguity removal.
Every detection is a 候補 (candidate); only human review makes it 確認済み.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

DETECTOR_VERSION = "cfar-trial-1"
PFA = 1e-5
K = 4.265          # one-sided Gaussian quantile for PFA 1e-5 (two-parameter CFAR)
GUARD_M = 70       # guard band around the test cell
BG_M = 250         # outer half-size of the background window
WATER_DB = -15.0   # smoothed VV below this is open water (trial water mask)
QUAY_M = 50        # quay band: water within this distance of land
MIN_PIXELS = 3
LENGTH_BINS = [(50, "〜50m"), (100, "50〜100m"), (200, "100〜200m"), (float("inf"), "200m〜")]


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-6))


def water_mask(vv, valid, px_m):
    smooth = ndi.median_filter(db(vv), size=max(3, int(round(60 / px_m)) | 1))
    water = (smooth < WATER_DB) & valid
    water = ndi.binary_opening(water, iterations=2)
    lab, n = ndi.label(water)               # drop tiny water specks (paddies, shadows)
    if n:
        sizes = ndi.sum(water, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sizes >= (300 / px_m) ** 2))
        water &= keep
    return water


def ring_stats(img, mask, guard, outer):
    """Mean and std of img over mask pixels in a square ring (outer half-size, minus guard half-size)."""
    def boxsum(a, h):
        return ndi.uniform_filter(a, size=2 * h + 1, mode="constant") * (2 * h + 1) ** 2
    m = mask.astype(float)
    v = np.where(mask, img, 0.0)
    n = boxsum(m, outer) - boxsum(m, guard)
    s1 = boxsum(v, outer) - boxsum(v, guard)
    s2 = boxsum(v * v, outer) - boxsum(v * v, guard)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    return mean, std, n


def length_class(length_m):
    return next(label for limit, label in LENGTH_BINS if length_m < limit)


def detect(vv, vh, valid, px_m):
    """Candidates as dicts (row/col centroid, pixels, peak dB, length, zone) plus the masks."""
    valid = valid.astype(bool) & np.isfinite(vv) & np.isfinite(vh)
    water = water_mask(vv, valid, px_m)
    land = valid & ~water
    quay = water & ndi.binary_dilation(land, iterations=max(1, int(round(QUAY_M / px_m))))
    guard, outer = max(1, int(round(GUARD_M / px_m))), max(3, int(round(BG_M / px_m)))
    # Background statistics over water only; bright targets would inflate them, so use a clipped copy.
    clip = np.minimum(vv, np.nanpercentile(vv[water], 99)) if water.any() else vv
    mean, std, n = ring_stats(clip, water, guard, outer)
    hit = water & (n >= 20) & (vv > mean + K * std)
    vh_mean, vh_std, _ = ring_stats(np.minimum(vh, np.nanpercentile(vh[water], 99)) if water.any() else vh, water, guard, outer)
    lab, count = ndi.label(hit, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        blob = lab[sl] == i
        npx = int(blob.sum())
        if npx < MIN_PIXELS:
            continue
        rows, cols = np.nonzero(blob)
        rows, cols = rows + sl[0].start, cols + sl[1].start
        if not (vh[rows, cols] > vh_mean[rows, cols] + 3 * vh_std[rows, cols]).any():
            continue  # VH must also stand out from the water around it
        if npx >= 2:
            cov = np.cov(np.vstack([rows, cols]) * px_m)
            major = 4 * np.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0))  # ~ full length of an elongated blob
        else:
            major = px_m
        length = max(major, px_m * np.sqrt(npx))
        out.append(dict(row=float(rows.mean()), col=float(cols.mean()), pixels=npx,
                        peak_vv_db=round(float(db(vv[rows, cols]).max()), 1),
                        peak_vh_db=round(float(db(vh[rows, cols]).max()), 1),
                        length_m=round(float(length)), length_class=length_class(length),
                        zone="岸壁帯" if quay[rows, cols].any() else "港内"))
    return out, {"water": water, "land": land, "quay": quay}
