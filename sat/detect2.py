"""cfar-v2 PR2 (experimental): ship candidates from the log-domain CFAR of ``sat/cfar2.py``.

Experimental. Nothing in ``sat/run.py`` (daily run, ZJG counts, public data) calls this module;
cfar-v1 (``sat/detect.py``) is unchanged. Depends on numpy, ``sat.grid`` and ``sat.cfar2`` only.

Steps (design ``cfar-v2`` 3.4-3.6):

1. ``cfar2.cfar2`` on VV and on VH (same windows). Pixels within ``edge_px`` (3) of no data are
   not tested.
2. Hysteresis: seeds ``z >= z_t(pfa_seed=1e-7)``; the 8-connected regions of
   ``z >= z_t(pfa_grow=1e-3)`` that contain a seed are kept.
3. Binary 3x3 closing, hole filling, area filter (2 px at >= 15 m pixels, 4 px finer). No opening
   (the design's directional 2x1 opening at 10 m is not implemented).
4. 8-connected components (``grid.label``).
5. White top-hat (dB image minus a 400 m square opening, separable min/max filters) recorded as
   ``tophat_db``; not used to decide.
6. Confidence from VH: ``A`` if the component's max ``z_vh >= z_t_vh(1e-3)`` or its mean VH is
   >= +3 dB over the background; else ``B`` if ``length_m >= 80`` (VV only; recorded, not counted);
   else dropped.
7. Shape: length and width along the principal axes (one pixel of footprint included), heading of
   the major axis (degrees from true north, 0-180), area, length/width, fill (pixels / rotated
   bounding rectangle).
8. Side lobes: a brighter component (by peak dB) is a parent; a component whose centroid lies in
   the cross centred on the parent's peak pixel -- along azimuth and along range, half-width
   ``sidelobe_band_px`` (2) px, half-length ``sidelobe_len_m`` (600 m) -- and whose peak is at least
   ``sidelobe_drop_db`` (13 dB) below the parent's is marked ``sidelobe_of=<parent index>``.
   Azimuth = the platform heading (``heading_deg``); without it the image axes are used.
9. Azimuth ambiguities: ``delta = lambda * R * PRF / (2 V)`` (R = slant range from the incidence
   angle, PRF from the nominal per-sub-swath table ``IW_PRF_HZ`` below, V = orbital speed). A
   component within ``ambiguity_tol_m`` (400 m) of ``parent +- delta`` along the azimuth, whose VV
   peak is >= 15 dB and VH peak >= 10 dB below the parent's, is marked ``ambiguity_of=<parent>``.
   Needs ``lia`` (incidence, deg) and ``heading_deg``; skipped otherwise (``info``).
10. Wake (recorded only): mean dB on 3-px-wide lines continuing the major axis 1.5 km beyond each
    end, minus the local background (the darker end) -> ``wake_dark_db``.
11. ``large`` flag: ``length_m >= 80 and width_m >= 12 and lw_ratio >= 2.5 and confidence == "A"
    and not static`` (provisional numbers; lengths are blurred by about +20 m by the resolution).

Excluded candidates (side lobe, ambiguity) are returned with the reason set and are listed in
``info["excluded"]``; use :func:`kept` for the rest. ``lon``, ``lat`` and ``acquired_utc`` are None
until :func:`add_geo` fills them from a ``grid.Grid``.

Nominal IW sub-swath table (``IW_PRF_HZ``, ``IW_INCIDENCE_DEG``)
-----------------------------------------------------------------
Sources checked (ESA public documents):

* ESA, "Sentinel-1 System Data", PolSAR 2013 workshop (earth.esa.int/eogateway/documents/20142/
  37627/PolSAR-2013-Sentinel-1-System-Data.pdf): orbit ~698 km (``ORBIT_HEIGHT_M``), PRF range
  1000-3000 Hz, sub-swaths IW1-IW3. No per-sub-swath PRF.
* Geudtner et al., "Sentinel-1 System Overview and Performance", ESA Living Planet Symposium 2013
  (ESA SP-722): one PRF per (sub-)swath, IW incidence about 30-42 deg (Table 1), no PRF values.
* Centre frequency 5.405 GHz (``WAVELENGTH_M`` = c / 5.405 GHz).

**Provisional (not verified against an ESA document or a product annotation):** the PRF values
(IW1 1717.13, IW2 1451.37, IW3 1685.82 Hz; values commonly found in IW product annotation XML,
quoted from memory) and the sub-swath incidence bounds (29.1/36.0/41.6/46.0 deg). To be replaced
by values read from product annotation XML (``downlinkInformation/prf``) and checked on a bright
ship's real ambiguity.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import cfar2 as c2
from .grid import box_sum, dilate, label

DETECTOR_VERSION = "cfar-v2"
PROCESSING_VERSION = "cfar2-pr2"

# --- nominal Sentinel-1 IW geometry (see module doc: PRF and incidence bounds are provisional)
WAVELENGTH_M = 299_792_458.0 / 5.405e9
ORBIT_HEIGHT_M = 698_000.0
EARTH_RADIUS_M = 6_371_000.0
GM = 3.986004418e14
INCLINATION_DEG = 98.18
IW_PRF_HZ = {"IW1": 1717.13, "IW2": 1451.37, "IW3": 1685.82}          # provisional
IW_INCIDENCE_DEG = {"IW1": (29.1, 36.0), "IW2": (36.0, 41.6), "IW3": (41.6, 46.0)}  # provisional

DEFAULTS_DET2 = {
    "edge_px": 3,
    "coast_band_m": 1000.0,
    "quay_band_m": 50.0,
    "min_area_px": None,          # None: 2 px at >= 15 m pixels, 4 px finer
    "tophat_m": 400.0,
    "vh_ratio_min_db": 3.0,
    "pfa_vh": 1e-3,
    "b_min_length_m": 80.0,
    "sidelobe_band_px": 2.0,
    "sidelobe_len_m": 600.0,
    "sidelobe_drop_db": 13.0,
    "ambiguity_tol_m": 400.0,
    "ambiguity_drop_db": 15.0,
    "ambiguity_vh_drop_db": 10.0,   # "VH weak": VH peak this far below the parent's (provisional)
    "incidence_margin_deg": 0.5,    # near a sub-swath boundary both PRFs are tried
    "wake_len_m": 1500.0,
    "wake_band_px": 1,              # +-1 px = 3 px wide
    "wake_min_samples": 10,
    "large_length_m": 80.0,         # provisional
    "large_width_m": 12.0,
    "large_lw": 2.5,
}


# ---------------------------------------------------------------- geometry

def platform_heading(direction: str, lat_deg: float) -> float:
    """Nominal ground-track heading (deg from north) of S1 at ``lat_deg`` for "ascending"/"descending".

    Spherical Earth, inclination 98.18 deg, Earth rotation subtracted (about 348 deg ascending and
    192 deg descending at 22N).
    """
    phi = math.radians(lat_deg)
    s = math.cos(math.radians(INCLINATION_DEG)) / math.cos(phi)
    a = math.asin(max(-1.0, min(1.0, s)))
    if direction.lower().startswith("d"):
        a = math.pi - a
    elif not direction.lower().startswith("a"):
        raise ValueError(direction)
    vs = math.sqrt(GM / (EARTH_RADIUS_M + ORBIT_HEIGHT_M))
    vg = vs * EARTH_RADIUS_M / (EARTH_RADIUS_M + ORBIT_HEIGHT_M)
    east = vg * math.sin(a) - 7.2921159e-5 * EARTH_RADIUS_M * math.cos(phi)
    north = vg * math.cos(a)
    return math.degrees(math.atan2(east, north)) % 360.0


def slant_range_m(incidence_deg: float) -> float:
    """Slant range for an incidence angle on a spherical Earth (orbit height ``ORBIT_HEIGHT_M``)."""
    th = math.radians(incidence_deg)
    rs = EARTH_RADIUS_M + ORBIT_HEIGHT_M
    look = math.asin(EARTH_RADIUS_M * math.sin(th) / rs)
    beta = th - look
    return math.sqrt(EARTH_RADIUS_M ** 2 + rs ** 2 - 2 * EARTH_RADIUS_M * rs * math.cos(beta))


def subswaths(incidence_deg: float, margin: float = 0.5) -> list[str]:
    return [k for k, (lo, hi) in IW_INCIDENCE_DEG.items() if lo - margin <= incidence_deg <= hi + margin]


def ambiguity_offset_m(incidence_deg: float, swath: str | None = None) -> float:
    """Along-track displacement of the first azimuth ambiguity, lambda R PRF / (2 V_orbit)."""
    swath = swath or (subswaths(incidence_deg, 0.0) or ["IW1" if incidence_deg < 36 else "IW3"])[0]
    v = math.sqrt(GM / (EARTH_RADIUS_M + ORBIT_HEIGHT_M))
    return WAVELENGTH_M * slant_range_m(incidence_deg) * IW_PRF_HZ[swath] / (2 * v)


def _unit(heading_deg: float) -> tuple[float, float]:
    """(east, north) unit vector of a heading."""
    h = math.radians(heading_deg)
    return math.sin(h), math.cos(h)


# ---------------------------------------------------------------- morphology (numpy)

def _shift_or(m: np.ndarray, r: int) -> np.ndarray:
    h, w = m.shape
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= p[r + dy:r + dy + h, r + dx:r + dx + w]
    return out


def closing3(m: np.ndarray) -> np.ndarray:
    """Binary closing with a 3x3 square (outside the image counts as background for dilation)."""
    d = _shift_or(m, 1)
    return ~_shift_or(~d, 1)


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill background regions (4-connected) that do not reach the border."""
    bg, n = label(~m, connectivity=4)
    if n == 0:
        return m.copy()
    edge = np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    hole = (bg > 0) & ~np.isin(bg, edge)
    return m | hole


def _run_filter(a: np.ndarray, w: int, axis: int, fn, pad_value: float) -> np.ndarray:
    """Centred running min/max of odd width ``w`` along ``axis`` (doubling; O(log w) array ops)."""
    half = w // 2
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    p = np.concatenate([np.full((half,) + a.shape[1:], pad_value), a, np.full((half,) + a.shape[1:], pad_value)])
    acc = p.copy()          # acc[i] = fn(p[i : i + k])
    k = 1
    while 2 * k <= w:
        acc[:len(acc) - k] = fn(acc[:len(acc) - k], acc[k:])
        k *= 2
    out = fn(acc[0:n], acc[w - k:w - k + n])
    return np.moveaxis(out, 0, axis)


def min_filter(a: np.ndarray, w: int) -> np.ndarray:
    return _run_filter(_run_filter(a, w, 0, np.minimum, np.inf), w, 1, np.minimum, np.inf)


def max_filter(a: np.ndarray, w: int) -> np.ndarray:
    return _run_filter(_run_filter(a, w, 0, np.maximum, -np.inf), w, 1, np.maximum, -np.inf)


def white_tophat(db: np.ndarray, water: np.ndarray, w: int) -> np.ndarray:
    """``db`` minus its grey opening with a w x w square; non-water pixels are ignored."""
    ero = min_filter(np.where(water, db, np.inf), w)
    ero = np.where(water, ero, -np.inf)
    return db - max_filter(ero, w)


# ---------------------------------------------------------------- shape

def shape_features(rr: np.ndarray, cc: np.ndarray, pixel_m) -> dict:
    """Length, width (m, incl. one pixel of footprint), heading of the major axis (deg from north,
    0-180), area (m2), length/width, fill (pixels / rotated rectangle)."""
    ry, rx = pixel_m
    px = (ry + rx) / 2
    x = cc * rx                      # east
    y = -rr * ry                     # north (rows go south)
    area = len(rr) * ry * rx
    if len(rr) < 2:
        return {"length_m": px, "width_m": px, "heading_deg": None, "area_m2": area, "lw_ratio": 1.0,
                "fill": round(area / (px * px), 3)}
    pts = np.stack([x - x.mean(), y - y.mean()])
    cov = pts @ pts.T / len(x)
    _vals, vecs = np.linalg.eigh(cov)
    major, minor = vecs[:, 1], vecs[:, 0]
    pa, pb = major @ pts, minor @ pts
    length = float(pa.max() - pa.min() + px)
    width = float(pb.max() - pb.min() + px)
    heading = math.degrees(math.atan2(major[0], major[1])) % 180.0
    return {"length_m": length, "width_m": width, "heading_deg": round(heading, 1), "area_m2": area,
            "lw_ratio": round(length / width, 2), "fill": round(min(area / (length * width), 1.0), 3),
            "_axis": (float(major[0]), float(major[1]))}


def length_class(length_m: float) -> str:
    for name, lo, hi in (("lt50", 0, 50), ("50_100", 50, 100), ("100_200", 100, 200)):
        if lo <= length_m < hi:
            return name
    return "ge200"


def _at(a, r: int, c: int):
    if a is None:
        return None
    if np.ndim(a) == 0:
        return float(a)
    v = float(a[r, c])
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- main entry

def detect_v2(vv_db: np.ndarray, vh_db: np.ndarray, land: np.ndarray, pixel_m, cfg: dict | None = None, *,
              lia=None, heading_deg: float | None = None, static: np.ndarray | None = None):
    """Return (dets, info) like ``detect.detect`` (cfar-v1) with the cfar-v2 fields added.

    ``vv_db``/``vh_db``: dB, NaN = no data. ``land``: True on land. ``pixel_m``: (row m, col m).
    ``lia``: incidence angle (deg), scalar or image. ``heading_deg``: platform heading (deg from
    north; see :func:`platform_heading`). ``static``: True on known fixed structures (already dilated).
    """
    t0 = time.perf_counter()
    cfg = {**c2.DEFAULTS_V2, **DEFAULTS_DET2, **(cfg or {})}
    ry, rx = float(pixel_m[0]), float(pixel_m[1])
    px = (ry + rx) / 2
    vv_db = np.asarray(vv_db, float)
    vh_db = np.asarray(vh_db, float)
    land = np.zeros(vv_db.shape, bool) if land is None else np.asarray(land, bool)
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    water = valid & ~land
    nodata_band = dilate(~valid, cfg["edge_px"]) if (~valid).any() else np.zeros_like(valid)
    vv_in = np.where(valid, vv_db, np.nan)
    vh_in = np.where(valid, vh_db, np.nan)

    rv = c2.cfar2(vv_in, land, (ry, rx), cfg, invalid=nodata_band)
    rh = c2.cfar2(vh_in, land, (ry, rx), cfg, invalid=nodata_band)
    z, zh = rv["z"], rh["z"]
    zt_seed = c2.threshold_image(rv, cfg["pfa_seed"])
    zt_grow = c2.threshold_image(rv, cfg["pfa_grow"])
    zt_vh = c2.threshold_image(rh, cfg["pfa_vh"])
    with np.errstate(invalid="ignore"):
        seed = z >= zt_seed
        grow = (z >= zt_grow) | seed

    # ---- hysteresis
    glab, ng = label(grow, 8)
    keep_ids = np.unique(glab[seed])
    keep_ids = keep_ids[keep_ids > 0]
    hyst = np.isin(glab, keep_ids) if len(keep_ids) else np.zeros_like(grow)

    # ---- closing, hole filling (per component box), area filter
    closed = closing3(hyst) & water
    clab, nc = label(closed, 8)
    min_area = cfg["min_area_px"] or (2 if px >= 15.0 else 4)
    comps = []
    dropped_small = 0
    if nc:
        flat = clab.ravel()
        idx = np.flatnonzero(flat)
        lab_of = flat[idx]
        order = np.argsort(lab_of, kind="stable")
        idx, lab_of = idx[order], lab_of[order]
        for grp in np.split(idx, np.flatnonzero(np.diff(lab_of)) + 1):
            rr, cc = np.divmod(grp, clab.shape[1])
            r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            box = np.zeros((r1 - r0 + 2, c1 - c0 + 2), bool)
            box[rr - r0 + 1, cc - c0 + 1] = True
            filled = fill_holes(box)
            fr, fc = np.nonzero(filled)
            fr, fc = fr + r0 - 1, fc + c0 - 1
            ok = water[fr, fc]
            fr, fc = fr[ok], fc[ok]
            if len(fr) < min_area:
                dropped_small += 1
                continue
            comps.append((fr, fc))

    # ---- linear backgrounds for v1-style contrast / VH ratio (ring of pass-2 pixels)
    guard, outer, _tile, _thin = c2.window_px((ry, rx), cfg)
    with np.errstate(invalid="ignore"):
        cens_v = water & ~dilate(np.nan_to_num(z, nan=0.0) > cfg["censor_z"], cfg["censor_dilate_px"])
        cens_h = water & ~dilate(np.nan_to_num(zh, nan=0.0) > cfg["censor_z"], cfg["censor_dilate_px"])
    vv_lin = np.where(water, 10 ** (np.nan_to_num(vv_db) / 10), 0.0)
    vh_lin = np.where(water, 10 ** (np.nan_to_num(vh_db) / 10), 0.0)
    bg_vv, _s, _n = c2.ring_stats(vv_lin, cens_v, guard, outer)
    bg_vh, _s, _n = c2.ring_stats(vh_lin, cens_h, guard, outer)
    bg_db = rv["mean"] / c2.LN10_10            # local background mean of dB (log mean)

    coast_px = int(math.ceil(cfg["coast_band_m"] / px))
    near_coast_img = box_sum(land.astype(np.float64), coast_px) > 0 if land.any() else np.zeros_like(land)
    quay_img = dilate(land, cfg["quay_band_m"] / px) & ~land if land.any() else np.zeros_like(land)
    edge_img = dilate(nodata_band, 2) if nodata_band.any() else nodata_band
    tile = rv["tile_px"]
    tiles_v = {(t["ty"], t["tx"]): t for t in rv["tiles"]}
    th_w = 2 * int(round(cfg["tophat_m"] / px / 2)) + 1
    plat = heading_deg
    u_az = _unit(plat) if plat is not None else (0.0, 1.0)         # (east, north)
    u_rg = (u_az[1], -u_az[0])

    dets = []
    dropped_vh = 0
    for fr, fc in comps:
        sf = shape_features(fr, fc, (ry, rx))
        length, width = sf["length_m"], sf["width_m"]
        zc = z[fr, fc]
        k = int(np.nanargmax(np.where(np.isfinite(zc), zc, -np.inf)))
        pr, pc = int(fr[k]), int(fc[k])
        vh_z = float(np.nanmax(zh[fr, fc])) if np.isfinite(zh[fr, fc]).any() else None
        zt_vh_here = float(np.nanmax(zt_vh[fr, fc])) if np.isfinite(zt_vh[fr, fc]).any() else math.inf
        m_vh = float(vh_lin[fr, fc].mean())
        b_vh = float(np.nanmean(bg_vh[fr, fc]))
        ratio = m_vh / b_vh if b_vh > 0 else math.inf
        ratio_db = 10 * math.log10(ratio) if math.isfinite(ratio) and ratio > 0 else None
        if (vh_z is not None and vh_z >= zt_vh_here) or (ratio_db is not None and ratio_db >= cfg["vh_ratio_min_db"]) \
                or ratio == math.inf:
            conf = "A"
        elif length >= cfg["b_min_length_m"]:
            conf = "B"
        else:
            dropped_vh += 1
            continue
        b_vv = float(np.nanmean(bg_vv[fr, fc]))
        t = tiles_v.get((pr // tile, pc // tile), {})
        # top-hat on a crop around the component
        m = th_w + 1
        a0, a1 = max(0, fr.min() - m), min(z.shape[0], fr.max() + m + 1)
        b0, b1 = max(0, fc.min() - m), min(z.shape[1], fc.max() + m + 1)
        crop_db = np.where(water[a0:a1, b0:b1], vv_db[a0:a1, b0:b1], 0.0)
        th = white_tophat(crop_db, water[a0:a1, b0:b1], th_w)
        r0c, c0c = float(fr.mean()), float(fc.mean())
        ri, ci = int(round(r0c)), int(round(c0c))
        near_coast = bool(near_coast_img[ri, ci])
        d = {
            # cfar-v1 fields
            "row": r0c, "col": c0c,
            "rmin": int(fr.min()), "rmax": int(fr.max()), "cmin": int(fc.min()), "cmax": int(fc.max()),
            "pixels": int(len(fr)),
            "length_m": round(length), "width_m": round(width),
            "length_class": length_class(length),
            "zone": "quay" if quay_img[ri, ci] else ("coast" if near_coast else "open"),
            "peak_vv_db": round(float(vv_db[fr, fc].max()), 1),
            "contrast_vv_db": (round(float(10 * np.log10(vv_lin[fr, fc].mean() / b_vv)), 1)
                               if b_vv > 0 else None),
            "vh_ratio_db": round(ratio_db, 1) if ratio_db is not None else None,
            # cfar-v2 fields
            "lon": None, "lat": None, "acquired_utc": None,
            "heading_deg": sf["heading_deg"], "area_m2": round(sf["area_m2"]),
            "lw_ratio": sf["lw_ratio"], "fill": sf["fill"],
            "z_peak": round(float(z[pr, pc]), 2),
            "z_t": round(float(zt_seed[pr, pc]), 3) if np.isfinite(zt_seed[pr, pc]) else None,
            "model": t.get("model"), "tail_rmse": t.get("tail_rmse"),
            "vh_z": round(vh_z, 2) if vh_z is not None else None,
            "tophat_db": round(float(th[fr - a0, fc - b0].max()), 1),
            "incidence_deg": (round(_at(lia, ri, ci), 2) if _at(lia, ri, ci) is not None else None),
            "sea_state": t.get("sea_state"),
            "near_coast": near_coast,
            "static": bool(static is not None and np.asarray(static, bool)[fr, fc].any()),
            "edge": bool(edge_img[fr, fc].any()),
            "sidelobe_of": None, "ambiguity_of": None,
            "wake_dark_db": None,
            "confidence": conf, "large": False,
            "pixel_m": round(px, 2),
            "detector_version": DETECTOR_VERSION, "processing_version": PROCESSING_VERSION,
            "_peak": (pr, pc), "_axis": sf.get("_axis"), "_pix": (fr, fc),
            "_peak_vh_db": float(np.nanmax(vh_db[fr, fc])),
            "_peak_db": float(vv_db[fr, fc].max()),
        }
        dets.append(d)

    # ---- wake (recorded only)
    comp_mask = np.zeros(z.shape, bool)
    for d in dets:
        comp_mask[d["_pix"]] = True
    comp_mask = _shift_or(comp_mask, 1)
    for d in dets:
        if d["_axis"] is None:
            continue
        ax_e, ax_n = d["_axis"]
        half = d["length_m"] / 2
        steps = np.arange(half + px, half + cfg["wake_len_m"] + 1e-9, px)
        best = None
        for sgn in (1.0, -1.0):
            vals, bgs = [], []
            for off in range(-cfg["wake_band_px"], cfg["wake_band_px"] + 1):
                e = sgn * steps * ax_e + off * px * ax_n
                n = sgn * steps * ax_n - off * px * ax_e
                rr = np.round(d["row"] - n / ry).astype(int)
                cc = np.round(d["col"] + e / rx).astype(int)
                ok = (rr >= 0) & (rr < z.shape[0]) & (cc >= 0) & (cc < z.shape[1])
                rr, cc = rr[ok], cc[ok]
                ok = water[rr, cc] & ~comp_mask[rr, cc] & np.isfinite(bg_db[rr, cc])
                vals.append(vv_db[rr[ok], cc[ok]])
                bgs.append(bg_db[rr[ok], cc[ok]])
            v = np.concatenate(vals)
            if v.size >= cfg["wake_min_samples"]:
                diff = float(v.mean() - np.concatenate(bgs).mean())
                best = diff if best is None else min(best, diff)
        d["wake_dark_db"] = round(best, 2) if best is not None else None

    # ---- side lobes and azimuth ambiguities
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["_peak_db"])
    band = cfg["sidelobe_band_px"] * px
    amb_ok = lia is not None and heading_deg is not None
    deltas_used = {}
    for rank, i in enumerate(order):
        P = dets[i]
        if P["sidelobe_of"] is not None or P["ambiguity_of"] is not None:
            continue
        prow, pcol = P["_peak"]
        deltas = []
        if amb_ok:
            inc = _at(lia, int(round(P["row"])), int(round(P["col"])))
            if inc is not None:
                for sw in subswaths(inc, cfg["incidence_margin_deg"]) or []:
                    deltas.append(ambiguity_offset_m(inc, sw))
                    deltas_used[sw] = round(deltas[-1])
        for j in order[rank + 1:]:
            C = dets[j]
            if C["sidelobe_of"] is not None or C["ambiguity_of"] is not None:
                continue
            de = (C["col"] - pcol) * rx
            dn = -(C["row"] - prow) * ry
            along = de * u_az[0] + dn * u_az[1]
            across = de * u_rg[0] + dn * u_rg[1]
            drop = P["_peak_db"] - C["_peak_db"]
            reach = cfg["sidelobe_len_m"] + C["length_m"] / 2      # the piece may start inside the arm
            if drop >= cfg["sidelobe_drop_db"] and (
                    (abs(across) <= band and abs(along) <= reach) or
                    (abs(along) <= band and abs(across) <= reach)):
                C["sidelobe_of"] = i
                continue
            if deltas and drop >= cfg["ambiguity_drop_db"] and \
                    P["_peak_vh_db"] - C["_peak_vh_db"] >= cfg["ambiguity_vh_drop_db"]:
                for dl in deltas:
                    if min(math.hypot(along - dl, across), math.hypot(along + dl, across)) <= cfg["ambiguity_tol_m"]:
                        C["ambiguity_of"] = i
                        break

    excl = {"small": dropped_small, "vh": dropped_vh, "sidelobe": 0, "ambiguity": 0, "static": 0, "near_coast": 0}
    for d in dets:
        excl["sidelobe"] += d["sidelobe_of"] is not None
        excl["ambiguity"] += d["ambiguity_of"] is not None
        excl["static"] += d["static"]
        excl["near_coast"] += d["near_coast"]
        d["large"] = bool(d["length_m"] >= cfg["large_length_m"] and d["width_m"] >= cfg["large_width_m"]
                          and d["lw_ratio"] >= cfg["large_lw"] and d["confidence"] == "A" and not d["static"])
        for k in [k for k in d if k.startswith("_")]:
            del d[k]

    iv = rv["info"]
    info = {
        # cfar-v1 fields
        "valid_fraction": round(float(valid.mean()), 4),
        "water_fraction": round(float(water.mean()), 4),
        "threshold_sigma": (iv["z_t_seed"] or {}).get("median"),
        "guard_px": guard, "outer_px": outer,
        "hit_pixels": int(seed.sum()), "clusters": int(nc),
        "dropped_small": dropped_small, "dropped_vh": dropped_vh,
        # cfar-v2 fields
        "detector_version": DETECTOR_VERSION, "processing_version": PROCESSING_VERSION,
        "grow_pixels": int(grow.sum()), "hysteresis_pixels": int(hyst.sum()),
        "model_counts": iv["model_counts"], "model_counts_vh": rh["info"]["model_counts"],
        "z_t_seed": iv["z_t_seed"], "exceedance": iv["exceedance"],
        "tiles": iv["tiles"], "tiles_borrowed": iv["tiles_borrowed"],
        "tiles_untested": sum(1 for t in rv["tiles"] if t["model"] is None),
        "excluded": excl,
        "platform_heading_deg": heading_deg,
        "sidelobe_axes": "platform" if heading_deg is not None else "image (no heading)",
        "ambiguity_check": ("on" if amb_ok else "skipped (needs lia and heading_deg)"),
        "ambiguity_offset_m": deltas_used,
        "tile_table": c2.tile_table(rv),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return dets, info


def kept(dets: list[dict]) -> list[dict]:
    """Candidates not excluded as side lobe or ambiguity."""
    return [d for d in dets if d["sidelobe_of"] is None and d["ambiguity_of"] is None]


def add_geo(dets: list[dict], grid, acquired_utc: str | None = None) -> list[dict]:
    """Fill ``lon``/``lat`` (pixel centres, as run.py does for v1) and ``acquired_utc``."""
    for d in dets:
        lon, lat = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
        d["lon"], d["lat"] = round(float(lon), 6), round(float(lat), 6)
        d["acquired_utc"] = acquired_utc
    return dets


def to_features(dets: list[dict]) -> list[dict]:
    """GeoJSON Point features (one per candidate); ``lon``/``lat`` must be filled (:func:`add_geo`)."""
    return [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
             "properties": {k: v for k, v in d.items() if k not in ("lon", "lat")}} for d in dets]
