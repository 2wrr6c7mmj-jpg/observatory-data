"""occ-2: presence / absence of a ship at a quay section as a statistical test against the scene's own empty-quay distribution.

A second series beside the fixed-dB quay count (sat/quay.py quay-v2 / est-2) and after the occ-1 trial (sat/occ_test.py).
Inputs: the quay sample geometry (quay.Geometry), the sections (s2_labels.sections), the bl-1 layers of the scene's orbit
(vv_p10, vh_p10, vv_p90) and the scene's VV / VH; no extra fetch (PU 0). Output: one row per scene x section
(sat/occ2_schema.FIELDS) and a per-scene record of the null distribution.

Method (設計書_occ-2 2026-11-08, 2部):

1. Position and width per scene (§1.2). d = scene dB (3x3 mean) - baseline p10 at the pixel. The empty quay is not at
   d = 0: wind and sea state lift a whole scene by 4-7 dB. So the pool = inner band (0-30 m) of the non-blind naval
   sections gives Delta_b = median(d) and sigma_b = median - 31.7th percentile of the lower half (ships are the upper tail),
   per band (VV, VH). sigma is never below the speckle width 4.34 / sqrt(ENL) (ENL of the 3x3 mean measured on open water,
   §1.1), Delta never below the open-water lift minus 1 dB (§8: a scene with over half the berths taken), and a face
   (toward / along / away) with enough sections gets a fixed offset of its own (§8, shrunk and clipped).
   z = (d_VV - Delta_VV) / sigma_VV + w (d_VH - Delta_VH) / sigma_VH.
2. Sample and window statistics (§1.3). s_i = min of z over the depth pixels 5-25 m off the edge (5-35 m for a face toward
   the radar): two or three distinct pixels must be raised, so a one-pixel bright edge (type F) does not count.
   T = the largest sum of five consecutive s_i (50 m, the shortest ship) over a window that covers at least 20 m of the
   section (a ship across a section border is taken whole). The null of T is the pool sections' own T: the lower half
   folded about the median, empirical within its range and a normal tail beyond. p_test = P(T >= T_obs | empty).
   The seed is a requirement: at least one sample of the best window has s_i above the null's 1 - seed_p quantile
   (z_seed); without it p_test is floored at no_seed_floor_p (z_seed_ok 0 in the row).
3. Blind sections (§2.3). A section whose inner p10 median is above the p90 of the same-face, same-kind non-blind sections
   (or above the fixed blind_p10_db) is blind. It is tested only against its own empty days (the section reference below,
   occ2-2b); without them p_test is empty and p_occ = blind_prior (ref_kind prior). The design's spatial p90 reference is
   kept as an option (blind_test "spatial"): 1部's review of #87 showed it tests every blind berth "present" (the blind
   definition and the reference are the same quantity).
3b. Section reference (occ2-2b, 設計 §2.1 third item). The owner's "absent" days of a section (within 48 h of the scene)
   give its own empty T's; the median of their deviation from each scene's null centre (the scene itself left out) is the
   section's offset, subtracted from T before the test (ref_kind section). Needs section_ref_min_days such days.
4. The edge (§3.1). The baseline p10 across the OSM line (-40 .. +30 m, median over the section's samples) has a ridge
   (a local maximum 3 dB above the value 20 m waterward) where the quay's double-bounce line is: the band starts 10 m
   waterward of it (edge_source ridge, edge_offset_m = the ridge's offset from the line). Otherwise the land mask's edge
   (Geometry, edge_source mask). The band may enter the land mask 2.5 pixels (Geometry: 1.5); Geometry itself is not
   changed, so quay-v2 / est-2 and Zhanjiang are untouched.
5. Length (§3.2). Samples with s_i above the null's 1 - on_p quantile, gaps under 20 m joined, make stretches along the
   whole profile; a section reports the full length of the stretch covering at least 20 m of it (stretch_len_m, the
   quantity of est-2's ship_length_m) and its depth (the reach of z above the same level off the edge).
6. p_occ (§1.3-5): logit p_occ = a + b (-log10 p_test) + c ln(len / 50), a = logit(prior), b = 1, c = 0 before the
   calibration (occ2-4); sat/occ2_schema.p_occ_from_test.

Only numpy (sat/occ_test.enl_estimate for the ENL). Nothing here writes files: the daily flow (occ2-3) and the
evaluation (sat/eval_occ2.py) call ``run_scene``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from . import occ_test, quay
from .baseline import smooth_db
from .grid import dilate
from .occ2_schema import DEFAULTS, FIELDS, OCC_VERSION, p_occ_from_test
from .osm_mask import LAND, WATER



# ---------------------------------------------------------------- normal distribution (scalar helpers)
def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_isf(p: float) -> float:
    """x with P(Z >= x) = p (bisection on erfc; p in (0, 1))."""
    p = min(max(p, 1e-15), 1 - 1e-15)
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_sf(mid) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def robust_centre_scale(x: np.ndarray, floor: float = 0.0) -> tuple[float, float, int]:
    """(median, one sigma from the lower half: median - its 31.7th percentile, n) of finite values; the upper tail is left
    out (ships). Scale at least ``floor``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n == 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(x))
    lo = x[x <= med]
    scale = med - float(np.percentile(lo, 31.7)) if len(lo) >= 10 else float(np.std(x))
    return med, max(scale, floor), n


def folded_tail_p(null: np.ndarray, t: float, centre: float, scale: float) -> float:
    """P(T >= t) under the null made of the lower half of ``null`` folded about ``centre``: the empirical share within the
    folded values' range (with the +1 of the unseen), the normal tail N(centre, scale) beyond it."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(t):
        return float("nan")
    lo = null[null <= centre]
    folded = np.concatenate([lo, 2.0 * centre - lo])
    n = len(folded)
    if t <= folded.max():
        return float((np.sum(folded >= t) + 1.0) / (n + 1.0))
    p_norm = norm_sf((t - centre) / max(scale, 1e-9))
    return float(min(p_norm, 1.0 / (n + 1.0)))


def sample_stat(z: np.ndarray, cols: np.ndarray, mode: str = "min2") -> np.ndarray:
    """s_i per sample from z (samples x depth pixels). min2: the best adjacent pair of depth pixels within ``cols``, both
    raised (max over the pairs of the min of the two); w15: 0.5 z(first) + 0.5 z(second) (§8 variant). NaN when no pair
    has two values."""
    cols = np.asarray(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "w15":
            zz = z[:, cols[:2]]
            return np.where(np.isfinite(zz).all(1), 0.5 * zz[:, 0] + 0.5 * zz[:, 1], np.nan)
        best = np.full(z.shape[0], np.nan)
        for k0, k1 in zip(cols[:-1], cols[1:]):
            pair = np.minimum(z[:, k0], z[:, k1])        # NaN when either is NaN
            best = np.where(np.isfinite(pair) & (~np.isfinite(best) | (pair > best)), pair, best)
        return best


# ---------------------------------------------------------------- band pixels off any edge
class Sampler:
    """Band pixels of a profile's samples at offsets from the OSM line (water side positive), with the validity rule of
    quay.Geometry (inside the grid, not deep in the land mask, owned by the line) but a wider land allowance."""

    def __init__(self, geom: quay.Geometry, mask: np.ndarray, lines: dict | None = None, cfg: dict | None = None):
        self.geom = geom
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.h, self.w = mask.shape
        notland = mask != LAND
        self.allowed = notland | dilate(notland, float(self.cfg["allowed_dilate_px"]))
        self.way_k = {w.get("id"): k for k, w in enumerate(lines.get("ways", []))} if lines else None
        self._base: dict = {}

    def base(self, pi: int):
        """(x, y, nx, ny) of a profile's samples: the point on the OSM line and the water-side unit normal (x east, y south, m)."""
        if pi not in self._base:
            pr = self.geom.profiles[pi]
            g = self.geom.grid
            r, c = g.to_pixel(np.asarray(pr["lon"], float), np.asarray(pr["lat"], float))
            x, y = np.asarray(c, float) * self.geom.rx, np.asarray(r, float) * self.geom.ry
            naz = np.radians(np.asarray(pr["naz"], float))
            nx, ny = np.sin(naz), -np.cos(naz)
            e = np.asarray(pr["edge_m"], float) + 15.0     # the display line is 15 m off the mask's edge
            self._base[pi] = (x - e * nx, y - e * ny, nx, ny)
        return self._base[pi]

    def pixels(self, pi: int, idx: np.ndarray, line_off: np.ndarray, raw: bool = False):
        """(rr, cc, valid) for samples ``idx`` of profile ``pi`` at ``line_off`` (len(idx), m) metres from the OSM line.
        ``raw``: valid = inside the grid only (land read too, for the ridge profile)."""
        x, y, nx, ny = self.base(pi)
        idx = np.asarray(idx)
        qx = x[idx][:, None] + line_off * nx[idx][:, None]
        qy = y[idx][:, None] + line_off * ny[idx][:, None]
        rr = np.floor(qy / self.geom.ry).astype(int)
        cc = np.floor(qx / self.geom.rx).astype(int)
        inside = (rr >= 0) & (rr < self.h) & (cc >= 0) & (cc < self.w)
        rr, cc = np.clip(rr, 0, self.h - 1), np.clip(cc, 0, self.w - 1)
        if raw:
            return rr, cc, inside
        valid = inside & self.allowed[rr, cc]
        pr = self.geom.profiles[pi]
        if self.way_k is not None and pr.get("way_id") in self.way_k and len(self.geom.seg_a):
            k = self.way_k[pr["way_id"]]
            own, other = quay._own_other(np.stack([qx.ravel(), qy.ravel()], 1), self.geom.seg_a, self.geom.seg_b, self.geom.seg_way, k)
            valid &= (own <= other + self.geom.cfg["own_tolerance_m"]).reshape(qx.shape)
        return rr, cc, valid


def section_face(geom: quay.Geometry, sec: dict, look_az_deg: float | None) -> str:
    if look_az_deg is None:
        return ""
    pr = geom.profiles[sec["profile"]]
    return quay.face_class(float(np.median(quay.cos_to_radar(pr["naz"][np.asarray(sec["idx"])], look_az_deg))))


# ---------------------------------------------------------------- the edge (§3.1)
def ridge_offset(sampler: Sampler, sec: dict, vv_p10: np.ndarray, cfg: dict) -> float | None:
    """The offset (m from the OSM line, water side positive) of the baseline p10 ridge across the line, or None."""
    lo, hi = cfg["ridge_search_m"]
    step = float(cfg["ridge_step_m"])
    offs = np.arange(lo, hi + 0.1, step)
    idx = np.asarray(sec["idx"])
    rr, cc, inside = sampler.pixels(sec["profile"], idx, np.broadcast_to(offs[None, :], (len(idx), len(offs))).copy(), raw=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.nanmedian(np.where(inside, vv_p10[rr, cc], np.nan), 0)
    k_check = int(round(cfg["ridge_check_m"] / step))
    best = None
    for i in range(1, len(offs) - k_check):
        if not (np.isfinite(prof[i]) and np.isfinite(prof[i - 1]) and np.isfinite(prof[i + 1]) and np.isfinite(prof[i + k_check])):
            continue
        if not (offs[i] >= cfg["ridge_min_m"] and offs[i] <= cfg["ridge_max_m"]):
            continue
        # a local maximum; on a plateau (two 5 m offsets in one 10 m pixel) its waterward end, the wall
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] - prof[i + k_check] >= cfg["ridge_rel_db"]:
            if best is None or prof[i] > prof[best]:
                best = i
    if best is None:
        return None
    r = float(offs[best])
    # the band after the ridge must have pixels to read (not deep land)
    e = r + cfg["ridge_after_m"]
    _, _, v = sampler.pixels(sec["profile"], idx, np.broadcast_to(np.array([[e + 5.0, e + 15.0]]), (len(idx), 2)).copy())
    if v.mean() < 0.5:
        return None
    return r


# ---------------------------------------------------------------- the scene
def scene_water_stats(svv: np.ndarray, svh: np.ndarray, layers: dict, mask: np.ndarray, cfg: dict) -> dict:
    """ENL of the 3x3 mean (VV, VH) and the lift of open water (median d) of the scene."""
    water = mask == WATER
    far = water & ~dilate(~water, int(cfg["enl_far_px"]))
    out = {"enl_vv": occ_test.enl_estimate(svv, far), "enl_vh": occ_test.enl_estimate(svh, far), "water_px": int(far.sum())}
    out["enl_source"] = "scene" if out["enl_vv"] is not None else "default"
    if out["enl_vv"] is None:
        out["enl_vv"] = float(cfg["enl_default_vv"])
    if out["enl_vh"] is None:
        out["enl_vh"] = float(cfg["enl_default_vh"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dv = (svv - layers["vv_p10"])[far]
        dh = (svh - layers["vh_p10"])[far]
        out["water_lift_vv_db"] = float(np.nanmedian(dv)) if np.isfinite(dv).sum() >= 50 else None
        out["water_lift_vh_db"] = float(np.nanmedian(dh)) if np.isfinite(dh).sum() >= 50 else None
        out["water_vv_median_db"] = float(np.nanmedian(svv[far])) if far.any() else None
    return out


def _group_ref(groups: dict, face: str, kind: str, pct: float, min_n: int):
    """The reference of a (face, kind) group: its p90 of the sections' inner p10 medians, borrowing the face's groups then
    all when the group is small. (vv, vh, group name, n) or None."""
    for key, name in (((face, kind), f"{face}/{kind}"), ((face, None), face), ((None, None), "all")):
        vals = groups.get(key, [])
        if len(vals) >= min_n:
            a = np.array(vals)
            return float(np.percentile(a[:, 0], pct)), float(np.percentile(a[:, 1], pct)), name, len(vals)
    return None


def run_scene(geom: quay.Geometry, mask: np.ndarray, sections: list[dict], layers: dict, vv_db: np.ndarray, vh_db: np.ndarray,
              acq: dict, lines: dict | None = None, cfg: dict | None = None, eligible=None, blind_prior: dict | None = None,
              look_az_deg: float | None = None, presmoothed: bool = False, calib: dict | None = None,
              baseline_period: str = "") -> tuple[list[dict], dict]:
    """The occ-2 rows of one scene and the scene's record (pool statistics).

    ``sections``: s2_labels.sections(geom); ``eligible``: f(section) -> bool (the naval berth sections; None: every
    section of min_section_m); ``blind_prior``: {section_id: P(present)} of the blind sections (default
    blind_prior_default); ``acq``: site_id, acquired_utc, relative_orbit, orbit_direction; ``calib``: the calibration
    (occ2_schema DEFAULTS["calib"] keys). ``presmoothed``: vv_db / vh_db are 3x3 means already. ``baseline_period``: the
    scene's place in the baseline (sat/run.in_sample: "tuning" / "baseline" / "", recorded in the row)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    step = geom.cfg["step_m"]
    sampler = Sampler(geom, mask, lines, cfg)
    svv = vv_db if presmoothed else smooth_db(vv_db)
    svh = vh_db if presmoothed else smooth_db(vh_db)
    p10v, p10h = layers["vv_p10"], layers["vh_p10"]
    if look_az_deg is None:
        lat0 = float(geom.grid.to_lonlat(geom.grid.height / 2.0, geom.grid.width / 2.0)[1])
        look_az_deg = quay.look_azimuth(acq.get("orbit_direction"), lat0)
    offs = np.arange(5.0, geom.cfg["depth_max_m"] + 0.1, 10.0)   # depth pixels 5, 15, 25 .. off the edge: distinct 10 m pixels
    inner = offs <= cfg["pool_inner_m"]
    water = scene_water_stats(svv, svh, layers, mask, cfg)
    sig_floor_vv = max(4.343 / math.sqrt(max(water["enl_vv"], 1.0)), cfg["sigma_db_min"])
    sig_floor_vh = max(4.343 / math.sqrt(max(water["enl_vh"], 1.0)), cfg["sigma_db_min"])
    blind_prior = blind_prior or {}

    # the sections of the test and their baseline facts
    secs = []
    for s in sections:
        if s["length_m"] < cfg["min_section_m"]:
            continue
        if eligible is not None and not eligible(s):
            continue
        pr = geom.profiles[s["profile"]]
        idx = np.asarray(s["idx"])
        edge = np.asarray(pr["edge_m"], float)[idx]
        rr, cc, valid = sampler.pixels(s["profile"], idx, edge[:, None] + offs[None, :])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a10 = np.where(valid, p10v[rr, cc], np.nan)[:, inner]
            h10 = np.where(valid, p10h[rr, cc], np.nan)[:, inner]
            d = {"sec": s, "idx": idx, "edge": edge, "face": section_face(geom, s, look_az_deg),
                 "p10_med_vv": float(np.nanmedian(a10)) if np.isfinite(a10).any() else float("nan"),
                 "p10_med_vh": float(np.nanmedian(h10)) if np.isfinite(h10).any() else float("nan"),
                 "blind_share": float(np.mean(np.nanmax(a10, 1) > cfg["blind_p10_db"])) if np.isfinite(a10).any() else 1.0}
        secs.append(d)
    if not secs:
        return [], {"pool_n": 0, **water}

    # blind (§2.3): the fixed rule first, then the spatial reference from the sections it leaves clear
    groups: dict = {}
    for d in secs:
        if d["blind_share"] < 0.5 and np.isfinite(d["p10_med_vv"]) and np.isfinite(d["p10_med_vh"]):
            v = (d["p10_med_vv"], d["p10_med_vh"])
            for key in ((d["face"], d["sec"]["kind"]), (d["face"], None), (None, None)):
                groups.setdefault(key, []).append(v)
    for d in secs:
        ref = _group_ref(groups, d["face"], d["sec"]["kind"], cfg["blind_ref_pct"], cfg["blind_ref_min"])
        d["ref"] = ref
        d["blind"] = d["blind_share"] >= 0.5 or (ref is not None and np.isfinite(d["p10_med_vv"]) and d["p10_med_vv"] > ref[0])

    # the edge (§3.1) and the band values (d = scene - reference) at the depth pixels
    for d in secs:
        d["edge_offset_m"], d["edge_source"] = None, "mask"
        edge = d["edge"]
        if not d["blind"]:
            r = ridge_offset(sampler, d["sec"], p10v, cfg)
            if r is not None:
                d["edge_offset_m"] = r
                if float(np.median(edge)) - (r + cfg["ridge_after_m"]) >= cfg["ridge_shift_min_m"]:
                    d["edge_source"] = "ridge"
                    edge = np.full(len(d["idx"]), r + cfg["ridge_after_m"])
        d["band_edge"] = edge
        rr, cc, valid = sampler.pixels(d["sec"]["profile"], d["idx"], edge[:, None] + offs[None, :])
        xv = np.where(valid, svv[rr, cc], np.nan)
        xh = np.where(valid, svh[rr, cc], np.nan)
        if d["blind"] and d["ref"] is not None and cfg["blind_test"] == "spatial":
            d["dvv"], d["dvh"], d["ref_kind"] = xv - d["ref"][0], xh - d["ref"][1], "spatial"
        else:
            d["dvv"], d["dvh"], d["ref_kind"] = xv - p10v[rr, cc], xh - p10h[rr, cc], "time"
        d["dvv_time"] = xv - p10v[rr, cc]   # the pool reads the time reference whatever the section's own is
        d["dvh_time"] = xh - p10h[rr, cc]

    # the pool (§1.2): position and width of the empty quay in this scene
    pool = [d for d in secs if not d["blind"]]
    pv = np.concatenate([d["dvv_time"][:, inner].ravel() for d in pool]) if pool else np.empty(0)
    ph = np.concatenate([d["dvh_time"][:, inner].ravel() for d in pool]) if pool else np.empty(0)
    dvv0, svv0, nvv = robust_centre_scale(pv, sig_floor_vv)
    dvh0, svh0, nvh = robust_centre_scale(ph, sig_floor_vh)
    rec = {"site_id": acq.get("site_id"), "acquired_utc": acq.get("acquired_utc"), "relative_orbit": acq.get("relative_orbit"),
           "occ_version": OCC_VERSION, "sections": len(secs), "pool_n": len(pool), "pool_px": int(min(nvv, nvh)),
           "blind_sections": sum(1 for d in secs if d["blind"]), "ridge_sections": sum(1 for d in secs if d["edge_source"] == "ridge"),
           **water, "sigma_floor_vv_db": sig_floor_vv, "sigma_floor_vh_db": sig_floor_vh}
    if min(nvv, nvh) < cfg["pool_min_samples"]:
        rec.update(status="no_pool")
        rows = [_row(acq, d, None, None, None, None, 0, None, None, rec, cfg, calib, baseline_period, blind_prior) for d in secs]
        return rows, rec
    rec["delta_raw_vv_db"], rec["delta_raw_vh_db"] = dvv0, dvh0
    if water.get("water_lift_vv_db") is not None:       # §8: the water's lift is a floor of Delta
        dvv0 = max(dvv0, water["water_lift_vv_db"] - cfg["delta_water_margin_db"])
    if water.get("water_lift_vh_db") is not None:
        dvh0 = max(dvh0, water["water_lift_vh_db"] - cfg["delta_water_margin_db"])
    rec.update(delta_vv_db=dvv0, delta_vh_db=dvh0, sigma_vv_db=svv0, sigma_vh_db=svh0)
    # §8: a fixed offset per face, shrunk by n / (n + k) and clipped
    face_corr: dict = {}
    for face in ("toward", "along", "away"):
        fs = [d for d in pool if d["face"] == face]
        if len(fs) < cfg["face_corr_min_sections"]:
            continue
        fv = np.concatenate([d["dvv_time"][:, inner].ravel() for d in fs])
        fh = np.concatenate([d["dvh_time"][:, inner].ravel() for d in fs])
        mv, _, _ = robust_centre_scale(fv)
        mh, _, _ = robust_centre_scale(fh)
        shrink = len(fs) / (len(fs) + cfg["face_corr_k"])
        cmax = cfg["face_corr_max_db"]
        face_corr[face] = (float(np.clip((mv - rec["delta_raw_vv_db"]) * shrink, -cmax, cmax)),
                           float(np.clip((mh - rec["delta_raw_vh_db"]) * shrink, -cmax, cmax)), len(fs))
    rec["face_corr_db"] = {f: {"vv": round(v[0], 2), "vh": round(v[1], 2), "n": v[2]} for f, v in face_corr.items()}

    # z, s_i per sample on the whole profile, T per section (§1.3)
    w = max(1, int(round(cfg["window_m"] / step)))
    ov = max(1, int(round(cfg["overlap_m"] / step)))
    s_prof: dict = {}
    for d in secs:
        cv, ch = face_corr.get(d["face"], (0.0, 0.0, 0))[:2]
        with np.errstate(invalid="ignore"):
            z = (d["dvv"] - dvv0 - cv) / svv0 + cfg["w_vh"] * (d["dvh"] - dvh0 - ch) / svh0
        d["z"] = z
        lo, hi = cfg["depth_toward_m"] if d["face"] == "toward" else cfg["depth_m"]
        use = np.flatnonzero((offs >= lo - 0.1) & (offs <= hi + 0.1))
        d["s"] = sample_stat(z, use, cfg["depth_mode"])
        n = len(geom.profiles[d["sec"]["profile"]]["ok"])
        arr = s_prof.setdefault(d["sec"]["profile"], np.full(n, np.nan))
        arr[d["idx"]] = d["s"]
    # the null of s from the pool (§1.3-3); samples without a value (band pixels invalid) count as the null's centre
    s_pool = np.concatenate([d["s"] for d in pool])
    ms, ss, ns = robust_centre_scale(s_pool, 0.05)
    sums: dict = {}
    for pi, arr in s_prof.items():
        fin = np.isfinite(arr)
        a0 = np.where(fin, arr, ms if np.isfinite(ms) else 0.0)
        ww = min(w, len(a0))
        sums[pi] = (np.convolve(a0, np.ones(ww), "valid"), np.convolve(fin.astype(float), np.ones(ww), "valid"), ww)
    min_fin = max(1, int(math.ceil(0.6 * w)))
    for d in secs:
        su, nf, ww = sums[d["sec"]["profile"]]
        a, b = int(d["idx"][0]), int(d["idx"][-1]) + 1
        starts = np.arange(len(su))
        overlap = np.minimum(starts + ww, b) - np.maximum(starts, a)
        ok = (overlap >= min(ov, b - a)) & (nf >= min(min_fin, ww))
        if not ok.any():
            d["T"], d["win"] = float("nan"), None
            continue
        j = int(starts[ok][np.argmax(su[ok])])
        d["T"], d["win"] = float(su[j]), (j, j + ww)
    # the null of T from the pool sections, the seed and the "on" level (§1.3-4, §3.2)
    t_pool = np.array([d["T"] for d in pool if np.isfinite(d["T"])])
    if len(t_pool) >= cfg["pool_min_sections"]:
        mt, st, _ = robust_centre_scale(t_pool, 0.05)
        t_null_src = "pool_sections"
    else:   # §1.3-3 normal approximation from the samples: T ~ N(w m_s, w v_s)
        mt, st, t_null_src = w * ms, math.sqrt(w) * ss, "samples_normal"
        t_pool = np.empty(0)
    z_seed = ms + norm_isf(cfg["seed_p"]) * ss
    z_on = ms + norm_isf(cfg["on_p"]) * ss
    rec.update(s_centre=ms, s_sigma=ss, t_centre=mt, t_sigma=st, t_null=t_null_src, z_seed=z_seed, z_on=z_on,
               seed_db_vv=dvv0 + z_seed * svv0, status="ok", t_pool=[round(float(t), 3) for t in t_pool])

    # stretches along each profile (§3.2)
    on_prof: dict = {}
    for pi, arr in s_prof.items():
        on = np.isfinite(arr) & (arr >= z_on)
        on_prof[pi] = (on, quay.runs(on, step, cfg["gap_m"], cfg["stretch_min_m"], False))
    rows = []
    for d in secs:
        on, runs_ = on_prof[d["sec"]["profile"]]
        a, b = int(d["idx"][0]), int(d["idx"][-1]) + 1
        best_len, depth = 0.0, None
        for s0, e0 in runs_:
            if min(e0, b) - max(s0, a) >= min(ov, b - a):
                length = (e0 - s0) * step
                if length > best_len:
                    best_len = length
                    # depth: how far z stays above z_on off the edge, median over the stretch's samples in this section
                    sel = np.arange(max(s0, a), min(e0, b)) - a
                    reach = []
                    for row in d["z"][sel]:
                        k = 0
                        while k < len(row) and np.isfinite(row[k]) and row[k] >= z_on:
                            k += 1
                        reach.append(k * 10.0)
                    depth = float(np.median(reach)) if reach else None
        seed_ok = None
        if d["win"] is not None:
            j0, j1 = d["win"]
            seg = s_prof[d["sec"]["profile"]][j0:j1]
            seed_ok = int(bool(np.any(np.isfinite(seg) & (seg >= z_seed))))
        p_test = folded_tail_p(t_pool, d["T"], mt, st) if t_null_src == "pool_sections" and np.isfinite(d["T"]) else \
            (norm_sf((d["T"] - mt) / st) if np.isfinite(d["T"]) else None)
        if p_test is not None and np.isnan(p_test):
            p_test = None
        rows.append(_row(acq, d, p_test, d["T"] if np.isfinite(d["T"]) else None, seed_ok, best_len if best_len else 0.0,
                         1 if d["blind"] else 0, depth, (dvv0, dvh0, svv0, svh0), rec, cfg, calib, baseline_period, blind_prior))
    return rows, rec


def finalize(row: dict, p_raw, cfg: dict, calib: dict | None, blind_prior: dict) -> dict:
    """p_test and p_occ of a row from the raw test p: the seed requirement (floor without a seed), a blind section without a
    section reference gets no test (ref_kind prior, p_occ = blind_prior), otherwise the placeholder calibration and, for a
    blind section, the mixture with its prior."""
    seed_ok = row.get("z_seed_ok")
    blind = bool(row.get("blind"))
    prior = blind_prior.get(row["section_id"], cfg["blind_prior_default"]) if blind else None
    row["blind_prior"] = prior
    if p_raw is not None and not np.isfinite(p_raw):
        p_raw = None
    if blind and row.get("ref_kind") == "time" and cfg["blind_test"] != "spatial":
        row["ref_kind"], row["p_test"], row["p_occ"] = "prior", None, prior
        return row
    p = p_raw
    if p is not None and seed_ok == 0:
        p = max(p, cfg["no_seed_floor_p"])
    row["p_test"] = p
    row["p_occ"] = p_occ_from_test(p, row.get("stretch_len_m"), calib, blind=blind, blind_prior=prior, lam=cfg["lambda_blind"])
    return row


def _row(acq, d, p_test, t_stat, seed_ok, stretch_len, blind, depth, null, rec, cfg, calib, baseline_period, blind_prior) -> dict:
    sec = d["sec"]
    c = {**DEFAULTS["calib"], **(calib or {})}
    row = {k: None for k in FIELDS}
    row.update({
        "site_id": acq.get("site_id"), "acquired_utc": acq.get("acquired_utc"), "relative_orbit": acq.get("relative_orbit"),
        "orbit_direction": acq.get("orbit_direction"), "section_id": sec["section_id"], "kind": sec["kind"], "face": d["face"],
        "occ_version": OCC_VERSION, "p_test": None, "p_occ": None, "t_stat": t_stat, "z_seed_ok": seed_ok,
        "stretch_len_m": stretch_len, "stretch_depth_m": depth, "blind": blind, "blind_prior": None,
        "ref_kind": d.get("ref_kind"), "edge_offset_m": d.get("edge_offset_m"), "edge_source": d.get("edge_source"),
        "delta_vv_db": null[0] if null else None, "delta_vh_db": null[1] if null else None,
        "sigma_vv_db": null[2] if null else None, "sigma_vh_db": null[3] if null else None,
        "enl_vv": rec.get("enl_vv"), "enl_vh": rec.get("enl_vh"), "pool_n": rec.get("pool_n"),
        "calib_source": c.get("source", "none"), "baseline_period": baseline_period,
    })
    return finalize(row, p_test, cfg, calib, blind_prior)


# ---------------------------------------------------------------- the section reference (occ2-2b)
def section_offsets(rows_by_scene: dict, recs_by_scene: dict, absent: dict, min_days: int = DEFAULTS["section_ref_min_days"]) -> dict:
    """{(acquired_utc, section_id): offset} — the median, over the section's empty scenes other than this one (LOO), of
    T - the scene's null centre. ``absent``: {section_id: set of acquired_utc the owner called absent}. Only scenes with a
    record (status ok) count; a section needs ``min_days`` such scenes."""
    dev: dict = {}
    for acq, rows in rows_by_scene.items():
        rec = recs_by_scene.get(acq) or {}
        if rec.get("status") != "ok" or rec.get("t_centre") is None:
            continue
        for r in rows:
            if r.get("t_stat") is not None and acq in absent.get(r["section_id"], ()):
                dev.setdefault(r["section_id"], {})[acq] = float(r["t_stat"]) - float(rec["t_centre"])
    out = {}
    for sid, by in dev.items():
        for acq in rows_by_scene:
            others = [v for a, v in by.items() if a != acq]
            if len(others) >= min_days:
                out[(acq, sid)] = float(np.median(others))
    return out


def apply_section_reference(rows_by_scene: dict, recs_by_scene: dict, offsets: dict, cfg: dict | None = None,
                            calib: dict | None = None, blind_prior: dict | None = None, source_tag: str = "ref-own-absent-days") -> int:
    """Re-test the rows that have an offset: T - offset against the scene's pool null (ref_kind section). Returns the count."""
    cfg = {**DEFAULTS, **(cfg or {})}
    blind_prior = blind_prior or {}
    n = 0
    for acq, rows in rows_by_scene.items():
        rec = recs_by_scene.get(acq) or {}
        if rec.get("status") != "ok":
            continue
        t_pool = np.asarray(rec.get("t_pool") or [], float)
        mt, st = float(rec["t_centre"]), float(rec["t_sigma"])
        for r in rows:
            off = offsets.get((acq, r["section_id"]))
            if off is None or r.get("t_stat") is None or r.get("ref_kind") == "spatial":
                continue
            t_adj = float(r["t_stat"]) - off
            p = folded_tail_p(t_pool, t_adj, mt, st) if len(t_pool) >= cfg["pool_min_sections"] else norm_sf((t_adj - mt) / st)
            r["ref_kind"] = "section"
            r["t_stat"] = t_adj
            finalize(r, p, cfg, calib, blind_prior)
            r["ref_kind"] = "section"
            base = r.get("calib_source") or "none"
            r["calib_source"] = base if source_tag in base else (source_tag if base == "none" else f"{base};{source_tag}")
            n += 1
    return n


def eligible_naval(bt, min_m: float = DEFAULTS["min_section_m"]):
    """f(section) -> bool: on a naval-1 line, a berth (not berth false), at least ``min_m`` long (as sat/eval_presence)."""
    def f(s):
        try:
            return bool(s["length_m"] >= min_m and bt.stretch_class(s["way_id"], s["line"]) == "naval"
                        and bt.stretch_ok(s["kind"], s["way_id"], s["line"]))
        except Exception:
            return False
    return f


def settings(cfg: dict | None = None) -> dict:
    from .occ2_schema import settings as _s
    return _s(cfg)
