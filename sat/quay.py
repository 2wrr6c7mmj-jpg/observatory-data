"""Ships alongside quays and piers: occupied lengths from the change against the baseline (quay-v1, trial).

The CFAR count (sat/detect.py) misses ships lying against a quay: their echo merges with
the quay's. Here the scene is compared with the per-orbit "empty quay" baseline
(sat/baseline.py, VV/VH 10th percentile) along the OSM quay lines (sat/quay_lines.py):

1. Every line (pier, quay, coastline) is sampled every 10 m. At each sample, on each side
   that is water (by the land mask), two bands are read along the normal: inner 0-30 m and
   outer 30-60 m. A band pixel belongs to the line only if no other line is closer (a ship
   between two piers is counted once, by the nearer pier).
2. The profile along the line is the maximum change in the band: dVV = VV - VV_p10 and
   dVH = VH - VH_p10 (both 3x3 means, as in the baseline). A sample is occupied when
   dVV >= +6 dB and dVH >= +4 dB (and the scene's VV there is at least VV_MIN_DB).
3. Occupied samples form runs; gaps under 20 m are joined, runs of 30 m or more are kept
   (occupied stretches). Never mixes orbits: the baseline is the scene's own orbit.
4. The number is a range. Minimum = stretches, plus one for each valley of 3 dB or more
   in the dVV profile (two ships bow to stern). Maximum = sum of max(pieces, floor(length /
   60 m)) (the shortest warship assumed 60 m), plus one where the outer band is occupied
   alongside the inner band for 30 m or more (a ship moored outboard of another).
   Estimated length of a stretch = length - 20 m (the 3x3 mean widens it by about 20 m).

Trial: the accuracy has not been measured. Credibility 3 (4 when the baseline has fewer
than 24 scenes).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .baseline import smooth_db
from .grid import Grid, dilate
from .osm_mask import LAND, WATER

QUAY_VERSION = "quay-v1"
DEFAULTS = {
    "step_m": 10.0,
    "inner_m": (5.0, 15.0, 25.0),       # 0-30 m band (pixel centres every 10 m)
    "outer_m": (35.0, 45.0, 55.0),      # 30-60 m band
    "edge_search_m": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),  # where the water starts (a pier line is its centre)
    "raft_from_m": 40.0,                # outboard ship: outer band beyond the 3x3 spread (~10 m) of the inner ship
    "water_side_min": 0.6,              # share of water (land mask) at 15-55 m for a side to count
    "own_tolerance_m": 5.0,             # a band pixel belongs to the nearest line (within this)
    "dvv_db": 6.0,
    "dvh_db": 4.0,
    "vv_min_db": -10.0,                 # the scene's VV at the hit (guards against wind on calm-water baselines)
    "seed_dvv_db": 12.0,                # a stretch needs a sample with dVV at least this high (hysteresis)
    "min_run_m": 30.0,
    "max_gap_m": 20.0,                  # gaps shorter than this are joined
    "valley_db": 3.0,
    "min_piece_m": 30.0,                # a valley splits only pieces at least this long
    "ship_min_m": 60.0,                 # shortest warship assumed (maximum = floor(length / this))
    "raft_min_m": 30.0,
    "length_trim_m": 20.0,
    "cluster_m": 60.0,                  # stretches this close (other side of a pier, a corner) may be one ship
    "depth_step_m": 5.0,                # est-1: occupied depth off the water's edge, 5-120 m every 5 m
    "depth_max_m": 120.0,               # 90 m until 2026-10-30 (feat-1: three ships abreast reach ~100 m)
    "face_cos": 0.5,                    # feat-1: |cos phi| above this is a face toward / away from the radar
    "db_rel_db": 6.0,                   # feat-1: peaks of the cross-section within this of the strongest
}

# feat-1 (record only, never used in the count or the estimate): face orientation against the radar,
# depth without the mean across the line, the outer double-bounce line, orbit pairs.
FEATURES_VERSION = "feat-1"
S1_INCLINATION_DEG = 98.18     # Sentinel-1 orbit (sun-synchronous, 693 km)
S1_PERIOD_S = 98.6 * 60.0
EARTH_OMEGA = 7.2921159e-5     # rad/s


def look_azimuth(orbit_direction: str | None, lat_deg: float, right_looking: bool = True) -> float | None:
    """Azimuth (deg from north, clockwise) the radar looks along, on the ground, from the orbit:
    the ground-track heading (inclination and Earth rotation at ``lat_deg``) plus 90 deg for a
    right-looking sensor. Sentinel-1 ascending at 21.2N: heading ~-12 deg, look ~78 deg (ENE)."""
    od = (orbit_direction or "").lower()
    if od not in ("ascending", "descending"):
        return None
    i, p = math.radians(S1_INCLINATION_DEG), math.radians(lat_deg)
    s = max(-1.0, min(1.0, math.cos(i) / math.cos(p)))
    a = math.asin(s)                       # inertial azimuth of the ascending track (negative: west of north)
    if od == "descending":
        a = math.pi - a
    ws = 2 * math.pi / S1_PERIOD_S
    vn = ws * math.cos(a)
    ve = ws * math.sin(a) - EARTH_OMEGA * math.cos(p)
    heading = math.degrees(math.atan2(ve, vn))
    return round((heading + (90.0 if right_looking else -90.0)) % 360.0, 1)


def face_class(cos_phi: float, thr: float = DEFAULTS["face_cos"]) -> str:
    """"toward" (the water side faces the radar), "away" or "along"."""
    if not np.isfinite(cos_phi):
        return ""
    return "toward" if cos_phi > thr else "away" if cos_phi < -thr else "along"


def cos_to_radar(normal_az_deg, look_az_deg: float):
    """cos of the angle between the water-side normal and the direction to the radar
    (look azimuth + 180 deg): +1 when the face looks straight at the sensor."""
    return np.cos(np.radians(np.asarray(normal_az_deg, float) - (look_az_deg + 180.0)))


def _resample(x: np.ndarray, y: np.ndarray, step: float):
    """Points every ``step`` (metres) along a polyline, with unit tangents."""
    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate([[True], seg > 1e-6])
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(1, int(total // step))
    t = (np.arange(n) + 0.5) * total / n
    i = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(seg) - 1)
    f = (t - s[i]) / seg[i]
    px = x[i] + f * (x[i + 1] - x[i])
    py = y[i] + f * (y[i + 1] - y[i])
    tx = (x[i + 1] - x[i]) / seg[i]
    ty = (y[i + 1] - y[i]) / seg[i]
    return np.stack([px, py], 1), np.stack([tx, ty], 1)


def _seg_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Distance from each point to each segment a->b: (n_points, n_segments)."""
    out = []
    d = b - a
    dd = np.maximum((d * d).sum(1), 1e-12)
    for k in range(0, len(points), chunk):
        p = points[k:k + chunk, None, :]
        t = np.clip(((p - a[None]) * d[None]).sum(2) / dd[None], 0, 1)
        q = a[None] + t[..., None] * d[None]
        out.append(np.hypot(p[..., 0] - q[..., 0], p[..., 1] - q[..., 1]))
    return np.concatenate(out) if out else np.empty((0, len(a)))


class Geometry:
    """Sample points, sides and band pixels of a site's quay lines (fixed per site and mask)."""

    def __init__(self, grid: Grid, mask: np.ndarray, lines: dict, cfg: dict | None = None):
        self.cfg = cfg = {**DEFAULTS, **(cfg or {})}
        self.grid = grid
        ry, rx = grid.pixel_m
        self.ry, self.rx = ry, rx
        h, w = mask.shape
        water = mask == WATER
        # band pixels may be water, structure, or land right at the edge (OSM lines are off by 10-20 m)
        allowed = (mask != LAND) | dilate(mask != LAND, 1.5)
        ways = []
        seg_a, seg_b, seg_way = [], [], []
        for k, way in enumerate(lines.get("ways", [])):
            lon = np.array([c[0] for c in way["coords"]])
            lat = np.array([c[1] for c in way["coords"]])
            r, c = grid.to_pixel(lon, lat)
            x, y = c * rx, r * ry
            ways.append((way, x, y))
            seg_a.append(np.stack([x[:-1], y[:-1]], 1))
            seg_b.append(np.stack([x[1:], y[1:]], 1))
            seg_way.append(np.full(len(x) - 1, k))
        self.seg_a = np.concatenate(seg_a) if seg_a else np.empty((0, 2))
        self.seg_b = np.concatenate(seg_b) if seg_b else np.empty((0, 2))
        self.seg_way = np.concatenate(seg_way) if seg_way else np.empty(0, int)
        inner = np.array(cfg["inner_m"])
        outer = np.array(cfg["outer_m"])
        offs = np.concatenate([inner, outer])
        side_offs = offs[offs >= 15.0]
        self.depth_offs = np.arange(cfg["depth_step_m"], cfg["depth_max_m"] + 0.1, cfg["depth_step_m"])
        self.profiles = []  # one per (way, side) with at least one valid sample
        for k, (way, x, y) in enumerate(ways):
            pts, tan = _resample(x, y, cfg["step_m"])
            if not len(pts):
                continue
            nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
            for side in (1, -1):
                # the bands start at the water's edge: 0 m for a coastline drawn on the edge,
                # half the deck width for a pier drawn along its centre
                es = np.array(cfg["edge_search_m"])
                qe = pts[:, None, :] + side * es[None, :, None] * nrm[:, None, :]
                re_ = np.clip(np.floor(qe[..., 1] / ry).astype(int), 0, h - 1)
                ce_ = np.clip(np.floor(qe[..., 0] / rx).astype(int), 0, w - 1)
                we = water[re_, ce_]
                first = np.where(we.any(1), we.argmax(1), 0)
                edge = es[first]
                q = pts[:, None, :] + side * (edge[:, None] + offs[None, :])[..., None] * nrm[:, None, :]  # (n, n_off, 2)
                rr = np.floor(q[..., 1] / ry).astype(int)
                cc = np.floor(q[..., 0] / rx).astype(int)
                inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr_c, cc_c = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
                is_water = water[rr_c, cc_c] & inside
                sel = np.isin(offs, side_offs)
                water_share = is_water[:, sel].mean(1)
                ok_side = water_share >= cfg["water_side_min"]
                if not ok_side.any():
                    continue
                # ownership: no other line closer than this one (plus a tolerance)
                flat = q.reshape(-1, 2)
                dist = _seg_dist(flat, self.seg_a, self.seg_b)
                own = dist[:, self.seg_way == k].min(1)
                others = dist[:, self.seg_way != k]
                other = others.min(1) if others.shape[1] else np.full(len(flat), np.inf)
                owned = (own <= other + cfg["own_tolerance_m"]).reshape(q.shape[:2])
                valid = inside & owned & allowed[rr_c, cc_c] & ok_side[:, None]
                if not valid.any():
                    continue
                # depth profile (est-1): 0-90 m off the water's edge every 5 m, same ownership rule
                dq = pts[:, None, :] + side * (edge[:, None] + self.depth_offs[None, :])[..., None] * nrm[:, None, :]
                drr = np.floor(dq[..., 1] / ry).astype(int)
                dcc = np.floor(dq[..., 0] / rx).astype(int)
                dinside = (drr >= 0) & (drr < h) & (dcc >= 0) & (dcc < w)
                drr, dcc = np.clip(drr, 0, h - 1), np.clip(dcc, 0, w - 1)
                ddist = _seg_dist(dq.reshape(-1, 2), self.seg_a, self.seg_b)
                down = ddist[:, self.seg_way == k].min(1)
                dothers = ddist[:, self.seg_way != k]
                dother = dothers.min(1) if dothers.shape[1] else np.full(len(down), np.inf)
                dvalid = dinside & (down <= dother + cfg["own_tolerance_m"]).reshape(dq.shape[:2]) & allowed[drr, dcc]
                # feat-1: the same depth pixels without the mean across the line: 3 pixels along it (-10, 0, +10 m)
                tq = dq[:, :, None, :] + np.array([-1.0, 0.0, 1.0])[None, None, :, None] * cfg["step_m"] * tan[:, None, None, :]
                trr = np.clip(np.floor(tq[..., 1] / ry).astype(int), 0, h - 1)
                tcc = np.clip(np.floor(tq[..., 0] / rx).astype(int), 0, w - 1)
                # water-side normal as an azimuth (x east, y south)
                naz = np.degrees(np.arctan2(side * nrm[:, 0], -side * nrm[:, 1])) % 360.0
                # display line: the middle of the inner band (15 m off the water's edge)
                mid = pts + side * (edge + 15.0)[:, None] * nrm
                lon, lat = grid.to_lonlat(mid[:, 1] / ry, mid[:, 0] / rx)
                self.profiles.append({
                    "way_id": way.get("id"), "kind": way["kind"], "closed": bool(way.get("closed")), "side": side,
                    "rr": rr_c, "cc": cc_c, "valid": valid,
                    "n_inner": len(inner), "ok": ok_side, "edge_m": edge,
                    "raft_cols": np.flatnonzero(offs >= cfg["raft_from_m"]),
                    "drr": drr, "dcc": dcc, "dvalid": dvalid, "trr": trr, "tcc": tcc, "naz": naz,
                    "lon": np.asarray(lon), "lat": np.asarray(lat),
                })

    @property
    def samples(self) -> int:
        return int(sum(p["ok"].sum() for p in self.profiles))


def runs(flags: np.ndarray, step: float, max_gap: float, min_run: float, closed: bool = False) -> list[tuple[int, int]]:
    """[start, end) index runs of True after joining gaps shorter than ``max_gap`` metres and
    dropping runs shorter than ``min_run``. A closed line wraps around."""
    f = np.asarray(flags, bool).copy()
    n = len(f)
    if n == 0 or not f.any():
        return []
    shift = 0
    if closed and not f.all():
        shift = int(np.flatnonzero(~f)[0])  # start at a free sample so no run wraps
        f = np.roll(f, -shift)
    max_gap_n = int(math.ceil(max_gap / step)) - 1  # gaps of at most this many samples are joined
    idx = np.flatnonzero(f)
    out = []
    s = e = idx[0]
    for i in idx[1:]:
        if i - e - 1 <= max_gap_n:
            e = i
        else:
            out.append((s, e + 1))
            s = e = i
    out.append((s, e + 1))
    out = [(a, b) for a, b in out if (b - a) * step >= min_run]
    if shift:
        out = [((a + shift) % n, (a + shift) % n + (b - a)) for a, b in out]
    return out


def _depth(pr: dict, dvv, dvh, svv, cfg: dict, offs: np.ndarray) -> np.ndarray:
    """Per sample: how far off the water's edge the occupation reaches without a break (m),
    NaN where the first depth pixel is not occupied."""
    if "drr" not in pr:
        return np.full(len(pr["ok"]), np.nan)
    rr, cc, v = pr["drr"], pr["dcc"], pr["dvalid"]
    with np.errstate(invalid="ignore"):
        hit = v & (dvv[rr, cc] >= cfg["dvv_db"]) & (dvh[rr, cc] >= cfg["dvh_db"]) & (svv[rr, cc] >= cfg["vv_min_db"])
    n_run = np.where(hit.all(1), hit.shape[1], np.argmin(hit, 1))  # leading occupied pixels
    out = np.where(n_run > 0, offs[np.maximum(n_run - 1, 0)] + cfg["depth_step_m"] / 2, np.nan)
    return out.astype(float)


def _lin_mean3(db: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Mean in linear power of the pixels on the last axis (NaN ignored), in dB."""
    v = db[rr, cc]
    ok = np.isfinite(v)
    lin = np.where(ok, 10.0 ** (np.where(ok, v, 0.0) / 10.0), 0.0)
    n = ok.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(lin.sum(-1) / n)
    return np.where(n > 0, out, np.nan)


def _depth_1x3(pr: dict, vv_db, vh_db, base: dict, cfg: dict, offs: np.ndarray):
    """feat-1: per sample, the occupied depth as ``_depth`` but with the scene's 1x3 mean (3 pixels along
    the line, none across it) against the baseline p10 (3x3), and the scene's 1x3 VV cross-section (dB)."""
    if "trr" not in pr or vv_db is None:
        n = len(pr["ok"])
        return np.full(n, np.nan), np.full((n, len(offs)), np.nan)
    rr, cc, v = pr["trr"], pr["tcc"], pr["dvalid"]
    vv3 = _lin_mean3(vv_db, rr, cc)
    vh3 = _lin_mean3(vh_db, rr, cc)
    with np.errstate(invalid="ignore"):
        hit = (v & (vv3 - base["vv_p10"][pr["drr"], pr["dcc"]] >= cfg["dvv_db"])
               & (vh3 - base["vh_p10"][pr["drr"], pr["dcc"]] >= cfg["dvh_db"]) & (vv3 >= cfg["vv_min_db"]))
    n_run = np.where(hit.all(1), hit.shape[1], np.argmin(hit, 1))
    depth = np.where(n_run > 0, offs[np.maximum(n_run - 1, 0)] + cfg["depth_step_m"] / 2, np.nan).astype(float)
    return depth, np.where(v, vv3, np.nan)


def outer_line(section_db: np.ndarray, offs: np.ndarray, reach_m: float | None, rel_db: float = 6.0) -> dict:
    """feat-1: the outer double-bounce line of a stretch from its mean cross-section (scene VV, dB per
    offset off the water's edge, mean over the samples in linear power). Local maxima within ``rel_db``
    of the strongest, out to ``reach_m`` (the occupied depth + 15 m; 60 m when unknown); the outermost
    one's lobe (down to -3 dB of it) gives the centroid (power-weighted distance from the water's edge).
    Returns {"m", "db", "peaks"} (None where there is no data)."""
    p_db = np.asarray(section_db, float)
    lim = (reach_m if reach_m is not None and np.isfinite(reach_m) else 45.0) + 15.0
    keep = (offs <= lim) & np.isfinite(p_db)
    if keep.sum() < 2:
        return {"m": None, "db": None, "peaks": 0}
    idx = np.flatnonzero(keep)
    d, p = offs[idx], 10.0 ** (p_db[idx] / 10.0)
    n = len(p)
    loc = [i for i in range(n) if (i == 0 or p[i] >= p[i - 1]) and (i == n - 1 or p[i] >= p[i + 1])]
    top = max(p[i] for i in loc)
    cand = [i for i in loc if p[i] >= top / 10 ** (rel_db / 10)]
    # neighbouring maxima of a flat top (the same pixel sampled twice every 5 m) count once
    peaks = [i for k, i in enumerate(cand) if k == 0 or i - cand[k - 1] > 2]
    o = cand[-1]
    a = b = o
    while a > 0 and p[a - 1] >= p[o] / 2 and p[a - 1] <= p[a]:
        a -= 1
    while b < n - 1 and p[b + 1] >= p[o] / 2 and p[b + 1] <= p[b]:
        b += 1
    w = p[a:b + 1]
    return {"m": int(round(float((d[a:b + 1] * w).sum() / w.sum()))), "db": round(float(10 * np.log10(p[o])), 1),
            "peaks": len(peaks)}


def pieces(profile: np.ndarray, valley_db: float, min_piece_n: int) -> int:
    """1 + number of valleys at least ``valley_db`` below the peaks on both sides,
    each piece at least ``min_piece_n`` samples long."""
    p = np.nan_to_num(np.asarray(profile, float), nan=-np.inf)
    n = len(p)
    count, start = 1, 0
    peak, valley, vi = p[0] if n else 0, None, None
    for i in range(1, n):
        x = p[i]
        if valley is None:
            if x > peak:
                peak = x
            elif x <= peak - valley_db:
                valley, vi = x, i
        else:
            if x < valley:
                valley, vi = x, i
            elif x >= valley + valley_db:
                if vi - start >= min_piece_n and n - vi >= min_piece_n:
                    count += 1
                    start = vi
                peak, valley, vi = x, None, None
    return count


def count(geom: Geometry, vv_db: np.ndarray, vh_db: np.ndarray, base: dict, sector_of=None,
          presmoothed: bool = False, look_az_deg: float | None = None) -> dict:
    """Occupied stretches of one scene against the baseline of its orbit.

    ``base``: {"vv_p10", "vh_p10"} in dB (NaN = no data). ``sector_of(lon, lat)`` -> sector id.
    ``look_az_deg``: the radar's look azimuth (``look_azimuth``) for the face features (feat-1; record only).
    Returns {"min", "max", "segments": [...], "rafted", "coverage", "by_sector": {id: [min, max]}}.
    """
    cfg = geom.cfg
    svv, svh = (vv_db, vh_db) if presmoothed else (smooth_db(vv_db), smooth_db(vh_db))
    dvv = svv - base["vv_p10"]
    dvh = svh - base["vh_p10"]
    step = cfg["step_m"]
    segs = []
    n_ok = n_data = 0
    for pr in geom.profiles:
        rr, cc, valid, ni = pr["rr"], pr["cc"], pr["valid"], pr["n_inner"]
        a_vv = np.where(valid, dvv[rr, cc], np.nan)
        a_vh = np.where(valid, dvh[rr, cc], np.nan)
        a_abs = np.where(valid, svv[rr, cc], np.nan)
        hit = (a_vv >= cfg["dvv_db"]) & (a_vh >= cfg["dvh_db"]) & (a_abs >= cfg["vv_min_db"])
        with np.errstate(invalid="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof = np.nanmax(a_vv[:, :ni], 1)
                prof_vh = np.nanmax(a_vh[:, :ni], 1)
        inner_hit = hit[:, :ni].any(1)
        depth = _depth(pr, dvv, dvh, svv, cfg, geom.depth_offs)
        depth13, section = _depth_1x3(pr, None if presmoothed else vv_db, vh_db, base, cfg, geom.depth_offs)
        cphi = cos_to_radar(pr["naz"], look_az_deg) if look_az_deg is not None and "naz" in pr else None
        # outboard ship: the outer band (beyond the spread of the inner ship) also reaches the seed level,
        # so that side lobes and the spread of a bright inner ship do not count as a second ship
        rc = pr["raft_cols"]
        outer_hit = (hit[:, rc] & (a_vv[:, rc] >= cfg["seed_dvv_db"])).any(1)
        has = np.isfinite(a_vv[:, :ni]).any(1) & pr["ok"]
        n_ok += int(pr["ok"].sum())
        n_data += int(has.sum())
        outer_runs = runs(outer_hit & inner_hit, step, cfg["max_gap_m"], cfg["raft_min_m"], pr["closed"])
        n = len(inner_hit)
        for a, b in runs(inner_hit, step, cfg["max_gap_m"], cfg["min_run_m"], pr["closed"]):
            idx = np.arange(a, b) % n
            length = (b - a) * step
            if not np.nanmax(prof[idx]) >= cfg["seed_dvv_db"]:
                continue
            k = pieces(prof[idx], cfg["valley_db"], int(math.ceil(cfg["min_piece_m"] / step)))
            mx = max(k, int(length // cfg["ship_min_m"]))
            covered = set(idx.tolist())
            overlaps = [len(covered & set((np.arange(oa, ob) % n).tolist())) * step for oa, ob in outer_runs]
            rafted = any(o >= cfg["raft_min_m"] for o in overlaps)
            outer_len = int(sum(o for o in overlaps if o >= cfg["raft_min_m"]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dmed = np.nanmedian(depth[idx]) if np.isfinite(depth[idx]).any() else np.nan
                d13 = np.nanmedian(depth13[idx]) if np.isfinite(depth13[idx]).any() else np.nan
                sec = section[idx]
                ok_sec = np.isfinite(sec)
                lin = np.where(ok_sec, 10.0 ** (np.where(ok_sec, sec, 0.0) / 10.0), 0.0).sum(0)
                nsec = ok_sec.sum(0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sec_db = np.where(nsec > 0, 10.0 * np.log10(lin / np.maximum(nsec, 1)), np.nan)
                pvh = np.nanmax(prof_vh[idx]) if np.isfinite(prof_vh[idx]).any() else np.nan
            mid = idx[len(idx) // 2]
            lon, lat = float(pr["lon"][mid]), float(pr["lat"][mid])
            ol = outer_line(sec_db, geom.depth_offs, d13 if np.isfinite(d13) else (dmed if np.isfinite(dmed) else None),
                            cfg["db_rel_db"])
            feat = {
                "depth_1x3_m": int(round(float(d13) / 5.0) * 5) if np.isfinite(d13) else None,
                "db_line_m": ol["m"], "db_line_db": ol["db"], "db_peaks": ol["peaks"],
                "look_az_deg": look_az_deg, "cos_phi": None, "face_look": None, "face_shares": None,
            }
            if cphi is not None:
                cs = cphi[idx]
                cls = [face_class(float(c), cfg["face_cos"]) for c in cs]
                shares = {k: round(cls.count(k) / len(cls), 2) for k in ("toward", "along", "away")}
                feat.update(cos_phi=round(float(np.median(cs)), 2), face_look=max(shares, key=lambda k: (shares[k], k == "along")),
                            face_shares=shares)
            segs.append({**feat,
                "way_id": pr["way_id"], "kind": pr["kind"], "side": pr["side"],
                "length_m": int(length), "est_length_m": int(max(length - cfg["length_trim_m"], step)),
                "min": k, "max": mx + (1 if rafted else 0), "rafted": rafted,
                "peak_dvv_db": round(float(np.nanmax(prof[idx])), 1),
                "peak_dvh_db": round(float(pvh), 1) if np.isfinite(pvh) else None,
                "outer_len_m": outer_len,
                "depth_m": int(round(float(dmed) / 5.0) * 5) if np.isfinite(dmed) else None,
                "lon": lon, "lat": lat,
                "sector": sector_of(lon, lat) if sector_of else None,
                "line": [[round(float(pr["lon"][i]), 6), round(float(pr["lat"][i]), 6)] for i in idx],
            })
    clusters = cluster(segs, cfg["cluster_m"])
    by_sector: dict = {}
    total_min = 0
    for ci, members in enumerate(clusters, 1):
        lead = max(members, key=lambda i: (segs[i]["min"], segs[i]["peak_dvv_db"]))
        cmin = segs[lead]["min"]
        total_min += cmin
        by_sector.setdefault(segs[lead]["sector"], [0, 0])[0] += cmin
        for i in members:
            segs[i]["cluster"] = ci
            segs[i]["cluster_lead"] = i == lead
            by_sector.setdefault(segs[i]["sector"], [0, 0])[1] += segs[i]["max"]
    return {
        "min": total_min, "max": sum(s["max"] for s in segs), "clusters": len(clusters),
        "segments": segs, "rafted": sum(s["rafted"] for s in segs),
        "coverage": round(n_data / n_ok, 3) if n_ok else 0.0,
        "by_sector": by_sector,
    }


BLIND_P10_DB = -8.0  # the baseline's VV p10 above this in the inner band: a ship there is unlikely to show


def blind_stretches(geom: Geometry, base: dict, p10_db: float = BLIND_P10_DB) -> dict:
    """Stretches of the quay lines where the count is likely blind: the baseline VV p10 in the
    inner band (0-30 m) is above ``p10_db`` (a ship was there in about 90 % of the baseline
    scenes, the berth was always taken by turns, or a fixed structure). A ship there adds
    little to the p10 and is likely not counted.

    ``blind_m``: the maximum over the inner band is above ``p10_db`` (the stretches; an upper
    figure: a bright pixel at the edge, e.g. the quay itself spreading in the 3x3 mean, is enough).
    ``strict_m``: even the darkest pixel of the inner band is above it (a lower figure).
    ``total_m`` is the length of all sides with water (as ``Geometry.samples``).
    Returns {"total_m", "blind_m", "share", "strict_m", "p10_db", "stretches": [{"length_m", "line", ...}]}."""
    cfg = geom.cfg
    step = cfg["step_m"]
    vv = base["vv_p10"]
    total = blind = strict = 0
    out = []
    import warnings
    for pr in geom.profiles:
        rr, cc, valid, ni = pr["rr"], pr["cc"], pr["valid"], pr["n_inner"]
        a = np.where(valid[:, :ni], vv[rr[:, :ni], cc[:, :ni]], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = np.nanmax(a, 1)
            lo = np.nanmin(a, 1)
        with np.errstate(invalid="ignore"):
            flags = pr["ok"] & np.isfinite(p) & (p > p10_db)
            strict += int((pr["ok"] & np.isfinite(lo) & (lo > p10_db)).sum()) * step
        total += int(pr["ok"].sum()) * step
        blind += int(flags.sum()) * step
        n = len(flags)
        for s, e in runs(flags, step, step, step, pr["closed"]):  # no joining, every run kept
            idx = np.arange(s, e) % n
            out.append({"way_id": pr["way_id"], "kind": pr["kind"], "side": pr["side"], "length_m": int((e - s) * step),
                        "line": [[round(float(pr["lon"][i]), 6), round(float(pr["lat"][i]), 6)] for i in idx]})
    return {"total_m": int(total), "blind_m": int(blind), "share": round(blind / total, 3) if total else 0.0,
            "strict_m": int(strict), "p10_db": p10_db, "stretches": out}


def cluster(segs: list[dict], dist_m: float) -> list[list[int]]:
    """Groups of stretches whose lines come within ``dist_m`` of each other (union-find).

    The same ship can light up both sides of a narrow pier (its bright echo and side lobes)
    or two lines at a corner. A group counts as at least its largest member (minimum) and
    at most all members (maximum)."""
    n = len(segs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if n:
        lat0 = math.radians(sum(s["lat"] for s in segs) / n)
        mlat, mlon = 111132.0, 111320.0 * math.cos(lat0)
        pts = [np.array([[p[0] * mlon, p[1] * mlat] for p in s["line"]]) for s in segs]
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(pts[i][:, None, 0] - pts[j][None, :, 0], pts[i][:, None, 1] - pts[j][None, :, 1])
                if d.min() <= dist_m:
                    parent[find(j)] = find(i)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def settings() -> dict:
    c = DEFAULTS
    return {
        "version": QUAY_VERSION,
        "lines": "OSM の man_made=pier・quay と natural=coastline（sat/masks/<地点>-quay-v<版>.json）。水側は陸マスクで決める（桟橋は両側）",
        "bands_m": {"inner": [0, 30], "outer": [30, 60]},
        "step_m": c["step_m"],
        "occupied": f"帯の中の ΔVV（今回−基準 p10、3×3 平均）≥ +{c['dvv_db']:g} dB かつ ΔVH ≥ +{c['dvh_db']:g} dB、かつ今回の VV ≥ {c['vv_min_db']:g} dB",
        "runs": f"{c['max_gap_m']:g} m 未満の切れ目はつなぎ、{c['min_run_m']:g} m 以上続く区間を占有区間とする",
        "minimum": f"占有区間の数（ΔVV の断面に {c['valley_db']:g} dB 以上の谷があれば切って加える。切った片は {c['min_piece_m']:g} m 以上）",
        "maximum": (f"Σ max(片の数, floor(区間長／{c['ship_min_m']:g} m))。外側の帯のうち {c['raft_from_m']:g}〜60 m（内側の船の 3×3 のにじみ約 10 m を除く）も"
                    f"内側と並んで {c['raft_min_m']:g} m 以上占有なら横抱きの可能性として +1"),
        "edge": "帯は水際から測る：線から法線方向に 0〜30 m を 5 m おきに見て、最初に陸マスクの水域になる所を水際とする（海岸線は線上、桟橋は中心線から甲板の半幅）",
        "length": f"区間長 − {c['length_trim_m']:g} m（目安・未検証）",
        "ownership": f"帯の画素は最も近い線のもの（差 {c['own_tolerance_m']:g} m まで）。桟橋の間の船を二重に数えない",
        "orbits": "基準は撮影と同じ相対軌道のものだけを使う（軌道を混ぜない）",
        "features": {
            "version": FEATURES_VERSION,
            "status": "記録だけ（数・推定には使わない）。2部の設計 est-2 の準備",
            "look_az_deg": ("レーダーが地上で見る方位（北から時計回り）。撮影の軌道の向きと地点の緯度から、Sentinel-1 の軌道（傾斜角 98.18°、周期 98.6 分）と地球の自転で"
                            "地上の進行方向を求め、右向き観測として +90°（湛江の上昇軌道で約 77.6°＝東北東、下降で約 282°）"),
            "cos_phi": (f"水側の法線と、レーダーのある方向（look_az_deg＋180°）のなす角 φ の cos（区間の点の中央値）。+1＝水側の面がレーダーを正面に向く。"
                        f"face_look＝区間の点の多数決で toward（cosφ＞{c['face_cos']:g}、レーダー向き）／away（cosφ＜−{c['face_cos']:g}、反対向き）／along（沿う）、face_shares＝その割合"),
            "depth": (f"depth_m＝ΔVV・ΔVH（今回の 3×3 平均−基準）で水際から 5〜{c['depth_max_m']:g} m に占有が続く奥行き（2026-10-30 の feat-1 で 90→{c['depth_max_m']:g} m）。"
                      "depth_1x3_m＝今回を線に沿う 3 画素（−10・0・+10 m）だけで平均し法線方向には平均しない版（基準は 3×3 の p10 のまま）。どちらも区間の点の中央値、5 m 単位"),
            "db_line": (f"外側の二回反射の線：区間の点の今回の VV（1×3）の断面を線形で平均し、奥行き（depth_1x3_m、無ければ depth_m、どちらも無ければ 45 m）＋15 m までの極大のうち"
                        f"最も強いものから {c['db_rel_db']:g} dB 以内で最も外側のものを取り、その山（−3 dB まで）のパワー重心の水際からの距離を db_line_m、山の値を db_line_db、"
                        "そうした山の数（10 m 以内の重なりは 1 つ）を db_peaks とする（未検証。10 m 画素なので距離の分解能は約 10 m）"),
            "pair_id": ("同じ岸壁を2つの軌道で見る組（36 時間以内。その間に入出港があれば船は違う）：時刻順に、別の相対軌道の撮影で 36 時間以内のものと組にする（湛江では 157 と翌日の 84 など）。"
                        "組は保存のたびに全行で作り直し、過去の撮影にも付く（過去の月の GeoJSON も書き直す）。時刻順に後から来る撮影では既存の組は変わらない"),
        },
    }
