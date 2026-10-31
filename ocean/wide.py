"""cfar-v2 PR3 probe, mode ``wide``: one wide-area snapshot of large-ship candidates (probe only).

    python -m ocean.cfar_probe --mode wide --out DIR ...

Regions in priority order (``REGIONS``): Senkaku, east of Taiwan (to 21.5 N), Mischief Reef, the Paracels, and the
East China Sea to south of the Sakishima islands (23.5-31.5 N, 121.5-129.5 E); Sentinel-1 IW
scenes of the last ``LOOKBACK_DAYS`` (7) days, fetched at 40 m (1/4 of the PU of 20 m) with BILINEAR.

1. Catalog (no PU) over the region; passes grouped into acquisitions (``cdse.acquisitions``).
2. Plan on a 0.2 x 0.2 deg cell grid: share of each cell inside each acquisition footprint, and the
   sea share of each cell from coarse land polygons (mainland China coast incl. Zhoushan, north-east
   Taiwan; small islands are left to the OSM mask). Greedy: rectangles of whole cells, at most 2,500 px
   (100 km at 40 m) per side, chosen by new sea cells covered (cells already covered count as zero),
   rectangles with at least 80% useful cells first (then 60%, 40%), newer acquisitions preferred on
   ties, until the PU budget is used. Overlap between requests is avoided this way.
3. Per request: OSM land/fixed objects (Overpass, one query per request, 10 s apart), Process API
   (VV, VH, LIA, 40 m), cfar-v1 (``sat.detect``) as the detector of record, with the guard and ring
   scaled to keep the pixel counts of 20 m (guard 160 m = 4 px, ring 320 m = 8 px), and cfar-v2
   (recorded only: counts and the match with v1 within 100 m).
4. Output (no sigma0 is written): ``candidates.geojson`` (v1 candidates with length >= 80 m, position,
   time, length estimate, confidence, near_coast, static (fixed object), edge, v2 match),
   ``coverage.geojson`` (requests), ``holes.geojson`` (sea cells not seen: no scene in 7 days, not
   fetched within the budget, no data in the fetched image, or no OSM mask), ``summary.json`` and
   ``report.txt``. Candidates within 1 km of the Chinese mainland coast and Zhoushan (west of 123 E,
   north of 27 N, near_coast) are left out.

Limits: at 40 m a ship of 80 m is 2 pixels; the length is a rough estimate (footprint incl. one pixel,
BILINEAR blur), and fishing vessels of 40-60 m can pass the 80 m cut. Candidates are unconfirmed.
"""
from __future__ import annotations

import json
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from sat import cdse
from sat import detect as v1
from sat import detect2 as v2
from sat.grid import Grid, dilate, m_per_deg

WIDE_VERSION = "ocean-wide-1"
# Regions in priority order (HQ, 2026-10-30): Senkaku -> east of Taiwan -> Mischief Reef -> Paracels -> rest
# of the East China Sea. ``min_eff``: lowest share of useful cells in a request (Senkaku: whatever it takes).
# ``fallback_days``: if no scene of the last LOOKBACK_DAYS covers half of the region, scenes this old are used.
REGIONS = [
    {"id": "SENKAKU", "name": "尖閣諸島周辺", "bbox": (123.3, 25.4, 124.0, 26.2), "min_eff": 0.05, "fallback_days": 12},
    {"id": "TWE", "name": "台湾東方", "bbox": (121.3, 21.5, 124.0, 25.4), "min_eff": 0.3},
    {"id": "MISCHIEF", "name": "ミスチーフ礁周辺", "bbox": (115.0, 9.0, 116.5, 10.5), "min_eff": 0.3},
    {"id": "PARACEL", "name": "パラセル（西沙）諸島周辺", "bbox": (111.0, 15.5, 113.0, 17.5), "min_eff": 0.3},
    {"id": "ECS", "name": "東シナ海〜先島諸島南方", "bbox": (121.5, 23.5, 129.5, 31.5), "min_eff": 0.4},
]
REGION = None  # bounding box of REGIONS (computed); tests may set it
CELL_DEG = 0.2
PIXEL_M = 40.0
MAX_PX = 2500
LOOKBACK_DAYS = 7
MIN_LENGTH_M = 80.0
V1_CFG_40M = {"guard_m": 160.0, "background_m": 320.0}
CN_EXCLUDE = {"west_of": 123.0, "north_of": 27.0}
ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026"
# coarse land for planning only (lon, lat); detection uses the OSM mask
LAND_POLYS = {
    "cn_shanghai": [(121.4, 31.6), (121.98, 31.6), (121.98, 30.85), (121.8, 30.62), (121.4, 30.62)],
    "cn_zhejiang": [(121.4, 30.32), (121.9, 30.32), (122.3, 30.12), (122.05, 29.62), (121.97, 29.0),
                    (121.72, 28.6), (121.55, 28.25), (121.4, 28.1)],
    "tw_east": [(120.9, 22.4), (121.2, 22.75), (121.37, 23.0), (121.45, 23.3), (121.5, 23.5), (120.9, 23.5)],
    "tw_northeast": [(121.4, 23.4), (121.4, 25.32), (121.6, 25.32), (121.75, 25.17), (121.93, 25.06),
                     (122.0, 25.0), (121.86, 24.85), (121.86, 24.6), (121.76, 24.3), (121.63, 24.0),
                     (121.52, 23.6), (121.48, 23.4)],
}
SUBREGIONS = [  # (id, name, west, south, east, north) -- first match wins
    ("SENKAKU", "尖閣諸島周辺（25.4–26.2N、123.3–124.0E）", 123.3, 25.4, 124.0, 26.2),
    ("TWE", "台湾東方（21.5–25.4N、121.3–124.0E）", 121.3, 21.5, 124.0, 25.4),
    ("MISCHIEF", "ミスチーフ礁周辺（9.0–10.5N、115.0–116.5E）", 115.0, 9.0, 116.5, 10.5),
    ("PARACEL", "パラセル諸島周辺（15.5–17.5N、111.0–113.0E）", 111.0, 15.5, 113.0, 17.5),
    ("ECS-N", "東シナ海北部（28.5–31.5N、121.5–126E）", 121.5, 28.5, 126.0, 31.5),
    ("ECS-S", "東シナ海南部・台湾北方（25.5–28.5N、121.5–126E）", 121.5, 25.5, 126.0, 28.5),
    ("NANSEI", "沖縄トラフ・南西諸島（奄美〜沖縄）周辺（25.5–31.5N、126–129.5E）", 126.0, 25.5, 129.5, 31.5),
    ("SAKI", "先島諸島周辺・台湾東方（23.5–25.5N、121.5–126E）", 121.5, 23.5, 126.0, 25.5),
    ("SOUTH", "沖縄本島〜先島の南方の太平洋（23.5–25.5N、126–129.5E）", 126.0, 23.5, 129.5, 25.5),
]


def regions_bbox() -> tuple[float, float, float, float]:
    if REGION:
        return REGION
    b = [r["bbox"] for r in REGIONS]
    return (min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b))


def subregion(lon: float, lat: float) -> str:
    for sid, _n, w, s, e, n in SUBREGIONS:
        if w <= lon <= e and s <= lat <= n:
            return sid
    return "OUT"


def _inside_ring(x, y, ring) -> np.ndarray:
    ring = np.asarray(ring, float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    xs, ys = ring[:, 0], ring[:, 1]
    res = np.zeros(np.shape(x), bool)
    for k in range(len(ring) - 1):
        x0, y0, x1, y1 = xs[k], ys[k], xs[k + 1], ys[k + 1]
        if y0 == y1:
            continue
        c = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        res ^= c & (x < xc)
    return res


class Cells:
    """0.2 deg cell grid of the region; sample points (k x k per cell) for shares."""

    def __init__(self, region=None, cell=CELL_DEG, k: int = 4):
        self.w, self.s, self.e, self.n = region or regions_bbox()
        self.cell = cell
        self.nx = int(round((self.e - self.w) / cell))
        self.ny = int(round((self.n - self.s) / cell))
        off = (np.arange(k) + 0.5) / k * cell
        cx = self.w + np.arange(self.nx) * cell
        cy = self.s + np.arange(self.ny) * cell
        # sample points shape (ny, nx, k*k)
        sx = (cx[None, :, None] + np.tile(off, k)[None, None, :]) * np.ones((self.ny, 1, 1))
        sy = (cy[:, None, None] + np.repeat(off, k)[None, None, :]) * np.ones((1, self.nx, 1))
        self.sx, self.sy = sx, sy
        land = np.zeros(sx.shape, bool)
        for ring in LAND_POLYS.values():
            land |= _inside_ring(sx, sy, ring)
        self.frac = {}
        for r in REGIONS:
            w, s, e, n = r["bbox"]
            self.frac[r["id"]] = ((sx >= w) & (sx < e) & (sy >= s) & (sy < n)).mean(axis=2)
        inside = np.max(np.stack(list(self.frac.values())), axis=0) if self.frac else np.ones(sx.shape[:2])
        self.sea_all = 1.0 - land.mean(axis=2)
        self.sea = self.sea_all * inside          # sea of the regions only (0 outside every region)
        # the region of each cell = the first (highest priority) region it is in
        self.region = np.full(self.sea.shape, "", dtype=object)
        for r in reversed(REGIONS):
            self.region[self.frac[r["id"]] > 0] = r["id"]

    def bounds(self, j: int, i: int, h: int = 1, w: int = 1) -> tuple[float, float, float, float]:
        """(west, south, east, north) of rows j..j+h-1 (south to north) and cols i..i+w-1."""
        return (round(self.w + i * self.cell, 6), round(self.s + j * self.cell, 6),
                round(self.w + (i + w) * self.cell, 6), round(self.s + (j + h) * self.cell, 6))

    def rect_of(self, bbox) -> tuple[int, int, int, int]:
        """(j, i, h, w) of a cell-aligned bbox."""
        w, s, e, n = bbox
        i, j = int(round((w - self.w) / self.cell)), int(round((s - self.s) / self.cell))
        return j, i, int(round((n - s) / self.cell)), int(round((e - w) / self.cell))

    def area_km2(self, j: int) -> float:
        lat = self.s + (j + 0.5) * self.cell
        my, mx = m_per_deg(lat)
        return self.cell * my * self.cell * mx / 1e6

    def footprint_share(self, geoms: list[dict]) -> np.ndarray:
        rings = []
        for g in geoms:
            if not g:
                continue
            if g.get("type") == "Polygon":
                rings.append(g["coordinates"][0])
            elif g.get("type") == "MultiPolygon":
                rings.extend(p[0] for p in g["coordinates"])
        hit = np.zeros(self.sx.shape, bool)
        for r in rings:
            hit |= _inside_ring(self.sx, self.sy, r)
        return hit.mean(axis=2)


def rect_px(b: tuple[float, float, float, float], pixel_m: float | None = None) -> tuple[int, int]:
    pixel_m = pixel_m or PIXEL_M
    w, s, e, n = b
    my, mx = m_per_deg((s + n) / 2)
    return int(round((e - w) * mx / pixel_m)), int(round((n - s) * my / pixel_m))


def _rect_sums(v: np.ndarray, h: int, w: int) -> np.ndarray:
    ii = np.zeros((v.shape[0] + 1, v.shape[1] + 1))
    ii[1:, 1:] = v.cumsum(0).cumsum(1)
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def greedy_plan(cells: Cells, fp: list[np.ndarray], recency: list[float], pu_left: float,
                max_requests: int = 200, thresholds=(0.8, 0.6, 0.4), max_cells: int = 5,
                passes: list[dict] | None = None, cov: np.ndarray | None = None,
                state: dict | None = None) -> tuple[list[dict], np.ndarray]:
    """Rectangles (acquisition index, j, i, h, w) chosen greedily; returns (plan, covered share per cell).

    ``passes`` (priority order): {"id", "weight" (cells of this pass), "acqs" (allowed acquisition indices),
    "min_eff"}. In a pass a request must contain uncovered sea of the pass; sea of other regions in the
    rectangle counts half. Without ``passes`` there is one pass over all sea cells."""
    state = state if state is not None else {}
    cov = state.setdefault("cov", np.zeros((cells.ny, cells.nx)) if cov is None else cov)
    state.setdefault("covA", {})
    state.setdefault("cov2", np.zeros((cells.ny, cells.nx)))
    passes = passes or [{"id": None, "weight": np.ones((cells.ny, cells.nx)), "acqs": list(range(len(fp))),
                         "min_eff": thresholds[-1]}]
    plan, spent = [], 0.0
    for ps in passes:
        thr_list = [t for t in thresholds if t >= ps["min_eff"]] + ([ps["min_eff"]] if ps["min_eff"] < thresholds[-1] else [])
        p2, spent = _greedy_pass(cells, fp, recency, pu_left, max_requests - len(plan), thr_list, max_cells,
                                 ps, cov, spent, state)
        for p in p2:
            p["region"] = ps["id"]
            p["kind"] = ps.get("kind", "first")
        plan += p2
    return plan, cov


def _greedy_pass(cells, fp, recency, pu_left, max_requests, thresholds, max_cells, ps, cov, spent, state=None):
    """kind "first": cells not yet seen. kind "second": cells already seen by another scene of
    ``ps["other_acqs"]`` (a second time over the same sea), not yet seen twice."""
    state = state if state is not None else {"covA": {}, "cov2": np.zeros_like(cov)}
    covA, cov2 = state["covA"], state["cov2"]
    zero = np.zeros_like(cov)
    kind = ps.get("kind", "first")
    plan = []
    sizes = [(h, w) for h in range(1, max_cells + 1) for w in range(1, max_cells + 1)]
    cur_w = ps["weight"]
    wgt = np.maximum(cur_w, 0.5 * (cells.sea > 0))
    while len(plan) < max_requests:
        best = None
        for thr in thresholds:
            for a in ps["acqs"]:
                f = fp[a]
                if kind == "second":
                    others = [covA[b] for b in ps.get("other_acqs", list(covA)) if b != a and b in covA]
                    other = np.max(np.stack(others), axis=0) if others else zero
                    free = np.minimum((f - covA.get(a, zero)).clip(0), (other - cov2).clip(0))
                else:
                    free = (f - cov).clip(0)   # new share only (a scene never pays twice for its own cells)
                val = cells.sea * free * wgt
                cur = cells.sea * free * cur_w
                if cur.sum() <= 1e-6:
                    continue
                for h, w in sizes:
                    if h > cells.ny or w > cells.nx:
                        continue
                    sums = _rect_sums(val, h, w)
                    sums = np.where(_rect_sums(cur, h, w) > 0.05, sums, 0.0)
                    if sums.max() <= 0:
                        continue
                    # px of a rect depends on latitude: check the widest (southern) row of the grid
                    for j in np.unique(np.argwhere(sums > 0)[:, 0]):
                        b = cells.bounds(j, 0, h, w)
                        wp, hp = rect_px(b)
                        if wp > MAX_PX or hp > MAX_PX:
                            continue
                        cost = cdse.estimate_pu(wp, hp, 3)
                        if spent + cost > pu_left + 1e-9:
                            continue
                        row = sums[j]
                        eff = row / (h * w)
                        ok = eff >= thr
                        if not ok.any():
                            continue
                        i = int(np.argmax(np.where(ok, row, -1)))
                        score = row[i] * (1 + 0.1 * recency[a])
                        if best is None or score > best[0] + 1e-9:
                            best = (score, a, int(j), i, h, w, cost, float(row[i]), float(eff[i]))
            if best is not None:
                break
        if best is None:
            break
        _s, a, j, i, h, w, cost, value, eff = best
        if kind == "second":
            others = [covA[b] for b in ps.get("other_acqs", list(covA)) if b != a and b in covA]
            other = np.max(np.stack(others), axis=0) if others else zero
            cov2[j:j + h, i:i + w] = np.maximum(cov2[j:j + h, i:i + w], np.minimum(fp[a], other)[j:j + h, i:i + w])
        ca = covA.setdefault(a, np.zeros_like(cov))
        ca[j:j + h, i:i + w] = np.maximum(ca[j:j + h, i:i + w], fp[a][j:j + h, i:i + w])
        cov[j:j + h, i:i + w] = np.maximum(cov[j:j + h, i:i + w], fp[a][j:j + h, i:i + w])
        spent += cost
        plan.append({"acq_index": a, "j": j, "i": i, "h": h, "w": w, "est_pu": cost,
                     "value_cells": round(value, 3), "efficiency": round(eff, 3)})
    return plan, spent


def confidence(d: dict) -> str:
    c = d.get("contrast_vv_db") or 0.0
    r = d.get("vh_ratio_db") or 0.0
    level = "high" if (c >= 10 and r >= 6 and d["pixels"] >= 4) else "medium" if c >= 6 else "low"
    if d.get("edge") or d.get("static"):
        level = {"high": "medium", "medium": "low", "low": "low"}[level]
    return level


def detect_rect(vv, vh, lia, land, static, grid: Grid, acq: dict) -> tuple[list[dict], dict]:
    """cfar-v1 (record) + cfar-v2 (counts and match only) on one 40 m request."""
    px = float(np.mean(grid.pixel_m))
    valid = np.isfinite(vv) & np.isfinite(vh)
    t0 = time.perf_counter()
    d1, i1 = v1.detect(vv, vh, land, grid.pixel_m, V1_CFG_40M)
    s1 = round(time.perf_counter() - t0, 2)
    edge = dilate(~valid, 3) if (~valid).any() else np.zeros_like(valid)
    near = dilate(land, 1000.0 / px) & ~land if land.any() else np.zeros_like(land)
    lat = (grid.south + grid.north) / 2
    heading = v2.platform_heading((acq.get("orbit_direction") or "").lower(), lat) if acq.get("orbit_direction") else None
    t0 = time.perf_counter()
    try:
        d2, i2 = v2.detect_v2(vv, vh, land, grid.pixel_m, lia=lia, heading_deg=heading, static=static)
        k2 = v2.kept(d2)
        v2err = None
    except Exception as e:  # v2 is recorded only; a failure must not stop v1
        k2, i2, v2err = [], {}, f"{type(e).__name__}: {str(e)[:200]}"
    s2 = round(time.perf_counter() - t0, 2)
    pts2 = np.array([[d["row"], d["col"]] for d in k2]) if k2 else np.zeros((0, 2))
    out = []
    for d in d1:
        r, c = int(round(d["row"])), int(round(d["col"]))
        r, c = min(max(r, 0), grid.height - 1), min(max(c, 0), grid.width - 1)
        d["static"], d["edge"], d["near_coast"] = bool(static[r, c]), bool(edge[r, c]), bool(near[r, c])
        if d["length_m"] < MIN_LENGTH_M:
            continue
        lon, la = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
        lon, la = float(lon), float(la)
        if d["near_coast"] and lon < CN_EXCLUDE["west_of"] and la > CN_EXCLUDE["north_of"]:
            continue
        m = None
        if len(pts2):
            dist = np.hypot(pts2[:, 0] - d["row"], pts2[:, 1] - d["col"]) * px
            k = int(np.argmin(dist))
            if dist[k] <= 100.0:
                m = k2[k]
        out.append({
            "lon": round(lon, 5), "lat": round(la, 5), "acquired_utc": acq["acquired_utc"],
            "platform": acq.get("platform"), "orbit_direction": acq.get("orbit_direction"),
            "relative_orbit": acq.get("relative_orbit"),
            "length_m_est": d["length_m"], "width_m_est": d["width_m"], "length_class": d["length_class"],
            "pixels_40m": d["pixels"], "peak_vv_db": d["peak_vv_db"], "contrast_vv_db": d["contrast_vv_db"],
            "vh_ratio_db": d["vh_ratio_db"], "near_coast": d["near_coast"], "static": d["static"], "edge": d["edge"],
            "confidence": confidence(d), "subregion": subregion(lon, la), "detector": "cfar-v1 (40 m, guard 160 m, ring 320 m)",
            "v2_match": m is not None, "v2_confidence": m.get("confidence") if m else None,
            "v2_length_m": m.get("length_m") if m else None,
        })
    water = valid & ~land
    km2 = float(water.sum()) * grid.pixel_m[0] * grid.pixel_m[1] / 1e6
    info = {"valid_fraction": round(float(valid.mean()), 4), "water_km2": round(km2, 1),
            "vv_median_db": round(float(np.nanmedian(vv)), 2) if valid.any() else None,
            "v1": {"candidates": len(d1), "large_kept": len(out), "seconds": s1,
                   "per_1000km2": round(len(d1) / km2 * 1000, 2) if km2 else None,
                   **{k: i1[k] for k in ("hit_pixels", "clusters", "dropped_small", "dropped_vh")}},
            "v2": {"candidates": len(k2), "confidence_A": sum(1 for d in k2 if d.get("confidence") == "A"),
                   "large": sum(1 for d in k2 if d.get("large")), "seconds": s2, "error": v2err,
                   "tiles": i2.get("tiles"), "tiles_untested": i2.get("tiles_untested"),
                   "model_counts": i2.get("model_counts"),
                   "matched_v1_large": sum(1 for o in out if o["v2_match"])}}
    return out, info


def cell_valid(valid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Share of valid pixels per cell (h x w cells, row 0 = north) -> (h, w) with row 0 = south."""
    rows = np.array_split(np.arange(valid.shape[0]), h)
    cols = np.array_split(np.arange(valid.shape[1]), w)
    out = np.zeros((h, w))
    for a, rr in enumerate(rows):
        for b, cc in enumerate(cols):
            out[h - 1 - a, b] = valid[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1].mean() if len(rr) and len(cc) else 0
    return out


def dedupe(cands: list[dict], tol_m: float = 200.0) -> list[dict]:
    out = []
    for c in cands:
        dup = False
        for o in out:
            if o["acquired_utc"] != c["acquired_utc"]:
                continue
            my, mx = m_per_deg(c["lat"])
            if math.hypot((o["lat"] - c["lat"]) * my, (o["lon"] - c["lon"]) * mx) <= tol_m:
                dup = True
                break
        if not dup:
            out.append(c)
    return out


def _poly(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def holes_features(cells: Cells, reason: np.ndarray) -> list[dict]:
    """Merge runs of cells with the same reason along each row into rectangles."""
    feats = []
    for j in range(cells.ny):
        i = 0
        while i < cells.nx:
            r = reason[j, i]
            if not r:
                i += 1
                continue
            k = i
            while k + 1 < cells.nx and reason[j, k + 1] == r and cells.region[j, k + 1] == cells.region[j, i]:
                k += 1
            w, s, e, n = cells.bounds(j, i, 1, k - i + 1)
            km2 = sum(cells.area_km2(j) * cells.sea[j, x] for x in range(i, k + 1))
            feats.append({"type": "Feature", "geometry": _poly(w, s, e, n),
                          "properties": {"reason": r, "region": cells.region[j, i], "sea_km2_approx": round(km2, 1)}})
            i = k + 1
    return feats


# extend run (--extend): hole filling with older scenes, in days (HQ 2026-10-30: fill the holes, then a
# second look within 7 days, priority Senkaku -> east of Taiwan -> Mischief -> Paracels -> East China Sea)
# second look window (days): Senkaku is imaged by one track only (171, every 12 days) -> the previous pass
SECOND_DAYS = {"SENKAKU": 13}
EXTEND_HOLE_DAYS = {"SENKAKU": 12, "TWE": 12, "MISCHIEF": 30, "PARACEL": 30, "ECS": 12}


def load_prior(out: Path) -> list[dict]:
    """Results of earlier (not dry-run) wide runs next to ``out`` (same probe branch)."""
    rows = []
    for f in sorted(out.parent.glob("*/summary.json")):
        if f.parent == out:
            continue
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if s.get("mode") != "wide" or s.get("dry_run"):
            continue
        for r in s.get("results", []):
            rows.append({**r, "_run": f.parent.name})
    return rows


HOLE_REASON = {"no_scene": "7 日以内に撮影なし", "budget": "撮影はあるが PU の上限で取得せず",
               "no_data": "取得したが画像にデータなし（撮影の端など）", "osm_failed": "陸地マスク（OSM）を取れず取得せず"}


def run_wide(args, cp, client, overpass, now, pacer, sleep, log, clock) -> dict:
    month = now.strftime("%Y-%m")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sat_rows = [r for f in args.sat_usage for r in cp.read_csv(Path(f))]
    ocean_rows = [r for f in [args.usage, *args.ocean_usage] for r in cp.read_csv(Path(f))]
    bud = cp.budget(month, sat_rows, ocean_rows, args.max_pu)
    lookback = int(args.lookback_days or LOOKBACK_DAYS)
    summary = {"probe_version": WIDE_VERSION, "mode": "wide", "run_utc": cdse.iso(now), "region": REGION,
               "pixel_m": PIXEL_M, "interpolation": cp.INTERPOLATION, "lookback_days": lookback,
               "dry_run": bool(args.dry_run), "budget": bud, "attribution": ATTRIBUTION,
               "min_length_m": MIN_LENGTH_M, "v1_cfg": V1_CFG_40M, "subregions": SUBREGIONS,
               "hole_reasons": HOLE_REASON, "acquisitions": [], "plan": {}, "results": [], "errors": []}
    log(f"今月 {month}：sat 等 {bud['sat_month_pu']} PU・外洋 {bud['ocean_month_pu']} PU 使用済み → 使える {bud['pu_left']} PU")
    summary["region"] = regions_bbox()
    summary["regions"] = REGIONS
    extend = bool(getattr(args, "extend", False))
    summary["extend"] = extend
    max_days = max([lookback] + [int(r.get("fallback_days") or 0) for r in REGIONS]
                   + (list(EXTEND_HOLE_DAYS.values()) if extend else []))
    by_id = {}
    for r in REGIONS:
        try:
            for f in client.catalog(list(r["bbox"]), now - timedelta(days=max_days), now):
                by_id[f.get("id")] = f
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as ex:
            summary["errors"].append({"stage": "catalog", "region": r["id"], "error": str(ex)[:300]})
    feats = list(by_id.values())
    acqs = [a for a in cdse.acquisitions(feats)]
    cells = Cells()
    fp = [cells.footprint_share([(by_id.get(i) or {}).get("geometry") for i in a["ids"]]) for a in acqs]
    ages = [(now - a["time"]).total_seconds() / 86400 for a in acqs]
    recency = [max(0.0, 1 - g / max_days) for g in ages]
    passes = []
    for r in REGIONS:
        fr = cells.frac[r["id"]] * (cells.sea_all > 0)
        recent = [k for k, g in enumerate(ages) if g <= lookback]
        share = lambda ks: max([float((np.minimum(fp[k], 1) * fr).sum() / max(fr.sum(), 1e-9)) for k in ks] or [0.0])
        use, days = recent, lookback
        if r.get("fallback_days") and share(recent) < 0.5:
            use, days = [k for k, g in enumerate(ages) if g <= r["fallback_days"]], r["fallback_days"]
        passes.append({"id": r["id"], "weight": cells.frac[r["id"]], "acqs": use, "min_eff": r["min_eff"], "days": days})
    for k, (a, f) in enumerate(zip(acqs, fp)):
        summary["acquisitions"].append({**cp._acq_meta(a), "absolute_orbit": a.get("absolute_orbit"), "slices": len(a["ids"]),
                                        "age_days": round(ages[k], 2),
                                        "region_sea_share": {r["id"]: round(float((f * cells.frac[r["id"]] * cells.sea).sum()
                                                                                  / max((cells.frac[r["id"]] * cells.sea).sum(), 1e-9)), 4)
                                                             for r in REGIONS}})
    state = {"cov": np.zeros((cells.ny, cells.nx)), "covA": {}, "cov2": np.zeros((cells.ny, cells.nx))}
    observed = np.zeros((cells.ny, cells.nx))
    obs_by_acq: dict = {}
    prior = []
    if extend:
        prior = load_prior(Path(args.out))
        summary["prior_runs"] = sorted({r["_run"] for r in prior})
        idx = {a["acquired_utc"]: k for k, a in enumerate(acqs)}
        for r in prior:
            j, i, h, w = cells.rect_of(r["bbox"])
            if h <= 0 or w <= 0:
                continue
            sl = (slice(j, j + h), slice(i, i + w))
            k = idx.get(r["acquired_utc"])
            share = fp[k][sl] if k is not None else np.full((h, w), float(r.get("valid_fraction") or 0))
            if k is not None:
                ca = state["covA"].setdefault(k, np.zeros((cells.ny, cells.nx)))
                ca[sl] = np.maximum(ca[sl], share)
            state["cov"][sl] = np.maximum(state["cov"][sl], share)
            observed[sl] = np.maximum(observed[sl], share)
            oa = obs_by_acq.setdefault(r["acquired_utc"], np.zeros((cells.ny, cells.nx)))
            oa[sl] = np.maximum(oa[sl], share)
        recent = [k for k, g in enumerate(ages) if g <= lookback]
        byid = {p["id"]: p for p in passes}
        hole = lambda rid: {"id": rid, "kind": "first", "weight": cells.frac[rid], "min_eff": byid[rid]["min_eff"],
                            "days": EXTEND_HOLE_DAYS.get(rid, 12),
                            "acqs": [k for k, g in enumerate(ages) if g <= EXTEND_HOLE_DAYS.get(rid, 12)]}
        win = lambda rid: [k for k, g in enumerate(ages) if g <= SECOND_DAYS.get(rid, lookback)]
        second = lambda rid: {"id": rid, "kind": "second", "weight": cells.frac[rid], "min_eff": byid[rid]["min_eff"],
                              "days": SECOND_DAYS.get(rid, lookback), "acqs": win(rid), "other_acqs": win(rid)}
        ids = [r["id"] for r in REGIONS]
        # holes of the priority regions, a second look at the first two, holes of the last (largest) region,
        # then second looks at the rest
        passes = ([hole(r) for r in ids[:-1]] + [second(r) for r in ids[:2] if r != ids[-1]] + [hole(ids[-1])]
                  + [second(r) for r in ids[2:]] + ([second(ids[-1])] if len(ids) <= 2 else []))
    summary["passes"] = [{"id": p["id"], "kind": p.get("kind", "first"), "days": p["days"],
                          "acquisitions": [acqs[k]["acquired_utc"] for k in p["acqs"]]} for p in passes]
    plan, cov = greedy_plan(cells, fp, recency, max(0.0, bud["pu_left"] - 2.0), max_requests=min(200, bud["requests_left"]),
                            passes=passes, state=state)
    any_fp = np.zeros((cells.ny, cells.nx))
    for ps in passes:  # imaged within the region's own window
        for k in ps["acqs"]:
            any_fp = np.maximum(any_fp, fp[k] * (cells.frac[ps["id"]] > 0))
    sea_km2 = float(sum(cells.area_km2(j) * cells.sea[j].sum() for j in range(cells.ny)))
    plan_km2 = float(sum(cells.area_km2(j) * (cells.sea[j] * cov[j]).sum() for j in range(cells.ny)))
    seen_km2 = float(sum(cells.area_km2(j) * (cells.sea[j] * any_fp[j]).sum() for j in range(cells.ny)))
    for p in plan:
        a = acqs[p["acq_index"]]
        p["bbox"] = list(cells.bounds(p["j"], p["i"], p["h"], p["w"]))
        p["width"], p["height"] = rect_px(tuple(p["bbox"]))
        p["acquired_utc"] = a["acquired_utc"]
    summary["plan"] = {"requests": len(plan), "est_pu": round(sum(p["est_pu"] for p in plan), 3),
                       "region_sea_km2": round(sea_km2), "imaged_in_lookback_sea_km2": round(seen_km2),
                       "planned_sea_km2": round(plan_km2), "items": plan}
    log(f"計画: {len(plan)} 件 {summary['plan']['est_pu']} PU、海域 {sea_km2:.0f} km² のうち 7 日以内の撮影 {seen_km2:.0f}・取得予定 {plan_km2:.0f} km²")
    osm_fail = np.zeros((cells.ny, cells.nx), bool)
    cands = []
    if not args.dry_run and plan:
        cfg = cp.load_areas(args.areas)
        seamark_re = cfg["osm_static"]["seamark_types"]
        deadline = time.monotonic() + args.max_minutes * 60
        queue = [(k, 0) for k in range(len(plan))]
        n_done = 0
        while queue:
            k, attempt = queue.pop(0)
            p = plan[k]
            a = acqs[p["acq_index"]]
            g = Grid(p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3], p["width"], p["height"])
            sl = (slice(p["j"], p["j"] + p["h"]), slice(p["i"], p["i"] + p["w"]))
            if clock().strftime("%Y-%m") != month:
                summary["errors"].append({"stage": "stop", "error": "月が替わったため以降の取得をしない"})
                break
            if time.monotonic() > deadline:
                summary["errors"].append({"stage": "stop", "error": f"{args.max_minutes} 分を過ぎたため止めた"})
                break
            if client.processing_units + p["est_pu"] > bud["pu_left"] + 1e-9:
                summary["errors"].append({"stage": "stop", "error": "実際の PU が上限に届くため止めた"})
                break
            if n_done:
                sleep(10 if not attempt else 60)
            n_done += 1
            try:
                osm = overpass(cp.overpass_query(g, seamark_re))
            except Exception as ex:
                if attempt == 0:  # Overpass busy (e.g. 502): try this request once more at the end
                    queue.append((k, 1))
                    continue
                osm_fail[sl] = True
                summary["errors"].append({"request": k, "stage": "overpass", "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
                continue
            pacer.wait(p["est_pu"])
            try:
                t0 = time.perf_counter()
                data, pu = client.process(g.bbox, g.width, g.height, a["t_from"], a["t_to"], with_lia=True,
                                          upsampling=cp.INTERPOLATION)
                pacer.spent(pu)
                tf = round(time.perf_counter() - t0, 1)
                vv, vh, lia = cp.decode(data)
                del data
                land, static, minfo = cp.masks(g, osm, seamark_re, float(cfg["static_dilate_m"]))
                c, info = detect_rect(vv, vh, lia, land, static, g, a)
                valid = np.isfinite(vv) & np.isfinite(vh)
                cv = cell_valid(valid, p["h"], p["w"])
                observed[sl] = np.maximum(observed[sl], cv)
                oa = obs_by_acq.setdefault(a["acquired_utc"], np.zeros((cells.ny, cells.nx)))
                oa[sl] = np.maximum(oa[sl], cv)
                cands.extend(c)
                row = {"request": k, **cp._acq_meta(a), "bbox": g.bbox, "width": g.width, "height": g.height,
                       "pu": pu, "fetch_seconds": tf, **info,
                       "mask": {kk: minfo.get(kk) for kk in ("land_fraction", "static_objects", "static_fraction", "osm_base")}}
                summary["results"].append(row)
                log(f"#{k} {a['acquired_utc']} {g.bbox}: PU {pu}、有効 {info['valid_fraction']}、VV {info['vv_median_db']} dB、"
                    f"v1 {info['v1']['candidates']}（80 m 以上 {info['v1']['large_kept']}）、v2 {info['v2']['candidates']}")
            except cdse.CredentialError:
                raise
            except cdse.CDSEError as ex:
                summary["errors"].append({"request": k, "acquired_utc": a["acquired_utc"], "stage": "fetch", "error": str(ex)[:300]})
            cp._write_summary(out, summary, client, pacer)
    cands = dedupe(cands)
    # holes
    reason = np.full((cells.ny, cells.nx), "", dtype=object)
    sea_cell = cells.sea > 0.05
    fetched_any = np.zeros((cells.ny, cells.nx), bool)
    for r in summary["results"]:
        p = plan[r["request"]]
        fetched_any[p["j"]:p["j"] + p["h"], p["i"]:p["i"] + p["w"]] = True
    for j in range(cells.ny):
        for i in range(cells.nx):
            if not sea_cell[j, i] or observed[j, i] >= 0.5:
                continue
            if osm_fail[j, i] and not fetched_any[j, i]:
                reason[j, i] = "osm_failed"
            elif fetched_any[j, i]:
                reason[j, i] = "no_data"
            elif any_fp[j, i] >= 0.5:
                reason[j, i] = "budget"
            else:
                reason[j, i] = "no_scene"
    hf = holes_features(cells, reason)
    per_region = {}
    for rg in REGIONS:
        fr = cells.frac[rg["id"]]
        km = lambda arr: float(sum(cells.area_km2(j) * (cells.sea[j] * fr[j] * arr[j]).sum() for j in range(cells.ny)))
        rs = [r for r in summary["results"] if fr[plan[r["request"]]["j"]:plan[r["request"]]["j"] + plan[r["request"]]["h"],
                                                   plan[r["request"]]["i"]:plan[r["request"]]["i"] + plan[r["request"]]["w"]].sum() > 0]
        w0, s0, e0, n0 = rg["bbox"]
        cs = [c for c in cands if w0 <= c["lon"] <= e0 and s0 <= c["lat"] <= n0]
        tot = km(np.ones_like(observed))
        per_region[rg["id"]] = {
            "name": rg["name"], "bbox": rg["bbox"], "sea_km2": round(tot),
            "observed_sea_km2": round(km((observed >= 0.5).astype(float))),
            "observed_share": round(km((observed >= 0.5).astype(float)) / tot, 3) if tot else None,
            "imaged_in_window_share": round(km(np.minimum(any_fp, 1.0)) / tot, 3) if tot else None,
            "window_days": next(p["days"] for p in passes if p["id"] == rg["id"]),
            "acquisitions_used": sorted({r["acquired_utc"] for r in rs}),
            "requests": len(rs), "pu": round(sum(r["pu"] for r in rs), 3),
            "seen_twice_sea_km2": round(km((sum((o >= 0.5).astype(int) for o in obs_by_acq.values()) >= 2).astype(float)))
                                  if obs_by_acq else 0,
            "acquisitions_incl_prior": sorted(t for t, o in obs_by_acq.items() if ((o >= 0.5) * fr * cells.sea).sum() > 0),
            "candidates": len(cs), "candidates_high": sum(1 for c in cs if c["confidence"] == "high"),
            "candidates_medium": sum(1 for c in cs if c["confidence"] == "medium"),
            "candidates_static": sum(1 for c in cs if c["static"]),
            "candidates_near_coast": sum(1 for c in cs if c["near_coast"])}
    summary["per_region"] = per_region
    obs_km2 = float(sum(cells.area_km2(j) * (cells.sea[j] * (observed[j] >= 0.5)).sum() for j in range(cells.ny)))
    hole_km2 = {r: round(sum(f["properties"]["sea_km2_approx"] for f in hf if f["properties"]["reason"] == r)) for r in HOLE_REASON}
    by_sub = {}
    for c in cands:
        d = by_sub.setdefault(c["subregion"], {"all": 0, "high": 0, "medium": 0, "low": 0, "static": 0,
                                               "near_coast": 0, "v2_match": 0, "ge_150m_est": 0})
        d["all"] += 1
        d[c["confidence"]] += 1
        d["static"] += c["static"]
        d["near_coast"] += c["near_coast"]
        d["v2_match"] += c["v2_match"]
        d["ge_150m_est"] += c["length_m_est"] >= 150
    summary["coverage"] = {"observed_sea_km2": round(obs_km2), "region_sea_km2": round(sea_km2), "holes_sea_km2": hole_km2,
                           "observed_share": round(obs_km2 / sea_km2, 3) if sea_km2 else None}
    summary["candidates"] = {"total": len(cands), "by_subregion": by_sub}
    cand_fc = {"type": "FeatureCollection", "name": "大型船の候補（長さ約 80 m 以上、未確認）",
               "attribution": ATTRIBUTION + "; land and fixed objects (c) OpenStreetMap contributors, ODbL 1.0",
               "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
                             "properties": {k: v for k, v in c.items() if k not in ("lon", "lat")}} for c in cands]}
    cov_fc = {"type": "FeatureCollection", "attribution": ATTRIBUTION, "features": [
        {"type": "Feature", "geometry": _poly(*r["bbox"]),
         "properties": {"acquired_utc": r["acquired_utc"], "platform": r.get("platform"), "orbit_direction": r.get("orbit_direction"),
                        "relative_orbit": r.get("relative_orbit"), "valid_fraction": r["valid_fraction"], "pu": r["pu"],
                        "v1_large": r["v1"]["large_kept"]}} for r in summary["results"]]}
    hole_fc = {"type": "FeatureCollection", "reasons": HOLE_REASON, "features": hf}
    for name, fc in (("candidates.geojson", cand_fc), ("coverage.geojson", cov_fc), ("holes.geojson", hole_fc)):
        (out / name).write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":"), default=cp._jsonable), encoding="utf-8")
    cp._write_summary(out, summary, client, pacer)
    cp.write_usage(Path(args.usage), {"run_utc": cdse.iso(now), "requests": client.requests,
                                      "processing_units": round(client.processing_units, 3),
                                      "pu_source": "header" if getattr(client, "pu_reported", True) else "estimate",
                                      "scenes": len({r["acquired_utc"] for r in summary["results"]}),
                                      "units": len(summary["results"]), "probe_version": WIDE_VERSION,
                                      "note": "dry-run" if args.dry_run else "wide 40 m"})
    rep = report(summary, client)
    (out / "report.txt").write_text(rep, encoding="utf-8")
    log(rep)
    return summary


def report(s: dict, client) -> str:
    b, p = s["budget"], s["plan"]
    L = [f"{WIDE_VERSION} {s['run_utc']}{'（dry-run）' if s['dry_run'] else ''} 範囲 {s['region']} 40 m、直近 {s['lookback_days']} 日",
         f"使える {b['pu_left']} PU、計画 {p['requests']} 件 {p['est_pu']} PU、実績 {client.requests} リクエスト・{round(client.processing_units, 3)} PU",
         f"海域 {p['region_sea_km2']} km²、7 日以内の撮影 {p['imaged_in_lookback_sea_km2']}、取得予定 {p['planned_sea_km2']}、"
         f"実際に見た {s.get('coverage', {}).get('observed_sea_km2')} km²、穴 {s.get('coverage', {}).get('holes_sea_km2')}",
         "撮影（カタログ）:"]
    for a in s["acquisitions"]:
        L.append(f"  {a['acquired_utc']} {a.get('platform')} {a.get('orbit_direction')} 軌道 {a.get('relative_orbit')} 範囲別の海の被覆 "
                 + json.dumps({k: v for k, v in a['region_sea_share'].items() if v}, ensure_ascii=False))
    for r in s["results"]:
        L.append(f"#{r['request']} {r['acquired_utc']} {r['bbox']} PU {r['pu']} 有効 {r['valid_fraction']} VV {r['vv_median_db']} "
                 f"v1 {r['v1']['candidates']}（80 m 以上 {r['v1']['large_kept']}、{r['v1']['per_1000km2']}/1000km²）v2 {r['v2']['candidates']}")
    for rid, v in (s.get("per_region") or {}).items():
        L.append(f"[{rid}] {v['name']}：海 {v['sea_km2']} km²、見た {v['observed_sea_km2']}（{v['observed_share']}）、"
                 f"窓 {v['window_days']} 日の撮影の被覆 {v['imaged_in_window_share']}、撮影 {v['acquisitions_used']}、"
                 f"{v['requests']} 件 {v['pu']} PU、候補 {v['candidates']}（high {v['candidates_high']}・medium {v['candidates_medium']}"
                 f"・固定物 {v['candidates_static']}・沿岸 {v['candidates_near_coast']}）")
    L.append("候補（範囲別）: " + json.dumps(s.get("candidates", {}), ensure_ascii=False))
    for e in s["errors"]:
        L.append("エラー: " + json.dumps(e, ensure_ascii=False, default=str))
    return "\n".join(L) + "\n"
