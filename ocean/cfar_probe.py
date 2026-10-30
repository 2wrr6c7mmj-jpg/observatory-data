"""cfar-v2 PR3 probe: open-sea Sentinel-1 chips, cfar-v1 and cfar-v2 on the same arrays.

Experimental, probe only (``.github/workflows/ocean-cfar-probe.yml``, push to ``probe/ocean-cfar*``).
Nothing in the daily run (``sat/run.py``), the Zhanjiang counts or the public data uses this.

    python -m ocean.cfar_probe --out DIR [--usage docs/data/ocean/usage.csv]
                               [--sat-usage FILE ...] [--ocean-usage FILE ...] [--dry-run]

Steps
-----
1. Areas from ``ocean/areas.json`` (centre + side, 20 m pixels), split into units of at most
   2,500 x 2,500 px (the Process API limit).
2. Catalog (Sentinel Hub, no PU): IW scenes with VV+VH of the last ``lookback_days`` whose footprint
   covers at least ``min_footprint_coverage`` of the unit; per area up to ``max_scenes``, spread over
   the period (evenly spaced in time order, so that different sea states/winds come in).
3. Plan and caps, before any PU is spent: the estimated PU of every request (``cdse.estimate_pu``)
   must fit (a) the ocean cap of the month (4,000 PU in 2026-10, 400 PU otherwise) with this month's ocean usage (``docs/data/ocean/usage.csv`` of every probe branch),
   (b) the guard on the whole account (``TOTAL_MONTHLY_PU_GUARD``: 9,000 PU in 2026-10) with this
   month's ``docs/data/sat/usage.csv`` of main and any other usage file given with ``--sat-usage``, (c) ``--max-pu`` for the run, (d) the
   monthly request cap. Requests are taken in priority order (the first scene of every area, then
   the second, ...) and the plan is cut where a cap would be passed. The ocean usage is kept out of
   ``docs/data/sat/usage.csv``: the daily run's own cap (1,000 PU) is not used up by this probe.
4. Rate: CDSE Sentinel Hub allows 300 requests and 300 PU per minute (HTTP 429 above). A pacer keeps
   the last 60 s under ``PU_PER_MIN`` (200) and ``REQ_PER_MIN`` (200) by waiting; ``cdse.Client``
   also waits and retries on 429. Months are UTC calendar months; a run stops before starting a
   request in a month other than the one it was planned in.
5. Fetch (Process API): VV, VH, local incidence angle, SIGMA0_ELLIPSOID, orthorectified, 20 m.
   ``downsampling`` is ignored for Sentinel-1 (Sentinel Hub docs, "Not used, use upsampling"):
   ``upsampling`` decides the interpolation "regardless of the resolution". The run uses BILINEAR
   and, once, measures the speckle (ENL) of a 5 km chip fetched with NEAREST and BILINEAR at 20 m
   and at 10 m (design 7.2 (a)). No-data pixels (dataMask 0) are NaN; a unit with less than
   ``min_valid_fraction`` valid pixels is recorded and not detected.
6. Masks: land from OSM coastlines (``sat.osm_mask.build_mask``); the 1 km coast band is handled
   by ``detect_v2`` (``near_coast``). Fixed objects (offshore wind turbines, platforms, buoys and
   beacons) from OSM, dilated by 300 m -> ``static``. One Overpass query per area and run.
7. ``detect.detect`` (cfar-v1) and ``detect2.detect_v2`` (cfar-v2) on the same arrays; counts,
   confidence, exclusions, per-tile model choice, z_t, measured exceedance at 1e-3 and 1e-4, time,
   PU, and the v1/v2 match (within 100 m).

Output (small files only; no sigma0 is written): ``DIR/summary.json``, ``DIR/report.txt``,
``DIR/<area>-<unit>-<time>.geojson`` (v1 and v2 candidates), and one row added to ``--usage``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sat import cdse, png
from sat import detect as v1
from sat import detect2 as v2
from sat.grid import Grid, dilate, draw_line, fill_polygon
from sat.osm_mask import OVERPASS_URL, USER_AGENT, WATER, build_mask

PROBE_VERSION = "ocean-cfar-probe-1"
AREAS_FILE = Path(__file__).resolve().parent / "areas.json"
MAX_PX = 2500
# Own ocean cap per UTC month; 2026-10: the October free tier is used before it lapses (HQ, 2026-10-30;
# raised from 4,000 to 4,600 PU on 2026-10-30 when the wide snapshot was extended, ports reduced to match).
OCEAN_MONTHLY_PU_CAP = {"2026-10": 4600.0}
OCEAN_DEFAULT_PU_CAP = 400.0
# Safety guard on the whole CDSE account: Sentinel-1 daily run (docs/data/sat/usage.csv) + other
# probes passed with --sat-usage (e.g. other harbours) + ocean. 2026-10: 9,000 of the 10,000 PU free tier.
TOTAL_MONTHLY_PU_GUARD = {"2026-10": 9000.0}
TOTAL_DEFAULT_PU_GUARD = 1400.0  # = the daily run's own 1,000 + the ocean 400
MONTHLY_REQUEST_CAP = 2000       # same number as sat/run.py, counted on sat + ocean together
# 2026-10: the ports probe's requests (ports_usage.csv, passed with --sat-usage) are counted too, so the cap is
# the account's 10,000 requests a month less a margin of 1,500 (the same numbers as sat/port_archive.py).
MONTHLY_REQUEST_CAP_BY_MONTH = {"2026-10": 8500}
PU_PER_MIN = 200.0               # CDSE limit 300 PU/min
REQ_PER_MIN = 200                # CDSE limit 300 requests/min
DEFAULT_MAX_PU = 3900.0
INTERPOLATION = "BILINEAR"
CHECK_CHIP_M = 5000
MATCH_M = 100.0
USAGE_FIELDS = ["run_utc", "requests", "processing_units", "pu_source", "scenes", "units", "probe_version", "note"]


# ---------------------------------------------------------------- areas and units

def load_areas(path: Path = AREAS_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def area_grid(area: dict, pixel_m: float) -> Grid:
    return Grid.for_site({**area, "pixel_m": pixel_m})


def units(area: dict, pixel_m: float, max_px: int = MAX_PX) -> list[dict]:
    """Split an area into equal units of at most ``max_px`` on each side (Process API limit)."""
    g = area_grid(area, pixel_m)
    ny, nx = math.ceil(g.height / max_px), math.ceil(g.width / max_px)
    out = []
    for j in range(ny):
        for i in range(nx):
            r0, r1 = round(j * g.height / ny), round((j + 1) * g.height / ny)
            c0, c1 = round(i * g.width / nx), round((i + 1) * g.width / nx)
            w0, n0 = g.to_lonlat(r0, c0)
            e1, s1 = g.to_lonlat(r1, c1)
            ug = Grid(round(float(w0), 6), round(float(s1), 6), round(float(e1), 6), round(float(n0), 6), c1 - c0, r1 - r0)
            out.append({"area": area["id"], "unit": f"{j}{i}", "grid": ug})
    return out


# ---------------------------------------------------------------- catalog and scene choice

def _rings(geom: dict) -> list[list]:
    if not geom:
        return []
    if geom.get("type") == "Polygon":
        return [geom["coordinates"][0]]
    if geom.get("type") == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


def _inside(x: np.ndarray, y: np.ndarray, ring) -> np.ndarray:
    ring = np.asarray(ring, float)
    xs, ys = ring[:, 0], ring[:, 1]
    res = np.zeros(x.shape, bool)
    for k in range(len(ring) - 1):
        x0, y0, x1, y1 = xs[k], ys[k], xs[k + 1], ys[k + 1]
        if y0 == y1:
            continue
        c = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        res ^= c & (x < xc)
    return res


def footprint_coverage(geoms: list[dict], grid: Grid, n: int = 21) -> float:
    """Share of a unit (n x n sample points) inside the union of the scene footprints.
    No geometry -> 1.0 (the fetched valid fraction decides later)."""
    rings = [r for g in geoms for r in _rings(g)]
    if not rings:
        return 1.0
    xs = np.linspace(grid.west, grid.east, n)
    ys = np.linspace(grid.south, grid.north, n)
    x, y = np.meshgrid(xs, ys)
    hit = np.zeros(x.shape, bool)
    for r in rings:
        hit |= _inside(x, y, r)
    return round(float(hit.mean()), 3)


def spread(items: list, n: int) -> list:
    """``n`` items evenly spaced over a time-ordered list (all when there are no more than n)."""
    if len(items) <= n:
        return list(items)
    idx = sorted({int(round(v)) for v in np.linspace(0, len(items) - 1, n)})
    return [items[i] for i in idx]


def choose_scenes(features: list[dict], grid: Grid, max_scenes: int, min_cov: float) -> tuple[list[dict], list[dict]]:
    """Acquisitions covering the unit, spread over time (newest first in the result)."""
    by_id = {f.get("id"): f for f in features}
    acqs = cdse.acquisitions(features)
    cands, rejected = [], []
    for a in acqs:
        cov = footprint_coverage([(by_id.get(i) or {}).get("geometry") for i in a["ids"]], grid)
        a["footprint_coverage"] = cov
        (cands if cov >= min_cov else rejected).append(a)
    chosen = spread(cands, max_scenes)
    chosen.sort(key=lambda a: a["time"], reverse=True)
    return chosen, rejected


# ---------------------------------------------------------------- usage and caps

def read_csv(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def month_totals(rows: list[dict], month: str) -> tuple[float, int]:
    seen, pu, req = set(), 0.0, 0
    for r in rows:
        key = (r.get("run_utc"), r.get("processing_units"), r.get("requests"))
        if not (r.get("run_utc") or "").startswith(month) or key in seen:
            continue  # the same row read from several branches counts once
        seen.add(key)
        pu += float(r.get("processing_units") or 0)
        req += int(float(r.get("requests") or 0))
    return pu, req


def caps_for(month: str) -> dict:
    return {"ocean_pu": OCEAN_MONTHLY_PU_CAP.get(month, OCEAN_DEFAULT_PU_CAP),
            "total_pu": TOTAL_MONTHLY_PU_GUARD.get(month, TOTAL_DEFAULT_PU_GUARD),
            "requests": MONTHLY_REQUEST_CAP_BY_MONTH.get(month, MONTHLY_REQUEST_CAP)}


def budget(month: str, sat_rows: list[dict], ocean_rows: list[dict], max_pu: float) -> dict:
    caps = caps_for(month)
    sat_pu, sat_req = month_totals(sat_rows, month)
    oc_pu, oc_req = month_totals(ocean_rows, month)
    left = min(caps["ocean_pu"] - oc_pu, caps["total_pu"] - sat_pu - oc_pu, max_pu)
    return {"month": month, **{f"cap_{k}": v for k, v in caps.items()}, "max_pu_run": max_pu,
            "sat_month_pu": round(sat_pu, 3), "sat_month_requests": sat_req,
            "ocean_month_pu": round(oc_pu, 3), "ocean_month_requests": oc_req,
            "pu_left": round(max(0.0, left), 3),
            "requests_left": max(0, caps["requests"] - sat_req - oc_req)}


def plan_requests(per_area: dict, check: list[dict], pu_left: float, req_left: int) -> tuple[list[dict], list[dict]]:
    """Priority order: the check chips, then round-robin over areas (1st scene of each, 2nd, ...).
    Cut at the first request that would pass a cap (the rest are deferred, nothing is reordered)."""
    queue = list(check)
    depth = max((len(v) for v in per_area.values()), default=0)
    for k in range(depth):
        for aid in per_area:
            if k < len(per_area[aid]):
                queue.append(per_area[aid][k])
    plan, deferred, pu, req = [], [], 0.0, 0
    for item in queue:
        if deferred or pu + item["est_pu"] > pu_left + 1e-9 or req + 1 > req_left:
            deferred.append(item)
            continue
        pu += item["est_pu"]
        req += 1
        plan.append(item)
    return plan, deferred


class Pacer:
    """Keep the PU and requests of the last 60 s under the per-minute limits (sleep before a request)."""

    def __init__(self, pu_per_min: float = PU_PER_MIN, req_per_min: int = REQ_PER_MIN,
                 clock=time.monotonic, sleep=time.sleep):
        self.pu_per_min, self.req_per_min = pu_per_min, req_per_min
        self.clock, self.sleep = clock, sleep
        self.events: deque = deque()
        self.waited = 0.0

    def _trim(self, now):
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def wait(self, pu: float) -> None:
        while True:
            now = self.clock()
            self._trim(now)
            used = sum(p for _t, p in self.events)
            if (used + pu <= self.pu_per_min and len(self.events) < self.req_per_min) or not self.events:
                return
            dt = max(0.5, 60.0 - (now - self.events[0][0]))
            self.waited += dt
            self.sleep(dt)

    def spent(self, pu: float) -> None:
        self.events.append((self.clock(), pu))


# ---------------------------------------------------------------- OSM masks

def overpass_query(grid: Grid, seamark_re: str) -> str:
    b = f"({grid.south},{grid.west},{grid.north},{grid.east})"
    return (f'[out:json][timeout:120];('
            f'way["natural"="coastline"]{b};'
            f'way["man_made"~"^(pier|breakwater|groyne|quay)$"]{b};'
            f'way["bridge"]["highway"]{b};way["bridge"]["railway"]{b};'
            f'nwr["man_made"="offshore_platform"]{b};'
            f'nwr["power"="generator"]["generator:source"="wind"]{b};'
            f'nwr["power"="platform"]{b};'
            f'node["seamark:type"~"{seamark_re}"]{b};'
            f');out geom;')


def fetch_overpass(query: str, opener=urllib.request.urlopen, sleep=time.sleep) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with opener(req, timeout=180) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 2:
                raise
            sleep(60)
    raise RuntimeError("unreachable")


def is_static(tags: dict, seamark_re: str) -> bool:
    return (tags.get("man_made") == "offshore_platform" or tags.get("power") == "platform"
            or (tags.get("power") == "generator" and tags.get("generator:source") == "wind")
            or bool(re.match(seamark_re, tags.get("seamark:type", ""))))


def masks(grid: Grid, osm: dict, seamark_re: str, static_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """(land bool, static bool dilated by ``static_m``, info) on a unit grid."""
    els = osm.get("elements", [])
    land_osm = {"elements": [e for e in els if e.get("type") == "way" and not is_static(e.get("tags", {}), seamark_re)]}
    mask, info = build_mask(grid, land_osm)
    land = mask != WATER
    static = np.zeros(land.shape, bool)
    n_static = 0
    for e in els:
        if not is_static(e.get("tags", {}), seamark_re):
            continue
        if e.get("type") == "node":
            r, c = grid.to_pixel(e["lon"], e["lat"])
            r, c = int(math.floor(r)), int(math.floor(c))
            if 0 <= r < grid.height and 0 <= c < grid.width:
                static[r, c] = True
                n_static += 1
        elif e.get("geometry"):
            rows, cols = grid.to_pixel([p["lon"] for p in e["geometry"]], [p["lat"] for p in e["geometry"]])
            before = static.sum()
            draw_line(static, rows, cols)
            if len(rows) > 3 and e["geometry"][0] == e["geometry"][-1]:
                fill_polygon(static, rows, cols)
            n_static += int(static.sum() > before)
        elif "center" in e or ("bounds" in e):
            pass
    py = np.mean(grid.pixel_m)
    if static.any():
        static = dilate(static, static_m / py)
    info.update(static_objects=n_static, static_fraction=round(float(static.mean()), 5),
                osm_base=(osm.get("osm3s") or {}).get("timestamp_osm_base"))
    return land, static, info


# ---------------------------------------------------------------- decoding and statistics

def decode(data: bytes):
    arr = png.decode(data).astype(np.float64)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise cdse.CDSEError(f"想定外の画像（形 {arr.shape}）")
    dm = arr[..., -1] > 0

    def db(ch):
        return np.where(dm & (ch > 0), ch / cdse.DB_SCALE - cdse.DB_OFFSET, np.nan)
    lia = np.where(dm, arr[..., 2] / cdse.LIA_SCALE, np.nan) if arr.shape[2] == 4 else None
    return db(arr[..., 0]), db(arr[..., 1]), lia


def enl(vv_db: np.ndarray, block: int = 25) -> dict:
    """Equivalent number of looks of linear VV, mean^2/var per block (fully valid blocks), median.
    Sea texture lowers it; the comparison between interpolations on the same chip is what counts."""
    x = 10 ** (vv_db / 10)
    h, w = x.shape
    vals = []
    for r in range(0, h - block + 1, block):
        for c in range(0, w - block + 1, block):
            b = x[r:r + block, c:c + block]
            if np.isfinite(b).all():
                m, v = b.mean(), b.var()
                if v > 0:
                    vals.append(m * m / v)
    if not vals:
        return {"enl_median": None, "blocks": 0}
    return {"enl_median": round(float(np.median(vals)), 3), "enl_p25": round(float(np.percentile(vals, 25)), 3),
            "enl_p75": round(float(np.percentile(vals, 75)), 3), "blocks": len(vals)}


def match(a: list[dict], b: list[dict], pixel_m: float, tol_m: float = MATCH_M) -> dict:
    """Greedy nearest match of candidate centres within ``tol_m``."""
    if not a or not b:
        return {"both": 0, "v1_only": len(a), "v2_only": len(b)}
    pa = np.array([[d["row"], d["col"]] for d in a]) * pixel_m
    pb = np.array([[d["row"], d["col"]] for d in b]) * pixel_m
    d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
    used_a, used_b, both = set(), set(), 0
    for k in np.argsort(d, axis=None):
        i, j = divmod(int(k), len(b))
        if d[i, j] > tol_m:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        both += 1
    return {"both": both, "v1_only": len(a) - both, "v2_only": len(b) - both}


def tile_exceedance_check(tiles: list[dict], dets_v2: list[dict], static: np.ndarray) -> dict:
    """Share of tiles whose measured exceedance is within 1/3..3x the nominal (design 5.3-3), over
    tiles with a model, no v2 candidate and no fixed object (AIS ships are removed in PR4)."""
    out = {}
    busy = set()
    for d in dets_v2:
        busy.add((int(d["row"]) // 128, int(d["col"]) // 128))
    for key, nominal in (("1e-03", 1e-3), ("1e-04", 1e-4)):
        n = ok = 0
        for t in tiles:
            if t.get("model") is None or (t["ty"], t["tx"]) in busy:
                continue
            sl = static[t["ty"] * 128:(t["ty"] + 1) * 128, t["tx"] * 128:(t["tx"] + 1) * 128]
            if sl.any():
                continue
            e = (t.get("exceed") or {}).get(key)
            if e is None:
                continue
            n += 1
            ok += nominal / 3 <= e <= nominal * 3
        out[key] = {"tiles": n, "within_1_3_to_3x": ok, "share": round(ok / n, 3) if n else None}
    return out


def compact_tiles(tt: list[dict]) -> list[list]:
    """[ty, tx, model, own_model, borrowed, z_t(1e-7), exceed 1e-3, exceed 1e-4, vv_median_db, sea_state]"""
    return [[t["ty"], t["tx"], t["model"], t["own_model"], t["borrowed"], (t.get("z_t") or {}).get("1e-07"),
             (t.get("exceed") or {}).get("1e-03"), (t.get("exceed") or {}).get("1e-04"),
             t.get("vv_median_db"), t.get("sea_state")] for t in tt]


# ---------------------------------------------------------------- one unit x scene

V1_KEYS = ("row", "col", "pixels", "length_m", "width_m", "length_class", "peak_vv_db", "contrast_vv_db", "vh_ratio_db")


def compare(vv, vh, lia, land, static, grid: Grid, acq: dict) -> tuple[dict, list[dict]]:
    pixel_m = grid.pixel_m
    px = float(np.mean(pixel_m))
    valid = np.isfinite(vv) & np.isfinite(vh)
    lat = (grid.south + grid.north) / 2
    heading = v2.platform_heading((acq.get("orbit_direction") or "").lower(), lat) if acq.get("orbit_direction") else None
    lia_med = float(np.nanmedian(lia)) if lia is not None and np.isfinite(lia).any() else None

    t0 = time.perf_counter()
    d1, i1 = v1.detect(vv, vh, land, pixel_m)
    s1 = round(time.perf_counter() - t0, 2)
    edge = dilate(~valid, 3) if (~valid).any() else np.zeros_like(valid)
    near = dilate(land, v2.DEFAULTS_DET2["coast_band_m"] / px) & ~land if land.any() else np.zeros_like(land)
    for d in d1:
        r, c = int(round(d["row"])), int(round(d["col"]))
        r, c = min(max(r, 0), grid.height - 1), min(max(c, 0), grid.width - 1)
        d["static"], d["edge"], d["near_coast"] = bool(static[r, c]), bool(edge[r, c]), bool(near[r, c])

    t0 = time.perf_counter()
    d2, i2 = v2.detect_v2(vv, vh, land, pixel_m, lia=lia_med if lia is None else lia, heading_deg=heading, static=static)
    s2 = round(time.perf_counter() - t0, 2)
    k2 = v2.kept(d2)
    a2 = [d for d in k2 if d["confidence"] == "A"]
    open_v1 = [d for d in d1 if not (d["static"] or d["near_coast"] or d["edge"])]
    open_v2 = [d for d in a2 if not (d["static"] or d["near_coast"] or d["edge"])]

    res = {
        "valid_fraction": round(float(valid.mean()), 4),
        "water_fraction": round(float((valid & ~land).mean()), 4),
        "incidence_median_deg": round(lia_med, 2) if lia_med is not None else None,
        "vv_median_db": round(float(np.nanmedian(vv)), 2) if valid.any() else None,
        "vh_median_db": round(float(np.nanmedian(vh)), 2) if valid.any() else None,
        "platform_heading_deg": round(heading, 2) if heading is not None else None,
        "v1": {"candidates": len(d1), "open_sea": len(open_v1),
               "static": sum(d["static"] for d in d1), "near_coast": sum(d["near_coast"] for d in d1),
               "edge": sum(d["edge"] for d in d1),
               "length_classes": _classes(d1), "seconds": s1,
               **{k: i1[k] for k in ("hit_pixels", "clusters", "dropped_small", "dropped_vh", "threshold_sigma")}},
        "v2": {"candidates": len(k2), "confidence_A": len(a2), "confidence_B": len(k2) - len(a2),
               "open_sea_A": len(open_v2), "large": sum(d["large"] for d in k2),
               "static": sum(d["static"] for d in k2), "near_coast": sum(d["near_coast"] for d in k2),
               "edge": sum(d["edge"] for d in k2), "length_classes": _classes(a2),
               "sea_state": _count(d["sea_state"] for d in k2), "seconds": s2,
               **{k: i2.get(k) for k in ("hit_pixels", "grow_pixels", "clusters", "dropped_small", "dropped_vh",
                                         "excluded", "model_counts", "model_counts_vh", "z_t_seed", "exceedance",
                                         "tiles", "tiles_borrowed", "tiles_untested", "ambiguity_check",
                                         "ambiguity_offset_m", "sidelobe_axes")},
               "exceedance_check": tile_exceedance_check(i2["tile_table"], k2, static),
               "tile_table_cols": ["ty", "tx", "model", "own_model", "borrowed", "z_t_1e-07", "exceed_1e-03",
                                   "exceed_1e-04", "vv_median_db", "sea_state"],
               "tile_table": compact_tiles(i2["tile_table"])},
        "match_v1_all_v2_A_100m": match(d1, a2, px),
        "match_open_sea_100m": match(open_v1, open_v2, px),
    }
    feats = []
    for d in d1:
        lon, lat_ = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
        props = {k: d.get(k) for k in V1_KEYS}
        props.update(detector="cfar-v1", static=d["static"], edge=d["edge"], near_coast=d["near_coast"],
                     acquired_utc=acq["acquired_utc"])
        feats.append(_point(lon, lat_, props))
    v2.add_geo(d2, grid, acq["acquired_utc"])
    for f in v2.to_features(d2):
        f["properties"]["detector"] = "cfar-v2"
        f["properties"] = {k: _jsonable(v) for k, v in f["properties"].items()}
        feats.append(f)
    return res, feats


def _point(lon, lat, props):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(lon), 6), round(float(lat), 6)]},
            "properties": {k: _jsonable(v) for k, v in props.items()}}


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else None
    return v


def _classes(dets):
    return _count(d["length_class"] for d in dets)


def _count(it):
    out = {}
    for k in it:
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------------- main

def stamp(s: str) -> str:
    return s.replace(":", "-")


def write_usage(path: Path, row: dict) -> None:
    rows = read_csv(path)
    rows.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=USAGE_FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in USAGE_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def run(args, client=None, overpass=None, now=None, pacer=None, sleep=time.sleep, log=None, clock=None) -> dict:
    log = log or (lambda m: print(m, file=sys.stderr))
    clock = clock or (lambda: datetime.now(timezone.utc))
    now = now or clock()
    month = now.strftime("%Y-%m")
    cfg = load_areas(args.areas)
    pixel_m = float(cfg["pixel_m"])
    client = client if client is not None else cdse.Client()
    overpass = overpass or fetch_overpass
    pacer = pacer or Pacer(sleep=sleep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sat_rows = [r for f in args.sat_usage for r in read_csv(Path(f))]
    ocean_rows = [r for f in [args.usage, *args.ocean_usage] for r in read_csv(Path(f))]
    bud = budget(month, sat_rows, ocean_rows, args.max_pu)
    summary = {"probe_version": PROBE_VERSION, "run_utc": cdse.iso(now), "areas_version": cfg["version"],
               "pixel_m": pixel_m, "interpolation": INTERPOLATION, "dry_run": bool(args.dry_run),
               "budget": bud, "areas": {}, "results": [], "errors": []}
    log(f"今月 {month}：sat {bud['sat_month_pu']} PU・外洋 {bud['ocean_month_pu']} PU 使用済み、"
        f"外洋の上限 {bud['cap_ocean_pu']}・合計のガード {bud['cap_total_pu']}・この実行の上限 {args.max_pu} → 使える {bud['pu_left']} PU")

    # ---- catalog and plan (no PU)
    if getattr(args, "lookback_days", None):
        cfg["lookback_days"] = args.lookback_days
    if getattr(args, "min_coverage", None) is not None:
        cfg["min_footprint_coverage"] = args.min_coverage
    summary["lookback_days"] = cfg["lookback_days"]
    summary["min_footprint_coverage"] = cfg["min_footprint_coverage"]
    start = now - timedelta(days=int(cfg["lookback_days"]))
    per_area: dict[str, list] = {}
    for area in cfg["areas"]:
        if area.get("enabled") is False:
            continue
        for u in units(area, pixel_m):
            g = u["grid"]
            try:
                feats = client.catalog(g.bbox, start, now)
            except cdse.CredentialError:
                raise
            except cdse.CDSEError as e:
                summary["errors"].append({"area": area["id"], "unit": u["unit"], "stage": "catalog", "error": str(e)[:300]})
                continue
            chosen, rejected = choose_scenes(feats, g, int(area.get("max_scenes", 1)), float(cfg["min_footprint_coverage"]))
            est = cdse.estimate_pu(g.width, g.height, 3)
            per_area.setdefault(area["id"], [])
            for a in chosen:
                per_area[area["id"]].append({"kind": "scene", "area": area["id"], "unit": u["unit"], "grid": g,
                                             "acq": a, "est_pu": est})
            summary["areas"].setdefault(area["id"], {"name": area["name"], "units": []})["units"].append({
                "unit": u["unit"], "bbox": g.bbox, "width": g.width, "height": g.height,
                "area_km2": round(g.area_km2(), 1), "acquisitions_in_catalog": len(chosen) + len(rejected),
                "chosen": [_acq_meta(a) for a in chosen],
                "rejected_low_coverage": len(rejected), "est_pu_per_scene": est,
                "est_pu": round(est * len(chosen), 3)})
    check = []
    first = next((v[0] for v in per_area.values() if v), None)
    if first is not None and not args.no_check:
        g = first["grid"]
        mlat, mlon = (g.north - g.south) / (g.height * g.pixel_m[0]), (g.east - g.west) / (g.width * g.pixel_m[1])
        cx, cy = (g.west + g.east) / 2, (g.south + g.north) / 2
        h = CHECK_CHIP_M / 2
        cb = [round(cx - h * mlon, 6), round(cy - h * mlat, 6), round(cx + h * mlon, 6), round(cy + h * mlat, 6)]
        for interp, pm in (("NEAREST", 20), ("BILINEAR", 20), ("NEAREST", 10)):
            n = int(CHECK_CHIP_M / pm)
            check.append({"kind": "check", "area": first["area"], "unit": first["unit"], "acq": first["acq"],
                          "bbox": cb, "px": n, "pixel_m": pm, "interp": interp, "est_pu": cdse.estimate_pu(n, n, 3)})
    plan, deferred = plan_requests(per_area, check, bud["pu_left"], bud["requests_left"] - 2 * len(cfg["areas"]))
    summary["plan"] = {"requests": len(plan), "est_pu": round(sum(p["est_pu"] for p in plan), 3),
                       "deferred": len(deferred), "deferred_est_pu": round(sum(p["est_pu"] for p in deferred), 3),
                       "per_area_est_pu": {aid: round(sum(p["est_pu"] for p in plan if p["area"] == aid and p["kind"] == "scene"), 3)
                                           for aid in per_area},
                       "per_area_scenes": {aid: sum(1 for p in plan if p["area"] == aid and p["kind"] == "scene") for aid in per_area},
                       "check_est_pu": round(sum(p["est_pu"] for p in plan if p["kind"] == "check"), 3)}
    log("計画: " + json.dumps(summary["plan"], ensure_ascii=False))
    if args.dry_run or not plan:
        _finish(args, summary, client, now, out, log)
        return summary

    # ---- OSM masks (one Overpass query per area that has a planned scene)
    seamark_re = cfg["osm_static"]["seamark_types"]
    mask_cache: dict = {}
    osm_cache: dict = {}
    for k, aid in enumerate(sorted({p["area"] for p in plan if p["kind"] == "scene"})):
        area = next(a for a in cfg["areas"] if a["id"] == aid)
        if k:
            sleep(10)
        try:
            osm_cache[aid] = overpass(overpass_query(area_grid(area, pixel_m), seamark_re))
        except Exception as e:  # land unknown: the area is skipped (no PU spent on it)
            summary["errors"].append({"area": aid, "stage": "overpass", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    summary["osm"] = {"source": "OpenStreetMap via Overpass API", "license": "(c) OpenStreetMap contributors, ODbL 1.0"}

    # ---- fetch and compare
    deadline = time.monotonic() + args.max_minutes * 60
    for item in plan:
        if clock().strftime("%Y-%m") != month:
            summary["errors"].append({"stage": "stop", "error": "月が替わったため以降の取得をしない"})
            break
        if time.monotonic() > deadline:
            summary["errors"].append({"stage": "stop", "error": f"{args.max_minutes} 分を過ぎたため以降の取得をしない"})
            break
        if item["kind"] == "scene" and item["area"] not in osm_cache:
            continue
        spent = client.processing_units  # from the x-processingunits-spent header (estimate if absent)
        if spent + item["est_pu"] > bud["pu_left"] + 1e-9:
            summary["errors"].append({"stage": "stop", "error": f"実際の PU {spent:.1f} に次の見込みを足すと上限に届くため止めた"})
            break
        a = item["acq"]
        pacer.wait(item["est_pu"])
        try:
            if item["kind"] == "check":
                data, pu = client.process(item["bbox"], item["px"], item["px"], a["t_from"], a["t_to"],
                                          with_lia=True, upsampling=item["interp"])
                pacer.spent(pu)
                vv, _vh, _lia = decode(data)
                summary.setdefault("interpolation_check", []).append({
                    "interp": item["interp"], "pixel_m": item["pixel_m"], "px": item["px"], "bbox": item["bbox"],
                    "acquired_utc": a["acquired_utc"], "pu": pu, "valid_fraction": round(float(np.isfinite(vv).mean()), 4),
                    **enl(vv, 25 if item["pixel_m"] == 20 else 50)})
                continue
            g = item["grid"]
            key = (item["area"], item["unit"])
            if key not in mask_cache:
                mask_cache[key] = masks(g, osm_cache[item["area"]], seamark_re, float(cfg["static_dilate_m"]))
            land, static, minfo = mask_cache[key]
            t0 = time.perf_counter()
            data, pu = client.process(g.bbox, g.width, g.height, a["t_from"], a["t_to"], with_lia=True,
                                      upsampling=INTERPOLATION)
            pacer.spent(pu)
            t_fetch = round(time.perf_counter() - t0, 1)
            vv, vh, lia = decode(data)
            del data
            row = {"area": item["area"], "unit": item["unit"], **_acq_meta(a), "pu": pu, "fetch_seconds": t_fetch,
                   "mask": {k: minfo.get(k) for k in ("water_fraction", "land_fraction", "static_objects",
                                                       "static_fraction", "coastline_ways", "osm_base")}}
            valid = float((np.isfinite(vv) & np.isfinite(vh)).mean())
            if valid < float(cfg["min_valid_fraction"]):
                row.update(valid_fraction=round(valid, 4), skipped="no data")
            else:
                res, feats = compare(vv, vh, lia, land, static, g, a)
                row.update(res)
                name = f"{item['area']}-{item['unit']}-{stamp(a['acquired_utc'])}.geojson"
                (out / name).write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                row["geojson"] = name
            summary["results"].append(row)
            log(_line(row))
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as e:
            summary["errors"].append({"area": item["area"], "acquired_utc": a["acquired_utc"], "stage": item["kind"],
                                      "error": str(e)[:300]})
            log(f"{item['area']} {a['acquired_utc']}: 失敗 {str(e)[:200]}")
        _write_summary(out, summary, client, pacer)  # after every scene: a timeout keeps what was done
    _finish(args, summary, client, now, out, log, pacer)
    return summary


def _acq_meta(a: dict) -> dict:
    return {"acquired_utc": a["acquired_utc"], "platform": a.get("platform"), "orbit_direction": a.get("orbit_direction"),
            "relative_orbit": a.get("relative_orbit"), "footprint_coverage": a.get("footprint_coverage")}


def _line(r: dict) -> str:
    if r.get("skipped"):
        return f"{r['area']} {r['acquired_utc']}: 無データ（有効 {r.get('valid_fraction')}）"
    return (f"{r['area']} {r['acquired_utc']} {r.get('orbit_direction')}: VV中央値 {r['vv_median_db']} dB、"
            f"v1 {r['v1']['candidates']}（外洋 {r['v1']['open_sea']}、{r['v1']['seconds']}s）、"
            f"v2 A {r['v2']['confidence_A']}・B {r['v2']['confidence_B']}（外洋A {r['v2']['open_sea_A']}、大型 {r['v2']['large']}、"
            f"{r['v2']['seconds']}s）、一致 {r['match_open_sea_100m']}、超過率 {r['v2']['exceedance']}、PU {r['pu']}")


def _write_summary(out: Path, summary: dict, client, pacer=None) -> None:
    summary["requests"] = client.requests
    summary["processing_units"] = round(client.processing_units, 3)
    summary["pu_source"] = "header" if getattr(client, "pu_reported", True) else "estimate"
    if pacer is not None:
        summary["pacer_wait_seconds"] = round(pacer.waited, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), default=_jsonable),
                                      encoding="utf-8")


def _finish(args, summary, client, now, out, log, pacer=None) -> None:
    _write_summary(out, summary, client, pacer)
    scenes = len(summary["results"])
    write_usage(Path(args.usage), {"run_utc": cdse.iso(now), "requests": client.requests,
                                   "processing_units": round(client.processing_units, 3),
                                   "pu_source": "header" if getattr(client, "pu_reported", True) else "estimate",
                                   "scenes": scenes, "units": scenes, "probe_version": PROBE_VERSION,
                                   "note": "dry-run" if args.dry_run else ""})
    (out / "report.txt").write_text(report(summary), encoding="utf-8")
    log(report(summary))


def report(s: dict) -> str:
    b = s["budget"]
    lines = [f"{s['probe_version']} {s['run_utc']} 補間 {s['interpolation']}{'（dry-run）' if s['dry_run'] else ''}",
             f"今月 {b['month']}：sat {b['sat_month_pu']} PU + 外洋 {b['ocean_month_pu']} PU（実行前）。上限 外洋 {b['cap_ocean_pu']}・"
             f"合計 {b['cap_total_pu']}・実行 {b['max_pu_run']} → 使える {b['pu_left']} PU",
             f"計画 {s['plan']['requests']} 件 見込み {s['plan']['est_pu']} PU（範囲別 {s['plan']['per_area_est_pu']}、"
             f"撮影数 {s['plan']['per_area_scenes']}、見送り {s['plan']['deferred']} 件 {s['plan']['deferred_est_pu']} PU）",
             f"実績 {s.get('requests')} リクエスト・{s.get('processing_units')} PU（{s.get('pu_source')}）"]
    for c in s.get("interpolation_check", []):
        lines.append(f"補間の確認 {c['interp']} {c['pixel_m']}m: ENL 中央値 {c.get('enl_median')}（{c.get('blocks')} ブロック）")
    for r in s["results"]:
        lines.append(_line(r))
    for e in s["errors"]:
        lines.append("エラー: " + json.dumps(e, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def run_mode(args, client=None, overpass=None, now=None, pacer=None, sleep=time.sleep, log=None, clock=None) -> dict:
    """``--mode raw`` (ocean/raw_cut.py) or ``--mode wide`` (ocean/wide.py), with this module's caps and pacing."""
    from ocean import raw_cut, wide
    log = log or (lambda m: print(m, file=sys.stderr))
    clock = clock or (lambda: datetime.now(timezone.utc))
    now = now or clock()
    client = client if client is not None else cdse.Client()
    overpass = overpass or fetch_overpass
    lim = min(getattr(args, "pu_per_min", PU_PER_MIN) or PU_PER_MIN, PU_PER_MIN)
    pacer = pacer or Pacer(lim, int(min(lim, REQ_PER_MIN)), sleep=sleep)
    fn = raw_cut.run_raw if args.mode == "raw" else wide.run_wide
    return fn(args, sys.modules[__name__], client, overpass, now, pacer, sleep, log, clock)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cfar-v1 / cfar-v2 on open-sea Sentinel-1 chips (probe)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--areas", type=Path, default=AREAS_FILE)
    ap.add_argument("--usage", type=Path, default=Path("docs/data/ocean/usage.csv"),
                    help="the ocean usage file this run appends to")
    ap.add_argument("--sat-usage", nargs="*", default=[], help="usage.csv of the daily Sentinel-1 run (main)")
    ap.add_argument("--ocean-usage", nargs="*", default=[], help="ocean usage.csv of other probe branches")
    ap.add_argument("--max-pu", type=float, default=DEFAULT_MAX_PU)
    ap.add_argument("--max-minutes", type=float, default=200)
    ap.add_argument("--lookback-days", type=int, help="override areas.json lookback_days")
    ap.add_argument("--min-coverage", type=float, help="override areas.json min_footprint_coverage")
    ap.add_argument("--dry-run", action="store_true", help="catalog and plan only (no PU)")
    ap.add_argument("--no-check", action="store_true", help="skip the interpolation check chips")
    ap.add_argument("--mode", choices=("compare", "raw", "wide"), default="compare",
                    help="compare: v1/v2 on the areas (default); raw: raw sigma0 tiles of selected scenes "
                         "(ocean/raw_cut.py); wide: 40 m wide-area snapshot (ocean/wide.py)")
    ap.add_argument("--raw-out", type=Path, help="raw mode: directory for the .npz tiles (probe branch only)")
    ap.add_argument("--targets", type=Path, help="raw mode: targets file (default ocean/raw_targets.json)")
    ap.add_argument("--pu-per-min", type=float, default=PU_PER_MIN, help="pacer limit (PU and requests per minute)")
    args = ap.parse_args(argv)
    try:
        if args.mode != "compare":
            s = run_mode(args)
        else:
            s = run(args, pacer=Pacer(min(args.pu_per_min, PU_PER_MIN), int(min(args.pu_per_min, REQ_PER_MIN))))
    except cdse.CredentialError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    return 1 if any(e.get("stage") not in ("stop",) for e in s["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
