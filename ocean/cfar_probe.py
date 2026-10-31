"""cfar-v2 PR3 probe: open-sea Sentinel-1 chips, cfar-v1 and cfar-v2 on the same arrays.

Experimental, probe only (``.github/workflows/ocean-cfar-probe.yml``, push to ``probe/ocean-cfar*``).
Nothing in the daily run (``sat/run.py``), the Zhanjiang counts or the public data uses this.

    python -m ocean.cfar_probe --out DIR --budget-month YYYY-MM --ocean-cap PU --global-cap PU --global-margin PU
                               [--usage docs/data/ocean/usage.csv] [--sat-usage FILE ...] [--other-usage FILE ...]
                               [--ports-usage FILE ...] [--ocean-usage FILE ...] [--max-pu PU] [--dry-run]

The caps have no defaults and are given with the month they were set for (``--budget-month``). In any
other UTC month nothing is sent to CDSE (not even the catalog) and the exit code is 2. Exit codes: 0 no
error, 1 errors of single requests (or an unexpected exception, with its traceback), 2 the month, 3 CDSE
refused the credentials (``exit_code``).

Usage record: the run's row in ``--usage`` is written after the catalog and after every request (rewritten
in place, one row per run), and once more however the run ends -- an exception of any kind (URLError,
TimeoutError, IncompleteRead, a PNG or detector error, CredentialError, a 403 when the quota is used up)
or SIGTERM from a cancelled / timed-out job -- before the exception is raised again (``guarded``).

Limits: ``cdse.Client._post_sh`` retries 429 and 5xx inside one call (up to 3 more sends, waits of up to
120 s each) without the month check of ``MonthClient`` and without the pacer: a call started just before
the end of the month can send a few minutes into the next month, and the per-minute rate and the request
count can be passed by up to 3 requests per call. The pacer keeps this run under 200 PU and 200 requests
a minute, but the port archive (ports-archive.yml, up to 3 jobs at 80 PU/min) may run at the same time
and together they can pass CDSE's 300 PU/min (429, retried as above); the request cap is not guaranteed
across jobs running at the same time either (each reads the others' usage only when it starts).

Steps
-----
1. Areas from ``ocean/areas.json`` (centre + side, 20 m pixels), split into units of at most
   2,500 x 2,500 px (the Process API limit).
2. Catalog (Sentinel Hub, no PU): IW scenes with VV+VH of the last ``lookback_days`` whose footprint
   covers at least ``min_footprint_coverage`` of the unit; per area up to ``max_scenes``, spread over
   the period (evenly spaced in time order, so that different sea states/winds come in).
3. Plan and caps, before any PU is spent (``budget``, the guard of ``sat.port_archive``): the
   estimated PU of every request (``cdse.estimate_pu``) must fit (a) ``--ocean-cap`` less this month's
   ocean usage (``docs/data/ocean/usage.csv`` of every branch), (b) ``--global-cap`` less
   ``--global-margin`` (kept for the daily Zhanjiang run) less this month's usage of everything else:
   main's ``docs/data/sat/usage.csv`` (``--sat-usage``), the port archive (``--ports-usage``) and any
   other usage CSV of any branch (``--other-usage``), (c) ``--max-pu`` for the run if given, (d) the
   account's requests (``port_archive.REQUEST_CAP`` less ``REQUEST_MARGIN``, all uses together).
   The usage copies of all branches are gathered with ``python -m sat.port_archive gather`` and summed
   with each record once (``port_archive.union_usage``). Requests are taken in priority order (the
   first scene of every area, then the second, ...) and the plan is cut where a cap would be passed.
   The ocean usage is kept out of ``docs/data/sat/usage.csv``: the daily run's own cap is not used up.
4. Rate: CDSE Sentinel Hub allows 300 requests and 300 PU per minute (HTTP 429 above). A pacer keeps
   the last 60 s under ``PU_PER_MIN`` (200) and ``REQ_PER_MIN`` (200) by waiting; ``cdse.Client``
   also waits and retries on 429. Months are UTC calendar months: before every request (catalog
   included, ``MonthClient``) and before every scene the UTC month is checked against
   ``--budget-month``; once it differs the run stops (the usage row is still written, exit code 2).
5. Fetch (Process API): VV, VH, local incidence angle, SIGMA0_ELLIPSOID, orthorectified, 20 m.
   ``downsampling`` is ignored for Sentinel-1 (Sentinel Hub docs, "Not used, use upsampling"):
   ``upsampling`` decides the interpolation "regardless of the resolution". The run uses BILINEAR
   and, once, measures the speckle (ENL) of a 5 km chip fetched with NEAREST and BILINEAR at 20 m
   and at 10 m (design 7.2 (a)). No data = dataMask 0 or VV code 0 (``cfar2-pr2.1``); a VH code 0
   with data is a VH sigma0 <= 0, not no data. The dataMask is passed to ``detect_v2`` (``datamask``).
   A unit with less than ``min_valid_fraction`` valid pixels is recorded and not detected.
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
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sat import cdse, png
from sat import port_archive as pa
from sat import detect as v1
from sat import detect2 as v2
from sat.grid import Grid, dilate, draw_line, fill_polygon
from sat.osm_mask import OVERPASS_URL, USER_AGENT, WATER, build_mask

PROBE_VERSION = "ocean-cfar-probe-1"
AREAS_FILE = Path(__file__).resolve().parent / "areas.json"
MAX_PX = 2500
# The monthly caps (ocean, the whole account, the margin kept for the daily run) have no defaults: they are
# required arguments, set with the month they are for (--budget-month), as in sat/port_archive.py.
# The ocean usage CSV as named by ``port_archive.gather_usage`` (other-<path with _>--<n>.csv).
OCEAN_USAGE_KEY = "other-docs_data_ocean_usage.csv"
PU_PER_MIN = 200.0               # CDSE limit 300 PU/min
REQ_PER_MIN = 200                # CDSE limit 300 requests/min
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


class MonthStop(pa.BudgetStop):
    """The UTC month is no longer ``--budget-month`` (raised before a request is sent)."""


class MonthClient(cdse.Client):
    """cdse.Client that sends nothing (catalog or process) once the UTC month is not ``budget_month``."""

    def __init__(self, budget_month: str, utcnow=None, **kw):
        super().__init__(**kw)
        self.budget_month = budget_month
        self.utcnow = utcnow or (lambda: datetime.now(timezone.utc))

    def _post_sh(self, url, payload, accept):
        month = self.utcnow().strftime("%Y-%m")
        if month != self.budget_month:
            raise MonthStop(f"上限は {self.budget_month} の値（今は {month}）。月が変わったため止める")
        return super()._post_sh(url, payload, accept)


def split_usage(args) -> tuple[list[Path], list[Path], list[Path]]:
    """(ocean, ports, other) usage CSVs. Ocean: this run's ``--usage``, ``--ocean-usage`` and the ocean copies
    among ``--other-usage`` (gather's other-docs_data_ocean_usage.csv--<n>.csv); ports: ``--ports-usage``;
    other: ``--sat-usage`` and the rest of ``--other-usage``."""
    other_in = [Path(f) for f in (getattr(args, "other_usage", None) or [])]
    ocean = [Path(args.usage), *[Path(f) for f in (getattr(args, "ocean_usage", None) or [])],
             *[f for f in other_in if pa.usage_key(f) == OCEAN_USAGE_KEY]]
    ports = [Path(f) for f in (getattr(args, "ports_usage", None) or [])]
    other = [Path(f) for f in (getattr(args, "sat_usage", None) or [])] + \
            [f for f in other_in if pa.usage_key(f) != OCEAN_USAGE_KEY]
    return ocean, ports, other


def budget(args, now: datetime) -> dict:
    """PU and requests left for the ocean probe in ``args.budget_month``, with ``sat.port_archive.guard``.

    Roles in the shared guard: the ocean is the job being capped (``cap`` = ``--ocean-cap``, its own usage
    = the ocean CSVs as one pool), the port archive's usage (all its copies as one pool, ``union_usage``)
    is passed as ``reserve`` and its requests are taken off the request cap, the rest (main's sat usage,
    other usage CSVs) as the other usage. So the ocean's left is
    min(ocean cap - ocean used, global cap - margin - sat/others - ports - ocean used, --max-pu).
    Raises MonthStop (exit code 2) if ``now`` is not in ``args.budget_month``."""
    month = args.budget_month
    ocean, ports, other = split_usage(args)
    ports_pu, ports_req = pa.union_usage(ports, month) if now.strftime("%Y-%m") == month else (0.0, 0)
    try:
        g = pa.guard(month, now, cap=float(args.ocean_cap), global_cap=float(args.global_cap),
                     global_margin=float(args.global_margin), ocean_cap=None, usage=other, ports_usage=ocean,
                     reserve=ports_pu, max_pu=getattr(args, "max_pu", None))
    except ValueError as e:  # guard: ``now`` is not in the budget month
        raise MonthStop(str(e)) from None
    return {"month": month, "cap_ocean_pu": float(args.ocean_cap), "cap_total_pu": float(args.global_cap),
            "global_margin": float(args.global_margin), "cap_requests": pa.REQUEST_CAP - pa.REQUEST_MARGIN,
            "max_pu_run": getattr(args, "max_pu", None),
            "sat_month_pu": round(g["other_pu"], 3), "sat_month_requests": g["other_req"],
            "ports_month_pu": round(ports_pu, 3), "ports_month_requests": ports_req,
            "ocean_month_pu": round(g["done_pu"], 3), "ocean_month_requests": g["done_req"],
            "pu_left": round(g["cap"], 3), "requests_left": max(0, g["req_cap"] - ports_req)}


def month_changed(args, clock) -> bool:
    return clock().strftime("%Y-%m") != args.budget_month


MONTH_STOP = "月が替わったため以降の取得をしない"


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
    """(vv dB, vh dB, lia, datamask). Code 0 is NaN in dB; datamask = Sentinel Hub dataMask (last band).
    No data is datamask False or VV NaN; a VH NaN with data is VH sigma0 <= 0 (``detect2``, cfar2-pr2.1)."""
    arr = png.decode(data).astype(np.float64)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise cdse.CDSEError(f"想定外の画像（形 {arr.shape}）")
    dm = arr[..., -1] > 0

    def db(ch):
        return np.where(dm & (ch > 0), ch / cdse.DB_SCALE - cdse.DB_OFFSET, np.nan)
    lia = np.where(dm, arr[..., 2] / cdse.LIA_SCALE, np.nan) if arr.shape[2] == 4 else None
    return db(arr[..., 0]), db(arr[..., 1]), lia, dm


def data_valid(vv: np.ndarray, datamask: np.ndarray | None = None) -> np.ndarray:
    """Pixels with data, as ``detect2.detect_v2``: datamask (if given) and VV finite. VH is not looked at."""
    valid = np.isfinite(vv)
    if datamask is not None:
        valid &= np.asarray(datamask, bool)
    return valid


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


def compare(vv, vh, lia, land, static, grid: Grid, acq: dict, datamask=None) -> tuple[dict, list[dict]]:
    pixel_m = grid.pixel_m
    px = float(np.mean(pixel_m))
    valid = data_valid(vv, datamask)
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
    d2, i2 = v2.detect_v2(vv, vh, land, pixel_m, lia=lia_med if lia is None else lia, heading_deg=heading, static=static,
                         datamask=datamask)
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
        "vh_median_db": _median_db(vh, valid),
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


def _median_db(x: np.ndarray, valid: np.ndarray):
    """Median dB over the valid pixels with a finite value; None (JSON null) when there is none (e.g. VH
    sigma0 <= 0 everywhere at very low wind: VH code 0 is data but has no dB value)."""
    v = x[valid & np.isfinite(x)]
    return round(float(np.median(v)), 2) if v.size else None


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


def write_usage(path: Path, row: dict, replace: bool = False) -> None:
    """Add ``row`` to the usage CSV. ``replace``: the last row with the same run_utc and probe_version is
    replaced instead (the running record of one run, see ``UsageRecorder``). Written atomically."""
    path = Path(path)
    rows = read_csv(path)
    key = (row.get("run_utc"), row.get("probe_version"))
    k = next((i for i in range(len(rows) - 1, -1, -1)
              if (rows[i].get("run_utc"), rows[i].get("probe_version")) == key), None) if replace else None
    if k is None:
        rows.append(row)
    else:
        rows[k] = row
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=USAGE_FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in USAGE_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    os.replace(tmp, path)


class UsageRecorder:
    """The usage row of one run, written as the run goes (after the catalog and after every request) and
    once more however the run ends (normal end, an exception of any kind, SIGTERM): the PU spent are in
    ``--usage`` even if the job is cut off by its timeout. One row per run (rewritten in place, key
    run_utc + probe_version), so the sum over the branches (``port_archive.union_usage``) counts it once."""

    def __init__(self, path: Path, now: datetime, client, version: str, note: str, counts=None):
        self.path, self.now, self.client, self.version, self.note = Path(path), now, client, version, note
        self.counts = counts or (lambda s: (len(s["results"]), len(s["results"])))

    def write(self, summary: dict | None, note: str | None = None) -> None:
        scenes, units = self.counts(summary) if summary else (0, 0)
        write_usage(self.path, {"run_utc": cdse.iso(self.now), "requests": self.client.requests,
                                "processing_units": round(self.client.processing_units, 3),
                                "pu_source": "header" if getattr(self.client, "pu_reported", True) else "estimate",
                                "scenes": scenes, "units": units, "probe_version": self.version,
                                "note": self.note if note is None else note}, replace=True)


def guarded(rec: UsageRecorder, ctx: dict, out: Path, client, pacer, log, body):
    """Run ``body()``; whatever it raises (URLError, TimeoutError, IncompleteRead, a PNG or detector error,
    CredentialError, SystemExit from SIGTERM, ...) the usage row is written first, then summary.json and
    report.txt as far as they go, and the exception is raised again."""
    try:
        return body()
    except BaseException as e:
        s = ctx.get("summary")
        why = f"{type(e).__name__}: {str(e)[:200]}"
        try:
            rec.write(s, note=f"aborted ({type(e).__name__})")
        except Exception as e2:  # pragma: no cover - the disk itself failed
            log(f"usage を書けなかった: {e2}")
        if s is not None:
            try:
                s["errors"].append({"stage": "abort", "error": why})
                _write_summary(out, s, client, pacer)
                if ctx.get("report"):
                    (out / "report.txt").write_text(ctx["report"](s), encoding="utf-8")
            except Exception:
                pass
        log(f"中断: {why}（使った {client.requests} リクエスト・{round(client.processing_units, 3)} PU は usage に記録）")
        raise


def attribution(times, now: datetime | None = None, osm: bool = True) -> str:
    """Credit with the years of the acquisitions (``times``: acquired_utc strings or datetimes), e.g.
    "Contains modified Copernicus Sentinel data 2026" or "... 2025–2026" across a year; the year of
    ``now`` when there is no acquisition. ``osm``: add the OpenStreetMap credit (land, fixed objects)."""
    years = sorted({(cdse.parse_time(t) if isinstance(t, str) else t).year for t in times if t})
    if not years and now is not None:
        years = [now.year]
    y = "" if not years else str(years[0]) if years[0] == years[-1] else f"{years[0]}–{years[-1]}"
    text = f"Contains modified Copernicus Sentinel data {y}".rstrip()
    return text + ("; " + OSM_CREDIT if osm else "")


OSM_CREDIT = "land and fixed objects © OpenStreetMap contributors, ODbL 1.0"


def clean(v):
    """JSON-safe copy: numpy scalars to Python, NaN/Inf to None (no NaN in any JSON written here)."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):  # as ``_jsonable`` (4 decimals)
        return round(float(v), 4) if math.isfinite(v) else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return v


def dumps(obj, **kw) -> str:
    """json.dumps of ``clean(obj)`` with ``allow_nan=False`` (fails rather than writing NaN)."""
    kw.setdefault("ensure_ascii", False)
    kw.setdefault("separators", (",", ":"))
    return json.dumps(clean(obj), allow_nan=False, default=_jsonable, **kw)


def run(args, client=None, overpass=None, now=None, pacer=None, sleep=time.sleep, log=None, clock=None) -> dict:
    log = log or (lambda m: print(m, file=sys.stderr))
    clock = clock or (lambda: datetime.now(timezone.utc))
    now = now or clock()
    month = args.budget_month
    bud = budget(args, now)  # MonthStop outside the budget month, before anything is sent
    cfg = load_areas(args.areas)
    pixel_m = float(cfg["pixel_m"])
    client = client if client is not None else MonthClient(month, utcnow=clock)
    overpass = overpass or fetch_overpass
    pacer = pacer or Pacer(sleep=sleep)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"probe_version": PROBE_VERSION, "run_utc": cdse.iso(now), "areas_version": cfg["version"],
               "pixel_m": pixel_m, "interpolation": INTERPOLATION, "dry_run": bool(args.dry_run),
               "attribution": attribution([], now), "budget": bud, "areas": {}, "results": [], "errors": []}
    log(f"今月 {month}：sat {bud['sat_month_pu']} PU・外洋 {bud['ocean_month_pu']} PU 使用済み、"
        f"港 {bud['ports_month_pu']} PU、外洋の上限 {bud['cap_ocean_pu']}・合計 {bud['cap_total_pu']}（余裕 {bud['global_margin']}）・"
        f"この実行の上限 {bud['max_pu_run']} → 使える {bud['pu_left']} PU")
    rec = UsageRecorder(Path(args.usage), now, client, PROBE_VERSION, "dry-run" if args.dry_run else "")
    ctx = {"summary": summary, "report": report}
    return guarded(rec, ctx, out, client, pacer, log,
                   lambda: _run_compare(args, cfg, pixel_m, client, overpass, now, pacer, sleep, log, clock,
                                        bud, out, summary, rec))


def _run_compare(args, cfg, pixel_m, client, overpass, now, pacer, sleep, log, clock, bud, out, summary, rec) -> dict:
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
            if month_changed(args, clock):
                return _month_stop(args, summary, client, now, out, log, MONTH_STOP, rec)
            try:
                feats = client.catalog(g.bbox, start, now)
            except MonthStop as e:
                return _month_stop(args, summary, client, now, out, log, str(e), rec)
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
    summary["attribution"] = attribution([p["acq"]["acquired_utc"] for p in plan], now)
    rec.write(summary)  # the catalog requests
    if args.dry_run or not plan:
        _finish(args, summary, client, now, out, log, rec=rec)
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
    summary["osm"] = {"source": "OpenStreetMap via Overpass API", "license": "© OpenStreetMap contributors, ODbL 1.0"}

    # ---- fetch and compare
    deadline = time.monotonic() + args.max_minutes * 60
    for item in plan:
        if month_changed(args, clock):
            summary["errors"].append({"stage": "stop", "reason": "month", "error": MONTH_STOP})
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
                vv, _vh, _lia, dm = decode(data)
                summary.setdefault("interpolation_check", []).append({
                    "interp": item["interp"], "pixel_m": item["pixel_m"], "px": item["px"], "bbox": item["bbox"],
                    "acquired_utc": a["acquired_utc"], "pu": pu, "valid_fraction": round(float(data_valid(vv, dm).mean()), 4),
                    **enl(vv, 25 if item["pixel_m"] == 20 else 50)})
                _write_summary(out, summary, client, pacer)
                rec.write(summary)
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
            vv, vh, lia, dm = decode(data)
            del data
            row = {"area": item["area"], "unit": item["unit"], **_acq_meta(a), "pu": pu, "fetch_seconds": t_fetch,
                   "mask": {k: minfo.get(k) for k in ("water_fraction", "land_fraction", "static_objects",
                                                       "static_fraction", "coastline_ways", "osm_base")}}
            valid = float(data_valid(vv, dm).mean())
            if valid < float(cfg["min_valid_fraction"]):
                row.update(valid_fraction=round(valid, 4), skipped="no data")
            else:
                res, feats = compare(vv, vh, lia, land, static, g, a, datamask=dm)
                row.update(res)
                name = f"{item['area']}-{item['unit']}-{stamp(a['acquired_utc'])}.geojson"
                (out / name).write_text(dumps({"type": "FeatureCollection",
                                               "attribution": attribution([a["acquired_utc"]], now),
                                               "features": feats}), encoding="utf-8")
                row["geojson"] = name
            summary["results"].append(row)
            log(_line(row))
        except MonthStop as e:
            summary["errors"].append({"stage": "stop", "reason": "month", "error": str(e)})
            break
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as e:
            summary["errors"].append({"area": item["area"], "acquired_utc": a["acquired_utc"], "stage": item["kind"],
                                      "error": str(e)[:300]})
            log(f"{item['area']} {a['acquired_utc']}: 失敗 {str(e)[:200]}")
        _write_summary(out, summary, client, pacer)  # after every scene: a timeout keeps what was done
        rec.write(summary)                            # and the PU spent so far are in --usage
    _finish(args, summary, client, now, out, log, pacer, rec=rec)
    return summary


def _month_stop(args, summary, client, now, out, log, why: str, rec=None) -> dict:
    """Stop during the catalog (no PU spent): the usage row (requests) is still written."""
    summary["errors"].append({"stage": "stop", "reason": "month", "error": why})
    summary["plan"] = {"requests": 0, "est_pu": 0.0, "deferred": 0, "deferred_est_pu": 0.0,
                       "per_area_est_pu": {}, "per_area_scenes": {}, "check_est_pu": 0.0}
    _finish(args, summary, client, now, out, log, rec=rec)
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
    (out / "summary.json").write_text(dumps(summary), encoding="utf-8")


def _finish(args, summary, client, now, out, log, pacer=None, rec=None) -> None:
    _write_summary(out, summary, client, pacer)
    rec = rec or UsageRecorder(Path(args.usage), now, client, PROBE_VERSION, "dry-run" if args.dry_run else "")
    rec.write(summary)
    (out / "report.txt").write_text(report(summary), encoding="utf-8")
    log(report(summary))


def report(s: dict) -> str:
    b = s["budget"]
    pl = {"requests": 0, "est_pu": 0.0, "per_area_est_pu": {}, "per_area_scenes": {}, "deferred": 0,
          "deferred_est_pu": 0.0, **(s.get("plan") or {})}
    s = {**s, "plan": pl}
    lines = [f"{s['probe_version']} {s['run_utc']} 補間 {s['interpolation']}{'（dry-run）' if s['dry_run'] else ''}",
             f"出典: {s.get('attribution')}",
             f"今月 {b['month']}：sat 等 {b['sat_month_pu']} PU + 港 {b['ports_month_pu']} PU + 外洋 {b['ocean_month_pu']} PU（実行前）。"
             f"上限 外洋 {b['cap_ocean_pu']}・合計 {b['cap_total_pu']}（余裕 {b['global_margin']}）・実行 {b['max_pu_run']} → 使える {b['pu_left']} PU",
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
    budget(args, now)  # MonthStop outside the budget month, before anything is sent
    client = client if client is not None else MonthClient(args.budget_month, utcnow=clock)
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
    ap.add_argument("--budget-month", required=True,
                    help="YYYY-MM the caps were set for; in any other month (UTC) nothing is requested (exit 2)")
    ap.add_argument("--ocean-cap", type=float, required=True, help="the ocean's own cap of this month (PU)")
    ap.add_argument("--global-cap", type=float, required=True,
                    help="the month's own cap of all CDSE use (sat + ports + ocean + others, PU)")
    ap.add_argument("--global-margin", type=float, required=True,
                    help="PU kept free under --global-cap for the daily Zhanjiang run to the end of the month")
    ap.add_argument("--sat-usage", nargs="*", default=[], help="usage.csv of the daily Sentinel-1 run (main)")
    ap.add_argument("--other-usage", nargs="*", default=[],
                    help="other usage CSVs of every branch (sat.port_archive gather --other; ocean copies are told apart)")
    ap.add_argument("--ports-usage", nargs="*", default=[],
                    help="the port archive's usage CSVs of every branch (sat.port_archive gather --ports)")
    ap.add_argument("--ocean-usage", nargs="*", default=[], help="more ocean usage CSVs (summed with the ocean pool)")
    ap.add_argument("--max-pu", type=float, help="cap of this run (inside the month's caps); none if omitted")
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
    ap.add_argument("--extend", action="store_true", help="wide mode: fill holes and take second looks after earlier wide runs")
    ap.add_argument("--pu-per-min", type=float, default=PU_PER_MIN, help="pacer limit (PU and requests per minute)")
    args = ap.parse_args(argv)
    now_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if args.budget_month != now_month:
        print(f"上限は {args.budget_month} の値（今は {now_month}）。その月の上限（--budget-month・--ocean-cap・"
              f"--global-cap・--global-margin）を書き直すまで取得しない", file=sys.stderr)
        return EXIT_MONTH
    # A cancelled or timed-out job ends with SIGTERM: raise SystemExit so that the usage row is written
    # (``guarded``) before the process ends.
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.mode != "compare":
            s = run_mode(args)
        else:
            s = run(args, pacer=Pacer(min(args.pu_per_min, PU_PER_MIN), int(min(args.pu_per_min, REQ_PER_MIN))))
    except MonthStop as e:  # the month changed between the check above and budget(): nothing was sent
        print(f"停止: {e}", file=sys.stderr)
        return EXIT_MONTH
    except cdse.CredentialError as e:  # the usage row is already written (guarded)
        print(f"エラー（認証）: {e}", file=sys.stderr)
        return EXIT_CREDENTIAL
    return exit_code(s)


EXIT_ERRORS, EXIT_MONTH, EXIT_CREDENTIAL = 1, 2, 3


def _on_sigterm(signum, frame):
    raise SystemExit(128 + signum)


def exit_code(s: dict) -> int:
    """Exit code of ``main``: 0 no error; 1 errors of single requests (catalog, fetch, Overpass) recorded in
    the summary; 2 (``EXIT_MONTH``) the UTC month is not / is no longer ``--budget-month`` (nothing more was
    sent); 3 (``EXIT_CREDENTIAL``, from ``main``) CDSE refused the credentials. Any other exception ends the
    run with Python's traceback (exit 1) after the usage row is written."""
    if any(e.get("reason") == "month" for e in s["errors"]):
        return EXIT_MONTH
    return EXIT_ERRORS if any(e.get("stage") not in ("stop",) for e in s["errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
