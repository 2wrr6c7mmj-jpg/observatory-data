"""Open-sea monitoring with Sentinel-1 (ocean-monitor-1): large-ship candidates not yet matched with AIS.

    python -m ocean.monitor run  --out docs/data/ocean/monitor --usage docs/data/ocean/usage.csv \
        --budget-month YYYY-MM --ocean-cap N --global-cap N --global-margin N --max-pu 400 [--dry-run] ...
    python -m ocean.monitor plan --month 2026-10 --out DIR      # catalogue only (no PU, no Sentinel Hub request)

Design: 2部 ocean-monitoring-design (2026-10-31), HQ chose plan B: the key areas (Senkaku, Yonaguni,
Miyako Strait, east of Taiwan, south of Sakishima, Taiwan Strait) at every pass, Okinawa-Amami once a month,
40 m, in ``ocean/monitor.json``.

``run`` (daily, ``.github/workflows/ocean-monitor.yml``):

1. New passes: the CDSE OData catalogue (catalogue.dataspace.copernicus.eu: no login, no processing units,
   no Sentinel Hub request) over the enabled areas for the last ``lookback_days`` (3); IW GRDH 1SDV
   products grouped into acquisitions (``sat.cdse.acquisitions``). Passes already in ``passes.json`` are
   skipped (a pass that hit a cap is tried again while it is inside the window).
2. Per area (priority order) and pass (time order): share of the area's sea inside the footprint on a 0.1
   degree cell grid. ``every_pass`` areas take a pass covering at least ``min_area_share``; areas with
   ``min_revisit_days`` skip a pass within that many days of one already processed. ``monthly`` areas take
   only the cells not yet seen this month. Rectangles of whole cells (``ocean.wide.greedy_plan``), at most
   2,500 px a side, within the area's month cap (``pu_month_cap``) and the run's cap (``--max-pu`` and the
   month's caps, ``ocean.cfar_probe.budget``: the same guard as the ocean probe and the port archive).
3. Per rectangle: land and fixed objects from OpenStreetMap (one Overpass query per area, kept in
   ``osm/<AREA>.json`` and renewed after ``osm_refresh_days``), the Process API at 40 m (VV, VH, LIA,
   BILINEAR), cfar-v1 as the detector of record and cfar-v2 for the match (``ocean.wide.detect_rect``).
4. Flags: sea state per cell (VV median outside [low_db, high_db] -> not counted and not seen), azimuth
   ambiguity (a pixel >= 15 dB brighter at the ambiguity offset along the platform heading, land included),
   fishing clusters (>= 5 within 500 m), learned fixed objects (``static_learned.json``: >= 3 of the last
   5 looks of the same satellite and track within 200 m).
5. Classes: A = cfar-v1 high and cfar-v2 A within 100 m, not coast/fixed/edge/ambiguity/sea state/cluster
   (headline; "低〜中" until the 10 m chip check, M3); B = the rest of v1 medium/high and v2-only
   (published, not counted); reference = v1 low, coast band, fixed objects, sea state (counts only).
6. Output in ``--out`` (docs/data/ocean/monitor): ``meta.json``, ``areas.geojson``, ``passes.json``,
   ``candidates/YYYY-MM-DD.geojson``, ``coverage/YYYY-MM-DD.geojson``, ``series.csv``, ``series.json``,
   ``static_learned.json``. Counts are null (not seen) when less than ``count_min_share`` of the area was
   seen; 0 only when that much was seen and nothing was found. No sigma0 is written.

PU spent are added to ``--usage`` (docs/data/ocean/usage.csv) as the run goes and however it ends
(``ocean.cfar_probe.UsageRecorder``/``guarded``). Exit codes as ``ocean.cfar_probe``: 2 month, 3 credentials.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import signal
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ocean import cfar_probe as cp
from ocean import coverage_probe as covp
from ocean import wide
from sat import cdse
from sat import detect2 as v2
from sat.grid import Grid, m_per_deg

VERSION = "ocean-monitor-1"
CONFIG_FILE = Path(__file__).resolve().parent / "monitor.json"
PAGE = 1000
WINDOW_DAYS = 5
DONE = ("ok", "partial", "no_data", "skipped_revisit")
# coarse land for planning only (lon, lat); detection uses the OSM mask
LAND_POLYS = {
    **wide.LAND_POLYS,
    "tw_west": [(120.45, 22.85), (120.13, 22.9), (120.1, 23.2), (120.12, 23.5), (120.16, 23.8), (120.3, 23.95),
                (120.45, 24.1)],
}
SERIES_FIELDS = ["area", "acquired_utc", "date", "platform", "relative_orbit", "orbit_direction", "status",
                 "area_sea_km2", "observed_sea_km2", "observed_share", "count_A", "count_A_ge120_confirmed",
                 "count_A_ge120_unconfirmed", "count_B", "clusters", "density_A_per_1000km2",
                 "sea_state_excluded_km2", "vv_median_db", "pu", "source"]
CLASS_LABEL = {"A": "A（確度：低〜中、10 m の切り出しで未確認）", "B": "B（確度：低）"}
LIMITS = [
    "S1 の分解能は約 20 m。40 m の取得では 80 m 未満の船は数えない（見逃す）。艦種・国籍・軍民の区別は S1 だけではできない。",
    "見えるのは撮影の瞬間（昇交 18:20–19:32 JST・降交 06:17–06:51 JST）だけ。域ごとに月 2〜5 回。撮影の間の動きは分からない。",
    "動く船は方位方向に最大数百 m ずれて写る（視線速度 5 m/s で約 600 m）。向きは 180° の区別がつかない。",
    "撮られていない海（監視域の外、撮影の穴）は「いない」ではなく「見ていない」。",
    "検出の精度は香港の AIS 照合（未実施）で測るまで未検証。40 m の長さは 40 m 刻みで長めに出るので目安にしない。",
    "AIS は未照合。照合後も外洋（尖閣・宮古・先島南方）は陸上局の受信が届かず「AIS 未確認」としか言えない。",
]


# ---------------------------------------------------------------- config and areas

def load_config(path: Path = CONFIG_FILE) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [a["id"] for a in cfg["areas"]]
    if len(set(ids)) != len(ids):
        raise ValueError("monitor.json: 域の id が重複")
    for a in cfg["areas"]:
        w, s, e, n = a["bbox"]
        if not (w < e and s < n):
            raise ValueError(f"{a['id']}: bbox は [西, 南, 東, 北]")
        if a["mode"] not in ("every_pass", "monthly"):
            raise ValueError(f"{a['id']}: mode は every_pass か monthly")
    return cfg


def areas_in_order(cfg: dict, include_disabled: bool = False) -> list[dict]:
    """Areas by priority (then file order); the enabled ones only unless ``include_disabled``."""
    order = {a["id"]: k for k, a in enumerate(cfg["areas"])}
    return sorted([a for a in cfg["areas"] if include_disabled or a.get("enabled")],
                  key=lambda a: (a["priority"], order[a["id"]]))


class AreaCells(wide.Cells):
    """0.1 deg cells of one area. ``others``: bboxes of the areas before it (their sea is not this area's)."""

    def __init__(self, area: dict, others=(), cell: float = 0.1, k: int = 4):
        self.area_id = area["id"]
        self.w, self.s, self.e, self.n = area["bbox"]
        self.cell = cell
        self.nx = int(round((self.e - self.w) / cell))
        self.ny = int(round((self.n - self.s) / cell))
        off = (np.arange(k) + 0.5) / k * cell
        cx = self.w + np.arange(self.nx) * cell
        cy = self.s + np.arange(self.ny) * cell
        self.sx = (cx[None, :, None] + np.tile(off, k)[None, None, :]) * np.ones((self.ny, 1, 1))
        self.sy = (cy[:, None, None] + np.repeat(off, k)[None, None, :]) * np.ones((1, self.nx, 1))
        land = np.zeros(self.sx.shape, bool)
        for ring in LAND_POLYS.values():
            land |= wide._inside_ring(self.sx, self.sy, ring)
        other = np.zeros(self.sx.shape, bool)
        for w, s, e, n in others:
            other |= (self.sx >= w) & (self.sx < e) & (self.sy >= s) & (self.sy < n)
        self.sea_all = 1.0 - land.mean(axis=2)
        self.own = 1.0 - other.mean(axis=2)
        self.sea = self.sea_all * self.own
        self.frac = {self.area_id: self.own}
        self.region = np.full(self.sea.shape, self.area_id, dtype=object)
        self.km2 = np.array([self.area_km2(j) for j in range(self.ny)])[:, None] * np.ones((1, self.nx))

    def sea_km2(self, weight=None) -> float:
        w = 1.0 if weight is None else weight
        return float((self.km2 * self.sea * w).sum())

    def share(self, weight) -> float:
        tot = self.sea_km2()
        return self.sea_km2(np.minimum(weight, 1.0)) / tot if tot else 0.0

    def cell_of(self, lon: float, lat: float):
        i, j = int(math.floor((lon - self.w) / self.cell)), int(math.floor((lat - self.s) / self.cell))
        return (j, i) if 0 <= i < self.nx and 0 <= j < self.ny else None

    def contains(self, lon: float, lat: float) -> bool:
        c = self.cell_of(lon, lat)
        return c is not None and self.own[c] > 0


def area_cells(cfg: dict, areas: list[dict]) -> dict:
    """AreaCells of each area; the sea of an earlier (higher priority) area is not counted again."""
    out, before = {}, []
    for a in areas:
        out[a["id"]] = AreaCells(a, before, cell=float(cfg.get("cell_deg", 0.1)))
        before.append(tuple(a["bbox"]))
    return out


def union_bbox(areas: list[dict]) -> list[float]:
    b = [a["bbox"] for a in areas]
    return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]


# ---------------------------------------------------------------- OData catalogue (no PU, no login)

def odata_url(bbox: list[float], t0: datetime, t1: datetime, skip: int = 0) -> str:
    flt = (f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;{covp.wkt_box(bbox)}') "
           f"and ContentDate/Start ge {covp.iso_ms(t0)} and ContentDate/Start lt {covp.iso_ms(t1)} "
           f"and contains(Name,'_IW_GRDH_1SDV_')")
    q = {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": str(PAGE)}
    if skip:
        q["$skip"] = str(skip)
    return covp.ODATA_URL + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def fetch_odata(bbox: list[float], start: datetime, end: datetime, get=covp.http_json, log=print,
                sleep=time.sleep) -> tuple[list[dict], int]:
    """IW GRDH dual-pol products starting in [start, end) over ``bbox``. Returns (products, OData requests)."""
    out, seen, n = [], set(), 0
    t0 = start
    while t0 < end:
        t1 = min(end, t0 + timedelta(days=WINDOW_DAYS))
        url, skip = odata_url(bbox, t0, t1), 0
        while url:
            page = get(url)
            n += 1
            items = page.get("value", [])
            for p in items:
                if p.get("Id") not in seen:
                    seen.add(p.get("Id"))
                    out.append(p)
            if page.get("@odata.nextLink"):
                url = page["@odata.nextLink"]
            elif len(items) >= PAGE:
                skip += PAGE
                url = odata_url(bbox, t0, t1, skip)
            else:
                url = None
            if url:
                sleep(0.5)
        t0 = t1
    log(f"OData: {start:%Y-%m-%d %H:%M}〜{end:%Y-%m-%d %H:%M} UTC の IW GRDH 1SDV {len(out)} 件（{n} リクエスト）")
    return out, n


def to_features(products: list[dict]) -> list[dict]:
    """OData products as Sentinel Hub catalogue features (for ``cdse.acquisitions``); the COG copy of a
    product (same satellite and start) is dropped."""
    feats, seen = [], set()
    for p in products:
        info = covp.product_info(p)
        if not info or info["mode"] != "IW" or info["kind"] != "GRD" or not info["start"] or not info["rings"]:
            continue
        key = (info["platform"], info["start"])
        if key in seen:
            continue
        seen.add(key)
        rel = info["relative_orbit"]
        feats.append({"id": info["name"], "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in info["rings"]]},
                      "properties": {"datetime": cdse.iso(info["start"]), "platform": "sentinel-" + info["platform"][1:].lower(),
                                     "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"],
                                     "sat:orbit_state": info["direction"], "sat:absolute_orbit": info["absolute_orbit"],
                                     "sat:relative_orbit": rel,
                                     "published": cdse.iso(info["published"]) if info["published"] else None}})
    return feats


def catalogue(bbox, start, end, get=covp.http_json, log=print, sleep=time.sleep) -> tuple[list[dict], dict, int]:
    """(acquisitions, features by id, OData requests)."""
    prods, n = fetch_odata(bbox, start, end, get=get, log=log, sleep=sleep)
    feats = to_features(prods)
    by_id = {f["id"]: f for f in feats}
    acqs = cdse.acquisitions(feats)
    for a in acqs:
        pubs = [by_id[i]["properties"].get("published") for i in a["ids"]]
        pubs = [x for x in pubs if x]
        a["published_utc"] = max(pubs) if pubs else None
    return acqs, by_id, n


def footprints(cells: AreaCells, acqs: list[dict], by_id: dict) -> list[np.ndarray]:
    return [cells.footprint_share([(by_id.get(i) or {}).get("geometry") for i in a["ids"]]) for a in acqs]


# ---------------------------------------------------------------- decisions shared by run and plan

def pass_key(acquired_utc: str, area_id: str) -> str:
    return f"{acquired_utc}|{area_id}"


def rects_cov(cells: AreaCells, rows: list[dict]) -> np.ndarray:
    """Cells seen (observed share, max) by the rectangles of earlier passes."""
    cov = np.zeros((cells.ny, cells.nx))
    for r in rows:
        for o in r.get("observed_cells") or []:
            j, i = cells.cell_of(o[0] + cells.cell / 2, o[1] + cells.cell / 2) or (None, None)
            if j is not None:
                cov[j, i] = max(cov[j, i], float(o[2]))
    return cov


def decide(area: dict, cfg: dict, cells: AreaCells, acq: dict, fp: np.ndarray, history: list[dict]):
    """Whether this pass is to be taken for this area. ``history``: the area's earlier pass records
    (passes.json rows and this run's). Returns (verdict, cov0, share): verdict "take", "low_share",
    "revisit" or "seen_this_month"; ``cov0`` the cells counted as already covered for the planner."""
    share = cells.share(fp)
    cov0 = np.zeros((cells.ny, cells.nx))
    if area["mode"] == "monthly":
        month = acq["acquired_utc"][:7]
        rows = [h for h in history if h["acquired_utc"][:7] == month and h.get("status") in ("ok", "partial")]
        cov0 = (rects_cov(cells, rows) >= 0.5).astype(float)
        new = cells.share((fp - cov0).clip(0))
        if new < float(cfg.get("min_new_share_monthly", 0.1)):
            return "seen_this_month", cov0, share
        return "take", cov0, share
    if share < float(area.get("min_area_share", cfg.get("min_area_share", 0.3))):
        return "low_share", cov0, share
    days = area.get("min_revisit_days")
    if days:
        t = cdse.parse_time(acq["acquired_utc"])
        for h in history:
            if h.get("status") in ("ok", "partial") and h["acquired_utc"] != acq["acquired_utc"] \
                    and abs((cdse.parse_time(h["acquired_utc"]) - t).total_seconds()) < float(days) * 86400:
                return "revisit", cov0, share
    return "take", cov0, share


def plan_rects(cells: AreaCells, area: dict, fp: np.ndarray, cov0: np.ndarray, pu_left: float,
               max_cells: int = 8) -> list[dict]:
    """Rectangles of whole cells over the area's sea in the footprint (``ocean.wide.greedy_plan``)."""
    if pu_left <= 0:
        return []
    ps = [{"id": area["id"], "weight": cells.own, "acqs": [0], "min_eff": float(area.get("min_eff", 0.3))}]
    plan, _cov = wide.greedy_plan(cells, [fp], [1.0], pu_left, max_requests=60, max_cells=max_cells, passes=ps,
                                  cov=cov0.copy())
    for p in plan:
        p["bbox"] = list(cells.bounds(p["j"], p["i"], p["h"], p["w"]))
        p["width"], p["height"] = wide.rect_px(tuple(p["bbox"]))
    return plan


# ---------------------------------------------------------------- per-image flags

def cell_stats(vv: np.ndarray, valid: np.ndarray, water: np.ndarray, h: int, w: int):
    """Per cell (row 0 = south, as the planner): (valid share, VV median dB of valid water or NaN)."""
    rows = np.array_split(np.arange(vv.shape[0]), h)
    cols = np.array_split(np.arange(vv.shape[1]), w)
    vshare, med = np.zeros((h, w)), np.full((h, w), np.nan)
    for a, rr in enumerate(rows):
        for b, cc in enumerate(cols):
            if not len(rr) or not len(cc):
                continue
            sl = (slice(rr[0], rr[-1] + 1), slice(cc[0], cc[-1] + 1))
            vshare[h - 1 - a, b] = valid[sl].mean()
            m = valid[sl] & water[sl]
            if m.sum() >= 50:
                med[h - 1 - a, b] = float(np.median(vv[sl][m]))
    return vshare, med


def sea_state(med: float, cfg: dict) -> str | None:
    if med is None or not math.isfinite(med):
        return None
    ss = cfg["sea_state"]
    return "low" if med < ss["low_db"] else "high" if med > ss["high_db"] else "ok"


def ambiguity_check(cands: list[dict], vv: np.ndarray, lia, land: np.ndarray, grid: Grid, heading: float | None,
                    cfg: dict) -> None:
    """Sets ``ambiguity_suspect`` (and ``ambiguity_parent`` land/sea) on each candidate in place."""
    amb = cfg["ambiguity"]
    py, px = grid.pixel_m
    for c in cands:
        c["ambiguity_suspect"], c["ambiguity_parent"] = False, None
        if heading is None:
            continue
        r0, c0 = grid.to_pixel(c["lon"], c["lat"])
        r0, c0 = float(r0) - 0.5, float(c0) - 0.5
        ri, ci = int(round(r0)), int(round(c0))
        inc = None
        if lia is not None and 0 <= ri < grid.height and 0 <= ci < grid.width and math.isfinite(lia[ri, ci]):
            inc = float(lia[ri, ci])
        delta = v2.ambiguity_offset_m(inc if inc and 25 <= inc <= 50 else 37.0)
        ue, un = v2._unit(heading)
        along = np.arange(-amb["tol_az_m"], amb["tol_az_m"] + 1e-9, min(py, px))
        across = np.arange(-amb["tol_rg_m"], amb["tol_rg_m"] + 1e-9, min(py, px))
        best, best_land = -np.inf, False
        for sgn in (1, -1):
            d = sgn * delta + along[:, None]
            x = d * ue + across[None, :] * un      # east (across = perpendicular: (un, -ue))
            y = d * un - across[None, :] * ue      # north
            rr = np.round(r0 - y / py).astype(int).ravel()
            cc = np.round(c0 + x / px).astype(int).ravel()
            ok = (rr >= 0) & (rr < grid.height) & (cc >= 0) & (cc < grid.width)
            if not ok.any():
                continue
            vals = vv[rr[ok], cc[ok]]
            if np.isfinite(vals).any():
                k = int(np.nanargmax(vals))
                if vals[k] > best:
                    best, best_land = float(vals[k]), bool(land[rr[ok][k], cc[ok][k]])
        if c.get("peak_vv_db") is not None and best >= float(c["peak_vv_db"]) + amb["drop_db"]:
            c["ambiguity_suspect"], c["ambiguity_parent"] = True, "land" if best_land else "sea"


def _xy(cands: list[dict]) -> np.ndarray:
    if not cands:
        return np.zeros((0, 2))
    lat0 = float(np.mean([c["lat"] for c in cands]))
    my, mx = m_per_deg(lat0)
    return np.array([[c["lon"] * mx, c["lat"] * my] for c in cands])


def clusters(cands: list[dict], radius_m: float, min_n: int) -> int:
    """Groups linked within ``radius_m`` with at least ``min_n`` members: sets ``cluster`` (group number)
    on their members (None for the others). Returns the number of groups."""
    for c in cands:
        c["cluster"] = None
    n = len(cands)
    if n < min_n:
        return 0
    xy = _xy(cands)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        d = np.hypot(xy[i + 1:, 0] - xy[i, 0], xy[i + 1:, 1] - xy[i, 1])
        for k in np.nonzero(d <= radius_m)[0]:
            a, b = find(i), find(i + 1 + int(k))
            if a != b:
                parent[a] = b
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    g = 0
    for members in groups.values():
        if len(members) >= min_n:
            g += 1
            for i in members:
                cands[i]["cluster"] = g
    return g


def near(points: list[dict], lon: float, lat: float, radius_m: float) -> bool:
    my, mx = m_per_deg(lat)
    return any(math.hypot((p["lon"] - lon) * mx, (p["lat"] - lat) * my) <= radius_m for p in points)


def classify(c: dict) -> tuple[str, str | None]:
    """("A" | "B" | "ref", reason of ref or None)."""
    for k in ("near_coast", "static", "static_learned"):
        if c.get(k):
            return "ref", k
    if c.get("sea_state") in ("low", "high"):
        return "ref", "sea_state"
    if c.get("confidence") == "low":
        return "ref", "v1_low"
    if (c.get("confidence") == "high" and c.get("v2_confidence") == "A" and not c.get("edge")
            and not c.get("ambiguity_suspect") and c.get("cluster") is None):
        return "A", None
    return "B", None


# ---------------------------------------------------------------- learned fixed objects

def learn_static(cands: list[dict], acq: dict, history: list[dict], prior_points, learned: dict, cfg: dict) -> int:
    """Adds points seen in >= min_hits of the last ``looks`` looks of the same satellite and track (this pass
    included) to ``learned["points"]``. ``history``: pass records (any area) newest last; ``prior_points``:
    function(acquired_utc) -> candidate points of that pass. Returns the number added."""
    rule = cfg["static_learned"]
    key = (acq.get("platform"), acq.get("relative_orbit"))
    same = [h for h in history if (h.get("platform"), h.get("relative_orbit")) == key
            and h.get("status") in ("ok", "partial") and h["acquired_utc"] < acq["acquired_utc"]]
    added = 0
    for c in cands:
        if c.get("near_coast") or c.get("static") or c.get("static_learned"):
            continue
        looks = [h for h in same if any(b[0] <= c["lon"] <= b[2] and b[1] <= c["lat"] <= b[3] for b in h.get("rects") or [])]
        looks = sorted({h["acquired_utc"] for h in looks})[-(int(rule["looks"]) - 1):]
        hits = 1 + sum(1 for t in looks if near(prior_points(t), c["lon"], c["lat"], float(rule["radius_m"])))
        if hits >= int(rule["min_hits"]) and not near(learned["points"], c["lon"], c["lat"], float(rule["radius_m"])):
            learned["points"].append({"lon": c["lon"], "lat": c["lat"], "platform": acq.get("platform"),
                                      "relative_orbit": acq.get("relative_orbit"), "hits": hits, "looks": len(looks) + 1,
                                      "added_utc": acq["acquired_utc"], "area": c.get("area")})
            c["static_learned"] = True
            added += 1
    return added


# ---------------------------------------------------------------- state and outputs

def _read_json(p: Path, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def _fc(features: list[dict], attribution: str, **extra) -> dict:
    return {"type": "FeatureCollection", "attribution": attribution, **extra, "features": features}


def _poly(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def merge_day(path: Path, features: list[dict], key, attribution_fn, **extra) -> None:
    """Replace this pass's features (``key(props)``) in the day file and keep the rest."""
    old = _read_json(path, {"features": []}).get("features", [])
    keys = {key(f["properties"]) for f in features}
    feats = [f for f in old if key(f["properties"]) not in keys] + features
    feats.sort(key=lambda f: (f["properties"].get("acquired_utc") or "", f["properties"].get("area") or ""))
    times = sorted({f["properties"].get("acquired_utc") for f in feats if f["properties"].get("acquired_utc")})
    _write(path, cp.dumps(_fc(feats, attribution_fn(times), **extra)))


def series_rows(passes: list[dict]) -> list[dict]:
    rows = []
    for p in sorted(passes, key=lambda r: (r["acquired_utc"], r["area"])):
        if p.get("status") not in ("ok", "partial", "no_data"):
            continue
        c = p.get("counts") or {}
        rows.append({"area": p["area"], "acquired_utc": p["acquired_utc"], "date": p["acquired_utc"][:10],
                     "platform": p.get("platform"), "relative_orbit": p.get("relative_orbit"),
                     "orbit_direction": p.get("orbit_direction"), "status": p["status"],
                     "area_sea_km2": p.get("area_sea_km2"), "observed_sea_km2": p.get("observed_sea_km2"),
                     "observed_share": p.get("observed_share"), "count_A": c.get("A"),
                     "count_A_ge120_confirmed": c.get("A_ge120_confirmed"),
                     "count_A_ge120_unconfirmed": c.get("A_ge120_unconfirmed"), "count_B": c.get("B"),
                     "clusters": c.get("clusters"), "density_A_per_1000km2": c.get("density_A_per_1000km2"),
                     "sea_state_excluded_km2": p.get("sea_state_excluded_km2"), "vv_median_db": p.get("vv_median_db"),
                     "pu": p.get("pu"), "source": cp.attribution([p["acquired_utc"]], osm=False)})
    return rows


def write_series(out: Path, passes: list[dict], now: datetime) -> None:
    rows = series_rows(passes)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SERIES_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if r.get(k) is None else r.get(k) for k in SERIES_FIELDS})
    _write(out / "series.csv", buf.getvalue())
    times = [r["acquired_utc"] for r in rows]
    _write(out / "series.json", cp.dumps({"version": VERSION, "generated_utc": cdse.iso(now),
                                          "attribution": cp.attribution(times, now),
                                          "null_rule": NULL_RULE, "fields": SERIES_FIELDS[:-1], "rows": rows}))


NULL_RULE = ("件数（count_*）は、域の海のうち実際に見た（データがあり海況が普通の）割合が count_min_share（30%）未満の撮影では "
             "null（見ていない）。30% 以上見て候補がないときだけ 0。count_A_ge120_confirmed は 10 m の切り出しによる確認（未実装）が"
             "済むまで null。海況不良のセル（VV 中央値が low_db 未満か high_db 超）は件数に入れず、見た面積からも除く。")


def write_meta(out: Path, cfg: dict, passes: list[dict], learned: dict, now: datetime) -> None:
    times = [p["acquired_utc"] for p in passes if p.get("status") in ("ok", "partial")]
    meta = {
        "version": VERSION, "config_version": cfg["version"], "generated_utc": cdse.iso(now),
        "attribution": cp.attribution(times, now),
        "credits": ["Contains modified Copernicus Sentinel data (Copernicus Sentinel-1, ESA; CDSE Sentinel Hub)",
                    "land and fixed objects © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"],
        "method": {
            "data": "Sentinel-1 IW GRD（VV・VH・入射角）、40 m（BILINEAR）、Sentinel Hub Process API。新しい撮影は CDSE OData カタログで確かめる",
            "detector": "cfar-v1（記録の検出器、40 m、守り帯 160 m・リング 320 m、長さ 80 m 以上）と cfar-v2（100 m 以内の一致）",
            "masks": "OSM の陸・海岸（海岸から 1 km は near_coast）・洋上の固定物（風車・掘削台・航路標識、300 m 膨張）、学習した固定物",
            "sea_state": cfg["sea_state"], "cluster": cfg["cluster"], "ambiguity": cfg["ambiguity"],
            "static_learned": cfg["static_learned"],
            "schedule": "毎日 19:41 UTC（04:41 JST）。昇交（09:20–10:32 UTC）は同じ日、降交（21:17–21:51 UTC）は翌日に処理",
        },
        "classes": {
            "A": "cfar-v1 が high かつ cfar-v2 の A と 100 m 以内で一致し、海岸帯・固定物・端・方位アンビギュイティ・海況不良・群れでない。"
                 "見出しの数に入れる。10 m の切り出しで船の形と長さ 80 m 以上を確認するまで確度は「低〜中」（chip_checked=false）",
            "B": "cfar-v1 high か cfar-v2 A の片方だけ、または cfar-v1 medium、または端・アンビギュイティ・群れの印あり。確度「低」。"
                 "GeoJSON に出すが見出しの数に入れない",
            "reference": "cfar-v1 low・海岸帯・固定物（OSM・学習）・海況不良。出さない（passes.json の内訳だけ）",
            "high": "「高」は S1 だけでは付けない。統幕の通峡の発表・海保の尖閣の発表・AIS の照合と一致したときだけ分析の文章で付ける",
        },
        "ais": "全件 AIS 未照合（ais=unmatched）。AIS の生データ・MMSI は出さない",
        "null_rule": NULL_RULE, "count_min_share": cfg["count_min_share"],
        "limits": LIMITS,
        "areas": [{k: a.get(k) for k in ("id", "name", "bbox", "priority", "mode", "orbits", "expected_per_month")}
                  for a in areas_in_order(cfg)],
        "files": {"areas.geojson": "監視域", "passes.json": "処理した撮影（域ごと）", "candidates/YYYY-MM-DD.geojson": "撮影日（UTC）ごとの候補（A・B）",
                  "coverage/YYYY-MM-DD.geojson": "その日に実際に見た範囲（0.1° のセル、海況不良のセルは state で区別）",
                  "series.csv・series.json": "域×撮影ごとの件数の時系列", "static_learned.json": "学習した固定物"},
        "passes": len(passes), "static_learned_points": len(learned.get("points", [])),
    }
    _write(out / "meta.json", cp.dumps(meta, indent=1))


def write_areas(out: Path, cfg: dict, now: datetime) -> None:
    feats = [{"type": "Feature", "geometry": _poly(*a["bbox"]),
              "properties": {k: a.get(k) for k in ("id", "name", "priority", "mode", "orbits", "expected_per_month",
                                                   "min_revisit_days", "note")}} for a in areas_in_order(cfg)]
    _write(out / "areas.geojson", cp.dumps(_fc(feats, cp.attribution([], now, osm=False), version=cfg["version"])))


def cell_features(cells: AreaCells, arr: np.ndarray, props: dict, state: np.ndarray | None = None) -> list[dict]:
    """Runs of seen cells (``arr`` >= 0.5) along each row, split by ``state``."""
    feats = []
    for j in range(cells.ny):
        i = 0
        while i < cells.nx:
            if arr[j, i] < 0.5 or cells.own[j, i] <= 0:
                i += 1
                continue
            st = state[j, i] if state is not None else "seen"
            k = i
            while k + 1 < cells.nx and arr[j, k + 1] >= 0.5 and cells.own[j, k + 1] > 0 and \
                    (state[j, k + 1] if state is not None else "seen") == st:
                k += 1
            feats.append({"type": "Feature", "geometry": _poly(*cells.bounds(j, i, 1, k - i + 1)),
                          "properties": {**props, "state": st}})
            i = k + 1
    return feats


# ---------------------------------------------------------------- OSM per area (cached)

def osm_path(out: Path, area_id: str) -> Path:
    return out / "osm" / f"{area_id}.json"


def compact_osm(osm: dict, seamark_re: str) -> dict:
    els = []
    for e in osm.get("elements", []):
        tags = {k: v for k, v in (e.get("tags") or {}).items()
                if k in ("natural", "man_made", "bridge", "highway", "railway", "power", "generator:source",
                         "seamark:type", "area")}
        if e.get("type") == "node" and "lat" in e:
            els.append({"type": "node", "lat": round(e["lat"], 6), "lon": round(e["lon"], 6), "tags": tags})
        elif e.get("geometry"):
            els.append({"type": e.get("type"), "tags": tags,
                        "geometry": [{"lat": round(p["lat"], 6), "lon": round(p["lon"], 6)} for p in e["geometry"] if p]})
    return {"osm3s": {"timestamp_osm_base": (osm.get("osm3s") or {}).get("timestamp_osm_base")}, "elements": els}


def area_osm(out: Path, area: dict, cfg: dict, overpass, now: datetime, log, dry: bool = False) -> dict:
    """The area's OSM elements from ``osm/<AREA>.json``; fetched again (one Overpass query) when missing or
    older than ``osm_refresh_days``."""
    p = osm_path(out, area["id"])
    old = _read_json(p, None)
    if old and old.get("fetched_utc"):
        age = (now - cdse.parse_time(old["fetched_utc"])).total_seconds() / 86400
        if age < float(cfg.get("osm_refresh_days", 30)):
            return old
    if dry:
        return old or {"elements": []}
    w, s, e, n = area["bbox"]
    g = Grid(w, s, e, n, 10, 10)
    seamark_re = cfg["osm_static"]["seamark_types"]
    q = cp.overpass_query(g, seamark_re).replace("[timeout:120]", "[timeout:300]")
    data = compact_osm(overpass(q), seamark_re)
    data.update({"area": area["id"], "bbox": area["bbox"], "fetched_utc": cdse.iso(now),
                 "source": "OpenStreetMap via Overpass API",
                 "license": "© OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"})
    _write(p, json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    log(f"OSM {area['id']}: 要素 {len(data['elements'])}（{data['osm3s']['timestamp_osm_base']}）")
    return data


# ---------------------------------------------------------------- run

class Recorder(cp.UsageRecorder):
    """UsageRecorder that writes nothing while no Sentinel Hub request was sent (days without a new pass)."""

    def write(self, summary, note=None):
        if self.client.requests == 0 and not getattr(self, "_written", False):
            return
        self._written = True
        super().write(summary, note)


def run(args, client=None, odata_get=None, overpass=None, now=None, pacer=None, sleep=time.sleep, log=None,
        clock=None) -> dict:
    log = log or (lambda m: print(m, file=sys.stderr))
    clock = clock or (lambda: datetime.now(timezone.utc))
    now = now or clock()
    cfg = load_config(args.config)
    bud = cp.budget(args, now)  # MonthStop outside the budget month, before anything is sent
    client = client if client is not None else cp.MonthClient(args.budget_month, utcnow=clock)
    overpass = overpass or cp.fetch_overpass
    lim = float(cfg.get("pu_per_min", 100))
    pacer = pacer or cp.Pacer(lim, int(lim), sleep=sleep)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {"version": VERSION, "run_utc": cdse.iso(now), "dry_run": bool(args.dry_run), "budget": bud,
               "lookback_days": int(args.lookback_days or cfg["lookback_days"]), "acquisitions": [], "tasks": [],
               "results": [], "errors": [], "attribution": cp.attribution([], now)}
    log(f"今月 {bud['month']}：sat 等 {bud['sat_month_pu']}・港 {bud['ports_month_pu']}・外洋 {bud['ocean_month_pu']} PU 使用済み"
        f" → 使える {bud['pu_left']} PU（この実行の上限 {bud['max_pu_run']}）")
    rec = Recorder(Path(args.usage), now, client, VERSION, "dry-run" if args.dry_run else "monitor 40 m",
                   counts=lambda s: (len({t["acquired_utc"] for t in s["tasks"] if t.get("pu")}), len(s["results"])))
    ctx = {"summary": summary, "report": report}
    return cp.guarded(rec, ctx, report_dir, client, pacer, log,
                      lambda: _run(args, cfg, client, odata_get or covp.http_json, overpass, now, pacer, sleep, log,
                                   clock, bud, summary, rec, report_dir))


def _run(args, cfg, client, odata_get, overpass, now, pacer, sleep, log, clock, bud, summary, rec, report_dir) -> dict:
    out = Path(args.out)
    areas = areas_in_order(cfg)
    cells = area_cells(cfg, areas)
    passes = _read_json(out / "passes.json", {}).get("passes", [])
    learned = _read_json(out / "static_learned.json", {}).get("points") is not None and \
        _read_json(out / "static_learned.json", {}) or {"points": []}
    learned.setdefault("points", [])
    lookback = summary["lookback_days"]
    acqs, by_id, nreq = catalogue(union_bbox(areas), now - timedelta(days=lookback), now, get=odata_get, log=log, sleep=sleep)
    summary["odata_requests"] = nreq
    done = {pass_key(p["acquired_utc"], p["area"]) for p in passes if p.get("status") in DONE}
    fps = {a["id"]: footprints(cells[a["id"]], acqs, by_id) for a in areas}
    for k, a in enumerate(acqs):
        summary["acquisitions"].append({**cp._acq_meta(a), "published_utc": a.get("published_utc"),
                                        "area_share": {ar["id"]: round(cells[ar["id"]].share(fps[ar["id"]][k]), 3)
                                                       for ar in areas if cells[ar["id"]].share(fps[ar["id"]][k]) > 0}})
    month = bud["month"]
    run_left = float(bud["pu_left"])
    spent = 0.0
    new_rows: list[dict] = []
    deadline = time.monotonic() + float(args.max_minutes) * 60
    stop = None
    for area in areas:
        ac = cells[area["id"]]
        for k in sorted(range(len(acqs)), key=lambda x: acqs[x]["time"]):
            a = acqs[k]
            key = pass_key(a["acquired_utc"], area["id"])
            if key in done:
                continue
            fp = fps[area["id"]][k]
            if ac.share(fp) <= 0.001:
                continue
            hist = [p for p in passes + new_rows if p["area"] == area["id"]]
            verdict, cov0, share = decide(area, cfg, ac, a, fp, hist)
            task = {"area": area["id"], "acquired_utc": a["acquired_utc"], "share": round(share, 3), "verdict": verdict}
            summary["tasks"].append(task)
            if verdict == "revisit":
                new_rows.append(_pass_row(a, area, ac, "skipped_revisit", share, now))
                continue
            if verdict != "take":
                continue
            area_used = sum(float(p.get("pu") or 0) for p in passes + new_rows
                            if p["area"] == area["id"] and p["acquired_utc"][:7] == month)
            left = min(float(area["pu_month_cap"]) - area_used, run_left - spent)
            items = plan_rects(ac, area, fp, cov0, max(0.0, left - 0.5), int(cfg.get("max_cells", 8)))
            task.update(requests=len(items), est_pu=round(sum(i["est_pu"] for i in items), 3))
            if not items:
                task["verdict"] = "budget"
                continue
            if args.dry_run or stop:
                task["verdict"] = "planned" if args.dry_run else "stopped"
                continue
            row, stop = _fetch_pass(args, cfg, client, overpass, now, pacer, sleep, log, clock, bud, summary, rec,
                                    out, area, ac, a, items, cov0, passes + new_rows, learned, deadline)
            spent += row.get("pu") or 0.0
            task["pu"] = row.get("pu")
            task["status"] = row["status"]
            if row["status"] != "error":
                new_rows.append(row)
                done.add(key)
    summary["planned_pu"] = round(sum(t.get("est_pu") or 0 for t in summary["tasks"]
                                      if t["verdict"] in ("take", "planned")), 3)
    if not args.dry_run:
        all_passes = passes + [r for r in new_rows if r.get("status") != "error"]
        all_passes.sort(key=lambda r: (r["acquired_utc"], r["area"]))
        if new_rows or not (out / "meta.json").exists():
            _write(out / "passes.json", cp.dumps({"version": VERSION, "attribution": cp.attribution(
                [p["acquired_utc"] for p in all_passes], now), "passes": all_passes}, indent=None))
            learned.update({"version": VERSION, "rule": cfg["static_learned"],
                            "attribution": cp.attribution([p["added_utc"] for p in learned["points"]], now)})
            _write(out / "static_learned.json", cp.dumps(learned, indent=1))
            write_series(out, all_passes, now)
            write_areas(out, cfg, now)
            write_meta(out, cfg, all_passes, learned, now)
    summary["new_passes"] = [{k: r.get(k) for k in ("area", "acquired_utc", "status", "observed_share", "counts", "pu")}
                             for r in new_rows]
    cp._write_summary(report_dir, summary, client, pacer)
    rec.write(summary)
    rep = report(summary)
    (report_dir / "report.txt").write_text(rep, encoding="utf-8")
    log(rep)
    return summary


def _pass_row(a: dict, area: dict, ac: AreaCells, status: str, share: float, now: datetime) -> dict:
    return {"area": area["id"], "acquired_utc": a["acquired_utc"], "platform": a.get("platform"),
            "relative_orbit": a.get("relative_orbit"), "orbit_direction": a.get("orbit_direction"),
            "published_utc": a.get("published_utc"), "processed_utc": cdse.iso(now),
            "hours_to_processed": round((now - a["time"]).total_seconds() / 3600, 1),
            "footprint_share": round(share, 3), "area_sea_km2": round(ac.sea_km2()), "status": status}


def _fetch_pass(args, cfg, client, overpass, now, pacer, sleep, log, clock, bud, summary, rec, out, area, ac, a,
                items, cov0, history, learned, deadline):
    """Fetch and detect one pass over one area. Returns (pass record, stop reason or None)."""
    row = _pass_row(a, area, ac, "ok", ac.share(np.zeros(1)) if False else 0.0, now)
    row["footprint_share"] = next((t["share"] for t in reversed(summary["tasks"])
                                   if t["area"] == area["id"] and t["acquired_utc"] == a["acquired_utc"]), None)
    seamark_re = cfg["osm_static"]["seamark_types"]
    observed = np.zeros((ac.ny, ac.nx))
    state = np.full((ac.ny, ac.nx), "seen", dtype=object)
    meds = np.full((ac.ny, ac.nx), np.nan)
    cands, rects, errors, pu, nreq, stop = [], [], 0, 0.0, 0, None
    try:
        osm = area_osm(out, area, cfg, overpass, now, log)
    except Exception as ex:  # Overpass busy: the pass is tried again by the next run (inside the window)
        summary["errors"].append({"area": area["id"], "acquired_utc": a["acquired_utc"], "stage": "overpass",
                                  "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
        row["status"] = "error"
        return row, None
    for n, p in enumerate(items):
        if cp.month_changed(args, clock):
            stop = "month"
            summary["errors"].append({"stage": "stop", "reason": "month", "error": cp.MONTH_STOP})
            break
        if time.monotonic() > deadline:
            stop = "time"
            summary["errors"].append({"stage": "stop", "error": f"{args.max_minutes} 分を過ぎたため止めた"})
            break
        if client.processing_units + p["est_pu"] > bud["pu_left"] + 1e-9:
            stop = "budget"
            summary["errors"].append({"stage": "stop", "error": "実際の PU が上限に届くため止めた"})
            break
        if n:
            sleep(2)
        g = Grid(p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3], p["width"], p["height"])
        pacer.wait(p["est_pu"])
        before = client.processing_units
        try:
            nreq += 1
            data, pu1 = client.process(g.bbox, g.width, g.height, a["t_from"], a["t_to"], with_lia=True,
                                       upsampling=cp.INTERPOLATION)
            pacer.spent(pu1)
            vv, vh, lia, dm = cp.decode(data)
            del data
            land, static, minfo = cp.masks(g, osm, seamark_re, float(cfg["static_dilate_m"]))
            c, info = wide.detect_rect(vv, vh, lia, land, static, g, a, datamask=dm)
            valid = cp.data_valid(vv, dm)
            vshare, med = cell_stats(vv, valid, valid & ~land, p["h"], p["w"])
            sl = (slice(p["j"], p["j"] + p["h"]), slice(p["i"], p["i"] + p["w"]))
            observed[sl] = np.maximum(observed[sl], vshare)
            meds[sl] = np.where(np.isfinite(med), med, meds[sl])
            heading = v2.platform_heading(a["orbit_direction"], (g.south + g.north) / 2) if a.get("orbit_direction") else None
            c = [x for x in c if ac.contains(x["lon"], x["lat"])]
            ambiguity_check(c, vv, lia, land, g, heading, cfg)
            cands.extend(c)
            rects.append(g.bbox)
            summary["results"].append({"area": area["id"], "acquired_utc": a["acquired_utc"], "bbox": g.bbox,
                                       "width": g.width, "height": g.height, "pu": pu1, **info,
                                       "mask": {kk: minfo.get(kk) for kk in ("land_fraction", "static_objects", "osm_base")}})
            log(f"{area['id']} {a['acquired_utc']} {g.bbox}: PU {pu1}、有効 {info['valid_fraction']}、VV {info['vv_median_db']} dB、"
                f"v1 {info['v1']['large_kept']}、v2 {info['v2']['candidates']}")
        except cp.MonthStop as ex:
            stop = "month"
            summary["errors"].append({"stage": "stop", "reason": "month", "error": str(ex)})
            break
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as ex:
            errors += 1
            summary["errors"].append({"area": area["id"], "acquired_utc": a["acquired_utc"], "stage": "fetch",
                                      "bbox": p["bbox"], "error": str(ex)[:300]})
        finally:
            pu += client.processing_units - before
        rec.write(summary)  # the PU spent so far are in --usage
    row["pu"] = round(pu, 3)
    row["requests"] = nreq
    row["rects"] = rects
    if not rects:
        row["status"] = "error" if (errors or stop) and pu == 0 else "no_data"
        if stop and not errors and pu == 0:
            row["status"] = "error"  # nothing sent: tried again by the next run
        return row, stop
    # sea state per cell
    for j in range(ac.ny):
        for i in range(ac.nx):
            st = sea_state(meds[j, i], cfg)
            if st in ("low", "high"):
                state[j, i] = f"sea_state_{st}"
    good = (state == "seen")
    seen = (observed >= 0.5) & (ac.own > 0)
    obs_km2 = float((ac.km2 * ac.sea * observed * good * seen).sum())
    bad_km2 = float((ac.km2 * ac.sea * observed * ~good * seen).sum())
    for c in cands:
        cell = ac.cell_of(c["lon"], c["lat"])
        st = sea_state(meds[cell], cfg) if cell else None
        c["sea_state"] = st
        c["area"] = area["id"]
        c["static_learned"] = near(learned["points"], c["lon"], c["lat"], float(cfg["static_learned"]["radius_m"]))
    cands = wide.dedupe(cands)
    # learned fixed objects (before the classes: a point learned now is a reference too)
    prior_cache: dict = {}

    def prior_points(t):
        if t not in prior_cache:
            fc = _read_json(out / "candidates" / f"{t[:10]}.geojson", {"features": []})
            prior_cache[t] = [{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                              for f in fc.get("features", []) if f["properties"].get("acquired_utc") == t]
        return prior_cache[t]
    row["static_learned_added"] = learn_static(cands, a, history, prior_points, learned, cfg)
    pool = [c for c in cands if not (c.get("near_coast") or c.get("static") or c.get("static_learned"))]
    n_clusters = clusters(pool, float(cfg["cluster"]["radius_m"]), int(cfg["cluster"]["min_n"]))
    excl = {"near_coast": 0, "static": 0, "static_learned": 0, "sea_state": 0, "v1_low": 0, "edge": 0,
            "ambiguity": 0, "cluster": 0}
    counts = {"A": 0, "B": 0, "ref": 0}
    pub = []
    for c in cands:
        c.setdefault("cluster", None)
        cls, why = classify(c)
        counts[cls] += 1
        if why:
            excl[why] += 1
        else:
            excl["edge"] += bool(c.get("edge"))
            excl["ambiguity"] += bool(c.get("ambiguity_suspect"))
            excl["cluster"] += c.get("cluster") is not None
        if cls == "ref":
            continue
        props = {"area": area["id"], "acquired_utc": c["acquired_utc"], "platform": c.get("platform"),
                 "relative_orbit": c.get("relative_orbit"), "orbit_direction": c.get("orbit_direction"),
                 "class": cls, "confidence_label": CLASS_LABEL[cls],
                 "length_m_est_40m": c["length_m_est"], "length_class_40m": c["length_class"],
                 "large_120m_est": c["length_m_est"] >= float(cfg["large_length_m"]),
                 "pixels_40m": c["pixels_40m"], "peak_vv_db": c["peak_vv_db"], "contrast_vv_db": c["contrast_vv_db"],
                 "vh_ratio_db": c["vh_ratio_db"], "v1_confidence": c["confidence"], "v2_confidence": c["v2_confidence"],
                 "v2_match": c["v2_match"], "edge": c["edge"], "ambiguity_suspect": c["ambiguity_suspect"],
                 "ambiguity_parent": c["ambiguity_parent"], "cluster": c["cluster"], "sea_state": c["sea_state"],
                 "chip_checked": False, "length_m_10m": None, "ais": "unmatched"}
        pub.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
                    "properties": props})
    area_km2 = ac.sea_km2()
    share = obs_km2 / area_km2 if area_km2 else 0.0
    counted = share >= float(cfg["count_min_share"])
    a_large = sum(1 for f in pub if f["properties"]["class"] == "A" and f["properties"]["large_120m_est"])
    row.update({
        "observed_sea_km2": round(obs_km2), "observed_share": round(share, 3), "sea_state_excluded_km2": round(bad_km2),
        "vv_median_db": round(float(np.nanmedian(meds)), 2) if np.isfinite(meds).any() else None,
        "status": "partial" if (errors or stop) else ("ok" if obs_km2 > 0 or bad_km2 > 0 else "no_data"),
        "counts": {"A": counts["A"] if counted else None, "B": counts["B"] if counted else None,
                   "A_ge120_unconfirmed": a_large if counted else None, "A_ge120_confirmed": None,
                   "clusters": n_clusters if counted else None,
                   "density_A_per_1000km2": round(counts["A"] / obs_km2 * 1000, 3) if counted and obs_km2 else None,
                   "published_A": counts["A"], "published_B": counts["B"], "reference": counts["ref"]},
        "excluded": excl,
        "observed_cells": [[round(ac.w + i * ac.cell, 6), round(ac.s + j * ac.cell, 6), round(float(observed[j, i]), 3)]
                           for j in range(ac.ny) for i in range(ac.nx) if observed[j, i] >= 0.5 and ac.own[j, i] > 0],
    })
    day = a["acquired_utc"][:10]
    att = lambda times: cp.attribution(times, now)
    merge_day(out / "candidates" / f"{day}.geojson", pub, lambda q: (q.get("acquired_utc"), q.get("area")), att,
              version=VERSION, name="大型船の候補（AIS 未照合、未確認）", classes=CLASS_LABEL)
    cov_props = {"area": area["id"], "acquired_utc": a["acquired_utc"], "platform": a.get("platform"),
                 "relative_orbit": a.get("relative_orbit"), "orbit_direction": a.get("orbit_direction")}
    merge_day(out / "coverage" / f"{day}.geojson", cell_features(ac, observed, cov_props, state),
              lambda q: (q.get("acquired_utc"), q.get("area")), lambda t: cp.attribution(t, now, osm=False),
              version=VERSION, note="見た範囲（0.1° のセル）。state: seen＝件数に入る、sea_state_low/high＝海況不良で件数に入れない")
    log(f"{area['id']} {a['acquired_utc']}: 見た {row['observed_sea_km2']} km²（{row['observed_share']}）、"
        f"A {counts['A']}・B {counts['B']}・参考 {counts['ref']}、群れ {n_clusters}、PU {row['pu']}")
    return row, stop


def report(s: dict) -> str:
    b = s["budget"]
    L = [f"{VERSION} {s['run_utc']}{'（dry-run）' if s['dry_run'] else ''}、直近 {s['lookback_days']} 日",
         f"出典: {s.get('attribution')}",
         f"今月 {b['month']}：使える {b['pu_left']} PU。OData {s.get('odata_requests')} リクエスト、"
         f"Sentinel Hub {s.get('requests')} リクエスト・{s.get('processing_units')} PU"]
    for a in s["acquisitions"]:
        if a["area_share"]:
            L.append(f"  撮影 {a['acquired_utc']} {a.get('platform')} {a.get('orbit_direction')} 軌道 {a.get('relative_orbit')} "
                     f"公開 {a.get('published_utc')} 域の覆い {json.dumps(a['area_share'], ensure_ascii=False)}")
    for t in s["tasks"]:
        L.append(f"  {t['area']} {t['acquired_utc']}: {t['verdict']}（覆い {t['share']}、計画 {t.get('requests', 0)} 件 "
                 f"{t.get('est_pu', 0)} PU、実績 {t.get('pu')} PU {t.get('status') or ''}）")
    for r in s.get("new_passes") or []:
        L.append(f"  → {r['area']} {r['acquired_utc']} {r['status']} 見た割合 {r.get('observed_share')} 件数 {json.dumps(r.get('counts'), ensure_ascii=False)}")
    for e in s["errors"]:
        L.append("エラー: " + json.dumps(e, ensure_ascii=False, default=str))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- plan (M0: catalogue of a past month)

def plan_month(cfg: dict, month: str, get=covp.http_json, log=print, sleep=time.sleep) -> dict:
    """Passes, cover and PU of each area (all areas, enabled or not) over one month from the OData
    catalogue, with the rules of ``run`` (plan B), without any Sentinel Hub request."""
    t0 = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    t1 = (t0 + timedelta(days=32)).replace(day=1)
    allareas = areas_in_order(cfg, include_disabled=True)
    enabled = [a for a in allareas if a.get("enabled")]
    # sea of an enabled area of higher priority is not counted again; a disabled area is planned on its own
    cells = area_cells(cfg, enabled)
    for a in allareas:
        if not a.get("enabled"):
            cells[a["id"]] = AreaCells(a, [], cell=float(cfg.get("cell_deg", 0.1)))
    acqs, by_id, nreq = catalogue(union_bbox(allareas), t0, t1, get=get, log=log, sleep=sleep)
    lags = [(cdse.parse_time(a["published_utc"]) - a["time"]).total_seconds() / 3600 for a in acqs if a.get("published_utc")]
    res = {"version": VERSION, "month": month, "odata_requests": nreq, "acquisitions": len(acqs),
           "publication_lag_h": covp.lag_stats(lags) if lags else None, "waste_factor": cfg.get("waste_factor", 1.15),
           "source": "CDSE OData catalogue（認証なし、PU・Sentinel Hub のリクエストを使わない）", "areas": {}}
    for area in allareas:
        ac = cells[area["id"]]
        fps = footprints(ac, acqs, by_id)
        hist, rows = [], []
        full_once = (ac.sea > 0).astype(float)
        for k in sorted(range(len(acqs)), key=lambda x: acqs[x]["time"]):
            a, fp = acqs[k], fps[k]
            share = ac.share(fp)
            if share <= 0.01:
                continue
            verdict, cov0, _s = decide(area, cfg, ac, a, fp, hist)
            items = plan_rects(ac, area, fp, cov0, 1e9, int(cfg.get("max_cells", 8))) if verdict == "take" else []
            est = round(sum(i["est_pu"] for i in items), 2)
            fetched = np.zeros((ac.ny, ac.nx))
            for i in items:
                sl = (slice(i["j"], i["j"] + i["h"]), slice(i["i"], i["i"] + i["w"]))
                fetched[sl] = np.maximum(fetched[sl], fp[sl])
            row = {"acquired_utc": a["acquired_utc"], "platform": a.get("platform"), "relative_orbit": a.get("relative_orbit"),
                   "orbit_direction": a.get("orbit_direction"), "share": round(share, 3), "verdict": verdict,
                   "requests": len(items), "est_pu": est, "planned_share": round(ac.share(fetched), 3),
                   "requested_km2": round(sum((i["width"] * i["height"]) * (wide.PIXEL_M ** 2) / 1e6 for i in items)),
                   "published_utc": a.get("published_utc")}
            rows.append(row)
            if verdict == "take":
                hist.append({"acquired_utc": a["acquired_utc"], "status": "ok",
                             "observed_cells": [[round(ac.w + i * ac.cell, 6), round(ac.s + j * ac.cell, 6), float(fetched[j, i])]
                                                for j in range(ac.ny) for i in range(ac.nx) if fetched[j, i] >= 0.5]})
            elif verdict == "revisit":
                pass
        full = plan_rects(ac, area, full_once, np.zeros((ac.ny, ac.nx)), 1e9, int(cfg.get("max_cells", 8)))
        taken = [r for r in rows if r["verdict"] == "take"]
        orbits: dict = {}
        for r in rows:
            o = orbits.setdefault(f"{(r['platform'] or '')[-2:].upper()}/{r['relative_orbit']} {(r['orbit_direction'] or '')[:3]}",
                                  {"passes": 0, "share_mean": 0.0})
            o["passes"] += 1
            o["share_mean"] += r["share"]
        for o in orbits.values():
            o["share_mean"] = round(o["share_mean"] / o["passes"], 3)
        res["areas"][area["id"]] = {
            "name": area["name"], "bbox": area["bbox"], "priority": area["priority"], "mode": area["mode"],
            "enabled": bool(area.get("enabled")), "sea_km2": round(ac.sea_km2()),
            "passes_touching": len(rows), "passes_ge_30pct": sum(1 for r in rows if r["share"] >= 0.3),
            "cover_equivalent": round(sum(r["share"] for r in rows), 2),
            "taken": len(taken), "taken_cover_equivalent": round(sum(r["planned_share"] for r in taken), 2),
            "skipped_revisit": sum(1 for r in rows if r["verdict"] == "revisit"),
            "skipped_low_share": sum(1 for r in rows if r["verdict"] == "low_share"),
            "est_pu_month": round(sum(r["est_pu"] for r in taken), 1),
            "est_pu_month_with_waste": round(sum(r["est_pu"] for r in taken) * float(cfg.get("waste_factor", 1.15)), 1),
            "est_pu_full_cover": round(sum(i["est_pu"] for i in full), 1),
            "pu_month_cap": area.get("pu_month_cap"), "design_expected_pu_month": area.get("expected_pu_month"),
            "design_expected_per_month": area.get("expected_per_month"),
            "orbits": orbits, "passes": rows}
    en = [v for v in res["areas"].values() if v["enabled"]]
    res["total_enabled"] = {"est_pu_month": round(sum(v["est_pu_month"] for v in en), 1),
                            "est_pu_month_with_waste": round(sum(v["est_pu_month_with_waste"] for v in en), 1),
                            "passes_taken": sum(v["taken"] for v in en),
                            "requests": sum(r["requests"] for v in en for r in v["passes"])}
    return res


def plan_report(res: dict) -> str:
    L = [f"{VERSION} plan {res['month']}（OData のカタログだけ、PU 0）：撮影 {res['acquisitions']}、OData {res['odata_requests']} リクエスト、"
         f"公開の遅れ（時間）{json.dumps(res['publication_lag_h'], ensure_ascii=False)}",
         f"有効な域の計：取る撮影 {res['total_enabled']['passes_taken']}、{res['total_enabled']['requests']} リクエスト、"
         f"見込み {res['total_enabled']['est_pu_month']} PU（×{res['waste_factor']} で {res['total_enabled']['est_pu_month_with_waste']}）"]
    for aid, v in res["areas"].items():
        L.append(f"[{aid}] {v['name']}（{'有効' if v['enabled'] else '無効'}・{v['mode']}・海 {v['sea_km2']} km²）："
                 f"かかる撮影 {v['passes_touching']}（30% 以上 {v['passes_ge_30pct']}）、覆いの換算 {v['cover_equivalent']} 回、"
                 f"取る {v['taken']}（換算 {v['taken_cover_equivalent']} 回、4 日以内の再訪で見送り {v['skipped_revisit']}・覆い不足 {v['skipped_low_share']}）、"
                 f"PU {v['est_pu_month']}（×{res['waste_factor']}：{v['est_pu_month_with_waste']}、1 回全面 {v['est_pu_full_cover']}）"
                 f"／設計 {v['design_expected_per_month']} 回・{v['design_expected_pu_month']} PU、上限 {v['pu_month_cap']}")
        L.append("    軌道: " + json.dumps(v["orbits"], ensure_ascii=False))
        for r in v["passes"]:
            L.append(f"    {r['acquired_utc']} {r['platform']} {r['orbit_direction']} {r['relative_orbit']} 覆い {r['share']} "
                     f"{r['verdict']} {r['requests']} 件 {r['est_pu']} PU（取得で覆う {r['planned_share']}）")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="open-sea monitoring with Sentinel-1 (ocean-monitor-1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="passes, cover and PU of each area over a past month (OData only, no PU)")
    p.add_argument("--month", required=True)
    p.add_argument("--config", type=Path, default=CONFIG_FILE)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="the daily run")
    r.add_argument("--config", type=Path, default=CONFIG_FILE)
    r.add_argument("--out", type=Path, default=Path("docs/data/ocean/monitor"))
    r.add_argument("--report-dir", type=Path, required=True, help="summary.json and report.txt (not published)")
    r.add_argument("--usage", type=Path, default=Path("docs/data/ocean/usage.csv"))
    r.add_argument("--budget-month", required=True)
    r.add_argument("--ocean-cap", type=float, required=True)
    r.add_argument("--global-cap", type=float, required=True)
    r.add_argument("--global-margin", type=float, required=True)
    r.add_argument("--sat-usage", nargs="*", default=[])
    r.add_argument("--other-usage", nargs="*", default=[])
    r.add_argument("--ports-usage", nargs="*", default=[])
    r.add_argument("--ocean-usage", nargs="*", default=[])
    r.add_argument("--max-pu", type=float, required=True, help="cap of this run (inside the month's caps)")
    r.add_argument("--max-minutes", type=float, default=100)
    r.add_argument("--lookback-days", type=int)
    r.add_argument("--dry-run", action="store_true", help="catalogue and plan only (no PU, nothing written to --out)")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        res = plan_month(load_config(args.config), args.month, log=lambda m: print(m, file=sys.stderr))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(cp.dumps(res, indent=1), encoding="utf-8")
        rep = plan_report(res)
        (args.out / "report.txt").write_text(rep, encoding="utf-8")
        print(rep)
        return 0
    now_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if args.budget_month != now_month:
        print(f"上限は {args.budget_month} の値（今は {now_month}）。その月の上限を書き直すまで取得しない", file=sys.stderr)
        return cp.EXIT_MONTH
    signal.signal(signal.SIGTERM, cp._on_sigterm)
    try:
        s = run(args)
    except cp.MonthStop as e:
        print(f"停止: {e}", file=sys.stderr)
        return cp.EXIT_MONTH
    except cdse.CredentialError as e:
        print(f"エラー（認証）: {e}", file=sys.stderr)
        return cp.EXIT_CREDENTIAL
    return cp.exit_code(s)


if __name__ == "__main__":
    raise SystemExit(main())
