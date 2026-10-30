"""Sentinel-1 / Sentinel-2 archive of the main PLAN ports other than Zhanjiang (trial, probe only).

The sites are in sat/ports.json (same format as sat/sites.json). The daily run (sat/run.py) does not
read them, and nothing here writes to docs/data/sat outside docs/data/sat/ports/ and
docs/data/sat/ports_usage.csv (the PU of this archive; not counted in the 1,000 PU cap of the daily run).

    python -m sat.port_archive plan  --out plan.json [--cap 4000] [--usage CSV ...]  # catalog only (no PU)
    python -m sat.port_archive mask  --mask-dir DIR [--port ID ...]                 # OSM land masks
    python -m sat.port_archive run   --port ID --cap PU --out DIR --raw-dir DIR --mask-dir DIR

Per port (``run``):
1. Sentinel-1 IW GRD, 10 m, VV + VH (+ local incidence angle), every acquisition of the last 12 months
   (both orbit directions). Per scene: the false colour WebP of sat/color.py (s1fc-1) and its thumbnail,
   the cfar-v1 candidates (sat/detect.py) as GeoJSON (one file per month), and one row of counts
   (candidates by zone and by length class). The raw sigma0 is not kept, except three representative
   chips per port (the UINT16 PNG of the Process API, dB = v / 800 - 50) under ``--raw-dir``.
2. Sentinel-2 L2A natural colour on clear days of the last 12 months, with sat/s2.py (tile cloud cover
   and a 60 m SCL pre-check before the 10 m request), in 30-day windows.

Budget: the PU of every response header are summed; a request is not sent if the port's cap would be
exceeded (the estimate of the next request is checked first). Rate: at most ``--pu-per-min`` PU and
``--req-per-min`` requests in any 60 s (the account limit is 300 / 300 shared by all jobs); HTTP 429 is
waited for and retried.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, s2
from . import color as fcolor
from .detect import DETECTOR_VERSION, LENGTH_CLASSES, detect
from .grid import Grid

ARCHIVE_VERSION = "ports-1"
PORTS_FILE = Path(__file__).resolve().parent / "ports.json"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "data" / "sat" / "ports"
LOOKBACK_DAYS = 365
PORTS_CAP = 4000.0          # the whole port archive (October 2026)
GLOBAL_CAP = 9000.0         # sat + ocean + ports in the month (safety guard)
GLOBAL_MARGIN = 300.0       # kept free for the daily runs to the end of the month
REQUEST_CAP = 10000         # CDSE requests per month
REQUEST_MARGIN = 1500
MIN_COVERAGE = 0.5          # as sat/run.py
S2_WINDOW_DAYS = 30
S2_CLEAR_SHARE = 0.35       # plan only: share of days with a 10 m request
S2_PRE_PU = 0.03
RAW_KEEP = 3
DONE_RESULTS = ("counted", "skipped_low_coverage", "skipped_footprint")
SCENE_FIELDS = ["site_id", "acquired_utc", "platform", "orbit_direction", "relative_orbit", "result",
                "coverage", "incidence_deg", "candidates", "harbour_candidates", "quay_candidates"] + \
               [f"len_{n}" for n, _lo, _hi in LENGTH_CLASSES] + \
               ["processing_units", "color_image", "color_thumb", "raw_chip", "mask_version",
                "detector_version", "archive_version", "retrieved_utc", "error"]
USAGE_FIELDS = ["run_utc", "site_id", "requests", "processing_units", "pu_source", "s1_scenes",
                "s2_days", "s2_clear", "s1_processing_units", "s2_processing_units", "cap_pu", "archive_version"]


class BudgetStop(cdse.CredentialError):
    """The next request would exceed the cap (PU or requests) or the time limit. Subclass of
    CredentialError so that sat/s2.py does not swallow it as a one-day failure."""


class LimitedClient(cdse.Client):
    """cdse.Client with a PU / request cap, a sliding-window rate limit and patient 429 retries."""

    def __init__(self, cap_pu: float, cap_req: int, pu_per_min: float, req_per_min: int,
                 deadline: float | None = None, clock=time.monotonic, sleep=time.sleep, **kw):
        super().__init__(**kw)
        self.cap_pu = cap_pu
        self.cap_req = cap_req
        self.pu_per_min = pu_per_min
        self.req_per_min = req_per_min
        self.deadline = deadline
        self.clock = clock
        self.sleep = sleep
        self.window: deque = deque()  # (t, pu) of the last 60 s
        self.next_est = 0.0
        self.next_key = None
        self.seen_pu: dict = {}  # (width, height, input bands) -> largest PU of the header so far
        self.waited_s = 0.0
        self.retries_429 = 0

    def _trim(self, now: float) -> None:
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()

    def _gate(self, est: float) -> None:
        if self.deadline is not None and self.clock() > self.deadline:
            raise BudgetStop("時間の上限に達したため止める")
        if self.processing_units + est > self.cap_pu:
            raise BudgetStop(f"PU の上限（{self.cap_pu:.1f}）を超えるため止める（使用 {self.processing_units:.1f}、次の見込み {est:.2f}）")
        if self.requests + 1 > self.cap_req:
            raise BudgetStop(f"リクエスト数の上限（{self.cap_req}）に達したため止める")
        while True:
            now = self.clock()
            self._trim(now)
            pu = sum(p for _t, p in self.window)
            if len(self.window) < self.req_per_min and pu + est <= self.pu_per_min:
                return
            wait = max(0.5, 60 - (now - self.window[0][0])) if self.window else 1.0
            self.waited_s += wait
            self.sleep(wait)

    def _post_sh(self, url, payload, accept):
        est, key = (self.next_est, self.next_key) if url == cdse.PROCESS_URL else (0.0, None)
        if key is not None:
            est = max(est, self.seen_pu.get(key, 0.0))  # the header is the truth; never assume less
        self.next_est, self.next_key = 0.0, None
        for attempt in range(6):
            self._gate(est)
            try:
                h, data = super()._post_sh(url, payload, accept)
            except cdse.CredentialError:
                raise
            except cdse.CDSEError as e:
                if "429" in str(e) and attempt < 5:
                    self.retries_429 += 1
                    self.sleep(60)
                    continue
                raise
            pu = h.get("x-processingunits-spent")
            try:
                pu = float(pu) if pu is not None else est
            except ValueError:
                pu = est
            self.window.append((self.clock(), pu))
            if key is not None:
                self.seen_pu[key] = max(self.seen_pu.get(key, 0.0), pu)
            return h, data
        raise cdse.CDSEError("HTTP 429 が続いた")

    def process(self, bbox, width, height, t_from, t_to, with_lia=True):
        self.next_est = cdse.estimate_pu(width, height, 3 if with_lia else 2)
        self.next_key = (width, height, 3 if with_lia else 2)
        return super().process(bbox, width, height, t_from, t_to, with_lia)

    def process_raw(self, payload, accept, input_bands):
        out = payload.get("output", {})
        self.next_est = round(max(0.005, out.get("width", 0) * out.get("height", 0) / 512 ** 2 * max(input_bands / 3, 1)), 3)
        self.next_key = (out.get("width", 0), out.get("height", 0), input_bands)
        return super().process_raw(payload, accept, input_bands)


# ---------- sites, files ----------

def load_ports(path: Path = PORTS_FILE) -> list[dict]:
    return [p for p in json.loads(path.read_text(encoding="utf-8"))["sites"] if p.get("enabled", True)]


def mask_tag(site: dict) -> str:
    return f"{site['id']}-land-v{int(site.get('mask_version', 1))}"


def stamp(acquired_utc: str) -> str:
    return acquired_utc.replace(":", "-")


def color_paths(site: dict, acquired_utc: str) -> tuple[str, str]:
    base = f"color/v{int(site.get('mask_version', 1))}/{stamp(acquired_utc)}"
    return base + ".webp", base + ".thumb.webp"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    path.write_text(buf.getvalue(), encoding="utf-8")


def month_usage(paths: list[Path], month: str) -> tuple[float, int]:
    """PU and requests of ``month`` in usage CSVs (columns run_utc, requests, processing_units)."""
    pu, req = 0.0, 0
    for p in paths:
        for r in read_csv(p):
            if (r.get("run_utc") or "").startswith(month):
                pu += float(r.get("processing_units") or 0)
                req += int(float(r.get("requests") or 0))
    return pu, req


# ---------- catalog footprints ----------

def _rings(geom: dict | None) -> list:
    if not geom:
        return []
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return [c[0]] if c else []
    if t == "MultiPolygon":
        return [p[0] for p in c if p]
    return []


def _inside(x: np.ndarray, y: np.ndarray, ring) -> np.ndarray:
    r = np.asarray(ring, float)
    xs, ys = r[:, 0], r[:, 1]
    out = np.zeros(x.shape, bool)
    j = len(r) - 1
    for i in range(len(r)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if yi != yj:
            out ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        j = i
    return out


def footprint_share(geoms: list, grid: Grid, water: np.ndarray, step: int = 8) -> float | None:
    """Share of the site's water (sampled every ``step`` pixels) inside the union of the catalog footprints.
    None when no footprint geometry is known (the scene is then fetched as before)."""
    rings = [r for g in geoms for r in _rings(g)]
    if not rings:
        return None
    rr = np.arange(step // 2, grid.height, step)
    cc = np.arange(step // 2, grid.width, step)
    lon, lat = grid.to_lonlat(rr[:, None] + 0.5, cc[None, :] + 0.5)
    lon, lat = np.broadcast_to(lon, (len(rr), len(cc))), np.broadcast_to(lat, (len(rr), len(cc)))
    sel = water[np.ix_(rr, cc)]
    if not sel.any():
        return None
    inside = np.zeros(sel.shape, bool)
    for ring in rings:
        inside |= _inside(lon, lat, ring)
    return round(float(inside[sel].mean()), 3)


def acquisitions_with_footprints(feats: list[dict]) -> list[dict]:
    geo = {f.get("id"): f.get("geometry") for f in feats}
    acqs = cdse.acquisitions(feats)
    for a in acqs:
        a["geometries"] = [geo.get(i) for i in a["ids"] if geo.get(i)]
    return acqs


def water_of(site: dict, mask: np.ndarray | None) -> np.ndarray:
    grid = Grid.for_site(site)
    return (mask == 0) if mask is not None else np.ones((grid.height, grid.width), bool)


def row_tag(site: dict, has_mask: bool) -> str:
    return mask_tag(site) if has_mask else "none"


# ---------- plan ----------

def s2_full_pu(grid: Grid) -> float:
    return round(grid.width * grid.height / 512 ** 2 * 5 / 3, 3)


def estimate_port(site: dict, n_s1: int, n_s2_days: int) -> dict:
    grid = Grid.for_site(site)
    s1_pu = cdse.estimate_pu(grid.width, grid.height, 3)
    s1 = n_s1 * s1_pu
    s2_total = n_s2_days * S2_PRE_PU + n_s2_days * S2_CLEAR_SHARE * s2_full_pu(grid)
    return {"site_id": site["id"], "width": grid.width, "height": grid.height, "area_km2": round(grid.area_km2(), 1),
            "s1_scenes": n_s1, "s1_pu_per_scene": s1_pu, "s1_pu": round(s1, 1),
            "s2_days": n_s2_days, "s2_pu": round(s2_total, 1), "pu": round(s1 + s2_total, 1),
            "s1_requests": n_s1, "s2_requests": int(n_s2_days * (1 + S2_CLEAR_SHARE)) + 13,
            "priority": int(site.get("priority", 1))}


def allocate(estimates: list[dict], cap: float, req_cap: int) -> list[dict]:
    """Per-port caps (PU, requests). Priority-1 ports first; a priority-2 port only if all of it fits.
    Each port gets 1.25 x its estimate (it stops on its own cap), scaled down if the sum is over ``cap``."""
    chosen, total, total_req = [], 0.0, 0
    for e in sorted(estimates, key=lambda e: (e["priority"], e["site_id"])):
        need = e["pu"] * 1.25
        nreq = int((e["s1_requests"] + e["s2_requests"]) * 1.25) + 20
        if e["priority"] > 1 and (total + need > cap or total_req + nreq > req_cap):
            e["skipped"] = "枠が足りないため取らない（priority 2）"
            continue
        chosen.append((e, need, nreq))
        total += need
        total_req += nreq
    f = min(1.0, cap / total) if total else 1.0
    fr = min(1.0, req_cap / total_req) if total_req else 1.0
    out = []
    for e, need, nreq in chosen:
        e["cap_pu"] = round(need * f, 1)
        e["cap_req"] = int(nreq * fr)
        out.append(e)
    return out


def allocate_ordered(estimates: list[dict], cap: float, req_cap: int) -> list[dict]:
    """In the given order: each port gets 1.25 x its estimate while the cap lasts; the next one gets what is
    left (if at least 5 PU), the rest are skipped."""
    out, left, left_req = [], cap, req_cap
    for e in estimates:
        need = e["pu"] * 1.25
        nreq = int((e["s1_requests"] + e["s2_requests"]) * 1.25) + 20
        if left < 5 or left_req < 30:
            e["skipped"] = "枠が足りないため取らない（優先順の後ろ）"
            continue
        e["cap_pu"] = round(min(need, left), 1)
        e["cap_req"] = min(nreq, left_req)
        left -= e["cap_pu"]
        left_req -= e["cap_req"]
        out.append(e)
    return out


def plan(client, ports: list[dict], now: datetime, cap: float, req_cap: int, log=print,
         out_root: Path | None = None, mask_dir: Path | None = None, days_from: int = LOOKBACK_DAYS,
         days_to: int = 0, s2_on: bool = True, ordered: bool = False) -> dict:
    """Only the work not done yet (S1 scenes and S2 days of the current mask version) is estimated; scenes
    whose catalog footprint covers less than MIN_COVERAGE of the water are not counted (not fetched)."""
    from .osm_mask import load_mask
    est = []
    for site in ports:
        grid = Grid.for_site(site)
        t0, t1 = now - timedelta(days=days_from), now - timedelta(days=days_to)
        mask = None
        if mask_dir is not None:
            try:
                mask, _m = load_mask(site["id"], int(site.get("mask_version", 1)), mask_dir)
            except FileNotFoundError:
                mask = None
        tag = row_tag(site, mask is not None)
        out = (out_root / site["id"]) if out_root is not None else None
        done1 = {r["acquired_utc"] for r in read_csv(out / "s1_scenes.csv") if r.get("mask_version") == tag
                 and r.get("result") in DONE_RESULTS} if out else set()
        done2 = {r["date"] for r in read_csv(out / "s2_scenes.csv") if r.get("mask_version") == s2.mask_tag(site)
                 and r.get("result") != "error"} if out else set()
        feats1 = client.catalog(grid.bbox, t0, t1)
        modes: dict = {}
        for f in feats1:
            pr = f.get("properties") or {}
            k = f"{pr.get('sar:instrument_mode', '?')}/{'+'.join(pr.get('sar:polarizations') or [])}"
            modes[k] = modes.get(k, 0) + 1
        acqs = acquisitions_with_footprints(feats1)
        water = water_of(site, mask)
        new1 = [a for a in acqs if a["acquired_utc"] not in done1]
        low = [a for a in new1 if (lambda f: f is not None and f < MIN_COVERAGE)(footprint_share(a["geometries"], grid, water))]
        n1 = len(new1) - len(low)
        # S2 is always the last 12 months (run_s2), whatever the S1 window
        n2 = len([d for d in s2.day_groups(client.catalog(grid.bbox, now - timedelta(days=LOOKBACK_DAYS), now,
                                                          collection=s2.COLLECTION))
                  if d["date"] not in done2]) if s2_on else 0
        e = estimate_port(site, n1, n2)
        e.update(s1_done=len(done1), s2_done=len(done2), s1_footprint_skip=len(low), mask_version=tag,
                 s1_catalog_items=len(feats1), s1_catalog_modes=modes)
        est.append(e)
        log(f"{site['id']} {site['name']} [{tag}]: S1 新 {n1} 撮影（済み {len(done1)}、足跡で除く {len(low)}）× "
            f"{e['s1_pu_per_scene']} PU = {e['s1_pu']} PU（カタログ {len(feats1)} 件 {modes}）、S2 新 {n2} 日（済み {len(done2)}）→ 約 {e['s2_pu']} PU、計 {e['pu']} PU")
    todo = [e for e in est if e["s1_scenes"] or e["s2_days"]]
    if not s2_on:
        for e in todo:
            e["s2_requests"] = 0
    chosen = allocate_ordered(todo, cap, req_cap) if ordered else allocate(todo, cap, req_cap)
    return {"plan_utc": cdse.iso(now), "cap_pu": cap, "cap_req": req_cap,
            "estimate_pu": round(sum(e["pu"] for e in est), 1),
            "chosen_estimate_pu": round(sum(e["pu"] for e in chosen), 1),
            "catalog_requests": client.requests, "ports": est,
            "matrix": [{"port": e["site_id"], "cap": e["cap_pu"], "req": e["cap_req"]} for e in chosen]}


# ---------- run ----------

def representative(rows: list[dict], k: int = RAW_KEEP) -> list[str]:
    """acquired_utc of k counted scenes: one per relative orbit (most scenes first), the one nearest the
    middle of that orbit's series; filled with the earliest / latest of the main orbit."""
    counted = [r for r in rows if r.get("result") == "counted"]
    by_orbit: dict = {}
    for r in sorted(counted, key=lambda r: r["acquired_utc"]):
        by_orbit.setdefault(str(r.get("relative_orbit")), []).append(r)
    groups = sorted(by_orbit.values(), key=lambda g: (-len(g), g[0]["acquired_utc"]))
    out: list[str] = []
    for g in groups:
        if len(out) >= k:
            break
        out.append(g[len(g) // 2]["acquired_utc"])
    if groups:
        for r in (groups[0][0], groups[0][-1]) + tuple(counted):
            if len(out) >= k:
                break
            if r["acquired_utc"] not in out:
                out.append(r["acquired_utc"])
    return out[:k]


def run_s1(client: LimitedClient, site: dict, out: Path, tmp_raw: Path, mask: np.ndarray | None, now: datetime,
           days: int = LOOKBACK_DAYS, log=print, days_to: int = 0) -> dict:
    """S1 scenes acquired between ``days`` and ``days_to`` days before ``now`` (default: the last 12 months)."""
    from .run import decode_chip
    grid = Grid.for_site(site)
    path = out / "s1_scenes.csv"
    tag = row_tag(site, mask is not None)
    rows_all = read_csv(path)
    others = [r for r in rows_all if r.get("mask_version") != tag]  # earlier mask versions stay as they are
    rows = [r for r in rows_all if r.get("mask_version") == tag]
    done = {r["acquired_utc"] for r in rows if r.get("result") in DONE_RESULTS}
    rows = [r for r in rows if r["acquired_utc"] in done]
    acqs = acquisitions_with_footprints(client.catalog(grid.bbox, now - timedelta(days=days), now - timedelta(days=days_to)))
    land = (mask > 0) if mask is not None else np.zeros((grid.height, grid.width), bool)
    if land.shape != (grid.height, grid.width):
        raise ValueError(f"{site['id']}: 陸マスクの大きさ {land.shape} が範囲 {(grid.height, grid.width)} と合わない")
    water_px = int((~land).sum())
    feats_by_month: dict = {}
    summary = {"catalog_acquisitions": len(acqs), "counted": 0, "skipped": 0, "footprint_skipped": 0, "errors": 0,
               "stopped": None}
    pu0 = client.processing_units
    log(f"{site['id']} S1: 撮影 {len(acqs)} 回（過去 {days} 日）、済み {len(done)}")
    for acq in acqs:
        a = acq["acquired_utc"]
        if a in done:
            continue
        row = {"site_id": site["id"], "acquired_utc": a, "platform": acq["platform"],
               "orbit_direction": acq["orbit_direction"], "relative_orbit": acq["relative_orbit"],
               "mask_version": tag, "detector_version": DETECTOR_VERSION,
               "archive_version": ARCHIVE_VERSION, "retrieved_utc": cdse.iso(datetime.now(timezone.utc))}
        share = footprint_share(acq.get("geometries") or [], grid, ~land)
        if share is not None and share < MIN_COVERAGE:  # catalog footprint: no request, no PU
            row.update(result="skipped_footprint", coverage=share, processing_units=0)
            rows.append(row)
            summary["footprint_skipped"] += 1
            log(f"{site['id']} S1 {a}: 足跡が水域の {share:.0%} しか覆わないため取らない")
            continue
        try:
            data, pu = client.process(grid.bbox, grid.width, grid.height, acq["t_from"], acq["t_to"], True)
        except BudgetStop as e:
            summary["stopped"] = str(e)
            log(f"{site['id']} S1: {e}")
            break
        except cdse.CredentialError:
            raise
        except Exception as e:
            row.update(result="error", error=str(e)[:300])
            rows.append(row)
            summary["errors"] += 1
            log(f"{site['id']} S1 {a}: 失敗 {str(e)[:200]}")
            continue
        row["processing_units"] = pu
        try:
            vv, vh, lia = decode_chip(data)
            valid_water = np.isfinite(vv) & ~land
            coverage = valid_water.sum() / water_px if water_px else 0.0
            row["coverage"] = round(float(coverage), 3)
            if coverage < MIN_COVERAGE:
                row["result"] = "skipped_low_coverage"
                summary["skipped"] += 1
                rows.append(row)
                log(f"{site['id']} S1 {a}: 水域の {coverage:.0%} しか写っていない")
                continue
            tmp_raw.mkdir(parents=True, exist_ok=True)
            (tmp_raw / f"{site['id']}-{stamp(a)}.png").write_bytes(data)
            if lia is not None and np.isfinite(lia[valid_water]).any():
                row["incidence_deg"] = round(float(np.nanmedian(lia[valid_water])), 2)
            if mask is not None:
                dets, _info = detect(vv, vh, land, grid.pixel_m)
            else:
                dets = []
            row["candidates"] = len(dets) if mask is not None else ""
            row["harbour_candidates"] = sum(d["zone"] == "harbour" for d in dets) if mask is not None else ""
            row["quay_candidates"] = sum(d["zone"] == "quay" for d in dets) if mask is not None else ""
            for name, _lo, _hi in LENGTH_CLASSES:
                row[f"len_{name}"] = sum(d["length_class"] == name for d in dets) if mask is not None else ""
            cpath, tpath = color_paths(site, a)
            label = (f"{site['id']} {site.get('name', '')}  {a}  {acq['platform']} {acq['orbit_direction']} "
                     f"orbit {acq['relative_orbit']}  (provisional)")
            full, thumb = fcolor.render(vv, vh, label, f"Contains modified Copernicus Sentinel data {a[:4]}")
            for rel, b in ((cpath, full), (tpath, thumb)):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes(b)
            row.update(color_image=cpath, color_thumb=tpath, result="counted")
            for i, d in enumerate(sorted(dets, key=lambda d: (d["row"], d["col"]))):
                lon, lat = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
                feats_by_month.setdefault(a[:7], []).append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(float(lon), 6), round(float(lat), 6)]},
                    "properties": {"site_id": site["id"], "acquired_utc": a, "n": i + 1,
                                   "platform": acq["platform"], "orbit_direction": acq["orbit_direction"],
                                   "relative_orbit": acq["relative_orbit"], "zone": d["zone"],
                                   "length_m": d["length_m"], "width_m": d["width_m"], "length_class": d["length_class"],
                                   "pixels": d["pixels"], "peak_vv_db": d["peak_vv_db"],
                                   "contrast_vv_db": d["contrast_vv_db"], "vh_ratio_db": d["vh_ratio_db"],
                                   "status": "candidate", "reliability": "A", "credibility": "3",
                                   "mask_version": row["mask_version"], "detector_version": DETECTOR_VERSION,
                                   "image": cpath}})
            summary["counted"] += 1
            log(f"{site['id']} S1 {a} {acq['platform']} {acq['orbit_direction']} 軌道{acq['relative_orbit']}: "
                f"候補 {row['candidates']}（港内 {row['harbour_candidates']}、岸壁帯 {row['quay_candidates']}）、"
                f"{pu} PU、水域の {coverage:.0%}")
        except Exception as e:  # PU spent: record the failure, go on
            row.update(result="error", error=f"検出・描画で失敗: {str(e)[:280]}")
            summary["errors"] += 1
            log(f"{site['id']} S1 {a}: 検出・描画で失敗 {str(e)[:200]}")
        rows.append(row)
    write_csv(path, sorted(others + rows, key=lambda r: (r["acquired_utc"], r.get("mask_version") or "")), SCENE_FIELDS)
    for month, feats in feats_by_month.items():
        p = out / "detections" / f"{month}.geojson"
        fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"type": "FeatureCollection", "features": []}
        keep = {(f["properties"]["acquired_utc"], f["properties"]["mask_version"]) for f in feats}
        fc["features"] = [f for f in fc["features"]
                          if (f["properties"]["acquired_utc"], f["properties"].get("mask_version")) not in keep] + feats
        fc["features"].sort(key=lambda f: (f["properties"]["acquired_utc"], f["properties"].get("mask_version") or "",
                                           f["properties"]["n"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    summary["processing_units"] = round(client.processing_units - pu0, 3)
    return summary


def raw_subdir(site: dict) -> str:
    v = int(site.get("mask_version", 1))
    return site["id"] if v == 1 else f"{site['id']}/v{v}"  # v1 keeps the path of the first run


def keep_raw(site: dict, out: Path, tmp_raw: Path, raw_dir: Path, tag: str | None = None) -> list[str]:
    path = out / "s1_scenes.csv"
    rows = read_csv(path)
    tag = tag or mask_tag(site)
    mine = [r for r in rows if r.get("mask_version") == tag]
    if any(r.get("raw_chip") for r in mine):
        return []  # already kept for this version
    have = {r["acquired_utc"] for r in mine if (tmp_raw / f"{site['id']}-{stamp(r['acquired_utc'])}.png").exists()}
    picks = representative([r for r in mine if r["acquired_utc"] in have])
    sub = raw_subdir(site)
    kept = []
    for a in picks:
        src = tmp_raw / f"{site['id']}-{stamp(a)}.png"
        dst = raw_dir / sub / f"{stamp(a)}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        kept.append(str(dst.relative_to(raw_dir)))
    if kept:
        for r in mine:
            if r["acquired_utc"] in picks:
                r["raw_chip"] = f"probe-raw/{sub}/{stamp(r['acquired_utc'])}.png"
        write_csv(path, rows, SCENE_FIELDS)
        grid = Grid.for_site(site)
        (raw_dir / sub / "README.json").write_text(json.dumps({
            "site": site["id"], "mask_version": tag, "bbox": grid.bbox, "width": grid.width, "height": grid.height,
            "encoding": "PNG UINT16, 4 bands: VV, VH (dB = v / 800 - 50, 0 = no data), local incidence angle x 100, dataMask",
            "source": "Copernicus Sentinel-1 IW GRDH via CDSE Sentinel Hub Process API (SIGMA0_ELLIPSOID, COPERNICUS_30 DEM, NEAREST)",
            "credit": "Contains modified Copernicus Sentinel data",
            "chips": kept, "why": "岸壁の数え方を作るための代表 3 撮影（主な軌道ごとに系列の中ほど）",
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return kept


def run_s2(client: LimitedClient, site: dict, out: Path, now: datetime, cap_left, deadline_left, log=print) -> dict:
    tot = {"days": 0, "clear": 0, "cloudy": 0, "skipped": 0, "deferred": 0, "errors": 0, "processing_units": 0.0,
           "stopped": None}
    s2site = {**site, "sectors": []}
    end = now
    start = now - timedelta(days=LOOKBACK_DAYS)
    while end > start:
        days = min(S2_WINDOW_DAYS, (end - start).days or 1)
        if deadline_left() <= 0:
            tot["stopped"] = "時間の上限"
            break
        left = cap_left() - 3.0
        if left <= 0:
            tot["stopped"] = "PU の上限"
            break
        try:
            s = s2.update_site(client, out, s2site, end, days=days, max_pu=left, log=log)
        except BudgetStop as e:
            tot["stopped"] = str(e)
            log(f"{site['id']} S2: {e}（この 30 日の行は記録されない）")
            break
        for k in ("days", "clear", "cloudy", "skipped", "deferred", "errors"):
            tot[k] += s.get(k, 0)
        tot["processing_units"] = round(tot["processing_units"] + s.get("processing_units", 0), 3)
        end -= timedelta(days=days)
    return tot


def run_port(client: LimitedClient, site: dict, out_root: Path, raw_dir: Path, tmp_raw: Path,
             mask_dir: Path, now: datetime, log=print, days_from: int = LOOKBACK_DAYS, days_to: int = 0,
             s2_on: bool = True) -> dict:
    from .osm_mask import load_mask
    out = out_root / site["id"]
    try:
        mask, mmeta = load_mask(site["id"], int(site.get("mask_version", 1)), mask_dir)
    except FileNotFoundError as e:
        log(f"{site['id']}: 陸マスクがない（{e}）。検出なしで画像だけ作る")
        mask, mmeta = None, None
    req0 = client.requests
    s1 = run_s1(client, site, out, tmp_raw, mask, now, days=days_from, log=log, days_to=days_to)
    kept = keep_raw(site, out, tmp_raw, raw_dir, row_tag(site, mask is not None))
    s1_req = client.requests - req0
    deadline_left = (lambda: client.deadline - client.clock()) if client.deadline is not None else (lambda: 1e9)
    if not s2_on:
        s2s = {"stopped": "S2 なし（S1 だけの取得）"}
    elif s1.get("stopped"):
        s2s = {"stopped": "S1 で上限"}
    else:
        s2s = run_s2(client, site, out, now, lambda: client.cap_pu - client.processing_units, deadline_left, log=log)
    return {"site_id": site["id"], "mask": (mmeta or {}).get("version"), "s1": s1, "raw_kept": kept, "s2": s2s,
            "requests": client.requests, "s1_requests": s1_req, "processing_units": round(client.processing_units, 3),
            "pu_source": "header" if client.pu_reported else "estimate", "cap_pu": client.cap_pu,
            "waited_s": round(client.waited_s, 1), "retries_429": client.retries_429}


# ---------- per-orbit baselines (as Zhanjiang, sat/baseline.py) ----------

BASELINE_LOOKBACK = 365
BASELINE_MAX_SCENES = 24
BASELINE_MIN_SCENES = 24      # below this the quay count is one credibility step lower (as Zhanjiang)


class _CatalogView:
    """A client whose catalog returns fixed features (the scenes chosen beforehand); the rest is the real client."""

    def __init__(self, client, feats):
        self._c, self._feats = client, feats

    def catalog(self, *a, **k):
        return list(self._feats)

    def __getattr__(self, name):
        return getattr(self._c, name)


def baseline_port(client: LimitedClient, site: dict, mask: np.ndarray, base_dir: Path, now: datetime,
                  log=print, lookback: int = BASELINE_LOOKBACK, max_scenes: int = BASELINE_MAX_SCENES,
                  only_orbits: list[int] | None = None) -> dict:
    """Per-orbit baselines of a port with sat/baseline.py: scenes of the last ``lookback`` days whose catalog
    footprint covers at least the baseline's min_coverage of the water (others are not fetched); every orbit with
    at least 3 such scenes; the newest ``max_scenes`` per orbit, fewer if the cap would be exceeded. Nothing
    but the baseline layers is kept (no raw sigma0)."""
    from . import baseline as fbase
    grid = Grid.for_site(site)
    water = mask == 0
    feats = client.catalog(grid.bbox, now - timedelta(days=lookback), now)
    acqs = acquisitions_with_footprints(feats)
    min_cov = fbase.DEFAULTS["min_coverage"]
    keep_ids, by_orbit, dropped = set(), {}, 0
    for a in acqs:
        sh = footprint_share(a["geometries"], grid, water)
        if sh is not None and sh < min_cov:
            dropped += 1
            continue
        by_orbit.setdefault(a["relative_orbit"], []).append(a)
    orbits = sorted(o for o, v in by_orbit.items() if o is not None and len(v) >= 3
                    and (only_orbits is None or int(o) in only_orbits))
    per = cdse.estimate_pu(grid.width, grid.height, 2)
    left = client.cap_pu - client.processing_units
    n = max_scenes
    while n > 3 and per * sum(min(len(by_orbit[o]), n) for o in orbits) > left:
        n -= 1
    for o in orbits:
        for a in sorted(by_orbit[o], key=lambda a: a["acquired_utc"])[-n:]:
            keep_ids.update(a["ids"])
    counts = {str(o): len(v) for o, v in sorted(by_orbit.items(), key=lambda kv: str(kv[0]))}
    log(f"{site['id']}: 撮影 {len(acqs)}（足跡で除く {dropped}）、軌道別 {counts}、基準に使う軌道 {orbits}、"
        f"1 軌道あたり最大 {n} 撮影（{per} PU／撮影、残り {left:.1f} PU）")
    summary = {"site_id": site["id"], "mask_version": mask_tag(site), "lookback_days": lookback,
               "catalog_by_orbit": counts, "footprint_dropped": dropped, "max_scenes_used": n,
               "min_scenes": BASELINE_MIN_SCENES, "orbits": {}}
    if not orbits:
        summary["note"] = "3 撮影以上の軌道がないため基準を作らない"
        return summary
    bsite = {**site, "baseline": {"orbits": orbits, "lookback_days": lookback, "max_scenes": n,
                                  "min_scenes": BASELINE_MIN_SCENES}}
    view = _CatalogView(client, [f for f in feats if f.get("id") in keep_ids])
    res = fbase.build(view, bsite, mask, base_dir, now, max(0.0, client.cap_pu - client.processing_units) + 0.5, log=log)
    if lookback > BASELINE_LOOKBACK:  # record that scenes older than 12 months are mixed in
        for o, r in res.get("orbits", {}).items():
            pj = fbase.paths(bsite, int(o), base_dir)["json"]
            if r.get("built") and pj.exists():
                meta = json.loads(pj.read_text(encoding="utf-8"))
                old = [x["acquired_utc"] for x in meta.get("scene_list", [])
                       if x.get("used") and x["acquired_utc"] < cdse.iso(now - timedelta(days=BASELINE_LOOKBACK))]
                meta["period"] = {"lookback_days": lookback, "first_utc": meta.get("first_utc"), "last_utc": meta.get("last_utc"),
                                  "older_than_12_months": len(old),
                                  "note": (f"直近 12 か月では {BASELINE_MIN_SCENES} 撮影に届かないため、さかのぼりを {lookback} 日に延ばし、"
                                           f"13〜24 か月前の撮影 {len(old)} 枚を含む（最新から順に {max_scenes} 撮影まで）。"
                                           "その間に岸壁・桟橋が変わっていれば基準がずれる")}
                pj.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                r["older_than_12_months"] = len(old)
    for o, r in res.get("orbits", {}).items():
        r = dict(r)
        r.pop("stats", None)
        r["enough"] = bool(r.get("built") and r.get("scenes", r.get("used", 0)) >= BASELINE_MIN_SCENES)
        summary["orbits"][o] = r
    return summary


def usage_row(res: dict, now: datetime) -> dict:
    return {"run_utc": cdse.iso(now), "site_id": res["site_id"], "requests": res["requests"],
            "processing_units": res["processing_units"], "pu_source": res["pu_source"],
            "s1_scenes": res["s1"].get("counted", 0), "s2_days": res["s2"].get("days", 0),
            "s2_clear": res["s2"].get("clear", 0), "s1_processing_units": res["s1"].get("processing_units", 0),
            "s2_processing_units": res["s2"].get("processing_units", 0), "cap_pu": res["cap_pu"],
            "archive_version": ARCHIVE_VERSION}


def trend(out: Path, tag: str | None = None) -> str:
    """Monthly median of the candidates (counted scenes of one mask version) — the rough series of the summary."""
    rows = [r for r in read_csv(out / "s1_scenes.csv") if r.get("result") == "counted" and r.get("candidates") != ""
            and (tag is None or r.get("mask_version") == tag)]
    by: dict = {}
    for r in rows:
        by.setdefault(r["acquired_utc"][:7], []).append(int(r["candidates"]))
    return " ".join(f"{m[2:]}:{int(np.median(v))}({len(v)})" for m, v in sorted(by.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PLAN ports: Sentinel-1/2 archive (probe)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--ports", type=Path, default=PORTS_FILE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cap", type=float, default=PORTS_CAP)
    p.add_argument("--usage", type=Path, nargs="*", default=[], help="other usage CSVs of this month (sat, ocean)")
    p.add_argument("--ports-usage", type=Path, nargs="*", default=[],
                   help="usage CSVs of earlier runs of this archive (docs/data/sat/ports/*/usage.csv)")
    p.add_argument("--ocean-cap", type=float,
                   help="ocean's monthly cap: reserve = cap - the ocean usage already in --usage files named other-*")
    p.add_argument("--reserve", type=float, default=0.0,
                   help="PU announced by other jobs but not in their usage CSV yet (counted in the 9,000 PU guard)")
    p.add_argument("--out-root", type=Path, default=OUT_DIR, help="earlier results (only new work is planned)")
    p.add_argument("--mask-dir", type=Path, default=OUT_DIR / "masks")
    p.add_argument("--max-pu", type=float, help="cap of this run (e.g. a backfill), inside the ports cap")
    p.add_argument("--only", nargs="*", help="these ports, in this order of priority (ordered allocation)")
    p.add_argument("--proportional", action="store_true", help="with --only: share the cap in proportion to the estimates")
    m = sub.add_parser("mask")
    m.add_argument("--ports", type=Path, default=PORTS_FILE)
    m.add_argument("--mask-dir", type=Path, required=True)
    m.add_argument("--port", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--ports", type=Path, default=PORTS_FILE)
    r.add_argument("--port", required=True)
    r.add_argument("--cap", type=float, required=True)
    r.add_argument("--req", type=int, default=2000)
    r.add_argument("--out", type=Path, default=OUT_DIR)
    r.add_argument("--raw-dir", type=Path, required=True)
    r.add_argument("--tmp", type=Path, required=True)
    r.add_argument("--mask-dir", type=Path, required=True)
    r.add_argument("--pu-per-min", type=float, default=90)
    r.add_argument("--req-per-min", type=int, default=90)
    r.add_argument("--max-minutes", type=float, default=300)
    r.add_argument("--summary", type=Path)
    for q in (p, r):
        q.add_argument("--days-from", type=int, default=LOOKBACK_DAYS, help="oldest acquisition, days before now")
        q.add_argument("--days-to", type=int, default=0, help="newest acquisition, days before now")
        q.add_argument("--s1-only", action="store_true")
    b = sub.add_parser("baseline")
    b.add_argument("--ports", type=Path, default=PORTS_FILE)
    b.add_argument("--only", nargs="+", required=True, help="ports in this order of priority")
    b.add_argument("--cap", type=float, required=True, help="PU of this run (already inside the ports cap and the guard)")
    b.add_argument("--req", type=int, default=1500)
    b.add_argument("--mask-dir", type=Path, default=OUT_DIR / "masks")
    b.add_argument("--base-dir", type=Path, default=OUT_DIR / "baselines")
    b.add_argument("--out", type=Path, default=OUT_DIR)
    b.add_argument("--pu-per-min", type=float, default=150)
    b.add_argument("--req-per-min", type=int, default=150)
    b.add_argument("--max-minutes", type=float, default=300)
    b.add_argument("--summary", type=Path)
    b.add_argument("--lookback", type=int, default=BASELINE_LOOKBACK)
    b.add_argument("--orbits", nargs="*", default=[],
                   help="ID:orbit[,orbit][:max scenes], e.g. SYL:157:26 QGZ:69,171 (only these orbits of these ports)")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    ports = load_ports(args.ports)

    if args.cmd == "baseline":
        from .osm_mask import load_mask
        client = LimitedClient(args.cap, args.req, args.pu_per_min, args.req_per_min,
                               deadline=time.monotonic() + args.max_minutes * 60)
        byid = {p["id"]: p for p in ports}
        out = {"run_utc": cdse.iso(now), "cap_pu": args.cap, "ports": []}
        for pid in args.only:
            site = byid[pid]
            pu0, req0 = client.processing_units, client.requests
            try:
                mask, _m = load_mask(site["id"], int(site.get("mask_version", 1)), args.mask_dir)
                spec = {x.split(":")[0]: x.split(":")[1:] for x in args.orbits}
                sp = spec.get(pid)
                s = baseline_port(client, site, mask, args.base_dir, now, log=lambda t: print(t, file=sys.stderr, flush=True),
                                  lookback=args.lookback,
                                  max_scenes=int(sp[1]) if sp and len(sp) > 1 else BASELINE_MAX_SCENES,
                                  only_orbits=[int(o) for o in sp[0].split(",")] if sp else None)
            except (BudgetStop, FileNotFoundError, cdse.CDSEError) as e:
                if isinstance(e, cdse.CredentialError) and not isinstance(e, BudgetStop):
                    raise
                s = {"site_id": pid, "error": str(e)[:300]}
                print(f"{pid}: {e}", file=sys.stderr)
            s.update(processing_units=round(client.processing_units - pu0, 3), requests=client.requests - req0)
            out["ports"].append(s)
            d = args.out / pid
            write_csv(d / "usage.csv", read_csv(d / "usage.csv") + [{
                "run_utc": cdse.iso(now), "site_id": pid, "requests": s["requests"], "processing_units": s["processing_units"],
                "pu_source": "header" if client.pu_reported else "estimate", "s1_processing_units": s["processing_units"],
                "cap_pu": args.cap, "archive_version": "baseline-bl-1"}], USAGE_FIELDS)
        out.update(processing_units=round(client.processing_units, 3), requests=client.requests)
        text = json.dumps(out, ensure_ascii=False, indent=1)
        if args.summary:
            args.summary.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.cmd == "mask":
        from .osm_mask import build_mask, fetch_osm, save_mask
        rc = 0
        for site in ports:
            if args.port and site["id"] not in args.port:
                continue
            from .osm_mask import mask_paths
            if mask_paths(site["id"], int(site.get("mask_version", 1)), args.mask_dir)[0].exists():
                continue  # keep the mask of the earlier run (the series stays on one mask)
            grid = Grid.for_site(site)
            for attempt in range(3):
                try:
                    osm = fetch_osm(grid)
                    mask, info = build_mask(grid, osm)
                    meta = save_mask(site, grid, mask, info, osm, int(site.get("mask_version", 1)), args.mask_dir)
                    print(json.dumps({k: meta[k] for k in meta if k != "query"}, ensure_ascii=False), file=sys.stderr)
                    break
                except Exception as e:
                    print(f"{site['id']}: OSM のマスクに失敗（{attempt + 1} 回目）{str(e)[:200]}", file=sys.stderr)
                    time.sleep(30 * (attempt + 1))
            else:
                rc = 1
            time.sleep(5)
        return rc

    if args.cmd == "plan":
        month = now.strftime("%Y-%m")
        other_pu, other_req = month_usage(args.usage, month)
        if args.ocean_cap is not None:
            ocean_pu, _r = month_usage([u for u in args.usage if u.name.startswith("other-")], month)
            args.reserve = max(args.reserve, round(args.ocean_cap - ocean_pu, 1))
        done_pu, done_req = month_usage(args.ports_usage, month)
        cap = min(args.cap - done_pu, GLOBAL_CAP - GLOBAL_MARGIN - other_pu - args.reserve - done_pu)
        if args.max_pu is not None:
            cap = min(cap, args.max_pu)
        if args.only:
            order = {pid: i for i, pid in enumerate(args.only)}
            ports = sorted([p for p in ports if p["id"] in order], key=lambda p: order[p["id"]])
        req_cap = REQUEST_CAP - REQUEST_MARGIN - other_req - done_req
        client = cdse.Client()
        res = plan(client, ports, now, max(0.0, cap), max(0, req_cap - 200),
                   log=lambda s: print(s, file=sys.stderr), out_root=args.out_root, mask_dir=args.mask_dir,
                   days_from=args.days_from, days_to=args.days_to, s2_on=not args.s1_only, ordered=bool(args.only) and not args.proportional)
        res.update(other_usage_pu=round(other_pu, 1), reserve_pu=args.reserve, other_usage_requests=other_req,
                   ports_done_pu=round(done_pu, 1), global_cap=GLOBAL_CAP, ports_cap=args.cap, max_pu=args.max_pu,
                   window_days=[args.days_from, args.days_to], s1_only=args.s1_only, order=args.only)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "ports"}, ensure_ascii=False))
        return 0

    site = next(s for s in ports if s["id"] == args.port)
    client = LimitedClient(args.cap, args.req, args.pu_per_min, args.req_per_min,
                           deadline=time.monotonic() + args.max_minutes * 60)
    res = run_port(client, site, args.out, args.raw_dir, args.tmp, args.mask_dir, now,
                   log=lambda s: print(s, file=sys.stderr, flush=True), days_from=args.days_from,
                   days_to=args.days_to, s2_on=not args.s1_only)
    res["trend"] = trend(args.out / site["id"], mask_tag(site))
    out = args.out / site["id"]
    write_csv(out / "usage.csv", read_csv(out / "usage.csv") + [usage_row(res, now)], USAGE_FIELDS)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if args.summary:
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
