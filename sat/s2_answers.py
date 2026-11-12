"""Review sets and answers for the owner's eye check of the quay sections (est-2 PR-2, design 1.4 / PR-2).

The check that calibrated Zhanjiang (2026-10-30: the owner looked at Sentinel-2 10 m natural colour on clear
days and said, per quay section and day, how many rows of ships and how many - or that it cannot be judged)
is repeated for the ports. Nothing here connects to CDSE (PU 0): the pictures are the Sentinel-2 images already
archived (sat/port_archive.py, natural colour, PU 0 to look at them), the sections are the same pieces of the
OSM quay lines as in Zhanjiang (sat/s2_labels.py sections()).

    python -m sat.s2_answers prepare --site SYL --archive-ref origin/probe/ports-s2ext-ops3
    python -m sat.s2_answers prepare --site LSN --archive-ref origin/probe/ports-s2ext-ops3 --naval-only   # naval lines only (A-0)
    python -m sat.s2_answers prepare --site ZJG --oos                    # out-of-sample days (from the freeze on)
    python -m sat.s2_answers merge   --site SYL --file ~/Downloads/SYL-s2-answers.csv
    python -m sat.s2_answers plan                                        # S1 re-fetch list and PU, every port with a review set
    python -m sat.s2_answers prepare --s1 --site QGZ --raw-ref origin/probe/sat-s1-recount-ops4   # the S1 check (same-time truth)
    python -m sat.s2_answers merge   --s1 --site QGZ --file ~/Downloads/QGZ-s1-answers.csv

prepare   writes docs/data/sat/labels/<site>/review.json (days, candidate sections as pixel rings, the S1 scenes
          each day can be paired with), the pictures docs/data/sat/labels/<site>/img/<date>.webp (as archived,
          with the credit strip) and s1_recount_plan.json. The page docs/sat/label/ shows them.
merge     adds the answers file the page downloads to docs/data/sat/labels/<site>-s2-answers.csv (ports) or
          <site>-oos-answers.csv (Zhanjiang, out of sample). A later answer replaces an earlier one of the same
          section and day. Only sections and days of the review set are accepted.
plan      what has to be fetched again from Sentinel-1 for the answered days (not done here): scenes within
          +-4 days of an answered day, of the orbits with a baseline, and the PU estimate (cdse.estimate_pu).

Candidate sections: the pieces whose stored Sentinel-1 stretches change between scenes come first (they
carry the information on how many rows abreast), then the ones occupied every time, then never; at most
two thirds of the picks from one OSM way is avoided by a round robin over the ways.

The S1 check (``--s1``; design 設計_同日の正解と港の誤り検品 2026-11 §2, PU 0): the unit is one Sentinel-1 scene x one
section. The raw chips of the re-count (probe-raw/s1-recount/<site>/s1-<stamp>.npz, VV and VH in dB, on a probe
branch: ``--raw-ref``, or a directory: ``--raw-dir``) are drawn as the same false colour as the published pictures
(sat/color.render, s1fc-1) into labels/<site>/s1/<stamp>.webp, and review-s1.json lists the scenes (acquired_utc,
relative_orbit, image, the nearest Sentinel-2 day of the review set within +-4 days with its time difference) and the
same sections as review.json, each with the mechanical "readable in S1 / needs S2" mark per orbit (§2.4: the baseline
VV p10 of the inner band under READABLE["p10_max_db"], p90 - p10 at least READABLE["spread_min_db"], land under
READABLE["land_max"] of the pixels off the edge; a blind section is never readable). The page docs/sat/label/?s1=1
shows S1 and S2 side by side, asks present / absent / cannot judge, and pre-fills the owner's S2 answer when the S2
day is within S1_PREFILL_H. ``merge --s1`` writes labels/<site>-s1-answers.csv, key (section_id, s1_acquired_utc).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, quay, quay_est2 as e2, quay_lines
from .grid import Grid, m_per_deg
from .s1_recount import RAW_SUBDIR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
REVIEW_VERSION = "s2rev-1"
N_DAYS = 8
N_SECTIONS = 24
MIN_DAYS = 3
S1_WINDOW_DAYS = 4.0
FRAME_M = 120.0              # the blue frame of the page: off the water's edge, as in the Zhanjiang check
STRIP_H = 66                 # the legend strip below the chip in the archived WebP (s2.render)
ANSWER_FIELDS = ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c", "verified_color",
                 "verified_by", "verified_note"]
SCENE_FIELDS = ["site_id", "date", "warships_60", "note"]
CHOICES = {"unsure": ("0", "", ""), "none": ("1", "0", "0"), "1": ("1", "1", "1"), "2": ("1", "1", "2"), "3": ("1", "1", "3")}
# the S1 check (same-time truth)
S1_REVIEW_VERSION = "s1rev-1"
S1_ANSWER_FIELDS = ["site_id", "section_id", "s1_acquired_utc", "relative_orbit", "s2_date", "dt_h", "verified", "verified_occupied",
                    "verified_by", "verified_note"]
S1_PREFILL_H = 48.0          # the owner's S2 answer is the default when the S2 day is within this of the S1 scene
S1_S2_WINDOW_DAYS = 4.0      # the nearest S2 day shown beside the S1 scene
READABLE = {"p10_max_db": -14.0, "spread_min_db": 8.0, "land_max": 0.4, "blind_p10_db": quay.BLIND_P10_DB, "blind_share_max": 0.5}
SEC_PER_CELL = {"prefilled": 5, "read": 12}   # seconds per cell for the time estimate


# ---------------------------------------------------------------- helpers
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def answers_path(out: Path, site_id: str, oos: bool = False) -> Path:
    return out / "labels" / (f"{site_id}-oos-answers.csv" if oos else f"{site_id}-s2-answers.csv")


def review_path(out: Path, site_id: str, oos: bool = False) -> Path:
    return out / "labels" / site_id / ("review-oos.json" if oos else "review.json")


class Archive:
    """Where the archived pictures and catalogues of a port are: a git ref (the probe branch) or a directory."""

    def __init__(self, ref: str | None = None, directory: Path | None = None, repo: Path = ROOT):
        self.ref, self.dir, self.repo = ref, directory, repo

    def read(self, site_id: str, rel: str) -> bytes | None:
        path = f"docs/data/sat/ports/{site_id}/{rel}"
        if self.dir is not None:
            p = self.dir / site_id / rel
            return p.read_bytes() if p.exists() else None
        if self.ref:
            r = subprocess.run(["git", "show", f"{self.ref}:{path}"], cwd=self.repo, capture_output=True)
            return r.stdout if r.returncode == 0 else None
        return None

    def ports_site(self, site_id: str) -> dict | None:
        """The port's entry of sat/ports.json at the archive's ref (the range the pictures were taken for)."""
        if not self.ref:
            return None
        r = subprocess.run(["git", "show", f"{self.ref}:sat/ports.json"], cwd=self.repo, capture_output=True)
        if r.returncode != 0:
            return None
        return next((x for x in json.loads(r.stdout.decode("utf-8")).get("sites", []) if x.get("id") == site_id), None)

    def csv(self, site_id: str, name: str) -> list[dict]:
        b = self.read(site_id, name)
        return list(csv.DictReader(io.StringIO(b.decode("utf-8")))) if b else []


# ---------------------------------------------------------------- the grid of a picture
def image_grid(site_grid: Grid, chip_w: int, chip_h: int) -> Grid:
    """The grid of an archived picture: same west, south and pixel size as the site's current grid, the height
    of the picture (an older, smaller range keeps its south-west corner: SLP v1 stops at 18.230 N).
    A range narrower than the strip below the picture (sat/color.py LEGEND_MIN_WIDTH, 300 px: Jianggezhuang is 288 px wide,
    the ports of 2026-11-12 about 220) is archived on a canvas as wide as the strip, the picture at its top left: the
    grid keeps the site's width, and the page reads the pixels from the top left corner."""
    from .color import LEGEND_MIN_WIDTH
    narrow = site_grid.width < LEGEND_MIN_WIDTH == chip_w
    if chip_w != site_grid.width and not narrow:
        raise ValueError(f"picture width {chip_w} != grid width {site_grid.width}")
    dlat = (site_grid.north - site_grid.south) / site_grid.height
    return Grid(site_grid.west, site_grid.south, site_grid.east, round(site_grid.south + dlat * chip_h, 6), site_grid.width, chip_h)


def webp_size(data: bytes) -> tuple[int, int]:
    from PIL import Image
    return Image.open(io.BytesIO(data)).size


# ---------------------------------------------------------------- days
def eligible_days(s2_rows: list[dict], s1_rows: list[dict], orbits: set, mask_tags: list[str], freeze: str,
                  oos: bool = False) -> list[dict]:
    """Clear Sentinel-2 days (a mask tag in ``mask_tags``, the first one that has the day), before ``freeze`` (or from it
    on when ``oos``), with the Sentinel-1 scenes of ``orbits`` within +-4 days (for ports; oos: no S1 needed here)."""
    seen: dict = {}
    for tag in mask_tags:
        for r in s2_rows:
            if r.get("mask_version") == tag and r.get("result") == "clear" and r.get("image") and r["date"] not in seen:
                if (r["date"] >= freeze) == oos:
                    seen[r["date"]] = r
    s1 = [r for r in s1_rows if str(r.get("relative_orbit")) in orbits and r.get("result") in ("counted", "ok", None, "")]
    out = []
    for d, r in sorted(seen.items()):
        t = cdse.parse_time(r["acquired_utc"])
        near = []
        for a in s1:
            dt = (cdse.parse_time(a["acquired_utc"]) - t).total_seconds() / 3600.0
            if abs(dt) <= S1_WINDOW_DAYS * 24 and (a["acquired_utc"][:10] < freeze) == (not oos):
                near.append({"acquired_utc": a["acquired_utc"], "relative_orbit": str(a["relative_orbit"]), "dt_h": round(dt, 1)})
        near.sort(key=lambda x: abs(x["dt_h"]))
        if near or oos:
            out.append({**r, "s1_near": near})
    return out


def pick_days(days: list[dict], n: int = N_DAYS) -> list[dict]:
    """``n`` days spread over the period (evenly by position), the nearest S1 scene first among equals."""
    if len(days) <= n:
        return days
    idx = sorted({int(round(i)) for i in np.linspace(0, len(days) - 1, n)})
    return [days[i] for i in idx]


# ---------------------------------------------------------------- sections
def stored_stretch_frequency(out: Path, site_id: str, secs: list[dict], lat0: float) -> dict:
    """{section_id: (occupied scenes, scenes)} from the stored Sentinel-1 stretches (quay/<site>/*.geojson, or the
    shared quay/*.geojson for Zhanjiang): a stretch of the same OSM way and side within 20 m of the section."""
    month = "[0-9][0-9][0-9][0-9]-[0-9][0-9].geojson"
    files = sorted((out / "quay" / site_id).glob(month)) if (out / "quay" / site_id).is_dir() else sorted((out / "quay").glob(month))
    feats = []
    for p in files:
        for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            if pr.get("site_id") == site_id:
                feats.append({**pr, "xy": e2._xy(f["geometry"]["coordinates"], lat0)})
    scenes = sorted({f["acquired_utc"] for f in feats})
    cq = [r["acquired_utc"] for r in read_csv(out / "quay_counts.csv") if r["site_id"] == site_id]
    total = len(set(cq) | set(scenes)) or 0
    res = {}
    for s in secs:
        sxy = e2._xy(s["line"], lat0)
        hit = set()
        for f in feats:
            if f.get("osm_way") != s["way_id"] or e2._side(f) != int(s["side"]):
                continue
            if float(e2._min_dist(sxy, f["xy"]).min()) <= e2.SECTION_MATCH_M:
                hit.add(f["acquired_utc"])
        res[s["section_id"]] = (len(hit), total)
    return res


def pick_sections(secs: list[dict], freq: dict, n: int = N_SECTIONS, allowed=None) -> list[dict]:
    """Sections that change between scenes first (p(1-p)), then the always occupied, then the never occupied; a round
    robin over the OSM ways so that one long quay does not take all picks."""
    def score(s):
        k, t = freq.get(s["section_id"], (0, 0))
        p = k / t if t else 0.0
        return (round(4 * p * (1 - p), 3), p, s["length_m"])
    pool = [s for s in secs if allowed is None or allowed(s)]
    by_way: dict = {}
    for s in sorted(pool, key=score, reverse=True):
        by_way.setdefault(s["way_id"], []).append(s)
    ways = sorted(by_way, key=lambda w: score(by_way[w][0]), reverse=True)
    out = []
    while len(out) < n and any(by_way.values()):
        for w in ways:
            if by_way[w] and len(out) < n:
                out.append(by_way[w].pop(0))
    return sorted(out, key=lambda s: (str(s["way_id"]), s["side"], s["section_id"]))


def section_rings(geom, sc: dict, grid: Grid) -> dict:
    """The section's line and the frame (FRAME_M off the water's edge along the normals) in lon/lat."""
    pr = geom.profiles[sc["profile"]]
    idx = sc["idx"]
    mlat, mlon = m_per_deg(float(np.mean(pr["lat"][idx])))
    far = []
    for i in idx:
        a = math.radians(float(pr["naz"][i]))
        far.append([round(float(pr["lon"][i]) + FRAME_M * math.sin(a) / mlon, 6), round(float(pr["lat"][i]) + FRAME_M * math.cos(a) / mlat, 6)])
    near = sc["line"]
    return {"line": near, "frame": near + far[::-1]}


# ---------------------------------------------------------------- prepare
def site_config(site_id: str) -> dict:
    from .run import load_sites
    return next(s for s in load_sites() if s["id"] == site_id)


def prepare(site_id: str, out: Path = OUT, archive: Archive | None = None, oos: bool = False,
            n_days: int = N_DAYS, n_sections: int = N_SECTIONS, site: dict | None = None, log=print,
            naval_only: bool = False) -> dict:
    """``naval_only`` (plan A, A-0): only the sections on a line classed naval by naval-1 (sat/berth_class/<site>.json, with
    the sections the class file adds) are candidates, so that the owner's answers all serve the presence evaluation
    (sat/eval_presence.py) of the naval quays."""
    from .osm_mask import MASK_DIR, load_mask
    from .run import site_sectors, sector_of as _sector_of
    from . import berths, s2_labels
    site = site or site_config(site_id)
    grid = Grid.for_site(site)
    q = site.get("quay") or {}
    freeze = q.get("label_freeze_utc") or e2.FREEZE_UTC
    mver = int(site.get("mask_version", 1))
    mask, _ = load_mask(site_id, mver, MASK_DIR)
    lines = quay_lines.load(site_id, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), MASK_DIR)
    geom = quay.Geometry(grid, mask, lines)
    sectors = site_sectors(site)
    secs = s2_labels.sections(geom, lambda lo, la: _sector_of(sectors, lo, la))
    orbits = {str(o) for o in ((site.get("baseline") or {}).get("orbits") or [])}
    # the pictures
    if site_id == "ZJG":
        s2_rows = read_csv(out / "s2_scenes.csv")
        s2_rows = [r for r in s2_rows if r["site_id"] == "ZJG"]
        s1_rows = []

        def read_img(r): return (out / r["image"]).read_bytes() if (out / r["image"]).exists() else None
        tags = [f"ZJG-land-v{mver}"]
    else:
        archive = archive or Archive()
        s2_rows = archive.csv(site_id, "s2_scenes.csv")
        s1_rows = archive.csv(site_id, "s1_scenes.csv")

        def read_img(r): return archive.read(site_id, r["image"])
        tags = [f"{site_id}-land-v{mver}"] + sorted({r["mask_version"] for r in s2_rows} - {f"{site_id}-land-v{mver}"}, reverse=True)
    elig = eligible_days(s2_rows, s1_rows, orbits, tags, freeze, oos) if site_id != "ZJG" else \
        [{**r, "s1_near": []} for r in eligible_days(s2_rows, [], orbits, tags, freeze, oos)] if not oos else \
        eligible_days(s2_rows, [], orbits, tags, freeze, True)
    days_info, imgs = [], {}
    for r in pick_days(elig, n_days):
        data = read_img(r)
        if data is None:
            log(f"{site_id} {r['date']}: picture not found, skipped")
            continue
        w, h = webp_size(data)
        try:
            g = image_grid(grid, w, h - STRIP_H)
        except ValueError as e:   # another range than today's grid: the range the archive was taken for, when it fits
            ag = None
            if archive is not None and site_id != "ZJG":
                ps = archive.ports_site(site_id)
                ag = Grid.for_site(ps) if ps and ps.get("bbox") else None
            if ag is not None and (ag.width, ag.height) == (w, h - STRIP_H):
                g = ag
            else:
                log(f"{site_id} {r['date']}: {e}, skipped")
                continue
        imgs[r["date"]] = data
        days_info.append({"date": r["date"], "acquired_utc": r["acquired_utc"], "image": f"img/{r['date']}.webp",
                          "grid": {"west": g.west, "south": g.south, "east": g.east, "north": g.north, "width": g.width, "height": g.height},
                          "tile_cloud_pct": r.get("tile_cloud_pct"), "s1_near": r["s1_near"]})
    if len(days_info) < (1 if oos else MIN_DAYS):
        log(f"{site_id}: {len(days_info)} usable day(s)" + ("" if oos else f" (< {MIN_DAYS}): no review set"))
        if not days_info:
            return {"site_id": site_id, "days": 0}
    bnd = {"west": max(d["grid"]["west"] for d in days_info), "south": max(d["grid"]["south"] for d in days_info),
           "east": min(d["grid"]["east"] for d in days_info), "north": min(d["grid"]["north"] for d in days_info)} if days_info else None
    pad = FRAME_M / 111000 + 1e-4   # the frame (and a margin) must lie in every picture

    def inside(sc):
        if bnd is None:
            return True
        lo, la = [p[0] for p in sc["line"]], [p[1] for p in sc["line"]]
        return (min(lo) - pad >= bnd["west"] and max(lo) + pad <= bnd["east"] and min(la) - pad >= bnd["south"] and max(la) + pad <= bnd["north"])
    lat0 = float(q.get("est2_lat0") or site.get("center_lat") or (grid.south + grid.north) / 2)
    freq = stored_stretch_frequency(out, site_id, secs, lat0)
    if days_info and len(days_info) < n_days:   # fewer days: more sections, so that the owner's work stays about the same
        n_sections = min(40, int(round(n_sections * n_days / len(days_info))))
    allowed = inside
    if naval_only:
        b = berths.Berths(site, lines, naval=True)
        secs = [sc for sc in secs if b.line_class(sc["way_id"], sc["lon"], sc["lat"]) == "naval"]
    chosen = pick_sections(secs, freq, n_sections, allowed=allowed)
    out_secs = []
    for sc in chosen:
        ring = section_rings(geom, sc, grid)
        k, t = freq.get(sc["section_id"], (0, 0))
        out_secs.append({"id": sc["section_id"], "way": sc["way_id"], "kind": sc["kind"], "side": sc["side"], "sector": sc["sector"],
                         "length_m": sc["length_m"], "line": ring["line"], "frame": ring["frame"],
                         "s1_occupied": k, "s1_scenes": t})
    ddir = out / "labels" / site_id
    (ddir / "img").mkdir(parents=True, exist_ok=True)
    for old in (ddir / "img").glob("*.webp") if not oos else []:
        if old.stem not in imgs:
            old.unlink()
    for d, data in imgs.items():
        (ddir / "img" / f"{d}.webp").write_bytes(data)
    n_cells = len(out_secs) * len(days_info)
    doc = {"version": REVIEW_VERSION, "site_id": site_id, "name": site.get("name"), "kind": "oos" if oos else "calibration",
           "freeze_utc": freeze, "frame_m": FRAME_M, "zoom": 8, "strip_h": STRIP_H,
           "days": days_info, "sections": out_secs, "cells": n_cells,
           "minutes_estimate": round(n_cells * 8 / 60),
           "answers_file": answers_path(out, site_id, oos).name,
           "credit": "Contains modified Copernicus Sentinel data 2025, 2026", "attribution": "岸壁の線：(c) OpenStreetMap contributors（ODbL 1.0）",
           "selection": {"sections": f"保存済みの S1 の区間（同じ OSM の線・向き 20 m 以内）が撮影ごとに変わる区間を先に、次にいつも占有、最後にいつも空。OSM の線ごとに順番に取る（最大 {n_sections}）",
                         "days": f"S2 の晴れの日（凍結 {freeze} {'以後' if oos else 'より前'}）のうち" + ("" if oos else "、前後 4 日以内に基準のある軌道の S1 があるものから、期間に等間隔で最大 " + str(n_days) + " 日"),
                         "candidate_sections": len(secs), "eligible_days": len(elig),
                         "naval_only": naval_only,
                         **({"naval_only_note": "naval-1 で naval の線（設定が足す区間を含む）の区間だけを候補にした（在/不在の評価票 eval/presence.json の対象に合わせる）"} if naval_only else {})}}
    review_path(out, site_id, oos).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    stored = {r["acquired_utc"] for r in read_csv(out / "quay_counts.csv") if r["site_id"] == site_id}
    plan = s1_plan(doc, site, grid, stored) if not oos else None
    if plan is not None:
        (ddir / "s1_recount_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"{site_id}: {len(days_info)} days x {len(out_secs)} sections = {n_cells} cells (about {doc['minutes_estimate']} min); "
        f"images {sum(len(v) for v in imgs.values()) // 1024} KiB")
    return {"site_id": site_id, "days": len(days_info), "sections": len(out_secs), "cells": n_cells, "plan": plan}


# ---------------------------------------------------------------- the S1 re-fetch list
def s1_plan(review: dict, site: dict, grid: Grid | None = None, stored: set | None = None) -> dict:
    """The Sentinel-1 scenes the answered days need again (per-section features are not stored): one scene per day,
    the nearest of the baseline orbits within +-4 days (as sat/s2_labels.nearest_s1). Not fetched here."""
    grid = grid or Grid.for_site(site)
    est = cdse.estimate_pu(grid.width, grid.height, 3)
    stored = stored or set()
    scenes: dict = {}
    per_day = []
    for d in review["days"]:
        a = (d.get("s1_near") or [None])[0]
        per_day.append({"date": d["date"], "s1": None if a is None else a["acquired_utc"], "dt_h": None if a is None else a["dt_h"]})
        if a is not None:
            scenes.setdefault(a["acquired_utc"], {"acquired_utc": a["acquired_utc"], "relative_orbit": a["relative_orbit"], "days": []})["days"].append(d["date"])
    for sc in scenes.values():
        sc["stored_stretches"] = sc["acquired_utc"] in stored
    n_new = sum(1 for sc in scenes.values() if not sc["stored_stretches"])
    return {"site_id": review["site_id"], "version": REVIEW_VERSION, "scenes": sorted(scenes.values(), key=lambda x: x["acquired_utc"]),
            "per_day": per_day, "pu_per_scene": round(est, 3), "pu_total": round(est * len(scenes), 2),
            "pu_if_stored_stretches_used": round(est * n_new, 2),
            "note": ("取り直しは python -m sat.s2_labels 系の S1 だけの取得（この PR にはまだ無い）で、答えの日の前後 4 日以内の撮影を 1 枚ずつ。"
                     "stored_stretches＝その撮影の岸壁の片（区間の奥行きなど）は quay/<地点>/ に保存済み（取り直さずに突き合わせを組める。組む処理は未実装なので pu_total には含めてある）。S2 の自動ラベルは使わない（人の答えだけを正解にする）ので S2 の取得（約 2.7〜3.2 PU/日）は要らない。PU は cdse.estimate_pu の見積もり（3 帯、LIA なし）"),
            "needs_cdse": True, "fetched": False}


def plan_all(out: Path = OUT) -> dict:
    res = {"scenes": 0, "pu": 0.0, "sites": {}}
    for p in sorted((out / "labels").glob("*/review.json")):
        rv = json.loads(p.read_text(encoding="utf-8"))
        pl = (p.parent / "s1_recount_plan.json")
        if pl.exists():
            d = json.loads(pl.read_text(encoding="utf-8"))
            res["sites"][rv["site_id"]] = {"scenes": len(d["scenes"]), "pu": d["pu_total"], "days": len(rv["days"])}
            res["scenes"] += len(d["scenes"])
            res["pu"] += d["pu_total"]
    res["pu"] = round(res["pu"], 2)
    return res


# ---------------------------------------------------------------- the S1 check (same-time truth, PU 0)
def s1_answers_path(out: Path, site_id: str) -> Path:
    return out / "labels" / f"{site_id}-s1-answers.csv"


def s1_review_path(out: Path, site_id: str) -> Path:
    return out / "labels" / site_id / "review-s1.json"


class RawStore:
    """The saved Sentinel-1 chips of the re-count (sat/s1_recount.save): <raw>/<site>/s1-<stamp>.npz + .json, on a git
    ref (the probe branch, read with git show: nothing is checked out) or in a directory."""

    def __init__(self, ref: str | None = None, directory: Path | None = None, repo: Path = ROOT, subdir: str = RAW_SUBDIR):
        self.ref, self.dir, self.repo, self.subdir = ref, directory, repo, subdir

    def _bytes(self, site_id: str, name: str) -> bytes | None:
        if self.dir is not None:
            p = self.dir / site_id / name
            return p.read_bytes() if p.exists() else None
        if self.ref:
            r = subprocess.run(["git", "show", f"{self.ref}:{self.subdir}/{site_id}/{name}"], cwd=self.repo, capture_output=True)
            return r.stdout if r.returncode == 0 else None
        return None

    def stems(self, site_id: str) -> list[str]:
        """The chips of a site (s1-<stamp>), sorted."""
        if self.dir is not None:
            return sorted(p.stem for p in (self.dir / site_id).glob("s1-*.npz")) if (self.dir / site_id).is_dir() else []
        if self.ref:
            r = subprocess.run(["git", "ls-tree", "--name-only", f"{self.ref}:{self.subdir}/{site_id}"], cwd=self.repo, capture_output=True)
            if r.returncode != 0:
                return []
            return sorted(n[:-4] for n in r.stdout.decode("utf-8").split() if n.startswith("s1-") and n.endswith(".npz"))
        return []

    def load(self, site_id: str, stem: str):
        """(vv dB, vh dB, meta) or None."""
        b = self._bytes(site_id, stem + ".npz")
        if b is None:
            return None
        z = np.load(io.BytesIO(b))
        m = self._bytes(site_id, stem + ".json")
        return z["vv"].astype(np.float64), z["vh"].astype(np.float64), (json.loads(m.decode("utf-8")) if m else {})


def readability(sec: dict, geom, mask, base: dict, rules: dict = READABLE) -> dict:
    """The mechanical mark of one section under one orbit's baseline (§2.4): the baseline VV p10 of the inner band (its
    median over the section) darker than ``p10_max_db``, the spread p90 - p10 at least ``spread_min_db``, land (the land
    mask over the depth pixels 5-120 m off the edge) under ``land_max``, and the blind share (samples whose inner p10 is above
    ``blind_p10_db``, as quay.blind_stretches) under ``blind_share_max``. Returns the figures and ``readable``."""
    import warnings
    from .osm_mask import WATER
    pr = geom.profiles[sec["profile"]]
    idx = np.asarray(sec["idx"])
    ni = pr["n_inner"]
    rr, cc, valid = pr["rr"][idx, :ni], pr["cc"][idx, :ni], pr["valid"][idx, :ni]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p10 = np.where(valid, base["vv_p10"][rr, cc], np.nan)
        p90 = np.where(valid, base["vv_p90"][rr, cc], np.nan)
        smax = np.nanmax(p10, 1)
        blind = float(np.mean(np.isfinite(smax) & (smax > rules["blind_p10_db"]))) if len(smax) else 1.0
        med10 = float(np.nanmedian(p10)) if np.isfinite(p10).any() else None
        med90 = float(np.nanmedian(p90)) if np.isfinite(p90).any() else None
    drr, dcc = pr["drr"][idx], pr["dcc"][idx]
    land = float(np.mean(mask[drr, dcc] != WATER))
    why = []
    if med10 is None or med10 >= rules["p10_max_db"]:
        why.append("基準 p10 が明るい")
    if med10 is None or med90 is None or med90 - med10 < rules["spread_min_db"]:
        why.append("p90−p10 が小さい")
    if land >= rules["land_max"]:
        why.append("陸が多い")
    if blind >= rules["blind_share_max"]:
        why.append("盲域")
    return {"p10_db": None if med10 is None else round(med10, 1), "p90_db": None if med90 is None else round(med90, 1),
            "land_share": round(land, 2), "blind_share": round(blind, 2), "readable": not why, "why": "・".join(why)}


def nearest_s2(days: list[dict], t: datetime, window_days: float = S1_S2_WINDOW_DAYS) -> dict | None:
    """The review day nearest to ``t`` within the window: {date, image, grid, dt_h (S1 - S2, hours)}."""
    best = None
    for d in days:
        dt = (t - cdse.parse_time(d["acquired_utc"])).total_seconds() / 3600.0
        if abs(dt) <= window_days * 24 and (best is None or abs(dt) < abs(best["dt_h"])):
            best = {"date": d["date"], "acquired_utc": d["acquired_utc"], "image": d["image"], "grid": d["grid"], "dt_h": round(dt, 1)}
    return best


def s2_prefill(answers: list[dict], section_ids: set, s2: dict | None, max_h: float = S1_PREFILL_H) -> dict:
    """{section_id: "present" | "absent"} from the owner's S2 answers of the scene's S2 day when it is within ``max_h``."""
    if s2 is None or abs(float(s2["dt_h"])) > max_h:
        return {}
    res = {}
    for r in answers:
        if r.get("date") == s2["date"] and r.get("section_id") in section_ids and r.get("verified") == "1":
            res[r["section_id"]] = "present" if r.get("verified_occupied") == "1" else "absent"
    return res


def prepare_s1(site_id: str, out: Path = OUT, raw: RawStore | None = None, site: dict | None = None, log=print) -> dict:
    """review-s1.json and labels/<site>/s1/<stamp>.webp from the saved chips (no CDSE). The sections are those of review.json
    (the owner's S2 check); the scenes every chip of the site whose grid is the site's."""
    from . import baseline as fbase
    from . import color, s2_labels
    from .osm_mask import MASK_DIR, load_mask
    from .run import site_sectors, sector_of as _sector_of
    site = site or site_config(site_id)
    raw = raw or RawStore()
    rv_path = review_path(out, site_id)
    if not rv_path.exists():
        log(f"{site_id}: review.json がない（先に prepare）")
        return {"site_id": site_id, "scenes": 0}
    rv = json.loads(rv_path.read_text(encoding="utf-8"))
    grid = Grid.for_site(site)
    mver = int(site.get("mask_version", 1))
    mask, _ = load_mask(site_id, mver, MASK_DIR)
    lines = quay_lines.load(site_id, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), MASK_DIR)
    geom = quay.Geometry(grid, mask, lines)
    sectors = site_sectors(site)
    by_id = {sc["section_id"]: sc for sc in s2_labels.sections(geom, lambda lo, la: _sector_of(sectors, lo, la))}
    bls = {}
    for o in fbase.site_config(site)["orbits"]:
        b = fbase.load(site, o, fbase.BASELINE_DIR)
        if b is not None and (b[1].get("width"), b[1].get("height")) == (grid.width, grid.height):
            bls[str(o)] = b[0]
    ans = read_csv(answers_path(out, site_id))
    sec_ids = {s["id"] for s in rv["sections"]}
    sdir = out / "labels" / site_id / "s1"
    sdir.mkdir(parents=True, exist_ok=True)
    scenes, kept, skipped = [], set(), []
    gdoc = {"west": grid.west, "south": grid.south, "east": grid.east, "north": grid.north, "width": grid.width, "height": grid.height}
    for stem in raw.stems(site_id):
        z = raw.load(site_id, stem)
        if z is None:
            continue
        vv, vh, meta = z
        if vv.shape != (grid.height, grid.width):
            skipped.append(f"{stem}: {vv.shape[1]}x{vv.shape[0]} != grid {grid.width}x{grid.height}")
            continue
        acq = meta.get("acquired_utc") or (stem[3:13] + "T" + stem[14:22].replace("-", ":") + "Z")
        ro = str(meta.get("relative_orbit", ""))
        label = f"{site_id} S1 {acq} RO{ro} {meta.get('orbit_direction', '')}".strip()
        credit = meta.get("credit") or f"Contains modified Copernicus Sentinel data {acq[:4]}"
        full, _thumb = color.render(vv, vh, label, credit)
        (sdir / f"{stem}.webp").write_bytes(full)
        kept.add(stem)
        s2 = nearest_s2(rv["days"], cdse.parse_time(acq))
        scenes.append({"acquired_utc": acq, "relative_orbit": ro, "platform": meta.get("platform"), "orbit_direction": meta.get("orbit_direction"),
                       "image": f"s1/{stem}.webp", "grid": gdoc, "s2": s2, "prefill": s2_prefill(ans, sec_ids, s2),
                       "baseline": ro in bls})
    for old in sdir.glob("*.webp"):
        if old.stem not in kept:
            old.unlink()
    scenes.sort(key=lambda x: x["acquired_utc"])
    out_secs = []
    for s in rv["sections"]:
        sc = by_id.get(s["id"])
        marks = {}
        for ro, base in bls.items():
            marks[ro] = readability(sc, geom, mask, base) if sc is not None else {"readable": False, "why": "区間が今の線に無い"}
        out_secs.append({**s, "s1": marks})
    n_cells = len(out_secs) * len(scenes)
    n_pre = sum(sum(1 for s in out_secs if s["id"] in sc["prefill"]) for sc in scenes)
    minutes = round((n_pre * SEC_PER_CELL["prefilled"] + (n_cells - n_pre) * SEC_PER_CELL["read"]) / 60)
    doc = {"version": S1_REVIEW_VERSION, "site_id": site_id, "name": site.get("name"), "kind": "s1", "frame_m": rv.get("frame_m", FRAME_M),
           "zoom": rv.get("zoom", 8), "grid": gdoc, "scenes": scenes, "sections": out_secs, "cells": n_cells, "prefilled": n_pre,
           "minutes_estimate": minutes, "prefill_h": S1_PREFILL_H, "s2_window_days": S1_S2_WINDOW_DAYS,
           "answers_file": s1_answers_path(out, site_id).name, "color_version": color.COLOR_VERSION,
           "credit": "Contains modified Copernicus Sentinel data " + ", ".join(sorted({x["acquired_utc"][:4] for x in scenes})),
           "attribution": "岸壁の線：(c) OpenStreetMap contributors（ODbL 1.0）",
           "readable_rule": {**READABLE, "note": ("S1 で判断できる区間の機械的な印（軌道ごと）：基準の VV p10（内側 0〜30 m）の区間の中央値が p10_max_db 未満、"
                                                 "p90−p10 が spread_min_db 以上、水際から 120 m までの陸の割合が land_max 未満、盲域（p10 > blind_p10_db）の割合が "
                                                 "blind_share_max 未満。満たさない区間は「要 S2」（既定は判断できない）。所有者は見て直せる")},
           "skipped": skipped,
           "selection": {"sections": "S2 の確認セット（review.json）と同じ区間", "scenes": "取り直しの S1（probe-raw/s1-recount）の全撮影のうち格子が今の地点のもの",
                         "s2": f"各撮影の前後 {S1_S2_WINDOW_DAYS:g} 日以内で最も近い確認セットの S2 の日（無ければ null）",
                         "prefill": f"S2 が {S1_PREFILL_H:g} 時間以内なら所有者の S2 の答えを初期値に（確認だけ）"}}
    s1_review_path(out, site_id).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    log(f"{site_id}: {len(scenes)} S1 scenes x {len(out_secs)} sections = {n_cells} cells ({n_pre} prefilled, about {minutes} min)"
        + (f"; skipped {skipped}" if skipped else ""))
    return {"site_id": site_id, "scenes": len(scenes), "sections": len(out_secs), "cells": n_cells, "prefilled": n_pre, "minutes": minutes}


def merge_s1(site_id: str, file: Path, out: Path = OUT, by: str = "owner") -> dict:
    """Add the page's S1 answers to labels/<site>-s1-answers.csv: key (section_id, s1_acquired_utc), the later answer replaces the
    earlier; rows outside review-s1.json or with an unknown verified / occupied value are rejected and counted."""
    rv = json.loads(s1_review_path(out, site_id).read_text(encoding="utf-8"))
    secs = {s["id"] for s in rv["sections"]}
    scenes = {s["acquired_utc"]: s for s in rv["scenes"]}
    path = s1_answers_path(out, site_id)
    cur = {(r["section_id"], r["s1_acquired_utc"]): r for r in read_csv(path)}
    n_new = n_rep = 0
    bad = []
    for r in read_csv(file):
        v, occ = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip()
        acq = (r.get("s1_acquired_utc") or "").strip()
        ok = r.get("section_id") in secs and acq in scenes and ((v == "0" and occ == "") or (v == "1" and occ in ("0", "1")))
        if not ok:
            bad.append(f"{r.get('section_id')}@{acq}")
            continue
        k = (r["section_id"], acq)
        n_rep += k in cur
        n_new += k not in cur
        sc = scenes[acq]
        cur[k] = {"site_id": site_id, "section_id": r["section_id"], "s1_acquired_utc": acq, "relative_orbit": sc.get("relative_orbit"),
                  "s2_date": (sc.get("s2") or {}).get("date"), "dt_h": (sc.get("s2") or {}).get("dt_h"),
                  "verified": v, "verified_occupied": occ, "verified_by": by, "verified_note": (r.get("verified_note") or "").strip()}
    order = {s["id"]: i for i, s in enumerate(rv["sections"])}
    write_csv(path, sorted(cur.values(), key=lambda r: (r["s1_acquired_utc"], order.get(r["section_id"], 0))), S1_ANSWER_FIELDS)
    return {"site_id": site_id, "added": n_new, "replaced": n_rep, "rejected": bad, "total": len(cur)}


# ---------------------------------------------------------------- merge
def merge(site_id: str, file: Path, out: Path = OUT, oos: bool = False, by: str = "owner") -> dict:
    """Add a downloaded answers file to labels/<site>-s2-answers.csv (or -oos-answers.csv). Rows for a section or day outside
    the review set, or with an unknown verified / c value, are rejected and counted."""
    rv = json.loads(review_path(out, site_id, oos).read_text(encoding="utf-8"))
    secs = {s["id"] for s in rv["sections"]}
    days = {d["date"] for d in rv["days"]}
    path = answers_path(out, site_id, oos)
    cur = {(r["section_id"], r["date"]): r for r in read_csv(path)}
    n_new = n_rep = 0
    bad = []
    rows = read_csv(file)
    scene_rows = [r for r in rows if "warships_60" in r]
    for r in rows:
        if "warships_60" in r:
            continue
        v, occ, c = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip(), (r.get("verified_c") or "").strip()
        ok = (r.get("section_id") in secs and r.get("date") in days and
              ((v == "0" and occ == "" and c == "") or (v == "1" and ((occ == "0" and c == "0") or (occ == "1" and c in ("1", "2", "3"))))))
        if not ok:
            bad.append(f"{r.get('section_id')}@{r.get('date')}")
            continue
        k = (r["section_id"], r["date"])
        n_rep += k in cur
        n_new += k not in cur
        cur[k] = {"site_id": site_id, "section_id": r["section_id"], "date": r["date"], "verified": v, "verified_occupied": occ,
                  "verified_c": c, "verified_color": (r.get("verified_color") or "").strip(),
                  "verified_by": by, "verified_note": (r.get("verified_note") or "").strip()}
    order = {s["id"]: i for i, s in enumerate(rv["sections"])}
    write_csv(path, sorted(cur.values(), key=lambda r: (r["date"], order.get(r["section_id"], 0))), ANSWER_FIELDS)
    if oos and scene_rows:
        sp = out / "labels" / f"{site_id}-oos-scene.csv"
        sc = {r["date"]: r for r in read_csv(sp)}
        for r in scene_rows:
            if r.get("date") in days and (r.get("warships_60") or "").strip().isdigit():
                sc[r["date"]] = {"site_id": site_id, "date": r["date"], "warships_60": r["warships_60"].strip(), "note": (r.get("note") or "").strip()}
        write_csv(sp, [sc[k] for k in sorted(sc)], SCENE_FIELDS)
    return {"site_id": site_id, "added": n_new, "replaced": n_rep, "rejected": bad, "total": len(cur)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="review sets and answers of the owner's eye check (no CDSE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--site", required=True)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--archive-ref", help="git ref with docs/data/sat/ports/<site>/ (e.g. origin/probe/ports-s2ext-ops3)")
    p.add_argument("--archive-dir", type=Path)
    p.add_argument("--oos", action="store_true")
    p.add_argument("--days", type=int, default=N_DAYS)
    p.add_argument("--sections", type=int, default=N_SECTIONS)
    p.add_argument("--naval-only", action="store_true", help="candidates: the sections on naval-1's naval lines only")
    p.add_argument("--s1", action="store_true", help="the S1 check: review-s1.json and labels/<site>/s1/*.webp from the saved chips")
    p.add_argument("--raw-ref", help="--s1: git ref holding probe-raw/s1-recount/<site>/ (e.g. origin/probe/sat-s1-recount-ops4)")
    p.add_argument("--raw-dir", type=Path, help="--s1: directory holding <site>/s1-*.npz instead")
    m = sub.add_parser("merge")
    m.add_argument("--site", required=True)
    m.add_argument("--file", type=Path, required=True)
    m.add_argument("--out", type=Path, default=OUT)
    m.add_argument("--oos", action="store_true")
    m.add_argument("--s1", action="store_true", help="the S1 check's answers -> labels/<site>-s1-answers.csv")
    pl = sub.add_parser("plan")
    pl.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "prepare" and a.s1:
        print(json.dumps(prepare_s1(a.site, a.out, RawStore(a.raw_ref, a.raw_dir)), ensure_ascii=False))
    elif a.cmd == "prepare":
        r = prepare(a.site, a.out, Archive(a.archive_ref, a.archive_dir), a.oos, a.days, a.sections, naval_only=a.naval_only)
        print(json.dumps({k: v for k, v in r.items() if k != "plan"}, ensure_ascii=False))
    elif a.cmd == "merge" and a.s1:
        print(json.dumps(merge_s1(a.site, a.file, a.out), ensure_ascii=False))
    elif a.cmd == "merge":
        print(json.dumps(merge(a.site, a.file, a.out, a.oos), ensure_ascii=False))
    else:
        print(json.dumps(plan_all(a.out), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
