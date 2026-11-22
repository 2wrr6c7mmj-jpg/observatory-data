"""Presence / absence of a surface ship of 50 m or more at the naval quays (plan A, A-0): the judge against the owner's eye.

The unit is one Sentinel-1 scene x one quay section (the ~150 m pieces of the quay lines of sat/s2_labels.sections()). The
sections are limited to the ones a warship of 50 m or more can lie at: on a line classed ``naval`` by naval-1
(sat/berth_class/<site>.json, Berths.line_class at the section's middle point, with the sections the class file adds) and at
least MIN_LENGTH_M long (the section's own ``length_m``). The truth is the owner's answer of the Sentinel-2 day
(labels/<site>-s2-answers.csv, labels/<site>-oos-answers.csv; Zhanjiang: the verified rows of labels/ZJG-s2-labels.csv):
verified_occupied 1 = present, 0 = absent; "cannot judge" (verified 0) is left out. The Sentinel-1 scene of a label day is the one
the review set paired it with (labels/<site>/review*.json ``s1_near`` [0]: the nearest scene of an orbit with a baseline within
+-4 days, the rule of sat/s2_answers.eligible_days; Zhanjiang: labels/ZJG-s2-s1-match.csv, sat/s2_labels.nearest_s1, the same rule).

The judge's probability of "a ship of 50 m or more is present" on the section:

* est-2 (quay_est2.json / quay_est2/<site>.json) when the scene is in it: the stretch of the same OSM way and side with the
  most points within SECTION_MATCH_M of the section (its line is the quay feature of the same ``n`` in quay/<site>/*.geojson);
  p = p_real when its ship_length_m >= MIN_LENGTH_M, else 0; no stretch: 0.
* otherwise the stored Sentinel-1 re-count (labels/<site>-s2-s1-match.csv): p = 1 when the stretch that reaches into the
  section (s1_stretch_len_m: the full length of the stretch, the same quantity as est-2's ship_length_m; ports, s1-match-3) is
  MIN_LENGTH_M or longer, else 0; a table without that column (Zhanjiang, s1-match-1): s1_occupied 1 and s1_occ_len_m >=
  MIN_LENGTH_M (the occupied length inside the section). A 0/1 judge: its Brier score is its error rate.
* a scene stored in neither is not judged: the cell is counted under ``no_scene`` and left out of the metrics (an unstored scene
  is not "absent").

Metrics per site and over all sites, in sample (label day before the site's label_freeze_utc) and out of sample (from the freeze
on) apart: confusion matrix at p >= 0.5 (tp, fp, fn, tn), recall = tp / (tp + fn), precision = tp / (tp + fp), Brier = mean (p - y)^2,
the base rate of the owner's "present" and the judge's mean p. ``any_length`` repeats the metrics with the length rule off
(the owner's label does not carry the length of the ship: a quay occupied by a boat under 50 m is "present" to the owner and
"absent" to the 50 m target; this variant shows how much of the disagreement is that).

Two truths (``--truth``, design 設計_同日の正解と港の誤り検品 2026-11 §2): ``s2`` (default, above) and ``s1``, the owner's answer
at the Sentinel-1 scene itself (labels/<site>-s1-answers.csv of the S1 check page, key (section_id, s1_acquired_utc): no pairing,
dt_h 0). presence.json carries the chosen truth as its body and the other under ``other_truth`` (metrics and rows), so both are
always published side by side. Every row carries ``blind_share`` (the share of the section's line on a likely-blind stretch of
the scene's orbit, quay/<site>/blind.geojson: the baseline VV p10 of the inner band above -8 dB) and ``line_conf`` (naval-1's
confidence of the line: 確認寄り / 推定 / fixed / unknown), and the metrics are repeated by blind (blind_share >= BLIND_SHARE) /
clear and by line confidence.

occ-2 (eval-presence-3, 設計書_occ-2 §4.3・§5): when the scene has occ-2 rows (docs/data/sat/quay_occ2/<site>/YYYY-MM.csv,
sat/occ2_schema; the evaluation sat/eval_occ2.py writes them for the re-count scenes, the daily flow later) every row also
carries the occ-2 judge beside the current one: ``p_occ2`` (the occupation probability, before the calibration a placeholder
a = logit(prior), b = 1, c = 0), ``p_test_occ2`` (the test's P(T >= T_obs | empty)), ``occ2_blind`` (occ-2's own blind flag:
spatial reference), ``edge_offset_m`` / ``edge_source`` (the baseline's ridge) and ``a1_target`` (a section the inspection
left to A-1, the line map: A = line geometry, D = boat basins; sat/occ2_schema.A1_TARGETS). The metrics blocks carry an
``occ2`` entry with occ-2's numbers and the current judge's numbers on the same cells (the body ``p`` stays the current judge;
nothing of the published presence changes where occ-2 has no rows). ``auc`` (ROC-AUC, Mann-Whitney, ties half) is in every
metrics block.

pres-1 (eval-presence-6, sat/pres1, design docs/data/sat/eval/pres1-design.md): every row also carries pres-1a (trial, a parallel
series) beside the current judge: ``res_rate`` / ``res_n`` / ``prev_y`` (the owner's answers of the section on the days before the
scene only), ``p_pres1`` (the logistic regression of p, res_rate and res_n with the coefficients of sat/pres1_calib.json, fitted on
the cells before its ``fit_before``), ``pres1_oos`` and ``change_flag`` (the judge disagrees with the section's previous answer).
The blocks carry ``pres1`` (pres-1's out-of-sample cells only; ``changed``: the cells whose answer changed, where pres-1a is
weaker than the current judge). The body ``p`` stays the current judge. refresh also writes quay_pres1/<site>.json for the ports
whose sites.json quay.pres1 is trial / on (Zhanjiang: off, no file).

Pure Python (no numpy, no CDSE: PU 0). Output docs/data/sat/eval/presence.json and docs/data/sat/eval/meta.json.

Regression guard (tests/test_sat_eval_presence.py): ``presence_table`` is the judge's presence (p >= 0.5, 50 m) of every stored
scene of the last SNAPSHOT_DAYS days on the labelled sections, under the rules of the working tree (naval-1 class files, the
berth rule applied to the stored stretches, the section set). ``snapshot`` writes it with the rule versions to
docs/data/sat/eval/presence_snapshot.json; the test recomputes it and fails when more than SNAPSHOT_MAX_CHANGE of the common cells
differ (a rule change that flips the presence of over a fifth of the sections must be looked at, not merged by accident).

    python -m sat.eval_presence [--out docs/data/sat]            # the evaluation
    python -m sat.eval_presence --snapshot [--out docs/data/sat] # refresh the regression snapshot (after an intended rule change)

The daily run (sat/run.py save) calls ``refresh``: presence.json and meta.json are rewritten from the stored files, and the
snapshot too when the rule versions are unchanged, so the published evaluation never goes stale.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import io

from . import atomic, berths
from . import blind_prior as bp
from . import radar_dark
from . import opt_fill
from . import occ2_schema as o2s
from . import pres1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
EVAL_VERSION = "eval-presence-6"   # eval-presence-5 (blind-1 beside the judge) + pres-1a and the change flag beside the judge (sat/pres1)
MIN_LENGTH_M = 50.0
SECTION_MATCH_M = 20.0        # as sat/quay_est2.SECTION_MATCH_M (not imported: quay_est2 needs numpy)
# assign-2 (2026-11-10): which stretch belongs to a section; the same rule and numbers as sat/s2_labels (not imported: numpy)
STRETCH_ASSIGN = "assign-2"
STRETCH_INSIDE_M = 50.0
ASSIGN_1_SITES = ("ZJG",)       # Zhanjiang keeps assign-1 (its calibration and its evaluation were made with it)
THRESHOLD = 0.5
FAR_H = 48.0                  # |S1 - S2| beyond this is "far" in the pairing summary
SOURCES = ("est-2", "s1-recount")
TEST_ALPHA = 0.05             # occ-2: a cell is "present" by the test alone when p_test < this (beside p_occ >= THRESHOLD)
TRUTHS = ("s2", "s1")
BLIND_SHARE = 0.5             # a cell whose section lies on a blind stretch for at least this share is "blind"
BLIND_MATCH_M = 20.0          # a point of the section line within this of a blind stretch of the same OSM way is blind
NAMES = {"ZJG": "湛江", "SYL": "三亜・楡林", "SLP": "三亜・亜龍湾", "QGZ": "青島・古鎮口", "ZDH": "舟山・定海",
         "LSN": "大連・旅順", "QDH": "青島港内", "QJG": "青島・姜哥荘", "XSN": "象山", "SWS": "上海・呉淞", "STU": "汕頭",
         "XMN": "廈門", "MAW": "馬尾"}     # SWS was "汕尾" (Shanwei, another city) until 2026-11-13: SWS is Wusong, Shanghai


# ---------------------------------------------------------------- small helpers
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_sites(path: Path = ROOT / "sat" / "sites.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["sites"]


def _t(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _xy(coords, lat0: float) -> list[tuple[float, float]]:
    mx, my = 111320.0 * math.cos(math.radians(lat0)), 111132.0
    return [(float(c[0]) * mx, float(c[1]) * my) for c in coords]


def _min_dist(p, line) -> float:
    """Distance (m) of a point to a polyline, both in metres."""
    best = math.inf
    px, py = p
    if len(line) == 1:
        return math.hypot(px - line[0][0], py - line[0][1])
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        dx, dy = bx - ax, by - ay
        dd = dx * dx + dy * dy
        t = 0.0 if dd == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
        best = min(best, math.hypot(px - ax - t * dx, py - ay - t * dy))
    return best


def _side(f: dict) -> int:
    face = str(f.get("face") or "")
    return -1 if face.endswith("-") else 1


def _side_of(s) -> int:
    return 1 if str(s) in ("1", "L") else -1


def load_lines(site: dict) -> dict | None:
    """The site's quay lines (sat/masks/<site>-quay-v<n>.json, as sat/quay_lines.load; read here without numpy)."""
    p = ROOT / "sat" / "masks" / f"{site['id']}-quay-v{int(site.get('quay_lines_version', 1))}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- the labelled sections and their class
def site_sections(out: Path, site: dict) -> dict[str, dict]:
    """{section_id: {way_id, side, kind, length_m, line, lon, lat, naval}} of the sections the owner was asked about: the review
    sets (review.json, review-oos.json) and, for Zhanjiang, labels/ZJG-s2-sections.geojson. ``naval``: the line is classed naval
    by naval-1 at the section's middle point."""
    sid = site["id"]
    b = berths.Berths(site, load_lines(site), naval=True)
    secs: dict[str, dict] = {}
    for name in ("review.json", "review-oos.json"):
        p = out / "labels" / sid / name
        if not p.exists():
            continue
        for s in json.loads(p.read_text(encoding="utf-8"))["sections"]:
            line = s["line"]
            mid = line[len(line) // 2]
            secs.setdefault(s["id"], {"way_id": s["way"], "side": _side_of(s["side"]), "kind": s.get("kind"),
                                      "length_m": float(s.get("length_m") or 0), "line": line, "lon": mid[0], "lat": mid[1]})
    p = out / "labels" / f"{sid}-s2-sections.geojson"
    if p.exists():
        for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            line = f["geometry"]["coordinates"]
            secs.setdefault(pr["section_id"], {"way_id": pr["way_id"], "side": _side_of(pr.get("side", 1)), "kind": pr.get("kind"),
                                                "length_m": float(pr.get("length_m") or 0), "line": line,
                                                "lon": pr.get("lon", line[len(line) // 2][0]), "lat": pr.get("lat", line[len(line) // 2][1])})
    for s in secs.values():
        s["class"] = b.line_class(s["way_id"], float(s["lon"]), float(s["lat"]))
        s["conf"] = line_conf(b, site, s["way_id"], float(s["lon"]), float(s["lat"]))
        s["naval"] = s["class"] == "naval"
        # naval-1 "berth": false (a naval line where no warship berths: a boat basin's shore, a jetty for boats): not evaluated
        s["berth"] = not b.no_berth_at(s["way_id"], float(s["lon"]), float(s["lat"]))
        s["eligible"] = s["naval"] and s["berth"] and s["length_m"] >= MIN_LENGTH_M
    return secs


def line_conf(b: "berths.Berths", site: dict, way_id, lon: float, lat: float) -> str:
    """naval-1's confidence of the line's class at (lon, lat): the entry's ``conf`` (確認寄り, 推定), "fixed" for a decision by
    hand without one, "unknown" when the way has no entry (the same entry line_class picks)."""
    areas = {x["id"]: x for x in (site.get("sectors") or [])}
    for e in (b.classes or {}).get("lines") or []:
        if str(e.get("way")) != str(way_id):
            continue
        part = e.get("part") or {}
        if part.get("sector"):
            area = areas.get(part["sector"], {"bbox": [0, 0, 0, 0]})
        elif part.get("bbox") or part.get("polygon"):
            area = part
        else:
            area = None
        if area is None or berths._in_area(area, lon, lat):
            return str(e.get("conf") or ("fixed" if e.get("fixed") else "unknown"))
    return "unknown"


def blind_stretches(out: Path, site_id: str, lat0: float) -> dict:
    """{relative_orbit: [(osm_way, [(x, y), ...]), ...]} of the likely-blind stretches (quay/<site>/blind.geojson, or the shared
    quay/blind.geojson), in metres (``_xy`` with the site's ``lat0``, the same as the sections)."""
    p = out / "quay" / site_id / "blind.geojson"
    if not p.exists():
        p = out / "quay" / "blind.geojson"
    if not p.exists():
        return {}
    res: dict = {}
    for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
        pr = f["properties"]
        if pr.get("site_id") != site_id:
            continue
        g = f["geometry"]
        coords = [g["coordinates"]] if g["type"] == "Point" else g["coordinates"]
        res.setdefault(str(pr.get("relative_orbit")), []).append((str(pr.get("osm_way")), _xy(coords, lat0)))
    return res


def blind_share(sec: dict, blind: dict, orbit, lat0: float) -> float | None:
    """The share of the section's line points within BLIND_MATCH_M of a blind stretch of the same OSM way and orbit (None: the
    orbit has no blind layer)."""
    sts = blind.get(str(orbit))
    if sts is None:
        return None
    pts = _xy(sec["line"], lat0)
    if not pts:
        return None
    n = 0
    for p in pts:
        for way, line in sts:
            if way == str(sec["way_id"]) and _min_dist(p, line) <= BLIND_MATCH_M:
                n += 1
                break
    return round(n / len(pts), 2)


# ---------------------------------------------------------------- the labels and their Sentinel-1 scene
def labels(out: Path, site_id: str) -> list[dict]:
    """The owner's answers: {section_id, date, y (1/0), verified_c, source}; verified 0 (cannot judge) left out."""
    rows = []
    for name in (f"{site_id}-s2-answers.csv", f"{site_id}-oos-answers.csv", f"{site_id}-s2-labels.csv"):
        for r in read_csv(out / "labels" / name):
            if r.get("site_id", site_id) != site_id or r.get("verified") != "1":
                continue
            rows.append({"section_id": r["section_id"], "date": r["date"], "y": 1 if r.get("verified_occupied") == "1" else 0,
                         "verified_c": r.get("verified_c") or "", "source": name})
    seen: dict = {}
    for r in rows:   # one answer per section and day (the answers files are already so; the labels file is a different set)
        seen[(r["section_id"], r["date"])] = r
    return list(seen.values())


def labels_s1(out: Path, site_id: str) -> list[dict]:
    """The owner's answers at the Sentinel-1 scene (labels/<site>-s1-answers.csv, the S1 check): {section_id, s1, y, s2_date, s2_dt_h};
    verified 0 left out."""
    rows = {}
    for r in read_csv(out / "labels" / f"{site_id}-s1-answers.csv"):
        if r.get("site_id", site_id) != site_id or r.get("verified") != "1" or not r.get("s1_acquired_utc"):
            continue
        rows[(r["section_id"], r["s1_acquired_utc"])] = {"section_id": r["section_id"], "s1": r["s1_acquired_utc"],
                                                          "y": 1 if r.get("verified_occupied") == "1" else 0,
                                                          "s2_date": r.get("s2_date") or None, "s2_dt_h": float(r["dt_h"]) if r.get("dt_h") else None,
                                                          "note": r.get("verified_note") or ""}
    return list(rows.values())


def s1_of_days(out: Path, site_id: str) -> dict:
    """{date: (s1 acquired_utc, dt_h)} from the review sets (s1_near[0])."""
    res = {}
    for name in ("review.json", "review-oos.json"):
        p = out / "labels" / site_id / name
        if not p.exists():
            continue
        for d in json.loads(p.read_text(encoding="utf-8"))["days"]:
            near = d.get("s1_near") or []
            if near:
                res.setdefault(d["date"], (near[0]["acquired_utc"], near[0].get("dt_h")))
    return res


def s1_match(out: Path, site_id: str, by_scene: bool = False) -> dict:
    """The re-count: {(section_id, s2 date): row} of labels/<site>-s2-s1-match.csv (``by_scene``: keyed by (section_id, s1_acquired_utc))."""
    k = "s1_acquired_utc" if by_scene else "s2_date"
    return {(r["section_id"], r[k]): r for r in read_csv(out / "labels" / f"{site_id}-s2-s1-match.csv")}


# ---------------------------------------------------------------- the judge
def est2_scenes(out: Path, site_id: str) -> dict:
    """{acquired_utc: scene} of est-2 (quay_est2.json for Zhanjiang, quay_est2/<site>.json otherwise)."""
    p = out / ("quay_est2.json" if site_id == "ZJG" else f"quay_est2/{site_id}.json")
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {s["acquired_utc"]: s for s in d.get("scenes", []) if s.get("site_id") == site_id}


def quay_features(out: Path, site_id: str) -> dict:
    """{(acquired_utc, mask_version): [feature properties + coords]} of the stored quay stretches (quay/<site>/*.geojson or the
    shared quay/*.geojson). A scene counted with two masks (v1 and v2 rows of the same acquisition) keeps both sets apart."""
    month = "[0-9][0-9][0-9][0-9]-[0-9][0-9].geojson"
    files = sorted((out / "quay" / site_id).glob(month)) if (out / "quay" / site_id).is_dir() else sorted((out / "quay").glob(month))
    res: dict = {}
    for p in files:
        for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            if pr.get("site_id") == site_id:
                res.setdefault((pr["acquired_utc"], pr.get("mask_version") or ""), []).append({**pr, "coords": f["geometry"]["coordinates"]})
    return res


def scene_features(feats: dict, sc: dict) -> list[dict]:
    """The quay features of an est-2 scene: the same acquisition and mask version (an est-2 scene without a mask version:
    the features of any mask of that acquisition)."""
    mv = sc.get("mask_version")
    if mv:
        return feats.get((sc["acquired_utc"], mv), [])
    return [f for (acq, _m), fs in feats.items() if acq == sc["acquired_utc"] for f in fs]


def calibrated(out: Path, site_id: str) -> bool:
    """Was est-2 calibrated on this site's own labels? (quay_est2 calib_source: none or "self" = its own; a site id = that
    site's weights were borrowed, so none of its labels is "in sample")."""
    p = out / ("quay_est2.json" if site_id == "ZJG" else f"quay_est2/{site_id}.json")
    if not p.exists():
        return False
    src = json.loads(p.read_text(encoding="utf-8")).get("calib_source")
    return src in (None, "", "self", site_id)


def _midpoint(line: list[tuple[float, float]]) -> tuple[float, float]:
    """The point halfway along a polyline (metres)."""
    seg = [math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(line, line[1:])]
    half, run = sum(seg) / 2.0, 0.0
    for d, (a, b) in zip(seg, zip(line, line[1:])):
        if d > 0 and run + d >= half:
            t = (half - run) / d
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        run += d
    return line[len(line) // 2]


def stretch_of_section(sec: dict, feats: list[dict], stretches: list[dict], lat0: float, assign: str = STRETCH_ASSIGN):
    """The est-2 stretch (same OSM way and side, the most points within SECTION_MATCH_M) or None. A stretch carries no
    geometry: its ``n`` is the ``n`` of the scene's quay feature (quay/<site>/*.geojson), whose line is used. est-2 keeps
    fewer stretches than the scene has features (berth-1 / naval-1 drop some), so the pairing is by ``n``, not by position.
    ``assign`` assign-2 (the ports, as sat/s2_labels.stretch_belongs): only a stretch that belongs to the section, i.e.
    STRETCH_INSIDE_M of the section's line (or all of it, when shorter) lies along the stretch, or the stretch's midpoint lies
    on the section. assign-1 (Zhanjiang, unchanged): any stretch with a point of the section within SECTION_MATCH_M."""
    sxy = _xy(sec["line"], lat0)
    by_n = {f.get("n"): f for f in feats}
    spacing = float(sec.get("length_m") or 0) / max(len(sxy), 1) or 10.0
    need = min(STRETCH_INSIDE_M, len(sxy) * spacing)
    best = None
    for st in stretches:
        f = by_n.get(st.get("n"))
        if f is None or str(st.get("osm_way")) != str(sec["way_id"]) or int(st.get("side", _side(f))) != int(sec["side"]):
            continue
        fxy = _xy(f["coords"], lat0)
        n = sum(1 for p in sxy if _min_dist(p, fxy) <= SECTION_MATCH_M)
        if n and assign != "assign-1" and not (n * spacing >= need or _min_dist(_midpoint(fxy), sxy) <= SECTION_MATCH_M / 2):
            continue
        if n and (best is None or n > best[0]):
            best = (n, st)
    return best[1] if best else None


def judge_est2(st: dict | None, min_len: float) -> float:
    if st is None:
        return 0.0
    if min_len and float(st.get("ship_length_m") or 0) < min_len:
        return 0.0
    return float(st.get("p_real", 1.0))


def judge_recount(r: dict | None, min_len: float) -> float | None:
    """The re-count's presence. With ``s1_stretch_len_m`` (ports, s1-match-3): the full length of the stretch that belongs to the
    section (assign-2, sat/s2_labels.stretch_belongs) against ``min_len`` (the length rule on the whole stretch, as est-2's
    ship_length_m: a 60 m ship across a section border is present on the side of its midpoint); without it (Zhanjiang):
    s1_occupied and the occupied length inside the section."""
    if r is None or r.get("s1_occupied") in (None, ""):
        return None
    if "s1_stretch_len_m" in r:
        n = float(r.get("s1_stretch_len_m") or 0)
        if n <= 0 and r["s1_occupied"] != "1":
            return 0.0
        return 1.0 if n >= min_len or (not min_len and r["s1_occupied"] == "1") else 0.0
    if r["s1_occupied"] != "1":
        return 0.0
    if min_len and float(r.get("s1_occ_len_m") or 0) < min_len:
        return 0.0
    return 1.0


def recount_length(r: dict | None):
    if r is None:
        return None
    return (r.get("s1_stretch_len_m") if "s1_stretch_len_m" in r else r.get("s1_occ_len_m")) or None


# ---------------------------------------------------------------- metrics
def metrics(rows: list[dict], key: str = "p") -> dict:
    n = len(rows)
    m = {"n": n, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "recall": None, "precision": None, "accuracy": None, "brier": None,
         "base_rate": None, "mean_p": None, "auc": None}
    if not n:
        return m
    for r in rows:
        y, pred = r["y"], int(r[key] >= THRESHOLD)
        m["tp" if (y and pred) else "fn" if y else "fp" if pred else "tn"] += 1
    m["recall"] = round(m["tp"] / (m["tp"] + m["fn"]), 3) if (m["tp"] + m["fn"]) else None
    m["precision"] = round(m["tp"] / (m["tp"] + m["fp"]), 3) if (m["tp"] + m["fp"]) else None
    m["accuracy"] = round((m["tp"] + m["tn"]) / n, 3)
    m["brier"] = round(sum((r[key] - r["y"]) ** 2 for r in rows) / n, 3)
    m["base_rate"] = round(sum(r["y"] for r in rows) / n, 3)
    m["mean_p"] = round(sum(r[key] for r in rows) / n, 3)
    m["auc"] = auc(rows, key)
    return m


def auc(rows: list[dict], key: str = "p"):
    """ROC-AUC by Mann-Whitney (ties count half); None without both classes."""
    pos = sorted(float(r[key]) for r in rows if r["y"])
    neg = sorted(float(r[key]) for r in rows if not r["y"])
    if not pos or not neg:
        return None
    import bisect
    s = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        s += lo + 0.5 * (hi - lo)
    return round(s / (len(pos) * len(neg)), 3)


def is_near(r: dict, far_h: float = FAR_H) -> bool:
    """|S1 - S2| <= far_h hours (a pairing the ships are less likely to have changed in)."""
    return r.get("dt_h") is not None and abs(float(r["dt_h"])) <= far_h


def is_blind(r: dict, share: float = BLIND_SHARE) -> bool:
    return r.get("blind_share") is not None and float(r["blind_share"]) >= share


def occ2_metrics(rows: list[dict]) -> dict:
    """occ-2 on the cells that have an occ-2 row, beside the current judge on the same cells: p_occ (min50 at THRESHOLD),
    the test alone (p_test < TEST_ALPHA), near pairs, blind / clear by occ-2's own flag, and without the A-1 sections."""
    o = [r for r in rows if r.get("p_occ2") is not None]
    for r in o:
        pt = r.get("p_test_occ2")
        r["_t05"] = 1.0 if (pt is not None and pt < TEST_ALPHA) else 0.0
        r["_negp"] = 1.0 - (pt if pt is not None else 1.0)   # the test's ranking (higher = more surely present), for its AUC

    def tm(rs):   # the test alone, on the cells that have a test (a blind cell without its own empty days has none): the confusion
        rs = [r for r in rs if r.get("p_test_occ2") is not None]   # at p_test < TEST_ALPHA, the AUC of the ranking by p_test
        m = metrics(rs, "_t05")
        m["auc"] = auc(rs, "_negp")
        m["brier"] = None   # a 0/1 cut of a p-value has no Brier worth reading
        return m
    near = [r for r in o if is_near(r)]
    blind = [r for r in o if r.get("occ2_blind")]
    clear = [r for r in o if not r.get("occ2_blind")]
    no_a1 = [r for r in o if not r.get("a1_target")]
    res = {"n": len(o), "alpha": TEST_ALPHA,
           "p_occ": metrics(o, "p_occ2"), "test": tm(o), "current": metrics(o),
           "near": {"far_h": FAR_H, "p_occ": metrics(near, "p_occ2"), "test": tm(near), "current": metrics(near)},
           "by_blind": {"blind": {"p_occ": metrics(blind, "p_occ2"), "test": tm(blind), "current": metrics(blind)},
                        "clear": {"p_occ": metrics(clear, "p_occ2"), "test": tm(clear), "current": metrics(clear)}},
           "excl_a1": {"n": len(no_a1), "p_occ": metrics(no_a1, "p_occ2"), "test": tm(no_a1), "current": metrics(no_a1)}}
    for r in o:
        r.pop("_t05", None)
        r.pop("_negp", None)
    return res


def blind1_metrics(rows: list[dict]) -> dict:
    """blind-1 beside the current judge: p_blind1 = max(p, the section's resident probability from the owner's other S1 scenes)."""
    b = [r for r in rows if r.get("p_blind1") is not None]
    res = [r for r in b if r.get("p_resident")]
    return {"version": bp.VERSION, "n": len(b), "n_resident": len(res), "min50": metrics(b, "p_blind1"), "current": metrics(b),
            "resident": {"p_blind1": metrics(res, "p_blind1"), "current": metrics(res)}}


def split_metrics(rows: list[dict]) -> dict:
    """min50 / any_length of the rows, the same of the near pairs only (|dt_h| <= FAR_H), min50 by blind / clear (blind_share >=
    BLIND_SHARE) and by the line's confidence, and occ-2 beside the current judge on the cells that have an occ-2 row."""
    near = [r for r in rows if is_near(r)]
    confs = sorted({r.get("line_conf") or "unknown" for r in rows})
    notes = {"prefill": lambda n: n.startswith("s2-prefill"), "default": lambda n: n.startswith("s1-unreadable-default"),
             "manual": lambda n: not (n.startswith("s2-prefill") or n.startswith("s1-unreadable-default"))}
    return {"min50": metrics(rows), "any_length": metrics(rows, "p_any"),
            "by_note": {k: metrics([r for r in rows if f(r.get("note") or "")]) for k, f in notes.items()},
            "near": {"far_h": FAR_H, "min50": metrics(near), "any_length": metrics(near, "p_any")},
            "by_blind": {"share": BLIND_SHARE, "blind": metrics([r for r in rows if is_blind(r)]),
                         "clear": metrics([r for r in rows if not is_blind(r)]),
                         "clear_near": metrics([r for r in near if not is_blind(r)])},
            "by_conf": {c: metrics([r for r in rows if (r.get("line_conf") or "unknown") == c]) for c in confs},
            "occ2": occ2_metrics(rows), "blind1": blind1_metrics(rows), "answers": answers_metrics(rows),
            "pres1": pres1.safe_block(rows, metrics)}


# Sections whose answer still waits for the owner's ruling on what counts as present (QGZ 219848841:L:00: a grey
# object that is always there; the S2 answers say present, the S1 answers absent). Marked, and left out of
# answers.settled, until the basis is decided.
PENDING_BASIS = {("QGZ", "219848841:L:00")}


def answers_metrics(rows: list[dict]) -> dict:
    """The current judge on the cells whose S1 and S2 answers agree (``agree`` True: both answers exist for the section,
    the S1 scene and its paired S2 day, and say the same), the count where they disagree, and the cells without a
    section waiting for the answer basis (PENDING_BASIS). Side by side; the main figures are unchanged."""
    agree = [r for r in rows if r.get("agree") is True]
    return {"agree": metrics(agree), "disagree_n": sum(1 for r in rows if r.get("agree") is False),
            "both_answered_n": sum(1 for r in rows if r.get("agree") is not None),
            "settled": metrics([r for r in rows if not r.get("pending_basis")]),
            "pending_basis_n": sum(1 for r in rows if r.get("pending_basis"))}


def _freeze(site: dict) -> str:
    return (site.get("quay") or {}).get("label_freeze_utc") or "2026-10-27"


# ---------------------------------------------------------------- one site
def evaluate_site(out: Path, site: dict, truth: str = "s2") -> dict:
    sid = site["id"]
    freeze = _freeze(site)
    secs = site_sections(out, site)
    s1_truth = truth == "s1"
    lab = labels_s1(out, sid) if s1_truth else labels(out, sid)
    days = {} if s1_truth else s1_of_days(out, sid)
    match = s1_match(out, sid, by_scene=s1_truth)
    # the other kind of answer, for answers.agree: S1 answers by (section, scene), S2 answers by (section, day)
    other_s1 = {} if s1_truth else {(a["section_id"], a["s1"]): a["y"] for a in labels_s1(out, sid)}
    other_s2 = {(a["section_id"], a["date"]): a["y"] for a in labels(out, sid)} if s1_truth else {}
    est = est2_scenes(out, sid)
    feats = quay_features(out, sid)
    own = calibrated(out, sid)
    lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 21.2)
    blind = blind_stretches(out, sid, lat0)
    orbit_of = scene_orbits(out, sid, est)
    occ2 = o2s.load_site_rows(out, sid)
    resident = bp.answers(out, sid)   # blind-1: the owner's S1 answers per section (the judged scene is left out below)
    cnt = {"answers": len(lab), "prefilled": sum(1 for a in lab if (a.get("note") or "").startswith("s2-prefill")),
           "unknown_section": 0, "not_naval": 0, "no_berth": 0, "short": 0, "no_s1_pairing": 0, "no_scene": 0, "other_day_of_scene": 0, "used": 0}
    rows = []
    for a in lab:
        s = secs.get(a["section_id"])
        if s is None:
            cnt["unknown_section"] += 1
            continue
        if not s["naval"]:
            cnt["not_naval"] += 1
            continue
        if not s["berth"]:
            cnt["no_berth"] += 1
            continue
        if s["length_m"] < MIN_LENGTH_M:
            cnt["short"] += 1
            continue
        if s1_truth:
            s1, dt = a["s1"], 0.0
            mr = match.get((a["section_id"], s1))
        else:
            mr = match.get((a["section_id"], a["date"]))
            if a["date"] in days:
                s1, dt = days[a["date"]]
            elif mr is not None and mr.get("s1_acquired_utc"):
                s1, dt = mr["s1_acquired_utc"], float(mr["dt_h"]) if mr.get("dt_h") else None
            else:
                cnt["no_s1_pairing"] += 1
                continue
        sc = est.get(s1)
        if sc is not None:
            fs = scene_features(feats, sc)
            sts = sc.get("stretches") or []
            st = stretch_of_section(s, fs, sts, lat0, "assign-1" if sid in ASSIGN_1_SITES else STRETCH_ASSIGN) if fs else None
            p, p_any, src = judge_est2(st, MIN_LENGTH_M), judge_est2(st, 0.0), "est-2"
            length = None if st is None else st.get("ship_length_m")
        else:
            p, p_any = judge_recount(mr, MIN_LENGTH_M), judge_recount(mr, 0.0)
            src, length = "s1-recount", recount_length(mr)
            if p is None:
                cnt["no_scene"] += 1
                continue
        orbit = orbit_of.get(s1) or (mr or {}).get("relative_orbit")
        row = {"section_id": a["section_id"], "date": a.get("date") or s1[:10], "s1": s1, "dt_h": dt, "source": src, "y": a["y"],
               "owner_c": a.get("verified_c", ""), "p": round(p, 3), "p_any": round(p_any, 3), "length_m": length,
               "blind_share": blind_share(s, blind, orbit, lat0), "line_conf": s.get("conf") or "unknown",
               "oos": (not own) or (a.get("date") or s1[:10]) >= freeze}
        if s1_truth:
            row.update(s2_date=a.get("s2_date"), s2_dt_h=a.get("s2_dt_h"), note=a.get("note") or "")
            other = other_s2.get((a["section_id"], a.get("s2_date") or ""))
        else:
            other = other_s1.get((a["section_id"], s1))
        row.update(agree=None if other is None else other == a["y"], pending_basis=(sid, a["section_id"]) in PENDING_BASIS)
        pb =bp.p_blind(resident, a["section_id"], exclude_s1={s1, a.get("date") or s1[:10], a.get("s2_date") or ""})
        row.update(p_resident=pb, p_blind1=round(max(p, pb), 3))
        o2 = occ2.get(s1, {}).get(a["section_id"])
        row.update(p_occ2=None if o2 is None or o2.get("p_occ") is None else round(float(o2["p_occ"]), 3),
                   p_test_occ2=None if o2 is None or o2.get("p_test") is None else float(o2["p_test"]),
                   occ2_blind=None if o2 is None else o2.get("blind"), edge_offset_m=None if o2 is None else o2.get("edge_offset_m"),
                   edge_source=None if o2 is None else o2.get("edge_source"), occ2_len_m=None if o2 is None else o2.get("stretch_len_m"),
                   a1_target=o2s.a1_target(sid, a["section_id"]))
        rows.append(row)
    # one label day per Sentinel-1 scene: the nearest in time (two S2 days within +-4 days of the same S1 scene would count
    # the same judgement twice); the other days' rows are dropped
    rows, dropped = nearest_day_per_scene(rows)
    cnt["other_day_of_scene"] = dropped
    # pres-1a and the change flag beside the judge (sat/pres1): the section's answers before the scene only
    # (never raises: on an error the pres-1 fields are null and a warning is printed; the rest of the evaluation goes on)
    pres1.safe_annotate(rows, out, sid)
    cnt["used"] = len(rows)
    ins = [r for r in rows if not r["oos"]]
    oos = [r for r in rows if r["oos"]]
    n_el = sum(1 for s in secs.values() if s["eligible"])
    scenes = sorted({r["s1"] for r in rows})
    far = [r for r in rows if r["dt_h"] is not None and abs(float(r["dt_h"])) > FAR_H]
    return {"site_id": sid, "name": NAMES.get(sid, site.get("name")), "freeze_utc": freeze, "calibrated": own, "truth": truth,
            "sample_note": ("標本内＝ラベルの日が凍結より前（est-2 の較正に使った期間）、標本外＝凍結以後" if own else
                            "est-2 はこの港の正解で較正していない（湛江の重みの流用）ので、全部を標本外とする"),
            "sections": {"labelled": len(secs), "naval": sum(1 for s in secs.values() if s["naval"]),
                         "no_berth": sum(1 for s in secs.values() if s["naval"] and not s["berth"]), "eligible": n_el},
            "labels": cnt, "scenes_judged": scenes,
            "pairing": {"scenes": len(scenes), "far_h": FAR_H, "far": len(far), "far_share": round(len(far) / len(rows), 3) if rows else None},
            "by_source": {src: {"n": sum(1 for r in rows if r["source"] == src),
                                "min50": metrics([r for r in rows if r["source"] == src]),
                                "any_length": metrics([r for r in rows if r["source"] == src], "p_any")} for src in SOURCES},
            "in_sample": split_metrics(ins), "out_of_sample": split_metrics(oos), "pres1": pres1.safe_block(rows, metrics),
            "status": _status(len(ins), len(oos), own), "rows": rows}


def scene_orbits(out: Path, site_id: str, est: dict) -> dict:
    """{acquired_utc: relative_orbit} of the scenes: est-2's scenes and the review sets' pairings."""
    res = {k: str(v.get("relative_orbit")) for k, v in est.items() if v.get("relative_orbit") is not None}
    for name in ("review.json", "review-oos.json"):
        p = out / "labels" / site_id / name
        if not p.exists():
            continue
        for d in json.loads(p.read_text(encoding="utf-8"))["days"]:
            for a in d.get("s1_near") or []:
                res.setdefault(a["acquired_utc"], str(a.get("relative_orbit")))
    return res


def nearest_day_per_scene(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep, for every Sentinel-1 scene, only the rows of the label day nearest in time (|dt_h|, then the earlier day)."""
    best: dict = {}
    for r in rows:
        k = (r["s1"], r["date"])
        d = abs(float(r["dt_h"])) if r["dt_h"] is not None else math.inf
        cur = best.get(r["s1"])
        if cur is None or (d, r["date"]) < cur:
            best[r["s1"]] = (d, r["date"])
    keep = [r for r in rows if r["date"] == best[r["s1"]][1]]
    return keep, len(rows) - len(keep)


def _status(n_in: int, n_oos: int, own: bool = True) -> str:
    if not n_in and not n_oos:
        return "標本 0"
    if not own:
        return f"標本外 {n_oos} 撮影×区間（較正の流用）"
    return f"標本 {n_in} 撮影×区間（標本外 {n_oos}）"


TRUTH_NOTE = {"s2": "所有者の S2 の日の答え（labels/<地点>-s2-answers.csv ほか）を、その日に最も近い S1 撮影（前後 4 日）に対応させたもの。dt_h＝S1−S2",
              "s1": "所有者が S1 の撮影そのものを見て答えたもの（labels/<地点>-s1-answers.csv、確認ページ ?s1=1：S1 の疑似カラーと直近の S2 の 2 枚組）。鍵 (section_id, s1_acquired_utc)、dt_h 0。s2_date・s2_dt_h は並べて見た S2 の日とその時間差、note s2-prefill＝S2 の答えを初期値のまま確認した升（by_note で手動の升と分けて出す）"}


def evaluate_one(out: Path, sites: list[dict], truth: str) -> dict:
    """{sites, total} of one truth (``s2`` or ``s1``)."""
    ids = []
    for s in sites:
        sid = s["id"]
        if truth == "s1":
            has = (out / "labels" / f"{sid}-s1-answers.csv").exists()
        else:
            has = (out / "labels" / sid / "review.json").exists() or (out / "labels" / f"{sid}-s2-labels.csv").exists() or \
                (out / "labels" / f"{sid}-s2-answers.csv").exists()
        if has:
            ids.append(sid)
    res = {sid: evaluate_site(out, next(s for s in sites if s["id"] == sid), truth) for sid in ids}
    all_rows = [r for r in res.values() for r in r["rows"]]
    ins = [r for r in all_rows if not r["oos"]]
    oos = [r for r in all_rows if r["oos"]]
    tot = {"in_sample": split_metrics(ins), "out_of_sample": split_metrics(oos), "pres1": pres1.safe_block(all_rows, metrics),
           "by_source": {src: {"n": sum(1 for r in all_rows if r["source"] == src), "min50": metrics([r for r in all_rows if r["source"] == src])}
                         for src in SOURCES}}
    return {"truth": truth, "truth_note": TRUTH_NOTE[truth], "total": tot, "sites": res}


def evaluate(out: Path, sites: list[dict] | None = None, now: datetime | None = None, truth: str = "s2") -> dict:
    """``now``: the time written as generated_utc (the daily run passes its own, so that two runs of one frozen time are
    byte for byte the same; default the clock). ``truth``: s2 (the S2 day's answer paired to the nearest S1 scene) or s1 (the
    answer at the S1 scene itself); the other truth's result is carried under ``other_truth``."""
    if truth not in TRUTHS:
        raise ValueError(f"truth: {TRUTHS}")
    sites = sites or load_sites()
    now = now or datetime.now(timezone.utc)
    main_ = evaluate_one(out, sites, truth)
    other = evaluate_one(out, sites, "s1" if truth == "s2" else "s2")
    return {"version": EVAL_VERSION, "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": "海軍の岸壁（naval-1 の naval の線）に着く 50 m 以上の水上艦の、撮影×岸壁区間ごとの在/不在",
            "min_length_m": MIN_LENGTH_M, "threshold": THRESHOLD, "truth": truth, "truth_note": main_["truth_note"],
            "total": main_["total"], "sites": main_["sites"],
            "other_truth": {"truth": other["truth"], "truth_note": other["truth_note"], "total": other["total"],
                            "sites": {sid: {k: v for k, v in r.items()} for sid, r in other["sites"].items()}}}


# ---------------------------------------------------------------- meta
def occ2_columns(rows: list[dict]) -> dict:
    """The occ-2 columns of the table from the rows of a site (or of all): occ-2 and the current judge on the same cells."""
    m = occ2_metrics(rows)

    def pick(block, tag):
        return {f"recall_{tag}": block["recall"], f"precision_{tag}": block["precision"], f"brier_{tag}": block["brier"], f"auc_{tag}": block["auc"]}
    out = {"n_occ2": m["n"], **pick(m["p_occ"], "occ2"), **pick(m["current"], "cur_occ2"),
           "recall_occ2_test": m["test"]["recall"], "precision_occ2_test": m["test"]["precision"],
           "n_occ2_near": m["near"]["p_occ"]["n"], **pick(m["near"]["p_occ"], "occ2_near"), **pick(m["near"]["current"], "cur_occ2_near"),
           "n_occ2_blind": m["by_blind"]["blind"]["p_occ"]["n"], "recall_occ2_blind": m["by_blind"]["blind"]["p_occ"]["recall"],
           "recall_cur_occ2_blind": m["by_blind"]["blind"]["current"]["recall"],
           "n_occ2_excl_a1": m["excl_a1"]["n"], "recall_occ2_excl_a1": m["excl_a1"]["p_occ"]["recall"],
           "precision_occ2_excl_a1": m["excl_a1"]["p_occ"]["precision"]}
    return out


def pres1_columns(b: dict) -> dict:
    """The pres-1 columns of the table from a pres1 block (sat/pres1.block): pres-1a and the current judge on pres-1's
    out-of-sample cells, on the cells whose answer changed, and the change flag."""
    if not b or b.get("error") or "pres1" not in b:   # pres-1 failed (pres1.safe_block): the columns are empty
        b = {"n": None, "pres1": {}, "current": {}, "changed": {"n": None, "pres1": {}, "current": {}}, "change_flag": {}}
    out = {"n_pres1": b["n"]}
    for tag, m in (("pres1", b["pres1"]), ("cur_pres1", b["current"])):
        out.update({f"recall_{tag}": m.get("recall"), f"precision_{tag}": m.get("precision"), f"f1_{tag}": m.get("f1"), f"brier_{tag}": m.get("brier")})
    out.update({"n_pres1_changed": b["changed"]["n"], "accuracy_pres1_changed": b["changed"]["pres1"].get("accuracy"),
                "accuracy_cur_pres1_changed": b["changed"]["current"].get("accuracy"),
                "change_flags": b["change_flag"].get("flags"), "change_flag_precision": b["change_flag"].get("precision"),
                "change_flag_recall": b["change_flag"].get("recall")})
    return out


def table(res: dict) -> list[dict]:
    rows = []
    other = (res.get("other_truth") or {}).get("sites") or {}
    for sid, r in res["sites"].items():
        m, o = r["in_sample"]["min50"], r["out_of_sample"]["min50"]
        mn, on = r["in_sample"]["near"]["min50"], r["out_of_sample"]["near"]["min50"]
        ob = r["out_of_sample"]["by_blind"]
        ot = (other.get(sid) or {}).get("out_of_sample", {}).get("min50") or {"n": 0, "recall": None, "precision": None, "brier": None}
        rows.append({"site_id": sid, "name": r["name"], "calibrated": r["calibrated"], "sections_eligible": r["sections"]["eligible"],
                     "answers": r["labels"]["answers"],
                     "n_blind_oos": ob["blind"]["n"], "n_clear_oos": ob["clear"]["n"], "recall_clear_oos": ob["clear"]["recall"],
                     "precision_clear_oos": ob["clear"]["precision"],
                     "n_other": ot["n"], "recall_other": ot["recall"], "precision_other": ot["precision"], "brier_other": ot["brier"],
                     "n": m["n"], "recall": m["recall"], "precision": m["precision"], "brier": m["brier"], "base_rate": m["base_rate"],
                     "n_near": mn["n"], "recall_near": mn["recall"], "precision_near": mn["precision"], "brier_near": mn["brier"],
                     "n_oos": o["n"], "recall_oos": o["recall"], "precision_oos": o["precision"], "brier_oos": o["brier"],
                     "n_near_oos": on["n"], "recall_near_oos": on["recall"], "precision_near_oos": on["precision"], "brier_near_oos": on["brier"],
                     "n_est2": r["by_source"]["est-2"]["n"], "n_recount": r["by_source"]["s1-recount"]["n"],
                     "scenes": r["pairing"]["scenes"], "far_share": r["pairing"]["far_share"],
                     "not_naval": r["labels"]["not_naval"], "no_berth": r["labels"].get("no_berth", 0),
                     "no_scene": r["labels"]["no_scene"] + r["labels"]["no_s1_pairing"],
                     "other_day_of_scene": r["labels"]["other_day_of_scene"], "status": r["status"], **occ2_columns(r.get("rows") or []),
                     **pres1_columns(r.get("pres1") or pres1.safe_block(r.get("rows") or [], metrics))})
    t = res["total"]["in_sample"]["min50"]
    to = res["total"]["out_of_sample"]["min50"]
    tn, ton = res["total"]["in_sample"]["near"]["min50"], res["total"]["out_of_sample"]["near"]["min50"]
    n_all = t["n"] + to["n"]
    far = sum(r["pairing"]["far"] for r in res["sites"].values())
    tob = res["total"]["out_of_sample"]["by_blind"]
    tot_other = (res.get("other_truth") or {}).get("total", {}).get("out_of_sample", {}).get("min50") or {"n": 0, "recall": None, "precision": None, "brier": None}
    rows.append({"site_id": "ALL", "name": "全港", "calibrated": None, "sections_eligible": sum(x["sections_eligible"] for x in rows),
                 "answers": sum(x["answers"] for x in rows),
                 "n_blind_oos": tob["blind"]["n"], "n_clear_oos": tob["clear"]["n"], "recall_clear_oos": tob["clear"]["recall"],
                 "precision_clear_oos": tob["clear"]["precision"],
                 "n_other": tot_other["n"], "recall_other": tot_other["recall"], "precision_other": tot_other["precision"], "brier_other": tot_other["brier"],
                 "n": t["n"], "recall": t["recall"], "precision": t["precision"], "brier": t["brier"], "base_rate": t["base_rate"],
                 "n_near": tn["n"], "recall_near": tn["recall"], "precision_near": tn["precision"], "brier_near": tn["brier"],
                 "n_oos": to["n"], "recall_oos": to["recall"], "precision_oos": to["precision"], "brier_oos": to["brier"],
                 "n_near_oos": ton["n"], "recall_near_oos": ton["recall"], "precision_near_oos": ton["precision"], "brier_near_oos": ton["brier"],
                 "n_est2": res["total"]["by_source"]["est-2"]["n"], "n_recount": res["total"]["by_source"]["s1-recount"]["n"],
                 "scenes": sum(x["scenes"] for x in rows), "far_share": round(far / n_all, 3) if n_all else None,
                 "not_naval": sum(x["not_naval"] for x in rows), "no_berth": sum(x["no_berth"] for x in rows), "no_scene": sum(x["no_scene"] for x in rows),
                 "other_day_of_scene": sum(x["other_day_of_scene"] for x in rows), "status": _status(t["n"], to["n"]),
                 **occ2_columns([x for r in res["sites"].values() for x in (r.get("rows") or [])]),
                 **pres1_columns(res["total"].get("pres1") or pres1.safe_block([x for r in res["sites"].values() for x in (r.get("rows") or [])], metrics))})
    return rows


def meta(res: dict) -> dict:
    return {
        "generated_utc": res["generated_utc"],
        "version": EVAL_VERSION,
        "files": {"presence.json": "評価票の本体：全体と港別の指標、撮影×区間の行（rows）", "meta.json": "この定義と港別の表（table）",
                  f"../{NAVAL50_FILE}": (f"{NAVAL50_VERSION}：地図の札の数。撮影ごとに、軍の着く線（naval-1）の上の片のうち船の長さが {MIN_LENGTH_M:g} m 以上のものを、"
                                         "艦種の重みや区画の役割で割り引かずに数える（ユーザーの決定 2026-11-11）。warship_50＝Σ (1＋0.5(k−1))×(1＋0.5(c_mean−1))×p_real の四捨五入"
                                         "（k＝縦に並ぶ数、c_mean＝横抱きの列数の期待値、p_real＝にじみでない確率。片の 1 隻目はそのまま、2 隻目からは半分に数える："
                                         "naval50-3、2026-11-15。撮影時刻の近い S2 との組 66 組で、大きな基地の数が目視の 1.6〜2.9 倍だったため）、stretches_50＝数えた片の数、"
                                         "warship_est2＝est-2 の中央値（割り引いた数）、ships_all＝全長・全種の中央値、min_length_m＝その港の長さの線引き"
                                         f"（既定 {MIN_LENGTH_M:g} m。水警区の港 {'・'.join(sorted(NAVAL50_MIN_LEN))} は {min(NAVAL50_MIN_LEN.values()):g} m：022 型ミサイル艇など小型の戦闘艦艇が主のため。列名は warship_50 のまま）"),
                  f"../{BLIND_DIR}/<地点>.csv": (f"{bp.VERSION}：常在の岸壁の加算。撮影ごとに resident_sections（所有者の答えで常在とした区間の数）、resident_seen（そのうち計数が在とした数）、"
                                                 "warship_blind（計数が在としなかった常在の区間の常在度の和の四捨五入。1 区間＝50 m 以上の軍艦 1 隻の下限、横抱きは数えない。稼働状態は問わない）。"
                                                 "地図の札はこれを足す。answers＝常在の表の出所（s1＝所有者の同日の答え、s2＝所有者の S2 の日の答え、ai＝AI の判断による仮の表：所有者の答えが無い港だけ）。"
                                                 "radar_dark_sections・radar_dark_seen・radar_dark_reasons＝レーダーでは判定できない区間（dark-1、../radar_dark.json）。"
                                                 f"末尾の {'・'.join(opt_fill.COLUMNS)}＝その区間を光学（pres-3o）の点数で埋める試行の系列（{opt_fill.VERSION}、../{opt_fill.OPT_FILL}。地図の札にも warship_blind にも足さない）。"
                                                 "方針（司令部の決定 2026-11-22、数は変えない）：潜水艦の桟橋の在は 50 m 以上の艦（潜水艦を含む）として軍艦の数に数える。"
                                                 "長さの規則は札の規則（片の長さ − 20 m を山の数で割り 50 m 以上）を正とし、評価もこれに揃える（評価票の変更は別の PR）"),
                  f"../{pres1.OUT_DIR}/<地点>.json": (f"{pres1.VERSION}（試行・並行系列、sat/pres1）：sites.json の quay.pres1 が trial／on の港の、直近 {pres1.KEEP_SCENES} 撮影の区間ごとの "
                                                    "p_current（現行の判定）・res_rate・res_n・prev_y（撮影の日より前の所有者の答え）・p_pres1・change_flag。湛江は off で書かない。公開の画面は読まない")},
        "target": res["target"],
        "truth_source": {"body": res.get("truth", "s2"), "other": (res.get("other_truth") or {}).get("truth"),
                         "note": "本体（total・sites・table）は truth_source.body の正解、other_truth は他方の正解で同じ指標。s2＝S2 の日の答えを直近の S1 に対応、s1＝S1 の撮影そのものの答え（確認ページ ?s1=1）。python -m sat.eval_presence --truth s1 で入れ替わる",
                         **{k: TRUTH_NOTE[k] for k in TRUTHS}},
        "unit": "撮影×岸壁区間。区間は sat/s2_labels.sections() の約 150 m の岸壁線の片（確認ページの区間、湛江は labels/ZJG-s2-sections.geojson）",
        "sections": f"naval-1（sat/berth_class/<地点>.json）で区間の中点の線の区分が naval（海軍の桟橋・岸壁。設定が足す区間を含む）で、区分ファイルが berth false（海軍の区域だが艦の着かない線：艇溜まりの岸・小型艇の桟橋）としていない線で、区間の長さ（length_m）が {MIN_LENGTH_M:g} m 以上のもの。civil・unknown の線の区間、berth false の区間、{MIN_LENGTH_M:g} m 未満の区間は数えない（labels.not_naval・labels.no_berth・labels.short）",
        "truth": "所有者の目視（labels/<地点>-s2-answers.csv、-oos-answers.csv、湛江は ZJG-s2-labels.csv の verified 1 の行）：verified_occupied 1＝在、0＝不在。判断できない（verified 0）は除く。所有者の答えは船の長さを持たない（50 m 未満の船の着いた区間も「在」になる）ので、any_length（長さの規則なし）も併記する",
        "pairing": "ラベルの日（S2 の晴れの日）の S1 撮影＝確認セット（labels/<地点>/review*.json の days.s1_near[0]）の対応：前後 4 日以内で基準のある軌道の最も近い撮影（sat/s2_answers.eligible_days）。湛江は labels/ZJG-s2-s1-match.csv の s1_acquired_utc（sat/s2_labels.nearest_s1、同じ規則）。dt_h＝S1−S2（時間）、同じ日でも約 7 時間ずれ、船が入れ替わることがある",
        "judge": {
            "est-2": f"その撮影が quay_est2.json・quay_est2/<地点>.json にあるとき：区間と同じ OSM の線・向きで {SECTION_MATCH_M:g} m 以内の点が最も多い区間片（stretch）のうち、その区間に属する片（{STRETCH_ASSIGN}：区間内に {STRETCH_INSIDE_M:g} m 以上あるか、片の中点が区間内。湛江は従来どおり 1 点でも掛かれば可）。p＝p_real（ship_length_m ≥ {MIN_LENGTH_M:g} m のとき、未満は 0）。片が無ければ 0",
            "s1-recount": (f"est-2 に無い撮影（ラベル日の S1 取り直し、labels/<地点>-s2-s1-match.csv）：区間に属する片（{STRETCH_ASSIGN}：区間内に {STRETCH_INSIDE_M:g} m 以上あるか、片の中点が区間内。2026-11-10 までは 20 m 掛かれば可）の全長 s1_stretch_len_m ≥ {MIN_LENGTH_M:g} m で 1、それ以外 0"
                           f"（港の表、s1-match-3。est-2 の ship_length_m と同じ量：区間の境をまたぐ艦は中点のある側で在、両側に {STRETCH_INSIDE_M:g} m 以上あれば両区間で在）。湛江の表（s1-match-1、列なし）は s1_occupied 1 かつ s1_occ_len_m ≥ {MIN_LENGTH_M:g} m。0/1 の判定：ブライアは誤り率に等しい"),
            "blind-1": (f"{bp.VERSION}（sat/blind_prior）：区間の常在度を併記。所有者の同日の答え（labels/<地点>-s1-answers.csv）で、判定する撮影を除いて {bp.MIN_ANSWERS} 撮影以上あり、"
                        f"在の割合が {bp.RESIDENT_MIN:g} 以上の区間は p_resident＝(在＋1)／(答え＋2)、ほかは 0。"
                        f"blind-2（2026-11-13）：同日の答えで常在でも、光学写真の答えが {bp.MIN_ANSWERS} 日以上あって在の割合が {bp.RESIDENT_MIN:g} 未満の区間は 0（判定する撮影の日の答えは除く）。p_blind1＝max(p, p_resident)。本体の p は現行の判定のまま。"
                        f"指標は各ブロックの blind1（min50＝p_blind1、current＝同じ升の現行、resident＝常在の升だけ）。対象は軍艦で稼働状態は問わない"),
            "pres-1": (f"{pres1.VERSION}（試行・並行系列、sat/pres1、係数 sat/pres1_calib.json）：升に res_rate・res_n（区間の、撮影の日＝ラベルの日と S1 の日の早い方より前の日の所有者の答え："
                       "(在＋1)／(答え＋2) と答えの数。S2 の日の答えと S1 の撮影の答え、1 日 1 つ）、prev_y（直前の答え）、p_pres1（logit＝b0＋b1·z(p)＋b2·z(res_rate)＋b3·z(res_n)、"
                       "fit_before より前の升だけで当てた係数）、pres1_oos（ラベルの日が fit_before 以後＝pres-1 の標本外）、change_flag（p ≥ 0.5 が prev_y と食い違うと 1、同じ 0、直前なし null）を併記。"
                       "本体の p は現行の判定のまま。pres-1a は「前と同じ」に強く、答えが前回から変わった升では現行の判定より弱い（pres1.changed）"),
            "no_scene": "どちらにも無い撮影は判定しない（labels.no_scene・no_s1_pairing）。保存の無い撮影を「不在」とはしない。港の確認セットの日の S1 は多くが未保存（s1_recount_plan.json の取り直し待ち）",
            "occ-2": (f"撮影に occ-2 の行（quay_occ2/<地点>/YYYY-MM.csv、sat/occ2）があれば、升に p_occ2（占有の確率。較正前は a=logit(事前)・b=1・c=0 の仮置き、"
                      f"盲域は blind_prior と λ=0.5 で混合）、p_test_occ2（検定の有意確率 P(T ≥ T_obs | 空)）、occ2_blind（occ-2 の盲域：空間の参照）、"
                      f"edge_offset_m・edge_source（基準 p10 の尾根の縁）、a1_target（検品 §1.2 で A-1 の線の地図に残した区間：A＝線の幾何、D＝艇溜まり）を併記。"
                      f"本体の p は現行の判定のまま。occ-2 の指標は各ブロックの occ2（p_occ ≥ {THRESHOLD:g}、検定だけ p_test < {TEST_ALPHA:g}、同じ升の現行 current）"),
        },
        "metrics": {
            "confusion": f"p ≥ {THRESHOLD:g} を「在」として tp・fp・fn・tn",
            "recall": "tp／(tp＋fn)：所有者が在とした区間のうち判定も在",
            "precision": "tp／(tp＋fp)：判定が在とした区間のうち所有者も在",
            "brier": "mean (p − y)²（0 が最良。常に基準率を答えると base_rate×(1−base_rate)）",
            "base_rate": "所有者の在の割合", "mean_p": "判定の p の平均",
            "in_sample / out_of_sample": ("標本内＝est-2 をその港の正解で較正した（calibrated）港で、ラベルの日が label_freeze_utc（sites.json の quay.label_freeze_utc、湛江 2026-10-27）より前。"
                                         "標本外＝凍結以後の日と、est-2 の重みを湛江から流用した港（calib_source が自港でない）の全部。港の数字は較正に使っていない正解での評価"),
            "by_source": "判定の出所別（est-2／s1-recount）の n と指標。湛江の見出しの数字は大半が s1-recount（S1 取り直しの生の占有の評価）で、est-2 の評価ではない",
            "pairing": f"scenes＝判定に使った S1 撮影の数、far_share＝|dt_h| > {FAR_H:g} 時間の升の割合。1 つの S1 撮影には最も近いラベル日だけを使う（他の日の升は labels.other_day_of_scene）",
            "near": f"in_sample・out_of_sample の near＝|S1−S2| ≤ {FAR_H:g} 時間の組だけの同じ指標（見出しの数字は全部の組。遠い組では船の入れ替わりが不一致に混じるので、近い組の数字を隣に置く）",
            "by_blind": f"升の blind_share（区間の線の点のうち、その撮影の軌道の盲域の片（quay/<地点>/blind.geojson：基準の VV p10 が内側の帯で −8 dB 超）から {BLIND_MATCH_M:g} m 以内の割合）が {BLIND_SHARE:g} 以上を盲域（blind）、未満を clear として分けた min50 の指標。clear_near＝clear のうち近い組",
            "by_conf": "升の line_conf（naval-1 の線の区分の確信：確認寄り／推定／fixed／unknown）ごとの min50 の指標",
            "auc": "ROC-AUC（Mann-Whitney、同値は半分）。0/1 の判定では精度の平均に近い",
            "occ2": (f"occ-2 の行のある升だけの指標：p_occ（p_occ2 ≥ {THRESHOLD:g}）、test（p_test_occ2 のある升だけ：< {TEST_ALPHA:g} を在。盲域で区間別の参照の無い升は検定が無く p_occ＝blind_prior）、current（同じ升の現行の判定）。"
                     f"near＝|S1−S2| ≤ {FAR_H:g} 時間、by_blind＝occ-2 の盲域（occ2_blind）で分けた同じ 3 つ、excl_a1＝A-1 対象の区間（a1_target）を除いた同じ 3 つ。"
                     "取り直しの 19 撮影の行は sat/eval_occ2 が書いた（較正前、全部が凍結前の答え＝較正に使う標本なので in-sample の数字）。"
                     "較正前の仮置き a＝logit(事前) では p_occ ≥ 事前 が常に成り立ち、事前 ≥ 0.5 の港では p_occ ≥ 0.5 が全升で在になる：較正（occ2-4）までは test の列（p_test）を読む。"
                     "正解を使わない変種（事前 0.5・盲域の事前 0.5）との比較は eval/occ2.json"),
            "pres1": ("pres-1 の標本外（pres1_oos：ラベルの日が sat/pres1_calib.json の fit_before 以後）の升だけの指標。各ブロック（in_sample・out_of_sample は est-2 の凍結での分け方のまま）と、"
                      "港の pres1・total.pres1（est-2 の分け方によらない全部の標本外の升）。pres1＝p_pres1 ≥ 0.5、current＝同じ升の現行の判定、f1＝2tp／(2tp＋fp＋fn)。"
                      "changed＝区間の答えが直前の答え（prev_y）から変わった升（入港・出港）、same＝変わらない升、no_prev_n＝直前の答えの無い升。"
                      "change_flag：cells＝直前の答えのある升、flags＝印の数、true＝印のうち本当に変わった升、precision＝true／flags、recall＝true／changes。"
                      "較正に使った升（fit_before より前）の指標は出さない。前進検証と区間は sat/pres1_calib.json"),
            "by_note": "s1 の正解だけ：升の verified_note 別の min50 の指標。prefill＝S2 の答えを初期値のまま確認した升（s2-prefill。所有者が押していない）、default＝要 S2 の区間の既定を上書きしなかった升（在/不在の答えにはならないので通常 0）、manual＝所有者が押した升。prefill の升は S2 の時刻の答えの流用なので、S1 の時刻の正解としては manual より弱い（labels.prefilled に件数）",
        },
        "table_columns": {"site_id": "地点", "name": "地名", "calibrated": "est-2 をその港の正解で較正したか（false＝湛江の流用、全部標本外）",
                          "sections_eligible": "対象の区間数（確認セットのうち naval・50 m 以上）", "answers": "所有者の答え（判断できないを除く）",
                          "n": "標本内の撮影×区間", "recall": "再現率", "precision": "適合率", "brier": "ブライア", "base_rate": "在の割合",
                          "n_near・recall_near・precision_near・brier_near": f"標本内のうち |S1−S2| ≤ {FAR_H:g} 時間の組だけの同じ指標",
                          "n_oos・recall_oos・precision_oos・brier_oos": "標本外の同じ指標",
                          "n_near_oos・recall_near_oos・precision_near_oos・brier_near_oos": f"標本外のうち |S1−S2| ≤ {FAR_H:g} 時間の組だけの同じ指標",
                          "n_est2 / n_recount": "判定の出所別の升の数（est-2 の区間片／S1 取り直しの占有）",
                          "n_blind_oos・n_clear_oos・recall_clear_oos・precision_clear_oos": f"標本外のうち盲域（blind_share ≥ {BLIND_SHARE:g}）の升の数と、盲域でない升の数・再現率・適合率",
                          "n_other・recall_other・precision_other・brier_other": "他方の正解（truth_source.other）での標本外の同じ指標（s1 の正解は S1 の撮影そのものの答え）",
                          "scenes": "判定に使った S1 撮影の数", "far_share": f"|S1−S2| > {FAR_H:g} 時間の升の割合",
                          "not_naval": "naval でない線の答え（除外）", "no_berth": "naval だが berth false の線の答え（除外）", "no_scene": "S1 の対応か保存が無い答え（除外）",
                          "prefilled（labels）": "s1 の正解のうち verified_note が s2-prefill（初期値のまま確認）の答えの数",
                          "other_day_of_scene": "同じ S1 撮影に対応した別のラベル日の答え（最も近い日だけ使う。除外）",
                          "n_occ2・recall_occ2・precision_occ2・brier_occ2・auc_occ2": f"occ-2 の行のある升（標本内外とも）の数と、p_occ2 ≥ {THRESHOLD:g} の再現率・適合率・ブライア・AUC",
                          "recall_cur_occ2・precision_cur_occ2・brier_cur_occ2・auc_cur_occ2": "同じ升での現行の判定（est-2／s1-recount）の同じ指標",
                          "recall_occ2_test・precision_occ2_test": f"検定だけ（p_test_occ2 < {TEST_ALPHA:g} を在）の再現率・適合率",
                          "n_occ2_near・…_occ2_near・…_cur_occ2_near": f"occ-2 の行のある升のうち |S1−S2| ≤ {FAR_H:g} 時間の同じ指標（occ-2 と現行）",
                          "n_occ2_blind・recall_occ2_blind・recall_cur_occ2_blind": "occ-2 の盲域（occ2_blind）の升の数と再現率（occ-2 と現行）",
                          "n_occ2_excl_a1・recall_occ2_excl_a1・precision_occ2_excl_a1": "A-1 対象の区間を除いた occ-2 の再現率・適合率",
                          "n_pres1・recall_pres1・precision_pres1・f1_pres1・brier_pres1": "pres-1 の標本外の升の数と、p_pres1 ≥ 0.5 の再現率・適合率・F1・ブライア（港の pres1 ブロック）",
                          "recall_cur_pres1・precision_cur_pres1・f1_cur_pres1・brier_cur_pres1": "同じ升での現行の判定の同じ指標",
                          "n_pres1_changed・accuracy_pres1_changed・accuracy_cur_pres1_changed": "そのうち答えが直前から変わった升の数と、pres-1a・現行の正答率（pres-1a の弱い所）",
                          "change_flags・change_flag_precision・change_flag_recall": "変化の印の数・適合率（印のうち本当に変わった割合）・再現率（変わった升のうち印の付いた割合）"},
        "table": table(res),
        "limits": [
            "湛江の見出しの数字（n の大半が s1-recount）は S1 取り直しの占有（quay-v1 の規則）の評価で、est-2 の評価は n_est2 の升だけ。両方の判定を混ぜて一つの指標にしている",
            f"{MIN_LENGTH_M:g} m の規則：est-2 は艦の長さの推定（ship_length_m、区間をまたぐ片の全長）、港の s1-recount も片の全長（s1_stretch_len_m、区間に属する片）。"
            "湛江の s1-recount の表（s1-match-1）だけは区間の中の占有の長さ（s1_occ_len_m）のままで、60 m の艦が区間の境にまたがると 30／30 に割れて両区間が「不在」になる（湛江 2026-01-03 の見逃しがこの型）",
            "標本が小さい港の数字は目安。n が 20 未満の再現率・適合率は一桁の件数で動く",
            "三亜・楡林、三亜・亜龍湾、青島・古鎮口、舟山・定海の確認セットは naval-1 より前に組んだため、海軍の線でない区間が多い（not_naval）。大連・旅順と青島港内の確認セットは naval の線だけで組んだ（s2_answers prepare --naval-only）が、答えはまだ無い",
            "所有者のラベルは S2（03 UTC 頃）、判定は S1（10 UTC 頃、最大 4 日差）：時刻差の入れ替わりは判定の誤りに数えられる",
            "判定の長さ（ship_length_m・s1_occ_len_m）は推定で、50 m 付近の艦は在/不在が揺れる",
        ],
        "credit": "Contains modified Copernicus Sentinel data 2024, 2025, 2026",
        "attribution": "岸壁の線・区間：(c) OpenStreetMap contributors（ODbL 1.0）",
    }


# ---------------------------------------------------------------- regression snapshot
SNAPSHOT_DAYS = 30
SNAPSHOT_MAX_CHANGE = 0.2
SNAPSHOT_MIN_SITE_CELLS = 30   # the per-site cap applies to sites with at least this many common cells
SNAPSHOT_FILE = "presence_snapshot.json"


def _quay_version() -> str:
    """quay.QUAY_VERSION read from the source (sat/quay.py needs numpy at import)."""
    m = re.search(r'^QUAY_VERSION\s*=\s*"([^"]+)"', (ROOT / "sat" / "quay.py").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "?"


def rule_versions(sites: list[dict] | None = None) -> dict:
    """The versions of the rules the presence depends on: berth-1, naval-1 (and a hash of every class file), quay, the quay
    lines per site, and berths.json's hash."""
    sites = sites or load_sites()
    cls = {}
    for p in sorted(berths.CLASS_DIR.glob("*.json")) if berths.CLASS_DIR.is_dir() else []:
        # without the radar-dark marks (sat/radar_dark.py, dark-1): a display mark, not a rule of the presence
        cls[p.stem] = hashlib.sha1(radar_dark.class_bytes_without_marks(p.read_text(encoding="utf-8"))).hexdigest()[:12]
    return {"berths": berths.VERSION, "berths_hash": hashlib.sha1(berths.BERTHS_PATH.read_bytes()).hexdigest()[:12],
            "naval": berths.NAVAL_VERSION, "naval_since": berths.NAVAL_SINCE, "classes": cls, "quay": _quay_version(),
            "lines": {s["id"]: int(s.get("quay_lines_version", 1)) for s in sites}}


def presence_table(out: Path, sites: list[dict] | None = None, days: int = SNAPSHOT_DAYS, since: str | None = None) -> dict:
    """{site_id: {acquired_utc: {section_id: 0/1}}}: the judge's presence (est-2, p >= THRESHOLD, MIN_LENGTH_M) on the eligible
    labelled sections (naval, >= 50 m under the current class files) for the stored scenes of the last ``days`` days (before the
    newest stored scene of the site, or from ``since``). A stretch counts only when the berth rule of the working tree keeps its
    feature (berths.feature_ok with the naval variant), so a change of berths.json or a class file shows here."""
    sites = sites or load_sites()
    res: dict = {}
    for site in sites:
        sid = site["id"]
        secs = {k: v for k, v in site_sections(out, site).items() if v["eligible"]}
        est = est2_scenes(out, sid)
        if not secs or not est:
            continue
        b = berths.Berths(site, load_lines(site), naval=True)
        feats = quay_features(out, sid)
        lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 21.2)
        newest = max(est)
        start = since or (_t(newest) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        tab: dict = {}
        for acq in sorted(a for a in est if a >= start):
            fs = [f for f in scene_features(feats, est[acq]) if berths.feature_ok(b, f, f["coords"])]
            sts = est[acq].get("stretches") or []
            row = {}
            for sec_id, sec in secs.items():
                st = stretch_of_section(sec, fs, sts, lat0, "assign-1" if sid in ASSIGN_1_SITES else STRETCH_ASSIGN) if fs else None
                row[sec_id] = int(judge_est2(st, MIN_LENGTH_M) >= THRESHOLD)
            tab[acq] = row
        if tab:
            res[sid] = tab
    return res


def snapshot(out: Path, sites: list[dict] | None = None, now: datetime | None = None) -> dict:
    sites = sites or load_sites()
    now = now or datetime.now(timezone.utc)
    doc = {"version": EVAL_VERSION, "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "days": SNAPSHOT_DAYS, "max_change": SNAPSHOT_MAX_CHANGE, "rules": rule_versions(sites),
           "min_site_cells": SNAPSHOT_MIN_SITE_CELLS,
           "note": ("回帰の見張り：規則（berths.json とそのハッシュ、naval-1 の区分ファイルのハッシュ、quay、岸壁線の版）が写しと違うとき、直近 30 日の撮影の在/不在"
                    "（est-2、p ≥ 0.5、50 m 以上、確認セットの naval の区間）を写しと比べ、共通の升の 2 割を超えて変わると（全体、または升が 30 以上の港のどれか）"
                    "tests/test_sat_eval_presence.py が失敗する。写しの撮影のうち半分未満しか見つからない港があっても失敗（写しが古い）。規則の版が同じ日は毎日の実行（sat/run.py）が"
                    "写しを更新する。規則を変えた PR では python -m sat.eval_presence --snapshot で写しを更新し、変わった升の数を PR に書く"),
           "table": presence_table(out, sites)}
    (out / "eval").mkdir(parents=True, exist_ok=True)
    atomic.write_text(out / "eval" / SNAPSHOT_FILE, json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return doc


def refresh(out: Path, sites: list[dict] | None = None, log=None, now: datetime | None = None) -> dict:
    """The daily run (sat/run.py save): rewrite eval/presence.json and eval/meta.json from the stored files (PU 0), and the
    regression snapshot when its rule versions are those of the working tree (a rule change refreshes it by hand, after the
    guard has been looked at). Returns {"presence": n rows, "snapshot": "updated" | "kept" | "created"}."""
    sites = sites or load_sites()
    now = now or datetime.now(timezone.utc)
    res = evaluate(out, sites, now)
    write(out, res)
    p = out / "eval" / SNAPSHOT_FILE
    state = "kept"
    if not p.exists():
        snapshot(out, sites, now)
        state = "created"
    else:
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("rules") == rule_versions(sites):
            snapshot(out, sites, now)
            state = "updated"
    # dark-1: the radar-dark marks follow the sheet just written (docs/data/sat/radar_dark.json), before quay_blind reads them
    # The marks change no count: a failure here (a record that does not read, a bug) must not stop quay_blind, quay_naval50 and
    # quay_pres1. It is reported, and quay_blind is made from the list of an earlier day (none: no mark)
    try:
        dark = radar_dark.refresh(out)
    except Exception as e:                      # noqa: BLE001 (any failure of the marks alone)
        dark = {"counts": {}, "stale": [], "error": repr(e)[:300]}
        print(f"::warning title=sat radar_dark::印の作り直しに失敗（前の radar_dark.json で quay_blind を作る。数は変わらない）: {dark['error']}", flush=True)
    if dark["stale"] and log:
        print(f"::warning title=sat radar_dark::sat/berth_class の印（radar_dark）が評価票より古い: {', '.join(dark['stale'])}"
              "（python -m sat.radar_dark build で作り直す。公開の radar_dark.json と quay_blind は今日の評価票で作った）", flush=True)
    # opt-fill-1 (sat/opt_fill.py): the trial series beside the marks (the radar-dark sections from the optical judge's
    # scores of the confirm run). It adds to no count: a failure leaves its columns of quay_blind empty and stops nothing
    fill_err = None
    try:
        fill = opt_fill.prepare(out, [s["id"] for s in sites])
    except Exception as e:                      # noqa: BLE001 (any failure of the trial series alone)
        fill, fill_err = {}, repr(e)[:300]
        print(f"::warning title=sat opt_fill::光学で埋める試行（opt-fill-1）の準備に失敗（quay_blind の opt_fill の列は空。数は変わらない）: {fill_err}", flush=True)
    r = {"presence": sum(len(x["rows"]) for x in res["sites"].values()), "snapshot": state, "radar_dark": dark, "blind": write_blind(out, sites, fill),
         "naval50": write_naval50(out, sites), "pres1": pres1.safe_write(out, sites)}
    try:
        r["opt_fill"] = {"error": fill_err} if fill_err else opt_fill.write_public(out, fill)
    except Exception as e:                      # noqa: BLE001
        r["opt_fill"] = {"error": repr(e)[:300]}
        print(f"::warning title=sat opt_fill::opt_fill.json の書き出しに失敗（数は変わらない）: {r['opt_fill']['error']}", flush=True)
    if log:
        log(f"eval/presence.json: {r['presence']} 升、写し {state}、quay_blind {sum(r['blind'].values())} 行、quay_naval50 {r['naval50']} 行、"
            f"quay_pres1 {len(r['pres1'])} 港、radar_dark {sum(r['radar_dark']['counts'].values())} 区間（区分ファイルの記録が古い港 {r['radar_dark']['stale'] or 'なし'}"
            f"{'、作り直しに失敗' if r['radar_dark'].get('error') else ''}）、opt-fill（試行）"
            f"{'失敗' if r['opt_fill'].get('error') else {k: v['present_sum'] for k, v in r['opt_fill'].items()}}")
    return r


BLIND_DIR = "quay_blind"
BLIND_COLUMNS = ("site_id", "acquired_utc", "blind_version", "answers", "resident_sections", "resident_seen", "warship_blind",
                 # dark-1 (sat/radar_dark.py): the eligible sections the radar cannot judge on the scene's day, how many of them
                 # the count saw, and why ("+"-joined: submarine_berth, owner_present_radar_none). warship_blind is not changed
                 "radar_dark_sections", "radar_dark_seen", "radar_dark_reasons") + opt_fill.COLUMNS
# opt-fill-1 (sat/opt_fill.py, trial): the radar-dark sections filled from the optical judge pres-3o, at the end. Added to no count


def blind_counts(out: Path, sites: list[dict] | None = None, fill: dict | None = None) -> dict:
    """blind-1 (2/3): {site_id: [row, ...]} for every stored est-2 scene of the sites that have a resident table (the owner's
    answers, sat/blind_prior): the resident sections among the eligible ones, how many of them the count saw (the judge's
    presence, as presence_table) and ``warship_blind``: the sum of the resident probability of the ones it did not see, rounded.
    One section (about 150 m) stands for one warship of 50 m or more: a lower bound, ships abreast are not counted. The number is
    added to est-2's warship_median for the map; est-2's own files are not touched."""
    sites = sites or load_sites()
    if fill is None:        # not given (by hand, the tests): the trial series as the daily run makes it (refresh gives {} on a failure)
        try:
            fill = opt_fill.prepare(out, [s["id"] for s in sites])
        except Exception:                       # noqa: BLE001 (the trial series alone: its columns empty)
            fill = {}
    res: dict = {}
    for site in sites:
        sid = site["id"]
        ans = bp.answers(out, sid)
        prior = {sec: bp.p_blind(ans, sec) for sec in ans}
        prior = {k: v for k, v in prior.items() if v > 0}
        if not prior:
            continue
        tab = presence_table(out, [site], since="").get(sid) or {}
        try:
            dark = radar_dark.marks(sid, out)
        except Exception:                       # noqa: BLE001 (a list that does not read: no mark, the counts as always)
            dark = {}
        rows = []
        for acq in sorted(tab):
            mine = [k for k in sorted(prior) if k in tab[acq]]
            why = {k: radar_dark.is_dark(dark[k], acq) for k in sorted(tab[acq]) if k in dark}
            dk = [k for k, w in why.items() if w]
            rows.append({"site_id": sid, "acquired_utc": acq, "blind_version": bp.VERSION, "answers": bp.answers_source(out, sid),
                         "resident_sections": len(mine), "resident_seen": sum(tab[acq][k] for k in mine),
                         "warship_blind": int(sum(prior[k] for k in mine if not tab[acq][k]) + 0.5),
                         "radar_dark_sections": len(dk), "radar_dark_seen": sum(tab[acq][k] for k in dk),
                         "radar_dark_reasons": "+".join(r for r in radar_dark.REASONS if any(r in why[k] for k in dk)),
                         **_opt_fill_row(fill, sid, acq, dk, tab[acq], mine)})
        if rows:
            res[sid] = rows
    return res


def _opt_fill_row(fill: dict | None, sid: str, acq: str, dark: list, seen: dict, resident: list) -> dict:
    """The opt-fill columns of a quay_blind row (sat/opt_fill.py): empty when the series is not made (no context, Zhanjiang, a
    failure). A failure here leaves the columns empty and changes no other column."""
    try:
        return opt_fill.fill((fill or {}).get(sid), acq, dark, seen, resident)
    except Exception:                           # noqa: BLE001 (the trial series alone)
        return dict(opt_fill.EMPTY)


def write_blind(out: Path, sites: list[dict] | None = None, fill: dict | None = None) -> dict:
    """Write quay_blind/<site>.csv (all stored scenes, rewritten whole: PU 0). Returns {site_id: number of rows}.
    A site that has no row any more loses the file of an earlier run (its table lost its last resident section: the owner's
    answers replaced the AI's table, or were corrected). The map adds whatever the file says and reads a missing file as +0,
    so a file left behind would keep adding the old number (Lushun 2026-11-13: the AI's +2 after the owner's answers gave 0)."""
    sites = sites or load_sites()
    res = blind_counts(out, sites, fill)
    d = out / BLIND_DIR
    for site in sites:
        stale = d / f"{site['id']}.csv"
        if site["id"] not in res and stale.exists():
            stale.unlink()
    for sid, rows in res.items():
        d.mkdir(parents=True, exist_ok=True)
        f = io.StringIO()
        w = csv.DictWriter(f, fieldnames=BLIND_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        atomic.write_text(d / f"{sid}.csv", f.getvalue())
    return {sid: len(rows) for sid, rows in res.items()}


NAVAL50_FILE = "quay_naval50.csv"
NAVAL50_VERSION = "naval50-3"   # naval50-2 + half of the ships beyond the first (NAVAL50_K_DAMP, NAVAL50_C_DAMP)
# naval50-3 (2026-11-15): the count of a stretch keeps the first ship whole and half of the rest, end to end (k) and abreast
# (c_mean): (1 + K_DAMP (k - 1)) x (1 + C_DAMP (c_mean - 1)) x p_real. The close pairs (close_pairs.csv: 66 pairs of a radar
# scene and a Sentinel-2 image within 24 h, the AI's count by eye) showed naval50-2 over the eye count at the big bases
# (Zhanjiang, Guzhenkou, Dinghai, Yulin, Longpo: 17 of 24 pairs over, 1.6-2.9 times), mostly from k (a long bright stretch read as
# several ships end to end). With these values: 9 of 18 pairs within the eye range (1 of 18 before), Zhanjiang 1.08 times, Dinghai
# 1.20, Guzhenkou 1.52. Abreast ships are half kept: the eye at 10 m misses some of them. The small ports barely change (k ~ 1).
NAVAL50_K_DAMP = 0.5
NAVAL50_C_DAMP = 0.5
NAVAL50_COLUMNS = ("site_id", "acquired_utc", "naval50_version", "stretches_50", "warship_50", "warship_est2", "ships_all", "min_length_m")
# The length line of a site when it is not MIN_LENGTH_M. The maritime garrison ports (水警区: the navy's coastal patrol units, not
# the coast guard) are home to small combatants: the Type 022 missile boat is about 43 m, and at Shantou 20 ships were seen on the
# naval berths against 2 of 50 m or more. User's decision 2026-11-11: lower the line to 40 m for these ports only.
NAVAL50_MIN_LEN = {"STU": 40.0, "XMN": 40.0, "SWS": 40.0}


def naval50_sum(stretches: list[dict], min_len: float = MIN_LENGTH_M) -> tuple[int, float]:
    """(stretches of ``min_len`` or more, their ships) of est-2 stretches already on naval berths. The ships of a stretch
    (naval50-3) = (1 + K_DAMP (k - 1)) x (1 + C_DAMP (c_mean - 1)) x p_real: the first ship whole, half of the rest."""
    n, total = 0, 0.0
    for st in stretches:
        if float(st.get("ship_length_m") or 0) < min_len:
            continue
        n += 1
        k, c = float(st.get("k") or 1), float(st.get("c_mean") or 1)
        total += ((1 + NAVAL50_K_DAMP * (k - 1)) * (1 + NAVAL50_C_DAMP * (c - 1))
                  * float(1 if st.get("p_real") is None else st["p_real"]))
    return n, total


NAVAL50_BOX_DIR = "quay_naval50"   # <site>.json: where the counted stretches are (for the red frames on the colour image)
NAVAL50_BOX_SCENES = 10            # the newest scenes kept there (the colour images keep the same number)
NAVAL50_BOX_PAD_M = 15.0           # a stretch is a line along the quay: its frame is the line's bounds widened by this much


def stretch_box(coords, pad_m: float = NAVAL50_BOX_PAD_M) -> list[float] | None:
    """[west, south, east, north] of a stretch's line (LineString or MultiLineString coordinates), widened by ``pad_m``."""
    pts = [c for c in coords if isinstance(c[0], (int, float))] or [c for part in coords for c in part]
    if not pts:
        return None
    lons, lats = [c[0] for c in pts], [c[1] for c in pts]
    dlat = pad_m / 111320.0
    dlon = dlat / max(0.1, math.cos(math.radians((min(lats) + max(lats)) / 2)))
    return [round(min(lons) - dlon, 6), round(min(lats) - dlat, 6), round(max(lons) + dlon, 6), round(max(lats) + dlat, 6)]


def naval50_counts(out: Path, sites: list[dict] | None = None, boxes: dict | None = None) -> list[dict]:
    """naval50-1 (user's decision 2026-11-11: "a ship of 50 m or more seen at a naval pier or quay is a warship; no discount"):
    for every stored est-2 scene, the ships of the stretches on a naval berth line (naval-1, the berth rule of the working tree)
    whose ship length is MIN_LENGTH_M or more, counted without est-2's class weights and sector roles:
    ``warship_50`` = round(sum of k x c_mean x p_real), k = the ships end to end in the stretch, c_mean = the expected number
    abreast, p_real = est-2's probability that the stretch is a real ship (smear). ``stretches_50`` = the stretches counted,
    ``warship_est2`` = est-2's own median (class weights, sector roles: the discounted number), ``ships_all`` = est-2's median of
    all ships of any length. The map shows warship_50; est-2's files are not touched.
    ``boxes`` (a dict, filled when given): {site_id: {acquired_utc: [{n, box_lonlat, ships, length_m}]}} of the stretches counted."""
    sites = sites or load_sites()
    rows = []
    for site in sites:
        sid = site["id"]
        est = est2_scenes(out, sid)
        if not est:
            continue
        b = berths.Berths(site, load_lines(site), naval=True)
        feats = quay_features(out, sid)
        for acq in sorted(est):
            sc = est[acq]
            fs = {f.get("n"): f for f in scene_features(feats, sc)}
            kept = [st for st in sc.get("stretches") or []
                    if fs.get(st.get("n")) is not None and berths.feature_ok(b, fs[st.get("n")], fs[st.get("n")]["coords"])]
            min_len = NAVAL50_MIN_LEN.get(sid, MIN_LENGTH_M)
            n50, total = naval50_sum(kept, min_len)
            if boxes is not None:
                boxes.setdefault(sid, {})[acq] = [
                    {"n": st.get("n"), "box_lonlat": stretch_box(fs[st.get("n")]["coords"]),
                     "ships": round(naval50_sum([st], min_len)[1], 2), "length_m": st.get("ship_length_m")}
                    for st in kept if naval50_sum([st], min_len)[0]]
            rows.append({"site_id": sid, "acquired_utc": acq, "naval50_version": NAVAL50_VERSION, "stretches_50": n50,
                         "warship_50": int(total + 0.5), "warship_est2": (sc.get("warship") or {}).get("median"),
                         "ships_all": (sc.get("ships") or {}).get("median"), "min_length_m": f"{min_len:g}"})
    return rows


def write_naval50(out: Path, sites: list[dict] | None = None) -> int:
    """Write quay_naval50.csv (all sites, all stored scenes, rewritten whole: PU 0). Returns the number of rows (no file when 0)."""
    boxes: dict = {}
    rows = naval50_counts(out, sites, boxes)
    if rows:
        f = io.StringIO()
        w = csv.DictWriter(f, fieldnames=NAVAL50_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        atomic.write_text(out / NAVAL50_FILE, f.getvalue())
        write_naval50_boxes(out, rows, boxes)
        from . import close_pairs   # the S1 scenes with a clear S2 image within a day (user 2026-11-15: keep them every day)
        close_pairs.update(out, rows, boxes)
    return len(rows)


def naval50_box_doc(site_id: str, rows: list[dict], scenes: dict) -> dict:
    """quay_naval50/<site>.json: the newest NAVAL50_BOX_SCENES scenes of a site, each with its count and the counted stretches."""
    mine = {r["acquired_utc"]: r for r in rows if r["site_id"] == site_id}
    keep = sorted(mine)[-NAVAL50_BOX_SCENES:]
    return {"version": NAVAL50_VERSION, "site_id": site_id, "min_length_m": float(mine[keep[-1]]["min_length_m"]) if keep else None,
            "note": "box_lonlat は数えた片（軍の着く線の上で、船の長さが線引き以上）の線の範囲を 15 m 広げた四角 [西, 南, 東, 北]。"
                    "ships はその片の隻数（縦に並ぶ数と横抱きの期待値の、1 隻目はそのまま・2 隻目からは半分 × 実在の確率。naval50-3）。船体の輪郭ではない",
            "scenes": {a: {"warship_50": mine[a]["warship_50"], "boxes": scenes.get(a, [])} for a in keep}}


def write_naval50_boxes(out: Path, rows: list[dict], boxes: dict) -> int:
    """Write quay_naval50/<site>.json for every site with rows (rewritten whole: PU 0). Returns the number of files."""
    d = out / NAVAL50_BOX_DIR
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in sorted({r["site_id"] for r in rows}):
        doc = naval50_box_doc(sid, rows, boxes.get(sid, {}))
        atomic.write_text(d / f"{sid}.json", json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
        n += 1
    return n


def compare_tables(old: dict, new: dict) -> dict:
    """Common scenes: cells changed (presence flipped, or a section only in one table) over the union of cells, per site and in all."""
    sites, n_all, c_all = {}, 0, 0
    for sid in sorted(set(old) | set(new)):
        o, n = old.get(sid, {}), new.get(sid, {})
        scenes = sorted(set(o) & set(n))
        cells = changed = 0
        for acq in scenes:
            keys = set(o[acq]) | set(n[acq])
            cells += len(keys)
            changed += sum(1 for k in keys if o[acq].get(k) != n[acq].get(k))
        sites[sid] = {"scenes": len(scenes), "scenes_old": len(o), "cells": cells, "changed": changed,
                      "share": round(changed / cells, 3) if cells else None}
        n_all += cells
        c_all += changed
    return {"sites": sites, "cells": n_all, "changed": c_all, "share": round(c_all / n_all, 3) if n_all else None}


def write(out: Path, res: dict) -> None:
    d = out / "eval"
    d.mkdir(parents=True, exist_ok=True)
    atomic.write_text(d / "presence.json", json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    atomic.write_text(d / "meta.json", json.dumps(meta(res), ensure_ascii=False, indent=1) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="presence / absence of a 50 m warship at the naval quays: the judge against the owner's labels (no fetch)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--snapshot", action="store_true", help="refresh eval/presence_snapshot.json (the regression guard) instead")
    ap.add_argument("--truth", choices=TRUTHS, default="s2", help="the truth of the body of presence.json (the other one goes under other_truth)")
    a = ap.parse_args(argv)
    if a.snapshot:
        old = a.out / "eval" / SNAPSHOT_FILE
        before = json.loads(old.read_text(encoding="utf-8"))["table"] if old.exists() else {}
        doc = snapshot(a.out)
        cmp = compare_tables(before, doc["table"])
        print(json.dumps({"scenes": {k: len(v) for k, v in doc["table"].items()}, "changed_vs_previous": cmp}, ensure_ascii=False), file=sys.stderr)
        return 0
    res = evaluate(a.out, truth=a.truth)
    write(a.out, res)
    write_blind(a.out)
    write_naval50(a.out)
    pres1.safe_write(a.out, load_sites())
    print(f"正解 {res['truth']}（other_truth {res['other_truth']['truth']}）", file=sys.stderr)
    for row in table(res):
        print(f"{row['name']}: 標本内 n={row['n']} 再現率={row['recall']} 適合率={row['precision']} ブライア={row['brier']}"
              f"（近い組 n={row['n_near']} {row['recall_near']}／{row['precision_near']}／{row['brier_near']}）；標本外 n={row['n_oos']} "
              f"再現率={row['recall_oos']} 適合率={row['precision_oos']} ブライア={row['brier_oos']}"
              f"（近い組 n={row['n_near_oos']} {row['recall_near_oos']}／{row['precision_near_oos']}／{row['brier_near_oos']}）（est-2 {row['n_est2']}／取り直し {row['n_recount']}、"
              f"撮影 {row['scenes']}、海軍でない線 {row['not_naval']}、撮影なし {row['no_scene']}；標本外 盲域 {row['n_blind_oos']}・非盲域 {row['n_clear_oos']} "
              f"{row['recall_clear_oos']}／{row['precision_clear_oos']}；他方の正解 n={row['n_other']} {row['recall_other']}／{row['precision_other']}）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
