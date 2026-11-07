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
* otherwise the stored Sentinel-1 re-count of Zhanjiang's labelled days (labels/ZJG-s2-s1-match.csv): p = 1 when s1_occupied
  is 1 and s1_occ_len_m >= MIN_LENGTH_M, else 0 (a 0/1 judge: its Brier score is its error rate).
* a scene stored in neither is not judged: the cell is counted under ``no_scene`` and left out of the metrics (an unstored scene
  is not "absent").

Metrics per site and over all sites, in sample (label day before the site's label_freeze_utc) and out of sample (from the freeze
on) apart: confusion matrix at p >= 0.5 (tp, fp, fn, tn), recall = tp / (tp + fn), precision = tp / (tp + fp), Brier = mean (p - y)^2,
the base rate of the owner's "present" and the judge's mean p. ``any_length`` repeats the metrics with the length rule off
(the owner's label does not carry the length of the ship: a quay occupied by a boat under 50 m is "present" to the owner and
"absent" to the 50 m target; this variant shows how much of the disagreement is that).

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

from . import berths

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
EVAL_VERSION = "eval-presence-1"
MIN_LENGTH_M = 50.0
SECTION_MATCH_M = 20.0        # as sat/quay_est2.SECTION_MATCH_M (not imported: quay_est2 needs numpy)
THRESHOLD = 0.5
FAR_H = 48.0                  # |S1 - S2| beyond this is "far" in the pairing summary
SOURCES = ("est-2", "s1-recount")
NAMES = {"ZJG": "湛江", "SYL": "三亜・楡林", "SLP": "三亜・亜龍湾", "QGZ": "青島・古鎮口", "ZDH": "舟山・定海",
         "LSN": "大連・旅順", "QDH": "青島港内", "QJG": "青島・姜哥荘", "XSN": "象山", "SWS": "汕尾", "STU": "汕頭",
         "XMN": "厦門", "MAW": "馬尾"}


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
        s["naval"] = s["class"] == "naval"
        s["eligible"] = s["naval"] and s["length_m"] >= MIN_LENGTH_M
    return secs


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


def s1_match(out: Path, site_id: str) -> dict:
    """Zhanjiang's re-count: {(section_id, date): row} of labels/<site>-s2-s1-match.csv."""
    return {(r["section_id"], r["s2_date"]): r for r in read_csv(out / "labels" / f"{site_id}-s2-s1-match.csv")}


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


def stretch_of_section(sec: dict, feats: list[dict], stretches: list[dict], lat0: float):
    """The est-2 stretch (same OSM way and side, the most points within SECTION_MATCH_M) or None. A stretch carries no
    geometry: its ``n`` is the ``n`` of the scene's quay feature (quay/<site>/*.geojson), whose line is used. est-2 keeps
    fewer stretches than the scene has features (berth-1 / naval-1 drop some), so the pairing is by ``n``, not by position."""
    sxy = _xy(sec["line"], lat0)
    by_n = {f.get("n"): f for f in feats}
    best = None
    for st in stretches:
        f = by_n.get(st.get("n"))
        if f is None or str(st.get("osm_way")) != str(sec["way_id"]) or int(st.get("side", _side(f))) != int(sec["side"]):
            continue
        n = sum(1 for p in sxy if _min_dist(p, _xy(f["coords"], lat0)) <= SECTION_MATCH_M)
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
    if r is None or r.get("s1_occupied") in (None, ""):
        return None
    if r["s1_occupied"] != "1":
        return 0.0
    if min_len and float(r.get("s1_occ_len_m") or 0) < min_len:
        return 0.0
    return 1.0


# ---------------------------------------------------------------- metrics
def metrics(rows: list[dict], key: str = "p") -> dict:
    n = len(rows)
    m = {"n": n, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "recall": None, "precision": None, "accuracy": None, "brier": None,
         "base_rate": None, "mean_p": None}
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
    return m


def _freeze(site: dict) -> str:
    return (site.get("quay") or {}).get("label_freeze_utc") or "2026-10-27"


# ---------------------------------------------------------------- one site
def evaluate_site(out: Path, site: dict) -> dict:
    sid = site["id"]
    freeze = _freeze(site)
    secs = site_sections(out, site)
    lab = labels(out, sid)
    days = s1_of_days(out, sid)
    match = s1_match(out, sid)
    est = est2_scenes(out, sid)
    feats = quay_features(out, sid)
    own = calibrated(out, sid)
    lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 21.2)
    cnt = {"answers": len(lab), "unknown_section": 0, "not_naval": 0, "short": 0, "no_s1_pairing": 0, "no_scene": 0,
           "other_day_of_scene": 0, "used": 0}
    rows = []
    for a in lab:
        s = secs.get(a["section_id"])
        if s is None:
            cnt["unknown_section"] += 1
            continue
        if not s["naval"]:
            cnt["not_naval"] += 1
            continue
        if s["length_m"] < MIN_LENGTH_M:
            cnt["short"] += 1
            continue
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
            st = stretch_of_section(s, fs, sts, lat0) if fs else None
            p, p_any, src = judge_est2(st, MIN_LENGTH_M), judge_est2(st, 0.0), "est-2"
            length = None if st is None else st.get("ship_length_m")
        else:
            p, p_any = judge_recount(mr, MIN_LENGTH_M), judge_recount(mr, 0.0)
            src, length = "s1-recount", None if mr is None else (mr.get("s1_occ_len_m") or None)
            if p is None:
                cnt["no_scene"] += 1
                continue
        rows.append({"section_id": a["section_id"], "date": a["date"], "s1": s1, "dt_h": dt, "source": src, "y": a["y"],
                     "owner_c": a["verified_c"], "p": round(p, 3), "p_any": round(p_any, 3), "length_m": length,
                     "oos": (not own) or a["date"] >= freeze})
    # one label day per Sentinel-1 scene: the nearest in time (two S2 days within +-4 days of the same S1 scene would count
    # the same judgement twice); the other days' rows are dropped
    rows, dropped = nearest_day_per_scene(rows)
    cnt["other_day_of_scene"] = dropped
    cnt["used"] = len(rows)
    ins = [r for r in rows if not r["oos"]]
    oos = [r for r in rows if r["oos"]]
    n_el = sum(1 for s in secs.values() if s["eligible"])
    scenes = sorted({r["s1"] for r in rows})
    far = [r for r in rows if r["dt_h"] is not None and abs(float(r["dt_h"])) > FAR_H]
    return {"site_id": sid, "name": NAMES.get(sid, site.get("name")), "freeze_utc": freeze, "calibrated": own,
            "sample_note": ("標本内＝ラベルの日が凍結より前（est-2 の較正に使った期間）、標本外＝凍結以後" if own else
                            "est-2 はこの港の正解で較正していない（湛江の重みの流用）ので、全部を標本外とする"),
            "sections": {"labelled": len(secs), "naval": sum(1 for s in secs.values() if s["naval"]), "eligible": n_el},
            "labels": cnt, "scenes_judged": scenes,
            "pairing": {"scenes": len(scenes), "far_h": FAR_H, "far": len(far), "far_share": round(len(far) / len(rows), 3) if rows else None},
            "by_source": {src: {"n": sum(1 for r in rows if r["source"] == src),
                                "min50": metrics([r for r in rows if r["source"] == src]),
                                "any_length": metrics([r for r in rows if r["source"] == src], "p_any")} for src in SOURCES},
            "in_sample": {"min50": metrics(ins), "any_length": metrics(ins, "p_any")},
            "out_of_sample": {"min50": metrics(oos), "any_length": metrics(oos, "p_any")},
            "status": _status(len(ins), len(oos), own), "rows": rows}


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


def evaluate(out: Path, sites: list[dict] | None = None) -> dict:
    sites = sites or load_sites()
    ids = []
    for s in sites:
        sid = s["id"]
        if (out / "labels" / sid / "review.json").exists() or (out / "labels" / f"{sid}-s2-labels.csv").exists() or \
                (out / "labels" / f"{sid}-s2-answers.csv").exists():
            ids.append(sid)
    res = {sid: evaluate_site(out, next(s for s in sites if s["id"] == sid)) for sid in ids}
    all_rows = [r for r in res.values() for r in r["rows"]]
    ins = [r for r in all_rows if not r["oos"]]
    oos = [r for r in all_rows if r["oos"]]
    tot = {"in_sample": {"min50": metrics(ins), "any_length": metrics(ins, "p_any")},
           "out_of_sample": {"min50": metrics(oos), "any_length": metrics(oos, "p_any")},
           "by_source": {src: {"n": sum(1 for r in all_rows if r["source"] == src), "min50": metrics([r for r in all_rows if r["source"] == src])}
                         for src in SOURCES}}
    return {"version": EVAL_VERSION, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target": "海軍の岸壁（naval-1 の naval の線）に着く 50 m 以上の水上艦の、撮影×岸壁区間ごとの在/不在",
            "min_length_m": MIN_LENGTH_M, "threshold": THRESHOLD, "total": tot, "sites": res}


# ---------------------------------------------------------------- meta
def table(res: dict) -> list[dict]:
    rows = []
    for sid, r in res["sites"].items():
        m, o = r["in_sample"]["min50"], r["out_of_sample"]["min50"]
        rows.append({"site_id": sid, "name": r["name"], "calibrated": r["calibrated"], "sections_eligible": r["sections"]["eligible"],
                     "answers": r["labels"]["answers"],
                     "n": m["n"], "recall": m["recall"], "precision": m["precision"], "brier": m["brier"], "base_rate": m["base_rate"],
                     "n_oos": o["n"], "recall_oos": o["recall"], "precision_oos": o["precision"], "brier_oos": o["brier"],
                     "n_est2": r["by_source"]["est-2"]["n"], "n_recount": r["by_source"]["s1-recount"]["n"],
                     "scenes": r["pairing"]["scenes"], "far_share": r["pairing"]["far_share"],
                     "not_naval": r["labels"]["not_naval"], "no_scene": r["labels"]["no_scene"] + r["labels"]["no_s1_pairing"],
                     "other_day_of_scene": r["labels"]["other_day_of_scene"], "status": r["status"]})
    t = res["total"]["in_sample"]["min50"]
    to = res["total"]["out_of_sample"]["min50"]
    n_all = t["n"] + to["n"]
    far = sum(r["pairing"]["far"] for r in res["sites"].values())
    rows.append({"site_id": "ALL", "name": "全港", "calibrated": None, "sections_eligible": sum(x["sections_eligible"] for x in rows),
                 "answers": sum(x["answers"] for x in rows),
                 "n": t["n"], "recall": t["recall"], "precision": t["precision"], "brier": t["brier"], "base_rate": t["base_rate"],
                 "n_oos": to["n"], "recall_oos": to["recall"], "precision_oos": to["precision"], "brier_oos": to["brier"],
                 "n_est2": res["total"]["by_source"]["est-2"]["n"], "n_recount": res["total"]["by_source"]["s1-recount"]["n"],
                 "scenes": sum(x["scenes"] for x in rows), "far_share": round(far / n_all, 3) if n_all else None,
                 "not_naval": sum(x["not_naval"] for x in rows), "no_scene": sum(x["no_scene"] for x in rows),
                 "other_day_of_scene": sum(x["other_day_of_scene"] for x in rows), "status": _status(t["n"], to["n"])})
    return rows


def meta(res: dict) -> dict:
    return {
        "generated_utc": res["generated_utc"],
        "version": EVAL_VERSION,
        "files": {"presence.json": "評価票の本体：全体と港別の指標、撮影×区間の行（rows）", "meta.json": "この定義と港別の表（table）"},
        "target": res["target"],
        "unit": "撮影×岸壁区間。区間は sat/s2_labels.sections() の約 150 m の岸壁線の片（確認ページの区間、湛江は labels/ZJG-s2-sections.geojson）",
        "sections": f"naval-1（sat/berth_class/<地点>.json）で区間の中点の線の区分が naval（海軍の桟橋・岸壁。設定が足す区間を含む）で、区間の長さ（length_m）が {MIN_LENGTH_M:g} m 以上のもの。civil・unknown の線の区間と {MIN_LENGTH_M:g} m 未満の区間は数えない（labels.not_naval・labels.short）",
        "truth": "所有者の目視（labels/<地点>-s2-answers.csv、-oos-answers.csv、湛江は ZJG-s2-labels.csv の verified 1 の行）：verified_occupied 1＝在、0＝不在。判断できない（verified 0）は除く。所有者の答えは船の長さを持たない（50 m 未満の船の着いた区間も「在」になる）ので、any_length（長さの規則なし）も併記する",
        "pairing": "ラベルの日（S2 の晴れの日）の S1 撮影＝確認セット（labels/<地点>/review*.json の days.s1_near[0]）の対応：前後 4 日以内で基準のある軌道の最も近い撮影（sat/s2_answers.eligible_days）。湛江は labels/ZJG-s2-s1-match.csv の s1_acquired_utc（sat/s2_labels.nearest_s1、同じ規則）。dt_h＝S1−S2（時間）、同じ日でも約 7 時間ずれ、船が入れ替わることがある",
        "judge": {
            "est-2": f"その撮影が quay_est2.json・quay_est2/<地点>.json にあるとき：区間と同じ OSM の線・向きで {SECTION_MATCH_M:g} m 以内の点が最も多い区間片（stretch）。p＝p_real（ship_length_m ≥ {MIN_LENGTH_M:g} m のとき、未満は 0）。片が無ければ 0",
            "s1-recount": f"est-2 に無い撮影（湛江のラベル日の S1 取り直し、labels/ZJG-s2-s1-match.csv）：s1_occupied 1 かつ s1_occ_len_m ≥ {MIN_LENGTH_M:g} m で 1、それ以外 0（0/1 の判定：ブライアは誤り率に等しい）",
            "no_scene": "どちらにも無い撮影は判定しない（labels.no_scene・no_s1_pairing）。保存の無い撮影を「不在」とはしない。港の確認セットの日の S1 は多くが未保存（s1_recount_plan.json の取り直し待ち）",
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
        },
        "table_columns": {"site_id": "地点", "name": "地名", "calibrated": "est-2 をその港の正解で較正したか（false＝湛江の流用、全部標本外）",
                          "sections_eligible": "対象の区間数（確認セットのうち naval・50 m 以上）", "answers": "所有者の答え（判断できないを除く）",
                          "n": "標本内の撮影×区間", "recall": "再現率", "precision": "適合率", "brier": "ブライア", "base_rate": "在の割合",
                          "n_oos・recall_oos・precision_oos・brier_oos": "標本外の同じ指標",
                          "n_est2 / n_recount": "判定の出所別の升の数（est-2 の区間片／湛江の S1 取り直しの占有）",
                          "scenes": "判定に使った S1 撮影の数", "far_share": f"|S1−S2| > {FAR_H:g} 時間の升の割合",
                          "not_naval": "naval でない線の答え（除外）", "no_scene": "S1 の対応か保存が無い答え（除外）",
                          "other_day_of_scene": "同じ S1 撮影に対応した別のラベル日の答え（最も近い日だけ使う。除外）"},
        "table": table(res),
        "limits": [
            "湛江の見出しの数字（n の大半が s1-recount）は S1 取り直しの占有（quay-v1 の規則）の評価で、est-2 の評価は n_est2 の升だけ。両方の判定を混ぜて一つの指標にしている",
            f"{MIN_LENGTH_M:g} m の規則は出所で別の量に当たる：est-2 は艦の長さの推定（ship_length_m、区間をまたぐ片の全長）、s1-recount は区間の中の占有の長さ（s1_occ_len_m）。"
            "60 m の艦が区間の境にまたがると s1_occ_len_m は 30／30 に割れて両区間が「不在」になる（湛江 2026-01-03 の見逃しがこの型）。所有者の答えは区間に少しでも船が入れば「在」",
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
        cls[p.stem] = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
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
                st = stretch_of_section(sec, fs, sts, lat0) if fs else None
                row[sec_id] = int(judge_est2(st, MIN_LENGTH_M) >= THRESHOLD)
            tab[acq] = row
        if tab:
            res[sid] = tab
    return res


def snapshot(out: Path, sites: list[dict] | None = None) -> dict:
    sites = sites or load_sites()
    doc = {"version": EVAL_VERSION, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "days": SNAPSHOT_DAYS, "max_change": SNAPSHOT_MAX_CHANGE, "rules": rule_versions(sites),
           "min_site_cells": SNAPSHOT_MIN_SITE_CELLS,
           "note": ("回帰の見張り：規則（berths.json とそのハッシュ、naval-1 の区分ファイルのハッシュ、quay、岸壁線の版）が写しと違うとき、直近 30 日の撮影の在/不在"
                    "（est-2、p ≥ 0.5、50 m 以上、確認セットの naval の区間）を写しと比べ、共通の升の 2 割を超えて変わると（全体、または升が 30 以上の港のどれか）"
                    "tests/test_sat_eval_presence.py が失敗する。写しの撮影のうち半分未満しか見つからない港があっても失敗（写しが古い）。規則の版が同じ日は毎日の実行（sat/run.py）が"
                    "写しを更新する。規則を変えた PR では python -m sat.eval_presence --snapshot で写しを更新し、変わった升の数を PR に書く"),
           "table": presence_table(out, sites)}
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "eval" / SNAPSHOT_FILE).write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def refresh(out: Path, sites: list[dict] | None = None, log=None) -> dict:
    """The daily run (sat/run.py save): rewrite eval/presence.json and eval/meta.json from the stored files (PU 0), and the
    regression snapshot when its rule versions are those of the working tree (a rule change refreshes it by hand, after the
    guard has been looked at). Returns {"presence": n rows, "snapshot": "updated" | "kept" | "created"}."""
    sites = sites or load_sites()
    res = evaluate(out, sites)
    write(out, res)
    p = out / "eval" / SNAPSHOT_FILE
    state = "kept"
    if not p.exists():
        snapshot(out, sites)
        state = "created"
    else:
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("rules") == rule_versions(sites):
            snapshot(out, sites)
            state = "updated"
    r = {"presence": sum(len(x["rows"]) for x in res["sites"].values()), "snapshot": state}
    if log:
        log(f"eval/presence.json: {r['presence']} 升、写し {state}")
    return r


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
    (d / "presence.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta(res), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="presence / absence of a 50 m warship at the naval quays: the judge against the owner's labels (no fetch)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--snapshot", action="store_true", help="refresh eval/presence_snapshot.json (the regression guard) instead")
    a = ap.parse_args(argv)
    if a.snapshot:
        old = a.out / "eval" / SNAPSHOT_FILE
        before = json.loads(old.read_text(encoding="utf-8"))["table"] if old.exists() else {}
        doc = snapshot(a.out)
        cmp = compare_tables(before, doc["table"])
        print(json.dumps({"scenes": {k: len(v) for k, v in doc["table"].items()}, "changed_vs_previous": cmp}, ensure_ascii=False), file=sys.stderr)
        return 0
    res = evaluate(a.out)
    write(a.out, res)
    for row in table(res):
        print(f"{row['name']}: 標本内 n={row['n']} 再現率={row['recall']} 適合率={row['precision']} ブライア={row['brier']}；標本外 n={row['n_oos']} "
              f"再現率={row['recall_oos']} 適合率={row['precision_oos']} ブライア={row['brier_oos']}（est-2 {row['n_est2']}／取り直し {row['n_recount']}、"
              f"撮影 {row['scenes']}、海軍でない線 {row['not_naval']}、撮影なし {row['no_scene']}）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
