"""dark-1: the quay sections the radar (Sentinel-1) cannot judge, marked apart from "0 ships" (miss-1 design §5.2-1, PR-2).

The radar count misses the submarines at their piers almost always (miss-1 §4.1: recall 0.05 at the piers hires-1 read as
submarine berths; no threshold, joining, length rule or baseline moves it). A 0 there means "the radar cannot see", not "no
ship". This module marks such sections; it changes no count (quay_naval50, quay_est2 and eval/presence.json are not touched).

A section is radar-dark for the scenes of a day ``d`` for one of two reasons:

  submarine_berth            hires-1 (docs/data/sat/labels/hires-berth-read.csv, verdict U) read it as a pier where submarines
                             lie. A trait of the berth, not an answer: it holds for every day
  owner_present_radar_none   of the evaluation cells of the section (eval/presence.json: the owner's Sentinel-2 answer y and the
                             radar's judgement p of the paired scene) dated before d, the owner said "present" at least
                             MIN_PRESENT times and the radar said present (p >= 0.5) not once. Only answers of days before d
                             count: the mark starts after the day of the MIN_PRESENT-th "present" (``since``) and ends with the
                             day the radar first said present (``until``, inclusive), if it ever did

    python -m sat.radar_dark build     # write the "radar_dark" list into sat/berth_class/<site>.json and docs/data/sat/radar_dark.json
    (the daily run: sat.eval_presence.refresh calls refresh() right after it writes eval/presence.json: docs/data/sat/radar_dark.json
    and quay_blind follow the answers by themselves; the record in sat/berth_class/, which the daily workflow does not publish,
    is rebuilt by hand, and the run warns while it is behind)
    python -m sat.radar_dark check     # the marks against the current hires-1 record and eval/presence.json (exit 1 on a drift)

Where the marks go:
  sat/berth_class/<site>.json "radar_dark"   [{section, way, radar_dark: true, reasons, since, until, basis}] (the naval-1 class
                                             of the lines is not touched; eval_presence.rule_versions hashes the file without
                                             this key, so the presence rules and their regression snapshot do not change)
  docs/data/sat/radar_dark.json              the same for every port (the site and the trial page read it), with the rule and the
                                             evidence (the evaluation cells the second reason was decided on)
  docs/data/sat/quay_blind/<site>.csv        per scene: radar_dark_sections (eligible sections that are dark on the scene's day),
                                             radar_dark_seen (of them, the radar count saw), radar_dark_reasons

Pure Python (no numpy), no network, PU 0. hires-1 was read from Esri World Imagery Wayback by the AI (berth_class.py
ESRI_WAYBACK_CREDIT); the owner's answers are of Sentinel-2 images (Contains modified Copernicus Sentinel data).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
CLASS_DIR = Path(__file__).resolve().parent / "berth_class"
HIRES = OUT / "labels" / "hires-berth-read.csv"
PUBLIC = "radar_dark.json"
KEY = "radar_dark"
VERSION = "dark-1"
MIN_PRESENT = 3            # miss-1 §3.1: "the owner 3 times or more present, the radar never"
THRESHOLD = 0.5            # the radar's present (eval_presence.THRESHOLD)
SUBMARINE = "submarine_berth"
OWNER_NONE = "owner_present_radar_none"
REASONS = {SUBMARINE: "hires-1 が潜水艦の桟橋と読んだ区間（高解像度の光学、AI の判読）。レーダーは潜水艦をほとんど数えない（miss-1 §4.1：再現率 0.05）",
           OWNER_NONE: f"所有者の Sentinel-2 の答えで {MIN_PRESENT} 回以上「在」なのに、対のレーダーの判定が一度も在にしない区間（答えは判定する日より前のものだけ）"}
NOTE = ("レーダー（Sentinel-1）では判定できない区間。0 隻は「居ない」ではなく「レーダーでは見えない」。札の数（quay_naval50）・est-2・評価票は変えない"
        "（印を出すだけ）。since の翌日から until の日まで（until が null なら以後ずっと）。submarine_berth は日付によらない。"
        "印の日付は撮影の日：過去の撮影の日付の答えが後から入ると、過去の撮影の行（quay_blind）に印が付くことがある。"
        "レーダーの判定は評価票の p（50 m 以上の片で在）を使う：評価票と札の対象が 50 m 以上の艦で、潜水艦（039 系は約 75 m）もこの線の上にあるため。"
        "長さを問わない p_any で数える案もある（短く測られた片も在とする。未採用）。毎日の実行（eval_presence.refresh）が評価票を書いた直後に作り直す。"
        "方針（司令部の決定 2026-11-22、数は変えない）：長さの規則は表示の札の規則（片の長さ − 20 m を山の数で割り 50 m 以上。−20 m は未較正の目安）を正とし、評価もこれに揃える"
        "（評価票の変更は別の PR）。この印の判定だけは札の規則ではなく全長で判定する（緩い側。印を増やさないため）。"
        "潜水艦の桟橋の在は「50 m 以上の艦（潜水艦を含む）」として軍艦の数に数える（別の欄に分けない）。"
        "印の区間を光学（pres-3o）の点数で埋める試行の系列は opt_fill.json（opt-fill-1、本体の数には足さない）")


def _hires(path: Path = HIRES) -> dict:
    """site -> {section: (way, note, read_date)} of the hires-1 sections read as submarine berths (verdict U)."""
    res: dict = {}
    if not Path(path).exists():
        return res
    with Path(path).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("verdict") == "U":
                res.setdefault(r["site_id"], {})[r["section_id"]] = (r.get("way"), r.get("note") or "", r.get("read_date") or "")
    return res


def evidence_of(presence: dict) -> dict:
    """site -> section -> [[date, y, s1], ...] (date order) of the evaluation cells with an answer y in {0, 1}, for the
    sections with at least MIN_PRESENT "present" answers (the others cannot be marked by the second reason)."""
    res: dict = {}
    for site_id, s in sorted((presence.get("sites") or {}).items()):
        by: dict = {}
        for r in s.get("rows") or []:
            if r.get("y") in (0, 1) and r.get("p") is not None:
                by.setdefault(r["section_id"], []).append([r["date"], int(r["y"]), int(float(r["p"]) >= THRESHOLD)])
        for sec, rows in sorted(by.items()):
            if sum(y for _, y, _ in rows) >= MIN_PRESENT:
                res.setdefault(site_id, {})[sec] = sorted(rows)
    return res


def owner_window(rows: list) -> tuple[str, str | None] | None:
    """(since, until) of the second reason from the cells [[date, y, s1], ...]: dark for the days d with since < d <= until
    (until None: no end), judged only on the cells dated before d. None when it never holds."""
    rows = sorted(rows)
    present = [d for d, y, _ in rows if y == 1]
    if len(present) < MIN_PRESENT:
        return None
    since = present[MIN_PRESENT - 1]
    seen = [d for d, _, s in rows if s == 1]
    until = seen[0] if seen else None
    if until is not None and until <= since:
        return None
    return since, until


def is_dark(entry: dict, day: str) -> list[str]:
    """The reasons a mark holds for the scenes of ``day`` (YYYY-MM-DD...); [] when none."""
    d = str(day)[:10]
    out = []
    for why in entry.get("reasons", []):
        if why == SUBMARINE:
            out.append(why)
        elif why == OWNER_NONE:
            w = entry.get("owner") or {}
            if w.get("since") and w["since"] < d and (w.get("until") is None or d <= w["until"]):
                out.append(why)
    return out


def build_marks(hires: dict, evidence: dict) -> dict:
    """site -> [entry] sorted by section: the union of the two reasons."""
    sites = sorted(set(hires) | set(evidence))
    res = {}
    for site_id in sites:
        ent = {}
        for sec, (way, note, rd) in sorted((hires.get(site_id) or {}).items()):
            ent.setdefault(sec, {"section": sec, "way": _way(way, sec), "radar_dark": True, "reasons": [], "basis": []})
            ent[sec]["reasons"].append(SUBMARINE)
            ent[sec]["basis"].append(f"hires-1（{rd}、AI の判読）：{note}".rstrip("："))
        for sec, rows in sorted((evidence.get(site_id) or {}).items()):
            w = owner_window(rows)
            if w is None:
                continue
            e = ent.setdefault(sec, {"section": sec, "way": _way(None, sec), "radar_dark": True, "reasons": [], "basis": []})
            e["reasons"].append(OWNER_NONE)
            e["owner"] = {"since": w[0], "until": w[1], "present": sum(y for _, y, _ in rows), "cells": len(rows),
                          "radar_present": sum(s for _, _, s in rows)}
            e["basis"].append(f"評価票（eval/presence.json）の升 {len(rows)}：所有者の在 {e['owner']['present']}、レーダーの在 {e['owner']['radar_present']}。"
                              f"{w[0]} の答えで {MIN_PRESENT} 回目の在")
        if ent:
            res[site_id] = [ent[k] for k in sorted(ent)]
    return res


def _way(way, sec: str):
    w = way if way not in (None, "") else str(sec).split(":")[0]
    try:
        return int(w)
    except (TypeError, ValueError):
        return w


def load_presence(out: Path = OUT) -> dict:
    p = Path(out) / "eval" / "presence.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def current(out: Path = OUT, hires_path: Path = HIRES) -> tuple[dict, dict]:
    """(marks, evidence) from the current hires-1 record and eval/presence.json."""
    ev = evidence_of(load_presence(out))
    return build_marks(_hires(hires_path), ev), ev


def marks(site_id: str, out: Path = OUT) -> dict:
    """section -> entry of the published list (docs/data/sat/radar_dark.json, made by refresh() from the current evaluation
    sheet): what quay_blind reads. {} when none."""
    p = Path(out) / PUBLIC
    if not p.exists():
        return {}
    return {e["section"]: e for e in (json.loads(p.read_text(encoding="utf-8")).get("sites") or {}).get(site_id) or []}


def class_marks(site_id: str, class_dir: Path = CLASS_DIR) -> list:
    """The record in the site's class file (sat/berth_class/<site>.json "radar_dark")."""
    p = Path(class_dir) / f"{site_id}.json"
    return (json.loads(p.read_text(encoding="utf-8")).get(KEY) or []) if p.exists() else []


def stale_class_files(want: dict, class_dir: Path = CLASS_DIR) -> list[str]:
    """The sites whose class-file record differs from the marks ``want`` (site -> [entry])."""
    return [p.stem for p in sorted(Path(class_dir).glob("*.json")) if class_marks(p.stem, class_dir) != want.get(p.stem, [])]


def refresh(out: Path = OUT, class_dir: Path = CLASS_DIR) -> dict:
    """The daily part (PU 0, the same result for the same files): the marks from the hires-1 record and the evaluation sheet just
    written, into docs/data/sat/radar_dark.json only (the class files are not touched). Returns {"counts", "stale"}: stale =
    the sites whose sat/berth_class record is behind (python -m sat.radar_dark build)."""
    ev = evidence_of(load_presence(out))
    m = build_marks(_hires(Path(out) / "labels" / "hires-berth-read.csv"), ev)
    # a set of files that never had a list and gives no mark (Zhanjiang alone: tests/sat_zjg_golden.py) gets no new file
    if not m and not (Path(out) / PUBLIC).exists():
        return {"counts": {}, "stale": []}
    return {"counts": write_public(m, ev, out), "stale": stale_class_files(m, class_dir)}


def class_bytes_without_marks(text: str) -> bytes:
    """The bytes a class file had before the marks were put in (the file is json.dumps(indent=1, ensure_ascii=False) + "\\n",
    as sat/berth_class.py writes it, and the marks are its last key): what the presence rules hash."""
    doc = json.loads(text)
    if KEY not in doc:
        return text.encode("utf-8")
    doc.pop(KEY)
    return (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def write(marks_by_site: dict, evidence: dict, out: Path = OUT, class_dir: Path = CLASS_DIR, source: str = "") -> dict:
    """Put the marks into the class files (only the "radar_dark" key: added last, replaced, or removed when a site has none)
    and write docs/data/sat/radar_dark.json. Returns {site: number of marks}."""
    for p in sorted(Path(class_dir).glob("*.json")):
        text = p.read_text(encoding="utf-8")
        doc = json.loads(text)
        doc.pop(KEY, None)
        if marks_by_site.get(p.stem):
            doc[KEY] = marks_by_site[p.stem]
        new = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
        if new != text:
            p.write_text(new, encoding="utf-8")
    return write_public(marks_by_site, evidence, out, source)


def write_public(marks_by_site: dict, evidence: dict, out: Path = OUT, source: str = "") -> dict:
    """docs/data/sat/radar_dark.json (left alone when its content would not change). Returns {site: number of marks}."""
    pres = load_presence(out)
    doc = {"version": VERSION, "note": NOTE, "reasons": REASONS, "min_present": MIN_PRESENT, "threshold": THRESHOLD,
           "sources": {"hires": "docs/data/sat/labels/hires-berth-read.csv（verdict U）", "evaluation": f"docs/data/sat/eval/presence.json（{pres.get('version')}）"},
           # no time stamp of the sheet: it is rewritten every day, and the list only when the marks or their evidence change
           "credit": ["区分の一部は Esri World Imagery Wayback の画像の判読（hires-1、2026-11-18、読んだのは AI）。Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community; imagery not redistributed",
                      "所有者の答えは Sentinel-2 の画像の目視。Contains modified Copernicus Sentinel data 2023-2026"],
           "null": "since・until が null：submarine_berth だけの印（日付によらない）、または終わりの無い印",
           "counts": {s: len(v) for s, v in sorted(marks_by_site.items())},
           "sites": marks_by_site, "evidence": evidence}
    Path(out).mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    p = Path(out) / PUBLIC
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.write_text(text, encoding="utf-8")
    return doc["counts"]


def check(out: Path = OUT, class_dir: Path = CLASS_DIR) -> list[str]:
    """Drift: the marks of the files against the marks the current hires-1 record and eval/presence.json give."""
    want, ev = current(out)
    bad = []
    for p in sorted(Path(class_dir).glob("*.json")):
        have = json.loads(p.read_text(encoding="utf-8")).get(KEY) or []
        if have != want.get(p.stem, []):
            bad.append(f"{p.stem}: the marks of sat/berth_class/{p.name} differ from the current record (python -m sat.radar_dark build)")
    pub = Path(out) / PUBLIC
    doc = json.loads(pub.read_text(encoding="utf-8")) if pub.exists() else {}
    if doc.get("sites") != want or doc.get("evidence") != ev:
        bad.append(f"{PUBLIC}: differs from the current record (python -m sat.radar_dark build)")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "check"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m, ev = current()
        print(json.dumps(write(m, ev), ensure_ascii=False))
        return 0
    bad = check()
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
