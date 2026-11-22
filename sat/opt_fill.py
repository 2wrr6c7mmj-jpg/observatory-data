"""opt-fill-1 (trial series): the radar-dark sections of a radar scene, filled from the optical judge pres-3o (Sentinel-2).

The radar (Sentinel-1) cannot judge the sections of radar_dark.json (dark-1, sat/radar_dark.py): there a 0 means "the radar
cannot see", not "no ship". The daily confirm run (confirm-1, sat/confirm.py) already scores each quay section x Sentinel-2
day with pres-3o (docs/data/sat/confirm/state/<site>/<month>.csv, column p; it uses no S1). This module puts the two side by
side, as a trial series apart from the counts (2部の評価「光学で埋める」§4, the rule opt-fill-1):

  sections   of a scene of quay_blind (an S1 scene of a site with a resident table): the eligible sections that are radar-dark
             on the scene's day, except
               (a) the ones the radar count saw in that scene (radar_dark_seen),
               (b) the resident sections of blind-2 (already one ship each in warship_blind: never added twice),
               (c) the ones not in the confirm-1 section ledger (docs/data/sat/confirm/sections/<site>.json, status active)
  value      the Sentinel-2 day nearest to the scene's day within WINDOW_D days (either side; a tie takes the earlier day):
             the owner's answer of that day when there is one (labels/<site>-s2-answers.csv and -confirm-answers.csv, read
             as the confirm series reads them: verified_by owner, the later file wins), else the pres-3o score p of the state. A day without an image
             (no_image, no p) or that the owner could not judge is not used. No usable day in the window: not filled
  present    owner "present", or p >= THRESHOLD: one ship for the section (the lower bound, as the resident sections)

Columns added at the end of quay_blind/<site>.csv (eval_presence.BLIND_COLUMNS); no earlier column changes, and nothing here
is added to quay_naval50, quay_est2, warship_blind, est-2 or eval/presence.json:
  opt_fill_version    VERSION ("" when the series could not be made that day: the columns are empty, the counts as always)
  opt_fill_sections   the sections of the rule (after (a)-(c))
  opt_fill_scored     of them, those with a usable Sentinel-2 day in the window
  opt_fill_present    of them, present (= the ships the trial would add)
  opt_fill_basis      per scored section "section|S2 day|S2 day - S1 day (d)|owner or p|value", ";"-joined
The rule, the caveats and the sources go to docs/data/sat/opt_fill.json (OPT_FILL).

Zhanjiang (ZJG) is left out (EXCLUDED; it has neither marks nor a quay_blind file; tests/sat_zjg_golden.py). Pure Python (no
numpy), reads published files only, no network, PU 0. Sentinel-2 images: Contains modified Copernicus Sentinel data.
The confirm series' code stays outside the daily collection (tests/test_sat_confirm.py NotWiredTest): this module reads its
published files (state, section ledger, answers) with its own small readers below and imports none of its modules.
"""
from __future__ import annotations

import csv
import json
from datetime import date as _date
from pathlib import Path

CONFIRM_DIR = "confirm"   # docs/data/sat/confirm/ (state/<site>/<month>.csv, sections/<site>.json): published by the confirm run

VERSION = "opt-fill-1"
THRESHOLD = 0.9        # 2部の評価 §4：前進検証で合計が所有者と釣り合う（43 対 42）、誤りの増え方が小さい（+3）
WINDOW_D = 4           # the S1-S2 pairing of the evaluation sheet (sat/s2_answers.eligible_days): 4 days either side
EXCLUDED = ("ZJG",)    # Zhanjiang: left out (its outputs must not change)
OPT_FILL = "opt_fill.json"
COLUMNS = ("opt_fill_version", "opt_fill_sections", "opt_fill_scored", "opt_fill_present", "opt_fill_basis")
EMPTY = {k: "" for k in COLUMNS}

RULE = (f"S1 の撮影の日に radar_dark（dark-1）の印がある対象の区間から、(a) その撮影でレーダーが数えた区間、(b) blind-2 の常在の区間（すでに 1 隻足されている）、"
        f"(c) 確認回し（confirm-1）の区間の表に無い区間を除く。値は S1 の撮影の日の前後 {WINDOW_D} 日以内でいちばん近い Sentinel-2 の日のもの（同じ近さなら前の日）："
        f"その日に所有者の答えがあれば答え、無ければ pres-3o の点数 p（確認回しの state の p）。像の無い日（no_image）と所有者が判断できないとした日は使わない。"
        f"窓に使える日が無ければ埋めない。所有者の在、または p ≥ {THRESHOLD:g} で区間 1 隻（常在と同じ下限の数え方）")
NOTE = ("試行の系列。公開の本体の数（quay_naval50 の札・quay_est2・warship_blind・評価票）には足さない。画面に出すときは「光学で補った数（試行）」と書き、本体の数字と分ける。"
        "8 週間後（凍結 2026-11-22 以降の答えで）0.9 と 0.7 を比べ直し、足した升の適合率が 0.8 を超えるまでは本体に足さない")
CAVEATS = [
    "升が少ない：2部の評価の前進検証で、その日に印のある升 177・28 撮影、在のある港は 4 つ（象山・姜哥荘・青島港内・旅順）。95% CI は広い",
    "点数は過信気味：前進検証で p 0.70〜0.90 の帯は平均 p 0.83 に対して実際の在 0.67、0.90〜0.97 の帯は 0.92 に対して 0.63（升が少ない）",
    f"S1 と S2 の日のずれ：評価は所有者の答えた S2 の日そのものの升で、運用での S1 の撮影と S2 の日のずれ（最大 {WINDOW_D} 日）の分の誤りは測っていない。ずれの間に艦が出入りする",
    "像なしの升：11 月の dark の区間の升（区間 × S2 の日）177 のうち 80 は像が無い（no_image）。窓の中に使える日が無い区間は埋めない（scored に入らない）",
    "大榭島・下川島・楡林など確認回しの区間の表に無い港・区間は対象外（(c)）。旅順など常在の表の無い港は quay_blind の行が無いので、この系列も無い",
    "足す区間はほぼ潜水艦の桟橋（hires-1 の U）。所有者が 8 回とも不在と答えた U の区間もあり、正解の側にも穴がある",
    "pres-3o は区間の答えの履歴を特徴に使う：答えの少ない区間では弱い",
]
# 司令部の決定（所有者に代わって、2026-11-22。数は変えない）
POLICY = {
    "submarine": ("潜水艦の桟橋の在は「50 m 以上の艦（潜水艦を含む）」として軍艦の数に数える（既定。別の欄には分けない）。所有者への質問が「50 m 以上の船がいるか」で、"
                  "配属の数（分母）も潜水艦を含むため。詳細の注記で「うち潜水艦の桟橋の分」を示すのは後のサイトの作業"),
    "length": ("長さの規則は表示の札の規則（片の長さ − 20 m を山の数で割り 50 m 以上）を正とし、評価もこれに揃える方針（評価票の変更は別の PR）。"
               "−20 m は未較正の目安。radar_dark の印の判定だけは、緩い側として今の全長 ≥ 50 m のまま"),
}
SOURCES = {
    "marks": "docs/data/sat/radar_dark.json（dark-1）",
    "scores": "docs/data/sat/confirm/state/<港>/<月>.csv の p（pres-3o、確認回し confirm-1。S1 は使わない）",
    "answers": "docs/data/sat/labels/<港>-s2-answers.csv・<港>-confirm-answers.csv（verified_by owner）",
    "sections": "docs/data/sat/confirm/sections/<港>.json（status active）",
    "resident": "sat/blind_prior（blind-2、quay_blind の resident_sections）",
    "evaluation": "2部の評価「光学で埋める」（2026-11-22）§4 の規則の案 opt-fill-1",
}
CREDIT = ["Contains modified Copernicus Sentinel data 2023-2026（Sentinel-1・Sentinel-2。pres-3o の点数と所有者の答えは Sentinel-2 L2A の像による）",
          "区分の一部（潜水艦の桟橋）は Esri World Imagery Wayback の画像の判読（hires-1、2026-11-18、読んだのは AI）。Source: Esri, Vantor, Earthstar Geographics, and the GIS User Community; imagery not redistributed"]


def _day(s: str) -> _date:
    return _date.fromisoformat(str(s)[:10])


def _csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def load_state(out: Path, site_id: str) -> dict:
    """(section_id, date) -> row of the confirm state (docs/data/sat/confirm/state/<site>/*.csv; a later month file wins)."""
    d = Path(out) / CONFIRM_DIR / "state" / site_id
    return {(r["section_id"], r["date"]): r for f in (sorted(d.glob("*.csv")) if d.exists() else []) for r in _csv(f)}


def active_sections(out: Path, site_id: str) -> set:
    """The ids of the active sections of the confirm ledger (docs/data/sat/confirm/sections/<site>.json); none: empty."""
    p = Path(out) / CONFIRM_DIR / "sections" / f"{site_id}.json"
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {s["id"] for s in doc.get("sections", []) if s.get("status", "active") == "active"}


def owner_answers(out: Path, site_id: str) -> dict:
    """(section_id, date) -> 1 / 0 / None (cannot judge) of the owner's answers: labels/<site>-s2-answers.csv, then
    labels/<site>-confirm-answers.csv (the later one wins); rows of verified_by owner only (as the confirm series reads them)."""
    res = {}
    for name in (f"{site_id}-s2-answers.csv", f"{site_id}-confirm-answers.csv"):
        for r in _csv(Path(out) / "labels" / name):
            v, occ = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip()
            if (r.get("verified_by") or "").strip() != "owner" or (r.get("site_id") or site_id) != site_id:
                continue
            if v == "1" and occ in ("0", "1"):
                res[(r["section_id"], r["date"])] = int(occ)
            elif v == "0":
                res[(r["section_id"], r["date"])] = None
    return res


def site_cells(out: Path, site_id: str) -> dict:
    """section -> {S2 day: ("owner", 0/1) or ("p", float)} of the usable Sentinel-2 days of the site: the owner's answer when
    there is one (an answer "cannot judge" makes the day unusable), else the confirm state's p (no_image or no p: not used)."""
    cells: dict = {}
    for (sec, day), r in load_state(out, site_id).items():
        if (r.get("state") or "") == "no_image":
            continue
        p = _num(r.get("p"))
        if p is not None:
            cells.setdefault(sec, {})[day[:10]] = ("p", p)
    for (sec, day), y in owner_answers(out, site_id).items():
        d = cells.setdefault(sec, {})
        if y is None:
            d.pop(day[:10], None)
        else:
            d[day[:10]] = ("owner", y)
    return {k: v for k, v in cells.items() if v}


def prepare(out: Path, site_ids) -> dict:
    """site -> context for fill(): the active sections of the confirm ledger and the usable cells. Excluded sites get none."""
    res = {}
    for sid in site_ids:
        if sid in EXCLUDED:
            continue
        res[sid] = {"sections": active_sections(out, sid), "cells": site_cells(out, sid), "rows": []}
    return res


def nearest(days: dict, acq: str):
    """(S2 day, signed days S2 - S1, (kind, value)) of the usable day nearest to the scene within WINDOW_D (tie: the earlier)."""
    d0 = _day(acq)
    best = None
    for day in sorted(days):
        dt = (_day(day) - d0).days
        if abs(dt) > WINDOW_D:
            continue
        if best is None or abs(dt) < abs(best[1]):
            best = (day, dt, days[day])
    return best


def is_present(kind: str, v) -> bool:
    return v == 1 if kind == "owner" else float(v) >= THRESHOLD


def fill(ctx: dict | None, acq: str, dark: list, seen: dict, resident) -> dict:
    """The opt-fill columns of one scene. dark = the eligible sections that are radar-dark on the scene's day, seen = section ->
    0/1 of the radar count of the scene, resident = the resident sections (blind-2). ctx None: the empty columns."""
    if ctx is None:
        return dict(EMPTY)
    res_set = set(resident)
    secs = [k for k in sorted(dark) if not seen.get(k) and k not in res_set and k in ctx["sections"]]
    basis, present = [], 0
    for k in secs:
        hit = nearest(ctx["cells"].get(k) or {}, acq)
        if hit is None:
            continue
        day, dt, (kind, v) = hit
        present += is_present(kind, v)
        basis.append(f"{k}|{day}|{dt:+d}|{kind}|{v if kind == 'owner' else f'{v:.3f}'}")
    row = {"opt_fill_version": VERSION, "opt_fill_sections": len(secs), "opt_fill_scored": len(basis),
           "opt_fill_present": present, "opt_fill_basis": ";".join(basis)}
    ctx["rows"].append({"acquired_utc": acq, **{k: row[k] for k in COLUMNS[1:4]}})
    return row


def write_public(out: Path, ctx_by_site: dict) -> dict:
    """docs/data/sat/opt_fill.json: the rule, the caveats, the sources and per site the scenes and the sums of the trial (left
    alone when its content would not change; no time stamp). No file is made when no site has a row (Zhanjiang alone)."""
    counts = {}
    for sid, ctx in sorted(ctx_by_site.items()):
        rows = sorted(ctx["rows"], key=lambda r: r["acquired_utc"])
        if not rows:
            continue
        counts[sid] = {"scenes": len(rows), "scenes_filled": sum(1 for r in rows if r["opt_fill_present"]),
                       "present_sum": sum(r["opt_fill_present"] for r in rows), "latest": rows[-1]}
    p = Path(out) / OPT_FILL
    if not counts and not p.exists():
        return counts
    doc = {"version": VERSION, "status": "trial", "note": NOTE, "rule": RULE, "threshold": THRESHOLD, "window_d": WINDOW_D,
           "excluded_sites": list(EXCLUDED), "columns": f"quay_blind/<港>.csv の末尾の列 {', '.join(COLUMNS)}（sat/opt_fill.py の docstring）",
           "caveats": CAVEATS, "policy": POLICY, "sources": SOURCES, "credit": CREDIT,
           "null": "列が空：その日の系列が作れなかった（失敗。本体の数はいつもどおり）、または対象外の港",
           "counts": counts}
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.write_text(text, encoding="utf-8")
    return counts
