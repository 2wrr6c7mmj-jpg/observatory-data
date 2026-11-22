"""bl-cap (trial, a parallel series): the quay count again on the "resident" sections, with a ceiling on the empty-quay baseline.

Design: 2部 設計書_レーダーの見落とし（miss-1）§4.2・§5.1・§7 PR-3. Where a warship lies at a berth in most baseline scenes, the
baseline's p10 (sat/baseline.py, "the empty quay") holds the ship, so a ship there gives no change and is not counted. bl-cap lowers
the baseline of every line side to the 25 % point of its inner band + MARGIN_DB (sat/quay.cap_baseline) and counts the scene again,
but uses that count only on the sections the owner's earlier answers call resident:

* resident on a scene = the section's owner answers dated strictly before the scene's day (UTC) are at least MIN_ANSWERS and at
  least RESIDENT_MIN of them are "present" (the line of blind-1, sat/blind_prior.py). An answer of the scene's own day or later is
  never read (no future answer in the mark). The answers: labels/<site>-s2-answers.csv and -oos-answers.csv (Zhanjiang: the
  verified rows of ZJG-s2-labels.csv) and labels/<site>-confirm-answers.csv of the confirmation queue (confirm-1; the later file
  wins on the same section and day, and its "cannot tell" withdraws the answer), only rows whose verified_by is "owner" where the
  file has that column. The S1 check (labels/<site>-s1-answers.csv) is not read: the design measured the S2 day's answers.
* the judge on a section = the rule of the daily count (naval50 / est-2's ship length): the longest stretch of the section's way
  and side that belongs to it (sat/s2_labels.stretch_belongs, assign-2), ship length = (length - 20 m - 25 m x (pieces - 1)) /
  pieces, at least 20 m; present when the ship length is MIN_LENGTH_M or more.

MARGIN_DB is fixed at +3 dB, as the design chose it on the 633 cells; it is not chosen again here (choosing it again on the same
cells adds to the optimism). Nothing published changes: quay_counts.csv, quay_est2*, quay_naval50*, eval/presence.json and
Zhanjiang's outputs are not touched. The daily run (sat/run.py) writes quay_blcap/<site>.json for the ports whose sites.json
``quay.blcap`` is "trial" (default "off"; Zhanjiang off), from the scene it already holds: no request and no PU of its own.

    python -m sat.eval_blcap --raw <dir> [<dir> ...]     # the evaluation on the stored chips (design §5.1, PU 0)
"""
from __future__ import annotations

import csv
from pathlib import Path

from .blind_prior import MIN_ANSWERS, RESIDENT_MIN

VERSION = "blcap-1"
MODES = ("off", "trial")
MARGIN_DB = 3.0          # fixed (design §5.1); never tuned again on the evaluation's cells
QUANTILE = 25.0          # the ceiling: this percentile of the inner band's p10 of the line side, + MARGIN_DB
MIN_LENGTH_M = 50.0      # a ship of 50 m or more (the target of naval50 and of the presence evaluation)
TRIM_M = 20.0            # est-2's ship length: the 3x3 mean widens a stretch by about 20 m
SPLIT_M = 25.0           # and every further piece by 25 m
SHIP_MIN_M = 20.0
REVIEW_FILES = ("{site}-s2-answers.csv", "{site}-oos-answers.csv")
ZJG_FILE = "ZJG-s2-labels.csv"
CONFIRM_FILE = "{site}-confirm-answers.csv"


def mode(site: dict) -> str:
    """sites.json quay.blcap: off (default) / trial. Anything else is an error (a typo must not switch a port on)."""
    m = (site.get("quay") or {}).get("blcap") or "off"
    if m not in MODES:
        raise ValueError(f"{site.get('id')}: quay.blcap は {MODES} のどれか（{m!r}）")
    return m


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def owner_answers(out: Path, site_id: str) -> dict[str, list[tuple[str, int]]]:
    """{section_id: [(YYYY-MM-DD, 1 present / 0 absent), ...]} (sorted by day) of the owner's answers: the review sets and the
    confirmation queue's answers (see the module's docstring). "Cannot judge" is no answer."""
    lab = Path(out) / "labels"
    seen: dict[tuple[str, str], int] = {}
    review = [lab / ZJG_FILE] if site_id == "ZJG" else [lab / n.format(site=site_id) for n in REVIEW_FILES]
    for path, confirm in [(p, False) for p in review] + [(lab / CONFIRM_FILE.format(site=site_id), True)]:
        for r in _rows(path):
            if (r.get("site_id") or site_id) != site_id or not r.get("section_id") or not r.get("date"):
                continue
            if "verified_by" in r and (r.get("verified_by") or "").strip() != "owner":
                continue
            key = (r["section_id"], r["date"][:10])
            v, occ = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip()
            if v == "1" and occ in ("0", "1"):
                seen[key] = int(occ)
            elif confirm and v == "0":
                seen.pop(key, None)      # a later "cannot tell" of the confirmation queue withdraws the answer
    out_: dict[str, list[tuple[str, int]]] = {}
    for (sec, day), y in sorted(seen.items()):
        out_.setdefault(sec, []).append((day, y))
    return out_


def resident(ans: dict, section_id: str, scene_utc: str, exclude=()) -> dict:
    """The mark of one section on one scene from the answers strictly before the scene's day:
    {"resident", "answers", "present", "last_answer"}. ``exclude``: days left out besides (the evaluation leaves out the
    answer that is the cell's own truth: an S2 day up to 4 days before its scene)."""
    day = scene_utc[:10]
    ys = [(d, y) for d, y in ans.get(section_id, []) if d < day and d not in exclude]
    n, k = len(ys), sum(y for _, y in ys)
    return {"resident": bool(n >= MIN_ANSWERS and k / n >= RESIDENT_MIN), "answers": n, "present": k,
            "last_answer": ys[-1][0] if ys else None}


def resident_sections(ans: dict, scene_utc: str) -> dict[str, dict]:
    """{section_id: mark} of the sections resident on the scene."""
    out = {}
    for sec in ans:
        m = resident(ans, sec, scene_utc)
        if m["resident"]:
            out[sec] = m
    return out


def ship_length(length_m: float, pieces: int) -> float:
    """est-2's ship length of a stretch (0 when there is none)."""
    if not length_m:
        return 0.0
    k = max(1, int(pieces or 1))
    return max(SHIP_MIN_M, (float(length_m) - TRIM_M - SPLIT_M * (k - 1)) / k)


def section_stretches(secs: list[dict], q: dict, step: float, n_of) -> dict[str, dict]:
    """Per section: the longest stretch of ``q`` (sat/quay.count, restrict) of the same way and side that belongs to it
    (assign-2), with its pieces, the ship length and the judge. ``n_of(profile index)`` = the profile's number of samples."""
    from .s2_labels import stretch_belongs
    by: dict = {}
    for sg in q["segments"]:
        by.setdefault((sg["way_id"], sg["side"]), []).append(sg)
    out = {}
    for sc in secs:
        n = n_of(sc["profile"])
        best = None
        for sg in by.get((sc["way_id"], sc["side"]), []):
            a, b = sg["idx"]
            if stretch_belongs(int(a), int(b), n, sc["idx"], step)[0] and (best is None or int(sg["length_m"]) > int(best["length_m"])):
                best = sg
        L = int(best["length_m"]) if best else 0
        k = int(best["min"]) if best else 1
        ship = ship_length(L, k)
        out[sc["section_id"]] = {"stretch_m": L, "pieces": k if best else 0, "ship_m": round(ship, 1),
                                 "present": int(ship >= MIN_LENGTH_M), "stretch": best}
    return out


def scene_entry(geom, secs: list[dict], marks: dict[str, dict], q_now: dict, q_cap: dict, acq: dict, bver: str) -> dict:
    """The record of one scene: per resident section the current judge (the published count's stretch) and bl-cap's, and the
    capped count's stretches (pieces) that belong to the resident sections."""
    step = geom.cfg["step_m"]
    n_of = lambda pi: len(geom.profiles[pi]["ok"])
    rs = [sc for sc in secs if sc["section_id"] in marks]
    now = section_stretches(rs, q_now, step, n_of)
    cap = section_stretches(rs, q_cap, step, n_of)
    sections, pieces, seen = [], [], set()
    for sc in rs:
        sid = sc["section_id"]
        a, b = now[sid], cap[sid]
        sections.append({"section_id": sid, **{k: marks[sid][k] for k in ("answers", "present", "last_answer")},
                         "now": {k: a[k] for k in ("stretch_m", "pieces", "ship_m", "present")},
                         "blcap": {k: b[k] for k in ("stretch_m", "pieces", "ship_m", "present")},
                         "differs": a["present"] != b["present"]})
        sg = b["stretch"]
        if sg is not None and id(sg) not in seen:
            seen.add(id(sg))
            pieces.append({"section_id": sid, "way_id": sg["way_id"], "side": sg["side"], "length_m": sg["length_m"],
                           "min": sg["min"], "ship_m": b["ship_m"], "peak_dvv_db": sg["peak_dvv_db"], "line": sg["line"]})
    return {"acquired_utc": acq["acquired_utc"], "relative_orbit": acq.get("relative_orbit"), "baseline_version": bver,
            "resident_sections": len(rs), "now_present": sum(s["now"]["present"] for s in sections),
            "blcap_present": sum(s["blcap"]["present"] for s in sections), "sections": sections, "pieces": pieces}


def settings() -> dict:
    return {
        "version": VERSION, "status": "試行（並べて出すだけ。公開の隻数・評価票・湛江の出力は変えない）",
        "design": "2部 設計書_レーダーの見落とし（miss-1）§4.2・§5.1・§7 PR-3",
        "cap": (f"線の片側ごとに、内側の帯（0〜30 m）の基準 p10 の {QUANTILE:g}% 点 + {MARGIN_DB:g} dB を上限に、帯と奥行き（0〜120 m）の画素の基準を下げて"
                f"数え直す（VV・VH 別。上限の幅 +{MARGIN_DB:g} dB は設計書のまま固定、ここで選び直さない）"),
        "resident": (f"その撮影の日（UTC）より前の所有者の答え（labels/<港>-s2-answers.csv・-oos-answers.csv、湛江は ZJG-s2-labels.csv、"
                     f"確認回しの labels/<港>-confirm-answers.csv）が {MIN_ANSWERS} 件以上で、在の割合が {RESIDENT_MIN:g} 以上の区間だけ。撮影の日以後の答えは使わない"),
        "judge": (f"区間に属する最長の片（assign-2）の船の長さ＝（全長 − {TRIM_M:g} m − {SPLIT_M:g} m ×（山の数 − 1））÷ 山の数（最小 {SHIP_MIN_M:g} m）≥ "
                  f"{MIN_LENGTH_M:g} m で在（毎日の札の規則）。now＝公開の数え方、blcap＝上限をかけた数え方"),
        "sites": "sites.json の quay.blcap：off（既定）／trial（quay_blcap/<港>.json を出すだけ）。湛江は off",
        "pu": "毎日の流れで既に取った撮影から計算するだけ（要求・PU の追加なし）",
        "limits": "効果の大きさは同じ 633 升で上限の幅を選んだので楽観がある（設計書 §5.1）。答えのある港にしか使えない",
    }
