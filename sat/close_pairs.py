"""Close pairs: a Sentinel-1 scene and a clear Sentinel-2 image of the same port taken within CLOSE_PAIR_H hours.

User's request (2026-11-15): "record the pairs taken close in time every day, and keep them". The big bases count more warships
than are seen in the optical image (Guzhenkou 32 against 14-18), and the number a stretch stands for (ships end to end x the
expected number abreast) can only be calibrated against pairs close in time, of which only two existed (the AI's answers,
labels/ai-port-counts.csv). This module keeps every such pair as the daily run finds it, so the answers can be made later.

- close_pairs.csv: one row per pair (site_id, s1_utc, s2_utc). Rows are never removed: a pair whose S1 scene has left the
  stored est-2 scenes keeps its last numbers. While the scene is stored, its numbers follow the rules of the working tree.
- close_pairs/<site>.json: the counted stretches of each paired S1 scene (box_lonlat, ships, length_m), kept the same way, so the
  boxes outlive quay_naval50/<site>.json (which keeps the newest 10 scenes only).
- The answer columns come from labels/ai-port-counts.csv (the AI's eye count on the S2 image) when the pair was answered.
- The S2 images are the clear ones of s2_scenes.csv and the days of the eye check (labels/<site>/review.json).
- ``source``: daily (an S1 scene of the daily run) or replay (an older S1 scene counted again by sat/replay.py, user
  2026-11-15 "can't we do the same from the past?"; its numbers are kept as recorded, the daily run never recounts it).

PU 0: everything is read from the stored files. Called by eval_presence.write_naval50 (the daily refresh).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

CLOSE_PAIR_H = 24.0            # S2 ~03 UTC, S1 ~10 UTC: the same day is ~7 h apart, the next morning ~17 h
PAIRS_FILE = "close_pairs.csv"
PAIRS_DIR = "close_pairs"
PAIRS_VERSION = "pairs-1"
COLUMNS = ("site_id", "source", "s1_utc", "s2_utc", "dt_h", "stretches_50", "warship_50", "warship_blind", "card", "warship_est2",
           "ships_all", "min_length_m", "s2_image", "recorded_utc", "ans_min", "ans_max", "ans_by")
ANSWER_FILES = (("labels/ai-port-counts.csv", "ai"),)


def _t(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clear_s2(out: Path) -> dict:
    """{site_id: [(acquired_utc, image), ...]} of the clear S2 scenes that have an image, and of the days of the eye
    check (labels/<site>/review.json, image under labels/<site>/). One image per acquisition (s2_scenes first)."""
    res: dict = {}
    for r in _read(out / "s2_scenes.csv"):
        if r.get("result") == "clear" and r.get("image") and r.get("acquired_utc"):
            res.setdefault(r["site_id"], {}).setdefault(r["acquired_utc"], r["image"])
    for p in sorted((out / "labels").glob("*/review.json")):
        sid = p.parent.name
        try:
            days = json.loads(p.read_text(encoding="utf-8")).get("days") or []
        except (OSError, ValueError):
            continue
        for d in days:
            if d.get("acquired_utc") and d.get("image"):
                res.setdefault(sid, {}).setdefault(d["acquired_utc"], f"labels/{sid}/{d['image']}")
    return {sid: sorted(v.items()) for sid, v in res.items()}


def blind_numbers(out: Path, site_id: str) -> dict:
    """{acquired_utc: warship_blind} of quay_blind/<site>.csv (none: +0, as the map reads it)."""
    return {r["acquired_utc"]: int(r.get("warship_blind") or 0) for r in _read(out / "quay_blind" / f"{site_id}.csv")}


def answers(out: Path) -> dict:
    """{(site_id, s2_utc): (min, max, by)} from the answer files (the first file that has the pair wins)."""
    res: dict = {}
    for name, by in ANSWER_FILES:
        for r in _read(out / name):
            key = (r.get("site_id"), r.get("s2_utc"))
            if key not in res and r.get("ai_min") not in (None, ""):
                res[key] = (r["ai_min"], r["ai_max"], r.get("verified_by") or by)
    return res


def find_pairs(rows: list[dict], s2: dict, max_h: float = CLOSE_PAIR_H) -> list[tuple[dict, str, str, float]]:
    """For every clear S2 image, the stored S1 scene of the same site nearest in time if within ``max_h`` hours:
    [(naval50 row, s2_utc, s2_image, dt_h = S1 - S2), ...]. One S1 scene can pair with two S2 images (the same morning and the next)."""
    by_site: dict = {}
    for r in rows:
        by_site.setdefault(r["site_id"], []).append(r)
    res = []
    for sid, images in sorted(s2.items()):
        scenes = by_site.get(sid)
        if not scenes:
            continue
        for acq2, image in sorted(images):
            best = min(scenes, key=lambda r: abs(_t(r["acquired_utc"]) - _t(acq2)))
            dt = (_t(best["acquired_utc"]) - _t(acq2)) / 3600
            if abs(dt) <= max_h:
                res.append((best, acq2, image, round(dt, 1)))
    return res


def update(out: Path, rows: list[dict], boxes: dict, now: datetime | None = None, source: str = "daily",
           numbers_out: Path | None = None) -> int:
    """Add the new pairs and refresh the numbers of the stored ones (close_pairs.csv and close_pairs/<site>.json).
    ``rows`` and ``boxes`` are those of eval_presence.naval50_counts. ``numbers_out``: where quay_blind/ of those rows is
    (sat/replay.py's work directory; default ``out``). Returns the number of pairs kept. No file when none."""
    now = now or datetime.now(timezone.utc)
    numbers_out = numbers_out or out
    path = out / PAIRS_FILE
    kept = {(r["site_id"], r["s1_utc"], r["s2_utc"]): r for r in _read(path)}
    blind = {}
    ans = answers(out)
    for row, acq2, image, dt in find_pairs(rows, clear_s2(out)):
        sid, acq1 = row["site_id"], row["acquired_utc"]
        if source == "replay" and any(k[0] == sid and k[2] == acq2 and k[1] != acq1 for k in kept):
            continue   # the S2 image has its pair already (one S1 scene per S2 image)
        if sid not in blind:
            blind[sid] = blind_numbers(numbers_out, sid)
        b = blind[sid].get(acq1, 0)
        old = kept.get((sid, acq1, acq2)) or {}
        kept[(sid, acq1, acq2)] = {
            "site_id": sid, "source": old.get("source") or source, "s1_utc": acq1, "s2_utc": acq2, "dt_h": f"{dt:g}", "stretches_50": row["stretches_50"],
            "warship_50": row["warship_50"], "warship_blind": b, "card": int(row["warship_50"]) + b,
            "warship_est2": row.get("warship_est2"), "ships_all": row.get("ships_all"), "min_length_m": row.get("min_length_m"),
            "s2_image": image, "recorded_utc": old.get("recorded_utc") or now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ans_min": "", "ans_max": "", "ans_by": ""}
    for key, r in kept.items():
        r["source"] = r.get("source") or "daily"
        a = ans.get((r["site_id"], r["s2_utc"]))
        r["ans_min"], r["ans_max"], r["ans_by"] = a if a else ("", "", "")
    if not kept:
        return 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(kept.values(), key=lambda r: (r["site_id"], r["s1_utc"], r["s2_utc"])))
    d = out / PAIRS_DIR
    d.mkdir(parents=True, exist_ok=True)
    for sid in sorted({k[0] for k in kept}):
        p = d / f"{sid}.json"
        doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        scenes = doc.get("scenes") or {}
        for (s, acq1, _), r in kept.items():
            if s == sid and acq1 in (boxes.get(sid) or {}):
                scenes[acq1] = {"warship_50": int(r["warship_50"]), "boxes": boxes[sid][acq1]}
        doc = {"version": PAIRS_VERSION, "site_id": sid,
               "note": "撮影時刻の近い S2 と組になった S1 の撮影の、数えた片（quay_naval50/<港>.json と同じ形）。"
                       "組の撮影が est-2 の保存から外れても消さない",
               "scenes": dict(sorted(scenes.items()))}
        p.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return len(kept)
