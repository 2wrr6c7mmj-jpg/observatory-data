"""Trial sections of the confirmation queue (confirm-1, design §6.5 "sections are added later"): quay lines found by the pier
search (2部 調査報告「桟橋と見落とし」§7.4, 2026-11-20) that are put before the owner, but are NOT counted anywhere else.

    sat/confirm_trial.json   "sources": the lines as the report gives them (candidate, port, OSM way or none, the line, the
                             sides of it that face the water, the role, the basis); "sections": the sections made from them
                             (build()), which is what the ledger, the pres-3 patches and the queue use

What a trial section is, and is not:
  - it enters the section ledger of the port (confirm/sections/<site>.json) as a new sections_version (sat.confirm
    sync_sections: existing ids, labels and states are never touched). It has no earlier answer of the owner, so rule v1
    never settles it automatically at first: every cell of it goes to the owner (few_answers; and no_reference until the
    section has 6 earlier clear days in the patch store)
  - its patches are cut from the daily Sentinel-2 window of the port (Earth Search / AWS COG, PU 0). When the line lies
    outside the port's window, the window is widened (fetch_bbox) and the cells of the existing sections are still cut from
    the original window inside it (the pres-3 package: fetch.one ``base``, build.day_cells ``trial``), so their patches are
    the same bytes as before; the water colour and the clear-day flag of a day are made from the existing sections only
    (the patch store of the daily judge: Store.put ``trial``), so the scores of the existing sections do not move
  - it is not in the published counts (quay_naval50, quay_est2, eval/presence.json: those read sat/berths.json,
    sat/berth_class and the review sets, none of which this file touches), not in the re-fit of the judge
    (its training cells leave them out), and Zhanjiang has none

    python -m sat.confirm_trial build     # write "sections" from "sources" (the test checks that the file is up to date)
    python -m sat.confirm_trial check

To draw a line again: give it a new id (e.g. the candidate with a version, "XSN-W-L2") and retire the old id in the ledger
(python -m sat.confirm sections --site <site> --retire <id>). A trial section whose line differs from the ledger under the same
id is left out by the daily run (not synced, not judged, not stored, with a warning): the old id's patches of the old line are
not mixed with the new line in its reference. Pure Python, no network.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path



def m_per_deg(lat: float) -> tuple[float, float]:
    """Metres per degree of latitude and of longitude at ``lat``: the same series as sat.grid.m_per_deg (that module imports
    numpy; this one stays pure Python, the collect workflows run its tests without numpy)."""
    p = math.radians(lat)
    return (111132.954 - 559.822 * math.cos(2 * p) + 1.175 * math.cos(4 * p), 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p))

PATH = Path(__file__).resolve().parent / "confirm_trial.json"
VERSION = "confirm-trial-1"
FRAME_M = 120.0          # as the review sets and sat.confirm (the frame: the line and its copy 120 m off the water side)
PIECE_MAX_M = 170.0      # a line is cut into ceil(length / 170) equal pieces (the review sets' pieces are 70-170 m)
STEP_M = 10.0            # the points of a piece about every 10 m (as the review sets' lines)
SECTOR = "試行"           # the sector of every trial section: its label is "区画 試行・桟橋 n・左側 k", numbered apart from the
                         # existing sections (sat.confirm.section_labels numbers per sector), so no existing label changes
SIDES = {"L": 1, "R": -1}   # the side as in the ids of the review sets (labels.normals of the pres-3 package: n = (t_y, -t_x) x side)
MARGIN_DEG = 0.001       # fetch_bbox: the trial frames' extent plus this (the window reader adds its own 0.004)


def load(path: Path = PATH) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"version": VERSION, "sources": [], "sections": {}}


def _xy(pts, lat0):
    mlat, mlon = m_per_deg(lat0)
    return [(p[0] * mlon, p[1] * mlat) for p in pts], mlat, mlon


def _length(pts) -> float:
    lat0 = sum(p[1] for p in pts) / len(pts)
    xy, _, _ = _xy(pts, lat0)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:]))


def _along(pts, s0: float, s1: float, step: float) -> list:
    """The points of the polyline from s0 to s1 metres along it, about every ``step`` metres (both ends included)."""
    lat0 = sum(p[1] for p in pts) / len(pts)
    xy, mlat, mlon = _xy(pts, lat0)
    cum = [0.0]
    for a, b in zip(xy, xy[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    n = max(1, int(math.ceil((s1 - s0) / step - 1e-9)))
    out = []
    for k in range(n + 1):
        s = s0 + (s1 - s0) * k / n
        j = max(0, min(len(cum) - 2, next((i for i in range(len(cum) - 1) if cum[i + 1] >= s), len(cum) - 2)))
        f = 0.0 if cum[j + 1] == cum[j] else (s - cum[j]) / (cum[j + 1] - cum[j])
        x = xy[j][0] + f * (xy[j + 1][0] - xy[j][0]); y = xy[j][1] + f * (xy[j + 1][1] - xy[j][1])
        out.append([round(x / mlon, 6), round(y / mlat, 6)])
    return out


def frame(line: list, side: int, frame_m: float = FRAME_M) -> list:
    """The line and its copy frame_m off the water side, reversed (the form of the review sets' frames): the normal of each
    point from its neighbours, n = (t_y, -t_x) x side, as labels.normals of the pres-3 package."""
    lat0 = sum(p[1] for p in line) / len(line)
    xy, mlat, mlon = _xy(line, lat0)
    far = []
    for i, (x, y) in enumerate(xy):
        a, b = xy[max(i - 1, 0)], xy[min(i + 1, len(xy) - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        n = math.hypot(tx, ty) or 1.0
        far.append([round((x + frame_m * side * ty / n) / mlon, 6), round((y - frame_m * side * tx / n) / mlat, 6)])
    return [list(p) for p in line] + far[::-1]


def build(doc: dict) -> dict:
    """site -> the trial sections made from doc["sources"]: each line cut into equal pieces of at most PIECE_MAX_M, one section
    per piece and per side given. Ids: "<OSM way>:<L|R>:<piece, 2 digits>" (the form of the review sets), or, for a line that
    is not an OSM way, "<candidate>:<L|R>:<piece>"."""
    out = {}
    for src in doc.get("sources", []):
        line = src["line"]
        length = _length(line)
        n = max(1, int(math.ceil(length / PIECE_MAX_M - 1e-9)))
        stem = str(src["way"]) if src.get("way") else src["candidate"]
        for side_name in src["sides"]:
            for k in range(n):
                piece = _along(line, length * k / n, length * (k + 1) / n, STEP_M)
                out.setdefault(src["site_id"], []).append({
                    "id": f"{stem}:{side_name}:{k:02d}", "way": src.get("way"), "kind": src.get("kind", "pier"), "side": SIDES[side_name],
                    "sector": SECTOR, "length_m": int(round(length / n)), "line": piece, "frame": frame(piece, SIDES[side_name]),
                    "trial": True, "counted": False, "candidate": src["candidate"], "role": src["role"], "added": src["added"]})
    return out


def sections(site_id: str, doc: dict | None = None) -> list[dict]:
    """The trial sections of a port, in the form of sat.confirm.review_sections (plus trial, counted, candidate, role, added)."""
    return list((doc or load()).get("sections", {}).get(site_id, []))


def trial_ids(site_id: str, doc: dict | None = None) -> frozenset:
    return frozenset(s["id"] for s in sections(site_id, doc))


def extent(site_id: str, doc: dict | None = None) -> list | None:
    """[west, south, east, north] of the frames of the port's trial sections, or None."""
    pts = [p for s in sections(site_id, doc) for p in s["frame"]]
    if not pts:
        return None
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def fetch_bbox(site_id: str, bbox: list, doc: dict | None = None) -> list:
    """The bbox the port's Sentinel-2 window is read for: the site's bbox, widened to the trial frames (plus MARGIN_DEG) when a
    trial section lies outside it. The site's bbox itself when there is none (Zhanjiang and the ports without trial sections:
    nothing changes for them)."""
    e = extent(site_id, doc)
    bb = [float(x) for x in bbox]
    if e is None:
        return bb
    w = [round(e[0] - MARGIN_DEG, 6), round(e[1] - MARGIN_DEG, 6), round(e[2] + MARGIN_DEG, 6), round(e[3] + MARGIN_DEG, 6)]
    return [min(bb[0], w[0]), min(bb[1], w[1]), max(bb[2], w[2]), max(bb[3], w[3])]


LEDGER_DIR = Path(__file__).resolve().parent.parent / "docs" / "data" / "sat" / "confirm" / "sections"


def check(doc: dict | None = None, review_ids: dict | None = None, ledger_dir: Path = LEDGER_DIR) -> list[str]:
    """Problems of the file: sections not up to date with the sources, an id twice, an id of the port's review set, Zhanjiang,
    and a section whose id is in the published ledger (confirm/sections/<site>.json) with another line or frame: a line drawn
    again needs a new id (e.g. the candidate with a version, "XSN-W-L2"); the old id stays in the ledger (retire it with
    python -m sat.confirm sections --site <site> --retire <id>)."""
    doc = doc or load()
    bad = []
    for site_id, secs in doc.get("sections", {}).items():
        p = Path(ledger_dir) / f"{site_id}.json"
        known = {s["id"]: s for s in json.loads(p.read_text(encoding="utf-8")).get("sections", [])} if p.exists() else {}
        for s in secs:
            k = known.get(s["id"])
            if k is not None and (k.get("line") != s["line"] or k.get("frame") != s["frame"]):
                bad.append(f"{site_id} {s['id']}: the line differs from the ledger under the same id. A line drawn again needs a new id "
                           "(the candidate with a version); retire the old id (python -m sat.confirm sections --site <site> --retire <id>). "
                           "Until then the daily run neither judges nor stores that section (its reference would mix the two lines)")
    if doc.get("sections", {}) != build(doc):
        bad.append("sections differ from build(sources): run python -m sat.confirm_trial build")
    for site_id, secs in doc.get("sections", {}).items():
        ids = [s["id"] for s in secs]
        if len(ids) != len(set(ids)):
            bad.append(f"{site_id}: an id twice")
        if site_id == "ZJG":
            bad.append("ZJG: no trial section in Zhanjiang (its outputs do not change)")
        clash = set(ids) & set((review_ids or {}).get(site_id, ()))
        if clash:
            bad.append(f"{site_id}: ids of the review set: {sorted(clash)}")
        if not all(s.get("trial") is True and s.get("counted") is False for s in secs):
            bad.append(f"{site_id}: every trial section has trial true and counted false")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "check"))
    a = ap.parse_args(argv)
    doc = load()
    if a.cmd == "build":
        doc["sections"] = build(doc)
        PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: len(v) for k, v in doc["sections"].items()}))
        return 0
    bad = check(doc)
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
