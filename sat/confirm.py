"""confirm-1 (trial): cells the judge is not sure of go to the owner, and the owner's answers come back.

Design: 設計書_確認回し（2部、2026-11-20）§2.5・§4・§6・§7, PR-3. A cell is one quay section x one Sentinel-2 day. This module
takes a table of scores per cell and does the bookkeeping around it. It does not make the scores and does not care where
they come from (pres-3o, or any other judge): it reads the columns below and nothing else. Pure Python (no numpy), no
network, no CDSE (PU 0). It writes only under <out>/confirm/ and <out>/labels/<site>-confirm-answers.csv.

    python -m sat.confirm sections --site QDH                 # the section ledger, from the review set of the port
    python -m sat.confirm run --scores scores.csv             # judge the cells of the table, write state / log / queue / summary
    python -m sat.confirm merge --file confirm-answers-2026-11-27.csv   # the owner's answers (the file the page downloads)
    python -m sat.confirm summary

The score table (CSV, one row per cell; SCORE_FIELDS):
    site_id, section_id, date      the key of the cell (date = the UTC day of the Sentinel-2 scene)
    acquired_utc                   the time of the scene (optional)
    p                              the judge's score in 0..1 (present = high). Empty: the judge had not enough input
                                   (reason no_reference); the cell is shown to the owner
    model_version                  the version of the judge (recorded, never interpreted)
    cloud_shadow | cloud, shadow   the share of the section's pixels under cloud and shadow (SCL); and
    nodata                         the share without data. Over the limits of confirm.json "clear" -> no_image.
    status                         optional: "no_image" says the same directly
    answers_through                optional: the last answer day the judge used for this score (recorded in the log)
    n_answers                      the number of the owner's answers the judge used for this score (an integer; pres-3o:
                                   the answers whose day has a patch in the store). Missing or empty: not known, and the
                                   cell is not settled automatically (reason answers_unknown)
    before_cutoff                  1 (any value but empty) when the day is before the cutoff of the judge's coefficients: the
                                   score is from the judge's training period, not out of sample (§6.4-2). Such a row is not
                                   judged: no state, no log row, not in A; it is listed in "skipped" (reason before_cutoff)

The rule v1 (confirm.json; decide()): automatic only when all of
    p <= p_low or p >= p_high / at least min_answers answers of the owner in the section before the day, counted both
    in the answer files and by the judge (n_answers of the table; the smaller of the two counts) / the last of them at
    most max_prev_days days before the day (from the answer files) / the port is in auto_sites.
Everything else is shown to the owner. Only the owner's answers count as answers: an automatic state never enters the
history (history()), so the judge cannot confirm itself.

Files (under <out>/confirm/):
    sections/<site>.json     the section ledger (id, line, frame, since, sections_version, active / retired)
    state/<site>/YYYY-MM.csv one row per cell: its state now (STATES). No audit mark here
    log/<site>/YYYY-MM.csv   append-only: every judgement of a cell, written before the owner's answer comes in. A cell is
                             scored against its last row from before its answer (summary()). Never rewritten
    queue.json               what the page docs/sat/label/?queue=1 shows. No score, no judgement, no audit mark
    summary.json             the numbers of §7 (A: accuracy of the automatic cells; B: composition; C: total, labelled)
    latest.json              what the trial page (docs/sat/confirm/) and the site show (§6.6): per port and active section the
                             newest cell's state, and the numbers always shown beside it. Written with summary.json. No score
                             and no judgement of a cell the owner has not seen (in the shadow operation no automatic value)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
from datetime import date as _date, datetime, timezone
from pathlib import Path

from . import atomic, berths

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
CONFIG_PATH = Path(__file__).resolve().parent / "confirm.json"
VERSION = "confirm-1"
QUEUE_VERSION = "confirm-queue-1"
LATEST_VERSION = "confirm-latest-1"
DIR = "confirm"
FRAME_M = 120.0
STATES = ("auto", "human", "pending", "unconfirmed", "unsure", "no_image", "excluded")
# excluded (confirm-x1): a cell of a section no warship can lie at (exclude_sections), or whose frame is not in the picture. It
# keeps its score p (opt-fill reads it) and goes on record, but is never in the queue, never expires and is not in A or C
FINAL = ("human", "unsure")                 # the owner has spoken: never judged again
SCORE_FIELDS = ["site_id", "section_id", "date", "acquired_utc", "p", "model_version", "cloud_shadow", "nodata", "status",
                "answers_through", "n_answers", "before_cutoff"]
STATE_FIELDS = ["site_id", "section_id", "date", "acquired_utc", "state", "value", "source", "p", "model_version",
                "rule_version", "reason", "prev_answer_date", "prev_answer", "n_answers", "n_answers_judge", "answered_utc", "sections_version"]
LOG_FIELDS = ["site_id", "section_id", "date", "judged_utc", "model_version", "rule_version", "p", "state", "value", "source",
              "reason", "v1_state", "v1_value", "answers_through", "n_answers", "n_answers_judge", "prev_answer_date", "prev_answer", "audit",
              "audit_rate", "queued", "answered_utc", "queue_version", "sections_version"]
ANSWER_FIELDS = ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_by", "verified_note",
                 "answered_utc", "queue_version"]
# what makes a new log row: a change in any of these
LOG_KEY = ["model_version", "rule_version", "p", "state", "value", "source", "reason", "v1_state", "v1_value", "n_answers",
           "n_answers_judge",
           "prev_answer_date", "prev_answer", "audit", "queued", "answered_utc"]
CREDIT = "Contains modified Copernicus Sentinel data"
ATTRIBUTION = "岸壁の線：(c) OpenStreetMap contributors（ODbL 1.0）"


# ---------------------------------------------------------------- small helpers
def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("rule_version", "p_low", "p_high", "min_answers", "max_prev_days", "auto_sites", "expire_days", "audit", "clear"):
        if k not in cfg:
            raise ValueError(f"confirm.json: {k} is missing")
    if not (0.0 <= cfg["p_low"] < 0.5 < cfg["p_high"] <= 1.0):
        raise ValueError("confirm.json: p_low < 0.5 < p_high")
    return cfg


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(s: str) -> _date:
    return _date.fromisoformat(str(s)[:10])


def _days(a: str, b: str) -> int:
    """b - a in days."""
    return (_day(b) - _day(a)).days


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_text(rows: list[dict], fields: list[str], header: bool = True) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    if header:
        w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return buf.getvalue()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> bool:
    """Atomic; the file is left alone when its content would not change. True when written."""
    text = _csv_text(rows, fields)
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    atomic.write_text(path, text)
    return True


def append_log(path: Path, rows: list[dict]) -> None:
    """The log only grows: rows are added at the end, what is there is never touched."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with path.open(encoding="utf-8", newline="") as f:
            head = f.readline().rstrip("\n")
        if head != ",".join(LOG_FIELDS):
            raise ValueError(f"{path}: the header of the log differs from LOG_FIELDS (the log is never rewritten: start a new file)")
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(_csv_text(rows, LOG_FIELDS, header=new))


def _write_json(path: Path, doc: dict) -> None:
    atomic.write_text(Path(path), json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def _month(date: str) -> str:
    return str(date)[:7]


def _fnum(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt_p(p: float | None) -> str:
    return "" if p is None else repr(float(p))      # as read: the record must show on which side of a threshold it was


# ---------------------------------------------------------------- the section ledger
def sections_path(out: Path, site_id: str) -> Path:
    return Path(out) / DIR / "sections" / f"{site_id}.json"


def load_sections(out: Path, site_id: str) -> dict | None:
    p = sections_path(out, site_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def active_sections(ledger: dict | None) -> dict:
    return {s["id"]: s for s in (ledger or {}).get("sections", []) if s.get("status", "active") == "active"}


def _zjg_frame(line: list, side: int) -> list:
    """Zhanjiang has no review set: the frame is the line and its copy FRAME_M off the water side (the same sense as
    sat/pres3/labels.py normals(): n = (t_y, -t_x) x side)."""
    lat0 = sum(p[1] for p in line) / len(line)
    mlat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat0))
    mlon = 111412.84 * math.cos(math.radians(lat0)) - 93.5 * math.cos(3 * math.radians(lat0))
    xy = [(p[0] * mlon, p[1] * mlat) for p in line]
    far = []
    for i, (x, y) in enumerate(xy):
        a, b = xy[max(i - 1, 0)], xy[min(i + 1, len(xy) - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        n = math.hypot(tx, ty) or 1.0
        far.append([round((x + FRAME_M * side * ty / n) / mlon, 6), round((y - FRAME_M * side * tx / n) / mlat, 6)])
    return [list(p) for p in line] + far[::-1]


def review_sections(out: Path, site_id: str) -> list[dict]:
    """The sections the ledger starts from: the review set of the port (labels/<site>/review.json); Zhanjiang from
    labels/ZJG-s2-sections.geojson."""
    rv = Path(out) / "labels" / site_id / "review.json"
    if rv.exists():
        return [{k: s.get(k) for k in ("id", "way", "kind", "side", "sector", "length_m", "line", "frame")}
                for s in json.loads(rv.read_text(encoding="utf-8"))["sections"]]
    gj = Path(out) / "labels" / f"{site_id}-s2-sections.geojson"
    if gj.exists():
        res = []
        for f in json.loads(gj.read_text(encoding="utf-8"))["features"]:
            pr, line = f["properties"], f["geometry"]["coordinates"]
            res.append({"id": pr["section_id"], "way": pr.get("way_id"), "kind": pr.get("kind"), "side": pr.get("side"),
                        "sector": pr.get("sector"), "length_m": pr.get("length_m"), "line": line,
                        "frame": _zjg_frame(line, int(pr.get("side") or 1))})
        return res
    raise FileNotFoundError(f"{site_id}: no review set (labels/{site_id}/review.json) and no sections geojson")


def sync_sections(out: Path, site_id: str, sections: list[dict], today: str, retire: tuple = ()) -> dict:
    """Add the sections that are not in the ledger yet (a new sections_version); mark ``retire`` as retired. A section
    already in the ledger is never changed: a line drawn again needs a new id (the patches and exemplars of the old id
    do not fit the new line), and this raises when the geometry of a known id differs."""
    led = load_sections(out, site_id) or {"version": VERSION, "site_id": site_id, "sections_version": 0, "sections": []}
    known = {s["id"]: s for s in led["sections"]}
    new = []
    for s in sections:
        k = known.get(s["id"])
        if k is None:
            new.append(s)
        elif k.get("line") != s.get("line") or k.get("frame") != s.get("frame"):
            raise ValueError(f"{site_id} {s['id']}: the line or frame differs from the ledger; a section drawn again needs a new id")
    gone = [i for i in retire if i in known and known[i].get("status", "active") == "active"]
    if not new and not gone:
        return led
    led["sections_version"] = int(led["sections_version"]) + 1
    for s in new:
        led["sections"].append({**s, "since": today[:10], "sections_version": led["sections_version"], "status": "active"})
    for i in gone:
        known[i]["status"] = "retired"
        known[i]["retired"] = today[:10]
    led["updated_utc"] = today
    _write_json(sections_path(out, site_id), led)
    return led


# ---------------------------------------------------------------- the owner's answers
def confirm_answers_path(out: Path, site_id: str) -> Path:
    return Path(out) / "labels" / f"{site_id}-confirm-answers.csv"


def owner_answers(out: Path, site_id: str) -> dict:
    """(section_id, date) -> {"y": 1 / 0 / None (cannot judge), "answered_utc", "queue_version", "from"} for every answer of
    the owner: the review sets (labels/<site>-s2-answers.csv; Zhanjiang: the verified rows of ZJG-s2-labels.csv; read only)
    and the confirm answers (labels/<site>-confirm-answers.csv; the later one wins). Nothing else is an answer."""
    out = Path(out)
    res = {}
    base = out / "labels" / ("ZJG-s2-labels.csv" if site_id == "ZJG" else f"{site_id}-s2-answers.csv")
    for src, path in (("review", base), ("confirm", confirm_answers_path(out, site_id))):
        for r in read_csv(path):
            v, occ = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip()
            if (r.get("verified_by") or "").strip() != "owner" or (r.get("site_id") or site_id) != site_id:
                continue
            if v == "1" and occ in ("0", "1"):
                y = int(occ)
            elif v == "0":
                y = None
            else:
                continue
            res[(r["section_id"], r["date"])] = {"y": y, "answered_utc": (r.get("answered_utc") or "").strip(),
                                                 "queue_version": (r.get("queue_version") or "").strip(), "from": src}
    return res


def history(answers: dict, section_id: str, date: str) -> dict:
    """What the owner said about the section strictly before ``date``. "Cannot judge" is not counted."""
    days = sorted((d, a["y"]) for (s, d), a in answers.items() if s == section_id and d < date[:10] and a["y"] is not None)
    if not days:
        return {"n_answers": 0, "prev_answer_date": None, "prev_answer": None, "prev_days": None}
    return {"n_answers": len(days), "prev_answer_date": days[-1][0], "prev_answer": days[-1][1],
            "prev_days": _days(days[-1][0], date)}


# ---------------------------------------------------------------- confirm-x1: the sections no warship can lie at
# 2026-11-23 (the owner, after 297 cells: "too many cells that are plainly land or not a pier"). Design: 2部「確認回しの
# キューから艦が着き得ない区間を外す規則」§2.2・§3. A section is shown to the owner when it is a counted section (the middle
# point of its line is on a berth line, Berths.point_ok, and the line's naval-1 class there is naval and not "berth": false:
# the sections of eval/presence.json and hires-1's counted_after), or when it is guarded (below). The rule reads the class
# files and quay lines as they are at the time of the run, so a change of berth_class or of the lines decides again.
EXCLUDE_REASONS = {"off_berth_line": "着く線の外", "berth_false": "艦の着かない線（berth false）", "civil": "民用の線",
                   "not_naval": "区分が海軍でない線", "out_of_image": "枠が画像の範囲の外"}
SITES_PATH = ROOT / "sat" / "sites.json"


def _file_hash(path: Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def berth_rule(site_id: str):
    """(Berths of the site with the naval rule, the ids of its quay lines, the basis) or None when the site has no entry in
    sat/sites.json or no quay lines (then nothing is excluded). The basis names the versions of what the rule read."""
    sites = json.loads(SITES_PATH.read_text(encoding="utf-8")).get("sites", []) if SITES_PATH.exists() else []
    site = next((s for s in sites if s.get("id") == site_id), None)
    if site is None:
        return None
    lp = ROOT / "sat" / "masks" / f"{site_id}-quay-v{int(site.get('quay_lines_version', 1))}.json"
    if not lp.exists():
        return None
    lines = json.loads(lp.read_text(encoding="utf-8"))
    b = berths.Berths(site, lines, naval=True)
    basis = {"berth_class": _file_hash(berths.CLASS_DIR / f"{site_id}.json"), "quay_lines": f"{lp.name}:{_file_hash(lp)}",
             "berths": f"{berths.VERSION}:{_file_hash(berths.BERTHS_PATH)}", "naval": berths.NAVAL_VERSION}
    return b, {str(w.get("id")) for w in lines.get("ways", [])}, basis


UNKNOWN, COUNTED = "unknown", "counted"


def not_counted(rule, s: dict) -> str | None:
    """Why the section is not a counted section (EXCLUDE_REASONS), or None when it is one or is not known."""
    c = section_class(rule, s)
    return None if c in (UNKNOWN, COUNTED) else c


def section_class(rule, s: dict) -> str:
    """COUNTED, a reason of EXCLUDE_REASONS, or UNKNOWN (a site without the rule's files, a line the site's quay lines do not
    have, a trial section: shown, no mark). Judged at the middle point of the line, as hires-1 and eval/presence.json."""
    if rule is None or s.get("trial"):
        return UNKNOWN
    b, ways, _ = rule
    line, w = s.get("line") or [], str(s.get("way"))
    if not line or w not in ways:
        return UNKNOWN
    lon, lat = float(line[len(line) // 2][0]), float(line[len(line) // 2][1])
    if b.no_berth_at(w, lon, lat):
        return "berth_false"
    if not b.point_ok(s.get("kind"), w, lon, lat):
        return "off_berth_line"
    c = b.line_class(w, lon, lat)
    return COUNTED if c == "naval" else "civil" if c == "civil" else "not_naval"


def _pip(x: float, y: float, poly) -> bool:
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i][:2], poly[(i + 1) % len(poly)][:2]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _lattice(frame, n: int = 32) -> list:
    """Points on an n x n lattice over the frame's box that are inside the frame: the frame's area for the shares below."""
    xs, ys = [p[0] for p in frame], [p[1] for p in frame]
    w, s, e, nn = min(xs), min(ys), max(xs), max(ys)
    pts = [(w + (e - w) * (i + 0.5) / n, s + (nn - s) * (j + 0.5) / n) for i in range(n) for j in range(n)]
    return [p for p in pts if _pip(p[0], p[1], frame)]


def overlap(a: list, b: list) -> float:
    """The share of the frame ``a`` that the frame ``b`` covers (0..1; a lattice of points, pure Python)."""
    if not a or not b:
        return 0.0
    bx, by = [p[0] for p in b], [p[1] for p in b]
    if max(p[0] for p in a) < min(bx) or min(p[0] for p in a) > max(bx) or max(p[1] for p in a) < min(by) or min(p[1] for p in a) > max(by):
        return 0.0
    pts = _lattice(a)
    return sum(1 for x, y in pts if _pip(x, y, b)) / len(pts) if pts else 0.0


def in_image(frame: list, grid: dict | None) -> float:
    """The share of the frame inside the picture's grid (1.0 when there is no grid to check against)."""
    if not grid or not frame:
        return 1.0
    pts = _lattice(frame)
    if not pts:
        return 1.0
    return sum(1 for x, y in pts if grid["west"] <= x <= grid["east"] and grid["south"] <= y <= grid["north"]) / len(pts)


def guard_counts(secs: dict, why: dict, answers: dict, overlap_min: float, counted: list | None = None) -> dict:
    """§2.2-2: for each section not counted (``why``), the owner's "present" answers that no counted section whose frame covers
    at least ``overlap_min`` of this one's answered "present" on the same day (an answer the neighbours do not explain)."""
    counted = [i for i in secs if why.get(i) is None] if counted is None else counted
    pres = {}
    for (sec, day), a in answers.items():
        if a["y"] == 1:
            pres.setdefault(sec, set()).add(day)
    res = {}
    for sid in why:
        if why[sid] is None or not pres.get(sid):
            continue
        near = [t for t in counted if overlap(secs[sid].get("frame") or [], secs[t].get("frame") or []) >= overlap_min]
        res[sid] = sum(1 for d in pres[sid] if not any(d in pres.get(t, ()) for t in near))
    return res


def exclude_sections(out: Path, site_id: str, ledger: dict | None, answers: dict, cfg: dict, today: str, rule=False) -> dict:
    """confirm-x1 on the active sections of the ledger: {section_id: reason} of the sections not shown to the owner. The ledger
    keeps every section active (sat/opt_fill.py reads the active sections and the p of their cells) and gets a mark per
    section: queue "excluded" with excluded_reason / excluded_rule / excluded_since / excluded_basis, or queue "shown" (with
    guard_reason when a guard keeps it). The line and frame are never touched. cfg "exclude" null: nothing is excluded (marks
    removed). ``rule``: the Berths rule to use (tests); False reads the site's files."""
    x = cfg.get("exclude")
    secs = active_sections(ledger)
    if ledger is None:
        return {}
    rule = berth_rule(site_id) if rule is False else rule
    cls = {i: (section_class(rule, s) if x else UNKNOWN) for i, s in secs.items()}
    why = {i: (None if c in (UNKNOWN, COUNTED) else c) for i, c in cls.items()}
    guard = guard_counts(secs, why, answers, float(x.get("overlap_min", 0.3)), [i for i, c in cls.items() if c == COUNTED]) if x else {}
    res, changed = {}, False
    for i, s in secs.items():
        keep = {k: s.get(k) for k in ("queue", "excluded_reason", "excluded_rule", "excluded_since", "excluded_basis", "guard_reason")}
        for k in keep:
            s.pop(k, None)
        if x:
            g = guard.get(i, 0)
            if why[i] is not None and g < int(x.get("guard_min_present", 2)):
                res[i] = why[i]
                same = keep["queue"] == "excluded" and keep["excluded_reason"] == why[i] and keep["excluded_rule"] == x["rule"]
                s.update(queue="excluded", excluded_reason=why[i], excluded_rule=x["rule"],
                         excluded_since=keep["excluded_since"] if same else today[:10], excluded_basis=rule[2] if rule else None)
            elif cls[i] != UNKNOWN:
                s["queue"] = "shown"
                if why[i] is not None:
                    s["guard_reason"] = f"{why[i]}: owner_present_unexplained={g}"
        changed |= any(s.get(k) != v for k, v in keep.items())
    if changed:
        ledger["updated_utc"] = today
        _write_json(sections_path(out, site_id), ledger)
    return res


def _shown_by(log: list, sec: str | None, kind: str, since: str = "") -> list:
    """The cells a check (valve / sample) put in a queue: log rows queued = 1 with that reason (``sec`` None: any section)."""
    return [(r["section_id"], r["date"]) for r in log
            if r.get("queued") == "1" and (r.get("reason") or "") == kind and (sec is None or r["section_id"] == sec)
            and (r.get("judged_utc") or "") >= since]


def _exclude_cells(site_id: str, state: dict, excluded: dict, cfg: dict, today: str, log: list) -> None:
    """The waiting cells of excluded sections become excluded (reason excluded:<why>); an excluded cell of a section shown
    again waits again (readmitted). Two checks put an excluded cell in the queue once (§2.2-4): valve, the newest of the
    last valve_runs scored days of the section all with p >= valve_p; sample, one excluded section per port and month
    (fixed by port, month and rule version). Their answers are kept, but the section stays excluded (not in A or C)."""
    x = cfg.get("exclude") or {}
    for k, r in state.items():
        if r["state"] == "pending" and k[0] in excluded:
            r.update(state="excluded", reason=f"excluded:{excluded[k[0]]}")
        elif r["state"] == "excluded" and k[0] not in excluded and r["reason"] != "excluded:out_of_image":
            r.update(state="pending", reason="readmitted")
    if not x or not excluded:
        return

    def live(k):
        return k in state and state[k]["state"] in ("excluded", "pending") and _days(k[1], today) <= cfg["expire_days"]

    def show(k, kind):
        state[k].update(state="pending", reason=kind)

    vp, vn = float(x.get("valve_p", 0.9)), int(x.get("valve_runs", 3))
    for sid in sorted(excluded):
        days = sorted((k for k, r in state.items() if k[0] == sid and _fnum(r["p"]) is not None and r["state"] != "no_image"),
                      key=lambda k: k[1])
        streak = []
        for k in reversed(days):
            if _fnum(state[k]["p"]) < vp:
                break
            streak.append(k)
        if len(streak) < vn:
            continue
        before = [k for k in _shown_by(log, sid, "valve") if k[1] >= streak[-1][1]]
        if before:
            if live(before[-1]):
                show(before[-1], "valve")
        elif live(streak[0]):
            show(streak[0], "valve")
    if int(x.get("sample_per_month", 0)) < 1:
        return
    month = today[:7]
    before = _shown_by(log, None, "sample", since=month + "-01")
    if before:
        if live(before[-1]) and before[-1][0] in excluded:
            show(before[-1], "sample")
        return
    pool = sorted(sid for sid in excluded if excluded[sid] != "out_of_image")
    if not pool:
        return
    sid = pool[int(_unit("sample", site_id, month, cfg["rule_version"]) * len(pool))]
    cells = sorted((k for k, r in state.items() if k[0] == sid and r["state"] == "excluded" and r["reason"] != "excluded:out_of_image"
                    and live(k)), key=lambda k: k[1])
    if cells:
        show(cells[-1], "sample")


# ---------------------------------------------------------------- the rule
def is_clear(row: dict, cfg: dict) -> bool:
    if (row.get("status") or "").strip() == "no_image":
        return False
    cs = _fnum(row.get("cloud_shadow"))
    if cs is None and (row.get("cloud") not in (None, "") or row.get("shadow") not in (None, "")):
        cs = (_fnum(row.get("cloud")) or 0.0) + (_fnum(row.get("shadow")) or 0.0)
    nd = _fnum(row.get("nodata"))
    return not ((cs is not None and cs > cfg["clear"]["cloud_shadow_max"]) or (nd is not None and nd > cfg["clear"]["nodata_max"]))


def judge_count(v) -> int | None:
    """The n_answers column of the score table: a whole number >= 0, else None (not known)."""
    s = "" if v is None else str(v).strip()
    return int(s) if s.isascii() and s.isdigit() else None


def decide(site_id: str, p: float | None, clear: bool, hist: dict, cfg: dict, trial: bool = False) -> dict:
    """Rule v1 for one cell: {"state": auto / pending / no_image, "value": 1 / 0 / None, "reason"}. The reasons a cell is
    shown for are joined with "+", in the order of the design (§2.5, §4).

    The number of answers is the smaller of hist["n_answers"] (the answer files) and hist["n_answers_judge"] (the answers
    the judge used for this score). A judge count that is missing or None is not known: the cell is shown
    (answers_unknown), never settled on the count of the files alone. The days since the last answer come from the files.

    trial: a trial section of the ledger (sat/confirm_trial.json): always shown to the owner (reason trial), never
    automatic, whatever its answers and score (an empty pier is watched for the day it is first used)."""
    if not clear:
        return {"state": "no_image", "value": None, "reason": "no_image"}
    why = []
    if p is None:
        why.append("no_reference")
    elif cfg["p_low"] < p < cfg["p_high"]:
        why.append("uncertain")
    nj = hist.get("n_answers_judge")
    if min(hist["n_answers"], nj if nj is not None else hist["n_answers"]) < cfg["min_answers"]:
        why.append("few_answers")
    elif nj is None:
        why.append("answers_unknown")
    elif hist["prev_days"] is None or hist["prev_days"] > cfg["max_prev_days"]:
        why.append("stale_answer")
    if site_id not in cfg["auto_sites"]:
        why.append("site_off")
    if trial:
        why.append("trial")
    if why:
        return {"state": "pending", "value": None, "reason": "+".join(why)}
    return {"state": "auto", "value": 1 if p >= cfg["p_high"] else 0, "reason": "confident"}


def _unit(*parts) -> float:
    """A number in [0, 1) fixed by the parts (the same cell gives the same number on every run and every machine)."""
    h = hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def audit_rate(value: int, prev_days: int | None, cfg: dict) -> float:
    a = cfg["audit"]
    r = float(a["rate"])
    if value == 1:
        r *= float(a.get("present_factor", 1.0))
    if prev_days is not None and prev_days > int(a.get("stale_days", 10 ** 9)):
        r *= float(a.get("stale_factor", 1.0))
    return min(1.0, r)


def audit_pick(site_id: str, section_id: str, date: str, value: int, prev_days: int | None, cfg: dict) -> tuple[bool, float]:
    """§7.3: is this automatic cell one of the sample the owner is asked about? Decided by (site, section, day, rule
    version) only, so a run repeated picks the same cells and a cell cannot be swapped afterwards."""
    r = audit_rate(value, prev_days, cfg)
    return _unit("audit", site_id, section_id, date[:10], cfg["rule_version"]) < r, r


# ---------------------------------------------------------------- state and log files
def state_dir(out: Path, site_id: str) -> Path:
    return Path(out) / DIR / "state" / site_id


def log_dir(out: Path, site_id: str) -> Path:
    return Path(out) / DIR / "log" / site_id


def load_state(out: Path, site_id: str) -> dict:
    res = {}
    d = state_dir(out, site_id)
    for f in sorted(d.glob("*.csv")) if d.exists() else []:
        for r in read_csv(f):
            res[(r["section_id"], r["date"])] = r
    return res


def load_log(out: Path, site_id: str) -> list[dict]:
    d = log_dir(out, site_id)
    return [r for f in (sorted(d.glob("*.csv")) if d.exists() else []) for r in read_csv(f)]


def known_sites(out: Path) -> list[str]:
    d = Path(out) / DIR / "sections"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def _save_state(out: Path, site_id: str, state: dict) -> None:
    by = {}
    for r in state.values():
        by.setdefault(_month(r["date"]), []).append(r)
    for m, rows in by.items():
        write_csv(state_dir(out, site_id) / f"{m}.csv", sorted(rows, key=lambda r: (r["date"], r["section_id"])), STATE_FIELDS)


def _log_changes(out: Path, site_id: str, state: dict, extra: dict, now: str, queue_version: str = "") -> int:
    """Append a log row for every cell whose judgement differs from its last log row."""
    last = {}
    for r in load_log(out, site_id):
        last[(r["section_id"], r["date"])] = r
    add = {}
    for k, r in state.items():
        row = {**r, **extra.get(k, {})}
        row.setdefault("audit", "0")
        row.setdefault("queued", "0")
        row = {f: ("" if row.get(f) is None else str(row.get(f))) for f in LOG_FIELDS}
        prev = last.get(k)
        if prev is not None and "v1_state" not in extra.get(k, {}):
            # a cell not judged in this run keeps what the rule said last time
            for f in ("v1_state", "v1_value", "answers_through", "audit_rate"):
                row[f] = prev.get(f, "")
        if prev is not None and all((prev.get(f) or "") == row[f] for f in LOG_KEY):
            continue
        row["judged_utc"] = now
        row["queue_version"] = row["queue_version"] or (queue_version if row["queued"] == "1" else "")
        add.setdefault(_month(r["date"]), []).append(row)
    for m, rows in add.items():
        append_log(log_dir(out, site_id) / f"{m}.csv", sorted(rows, key=lambda r: (r["date"], r["section_id"])))
    return sum(len(v) for v in add.values())


# ---------------------------------------------------------------- the run
def rejudge_scope(state: dict, answered: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """§6.3 (2): the cells to judge again after answers came in: the same section, days after the answered day, not yet
    answered by the owner. Days before the answer are never judged again (a later answer does not reach back)."""
    first = {}
    for sec, day in answered:
        first[sec] = min(day, first.get(sec, day))
    return sorted(k for k, r in state.items() if k[0] in first and k[1] > first[k[0]] and r["state"] not in FINAL + ("no_image",))


def _apply_answers(state: dict, answers: dict) -> list[tuple[str, str]]:
    """A cell the owner answered takes the answer as its state. Returns the cells that changed."""
    ch = []
    for k, r in state.items():
        a = answers.get(k)
        if a is None:
            continue
        st, val = ("unsure", "") if a["y"] is None else ("human", str(a["y"]))
        if (r["state"], r["value"], r.get("answered_utc") or "") != (st, val, a["answered_utc"]):
            r.update(state=st, value=val, source="owner", answered_utc=a["answered_utc"],
                     reason="owner" if r["state"] not in FINAL else r["reason"])
            ch.append(k)
    return ch


def _queue_rules(site_id: str, state: dict, cfg: dict, today: str) -> set:
    """§3.3: a waiting cell older than expire_days is unconfirmed (expired); of the waiting cells of a section only the
    newest stays, and none older than a cell the owner has answered (the others: unconfirmed, superseded; §3.3-2: an
    answer to the newest cell does not bring back the cells it replaced, so a queue left for weeks does not double).
    Returns the keys of the cells in the queue: the waiting ones and the automatic ones picked for the audit (not
    answered, not expired; exempt from "newest only" so that the sample stays a random one)."""
    q = set()
    newest = {}
    for k, r in state.items():
        if r["state"] in FINAL and cfg.get("latest_only", True):
            newest[k[0]] = max(k[1], newest.get(k[0], k[1]))
        if r["state"] != "pending":
            continue
        if _days(r["date"], today) > cfg["expire_days"]:
            r.update(state="unconfirmed", reason="expired")
        elif cfg.get("latest_only", True):
            newest[k[0]] = max(k[1], newest.get(k[0], k[1]))
    for k, r in state.items():
        if r["state"] == "pending":
            if cfg.get("latest_only", True) and k[1] < newest[k[0]]:
                r.update(state="unconfirmed", reason="superseded")
            else:
                q.add(k)
        elif r["state"] == "auto" and cfg.get("auto_display") and _days(r["date"], today) <= cfg["expire_days"]:
            pd = _days(r["prev_answer_date"], r["date"]) if r.get("prev_answer_date") else None
            if audit_pick(site_id, k[0], k[1], int(r["value"]), pd, cfg)[0]:
                q.add(k)
    return q


def run(scores: list[dict], out: Path = OUT, cfg: dict | None = None, now: str | None = None, hold_without_image: bool = False) -> dict:
    """Judge the cells of the score table and bring state, log, queue and summary up to date. A cell the owner answered is
    never judged again; a cell outside the table keeps its judgement (only the queue rules and new answers reach it).
    hold_without_image (the daily run): a cell whose day has no picture yet is held back from the queue (queued = 0 in the
    log, so merge takes no answer for it): the page would show it with nothing to answer from. It keeps its state and
    waits (the expiry of §3.3 still counts); queue.json lists the held days ("held": port, day, cells, the days of their
    last answers) so that the pictures can be made, and the next run puts the cell in once its picture is there."""
    out, cfg, now = Path(out), cfg or load_config(), now or utcnow()
    today = now[:10]
    by_site = {}
    for r in scores:
        by_site.setdefault((r.get("site_id") or "").strip(), []).append(r)
    res = {"now": now, "sites": {}, "skipped": []}
    queued = {}
    held = {}
    pending_log = {}
    for site_id in sorted(set(known_sites(out)) | set(by_site)):
        ledger = load_sections(out, site_id)
        secs = active_sections(ledger)
        if ledger is None:
            res["skipped"] += [f"{site_id}:{r.get('section_id')}@{r.get('date')} (no ledger)" for r in by_site.get(site_id, [])]
            continue
        state = load_state(out, site_id)
        answers = owner_answers(out, site_id)
        excluded = exclude_sections(out, site_id, ledger, answers, cfg, now)      # confirm-x1 (marks the ledger)
        extra = {}
        for r in by_site.get(site_id, []):
            sec, day = (r.get("section_id") or "").strip(), (r.get("date") or "").strip()[:10]
            k = (sec, day)
            if sec not in secs:
                res["skipped"].append(f"{site_id}:{sec}@{day} (not an active section of the ledger)")
                continue
            try:
                _day(day)
            except ValueError:
                res["skipped"].append(f"{site_id}:{sec}@{day} (date)")
                continue
            if (r.get("before_cutoff") or "").strip():
                # §6.4-2: a version judges only the days after its cutoff. A score of the training period is not a judgement
                # to record (it would be scored against answers the judge learnt from)
                res["skipped"].append(f"{site_id}:{sec}@{day} (before_cutoff: a score of the judge's training period)")
                continue
            if k in state and state[k]["state"] in FINAL:
                continue
            p = _fnum(r.get("p"))
            if p is not None and not 0.0 <= p <= 1.0:
                res["skipped"].append(f"{site_id}:{sec}@{day} (p outside 0..1)")
                continue
            h = {**history(answers, sec, day), "n_answers_judge": judge_count(r.get("n_answers"))}
            v1 = decide(site_id, p, is_clear(r, cfg), h, cfg, trial=bool(secs[sec].get("trial")))
            d = dict(v1)
            aud, rate = (False, "")
            if v1["state"] == "auto":
                if cfg.get("auto_display"):
                    aud, rate = audit_pick(site_id, sec, day, v1["value"], h["prev_days"], cfg)
                else:          # shadow operation (§7.2): the rule is recorded, the owner sees every cell
                    d = {"state": "pending", "value": None, "reason": "shadow"}
            state[k] = {"site_id": site_id, "section_id": sec, "date": day, "acquired_utc": (r.get("acquired_utc") or "").strip(),
                        "state": d["state"], "value": "" if d["value"] is None else str(d["value"]),
                        "source": "auto" if d["state"] == "auto" else "", "p": _fmt_p(p),
                        "model_version": (r.get("model_version") or "").strip(), "rule_version": cfg["rule_version"],
                        "reason": d["reason"], "prev_answer_date": h["prev_answer_date"] or "",
                        "prev_answer": "" if h["prev_answer"] is None else str(h["prev_answer"]), "n_answers": str(h["n_answers"]),
                        "n_answers_judge": "" if h["n_answers_judge"] is None else str(h["n_answers_judge"]),
                        "answered_utc": "", "sections_version": str(secs[sec].get("sections_version", ""))}
            extra[k] = {"v1_state": v1["state"], "v1_value": "" if v1["value"] is None else str(v1["value"]),
                        "answers_through": (r.get("answers_through") or "").strip() or (h["prev_answer_date"] or ""),
                        "audit": "1" if aud else "0", "audit_rate": "" if rate == "" else f"{rate:.4f}"}
        # a cell judged now that the owner has answered already (a day of the review set): the judgement goes on record
        # as it was made, without the answer of the cell, before the answer is applied (§6.4-1)
        pre = {k: dict(state[k]) for k in extra if k in answers}
        _log_changes(out, site_id, pre, extra, now)
        _apply_answers(state, answers)
        _exclude_cells(site_id, state, excluded, cfg, today, load_log(out, site_id))
        q = _queue_rules(site_id, state, cfg, today)
        q = {k for k in q if not (k[0] in excluded and state[k]["state"] == "auto")}     # no audit of an excluded section
        im = _images(out, site_id)
        if hold_without_image:
            hk = {k for k in q if k[1] not in im}
            q -= hk
            held[site_id] = {k: state[k] for k in hk}
        q -= _out_of_image(state, secs, q, im, cfg)
        queued[site_id] = (state, secs, q, answers)
        pending_log[site_id] = extra
    waiting = _daily_limit(queued, cfg)
    queue = build_queue(out, queued, cfg, now, held if hold_without_image else None, waiting)
    for site_id, (state, secs, q, answers) in queued.items():
        extra = pending_log[site_id]
        for k in state:
            e = extra.setdefault(k, {})
            e["queued"] = "1" if k in q else "0"
            if state[k]["state"] == "auto" and cfg.get("auto_display") and "audit" not in e:
                r = state[k]
                pd = _days(r["prev_answer_date"], r["date"]) if r.get("prev_answer_date") else None
                e["audit"] = "1" if audit_pick(site_id, k[0], k[1], int(r["value"]), pd, cfg)[0] else "0"
            if state[k]["state"] in FINAL:
                e["queue_version"] = (answers.get(k) or {}).get("queue_version", "")
        n_log = _log_changes(out, site_id, state, extra, now, queue["queue_version"])
        _save_state(out, site_id, state)
        cnt = {}
        for r in state.values():
            cnt[r["state"]] = cnt.get(r["state"], 0) + 1
        res["sites"][site_id] = {"cells": len(state), "states": cnt, "queued": len(q), "log_rows": n_log}
    _write_json(out / DIR / "queue.json", queue)
    summ = summary(out, cfg, now)
    _write_json(out / DIR / "summary.json", summ)
    _write_json(out / DIR / "latest.json", latest(out, cfg, now, summ))
    res["queue"] = {"queue_version": queue["queue_version"], "cells": len(queue["cells"]), "waiting": sum(waiting.values())}
    if hold_without_image:
        res["held"] = {s: len(h) for s, h in sorted(held.items()) if h}
    return res


# ---------------------------------------------------------------- the queue
def _images(out: Path, site_id: str) -> dict:
    """date -> {"src" (relative to <out>), "grid"}: the pictures made for the queue (confirm/img/<site>/index.json), else the
    pictures of the review set (labels/<site>/img/, for the day of an earlier answer)."""
    res = {}
    rv = Path(out) / "labels" / site_id / "review.json"
    if rv.exists():
        for d in json.loads(rv.read_text(encoding="utf-8")).get("days", []):
            res[d["date"]] = {"src": f"labels/{site_id}/{d['image']}", "grid": d["grid"]}
    ix = Path(out) / DIR / "img" / site_id / "index.json"
    if ix.exists():
        for day, d in json.loads(ix.read_text(encoding="utf-8")).get("days", {}).items():
            res[day] = {"src": f"{DIR}/img/{site_id}/{d['image']}", "grid": d["grid"]}
    return res


def _out_of_image(state: dict, secs: dict, q: set, im: dict, cfg: dict) -> set:
    """The cells of ``q`` whose frame is not in the picture of their day (less than exclude.frame_in_image_min of it): the
    owner would see the black around the picture or the credit strip. They become excluded (reason excluded:out_of_image) and
    are returned. A day without a picture is not checked here (the daily run holds it back until its picture is made)."""
    m = (cfg.get("exclude") or {}).get("frame_in_image_min")
    if not m:
        return set()
    bad = set()
    for k in sorted(q):
        s = secs.get(k[0])
        if s and k[1] in im and in_image(s.get("frame") or [], im[k[1]]["grid"]) < float(m):
            state[k].update(state="excluded", reason="excluded:out_of_image")
            bad.add(k)
    return bad


def _queue_cell(site_id: str, k: tuple, r: dict) -> dict:
    return {"key": (site_id, k[0], k[1]), "state": r["state"], "reason": r["reason"], "p": _fnum(r["p"]),
            "prev_answer": int(r["prev_answer"]) if r["prev_answer"] != "" else None,
            "prev_answer_date": r["prev_answer_date"] or None, "acquired_utc": r["acquired_utc"]}


def _daily_limit(queued: dict, cfg: dict) -> dict:
    """§4-3: at most daily_max cells a day, the first ones of the order (_order); the others stay waiting for the next run
    (their expiry still counts; queued = 0, so merge takes no answer for them). Takes them out of the queued sets and returns
    {site: cells left for the next run}."""
    n = cfg.get("daily_max")
    cells = [_queue_cell(site_id, k, state[k]) for site_id, (state, _secs, q, _a) in queued.items() for k in q]
    if not n or len(cells) <= int(n):
        return {}
    left = {}
    for c in _order(cells, cfg)[int(n):]:
        site_id, sec, day = c["key"]
        queued[site_id][2].discard((sec, day))
        left[site_id] = left.get(site_id, 0) + 1
    return dict(sorted(left.items()))


def _order(cells: list[dict], cfg: dict) -> list[dict]:
    """§3.3-3: (1) the judge says the section changed since the last answer, (2) scores near 0.5, (3) sections with few
    answers, (4) sections whose last answer is old; the audit cells and the other confident cells are put among (2)-(4) at
    places fixed by their key, so that the place does not tell which they are."""
    def changed(c):
        return c["p"] is not None and c["prev_answer"] is not None and (c["p"] >= 0.5) != bool(c["prev_answer"])
    first = sorted((c for c in cells if changed(c)), key=lambda c: (abs(c["p"] - 0.5) * -1, c["key"]))
    rest = [c for c in cells if not changed(c)]
    main, mix = [], []
    for c in rest:
        why = c["reason"].split("+")
        g = 2 if "uncertain" in why else 3 if ("few_answers" in why or "answers_unknown" in why or "no_reference" in why) else 4 if "stale_answer" in why else None
        if g is None or c["state"] == "auto":
            mix.append(c)
        else:
            main.append((g, abs(c["p"] - 0.5) if c["p"] is not None else 0.5, c["key"], c))
    main.sort(key=lambda t: t[:3])
    keyed = [((float(i), 0, ""), t[3]) for i, t in enumerate(main)]
    for c in mix:
        u = _unit("order", *c["key"], cfg["rule_version"])
        keyed.append(((u * len(main) - 0.5, 1, "|".join(c["key"])), c))
    keyed.sort(key=lambda t: t[0])
    return first + [c for _, c in keyed]


def held_days(held: dict) -> list[dict]:
    """The days held back for want of a picture (run(hold_without_image=True)): port, day, number of cells and the days of
    their last answers. No section, score or judgement (§6.2)."""
    by = {}
    for site_id, cells in held.items():
        for (_sec, day), r in cells.items():
            d = by.setdefault((site_id, day), {"site_id": site_id, "date": day, "cells": 0, "prev_dates": set()})
            d["cells"] += 1
            if r.get("prev_answer_date"):
                d["prev_dates"].add(r["prev_answer_date"])
    return [dict(d, prev_dates=sorted(d["prev_dates"])) for _, d in sorted(by.items())]


def build_queue(out: Path, queued: dict, cfg: dict, now: str, held: dict | None = None, waiting: dict | None = None) -> dict:
    """queue.json for the page. Per cell: port, section, day, picture, the last answer (day, present / absent, picture) and
    its place. No score, no judgement and no audit mark go in (§6.2): the owner reads without seeing the judge. No file
    name either: the page names the file it downloads after the day of the download (confirm-answers-YYYY-MM-DD.csv;
    it holds every stored answer, of this queue and of earlier ones), and merge takes any name."""
    cells, sites, images = [], {}, {}
    for site_id, (state, secs, q, _answers) in queued.items():
        if not q:
            continue
        im = _images(out, site_id)
        used = {}
        for k in q:
            r = state[k]
            cells.append(_queue_cell(site_id, k, r))
            used[k[0]] = secs.get(k[0]) or next((s for s in (load_sections(out, site_id) or {}).get("sections", []) if s["id"] == k[0]), None)
            for day in (k[1], r["prev_answer_date"]):
                if day and day in im:
                    images[f"{site_id}|{day}"] = im[day]
        labels = section_labels(load_sections(out, site_id))
        sites[site_id] = {"sections": {i: {**{f: s.get(f) for f in ("id", "way", "kind", "sector", "length_m", "line", "frame")},
                                           "label": labels.get(i, ("", None))[0] or None}
                                       for i, s in sorted(used.items()) if s}}
    # §4-1: the cells of one port and day together (one picture after the other), the boxes in the order of their first cell
    # and the cells of a box in the order of _order (the place of a confident cell stays fixed by its key)
    boxes = {}
    for c in _order(cells, cfg):
        boxes.setdefault((c["key"][0], c["key"][2]), []).append(c)
    m = (cfg.get("exclude") or {}).get("frame_in_image_min")
    out_cells = []
    for n, c in enumerate(c for b in boxes.values() for c in b):
        site_id, sec, day = c["key"]
        prev = None
        if c["prev_answer_date"]:
            pk = f"{site_id}|{c['prev_answer_date']}"
            fr = ((sites.get(site_id) or {}).get("sections", {}).get(sec) or {}).get("frame") or []
            ok = pk in images and (not m or in_image(fr, images[pk]["grid"]) >= float(m))   # an older picture of other bounds
            prev = {"date": c["prev_answer_date"], "answer": c["prev_answer"], "image": pk if ok else None}
        ck = f"{site_id}|{day}"
        out_cells.append({"site_id": site_id, "section_id": sec, "date": day, "acquired_utc": c["acquired_utc"], "order": n,
                          "image": ck if ck in images else None, "prev": prev})
    dig = hashlib.sha256(json.dumps([[c["site_id"], c["section_id"], c["date"]] for c in out_cells]).encode()).hexdigest()[:8]
    years = sorted({k.split("|")[1][:4] for k in images} | {c["date"][:4] for c in out_cells})
    return {"version": QUEUE_VERSION, "queue_version": f"{now[:10]}-{dig}", "generated_utc": now, "rule_version": cfg["rule_version"],
            "zoom": 8, "frame_m": FRAME_M, "expire_days": cfg["expire_days"], "seconds_per_cell": 10,
            "daily_max": cfg.get("daily_max"), "waiting": {"cells": sum((waiting or {}).values()), "by_site": waiting or {},
                                                           "note": "1 日の上限（daily_max）を超えた升。翌日以降に出す（28 日の期限はそのまま数える）"},
            "answer_fields": ANSWER_FIELDS,
            "credit": f"{CREDIT} {', '.join(years)}".strip(), "attribution": ATTRIBUTION,
            "note": "判定器の点数・判定・抜き取り検査の印は入れていない（所有者の読みが判定器に引きずられないように）",
            "sites": sites, "images": images, "cells": out_cells,
            **({} if held is None else {"held": held_days(held),
                                        "held_note": "画像がまだ無い日の升は出さない（画像ができた次の実行で出す。28 日の期限はそのまま数える）"})}


# ---------------------------------------------------------------- merge
def merge(file: Path, out: Path = OUT, cfg: dict | None = None, now: str | None = None, by: str = "owner") -> dict:
    """Add the answers file the page downloads to labels/<site>-confirm-answers.csv. Only cells that were in a queue are
    accepted (a log row with queued = 1). A later answer replaces the earlier one of the same cell; the earlier one stays
    in the log. The state and the log are brought up to date, and the cells to judge again are returned (rejudge)."""
    out, cfg, now = Path(out), cfg or load_config(), now or utcnow()
    rows = read_csv(file)
    by_site = {}
    for r in rows:
        by_site.setdefault((r.get("site_id") or "").strip(), []).append(r)
    res = {"added": 0, "replaced": 0, "unchanged": 0, "rejected": [], "rejudge": {}}
    for site_id, rs in sorted(by_site.items()):
        was_queued = {(r["section_id"], r["date"]) for r in load_log(out, site_id) if r.get("queued") == "1"}
        path = confirm_answers_path(out, site_id)
        cur = {(r["section_id"], r["date"]): r for r in read_csv(path)}
        touched = []
        for r in rs:
            sec, day = (r.get("section_id") or "").strip(), (r.get("date") or "").strip()
            v, occ = (r.get("verified") or "").strip(), (r.get("verified_occupied") or "").strip()
            k = (sec, day)
            if k not in was_queued or not ((v == "1" and occ in ("0", "1")) or (v == "0" and occ == "")):
                res["rejected"].append(f"{site_id}:{sec}@{day}")
                continue
            new = {"site_id": site_id, "section_id": sec, "date": day, "verified": v, "verified_occupied": occ, "verified_by": by,
                   "verified_note": (r.get("verified_note") or "").strip(),
                   "answered_utc": (r.get("answered_utc") or "").strip() or now, "queue_version": (r.get("queue_version") or "").strip()}
            old = cur.get(k)
            if old and all((old.get(f) or "") == new[f] for f in ("verified", "verified_occupied", "answered_utc")):
                res["unchanged"] += 1
                continue
            res["replaced" if old else "added"] += 1
            cur[k] = new
            touched.append(k)
        if not touched:
            continue
        write_csv(path, sorted(cur.values(), key=lambda r: (r["date"], r["section_id"])), ANSWER_FIELDS)
        state = load_state(out, site_id)
        answers = owner_answers(out, site_id)
        _apply_answers(state, answers)
        extra = {k: {"queue_version": cur[k]["queue_version"]} for k in touched if k in state}
        _log_changes(out, site_id, {k: state[k] for k in touched if k in state}, extra, now)
        _save_state(out, site_id, state)
        res["rejudge"][site_id] = [f"{s}@{d}" for s, d in rejudge_scope(state, touched)]
    return res


# ---------------------------------------------------------------- summary (§7)
def _ci(units: dict, n_boot: int = 1000, seed: int = 0) -> list | None:
    """95% interval of sum(agree x w) / sum(w), resampling the scenes (site x day) with replacement. units: scene ->
    [(agree, w), ...]."""
    keys = sorted(units)
    if not keys:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        a = w = 0.0
        for _ in keys:
            for g, ww in units[keys[rng.randrange(len(keys))]]:
                a += g * ww
                w += ww
        if w:
            vals.append(a / w)
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(math.ceil(0.975 * (len(vals) - 1)))], 4)] if vals else None


def _rate(pairs: list, ci: bool = True) -> dict:
    """pairs: (scene, agree 0/1, weight)."""
    n, k = len(pairs), sum(a for _, a, _ in pairs)
    w = sum(ww for _, _, ww in pairs)
    doc = {"n": n, "agree": k, "rate": round(sum(a * ww for _, a, ww in pairs) / w, 4) if w else None}
    if ci:
        units = {}
        for sc, a, ww in pairs:
            units.setdefault(sc, []).append((a, ww))
        doc["ci95"] = _ci(units)
        doc["scenes"] = len(units)
    return doc


def scored_cells(out: Path, site_id: str) -> list[dict]:
    """Every cell the owner answered with present / absent, with its last log row from before the answer (§6.4-1): the
    judge is scored on what it said without having seen the answer of the cell."""
    last = {}
    for r in load_log(out, site_id):
        if r["state"] not in FINAL:
            last[(r["section_id"], r["date"])] = r
    res = []
    for k, a in owner_answers(out, site_id).items():
        r = last.get(k)
        if r is not None and a["y"] is not None:
            res.append({**r, "y": a["y"]})
    return res


A_RECENT_DAYS = 84          # §7.5: A over the last 12 weeks too


def _a_pairs(scored: list[dict], cfg: dict) -> list:
    """The pairs of A: the audit sample (weighted by 1 / audit_rate) when automatic states are shown, else (the shadow
    operation) every cell rule v1 would have made automatic."""
    res = []
    for r in scored:
        if r.get("trial"):
            continue                      # a trial section is never automatic: not in A
        if r["v1_state"] == "auto" and r["v1_value"] != "":
            agree = int(int(r["v1_value"]) == r["y"])
            if not cfg.get("auto_display"):
                res.append(((r["site_id"], r["date"]), agree, 1.0))
            elif r["audit"] == "1" and r["state"] == "auto":
                res.append(((r["site_id"], r["date"]), agree, 1.0 / (_fnum(r["audit_rate"]) or 1.0)))
    return res


def _block(states: list[dict], scored: list[dict], cfg: dict, picked: frozenset = frozenset(), now: str | None = None) -> dict:
    """picked: the (site, section, day) of the cells chosen for the audit (a log row with audit = 1)."""
    cnt = {s: 0 for s in STATES}
    for r in states:
        cnt[r["state"]] = cnt.get(r["state"], 0) + 1
    n_all = len(states)
    n_img = n_all - cnt["no_image"] - cnt["excluded"]
    would, audit, routed, changed, changed_routed = [], [], [], 0, 0
    for r in scored:
        sc, y, p = (r["site_id"], r["date"]), r["y"], _fnum(r["p"])
        if r["v1_state"] == "auto" and r["v1_value"] != "" and not r.get("trial"):
            would.append((sc, int(int(r["v1_value"]) == y), 1.0))
            if r["audit"] == "1" and r["state"] == "auto":
                audit.append((sc, int(int(r["v1_value"]) == y), 1.0 / (_fnum(r["audit_rate"]) or 1.0)))
        elif r["v1_state"] == "pending" and p is not None:
            routed.append((sc, int((p >= 0.5) == bool(y)), 1.0))
        if r["prev_answer"] != "" and int(r["prev_answer"]) != y:
            changed += 1
            changed_routed += r["v1_state"] == "pending"
    a = _rate(audit) if cfg.get("auto_display") else _rate(would)
    rt = _rate(routed, ci=False)
    # §7.5 / C5: what became of the audit sample (an expired one stays unanswered: A is computed without it), the two
    # kinds of unconfirmed cells, and the cells of the sections added after the ledger was started
    smp = {"picked": 0, "answered": 0, "expired": 0, "waiting": 0}
    for r in states:
        if (r["site_id"], r["section_id"], r["date"]) in picked:
            smp["picked"] += 1
            smp["answered" if r["state"] in FINAL else "expired" if now and _days(r["date"], now) > cfg["expire_days"] else "waiting"] += 1
    unc = {"expired": 0, "superseded": 0}
    for r in states:
        if r["state"] == "unconfirmed" and r["reason"] in unc:
            unc[r["reason"]] += 1
    new = [r for r in states if (r.get("sections_version") or "1") != "1"]
    ncnt = {}
    for r in new:
        ncnt[r["state"]] = ncnt.get(r["state"], 0) + 1
    recent = [r for r in scored if now and 0 <= _days(r["date"], now) < A_RECENT_DAYS]
    den = cnt["auto"] + cnt["human"] + cnt["unconfirmed"]
    c = None
    if den and (cnt["auto"] == 0 or a["rate"] is not None) and (cnt["unconfirmed"] == 0 or rt["rate"] is not None):
        c = round((cnt["auto"] * (a["rate"] or 0) + cnt["human"] + cnt["unconfirmed"] * (rt["rate"] or 0)) / den, 4)
    return {"cells": n_all, "states": cnt,
            "composition": {"of_cells_with_image": {s: (round(cnt[s] / n_img, 4) if n_img else None) for s in ("auto", "human", "pending", "unconfirmed", "unsure")},
                            "no_image_of_all": round(cnt["no_image"] / n_all, 4) if n_all else None,
                            "excluded_of_all": round(cnt["excluded"] / n_all, 4) if n_all else None,
                            "judgeable_of_all": round(n_img / n_all, 4) if n_all else None},
            "answered": len(scored),
            "A_auto_accuracy": {**a, "basis": "audit sample, weighted by 1 / audit_rate" if cfg.get("auto_display")
                                else "shadow operation: the cells rule v1 would have made automatic, all answered by the owner"},
            "A_auto_accuracy_last_12_weeks": _rate(_a_pairs(recent, cfg)),
            "would_be_auto": _rate(would), "audit": _rate(audit), "audit_sample": smp,
            "unconfirmed": unc, "new_sections": {"cells": len(new), "states": ncnt},
            "routed_judge_accuracy": rt,
            "changed": {"n": changed, "routed": changed_routed},
            "C_total_owner_counted_correct": c}


def summary(out: Path = OUT, cfg: dict | None = None, now: str | None = None) -> dict:
    """The numbers of §7 from the state files, the log and the answers. Split by rule / judge / sections version and by
    "after frozen_utc or not" (never mixed); and by ISO week. Agreement is agreement with the owner's reading."""
    out, cfg, now = Path(out), cfg or load_config(), now or utcnow()
    frozen = (cfg.get("frozen_utc") or "")[:10]
    states, scored, picked = [], [], set()
    for site_id in known_sites(out):
        led = (load_sections(out, site_id) or {}).get("sections", [])
        xs = {s["id"] for s in led if s.get("queue") == "excluded"}
        # confirm-x1: every cell of an excluded section is counted as excluded (B), and none is in A or C
        states += [{**r, "state": "excluded"} if r["section_id"] in xs else r for r in load_state(out, site_id).values()]
        tr = {s["id"] for s in led if s.get("trial")}
        scored += [{**r, "trial": r["section_id"] in tr} for r in scored_cells(out, site_id) if r["section_id"] not in xs]
        picked |= {(site_id, r["section_id"], r["date"]) for r in load_log(out, site_id) if r.get("audit") == "1"}
    picked = frozenset(picked)

    def part(r):
        return "after_freeze" if frozen and r["date"] > frozen else "before_freeze"

    def ver(r):
        return f"{r.get('rule_version', '')}|{r.get('model_version', '')}|sections-{r.get('sections_version', '')}"

    def week(r):
        y, w, _ = _day(r["date"]).isocalendar()
        return f"{y}-W{w:02d}"

    doc = {"version": VERSION, "generated_utc": now, "rule_version": cfg["rule_version"], "frozen_utc": cfg.get("frozen_utc"),
           "auto_display": bool(cfg.get("auto_display")),
           "notes": {"A": "自動確定の升の正答率＝所有者が判定器の答えを見ずに答えた結果との一致率。約束の数字（目標：CI の下限で 0.95 以上）",
                     "B": "構成比。像のある升のうちの 自動確定／人が確認／確認待ち／未確認／判断できない と、全部の升のうちの像なし・確認の対象外（excluded：confirm-x1 で外した区間の升と、枠が画像の外の升。像のある升にも A・C にも入れない）",
                     "C": "合計の一致率（参考）：人が確認した升を正答と数えた値。未確認の升は正答に数えず、回した升の判定器の正答率を掛ける。判定器の成績ではない",
                     "agreement": "どの数字も所有者の読みとの一致。所有者の読みの誤りは入っていない",
                     "ci": "95% CI は撮影（港×日）単位のブートストラップ 1,000 回、種 0",
                     "freeze": "frozen_utc より後に撮られた升（after_freeze）だけが、規則を先に決めてから入った答え。before_freeze の数字は規則を選んだ升を含み楽観がある",
                     "null": "null は数えられない（升が無い）。0 は数えて 0",
                     "A_last_12_weeks": "A と同じ数え方で、撮影日が generated_utc の日から 84 日（12 週）以内の升だけ",
                     "audit_sample": "抜き取りに選んだ升（記録に audit=1 の行がある升）と、そのうち答えがあった／期限（expire_days）を過ぎて答えのない／まだ待っている升の数。期限を過ぎた升は A に入らない（A が偏りうるので数を出す）",
                     "unconfirmed": "未確認の升の内訳：期限切れ（expired）と、後の撮影に置き換え（superseded）",
                     "new_sections": "区間の台帳を作った後に足された区間（sections_version が 2 以上）の升の数と状態",
                     "trial": "試行の区間（台帳で trial: true、sat/confirm_trial.json）は規則 v1 でも必ず所有者に回し（理由 trial）、A（自動確定の一致率・抜き取り）に入れない。回した升の判定器の正答率（routed_judge_accuracy）と変わった升（changed）には試行の区間の升も入る"},
           "parts": {}}
    for name in ("before_freeze", "after_freeze"):
        st = [r for r in states if part(r) == name]
        sc = [r for r in scored if part(r) == name]
        blk = {"total": _block(st, sc, cfg, picked, now), "by_site": {}, "by_version": {}, "by_week": {}}
        for label, fn in (("by_site", lambda r: r["site_id"]), ("by_version", ver), ("by_week", week)):
            for g in sorted({fn(r) for r in st} | {fn(r) for r in sc}):
                blk[label][g] = _block([r for r in st if fn(r) == g], [r for r in sc if fn(r) == g], cfg, picked, now)
        doc["parts"][name] = blk
    return doc


# ---------------------------------------------------------------- readable names of the sections
KIND_JA = {"pier": "桟橋", "coastline": "海岸線", "quay": "岸壁", "breakwater": "防波堤"}


def section_labels(ledger: dict | None) -> dict:
    """id -> (label, sort key): "区画 A1・桟橋 3・左側 1" = the sector, the kind and the number of its OSM line among the lines of
    that kind in the sector, the side of the line (L 左側 / R 右側, as in the id) and the piece on that side (the id's last
    part + 1). The lines are numbered along the shore (by longitude or latitude, whichever spreads more) among the lines
    that entered the ledger in the same sections_version, and the lines of a later version are numbered after the earlier
    ones: a label never changes while the ledger keeps its ids (retired sections keep their numbers)."""
    secs = (ledger or {}).get("sections", [])
    lines = {}
    for s in secs:
        key = (s.get("sector") or "", s.get("kind") or "", s.get("way"))
        pts = s.get("line") or []
        e = lines.setdefault(key, {"v": int(s.get("sections_version") or 0), "pts": [], "first": len(lines)})
        e["v"] = min(e["v"], int(s.get("sections_version") or 0))
        e["pts"] += [p for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    groups = {}
    for (sector, kind, way), e in lines.items():
        groups.setdefault((sector, kind), {}).setdefault(e["v"], []).append((way, e))
    number = {}
    for (sector, kind), by_v in groups.items():
        n = 0
        for v in sorted(by_v):
            ws = by_v[v]

            def centre(e):
                pts = e["pts"]
                return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else (0.0, 0.0)
            cs = [centre(e) for _, e in ws]
            lat0 = sum(c[1] for c in cs) / len(cs)
            spread_x = (max(c[0] for c in cs) - min(c[0] for c in cs)) * math.cos(math.radians(lat0))
            spread_y = max(c[1] for c in cs) - min(c[1] for c in cs)
            along = (lambda c: c[0]) if spread_x >= spread_y else (lambda c: -c[1])   # west to east, or north to south
            for (way, e), c in sorted(zip(ws, cs), key=lambda t: (along(t[1]), t[0][1]["first"])):
                n += 1
                number[(sector, kind, way)] = n
    res = {}
    for s in secs:
        sector, kind = s.get("sector") or "", s.get("kind") or ""
        n = number[(sector, kind, s.get("way"))]
        parts = str(s["id"]).split(":")
        side = {"L": "左側", "R": "右側"}.get(parts[1] if len(parts) > 2 else "", "")
        piece = int(parts[-1]) + 1 if len(parts) > 2 and parts[-1].isdigit() else None
        tail = (side + (f" {piece}" if piece is not None else "")).strip() if len(parts) > 2 else str(s["id"])   # another form of id: itself
        label = "・".join(x for x in (f"区画 {sector}" if sector else "", f"{KIND_JA.get(kind, kind)} {n}", tail) if x)
        res[s["id"]] = (label, (sector, kind, n, parts[1] if len(parts) > 2 else "", piece or 0, str(s["id"])))
    return res


# ---------------------------------------------------------------- latest (§6.6: the trial page and the site)
LATEST_NOTE = "試行。在／不在は Sentinel-2（10 m）の像についての判定と、所有者が同じ像を目で見た確認。真の在／不在ではない"
LATEST_SOURCE = "Element84 Earth Search／AWS Open Data 経由"


def port_name(out: Path, site_id: str) -> str:
    rv = Path(out) / "labels" / site_id / "review.json"
    if rv.exists():
        return json.loads(rv.read_text(encoding="utf-8")).get("name") or site_id
    sites = Path(__file__).resolve().parent / "sites.json"
    for s in json.loads(sites.read_text(encoding="utf-8")).get("sites", []) if sites.exists() else []:
        if s.get("id") == site_id:
            return s.get("name") or site_id
    return site_id


SITE_A_MIN_CELLS = 30      # a port's A is withheld below this many cells: one new answer would move it and tell the rule's answer


def _numbers(block: dict, floor: int = 0) -> dict:
    """The numbers shown beside the table (§6.6): A with its count and CI, the composition, the share that could be judged.
    With ``floor``, A of fewer cells than that is withheld (rate and CI null, with the reason)."""
    a = dict(block["A_auto_accuracy"])
    if floor and a["n"] < floor:
        a.update(rate=None, ci95=None, withheld=f"升が少ない（{floor} 升未満）")
    # basis: "sample" = the audit sample of the automatic cells (§7.3); "shadow" = the cells rule v1 would have made automatic,
    # all of them shown to the owner (§7.2)
    return {"A": {"rate": a["rate"], "n": a["n"], "ci95": a.get("ci95"), "basis": "shadow" if a["basis"].startswith("shadow") else "sample",
                  "withheld": a.get("withheld")},
            "composition": block["composition"]["of_cells_with_image"], "no_image_of_all": block["composition"]["no_image_of_all"],
            "excluded_of_all": block["composition"].get("excluded_of_all"),
            "judgeable_of_all": block["composition"]["judgeable_of_all"], "cells": block["cells"], "states": block["states"]}


def latest(out: Path = OUT, cfg: dict | None = None, now: str | None = None, summ: dict | None = None) -> dict:
    """Per port and active section of the ledger: the newest cell (its state, day, and value when that is a value shown:
    the owner's, or an automatic one while auto_display is on), and for a cell without a value the last value shown and its
    day. In the shadow operation an automatic state is shown as waiting (the judge's answer is not shown before the owner's).
    null is unknown; 0 is absent, only for a cell judged or confirmed absent."""
    out, cfg, now = Path(out), cfg or load_config(), now or utcnow()
    summ = summ or summary(out, cfg, now)
    show_auto = bool(cfg.get("auto_display"))
    frozen = bool(cfg.get("frozen_utc"))
    part = "after_freeze" if frozen and summ["parts"]["after_freeze"]["total"]["cells"] else "before_freeze"
    blk = summ["parts"][part]
    sites, years = {}, set()
    for site_id in known_sites(out):
        ledger = load_sections(out, site_id)
        secs = active_sections(ledger)
        labels = section_labels(ledger)
        by_sec = {}
        for r in load_state(out, site_id).values():
            by_sec.setdefault(r["section_id"], []).append(r)
        rows, cnt = [], {}
        for sid in sorted(secs, key=lambda i: labels[i][1]):
            s = secs[sid]
            cells = sorted(by_sec.get(sid, []), key=lambda r: r["date"])
            row = {"id": sid, "label": labels[sid][0], "kind": s.get("kind"), "sector": s.get("sector"), "length_m": s.get("length_m"),
                   "state": None, "date": None, "value": None, "last_date": None, "last_value": None, "last_source": None}
            if s.get("trial"):          # a trial section (sat/confirm_trial.json): shown, never counted elsewhere
                row.update(trial=True, role=s.get("role"), candidate=s.get("candidate"))
            if s.get("queue") == "excluded":    # confirm-x1: kept in the list, marked
                row.update(excluded=True, excluded_reason=s.get("excluded_reason"),
                           excluded_note=EXCLUDE_REASONS.get(s.get("excluded_reason"), s.get("excluded_reason")))
            shown = [r for r in cells if (r["state"] == "human" or (show_auto and r["state"] == "auto")) and r["value"] in ("0", "1")]
            if cells:
                r = cells[-1]
                years.add(r["date"][:4])
                st = r["state"] if (show_auto or r["state"] != "auto") else "pending"
                row.update(state=st, date=r["date"])
                if st in ("human", "auto") and r["value"] in ("0", "1"):
                    row["value"] = int(r["value"])
                elif shown:
                    q = shown[-1]
                    row.update(last_date=q["date"], last_value=int(q["value"]), last_source="owner" if q["state"] == "human" else "auto")
                cnt[st] = cnt.get(st, 0) + 1
            rows.append(row)
        sites[site_id] = {"name": port_name(out, site_id), "sections": rows, "sections_by_state": cnt,
                          "numbers": _numbers(blk["by_site"][site_id], SITE_A_MIN_CELLS) if site_id in blk["by_site"] else None}
    return {"version": LATEST_VERSION, "generated_utc": now, "rule_version": cfg["rule_version"], "auto_display": show_auto,
            "frozen_utc": cfg.get("frozen_utc"), "expire_days": cfg["expire_days"], "part": part,
            "note": LATEST_NOTE, "credit": f"{CREDIT} {', '.join(sorted(years))}".strip(), "source": LATEST_SOURCE,
            "attribution": ATTRIBUTION,
            "null": "null は不明（未確認・像なし・判断できない・まだ升が無い）。0 は不在と判定または確認された升だけ",
            "notes": {"sections_by_state": "港ごとの区間の数：各区間のいちばん新しい升の状態で数えた区間の数（1 区間＝1）",
                      "numbers.states": "升の数：全部の日の全部の升を状態ごとに数えた数（numbers.composition はその割合）",
                      "trial": "trial: true の区間は試行の区間（sat/confirm_trial.json、区画「試行」）。所有者の確認に出すだけで、公開の隻数・評価票・判定器の再学習には入れない。role：naval_empty＝軍の桟橋で今は空（使い始めを見る）、fitout＝修理・艤装の岸壁（配備の数には入れない）",
                      "excluded": "excluded: true の区間は確認の対象外（confirm-x1：区間の中点の線が着く線の外・berth false・民用・区分が海軍でない）。判定と記録は続け、所有者の確認には出さない。区間の線の区分（sat/berth_class）が変われば次の実行で決め直す。数（A・C）に入れない",
                      "label": "区間の読める名前：区画・種別とその区画の中での線の番号（岸に沿った順）・線の左側／右側（OSM の線の向きに対して）とその側の何番目（id の最後の番号＋1）。区間の id（OSM の線の番号:側:番号）が変わらない限り変わらない",
                      "A": f"規則 v1 なら自動確定の升の一致率（影の運用）／自動確定の正答率（抜き取り）。港ごとの値は {SITE_A_MIN_CELLS} 升未満なら出さない（rate・ci95 は null、withheld に理由）。全体の値は升の数によらず出す"},
            "numbers": _numbers(blk["total"]), "sites": sites}


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="confirm-1: cells the judge is not sure of go to the owner (no network, PU 0)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sections", "run", "merge", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=OUT)
        p.add_argument("--config", type=Path, default=CONFIG_PATH)
        p.add_argument("--now", help="UTC time of the run (default: now)")
        if name == "sections":
            p.add_argument("--site", required=True)
            p.add_argument("--retire", nargs="*", default=[], help="section ids to mark as retired")
        elif name == "run":
            p.add_argument("--scores", type=Path, required=True, help="the score table (CSV; see the module text)")
        elif name == "merge":
            p.add_argument("--file", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg, now = load_config(a.config), a.now or utcnow()
    if a.cmd == "sections":
        led = sync_sections(a.out, a.site, review_sections(a.out, a.site), now, tuple(a.retire))
        doc = {"site_id": a.site, "sections_version": led["sections_version"], "sections": len(led["sections"]),
               "active": len(active_sections(led))}
    elif a.cmd == "run":
        doc = run(read_csv(a.scores), a.out, cfg, now)
    elif a.cmd == "merge":
        doc = merge(a.file, a.out, cfg, now)
    else:
        doc = summary(a.out, cfg, now)
        _write_json(a.out / DIR / "summary.json", doc)
        _write_json(a.out / DIR / "latest.json", latest(a.out, cfg, now, doc))
        doc = {"written": str(a.out / DIR / "summary.json"), "parts": {k: v["total"]["states"] for k, v in doc["parts"].items()}}
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
