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

from . import atomic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
CONFIG_PATH = Path(__file__).resolve().parent / "confirm.json"
VERSION = "confirm-1"
QUEUE_VERSION = "confirm-queue-1"
DIR = "confirm"
FRAME_M = 120.0
STATES = ("auto", "human", "pending", "unconfirmed", "unsure", "no_image")
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


def decide(site_id: str, p: float | None, clear: bool, hist: dict, cfg: dict) -> dict:
    """Rule v1 for one cell: {"state": auto / pending / no_image, "value": 1 / 0 / None, "reason"}. The reasons a cell is
    shown for are joined with "+", in the order of the design (§2.5, §4).

    The number of answers is the smaller of hist["n_answers"] (the answer files) and hist["n_answers_judge"] (the answers
    the judge used for this score). A judge count that is missing or None is not known: the cell is shown
    (answers_unknown), never settled on the count of the files alone. The days since the last answer come from the files."""
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


def run(scores: list[dict], out: Path = OUT, cfg: dict | None = None, now: str | None = None) -> dict:
    """Judge the cells of the score table and bring state, log, queue and summary up to date. A cell the owner answered is
    never judged again; a cell outside the table keeps its judgement (only the queue rules and new answers reach it)."""
    out, cfg, now = Path(out), cfg or load_config(), now or utcnow()
    today = now[:10]
    by_site = {}
    for r in scores:
        by_site.setdefault((r.get("site_id") or "").strip(), []).append(r)
    res = {"now": now, "sites": {}, "skipped": []}
    queued = {}
    pending_log = {}
    for site_id in sorted(set(known_sites(out)) | set(by_site)):
        ledger = load_sections(out, site_id)
        secs = active_sections(ledger)
        if ledger is None:
            res["skipped"] += [f"{site_id}:{r.get('section_id')}@{r.get('date')} (no ledger)" for r in by_site.get(site_id, [])]
            continue
        state = load_state(out, site_id)
        answers = owner_answers(out, site_id)
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
            v1 = decide(site_id, p, is_clear(r, cfg), h, cfg)
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
        q = _queue_rules(site_id, state, cfg, today)
        queued[site_id] = (state, secs, q, answers)
        pending_log[site_id] = extra
    queue = build_queue(out, queued, cfg, now)
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
    _write_json(out / DIR / "summary.json", summary(out, cfg, now))
    res["queue"] = {"queue_version": queue["queue_version"], "cells": len(queue["cells"])}
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


def build_queue(out: Path, queued: dict, cfg: dict, now: str) -> dict:
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
            cells.append({"key": (site_id, k[0], k[1]), "state": r["state"], "reason": r["reason"], "p": _fnum(r["p"]),
                          "prev_answer": int(r["prev_answer"]) if r["prev_answer"] != "" else None,
                          "prev_answer_date": r["prev_answer_date"] or None, "acquired_utc": r["acquired_utc"]})
            used[k[0]] = secs.get(k[0]) or next((s for s in (load_sections(out, site_id) or {}).get("sections", []) if s["id"] == k[0]), None)
            for day in (k[1], r["prev_answer_date"]):
                if day and day in im:
                    images[f"{site_id}|{day}"] = im[day]
        sites[site_id] = {"sections": {i: {f: s.get(f) for f in ("id", "way", "kind", "sector", "length_m", "line", "frame")}
                                       for i, s in sorted(used.items()) if s}}
    out_cells = []
    for n, c in enumerate(_order(cells, cfg)):
        site_id, sec, day = c["key"]
        prev = None
        if c["prev_answer_date"]:
            pk = f"{site_id}|{c['prev_answer_date']}"
            prev = {"date": c["prev_answer_date"], "answer": c["prev_answer"], "image": pk if pk in images else None}
        ck = f"{site_id}|{day}"
        out_cells.append({"site_id": site_id, "section_id": sec, "date": day, "acquired_utc": c["acquired_utc"], "order": n,
                          "image": ck if ck in images else None, "prev": prev})
    dig = hashlib.sha256(json.dumps([[c["site_id"], c["section_id"], c["date"]] for c in out_cells]).encode()).hexdigest()[:8]
    years = sorted({k.split("|")[1][:4] for k in images} | {c["date"][:4] for c in out_cells})
    return {"version": QUEUE_VERSION, "queue_version": f"{now[:10]}-{dig}", "generated_utc": now, "rule_version": cfg["rule_version"],
            "zoom": 8, "frame_m": FRAME_M, "expire_days": cfg["expire_days"], "seconds_per_cell": 10,
            "answer_fields": ANSWER_FIELDS,
            "credit": f"{CREDIT} {', '.join(years)}".strip(), "attribution": ATTRIBUTION,
            "note": "判定器の点数・判定・抜き取り検査の印は入れていない（所有者の読みが判定器に引きずられないように）",
            "sites": sites, "images": images, "cells": out_cells}


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
    n_img = n_all - cnt["no_image"]
    would, audit, routed, changed, changed_routed = [], [], [], 0, 0
    for r in scored:
        sc, y, p = (r["site_id"], r["date"]), r["y"], _fnum(r["p"])
        if r["v1_state"] == "auto" and r["v1_value"] != "":
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
        states += list(load_state(out, site_id).values())
        scored += scored_cells(out, site_id)
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
                     "B": "構成比。像のある升のうちの 自動確定／人が確認／確認待ち／未確認／判断できない と、全部の升のうちの像なし",
                     "C": "合計の一致率（参考）：人が確認した升を正答と数えた値。未確認の升は正答に数えず、回した升の判定器の正答率を掛ける。判定器の成績ではない",
                     "agreement": "どの数字も所有者の読みとの一致。所有者の読みの誤りは入っていない",
                     "ci": "95% CI は撮影（港×日）単位のブートストラップ 1,000 回、種 0",
                     "freeze": "frozen_utc より後に撮られた升（after_freeze）だけが、規則を先に決めてから入った答え。before_freeze の数字は規則を選んだ升を含み楽観がある",
                     "null": "null は数えられない（升が無い）。0 は数えて 0",
                     "A_last_12_weeks": "A と同じ数え方で、撮影日が generated_utc の日から 84 日（12 週）以内の升だけ",
                     "audit_sample": "抜き取りに選んだ升（記録に audit=1 の行がある升）と、そのうち答えがあった／期限（expire_days）を過ぎて答えのない／まだ待っている升の数。期限を過ぎた升は A に入らない（A が偏りうるので数を出す）",
                     "unconfirmed": "未確認の升の内訳：期限切れ（expired）と、後の撮影に置き換え（superseded）",
                     "new_sections": "区間の台帳を作った後に足された区間（sections_version が 2 以上）の升の数と状態"},
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
        doc = {"written": str(a.out / DIR / "summary.json"), "parts": {k: v["total"]["states"] for k, v in doc["parts"].items()}}
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
