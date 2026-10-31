"""Collect the PLA Activity Tracker daily counts into docs/data/pla-tracker/.

Source: PLA Activity Tracker (https://pla-tracker.skyfaring.net/), one CSV
(``/data/records.csv``) compiled by an individual from the Taiwan Ministry of
National Defense (國防部) daily reports. CC BY 4.0. Used from 2026-10-31 on,
because the Ministry's own site disallows automated access in robots.txt
(collectors/taiwan_mnd.py SUSPENDED).

  python -m collectors.pla_tracker                 # daily run (one request)
  python -m collectors.pla_tracker --out DIR       # write to DIR instead (probe)
  python -m collectors.pla_tracker --from-file F   # rebuild from a saved CSV (no request)
  python -m collectors.pla_tracker --about         # probe: also read the about page once

Politeness: robots.txt is checked first (collectors/polite.py); the run is
skipped when it disallows us or cannot be checked, and a bot-challenge page
stops it. One CSV per run, no retries.

Explicit permission: the maker's permission is recorded in
collectors/permissions.json ("pla-tracker"). It is used only if robots.txt
disallows the CSV; then the CSV (and nothing else) is fetched, and status.json says so (``last_run.access`` = "permission" and
``last_run.permission``; otherwise ``access`` = "robots"). An expired or
missing permission means the run is skipped as before.

At most one request for the CSV per UTC day, whichever way it is allowed and
also on the probe path (--save-raw): ``last_request_utc`` of status.json (the
output directory's and the published one) is checked before the request. A run
that stops for this reason leaves status.json untouched.

Rows are merged by date: a value changed by the source replaces the stored
one (listed in status.json ``last_run.revised``); a date that disappears from
the source is kept. A CSV that is much shorter than what is stored, or does
not have the expected columns, is not used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import re
import sys
import urllib.error
from datetime import date, datetime, timezone
from pathlib import Path

from collectors import polite

CSV_URL = "https://pla-tracker.skyfaring.net/data/records.csv"
ABOUT_URL = "https://pla-tracker.skyfaring.net/en/about"
SITE_URL = "https://pla-tracker.skyfaring.net/"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "pla-tracker"
MND = ROOT / "docs" / "data" / "taiwan-mnd"
SOURCE_ID = "pla-tracker"  # key in collectors/permissions.json
# A plain bot name without contact details (decided 2026-10-31). Never a browser-like string.
UA = "observatory-data/1"
TIMEOUT = 60

SOURCE_COLUMNS = ["date", "aircraft_total", "median_line_cross", "cross_rate", "aircraft_type",
                  "ships_total", "activity_start", "activity_end", "special_event"]
INT_COLUMNS = ("aircraft_total", "median_line_cross", "ships_total")
FIELDS = SOURCE_COLUMNS + ["retrieved_utc"]
MIN_KEEP_RATIO = 0.8  # a CSV with fewer rows than this share of the stored ones is not used
MAX_COUNT = 400       # larger counts are treated as broken rows

# The Taiwan MND series and the fields compared with it (tracker -> MND daily.csv).
COMPARE = {"aircraft_total": "aircraft", "median_line_cross": "median_line_or_adiz", "ships_total": "plan_vessels"}
# Extra pairs reported in compare.json but not in the "all three equal" count.
COMPARE_EXTRA = {"ships_total_vs_plan_plus_official": ("ships_total", "plan_plus_official")}

# Repeated in daily.json so the data carries its credit wherever it is copied (same as meta.json).
ATTRIBUTION = {
    "author": "Adam Pan（Skyfaring）, PLA Activity Tracker",
    "url": SITE_URL,
    # The wording the site asks for (about page, "Citation & license", read 2026-10-30).
    "credit": "Source: ROC Ministry of National Defense; compilation & charts: PLA Activity Tracker (https://pla-tracker.skyfaring.net), "
              "by Adam Pan (Skyfaring), CC BY 4.0. Republished by observatory-data; this does not imply official involvement or endorsement by the PLA Activity Tracker or its maker.",
    "credit_ja": "出典：中華民國（台湾）國防部。集約：PLA Activity Tracker（https://pla-tracker.skyfaring.net、Adam Pan／Skyfaring、CC BY 4.0）",
    "disclaimer": "observatory-data による再掲であり、PLA Activity Tracker およびその制作者の公式の関与・承認を示すものではない（制作者の条件）",
    "license": "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/",
    "processing": "observatory-data（https://github.com/2wrr6c7mmj-jpg/observatory-data）が records.csv を日ごとに取り込み、"
                  "数値の列を整数にし、取得時刻を付け、国防部の日報（docs/data/taiwan-mnd/）と突き合わせた。値そのものは変えていない。",
}


class Stop(Exception):
    """The source cannot be used in this run."""


class AlreadyToday(Stop):
    """The CSV was already requested this UTC day (no request, status.json untouched)."""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def notice(msg: str, level: str = "notice", title: str = "pla-tracker") -> None:
    text = msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{text[:3900]}")


# ---------------------------------------------------------------- parsing

def _int(v: str, col: str, day: str) -> int | None:
    v = (v or "").strip()
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"{day} {col}: not a number: {v!r}") from None
    if f != int(f) or f < 0 or f > MAX_COUNT:
        raise ValueError(f"{day} {col}: out of range: {v!r}")
    return int(f)


def parse_csv(text: str) -> tuple[dict[str, dict], list[str]]:
    """Rows by date, and the rows that were skipped (with the reason)."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    cols = [c.strip() for c in (reader.fieldnames or [])]
    missing = [c for c in SOURCE_COLUMNS if c not in cols]
    if missing:
        raise Stop(f"columns missing: {missing} (got {cols})")
    rows: dict[str, dict] = {}
    skipped: list[str] = []
    for raw in reader:
        raw = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k is not None}
        day = raw.get("date", "")
        try:
            date.fromisoformat(day)
        except ValueError:
            skipped.append(f"bad date {day!r}")
            continue
        try:
            rec = {c: raw.get(c, "") for c in SOURCE_COLUMNS}
            for c in INT_COLUMNS:
                rec[c] = _int(raw.get(c, ""), c, day)
        except ValueError as e:
            skipped.append(str(e))
            continue
        if day in rows and rows[day] != rec:
            skipped.append(f"{day}: duplicate date with different values (first kept)")
            continue
        rows[day] = rec
    return rows, skipped


def _cell(v) -> str:
    return "" if v is None else str(v)


def read_stored(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            rec = {c: r.get(c, "") for c in FIELDS}
            for c in INT_COLUMNS:
                rec[c] = int(rec[c]) if rec[c] != "" else None
            out[r["date"]] = rec
        return out


def merge(stored: dict[str, dict], fresh: dict[str, dict], retrieved: str) -> tuple[dict[str, dict], dict]:
    if stored and len(fresh) < MIN_KEEP_RATIO * len(stored):
        raise Stop(f"the CSV has {len(fresh)} rows, fewer than {MIN_KEEP_RATIO:.0%} of the {len(stored)} stored; not used")
    out = dict(stored)
    added, revised = [], []
    for day, rec in fresh.items():
        old = stored.get(day)
        if old is None:
            added.append(day)
        elif any(old.get(c) != rec.get(c) for c in SOURCE_COLUMNS):
            revised.append({"date": day, "changed": {c: [old.get(c), rec.get(c)] for c in SOURCE_COLUMNS
                                                     if old.get(c) != rec.get(c)}})
        else:
            continue
        out[day] = {**rec, "retrieved_utc": retrieved}
    gone = sorted(set(stored) - set(fresh))
    return out, {"added": len(added), "added_range": [min(added), max(added)] if added else None,
                 "revised": revised[:50], "revised_count": len(revised),
                 "kept_not_in_source": gone[:50], "kept_not_in_source_count": len(gone)}


# ---------------------------------------------------------------- comparison with the MND reports

def read_mnd(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("format") != "around-taiwan":
                continue
            rec = {k: (int(r[k]) if r.get(k, "") != "" else None) for k in list(COMPARE.values()) + ["official_ships"]}
            # ships_total of the tracker includes the official ships from about 2026-03 (see meta.json)
            rec["plan_plus_official"] = (None if rec["plan_vessels"] is None
                                         else rec["plan_vessels"] + (rec["official_ships"] or 0))
            out[r["report_date"]] = rec
    return out


def _shift(day: str, n: int) -> str:
    return date.fromordinal(date.fromisoformat(day).toordinal() + n).isoformat()


def compare(rows: dict[str, dict], mnd: dict[str, dict], examples: int = 15) -> dict:
    """Agreement between the tracker and the MND around-taiwan reports (2022-11-16 on).

    ``offsets`` checks the date alignment: tracker date + n days against the
    MND report date (0 = same date)."""
    both = sorted(set(rows) & set(mnd))
    res = {"mnd_format": "around-taiwan", "overlap_days": len(both),
           "range": [both[0], both[-1]] if both else None, "fields": {}, "offsets": {}}
    for n in (-1, 0, 1):
        eq = tot = 0
        for day, rec in rows.items():
            m = mnd.get(_shift(day, n))
            if m is None:
                continue
            for t, k in COMPARE.items():
                if rec.get(t) is not None and m.get(k) is not None:
                    tot += 1
                    eq += rec[t] == m[k]
        res["offsets"][str(n)] = {"pairs": tot, "equal": eq, "rate": round(eq / tot, 4) if tot else None}
    all_equal = 0
    pairs_to_check = [(t, t, k) for t, k in COMPARE.items()] + [(name, t, k) for name, (t, k) in COMPARE_EXTRA.items()]
    for name, t, k in pairs_to_check:
        pairs = [(d, rows[d][t], mnd[d].get(k)) for d in both]
        comparable = [(d, a, b) for d, a, b in pairs if a is not None and b is not None]
        diff = [(d, a, b) for d, a, b in comparable if a != b]
        only_t = sum(1 for _, a, b in pairs if a is not None and b is None)
        only_m = sum(1 for _, a, b in pairs if a is None and b is not None)
        big = sorted(diff, key=lambda x: (-abs(x[1] - x[2]), x[0]))[:examples]
        by_month: dict[str, list[int]] = {}
        for d, a, b in comparable:
            by_month.setdefault(d[:7], [0, 0])
            by_month[d[:7]][0] += a == b
            by_month[d[:7]][1] += 1
        res["fields"][name] = {
            "tracker_field": t, "mnd_field": k,
            "by_month": {mo: f"{e}/{n}" for mo, (e, n) in sorted(by_month.items())}, "comparable": len(comparable), "equal": len(comparable) - len(diff),
            "rate": round((len(comparable) - len(diff)) / len(comparable), 4) if comparable else None,
            "tracker_only": only_t, "mnd_only": only_m,
            "mean_abs_diff": round(sum(abs(a - b) for _, a, b in diff) / len(comparable), 3) if comparable else None,
            "examples_latest": [{"date": d, "tracker": a, "mnd": b} for d, a, b in diff[-examples:]],
            "examples_largest": [{"date": d, "tracker": a, "mnd": b} for d, a, b in big],
        }
    # Headline: the two counts whose definitions match (median_line_cross is narrower, see meta.json).
    for d in both:
        if (rows[d]["aircraft_total"] == mnd[d]["aircraft"]
                and rows[d]["ships_total"] == mnd[d]["plan_plus_official"]):
            all_equal += 1
    res["days_aircraft_and_ships_equal"] = all_equal
    res["days_aircraft_and_ships_rate"] = round(all_equal / len(both), 4) if both else None
    return res


def compare_text(c: dict) -> str:
    lines = [f"重なり {c['overlap_days']} 日 {c['range']}、機数と艦艇（国防部は海軍艦艇＋公務船）がともに一致 "
             f"{c['days_aircraft_and_ships_equal']} 日（{c['days_aircraft_and_ships_rate']}）",
             "日付のずれ（tracker の日付+n と国防部の報告日）：" + "、".join(
                 f"n={n} {v['equal']}/{v['pairs']}={v['rate']}" for n, v in c["offsets"].items())]
    for t, f in c["fields"].items():
        lines.append(f"{f['tracker_field']}↔{f['mnd_field']}：一致 {f['equal']}/{f['comparable']}={f['rate']}、平均絶対差 {f['mean_abs_diff']}、"
                     f"tracker のみ {f['tracker_only']}・国防部のみ {f['mnd_only']}")
        lines.append("  差の大きい例：" + "; ".join(f"{e['date']} {e['tracker']}/{e['mnd']}" for e in f["examples_largest"][:8]))
        lines.append("  最近の例：" + "; ".join(f"{e['date']} {e['tracker']}/{e['mnd']}" for e in f["examples_latest"][-6:]))
    return "\n".join(lines)


# ---------------------------------------------------------------- writing

def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def dump(v) -> str:
    return json.dumps(v, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def write_all(out: Path, rows: dict[str, dict], status: dict, comparison: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for d in days:
        w.writerow({c: _cell(rows[d].get(c)) for c in FIELDS})
    write_atomic(out / "daily.csv", buf.getvalue())
    write_atomic(out / "daily.json", dump({
        "attribution": ATTRIBUTION, "fields": FIELDS, "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "rows": [[rows[d].get(c) for c in FIELDS] for d in days]}))
    write_atomic(out / "compare.json", dump(comparison))
    write_atomic(out / "meta.json", dump(META))
    write_atomic(out / "status.json", dump({**status, "days": len(days), "first_day": days[0] if days else None,
                                            "last_day": days[-1] if days else None}))


def _same_utc_day(a: str | None, b: datetime) -> bool:
    return bool(a) and str(a)[:10] == b.strftime("%Y-%m-%d")


def last_request(*dirs: Path) -> str | None:
    """The latest ``last_request_utc`` in the status.json of these directories."""
    seen = []
    for d in dirs:
        try:
            v = json.loads((d / "status.json").read_text(encoding="utf-8")).get("last_request_utc")
        except (OSError, ValueError):
            continue
        if v:
            seen.append(str(v))
    return max(seen) if seen else None


def fetch_csv(last_request_utc: str | None = None, trace: dict | None = None) -> tuple[str, dict, polite.RobotsResult]:
    """One GET of the CSV, at most once per UTC day. ``trace["requested_utc"]`` is set once the request is sent."""
    if _same_utc_day(last_request_utc, now_utc()):
        raise AlreadyToday(f"the CSV was already requested today (UTC) at {last_request_utc}; one request per day")
    robots = polite.require_allowed(CSV_URL, UA, source=SOURCE_ID)
    if trace is not None:
        trace["requested_utc"] = iso(now_utc())
    try:
        resp = polite.get(CSV_URL, UA, TIMEOUT, source=SOURCE_ID)
    except polite.SourceBlocked:
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
        raise Stop(f"{CSV_URL}: {type(e).__name__}: {e}") from None
    if resp.status != 200:
        raise Stop(f"{CSV_URL}: HTTP {resp.status}")
    text = resp.body.decode("utf-8", errors="replace")
    if text.lstrip()[:1] == "<":
        raise Stop(f"{CSV_URL}: an HTML page was returned instead of the CSV")
    info = {"url": CSV_URL, "http": resp.status, "bytes": len(resp.body),
            "sha256": hashlib.sha256(resp.body).hexdigest(),
            "content_type": resp.headers.get("content-type"),
            "last_modified": resp.headers.get("last-modified"), "etag": resp.headers.get("etag")}
    return text, info, robots


def about_text() -> list[str]:
    """Probe only: the about page once, as plain text in annotation-sized parts."""
    resp = polite.get(ABOUT_URL, UA, TIMEOUT)
    body = resp.body.decode("utf-8", errors="replace")
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    text = f"HTTP {resp.status}, {len(resp.body)} bytes\n" + text
    return [text[i:i + 3800] for i in range(0, min(len(text), 5 * 3800), 3800)]


def run(out: Path, from_file: Path | None = None, mnd_csv: Path | None = None) -> int:
    started = now_utc()
    status_path = out / "status.json"
    old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    stored = read_stored(out / "daily.csv")
    trace: dict = {}
    access: dict = {}
    try:
        if from_file:
            text = from_file.read_text(encoding="utf-8")
            info = {"url": CSV_URL, "from_file": str(from_file),
                    "sha256": hashlib.sha256(from_file.read_bytes()).hexdigest()}
            robots = "not checked (rebuilt from a saved file)"
        else:
            text, info, rr = fetch_csv(last_request(out, OUT), trace)
            robots = rr.reason
            access = {"access": rr.access, **({"permission": rr.permission} if rr.permission else {})}
            if rr.permission:
                notice(f"robots.txt は拒否、制作者の明示の許可で取得した（{rr.reason}）", "warning", title="pla-tracker permission")
        fresh, skipped = parse_csv(text)
        if not fresh:
            raise Stop("no rows")
        retrieved = iso(now_utc())
        rows, changes = merge(stored, fresh, retrieved)
    except AlreadyToday as e:
        notice(f"PLA Activity Tracker は今日は取得済み：{e}")
        return 0
    except (Stop, polite.SourceBlocked) as e:
        notice(f"PLA Activity Tracker を使わない：{e}", "warning")
        last_run = {"started_utc": iso(started), "finished_utc": iso(now_utc()), "result": "skipped", "reason": str(e),
                    **access}
        if stored:
            write_atomic(status_path, dump({**old_status, "last_run": last_run,
                                            **({"last_request_utc": trace["requested_utc"]} if trace else {})}))
        return 1 if isinstance(e, polite.Challenge) else 0
    comparison = compare(rows, read_mnd(mnd_csv or MND / "daily.csv"))
    src_days = sorted(fresh)
    status = {
        "generated_utc": iso(now_utc()),
        "last_success_utc": iso(now_utc()),
        **({"last_request_utc": trace["requested_utc"]} if trace
           else {"last_request_utc": old_status["last_request_utc"]} if old_status.get("last_request_utc") else {}),
        "source": {**info, "rows": len(fresh), "first_day": src_days[0], "last_day": src_days[-1],
                   "skipped_rows": skipped[:30], "skipped_count": len(skipped)},
        "last_run": {"started_utc": iso(started), "finished_utc": iso(now_utc()), "result": "ok",
                     "robots": robots, **access, **changes},
    }
    write_all(out, rows, status, comparison)
    notice(f"取得 {len(fresh)} 行（{src_days[0]}〜{src_days[-1]}）、追加 {changes['added']}・改訂 {changes['revised_count']}・"
           f"元に無く保持 {changes['kept_not_in_source_count']}・読めない行 {len(skipped)}"
           + (f"\n読めない行の例：{skipped[:5]}" if skipped else ""), title="pla-tracker summary")
    notice(compare_text(comparison), title="pla-tracker vs taiwan-mnd")
    return 0


META = {
    "dataset": "pla-tracker",
    "title": "PLA Activity Tracker：台湾周辺の中国軍機・艦艇の日ごとの数（台湾国防部の日報の集約）",
    "attribution": ATTRIBUTION,
    "source": {
        "name": "PLA Activity Tracker（個人による集約サイト）",
        "urls": [SITE_URL, CSV_URL, ABOUT_URL],
        "original": "中華民國國防部の日報「中共解放軍臺海周邊海、空域動態」（https://www.mnd.gov.tw/）",
        "position": "原典は紛争当事者（台湾）の公式発表。集約者は個人で、国防部のサイトを自動で読み取って表にしている",
        "reliability": "B",
        "credibility": "2",
        "compiler": "Adam Pan（Skyfaring、about ページの記名）",
        "rating_note": "暫定評価 B-2 目安（アドミラルティ・コード）。数値の原典は当事者である台湾国防部の発表で、探知範囲・計上基準・公表方針は国防部に依存する。"
                       "集約者は個人であり、読み取りの誤り・修正の有無は国防部の原文で確かめる必要がある。",
    },
    "access": "取得の前に robots.txt を確かめる（collectors/polite.py）。2026-10-31 10:22 UTC に収集器の User-Agent で読んだ robots.txt は "
              "User-agent: * に Allow: /（全体を許可）。制作者の許可（2026-10-31、運営者経由。私信のため非公開）を "
              "collectors/permissions.json に記録しており、robots.txt が CSV を拒むようになった場合も、許可の見直し日までは CSV 1 本だけを "
              "1 日 1 回以内・再試行なしで取得する。どちらで取得したかは status.json の last_run.access（robots / permission）に書く",
    "use": "台湾国防部の日報の自動取得は robots.txt（User-agent: * に Disallow: /）で拒否されているため 2026-10-30 で止めた"
           "（collectors/taiwan_mnd.py SUSPENDED）。2026-10-31 以降の日ごとの数はこのデータで補う。",
    "precedence": "同じ日に国防部の日報（docs/data/taiwan-mnd/、around-taiwan 書式）とこのデータの両方がある場合は、原典に近い国防部の日報の値を使う。"
                  "このデータは国防部の日報の行が無い日（主に 2026-10-31 以降）だけに使い、使った日は出典をこのデータとして区別する。"
                  "つなぐときは aircraft_total→aircraft はそのまま、ships_total は海軍艦艇＋公務船（plan_vessels とは別の量）、"
                  "median_line_cross は median_line_or_adiz より狭い量として扱い、同じ列として続けない。食い違いは compare.json に毎回記録する。",
    "window": "date は国防部の日報の報告日と同じとみなす（台湾時間。前日 06:00〜当日 06:00 の集計）。集約者は国防部の公表（台湾時間の正午〜14 時ごろ）の後に自動で取り込む（about ページ）。compare.json の offsets で日付のずれを確かめている",
    "fields": {
        "date": "日付（国防部の報告日、台湾時間とみなす）。2026-03-05〜03-19 と 03-31〜04-02 には、国防部の報告日より 1 日早い日付で載っているとみられる行がある（compare.json）",
        "aircraft_total": "中国軍機の延べ機数（国防部 aircraft に相当。定義は同じ）",
        "median_line_cross": "中間線を越えた機数。国防部 median_line_or_adiz（中間線越え＋北部・中部・西南・東部空域への進入）より狭く、"
                             "空域への進入だけの日は 0 になることが多い（例：2026-10-27 は 0、国防部 3「進入北部及西南空域(3架次)」）。"
                             "ただし進入の機数を入れた日もあり一定しない（2026-10-29 は 5「進入北部、中部及西南空域(5架次)」）。"
                             "国防部の原文に記載のない日も 0 になっている（国防部の側では null）",
        "cross_rate": "median_line_cross / aircraft_total の百分率（集約者の計算、文字列のまま。0.0〜100）",
        "aircraft_type": "機の種類の区分（Manned・Zero・Mixed・UAV・Helicopter。大文字小文字の揺れは原文のまま）。機種名ではない",
        "ships_total": "艦艇の数。2026-01〜02 は中国海軍艦艇だけ（国防部 plan_vessels と同じ）、2026-03 下旬（03-23 ごろ）以降は"
                       "海軍艦艇＋公務船（国防部 plan_vessels + official_ships）。集約者の定義変更とみられ、内訳は分からない",
        "activity_start": "活動の始まり（台湾時間の時刻、集約者の表記のまま。記載はまれ）",
        "activity_end": "活動の終わり（同上）",
        "special_event": "国防部の原文（中国語）の一部の写し：空域・中間線越えの記述、気球など（集約者の表記のまま）",
        "retrieved_utc": "その行を取り込んだ（または値が変わった）時刻（UTC）。公表時刻ではない",
    },
    "comparison_2026": "国防部の日報（around-taiwan 書式）と重なる 2026-01-01〜2026-10-30 の 299 日（2026-10-30 取得の CSV）："
                       "機数 285/299 一致（95.3%、食い違い 14 日は 2026-02-01・03-05〜03-19・03-31〜04-02・08-05 で、多くは 1 日のずれ）、"
                       "艦艇は国防部の海軍艦艇＋公務船と 261/299 一致（87.3%。2026-05 以降は 147/149）、海軍艦艇だけとは 92/299、"
                       "中間線越えは国防部に値のある 221 日で 144 一致（65.2%、定義が狭いため）。毎回の結果は compare.json",
    "missing_values": "空欄は不明（null）で、0 とはみなさない。集約者が空欄にした理由は分からない",
    "merge": "日付ごとに取り込み、集約者が値を変えた日は新しい値で置き換えて status.json の last_run.revised に記録する。"
             "元の CSV から消えた日は残す。保存済みの 80% より行が少ない CSV、列の足りない CSV は使わない",
    "limits": [
        "集約者が国防部のサイトを自動で取得して作った表であり、国防部自身の robots.txt は自動取得を拒んでいる。集約者の取得方法・許諾の有無は確認していない",
        "集約者は個人で、更新が止まる・形式が変わる・公開をやめる可能性がある。status.json の source.last_day で最新日を確かめる",
        "集約の過程での読み取り誤り・転記の遅れがありうる。国防部の日報との突き合わせ（compare.json）で一致率を確かめ、食い違いは国防部の原文を優先する",
        "国防部の日報の書式・定義の変更（taiwan-mnd/meta.json の definition_changes）は、このデータでは記録されない",
        "about ページは「vessels は PLA Navy の艦艇数」と説明しているが、実際の ships_total は 2026-03 下旬から公務船を含む（compare.json）。集約者の説明と値が合っていない",
        "about ページは「記録は追記のみで書き換えない」としている。過去の日付のずれ（2026-03 の 1 日ずれ等）も直されない可能性が高い",
        "収録は 2026-01-01 から（それより前は無い）",
    ],
    "files": {
        "daily.csv": "日ごとの値（元の列＋取り込み時刻）",
        "daily.json": "同じ内容の JSON（出典表記つき）",
        "compare.json": "国防部の日報（around-taiwan 書式）との突き合わせ：重なる期間の項目ごとの一致率、日付のずれ、食い違いの例",
        "status.json": "最後の取得の結果（HTTP、行数、最新日、追加・改訂、robots.txt の判定）",
    },
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--from-file", type=Path)
    ap.add_argument("--about", action="store_true", help="probe: also read the about page once")
    ap.add_argument("--save-raw", type=Path, help="probe: keep the fetched CSV here")
    a = ap.parse_args(argv)
    if a.save_raw and not a.from_file:
        try:
            text, info, rr = fetch_csv(last_request(a.out, OUT))
        except AlreadyToday as e:
            notice(f"probe: 今日は取得済みのため CSV を取らない：{e}")
            return 0
        a.save_raw.write_text(text, encoding="utf-8")
        notice(f"{rr.reason}（access: {rr.access}）\n{json.dumps(info, ensure_ascii=False)}", title="pla-tracker fetch")
        rc = run(a.out, from_file=a.save_raw)
    else:
        rc = run(a.out, from_file=a.from_file)
    if a.about:
        try:
            parts = about_text()
            for i, part in enumerate(parts, 1):
                notice(part, title=f"pla-tracker about {i}/{len(parts)}")
        except Exception as e:  # noqa: BLE001 - probe output only
            notice(f"about: {type(e).__name__}: {e}", "warning", title="pla-tracker about")
    return rc


if __name__ == "__main__":
    sys.exit(main())
