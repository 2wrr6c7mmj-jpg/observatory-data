"""Collect Taiwan MND daily PLA activity reports into docs/data/taiwan-mnd/.

Runs in GitHub Actions of this public repository (no Cloudflare usage). The
result is served by GitHub Pages and read by the Observatory site.

  python -m collectors.taiwan_mnd                 # daily: newest list pages
  python -m collectors.taiwan_mnd --pages 300     # backfill, resumable
  python -m collectors.taiwan_mnd --probe         # print page structure only
  python -m collectors.taiwan_mnd --check-robots  # read robots.txt once and report
  python -m collectors.taiwan_mnd --reparse       # offline: re-read stored text

Politeness: robots.txt is checked (collectors/polite.py) before any list or
report page is fetched; a disallowed or unverifiable answer means nothing is
fetched and the stored data is kept. One request at a time, a pause between
requests, an identifying User-Agent, and reports already stored are never
fetched again.

SUSPENDED: collection is stopped while this is set (see the note there).
"""
from __future__ import annotations

import argparse
import http.client
import re
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin
from datetime import datetime, timezone
from pathlib import Path

from collectors import polite
from collectors.mnd_parse import PARSER_VERSION, TYPE_FORMATS, page_text, parse_report, reparse, report_links

BASE = "https://www.mnd.gov.tw"
LIST_URLS = [  # tried in order until one yields report links
    BASE + "/en/news/plaactlist/{page}",
    BASE + "/news/plaactlist/{page}",
]
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "taiwan-mnd"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data; public research)"
PAUSE = float(os.environ.get("MND_PAUSE_SECONDS", "2.5"))
FETCH_TIMEOUT = 30  # seconds per connect / read
# Collection is stopped: on 2026-10-30 https://www.mnd.gov.tw/robots.txt (Last-Modified
# 2026-10-13) allowed only Googlebot (a few paths disallowed) and had
# "User-agent: *" / "Disallow: /", which covers our User-Agent and every page we read.
# While this is set, collect / probe make no request at all (--reparse still works
# offline, --check-robots reads robots.txt once). Clear it only after robots.txt allows
# /en/news/plaactlist/ and /en/News/PLAAct/ for our User-Agent; the robots check
# below stays in force either way.
SUSPENDED = ("robots.txt（2026-10-30 確認）が User-agent: * に Disallow: / を指定しているため、"
             "台湾国防部の取得を停止中（保存済みのデータはそのまま）")
FIELDS = ["report_date", "format", "aircraft", "median_line_or_adiz", "plan_vessels", "official_ships",
          "balloons", "adiz_aircraft", "adiz_types", "window_start_utc", "window_end_utc",
          "source_url", "retrieved_utc", "parser_version", "reports", "merge_note"]


def daily_rows(reports: dict) -> list[dict]:
    """One row per report date.

    around-taiwan: the parsed report with the highest id wins (a later
    correction on the same day replaces the earlier figures).
    sw-adiz / afc-around-roc (until 2022-11-12): every report is one ADIZ entry
    event, so the reports of one day (morning, afternoon, night, or several
    without a time of day) are added together. A report with the same time of
    day and the same aircraft as an earlier one of that day is a re-post and is
    not counted twice; one with the same stated time of day but different
    aircraft is taken as a correction of the earlier one (noted in merge_note).
    """
    by_day: dict[tuple, list[tuple[int, dict]]] = {}
    for rid in sorted(reports, key=int):
        r = reports[rid]
        if r.get("parsed") and r.get("report_date"):
            by_day.setdefault((r["report_date"], r.get("format", "around-taiwan")), []).append((int(rid), r))
    days: dict[str, dict] = {}
    for (day, fmt), group in sorted(by_day.items()):
        if fmt not in TYPE_FORMATS:
            days[day] = {**group[-1][1], "format": fmt}
            continue
        if day in days and days[day].get("format") not in TYPE_FORMATS:
            continue  # an around-taiwan report for the same day takes precedence
        events: list[dict] = []
        notes: list[str] = []
        for rid, r in group:
            types = dict(r.get("adiz_types") or {})
            per = r.get("period")
            same = next((e for e in events if e["period"] == per and e["types"] == types), None)
            if same is not None:
                notes.append(f"{rid}: re-post of {same['id']} (same time of day and aircraft), counted once")
                same.update(id=rid, url=r["source_url"])
                continue
            fix = next((e for e in events if per is not None and e["period"] == per), None)
            if fix is not None:
                notes.append(f"{rid}: same time of day ({per}) as {fix['id']} with different aircraft; "
                             f"taken as a correction (unverified)")
                fix.update(id=rid, url=r["source_url"], types=types, n=r.get("adiz_aircraft") or 0)
                continue
            events.append({"id": rid, "url": r["source_url"], "period": per, "types": types,
                           "n": r.get("adiz_aircraft") or 0})
        total: dict[str, int] = {}
        for e in events:
            for k, v in e["types"].items():
                total[k] = total.get(k, 0) + v
        last = group[-1][1]
        days[day] = {**last, "format": fmt, "adiz_aircraft": sum(e["n"] for e in events), "adiz_types": total,
                     "source_url": " ".join(e["url"] for e in events), "reports": len(events),
                     "merge_note": "; ".join(notes) or None}
    out = []
    for day in sorted(days):
        row = dict(days[day])
        if isinstance(row.get("adiz_types"), dict):
            row["adiz_types"] = ";".join(f"{k}:{v}" for k, v in sorted(row["adiz_types"].items())) or None
        out.append(row)
    return out


def require_robots(url: str) -> polite.RobotsResult:
    """Raise polite.SourceBlocked unless robots.txt allows ``url`` for our User-Agent."""
    return polite.require_allowed(url, UA)


def fetch(url: str) -> str:
    require_robots(url)  # robots.txt is read once per run (cached), before any page
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,zh-TW;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
            body = r.read(3_000_000)
            charset = r.headers.get_content_charset() or "utf-8"
    finally:
        time.sleep(PAUSE)  # keep the pace after a failed request too
    return body.decode(charset, errors="replace")


def notice(message: str) -> None:
    """Visible as an annotation on the workflow run (readable through the API)."""
    print("::notice::" + message.replace("%", "%25").replace("\n", "%0A")[:4000])


def warning(message: str) -> None:
    """Like notice(), but shown as a warning so a cut-short run stands out."""
    print("::warning::" + message.replace("%", "%25").replace("\n", "%0A")[:4000])


def load() -> dict:
    path = OUT / "reports.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"reports": {}}


def save(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = store["reports"]
    (OUT / "reports.json").write_text(json.dumps(store, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    rows = daily_rows(reports)
    with open(OUT / "daily.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "daily.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fields": FIELDS,
        "rows": [[r.get(k) for k in FIELDS] for r in rows],
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    unparsed = sorted((rid for rid, r in reports.items() if not r.get("parsed")), key=int)
    (OUT / "status.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reports": len(reports), "days": len(rows),
        "first_day": rows[0]["report_date"] if rows else None,
        "last_day": rows[-1]["report_date"] if rows else None,
        "unparsed_ids": unparsed, "parser_version": PARSER_VERSION,
        "parse_errors": {rid: reports[rid]["parse_error"] for rid in unparsed if reports[rid].get("parse_error")},
        "collection": SUSPENDED or "active",
        "days_by_format": {f: sum(1 for r in rows if r.get("format") == f) for f in sorted({r.get("format") for r in rows})},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def page_base(raw_html: str, page_url: str) -> str:
    """URL that relative links on the page resolve against (honours <base href>)."""
    m = re.search(r"""<base[^>]*href\s*=\s*["']([^"']+)["']""", raw_html, re.I)
    return urljoin(page_url, m[1]) if m else page_url


LIST_RETRY_WAITS = (30, 120)  # seconds before the 2nd and 3rd attempt
RETRY_AFTER_CAP = 300


class ListPageUnavailable(Exception):
    """A list page kept failing with a temporary error; end this run, resume next time."""


def transient(e: Exception) -> bool:
    """Temporary failure worth retrying later.

    A read timeout or a dropped connection while waiting for the response is
    raised by urllib as a bare TimeoutError / ConnectionError / HTTPException
    (only errors while sending are wrapped in URLError), so all are listed.
    """
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException))


def retry_wait(e: Exception, default: float) -> float:
    after = getattr(e, "headers", None) and e.headers.get("Retry-After")
    if after and str(after).strip().isdigit():
        # never wait less than our own pace, never longer than the cap
        return max(min(float(after), RETRY_AFTER_CAP), default)
    return default


def fetch_list(url: str) -> str:
    """fetch() with a few slow retries on 429 / 5xx / network errors."""
    for attempt in range(len(LIST_RETRY_WAITS) + 1):
        try:
            return fetch(url)
        except Exception as e:  # noqa: BLE001 - classified below
            if not transient(e):
                raise
            if attempt == len(LIST_RETRY_WAITS):
                raise ListPageUnavailable(f"{url}: {e}") from e
            wait = retry_wait(e, LIST_RETRY_WAITS[attempt])
            notice(f"{url}: {e}; retrying in {wait:.0f}s")
            _sleep(wait)
    raise AssertionError("unreachable")


_sleep = time.sleep

REPORT_RETRY_WAIT = 30  # seconds before the one retry of a report page
REPORT_FAILURE_LIMIT = 3  # consecutive reports given up on before the run ends


def fetch_report(url: str) -> str:
    """fetch() with one slow retry on a temporary error (raised again if it persists)."""
    try:
        return fetch(url)
    except Exception as e:  # noqa: BLE001 - classified below
        if not transient(e):
            raise
        notice(f"{url}: {e}; retrying in {REPORT_RETRY_WAIT}s")
        _sleep(retry_wait(e, REPORT_RETRY_WAIT))
    return fetch(url)


def list_page(page: int) -> list[tuple[str, int]]:
    """Absolute report URLs and ids on one list page."""
    for template in LIST_URLS:
        url = template.format(page=page)
        try:
            raw = fetch_list(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        base = page_base(raw, url)
        links = [(urljoin(base, href), rid) for href, rid in report_links(raw)]
        if links:
            return links
    return []


def collect(pages: int, stop_when_known: bool) -> int:
    robots = polite.check_robots(LIST_URLS[0].format(page=1), UA)
    if not robots.allowed:  # nothing fetched, nothing rewritten
        warning(f"taiwan-mnd: not collecting ({robots.reason})")
        return 0 if robots.verified else 75
    store = load()
    reports = store["reports"]
    aborted = False
    count = {"added": 0}
    try:
        aborted = _walk(store, pages, stop_when_known, count)
    finally:
        save(store)  # whatever was fetched is kept, even on an unexpected error
    bad = [rid for rid, r in reports.items() if not r.get("parsed")]
    notice(f"taiwan-mnd: {count['added']} fetched, {len(reports)} stored, {len(bad)} unparsed")
    # Non-zero marks the run as cut short; the publish step (always()) still
    # commits what was saved. 75 = EX_TEMPFAIL.
    return 75 if aborted else 0


def _walk(store: dict, pages: int, stop_when_known: bool, count: dict) -> bool:
    """Walk the list pages and fetch new reports; True if the run was cut short."""
    reports = store["reports"]
    failures = 0  # consecutive report pages given up on
    for page in range(1, pages + 1):
        try:
            links = list_page(page)
        except polite.SourceBlocked as e:
            warning(f"list page {page}: not fetched ({e}); ending this run")
            return True
        except ListPageUnavailable as e:
            warning(f"list page {page}: still failing after retries ({e}); ending this run, next run resumes")
            return True
        if not links:
            notice(f"list page {page}: no report links; stopping")
            break
        for _, rid in links:  # a newer parser re-reads kept text without fetching again
            r = reports.get(str(rid))
            if r and r.get("parser_version", 0) < PARSER_VERSION and r.get("main_text"):
                reports[str(rid)] = reparse(r)
        fresh = [(href, rid) for href, rid in links if str(rid) not in reports
                 or reports[str(rid)].get("parser_version", 0) < PARSER_VERSION]
        for href, rid in fresh:
            url = href
            try:
                raw = fetch_report(url)
            except polite.SourceBlocked as e:
                warning(f"report {rid}: not fetched ({e}); ending this run")
                return True
            except urllib.error.HTTPError as e:
                if transient(e):
                    failures += 1
                notice(f"report {rid}: fetch failed ({e}); will retry next run")
            except Exception as e:  # noqa: BLE001 - classified below
                if not transient(e):
                    raise
                failures += 1
                notice(f"report {rid}: fetch failed ({e}); will retry next run")
            else:
                failures = 0
                reports[str(rid)] = parse_report(raw, url, datetime.now(timezone.utc))
                count["added"] += 1
                continue
            if failures >= REPORT_FAILURE_LIMIT:
                warning(f"report pages: {failures} in a row failed; ending this run, next run resumes")
                return True
        if fresh:
            save(store)  # resumable: progress survives a timeout
        if stop_when_known and not fresh:
            break
    return False


def probe() -> int:
    """Print what the parser sees on the first list page and its newest report."""
    for template in LIST_URLS:
        url = template.format(page=1)
        try:
            raw = fetch(url)
        except Exception as e:  # noqa: BLE001 - diagnostics only
            notice(f"probe {url}: {e}")
            continue
        links = report_links(raw)
        hrefs = sorted(set(re.findall(r"""(?:href|data-href|data-url|action|src)\s*=\s*["']([^"']+)["']""", raw, re.I)))
        hints = sorted(set(re.findall(r"""["']([^"'\s]*(?:api|ajax|json|plaact|Publication|news)[^"'\s]*)["']""", raw, re.I)))
        base = re.findall(r"<base[^>]*>", raw, re.I)
        pages = [h for h in hrefs if re.search(r"plaactlist|page", h, re.I)]
        notice(f"probe {url}: {len(raw)} bytes, {len(links)} report links, first {links[:3]}, base {base}, pagination {pages[:40]}")
        notice("probe hrefs:\n" + "\n".join(hrefs[:150]))
        notice("probe hints:\n" + "\n".join(hints[:80]))
        body = re.sub(r"\s+", " ", page_text(raw))
        i = body.find("PLA")
        notice("probe text around first 'PLA':\n" + body[max(0, i - 200):i + 1500])
        if links:
            href, rid = links[0]
            href = urljoin(page_base(raw, url), href)
            report = fetch(href)
            notice(f"probe report {rid}:\n" + page_text(report)[:2500])
            notice("probe parsed: " + json.dumps(parse_report(report, href, datetime.now(timezone.utc)), ensure_ascii=False))
            return 0
    return 1


def probe_ids(ids: list[str]) -> int:
    """Print the main text of specific reports (for adapting the parser)."""
    for rid in ids:
        url = f"{BASE}/en/News/PLAAct/{rid}"
        raw = fetch(url)
        body = re.sub(r"\s+", " ", page_text(raw))
        i = body.find("PLA Activities 20")
        imgs = sorted(set(re.findall(r"""(?:src|href)\s*=\s*["']([^"']+\.(?:jpe?g|png|pdf))["']""", raw, re.I)))[:10]
        notice(f"report {rid}: images {imgs}\n" + body[max(0, i):i + 1800])
    return 0


def check_robots() -> int:
    """Read robots.txt once and report whether the pages we read are allowed (no page fetched)."""
    lines = []
    for url in (LIST_URLS[0].format(page=1), LIST_URLS[1].format(page=1), BASE + "/en/News/PLAAct/1"):
        r = polite.check_robots(url, UA)
        lines.append(f"{url}: {'allowed' if r.allowed else 'not allowed'} ({r.reason})")
    (warning if any("not allowed" in x for x in lines) else notice)("taiwan-mnd robots.txt:\n" + "\n".join(lines))
    return 0


def mark_suspended() -> None:
    """Record the suspension in status.json once (no rewrite while it is unchanged)."""
    path = OUT / "status.json"
    if not path.exists():
        return
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("collection") != SUSPENDED:
        status["collection"] = SUSPENDED
        path.write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def reparse_all() -> int:
    """Re-read every stored report from its kept text (no network) and rewrite the outputs."""
    store = load()
    for rid, r in store["reports"].items():
        if r.get("main_text"):
            store["reports"][rid] = reparse(r)
    save(store)
    notice(f"taiwan-mnd: {len(store['reports'])} reports re-read with parser {PARSER_VERSION}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="list pages to walk (10 reports each)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--reparse", action="store_true", help="re-read stored reports offline and rewrite the outputs")
    ap.add_argument("--probe-ids", default="", help="comma-separated report ids to print")
    ap.add_argument("--check-robots", action="store_true", help="read robots.txt once and report")
    a = ap.parse_args()
    if a.check_robots:
        return check_robots()
    if a.reparse:
        return reparse_all()
    if SUSPENDED:
        warning("taiwan-mnd: " + SUSPENDED)
        mark_suspended()
        return 0
    if a.probe_ids:
        return probe_ids([x for x in a.probe_ids.split(",") if x.strip()])
    if a.probe:
        return probe()
    # The daily run stops at the first page with nothing new; a backfill walks all pages.
    return collect(a.pages, stop_when_known=a.pages <= 3)


if __name__ == "__main__":
    sys.exit(main())
