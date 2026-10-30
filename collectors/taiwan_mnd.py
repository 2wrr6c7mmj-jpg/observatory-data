"""Collect Taiwan MND daily PLA activity reports into docs/data/taiwan-mnd/.

Runs in GitHub Actions of this public repository (no Cloudflare usage). The
result is served by GitHub Pages and read by the Observatory site.

  python -m collectors.taiwan_mnd                 # daily: newest list pages
  python -m collectors.taiwan_mnd --pages 300     # backfill, resumable
  python -m collectors.taiwan_mnd --probe         # print page structure only

Politeness: one request at a time, a pause between requests, an identifying
User-Agent, and reports already stored are never fetched again.
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

from collectors.mnd_parse import PARSER_VERSION, page_text, parse_report, reparse, report_links

BASE = "https://www.mnd.gov.tw"
LIST_URLS = [  # tried in order until one yields report links
    BASE + "/en/news/plaactlist/{page}",
    BASE + "/news/plaactlist/{page}",
]
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "taiwan-mnd"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data; public research)"
PAUSE = float(os.environ.get("MND_PAUSE_SECONDS", "2.5"))
FETCH_TIMEOUT = 30  # seconds per connect / read
FIELDS = ["report_date", "format", "aircraft", "median_line_or_adiz", "plan_vessels", "official_ships",
          "balloons", "adiz_aircraft", "adiz_types", "window_start_utc", "window_end_utc",
          "source_url", "retrieved_utc", "parser_version"]


def daily_rows(reports: dict) -> list[dict]:
    """One row per report date.

    around-taiwan: the parsed report with the highest id wins (a later
    correction on the same day replaces the earlier figures).
    sw-adiz (until late 2022): daytime and nighttime reports of the same day are
    added together; for the same day and period the highest id wins.
    """
    latest: dict[tuple, dict] = {}
    for rid in sorted(reports, key=int):
        r = reports[rid]
        if r.get("parsed") and r.get("report_date"):
            key = (r["report_date"], r.get("format", "around-taiwan"),
                   r.get("period") if r.get("format") == "sw-adiz" else None)
            latest[key] = r
    days: dict[str, dict] = {}
    for (day, fmt, _), r in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        if fmt != "sw-adiz":
            days[day] = {**r, "format": fmt}
            continue
        row = days.get(day)
        if row is None or row.get("format") != "sw-adiz":
            if row is not None:  # an around-taiwan report for the same day takes precedence
                continue
            row = days[day] = {**r, "adiz_types": dict(r.get("adiz_types") or {}), "sources": [r["source_url"]]}
            continue
        row["adiz_aircraft"] = (row.get("adiz_aircraft") or 0) + (r.get("adiz_aircraft") or 0)
        for k, v in (r.get("adiz_types") or {}).items():
            row["adiz_types"][k] = row["adiz_types"].get(k, 0) + v
        row["sources"].append(r["source_url"])
        row["source_url"] = " ".join(row["sources"])
    out = []
    for day in sorted(days):
        row = dict(days[day])
        if isinstance(row.get("adiz_types"), dict):
            row["adiz_types"] = ";".join(f"{k}:{v}" for k, v in sorted(row["adiz_types"].items())) or None
        out.append(row)
    return out


def fetch(url: str) -> str:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="list pages to walk (10 reports each)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-ids", default="", help="comma-separated report ids to print")
    a = ap.parse_args()
    if a.probe_ids:
        return probe_ids([x for x in a.probe_ids.split(",") if x.strip()])
    if a.probe:
        return probe()
    # The daily run stops at the first page with nothing new; a backfill walks all pages.
    return collect(a.pages, stop_when_known=a.pages <= 3)


if __name__ == "__main__":
    sys.exit(main())
