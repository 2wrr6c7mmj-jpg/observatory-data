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
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from collectors.mnd_parse import PARSER_VERSION, page_text, parse_report, report_links

BASE = "https://www.mnd.gov.tw"
LIST_URLS = [  # tried in order until one yields report links
    BASE + "/en/tag/pla%20activities/{page}",
    BASE + "/tag/pla%20activities/{page}",
]
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "taiwan-mnd"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data; public research)"
PAUSE = float(os.environ.get("MND_PAUSE_SECONDS", "2.5"))
FIELDS = ["report_date", "aircraft", "median_line_or_adiz", "plan_vessels", "official_ships",
          "balloons", "window_start_utc", "window_end_utc", "source_url", "retrieved_utc", "parser_version"]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,zh-TW;q=0.8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read(3_000_000)
        charset = r.headers.get_content_charset() or "utf-8"
    time.sleep(PAUSE)
    return body.decode(charset, errors="replace")


def notice(message: str) -> None:
    """Visible as an annotation on the workflow run (readable through the API)."""
    print("::notice::" + message.replace("%", "%25").replace("\n", "%0A")[:4000])


def load() -> dict:
    path = OUT / "reports.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"reports": {}}


def save(store: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = store["reports"]
    (OUT / "reports.json").write_text(json.dumps(store, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    # One row per report date: the parsed report with the highest id wins
    # (a later correction on the same day replaces the earlier figures).
    daily: dict[str, dict] = {}
    for rid in sorted(reports, key=int):
        r = reports[rid]
        if r.get("parsed") and r.get("report_date"):
            daily[r["report_date"]] = r
    rows = [daily[d] for d in sorted(daily)]
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


def list_page(page: int) -> list[tuple[str, int]]:
    for template in LIST_URLS:
        try:
            links = report_links(fetch(template.format(page=page)))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        if links:
            return links
    return []


def collect(pages: int, stop_when_known: bool) -> int:
    store = load()
    reports = store["reports"]
    added = 0
    for page in range(1, pages + 1):
        links = list_page(page)
        if not links:
            notice(f"list page {page}: no report links; stopping")
            break
        fresh = [(href, rid) for href, rid in links if str(rid) not in reports
                 or reports[str(rid)].get("parser_version", 0) < PARSER_VERSION]
        for href, rid in fresh:
            url = href if href.startswith("http") else BASE + href
            try:
                raw = fetch(url)
            except urllib.error.URLError as e:
                notice(f"report {rid}: fetch failed ({e}); will retry next run")
                continue
            reports[str(rid)] = parse_report(raw, url, datetime.now(timezone.utc))
            added += 1
        if fresh:
            save(store)  # resumable: progress survives a timeout
        if stop_when_known and not fresh:
            break
    save(store)
    bad = [rid for rid, r in reports.items() if not r.get("parsed")]
    notice(f"taiwan-mnd: {added} fetched, {len(reports)} stored, {len(bad)} unparsed")
    return 0


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
        notice(f"probe {url}: {len(raw)} bytes, {len(links)} report links, first {links[:3]}\n"
               + page_text(raw)[:1500])
        if links:
            href, rid = links[0]
            report = fetch(href if href.startswith("http") else BASE + href)
            notice(f"probe report {rid}:\n" + page_text(report)[:2500])
            notice("probe parsed: " + json.dumps(parse_report(report, href, datetime.now(timezone.utc)), ensure_ascii=False))
            return 0
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="list pages to walk (10 reports each)")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe()
    # The daily run stops at the first page with nothing new; a backfill walks all pages.
    return collect(a.pages, stop_when_known=a.pages <= 3)


if __name__ == "__main__":
    sys.exit(main())
