"""Collect the Japan Coast Guard Senkaku counts into docs/data/jcg-senkaku/.

Source: 海上保安庁「尖閣諸島周辺海域における中国海警局に所属する船舶等の動向と
我が国の対処」 (one PDF per month since 2012-09, linked from the page).

  python -m collectors.jcg_senkaku               # daily run
  python -m collectors.jcg_senkaku --probe DIR   # fetch the page and sample PDFs, print, keep nothing
  python -m collectors.jcg_senkaku --dump DIR --start 0 --end 6   # print stored probe samples

Politeness: robots.txt is checked first (collectors/polite.py) and the source
is skipped when it disallows us or cannot be checked; a bot-challenge page
stops the run. One request at a time with a pause, an identifying User-Agent,
the page once per run, and PDF links taken from the page (URLs are never
built). A month's PDF is fetched once; only the current month (and the
previous one until it is complete) is fetched again, with If-None-Match /
If-Modified-Since so an unchanged file is not downloaded.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import http.client
import io
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from collectors import polite
from collectors.jcg_parse import PARSER_VERSION, check, decode_html, lines, page_links, parse_words

PAGE_URL = "https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "jcg-senkaku"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data; public research)"
PAUSE = float(os.environ.get("JCG_PAUSE_SECONDS", "5"))
TIMEOUT = 60
JST = timezone(timedelta(hours=9))
FIELDS = ["date", "contiguous_zone", "territorial_sea", "subject", "check", "window_start_utc", "window_end_utc",
          "as_of", "source_url", "source_last_modified_utc", "retrieved_utc", "parser_version"]

# Repeated in daily.json so the data carries its credit wherever it is copied (same as meta.json).
ATTRIBUTION = {
    "credit": "出典：海上保安庁ホームページ（https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html）",
    "license": "公共データ利用規約 第1.0版（PDL1.0） https://www.digital.go.jp/resources/open_data/public_data_license_v1.0",
    "processing": "observatory-data（https://github.com/2wrr6c7mmj-jpg/observatory-data）が月ごとの PDF から数値を自動で抜き出し、"
                  "日ごとの表に加工した。加工したデータは海上保安庁が作成したものではなく、誤りがあれば加工者の責任による。",
    "note": "海上保安庁のロゴ等は使用しない。",
}

_sleep = time.sleep


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def notice(message: str) -> None:
    print("::notice::" + message.replace("%", "%25").replace("\n", "%0A")[:4000], flush=True)


def warning(message: str) -> None:
    print("::warning::" + message.replace("%", "%25").replace("\n", "%0A")[:4000], flush=True)


def get(url: str, headers: dict | None = None) -> polite.Response:
    try:
        return polite.get(url, UA, TIMEOUT, headers)
    finally:
        _sleep(PAUSE)


def pdf_words(data: bytes) -> list[list]:
    """[page, text, x0, x1, top, bottom] for every word (pdfplumber, imported lazily)."""
    import pdfplumber  # installed by the workflow; tests do not need it

    out = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pno, page in enumerate(pdf.pages):
            for w in page.extract_words():
                out.append([pno, w["text"], round(w["x0"], 1), round(w["x1"], 1),
                            round(w["top"], 1), round(w["bottom"], 1)])
    return out


# ---------------------------------------------------------------- store

def load() -> dict:
    path = OUT / "pdfs.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"pdfs": {}}


def month_key(rec: dict) -> str | None:
    y, m = rec.get("year"), rec.get("month")
    if y and m:
        return f"{y:04d}-{m:02d}"
    return (rec.get("parse") or {}).get("title_year_month")


def is_final(rec: dict) -> bool:
    key, as_of = month_key(rec), (rec.get("parse") or {}).get("as_of")
    if not key or not as_of:
        return False
    y, m = (int(x) for x in key.split("-"))
    return as_of >= date(y, m, calendar.monthrange(y, m)[1]).isoformat()


def needs_fetch(link: dict, rec: dict | None, today: date) -> bool:
    """New PDFs always; a stored one only while its month may still change."""
    if rec is None:
        return True
    if is_final(rec):
        return False
    key = month_key(rec) or month_key(link)
    if not key:
        return False
    y, m = (int(x) for x in key.split("-"))
    months_ago = (today.year - y) * 12 + today.month - m
    return months_ago <= 1


def last_day(rec: dict, today: date) -> int | None:
    """Last day of the month covered by the PDF (as-of date, else month end if past)."""
    key = month_key(rec)
    if not key:
        return None
    y, m = (int(x) for x in key.split("-"))
    end = calendar.monthrange(y, m)[1]
    as_of = (rec.get("parse") or {}).get("as_of")
    if as_of:
        a = date.fromisoformat(as_of)
        return end if (a.year, a.month) > (y, m) else min(a.day, end)
    if (y, m) < (today.year, today.month):
        return end
    return None


def newness(rec: dict) -> tuple:
    return ((rec.get("parse") or {}).get("as_of") or "", rec.get("last_modified_utc") or "")


def daily_rows(store: dict, today: date) -> tuple[list[dict], dict]:
    rows, months, issues = [], {}, {}
    for url, rec in store["pdfs"].items():
        key = month_key(rec)
        if not key:
            issues[url] = "year/month unknown"
            continue
        months.setdefault(key, []).append(rec)
    by_month = {}
    for key, recs in months.items():
        # Two PDFs for one month: the newer (later as-of date, then Last-Modified) is
        # published; if neither is known to be newer, neither is.
        recs.sort(key=newness, reverse=True)
        if len(recs) > 1 and newness(recs[0]) == newness(recs[1]):
            for r in recs:
                issues[r["url"]] = f"{len(recs)} PDFs for {key}, none known to be newer"
            continue
        for r in recs[1:]:
            issues[r["url"]] = f"older PDF for {key} (published {recs[0]['url']})"
        by_month[key] = recs[0]
    for key in sorted(by_month):
        rec = by_month[key]
        p = rec.get("parse") or {}
        if not p.get("parsed"):
            issues[rec["url"]] = "; ".join(p.get("problems") or ["not parsed"])
            continue
        end = last_day(rec, today)
        if end is None:
            issues[rec["url"]] = "as-of date unknown"
            continue
        result = check(p, end)
        if result == "mismatch":
            issues[rec["url"]] = f"rows do not match the monthly totals {p.get('totals')}"
            continue
        verified = result in ("match", "match_days")
        ts_verified = verified and "ts_days" in p["totals"]
        y, m = (int(x) for x in key.split("-"))
        first = min(int(d) for d in p["days"])
        if key != "2012-09" and first != 1:  # only the first month (2012-09) starts later, on the 14th
            issues.setdefault(rec["url"], f"table starts on day {first}, not 1")
            first = 1  # the days before are published as unknown (null)
        for d in range(first, end + 1):
            r = p["days"].get(str(d))
            if r is None:
                issues.setdefault(rec["url"], f"day {d} missing")
                cz = ts = None
            else:
                cz = r["cz"] if r["cz"] is not None else (0 if verified else None)
                ts = r["ts"] if r["ts"] is not None else (0 if ts_verified or cz == 0 else None)
            start = datetime(y, m, d, tzinfo=JST)
            rows.append({
                "date": f"{key}-{d:02d}", "contiguous_zone": cz, "territorial_sea": ts, "subject": p.get("subject"), "check": result,
                "window_start_utc": iso(start), "window_end_utc": iso(start + timedelta(days=1)),
                "as_of": p.get("as_of"), "source_url": rec["url"],
                "source_last_modified_utc": rec.get("last_modified_utc"),
                "retrieved_utc": rec.get("retrieved_utc"), "parser_version": p.get("parser_version"),
            })
    return rows, issues


def dump_store(store: dict) -> str:
    """JSON with one line per PDF (small diffs when one month changes)."""
    def one(v):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    head = [f" {one(k)}:{one(v)}" for k, v in sorted(store.items()) if k != "pdfs"]
    pdfs = [f"  {one(u)}:{one(r)}" for u, r in sorted(store["pdfs"].items())]
    body = ",\n".join(head + [' "pdfs":{\n' + ",\n".join(pdfs) + "\n }"])
    return "{\n" + body + "\n}\n"


def write_atomic(path: Path, text: str) -> None:
    """Write beside the file and rename, so a killed run never leaves half a file to commit."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(store: dict, run: dict, today: date) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT / "pdfs.json", dump_store(store))
    rows, issues = daily_rows(store, today)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    write_atomic(OUT / "daily.csv", buf.getvalue())
    generated = iso(now_utc())
    write_atomic(OUT / "daily.json", json.dumps({
        "generated_utc": generated, "attribution": ATTRIBUTION, "fields": FIELDS,
        "rows": [[r[k] for k in FIELDS] for r in rows],
    }, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_atomic(OUT / "status.json", json.dumps({
        "generated_utc": generated, "last_run": run,
        "pdfs": len(store["pdfs"]), "days": len(rows),
        "first_day": rows[0]["date"] if rows else None, "last_day": rows[-1]["date"] if rows else None,
        "page": store.get("page"), "not_published": issues, "parser_version": PARSER_VERSION,
    }, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def record(link: dict, resp: polite.Response, retrieved: datetime) -> dict:
    lm = resp.headers.get("last-modified")
    try:
        lm_dt = parsedate_to_datetime(lm) if lm else None
    except (TypeError, ValueError):
        lm_dt = None
    try:
        words, error = pdf_words(resp.body), None
    except Exception as e:  # noqa: BLE001 - a broken PDF is recorded, not fatal
        words, error = [], f"PDF could not be read: {type(e).__name__}: {e}"
    rec = {
        "url": link["url"], "label": link["label"], "year": link["year"], "month": link["month"],
        "sha256": hashlib.sha256(resp.body).hexdigest(), "bytes": len(resp.body),
        "etag": resp.headers.get("etag"), "last_modified": lm, "last_modified_utc": iso(lm_dt),
        "retrieved_utc": iso(retrieved), "words": words, "read_error": error,
    }
    return reparse(rec)


def reparse(rec: dict) -> dict:
    p = parse_words(rec.get("words") or [], rec.get("year"), rec.get("month"))
    p["parser_version"] = PARSER_VERSION
    if rec.get("read_error"):
        p["problems"].insert(0, rec["read_error"])
        p["parsed"] = False
    title = p.get("title_year_month")
    if title and rec.get("year") and rec.get("month") and title != f"{rec['year']:04d}-{rec['month']:02d}":
        p["problems"].append(f"title says {title}, page says {rec['year']}-{rec['month']:02d}")
        p["parsed"] = False
    return {**rec, "parse": p}


# ---------------------------------------------------------------- run

class Stop(Exception):
    """End this run (temporary failure); what was fetched so far is kept."""


def fetch_pdf(link: dict, rec: dict | None) -> polite.Response:
    headers = {}
    if rec and rec.get("etag"):
        headers["If-None-Match"] = rec["etag"]
    if rec and rec.get("last_modified"):
        headers["If-Modified-Since"] = rec["last_modified"]
    resp = get(link["url"], headers)
    if resp.status == 304:
        return resp
    if resp.status != 200:
        raise Stop(f"{link['url']}: HTTP {resp.status}")
    if not resp.body.startswith(b"%PDF"):
        raise Stop(f"{link['url']}: not a PDF ({resp.headers.get('content-type')})")
    return resp


def collect(max_fetch: int) -> int:
    store = load()
    today = now_utc().astimezone(JST).date()
    run = {"started_utc": iso(now_utc()), "result": "ok", "fetched": 0, "not_modified": 0}
    code = 0
    try:
        code = _collect(store, run, today, max_fetch)
    finally:
        run["finished_utc"] = iso(now_utc())
        save(store, run, today)
    notice(f"jcg-senkaku: {run['result']}; {run['fetched']} PDFs fetched, {run['not_modified']} unchanged, "
           f"{len(store['pdfs'])} stored")
    return code


def _collect(store: dict, run: dict, today: date, max_fetch: int) -> int:
    pdfs = store["pdfs"]
    for url, rec in list(pdfs.items()):  # a newer parser re-reads stored words without fetching
        if (rec.get("parse") or {}).get("parser_version", 0) < PARSER_VERSION:
            pdfs[url] = reparse(rec)
    try:
        robots = polite.check_robots(PAGE_URL, UA)
        run["robots"] = robots.reason
        if not robots.allowed:
            run["result"] = "skipped: robots.txt " + ("disallows" if robots.verified else "could not be checked")
            warning(f"jcg-senkaku skipped: {robots.reason}")
            return 0 if robots.verified else 75
        page = get(PAGE_URL)
        if page.status != 200:
            raise Stop(f"{PAGE_URL}: HTTP {page.status}")
        retrieved = now_utc()
        links = page_links(decode_html(page.body, page.headers.get("content-type", "")), PAGE_URL)
        if not links:
            raise Stop(f"{PAGE_URL}: no PDF links found (page layout changed?)")
        store["page"] = {"url": PAGE_URL, "sha256": hashlib.sha256(page.body).hexdigest(),
                         "retrieved_utc": iso(retrieved), "links": len(links)}
        todo = [l for l in reversed(links) if needs_fetch(l, pdfs.get(l["url"]), today)]  # newest first
        if len(todo) > max_fetch:
            notice(f"jcg-senkaku: {len(todo)} PDFs to fetch, {max_fetch} this run")
        for i, link in enumerate(todo[:max_fetch]):
            rec = pdfs.get(link["url"])
            resp = fetch_pdf(link, rec)
            if resp.status == 304:
                run["not_modified"] += 1
                continue
            pdfs[link["url"]] = record(link, resp, now_utc())
            run["fetched"] += 1
            if i % 10 == 9:
                save(store, run, today)  # resumable
    except polite.SourceBlocked as e:
        run["result"] = "stopped: " + str(e)
        warning(f"jcg-senkaku stopped: {e}")
        return 75
    except (Stop, OSError, http.client.HTTPException) as e:  # network errors
        run["result"] = "stopped: " + str(e)
        warning(f"jcg-senkaku stopped: {e}; next run resumes")
        return 75
    return 0


# ---------------------------------------------------------------- probe

def layout(words: list[list], limit: int = 3800) -> str:
    """Compact positional dump: 'top|x0:text x0:text ...' per visual line."""
    from collectors.jcg_parse import _w
    out = []
    for p in sorted({w[0] for w in words}):
        out.append(f"--page {p}")
        for line in lines([_w(w) for w in words if w[0] == p]):
            out.append(f"{line[0]['top']:.0f}|" + " ".join(f"{w['x0']:.0f}:{w['t']}" for w in line))
    s = "\n".join(out)
    return s if len(s) <= limit else s[:limit] + "…"


def summary(rec: dict, today: date) -> str:
    p = rec["parse"]
    end = last_day(rec, today)
    vals = [p["days"].get(str(d), {}).get("cz") for d in range(1, (end or 0) + 1)]
    return (f"{month_key(rec)} {rec['url'].rsplit('/', 1)[-1]} {rec['bytes']}B lm={rec['last_modified_utc']} "
            f"parsed={p['parsed']} days={len(p['days'])} as_of={p['as_of']} title={p['title_year_month']} "
            f"totals={p['totals']} check={check(p, end) if end else None} cz={vals} "
            f"problems={p['problems'][:4]}")


def probe(sample_dir: str) -> int:
    """Fetch the page and a sample of PDFs (about one per year); print; write nothing to docs/."""
    d = Path(sample_dir)
    d.mkdir(parents=True, exist_ok=True)
    today = now_utc().astimezone(JST).date()
    robots = polite.check_robots(PAGE_URL, UA)
    notice(f"probe robots: allowed={robots.allowed} verified={robots.verified} {robots.reason}")
    if not robots.allowed:
        return 0 if robots.verified else 75
    try:
        page = get(PAGE_URL)
    except polite.SourceBlocked as e:
        warning(f"probe page: {e}")
        return 75
    links = page_links(decode_html(page.body, page.headers.get("content-type", "")), PAGE_URL)
    unknown = [l for l in links if not (l["year"] and l["month"])]
    keys = [f"{l['year']}-{l['month']}" for l in links]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    notice(f"probe page: HTTP {page.status}, {len(page.body)} bytes, {page.headers.get('content-type')}, "
           f"last-modified {page.headers.get('last-modified')}; {len(links)} PDF links, "
           f"first {links[:2]}, last {links[-2:]}, without year/month {unknown[:10]}, duplicate months {dup}")
    picks, years = [], set()
    for i, l in enumerate(links):
        name = l["url"].rsplit("/", 1)[-1]
        if l["year"] not in years or i >= len(links) - 2 or re.search(r"h30_0?6|_R_0?5", name):
            years.add(l["year"])
            picks.append(l)
    lines_out, index = [], []
    for i, link in enumerate(picks):
        try:
            resp = fetch_pdf(link, None)
        except (polite.SourceBlocked, Stop, OSError, http.client.HTTPException) as e:
            lines_out.append(f"#{i} {link['url']}: {e}")
            if isinstance(e, polite.Challenge):
                break
            continue
        (d / f"{i}.pdf").write_bytes(resp.body)
        rec = record(link, resp, now_utc())
        rec.pop("words")
        index.append(rec)
        lines_out.append(f"#{i} " + summary({**rec, "words": None}, today))
    (d / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    chunk = ""
    for s in lines_out:  # at most a few annotations
        if len(chunk) + len(s) > 3800:
            notice("probe samples:\n" + chunk)
            chunk = ""
        chunk += s + "\n"
    if chunk:
        notice("probe samples:\n" + chunk)
    return 0


def dump(sample_dir: str, start: int, end: int) -> int:
    d = Path(sample_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8")) if (d / "index.json").exists() else []
    for i in range(start, min(end, len(index))):
        rec = index[i]
        path = d / f"{i}.pdf"
        if not path.exists():
            continue
        words = pdf_words(path.read_bytes())
        text = layout(words, limit=7600)
        name = rec["url"].rsplit("/", 1)[-1]
        for part in range(0, len(text), 3800):
            notice(f"layout #{i} {name} ({len(words)} words) part {part // 3800 + 1}\n" + text[part:part + 3800])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fetch", type=int, default=250, help="PDF requests at most in this run")
    ap.add_argument("--probe", metavar="DIR")
    ap.add_argument("--dump", metavar="DIR")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=6)
    a = ap.parse_args()
    if a.probe:
        return probe(a.probe)
    if a.dump:
        return dump(a.dump, a.start, a.end)
    return collect(a.max_fetch)


if __name__ == "__main__":
    sys.exit(main())
