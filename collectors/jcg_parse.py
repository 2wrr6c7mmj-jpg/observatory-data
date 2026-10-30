"""Parse the Japan Coast Guard Senkaku page and its monthly PDFs.

The index page links one PDF per month. Each PDF holds a table with one row
per day: day, vessels that entered the contiguous zone, and vessels that
entered the territorial sea (a subset of the former), plus monthly totals
("接続水域入域；N日、のべM隻") and an as-of date ("( 9/30 現在)").

The text stream of these PDFs is jumbled, so the table is read from word
positions (pdfplumber words: text, x0, x1, top, bottom). Parsing works on a
plain word list so it can be tested without pdfplumber and re-run on stored
words without fetching the PDF again.
"""
from __future__ import annotations

import calendar
import html
import re
import unicodedata
from datetime import date
from urllib.parse import urljoin, urlsplit

PARSER_VERSION = 2

ERA = {"平成": 1988, "令和": 2018, "H": 1988, "R": 2018}
DASHES = {"-", "−", "－", "―", "‐", "ー", "—", "–"}


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def era_year(era: str, n: str) -> int:
    return ERA[era] + (1 if n in ("元", "1") else int(n))


# ---------------------------------------------------------------- index page

_TOKEN = re.compile(
    r"(?P<year>(?:平成|令和)\s*(?:元|\d{1,2})\s*年)"
    r"|<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a\s*>",
    re.I | re.S,
)
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)


def page_links(raw_html: str, page_url: str) -> list[dict]:
    """Monthly PDF links in page order, with the year of the table row they are in.

    The year is the last era year (平成NN年 / 令和N年) seen before the link in the
    document; the month comes from the link text. Either may be None when the
    page does not say; the PDF title is used then.
    """
    out, seen = [], set()
    year = None
    text = re.sub(r"<!--.*?-->", "", raw_html, flags=re.S)
    text = re.sub(r"<(script|style)\b.*?</\1\s*>", "", text, flags=re.S | re.I)
    for m in _TOKEN.finditer(nfkc_keep(text)):
        if m["year"]:
            y = re.match(r"(平成|令和)\s*(元|\d+)", m["year"])
            year = era_year(y[1], y[2])
            continue
        href = _HREF.search(m["attrs"] or "")
        if not href:
            continue
        url = urljoin(page_url, html.unescape(href[1]).strip())
        path = urlsplit(url).path
        if not path.lower().endswith(".pdf") or "/mission/senkaku/" not in path:
            continue
        if url in seen:
            continue
        seen.add(url)
        label = nfkc(html.unescape(re.sub(r"<[^>]+>", "", m["text"])))
        mm = re.search(r"(\d{1,2})", label)
        month = int(mm[1]) if mm and 1 <= int(mm[1]) <= 12 else None
        out.append({"url": url, "label": label, "year": year, "month": month})
    return out


def decode_html(body: bytes, content_type: str = "") -> str:
    """Bytes of the page as text: HTTP charset, then <meta charset>, then UTF-8, then CP932."""
    cands = []
    m = re.search(r"charset=([\w.-]+)", content_type or "", re.I)
    if m:
        cands.append(m[1])
    m = re.search(rb"""<meta[^>]+charset\s*=\s*["']?([\w.-]+)""", body[:4096], re.I)
    if m:
        cands.append(m[1].decode("ascii", "replace"))
    for enc in cands + ["utf-8", "cp932"]:
        enc = {"shift_jis": "cp932", "sjis": "cp932", "x-sjis": "cp932"}.get(enc.lower(), enc)
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def nfkc_keep(s: str) -> str:
    """NFKC without stripping (positions in the document do not matter here)."""
    return unicodedata.normalize("NFKC", s)


# ---------------------------------------------------------------- PDF words

def _w(w) -> dict:
    """Accept pdfplumber dicts or stored [page, text, x0, x1, top, bottom] lists."""
    if isinstance(w, dict):
        return {"p": w.get("p", 0), "t": nfkc(w["t"] if "t" in w else w["text"]),
                "x0": float(w["x0"]), "x1": float(w["x1"]), "top": float(w["top"]), "bottom": float(w["bottom"])}
    p, t, x0, x1, top, bottom = w
    return {"p": p, "t": nfkc(t), "x0": float(x0), "x1": float(x1), "top": float(top), "bottom": float(bottom)}


def _cx(w):
    return (w["x0"] + w["x1"]) / 2


def _cy(w):
    return (w["top"] + w["bottom"]) / 2


def _h(w):
    return max(w["bottom"] - w["top"], 1.0)


def lines(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines (by vertical centre), each sorted left to right."""
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (_cy(w), w["x0"])):
        if out:
            cur = out[-1]
            mean = sum(_cy(x) for x in cur) / len(cur)
            tol = max(2.5, 0.45 * min(_h(w), min(_h(x) for x in cur)))
            if abs(_cy(w) - mean) <= tol:
                cur.append(w)
                continue
        out.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in out]


def line_text(line: list[dict], sep: str = "") -> str:
    return sep.join(w["t"] for w in line)


_DAY = re.compile(r"^(\d{1,2})日?$")
_VAL = re.compile(r"^(\d{1,3})隻?$")


def _dnum(w) -> int:
    return int(_DAY.match(w["t"])[1])


def _day_columns(words: list[dict]) -> list[list[dict]]:
    """Vertical runs of consecutive day numbers (n, n+1, ... going down)."""
    cands = [w for w in words if _DAY.match(w["t"]) and 1 <= _dnum(w) <= 31]
    clusters: list[list[dict]] = []
    for w in sorted(cands, key=lambda w: w["x1"]):
        for c in clusters:  # day numbers are right-aligned or centred: compare both
            ref = c[0]
            if abs(w["x1"] - ref["x1"]) <= 4 or abs(_cx(w) - _cx(ref)) <= 4:
                c.append(w)
                break
        else:
            clusters.append([w])
    runs = []
    for c in clusters:
        c.sort(key=lambda w: w["top"])
        run: list[dict] = []
        for w in c:
            if run and _dnum(w) == _dnum(run[-1]) + 1:
                run.append(w)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = [w]
        if len(run) >= 2:
            runs.append(run)
    return runs


def _chain(runs: list[list[dict]]) -> list[list[dict]]:
    """The longest sequence of runs continuing each other left to right.

    The PDFs have two tables side by side (days 1-16 | 17-31); the first
    month (2012-09) starts at day 14.
    """
    best: list[list[dict]] = []
    for start in runs:
        chain = [start]
        while True:
            nxt = _dnum(chain[-1][-1]) + 1
            follow = [r for r in runs if _dnum(r[0]) == nxt and r[0]["x0"] > chain[-1][0]["x1"]]
            if not follow:
                break
            chain.append(max(follow, key=len))
        if sum(map(len, chain)) > sum(map(len, best)):
            best = chain
    return best


def _header(words, kind, run, right_bound):
    """Column header above a run of days: the nearest one to the right of the day column."""
    def ok(t):
        if kind == "cz":
            return "接続水域" in t and "領海" not in t and "内数" not in t
        return "領海" in t and "接続" not in t and "内数" not in t
    first = run[0]
    hs = [w for w in words if ok(w["t"]) and w["bottom"] <= first["top"] + 1
          and w["x0"] >= first["x1"] - 2 and _cx(w) < right_bound]
    return min(hs, key=lambda w: (w["x0"] - first["x1"], -w["top"])) if hs else None


# "2【3】": 2 vessels, entering the territorial sea 3 times in all (the note of the
# 2016-08 PDF: the number in 【 】 is the number of entries). NFKC keeps 【 】.
_VAL_ENTRIES = re.compile(r"^(\d{1,3})隻?[【\[](\d{1,3})[】\]]$")
_MARKS = {"※", "*", "＊"}  # a footnote mark beside a cell, not a value


def _cell(t: str):
    if t in DASHES:
        return 0, "dash"
    m = _VAL.match(t)
    if m:
        return int(m[1]), "number"
    m = _VAL_ENTRIES.match(t)
    return (int(m[1]), "number", int(m[2])) if m else (None, None)


# "###" is a spreadsheet cell too narrow for its number (seen in a current-month PDF).
_TOTAL_CZ = re.compile(r"接続水域入域\)?[;:]\s*(\d+)日[、,]?\s*(?:のべ|延べ)?\s*(\d+|#+)隻")
_TOTAL_TS = re.compile(r"領海侵入.*?[;:]\s*(\d+)日[、,]?\s*(?:のべ|延べ)?\s*(\d+|#+)隻")
_TOTAL_TS_NONE = re.compile(r"領海侵入.*?[;:]\s*(?:なし|0日)")
_ASOF = [re.compile(r"\(?\s*(\d{1,2})\s*/\s*(\d{1,2})\s*現在"),
         re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*現在")]
_TITLE = re.compile(r"[-−－―]\s*(平成|令和|H|R)\s*(元|\d{1,2})\s*年\s*(\d{1,2})\s*月\s*[-−－―]")


def _num(s: str) -> int | None:
    return int(s) if s.isdigit() else None


def parse_words(raw_words: list, year: int | None = None, month: int | None = None) -> dict:
    """Days, totals, as-of date and title month from PDF words.

    Returns {"parsed": bool, "problems": [...], "days": {"1": {"cz": n|None,
    "ts": n|None, "cz_cell": "number"|"dash"|"blank", ...}}, "totals": {...},
    "as_of": "YYYY-MM-DD"|None, "title_year_month": "YYYY-MM"|None}.
    Blank cells are None here; whether they mean zero is decided against the
    totals in check().
    """
    words = [_w(w) for w in raw_words]
    out = {"parsed": False, "problems": [], "days": {}, "totals": {}, "as_of": None, "title_year_month": None,
           "subject": None}
    if not words:
        out["problems"].append("no text in PDF (image only?)")
        return out
    pages = sorted({w["p"] for w in words})
    best = None
    for p in pages:  # the page with the longest day run holds the table
        pw = [w for w in words if w["p"] == p]
        chain = _chain(_day_columns(pw))
        if best is None or len(chain) > len(best[1]):
            best = (pw, chain)
    pw, chain = best
    alltext = [line_text(l) for p in pages for l in lines([w for w in words if w["p"] == p])]

    for t in alltext:
        if out["subject"] is None and (m := re.search(r"^(.+?)による尖閣諸島接近", t)):
            out["subject"] = m[1]  # 中国公船 / 中国海警局に所属する船舶等
        m = _TITLE.search(t)
        if m and out["title_year_month"] is None:
            out["title_year_month"] = f"{era_year(m[1], m[2]):04d}-{int(m[3]):02d}"
    if (year is None or month is None) and out["title_year_month"]:
        year, month = (int(x) for x in out["title_year_month"].split("-"))

    for t in alltext:
        if "領海" not in t and (m := _TOTAL_CZ.search(t)):
            out["totals"]["cz_days"], out["totals"]["cz_vessels"] = int(m[1]), _num(m[2])
        elif m := _TOTAL_TS.search(t):
            out["totals"]["ts_days"], out["totals"]["ts_vessels"] = int(m[1]), _num(m[2])
        elif _TOTAL_TS_NONE.search(t):
            out["totals"]["ts_days"], out["totals"]["ts_vessels"] = 0, 0
        for rx in _ASOF:
            if out["as_of"] is None and (m := rx.search(t)) and year:
                am, ad = int(m[1]), int(m[2])
                ay = year + 1 if month == 12 and am == 1 else year
                try:
                    out["as_of"] = date(ay, am, ad).isoformat()
                except ValueError:
                    out["problems"].append(f"bad as-of date {m[0]!r}")

    ndays = sum(map(len, chain))
    if ndays < 5:
        out["problems"].append(f"day column not found ({ndays} consecutive days)")
        return out
    for i, run in enumerate(chain):
        right = chain[i + 1][0]["x0"] if i + 1 < len(chain) else float("inf")
        cz_h = _header(pw, "cz", run, right)
        ts_h = _header(pw, "ts", run, right)
        if cz_h and ts_h and _cx(ts_h) <= _cx(cz_h):
            cz_h = ts_h = None  # unexpected order; fall back to left-to-right
        if cz_h and ts_h and right == float("inf"):
            right = ts_h["x1"] + max(30.0, ts_h["x1"] - ts_h["x0"])
        for dw in run:
            d = _dnum(dw)
            tol = max(2.5, 0.5 * _h(dw))
            row = sorted((w for w in pw if w is not dw and abs(_cy(w) - _cy(dw)) <= tol
                          and w["x0"] >= dw["x1"] - 0.5 and _cx(w) < right), key=lambda w: w["x0"])
            row = [w for w in row if w["t"] not in _MARKS]
            cells, entries = [], {}
            for w in row:
                v, k, *e = _cell(w["t"])
                cells.append((w, v, k))
                if e:
                    entries[id(w)] = e[0]
            stray = [w["t"] for w, v, _ in cells if v is None]
            cells = [(w, v, k) for w, v, k in cells if v is not None]
            rec = {"cz": None, "ts": None, "cz_cell": "blank", "ts_cell": "blank"}
            if cz_h and ts_h:
                for w, v, k in cells:
                    col = "cz" if abs(_cx(w) - _cx(cz_h)) <= abs(_cx(w) - _cx(ts_h)) else "ts"
                    if rec[col + "_cell"] != "blank":
                        out["problems"].append(f"day {d}: two values in one column")
                    rec[col], rec[col + "_cell"] = v, k
                    if id(w) in entries:
                        if col != "ts":
                            out["problems"].append(f"day {d}: entry count 【 】 outside the territorial-sea column")
                        rec["ts_entries"] = entries[id(w)]
            else:
                if len(cells) > 2:
                    out["problems"].append(f"day {d}: {len(cells)} values")
                for col, (w, v, k) in zip(("cz", "ts"), cells):
                    rec[col], rec[col + "_cell"] = v, k
                    if id(w) in entries:
                        if col != "ts":
                            out["problems"].append(f"day {d}: entry count 【 】 outside the territorial-sea column")
                        rec["ts_entries"] = entries[id(w)]
            if stray:
                out["problems"].append(f"day {d}: unreadable cell {stray}")
            out["days"][str(d)] = rec
    if year and month:
        last = calendar.monthrange(year, month)[1]
        extra = [d for d in out["days"] if int(d) > last]
        for d in extra:
            if out["days"][d]["cz"] is not None:
                out["problems"].append(f"day {d} does not exist in {year}-{month:02d}")
            del out["days"][d]
    for d, r in out["days"].items():
        if r["cz"] is not None and r["ts"] is not None and r["ts"] > r["cz"]:
            out["problems"].append(f"day {d}: territorial sea {r['ts']} > contiguous zone {r['cz']}")
    out["parsed"] = not out["problems"]
    return out


def check(parsed: dict, last_day: int) -> str:
    """Compare the day rows (up to last_day) with the monthly totals.

    "match": the totals (days and vessels) equal the rows, so blank cells mean
    none confirmed (0); "match_days": the same, but a vessel total was
    unreadable ("###") and only the day counts could be compared;
    "no_totals": totals not found; "mismatch": they differ.
    """
    t = parsed.get("totals") or {}
    days = [r for d, r in parsed.get("days", {}).items() if int(d) <= last_day]
    if "cz_days" not in t:
        return "no_totals"
    partial = False
    for key in ("cz", "ts"):
        if f"{key}_days" not in t:
            continue
        vals = [r[key] or 0 for r in days]
        if sum(1 for v in vals if v > 0) != t[f"{key}_days"]:
            return "mismatch"
        if t.get(f"{key}_vessels") is None:
            partial = True
        elif sum(vals) != t[f"{key}_vessels"]:
            return "mismatch"
    return "match_days" if partial else "match"
