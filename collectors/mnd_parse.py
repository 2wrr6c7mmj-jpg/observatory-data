"""Parse Taiwan MND daily reports on PLA activity around Taiwan.

The reports (English: /en/news/plaact/<id>, Chinese: /news/plaact/<id>) cover the
24 hours from 06:00 to 06:00 (UTC+8) ending on the report date, and state the
counts in a fixed sentence, for example:

  "8 sorties of PLA aircraft and 6 PLAN vessels operating around Taiwan were
   detected up until 6 a.m. (UTC+8) today. 6 out of 8 sorties crossed the
   median line ..."
  "偵獲共機27架次、共艦9艘及公務船2艘 ... 其中22架次逾越海峽中線"

Only numbers stated in the text are recorded. A count the text does not state
stays None (unknown), never 0, and a page whose main counts cannot be read is
returned as unparsed so it can be reviewed instead of guessed.
"""
from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta, timezone

PARSER_VERSION = 1
TAIPEI = timezone(timedelta(hours=8))

WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM = r"\b(\d{1,4}|" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")"
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _n(value: str | None) -> int | None:
    if value is None:
        return None
    v = value.strip().lower()
    return int(v) if v.isdigit() else WORDS.get(v)


def page_text(raw_html: str) -> str:
    """Visible text of a page, with tags removed and whitespace collapsed."""
    s = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", raw_html)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h\d|tr)>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("　", " ").replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", s)


def report_date(text: str) -> date | None:
    """Report date (the day the 06:00 window ends), in Taiwan local time."""
    m = re.search(r"\b(20\d\d)[./-](\d{1,2})[./-](\d{1,2})\b", text)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"\b(1[01]\d)[./](\d{1,2})[./](\d{1,2})\b", text)  # ROC calendar
    if m:
        return date(int(m[1]) + 1911, int(m[2]), int(m[3]))
    m = re.search(r"中華民國(1[01]\d)年(\d{1,2})月(\d{1,2})日[^至]*至(\d{1,2})月(\d{1,2})日", text)
    if m:
        return date(int(m[1]) + 1911, int(m[4]), int(m[5]))
    m = re.search(r"\b([A-Z][a-z]{2})\.?\s+(\d{1,2}),\s+(20\d\d)", text)
    if m and m[1].lower() in MONTHS:
        return date(int(m[3]), MONTHS[m[1].lower()], int(m[2]))
    return None


def _first(patterns: list[str], text: str, group: int = 1) -> int | None:
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return _n(m[group])
    return None


def counts(text: str) -> dict:
    t = re.sub(r"\s+", " ", text)
    aircraft = _first([
        NUM + r" sorties? of (?:PLA|Chinese) (?:military )?aircraft",
        NUM + r" (?:PLA|Chinese) (?:military )?aircraft",
        r"共機\s*(\d+)\s*架",
        r"軍機\s*(\d+)\s*架",
    ], t)
    vessels = _first([
        NUM + r" PLAN (?:vessels?|ships?)",
        NUM + r" (?:PLA )?naval (?:vessels?|ships?)",
        r"共艦\s*(\d+)\s*艘",
        r"(?:軍艦|艦艇)\s*(\d+)\s*艘",
    ], t)
    official = _first([
        NUM + r" (?:PRC |Chinese )?official (?:ships?|vessels?)",
        r"公務船\s*(\d+)\s*艘",
    ], t)
    balloons = _first([
        NUM + r" (?:PRC |Chinese )?balloons?",
        r"(?:空飄)?氣球\s*(\d+)\s*(?:顆|個|枚)",
    ], t)
    crossed = _first([
        NUM + r" (?:out )?of (?:the )?(?:\d+ )?sorties? (?:crossed|entered)",
        r"\ball " + NUM + r" sorties? (?:crossed|entered)",
        NUM + r" sorties? (?:crossed|entered)",
        r"其中\s*(\d+)\s*架次(?:逾越|進入)",
    ], t)
    if crossed is None and re.search(r"no PLA aircraft crossed|無逾越海峽中線|did not cross", t, re.I):
        crossed = 0
    if aircraft is None and re.search(r"\bno PLA aircraft\b|未偵獲共機", t, re.I):
        aircraft = 0
    return {"aircraft": aircraft, "median_line_or_adiz": crossed,
            "plan_vessels": vessels, "official_ships": official, "balloons": balloons}


def parse_report(raw_html: str, url: str, retrieved: datetime) -> dict:
    text = page_text(raw_html)
    d = report_date(text)
    c = counts(text)
    ok = d is not None and (c["aircraft"] is not None or c["plan_vessels"] is not None)
    record = {
        "source_url": url,
        "retrieved_utc": retrieved.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parser_version": PARSER_VERSION,
        "parsed": ok,
    }
    if d is not None:
        end = datetime(d.year, d.month, d.day, 6, tzinfo=TAIPEI)
        record.update({
            "report_date": d.isoformat(),
            "window_start_utc": (end - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "window_end_utc": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        })
    record.update(c)
    if not ok:
        record["text_excerpt"] = text.strip()[:600]
    return record


def report_links(list_html: str) -> list[tuple[str, int]]:
    """Report links on a list or tag page, newest first, without duplicates."""
    seen, out = set(), []
    for m in re.finditer(r"""href\s*=\s*["']((?:https?://[^"']*?/|/)?(?:en/)?news/plaact/(\d+))["']""", list_html, flags=re.I):
        rid = int(m[2])
        if rid not in seen:
            seen.add(rid)
            out.append((m[1], rid))
    return out
