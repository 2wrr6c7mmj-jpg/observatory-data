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

PARSER_VERSION = 7
MAX_PLAUSIBLE = 200  # a daily count above this is a misread (e.g. phone numbers in the page footer)
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


def window_end(text: str, header: date | None) -> tuple[date | None, str]:
    """Report date from the stated 06:00-06:00 window, checked against the page date.

    Returns (date, how): "window" when the window text gives the end day,
    "header" when only the page date is available.
    """
    t = re.sub(r"\s+", " ", text)
    month = day = None
    m = re.search(r"to 6 ?a\.?m\.? ([A-Z][a-z]{2})[a-z]*\.? (\d{1,2})\b", t)
    if m and m[1].lower() in MONTHS:
        month, day = MONTHS[m[1].lower()], int(m[2])
    if month is None:  # "Sunday to Monday, Dec. 24-25" / "Dec. 31-Jan. 1"
        m = re.search(r"\b([A-Z][a-z]{2})\.? \d{1,2}\s*-\s*(?:([A-Z][a-z]{2})\.? )?(\d{1,2})\b", t)
        if m and m[1].lower() in MONTHS:
            month, day = MONTHS[(m[2] or m[1]).lower()], int(m[3])
    if month is None:
        m = re.search(r"至\s*(\d{1,2})月(\d{1,2})日", t)
        if m:
            month, day = int(m[1]), int(m[2])
    if month is None or header is None:
        return header, "header"
    year = header.year
    if month == 12 and header.month == 1:
        year -= 1
    elif month == 1 and header.month == 12:
        year += 1
    try:
        return date(year, month, day), "window"
    except ValueError:
        return header, "header"


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
        NUM + r" ?(?:PLA|Chinese) (?:military )?aircraft",
        r"共機\s*(\d+)\s*架",
        r"軍機\s*(\d+)\s*架",
    ], t)
    vessels = _first([
        NUM + r" PLAN (?:vessels?|ships?)",
        NUM + r" (?:PLA )?naval (?:vessels?|ships?)",
        NUM + r" PLAN (?=around|operating)",  # "4 PLA aircraft and 10 PLAN around Taiwan"
        r"共艦\s*(\d+)\s*艘",
        r"(?:軍艦|艦艇)\s*(\d+)\s*艘",
    ], t)
    official = _first([
        NUM + r" (?:PRC |Chinese )?official (?:ships?|vessels?)",
        NUM + r" (?:CCG|China Coast Guard|coast guard) (?:ships?|vessels?)",
        r"公務船\s*(\d+)\s*艘",
    ], t)
    balloons = _first([
        NUM + r" (?:PRC |Chinese )?balloons?",
        r"(?:空飄)?氣球\s*(\d+)\s*(?:顆|個|枚)",
    ], t)
    crossed = _first([
        NUM + r" (?:out )?of (?:the )?(?:\d+ )?sorties? (?:had )?(?:crossed|entered)",
        NUM + r" of (?:the |them )?(?:detected )?(?:PLA )?aircraft (?:had )?(?:crossed|entered)",
        r"\ball " + NUM + r" sorties? (?:crossed|entered)",
        NUM + r" sorties? (?:crossed|entered)",
        r"\bThe " + NUM + r" (?:detected )?aircraft (?:had )?(?:crossed|entered)",
        NUM + r" helicopters? (?:had )?(?:crossed|entered)",
        r"其中\s*(\d+)\s*架次(?:逾越|進入)",
    ], t)
    if crossed is None and re.search(r"no PLA aircraft crossed|無逾越海峽中線|did not cross", t, re.I):
        crossed = 0
    if aircraft is None and re.search(r"\bno PLA aircraft\b|未偵獲共機", t, re.I):
        aircraft = 0
    if vessels is None and re.search(r"\bno PLA aircraft and (?:PLAN )?(?:vessels?|ships?)\b", t, re.I):
        vessels = 0  # "No PLA aircraft and PLAN vessel(s) operating around Taiwan were detected"
    if vessels is None and aircraft is not None and re.search(
            r"activities\s*[:：]\s*" + NUM + r" ?PLA aircraft (?:operating )?around Taiwan were detected", t, re.I):
        vessels = 0  # the daily tally names only aircraft: no ships were detected
    if crossed is None and aircraft and re.search(r"\bThe (?:detected )?aircraft (?:had )?(?:crossed|entered)", t):
        crossed = aircraft  # "The detected aircraft had entered ...": all of them
    if aircraft is None and vessels is not None and re.search(
            r"activities\s*[:：]\s*" + NUM + r" PLAN (?:vessels?|ships?)\b(?: and \d+ (?:PRC |Chinese )?official (?:ships?|vessels?))?"
            r" (?:operating )?around Taiwan were detected", t, re.I):
        # The daily tally lists what was detected; a report naming only ships detected no aircraft.
        aircraft = 0
    return {"aircraft": aircraft, "median_line_or_adiz": crossed,
            "plan_vessels": vessels, "official_ships": official, "balloons": balloons}


TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
UNITS = [w for w in WORDS if 0 < WORDS[w] < 20]
COUNT = re.compile(
    r"(?<![\w-])(?:(" + "|".join(TENS) + r")(?:[\s-]+(" + "|".join(w for w in UNITS if WORDS[w] < 10) + r"))?"
    r"|(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")|(\d{1,3}))"
    # A number word may be joined to the type by a hyphen ("One-Y-8 EW"); a digit may not
    # (the "8" of "Y-8" and the "16" of "J-16" are preceded by "-" and never start a count).
    r"(?:(?![\w-])|(?<=[a-z])(?=-[a-z]))", re.I)


def adiz_types(text: str) -> dict | None:
    """Older reports (until late 2022): aircraft entering the southwestern ADIZ, by type.

    "二、Aircraft Type: Thirty four J-16 Two SU-30 Two Y-8 ASW ..." → {"J-16": 34, ...}
    These count ADIZ entries only and are not comparable with the later
    "around Taiwan" totals, so they are kept in separate fields.
    """
    t = re.sub(r"\s+", " ", text)
    end = r"(?:[三3][、.]|Activit(?:y|ies) Areas?|Reactions?\s*[:：]|Keywords|$)"
    m = re.search(r"Aircraft Types?\s*[:：]\s*(.*?)" + end, t, re.I)
    if not m:  # no "Aircraft Type:" label: the list follows the date line, e.g.
        # "Date: October 21,2021（Thursday daytime） Two J-16 Two J-11 Activity Area: ..."
        m = re.search(r"Date\s*[:：][^()（）]{0,40}[(（][^)）]{0,30}[)）]\s*(.*?)" + end, t, re.I)
        if not m or not re.match(r"(?:" + COUNT.pattern + r")\s+[A-Z]", m[1].strip(), re.I):
            return None
    section, out = m[1], {}
    hits = list(COUNT.finditer(section))
    for i, h in enumerate(hits):
        n = TENS[h[1].lower()] + (WORDS[h[2].lower()] if h[2] else 0) if h[1] else WORDS[h[3].lower()] if h[3] else int(h[4])
        # Brackets around each entry ("(One Y-8 ASW) (Two J-10)") or a stray ")" are not
        # part of the type name (parser 7).
        name = section[h.end():hits[i + 1].start() if i + 1 < len(hits) else len(section)].strip(" ,;、.-()（）")
        if name:
            out[name] = out.get(name, 0) + n
    return out or None


def main_text(text: str) -> str:
    """The report itself: from the "PLA Activities YYYY.MM.DD" heading to "Keywords".

    Site menus and the page footer (addresses and telephone numbers such as
    "886" or "409") are left out, so they are never read as counts.
    """
    body = re.sub(r"\s+", " ", text)
    i = body.find("PLA Activities 20")
    j = body.find("Keywords", i if i >= 0 else 0)
    return body[max(i, 0):j if j > i >= 0 else max(i, 0) + 2000][:3000].strip()


TYPE_FORMATS = ("sw-adiz", "afc-around-roc")


AFC_AROUND_ROC_FROM = date(2022, 8, 3)


def type_format(text: str, d: date | None) -> tuple[str, str]:
    """Which of the older by-type formats a report is in.

    sw-adiz: "Air activities in the southwestern ADIZ of R.O.C." (aircraft that
    entered the southwestern ADIZ only).
    afc-around-roc: "Air activities of PLA aircraft around the R.O.C. airspace"
    (Air Force Command, 2022-08-03 to 2022-11-12: around Taiwan as a whole,
    not only the southwestern ADIZ).
    Returns (format, basis): "heading" when the report says which, "series" when
    it has no heading and is an Air Force Command report from 2022-08-03 on
    (all of its headed reports from then on are around-the-R.O.C.).
    """
    if re.search(r"around the R\.?\s?O\.?\s?C\.? airspace", text, re.I):
        return "afc-around-roc", "heading"
    if re.search(r"southwestern ADIZ|southwest ADIZ", text, re.I):
        return "sw-adiz", "heading"
    if d is not None and d >= AFC_AROUND_ROC_FROM and re.search(r"Issuing Authority\s*[:：]\s*Air Force Command", text, re.I):
        return "afc-around-roc", "series"
    return "sw-adiz", "series"


def period(text: str) -> str | None:
    """Part of the day in the date line: "(Thursday morning)", "（Monday night）"."""
    m = re.search(r"Date\s*[:：][^()（）]{0,40}[(（]([^)）]{0,30})[)）]", re.sub(r"\s+", " ", text), re.I)
    if not m:
        return None
    w = m[1].lower()
    for key, name in (("morning", "morning"), ("afternoon", "afternoon"), ("afteroon", "afternoon"),
                      ("daytime", "daytime"), ("night", "nighttime")):
        if key in w:
            return name
    return None


def parse_report(raw_html: str, url: str, retrieved: datetime) -> dict:
    full = page_text(raw_html)
    body = main_text(full)
    text = body if "PLA Activities 20" in body else full
    d, how = window_end(text, report_date(text))
    c = counts(text)
    types = adiz_types(text)
    fmt = "around-taiwan"
    if types is not None and c["aircraft"] is None and c["plan_vessels"] is None:
        c["adiz_aircraft"] = sum(types.values())
        c["adiz_types"] = types
        c["period"] = period(text)
        if d is None or how == "header":
            dm = re.search(r"Date\s*[:：]\s*([A-Z][a-z]+)\.? ?(\d{1,2}), ?(20\d\d)", re.sub(r"\s+", " ", text))
            if dm and dm[1][:3].lower() in MONTHS:
                d, how = date(int(dm[3]), MONTHS[dm[1][:3].lower()], int(dm[2])), "date-line"
        fmt, c["format_basis"] = type_format(text, d)
    ok = d is not None and (c["aircraft"] is not None or c["plan_vessels"] is not None or c.get("adiz_aircraft") is not None)
    too_big = {k: v for k, v in c.items() if isinstance(v, int) and v > MAX_PLAUSIBLE}
    record = {
        "source_url": url,
        "retrieved_utc": retrieved.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parser_version": PARSER_VERSION,
        "parsed": ok and not too_big,
        "date_from": how,
        "format": fmt,
    }
    if too_big:  # a misread, not a count: kept for review, never published
        record["parse_error"] = f"implausible count (> {MAX_PLAUSIBLE}): {too_big}"
    if d is not None and fmt in TYPE_FORMATS:  # one local calendar day (morning / afternoon / night reports)
        start = datetime(d.year, d.month, d.day, tzinfo=TAIPEI)
        record.update({
            "report_date": d.isoformat(),
            "window_start_utc": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "window_end_utc": (start + timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        })
    elif d is not None:
        end = datetime(d.year, d.month, d.day, 6, tzinfo=TAIPEI)
        record.update({
            "report_date": d.isoformat(),
            "window_start_utc": (end - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "window_end_utc": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        })
    record.update(c)
    # Keep the report's own text (without site menus) so that a later parser
    # version can re-read every report offline instead of crawling the site again.
    record["main_text"] = body
    return record


def reparse(record: dict) -> dict:
    """Re-read a stored report from its kept text (no network)."""
    if not record.get("main_text"):
        return record
    retrieved = datetime.strptime(record["retrieved_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return parse_report("<p>" + html.escape(record["main_text"]) + "</p>", record["source_url"], retrieved)


def report_links(list_html: str) -> list[tuple[str, int]]:
    """Report links on a list or tag page, newest first, without duplicates."""
    seen, out = set(), []
    for m in re.finditer(r"""href\s*=\s*["']((?:https?://[^"']*?/|/)?(?:en/)?news/plaact/(\d+))["']""", list_html, flags=re.I):
        rid = int(m[2])
        if rid not in seen:
            seen.add(rid)
            out.append((m[1], rid))
    return out
