"""Dates from the words of the source, by rule (the model's own dates are not used).

The model quotes the words that give the period (dateEvidence). This module reads those
words and returns the period only when they state it: a day, a span of days, or a part of
a month (上旬・中旬・下旬, early/mid/late). Relative words (先日, 日前, this week, Tuesday,
early this month) are not turned into days; neither is a page's own time stamp
("2026-11-07 16:07"). A year missing from the words is taken from the publication day,
and only when the publication day is known.

    parse("令和８年１１月１７日（火）から同月２０日（金）までの間", "2026-11-13")
        → ("2026-11-17", "2026-11-20", "day")
    parse("9月中旬至10月上旬", "2026-10-12") → ("2026-09-11", "2026-10-10", "part")
    parse("日前", "2026-11-14") → None
"""
import calendar
import re
from datetime import date

_Z2H = str.maketrans("０１２３４５６７８９／－～：", "0123456789/-~:")
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
_PART = {"上旬": (1, 10), "中旬": (11, 20), "下旬": (21, 31), "上中旬": (1, 20), "中下旬": (11, 31),
         "early": (1, 10), "mid": (11, 20), "late": (21, 31)}
_RELATIVE = re.compile(r"日前|近日|先日|最近|本周|上周|今週|先週|昨日|今日|明日|this (week|month)|last (week|month)|next (week|month)|"
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow|recently|earlier this", re.I)


def _day(y, m, d):
    try:
        last = calendar.monthrange(y, m)[1]
        return date(y, m, min(d, last))
    except (ValueError, TypeError):
        return None


def _year_for(m, published):
    """The year of a month without a year: the publication year, or the one before/after when that is closer."""
    if not published:
        return None
    p = date.fromisoformat(published)
    best = None
    for y in (p.year - 1, p.year, p.year + 1):
        d = _day(y, m, 15)
        if d and (best is None or abs((d - p).days) < abs((best - p).days)):
            best = d
    return best.year


def _clean(text):
    t = (text or "").translate(_Z2H)
    t = re.sub(r"[（(][月火水木金土日一二三四五六七][)）]", "", t)   # weekdays in brackets
    return re.sub(r"\s+", " ", t).strip()


def parse(evidence, published=None):
    """(start, end, precision) or None. precision: 'day' or 'part' (a part of a month)."""
    t = _clean(evidence)
    if not t:
        return None
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}", t):
        return None   # a page's time stamp, not the event's period
    era = re.search(r"令和(\d+)年", t)
    year_in_text = 2018 + int(era.group(1)) if era else None
    m = re.search(r"(20\d{2})\s*(?:年|/|-|\.)", t) or re.search(r"\b(20\d{2})\b", t)
    if not year_in_text and m:
        year_in_text = int(m.group(1))

    def year(mo):
        return year_in_text or _year_for(mo, published)

    # Chinese / Japanese: M月D日 … (至|から|～|-) (M月)?D日, and parts of months
    ja = list(re.finditer(r"(?:(\d{1,2})月)?(\d{1,2})日|(\d{1,2})月(上中旬|中下旬|上旬|中旬|下旬)", t))
    if ja:
        points, month, precision = [], None, "day"
        for x in ja:
            if x.group(4):
                month = int(x.group(3))
                lo, hi = _PART[x.group(4)]
                points.append((month, lo, hi))
                precision = "part"
            else:
                if x.group(1):
                    month = int(x.group(1))
                elif re.search(r"同月", t[max(0, x.start() - 3):x.start()]) is None and month is None:
                    month = None
                if month is None:
                    return None
                d = int(x.group(2))
                points.append((month, d, d))
        y0 = year(points[0][0])
        if y0 is None:
            return None
        s = _day(y0, points[0][0], points[0][1])
        last = points[-1]
        y1 = y0 + (1 if last[0] < points[0][0] else 0)
        e = _day(y1, last[0], last[2])
        if not s or not e or e < s:
            return None
        return s.isoformat(), e.isoformat(), precision
    if _RELATIVE.search(t) and not re.search(_MON + r"\s+\d", t, re.I):
        return None
    # English: (early|mid|late) Month; Month D(–D| to Month D)
    part = re.search(r"\b(early|mid|late)[- ]" + _MON, t, re.I)
    if part:
        name = part.group(2).lower().rstrip(".")
        mo = MONTHS.get(name) or MONTHS.get(name[:3])
        y0 = year(mo)
        if y0 is None:
            return None
        lo, hi = _PART[part.group(1).lower()]
        return _day(y0, mo, lo).isoformat(), _day(y0, mo, hi).isoformat(), "part"
    # "19 to 29 October", "19–29 Oct 2026"
    span = re.search(r"\b(\d{1,2})\s*(?:to|through|-|–|~)\s*(\d{1,2})\s+" + _MON, t, re.I)
    if span:
        name = span.group(3).lower().rstrip(".")
        mo = MONTHS.get(name) or MONTHS.get(name[:3])
        y0 = year(mo)
        s, e = (_day(y0, mo, int(span.group(1))), _day(y0, mo, int(span.group(2)))) if y0 else (None, None)
        return (s.isoformat(), e.isoformat(), "day") if s and e and e >= s else None
    en = list(re.finditer(_MON + r"\s+(\d{1,2})\b|\b(\d{1,2})\s+" + _MON, t, re.I))
    if en:
        points = []
        for x in en:
            name = (x.group(1) or x.group(4)).lower().rstrip(".")
            mo = MONTHS.get(name) or MONTHS.get(name[:3])
            d = int(x.group(2) or x.group(3))
            points.append((mo, d))
        # "Oct. 24–30": a second day after a dash belongs to the same month
        tail = re.search(r"\d{1,2}\s*[-–~]\s*(\d{1,2})\b(?!\s*[:/])", t[en[-1].end() - 2:])
        if len(points) == 1 and tail:
            points.append((points[0][0], int(tail.group(1))))
        y0 = year(points[0][0])
        if y0 is None:
            return None
        s = _day(y0, *points[0])
        y1 = y0 + (1 if points[-1][0] < points[0][0] else 0)
        e = _day(y1, *points[-1])
        if not s or not e or e < s:
            return None
        return s.isoformat(), e.isoformat(), "day"
    iso = list(re.finditer(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", t))
    if iso and re.fullmatch(r"20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", t) and (not published or t.replace("/", "-") == published):
        return None   # a bare date in the page's own format: the page's date, not the event's
    if iso:
        s = _day(*map(int, iso[0].groups()))
        e = _day(*map(int, iso[-1].groups()))
        if s and e and e >= s:
            return s.isoformat(), e.isoformat(), "day"
    return None
