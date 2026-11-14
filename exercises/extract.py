"""One model call per source document, then checks against the source text.

The model's answer is never published as is: names must appear in the text,
dates need words quoted from the text, and counts (ships, aircraft, people)
must appear as numbers in the text. The area is one id from regions.py.
"""
import json
import re
import urllib.request

from . import dates, regions

KINDS = ["exercise", "live_fire", "deployment"]
STATUSES = ["planned", "ongoing", "completed", "unknown"]

SYSTEM = f"""You read one news article or official press release and extract a concrete military exercise, drill, training event, live-fire event or large deployment of armed forces, for a Japanese situational-awareness site. Answer with JSON only.

relevant=true ONLY when the text reports a specific, real-world event of this kind with participants and a place or time: a named or unnamed exercise or drill, joint training between countries, live-fire or missile firing drills, or a large naval/air deployment that the text itself presents as an exercise, drill or show of force.
relevant=false for: port visits alone, strait transits alone, ceremonies, individual or classroom training, procurement, policy, budgets, opinion, and texts that only mention an exercise in passing or as background. If the text reports several events, pick the main one.
When relevant=false, answer only {{"relevant": false}} with every other field null, "" or [] (region "other", status "unknown"): write nothing else.

Fields (Japanese for text fields):
- kind: exercise (exercise, drill, joint training), live_fire (firing or missile drills), deployment (large deployment presented as a drill or show of force).
- name: the exercise's own name exactly as written in the text, often in quotes or with a year; null if none. Not the name of an earlier exercise the text only recalls, and not a description.
- titleJa: a short factual Japanese headline (who, what, where), at most 30 characters.
- countries: participating countries, always in Japanese katakana/kanji as listed here (日本, 米国, 中国, ロシア, 韓国, 北朝鮮, 台湾, フィリピン, 豪州, インド, インドネシア, ベトナム, パキスタン, …), even when the text is Chinese or English. Not countries only mentioned. In Chinese, 中巴 is China–Pakistan, 印尼 is Indonesia (not India), 中越 is China–Vietnam.
- region: exactly one id from this list (the main area of the event; a sea named in the text wins over a nearby island):
{regions.prompt_list()}
- place: the place in Japanese as specific as the text states (e.g. 日向灘, 嘉手納基地); null if not stated.
- forces: the participating units, ships, aircraft and people of THIS text, in Japanese, concise, at most 120 characters; null if the text does not state them. Never fill it from anything but this text.
- ships, aircraft, personnel: totals the text states as numbers; null when the text gives no number. Never estimate.
- content: what is trained or done, in Japanese, one sentence of at most 60 characters, facts from the text only; null if not stated.
- start, end: YYYY-MM-DD only when the text states the event's dates (not the publication date or the page's time stamp). Same date for both for a single day; end=null when unknown.
- dateEvidence: the exact words in the text that state when the event takes place, copied as written (e.g. a date, a span of dates, a part of a month), or null. Never the page's publication time.
- status: planned (announced for the future), ongoing (underway at publication), completed (finished), unknown."""

SCHEMA = {
    "type": "object",
    "required": ["relevant", "kind", "name", "titleJa", "countries", "region", "place", "forces", "ships",
                 "aircraft", "personnel", "content", "start", "end", "dateEvidence", "status"],
    "properties": {
        "relevant": {"type": "boolean"},
        "kind": {"enum": KINDS + [None]},
        "name": {"type": ["string", "null"]},
        "titleJa": {"type": "string"},
        "countries": {"type": "array", "items": {"type": "string"}},
        "region": {"type": "string", "enum": regions.IDS},
        "place": {"type": ["string", "null"]},
        "forces": {"type": ["string", "null"]},
        "ships": {"type": ["integer", "null"]},
        "aircraft": {"type": ["integer", "null"]},
        "personnel": {"type": ["integer", "null"]},
        "content": {"type": ["string", "null"]},
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
        "dateEvidence": {"type": ["string", "null"]},
        "status": {"type": "string", "enum": STATUSES},
    },
}

_Z2H = str.maketrans("０１２３４５６７８９，", "0123456789,")

# A country stays only when the text names it (the model has mixed up 中巴 → Bahrain and 印尼 → India).
COUNTRY_WORDS = {
    "日本": r"日本|japan|自衛隊|jsdf|jmsdf|jgsdf|jasdf|海自|陸自|空自|日米|日豪|日比|日印|日英|日仏|日韓|日方",
    "米国": r"米国|米軍|米海|米空|米陸|米海兵|アメリカ|united states|\bu\.?s\.?\b|\bus (navy|army|air|marine)|usmc|american|美国|美军|美方|日米|米比|米韓|米豪|indopacom|7th fleet",
    "中国": r"中国|中华|中華|china|chinese|\bpla\b|plan\b|解放军|解放軍|中方|中俄|中巴|中越|中印|中泰|中柬|战区|戦区|武警|人民武装警察|中共",
    "ロシア": r"ロシア|russia|露|俄|россия|российск",
    "韓国": r"韓国|south korea|\brok\b|韩国|米韓|日米韓|대한민국",
    "北朝鮮": r"北朝鮮|north korea|dprk|朝鲜|朝鮮人民軍",
    "台湾": r"台湾|taiwan|台灣|國軍|国軍",
    "フィリピン": r"フィリピン|philippine|菲律宾|日米比|米比|日比|比海軍|比軍|\bafp\b",
    "豪州": r"豪州|オーストラリア|australia|澳大利亚|日豪|米豪|豪海軍|豪軍|\badf\b",
    "インド": r"インド(?!ネシア|洋)|\bindia\b|indian (navy|army|air force|armed)|印度(?!尼|洋)|日印",
    "インドネシア": r"インドネシア|indonesia|印度尼西亚|印尼",
    "ベトナム": r"ベトナム|vietnam|viet nam|越南|中越",
    "パキスタン": r"パキスタン|pakistan|巴基斯坦|中巴",
    "バーレーン": r"バーレーン|bahrain|巴林",
    "タイ": r"タイ王国|タイ軍|タイ海軍|日タイ|米タイ|thailand|\bthai\b|泰国|中泰",
    "マレーシア": r"マレーシア|malaysia|马来西亚",
    "シンガポール": r"シンガポール|singapore|新加坡",
    "カンボジア": r"カンボジア|cambodia|柬埔寨|中柬",
    "英国": r"英国|イギリス|british|united kingdom|royal navy|日英|英海軍|英軍",
    "フランス": r"フランス|france|french|法国|日仏|仏海軍|仏軍",
    "ドイツ": r"ドイツ|germany|german|德国",
    "カナダ": r"カナダ|canada|canadian|加拿大",
    "ニュージーランド": r"ニュージーランド|new zealand|新西兰",
    "モンゴル": r"モンゴル|mongolia|蒙古国",
}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "eleven": 11, "twelve": 12, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9, "十": 10}


def user_message(doc: dict, text: str, published_day: str) -> str:
    return (f"Publication date: {published_day}\nSource: {doc.get('sourceName') or doc.get('source')}\n"
            f"Headline: {doc.get('title')}\nSummary: {doc.get('summary') or ''}\n"
            f"Text:\n{text or '(body unavailable; use the headline and summary only)'}")


def ask(server: str, doc: dict, text: str, published_day: str, timeout: int = 900) -> dict:
    body = {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_message(doc, text, published_day)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0, "max_tokens": 700, "cache_prompt": True,
        "response_format": {"type": "json_object", "schema": SCHEMA},
    }
    req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode("utf-8"))
    content = res["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}")
    out = json.loads(content[start:end + 1])
    t = res.get("timings") or {}
    ask.last_timings = {k: t.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")}
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").translate(_Z2H)).lower()


def _contains(source: str, needle: str) -> bool:
    n = _norm(needle)
    return bool(n) and n in _norm(source)


def _number_in(source: str, n) -> bool:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        return False
    digits = re.sub(r"[,\s]", "", source.translate(_Z2H))
    return re.search(rf"(?<!\d){n}(?!\d)", digits) is not None


def _iso(v):
    return v if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) else None


def _days_apart(a: str, b: str) -> int:
    from datetime import date
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


_QUOTED = re.compile(r"[“「『\"']([^”」』\"']{2,40}?[-－‐ ]?(?:20\d{2}|\d{2})[A-Za-z]?)[”」』\"']")


def quoted_name(title: str):
    """A name the headline quotes with a year in it: 中俄“海上联合-2026”联合演习 → 海上联合-2026."""
    m = _QUOTED.search(title)
    return m.group(1).strip() if m else None


def canonical_country(name):
    """The table's Japanese name for a country the model wrote in another form (俄罗斯, 越南, Russia → ロシア, ベトナム)."""
    if name in COUNTRY_WORDS:
        return name
    for key, pat in COUNTRY_WORDS.items():
        if re.fullmatch(f"(?:{pat})", name, re.I) or re.fullmatch(r"[^\s]{1,8}", name) and re.search(pat, name, re.I) and len(name) <= 8:
            return key
    return name


def supported_countries(countries, source):
    out = []
    for c in (canonical_country(x) for x in countries):
        pat = COUNTRY_WORDS.get(c)
        ok = re.search(pat, source, re.I) if pat else _contains(source, c)
        if ok and c not in out:
            out.append(c)
    return out


def _numbers(text):
    return {int(x) for x in re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d{3})", "", (text or "").translate(_Z2H)))}


def numbers_supported(text, source):
    """Every number of 3 or more in `text` is a number of the source (digits or a number word)."""
    have = _numbers(source) | {v for w, v in _NUM_WORDS.items() if re.search(rf"\b{w}\b" if w.isascii() else w, source, re.I)}
    return all(n in have for n in _numbers(text) if n >= 3)


_GENERIC_JA = re.compile(r"^(共同|合同|多国間|二国間|海上|統合|実動|指揮所|反恐|联合|聯合|軍事)*(訓練|演習|演习|训练|演練)(について)?$")
_ACTORS = re.compile(r"日本軍?|米軍?|アメリカ軍?|中国軍?|ロシア軍?|露|中|日|自衛隊|海自|陸自|空自|[米日中露韓豪比印英仏加独伊新]|軍|・|、|\s")
_NOT_IN_NAMES = {"in", "with", "for", "during", "at", "held", "hold", "holds", "conduct", "conducts", "conducted", "near"}


def plausible_name(name, year=None):
    """A real exercise name, not a description or an earlier exercise."""
    if not name:
        return None
    if re.search(r"\d{4}-\d{2}-\d{2}", name):
        return None
    if re.fullmatch(r"[\x00-\x7f]+", name):
        words = name.split()
        if not re.search(r"[A-Z0-9]", name) or len(words) > 6 or any(w.lower() in _NOT_IN_NAMES for w in words):
            return None
    if _GENERIC_JA.match(_ACTORS.sub("", name)):
        return None
    m = re.search(r"(20\d{2})", name)
    if m and year and int(m.group(1)) != int(year):
        return None
    return name


def validate(value: dict, doc: dict, text: str, published_day: str) -> dict:
    """Keeps only what the source supports. Returns {'relevant': False} when the
    answer is not a concrete event."""
    v = value if isinstance(value, dict) else {}
    if v.get("relevant") is not True or v.get("kind") not in KINDS:
        return {"relevant": False}
    source = " ".join([doc.get("title") or "", doc.get("summary") or "", text or ""])
    name = v.get("name") if isinstance(v.get("name"), str) and _contains(source, v["name"]) else None
    if not name:
        name = quoted_name(doc.get("title") or "")
    # Dates are read from the quoted words by rule (exercises/dates.py); the model's own dates are not used.
    evidence = v.get("dateEvidence") if isinstance(v.get("dateEvidence"), str) and v["dateEvidence"].strip() else None
    if evidence and not _contains(source, evidence[:60]):
        evidence = None
    parsed = dates.parse(evidence, published_day) if evidence else None
    start, end, precision = parsed if parsed else (None, None, None)
    if start and published_day and _days_apart(start, published_day) > 400:
        start = end = precision = None
    counts = {k: (v.get(k) if _number_in(source, v.get(k)) else None) for k in ("ships", "aircraft", "personnel")}
    name = plausible_name(name, (start or published_day or "")[:4] or None)

    def s(key, n):
        x = v.get(key)
        return x.strip()[:n] if isinstance(x, str) and x.strip() else None

    countries = supported_countries([c.strip()[:20] for c in (v.get("countries") or []) if isinstance(c, str) and c.strip()], source)
    forces = s("forces", 400)
    if forces and not numbers_supported(forces, source):
        forces = None
    content = s("content", 300)
    if content and not numbers_supported(content, source):
        content = None
    place = s("place", 80)
    model_region = v.get("region") if v.get("region") in regions.IDS else "other"
    region = (regions.from_place(place) or regions.from_place(doc.get("title") or "")
              or (model_region if regions.supported(model_region, source) else "other"))
    return {
        "relevant": True, "kind": v["kind"], "name": name,
        "titleJa": s("titleJa", 120) or (doc.get("titleJa") or doc.get("title") or "")[:120],
        "countries": countries[:12],
        "region": region,
        "place": place, "forces": forces, **counts,
        "content": content, "start": start, "end": end,
        "dateText": evidence[:120] if evidence else None, "datePrecision": precision,
        "status": v.get("status") if v.get("status") in STATUSES else "unknown",
    }
