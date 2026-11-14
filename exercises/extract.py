"""One model call per source document, then checks against the source text.

The model's answer is never published as is: names must appear in the text,
dates need words quoted from the text, and counts (ships, aircraft, people)
must appear as numbers in the text. The area is one id from regions.py.
"""
import json
import re
import urllib.request

from . import regions

KINDS = ["exercise", "live_fire", "deployment"]
STATUSES = ["planned", "ongoing", "completed", "unknown"]

SYSTEM = f"""You read one news article or official press release and extract a concrete military exercise, drill, training event, live-fire event or large deployment of armed forces, for a Japanese situational-awareness site. Answer with JSON only.

relevant=true ONLY when the text reports a specific, real-world event of this kind with participants and a place or time: a named or unnamed exercise or drill, joint training between countries, live-fire or missile firing drills, or a large naval/air deployment that the text itself presents as an exercise, drill or show of force.
relevant=false for: port visits alone, strait transits alone, ceremonies, individual or classroom training, procurement, policy, budgets, opinion, and texts that only mention an exercise in passing or as background. If the text reports several events, pick the main one.

Fields (Japanese for text fields):
- kind: exercise (exercise, drill, joint training), live_fire (firing or missile drills), deployment (large deployment presented as a drill or show of force).
- name: the exercise's own name exactly as written in the text (e.g. "Keen Sword 27", "Balikatan 2026", "海峡雷霆-2025A"); null if none.
- titleJa: a short factual Japanese headline (who, what, where).
- countries: participating countries in Japanese (日本, 米国, 中国, ロシア, 韓国, 台湾, フィリピン, 豪州, インド, …). Not countries only mentioned.
- region: exactly one id from this list (the main area of the event):
{regions.prompt_list()}
- place: the place in Japanese as specific as the text states (e.g. 日向灘, 嘉手納基地); null if not stated.
- forces: participating units, ships, aircraft and people in Japanese, as the text states them (e.g. "海自 艦艇16隻（掃海艦3・掃海艇8）、MCH-101×2／米海軍 掃海艦1"); null if not stated.
- ships, aircraft, personnel: totals the text states as numbers; null when the text gives no number. Never estimate.
- content: what is trained or done, in Japanese, 1–2 sentences, facts from the text only; null if not stated.
- start, end: YYYY-MM-DD only when the text states the event's dates (not the publication date). Resolve relative words ("Monday", "this week", 令和8年11月16日) using the publication date. Same date for both for a single day; end=null when unknown.
- dateEvidence: the exact words in the text that give the dates (copy them), or null.
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


def user_message(doc: dict, text: str, published_day: str) -> str:
    return (f"Publication date: {published_day}\nSource: {doc.get('sourceName') or doc.get('source')}\n"
            f"Headline: {doc.get('title')}\nSummary: {doc.get('summary') or ''}\n"
            f"Text:\n{text or '(body unavailable; use the headline and summary only)'}")


def ask(server: str, doc: dict, text: str, published_day: str, timeout: int = 900) -> dict:
    body = {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user_message(doc, text, published_day)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0, "max_tokens": 700,
        "response_format": {"type": "json_object", "schema": SCHEMA},
    }
    req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode("utf-8"))
    content = res["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end + 1])


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


def validate(value: dict, doc: dict, text: str, published_day: str) -> dict:
    """Keeps only what the source supports. Returns {'relevant': False} when the
    answer is not a concrete event."""
    v = value if isinstance(value, dict) else {}
    if v.get("relevant") is not True or v.get("kind") not in KINDS:
        return {"relevant": False}
    source = " ".join([doc.get("title") or "", doc.get("summary") or "", text or ""])
    name = v.get("name") if isinstance(v.get("name"), str) and _contains(source, v["name"]) else None
    start, end = _iso(v.get("start")), _iso(v.get("end"))
    evidence = v.get("dateEvidence") if isinstance(v.get("dateEvidence"), str) else None
    if not evidence or not _contains(source, evidence[:60]):
        start = end = evidence = None
    for d in (start, end):
        if d and published_day and _days_apart(d, published_day) > 400:
            start = end = evidence = None
    if start and end and end < start:
        end = None
    counts = {k: (v.get(k) if _number_in(source, v.get(k)) else None) for k in ("ships", "aircraft", "personnel")}

    def s(key, n):
        x = v.get(key)
        return x.strip()[:n] if isinstance(x, str) and x.strip() else None

    countries = []
    for c in v.get("countries") or []:
        if isinstance(c, str) and c.strip() and c.strip() not in countries:
            countries.append(c.strip()[:20])
    return {
        "relevant": True, "kind": v["kind"], "name": name,
        "titleJa": s("titleJa", 120) or (doc.get("titleJa") or doc.get("title") or "")[:120],
        "countries": countries[:12],
        "region": v.get("region") if v.get("region") in regions.IDS else "other",
        "place": s("place", 80), "forces": s("forces", 400), **counts,
        "content": s("content", 300), "start": start, "end": end,
        "dateEvidence": evidence[:200] if evidence else None,
        "status": v.get("status") if v.get("status") in STATUSES else "unknown",
    }
