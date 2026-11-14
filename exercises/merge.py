"""Answers (one per source document) → events (one per real-world activity).

Two answers are the same event when
  - both carry a name and the names match after normalisation, and their dates
    overlap (3 days' slack) or one of them is undated; or
  - neither has a name, they share the kind, area and at least one country,
    and their dates overlap (or, undated, they were reported within 5 days).
An unnamed answer also joins a named event of the same area and countries
whose dates overlap it.
"""
import hashlib
import re
from datetime import date, timedelta

from . import regions

DAY = 86400000
OFFICIAL_KINDS = {"公式", "政府発表"}


GENERIC = re.compile(r"^(日[米豪比印英仏加独伊新韓]+|米[比豪韓]+|中露|露中)?(共同|合同|多国間|二国間|海上|統合|実動|指揮所)*(訓練|演習)(について)?$|"
                     r"^(joint|bilateral|multilateral|military|naval|combined|live[- ]fire)?\s*(exercises?|drills?|training)$", re.I)


def norm_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"^(exercise|operation|op\.?)\s+", "", s)
    s = re.sub(r"[「」『』\"'（）()・\s\-_/]+", "", s)
    return s


def _span(e):
    if not e.get("start"):
        return None
    a = date.fromisoformat(e["start"])
    b = date.fromisoformat(e.get("end") or e["start"])
    return a, b


def _overlap(a, b, slack=3):
    sa, sb = _span(a), _span(b)
    if sa is None or sb is None:
        return None
    d = timedelta(days=slack)
    return sa[0] <= sb[1] + d and sb[0] <= sa[1] + d


def _reported_close(a, b, days=5):
    return abs((a.get("lastReported") or 0) - (b.get("firstReported") or 0)) <= days * DAY or \
        abs((b.get("lastReported") or 0) - (a.get("firstReported") or 0)) <= days * DAY


def same_event(a, b) -> bool:
    ov = _overlap(a, b)
    if a.get("name") and b.get("name"):
        return norm_name(a["name"]) == norm_name(b["name"]) and ov is not False
    if a["region"] != b["region"] or not set(a["countries"]) & set(b["countries"]):
        return False
    if a.get("name") or b.get("name"):
        return ov is True
    if a["kind"] != b["kind"]:
        return False
    return ov is True or (ov is None and _reported_close(a, b))


def event_from(doc: dict, ans: dict) -> dict:
    at = doc.get("publishedAt") or doc.get("analyzedAt") or 0
    src = {"url": doc["url"], "title": doc.get("title") or "", "titleJa": doc.get("titleJa"),
           "source": doc.get("source"), "sourceName": doc.get("sourceName"), "kind": doc.get("kind") or "報道",
           "publishedAt": doc.get("publishedAt")}
    ans = dict(ans)
    if ans.get("name") and GENERIC.match(ans["name"].strip()):
        ans["name"] = None
    keep = {k: ans.get(k) for k in ("kind", "name", "titleJa", "countries", "region", "place", "forces", "ships",
                                     "aircraft", "personnel", "content", "start", "end", "status")}
    return {**keep, "sources": [src], "firstReported": at, "lastReported": at,
            "_official": src["kind"] in OFFICIAL_KINDS}


def _merge(a: dict, b: dict) -> dict:
    """Official answers win for each field; then the latest report."""
    def rank(e):
        return (1 if e["_official"] else 0, e["lastReported"])
    first, second = (a, b) if rank(a) >= rank(b) else (b, a)
    out = dict(first)
    for k in ("name", "place", "forces", "ships", "aircraft", "personnel", "content", "start", "end"):
        if out.get(k) in (None, ""):
            out[k] = second.get(k)
    if first.get("start") and second.get("start") and not first["_official"]:
        out["start"] = min(first["start"], second["start"])
        ends = [x for x in (first.get("end"), second.get("end")) if x]
        out["end"] = max(ends) if ends else None
    out["countries"] = list(dict.fromkeys(first["countries"] + second["countries"]))
    srcs = {s["url"]: s for s in second["sources"] + first["sources"]}
    out["sources"] = sorted(srcs.values(), key=lambda s: s.get("publishedAt") or 0, reverse=True)
    out["firstReported"] = min(a["firstReported"], b["firstReported"])
    out["lastReported"] = max(a["lastReported"], b["lastReported"])
    out["_official"] = a["_official"] or b["_official"]
    latest = a if a["lastReported"] >= b["lastReported"] else b
    out["status"] = latest["status"] if latest["status"] != "unknown" else out["status"]
    return out


def merge_all(items: list) -> list:
    events = []
    # Named first, so that unnamed reports join them.
    for e in sorted(items, key=lambda e: (0 if e.get("name") else 1, e["firstReported"])):
        for i, x in enumerate(events):
            if same_event(x, e):
                events[i] = _merge(x, e)
                break
        else:
            events.append(e)
    return events


def phase(e: dict, today: date) -> str:
    """now / next / done from the dates; from the reported status when undated."""
    sp = _span(e)
    if sp:
        if sp[1] < today and (e.get("end") or sp[0] < today - timedelta(days=1)):
            return "done"
        if sp[0] > today:
            return "next"
        if e.get("end") is None and sp[0] < today - timedelta(days=14) and e.get("status") == "completed":
            return "done"
        return "now"
    return {"planned": "next", "ongoing": "now", "completed": "done"}.get(e.get("status"), "now")


def confidence(e: dict) -> str:
    """高: an official release; 中: two or more different outlets; 低: one report."""
    if e["_official"]:
        return "高"
    outlets = {s.get("source") for s in e["sources"]}
    return "中" if len(outlets) >= 2 else "低"


def finish(e: dict, today: date) -> dict:
    out = {k: v for k, v in e.items() if not k.startswith("_")}
    anchor = e.get("start") or str(e["firstReported"] // DAY)
    key = "|".join([norm_name(e["name"]), anchor[:7]]) if e.get("name") else "|".join(
        [e["kind"], e["region"], ",".join(sorted(e["countries"])), anchor])
    out["id"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    out["regionName"] = regions.name(e["region"])
    r = regions.BY_ID.get(e["region"]) or regions.BY_ID["other"]
    out["lat"], out["lon"] = r[2], r[3]
    out["phase"] = phase(e, today)
    out["confidence"] = confidence(e)
    return out
