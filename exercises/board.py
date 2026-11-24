"""The 演習 screen's one list (board.json): what is under way now or starts within BOARD_DAYS days, from three kinds of
material, in time order.

Owner, 2026-11-23/24: the screen shows only exercises under way or within a month; confirmed ones and expected ones in
one list, an expected one marked with the single word 見込み; the screen is kept simple, the work behind it is not.

- event:    an exercise read from official releases and reports (current.json). Shown when it is under way, or starts
            within BOARD_DAYS days. Undated ones, ones outside the East Asian and Western Pacific areas, and reports of
            accidents or disaster relief are left out (they stay in the yearly files and the history search).
            The firing/exercise notices of any issuer that overlap it in time and lie within NEAR_KM of its area are
            attached as supporting detail.
- notice:   China's, Russia's and North Korea's own notices of firing, exercises and military activity (notam/military.json),
            grouped by issuer, overlapping time and nearby areas. Other countries' routine range notices (Taiwan alone has
            ~500) are not listed on their own.
- forecast: recurring exercises expected within the month (forecast.json, exercises/forecast.py), with their basis line.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BOARD_DAYS = 31
NEAR_KM = 500
GROUP_KM = 300
OWN = {"CN": "中国", "RU": "ロシア", "KP": "北朝鮮"}
KINDS = {"firing", "exercise", "military"}
FAR = {"hawaii", "australia-north", "australia", "indonesia", "southeast-asia", "bay-of-bengal", "india", "indian-ocean",
       "russia", "north-pacific", "other"}
NOT_EXERCISE = re.compile(r"事故|死傷|殉職|墜落|災害派遣|防災|医療活動|救難|捜索|爆発事故")


def _d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _centre(n: dict):
    pts = [p for s in n.get("area", {}).get("shapes", []) for p in s.get("points", [])]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _span(n: dict):
    v = n.get("valid", {})
    return _d(v.get("start_utc")), _d(v.get("end_utc"))


def _notice_out(n: dict) -> dict:
    return {"key": n["key"], "id": n["id"], "issuer": n["issuer"].get("name") or n["issuer"].get("code"), "country": n["issuer"].get("country"),
            "category": n["category"], "start_utc": n["valid"].get("start_utc"), "end_utc": n["valid"].get("end_utc"),
            "polygons": [s["points"] for s in n["area"]["shapes"] if s.get("kind") == "polygon"],
            "text": n.get("original", {}).get("text", "")[:1200], "url": n.get("source", {}).get("url")}


def in_board(start, end, today) -> bool:
    if start is None:
        return False
    return start <= today + timedelta(days=BOARD_DAYS) and (end or start) >= today


def events_part(current: dict, notices: list[dict], today: date) -> list[dict]:
    out = []
    for e in current.get("events", []):
        s, en = _d(e.get("start")), _d(e.get("end"))
        # An exercise reported as under way with no end given is still under way today (current.json's phase).
        ongoing = en is None and e.get("phase") == "now"
        if ongoing:
            en = today
        if not in_board(s, en, today) or e.get("region") in FAR or e.get("lat") is None:
            continue
        if NOT_EXERCISE.search(" ".join(str(e.get(k) or "") for k in ("name", "titleJa", "content"))):
            continue
        at = (e["lat"], e["lon"])
        near = []
        for n in notices:
            c, (ns, ne) = _centre(n), _span(n)
            if c and ns and _km(at, c) <= NEAR_KM and ns <= (en or s) and (ne or ns) >= s:
                near.append(_notice_out(n))
        out.append({"id": "event:" + e["id"], "kind": "event", "name": e.get("name") or e.get("titleJa"), "countries": e.get("countries", []),
                    "start": e.get("start"), "end": e.get("end"), "lat": e["lat"], "lon": e["lon"], "region": e.get("regionName"),
                    "confidence": e.get("confidence"), "confidenceBasis": e.get("confidenceBasis"), "content": e.get("content"),
                    "forces": e.get("forces"), "sources": e.get("sources", [])[:8], "notices": near[:12], "event_id": e["id"],
                    **({"ongoing": True} if ongoing else {})})
    return out


def notices_part(notices: list[dict], today: date) -> list[dict]:
    own = [n for n in notices if n["issuer"].get("country") in OWN and n["category"] in KINDS and _centre(n)]
    own = [n for n in own if in_board(*_span(n), today) or (_span(n)[0] is None and n["valid"].get("status") == "active")]
    groups: list[list[dict]] = []
    for n in sorted(own, key=lambda n: (n["valid"].get("start_utc") or "")):
        c, (s, e) = _centre(n), _span(n)
        for g in groups:
            g0 = g[0]
            gs, ge = _span(g0)
            if g0["issuer"].get("name") == n["issuer"].get("name") and _km(_centre(g0), c) <= GROUP_KM and \
               (s is None or gs is None or (s <= (ge or gs) + timedelta(days=1) and (e or s) >= gs - timedelta(days=1))):
                g.append(n)
                break
        else:
            groups.append([n])
    out = []
    for g in groups:
        cs = [_centre(n) for n in g]
        starts = [x for x in (_span(n)[0] for n in g) if x]
        ends = [x for x in (_span(n)[1] for n in g) if x]
        cc = g[0]["issuer"].get("country")
        cats = sorted({n["category"] for n in g}, key=["firing", "exercise", "military"].index)
        word = {"firing": "射撃", "exercise": "演習", "military": "軍事活動"}[cats[0]]
        out.append({"id": "notice:" + g[0]["key"], "kind": "notice", "name": f"{word}（{g[0]['issuer'].get('name') or g[0]['issuer'].get('code') or OWN[cc]}）",
                    "countries": [OWN[cc]], "start": min(starts).isoformat() if starts else None, "end": max(ends).isoformat() if ends else None,
                    "lat": round(sum(c[0] for c in cs) / len(cs), 2), "lon": round(sum(c[1] for c in cs) / len(cs), 2),
                    "region": None, "notices": [_notice_out(n) for n in g][:12]})
    return out


def forecast_part(fc: dict) -> list[dict]:
    from exercises.forecast import COMPILED
    from exercises.regions import REGIONS
    return [{"id": f"forecast:{c['series']}:{c['year']}", "kind": "forecast", "name": c["name"], "countries": c["countries"].split("・"),
             "start": c["window_start"], "end": c["window_end"], "lat": None, "lon": None, "region": c["area"],
             "basis": c["basis"], "probability": c["probability"]} for c in fc.get("calls", [])]


def build(current: dict, military: dict, fc: dict, today: date) -> dict:
    notices = military.get("items", [])
    items = events_part(current, notices, today) + notices_part(notices, today) + forecast_part(fc)
    items.sort(key=lambda x: (x["start"] or "", {"event": 0, "notice": 1, "forecast": 2}[x["kind"]], x["id"]))
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "today": today.isoformat(), "days": BOARD_DAYS,
            "counts": {k: sum(1 for x in items if x["kind"] == k) for k in ("event", "notice", "forecast")},
            "rules": {"event": "公式の発表・報道から読んだ演習。実施中か今後31日に始まるもの。日付不明・東アジアと西太平洋の外・事故や災害対応は除く。期間が重なり500km以内の告知を添える",
                      "notice": "中国・ロシア・北朝鮮が出した射撃・演習・軍事活動の告知（NOTAM・航行警報）。発出元・期間・場所の近いものをまとめる",
                      "forecast": "定例の演習の見込み（forecast.json）。根拠は過去の記録"},
            "items": items}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", default=str(root / "docs" / "data" / "exercises"))
    ap.add_argument("--military", default=str(root / "docs" / "data" / "notam" / "military.json"))
    ap.add_argument("--today")
    a = ap.parse_args()
    out = Path(a.out)
    read = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else {}
    today = date.fromisoformat(a.today) if a.today else datetime.now(timezone(timedelta(hours=9))).date()
    board = build(read(out / "current.json"), read(a.military), read(out / "forecast.json"), today)
    (out / "board.json").write_text(json.dumps(board, ensure_ascii=False, indent=1) + "\n")
    print(f"::notice title=exercises board::{board['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
