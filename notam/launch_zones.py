"""The launch zones of 世界情勢観測's 区域 screen: NOTAMs and navigational warnings
about rocket / missile launches that are in force or coming, grouped into
launches, with the inferred launch site and the likely launch vehicle.

Called from notam/launches.py (docs/data/launches/zones.json). Everything here
is a pure function of the notice history, Launch Library 2 records and `now`.

What is added to the history kept by launches.py:
  - the kind of each notice (ロケット打上げ・落下区域・打上げ関連・迎撃の備え),
    from its own text in English, Chinese or Japanese;
  - periods read from the text when NOTAM Watch gave none (NAVAREA XI and
    China MSA English: "130500Z TO 130900Z NOV" / "121400 UTC TO 122100UTC NOV";
    China MSA Chinese, Beijing time: "11月12日2200时至13日0500时");
  - NAVAREA XI cancellations ("CANCEL 0508/26", "... SUSPENDED");
  - the vehicle: the Launch Library 2 launch that falls in the window from the
    inferred site when there is one, otherwise the vehicles flown from that site
    whose usual orbits fit the inferred inclination.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

HORIZON_DAYS = 30
ISO = "%Y-%m-%dT%H:%M:%SZ"

# ---------- kinds ----------
INTERCEPT = re.compile(r"ANTI-?BALLISTIC|DESTRUCTION OF (AN OBJECT|THE ROCKET)", re.I)
DETOUR = re.compile(r"\bQARCH\b|\bALTN\s+RTE\b|\bALTERNATE\s+ROUTE|\bRTE\s+CLSD\b", re.I)
GROUND_FIRING = re.compile(r"\bFRNG\b")
RANGE_NAME = re.compile(r"MANEUVERING AREA|MANEUVER AREA|TRAINING AREA|FIRING RANGE|演習場", re.I)
FALL = re.compile(r"\bSTAGE\s+FALLING|FALLING AREA|FALL AREA OF[\s\S]{0,60}DEBRIS|SPACE DEBRIS|ROCKET DEBRIS|DEBRIS FALL|火箭残骸|残骸坠落|残骸落|落下物", re.I)
ROCKET = re.compile(r"ROCKET LAUNCH|SPACE LAUNCH|LAUNCH VEHICLE|MISSILE LAUNCH|火箭发射|航天发射|发射火箭|ロケット打上げ|ロケット発射", re.I)
RELATED = re.compile(r"AEROSPACE\s+(FLIGHT|FLT)\b|航天|火箭", re.I)
KINDS = {"rocket": "ロケット打上げ", "fall": "落下区域", "related": "打上げ関連", "intercept": "迎撃の備え"}


def kind_of(text: str) -> str | None:
    if not text:
        return None
    if GROUND_FIRING.search(text) and RANGE_NAME.search(text):
        return None
    if DETOUR.search(text):
        return None
    if INTERCEPT.search(text):
        return "intercept"
    if FALL.search(text):
        return "fall"
    if ROCKET.search(text):
        return "rocket"
    if RELATED.search(text):
        return "related"
    return None


# ---------- issuers ----------
FIR_COUNTRY = {"RP": "PH", "RJ": "JP", "RO": "JP", "RK": "KR", "RC": "TW", "ZK": "KP", "UH": "RU", "VV": "VN"}
MSA_PROVINCE = "琼浙鲁沪云冀辽闽粤桂津苏"
COUNTRY_JA = {"JP": "日本", "PH": "フィリピン", "CN": "中国", "KR": "韓国", "KP": "北朝鮮", "TW": "台湾", "RU": "ロシア",
              "VN": "ベトナム", "US": "米国"}


def issuer_of(n: dict) -> tuple[str | None, str]:
    """(country code, issuer in Japanese) of a stored notice."""
    nid, text = n.get("id", ""), n.get("text", "")
    if n.get("source") == "aviation":
        q = re.search(r"Q\)\s*([A-Z]{4})/", text)
        fir = q.group(1) if q else ""
        cc = "CN" if fir.startswith("Z") and not fir.startswith("ZK") else FIR_COUNTRY.get(fir[:2])
        return cc, f"{COUNTRY_JA.get(cc, cc or '')}（航空当局 {fir}）" if fir else (COUNTRY_JA.get(cc, "") or "航空当局")
    if nid.startswith("NAVAREA XI"):
        return "JP", "日本（海上保安庁）"
    if re.match(rf"[{MSA_PROVINCE}]航警", nid) or "MSA CHINA" in text:
        return "CN", "中国（海事局）"
    if nid.startswith("TW-"):
        return "TW", "台湾（漁業署）"
    return None, n.get("authority") or "発出者の記載なし"


# ---------- periods from the text ----------
MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
EN_PERIOD = re.compile(r"(\d{2})(\d{2})(\d{2})\s*(?:Z|UTC)\s*(?:([A-Z]{3})\s*)?TO\s*(\d{2})(\d{2})(\d{2})\s*(?:Z|UTC)\s*([A-Z]{3})")
CN_PERIOD = re.compile(r"(?:(\d{1,2})月)?(\d{1,2})日(\d{2})(\d{2})时至(?:(?:(\d{1,2})月)?(\d{1,2})日)?(\d{2})(\d{2})时")
CANCEL_SELF = re.compile(r"CANCEL THIS MSG\s+(\d{2})(\d{2})(\d{2})Z\s+([A-Z]{3})")
BJT = timedelta(hours=8)


def _near_year(month: int, day: int, h: int, m: int, ref: datetime, shift=timedelta(0)) -> datetime | None:
    """The date with this month/day closest to `ref` (notices are issued days before they apply)."""
    best = None
    for y in (ref.year - 1, ref.year, ref.year + 1):
        try:
            t = datetime(y, month, day, h % 24, m, tzinfo=timezone.utc) - shift + (timedelta(days=1) if h == 24 else timedelta(0))
        except ValueError:
            continue
        if best is None or abs(t - ref) < abs(best - ref):
            best = t
    return best


def text_periods(text: str, ref: datetime) -> list[tuple[datetime, datetime]]:
    out = []
    for m in EN_PERIOD.finditer(text):
        d1, h1, n1, mon1, d2, h2, n2, mon2 = m.groups()
        a = _near_year(MON.get(mon1 or mon2, 0) or 1, int(d1), int(h1), int(n1), ref) if (mon1 or mon2) in MON else None
        b = _near_year(MON[mon2], int(d2), int(h2), int(n2), ref) if mon2 in MON else None
        if a and b and b > a:
            out.append((a, b))
    for m in CN_PERIOD.finditer(text):
        mo1, d1, h1, n1, mo2, d2, h2, n2 = m.groups()
        mo1 = int(mo1) if mo1 else ref.month
        a = _near_year(mo1, int(d1), int(h1), int(n1), ref, BJT)
        b = _near_year(int(mo2) if mo2 else mo1, int(d2) if d2 else int(d1), int(h2), int(n2), ref, BJT)
        if a and b and b < a and not d2:   # "2200时至0500时": the next morning
            b += timedelta(days=1)
        if a and b and b > a:
            out.append((a, b))
    return out


def parse_iso(s: str | None) -> datetime | None:
    return datetime.strptime(s, ISO).replace(tzinfo=timezone.utc) if s else None


def fmt(t: datetime | None) -> str | None:
    return t.strftime(ISO) if t else None


STATED_SITE = re.compile(r"LAUNCH (?:SITE|PAD|POSITION)\s*:?\s*(\d{2})(\d{2})(\d{2}(?:\.\d+)?)N\s*(\d{3})(\d{2})(\d{2}(?:\.\d+)?)E")


def stated_site(text: str) -> list[float] | None:
    """The launch site when the notice gives it ("LAUNCH SITE : 330842N1262718E")."""
    m = STATED_SITE.search(text or "")
    if not m:
        return None
    a, b, c, d, e, f = map(float, m.groups())
    return [round(a + b / 60 + c / 3600, 4), round(d + e / 60 + f / 3600, 4)]


def normalise(n: dict) -> dict:
    """A copy of a stored notice with its kind, issuer and, when missing, periods read from the text."""
    text = n.get("text", "")
    ref = parse_iso(n.get("first_seen_utc")) or datetime.now(timezone.utc)
    cc, issuer = issuer_of(n)
    out = {**n, "kind": kind_of(text), "stated_site": stated_site(text), "country": cc, "issuer": issuer, "period_basis": "notam-watch"}
    alternate = bool(re.search(r"\bALTERNATE\b|备用|予備", text))
    if not n.get("start_utc"):
        ps = text_periods(text, ref)
        if ps:
            out["start_utc"], out["end_utc"] = fmt(min(p[0] for p in ps)), fmt(max(p[1] for p in ps))
            out["period_basis"] = "text"
            if len(ps) > 1:
                out["periods"] = [{"start_utc": fmt(a), "end_utc": fmt(b)} for a, b in ps]
        else:
            out["period_basis"] = "none"
    if alternate:
        out["alternate"] = True
        c = CANCEL_SELF.search(text)
        if c and c.group(4) in MON:
            t = _near_year(MON[c.group(4)], int(c.group(1)), int(c.group(2)), int(c.group(3)), ref)
            if t and (not out.get("end_utc") or t > parse_iso(out["end_utc"])):
                out["end_utc"] = fmt(t)   # the reserve days run until the notice cancels itself
    return out


def cancelled_keys(notices: list[dict]) -> set[str]:
    """NAVAREA XI notices cancelled by a later one ("CANCEL 0508/26") or that suspend themselves."""
    out = set()
    for n in notices:
        if not n.get("id", "").startswith("NAVAREA XI"):
            continue
        text = n.get("text", "")
        for num, yy in re.findall(r"CANCEL\s+(\d{4})/(\d{2})", text):
            out.add(f"maritime:NAVAREA XI {yy}-{num}")
        if re.search(r"\bSUSPENDED\b|\bCANCELLED\b", text):
            out.add(n["key"])
    return out


# ---------- vehicles ----------
# Vehicles flown from each site in recent years, with the orbits they usually go to (inclination, degrees).
# Sources: Launch Library 2 records (docs/data/launches/ll2.json) and the operators' announcements.
SITE_VEHICLES = {
    "文昌": [("長征5号", "低軌道・静止遷移・月惑星"), ("長征7号/7A号", "低軌道・静止遷移"), ("長征8号/8A号", "低軌道・太陽同期"),
             ("長征12号", "低軌道（千帆などの通信衛星群）")],
    "海陽（海上発射）": [("捷龍3号", "太陽同期"), ("長征11号（海上）", "低軌道・太陽同期"), ("引力1号", "太陽同期"),
                    ("谷神星1号（海上）", "太陽同期")],
    "酒泉": [("長征2C/2D/4C号", "低軌道・太陽同期"), ("長征2F号（有人）", "宇宙ステーション（約41度）"), ("快舟1A/11号", "太陽同期"),
             ("谷神星1号", "太陽同期"), ("朱雀2E/3号", "太陽同期"), ("力箭1号", "太陽同期"), ("双曲線1号", "太陽同期")],
    "太原": [("長征2C/2D号", "太陽同期"), ("長征4B/4C号", "太陽同期"), ("長征6号/6A号", "太陽同期・低軌道")],
    "西昌": [("長征3B/3C号", "静止遷移・北斗"), ("長征2C/2D号", "低軌道")],
    "西海（東倉里）": [("千里馬1型（衛星打上げ）", "太陽同期・極軌道"), ("弾道ミサイル", "—")],
    "平壌周辺": [("弾道ミサイル", "—")],
    "種子島": [("H3", "太陽同期・静止遷移"), ("H-IIA", "太陽同期・静止遷移")],
    "内之浦": [("イプシロン", "太陽同期・低軌道")],
    "羅老": [("ヌリ", "太陽同期")],
    "済州沖（海上発射）": [("固体燃料の宇宙発射体（国防科学研究所）", "低軌道・太陽同期")],
    "ボストーチヌイ": [("ソユーズ2", "太陽同期・低軌道"), ("アンガラ", "低軌道")],
}


def orbit_hint(inclination: float | None) -> str | None:
    if inclination is None:
        return None
    if inclination >= 95:
        return "太陽同期軌道"
    if inclination >= 80:
        return "極軌道"
    if 38 <= inclination <= 44:
        return "低軌道（宇宙ステーションと同じ約41度）"
    if inclination <= 30:
        return "低傾斜（静止遷移など）"
    return "低軌道"


ROCKET_JA = [(r"^Long March ([\w/]+)(.*)$", r"長征\1号\2"), (r"^Kinetica[- ](\d)", r"力箭\1号"), (r"^Ceres-(\d)(.*)$", r"谷神星\1号\2"),
             (r"^Zhuque-(\d)(.*)$", r"朱雀\1号\2"), (r"^Kuaizhou[- ](\w+)", r"快舟\1"), (r"^Jielong-3", "捷龍3号"),
             (r"^Gravity-(\d)", r"引力\1号"), (r"^Hyperbola-(\d)(.*)$", r"双曲線\1号\2"), (r"^KSLV-2 Nuri", "ヌリ"),
             (r"^Mir$", "固体燃料の宇宙発射体（国防科学研究所）"), (r"^Chollima-1", "千里馬1型")]


def rocket_ja(name: str | None) -> str | None:
    for pat, rep in ROCKET_JA:
        if name and re.match(pat, name):
            return re.sub(pat, rep, name)
    return name


def vehicle_of(ev: dict) -> dict | None:
    rec = ev.get("recorded_launches") or []
    if rec:
        r = rec[0]
        return {"basis": "ll2", "name": rocket_ja(r.get("rocket")), "rocket_ll2": r.get("rocket"), "mission": r.get("name"), "orbit": r.get("orbit"),
                "status": r.get("status"), "net_utc": r.get("net_utc"), "pad": r.get("pad")}
    site = ev.get("site")
    if not site or site not in SITE_VEHICLES:
        return None
    return {"basis": "site", "candidates": [{"name": a, "orbit": b} for a, b in SITE_VEHICLES[site]]}


# ---------- zones.json ----------
def _window(e: dict) -> tuple[datetime, datetime]:
    f = "%Y-%m-%dT%H:%MZ"
    return (datetime.strptime(e["window_start_utc"], f).replace(tzinfo=timezone.utc),
            datetime.strptime(e["window_end_utc"], f).replace(tzinfo=timezone.utc))


def _mission(e: dict) -> str | None:
    rec = e.get("recorded_launches") or []
    return rec[0].get("name") if rec else None


def _kinds(e: dict, notices: dict[str, dict]) -> frozenset:
    return frozenset(notices[k].get("kind") for k in e["notices"])


def merge_attempts(events: list[dict], notices: dict[str, dict]) -> list[dict]:
    """One launch, one entry. A launch put off day after day is notified again for each attempt (済州沖 11/10, 11/11,
    11/13, each with reserve days to 11/30); these are the same launch when the site is the same, the windows overlap and
    the launch schedule does not name two different launches. Notices with no site are one entry when they are of the
    same kinds over the same window (interception airspace: over overlapping windows)."""
    out: list[dict] = []
    for e in sorted(events, key=lambda x: x["window_start_utc"]):
        a, b = _window(e)
        for g in out:
            ga, gb = _window(g)
            if a > gb or b < ga:
                continue
            if e.get("site") and g.get("site"):
                m1, m2 = _mission(e), _mission(g)
                same = e["site"] == g["site"] and not (m1 and m2 and m1 != m2)
            elif not e.get("site") and not g.get("site"):
                k = _kinds(e, notices)
                same = k == _kinds(g, notices) and (k == {"intercept"} or (a, b) == (ga, gb))
            else:
                same = False
            if same:
                g["notices"] = g["notices"] + [k for k in e["notices"] if k not in g["notices"]]
                g["areas"] = g.get("areas", []) + e.get("areas", [])
                g["window_start_utc"] = min(g["window_start_utc"], e["window_start_utc"])
                g["window_end_utc"] = max(g["window_end_utc"], e["window_end_utc"])
                if not g.get("recorded_launches") and e.get("recorded_launches"):
                    g["recorded_launches"] = e["recorded_launches"]
                for k in ("azimuth_deg", "inclination_deg", "bearing_spread_deg", "distance_km"):
                    if k not in g and k in e:
                        g[k] = e[k]
                break
        else:
            out.append(dict(e))
    return out


def build(events: list[dict], notices: dict[str, dict], sites: dict, now: datetime) -> dict:
    """Launches in force at `now` or starting within HORIZON_DAYS, each with its notices."""
    horizon = now + timedelta(days=HORIZON_DAYS)
    out = []
    live = [e for e in events if _window(e)[1] >= now and _window(e)[0] <= horizon]
    for e in merge_attempts(live, notices):
        ns = []
        for k in e["notices"]:
            n = notices[k]
            polys = [s["points"] for s in n.get("shapes", []) if s.get("kind") == "polygon" and len(s.get("points", [])) >= 3]
            ns.append({"key": k, "id": n["id"], "dataset": n["source"], "kind": n["kind"], "issuer": n["issuer"],
                       "country": n["country"], "start_utc": n["start_utc"], "end_utc": n["end_utc"],
                       "period_basis": n["period_basis"], **({"periods": n["periods"]} if n.get("periods") else {}),
                       **({"alternate": True} if n.get("alternate") else {}), "polygons": polys,
                       "circles": [s for s in n.get("shapes", []) if s.get("kind") == "circle"], "text": n.get("text", "")})
        site = None
        if e.get("method") == "stated":
            site = {"name": e.get("site") or "告知に記載の座標", "lat": e["site_latlon"][0], "lon": e["site_latlon"][1],
                    "country": e.get("country"), "method": "stated",
                    **{k: e[k] for k in ("azimuth_deg", "inclination_deg") if k in e}}
        elif e.get("site") and e.get("method") in ("aligned", "nearest-only"):
            lat, lon, cc = sites[e["site"]]
            site = {"name": e["site"], "lat": lat, "lon": lon, "country": cc, "method": e["method"],
                    **{k: e[k] for k in ("azimuth_deg", "bearing_spread_deg", "inclination_deg", "distance_km") if k in e}}
        out.append({
            "id": f"{e['window_start_utc']}:{(site or {}).get('name') or '+'.join(sorted({str(n['kind']) for n in ns}))}",
            "window_start_utc": e["window_start_utc"], "window_end_utc": e["window_end_utc"],
            "site": site, "orbit": orbit_hint(e.get("inclination_deg")), "vehicle": vehicle_of(e), "notices": ns,
        })
    out.sort(key=lambda x: x["window_start_utc"])
    return {"generated_utc": now.strftime(ISO), "horizon_days": HORIZON_DAYS, "kinds": KINDS, "events": out}
