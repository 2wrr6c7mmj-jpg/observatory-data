#!/usr/bin/env python3
"""Parser for JSO press releases on foreign naval vessel / aircraft movements.

parse_release(release_id, pub_date, text) -> dict
  release: {domain, nation, has_track_map, corrects}
  sightings: [ {seq, observed_date, observed_date_end, observed_time, ref_point, bearing_jp,
                distance_km, lat, lon, heading_jp, heading_deg, strait, strait_dir,
                waters_from, waters_to, activity, activity_count, same_as_text, unit_count,
                sentence, units:[{nation,kind,class_jp,class_key,hull_no,name,count}] } ]
  jp_units: [ {unit_kind, unit_name, parent_unit, base} ]
"""
import csv, json, math, os, re, unicodedata, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- reference data
REF_POINTS = {}
with open(os.path.join(HERE, "ref", "ref_points.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        REF_POINTS[row["name_jp"]] = (float(row["lat"]), float(row["lon"]))
        for a in (row.get("aliases") or "").split("|"):
            if a.strip():
                REF_POINTS[a.strip()] = (float(row["lat"]), float(row["lon"]))

with open(os.path.join(HERE, "ref", "straits.json"), encoding="utf-8") as f:
    STRAITS = json.load(f)
PREF_OPT = r"(?:（[^）]{2,6}）)?"
def _pat(pt):
    return re.sub(r"(島|台湾|岬|崎|半島|諸島|列島|海峡)(?=と|の|$|\(|\[|間|\))", lambda m: m.group(1) + PREF_OPT, pt)
for s in STRAITS:
    s["_rx"] = re.compile("|".join(_pat(x) for x in s["patterns"]))

# ---------------------------------------------------------------- normalisation
_Z2H = {ord(c): ord(c) - 0xFEE0 for c in "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}
_Z2H.update({ord("－"): ord("-"), ord("ー"): None, ord("．"): ord("."), ord("～"): ord("~"), ord("〜"): ord("~")})
_KEEP_LONG = "ー"  # we must NOT strip ー inside katakana; handle separately

def normalize(text: str) -> str:
    """Full-width digits/latin -> ASCII, drop line breaks & spaces (Japanese text has no word spaces)."""
    t = text.replace("\r", "")
    # protect katakana long vowel mark from the '-' mapping: only map '－' (U+FF0D), keep 'ー' (U+30FC)
    t = t.translate({ord(c): ord(c) - 0xFEE0 for c in "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"})
    t = t.replace("－", "-").replace("．", ".").replace("～", "~").replace("〜", "~")
    t = t.replace("ｋｍ", "km").replace("ＫＭ", "km")
    t = re.sub(r"[ 　\t]+", "", t)
    t = re.sub(r"(?<=[ァ-ヶー])III(?=改?級)", "Ⅲ", t)
    t = re.sub(r"(?<=[ァ-ヶー])II(?=改?級)", "Ⅱ", t)
    t = re.sub(r"(?<=[ァ-ヶー])IV(?=改?級)", "Ⅳ", t)
    t = re.sub(r"(?<=[ァ-ヶー])I(?=改?級)", "Ⅰ", t)
    t = re.sub(r"(?<=[ァ-ヶー])V(?=改?級)", "Ⅴ", t)
    t = re.sub(r"\n+", "\n", t)
    return t

def body_text(norm: str) -> str:
    """Join lines into one string for sentence splitting (drop lone newlines)."""
    return norm.replace("\n", "")

# ---------------------------------------------------------------- small lexicons
ERA = {"令和": 2018, "平成": 1988, "昭和": 1925}
WEEKDAY = r"(?:（[月火水木金土日]）)?"
TIME_RX = re.compile(r"(午前|午後)?(\d{1,2})時(半|\d{1,2}分)?(頃|過ぎ|前)?")

BEARINGS = {
    "北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5,
    "南": 180, "南南西": 202.5, "南西": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5,
}
BEARING_RX = "(北北東|東北東|東南東|南南東|南南西|西南西|西北西|北北西|北東|南東|南西|北西|北|東|南|西)"
HEADING_DEG = {"北進": 0, "北上": 0, "北航": 0, "北北東進": 22.5, "北東進": 45, "東北東進": 67.5, "東進": 90, "東航": 90,
               "東南東進": 112.5, "南東進": 135, "南南東進": 157.5, "南進": 180, "南下": 180, "南航": 180,
               "南南西進": 202.5, "南西進": 225, "西南西進": 247.5, "西進": 270, "西航": 270, "西北西進": 292.5,
               "北西進": 315, "北北西進": 337.5}
HEADING_RX = re.compile("(" + "|".join(sorted(HEADING_DEG, key=len, reverse=True)) + ")")
DIR8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

WATERS = ["東シナ海", "太平洋", "日本海", "オホーツク海", "南シナ海", "フィリピン海", "黄海", "ベーリング海", "北極海", "台湾海峡"]

SHIP_TYPES = ("空母|ミサイル駆逐艦|駆逐艦|小型フリゲート|フリゲート|ミサイル巡洋艦|原子力ミサイル巡洋艦|巡洋艦|"
              "高速戦闘支援艦|補給艦|給油艦|情報収集艦|測量艦|測定艦|弾道ミサイル原子力潜水艦|巡航ミサイル原子力潜水艦|"
              "原子力潜水艦|潜水艦救難艦|潜水艦|戦車揚陸艦|ドック型揚陸艦|強襲揚陸艦|揚陸艦|ミサイル護衛哨戒艇|護衛哨戒艇|"
              "哨戒艇|ミサイル艇|機雷掃討艇|掃海艇|掃海艦|救難えい船|えい船|設標艦|ミサイル輸送艦|輸送艦|病院船|訓練艦|"
              "護衛艦|コルベット|哨戒艦|戦闘支援艦|多用途支援艦|調査船|海洋観測艦|工作艦|母艦|兵器輸送艦|練習艦|ミサイル観測支援艦|観測支援艦|砕氷船|給兵艦|艦隊航洋曳船|航洋曳船|曳船|水路測量艦|海洋調査船|砕氷艦|ミサイル追跡艦|艦艇")
# class name: katakana (incl. roman numerals / 改 / ・) + 級 + type ; or 艦級不明 / 艦種不明
CLASS_RX = re.compile(
    r"((?:[ァ-ヶー・ⅠⅡⅢⅣⅤⅥ]{2,}|[A-Z][A-Z\-]{2,}|艦級不明|艦種艦級不明|艦種不明|級不明)(?:改)?級?(?:" + SHIP_TYPES + r"))"
    r"(?:「([^」]{1,12})」)?"            # ship name
    r"(?:(\d+)隻)?"                       # count
    r"(?:（艦番号「?(\d{2,4})」?((?:(?:及び|、|，)(?:艦番号)?「?\d{2,4}」?)*)）)?"      # hull no(s) (new style)
    r"(?:(\d+)隻)?"                       # count after hull
)
HULL_CAPTION_RX = re.compile(r"((?:[ァ-ヶー・ⅠⅡⅢⅣⅤⅥ]{2,}|[A-Z][A-Z\-]{2,})(?:改)?級(?:" + SHIP_TYPES + r"))(?:「([^」]{1,12})」)?（(?:艦番号「)?(\d{2,4})」?）")
TOTAL_RX = re.compile(r"(?:計|合計)(\d+)隻")

AIRCRAFT_RX = re.compile(r"((?:[A-Z]{1,3}-\d{1,3}[A-Z]?|[A-Z]{2,4}-?\d{2,4}|無人機|推定無人機)(?:[ぁ-ん一-龥]{0,6}(?:機|爆撃機|戦闘機|輸送機|情報収集機|早期警戒機|哨戒機|偵察機|電子戦機|無人機|ヘリ(?:コプター)?))?)(\d+)?機?")

SAME_RX = re.compile(r"(?:なお|また)、?(?:これら|当該|同)(?:\d+隻)?の?艦艇(?:等)?(?:は|が)、?(.+?)(?:したもの|した艦艇|艦艇)と同一(?:である|のもの)")

JP_UNIT_RX = re.compile(r"(第\d+(?:護衛隊群|護衛隊|航空群|航空隊|掃海隊群|掃海隊|ミサイル艇隊|補給隊|海上補給隊|輸送隊|潜水隊群|潜水隊)|海上自衛隊|航空自衛隊)?(?:所属)?「([^」]+)」（([^）]+)）")

ACTIVITY_RX = [
    ("発着艦", re.compile(r"発着艦")),
    ("陣形運動", re.compile(r"陣形運動")),
    ("遊弋", re.compile(r"遊弋")),
    ("射撃訓練", re.compile(r"射撃(?:訓練)?")),
    ("給油", re.compile(r"(?:洋上)?給油")),
    ("飛行訓練", re.compile(r"飛行訓練")),
    ("中露共同航行", re.compile(r"中露(?:共同)?航行|共同航行|ロシア海軍艦艇と(?:ともに|共に)")),
    ("反転", re.compile(r"反転")),
    ("接続水域", re.compile(r"接続水域")),
    ("領海", re.compile(r"領海")),
    ("ヘリ発着", re.compile(r"ヘリ(?:コプター)?の?発着")),
    ("潜没航行", re.compile(r"潜没")),
    ("曳航", re.compile(r"曳航|えい航")),
]
COUNT_RX = re.compile(r"計約?(\d+)回")

# ---------------------------------------------------------------- helpers
def jdate(era, y, m, d):
    return dt.date(ERA[era] + int(y), int(m), int(d))

DATE_FULL_RX = re.compile(r"(令和|平成|昭和)(\d{1,2}|元)年(\d{1,2})月(\d{1,2})日")
DATE_MD_RX = re.compile(r"(?<![\d年])(\d{1,2})月(\d{1,2})日")
DATE_D_RX = re.compile(r"(?<![\d月])(\d{1,2})日")

class DateCtx:
    def __init__(self, pub_date: dt.date):
        self.pub = pub_date
        self.year = pub_date.year
        self.month = pub_date.month

    def resolve(self, s: str, start=0):
        """Return (date, end_pos) for the first date mention in s[start:], updating context."""
        m = DATE_FULL_RX.search(s, start)
        m2 = DATE_MD_RX.search(s, start)
        m3 = DATE_D_RX.search(s, start)
        cands = [x for x in (m, m2, m3) if x]
        if not cands:
            return None, start
        m = min(cands, key=lambda x: x.start())
        if m.re is DATE_FULL_RX:
            y = 1 if m.group(2) == "元" else int(m.group(2))
            d = jdate(m.group(1), y, m.group(3), m.group(4))
        elif m.re is DATE_MD_RX:
            mo, da = int(m.group(1)), int(m.group(2))
            y = self.pub.year
            # a December event published in January
            if mo > self.pub.month + 1:
                y -= 1
            try:
                d = dt.date(y, mo, da)
            except ValueError:
                return None, m.end()
        else:
            da = int(m.group(1))
            try:
                d = dt.date(self.year, self.month, da)
            except ValueError:
                return None, m.end()
            if d > self.pub + dt.timedelta(days=3):  # "27日" referring to previous month
                mo = self.month - 1 or 12
                y = self.year if self.month > 1 else self.year - 1
                try:
                    d = dt.date(y, mo, da)
                except ValueError:
                    pass
        self.year, self.month = d.year, d.month
        return d, m.end()

def parse_time(s):
    m = TIME_RX.search(s)
    if not m:
        return None
    ap, h, mi, _ = m.groups()
    h = int(h)
    if ap == "午後" and h < 12:
        h += 12
    if ap == "午前" and h == 12:
        h = 0
    minute = 30 if mi == "半" else (int(mi[:-1]) if mi else 0)
    return f"{h:02d}:{minute:02d}"

POS_RX = re.compile(r"([ぁ-んァ-ヶー一-龥]{2,8}?)(?:（[^）]{2,5}）)?の" + BEARING_RX + r"約?(\d+(?:\.\d+)?)km")

def parse_position(s):
    best = None
    for m in POS_RX.finditer(s):
        name, bear, dist = m.group(1), m.group(2), float(m.group(3))
        # strip leading particles/junk like "海上自衛隊は、" captured by greedy kana
        for key in sorted(REF_POINTS, key=len, reverse=True):
            if name.endswith(key):
                name = key
                break
        cand = (name, bear, dist, m.start())
        if name in REF_POINTS:
            best = cand
            break
        if best is None:
            best = cand
    if not best:
        return None
    name, bear, dist, _ = best
    lat = lon = None
    if name in REF_POINTS:
        la, lo = REF_POINTS[name]
        b = math.radians(BEARINGS[bear])
        lat = round(la + dist * math.cos(b) / 111.0, 3)
        lon = round(lo + dist * math.sin(b) / (111.0 * math.cos(math.radians(la))), 3)
    return {"ref_point": name, "bearing_jp": bear, "distance_km": dist, "lat": lat, "lon": lon}

def heading_to_dir8(deg):
    return DIR8[int(((deg + 22.5) % 360) // 45)]

def parse_strait(s, heading_deg):
    for st in STRAITS:
        if st["_rx"].search(s):
            out = {"strait": st["strait_key"], "strait_dir": None, "waters_from": None, "waters_to": None}
            if heading_deg is not None:
                out["strait_dir"] = heading_to_dir8(heading_deg)
                diff = math.cos(math.radians(heading_deg - st["b_bearing"]))
                if diff > 0.05:
                    out["waters_from"], out["waters_to"] = st["side_a"], st["side_b"]
                elif diff < -0.05:
                    out["waters_from"], out["waters_to"] = st["side_b"], st["side_a"]
            # explicit waters in sentence override
            for w in WATERS:
                if re.search(w + r"(?:へ|に)向けて|" + w + r"へ(?:進出|航行)", s):
                    out["waters_to"] = w
                if re.search(w + r"から", s):
                    out["waters_from"] = w
            return out
    return None

WATERS_RX = re.compile("(" + "|".join(WATERS) + r")(?:上)?から(?:、)?(" + "|".join(WATERS) + r")(?:へ|に)向けて")
WATERS_TO_RX = re.compile("(" + "|".join(WATERS) + r")(?:へ|に)向けて")
WATERS_FROM_RX = re.compile("(" + "|".join(WATERS) + r")(?:上)?から")

def parse_waters(s):
    m = WATERS_RX.search(s)
    if m:
        return m.group(1), m.group(2)
    f = WATERS_FROM_RX.search(s); t = WATERS_TO_RX.search(s)
    return (f.group(1) if f else None), (t.group(1) if t else None)

# ref_point -> candidate strait when a passage is described without naming the strait
NEAR_STRAIT = {
    "宮古島": "miyako", "久米島": "miyako", "沖縄本島": "miyako", "多良間島": "miyako_ishigaki",
    "上対馬": "tsushima", "下対馬": "tsushima", "対馬": "tsushima", "壱岐": "tsushima", "沖ノ島": "tsushima", "見島": "tsushima", "福江島": "tsushima", "五島列島": "tsushima", "男女群島": "tsushima",
    "大隅半島": "osumi", "佐多岬": "osumi", "種子島": "osumi", "屋久島": "osumi", "都井岬": "osumi", "口永良部島": "osumi",
    "竜飛崎": "tsugaru", "尻屋崎": "tsugaru", "大間崎": "tsugaru", "恵山岬": "tsugaru", "白神岬": "tsugaru", "津軽半島": "tsugaru", "下北半島": "tsugaru", "松前": "tsugaru", "奥尻島": "tsugaru",
    "宗谷岬": "soya", "礼文島": "soya", "利尻島": "soya", "稚内": "soya", "知床岬": "soya",
    "与那国島": "yonaguni_taiwan", "仲ノ神島": "yonaguni_taiwan", "波照間島": "yonaguni_iriomote", "西表島": "yonaguni_iriomote", "石垣島": "ishigaki_iriomote",
    "奄美大島": "amami_yokoate", "横当島": "amami_yokoate", "喜界島": "kikai", "徳之島": "amami_tokunoshima", "沖永良部島": "amami_tokunoshima", "与論島": "amami_tokunoshima",
    "宝島": "tokara", "小宝島": "tokara", "悪石島": "tokara", "諏訪之瀬島": "tokara", "中之島": "tokara", "口之島": "tokara",
    "御蔵島": "izu_mikura_hachijo", "八丈島": "izu_mikura_hachijo", "三宅島": "izu_mikura_hachijo", "青ヶ島": "izu_mikura_hachijo",
    "納沙布岬": "nemuro", "国後島": "etorofu", "択捉島": "etorofu",
}
STRAIT_BY_KEY = {s["strait_key"]: s for s in STRAITS}

def infer_strait(ref_point, distance_km, heading_deg, wf, wt):
    """Passage described as 'X の NE 110km を 東シナ海から太平洋に向けて 南東進' -> strait key."""
    if not ref_point or ref_point not in NEAR_STRAIT:
        return None
    if distance_km is not None and distance_km > 250:
        return None
    key = NEAR_STRAIT[ref_point]
    st = STRAIT_BY_KEY[key]
    if wf and wt and {wf, wt} <= {st["side_a"], st["side_b"]}:
        return key
    if heading_deg is not None and wf is None and wt is None:
        # heading must cross the strait axis
        if abs(math.cos(math.radians(heading_deg - st["b_bearing"]))) > 0.7 and (distance_km or 0) <= 150:
            return key
    return None

def class_key(class_jp: str) -> str:
    k = class_jp
    k = re.sub("(" + SHIP_TYPES + ")$", "", k)
    k = k.replace("級", "").replace("改", "kai")
    k = k.replace("Ⅰ", "1").replace("Ⅱ", "2").replace("Ⅲ", "3").replace("Ⅳ", "4").replace("Ⅴ", "5")
    return k

def parse_units(s, nation_default):
    units = []
    for m in CLASS_RX.finditer(s):
        cj, name, c1, hull, more, c2 = m.groups()
        if cj in ("艦艇",):
            continue
        cnt = int(c1 or c2 or 1)
        hulls = [hull] + re.findall(r"\d{2,4}", more or "") if hull else []
        nation = nation_default
        pre = s[max(0, m.start() - 12):m.start()]
        if "ロシア" in pre:
            nation = "RU"
        elif "中国" in pre:
            nation = "CN"
        kind = "sub" if "潜水艦" in cj and "救難" not in cj else "ship"
        if len(hulls) > 1:
            for h in hulls:
                units.append({"nation": nation, "kind": kind, "class_jp": cj, "class_key": class_key(cj), "hull_no": h, "name": name, "count": 1})
        else:
            units.append({"nation": nation, "kind": kind, "class_jp": cj, "class_key": class_key(cj),
                          "hull_no": hull, "name": name, "count": cnt})
    return units

def parse_aircraft(s, nation_default):
    units = []
    for m in AIRCRAFT_RX.finditer(s):
        t, cnt = m.group(1), m.group(2)
        kind = "uav" if "無人" in t or re.match(r"(BZK|TB|WZ|GJ|WL|CH)-?\d", t) else "air"
        units.append({"nation": nation_default, "kind": kind, "class_jp": t, "class_key": re.sub(r"[^A-Z0-9]", "", t.split("機")[0]).lower(),
                      "hull_no": None, "name": None, "count": int(cnt) if cnt else 1})
    return units

def parse_jp_units(s):
    out = []
    if "自衛隊" not in s:
        return out
    for m in JP_UNIT_RX.finditer(s):
        parent, name, base = m.groups()
        kind = "air" if re.match(r"[A-Z]{1,2}-\d", name) else "ship"
        out.append({"unit_kind": kind, "unit_name": name, "parent_unit": parent, "base": base})
    return out

# ---------------------------------------------------------------- main
def split_sentences(body):
    return [x for x in re.split(r"(?<=。)", body) if x.strip()]

def parse_release(release_id, pub_date, text, domain_hint=None, nation_hint=None):
    norm = normalize(text)
    body = body_text(norm)
    pub = pub_date if isinstance(pub_date, dt.date) else dt.date.fromisoformat(pub_date)

    nation = nation_hint or ("CNRU" if ("中国" in body[:400] and "ロシア" in body[:400]) else "CN" if "中国" in body[:400] else "RU" if "ロシア" in body[:400] else "OTHER")
    domain = domain_hint or ("correction" if "一部訂正" in body[:200] else "air" if re.search(r"軍機|航空機|飛行について", body[:200]) else "ship")

    # cut off trailing caption/map text for sentence parsing, but keep captions for hull numbers
    cut = len(body)
    for marker in ("統合幕僚監部報道発表資料", "行動概要", "http://www.mod", "https://www.mod"):
        i = body.find(marker, 60)
        if i > 0:
            cut = min(cut, i)
    main = body[:cut]
    tail = body[cut:]
    # drop the header (date / 統合幕僚監部 / （お知らせ） / title) – body starts after the last "について" in the first ~150 chars
    hdr = [m.end() for m in re.finditer(r"について", main[:200])]
    if hdr:
        main = main[hdr[-1]:]

    corrects = None
    if domain == "correction":
        m = re.search(r"(\d{1,2})月(\d{1,2})日" + WEEKDAY + r"に公表した", body)
        if m:
            corrects = f"{pub.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    dctx = DateCtx(pub)
    sightings, jp_units = [], []
    mo = re.search(r"期日(?:令和|平成)?(?:\d{1,2}|元)?年?(\d{1,2})月(\d{1,2})日", body)
    if domain == "air" and mo and "国籍等" in body:
        d1, _ = dctx.resolve(body, mo.start())
        seg = body[body.find("国籍等"):]
        seg = seg[:seg.find("行動概要")] if "行動概要" in seg else seg[:120]
        nat = "RU" if "ロシア" in seg else "CN" if "中国" in seg else default_nation if False else ("RU" if nation == "RU" else "CN")
        units = []
        for m in re.finditer(r"([A-Z]{1,3}-\d{1,3}[A-Z]?)(?:型)?(?:（([^）]+)）)?(?:\s*)(\d+)?機?", seg):
            units.append({"nation": nat, "kind": "air", "class_jp": m.group(1) + ("（" + m.group(2) + "）" if m.group(2) else ""), "class_key": m.group(1).replace("-", "").lower(), "hull_no": None, "name": None, "count": int(m.group(3)) if m.group(3) else 1})
        resp = re.search(r"自衛隊の対応(.{0,40}?)。", body)
        acts = []
        if resp:
            if "緊急発進" in resp.group(1): acts.append("スクランブル")
        area = re.search(r"(日本海|太平洋|東シナ海|オホーツク海|沖縄方面|南西諸島)", body[:120])
        sightings.append({"seq": 1, "date_inherited": False, "observed_date": d1.isoformat() if d1 else None, "observed_date_end": None, "observed_time": None,
                          "ref_point": None, "bearing_jp": None, "distance_km": None, "lat": None, "lon": None, "heading_jp": None, "heading_deg": None,
                          "strait": None, "strait_dir": None, "strait_inferred": 0, "waters_from": None, "waters_to": area.group(1) if area else None,
                          "activity": "/".join(acts) or None, "activity_count": None, "same_as_text": None, "unit_count": sum(u["count"] for u in units) or None,
                          "sentence": body[:200], "units": units, "units_inherited": False})
        return {"release": {"release_id": release_id, "pub_date": pub.isoformat(), "nation": nation, "domain": domain, "has_track_map": 1, "corrects": None, "format": "air_old"},
                "sightings": sightings, "jp_units": [], "same_as_text": None}
    current_units = []
    same_as_text = None
    seq = 0
    default_nation = "RU" if nation == "RU" else "CN"

    for sent in split_sentences(main):
        if "警戒監視" in sent or "情報収集を行" in sent or "所要の" in sent:
            jp_units.extend(parse_jp_units(sent))
            continue
        m_same = SAME_RX.search(sent)
        if m_same:
            same_as_text = m_same.group(1)
            # attach to last sighting; also keep as its own record for later resolution
            if sightings:
                sightings[-1]["same_as_text"] = same_as_text
            continue
        if not re.search(r"確認|航行|出域|入域|遊弋|通過|進出|反転", sent):
            continue
        for ju in parse_jp_units(sent):
            if ju not in jp_units:
                jp_units.append(ju)
        if re.search(r"^(※|（訂正)", sent):
            continue
        # units
        if domain == "air":
            units = parse_aircraft(sent, default_nation)
        else:
            units = parse_units(sent, default_nation)
        if units:
            current_units = units
            inherit = False
        else:
            units = [dict(u) for u in current_units]
            inherit = True
        # per-sentence partial subset (e.g. "ジャンカイⅡ級フリゲート（艦番号「546」）は…反転")
        # dates
        d1, pos = dctx.resolve(sent)
        d2 = None
        if d1 and re.search(r"(?:から|~)", sent[pos:pos + 12]):
            d2, _ = dctx.resolve(sent, pos)
        hm = HEADING_RX.search(sent)
        heading_jp = hm.group(1) if hm else ("遊弋" if "遊弋" in sent else None)
        heading_deg = HEADING_DEG.get(heading_jp) if heading_jp else None
        # prefer the heading attached to the strait phrase ("…の間の海域を南東進")
        st_info = None
        for st in STRAITS:
            sm = st["_rx"].search(sent)
            if sm:
                after = sent[sm.end():sm.end() + 20]
                hm2 = HEADING_RX.search(after)
                if hm2:
                    heading_jp = hm2.group(1); heading_deg = HEADING_DEG[heading_jp]
                st_info = parse_strait(sent, heading_deg)
                # if several dates precede the strait phrase, take the one nearest before it
                pre = sent[:sm.start()]
                dm = list(re.finditer(r"(?:(?:令和|平成)(?:\d{1,2}|元)年)?(?:\d{1,2}月)?\d{1,2}日" + WEEKDAY + r"(?:から(?:\d{1,2}月)?\d{1,2}日" + WEEKDAY + r")?(?:にかけて|に|、)?", pre))
                if len(dm) > 1:
                    last = dm[-1]
                    d1x, posx = dctx.resolve(pre, last.start())
                    if d1x:
                        d1 = d1x; d2 = None
                        if re.search(r"から", pre[posx:posx + 6]):
                            d2, _ = dctx.resolve(pre, posx)
                break
        posinfo = parse_position(sent) or {}
        wf, wt = parse_waters(sent)
        strait_inferred = 0
        if st_info is None:
            key = infer_strait(posinfo.get("ref_point"), posinfo.get("distance_km"), heading_deg, wf, wt)
            if key:
                st = STRAIT_BY_KEY[key]
                st_info = {"strait": key, "strait_dir": heading_to_dir8(heading_deg) if heading_deg is not None else None, "waters_from": wf, "waters_to": wt}
                if not (wf and wt) and heading_deg is not None:
                    diff = math.cos(math.radians(heading_deg - st["b_bearing"]))
                    st_info["waters_from"], st_info["waters_to"] = (st["side_a"], st["side_b"]) if diff > 0 else (st["side_b"], st["side_a"])
                strait_inferred = 1
        if st_info is not None:
            if wf: st_info["waters_from"] = wf
            if wt: st_info["waters_to"] = wt
        acts = [n for n, rx in ACTIVITY_RX if rx.search(sent)]
        cm = COUNT_RX.search(sent)
        tm = TOTAL_RX.search(sent)
        seq += 1
        date_inherited = False
        if not d1 and sightings and sightings[-1]["observed_date"]:
            d1 = dt.date.fromisoformat(sightings[-1]["observed_date_end"] or sightings[-1]["observed_date"])
            date_inherited = True
        sightings.append({
            "seq": seq,
            "date_inherited": date_inherited,
            "observed_date": d1.isoformat() if d1 else None,
            "observed_date_end": d2.isoformat() if d2 else None,
            "observed_time": parse_time(sent),
            **{k: posinfo.get(k) for k in ("ref_point", "bearing_jp", "distance_km", "lat", "lon")},
            "heading_jp": heading_jp, "heading_deg": heading_deg,
            "strait": st_info["strait"] if st_info else None,
            "strait_dir": st_info["strait_dir"] if st_info else None,
            "strait_inferred": strait_inferred,
            "waters_from": st_info["waters_from"] if st_info else wf,
            "waters_to": st_info["waters_to"] if st_info else wt,
            "activity": "/".join(acts) or None,
            "activity_count": int(cm.group(1)) if cm else None,
            "same_as_text": None,
            "unit_count": int(tm.group(1)) if tm else sum(u["count"] for u in units) or None,
            "sentence": sent,
            "units": units,
            "units_inherited": inherit,
        })

    # table-style releases ("以下のとおり"): no units in sentences -> take them from captions/table
    if sightings and not any(sg["units"] for sg in sightings):
        caps = []
        for c, n_, h in HULL_CAPTION_RX.findall(body):
            if (c, h) not in {(x["class_jp"], x["hull_no"]) for x in caps}:
                caps.append({"nation": default_nation, "kind": "sub" if "潜水艦" in c and "救難" not in c else "ship", "class_jp": c, "class_key": class_key(c), "hull_no": h, "name": n_ or None, "count": 1})
        if caps:
            for sg in sightings:
                sg["units"] = [dict(x) for x in caps]; sg["units_inherited"] = True
    # hull numbers from photo captions (older releases) -> fill missing hull_no by class, in order
    cap_hulls, _seen = [], set()
    for c, n, h in HULL_CAPTION_RX.findall(body):
        if (c, h) not in _seen:
            _seen.add((c, h)); cap_hulls.append((c, n, h))
    if cap_hulls:
        for sg in sightings:
            expanded = []
            for u in sg["units"]:
                if u["hull_no"] or u["kind"] == "air":
                    expanded.append(u); continue
                matches = [h for c, n, h in cap_hulls if c == u["class_jp"]]
                if matches and len(matches) >= u["count"]:
                    for h in matches[:u["count"]]:
                        nu = dict(u); nu["hull_no"] = h; nu["count"] = 1
                        names = [n for c, n, h2 in cap_hulls if h2 == h and n]
                        if names and not nu["name"]:
                            nu["name"] = names[0]
                        expanded.append(nu)
                else:
                    expanded.append(u)
            sg["units"] = expanded

    return {
        "release": {"release_id": release_id, "pub_date": pub.isoformat(), "nation": nation, "domain": domain,
                    "has_track_map": 1 if "行動概要" in body else 0, "corrects": corrects},
        "sightings": sightings,
        "jp_units": jp_units,
        "same_as_text": same_as_text,
    }

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        rid = os.path.splitext(os.path.basename(p))[0]
        pd = dt.date(int(rid[1:5]), int(rid[5:7]), int(rid[7:9]))
        res = parse_release(rid, pd, open(p, encoding="utf-8").read())
        print(json.dumps(res, ensure_ascii=False, indent=1))
