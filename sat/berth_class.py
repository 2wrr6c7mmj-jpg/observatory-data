"""The class of every berth line (naval-1): naval, civil or unknown, with the evidence per line.

User's instruction (2026-11-03): the harbour collector also picks up civil ships; take only what is clearly a
warship, from the naval piers and quays. Each berth line of a site (sat/berths.py: the pier and quay lines of
sat/masks/<site>-quay-v<n>.json, the hand-drawn lines among them, and the coastline sections of sat/berths.json)
gets a class in sat/berth_class/<site>.json. The headline counts use the naval lines only (sat/run.py, from
berths.NAVAL_SINCE).

The automatic class of a line is built from public sources only, and every entry keeps its evidence:

* OpenStreetMap (sat/berth_osm.py, the probe file <site>-osm-areas.json): a line touching (a vertex inside, or
  within ``TOUCH_M``) an area of landuse=military or military=naval_base/base/barracks/... is naval; touching
  only civil port areas (landuse=port, industrial=port/fishing, harbour=*, ferry terminals, marinas) it is civil;
  touching both, the side holding the line's middle point decides (neither: unknown);
* shipyards (industrial=shipyard, waterway=dock/boatyard): naval at a shipyard site (kind yard, e.g. Mawei: ships
  being built or refitted are never dropped as civil) or when the yard is a naval yard (a military area or a Wen
  point of a military yard within ``WEN_M``), else unknown;
* Joseph Wen's PLA map (docs/data/ref/wen-pla-bases.geojson, points only): a navy point within ``WEN_M`` of the line
  in a naval sector of sites.json supports naval when OSM is silent (confidence "推定");
* anything else is unknown (never counted in the headline: "clearly a warship" only).

Entries with ``"fixed": true`` (a decision by hand: imagery, a published source) are kept as they are when the file
is rebuilt; their ``basis`` says why.

    python -m sat.berth_class build --osm DIR [SITE ...]     # writes sat/berth_class/<site>.json
    python -m sat.berth_class geojson [--out docs/data/sat/berth_class.geojson]

Data (c) OpenStreetMap contributors, ODbL 1.0. Wen's map: 出典は docs/data/ref/wen-pla-bases.meta.json の credit.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import berths, quay_lines

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASS_DIR = berths.CLASS_DIR
WEN_FILE = ROOT / "docs" / "data" / "ref" / "wen-pla-bases.geojson"
WEN_META = ROOT / "docs" / "data" / "ref" / "wen-pla-bases.meta.json"
GEOJSON_OUT = ROOT / "docs" / "data" / "sat" / "berth_class.geojson"
TOUCH_M = 40.0
WEN_M = 600.0
STEP_M = 20.0
MIL_VALUES = {"naval_base", "base", "barracks", "office", "training_area", "airfield", "launchpad", "obstacle_course"}
NAVY_LAYER = "解放軍海軍、海軍陸戰隊基地及設施"
INDUSTRY_LAYER = "中國軍工及航天產業"


def _mxy(lat0: float) -> tuple[float, float]:
    return 111320.0 * math.cos(math.radians(lat0)), 111132.0


def _in_ring(x: float, y: float, ring) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = 0.0 if dd == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def densify(coords, mx: float, my: float, step: float = STEP_M) -> list[tuple[float, float]]:
    out = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        n = max(1, int(math.ceil(math.hypot((x2 - x1) * mx, (y2 - y1) * my) / step)))
        out += [(x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n) for i in range(n)]
    out.append(tuple(coords[-1]))
    return out


class Area:
    """An OSM area (closed rings) or a point, with what it says."""

    def __init__(self, f: dict, mx: float, my: float):
        self.osm, self.tags = f["osm"], f.get("tags") or {}
        self.rings = [[(x * mx, y * my) for x, y in r] for r in f["rings"]]
        self.closed = [r for r in self.rings if len(r) >= 4 and r[0] == r[-1]]
        self.kind = area_kind(self.tags)

    def label(self) -> str:
        t = self.tags
        keys = [f"{k}={t[k]}" for k in ("landuse", "military", "industrial", "harbour", "amenity", "leisure", "waterway",
                                        "seamark:type", "man_made") if t.get(k)]
        name = t.get("name") or t.get("name:en") or ""
        return f"OSM {self.osm}{('「' + name + '」') if name else ''}（{'・'.join(keys)}）"

    def touch(self, pts) -> tuple[bool, float]:
        """(a point of the line inside, the least distance from the line's points to the outline) in metres."""
        inside = any(_in_ring(x, y, r) for r in self.closed for x, y in pts)
        d = min((_seg_dist(x, y, *a, *b) for r in self.rings for a, b in zip(r, r[1:]) for x, y in pts), default=math.inf)
        if len(self.rings) == 1 and len(self.rings[0]) == 1:   # a node
            d = min(math.hypot(x - self.rings[0][0][0], y - self.rings[0][0][1]) for x, y in pts)
        return inside, d


def area_kind(t: dict) -> str | None:
    """military, yard, civil or None (not evidence for a class)."""
    if t.get("landuse") == "military" or t.get("military") in MIL_VALUES:
        return "military"
    if t.get("industrial") == "shipyard" or t.get("waterway") in ("dock", "boatyard") or \
            (t.get("landuse") == "industrial" and any(k in (t.get("name") or "") for k in ("船厂", "船廠", "造船", "Shipyard", "shipyard"))):
        return "yard"
    if t.get("landuse") in ("port", "harbour") or t.get("industrial") in ("port", "fishing") or t.get("harbour") \
            or t.get("seamark:type") in ("harbour", "small_craft_facility") or t.get("amenity") == "ferry_terminal" \
            or t.get("leisure") == "marina" or (t.get("landuse") == "industrial" and t.get("name")):
        return "civil"
    return None


def load_osm(osm_dir: Path | None, site_id: str) -> dict | None:
    if osm_dir is None:
        return None
    p = osm_dir / f"{site_id}-osm-areas.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wen_points() -> list[dict]:
    if not WEN_FILE.exists():
        return []
    return [f for f in json.loads(WEN_FILE.read_text(encoding="utf-8"))["features"]
            if f["properties"]["layer"] in (NAVY_LAYER, INDUSTRY_LAYER)]


def berth_items(site: dict, lines: dict) -> list[dict]:
    """The berth lines of a site: {"way", "kind", "coords", "part"} (a coastline section has its part)."""
    bt = berths.Berths(site, lines, classes=None)
    out = []
    for w in lines.get("ways", []):
        wid = str(w.get("id"))
        if w["kind"] in berths.BERTH_KINDS and wid not in bt.exclude:
            out.append({"way": w["id"], "kind": w["kind"], "coords": w["coords"], "part": None})
        elif w["kind"] == "coastline":
            for e in (berths._CONF.get("sites") or {}).get(site["id"], {}).get("coastline") or []:
                if str(e["way"]) != wid:
                    continue
                part = {"sector": e["sector"]} if e.get("sector") else ({"bbox": e["bbox"]} if e.get("bbox") else None)
                area = None
                if part and part.get("sector"):
                    area = next((s for s in site.get("sectors") or [] if s["id"] == part["sector"]), None)
                elif part:
                    area = part
                coords = [c for c in w["coords"] if area is None or berths._in_area(area, c[0], c[1])]
                if len(coords) >= 2:
                    out.append({"way": w["id"], "kind": "coastline", "coords": coords, "part": part})
    return out


def sector_role(site: dict, lon: float, lat: float) -> tuple[str | None, str | None]:
    for s in site.get("sectors") or []:
        if berths._in_area(s, lon, lat):
            return s["id"], s.get("role") or ("naval" if s["id"] == "A" else None)
    rest = site.get("sector_rest") or {}
    return rest.get("id"), rest.get("role") or "other"


def classify_line(site: dict, item: dict, areas: list[Area], wen: list[dict], mx: float, my: float) -> dict:
    pts = [(x * mx, y * my) for x, y in densify(item["coords"], mx, my)]
    mid = item["coords"][len(item["coords"]) // 2]
    mpt = [(mid[0] * mx, mid[1] * my)]
    hits = {"military": [], "yard": [], "civil": []}
    mid_in = {"military": False, "yard": False, "civil": False}
    for a in areas:
        if a.kind is None:
            continue
        inside, d = a.touch(pts)
        if inside or d <= TOUCH_M:
            hits[a.kind].append((a, inside, d))
            if a.closed and any(_in_ring(*mpt[0], r) for r in a.closed):
                mid_in[a.kind] = True
    near_wen = []
    for f in wen:
        x, y = f["geometry"]["coordinates"]
        d = min(math.hypot(x * mx - px, y * my - py) for px, py in pts)
        if d <= WEN_M:
            near_wen.append((f, d))
    sec, role = sector_role(site, mid[0], mid[1])
    basis = []
    for k in ("military", "yard", "civil"):
        for a, inside, d in sorted(hits[k], key=lambda h: h[2])[:3]:
            basis.append(f"{a.label()}に{'含まれる' if inside else f'接する（{d:.0f} m）'}")
    for f, d in sorted(near_wen, key=lambda x: x[1])[:2]:
        basis.append(f"溫約瑟の地図 {f['id']}「{f['properties']['name']}」まで {d:.0f} m")
    basis.append(f"区画 {sec}（role {role}）")
    navy_wen = [f for f, _ in near_wen if f["properties"]["layer"] == NAVY_LAYER]
    yard_wen = [f for f, _ in near_wen if f["properties"]["layer"] == INDUSTRY_LAYER and
                any(k in f["properties"]["name"] + f["properties"]["description"] for k in ("解放軍", "軍用", "潛艇", "軍艦", "護衛艦"))]
    mil, yard, civ = bool(hits["military"]), bool(hits["yard"]), bool(hits["civil"])
    if mil and not civ:
        cls, conf, why = "naval", "確認寄り", "OSM の軍の区域に接し、民間の港の区域に接しない"
    elif mil and civ:
        if mid_in["military"] and not mid_in["civil"]:
            cls, conf, why = "naval", "推定", "OSM の軍と民間の区域の両方に接する。線の中点は軍の区域の内側"
        elif mid_in["civil"] and not mid_in["military"]:
            cls, conf, why = "civil", "推定", "OSM の軍と民間の区域の両方に接する。線の中点は民間の区域の内側"
        else:
            cls, conf, why = "unknown", "—", "OSM の軍と民間の区域の両方に接し、どちらとも言えない"
    elif yard and (site.get("kind") == "yard" or "造船" in (site.get("name") or "") or yard_wen):
        cls, conf, why = "naval", "推定", "造船所の岸壁（建造・整備中の艦を落とさないため海軍に数える。民間船の建造・修理も入りうる）"
    elif yard:
        cls, conf, why = "unknown", "—", "造船所の岸壁だが、軍の造船所と言える根拠が無い"
    elif civ:
        cls, conf, why = "civil", "推定", "OSM の民間の港の区域に接し、軍の区域に接しない"
    elif navy_wen and role == "naval":
        cls, conf, why = "naval", "推定", "OSM に区域は無いが、海軍の区画の中で溫約瑟の地図の海軍の地点に近い"
    else:
        cls, conf, why = "unknown", "—", "軍・民間のどちらの根拠も無い"
    e = {"way": item["way"], "kind": item["kind"], "class": cls, "conf": conf, "why": why, "basis": basis}
    if item["part"]:
        e["part"] = item["part"]
    return e


def build(site: dict, osm_dir: Path | None, out_dir: Path = CLASS_DIR) -> dict:
    lv = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
    lines = quay_lines.load(site["id"], lv) or {"ways": []}
    items = berth_items(site, lines)
    pts = [c for it in items for c in it["coords"]] or [[0, (site["bbox"][1] + site["bbox"][3]) / 2]]
    mx, my = _mxy(sum(c[1] for c in pts) / len(pts))
    osm = load_osm(osm_dir, site["id"])
    areas = [Area(f, mx, my) for f in (osm or {}).get("features", [])]
    wen = wen_points()
    p = out_dir / f"{site['id']}.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    fixed = {(str(e["way"]), json.dumps(e.get("part"), sort_keys=True)): e for e in old.get("lines", []) if e.get("fixed")}
    out_lines = []
    for it in items:
        k = (str(it["way"]), json.dumps(it["part"], sort_keys=True))
        out_lines.append(fixed.pop(k) if k in fixed else classify_line(site, it, areas, wen, mx, my))
    out_lines += list(fixed.values())   # a hand-fixed entry of a line no longer drawn stays (reported by the tests)
    doc = {"site": site["id"], "name": site.get("name"), "version": berths.NAVAL_VERSION,
           "lines_version": f"{site['id']}-quay-v{lv}", "default": "unknown",
           "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "osm_base": (osm or {}).get("osm_base"), "osm_bbox": (osm or {}).get("bbox"),
           "confidence": old.get("confidence"), "note": old.get("note"),
           "sources": ["OpenStreetMap（© OpenStreetMap contributors, ODbL 1.0）、sat/berth_osm.py の probe",
                       wen_credit()],
           "rule": "sat/berth_class.py の docstring（軍の区域に接する→海軍、民間の港の区域だけ→民間、造船所は造船所の地点と軍の造船所だけ海軍、ほかは不明）。fixed＝手で決めた区分",
           "lines": out_lines}
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


def wen_credit() -> str:
    if WEN_META.exists():
        return json.loads(WEN_META.read_text(encoding="utf-8"))["attribution"]["credit"]
    return "溫約瑟（Joseph Wen）「中國人民解放軍基地及設施 (持續更新)」"


def geojson(sites: list[dict]) -> dict:
    """Every classed berth line as a LineString (the part of a coastline section only), for the eye check on a map."""
    feats = []
    counts = {}
    for site in sites:
        c = berths.load_classes(site["id"])
        if c is None:
            continue
        lv = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
        lines = quay_lines.load(site["id"], lv) or {"ways": []}
        geo = {}
        for it in berth_items(site, lines):
            geo[(str(it["way"]), json.dumps(it["part"], sort_keys=True))] = it["coords"]
        n = {k: 0 for k in berths.CLASSES}
        for e in c["lines"]:
            coords = geo.get((str(e["way"]), json.dumps(e.get("part"), sort_keys=True)))
            if not coords:
                continue
            n[e["class"]] += 1
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                          "properties": {"site_id": site["id"], "site_name": site.get("name"), "way": e["way"],
                                         "osm": f"https://www.openstreetmap.org/way/{e['way']}" if int(e["way"]) > 0 else None,
                                         "kind": e["kind"], "class": e["class"], "conf": e.get("conf"), "why": e.get("why"),
                                         "basis": e.get("basis"), "fixed": bool(e.get("fixed")), "part": e.get("part"),
                                         "version": berths.NAVAL_VERSION}})
        counts[site["id"]] = n
    return {"type": "FeatureCollection", "version": berths.NAVAL_VERSION, "since_utc": berths.NAVAL_SINCE,
            "note": ("岸壁・桟橋の線の区分（naval＝海軍、civil＝民間、unknown＝不明）。見出しの数（港内・岸壁帯の候補、岸壁の占有、est-1・est-2、赤枠）"
                     f"は {berths.NAVAL_SINCE} 以降の撮影で naval の線に沿うものだけ。区分の設定は sat/berth_class/<地点>.json、根拠は basis"),
            "credit": ["Data © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)", wen_credit()],
            "counts": counts, "features": feats}


def main(argv=None) -> int:
    from .run import load_sites
    ap = argparse.ArgumentParser(description="naval-1: the class of every berth line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--osm", type=Path, help="folder of <site>-osm-areas.json (sat/berth_osm.py)")
    b.add_argument("sites", nargs="*")
    g = sub.add_parser("geojson")
    g.add_argument("--out", type=Path, default=GEOJSON_OUT)
    args = ap.parse_args(argv)
    sites = [s for s in load_sites() if s.get("enabled")]
    if args.cmd == "build":
        for s in sites:
            if args.sites and s["id"] not in args.sites:
                continue
            d = build(s, args.osm)
            n = {k: sum(e["class"] == k for e in d["lines"]) for k in berths.CLASSES}
            print(s["id"], n)
    else:
        fc = geojson(sites)
        args.out.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(args.out, len(fc["features"]), fc["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
