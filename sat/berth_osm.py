"""OSM areas that tell a naval berth from a civil one (naval-1, probe only: the work shell cannot reach Overpass).

For each site of sat/sites.json (its bbox plus ``--margin`` degrees, or a wider box given with ``--wide SITE=w,s,e,n``):
military areas (landuse=military, military=*), ports and shipyards (landuse=port, industrial=port/shipyard,
harbour=*, seamark harbours, ferry terminals, marinas, fishing harbours) with their outlines, and the named
piers, quays and breakwaters (all piers and quays, named or not, in a --wide box: berths outside the site). Written to ``<out>/<site>-osm-areas.json``; sat/berth_class.py builds the class of
each berth line from them (sat/berth_class.json keeps the result and the evidence per line).

    python -m sat.berth_osm --out DIR [--margin 0.01] [--wide-margin 0.08] [--wide XSN=121.55,29.42,121.95,29.65] [SITE ...]

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .osm_mask import OVERPASS_URL, USER_AGENT

KEEP_TAGS = ("name", "name:en", "name:zh", "landuse", "military", "industrial", "harbour", "man_made", "amenity",
             "leisure", "seamark:type", "seamark:harbour:category", "operator", "access", "waterway", "building")


def query(b, all_piers: bool = False) -> str:
    w, s, e, n = b
    bb = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:150];('
            f'nwr["landuse"="military"]{bb};'
            f'nwr["military"]{bb};'
            f'nwr["landuse"~"^(port|harbour)$"]{bb};'
            f'nwr["industrial"~"^(port|shipyard|fishing)$"]{bb};'
            f'nwr["landuse"="industrial"]["name"~"船|港|码头|碼頭|Shipyard|shipyard|Port|port|渔|漁"]{bb};'
            f'nwr["harbour"]{bb};'
            f'nwr["seamark:type"~"^(harbour|small_craft_facility)$"]{bb};'
            f'nwr["amenity"="ferry_terminal"]{bb};'
            f'nwr["leisure"="marina"]{bb};'
            f'way["man_made"~"^(pier|quay|breakwater|dock)$"]["name"]{bb};'
            f'way["waterway"~"^(dock|boatyard)$"]{bb};'
            + (f'way["man_made"~"^(pier|quay)$"]{bb};' if all_piers else '')
            + ');out tags geom;')


def fetch(b, url: str = OVERPASS_URL, all_piers: bool = False) -> dict:
    data = urllib.parse.urlencode({"data": query(b, all_piers)}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=200) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(45)
    return {}


def rings(el: dict) -> list[list[list[float]]]:
    if el.get("type") == "way" and el.get("geometry"):
        return [[[round(p["lon"], 6), round(p["lat"], 6)] for p in el["geometry"]]]
    if el.get("type") == "relation":
        return [[[round(p["lon"], 6), round(p["lat"], 6)] for p in m["geometry"]] for m in el.get("members") or []
                if m.get("geometry") and m.get("role") in ("outer", "")]
    if el.get("type") == "node":
        return [[[round(el["lon"], 6), round(el["lat"], 6)]]]
    return []


def simplify(osm: dict) -> list[dict]:
    out = []
    for el in osm.get("elements", []):
        rs = [r for r in rings(el) if r]
        if not rs:
            continue
        tags = el.get("tags") or {}
        out.append({"osm": f"{el['type']}/{el['id']}", "tags": {k: v for k, v in tags.items() if k in KEEP_TAGS},
                    "rings": rs})
    out.sort(key=lambda f: f["osm"])
    return out


def main(argv=None) -> int:
    from .run import load_sites
    ap = argparse.ArgumentParser(description="OSM military / port / shipyard areas around the sites (naval-1)")
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--wide", action="append", default=[], help="SITE=w,s,e,n: a wider box for that site")
    ap.add_argument("--wide-margin", type=float, default=None,
                    help="every site: a wide box of bbox + this many degrees with all piers and quays (naval-1 survey)")
    ap.add_argument("--sleep", type=float, default=40.0)
    args = ap.parse_args(argv)
    wide = {k: [float(x) for x in v.split(",")] for k, v in (a.split("=", 1) for a in args.wide)}
    sites = [s for s in load_sites() if s.get("enabled") and (not args.sites or s["id"] in args.sites)]
    args.out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(sites):
        if i:
            time.sleep(args.sleep)
        m = args.wide_margin if args.wide_margin is not None and s["id"] not in wide else args.margin
        b = wide.get(s["id"]) or [round(s["bbox"][0] - m, 4), round(s["bbox"][1] - m, 4),
                                  round(s["bbox"][2] + m, 4), round(s["bbox"][3] + m, 4)]
        all_piers = s["id"] in wide or args.wide_margin is not None
        try:
            osm = fetch(b, all_piers=all_piers)
        except Exception as e:
            print(f"FAILED {s['id']}: {str(e)[:200]}")
            continue
        doc = {"site": s["id"], "bbox": b, "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "osm_base": (osm.get("osm3s") or {}).get("timestamp_osm_base"), "query": query(b, all_piers), "all_piers": all_piers,
               "license": "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
               "features": simplify(osm)}
        (args.out / f"{s['id']}-osm-areas.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n",
                                                            encoding="utf-8")
        print(f"{s['id']}: {len(doc['features'])} features, bbox {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
