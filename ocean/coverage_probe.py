"""Probe: how often Sentinel-1 images the open sea around Japan, and how late it is published.

Reads the CDSE OData catalogue (no login, no processing units) for the past N days
over three sea areas and counts, per 0.25 degree cell, the distinct Sentinel-1 IW
passes whose GRD footprint covers the cell centre; per area, the relative orbits,
orbit directions and acquisition times (UTC); which other modes (EW, WV, SM) appear;
and the publication delay (PublicationDate - ContentDate/Start) per product type.

    python -m ocean.coverage_probe --days 60 --out coverage_probe.json --report DIR

Probe only: it writes nowhere in the repository unless --out points there. Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
USER_AGENT = "observatory-data/1.0 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"
STEP = 0.25
PAGE = 1000
WINDOW_DAYS = 5
PROBE_VERSION = "ocean-coverage-probe-1"

# [west, south, east, north]
AREAS = {
    "T": {"name": "台湾周辺", "bbox": [118.5, 21.0, 123.5, 26.5]},
    "E": {"name": "東シナ海", "bbox": [122.0, 26.0, 129.0, 32.0]},
    "R": {"name": "南西諸島・沖縄周辺", "bbox": [122.0, 23.0, 131.0, 29.0]},
}

NAME_RE = re.compile(
    r"^S1(?P<sat>[A-D])_(?P<mode>[A-Z0-9]{2})_(?P<type>[A-Z]{3})(?P<res>[A-Z_])_(?P<level>\d)(?P<cls>[A-Z])(?P<pol>[A-Z]{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<abs>\d{6})_(?P<dt>[0-9A-F]{6})_")


def iso_ms(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_time(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "." in s:  # OData gives up to 6-7 fraction digits
        head, rest = s.split(".", 1)
        frac = re.match(r"\d+", rest).group(0)
        s = head + "." + frac[:6].ljust(6, "0") + rest[len(frac):]
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def union_bbox(areas: dict) -> list[float]:
    b = [a["bbox"] for a in areas.values()]
    return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]


def wkt_box(b: list[float]) -> str:
    w, s, e, n = b
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def query_url(bbox: list[float], t0: datetime, t1: datetime, skip: int = 0) -> str:
    # GRD and OCN only (no RAW/SLC): enough to see every mode (IW, EW, WV, SM) that was acquired.
    flt = (f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_box(bbox)}') "
           f"and ContentDate/Start ge {iso_ms(t0)} and ContentDate/Start lt {iso_ms(t1)} "
           f"and (contains(Name,'_GRD') or contains(Name,'_OCN_'))")
    q = {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": str(PAGE)}
    if skip:
        q["$skip"] = str(skip)
    return ODATA_URL + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def http_json(url: str, tries: int = 4, timeout: float = 120) -> dict:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = str(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"OData の問い合わせに失敗: {last}")


def fetch_products(bbox: list[float], start: datetime, end: datetime, get=http_json, log=print) -> tuple[list[dict], int]:
    """All products in [start, end), queried in WINDOW_DAYS windows and pages. Returns (products, requests)."""
    out, seen, requests = [], set(), 0
    t0 = start
    while t0 < end:
        t1 = min(end, t0 + timedelta(days=WINDOW_DAYS))
        url, skip = query_url(bbox, t0, t1), 0
        while url:
            page = get(url)
            requests += 1
            items = page.get("value", [])
            for p in items:
                if p.get("Id") not in seen:
                    seen.add(p.get("Id"))
                    out.append(p)
            nxt = page.get("@odata.nextLink")
            if nxt:
                url = nxt
            elif len(items) >= PAGE:
                skip += PAGE
                url = query_url(bbox, t0, t1, skip)
            else:
                url = None
            time.sleep(0.5 if url else 0)
        log(f"{iso_ms(t0)[:10]}〜{iso_ms(t1)[:10]}: 累計 {len(out)} 件")
        t0 = t1
    return out, requests


def attrs(p: dict) -> dict:
    return {a.get("Name"): a.get("Value") for a in p.get("Attributes") or []}


def rings(geom: dict) -> list[list[list[float]]]:
    """Outer rings of a GeoJSON Polygon / MultiPolygon (holes are ignored: footprints have none)."""
    if not geom:
        return []
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return [c[0]] if c else []
    if t == "MultiPolygon":
        return [poly[0] for poly in c if poly]
    return []


def wkt_rings(s: str) -> list[list[list[float]]]:
    """Rings of a WKT (MULTI)POLYGON, as a fallback when GeoFootprint is missing."""
    s = s.split(";", 1)[-1].rstrip("'")
    out = []
    for grp in re.findall(r"\(\(([^()]*)\)", s):
        out.append([[float(v) for v in pt.split()[:2]] for pt in grp.split(",")])
    return out


def inside(ring: list[list[float]], x: float, y: float) -> bool:
    res = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            res = not res
        j = i
    return res


def ring_bbox(r):
    xs, ys = [p[0] for p in r], [p[1] for p in r]
    return min(xs), min(ys), max(xs), max(ys)


def product_info(p: dict) -> dict | None:
    m = NAME_RE.match(p.get("Name", ""))
    if not m:
        return None
    a = attrs(p)
    start = parse_time(p["ContentDate"]["Start"]) if p.get("ContentDate", {}).get("Start") else None
    pub = parse_time(p["PublicationDate"]) if p.get("PublicationDate") else None
    ptype = a.get("productType") or f"{m['mode']}_{m['type']}{m['res']}_{m['level']}{m['cls']}"
    if p.get("Name", "").endswith("_COG.SAFE") and not ptype.endswith("COG"):
        ptype += "-COG"
    rr = rings(p.get("GeoFootprint") or {})
    if not rr and p.get("Footprint"):
        rr = wkt_rings(p["Footprint"])
    rel = a.get("relativeOrbitNumber")
    return {
        "name": p["Name"], "platform": "S1" + m["sat"], "mode": m["mode"], "kind": m["type"],
        "product_type": ptype, "cog": p["Name"].endswith("_COG.SAFE"),
        "absolute_orbit": int(m["abs"]), "relative_orbit": int(rel) if rel not in (None, "") else None,
        "direction": (a.get("orbitDirection") or "").lower() or None,
        "start": start, "published": pub, "rings": rr,
    }


def area_cells(bbox: list[float], step: float = STEP) -> tuple[int, int]:
    w, s, e, n = bbox
    return int(round((e - w) / step)), int(round((n - s) / step))


def cell_centre(bbox, ix, iy, step=STEP):
    """ix from the west, iy from the north (row 0 = northern edge)."""
    return bbox[0] + (ix + 0.5) * step, bbox[3] - (iy + 0.5) * step


def touches(rr, bbox) -> bool:
    for r in rr:
        x0, y0, x1, y1 = ring_bbox(r)
        if x1 >= bbox[0] and x0 <= bbox[2] and y1 >= bbox[1] and y0 <= bbox[3]:
            return True
    return False


def quantiles(vals: list[float], qs=(0.1, 0.5, 0.9)) -> list[float]:
    v = sorted(vals)
    out = []
    for q in qs:
        k = (len(v) - 1) * q
        f, c = math.floor(k), math.ceil(k)
        out.append(v[f] + (v[c] - v[f]) * (k - f))
    return out


def lag_stats(hours: list[float]) -> dict:
    if not hours:
        return {"n": 0}
    p10, med, p90 = quantiles(hours)
    return {"n": len(hours), "min_h": round(min(hours), 2), "p10_h": round(p10, 2), "median_h": round(med, 2),
            "p90_h": round(p90, 2), "max_h": round(max(hours), 2)}


def analyse(products: list[dict], start: datetime, end: datetime, areas: dict = AREAS) -> dict:
    infos = [i for i in (product_info(p) for p in products) if i]
    types: dict[str, int] = {}
    for i in infos:
        types[i["product_type"]] = types.get(i["product_type"], 0) + 1
    # Publication delay per product type (all products of the union box).
    lags: dict[str, list[float]] = {}
    for i in infos:
        if i["start"] and i["published"]:
            lags.setdefault(i["product_type"], []).append((i["published"] - i["start"]).total_seconds() / 3600)
    # IW GRD (one product per slice; the COG copy is the same slice): the coverage source.
    iw = [i for i in infos if i["mode"] == "IW" and i["kind"] == "GRD"]
    out_areas = {}
    for aid, area in areas.items():
        bbox = area["bbox"]
        nx, ny = area_cells(bbox)
        passes: dict[tuple, dict] = {}
        for i in iw:
            if not i["rings"] or not touches(i["rings"], bbox):
                continue
            key = (i["platform"], i["absolute_orbit"])
            ps = passes.setdefault(key, {"platform": i["platform"], "absolute_orbit": i["absolute_orbit"],
                                         "relative_orbit": i["relative_orbit"], "direction": i["direction"],
                                         "start": i["start"], "rings": []})
            if i["start"] and (ps["start"] is None or i["start"] < ps["start"]):
                ps["start"] = i["start"]
            ps["rings"].extend(i["rings"])
        counts = [[0] * nx for _ in range(ny)]
        by_dir = {"ascending": [[0] * nx for _ in range(ny)], "descending": [[0] * nx for _ in range(ny)]}
        for ps in passes.values():
            hit = set()
            for r in ps["rings"]:
                x0, y0, x1, y1 = ring_bbox(r)
                ix0 = max(0, int(math.floor((x0 - bbox[0]) / STEP)))
                ix1 = min(nx - 1, int(math.floor((x1 - bbox[0]) / STEP)))
                iy0 = max(0, int(math.floor((bbox[3] - y1) / STEP)))
                iy1 = min(ny - 1, int(math.floor((bbox[3] - y0) / STEP)))
                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        if (ix, iy) not in hit and inside(r, *cell_centre(bbox, ix, iy)):
                            hit.add((ix, iy))
            ps["cells"] = len(hit)
            for ix, iy in hit:
                counts[iy][ix] += 1
                if ps["direction"] in by_dir:
                    by_dir[ps["direction"]][iy][ix] += 1
        covering = [p for p in passes.values() if p["cells"] > 0]
        flat = [c for row in counts for c in row]
        orbits: dict[tuple, dict] = {}
        for p in covering:
            k = (p["direction"], p["relative_orbit"])
            o = orbits.setdefault(k, {"direction": p["direction"], "relative_orbit": p["relative_orbit"],
                                      "platforms": set(), "passes": 0, "times_utc": [], "max_cells": 0})
            o["platforms"].add(p["platform"])
            o["passes"] += 1
            o["max_cells"] = max(o["max_cells"], p["cells"])
            if p["start"]:
                o["times_utc"].append(p["start"].strftime("%H:%M"))
        orbit_list = []
        for o in sorted(orbits.values(), key=lambda o: (o["direction"] or "", o["relative_orbit"] or 0)):
            t = sorted(o["times_utc"])
            orbit_list.append({"direction": o["direction"], "relative_orbit": o["relative_orbit"],
                               "platforms": sorted(o["platforms"]), "passes": o["passes"],
                               "cells_max": o["max_cells"], "share_of_cells_max": round(o["max_cells"] / (nx * ny), 3),
                               "utc_first": t[0] if t else None, "utc_last": t[-1] if t else None})
        hours = {}
        for p in covering:
            if p["start"]:
                h = f"{p['start'].hour:02d}"
                d = hours.setdefault(h, {"ascending": 0, "descending": 0, "other": 0})
                d[p["direction"] if p["direction"] in ("ascending", "descending") else "other"] += 1
        # Revisit: gaps between consecutive passes over the area (any cell).
        times = sorted(p["start"] for p in covering if p["start"])
        gaps = [(b - a).total_seconds() / 3600 for a, b in zip(times, times[1:])]
        other = {}
        for i in infos:
            if i["mode"] != "IW" and i["rings"] and touches(i["rings"], bbox):
                o = other.setdefault(i["mode"], {"products": 0, "product_types": set(), "passes": set(),
                                                 "first_utc": None, "last_utc": None})
                o["products"] += 1
                o["product_types"].add(i["product_type"])
                o["passes"].add((i["platform"], i["absolute_orbit"]))
                ts = i["start"].strftime("%Y-%m-%dT%H:%MZ") if i["start"] else None
                if ts:
                    o["first_utc"] = min(filter(None, [o["first_utc"], ts]))
                    o["last_utc"] = max(filter(None, [o["last_utc"], ts]))
        out_areas[aid] = {
            "name": area["name"], "bbox": bbox,
            "passes_iw": len(covering),
            "passes_iw_by_direction": {d: sum(p["direction"] == d for p in covering) for d in ("ascending", "descending")},
            "passes_iw_by_platform": {pl: sum(p["platform"] == pl for p in covering)
                                      for pl in sorted({p["platform"] for p in covering})},
            "cells": {"step_deg": STEP, "nx": nx, "ny": ny, "row0": "north", "col0": "west",
                      "min": min(flat) if flat else 0, "median": quantiles(flat, (0.5,))[0] if flat else 0,
                      "max": max(flat) if flat else 0, "zero": sum(c == 0 for c in flat),
                      "mean": round(sum(flat) / len(flat), 2) if flat else 0},
            "revisit_gap_h": lag_stats(gaps),
            "orbits": orbit_list,
            "utc_hours": dict(sorted(hours.items())),
            "other_modes": {m: {"products": o["products"], "passes": len(o["passes"]),
                                "product_types": sorted(o["product_types"]),
                                "first_utc": o["first_utc"], "last_utc": o["last_utc"]} for m, o in sorted(other.items())},
            "counts": counts,
            "counts_by_direction": by_dir,
        }
    return {
        "probe": PROBE_VERSION,
        "window_utc": {"start": iso_ms(start)[:19] + "Z", "end": iso_ms(end)[:19] + "Z",
                       "days": round((end - start).total_seconds() / 86400, 2)},
        "source": "CDSE OData catalogue (catalogue.dataspace.copernicus.eu)。認証なし、処理単位（PU）は使わない",
        "method": ("Sentinel-1 の GRD・OCN の製品を和の矩形で問い合わせ、IW GRD のスライスを (衛星, 絶対軌道) で 1 通過にまとめ、"
                   "0.25° の格子の中心が足跡に入る通過を数える（COG の写しは同じスライスなので重ねて数えない）。"
                   "公開の遅れ＝PublicationDate − ContentDate/Start（製品の種類ごと）"),
        "attribution": attribution(start, end),
        "notes": NOTES,
        "products": len(infos),
        "product_types": dict(sorted(types.items())),
        "publication_lag": {t: lag_stats(v) for t, v in sorted(lags.items())},
        "areas": out_areas,
    }


NOTES = [
    "counts は 0.25° 格子の中心が IW GRD の足跡に入った通過の数（行 0＝北端、列 0＝西端）。陸の格子も含む",
    "1 通過＝同じ衛星・同じ絶対軌道のスライスの集まり。COG の写し（-COG）は同じスライスなので数えない",
    "revisit_gap_h は海域のどこかを撮った通過どうしの間隔（格子ごとの再訪ではない。格子ごとは counts を 60 日で割る）",
    "other_modes は GRD・OCN の製品がある IW 以外のモード（RAW・SLC は問い合わせていない）",
    "公開の遅れはカタログへの公開（PublicationDate）まで。Sentinel Hub で処理できるまでの遅れは別",
]


def attribution(start: datetime, end: datetime) -> str:
    y0, y1 = start.year, end.year
    return f"Contains modified Copernicus Sentinel data {y0 if y0 == y1 else f'{y0}-{y1}'}（カタログの情報）"


SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyz"


def char_map(counts: list[list[int]]) -> str:
    return "\n".join("".join(SYMBOLS[min(c, len(SYMBOLS) - 1)] for c in row) for row in counts)


def report(res: dict) -> dict[str, str]:
    """Texts for annotations (each well under 4096 characters)."""
    w = res["window_utc"]
    lines = [f"窓 {w['start']}〜{w['end']}（{w['days']} 日）、製品 {res['products']} 件",
             "種類別: " + ", ".join(f"{k}={v}" for k, v in res["product_types"].items())]
    for t, s in res["publication_lag"].items():
        if s.get("n"):
            lines.append(f"公開の遅れ {t}: n={s['n']} 中央値 {s['median_h']} h（10% {s['p10_h']}・90% {s['p90_h']}、最小 {s['min_h']}・最大 {s['max_h']}）")
    for aid, a in res["areas"].items():
        c = a["cells"]
        lines.append(f"[{aid} {a['name']} {a['bbox']}] IW 通過 {a['passes_iw']}（{a['passes_iw_by_direction']}、{a['passes_iw_by_platform']}）"
                     f" 格子あたり 最小 {c['min']}・中央値 {c['median']}・平均 {c['mean']}・最大 {c['max']}、0 の格子 {c['zero']}/{c['nx'] * c['ny']}"
                     f"、通過の間隔 中央値 {a['revisit_gap_h'].get('median_h')} h（最大 {a['revisit_gap_h'].get('max_h')} h）"
                     f"、他のモード {json.dumps({m: o['passes'] for m, o in a['other_modes'].items()}, ensure_ascii=False)}")
    out = {"summary": "\n".join(lines)}
    for aid, a in res["areas"].items():
        orb = "\n".join(f"{o['direction']} RO{o['relative_orbit']} {','.join(o['platforms'])} 通過{o['passes']}"
                        f" UTC {o['utc_first']}〜{o['utc_last']} 最大{o['share_of_cells_max']:.0%}" for o in a["orbits"])
        hrs = " ".join(f"{h}時:{d['ascending']}/{d['descending']}" for h, d in a["utc_hours"].items())
        out[f"map-{aid}"] = (f"{aid} {a['name']}：0.25° 格子ごとの IW 通過数（60 日、北が上・西が左、0-9 のあと a=10…）\n"
                             + char_map(a["counts"]))
        out[f"orbits-{aid}"] = f"{aid} 相対軌道（向き RO 衛星 通過数 時刻 格子の最大被覆）\n{orb}\nUTC 時（上昇/下降）: {hrs}"
        out[f"modes-{aid}"] = f"{aid} IW 以外: " + json.dumps(a["other_modes"], ensure_ascii=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, help="directory for annotation texts")
    args = ap.parse_args(argv)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    products, requests = fetch_products(union_bbox(AREAS), start, end)
    res = analyse(products, start, end)
    res["requests"] = requests
    res["generated_utc"] = iso_ms(datetime.now(timezone.utc))[:19] + "Z"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Compact: one line per grid row keeps the file small and diffable.
    args.out.write_text(dumps(res) + "\n", encoding="utf-8")
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        for k, v in report(res).items():
            (args.report / f"{k}.txt").write_text(v + "\n", encoding="utf-8")
    print(report(res)["summary"])
    return 0


def dumps(res: dict) -> str:
    s = json.dumps(res, ensure_ascii=False, indent=1)
    # put each list of ints on one line
    return re.sub(r"\[\s+((?:-?\d+,\s+)*-?\d+)\s+\]", lambda m: "[" + re.sub(r",\s+", ",", m.group(1)) + "]", s)


if __name__ == "__main__":
    sys.exit(main())
