"""Locate naval quays and shipyards for sat/ports.json (probe only): OSM features and a cheap S2 overview.

For each region of sat/port_regions.json (a wide box, about 15-40 km):
- OpenStreetMap (Overpass): military=naval_base / landuse=military / industrial=shipyard / man_made=pier
  with a name, and their centres and outlines (a public source with coordinates).
- One Sentinel-2 L2A natural-colour overview at ``--m`` metres (least cloudy mosaic of the last 120 days,
  B04/B03/B02, about 1-3 PU) with a 0.01 degree grid, labels every 0.05 degree, and the OSM outlines drawn.

    python -m sat.port_locate --out DIR --budget-month YYYY-MM --cap PU --global-cap PU --ocean-cap PU \
        [--global-margin PU] [--usage CSV ...] [--ports-usage CSV ...] [--m 30]

The overview requests go through the same monthly guard as ``sat.port_archive plan`` (the month the caps were
set for, the ports cap, the global cap with its margin, the ocean reserve; the rows of every branch's usage
CSVs summed, each record once) and the same capped client (sat.port_archive.LimitedClient), and the PU are recorded in the
archive's usage (docs/data/sat/ports/locate/usage.csv, by the workflow).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, png
from .grid import Grid, m_per_deg
from .osm_mask import OVERPASS_URL, USER_AGENT

REGIONS_FILE = Path(__file__).resolve().parent / "port_regions.json"
OVERVIEW_DAYS = 120

OV_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "dataMask"]}], output: {bands: 3, sampleType: "UINT8"}};
}
function f(x) { return Math.max(0, Math.min(255, Math.round(Math.pow(Math.min(x / 0.3, 1), 1 / 2.2) * 255))); }
function evaluatePixel(s) { return s.dataMask ? [f(s.B04), f(s.B03), f(s.B02)] : [0, 0, 0]; }
"""


def overpass_query(b: list[float]) -> str:
    w, s, e, n = b
    bb = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:120];('
            f'nwr["military"]{bb};'
            f'nwr["landuse"="military"]{bb};'
            f'nwr["industrial"="shipyard"]{bb};'
            f'nwr["landuse"="industrial"]["name"~"船|Shipyard|shipyard|Shipbuilding"]{bb};'
            f'nwr["natural"~"^(reef|island|islet)$"]["name"]{bb};'
            f'nwr["man_made"~"^(pier|breakwater)$"]["name"]{bb};'
            f');out tags geom;')


def fetch(b: list[float]) -> dict:
    data = urllib.parse.urlencode({"data": overpass_query(b)}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(30)
    return {}


def outlines(el: dict) -> list[list[tuple[float, float]]]:
    if el.get("type") == "way" and el.get("geometry"):
        return [[(p["lon"], p["lat"]) for p in el["geometry"]]]
    if el.get("type") == "relation":
        return [[(p["lon"], p["lat"]) for p in m.get("geometry") or []] for m in el.get("members") or []
                if m.get("geometry") and m.get("role") in ("outer", "")]
    if el.get("type") == "node":
        return [[(el["lon"], el["lat"])]]
    return []


def summarize(osm: dict) -> list[dict]:
    out = []
    for el in osm.get("elements") or []:
        tags = el.get("tags") or {}
        pts = [p for o in outlines(el) for p in o]
        if not pts:
            continue
        lon = [p[0] for p in pts]
        lat = [p[1] for p in pts]
        out.append({"osm": f"{el['type']}/{el['id']}", "name": tags.get("name") or tags.get("name:en") or "",
                    "tags": {k: v for k, v in tags.items() if k in ("military", "landuse", "industrial", "name:en", "operator")},
                    "center": [round((min(lon) + max(lon)) / 2, 5), round((min(lat) + max(lat)) / 2, 5)],
                    "bbox": [round(min(lon), 5), round(min(lat), 5), round(max(lon), 5), round(max(lat), 5)]})
    out.sort(key=lambda x: (x["tags"].get("military") is None, -((x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))))
    return out


def overview(client, b: list[float], m: float, now: datetime) -> tuple[np.ndarray, float]:
    mlat, mlon = m_per_deg((b[1] + b[3]) / 2)
    w = int(round((b[2] - b[0]) * mlon / m))
    h = int(round((b[3] - b[1]) * mlat / m))
    payload = {
        "input": {"bounds": {"bbox": b, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": cdse.iso(now - timedelta(days=OVERVIEW_DAYS)), "to": cdse.iso(now)},
                                           "mosaickingOrder": "leastCC", "maxCloudCoverage": 30}}]},
        "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": OV_EVALSCRIPT,
    }
    data, pu = client.process_raw(payload, "image/png", 3)
    return png.decode(data), pu


def draw(rgb: np.ndarray, b: list[float], feats: list[dict], title: str, credit: str = "") -> bytes:
    from PIL import Image, ImageDraw
    from .color import _font
    h, w = rgb.shape[:2]
    im = Image.fromarray(rgb.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    font = _font(11)
    X = lambda lon: (lon - b[0]) / (b[2] - b[0]) * w
    Y = lambda lat: (b[3] - lat) / (b[3] - b[1]) * h
    k0 = int(np.ceil(b[0] * 100))
    for k in range(k0, int(b[2] * 100) + 1):
        x = X(k / 100)
        major = k % 5 == 0
        d.line([(x, 0), (x, h)], fill=(255, 255, 0) if major else (120, 120, 0), width=1)
        if major:
            d.text((x + 2, 2), f"{k / 100:.2f}E", fill=(255, 255, 0), font=font)
    for k in range(int(np.ceil(b[1] * 100)), int(b[3] * 100) + 1):
        y = Y(k / 100)
        major = k % 5 == 0
        d.line([(0, y), (w, y)], fill=(255, 255, 0) if major else (120, 120, 0), width=1)
        if major:
            d.text((2, y + 2), f"{k / 100:.2f}N", fill=(255, 255, 0), font=font)
    for i, f in enumerate(feats[:40]):
        col = (255, 60, 60) if f["tags"].get("military") or f["tags"].get("landuse") == "military" else (60, 200, 255)
        x0, y0, x1, y1 = X(f["bbox"][0]), Y(f["bbox"][3]), X(f["bbox"][2]), Y(f["bbox"][1])
        d.rectangle([x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)], outline=col, width=2)
        d.text((x0 + 2, y1 + 1), str(i + 1), fill=col, font=font)
    d.text((4, h - 30 if credit else h - 16), title, fill=(255, 255, 255), font=font)
    if credit:
        d.text((4, h - 16), credit, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=82, method=6)
    return buf.getvalue()


def overview_credit(now: datetime) -> str:
    """The credit of a least-cloud mosaic of the last OVERVIEW_DAYS days (a year range across the new year)."""
    from .port_archive import credit
    return credit([(now - timedelta(days=OVERVIEW_DAYS)).year, now.year])


def main(argv=None) -> int:
    from . import port_archive as pa
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=Path, default=REGIONS_FILE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--m", type=float, default=30)
    ap.add_argument("--only", nargs="*")
    pa.add_guard_args(ap)
    ap.add_argument("--max-pu", type=float, help="cap of this run, inside the guard")
    ap.add_argument("--pu-per-min", type=float, default=80)
    ap.add_argument("--req-per-min", type=int, default=80)
    ap.add_argument("--max-minutes", type=float, default=30)
    args = ap.parse_args(argv)
    regions = json.loads(args.regions.read_text(encoding="utf-8"))["regions"]
    now = pa._utcnow()
    try:
        gd = pa.guard(args.budget_month, now, args.cap, args.global_cap, args.global_margin, args.ocean_cap,
                      args.usage, args.ports_usage, args.reserve, args.max_pu)
    except ValueError as e:
        print(f"locate: {e}", file=sys.stderr)
        return 2
    print(f"locate: 残り {gd['cap']:.1f} PU（港の使用 {gd['done_pu']:.1f}、他 {gd['other_pu']:.1f}、予約 {gd['reserve']:.1f}）",
          file=sys.stderr)
    client = pa.LimitedClient(gd["cap"], max(0, gd["req_cap"]), args.pu_per_min, args.req_per_min,
                              deadline=time.monotonic() + args.max_minutes * 60, budget_month=args.budget_month)
    args.out.mkdir(parents=True, exist_ok=True)
    res = []
    for r in regions:
        if (args.only and r["id"] not in args.only) or (not args.only and r.get("done")):
            continue
        b = r["bbox"]
        try:
            feats = summarize(fetch(b))
        except Exception as e:
            feats = []
            print(f"{r['id']}: OSM 失敗 {str(e)[:200]}", file=sys.stderr)
        time.sleep(10)
        pu = None
        stop = None
        try:
            rgb, pu = overview(client, b, args.m, now)
            (args.out / f"{r['id']}.webp").write_bytes(draw(rgb, b, feats, f"{r['id']} {r.get('name', '')} S2 L2A leastCC {OVERVIEW_DAYS}d",
                                                            overview_credit(now)))
        except pa.BudgetStop as e:
            stop = str(e)
            print(f"{r['id']}: {e}", file=sys.stderr)
        except cdse.CredentialError:
            raise
        except Exception as e:
            print(f"{r['id']}: S2 失敗 {str(e)[:200]}", file=sys.stderr)
        res.append({"id": r["id"], "bbox": b, "pu": pu, "features": feats[:60]})
        print(f"{r['id']}: OSM {len(feats)} 件、S2 {pu} PU", file=sys.stderr)
        if stop:
            break
    prev = args.out / "locate.json"
    if prev.exists():
        ids = {r["id"] for r in res}
        res = [r for r in json.loads(prev.read_text(encoding="utf-8"))["regions"] if r["id"] not in ids] + res
    (args.out / "locate.json").write_text(json.dumps({"run_utc": cdse.iso(now), "regions": res,
                                                        "requests": client.requests, "pu": round(client.processing_units, 3),
                                                        "osm_license": "(c) OpenStreetMap contributors, ODbL 1.0"},
                                                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"requests": client.requests, "pu": round(client.processing_units, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
