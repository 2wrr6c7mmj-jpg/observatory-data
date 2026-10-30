"""Trial: Sentinel-1 ship-candidate counts for one site (仕様書 v0 第8節の2).

  python -m sat.run --site ZJG --days 12 --max-scenes 3

Writes docs/data/sat/: ship_counts.csv, detections/YYYY-MM.geojson,
quicklook/<site>/<time>.png and usage.csv (Sentinel Hub processing units per
request, for 4部). Already processed scenes are skipped. No Cloudflare usage.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

from sat import detect, sentinel_hub as sh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
SITES = json.loads((Path(__file__).with_name("sites.json")).read_text(encoding="utf-8"))
PIXEL_M = 10
RUN_VERSION = "sat-trial-1"
COUNT_FIELDS = ["site", "acquired_utc", "retrieved_utc", "platform", "orbit_state", "relative_orbit", "scene_id",
                "harbour_candidates", "quay_candidates", "confirmed", "water_mask", "detector", "run_version", "status"]
USAGE_FIELDS = ["requested_utc", "site", "scene_id", "request", "processing_units", "width", "height"]
ATTRIBUTION = "Contains modified Copernicus Sentinel data %d"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, rows):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def grid(site):
    box = sh.bbox(site)
    w = max(1, round(site["width_km"] * 1000 / PIXEL_M))
    h = max(1, round(site["height_km"] * 1000 / PIXEL_M))
    return box, w, h


def to_lonlat(box, w, h, row, col):
    west, south, east, north = box
    return round(west + (col + 0.5) / w * (east - west), 6), round(north - (row + 0.5) / h * (north - south), 6)


def quicklook(path, vv, masks, cands, year):
    from PIL import Image, ImageDraw
    v = np.clip((detect.db(vv) + 25) / 25, 0, 1)
    img = Image.fromarray((v * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for c in cands:
        r = 6
        d.ellipse([c["col"] - r, c["row"] - r, c["col"] + r, c["row"] + r], outline=(255, 60, 60) if c["zone"] == "港内" else (255, 200, 0), width=2)
    edge = masks["land"] ^ np.roll(masks["land"], 1, 0)  # rough land/water outline for checking the mask
    arr = np.array(img)
    arr[edge] = (0.6 * arr[edge] + 0.4 * np.array([0, 160, 255])).astype(np.uint8)
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).text((6, img.height - 16), ATTRIBUTION % year + " / candidates, unconfirmed", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def process(site_id, days, max_scenes, client=sh):
    site = SITES["sites"][site_id]
    box, w, h = grid(site)
    done = {r["scene_id"] for r in read_csv(OUT / "ship_counts.csv") if r["site"] == site_id}
    tok = client.token()
    end = dt.datetime.now(dt.timezone.utc)
    scenes = client.search(tok, box, (end - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ"))
    todo = [s for s in scenes if s["id"] not in done][-max_scenes:]
    usage = [dict(requested_utc=now(), site=site_id, scene_id="", request="catalog", processing_units="", width="", height="")]
    print(json.dumps({"site": site_id, "bbox": box, "scenes_found": len(scenes), "to_process": len(todo)}, ensure_ascii=False))
    for s in todo:
        import tifffile
        raw, pu = client.cutout(tok, box, s["datetime"], w, h)
        usage.append(dict(requested_utc=now(), site=site_id, scene_id=s["id"], request="process",
                          processing_units="" if pu is None else pu, width=w, height=h))
        arr = tifffile.imread(io.BytesIO(raw)).astype("float32")
        vv, vh, valid = arr[..., 0], arr[..., 1], arr[..., 2] > 0
        row = dict(site=site_id, acquired_utc=s["datetime"], retrieved_utc=now(), platform=s["platform"], orbit_state=s["orbit_state"],
                   relative_orbit=s["relative_orbit"], scene_id=s["id"], confirmed="", water_mask="image-trial",
                   detector=detect.DETECTOR_VERSION, run_version=RUN_VERSION)
        if valid.mean() < 0.5:
            row.update(harbour_candidates="", quay_candidates="", status="範囲の半分以上が撮影外")
            append_csv(OUT / "ship_counts.csv", COUNT_FIELDS, [row])
            continue
        lat_m = (box[3] - box[1]) * 110574 / h
        lon_m = (box[2] - box[0]) * 111320 * math.cos(math.radians(site["lat"])) / w
        cands, masks = detect.detect(vv, vh, valid, (lat_m + lon_m) / 2)
        row.update(harbour_candidates=sum(c["zone"] == "港内" for c in cands), quay_candidates=sum(c["zone"] == "岸壁帯" for c in cands),
                   status="試行（陸マスクは画像から・固定物マスクなし）")
        append_csv(OUT / "ship_counts.csv", COUNT_FIELDS, [row])
        month = s["datetime"][:7]
        gpath = OUT / "detections" / f"{month}.geojson"
        fc = json.loads(gpath.read_text(encoding="utf-8")) if gpath.exists() else {"type": "FeatureCollection", "features": []}
        for c in cands:
            lon, lat = to_lonlat(box, w, h, c["row"], c["col"])
            fc["features"].append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                                   "properties": {"site": site_id, "acquired_utc": s["datetime"], "scene_id": s["id"],
                                                  "pixels": c["pixels"], "peak_vv_db": c["peak_vv_db"], "peak_vh_db": c["peak_vh_db"],
                                                  "length_class": c["length_class"], "zone": c["zone"], "status": "候補",
                                                  "reliability": "A", "credibility": "3", "detector": detect.DETECTOR_VERSION}})
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gpath.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        quicklook(OUT / "quicklook" / site_id / (s["datetime"][:19].replace(":", "") + "Z.png"), vv, masks, cands, int(s["datetime"][:4]))
    append_csv(OUT / "usage.csv", USAGE_FIELDS, usage)
    spent = [float(u["processing_units"]) for u in usage if u["processing_units"] != ""]
    print(json.dumps({"processed": len(todo), "processing_units": round(sum(spent), 3) if spent else None}, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="ZJG")
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--max-scenes", type=int, default=3)
    a = ap.parse_args()
    try:
        return process(a.site, a.days, a.max_scenes)
    except sh.AuthError as e:
        print("::error title=CDSE 認証エラー::" + str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
