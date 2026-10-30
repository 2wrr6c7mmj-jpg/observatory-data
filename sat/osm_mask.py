"""Fixed land mask per site from OpenStreetMap (versioned, committed to the repo).

Built once per version (``python -m sat.osm_mask ZJG``), not on every run:
natural=coastline ways are drawn as barriers, the free pixels are split into
4-connected regions, and each region is called land or water by the side of
the coastline it lies on (OSM convention: land on the left of the way).
Piers, breakwaters, groynes and road/rail bridges are added as structures.

Mask values: 0 water, 1 land, 2 structure (pier / breakwater / bridge).
Data (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import png
from .grid import Grid, dilate, draw_line, fill_polygon, label

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "observatory-data/1.0 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"
MASK_DIR = Path(__file__).resolve().parent / "masks"
WATER, LAND, STRUCTURE = 0, 1, 2


def overpass_query(grid: Grid) -> str:
    s, w, n, e = grid.south, grid.west, grid.north, grid.east
    b = f"({s},{w},{n},{e})"
    return (f'[out:json][timeout:90];('
            f'way["natural"="coastline"]{b};'
            f'way["man_made"~"^(pier|breakwater|groyne|quay)$"]{b};'
            f'way["bridge"]["highway"]{b};'
            f'way["bridge"]["railway"]{b};'
            f');out geom;')


def fetch_osm(grid: Grid, url: str = OVERPASS_URL) -> dict:
    data = urllib.parse.urlencode({"data": overpass_query(grid)}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(30)
    raise RuntimeError("unreachable")


def _way_pixels(grid: Grid, way: dict):
    geom = way.get("geometry") or []
    lon = [p["lon"] for p in geom]
    lat = [p["lat"] for p in geom]
    return grid.to_pixel(lon, lat)


def build_mask(grid: Grid, osm: dict) -> tuple[np.ndarray, dict]:
    h, w = grid.height, grid.width
    coast = np.zeros((h, w), bool)
    coast_ways = []
    struct = np.zeros((h, w), bool)
    n_struct = 0
    for el in osm.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {})
        rows, cols = _way_pixels(grid, el)
        if tags.get("natural") == "coastline":
            draw_line(coast, rows, cols)
            coast_ways.append((rows, cols))
            continue
        n_struct += 1
        line = np.zeros((h, w), bool)
        draw_line(line, rows, cols)
        closed = len(rows) > 3 and el["geometry"][0] == el["geometry"][-1]
        if closed and tags.get("man_made") in ("pier", "breakwater", "groyne", "quay") and tags.get("area") != "no":
            fill_polygon(line, rows, cols)
        radius = 2 if "bridge" in tags else 1
        struct |= dilate(line, radius)

    labels, n = label(~coast, connectivity=4)
    votes = np.zeros((n + 1, 2), dtype=np.int64)  # [:, 0] land, [:, 1] water
    for rows, cols in coast_ways:
        for k in range(len(rows) - 1):
            r0, c0, r1, c1 = rows[k], cols[k], rows[k + 1], cols[k + 1]
            seg = math.hypot(r1 - r0, c1 - c0)
            if seg == 0:
                continue
            # Unit normal pointing to the left of travel. Rows grow southward,
            # so in (x=col, y=-row) terms left of (dx, dy) is (-dy, dx).
            dx, dy = (c1 - c0) / seg, -(r1 - r0) / seg
            lx, ly = -dy, dx
            for t in np.linspace(0, 1, max(2, int(seg / 3) + 1)):
                rm, cm = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t
                for off in (1.5, 2.5, 3.5):
                    hits = []
                    for side, sgn in ((0, 1), (1, -1)):
                        rr = int(math.floor(rm - sgn * ly * off))
                        cc = int(math.floor(cm + sgn * lx * off))
                        if 0 <= rr < h and 0 <= cc < w and labels[rr, cc]:
                            hits.append((side, labels[rr, cc]))
                    for side, lab in hits:
                        votes[lab, side] += 1
                    if hits:
                        break
    land_comp = votes[:, 0] > votes[:, 1]
    unvoted = int(((votes[1:, 0] + votes[1:, 1]) == 0).sum())
    mask = np.where(land_comp[labels], LAND, WATER).astype(np.uint8)
    mask[coast] = LAND
    mask[struct & (mask == WATER)] = STRUCTURE
    info = {
        "coastline_ways": len(coast_ways),
        "structure_ways": n_struct,
        "regions": int(n),
        "regions_without_votes": unvoted,
        "water_fraction": round(float((mask == WATER).mean()), 4),
        "land_fraction": round(float((mask == LAND).mean()), 4),
        "structure_fraction": round(float((mask == STRUCTURE).mean()), 4),
    }
    return mask, info


def mask_paths(site_id: str, version: int, mask_dir: Path = MASK_DIR) -> tuple[Path, Path]:
    stem = f"{site_id}-land-v{version}"
    return mask_dir / f"{stem}.png", mask_dir / f"{stem}.json"


def save_mask(site: dict, grid: Grid, mask: np.ndarray, info: dict, osm: dict, version: int,
              mask_dir: Path = MASK_DIR) -> dict:
    mask_dir.mkdir(parents=True, exist_ok=True)
    p_png, p_json = mask_paths(site["id"], version, mask_dir)
    p_png.write_bytes(png.encode(mask))
    meta = {
        "version": f"{site['id']}-land-v{version}",
        "site": site["id"],
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": grid.bbox,
        "width": grid.width,
        "height": grid.height,
        "values": {"0": "water", "1": "land", "2": "structure (pier/breakwater/groyne/quay/bridge)"},
        "source": "OpenStreetMap via Overpass API",
        "osm_base": (osm.get("osm3s") or {}).get("timestamp_osm_base"),
        "query": overpass_query(grid),
        "license": "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        **info,
    }
    p_json.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def load_mask(site_id: str, version: int, mask_dir: Path = MASK_DIR) -> tuple[np.ndarray, dict]:
    p_png, p_json = mask_paths(site_id, version, mask_dir)
    if not p_png.exists():
        raise FileNotFoundError(f"land mask {p_png.name} がありません。python -m sat.osm_mask {site_id} で作り、確かめてから commit する")
    return png.decode(p_png.read_bytes()), json.loads(p_json.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    from .run import load_sites
    ap = argparse.ArgumentParser(description="Build a site's land mask from OpenStreetMap")
    ap.add_argument("site")
    ap.add_argument("--version", type=int, help="mask version (default: the site's mask_version in sites.json)")
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    args = ap.parse_args(argv)
    site = next(s for s in load_sites() if s["id"] == args.site)
    grid = Grid.for_site(site)
    osm = fetch_osm(grid)
    mask, info = build_mask(grid, osm)
    from .run import mask_version
    meta = save_mask(site, grid, mask, info, osm, args.version or mask_version(site), args.mask_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
