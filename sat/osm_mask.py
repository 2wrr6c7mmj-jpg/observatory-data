"""Fixed land mask per site from OpenStreetMap (versioned, committed to the repo).

Built once per version (``python -m sat.osm_mask ZJG``), not on every run:
natural=coastline ways are drawn as barriers, the free pixels are split into
4-connected regions, and each region is called land or water by the side of
the coastline it lies on (OSM convention: land on the left of the way).
Piers, breakwaters, groynes and road/rail bridges are added as structures.

Mask values: 0 water, 1 land, 2 structure (pier / breakwater / bridge).
Data (c) OpenStreetMap contributors, ODbL 1.0.

Land fixes (``sat/masks/<site>-land-fix.geojson``, optional): polygons made land after the OSM
classification, for places where the OSM coastline misses reclaimed land (checked against
Sentinel-2). Each feature has ``id``, ``since_mask_version`` (applied to that mask version and
later ones only) and ``source``; ``whole_regions`` makes land the whole 4-connected water regions
the polygon touches (a small pocket cut off by the coastline), up to ``max_pixels``. Only water
pixels change (structures stay). ``--from-version N`` applies the fixes of the new version to the
committed mask N without a new Overpass request (the OSM data stay those of mask N).
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


def fix_path(site_id: str, mask_dir: Path = MASK_DIR) -> Path:
    return mask_dir / f"{site_id}-land-fix.geojson"


def load_land_fix(site_id: str, version: int, mask_dir: Path = MASK_DIR) -> list[dict]:
    """The land fixes of a site that apply to mask ``version`` ([] without a fix file)."""
    p = fix_path(site_id, mask_dir)
    if not p.exists():
        return []
    out = []
    for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
        pr, geom = f["properties"], f["geometry"]
        if geom["type"] != "Polygon":
            raise ValueError(f"{p.name}: {pr.get('id')} は Polygon でない")
        if int(pr["since_mask_version"]) > version:
            continue
        out.append({"id": pr["id"], "polygon": [[float(x), float(y)] for x, y in geom["coordinates"][0]],
                    "whole_regions": bool(pr.get("whole_regions")), "max_pixels": int(pr.get("max_pixels", 500))})
    return out


def apply_land_fix(grid: Grid, mask: np.ndarray, fixes: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Water pixels whose centre is in a fix polygon (or, with whole_regions, the water regions the
    polygon touches) made land. Returns (new mask, [{id, pixels}])."""
    from .sectors import _in_polygon
    out = mask.copy()
    if not fixes:
        return out, []
    lon, lat = grid.to_lonlat(np.arange(grid.height)[:, None] + 0.5, np.arange(grid.width)[None, :] + 0.5)
    lon, lat = np.broadcast_to(lon, out.shape), np.broadcast_to(lat, out.shape)
    report = []
    for fx in fixes:
        target = _in_polygon(lon, lat, fx["polygon"]) & (out == WATER)
        if fx.get("whole_regions") and target.any():
            labels, _ = label(out == WATER, connectivity=4)
            ids = np.unique(labels[target])
            target = np.isin(labels, ids[ids > 0])
            if int(target.sum()) > fx.get("max_pixels", 500):
                raise ValueError(f"land_fix {fx['id']}: 水の領域が {int(target.sum())} 画素で max_pixels を超える（海につながっている）")
        out[target] = LAND
        report.append({"id": fx["id"], "pixels": int(target.sum())})
    return out, report


def _fractions(mask: np.ndarray) -> dict:
    return {
        "water_fraction": round(float((mask == WATER).mean()), 4),
        "land_fraction": round(float((mask == LAND).mean()), 4),
        "structure_fraction": round(float((mask == STRUCTURE).mean()), 4),
    }


def build_mask(grid: Grid, osm: dict, land_fix: list[dict] | None = None) -> tuple[np.ndarray, dict]:
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
    fixed = None
    if land_fix:
        mask, fixed = apply_land_fix(grid, mask, land_fix)
    info = {
        "coastline_ways": len(coast_ways),
        "structure_ways": n_struct,
        "regions": int(n),
        "regions_without_votes": unvoted,
        **_fractions(mask),
    }
    if fixed is not None:
        info["land_fix"] = fixed
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


def fix_from_version(site: dict, grid: Grid, base: int, version: int, mask_dir: Path = MASK_DIR) -> dict:
    """Mask ``version`` = the committed mask ``base`` + the land fixes of ``version`` (no Overpass)."""
    if version <= base:
        raise ValueError("--version は --from-version より大きいこと")
    mask, bmeta = load_mask(site["id"], base, mask_dir)
    if list(bmeta["bbox"]) != grid.bbox or mask.shape != (grid.height, grid.width):
        raise ValueError(f"{bmeta['version']} の範囲が sites.json と違う")
    fixes = load_land_fix(site["id"], version, mask_dir)
    if not fixes:
        raise ValueError(f"{fix_path(site['id'], mask_dir).name} に版 {version} の修正が無い")
    new, report = apply_land_fix(grid, mask, fixes)
    p_png, p_json = mask_paths(site["id"], version, mask_dir)
    p_png.write_bytes(png.encode(new))
    meta = {**bmeta,
            "version": f"{site['id']}-land-v{version}",
            "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": f"{bmeta['source']} (= {bmeta['version']}) + land_fix ({fix_path(site['id'], mask_dir).name})",
            "base_version": bmeta["version"],
            **_fractions(new),
            "land_fix": report}
    p_json.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    from .run import load_sites
    ap = argparse.ArgumentParser(description="Build a site's land mask from OpenStreetMap")
    ap.add_argument("site")
    ap.add_argument("--version", type=int, help="mask version (default: the site's mask_version in sites.json)")
    ap.add_argument("--from-version", type=int,
                    help="apply the land fixes of --version to this committed mask instead of asking Overpass")
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    args = ap.parse_args(argv)
    site = next(s for s in load_sites() if s["id"] == args.site)
    grid = Grid.for_site(site)
    from .run import mask_version
    version = args.version or mask_version(site)
    if args.from_version:
        print(json.dumps(fix_from_version(site, grid, args.from_version, version, args.mask_dir), ensure_ascii=False, indent=1))
        return 0
    osm = fetch_osm(grid)
    mask, info = build_mask(grid, osm, load_land_fix(site["id"], version, args.mask_dir))
    meta = save_mask(site, grid, mask, info, osm, version, args.mask_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
