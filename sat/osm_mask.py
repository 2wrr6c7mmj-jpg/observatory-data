"""Fixed land mask per site from OpenStreetMap (versioned, committed to the repo).

Built once per version (``python -m sat.osm_mask ZJG``), not on every run:
natural=coastline ways are drawn as barriers, the free pixels are split into
4-connected regions, and each region is called land or water by the side of
the coastline it lies on (OSM convention: land on the left of the way).
Piers, breakwaters, groynes and road/rail bridges are added as structures.

Water polygons (``"mask_water": "osm_polygons"`` in sites.json): for a site on a river with no natural=coastline
(Mawei on the Min River), water is the inside of the OSM water areas (natural=water, waterway=riverbank; ways and
multipolygon relations, inner rings are land) and everything else is land; structures are added as above. Without the
setting the coastline method is used and the query is unchanged.

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
USER_AGENT = "observatory-data/1"
MASK_DIR = Path(__file__).resolve().parent / "masks"
WATER, LAND, STRUCTURE = 0, 1, 2


WATER_MODES = ("coastline", "osm_polygons")


def water_mode(site: dict) -> str:
    m = site.get("mask_water", "coastline")
    if m not in WATER_MODES:
        raise ValueError(f"{site['id']}: mask_water {m} は不明")
    return m


def overpass_query(grid: Grid, water_polygons: bool = False) -> str:
    s, w, n, e = grid.south, grid.west, grid.north, grid.east
    b = f"({s},{w},{n},{e})"
    water = (f'way["natural"="water"]{b};relation["natural"="water"]{b};'
             f'way["waterway"="riverbank"]{b};relation["waterway"="riverbank"]{b};') if water_polygons else ""
    return (f'[out:json][timeout:{180 if water_polygons else 90}];('
            f'way["natural"="coastline"]{b};'
            f'way["man_made"~"^(pier|breakwater|groyne|quay)$"]{b};'
            f'way["bridge"]["highway"]{b};'
            f'way["bridge"]["railway"]{b};'
            f'{water});out geom;')


def fetch_osm(grid: Grid, url: str = OVERPASS_URL, water_polygons: bool = False) -> dict:
    data = urllib.parse.urlencode({"data": overpass_query(grid, water_polygons)}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
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


def is_water_area(tags: dict) -> bool:
    return tags.get("natural") == "water" or tags.get("waterway") == "riverbank"


def _rings(ways: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Closed rings from the member ways of a multipolygon (joined end to end, reversed where needed). A chain that
    cannot be closed (a member outside the data) is closed by its end points, as OSM tools do."""
    todo = [list(w) for w in ways if len(w) >= 2]
    rings = []
    while todo:
        ring = todo.pop(0)
        while ring[0] != ring[-1]:
            for k, w in enumerate(todo):
                if w[0] == ring[-1]:
                    ring += w[1:]
                elif w[-1] == ring[-1]:
                    ring += w[::-1][1:]
                elif w[-1] == ring[0]:
                    ring = w[:-1] + ring
                elif w[0] == ring[0]:
                    ring = w[::-1][:-1] + ring
                else:
                    continue
                todo.pop(k)
                break
            else:
                ring.append(ring[0])
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def water_areas(osm: dict) -> list[tuple[list, list]]:
    """[(outer rings, inner rings)] of the OSM water areas (lon, lat) in ``osm`` (Overpass ``out geom``)."""
    out = []
    for el in osm.get("elements", []):
        if not is_water_area(el.get("tags", {})):
            continue
        if el.get("type") == "way" and el.get("geometry"):
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(ring) >= 4 and ring[0] == ring[-1]:
                out.append(([ring], []))
        elif el.get("type") == "relation":
            members = [m for m in el.get("members", []) if m.get("type") == "way" and m.get("geometry")]
            parts = {role: _rings([[(p["lon"], p["lat"]) for p in m["geometry"]] for m in members
                                   if (m.get("role") or "outer") == role]) for role in ("outer", "inner")}
            if parts["outer"]:
                out.append((parts["outer"], parts["inner"]))
    return out


def water_polygon_mask(grid: Grid, osm: dict) -> tuple[np.ndarray, dict]:
    """True = inside an OSM water area (outer rings filled, inner rings taken out)."""
    h, w = grid.height, grid.width
    water = np.zeros((h, w), bool)
    n_outer = n_inner = 0
    areas = water_areas(osm)
    for outers, inners in areas:
        area = np.zeros((h, w), bool)
        for ring in outers:
            r, c = grid.to_pixel([p[0] for p in ring], [p[1] for p in ring])
            fill_polygon(area, r, c)
            n_outer += 1
        for ring in inners:
            hole = np.zeros((h, w), bool)
            r, c = grid.to_pixel([p[0] for p in ring], [p[1] for p in ring])
            fill_polygon(hole, r, c)
            area &= ~hole
            n_inner += 1
        water |= area
    return water, {"water_areas": len(areas), "water_outer_rings": n_outer, "water_inner_rings": n_inner}


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


def build_mask(grid: Grid, osm: dict, land_fix: list[dict] | None = None,
               water_polygons: bool = False) -> tuple[np.ndarray, dict]:
    h, w = grid.height, grid.width
    coast = np.zeros((h, w), bool)
    coast_ways = []
    struct = np.zeros((h, w), bool)
    n_struct = 0
    for el in osm.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {})
        if is_water_area(tags):
            continue   # water areas: only with water_polygons (below)
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
    winfo = None
    if water_polygons:   # a river site: the OSM water areas decide land and water (the coastline is not used)
        water, winfo = water_polygon_mask(grid, osm)
        mask = np.where(water, WATER, LAND).astype(np.uint8)
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
    if winfo is not None:
        info.update(water_mode="osm_polygons", **winfo)
    if fixed is not None:
        info["land_fix"] = fixed
    return mask, info


def mask_paths(site_id: str, version: int, mask_dir: Path = MASK_DIR) -> tuple[Path, Path]:
    stem = f"{site_id}-land-v{version}"
    return mask_dir / f"{stem}.png", mask_dir / f"{stem}.json"


def save_mask(site: dict, grid: Grid, mask: np.ndarray, info: dict, osm: dict, version: int,
              mask_dir: Path = MASK_DIR, water_polygons: bool = False) -> dict:
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
        "query": overpass_query(grid, water_polygons),
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
    ap.add_argument("--osm-json", type=Path, help="build from this saved Overpass answer instead of asking Overpass")
    ap.add_argument("--save-osm", type=Path, help="also write the Overpass answer here (probe)")
    args = ap.parse_args(argv)
    site = next(s for s in load_sites() if s["id"] == args.site)
    grid = Grid.for_site(site)
    from .run import mask_version
    version = args.version or mask_version(site)
    if args.from_version:
        print(json.dumps(fix_from_version(site, grid, args.from_version, version, args.mask_dir), ensure_ascii=False, indent=1))
        return 0
    wp = water_mode(site) == "osm_polygons"
    osm = json.loads(args.osm_json.read_text(encoding="utf-8")) if args.osm_json else fetch_osm(grid, water_polygons=wp)
    if args.save_osm:
        args.save_osm.parent.mkdir(parents=True, exist_ok=True)
        args.save_osm.write_text(json.dumps(osm, ensure_ascii=False), encoding="utf-8")
    mask, info = build_mask(grid, osm, load_land_fix(site["id"], version, args.mask_dir), water_polygons=wp)
    meta = save_mask(site, grid, mask, info, osm, version, args.mask_dir, water_polygons=wp)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
