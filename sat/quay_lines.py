"""Quay lines of a site from OpenStreetMap (versioned, committed like the land masks).

man_made=pier / quay and natural=coastline ways in the site's bbox, kept as lon/lat
polylines. The quay count (sat/quay.py) samples the water side of these lines; which side
is water is decided from the land mask, so the drawing direction of a way does not matter
(a pier has water on both sides, a coastline on one).

    python -m sat.quay_lines ZJG --out DIR     # probe: the work shell cannot reach Overpass

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

from .grid import Grid
from .osm_mask import MASK_DIR, OVERPASS_URL, USER_AGENT

LINES_VERSION = 1
KINDS = ("pier", "quay", "coastline")


def query(grid: Grid) -> str:
    b = f"({grid.south},{grid.west},{grid.north},{grid.east})"
    return (f'[out:json][timeout:90];('
            f'way["man_made"~"^(pier|quay)$"]{b};'
            f'way["natural"="coastline"]{b};'
            f');out geom;')


def fetch(grid: Grid, url: str = OVERPASS_URL) -> dict:
    data = urllib.parse.urlencode({"data": query(grid)}).encode()
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


def kind_of(tags: dict) -> str | None:
    if tags.get("natural") == "coastline":
        return "coastline"
    if tags.get("man_made") in ("pier", "quay"):
        return tags["man_made"]
    return None


def lines_from_osm(osm: dict) -> list[dict]:
    out = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        k = kind_of(el.get("tags", {}))
        if k is None:
            continue
        coords = [[round(p["lon"], 6), round(p["lat"], 6)] for p in el["geometry"]]
        if len(coords) < 2:
            continue
        out.append({"id": el.get("id"), "kind": k, "closed": coords[0] == coords[-1], "coords": coords})
    out.sort(key=lambda w: (KINDS.index(w["kind"]), w["id"] or 0))
    return out


def path(site_id: str, version: int = LINES_VERSION, mask_dir: Path = MASK_DIR) -> Path:
    return mask_dir / f"{site_id}-quay-v{version}.json"


def save(site: dict, grid: Grid, osm: dict, out_dir: Path, version: int = LINES_VERSION) -> dict:
    ways = lines_from_osm(osm)
    doc = {
        "version": f"{site['id']}-quay-v{version}",
        "site": site["id"],
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": grid.bbox,
        "source": "OpenStreetMap via Overpass API",
        "osm_base": (osm.get("osm3s") or {}).get("timestamp_osm_base"),
        "query": query(grid),
        "license": "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "counts": {k: sum(w["kind"] == k for w in ways) for k in KINDS},
        "ways": ways,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = path(site["id"], version, out_dir)
    p.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return doc


def load(site_id: str, version: int = LINES_VERSION, mask_dir: Path = MASK_DIR) -> dict | None:
    p = path(site_id, version, mask_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None) -> int:
    from .run import load_sites
    ap = argparse.ArgumentParser(description="Fetch a site's quay lines (pier, quay, coastline) from OSM")
    ap.add_argument("site")
    ap.add_argument("--out", type=Path, default=MASK_DIR)
    ap.add_argument("--version", type=int, default=LINES_VERSION)
    args = ap.parse_args(argv)
    site = next(s for s in load_sites() if s["id"] == args.site)
    grid = Grid.for_site(site)
    doc = save(site, grid, fetch(grid), args.out, args.version)
    print(json.dumps({k: v for k, v in doc.items() if k != "ways"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
