"""OSM data a new port needs before its first run (probe only: the work shell cannot reach Overpass; PU 0).

For each site of sat/sites.json given (default: every enabled site without a land mask of its mask_version):

- the land mask (sat/osm_mask.py). A site with no natural=coastline in its range (a river port) comes out all water or
  all land; then the mask is built again from the OSM water areas (``mask_water: osm_polygons``) and the result says
  so, and sites.json must be given that setting before the mask is used;
- the quay lines (sat/quay_lines.py);
- the military / port areas around it and every pier (sat/berth_osm.py, bbox + ``--margin`` degrees).

    python -m tools.ports_prep [SITE ...] [--sleep 25] [--margin 0.02]

Writes sat/masks/<site>-land-v<n>.png/.json, sat/masks/<site>-quay-v<n>.json, probe-raw/berth-osm/<site>-osm-areas.json
and probe-raw/ports-prep/summary.json. A failed request is tried once more at the end; what still fails is listed.

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "probe-raw"
# a coastline mask this one-sided has no usable coastline (Mawei v1: land 0.0)
ONE_SIDED = 0.02


def todo_sites(sites: list[dict], only: list[str], mask_dir: Path) -> list[dict]:
    from sat.osm_mask import mask_paths
    from sat.run import mask_version
    out = []
    for s in sites:
        if not s.get("enabled", True):
            continue
        if only:
            if s["id"] in only:
                out.append(s)
        elif not mask_paths(s["id"], mask_version(s), mask_dir)[0].exists():
            out.append(s)
    return out


def needs_water_polygons(info: dict) -> bool:
    return not info.get("coastline_ways") or min(info.get("land_fraction", 0.0), info.get("water_fraction", 0.0)) < ONE_SIDED


def land_mask(site: dict, mask_dir: Path, sleep: float, fetch=None) -> dict:
    from sat import osm_mask
    from sat.grid import Grid
    from sat.run import mask_version
    fetch = fetch or osm_mask.fetch_osm
    grid, version = Grid.for_site(site), mask_version(site)
    wp = osm_mask.water_mode(site) == "osm_polygons"
    osm = fetch(grid, water_polygons=wp)
    mask, info = osm_mask.build_mask(grid, osm, osm_mask.load_land_fix(site["id"], version, mask_dir), water_polygons=wp)
    switched = False
    if not wp and needs_water_polygons(info):
        time.sleep(sleep)
        first = {k: info.get(k) for k in ("coastline_ways", "land_fraction", "water_fraction")}
        wp, switched = True, True
        osm = fetch(grid, water_polygons=True)
        mask, info = osm_mask.build_mask(grid, osm, osm_mask.load_land_fix(site["id"], version, mask_dir), water_polygons=True)
        info = {**info, "coastline_attempt": first}
    meta = osm_mask.save_mask(site, grid, mask, info, osm, version, mask_dir, water_polygons=wp)
    return {"version": meta["version"], "water_polygons": wp, "switched_to_water_polygons": switched,
            **{k: meta.get(k) for k in ("coastline_ways", "structure_ways", "water_areas", "water_fraction", "land_fraction",
                                        "structure_fraction", "osm_base")}}


def quay_lines(site: dict, mask_dir: Path) -> dict:
    from sat import quay_lines as ql
    from sat.grid import Grid
    grid = Grid.for_site(site)
    doc = ql.save(site, grid, ql.fetch(grid), mask_dir, int(site.get("quay_lines_version", ql.LINES_VERSION)))
    return {"version": doc["version"], "counts": doc["counts"]}


def areas(site: dict, out: Path, margin: float) -> dict:
    from sat import berth_osm as bo
    b = [round(site["bbox"][0] - margin, 4), round(site["bbox"][1] - margin, 4),
         round(site["bbox"][2] + margin, 4), round(site["bbox"][3] + margin, 4)]
    osm = bo.fetch(b, all_piers=True)
    doc = {"site": site["id"], "bbox": b, "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "osm_base": (osm.get("osm3s") or {}).get("timestamp_osm_base"), "query": bo.query(b, True), "all_piers": True,
           "license": "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
           "features": bo.simplify(osm)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{site['id']}-osm-areas.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    mil = [f for f in doc["features"] if (f.get("tags") or {}).get("landuse") == "military" or (f.get("tags") or {}).get("military")]
    return {"features": len(doc["features"]), "military": len(mil),
            "military_names": [(f.get("tags") or {}).get("name", "") for f in mil][:12]}


def main(argv=None) -> int:
    from sat.osm_mask import MASK_DIR
    from sat.run import load_sites
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--sleep", type=float, default=25.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    ap.add_argument("--raw", type=Path, default=RAW)
    args = ap.parse_args(argv)
    sites = todo_sites(load_sites(), args.sites, args.mask_dir)
    steps = {"mask": lambda s: land_mask(s, args.mask_dir, args.sleep), "lines": lambda s: quay_lines(s, args.mask_dir),
             "areas": lambda s: areas(s, args.raw / "berth-osm", args.margin)}
    res: dict = {s["id"]: {} for s in sites}
    failed: list[tuple[dict, str]] = []
    first = True
    for rnd in (1, 2):
        work = [(s, k) for s in sites for k in steps] if rnd == 1 else failed
        failed = []
        for s, k in work:
            if not first:
                time.sleep(args.sleep)
            first = False
            try:
                res[s["id"]][k] = steps[k](s)
                print(f"{s['id']} {k}: {json.dumps(res[s['id']][k], ensure_ascii=False)}", flush=True)
            except Exception as e:  # one port's failure must not stop the others
                res[s["id"]][k] = {"error": str(e)[:200]}
                failed.append((s, k))
                print(f"FAILED {s['id']} {k} (round {rnd}): {str(e)[:200]}", flush=True)
        if not failed:
            break
    out = args.raw / "ports-prep"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                  "sites": res, "failed": [f"{s['id']}:{k}" for s, k in failed]},
                                                 ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
