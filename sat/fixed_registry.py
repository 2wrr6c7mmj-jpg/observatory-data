"""The registry of confirmed fixed objects in the ports (fix-2, 2部の設計書 fixed_eval 4.1): sat/fixed_registry/<site>.json.

A person checked each entry on optical imagery (Esri World Imagery, Sentinel-2) and found a structure, not a ship: a pier
part, a pontoon, a buoy, a beacon, a bridge pier, fishing gear, a raft, or a hole in the land mask. From fixed.FIX2_SINCE
a candidate whose centre is within ``radius_m`` of an active entry is not counted (``fixed_excluded`` = "registry",
``fixed_registry_id``), except that fix-2's guard (a naval berth candidate of 60 m and more, its pieces, a shipyard) wins
over every kind but ``pier_part`` and ``mask_error`` (sat/fixed.py).

File::

    {"site": "QGZ", "note": "...", "entries": [
        {"id": "QGZ-001", "lon": 119.96395, "lat": 35.74132, "radius_m": 30, "kind": "mask_error",
         "status": "confirmed",                     # confirmed (used) | proposed (listed only, never used)
         "confirmed_by": "2部",                      # 2部 | owner ("" for proposed)
         "evidence": {"base_db": 12.0, "s_scenes": "...", "imagery": [{"source": "Esri World Imagery", "viewed_utc": "2026-11-06", ...}],
                      "note": "..."},
         "added_utc": "2026-11-06", "expires_utc": null}]}      # gear and raft: expires 120 days after added

An entry is active for a scene when it is confirmed, has ``confirmed_by`` and has not expired at the scene's date.
Each scene records the hash of the site's active entries (``registry_hash``): changing the registry changes the counts.
Pure Python (the collect workflows run without numpy).

    python -m sat.fixed_registry geojson          # writes docs/data/sat/fixed/<site>.geojson (checked by the tests)
    python -m sat.fixed_registry review           # the month's list for 2部: guarded / berthed_long and repeated points
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY_DIR = HERE / "fixed_registry"
GEOJSON_DIR = HERE.parent / "docs" / "data" / "sat" / "fixed"
DETECTIONS_DIR = HERE.parent / "docs" / "data" / "sat" / "detections"
KINDS = ("pier_part", "pontoon", "buoy", "beacon", "bridge_pier", "gear", "raft", "mask_error")
EXPIRING = {"gear": 120, "raft": 120}        # days: nets and rafts move with the season
OVERRIDES_GUARD = ("pier_part", "mask_error")
STATUSES = ("confirmed", "proposed")
CONFIRMERS = ("2部", "owner")
DEFAULT_RADIUS_M = 30.0
MAX_RADIUS_M = 100.0                          # a larger circle could swallow a moored ship next to the structure


def load(site_id: str, d: Path | None = None) -> dict | None:
    p = (d or REGISTRY_DIR) / f"{site_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check(reg: dict, site_id: str | None = None) -> list[str]:
    """The problems of a registry file ([] = fine)."""
    errs = []
    if site_id and reg.get("site") != site_id:
        errs.append(f"site {reg.get('site')} != {site_id}")
    seen = set()
    for e in reg.get("entries") or []:
        i = e.get("id")
        if not i or i in seen:
            errs.append(f"id {i!r} missing or repeated")
        seen.add(i)
        if not (isinstance(e.get("lon"), (int, float)) and isinstance(e.get("lat"), (int, float))
                and -180 <= e["lon"] <= 180 and -90 <= e["lat"] <= 90):
            errs.append(f"{i}: lon/lat")
        r = e.get("radius_m", DEFAULT_RADIUS_M)
        if not isinstance(r, (int, float)) or not 0 < r <= MAX_RADIUS_M:
            errs.append(f"{i}: radius_m {r} not in (0, {MAX_RADIUS_M:g}]")
        if e.get("kind") not in KINDS:
            errs.append(f"{i}: kind {e.get('kind')}")
        if e.get("status", "confirmed") not in STATUSES:
            errs.append(f"{i}: status {e.get('status')}")
        if e.get("status", "confirmed") == "confirmed" and e.get("confirmed_by") not in CONFIRMERS:
            errs.append(f"{i}: confirmed without confirmed_by in {CONFIRMERS}")
        try:
            added = date.fromisoformat(str(e.get("added_utc"))[:10])
        except ValueError:
            errs.append(f"{i}: added_utc")
            continue
        exp = e.get("expires_utc")
        if e.get("kind") in EXPIRING:
            if not exp:
                errs.append(f"{i}: {e['kind']} needs expires_utc")
            elif date.fromisoformat(exp[:10]) > added + timedelta(days=EXPIRING[e["kind"]]):
                errs.append(f"{i}: expires_utc later than {EXPIRING[e['kind']]} days after added_utc")
        elif exp:
            try:
                date.fromisoformat(exp[:10])
            except ValueError:
                errs.append(f"{i}: expires_utc")
        ev = e.get("evidence") or {}
        if e.get("status", "confirmed") == "confirmed" and not ev.get("imagery"):
            errs.append(f"{i}: confirmed without evidence.imagery")
    return errs


def active(reg: dict | None, acquired_utc: str) -> list[dict]:
    """The entries used for a scene acquired at ``acquired_utc``: confirmed, confirmed_by set, not expired."""
    day = acquired_utc[:10]
    out = []
    for e in (reg or {}).get("entries") or []:
        if e.get("status", "confirmed") != "confirmed" or e.get("confirmed_by") not in CONFIRMERS:
            continue
        if e.get("expires_utc") and day >= e["expires_utc"][:10]:
            continue
        out.append(e)
    return out


def confirmed(reg: dict | None) -> list[dict]:
    """Every confirmed entry with confirmed_by, expired or not (meta.json: the registry as filed)."""
    return [e for e in (reg or {}).get("entries") or []
            if e.get("status", "confirmed") == "confirmed" and e.get("confirmed_by") in CONFIRMERS]


def entries_hash(entries: list[dict]) -> str:
    """sha256 (first 12) of the active entries that decide the counts (id, place, radius, kind); "" when none."""
    if not entries:
        return ""
    key = sorted((e["id"], round(float(e["lon"]), 6), round(float(e["lat"]), 6),
                  float(e.get("radius_m", DEFAULT_RADIUS_M)), e["kind"]) for e in entries)
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:12]


def dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    mx = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lon1 - lon2) * mx, (lat1 - lat2) * 111132.0)


def match(entries: list[dict], lon: float, lat: float) -> dict | None:
    """The nearest active entry whose circle holds (lon, lat), or None."""
    best, bd = None, math.inf
    for e in entries:
        x = dist_m(lon, lat, e["lon"], e["lat"])
        if x <= float(e.get("radius_m", DEFAULT_RADIUS_M)) and x < bd:
            best, bd = e, x
    return best


def geojson(reg: dict) -> dict:
    """docs/data/sat/fixed/<site>.geojson: every entry (confirmed and proposed) as a point, to be checked by eye."""
    feats = []
    for e in reg.get("entries") or []:
        props = {k: e.get(k) for k in ("id", "kind", "status", "confirmed_by", "radius_m", "added_utc", "expires_utc")}
        props["radius_m"] = e.get("radius_m", DEFAULT_RADIUS_M)
        props["note"] = (e.get("evidence") or {}).get("note")
        props["imagery"] = [{k: v for k, v in im.items() if k in ("source", "viewed_utc", "captured")}
                            for im in (e.get("evidence") or {}).get("imagery") or []]
        if e.get("osm"):
            props["osm"] = e["osm"]
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [e["lon"], e["lat"]]},
                      "properties": props})
    attribution = ["固定物の台帳（人が光学画像で確かめた点。sat/fixed_registry）"]
    if any(e.get("osm") for e in reg.get("entries") or []):
        attribution.append("© OpenStreetMap contributors, ODbL 1.0")
    return {"type": "FeatureCollection", "site_id": reg["site"], "registry_hash": entries_hash(confirmed(reg)),
        "attribution": attribution, "features": feats}


def sites_with_registry(d: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (d or REGISTRY_DIR).glob("*.json"))


def write_geojson(d: Path | None = None, out: Path | None = None) -> list[Path]:
    out = out or GEOJSON_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid in sites_with_registry(d):
        p = out / f"{sid}.geojson"
        p.write_text(json.dumps(geojson(load(sid, d)), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def review(features: list[dict], point_m: float = 30.0, min_scenes: int = 2) -> dict:
    """The list 2部 checks each month (design 4.4 / 6): the guarded and berthed_long candidates, and the points flagged
    ``point`` in ``min_scenes`` or more scenes within ``point_m`` of each other (proposed registry entries)."""
    guarded = [f for f in features if f["properties"].get("fixed_flag") == "guarded"]
    pts = [f for f in features if f["properties"].get("fixed_flag") == "point"]
    groups: list[list[dict]] = []
    for f in pts:
        lon, lat = f["geometry"]["coordinates"]
        for g in groups:
            g0 = g[0]["geometry"]["coordinates"]
            if g[0]["properties"].get("site_id") == f["properties"].get("site_id") and dist_m(lon, lat, *g0) <= point_m:
                g.append(f)
                break
        else:
            groups.append([f])
    proposed = []
    for g in groups:
        scenes = sorted({f["properties"].get("acquired_utc") for f in g})
        if len(scenes) >= min_scenes:
            lon = sum(f["geometry"]["coordinates"][0] for f in g) / len(g)
            lat = sum(f["geometry"]["coordinates"][1] for f in g) / len(g)
            proposed.append({"site_id": g[0]["properties"].get("site_id"), "lon": round(lon, 6), "lat": round(lat, 6),
                             "scenes": scenes, "status": "proposed"})
    return {"guarded": [{"site_id": f["properties"].get("site_id"), "acquired_utc": f["properties"].get("acquired_utc"),
                         "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1],
                         "length_m": f["properties"].get("length_m"), "fixed_guard": f["properties"].get("fixed_guard"),
                         "berthed_long": f["properties"].get("berthed_long")} for f in guarded],
            "proposed_points": proposed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fix-2: the registry of confirmed fixed objects")
    ap.add_argument("cmd", choices=("check", "geojson", "review"))
    ap.add_argument("--months", nargs="*", help="review: detections/**/<YYYY-MM>.geojson to read (default: all)")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        bad = {sid: check(load(sid), sid) for sid in sites_with_registry()}
        bad = {k: v for k, v in bad.items() if v}
        print(json.dumps(bad, ensure_ascii=False, indent=1) if bad else "ok")
        return 1 if bad else 0
    if a.cmd == "geojson":
        for p in write_geojson():
            print(p)
        return 0
    files = sorted(p for m in a.months for p in DETECTIONS_DIR.rglob(f"{m}.geojson")) if a.months \
        else sorted(DETECTIONS_DIR.rglob("*.geojson"))
    feats = [f for p in files if p.exists() for f in json.loads(p.read_text(encoding="utf-8"))["features"]]
    print(json.dumps(review(feats), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
