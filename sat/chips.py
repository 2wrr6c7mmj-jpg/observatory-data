"""Chip index for the public site (chips-1): the latest colour images of each port with the
candidate boxes as data (the site draws the red frames; nothing is burned into the image).

Written under docs/data/sat/chips/:

    index.json                 every site's latest scenes (newest first), where each image and chip file is
    <SITE>/<stamp>.json        one scene: image, bbox, pixel size, image size and the candidates' boxes

Derived only from files already published (ship_counts.csv colour columns and the detections GeoJSON): no
fetch, no extra Sentinel Hub request. Only the latest KEEP scenes of a site are kept: the colour image,
its thumbnail and the chip file of an older scene are deleted (they can be made again; the VV PNG of
Zhanjiang and every table are untouched). Pure Python (no numpy / Pillow): the WebP size is read from
its header.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Callable

from .grid import Grid, m_per_deg

CHIPS_VERSION = "chips-1"
KEEP = 10                   # scenes kept per site
MIN_BOX_M = 30.0            # a box is at least this wide (a few pixels), as on the trial page
BOX_PAD_PX = 2.0
MAX_IMAGE_BYTES = 300_000   # target for one colour image
CREDIT = "Contains modified Copernicus Sentinel data {year}"
WINDOW_LABELS = {"anchorage": "港内の錨泊（船が着く線から離れた水面。岸壁の候補とは別）"}
CAPTION = "擬似カラー（レーダー VV・VH）。赤枠は自動検出の候補で、艦艇とは限らない。試行"
OCEAN_FRAGMENT = "chips/ocean/index.json"   # written by ocean.chips (ocean.monitor); its "ocean" list is copied here
INDEX_NOTE = ("各地点の最新 keep 撮影（新しい順）。それより古い疑似カラー画像・切り出しファイルは消す（作り直せる派生物）。"
              "外洋（ocean）は域ごとの最新撮影（ocean.chips、chips/ocean/index.json から写す。保留の域は出さない）")
_SAFE = re.compile(r"^(color|chips)/[A-Za-z0-9_\-./]+$")


def credit(year: str) -> str:
    return CREDIT.format(year=year)


def webp_size(data: bytes) -> tuple[int, int]:
    """(width, height) of a WebP from its header (lossy VP8, lossless VP8L or extended VP8X)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP")
    kind = data[12:16]
    if kind == b"VP8 ":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        b = struct.unpack("<I", data[21:25])[0]
        return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    raise ValueError("unknown WebP chunk")


def candidate_box(grid: Grid, lon: float, lat: float, length_m: float, width_m: float) -> dict:
    """Square frame around a candidate, in chip pixels [x0, y0, x1, y1] and in lon/lat [w, s, e, n],
    clipped to the chip. Side = the longer of length / width, at least MIN_BOX_M (as the trial page)."""
    row, col = grid.to_pixel(lon, lat)
    mr, mc = grid.pixel_m
    half_x = max(length_m or 0, width_m or 0, MIN_BOX_M) / mc / 2 + BOX_PAD_PX
    half_y = max(length_m or 0, width_m or 0, MIN_BOX_M) / mr / 2 + BOX_PAD_PX
    x0, x1 = max(0.0, float(col) - half_x), min(float(grid.width), float(col) + half_x)
    y0, y1 = max(0.0, float(row) - half_y), min(float(grid.height), float(row) + half_y)
    w0, n0 = grid.to_lonlat(y0, x0)
    e0, s0 = grid.to_lonlat(y1, x1)
    return {"box_px": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "box_lonlat": [round(float(w0), 6), round(float(s0), 6), round(float(e0), 6), round(float(n0), 6)]}


def scene_doc(site: dict, grid: Grid, row: dict, features: list[dict], image_px: tuple[int, int],
              thumb_px: tuple[int, int] | None, tag: str, excluded: dict | None = None) -> dict:
    """The chip file of one scene. Candidate order = the GeoJSON's n. ``excluded``: counts of the
    candidates left out (berth-1, by reason), given only when the berth rule was applied."""
    cands = []
    for f in sorted(features, key=lambda f: f["properties"]["n"]):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        c = {"n": p["n"], "center": [lon, lat], **candidate_box(grid, lon, lat, p.get("length_m"), p.get("width_m")),
             "zone": p.get("zone"), "sector": p.get("sector"),
             "length_m": p.get("length_m"), "width_m": p.get("width_m"), "length_class": p.get("length_class"),
             "contrast_vv_db": p.get("contrast_vv_db"), "vh_ratio_db": p.get("vh_ratio_db"),
             "credibility": p.get("credibility"), "reliability": p.get("reliability"), "status": p.get("status")}
        if p.get("fixed_version"):   # fix-1: how ship-like against the empty-berth baseline
            c.update({k: p.get(k) for k in ("ship_conf", "rise_db", "rise_z", "base_db")})
        if p.get("fixed_flag"):      # fix-2: guarded (framed as usual, berthed long) / point, field (framed in grey)
            c["fixed_flag"] = p["fixed_flag"]
            c["frame"] = "grey" if p["fixed_flag"] in ("point", "field") else "red"
            if p.get("berthed_long"):
                c["berthed_long"] = True
        if p.get("window"):   # win-1: an anchorage candidate (open harbour water beyond the berth band), framed apart
            c["window"] = p["window"]
            c["window_label"] = WINDOW_LABELS.get(p["window"], p["window"])
        cands.append(c)
    mr, mc = grid.pixel_m
    return {
        "version": CHIPS_VERSION, "site_id": site["id"], "site_name": site.get("name"), "mask_version": tag,
        "acquired_utc": row["acquired_utc"], "platform": row.get("platform"),
        "orbit_direction": row.get("orbit_direction"), "relative_orbit": row.get("relative_orbit"),
        "image": row["color_image"], "thumb": row.get("color_thumb") or None,
        "image_px": list(image_px), "thumb_px": list(thumb_px) if thumb_px else None,
        "chip_px": [grid.width, grid.height],
        "bbox": [grid.west, grid.south, grid.east, grid.north],
        "pixel_m": {"row": round(mr, 2), "col": round(mc, 2)},
        "note": "box_px は画像の左上を原点とする画素（凡例は画像の下で、chip_px の外）。枠は候補の中心に長さの目安で描いた正方形で、船体の輪郭ではない",
        "candidates": cands,
        **({"excluded_counts": excluded} if excluded is not None else {}),
        "caption": CAPTION,
        "credit": credit(row["acquired_utc"][:4]),
    }


def _naval_scene(site: dict, acquired_utc: str) -> bool:
    """naval-1 applies to the scene: from berths.NAVAL_SINCE, for a site with a berth class file."""
    from . import berths   # noqa: PLC0415 - keep this module light for the site-side tests
    return berths.naval_active(site["id"], acquired_utc)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _remove(out: Path, rel: str) -> None:
    if rel and _SAFE.match(rel) and ".." not in rel:
        (out / rel).unlink(missing_ok=True)


def ocean_entries(out: Path) -> list[dict]:
    """The ``ocean`` list of index.json: copied from chips/ocean/index.json (ocean.chips); [] without one."""
    try:
        frag = json.loads((out / OCEAN_FRAGMENT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    ocean = frag.get("ocean") if isinstance(frag, dict) else None
    return ocean if isinstance(ocean, list) else []


def chip_rel(site_id: str, stamp: str) -> str:
    return f"chips/{site_id}/{stamp}.json"


def update(out: Path, sites: list[dict], rows: list[dict], features_for: Callable[[dict, str], list[dict]],
           mask_tag: Callable[[dict], str], row_tag: Callable[[dict], str], keep: int = KEEP,
           classify: Callable[[dict, list], list | None] | None = None) -> dict:
    """Rewrite chips/ from the published rows; delete the colour files of scenes beyond the latest ``keep`` of a site
    (and blank their colour columns in ``rows``, which the caller saves). Returns {"scenes", "removed", "cleared"}.

    ``classify(site, [(lon, lat), ...], acquired_utc)`` (berth-1, sat/berths.py): for the quay-band candidates, "" (kept),
    "coastline" or "off_berth"; the others are left out of ``candidates`` and only counted in
    ``excluded_counts``. Open-water candidates are always kept. None (or no quay lines): every candidate."""
    index_sites, removed, cleared, n_scenes = [], 0, 0, 0
    for site in sites:
        tag = mask_tag(site)
        mine = [r for r in rows if r["site_id"] == site["id"] and r.get("color_image") and row_tag(r) == tag]
        mine.sort(key=lambda r: r["acquired_utc"], reverse=True)
        if not mine:
            continue
        grid = Grid.for_site(site)
        scenes = []
        by_month: dict[str, list[dict]] = {}
        for r in mine[keep:]:       # older than the latest `keep`: delete the derived files
            for k in ("color_image", "color_thumb"):
                _remove(out, r.get(k) or "")
                r[k] = ""
            stamp = r["acquired_utc"].replace(":", "-")
            _remove(out, chip_rel(site["id"], stamp))
            removed += 1
            cleared += 1
        for r in mine[:keep]:
            img = out / r["color_image"]
            if not img.exists():
                continue
            thumb = out / r["color_thumb"] if r.get("color_thumb") else None
            try:
                image_px = webp_size(img.read_bytes())
                thumb_px = webp_size(thumb.read_bytes()) if thumb is not None and thumb.exists() else None
            except (ValueError, struct.error):
                continue
            month = r["acquired_utc"][:7]
            if month not in by_month:
                by_month[month] = features_for(site, month)
            feats = [f for f in by_month[month] if f["properties"]["acquired_utc"] == r["acquired_utc"]
                     and f["properties"]["site_id"] == site["id"] and row_tag(f["properties"]) == tag]
            excluded = None
            # win-1: candidates the window rule left out at detection time (property window_excluded) are not framed
            win = [f for f in feats if f["properties"].get("window_excluded")]
            if win:
                excluded = {}
                for f in win:
                    k = "window_" + f["properties"]["window_excluded"]
                    excluded[k] = excluded.get(k, 0) + 1
                feats = [f for f in feats if not f["properties"].get("window_excluded")]
            # fix-1: candidates the empty-berth baseline shows as a structure (property fixed_excluded) are not framed
            fx = [f for f in feats if f["properties"].get("fixed_excluded")]
            if fx:
                excluded = dict(excluded or {})
                for f in fx:
                    k = "fixed_" + f["properties"]["fixed_excluded"]
                    excluded[k] = excluded.get(k, 0) + 1
                feats = [f for f in feats if not f["properties"].get("fixed_excluded")]
            # naval-1: candidates on a civil or unknown berth line (property naval_excluded) are not framed; counted.
            # From berths.NAVAL_SINCE the open-water anchorage (win-1, no berth line) is not framed either: only naval
            # (user's instruction 2026-11-05: the counts and the squares show the navy only)
            if _naval_scene(site, r["acquired_utc"]):
                an = [f for f in feats if f["properties"].get("window")]
                if an:
                    excluded = dict(excluded or {})
                    excluded["naval_anchorage"] = excluded.get("naval_anchorage", 0) + len(an)
                    feats = [f for f in feats if not f["properties"].get("window")]
            nav = [f for f in feats if f["properties"].get("naval_excluded")]
            if nav:
                excluded = dict(excluded or {})
                for f in nav:
                    k = "naval_" + f["properties"]["naval_excluded"]
                    excluded[k] = excluded.get(k, 0) + 1
                feats = [f for f in feats if not f["properties"].get("naval_excluded")]
            if classify is not None:
                band = [f for f in feats if f["properties"].get("zone") == "quay"]
                why = classify(site, [tuple(f["geometry"]["coordinates"][:2]) for f in band], r["acquired_utc"])
                if why is not None:
                    excluded = {**{k: 0 for k in ("off_berth", "coastline")}, **(excluded or {})}
                    drop = set()
                    for f, w in zip(band, why):
                        if w:
                            excluded[w] = excluded.get(w, 0) + 1
                            drop.add(id(f))
                    feats = [f for f in feats if id(f) not in drop]
            doc = scene_doc(site, grid, r, feats, image_px, thumb_px if thumb is not None and thumb.exists() else None, tag,
                            excluded)
            stamp = r["acquired_utc"].replace(":", "-")
            _write_if_changed(out / chip_rel(site["id"], stamp), _dump(doc))
            scenes.append({"acquired_utc": r["acquired_utc"], "platform": r.get("platform"),
                           "orbit_direction": r.get("orbit_direction"), "relative_orbit": r.get("relative_orbit"),
                           "chip": chip_rel(site["id"], stamp), "image": r["color_image"],
                           "thumb": r.get("color_thumb") or None, "candidates": len(doc["candidates"]),
                           **({"excluded": sum(doc["excluded_counts"].values())} if "excluded_counts" in doc else {}),
                           "retrieved_utc": r.get("retrieved_utc")})
        # chip files that no row points to any more (a scene deleted by hand)
        keepers = {Path(s["chip"]).name for s in scenes}
        d = out / "chips" / site["id"]
        if d.is_dir():
            for p in d.glob("*.json"):
                if p.name not in keepers:
                    p.unlink()
        if scenes:
            n_scenes += len(scenes)
            index_sites.append({"site_id": site["id"], "name": site.get("name"), "mask_version": tag,
                                "bbox": [grid.west, grid.south, grid.east, grid.north],
                                "center": [round((grid.west + grid.east) / 2, 6), round((grid.south + grid.north) / 2, 6)],
                                "scenes": scenes})
    if index_sites:
        stamps = [s["retrieved_utc"] for x in index_sites for s in x["scenes"] if s.get("retrieved_utc")]
        for x in index_sites:
            for s in x["scenes"]:
                s.pop("retrieved_utc", None)
        ocean = ocean_entries(out)
        idx = {"version": CHIPS_VERSION, "keep": keep, "updated_utc": max(stamps) if stamps else None,
               "caption": CAPTION,
               "credit": credit(max(s["acquired_utc"][:4] for x in index_sites + ocean for s in x["scenes"])),
               "note": INDEX_NOTE, "sites": index_sites, "ocean": ocean}
        _write_if_changed(out / "chips" / "index.json", _dump(idx))
    return {"scenes": n_scenes, "removed": removed, "cleared": cleared}


def splice_sites(out: Path, ports_index: Path) -> bool:
    """Writes the ports' part of ``ports_index`` (chips/index.json as this run wrote it) into chips/index.json, with
    ``ocean`` from the fragment on disk (the latest main's, written by ocean.chips). Used by sat/publish.sh after
    ``git pull --rebase`` onto a main where ocean-monitor may have rewritten the shared index. True if changed;
    False (nothing written) when ``ports_index`` cannot be read."""
    try:
        ports = json.loads(ports_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(ports, dict) or not isinstance(ports.get("sites"), list):
        return False
    ocean = ocean_entries(out)
    years = [s["acquired_utc"][:4] for x in ports["sites"] + ocean for s in x.get("scenes") or [] if s.get("acquired_utc")]
    new = {**ports, "ocean": ocean, "credit": credit(max(years)) if years else ports.get("credit"), "note": INDEX_NOTE}
    p = out / "chips" / "index.json"
    text = _dump(new)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return False
    _write_if_changed(p, text)
    return True


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="chips-1 index; no network")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splice", help="copy the ports' sites of --sites into chips/index.json (ocean from the fragment)")
    s.add_argument("--sat", type=Path, default=Path("docs/data/sat"))
    s.add_argument("--sites", type=Path, required=True)
    a = ap.parse_args(argv)
    print("chips/index.json: " + ("更新" if splice_sites(a.sat, a.sites) else "変更なし"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
