"""Colour images of the open-sea areas for the site (ocean-chips-1): the latest passes of each published area,
in the same form as the port chips (``sat/chips.py``) so that the site reads them unchanged.

    python -m ocean.chips splice  --sat docs/data/sat                 # chips/ocean/index.json -> chips/index.json "ocean"
    python -m ocean.chips refresh --sat docs/data/sat [--config ...]  # prune (keep, held areas) and rewrite the fragment

Made by ``ocean.monitor`` from the arrays it has already fetched for detection (VV, VH at 40 m): no extra
request, no extra PU. One image per pass and area, written under ``docs/data/sat/chips/ocean/<AREA>/``:

    <stamp>.webp   the area at a reduced scale (top left, ``chip_px``; the site draws the red frames of the A and
                   B candidates from the chip file), below it the zoom tiles (40 m, x2) of the strongest A
                   candidates with their frame and number burned in, then the legend and the credit
    <stamp>.json   the chip file (``sat/chips.py`` scene_doc form; ``site_id`` is the area id)

``chips/ocean/index.json`` lists the latest ``KEEP`` passes of each published area; older images are deleted (they
cannot be made again without a new fetch, so only the latest few are kept, as the ports' keep). Areas with
``publish: hold`` (or disabled / removed) get no image: their directory is deleted. ``chips/index.json`` (the file the
site reads, shared with the port chips of ``sat.run``) takes its ``ocean`` list from the fragment: ``splice`` here and
``sat.chips.update`` both copy it, so neither writer drops the other's part.

The arrays are not kept after the run: a pass fetched before this module existed has no image and none can be made
from the published data (the candidates hold no pixels).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from sat import chips as satchips
from sat import color
from sat.grid import Grid

VERSION = "ocean-chips-2"
SUBDIR = "chips/ocean"
FRAGMENT = "chips/ocean/index.json"
INDEX = "chips/index.json"
KEEP = 3                    # passes kept per area
OVERVIEW_MAX = 600          # the area's longer side in the overview, px (fits a phone at 2x zoom)
TILE_NATIVE = 64            # zoom tile: 64 px at 40 m = 2.56 km square
TILE_SCALE = 2              # shown at x2 (nearest)
TILE_GAP = 4
MAX_TILES = 0               # ocean-chips-2: no zoom tiles in the overview (every candidate has its own in the sheet)
SHEET_TILE = 64             # one candidate in the sheet: 1.28 km square (40 m x2 or 20 m x1)
SHEET_COLS = 16
SHEET_MAX = 256             # candidates with a tile per pass (A first)
MAX_SHEET_BYTES = 350_000
MIN_BOX_PX = 8.0            # a frame on the overview is at least this wide (a ship is below one overview pixel)
MAX_IMAGE_BYTES = 250_000
QUALITY_STEPS = (70, 60, 50, 40, 30)
RED = (255, 40, 40)
CAPTION = satchips.CAPTION
NOTE = ("外洋の域の縮小画像（chip_px の範囲。赤枠は A・B 候補の位置を示す最小 8 画素の正方形で、船体の大きさではない）。"
        "候補ごとの切り出しは sheet（SHEET_NOTE）。取得しなかった所は黒")
SHEET_NOTE = ("候補ごとの切り出し（1.28 km 四方、候補が中央）。tile.res_m＝20 は 20 m の取り直し（過去の撮影と比べた確認つき）、"
              "40 は検出に使った 40 m の画像を 2 倍表示。赤枠（tile.frame、タイル内の画素）は推定の長さ")
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")
SHEET_SUFFIX = "-c"         # <stamp>-c.webp: the sheet of candidate tiles


def stamp_of(acquired_utc: str) -> str:
    return acquired_utc.replace(":", "-")


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- overview of one area

class Canvas:
    """The area at a reduced scale: mean linear sigma0 of the 40 m pixels falling in each overview pixel.
    k overview px per 0.1 deg cell (k = OVERVIEW_MAX // the longer side in cells, 4..60)."""

    def __init__(self, ac):
        self.k = int(min(60, max(4, OVERVIEW_MAX // max(ac.nx, ac.ny))))
        self.grid = Grid(ac.w, ac.s, ac.w + ac.nx * ac.cell, ac.s + ac.ny * ac.cell, ac.nx * self.k, ac.ny * self.k)
        n = self.grid.width * self.grid.height
        self.vv, self.vh, self.n, self.nh = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    def add(self, g: Grid, vv: np.ndarray, vh: np.ndarray, valid: np.ndarray) -> None:
        G = self.grid
        lon = g.west + (np.arange(g.width) + 0.5) / g.width * (g.east - g.west)
        lat = g.north - (np.arange(g.height) + 0.5) / g.height * (g.north - g.south)
        cols = np.floor((lon - G.west) / (G.east - G.west) * G.width).astype(np.int64)
        rows = np.floor((G.north - lat) / (G.north - G.south) * G.height).astype(np.int64)
        okc, okr = (cols >= 0) & (cols < G.width), (rows >= 0) & (rows < G.height)
        m = np.asarray(valid, bool) & okr[:, None] & okc[None, :] & np.isfinite(vv)
        if not m.any():
            return
        idx = (np.clip(rows, 0, G.height - 1)[:, None] * G.width + np.clip(cols, 0, G.width - 1)[None, :])[m]
        size = G.width * G.height
        self.vv += np.bincount(idx, weights=10.0 ** (vv[m] / 10.0), minlength=size)
        self.n += np.bincount(idx, minlength=size)
        hm = np.isfinite(vh[m])
        self.vh += np.bincount(idx[hm], weights=10.0 ** (vh[m][hm] / 10.0), minlength=size)
        self.nh += np.bincount(idx[hm], minlength=size)

    @property
    def has_data(self) -> bool:
        return bool(self.n.any())

    def covered(self) -> np.ndarray:
        return (self.n > 0).reshape(self.grid.height, self.grid.width)

    def rgb(self) -> np.ndarray:
        H, W = self.grid.height, self.grid.width
        with np.errstate(divide="ignore", invalid="ignore"):
            vv = np.where(self.n > 0, 10.0 * np.log10(self.vv / np.maximum(self.n, 1)), np.nan)
            # VH NaN with data is sigma0 <= 0 (below the noise floor): shown as the bottom of the range
            vh = np.where(self.nh > 0, 10.0 * np.log10(self.vh / np.maximum(self.nh, 1)),
                          np.where(self.n > 0, color.VH_RANGE[0], np.nan))
        return color.false_color(vv.reshape(H, W), vh.reshape(H, W), filtered=False)


def crop(vv: np.ndarray, vh: np.ndarray, g: Grid, lon: float, lat: float, size: int = TILE_NATIVE) -> np.ndarray:
    """uint8 RGB (size, size) of the 40 m pixels around a candidate (Lee filtered as the port images; outside = no data)."""
    r, c = g.to_pixel(lon, lat)
    r0, c0 = int(math.floor(float(r))) - size // 2, int(math.floor(float(c))) - size // 2
    a = np.full((size, size), np.nan)
    b = np.full((size, size), np.nan)
    ys, xs = max(0, r0), max(0, c0)
    ye, xe = min(vv.shape[0], r0 + size), min(vv.shape[1], c0 + size)
    if ye > ys and xe > xs:
        a[ys - r0:ye - r0, xs - c0:xe - c0] = vv[ys:ye, xs:xe]
        bb = vh[ys:ye, xs:xe]
        # VH NaN where VV has data: below the noise floor (bottom of the range), not "no data"
        b[ys - r0:ye - r0, xs - c0:xe - c0] = np.where(np.isfinite(bb), bb, np.where(np.isfinite(vv[ys:ye, xs:xe]),
                                                                                      color.VH_RANGE[0], np.nan))
    return color.false_color(a, b, filtered=True)


# ---------------------------------------------------------------- one pass: image and chip file

def _order(feats: list[dict]) -> list[dict]:
    """A first, then B; inside a class the stronger VV contrast first (n = position + 1)."""
    def key(f):
        p = f["properties"]
        return (0 if p.get("class") == "A" else 1, -(p.get("contrast_vv_db") or 0.0), f["geometry"]["coordinates"][0],
                f["geometry"]["coordinates"][1])
    return sorted(feats, key=key)


def overview_box(grid: Grid, lon: float, lat: float) -> dict:
    row, col = grid.to_pixel(lon, lat)
    h = MIN_BOX_PX / 2
    x0, x1 = max(0.0, float(col) - h), min(float(grid.width), float(col) + h)
    y0, y1 = max(0.0, float(row) - h), min(float(grid.height), float(row) + h)
    w0, n0 = grid.to_lonlat(y0, x0)
    e0, s0 = grid.to_lonlat(y1, x1)
    return {"box_px": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "box_lonlat": [round(float(w0), 6), round(float(s0), 6), round(float(e0), 6), round(float(n0), 6)]}


def _tile_image(rgb: np.ndarray, n: int, cls: str, length_m):
    from PIL import Image, ImageDraw
    big = np.repeat(np.repeat(rgb, TILE_SCALE, axis=0), TILE_SCALE, axis=1)
    im = Image.fromarray(big, "RGB")
    d = ImageDraw.Draw(im)
    s = im.width
    half = max(6.0, (float(length_m or 0) / 40.0) * TILE_SCALE / 2 + 4)
    c = s / 2
    d.rectangle([c - half, c - half, c + half, c + half], outline=RED, width=2)
    font = color._font(10)
    d.rectangle([0, 0, 40, 13], fill=(0, 0, 0))
    d.text((2, 1), f"#{n} {cls}", fill=(255, 255, 255), font=font)
    return im


def render(over: np.ndarray, tiles: list[tuple[int, str, np.ndarray, float | None]], label: str, credit: str,
           quality: int = QUALITY_STEPS[0]) -> tuple[bytes, tuple[int, int], list[dict]]:
    """(WebP bytes, image (w, h), tile boxes [{n, box_px}] in image px). The overview at the top left."""
    from PIL import Image, ImageDraw
    H, W = over.shape[:2]
    ts = TILE_NATIVE * TILE_SCALE
    width = max(W, color.LEGEND_MIN_WIDTH, 3 * (ts + TILE_GAP) + TILE_GAP)   # three tiles a row at least
    per_row = max(1, (width - TILE_GAP) // (ts + TILE_GAP))
    nrows = math.ceil(len(tiles) / per_row) if tiles else 0
    head = 16 if tiles else 0
    tiles_h = head + nrows * (ts + TILE_GAP)
    lg = color.legend(width, label, credit)
    im = Image.new("RGB", (width, H + tiles_h + lg.height), color.BG)
    im.paste(Image.fromarray(over, "RGB"), (0, 0))
    meta = []
    if tiles:
        ImageDraw.Draw(im).text((4, H + 3), f"Zoom: strongest A candidates, 40 m px x{TILE_SCALE}, "
                                f"{TILE_NATIVE * 40 / 1000:g} km square", fill=color.FG, font=color._font(10))
        for k, (n, cls, rgb, length) in enumerate(tiles):
            x = TILE_GAP + (k % per_row) * (ts + TILE_GAP)
            y = H + head + (k // per_row) * (ts + TILE_GAP)
            im.paste(_tile_image(rgb, n, cls, length), (x, y))
            meta.append({"n": n, "box_px": [x, y, x + ts, y + ts]})
    im.paste(lg, (0, H + tiles_h))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=color.WEBP_METHOD)
    return buf.getvalue(), im.size, meta


def render_capped(over, tiles, label, credit, max_bytes: int = MAX_IMAGE_BYTES):
    out = None
    for q in QUALITY_STEPS:
        out = render(over, tiles, label, credit, quality=q)
        if len(out[0]) <= max_bytes:
            break
    return out


def _label(area_id: str, acq: dict, canvas: Canvas) -> str:
    mr, mc = canvas.grid.pixel_m
    plat = (acq.get("platform") or "").replace("sentinel-", "S").upper()
    return (f"{area_id} {acq['acquired_utc']} {plat} {acq.get('orbit_direction') or ''} RO{acq.get('relative_orbit') or ''}"
            f"  overview ~{round((mr + mc) / 2, -1):g} m/px")


def sheet_tile(c40: np.ndarray | None, c20: np.ndarray | None) -> tuple[np.ndarray, int] | None:
    """(64 px RGB, pixel metres) of one candidate: the 20 m cut-out of the second look if there is one, else the
    middle 32 px of the 40 m crop shown x2. Both 1.28 km square, centred on the candidate."""
    if c20 is not None and c20.shape[:2] == (SHEET_TILE, SHEET_TILE):
        return c20, 20
    if c40 is not None and c40.shape[0] >= SHEET_TILE // 2:
        h = c40.shape[0] // 2
        q = SHEET_TILE // 4
        mid = c40[h - q:h + q, h - q:h + q]
        return np.repeat(np.repeat(mid, 2, axis=0), 2, axis=1), 40
    return None


def tile_frame(length_m, res_m: int) -> list[float]:
    """The red frame inside a tile (tile px): the candidate's length (40 m or 20 m estimate) as a square, 6 px at least."""
    half = max(3.0, (float(length_m or 0) / 20.0) / 2 + 1.5)    # 20 m per tile px in both kinds of tile
    c = SHEET_TILE / 2
    return [round(c - half, 1), round(c - half, 1), round(c + half, 1), round(c + half, 1)]


def render_sheet(tiles: list[np.ndarray], max_bytes: int = MAX_SHEET_BYTES) -> tuple[bytes, tuple[int, int]]:
    """The tiles in rows of SHEET_COLS, no gaps, nothing burned in (the site draws the frames)."""
    from PIL import Image
    cols = min(SHEET_COLS, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    im = Image.new("RGB", (cols * SHEET_TILE, rows * SHEET_TILE), color.BG)
    for k, t in enumerate(tiles):
        im.paste(Image.fromarray(t, "RGB"), ((k % cols) * SHEET_TILE, (k // cols) * SHEET_TILE))
    out = b""
    for q in QUALITY_STEPS:
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=q, method=color.WEBP_METHOD)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            break
    return out, im.size


def _previous_tiles(sat_out: Path, old: dict | None) -> dict:
    """{(lon, lat) rounded 5: (rgb, res_m)} from the sheet already written for this pass (a pass fetched in parts)."""
    if not old or not old.get("sheet"):
        return {}
    try:
        from PIL import Image
        im = np.asarray(Image.open(sat_out / old["sheet"]).convert("RGB"))
    except (OSError, ValueError):
        return {}
    out = {}
    for c in old.get("candidates") or []:
        t, ce = c.get("tile"), c.get("center")
        if not t or not ce or len(t.get("xy") or []) != 2:
            continue
        x, y = t["xy"]
        sz = int(t.get("size") or SHEET_TILE)
        tile = im[y:y + sz, x:x + sz]
        if tile.shape[:2] == (SHEET_TILE, SHEET_TILE):
            out[(round(ce[0], 5), round(ce[1], 5))] = (tile.copy(), int(t.get("res_m") or 40))
    return out


def write_pass(sat_out: Path, area: dict, acq: dict, canvas: Canvas, feats: list[dict], crops: dict,
               generated_utc: str, crops20: dict | None = None) -> dict | None:
    """Image and chip file of one pass over one area. ``feats``: the pass's published A/B features (all parts);
    ``crops``: {(lon, lat): rgb} of the candidates fetched by this run (40 m), ``crops20`` the 20 m cut-outs of the
    second look (ocean.verify). Every candidate with a crop gets a tile in the sheet <stamp>-c.webp. A pass continued from an earlier run keeps
    the earlier part of the overview from the image already written. Returns the scene entry or None."""
    from PIL import Image
    if not canvas.has_data:
        return None
    d = sat_out / SUBDIR / area["id"]
    st = stamp_of(acq["acquired_utc"])
    img_rel, doc_rel = f"{SUBDIR}/{area['id']}/{st}.webp", f"{SUBDIR}/{area['id']}/{st}.json"
    over = canvas.rgb()
    old = _read(sat_out / doc_rel, None)
    if old and old.get("chip_px") == [canvas.grid.width, canvas.grid.height] and (sat_out / img_rel).exists():
        try:   # a pass fetched in parts: the part fetched earlier stays
            prev = np.asarray(Image.open(sat_out / img_rel).convert("RGB"))[:canvas.grid.height, :canvas.grid.width]
            if prev.shape == over.shape:
                over = np.where(canvas.covered()[..., None], over, prev)
        except OSError:
            pass
    ordered = _order([f for f in feats if f["properties"].get("class") in ("A", "B")])
    cands, tiles, sheet = [], [], []
    prev_tiles = _previous_tiles(sat_out, old)
    for n, f in enumerate(ordered, 1):
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        box = overview_box(canvas.grid, lon, lat)
        cands.append({"n": n, "center": [round(lon, 5), round(lat, 5)], "box_px": box["box_px"],
                      "zone": "open_sea", "sector": None, "class": p.get("class"),
                      # the site's text says "10 m pixels" for length_m: the 40 m estimate is given apart
                      "length_m": None, "width_m": None, "length_class": None,
                      "length_m_est_40m": p.get("length_m_est_40m"),
                      "contrast_vv_db": p.get("contrast_vv_db"), "vh_ratio_db": p.get("vh_ratio_db"),
                      "credibility": p.get("class"), "reliability": None})
        cands[-1].update({k: p.get(k) for k in ("confidence_label", "verified", "verify_verdict", "persist_db",
                                                 "n_prior", "length_m_20m", "width_m_20m", "heading_deg_20m")
                          if k in p})
        rgb = crops.get((lon, lat))
        if p.get("class") == "A" and rgb is not None and len(tiles) < MAX_TILES:
            tiles.append((n, "A", rgb, p.get("length_m_est_40m")))
        t = sheet_tile(rgb, (crops20 or {}).get((lon, lat))) if len(sheet) < SHEET_MAX else None
        if t is None and len(sheet) < SHEET_MAX:   # a pass fetched in parts: the tile of an earlier part stays
            t = prev_tiles.get((round(lon, 5), round(lat, 5)))
        if t is not None:
            k = len(sheet)
            sheet.append(t[0])
            x, y = (k % SHEET_COLS) * SHEET_TILE, (k // SHEET_COLS) * SHEET_TILE
            length = p.get("length_m_20m") if t[1] == 20 else p.get("length_m_est_40m")
            cands[-1]["tile"] = {"xy": [x, y], "size": SHEET_TILE, "res_m": t[1], "frame": tile_frame(length, t[1])}
    year = acq["acquired_utc"][:4]
    data, size, tmeta = render_capped(over, tiles, _label(area["id"], acq, canvas), satchips.credit(year))
    mr, mc = canvas.grid.pixel_m
    G = canvas.grid
    doc = {
        "version": VERSION, "site_id": area["id"], "area_id": area["id"], "site_name": area.get("name"),
        "acquired_utc": acq["acquired_utc"], "platform": acq.get("platform"),
        "orbit_direction": acq.get("orbit_direction"), "relative_orbit": acq.get("relative_orbit"),
        "image": img_rel, "thumb": None, "image_px": list(size), "thumb_px": None,
        "chip_px": [G.width, G.height], "bbox": [G.west, G.south, G.east, G.north],
        "pixel_m": {"row": round(mr, 2), "col": round(mc, 2)}, "note": NOTE,
        "candidates": cands, "tiles": tmeta,
        "counts": {"A": sum(c["class"] == "A" for c in cands), "B": sum(c["class"] == "B" for c in cands)},
        "caption": CAPTION, "credit": satchips.credit(year), "generated_utc": generated_utc,
    }
    sheet_rel = f"{SUBDIR}/{area['id']}/{st}{SHEET_SUFFIX}.webp"
    if sheet:
        sdata, spx = render_sheet(sheet)
        doc.update(sheet=sheet_rel, sheet_px=list(spx), sheet_note=SHEET_NOTE, sheet_v=generated_utc)
    d.mkdir(parents=True, exist_ok=True)
    if sheet:
        (sat_out / sheet_rel).write_bytes(sdata)
    (sat_out / img_rel).write_bytes(data)
    (sat_out / doc_rel).write_text(_dump(doc), encoding="utf-8")
    return {"area": area["id"], "acquired_utc": acq["acquired_utc"], "bytes": len(data) + (len(sdata) if sheet else 0),
            "image_px": list(size), "candidates": len(cands), "tiles": len(sheet)}


# ---------------------------------------------------------------- keep, held areas, index fragment and splice

def published_areas(cfg: dict) -> list[dict]:
    """Enabled, not held, in priority order (as ocean.monitor.areas_in_order)."""
    return sorted([a for a in cfg["areas"] if a.get("enabled", True) and a.get("publish", "publish") != "hold"],
                  key=lambda a: (a["priority"], a["id"]))


def refresh(sat_out: Path, cfg: dict, keep: int = KEEP) -> dict:
    """Deletes the images of held / unknown areas and those beyond the latest ``keep`` passes of an area, and
    rewrites chips/ocean/index.json. Returns {"areas", "scenes", "removed", "bytes"}."""
    root = sat_out / SUBDIR
    areas = published_areas(cfg)
    ids = {a["id"] for a in areas}
    removed = 0
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name not in ids:   # held, disabled or removed: nothing of it stays public
                for p in d.iterdir():
                    p.unlink()
                    removed += 1
                d.rmdir()
    entries, total, n_scenes, stamps = [], 0, 0, []
    for a in areas:
        d = root / a["id"]
        if not d.is_dir():
            continue
        docs = []
        for p in d.glob("*.json"):
            doc = _read(p, None)
            if not doc or not _STAMP.match(p.stem) or doc.get("acquired_utc") is None:
                p.unlink()
                removed += 1
                continue
            docs.append((doc["acquired_utc"], p, doc))
        docs.sort(key=lambda x: x[0], reverse=True)
        for _t, p, _doc in docs[keep:]:
            p.unlink()
            p.with_suffix(".webp").unlink(missing_ok=True)
            p.with_name(p.stem + SHEET_SUFFIX + ".webp").unlink(missing_ok=True)
            removed += 2
        kept = {p.stem for _t, p, _d in docs[:keep]} | {p.stem + SHEET_SUFFIX for _t, p, _d in docs[:keep]}
        for p in d.glob("*.webp"):     # an image without a chip file
            if p.stem not in kept:
                p.unlink()
                removed += 1
        scenes = []
        for t, p, doc in docs[:keep]:
            img = p.with_suffix(".webp")
            if not img.exists():
                continue
            total += img.stat().st_size + p.stat().st_size
            sh = p.with_name(p.stem + SHEET_SUFFIX + ".webp")
            total += sh.stat().st_size if sh.exists() else 0
            stamps.append(doc.get("generated_utc") or t)
            scenes.append({"acquired_utc": t, "platform": doc.get("platform"), "orbit_direction": doc.get("orbit_direction"),
                           "relative_orbit": doc.get("relative_orbit"), "chip": f"{SUBDIR}/{a['id']}/{p.name}",
                           "image": doc["image"], "thumb": None, "candidates": len(doc.get("candidates") or []),
                           "counts": doc.get("counts")})
        if scenes:
            n_scenes += len(scenes)
            w, s, e, n = a["bbox"]
            entries.append({"area_id": a["id"], "name": a.get("name"), "bbox": [w, s, e, n],
                            "center": [round((w + e) / 2, 6), round((s + n) / 2, 6)], "scenes": scenes})
    frag = {"version": VERSION, "keep": keep, "updated_utc": max(stamps) if stamps else None,
            "caption": CAPTION,
            "credit": satchips.credit(max(s["acquired_utc"][:4] for x in entries for s in x["scenes"])) if entries else None,
            "note": "外洋の域ごとの最新 keep 撮影（新しい順）。公開を保留した域（publish＝hold）は作らない。"
                    "古い撮影の画像は消す（取得した配列は残さないので作り直せない）",
            "ocean": entries}
    satchips._write_if_changed(sat_out / FRAGMENT, _dump(frag))
    return {"areas": len(entries), "scenes": n_scenes, "removed": removed, "bytes": total}


def splice(sat_out: Path) -> bool:
    """Writes the fragment's ``ocean`` into chips/index.json (the ports' ``sites`` untouched). True if changed."""
    p = sat_out / INDEX
    idx = _read(p, None)
    if not isinstance(idx, dict):
        idx = {"version": satchips.CHIPS_VERSION, "keep": satchips.KEEP, "updated_utc": None, "caption": CAPTION,
               "credit": None, "note": "", "sites": []}
    ocean = satchips.ocean_entries(sat_out)
    years = [s["acquired_utc"][:4] for x in (idx.get("sites") or []) + ocean for s in x.get("scenes") or []]
    new = {**idx, "ocean": ocean, "credit": satchips.credit(max(years)) if years else idx.get("credit"),
           "note": satchips.INDEX_NOTE}
    text = _dump(new)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return False
    satchips._write_if_changed(p, text)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="open-sea chips for the site (ocean-chips-1); no network")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splice")
    s.add_argument("--sat", type=Path, default=Path("docs/data/sat"))
    r = sub.add_parser("refresh")
    r.add_argument("--sat", type=Path, default=Path("docs/data/sat"))
    r.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "monitor.json")
    a = ap.parse_args(argv)
    if a.cmd == "splice":
        print("chips/index.json: " + ("更新" if splice(a.sat) else "変更なし"))
        return 0
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    print(json.dumps(refresh(a.sat, cfg), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
