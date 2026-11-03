"""Where a harbour candidate may stand (win-1): water joined to the sea, along the berth lines.

User's instruction (2026-11-03): the candidate windows (the red frames of the public chips and the counts
harbour_candidates / quay_candidates) were found on land, on ponds and in water closed in by piers, where no
ship berths. From ``since_utc`` of a site's ``windows`` block (sat/sites.json) a CFAR candidate is kept only
where its centre pixel is

1. water of the land mask and not a stable bright target of the baseline of the scene's orbit (vv_p10 above
   ``land_p10_db`` with vv_p90 - vv_p10 under 3 dB, sat/baseline.always_occupied: land or a structure the
   mask misses) - else ``land``;
2. joined to the sea: the water component touches the edge of the chip (the sea goes on beyond it) - else
   ``pond`` (water not joined to the sea: ponds, inland water, mask errors);
3. reachable by a ship: still joined to the sea after an opening with a disc of ``min_channel_m`` (water
   narrower than that, closed in by piers, is cut off) - else ``enclosed``;
4. for a candidate of the open harbour water (zone harbour): within ``band_m`` of a berth line (sat/berths.py:
   piers, quays and the listed coastline sections; a ship's beam and the ships moored outboard of it) - else it
   is kept but counted apart as ``anchorage`` (a ship at anchor in the harbour, not at a berth; user's decision
   2026-11-03: not thrown away). A site without berth lines skips this step.

Candidates of steps 1-3 stay in detections/ with the property ``window_excluded`` (the reason) and are counted in
status.json (``window_excluded``); anchorage candidates carry ``window`` = "anchorage", are counted in
anchorage_counts.csv and get their own frame (``window``) in the chips. The water regions of pond, enclosed and
anchorage are written per site to windows/<site>.geojson so they can be checked by eye. Scenes before
``since_utc`` are never rewritten.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .grid import dilate, label
from .osm_mask import WATER

VERSION = "win-1"
REASONS = ("land", "pond", "enclosed")          # left out
ANCHORAGE = "anchorage"                          # kept, counted apart from the berths
DEFAULTS = {
    "band_m": 80.0,          # a ship's beam (~20 m) twice abreast, the pier's half deck and the 3x3 spread
    "min_channel_m": 30.0,   # narrower water is closed in (no ship gets in)
    "land_p10_db": -8.0,     # sat/baseline.OCC_P10_DB
    "land_spread_db": 3.0,   # sat/baseline.OCC_SPREAD_DB
    "region_min_px": 4,      # regions smaller than this are not written (still applied)
}
NOTE = {
    "land": "陸マスクの外（陸・構造物）か、基準で常に明るい画素（vv_p10 が高く p90−p10 が小さい：マスクに無い陸・構造物）",
    "pond": "海とつながらない水面（池・内水・マスクの誤り）",
    "enclosed": "海とはつながるが、幅 min_channel_m 未満の水路の奥（桟橋に囲まれて艦が入れない水面）",
    "anchorage": "船が着く線（sat/berths.py）から band_m より遠い開けた水面（外さない。港内の錨泊として岸壁とは別に数える）",
}


def config(site: dict) -> dict | None:
    """The site's windows block merged with the defaults, or None (no rule for the site)."""
    w = site.get("windows")
    if not w:
        return None
    return {**DEFAULTS, **w}


def active(site: dict, acquired_utc: str) -> bool:
    cfg = config(site)
    return bool(cfg) and bool(cfg.get("since_utc")) and acquired_utc[:10] >= cfg["since_utc"][:10]


def _erode(m: np.ndarray, r: float) -> np.ndarray:
    """Erosion with a disc; outside the chip counts as inside (the sea goes on)."""
    if r <= 0:
        return m.copy()
    k = int(math.floor(r))
    pad = np.pad(~m, k, constant_values=False)
    h, w = m.shape
    out = np.zeros_like(m)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dy * dy + dx * dx <= r * r:
                out |= pad[k + dy:k + dy + h, k + dx:k + dx + w]
    return m & ~out


def _touching_edge(m: np.ndarray) -> np.ndarray:
    lab, n = label(m, 4)
    if not n:
        return np.zeros_like(m)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    if not len(edge):  # nothing touches the edge (a lake chip): the largest component is the "sea"
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        edge = np.array([int(sizes.argmax())])
    return np.isin(lab, edge)


def berth_raster(grid, berths, step_m: float = 5.0) -> np.ndarray:
    """Pixels on the berth lines of ``berths`` (sat/berths.Berths)."""
    out = np.zeros((grid.height, grid.width), bool)
    if berths is None or not berths.lines.get("ways"):
        return out
    a, b, ok = berths._segments()
    for (ax, ay), (bx, by) in zip(a[ok], b[ok]):
        n = max(2, int(math.ceil(math.hypot(bx - ax, by - ay) / step_m)) + 1)
        t = np.linspace(0, 1, n)
        lon = (ax + (bx - ax) * t) / berths._mx
        lat = (ay + (by - ay) * t) / berths._my
        r, c = grid.to_pixel(lon, lat)
        r, c = np.floor(r).astype(int), np.floor(c).astype(int)
        keep = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
        out[r[keep], c[keep]] = True
    return out


class Windows:
    """The window rule of one site on its grid and land mask (fixed per site, mask and lines)."""

    def __init__(self, site: dict, grid, mask: np.ndarray, berths, cfg: dict | None = None):
        self.site_id = site["id"]
        self.cfg = cfg = {**DEFAULTS, **(cfg if cfg is not None else (config(site) or {}))}
        self.grid = grid
        ry, rx = grid.pixel_m
        px = (ry + rx) / 2.0
        self.px = px
        water = np.asarray(mask) == WATER
        self.water = water
        sea = _touching_edge(water) & water
        core = _touching_edge(_erode(sea, cfg["min_channel_m"] / 2.0 / px))
        navigable = dilate(core, cfg["min_channel_m"] / 2.0 / px) & sea
        lines = berth_raster(grid, berths)
        band = dilate(lines, cfg["band_m"] / px) if lines.any() else np.zeros_like(water)
        self.has_lines = bool(lines.any())
        self.code = np.zeros(water.shape, np.uint8)   # 0 berth band, 1 land, 2 pond, 3 enclosed, 4 anchorage
        self.code[~water] = 1
        self.code[water & ~sea] = 2
        self.code[sea & ~navigable] = 3
        if self.has_lines:                            # no berth line: step 4 is skipped (no band to measure from)
            self.code[navigable & ~band] = 4

    def classify(self, dets: list[dict], base: dict | None = None) -> list[str]:
        """Per candidate from its centre pixel (``row``, ``col``): "" (along a berth), "anchorage" (open harbour water
        beyond band_m, zone harbour only) or the reason it is left out (land, pond, enclosed). ``base``: the baseline
        layers of the scene's orbit (vv_p10, vv_p90; None = the mask alone decides "land")."""
        h, w = self.code.shape
        out = []
        for d in dets:
            r = min(max(int(round(d["row"])), 0), h - 1)
            c = min(max(int(round(d["col"])), 0), w - 1)
            k = int(self.code[r, c])
            if k == 0 and base is not None and "vv_p10" in base and "vv_p90" in base:
                p10, p90 = base["vv_p10"][r, c], base["vv_p90"][r, c]
                if np.isfinite(p10) and np.isfinite(p90) and p10 > self.cfg["land_p10_db"] and (p90 - p10) < self.cfg["land_spread_db"]:
                    k = 1
            if k == 4:
                out.append(ANCHORAGE if d.get("zone", "harbour") == "harbour" else "")
            else:
                out.append("" if k == 0 else REASONS[k - 1])
        return out

    def regions(self) -> dict:
        """GeoJSON of the water regions of pond, enclosed (left out) and anchorage (counted apart): one feature per
        region, its outline as the bounding box, centre, area and reason (the land is the mask itself, not written)."""
        feats = []
        g = self.grid
        for k, why in ((2, "pond"), (3, "enclosed"), (4, ANCHORAGE)):
            lab, n = label(self.code == k, 8)
            if not n:
                continue
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            for i in range(1, n + 1):
                if sizes[i] < self.cfg["region_min_px"]:
                    continue
                rr, cc = np.nonzero(lab == i)
                w_, n_ = g.to_lonlat(rr.max() + 1.0, cc.min() + 0.0)
                e_, s_ = g.to_lonlat(rr.min() + 0.0, cc.max() + 1.0)
                lo, la = g.to_lonlat(rr.mean() + 0.5, cc.mean() + 0.5)
                west, east = float(min(w_, e_)), float(max(w_, e_))
                south, north = float(min(n_, s_)), float(max(n_, s_))
                ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
                feats.append({"type": "Feature",
                              "geometry": {"type": "Polygon", "coordinates": [[[round(x, 6), round(y, 6)] for x, y in ring]]},
                              "properties": {"site_id": self.site_id, "reason": why, "pixels": int(sizes[i]),
                                             "area_m2": int(round(sizes[i] * self.px * self.px)),
                                             "center": [round(float(lo), 6), round(float(la), 6)],
                                             "version": VERSION}})
        feats.sort(key=lambda f: (f["properties"]["reason"], -f["properties"]["pixels"]))
        return {"type": "FeatureCollection", "version": VERSION, "site_id": self.site_id,
                "settings": {k: self.cfg.get(k) for k in ("since_utc", "band_m", "min_channel_m", "land_p10_db", "land_spread_db")},
                "note": ("外した水面（pond・enclosed）と港内の錨泊として別に数える水面（anchorage）。輪郭は外接矩形。陸（land）は陸マスクそのものと"
                         "基準の常に明るい画素で、ここには書かない。" + "／".join(f"{k}：{v}" for k, v in NOTE.items())),
                "has_berth_lines": self.has_lines,
                "counts_px": {r: int((self.code == i + 1).sum()) for i, r in enumerate(REASONS + (ANCHORAGE,))},
                "features": feats}


def dumps(fc: dict) -> str:
    return json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n"


def settings() -> dict:
    return {"version": VERSION, "defaults": DEFAULTS, "reasons": NOTE,
            "rule": ("候補は中心の画素が陸・池・幅 min_channel_m 未満の水路の奥なら外す（detections/ に window_excluded 付きで残し、件数は status.json の window_excluded）。"
                     "港内水域の候補で船が着く線から band_m より遠いものは外さず、港内の錨泊（window＝anchorage）として anchorage_counts.csv に別に数え、"
                     "赤枠も岸壁の候補と区別して出す（chips の window）。船が着く線の無い地点は錨泊の区別をしない。水面の区分は windows/<地点>.geojson")}
