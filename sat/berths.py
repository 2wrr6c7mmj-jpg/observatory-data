"""Where ships berth (berth-1): the quay count, est-1, est-2 and the published chips use only these.

User's instruction (2026-11-01): parts of the shore where ships never berth are left out of the counts
and the judgement. A berth line is a line of the site's quay lines (sat/masks/<site>-quay-v<n>.json)
of kind pier or quay (the hand-drawn lines are in it with negative ids), except ``exclude_ways``, plus
the coastline sections listed in sat/berths.json (a way inside a sector or a bbox). Any other coastline
is not a berth.

* An occupied stretch of the quay count (quay.count) is kept when its line is a berth line at the
  stretch's middle point (``Berths.stretch_ok``).
* A candidate of the quay band (CFAR, within 50 m of the land mask) is kept when a berth line is
  within ``keep_m`` and no non-berth line is closer by more than ``own_tolerance_m``; otherwise it is
  "coastline" (a non-berth line is nearer) or "off_berth" (no line near: islands, breakwaters).
  Candidates of the open harbour water are kept as before.

naval-1 (user's instruction 2026-11-03: "the collector also picks up civil ships; take only what is clearly a warship
from the naval piers and quays"): every berth line of a site gets a class in sat/berth_class/<site>.json,
``naval`` (a pier or quay of a naval area, or of a shipyard building or refitting warships), ``civil`` or
``unknown``, with the evidence per line (``Berths.line_class``). From ``NAVAL_SINCE`` (scenes acquired from that
date) only the stretches and the candidates on a naval line go into the headline counts (quay_counts, est-1, est-2,
ship_counts, the red frames); the civil and unknown ones are kept with their class (detections/ property
``berth_class``) and counted apart (berth_class_counts.csv). Scenes before that date are never rewritten.

Pure Python at import (the collect workflows run without numpy).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BERTHS_PATH = HERE / "berths.json"
BERTH_KINDS = ("pier", "quay")
REASONS = ("off_berth", "coastline")
DENSIFY_M = 20.0


def load(path: Path = BERTHS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_CONF = load()
CLASS_DIR = HERE / "berth_class"
CLASSES = ("naval", "civil", "unknown")
NAVAL_VERSION = "naval-1"
NAVAL_SINCE = "2026-11-05"


def load_classes(site_id: str, d: Path | None = None) -> dict | None:
    """The class file of a site (sat/berth_class/<site>.json) or None (no class: the naval rule is not applied)."""
    p = (d or CLASS_DIR) / f"{site_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def naval_active(site_id: str, acquired_utc: str, since: str | None = None) -> bool:
    """Is the naval rule applied to a scene of the site acquired at ``acquired_utc``? (a class file and the date)"""
    since = since or NAVAL_SINCE
    return bool(acquired_utc) and acquired_utc[:10] >= since[:10] and (CLASS_DIR / f"{site_id}.json").exists()
VERSION = _CONF["version"]
SINCE_UTC = _CONF.get("since_utc")
KEEP_M = float(_CONF.get("keep_m", 100.0))
OWN_TOL_M = float(_CONF.get("own_tolerance_m", 30.0))


def _in_poly(lon: float, lat: float, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _in_area(area: dict, lon: float, lat: float) -> bool:
    if area.get("polygon"):
        return _in_poly(lon, lat, area["polygon"])
    b = area.get("bbox")
    return b is not None and b[0] <= lon < b[2] and b[1] <= lat < b[3]


def _mid(coords) -> tuple[float, float]:
    c = coords[len(coords) // 2]
    return float(c[0]), float(c[1])


class Berths:
    """The berth rule of one site. ``lines``: the site's quay lines (quay_lines.load); ``site``: its sites.json entry;
    ``classes``: the naval-1 classes of its lines (default: sat/berth_class/<site>.json)."""

    def __init__(self, site: dict, lines: dict | None, conf: dict | None = None, classes: dict | None | bool = True):
        conf = conf if conf is not None else _CONF
        sc = (conf.get("sites") or {}).get(site["id"]) or {}
        self.site_id = site["id"]
        self.keep_m = float(conf.get("keep_m", KEEP_M))
        self.tol_m = float(conf.get("own_tolerance_m", OWN_TOL_M))
        self.exclude = {str(w) for w in sc.get("exclude_ways") or []}
        areas = {x["id"]: x for x in (site.get("sectors") or [])}
        self.allow: dict[str, list[dict]] = {}
        for e in sc.get("coastline") or []:
            if e.get("sector"):
                area = areas.get(e["sector"], {})   # a sector not in the site (tests check sites.json): never matches
            elif e.get("bbox"):
                area = {"bbox": e["bbox"]}
            else:
                area = {"all": True}
            self.allow.setdefault(str(e["way"]), []).append(area)
        self.lines = lines or {"ways": []}
        self._segs = None
        # naval-1 classes: True = the site's file (sat/berth_class/<site>.json); a dict = these; None/False = none
        self.classes = load_classes(site["id"]) if classes is True else (classes or None)
        self._cls: dict[str, list[tuple[dict | None, str]]] = {}
        for e in (self.classes or {}).get("lines") or []:
            part = e.get("part") or {}
            if part.get("sector"):
                area = areas.get(part["sector"], {"bbox": [0, 0, 0, 0]})
            elif part.get("bbox") or part.get("polygon"):
                area = part
            else:
                area = None
            self._cls.setdefault(str(e["way"]), []).append((area, e["class"]))
        self.default_class = (self.classes or {}).get("default", "unknown")

    def line_class(self, way_id, lon: float, lat: float) -> str:
        """naval-1: the class of the berth line ``way_id`` at (lon, lat): naval, civil or unknown (no entry: the file's
        default, unknown). The first entry of the way whose part (sector, bbox or polygon) holds the point, or that has none."""
        for area, c in self._cls.get(str(way_id), []):
            if area is None or _in_area(area, lon, lat):
                return c
        return self.default_class

    def stretch_class(self, way_id, coords) -> str:
        if not coords:
            return self.default_class
        lon, lat = _mid(coords)
        return self.line_class(way_id, lon, lat)

    def point_ok(self, kind: str | None, way_id, lon: float, lat: float) -> bool:
        """Is the line ``way_id`` (of ``kind``) a berth at (lon, lat)?"""
        w = str(way_id)
        if w in self.exclude:
            return False
        if kind in BERTH_KINDS:
            return True
        return any(a.get("all") or _in_area(a, lon, lat) for a in self.allow.get(w, []))

    def stretch_ok(self, kind: str | None, way_id, coords) -> bool:
        """An occupied stretch (its display line ``coords``) is on a berth: judged at its middle point."""
        if not coords:
            return False
        lon, lat = _mid(coords)
        return self.point_ok(kind, way_id, lon, lat)

    def counts(self) -> dict:
        """Berth lines of the site: {"berth_lines", "coastline_sections", "excluded_ways"}."""
        n = sum(1 for w in self.lines.get("ways", []) if w["kind"] in BERTH_KINDS and str(w.get("id")) not in self.exclude)
        return {"berth_lines": n, "coastline_sections": sum(len(v) for v in self.allow.values()),
                "excluded_ways": sorted(self.exclude)}

    def _segments(self):
        """Line pieces (<= DENSIFY_M) in metres around the site's first point: (a, b, is_berth)."""
        if self._segs is None:
            import numpy as np
            ways = self.lines.get("ways", [])
            pts = [c for w in ways for c in w["coords"]]
            lat0 = sum(c[1] for c in pts) / len(pts) if pts else 0.0
            self._lat0 = lat0
            self._mx, self._my = 111320.0 * math.cos(math.radians(lat0)), 111132.0
            A, B, OK, CL = [], [], [], []
            for w in ways:
                c = w["coords"]
                for (x1, y1), (x2, y2) in zip(c, c[1:]):
                    d = math.hypot((x2 - x1) * self._mx, (y2 - y1) * self._my)
                    n = max(1, int(math.ceil(d / DENSIFY_M)))
                    for i in range(n):
                        ax, ay = x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n
                        bx, by = x1 + (x2 - x1) * (i + 1) / n, y1 + (y2 - y1) * (i + 1) / n
                        A.append((ax * self._mx, ay * self._my))
                        B.append((bx * self._mx, by * self._my))
                        OK.append(self.point_ok(w["kind"], w.get("id"), (ax + bx) / 2, (ay + by) / 2))
                        CL.append(CLASSES.index(self.line_class(w.get("id"), (ax + bx) / 2, (ay + by) / 2)))
            self._segs = (np.array(A, float).reshape(-1, 2), np.array(B, float).reshape(-1, 2), np.array(OK, bool))
            self._seg_cls = np.array(CL, int)
        return self._segs

    def classify(self, points) -> list[str]:
        """For quay-band candidates [(lon, lat), ...]: "" (kept), "coastline" or "off_berth"."""
        if not points:
            return []
        import numpy as np
        a, b, ok = self._segments()
        if not len(a):
            return ["off_berth"] * len(points)
        p = np.array([(lo * self._mx, la * self._my) for lo, la in points], float)
        d = b - a
        dd = np.maximum((d * d).sum(1), 1e-12)
        out = []
        for k in range(0, len(p), 256):
            q = p[k:k + 256, None, :]
            t = np.clip(((q - a[None]) * d[None]).sum(2) / dd[None], 0, 1)
            c = a[None] + t[..., None] * d[None]
            dist = np.hypot(q[..., 0] - c[..., 0], q[..., 1] - c[..., 1])
            db = np.where(ok[None], dist, np.inf).min(1)
            dn = np.where(~ok[None], dist, np.inf).min(1)
            for x, y in zip(db, dn):
                if x <= self.keep_m and x <= y + self.tol_m:
                    out.append("")
                elif min(x, y) <= self.keep_m:
                    out.append("coastline")
                else:
                    out.append("off_berth")
        return out


    def nearest_class(self, points, max_m: float) -> list[str | None]:
        """naval-1: for [(lon, lat), ...] the class of the nearest berth line within ``max_m`` (None: no berth line so near)."""
        if not points:
            return []
        import numpy as np
        a, b, ok = self._segments()
        if not ok.any():
            return [None] * len(points)
        a, b, cl = a[ok], b[ok], self._seg_cls[ok]
        p = np.array([(lo * self._mx, la * self._my) for lo, la in points], float)
        d = b - a
        dd = np.maximum((d * d).sum(1), 1e-12)
        out: list[str | None] = []
        for k in range(0, len(p), 256):
            q = p[k:k + 256, None, :]
            t = np.clip(((q - a[None]) * d[None]).sum(2) / dd[None], 0, 1)
            c = a[None] + t[..., None] * d[None]
            dist = np.hypot(q[..., 0] - c[..., 0], q[..., 1] - c[..., 1])
            i = dist.argmin(1)
            for j, x in zip(i, dist[np.arange(len(i)), i]):
                out.append(CLASSES[int(cl[j])] if x <= max_m else None)
        return out


_CACHE: dict = {}


def for_site(site: dict, lines_dir: Path | None = None) -> Berths:
    """The site's berth rule on its current quay lines (cached per site and lines version)."""
    from . import quay_lines
    lv = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
    key = (site["id"], lv, str(lines_dir), json.dumps(site.get("sectors"), sort_keys=True))
    if key not in _CACHE:
        lines = quay_lines.load(site["id"], lv, lines_dir) if lines_dir else quay_lines.load(site["id"], lv)
        _CACHE[key] = Berths(site, lines)
    return _CACHE[key]


def feature_ok(b: Berths, props: dict, coords) -> bool:
    """A stored quay feature (quay/*.geojson) on a berth; from NAVAL_SINCE (naval-1) on a naval berth only."""
    if not b.stretch_ok(props.get("line_kind"), props.get("osm_way"), coords):
        return False
    if b.classes and (props.get("acquired_utc") or "")[:10] >= NAVAL_SINCE:
        return b.stretch_class(props.get("osm_way"), coords) == "naval"
    return True


def class_settings() -> dict:
    """naval-1 for meta.json: the rule and, per site, the number of lines by class."""
    sites = {}
    for p in sorted(CLASS_DIR.glob("*.json")) if CLASS_DIR.is_dir() else []:
        c = json.loads(p.read_text(encoding="utf-8"))
        n = {k: 0 for k in CLASSES}
        for e in c.get("lines") or []:
            n[e["class"]] = n.get(e["class"], 0) + 1
        sites[c["site"]] = {"lines": n, "lines_version": c.get("lines_version"), "confidence": c.get("confidence"),
                            "note": c.get("note")}
    return {"version": NAVAL_VERSION, "since_utc": NAVAL_SINCE, "files": "sat/berth_class/<地点>.json",
            "geojson": "berth_class.geojson",
            "rule": ("船が着く線を一本ずつ海軍（naval：海軍の区域の桟橋・岸壁、建造・整備中の艦の着く造船所の岸壁を含む）・民間（civil）・"
                     "不明（unknown）に分ける。since_utc 以降の撮影では、見出しの数（ship_counts の harbour_candidates・quay_candidates、"
                     "quay_counts の占有、est-1・est-2、赤枠）に海軍の線に沿うものだけを入れる。民間・不明の候補は detections の berth_class に"
                     "区分を付けて残し、件数は berth_class_counts.csv に別に出す。それより前の撮影の行は書き換えない"),
            "sites": sites}


def settings() -> dict:
    return {"version": VERSION, "since_utc": SINCE_UTC, "file": "sat/berths.json", "note": _CONF.get("note"),
            "keep_m": KEEP_M, "own_tolerance_m": OWN_TOL_M, "rules": _CONF.get("rules"),
            "sites": {k: {"coastline": v.get("coastline") or [], "exclude_ways": v.get("exclude_ways") or [],
                          "note": v.get("note")} for k, v in (_CONF.get("sites") or {}).items()}}
