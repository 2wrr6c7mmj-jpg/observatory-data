"""Site grid (lat/lon, ~10 m pixels) and small raster helpers (numpy only)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def m_per_deg(lat: float) -> tuple[float, float]:
    """Metres per degree of latitude and of longitude at ``lat`` (WGS84 series)."""
    p = math.radians(lat)
    mlat = 111132.954 - 559.822 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    mlon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
    return mlat, mlon


@dataclass(frozen=True)
class Grid:
    """North-up EPSG:4326 grid; row 0 is the northern edge."""
    west: float
    south: float
    east: float
    north: float
    width: int
    height: int

    @classmethod
    def for_site(cls, site: dict) -> "Grid":
        """A rectangle ``bbox`` [west, south, east, north] if the site has one, else a square
        of ``size_m`` around the centre. Pixels are about ``pixel_m`` on each side."""
        if site.get("bbox"):
            w, s, e, n = (float(v) for v in site["bbox"])
            if not (w < e and s < n):
                raise ValueError(f"{site.get('id')}: bbox は [西, 南, 東, 北] の順で、西<東・南<北であること")
            mlat, mlon = m_per_deg((s + n) / 2)
            width = max(1, int(round((e - w) * mlon / site["pixel_m"])))
            height = max(1, int(round((n - s) * mlat / site["pixel_m"])))
            return cls(w, s, e, n, width, height)
        n = int(round(site["size_m"] / site["pixel_m"]))
        mlat, mlon = m_per_deg(site["center_lat"])
        half = site["size_m"] / 2
        dlat, dlon = half / mlat, half / mlon
        return cls(round(site["center_lon"] - dlon, 6), round(site["center_lat"] - dlat, 6),
                   round(site["center_lon"] + dlon, 6), round(site["center_lat"] + dlat, 6), n, n)

    @property
    def bbox(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @property
    def pixel_m(self) -> tuple[float, float]:
        """(metres per row, metres per column)."""
        mlat, mlon = m_per_deg((self.south + self.north) / 2)
        return ((self.north - self.south) / self.height * mlat, (self.east - self.west) / self.width * mlon)

    def to_pixel(self, lon, lat):
        """Fractional (row, col) of a lon/lat (pixel centres at .5)."""
        col = (np.asarray(lon, float) - self.west) / (self.east - self.west) * self.width
        row = (self.north - np.asarray(lat, float)) / (self.north - self.south) * self.height
        return row, col

    def area_km2(self) -> float:
        mr, mc = self.pixel_m
        return self.width * mc * self.height * mr / 1e6

    def to_lonlat(self, row, col):
        lon = self.west + np.asarray(col, float) / self.width * (self.east - self.west)
        lat = self.north - np.asarray(row, float) / self.height * (self.north - self.south)
        return lon, lat


def label(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Connected components of a boolean mask by row runs (union-find).

    Returns (labels, n) with labels 1..n and 0 for background.
    """
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    ext = 1 if connectivity == 8 else 0
    runs = []  # (row, start, end)
    row_runs = []
    for r in range(h):
        d = np.diff(np.concatenate([[0], mask[r].astype(np.int8), [0]]))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        first = len(runs)
        runs.extend((r, int(s), int(e)) for s, e in zip(starts, ends))
        row_runs.append((first, len(runs)))
    parent = list(range(len(runs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in range(1, h):
        a0, a1 = row_runs[r - 1]
        b0, b1 = row_runs[r]
        i, j = a0, b0
        while i < a1 and j < b1:
            _, ps, pe = runs[i]
            _, cs, ce = runs[j]
            if ps < ce + ext and pe > cs - ext:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if pe < ce:
                i += 1
            else:
                j += 1
    labels = np.zeros((h, w), dtype=np.int32)
    ids = {}
    for k, (r, s, e) in enumerate(runs):
        root = find(k)
        if root not in ids:
            ids[root] = len(ids) + 1
        labels[r, s:e] = ids[root]
    return labels, len(ids)


def dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary dilation with a disc of ``radius`` pixels."""
    mask = np.asarray(mask, bool)
    if radius <= 0:
        return mask.copy()
    r = int(math.floor(radius))
    h, w = mask.shape
    pad = np.pad(mask, r)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= pad[r + dy:r + dy + h, r + dx:r + dx + w]
    return out


def draw_line(mask: np.ndarray, rows, cols) -> None:
    """Mark the pixels a polyline (fractional row/col) passes through.

    Sampled at 1/4 pixel so consecutive pixels touch at least at a corner:
    a 4-connected free region cannot cross the line.
    """
    h, w = mask.shape
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    for k in range(len(rows) - 1):
        r0, c0, r1, c1 = rows[k], cols[k], rows[k + 1], cols[k + 1]
        if max(r0, r1) < -1 or min(r0, r1) > h + 1 or max(c0, c1) < -1 or min(c0, c1) > w + 1:
            continue
        n = int(math.ceil(max(abs(r1 - r0), abs(c1 - c0)) * 4)) + 1
        t = np.linspace(0.0, 1.0, n + 1)
        rr = np.floor(r0 + (r1 - r0) * t).astype(int)
        cc = np.floor(c0 + (c1 - c0) * t).astype(int)
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        mask[rr[ok], cc[ok]] = True


def fill_polygon(mask: np.ndarray, rows, cols) -> None:
    """Even-odd fill of a closed ring (fractional row/col) at pixel centres."""
    h, w = mask.shape
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    r_lo = max(0, int(math.floor(rows.min())))
    r_hi = min(h - 1, int(math.ceil(rows.max())))
    y0, y1 = rows[:-1], rows[1:]
    x0, x1 = cols[:-1], cols[1:]
    for r in range(r_lo, r_hi + 1):
        y = r + 0.5
        cross = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        if not cross.any():
            continue
        xs = np.sort(x0[cross] + (y - y0[cross]) * (x1[cross] - x0[cross]) / (y1[cross] - y0[cross]))
        for a, b in zip(xs[0::2], xs[1::2]):
            c0 = max(0, int(math.ceil(a - 0.5)))
            c1 = min(w - 1, int(math.floor(b - 0.5)))
            if c1 >= c0:
                mask[r, c0:c1 + 1] = True


def box_sum(a: np.ndarray, half: int) -> np.ndarray:
    """Sum over the (2*half+1)^2 window centred on each pixel (clipped at edges)."""
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1)
    r = np.arange(h)
    c = np.arange(w)
    r0 = np.clip(r - half, 0, h)[:, None]
    r1 = np.clip(r + half + 1, 0, h)[:, None]
    c0 = np.clip(c - half, 0, w)[None, :]
    c1 = np.clip(c + half + 1, 0, w)[None, :]
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
