"""Sectors of a site (multi-site PR-1): N listed sectors (bbox or polygon) and the rest of the chip.

A listed sector has ``id``, ``name``, ``role`` and either ``bbox`` [w, s, e, n] or ``polygon``
[[lon, lat], ...] (closed or not). The rest of the chip (``sector_rest``, default id "B") is added last.
A point belongs to the first listed sector that contains it (listed sectors must not overlap:
``overlaps``), otherwise to the rest.

role (used by est-1, est-2 and the S2 cloud check): naval (piers and quays of a naval base),
other (channel, commercial port), dock / slip / fitout (shipyards), lagoon / basin / outer (islands).
Zhanjiang has no explicit roles: a listed sector is naval and the rest other (the old A / B).
"""
from __future__ import annotations

import numpy as np

from .grid import Grid

ROLES = ("naval", "other", "dock", "slip", "fitout", "hall_apron", "lagoon", "basin", "outer")
DEFAULT_LISTED_ROLE = "naval"
DEFAULT_REST_ROLE = "other"


def site_sectors(site: dict) -> list[dict]:
    """[{id, name, role, bbox|None, polygon?}] for a site: its listed sectors, then the rest of the chip.
    No sectors: []. The keys of Zhanjiang's entries stay {id, name, bbox} plus role."""
    listed = site.get("sectors") or []
    if not listed:
        return []
    out = []
    seen = set()
    for sc in listed:
        if sc["id"] in seen:
            raise ValueError(f"{site['id']}: 区画 {sc['id']} が重複")
        seen.add(sc["id"])
        if sc.get("bbox") is None and not sc.get("polygon"):
            raise ValueError(f"{site['id']}: 区画 {sc['id']} に bbox も polygon も無い")
        role = sc.get("role") or DEFAULT_LISTED_ROLE
        if role not in ROLES:
            raise ValueError(f"{site['id']}: 区画 {sc['id']} の role {role} は不明")
        d = {"id": sc["id"], "name": sc.get("name"), "bbox": sc.get("bbox")}
        if sc.get("polygon"):
            d["polygon"] = [[float(x), float(y)] for x, y in sc["polygon"]]
        d["role"] = role
        out.append(d)
    rest = site.get("sector_rest") or {"id": "B", "name": "それ以外"}
    if rest["id"] in seen:
        raise ValueError(f"{site['id']}: 残りの区画 {rest['id']} が区画の id と重複")
    out.append({"id": rest["id"], "name": rest.get("name"), "bbox": None, "role": rest.get("role") or DEFAULT_REST_ROLE})
    return out


def sector_roles(sectors: list[dict]) -> dict[str, str]:
    return {sc["id"]: sc["role"] for sc in sectors}


def _in_polygon(lon, lat, poly) -> np.ndarray:
    """Even-odd rule (ray casting) for scalars or arrays of points."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    inside = np.zeros(np.broadcast(lon, lat).shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        cross = (y1 > lat) != (y2 > lat)
        xint = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (lon < xint)
    return inside


def contains(sc: dict, lon, lat):
    """True where (lon, lat) is in a listed sector (the rest, bbox None and no polygon, contains everything)."""
    if sc.get("polygon"):
        return _in_polygon(lon, lat, sc["polygon"])
    b = sc.get("bbox")
    if b is None:
        return np.ones(np.broadcast(np.asarray(lon), np.asarray(lat)).shape, bool)
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lon >= b[0]) & (lon < b[2]) & (lat >= b[1]) & (lat < b[3])


def sector_of(sectors: list[dict], lon: float, lat: float) -> str | None:
    for sc in sectors:
        b = sc["bbox"]
        if sc.get("polygon"):
            if bool(_in_polygon(lon, lat, sc["polygon"])):
                return sc["id"]
        elif b is None or (b[0] <= lon < b[2] and b[1] <= lat < b[3]):
            return sc["id"]
    return None


def _centres(grid: Grid):
    lon, lat = grid.to_lonlat(np.arange(grid.height)[:, None] + 0.5, np.arange(grid.width)[None, :] + 0.5)
    return np.broadcast_to(lon, (grid.height, grid.width)), np.broadcast_to(lat, (grid.height, grid.width))


def sector_raster(grid: Grid, sectors: list[dict]) -> np.ndarray | None:
    """Index (uint8) of the sector of each pixel centre (0 = the first listed one, the last = the rest),
    or None without sectors."""
    if not sectors:
        return None
    lon, lat = _centres(grid)
    out = np.full((grid.height, grid.width), len(sectors) - 1, np.uint8)
    done = np.zeros((grid.height, grid.width), bool)
    for i, sc in enumerate(sectors[:-1]):
        m = contains(sc, lon, lat) & ~done
        out[m] = i
        done |= m
    return out


def role_index(grid: Grid, site: dict, roles: tuple = ("naval",)) -> np.ndarray | None:
    """True where a pixel centre is in a listed sector whose role is in ``roles`` (None without sectors).
    Zhanjiang: its only listed sector (A, naval) — the old "sector A"."""
    sectors = site_sectors(site)
    if not sectors:
        return None
    lon, lat = _centres(grid)
    out = np.zeros((grid.height, grid.width), bool)
    for sc in sectors[:-1]:
        if sc["role"] in roles:
            out |= contains(sc, lon, lat)
    return out


def overlaps(site: dict, grid: Grid | None = None) -> list[tuple[str, str, int]]:
    """Pairs of listed sectors that share pixel centres: [(id, id, pixels)] (must be empty)."""
    sectors = site_sectors(site)[:-1]
    if len(sectors) < 2:
        return []
    grid = grid or Grid.for_site(site)
    lon, lat = _centres(grid)
    ms = [contains(sc, lon, lat) for sc in sectors]
    out = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            n = int((ms[i] & ms[j]).sum())
            if n:
                out.append((sectors[i]["id"], sectors[j]["id"], n))
    return out


def outside(site: dict, grid: Grid | None = None) -> list[str]:
    """Listed sectors with no pixel centre inside the site (a typo in the coordinates)."""
    grid = grid or Grid.for_site(site)
    lon, lat = _centres(grid)
    return [sc["id"] for sc in site_sectors(site)[:-1] if not contains(sc, lon, lat).any()]
