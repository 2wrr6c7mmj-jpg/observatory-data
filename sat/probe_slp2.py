"""Probe only (probe/ports-slp2-*; not merged): SLP-land-v2 (bbox north to 18.242N), its quay lines v2 and the
orbit-157 baseline ro157-bl-1 with the same sat/baseline.py as Zhanjiang.

Scenes: catalog of the last ``--lookback`` days, scenes whose catalog footprint covers less than the
baseline's min_coverage of the water are not fetched (no PU), the newest ``--scenes`` of orbit 157.
PU: a hard cap on the PU of the response headers (``--cap``; the next request is not sent if the PU so far
plus the largest header PU so far would exceed it). Only the baseline layers are kept (no raw sigma0).

The October guard (``guard``): sat (main docs/data/sat/usage.csv) + ocean (largest docs/data/ocean/usage.csv
of the probe/ocean-* branches, at least the 5,000 PU reserve) + ports (largest docs/data/sat/ports_usage.csv of
the probe/ports-* branches) + this run <= 9,945 PU.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import baseline as fbase
from . import cdse
from .grid import Grid
from .osm_mask import MASK_DIR, WATER, load_mask
from .run import load_sites

MONTH = "2026-10"
GLOBAL_CAP = 9945.0
OCEAN_RESERVE = 5000.0


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def _sum_month(text: str, month: str = MONTH) -> float:
    if not text.strip():
        return 0.0
    return round(sum(float(r.get("processing_units") or 0) for r in csv.DictReader(io.StringIO(text))
                     if (r.get("run_utc") or "").startswith(month)), 3)


def guard() -> dict:
    refs = [r for r in _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split() if r]
    sat = _sum_month(_git("show", "origin/main:docs/data/sat/usage.csv"))
    ocean = {r: _sum_month(_git("show", f"{r}:docs/data/ocean/usage.csv")) for r in refs if r.startswith("origin/probe/ocean-")}
    ports = {r: _sum_month(_git("show", f"{r}:docs/data/sat/ports_usage.csv")) for r in refs if r.startswith("origin/probe/ports-")}
    o_max = max(ocean.values(), default=0.0)
    p_max = max(ports.values(), default=0.0)
    used = sat + max(o_max, OCEAN_RESERVE) + p_max
    return {"month": MONTH, "sat_main": sat, "ocean_max": o_max, "ocean_counted": max(o_max, OCEAN_RESERVE),
            "ocean_by_branch": ocean, "ports_max": p_max, "ports_by_branch": ports, "used": round(used, 3),
            "global_cap": GLOBAL_CAP, "room": round(GLOBAL_CAP - used, 3)}


class HardCapClient(cdse.Client):
    def __init__(self, cap: float, **kw):
        super().__init__(**kw)
        self.cap = cap
        self.max_seen = None

    def process(self, bbox, width, height, t_from, t_to, with_lia=True):
        est = self.max_seen if self.max_seen is not None else cdse.estimate_pu(width, height, 3 if with_lia else 2)
        if self.processing_units + est > self.cap:
            raise cdse.CDSEError(f"PU の上限 {self.cap:.1f} を超えるため送らない（使用 {self.processing_units:.2f}、次の見込み {est:.2f}）")
        data, pu = super().process(bbox, width, height, t_from, t_to, with_lia)
        self.max_seen = max(self.max_seen or 0.0, float(pu))
        return data, pu


class CatalogView:
    def __init__(self, client, feats):
        self._c, self._feats = client, feats

    def catalog(self, *a, **k):
        return list(self._feats)

    def __getattr__(self, name):
        return getattr(self._c, name)


def _rings(geom):
    if not geom:
        return []
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return [c[0]] if c else []
    if t == "MultiPolygon":
        return [p[0] for p in c if p]
    return []


def _inside(x, y, ring):
    r = np.asarray(ring, float)
    xs, ys = r[:, 0], r[:, 1]
    out = np.zeros(x.shape, bool)
    j = len(r) - 1
    for i in range(len(r)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if yi != yj:
            out ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        j = i
    return out


def footprint_share(geoms, grid: Grid, water: np.ndarray, step: int = 8):
    rings = [r for g in geoms for r in _rings(g)]
    if not rings:
        return None
    rr = np.arange(step // 2, grid.height, step)
    cc = np.arange(step // 2, grid.width, step)
    lon, lat = grid.to_lonlat(rr[:, None] + 0.5, cc[None, :] + 0.5)
    lon, lat = np.broadcast_to(lon, (len(rr), len(cc))), np.broadcast_to(lat, (len(rr), len(cc)))
    sel = water[np.ix_(rr, cc)]
    if not sel.any():
        return None
    inside = np.zeros(sel.shape, bool)
    for ring in rings:
        inside |= _inside(lon, lat, ring)
    return round(float(inside[sel].mean()), 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["guard", "baseline"])
    ap.add_argument("--site", default="SLP")
    ap.add_argument("--orbit", type=int, default=157)
    ap.add_argument("--scenes", type=int, default=24)
    ap.add_argument("--lookback", type=int, default=730)
    ap.add_argument("--cap", type=float, default=60.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "guard":
        g = guard()
        g["cap_this_run"] = round(max(0.0, min(args.cap, g["room"])), 3)
        print(json.dumps(g, ensure_ascii=False, indent=1))
        return 0
    now = datetime.now(timezone.utc)
    site = next(s for s in load_sites() if s["id"] == args.site)
    mask, _ = load_mask(site["id"], int(site["mask_version"]), MASK_DIR)
    grid = Grid.for_site(site)
    water = mask == WATER
    client = HardCapClient(args.cap)
    summary = {"run_utc": cdse.iso(now), "site_id": site["id"], "mask_version": fbase.tag(site), "cap_pu": args.cap,
               "lookback_days": args.lookback}
    rc = 0
    try:
        feats = client.catalog(grid.bbox, now - timedelta(days=args.lookback), now)
        geo = {f.get("id"): f.get("geometry") for f in feats}
        acqs = cdse.acquisitions(feats)
        min_cov = fbase.DEFAULTS["min_coverage"]
        ok, dropped = [], 0
        for a in acqs:
            if a["relative_orbit"] != args.orbit:
                continue
            sh = footprint_share([geo.get(i) for i in a["ids"] if geo.get(i)], grid, water)
            if sh is not None and sh < min_cov:
                dropped += 1
                continue
            ok.append(a)
        ok.sort(key=lambda a: a["acquired_utc"])
        pick = ok[-args.scenes:]
        keep = {i for a in pick for i in a["ids"]}
        summary.update(catalog_features=len(feats), orbit_acquisitions=len(ok) + dropped, footprint_dropped=dropped,
                       picked=len(pick), picked_first=pick[0]["acquired_utc"] if pick else None,
                       picked_last=pick[-1]["acquired_utc"] if pick else None)
        print(f"catalog {len(feats)}; orbit {args.orbit}: {len(ok) + dropped} acquisitions, footprint drop {dropped}, "
              f"pick {len(pick)}", file=sys.stderr)
        bsite = {**site, "baseline": {"orbits": [args.orbit], "lookback_days": args.lookback,
                                      "max_scenes": args.scenes, "min_scenes": 24}}
        # the estimate check of build() uses the conservative rule; the hard cap above uses the header PU
        need = cdse.estimate_pu(grid.width, grid.height, 2) * len(pick)
        res = fbase.build(CatalogView(client, [f for f in feats if f.get("id") in keep]), bsite, mask, args.out, now,
                          need + 1.0, log=lambda m: print(m, file=sys.stderr))
        summary.update(res)
        pj = fbase.paths(bsite, args.orbit, args.out)["json"]
        if pj.exists():
            meta = json.loads(pj.read_text(encoding="utf-8"))
            old = [x["acquired_utc"] for x in meta.get("scene_list", [])
                   if x.get("used") and x["acquired_utc"] < cdse.iso(now - timedelta(days=365))]
            meta["period"] = {"lookback_days": args.lookback, "first_utc": meta.get("first_utc"),
                              "last_utc": meta.get("last_utc"), "older_than_12_months": len(old),
                              "note": (f"範囲 v2（北端 18.242N）で作り直し。カタログの足跡で水域の {min_cov:.0%} 未満の撮影は取らず、"
                                       f"最新から {args.scenes} 撮影（さかのぼり {args.lookback} 日、12 か月より前の撮影 {len(old)} 枚）")}
            pj.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            summary["older_than_12_months"] = len(old)
    except cdse.CDSEError as e:
        summary["error"] = str(e)[:300]
        rc = 1
    finally:
        summary.update(requests=client.requests, processing_units=round(client.processing_units, 3),
                       pu_source="header" if client.pu_reported else "estimate")
        text = json.dumps(summary, ensure_ascii=False, indent=1)
        print(text)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
