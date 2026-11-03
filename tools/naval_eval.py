"""naval-1: what the naval rule would have changed on the stored candidates of a month (no fetch, no PU).

For each site with a class file (sat/berth_class/<site>.json) and every scene of the month in docs/data/sat:
* candidates: the headline before = the rules now in force applied again to the stored candidates (berth-1 on the
  quay band; win-1 from the site's mask: land, pond, enclosed, anchorage - without the baseline's bright-pixel test);
  after = those of them on a naval line (sat/run.classify_naval);
* quay stretches: the stored stretches on a berth line (berth-1), and those on a naval line; the sum of their
  upper counts (max) and their number.

    python -m tools.naval_eval [--month 2026-10] [SITE ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat import berths, quay_lines, run
from sat import windows as fwindows
from sat.grid import Grid
from sat.osm_mask import load_mask

OUT = run.OUT_DIR


def evaluate(site: dict, month: str) -> dict | None:
    b0 = berths.for_site(site)                 # berth-1 (the rule now)
    bt = berths.for_site(site, naval=True)     # naval-1 (the classes and the added berth sections)
    if not bt.classes:
        return None
    det_p = OUT / run.detections_rel(site, month)
    q_p = OUT / run.quay_rel(site, month)
    dets_all = json.loads(det_p.read_text(encoding="utf-8"))["features"] if det_p.exists() else []
    quay_all = json.loads(q_p.read_text(encoding="utf-8"))["features"] if q_p.exists() else []
    tag = run.mask_tag(site)
    grid = Grid.for_site(site)
    mask, _ = load_mask(site["id"], int(site.get("mask_version", 1)))
    w0 = fwindows.Windows(site, grid, mask, b0) if b0.lines.get("ways") else None
    w1 = fwindows.Windows(site, grid, mask, bt) if bt.lines.get("ways") else None
    scenes = sorted({f["properties"]["acquired_utc"] for f in dets_all if f["properties"]["site_id"] == site["id"]
                     and run.row_mask_tag(f["properties"]) == tag})
    tot = {"scenes": len(scenes), "before": 0, "after": 0, "civil": 0, "unknown": 0,
           "stretch_before": 0, "stretch_after": 0, "max_before": 0, "max_after": 0, "max_civil": 0, "max_unknown": 0}

    class Q:   # the parts of QuayContext classify_naval uses
        def berths(self, naval=False):
            return bt if naval else b0

        def naval(self, acquired_utc):
            return True

    def windows(w, dets):
        if w is None:
            return
        for d, why in zip(dets, w.classify(dets, None)):
            d["window_excluded"] = why if why in fwindows.REASONS else ""
            if why == fwindows.ANCHORAGE:
                d["window"] = fwindows.ANCHORAGE
    for t in scenes:
        base = []
        for f in dets_all:
            p = f["properties"]
            if p["acquired_utc"] != t or p["site_id"] != site["id"] or run.row_mask_tag(p) != tag:
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            r, c = grid.to_pixel(lon, lat)
            base.append({"zone": p["zone"], "lon": lon, "lat": lat, "row": float(r), "col": float(c)})
        dets = [dict(d) for d in base]
        run.classify_band(b0, dets)
        windows(w0, dets)
        tot["before"] += sum(run.at_berth(d) for d in dets)
        dets = [dict(d) for d in base]
        run.classify_band(bt, dets)
        windows(w1, dets)
        n = run.classify_naval(Q(), site, {"acquired_utc": t}, dets)
        tot["after"] += sum(run.at_berth(d) for d in dets)
        tot["civil"] += n["harbour_civil"] + n["quay_civil"]
        tot["unknown"] += n["harbour_unknown"] + n["quay_unknown"]
    for f in quay_all:
        p = f["properties"]
        if p["site_id"] != site["id"] or run.row_mask_tag(p) != tag:
            continue
        coords = f["geometry"]["coordinates"]
        if b0.stretch_ok(p.get("line_kind"), p.get("osm_way"), coords):
            tot["stretch_before"] += 1
            tot["max_before"] += int(p["max"])
        if not bt.stretch_ok(p.get("line_kind"), p.get("osm_way"), coords):
            continue
        c = bt.stretch_class(p.get("osm_way"), coords)
        if c == "naval":
            tot["stretch_after"] += 1
            tot["max_after"] += int(p["max"])
        else:
            tot[f"max_{c}"] += int(p["max"])
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-10")
    ap.add_argument("sites", nargs="*")
    a = ap.parse_args(argv)
    berths.NAVAL_SINCE = "0000-00-00"   # evaluation only: the rule on every stored scene
    out = {}
    for s in run.load_sites():
        if not s.get("enabled") or (a.sites and s["id"] not in a.sites):
            continue
        r = evaluate(s, a.month)
        if r is not None:
            out[s["id"]] = r
            print(s["id"], json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
