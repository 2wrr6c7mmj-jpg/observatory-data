"""bl-cap's effect on the stored Sentinel-1 chips (design 設計書_レーダーの見落とし §5.1, the forward form; PU 0, no request).

The cells are the presence evaluation's re-count cells (eval/presence.json, rows with source "s1-recount": one section x one
answered Sentinel-2 day, judged on the paired Sentinel-1 scene). Each scene is counted twice with sat/quay.count on the stored chip
(the probe branches' probe-raw/s1-recount/<site>/s1-<stamp>.npz) and the baseline of its orbit (sat/baselines): as published, and
with the baseline capped (sat/quay.cap_baseline, +3 dB, fixed). Both are judged with the daily count's rule (sat/blcap.ship_length
>= 50 m on the longest stretch that belongs to the section, the count not restricted to berth lines, as the evaluation's re-count).

* ``now``: the daily count's rule on the published count, every cell.
* ``blcap``: the same, but a cell whose section is resident on the scene (sat/blcap.resident: the owner's answers strictly before
  the scene's day; nothing of the scene's day or later, and never the cell's own answer, the truth it is scored against, which
  can be up to 4 days before the scene) takes the capped count's judge.
* ``blcap_design``: the resident mark as the design's run made it (the evaluation's own cells of the section on the S2 days
  before the cell's S2 day), for the comparison with the design's numbers only.

Differences blcap - now in misses (fn) and false counts (fp) with 95 % intervals: the bootstrap over the scenes (site x S1), 2,000
draws, seed 0, paired (the design's method). The margin is not chosen here (choosing it again on these cells adds to the optimism).

    git archive origin/probe/sat-s1-recount-hq1113 probe-raw/s1-recount | tar -x -C <dir>/hq1113
    git archive origin/probe/sat-s1-recount-ops3 probe-raw/s1-recount | tar -x -C <dir>/ops3
    git archive origin/probe/sat-s1-recount-ops4 probe-raw/s1-recount | tar -x -C <dir>/ops4
    python -m sat.eval_blcap --raw <dir>/hq1113/probe-raw/s1-recount <dir>/ops3/probe-raw/s1-recount <dir>/ops4/probe-raw/s1-recount \\
        [--json docs/data/sat/eval/blcap-trial.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from . import blcap, quay, quay_lines, s1_recount, s2_labels
from . import baseline as fbase
from .grid import Grid
from .osm_mask import MASK_DIR, load_mask

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
PORTS = ("SYL", "SLP", "QGZ", "ZDH", "QJG", "QDH", "XSN", "LSN")   # the ports with re-count cells (Zhanjiang has none)
N_BOOT, SEED = 2000, 0
DESIGN = {"dfn": -8, "dfn_ci": [-15, -2], "dfp": 1, "dfp_ci": [0, 3], "cells": 633, "resident_cells": 77,
          "source": "2部 設計書_レーダーの見落とし §5.1（633 升、対 札の規則）"}


def _chip(raws: list[Path], sid: str, acq: str):
    for r in raws:
        z = s1_recount.load(s1_recount.raw_path(r, sid, acq))
        if z is not None:
            return z
    return None


def site_cells(out: Path, sid: str, presence: dict, raws: list[Path], log=print) -> list[dict]:
    """The site's re-count cells with the two judges and the two resident marks."""
    from .run import load_sites, site_sectors, sector_of
    site = {s["id"]: s for s in load_sites()}[sid]
    rows = presence["sites"][sid]["rows"]
    grid = Grid.for_site(site)
    mask, _ = load_mask(sid, int(site.get("mask_version", 1)), MASK_DIR)
    lines = quay_lines.load(sid, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), MASK_DIR)
    geom = quay.Geometry(grid, mask, lines)
    sectors = site_sectors(site)
    secs = {sc["section_id"]: sc for sc in s2_labels.sections(geom, lambda lo, la: sector_of(sectors, lo, la))}
    bls = {str(o): v for o in fbase.site_config(site)["orbits"] if (v := fbase.load(site, o)) is not None}
    lat = float(site.get("center_lat") or (grid.south + grid.north) / 2)
    ans = blcap.owner_answers(out, sid)
    # the design's mark: the evaluation's own cells of the section, on the S2 days before the cell's
    hist: dict = collections.defaultdict(list)
    design_mark = {}
    for r in sorted(rows, key=lambda r: r["date"]):
        prev = hist[r["section_id"]]
        design_mark[(r["section_id"], r["date"])] = len(prev) >= 3 and sum(prev) / len(prev) >= 0.75
        prev.append(r["y"])
    cells = [r for r in rows if r.get("source") == "s1-recount"]
    step = geom.cfg["step_m"]
    n_of = lambda pi: len(geom.profiles[pi]["ok"])
    judged: dict = {}
    for acq in sorted({r["s1"] for r in cells}):
        z = _chip(raws, sid, acq)
        if z is None:
            log(f"{sid} {acq}: no chip")
            continue
        vv, vh, meta = z
        bl = bls.get(str(meta["relative_orbit"]))
        if bl is None:
            log(f"{sid} {acq}: no baseline of orbit {meta['relative_orbit']}")
            continue
        layers = bl[0]
        look = quay.look_azimuth(meta.get("orbit_direction"), lat)
        q0 = quay.count(geom, vv, vh, layers, look_az_deg=look)
        qc = quay.count(geom, vv, vh, quay.cap_baseline(geom, layers, blcap.MARGIN_DB, blcap.QUANTILE), look_az_deg=look)
        want = [secs[s] for s in {r["section_id"] for r in cells if r["s1"] == acq} if s in secs]
        judged[acq] = (blcap.section_stretches(want, q0, step, n_of), blcap.section_stretches(want, qc, step, n_of))
    out_rows = []
    for r in cells:
        if r["s1"] not in judged or r["section_id"] not in judged[r["s1"]][0]:
            continue
        a, b = (j[r["section_id"]] for j in judged[r["s1"]])
        m = blcap.resident(ans, r["section_id"], r["s1"], exclude={r["date"]})   # never the cell's own truth
        out_rows.append({"site": sid, "section_id": r["section_id"], "date": r["date"], "s1": r["s1"], "y": int(r["y"]),
                         "p_eval": r["p"], "stretch_now": a["stretch_m"], "now": a["present"], "cap": b["present"],
                         "res": m["resident"], "res_design": design_mark[(r["section_id"], r["date"])]})
    log(f"{sid}: {len(out_rows)} cells, {sum(c['res'] for c in out_rows)} resident")
    return out_rows


JUDGES = {
    "now": lambda c: c["now"],
    "blcap": lambda c: c["cap"] if c["res"] else c["now"],
    "blcap_design": lambda c: c["cap"] if c["res_design"] else c["now"],
}


def metrics(cells: list[dict], j) -> dict:
    tp = sum(1 for c in cells if c["y"] and j(c))
    fn = sum(1 for c in cells if c["y"] and not j(c))
    fp = sum(1 for c in cells if not c["y"] and j(c))
    tn = len(cells) - tp - fn - fp
    r3 = lambda x: round(x, 3)
    return {"cells": len(cells), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": r3(tp / (tp + fn)) if tp + fn else None, "precision": r3(tp / (tp + fp)) if tp + fp else None,
            "f1": r3(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None, "accuracy": r3((tp + tn) / len(cells)) if cells else None}


def boot(cells: list[dict], ja, jb, n: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired differences (ja - jb) of fn, fp, f1 and accuracy: the 2.5 and 97.5 percentiles over draws of the scenes."""
    by: dict = collections.defaultdict(list)
    for c in cells:
        by[(c["site"], c["s1"])].append(c)
    keys = list(by)
    rng = np.random.default_rng(seed)
    pre = {k: np.array([[c["y"], ja(c), jb(c)] for c in by[k]], int) for k in keys}
    out = collections.defaultdict(list)
    for _ in range(n):
        a = np.concatenate([pre[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        y, A, B = a[:, 0], a[:, 1], a[:, 2]

        def m(P):
            tp = (y & P).sum(); fn = (y & ~P & 1).sum(); fp = ((1 - y) & P).sum(); tn = len(y) - tp - fn - fp
            return fn, fp, (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan), (tp + tn) / len(y)
        for name, x, z in zip(("dfn", "dfp", "df1", "dacc"), m(A), m(B)):
            out[name].append(x - z)
    return {k: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)] for k, v in out.items()}


def compare(cells: list[dict], name: str, ref: str = "now", with_ci: bool = True) -> dict:
    a, b = metrics(cells, JUDGES[name]), metrics(cells, JUDGES[ref])
    d = {"dfn": a["fn"] - b["fn"], "dfp": a["fp"] - b["fp"],
         "df1": round(a["f1"] - b["f1"], 3) if a["f1"] is not None and b["f1"] is not None else None}
    if with_ci:
        ci = boot(cells, JUDGES[name], JUDGES[ref])
        d.update(dfn_ci=ci["dfn"], dfp_ci=ci["dfp"], df1_ci=ci["df1"], dacc_ci=ci["dacc"])
    return {"metrics": a, "diff": d}


def evaluate(out: Path, raws: list[Path], ports=PORTS, log=print) -> dict:
    presence = json.loads((out / "eval" / "presence.json").read_text(encoding="utf-8"))
    cells = [c for sid in ports for c in site_cells(out, sid, presence, raws, log)]
    # the re-count must give the evaluation's own judge (length of the stretch >= 50 m) on every cell
    mismatch = sum(1 for c in cells if (1.0 if c["stretch_now"] >= 50 else 0.0) != c["p_eval"])
    per_port = {}
    for sid in ports:
        cs = [c for c in cells if c["site"] == sid]
        res = [c for c in cs if c["res"]]
        per_port[sid] = {"cells": len(cs), "resident_cells": len(res), "resident_present": sum(c["y"] for c in res),
                         "now": metrics(cs, JUDGES["now"]), "blcap": metrics(cs, JUDGES["blcap"]),
                         "dfn": metrics(cs, JUDGES["blcap"])["fn"] - metrics(cs, JUDGES["now"])["fn"],
                         "dfp": metrics(cs, JUDGES["blcap"])["fp"] - metrics(cs, JUDGES["now"])["fp"]}
    res = [c for c in cells if c["res"]]
    return {
        "version": blcap.VERSION, "margin_db": blcap.MARGIN_DB, "quantile": blcap.QUANTILE,
        "resident_rule": f"answers before the scene's day >= {blcap.MIN_ANSWERS}, share present >= {blcap.RESIDENT_MIN}",
        "cells": len(cells), "recount_mismatch_vs_presence": mismatch,
        "resident_cells": len(res), "resident_present": sum(c["y"] for c in res),
        "resident_cells_design_mark": sum(1 for c in cells if c["res_design"]),
        "now": metrics(cells, JUDGES["now"]),
        "blcap": compare(cells, "blcap"),
        "blcap_design_mark": compare(cells, "blcap_design"),
        "resident_only": {"now": metrics(res, JUDGES["now"]), "blcap": metrics(res, JUDGES["blcap"])},
        "per_port": per_port,
        "design": DESIGN,
        "bootstrap": f"scenes (site x S1) drawn with replacement, {N_BOOT} draws, seed {SEED}, paired differences",
        "pu": 0,
        "credit": "Contains modified Copernicus Sentinel data 2025–2026",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, nargs="+", required=True, help="probe-raw/s1-recount folders of the probe branches")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ports", nargs="*", default=list(PORTS))
    ap.add_argument("--json", type=Path, help="write the result here")
    a = ap.parse_args(argv)
    res = evaluate(a.out, a.raw, a.ports, log=lambda m: print(m, file=sys.stderr))
    txt = json.dumps(res, ensure_ascii=False, indent=1) + "\n"
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(txt, encoding="utf-8")
    b = res["blcap"]["diff"]
    print(f"cells {res['cells']} (re-count mismatch {res['recount_mismatch_vs_presence']}), resident {res['resident_cells']}: "
          f"misses {b['dfn']:+d} {b.get('dfn_ci')}, false counts {b['dfp']:+d} {b.get('dfp_ci')}, F1 {b['df1']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
