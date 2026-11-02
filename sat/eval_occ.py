"""The current quay occupancy (quay-v2 / est-2) against the change test (occ-1) on the owner's eye labels (no fetch, PU 0).

Labels: docs/data/sat/labels/<site>-s2-answers.csv (the owner's answers per quay section and Sentinel-2 day:
verified_occupied, verified_c; the sections and days are those of labels/<site>/review.json). The answers can also
be read from a git ref (``--answers-ref``, e.g. a branch not merged yet).

Each S2 day is paired with the Sentinel-1 scene nearest in time within S2_MAX_H (96 h) among the scenes whose
inputs are at hand without a fetch. What is at hand (2026-11-03): for SYL and SLP only the three scenes of the
public chips (docs/data/sat/chips/<site>/, orbit 157, 2026-10-02 / 10-14 / 10-26). Their pixels are the display
image (sat/color.py: Lee 3x3, VV -22..0 dB and VH -28..-6 dB in 8 bits, lossy WebP), turned back into dB here;
values outside the ranges are clipped. So three methods are compared on the same days:

* ``est2``      the published est-2 (from the raw sigma0 of the daily run): P(occupied) = P(real) of the stretch on
                the section (0 without one), c = argmax of the 4 classes (as sat/eval_est2.py);
* ``quay_webp`` the fixed-dB rule of sat/quay.py (dVV >= 6 dB and dVH >= 4 dB, runs of 30 m) on the display image:
                P(occupied) is 0 or 1 (the rule has no probability);
* ``occ1``      sat/occ_test.py on the display image: P(occupied) and c.

The display image is a degraded input (clipping above 0 dB VV, 8-bit steps, lossy compression, Lee filter), so the
comparison of quay_webp and occ1 is like for like, while est2 had the raw data. A fair test of occ1 on raw sigma0
needs the scenes again (``plan``: PU estimate, nothing is fetched).

Metrics per site and pooled: occupancy Brier score, ROC-AUC (Mann-Whitney, ties counted half), agreement at
P >= 0.5; rows abreast: confusion matrix owner c (0-3) -> predicted c.

    python -m sat.eval_occ [--answers-ref origin/hq/labels-syl-slp] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import baseline as fbase, cdse, color, occ_test, quay, quay_est2 as e2, quay_lines
from .eval_est2 import p4, stretch_of_section  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
EVAL_VERSION = "eval-occ-1"
SITES = ("SYL", "SLP")


# ---------------------------------------------------------------- inputs

def read_answers(site_id: str, out: Path = OUT, ref: str | None = None) -> list[dict]:
    rel = f"docs/data/sat/labels/{site_id}-s2-answers.csv"
    if ref:
        r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return []
        text = r.stdout
    else:
        p = out / "labels" / f"{site_id}-s2-answers.csv"
        if not p.exists():
            return []
        text = p.read_text(encoding="utf-8")
    return [row for row in csv.DictReader(io.StringIO(text)) if row.get("site_id") == site_id]


def decode_display(data: bytes, chip_px: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(VV dB, VH dB) of a display WebP (sat/color.py, gamma 1 on R and G); no data (0, 0, 0) -> NaN."""
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float)
    w, h = chip_px
    im = im[:h, :w]
    lo, hi = color.VV_RANGE
    vv = lo + im[..., 0] / 255.0 * (hi - lo)
    lo, hi = color.VH_RANGE
    vh = lo + im[..., 1] / 255.0 * (hi - lo)
    nod = (im == 0).all(-1)
    vv[nod] = np.nan
    vh[nod] = np.nan
    return vv, vh


def display_scenes(site_id: str, out: Path = OUT) -> list[dict]:
    rows = []
    for p in sorted((out / "chips" / site_id).glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        img = out / c["image"]
        if img.exists():
            rows.append({"acquired_utc": c["acquired_utc"], "orbit": int(c["relative_orbit"]), "image": img,
                         "chip_px": tuple(c["chip_px"]), "bbox": c["bbox"], "mask_version": c["mask_version"]})
    return rows


# ---------------------------------------------------------------- metrics

def brier(p, y) -> float | None:
    return round(float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)), 4) if len(y) else None


def auc(p, y) -> float | None:
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return round(float((gt + 0.5 * eq) / (len(pos) * len(neg))), 4)


def confusion(y, c) -> dict:
    out = {}
    for a, b in zip(y, c):
        k = f"{a}->{b}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summary(rows: list[dict], key: str) -> dict:
    rr = [r for r in rows if r.get(f"{key}_p") is not None]
    p = [r[f"{key}_p"] for r in rr]
    y = [int(r["owner_c"] > 0) for r in rr]
    res = {"n": len(rr), "brier": brier(p, y), "auc": auc(p, y),
           "agree_at_0.5": round(float(np.mean([(a >= 0.5) == bool(b) for a, b in zip(p, y)])), 4) if rr else None}
    cr = [r for r in rr if r.get(f"{key}_c") is not None]
    if cr:
        conf = confusion([r["owner_c"] for r in cr], [r[f"{key}_c"] for r in cr])
        res["columns"] = {"n": len(cr), "confusion": conf,
                          "agree": sum(v for k, v in conf.items() if k.split("->")[0] == k.split("->")[1])}
    return res


# ---------------------------------------------------------------- evaluation

def evaluate_site(site_id: str, out: Path = OUT, ref: str | None = None, occ_cfg: dict | None = None, log=print) -> dict:
    from .grid import Grid
    from .osm_mask import MASK_DIR, load_mask
    from .run import load_sites
    from . import s2_labels
    site = next(s for s in load_sites() if s["id"] == site_id)
    grid = Grid.for_site(site)
    mask, _ = load_mask(site_id, int(site.get("mask_version", 1)), MASK_DIR)
    lines = quay_lines.load(site_id, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), MASK_DIR)
    geom = quay.Geometry(grid, mask, lines)
    secs = {s["section_id"]: s for s in s2_labels.sections(geom)}
    rv = json.loads((out / "labels" / site_id / "review.json").read_text(encoding="utf-8"))
    rv_secs = {s["id"]: {"way_id": s["way"], "side": s["side"], "line": s["line"]} for s in rv["sections"]}
    day_t = {d["date"]: d["acquired_utc"] for d in rv["days"]}
    answers = [a for a in read_answers(site_id, out, ref) if a.get("verified") == "1" and a["section_id"] in secs]
    scenes = [s for s in display_scenes(site_id, out) if (s["chip_px"][0], s["chip_px"][1]) == (grid.width, grid.height)]
    est = json.loads((out / "quay_est2" / f"{site_id}.json").read_text(encoding="utf-8")) if (out / "quay_est2" / f"{site_id}.json").exists() else {"scenes": []}
    est_by = {s["acquired_utc"]: s for s in est["scenes"]}
    feats_by: dict = {}
    for f in e2.load_features(out):
        if f["site_id"] == site_id:
            feats_by.setdefault(f["acquired_utc"], []).append(f)
    lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 18.2)

    def pair(date):
        t = day_t.get(date) or f"{date}T03:00:00Z"
        best = None
        for s in scenes:
            dt = (e2._t(s["acquired_utc"]) - e2._t(t)).total_seconds() / 3600.0
            if abs(dt) <= e2.S2_MAX_H and (best is None or abs(dt) < abs(best[0])):
                best = (dt, s)
        return best

    cache: dict = {}
    rows = []
    for a in answers:
        pr = pair(a["date"])
        if pr is None:
            continue
        dt, sc = pr
        key = sc["acquired_utc"]
        if key not in cache:
            bl = fbase.load(site, sc["orbit"])
            if bl is None:
                cache[key] = None
                continue
            layers, meta = bl
            vv, vh = decode_display(sc["image"].read_bytes(), sc["chip_px"])
            pv = occ_test.pvalues(vv, vh, layers, occ_cfg, n_scenes=int(meta.get("scenes") or 24), presmoothed=True)
            occ = occ_test.section_occupancy(geom, list(secs.values()), pv["p"], occ_cfg)
            samples: list = []
            quay.count(geom, vv, vh, layers, presmoothed=True, samples_out=samples)
            qocc = {}
            for sid, s in secs.items():
                hit = np.asarray(samples[s["profile"]]["inner_hit"], bool)
                runs = quay.runs(hit, geom.cfg["step_m"], geom.cfg["max_gap_m"], geom.cfg["min_run_m"], False)
                idx = set(int(i) for i in s["idx"])
                qocc[sid] = int(any(idx & set(range(r0, r1)) for r0, r1 in runs))
            cache[key] = {"occ": occ, "quay": qocc}
        got = cache[key]
        if got is None:
            continue
        y = int(a["verified_c"] or 0) if a.get("verified_occupied") == "1" else 0
        row = {"section_id": a["section_id"], "date": a["date"], "s1": key, "dt_h": round(dt, 1), "owner_c": y}
        o = got["occ"].get(a["section_id"]) or {}
        row["occ1_p"], row["occ1_c"], row["occ1_llr"] = o.get("p_occ"), o.get("c"), o.get("llr")
        row["quay_webp_p"] = got["quay"].get(a["section_id"])
        es = est_by.get(key)
        if es is not None:
            feats = feats_by.get(key, [])
            by_n = {f.get("n"): f for f in feats}
            pairs = [(by_n[s_["n"]], s_) for s_ in es["stretches"] if s_.get("n") in by_n]
            st = stretch_of_section(rv_secs[a["section_id"]], [f_ for f_, _ in pairs], [s_ for _, s_ in pairs], lat0)
            p = p4(st)
            row["est2_p"], row["est2_c"] = round(1.0 - float(p[0]), 4), int(np.argmax(p))
        rows.append(row)
    return {"site_id": site_id, "answers": len(answers), "days_answered": len({a["date"] for a in answers}),
            "scenes_at_hand": [s["acquired_utc"] for s in scenes], "rows_paired": len(rows),
            "days_paired": sorted({r["date"] for r in rows}),
            "metrics": {k: summary(rows, k) for k in ("est2", "quay_webp", "occ1")}, "rows": rows}


def plan(out: Path = OUT) -> dict:
    """PU to evaluate on raw sigma0 (nothing fetched): the S1 scenes of the answered days (labels/<site>/
    s1_recount_plan.json) and a longer baseline with the low percentiles (bl-2: one year, ~36 scenes of orbit 157)."""
    from .grid import Grid
    from .run import load_sites
    res, tot = {}, 0.0
    for sid in SITES:
        site = next(s for s in load_sites() if s["id"] == sid)
        g = Grid.for_site(site)
        p = out / "labels" / sid / "s1_recount_plan.json"
        rc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        orbits = (site.get("baseline") or {}).get("orbits") or []
        per = cdse.estimate_pu(g.width, g.height, 2)
        bl2 = round(per * 36 * len(orbits), 2)
        res[sid] = {"scenes_pu": rc.get("pu_total"), "scenes": len(rc.get("scenes", [])), "baseline_bl2_pu": bl2,
                    "baseline_bl2_note": f"{len(orbits)} 軌道 × 36 撮影 × {per} PU（VV・VH、1 年）"}
        tot += float(rc.get("pu_total") or 0) + bl2
    res["total_pu"] = round(tot, 2)
    return res


def evaluate(out: Path = OUT, ref: str | None = None, occ_cfg: dict | None = None) -> dict:
    sites = {sid: evaluate_site(sid, out, ref, occ_cfg) for sid in SITES}
    pooled_rows = [r for s in sites.values() for r in s["rows"]]
    return {"version": EVAL_VERSION, "occ": occ_test.settings() | (occ_cfg or {}),
            "input": "表示用の WebP（chips、Lee 3×3・8 bit・非可逆）を dB に戻した画素。est2 は毎日の実行の生の σ0 から",
            "pooled": {"rows": len(pooled_rows), "metrics": {k: summary(pooled_rows, k) for k in ("est2", "quay_webp", "occ1")}},
            "sites": sites, "plan": plan(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="current quay occupancy vs occ-1 on the owner's labels (no fetch)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--answers-ref", help="read the answers CSVs from this git ref")
    ap.add_argument("--json", type=Path, help="write the full result here")
    a = ap.parse_args(argv)
    res = evaluate(a.out, a.answers_ref)
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    brief = {"pooled": res["pooled"], "plan": res["plan"],
             "sites": {k: {kk: v[kk] for kk in ("answers", "days_answered", "rows_paired", "days_paired", "metrics")} for k, v in res["sites"].items()}}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
