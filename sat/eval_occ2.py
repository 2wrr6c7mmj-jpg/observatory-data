"""occ-2 on the re-count scenes: the rows for the evaluation and the in-sample numbers against the current judge (PU 0).

Supersedes sat/eval_occ.py (occ-1 on the display images). The inputs are the raw VV / VH of the 19 re-count scenes
(probe-raw/s1-recount/<site>/*.npz on the probe branch, read with ``git show``: Gu Zhenkou 8, Yalong Bay 7, Yulin 4), the
sites' bl-1 baselines and quay lines. For every scene sat/occ2.run_scene gives one row per naval berth section
(sat/occ2_schema.FIELDS); the rows are written to docs/data/sat/quay_occ2/<site>/YYYY-MM.csv and the per-scene pool
statistics to docs/data/sat/quay_occ2.json, where sat/eval_presence reads them (judge occ-2, eval-presence-3). Zhanjiang
is not run (sites.json quay.occ2 is set in occ2-3; this script takes the three ports).

Before the calibration (occ2-4) p_occ is the placeholder a = logit(prior), b = 1, c = 0 with the prior = the owner's base
rate of "present" over the site's answers, and the blind prior of a section = (present + 1) / (answers + 2) of the owner's
answers on it (both taken from the labels: these numbers are in sample, 設計書 §5.3; calib_source prior-own-labels). The
rows written also carry the section reference of occ2-2b: the owner's "absent" days of a section (s2-s1-match.csv
s2_occupied 0 within 48 h, or s1-answers.csv when it exists), the scene itself left out, shift the null's centre
(sat/occ2.section_offsets / apply_section_reference; ref_kind section, calib_source ...;ref-own-absent-days). The occupation
probability (p_occ) and the test (p_test < 0.05) are reported apart.

Output docs/data/sat/eval/occ2.json: per site and over all, for the s2 truth (and the s1 truth where answers exist), the
occ-2 metrics beside the current judge on the same cells (eval_presence.occ2_metrics) for the variants: min2 (the rows
written), noprior (nothing from the labels: a anchored at p_test 0.05, pool reference only), 2b (noprior + the section
reference), w15 (§8 depth variant), noridge (§3.1 off), and the scene records. The tables of the PR body are printed to
stdout.

    python -m sat.eval_occ2 [--sites QGZ,SLP,SYL] [--raw-ref origin/probe/sat-s1-recount-ops4 | --raw DIR] [--out docs/data/sat]
                            [--no-write]   # metrics only, nothing written
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import baseline as fbase, eval_presence as ep, occ2, occ2_schema as o2s, s2_labels
from .grid import Grid
from .osm_mask import MASK_DIR, load_mask
from .run import QuayContext, in_sample, load_sites, sector_of, site_sectors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
EVAL_VERSION = "eval-occ2-1"
RAW_REF = "origin/probe/sat-s1-recount-ops4"
RAW_SUBDIR = "probe-raw/s1-recount"
SITES = ("QGZ", "SLP", "SYL")
# the variants run: min2 (two adjacent depth pixels; the owner's answers give the prior and the blind priors, §1.3-5 / §2.3 as
# designed: in sample; plus the section reference of occ2-2b: the rows written), w15 (the 15 m depth of §8, same priors), noprior
# (nothing from the labels: a anchored so that p_occ = 0.5 at p_test = TEST_ALPHA, blind prior 0.5, pool reference only) and
# noridge (noprior without the ridge edge, §3.1 off). Derived without a new run: 2b = noprior + the section reference.
NO_LABEL_CALIB = {"a": __import__("math").log10(ep.TEST_ALPHA), "b": 1.0, "c": 0.0, "source": "none"}
VARIANTS = {"min2": {"cfg": {"depth_mode": "min2"}, "labels": True, "section_ref": True},
            "w15": {"cfg": {"depth_mode": "w15"}, "labels": True, "section_ref": True},
            "noprior": {"cfg": {"depth_mode": "min2"}, "labels": False, "section_ref": False},
            "noridge": {"cfg": {"depth_mode": "min2", "ridge_rel_db": 99.0}, "labels": False, "section_ref": False}}
DERIVED = {"2b": ("noprior", False), "2c": ("noprior", True)}   # (base variant, section reference on the blind sections only)
DEFAULT = "min2"
ABSENT_MAX_H = 48.0


# ---------------------------------------------------------------- inputs
def raw_scenes(site_id: str, ref: str | None, raw: Path | None, tmp: Path) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """[(meta, vv, vh)] of the saved scenes of a site: from ``raw``/<site>/*.npz, or from the git ref's probe-raw files."""
    out = []
    if raw is not None:
        files = sorted((raw / site_id).glob("s1-*.npz"))
        for f in files:
            z = np.load(f)
            meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
            out.append((meta, z["vv"].astype(np.float64), z["vh"].astype(np.float64)))
        return out
    ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", f"{RAW_SUBDIR}/{site_id}/"], cwd=ROOT, capture_output=True, text=True)
    if ls.returncode != 0:
        raise SystemExit(f"{site_id}: git ls-tree {ref} に失敗: {ls.stderr.strip()}")
    for rel in sorted(l for l in ls.stdout.split() if l.endswith(".npz")):
        local = tmp / Path(rel).name
        with local.open("wb") as fh:
            subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, stdout=fh, check=True)
        meta_txt = subprocess.run(["git", "show", f"{ref}:{rel[:-4]}.json"], cwd=ROOT, capture_output=True, text=True)
        meta = json.loads(meta_txt.stdout) if meta_txt.returncode == 0 else {}
        z = np.load(local)
        out.append((meta, z["vv"].astype(np.float64), z["vh"].astype(np.float64)))
    return out


def absent_days(tables: dict) -> dict:
    """{section_id: set of S1 acquired_utc the owner called absent}: from <site>-s1-answers.csv (verified 1, occupied 0) when it has
    rows, else from <site>-s2-s1-match.csv (s2_occupied 0 and |dt_h| <= ABSENT_MAX_H). ``tables``: {file name: rows}."""
    out: dict = {}
    s1 = [r for k, rs in tables.items() if k.endswith("s1-answers.csv") for r in rs]
    if s1:
        for r in s1:
            if r.get("verified") == "1" and r.get("verified_occupied") == "0" and r.get("s1_acquired_utc"):
                out.setdefault(r["section_id"], set()).add(r["s1_acquired_utc"])
        return out
    for k, rs in tables.items():
        if not k.endswith("s2-s1-match.csv"):
            continue
        for r in rs:
            try:
                dt = abs(float(r.get("dt_h") or "nan"))
            except ValueError:
                continue
            if r.get("s2_occupied") == "0" and dt <= ABSENT_MAX_H and r.get("s1_acquired_utc"):
                out.setdefault(r["section_id"], set()).add(r["s1_acquired_utc"])
    return out


def load_absent(out: Path, site_id: str) -> dict:
    names = (f"{site_id}-s1-answers.csv", f"{site_id}-s2-s1-match.csv")
    return absent_days({n: ep.read_csv(out / "labels" / n) for n in names})


def owner_priors(out: Path, site_id: str, eligible_ids: set) -> tuple[dict, float]:
    """({section_id: (present + 1) / (answers + 2)}, base rate over the eligible sections' answers) from the owner's labels."""
    lab = ep.labels(out, site_id)
    by: dict = {}
    for a in lab:
        if a["section_id"] in eligible_ids:
            by.setdefault(a["section_id"], []).append(a["y"])
    ys = [y for v in by.values() for y in v]
    prior = {k: (sum(v) + 1.0) / (len(v) + 2.0) for k, v in by.items()}
    return prior, (sum(ys) / len(ys)) if ys else 0.5


# ---------------------------------------------------------------- one site
def run_site(site: dict, out: Path, scenes: list, variants=VARIANTS, derived=DERIVED, log=print) -> tuple[dict, list[dict], dict]:
    """({variant: [rows]} (the derived variants included), [scene records of the default variant], {prior, blind_prior, ...})."""
    sid = site["id"]
    grid = Grid.for_site(site)
    mask, _ = load_mask(sid, int(site.get("mask_version", 1)), MASK_DIR)
    ctx = QuayContext(site, grid, mask)
    geom = ctx.geometry()
    sectors = site_sectors(site)
    secs = s2_labels.sections(geom, lambda lo, la: sector_of(sectors, lo, la))
    elig = occ2.eligible_naval(ctx.berths(True))
    elig_ids = {s["section_id"] for s in secs if elig(s)}
    blind_prior, base = owner_priors(out, sid, elig_ids)
    absent = load_absent(out, sid)
    rows: dict = {v: [] for v in variants}
    recs = []
    recs_by: dict = {v: {} for v in variants}
    calibs: dict = {}
    for meta, vv, vh in scenes:
        if vv.shape != (grid.height, grid.width):
            log(f"{sid} {meta.get('acquired_utc')}: 大きさ {vv.shape} が範囲 {(grid.height, grid.width)} と合わない、飛ばす")
            continue
        bl = ctx.baseline(meta.get("relative_orbit"))
        if bl is None:
            log(f"{sid} {meta.get('acquired_utc')}: 軌道 {meta.get('relative_orbit')} の基準が無い、飛ばす")
            continue
        layers, bmeta = bl
        acq = {"site_id": sid, "acquired_utc": meta["acquired_utc"], "relative_orbit": int(meta["relative_orbit"]),
               "orbit_direction": meta.get("orbit_direction") or (bmeta.get("scene_list") or [{}])[0].get("orbit_direction")}
        period = in_sample(site, acq["acquired_utc"], bmeta)
        svv, svh = fbase.smooth_db(vv), fbase.smooth_db(vh)
        for v, spec in variants.items():
            use_labels = spec["labels"]
            calibs[v] = {"prior": base, "source": "prior-own-labels"} if use_labels else NO_LABEL_CALIB
            r, rec = occ2.run_scene(geom, mask, secs, layers, svv, svh, acq, lines=ctx.lines, cfg=spec["cfg"], eligible=elig,
                                    blind_prior=blind_prior if use_labels else {}, presmoothed=True, calib=calibs[v], baseline_period=period)
            rows[v].extend(r)
            recs_by[v][acq["acquired_utc"]] = rec
            if v == DEFAULT:
                recs.append({"depth_mode": spec["cfg"].get("depth_mode", "min2"), "baseline_period": period,
                             **{k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items() if k != "t_pool"}})
        log(f"{sid} {acq['acquired_utc']} ro{acq['relative_orbit']}: 区間 {recs[-1]['sections']} pool {recs[-1]['pool_n']} 盲域 {recs[-1]['blind_sections']} "
            f"尾根 {recs[-1]['ridge_sections']} Δ {recs[-1].get('delta_vv_db')}／{recs[-1].get('delta_vh_db')} σ {recs[-1].get('sigma_vv_db')}／{recs[-1].get('sigma_vh_db')} "
            f"ENL {recs[-1]['enl_vv']}／{recs[-1]['enl_vh']} T 帰無 {recs[-1].get('t_centre')}±{recs[-1].get('t_sigma')} {recs[-1].get('status')}")
    # the section reference (occ2-2b): the derived variants first (from the base variant's rows before its own shift)
    for name, (base_v, _only) in derived.items():
        rows[name] = [dict(r) for r in rows[base_v]]
        recs_by[name] = recs_by[base_v]
        calibs[name] = calibs[base_v]
    n_ref = {}
    for v in list(rows):
        spec = variants.get(v)
        if (spec is not None and not spec["section_ref"]) or not rows[v]:
            continue
        by_scene: dict = {}
        for r in rows[v]:
            by_scene.setdefault(r["acquired_utc"], []).append(r)
        offs = occ2.section_offsets(by_scene, recs_by[v], absent)
        base_cfg = (spec or variants[derived[v][0]])["cfg"]
        n_ref[v] = occ2.apply_section_reference(by_scene, recs_by[v], offs, cfg=base_cfg, calib=calibs[v],
                                                blind_prior=blind_prior if calibs[v].get("source") == "prior-own-labels" else {},
                                                only_blind=bool(spec is None and derived[v][1]))
    log(f"{sid}: 不在の日のある区間 {sum(1 for v in absent.values() if len(v) >= 2)}（答え {sum(len(v) for v in absent.values())}）、区間別参照を当てた升 {n_ref}")
    ridge = sorted({(r.get("relative_orbit"), r.get("ridge_sections")) for r in recs})
    info = {"prior": round(base, 3), "blind_prior": {k: round(v, 3) for k, v in sorted(blind_prior.items())}, "sections_eligible": len(elig_ids),
            "absent_days": {k: sorted(v) for k, v in sorted(absent.items())}, "section_ref_cells": n_ref,
            "ridge_sections_by_orbit": [{"relative_orbit": o, "ridge_sections": n} for o, n in ridge]}
    return rows, recs, info


# ---------------------------------------------------------------- the comparison
def with_variant(rows: list[dict], var_rows: list[dict]) -> list[dict]:
    """The evaluation rows with p_occ2 / p_test_occ2 / occ2_blind taken from another variant's occ-2 rows (same cells)."""
    by = {(r["acquired_utc"], r["section_id"]): r for r in var_rows}
    out = []
    for r in rows:
        o = by.get((r["s1"], r["section_id"]))
        if o is None or o.get("p_occ") is None:
            continue
        out.append({**r, "p_occ2": round(float(o["p_occ"]), 3), "p_test_occ2": o["p_test"], "occ2_blind": o["blind"]})
    return out


def site_block(ev: dict, var_rows: dict) -> dict:
    rows = ev["rows"]
    blk = {"n_rows": len(rows), "occ2": ep.occ2_metrics(rows), "scenes": sorted({r["s1"] for r in rows if r.get("p_occ2") is not None}),
           "variants": {v: ep.occ2_metrics(with_variant(rows, var_rows[v])) for v in var_rows}}
    return blk


def fmt(x, nd=3):
    if x is None:
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def md_tables(res: dict) -> str:
    lines = []
    lines.append("### 現行（est-2／s1-recount）と occ-2（較正前の仮置き）、同じ升（s2 の正解）\n")
    lines.append("| 港 | 升 | 現行 再現率／適合率／ブライア／AUC | occ-2 p_occ 再現率／適合率／ブライア／AUC | occ-2 検定 p<0.05 再現率／適合率 | 48 h 以内 升 | 48 h 現行 再現率／適合率 | 48 h occ-2 再現率／適合率／AUC | 盲域 升 | 盲域 再現率 現行→occ-2 | A-1 除外 occ-2 再現率／適合率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sid, b in list(res["sites"].items()) + [("ALL", res["total"])]:
        m = b["s2"]["occ2"] if sid != "ALL" else b["s2"]["occ2"]
        c, o, t, nr = m["current"], m["p_occ"], m["test"], m["near"]
        bl = m["by_blind"]["blind"]
        ex = m["excl_a1"]
        name = ep.NAMES.get(sid, sid) if sid != "ALL" else "全港"
        lines.append(f"| {name} | {m['n']} | {fmt(c['recall'])}／{fmt(c['precision'])}／{fmt(c['brier'])}／{fmt(c['auc'])} "
                     f"| {fmt(o['recall'])}／{fmt(o['precision'])}／{fmt(o['brier'])}／{fmt(o['auc'])} | {fmt(t['recall'])}／{fmt(t['precision'])} "
                     f"| {nr['p_occ']['n']} | {fmt(nr['current']['recall'])}／{fmt(nr['current']['precision'])} "
                     f"| {fmt(nr['p_occ']['recall'])}／{fmt(nr['p_occ']['precision'])}／{fmt(nr['p_occ']['auc'])} | {bl['p_occ']['n']} "
                     f"| {fmt(bl['current']['recall'])} → {fmt(bl['p_occ']['recall'])} | {fmt(ex['p_occ']['recall'])}／{fmt(ex['p_occ']['precision'])} |")
    for truth, title in (("s2", "s2 の正解（S2 の日の答えを直近の S1 に対応、dt_h あり）"), ("s1", "s1 の正解（S1 の撮影そのものの答え、|dt|＝0。古鎮口のみ）")):
        blocks = [(sid, b) for sid, b in res["sites"].items() if truth in b] + ([("ALL", res["total"])] if truth in res["total"] else [])
        if not blocks:
            continue
        lines.append(f"\n### {title}：正解を事前に使わない occ-2（a＝log10({ep.TEST_ALPHA:g})：p_test＝{ep.TEST_ALPHA:g} で p_occ＝0.5、盲域の事前 0.5）、pool の参照だけ（noprior）、区間別の空の参照（2b：全区間、2c：盲域だけ）、現行と同じ升\n")
        lines.append("| 港 | 升 | 判定 | 全体 再現率／適合率／ブライア／AUC | 検定 p<0.05 再現率／適合率／AUC(p_test) | 48 h 升 | 48 h 再現率／適合率／AUC | 盲域 升 | 盲域 再現率／適合率 | 非盲域 升 | 非盲域 再現率／適合率／AUC | A-1 除外 再現率／適合率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for sid, b in blocks:
            name = ep.NAMES.get(sid, sid) if sid != "ALL" else "全港"
            for label, var, key in (("現行", "noprior", "current"), ("noprior", "noprior", "p_occ"), ("2b", "2b", "p_occ"), ("2c", "2c", "p_occ")):
                m = b[truth]["variants"][var]
                o, nr, bl, cl, ex = m[key], m["near"][key], m["by_blind"]["blind"][key], m["by_blind"]["clear"][key], m["excl_a1"][key]
                t = m["test"]
                test_s = "–" if key == "current" else f"{fmt(t['recall'])}／{fmt(t['precision'])}／{fmt(t['auc'])}"
                lines.append(f"| {name} | {m['n']} | {label} | {fmt(o['recall'])}／{fmt(o['precision'])}／{fmt(o['brier'])}／{fmt(o['auc'])} "
                             f"| {test_s} "
                             f"| {nr['n']} | {fmt(nr['recall'])}／{fmt(nr['precision'])}／{fmt(nr['auc'])} | {bl['n']} | {fmt(bl['recall'])}／{fmt(bl['precision'])} "
                             f"| {cl['n']} | {fmt(cl['recall'])}／{fmt(cl['precision'])}／{fmt(cl['auc'])} | {fmt(ex['recall'])}／{fmt(ex['precision'])} |")
    lines.append("\n### 尾根の縁（§3.1）あり／なし（noprior、同じ升）\n")
    lines.append("| 港 | 升 | 尾根あり 検定 AUC(p_test) | なし | 尾根あり p<0.05 再現率／適合率 | なし | 尾根で帯を動かした区間（軌道別、全撮影で同じ） |")
    lines.append("|---|---|---|---|---|---|---|")
    for sid, b in list(res["sites"].items()) + [("ALL", res["total"])]:
        d = b["s2"]["variants"]
        a, w = d["noprior"], d["noridge"]
        name = ep.NAMES.get(sid, sid) if sid != "ALL" else "全港"
        nr = "–" if sid == "ALL" else "、".join(f"ro{x['relative_orbit']} {x['ridge_sections']}" for x in res["records"][sid].get("ridge_sections_by_orbit", [])) + f" of {res['records'][sid]['scenes'][0]['sections']}"
        lines.append(f"| {name} | {a['n']} | {fmt(a['test']['auc'])} | {fmt(w['test']['auc'])} | {fmt(a['test']['recall'])}／{fmt(a['test']['precision'])} "
                     f"| {fmt(w['test']['recall'])}／{fmt(w['test']['precision'])} | {nr} |")
    lines.append("\n### 奥行きの案（§8）：2 画素の対（min2、既定）と 15 m 案（w15）\n")
    lines.append("| 港 | 升 | min2 検定 AUC(p_test) | w15 検定 AUC(p_test) | min2 p<0.05 再現率／適合率 | w15 p<0.05 再現率／適合率 | min2 ブライア | w15 ブライア |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sid, b in list(res["sites"].items()) + [("ALL", res["total"])]:
        d = b["s2"]["variants"]
        a, w = d["min2"], d["w15"]
        name = ep.NAMES.get(sid, sid) if sid != "ALL" else "全港"
        lines.append(f"| {name} | {a['n']} | {fmt(a['test']['auc'])} | {fmt(w['test']['auc'])} | {fmt(a['test']['recall'])}／{fmt(a['test']['precision'])} "
                     f"| {fmt(w['test']['recall'])}／{fmt(w['test']['precision'])} | {fmt(a['p_occ']['brier'])} | {fmt(w['p_occ']['brier'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="occ-2 on the re-count scenes: rows for the evaluation and the in-sample numbers (PU 0)")
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--raw-ref", default=RAW_REF, help="git ref holding probe-raw/s1-recount/<site>/*.npz")
    ap.add_argument("--raw", type=Path, default=None, help="a directory with <site>/*.npz instead of the git ref")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true", help="metrics only: no quay_occ2 rows, no eval/occ2.json")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    ids = [s.strip() for s in a.sites.split(",") if s.strip()]
    sites = {s["id"]: s for s in load_sites()}
    log = lambda m: print(m, file=sys.stderr)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = {"version": EVAL_VERSION, "generated_utc": now, "occ_version": o2s.OCC_VERSION, "raw_ref": None if a.raw else a.raw_ref,
           "settings": occ2.settings(), "sites": {}, "total": {}}
    all_rows_s2, all_var_s2, all_rows_s1 = [], {v: [] for v in list(VARIANTS) + list(DERIVED)}, []
    recs_all = {}
    with tempfile.TemporaryDirectory() as td:
        for sid in ids:
            site = sites[sid]
            scenes = raw_scenes(sid, a.raw_ref, a.raw, Path(td))
            log(f"{sid}: 撮影 {len(scenes)}")
            rows, recs, info = run_site(site, a.out, scenes, log=log)
            recs_all[sid] = {"scenes": recs, **info}
            if not a.no_write:
                o2s.save_rows(a.out, sid, rows[DEFAULT])
            ev = ep.evaluate_site(a.out, site, truth="s2")
            if a.no_write:   # nothing written: join the rows in memory
                ev["rows"] = with_variant(ev["rows"], rows[DEFAULT])
            blk = {"s2": site_block(ev, rows), "prior": info["prior"], "sections_eligible": info["sections_eligible"],
                   "scenes_run": [r["acquired_utc"] for r in recs]}
            if (a.out / "labels" / f"{sid}-s1-answers.csv").exists():
                ev1 = ep.evaluate_site(a.out, site, truth="s1")
                if a.no_write:
                    ev1["rows"] = with_variant(ev1["rows"], rows[DEFAULT])
                blk["s1"] = site_block(ev1, rows)
                all_rows_s1.extend(ev1["rows"])
            res["sites"][sid] = blk
            all_rows_s2.extend(ev["rows"])
            for v in all_var_s2:
                all_var_s2[v].extend(rows[v])
    res["total"] = {"s2": site_block({"rows": all_rows_s2}, all_var_s2)}
    if all_rows_s1:
        res["total"]["s1"] = site_block({"rows": all_rows_s1}, all_var_s2)
    res["records"] = recs_all
    res["notes"] = [
        "s1 の正解（labels/<港>-s1-answers.csv、S1 の撮影そのものの答え）がある港は、区間別の空の参照の不在の日もその答えから取り（|dt|＝0）、s1 のブロックに同じ指標を出す（古鎮口 192 升、2026-11-08）",
        "較正前：p_occ は a=logit(事前：その港の所有者の在の割合)・b=1・c=0 の仮置き、盲域は所有者の答えの在の割合（(在+1)/(答え+2)）を blind_prior に λ=0.5 で混合。"
        "どちらも正解から取った値なので、ここの数字は in-sample（較正に使う標本の数字）",
        "升は eval_presence と同じ（確認セットの区間×ラベル日の S1、naval・berth・50 m 以上、1 撮影 1 ラベル日）。current は同じ升の現行の判定（est-2／s1-recount）",
        "test は検定だけ（p_test < 0.05 を在）、p_occ は p_occ ≥ 0.5 を在。AUC は順位だけで決まるので p_occ と p_test で同じ（盲域の混合の分だけ違う）",
        "variants：min2＝奥行き 5〜25 m（toward は 5〜35 m）の隣り合う 2 画素の対の min の最大＋区間別の空の参照（既定、quay_occ2 の行。事前と盲域の事前は所有者の答え）、"
        "w15＝1 画素目と 2 画素目の 0.5／0.5 加重（同じ事前・参照）、"
        f"noprior＝正解を事前に使わず pool の参照だけ（a＝log10({ep.TEST_ALPHA:g})：p_test＝{ep.TEST_ALPHA:g} で p_occ＝0.5、盲域の事前 0.5）、2b＝noprior＋区間別の空の参照（所有者の不在の日、当該撮影は除く、全区間）、"
        "2c＝区間別の参照を盲域にだけ当て（無い盲域は blind_prior のみ）、非盲域は pool の参照のまま（種の要求は共通）、"
        "noridge＝noprior で尾根の縁を切る。test の AUC は p_test の順位（連続）、p_occ の AUC は p_occ の順位",
        "盲域：区間別の参照（不在の日 2 以上）があるときだけ検定し、無ければ p_test 空欄・p_occ＝盲域の事前（不在とは出さない）。設計 §2.3 の空間の p90 を参照にする検定は 1部の審査で棄却"
        "（盲域の定義と参照が同じ量なので、空でも構造的に在になる：#87 の初版では盲域 56 升が tp29 fp14 fn12 tn1、AUC(p_test) 0.49）",
        "種（seed）は検定の要求：窓に z_seed 以上の標本が無い升は p_test を 0.1 で止める（z_seed_ok 0）",
        "設計の仮置き a＝logit(事前) では −log10 p_test ≥ 0 なので p_occ ≥ 事前 が常に成り立ち、事前 ≥ 0.5 の港では全升が「在」になる（min2 の p_occ の再現率 1.0 はそれ）。"
        "較正（occ2-4）までは検定の p と min2_noprior の p_occ を読む",
    ]
    res["credit"] = "Contains modified Copernicus Sentinel data 2025–2026"
    if not a.no_write:
        qj = a.out / "quay_occ2.json"
        old = json.loads(qj.read_text(encoding="utf-8")) if qj.exists() else {}
        sha = subprocess.run(["git", "rev-parse", "--short", a.raw_ref], cwd=ROOT, capture_output=True, text=True).stdout.strip() if not a.raw else None
        doc = {"version": o2s.OCC_VERSION, "generated_utc": now, "settings": occ2.settings(),
               "note": "occ-2（岸壁の在不在の統計検定）の撮影ごとの帰無分布の記録：pool（非盲域の海軍区間）の Δ・σ、沖の水面の ENL と持ち上がり、T の帰無の中央値と幅、種の z。"
                       "行は quay_occ2/<港>/YYYY-MM.csv。取り直しの撮影は sat/eval_occ2 が書く。湛江には書かない",
               "calibration": "較正前。行の p_occ は仮置き（a＝logit(所有者の在の割合)・b＝1・c＝0、盲域は所有者の答えの在の割合と λ＝0.5 で混合、区間別の空の参照も所有者の不在の日から：calib_source の prior-own-labels／ref-own-absent-days）。"
                              "p_occ は判定に使えない（事前 ≥ 0.5 なら p_test ≤ 0.5 で全升が在）。検定の p_test は事前を使わない。較正は occ2-4",
               "raw_source": {"ref": None if a.raw else a.raw_ref, "commit": sha, "path": f"{RAW_SUBDIR}/<site>/s1-<acquired>.npz",
                              "processing_units": 0, "note": "取り直しの 19 撮影（s1-recount-1、ops-4）の生データを読むだけで新しい取得は無い"},
               "columns": {"baseline_period": "撮影が基準期間の中か（sat/run.in_sample：tuning＝閾値の調整に使った撮影、baseline＝基準の撮影、空＝どちらでもない）。較正の標本内外（label_freeze_utc）とは別",
                           "ref_kind": "time＝画素の自分の p10、section＝自分の p10＋区間の不在の日で中心をずらす、prior＝検定なし（盲域、p_occ＝blind_prior）、spatial＝設計 §2.3 の空間の p90（選択肢、既定では使わない）",
                           "calib_source": "p_occ と参照の出所：none＝正解を使わない、prior-own-labels＝事前と盲域の事前が所有者の答え、ref-own-absent-days＝区間別の参照が所有者の不在の日"},
               "sites": {**(old.get("sites") or {}), **recs_all}, "credit": res["credit"]}
        qj.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (a.out / "eval").mkdir(parents=True, exist_ok=True)
        (a.out / "eval" / "occ2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(md_tables(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
