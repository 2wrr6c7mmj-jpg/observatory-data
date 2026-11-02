"""Rows abreast from the cross-section (cols-1) on the owner's labels of SYL and SLP (no fetch, PU 0).

The rows are the labelled sections the owner saw occupied (verified_c 1-3), paired with a Sentinel-1 scene within
96 h (as sat/eval_occ.py: what is at hand are the public display images, orbit 157 ascending). For each row:

* ``xsec``      the protrusion width W from the display image (sat/cols_xsec.section_width) and c from the
                calibrated model, calibrated leave-one-day-out (the day's rows never in its own fit), per direction;
* ``xsec_raw``  the same model on the occupied depth of the stored quay feature (est-1 ``depth_m``, from the raw
                sigma0 of the daily run) where a stretch lies on the section;
* ``est2`` / ``occ1``  c of the published est-2 and of occ-1 (sat/eval_occ.py) on the same rows;
* ``xsec_raw_zjgcal``  the raw-depth model fitted on Zhanjiang's verified labels (labels/ZJG-s2-labels.csv with the
                paired raw depth of labels/ZJG-s2-s1-match.csv; another port, so out of sample for SYL and SLP);
* ``prior``     always c = 1 (the most common).

``zjg_raw_lodo``: the same cross-section model on Zhanjiang's labels themselves, leave one day out (more rows, with
c = 2 and 3). Every scene at hand is ascending (SYL and SLP orbit 157; Zhanjiang 157 and 84): the descending model
stays the default until descending scenes and a descending baseline exist.

Output: the confusion matrices owner c -> c, the agreement and the 3-class Brier score of the posterior. With
``--write-calib`` the raw-depth model fitted on Zhanjiang's labels is written to sat/cols_xsec_calib.json (per
direction; a direction without labels keeps the default).

    python -m sat.eval_cols [--answers-ref origin/hq/labels-syl-slp] [--write-calib] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import baseline as fbase, cols_xsec as cx, eval_occ, quay, quay_est2 as e2, quay_lines
from .eval_est2 import stretch_of_section

OUT = eval_occ.OUT
EVAL_VERSION = "eval-cols-1"


def rows_for_site(site_id: str, out: Path = OUT, ref: str | None = None) -> list[dict]:
    from .grid import Grid
    from .osm_mask import MASK_DIR, load_mask
    from .run import load_sites
    from . import s2_labels
    base_rows = {(r["section_id"], r["date"]): r for r in eval_occ.evaluate_site(site_id, out, ref)["rows"]}
    site = next(s for s in load_sites() if s["id"] == site_id)
    grid = Grid.for_site(site)
    mask, _ = load_mask(site_id, int(site.get("mask_version", 1)), MASK_DIR)
    lines = quay_lines.load(site_id, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), MASK_DIR)
    geom = quay.Geometry(grid, mask, lines)
    secs = {s["section_id"]: s for s in s2_labels.sections(geom)}
    rv = json.loads((out / "labels" / site_id / "review.json").read_text(encoding="utf-8"))
    rv_secs = {s["id"]: {"way_id": s["way"], "side": s["side"], "line": s["line"]} for s in rv["sections"]}
    scenes = {s["acquired_utc"]: s for s in eval_occ.display_scenes(site_id, out)}
    chips = {}
    for p in (out / "chips" / site_id).glob("*.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        chips[c["acquired_utc"]] = c.get("orbit_direction")
    feats_by: dict = {}
    for f in e2.load_features(out):
        if f["site_id"] == site_id:
            feats_by.setdefault(f["acquired_utc"], []).append(f)
    lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 18.2)
    dvv_cache: dict = {}
    rows = []
    for (sid, date), r in base_rows.items():
        if r["owner_c"] < 1:
            continue
        key = r["s1"]
        if key not in dvv_cache:
            sc = scenes[key]
            bl = fbase.load(site, sc["orbit"])
            vv, _vh = eval_occ.decode_display(sc["image"].read_bytes(), sc["chip_px"])
            dvv_cache[key] = vv - bl[0]["vv_p10"] if bl else None
        dvv = dvv_cache[key]
        w = cx.section_width(geom, secs[sid], dvv) if dvv is not None else None
        feats = feats_by.get(key, [])
        f = stretch_of_section(rv_secs[sid], feats, feats, lat0) if feats else None
        rows.append({"site_id": site_id, "section_id": sid, "date": date, "s1": key, "direction": chips.get(key) or "ascending",
                     "owner_c": min(int(r["owner_c"]), 3), "width_m": w["width_m"] if w else None,
                     "raw_depth_m": (float(f["depth_m"]) if f and f.get("depth_m") not in (None, "") else None),
                     "est2_c": r.get("est2_c"), "occ1_c": r.get("occ1_c")})
    return rows


def zjg_raw_rows(out: Path = OUT) -> list[dict]:
    """Zhanjiang's owner-verified sections (labels/ZJG-s2-labels.csv, verified_c 1-3) with the occupied depth of the
    paired S1 scene from the raw sigma0 (labels/ZJG-s2-s1-match.csv s1_depth_m, est-1). Both of Zhanjiang's orbits
    (157, 84) are ascending."""
    lab = {(r["section_id"], r["date"]): r for r in eval_occ_csv(out / "labels" / "ZJG-s2-labels.csv")
           if r.get("verified") == "1" and r.get("verified_occupied") == "1" and (r.get("verified_c") or "") in ("1", "2", "3")}
    rows = []
    for m in eval_occ_csv(out / "labels" / "ZJG-s2-s1-match.csv"):
        r = lab.get((m["section_id"], m["s2_date"]))
        if r is None or not m.get("s1_depth_m"):
            continue
        rows.append({"site_id": "ZJG", "section_id": m["section_id"], "date": m["s2_date"], "direction": "ascending",
                     "relative_orbit": m.get("relative_orbit"), "owner_c": int(r["verified_c"]),
                     "raw_depth_m": float(m["s1_depth_m"])})
    return rows


def eval_occ_csv(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _lodo(rows: list[dict], key: str) -> list[list[float]]:
    """Posterior per row, the model fitted on the other days' rows (per direction)."""
    post = []
    for r in rows:
        train = [x for x in rows if x["date"] != r["date"] and x[key] is not None]
        calib = {}
        for d in cx.DIRECTIONS:
            t = [x for x in train if x["direction"] == d]
            calib[d] = cx.fit([x[key] for x in t], [x["owner_c"] for x in t])
        post.append(cx.posterior({r["direction"]: r[key]}, calib))
    return post


def _summary(y: list[int], c: list[int], p: list[list[float]] | None = None) -> dict:
    conf = eval_occ.confusion(y, c)
    out = {"n": len(y), "confusion": conf, "agree": sum(v for k, v in conf.items() if k.split("->")[0] == k.split("->")[1])}
    if p is not None and y:
        out["brier3"] = round(float(np.mean([((np.array(pp) - np.eye(3)[yy - 1]) ** 2).sum() for pp, yy in zip(p, y)])), 4)
    return out


def evaluate(out: Path = OUT, ref: str | None = None) -> dict:
    rows = [r for sid in eval_occ.SITES for r in rows_for_site(sid, out, ref)]
    y = [r["owner_c"] for r in rows]
    res = {"version": EVAL_VERSION, "cols": cx.settings(), "rows": rows, "n": len(rows),
           "by_direction": {d: sum(r["direction"] == d for r in rows) for d in cx.DIRECTIONS},
           "owner_c": {str(k): y.count(k) for k in (1, 2, 3)}, "methods": {}}
    for key, name in (("width_m", "xsec"), ("raw_depth_m", "xsec_raw")):
        post = _lodo(rows, key)
        res["methods"][name] = {**_summary(y, [int(np.argmax(p)) + 1 for p in post], post),
                                "with_width": sum(r[key] is not None for r in rows)}
    # the raw-depth model fitted on Zhanjiang's labels (another port, out of sample for SYL and SLP)
    z = zjg_raw_rows(out)
    zcal = {d: cx.fit([x["raw_depth_m"] for x in z if x["direction"] == d], [x["owner_c"] for x in z if x["direction"] == d])
            for d in cx.DIRECTIONS}
    post = [cx.posterior({r["direction"]: r["raw_depth_m"]}, zcal) for r in rows]
    res["methods"]["xsec_raw_zjgcal"] = {**_summary(y, [int(np.argmax(p)) + 1 for p in post], post),
                                         "with_width": sum(r["raw_depth_m"] is not None for r in rows), "calib": zcal}
    # Zhanjiang itself, leave one day out (raw depth, ascending): the cross-section on more labels
    zy = [x["owner_c"] for x in z]
    zpost = _lodo(z, "raw_depth_m")
    res["zjg_raw_lodo"] = {**_summary(zy, [int(np.argmax(p)) + 1 for p in zpost], zpost),
                           "owner_c": {str(k): zy.count(k) for k in (1, 2, 3)},
                           "prior": _summary(zy, [1] * len(zy), [list(cx.PRIOR_C)] * len(zy))}
    prior = [list(cx.PRIOR_C)] * len(rows)
    res["methods"]["prior"] = _summary(y, [1] * len(rows), prior)
    for k in ("est2_c", "occ1_c"):
        rr = [(r["owner_c"], r[k]) for r in rows if r.get(k) is not None]
        res["methods"][k[:-2]] = _summary([a for a, _ in rr], [b for _, b in rr])
    res["calib_all"] = {d: cx.fit([r["width_m"] for r in rows if r["direction"] == d and r["width_m"] is not None],
                                  [r["owner_c"] for r in rows if r["direction"] == d and r["width_m"] is not None])
                        for d in cx.DIRECTIONS}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rows abreast from the cross-section (cols-1) on the owner's labels (no fetch)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--answers-ref")
    ap.add_argument("--write-calib", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = evaluate(a.out, a.answers_ref)
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.write_calib:
        z = res["methods"]["xsec_raw_zjgcal"]["calib"]
        doc = {"version": cx.COLS_VERSION, "measure": "生の σ0 からの占有の奥行き（est-1 depth_m、水際から途切れずに続く距離）",
               "source": (f"湛江の目視ラベル（labels/ZJG-s2-labels.csv の verified_c、{z['ascending']['n']} 区間×日）と組にした S1 の奥行き"
                          "（labels/ZJG-s2-s1-match.csv）。昇交（軌道 157・84）だけ。降交は撮影が無く既定値"),
               "check": {"SYL_SLP": {k: res["methods"]["xsec_raw_zjgcal"][k] for k in ("n", "agree", "brier3", "confusion")},
                         "ZJG_lodo": {k: res["zjg_raw_lodo"][k] for k in ("n", "agree", "brier3", "confusion")}},
               "directions": z}
        cx.CALIB_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n", "by_direction", "owner_c", "methods", "calib_all", "zjg_raw_lodo")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
