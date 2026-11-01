"""Out-of-sample evaluation of est-2 (design 1.4 / PR-2, step 3): the frozen scenes against the owner's new eye labels.

The first calibration (est-2-cal1, sat/quay_est2_calib.py) used the owner's labels of the Sentinel-2 days before the
freeze (2026-10-27) and was evaluated by leave-one-date-out inside that sample. From the freeze on the scenes are
out of sample: nothing was fitted on them. This script compares est-2's output for those scenes with the owner's labels
of the Sentinel-2 days from the freeze on (labels/<site>-oos-answers.csv from the review page docs/sat/label/?site=ZJG&oos=1,
merged with ``python -m sat.s2_answers merge --oos``; the sections and days are in labels/<site>/review-oos.json):

* columns abreast: confusion matrix of the owner's c (0 = none, 1-3) against est-2's c for the section (0 when no stretch
  of the S1 scene lies on the section or P(real) < 0.5), the agreement, and the Brier score of the 4-class probability
  [1 - P(real), P(real) x p_c] against the owner's class;
* occupancy: agreement and Brier of P(real) (0 without a stretch) against the owner's occupied;
* the 80% interval of the warship count (60 m or more) against the owner's count of the day (labels/<site>-oos-scene.csv,
  the whole port, optional): the share of days whose count lies in [p10, p90] of the est-2 scene (nominal 80%).

A Sentinel-2 day is paired with the est-2 scene (S1 on or after the freeze) nearest in time within S2_MAX_H (96 h, the
window of the colour clue). The owner's label is of the S2 time (about 03 UTC), the S1 of 10-11 UTC and up to 4 days
away: ships may have changed in between (dt_h is kept per row). With no label or no scene from the freeze on the result is
"標本 0" (n = 0, nothing computed). Output: docs/data/sat/eval_est2.json (referenced from meta.json, sat/run.py).

    python -m sat.eval_est2 [--out docs/data/sat]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import quay_est2 as e2

EVAL_VERSION = "eval-est2-1"
NOMINAL = 0.8
CLASSES = (0, 1, 2, 3)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _freeze(site_id: str) -> str:
    try:
        from .run import load_sites
        s = next(x for x in load_sites() if x["id"] == site_id)
        return (s.get("quay") or {}).get("label_freeze_utc") or e2.FREEZE_UTC
    except (StopIteration, Exception):
        return e2.FREEZE_UTC


def _lat0(site_id: str) -> float:
    try:
        from .run import load_sites
        s = next(x for x in load_sites() if x["id"] == site_id)
        q = s.get("quay") or {}
        return float(q.get("est2_lat0") or s.get("center_lat") or 21.2)
    except (StopIteration, Exception):
        return 21.2


def load_sections(out: Path, site_id: str) -> dict:
    p = out / "labels" / site_id / "review-oos.json"
    if p.exists():
        rv = json.loads(p.read_text(encoding="utf-8"))
        return {s["id"]: {"way_id": s["way"], "side": s["side"], "line": s["line"]} for s in rv["sections"]}, {d["date"]: d["acquired_utc"] for d in rv["days"]}
    return {s["section_id"]: {"way_id": s["way_id"], "side": s["side"], "line": s["coords"]} for s in e2.load_sections(out, site_id)}, {}


def s2_times(out: Path, site_id: str) -> dict:
    return {r["date"]: r["acquired_utc"] for r in read_csv(out / "s2_scenes.csv") if r["site_id"] == site_id and r.get("result") == "clear"}


def stretch_of_section(sec: dict, feats: list[dict], stretches: list[dict], lat0: float):
    """The est-2 stretch (with the most points within SECTION_MATCH_M) of the same OSM way and side, or None."""
    sxy = e2._xy(sec["line"], lat0)
    best = None
    for f, st in zip(feats, stretches):
        if f.get("osm_way") != sec["way_id"] or e2._side(f) != int(sec["side"]):
            continue
        n = int((e2._min_dist(sxy, e2._xy(f["coords"], lat0)) <= e2.SECTION_MATCH_M).sum())
        if n and (best is None or n > best[0]):
            best = (n, st)
    return best[1] if best else None


def p4(st: dict | None) -> np.ndarray:
    """[P(none), P(c=1), P(c=2), P(c=3)] of the section from its stretch (no stretch: none)."""
    if st is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    pr = float(st.get("p_real", 1.0))
    return np.array([1 - pr] + [pr * float(x) for x in st["p_c"]])


def evaluate_site(out: Path, site_id: str = "ZJG") -> dict:
    freeze = _freeze(site_id)
    base = {"site_id": site_id, "freeze_utc": freeze}
    ans = [r for r in read_csv(out / "labels" / f"{site_id}-oos-answers.csv") if r.get("date", "") >= freeze]
    scene_lab = [r for r in read_csv(out / "labels" / f"{site_id}-oos-scene.csv") if r.get("date", "") >= freeze]
    path = out / ("quay_est2.json" if site_id == "ZJG" else f"quay_est2/{site_id}.json")
    est = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scenes": []}
    scenes = [s for s in est["scenes"] if s["site_id"] == site_id and s["acquired_utc"][:10] >= freeze]
    secs, rv_days = load_sections(out, site_id)
    times = {**s2_times(out, site_id), **rv_days}
    lat0 = _lat0(site_id)
    feats_by: dict = {}
    for f in e2.load_features(out):
        if f["site_id"] == site_id:
            feats_by.setdefault(f["acquired_utc"], []).append(f)

    def pair(date: str):
        t = times.get(date) or f"{date}T03:00:00Z"
        best = None
        for s in scenes:
            dt = (e2._t(s["acquired_utc"]) - e2._t(t)).total_seconds() / 3600.0
            if abs(dt) <= e2.S2_MAX_H and (best is None or abs(dt) < abs(best[0])):
                best = (dt, s)
        return best

    rows, conf = [], {}
    pv, yv, o_p, o_y = [], [], [], []
    for a in ans:
        if a.get("verified") != "1" or a["section_id"] not in secs:
            continue
        pr = pair(a["date"])
        if pr is None:
            continue
        dt, sc = pr
        feats = feats_by.get(sc["acquired_utc"], [])
        sts = sc["stretches"]
        st = stretch_of_section(secs[a["section_id"]], feats, sts, lat0) if len(feats) == len(sts) else None
        p = p4(st)
        y = int(a["verified_c"]) if a.get("verified_occupied") == "1" else 0
        pred = int(np.argmax(p))
        key = f"{y}->{pred}"
        conf[key] = conf.get(key, 0) + 1
        pv.append(p)
        yv.append(y)
        o_p.append(1 - float(p[0]))
        o_y.append(int(y > 0))
        rows.append({"section_id": a["section_id"], "date": a["date"], "s1": sc["acquired_utc"], "dt_h": round(dt, 1), "owner_c": y, "est2_c": pred,
                     "p": [round(float(x), 3) for x in p]})
    n = len(rows)
    cols = {"n": n, "confusion": dict(sorted(conf.items())), "agree": sum(v for k, v in conf.items() if k.split("->")[0] == k.split("->")[1]),
            "share": None, "brier4": None}
    occ = {"n": n, "agree": 0, "share": None, "brier": None}
    if n:
        cols["share"] = round(cols["agree"] / n, 3)
        cols["brier4"] = round(float(np.mean([((p - np.eye(4)[y]) ** 2).sum() for p, y in zip(pv, yv)])), 3)
        occ["agree"] = sum(1 for p, y in zip(o_p, o_y) if (p >= 0.5) == bool(y))
        occ["share"] = round(occ["agree"] / n, 3)
        occ["brier"] = round(float(np.mean([(p - y) ** 2 for p, y in zip(o_p, o_y)])), 3)
        occ["mean_p_occupied"] = round(float(np.mean(o_p)), 3)
        occ["owner_occupied_share"] = round(float(np.mean(o_y)), 3)
    wrows = []
    for r in scene_lab:
        if not (r.get("warships_60") or "").strip().isdigit():
            continue
        pr = pair(r["date"])
        if pr is None:
            continue
        dt, sc = pr
        w = sc["warship"]
        k = int(r["warships_60"])
        wrows.append({"date": r["date"], "s1": sc["acquired_utc"], "dt_h": round(dt, 1), "owner": k, "p10": w["p10"], "median": w["median"], "p90": w["p90"],
                      "hit": bool(w["p10"] <= k <= w["p90"])})
    wi = {"n": len(wrows), "hit": sum(1 for x in wrows if x["hit"]), "nominal": NOMINAL, "share": None, "rows": wrows}
    if wrows:
        wi["share"] = round(wi["hit"] / len(wrows), 3)
    status = "標本 0" if (n == 0 and not wrows) else f"標本 {n} 区間×日、軍艦の日 {len(wrows)}"
    return {**base, "status": status, "labels": {"answers": len(ans), "used": n, "days": len({r["date"] for r in rows}),
                                                  "scene_labels": len(scene_lab)},
            "scenes_out_of_sample": len(scenes), "columns": cols, "occupied": occ, "warship_interval": wi, "rows": rows}


def evaluate(out: Path) -> dict:
    ids = ["ZJG"] + sorted(p.parent.name for p in (out / "labels").glob("*/review-oos.json") if p.parent.name != "ZJG")
    sites = {sid: evaluate_site(out, sid) for sid in ids}
    return {"version": EVAL_VERSION, "method": ("凍結（2026-10-27）より後の est-2 の出力を、所有者の新しい目視ラベル（S2 の晴れの日）と突き合わせる。標本外：この撮影・ラベルは較正に使っていない。"
                       "列数の混同行列（人の列数 0〜3 と est-2 の列数）、占有と 4 区分のブライア、軍艦隻数の 80% 区間の的中率"),
            "rules": {"freeze": "2026-10-27（湛江。他の港は sites.json の quay.label_freeze_utc）以降の撮影と、その日以降の S2 の晴れの日の人の答えだけ",
                      "pair": f"S2 の日に最も近い est-2 の撮影（前後 {e2.S2_MAX_H:g} 時間以内）",
                      "interval": f"軍艦隻数の 80% 区間の的中率＝人の数が est-2 の p10〜p90 に入った日の割合（目安 {NOMINAL:.0%}）。人の軍艦の数は港全体・60 m 以上と思う艦（任意）",
                      "none": "標本 0＝凍結後の撮影か人の答えがまだ無い。計算しない"},
            "sites": sites}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="out-of-sample evaluation of est-2 (no fetch)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "data" / "sat")
    a = ap.parse_args(argv)
    res = evaluate(a.out)
    (a.out / "eval_est2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for sid, r in res["sites"].items():
        print(sid, r["status"], f"（凍結後の撮影 {r['scenes_out_of_sample']}）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
