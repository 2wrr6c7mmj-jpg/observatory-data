"""est-2 #7: calibration of est-2 on the Sentinel-2 labels checked by eye (first round).

Input (already stored, no fetch, no PU):
  labels/<site>-s2-labels.csv     verified_* filled by the owner (one person, 10 m natural colour x8)
  labels/<site>-s2-s1-match.csv   the same section x day with the nearest Sentinel-1 re-count (#5)
  labels/<site>-s2-sections.geojson

Only rows with status ok, verified 1 and a date before FREEZE_UTC (2026-10-27) are used.
The human label is of the Sentinel-2 time (about 03:20 UTC); the Sentinel-1 depth is of a scene
-40.5 to +79.4 h away, so ships may have changed in between (it widens the fitted spread).

What is fitted (each shrunk toward the placeholder by a pseudo-count, because the samples are small):
1. Depth likelihood: mean of the S1 occupied depth for every (face/orbit, c) cell and one common sd.
   Cell mean = (n * cell average + N0_MU * target) / (n + N0_MU), target = placeholder + common offset
   (the offset = n-weighted mean difference of all cells, itself shrunk to 0 with N0_OFFSET). Cells with
   no data take the target. Means are kept at least MIN_GAP_M apart in c. sd: the pooled within-cell
   sd, shrunk to the placeholder 12 m with NU0_SD degrees of freedom.
2. Smear (away face of a pier, the toward face of the same pier occupied in the same S1 scene):
   P(real) = (k + A0 * placeholder) / (n + A0), k of n such rows with a ship seen by eye.
3. Colour: the owner's note is that grey and white cannot be told apart at 10 m. Grey, white and
   black/white (human grey, bw) are merged into "achromatic". There is no warship truth, so the
   likelihood ratios are not fitted: they are capped (achromatic LR_ACHROMATIC, strong colour LR_STRONG)
   and an automatic class gets 1 + r_a (LR_ACHROMATIC - 1) + r_s (LR_STRONG - 1), r = share of the
   rows of that automatic class that the owner called achromatic / strong colour, shrunk toward 0
   (no information, LR 1) by N0_COLOUR. A row the owner checked uses the owner's colour directly.
4. Hard sections: an OSM way where the owner could not judge at least HARD_SHARE of the answered rows
   (at least HARD_MIN_N answers) is noted as "hard to judge" in the est-2 output (the estimate itself
   is not changed: not being able to judge it by eye at 10 m is not evidence about the ships there).

Evaluation: leave-one-date-out (each S2 day out in turn, fitted on the other days). Column agreement:
argmax of the est-2 column posterior from the S1 depth (place prior x depth likelihood) against the
owner's c. Occupation: S1 occupied x P(real) >= 0.5 against the owner's occupied, and the Brier score.

    python -m sat.quay_est2_calib --out docs/data/sat     # writes sat/quay_est2_calib.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import quay_est2 as e2

CALIB_VERSION = "est-2-cal1"
N0_MU = 5.0
N0_OFFSET = 5.0
NU0_SD = 5.0
MIN_GAP_M = 10.0
A0_SMEAR = 5.0
N0_COLOUR = 5.0
LR_ACHROMATIC = 1.5
LR_STRONG = 0.3
HARD_MIN_N = 5
HARD_SHARE = 0.5
GROUPS = ("away", "along", "toward_157", "toward_84")
GROUP_NAMES = {"away": "反対向き（両軌道）", "along": "沿う（両軌道）",
               "toward_157": "レーダー向き・軌道 157（入射約 45°）", "toward_84": "レーダー向き・軌道 84（入射約 32°）"}
HUMAN_ACHROMATIC = ("grey", "bw")
HUMAN_STRONG = ("vivid",)
OUT_FILE = Path(__file__).with_name("quay_est2_calib.json")


def placeholder() -> dict:
    return {"depth_mu": {"away": list(e2.PH_DEPTH_MU["away"]), "along": list(e2.PH_DEPTH_MU["along"]),
                         "toward_157": list(e2.PH_TOWARD_MU[45.0]), "toward_84": list(e2.PH_TOWARD_MU[32.0])},
            "depth_sd": e2.PH_DEPTH_SD_M, "smear_p_real": e2.PH_SMEAR_P_REAL}


def group_of(face: str, orbit: str) -> str | None:
    if face == "toward":
        return f"toward_{orbit}" if f"toward_{orbit}" in GROUPS else None
    return face if face in GROUPS else None


# --- rows ------------------------------------------------------------------------------------------
def load_rows(out: Path, site_id: str = "ZJG") -> tuple[list[dict], list[dict]]:
    """(verified rows joined with the S1 match, all answered rows incl. verified 0)."""
    ldir = out / "labels"
    labels = {(r["section_id"], r["date"]): r for r in e2.read_csv(ldir / f"{site_id}-s2-labels.csv")}
    match = e2.read_csv(ldir / f"{site_id}-s2-s1-match.csv")
    secs = {x["section_id"]: e2._xy(x["coords"], 21.2) for x in e2.load_sections(out, site_id)}
    toward_occ: dict = {}
    for m in match:
        if m.get("face_look") == "toward" and m.get("s1_occupied") == "1":
            toward_occ.setdefault((m["s1_acquired_utc"], m["section_id"].split(":")[0]), []).append(m["section_id"])
    answered, rows = [], []
    for m in match:
        lab = labels.get((m["section_id"], m["s2_date"]))
        if lab is None or lab.get("verified") in (None, "") or lab.get("status") != "ok" or m["s2_date"] >= e2.FREEZE_UTC:
            continue
        answered.append(lab)
        if lab["verified"] != "1":
            continue
        sid = m["section_id"]
        smear = (m.get("face_look") == "away" and m.get("kind") == "pier" and m.get("s1_occupied") == "1" and any(
            sid in secs and o in secs and float(e2._min_dist(secs[sid], secs[o]).min()) <= e2.SMEAR_MAX_M
            for o in toward_occ.get((m["s1_acquired_utc"], sid.split(":")[0]), [])))
        rows.append({
            "date": m["s2_date"], "section_id": sid, "way": sid.split(":")[0],
            "place": e2.place_of(m.get("sector"), m.get("kind")), "face": m.get("face_look") or "",
            "orbit": str(m.get("relative_orbit") or ""), "group": group_of(m.get("face_look") or "", str(m.get("relative_orbit") or "")),
            "s1_occ": m.get("s1_occupied") if m.get("s1_occupied") in ("0", "1") else None,
            "s1_depth": float(m["s1_depth_m"]) if m.get("s1_depth_m") not in (None, "") else None,
            "smear": smear,
            "v_occ": lab.get("verified_occupied"), "v_c": lab.get("verified_c"), "v_color": lab.get("verified_color") or "",
            "auto_occ": lab.get("occupied"), "auto_c": lab.get("c"), "auto_color": lab.get("color_class") or "",
        })
    return rows, answered


def column_rows(rows: list[dict]) -> list[dict]:
    """Rows usable for the depth likelihood: occupied by eye with c 1-3, S1 occupied with a depth."""
    return [r for r in rows if r["v_occ"] == "1" and r["v_c"] in ("1", "2", "3") and r["s1_occ"] == "1"
            and r["s1_depth"] is not None and r["group"]]


def occupancy_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["v_occ"] in ("0", "1") and r["s1_occ"] is not None]


# --- fit -------------------------------------------------------------------------------------------
def fit_depth(rows: list[dict]) -> dict:
    ph = placeholder()
    cells: dict = {}
    for r in column_rows(rows):
        cells.setdefault((r["group"], int(r["v_c"])), []).append(r["s1_depth"])
    n_all = sum(len(v) for v in cells.values())
    off_sum = sum(len(v) * (float(np.mean(v)) - ph["depth_mu"][g][c - 1]) for (g, c), v in cells.items())
    offset = off_sum / (n_all + N0_OFFSET) if n_all else 0.0
    mu, table = {}, []
    for g in GROUPS:
        m = []
        for c in e2.C_VALUES:
            v = cells.get((g, c), [])
            t = ph["depth_mu"][g][c - 1] + offset
            x = (len(v) * float(np.mean(v)) + N0_MU * t) / (len(v) + N0_MU) if v else t
            m.append(x)
            table.append({"group": g, "c": c, "n": len(v), "mean_obs": round(float(np.mean(v)), 1) if v else None,
                          "sd_obs": round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else None,
                          "placeholder": ph["depth_mu"][g][c - 1]})
        for i in (1, 2):  # keep the columns ordered
            m[i] = max(m[i], m[i - 1] + MIN_GAP_M)
        mu[g] = [round(x, 1) for x in m]
        for row in table:
            if row["group"] == g:
                row["calibrated"] = mu[g][row["c"] - 1]
    ss = sum(float(((np.array(v) - np.mean(v)) ** 2).sum()) for v in cells.values())
    df = n_all - len(cells)
    sd = math.sqrt((ss + NU0_SD * ph["depth_sd"] ** 2) / (df + NU0_SD)) if df > 0 else ph["depth_sd"]
    return {"depth_mu": mu, "depth_sd": round(sd, 1), "offset": round(offset, 1), "n": n_all,
            "pooled_sd_obs": round(math.sqrt(ss / df), 1) if df > 0 else None, "table": table}


def fit_smear(rows: list[dict]) -> dict:
    s = [r for r in occupancy_rows(rows) if r["smear"]]
    k = sum(1 for r in s if r["v_occ"] == "1")
    p = (k + A0_SMEAR * e2.PH_SMEAR_P_REAL) / (len(s) + A0_SMEAR)
    other = [r for r in occupancy_rows(rows) if r["s1_occ"] == "1" and not r["smear"]]
    return {"smear_p_real": round(p, 2), "n": len(s), "seen": k,
            "other_s1_occupied": {"n": len(other), "seen": sum(1 for r in other if r["v_occ"] == "1")}}


def fit_colour(rows: list[dict]) -> dict:
    out, tab = {}, {}
    occ = [r for r in rows if r["v_occ"] == "1"]
    for cls in ("grey", "colour", "contrast", "mixed"):
        s = [r for r in occ if r["auto_color"] == cls]
        a = sum(1 for r in s if r["v_color"] in HUMAN_ACHROMATIC)
        st = sum(1 for r in s if r["v_color"] in HUMAN_STRONG)
        ra, rs = a / (len(s) + N0_COLOUR), st / (len(s) + N0_COLOUR)
        out[cls] = round(1 + ra * (LR_ACHROMATIC - 1) + rs * (LR_STRONG - 1), 2)
        tab[cls] = {"n": len(s), "owner_achromatic": a, "owner_strong": st, "owner_blank": sum(1 for r in s if not r["v_color"])}
    return {"s2_lr_auto": out, "s2_lr_owner": {"achromatic": LR_ACHROMATIC, "strong": LR_STRONG, "blank": 1.0},
            "table": tab}


def hard_ways(answered: list[dict]) -> dict:
    by: dict = {}
    for r in answered:
        w = r["section_id"].split(":")[0]
        by.setdefault(w, [0, 0])
        by[w][0] += 1
        by[w][1] += r["verified"] == "0"
    return {w: {"answered": n, "cannot_judge": k, "share": round(k / n, 2)}
            for w, (n, k) in sorted(by.items()) if n >= HARD_MIN_N and k / n >= HARD_SHARE}


def fit(rows: list[dict]) -> dict:
    d = fit_depth(rows)
    s = fit_smear(rows)
    return {"depth_mu": d["depth_mu"], "depth_sd": d["depth_sd"], "smear_p_real": s["smear_p_real"]}


# --- evaluate --------------------------------------------------------------------------------------
def posterior_c(r: dict, params: dict) -> np.ndarray:
    mu = params["depth_mu"][r["group"]]
    lk, _ = e2.depth_like(r["s1_depth"], "x", None, e2.DEPTH_LIMIT_M, mu=tuple(mu), sd=params["depth_sd"])
    return e2.c_posterior(r["place"], [lk])


def predict_c(r: dict, params: dict) -> int:
    return int(np.argmax(posterior_c(r, params))) + 1


def p_occ(r: dict, params: dict) -> float:
    if r["s1_occ"] != "1":
        return 0.0
    return params["smear_p_real"] if r["smear"] else 1.0


def score(rows: list[dict], params_of) -> dict:
    """params_of(row) -> params. Column agreement and occupation agreement / Brier."""
    cr = column_rows(rows)
    conf: dict = {}
    for r in cr:
        k = f"{r['v_c']}->{predict_c(r, params_of(r))}"
        conf[k] = conf.get(k, 0) + 1
    ca = sum(v for k, v in conf.items() if k[0] == k[-1])
    pt = [float(posterior_c(r, params_of(r))[int(r["v_c"]) - 1]) for r in cr]
    ec = sum(float((posterior_c(r, params_of(r)) * np.array(e2.C_VALUES)).sum()) for r in cr)
    orows = occupancy_rows(rows)
    ps = [(p_occ(r, params_of(r)), r["v_occ"] == "1") for r in orows]
    oa = sum(1 for p, y in ps if (p >= 0.5) == y)
    brier = sum((p - y) ** 2 for p, y in ps) / len(ps) if ps else None
    sm = [(p_occ(r, params_of(r)), r["v_occ"] == "1") for r in orows if r["smear"]]
    return {"columns": {"n": len(cr), "agree": ca, "share": round(ca / len(cr), 3) if cr else None, "confusion": dict(sorted(conf.items())),
                        "mean_p_true": round(float(np.mean(pt)), 3) if pt else None,
                        "sum_c_owner": sum(int(r["v_c"]) for r in cr), "sum_c_expected": round(ec, 1),
                        "log_score": round(float(np.mean(np.log(np.maximum(pt, 1e-9)))), 3) if pt else None},
            "occupied": {"n": len(ps), "agree": oa, "share": round(oa / len(ps), 3) if ps else None,
                         "brier": round(brier, 3) if brier is not None else None},
            "smear_rows": {"n": len(sm), "brier": round(sum((p - y) ** 2 for p, y in sm) / len(sm), 3) if sm else None}}


def cross_validate(rows: list[dict]) -> dict:
    ph = placeholder()
    days = sorted({r["date"] for r in rows})
    folds = {d: fit([r for r in rows if r["date"] != d]) for d in days}
    always_one = {"depth_mu": {g: [0.0, 1e3, 2e3] for g in GROUPS}, "depth_sd": 1e3, "smear_p_real": 1.0}
    return {"method": "日付ごとの leave-one-out（S2 の晴れの日 1 日を外し、残りの日で較正して外した日を評価）",
            "days": len(days),
            "before": score(rows, lambda r: ph),
            "after_cv": score(rows, lambda r: folds[r["date"]]),
            "after_in_sample": score(rows, lambda r, _p=fit(rows): _p),
            "always_c1": {k: v for k, v in score(column_rows(rows), lambda r: always_one)["columns"].items()
                          if k in ("n", "agree", "share", "confusion")},
            "note": ("列数は人の c（S2 の時刻）と S1 の奥行き（−40.5〜+79.4 時間ずれる）の組で 45 件と少ない。"
                     "占有は S1 の占有×本当に船である確率（にじみ）が 0.5 以上を「いる」として人と比べた。mean_p_true＝人の c に付けた事後確率の平均、"
                     "log_score＝その対数の平均（0 に近いほどよい）。sum_c_owner／sum_c_expected＝人の列数の合計と、事後の列数の期待値の合計（隻数の偏りの目安）")}


def auto_agreement(rows: list[dict]) -> dict:
    """The S2 automatic labels (#5) against the owner (for the record; not part of est-2)."""
    occ = [r for r in rows if r["v_occ"] in ("0", "1")]
    both = [r for r in occ if r["v_occ"] == "1" and r["auto_occ"] == "1" and r["v_c"] in ("1", "2", "3")]
    conf: dict = {}
    for r in both:
        k = f"auto {r['auto_c']} / owner {r['v_c']}"
        conf[k] = conf.get(k, 0) + 1
    return {"occupied": {"n": len(occ), "agree": sum(1 for r in occ if r["v_occ"] == r["auto_occ"]),
                         "auto_yes_owner_no": sum(1 for r in occ if r["auto_occ"] == "1" and r["v_occ"] == "0"),
                         "auto_no_owner_yes": sum(1 for r in occ if r["auto_occ"] == "0" and r["v_occ"] == "1")},
            "columns": {"n": len(both), "agree": sum(1 for r in both if r["auto_c"] == r["v_c"]), "confusion": dict(sorted(conf.items()))},
            "s1_occupied": {"n": len(occupancy_rows(rows)), "agree": sum(1 for r in occupancy_rows(rows) if r["s1_occ"] == r["v_occ"]),
                            "s1_yes_owner_no": sum(1 for r in occupancy_rows(rows) if r["s1_occ"] == "1" and r["v_occ"] == "0"),
                            "s1_no_owner_yes": sum(1 for r in occupancy_rows(rows) if r["s1_occ"] == "0" and r["v_occ"] == "1")}}


def calibrate(out: Path, site_id: str = "ZJG") -> dict:
    rows, answered = load_rows(out, site_id)
    d, s, c = fit_depth(rows), fit_smear(rows), fit_colour(rows)
    return {
        "version": CALIB_VERSION, "site_id": site_id,
        "labels": {"answered": len(answered), "verified": len(rows), "cannot_judge": len(answered) - len(rows),
                   "by": "所有者 1 人の目視（確認ページ 2026-10-30、Sentinel-2 10 m 自然色を 8 倍に拡大、水色の枠は水際から 120 m）",
                   "used": "status＝ok かつ verified＝1、2026-10-27 より前の日だけ",
                   "column_rows": d["n"], "occupancy_rows": len(occupancy_rows(rows)), "days": len({r['date'] for r in rows})},
        "params": {"depth_mu": d["depth_mu"], "depth_sd": d["depth_sd"], "smear_p_real": s["smear_p_real"],
                   "s2_lr_auto": c["s2_lr_auto"], "s2_lr_owner": c["s2_lr_owner"], "hard_ways": sorted(hard_ways(answered))},
        "fit": {"depth": {k: d[k] for k in ("offset", "n", "pooled_sd_obs", "table")}, "smear": s, "colour": c["table"],
                "hard_ways": hard_ways(answered)},
        "shrinkage": {"depth_mean_pseudo_n": N0_MU, "offset_pseudo_n": N0_OFFSET, "sd_pseudo_df": NU0_SD,
                      "min_gap_m": MIN_GAP_M, "smear_pseudo_n": A0_SMEAR, "colour_pseudo_n": N0_COLOUR,
                      "lr_caps": {"achromatic": LR_ACHROMATIC, "strong": LR_STRONG},
                      "hard": {"min_answers": HARD_MIN_N, "cannot_share": HARD_SHARE}},
        "evaluation": cross_validate(rows),
        "auto_vs_owner": auto_agreement(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="est-2 #7: calibrate on the verified S2 labels (no fetch)")
    ap.add_argument("--out", default="docs/data/sat")
    ap.add_argument("--site", default="ZJG")
    ap.add_argument("--write", default=str(OUT_FILE))
    a = ap.parse_args(argv)
    res = calibrate(Path(a.out), a.site)
    Path(a.write).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ev = res["evaluation"]
    print(json.dumps({"params": res["params"], "before": ev["before"], "after_cv": ev["after_cv"]}, ensure_ascii=False, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
