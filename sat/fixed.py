"""Fixed objects and the pier/ship decision in the ports (fix-1): every candidate's pixels against the empty-berth
baseline of the scene's orbit.

User's instruction (2026-11-04): a fixed-object mask for every site, telling piers from ships, and a stated
confidence per candidate. win-1 (sat/windows.py) looks at the candidate's centre pixel only; a CFAR candidate on the
edge of a pier, a dolphin, a buoy or a crane barge has its centre on water and stays. fix-1 looks at the candidate's
own pixels (the brightest ``pixels`` of its box, the 3x3 mean as the baseline was made) and asks two questions:

1. Is the spot bright when the berth is empty?  ``base_db`` = the baseline VV p10 there minus the site's water p10
   (sat/baseline meta ``water_vv_p10_median_db``). A ship is not there in 9 of 10 scenes; a pier edge or a buoy is.
2. Did today's scene rise above that?  ``rise_db`` = median of (scene VV - baseline VV p10) over the pixels, with a
   z-score for n pixels (speckle of the 3x3 mean of GRD, ``sigma_db`` per pixel, ``corr_px`` pixels per independent
   sample: BILINEAR/3x3 smoothing correlate neighbours).

    fixed ("structure"):  base_db >= bright_db  and  rise_db < rise_db  and  p90 - p10 < spread_db
                          (bright when empty, nothing new today, never varies: a structure the mask misses)
    otherwise the candidate stays and gets ``ship_conf``:
        high    rise_db >= high_rise_db and z >= high_z and VH ratio >= high_vh_db and length >= high_len_m
        medium  rise_db >= rise_db
        low     the rest (a spot that is often occupied, or a weak change)
    no baseline for the orbit: ``ship_conf`` from the scene's own contrast (medium at best), fixed never set.

From ``SINCE_UTC`` only (earlier rows are not rewritten): a fixed candidate stays in detections/ with
``fixed_excluded`` = "structure" and is not counted (``at_berth``), like win-1's ``window_excluded``. Zhanjiang's
golden scenes are older than SINCE_UTC, so tests/sat_zjg_golden.py is unchanged.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

VERSION = "fix-1"
SINCE_UTC = "2026-11-05"
DEFAULTS = {
    "bright_db": 8.0,      # baseline p10 this far above the site's water p10: bright when the berth is empty
    "rise_db": 3.0,        # today's VV over the baseline p10, median of the candidate's pixels
    "spread_db": 3.5,      # baseline p90 - p10 under this: never varies (Zhanjiang 2026-10: the structures 1.8-2.5,
                           # berths that are mostly occupied 3.6-5)
    "sigma_db": 1.6,       # speckle of one 3x3-mean pixel (dB), scene minus baseline
    "corr_px": 3.0,        # pixels per independent sample
    "high_rise_db": 6.0,
    "high_z": 5.0,
    "high_vh_db": 3.0,
    "high_len_m": 40.0,
    "pad_px": 1,
}
CONF_LABEL = {"high": "高", "medium": "中", "low": "低"}
NOTE = {
    "structure": "空の岸壁の基準でも明るく（水面の p10 より bright_db 以上）、当日も上がっておらず（rise_db 未満）、"
                 "基準の p90−p10 も小さい：桟橋の縁・係船柱・ブイ・クレーン台船など。数えない",
    "ship_conf": "船らしさ。high＝当日だけ基準より大きく上がり（high_rise_db 以上、z high_z 以上）、VH も強く、長さ high_len_m 以上。"
                 "medium＝基準より rise_db 以上上がった。low＝それ以外（よく船がいる所か、変化が弱い）。基準の無い軌道は自分の画像の"
                 "コントラストだけで、medium まで",
}


def active(acquired_utc: str) -> bool:
    return acquired_utc[:10] >= SINCE_UTC


def smooth3_db(db: np.ndarray) -> np.ndarray:
    """3x3 mean in linear power of the finite pixels (as sat/baseline), back in dB; NaN where none."""
    lin = np.where(np.isfinite(db), 10.0 ** (np.nan_to_num(db) / 10.0), 0.0)
    ok = np.isfinite(db).astype(np.float64)
    pad = lambda a: np.pad(a, 1)  # noqa: E731
    lp, op = pad(lin), pad(ok)
    h, w = db.shape
    s = sum(lp[i:i + h, j:j + w] for i in range(3) for j in range(3))
    n = sum(op[i:i + h, j:j + w] for i in range(3) for j in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, 10.0 * np.log10(s / np.maximum(n, 1)), np.nan)


def water_p10(base: dict, meta: dict | None, water: np.ndarray | None) -> float | None:
    v = (meta or {}).get("water_vv_p10_median_db")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if water is not None:
        p = base["vv_p10"][np.asarray(water, bool)]
        p = p[np.isfinite(p)]
        if p.size:
            return float(np.median(p))
    return None


def assess(d: dict, vv_s: np.ndarray, base: dict | None, wp10: float | None, cfg: dict | None = None) -> dict:
    """The decision on one candidate (``d`` from sat/detect.py: rmin..cmax, pixels, vh_ratio_db, length_m,
    contrast_vv_db). ``vv_s``: the scene's VV, 3x3 mean (smooth3_db). Returns the fields written to the candidate."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out = {"fixed_version": VERSION, "fixed_excluded": "", "ship_conf": "low", "rise_db": None, "rise_z": None,
           "base_db": None}
    h, w = vv_s.shape
    pad = int(cfg["pad_px"])
    r0, r1 = max(0, d["rmin"] - pad), min(h, d["rmax"] + pad + 1)
    c0, c1 = max(0, d["cmin"] - pad), min(w, d["cmax"] + pad + 1)
    win = vv_s[r0:r1, c0:c1]
    vh = d.get("vh_ratio_db") or 0.0
    if base is None or wp10 is None or not np.isfinite(win).any():
        strong = (d.get("contrast_vv_db") or 0.0) >= 10.0 and vh >= float(cfg["high_vh_db"])
        out["ship_conf"] = "medium" if strong else "low"
        out["basis"] = "no_baseline"
        return out
    p10 = base["vv_p10"][r0:r1, c0:c1]
    p90 = base["vv_p90"][r0:r1, c0:c1]
    ok = np.isfinite(win) & np.isfinite(p10)
    if not ok.any():
        out["basis"] = "no_baseline_here"
        strong = (d.get("contrast_vv_db") or 0.0) >= 10.0 and vh >= float(cfg["high_vh_db"])
        out["ship_conf"] = "medium" if strong else "low"
        return out
    k = max(1, min(int(d.get("pixels") or 1), int(ok.sum())))
    vals = np.where(ok, win, -np.inf).ravel()
    idx = np.argsort(vals)[::-1][:k]          # the candidate's brightest pixels today
    rise = win.ravel()[idx] - p10.ravel()[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = float(np.nanmedian((p90 - p10).ravel()[idx]))
    rise_med = float(np.median(rise))
    base_db = float(np.median(p10.ravel()[idx])) - wp10
    n_eff = max(1.0, k / float(cfg["corr_px"]))
    z = rise_med * math.sqrt(n_eff) / (float(cfg["sigma_db"]) * math.sqrt(2.0))
    out.update(rise_db=round(rise_med, 1), rise_z=round(z, 1), base_db=round(base_db, 1),
               spread_db=round(spread, 1) if math.isfinite(spread) else None, basis="baseline")
    if base_db >= float(cfg["bright_db"]) and rise_med < float(cfg["rise_db"]) \
            and (not math.isfinite(spread) or spread < float(cfg["spread_db"])):
        out["fixed_excluded"] = "structure"
        out["ship_conf"] = None
        return out
    if (rise_med >= float(cfg["high_rise_db"]) and z >= float(cfg["high_z"]) and vh >= float(cfg["high_vh_db"])
            and (d.get("length_m") or 0) >= float(cfg["high_len_m"])):
        out["ship_conf"] = "high"
    elif rise_med >= float(cfg["rise_db"]):
        out["ship_conf"] = "medium"
    else:
        out["ship_conf"] = "low"
    return out


def classify(dets: list[dict], vv_db: np.ndarray, base: dict | None, meta: dict | None, water: np.ndarray | None,
             cfg: dict | None = None) -> dict:
    """Writes ``assess`` into every candidate. Returns the counts (structure, high, medium, low, no_baseline)."""
    vv_s = smooth3_db(vv_db)
    wp10 = water_p10(base, meta, water) if base is not None else None
    counts = {"structure": 0, "high": 0, "medium": 0, "low": 0, "no_baseline": 0}
    for d in dets:
        r = assess(d, vv_s, base, wp10, cfg)
        d.update(r)
        if r["fixed_excluded"]:
            counts["structure"] += 1
        else:
            counts[r["ship_conf"]] += 1
        if r.get("basis", "").startswith("no_baseline"):
            counts["no_baseline"] += 1
    counts["version"] = VERSION
    return counts


def settings() -> dict:
    return {"version": VERSION, "since_utc": SINCE_UTC, "defaults": DEFAULTS, "notes": NOTE, "labels": CONF_LABEL,
            "rule": ("候補の画素（その日の明るい順に pixels 個、3×3 平均）を、同じ軌道の空の岸壁の基準（VV p10・p90）と比べる。"
                     "base_db＝基準の p10 と水面の p10 の差、rise_db＝その日の VV と基準の p10 の差の中央値、rise_z＝画素数で割り引いた z。"
                     "structure の候補は detections/ に fixed_excluded 付きで残し、件数（harbour_candidates・quay_candidates・錨泊・区画）に入れない。"
                     "SINCE_UTC より前の行は書き換えない")}
