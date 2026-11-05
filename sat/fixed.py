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

fix-2, guard only (2部の設計書 fixed_eval 4.2 の 3, from ``FIX2_SINCE``): the structure test cannot tell a structure from
a ship that stays at one berth for months (bright when "empty", nothing new today). A candidate that meets it is kept
(``fixed_flag`` = "guarded", ``ship_conf`` = "low", counted as before fix-1) when any of these holds (``fixed_guard``):

    naval_long   the nearest berth line (berths.py, naval-1: keep_m in the quay band, the window's band_m in the open
                 water) is naval and the candidate is ``GUARD["naval_len_m"]`` or longer
    naval_near   that line is naval and a candidate of the same scene on a naval line, ``naval_len_m`` or longer, is
                 within ``GUARD["near_m"]`` (a split piece of a moored warship; until split pieces are merged)
    yard_site    the site's kind is yard
    yard_sector  the candidate is in a shipyard sector (role dock / slip / fitout / hall_apron)
    yard_line    the nearest berth line is a shipyard berth of the class file (why "造船所の岸壁…")

``fixed_guard`` (the reasons, "+"-joined) and ``fixed_guard_class`` (the line class used) are written on every fix-2
candidate, excluded or not, so the decision can be traced either way. The registry of confirmed fixed objects, the point
and field flags and ``berthed_long`` of the design are not in this version yet (they do not change the counts, except the
registry, which will carry its own hash per scene).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

VERSION = "fix-2"
SINCE_UTC = "2026-11-05"          # the fixed-object rule at all (fix-1 from this date; no published scene had fix-1)
FIX1_VERSION = "fix-1"
FIX2_SINCE = "2026-11-05"         # fix-2 (the guard) from this date; earlier scenes from SINCE_UTC keep fix-1
GUARD = {
    "naval_len_m": 60.0,   # est-2's "warship = 60 m and more"; the structures dropped on naval berths were 30-46 m
    "near_m": 100.0,       # a naval candidate this near a naval one of naval_len_m: a split piece of the same ship
}
YARD_ROLES = ("dock", "slip", "fitout", "hall_apron")
GUARD_REASONS = ("naval_long", "naval_near", "yard_site", "yard_sector", "yard_line")
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
    "guarded": "fix-2：structure の条件に当たるが、海軍の岸壁の 60 m 以上の候補（またはその片）か造船所の候補なので外さずに数える"
               "（長く係留している艦は構造物と同じく基準でも明るく当日も変化しないため）。理由は fixed_guard",
    "ship_conf": "船らしさ。high＝当日だけ基準より大きく上がり（high_rise_db 以上、z high_z 以上）、VH も強く、長さ high_len_m 以上。"
                 "medium＝基準より rise_db 以上上がった。low＝それ以外（よく船がいる所か、変化が弱い）。基準の無い軌道は自分の画像の"
                 "コントラストだけで、medium まで",
}


def active(acquired_utc: str) -> bool:
    return acquired_utc[:10] >= SINCE_UTC


def version_for(acquired_utc: str) -> str | None:
    """The fixed-object rule of a scene: None before SINCE_UTC, fix-1 before FIX2_SINCE, else fix-2."""
    if not active(acquired_utc):
        return None
    return VERSION if acquired_utc[:10] >= FIX2_SINCE else FIX1_VERSION


def _dist_m(a: dict, b: dict) -> float:
    my = 111132.0
    mx = 111320.0 * math.cos(math.radians((a["lat"] + b["lat"]) / 2))
    return math.hypot((a["lon"] - b["lon"]) * mx, (a["lat"] - b["lat"]) * my)


def guard_reasons(dets: list[dict], ctx: list[dict], cfg: dict | None = None) -> list[list[str]]:
    """fix-2's guard for every candidate (all of them, structure or not). ``ctx[i]``: {"berth_class": naval / civil /
    unknown / None, "yard_site", "yard_sector", "yard_line": bool} for ``dets[i]`` (which has lon, lat, length_m)."""
    g = {**GUARD, **(cfg or {})}
    long_m, near_m = float(g["naval_len_m"]), float(g["near_m"])
    naval = [c.get("berth_class") == "naval" for c in ctx]
    anchors = [i for i, d in enumerate(dets) if naval[i] and float(d.get("length_m") or 0) >= long_m]
    out = []
    for i, (d, c) in enumerate(zip(dets, ctx)):
        r = []
        if naval[i] and float(d.get("length_m") or 0) >= long_m:
            r.append("naval_long")
        elif naval[i] and any(j != i and _dist_m(d, dets[j]) <= near_m for j in anchors):
            r.append("naval_near")
        r += [k for k in ("yard_site", "yard_sector", "yard_line") if c.get(k)]
        out.append(r)
    return out


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


def assess(d: dict, vv_s: np.ndarray, base: dict | None, wp10: float | None, cfg: dict | None = None,
           version: str = FIX1_VERSION) -> dict:
    """The fix-1 decision on one candidate (``d`` from sat/detect.py: rmin..cmax, pixels, vh_ratio_db, length_m,
    contrast_vv_db). ``vv_s``: the scene's VV, 3x3 mean (smooth3_db). Returns the fields written to the candidate
    (``version`` is only the label; the guard of fix-2 is applied by ``classify``)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out = {"fixed_version": version, "fixed_excluded": "", "ship_conf": "low", "rise_db": None, "rise_z": None,
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
             cfg: dict | None = None, version: str = FIX1_VERSION, guard_ctx: list[dict] | None = None) -> dict:
    """Writes ``assess`` into every candidate. Returns the counts (structure, high, medium, low, no_baseline; fix-2 also
    guarded and guarded_by: reason -> count). ``version`` fix-2: the guard (``guard_ctx``, one dict per candidate, see
    ``guard_reasons``; None = no berth class and no shipyard, so no guard) keeps a structure candidate."""
    vv_s = smooth3_db(vv_db)
    wp10 = water_p10(base, meta, water) if base is not None else None
    counts = {"structure": 0, "high": 0, "medium": 0, "low": 0, "no_baseline": 0}
    fix2 = version == VERSION
    if fix2:
        ctx = guard_ctx if guard_ctx is not None else [{} for _ in dets]
        reasons = guard_reasons(dets, ctx, cfg)
        counts.update(guarded=0, guarded_by={k: 0 for k in GUARD_REASONS})
    for i, d in enumerate(dets):
        r = assess(d, vv_s, base, wp10, cfg, version)
        if fix2:
            why = reasons[i]
            r.update(fixed_flag="", fixed_guard="+".join(why), fixed_guard_class=ctx[i].get("berth_class") or "")
            if r["fixed_excluded"] == "structure" and why:   # a structure by fix-1's test, kept by the guard
                r.update(fixed_excluded="", fixed_flag="guarded", ship_conf="low")
                counts["guarded"] += 1
                for k in why:
                    counts["guarded_by"][k] += 1
        d.update(r)
        if r["fixed_excluded"]:
            counts["structure"] += 1
        else:
            counts[r["ship_conf"]] += 1
        if r.get("basis", "").startswith("no_baseline"):
            counts["no_baseline"] += 1
    counts["version"] = version
    return counts


def settings() -> dict:
    return {"version": VERSION, "since_utc": SINCE_UTC, "defaults": DEFAULTS, "notes": NOTE, "labels": CONF_LABEL,
            "versions": [{"version": FIX1_VERSION, "since_utc": SINCE_UTC, "until_utc": FIX2_SINCE},
                         {"version": VERSION, "since_utc": FIX2_SINCE}],
            "guard": {**GUARD, "yard_roles": list(YARD_ROLES), "reasons": list(GUARD_REASONS),
                      "rule": ("fix-2：structure の条件に当たる候補でも、次のどれかに当たれば外さずに数える（fixed_flag＝guarded、"
                               "ship_conf＝low）。naval_long＝最も近い着く線（naval-1 と同じ距離：岸壁帯は keep_m、開けた水面は窓の band_m）が"
                               "海軍で長さ naval_len_m 以上。naval_near＝その線が海軍で、同じ撮影の near_m 以内に海軍の線の naval_len_m 以上の候補"
                               "（係留中の艦の割れた片。片をまとめる処理が入るまでの代わり）。yard_site＝地点の kind が yard。"
                               "yard_sector＝造船所の区画（role が yard_roles）。yard_line＝最も近い着く線が造船所の岸壁"
                               "（sat/berth_class の why が「造船所の岸壁」）。理由 fixed_guard と、使った線の区分 fixed_guard_class は、"
                               "外した候補にも残した候補にも記録する。人が確かめた固定物の台帳・点状（P）・密集（F）の印・berthed_long は未実装"),
                      "source": "2部の設計書 fixed_eval（2026-11-06）4.2 の 3"},
            "rule": ("候補の画素（その日の明るい順に pixels 個、3×3 平均）を、同じ軌道の空の岸壁の基準（VV p10・p90）と比べる。"
                     "base_db＝基準の p10 と水面の p10 の差、rise_db＝その日の VV と基準の p10 の差の中央値、rise_z＝画素数で割り引いた z。"
                     "structure の候補は detections/ に fixed_excluded 付きで残し、件数（harbour_candidates・quay_candidates・錨泊・区画）に入れない。"
                     "SINCE_UTC より前の行は書き換えない")}
