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

fix-2 (2部の設計書 fixed_eval 4.2, from ``FIX2_SINCE``): the structure test cannot tell a structure from a ship that
stays at one berth for months (bright when "empty", nothing new today). In two steps, so that the merge of split pieces
(split_eval (b)) can come in between without changing this module:

1. ``score`` (per candidate, nothing is excluded yet): fix-1's ``assess`` (``structure`` = S) and the match with the
   registry of confirmed fixed objects (sat/fixed_registry.py; centre within the entry's radius_m). ``bridge_ok`` says
   whether a candidate may join two others in the merge: never an S or a registry candidate (a structure must not join a
   ship to a pier or two ships into one long object).
2. ``decide`` (per object: a merged group of candidates, or each candidate alone until the merge exists):
   - every piece is S or in the registry, and no guard: not counted, ``fixed_excluded`` = "registry" (all pieces in
     the registry) or "structure";
   - the guard (``fixed_guard``) keeps it (``fixed_flag`` = "guarded", ``berthed_long`` = true, ``ship_conf`` low):
       naval_long   the nearest berth line (berths.py, naval-1: keep_m in the quay band, the window's band_m in the open
                    water) is naval and the object is ``GUARD["naval_len_m"]`` or longer
       naval_near   that line is naval and a counted naval object of ``naval_len_m`` or more (not left out by berth-1 or
                    win-1, not dropped by the registry) is within ``GUARD["near_m"]`` (a split piece; until the merge)
       yard_site    the site's kind is yard
       yard_sector  in a shipyard sector (role dock / slip / fitout / hall_apron)
       yard_line    the nearest berth line is a naval shipyard berth of the class file (why "造船所の岸壁…", class naval)
     except that registry entries of kind pier_part or mask_error (a person saw a structure) win over the guard when
     every piece is in them, did not rise today (rise_db < DEFAULTS["rise_db"]: a ship that came alongside is kept) and
     the object is no longer than the entry's ``evidence.length_m`` plus ``REGISTRY_LEN`` (a ship moored for months
     next to the structure is bright in the baseline too and does not rise: only its length tells it apart). Those are
     counted as ``registry_over_guard`` and marked on the candidate (``registry_over_guard`` = true) for 2部's review;
   - flags that never change the counts: ``point`` (open water, 40 m or less, weak but always there: a buoy or a
     beacon) and ``field`` (5 or more small open-water candidates within 300 m: rafts, nets or a fishing fleet).

``fixed_guard`` (the reasons, "+"-joined), ``fixed_guard_class`` (the line class used) and ``fixed_registry_id`` are
written on every fix-2 candidate, excluded or not. Each scene records ``registry_hash`` (sat/fixed_registry.py) in
status.json: a registry change changes the counts.
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
REVISION = 2              # fix-2 revisions: 1 = the guard only (PR #76), 2 = registry, point / field, object decision
POINT = {                 # a weak spot that is always there (design 4.2 の 4); flag only
    "max_len_m": 40.0,    # the median length of split pieces (split_eval)
    "base_db": 4.0, "spread_db": 5.0, "rise_db": 6.0,   # looser than fix-1: weak but always there
}                          # and in the open water (zone harbour: 50 m or more from the land mask, cfar-v1's quay band)
FIELD = {"radius_m": 300.0, "min_n": 5, "max_len_m": 40.0}   # the ocean cluster rule scaled to 10 m pixels; flag only,
                                                            # provisional (no field in the 2026-11 sample)
FLAGS = ("guarded", "point", "field")
REGISTRY_LEN = {"pad_m": 10.0, "pad_frac": 0.2}   # a registry entry wins over the guard only for an object up to its
                                                  # evidence.length_m + max(pad_m, pad_frac x length_m) (1部の審査
                                                  # 2026-11-06: QJG-002 46 m -> 56 m, under the guard's 60 m)
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
               "（長く係留している艦は構造物と同じく基準でも明るく当日も変化しないため）。理由は fixed_guard。berthed_long＝true",
    "registry": "fix-2：人が光学画像で構造物と確かめた台帳（sat/fixed_registry）の点の radius_m 以内。数えない。項目は fixed_registry_id",
    "registry_over_guard": "fix-2：歯止めに当たるが、台帳の pier_part・mask_error で、当日も上がらず（rise_db 未満）、物体の長さが台帳の"
                           "evidence.length_m＋余裕（REGISTRY_LEN）以下なので外した候補。2部の毎月の見直しの一覧に載る",
    "point": "fix-2：開けた水面の小さく弱いがいつもある点（ブイ・灯標など）。数は変えない。2 撮影以上で同じ点なら台帳の候補",
    "field": "fix-2：開けた水面の小さい候補の密集（筏・網・漁船団）。数は変えない",
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


def counted_base(d: dict) -> bool:
    """Not left out before the fixed decision (berth-1, win-1, anchorage): may anchor naval_near."""
    return not d.get("berth_excluded") and not d.get("window_excluded") and d.get("window") != "anchorage"


def guard_reasons(dets: list[dict], ctx: list[dict], cfg: dict | None = None, objects: list[list[int]] | None = None,
                  lengths: list[float] | None = None, anchor_ok: list[bool] | None = None) -> list[list[str]]:
    """fix-2's guard for every object (default: each candidate alone), structure or not. ``ctx[i]``: {"berth_class":
    naval / civil / unknown / None, "yard_site", "yard_sector", "yard_line": bool} for ``dets[i]`` (lon, lat,
    length_m). ``lengths``: the objects' lengths (default: the longest piece). ``anchor_ok``: per object, may it anchor
    naval_near (default: every piece ``counted_base``)."""
    g = {**GUARD, **(cfg or {})}
    long_m, near_m = float(g["naval_len_m"]), float(g["near_m"])
    objs = objects if objects is not None else [[i] for i in range(len(dets))]
    lens = lengths if lengths is not None else [max(float(dets[i].get("length_m") or 0) for i in o) for o in objs]
    ok = anchor_ok if anchor_ok is not None else [all(counted_base(dets[i]) for i in o) for o in objs]
    naval = [any(ctx[i].get("berth_class") == "naval" for i in o) for o in objs]
    anchors = [k for k in range(len(objs)) if naval[k] and lens[k] >= long_m and ok[k]]
    out = []
    for k, o in enumerate(objs):
        r = []
        if naval[k] and lens[k] >= long_m:
            r.append("naval_long")
        elif naval[k] and any(j != k and any(_dist_m(dets[i], dets[m]) <= near_m for i in o for m in objs[j])
                              for j in anchors):
            r.append("naval_near")
        if any(ctx[i].get("yard_site") for i in o):
            r.append("yard_site")
        if any(ctx[i].get("yard_sector") for i in o):
            r.append("yard_sector")
        if any(ctx[i].get("yard_line") and ctx[i].get("berth_class") == "naval" for i in o):
            r.append("yard_line")
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


def score(dets: list[dict], vv_s: np.ndarray, base: dict | None, wp10: float | None, cfg: dict | None = None,
          registry: list[dict] | None = None, version: str = VERSION) -> list[dict]:
    """Step 1 of fix-2, per candidate (nothing excluded yet): fix-1's ``assess`` with its exclusion taken back into
    ``S``, and the registry entry the candidate's centre falls in (``reg``). Returns one dict per candidate."""
    from . import fixed_registry as freg
    out = []
    for d in dets:
        r = assess(d, vv_s, base, wp10, cfg, version)
        r["S"] = r["fixed_excluded"] == "structure"
        r["fixed_excluded"] = ""
        e = freg.match(registry, d["lon"], d["lat"]) if registry and "lon" in d else None
        r["reg"] = e
        out.append(r)
    return out


def registry_len_limit(e: dict | None) -> float | None:
    """The longest object an entry may drop over the guard: evidence.length_m + max(pad_m, pad_frac x it); None when
    the entry has no length (then the guard always wins)."""
    ln = ((e or {}).get("evidence") or {}).get("length_m")
    if isinstance(ln, bool) or not isinstance(ln, (int, float)) or not ln > 0:
        return None
    return float(ln) + max(REGISTRY_LEN["pad_m"], REGISTRY_LEN["pad_frac"] * float(ln))


def bridge_ok(sc: dict) -> bool:
    """May this candidate join others in the merge of split pieces? Never an S or a registry candidate (design 5)."""
    return not sc.get("S") and not sc.get("reg")


def decide(dets: list[dict], scores: list[dict], ctx: list[dict] | None, cfg: dict | None = None,
           objects: list[list[int]] | None = None, lengths: list[float] | None = None) -> dict:
    """Step 2 of fix-2, per object (``objects``: lists of candidate indices; default each candidate alone; ``lengths``:
    the merged lengths). Writes the fields into every candidate and returns the counts."""
    from . import fixed_registry as freg
    c = {**DEFAULTS, **(cfg or {})}
    n = len(dets)
    ctx = ctx if ctx is not None else [{} for _ in range(n)]
    objs = objects if objects is not None else [[i] for i in range(n)]
    lens = lengths if lengths is not None else [max(float(dets[i].get("length_m") or 0) for i in o) for o in objs]

    def overrides(i, length):  # a confirmed structure kind, nothing new today and no longer than the entry's evidence
        e, r = scores[i]["reg"], scores[i].get("rise_db")
        lim = registry_len_limit(e)
        return (bool(e) and e["kind"] in freg.OVERRIDES_GUARD and r is not None and r < float(c["rise_db"])
                and lim is not None and length <= lim)

    dropped_by_reg = [all(overrides(i, lens[k]) for i in o) for k, o in enumerate(objs)]
    anchor_ok = [all(counted_base(dets[i]) for i in o) and not dropped_by_reg[k] for k, o in enumerate(objs)]
    reasons = guard_reasons(dets, ctx, cfg, objs, lens, anchor_ok)
    counts = {"structure": 0, "registry": 0, "registry_over_guard": 0, "guarded": 0, "point": 0, "field": 0, "high": 0, "medium": 0, "low": 0,
              "no_baseline": 0, "guarded_by": {k: 0 for k in GUARD_REASONS}}
    small = [dets[i].get("zone") == "harbour" for i in range(n)]
    for k, o in enumerate(objs):
        all_reg = all(scores[i]["reg"] for i in o)
        fixed = all(scores[i]["S"] or scores[i]["reg"] for i in o)
        why = reasons[k]
        excl, flag, over = "", "", False
        if fixed:
            if not why or dropped_by_reg[k]:
                excl = "registry" if all_reg else "structure"
                over = bool(why)       # the guard applied and a registry entry won over it
            else:
                flag = "guarded"
        if not excl and not flag and lens[k] <= POINT["max_len_m"] and all(small[i] for i in o):
            bs = [(scores[i].get("base_db"), scores[i].get("spread_db"), scores[i].get("rise_db")) for i in o]
            if all(b is not None and sp is not None and r is not None and b >= POINT["base_db"]
                   and sp < POINT["spread_db"] and r < POINT["rise_db"] for b, sp, r in bs):
                flag = "point"
            elif lens[k] <= FIELD["max_len_m"]:
                near = sum(1 for j in range(n) if small[j] and float(dets[j].get("length_m") or 0) <= FIELD["max_len_m"]
                           and any(_dist_m(dets[i], dets[j]) <= FIELD["radius_m"] for i in o))
                if near >= FIELD["min_n"]:
                    flag = "field"
        for i in o:
            r = {kk: v for kk, v in scores[i].items() if kk not in ("S", "reg")}
            e = scores[i]["reg"]
            r.update(fixed_excluded=excl, fixed_flag=flag, fixed_guard="+".join(why),
                     fixed_guard_class=ctx[i].get("berth_class") or "", fixed_registry_id=e["id"] if e else "",
                     berthed_long=flag == "guarded", registry_over_guard=over)
            if excl:
                r["ship_conf"] = None
            elif r.get("ship_conf") is None:
                r["ship_conf"] = "low"
            dets[i].update(r)
            if excl:
                counts[excl] += 1
                if over:
                    counts["registry_over_guard"] += 1
            else:
                counts[r["ship_conf"]] += 1
                if flag:
                    counts[flag] += 1
                    if flag == "guarded":
                        for kk in why:
                            counts["guarded_by"][kk] += 1
            if r.get("basis", "").startswith("no_baseline"):
                counts["no_baseline"] += 1
    counts["version"] = VERSION
    counts["revision"] = REVISION
    return counts


def classify(dets: list[dict], vv_db: np.ndarray, base: dict | None, meta: dict | None, water: np.ndarray | None,
             cfg: dict | None = None, version: str = FIX1_VERSION, guard_ctx: list[dict] | None = None,
             registry: list[dict] | None = None, registry_hash: str = "") -> dict:
    """Writes the decision into every candidate. fix-1: ``assess`` (counts structure, high, medium, low, no_baseline).
    fix-2: ``score`` then ``decide`` with each candidate as its own object (``guard_ctx``: one dict per candidate, see
    ``guard_reasons``; None = no berth class and no shipyard; ``registry``: the site's active entries)."""
    vv_s = smooth3_db(vv_db)
    wp10 = water_p10(base, meta, water) if base is not None else None
    if version == VERSION:
        counts = decide(dets, score(dets, vv_s, base, wp10, cfg, registry, version), guard_ctx, cfg)
        counts["registry_hash"] = registry_hash
        return counts
    counts = {"structure": 0, "high": 0, "medium": 0, "low": 0, "no_baseline": 0}
    for d in dets:
        r = assess(d, vv_s, base, wp10, cfg, version)
        d.update(r)
        if r["fixed_excluded"]:
            counts["structure"] += 1
        else:
            counts[r["ship_conf"]] += 1
        if r.get("basis", "").startswith("no_baseline"):
            counts["no_baseline"] += 1
    counts["version"] = version
    return counts


def settings(registry_hashes: dict | None = None, registry_errors: dict | None = None) -> dict:
    """meta.json "fixed". ``registry_hashes``: {site: hash of its active registry entries}; ``registry_errors``:
    {site: why its file could not be read} (written only when there is one)."""
    if FIX2_SINCE <= SINCE_UTC:   # fix-1 never applied to a scene: fix-2 from the start
        versions = [{"version": VERSION, "since_utc": FIX2_SINCE, "revision": REVISION,
                     "note": "fix-1 は公開データの撮影に一度も適用されないまま fix-2 に置き換えた（適用期間なし）"}]
    else:
        versions = [{"version": FIX1_VERSION, "since_utc": SINCE_UTC, "until_utc": FIX2_SINCE},
                    {"version": VERSION, "since_utc": FIX2_SINCE, "revision": REVISION}]
    return {"version": VERSION, "revision": REVISION, "since_utc": SINCE_UTC, "defaults": DEFAULTS, "notes": NOTE,
            "labels": CONF_LABEL, "versions": versions,
            "guard": {**GUARD, "yard_roles": list(YARD_ROLES), "reasons": list(GUARD_REASONS),
                      "rule": ("fix-2：structure の条件に当たる候補（または台帳に載った候補）でも、次のどれかに当たれば外さずに数える"
                               "（fixed_flag＝guarded、berthed_long＝true、ship_conf＝low）。naval_long＝最も近い着く線（naval-1 と同じ距離："
                               "岸壁帯は keep_m、開けた水面は窓の band_m）が海軍で、物体の長さ naval_len_m 以上。naval_near＝その線が海軍で、"
                               "同じ撮影の near_m 以内に、数えている（berth-1・win-1 で外れず、台帳で外れない）海軍の naval_len_m 以上の物体がある"
                               "（係留中の艦の割れた片。片をまとめる処理が入るまでの代わり）。yard_site＝地点の kind が yard。"
                               "yard_sector＝造船所の区画（role が yard_roles）。yard_line＝最も近い着く線が海軍の造船所の岸壁"
                               "（sat/berth_class の why が「造船所の岸壁」で区分 naval）。ただし台帳の pier_part・mask_error（人が構造物と確かめた点）"
                               "で、当日も上がっていない（rise_db が defaults の rise_db 未満）うえ、物体の長さが台帳の evidence.length_m＋余裕"
                               "（registry_len）以下のものは歯止めより優先して外す（長く係留する艦は基準でも明るく当日も上がらないので、"
                               "長さで見分ける。外した数は status.json の fixed.registry_over_guard、候補の registry_over_guard＝true）。"
                               "理由 fixed_guard と、"
                               "使った線の区分 fixed_guard_class は、外した候補にも残した候補にも記録する"),
                      "registry_len": {**REGISTRY_LEN, "rule": "evidence.length_m + max(pad_m, pad_frac × evidence.length_m)。"
                                       "evidence.length_m が無い項目は歯止めに勝たない"},
                      "source": "2部の設計書 fixed_eval（2026-11-06）4.2"},
            "registry": {"files": "sat/fixed_registry/<地点>.json", "geojson": "fixed/<地点>.geojson",
                         "match": "候補の中心が項目の radius_m 以内（既定 30 m、上限 100 m）",
                         "active": "status が confirmed で confirmed_by（2部・owner）があり、撮影日が expires_utc より前",
                         "expires_days": {"gear": 120, "raft": 120}, "hashes": registry_hashes or {},
                         "per_scene": "status.json の撮影ごとの fixed.registry_hash（読めなかった日は registry_error）",
                         **({"errors": registry_errors} if registry_errors else {})},
            "point": {**POINT, "rule": "開けた水面（陸マスクから 50 m 以上、zone harbour）で長さ max_len_m 以下、base_db 以上・"
                                       "spread_db 未満・rise_db 未満：ブイ・灯標などの弱いがいつもある点。数は変えない（fixed_flag＝point）"},
            "field": {**FIELD, "rule": "開けた水面の max_len_m 以下の候補が同じ撮影の radius_m 以内に min_n 件以上（自分を含む）："
                                       "筏・網・漁船団。1 撮影では区別できないので数は変えない（fixed_flag＝field）。値は仮置き"},
            "order": ("検出 → win-1 → fix-2 の採点（score：S と台帳の一致。まだ外さない）→ 割れのまとめ（入れる場合。S と台帳の候補は"
                      "つなぎにしない：bridge_ok）→ fix-2 の判定（decide：物体の単位。物体を外すのは片がすべて S か台帳で、"
                      "物体の長さで歯止めにかからないとき）→ size-1（まとめた後の長さ）→ naval-1 の数・quay・est-1・est-2。"
                      "まとめの処理が無いうちは候補 1 つを物体とする"),
            "rule": ("候補の画素（その日の明るい順に pixels 個、3×3 平均）を、同じ軌道の空の岸壁の基準（VV p10・p90）と比べる。"
                     "base_db＝基準の p10 と水面の p10 の差、rise_db＝その日の VV と基準の p10 の差の中央値、rise_z＝画素数で割り引いた z。"
                     "structure・registry の候補は detections/ に fixed_excluded 付きで残し、件数（harbour_candidates・quay_candidates・"
                     "錨泊・区画）に入れない。point・field は印だけで数は変えない。SINCE_UTC より前の行は書き換えない")}
