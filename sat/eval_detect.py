"""Evaluation of ship detectors (2部 sar-literature-plan section 3, steps 1-2; numpy only, no PU).

Step 1, the ground for the comparison:

* loaders for the saved chips: the open-sea raw tiles of the cfar-v2 probe
  (``probe/ocean-cfar-raw-ops3``: ``probe-raw/ocean/<run>/<scene>.npz`` + ``.json``, 12 tiles of
  128 x 128 px at 20 m per scene; set A) and the Zhanjiang Sentinel-1 / Sentinel-2 pairs of the
  quay labels (``probe/sat-s2-labels-ops3``: ``probe-raw/s2-labels/ZJG/``; set B);
* matching of predictions to truth by the Hungarian method (optimal one-to-one assignment) within
  200 m (as xView3) or 100 m. ``linear_sum_assignment`` is implemented here (shortest augmenting
  path, O(n^2 m)): scipy is not installed in the Tests workflow and the collectors do not need it;
  the unit test compares it with scipy only where scipy happens to be installed;
* metrics: precision, recall and F1, false alarms per 1,000 km2, by distance from shore
  (0-1 / 1-2 / 2-5 / > 5 km, each band filtering truth and predictions by their own distance), the
  near-shore F1 as xView3's ``reference/metric.py``: truth within 2 km of shore, predictions within
  2 km + the matching distance, and only the scenes that have both near-shore truth and near-shore
  predictions; detection rate by length band (< 25, 25-50, 50-100, 100-150, >= 150 m) and the length
  performance as xView3's ``compute_length_performance`` (lengths capped at 500 m, mean of
  ``|l_pred - l_true| / l_true`` over the true positives, score ``1 - min(mean, 1)``). Every
  proportion has a Wilson 95% interval;
* low-confidence truth (``likely`` / ``unknown``; たぶん船 / 不明) is neither a hit nor a miss: a
  prediction assigned to it is not counted at all. This is the plan's rule (2部 section 3.3), not
  xView3's: xView3 leaves out only LOW confidence and counts MEDIUM;
* the blind review sheet for the pooled candidates (all methods, merged within 200 m):
  ``vid, scene, lon, lat, sources, label, note`` with ``sources`` left empty (hidden); the key
  (vid -> methods) is written to a separate file.

Step 2, cfar-v2 on set A (108 open-sea tiles) and set B (15 Zhanjiang scenes), with the processing of
main (``cfar2-pr2.1``: VH sigma0 <= 0 has data) and, for comparison, the handling before PR #33 (VH code 0
= no data). The old handling is reproduced exactly by giving ``detect_v2`` a VV that is NaN wherever
VH is NaN (then ``valid``, the no-data band, the VH statistics and the linear VH means are the same as
in ``cfar2-pr2``). Per tile / scene: own-fit failures (VV and VH) and candidates; for set B the
candidates are assigned to the quay sections and the section occupation compared with the labels.

    python -m sat.eval_detect --ocean-raw DIR --zjg-raw DIR --out DIR [--verified CSV]

Sentinel-1 / -2 inputs: "Contains modified Copernicus Sentinel data [year]" (written into the outputs).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

EVAL_VERSION = "eval-detect-1"
MATCH_M = (200.0, 100.0)
SHORE_BANDS = ((0.0, 1.0, "0-1"), (1.0, 2.0, "1-2"), (2.0, 5.0, "2-5"), (5.0, math.inf, ">5"))
NEAR_SHORE_KM = 2.0
FULL_NODATA_MAX = 0.3          # a raw tile with a larger no-data band (pr2.1) lies on the scene border
LENGTH_BANDS = ((0.0, 25.0, "<25"), (25.0, 50.0, "25-50"), (50.0, 100.0, "50-100"),
                (100.0, 150.0, "100-150"), (150.0, math.inf, ">=150"))
POOL_FIELDS = ["vid", "scene", "lon", "lat", "sources", "label", "note"]
Z95 = 1.959963984540054

# truth labels: high confidence counts, low confidence is ignored, "not a ship" is no truth at all
LABELS = {
    "ship": "ship", "船": "ship", "vessel": "ship",
    "likely": "low", "たぶん船": "low", "unknown": "low", "不明": "low", "low": "low",
    "not_ship": "no", "船でない": "no", "no": "no", "": None,
}
MODES = ("pr2.1", "pre33")
HARBOUR_CFG = {   # reference only for set B (narrow strips of saved pixels); not the cfar-v2 setting
    "outer_m": 300.0, "guard_m": 60.0, "min_background_fraction": 0.1,
    "min_tile_water": 0.05, "min_fit_samples": 300,
}


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = Z95) -> dict:
    """Proportion k/n with the Wilson score interval; all None when n is 0."""
    if n <= 0:
        return {"k": int(k), "n": int(n), "p": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "p": round(p, 4), "lo": round(max(0.0, mid - half), 4),
            "hi": round(min(1.0, mid + half), 4)}


def f1_interval(tp: int, fp: int, fn: int, z: float = Z95) -> dict:
    """F1 = 2TP / (2TP + FP + FN) = TP / (TP + (FP + FN) / 2): Wilson interval on that ratio with the
    effective count TP + (FP + FN) / 2 (an approximation; F1 is not a binomial proportion)."""
    n = tp + (fp + fn) / 2
    if n <= 0:
        return {"p": None, "lo": None, "hi": None, "n_eff": 0}
    p = tp / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": round(p, 4), "lo": round(max(0.0, mid - half), 4), "hi": round(min(1.0, mid + half), 4),
            "n_eff": round(n, 1)}


def rate_interval(k: int, area_km2: float, per_km2: float = 1000.0, z: float = Z95) -> dict:
    """Events per ``per_km2`` with the Poisson score interval (the Wilson-type interval for a count)."""
    if not area_km2 or area_km2 <= 0:
        return {"k": int(k), "area_km2": area_km2, "rate": None, "lo": None, "hi": None}
    s = per_km2 / area_km2
    mid = k + z * z / 2
    half = z * math.sqrt(k + z * z / 4)
    return {"k": int(k), "area_km2": round(area_km2, 3), "rate": round(k * s, 3),
            "lo": round(max(0.0, mid - half) * s, 3), "hi": round((mid + half) * s, 3)}


def spearman(x, y) -> float | None:
    """Spearman rank correlation (average ranks for ties)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return None

    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size)
        r[order] = np.arange(a.size, dtype=float)
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


# ---------------------------------------------------------------- Hungarian matching

def linear_sum_assignment(cost) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost one-to-one assignment of a (rectangular) cost matrix, like
    ``scipy.optimize.linear_sum_assignment``: returns (rows, cols), rows ascending, min(n, m) pairs."""
    c = np.asarray(cost, float)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    if not np.isfinite(c).all():
        raise ValueError("cost must be finite")
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T
    n, m = c.shape
    if n == 0:
        e = np.zeros(0, int)
        return e, e.copy()
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)      # p[j]: row (1-based) assigned to column j; column 0 is the root
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            mv = minv[1:]
            mv[upd] = cur[upd]
            wy = way[1:]
            wy[upd] = j0
            cand = np.where(free, mv, np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            uj = np.flatnonzero(used)
            u[p[uj]] += delta
            v[uj] -= delta
            mv[free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.flatnonzero(p[1:])
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order].astype(int), cols[order].astype(int)


def _xy(points: list[dict], lat0: float) -> np.ndarray:
    from .grid import m_per_deg
    mlat, mlon = m_per_deg(lat0)
    return np.array([[float(p["lon"]) * mlon, float(p["lat"]) * mlat] for p in points], float).reshape(-1, 2)


def distances_m(a: list[dict], b: list[dict]) -> np.ndarray:
    """Pairwise ground distances (m) between points with ``lon``/``lat`` (local equirectangular)."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    lat0 = float(np.mean([p["lat"] for p in a] + [p["lat"] for p in b]))
    xa, xb = _xy(a, lat0), _xy(b, lat0)
    return np.hypot(xa[:, None, 0] - xb[None, :, 0], xa[:, None, 1] - xb[None, :, 1])


def match(preds: list[dict], truths: list[dict], tol_m: float = 200.0) -> list[tuple[int, int, float]]:
    """Hungarian assignment within ``tol_m``: (pred index, truth index, distance m). Pairs beyond the
    tolerance cost a large constant, so the number of pairs within it is maximised first and the
    summed distance minimised second. Points are matched only within the same ``scene`` (if given)."""
    out = []
    groups: dict = {}
    for i, p in enumerate(preds):
        groups.setdefault(p.get("scene"), ([], []))[0].append(i)
    for j, t in enumerate(truths):
        groups.setdefault(t.get("scene"), ([], []))[1].append(j)
    for _scene, (pi, ti) in groups.items():
        if not pi or not ti:
            continue
        d = distances_m([preds[i] for i in pi], [truths[j] for j in ti])
        big = (min(len(pi), len(ti)) + 1) * tol_m   # one pair beyond tol costs more than all pairs within it
        cost = np.where(d <= tol_m, d, big)
        r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            if d[a, b] <= tol_m:
                out.append((pi[a], ti[b], float(d[a, b])))
    out.sort()
    return out


# ---------------------------------------------------------------- metrics

def band_of(value, bands) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "unknown"
    for lo, hi, name in bands:
        if lo <= value < hi:
            return name
    return "unknown"


def truth_class(t: dict) -> str | None:
    lab = t.get("label")
    lab = "" if lab is None else str(lab).strip()
    if lab not in LABELS:
        raise ValueError(f"unknown truth label {lab!r}")
    return LABELS[lab]


LENGTH_CAP_M = 500.0


def length_error(pairs: list[tuple[float, float]]) -> dict:
    """xView3 length performance (``compute_length_performance``) over (pred, true) pairs of true
    positives: both lengths capped at 500 m, relative error ``|p - t| / t`` (not capped per pair),
    ``mean``, ``median``, score ``1 - min(mean, 1)``; percentile bootstrap 95% interval of the mean
    (seed 0; the error is not a proportion, so no Wilson interval)."""
    e = np.array([abs(min(p, LENGTH_CAP_M) - min(t, LENGTH_CAP_M)) / min(t, LENGTH_CAP_M)
                  for p, t in pairs if t and t > 0 and p is not None], float)
    if e.size == 0:
        return {"n": 0, "mean": None, "median": None, "score": None, "mean_lo": None, "mean_hi": None}
    rng = np.random.default_rng(0)
    boots = e[rng.integers(0, e.size, (2000, e.size))].mean(axis=1) if e.size > 1 else np.array([e[0]])
    return {"n": int(e.size), "mean": round(float(e.mean()), 4), "median": round(float(np.median(e)), 4),
            "score": round(1 - min(float(e.mean()), 1.0), 4),
            "mean_lo": round(float(np.percentile(boots, 2.5)), 4), "mean_hi": round(float(np.percentile(boots, 97.5)), 4)}


def score(preds: list[dict], truths: list[dict], tol_m: float = 200.0, area_km2: float | None = None) -> dict:
    """Counts and rates of one comparison. ``truths`` with label ship (high), likely/unknown (low,
    ignored) or not_ship (dropped, so a prediction there is a false alarm)."""
    tr = [t for t in truths if truth_class(t) in ("ship", "low")]
    pairs = match(preds, tr, tol_m)
    hit_p, hit_t = set(), set()
    tp = ign = 0
    lens = []
    for i, j, _d in pairs:
        hit_p.add(i)
        hit_t.add(j)
        if truth_class(tr[j]) == "ship":
            tp += 1
            if tr[j].get("length_m") and preds[i].get("length_m") is not None:
                lens.append((float(preds[i]["length_m"]), float(tr[j]["length_m"])))
        else:
            ign += 1
    fp = len(preds) - len(hit_p)
    fn = sum(1 for j, t in enumerate(tr) if truth_class(t) == "ship" and j not in hit_t)
    by_len = {}
    for _lo, _hi, name in LENGTH_BANDS + ((None, None, "unknown"),):
        idx = [j for j, t in enumerate(tr) if truth_class(t) == "ship" and band_of(t.get("length_m"), LENGTH_BANDS) == name]
        if idx:
            by_len[name] = wilson(sum(1 for j in idx if j in hit_t), len(idx))
    res = {"tol_m": tol_m, "tp": tp, "fp": fp, "fn": fn, "ignored_low": ign,
           "precision": wilson(tp, tp + fp), "recall": wilson(tp, tp + fn), "f1": f1_interval(tp, fp, fn),
           "recall_by_length": by_len, "length_error": length_error(lens)}
    if area_km2:
        res["false_alarms_per_1000km2"] = rate_interval(fp, area_km2)
    return res


def near_shore(preds: list[dict], truths: list[dict], tol_m: float = 200.0, near_km: float = NEAR_SHORE_KM) -> dict:
    """xView3 close-to-shore score: truth with ``shore_km <= near_km``, predictions with
    ``shore_km <= near_km + tol_m / 1000`` (a ship just inside may be detected just outside), and only
    the scenes that have at least one of each (the others are left out of this score)."""
    def near(p, lim):
        d = p.get("shore_km")
        return d is not None and d <= lim
    tr = [t for t in truths if near(t, near_km) and truth_class(t) in ("ship", "low")]
    pr = [p for p in preds if near(p, near_km + tol_m / 1000.0)]
    scenes = {t.get("scene") for t in tr} & {p.get("scene") for p in pr}
    res = score([p for p in pr if p.get("scene") in scenes], [t for t in tr if t.get("scene") in scenes], tol_m)
    res["scenes"] = len(scenes)
    res["scenes_left_out"] = len(({t.get("scene") for t in tr} | {p.get("scene") for p in pr}) - scenes)
    return res


def score_by_shore(preds: list[dict], truths: list[dict], tol_m: float = 200.0) -> dict:
    """``score`` within each shore band (truth and predictions filtered by their own distance) and the
    xView3 near-shore score (:func:`near_shore`); points carry ``shore_km`` (None = unknown)."""
    out = {}
    for lo, hi, name in SHORE_BANDS:
        def inb(p):
            d = p.get("shore_km")
            return d is not None and lo <= d < hi
        out[name] = score([p for p in preds if inb(p)], [t for t in truths if inb(t)], tol_m)
    out["near_shore_0-2"] = near_shore(preds, truths, tol_m)
    unk = [p for p in preds if p.get("shore_km") is None]
    if unk or any(t.get("shore_km") is None for t in truths):
        out["unknown"] = score(unk, [t for t in truths if t.get("shore_km") is None], tol_m)
    return out


def shore_km(rows, cols, land: np.ndarray, pixel_m, coarse: list[tuple] | None = None) -> list[dict]:
    """Distance (km) from each pixel position to the nearest land pixel of ``land``; with ``coarse``
    (rectangles (r0, c0, r1, c1) in the same pixel frame that contain land somewhere outside ``land``)
    the distance to the nearest rectangle is a lower bound. Returns [{"km", "exact"}]; km None when
    nothing is known."""
    mr, mc = (float(pixel_m[0]), float(pixel_m[1])) if np.ndim(pixel_m) else (float(pixel_m),) * 2
    land = np.asarray(land, bool)
    if land.any():
        # only land pixels with a non-land 4-neighbour can be nearest
        pad = np.pad(land, 1, constant_values=True)
        inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        lr, lc = np.nonzero(land & ~inner)
    else:
        lr = lc = np.zeros(0, int)
    out = []
    for r, c in zip(np.atleast_1d(rows), np.atleast_1d(cols)):
        best, exact = math.inf, False
        if lr.size:
            d = float(np.sqrt(((lr - r) * mr) ** 2 + ((lc - c) * mc) ** 2).min()) / 1000
            best, exact = d, True
        for r0, c0, r1, c1 in coarse or ():
            dr = max(r0 - r, 0, r - r1)
            dc = max(c0 - c, 0, c - c1)
            d = math.hypot(dr * mr, dc * mc) / 1000
            if d < best:
                best, exact = d, False
        out.append({"km": None if not math.isfinite(best) else round(best, 3), "exact": exact})
    return out


# ---------------------------------------------------------------- pooled blind review sheet

def pool(cands: dict[str, list[dict]], tol_m: float = 200.0, salt: str = "") -> tuple[list[dict], dict]:
    """Merge the candidates of all methods (``{method: [{scene, lon, lat}, ...]}``) within ``tol_m``
    (single linkage, per scene). Returns (rows for :data:`POOL_FIELDS` with sources hidden, key
    {vid: {"sources": [...], "n": ...}}). The order of the vids is a hash of position (and ``salt``),
    so it does not tell the method either."""
    pts = [(m, p) for m in sorted(cands) for p in cands[m]]
    by_scene: dict = {}
    for k, (_m, p) in enumerate(pts):
        by_scene.setdefault(p.get("scene"), []).append(k)
    clusters = []
    for scene, idx in by_scene.items():
        parent = list(range(len(idx)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        d = distances_m([pts[k][1] for k in idx], [pts[k][1] for k in idx])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if d[a, b] <= tol_m:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        groups: dict = {}
        for a in range(len(idx)):
            groups.setdefault(find(a), []).append(idx[a])
        for members in groups.values():
            lon = float(np.mean([pts[k][1]["lon"] for k in members]))
            lat = float(np.mean([pts[k][1]["lat"] for k in members]))
            clusters.append({"scene": scene, "lon": round(lon, 6), "lat": round(lat, 6),
                             "sources": sorted({pts[k][0] for k in members}), "n": len(members)})
    clusters.sort(key=lambda c: hashlib.sha1(f"{salt}|{c['scene']}|{c['lon']:.6f}|{c['lat']:.6f}".encode()).hexdigest())
    rows, key = [], {}
    for i, c in enumerate(clusters, 1):
        vid = f"p{i:04d}"
        rows.append({"vid": vid, "scene": c["scene"], "lon": c["lon"], "lat": c["lat"],
                     "sources": "", "label": "", "note": ""})
        key[vid] = {"sources": c["sources"], "n": c["n"]}
    return rows, key


def write_pool_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POOL_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in POOL_FIELDS})


def read_judged(path: Path) -> list[dict]:
    """The review sheet back as truth points (rows with an empty label are left out)."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("label") or "").strip() == "":
                continue
            truth_class(r)
            out.append({"vid": r["vid"], "scene": r["scene"], "lon": float(r["lon"]), "lat": float(r["lat"]),
                        "label": r["label"].strip(), "note": r.get("note", "")})
    return out


# ---------------------------------------------------------------- loaders

def _db(code: np.ndarray, dm: np.ndarray) -> np.ndarray:
    from .cdse import DB_OFFSET, DB_SCALE
    code = np.asarray(code, np.float64)
    return np.where(dm & (code > 0), code / DB_SCALE - DB_OFFSET, np.nan)


def load_ocean_raw(root: Path) -> list[dict]:
    """Scenes of the raw probe (set A): every ``<scene>.npz`` with its ``.json`` under ``root``
    (searched recursively). Each tile: vv/vh dB (NaN for code 0), lia (deg), datamask, land, static,
    its grid, and the zero-code shares."""
    from .cdse import LIA_SCALE
    from .grid import Grid
    scenes = []
    for npz in sorted(Path(root).rglob("*.npz")):
        meta_p = npz.with_suffix(".json")
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        req = meta["request"]
        w, s, e, n = req["bbox_epsg4326"]
        W, H = int(req["width"]), int(req["height"])
        with np.load(npz) as z:
            a = {k: z[k] for k in z.files}
        tiles = []
        for k, (ty, tx, r0, c0) in enumerate(a["tiles"].tolist()):
            dm = a["datamask"][k] > 0
            th, tw = a["vv"][k].shape
            g = Grid(w + c0 / W * (e - w), n - (r0 + th) / H * (n - s), w + (c0 + tw) / W * (e - w), n - r0 / H * (n - s), tw, th)
            land = a["land"][k] > 0
            sea = dm & ~land
            vh_code, vv_code = a["vh"][k], a["vv"][k]
            tiles.append({
                "ty": int(ty), "tx": int(tx), "r0": int(r0), "c0": int(c0), "grid": g,
                "vv": _db(vv_code, dm), "vh": _db(vh_code, dm),
                "lia": np.where(dm, a["lia"][k] / LIA_SCALE, np.nan), "datamask": dm, "land": land,
                "static": a["static"][k] > 0,
                "vh_zero": round(float((sea & (vh_code == 0)).sum() / max(sea.sum(), 1)), 4),
                "vv_zero": round(float((sea & (vv_code == 0)).sum() / max(sea.sum(), 1)), 4),
            })
        acq = meta.get("acquisition") or {}
        scenes.append({"scene": f"{meta.get('area')}-{acq.get('acquired_utc')}", "file": npz.name, "area": meta.get("area"),
                       "acquired_utc": acq.get("acquired_utc"), "orbit_direction": acq.get("orbit_direction"),
                       "pixel_m": tuple(req["pixel_m"]), "scene_shape": (H, W), "tiles": tiles, "meta": meta})
    return scenes


def load_zjg_pairs(root: Path) -> list[dict]:
    """Set B: one entry per Sentinel-2 day (``s1-for-<date>.json``) with its Sentinel-1 chip (vv/vh dB,
    float16; NaN outside the pixels the labels read, and NaN where VH sigma0 <= 0) and the S2 arrays."""
    root = Path(root)
    out = []
    for mp in sorted(root.glob("s1-for-*.json")):
        date = mp.stem[len("s1-for-"):]
        a = json.loads(mp.read_text(encoding="utf-8"))
        if not a:
            continue
        stamp = a["acquired_utc"].replace(":", "-")
        p1 = root / f"s1-{stamp}.npz"
        if not p1.exists():
            continue
        with np.load(p1) as z:
            vv, vh = z["vv"].astype(np.float64), z["vh"].astype(np.float64)
        rec = {"date": date, "s1": a, "s1_file": p1.name, "vv": vv, "vh": vh, "keep": np.isfinite(vv)}
        p2 = root / f"s2-{date}.npz"
        if p2.exists():
            with np.load(p2) as z:
                rec["s2"] = {k: z[k] for k in z.files}
        out.append(rec)
    return out


# ---------------------------------------------------------------- step 2: cfar-v2 runs

def mode_inputs(vv: np.ndarray, vh: np.ndarray, datamask, mode: str):
    """(vv, vh, datamask) for ``detect_v2`` in ``mode``: ``pr2.1`` as main; ``pre33`` = VH code 0 is no
    data (VV set to NaN there; exact reproduction of cfar2-pr2's ``valid``)."""
    if mode == "pr2.1":
        return vv, vh, datamask
    if mode == "pre33":
        return np.where(np.isfinite(vh), vv, np.nan), vh, None
    raise ValueError(mode)


def run_v2(vv, vh, land, pixel_m, mode: str, cfg: dict | None = None, **kw):
    from . import detect2 as v2
    a, b, dm = mode_inputs(vv, vh, kw.pop("datamask", None), mode)
    t0 = time.perf_counter()
    dets, info = v2.detect_v2(a, b, land, pixel_m, cfg, datamask=dm, **kw)
    sec = time.perf_counter() - t0
    tt = info["tile_table"] if "tile_table" in info else []
    return dets, info, sec, tt


def _own_failed(tile_table: list[dict]) -> int:
    return sum(1 for t in tile_table if t.get("own_model") is None)


def ocean_runs(scenes: list[dict], modes=MODES) -> tuple[list[dict], dict]:
    """cfar-v2 on every tile of set A, alone (12 tiles per scene, independently). Returns (rows,
    {mode: [candidate dicts with lon/lat]})."""
    from . import detect2 as v2
    rows, cands = [], {m: [] for m in modes}
    for sc in scenes:
        H, W = sc["scene_shape"]
        lat = (sc["tiles"][0]["grid"].south + sc["tiles"][0]["grid"].north) / 2 if sc["tiles"] else 0
        heading = v2.platform_heading(sc["orbit_direction"].lower(), lat) if sc.get("orbit_direction") else None
        meta_tiles = {(r[0], r[1]): dict(zip(sc["meta"].get("all_tiles_cols", []), r)) for r in sc["meta"].get("all_tiles", [])}
        coarse_land = []
        for (ty, tx), r in meta_tiles.items():
            if (r.get("land") or 0) > 0:
                coarse_land.append((ty, tx))
        for t in sc["tiles"]:
            g = t["grid"]
            th, tw = t["vv"].shape
            # land tiles of the scene (128 px) in this tile's pixel frame, except the tile itself
            coarse = [((ty * 128) - t["r0"], (tx * 128) - t["c0"], (ty * 128) - t["r0"] + 127, (tx * 128) - t["c0"] + 127)
                      for ty, tx in coarse_land if (ty, tx) != (t["ty"], t["tx"])]
            mt = meta_tiles.get((t["ty"], t["tx"]), {})
            for mode in modes:
                dets, info, sec, tt = run_v2(t["vv"], t["vh"], t["land"], sc["pixel_m"], mode,
                                             datamask=t["datamask"], lia=t["lia"], heading_deg=heading, static=t["static"])
                k = v2.kept(dets)
                v2.add_geo(k, g, sc["acquired_utc"])
                edge_px = 0.1 * th
                sd = shore_km([d["row"] for d in k], [d["col"] for d in k], t["land"], sc["pixel_m"], coarse)
                for d, s in zip(k, sd):
                    cands[mode].append({
                        "scene": sc["scene"], "lon": d["lon"], "lat": d["lat"], "mode": mode, "ty": t["ty"], "tx": t["tx"],
                        "confidence": d["confidence"], "length_m": d.get("length_m"), "width_m": d.get("width_m"),
                        "large": d.get("large"), "static": bool(t["static"][min(int(d["row"]), th - 1), min(int(d["col"]), tw - 1)]),
                        "edge10": bool(min(d["row"], d["col"], th - 1 - d["row"], tw - 1 - d["col"]) < edge_px),
                        "shore_km": s["km"], "shore_exact": s["exact"], "sea_state": d.get("sea_state"),
                        "acquired_utc": sc["acquired_utc"]})
                vh_fit = info.get("model_counts_vh") or {}
                rows.append({
                    "scene": sc["scene"], "area": sc["area"], "ty": t["ty"], "tx": t["tx"], "mode": mode,
                    "vh_zero": t["vh_zero"], "vv_zero": t["vv_zero"],
                    "nodata_band": info.get("nodata_band_fraction"),
                    "vv_own_fit_failed": _own_failed(tt), "tiles_untested": info.get("tiles_untested"),
                    "vh_untested": info.get("tiles_untested_vh"), "vh_models": sum(vh_fit.values()),
                    "vv_model": (tt[0].get("model") if tt else None), "usable": (tt[0].get("usable_fraction") if tt else None),
                    "candidates": len(k), "A": sum(d["confidence"] == "A" for d in k),
                    "B": sum(d["confidence"] == "B" for d in k),
                    "edge10": sum(1 for c in cands[mode][len(cands[mode]) - len(k):] if c["edge10"]),
                    "scene_fit_pre33": mt.get("own_model") is not None if mt else None,
                    "scene_usable_pre33": mt.get("usable"), "seconds": round(sec, 2),
                })
    return rows, cands


def scene_table(rows: list[dict], scenes: list[dict]) -> list[dict]:
    """Per scene: VH code-0 share (12 tiles, and all scene tiles from the probe meta), fit failures of
    the 12 tiles under both modes, the scene-level failures recorded by the probe (cfar2-pr2), candidates."""
    out = []
    for sc in scenes:
        r = [x for x in rows if x["scene"] == sc["scene"]]
        at = [dict(zip(sc["meta"].get("all_tiles_cols", []), t)) for t in sc["meta"].get("all_tiles", [])]
        sea_at = [t for t in at if (t.get("datamask") or 0) > 0.5]
        rec = {"scene": sc["scene"], "area": sc["area"], "tiles": len(sc["tiles"]),
               "vh_zero_12": round(float(np.mean([t["vh_zero"] for t in sc["tiles"]])), 4) if sc["tiles"] else None,
               "vh_zero_scene": round(float(np.mean([t["vh_zero"] for t in sea_at])), 4) if sea_at else None,
               "scene_tiles": len(at), "scene_failed_pre33": sum(1 for t in at if t.get("own_model") is None),
               "vv_median_db": round(float(np.median([t["vv_med_db"] for t in sea_at if t.get("vv_med_db") is not None])), 2) if sea_at else None}
        for m in MODES:
            rm = [x for x in r if x["mode"] == m]
            rec[f"failed_{m}"] = sum(x["vv_own_fit_failed"] for x in rm)
            rec[f"vh_untested_{m}"] = sum(x["vh_untested"] or 0 for x in rm)
            rec[f"cand_{m}"] = sum(x["candidates"] for x in rm)
            rec[f"A_{m}"] = sum(x["A"] for x in rm)
        out.append(rec)
    return out


def hypothesis(rows: list[dict], scenes_tab: list[dict]) -> dict:
    """Does the VH code-0 share go with fit failures? Tile level (12 x 9 tiles) by VH zero bins, both
    modes, and scene level (Spearman over the scenes)."""
    bins = ((0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 0.10, "5-10%"), (0.10, 1.01, ">=10%"))
    # true no-data band of each tile (pr2.1: datamask / VV only); >= FULL_NODATA_MAX = a tile at the
    # scene's border (half empty), where a lone 128 px tile cannot be fitted whatever VH does
    nod = {(x["scene"], x["ty"], x["tx"]): x["nodata_band"] or 0.0 for x in rows if x["mode"] == "pr2.1"}
    tile = {}
    for part, keep in (("all", lambda x: True), ("full", lambda x: nod.get((x["scene"], x["ty"], x["tx"]), 0) < FULL_NODATA_MAX),
                       ("border", lambda x: nod.get((x["scene"], x["ty"], x["tx"]), 0) >= FULL_NODATA_MAX)):
        tp = {}
        for lo, hi, name in bins:
            rec = {}
            for m in MODES:
                rm = [x for x in rows if x["mode"] == m and lo <= x["vh_zero"] < hi and keep(x)]
                rec[m] = wilson(sum(x["vv_own_fit_failed"] for x in rm), len(rm))
            tp[name] = rec
        tile[part] = tp
    scene = {}
    for m in MODES:
        scene[m] = spearman([s["vh_zero_12"] for s in scenes_tab], [s[f"failed_{m}"] for s in scenes_tab])
    scene["scene_meta_pre33"] = spearman([s["vh_zero_scene"] or 0 for s in scenes_tab],
                                         [s["scene_failed_pre33"] / max(s["scene_tiles"], 1) for s in scenes_tab])
    return {"tile_failed_by_vh_zero": tile, "spearman_scene_vh_zero_vs_failed": scene,
            "tiles_failed_pre33_fitted_pr2.1": sum(
                1 for a in rows if a["mode"] == "pre33" and a["vv_own_fit_failed"]
                for b in rows if b["mode"] == "pr2.1" and (b["scene"], b["ty"], b["tx"]) == (a["scene"], a["ty"], a["tx"])
                and not b["vv_own_fit_failed"])}


# ---------------------------------------------------------------- set B: quay sections

def load_sections(path: Path) -> list[dict]:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"section_id": f["properties"]["section_id"], "sector": f["properties"].get("sector"),
             "line": f["geometry"]["coordinates"]} for f in fc["features"]]


def assign_sections(points: list[dict], sections: list[dict], max_m: float = 130.0) -> list[str | None]:
    """Nearest section (distance to its line, m) within ``max_m`` for each point, else None."""
    from .grid import m_per_deg
    if not points or not sections:
        return [None] * len(points)
    lat0 = float(np.mean([p["lat"] for p in points]))
    mlat, mlon = m_per_deg(lat0)
    out = []
    segs = []
    for si, s in enumerate(sections):
        xy = np.array(s["line"], float) * [mlon, mlat]
        if len(xy) == 1:
            xy = np.vstack([xy, xy])
        for a, b in zip(xy[:-1], xy[1:]):
            segs.append((si, a, b))
    A = np.array([s[1] for s in segs])
    B = np.array([s[2] for s in segs])
    sid = np.array([s[0] for s in segs])
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    for p in points:
        P = np.array([p["lon"] * mlon, p["lat"] * mlat])
        tpar = np.clip(((P - A) * AB).sum(1) / L2, 0, 1)
        d = np.hypot(*(A + tpar[:, None] * AB - P).T)
        k = int(np.argmin(d))
        out.append(sections[sid[k]]["section_id"] if d[k] <= max_m else None)
    return out


def zjg_truth(labels_csv: Path, verified_csv: Path | None = None) -> dict:
    """{(section_id, S2 date): {"occupied": 0/1/None, "source": verified|auto|unknown}}. Priority: the
    verified answers (``verified`` 1: ``verified_occupied``; 0: unknown, ignored), then the labels'
    own verified columns, then the automatic label (status ok only)."""
    truth = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["section_id"], r["date"])
            if r.get("verified") == "1" and r.get("verified_occupied") not in ("", None):
                truth[key] = {"occupied": int(r["verified_occupied"]), "source": "verified"}
            elif r.get("verified") == "0":
                truth[key] = {"occupied": None, "source": "unknown"}
            elif r["status"] == "ok" and r.get("occupied") not in ("", None):
                truth[key] = {"occupied": int(r["occupied"]), "source": "auto"}
    if verified_csv and Path(verified_csv).exists():
        with open(verified_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["section_id"], r["date"])
                if r.get("verified") == "1" and r.get("verified_occupied") not in ("", None):
                    truth[key] = {"occupied": int(r["verified_occupied"]), "source": "verified"}
                elif r.get("verified") == "0":
                    truth[key] = {"occupied": None, "source": "unknown"}
    return truth


def occupancy_compare(pred_occ: dict, truth: dict) -> dict:
    """Section occupation (prediction: any candidate assigned) against the truth; Wilson intervals."""
    out = {}
    for src in ("all", "verified", "auto"):
        tp = fp = fn = tn = 0
        for key, t in truth.items():
            if t["occupied"] is None or key not in pred_occ or (src != "all" and t["source"] != src):
                continue
            p = pred_occ[key]
            if p and t["occupied"]:
                tp += 1
            elif p:
                fp += 1
            elif t["occupied"]:
                fn += 1
            else:
                tn += 1
        n = tp + fp + fn + tn
        out[src] = {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "agreement": wilson(tp + tn, n),
                    "precision": wilson(tp, tp + fp), "recall": wilson(tp, tp + fn)}
    return out


def zjg_runs(pairs: list[dict], site: dict, land: np.ndarray, sections: list[dict], truth: dict,
             settings=(("default", None), ("harbour_ref", HARBOUR_CFG)), modes=MODES, conf=("A",)) -> tuple[list[dict], dict, list[dict]]:
    """cfar-v2 on each Zhanjiang S1 chip (once per scene), candidates assigned to the sections.
    ``default``: the cfar-v2 setting, pixels not saved = no data. ``harbour_ref``: reference only --
    pixels not saved count as land (so their border is not dilated) and the windows are shrunk
    (:data:`HARBOUR_CFG`)."""
    from . import detect2 as v2
    from .grid import Grid
    grid = Grid.for_site(site)
    rows, comp, feats = [], {}, []
    by_s1: dict = {}
    for p in pairs:
        by_s1.setdefault(p["s1_file"], []).append(p)
    days_all = {p["date"] for p in pairs}
    for sname, cfg in settings:
        for mode in modes:
            pred_occ = {}
            for s1f, ps in sorted(by_s1.items()):
                p0 = ps[0]
                vv, vh, keep = p0["vv"], p0["vh"], p0["keep"]
                if sname == "default":
                    lnd = land
                    vvi, vhi = vv, vh
                else:
                    lnd = land | ~keep
                    vvi = np.where(keep, vv, -30.0)
                    vhi = np.where(keep, vh, -35.0)
                dets, info, sec, tt = run_v2(vvi, vhi, lnd, grid.pixel_m, mode, cfg)
                k = [d for d in v2.kept(dets) if d["confidence"] in conf]
                v2.add_geo(k, grid, p0["s1"]["acquired_utc"])
                secs = assign_sections(k, sections)
                occ = {s for s in secs if s}
                sea = keep & ~land
                for p in ps:
                    for s in sections:
                        pred_occ[(s["section_id"], p["date"])] = int(s["section_id"] in occ)
                for d, s in zip(k, secs):
                    feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
                                  "properties": {"set": "B", "setting": sname, "mode": mode, "scene": p0["s1"]["acquired_utc"],
                                                 "confidence": d["confidence"], "length_m": d.get("length_m"),
                                                 "section_id": s}})
                rows.append({"setting": sname, "mode": mode, "s1": p0["s1"]["acquired_utc"],
                             "s2_days": ";".join(p["date"] for p in ps),
                             "vh_zero": round(float((sea & ~np.isfinite(vh)).sum() / max(sea.sum(), 1)), 4),
                             "tiles": len(tt), "vv_own_fit_failed": _own_failed(tt), "tiles_untested": info.get("tiles_untested"),
                             "candidates": len(k), "assigned": sum(1 for s in secs if s), "sections_occupied": len(occ),
                             "seconds": round(sec, 2)})
            comp[f"{sname}/{mode}"] = occupancy_compare(pred_occ, {k_: v for k_, v in truth.items() if k_[1] in days_all})
    return rows, comp, feats


# ---------------------------------------------------------------- output

def attribution(times) -> str:
    years = sorted({str(t)[:4] for t in times if t})
    return f"Contains modified Copernicus Sentinel data {', '.join(years)}" if years else "Contains modified Copernicus Sentinel data"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _fc(feats: list[dict], attr: str) -> str:
    return json.dumps({"type": "FeatureCollection", "attribution": attr, "features": feats},
                      ensure_ascii=False, separators=(",", ":"))


def check_split(out: Path, key_out: Path) -> None:
    """The blind sheet and its key must not end up in one place (one folder, or one folder inside the
    other): whoever judges the sheet is given ``out`` only."""
    a, b = Path(out).resolve(), Path(key_out).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise SystemExit("--key-out は --out と別の場所に（入れ子も不可）。鍵と手法別の候補は判定が済むまで判定者に渡さない")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ocean-raw", type=Path, help="probe-raw/ocean (set A)")
    ap.add_argument("--zjg-raw", type=Path, help="probe-raw/s2-labels/ZJG (set B)")
    ap.add_argument("--labels", type=Path, default=Path("docs/data/sat/labels/ZJG-s2-labels.csv"))
    ap.add_argument("--sections", type=Path, default=Path("docs/data/sat/labels/ZJG-s2-sections.geojson"))
    ap.add_argument("--verified", type=Path, help="verified answers CSV (section_id, date, verified, verified_occupied)")
    ap.add_argument("--out", type=Path, required=True,
                    help="reviewer side: the blind sheet (pool-*.csv), the tables and summary.json only")
    ap.add_argument("--key-out", type=Path, required=True,
                    help="kept from the reviewer until the review is done: the pool key (vid -> methods) and the "
                         "per-method candidate files; must not be inside --out (or contain it)")
    args = ap.parse_args(argv)
    check_split(args.out, args.key_out)
    args.out.mkdir(parents=True, exist_ok=True)
    args.key_out.mkdir(parents=True, exist_ok=True)
    summary = {"version": EVAL_VERSION, "modes": {"pr2.1": "main (VH sigma0 <= 0 has data)",
                                                    "pre33": "before PR #33 (VH code 0 = no data)"}}
    times = []
    if args.ocean_raw:
        scenes = load_ocean_raw(args.ocean_raw)
        times += [s["acquired_utc"] for s in scenes]
        rows, cands = ocean_runs(scenes)
        tab = scene_table(rows, scenes)
        _write_csv(args.out / "ocean_tiles.csv", rows)
        _write_csv(args.out / "ocean_scenes.csv", tab)
        summary["ocean"] = {"scenes": len(scenes), "tiles": sum(len(s["tiles"]) for s in scenes),
                            "hypothesis": hypothesis(rows, tab),
                            "candidates": {m: {"all": len(c), "A": sum(x["confidence"] == "A" for x in c),
                                               "edge10": sum(x["edge10"] for x in c),
                                               "by_shore": {b: sum(1 for x in c if band_of(x["shore_km"], SHORE_BANDS) == b)
                                                            for b in [s[2] for s in SHORE_BANDS] + ["unknown"]}}
                                           for m, c in cands.items()}}
        attr = attribution(times)
        for m, c in cands.items():
            feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [x["lon"], x["lat"]]},
                      "properties": {k: v for k, v in x.items() if k not in ("lon", "lat")}} for x in c]
            (args.key_out / f"candidates-ocean-{m}.geojson").write_text(_fc(feats, attr), encoding="utf-8")
        prow, key = pool({f"cfar-v2/{m}": c for m, c in cands.items()}, salt=EVAL_VERSION)
        write_pool_csv(args.out / "pool-ocean.csv", prow)
        (args.key_out / "pool-ocean-key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
        summary["ocean"]["pool_rows"] = len(prow)
    if args.zjg_raw:
        from .osm_mask import WATER, load_mask
        sites = json.loads(Path(__file__).with_name("sites.json").read_text(encoding="utf-8"))["sites"]
        site = next(s for s in sites if s["id"] == "ZJG")
        mask, _ = load_mask("ZJG", int(site.get("mask_version", 1)))
        pairs = load_zjg_pairs(args.zjg_raw)
        times += [p["s1"]["acquired_utc"] for p in pairs]
        truth = zjg_truth(args.labels, args.verified)
        rows, comp, feats = zjg_runs(pairs, site, mask != WATER, load_sections(args.sections), truth)
        _write_csv(args.out / "zjg_scenes.csv", rows)
        (args.key_out / "candidates-zjg.geojson").write_text(_fc(feats, attribution(times)), encoding="utf-8")
        summary["zjg"] = {"s2_days": len(pairs), "s1_scenes": len({p["s1_file"] for p in pairs}), "occupancy": comp,
                          "truth_sources": {s: sum(1 for k, v in truth.items() if v["source"] == s and k[1] in {p["date"] for p in pairs})
                                            for s in ("verified", "auto", "unknown")}}
    summary["attribution"] = attribution(times)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
