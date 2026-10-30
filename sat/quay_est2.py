"""Warship-likely ships alongside (est-2, trial, not calibrated): ships abreast and the warship odds.

Runs after the quay count on what is already stored (quay_counts.csv, quay/YYYY-MM.geojson and the
Sentinel-2 files under labels/); it fetches nothing (no PU). est-1 (sat/quay_est.py) stays as it is;
est-2 is a separate output next to it (quay_est2.csv, quay_est2.json).

For every occupied stretch of quay-v1 in a scene:

1. Columns abreast c in {1, 2, 3} (hidden). Prior by place: sector A pier 0.5/0.35/0.15, sector A
   coastline and sector B 0.85/0.13/0.02.
2. The occupied depth (depth_m) is N(mu[face, orbit][c], 12 m); a depth at the depth limit of the
   profile (120 m, 90 m before feat-1) is censored (P(D >= limit)). On a face toward the radar the
   outer double-bounce line (db_line_m) is N(22 / 42 / 62 m, 10 m) when it is there. The face
   (toward / along / away) is the stretch's face_look (feat-1), or else it is worked out from the
   normal of the Sentinel-2 sections (labels/<site>-s2-sections.geojson, the same OSM lines) and the
   look azimuth of the orbit. The toward means depend on the incidence angle (45 deg: orbit 157,
   32 deg: orbit 84 at Zhanjiang; other angles are interpolated).
3. Scenes of an orbit pair (pair_id, within 36 h) multiply the likelihoods of the same face seen from
   the other orbit (ships arriving or leaving in between are ignored).
4. Smear: a stretch on a face away from the radar, with the opposite side of the same pier occupied
   in the same scene, is a real ship with probability 0.3 (else 0 ships).
5. Warship odds of each ship: logit P = logit pi(place, length) + 0.5 * kappa * ln LR_S2. Length:
   (stretch length - 20 m - gaps) / k pieces, spread N(., 20 m) over <60 / 60-100 / >=100 m.
   LR_S2 from the hull colour of the nearest clear Sentinel-2 day (grey 3, colour or contrast 0.2,
   mixed 1), weakened by kappa from the time difference (<=8 h 0.85, <=24 h 0.6, <=48 h 0.4, more 0.25).
   No Sentinel-2 day: 0. SAR features (dVV, dVH) are recorded only. AIS: not used (m_AIS = 0).
6. Per stretch: ships = c x k, warships ~ Binomial(c x k, P); a sector and the site are the
   convolution (stretches independent). Median and the model's 80 % interval (p10-p90).

All numbers above are placeholders (HQ's 1-D simulation and settings, 2026-10-30), not calibrated.
The Sentinel-2 labels (#5, automatic, unverified) are not an input for the counts (occupied, c) and
not a truth either; only their hull colour class is used as the colour clue.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EST2_VERSION = "est-2"
STATUS = "仮置き・較正前（司令部の 1 次元模擬と仮置き値。正解データでの較正は #7）"

C_VALUES = (1, 2, 3)
PRIOR_C = {"A_pier": (0.5, 0.35, 0.15), "A_shore": (0.85, 0.13, 0.02), "B": (0.85, 0.13, 0.02)}
DEPTH_SD_M = 12.0
DEPTH_MU = {  # mean occupied depth (m) for c = 1, 2, 3
    "away": (35.0, 55.0, 75.0),
    "along": (40.0, 60.0, 80.0),
}
TOWARD_MU = {45.0: (45.0, 65.0, 95.0), 32.0: (55.0, 75.0, 95.0)}  # by incidence angle (deg)
INCIDENCE_BY_ORBIT = {"ZJG": {"157": 45.0, "84": 32.0}}
DB_MU = (22.0, 42.0, 62.0)  # outer double-bounce line off the water's edge (toward faces only)
DB_SD_M = 10.0
DEPTH_LIMIT_M = 120.0       # feat-1 profile (0-120 m)
DEPTH_LIMIT_OLD_M = 90.0    # before feat-1
FACE_COS = 0.5
SMEAR_P_REAL = 0.3
SMEAR_MAX_M = 120.0         # opposite side of the pier: display lines this close
PAIR_MATCH_M = 30.0         # the same face in the partner scene: points this close
SECTION_MATCH_M = 20.0
LENGTH_TRIM_M = 20.0
GAP_M = 25.0
LEN_SD_M = 20.0
LEN_BINS = (60.0, 100.0)
PI = {  # warship share by place and length (<60, 60-100, >=100 m)
    "A_pier": (0.4, 0.7, 0.9),
    "A_shore": (0.3, 0.5, 0.7),
    "B": (0.1, 0.2, 0.3),
}
S2_LR = {"grey": 3.0, "colour": 0.2, "contrast": 0.2, "mixed": 1.0}
S2_WEIGHT = 0.5
KAPPA_STEPS = ((8.0, 0.85), (24.0, 0.6), (48.0, 0.4))
KAPPA_FAR = 0.25
S2_MAX_H = 96.0             # the S2 day within this of the S1 time (the S1 window of #5, 4 days)
PLACE_NAMES = {"A_pier": "区画 A の桟橋", "A_shore": "区画 A の海岸線", "B": "区画 B"}
FACE_NAMES = {"toward": "レーダー向き", "along": "沿う", "away": "反対向き"}
CLUE_NAMES = {
    "depth": "占有の奥行き（S1）",
    "depth_censored": "奥行きが上限に達した（以上として扱う）",
    "db_line": "外側の二回反射の線（S1、レーダー向きの面）",
    "pair": "軌道の組のもう一方の撮影",
    "face_feature": "面の向き（feat-1 の記録）",
    "face_sections": "面の向き（S2 区間の法線から計算）",
    "smear": "反対側の占有によるにじみの割り引き",
    "s2_colour": "S2 の船体の色",
    "place": "区画と岸壁の種類",
    "length": "占有の長さ",
}
CSV_FIELDS = [
    "site_id", "acquired_utc", "relative_orbit", "mask_version", "est2_version", "in_sample", "pair_id",
    "segments", "warship_median", "warship_p10", "warship_p90", "warship_mean",
    "a_median", "a_p10", "a_p90", "b_median", "b_p10", "b_p90",
    "ships_median", "ships_p10", "ships_p90", "c_mean",
    "used_depth", "used_db_line", "used_pair", "used_s2_colour", "smear_discounted", "face_from_sections", "no_face",
    "est1_warship_mode", "est1_warship_p10", "est1_warship_p90",
]
FREEZE_UTC = "2026-10-27"   # the S2 labels before this day only (later scenes are frozen as out-of-sample)


# --- small helpers --------------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x: float, mu: float, sd: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def _quant(p: np.ndarray, q: float) -> int:
    c = np.cumsum(p)
    return int(np.searchsorted(c, q * c[-1] - 1e-12))


def dist_stats(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    p = p / p.sum()
    x = np.arange(len(p))
    return {"median": _quant(p, 0.5), "p10": _quant(p, 0.1), "p90": _quant(p, 0.9), "mean": round(float((x * p).sum()), 1)}


def _t(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def _xy(coords, lat0: float) -> np.ndarray:
    a = np.asarray(coords, float).reshape(-1, 2)
    return np.stack([a[:, 0] * 111320.0 * math.cos(math.radians(lat0)), a[:, 1] * 110540.0], 1)


def _min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every point of a, the distance to the nearest point of b."""
    if not len(a) or not len(b):
        return np.full(len(a), np.inf)
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)


def kappa(dt_h: float | None) -> float:
    """Probability that the ship of the Sentinel-2 day is still there at the Sentinel-1 time."""
    if dt_h is None:
        return 0.0
    for lim, k in KAPPA_STEPS:
        if abs(dt_h) <= lim:
            return k
    return KAPPA_FAR


def place_of(sector, kind) -> str:
    if sector == "A":
        return "A_pier" if kind == "pier" else "A_shore"
    return "B"


def toward_mu(incidence: float | None) -> tuple:
    if incidence is None:
        return tuple((a + b) / 2 for a, b in zip(TOWARD_MU[45.0], TOWARD_MU[32.0]))
    lo, hi = 32.0, 45.0
    w = min(1.0, max(0.0, (incidence - lo) / (hi - lo)))
    return tuple((1 - w) * a + w * b for a, b in zip(TOWARD_MU[lo], TOWARD_MU[hi]))


def depth_mu(face: str, incidence: float | None) -> tuple | None:
    if face == "toward":
        return toward_mu(incidence)
    return DEPTH_MU.get(face)


def face_class(cos_phi: float) -> str:
    return "toward" if cos_phi > FACE_COS else "away" if cos_phi < -FACE_COS else "along"


def look_azimuth(orbit_direction, lat):
    from . import quay
    return quay.look_azimuth(orbit_direction, lat)


# --- likelihoods ----------------------------------------------------------------------------------
def depth_like(depth, face: str, incidence, limit: float) -> tuple[np.ndarray, str]:
    """Likelihood of the occupied depth for c = 1, 2, 3 and the clue used ("" when none)."""
    mu = depth_mu(face, incidence)
    if depth is None or mu is None:
        return np.ones(3), ""
    d = float(depth)
    if d >= limit - 2.5:  # reached the end of the profile: D >= limit
        return np.array([1 - _ncdf((limit - m) / DEPTH_SD_M) for m in mu]) + 1e-12, "depth_censored"
    return np.array([_npdf(d, m, DEPTH_SD_M) for m in mu]) + 1e-300, "depth"


def db_like(db_line, face: str) -> tuple[np.ndarray, str]:
    if face != "toward" or db_line in (None, ""):
        return np.ones(3), ""
    x = float(db_line)
    return np.array([_npdf(x, m, DB_SD_M) for m in DB_MU]) + 1e-300, "db_line"


def warship_p(place: str, length_m: float, ln_lr: float, kap: float) -> float:
    """P(warship) of one ship: logit pi(place, length) + 0.5 kappa ln LR, mixed over the length bins."""
    pis = PI[place]
    edges = (-math.inf,) + LEN_BINS + (math.inf,)
    out = 0.0
    for b in range(3):
        lo, hi = edges[b], edges[b + 1]
        m = (_ncdf((hi - length_m) / LEN_SD_M) if hi < math.inf else 1.0) - (_ncdf((lo - length_m) / LEN_SD_M) if lo > -math.inf else 0.0)
        z = math.log(pis[b] / (1 - pis[b])) + S2_WEIGHT * kap * ln_lr
        out += m / (1 + math.exp(-z))
    return out


def c_posterior(place: str, likes: list[np.ndarray]) -> np.ndarray:
    p = np.array(PRIOR_C[place], float)
    for lk in likes:
        p = p * lk
    s = p.sum()
    return p / s if s > 0 else np.array(PRIOR_C[place], float)


def _binom(n: int, p: float) -> np.ndarray:
    return np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)])


def stretch_dists(pc: np.ndarray, k: int, p_war: float, p_real: float) -> tuple[np.ndarray, np.ndarray]:
    """(P(ships), P(warships)) of one stretch: c x k ships, each a warship with p_war."""
    n_max = 3 * k
    ships, war = np.zeros(n_max + 1), np.zeros(n_max + 1)
    for c, w in zip(C_VALUES, pc):
        ships[c * k] += w * p_real
        b = _binom(c * k, p_war)
        war[:len(b)] += w * p_real * b
    ships[0] += 1 - p_real
    war[0] += 1 - p_real
    return ships, war


# --- inputs ---------------------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_sections(out: Path, site_id: str) -> list[dict]:
    p = out / "labels" / f"{site_id}-s2-sections.geojson"
    if not p.exists():
        return []
    return [{**f["properties"], "coords": f["geometry"]["coordinates"]} for f in json.loads(p.read_text(encoding="utf-8"))["features"]]


def load_colours(out: Path, site_id: str) -> dict[str, list[tuple[str, str]]]:
    """{section_id: [(S2 time, colour class)]} from the S2 labels: the hull colour only (status ok)."""
    out_: dict = {}
    for r in read_csv(out / "labels" / f"{site_id}-s2-labels.csv"):
        if r.get("status") != "ok" or not r.get("color_class") or r.get("date", "") >= FREEZE_UTC:
            continue
        out_.setdefault(r["section_id"], []).append((r["acquired_utc"], r["color_class"]))
    return out_


def load_features(out: Path) -> list[dict]:
    feats = []
    for p in sorted((out / "quay").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].geojson")):
        for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
            feats.append({**f["properties"], "coords": f["geometry"]["coordinates"]})
    return feats


def _side(f: dict) -> int:
    face = str(f.get("face") or "")
    return -1 if face.endswith("-") else 1


# --- the estimate ---------------------------------------------------------------------------------
class Site:
    """Per-site lookups (sections and their normals, S2 colours)."""

    def __init__(self, site_id: str, sections: list[dict], colours: dict, lat0: float = 21.2):
        self.site_id, self.lat0 = site_id, lat0
        self.sections = sections
        for s in sections:
            s["xy"] = _xy(s["coords"], lat0)
        self.colours = colours

    def match_sections(self, seg: dict, xy: np.ndarray) -> list[tuple[dict, int]]:
        """Sections on the same way and side with points near the stretch: [(section, points)]."""
        out = []
        for s in self.sections:
            if s.get("way_id") != seg.get("osm_way") or int(s.get("side", 1)) != _side(seg):
                continue
            n = int((_min_dist(xy, s["xy"]) <= SECTION_MATCH_M).sum())
            if n:
                out.append((s, n))
        return sorted(out, key=lambda x: -x[1])

    def colour(self, secs: list[tuple[dict, int]], t_s1: str):
        """(ln LR, kappa, S2 time, class) of the nearest S2 day with a colour on the stretch's sections."""
        best = None
        for s, _n in secs:
            for t2, cls in self.colours.get(s["section_id"], []):
                dt = (_t(t_s1) - _t(t2)).total_seconds() / 3600.0
                if abs(dt) > S2_MAX_H:
                    continue
                if best is None or abs(dt) < abs(best[0]):
                    best = (dt, t2, cls)
        if best is None:
            return 0.0, 0.0, None, None, None
        dt, t2, cls = best
        return math.log(S2_LR.get(cls, 1.0)), kappa(dt), t2, cls, round(dt, 1)


def face_of(seg: dict, site: Site, xy: np.ndarray, look_az) -> tuple[str, str, list]:
    secs = site.match_sections(seg, xy) if site.sections else []
    if seg.get("face_look"):
        return seg["face_look"], "face_feature", secs
    if secs and look_az is not None:
        w = np.array([n for _s, n in secs], float)
        cs = np.array([math.cos(math.radians(float(s["normal_az_deg"]) - (look_az + 180.0))) for s, _n in secs])
        return face_class(float((w * cs).sum() / w.sum())), "face_sections", secs
    return "", "", secs


def estimate_scene(row: dict, segs: list[dict], site: Site, partner: tuple[dict, list[dict]] | None = None,
                   incidence=None) -> dict:
    """est-2 for one scene. ``segs``: the scene's quay features; ``partner``: (row, features) of the
    other scene of the orbit pair. Returns totals, per-stretch details and the clues used."""
    lat0 = site.lat0
    look = row.get("look_az_deg")
    look = float(look) if look not in (None, "") else look_azimuth(row.get("orbit_direction") or "ascending", lat0)
    orbit = str(row.get("relative_orbit"))
    inc = incidence if incidence is not None else INCIDENCE_BY_ORBIT.get(row["site_id"], {}).get(orbit)

    def prep(fs, lk):
        out = []
        for f in fs:
            xy = _xy(f["coords"], lat0)
            face, how, secs = face_of(f, site, xy, lk)
            out.append({"f": f, "xy": xy, "face": face, "how": how, "secs": secs})
        return out

    items = prep(segs, look)
    pitems = []
    p_inc = None
    if partner is not None:
        prow, pfs = partner
        pl = prow.get("look_az_deg")
        pl = float(pl) if pl not in (None, "") else look_azimuth(prow.get("orbit_direction") or "ascending", lat0)
        pitems = prep(pfs, pl)
        p_inc = INCIDENCE_BY_ORBIT.get(prow["site_id"], {}).get(str(prow.get("relative_orbit")))
    sectors: dict = {"A": [np.ones(1), np.ones(1)], "B": [np.ones(1), np.ones(1)]}
    tot_s, tot_w = np.ones(1), np.ones(1)
    details = []
    clues_count: dict = {}
    for it in items:
        f, face = it["f"], it["face"]
        place = place_of(f.get("sector"), f.get("line_kind"))
        limit = DEPTH_LIMIT_M if f.get("features_version") else DEPTH_LIMIT_OLD_M
        clues = ["place", "length"]
        if it["how"]:
            clues.append(it["how"])
        likes = []
        lk, cl = depth_like(f.get("depth_m"), face, inc, limit)
        likes.append(lk)
        if cl:
            clues.append(cl)
        lk, cl = db_like(f.get("db_line_m"), face)
        likes.append(lk)
        if cl:
            clues.append(cl)
        # the same face from the other orbit of the pair
        best = None
        for pi in pitems:
            pf = pi["f"]
            if pf.get("osm_way") != f.get("osm_way") or _side(pf) != _side(f):
                continue
            n = int((_min_dist(it["xy"], pi["xy"]) <= PAIR_MATCH_M).sum())
            if n and (best is None or n > best[0]):
                best = (n, pi)
        if best is not None:
            pi = best[1]
            pf = pi["f"]
            plim = DEPTH_LIMIT_M if pf.get("features_version") else DEPTH_LIMIT_OLD_M
            pface = pi["face"] or face
            l1, c1 = depth_like(pf.get("depth_m"), pface, p_inc, plim)
            l2, c2 = db_like(pf.get("db_line_m"), pface)
            if c1 or c2:
                likes += [l1, l2]
                clues.append("pair")
        pc = c_posterior(place, likes)
        # smear: a face away from the radar with the other side of the same pier occupied
        p_real = 1.0
        if face == "away" and f.get("line_kind") == "pier":
            for o in items:
                of = o["f"]
                # the opposite face of the pier: a face toward the radar on the same OSM way (a closed
                # pier outline has every face on one side of the way, so the side alone does not tell)
                if of is f or of.get("osm_way") != f.get("osm_way") or o["face"] != "toward":
                    continue
                if float(_min_dist(it["xy"], o["xy"]).min()) <= SMEAR_MAX_M:
                    p_real = SMEAR_P_REAL
                    clues.append("smear")
                    break
        k = max(1, int(f.get("min") or 1))
        length = max(20.0, (float(f.get("length_m") or 0) - LENGTH_TRIM_M - (k - 1) * GAP_M) / k)
        ln_lr, kap, t2, ccls, dt = site.colour(it["secs"], row["acquired_utc"])
        if t2 is not None:
            clues.append("s2_colour")
        pw = warship_p(place, length, ln_lr, kap)
        ps, pwar = stretch_dists(pc, k, pw, p_real)
        tot_s, tot_w = np.convolve(tot_s, ps), np.convolve(tot_w, pwar)
        sec = f.get("sector") if f.get("sector") in sectors else "B"
        sectors[sec][0] = np.convolve(sectors[sec][0], ps)
        sectors[sec][1] = np.convolve(sectors[sec][1], pwar)
        for c in set(clues):
            clues_count[c] = clues_count.get(c, 0) + 1
        sw = dist_stats(pwar)
        details.append({
            "n": f.get("n"), "sector": f.get("sector"), "line_kind": f.get("line_kind"), "osm_way": f.get("osm_way"),
            "side": _side(f), "place": place, "face": face or None, "face_from": it["how"] or None,
            "length_m": f.get("length_m"), "k": k, "ship_length_m": round(length), "depth_m": f.get("depth_m"),
            "db_line_m": f.get("db_line_m"),
            "p_c": [round(float(x), 3) for x in pc], "c_map": int(np.argmax(pc)) + 1,
            "c_mean": round(float((pc * np.array(C_VALUES)).sum()), 2),
            "p_real": p_real, "p_warship": round(pw, 3),
            "s2": ({"acquired_utc": t2, "colour": ccls, "dt_h": dt, "kappa": kap} if t2 else None),
            "warship_median": sw["median"], "warship_p10": sw["p10"], "warship_p90": sw["p90"],
            "sar_recorded": {"peak_dvv_db": f.get("peak_dvv_db"), "peak_dvh_db": f.get("peak_dvh_db")},
            "clues": sorted(set(clues)),
            "est1_n_mode": f.get("n_mode"), "est1_warship_mode": f.get("warship_mode"),
        })
    return {
        "warship": dist_stats(tot_w), "ships": dist_stats(tot_s),
        "by_sector": {s: {"warship": dist_stats(v[1]), "ships": dist_stats(v[0])} for s, v in sectors.items()},
        "c_mean": round(float(np.mean([d["c_mean"] for d in details])), 2) if details else None,
        "clues": clues_count, "stretches": details,
        "warship_dist": [round(float(x), 4) for x in tot_w / tot_w.sum()],
    }


def _mask_tag(r: dict) -> str:
    return r.get("mask_version") or ""


def pairs_of(rows: list[dict]) -> dict:
    """pair_id per (site, time, mask) from the rows, or the same rule as sat/run.py when empty."""
    if any(r.get("pair_id") for r in rows):
        return {(r["site_id"], r["acquired_utc"], _mask_tag(r)): r["pair_id"] for r in rows if r.get("pair_id")}
    from .run import orbit_pairs
    return orbit_pairs(rows)


def build(out: Path, quay_rows: list[dict] | None = None, ship_rows: list[dict] | None = None,
          pairs: dict | None = None) -> dict:
    """Recompute est-2 for every quay scene from stored files; write quay_est2.csv and quay_est2.json."""
    rows = quay_rows if quay_rows is not None else read_csv(out / "quay_counts.csv")
    ship_rows = ship_rows if ship_rows is not None else read_csv(out / "ship_counts.csv")
    inc_of = {(r["site_id"], r["acquired_utc"]): r.get("incidence_deg") for r in ship_rows}
    pairs = pairs if pairs is not None else (pairs_of(rows) if rows else {})
    feats = load_features(out)
    by_scene: dict = {}
    for f in feats:
        by_scene.setdefault((f["site_id"], f["acquired_utc"], f.get("mask_version") or ""), []).append(f)
    sites: dict = {}
    out_rows, scenes = [], []
    rows = sorted(rows, key=lambda r: (r["site_id"], r["acquired_utc"]))
    by_pid: dict = {}
    for r in rows:
        pid = pairs.get((r["site_id"], r["acquired_utc"], _mask_tag(r))) or ""
        if pid:
            by_pid.setdefault(pid, []).append(r)
    for r in rows:
        sid = r["site_id"]
        if sid not in sites:
            sites[sid] = Site(sid, load_sections(out, sid), load_colours(out, sid))
        key = (sid, r["acquired_utc"], _mask_tag(r))
        segs = by_scene.get(key, [])
        pid = pairs.get(key) or ""
        partner = None
        for pr in by_pid.get(pid, []):
            if pr["acquired_utc"] != r["acquired_utc"]:
                partner = (pr, by_scene.get((sid, pr["acquired_utc"], _mask_tag(pr)), []))
        inc = inc_of.get((sid, r["acquired_utc"]))
        try:
            inc = float(inc) if inc not in (None, "") else None
        except ValueError:
            inc = None
        e = estimate_scene(r, segs, sites[sid], partner, inc)
        cl = e["clues"]
        st = [d for d in e["stretches"]]
        out_rows.append({
            "site_id": sid, "acquired_utc": r["acquired_utc"], "relative_orbit": r.get("relative_orbit"),
            "mask_version": r.get("mask_version"), "est2_version": EST2_VERSION, "in_sample": r.get("in_sample", ""),
            "pair_id": pid, "segments": len(segs),
            "warship_median": e["warship"]["median"], "warship_p10": e["warship"]["p10"], "warship_p90": e["warship"]["p90"],
            "warship_mean": e["warship"]["mean"],
            "a_median": e["by_sector"]["A"]["warship"]["median"], "a_p10": e["by_sector"]["A"]["warship"]["p10"],
            "a_p90": e["by_sector"]["A"]["warship"]["p90"],
            "b_median": e["by_sector"]["B"]["warship"]["median"], "b_p10": e["by_sector"]["B"]["warship"]["p10"],
            "b_p90": e["by_sector"]["B"]["warship"]["p90"],
            "ships_median": e["ships"]["median"], "ships_p10": e["ships"]["p10"], "ships_p90": e["ships"]["p90"],
            "c_mean": e["c_mean"] if e["c_mean"] is not None else "",
            "used_depth": cl.get("depth", 0) + cl.get("depth_censored", 0), "used_db_line": cl.get("db_line", 0),
            "used_pair": cl.get("pair", 0), "used_s2_colour": cl.get("s2_colour", 0), "smear_discounted": cl.get("smear", 0),
            "face_from_sections": cl.get("face_sections", 0), "no_face": sum(1 for d in st if not d["face"]),
            "est1_warship_mode": r.get("est_warship_mode", ""), "est1_warship_p10": r.get("est_warship_p10", ""),
            "est1_warship_p90": r.get("est_warship_p90", ""),
        })
        scenes.append({"site_id": sid, "acquired_utc": r["acquired_utc"], "relative_orbit": r.get("relative_orbit"),
                       "pair_id": pid or None, "incidence_deg": inc, "in_sample": r.get("in_sample", ""),
                       "warship": e["warship"], "ships": e["ships"], "by_sector": e["by_sector"],
                       "warship_dist": e["warship_dist"],
                       "clues": {CLUE_NAMES.get(k, k): v for k, v in sorted(cl.items())},
                       "est1": {"all_mode": r.get("est_mode"), "warship_mode": r.get("est_warship_mode"),
                                "warship_p10": r.get("est_warship_p10"), "warship_p90": r.get("est_warship_p90"),
                                "a_mode": r.get("est_a_mode"), "b_mode": r.get("est_b_mode")},
                       "stretches": e["stretches"]})
    write_csv(out / "quay_est2.csv", out_rows, CSV_FIELDS)
    doc = {"version": EST2_VERSION, "status": STATUS, "settings": settings(), "scenes": scenes}
    (out / "quay_est2.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"rows": out_rows, "scenes": scenes}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


# --- check against the S2 labels (not a calibration) ----------------------------------------------
def label_check(out: Path, site_id: str = "ZJG") -> dict:
    """Columns abreast of est-2 (and est-1's cols_hint) against the automatic S2 labels' c, on the
    S1 re-fetch of #5 (labels/<site>-s2-s1-match.csv): S2 days before FREEZE_UTC, status ok, both
    occupied. A check only: the labels are automatic and unverified, and nothing is tuned on them."""
    rows = [r for r in read_csv(out / "labels" / f"{site_id}-s2-s1-match.csv")
            if r.get("s2_date", "") < FREEZE_UTC and r.get("s2_status") == "ok" and r.get("s1_occupied") == "1"]
    secs = {x["section_id"]: _xy(x["coords"], 21.2) for x in load_sections(out, site_id)}
    toward_occ: dict = {}
    for r in rows:
        if r.get("face_look") == "toward":
            toward_occ.setdefault((r["s1_acquired_utc"], r["section_id"].split(":")[0]), []).append(r["section_id"])
    tab = {f: {} for f in ("all", "toward", "along", "away")}
    tab_e1: dict = {}
    s2_empty: dict = {}
    n_used = 0
    for r in rows:
        face = r.get("face_look") or ""
        place = place_of(r.get("sector"), r.get("kind"))
        inc = INCIDENCE_BY_ORBIT.get(site_id, {}).get(str(r.get("relative_orbit")))
        d = r.get("s1_depth_m")
        l1, _ = depth_like(float(d) if d else None, face, inc, DEPTH_LIMIT_M)
        l2, _ = db_like(r.get("s1_db_line_m") or None, face)
        pc = c_posterior(place, [l1, l2])
        c2 = int(np.argmax(pc)) + 1
        way = r["section_id"].split(":")[0]
        smear = face == "away" and r.get("kind") == "pier" and any(
            r["section_id"] in secs and o in secs and float(_min_dist(secs[r["section_id"]], secs[o]).min()) <= SMEAR_MAX_M
            for o in toward_occ.get((r["s1_acquired_utc"], way), []))
        if r.get("s2_occupied") != "1":
            k = "smear" if smear else "no_smear"
            s2_empty[k] = s2_empty.get(k, 0) + 1
            continue
        s2c = r.get("s2_c") or ""
        if s2c not in ("1", "2", "3"):
            continue
        n_used += 1
        for f in ("all", face):
            if f in tab:
                tab[f].setdefault(s2c, {}).setdefault(str(c2), 0)
                tab[f][s2c][str(c2)] += 1
        h = r.get("s1_cols_hint") or "-"
        tab_e1.setdefault(s2c, {}).setdefault(h, 0)
        tab_e1[s2c][h] += 1

    def agree(t):
        n = sum(sum(v.values()) for v in t.values())
        a = sum(t.get(c, {}).get(c, 0) for c in ("1", "2", "3"))
        return {"n": n, "agree": a, "share": round(a / n, 3) if n else None}

    ones = sum(tab["all"].get("1", {}).values())
    return {"rows": n_used, "always_one": {"n": n_used, "agree": ones, "share": round(ones / n_used, 3) if n_used else None},
            "est2": tab, "est2_agree": {f: agree(t) for f, t in tab.items()},
            "est1_cols_hint": tab_e1, "est1_agree": agree(tab_e1), "s1_occupied_s2_empty": s2_empty,
            "note": "S2 ラベルは自動・未確認。確認のための表で、較正には使っていない"}


# --- description ----------------------------------------------------------------------------------
LIMITS = [
    "値はすべて仮置き・較正前（司令部の 1 次元模擬と仮置きの値）。正解データ（人の目で確認したラベル）での較正は #7 で行う",
    "隻数の 80% 区間はモデル内の区間（このモデルと仮置き値が正しいとしたときの p10〜p90）で、実数がこの区間に入る確率ではない",
    "「軍艦の可能性が高い船」は区画・長さ・S2 の色から引いた隻数のモデル上の分布で、艦種・所属は分からない。民間船・作業船も確率つきで入る",
    "AIS は使っていない（m_AIS＝0）：湛江の軍港は aisstream の受信網の外で、試行で 0 隻だった。軍艦は AIS を出さないことも多い",
    "S2 の色は自動判定（#5、未確認）の船体の色だけを使い、時刻差に応じて κ で弱める。前後 96 時間以内に S2 の晴れの日が無い撮影・2026-10-27 以降の S2 の日の色は 0（使わない）",
    "S2 ラベルの占有・列数（#5、自動・未確認）は推定の入力にも正解にも使っていない（列数の一致表は確認のためだけ）",
    "SAR の明るさ（ΔVV・ΔVH）は記録だけで判定に使っていない",
    "軌道の組（36 時間以内）の間の入出港は無いものとしている",
    "区間どうしは独立と仮定。est-1 の「60 m 以内のまとまり」のにじみの割り引きは入れていない（反対向きの面のにじみだけ割り引く）",
    "数えられない区間（常時占有・OSM に無い岸壁）は est-1 と同じく入らない",
]


def settings() -> dict:
    return {
        "version": EST2_VERSION,
        "status": STATUS,
        "inputs": "quay_counts.csv と quay/YYYY-MM.geojson（取得済みの S1）、labels/ の S2 区間の線と船体の色（取得済み）。新たな取得はしない（PU 0）",
        "columns_abreast": {
            "c": "列数 c∈{1,2,3}（横抱き）を隠れ変数とする",
            "prior": {PLACE_NAMES[k]: list(v) for k, v in PRIOR_C.items()},
            "depth_mean_m": {**{FACE_NAMES[k]: list(v) for k, v in DEPTH_MU.items()},
                              "レーダー向き・軌道 157（入射約 45°）": list(TOWARD_MU[45.0]),
                              "レーダー向き・軌道 84（入射約 32°）": list(TOWARD_MU[32.0])},
            "depth_sd_m": DEPTH_SD_M,
            "depth_limit_m": f"{DEPTH_LIMIT_M:g}（feat-1 の前の記録は {DEPTH_LIMIT_OLD_M:g}）。上限に達した奥行きは「以上」として扱う",
            "db_line_mean_m": list(DB_MU), "db_line_sd_m": DB_SD_M,
            "db_line_note": "外側の二回反射の線はレーダー向きの面だけ使う。無ければ使わない",
            "pair": "軌道の組（pair_id）があれば、同じ面のもう一方の撮影の尤度も掛ける",
            "smear": (f"反対向きの面の占有で、同じ桟橋（同じ OSM の線）のレーダー向きの面が {SMEAR_MAX_M:g} m 以内で同じ撮影で占有のとき、"
                      f"本当に船である確率を {SMEAR_P_REAL:g} とする（それ以外は 0 隻）"),
            "face": "feat-1 の face_look。無ければ S2 区間（同じ OSM の線）の法線とレーダーの見る方位から cosφ を出し、|cosφ|＞0.5 で向き・反対向き",
        },
        "warship": {
            "formula": "logit P＝logit π(区画, 長さ)＋0.5·κ·ln LR_S2（欠けた手がかりは 0）",
            "pi": {PLACE_NAMES[k]: dict(zip(("60 m 未満", "60〜100 m", "100 m 以上"), v)) for k, v in PI.items()},
            "length": f"1 隻の長さ＝(区間長−{LENGTH_TRIM_M:g} m−(k−1)×{GAP_M:g} m)／k（k＝quay-v1 の片の数）、N(・, {LEN_SD_M:g} m) で 3 階級に分ける",
            "s2_lr": S2_LR,
            "kappa": {"8 時間以内": 0.85, "24 時間以内": 0.6, "48 時間以内": 0.4, f"それ以上（{S2_MAX_H:g} 時間まで）": KAPPA_FAR},
            "s2_window": f"S1 の撮影の前後 {S2_MAX_H:g} 時間（#5 の突き合わせと同じ 4 日）以内の最も近い S2 の晴れの日。それより離れた日の色は使わない",
            "sar": "記録だけ", "ais": "使わない（m_AIS＝0）",
        },
        "count": "区間ごとに c×k 隻、各隻が確率 P で軍艦。区画・地点は畳み込み。中央値とモデル内 80% 区間（p10〜p90）",
        "limits": LIMITS,
        "accuracy": "2026-10-27 以降の撮影を標本外として凍結。列数の混同行列・80% 区間の的中率・ブライアスコアは #7（人の目で確認したラベル約 100 区間）",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="est-2 from the stored quay files (no fetch)")
    ap.add_argument("--out", default="docs/data/sat")
    ap.add_argument("--check", action="store_true", help="print the column check against the S2 labels")
    a = ap.parse_args(argv)
    out = Path(a.out)
    r = build(out)
    for x in r["rows"]:
        print(x["acquired_utc"], x["relative_orbit"], "A", x["a_median"], f"({x['a_p10']}-{x['a_p90']})",
              "B", x["b_median"], f"({x['b_p10']}-{x['b_p90']})", "total", x["warship_median"],
              f"({x['warship_p10']}-{x['warship_p90']})", file=sys.stderr)
    if a.check:
        print(json.dumps(label_check(out), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
