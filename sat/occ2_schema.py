"""occ-2: columns, version, settings and the pure-Python parts of the quay presence test (no numpy).

The statistics are in sat/occ2.py (numpy). This module is what the collect workflows (no numpy) and the evaluation
(sat/eval_presence.py, pure Python) read: the column list of the scene x section rows, the settings, the CSV reading
and writing, the calibration formula and the blind mixing.

Design: 設計書_occ-2（岸壁の在不在の統計検定）2026-11-08 (2部). occ-2 is a separate series beside quay-v2 / est-2:
docs/data/sat/quay_occ2/<site>/YYYY-MM.csv (scene x section rows), quay_occ2/<site>/stack.csv (the section stack, occ2-5)
and quay_occ2.json (settings and the per-scene pool statistics). Zhanjiang is never written (sites.json quay.occ2 false).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

OCC_VERSION = "occ-2"
SUBDIR = "quay_occ2"

# the settings (design §1-§3, §8); the keys are the names used in the design
DEFAULTS = {
    "pool_inner_m": 30.0,          # §1.2 pool: inner band 0-30 m off the edge of the non-blind naval sections
    "pool_min_sections": 10,       # fewer non-blind sections: the normal approximation from the samples (§1.3-3), below pool_min_samples: no test
    "pool_min_samples": 50,
    "w_vh": 0.5,                   # §1.2 z = z_VV + w_vh * z_VH
    "enl_default_vv": 16.0,        # §1.1 3x3-mean ENL on open water when the scene has too little water to measure
    "enl_default_vh": 7.0,
    "enl_far_px": 8,               # open water: at least this many pixels (80 m) from land / structures
    "sigma_db_min": 0.5,           # never below this (dB) whatever the ENL says
    "delta_water_margin_db": 1.0,  # §8 Delta >= (open-water lift) - this
    "face_corr_max_db": 2.0,       # §8 per-face offset of the pool's position, shrunk by n / (n + face_corr_k) and clipped
    "face_corr_k": 8.0,
    "face_corr_min_sections": 4,
    "depth_m": (5.0, 25.0),        # §1.3-1 depth offsets of s_i: 5 .. 25 m (two distinct pixels of 10 m)
    "depth_toward_m": (5.0, 35.0), # a face toward the radar: the double-bounce line lies outside (est-2's face depths)
    "depth_mode": "min2",          # min2: min over the depth pixels; w15: 0.5 z(5 m) + 0.5 z(15 m) (§8, compared in occ2-2)
    "window_m": 50.0,              # §1.3-2 the shortest ship: five 10 m samples
    "overlap_m": 20.0,             # §3.2 a stretch / window counts for a section when it covers at least this of it
    "seed_p": 1e-3,                # §1.3-4 the seed: one sample of the window at p_px < seed_p under the pool's null of s
    "on_p": 0.01,                  # §3.2 a sample is part of a stretch when its s_i is above the null's 1 - on_p quantile
    "gap_m": 20.0,                 # gaps shorter than this inside a stretch are joined (as quay.max_gap_m)
    "stretch_min_m": 20.0,
    "blind_p10_db": -8.0,          # §2.3 the current blind rule (inner p10 above this) is kept ...
    "blind_ref_pct": 90.0,         # ... and a section whose inner p10 median is above the p90 of the same-face, same-kind non-blind sections is blind too
    "blind_ref_min": 5,            # a face x kind group this small borrows the same-face group, then all non-blind sections
    "blind_prior_default": 0.5,    # §2.3 blind_prior without a label history
    "lambda_blind": 0.5,           # §2.3 p_occ = (1 - lambda) p_occ(test) + lambda blind_prior
    "ridge_search_m": (-40.0, 30.0),  # §3.1 across the OSM line, land side negative, every 5 m
    "ridge_step_m": 5.0,
    "ridge_rel_db": 3.0,           # the ridge is this much above the value ridge_check_m waterward
    "ridge_check_m": 20.0,
    "ridge_after_m": 5.0,          # the band starts this far waterward of the ridge (its first depth pixel is centred 10 m off it)
    "ridge_shift_min_m": 15.0,     # the ridge moves the band only when its start is this far landward of the mask's edge (type A')
    "ridge_min_m": -25.0,          # a ridge further inland than this is a building, not the quay's edge
    "ridge_max_m": 20.0,
    "allowed_dilate_px": 2.5,      # §3.1 the band may enter the land mask this far (Geometry uses 1.5)
    "calib": {"a": None, "b": 1.0, "c": 0.0, "prior": 0.5, "source": "none"},  # §1.3-5 logit p_occ = a + b (-log10 p_test) + c ln(len / 50)
    "min_section_m": 50.0,
}

# §4.2 the scene x section row
FIELDS = [
    "site_id", "acquired_utc", "relative_orbit", "orbit_direction", "section_id", "kind", "face",
    "occ_version", "p_test", "p_occ", "t_stat", "z_seed_ok",
    "stretch_len_m", "stretch_depth_m",
    "blind", "blind_prior", "ref_kind",
    "edge_offset_m", "edge_source",
    "delta_vv_db", "delta_vh_db", "sigma_vv_db", "sigma_vh_db", "enl_vv", "enl_vh", "pool_n",
    "calib_source", "in_sample",
]
KEY = ("site_id", "acquired_utc", "section_id")
FLOAT_FIELDS = ("p_test", "p_occ", "t_stat", "stretch_len_m", "stretch_depth_m", "blind_prior", "edge_offset_m",
                "delta_vv_db", "delta_vh_db", "sigma_vv_db", "sigma_vh_db", "enl_vv", "enl_vh")
INT_FIELDS = ("relative_orbit", "z_seed_ok", "blind", "pool_n")

# the sections of the inspection (設計_同日の正解と港の誤り検品 §1.2) that only A-1 (the line map) can fix: A = line geometry
# (a phantom line, pixels taken by a line 2 m away), D = boat basins and small jetties. Marked in the evaluation, never used
# in the test.
A1_TARGETS = {
    "QGZ": {"1564921421:R:00": "A", "219848837:L:07": "A", "582517902:L:01": "D", "582517902:R:00": "D", "15387629:L:06": "D"},
}


def settings(cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    return {"version": OCC_VERSION, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}}


def a1_target(site_id: str, section_id: str) -> str:
    return A1_TARGETS.get(site_id, {}).get(section_id, "")


# ---------------------------------------------------------------- probability of occupation (§1.3-5, §2.3)
def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    x = max(min(x, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def p_occ_from_test(p_test: float | None, stretch_len_m: float | None, calib: dict | None = None,
                    blind: bool = False, blind_prior: float | None = None, lam: float | None = None) -> float | None:
    """logit p = a + b (-log10 p_test) + c ln(max(len, 10) / 50); a = logit(prior) when the calibration has no a.
    A blind section mixes with its prior: (1 - lambda) p + lambda blind_prior. None without a test."""
    if p_test is None:
        return None
    c = {**DEFAULTS["calib"], **(calib or {})}
    a = c["a"] if c.get("a") is not None else logit(float(c.get("prior", 0.5)))
    x = -math.log10(max(float(p_test), 1e-12))
    ln_len = math.log(max(float(stretch_len_m or 0.0), 10.0) / 50.0) if c.get("c") else 0.0
    p = sigmoid(a + float(c.get("b", 1.0)) * x + float(c.get("c", 0.0)) * ln_len)
    if blind:
        lam_ = DEFAULTS["lambda_blind"] if lam is None else float(lam)
        prior = DEFAULTS["blind_prior_default"] if blind_prior is None else float(blind_prior)
        p = (1.0 - lam_) * p + lam_ * prior
    return p


# ---------------------------------------------------------------- rows
def _fmt(k: str, v):
    if v is None:
        return ""
    if k in FLOAT_FIELDS:
        v = float(v)
        if k.startswith("p_"):
            return f"{v:.6g}"
        return f"{v:.3g}" if abs(v) < 1 else f"{v:.4g}"
    if k in INT_FIELDS:
        return str(int(v))
    if isinstance(v, bool):
        return str(int(v))
    return str(v)


def parse_row(r: dict) -> dict:
    """Typed values of a CSV row (empty -> None)."""
    out = {}
    for k in FIELDS:
        v = r.get(k, "")
        if v in ("", None):
            out[k] = None
        elif k in FLOAT_FIELDS:
            out[k] = float(v)
        elif k in INT_FIELDS:
            out[k] = int(float(v))
        else:
            out[k] = v
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [parse_row(r) for r in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([_fmt(k, r.get(k)) for k in FIELDS])


def merge_rows(old: list[dict], new: list[dict]) -> list[dict]:
    """The rows keyed by (site, scene, section): a new row replaces the old one; sorted by scene then section."""
    d = {tuple(r[k] for k in KEY): r for r in old}
    for r in new:
        d[tuple(r[k] for k in KEY)] = r
    return [d[k] for k in sorted(d)]


def month_path(out: Path, site_id: str, acquired_utc: str) -> Path:
    return out / SUBDIR / site_id / f"{acquired_utc[:7]}.csv"


def save_rows(out: Path, site_id: str, rows: list[dict]) -> list[Path]:
    """Merge the rows into quay_occ2/<site>/YYYY-MM.csv by the scene's month. Returns the files written."""
    by_month: dict = {}
    for r in rows:
        by_month.setdefault(r["acquired_utc"][:7], []).append(r)
    paths = []
    for month, rs in sorted(by_month.items()):
        p = out / SUBDIR / site_id / f"{month}.csv"
        write_rows(p, merge_rows(read_rows(p), rs))
        paths.append(p)
    return paths


def load_site_rows(out: Path, site_id: str) -> dict:
    """{acquired_utc: {section_id: row}} of every month file of a site."""
    d = out / SUBDIR / site_id
    res: dict = {}
    if not d.is_dir():
        return res
    for p in sorted(d.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].csv")):
        for r in read_rows(p):
            res.setdefault(r["acquired_utc"], {})[r["section_id"]] = r
    return res
