"""Rows abreast from the cross-section across the quay line (cols-1, trial; 2部's survey, option 5).

A ship moored alongside widens the bright band off the water's edge by about its beam; a second ship moored
outboard (rafted) by about one beam more. Across the line (the depth pixels of sat/quay.Geometry, 5-120 m off
the water's edge every 5 m) the change against the empty-quay baseline dVV(d) = VV(d) - VV_p10(d) is taken at
the samples of the section that are occupied (dVV within ``start_m`` of the edge at least ``seed_db``); its
median over those samples is the section's cross-section, and the protrusion width W is how far it stays at or
above max(``level_db``, peak - ``drop_db``) without a break from the edge.

How W relates to the rows depends on the look: the facing side of a hull is bright, the far side in shadow,
and layover moves the echo toward the radar. So the model is calibrated per orbit direction:

    W | c ~ Normal(w0_dir + beam_dir * (c - 1), sigma_dir),  c = 1, 2, 3

fitted by least squares on the owner's labels (``fit``), and the directions are combined by multiplying their
likelihoods with the prior of c (``posterior``): an ascending and a descending scene a few days apart give one
estimate. A direction without labels uses the default (``status`` "default"). The calibration file is
sat/cols_xsec_calib.json (written by ``python -m sat.eval_cols --write-calib``): the width there is the occupied
depth of the raw sigma0 (est-1 depth_m, the same cross-section measure in the daily run), fitted on Zhanjiang's
eye labels; the width from a display image (``section_width`` on sat/eval_occ.decode_display) is a different,
degraded measure and is only evaluated, not calibrated into the file.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

COLS_VERSION = "cols-1"
CALIB_PATH = Path(__file__).resolve().parent / "cols_xsec_calib.json"
DEFAULTS = {
    "seed_db": 6.0,        # a sample is occupied when dVV within start_m reaches this
    "start_m": 15.0,
    "level_db": 3.0,       # the band ends below max(level_db, peak - drop_db)
    "drop_db": 6.0,
    "min_samples": 2,      # occupied samples needed for a width
}
DEFAULT_MODEL = {"w0": 30.0, "beam": 25.0, "sigma": 12.0, "n": 0, "status": "default",
                 "note": "較正なし：1 列目の帯の幅 30 m（船幅約 20 m＋3×3 平均の広がり約 10 m）、1 列ごとに 25 m"}
PRIOR_C = (0.70, 0.25, 0.05)   # P(c = 1, 2, 3) given occupied (Zhanjiang's eye labels: rafting is the exception)
MIN_FIT = 6                    # labels per direction for a fit; with one c level only w0 and sigma are fitted
DIRECTIONS = ("ascending", "descending")


def section_width(geom, sec: dict, dvv: np.ndarray, cfg: dict | None = None) -> dict | None:
    """{"width_m", "samples", "peak_db", "profile"} of a section (sat/s2_labels.sections()), or None if fewer than
    min_samples of its samples are occupied. ``dvv``: scene VV minus the baseline's vv_p10 (dB, same smoothing)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    pr = geom.profiles[sec["profile"]]
    idx = np.asarray(sec["idx"])
    offs = geom.depth_offs
    prof = np.where(pr["dvalid"][idx], dvv[pr["drr"][idx], pr["dcc"][idx]], np.nan)
    near = offs <= cfg["start_m"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seed = np.nanmax(prof[:, near], 1)
        occ = np.isfinite(seed) & (seed >= cfg["seed_db"])
        if occ.sum() < cfg["min_samples"]:
            return None
        med = np.nanmedian(prof[occ], 0)
    if not np.isfinite(med[near]).any():
        return None
    peak = float(np.nanmax(med))
    level = max(cfg["level_db"], peak - cfg["drop_db"])
    above = np.isfinite(med) & (med >= level)
    st = np.flatnonzero(above & near)
    if not len(st):
        return None
    k = int(st[0])
    while k + 1 < len(above) and above[k + 1]:
        k += 1
    width = float(offs[k]) + geom.cfg["depth_step_m"] / 2.0
    return {"width_m": width, "samples": int(occ.sum()), "peak_db": round(peak, 1),
            "profile": [None if not np.isfinite(x) else round(float(x), 1) for x in med]}


def fit(widths, cols) -> dict:
    """Least squares W = w0 + beam (c - 1); sigma from the residuals. One c level: beam stays the default."""
    w = np.asarray(widths, float)
    c = np.asarray(cols, float)
    n = len(w)
    if n < MIN_FIT:
        return {**DEFAULT_MODEL, "n": int(n), "status": "default",
                "note": f"ラベル {n} 件（{MIN_FIT} 件未満）のため既定値"}
    if len(np.unique(c)) < 2:
        beam = DEFAULT_MODEL["beam"]
        w0 = float(np.mean(w - beam * (c - 1)))
        status = "w0_only"
    else:
        A = np.stack([np.ones(n), c - 1], 1)
        (w0, beam), *_ = np.linalg.lstsq(A, w, rcond=None)
        w0, beam = float(w0), float(beam)
        if beam < 5.0:  # rows must widen the band: a negative or flat slope is noise in a small sample
            beam = DEFAULT_MODEL["beam"]
            w0 = float(np.mean(w - beam * (c - 1)))
            status = "w0_only"
        else:
            status = "fitted"
    res = w - (w0 + beam * (c - 1))
    dof = max(n - (2 if status == "fitted" else 1), 1)
    sigma = max(float(math.sqrt((res ** 2).sum() / dof)), 5.0)
    return {"w0": round(w0, 1), "beam": round(beam, 1), "sigma": round(sigma, 1), "n": int(n), "status": status}


def _npdf(x, mu, s):
    return math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def posterior(widths: dict, calib: dict | None = None, prior=PRIOR_C) -> list[float]:
    """P(c = 1, 2, 3) from the widths by direction ({"ascending": W or None, "descending": ...}); the directions'
    likelihoods are multiplied (independent looks). No width at all: the prior."""
    calib = calib or {}
    lik = np.array(prior, float)
    for d, wv in widths.items():
        if wv is None:
            continue
        m = calib.get(d) or DEFAULT_MODEL
        lik = lik * np.array([_npdf(wv, m["w0"] + m["beam"] * k, m["sigma"]) for k in range(3)])
    s = lik.sum()
    return (lik / s).tolist() if s > 0 else list(prior)


def load_calib(path: Path = CALIB_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("directions", {})


def settings() -> dict:
    return {"version": COLS_VERSION, **DEFAULTS, "default_model": DEFAULT_MODEL, "prior_c": list(PRIOR_C)}
