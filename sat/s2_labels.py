"""Sentinel-2 labels of the quay sections, matched with Sentinel-1 (s2lab-1, automatic and unverified).

2部の設計 est-2 の #5. On the Sentinel-2 clear days (s2_scenes.csv, result clear, before FREEZE_FROM),
for every quay section (the water sides of the OSM lines in sat/masks/<site>-quay-v<n>.json, cut into
pieces of about SECTION_M), an automatic label is made from the 10 m reflectance:

1. Water reference of the day: median and MAD of B8 (NIR) and of the visible mean (B2, B3, B4) over the
   water of the land mask that SCL calls water (6), away from the quay lines. A pixel is "not water"
   when B8 or the visible mean is above the reference by max(NW_MIN, NW_K x MAD).
2. Depth: along the same normals as the Sentinel-1 quay count (quay.Geometry, 5-120 m off the water's
   edge every 5 m, the nearest line owns a pixel), the leading run of not-water pixels (m). The fixed
   structure (pier deck, OSM line offset) is the minimum of that depth over all clear days of the sample
   (struct); the excess over it is the ship part.
3. Occupied samples: excess >= OCC_EXCESS_M; runs as in quay-v1 (gaps < 20 m joined, >= 30 m kept).
   A section is occupied when its occupied samples reach OCC_MIN_M. Its depth = median excess of the
   occupied samples (5 m); rows abreast c = clip(round(depth / BEAM_M), 1, 3), 0 when not occupied.
4. Hull colour (record only): reflectance of the ship pixels (between struct and the depth); the core
   are those whose 4 neighbours are not water either (all of them when fewer than CORE_MIN). Median
   B4/B3/B2/B8, HSV of the median, and the spread of the brightness (p90 - p10) / median. The class
   grey / colour / contrast / mixed is a clue only.

Sentinel-1: for each clear day, the nearest scene within +-S1_WINDOW_DAYS of a relative orbit with a
baseline (Zhanjiang: 157 and 84), before FREEZE_FROM, is fetched again and counted with quay.count
(quay-v1 with feat-1). Its per-sample occupation and depth are summed per section and matched with the
Sentinel-2 label (ZJG-s2-s1-match.csv). Nothing here changes quay_counts.csv, the est-1 prior or a count.

    python -m sat.s2_labels --site ZJG --max-pu 120 --raw DIR      # fetch (probe), PU in usage.csv
    python -m sat.s2_labels --site ZJG --from-raw DIR               # again from the saved chips, no PU
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import io
import json
import math
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import baseline as fbase
from . import cdse, quay, quay_lines, s2
from .grid import Grid, dilate
from .osm_mask import MASK_DIR, WATER, load_mask

LABELS_VERSION = "s2lab-1"
FREEZE_FROM = "2026-10-27"     # acquisitions from this day on are out of sample: never used here
S1_WINDOW_DAYS = 4.0
SECTION_M = 150.0              # target length of a section along the line
SECTION_MIN_M = 50.0           # shorter pieces are dropped
NW_K = 6.0                     # not water: above the water reference by max(NW_MIN, NW_K x MAD)
NW_MIN = 0.03                  # reflectance
OCC_EXCESS_M = 10.0            # ship part of a sample: depth beyond the structure
OCC_MIN_M = 30.0               # occupied samples in a section (after the runs) for "occupied"
RUN_GAP_M = 20.0
RUN_MIN_M = 30.0
BEAM_M = 18.0                  # rows abreast = depth / beam
STRUCT_MIN_DAYS = 8            # the structure needs this many days with data at the pixel
WATER_FREQ_MIN = 0.25          # a pixel that was water on fewer of the clear days is fixed (structure, land)
FAR_FROM_M = 65.0              # beyond three ships abreast
FAR_WATER_MIN = 0.9            # the far pixels of a section on open water are water on at least this share of the days
BLIND_STRUCT_M = 30.0          # structure this deep: the berth looked taken on every day (blind)
CLOUD_SECTION = 0.3            # SCL cloud + shadow share of the section's pixels: flagged "cloud"
CORE_MIN = 3
GREY_SAT = 0.15
COLOUR_SAT = 0.30
CONTRAST = 0.8
SCALE = s2.REFL_SCALE

LABEL_FIELDS = [
    "site_id", "section_id", "date", "acquired_utc", "labels_version", "label_source", "status",
    "occupied", "occ_len_m", "depth_m", "depth_max_m", "struct_m", "far_water", "c",
    "ship_px", "core_px", "core_mode", "med_r", "med_g", "med_b", "med_nir", "hue_deg", "sat", "val",
    "contrast", "color_class", "cloud_share",
    "verified", "verified_occupied", "verified_c", "verified_color", "verified_by", "verified_note",
]
MATCH_FIELDS = [
    "site_id", "section_id", "sector", "kind", "length_m", "s2_date", "s2_acquired_utc", "s1_acquired_utc",
    "relative_orbit", "dt_h", "s1_in_sample", "look_az_deg", "cos_phi", "face_look",
    "s2_status", "s2_occupied", "s2_depth_m", "s2_c", "s2_color_class",
    "s1_coverage", "s1_occupied", "s1_occ_len_m", "s1_depth_m", "s1_depth_1x3_m", "s1_cols_hint",
    "s1_db_line_m", "s1_rafted", "s1_min", "s1_max", "s1_peak_dvv_db", "d_depth_m", "agree",
]
SECTION_FIELDS = ["section_id", "way_id", "kind", "side", "sector", "length_m", "samples", "lon", "lat", "normal_az_deg"]


# ---------------------------------------------------------------- sections

def sections(geom: quay.Geometry, sector_of=None, section_m: float = SECTION_M) -> list[dict]:
    """Pieces of the water sides: each run of valid samples of a profile, cut into near-equal pieces of
    about ``section_m``. Ids: <osm way>:<L|R>:<k> (L = side +1)."""
    step = geom.cfg["step_m"]
    out = []
    for pi, pr in enumerate(geom.profiles):
        ok = np.asarray(pr["ok"], bool)
        k = 0
        for a, b in quay.runs(ok, step, step, step, False):  # no joining
            n = b - a
            parts = max(1, int(round(n * step / section_m)))
            edges = np.linspace(a, b, parts + 1).round().astype(int)
            for s, e in zip(edges[:-1], edges[1:]):
                if (e - s) * step < SECTION_MIN_M:
                    continue
                idx = np.arange(s, e)
                mid = idx[len(idx) // 2]
                lon, lat = float(pr["lon"][mid]), float(pr["lat"][mid])
                naz = pr["naz"][idx]
                mean_az = math.degrees(math.atan2(np.sin(np.radians(naz)).mean(), np.cos(np.radians(naz)).mean())) % 360
                out.append({
                    "section_id": f"{pr['way_id']}:{'L' if pr['side'] == 1 else 'R'}:{k:02d}",
                    "way_id": pr["way_id"], "kind": pr["kind"], "side": pr["side"], "profile": pi, "idx": idx,
                    "sector": sector_of(lon, lat) if sector_of else None, "length_m": int(len(idx) * step),
                    "samples": int(len(idx)), "lon": round(lon, 6), "lat": round(lat, 6),
                    "normal_az_deg": round(mean_az, 1),
                    "line": [[round(float(pr["lon"][i]), 6), round(float(pr["lat"][i]), 6)] for i in idx],
                })
                k += 1
    return out


def keep_mask(geom: quay.Geometry, shape: tuple[int, int]) -> np.ndarray:
    """Pixels the labels read (the depth pixels of every profile, 0-120 m off the water's edge, and 2 pixels
    around them for the 4 neighbours): the raw chips keep only these."""
    m = np.zeros(shape, bool)
    for pr in geom.profiles:
        m[pr["drr"], pr["dcc"]] = True
        m[pr["rr"], pr["cc"]] = True
    return dilate(m, 2.0)


# ---------------------------------------------------------------- Sentinel-2

def water_reference(refl: np.ndarray, scl: np.ndarray, mask: np.ndarray, near_lines: np.ndarray) -> dict:
    """Median and MAD of B8 and of the visible mean over the open water of the day (reflectance): the
    land mask's water that SCL calls water (6), away from the quay bands when enough is left (the median
    and MAD are robust to the few ships out there). Made from the whole chip, before the raw chip is cut."""
    nir = refl[..., 3].astype(float) / SCALE
    vis = refl[..., :3].astype(float).mean(-1) / SCALE
    water = (mask == WATER) & (scl == 6)
    sel = water & ~near_lines
    if sel.sum() < 2000:
        sel = water
    if sel.sum() < 50:
        return {}
    out = {"pixels": int(sel.sum())}
    for name, a in (("nir", nir[sel]), ("vis", vis[sel])):
        med = float(np.median(a))
        out[name] = med
        out[name + "_mad"] = float(np.median(np.abs(a - med)))
    return out


REF_KEYS = ("nir", "nir_mad", "vis", "vis_mad", "pixels")


def not_water(refl: np.ndarray, ref: dict) -> np.ndarray:
    nir = refl[..., 3].astype(float) / SCALE
    vis = refl[..., :3].astype(float).mean(-1) / SCALE
    t_nir = ref["nir"] + max(NW_MIN, NW_K * ref["nir_mad"])
    t_vis = ref["vis"] + max(NW_MIN, NW_K * ref["vis_mad"])
    return (nir > t_nir) | (vis > t_vis)


def leading_depth(flags: np.ndarray, offs: np.ndarray, step: float) -> np.ndarray:
    """Per sample (rows): metres of not-water from the water's edge without a break (0 when the first pixel is water)."""
    n_run = np.where(flags.all(1), flags.shape[1], np.argmin(flags, 1))
    return np.where(n_run > 0, offs[np.maximum(n_run - 1, 0)] + step / 2, 0.0)


def s2_profiles(geom: quay.Geometry, nw: np.ndarray, valid_px: np.ndarray) -> list[dict]:
    """Per profile: depth of not-water (m) and whether the sample had data."""
    step = geom.cfg["depth_step_m"]
    out = []
    for pr in geom.profiles:
        rr, cc, v = pr["drr"], pr["dcc"], pr["dvalid"]
        flags = v & nw[rr, cc]
        has = valid_px[rr[:, 0], cc[:, 0]] & pr["ok"]
        out.append({"depth": leading_depth(flags, geom.depth_offs, step), "has": has})
    return out


def water_freq(nws: list[np.ndarray], valids: list[np.ndarray]) -> np.ndarray:
    """Per pixel: the share of the clear days with data on which it was water (NaN with fewer than STRUCT_MIN_DAYS)."""
    n = np.sum(valids, 0)
    w = np.sum([v & ~nw for nw, v in zip(nws, valids)], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= STRUCT_MIN_DAYS, w / np.maximum(n, 1), np.nan)


def structure(geom: quay.Geometry, freq: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per profile and sample: (the fixed part: metres from the water's edge of pixels that were rarely water,
    water on fewer than WATER_FREQ_MIN of the days, without a break; the water share of the far pixels,
    FAR_FROM_M and beyond, where no ship should be on most days). NaN where there are too few days."""
    step = geom.cfg["depth_step_m"]
    far_cols = geom.depth_offs >= FAR_FROM_M
    st, far = [], []
    for pr in geom.profiles:
        f = freq[pr["drr"], pr["dcc"]]
        v = pr["dvalid"] & np.isfinite(f)
        with np.errstate(invalid="ignore"):
            rare = v & (f < WATER_FREQ_MIN)
        d = leading_depth(rare, geom.depth_offs, step)
        st.append(np.where(np.isfinite(f[:, 0]), d, np.nan))
        fv = np.where(v[:, far_cols], f[:, far_cols], np.nan)
        far.append(fv)
    return st, far


def hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """HSV of a reflectance triplet (0-REFL_MAX -> 0-1, linear)."""
    h, s, v = colorsys.rgb_to_hsv(*(min(1.0, max(0.0, x / s2.REFL_MAX)) for x in (r, g, b)))
    return h * 360.0, s, v


def color_class(sat: float | None, contrast: float | None) -> str:
    if sat is None:
        return ""
    if sat >= COLOUR_SAT:
        return "colour"
    if contrast is not None and contrast >= CONTRAST:
        return "contrast"
    if sat < GREY_SAT:
        return "grey"
    return "mixed"


def hull_colour(refl: np.ndarray, nw: np.ndarray, pix: set) -> dict:
    """Colour statistics of the ship pixels ``pix`` ({(r, c)}) of a section."""
    if not pix:
        return {}
    h, w = nw.shape
    core = [(r, c) for r, c in pix
            if all(0 <= r + dr < h and 0 <= c + dc < w and nw[r + dr, c + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    mode = "interior"
    if len(core) < CORE_MIN:
        core, mode = list(pix), "all"
    rc = np.array(core)
    px = refl[rc[:, 0], rc[:, 1]].astype(float) / SCALE
    allp = np.array(list(pix))
    bright = refl[allp[:, 0], allp[:, 1], :3].astype(float).mean(-1) / SCALE
    med = np.median(px, 0)
    hh, ss, vv = hsv(med[2], med[1], med[0])
    mb = float(np.median(bright))
    contrast = float((np.percentile(bright, 90) - np.percentile(bright, 10)) / mb) if mb > 0 and len(bright) >= 3 else None
    return {"ship_px": len(pix), "core_px": len(core), "core_mode": mode,
            "med_r": round(float(med[2]), 4), "med_g": round(float(med[1]), 4), "med_b": round(float(med[0]), 4),
            "med_nir": round(float(med[3]), 4), "hue_deg": round(hh, 0), "sat": round(ss, 3), "val": round(vv, 3),
            "contrast": None if contrast is None else round(contrast, 2), "color_class": color_class(ss, contrast)}


def label_day(geom: quay.Geometry, secs: list[dict], prof: list[dict], struct: list[np.ndarray], dry: list[np.ndarray],
              refl: np.ndarray,
              scl: np.ndarray, nw: np.ndarray, site_id: str, date: str, acquired: str) -> list[dict]:
    """Automatic labels of one clear day, one row per section."""
    step = geom.cfg["step_m"]
    dstep = geom.cfg["depth_step_m"]
    cloud_px = np.isin(scl, s2.CLOUD_SCL + s2.SHADOW_SCL)
    occ_by_prof = []
    for pi, pr in enumerate(geom.profiles):
        ex = prof[pi]["depth"] - struct[pi]
        with np.errstate(invalid="ignore"):
            flags = prof[pi]["has"] & np.isfinite(ex) & (ex >= OCC_EXCESS_M)
        occ = np.zeros(len(flags), bool)
        for a, b in quay.runs(flags, step, RUN_GAP_M, RUN_MIN_M, False):
            occ[a:b] = True
        occ_by_prof.append((occ & flags, ex))
    rows = []
    for sc in secs:
        pi, idx = sc["profile"], sc["idx"]
        pr = geom.profiles[pi]
        has = prof[pi]["has"][idx]
        st = struct[pi][idx]
        occ, ex = occ_by_prof[pi][0][idx], occ_by_prof[pi][1][idx]
        rr, cc, v = pr["drr"][idx], pr["dcc"][idx], pr["dvalid"][idx]
        cshare = float(cloud_px[rr[v], cc[v]].mean()) if v.any() else None
        row = {"site_id": site_id, "section_id": sc["section_id"], "date": date, "acquired_utc": acquired,
               "labels_version": LABELS_VERSION, "label_source": "auto（自動・未確認）",
               "cloud_share": None if cshare is None else round(cshare, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            smed = float(np.nanmedian(st)) if np.isfinite(st).any() else None
        row["struct_m"] = None if smed is None else int(round(smed / 5.0) * 5)
        fw = dry[pi][idx]
        fwat = float(np.nanmedian(fw)) if np.isfinite(fw).any() else None
        row["far_water"] = None if fwat is None else round(fwat, 2)
        if has.mean() < 0.5 or smed is None:
            row.update(status="nodata")
            rows.append(row)
            continue
        if fwat is not None and fwat < FAR_WATER_MIN:
            row.update(status="not_water")
            rows.append(row)
            continue
        status = "blind" if smed >= BLIND_STRUCT_M else "cloud" if (cshare or 0) > CLOUD_SECTION else "ok"
        occ_len = int(occ.sum() * step)
        occupied = occ_len >= OCC_MIN_M
        row.update(status=status, occupied=int(occupied), occ_len_m=occ_len, c=0)
        if occupied:
            e = ex[occ]
            dm = float(np.median(e))
            row.update(depth_m=int(round(dm / 5.0) * 5), depth_max_m=int(round(float(e.max()) / 5.0) * 5),
                       c=int(min(3, max(1, round(dm / BEAM_M)))))
            pix = set()
            for j in np.flatnonzero(occ):
                lo, hi = st[j], prof[pi]["depth"][idx[j]]
                for k, off in enumerate(geom.depth_offs):
                    if lo < off + dstep / 2 <= hi and v[j, k]:
                        pix.add((int(rr[j, k]), int(cc[j, k])))
            row.update(hull_colour(refl, nw, pix))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- Sentinel-1

def s1_sections(secs: list[dict], q: dict, samples: list[dict], step: float, face_cos: float) -> dict:
    """Per section id: the Sentinel-1 features of one scene (quay.count with samples_out)."""
    stretch = {}
    for sg in q["segments"]:
        key = (sg["way_id"], sg["side"])
        stretch.setdefault(key, []).append(sg)
    out = {}
    for sc in secs:
        sm = samples[sc["profile"]]
        idx = sc["idx"]
        n = len(sm["inner_hit"])
        in_run = np.zeros(n, bool)
        best, best_ov = None, 0
        for sg in stretch.get((sc["way_id"], sc["side"]), []):
            a, b = sg["idx"]
            ii = np.arange(a, b) % n
            in_run[ii] = True
            ov = len(set(ii.tolist()) & set(idx.tolist()))
            if ov > best_ov:
                best, best_ov = sg, ov
        occ = in_run[idx] & sm["inner_hit"][idx]
        occ_len = int(occ.sum() * step)
        cov = float(sm["has"][idx].mean())
        f = {"s1_coverage": round(cov, 2), "s1_occ_len_m": occ_len, "s1_occupied": int(occ_len >= OCC_MIN_M)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key, name in (("depth", "s1_depth_m"), ("depth13", "s1_depth_1x3_m")):
                d = sm[key][idx][occ]
                f[name] = int(round(float(np.nanmedian(d)) / 5.0) * 5) if occ.any() and np.isfinite(d).any() else None
        f["s1_cols_hint"] = None if f["s1_depth_m"] is None else int(min(3, max(1, round((f["s1_depth_m"] - 20) / 18))))
        if sm["cos_phi"] is not None:
            cphi = float(np.median(sm["cos_phi"][idx]))
            f["cos_phi"] = round(cphi, 2)
            f["face_look"] = quay.face_class(cphi, face_cos)
        if best is not None and best_ov * step >= OCC_MIN_M:
            f.update(s1_db_line_m=best.get("db_line_m"), s1_rafted=int(bool(best.get("rafted"))), s1_min=best.get("min"),
                     s1_max=best.get("max"), s1_peak_dvv_db=best.get("peak_dvv_db"))
        out[sc["section_id"]] = f
    return out


def nearest_s1(acqs: list[dict], t: datetime, orbits: set, window_days: float = S1_WINDOW_DAYS,
               freeze: str = FREEZE_FROM) -> dict | None:
    """The Sentinel-1 acquisition of one of ``orbits`` nearest to ``t`` within the window, before ``freeze``."""
    best = None
    for a in acqs:
        if str(a.get("relative_orbit")) not in orbits or a["acquired_utc"][:10] >= freeze:
            continue
        dt = abs((cdse.parse_time(a["acquired_utc"]) - t).total_seconds())
        if dt <= window_days * 86400 and (best is None or dt < best[0]):
            best = (dt, a)
    return best[1] if best else None


# ---------------------------------------------------------------- I/O

def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    path.write_text(buf.getvalue(), encoding="utf-8")


def save_raw(raw: Path, name: str, keep: np.ndarray, **arrays) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for k, a in arrays.items():
        if a.ndim == 1:
            pass
        elif a.ndim == 3:
            a = np.where(keep[..., None], a, 0)
        elif a.dtype.kind == "f":
            a = np.where(keep, a, np.nan)
        else:
            a = np.where(keep, a, 0)
        arrs[k] = a
    np.savez_compressed(raw / f"{name}.npz", **arrs)


def load_raw(raw: Path, name: str) -> dict | None:
    p = raw / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def settings() -> dict:
    return {
        "version": LABELS_VERSION,
        "status": "自動・未確認（人の目で確かめるまで正解として使わない。verified の列は空）",
        "days": f"s2_scenes.csv の clear の日のうち {FREEZE_FROM} より前（それ以降の撮影は標本外として凍結し、ここでは使わない）",
        "sections": (f"岸壁の線（OSM、quay-v1 と同じ Geometry）の水側を、有効な点の続く所ごとに約 {SECTION_M:g} m の等しい長さに切った区間"
                     f"（{SECTION_MIN_M:g} m 未満は捨てる）。id＝<OSM way>:<L|R>:<番号>（L＝線の左側）。線と区間の表は ZJG-s2-sections.geojson"),
        "water": (f"その日の水の基準：陸マスクの水域で SCL＝6（水）、岸壁の帯（水際から 120 m＋50 m）を除いた画素（切り出し全体から取得時に求める）の B8 と可視（B2・B3・B4 の平均）の中央値と MAD。"
                  f"どちらかが基準より max({NW_MIN:g}, {NW_K:g}×MAD) 以上明るい画素を「水でない」とする"),
        "depth": ("quay-v1 と同じ法線で水際から 5〜120 m を 5 m おきに見て、水でない画素が途切れず続く長さ（m）。"
                  f"画素ごとに晴れの日のうち水だった日の割合を数え（データのある日が {STRUCT_MIN_DAYS} 日以上の画素）、{WATER_FREQ_MIN:.0%} 未満の画素は動かないもの（岸壁・桟橋・陸。線のずれを含む）とする。"
                  "水際から動かないものが途切れず続く長さを struct_m、その日の奥行きのうちそれを超えた分を船の奥行きとする。"
                  f"水際から {FAR_FROM_M:g} m 以遠（3 隻横並びより外）の画素が晴れの日の {FAR_WATER_MIN:.0%} 未満でしか水でない区間（中央値、far_water）は、"
                  "開けた水面に面していない（陸マスクの水域が実は陸・干潟・養殖池・工事中の埋立地など。status＝not_water、ラベルなし）"),
        "occupied": (f"超えた分が {OCC_EXCESS_M:g} m 以上の点を、{RUN_GAP_M:g} m 未満の切れ目をつなぎ {RUN_MIN_M:g} m 以上続く所だけ残し、"
                     f"区間の中で {OCC_MIN_M:g} m 以上あれば occupied＝1。depth_m＝その点の超えた分の中央値、depth_max_m＝最大（5 m 単位）"),
        "c": f"列数の目安：c＝clip(round(depth_m／{BEAM_M:g} m), 1, 3)、占有なしは 0（艦の幅を約 {BEAM_M:g} m と仮定。10 m 画素のため ±1 列の誤りがあり得る）",
        "status": {"ok": "ラベルあり", "cloud": f"区間の画素の SCL の雲・影が {CLOUD_SECTION:.0%} 超（ラベルは作るが疑わしい。白い船が雲とされることもある）",
                   "blind": f"struct_m ≥ {BLIND_STRUCT_M:g} m：水際の画素が晴れの日の {1 - WATER_FREQ_MIN:.0%} 超で水でない（ほぼ常に船がいる・構造物）。占有を見分けられない",
                   "not_water": f"far_water < {FAR_WATER_MIN:g}：沖側が開けた水面でない（陸マスクの水域が実際は陸・干潟・養殖池など。ラベルなし）",
                   "nodata": "区間の半分以上でデータなし"},
        "colour": (f"船の画素（struct から奥行きまで）のうち、上下左右も水でない画素を芯とする（{CORE_MIN} 個未満なら全部、core_mode＝all）。"
                   "芯の反射率の中央値 med_r（B4）・med_g（B3）・med_b（B2）・med_nir（B8）、その HSV（反射率 0〜0.3 を 0〜1、線形）、"
                   "contrast＝船の画素の明るさ（可視の平均）の (p90−p10)／中央値"),
        "color_class": (f"記録だけ（判定に使わない）：colour（彩度 ≥ {COLOUR_SAT:g}）、contrast（≥ {CONTRAST:g}、白黒の対比）、"
                        f"grey（彩度 < {GREY_SAT:g}）、mixed（その他）"),
        "s1": (f"各晴れの日の ±{S1_WINDOW_DAYS:g} 日以内で基準のある相対軌道（湛江は 157 と 84。カタログではどちらも上昇＝夕方の軌道）の撮影のうち最も近いもの"
               f"（{FREEZE_FROM} より前）を取り直し、quay-v1（feat-1 込み）で数えた。quay_counts.csv・est-1 の事前分布は変えない。"
               "区間ごとに、数えた占有区間（stretch）に入る内側の帯の占有点の長さ s1_occ_len_m、その点の depth_m・depth_1x3_m の中央値、"
               "最も重なる stretch の db_line_m・横抱き・min/max、区間の cosφ と face_look"),
        "match": ("d_depth_m＝s1_depth_m−s2 の depth_m（両方占有のとき）。agree＝占有の有無が一致（1/0）。"
                  "時刻は S2 が 03:22 UTC ごろ、S1 が 10:41〜10:49 UTC ごろで、同じ日でも約 7.5 時間ずれる（dt_h）：入出港で船が違うことがある"),
        "raw": ("取得した切り出し（岸壁の帯だけ残した npz：S2 の B2・B3・B4・B8・SCL と水の基準、S1 の VV・VH の dB）は probe ブランチ "
                "probe/sat-s2-labels-ops3 の probe-raw/s2-labels/<地点>/ にだけ置く（main には入れない）。"
                "python -m sat.s2_labels --from-raw <そのディレクトリ> で PU を使わずに作り直せる"),
        "limits": [
            "10 m 画素：幅 18 m 前後の艦は 1〜3 画素で、縁は水と混ざる。c は ±1 列ずれ得る（特に 1 と 2）",
            f"struct は、晴れの日の {1 - WATER_FREQ_MIN:.0%} 超で船がいた所では船を構造物とみなす（status＝blind）。晴れの日は 16 日で季節に偏りがある",
            "水の基準は湾全体の 1 つの値：濁り・航跡・日の照り返しの差は見ていない",
            "色は水との混ざった画素を含み、彩度は高めに出やすい。L2A の大気補正の誤差もある。艦種や軍民の判定には使わない",
            "Sentinel-1 の撮影の一部は基準（bl-1、2026-05 以降）の期間に入っている（s1_in_sample＝baseline）：その撮影は基準の p10 に自分も入っている",
        ],
    }


# ---------------------------------------------------------------- main

def build(site: dict, out: Path, raw: Path, client=None, max_pu: float = 120.0, mask_dir: Path = MASK_DIR,
          baseline_dir: Path = fbase.BASELINE_DIR, log=print, budget: float = 1e9) -> dict:
    """Fetch what is missing in ``raw`` (only with a client) and write the labels and the match table."""
    from .run import site_sectors, sector_of as _sector_of
    grid = Grid.for_site(site)
    mver = int(site.get("mask_version", 1))
    mask, _ = load_mask(site["id"], mver, mask_dir)
    lines = quay_lines.load(site["id"], int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), mask_dir)
    geom = quay.Geometry(grid, mask, lines)
    sectors = site_sectors(site)
    secs = sections(geom, lambda lo, la: _sector_of(sectors, lo, la))
    keep = keep_mask(geom, mask.shape)
    near = dilate(keep, 3) if keep.any() else keep
    tag = f"{site['id']}-land-v{mver}"
    days = [r for r in s2.read_rows(out / "s2_scenes.csv")
            if r["site_id"] == site["id"] and r["mask_version"] == tag and r["result"] == "clear" and r["date"] < FREEZE_FROM]
    days.sort(key=lambda r: r["date"])
    bls = {str(o): fbase.load(site, o, baseline_dir) for o in fbase.site_config(site)["orbits"]}
    bls = {k: v for k, v in bls.items() if v is not None and (v[1].get("width"), v[1].get("height")) == (grid.width, grid.height)}
    limit = min(max_pu, budget)
    summ = {"site_id": site["id"], "days": len(days), "s2_fetched": 0, "s1_fetched": 0, "deferred": [], "errors": []}
    pu0 = client.processing_units if client else 0.0
    req0 = client.requests if client else 0
    s2_pu = s2_req = 0.0
    acqs = None
    if client is not None and days:
        t0 = cdse.parse_time(days[0]["acquired_utc"]) - timedelta(days=S1_WINDOW_DAYS + 1)
        t1 = cdse.parse_time(days[-1]["acquired_utc"]) + timedelta(days=S1_WINDOW_DAYS + 1)
        acqs = cdse.acquisitions(client.catalog(grid.bbox, t0, t1))
    s1_pick = {}
    s1_est = cdse.estimate_pu(grid.width, grid.height, 3)
    for d in days:
        name = f"s2-{d['date']}"
        if load_raw(raw, name) is None and client is not None:
            if client.processing_units - pu0 + s2.EST_FULL_PU > limit:
                summ["deferred"].append(name)
            else:
                r0, p0 = client.requests, client.processing_units
                try:
                    refl, scl, _pu = s2.fetch(client, grid, d["date"])
                    ref = water_reference(refl, scl, mask, near)
                    save_raw(raw, name, keep, refl=refl, scl=scl,
                             wref=np.array([ref.get(k, np.nan) for k in REF_KEYS], float))
                    summ["s2_fetched"] += 1
                except cdse.CredentialError:
                    raise
                except Exception as e:
                    summ["errors"].append(f"{name}: {str(e)[:200]}")
                s2_pu += client.processing_units - p0
                s2_req += client.requests - r0
        meta_p = raw / f"s1-for-{d['date']}.json"
        if meta_p.exists():
            a = json.loads(meta_p.read_text(encoding="utf-8"))
        elif acqs is not None:
            a = nearest_s1(acqs, cdse.parse_time(d["acquired_utc"]), set(bls))
            a = None if a is None else {k: a[k] for k in ("acquired_utc", "platform", "orbit_direction", "relative_orbit")} | {
                "t_from": cdse.iso(a["t_from"]), "t_to": cdse.iso(a["t_to"])}
            raw.mkdir(parents=True, exist_ok=True)
            meta_p.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
        else:
            a = None
        s1_pick[d["date"]] = a
        if a is None:
            continue
        sname = f"s1-{s2_stamp(a['acquired_utc'])}"
        if load_raw(raw, sname) is None and client is not None:
            if client.processing_units - pu0 + s1_est > limit:
                summ["deferred"].append(sname)
                continue
            try:
                from .run import decode_chip
                data, pu1 = client.process(grid.bbox, grid.width, grid.height, cdse.parse_time(a["t_from"]),
                                           cdse.parse_time(a["t_to"]), False)
                s1_est = max(s1_est, float(pu1 or 0))
                vv, vh, _ = decode_chip(data)
                save_raw(raw, sname, keep, vv=vv.astype(np.float16), vh=vh.astype(np.float16))
                summ["s1_fetched"] += 1
            except cdse.CredentialError:
                raise
            except Exception as e:
                summ["errors"].append(f"{sname}: {str(e)[:200]}")
    if client is not None:
        summ["processing_units"] = round(client.processing_units - pu0, 3)
        summ["requests"] = client.requests - req0
        summ["s2_processing_units"] = round(s2_pu, 3)
        summ["s2_requests"] = int(s2_req)

    # labels
    loaded = []
    for d in days:
        z = load_raw(raw, f"s2-{d['date']}")
        if z is None:
            continue
        refl, scl = z["refl"], z["scl"]
        ref = dict(zip(REF_KEYS, (float(x) for x in z["wref"])))
        if not np.isfinite(ref["nir"]):
            summ["errors"].append(f"{d['date']}: 水の基準を作れない")
            continue
        nw = not_water(refl, ref) & ~np.isin(scl, (0, s2.NODATA))
        valid_px = ~np.isin(scl, (0, s2.NODATA)) & keep
        loaded.append((d, refl, scl, nw, s2_profiles(geom, nw, valid_px), ref, valid_px))
    labels, match = [], []
    if loaded:
        freq = water_freq([x[3] for x in loaded], [x[6] for x in loaded])
        struct, dry = structure(geom, freq)
        for d, refl, scl, nw, prof, ref, _v in loaded:
            rows = label_day(geom, secs, prof, struct, dry, refl, scl, nw, site["id"], d["date"], d["acquired_utc"])
            labels.extend(rows)
            a = s1_pick.get(d["date"])
            z = load_raw(raw, f"s1-{s2_stamp(a['acquired_utc'])}") if a else None
            s1f, look, insample = {}, None, ""
            if z is not None:
                layers, bmeta = bls[str(a["relative_orbit"])]
                look = quay.look_azimuth(a.get("orbit_direction"), float(site.get("center_lat") or (grid.south + grid.north) / 2))
                samples: list = []
                q = quay.count(geom, z["vv"].astype(float), z["vh"].astype(float), layers, look_az_deg=look, samples_out=samples)
                s1f = s1_sections(secs, q, samples, geom.cfg["step_m"], geom.cfg["face_cos"])
                insample = "baseline" if any(sc.get("acquired_utc") == a["acquired_utc"] for sc in bmeta.get("scene_list") or []) else ""
            byid = {sc["section_id"]: sc for sc in secs}
            for r in rows:
                sc = byid[r["section_id"]]
                f = s1f.get(r["section_id"], {})
                m = {"site_id": site["id"], "section_id": r["section_id"], "sector": sc["sector"], "kind": sc["kind"],
                     "length_m": sc["length_m"], "s2_date": d["date"], "s2_acquired_utc": d["acquired_utc"],
                     "s2_status": r["status"], "s2_occupied": r.get("occupied"), "s2_depth_m": r.get("depth_m"),
                     "s2_c": r.get("c"), "s2_color_class": r.get("color_class"), **f}
                if a:
                    m.update(s1_acquired_utc=a["acquired_utc"], relative_orbit=a["relative_orbit"], look_az_deg=look,
                             s1_in_sample=insample,
                             dt_h=round((cdse.parse_time(a["acquired_utc"]) - cdse.parse_time(d["acquired_utc"])).total_seconds() / 3600, 1))
                if z is not None and r.get("occupied") is not None and f.get("s1_coverage", 0) >= 0.5:
                    m["agree"] = int(int(r["occupied"]) == f["s1_occupied"])
                    if r.get("depth_m") is not None and f.get("s1_depth_m") is not None:
                        m["d_depth_m"] = f["s1_depth_m"] - r["depth_m"]
                match.append(m)
    ldir = out / "labels"
    _write(ldir / f"{site['id']}-s2-labels.csv", labels, LABEL_FIELDS)
    _write(ldir / f"{site['id']}-s2-s1-match.csv", match, MATCH_FIELDS)
    fc = {"type": "FeatureCollection",
          "attribution": "線：(c) OpenStreetMap contributors（ODbL 1.0）", "license": "ODbL-1.0",
          "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": sc["line"]},
                        "properties": {k: sc[k] for k in SECTION_FIELDS}} for sc in secs]}
    (ldir / f"{site['id']}-s2-sections.geojson").write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    summ.update(sections=len(secs), labels=len(labels), stats=stats(labels, match))
    meta = {"generated_utc": cdse.iso(datetime.now(timezone.utc)), "site_id": site["id"], "method": settings(),
            "s2_days": [d["date"] for d, *_ in loaded],
            "s1_scenes": {k: (v or {}).get("acquired_utc") for k, v in s1_pick.items()},
            "water_reference": {d["date"]: {k: round(v, 4) if isinstance(v, float) else v for k, v in ref.items()} for d, *_, ref, _v in loaded},
            "summary": summ["stats"],
            "files": {f"{site['id']}-s2-labels.csv": "区間×晴れの日の自動ラベル（自動・未確認。verified_* は人が確かめて埋める）",
                      f"{site['id']}-s2-s1-match.csv": "同じ区間×晴れの日の S2 ラベルと、近い S1 の撮影の岸壁の特徴の突き合わせ",
                      f"{site['id']}-s2-sections.geojson": "区間の線（OSM 由来、ODbL）"},
            "verify": ("verified＝1（確かめた）／0（見たが判断できない）、verified_occupied・verified_c・verified_color に人の判定、"
                       "verified_by に確かめた人、verified_note に根拠（画像 s2/<地点>/v<版>/<撮影>.webp）。空欄は未確認"),
            "credit": "Contains modified Copernicus Sentinel data 2025, 2026",
            "attribution": "岸壁の線：(c) OpenStreetMap contributors（ODbL 1.0）"}
    (ldir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return summ


def s2_stamp(t: str) -> str:
    return t.replace(":", "-")


def stats(labels: list[dict], match: list[dict]) -> dict:
    from collections import Counter
    st = Counter(r["status"] for r in labels)
    use = [r for r in labels if r["status"] in ("ok", "cloud")]
    c = Counter(r.get("c") for r in use if r.get("c") is not None)
    occ = Counter(r.get("color_class") or "-" for r in use if r.get("occupied"))
    out = {"labels": len(labels), "status": dict(st), "c": {str(k): c[k] for k in sorted(c)}, "color_class": dict(occ)}
    m = [x for x in match if x.get("agree") is not None and x["s2_status"] == "ok"]
    if m:
        both = lambda a, b: sum(1 for x in m if x["s2_occupied"] == a and x["s1_occupied"] == b)
        out["occupancy_s2_vs_s1"] = {"both": both(1, 1), "s2_only": both(1, 0), "s1_only": both(0, 1), "neither": both(0, 0)}
        by = {}
        for x in m:
            if x.get("d_depth_m") is not None:
                by.setdefault(x.get("face_look") or "-", []).append(x["d_depth_m"])
        out["d_depth_by_face"] = {k: {"n": len(v), "median": float(np.median(v)), "mean": round(float(np.mean(v)), 1)}
                                  for k, v in sorted(by.items())}
        cc = Counter((x["s2_c"], x.get("s1_cols_hint")) for x in m if x["s2_occupied"] == 1 and x.get("s1_cols_hint") is not None)
        out["c_vs_s1_cols_hint"] = {f"{a}-{b}": n for (a, b), n in sorted(cc.items())}
    return out


def main(argv=None, client=None, sites=None) -> int:
    from .run import load_sites, read_csv, write_csv, USAGE_FIELDS, MONTHLY_PU_CAP, month_usage
    ap = argparse.ArgumentParser(description="Sentinel-2 labels of the quay sections (s2lab-1, automatic, unverified)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "data" / "sat")
    ap.add_argument("--site", default="ZJG")
    ap.add_argument("--raw", type=Path, help="directory of the fetched chips (fetches what is missing)")
    ap.add_argument("--from-raw", type=Path, help="only the saved chips: no request")
    ap.add_argument("--max-pu", type=float, default=120.0)
    args = ap.parse_args(argv)
    if bool(args.raw) == bool(args.from_raw):
        ap.error("--raw か --from-raw のどちらか 1 つ")
    site = next(s for s in (sites if sites is not None else load_sites()) if s["id"] == args.site)
    now = datetime.now(timezone.utc)
    usage = read_csv(args.out / "usage.csv")
    used_pu, _ = month_usage(usage, now.strftime("%Y-%m"))
    if args.from_raw:
        summ = build(site, args.out, args.from_raw, None, log=lambda m: print(m, file=sys.stderr))
    else:
        client = client if client is not None else cdse.Client()
        summ = {}
        try:
            summ = build(site, args.out, args.raw, client, args.max_pu, budget=MONTHLY_PU_CAP - used_pu,
                         log=lambda m: print(m, file=sys.stderr))
        finally:
            if client.requests:
                usage.append({"run_utc": cdse.iso(now), "requests": client.requests,
                              "processing_units": round(client.processing_units, 3),
                              "pu_source": "header" if client.pu_reported else "estimate",
                              "scenes": summ.get("s1_fetched", ""),
                              "s2_requests": summ.get("s2_requests", ""), "s2_processing_units": summ.get("s2_processing_units", "")})
                write_csv(args.out / "usage.csv", usage, USAGE_FIELDS)
    summ["month_pu_before"] = round(used_pu, 3)
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
