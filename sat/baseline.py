"""Per-orbit "empty quay" baseline of a site from past Sentinel-1 scenes (P2 of the quay count).

For each relative orbit listed in the site's ``baseline.orbits`` (sites.json), the newest
``max_scenes`` acquisitions within ``lookback_days`` are fetched once (Process API, VV and
VH, the same chip as the daily run), each band is averaged over 3x3 pixels (linear power,
valid pixels only) and the per-pixel 10th and 90th percentiles are taken over the scenes:

* VV p10 and VH p10 = the quay without ships (a ship is at a berth in fewer than 10 % of
  the scenes, so the 10th percentile shows the empty berth),
* VV p90 = the busy state; p10 > -8 dB and p90 - p10 < 3 dB = "always occupied" (a
  structure missing from OSM, or a ship that never left; 2部 decides from optical images).

The layers are stored as 8-bit PNGs (dB quantised, 0 = no data) under
``sat/baselines/<mask version>/``, only within ``KEEP_M`` of water (land further inland is
not needed by the quay count or the change image). Built by a probe (GitHub Actions; the
work shell cannot reach CDSE), checked, then committed by hand like the land masks.

    python -m sat.baseline ZJG --out DIR [--usage docs/data/sat/usage.csv] [--max-pu 130]
           [--export-scenes 4]   # also write the newest smoothed scenes per orbit (tuning only)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, png
from .grid import Grid, box_sum, dilate, label
from .osm_mask import LAND, WATER, load_mask

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
BASELINE_VERSION = "bl-1"
DEFAULTS = {
    "lookback_days": 200,   # about six months (S1C+S1D on track 157: ~30 scenes; track 84: ~15)
    "max_scenes": 30,       # newest scenes per orbit
    "min_scenes": 24,       # below this the quay count is one credibility step lower
    "p_low": 10,
    "p_high": 90,
    "min_coverage": 0.9,    # share of the site's water with data; partial scenes are left out
    "min_valid": 10,        # a pixel needs this many scenes with data, else no data
}
SMOOTH_HALF = 1             # 3x3 mean before the percentiles (and before the quay count)
KEEP_M = 200.0              # layers kept within this distance of water or structures
OCC_P10_DB = -8.0           # "always occupied": p10 above this ...
OCC_SPREAD_DB = 3.0         # ... and p90 - p10 below this
# 8-bit quantisation: dB = q * Q_STEP + Q_OFFSET (q = 1..255), q = 0 no data -> -45.0 .. +18.5 dB
Q_STEP = 0.25
Q_OFFSET = -45.25
LAYERS = ("vv_p10", "vv_p90", "vh_p10")


def quantise(db: np.ndarray) -> np.ndarray:
    q = np.round((np.nan_to_num(db, nan=Q_OFFSET) - Q_OFFSET) / Q_STEP)
    q = np.clip(q, 1, 255).astype(np.uint8)
    q[~np.isfinite(db)] = 0
    return q


def dequantise(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.where(q > 0, q.astype(np.float64) * Q_STEP + Q_OFFSET, np.nan)


def smooth_db(db: np.ndarray, half: int = SMOOTH_HALF, min_valid: int = 5) -> np.ndarray:
    """Mean over a (2*half+1)^2 window in linear power of the valid pixels, back to dB."""
    ok = np.isfinite(db)
    lin = np.where(ok, 10.0 ** (np.where(ok, db, 0.0) / 10.0), 0.0)
    n = box_sum(ok.astype(np.float64), half)
    s = box_sum(lin, half)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(s / n)
    out[(n < min_valid) | ~ok | ~np.isfinite(out)] = np.nan
    return out


def keep_region(mask: np.ndarray, pixel_m: float = 10.0) -> np.ndarray:
    """Pixels whose baseline is stored: within KEEP_M of water or structures."""
    return dilate(mask != LAND, KEEP_M / pixel_m)


def site_config(site: dict) -> dict:
    cfg = {**DEFAULTS, **(site.get("baseline") or {})}
    cfg["orbits"] = [int(o) for o in cfg.get("orbits") or []]
    return cfg


def tag(site: dict) -> str:
    return f"{site['id']}-land-v{int(site.get('mask_version', 1))}"


def stem(orbit: int, version: str = BASELINE_VERSION) -> str:
    return f"ro{int(orbit)}-{version}"


def paths(site: dict, orbit: int, base_dir: Path = BASELINE_DIR, version: str = BASELINE_VERSION) -> dict:
    d = base_dir / tag(site)
    s = stem(orbit, version)
    return {"json": d / f"{s}.json", **{k: d / f"{s}.{k}.png" for k in LAYERS}}


def credibility(n_scenes: int, min_scenes: int = DEFAULTS["min_scenes"]) -> str:
    """Credibility of a quay count made with this baseline: 3, or 4 (one step lower) when thin."""
    return "3" if n_scenes >= min_scenes else "4"


def compute(vv_stack: list[np.ndarray], vh_stack: list[np.ndarray], keep: np.ndarray, cfg: dict | None = None) -> dict:
    """Percentile layers (float dB, NaN = no data) from smoothed scenes of one orbit."""
    cfg = {**DEFAULTS, **(cfg or {})}
    vv = np.stack(vv_stack).astype(np.float32)
    vh = np.stack(vh_stack).astype(np.float32)
    n_valid = np.isfinite(vv).sum(0)
    few = (n_valid < cfg["min_valid"]) | ~keep
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        out = {
            "vv_p10": np.nanpercentile(vv, cfg["p_low"], axis=0).astype(np.float64),
            "vv_p90": np.nanpercentile(vv, cfg["p_high"], axis=0).astype(np.float64),
            "vh_p10": np.nanpercentile(vh, cfg["p_low"], axis=0).astype(np.float64),
        }
    for k in out:
        out[k][few] = np.nan
    out["n_valid"] = n_valid
    return out


def always_occupied(vv_p10: np.ndarray, vv_p90: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (vv_p10 > OCC_P10_DB) & ((vv_p90 - vv_p10) < OCC_SPREAD_DB)


def occupied_clusters(occ: np.ndarray, mask: np.ndarray, grid: Grid, min_px: int = 3) -> list[dict]:
    """Always-occupied clusters lying on OSM water (not on land or a mapped structure):
    a structure missing from OSM or a ship that stayed all the time. For 2部 to check.
    The first 20 m off land or a structure is left out: the 3x3 mean spreads their echo
    one pixel onto the water, and OSM lines are off by 10-20 m in places."""
    on_water = occ & (mask == WATER) & ~dilate(mask != WATER, 2)
    lab, n = label(on_water, 8)
    out = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) < min_px:
            continue
        lon, lat = grid.to_lonlat(rr.mean() + 0.5, cc.mean() + 0.5)
        ext = max(rr.max() - rr.min(), cc.max() - cc.min()) + 1
        out.append({"lon": round(float(lon), 6), "lat": round(float(lat), 6), "pixels": int(len(rr)),
                    "extent_m": int(round(ext * 10))})
    out.sort(key=lambda c: -c["pixels"])
    return out


def save(site: dict, grid: Grid, orbit: int, layers: dict, scenes: list[dict], cfg: dict, mask: np.ndarray,
         base_dir: Path, built_utc: str) -> dict:
    p = paths(site, orbit, base_dir)
    p["json"].parent.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for k in LAYERS:
        data = png.encode(quantise(layers[k]), filters=[4])
        p[k].write_bytes(data)
        sizes[k] = len(data)
    occ = always_occupied(layers["vv_p10"], layers["vv_p90"])
    keep = np.isfinite(layers["vv_p10"])
    water = (mask == WATER) & keep
    used = [s for s in scenes if s.get("used")]
    n = len(used)
    meta = {
        "version": BASELINE_VERSION,
        "site": site["id"],
        "mask_version": tag(site),
        "relative_orbit": int(orbit),
        "built_utc": built_utc,
        "bbox": grid.bbox, "width": grid.width, "height": grid.height,
        "scenes": n,
        "first_utc": min(s["acquired_utc"] for s in used) if used else None,
        "last_utc": max(s["acquired_utc"] for s in used) if used else None,
        "credibility": credibility(n, cfg["min_scenes"]),
        "credibility_note": (f"{n} シーン（{cfg['min_scenes']} 以上）" if n >= cfg["min_scenes"] else
                             f"{n} シーンで {cfg['min_scenes']} 未満のため、この基準による岸壁の数の確度を一段下げる（3→4）"),
        "method": {
            "smooth": f"{2 * SMOOTH_HALF + 1}x{2 * SMOOTH_HALF + 1} の平均（線形の σ0、有効画素のみ）を取ってから dB",
            "percentiles": [cfg["p_low"], cfg["p_high"]],
            "layers": {"vv_p10": "VV の 10 パーセンタイル（船のいない岸壁の基準）",
                       "vv_p90": "VV の 90 パーセンタイル",
                       "vh_p10": "VH の 10 パーセンタイル（岸壁の数の ΔVH の基準）"},
            "always_occupied": f"vv_p10 > {OCC_P10_DB:g} dB かつ vv_p90 − vv_p10 < {OCC_SPREAD_DB:g} dB",
            "min_valid": cfg["min_valid"], "min_coverage": cfg["min_coverage"],
            "lookback_days": cfg["lookback_days"], "max_scenes": cfg["max_scenes"],
        },
        "encoding": {"format": "PNG 8 bit grey", "db": f"q × {Q_STEP:g} + ({Q_OFFSET:g})（q = 1〜255、−45.0〜+18.5 dB）",
                     "no_data": "q = 0", "kept": f"水域・構造物から {KEEP_M:g} m 以内の画素のみ（それより内陸は 0）"},
        "files": {k: p[k].name for k in LAYERS},
        "png_bytes": sizes,
        "stats": {
            "kept_fraction": round(float(keep.mean()), 4),
            "water_vv_p10_median_db": round(float(np.nanmedian(layers["vv_p10"][water])), 2) if water.any() else None,
            "water_vh_p10_median_db": round(float(np.nanmedian(layers["vh_p10"][water])), 2) if water.any() else None,
            "always_occupied_px": int(occ.sum()),
            "always_occupied_on_water_px": int((occ & (mask == WATER) & ~dilate(mask != WATER, 2)).sum()),
            "min_valid_scenes_in_kept": int(layers["n_valid"][keep].min()) if keep.any() else 0,
        },
        "always_occupied_on_water": occupied_clusters(occ, mask, grid)[:60],
        "scene_list": scenes,
        "source": "Copernicus Sentinel-1 IW GRD（CDSE Sentinel Hub Process API、σ0 楕円体基準、COPERNICUS_30 で地形補正、10 m）",
        "credit": f"Contains modified Copernicus Sentinel data {built_utc[:4]}",
    }
    p["json"].write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return meta


def load(site: dict, orbit, base_dir: Path = BASELINE_DIR) -> tuple[dict, dict] | None:
    """(layers in float dB, meta) of a site's baseline for one relative orbit, or None."""
    try:
        orbit = int(orbit)
    except (TypeError, ValueError):
        return None
    p = paths(site, orbit, base_dir)
    if not p["json"].exists():
        return None
    meta = json.loads(p["json"].read_text(encoding="utf-8"))
    layers = {k: dequantise(png.decode(p[k].read_bytes())) for k in LAYERS}
    return layers, meta


def build(client, site: dict, mask: np.ndarray, out_dir: Path, now: datetime, budget_pu: float,
          export_scenes: int = 0, log=print) -> dict:
    """Fetch, compute and save the baselines of every listed orbit. Returns a summary."""
    cfg = site_config(site)
    grid = Grid.for_site(site)
    if mask.shape != (grid.height, grid.width):
        raise ValueError(f"{site['id']}: 陸マスクの大きさ {mask.shape} が範囲 {(grid.height, grid.width)} と合わない")
    feats = client.catalog(grid.bbox, now - timedelta(days=cfg["lookback_days"]), now)
    acqs = cdse.acquisitions(feats)
    by_orbit: dict = {}
    for a in acqs:
        by_orbit.setdefault(a["relative_orbit"], []).append(a)
    log(f"{site['id']}: カタログ {len(feats)} 件 → 撮影 {len(acqs)} 回（過去 {cfg['lookback_days']} 日）、相対軌道別 "
        + "、".join(f"{o}:{len(v)}（{v[0]['orbit_direction']}）" for o, v in sorted(by_orbit.items(), key=lambda kv: str(kv[0]))))
    plan = {o: sorted(by_orbit.get(o, []), key=lambda a: a["acquired_utc"])[-cfg["max_scenes"]:] for o in cfg["orbits"]}
    per = cdse.estimate_pu(grid.width, grid.height, 2)
    need = per * sum(len(v) for v in plan.values())
    log(f"取得予定 {sum(len(v) for v in plan.values())} シーン、見込み {need:.1f} PU（予算 {budget_pu:.1f} PU）")
    if need > budget_pu:
        raise cdse.CDSEError(f"見込みの処理単位 {need:.1f} PU が予算 {budget_pu:.1f} PU を超えるため取得しない")
    keep = keep_region(mask)
    water = mask == WATER
    water_px = int(water.sum())
    built = cdse.iso(now)
    summary = {"site_id": site["id"], "mask_version": tag(site), "orbits": {}, "catalog_by_orbit":
               {str(o): len(v) for o, v in by_orbit.items()}}
    for orbit, plist in plan.items():
        vvs, vhs, scenes = [], [], []
        for a in plist:
            try:
                data, pu = client.process(grid.bbox, grid.width, grid.height, a["t_from"], a["t_to"], False)
                from .run import decode_chip
                vv, vh, _ = decode_chip(data)
            except cdse.CredentialError:
                raise
            except Exception as e:
                log(f"  {orbit} {a['acquired_utc']}: 失敗 {str(e)[:200]}")
                scenes.append({"acquired_utc": a["acquired_utc"], "platform": a["platform"], "used": False,
                               "reason": "error", "error": str(e)[:200]})
                continue
            cov = float((np.isfinite(vv) & water).sum() / water_px) if water_px else 0.0
            s_vv, s_vh = smooth_db(vv), smooth_db(vh)
            wmed = float(np.nanmedian(s_vv[water])) if np.isfinite(s_vv[water]).any() else None
            sc = {"acquired_utc": a["acquired_utc"], "platform": a["platform"], "orbit_direction": a["orbit_direction"],
                  "coverage": round(cov, 3), "processing_units": pu,
                  "water_vv_median_db": round(wmed, 2) if wmed is not None else None}
            if cov < cfg["min_coverage"]:
                sc.update(used=False, reason="low_coverage")
                scenes.append(sc)
                log(f"  {orbit} {a['acquired_utc']}: 水域の {cov:.0%} しか写っていないため使わない")
                continue
            sc["used"] = True
            scenes.append(sc)
            vvs.append(s_vv.astype(np.float32))
            vhs.append(s_vh.astype(np.float32))
            log(f"  {orbit} {a['acquired_utc']} {a['platform']}: 水域の {cov:.0%}、水面 VV 中央値 {wmed} dB（{pu} PU）")
        used = [s for s in scenes if s.get("used")]
        if len(used) < 3:
            log(f"  軌道 {orbit}: 使えるシーンが {len(used)} しかないため基準を作らない")
            summary["orbits"][str(orbit)] = {"scenes": len(used), "built": False}
            continue
        layers = compute(vvs, vhs, keep, cfg)
        meta = save(site, grid, orbit, layers, scenes, cfg, mask, out_dir, built)
        summary["orbits"][str(orbit)] = {"scenes": meta["scenes"], "used": len(used), "built": True,
                                         "credibility": meta["credibility"], "first_utc": meta["first_utc"],
                                         "last_utc": meta["last_utc"], "png_bytes": meta["png_bytes"],
                                         "stats": meta["stats"]}
        if export_scenes:  # newest smoothed scenes, for tuning the quay count offline (not committed)
            ex = out_dir.parent / "scenes"
            ex.mkdir(parents=True, exist_ok=True)
            # only the shore zone (within 100 m of the water's edge on either side): smaller to export
            shore = dilate(mask != WATER, 10) & dilate(mask == WATER, 10)
            used_idx = [i for i, s in enumerate(scenes) if s.get("used")]
            for j, i in list(enumerate(used_idx))[-export_scenes:]:
                st = scenes[i]["acquired_utc"].replace(":", "-")
                for name, arr in (("vv", vvs[j]), ("vh", vhs[j])):
                    q = quantise(np.where(shore, arr, np.nan))
                    (ex / f"{site['id']}-ro{orbit}-{st}.{name}.png").write_bytes(png.encode(q, filters=[4]))
        log(f"  軌道 {orbit}: {len(used)} シーンで基準を作成（確度 {meta['credibility']}）、"
            f"常時占有 {meta['stats']['always_occupied_px']} 画素（うち OSM の水域 {meta['stats']['always_occupied_on_water_px']}）")
    return summary


def main(argv=None, client=None) -> int:
    from .run import MONTHLY_PU_CAP, OUT_DIR, load_sites, month_usage, read_csv
    from .osm_mask import MASK_DIR
    ap = argparse.ArgumentParser(description="Build the per-orbit quay baselines of a site (probe)")
    ap.add_argument("site")
    ap.add_argument("--out", type=Path, required=True, help="directory for the baseline files (not the repo)")
    ap.add_argument("--usage", type=Path, default=OUT_DIR / "usage.csv", help="published usage.csv (monthly cap)")
    ap.add_argument("--max-pu", type=float, default=130.0, help="cap for this run")
    ap.add_argument("--export-scenes", type=int, default=0)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    site = next(s for s in load_sites() if s["id"] == args.site)
    mask, _ = load_mask(site["id"], int(site.get("mask_version", 1)), MASK_DIR)
    used_pu, _req = month_usage(read_csv(args.usage), now.strftime("%Y-%m"))
    budget = min(args.max_pu, MONTHLY_PU_CAP - used_pu)
    client = client if client is not None else cdse.Client()
    rc = 0
    summary = {"run_utc": cdse.iso(now), "month_pu_before": round(used_pu, 3), "budget_pu": round(budget, 3)}
    try:
        summary.update(build(client, site, mask, args.out, now, budget, args.export_scenes))
    except cdse.CDSEError as e:
        print(f"エラー: {e}", file=sys.stderr)
        summary["error"] = str(e)[:300]
        rc = 1
    finally:
        summary.update(requests=client.requests, processing_units=round(client.processing_units, 3),
                       pu_source="header" if client.pu_reported else "estimate")
        text = json.dumps(summary, ensure_ascii=False, indent=1)
        print(text)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
