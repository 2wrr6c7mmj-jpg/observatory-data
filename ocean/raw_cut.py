"""cfar-v2 PR3 probe, mode ``raw``: raw sigma0 tiles for finding why cfar-v2 fits fail (probe only).

    python -m ocean.cfar_probe --mode raw --out DIR --raw-out probe-raw/ocean/RUN ...

Re-fetches selected open-sea scenes (``ocean/raw_targets.json``, chosen from an earlier probe run) at
20 m with BILINEAR, the same request as the compare probe (VV, VH, local incidence angle, dataMask,
UINT16, SIGMA0_ELLIPSOID, orthorectified). The whole scene is never written: cfar-v2 is run on it and
at most ``MAX_TILES`` 2.56 km tiles (128 x 128 px) are cut out -- tiles without a model fit (own fit
failed) and, for comparison, tiles with a normal fit (and tiles with fixed objects such as wind
turbines when there are any) -- and saved as one compressed ``.npz`` per scene with a JSON meta file.

The raw UINT16 codes are kept as delivered: ``dB = code / 800 - 50`` for ``code > 0``; ``code = 0`` with
``dataMask = 1`` means sigma0 <= 0 or NaN in the source (the evalscript writes 0 for ``!(x > 0)``);
``LIA deg = code / 100``. Per-tile diagnostics of all 400 tiles (zero-code shares, no-data band,
usable share, model) go into the meta file, so the cause can be studied without new requests.
"""
from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from sat import cdse, png
from sat import detect2 as v2
from sat.grid import dilate

TARGETS_FILE = Path(__file__).resolve().parent / "raw_targets.json"
RAW_VERSION = "ocean-raw-cut-1"
TILE_PX = 128
MAX_TILES = 12
N_FAILED = 8          # up to this many unfitted tiles, the rest normal (or fixed-object) tiles
ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026"
ENCODING = {
    "vv": "dB = code / 800 - 50 (code > 0); code 0 with datamask 1 = sigma0 <= 0 or NaN in the source",
    "vh": "same as vv",
    "lia": "local incidence angle deg = code / 100",
    "datamask": "Sentinel Hub dataMask (1 = data)",
    "land": "1 = land from OSM coastlines (sat.osm_mask), on the same grid",
    "static": "1 = fixed object from OSM (wind turbine, platform, buoy/beacon), dilated 300 m",
    "tiles": "rows of [ty, tx, r0, c0] in the 2,500 x 2,500 px scene grid (128 px tiles)",
}


def load_targets(path: Path = TARGETS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_acq(features: list[dict], when: str, tol_s: float = 600) -> dict | None:
    t = cdse.parse_time(when)
    best = None
    for a in cdse.acquisitions(features):
        d = abs((a["time"] - t).total_seconds())
        if d <= tol_s and (best is None or d < best[0]):
            best = (d, a)
    return best[1] if best else None


def decode_raw(data: bytes) -> np.ndarray:
    arr = png.decode(data)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise cdse.CDSEError(f"想定外の画像（形 {arr.shape}）")
    return arr.astype(np.uint16)


def to_db(code: np.ndarray, dm: np.ndarray) -> np.ndarray:
    c = code.astype(np.float64)
    return np.where(dm & (code > 0), c / cdse.DB_SCALE - cdse.DB_OFFSET, np.nan)


def tile_diagnostics(raw: np.ndarray, land: np.ndarray, static: np.ndarray, v2_tiles: list[dict],
                     edge_px: int = 3, tile: int = TILE_PX) -> list[dict]:
    """Per tile: shares of dataMask, zero VV/VH codes (inside dataMask), the no-data band that cfar-v2
    removes (``dilate(~valid, edge_px)``), land, and the cfar-v2 fit (model, usable, n_fit)."""
    dm = raw[..., 3] > 0
    vv0 = dm & (raw[..., 0] == 0)
    vh0 = dm & (raw[..., 1] == 0)
    valid = dm & ~vv0 & ~vh0
    band = dilate(~valid, edge_px) if (~valid).any() else np.zeros_like(valid)
    by = {(t["ty"], t["tx"]): t for t in v2_tiles}
    h, w = dm.shape
    out = []
    for ty in range(-(-h // tile)):
        for tx in range(-(-w // tile)):
            sl = (slice(ty * tile, min(h, (ty + 1) * tile)), slice(tx * tile, min(w, (tx + 1) * tile)))
            n = dm[sl].size
            ndm = int(dm[sl].sum())
            t = by.get((ty, tx), {})
            vvc = raw[..., 0][sl][valid[sl]]
            vhc = raw[..., 1][sl][valid[sl]]
            lia = raw[..., 2][sl][dm[sl]]
            out.append({
                "ty": ty, "tx": tx,
                "datamask": round(ndm / n, 4),
                "vv_zero": round(float(vv0[sl].sum()) / max(ndm, 1), 4),
                "vh_zero": round(float(vh0[sl].sum()) / max(ndm, 1), 4),
                "nodata_band": round(float((band[sl] & ~land[sl]).sum()) / n, 4),
                "land": round(float(land[sl].mean()), 4),
                "static": bool(static[sl].any()),
                "usable": t.get("usable_fraction"), "n_fit": t.get("n_fit"),
                "model": t.get("model"), "own_model": t.get("own_model"), "borrowed": t.get("borrowed"),
                "vv_med_db": round(float(np.median(vvc)) / cdse.DB_SCALE - cdse.DB_OFFSET, 2) if vvc.size else None,
                "vh_med_db": round(float(np.median(vhc)) / cdse.DB_SCALE - cdse.DB_OFFSET, 2) if vhc.size else None,
                "vh_p05_db": round(float(np.percentile(vhc, 5)) / cdse.DB_SCALE - cdse.DB_OFFSET, 2) if vhc.size else None,
                "lia_med": round(float(np.median(lia)) / cdse.LIA_SCALE, 2) if lia.size else None,
            })
    return out


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = sorted({int(round(v)) for v in np.linspace(0, len(items) - 1, n)})
    return [items[i] for i in idx]


def choose_tiles(diag: list[dict], max_tiles: int = MAX_TILES, n_failed: int = N_FAILED) -> list[dict]:
    """Unfitted sea tiles (own fit failed, data inside the tile) and normally fitted ones, spread over the
    scene; up to 2 tiles with fixed objects first among the normal ones when there are any."""
    sea = [d for d in diag if d["datamask"] >= 0.5 and d["land"] <= 0.5]
    full = [d for d in diag if d["datamask"] >= 0.95 and d["land"] <= 0.05]
    pool_fail = [d for d in (full or sea) if d["own_model"] is None] or [d for d in sea if d["own_model"] is None]
    pool_ok = [d for d in (full or sea) if d["own_model"] is not None] or [d for d in sea if d["own_model"] is not None]
    stat = [d for d in sea if d["static"]]
    k_fail = min(len(pool_fail), n_failed if pool_ok else max_tiles)
    pick = _spread(pool_fail, k_fail)
    rest = max_tiles - len(pick)
    st = _spread(stat, min(2, rest))
    pick += [d for d in st if d not in pick]
    rest = max_tiles - len(pick)
    pick += [d for d in _spread([d for d in pool_ok if d not in pick], rest)]
    if len(pick) < max_tiles:  # few normal tiles: fill with more unfitted ones
        pick += [d for d in _spread([d for d in pool_fail if d not in pick], max_tiles - len(pick))]
    return pick[:max_tiles]


def cut(raw: np.ndarray, land: np.ndarray, static: np.ndarray, chosen: list[dict], tile: int = TILE_PX) -> dict:
    h, w = raw.shape[:2]
    arrs = {"vv": [], "vh": [], "lia": [], "datamask": [], "land": [], "static": [], "tiles": []}
    for d in chosen:
        r0, c0 = d["ty"] * tile, d["tx"] * tile
        r1, c1 = min(h, r0 + tile), min(w, c0 + tile)
        pad = lambda a: np.pad(a, ((0, tile - (r1 - r0)), (0, tile - (c1 - c0))))
        arrs["vv"].append(pad(raw[r0:r1, c0:c1, 0]))
        arrs["vh"].append(pad(raw[r0:r1, c0:c1, 1]))
        arrs["lia"].append(pad(raw[r0:r1, c0:c1, 2]))
        arrs["datamask"].append(pad((raw[r0:r1, c0:c1, 3] > 0).astype(np.uint8)))
        arrs["land"].append(pad(land[r0:r1, c0:c1].astype(np.uint8)))
        arrs["static"].append(pad(static[r0:r1, c0:c1].astype(np.uint8)))
        arrs["tiles"].append([d["ty"], d["tx"], r0, c0])
    out = {k: np.stack(v).astype(np.uint16 if k in ("vv", "vh", "lia") else np.uint8) for k, v in arrs.items() if k != "tiles"}
    out["tiles"] = np.array(arrs["tiles"], dtype=np.int32).reshape(-1, 4)
    return out


def tile_bbox(grid, r0: int, c0: int, tile: int = TILE_PX) -> list[float]:
    w, n = grid.to_lonlat(r0, c0)
    e, s = grid.to_lonlat(min(grid.height, r0 + tile), min(grid.width, c0 + tile))
    return [round(float(w), 6), round(float(s), 6), round(float(e), 6), round(float(n), 6)]


def run_raw(args, cp, client, overpass, now, pacer, sleep, log, clock) -> dict:
    """``cp`` is the ocean.cfar_probe module (shared areas, masks, budget and pacing)."""
    month = args.budget_month
    bud = cp.budget(args, now)  # MonthStop outside the budget month (sat.port_archive.guard)
    cfg = cp.load_areas(args.areas)
    pixel_m = float(cfg["pixel_m"])
    tg = load_targets(args.targets or TARGETS_FILE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = Path(args.raw_out) if args.raw_out else None
    summary = {"probe_version": RAW_VERSION, "mode": "raw", "run_utc": cdse.iso(now), "targets_version": tg["version"],
               "pixel_m": pixel_m, "interpolation": cp.INTERPOLATION, "dry_run": bool(args.dry_run), "budget": bud,
               "raw_dir": str(raw_out) if raw_out else None, "attribution": ATTRIBUTION,
               "plan": {}, "results": [], "errors": []}
    log(f"今月 {month}：sat 等 {bud['sat_month_pu']} PU・港 {bud['ports_month_pu']} PU・外洋 {bud['ocean_month_pu']} PU 使用済み → この実行で使える {bud['pu_left']} PU")
    areas = {a["id"]: a for a in cfg["areas"]}
    plan = []
    for t in tg["targets"]:
        area = areas[t["area"]]
        u = cp.units(area, pixel_m)[0]
        g = u["grid"]
        tt = cdse.parse_time(t["acquired_utc"])
        try:
            if cp.month_changed(args, clock):
                raise cp.MonthStop(cp.MONTH_STOP)
            feats = client.catalog(g.bbox, tt - timedelta(hours=2), tt + timedelta(hours=2))
        except cp.MonthStop as e:  # nothing more is sent; the plan is dropped
            summary["errors"].append({"stage": "stop", "reason": "month", "error": str(e)})
            plan = []
            break
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as e:
            summary["errors"].append({"target": t, "stage": "catalog", "error": str(e)[:300]})
            continue
        a = find_acq(feats, t["acquired_utc"])
        if a is None:
            summary["errors"].append({"target": t, "stage": "catalog", "error": "撮影がカタログに無い"})
            continue
        by_id = {f.get("id"): f for f in feats}
        a["footprint_coverage"] = cp.footprint_coverage([(by_id.get(i) or {}).get("geometry") for i in a["ids"]], g)
        a["ids_"] = list(a["ids"])
        plan.append({"target": t, "area": area, "unit": u["unit"], "grid": g, "acq": a,
                     "est_pu": cdse.estimate_pu(g.width, g.height, 3)})
    total, kept_plan = 0.0, []
    for p in plan:
        if total + p["est_pu"] > bud["pu_left"] + 1e-9 or len(kept_plan) + 1 > bud["requests_left"]:
            summary["errors"].append({"target": p["target"], "stage": "plan", "error": "上限のため見送り"})
            continue
        total += p["est_pu"]
        kept_plan.append(p)
    summary["plan"] = {"requests": len(kept_plan), "est_pu": round(total, 3),
                       "scenes": [{"area": p["target"]["area"], **cp._acq_meta(p["acq"])} for p in kept_plan]}
    log("計画: " + json.dumps(summary["plan"], ensure_ascii=False))
    if args.dry_run or not kept_plan:
        _finish(args, cp, summary, client, now, out, log, pacer)
        return summary

    seamark_re = cfg["osm_static"]["seamark_types"]
    osm_cache, mask_cache = {}, {}
    for k, aid in enumerate(sorted({p["area"]["id"] for p in kept_plan})):
        if k:
            sleep(10)
        try:
            osm_cache[aid] = overpass(cp.overpass_query(cp.area_grid(areas[aid], pixel_m), seamark_re))
        except Exception as e:
            summary["errors"].append({"area": aid, "stage": "overpass", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    if raw_out:
        raw_out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_minutes * 60
    for p in kept_plan:
        aid, g, a = p["area"]["id"], p["grid"], p["acq"]
        if cp.month_changed(args, clock):
            summary["errors"].append({"stage": "stop", "reason": "month", "error": cp.MONTH_STOP})
            break
        if time.monotonic() > deadline:
            summary["errors"].append({"stage": "stop", "error": f"{args.max_minutes} 分を過ぎたため止めた"})
            break
        if aid not in osm_cache:
            continue
        if client.processing_units + p["est_pu"] > bud["pu_left"] + 1e-9:
            summary["errors"].append({"stage": "stop", "error": "実際の PU が上限に届くため止めた"})
            break
        pacer.wait(p["est_pu"])
        try:
            data, pu = client.process(g.bbox, g.width, g.height, a["t_from"], a["t_to"], with_lia=True,
                                      upsampling=cp.INTERPOLATION)
            pacer.spent(pu)
            raw = decode_raw(data)
            del data
            if aid not in mask_cache:
                mask_cache[aid] = cp.masks(g, osm_cache[aid], seamark_re, float(cfg["static_dilate_m"]))
            land, static, _minfo = mask_cache[aid]
            dm = raw[..., 3] > 0
            vv, vh = to_db(raw[..., 0], dm), to_db(raw[..., 1], dm)
            lia = np.where(dm, raw[..., 2] / cdse.LIA_SCALE, np.nan)
            lat = (g.south + g.north) / 2
            heading = v2.platform_heading((a.get("orbit_direction") or "").lower(), lat) if a.get("orbit_direction") else None
            _d2, i2 = v2.detect_v2(vv, vh, land, g.pixel_m, lia=lia, heading_deg=heading, static=static, datamask=dm)
            diag = tile_diagnostics(raw, land, static, i2["tile_table"])
            chosen = choose_tiles(diag)
            arrs = cut(raw, land, static, chosen)
            name = f"{aid}-{cp.stamp(a['acquired_utc'])}"
            row = {"area": aid, "why": p["target"]["why"], **cp._acq_meta(a), "absolute_orbit": a.get("absolute_orbit"),
                   "pu": pu, "tiles_cut": len(chosen),
                   "tiles_cut_unfitted": sum(1 for d in chosen if d["own_model"] is None),
                   "scene": _scene_stats(raw, land, diag)}
            if raw_out:
                np.savez_compressed(raw_out / f"{name}.npz", **arrs)
                meta = {"version": RAW_VERSION, "attribution": ATTRIBUTION,
                        "source": "Copernicus Data Space Ecosystem, Sentinel Hub Process API, collection sentinel-1-grd",
                        "acquisition": {**cp._acq_meta(a), "absolute_orbit": a.get("absolute_orbit"),
                                        "catalog_ids": a["ids_"], "t_from": cdse.iso(a["t_from"]), "t_to": cdse.iso(a["t_to"])},
                        "request": {"bbox_epsg4326": g.bbox, "width": g.width, "height": g.height, "pixel_m": list(g.pixel_m),
                                    "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH",
                                    "backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True, "demInstance": "COPERNICUS_30",
                                    "upsampling": cp.INTERPOLATION, "sampleType": "UINT16", "pu": pu},
                        "encoding": ENCODING, "evalscript": cdse.evalscript(True),
                        "tiles": [{"ty": d["ty"], "tx": d["tx"], "bbox": tile_bbox(g, d["ty"] * TILE_PX, d["tx"] * TILE_PX),
                                   "fitted": d["own_model"] is not None, **{k: d[k] for k in ("model", "own_model", "usable", "n_fit",
                                   "vv_zero", "vh_zero", "nodata_band", "static")}} for d in chosen],
                        "why": p["target"]["why"], "area": aid,
                        "v2_info": {k: i2.get(k) for k in ("model_counts", "model_counts_vh", "tiles", "tiles_borrowed",
                                                           "tiles_untested", "exceedance", "z_t_seed")},
                        "all_tiles_cols": list(diag[0].keys()) if diag else [],
                        "all_tiles": [list(d.values()) for d in diag]}
                (raw_out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":"),
                                                                 default=cp._jsonable), encoding="utf-8")
                row["npz_bytes"] = (raw_out / f"{name}.npz").stat().st_size
                row["files"] = [f"{name}.npz", f"{name}.json"]
            summary["results"].append(row)
            log(f"{aid} {a['acquired_utc']}: PU {pu}、切り出し {row['tiles_cut']}（未当てはめ {row['tiles_cut_unfitted']}）"
                f"、{row.get('npz_bytes')} B、{json.dumps(row['scene'], ensure_ascii=False)}")
        except cp.MonthStop as e:
            summary["errors"].append({"stage": "stop", "reason": "month", "error": str(e)})
            break
        except cdse.CredentialError:
            raise
        except cdse.CDSEError as e:
            summary["errors"].append({"area": aid, "acquired_utc": a["acquired_utc"], "stage": "fetch", "error": str(e)[:300]})
        cp._write_summary(out, summary, client, pacer)
    _finish(args, cp, summary, client, now, out, log, pacer)
    return summary


def _scene_stats(raw, land, diag) -> dict:
    dm = (raw[..., 3] > 0) & ~land
    n = max(int(dm.sum()), 1)
    sea_tiles = [d for d in diag if d["datamask"] >= 0.5 and d["land"] <= 0.5]
    return {"vv_zero": round(float((dm & (raw[..., 0] == 0)).sum()) / n, 4),
            "vh_zero": round(float((dm & (raw[..., 1] == 0)).sum()) / n, 4),
            "sea_tiles": len(sea_tiles), "sea_tiles_unfitted": sum(1 for d in sea_tiles if d["own_model"] is None),
            "sea_tiles_no_model": sum(1 for d in sea_tiles if d["model"] is None)}


def _finish(args, cp, summary, client, now, out, log, pacer=None) -> None:
    cp._write_summary(out, summary, client, pacer)
    cp.write_usage(Path(args.usage), {"run_utc": cdse.iso(now), "requests": client.requests,
                                      "processing_units": round(client.processing_units, 3),
                                      "pu_source": "header" if getattr(client, "pu_reported", True) else "estimate",
                                      "scenes": len(summary["results"]), "units": len(summary["results"]),
                                      "probe_version": RAW_VERSION, "note": "dry-run" if args.dry_run else "raw tiles"})
    lines = [f"{RAW_VERSION} {summary['run_utc']}{'（dry-run）' if summary['dry_run'] else ''}",
             f"使える {summary['budget']['pu_left']} PU、計画 {summary['plan'].get('requests')} 件 {summary['plan'].get('est_pu')} PU、"
             f"実績 {client.requests} リクエスト・{round(client.processing_units, 3)} PU"]
    for r in summary["results"]:
        lines.append(f"{r['area']} {r['acquired_utc']} {r.get('orbit_direction')}: 切り出し {r['tiles_cut']}"
                     f"（未当てはめ {r['tiles_cut_unfitted']}）{r.get('npz_bytes')} B {json.dumps(r['scene'], ensure_ascii=False)}")
    for e in summary["errors"]:
        lines.append("エラー: " + json.dumps(e, ensure_ascii=False, default=str))
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
