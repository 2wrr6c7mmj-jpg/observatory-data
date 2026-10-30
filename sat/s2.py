"""Sentinel-2 L2A natural colour of a site on clear days (s2nc-1, trial; display only).

2部の設計 est-2 の #4: optical images next to the Sentinel-1 colour and change images, to
check the quay counts (and, later, the hull colour of warships). Nothing here changes a count.

1. Catalog (sentinel-2-l2a) over the site's bbox; acquisitions are grouped by UTC day.
   Days whose tile cloud cover (eo:cloud_cover, the whole 110 km tile) is above
   CATALOG_MAX_CC are recorded as tile_cloudy without a request for pixels.
2. Pre-check: the scene classification (SCL) only, at 60 m (about 0.03 PU). Cloud share =
   SCL 8, 9, 10 (cloud medium / high probability, thin cirrus) and 3 (cloud shadow) over the
   valid pixels, in the whole site and in sector A. Days well above the limits stop here.
3. Full request: B2, B3, B4, B8 (reflectance x 10000, UINT16) and SCL at 10 m on the site's
   grid (about 1.5 PU), as one request with two outputs (tar). The day is clear when the
   10 m SCL gives cloud+shadow <= 10 % of the site and <= 2 % of sector A, with at least 90 %
   of the site covered. Only clear days get an image.
4. Natural colour: R = B4, G = B3, B = B2, reflectance 0-0.3 -> 0-1, gamma 1/2.2, WebP
   (with a strip: label and "Contains modified Copernicus Sentinel data <year>") and a thumbnail.

Records: s2_scenes.csv (one row per site x mask version x day, every result, with the PU of
that day), the images under s2/<site>/v<mask>/, and the PU in usage.csv (the run's row; a
standalone run, such as the backfill probe, appends its own row).

    python -m sat.s2 --days 365 [--out docs/data/sat] [--max-pu 100]   # backfill (probe)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, png
from .grid import Grid

S2_VERSION = "s2nc-1"
COLLECTION = "sentinel-2-l2a"
LOOKBACK_DAYS = 10            # the daily run looks at the last days only (new acquisitions)
CATALOG_MAX_CC = 90.0         # tile cloud cover (%) above this: no pixel request at all
PRE_M = 60                    # pre-check resolution (m)
PRE_MAX_CLOUD = 0.20          # pre-check (60 m, nearest): looser than the final limits
PRE_MAX_CLOUD_A = 0.06
MAX_CLOUD = 0.10              # final (10 m SCL): cloud + shadow in the site
MAX_CLOUD_A = 0.02            # ... and in sector A
MIN_VALID = 0.90              # share of the site with data
CLOUD_SCL = (8, 9, 10)        # cloud medium / high probability, thin cirrus
SHADOW_SCL = (3,)             # cloud shadows
NODATA = 255
REFL_SCALE = 10000.0
REFL_MAX = 0.3                # display: reflectance 0-0.3 -> 0-1
GAMMA = 2.2                   # display: n ** (1 / 2.2)
WEBP_QUALITY = 80
WEBP_METHOD = 6
THUMB_MAX = 320
THUMB_QUALITY = 70
EST_FULL_PU = 1.6             # budget check before a full request
EST_PRE_PU = 0.05
FIELDS = ["site_id", "mask_version", "date", "acquired_utc", "platform", "tile_cloud_pct",
          "pre_cloud", "pre_cloud_a", "cloud", "cloud_a", "valid", "result", "image", "thumb",
          "processing_units", "s2_version", "retrieved_utc"]
RESULTS = ("clear", "cloudy", "pre_cloudy", "tile_cloudy", "partial", "no_data", "error")

PRE_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["SCL", "dataMask"]}], output: {bands: 1, sampleType: "UINT8"}};
}
function evaluatePixel(s) {
  return [s.dataMask ? s.SCL : 255];
}
"""

FULL_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "B08", "SCL", "dataMask"]}],
          output: [{id: "refl", bands: 4, sampleType: "UINT16"}, {id: "scl", bands: 1, sampleType: "UINT8"}]};
}
function r(x) { return Math.max(0, Math.min(65535, Math.round(x * 10000))); }
function evaluatePixel(s) {
  return {refl: [r(s.B02), r(s.B03), r(s.B04), r(s.B08)], scl: [s.dataMask ? s.SCL : 255]};
}
"""


def mask_tag(site: dict) -> str:
    return f"{site['id']}-land-v{int(site.get('mask_version', 1))}"


def image_paths(site: dict, acquired_utc: str) -> tuple[str, str]:
    base = f"s2/{site['id']}/v{int(site.get('mask_version', 1))}/{acquired_utc.replace(':', '-')}"
    return base + ".webp", base + ".thumb.webp"


def credit(year: str) -> str:
    return f"Contains modified Copernicus Sentinel data {year}"


def day_groups(features: list[dict]) -> list[dict]:
    """Catalog features -> one entry per UTC day: earliest time, lowest tile cloud cover, platforms."""
    days: dict = {}
    for f in features:
        p = f.get("properties") or {}
        t = p.get("datetime")
        if not t:
            continue
        t = cdse.iso(cdse.parse_time(t))
        d = days.setdefault(t[:10], {"date": t[:10], "acquired_utc": t, "tile_cloud_pct": None, "platforms": set()})
        d["acquired_utc"] = min(d["acquired_utc"], t)
        cc = p.get("eo:cloud_cover")
        if cc is not None:
            cc = float(cc)
            d["tile_cloud_pct"] = cc if d["tile_cloud_pct"] is None else min(d["tile_cloud_pct"], cc)
        if p.get("platform"):
            d["platforms"].add(str(p["platform"]))
    out = []
    for d in sorted(days.values(), key=lambda d: d["date"]):
        d["platform"] = "+".join(sorted(d.pop("platforms")))
        out.append(d)
    return out


def sector_index(grid: Grid, site: dict) -> np.ndarray | None:
    """True where a pixel centre is in the site's first sector (A), None without sectors."""
    sec = site.get("sectors") or []
    if not sec:
        return None
    b = sec[0]["bbox"]
    lon, lat = grid.to_lonlat(np.arange(grid.height)[:, None] + 0.5, np.arange(grid.width)[None, :] + 0.5)
    lon, lat = np.broadcast_to(lon, (grid.height, grid.width)), np.broadcast_to(lat, (grid.height, grid.width))
    return (lon >= b[0]) & (lon < b[2]) & (lat >= b[1]) & (lat < b[3])


def cloud_shares(scl: np.ndarray, in_a: np.ndarray | None) -> dict:
    """Cloud + shadow share of the valid pixels (site, sector A) and the valid share of the site."""
    scl = np.asarray(scl)
    valid = (scl != NODATA) & (scl != 0)
    bad = np.isin(scl, CLOUD_SCL + SHADOW_SCL) & valid
    n = int(valid.sum())
    out = {"valid": round(n / scl.size, 4) if scl.size else 0.0,
           "cloud": round(float(bad.sum()) / n, 4) if n else None, "cloud_a": None}
    if in_a is not None and in_a.shape == scl.shape:
        na = int((valid & in_a).sum())
        out["cloud_a"] = round(float((bad & in_a).sum()) / na, 4) if na else None
    elif in_a is None:
        out["cloud_a"] = out["cloud"]
    return out


def resample_mask(in_a: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Sector mask on a coarser grid (nearest pixel centre)."""
    if in_a is None:
        return None
    h, w = in_a.shape
    r = np.minimum(((np.arange(shape[0]) + 0.5) * h / shape[0]).astype(int), h - 1)
    c = np.minimum(((np.arange(shape[1]) + 0.5) * w / shape[1]).astype(int), w - 1)
    return in_a[np.ix_(r, c)]


def natural_color(refl: np.ndarray, scl: np.ndarray | None = None) -> np.ndarray:
    """(h, w, 4) reflectance x 10000 (B2, B3, B4, B8) -> (h, w, 3) uint8: R = B4, G = B3, B = B2,
    0-0.3 -> 0-1, gamma 1/2.2. No data is black."""
    rgb = refl[..., [2, 1, 0]].astype(np.float64) / REFL_SCALE / REFL_MAX
    out = np.round(np.clip(rgb, 0.0, 1.0) ** (1.0 / GAMMA) * 255).astype(np.uint8)
    if scl is not None:
        out[(scl == NODATA) | (scl == 0)] = 0
    return out


def render(rgb: np.ndarray, label: str, credit_text: str) -> tuple[bytes, bytes]:
    """(WebP with a strip below: label, display range, credit; WebP thumbnail without it)."""
    from PIL import Image, ImageDraw
    from .color import BG, FG, LEGEND_MIN_WIDTH, _font
    chip = Image.fromarray(rgb, "RGB")
    w, h = chip.size
    font = _font(10)
    lh = 14
    lines = label.split("\n") + [f"R B4  G B3  B B2  reflectance 0-{REFL_MAX:g}, gamma 1/{GAMMA:g}  ({S2_VERSION})", credit_text]
    strip = Image.new("RGB", (max(w, LEGEND_MIN_WIDTH), 6 + lh * len(lines) + 4), BG)
    d = ImageDraw.Draw(strip)
    for i, t in enumerate(lines):
        d.text((4, 6 + i * lh), t, fill=FG, font=font)
    full = Image.new("RGB", (strip.width, h + strip.height), BG)
    full.paste(chip, (0, 0))
    full.paste(strip, (0, h))
    buf = io.BytesIO()
    full.save(buf, "WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
    th = chip.copy()
    th.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
    tbuf = io.BytesIO()
    th.save(tbuf, "WEBP", quality=THUMB_QUALITY, method=WEBP_METHOD)
    return buf.getvalue(), tbuf.getvalue()


def _payload(bbox, w, h, date: str, script: str, responses: list[str]) -> dict:
    return {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": COLLECTION,
                      "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                                     "mosaickingOrder": "leastCC"},
                      "processing": {"upsampling": "NEAREST", "downsampling": "NEAREST"}}],
        },
        "output": {"width": w, "height": h,
                   "responses": [{"identifier": r, "format": {"type": "image/png"}} for r in responses]},
        "evalscript": script,
    }


def precheck(client, grid: Grid, date: str) -> tuple[np.ndarray, float]:
    w = max(2, int(round(grid.width * 10 / PRE_M)))
    h = max(2, int(round(grid.height * 10 / PRE_M)))
    data, pu = client.process_raw(_payload(grid.bbox, w, h, date, PRE_EVALSCRIPT, ["default"]), "image/png", 1)
    return png.decode(data), pu


def fetch(client, grid: Grid, date: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(reflectance (h, w, 4) uint16, SCL (h, w) uint8, PU)."""
    data, pu = client.process_raw(_payload(grid.bbox, grid.width, grid.height, date, FULL_EVALSCRIPT, ["refl", "scl"]),
                                  "application/tar", 5)
    parts = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                parts[Path(m.name).stem] = tf.extractfile(m).read()
    refl, scl = png.decode(parts["refl"]), png.decode(parts["scl"])
    if refl.ndim != 3 or refl.shape[2] != 4 or scl.shape != refl.shape[:2]:
        raise cdse.CDSEError(f"Sentinel-2 の応答の形が想定と違う（{refl.shape} / {scl.shape}）")
    return refl, scl, pu


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["date"], r["site_id"], r["mask_version"])):
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    path.write_text(buf.getvalue(), encoding="utf-8")


def _pct(x) -> str:
    return "-" if x is None else f"{100 * float(x):.1f}%"


def update_site(client, out: Path, site: dict, now: datetime, days: int = LOOKBACK_DAYS, budget: float = 1e9,
                log=print, max_pu: float | None = None) -> dict:
    """New Sentinel-2 days of one site: records every day once (any result but error), an image on clear days.
    Stops before ``budget`` / ``max_pu`` would be exceeded. Returns a summary."""
    grid = Grid.for_site(site)
    tag = mask_tag(site)
    path = out / "s2_scenes.csv"
    rows = read_rows(path)
    done = {r["date"] for r in rows if r["site_id"] == site["id"] and r["mask_version"] == tag and r["result"] != "error"}
    pu0 = client.processing_units
    req0 = client.requests
    feats = client.catalog(grid.bbox, now - timedelta(days=days), now, collection=COLLECTION)
    in_a = sector_index(grid, site)
    summary = {"site_id": site["id"], "days": 0, "clear": 0, "cloudy": 0, "skipped": 0, "deferred": 0, "errors": 0}
    limit = budget if max_pu is None else min(budget, max_pu)
    for d in day_groups(feats):
        if d["date"] in done:
            continue
        spent = client.processing_units - pu0
        if spent + EST_PRE_PU + EST_FULL_PU > limit:
            summary["deferred"] += 1
            continue
        row = {"site_id": site["id"], "mask_version": tag, "date": d["date"], "acquired_utc": d["acquired_utc"],
               "platform": d["platform"], "tile_cloud_pct": d["tile_cloud_pct"], "s2_version": S2_VERSION,
               "retrieved_utc": cdse.iso(now)}
        pu_day = 0.0
        try:
            if d["tile_cloud_pct"] is not None and d["tile_cloud_pct"] > CATALOG_MAX_CC:
                row["result"] = "tile_cloudy"
            else:
                scl60, pu = precheck(client, grid, d["date"])
                pu_day += pu
                pc = cloud_shares(scl60, resample_mask(in_a, scl60.shape))
                row.update(pre_cloud=pc["cloud"], pre_cloud_a=pc["cloud_a"])
                if pc["cloud"] is None or pc["valid"] < MIN_VALID - 0.05:
                    row["result"] = "no_data" if pc["cloud"] is None else "partial"
                    row["valid"] = pc["valid"]
                elif pc["cloud"] > PRE_MAX_CLOUD or (pc["cloud_a"] or 0) > PRE_MAX_CLOUD_A:
                    row["result"] = "pre_cloudy"
                else:
                    refl, scl, pu = fetch(client, grid, d["date"])
                    pu_day += pu
                    c = cloud_shares(scl, in_a)
                    row.update(cloud=c["cloud"], cloud_a=c["cloud_a"], valid=c["valid"])
                    if c["cloud"] is None or c["valid"] < MIN_VALID:
                        row["result"] = "partial"
                    elif c["cloud"] <= MAX_CLOUD and (c["cloud_a"] if c["cloud_a"] is not None else 1.0) <= MAX_CLOUD_A:
                        row["result"] = "clear"
                        ip, tp = image_paths(site, d["acquired_utc"])
                        plat = (d["platform"] or "S2").replace("sentinel-", "S").upper()
                        label = (f"{site['id']} v{int(site.get('mask_version', 1))}  {d['acquired_utc']}  {plat} L2A\n"
                                 f"cloud+shadow {_pct(c['cloud'])} (sector A {_pct(c['cloud_a'])})")
                        full, thumb = render(natural_color(refl, scl), label, credit(d["acquired_utc"][:4]))
                        for rel, data in ((ip, full), (tp, thumb)):
                            (out / rel).parent.mkdir(parents=True, exist_ok=True)
                            (out / rel).write_bytes(data)
                        row.update(image=ip, thumb=tp)
                    else:
                        row["result"] = "cloudy"
        except cdse.CredentialError:
            raise
        except Exception as e:  # one bad day never stops the run
            row["result"] = "error"
            log(f"{site['id']} S2 {d['date']}: 失敗 {str(e)[:200]}")
        row["processing_units"] = round(pu_day, 3)
        rows = [r for r in rows if not (r["site_id"] == site["id"] and r["mask_version"] == tag and r["date"] == d["date"])]
        rows.append(row)
        summary["days"] += 1
        res = row["result"]
        summary["clear" if res == "clear" else "errors" if res == "error" else "cloudy" if res in ("cloudy", "pre_cloudy") else "skipped"] += 1
        log(f"{site['id']} S2 {d['date']} {res}：雲 {_pct(row.get('cloud', row.get('pre_cloud')))}"
            f"（A {_pct(row.get('cloud_a', row.get('pre_cloud_a')))}）、{pu_day:.3f} PU")
    if summary["days"] or not path.exists():
        write_rows(path, rows)
    summary["processing_units"] = round(client.processing_units - pu0, 3)
    summary["requests"] = client.requests - req0
    return summary


def settings() -> dict:
    return {
        "version": S2_VERSION,
        "status": "試行（表示だけ。数・推定には使わない）",
        "source": "Copernicus Sentinel-2 L2A（CDSE Sentinel Hub Process API、sentinel-2-l2a、日ごとに雲の少ない順のモザイク）",
        "bands": "B2・B3・B4・B8（反射率×10000、UINT16）と SCL を 10 m で 1 回の要求（2 出力の tar）。PU の見込み 約 1.5／撮影（湛江 v2）",
        "cloud": (f"範囲内の SCL の 8・9・10（雲・巻雲）と 3（雲の影）の画素の割合。範囲内 ≤ {MAX_CLOUD:.0%} かつ区画 A ≤ {MAX_CLOUD_A:.0%}、"
                  f"データのある画素 ≥ {MIN_VALID:.0%} の日だけ画像を作る（clear）"),
        "precheck": (f"カタログのタイル全体の雲量（eo:cloud_cover）が {CATALOG_MAX_CC:g}% を超える日は取得しない（tile_cloudy）。"
                     f"それ以外はまず SCL だけを {PRE_M} m で取得し（約 0.03 PU）、範囲 > {PRE_MAX_CLOUD:.0%} か区画 A > {PRE_MAX_CLOUD_A:.0%} なら 10 m は取らない（pre_cloudy）"),
        "display": (f"自然色：R＝B4、G＝B3、B＝B2、反射率 0〜{REFL_MAX:g} を 0〜1 に、γ 1/{GAMMA:g}。範囲は全日で固定。WebP（品質 {WEBP_QUALITY}、"
                    f"下に撮影・雲量・出典の帯）と縮小版（{THUMB_MAX}×{THUMB_MAX} に収める、品質 {THUMB_QUALITY}）"),
        "daily": f"毎日の実行で直近 {LOOKBACK_DAYS} 日の新しい撮影日だけを処理する（同じ日は 1 回だけ。error は取り直す）",
        "credit": "画像と meta に「Contains modified Copernicus Sentinel data <撮影の年>」",
        "results": {"clear": "晴れ（画像あり）", "cloudy": "10 m の SCL で雲が多い", "pre_cloudy": "60 m の SCL で雲が多い",
                    "tile_cloudy": "タイル全体の雲量が多く取得しなかった", "partial": "範囲の一部しか写っていない", "no_data": "データなし",
                    "error": "取得・処理の失敗（次の実行で取り直す）"},
        "limits": ["雲の判定は SCL に頼る（薄い雲・霞・雲の縁の見落とし、明るい屋根・船の雲への誤分類がある）",
                   "10 m の画素で艦の形や艦種は分からない。色（灰色の度合い）の手がかりは別の変更で記録する",
                   "Sentinel-1 との組は時刻が違う（同じ日でも数時間ずれる）：入出港で船が違うことがある"],
    }


def main(argv=None, client=None, sites=None) -> int:
    from .run import load_sites, read_csv, write_csv, USAGE_FIELDS, MONTHLY_PU_CAP, month_usage
    ap = argparse.ArgumentParser(description="Sentinel-2 L2A natural colour on clear days (trial)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "data" / "sat")
    ap.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--max-pu", type=float, default=100.0)
    ap.add_argument("--site", default=None)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    usage = read_csv(args.out / "usage.csv")
    used_pu, _ = month_usage(usage, now.strftime("%Y-%m"))
    budget = MONTHLY_PU_CAP - used_pu
    client = client if client is not None else cdse.Client()
    sums = []
    try:
        for site in (sites if sites is not None else load_sites()):
            if not site.get("enabled") or not (site.get("s2") or {}).get("enabled") or (args.site and site["id"] != args.site):
                continue
            left = min(budget, args.max_pu) - client.processing_units
            sums.append(update_site(client, args.out, site, now, args.days, left, log=lambda m: print(m, file=sys.stderr)))
    finally:
        if client.requests:
            usage.append({"run_utc": cdse.iso(now), "requests": client.requests,
                          "processing_units": round(client.processing_units, 3),
                          "pu_source": "header" if client.pu_reported else "estimate",
                          "scenes": sum(s["days"] for s in sums),
                          "s2_requests": client.requests, "s2_processing_units": round(client.processing_units, 3)})
            write_csv(args.out / "usage.csv", usage, USAGE_FIELDS)
    print(json.dumps({"run_utc": cdse.iso(now), "month_pu_before": round(used_pu, 3), "sites": sums,
                      "processing_units": round(client.processing_units, 3), "requests": client.requests},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
