"""Daily run: find new Sentinel-1 scenes per site, count ship candidates, publish.

    python -m sat.run                      # normal run, writes docs/data/sat/
    python -m sat.run --probe DIR          # writes only under DIR (no repo data change)

Each acquisition (one pass over a site) is processed once; processed
acquisitions are skipped on later runs. Processing units (PU) and requests
are recorded in usage.csv and a monthly cap stops the run before the free
tier is at risk.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import baseline as fbase
from . import cdse, png, quay, quay_lines, render
from . import color as fcolor
from .detect import DEFAULTS as DETECT_DEFAULTS, DETECTOR_VERSION, LENGTH_CLASSES, detect
from .grid import Grid
from .osm_mask import MASK_DIR, build_mask, fetch_osm, load_mask, save_mask

ROOT = Path(__file__).resolve().parent.parent
SITES_FILE = Path(__file__).resolve().parent / "sites.json"
OUT_DIR = ROOT / "docs" / "data" / "sat"
PROCESSING_VERSION = "sat-2"  # sat-2: rectangular sites and sectors (A/B) per site
FIRST_LOOKBACK_DAYS = 30   # first run for a site (free tier: keep the backfill short)
LOOKBACK_DAYS = 6          # later runs: the last few days
MAX_LOOKBACK_DAYS = 30
MAX_SCENES_PER_RUN = 12
MONTHLY_PU_CAP = 1000.0    # own cap for this workflow, well inside the CDSE free tier
MONTHLY_REQUEST_CAP = 2000
MIN_COVERAGE = 0.5         # share of the site's water that must have data
SOURCE_RELIABILITY = "A"
CANDIDATE_CREDIBILITY = "3"

CSV_FIELDS = [
    "site_id", "acquired_utc", "retrieved_utc", "platform", "orbit_direction", "relative_orbit",
    "incidence_deg", "harbour_candidates", "quay_candidates", "confirmed",
    "mask_version", "detector_version", "processing_version",
    "coverage", "len_lt50", "len_50_100", "len_100_200", "len_ge200", "processing_units", "image",
    "sector_a_harbour", "sector_a_quay", "sector_b_harbour", "sector_b_quay",
    "color_image", "color_thumb",
]
SECTOR_COLUMNS = ("a", "b")  # the first sector of a site is "a"; the rest of the chip is "b"

USAGE_FIELDS = ["run_utc", "requests", "processing_units", "pu_source", "scenes"]

# Quay count (quay-v1, trial): one row per scene x site x mask version, only where a baseline
# of the scene's relative orbit exists. Separate from ship_counts.csv (the CFAR count stays as it is).
QUAY_FIELDS = [
    "site_id", "acquired_utc", "platform", "orbit_direction", "relative_orbit", "mask_version",
    "quay_version", "lines_version", "baseline_version", "baseline_scenes", "credibility",
    "occupied_min", "occupied_max", "sector_a_min", "sector_a_max", "sector_b_min", "sector_b_max",
    "segments", "rafted", "quay_coverage", "change_image", "change_thumb",
]


def load_sites(path: Path = SITES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["sites"]


def mask_version(site: dict) -> int:
    return int(site.get("mask_version", 1))


def mask_tag(site: dict) -> str:
    return f"{site['id']}-land-v{mask_version(site)}"


def row_mask_tag(r: dict) -> str:
    """Mask version of a published row or scene (entries written before v2 have none: v1)."""
    return r.get("mask_version") or f"{r['site_id']}-land-v1"


def site_sectors(site: dict) -> list[dict]:
    """[{id, name, bbox|None}] for a site: its listed sectors, then the rest of the chip.

    At most one listed sector (columns sector_a_* / sector_b_*). No sectors: [].
    """
    listed = site.get("sectors") or []
    if not listed:
        return []
    if len(listed) > 1:
        raise ValueError(f"{site['id']}: 区画は1つまで（残りは自動で B）")
    rest = site.get("sector_rest") or {"id": "B", "name": "それ以外"}
    return [{"id": listed[0]["id"], "name": listed[0].get("name"), "bbox": listed[0]["bbox"]},
            {"id": rest["id"], "name": rest.get("name"), "bbox": None}]


def sector_of(sectors: list[dict], lon: float, lat: float) -> str | None:
    for sc in sectors:
        b = sc["bbox"]
        if b is None or (b[0] <= lon < b[2] and b[1] <= lat < b[3]):
            return sc["id"]
    return None


def sector_raster(grid: Grid, sectors: list[dict]) -> np.ndarray | None:
    """Index of the sector (0, 1) of each pixel centre, or None without sectors."""
    if not sectors:
        return None
    lon, lat = grid.to_lonlat(np.arange(grid.height)[:, None] + 0.5, np.arange(grid.width)[None, :] + 0.5)
    lon, lat = np.broadcast_to(lon, (grid.height, grid.width)), np.broadcast_to(lat, (grid.height, grid.width))
    b = sectors[0]["bbox"]
    inside = (lon >= b[0]) & (lon < b[2]) & (lat >= b[1]) & (lat < b[3])
    return np.where(inside, 0, 1).astype(np.uint8)


def image_path(site: dict, acquired_utc: str) -> str:
    v = mask_version(site)
    sub = "" if v == 1 else f"v{v}/"
    return f"images/{site['id']}/{sub}{stamp(acquired_utc)}.png"


def color_paths(site: dict, acquired_utc: str) -> tuple[str, str]:
    """False-colour WebP and its thumbnail (a directory of their own; the PNG stays)."""
    base = f"color/{site['id']}/v{mask_version(site)}/{stamp(acquired_utc)}"
    return base + ".webp", base + ".thumb.webp"


def change_paths(site: dict, acquired_utc: str) -> tuple[str, str]:
    """Change image (R = this VV, G = B = baseline VV) and its thumbnail."""
    base = f"change/{site['id']}/v{mask_version(site)}/{stamp(acquired_utc)}"
    return base + ".webp", base + ".thumb.webp"


def color_label(site: dict, acq: dict) -> str:
    plat = (acq.get("platform") or "").replace("sentinel-", "S").upper()
    od = {"ascending": "asc", "descending": "desc"}.get(acq.get("orbit_direction"), acq.get("orbit_direction") or "?")
    return f"{site['id']} {mask_tag(site)}  {acq['acquired_utc']}  {plat} {od} RO {acq.get('relative_orbit') or '?'}"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def month_usage(usage: list[dict], month: str) -> tuple[float, int]:
    pu = sum(float(u["processing_units"] or 0) for u in usage if u["run_utc"].startswith(month))
    req = sum(int(u["requests"] or 0) for u in usage if u["run_utc"].startswith(month))
    return pu, req


def decode_chip(data: bytes):
    arr = png.decode(data).astype(np.float64)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise cdse.CDSEError(f"想定外の画像（形 {arr.shape}）")
    dm = arr[..., -1] > 0
    def db(ch):
        return np.where(dm & (ch > 0), ch / cdse.DB_SCALE - cdse.DB_OFFSET, np.nan)
    vv, vh = db(arr[..., 0]), db(arr[..., 1])
    lia = np.where(dm, arr[..., 2] / cdse.LIA_SCALE, np.nan) if arr.shape[2] == 4 else None
    return vv, vh, lia


def stamp(acquired_utc: str) -> str:
    return acquired_utc.replace(":", "-")


class Store:
    """The published files under ``out``."""

    def __init__(self, out: Path, sites: list[dict] | None = None):
        self.out = out
        self.sites = sites if sites is not None else load_sites()
        self.rows = read_csv(out / "ship_counts.csv")
        self.usage = read_csv(out / "usage.csv")
        st = out / "status.json"
        self.status = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {"scenes": []}
        self.new_features: dict[str, list[dict]] = {}
        self.quay_rows = read_csv(out / "quay_counts.csv")
        self.new_quay: dict[str, list] = {}

    # Rows and scenes are kept per mask version: a new version (a new site extent) starts
    # its own series and re-reads recent scenes; rows of the old version stay as they were.
    def done(self, site_id: str, tag: str) -> set[str]:
        keys = {r["acquired_utc"] for r in self.rows if r["site_id"] == site_id and row_mask_tag(r) == tag}
        keys |= {s["acquired_utc"] for s in self.status.get("scenes", [])
                 if s["site_id"] == site_id and row_mask_tag(s) == tag and s.get("result") != "error"}
        return keys

    def quay_done(self, site_id: str, tag: str) -> set[str]:
        """Scenes with a quay row, or whose quay count was tried (an error is not retried)."""
        keys = {r["acquired_utc"] for r in self.quay_rows if r["site_id"] == site_id and row_mask_tag(r) == tag}
        keys |= {s["acquired_utc"] for s in self.status.get("scenes", [])
                 if s["site_id"] == site_id and row_mask_tag(s) == tag and s.get("quay_result") in ("counted", "error")}
        return keys

    def add_quay(self, row: dict, features: list[dict], extra: dict[str, bytes] | None = None) -> None:
        key = (row["site_id"], row["acquired_utc"], row_mask_tag(row))
        self.quay_rows = [r for r in self.quay_rows if (r["site_id"], r["acquired_utc"], row_mask_tag(r)) != key]
        self.quay_rows.append(row)
        self.quay_rows.sort(key=lambda r: (r["acquired_utc"], r["site_id"], row_mask_tag(r)))
        for rel, data in (extra or {}).items():
            q = self.out / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
        self.new_quay.setdefault(row["acquired_utc"][:7], []).append((row, features))

    def has_rows(self, site_id: str, tag: str) -> bool:
        return any(r["site_id"] == site_id and row_mask_tag(r) == tag for r in self.rows)

    def note_scene(self, entry: dict) -> None:
        key = (entry["site_id"], entry["acquired_utc"], row_mask_tag(entry))
        sc = [s for s in self.status.get("scenes", []) if (s["site_id"], s["acquired_utc"], row_mask_tag(s)) != key]
        sc.append(entry)
        sc.sort(key=lambda s: (s["acquired_utc"], s["site_id"], row_mask_tag(s)))
        self.status["scenes"] = sc

    def add(self, row: dict, features: list[dict], image: bytes, extra: dict[str, bytes] | None = None) -> None:
        key = (row["site_id"], row["acquired_utc"], row_mask_tag(row))
        self.rows = [r for r in self.rows if (r["site_id"], r["acquired_utc"], row_mask_tag(r)) != key]
        self.rows.append(row)
        self.rows.sort(key=lambda r: (r["acquired_utc"], r["site_id"], row_mask_tag(r)))
        p = self.out / row["image"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(image)
        for rel, data in (extra or {}).items():
            q = self.out / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
        self.new_features.setdefault(row["acquired_utc"][:7], []).append((row, features))

    def save(self, run: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        write_csv(self.out / "ship_counts.csv", self.rows, CSV_FIELDS)
        for month, items in self.new_features.items():
            p = self.out / "detections" / f"{month}.geojson"
            fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"type": "FeatureCollection", "features": []}
            fkey = lambda p: (p["site_id"], p["acquired_utc"], row_mask_tag(p))
            keys = {fkey(r) for r, _ in items}
            feats = [f for f in fc["features"] if fkey(f["properties"]) not in keys]
            for _r, fs in items:
                feats.extend(fs)
            feats.sort(key=lambda f: (f["properties"]["acquired_utc"], f["properties"]["site_id"],
                                      row_mask_tag(f["properties"]), f["properties"]["n"]))
            fc["features"] = feats
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        if self.quay_rows or self.new_quay:
            write_csv(self.out / "quay_counts.csv", self.quay_rows, QUAY_FIELDS)
        for month, items in self.new_quay.items():
            p = self.out / "quay" / f"{month}.geojson"
            fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"type": "FeatureCollection", "features": []}
            fkey = lambda p: (p["site_id"], p["acquired_utc"], row_mask_tag(p))
            keys = {fkey(r) for r, _ in items}
            feats = [f for f in fc["features"] if fkey(f["properties"]) not in keys]
            for _r, fs in items:
                feats.extend(fs)
            feats.sort(key=lambda f: (f["properties"]["acquired_utc"], f["properties"]["site_id"],
                                      row_mask_tag(f["properties"]), f["properties"]["n"]))
            fc["features"] = feats
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        if run.get("requests"):
            self.usage.append({k: run.get(k) for k in USAGE_FIELDS})
        write_csv(self.out / "usage.csv", self.usage, USAGE_FIELDS)
        self.status["last_run"] = run
        self.status["updated_utc"] = run["run_utc"]
        self.status["series_breaks"] = series_breaks(self.sites)
        (self.out / "status.json").write_text(json.dumps(self.status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (self.out / "meta.json").write_text(json.dumps(build_meta(self.rows, self.sites), ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")


def attribution(rows: list[dict]) -> str:
    years = sorted({r["acquired_utc"][:4] for r in rows})
    if not years:
        years = [str(datetime.now(timezone.utc).year)]
    span = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return f"Contains modified Copernicus Sentinel data {span}"


def series_breaks(sites: list[dict]) -> list[dict]:
    """Where a site's counts start a new series (a new mask version = a new extent)."""
    out = []
    for site in sites:
        for h in site.get("history") or []:
            out.append({
                "site_id": site["id"],
                "from_mask_version": f"{site['id']}-land-v{h['mask_version']}",
                "to_mask_version": mask_tag(site),
                "date_utc": h.get("until_utc"),
                "note": (f"監視範囲を変更（旧範囲 v{h['mask_version']}：{h.get('note', '')}）。mask_version が違う行の数は範囲が違うため直接比べない。"
                         "新しい版は過去 30 日の撮影も新しい範囲で数え直し、旧版の行は残す（同じ撮影に両方の行がある）"),
            })
    return out


def baseline_meta(sites: list[dict]) -> list[dict]:
    out = []
    for site in sites:
        if not site.get("enabled"):
            continue
        for o in fbase.site_config(site)["orbits"]:
            bl = fbase.load(site, o)
            if bl is None:
                continue
            m = bl[1]
            out.append({"site_id": site["id"], "mask_version": m["mask_version"], "relative_orbit": m["relative_orbit"],
                        "version": f"ro{m['relative_orbit']}-{m['version']}", "scenes": m["scenes"],
                        "first_utc": m["first_utc"], "last_utc": m["last_utc"], "credibility": m["credibility"],
                        "file": f"sat/baselines/{m['mask_version']}/ro{m['relative_orbit']}-{m['version']}.json"})
    return out


def site_meta(site: dict) -> dict:
    grid = Grid.for_site(site)
    m = {"id": site["id"], "name": site.get("name"), "mask_version": mask_tag(site), "bbox": grid.bbox,
         "width_px": grid.width, "height_px": grid.height, "area_km2": round(grid.area_km2(), 1),
         "provisional": bool(site.get("provisional")), "note": site.get("note")}
    sec = site_sectors(site)
    if sec:
        m["sectors"] = [{"column": f"sector_{c}", **sc} for c, sc in zip(SECTOR_COLUMNS, sec)]
    return m


def build_meta(rows: list[dict], sites: list[dict] | None = None) -> dict:
    sites = sites if sites is not None else load_sites()
    return {
        "dataset": "sat",
        "title": "Sentinel-1 による港内の艦船候補数（試行：湛江 1 地点）",
        "status": ("試行（陸マスクのみ。固定物マスクは 20 シーン蓄積後）。湛江（ZJG）は 2026-10-30 に監視範囲を v2（東岸の軍港を中心とする矩形）"
                   "に変えた。ZJG-land-v1 の行と ZJG-land-v2 の行は範囲が違うため直接比べない（series_breaks）"),
        "sites": [site_meta(s) for s in sites if s.get("enabled")],
        "series_breaks": series_breaks(sites),
        "source": {
            "name": "Copernicus Sentinel-1 IW GRD（Copernicus Data Space Ecosystem, Sentinel Hub Process API）",
            "position": "欧州連合 Copernicus 計画の公式データ",
            "reliability": SOURCE_RELIABILITY,
            "credibility_of_candidates": CANDIDATE_CREDIBILITY,
            "rating_note": "情報源は A（Copernicus 公式）。自動検出はすべて「候補」で、確度は 3（未確認）とする。",
        },
        "attribution": {
            "credit": attribution(rows),
            "land_mask": "陸マスク：(c) OpenStreetMap contributors, ODbL 1.0",
            "processing": "observatory-data が σ0（地形補正済み、10 m）の切り出しから自動で数えた。数値は ESA・欧州委員会が作成したものではない。",
        },
        "files": {
            "ship_counts.csv": "1 行＝1 撮影×1 地点×1 範囲版（mask_version）",
            "detections/YYYY-MM.geojson": "撮影月ごとの候補（点、長さの目安つき）",
            "color/<地点>/v<版>/<撮影時刻>.webp": "疑似カラー（R＝VV、G＝VH、B＝VV−VH を逆向き）の WebP。範囲は全地点・全日で固定、表示用だけ 3×3 Lee フィルタ、凡例と出典は画像の下。候補の枠は焼き込まない（detections の GeoJSON から重ねる）。表示の版 s1fc-1 を入れてから処理した撮影のみ（それより前の撮影は VV の PNG のみ。再取得はしない）",
            "color/<地点>/v<版>/<撮影時刻>.thumb.webp": "その縮小版（320×320 に収める、凡例なし）",
            "images/<地点>/[v<版>/]<撮影時刻>.png": "切り出しの VV 画像（灰）。赤枠＝港内水域の候補、黄枠＝岸壁帯の候補、水色＝陸マスクの縁、橙＝桟橋・橋などの構造物、紫＝区画 A/B の境界。v2 以降は版ごとの下位フォルダ",
            "status.json": "処理したシーンと結果、最後の実行",
            "usage.csv": "実行ごとの Sentinel Hub リクエスト数と処理単位（PU）",
            "CHANGES.md": "定義変更の記録",
            "quay_counts.csv": "岸壁の占有数（quay-v1、試行・精度未測定）。1 行＝1 撮影×1 地点×1 範囲版。撮影と同じ相対軌道の基準がある撮影だけ",
            "quay/YYYY-MM.geojson": "撮影月ごとの岸壁の占有区間（線、水際から 15 m 沖に描く。長さ・数の幅・横抱きの可能性つき）",
            "change/<地点>/v<版>/<撮影時刻>.webp": "変化カラー（R＝今回の VV、G・B＝同じ相対軌道の基準の VV）の WebP と縮小版 .thumb.webp。基準がある撮影だけ",
        },
        "columns": {
            "site_id": "地点 ID",
            "acquired_utc": "撮影時刻（UTC、同じ通過の最初のスライス）",
            "retrieved_utc": "取得時刻（UTC）",
            "platform": "衛星",
            "orbit_direction": "軌道の向き（ascending / descending）",
            "relative_orbit": "相対軌道（不明は空）",
            "incidence_deg": "入射角（地点の水域の中央値、度）",
            "harbour_candidates": "港内水域（岸壁帯を除く切り出し範囲の水域）の候補数",
            "quay_candidates": "岸壁帯（陸マスクから 50 m 以内）の候補数",
            "confirmed": "確認数（人が画像で確かめた数。未記入は空）",
            "mask_version": "陸マスクの版（＝監視範囲の版）。版が違う行は範囲が違い、直接比べない",
            "detector_version": "検出設定の版",
            "processing_version": "処理の版",
            "coverage": "地点の水域のうちデータがあった割合",
            "len_lt50 / len_50_100 / len_100_200 / len_ge200": "候補（両区域）の長さの区分ごとの数（〜50 / 50〜100 / 100〜200 / 200 m〜）",
            "processing_units": "その撮影の取得に使った処理単位（PU）",
            "image": "切り出し画像（VV の灰、枠つきの PNG）",
            "color_image": "疑似カラーの WebP（凡例つき、枠なし）。作っていない行は空",
            "color_thumb": "その縮小版の WebP。作っていない行は空",
            "sector_a_harbour / sector_a_quay": "区画 A（地点の sectors の1つ目。湛江は軍港区画）の港内水域・岸壁帯の候補数（重心で判定）。区画のない版の行は空",
            "sector_b_harbour / sector_b_quay": "区画 B（切り出しの残り。湛江は水道）の港内水域・岸壁帯の候補数。A と B の合計は harbour_candidates / quay_candidates と同じ",
        },
        "display": fcolor.settings(),
        "quay": {
            "status": "試行（精度未測定）。正解データとの突き合わせをしていない",
            "method": quay.settings(),
            "baselines": baseline_meta(sites),
            "change_image": fcolor.change_settings(),
            "credibility": "基準が 24 シーン以上なら 3、未満なら 4（一段下げる）",
            "columns": {
                "occupied_min / occupied_max": "岸壁の占有数（最小〜最大）。最小＝占有区間のまとまり（60 m 以内の区間は同じ船の可能性があり、最大の 1 つだけ）の数＋断面の谷で切れた数。最大＝全区間の Σ max(片の数, floor(区間長／60 m))＋横抱きの可能性",
                "sector_a_min / sector_a_max / sector_b_min / sector_b_max": "区画 A（軍港区画）・B（水道）ごとの最小〜最大（区間の中央の位置で判定。まとまりの最小はその代表の区画に数える）",
                "segments": "占有区間の数", "rafted": "横抱きの可能性がある区間の数",
                "quay_coverage": "線の水側の点のうち、今回と基準の両方にデータがあった割合",
                "baseline_version / baseline_scenes / credibility": "使った基準（ro<軌道>-bl-<版>）・そのシーン数・確度",
                "quay_version / lines_version": "数え方の版・OSM の線の版",
                "change_image / change_thumb": "変化カラー画像と縮小版",
            },
            "can_say": "岸壁・桟橋に横付けされた物の有無・おおよその数の幅・占有の長さ（±20 m 程度）。同じ相対軌道どうしの撮影の比較",
            "cannot_say": "艦種・国籍・所属。民間船・作業船・荷役機械・建設中の構造物も数に入る。基準の期間ずっと居た船（常時占有）は数えられない",
            "limits": [
                "試行で精度は測っていない。数は幅（最小〜最大）で、最大は区間長を 60 m で割った上限なので大きめに出る",
                "基準は 10 パーセンタイルで、基準期間の 1 割を超えて同じ場所に居た船は「船のいない岸壁」に含まれてしまい数えられない（常時占有は .json に一覧）",
                "風・雨で水面や陸の σ0 が上がった日は、岸に沿って弱い増分が出る。種（ΔVV 12 dB 以上）と今回の VV −10 dB 以上の条件で抑えている",
                "明るい船の横の反対側や桟橋の角に同じ船が出ることがある（60 m 以内のまとまりは最小では 1 つに数える）",
                "OSM の線の位置は 10〜20 m ずれることがあり、水際は陸マスクで探す。OSM に無い岸壁は数えない",
                "相対軌道 84 の基準は 15 シーン（24 未満）で確度 4",
            ],
        },
        "detector": {"version": DETECTOR_VERSION, **DETECT_DEFAULTS,
                     "length_classes_m": [[n, lo, None if hi == math.inf else hi] for n, lo, hi in LENGTH_CLASSES]},
        "can_say": "目標の有無・数・位置・長さの目安（±20〜30 m）",
        "cannot_say": "艦種・国籍・所属。民間船・作業船・ブイ・養殖施設なども候補に入る",
        "limits": [
            "「候補数」「確認数」であり在泊艦数ではない",
            "岸壁に横付けされた艦は陸と区別しにくく数え落とす前提（岸壁帯は別に数える。横付けの船は基準との変化で quay_counts.csv に別に数える：quay）",
            "画素 10 m・実分解能約 20 m。接して並ぶ艦は 1 つにまとまり、大型艦は分かれることがある",
            "陸マスクは OSM の海岸線から作った固定マスクで、埋立て・新しい岸壁は反映が遅れる",
            "固定物（ブイ・係留施設等）のマスクはまだ無く、固定物も候補に数える",
            "撮影は地点ごとに数日おき。軌道の向き・入射角で見え方が変わるため、比較は同じ相対軌道どうしで行う",
            "湛江の範囲 v2 は 2部の確定案（2026-10-30）。北端 21.265N で麻斜の桟橋群が切れていないかは次の撮影の画像で確かめる。確かめるまで公開画面の地図には出さない",
            "区画は候補の重心で決める。境界をまたぐ艦は重心のある側に数える",
        ],
    }


class QuayContext:
    """A site's quay lines, their sample geometry and its per-orbit baselines (loaded on demand)."""

    def __init__(self, site: dict, grid: Grid, mask: np.ndarray, lines_dir: Path = MASK_DIR,
                 baseline_dir: Path = fbase.BASELINE_DIR):
        self.site, self.grid, self.mask, self.baseline_dir = site, grid, mask, baseline_dir
        self.lines_version = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
        self.lines = quay_lines.load(site["id"], self.lines_version, lines_dir)
        self._geom = None
        self._bl: dict = {}

    def baseline(self, orbit):
        if self.lines is None:
            return None
        key = str(orbit)
        if key not in self._bl:
            bl = fbase.load(self.site, orbit, self.baseline_dir)
            if bl is not None and (bl[1].get("width"), bl[1].get("height")) != (self.grid.width, self.grid.height):
                bl = None  # a baseline of another extent: never used
            self._bl[key] = bl
        return self._bl[key]

    def geometry(self) -> quay.Geometry:
        if self._geom is None:
            self._geom = quay.Geometry(self.grid, self.mask, self.lines)
        return self._geom


def quay_features(site: dict, acq: dict, q: dict, credibility: str, bver: str) -> list[dict]:
    feats = []
    segs = sorted(q["segments"], key=lambda s: (-s["lat"], s["lon"]))
    for i, sg in enumerate(segs):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": sg["line"]},
            "properties": {
                "site_id": site["id"], "acquired_utc": acq["acquired_utc"], "n": i + 1,
                "relative_orbit": acq["relative_orbit"], "sector": sg["sector"],
                "line_kind": sg["kind"], "osm_way": sg["way_id"],
                "length_m": sg["length_m"], "est_length_m": sg["est_length_m"],
                "min": sg["min"], "max": sg["max"], "rafted": sg["rafted"], "peak_dvv_db": sg["peak_dvv_db"],
                "cluster": sg.get("cluster"), "cluster_lead": sg.get("cluster_lead"),
                "status": "trial", "reliability": SOURCE_RELIABILITY, "credibility": credibility,
                "mask_version": mask_tag(site), "quay_version": quay.QUAY_VERSION, "baseline_version": bver,
            },
        })
    return feats


def chip_features(site: dict, grid: Grid, acq: dict, dets: list[dict], image: str) -> list[dict]:
    feats = []
    for i, d in enumerate(sorted(dets, key=lambda d: (d["row"], d["col"]))):
        lon, lat = d.get("lon"), d.get("lat")
        if lon is None:
            lon, lat = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(lon), 6), round(float(lat), 6)]},
            "properties": {
                "site_id": site["id"], "acquired_utc": acq["acquired_utc"], "n": i + 1,
                "platform": acq["platform"], "orbit_direction": acq["orbit_direction"],
                "relative_orbit": acq["relative_orbit"],
                "zone": d["zone"], "sector": d.get("sector"), "length_m": d["length_m"], "width_m": d["width_m"],
                "length_class": d["length_class"], "pixels": d["pixels"],
                "peak_vv_db": d["peak_vv_db"], "contrast_vv_db": d["contrast_vv_db"], "vh_ratio_db": d["vh_ratio_db"],
                "status": "candidate", "reliability": SOURCE_RELIABILITY, "credibility": CANDIDATE_CREDIBILITY,
                "mask_version": mask_tag(site), "detector_version": DETECTOR_VERSION,
                "image": image,
            },
        })
    return feats


def count_quay(store: Store, qctx: QuayContext, site: dict, acq: dict, vv: np.ndarray, vh: np.ndarray,
               sectors: list[dict], entry: dict, log=print) -> str:
    """Quay count of one scene against the baseline of its orbit; never stops the run."""
    if qctx.lines is None:
        entry["quay_result"] = "no_lines"
        return ""
    bl = qctx.baseline(acq["relative_orbit"])
    if bl is None:
        entry["quay_result"] = "no_baseline"
        return f"、岸壁の数：軌道 {acq['relative_orbit']} の基準なし"
    layers, bmeta = bl
    try:
        q = quay.count(qctx.geometry(), vv, vh, layers, lambda lo, la: sector_of(sectors, lo, la))
        bver = f"ro{bmeta['relative_orbit']}-{bmeta['version']}"
        cred = fbase.credibility(int(bmeta["scenes"]), int(fbase.site_config(site)["min_scenes"]))
        row = {
            "site_id": site["id"], "acquired_utc": acq["acquired_utc"], "platform": acq["platform"],
            "orbit_direction": acq["orbit_direction"], "relative_orbit": acq["relative_orbit"],
            "mask_version": mask_tag(site), "quay_version": quay.QUAY_VERSION,
            "lines_version": f"{site['id']}-quay-v{qctx.lines_version}", "baseline_version": bver,
            "baseline_scenes": bmeta["scenes"], "credibility": cred,
            "occupied_min": q["min"], "occupied_max": q["max"],
            "segments": len(q["segments"]), "rafted": q["rafted"], "quay_coverage": q["coverage"],
        }
        for col, sc in zip(SECTOR_COLUMNS, sectors):
            mn, mx = q["by_sector"].get(sc["id"], [0, 0])
            row[f"sector_{col}_min"], row[f"sector_{col}_max"] = mn, mx
        extra = {}
        try:
            cpath, tpath = change_paths(site, acq["acquired_utc"])
            full, thumb = fcolor.render_change(vv, layers["vv_p10"], color_label(site, acq) + f"\nbase {bver} (VV p10, n={bmeta['scenes']})",
                                               f"Contains modified Copernicus Sentinel data {acq['acquired_utc'][:4]}")
            extra = {cpath: full, tpath: thumb}
            row.update(change_image=cpath, change_thumb=tpath)
        except Exception as e:
            log(f"{site['id']} {acq['acquired_utc']}: 変化画像を作れなかった {str(e)[:200]}")
        store.add_quay(row, quay_features(site, acq, q, cred, bver), extra)
    except Exception as e:
        entry.update(quay_result="error", quay_error=str(e)[:300])
        log(f"{site['id']} {acq['acquired_utc']}: 岸壁の数で失敗 {str(e)[:200]}")
        return "、岸壁の数：失敗"
    entry["quay_result"] = "counted"
    by = "".join(f" {sc['id']} {row.get(f'sector_{c}_min')}〜{row.get(f'sector_{c}_max')}" for c, sc in zip(SECTOR_COLUMNS, sectors))
    return f"、岸壁の占有 {q['min']}〜{q['max']}（{len(q['segments'])} 区間{by}、基準 {bver} n={bmeta['scenes']}）"


def process_site(client, store: Store, site: dict, mask: np.ndarray, now: datetime, days: int | None,
                 budget_left: float, max_scenes: int, log=print, probe: dict | None = None,
                 raw_dir: Path | None = None, lines_dir: Path = MASK_DIR,
                 baseline_dir: Path = fbase.BASELINE_DIR) -> dict:
    grid = Grid.for_site(site)
    tag = mask_tag(site)
    sectors = site_sectors(site)
    if mask.shape != (grid.height, grid.width):
        raise ValueError(f"{site['id']}: 陸マスク {tag} の大きさ {mask.shape} が範囲 {(grid.height, grid.width)} と合わない")
    qctx = QuayContext(site, grid, mask, lines_dir, baseline_dir)
    if days is None:
        days = LOOKBACK_DAYS if store.has_rows(site["id"], tag) else FIRST_LOOKBACK_DAYS
        # the first run with a baseline also gives the recent scenes a quay count (fetched again once)
        if not store.quay_done(site["id"], tag) and any(qctx.baseline(o) for o in fbase.site_config(site)["orbits"]):
            days = FIRST_LOOKBACK_DAYS
    days = min(days, MAX_LOOKBACK_DAYS)
    feats = client.catalog(grid.bbox, now - timedelta(days=days), now)
    if probe is not None and feats:
        probe["catalog_property_keys"] = sorted(feats[0].get("properties", {}).keys())
    acqs = cdse.acquisitions(feats)
    done = store.done(site["id"], tag)
    qdone = store.quay_done(site["id"], tag)
    # Also a counted scene without a quay count, once a baseline of its orbit exists (re-fetched once).
    new = [a for a in acqs if a["acquired_utc"] not in done
           or (a["acquired_utc"] not in qdone and qctx.baseline(a["relative_orbit"]) is not None)]
    log(f"{site['id']}: カタログ {len(feats)} 件 → 撮影 {len(acqs)} 回（過去 {days} 日）、新着 {len(new)}")
    summary = {"site_id": site["id"], "lookback_days": days, "catalog_items": len(feats),
               "acquisitions": len(acqs), "new": len(new), "counted": 0, "skipped": 0, "errors": 0}
    land = mask > 0
    water_px = int((~land).sum())
    with_lia = True
    for acq in new[:max_scenes]:
        est = cdse.estimate_pu(grid.width, grid.height, 3)
        if budget_left < est:
            log(f"{site['id']}: 今月の処理単位の上限に近いため残りを次回に回す")
            summary["deferred_budget"] = True
            break
        entry = {"site_id": site["id"], "acquired_utc": acq["acquired_utc"], "platform": acq["platform"],
                 "orbit_direction": acq["orbit_direction"], "relative_orbit": acq["relative_orbit"],
                 "mask_version": tag, "catalog_ids": acq["ids"]}
        retrieved = cdse.iso(datetime.now(timezone.utc))
        try:
            try:
                data, pu = client.process(grid.bbox, grid.width, grid.height, acq["t_from"], acq["t_to"], with_lia)
            except cdse.CredentialError:
                raise
            except cdse.CDSEError as e:
                if with_lia and "localIncidenceAngle" in str(e):
                    log("localIncidenceAngle が使えないため入射角なしで取り直す")
                    with_lia = False
                    data, pu = client.process(grid.bbox, grid.width, grid.height, acq["t_from"], acq["t_to"], False)
                else:
                    raise
            budget_left -= pu
            if raw_dir is not None:  # probe only: keep the chip to re-render without fetching again
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{site['id']}-v{mask_version(site)}-{stamp(acq['acquired_utc'])}.png").write_bytes(data)
            vv, vh, lia = decode_chip(data)
        except cdse.CredentialError:
            raise
        except Exception as e:  # one bad scene does not stop the run; it is retried next time
            entry.update(result="error", error=str(e)[:300], retrieved_utc=retrieved)
            store.note_scene(entry)
            summary["errors"] += 1
            log(f"{site['id']} {acq['acquired_utc']}: 失敗 {str(e)[:200]}")
            continue
        valid_water = np.isfinite(vv) & ~land
        coverage = valid_water.sum() / water_px if water_px else 0.0
        entry.update(retrieved_utc=retrieved, processing_units=pu, coverage=round(float(coverage), 3))
        if coverage < MIN_COVERAGE:
            entry["result"] = "skipped_low_coverage"
            store.note_scene(entry)
            summary["skipped"] += 1
            log(f"{site['id']} {acq['acquired_utc']}: 水域の {coverage:.0%} しか写っていないため数えない")
            continue
        try:  # a detection/rendering failure is recorded like a fetch failure (PU already spent)
            dets, info = detect(vv, vh, land, grid.pixel_m)
            for d in dets:
                lon, lat = grid.to_lonlat(d["row"] + 0.5, d["col"] + 0.5)
                d["lon"], d["lat"] = float(lon), float(lat)
                d["sector"] = sector_of(sectors, d["lon"], d["lat"])
            image = image_path(site, acq["acquired_utc"])
            inc = None
            if lia is not None and np.isfinite(lia[valid_water]).any():
                inc = round(float(np.nanmedian(lia[valid_water])), 2)
            nh = sum(d["zone"] == "harbour" for d in dets)
            nq = sum(d["zone"] == "quay" for d in dets)
            row = {
                "site_id": site["id"], "acquired_utc": acq["acquired_utc"], "retrieved_utc": retrieved,
                "platform": acq["platform"], "orbit_direction": acq["orbit_direction"],
                "relative_orbit": acq["relative_orbit"], "incidence_deg": inc,
                "harbour_candidates": nh, "quay_candidates": nq, "confirmed": "",
                "mask_version": tag, "detector_version": DETECTOR_VERSION,
                "processing_version": PROCESSING_VERSION, "coverage": round(float(coverage), 3),
                "processing_units": pu, "image": image,
            }
            for col, sc in zip(SECTOR_COLUMNS, sectors):
                for zone in ("harbour", "quay"):
                    row[f"sector_{col}_{zone}"] = sum(d["zone"] == zone and d["sector"] == sc["id"] for d in dets)
            for name, _lo, _hi in LENGTH_CLASSES:
                row[f"len_{name}"] = sum(d["length_class"] == name for d in dets)
            quick = render.quicklook(vv, mask, dets, sector_raster(grid, sectors))
            features = chip_features(site, grid, acq, dets, image)
        except Exception as e:
            entry.update(result="error", error=f"検出・描画で失敗: {str(e)[:280]}")
            store.note_scene(entry)
            summary["errors"] += 1
            log(f"{site['id']} {acq['acquired_utc']}: 検出・描画で失敗 {str(e)[:200]}")
            continue
        extra = {}
        try:  # the display image never stops the count (the PNG quick-look is the record)
            cpath, tpath = color_paths(site, acq["acquired_utc"])
            full, thumb = fcolor.render(vv, vh, color_label(site, acq),
                                        f"Contains modified Copernicus Sentinel data {acq['acquired_utc'][:4]}")
            extra = {cpath: full, tpath: thumb}
            row.update(color_image=cpath, color_thumb=tpath)
        except Exception as e:
            log(f"{site['id']} {acq['acquired_utc']}: カラー画像を作れなかった {str(e)[:200]}")
        store.add(row, features, quick, extra)
        entry.update(result="counted", detector_info=info)
        qtext = count_quay(store, qctx, site, acq, vv, vh, sectors, entry, log)
        store.note_scene(entry)
        summary["counted"] += 1
        by_sector = "".join(f"、{sc['id']} {row[f'sector_{c}_harbour']}+{row[f'sector_{c}_quay']}"
                            for c, sc in zip(SECTOR_COLUMNS, sectors))
        log(f"{site['id']} {tag} {acq['acquired_utc']} {acq['platform']} {acq['orbit_direction']} 軌道{acq['relative_orbit']}"
            f" 入射角{inc}: 港内候補 {nh}、岸壁帯候補 {nq}{by_sector}（{pu} PU、水域の {coverage:.0%}）{qtext}")
    if len(new) > max_scenes:
        summary["deferred"] = len(new) - max_scenes
    summary["budget_left_pu"] = round(budget_left, 2)
    return summary


def main(argv=None, client=None, sites=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel-1 harbour ship-candidate counts (trial)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    ap.add_argument("--probe", type=Path, help="write everything under this directory only; build missing masks from OSM")
    ap.add_argument("--days", type=int, help=f"look back this many days (max {MAX_LOOKBACK_DAYS})")
    ap.add_argument("--max-scenes", type=int, default=MAX_SCENES_PER_RUN)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    probe = None
    mask_dir = args.mask_dir
    published_usage: list[dict] = []
    if args.probe:
        # The probe writes its own usage.csv under the probe directory, but the same
        # CDSE account is used: count this month's published usage (args.out, the
        # repository data) against the cap too, so a probe cannot push the month over it.
        published_usage = read_csv(args.out / "usage.csv")
        args.out = args.probe / "out"
        probe = {}
    all_sites = sites if sites is not None else load_sites()
    store = Store(args.out, all_sites)
    month = now.strftime("%Y-%m")
    used_pu, used_req = month_usage(store.usage + published_usage, month)
    budget = MONTHLY_PU_CAP - used_pu
    run = {"run_utc": cdse.iso(now), "sites": []}
    if used_req >= MONTHLY_REQUEST_CAP:
        print(f"今月のリクエスト数が上限（{MONTHLY_REQUEST_CAP}）に達したため取得しない", file=sys.stderr)
        budget = 0
    client = client if client is not None else cdse.Client()
    rc = 0
    try:
        for site in all_sites:
            if not site.get("enabled"):
                continue
            try:
                mask, mmeta = load_mask(site["id"], mask_version(site), mask_dir)
            except FileNotFoundError:
                if probe is None:
                    raise
                # Probe only: build the missing mask from OSM into the probe directory
                # (to be checked and committed by hand).
                grid = Grid.for_site(site)
                osm = fetch_osm(grid)
                mask, info = build_mask(grid, osm)
                mmeta = save_mask(site, grid, mask, info, osm, mask_version(site), args.probe / "masks")
                print("陸マスクを OSM から作成:", json.dumps({k: mmeta[k] for k in mmeta if k != "query"}, ensure_ascii=False))
            if budget <= 0:
                run["sites"].append({"site_id": site["id"], "deferred_budget": True})
                continue
            try:
                s = process_site(client, store, site, mask, now, args.days, budget, args.max_scenes, probe=probe,
                                 raw_dir=args.probe / "raw" if probe is not None else None)
            except cdse.CredentialError:
                raise
            except cdse.CDSEError as e:  # e.g. catalog 429/5xx: record it, go on to the next site
                print(f"{site['id']}: 失敗 {str(e)[:200]}", file=sys.stderr)
                s = {"site_id": site["id"], "error": str(e)[:300]}
                rc = 1
            # Budget from what the client actually spent (also covers a site that failed midway).
            budget = MONTHLY_PU_CAP - used_pu - client.processing_units
            s["budget_left_pu"] = round(budget, 2)
            run["sites"].append(s)
    except cdse.CredentialError as e:
        print(f"エラー: {e}", file=sys.stderr)
        run["error"] = str(e)
        rc = 2
    except BaseException as e:  # unexpected: still record the PU spent (finally), then fail the run
        run["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        run.update(requests=client.requests, processing_units=round(client.processing_units, 3),
                   pu_source="header" if client.pu_reported else "estimate",
                   scenes=sum(s.get("counted", 0) + s.get("skipped", 0) for s in run["sites"]),
                   month_pu_before=round(used_pu, 3), month_pu_cap=MONTHLY_PU_CAP)
        store.save(run)
    if probe is not None:
        probe.update(run)
        probe["last_process_header_names"] = sorted(getattr(client, "last_headers", []))
        print(json.dumps(probe, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(run, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
