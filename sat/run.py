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
from . import cdse, png, quay, quay_est, quay_est2, quay_lines, render, s2
from . import sectors as fsectors
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
FIRST_RUNS_PER_RUN = 2     # sites whose first 30-day backfill may start in one run (do not add many ports at once)
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

USAGE_FIELDS = ["run_utc", "requests", "processing_units", "pu_source", "scenes",
                "s2_requests", "s2_processing_units"]  # the Sentinel-2 part of the run (s2nc-1), included in the totals

# Quay count (quay-v1, trial): one row per scene x site x mask version, only where a baseline
# of the scene's relative orbit exists. Separate from ship_counts.csv (the CFAR count stays as it is).
QUAY_FIELDS = [
    "site_id", "acquired_utc", "platform", "orbit_direction", "relative_orbit", "mask_version",
    "quay_version", "lines_version", "baseline_version", "baseline_scenes", "credibility",
    "occupied_min", "occupied_max", "sector_a_min", "sector_a_max", "sector_b_min", "sector_b_max",
    "segments", "rafted", "quay_coverage", "change_image", "change_thumb", "in_sample",
    # est-1 (trial): the number as a distribution (sat/quay_est.py); occupied_min/max stay as they are
    "est_version", "est_mode", "est_p10", "est_p90", "est_mean",
    "est_warship_mode", "est_warship_p10", "est_warship_p90",
    "est_a_mode", "est_a_p10", "est_a_p90", "est_b_mode", "est_b_p10", "est_b_p90", "est_conflicts", "est_capped", "est_prior_hash",
    # feat-1 (record only): the radar's look azimuth and the orbit pair (same ships seen from two orbits)
    "features_version", "look_az_deg", "pair_id",
]
# Every sector of every site, one row per scene x site x mask version x sector (multi-site PR-1).
SECTOR_FIELDS = [
    "site_id", "acquired_utc", "relative_orbit", "mask_version", "sector_id", "sector_name", "role",
    "occupied_min", "occupied_max", "est_version", "est_mode", "est_p10", "est_p90",
    "est2_version", "est2_median", "est2_p10", "est2_p90", "est2_calib_source", "source",
]
PAIR_MAX_H = 36.0  # feat-1: scenes of two relative orbits within this many hours form a pair

# Scenes the quay-v1 thresholds were tuned on (CHANGES.md, 2026-10-30): their counts are
# in-sample values, and they are also in the baselines. Accuracy is judged on later scenes.
# Per site in sites.json (quay.tuning_scenes); this dict only adds to it (tests).
QUAY_TUNING_SCENES: dict[str, set[str]] = {}
OSM_CREDIT = "© OpenStreetMap contributors"
OSM_LICENSE = "ODbL 1.0"


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
    """[{id, name, role, bbox|None, polygon?}]: the listed sectors (any number, bbox or polygon), then
    the rest of the chip (sat/sectors.py). No sectors: []."""
    return fsectors.site_sectors(site)


def sector_of(sectors: list[dict], lon: float, lat: float) -> str | None:
    return fsectors.sector_of(sectors, lon, lat)


def sector_raster(grid: Grid, sectors: list[dict]) -> np.ndarray | None:
    """Index of the sector of each pixel centre (0 = the first listed one), or None without sectors."""
    return fsectors.sector_raster(grid, sectors)


def wide_sectors(sectors: list[dict], site: dict | None = None) -> list[dict]:
    """The sectors of the old wide columns (sector_a_* = the listed sector, sector_b_* = the rest):
    only for a site of the shared layout (Zhanjiang) with one listed sector. Other sites leave them
    empty; every sector of every site is in quay_sectors.csv."""
    if site is not None and layout(site) != "shared":
        return []
    return sectors if len(sectors) == 2 else []


def layout(site: dict) -> str:
    """"shared": detections/YYYY-MM.geojson and quay/YYYY-MM.geojson for all such sites (Zhanjiang, as
    before); "per_site": detections/<site>/YYYY-MM.geojson, quay/<site>/YYYY-MM.geojson (new sites)."""
    return site.get("layout") or "shared"


def detections_rel(site: dict | None, month: str) -> str:
    return f"detections/{site['id']}/{month}.geojson" if site and layout(site) == "per_site" else f"detections/{month}.geojson"


def quay_rel(site: dict | None, month: str) -> str:
    return f"quay/{site['id']}/{month}.geojson" if site and layout(site) == "per_site" else f"quay/{month}.geojson"


def makes_png(site: dict) -> bool:
    """New sites keep no PNG quick-look (the colour WebP is the record, design §5.3)."""
    return site.get("png", True) is not False


def tuning_scenes(site: dict) -> set[str]:
    """Scenes the quay-v1 thresholds were tuned on: sites.json quay.tuning_scenes (+ QUAY_TUNING_SCENES)."""
    return set((site.get("quay") or {}).get("tuning_scenes") or []) | set(QUAY_TUNING_SCENES.get(site["id"], set()))


def runs_daily(site: dict) -> bool:
    """Fetched by the daily run: enabled and not ``daily: false`` (a site with settings only)."""
    return bool(site.get("enabled")) and site.get("daily", True) is not False


def site_month_pu(store: "Store", site_id: str, month: str) -> float:
    """PU a site used this month: its Sentinel-1 scenes (status.json) and Sentinel-2 days (s2_scenes.csv).
    usage.csv stays one row per run (unchanged): the per-site share comes from the scene records."""
    pu = 0.0
    for s in store.status.get("scenes", []):
        if s.get("site_id") == site_id and str(s.get("retrieved_utc") or "").startswith(month):
            pu += float(s.get("processing_units") or 0)
    for r in s2.read_rows(store.out / "s2_scenes.csv"):
        if r.get("site_id") == site_id and str(r.get("retrieved_utc") or "").startswith(month):
            pu += float(r.get("processing_units") or 0)
    return pu


def site_budget(store: "Store", site: dict, month: str, budget: float) -> float:
    """The month's budget left for a site: the run's budget, and its own pu_month_cap if it has one."""
    cap = site.get("pu_month_cap")
    if cap is None:
        return budget
    return min(budget, float(cap) - site_month_pu(store, site["id"], month))


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
        self.zone_priors: dict[str, quay_est.ZonePrior] = {}
        self.by_id = {s["id"]: s for s in self.sites}
        self.sector_rows = read_csv(out / "quay_sectors.csv")
        self.new_sector_rows = False

    def site(self, site_id: str) -> dict | None:
        return self.by_id.get(site_id)

    def zone_prior_path(self, site_id: str) -> Path:
        return self.out / "quay" / f"{site_id}-zone-prior.json"

    def add_sectors(self, rows: list[dict]) -> None:
        """quay_sectors.csv: one row per scene x site x mask version x sector (every sector of every site)."""
        if not rows:
            return
        key = lambda r: (r["site_id"], r["acquired_utc"], row_mask_tag(r))
        keys = {key(r) for r in rows}
        self.sector_rows = [r for r in self.sector_rows if key(r) not in keys] + rows
        self.new_sector_rows = True

    def zone_prior(self, site_id: str, model: quay_est.Model) -> quay_est.ZonePrior:
        if site_id not in self.zone_priors:
            self.zone_priors[site_id] = quay_est.ZonePrior.load(model, site_id, self.zone_prior_path(site_id))
        return self.zone_priors[site_id]

    # Rows and scenes are kept per mask version: a new version (a new site extent) starts
    # its own series and re-reads recent scenes; rows of the old version stay as they were.
    def done(self, site_id: str, tag: str) -> set[str]:
        keys = {r["acquired_utc"] for r in self.rows if r["site_id"] == site_id and row_mask_tag(r) == tag}
        keys |= {s["acquired_utc"] for s in self.status.get("scenes", [])
                 if s["site_id"] == site_id and row_mask_tag(s) == tag and s.get("result") != "error"}
        return keys

    def quay_done(self, site_id: str, tag: str) -> set[str]:
        """Scenes with a quay row, or whose quay count was tried or cannot be made: an error and a
        scene skipped for low coverage are not fetched again."""
        keys = {r["acquired_utc"] for r in self.quay_rows if r["site_id"] == site_id and row_mask_tag(r) == tag}
        keys |= {s["acquired_utc"] for s in self.status.get("scenes", [])
                 if s["site_id"] == site_id and row_mask_tag(s) == tag and s.get("quay_result") in ("counted", "error", "skipped")}
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
        self.new_quay.setdefault(quay_rel(self.site(row["site_id"]), row["acquired_utc"][:7]), []).append((row, features))

    def has_rows(self, site_id: str, tag: str) -> bool:
        return any(r["site_id"] == site_id and row_mask_tag(r) == tag for r in self.rows)

    def note_scene(self, entry: dict) -> None:
        key = (entry["site_id"], entry["acquired_utc"], row_mask_tag(entry))
        sc = [s for s in self.status.get("scenes", []) if (s["site_id"], s["acquired_utc"], row_mask_tag(s)) != key]
        sc.append(entry)
        sc.sort(key=lambda s: (s["acquired_utc"], s["site_id"], row_mask_tag(s)))
        self.status["scenes"] = sc

    def add(self, row: dict, features: list[dict], image: bytes | None, extra: dict[str, bytes] | None = None) -> None:
        key = (row["site_id"], row["acquired_utc"], row_mask_tag(row))
        self.rows = [r for r in self.rows if (r["site_id"], r["acquired_utc"], row_mask_tag(r)) != key]
        self.rows.append(row)
        self.rows.sort(key=lambda r: (r["acquired_utc"], r["site_id"], row_mask_tag(r)))
        if image is not None and row.get("image"):  # new sites keep no PNG (site png: false)
            p = self.out / row["image"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(image)
        for rel, data in (extra or {}).items():
            q = self.out / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
        self.new_features.setdefault(detections_rel(self.site(row["site_id"]), row["acquired_utc"][:7]), []).append((row, features))

    def save(self, run: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        write_csv(self.out / "ship_counts.csv", self.rows, CSV_FIELDS)
        for rel, items in self.new_features.items():
            p = self.out / rel
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
        pairs = orbit_pairs(self.quay_rows)
        qkey = lambda r: (r["site_id"], r["acquired_utc"], row_mask_tag(r))
        months = dict(self.new_quay)
        for r in self.quay_rows:
            pid = pairs.get(qkey(r), "")
            if (r.get("pair_id") or "") != pid:
                r["pair_id"] = pid
                # rewrite the partner's month with its pair id
                months.setdefault(quay_rel(self.site(r["site_id"]), r["acquired_utc"][:7]), [])
        if self.quay_rows or self.new_quay:
            write_csv(self.out / "quay_counts.csv", self.quay_rows, QUAY_FIELDS)
        for rel, items in months.items():
            p = self.out / rel
            month = rel.rsplit("/", 1)[-1][:7]
            if not items and not p.exists():
                continue
            fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"type": "FeatureCollection", "features": []}
            fkey = lambda p: (p["site_id"], p["acquired_utc"], row_mask_tag(p))
            keys = {fkey(r) for r, _ in items}
            feats = [f for f in fc["features"] if fkey(f["properties"]) not in keys]
            for _r, fs in items:
                feats.extend(fs)
            feats.sort(key=lambda f: (f["properties"]["acquired_utc"], f["properties"]["site_id"],
                                      row_mask_tag(f["properties"]), f["properties"]["n"]))
            for f in feats:
                fp = f["properties"]
                if "pair_id" in fp or pairs.get(qkey(fp)):
                    fp["pair_id"] = pairs.get(qkey(fp)) or None
            fc = {"type": "FeatureCollection", **quay_fc_attribution(month[:4]), "features": feats}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        for sid, zp in self.zone_priors.items():
            zp.save(self.zone_prior_path(sid))
        est2 = None
        if self.quay_rows:
            try:  # est-2 (trial): recomputed from the stored files only (no fetch); never stops the run
                est2 = quay_est2.build(self.out, self.quay_rows, self.rows, pairs, sites=self.sites)
            except Exception as e:  # noqa: BLE001
                print(f"est-2: {type(e).__name__}: {e}", file=sys.stderr)
        self.save_sectors(est2)
        # quay/blind.geojson: the shared-layout sites (Zhanjiang) as before; a per_site site in quay/<site>/blind.geojson
        groups: dict[str, list] = {}
        for b in blind_summary(self.sites):
            site = self.site(b["site_id"])
            rel = f"quay/{b['site_id']}/blind.geojson" if site and layout(site) == "per_site" else "quay/blind.geojson"
            groups.setdefault(rel, []).extend(b["features"])
        for rel, blind_feats in groups.items():
            if not blind_feats:
                continue
            p = self.out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            fc = {"type": "FeatureCollection", **quay_fc_attribution(None), "features": blind_feats}
            p.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        if run.get("requests"):
            self.usage.append({k: run.get(k) for k in USAGE_FIELDS})
        write_csv(self.out / "usage.csv", self.usage, USAGE_FIELDS)
        self.status["last_run"] = run
        self.status["updated_utc"] = run["run_utc"]
        self.status["series_breaks"] = series_breaks(self.sites)
        (self.out / "status.json").write_text(json.dumps(self.status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (self.out / "meta.json").write_text(json.dumps(build_meta(self.rows, self.sites, self.out), ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")


    def save_sectors(self, est2: dict | None) -> None:
        """quay_sectors.csv (new, every sector of every site): the rows of this run's scenes, the old
        wide columns (sector_a_* / sector_b_*) of earlier scenes of a site with one listed sector, and the
        est-2 values per sector. Written only once there is a row; never stops the run."""
        try:
            have = {(r["site_id"], r["acquired_utc"], row_mask_tag(r)) for r in self.sector_rows}
            back = []
            for r in self.quay_rows:
                if (r["site_id"], r["acquired_utc"], row_mask_tag(r)) in have:
                    continue
                site = self.site(r["site_id"])
                secs = wide_sectors(site_sectors(site), site) if site else []
                for col, sc in zip(SECTOR_COLUMNS, secs):
                    if r.get(f"sector_{col}_min") in (None, ""):
                        continue
                    back.append({"site_id": r["site_id"], "acquired_utc": r["acquired_utc"],
                                 "relative_orbit": r.get("relative_orbit"), "mask_version": row_mask_tag(r),
                                 "sector_id": sc["id"], "sector_name": sc.get("name"), "role": sc["role"],
                                 "occupied_min": r.get(f"sector_{col}_min"), "occupied_max": r.get(f"sector_{col}_max"),
                                 "est_version": r.get("est_version"), "est_mode": r.get(f"est_{col}_mode"),
                                 "est_p10": r.get(f"est_{col}_p10"), "est_p90": r.get(f"est_{col}_p90"),
                                 "source": "quay_counts（横持ちの列から）"})
            if back:
                self.sector_rows += back
                self.new_sector_rows = True
            if est2:
                by = {(x["site_id"], x["acquired_utc"]): x for x in est2.get("scenes", [])}
                for r in self.sector_rows:
                    e = by.get((r["site_id"], r["acquired_utc"]))
                    st = (e or {}).get("by_sector", {}).get(r["sector_id"]) if e else None
                    if st:
                        new = {"est2_version": quay_est2.EST2_VERSION, "est2_median": st["warship"]["median"],
                               "est2_p10": st["warship"]["p10"], "est2_p90": st["warship"]["p90"],
                               "est2_calib_source": e.get("calib_source", "")}
                        if any(str(r.get(k, "")) != str(v) for k, v in new.items()):
                            r.update(new)
                            self.new_sector_rows = True
            if self.new_sector_rows:
                self.sector_rows.sort(key=lambda r: (r["acquired_utc"], r["site_id"], row_mask_tag(r), r["sector_id"]))
                write_csv(self.out / "quay_sectors.csv", self.sector_rows, SECTOR_FIELDS)
        except Exception as e:  # noqa: BLE001
            print(f"quay_sectors.csv: {type(e).__name__}: {e}", file=sys.stderr)


def attribution(rows: list[dict]) -> str:
    years = sorted({r["acquired_utc"][:4] for r in rows})
    if not years:
        years = [str(datetime.now(timezone.utc).year)]
    span = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return f"Contains modified Copernicus Sentinel data {span}"


def s2_attribution(out: Path | None = None) -> str:
    """Credit of the Sentinel-2 images: the years of the clear days on record."""
    rows = [r for r in s2.read_rows((out or OUT_DIR) / "s2_scenes.csv") if r.get("image")]
    return attribution(rows) if rows else ""  # no image yet: no credit


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


def quay_fc_attribution(year: str | None) -> dict:
    """attribution / license of a quay GeoJSON: its lines follow the OSM quay lines (ODbL)."""
    cop = f"; contains modified Copernicus Sentinel data {year}" if year else ""
    return {"attribution": f"岸壁の線：{OSM_CREDIT}{cop}", "license": f"{OSM_LICENSE}（{OSM_CREDIT}。線の幾何は OpenStreetMap に由来）"}


_BLIND_CACHE: dict = {}
_GEOM_CACHE: dict = {}
GEOM_CACHE_MAX = 4


def blind_summary(sites: list[dict]) -> list[dict]:
    """Per site and orbit: where the quay count is likely blind (baseline VV p10 > -8 dB in the inner band).
    Fixed by the baseline, the lines and the mask; computed once per process."""
    out = []
    for site in sites:
        # only sites fetched daily: a site with settings only (daily false) is left out (its geometry costs
        # 10-40 s per run for nothing; turn daily on and the next run writes it)
        if not runs_daily(site):
            continue
        orbits = fbase.site_config(site)["orbits"]
        lv = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
        if not orbits or quay_lines.load(site["id"], lv) is None:
            continue
        for o in orbits:
            bl = fbase.load(site, o)
            if bl is None:
                continue
            key = (site["id"], mask_tag(site), lv, o, bl[1]["version"], bl[1].get("built_utc"))
            if key not in _BLIND_CACHE:
                try:
                    grid = Grid.for_site(site)
                    mask, _info = load_mask(site["id"], mask_version(site))
                    qctx = QuayContext(site, grid, mask)
                    layers = qctx.baseline(o)
                    if layers is None:
                        continue
                    r = quay.blind_stretches(qctx.geometry(), layers[0])
                except Exception as e:  # the summary never stops a run
                    print(f"{site['id']} 軌道 {o}: 数えられない区間を計算できなかった {str(e)[:200]}", file=sys.stderr)
                    continue
                bver = f"ro{o}-{bl[1]['version']}"
                geom = lambda ln: ({"type": "LineString", "coordinates": ln} if len(ln) >= 2
                                   else {"type": "Point", "coordinates": ln[0]})  # a single 10 m sample
                feats = [{"type": "Feature", "geometry": geom(st["line"]),
                          "properties": {"site_id": site["id"], "mask_version": mask_tag(site), "relative_orbit": o,
                                         "baseline_version": bver, "line_kind": st["kind"], "osm_way": st["way_id"],
                                         "length_m": st["length_m"], "status": "likely_blind"}}
                         for st in r["stretches"]]
                _BLIND_CACHE[key] = {
                    "site_id": site["id"], "mask_version": mask_tag(site), "relative_orbit": o, "baseline_version": bver,
                    "lines_version": f"{site['id']}-quay-v{lv}", "total_m": r["total_m"], "blind_m": r["blind_m"],
                    "share": r["share"], "strict_m": r["strict_m"], "stretches": len(r["stretches"]), "features": feats,
                }
            out.append(_BLIND_CACHE[key])
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
        wide = wide_sectors(sec, site)
        m["sectors"] = [({"column": f"sector_{SECTOR_COLUMNS[i]}"} if wide else {}) | sc for i, sc in enumerate(sec)]
    m.update(site_status(site))
    return m


def safe_site_meta(site: dict) -> dict:
    """site_meta, or a short error entry: one site's bad settings never stop meta.json."""
    try:
        return site_meta(site)
    except Exception as e:  # noqa: BLE001
        print(f"{site.get('id')}: meta.json の地点の項を作れなかった {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return {"id": site.get("id"), "name": site.get("name"), "error": f"{type(e).__name__}: {str(e)[:300]}",
                "daily": runs_daily(site), "publish": site.get("publish", "trial")}


def validate_site(site: dict) -> None:
    """Raises ValueError on settings the run cannot use: sectors (ids, roles, overlap, outside the site)."""
    if not site.get("id"):
        raise ValueError("地点の id が無い")
    fsectors.site_sectors(site)
    ov = fsectors.overlaps(site)
    if ov:
        raise ValueError(f"{site['id']}: 区画が重なっている {ov}")
    out = fsectors.outside(site)
    if out:
        raise ValueError(f"{site['id']}: 範囲の外の区画 {out}")
    # A site fetched daily must have its own monthly cap (1部 review of PR #35): the shared CDSE free tier keeps
    # Zhanjiang's reserve. Only a site written explicitly as layout "shared" (Zhanjiang; one only, usable_sites)
    # uses the run's cap as before; a site without "layout" is not exempt.
    if runs_daily(site) and site.get("layout") != "shared":
        cap = site.get("pu_month_cap")
        if (cap is None or isinstance(cap, bool) or not isinstance(cap, (int, float))
                or not math.isfinite(cap) or cap < 0):
            raise ValueError(f"{site['id']}: daily＝true の地点には pu_month_cap（PU／月、0 以上の有限の数）が要る")


def check_shared(sites: list[dict]) -> None:
    """At most one enabled site written as layout "shared" (Zhanjiang): the shared files and the exemption
    from pu_month_cap are for that one only."""
    shared = [s.get("id") for s in sites if s.get("enabled") and s.get("layout") == "shared"]
    if len(shared) > 1:
        raise ValueError(f"layout＝shared の地点は 1 つだけ（{'・'.join(map(str, shared))}）")


def usable_sites(sites: list[dict], log=lambda m: print(m, file=sys.stderr)) -> tuple[list[dict], list[dict]]:
    """(sites to use, errors). A site of its own folders (layout per_site: a new trial port) with bad
    settings is left out of this run and recorded; a shared-layout site (Zhanjiang) stops the run as before."""
    check_shared(sites)  # two shared sites would write into the same files: stop the run
    ok, errors = [], []
    for s in sites:
        if not s.get("enabled"):
            ok.append(s)
            continue
        try:
            validate_site(s)
        except Exception as e:  # noqa: BLE001
            if layout(s) == "shared":
                raise
            errors.append({"site_id": s.get("id"), "error": f"{type(e).__name__}: {str(e)[:300]}"})
            log(f"{s.get('id')}: 設定の誤りのため、この地点だけ飛ばす {str(e)[:200]}")
            continue
        ok.append(s)
    return ok, errors


def site_status(site: dict) -> dict:
    """Multi-site settings of a site for meta.json (sites[]): kind, priority, daily, publish, where its
    files are, what the quay count still lacks, the est-1 weights and the est-2 calibration it uses."""
    qc = site.get("quay") or {}
    lv = int(site.get("quay_lines_version", quay_lines.LINES_VERSION))
    orbits = fbase.site_config(site)["orbits"]
    have_bl = [o for o in orbits if fbase.load(site, o) is not None]
    lacks = []
    if quay_lines.load(site["id"], lv) is None:
        lacks.append(f"岸壁線 {site['id']}-quay-v{lv}（OSM、probe で取得）")
    if not have_bl:
        lacks.append("基準（" + "・".join(f"軌道 {o}" for o in orbits) + "）")
    ver = quay_est.classes_version()
    out = {
        "kind": site.get("kind", "naval"), "group": site.get("group"), "priority": int(site.get("priority", 1)),
        "daily": runs_daily(site), "publish": site.get("publish", "trial"),
        "mode": "毎日の取得" if runs_daily(site) else "設定のみ（毎日の取得なし）",
        "layout": layout(site), "png": makes_png(site), "pu_month_cap": site.get("pu_month_cap"),
        "quay": {
            "ready": not lacks, "lacks": lacks, "baseline_orbits": orbits, "baselines_present": have_bl,
            "lines_version": f"{site['id']}-quay-v{lv}",
            "class_weights": quay_est.site_weights_info(site["id"]), "classes_version": ver,
            "est2_calib_source": quay_est2.calib_source(site["id"]),
            "label_freeze_utc": qc.get("label_freeze_utc"),
        },
    }
    if site.get("sources"):
        out["sources"] = site["sources"]
    return out


def build_meta(rows: list[dict], sites: list[dict] | None = None, out: Path | None = None) -> dict:
    sites = sites if sites is not None else load_sites()
    return {
        "dataset": "sat",
        "title": "Sentinel-1 による港内の艦船候補数（試行：" + "・".join(
            f"{s.get('name') or s['id']}" for s in sites if runs_daily(s)) + "）",
        "status": ("試行（陸マスクのみ。固定物マスクは 20 シーン蓄積後）。湛江（ZJG）は 2026-10-30 に監視範囲を v2（東岸の軍港を中心とする矩形）"
                   "に変えた。ZJG-land-v1 の行と ZJG-land-v2 の行は範囲が違うため直接比べない（series_breaks）"),
        "sites": [safe_site_meta(s) for s in sites if s.get("enabled")],
        "sites_note": ("複数港の土台（multi-site PR-1）。publish＝trial の地点は試行ページだけで扱い、公開の地図には出さない。"
                       "daily＝false の地点は設定だけで毎日の取得をしない（岸壁線・基準がそろってから有効にする）。"
                       "kind・priority・pu_month_cap・区画の role は sat/sites.json。湛江（ZJG）のファイルの場所と列は従来どおり"),
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
            "sentinel2": (s2_attribution(out) + "（s2/ の自然色。画像の下の帯にも描く）") if s2_attribution(out) else "",
            "land_mask": "陸マスク：(c) OpenStreetMap contributors, ODbL 1.0",
            "quay_lines": "岸壁の数に使う桟橋・岸壁・海岸線の線（sat/masks/<地点>-quay-v<版>.json）と、それに沿う quay/*.geojson の線：(c) OpenStreetMap contributors, ODbL 1.0",
            "processing": "observatory-data が σ0（地形補正済み、10 m）の切り出しから自動で数えた。数値は ESA・欧州委員会が作成したものではない。",
        },
        "files": {
            "ship_counts.csv": "1 行＝1 撮影×1 地点×1 範囲版（mask_version）。区画の列 sector_a_*・sector_b_* は区画が 1 つ＋残りの地点（湛江）だけ。他の地点は空（区画ごとの値は quay_sectors.csv）",
            "detections/<地点>/YYYY-MM.geojson・quay/<地点>/YYYY-MM.geojson": "湛江以外の地点（layout＝per_site）の候補と岸壁の占有区間。湛江は従来の detections/YYYY-MM.geojson・quay/YYYY-MM.geojson",
            "quay_sectors.csv": ("区画ごとの岸壁の数（縦持ち）：1 行＝1 撮影×1 地点×1 範囲版×1 区画。role（naval＝軍港の桟橋・岸壁、other＝水道・商港 など）、"
                                 "quay-v1 の最小〜最大、est-1 の最頻値・80% 区間、est-2 の中央値・80% 区間と較正の出どころ（est2_calib_source）。"
                                 "source＝quay_counts（横持ちの列から）の行は、このファイルを作る前の撮影を quay_counts.csv の区画 A・B の列から写したもの"),
            "quay_est2/<地点>.csv・quay_est2/<地点>.json": "湛江以外の地点の est-2（列は quay_est2.csv と同じ、区画 A・B の列は空）。湛江は従来の quay_est2.csv・quay_est2.json",
            "detections/YYYY-MM.geojson": "撮影月ごとの候補（点、長さの目安つき）",
            "color/<地点>/v<版>/<撮影時刻>.webp": "疑似カラー（R＝VV、G＝VH、B＝VV−VH を逆向き）の WebP。範囲は全地点・全日で固定、表示用だけ 3×3 Lee フィルタ、凡例と出典は画像の下。候補の枠は焼き込まない（detections の GeoJSON から重ねる）。表示の版 s1fc-1 を入れてから処理した撮影のみ（それより前の撮影は VV の PNG のみ。再取得はしない）",
            "color/<地点>/v<版>/<撮影時刻>.thumb.webp": "その縮小版（320×320 に収める、凡例なし）",
            "images/<地点>/[v<版>/]<撮影時刻>.png": "切り出しの VV 画像（灰）。赤枠＝港内水域の候補、黄枠＝岸壁帯の候補、水色＝陸マスクの縁、橙＝桟橋・橋などの構造物、紫＝区画 A/B の境界。v2 以降は版ごとの下位フォルダ",
            "status.json": "処理したシーンと結果、最後の実行",
            "usage.csv": "実行ごとの Sentinel Hub リクエスト数と処理単位（PU）。s2_requests・s2_processing_units＝そのうち Sentinel-2（s2nc-1）の分",
            "s2_scenes.csv": "Sentinel-2 L2A の撮影日ごとの雲の判定（1 行＝1 日×1 地点×1 範囲版。result・雲の割合・PU）。clear の日だけ画像あり（s2）",
            "s2/<地点>/v<版>/<撮影時刻>.webp": "Sentinel-2 の自然色（試行、表示だけ）と縮小版 .thumb.webp",
            "CHANGES.md": "定義変更の記録",
            "quay_counts.csv": "岸壁の占有数（quay-v1、試行・精度未測定）。1 行＝1 撮影×1 地点×1 範囲版。撮影と同じ相対軌道の基準がある撮影だけ",
            "quay/YYYY-MM.geojson": "撮影月ごとの岸壁の占有区間（線、水際から 15 m 沖に描く。長さ・数の幅・横抱きの可能性つき）。線は OSM 由来（attribution・license）",
            "quay/<地点>-zone-prior.json": "推定 est-1 の区画ごとの長さ 6 階級のディリクレ（初期値 alpha0 と更新後の alpha、更新に使った撮影）",
            "quay_est2.csv": ("推定 est-2（試行・較正 1 回目、一部は仮置き）：1 行＝1 撮影。軍艦の可能性が高い船の隻数の中央値とモデル内の 80% 区間（地点・区画 A・B）、"
                              "全船の隻数、使えた手がかりの区間数（used_*）、人の目でも判断しにくい区間の数とその期待値（hard_*、注記だけ）、"
                              "並べて見るための est-1 の 60 m 以上の値。取得済みのデータだけから毎回計算し直す"),
            "quay_est2.json": "est-2 の設定（較正した値と仮置き値）と撮影ごと・区間ごとの詳細（列数の事後確率 p_c、軍艦の確率、使った手がかり、hard_to_judge）",
            "quay/blind.geojson": "数えられない可能性が高い区間（軌道ごと。内側の帯で基準の VV p10 が −8 dB を超える所。1 点だけの区間は Point）。湛江以外の地点（layout＝per_site）は quay/<地点>/blind.geojson",
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
        "s2": {**s2.settings(), "credit": s2_attribution(out)},
        "quay": {
            "status": "試行（精度未測定）。正解データとの突き合わせをしていない",
            "method": quay.settings(),
            "estimate": quay_est.settings(),
            "estimate2": quay_est2.settings(),
            "baselines": baseline_meta(sites),
            "change_image": fcolor.change_settings(),
            "credibility": "基準が 24 シーン以上なら 3、未満なら 4（一段下げる）",
            "columns": {
                "occupied_min / occupied_max": "岸壁の占有数（最小〜最大、信頼区間ではない）。最小＝占有区間のまとまり（60 m 以内の区間は同じ船の可能性があり、最大の 1 つだけ）の数＋断面の谷で切れた数。最大＝全区間の Σ max(片の数, floor(区間長／60 m))＋横抱きの可能性",
                "sector_a_min / sector_a_max / sector_b_min / sector_b_max": "区画 A（軍港区画）・B（水道）ごとの最小〜最大（区間の中央の位置で判定。まとまりの最小はその代表の区画に数える）",
                "segments": "占有区間の数", "rafted": "横抱きの可能性がある区間の数",
                "quay_coverage": "線の水側の点のうち、今回と基準の両方にデータがあった割合",
                "baseline_version / baseline_scenes / credibility": "使った基準（ro<軌道>-bl-<版>）・そのシーン数・確度",
                "quay_version / lines_version": "数え方の版・OSM の線の版",
                "change_image / change_thumb": "変化カラー画像と縮小版",
                "in_sample": "tuning＝閾値の調整に使った撮影（標本内の値。基準の期間にも含まれる）、baseline＝基準の期間に含まれる撮影、空＝どちらでもない（精度の評価に使える）",
                "est_version / est_mode / est_p10 / est_p90 / est_mean": ("推定 est-1（試行・精度未測定）：艦種の長さの分布と桟橋別の事前分布から求めた隻数（全船）のモデル上の最頻値・モデル内の 80% 区間（p10〜p90）・平均。"
                                                                          "モデル内の区間は実数が入る確率ではなく、入っていない不確かさは quay.estimate.limits。"
                                                                          "旧来の occupied_min / occupied_max（幅）はそのまま残す。前提（艦種表・重み・桟橋の倍率）はすべて推定を含む"),
                "est_warship_mode / est_warship_p10 / est_warship_p90": "同じく、全長 60 m 以上の船だけの隻数（艦種表の長さによる区別で、艦艇以外の船を含み得る）",
                "est_a_* / est_b_*": "区画 A・B ごとの推定（区間の中央の位置で判定）",
                "est_conflicts": "前提と合わない印（prior_conflict：valleys・length・cap）が付いた区間の数",
                "est_capped": "モデル内の 80% 区間の上端が上限 16 隻に達した区間の数（1 以上なら合計はもっと多い可能性）",
                "features_version / look_az_deg / pair_id": "feat-1（記録だけ）：レーダーが地上で見る方位と、同じ岸壁を2つの軌道で見る組の ID（36 時間以内。その間に入出港があれば船は違う。quay.method.features）",
                "est_prior_hash": "推定に使った区画の事前分布（quay/<地点>-zone-prior.json の alpha）のハッシュ。history に撮影ごとの alpha が残る",
            },
            "can_say": "岸壁・桟橋に横付けされた物の有無・おおよその数の幅・占有の長さの目安（未検証）。同じ相対軌道どうしの撮影の比較",
            "cannot_say": ("艦種・国籍・所属。民間船・作業船・荷役機械・建設中の構造物も数に入る。基準期間の約 9 割以上同じ場所に居た船"
                           "（または入れ替わりで常に埋まっていた岸壁）は数えられない（blind_stretches）"),
            "tuning": {
                "scenes": {s["id"]: sorted(tuning_scenes(s)) for s in sites if tuning_scenes(s)},
                "note": ("閾値（ΔVV 12 dB の種、今回の VV −10 dB、60 m のまとまり、横抱きの 40〜60 m など）は湛江 v2 のこの 6 撮影で調整した。"
                         "これらの撮影の数は調整に使った標本内の値で、基準の期間（bl-1）にも含まれる（quay_counts.csv の in_sample＝tuning）。"
                         "精度の評価は 2026-10-27 以降の撮影（基準・調整のどちらにも使っていないもの）と光学画像で行う"),
            },
            "blind_stretches": [{k: v for k, v in b.items() if k != "features"} for b in blind_summary(sites)],
            "blind_note": ("内側の帯（水際から 0〜30 m）の最大で基準の VV p10 が −8 dB を超える区間は、基準に船（または構造物）が入っていて"
                           "数えられない可能性が高い。total_m＝水側のある線の全長、blind_m＝その区間の長さの合計、share＝割合（判定は帯の 3 画素の最大なので、岸壁自体のにじみも拾う多めの値）。strict_m＝帯の 3 画素すべてが −8 dB を超える長さ（少なめの値）。線は quay/blind.geojson"),
            "limits": [
                "試行で精度は測っていない。数は幅（最小〜最大）だが信頼区間ではない：誤検出（荷役機械・作業船・建設中の構造物）と見落とし（常時占有・OSM に無い岸壁）があり、実数はこの幅の外にもあり得る",
                ("最小は 60 m 以内の区間を 1 つにまとめる。湛江の桟橋は両側の線の間隔の中央値が 60〜80 m で、両側に並んだ 2 隻が 1 隻になりやすく、"
                 "最小は実質「占有された桟橋のまとまりの数」に近い"),
                "最大は区間長を 60 m（最短の艦の想定）で割った上限で、長い岸壁では大きく出る",
                ("基準は 10 パーセンタイルなので、基準期間の約 9 割以上同じ場所に居た船（または入れ替わりで常に埋まっていた岸壁）は"
                 "「船のいない岸壁」に含まれてしまい数えられない。基準の .json の always_occupied_on_water は陸・構造物から 20 m 以内を除くため、"
                 "横付けの帯（0〜30 m）の常時占有はほぼ載らない。代わりに内側の帯で基準の p10 が −8 dB を超える区間を blind_stretches・quay/blind.geojson に出す"),
                ("閾値は湛江 v2 の 6 撮影（2026-10-02・10-08・10-14・10-26 の軌道 157、10-03・10-15 の軌道 84）で調整した。"
                 "これらの撮影の数は標本内の値で、基準の期間にも含まれる（in_sample＝tuning）。精度の評価は 2026-10-27 以降の撮影と光学画像で行う"),
                "風・雨で水面や陸の σ0 が上がった日は、岸に沿って弱い増分が出る。種（ΔVV 12 dB 以上）と今回の VV −10 dB 以上の条件で抑えている",
                "明るい船の横の反対側や桟橋の角に同じ船が出ることがある（60 m 以内のまとまりは最小では 1 つに数える）",
                "横抱きは外側の帯も占有された「可能性」で、確かめていない。長さの目安（区間長−20 m）も未検証",
                "OSM の線の位置は 10〜20 m ずれることがあり、水際は陸マスクで探す。OSM に無い岸壁は数えない",
                "相対軌道 84 の基準は 15 シーン（24 未満）で確度 4",
            ] + ["推定 est-1：" + x for x in quay_est.LIMITS] + ["推定 est-2：" + x for x in quay_est2.LIMITS],
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
        self._model = None

    def model(self) -> quay_est.Model:
        if self._model is None:
            self._model = quay_est.Model(site_id=self.site["id"],
                                         sector_roles=fsectors.sector_roles(site_sectors(self.site)))
        return self._model

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
        """The sample geometry of the lines; shared by every context of the same site, mask and lines in
        one process (it takes ~20 s for Zhanjiang and was built once per context: count + 2 orbits' blind)."""
        if self._geom is None:
            import hashlib
            key = (self.site["id"], mask_tag(self.site), self.lines_version, tuple(self.grid.bbox), self.grid.width, self.grid.height,
                   hashlib.sha256(np.ascontiguousarray(self.mask).tobytes()).hexdigest(),
                   hashlib.sha256(json.dumps(self.lines, sort_keys=True).encode()).hexdigest())
            if key not in _GEOM_CACHE:
                while len(_GEOM_CACHE) >= GEOM_CACHE_MAX:  # a geometry holds large arrays: keep a few sites
                    _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
                _GEOM_CACHE[key] = quay.Geometry(self.grid, self.mask, self.lines)
            self._geom = _GEOM_CACHE[key]
        return self._geom


def orbit_pairs(rows: list[dict], max_h: float = PAIR_MAX_H) -> dict:
    """feat-1: pair ids of quay rows. In time order, an unpaired scene is paired with the next unpaired
    scene of another relative orbit (same site and mask version) within ``max_h`` hours: the same quays
    seen from two geometries (e.g. 157 and the next day's 84 at Zhanjiang; ships may arrive or leave in
    between). Re-made over all rows on every save (past rows get ids too); a scene that comes later in
    time never breaks an existing pair (a scene inserted earlier in time may).
    Returns {(site_id, acquired_utc, mask tag): "<site>-<first scene's time>-ro<a>+ro<b>"}."""
    def t(r):
        return datetime.strptime(r["acquired_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    out: dict = {}
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["site_id"], row_mask_tag(r)), []).append(r)
    for (sid, tag), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["acquired_utc"])
        used = set()
        for i, r in enumerate(rs):
            if i in used:
                continue
            for j in range(i + 1, len(rs)):
                s = rs[j]
                if (t(s) - t(r)).total_seconds() > max_h * 3600:
                    break
                if j in used or str(s.get("relative_orbit")) == str(r.get("relative_orbit")):
                    continue
                pid = f"{sid}-{stamp(r['acquired_utc'])}-ro{r.get('relative_orbit')}+ro{s.get('relative_orbit')}"
                out[(sid, r["acquired_utc"], tag)] = out[(sid, s["acquired_utc"], tag)] = pid
                used.update((i, j))
                break
    return out


def in_sample(site: dict, acquired_utc: str, bmeta: dict) -> str:
    """"tuning" (the thresholds were tuned on it), "baseline" (in the baseline period) or ""."""
    if acquired_utc in tuning_scenes(site):
        return "tuning"
    if any(sc.get("acquired_utc") == acquired_utc and sc.get("used", True) for sc in bmeta.get("scene_list") or []):
        return "baseline"
    return ""


def quay_features(site: dict, acq: dict, q: dict, credibility: str, bver: str, sample: str = "") -> list[dict]:
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
                "peak_dvh_db": sg.get("peak_dvh_db"), "outer_len_m": sg.get("outer_len_m"),
                "depth_m": sg.get("depth_m"), "cols_hint": quay_est.cols_hint(sg.get("depth_m")),
                "features_version": quay.FEATURES_VERSION, "depth_1x3_m": sg.get("depth_1x3_m"),
                "db_line_m": sg.get("db_line_m"), "db_line_db": sg.get("db_line_db"), "db_peaks": sg.get("db_peaks"),
                "look_az_deg": sg.get("look_az_deg"), "cos_phi": sg.get("cos_phi"), "face_look": sg.get("face_look"),
                "face_shares": sg.get("face_shares"), "pair_id": None,
                "zone": sg.get("zone"), "face": sg.get("face"),
                "est_version": quay_est.EST_VERSION if "n_mode" in sg else None,
                "n_mode": sg.get("n_mode"), "n_p10": sg.get("n_p10"), "n_p90": sg.get("n_p90"), "n_mean": sg.get("n_mean"),
                "warship_mode": sg.get("warship_mode"), "prior_conflict": sg.get("prior_conflict"),
                "prior_update": sg.get("clean"),
                "status": "trial", "reliability": SOURCE_RELIABILITY, "credibility": credibility,
                "mask_version": mask_tag(site), "quay_version": quay.QUAY_VERSION, "baseline_version": bver,
                "in_sample": sample,
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
        clat = site.get("center_lat")
        if clat is None and site.get("bbox"):
            clat = (float(site["bbox"][1]) + float(site["bbox"][3])) / 2  # the bbox centre; no latitude: no look azimuth
        look = quay.look_azimuth(acq.get("orbit_direction"), float(clat)) if clat is not None else None
        q = quay.count(qctx.geometry(), vv, vh, layers, lambda lo, la: sector_of(sectors, lo, la), look_az_deg=look)
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
            "in_sample": in_sample(site, acq["acquired_utc"], bmeta),
            "features_version": quay.FEATURES_VERSION, "look_az_deg": look,
        }
        for col, sc in zip(SECTOR_COLUMNS, wide_sectors(sectors, site)):
            mn, mx = q["by_sector"].get(sc["id"], [0, 0])
            row[f"sector_{col}_min"], row[f"sector_{col}_max"] = mn, mx
        est = None
        try:  # est-1 never stops the quay-v1 count
            zp = store.zone_prior(site["id"], qctx.model())
            tuning = row["in_sample"] == "tuning"  # the thresholds were tuned on it: never informs the prior
            est = quay_est.estimate(q["segments"], site["id"], qctx.model(), zp, tuple(sc["id"] for sc in sectors),
                                    update_prior=not tuning)
            row.update(est_version=quay_est.EST_VERSION, est_mode=est["all"]["mode"], est_p10=est["all"]["p10"],
                       est_p90=est["all"]["p90"], est_mean=est["all"]["mean"],
                       est_warship_mode=est["warship"]["mode"], est_warship_p10=est["warship"]["p10"],
                       est_warship_p90=est["warship"]["p90"], est_conflicts=est["conflicts"],
                       est_capped=est["capped"], est_prior_hash=est["prior_hash"])
            for col, sc in zip(SECTOR_COLUMNS, wide_sectors(sectors, site)):
                st = est["by_sector"].get(sc["id"])
                if st:
                    row[f"est_{col}_mode"], row[f"est_{col}_p10"], row[f"est_{col}_p90"] = st["mode"], st["p10"], st["p90"]
            zp.record(acq["acquired_utc"], bool(est["soft"]) and not tuning, "tuning（更新に使わない）" if tuning else "")
            if not tuning:
                zp.update(acq["acquired_utc"], est["soft"])  # after the estimate: a scene never informs its own prior
        except Exception as e:
            est = None
            log(f"{site['id']} {acq['acquired_utc']}: 推定 est-1 で失敗 {str(e)[:200]}")
        extra = {}
        try:
            cpath, tpath = change_paths(site, acq["acquired_utc"])
            full, thumb = fcolor.render_change(vv, layers["vv_p10"], color_label(site, acq) + f"\nbase {bver} (VV p10, n={bmeta['scenes']})",
                                               f"Contains modified Copernicus Sentinel data {acq['acquired_utc'][:4]}")
            extra = {cpath: full, tpath: thumb}
            row.update(change_image=cpath, change_thumb=tpath)
        except Exception as e:
            log(f"{site['id']} {acq['acquired_utc']}: 変化画像を作れなかった {str(e)[:200]}")
        store.add_quay(row, quay_features(site, acq, q, cred, bver, row["in_sample"]), extra)
        srows = []
        for sc in sectors:
            mn, mx = q["by_sector"].get(sc["id"], [0, 0])
            st = (est or {}).get("by_sector", {}).get(sc["id"]) if est else None
            srows.append({"site_id": site["id"], "acquired_utc": acq["acquired_utc"], "relative_orbit": acq["relative_orbit"],
                          "mask_version": mask_tag(site), "sector_id": sc["id"], "sector_name": sc.get("name"),
                          "role": sc["role"], "occupied_min": mn, "occupied_max": mx,
                          "est_version": quay_est.EST_VERSION if st else "", "est_mode": st["mode"] if st else "",
                          "est_p10": st["p10"] if st else "", "est_p90": st["p90"] if st else "", "source": "quay-v1"})
        store.add_sectors(srows)
    except Exception as e:
        entry.update(quay_result="error", quay_error=str(e)[:300])
        log(f"{site['id']} {acq['acquired_utc']}: 岸壁の数で失敗 {str(e)[:200]}")
        return "、岸壁の数：失敗"
    entry["quay_result"] = "counted"
    by = "".join(f" {sc['id']} {q['by_sector'].get(sc['id'], [0, 0])[0]}〜{q['by_sector'].get(sc['id'], [0, 0])[1]}" for sc in sectors)
    et = ""
    if est is not None:
        et = (f"、推定 est-1 最頻 {est['all']['mode']}（80% {est['all']['p10']}〜{est['all']['p90']}、平均 {est['all']['mean']}、"
              f"60 m 以上 {est['warship']['mode']}）" + "".join(f" {k} {v['mode']}（{v['p10']}〜{v['p90']}）" for k, v in est["by_sector"].items()))
    return f"、岸壁の占有 {q['min']}〜{q['max']}（{len(q['segments'])} 区間{by}、基準 {bver} n={bmeta['scenes']}）{et}"


def process_site(client, store: Store, site: dict, mask: np.ndarray, now: datetime, days: int | None,
                 budget_left: float, max_scenes: int, log=print, probe: dict | None = None,
                 raw_dir: Path | None = None, lines_dir: Path = MASK_DIR,
                 baseline_dir: Path = fbase.BASELINE_DIR, only: str | None = None) -> dict:
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
    if only:
        new = [a for a in new if a["acquired_utc"].startswith(only)]
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
            entry["quay_result"] = "skipped"  # never re-fetched for the quay count either
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
            image = image_path(site, acq["acquired_utc"]) if makes_png(site) else ""
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
            for col, sc in zip(SECTOR_COLUMNS, wide_sectors(sectors, site)):
                for zone in ("harbour", "quay"):
                    row[f"sector_{col}_{zone}"] = sum(d["zone"] == zone and d["sector"] == sc["id"] for d in dets)
            for name, _lo, _hi in LENGTH_CLASSES:
                row[f"len_{name}"] = sum(d["length_class"] == name for d in dets)
            quick = render.quicklook(vv, mask, dets, sector_raster(grid, sectors)) if makes_png(site) else None
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
        by_sector = "".join(f"、{sc['id']} {sum(d['zone'] == 'harbour' and d['sector'] == sc['id'] for d in dets)}"
                            f"+{sum(d['zone'] == 'quay' and d['sector'] == sc['id'] for d in dets)}" for sc in sectors)
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
    ap.add_argument("--only", help="probe only: process only the scenes whose acquired_utc starts with this")
    args = ap.parse_args(argv)
    if args.only and not args.probe:
        ap.error("--only is for --probe")
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
    all_sites, site_errors = usable_sites(all_sites)
    store = Store(args.out, all_sites)
    month = now.strftime("%Y-%m")
    used_pu, used_req = month_usage(store.usage + published_usage, month)
    budget = MONTHLY_PU_CAP - used_pu
    run = {"run_utc": cdse.iso(now), "sites": []}
    if site_errors:  # recorded in status.json (last_run); absent when there is none (Zhanjiang's status as before)
        run["site_errors"] = site_errors
    if used_req >= MONTHLY_REQUEST_CAP:
        print(f"今月のリクエスト数が上限（{MONTHLY_REQUEST_CAP}）に達したため取得しない", file=sys.stderr)
        budget = 0
    client = client if client is not None else cdse.Client()
    rc = 0
    # priority order (1 first; the order of sites.json within a priority); sites with daily: false are not fetched
    daily = sorted((s for s in all_sites if runs_daily(s)), key=lambda s: int(s.get("priority", 1)))
    first_runs = 0
    try:
        for site in daily:
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
            sbudget = site_budget(store, site, month, budget)
            if sbudget <= 0:  # the site's own monthly cap (pu_month_cap)
                run["sites"].append({"site_id": site["id"], "deferred_site_cap": True})
                continue
            if not store.has_rows(site["id"], mask_tag(site)) and args.days is None:
                # the first 30-day backfill: at most FIRST_RUNS_PER_RUN sites in one run
                first_runs += 1
                if first_runs > FIRST_RUNS_PER_RUN:
                    run["sites"].append({"site_id": site["id"], "deferred_first_run": True})
                    continue
            try:
                s = process_site(client, store, site, mask, now, args.days, sbudget, args.max_scenes, probe=probe,
                                 raw_dir=args.probe / "raw" if probe is not None else None, only=args.only)
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
        # Sentinel-2 natural colour on clear days (s2nc-1, display only): new days only, after Sentinel-1.
        # Not in a probe (--probe): the Sentinel-1 probe must not spend PU on Sentinel-2 (its backfill has a probe of its own).
        if probe is None:
            req0, pu0 = client.requests, client.processing_units
            for site in daily:
                if not (site.get("s2") or {}).get("enabled"):
                    continue
                sbudget = site_budget(store, site, month, budget)
                if sbudget < s2.EST_PRE_PU + s2.EST_FULL_PU:
                    run.setdefault("s2", []).append({"site_id": site["id"], "deferred_budget": True})
                    continue
                try:
                    r2 = s2.update_site(client, args.out, site, now, s2.LOOKBACK_DAYS, sbudget,
                                        log=lambda m: print(m, file=sys.stderr))
                except cdse.CredentialError:
                    raise
                except Exception as e:  # never stops the Sentinel-1 record
                    print(f"{site['id']}: Sentinel-2 で失敗 {str(e)[:200]}", file=sys.stderr)
                    r2 = {"site_id": site["id"], "error": str(e)[:300]}
                budget = MONTHLY_PU_CAP - used_pu - client.processing_units
                run.setdefault("s2", []).append(r2)
            run["s2_requests"] = client.requests - req0
            run["s2_processing_units"] = round(client.processing_units - pu0, 3)
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
