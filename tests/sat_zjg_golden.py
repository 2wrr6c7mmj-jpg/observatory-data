"""Zhanjiang (ZJG) end-to-end snapshot for the multi-site refactor: every output file must stay the same.

``snapshot(work)`` copies the published Zhanjiang records (CSV, GeoJSON, zone prior, S2 labels) into
``work``, runs ``sat.run.main`` on sat/sites.json with a fake client (two new scenes,
orbits 157 and 84, a synthetic chip built from the real baselines with ships along the real quay lines;
no Sentinel-2 day) at a frozen time, and returns {relative path: sha256} of every file it wrote.

The golden (tests/fixtures/sat_zjg_golden.json) was made with the code of origin/main before the
multi-site change (PR-1):

    python -m tests.sat_zjg_golden --write
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from sat import baseline as fbase, cdse, png, run
from sat.grid import Grid
from sat.osm_mask import load_mask

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "data" / "sat"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "sat_zjg_golden.json"
NOW = datetime(2026, 11, 3, 1, 0, 0, tzinfo=timezone.utc)
COPY = ["ship_counts.csv", "quay_counts.csv", "status.json", "usage.csv", "s2_scenes.csv",
        "quay/2026-10.geojson", "quay/ZJG-zone-prior.json", "quay/blind.geojson", "detections/2026-10.geojson",
        "labels/ZJG-s2-labels.csv", "labels/ZJG-s2-s1-match.csv", "labels/ZJG-s2-sections.geojson", "labels/meta.json",
        "quay_est2.csv", "quay_est2.json"]
# meta.json is rebuilt from sites.json (new sites, new keys) and is compared separately (zjg_meta)
SKIP = {"meta.json"}


class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


class FakeClient:
    def __init__(self, chip):
        self.chip = chip
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True

    def catalog(self, bbox, start, end, collection=None):
        self.requests += 1
        if collection is not None:  # Sentinel-2: no day
            return []
        p = {"platform": "sentinel-1c", "sat:orbit_state": "ascending",
             "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": "g157", "properties": {**p, "sat:relative_orbit": 157, "datetime": "2026-11-01T10:48:45Z"}},
                {"id": "g84", "properties": {**p, "sat:relative_orbit": 84, "datetime": "2026-11-02T10:41:00Z"}}]

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 1.226
        return self.chip, 1.226


def zjg_site() -> dict:
    return next(s for s in run.load_sites() if s["id"] == "ZJG")


def make_chip(site: dict) -> bytes:
    """VV/VH = the orbit-157 baseline p10 + noise, with bright ships on the water next to land/structures."""
    grid = Grid.for_site(site)
    mask, _ = load_mask(site["id"], int(site.get("mask_version", 1)))
    layers, _m = fbase.load(site, 157)
    rng = np.random.default_rng(20261103)
    vv = np.where(np.isfinite(layers["vv_p10"]), layers["vv_p10"], -20.0) + rng.normal(0, 0.4, mask.shape)
    vh = np.where(np.isfinite(layers["vh_p10"]), layers["vh_p10"], -26.0) + rng.normal(0, 0.4, mask.shape)
    water = mask == 0
    near = np.zeros_like(water)
    land = mask > 0
    near[1:, :] |= land[:-1, :]
    near[:-1, :] |= land[1:, :]
    near[:, 1:] |= land[:, :-1]
    near[:, :-1] |= land[:, 1:]
    edge = np.argwhere(water & near)
    picks = edge[rng.choice(len(edge), size=40, replace=False)]
    for r, c in picks:
        vv[max(r - 4, 0):r + 5, max(c - 1, 0):c + 2] = 3.0
        vh[max(r - 4, 0):r + 5, max(c - 1, 0):c + 2] = -6.0
    enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
    arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 4000.0), np.ones(vv.shape)], -1).astype(np.uint16)
    assert arr.shape[:2] == (grid.height, grid.width)
    return png.encode(arr, depth=16, filters=[1])


def snapshot(work: Path) -> dict:
    out = work / "out"
    for rel in COPY:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, out / rel)
    site = zjg_site()
    cli = FakeClient(make_chip(site))
    run._BLIND_CACHE.clear()
    with mock.patch.object(run, "datetime", Frozen), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # every site of sites.json (on origin/main: Zhanjiang only; later the new ports with daily false too)
        rc = run.main(["--out", str(out)], client=cli, sites=run.load_sites())
    if rc != 0:
        raise RuntimeError(f"run.main returned {rc}")
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file() and str(p.relative_to(out)) not in SKIP}


def zjg_meta(work: Path) -> dict:
    """The Zhanjiang parts of meta.json that must not change."""
    m = json.loads((work / "out" / "meta.json").read_text(encoding="utf-8"))
    site = next(s for s in m["sites"] if s["id"] == "ZJG")
    keep = ("id", "name", "mask_version", "bbox", "width_px", "height_px", "area_km2", "provisional", "note", "sectors")
    return {"site": {k: site.get(k) for k in keep},
            "baselines": [b for b in m["quay"]["baselines"] if b["site_id"] == "ZJG"],
            "blind": [b for b in m["quay"]["blind_stretches"] if b["site_id"] == "ZJG"],
            "tuning": m["quay"]["tuning"]["scenes"].get("ZJG"),
            "credit": m["attribution"]["credit"]}


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        snap = snapshot(Path(d))
        meta = zjg_meta(Path(d))
    if "--write" in sys.argv:
        GOLDEN.write_text(json.dumps({"files": snap, "meta": meta}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {GOLDEN} ({len(snap)} files)")
    else:
        print(json.dumps(snap, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
