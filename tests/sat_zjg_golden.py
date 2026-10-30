"""Zhanjiang (ZJG) end-to-end comparison for the multi-site refactor: every output file must stay the same.

The same scenario is run twice on the same machine, in two subprocesses:

* old: the code of origin/main before PR-1 (9e86565). The files PR-1 changes are kept in
  tests/fixtures/sat_main_9e86565/ (``git show 9e86565:sat/<file>``); the others (and the masks,
  baselines and calibration, which PR-1 does not change) are copied from sat/. sat/sectors.py (new) is left out.
* new: the code of this checkout, with every site of sat/sites.json (the new ports have daily false).

Each run copies the published Zhanjiang records (CSV, GeoJSON, zone prior, S2 labels, quay_est2) into a
work folder, runs ``sat.run.main`` with a fake client (two new scenes, orbits 157 and 84, a chip built from
the real baseline with ships along the real quay lines; no Sentinel-2 day) at a frozen time, and returns
{relative path: sha256} of every file it wrote. The two runs are compared byte for byte; no hash is
stored, because the bytes depend on the machine (zlib of the Python build, float reductions of the CPU)
while the comparison of the two codes on one machine does not.

    python -m tests.sat_zjg_golden            # prints the files that differ (none expected)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "data" / "sat"
OLD_FILES = ROOT / "tests" / "fixtures" / "sat_main_9e86565"
NEW_ONLY = ("sectors.py",)          # modules that did not exist before PR-1
NEW_FILES = {"quay_sectors.csv"}    # outputs that did not exist before PR-1
COPY = ["ship_counts.csv", "quay_counts.csv", "status.json", "usage.csv", "s2_scenes.csv",
        "quay/2026-10.geojson", "quay/ZJG-zone-prior.json", "quay/blind.geojson", "detections/2026-10.geojson",
        "labels/ZJG-s2-labels.csv", "labels/ZJG-s2-s1-match.csv", "labels/ZJG-s2-sections.geojson", "labels/meta.json",
        "quay_est2.csv", "quay_est2.json"]

# Runs in the subprocess (``import sat`` resolves to the package under test).
SCRIPT = r'''
import contextlib, hashlib, io, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import numpy as np
from sat import baseline as fbase, cdse, png, quay, run
from sat.osm_mask import load_mask

src, work = Path(sys.argv[1]), Path(sys.argv[2])
copy = json.loads(sys.argv[3])
NOW = datetime(2026, 11, 3, 1, 0, 0, tzinfo=timezone.utc)

class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)

class FakeClient:
    def __init__(self, chip):
        self.chip, self.requests, self.processing_units, self.pu_reported = chip, 0, 0.0, True
    def catalog(self, bbox, start, end, collection=None):
        self.requests += 1
        if collection is not None:  # Sentinel-2: no day
            return []
        p = {"platform": "sentinel-1c", "sat:orbit_state": "ascending", "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": "g157", "properties": {**p, "sat:relative_orbit": 157, "datetime": "2026-11-01T10:48:45Z"}},
                {"id": "g84", "properties": {**p, "sat:relative_orbit": 84, "datetime": "2026-11-02T10:41:00Z"}}]
    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 1.226
        return self.chip, 1.226

# The quay sample geometry depends only on (grid, mask, lines) and is not changed by its users: build it
# once per process in both runs (the old code built it three times, ~20 s each).
_G = {}
_Geometry = quay.Geometry
def cached_geometry(grid, mask, lines, *a, **k):
    key = (tuple(grid.bbox), grid.width, grid.height, hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
           json.dumps(lines, sort_keys=True), repr(a), repr(sorted(k.items())))
    if key not in _G:
        _G[key] = _Geometry(grid, mask, lines, *a, **k)
    return _G[key]

site = next(s for s in run.load_sites() if s["id"] == "ZJG")
mask, _ = load_mask(site["id"], int(site.get("mask_version", 1)))
layers, _m = fbase.load(site, 157)
rng = np.random.default_rng(20261103)
vv = np.where(np.isfinite(layers["vv_p10"]), layers["vv_p10"], -20.0) + rng.normal(0, 0.4, mask.shape)
vh = np.where(np.isfinite(layers["vh_p10"]), layers["vh_p10"], -26.0) + rng.normal(0, 0.4, mask.shape)
water, land = mask == 0, mask > 0
near = np.zeros_like(water)
near[1:, :] |= land[:-1, :]; near[:-1, :] |= land[1:, :]; near[:, 1:] |= land[:, :-1]; near[:, :-1] |= land[:, 1:]
edge = np.argwhere(water & near)
for r, c in edge[rng.choice(len(edge), size=40, replace=False)]:  # 40 ships alongside
    vv[max(r - 4, 0):r + 5, max(c - 1, 0):c + 2] = 3.0
    vh[max(r - 4, 0):r + 5, max(c - 1, 0):c + 2] = -6.0
enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
chip = png.encode(np.stack([enc(vv), enc(vh), np.full(vv.shape, 4000.0), np.ones(vv.shape)], -1).astype(np.uint16), depth=16, filters=[1])

out = work / "out"
for rel in copy:
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / rel, out / rel)
with mock.patch.object(run, "datetime", Frozen), mock.patch.object(quay, "Geometry", cached_geometry), \
        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    rc = run.main(["--out", str(out)], client=FakeClient(chip), sites=run.load_sites())
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob("*")) if p.is_file() and p != out / "meta.json"}
m = json.loads((out / "meta.json").read_text(encoding="utf-8"))
z = next(s for s in m["sites"] if s["id"] == "ZJG")
keep = ("id", "name", "mask_version", "bbox", "width_px", "height_px", "area_km2", "provisional", "note", "sectors")
meta = {"site": {k: z.get(k) for k in keep},
        "baselines": [b for b in m["quay"]["baselines"] if b["site_id"] == "ZJG"],
        "blind": [b for b in m["quay"]["blind_stretches"] if b["site_id"] == "ZJG"],
        "tuning": m["quay"]["tuning"]["scenes"].get("ZJG"), "credit": m["attribution"]["credit"]}
scenes = json.loads((out / "status.json").read_text(encoding="utf-8"))["scenes"][-2:]
sectors = []
if (out / "quay_sectors.csv").exists():
    import csv
    with (out / "quay_sectors.csv").open(encoding="utf-8") as f:
        sectors = list(csv.DictReader(f))
print(json.dumps({"rc": rc, "files": files, "meta": meta, "sectors": sectors,
                  "quay_results": [s.get("quay_result") for s in scenes]}, ensure_ascii=False))
'''


def old_package(dest: Path) -> Path:
    """dest/sat = sat/ with the files PR-1 changed put back to 9e86565 and the new modules removed."""
    pkg = dest / "sat"
    shutil.copytree(ROOT / "sat", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    for f in OLD_FILES.iterdir():
        shutil.copyfile(f, pkg / f.name)
    for f in NEW_ONLY:
        (pkg / f).unlink()
    return dest


def start(root: Path, work: Path) -> subprocess.Popen:
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.Popen([sys.executable, "-c", SCRIPT, str(SRC), str(work), json.dumps(COPY)],
                            cwd=str(work), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(p: subprocess.Popen, name: str) -> dict:
    out, err = p.communicate(timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"{name}: exit {p.returncode}\n{err[-3000:]}")
    return json.loads(out.strip().splitlines()[-1])


def compare(tmp: Path) -> tuple[dict, dict]:
    """(old, new) results of the same scenario, run in parallel."""
    po = start(old_package(tmp / "oldroot"), tmp / "old")
    pn = start(ROOT, tmp / "new")
    return finish(po, "old (9e86565)"), finish(pn, "new")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        old, new = compare(Path(d))
    diff = sorted(k for k in set(old["files"]) | set(new["files"]) if old["files"].get(k) != new["files"].get(k))
    print(json.dumps({"files": len(old["files"]), "differ": diff, "quay_results": new["quay_results"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
