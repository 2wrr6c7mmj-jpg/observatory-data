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

    python -m tests.sat_zjg_golden            # prints differ / added / removed; exit 0 only if they are as expected

Same check as tests/test_sat_multisite.py (ZhanjiangUnchangedTest): the files both runs wrote must be
byte for byte the same except BERTH_CHANGED (berth-1, 2026-11-01: an intended change; the rows of the past
scenes in quay_counts.csv and ship_counts.csv must still be the same), the files only the new code writes
must be exactly NEW_FILES (added), and no file of the old code may be missing (removed empty). Otherwise
the exit code is 1.
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
# berth-1 (2026-11-01, the user's instruction: shores where ships never berth are left out of the counts):
# an intended change of Zhanjiang's outputs. Only these files may differ from 9e86565; inside them the rows
# of the past scenes stay the same (never rewritten) except est-2, which is recomputed for every scene.
# win-1 (2026-11-03, the user's decision: candidate windows only on sea-joined water along the berth lines, from
# windows.since_utc of sites.json): an intended change of Zhanjiang's outputs for the scenes from that date. Checked
# with the rule switched on for the scenario's scenes (compare(windows_since=...)): only these files may differ besides
# BERTH_CHANGED, and WINDOW_NEW_FILES is added (the regions left out, for the eye check).
WINDOW_VERSION = "win-1"
# Harbour candidates beyond band_m of the berth lines are not thrown away but counted apart as anchorage
# (anchorage_counts.csv, user's decision 2026-11-03).
WINDOW_CHANGED = {"ship_counts.csv", "detections/2026-11.geojson", "status.json"}
WINDOW_NEW_FILES = {"windows/ZJG.geojson", "anchorage_counts.csv"}
BERTH_CHANGED = {"quay/2026-11.geojson", "quay/ZJG-zone-prior.json", "quay_counts.csv", "quay_est2.csv", "quay_est2.json",
                 "ship_counts.csv", "status.json"}
# The two scenes the fake client serves (Zhanjiang, orbits 157 and 84). The scenario's new rows are picked by
# site and these times, never by position or by month: the copied records grow every day (other sites' scenes
# of 2026-11-01T22Z sort between them since the daily run of 2026-11-02, and Zhanjiang's own daily scenes of
# November will follow), and must not change what is compared.
FAKE_UTC = ("2026-11-01T10:48:45Z", "2026-11-02T10:41:00Z")
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
FAKE = tuple(json.loads(sys.argv[4]))
OVERRIDE = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}   # {site id: {key: value}} on sites.json (new code only)
NAVAL_SINCE = OVERRIDE.pop("_naval_since", None)   # naval-1 switched on from this date (new code only)
if NAVAL_SINCE:
    from sat import berths
    berths.NAVAL_SINCE = NAVAL_SINCE
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
        return [{"id": "g157", "properties": {**p, "sat:relative_orbit": 157, "datetime": FAKE[0]}},
                {"id": "g84", "properties": {**p, "sat:relative_orbit": 84, "datetime": FAKE[1]}}]
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

def only_zjg_daily(sites):
    """Zhanjiang is the only daily site this FakeClient (a Zhanjiang-shaped chip) can serve: leave out every other
    site fetched daily (SYL, SLP from 2026-11-01); sites with daily false stay in, as before."""
    return [s for s in sites if s["id"] == "ZJG" or s.get("daily", True) is False]


out = work / "out"
for rel in copy:
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / rel, out / rel)
with mock.patch.object(run, "datetime", Frozen), mock.patch.object(quay, "Geometry", cached_geometry), \
        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sites = [{**s, **OVERRIDE.get(s["id"], {})} for s in only_zjg_daily(run.load_sites())]
    rc = run.main(["--out", str(out)], client=FakeClient(chip), sites=sites)
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob("*")) if p.is_file() and p != out / "meta.json"}
m = json.loads((out / "meta.json").read_text(encoding="utf-8"))
z = next(s for s in m["sites"] if s["id"] == "ZJG")
keep = ("id", "name", "mask_version", "bbox", "width_px", "height_px", "area_km2", "provisional", "note", "sectors")
meta = {"site": {k: z.get(k) for k in keep},
        "baselines": [b for b in m["quay"]["baselines"] if b["site_id"] == "ZJG"],
        "blind": [b for b in m["quay"]["blind_stretches"] if b["site_id"] == "ZJG"],
        "tuning": m["quay"]["tuning"]["scenes"].get("ZJG"), "credit": m["attribution"]["credit"]}
is_fake = lambda r: r.get("site_id") == "ZJG" and r.get("acquired_utc") in FAKE
is_past = lambda r: r.get("site_id") == "ZJG" and r.get("acquired_utc") not in FAKE
scenes = sorted((s for s in json.loads((out / "status.json").read_text(encoding="utf-8"))["scenes"] if is_fake(s)),
                key=lambda s: s["acquired_utc"])
sectors = []
if (out / "quay_sectors.csv").exists():
    import csv
    with (out / "quay_sectors.csv").open(encoding="utf-8") as f:
        sectors = list(csv.DictReader(f))
import csv
def rows_of(name):
    with (out / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
sha = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
past = {n: sha([r for r in rows_of(n) if is_past(r)]) for n in ("quay_counts.csv", "ship_counts.csv")}
new_quay = [{k: r.get(k) for k in ("acquired_utc", "relative_orbit", "quay_version", "occupied_min", "occupied_max",
                                   "sector_a_min", "sector_a_max", "sector_b_min", "sector_b_max", "est_mode", "est_warship_mode")}
            for r in rows_of("quay_counts.csv") if is_fake(r)]
new_ship = [{k: r.get(k) for k in ("acquired_utc", "harbour_candidates", "quay_candidates", "sector_a_quay", "sector_b_quay")}
            for r in rows_of("ship_counts.csv") if is_fake(r)]
gj = [f for f in json.loads((out / "quay/2026-11.geojson").read_text(encoding="utf-8"))["features"] if is_fake(f["properties"])]
new_lines = sorted({(f["properties"]["line_kind"], f["properties"]["osm_way"]) for f in gj})
e2 = rows_of("quay_est2.csv")
anch = [{k: r[k] for k in ("acquired_utc", "anchorage_candidates", "window_version")} for r in rows_of("anchorage_counts.csv")
        if is_fake(r)] if (out / "anchorage_counts.csv").exists() else []
dets = [f["properties"] for f in json.loads((out / "detections/2026-11.geojson").read_text(encoding="utf-8"))["features"] if is_fake(f["properties"])]
cand = [{"utc": t, "harbour_kept": sum(d["zone"] == "harbour" and not d.get("window_excluded") and d.get("window") != "anchorage"
                                       for d in dets if d["acquired_utc"] == t),
         "anchorage": sum(d.get("window") == "anchorage" for d in dets if d["acquired_utc"] == t),
         "anchorage_zones": sorted({d["zone"] for d in dets if d["acquired_utc"] == t and d.get("window") == "anchorage"}),
         "window_excluded": sorted(d["window_excluded"] for d in dets if d["acquired_utc"] == t and d.get("window_excluded")),
         "all": sum(1 for d in dets if d["acquired_utc"] == t)} for t in FAKE]
class_rows = [{k: r[k] for k in ("acquired_utc", "harbour_naval", "harbour_civil", "harbour_unknown", "quay_naval", "quay_civil",
                                  "quay_unknown", "stretches_naval", "stretches_civil", "stretches_unknown")}
              for r in rows_of("berth_class_counts.csv") if is_fake(r)] if (out / "berth_class_counts.csv").exists() else []
naval_cand = [{"utc": t, "classes": sorted(d.get("berth_class", "") for d in dets if d["acquired_utc"] == t and d.get("berth_class")),
               "excluded": sum(bool(d.get("naval_excluded")) for d in dets if d["acquired_utc"] == t)} for t in FAKE]
print(json.dumps({"rc": rc, "files": files, "class_rows": class_rows, "naval_cand": naval_cand,
                  "naval_excluded": [s.get("naval_excluded") for s in scenes],
                  "quay_naval_excluded": [s.get("quay_naval_excluded") for s in scenes], "meta": meta, "sectors": sectors,
                  "quay_results": [s.get("quay_result") for s in scenes],
                  "excluded": [{k: s.get(k) for k in ("quay_excluded", "quay_band_excluded")} for s in scenes],
                  "window_excluded": [s.get("window_excluded") for s in scenes], "candidates": cand, "anchorage_rows": anch,
                  "past": past, "new_quay": new_quay, "new_ship": new_ship, "new_lines": new_lines,
                  "est2_version": sorted({r["est2_version"] for r in e2}), "est2_rows": len(e2),
                  "est2_warship": [(r["acquired_utc"], r["warship_median"]) for r in e2]}, ensure_ascii=False))
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


def start(root: Path, work: Path, override: dict | None = None) -> subprocess.Popen:
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.Popen([sys.executable, "-c", SCRIPT, str(SRC), str(work), json.dumps(COPY), json.dumps(FAKE_UTC), json.dumps(override or {})],
                            cwd=str(work), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(p: subprocess.Popen, name: str) -> dict:
    out, err = p.communicate(timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"{name}: exit {p.returncode}\n{err[-3000:]}")
    return json.loads(out.strip().splitlines()[-1])


def compare(tmp: Path, windows_since: str | None = None) -> tuple[dict, dict]:
    """(old, new) results of the same scenario, run in parallel. ``windows_since``: the new code runs with Zhanjiang's
    window rule (win-1) active from this date (the scenario's scenes are before the real since_utc)."""
    po = start(old_package(tmp / "oldroot"), tmp / "old")
    ov = None
    if windows_since:
        from sat.run import load_sites
        z = next(s for s in load_sites() if s["id"] == "ZJG")
        ov = {"ZJG": {"windows": {**(z.get("windows") or {}), "since_utc": windows_since}}}
    pn = start(ROOT, tmp / "new", ov)
    return finish(po, "old (9e86565)"), finish(pn, "new")


# naval-1 (2026-11-03, the user's instruction: only the naval piers and quays in the headline counts, from
# berths.NAVAL_SINCE): checked with win-1 and naval-1 switched on for the scenario's scenes (compare_naval), against the
# same code with win-1 only. Only these files may differ, berth_class_counts.csv is added, and the headline counts drop
# by exactly the candidates and stretches counted apart as civil or unknown.
NAVAL_CHANGED = {"ship_counts.csv", "detections/2026-11.geojson", "status.json", "quay/2026-11.geojson",
                 "quay/ZJG-zone-prior.json", "quay_counts.csv", "quay_est2.csv", "quay_est2.json", "quay_sectors.csv",
                 "windows/ZJG.geojson", "anchorage_counts.csv"}
NAVAL_NEW_FILES = {"berth_class_counts.csv"}


def compare_naval(tmp: Path, since: str = "2026-11-01") -> tuple[dict, dict]:
    """(win-1 only, win-1 and naval-1) results of the new code on the same scenario, run in parallel."""
    from sat.run import load_sites
    z = next(s for s in load_sites() if s["id"] == "ZJG")
    w = {"ZJG": {"windows": {**(z.get("windows") or {}), "since_utc": since}}}
    pa = start(ROOT, tmp / "win", w)
    pb = start(ROOT, tmp / "naval", {**w, "_naval_since": since})
    return finish(pa, "win-1"), finish(pb, "win-1 + naval-1")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        old, new = compare(Path(d))
    o, n = old["files"], new["files"]
    differ = sorted(k for k in set(o) & set(n) if o[k] != n[k])
    added = sorted(k for k in set(n) - set(o) if not k.startswith("chips/"))  # chips/ = the site's chip index (new)
    removed = sorted(set(o) - set(n))
    unexpected = sorted(set(differ) - BERTH_CHANGED)
    ok = not unexpected and added == sorted(NEW_FILES) and not removed and old["past"] == new["past"]
    print(json.dumps({"files": len(o), "differ": differ, "unexpected": unexpected, "added": added,
                      "expected_added": sorted(NEW_FILES), "removed": removed, "past_rows_same": old["past"] == new["past"],
                      "quay_results": new["quay_results"], "old_quay": old["new_quay"], "new_quay": new["new_quay"],
                      "old_ship": old["new_ship"], "new_ship": new["new_ship"], "old_lines": old["new_lines"],
                      "new_lines": new["new_lines"], "excluded": new["excluded"], "ok": ok}, ensure_ascii=False))
    if not ok:
        return 1
    with tempfile.TemporaryDirectory() as d:
        a, b = compare_naval(Path(d))
    differ = sorted(k for k in set(a["files"]) & set(b["files"]) if a["files"][k] != b["files"][k] and not k.startswith("chips/"))
    added = sorted(k for k in set(b["files"]) - set(a["files"]) if not k.startswith("chips/"))
    unexpected = sorted(set(differ) - NAVAL_CHANGED)
    ok = not unexpected and added == sorted(NAVAL_NEW_FILES) and a["past"] == b["past"]
    print(json.dumps({"naval": True, "differ": differ, "unexpected": unexpected, "added": added,
                      "win_ship": a["new_ship"], "naval_ship": b["new_ship"], "class_rows": b["class_rows"],
                      "win_quay": a["new_quay"], "naval_quay": b["new_quay"], "ok": ok}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
