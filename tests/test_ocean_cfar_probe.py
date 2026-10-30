"""cfar-v2 PR3: the open-sea probe (ocean/cfar_probe.py). No network: the CDSE client and Overpass are fakes."""
from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from ocean import cfar_probe as op
    from sat import cdse, png
    from sat.grid import Grid

NOW = datetime(2026, 10, 30, 18, 0, tzinfo=timezone.utc)


def chip_png(w, h, seed=0, ship=True):
    rng = np.random.default_rng(seed)
    vv = 10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-18 / 10))
    vh = 10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-26 / 10))
    if ship:
        vv[h // 2:h // 2 + 3, w // 2 - 6:w // 2 + 6] = 5.0
        vh[h // 2:h // 2 + 3, w // 2 - 6:w // 2 + 6] = -5.0
    vv[:, :8] = np.nan  # an edge without data
    vh[:, :8] = np.nan
    enc = lambda d: np.where(np.isfinite(d), np.clip(np.round((np.nan_to_num(d) + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535), 0)
    dm = np.isfinite(vv).astype(float)
    arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 3500.0), dm], axis=-1)
    return png.encode(arr.astype(np.uint16), depth=16, filters=[1])


class FakeClient:
    def __init__(self, n_scenes=3, footprint=None):
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True
        self.n = n_scenes
        self.footprint = footprint
        self.calls = []

    def catalog(self, bbox, start, end):
        self.requests += 1
        w, s, e, n = bbox
        geom = self.footprint or {"type": "Polygon", "coordinates": [[[w - 1, s - 1], [e + 1, s - 1], [e + 1, n + 1], [w - 1, n + 1], [w - 1, s - 1]]]}
        p = {"platform": "sentinel-1a", "sat:orbit_state": "descending", "sat:absolute_orbit": 61000,
             "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": f"s{k}", "geometry": geom, "properties": {**p, "datetime": f"2026-10-{10 + 2 * k:02d}T22:30:00Z"}}
                for k in range(self.n)]

    def process(self, bbox, w, h, t0, t1, with_lia=True, upsampling="NEAREST"):
        self.requests += 1
        pu = cdse.estimate_pu(w, h, 3)
        self.processing_units += pu
        self.calls.append((w, h, upsampling))
        return chip_png(w, h, seed=len(self.calls)), pu


def fake_overpass(query):
    return {"osm3s": {"timestamp_osm_base": "2026-10-30T00:00:00Z"}, "elements": [
        {"type": "node", "lon": 120.0, "lat": 24.0, "tags": {"power": "generator", "generator:source": "wind"}},
        {"type": "node", "lon": 120.01, "lat": 24.01, "tags": {"seamark:type": "separation_line"}},
    ]}


class FakePacer:
    def __init__(self):
        self.waits, self.spents, self.waited = [], [], 0.0

    def wait(self, pu):
        self.waits.append(pu)

    def spent(self, pu):
        self.spents.append(pu)


def areas_file(tmp: Path, size_m=6000, max_scenes=2) -> Path:
    cfg = json.loads(op.AREAS_FILE.read_text(encoding="utf-8"))
    cfg["areas"] = [{"id": "TS", "name": "t", "center_lon": 120.0, "center_lat": 24.0, "size_m": size_m, "max_scenes": max_scenes},
                    {"id": "HK", "name": "h", "center_lon": 114.0, "center_lat": 22.0, "size_m": size_m, "max_scenes": max_scenes}]
    p = tmp / "areas.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def args(tmp: Path, **kw):
    a = Namespace(out=tmp / "out", areas=areas_file(tmp), usage=tmp / "ocean_usage.csv", sat_usage=[], ocean_usage=[],
                  max_pu=op.DEFAULT_MAX_PU, max_minutes=60, dry_run=False, no_check=True)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@unittest.skipUnless(np is not None, "numpy not installed")
class AreasTest(unittest.TestCase):
    def test_real_areas_are_one_unit_each_within_limit(self):
        cfg = op.load_areas()
        self.assertEqual(cfg["pixel_m"], 20)
        ids = [a["id"] for a in cfg["areas"]]
        self.assertEqual(len(ids), len(set(ids)))
        for a in cfg["areas"]:
            us = op.units(a, cfg["pixel_m"])
            self.assertEqual(len(us), 1, a["id"])
            g = us[0]["grid"]
            self.assertLessEqual(max(g.width, g.height), op.MAX_PX)
            self.assertAlmostEqual(cdse.estimate_pu(g.width, g.height, 3), 47.684, places=2)

    def test_split_into_units(self):
        us = op.units({"id": "X", "center_lon": 120, "center_lat": 24, "size_m": 120000}, 20)
        self.assertEqual(len(us), 9)  # 6,000 px -> 3 x 2,000 px
        for u in us:
            self.assertLessEqual(max(u["grid"].width, u["grid"].height), op.MAX_PX)
        self.assertEqual(sum(u["grid"].width * u["grid"].height for u in us), 6000 * 6000)

    def test_footprint_and_spread(self):
        g = Grid(120.0, 24.0, 120.5, 24.5, 100, 100)
        half = {"type": "Polygon", "coordinates": [[[119, 23], [120.25, 23], [120.25, 25], [119, 25], [119, 23]]]}
        cov = op.footprint_coverage([half], g)
        self.assertTrue(0.4 < cov < 0.6, cov)
        self.assertEqual(op.footprint_coverage([None], g), 1.0)
        self.assertEqual(op.spread(list(range(10)), 3), [0, 4, 9])
        self.assertEqual(op.spread([1, 2], 5), [1, 2])


@unittest.skipUnless(np is not None, "numpy not installed")
class BudgetTest(unittest.TestCase):
    def test_caps_by_month(self):
        self.assertEqual(op.caps_for("2026-10")["ocean_pu"], 4000.0)
        self.assertEqual(op.caps_for("2026-11")["ocean_pu"], 400.0)
        self.assertEqual(op.caps_for("2026-10")["total_pu"], 9000.0)

    def test_budget_takes_smallest(self):
        sat = [{"run_utc": "2026-10-30T05:00:00Z", "processing_units": "8000", "requests": "10"},
               {"run_utc": "2026-09-30T05:00:00Z", "processing_units": "999", "requests": "10"}]
        oc = [{"run_utc": "2026-10-30T06:00:00Z", "processing_units": "100", "requests": "5"}]
        b = op.budget("2026-10", sat, oc + oc, 3900)  # the same row from two branches counts once
        self.assertEqual(b["ocean_month_pu"], 100)
        self.assertEqual(b["pu_left"], 900)       # 9000 - 8000 - 100
        b = op.budget("2026-11", [], [], 3900)
        self.assertEqual(b["pu_left"], 400)

    def test_plan_round_robin_and_cut(self):
        per = {"A": [{"area": "A", "k": i, "est_pu": 48} for i in range(3)],
               "B": [{"area": "B", "k": i, "est_pu": 48} for i in range(3)]}
        plan, deferred = op.plan_requests(per, [], 150, 100)
        self.assertEqual([(p["area"], p["k"]) for p in plan], [("A", 0), ("B", 0), ("A", 1)])
        self.assertEqual(len(deferred), 3)
        plan, _ = op.plan_requests(per, [], 1000, 2)
        self.assertEqual(len(plan), 2)

    def test_pacer_waits_under_limit(self):
        t = [0.0]
        slept = []
        p = op.Pacer(pu_per_min=200, clock=lambda: t[0], sleep=lambda d: (slept.append(d), t.__setitem__(0, t[0] + d)))
        for _ in range(4):
            p.wait(48)
            p.spent(48)
        self.assertEqual(slept, [])           # 192 PU in the minute
        p.wait(48)
        self.assertEqual(len(slept), 1)        # the 5th waits until the first leaves the window
        self.assertGreaterEqual(t[0], 60.0)


@unittest.skipUnless(np is not None, "numpy not installed")
class MaskTest(unittest.TestCase):
    def test_static_from_osm_not_lanes(self):
        g = Grid(119.99, 23.99, 120.02, 24.02, 150, 150)
        land, static, info = op.masks(g, fake_overpass(""), op.load_areas()["osm_static"]["seamark_types"], 300)
        self.assertFalse(land.any())
        self.assertEqual(info["static_objects"], 1)   # the wind turbine; the separation line is not a fixed object
        self.assertGreater(static.sum(), 100)
        q = op.overpass_query(g, "^(buoy_.*)$")
        self.assertIn('generator:source"="wind"', q)
        self.assertIn('offshore_platform', q)


@unittest.skipUnless(np is not None, "numpy not installed")
class RunTest(unittest.TestCase):
    def test_dry_run_spends_no_pu(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = FakeClient()
            s = op.run(args(tmp, dry_run=True), client=c, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                       sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.processing_units, 0)
            self.assertEqual(s["plan"]["per_area_scenes"], {"TS": 2, "HK": 2})
            self.assertTrue((tmp / "ocean_usage.csv").exists())

    def test_full_run_v1_v2_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = FakeClient(n_scenes=3)
            pacer = FakePacer()
            a = args(tmp, no_check=False)
            s = op.run(a, client=c, overpass=fake_overpass, now=NOW, pacer=pacer, sleep=lambda x: None,
                       log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(s["errors"], [])
            self.assertEqual(len(s["interpolation_check"]), 3)
            self.assertEqual({x[2] for x in c.calls[3:]}, {op.INTERPOLATION})
            self.assertEqual(len(s["results"]), 4)               # 2 areas x 2 scenes (spread out of 3)
            self.assertEqual(len(pacer.waits), len(c.calls))
            r = s["results"][0]
            self.assertGreaterEqual(r["v1"]["candidates"], 1)
            self.assertGreaterEqual(r["v2"]["confidence_A"], 1)
            self.assertIn("1e-03", r["v2"]["exceedance"])
            self.assertIn("model_counts", r["v2"])
            self.assertGreaterEqual(r["match_v1_all_v2_A_100m"]["both"], 1)
            fc = json.loads((a.out / r["geojson"]).read_text(encoding="utf-8"))
            self.assertEqual({f["properties"]["detector"] for f in fc["features"]}, {"cfar-v1", "cfar-v2"})
            # no sigma0 written: only small json/geojson/txt
            for f in a.out.iterdir():
                self.assertIn(f.suffix, (".json", ".geojson", ".txt"))
            rows = op.read_csv(a.usage)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["processing_units"]), c.processing_units, places=2)
            json.loads((a.out / "summary.json").read_text(encoding="utf-8"))

    def test_cap_cuts_plan_and_month_change_stops(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = FakeClient(n_scenes=3)
            est = cdse.estimate_pu(300, 300, 3)
            s = op.run(args(tmp, max_pu=est * 1.5), client=c, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                       sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(len(c.calls), 1)
            self.assertEqual(s["plan"]["deferred"], 3)
            c2 = FakeClient(n_scenes=3)
            later = datetime(2026, 11, 1, 0, 1, tzinfo=timezone.utc)
            s2 = op.run(args(tmp), client=c2, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                        sleep=lambda x: None, log=lambda m: None, clock=lambda: later)
            self.assertEqual(c2.calls, [])
            self.assertEqual(s2["errors"][0]["stage"], "stop")

    def test_low_coverage_scene_not_fetched(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tiny = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
            c = FakeClient(footprint=tiny)
            s = op.run(args(tmp), client=c, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                       sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.calls, [])
            self.assertEqual(s["plan"]["requests"], 0)


if __name__ == "__main__":
    unittest.main()
