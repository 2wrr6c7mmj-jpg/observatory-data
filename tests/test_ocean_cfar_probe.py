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
                  other_usage=[], ports_usage=[], budget_month="2026-10", ocean_cap=5000.0, global_cap=9995.0,
                  global_margin=50.0, max_pu=None, max_minutes=60, dry_run=False, no_check=True)
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
    def write(self, path: Path, rows: list[dict]) -> Path:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r})
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return path

    def test_no_month_caps_in_code(self):
        for name in ("OCEAN_MONTHLY_PU_CAP", "TOTAL_MONTHLY_PU_GUARD", "MONTHLY_REQUEST_CAP_BY_MONTH",
                     "DEFAULT_MAX_PU", "caps_for"):
            self.assertFalse(hasattr(op, name), name)
        src = (Path(op.__file__).parent / "cfar_probe.py").read_text(encoding="utf-8")
        self.assertNotIn('"2026-', src)

    def test_budget_sums_every_use_once_with_the_shared_guard(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            sat = self.write(tmp / "sat.csv", [
                {"run_utc": "2026-10-30T05:00:00Z", "processing_units": "3000", "requests": "10"},
                {"run_utc": "2026-09-30T05:00:00Z", "processing_units": "999", "requests": "10"}])
            port_row = {"run_utc": "2026-10-29T01:00:00Z", "site_id": "SYL", "archive_version": "ports-1",
                        "processing_units": "2000", "requests": "40"}
            # the same port record in ports/SYL/usage.csv and ports_usage.csv, on two branches: counted once
            ports = [self.write(tmp / "p" / "ports-SYL--0.csv", [port_row]),
                     self.write(tmp / "p" / "ports-all--1.csv", [port_row]),
                     self.write(tmp / "p" / "ports-SYL--2.csv", [port_row])]
            oc_row = {"run_utc": "2026-10-30T06:00:00Z", "probe_version": "ocean-cfar-probe-1",
                      "processing_units": "100", "requests": "5"}
            other = [self.write(tmp / "o" / "other-docs_data_ocean_usage.csv--0.csv", [oc_row]),
                     self.write(tmp / "o" / "other-docs_data_ocean_usage.csv--1.csv", [oc_row]),
                     self.write(tmp / "o" / "other-docs_data_x_usage.csv--2.csv",
                                [{"run_utc": "2026-10-02T00:00:00Z", "processing_units": "10", "requests": "1"}])]
            a = args(tmp, sat_usage=[sat], ports_usage=ports, other_usage=other)
            b = op.budget(a, NOW)
            self.assertEqual(b["ocean_month_pu"], 100)
            self.assertEqual(b["ports_month_pu"], 2000)
            self.assertEqual(b["sat_month_pu"], 3010)
            self.assertEqual(b["pu_left"], 9995 - 50 - 3010 - 2000 - 100)   # the global guard binds
            self.assertEqual(b["requests_left"], 8500 - 11 - 5 - 40)
            a.ocean_cap = 1000.0
            self.assertEqual(op.budget(a, NOW)["pu_left"], 900)             # the ocean cap binds
            a.max_pu = 300.0
            self.assertEqual(op.budget(a, NOW)["pu_left"], 300)             # the run's cap binds
            a.global_cap = 5000.0
            self.assertEqual(op.budget(a, NOW)["pu_left"], 0)               # never negative

    def test_other_month_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a = args(Path(d))
            with self.assertRaises(op.MonthStop):
                op.budget(a, datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc))
            c = FakeClient()
            with self.assertRaises(op.MonthStop):
                op.run(a, client=c, overpass=fake_overpass, now=datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc),
                       pacer=FakePacer(), sleep=lambda x: None, log=lambda m: None,
                       clock=lambda: datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc))
            self.assertEqual(c.requests, 0)
            self.assertFalse((Path(d) / "ocean_usage.csv").exists())

    def test_main_needs_every_cap_and_exits_2_in_another_month(self):
        import contextlib
        import io
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            base = ["--out", str(Path(d) / "o"), "--budget-month", "2000-01", "--ocean-cap", "10",
                    "--global-cap", "10", "--global-margin", "1"]
            for k in ("--budget-month", "--ocean-cap", "--global-cap", "--global-margin"):
                i = base.index(k)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    op.main(base[:i] + base[i + 2:])
            with mock.patch.object(op.cdse.Client, "_post_sh", side_effect=AssertionError("sent")), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(op.main(base), 2)
                self.assertEqual(op.main(base + ["--mode", "wide"]), 2)
                self.assertEqual(op.main(base + ["--dry-run"]), 2)
            self.assertFalse((Path(d) / "o").exists())

    def test_month_client_sends_nothing_after_the_month(self):
        from unittest import mock
        t = [datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc)]
        c = op.MonthClient("2026-10", utcnow=lambda: t[0], client_id="x", client_secret="y")
        with mock.patch.object(op.cdse.Client, "_post_sh", return_value=({}, b"[]")) as sent:
            c._post_sh(op.cdse.CATALOG_URL, {}, "application/json")
            t[0] = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
            with self.assertRaises(op.MonthStop):
                c._post_sh(op.cdse.CATALOG_URL, {}, "application/json")
        self.assertEqual(sent.call_count, 1)

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
            self.assertEqual(c2.requests, 0)                       # not even the catalog
            self.assertEqual(s2["errors"][0]["stage"], "stop")
            self.assertEqual(op.exit_code(s2), 2)

    def test_month_change_during_the_run_stops_before_the_next_request(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = FakeClient(n_scenes=3)
            nov = datetime(2026, 11, 1, 0, 0, 5, tzinfo=timezone.utc)
            clock = lambda: NOW if len(c.calls) < 2 else nov    # the month changes after the 2nd scene
            s = op.run(args(tmp), client=c, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                       sleep=lambda x: None, log=lambda m: None, clock=clock)
            self.assertEqual(len(c.calls), 2)
            self.assertEqual(len(s["results"]), 2)
            self.assertEqual([e["reason"] for e in s["errors"]], ["month"])
            self.assertEqual(op.exit_code(s), 2)
            rows = op.read_csv(tmp / "ocean_usage.csv")             # the PU spent are still recorded
            self.assertAlmostEqual(float(rows[0]["processing_units"]), c.processing_units, places=2)

    def test_low_coverage_scene_not_fetched(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tiny = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
            c = FakeClient(footprint=tiny)
            s = op.run(args(tmp), client=c, overpass=fake_overpass, now=NOW, pacer=FakePacer(),
                       sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.calls, [])
            self.assertEqual(s["plan"]["requests"], 0)


@unittest.skipUnless(np is not None, "numpy not installed")
class DatamaskTest(unittest.TestCase):
    """No data as in detect2 (cfar2-pr2.1): dataMask 0 or VV code 0; VH code 0 with data is data."""

    def scene(self):
        w = h = 120
        rng = np.random.default_rng(3)
        enc = lambda d: np.clip(np.round((d + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        vv = enc(10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-1.8)))
        vh = enc(10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-2.6)))
        dm = np.ones((h, w))
        dm[:, :10] = 0
        vv[:, :10] = vh[:, :10] = 0
        vh[rng.random((h, w)) < 0.2] = 0        # VH sigma0 <= 0 inside the data
        arr = np.stack([vv, vh, np.full((h, w), 3500.0), dm], axis=-1)
        return png.encode(arr.astype(np.uint16), depth=16, filters=[1]), dm

    def test_decode_returns_datamask_and_vh_zero_is_data(self):
        data, dm = self.scene()
        vv, vh, lia, mask = op.decode(data)
        self.assertTrue(np.array_equal(mask, dm > 0))
        self.assertGreater(np.isnan(vh[:, 10:]).mean(), 0.1)
        self.assertAlmostEqual(float(op.data_valid(vv, mask).mean()), 110 / 120, places=6)

    def test_compare_passes_datamask_to_detect_v2(self):
        from unittest import mock
        data, _dm = self.scene()
        vv, vh, lia, mask = op.decode(data)
        g = Grid(120.0, 24.0, 120.02, 24.02, 120, 120)
        land = np.zeros(vv.shape, bool)
        orig = op.v2.detect_v2
        with mock.patch.object(op.v2, "detect_v2", side_effect=orig) as m:
            res, _f = op.compare(vv, vh, lia, land, land.copy(), g, {"acquired_utc": "2026-10-16T21:51:39Z",
                                                                      "orbit_direction": "ASCENDING"}, datamask=mask)
        self.assertIs(m.call_args.kwargs["datamask"], mask)
        self.assertAlmostEqual(res["valid_fraction"], round(110 / 120, 4))   # VH zeros are not no data


@unittest.skipUnless(np is not None, "numpy not installed")
class ProductionUnchangedTest(unittest.TestCase):
    def test_only_the_probe_workflow_runs_the_ocean_probe(self):
        root = Path(__file__).resolve().parent.parent
        for wf in (root / ".github" / "workflows").glob("*.yml"):
            text = wf.read_text(encoding="utf-8")
            if wf.name == "ocean-cfar-probe.yml":
                on = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
                self.assertNotIn("workflow_dispatch", on)
                self.assertNotIn("schedule", on)
                self.assertIn("probe/ocean-cfar", on)
                continue
            self.assertNotIn("cfar_probe", text, wf.name)
        self.assertNotIn("ocean", (root / "sat" / "run.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
