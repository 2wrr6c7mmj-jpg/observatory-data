"""cfar-v2 PR3 probe modes ``raw`` (ocean/raw_cut.py) and ``wide`` (ocean/wide.py). No network: fakes only."""
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
    from ocean import raw_cut, wide
    from sat import cdse, png

NOW = datetime(2026, 10, 30, 18, 0, tzinfo=timezone.utc)


def chip(w, h, seed=0, vh_db=-26.0, zero_share=0.0):
    rng = np.random.default_rng(seed)
    vv = 10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-18 / 10))
    vh = 10 * np.log10(rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (vh_db / 10))
    vv[h // 2:h // 2 + 3, w // 2 - 4:w // 2 + 4] = 5.0
    vh[h // 2:h // 2 + 3, w // 2 - 4:w // 2 + 4] = -5.0
    enc = lambda d: np.clip(np.round((d + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
    v, hh = enc(vv), enc(vh)
    if zero_share:
        hh[rng.random((h, w)) < zero_share] = 0
    arr = np.stack([v, hh, np.full(vv.shape, 3500.0), np.ones(vv.shape)], axis=-1)
    return png.encode(arr.astype(np.uint16), depth=16, filters=[1])


class Fake:
    def __init__(self, times, geom=None, zero_share=0.0):
        self.times, self.geom, self.zero_share = times, geom, zero_share
        self.requests, self.processing_units, self.pu_reported = 0, 0.0, True
        self.calls = []

    def catalog(self, bbox, start, end):
        self.requests += 1
        w, s, e, n = bbox
        g = self.geom or {"type": "Polygon", "coordinates": [[[w - 1, s - 1], [e + 1, s - 1], [e + 1, n + 1], [w - 1, n + 1], [w - 1, s - 1]]]}
        p = {"platform": "sentinel-1a", "sat:orbit_state": "ascending", "sat:absolute_orbit": 61000, "sat:relative_orbit": 69,
             "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": f"s{k}", "geometry": g, "properties": {**p, "datetime": t}}
                for k, t in enumerate(self.times) if start <= cdse.parse_time(t) <= end]

    def process(self, bbox, w, h, t0, t1, with_lia=True, upsampling="NEAREST"):
        self.requests += 1
        pu = cdse.estimate_pu(w, h, 3)
        self.processing_units += pu
        self.calls.append((w, h, upsampling))
        return chip(w, h, seed=len(self.calls), zero_share=self.zero_share), pu


def overpass(q):
    return {"elements": []}


class Pacer:
    waited = 0.0

    def wait(self, pu):
        pass

    def spent(self, pu):
        pass


def small_areas(tmp: Path) -> Path:
    cfg = op.load_areas()
    for a in cfg["areas"]:
        a["size_m"] = 6000  # 300 px at 20 m -> 3 x 3 tiles
    p = tmp / "areas.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def args(tmp: Path, mode: str, **kw) -> Namespace:
    base = dict(out=tmp / "out", areas=small_areas(tmp), usage=tmp / "usage.csv", sat_usage=[], ocean_usage=[],
                other_usage=[], ports_usage=[], budget_month="2026-10", ocean_cap=5000.0, global_cap=9995.0,
                global_margin=50.0,
                max_pu=600.0, max_minutes=10, lookback_days=None, min_coverage=None, dry_run=False, no_check=True,
                mode=mode, raw_out=tmp / "raw", targets=None, pu_per_min=100.0, extend=False)
    base.update(kw)
    return Namespace(**base)


@unittest.skipUnless(np is not None, "numpy not installed")
class RawModeTest(unittest.TestCase):
    def test_cuts_tiles_and_meta_without_whole_scene(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "low"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            c = Fake(["2026-10-16T21:51:39Z"], zero_share=0.4)
            s = op.run_mode(args(tmp, "raw", targets=tmp / "t.json"), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.calls, [(300, 300, "BILINEAR")])
            r = s["results"][0]
            self.assertEqual(r["tiles_cut"], 9)
            z = np.load(tmp / "raw" / r["files"][0])
            self.assertEqual(z["vv"].shape, (9, 128, 128))
            self.assertEqual(z["vv"].dtype, np.uint16)
            meta = json.loads((tmp / "raw" / r["files"][1]).read_text(encoding="utf-8"))
            self.assertIn("Contains modified Copernicus Sentinel data 2026", meta["attribution"])
            self.assertIn("800", meta["encoding"]["vv"])
            self.assertEqual(len(meta["all_tiles"]), 9)
            # zero VH codes (sigma0 <= 0 after noise removal) are counted per tile
            self.assertGreater(r["scene"]["vh_zero"], 0.3)
            self.assertIn("raw tiles", (tmp / "usage.csv").read_text(encoding="utf-8"))

    def test_datamask_reaches_detect_v2(self):
        from unittest import mock
        from sat import detect2
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "low"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            c = Fake(["2026-10-16T21:51:39Z"], zero_share=0.3)
            orig = detect2.detect_v2
            with mock.patch.object(detect2, "detect_v2", side_effect=orig) as m:
                op.run_mode(args(tmp, "raw", targets=tmp / "t.json"), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(m.call_count, 1)
            self.assertTrue(m.call_args.kwargs["datamask"].all())

    def test_other_month_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "x"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            for mode in ("raw", "wide"):
                c = Fake(["2026-10-16T21:51:39Z"])
                with self.assertRaises(op.MonthStop):
                    op.run_mode(args(tmp, mode, targets=tmp / "t.json", budget_month="2026-09"), client=c,
                                overpass=overpass, now=NOW, pacer=Pacer(), sleep=lambda x: None, log=lambda m: None,
                                clock=lambda: NOW)
                self.assertEqual(c.requests, 0, mode)

    def test_month_change_stops_raw_before_the_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "x"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            c = Fake(["2026-10-16T21:51:39Z"])
            nov = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
            s = op.run_mode(args(tmp, "raw", targets=tmp / "t.json"), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: nov)
            self.assertEqual(c.requests, 0)
            self.assertEqual(op.exit_code(s), 2)
            self.assertTrue((tmp / "usage.csv").exists())

    def test_missing_scene_is_an_error_not_a_request(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "x"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            c = Fake(["2026-10-17T21:51:39Z"])
            s = op.run_mode(args(tmp, "raw", targets=tmp / "t.json"), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.calls, [])
            self.assertTrue(s["errors"])

    def test_budget_cuts_the_plan(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "x"},
                                              {"area": "TE", "acquired_utc": "2026-10-16T21:51:39Z", "why": "y"}]}
            (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
            c = Fake(["2026-10-16T21:51:39Z"])
            pu1 = cdse.estimate_pu(300, 300, 3)
            s = op.run_mode(args(tmp, "raw", targets=tmp / "t.json", max_pu=pu1 * 1.5), client=c, overpass=overpass,
                            now=NOW, pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(len(c.calls), 1)
            self.assertEqual(s["plan"]["requests"], 1)


@unittest.skipUnless(np is not None, "numpy not installed")
class WideModeTest(unittest.TestCase):
    def setUp(self):
        self._saved = (wide.REGION, wide.PIXEL_M, wide.MIN_LENGTH_M, wide.REGIONS)
        wide.REGION = (124.0, 25.0, 124.8, 25.6)   # 4 x 3 cells
        wide.REGIONS = [{"id": "T", "name": "t", "bbox": (124.0, 25.0, 124.8, 25.6), "min_eff": 0.4}]
        wide.PIXEL_M = 400.0                        # 0.2 deg = about 56 px: small images

    def tearDown(self):
        wide.REGION, wide.PIXEL_M, wide.MIN_LENGTH_M, wide.REGIONS = self._saved

    def test_plan_covers_footprint_and_reports_holes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # footprint covers the western half of the region only
            geom = {"type": "Polygon", "coordinates": [[[123.0, 24.0], [124.4, 24.0], [124.4, 27.0], [123.0, 27.0], [123.0, 24.0]]]}
            c = Fake(["2026-10-28T09:50:00Z", "2026-10-20T09:50:00Z"], geom=geom)
            s = op.run_mode(args(tmp, "wide", max_pu=500.0), client=c, overpass=overpass, now=NOW, pacer=Pacer(),
                            sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(len(s["acquisitions"]), 1)  # the one older than 7 days is not in the catalog window
            self.assertGreaterEqual(s["plan"]["requests"], 1)
            for p in s["plan"]["items"]:
                self.assertLessEqual(p["bbox"][2], 124.4 + 1e-9)  # nothing outside the footprint
            holes = json.loads((tmp / "out" / "holes.geojson").read_text(encoding="utf-8"))
            reasons = {f["properties"]["reason"] for f in holes["features"]}
            self.assertEqual(reasons, {"no_scene"})
            self.assertGreater(s["coverage"]["observed_sea_km2"], 0)
            json.loads((tmp / "out" / "candidates.geojson").read_text(encoding="utf-8"))
            self.assertIn("wide", (tmp / "usage.csv").read_text(encoding="utf-8"))

    def test_budget_leaves_budget_holes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = Fake(["2026-10-28T09:50:00Z"])
            one = cdse.estimate_pu(*wide.rect_px((124.0, 25.0, 124.2, 25.2)), 3)
            s = op.run_mode(args(tmp, "wide", max_pu=one * 1.5 + 2.0), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertLessEqual(c.processing_units, one * 1.5 + 2.0)
            self.assertIn("budget", s["coverage"]["holes_sea_km2"])
            self.assertGreater(s["coverage"]["holes_sea_km2"]["budget"], 0)

    def test_overpass_failure_is_retried_once_then_a_hole(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = Fake(["2026-10-28T09:50:00Z"])
            calls = []

            def flaky(q):
                calls.append(q)
                if len(calls) == 1:
                    raise OSError("502")
                return {"elements": []}
            s = op.run_mode(args(tmp, "wide", max_pu=500.0), client=c, overpass=flaky, now=NOW, pacer=Pacer(),
                            sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(len(c.calls), s["plan"]["requests"])  # the failed one was fetched on the retry
            self.assertEqual(s["coverage"]["holes_sea_km2"]["osm_failed"], 0)

            def down(q):
                raise OSError("502")
            c2 = Fake(["2026-10-28T09:50:00Z"])
            s2 = op.run_mode(args(tmp, "wide", max_pu=500.0, out=tmp / "o2"), client=c2, overpass=down, now=NOW,
                             pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c2.calls, [])
            self.assertGreater(s2["coverage"]["holes_sea_km2"]["osm_failed"], 0)

    def test_extend_fills_holes_then_second_look(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            half = {"type": "Polygon", "coordinates": [[[123.0, 24.0], [124.4, 24.0], [124.4, 27.0], [123.0, 27.0], [123.0, 24.0]]]}
            c = Fake(["2026-10-28T09:50:00Z"], geom=half)
            run1 = tmp / "res" / "r1"
            op.run_mode(args(tmp, "wide", max_pu=500.0, out=run1), client=c, overpass=overpass, now=NOW, pacer=Pacer(),
                        sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            full = None  # second scene: whole region, newer
            c2 = Fake(["2026-10-28T09:50:00Z", "2026-10-29T21:50:00Z"], geom=full)
            c2.geom = None
            s = op.run_mode(args(tmp, "wide", max_pu=500.0, out=tmp / "res" / "r2", extend=True), client=c2, overpass=overpass,
                            now=NOW, pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(s["prior_runs"], ["r1"])
            kinds = {p["kind"] for p in s["plan"]["items"]}
            self.assertEqual(kinds, {"first", "second"})
            # a second look never repeats the scene that already covers the same rectangle
            firsts = {(p["acquired_utc"], tuple(p["bbox"])) for p in s["plan"]["items"] if p["kind"] == "first"}
            for p in s["plan"]["items"]:
                if p["kind"] == "second":
                    self.assertNotIn((p["acquired_utc"], tuple(p["bbox"])), firsts)
            self.assertGreater(s["per_region"]["T"]["seen_twice_sea_km2"], 0)
            self.assertEqual(s["coverage"]["holes_sea_km2"]["no_scene"], 0)

    def test_detect_rect_passes_datamask_and_keeps_vh_zero_as_data(self):
        from unittest import mock
        from sat import detect2
        from sat.grid import Grid
        vv, vh, lia, dm = op.decode(chip(200, 200, zero_share=0.3))
        land = np.zeros(vv.shape, bool)
        g = Grid(122.0, 24.0, 122.08, 24.07, 200, 200)
        orig = detect2.detect_v2
        with mock.patch.object(detect2, "detect_v2", side_effect=orig) as m:
            _c, info = wide.detect_rect(vv, vh, lia, land, land.copy(), g,
                                        {"acquired_utc": "2026-10-16T21:51:39Z", "orbit_direction": "ASCENDING"}, datamask=dm)
        self.assertIs(m.call_args.kwargs["datamask"], dm)
        self.assertEqual(info["valid_fraction"], 1.0)      # 30% VH zeros, no dataMask 0

    def test_dry_run_spends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = Fake(["2026-10-28T09:50:00Z"])
            s = op.run_mode(args(tmp, "wide", dry_run=True), client=c, overpass=overpass, now=NOW, pacer=Pacer(),
                            sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            self.assertEqual(c.calls, [])
            self.assertGreater(s["plan"]["requests"], 0)

    def test_confidence_and_subregion(self):
        self.assertEqual(wide.confidence({"contrast_vv_db": 12, "vh_ratio_db": 7, "pixels": 5}), "high")
        self.assertEqual(wide.confidence({"contrast_vv_db": 12, "vh_ratio_db": 7, "pixels": 5, "edge": True}), "medium")
        self.assertEqual(wide.confidence({"contrast_vv_db": 3, "vh_ratio_db": 3, "pixels": 3}), "low")
        self.assertEqual(wide.subregion(124.5, 24.5), "SAKI")
        self.assertEqual(wide.subregion(123.6, 25.8), "SENKAKU")
        self.assertEqual(wide.subregion(122.5, 22.0), "TWE")

    def test_priority_senkaku_first(self):
        wide.REGIONS, wide.REGION = self._saved[3], None
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = Fake(["2026-10-28T09:50:00Z"])  # footprint around each catalog bbox: every region imaged
            s = op.run_mode(args(tmp, "wide", max_pu=2.0 + 0.3, dry_run=True), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            items = s["plan"]["items"]
            self.assertTrue(items)
            self.assertEqual(items[0]["region"], "SENKAKU")
            self.assertIn("SENKAKU", s["per_region"])

    def test_partial_cells_are_not_requested_twice(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # the footprint edge cuts through the middle of a column of cells
            geom = {"type": "Polygon", "coordinates": [[[123.0, 24.0], [124.3, 24.0], [124.3, 27.0], [123.0, 27.0], [123.0, 24.0]]]}
            c = Fake(["2026-10-28T09:50:00Z"], geom=geom)
            s = op.run_mode(args(tmp, "wide", max_pu=500.0, dry_run=True), client=c, overpass=overpass, now=NOW,
                            pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            keys = [(p["acquired_utc"], tuple(p["bbox"])) for p in s["plan"]["items"]]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertLessEqual(len(keys), 4)
        self.assertEqual(wide.subregion(123.0, 30.0), "ECS-N")


class Raising(Fake):
    """Fake whose process call number ``at`` raises ``exc`` (the request was sent, no PU reported)."""

    def __init__(self, *a, exc=None, at=2, **kw):
        super().__init__(*a, **kw)
        self.exc, self.at = exc, at

    def process(self, *a, **kw):
        if len(self.calls) + 1 == self.at:
            self.requests += 1
            raise self.exc
        return super().process(*a, **kw)


@unittest.skipUnless(np is not None, "numpy not installed")
class ModesUsageOnAnyEndTest(unittest.TestCase):
    """raw and wide write the PU spent to --usage however they end (1部 指摘 1)."""

    def setUp(self):
        self._saved = (wide.REGION, wide.PIXEL_M, wide.MIN_LENGTH_M, wide.REGIONS)
        wide.REGION = (124.0, 25.0, 124.8, 25.6)
        wide.REGIONS = [{"id": "T", "name": "t", "bbox": (124.0, 25.0, 124.8, 25.6), "min_eff": 0.4}]
        wide.PIXEL_M = 400.0

    def tearDown(self):
        wide.REGION, wide.PIXEL_M, wide.MIN_LENGTH_M, wide.REGIONS = self._saved

    def check(self, tmp, c, exc, version):
        rows = op.read_csv(tmp / "usage.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["probe_version"], version)
        self.assertAlmostEqual(float(rows[0]["processing_units"]), c.processing_units, places=2)
        self.assertGreater(c.processing_units, 0)
        self.assertEqual(int(rows[0]["requests"]), c.requests)
        self.assertIn(type(exc).__name__, rows[0]["note"])
        self.assertTrue((tmp / "out" / "report.txt").exists())

    def test_raw_urlerror_and_credential_error(self):
        import urllib.error
        for exc in (urllib.error.URLError("reset"), cdse.CredentialError("HTTP 403")):
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                tg = {"version": "t", "targets": [{"area": "TS", "acquired_utc": "2026-10-16T21:51:39Z", "why": "a"},
                                                  {"area": "HK", "acquired_utc": "2026-10-16T21:51:39Z", "why": "b"}]}
                (tmp / "t.json").write_text(json.dumps(tg), encoding="utf-8")
                c = Raising(["2026-10-16T21:51:39Z"], exc=exc, at=2)
                with self.assertRaises(type(exc)):
                    op.run_mode(args(tmp, "raw", targets=tmp / "t.json", max_pu=None), client=c, overpass=overpass,
                                now=NOW, pacer=Pacer(), sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
                self.check(tmp, c, exc, raw_cut.RAW_VERSION)

    def test_wide_urlerror_and_credential_error(self):
        import urllib.error
        from unittest import mock
        for exc in (urllib.error.URLError("reset"), cdse.CredentialError("401")):
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                c = Raising(["2026-10-28T09:50:00Z"], exc=exc, at=2)
                with mock.patch.object(wide, "MAX_PX", 60), self.assertRaises(type(exc)):  # one cell per request
                    op.run_mode(args(tmp, "wide", max_pu=500.0), client=c, overpass=overpass, now=NOW, pacer=Pacer(),
                                sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
                self.check(tmp, c, exc, wide.WIDE_VERSION)

    def test_wide_credits_from_the_acquisitions(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            c = Fake(["2026-10-28T09:50:00Z"])
            op.run_mode(args(tmp, "wide", max_pu=500.0), client=c, overpass=overpass, now=NOW, pacer=Pacer(),
                        sleep=lambda x: None, log=lambda m: None, clock=lambda: NOW)
            cand = json.loads((tmp / "out" / "candidates.geojson").read_text(encoding="utf-8"))
            cov = json.loads((tmp / "out" / "coverage.geojson").read_text(encoding="utf-8"))
            self.assertEqual(cov["attribution"], "Contains modified Copernicus Sentinel data 2026")
            self.assertIn("© OpenStreetMap contributors, ODbL", cand["attribution"])
            self.assertIn("出典: Contains modified Copernicus Sentinel data 2026",
                          (tmp / "out" / "report.txt").read_text(encoding="utf-8"))
            self.assertFalse(hasattr(wide, "ATTRIBUTION"))
            self.assertFalse(hasattr(raw_cut, "ATTRIBUTION"))


@unittest.skipUnless(np is not None, "numpy not installed")
class RawDiagnosticsTest(unittest.TestCase):
    """tile_diagnostics: no data as in cfar2-pr2.1 (VH code 0 inside the dataMask is data) (1部 指摘 5)."""

    def test_vh_zero_is_data_and_not_in_the_nodata_band(self):
        h = w = 256
        rng = np.random.default_rng(0)
        raw = np.zeros((h, w, 4), np.uint16)
        raw[..., 0] = 32000
        raw[..., 1] = rng.integers(25000, 27000, (h, w))
        raw[..., 2] = 3500
        raw[..., 3] = 1
        raw[..., 1][rng.random((h, w)) < 0.3] = 0       # VH sigma0 <= 0 inside the data
        raw[:, :20, 3] = 0                               # no data: dataMask 0
        raw[:, :20, :3] = 0
        raw[200:, 200:, 0] = 0                           # no data: VV code 0
        land = np.zeros((h, w), bool)
        diag = {(d["ty"], d["tx"]): d for d in raw_cut.tile_diagnostics(raw, land, land.copy(), [])}
        # the band is that of dataMask 0 / VV 0 only, as in detect_v2 (dilate(~valid, 3))
        from sat.grid import dilate
        valid = (raw[..., 3] > 0) & (raw[..., 0] > 0)
        band = dilate(~valid, 3)
        for (ty, tx), d in diag.items():
            sl = (slice(ty * 128, (ty + 1) * 128), slice(tx * 128, (tx + 1) * 128))
            self.assertAlmostEqual(d["nodata_band"], round(float(band[sl].mean()), 4))
        self.assertLess(diag[(0, 1)]["nodata_band"], 0.01)   # VH zeros alone add no band
        self.assertGreater(diag[(0, 1)]["vh_zero"], 0.25)
        # the VH median is over VH code > 0 only (not pulled down to -50 dB by the zeros)
        self.assertGreater(diag[(0, 1)]["vh_med_db"], 25000 / cdse.DB_SCALE - cdse.DB_OFFSET)
        self.assertGreater(diag[(0, 1)]["vh_p05_db"], 24000 / cdse.DB_SCALE - cdse.DB_OFFSET)


if __name__ == "__main__":
    unittest.main()
