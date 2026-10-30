import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, cdse, png, quay_lines
    from sat.grid import Grid

SITE = {"id": "TST", "bbox": [110.0, 21.0, 110.004, 21.004], "pixel_m": 10, "mask_version": 2,
        "baseline": {"orbits": [157, 84], "min_scenes": 4, "min_valid": 3}}


def chip_png(vv_db, vh_db):
    enc = lambda d: np.where(np.isfinite(d), np.clip(np.round((d + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535), 0)
    ok = np.isfinite(vv_db).astype(float)
    return png.encode(np.stack([enc(vv_db), enc(vh_db), ok], -1).astype(np.uint16), depth=16, filters=[1])


class FakeClient:
    """Catalog: orbit 157 every 6 days (8 scenes), orbit 84 every 12 days (3 scenes)."""

    def __init__(self, shape, ship_on=()):
        self.shape = shape
        self.ship_on = set(ship_on)
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True
        self.calls = []

    def catalog(self, bbox, start, end):
        self.requests += 1
        feats = []
        t0 = datetime(2026, 5, 1, 10, 48, 45, tzinfo=timezone.utc)
        for k in range(8):
            feats.append({"id": f"a{k}", "properties": {
                "datetime": cdse.iso(t0 + timedelta(days=6 * k)), "platform": "sentinel-1c",
                "sat:orbit_state": "ascending", "sat:relative_orbit": 157,
                "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}})
        for k in range(3):
            feats.append({"id": f"b{k}", "properties": {
                "datetime": cdse.iso(t0 + timedelta(days=12 * k + 1)), "platform": "sentinel-1d",
                "sat:orbit_state": "ascending", "sat:relative_orbit": 84,
                "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}})
        return feats

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 0.5
        k = len(self.calls)
        self.calls.append((t0, with_lia))
        rng = np.random.default_rng(k)
        vv = 10 * np.log10(rng.gamma(4.4, 0.01 / 4.4, size=(h, w)))
        vh = 10 * np.log10(rng.gamma(4.4, 0.002 / 4.4, size=(h, w)))
        vv[:, :10] = 5.0  # land, bright and steady
        vh[:, :10] = -5.0
        if k in self.ship_on:
            vv[10:20, 12:14] = 10.0
            vh[10:20, 12:14] = 0.0
        return chip_png(vv, vh), 0.5


@unittest.skipUnless(np is not None, "numpy not installed")
class QuantTest(unittest.TestCase):
    def test_roundtrip(self):
        db = np.array([[-45.0, -20.1, 0.0], [18.5, np.nan, 30.0]])
        q = baseline.quantise(db)
        self.assertEqual(q[1, 1], 0)
        back = baseline.dequantise(q)
        self.assertTrue(np.isnan(back[1, 1]))
        self.assertAlmostEqual(back[0, 0], -45.0)
        self.assertAlmostEqual(back[0, 1], -20.0)  # 0.25 dB steps
        self.assertAlmostEqual(back[1, 2], 18.5)   # clipped

    def test_smooth_in_linear_power(self):
        db = np.full((5, 5), -20.0)
        db[2, 2] = -10.0
        s = baseline.smooth_db(db)
        lin = (8 * 0.01 + 0.1) / 9
        self.assertAlmostEqual(s[2, 2], 10 * np.log10(lin), places=6)
        db[0, 0] = np.nan
        self.assertTrue(np.isnan(baseline.smooth_db(db)[0, 0]))

    def test_credibility(self):
        self.assertEqual(baseline.credibility(24), "3")
        self.assertEqual(baseline.credibility(23), "4")


@unittest.skipUnless(np is not None, "numpy not installed")
class ComputeTest(unittest.TestCase):
    def test_percentiles_ignore_a_ship_seen_once(self):
        rng = np.random.default_rng(3)
        stack = [np.full((4, 4), -20.0) + rng.normal(0, 0.1, (4, 4)) for _ in range(20)]
        stack[5][1, 1] = 5.0  # a ship in one scene
        keep = np.ones((4, 4), bool)
        out = baseline.compute(stack, [s - 6 for s in stack], keep, {"min_valid": 10})
        self.assertLess(out["vv_p10"][1, 1], -19.5)
        self.assertLess(out["vv_p90"][1, 1], -19.5)
        occ = baseline.always_occupied(out["vv_p10"], out["vv_p90"])
        self.assertFalse(occ.any())
        # outside the kept region: no data
        keep[0, 0] = False
        out = baseline.compute(stack, stack, keep, {"min_valid": 10})
        self.assertTrue(np.isnan(out["vv_p10"][0, 0]))

    def test_always_occupied(self):
        p10 = np.array([-5.0, -5.0, -15.0])
        p90 = np.array([-3.0, 2.0, -14.0])
        self.assertEqual(baseline.always_occupied(p10, p90).tolist(), [True, False, False])


@unittest.skipUnless(np is not None, "numpy not installed")
class BuildTest(unittest.TestCase):
    def test_build_save_load(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, :10] = 1
        cli = FakeClient((g.height, g.width), ship_on={2})
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "baselines"
            s = baseline.build(cli, SITE, mask, out, now, 100.0, export_scenes=2, log=lambda *a: None)
            self.assertEqual(s["orbits"]["157"]["scenes"], 8)
            self.assertEqual(s["orbits"]["157"]["credibility"], "3")
            self.assertEqual(s["orbits"]["84"]["scenes"], 3)
            self.assertEqual(s["orbits"]["84"]["credibility"], "4")
            self.assertEqual(len(cli.calls), 11)
            self.assertTrue(all(not lia for _t, lia in cli.calls))  # VV and VH only
            layers, meta = baseline.load(SITE, 157, out)
            self.assertEqual(meta["relative_orbit"], 157)
            self.assertEqual(meta["mask_version"], "TST-land-v2")
            self.assertEqual(len(meta["scene_list"]), 8)
            # the ship seen in one scene does not reach the 10th percentile
            self.assertLess(np.nanmax(layers["vv_p10"][10:20, 12:14]), -10)
            # land is steady and bright: always occupied (on land, so not listed for 2部)
            self.assertTrue(baseline.always_occupied(layers["vv_p10"], layers["vv_p90"])[:, 2:8].all())
            self.assertEqual(meta["always_occupied_on_water"], [])
            self.assertIsNone(baseline.load(SITE, 99, out))
            self.assertIsNone(baseline.load(SITE, None, out))
            self.assertEqual(len(list((Path(d) / "scenes").glob("*.png"))), 8)  # 2 scenes x 2 bands x 2 orbits

    def test_budget_refuses_before_fetching(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        cli = FakeClient((g.height, g.width))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(cdse.CDSEError):
                baseline.build(cli, SITE, mask, Path(d), datetime(2026, 10, 30, tzinfo=timezone.utc), 0.1,
                               log=lambda *a: None)
        self.assertEqual(cli.calls, [])

    def test_committed_baselines_match_the_site(self):
        from sat.run import load_sites
        zjg = next(s for s in load_sites() if s["id"] == "ZJG")
        g = Grid.for_site(zjg)
        for orbit, n in ((157, 30), (84, 15)):
            layers, meta = baseline.load(zjg, orbit)
            self.assertEqual((meta["width"], meta["height"]), (g.width, g.height))
            self.assertEqual(meta["scenes"], n)
            self.assertEqual(meta["credibility"], baseline.credibility(n))
            self.assertEqual(layers["vv_p10"].shape, (g.height, g.width))
            water = np.isfinite(layers["vv_p10"])
            self.assertGreater(water.mean(), 0.4)
            self.assertLess(np.nanmedian(layers["vv_p10"]), -5)
        lines = quay_lines.load("ZJG")
        self.assertEqual(lines["counts"]["pier"], 13)

    def test_real_site_config(self):
        from sat.run import load_sites
        zjg = next(s for s in load_sites() if s["id"] == "ZJG")
        cfg = baseline.site_config(zjg)
        self.assertEqual(cfg["orbits"], [157, 84])
        self.assertEqual(cfg["min_scenes"], 24)


@unittest.skipUnless(np is not None, "numpy not installed")
class LinesTest(unittest.TestCase):
    def test_lines_from_osm(self):
        geom = lambda pts: [{"lon": a, "lat": b} for a, b in pts]
        osm = {"elements": [
            {"type": "way", "id": 3, "tags": {"man_made": "pier"}, "geometry": geom([(1, 1), (1, 2)])},
            {"type": "way", "id": 1, "tags": {"natural": "coastline"}, "geometry": geom([(0, 0), (0, 1)])},
            {"type": "way", "id": 2, "tags": {"man_made": "breakwater"}, "geometry": geom([(2, 2), (3, 3)])},
            {"type": "way", "id": 4, "tags": {"man_made": "quay"}, "geometry": geom([(0, 0), (1, 0), (1, 1), (0, 0)])},
        ]}
        ways = quay_lines.lines_from_osm(osm)
        self.assertEqual([w["kind"] for w in ways], ["pier", "quay", "coastline"])
        self.assertTrue(ways[1]["closed"])
        self.assertFalse(ways[0]["closed"])


class AnnotateRangeTest(unittest.TestCase):
    def test_b64_parts_range(self):
        from sat import annotate
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(bytes(range(256)) * 10)  # 2560 B -> 3416 base64 chars -> 35 parts of 100
            buf = io.StringIO()
            with redirect_stdout(buf):
                annotate.main(["b64", "t", str(p), "100", "30", "10"])
            lines = buf.getvalue().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertTrue(lines[0].startswith("::notice title=t 31/35::"))
            buf = io.StringIO()
            with redirect_stdout(buf):
                annotate.main(["b64", "t", str(p), "100"])
            lines = buf.getvalue().splitlines()
            self.assertEqual(len(lines), 35)
            self.assertTrue(lines[10].startswith("::warning"))
            joined = "".join(l.split("::", 2)[2] for l in lines)
            import base64
            self.assertEqual(base64.b64decode(joined), p.read_bytes())


if __name__ == "__main__":
    unittest.main()
