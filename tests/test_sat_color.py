import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import PIL  # noqa: F401
except ImportError:  # the collect workflows install neither; Tests and sat-ships do
    np = None

if np is not None:
    from PIL import Image

    from sat import color, run
    from tests.test_sat import SITE, FakeClient, RunTest


def speckle(h, w, db, seed=0):
    rng = np.random.default_rng(seed)
    return 10 * np.log10(10 ** (np.asarray(db, float) / 10) * rng.gamma(color.LEE_ENL, 1 / color.LEE_ENL, (h, w)))


@unittest.skipUnless(np is not None, "numpy/Pillow not installed")
class ColorTest(unittest.TestCase):
    def test_lee_smooths_speckle_keeps_nodata_and_mean(self):
        db = speckle(120, 120, -18.0)
        db[0:5, 0:5] = np.nan
        f = color.lee(db)
        self.assertTrue(np.isnan(f[0:5, 0:5]).all())
        self.assertTrue(np.isfinite(f[10:, 10:]).all())
        self.assertLess(np.nanstd(f), 0.7 * np.nanstd(db))
        lin = lambda x: np.nanmean(10 ** (x / 10))
        self.assertAlmostEqual(10 * np.log10(lin(f)), 10 * np.log10(lin(db)), delta=0.3)

    def test_lee_keeps_a_bright_point(self):
        db = speckle(40, 40, -20.0)
        db[20, 20] = 10.0
        self.assertGreater(color.lee(db)[20, 20], 0.0)

    def test_fixed_ranges_not_stretched(self):
        vv = np.array([[-22.0, 0.0, -11.0, np.nan]])
        vh = np.array([[-28.0, -6.0, -26.0, -20.0]])
        rgb = color.false_color(vv, vh, filtered=False)
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[0, 1, 0], 255)
        self.assertIn(rgb[0, 2, 0], (127, 128))
        self.assertEqual(rgb[0, 0, 1], 0)
        self.assertEqual(rgb[0, 1, 1], 255)
        # VV-VH: 6 dB -> 0.75 blue, 6 dB, 15 dB -> 0
        self.assertEqual(rgb[0, 0, 2], round(0.75 * 255))
        self.assertEqual(rgb[0, 2, 2], 0)
        self.assertEqual(tuple(rgb[0, 3]), color.NODATA_RGB)
        # The same values give the same colour in a scene with a different spread.
        rgb2 = color.false_color(np.array([[-40.0, 20.0, -11.0]]), np.array([[-40.0, 10.0, -26.0]]), filtered=False)
        self.assertEqual(tuple(rgb2[0, 2]), tuple(rgb[0, 2]))

    def test_ratio_is_reversed(self):
        vv = np.array([[-10.0, -10.0]])
        vh = np.array([[-13.0, -25.0]])  # ratio 3 dB (blue) and 15 dB (no blue)
        rgb = color.false_color(vv, vh, filtered=False)
        self.assertEqual(rgb[0, 0, 2], 255)
        self.assertEqual(rgb[0, 1, 2], 0)

    def test_render_webp_legend_and_sizes(self):
        h, w = 775, 311
        base = np.full((h, w), -21.0)
        base[:, 220:] = -6.0
        vv, vh = speckle(h, w, base, 1), speckle(h, w, base - 8, 2)
        vv[300:306, 100:130], vh[300:306, 100:130] = 5.0, -4.0
        full, thumb = color.render(vv, vh, "TST label", "Contains modified Copernicus Sentinel data 2026")
        self.assertEqual(full[:4], b"RIFF")
        self.assertEqual(full[8:12], b"WEBP")
        im = Image.open(io.BytesIO(full))
        self.assertEqual(im.format, "WEBP")
        self.assertEqual(im.width, max(w, color.LEGEND_MIN_WIDTH))
        self.assertGreater(im.height, h)  # legend below
        th = Image.open(io.BytesIO(thumb))
        self.assertLessEqual(max(th.size), color.THUMB_MAX)
        self.assertLessEqual(len(full), 150_000)
        self.assertLessEqual(len(thumb), 25_000)
        # The chip area is not covered by the legend: the ship stays bright.
        px = np.asarray(im.convert("RGB"))[303, 115]
        self.assertGreater(int(px[0]), 200)

    def test_settings_match_constants(self):
        s = color.settings()
        self.assertEqual(s["channels"]["R"]["range_db"], list(color.VV_RANGE))
        self.assertEqual(s["channels"]["B"]["range_db"], list(color.RATIO_RANGE))


@unittest.skipUnless(np is not None, "numpy/Pillow not installed")
class RunColorTest(unittest.TestCase):
    def setUp(self):
        self.chip, self.land = RunTest().chip()

    def run_site(self, out, site):
        store = run.Store(out, [site])
        cli = FakeClient(self.chip)
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        s = run.process_site(cli, store, site, self.land.astype(np.uint8), now, None, 100.0, 12, log=lambda *a: None)
        store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                    "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
        return s

    def test_color_image_written_beside_png(self):
        site = {**SITE, "id": "TST"}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.assertEqual(self.run_site(out, site)["counted"], 2)
            rows = run.read_csv(out / "ship_counts.csv")
            r = rows[0]
            self.assertEqual(r["color_image"], "color/TST/v1/2026-10-20T22-30-00Z.webp")
            self.assertEqual(r["color_thumb"], "color/TST/v1/2026-10-20T22-30-00Z.thumb.webp")
            self.assertTrue(r["image"].endswith(".png"))
            for k in ("image", "color_image", "color_thumb"):
                self.assertTrue((out / r[k]).exists(), k)
            with Image.open(out / r["color_image"]) as im:
                self.assertEqual(im.format, "WEBP")
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["display"]["version"], color.COLOR_VERSION)
            self.assertIn("color_image", meta["columns"])

    def test_color_failure_does_not_stop_the_count(self):
        site = {**SITE, "id": "TST"}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run.fcolor, "render", side_effect=RuntimeError("x")):
            out = Path(d)
            self.assertEqual(self.run_site(out, site)["counted"], 2)
            r = run.read_csv(out / "ship_counts.csv")[0]
            self.assertEqual(r["color_image"], "")
            self.assertTrue((out / r["image"]).exists())
            self.assertFalse((out / "color").exists())

    def test_old_rows_without_color_columns_still_load(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            old = [f for f in run.CSV_FIELDS if not f.startswith("color_")]
            run.write_csv(out / "ship_counts.csv", [{"site_id": "TST", "acquired_utc": "2026-10-01T00:00:00Z"}], old)
            store = run.Store(out, [SITE])
            store.save({"run_utc": "2026-10-30T05:17:00Z"})
            text = (out / "ship_counts.csv").read_text().splitlines()
            self.assertTrue(text[0].endswith(",color_image,color_thumb"))
            self.assertTrue(text[1].endswith(",,"))


if __name__ == "__main__":
    unittest.main()
