"""chips-1: the chip index for the site (latest 10 scenes per site, candidate boxes as data)."""
import json
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import chips, run
    from sat.grid import Grid
    from tests.test_sat_multisite import NEW, FakeClient, NewSiteRunTest

SITE = {"id": "AAA", "name": "試験港", "bbox": [120.0, 30.0, 120.03, 30.03], "pixel_m": 10, "mask_version": 1, "enabled": True,
        "layout": "per_site"}


def fake_webp(w, h):
    """The smallest lossy-WebP header chips.webp_size reads (the bytes are not decodable)."""
    return b"RIFF" + struct.pack("<I", 22) + b"WEBP" + b"VP8 " + struct.pack("<I", 10) + b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", w, h)


def feature(site_id, acq, n, lon, lat, length=80, width=20, tag="AAA-land-v1"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"site_id": site_id, "acquired_utc": acq, "n": n, "zone": "harbour", "sector": None,
                           "length_m": length, "width_m": width, "length_class": "50_100", "contrast_vv_db": 20.0,
                           "vh_ratio_db": 10.0, "credibility": "3", "reliability": "A", "status": "candidate",
                           "mask_version": tag}}


@unittest.skipUnless(np is not None, "numpy not installed")
class ChipsTest(unittest.TestCase):
    def make(self, out: Path, n: int, site=SITE):
        g = Grid.for_site(site)
        rows, feats = [], []
        for k in range(n):
            acq = f"2026-10-{k + 1:02d}T10:48:45Z"
            st = acq.replace(":", "-")
            (out / "color" / site["id"] / "v1").mkdir(parents=True, exist_ok=True)
            ci, ct = f"color/{site['id']}/v1/{st}.webp", f"color/{site['id']}/v1/{st}.thumb.webp"
            (out / ci).write_bytes(fake_webp(max(g.width, 300), g.height + 76))
            (out / ct).write_bytes(fake_webp(200, 220))
            rows.append({"site_id": site["id"], "acquired_utc": acq, "platform": "sentinel-1a", "orbit_direction": "ascending",
                         "relative_orbit": "10", "mask_version": f"{site['id']}-land-v1", "color_image": ci, "color_thumb": ct,
                         "retrieved_utc": "2026-11-01T05:00:00Z"})
            # one candidate in the middle, one on the very corner (the box must be clipped to the chip)
            feats += [feature(site["id"], acq, 1, 120.015, 30.015, tag=f"{site['id']}-land-v1"),
                      feature(site["id"], acq, 2, 120.0001, 30.0299, length=300, tag=f"{site['id']}-land-v1")]
        return rows, feats

    def call(self, out, rows, feats, keep=10):
        return chips.update(out, [SITE], rows, lambda s, m: feats, lambda s: f"{s['id']}-land-v1", run.row_mask_tag, keep)

    def test_keeps_latest_10_and_deletes_older(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows, feats = self.make(out, 13)
            r = self.call(out, rows, feats)
            self.assertEqual((r["scenes"], r["removed"]), (10, 3))
            idx = json.loads((out / "chips" / "index.json").read_text(encoding="utf-8"))
            scenes = idx["sites"][0]["scenes"]
            self.assertEqual([s["acquired_utc"][:10] for s in scenes], [f"2026-10-{k:02d}" for k in range(13, 3, -1)])
            for k in range(1, 4):  # the three oldest: image, thumbnail and chip file gone, the columns blank
                st = f"2026-10-{k:02d}T10-48-45Z"
                self.assertFalse((out / f"color/AAA/v1/{st}.webp").exists())
                self.assertFalse((out / f"color/AAA/v1/{st}.thumb.webp").exists())
                self.assertFalse((out / f"chips/AAA/{st}.json").exists())
                self.assertEqual(rows[k - 1]["color_image"], "")
            for s in scenes:  # everything the index points to exists
                for key in ("chip", "image", "thumb"):
                    self.assertTrue((out / s[key]).exists(), s[key])
            self.assertEqual(len(list((out / "chips" / "AAA").glob("*.json"))), 10)
            # a second run changes nothing
            before = {p: p.read_bytes() for p in (out / "chips").rglob("*.json")}
            r2 = self.call(out, rows, feats)
            self.assertEqual((r2["removed"], r2["cleared"]), (0, 0))
            self.assertEqual(before, {p: p.read_bytes() for p in (out / "chips").rglob("*.json")})

    def test_boxes_inside_the_image_and_credit(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows, feats = self.make(out, 2)
            self.call(out, rows, feats)
            idx = json.loads((out / "chips" / "index.json").read_text(encoding="utf-8"))
            self.assertIn("Contains modified Copernicus Sentinel data 2026", idx["credit"])
            doc = json.loads((out / idx["sites"][0]["scenes"][0]["chip"]).read_text(encoding="utf-8"))
            self.assertIn("Contains modified Copernicus Sentinel data 2026", doc["credit"])
            W, H = doc["image_px"]
            cw, ch = doc["chip_px"]
            self.assertLessEqual(cw, W)
            self.assertLess(ch, H)  # the legend is below the chip
            self.assertEqual(len(doc["candidates"]), 2)
            w, s, e, n = doc["bbox"]
            for c in doc["candidates"]:
                x0, y0, x1, y1 = c["box_px"]
                self.assertTrue(0 <= x0 < x1 <= cw and 0 <= y0 < y1 <= ch, c["box_px"])
                bw, bs, be, bn = c["box_lonlat"]
                self.assertTrue(w <= bw < be <= e and s <= bs < bn <= n, c["box_lonlat"])
                self.assertEqual(c["credibility"], "3")
            mid = doc["candidates"][0]["box_px"]  # centre candidate: the box is centred on it, about 80 m + padding wide
            self.assertAlmostEqual((mid[0] + mid[2]) / 2, cw / 2, delta=1.5)
            self.assertAlmostEqual((mid[1] + mid[3]) / 2, ch / 2, delta=1.5)

    def test_only_the_site_and_scene_features(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows, feats = self.make(out, 2)
            feats += [feature("BBB", rows[1]["acquired_utc"], 9, 120.01, 30.01)]
            self.call(out, rows, feats)
            doc = json.loads((out / "chips" / "AAA" / "2026-10-02T10-48-45Z.json").read_text(encoding="utf-8"))
            self.assertEqual([c["n"] for c in doc["candidates"]], [1, 2])

    def test_unsafe_path_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "keep.txt").write_text("x")
            chips._remove(out, "../keep.txt")
            chips._remove(out, "ship_counts.csv")
            self.assertTrue((out / "keep.txt").exists())

    def test_webp_size_reads_real_files(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        import io
        for kw in ({"quality": 70}, {"lossless": True}):
            buf = io.BytesIO()
            Image.new("RGB", (37, 53), (1, 2, 3)).save(buf, "WEBP", **kw)
            self.assertEqual(chips.webp_size(buf.getvalue()), (37, 53))

    def test_render_capped(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow not installed")
        from sat import color
        rng = np.random.default_rng(1)
        vv = rng.normal(-12, 6, (700, 700))
        vh = rng.normal(-19, 6, (700, 700))
        full, _ = color.render(vv, vh, "x", "c")
        capped, _ = color.render_capped(vv, vh, "x", "c", max_bytes=len(full) // 2)
        self.assertLess(len(capped), len(full))             # a larger-than-limit image is saved again at a lower quality
        same, _ = color.render_capped(vv[:50, :50], vh[:50, :50], "x", "c")
        self.assertEqual(same, color.render(vv[:50, :50], vh[:50, :50], "x", "c")[0])  # under the limit: unchanged

    def test_end_to_end_new_site(self):
        """A site run: colour image, chip file and index come out of Store.save, with no PNG and no extra request."""
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = NewSiteRunTest.setup_dirs(None, d, NEW)
            out = d / "out"
            store = run.Store(out, [NEW])
            cli = FakeClient(chip)
            run.process_site(cli, store, NEW, mask, now, None, 100.0, 12, log=lambda *a: None, lines_dir=lines_dir, baseline_dir=bl_dir)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests, "processing_units": cli.processing_units,
                        "pu_source": "header", "scenes": 2})
            idx = json.loads((out / "chips" / "index.json").read_text(encoding="utf-8"))
            sc = idx["sites"][0]["scenes"]
            self.assertEqual(len(sc), 2)
            doc = json.loads((out / sc[0]["chip"]).read_text(encoding="utf-8"))
            self.assertTrue((out / doc["image"]).exists())
            self.assertGreater(len(doc["candidates"]), 0)
            self.assertEqual(doc["image_px"][1] > doc["chip_px"][1], True)
            self.assertFalse((out / "images").exists())
            self.assertLess((out / doc["image"]).stat().st_size, 300_000)
            for c in doc["candidates"]:
                x0, y0, x1, y1 = c["box_px"]
                self.assertTrue(0 <= x0 < x1 <= doc["chip_px"][0] and 0 <= y0 < y1 <= doc["chip_px"][1])


if __name__ == "__main__":
    unittest.main()
