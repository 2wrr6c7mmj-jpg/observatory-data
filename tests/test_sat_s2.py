import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import png, run, s2
    from sat.grid import Grid

SITE = {"id": "TS2", "bbox": [110.0, 21.0, 110.005, 21.008], "pixel_m": 10, "mask_version": 2, "enabled": True,
        "sectors": [{"id": "A", "name": "北", "bbox": [110.0, 21.004, 110.005, 21.008]}],
        "sector_rest": {"id": "B", "name": "南"}, "s2": {"enabled": True}}


def tar_of(parts: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in parts.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class FakeS2:
    """Days: {date: (tile cloud %, cloud rows from the south (A is the north half))}."""

    def __init__(self, days):
        self.days = days
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True
        self.calls = []

    def catalog(self, bbox, start, end, limit=100, collection="sentinel-1-grd"):
        self.requests += 1
        assert collection == "sentinel-2-l2a"
        out = []
        for d, (cc, _rows) in self.days.items():
            for k in range(2):  # two tiles of the same datatake
                out.append({"id": f"{d}-{k}", "properties": {"datetime": f"{d}T03:10:{20 + k}Z", "eo:cloud_cover": cc + k,
                                                              "platform": "sentinel-2b"}})
        return out

    def scl(self, date, h, w):
        a = np.full((h, w), 6, np.uint8)  # water
        frac = self.days[date][1]
        n = int(round(frac * h))
        if n:
            a[h - n:, :] = 9  # clouds over the southern rows
        return a

    def process_raw(self, payload, accept, input_bands):
        self.requests += 1
        date = payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10]
        w, h = payload["output"]["width"], payload["output"]["height"]
        scl = self.scl(date, h, w)
        if accept == "image/png":
            self.calls.append(("pre", date))
            pu = 0.03
            data = png.encode(scl)
        else:
            self.calls.append(("full", date))
            pu = 1.53
            refl = np.zeros((h, w, 4), np.uint16)
            refl[..., 0], refl[..., 1], refl[..., 2], refl[..., 3] = 600, 800, 700, 300
            data = tar_of({"refl.png": png.encode(refl, depth=16), "scl.png": png.encode(scl)})
        self.processing_units += pu
        return data, pu


@unittest.skipUnless(np is not None, "numpy not installed")
class S2Test(unittest.TestCase):
    NOW = datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)

    def test_day_groups(self):
        g = s2.day_groups([{"properties": {"datetime": "2026-10-20T03:10:21Z", "eo:cloud_cover": 40, "platform": "sentinel-2b"}},
                           {"properties": {"datetime": "2026-10-20T03:10:19Z", "eo:cloud_cover": 12, "platform": "sentinel-2b"}},
                           {"properties": {"datetime": "2026-10-25T03:10:19Z"}}])
        self.assertEqual([d["date"] for d in g], ["2026-10-20", "2026-10-25"])
        self.assertEqual(g[0]["acquired_utc"], "2026-10-20T03:10:19Z")
        self.assertEqual(g[0]["tile_cloud_pct"], 12.0)
        self.assertIsNone(g[1]["tile_cloud_pct"])

    def test_cloud_shares_count_cloud_and_shadow(self):
        scl = np.full((10, 10), 6, np.uint8)
        scl[0, :] = 8
        scl[1, :5] = 3
        scl[9, :] = 255  # no data
        in_a = np.zeros((10, 10), bool)
        in_a[:5] = True
        c = s2.cloud_shares(scl, in_a)
        self.assertAlmostEqual(c["valid"], 0.9)
        self.assertAlmostEqual(c["cloud"], 15 / 90, places=3)
        self.assertAlmostEqual(c["cloud_a"], 0.3)

    def test_natural_color(self):
        refl = np.zeros((1, 3, 4), np.uint16)
        refl[0, 0] = [0, 0, 3000, 0]        # red 0.3 -> 255
        refl[0, 1] = [600, 600, 600, 0]     # grey 0.06 -> 0.2 ** (1/2.2)
        refl[0, 2] = [9000, 9000, 9000, 0]  # bright cloud: clipped
        rgb = s2.natural_color(refl)
        self.assertEqual(rgb[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(int(rgb[0, 1, 0]), round(0.2 ** (1 / 2.2) * 255))
        self.assertEqual(rgb[0, 2].tolist(), [255, 255, 255])

    def test_clear_days_get_an_image_and_are_not_fetched_again(self):
        days = {"2026-10-21": (5.0, 0.0),     # clear
                "2026-10-23": (30.0, 0.08),   # 8 % cloud in the south (B): clear for the site and A
                "2026-10-25": (20.0, 0.6),    # cloud over A too: the 60 m pre-check stops it
                "2026-10-27": (95.0, 0.0),    # the tile is cloudy: no pixel request
                "2026-10-29": (10.0, 0.14)}   # passes the pre-check, fails the final 10 % limit
        cli = FakeS2(days)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            r = s2.update_site(cli, out, SITE, self.NOW, 10, log=lambda *a: None)
            self.assertEqual((r["clear"], r["cloudy"], r["skipped"]), (2, 2, 1))
            rows = {x["date"]: x for x in s2.read_rows(out / "s2_scenes.csv")}
            self.assertEqual({k: v["result"] for k, v in rows.items()},
                             {"2026-10-21": "clear", "2026-10-23": "clear", "2026-10-25": "pre_cloudy",
                              "2026-10-27": "tile_cloudy", "2026-10-29": "cloudy"})
            self.assertNotIn(("pre", "2026-10-27"), cli.calls)
            self.assertNotIn(("full", "2026-10-25"), cli.calls)
            c = rows["2026-10-21"]
            self.assertEqual(c["acquired_utc"], "2026-10-21T03:10:20Z")
            self.assertEqual(c["image"], "s2/TS2/v2/2026-10-21T03-10-20Z.webp")
            self.assertTrue((out / c["image"]).exists() and (out / c["thumb"]).exists())
            self.assertEqual(float(c["processing_units"]), 1.56)
            self.assertEqual(rows["2026-10-29"]["image"], "")
            self.assertAlmostEqual(r["processing_units"], 3 * 1.53 + 4 * 0.03, places=3)
            from PIL import Image
            with Image.open(out / c["image"]) as im:
                self.assertGreater(im.height, Grid.for_site(SITE).height)  # the strip with the credit below
            # the next run: nothing again
            cli2 = FakeS2(days)
            r2 = s2.update_site(cli2, out, SITE, self.NOW, 10, log=lambda *a: None)
            self.assertEqual((r2["days"], cli2.calls), (0, []))

    def test_budget_defers(self):
        cli = FakeS2({"2026-10-21": (5.0, 0.0), "2026-10-23": (5.0, 0.0)})
        with tempfile.TemporaryDirectory() as d:
            r = s2.update_site(cli, Path(d), SITE, self.NOW, 10, budget=2.0, log=lambda *a: None)
            self.assertEqual((r["clear"], r["deferred"]), (1, 1))

    def test_daily_run_records_s2_pu(self):
        cli = FakeS2({"2026-10-21": (5.0, 0.0)})
        cli.process = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no S1 scene"))
        orig = cli.catalog

        def catalog(bbox, start, end, limit=100, collection="sentinel-1-grd"):
            if collection == "sentinel-1-grd":
                cli.requests += 1
                return []
            return orig(bbox, start, end, limit, collection)
        cli.catalog = catalog
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            masks = d / "masks"
            masks.mkdir()
            g = Grid.for_site(SITE)
            from sat.osm_mask import save_mask
            mask = np.zeros((g.height, g.width), np.uint8)
            mask[:, :5] = 1
            save_mask(SITE, g, mask, {}, {"elements": []}, 2, masks)
            rc = run.main(["--out", str(d / "out"), "--mask-dir", str(masks)], client=cli, sites=[SITE])
            self.assertEqual(rc, 0)
            u = run.read_csv(d / "out" / "usage.csv")[-1]
            self.assertAlmostEqual(float(u["s2_processing_units"]), 1.56, places=3)
            self.assertAlmostEqual(float(u["processing_units"]), 1.56, places=3)
            meta = json.loads((d / "out" / "meta.json").read_text())
            self.assertEqual(meta["s2"]["version"], "s2nc-1")
            self.assertIn("Contains modified Copernicus Sentinel data 2026", meta["s2"]["credit"])
            self.assertEqual(s2.read_rows(d / "out" / "s2_scenes.csv")[0]["result"], "clear")

    def _s1_none(self, cli):
        cli.process = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no S1 scene"))
        orig = cli.catalog
        seen = []

        def catalog(bbox, start, end, limit=100, collection="sentinel-1-grd"):
            seen.append(collection)
            if collection == "sentinel-1-grd":
                cli.requests += 1
                return []
            return orig(bbox, start, end, limit, collection)
        cli.catalog = catalog
        return seen

    def _masks(self, d):
        masks = d / "masks"
        masks.mkdir()
        g = Grid.for_site(SITE)
        from sat.osm_mask import save_mask
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, :5] = 1
        save_mask(SITE, g, mask, {}, {"elements": []}, 2, masks)
        return masks

    def test_probe_does_not_fetch_s2(self):
        cli = FakeS2({"2026-10-21": (5.0, 0.0)})
        seen = self._s1_none(cli)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            masks = self._masks(d)
            (d / "data").mkdir()
            rc = run.main(["--out", str(d / "data"), "--mask-dir", str(masks), "--probe", str(d / "probe")],
                          client=cli, sites=[SITE])
            self.assertEqual(rc, 0)
            self.assertEqual(cli.calls, [])
            self.assertNotIn("sentinel-2-l2a", seen)
            self.assertFalse((d / "probe" / "out" / "s2_scenes.csv").exists())

    def test_standalone_skips_settings_only_sites_and_keeps_caps(self):
        """python -m sat.s2 (the backfill probe): a daily false site only with --site, and at most its pu_month_cap."""
        days = {"2026-10-21": (5.0, 0.0), "2026-10-23": (5.0, 0.0), "2026-10-25": (5.0, 0.0)}
        off = {**SITE, "id": "OFF", "daily": False, "pu_month_cap": 2}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            cli = FakeS2(days)
            self.assertEqual(s2.main(["--out", str(out), "--days", "365"], client=cli, sites=[off]), 0)
            self.assertEqual((cli.requests, cli.calls), (0, []))  # not even the catalog
            self.assertFalse((out / "s2_scenes.csv").exists())
            # named with --site: fetched, but only up to its own cap (2 PU: one clear day of 1.56 PU)
            cli = FakeS2(days)
            self.assertEqual(s2.main(["--out", str(out), "--days", "365", "--site", "OFF"], client=cli, sites=[off]), 0)
            self.assertEqual([c for c in cli.calls if c[0] == "full"], [("full", "2026-10-21")])
            # the next run of the same month: the cap is spent, no request at all
            cli = FakeS2(days)
            s2.main(["--out", str(out), "--days", "365", "--site", "OFF"], client=cli, sites=[off])
            self.assertEqual((cli.requests, cli.calls), (0, []))
            # a daily site without a cap (Zhanjiang) as before: every day within --max-pu
            cli = FakeS2(days)
            s2.main(["--out", str(out), "--days", "365"], client=cli, sites=[{**SITE, "daily": True}])
            self.assertEqual(len([c for c in cli.calls if c[0] == "full"]), 3)

    def test_errors_are_retried_at_most_twice(self):
        cli = FakeS2({"2026-10-21": (5.0, 0.0)})
        cli.process_raw = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken response"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for n in (1, 2):
                r = s2.update_site(cli, out, SITE, self.NOW, 10, log=lambda *a: None)
                self.assertEqual(r["errors"], 1)
                row = s2.read_rows(out / "s2_scenes.csv")[0]
                self.assertEqual((row["result"], row["attempts"]), ("error", str(n)))
            r = s2.update_site(cli, out, SITE, self.NOW, 10, log=lambda *a: None)
            self.assertEqual(r["days"], 0)  # not tried a third time


if __name__ == "__main__":
    unittest.main()
