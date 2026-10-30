import io
import json
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cdse, png, render, run
    from sat.detect import detect, length_class
    from sat.grid import Grid, box_sum, dilate, fill_polygon, label
    from sat.osm_mask import build_mask

SITE = {"id": "TST", "center_lat": 21.2, "center_lon": 110.41, "size_m": 2000, "pixel_m": 10, "enabled": True}


@unittest.skipUnless(np is not None, "numpy not installed")
class PNGTest(unittest.TestCase):
    def test_roundtrip_uint16_all_filters(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 65536, size=(9, 7, 4), dtype=np.uint16)
        for f in range(5):
            data = png.encode(a, filters=[f], depth=16)
            self.assertTrue(np.array_equal(png.decode(data), a), f"filter {f}")
        data = png.encode(a, filters=[0, 1, 2, 3, 4], depth=16)
        self.assertTrue(np.array_equal(png.decode(data), a))

    def test_roundtrip_8bit_and_palette(self):
        rng = np.random.default_rng(2)
        g = rng.integers(0, 256, size=(5, 6), dtype=np.uint8)
        self.assertTrue(np.array_equal(png.decode(png.encode(g, filters=[4, 3])), g))
        p = render.quicklook(np.full((20, 20), -20.0), np.zeros((20, 20), np.uint8), [])
        self.assertEqual(png.decode(p).shape, (20, 20))

    def test_rejects_garbage(self):
        with self.assertRaises(png.PNGError):
            png.decode(b"not a png")


@unittest.skipUnless(np is not None, "numpy not installed")
class GridTest(unittest.TestCase):
    def test_rect_grid(self):
        g = Grid.for_site({"id": "R", "bbox": [110.425, 21.195, 110.455, 21.265], "pixel_m": 10})
        self.assertEqual((g.width, g.height), (311, 775))
        self.assertAlmostEqual(g.pixel_m[0], 10, delta=0.1)
        self.assertAlmostEqual(g.pixel_m[1], 10, delta=0.1)
        with self.assertRaises(ValueError):
            Grid.for_site({"id": "R", "bbox": [110.455, 21.195, 110.425, 21.265], "pixel_m": 10})

    def test_site_grid_is_10m(self):
        g = Grid.for_site({**SITE, "size_m": 8000})
        self.assertEqual((g.width, g.height), (800, 800))
        ry, rx = g.pixel_m
        self.assertAlmostEqual(ry, 10, delta=0.05)
        self.assertAlmostEqual(rx, 10, delta=0.05)
        r, c = g.to_pixel(110.41, 21.2)
        self.assertAlmostEqual(float(r), 400, delta=0.1)
        self.assertAlmostEqual(float(c), 400, delta=0.1)

    def test_label_connectivity(self):
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0]], bool)
        self.assertEqual(label(m, 8)[1], 2)
        self.assertEqual(label(m, 4)[1], 3)
        u = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], bool)
        lab, n = label(u, 4)
        self.assertEqual(n, 1)
        self.assertEqual(lab[0, 0], lab[0, 2])

    def test_box_sum_and_dilate(self):
        a = np.ones((5, 5))
        self.assertEqual(box_sum(a, 1)[2, 2], 9)
        self.assertEqual(box_sum(a, 1)[0, 0], 4)
        m = np.zeros((9, 9), bool)
        m[4, 4] = True
        self.assertEqual(int(dilate(m, 2).sum()), 13)

    def test_fill_polygon(self):
        m = np.zeros((10, 10), bool)
        fill_polygon(m, [2, 2, 6, 6, 2], [2, 6, 6, 2, 2])
        self.assertEqual(int(m.sum()), 16)


def _way(coords, **tags):
    return {"type": "way", "tags": tags, "geometry": [{"lat": la, "lon": lo} for lo, la in coords]}


@unittest.skipUnless(np is not None, "numpy not installed")
class MaskTest(unittest.TestCase):
    def test_coastline_side_island_and_pier(self):
        g = Grid.for_site(SITE)  # 200 x 200
        lon_c = lambda c: g.west + c / g.width * (g.east - g.west)
        lat_r = lambda r: g.north - r / g.height * (g.north - g.south)
        # Coast running north at col 50: land (left) to the west.
        coast = _way([(lon_c(50.3), g.south - 0.01), (lon_c(50.3), g.north + 0.01)], natural="coastline")
        # Counter-clockwise island (land on the left) around rows 90-110, cols 140-160.
        ring = [(lon_c(140), lat_r(110)), (lon_c(160), lat_r(110)), (lon_c(160), lat_r(90)),
                (lon_c(140), lat_r(90)), (lon_c(140), lat_r(110))]
        island = _way(ring, natural="coastline")
        pier = _way([(lon_c(50.5), lat_r(30.5)), (lon_c(80.5), lat_r(30.5))], man_made="pier")
        mask, info = build_mask(g, {"elements": [coast, island, pier]})
        self.assertEqual(mask[100, 10], 1)
        self.assertEqual(mask[100, 100], 0)
        self.assertEqual(mask[100, 150], 1)
        self.assertEqual(mask[30, 70], 2)
        self.assertEqual(mask[60, 70], 0)
        self.assertEqual(info["coastline_ways"], 2)


def _sea(rng, h, w, mean, looks=10):
    return rng.gamma(looks, mean / looks, size=(h, w))


def _ship(arr, r, c, length, width, value):
    arr[r:r + width, c:c + length] = value


@unittest.skipUnless(np is not None, "numpy not installed")
class DetectTest(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(7)
        h = w = 200
        vv = _sea(rng, h, w, 0.01)
        vh = _sea(rng, h, w, 0.002)
        land = np.zeros((h, w), bool)
        land[:, :40] = True
        _ship(vv, 100, 100, 15, 3, 1.0); _ship(vh, 100, 100, 15, 3, 0.1)    # 150 m ship, harbour
        _ship(vv, 50, 40, 6, 2, 1.0); _ship(vh, 50, 40, 6, 2, 0.1)          # 60 m ship at the quay
        _ship(vv, 150, 150, 4, 2, 1.0)                                     # bright in VV only: rejected
        vv[20, 150] = vv[20, 151] = 1.0; vh[20, 150] = vh[20, 151] = 0.1   # 2 px: too small
        to_db = lambda x: 10 * np.log10(x)
        return to_db(vv), to_db(vh), land

    def test_counts_zones_and_lengths(self):
        vv, vh, land = self.scene()
        dets, info = detect(vv, vh, land, (10.0, 10.0))
        self.assertEqual(len(dets), 2, dets)
        by_zone = {d["zone"]: d for d in dets}
        self.assertEqual(set(by_zone), {"harbour", "quay"})
        self.assertAlmostEqual(by_zone["harbour"]["length_m"], 150, delta=20)
        self.assertEqual(by_zone["harbour"]["length_class"], "100_200")
        self.assertEqual(by_zone["quay"]["length_class"], "50_100")
        self.assertGreaterEqual(info["dropped_vh"], 1)
        self.assertGreaterEqual(info["dropped_small"], 1)

    def test_land_and_nodata_ignored(self):
        vv, vh, land = self.scene()
        vv[:, :40] = 10.0  # bright land
        vv[180:, :] = np.nan
        dets, _ = detect(vv, vh, land, (10.0, 10.0))
        self.assertEqual(len(dets), 2)

    def test_length_class_edges(self):
        self.assertEqual(length_class(49.9), "lt50")
        self.assertEqual(length_class(50), "50_100")
        self.assertEqual(length_class(200), "ge200")


class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status, self._body, self.headers = status, body, headers or {}

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_opener(routes):
    calls = []

    def opener(req, timeout=None):
        calls.append(req)
        for prefix, fn in routes:
            if req.full_url.startswith(prefix):
                status, body, headers = fn(req)
                if status != 200:
                    raise urllib.error.HTTPError(req.full_url, status, "err", headers, io.BytesIO(body))
                return FakeResp(status, body, headers)
        raise AssertionError(req.full_url)
    opener.calls = calls
    return opener


def token_ok(req):
    return 200, json.dumps({"access_token": "tok", "expires_in": 600}).encode(), {}


@unittest.skipUnless(np is not None, "numpy not installed")
class ClientTest(unittest.TestCase):
    def test_missing_credentials(self):
        c = cdse.Client("", "", opener=fake_opener([]))
        with self.assertRaises(cdse.CredentialError) as cm:
            c.token()
        self.assertIn("CDSE_SH_CLIENT_SECRET", str(cm.exception))

    def test_invalid_client_message_has_no_secret(self):
        op = fake_opener([(cdse.TOKEN_URL, lambda r: (401, b'{"error":"invalid_client","error_description":"Invalid client credentials"}', {}))])
        c = cdse.Client("my-id", "s3cr3t-value", opener=op)
        with self.assertRaises(cdse.CredentialError) as cm:
            c.token()
        msg = str(cm.exception)
        self.assertIn("期限切れ", msg)
        self.assertNotIn("s3cr3t-value", msg)

    def test_process_records_pu_header(self):
        img = png.encode(np.ones((4, 4, 3), np.uint16) * 30000, depth=16)
        op = fake_opener([(cdse.TOKEN_URL, token_ok),
                          (cdse.PROCESS_URL, lambda r: (200, img, {"x-processingunits-spent": "2.5"}))])
        c = cdse.Client("id", "secret", opener=op)
        t = datetime(2026, 10, 1, tzinfo=timezone.utc)
        data, pu = c.process([0, 0, 1, 1], 4, 4, t, t, with_lia=False)
        self.assertEqual(pu, 2.5)
        self.assertEqual(c.requests, 1)
        self.assertNotIn(b"secret", op.calls[1].data)

    def test_acquisitions_group_slices(self):
        def f(t, plat="sentinel-1a", orbit=60000):
            return {"id": t, "properties": {"datetime": t, "platform": plat, "sat:orbit_state": "ascending",
                                             "sat:absolute_orbit": orbit, "sar:instrument_mode": "IW",
                                             "sar:polarizations": ["VV", "VH"]}}
        feats = [f("2026-10-01T10:30:00Z"), f("2026-10-01T10:30:25Z"), f("2026-10-04T22:10:00Z", "sentinel-1c", 5000)]
        acqs = cdse.acquisitions(feats)
        self.assertEqual(len(acqs), 2)
        self.assertEqual(len(acqs[0]["ids"]), 2)
        self.assertEqual(acqs[0]["relative_orbit"], (60000 - 73) % 175 + 1)
        self.assertIsNone(acqs[1]["relative_orbit"])
        self.assertLess(acqs[0]["t_from"], acqs[0]["time"])
        self.assertGreater(acqs[0]["t_to"], acqs[0]["t_end"])

    def test_pu_estimate(self):
        self.assertAlmostEqual(cdse.estimate_pu(800, 800, 3), 4.883, places=3)


class FakeClient:
    """Catalog with two acquisitions; process returns a synthetic chip."""

    def __init__(self, chip):
        self.chip = chip
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True
        self.processed = []

    def catalog(self, bbox, start, end):
        self.requests += 1
        p = {"platform": "sentinel-1a", "sat:orbit_state": "descending", "sat:absolute_orbit": 61000,
             "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": "a", "properties": {**p, "datetime": "2026-10-20T22:30:00Z"}},
                {"id": "b", "properties": {**p, "datetime": "2026-10-26T22:30:00Z"}}]

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 4.883
        self.processed.append(t0)
        return self.chip, 4.883


@unittest.skipUnless(np is not None, "numpy not installed")
class RunTest(unittest.TestCase):
    def chip(self):
        vv_db, vh_db, land = DetectTest().scene()
        enc = lambda d: np.clip(np.round((d + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv_db), enc(vh_db), np.full(vv_db.shape, 3500.0), np.ones(vv_db.shape)], axis=-1)
        return png.encode(arr.astype(np.uint16), depth=16, filters=[1, 4]), land

    def test_process_store_and_skip(self):
        chip, land = self.chip()
        mask = land.astype(np.uint8)
        site = {**SITE, "id": "TST"}
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            store = run.Store(out)
            cli = FakeClient(chip)
            s = run.process_site(cli, store, site, mask, now, None, 100.0, 12, log=lambda *a: None)
            self.assertEqual(s["lookback_days"], run.FIRST_LOOKBACK_DAYS)
            self.assertEqual(s["counted"], 2)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                        "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
            rows = run.read_csv(out / "ship_counts.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["harbour_candidates"], "1")
            self.assertEqual(rows[0]["quay_candidates"], "1")
            self.assertEqual(rows[0]["confirmed"], "")
            self.assertEqual(rows[0]["incidence_deg"], "35.0")
            self.assertEqual(rows[0]["relative_orbit"], str((61000 - 73) % 175 + 1))
            self.assertTrue((out / rows[0]["image"]).exists())
            gj = json.loads((out / "detections" / "2026-10.geojson").read_text())
            self.assertEqual(len(gj["features"]), 4)
            self.assertEqual(gj["features"][0]["properties"]["status"], "candidate")
            self.assertEqual(gj["features"][0]["properties"]["credibility"], "3")
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["attribution"]["credit"], "Contains modified Copernicus Sentinel data 2026")
            # Second run: nothing new to process.
            store2 = run.Store(out)
            cli2 = FakeClient(chip)
            s2 = run.process_site(cli2, store2, site, mask, now, None, 100.0, 12, log=lambda *a: None)
            self.assertEqual(s2["lookback_days"], run.LOOKBACK_DAYS)
            self.assertEqual(s2["new"], 0)
            self.assertEqual(cli2.processed, [])

    def test_budget_stops_processing(self):
        chip, land = self.chip()
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d))
            cli = FakeClient(chip)
            s = run.process_site(cli, store, SITE, land.astype(np.uint8), datetime(2026, 10, 30, tzinfo=timezone.utc),
                                 None, 5.0, 12, log=lambda *a: None)
            self.assertEqual(s["counted"], 1)
            self.assertTrue(s.get("deferred_budget"))

    def run_main(self, argv, cli, land):
        from unittest import mock
        import contextlib
        with mock.patch.object(run, "load_mask", return_value=(land.astype(np.uint8), {})), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.main(argv, client=cli, sites=[SITE])

    def test_detect_failure_still_saves_usage(self):
        from unittest import mock
        chip, land = self.chip()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with mock.patch.object(run, "detect", side_effect=ValueError("boom")):
                rc = self.run_main(["--out", str(out)], FakeClient(chip), land)
            self.assertEqual(rc, 0)
            usage = run.read_csv(out / "usage.csv")
            self.assertEqual(len(usage), 1)
            self.assertAlmostEqual(float(usage[0]["processing_units"]), 2 * 4.883, places=3)
            st = json.loads((out / "status.json").read_text())
            self.assertEqual([s["result"] for s in st["scenes"]], ["error", "error"])
            self.assertEqual(run.read_csv(out / "ship_counts.csv"), [])

    def test_unexpected_error_saves_then_raises(self):
        from unittest import mock
        chip, land = self.chip()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with mock.patch.object(run.Store, "add", side_effect=RuntimeError("disk")):
                with self.assertRaises(RuntimeError):
                    self.run_main(["--out", str(out)], FakeClient(chip), land)
            usage = run.read_csv(out / "usage.csv")
            self.assertEqual(len(usage), 1)
            self.assertIn("RuntimeError", json.loads((out / "status.json").read_text())["last_run"]["error"])

    def test_catalog_error_updates_status(self):
        chip, land = self.chip()

        class Failing(FakeClient):
            def catalog(self, *a):
                self.requests += 4
                raise cdse.CDSEError("Sentinel Hub の利用制限（HTTP 429）")

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rc = self.run_main(["--out", str(out)], Failing(chip), land)
            self.assertEqual(rc, 1)
            st = json.loads((out / "status.json").read_text())
            self.assertIn("429", st["last_run"]["sites"][0]["error"])
            self.assertEqual(run.read_csv(out / "usage.csv")[0]["requests"], "4")

    def test_probe_counts_published_usage(self):
        chip, land = self.chip()
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            pub = Path(d) / "pub"
            run.write_csv(pub / "usage.csv", [{"run_utc": run.cdse.iso(now), "requests": 1,
                                               "processing_units": run.MONTHLY_PU_CAP, "pu_source": "header", "scenes": 1}],
                          run.USAGE_FIELDS)
            cli = FakeClient(chip)
            self.run_main(["--out", str(pub), "--probe", str(Path(d) / "probe")], cli, land)
            self.assertEqual(cli.processed, [])
            self.assertEqual(len(run.read_csv(pub / "usage.csv")), 1)  # published data untouched

    def sector_site(self):
        g = Grid.for_site(SITE)
        mid = round((g.west + g.east) / 2, 6)
        return {"id": "TSB", "bbox": g.bbox, "pixel_m": 10, "enabled": True, "mask_version": 2,
                "sectors": [{"id": "A", "name": "east", "bbox": [mid, g.south, g.east, g.north]}],
                "sector_rest": {"id": "B", "name": "west"},
                "history": [{"mask_version": 1, "until_utc": "2026-10-30", "note": "old square"}]}

    def test_rect_site_sectors_and_new_series(self):
        chip, land = self.chip()
        site = self.sector_site()
        self.assertEqual((Grid.for_site(site).height, Grid.for_site(site).width), land.shape)
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            # Rows of the old version (v1) for the same scenes are already published.
            run.write_csv(out / "ship_counts.csv", [
                {"site_id": "TSB", "acquired_utc": t, "harbour_candidates": 9, "quay_candidates": 9, "image": "images/TSB/x.png"}
                for t in ("2026-10-20T22:30:00Z", "2026-10-26T22:30:00Z")], run.CSV_FIELDS)
            store = run.Store(out, [site])
            cli = FakeClient(chip)
            s = run.process_site(cli, store, site, land.astype(np.uint8), now, None, 100.0, 12, log=lambda *a: None)
            self.assertEqual(s["lookback_days"], run.FIRST_LOOKBACK_DAYS)  # a new series for v2
            self.assertEqual(s["counted"], 2)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                        "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
            rows = run.read_csv(out / "ship_counts.csv")
            self.assertEqual(len(rows), 4)  # v1 rows kept
            v1 = [r for r in rows if r["mask_version"] == ""]
            v2 = [r for r in rows if r["mask_version"] == "TSB-land-v2"]
            self.assertEqual((len(v1), len(v2)), (2, 2))
            self.assertEqual(v1[0]["harbour_candidates"], "9")
            r = v2[0]
            # The 150 m ship (cols 100-114) is in the east sector A, the quay ship (cols 40-45) in B.
            self.assertEqual((r["sector_a_harbour"], r["sector_a_quay"], r["sector_b_harbour"], r["sector_b_quay"]),
                             ("1", "0", "0", "1"))
            self.assertTrue(r["image"].startswith("images/TSB/v2/"))
            self.assertTrue((out / r["image"]).exists())
            gj = json.loads((out / "detections" / "2026-10.geojson").read_text())
            self.assertEqual(sorted(f["properties"]["sector"] for f in gj["features"]), ["A", "A", "B", "B"])
            st = json.loads((out / "status.json").read_text())
            self.assertEqual({x["mask_version"] for x in st["scenes"]}, {"TSB-land-v2"})
            self.assertEqual(st["series_breaks"][0]["from_mask_version"], "TSB-land-v1")
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["series_breaks"][0]["to_mask_version"], "TSB-land-v2")
            self.assertEqual([x["id"] for x in meta["sites"][0]["sectors"]], ["A", "B"])
            # Next run: v2 scenes are done, v1 rows do not count as done for v2 and are not touched.
            s2 = run.process_site(FakeClient(chip), run.Store(out, [site]), site, land.astype(np.uint8), now, None,
                                  100.0, 12, log=lambda *a: None)
            self.assertEqual((s2["lookback_days"], s2["new"]), (run.LOOKBACK_DAYS, 0))

    def test_mask_size_must_match_site(self):
        chip, land = self.chip()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run.process_site(FakeClient(chip), run.Store(Path(d), [SITE]), SITE, land[:, :150].astype(np.uint8),
                                 datetime(2026, 10, 30, tzinfo=timezone.utc), None, 100.0, 12, log=lambda *a: None)

    def test_real_sites_file(self):
        sites = run.load_sites()
        zjg = next(x for x in sites if x["id"] == "ZJG")
        g = Grid.for_site(zjg)
        self.assertEqual(g.bbox, [110.425, 21.195, 110.455, 21.265])
        self.assertAlmostEqual(g.area_km2(), 24.1, places=1)
        self.assertEqual((g.width, g.height), (311, 775))
        self.assertEqual(run.mask_tag(zjg), "ZJG-land-v2")
        sec = run.site_sectors(zjg)
        self.assertEqual(run.sector_of(sec, 110.437, 21.22), "A")   # Mazhang piers
        self.assertEqual(run.sector_of(sec, 110.427, 21.22), "B")   # channel
        self.assertEqual(run.sector_of(sec, 110.440, 21.197), "B")  # south of sector A
        # The committed land mask of the current version covers exactly this extent.
        mask, meta = run.load_mask("ZJG", run.mask_version(zjg))
        self.assertEqual(mask.shape, (g.height, g.width))
        self.assertEqual(meta["bbox"], g.bbox)
        self.assertEqual(meta["version"], "ZJG-land-v2")

    def test_month_usage(self):
        u = [{"run_utc": "2026-10-01T05:17:00Z", "requests": "3", "processing_units": "9.7"},
             {"run_utc": "2026-09-30T05:17:00Z", "requests": "2", "processing_units": "4.9"}]
        self.assertEqual(run.month_usage(u, "2026-10"), (9.7, 3))


if __name__ == "__main__":
    unittest.main()


class FollowupTest(unittest.TestCase):
    def test_probe_pu_annotation_sums_the_probe_usage(self):
        from sat import annotate
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "usage.csv"
            p.write_text("run_utc,requests,processing_units,pu_source,scenes\n"
                         "2026-10-30T05:17:00Z,4,12.5,header,2\n2026-10-30T06:00:00Z,1,0.25,header,0\n", encoding="utf-8")
            text = annotate.pu_summary(p)
        self.assertIn("12.750 PU", text)
        self.assertIn("リクエスト 5", text)

    @unittest.skipUnless(np is not None, "numpy not installed")  # sat.run imports numpy
    def test_meta_describes_one_row_per_mask_version(self):
        from sat import run
        self.assertTrue(run.build_meta([])["files"]["ship_counts.csv"].startswith("1 行＝1 撮影×1 地点×1 範囲版（mask_version）"))
