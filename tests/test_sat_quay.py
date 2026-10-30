import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, cdse, png, quay, quay_lines, run
    from sat.grid import Grid

SITE = {"id": "TQY", "bbox": [110.0, 21.0, 110.005, 21.008], "pixel_m": 10, "mask_version": 1, "enabled": True,
        "sectors": [{"id": "A", "name": "北", "bbox": [110.0, 21.004, 110.005, 21.008]}],
        "sector_rest": {"id": "B", "name": "南"},
        "baseline": {"orbits": [157], "min_scenes": 24}}


def scene_parts():
    """89 x 52 px: land west of col 10, a pier east along row 30 (cols 10-30), and the lines."""
    g = Grid.for_site(SITE)
    h, w = g.height, g.width
    lon = lambda c: g.west + c / g.width * (g.east - g.west)
    lat = lambda r: g.north - r / g.height * (g.north - g.south)
    mask = np.zeros((h, w), np.uint8)
    mask[:, :10] = 1
    mask[29:32, 10:31] = 2
    lines = {"ways": [
        {"id": 1, "kind": "coastline", "closed": False, "coords": [[lon(10), lat(h + 1)], [lon(10), lat(-1)]]},
        {"id": 2, "kind": "pier", "closed": False, "coords": [[lon(10), lat(30.5)], [lon(30.5), lat(30.5)]]},
    ]}
    clean_vv = np.where(mask == 1, 5.0, -20.0)
    clean_vh = np.where(mask == 1, -5.0, -26.0)
    base = {"vv_p10": baseline.smooth_db(clean_vv), "vh_p10": baseline.smooth_db(clean_vh)}
    return g, mask, lines, clean_vv, clean_vh, base


def ships(clean_vv, clean_vh, raft=False, weak=False):
    rng = np.random.default_rng(0)
    vv = clean_vv + rng.normal(0, 0.5, clean_vv.shape)
    vh = clean_vh + rng.normal(0, 0.5, clean_vh.shape)
    vv[27:29, 14:23] = 5.0   # 90 m along the pier's north side
    vh[27:29, 14:23] = -5.0
    vv[50:61, 11:13] = 5.0   # 110 m along the coast (rows 50-60)
    vh[50:61, 11:13] = -5.0
    if raft:                 # a second ship outboard of the coastal one
        vv[50:61, 14:16] = 5.0
        vh[50:61, 14:16] = -5.0
    if weak:                 # a faint change along the coast: below the seed level
        vv[70:78, 10:12] = -12.5
        vh[70:78, 10:12] = -20.0
    return vv, vh


@unittest.skipUnless(np is not None, "numpy not installed")
class HelpersTest(unittest.TestCase):
    def test_runs_join_gaps_and_drop_short(self):
        f = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
        # one-sample gaps (10 m) are joined, two-sample gaps (20 m) are not; runs under 30 m dropped
        self.assertEqual(quay.runs(f, 10, 20, 30), [(0, 5), (7, 12)])
        self.assertEqual(quay.runs(np.array([1, 1, 0, 0, 0], bool), 10, 20, 30), [])
        self.assertEqual(quay.runs(np.zeros(4, bool), 10, 20, 30), [])

    def test_runs_wrap_on_closed_lines(self):
        f = np.array([1, 1, 0, 0, 0, 0, 1, 1], bool)
        self.assertEqual(quay.runs(f, 10, 20, 30, closed=False), [])
        r = quay.runs(f, 10, 20, 30, closed=True)
        self.assertEqual(len(r), 1)
        a, b = r[0]
        self.assertEqual(b - a, 4)
        self.assertEqual(sorted(np.arange(a, b) % 8), [0, 1, 6, 7])

    def test_pieces_split_on_valleys(self):
        two = np.array([20, 22, 21, 20, 12, 11, 19, 21, 22, 20], float)
        self.assertEqual(quay.pieces(two, 3, 3), 2)
        self.assertEqual(quay.pieces(two, 12, 3), 1)      # valley not deep enough
        self.assertEqual(quay.pieces(two, 3, 6), 1)       # pieces would be too short
        self.assertEqual(quay.pieces(np.array([20, 18, 20, 18, 20.0]), 3, 1), 1)

    def test_cluster(self):
        segs = [{"lat": 21.0, "line": [[110.0, 21.0], [110.0, 21.0005]]},
                {"lat": 21.0, "line": [[110.0004, 21.0], [110.0004, 21.0005]]},   # ~42 m east
                {"lat": 21.0, "line": [[110.002, 21.0], [110.002, 21.0005]]}]     # ~200 m east
        self.assertEqual(quay.cluster(segs, 60), [[0, 1], [2]])
        self.assertEqual(quay.cluster(segs, 30), [[0], [1], [2]])
        self.assertEqual(quay.cluster([], 60), [])


@unittest.skipUnless(np is not None, "numpy not installed")
class CountTest(unittest.TestCase):
    def setUp(self):
        self.g, self.mask, self.lines, self.cvv, self.cvh, self.base = scene_parts()
        self.geo = quay.Geometry(self.g, self.mask, self.lines)

    def test_sides_follow_the_land_mask(self):
        kinds = sorted((p["kind"], p["side"]) for p in self.geo.profiles)
        self.assertEqual([k for k, _ in kinds].count("coastline"), 1)  # the land side is not water
        self.assertEqual([k for k, _ in kinds].count("pier"), 2)       # both sides of a pier
        pier = [p for p in self.geo.profiles if p["kind"] == "pier"][0]
        self.assertGreaterEqual(float(pier["edge_m"].min()), 5.0)       # bands start at the deck edge

    def test_empty_scene_counts_nothing(self):
        rng = np.random.default_rng(1)
        q = quay.count(self.geo, self.cvv + rng.normal(0, 0.5, self.cvv.shape),
                       self.cvh + rng.normal(0, 0.5, self.cvh.shape), self.base)
        self.assertEqual((q["min"], q["max"], q["segments"]), (0, 0, []))
        self.assertGreater(q["coverage"], 0.9)

    def test_two_ships_alongside(self):
        vv, vh = ships(self.cvv, self.cvh, weak=True)
        q = quay.count(self.geo, vv, vh, self.base, lambda lo, la: "A" if la >= 21.004 else "B")
        segs = sorted(q["segments"], key=lambda s: s["kind"])
        self.assertEqual([s["kind"] for s in segs], ["coastline", "pier"])  # the weak change is left out
        coast, pier = segs
        self.assertEqual(coast["length_m"], 130)        # 110 m ship + about 20 m of 3x3 spread
        self.assertEqual(coast["est_length_m"], 110)
        self.assertEqual((coast["min"], coast["max"]), (1, 2))  # floor(130 / 60) = 2
        self.assertEqual(pier["length_m"], 110)
        self.assertEqual((pier["min"], pier["max"]), (1, 1))
        self.assertFalse(coast["rafted"] or pier["rafted"])
        self.assertEqual((q["min"], q["max"]), (2, 3))
        self.assertEqual(pier["sector"], "A")
        self.assertEqual(coast["sector"], "B")
        self.assertEqual(q["by_sector"], {"A": [1, 1], "B": [1, 2]})

    def test_rafted_ship_raises_the_maximum(self):
        vv, vh = ships(self.cvv, self.cvh, raft=True)
        q = quay.count(self.geo, vv, vh, self.base)
        coast = [s for s in q["segments"] if s["kind"] == "coastline"][0]
        self.assertTrue(coast["rafted"])
        self.assertEqual((coast["min"], coast["max"]), (1, 3))

    def test_wind_without_ships_is_not_counted(self):
        rng = np.random.default_rng(2)
        vv = np.where(self.mask == 1, 5.0, -13.0) + rng.normal(0, 0.5, self.cvv.shape)  # water +7 dB everywhere
        vh = np.where(self.mask == 1, -5.0, -21.0) + rng.normal(0, 0.5, self.cvh.shape)
        q = quay.count(self.geo, vv, vh, self.base)
        self.assertEqual(q["segments"], [])


class FakeClient:
    def __init__(self, chip, orbits=(157, 84)):
        self.chip = chip
        self.orbits = orbits
        self.requests = 0
        self.processing_units = 0.0
        self.pu_reported = True
        self.processed = []

    def catalog(self, bbox, start, end):
        self.requests += 1
        p = {"platform": "sentinel-1c", "sat:orbit_state": "ascending",
             "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
        return [{"id": f"s{k}", "properties": {**p, "sat:relative_orbit": o, "datetime": f"2026-10-{20 + k}T10:48:45Z"}}
                for k, o in enumerate(self.orbits)]

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 1.226
        self.processed.append(t0)
        return self.chip, 1.226


@unittest.skipUnless(np is not None, "numpy not installed")
class RunQuayTest(unittest.TestCase):
    def setup_dirs(self, d: Path):
        g, mask, lines, cvv, cvh, base = scene_parts()
        lines_dir, bl_dir = d / "masks", d / "baselines"
        lines_dir.mkdir()
        doc = {"version": "TQY-quay-v1", **lines}
        quay_lines.path("TQY", 1, lines_dir).write_text(json.dumps(doc), encoding="utf-8")
        layers = {"vv_p10": base["vv_p10"], "vv_p90": base["vv_p10"] + 2, "vh_p10": base["vh_p10"],
                  "n_valid": np.full(mask.shape, 30)}
        scenes = [{"acquired_utc": f"2026-09-{k + 1:02d}T10:48:45Z", "used": True} for k in range(30)]
        baseline.save(SITE, g, 157, layers, scenes, baseline.site_config(SITE), mask, bl_dir, "2026-10-30T00:00:00Z")
        vv, vh = ships(cvv, cvh)
        enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 4500.0), np.ones(vv.shape)], -1).astype(np.uint16)
        return mask, png.encode(arr, depth=16, filters=[1]), lines_dir, bl_dir

    def test_quay_rows_change_image_and_no_refetch(self):
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = self.setup_dirs(d)
            out = d / "out"
            store = run.Store(out, [SITE])
            cli = FakeClient(chip)
            s = run.process_site(cli, store, SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                                 lines_dir=lines_dir, baseline_dir=bl_dir)
            self.assertEqual(s["counted"], 2)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                        "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
            rows = run.read_csv(out / "quay_counts.csv")
            self.assertEqual(len(rows), 1)  # orbit 84 has no baseline: no quay row (orbits are not mixed)
            r = rows[0]
            self.assertEqual(r["relative_orbit"], "157")
            self.assertEqual(r["baseline_version"], "ro157-bl-1")
            self.assertEqual(r["credibility"], "3")
            self.assertEqual((r["occupied_min"], r["occupied_max"]), ("2", "3"))
            self.assertEqual((r["sector_a_min"], r["sector_a_max"], r["sector_b_min"], r["sector_b_max"]), ("1", "1", "1", "2"))
            self.assertTrue((out / r["change_image"]).exists())
            self.assertTrue((out / r["change_thumb"]).exists())
            gj = json.loads((out / "quay" / "2026-10.geojson").read_text())
            self.assertEqual(len(gj["features"]), 2)
            self.assertEqual(gj["features"][0]["geometry"]["type"], "LineString")
            self.assertEqual(gj["features"][0]["properties"]["status"], "trial")
            # est-1: the distribution next to the old range, per stretch and in the zone prior
            self.assertEqual(r["est_version"], "est-1")
            self.assertLessEqual(int(r["est_p10"]), int(r["est_mode"]))
            self.assertLessEqual(int(r["est_mode"]), int(r["est_p90"]))
            self.assertNotEqual(r["est_a_mode"], "")
            p0 = gj["features"][0]["properties"]
            for k in ("n_mode", "n_p10", "n_p90", "prior_conflict", "zone", "face", "outer_len_m", "depth_m",
                      "depth_1x3_m", "db_line_m", "cos_phi", "face_look", "pair_id"):
                self.assertIn(k, p0)
            # feat-1: the look azimuth from the orbit (TQY at 21.0N, ascending)
            self.assertEqual(r["features_version"], "feat-1")
            self.assertAlmostEqual(float(r["look_az_deg"]), 77.6, delta=0.5)
            self.assertEqual(r["pair_id"], "")       # orbit 84 has no quay row here
            zp = json.loads((out / "quay" / "TQY-zone-prior.json").read_text())
            self.assertEqual(zp["applied"], [r["acquired_utc"]])
            st = {x["relative_orbit"]: x for x in json.loads((out / "status.json").read_text())["scenes"]}
            self.assertEqual(st[157]["quay_result"], "counted")
            self.assertEqual(st[84]["quay_result"], "no_baseline")
            # ship_counts.csv (the CFAR count) is still written as before
            self.assertEqual(len(run.read_csv(out / "ship_counts.csv")), 2)
            # next run: nothing new, nothing fetched again
            cli2 = FakeClient(chip)
            s2 = run.process_site(cli2, run.Store(out, [SITE]), SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                                  lines_dir=lines_dir, baseline_dir=bl_dir)
            self.assertEqual(s2["new"], 0)
            self.assertEqual(cli2.processed, [])

    def test_counted_scene_without_quay_is_fetched_again_once(self):
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = self.setup_dirs(d)
            out = d / "out"
            store = run.Store(out, [SITE])
            # first run before the baseline existed
            run.process_site(FakeClient(chip), store, SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                             lines_dir=lines_dir, baseline_dir=d / "none")
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": 3, "processing_units": 2.4,
                        "pu_source": "header", "scenes": 2})
            self.assertFalse((out / "quay_counts.csv").exists())
            cli = FakeClient(chip)
            store = run.Store(out, [SITE])
            s = run.process_site(cli, store, SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                                 lines_dir=lines_dir, baseline_dir=bl_dir)
            self.assertEqual(s["lookback_days"], run.FIRST_LOOKBACK_DAYS)
            self.assertEqual(len(cli.processed), 1)  # only the orbit-157 scene
            store.save({"run_utc": "2026-10-31T05:17:00Z", "requests": cli.requests,
                        "processing_units": cli.processing_units, "pu_source": "header", "scenes": 1})
            self.assertEqual(len(run.read_csv(out / "quay_counts.csv")), 1)
            self.assertEqual(len(run.read_csv(out / "ship_counts.csv")), 2)

    def test_low_coverage_scene_is_not_fetched_again(self):
        """A scene of an orbit with a baseline, skipped for low coverage, is fetched once only
        (it used to be fetched again every day while it stayed in the lookback window)."""
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, _chip, lines_dir, bl_dir = self.setup_dirs(d)
            arr = png.decode(_chip).copy()
            arr[:, : arr.shape[1] * 3 // 4, 3] = 0  # most of the site without data
            chip = png.encode(arr, depth=16, filters=[1])
            out = d / "out"
            processed = 0
            for day in range(3):
                cli = FakeClient(chip, orbits=(157,))
                store = run.Store(out, [SITE])
                s = run.process_site(cli, store, SITE, mask, now + timedelta(days=day), None, 100.0, 12,
                                     log=lambda *a: None, lines_dir=lines_dir, baseline_dir=bl_dir)
                store.save({"run_utc": f"2026-10-{30 + day}T05:17:00Z", "requests": cli.requests,
                            "processing_units": cli.processing_units, "pu_source": "header", "scenes": 1})
                processed += len(cli.processed)
                if day == 0:
                    self.assertEqual(s["skipped"], 1)
            self.assertEqual(processed, 1)
            st = json.loads((out / "status.json").read_text())["scenes"]
            self.assertEqual([(x["result"], x["quay_result"]) for x in st], [("skipped_low_coverage", "skipped")])
            self.assertFalse((out / "quay_counts.csv").exists())

    def test_in_sample_and_quay_attribution(self):
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = self.setup_dirs(d)
            out = d / "out"
            store = run.Store(out, [SITE])
            old = dict(run.QUAY_TUNING_SCENES)
            run.QUAY_TUNING_SCENES["TQY"] = {"2026-10-20T10:48:45Z"}
            try:
                run.process_site(FakeClient(chip), store, SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                                 lines_dir=lines_dir, baseline_dir=bl_dir)
            finally:
                run.QUAY_TUNING_SCENES.clear()
                run.QUAY_TUNING_SCENES.update(old)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": 3, "processing_units": 2.4,
                        "pu_source": "header", "scenes": 2})
            self.assertEqual(run.read_csv(out / "quay_counts.csv")[0]["in_sample"], "tuning")
            gj = json.loads((out / "quay" / "2026-10.geojson").read_text())
            self.assertIn("OpenStreetMap", gj["attribution"])
            self.assertIn("ODbL 1.0", gj["license"])
            self.assertEqual(gj["features"][0]["properties"]["in_sample"], "tuning")

    def test_blind_stretches(self):
        g, mask, lines, cvv, cvh, base = scene_parts()
        geo = quay.Geometry(g, mask, lines)
        dark = np.full(cvv.shape, -20.0)  # no land spreading into the bands
        r = quay.blind_stretches(geo, {"vv_p10": dark})
        self.assertEqual((r["blind_m"], r["strict_m"], r["stretches"]), (0, 0, []))
        self.assertEqual(r["total_m"], geo.samples * 10)
        vv = dark.copy()
        vv[50:61, 10:13] = 0.0  # a ship in every baseline scene along the coast (rows 50-60)
        r = quay.blind_stretches(geo, {"vv_p10": baseline.smooth_db(vv)})
        self.assertTrue(100 <= r["blind_m"] <= 160, r["blind_m"])
        self.assertTrue(80 <= r["strict_m"] <= r["blind_m"], r["strict_m"])
        self.assertTrue(all(len(s["line"]) >= 1 for s in r["stretches"]))

    def test_meta_describes_quay(self):
        m = run.build_meta([], run.load_sites())
        self.assertIn("quay", m)
        self.assertEqual(m["quay"]["method"]["version"], quay.QUAY_VERSION)
        self.assertEqual({b["relative_orbit"] for b in m["quay"]["baselines"]}, {157, 84})
        self.assertIn("quay_counts.csv", m["files"])
        self.assertIn("quay_lines", m["attribution"])
        self.assertEqual(sorted(m["quay"]["tuning"]["scenes"]["ZJG"])[0], "2026-10-02T10:48:45Z")
        bl = {b["relative_orbit"]: b for b in m["quay"]["blind_stretches"]}
        self.assertEqual(set(bl), {157, 84})
        self.assertTrue(0.01 < bl[157]["share"] < bl[84]["share"] < 0.15)
        self.assertNotIn("1 割を超えて", json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(np is not None, "numpy not installed")
class FeaturesTest(unittest.TestCase):
    """feat-1: records only; the counts and the estimate stay as they were."""
    CORE = ("way_id", "kind", "side", "length_m", "est_length_m", "min", "max", "rafted", "peak_dvv_db",
            "peak_dvh_db", "outer_len_m", "lon", "lat", "sector", "line", "cluster", "cluster_lead")

    def setUp(self):
        self.g, self.mask, self.lines, self.cvv, self.cvh, self.base = scene_parts()
        self.geo = quay.Geometry(self.g, self.mask, self.lines)

    def test_look_azimuth_from_the_orbit(self):
        self.assertAlmostEqual(quay.look_azimuth("ascending", 21.23), 77.6, delta=0.5)   # ENE at Zhanjiang
        self.assertAlmostEqual(quay.look_azimuth("descending", 21.23), 282.4, delta=0.5)
        self.assertIsNone(quay.look_azimuth(None, 21.23))

    def test_faces_against_the_radar(self):
        vv, vh = ships(self.cvv, self.cvh)
        q = quay.count(self.geo, vv, vh, self.base, look_az_deg=78.0)
        by = {s["kind"]: s for s in q["segments"]}
        # the coast faces east, the radar is to the WSW: the water side looks away from it
        self.assertEqual(by["coastline"]["face_look"], "away")
        self.assertLess(by["coastline"]["cos_phi"], -0.9)
        self.assertEqual(by["pier"]["face_look"], "along")      # the pier's north side
        q2 = quay.count(self.geo, vv, vh, self.base, look_az_deg=282.4)
        self.assertEqual({s["kind"]: s["face_look"] for s in q2["segments"]}["coastline"], "toward")
        self.assertEqual(quay.face_class(0.6), "toward")
        self.assertEqual(quay.face_class(-0.6), "away")
        self.assertEqual(quay.face_class(0.2), "along")

    def test_depth_and_outer_line(self):
        self.assertEqual(float(self.geo.depth_offs.max()), 120.0)
        vv, vh = ships(self.cvv, self.cvh, raft=True)
        q = quay.count(self.geo, vv, vh, self.base, look_az_deg=78.0)
        by = {s["kind"]: s for s in q["segments"]}
        pier = by["pier"]
        self.assertLess(pier["depth_1x3_m"], pier["depth_m"])     # no spread across the line
        coast = by["coastline"]
        self.assertEqual(coast["db_peaks"], 2)                     # the inner and the outboard ship
        self.assertTrue(40 <= coast["db_line_m"] <= 55)            # the outboard ship: 40-60 m off the edge

    def test_counts_and_estimate_do_not_change(self):
        from sat import quay_est
        vv, vh = ships(self.cvv, self.cvh, raft=True, weak=True)
        old = quay.Geometry(self.g, self.mask, self.lines, {"depth_max_m": 90.0})
        a = quay.count(old, vv, vh, self.base)
        b = quay.count(self.geo, vv, vh, self.base, look_az_deg=78.0)
        for k in ("min", "max", "clusters", "rafted", "coverage", "by_sector"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual([{k: s.get(k) for k in self.CORE} for s in a["segments"]],
                         [{k: s.get(k) for k in self.CORE} for s in b["segments"]])
        m = quay_est.Model()
        ea = quay_est.estimate([dict(s, way_id=218405149) for s in a["segments"]], "ZJG", m)
        eb = quay_est.estimate([dict(s, way_id=218405149) for s in b["segments"]], "ZJG", m)
        self.assertEqual(ea["all"], eb["all"])
        self.assertEqual(ea["warship"], eb["warship"])

    def test_orbit_pairs(self):
        rows = [{"site_id": "ZJG", "mask_version": "2", "acquired_utc": t, "relative_orbit": o} for t, o in [
            ("2026-10-02T10:48:45Z", "157"), ("2026-10-03T10:41:00Z", "84"), ("2026-10-08T10:49:01Z", "157"),
            ("2026-10-14T10:48:45Z", "157"), ("2026-10-15T10:41:00Z", "84"), ("2026-10-20T10:48:45Z", "157")]]
        p = run.orbit_pairs(rows)
        k = lambda t: ("ZJG", t, run.row_mask_tag(rows[0]))
        self.assertEqual(p[k("2026-10-02T10:48:45Z")], p[k("2026-10-03T10:41:00Z")])
        self.assertEqual(p[k("2026-10-14T10:48:45Z")], p[k("2026-10-15T10:41:00Z")])
        self.assertTrue(p[k("2026-10-02T10:48:45Z")].endswith("ro157+ro84"))
        self.assertNotIn(k("2026-10-08T10:49:01Z"), p)               # no other orbit within 36 h
        self.assertNotIn(k("2026-10-20T10:48:45Z"), p)
        rows.append({"site_id": "ZJG", "mask_version": "2", "acquired_utc": "2026-10-21T10:41:00Z", "relative_orbit": "84"})
        p2 = run.orbit_pairs(rows)
        self.assertEqual({kk: v for kk, v in p2.items() if kk in p}, p)  # a new scene never breaks a pair
        self.assertIn(k("2026-10-20T10:48:45Z"), p2)
