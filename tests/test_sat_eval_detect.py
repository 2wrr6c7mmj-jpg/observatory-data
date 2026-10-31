"""sat/eval_detect.py (2部 sar-literature-plan section 3, steps 1-2): matching, metrics, pooled
review sheet, loaders and the two VH code-0 modes, on synthetic data only (the real npz stay on the
probe branches)."""
import csv
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

try:
    from scipy.optimize import linear_sum_assignment as scipy_lsa
except ImportError:  # not installed in the Tests workflow
    scipy_lsa = None

if np is not None:
    from sat import eval_detect as ev

LAT = 21.2
M_LAT = 110_700.0    # rough metres per degree, enough to place test points


def _pt(x_m, y_m, scene="s", **kw):
    """A point x_m east / y_m north of (110.0, LAT)."""
    return {"scene": scene, "lon": 110.0 + x_m / (M_LAT * math.cos(math.radians(LAT))), "lat": LAT + y_m / M_LAT, **kw}


@unittest.skipUnless(np is not None, "numpy not installed")
class StatsTest(unittest.TestCase):
    def test_wilson_known_values(self):
        w = ev.wilson(5, 10)
        self.assertAlmostEqual(w["lo"], 0.2366, places=3)
        self.assertAlmostEqual(w["hi"], 0.7634, places=3)
        w = ev.wilson(0, 10)
        self.assertEqual(w["lo"], 0.0)
        self.assertAlmostEqual(w["hi"], 0.2775, places=3)
        self.assertIsNone(ev.wilson(0, 0)["p"])

    def test_f1_interval_contains_point(self):
        f = ev.f1_interval(8, 2, 4)
        self.assertAlmostEqual(f["p"], 16 / 22, places=3)
        self.assertLess(f["lo"], f["p"])
        self.assertGreater(f["hi"], f["p"])

    def test_rate_interval(self):
        r = ev.rate_interval(3, 500.0)
        self.assertAlmostEqual(r["rate"], 6.0)
        self.assertLess(r["lo"], 6.0)
        self.assertGreater(r["hi"], 6.0)
        self.assertEqual(ev.rate_interval(0, 500.0)["lo"], 0.0)

    def test_spearman(self):
        self.assertAlmostEqual(ev.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(ev.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(ev.spearman([1, 2], [1, 2]))


@unittest.skipUnless(np is not None, "numpy not installed")
class HungarianTest(unittest.TestCase):
    def _brute(self, c):
        n, m = c.shape
        best = math.inf
        if n <= m:
            for perm in itertools.permutations(range(m), n):
                best = min(best, sum(c[i, perm[i]] for i in range(n)))
        else:
            for perm in itertools.permutations(range(n), m):
                best = min(best, sum(c[perm[j], j] for j in range(m)))
        return best

    def test_against_brute_force(self):
        rng = np.random.default_rng(1)
        for shape in [(1, 1), (3, 3), (4, 6), (6, 4), (5, 5), (2, 7)]:
            for _ in range(5):
                c = rng.integers(0, 20, shape).astype(float)
                r, k = ev.linear_sum_assignment(c)
                self.assertEqual(len(r), min(shape))
                self.assertEqual(len(set(r.tolist())), len(r))
                self.assertEqual(len(set(k.tolist())), len(k))
                self.assertAlmostEqual(c[r, k].sum(), self._brute(c))

    def test_empty(self):
        r, k = ev.linear_sum_assignment(np.zeros((0, 3)))
        self.assertEqual(len(r), 0)

    @unittest.skipUnless(scipy_lsa is not None, "scipy not installed")
    def test_same_cost_as_scipy(self):
        rng = np.random.default_rng(2)
        for shape in [(20, 20), (15, 30), (30, 12)]:
            c = rng.random(shape)
            r, k = ev.linear_sum_assignment(c)
            r2, k2 = scipy_lsa(c)
            self.assertAlmostEqual(c[r, k].sum(), c[r2, k2].sum(), places=9)

    def test_match_tolerance_and_one_to_one(self):
        preds = [_pt(0, 0), _pt(150, 0), _pt(1000, 0), _pt(0, 0, scene="t")]
        truths = [_pt(10, 0), _pt(120, 0)]
        m = ev.match(preds, truths, 200.0)
        # optimal: pred0-truth0 (10 m) and pred1-truth1 (30 m); the far pred and the other scene unmatched
        self.assertEqual([(i, j) for i, j, _ in m], [(0, 0), (1, 1)])
        m100 = ev.match([_pt(0, 0)], [_pt(150, 0)], 100.0)
        self.assertEqual(m100, [])

    def test_match_maximises_pairs_first(self):
        # greedy nearest would pair p0-t1 (5 m) and leave p1 alone; the assignment pairs both
        preds = [_pt(0, 0), _pt(190, 0)]
        truths = [_pt(-150, 0), _pt(5, 0)]
        m = ev.match(preds, truths, 200.0)
        self.assertEqual(len(m), 2)


@unittest.skipUnless(np is not None, "numpy not installed")
class ScoreTest(unittest.TestCase):
    def test_counts_low_confidence_and_not_ship(self):
        truths = [_pt(0, 0, label="ship", length_m=40), _pt(1000, 0, label="ship", length_m=120),
                  _pt(2000, 0, label="たぶん船"), _pt(3000, 0, label="船でない"), _pt(4000, 0, label="ship", length_m=200)]
        preds = [_pt(20, 0, length_m=50), _pt(2010, 0), _pt(3010, 0), _pt(4020, 0, length_m=180)]
        s = ev.score(preds, truths, 200.0, area_km2=100.0)
        self.assertEqual((s["tp"], s["fp"], s["fn"], s["ignored_low"]), (2, 1, 1, 1))
        self.assertAlmostEqual(s["precision"]["p"], 2 / 3, places=3)
        self.assertAlmostEqual(s["recall"]["p"], 2 / 3, places=3)
        self.assertEqual(s["recall_by_length"]["25-50"]["k"], 1)
        self.assertEqual(s["recall_by_length"]["100-150"]["k"], 0)
        self.assertEqual(s["recall_by_length"][">=150"]["k"], 1)
        self.assertEqual(s["length_error"]["n"], 2)
        self.assertAlmostEqual(s["length_error"]["mean"], (10 / 40 + 20 / 200) / 2, places=4)
        self.assertAlmostEqual(s["false_alarms_per_1000km2"]["rate"], 10.0)

    def test_length_error_as_xview3(self):
        # per-pair error is not capped at 1: (300 - 100) / 100 = 2, mean (2 + 0) / 2 = 1 -> score 0
        e = ev.length_error([(300.0, 100.0), (100.0, 100.0)])
        self.assertAlmostEqual(e["mean"], 1.0)
        self.assertAlmostEqual(e["score"], 0.0)
        # mean above 1 -> score floored at 0
        self.assertAlmostEqual(ev.length_error([(400.0, 100.0)])["score"], 0.0)
        self.assertAlmostEqual(ev.length_error([(400.0, 100.0)])["mean"], 3.0)
        # lengths capped at 500 m: (800 -> 500) vs 400 = 0.25; (600 -> 500) vs (700 -> 500) = 0
        e = ev.length_error([(800.0, 400.0), (600.0, 700.0)])
        self.assertAlmostEqual(e["mean"], 0.125)
        self.assertAlmostEqual(e["score"], 0.875)

    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            ev.score([], [_pt(0, 0, label="boat?")])

    def test_by_shore_filters_both_sides(self):
        truths = [_pt(0, 0, label="ship", shore_km=0.5), _pt(5000, 0, label="ship", shore_km=8.0)]
        preds = [_pt(10, 0, shore_km=0.6), _pt(5010, 0, shore_km=8.0), _pt(9000, 0, shore_km=1.5)]
        s = ev.score_by_shore(preds, truths)
        self.assertEqual((s["0-1"]["tp"], s["0-1"]["fp"]), (1, 0))
        self.assertEqual((s["1-2"]["tp"], s["1-2"]["fp"]), (0, 1))
        self.assertEqual(s[">5"]["tp"], 1)
        self.assertEqual((s["near_shore_0-2"]["tp"], s["near_shore_0-2"]["fp"], s["near_shore_0-2"]["fn"]), (1, 1, 0))

    def test_near_shore_as_xview3(self):
        truths = [_pt(0, 0, label="ship", shore_km=1.95),            # scene s: near-shore truth
                  _pt(0, 0, scene="t", label="ship", shore_km=1.0),  # scene t: no near-shore prediction
                  _pt(5000, 0, label="ship", shore_km=2.3)]          # beyond 2 km: not near-shore truth
        preds = [_pt(150, 0, shore_km=2.15),                         # within 2 km + 0.2 km: kept, matches
                 _pt(5000, 0, shore_km=2.3),                         # beyond 2.2 km: left out
                 _pt(0, 0, scene="u", shore_km=0.5)]                 # scene u: no near-shore truth
        n = ev.near_shore(preds, truths, 200.0)
        self.assertEqual((n["tp"], n["fp"], n["fn"]), (1, 0, 0))
        self.assertEqual((n["scenes"], n["scenes_left_out"]), (1, 2))
        # with a 100 m match distance the prediction at 2.15 km is outside 2.1 km and scene s drops out
        n = ev.near_shore(preds, truths, 100.0)
        self.assertEqual((n["tp"], n["scenes"]), (0, 0))

    def test_bands(self):
        self.assertEqual(ev.band_of(24.9, ev.LENGTH_BANDS), "<25")
        self.assertEqual(ev.band_of(150, ev.LENGTH_BANDS), ">=150")
        self.assertEqual(ev.band_of(None, ev.LENGTH_BANDS), "unknown")
        self.assertEqual(ev.band_of(2.0, ev.SHORE_BANDS), "2-5")

    def test_shore_km(self):
        land = np.zeros((50, 50), bool)
        land[:, :10] = True
        d = ev.shore_km([25, 25], [20, 49], land, (20.0, 20.0))
        self.assertAlmostEqual(d[0]["km"], 0.22, places=3)
        self.assertTrue(d[0]["exact"])
        d = ev.shore_km([25], [25], np.zeros((50, 50), bool), (20.0, 20.0), coarse=[(0, 200, 49, 260)])
        self.assertAlmostEqual(d[0]["km"], 3.5, places=3)
        self.assertFalse(d[0]["exact"])
        self.assertIsNone(ev.shore_km([1], [1], np.zeros((5, 5), bool), 20.0)[0]["km"])


@unittest.skipUnless(np is not None, "numpy not installed")
class PoolTest(unittest.TestCase):
    def test_pool_merges_and_hides_sources(self):
        cands = {"m1": [_pt(0, 0), _pt(5000, 0)], "m2": [_pt(150, 0), _pt(0, 0, scene="t")], "m3": [_pt(320, 0)]}
        rows, key = ev.pool(cands, 200.0)
        # 0, 150, 320 m chain into one cluster (single linkage); 5000 m alone; scene t alone
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["sources"] == "" and r["label"] == "" for r in rows))
        merged = [v for v in key.values() if v["n"] == 3]
        self.assertEqual(merged[0]["sources"], ["m1", "m2", "m3"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pool.csv"
            ev.write_pool_csv(p, rows)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.readline().strip(), ",".join(ev.POOL_FIELDS))
            # judged sheet back as truth
            with open(p, encoding="utf-8") as f:
                rr = list(csv.DictReader(f))
            rr[0]["label"] = "船"
            rr[1]["label"] = "不明"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ev.POOL_FIELDS)
                w.writeheader()
                w.writerows(rr)
            j = ev.read_judged(p)
            self.assertEqual(len(j), 2)
            self.assertEqual(ev.truth_class(j[0]), "ship")
            self.assertEqual(ev.truth_class(j[1]), "low")

    def test_key_must_be_apart_from_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            ev.check_split(Path(d) / "review", Path(d) / "key")
            for a, b in (("review", "review"), ("review", "review/key"), ("key/review", "key")):
                with self.assertRaises(SystemExit):
                    ev.check_split(Path(d) / a, Path(d) / b)

    def test_main_writes_key_apart(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw" / "run1"
            raw.mkdir(parents=True)
            _write_raw_scene(raw)
            out, key = Path(d) / "review", Path(d) / "key"
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                ev.main(["--ocean-raw", str(Path(d) / "raw"), "--out", str(out), "--key-out", str(key)])
            names = {p.name for p in out.iterdir()}
            self.assertIn("pool-ocean.csv", names)
            self.assertFalse(any("key" in n or n.startswith("candidates-") for n in names))
            self.assertTrue((key / "pool-ocean-key.json").exists())
            self.assertTrue((key / "candidates-ocean-pr2.1.geojson").exists())

    def test_pool_order_does_not_follow_method(self):
        a, _ = ev.pool({"x": [_pt(0, 0)], "y": [_pt(5000, 0)]}, salt="s")
        b, _ = ev.pool({"y": [_pt(0, 0)], "x": [_pt(5000, 0)]}, salt="s")
        self.assertEqual([(r["lon"], r["lat"]) for r in a], [(r["lon"], r["lat"]) for r in b])


def _clutter(rng, shape, mean_db, looks=4.4):
    return 10 * np.log10(10 ** (mean_db / 10) * rng.gamma(looks, 1 / looks, shape))


@unittest.skipUnless(np is not None, "numpy not installed")
class ModesTest(unittest.TestCase):
    def test_mode_inputs(self):
        vv = np.array([[-20.0, -21.0]])
        vh = np.array([[np.nan, -30.0]])
        a, b, dm = ev.mode_inputs(vv, vh, np.ones_like(vv, bool), "pr2.1")
        self.assertTrue(np.isfinite(a).all())
        self.assertIsNotNone(dm)
        a, b, dm = ev.mode_inputs(vv, vh, np.ones_like(vv, bool), "pre33")
        self.assertTrue(np.isnan(a[0, 0]) and np.isfinite(a[0, 1]))
        self.assertIsNone(dm)
        with self.assertRaises(ValueError):
            ev.mode_inputs(vv, vh, None, "x")

    def test_vh_zero_breaks_the_fit_only_before_pr33(self):
        """A lone 128 px tile with 6% scattered VH code 0: cfar2-pr2 (VH 0 = no data, dilated 3 px)
        has no usable water and no fit; pr2.1 fits it and finds the bright target."""
        rng = np.random.default_rng(3)
        vv = _clutter(rng, (128, 128), -20.0)
        vh = _clutter(rng, (128, 128), -33.0)
        vv[60:63, 60:70] += 18.0
        vh[60:63, 60:70] += 15.0
        zero = rng.random((128, 128)) < 0.06
        zero[55:68, 55:75] = False
        vh[zero] = np.nan
        land = np.zeros((128, 128), bool)
        dm = np.ones((128, 128), bool)
        d_new, i_new, _s, tt_new = ev.run_v2(vv, vh, land, (20.0, 20.0), "pr2.1", datamask=dm)
        d_old, i_old, _s, tt_old = ev.run_v2(vv, vh, land, (20.0, 20.0), "pre33", datamask=dm)
        self.assertEqual(ev._own_failed(tt_old), 1)
        self.assertEqual(ev._own_failed(tt_new), 0)
        self.assertEqual(len(d_old), 0)
        from sat import detect2 as v2
        k = v2.kept(d_new)
        self.assertTrue(any(58 <= d["row"] <= 65 and 58 <= d["col"] <= 72 for d in k))


def _write_raw_scene(root: Path, area="TE", when="2026-10-14T10:00:29Z"):
    rng = np.random.default_rng(4)
    n = 2
    code = lambda db: np.clip(np.round((db + 50) * 800), 1, 65535).astype(np.uint16)  # noqa: E731
    vv = code(_clutter(rng, (n, 128, 128), -20.0))
    vh = code(_clutter(rng, (n, 128, 128), -33.0))
    vh[0, :10, :10] = 0           # 100 code-0 pixels in tile 0
    np.savez_compressed(root / f"{area}-{when.replace(':', '-')}.npz", vv=vv, vh=vh,
                        lia=np.full((n, 128, 128), 3500, np.uint16), datamask=np.ones((n, 128, 128), np.uint8),
                        land=np.zeros((n, 128, 128), np.uint8), static=np.zeros((n, 128, 128), np.uint8),
                        tiles=np.array([[0, 0, 0, 0], [0, 1, 0, 128]], np.int32))
    meta = {"area": area, "acquisition": {"acquired_utc": when, "orbit_direction": "ascending"},
            "request": {"bbox_epsg4326": [120.0, 24.0, 120.05, 24.05], "width": 256, "height": 256, "pixel_m": [20.0, 20.0]},
            "all_tiles_cols": ["ty", "tx", "datamask", "vv_zero", "vh_zero", "nodata_band", "land", "static", "usable",
                               "n_fit", "model", "own_model", "borrowed", "vv_med_db", "vh_med_db", "vh_p05_db", "lia_med"],
            "all_tiles": [[0, 0, 1.0, 0, 0.006, 0.02, 0.0, False, 0.9, 100, "lognormal", "lognormal", False, -20, -33, -36, 35],
                          [0, 1, 1.0, 0, 0.0, 0.0, 0.0, False, 0.9, 100, "lognormal", None, True, -20, -33, -36, 35]]}
    (root / f"{area}-{when.replace(':', '-')}.json").write_text(json.dumps(meta), encoding="utf-8")


@unittest.skipUnless(np is not None, "numpy not installed")
class LoadersTest(unittest.TestCase):
    def test_load_ocean_raw_and_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run1"
            root.mkdir()
            _write_raw_scene(root)
            sc = ev.load_ocean_raw(Path(d))
        self.assertEqual(len(sc), 1)
        t0, t1 = sc[0]["tiles"]
        self.assertTrue(np.isnan(t0["vh"][0, 0]) and np.isfinite(t0["vv"][0, 0]))
        self.assertAlmostEqual(t0["vh_zero"], 100 / 128 ** 2, places=4)
        self.assertAlmostEqual(t0["lia"][5, 5], 35.0)
        self.assertAlmostEqual(t1["grid"].west, 120.025, places=6)
        self.assertAlmostEqual(t1["grid"].north, 24.05, places=6)
        rows, cands = ev.ocean_runs(sc)
        self.assertEqual(len(rows), 4)
        tab = ev.scene_table(rows, sc)
        self.assertEqual(tab[0]["scene_failed_pre33"], 1)
        h = ev.hypothesis(rows, tab)
        self.assertIn("full", h["tile_failed_by_vh_zero"])

    def test_load_zjg_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = {"acquired_utc": "2026-01-04T10:48:40Z", "relative_orbit": 157}
            (root / "s1-for-2026-01-03.json").write_text(json.dumps(a), encoding="utf-8")
            (root / "s1-for-2026-01-05.json").write_text("null", encoding="utf-8")
            vv = np.full((4, 3), np.nan, np.float16)
            vv[1, 1] = -20
            np.savez_compressed(root / "s1-2026-01-04T10-48-40Z.npz", vv=vv, vh=vv)
            np.savez_compressed(root / "s2-2026-01-03.npz", refl=np.zeros((4, 3, 4), np.uint16), scl=np.zeros((4, 3), np.uint8))
            p = ev.load_zjg_pairs(root)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]["date"], "2026-01-03")
        self.assertEqual(int(p[0]["keep"].sum()), 1)
        self.assertIn("s2", p[0])


@unittest.skipUnless(np is not None, "numpy not installed")
class SectionsTest(unittest.TestCase):
    def test_assign_and_compare(self):
        secs = [{"section_id": "a", "line": [[110.0, 21.2], [110.0, 21.201]]},
                {"section_id": "b", "line": [[110.01, 21.2], [110.01, 21.201]]}]
        pts = [{"lon": 110.0005, "lat": 21.2005}, {"lon": 110.0095, "lat": 21.2005}, {"lon": 110.005, "lat": 21.2005}]
        self.assertEqual(ev.assign_sections(pts, secs, 130.0), ["a", "b", None])
        truth = {("a", "d"): {"occupied": 1, "source": "verified"}, ("b", "d"): {"occupied": 0, "source": "auto"},
                 ("c", "d"): {"occupied": None, "source": "unknown"}}
        c = ev.occupancy_compare({("a", "d"): 1, ("b", "d"): 1, ("c", "d"): 1}, truth)
        self.assertEqual((c["all"]["tp"], c["all"]["fp"], c["all"]["n"]), (1, 1, 2))
        self.assertEqual(c["verified"]["n"], 1)

    def test_truth_priority(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d) / "labels.csv"
            lab.write_text("section_id,date,status,occupied,verified,verified_occupied\n"
                           "a,d,ok,1,,\nb,d,ok,0,1,1\nc,d,not_water,,,\ne,d,ok,1,0,\n", encoding="utf-8")
            ver = Path(d) / "ver.csv"
            ver.write_text("vid,section_id,date,verified,verified_occupied\nv1,a,d,1,0\n", encoding="utf-8")
            t = ev.zjg_truth(lab, ver)
        self.assertEqual(t[("a", "d")], {"occupied": 0, "source": "verified"})
        self.assertEqual(t[("b", "d")]["occupied"], 1)
        self.assertNotIn(("c", "d"), t)
        self.assertIsNone(t[("e", "d")]["occupied"])

    def test_attribution(self):
        self.assertEqual(ev.attribution(["2026-01-01T00:00:00Z", "2025-11-11T00:00:00Z"]),
                         "Contains modified Copernicus Sentinel data 2025, 2026")


if __name__ == "__main__":
    unittest.main()
