"""cols-1 (sat/cols_xsec.py): rows abreast from the cross-section across the quay line, per orbit direction."""
import unittest

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cols_xsec as cx, eval_cols, quay, s2_labels
    from tests.test_sat_quay import scene_parts


@unittest.skipUnless(np is not None, "numpy not installed")
class ModelTest(unittest.TestCase):
    def test_fit_recovers_width_per_row(self):
        rng = np.random.default_rng(0)
        c = np.repeat([1, 2, 3], 20)
        w = 28 + 24 * (c - 1) + rng.normal(0, 6, len(c))
        m = cx.fit(w, c)
        self.assertEqual(m["status"], "fitted")
        self.assertAlmostEqual(m["w0"], 28, delta=3)
        self.assertAlmostEqual(m["beam"], 24, delta=3)
        self.assertAlmostEqual(m["sigma"], 6, delta=2)

    def test_small_or_single_level_samples(self):
        self.assertEqual(cx.fit([30, 40], [1, 1])["status"], "default")
        m = cx.fit([30, 35, 40, 32, 38, 36], [1] * 6)
        self.assertEqual((m["status"], m["beam"]), ("w0_only", cx.DEFAULT_MODEL["beam"]))
        self.assertAlmostEqual(m["w0"], 35.2, delta=0.1)
        m = cx.fit([60, 50, 40, 60, 50, 40], [1, 2, 3, 1, 2, 3])      # narrower with more rows: noise
        self.assertEqual(m["status"], "w0_only")

    def test_posterior_combines_directions(self):
        cal = {"ascending": {"w0": 30, "beam": 25, "sigma": 10}, "descending": {"w0": 20, "beam": 30, "sigma": 10}}
        self.assertEqual(cx.posterior({}, cal), list(cx.PRIOR_C))
        a = cx.posterior({"ascending": 55}, cal)
        both = cx.posterior({"ascending": 55, "descending": 50}, cal)
        self.assertEqual(int(np.argmax(a)), 1)
        self.assertGreater(both[1], a[1])                                  # the descending look agrees: sharper
        self.assertAlmostEqual(sum(both), 1.0)
        self.assertEqual(int(np.argmax(cx.posterior({"ascending": 30, "descending": None}, cal))), 0)

    def test_calibration_file(self):
        cal = cx.load_calib()
        self.assertEqual(cal["ascending"]["status"], "fitted")
        self.assertGreater(cal["ascending"]["beam"], 10)
        self.assertEqual(cal["descending"]["status"], "default")           # no descending scene yet


@unittest.skipUnless(np is not None, "numpy not installed")
class CrossSectionTest(unittest.TestCase):
    def setUp(self):
        self.g, self.mask, self.lines, self.cvv, self.cvh, self.base = scene_parts()
        self.geo = quay.Geometry(self.g, self.mask, self.lines)
        self.secs = [s for s in s2_labels.sections(self.geo, section_m=50.0) if s["kind"] == "coastline"]

    def width_at(self, rows_px):
        vv = self.cvv.copy()
        vv[50:61, 10:10 + rows_px] = 5.0
        dvv = vv - self.base["vv_p10"]
        out = []
        for s in self.secs:
            rr = [self.g.to_pixel(lo, la)[0] for lo, la in s["line"]]
            if min(rr) <= 55 <= max(rr):
                w = cx.section_width(self.geo, s, dvv)
                if w:
                    out.append(w["width_m"])
        return max(out) if out else None

    def test_width_grows_with_rafted_ships(self):
        one, two = self.width_at(2), self.width_at(5)
        self.assertIsNotNone(one)
        self.assertGreater(two, one + 15)
        cal = {"ascending": {"w0": one, "beam": two - one, "sigma": 8}}
        self.assertEqual(int(np.argmax(cx.posterior({"ascending": one}, cal))) + 1, 1)
        self.assertEqual(int(np.argmax(cx.posterior({"ascending": two}, cal))) + 1, 2)

    def test_empty_section_has_no_width(self):
        dvv = self.cvv - self.base["vv_p10"]
        self.assertTrue(all(cx.section_width(self.geo, s, dvv) is None for s in self.secs))


@unittest.skipUnless(np is not None, "numpy not installed")
class EvalTest(unittest.TestCase):
    def test_leave_one_day_out_never_sees_its_day(self):
        rows = [{"date": "d1", "direction": "ascending", "owner_c": 3, "w": 200.0}] + \
               [{"date": f"d{k}", "direction": "ascending", "owner_c": 1 + k % 2, "w": 30.0 + 25 * (k % 2)} for k in range(2, 14)]
        post = eval_cols._lodo(rows, "w")
        self.assertEqual(len(post), len(rows))
        # a model fitted with the outlier day would put far less weight on c = 3 for the others; here d1 itself
        # is judged by the others' fit, which has seen no c = 3 width: still c = 3 is the most likely for 200 m
        self.assertEqual(int(np.argmax(post[0])), 2)

    def test_zjg_rows_have_raw_depth(self):
        z = eval_cols.zjg_raw_rows()
        self.assertGreater(len(z), 30)
        self.assertTrue(all(r["direction"] == "ascending" and r["owner_c"] in (1, 2, 3) for r in z))


if __name__ == "__main__":
    unittest.main()
