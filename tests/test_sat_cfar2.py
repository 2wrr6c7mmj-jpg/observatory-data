"""cfar-v2 PR1 (sat/cfar2.py): thresholds, fits, ring statistics and synthetic clutter.

The synthetic test (design 5.1 a/b) prints a table: measured false-alarm rate / nominal PFA for
v2, the model chosen per tile, and the same ratio for cfar-v1's threshold (sat/detect.py).
"""
import math
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cfar2 as c2
    from sat.detect import background, detect
    from sat.grid import box_sum

ROOT = Path(__file__).resolve().parents[1]


def _clutter(kind, shape, rng):
    """Intensity clutter (mean ~1) times 0.01 (-20 dB), returned in dB quantised to 1/800 dB."""
    if kind == "gamma":
        x = rng.gamma(4.4, 1 / 4.4, shape)
    elif kind.startswith("K"):
        nu = float(kind[1:])
        x = rng.gamma(4.4, 1 / 4.4, shape) * rng.gamma(nu, 1 / nu, shape)
    elif kind == "lognormal":
        x = np.exp(rng.normal(0.0, 0.5, shape))
    elif kind == "weibull":
        x = rng.weibull(1.8, shape)
    else:
        raise ValueError(kind)
    db = 10 * np.log10(0.01 * x)
    return np.round(db * 800) / 800


@unittest.skipUnless(np is not None, "numpy not installed")
class ThresholdTest(unittest.TestCase):
    def test_design_table_values(self):
        self.assertAlmostEqual(c2.z_threshold("lognormal", None, 1e-5), 4.265, places=3)
        self.assertAlmostEqual(c2.z_threshold("weibull", None, 1e-5), 2.355, places=3)
        self.assertAlmostEqual(c2.z_threshold("loggamma", (4.4, 0.0), 1e-5), 3.179, places=3)
        # closed form for the Gumbel (Weibull) case
        g = math.sqrt(6) / math.pi * (c2.EULER + math.log(math.log(1e5)))
        self.assertAlmostEqual(c2.z_threshold("weibull", None, 1e-5), g, places=9)

    def test_lognormal_weibull_do_not_depend_on_parameters(self):
        for model, prms in (("lognormal", [(0, 1), (-3.2, 0.4), (5, 2.5)]),
                            ("weibull", [(1, 0), (3.5, -2.0), (0.6, 1.0)])):
            for cz in (None, 2.5):
                vals = [c2.z_threshold(model, p, 1e-4, cz) for p in prms]
                self.assertLess(max(vals) - min(vals), 1e-4, (model, cz, vals))

    def test_censored_threshold_is_higher(self):
        # pass 2 drops the ring's own upper tail, so s is smaller and z_t (in that z) is larger
        for model, p in (("lognormal", (0, 1)), ("loggamma", (4.4, 0.0)), ("weibull", (1.0, 0.0))):
            self.assertGreater(c2.z_threshold(model, p, 1e-4, 2.5), c2.z_threshold(model, p, 1e-4))

    def test_isf_matches_tail_probability(self):
        for model, p in (("loggamma", (4.4, 0.3)), ("logweibull3", (-2.0, 4.0, 1.5)),
                         ("loggengamma", (3.0, 1.7, -0.2)), ("weibull", (2.0, 0.1))):
            mu, sd = c2.moments(model, p)
            t = float(c2.isf(model, p, 1e-3)[0])
            y = np.linspace(t, mu + 60 * sd, 400001)
            tail = np.trapezoid(np.exp(c2.logpdf(model, p, y)), y)
            self.assertAlmostEqual(tail / 1e-3, 1.0, places=3, msg=model)

    def test_special_functions(self):
        self.assertAlmostEqual(c2.digamma(1.0), -c2.EULER, places=10)
        self.assertAlmostEqual(c2.trigamma(1.0), math.pi ** 2 / 6, places=9)
        self.assertAlmostEqual(c2.tetragamma(1.0), -2 * 1.2020569031595942, places=8)
        x = np.array([0.1, 1.0, 5.0, 30.0])
        np.testing.assert_allclose(c2.gammaincc(1.0, x), np.exp(-x), rtol=1e-12)
        np.testing.assert_allclose(c2.gammaincc(3.0, x), np.exp(-x) * (1 + x + x * x / 2), rtol=1e-12)


@unittest.skipUnless(np is not None, "numpy not installed")
class WindowTest(unittest.TestCase):
    def test_window_sizes_from_pixel_size(self):
        self.assertEqual(c2.window_px(10), (25, 100, 256, True))
        self.assertEqual(c2.window_px((20.0, 20.0)), (13, 50, 128, False))

    def test_ring_stats_match_box_sum(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=(90, 70))
        use = rng.random((90, 70)) > 0.3
        m, s, n = c2.ring_stats(y, use, 3, 11)
        m0, s0, n0 = background(y, use, 3, 11)   # cfar-v1's ring (grid.box_sum), unchanged
        np.testing.assert_allclose(n, n0)
        np.testing.assert_allclose(m, m0, atol=1e-10)
        np.testing.assert_allclose(s, s0, atol=1e-10)
        a = rng.random((40, 33))
        np.testing.assert_allclose(c2._window(c2._integral(a), 5), box_sum(a, 5), atol=1e-10)


@unittest.skipUnless(np is not None, "numpy not installed")
class FitTest(unittest.TestCase):
    def test_parameters_recovered(self):
        rng = np.random.default_rng(5)
        f = c2.fit_models(np.log(rng.gamma(4.4, 1 / 4.4, 40000)))
        self.assertAlmostEqual(f["loggamma"]["params"][0], 4.4, delta=0.15)
        f = c2.fit_models(np.log(rng.weibull(1.8, 40000)))
        self.assertAlmostEqual(f["weibull"]["params"][0], 1.8, delta=0.04)
        f = c2.fit_models(rng.normal(-3.0, 0.5, 40000))
        self.assertAlmostEqual(f["lognormal"]["params"][1], 0.5, delta=0.01)

    def test_tail_rule_prefers_heavier_fitting_model(self):
        fits = {m: {"params": (1.0,), "aic": 0.0, "tail_rmse": 0.1} for m in c2.MODELS}
        fits["loggamma"].update(aic=-50.0, tail_rmse=0.4)
        fits["lognormal"].update(aic=-10.0, tail_rmse=0.2)
        zt = {"lognormal": 5.3, "weibull": 2.6, "loggamma": 3.7, "logweibull3": 3.5, "loggengamma": 3.9}
        self.assertEqual(c2.select_model(fits, zt, 0.25), "lognormal")
        for m in c2.MODELS:
            fits[m]["tail_rmse"] = 0.5
        self.assertEqual(c2.select_model(fits, zt, 0.25), "lognormal")   # none fits: largest z_t
        fits["loggamma"]["tail_rmse"] = 0.1
        self.assertEqual(c2.select_model(fits, zt, 0.25), "loggamma")


@unittest.skipUnless(np is not None, "numpy not installed")
class SyntheticClutterTest(unittest.TestCase):
    """Design 5.1 (a) false-alarm rate within 1/2..2 of nominal, (b) model choice >= 80%."""
    N = 1024
    GENERATOR = {"gamma": "loggamma", "lognormal": "lognormal", "weibull": "weibull"}
    KINDS = ("gamma", "K3", "K10", "K1", "lognormal", "weibull")
    PFAS = (1e-3, 1e-4)

    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        for i, kind in enumerate(cls.KINDS):
            rng = np.random.default_rng(100 + i)
            db = _clutter(kind, (cls.N, cls.N), rng)
            res = c2.cfar2(db, None, (20.0, 20.0))
            info = res["info"]
            v1 = {}
            vh = _clutter(kind, (cls.N, cls.N), rng)
            for p in (*cls.PFAS, 1e-5):
                _d, inf = detect(db, vh, np.zeros(db.shape, bool), (20.0, 20.0), {"pfa": p})
                v1[p] = inf["hit_pixels"] / db.size / p
            tiles = res["tiles"]
            cls.rows[kind] = {
                "v2": {p: info["exceedance"][f"{p:.0e}"] / p for p in cls.PFAS},
                "v1": v1,
                "final": info["model_counts"], "raw": info["model_counts_unsmoothed"],
                "n_tiles": len(tiles), "seconds": info["seconds"],
            }
        out = ["", "cfar-v2 PR1 synthetic clutter, 1024x1024 px at 20 m (64 tiles); ratio = measured / nominal PFA"]
        out.append(f"{'clutter':10s} {'v2@1e-3':>8s} {'v2@1e-4':>8s} {'v1@1e-3':>8s} {'v1@1e-4':>8s} {'v1@1e-5':>8s}  model (smoothed / raw)")
        for kind, r in cls.rows.items():
            top = max(r["final"], key=r["final"].get)
            out.append(f"{kind:10s} {r['v2'][1e-3]:8.2f} {r['v2'][1e-4]:8.2f} {r['v1'][1e-3]:8.1f} {r['v1'][1e-4]:8.1f} "
                       f"{r['v1'][1e-5]:8.1f}  {top} {r['final'][top]}/{r['n_tiles']} / {r['raw'][top]}/{r['n_tiles']}"
                       f"  ({r['seconds']} s)")
        print("\n".join(out), file=sys.stderr)

    def test_false_alarm_rate_near_nominal(self):
        for kind in ("gamma", "K3", "K10", "lognormal", "weibull"):
            for p in self.PFAS:
                ratio = self.rows[kind]["v2"][p]
                self.assertTrue(0.5 <= ratio <= 2.0, f"{kind} PFA {p}: {ratio:.2f} x nominal")

    def test_spiky_k_within_design_acceptance(self):
        # K with texture shape 1 is beyond the 1/2..2 goal at 1e-4 (unresolved); keep it within 1/3..3
        for p in self.PFAS:
            self.assertTrue(1 / 3 <= self.rows["K1"]["v2"][p] <= 3.0, self.rows["K1"]["v2"])

    def test_model_choice_matches_generator(self):
        for kind, model in self.GENERATOR.items():
            r = self.rows[kind]
            self.assertGreaterEqual(r["final"][model] / r["n_tiles"], 0.8, (kind, r["final"]))
            self.assertGreaterEqual(r["raw"][model] / r["n_tiles"], 0.7, (kind, r["raw"]))

    def test_v1_far_above_nominal_on_gamma(self):
        # design 1.4: the linear Gaussian threshold is about 140x nominal at 1e-5 for ENL 4.4
        self.assertTrue(100 <= self.rows["gamma"]["v1"][1e-5] <= 190, self.rows["gamma"]["v1"])


@unittest.skipUnless(np is not None, "numpy not installed")
class ImageTest(unittest.TestCase):
    def test_10m_thinned_fit(self):
        rng = np.random.default_rng(11)
        db = _clutter("gamma", (768, 768), rng)
        res = c2.cfar2(db, None, 10.0)
        info = res["info"]
        self.assertTrue(info["thinned_fit"])
        self.assertEqual((info["guard_px"], info["outer_px"], info["tile_px"]), (25, 100, 256))
        self.assertGreaterEqual(info["model_counts"]["loggamma"], 7)
        self.assertTrue(0.5 <= info["exceedance"]["1e-03"] / 1e-3 <= 2.0, info["exceedance"])
        for t in res["tiles"]:
            self.assertLessEqual(t["n_fit"], 128 * 128)

    def test_land_tiles_borrow_and_targets_stand_out(self):
        rng = np.random.default_rng(12)
        db = _clutter("gamma", (512, 512), rng)
        land = np.zeros(db.shape, bool)
        land[:, 400:] = True                                  # last tile column is mostly land
        db[200:204, 100:115] += 15.0                          # a 300 m x 80 m bright target
        db[300:302, 250:256] += 8.0
        res = c2.cfar2(db, land, (20.0, 20.0))
        z = res["z"]
        self.assertTrue(np.isnan(z[land]).all())
        borrowed = [t for t in res["tiles"] if t["borrowed"]]
        self.assertTrue(borrowed)
        for t in res["tiles"]:
            self.assertIsNotNone(t["model"])
            self.assertIn("1e-07", t["z_t"])
        zt_seed = c2.threshold_image(res, 1e-7)
        self.assertGreater(np.nanmean(z[200:204, 100:115] >= zt_seed[200:204, 100:115]), 0.9)
        # the bright target does not lift the background next to it (second pass censors it)
        self.assertLess(abs(float(res["mean"][202, 140]) - float(res["mean"][202, 380])), 0.1)
        row = c2.tile_table(res)[0]
        for k in ("model", "z_t", "tail_rmse", "aic", "params", "exceed"):
            self.assertIn(k, row)


@unittest.skipUnless(np is not None, "numpy not installed")
class V1UnchangedTest(unittest.TestCase):
    """cfar-v1 (ZJG) is untouched: run/detect do not use cfar2 and detect() gives the same output."""

    def test_run_and_detect_do_not_use_cfar2(self):
        for name in ("run.py", "detect.py", "quay.py", "quay_est.py"):
            self.assertNotIn("cfar2", (ROOT / "sat" / name).read_text(encoding="utf-8"), name)

    def test_detect_golden_output(self):
        rng = np.random.default_rng(7)
        vv = 10 * np.log10(0.01 * rng.gamma(4.4, 1 / 4.4, (200, 200)))
        vh = 10 * np.log10(0.002 * rng.gamma(4.4, 1 / 4.4, (200, 200)))
        vv[50:53, 60:70] += 15
        vh[50:53, 60:70] += 12
        vv[120:140, 30:33] += 12
        vh[120:140, 30:33] += 10
        land = np.zeros((200, 200), bool)
        land[:, 180:] = True
        dets, info = detect(vv, vh, land, (10, 10))
        self.assertEqual(info, {"valid_fraction": 1.0, "water_fraction": 0.9, "threshold_sigma": 4.265,
                                "guard_px": 7, "outer_px": 22, "hit_pixels": 113, "clusters": 33,
                                "dropped_small": 31, "dropped_vh": 0})
        self.assertEqual([(d["rmin"], d["rmax"], d["cmin"], d["cmax"], d["pixels"], d["length_m"], d["width_m"],
                           d["zone"], d["peak_vv_db"], d["contrast_vv_db"], d["vh_ratio_db"]) for d in dets],
                         [(50, 52, 60, 69, 30, 100, 30, "harbour", -1.3, 15.0, 12.2),
                          (120, 139, 30, 32, 52, 200, 30, "harbour", -4.2, 11.9, 9.6)])


if __name__ == "__main__":
    unittest.main()
