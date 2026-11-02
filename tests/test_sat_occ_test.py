"""occ-1 (sat/occ_test.py): the change test against the empty-quay baseline, and its evaluation (sat/eval_occ.py)."""
import io
import unittest

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, color, eval_occ, occ_test as o, quay, s2_labels
    from tests.test_sat_quay import scene_parts


@unittest.skipUnless(np is not None, "numpy not installed")
class SpecialFunctionsTest(unittest.TestCase):
    # reference values: scipy.special / scipy.stats (1.18)
    def test_betainc(self):
        x = np.array([0.01, 0.3, 0.5, 0.9])
        want = [1.40367894e-05, 0.455750304, 0.883165651, 0.999996408]
        got = o.betainc(3.5, 7.2, x)
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_f_sf(self):
        got = o.f_sf(np.array([0.2, 1.0, 2.0, 5.0, np.nan]), 24, 300)
        np.testing.assert_allclose(got[:4], [0.999988906, 0.465975656, 4.27396101e-03, 5.05374175e-12], rtol=1e-6)
        self.assertTrue(np.isnan(got[4]))

    def test_chi2_and_gamma(self):
        np.testing.assert_allclose(o.chi2_sf(np.array([1.0, 9.4877, 30.0]), 4), [0.909796, 0.05, 4.8944e-06], rtol=1e-4)
        self.assertAlmostEqual(o.gamma_ppf(0.1, 12.0), 7.829342, places=5)

    def test_reference_factor_recovers_the_mean(self):
        rng = np.random.default_rng(3)
        L, mu, n = 12.0, 0.05, 24
        stack = rng.gamma(L, mu / L, (n, 20000))
        k, lref = o.ref_factor(10, L, n)
        est = np.percentile(stack, 10, axis=0) * k
        self.assertAlmostEqual(float(est.mean()) / mu, 1.0, delta=0.06)   # p10 of 24 scenes: a few % bias
        # the reference's equivalent looks from the asymptotic quantile variance: close, on the safe side (spread
        # not underestimated by more than 20 %)
        r = float(est.var() / est.mean() ** 2) * lref
        self.assertTrue(0.6 < r < 1.2, r)

    def test_temporal_enl_inverts_the_spread(self):
        for L in (2.0, 5.0, 12.0):
            r = 10 * np.log10(o.gamma_ppf(0.9, L) / o.gamma_ppf(0.1, L))
            got = o.temporal_enl(np.array([-20.0]), np.array([-20.0 + r]), 30.0)[0]
            self.assertAlmostEqual(got, L, delta=0.1 * L)
        self.assertEqual(o.temporal_enl(np.array([-20.0]), np.array([-20.0]), 12.0)[0], 12.0)  # capped


@unittest.skipUnless(np is not None, "numpy not installed")
class TestsAreCalibratedTest(unittest.TestCase):
    def test_omnibus_size_and_power(self):
        rng = np.random.default_rng(0)
        a = rng.gamma(12, 1 / 12, (5, 100000))
        b = rng.gamma(12, 1 / 12, (5, 100000))
        self.assertAlmostEqual(float(np.mean(o.omnibus(a, b, 12) < 0.05)), 0.05, delta=0.006)
        self.assertAlmostEqual(float(np.mean(o.omnibus(a[:2], None, 12) < 0.05)), 0.05, delta=0.006)  # two dates
        a[4] *= 4
        self.assertGreater(float(np.mean(o.omnibus(a, b, 12) < 0.05)), 0.8)

    def test_pixel_pvalues_uniform_without_change_small_with_a_ship(self):
        rng = np.random.default_rng(5)
        L, n, shape = 12.0, 24, (60, 60)
        mu_vv, mu_vh = 10 ** (-2.0), 10 ** (-2.6)
        vv_st = 10 * np.log10(rng.gamma(L, mu_vv / L, (n,) + shape))
        vh_st = 10 * np.log10(rng.gamma(L, mu_vh / L, (n,) + shape))
        base = {"vv_p10": np.percentile(vv_st, 10, 0), "vv_p90": np.percentile(vv_st, 90, 0),
                "vh_p10": np.percentile(vh_st, 10, 0)}
        vv = 10 * np.log10(rng.gamma(L, mu_vv / L, shape))
        vh = 10 * np.log10(rng.gamma(L, mu_vh / L, shape))
        vv[20:30, 20:23] += 15.0
        vh[20:30, 20:23] += 12.0
        pv = o.pvalues(vv, vh, base, {"enl": L}, n_scenes=n, presmoothed=True)
        ship = np.zeros(shape, bool)
        ship[20:30, 20:23] = True
        self.assertLess(float(np.nanmax(pv["p"][ship])), 1e-4)
        self.assertLess(float(np.mean(pv["p"][~ship] < 0.01)), 0.03)   # about the nominal 1 %
        self.assertAlmostEqual(float(np.nanmedian(pv["p"][~ship])), 0.5, delta=0.12)

    def test_low_layer_replaces_a_busy_p10(self):
        base = {"vv_p10": np.array([[-5.0, -20.0]]), "vv_p5": np.array([[-20.0, -20.5]])}
        L = np.full((1, 2), 12.0)
        mu, _ = o.reference(base, "vv", {**o.DEFAULTS}, 24, L)
        self.assertAlmostEqual(10 * np.log10(mu[0, 0]) - 10 * np.log10(mu[0, 1]), 0.0, delta=1.0)  # p5 at the busy pixel
        mu1, _ = o.reference({"vv_p10": base["vv_p10"]}, "vv", {**o.DEFAULTS}, 24, L)
        self.assertGreater(10 * np.log10(mu1[0, 0]), -5.0)                                         # bl-1: p10 only


@unittest.skipUnless(np is not None, "numpy not installed")
class SectionOccupancyTest(unittest.TestCase):
    def setUp(self):
        self.g, self.mask, self.lines, self.cvv, self.cvh, self.base = scene_parts()
        self.geo = quay.Geometry(self.g, self.mask, self.lines)
        self.secs = s2_labels.sections(self.geo, section_m=50.0)

    def scene(self, ship=True, raft=False):
        rng = np.random.default_rng(7)
        L = 12.0
        lin = lambda db: 10 ** (db / 10)
        vv = 10 * np.log10(rng.gamma(L, lin(self.cvv) / L))
        vh = 10 * np.log10(rng.gamma(L, lin(self.cvh) / L))
        if ship:
            vv[50:61, 11:13] = 5.0      # 110 m alongside the coast, rows 50-60
            vh[50:61, 11:13] = -5.0
        if raft:
            vv[50:61, 13:16] = 5.0
            vh[50:61, 13:16] = -5.0
        return vv, vh

    def occ(self, vv, vh):
        base = {**self.base, "vv_p90": self.base["vv_p10"] + 10 * np.log10(o.gamma_ppf(0.9, 12) / o.gamma_ppf(0.1, 12))}
        pv = o.pvalues(vv, vh, base, {"enl": 12.0}, n_scenes=24, presmoothed=True)
        return o.section_occupancy(self.geo, self.secs, pv["p"])

    def coast_sections(self, res):
        out = {}
        for s in self.secs:
            if s["kind"] != "coastline":
                continue
            rows = [self.g.to_pixel(lo, la)[0] for lo, la in s["line"]]
            out[s["section_id"]] = (min(rows), max(rows), res[s["section_id"]])
        return out

    def test_ship_section_occupied_others_not(self):
        res = self.coast_sections(self.occ(*self.scene()))
        hit = [v for lo, hi, v in res.values() if lo <= 55 <= hi]
        free = [v for lo, hi, v in res.values() if hi < 40 or lo > 70]
        self.assertTrue(hit and all(v["p_occ"] > 0.9 for v in hit), hit)
        self.assertTrue(free and all(v["p_occ"] < 0.2 for v in free), free)
        self.assertTrue(all(v["c"] == 1 for v in hit))

    def test_empty_scene_low_everywhere(self):
        res = self.occ(*self.scene(ship=False))
        ps = [v["p_occ"] for v in res.values() if v["p_occ"] is not None]
        self.assertTrue(ps)
        self.assertLess(max(ps), 0.5)

    def test_rafted_ship_reaches_further(self):
        one = [v for lo, hi, v in self.coast_sections(self.occ(*self.scene())).values() if lo <= 55 <= hi]
        two = [v for lo, hi, v in self.coast_sections(self.occ(*self.scene(raft=True))).values() if lo <= 55 <= hi]
        self.assertGreater(max(v["reach_m"] for v in two), max(v["reach_m"] for v in one))
        self.assertEqual(max(v["c"] for v in two), 2)


@unittest.skipUnless(np is not None, "numpy not installed")
class EvalTest(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(eval_occ.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]), 1.0)
        self.assertEqual(eval_occ.auc([0.5, 0.5], [1, 0]), 0.5)
        self.assertIsNone(eval_occ.auc([0.5], [1]))
        self.assertEqual(eval_occ.brier([1.0, 0.0], [1, 1]), 0.5)
        self.assertEqual(eval_occ.confusion([0, 1, 1], [0, 1, 2]), {"0->0": 1, "1->1": 1, "1->2": 1})

    def test_display_image_round_trip(self):
        from PIL import Image
        vv = np.array([[-22.0, -11.0, 0.0, 3.0]])
        vh = np.array([[-28.0, -17.0, -6.0, -6.0]])
        rgb = color.false_color(vv, vh, filtered=False)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        gv, gh = eval_occ.decode_display(buf.getvalue(), (4, 1))
        np.testing.assert_allclose(gv[0], [-22.0, -11.0, 0.0, 0.0], atol=0.1)   # clipped at 0 dB
        np.testing.assert_allclose(gh[0], [-28.0, -17.0, -6.0, -6.0], atol=0.1)
        nod = io.BytesIO()
        Image.fromarray(np.zeros((1, 2, 3), np.uint8)).save(nod, "PNG")
        self.assertTrue(np.isnan(eval_occ.decode_display(nod.getvalue(), (2, 1))[0]).all())   # (0, 0, 0) = no data

    def test_bl2_low_layers(self):
        rng = np.random.default_rng(1)
        st = [rng.normal(-20, 1, (3, 3)) for _ in range(30)]
        keep = np.ones((3, 3), bool)
        low = baseline.compute_low(st, st, keep)
        self.assertEqual(sorted(low), sorted(baseline.BL2_LAYERS))
        self.assertTrue(np.all(low["vv_p0"] <= low["vv_p5"]))
        np.testing.assert_allclose(low["vv_p0"], np.min(np.stack(st), 0))


if __name__ == "__main__":
    unittest.main()
