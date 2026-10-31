"""cfar-v2 (sat/detect2.py): VH sigma0 <= 0 pixels are not "no data".

Real Sentinel-1 GRD over calm sea: after thermal-noise removal a few % to 25% of VH pixels are
sigma0 <= 0 and arrive as code 0 (NaN in dB). Until ``cfar2-pr2.1`` detect_v2 treated them as no
data and dilated them by 3 px, so 3-10% of such pixels removed most of a tile from the VV fit and the
tile was not fitted (probe-raw/ocean 2026-10-30: 0 of 66 tiles with >= 3% fitted). These tests
rebuild that situation with a small synthetic scene (noise-subtracted VH = speckle + zero-mean noise,
<= 0 -> NaN); the real tiles are not in the repository (see the PR for the check on them).
"""
import math
import unittest

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import detect2 as d2

PX = 20.0
N = 384                 # 3 x 3 tiles of 128 px
VV_MEAN = 10 ** (-2.0)  # -20 dB
VH_MEAN = 10 ** (-3.6)  # -36 dB (low wind)


def _scene(seed, noise_sd):
    """VV/VH dB with three targets; VH = speckle + N(0, noise_sd * VH_MEAN), <= 0 -> NaN.

    Returns (vv_db, vh_db, vh_true_db, targets): ``vh_true_db`` has the same VH pixels, with the
    ones <= 0 replaced by their speckle-only value (what a perfect noise estimate would give).
    """
    rng = np.random.default_rng(seed)
    shape = (N, N)
    vv = VV_MEAN * rng.gamma(4.4, 1 / 4.4, shape)
    sp = VH_MEAN * rng.gamma(4.4, 1 / 4.4, shape)
    vh = sp + noise_sd * VH_MEAN * rng.standard_normal(shape)
    targets = {
        "A": (slice(100, 104), slice(100, 112)),    # 240 m, bright in VV and VH
        "B": (slice(300, 302), slice(200, 208)),    # 160 m, VV only -> B
        "short": (slice(250, 252), slice(60, 62)),  # 40 m, VV only -> dropped
    }
    vv[targets["A"]] *= 10 ** 1.5
    vh[targets["A"]] = sp[targets["A"]] * 10 ** 1.2
    sp[targets["A"]] *= 10 ** 1.2
    vv[targets["B"]] *= 10 ** 1.4
    vv[targets["short"]] *= 10 ** 1.3
    with np.errstate(invalid="ignore", divide="ignore"):
        vh_db = np.where(vh > 0, 10 * np.log10(vh), np.nan)
    vh_true_db = np.where(vh > 0, vh_db, 10 * np.log10(sp))
    return 10 * np.log10(vv), vh_db, vh_true_db, targets


def _at(dets, sl):
    r = (sl[0].start + sl[0].stop - 1) / 2
    c = (sl[1].start + sl[1].stop - 1) / 2
    return [d for d in dets if d["rmin"] <= r <= d["rmax"] and d["cmin"] <= c <= d["cmax"]]


@unittest.skipUnless(np is not None, "numpy not installed")
class VhZeroTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for name, noise_sd, seed in (("few", 0.42, 31), ("many", 0.75, 32)):
            vv, vh, vh_true, tg = _scene(seed, noise_sd)
            zero = ~np.isfinite(vh)
            dets, info = d2.detect_v2(vv, vh, None, (PX, PX), None, lia=40.0,
                                      heading_deg=d2.platform_heading("ascending", 22.0))
            dets_t, info_t = d2.detect_v2(vv, vh_true, None, (PX, PX), None, lia=40.0,
                                          heading_deg=d2.platform_heading("ascending", 22.0))
            cls.cases[name] = dict(vv=vv, vh=vh, zero=zero, tg=tg, dets=dets, info=info,
                                   dets_t=dets_t, info_t=info_t)

    def test_scene_reproduces_the_real_shares(self):
        # probe-raw: failures started at 3% and were total above 10%
        self.assertTrue(0.03 <= self.cases["few"]["zero"].mean() <= 0.10, self.cases["few"]["zero"].mean())
        self.assertGreater(self.cases["many"]["zero"].mean(), 0.10)

    def test_old_rule_reproduces_the_failure(self):
        # the pre-fix rule: a VH NaN is no data (both bands NaN there, dilated by edge_px)
        for name, c in self.cases.items():
            vv_old = np.where(c["zero"], np.nan, c["vv"])
            _d, info = d2.detect_v2(vv_old, c["vh"], None, (PX, PX))
            self.assertGreater(info["nodata_band_fraction"], 0.5, name)
            self.assertEqual(sum(1 for t in info["tile_table"] if t["own_model"] is not None), 0, name)

    def test_all_tiles_fitted_and_nothing_dilated(self):
        for name, c in self.cases.items():
            info = c["info"]
            self.assertEqual(info["valid_fraction"], 1.0, name)
            self.assertEqual(info["nodata_band_fraction"], 0.0, name)
            self.assertAlmostEqual(info["vh_zero_fraction"], float(c["zero"].mean()), places=3)
            self.assertEqual(info["tiles_borrowed"], 0, name)
            self.assertEqual(info["tiles_untested"], 0, name)
            self.assertEqual(info["tiles_untested_vh"], 0, name)
            self.assertTrue(all(t["own_model"] is not None for t in info["tile_table"]), name)
            # VV at 1e-3 within the design's acceptance range (1/3..3x; at 1e-4 the 3 targets
            # alone are about 4e-4 of this small scene)
            self.assertTrue(1e-3 / 3 < info["exceedance"]["1e-03"] < 3e-3, (name, info["exceedance"]))

    def test_vv_does_not_depend_on_vh_zeros(self):
        # VV statistics, thresholds, exceedance and candidate pixels are the same as with no VH zeros
        for name, c in self.cases.items():
            self.assertEqual(c["info"]["tile_table"], c["info_t"]["tile_table"], name)
            self.assertEqual(c["info"]["exceedance"], c["info_t"]["exceedance"], name)
            self.assertEqual(c["info"]["hit_pixels"], c["info_t"]["hit_pixels"], name)
            self.assertEqual(c["info"]["clusters"], c["info_t"]["clusters"], name)

    def test_confidence_not_distorted(self):
        for name, c in self.cases.items():
            a, b = _at(c["dets"], c["tg"]["A"]), _at(c["dets"], c["tg"]["B"])
            self.assertEqual(len(a), 1, name)
            self.assertEqual(len(b), 1, name)
            self.assertEqual(a[0]["confidence"], "A", name)
            self.assertEqual(b[0]["confidence"], "B", name)          # VV only: not promoted to A
            self.assertEqual(_at(c["dets"], c["tg"]["short"]), [], name)
            # same decision as with the true (positive) VH values, and the VH contrast within 1 dB
            conf = {(d["rmin"], d["cmin"]): d["confidence"] for d in c["dets"]}
            conf_t = {(d["rmin"], d["cmin"]): d["confidence"] for d in c["dets_t"]}
            self.assertEqual(conf, conf_t, name)
            for d in c["dets"]:
                t = [e for e in c["dets_t"] if (e["rmin"], e["cmin"]) == (d["rmin"], d["cmin"])][0]
                if d["vh_ratio_db"] is not None and t["vh_ratio_db"] is not None:
                    self.assertLess(abs(d["vh_ratio_db"] - t["vh_ratio_db"]), 1.0, name)

    def test_vh_threshold_still_bounded(self):
        # VH exceedance at 1e-3 (tile fits on the positive VH pixels only) within 1/3..3x, and
        # within 1.5x of the same scene with the true positive VH values
        for name, c in self.cases.items():
            v, vt = c["info"]["exceedance_vh"]["1e-03"], c["info_t"]["exceedance_vh"]["1e-03"]
            self.assertTrue(1e-3 / 3 < v < 3e-3, (name, v))
            self.assertLess(max(v / vt, vt / v), 1.5, (name, v, vt))

    def test_target_with_no_positive_vh(self):
        # a VV target whose VH pixels are all <= 0: no crash, kept as B (VV length), not A
        c = self.cases["few"]
        vh = c["vh"].copy()
        vh[c["tg"]["B"]] = np.nan
        dets, _info = d2.detect_v2(c["vv"], vh, None, (PX, PX), None, lia=40.0,
                                   heading_deg=d2.platform_heading("ascending", 22.0))
        b = _at(dets, c["tg"]["B"])
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0]["confidence"], "B")

    def test_datamask_is_no_data(self):
        # dataMask 0 is no data even where VV has a value; its border is dilated by edge_px
        c = self.cases["few"]
        dm = np.ones((N, N), bool)
        dm[:, :20] = False
        _d, info = d2.detect_v2(c["vv"], c["vh"], None, (PX, PX), datamask=dm)
        self.assertAlmostEqual(info["valid_fraction"], 1 - 20 / N, places=4)
        self.assertAlmostEqual(info["nodata_band_fraction"], 23 / N, places=4)
        # VV NaN is no data too (dilated the same way)
        vv = c["vv"].copy()
        vv[:, :20] = np.nan
        _d, info2 = d2.detect_v2(vv, c["vh"], None, (PX, PX))
        self.assertEqual(info2["valid_fraction"], info["valid_fraction"])
        self.assertEqual(info2["nodata_band_fraction"], info["nodata_band_fraction"])

    def test_processing_version(self):
        self.assertEqual(self.cases["few"]["dets"][0]["processing_version"], "cfar2-pr2.1")
        self.assertTrue(math.isfinite(self.cases["few"]["info"]["seconds"]))


if __name__ == "__main__":
    unittest.main()
