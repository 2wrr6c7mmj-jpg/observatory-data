"""fix-1: fixed objects and the pier/ship decision in the ports, against the empty-berth baseline (synthetic)."""
from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from sat import chips as satchips
    from sat import fixed, run


def layers(h=40, w=40):
    """Baseline: water p10 -22 dB, a pier edge (rows 10-12, cols 5-15) +20 dB over the water and stable; a berth
    (rows 25-27, cols 5-15) empty in p10, busy in p90."""
    p10 = np.full((h, w), -22.0)
    p90 = np.full((h, w), -18.0)
    p10[10:13, 5:16], p90[10:13, 5:16] = -2.0, 0.0
    p90[25:28, 5:16] = 2.0
    p10, p90 = fixed.smooth3_db(p10), fixed.smooth3_db(p90)      # the baseline is made of 3x3 means
    return {"vv_p10": p10, "vv_p90": p90, "vh_p10": p10 - 7}


def scene(ship=True, pier=True, h=40, w=40):
    vv = np.full((h, w), -22.0)
    if pier:
        vv[10:13, 5:16] = -2.0           # the pier edge, as every day
    if ship:
        vv[25:28, 5:16] = 4.0            # a ship at the empty berth
    return vv


def det(r0, r1, c0, c1, **kw):
    d = {"rmin": r0, "rmax": r1, "cmin": c0, "cmax": c1, "pixels": (r1 - r0 + 1) * (c1 - c0 + 1),
         "vh_ratio_db": 8.0, "length_m": 110, "contrast_vv_db": 20.0}
    d.update(kw)
    return d


@unittest.skipUnless(np is not None, "numpy not installed")
class FixedTest(unittest.TestCase):
    def test_pier_edge_is_a_structure_and_the_ship_is_high(self):
        dets = [det(10, 12, 5, 15), det(25, 27, 5, 15)]
        counts = fixed.classify(dets, scene(), layers(), {"water_vv_p10_median_db": -22.0}, None)
        pier, ship = dets
        self.assertEqual(pier["fixed_excluded"], "structure")
        self.assertIsNone(pier["ship_conf"])
        self.assertGreater(pier["base_db"], 15)
        self.assertLess(abs(pier["rise_db"]), 0.5)
        self.assertEqual((ship["fixed_excluded"], ship["ship_conf"]), ("", "high"))
        self.assertGreater(ship["rise_db"], 20)
        self.assertGreater(ship["rise_z"], 5)
        self.assertEqual((counts["structure"], counts["high"], counts["version"]), (1, 1, "fix-1"))

    def test_a_ship_beside_the_pier_is_not_a_structure(self):
        vv = scene()
        vv[10:13, 5:16] = 6.0            # a ship moored on the pier edge today: +8 dB over the empty state
        d = det(10, 12, 5, 15)
        fixed.classify([d], vv, layers(), {"water_vv_p10_median_db": -22.0}, None)
        self.assertEqual(d["fixed_excluded"], "")
        self.assertEqual(d["ship_conf"], "high")

    def test_a_varying_bright_spot_stays_with_low_confidence(self):
        bl = layers()
        bl["vv_p90"][9:14, 4:17] = 4.0   # p90 - p10 = 6 dB: a berth that is mostly occupied, not a structure
        d = det(10, 12, 5, 15)
        fixed.classify([d], scene(), bl, {"water_vv_p10_median_db": -22.0}, None)
        self.assertEqual((d["fixed_excluded"], d["ship_conf"]), ("", "low"))

    def test_medium_and_small(self):
        vv = scene()
        vv[25:28, 5:16] = -18.0          # +4 dB over the empty berth
        d = det(25, 27, 5, 15)
        fixed.classify([d], vv, layers(), {"water_vv_p10_median_db": -22.0}, None)
        self.assertEqual(d["ship_conf"], "medium")
        d = det(25, 27, 5, 15, length_m=20)     # a strong change but a small target: not high
        fixed.classify([d], scene(), layers(), {"water_vv_p10_median_db": -22.0}, None)
        self.assertEqual(d["ship_conf"], "medium")

    def test_without_a_baseline(self):
        d = det(25, 27, 5, 15)
        counts = fixed.classify([d], scene(), None, None, None)
        self.assertEqual((d["fixed_excluded"], d["ship_conf"], d["basis"]), ("", "medium", "no_baseline"))
        self.assertEqual(counts["no_baseline"], 1)
        d = det(25, 27, 5, 15, contrast_vv_db=6.0)
        fixed.classify([d], scene(), None, None, None)
        self.assertEqual(d["ship_conf"], "low")

    def test_water_level_from_the_mask_when_the_meta_has_none(self):
        water = np.ones((40, 40), bool)
        self.assertEqual(fixed.water_p10(layers(), {}, water), -22.0)
        self.assertIsNone(fixed.water_p10(layers(), {}, None))

    def test_since_and_counting(self):
        self.assertFalse(fixed.active("2026-11-04T10:48:45Z"))
        self.assertTrue(fixed.active("2026-11-05T10:48:45Z"))
        self.assertFalse(run.at_berth({"fixed_excluded": "structure"}))
        self.assertTrue(run.at_berth({"fixed_excluded": ""}))

    def test_chips_carry_the_confidence(self):
        from sat.grid import Grid
        g = Grid(110.0, 21.0, 110.04, 21.04, 400, 400)
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [110.02, 21.02]},
                  "properties": {"n": 1, "zone": "quay", "length_m": 100, "width_m": 20, "fixed_version": "fix-1",
                                 "ship_conf": "high", "rise_db": 9.1, "rise_z": 12.0, "base_db": 1.0}}]
        row = {"acquired_utc": "2026-11-05T10:48:45Z", "color_image": "color/X/v1/a.webp"}
        doc = satchips.scene_doc({"id": "X"}, g, row, feats, (400, 450), None, "X-land-v1")
        c = doc["candidates"][0]
        self.assertEqual((c["ship_conf"], c["rise_db"]), ("high", 9.1))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(np is not None, "numpy not installed")
class SizeGateTest(unittest.TestCase):
    """size-1: from SIZE_SINCE the headline counts take candidates of 50 m and more; smaller ones are counted apart."""

    def test_small_candidates_leave_the_headline(self):
        from sat import run as r
        dets = [{"zone": "quay", "length_m": 30}, {"zone": "quay", "length_m": 120},
                {"zone": "quay", "length_m": 20, "naval_excluded": "civil"}]
        r.mark_small({"acquired_utc": "2026-11-05T10:00:00Z"}, dets)
        self.assertFalse(any(d.get("small_excluded") for d in dets))        # before SIZE_SINCE: unchanged
        r.mark_small({"acquired_utc": r.SIZE_SINCE + "T10:00:00Z"}, dets)
        self.assertEqual([bool(d.get("small_excluded")) for d in dets], [True, False, False])
        self.assertEqual([r.headline(d) for d in dets], [False, True, False])
        self.assertTrue(r.at_berth(dets[0]))                                 # still a naval candidate (framed)
