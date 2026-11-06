"""fix-1 / fix-2: fixed objects and the pier/ship decision in the ports, against the empty-berth baseline (synthetic)."""
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


# fixed_eval/out/margin.csv (2部、2026-11-06): the 8 candidates of 60 m and more on naval berths that are bright in the
# baseline and did not rise (base >= 8, rise < 3); fix-1 keeps them only because p90 - p10 is 0-2 dB over 3.5.
# (site, lon, lat, length_m, base_db, rise_db, spread_db)
MARGIN_8 = [
    ("QGZ", 119.963946, 35.741315, 187, 12.0, 2.8, 3.5),
    ("ZJG", 110.433462, 21.210922, 68, 30.0, 2.4, 3.5),
    ("ZDH", 122.006476, 30.024359, 118, 13.4, 1.6, 3.6),
    ("QJG", 120.59533, 36.103067, 102, 14.2, 2.0, 3.8),
    ("ZJG", 110.433621, 21.211177, 96, 29.2, 1.8, 4.0),
    ("ZJG", 110.433353, 21.210815, 118, 28.4, 1.2, 4.0),
    ("SYL", 109.545125, 18.223845, 106, 18.8, 2.7, 4.8),
    ("SYL", 109.553173, 18.229887, 88, 24.2, 1.5, 5.5),
]
# design 3.3: the 2 of the 8 that were structures on Esri (dropped by the registry from fix-2 revision 2) and the 6 ships
MARGIN_STRUCTURES = {("QGZ", 119.963946, 35.741315): "QGZ-001", ("QJG", 120.59533, 36.103067): "QJG-001"}
MARGIN_SHIPS = [m for m in MARGIN_8 if (m[0], m[1], m[2]) not in MARGIN_STRUCTURES]
WP10 = -22.0


def flat(base_db, rise_db, spread_db, h=20, w=20):
    """Uniform rasters with these base / rise / spread (uniform, so the 3x3 mean changes nothing)."""
    p10 = np.full((h, w), WP10 + base_db)
    return np.full((h, w), WP10 + base_db + rise_db), {"vv_p10": p10, "vv_p90": p10 + spread_db, "vh_p10": p10 - 7}


@unittest.skipUnless(np is not None, "numpy not installed")
class GuardTest(unittest.TestCase):
    """fix-2's guard (2部の設計書 fixed_eval 4.2 の 3): a structure by fix-1's test is kept on a naval berth when 60 m
    and more (or a piece near such a candidate) and at a shipyard; the decision is recorded either way."""

    def setUp(self):
        self.meta = {"water_vv_p10_median_db": WP10}

    def cands(self, *specs):
        return [{"rmin": 5, "rmax": 7, "cmin": 5, "cmax": 15, "pixels": 33, "vh_ratio_db": 2.0, "contrast_vv_db": 20.0,
                 "lon": lon, "lat": lat, "length_m": ln} for lon, lat, ln in specs]

    def test_versions_by_date(self):
        self.assertIsNone(fixed.version_for("2026-11-04T23:59:59Z"))
        self.assertEqual(fixed.version_for(fixed.FIX2_SINCE + "T00:00:00Z"), "fix-2")
        if fixed.FIX2_SINCE > fixed.SINCE_UTC:
            self.assertEqual(fixed.version_for(fixed.SINCE_UTC + "T10:00:00Z"), "fix-1")

    def test_reasons(self):
        dets = self.cands((110.0, 21.0, 120), (110.0005, 21.0, 40), (110.0030, 21.0, 40), (110.0, 21.002, 200),
                          (110.0, 21.004, 59))
        ctx = [{"berth_class": "naval"}, {"berth_class": "naval"}, {"berth_class": "naval"}, {"berth_class": "civil"},
               {"berth_class": "unknown", "yard_line": True}]
        # yard_line only on a naval shipyard berth (1部の指摘：「軍の造船所と言える根拠が無い」unknown の線には付けない)
        self.assertEqual(fixed.guard_reasons(dets, ctx), [["naval_long"], ["naval_near"], [], [], []])
        ctx[4]["berth_class"] = "naval"
        self.assertEqual(fixed.guard_reasons(dets, ctx)[4], ["yard_line"])
        ctx[4]["berth_class"] = "unknown"
        dets[0]["length_m"] = 59      # no naval candidate of 60 m left: no piece either
        self.assertEqual(fixed.guard_reasons(dets, ctx)[:2], [[], []])
        self.assertEqual(fixed.guard_reasons(dets[:1], [{"yard_site": True, "yard_sector": True}]),
                         [["yard_site", "yard_sector"]])

    def test_structure_is_guarded_or_excluded_and_recorded_either_way(self):
        vv, bl = flat(20.0, 0.5, 2.0)     # a clear structure by fix-1 (spread 2 dB)
        dets = self.cands((110.0, 21.0, 120), (110.0, 21.01, 120), (110.0, 21.02, 30))
        ctx = [{"berth_class": "naval"}, {"berth_class": "civil"}, {"berth_class": "naval"}]
        c1 = fixed.classify([dict(d) for d in dets], vv, bl, self.meta, None)                     # fix-1: all three
        self.assertEqual((c1["structure"], c1["version"]), (3, "fix-1"))
        c2 = fixed.classify(dets, vv, bl, self.meta, None, version="fix-2", guard_ctx=ctx)
        naval, civil, small = dets
        self.assertEqual((naval["fixed_excluded"], naval["fixed_flag"], naval["fixed_guard"], naval["ship_conf"]),
                         ("", "guarded", "naval_long", "low"))
        self.assertEqual((civil["fixed_excluded"], civil["fixed_flag"], civil["fixed_guard"], civil["fixed_guard_class"]),
                         ("structure", "", "", "civil"))
        self.assertEqual((small["fixed_excluded"], small["fixed_guard_class"]), ("structure", "naval"))
        self.assertEqual((c2["structure"], c2["guarded"], c2["guarded_by"]["naval_long"], c2["low"], c2["version"]),
                         (2, 1, 1, 1, "fix-2"))
        self.assertTrue(run.at_berth(naval))
        self.assertFalse(run.at_berth(civil))

    def test_a_ship_moored_for_months_is_kept(self):
        """模擬（設計書 6 の (2)）：a 120 m x 16 m hull +25 dB over the water, always in the baseline and there today."""
        h, w = 60, 60
        p10 = np.full((h, w), WP10)
        p10[20:22, 10:22] = WP10 + 25.0                       # along the quay, 12 x 2 px of 10 m
        p10 = fixed.smooth3_db(p10)
        bl = {"vv_p10": p10, "vv_p90": p10 + 1.0, "vh_p10": p10 - 7}
        vv = np.full((h, w), WP10)
        vv[20:22, 10:22] = WP10 + 25.0
        mk = lambda: [{"rmin": 20, "rmax": 21, "cmin": 10, "cmax": 21, "pixels": 24, "vh_ratio_db": 6.0,  # noqa: E731
                       "contrast_vv_db": 25.0, "lon": 110.0, "lat": 21.0, "length_m": 120}]
        d1, d2 = mk(), mk()
        fixed.classify(d1, vv, bl, self.meta, None)
        fixed.classify(d2, vv, bl, self.meta, None, version="fix-2", guard_ctx=[{"berth_class": "naval"}])
        self.assertEqual(d1[0]["fixed_excluded"], "structure")
        self.assertEqual((d2[0]["fixed_excluded"], d2[0]["fixed_flag"]), ("", "guarded"))

    def test_the_8_margin_candidates_stay_even_at_6_db(self):
        """The 8 of margin.csv on the real berth lines and classes (sat/berth_class): naval, so guarded by length;
        with fix-1's spread threshold raised to 6 dB fix-1 drops all 8 and fix-2 none. With the committed registry
        (sat/fixed_registry) as the daily run uses it, the 6 ships (MARGIN_SHIPS) are still never dropped, and only
        the 2 structures that 2部 saw on Esri (QGZ 187 m = QGZ-001, QJG 102 m = QJG-001) are dropped by the registry."""
        from sat import berths, quay_lines
        from sat import fixed_registry as freg
        sites = {s["id"]: s for s in run.load_sites()}
        day = fixed.FIX2_SINCE + "T10:00:00Z"
        self.assertEqual(len(MARGIN_SHIPS), 6)
        for site_id, lon, lat, ln, base, rise, spread in MARGIN_8:
            with self.subTest(site=site_id, lon=lon):
                s = sites[site_id]
                bt = berths.Berths(s, quay_lines.load(site_id, int(s.get("quay_lines_version", quay_lines.LINES_VERSION))),
                                   naval=True)
                (_way, cl), = bt.nearest_line([(lon, lat)], bt.keep_m)
                self.assertEqual(cl, "naval")
                reg = freg.active(freg.load(site_id), day)
                ship = (site_id, lon, lat) in {(m[0], m[1], m[2]) for m in MARGIN_SHIPS}
                vv, bl = flat(base, rise, spread)
                for thr in (3.5, 6.0):
                    d1, d2, d3 = (self.cands((lon, lat, ln)) for _ in range(3))
                    fixed.classify(d1, vv, bl, self.meta, None, cfg={"spread_db": thr})
                    fixed.classify(d2, vv, bl, self.meta, None, cfg={"spread_db": thr}, version="fix-2",
                                   guard_ctx=[{"berth_class": cl}])
                    fixed.classify(d3, vv, bl, self.meta, None, cfg={"spread_db": thr}, version="fix-2",
                                   guard_ctx=[{"berth_class": cl}], registry=reg)
                    self.assertEqual(d2[0]["fixed_excluded"], "")
                    if ship:
                        self.assertEqual((d3[0]["fixed_excluded"], d3[0]["fixed_registry_id"]), ("", ""))
                    else:
                        self.assertEqual((d3[0]["fixed_excluded"], d3[0]["registry_over_guard"]), ("registry", True))
                        self.assertEqual(d3[0]["fixed_registry_id"], MARGIN_STRUCTURES[(site_id, lon, lat)])
                    if thr == 6.0:
                        self.assertEqual(d1[0]["fixed_excluded"], "structure")
                        self.assertEqual((d2[0]["fixed_flag"], d2[0]["fixed_guard"]), ("guarded", "naval_long"))
                        if ship:
                            self.assertEqual((d3[0]["fixed_flag"], d3[0]["fixed_guard"]), ("guarded", "naval_long"))

    def test_run_context_from_the_berth_lines_and_sectors(self):
        """run.fixed_guard_ctx: the naval-1 class of the nearest line, the shipyard line and sector, on a real site."""
        site = next(s for s in run.load_sites() if s["id"] == "ZJG")
        q = run.QuayContext(site, None, None)
        dets = [{"zone": "quay", "lon": 110.433462, "lat": 21.210922, "sector": "A"},
                {"zone": "quay", "lon": 110.30, "lat": 21.10, "sector": "B"}]
        ctx = run.fixed_guard_ctx(q, site, {"acquired_utc": fixed.FIX2_SINCE + "T10:00:00Z"}, dets)
        self.assertEqual([c["berth_class"] for c in ctx], ["naval", "unknown"])
        self.assertFalse(any(c["yard_site"] or c["yard_sector"] or c["yard_line"] for c in ctx))
        yard = {**site, "kind": "yard", "sectors": [{**sc, "role": "dock"} for sc in site.get("sectors") or []]}
        ctx = run.fixed_guard_ctx(run.QuayContext(yard, None, None), yard, {"acquired_utc": fixed.FIX2_SINCE + "T10:00:00Z"},
                                  dets[:1])
        self.assertTrue(ctx[0]["yard_site"] and ctx[0]["yard_sector"])
        old = run.fixed_guard_ctx(q, site, {"acquired_utc": "2026-11-01T10:00:00Z"}, dets[:1])   # naval-1 not applied
        self.assertIsNone(old[0]["berth_class"])

    def test_yard_lines_of_the_class_files(self):
        from sat import berths
        sites = {s["id"]: s for s in run.load_sites()}
        b = berths.Berths(sites["MAW"], {"ways": []})
        self.assertTrue(b.yard_ways())
        self.assertFalse(berths.Berths(sites["QGZ"], {"ways": []}).yard_ways())

    def test_series_break_and_settings(self):
        br = [b for b in run.series_breaks(run.load_sites()) if b.get("kind") == "fixed" and b.get("site_id") == "ZJG"]
        st = fixed.settings()
        self.assertEqual(st["version"], "fix-2")
        self.assertEqual(st["guard"]["naval_len_m"], 60.0)
        self.assertEqual(br[-1]["to_version"], "fix-2")
        self.assertEqual(br[-1]["date_utc"], fixed.FIX2_SINCE)


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
