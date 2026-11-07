"""occ-2 (sat/occ2.py, sat/occ2_schema.py): the presence test against the scene's own empty-quay distribution.

The schema tests are pure Python (the collect workflows have no numpy); the numeric tests build a 100 x 100 synthetic
grid (no site baseline is read) and run in seconds. 設計書_occ-2 §6 occ2-1 (a)-(e).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sat import occ2_schema as sch

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, occ2, quay, s2_labels
    from sat.grid import Grid
    from sat.osm_mask import LAND, STRUCTURE, WATER

SITE = {"id": "TOC", "bbox": [110.0, 21.0, 110.0096, 21.009], "pixel_m": 10, "mask_version": 1, "enabled": True,
        "center_lat": 21.0045, "baseline": {"orbits": [157], "min_scenes": 24}}
ACQ = {"site_id": "TOC", "acquired_utc": "2026-03-01T10:48:00Z", "relative_orbit": 157, "orbit_direction": "ascending"}


class SchemaTest(unittest.TestCase):
    def test_fields_and_settings(self):
        for k in ("site_id", "acquired_utc", "section_id", "p_test", "p_occ", "t_stat", "z_seed_ok", "stretch_len_m", "blind",
                  "blind_prior", "ref_kind", "edge_offset_m", "edge_source", "delta_vv_db", "sigma_vh_db", "enl_vv", "pool_n",
                  "calib_source", "baseline_period"):
            self.assertIn(k, sch.FIELDS)
        self.assertEqual(sch.OCC_VERSION, "occ-2")
        st = sch.settings()
        self.assertEqual(st["version"], "occ-2")
        self.assertEqual(st["depth_m"], [5.0, 25.0])
        self.assertEqual(st["calib"]["b"], 1.0)

    def test_p_occ_placeholder_calibration(self):
        # a = logit(0.5) = 0, b = 1, c = 0: p_test 0.01 -> sigmoid(2)
        self.assertAlmostEqual(sch.p_occ_from_test(0.01, 100.0), 1 / (1 + 2.718281828 ** -2), places=4)
        self.assertAlmostEqual(sch.p_occ_from_test(1.0, 0.0), 0.5, places=6)
        self.assertIsNone(sch.p_occ_from_test(None, 0.0))
        # the prior moves a; the length term only with c
        self.assertLess(sch.p_occ_from_test(0.1, 100.0, {"prior": 0.3}), sch.p_occ_from_test(0.1, 100.0, {"prior": 0.5}))
        self.assertGreater(sch.p_occ_from_test(0.1, 150.0, {"c": 0.5}), sch.p_occ_from_test(0.1, 30.0, {"c": 0.5}))
        # blind: the mixture with the prior
        p = sch.p_occ_from_test(1.0, 0.0, blind=True, blind_prior=0.9, lam=0.5)
        self.assertAlmostEqual(p, 0.5 * 0.5 + 0.5 * 0.9, places=6)

    def test_rows_round_trip_and_merge(self):
        r1 = {k: None for k in sch.FIELDS}
        r1.update(site_id="QGZ", acquired_utc="2026-02-16T09:56:08Z", relative_orbit=171, orbit_direction="ascending",
                  section_id="219848841:L:00", kind="pier", face="toward", occ_version="occ-2", p_test=0.00012345, p_occ=0.8125,
                  t_stat=23.456, z_seed_ok=1, stretch_len_m=130.0, stretch_depth_m=35.0, blind=1, blind_prior=0.9, ref_kind="spatial",
                  edge_offset_m=-15.0, edge_source="ridge", delta_vv_db=4.33, delta_vh_db=4.4, sigma_vv_db=2.77, sigma_vh_db=3.06,
                  enl_vv=15.9, enl_vh=6.6, pool_n=112, calib_source="none", baseline_period="baseline")
        r2 = {**r1, "section_id": "219848841:L:01", "p_test": None, "p_occ": None, "t_stat": None, "blind": 0, "blind_prior": None}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            paths = sch.save_rows(out, "QGZ", [r1, r2])
            self.assertEqual([p.name for p in paths], ["2026-02.csv"])
            back = sch.read_rows(paths[0])
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0]["section_id"], "219848841:L:00")
            self.assertAlmostEqual(back[0]["p_test"], 0.00012345, places=9)
            self.assertEqual(back[0]["blind"], 1)
            self.assertEqual(back[0]["relative_orbit"], 171)
            self.assertIsNone(back[1]["p_test"])
            self.assertIsNone(back[1]["blind_prior"])
            # a new row of the same key replaces, another scene adds a file
            r3 = {**r1, "p_occ": 0.5}
            r4 = {**r1, "acquired_utc": "2026-03-01T09:56:08Z"}
            sch.save_rows(out, "QGZ", [r3, r4])
            by = sch.load_site_rows(out, "QGZ")
            self.assertEqual(sorted(by), ["2026-02-16T09:56:08Z", "2026-03-01T09:56:08Z"])
            self.assertAlmostEqual(by["2026-02-16T09:56:08Z"]["219848841:L:00"]["p_occ"], 0.5)
            self.assertEqual(len(by["2026-02-16T09:56:08Z"]), 2)

    def test_a1_targets(self):
        self.assertEqual(sch.a1_target("QGZ", "1564921421:R:00"), "A")
        self.assertEqual(sch.a1_target("QGZ", "15387629:L:06"), "D")
        self.assertEqual(sch.a1_target("QGZ", "219848841:L:00"), "")
        self.assertEqual(sch.a1_target("SLP", "x"), "")


# ---------------------------------------------------------------- synthetic grid
def scene_parts(line_shift_px: int = 0):
    """100 x 100 px: land west of col 10 (-14 dB) with the quay's apron and wall at cols 8-9 (-5 dB: the baseline's ridge),
    water east (-20 dB), three piers (STRUCTURE, 3 rows, cols 10-40, -6 dB) at rows 20, 45, 70. The coastline is drawn at
    col 10 + ``line_shift_px`` (a line off the true edge, type A'). The 3x3 mean of the baseline puts the first water pixel
    at about -9.5 dB (clear under the -8 dB blind rule, as a real quay). Returns (grid, mask, lines, clean_vv, clean_vh, base)."""
    g = Grid.for_site(SITE)
    h, w = g.height, g.width
    lon = lambda c: g.west + c / g.width * (g.east - g.west)
    lat = lambda r: g.north - r / g.height * (g.north - g.south)
    mask = np.zeros((h, w), np.uint8)
    mask[:, :10] = LAND
    ways = [{"id": 1, "kind": "coastline", "closed": False,
             "coords": [[lon(10 + line_shift_px), lat(h + 1)], [lon(10 + line_shift_px), lat(-1)]]}]
    for k, r in enumerate((20, 45, 70)):
        mask[r:r + 3, 10:41] = STRUCTURE
        ways.append({"id": 2 + k, "kind": "pier", "closed": False, "coords": [[lon(10), lat(r + 1.5)], [lon(40.5), lat(r + 1.5)]]})
    clean_vv = np.where(mask == LAND, -14.0, np.where(mask == STRUCTURE, -6.0, -20.0))
    clean_vv[:, 8:10] = -5.0
    clean_vh = clean_vv - 6.0
    base = {"vv_p10": baseline.smooth_db(clean_vv), "vh_p10": baseline.smooth_db(clean_vh)}
    base["vv_p90"] = base["vv_p10"] + 2.0
    return g, mask, {"ways": ways}, clean_vv, clean_vh, base


def noisy(clean_vv, clean_vh, seed=0, sd=0.5):
    rng = np.random.default_rng(seed)
    return clean_vv + rng.normal(0, sd, clean_vv.shape), clean_vh + rng.normal(0, sd, clean_vh.shape)


def coast_section(secs, row):
    """The coastline section (way 1) whose samples cover ``row``."""
    for s in secs:
        if s["way_id"] == 1 and s["lat_rows"][0] <= row <= s["lat_rows"][1]:
            return s
    raise AssertionError("no coastline section at row %d" % row)


@unittest.skipUnless(np is not None, "numpy not installed")
class Occ2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.mask, cls.lines, cls.cvv, cls.cvh, cls.base = scene_parts()
        cls.geom = quay.Geometry(cls.g, cls.mask, cls.lines)
        cls.secs = s2_labels.sections(cls.geom)
        cls._rows_of(cls.geom, cls.secs, cls.g)

    @staticmethod
    def _rows_of(geom, secs, g):
        for s in secs:
            pr = geom.profiles[s["profile"]]
            r, _ = g.to_pixel(np.asarray(pr["lon"])[s["idx"]], np.asarray(pr["lat"])[s["idx"]])
            s["lat_rows"] = (float(np.min(r)), float(np.max(r)))

    def run_scene(self, vv, vh, geom=None, secs=None, lines=None, mask=None, base=None, **kw):
        return occ2.run_scene(geom or self.geom, self.mask if mask is None else mask, secs or self.secs, base or self.base, vv, vh, ACQ,
                              lines=lines or self.lines, **kw)

    def test_sections_and_pool(self):
        self.assertGreaterEqual(len(self.secs), 12)
        vv, vh = noisy(self.cvv, self.cvh)
        rows, rec = self.run_scene(vv, vh)
        self.assertEqual(rec["status"], "ok")
        self.assertGreaterEqual(rec["pool_n"], 10)
        self.assertEqual(rec["t_null"], "pool_sections")
        self.assertEqual(len(rows), len([s for s in self.secs if s["length_m"] >= 50]))
        for r in rows:
            self.assertEqual(r["occ_version"], "occ-2")
            self.assertEqual(set(r), set(sch.FIELDS))

    def test_a_uniform_lift_leaves_p_test_unchanged(self):
        """(a) an empty scene lifted 5 dB (wind) gives the same p_test, and no section is significant beyond chance."""
        vv, vh = noisy(self.cvv, self.cvh)
        rows0, rec0 = self.run_scene(vv, vh)
        rows5, rec5 = self.run_scene(vv + 5.0, vh + 5.0)
        self.assertAlmostEqual(rec5["delta_vv_db"] - rec0["delta_vv_db"], 5.0, places=2)
        self.assertAlmostEqual(rec5["delta_vh_db"] - rec0["delta_vh_db"], 5.0, places=2)
        p0 = [r["p_test"] for r in rows0]
        p5 = [r["p_test"] for r in rows5]
        for a, b, r in zip(p0, p5, rows0):
            if r["blind"]:                 # a blind section without its own empty days has no test (ref_kind prior)
                self.assertIsNone(a)
                self.assertEqual(r["ref_kind"], "prior")
                continue
            self.assertIsNotNone(a)
            self.assertAlmostEqual(a, b, places=4)
        p0 = [p for p in p0 if p is not None]
        self.assertGreaterEqual(len(p0), 10)
        self.assertLessEqual(sum(1 for p in p0 if p < 0.05) / len(p0), 0.2)
        self.assertLessEqual(sum(1 for p in p0 if p < 0.01), 1)

    def test_b_ship_150m_is_significant(self):
        """(b) a 150 m x 20 m ship along the coast: p_test < 0.01, the stretch about its length, the seed found."""
        vv, vh = noisy(self.cvv, self.cvh)
        vv[50:65, 10:12] = 5.0
        vh[50:65, 10:12] = -1.0
        rows, _ = self.run_scene(vv, vh)
        sec = coast_section(self.secs, 57)
        r = next(r for r in rows if r["section_id"] == sec["section_id"])
        self.assertLess(r["p_test"], 0.01)
        self.assertEqual(r["z_seed_ok"], 1)
        self.assertGreaterEqual(r["stretch_len_m"], 100.0)
        self.assertLessEqual(r["stretch_len_m"], 200.0)
        self.assertGreater(r["p_occ"], 0.8)
        # the pier sections are untouched
        others = [x["p_test"] for x in rows if x["kind"] == "pier" and x["p_test"] is not None]
        self.assertLessEqual(sum(1 for p in others if p < 0.01), 1)

    def test_c_one_pixel_bright_edge_is_not_a_ship(self):
        """(c) the quay's edge pixel alone lights up (type F): the 3x3 mean raises the first water pixel only, the two-pixel
        statistic stays low."""
        vv, vh = noisy(self.cvv, self.cvh)
        vv[50:65, 9] = 25.0
        vh[50:65, 9] = 19.0
        rows, _ = self.run_scene(vv, vh)
        sec = coast_section(self.secs, 57)
        r = next(r for r in rows if r["section_id"] == sec["section_id"])
        self.assertGreater(r["p_test"], 0.01)
        self.assertEqual(r["stretch_len_m"], 0.0)

    def test_d_line_off_the_edge_recovered_by_the_ridge(self):
        """(d) the coastline drawn 20 m waterward of the wall (type A'): the band off the mask's edge misses the ship, the band
        off the baseline's ridge (the wall) finds it."""
        g, mask, lines, cvv, cvh, base = scene_parts(line_shift_px=2)
        geom = quay.Geometry(g, mask, lines)
        secs = s2_labels.sections(geom)
        self._rows_of(geom, secs, g)
        vv, vh = noisy(cvv, cvh)
        vv[50:65, 10:12] = 5.0
        vh[50:65, 10:12] = -1.0
        sec = coast_section(secs, 57)
        rows, rec = occ2.run_scene(geom, mask, secs, base, vv, vh, ACQ, lines=lines)
        r = next(r for r in rows if r["section_id"] == sec["section_id"])
        self.assertEqual(r["edge_source"], "ridge")
        self.assertLessEqual(r["edge_offset_m"], -20.0)
        self.assertLess(r["p_test"], 0.01)
        self.assertGreaterEqual(r["stretch_len_m"], 100.0)
        rows_no, _ = occ2.run_scene(geom, mask, secs, base, vv, vh, ACQ, lines=lines, cfg={"ridge_rel_db": 99.0})
        r_no = next(r for r in rows_no if r["section_id"] == sec["section_id"])
        self.assertEqual(r_no["edge_source"], "mask")
        self.assertGreater(r_no["p_test"], 0.01)

    def _blind_setup(self):
        base = {k: v.copy() for k, v in self.base.items()}
        cvv = self.cvv.copy()
        cvh = self.cvh.copy()
        cvv[17:20, 12:27] = 5.0     # a ship always at the north side of the first pier, 150 m
        cvh[17:20, 12:27] = -1.0
        base["vv_p10"], base["vh_p10"] = baseline.smooth_db(cvv), baseline.smooth_db(cvh)
        return base, cvv, cvh

    def test_e_blind_section(self):
        """(e) a berth taken in the baseline (its p10 is a ship) is blind by the spatial rule (its inner p10 median above the p90 of
        the clear sections of its face and kind). Default (occ2-2b): no test without its own empty days, p_occ = blind_prior, never
        "absent". Option blind_test spatial (the design's §2.3): the p90 reference finds the ship of the day."""
        base, cvv, cvh = self._blind_setup()
        vv, vh = noisy(cvv, cvh)
        rows, rec = self.run_scene(vv, vh, base=base, blind_prior={s["section_id"]: 0.9 for s in self.secs})
        blind = [r for r in rows if r["blind"] == 1]
        self.assertGreaterEqual(len(blind), 1)
        r = next(r for r in blind if r["kind"] == "pier" and r["section_id"].startswith("2:R"))   # side R is the north side here
        self.assertEqual(r["ref_kind"], "prior")
        self.assertIsNone(r["p_test"])
        self.assertEqual(r["p_occ"], 0.9)
        self.assertIsNotNone(r["t_stat"])             # the statistic is still computed (the section reference uses it)
        self.assertIsNone(r["edge_offset_m"])        # no ridge search on a blind section
        rows_s, _ = self.run_scene(vv, vh, base=base, blind_prior={}, cfg={"blind_test": "spatial"})
        r_s = next(x for x in rows_s if x["section_id"] == r["section_id"])
        self.assertEqual(r_s["ref_kind"], "spatial")
        self.assertLess(r_s["p_test"], 0.01)
        self.assertGreaterEqual(r_s["stretch_len_m"], 100.0)
        self.assertLessEqual(r_s["p_occ"], 0.75 + 1e-9)   # mixed with the prior 0.5, lambda 0.5
        # the same section against its own p10 (the blind rules switched off): nothing to see
        rows_t, _ = self.run_scene(vv, vh, base=base, cfg={"blind_p10_db": 99.0, "blind_ref_min": 10 ** 6})
        r_t = next(x for x in rows_t if x["section_id"] == r["section_id"])
        self.assertEqual(r_t["ref_kind"], "time")
        self.assertGreater(r_t["p_test"], 0.05)

    def test_section_reference_shifts_the_null_centre(self):
        """occ2-2b: a section that is brighter than the pool on every day (a permanent structure not in the baseline) tests "present"
        against the pool; its own empty days (LOO) give its offset and the test comes back to the null. A real ship on another day
        stays significant after the shift."""
        sec = coast_section(self.secs, 57)
        pr = self.geom.profiles[sec["profile"]]
        rows_by, recs_by = {}, {}
        scenes = {}
        for k, seed in enumerate((1, 2, 3, 4)):
            vv, vh = noisy(self.cvv, self.cvh, seed=seed)
            vv[50:65, 10:12] += 9.0      # the permanent bias on the section's band: 150 m, two pixels deep
            vh[50:65, 10:12] += 9.0
            if k == 3:
                vv[52:63, 10:12] = 5.0   # and on the last day a ship
                vh[52:63, 10:12] = -1.0
            acq = {**ACQ, "acquired_utc": f"2026-03-0{k + 1}T10:48:00Z"}
            rows, rec = occ2.run_scene(self.geom, self.mask, self.secs, self.base, vv, vh, acq, lines=self.lines)
            rows_by[acq["acquired_utc"]], recs_by[acq["acquired_utc"]] = rows, rec
            scenes[acq["acquired_utc"]] = next(r for r in rows if r["section_id"] == sec["section_id"])
        acqs = sorted(rows_by)
        for a in acqs[:3]:
            self.assertLess(scenes[a]["p_test"], 0.05, a)      # the bias alone tests present against the pool
        absent = {sec["section_id"]: set(acqs[:3])}             # the owner: empty on the first three days
        off = occ2.section_offsets(rows_by, recs_by, absent)
        self.assertEqual(set(off), {(a, sec["section_id"]) for a in acqs})   # LOO: each of the three has two others; the fourth has three
        self.assertGreater(off[(acqs[0], sec["section_id"])], 5.0)
        n = occ2.apply_section_reference(rows_by, recs_by, off)
        self.assertEqual(n, 4)
        for a in acqs[:3]:
            self.assertEqual(scenes[a]["ref_kind"], "section")
            self.assertGreater(scenes[a]["p_test"], 0.05, a)   # back to the null
            self.assertIn("ref-own-absent-days", scenes[a]["calib_source"])
        self.assertLess(scenes[acqs[3]]["p_test"], 0.01)       # the ship still stands out
        # a section with fewer empty days keeps the pool reference
        off1 = occ2.section_offsets(rows_by, recs_by, {sec["section_id"]: set(acqs[:2])})
        self.assertNotIn((acqs[0], sec["section_id"]), off1)
        self.assertIn((acqs[3], sec["section_id"]), off1)

    def test_depth_mode_w15_and_no_pool(self):
        vv, vh = noisy(self.cvv, self.cvh)
        vv[50:65, 10:12] = 5.0
        vh[50:65, 10:12] = -1.0
        rows, rec = self.run_scene(vv, vh, cfg={"depth_mode": "w15"})
        sec = coast_section(self.secs, 57)
        r = next(r for r in rows if r["section_id"] == sec["section_id"])
        self.assertLess(r["p_test"], 0.01)
        # too few sections for a pool: rows without a test, the record says so
        few = [s for s in self.secs if s["way_id"] == 1][:1]
        rows, rec = self.run_scene(vv, vh, secs=few, cfg={"pool_min_samples": 10 ** 6})
        self.assertEqual(rec["status"], "no_pool")
        self.assertTrue(all(r["p_test"] is None and r["p_occ"] is None for r in rows))

    def test_eval_occ2_helpers(self):
        from sat import eval_occ2 as e2, occ2_schema as sch2
        self.assertAlmostEqual(sch2.p_occ_from_test(0.05, 0.0, e2.NO_LABEL_CALIB), 0.5, places=9)   # anchored at the test's alpha
        self.assertGreater(sch2.p_occ_from_test(0.001, 0.0, e2.NO_LABEL_CALIB), 0.8)   # sigmoid(3 - 1.3)
        rows = [{"s1": "2026-02-16T09:56:08Z", "section_id": "a", "y": 1, "p": 0.0}, {"s1": "2026-02-16T09:56:08Z", "section_id": "b", "y": 0, "p": 0.0}]
        var = [{"acquired_utc": "2026-02-16T09:56:08Z", "section_id": "a", "p_occ": 0.7, "p_test": 0.01, "blind": 0}]
        j = e2.with_variant(rows, var)
        self.assertEqual(len(j), 1)
        self.assertEqual((j[0]["p_occ2"], j[0]["p_test_occ2"], j[0]["occ2_blind"]), (0.7, 0.01, 0))
        self.assertEqual(set(e2.VARIANTS), {"min2", "w15", "noprior", "noridge"})
        self.assertEqual(set(e2.DERIVED), {"2b"})
        self.assertEqual(e2.DEFAULT, "min2")
        ab = e2.absent_days({"QGZ-s2-s1-match.csv": [{"section_id": "a", "s2_occupied": "0", "dt_h": "-17.0", "s1_acquired_utc": "x"},
                                                    {"section_id": "a", "s2_occupied": "0", "dt_h": "70.0", "s1_acquired_utc": "y"},
                                                    {"section_id": "b", "s2_occupied": "1", "dt_h": "1.0", "s1_acquired_utc": "x"}]})
        self.assertEqual(ab, {"a": {"x"}})

    def test_helpers(self):
        self.assertAlmostEqual(occ2.norm_sf(0.0), 0.5, places=9)
        self.assertAlmostEqual(occ2.norm_isf(0.001), 3.0902, places=3)
        c, s, n = occ2.robust_centre_scale(np.concatenate([np.random.default_rng(1).normal(0, 2, 5000), np.full(500, 50.0)]))
        self.assertAlmostEqual(c, 0.0, delta=0.4)      # 10 % of ships move the median by 0.125 sigma
        self.assertAlmostEqual(s, 2.0, delta=0.2)
        null = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 60.0])
        self.assertAlmostEqual(occ2.folded_tail_p(null, 3.0, 3.0, 1.0), (5 + 1) / (8 + 1))   # folded: 0 1 2 3 3 4 5 6
        self.assertLess(occ2.folded_tail_p(null, 12.0, 3.0, 1.0), 1e-6)
        z = np.array([[5.0, 5.0, 0.0], [5.0, 0.0, 5.0], [np.nan, 5.0, 5.0], [np.nan, np.nan, 1.0]])
        s = occ2.sample_stat(z, np.array([0, 1, 2]))
        np.testing.assert_allclose(s[:3], [5.0, 0.0, 5.0])
        self.assertTrue(np.isnan(s[3]))


if __name__ == "__main__":
    unittest.main()
