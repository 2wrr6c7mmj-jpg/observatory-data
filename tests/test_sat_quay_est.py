import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sat import quay_est as e


def seg(length, k=1, way=218405149, side=1, sector="A", outer=0, lead=True, dvh=12.0):
    return {"way_id": way, "side": side, "sector": sector, "length_m": length, "min": k,
            "outer_len_m": outer, "cluster_lead": lead, "peak_dvh_db": dvh}


class TableTest(unittest.TestCase):
    def test_table_has_sources_and_groups(self):
        t = e.load_classes()
        for c in t["classes"]:
            self.assertIn(c["group"], e.GROUPS)
            self.assertIn(c["length_conf"], ("確認", "推定", "未確認"))
            self.assertTrue(c["source"])
            g = c["group"]
            L = c["length_m"]
            self.assertTrue({"X": L >= 200, "L": 150 <= L < 200, "M": 100 <= L < 150, "S": 55 <= L <= 90, "T": L < 50}[g], c["id"])

    def test_zones(self):
        m = e.Model()
        self.assertEqual(m.zone_of("ZJG", 218405149, "A"), ("P1", "large"))
        self.assertEqual(m.zone_of("ZJG", 218404502, "A"), ("P5", "medium"))
        self.assertEqual(m.zone_of("ZJG", 13977070, "A"), ("SH", "shore"))
        self.assertEqual(m.zone_of("ZJG", 218405149, "B"), ("SH", "shore"))
        # the large-ship pier puts more weight on long ships than the shore
        lw, sw = m.zone_weights("large"), m.zone_weights("shore")
        self.assertGreater((lw * m.lengths).sum(), (sw * m.lengths).sum())
        np.testing.assert_allclose(lw.sum(), 1.0)
        # 30 % of the overall mix is always kept
        self.assertTrue((lw >= 0.3 * m.global_w - 1e-12).all())


class EstimateTest(unittest.TestCase):
    def setUp(self):
        self.m = e.Model()

    def test_single_ship_and_long_stretch(self):
        segs = [seg(180), seg(700, k=2, way=218401620)]
        r = e.estimate(segs, "ZJG", self.m)
        self.assertEqual(segs[0]["n_mode"], 1)
        self.assertGreaterEqual(segs[1]["n_mode"], 4)
        self.assertLessEqual(segs[1]["n_p10"], segs[1]["n_mode"])
        self.assertLessEqual(segs[1]["n_mode"], segs[1]["n_p90"])
        self.assertEqual(r["scenarios"], 81)
        self.assertLessEqual(r["all"]["p10"], r["all"]["mode"])
        self.assertLessEqual(r["all"]["mode"], r["all"]["p90"])
        self.assertTrue(segs[0]["clean"])
        self.assertEqual(segs[0]["zone"], "P1")
        self.assertEqual(segs[0]["face"], "218405149+")

    def test_deterministic(self):
        a = e.estimate([seg(400, k=2)], "ZJG", e.Model())
        b = e.estimate([seg(400, k=2)], "ZJG", e.Model())
        self.assertEqual(a["all"], b["all"])

    def test_valleys_raise_the_count(self):
        s1, s3 = seg(300, k=1), seg(300, k=4)
        e.estimate([s1], "ZJG", self.m)
        e.estimate([s3], "ZJG", self.m)
        self.assertGreater(s3["n_mean"], s1["n_mean"])

    def test_valley_conflict(self):
        s = seg(100, k=6)
        e.estimate([s], "ZJG", self.m)
        self.assertIn("valleys", s["prior_conflict"])

    def test_raft_adds_ships(self):
        a, b = seg(180), seg(180, outer=170)
        e.estimate([a], "ZJG", self.m)
        e.estimate([b], "ZJG", self.m)
        self.assertEqual(b["n_mode"], 2)
        self.assertFalse(b["clean"])

    def test_cluster_member_may_be_a_smear(self):
        lead, other = seg(180), seg(180, side=-1, lead=False)
        r = e.estimate([lead, other], "ZJG", self.m)
        self.assertEqual(other["n_p10"], 0)
        self.assertLess(other["n_mean"], 1.0)
        self.assertLess(r["all"]["mean"], 2.0)

    def test_small_craft_not_warships(self):
        s = seg(50, way=13977070, dvh=3.0)
        r = e.estimate([s], "ZJG", self.m)
        self.assertEqual(r["all"]["mode"], 1)
        self.assertEqual(r["warship"]["mode"], 0)

    def test_sectors(self):
        r = e.estimate([seg(180), seg(110, way=13977070, sector="B")], "ZJG", self.m)
        self.assertEqual(r["by_sector"]["A"]["mode"], 1)
        self.assertEqual(r["by_sector"]["B"]["mode"], 1)
        self.assertEqual(r["all"]["mode"], 2)

    def test_false_alarm_and_raft_allow_zero(self):
        s = seg(60, way=13977070)
        e.estimate([s], "ZJG", self.m)
        self.assertEqual(s["n_mode"], 1)
        pn, _ = self.m.posterior(40.0, 1, self.m.zone_weights("shore"), "base")
        self.assertGreater(pn[0], 0.0)
        self.assertLess(pn[0], 0.2)
        pn, _ = self.m.posterior(400.0, 1, self.m.zone_weights("shore"), "base")
        self.assertEqual(pn[0], 0.0)  # longer than any false alarm
        b = seg(180, outer=170)
        e.estimate([b], "ZJG", self.m)
        self.assertEqual(b["n_p10"], 1)  # the outboard ship may be absent (p_raft 0.5-1.0)

    def test_cap_flag(self):
        s = seg(3800, k=1)
        r = e.estimate([s], "ZJG", self.m)
        self.assertIn("cap", s["prior_conflict"])
        self.assertEqual(r["capped"], 1)

    def test_empty(self):
        r = e.estimate([], "ZJG", self.m)
        self.assertEqual(r["all"]["mode"], 0)

    def test_zones_off(self):
        s = [seg(180)]
        r = e.estimate(s, "ZJG", e.Model(use_zones=False))
        self.assertEqual(r["all"]["mode"], 1)


class PriorTest(unittest.TestCase):
    def test_update_forget_and_idempotent(self):
        m = e.Model()
        zp = e.ZonePrior(m, "ZJG")
        a0 = np.array(zp.state["zones"]["P1"]["alpha0"])
        self.assertAlmostEqual(a0.sum(), e.KAPPA, places=2)
        segs = [seg(180)]
        r = e.estimate(segs, "ZJG", m, zp)
        self.assertTrue(zp.update("2026-10-26T10:48:45Z", r["soft"]))
        self.assertFalse(zp.update("2026-10-26T10:48:45Z", r["soft"]))
        a1 = np.array(zp.state["zones"]["P1"]["alpha"])
        self.assertAlmostEqual(a1.sum(), e.KAPPA + 1, places=2)
        self.assertGreater(a1[4], a0[4])  # 170-200 m
        # zones without data drift back toward alpha0 (unchanged here) and keep their total
        self.assertAlmostEqual(sum(zp.state["zones"]["P3"]["alpha"]), e.KAPPA, places=2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            zp.save(p)
            zp2 = e.ZonePrior.load(m, "ZJG", p)
            self.assertEqual(zp2.state["zones"]["P1"]["alpha"], zp.state["zones"]["P1"]["alpha"])
            self.assertIn("2026-10-26T10:48:45Z", json.loads(p.read_text())["applied"])

    def test_out_of_prior_is_flagged_and_not_used(self):
        m = e.Model()
        zp = e.ZonePrior(m, "ZJG")
        s = seg(270, way=218401620)  # one 250 m ship at a medium-ship pier: beyond the prior's 97.5 %
        r = e.estimate([s], "ZJG", m, zp)
        self.assertIn("length", s["prior_conflict"])
        self.assertEqual(r["soft"], {})

    def test_reweight_identity_at_start(self):
        m = e.Model()
        w = m.zone_weights("large")
        a = e.KAPPA * m.bin_mass(w)
        np.testing.assert_allclose(m.reweight(w, a), w, atol=1e-12)

    def test_cols_hint(self):
        self.assertIsNone(e.cols_hint(None))
        self.assertEqual(e.cols_hint(35), 1)
        self.assertEqual(e.cols_hint(60), 2)
        self.assertEqual(e.cols_hint(200), 3)


if __name__ == "__main__":
    unittest.main()
