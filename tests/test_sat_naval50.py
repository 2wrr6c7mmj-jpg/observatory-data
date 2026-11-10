"""naval50-1 (sat/eval_presence.py): ships of 50 m or more on the naval berth lines, counted without est-2's discounts."""
import tempfile
import unittest
from pathlib import Path

from sat import eval_presence as ep


class Naval50(unittest.TestCase):
    def test_sum_of_the_stretches(self):
        st = [{"ship_length_m": 150, "k": 1, "c_mean": 1.11, "p_real": 1.0},      # one long ship, a little chance of a second abreast
              {"ship_length_m": 53, "k": 3, "c_mean": 1.0, "p_real": 1.0},        # three ships end to end
              {"ship_length_m": 50, "k": 1, "c_mean": 2.0, "p_real": 0.3},        # exactly 50 m counts; a likely smear counts 0.3
              {"ship_length_m": 49, "k": 4, "c_mean": 1.0, "p_real": 1.0},        # boats under 50 m: left out
              {"ship_length_m": None, "k": 2, "c_mean": 1.0, "p_real": 1.0},
              {"ship_length_m": 90}]                                              # missing k / c_mean / p_real: one ship
        n, total = ep.naval50_sum(st)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(total, 1.11 + 3.0 + 0.6 + 1.0)
        self.assertEqual(ep.naval50_sum([]), (0, 0.0))

    def test_stored_scenes(self):
        rows = ep.naval50_counts(ep.OUT)
        if not rows:
            self.skipTest("no stored est-2 scenes")
        mine = {(r["site_id"], r["acquired_utc"]): {k: "" if v is None else str(v) for k, v in r.items()} for r in rows}
        self.assertEqual(len(mine), len(rows))                                    # one row per site and scene
        for r in rows:
            self.assertEqual(tuple(r), ep.NAVAL50_COLUMNS)
            self.assertGreaterEqual(r["warship_50"], 0)
            if r["stretches_50"] == 0:                                           # no stretch of 50 m: no ship
                self.assertEqual(r["warship_50"], 0)
        # the published rows are the ones this code gives for the same scenes (a scene stored after the last refresh may be missing)
        pub = ep.read_csv(ep.OUT / ep.NAVAL50_FILE)
        self.assertTrue(pub)
        for r in pub:
            self.assertEqual(r, mine.get((r["site_id"], r["acquired_utc"])), f"{r['site_id']} {r['acquired_utc']}")

    def test_nothing_written_without_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ep.write_naval50(Path(d)), 0)
            self.assertFalse((Path(d) / ep.NAVAL50_FILE).exists())


if __name__ == "__main__":
    unittest.main()
