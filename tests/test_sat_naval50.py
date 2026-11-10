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
        # a lower line (the maritime garrison ports, 40 m): the 44 m boats count, the 37 m ones do not
        boats = [{"ship_length_m": 44, "k": 4, "c_mean": 1.05, "p_real": 1.0}, {"ship_length_m": 37, "k": 3, "c_mean": 1.0, "p_real": 1.0}]
        self.assertEqual(ep.naval50_sum(boats), (0, 0.0))
        n, total = ep.naval50_sum(boats, 40.0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(total, 4.2)

    def test_length_line_per_site(self):
        self.assertEqual(set(ep.NAVAL50_MIN_LEN), {"STU", "XMN", "SWS"})          # the maritime garrison ports only
        self.assertTrue(all(v == 40.0 for v in ep.NAVAL50_MIN_LEN.values()))
        for r in ep.naval50_counts(ep.OUT):
            self.assertEqual(r["min_length_m"], "40" if r["site_id"] in ep.NAVAL50_MIN_LEN else "50", r["site_id"])

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

    def test_boxes_of_the_counted_stretches(self):
        boxes = {}
        rows = ep.naval50_counts(ep.OUT, boxes=boxes)
        if not rows:
            self.skipTest("no stored est-2 scenes")
        for r in rows:
            bs = boxes[r["site_id"]][r["acquired_utc"]]
            self.assertEqual(len(bs), r["stretches_50"])                          # one frame per counted stretch
            self.assertLessEqual(abs(sum(b["ships"] for b in bs) - r["warship_50"]), 0.5 + 0.005 * len(bs) + 1e-9)
            for b in bs:
                w, s, e, n = b["box_lonlat"]
                self.assertTrue(w < e and s < n)
        sid = rows[-1]["site_id"]
        doc = ep.naval50_box_doc(sid, rows, boxes[sid])
        self.assertLessEqual(len(doc["scenes"]), ep.NAVAL50_BOX_SCENES)
        last = max(r["acquired_utc"] for r in rows if r["site_id"] == sid)
        self.assertIn(last, doc["scenes"])                                        # the newest scene is always there

    def test_stretch_box(self):
        w, s, e, n = ep.stretch_box([[120.0, 35.0], [120.001, 35.0]], 15.0)      # a line along a parallel still gets a height
        self.assertAlmostEqual((n - s) * 111320, 30, delta=1)
        self.assertLess(w, 120.0)
        self.assertGreater(e, 120.001)
        self.assertEqual(ep.stretch_box([[[120.0, 35.0], [120.001, 35.001]]], 0.0), [120.0, 35.0, 120.001, 35.001])
        self.assertIsNone(ep.stretch_box([]))

    def test_nothing_written_without_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ep.write_naval50(Path(d)), 0)
            self.assertFalse((Path(d) / ep.NAVAL50_FILE).exists())


if __name__ == "__main__":
    unittest.main()
