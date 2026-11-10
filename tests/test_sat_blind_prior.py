"""blind-1 (sat/blind_prior.py): resident sections from the owner's S1 answers, and the judge beside the current one."""
import json
import tempfile
import unittest
from pathlib import Path

from sat import blind_prior as bp
from sat import eval_presence as ep

HEAD = "site_id,section_id,s1_acquired_utc,relative_orbit,s2_date,dt_h,verified,verified_occupied,verified_by,verified_note\n"


def _answers(rows) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "labels").mkdir()
    (d / "labels" / "TST-s1-answers.csv").write_text(
        HEAD + "".join(f"TST,{sec},2026-01-{day:02d}T10:00:00Z,1,,0,{ver},{occ},owner,\n" for sec, day, ver, occ in rows), encoding="utf-8")
    return d


class BlindPrior(unittest.TestCase):
    def test_resident_rule(self):
        out = _answers([("a", d, 1, 1) for d in range(1, 9)] +                      # 8/8 present
                       [("b", d, 1, 1 if d <= 6 else 0) for d in range(1, 9)] +     # 6/8
                       [("c", d, 1, 1 if d <= 4 else 0) for d in range(1, 9)] +     # 4/8
                       [("d", 1, 1, 1), ("d", 2, 1, 1)] +                           # too few answers
                       [("e", d, 0 if d <= 6 else 1, 1) for d in range(1, 9)])      # "cannot judge" rows are left out: 2 answers
        ans = bp.answers(out, "TST")
        self.assertEqual(bp.p_blind(ans, "a"), 0.9)                 # (8 + 1) / (8 + 2)
        self.assertEqual(bp.p_blind(ans, "b"), 0.7)                 # 6/8 = 0.75 is resident: (6 + 1) / (8 + 2)
        self.assertEqual(bp.p_blind(ans, "c"), 0.0)
        self.assertEqual(bp.p_blind(ans, "d"), 0.0)
        self.assertEqual(bp.p_blind(ans, "e"), 0.0)
        self.assertEqual(bp.p_blind(ans, "unknown"), 0.0)
        self.assertEqual(bp.p_blind({}, "a"), 0.0)                  # a site without answers: nothing is raised

    def test_the_judged_scene_is_left_out(self):
        # 6 present of 8: with a "present" scene left out the share is 5/7 < 0.75, with an "absent" one 6/7
        out = _answers([("b", d, 1, 1 if d <= 6 else 0) for d in range(1, 9)])
        ans = bp.answers(out, "TST")
        self.assertEqual(bp.p_blind(ans, "b", exclude_s1="2026-01-01T10:00:00Z"), 0.0)
        self.assertEqual(bp.p_blind(ans, "b", exclude_s1="2026-01-08T10:00:00Z"), round(7 / 9, 3))

    def test_table_and_stored_file(self):
        doc = bp.build(_answers([("a", d, 1, 1) for d in range(1, 5)] + [("c", d, 1, 0) for d in range(1, 5)]), "TST")
        self.assertEqual(doc["version"], "blind-1")
        self.assertEqual(doc["n_resident"], 1)
        self.assertEqual(doc["sections"]["a"], {"present": 4, "answers": 4, "resident": True, "p_blind": round(5 / 6, 3)})
        self.assertFalse(doc["sections"]["c"]["resident"])
        # the stored tables follow the stored answers
        for p in sorted(bp.PRIOR_DIR.glob("*.json")):
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), bp.build(bp.OUT, p.stem), p.name)

    def test_beside_the_current_judge(self):
        rows = [{"y": 1, "p": 0.0, "p_any": 0.0, "p_resident": 0.9, "p_blind1": 0.9},    # a resident quay the count missed
                {"y": 1, "p": 1.0, "p_any": 1.0, "p_resident": 0.0, "p_blind1": 1.0},
                {"y": 0, "p": 0.0, "p_any": 0.0, "p_resident": 0.0, "p_blind1": 0.0},
                {"y": 0, "p": 0.0, "p_any": 0.0}]                                         # a row without blind-1 is not in its block
        m = ep.blind1_metrics(rows)
        self.assertEqual((m["version"], m["n"], m["n_resident"]), ("blind-1", 3, 1))
        self.assertEqual((m["min50"]["recall"], m["current"]["recall"]), (1.0, 0.5))
        self.assertEqual(m["resident"]["p_blind1"]["tp"], 1)
        self.assertEqual(ep.metrics(rows)["fn"], 1)                                       # the body's judge is unchanged

    def test_published_evaluation(self):
        res = ep.evaluate(ep.OUT)
        q = res["other_truth"]["sites"].get("QGZ")
        if not q or not q["rows"]:
            self.skipTest("no S1 answers for QGZ")
        self.assertTrue(all("p_blind1" in r and r["p_blind1"] >= r["p"] for r in q["rows"]))
        b = q["out_of_sample"]["blind1"]
        self.assertGreater(b["min50"]["recall"], b["current"]["recall"])
        self.assertGreaterEqual(b["min50"]["accuracy"], b["current"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
