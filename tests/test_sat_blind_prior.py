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

    def test_exclude_a_set_of_keys_and_s2_answers(self):
        # a site without an S1 check: the S2 day's answers (key = the day); the judged cell's own day is left out
        d = Path(tempfile.mkdtemp())
        (d / "labels").mkdir()
        (d / "labels" / "TST-s2-answers.csv").write_text(
            "site_id,section_id,date,verified,verified_occupied\n" +
            "".join(f"TST,a,2026-01-{day:02d},1,{1 if day <= 6 else 0}\n" for day in range(1, 9)) +
            "TST,z,2026-01-01,0,1\n", encoding="utf-8")
        self.assertEqual(bp.answers_source(d, "TST"), "s2")
        ans = bp.answers(d, "TST")
        self.assertEqual(bp.p_blind(ans, "a"), 0.7)
        self.assertEqual(bp.p_blind(ans, "a", exclude_s1={"2026-01-01T10:00:00Z", "2026-01-01", ""}), 0.0)   # 5/7
        self.assertEqual(bp.p_blind(ans, "a", exclude_s1={"2026-01-08"}), round(7 / 9, 3))                    # 6/7
        self.assertNotIn("z", ans)
        self.assertEqual(bp.build(d, "TST")["answers"], "s2")
        # the AI's provisional answers: used only while the owner has given none, and marked "ai"
        ai = Path(tempfile.mkdtemp())
        (ai / "labels").mkdir()
        (ai / "labels" / "TST-ai-answers.csv").write_text(
            "site_id,section_id,date,verified,verified_occupied,verified_by,verified_note\n" +
            "".join(f"TST,q,2026-01-{day:02d},1,1,ai,x\n" for day in range(1, 9)), encoding="utf-8")
        self.assertEqual(bp.answers_source(ai, "TST"), "ai")
        self.assertEqual(bp.p_blind(bp.answers(ai, "TST"), "q"), 0.9)
        self.assertEqual(bp.build(ai, "TST")["answers"], "ai")
        (ai / "labels" / "TST-s2-answers.csv").write_text("site_id,section_id,date,verified,verified_occupied\nTST,q,2026-01-01,1,0\n", encoding="utf-8")
        self.assertEqual(bp.answers_source(ai, "TST"), "s2")           # the owner's answers replace the AI's table
        self.assertEqual(bp.p_blind(bp.answers(ai, "TST"), "q"), 0.0)
        self.assertEqual(ep.labels(ai, "TST"), [{"section_id": "q", "date": "2026-01-01", "y": 0, "verified_c": "", "source": "TST-s2-answers.csv"}])
        (ai / "labels" / "TST-s2-answers.csv").unlink()
        self.assertEqual(ep.labels(ai, "TST"), [])                      # the AI's answers are never the truth of the evaluation
        # the S1 check wins when the site has one; no answers at all: no table
        self.assertEqual(bp.answers_source(_answers([("a", 1, 1, 1)]), "TST"), "s1")
        self.assertIsNone(bp.answers_source(Path(tempfile.mkdtemp()), "TST"))

    def test_scene_counts_for_the_map(self):
        res = ep.blind_counts(ep.OUT)
        self.assertNotIn("ZJG", res)                      # Zhanjiang has no resident table: nothing is written for it
        if not res:
            self.skipTest("no site with answers")
        for sid, rows in res.items():
            prior = {k: v["p_blind"] for k, v in bp.build(ep.OUT, sid)["sections"].items() if v["resident"]}
            for r in rows:
                self.assertEqual(tuple(r), ep.BLIND_COLUMNS)
                self.assertLessEqual(r["resident_seen"], r["resident_sections"])
                self.assertLessEqual(r["resident_sections"], len(prior))
                self.assertLessEqual(r["warship_blind"], r["resident_sections"] - r["resident_seen"])   # at most one ship per unseen section
            # the published rows are the ones this code gives for the same scenes (a scene stored after the last refresh may be
            # missing from the file: the daily run rewrites it, and a missing row must not stop the daily tests)
            p = ep.OUT / ep.BLIND_DIR / f"{sid}.csv"
            mine = {r["acquired_utc"]: {k: str(v) for k, v in r.items()} for r in rows}
            pub = ep.read_csv(p)
            self.assertTrue(pub, p.name)
            for r in pub:
                self.assertEqual(r, mine.get(r["acquired_utc"]), f"{p.name} {r['acquired_utc']}")
        for site in ep.load_sites():                      # and no file is left for a site without a row (the map would add it)
            if site["id"] not in res:
                self.assertFalse((ep.OUT / ep.BLIND_DIR / f"{site['id']}.csv").exists(), site["id"])
        with tempfile.TemporaryDirectory() as d:          # an output folder without answers: no files
            self.assertEqual(ep.write_blind(Path(d)), {})
            self.assertFalse((Path(d) / ep.BLIND_DIR).exists())

    def test_a_site_without_a_resident_section_loses_its_old_file(self):
        """Lushun 2026-11-13: the AI's table had two resident sections (+2 on the map); the owner's answers have none. The file
        of the earlier run must go, and the files of the sites that were not asked for must stay."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / ep.BLIND_DIR).mkdir()
            head = ",".join(ep.BLIND_COLUMNS) + "\n"
            (d / ep.BLIND_DIR / "TST.csv").write_text(head + "TST,2026-11-09T09:48:11Z,blind-1,ai,2,0,2\n", encoding="utf-8")
            (d / ep.BLIND_DIR / "OTH.csv").write_text(head + "OTH,2026-11-09T09:48:11Z,blind-1,s1,1,0,1\n", encoding="utf-8")
            self.assertEqual(ep.write_blind(d, [{"id": "TST"}]), {})
            self.assertFalse((d / ep.BLIND_DIR / "TST.csv").exists())
            self.assertTrue((d / ep.BLIND_DIR / "OTH.csv").exists())

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
