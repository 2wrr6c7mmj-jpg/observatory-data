"""Plan A, A-0: the presence / absence evaluation (sat/eval_presence.py) and its regression guard.

Pure Python: these tests run in the collect workflows without numpy too.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sat import berths, eval_presence as ep

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
EVAL = OUT / "eval"


def _rows(spec):
    """[(y, p)] -> rows for metrics()."""
    return [{"y": y, "p": p, "p_any": p} for y, p in spec]


class MetricsTest(unittest.TestCase):
    def test_confusion_recall_precision_brier(self):
        m = ep.metrics(_rows([(1, 0.9), (1, 0.6), (1, 0.2), (0, 0.7), (0, 0.1), (0, 0.0)]))
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["recall"], 2 / 3, places=3)
        self.assertAlmostEqual(m["precision"], 2 / 3, places=3)
        self.assertAlmostEqual(m["brier"], round((0.01 + 0.16 + 0.64 + 0.49 + 0.01 + 0.0) / 6, 3), places=3)
        self.assertAlmostEqual(m["base_rate"], 0.5, places=3)

    def test_empty_and_undefined(self):
        m = ep.metrics([])
        self.assertEqual(m["n"], 0)
        self.assertIsNone(m["recall"])
        m = ep.metrics(_rows([(0, 0.0), (0, 0.1)]))   # the owner never saw a ship: recall undefined, precision undefined
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["precision"])
        self.assertEqual(m["tn"], 2)


class JudgeTest(unittest.TestCase):
    def test_est2_length_rule(self):
        st = {"p_real": 0.7, "ship_length_m": 42}
        self.assertEqual(ep.judge_est2(st, 50.0), 0.0)       # a boat under 50 m is not the target
        self.assertEqual(ep.judge_est2(st, 0.0), 0.7)        # any length: est-2's P(real)
        self.assertEqual(ep.judge_est2({"p_real": 0.68, "ship_length_m": 162}, 50.0), 0.68)
        self.assertEqual(ep.judge_est2(None, 50.0), 0.0)

    def test_recount_rule(self):
        self.assertEqual(ep.judge_recount({"s1_occupied": "1", "s1_occ_len_m": "60"}, 50.0), 1.0)
        self.assertEqual(ep.judge_recount({"s1_occupied": "1", "s1_occ_len_m": "40"}, 50.0), 0.0)
        self.assertEqual(ep.judge_recount({"s1_occupied": "1", "s1_occ_len_m": "40"}, 0.0), 1.0)
        self.assertEqual(ep.judge_recount({"s1_occupied": "0", "s1_occ_len_m": ""}, 50.0), 0.0)
        self.assertIsNone(ep.judge_recount({"s1_occupied": "", "s1_occ_len_m": ""}, 50.0))   # no coverage: not judged
        self.assertIsNone(ep.judge_recount(None, 50.0))

    def test_stretch_pairs_by_feature_n_not_position(self):
        # a 150 m section along a parallel; two features of the scene, est-2 kept only the second (n=7)
        lat0 = 21.0
        sec = {"way_id": 11, "side": 1, "line": [[110.0 + i * 0.0001, 21.0] for i in range(15)]}
        feats = [{"n": 3, "osm_way": 11, "face": "11+", "coords": [[110.0 + i * 0.0001, 21.0 + 0.0001] for i in range(5)]},   # 11 m off: also on the section
                 {"n": 7, "osm_way": 11, "face": "11+", "coords": [[110.0005 + i * 0.0001, 21.0] for i in range(5)]}]
        sts = [{"n": 7, "osm_way": 11, "side": 1, "p_real": 0.9, "ship_length_m": 80}]
        st = ep.stretch_of_section(sec, feats, sts, lat0)
        self.assertIsNotNone(st)
        self.assertEqual(st["n"], 7)
        # the other side of the line, or another way: no pairing
        self.assertIsNone(ep.stretch_of_section({**sec, "side": -1}, feats, sts, lat0))
        self.assertIsNone(ep.stretch_of_section({**sec, "way_id": 12}, feats, sts, lat0))
        # a stretch far from the section (300 m east)
        far = [{"n": 7, "osm_way": 11, "face": "11+", "coords": [[110.003 + i * 0.0001, 21.0] for i in range(5)]}]
        self.assertIsNone(ep.stretch_of_section(sec, far, sts, lat0))


class EvaluateRepoTest(unittest.TestCase):
    """The evaluation on the repository's own data (PU 0, well under a second)."""

    @classmethod
    def setUpClass(cls):
        cls.res = ep.evaluate(OUT)

    def test_sections_limited_to_naval_lines(self):
        for sid, r in self.res["sites"].items():
            site = next(s for s in ep.load_sites() if s["id"] == sid)
            secs = ep.site_sections(OUT, site)
            self.assertEqual(r["sections"]["naval"], sum(1 for s in secs.values() if s["class"] == "naval"))
            used = {x["section_id"] for x in r["rows"]}
            for sec_id in used:
                self.assertTrue(secs[sec_id]["naval"], f"{sid} {sec_id} is not on a naval line")
                self.assertGreaterEqual(secs[sec_id]["length_m"], ep.MIN_LENGTH_M)
            lab = r["labels"]
            self.assertEqual(lab["answers"], lab["unknown_section"] + lab["not_naval"] + lab["short"] + lab["no_s1_pairing"] + lab["no_scene"] + lab["used"])
            self.assertEqual(lab["used"], len(r["rows"]))

    def test_rows_and_split(self):
        for sid, r in self.res["sites"].items():
            for x in r["rows"]:
                self.assertIn(x["y"], (0, 1))
                self.assertTrue(0.0 <= x["p"] <= 1.0 and 0.0 <= x["p_any"] <= 1.0)
                self.assertIn(x["source"], ("est-2", "s1-recount"))
                self.assertEqual(x["oos"], x["date"] >= r["freeze_utc"])
            self.assertEqual(r["in_sample"]["min50"]["n"] + r["out_of_sample"]["min50"]["n"], len(r["rows"]))
        tot = self.res["total"]["in_sample"]["min50"]["n"] + self.res["total"]["out_of_sample"]["min50"]["n"]
        self.assertEqual(tot, sum(len(r["rows"]) for r in self.res["sites"].values()))

    def test_zhanjiang_has_a_sample_and_the_new_review_sets_are_listed(self):
        self.assertGreater(self.res["sites"]["ZJG"]["in_sample"]["min50"]["n"], 0)
        for sid in ("LSN", "QDH"):   # Lushun and Qingdao harbour: review sets of naval sections only (answers to come)
            self.assertIn(sid, self.res["sites"])
            rv = json.loads((OUT / "labels" / sid / "review.json").read_text(encoding="utf-8"))
            self.assertTrue(rv["selection"]["naval_only"])
            self.assertEqual(self.res["sites"][sid]["sections"]["naval"], len(rv["sections"]))

    def test_published_files_match(self):
        """docs/data/sat/eval/presence.json and meta.json are what the code writes for the committed data (apart from the time)."""
        pub = json.loads((EVAL / "presence.json").read_text(encoding="utf-8"))
        a = {k: v for k, v in pub.items() if k != "generated_utc"}
        b = {k: v for k, v in self.res.items() if k != "generated_utc"}
        self.assertEqual(a, b, "run python -m sat.eval_presence and commit docs/data/sat/eval/")
        meta = json.loads((EVAL / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["table"], ep.table(self.res))
        names = {r["site_id"]: r["name"] for r in meta["table"]}
        self.assertEqual(names["ZJG"], "湛江")
        self.assertEqual(names["LSN"], "大連・旅順")
        self.assertEqual(names["ALL"], "全港")

    def test_place_names_in_meta(self):
        meta = json.loads((EVAL / "meta.json").read_text(encoding="utf-8"))
        for r in meta["table"]:
            self.assertTrue(r["name"] and not r["name"].isascii(), r)


class PresenceTableTest(unittest.TestCase):
    def test_compare_tables(self):
        old = {"A": {"s1": {"x": 1, "y": 0, "z": 0}, "s0": {"x": 1}}}
        new = {"A": {"s1": {"x": 0, "y": 0}, "s2": {"x": 1}}}   # x flipped, z dropped: 2 of 3 cells of the common scene s1
        c = ep.compare_tables(old, new)
        self.assertEqual(c["sites"]["A"]["scenes"], 1)
        self.assertEqual((c["cells"], c["changed"]), (3, 2))
        self.assertAlmostEqual(c["share"], 2 / 3, places=3)
        self.assertIsNone(ep.compare_tables({}, new)["share"])

    def test_table_follows_the_class_file(self):
        """A section whose line stops being naval leaves the table (the guard sees a rule change as cells that differ)."""
        sites = [s for s in ep.load_sites() if s["id"] == "SLP"]
        tab = ep.presence_table(OUT, sites)
        self.assertIn("SLP", tab)
        n_sec = len(next(iter(tab["SLP"].values())))
        self.assertGreater(n_sec, 0)
        # the same with every class file hidden: no naval line, no section, no table
        with tempfile.TemporaryDirectory() as d:
            saved = berths.CLASS_DIR
            berths.CLASS_DIR = Path(d)
            try:
                self.assertEqual(ep.presence_table(OUT, sites), {})
            finally:
                berths.CLASS_DIR = saved


class RegressionGuardTest(unittest.TestCase):
    """The rules of the working tree against docs/data/sat/eval/presence_snapshot.json: the judge's presence (est-2, p >= 0.5,
    50 m) on the labelled naval sections of the stored scenes of the last 30 days. More than a fifth of the common cells
    changed (presence flipped, or a section that is naval in one version only) fails: a PR that changes berths.json, a
    naval-1 class file, the quay lines or the quay version has to show the change is intended, refresh the snapshot
    (python -m sat.eval_presence --snapshot) and write the number of changed cells in the PR."""

    def test_presence_within_a_fifth_of_the_snapshot(self):
        p = EVAL / ep.SNAPSHOT_FILE
        if not p.exists():
            self.skipTest("no snapshot: python -m sat.eval_presence --snapshot")
        snap = json.loads(p.read_text(encoding="utf-8"))
        cur = ep.presence_table(OUT, days=snap.get("days", ep.SNAPSHOT_DAYS))
        # the snapshot's scenes: recompute them even when they fell out of the 30-day window since
        since = min((min(v) for v in snap["table"].values() if v), default=None)
        if since:
            cur = ep.presence_table(OUT, since=since)
        cmp = ep.compare_tables(snap["table"], cur)
        if cmp["cells"] == 0:
            self.skipTest("the snapshot and the stored scenes share no scene: refresh with python -m sat.eval_presence --snapshot")
        rules_now = ep.rule_versions()
        changed_rules = {k: (snap["rules"].get(k), rules_now.get(k)) for k in rules_now if snap["rules"].get(k) != rules_now.get(k)}
        msg = (f"在/不在が写しと {cmp['changed']}/{cmp['cells']} 升（{cmp['share']:.0%}）違う（上限 {snap.get('max_change', ep.SNAPSHOT_MAX_CHANGE):.0%}）。"
               f"変わった規則: {changed_rules or 'なし'}; 港別: { {k: v['share'] for k, v in cmp['sites'].items()} }。"
               "意図した変更なら python -m sat.eval_presence --snapshot で写しを更新し、PR に升の数を書く")
        self.assertLessEqual(cmp["share"], snap.get("max_change", ep.SNAPSHOT_MAX_CHANGE), msg)


if __name__ == "__main__":
    unittest.main()
