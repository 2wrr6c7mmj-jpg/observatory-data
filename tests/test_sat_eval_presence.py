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


    def test_near_pairs_metrics_beside_all(self):
        """near: the same metrics of the pairs with |S1 - S2| <= FAR_H only; the headline keeps every pair."""
        rows = _rows([(1, 1.0), (1, 0.0), (0, 1.0), (0, 0.0)])
        for r, dt in zip(rows, [-16.6, 88.6, 7.4, -64.6]):
            r["dt_h"] = dt
            r["p_any"] = r["p"]
        sm = ep.split_metrics(rows)
        self.assertEqual(sm["min50"]["n"], 4)
        self.assertEqual(sm["near"]["far_h"], ep.FAR_H)
        self.assertEqual(sm["near"]["min50"]["n"], 2)                     # -16.6 h and 7.4 h
        self.assertEqual((sm["near"]["min50"]["tp"], sm["near"]["min50"]["fp"]), (1, 1))
        self.assertEqual(sm["min50"]["recall"], 0.5)
        self.assertEqual(sm["near"]["min50"]["recall"], 1.0)
        rows[0]["dt_h"] = None                                             # no pairing time: not near
        self.assertEqual(ep.split_metrics(rows)["near"]["min50"]["n"], 1)
        t = ep.table({"sites": {}, "total": {"in_sample": sm, "out_of_sample": sm, "by_source": {s: {"n": 0} for s in ep.SOURCES}}})
        self.assertEqual((t[0]["n"], t[0]["n_near"], t[0]["n_near_oos"], t[0]["recall_near_oos"]), (4, 2, 2, 1.0))

    def test_generated_utc_from_the_given_time(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as d:
            now = datetime(2026, 10, 30, 5, 17, tzinfo=timezone.utc)
            res = ep.evaluate(Path(d), sites=[], now=now)
            self.assertEqual(res["generated_utc"], "2026-10-30T05:17:00Z")
            doc = ep.snapshot(Path(d), sites=[], now=now)
            self.assertEqual(doc["generated_utc"], "2026-10-30T05:17:00Z")


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
                self.assertTrue(secs[sec_id]["berth"], f"{sid} {sec_id} is on a berth-false line")
                self.assertGreaterEqual(secs[sec_id]["length_m"], ep.MIN_LENGTH_M)
            lab = r["labels"]
            self.assertEqual(lab["answers"], lab["unknown_section"] + lab["not_naval"] + lab["no_berth"] + lab["short"] + lab["no_s1_pairing"]
                             + lab["no_scene"] + lab["other_day_of_scene"] + lab["used"])
            self.assertEqual(r["sections"]["no_berth"], sum(1 for s in secs.values() if s["naval"] and not s["berth"]))
            self.assertEqual(lab["used"], len(r["rows"]))
        # Gu Zhenkou (A-1, 2026-11-07): the hooked jetty 582517902 and the yard shore 15387629:L:06 are out of the evaluation
        q = self.res["sites"]["QGZ"]
        self.assertEqual(q["sections"]["no_berth"], 3)
        self.assertFalse({x["section_id"] for x in q["rows"]} & {"582517902:L:01", "582517902:R:00", "15387629:L:06"})
        self.assertGreater(q["labels"]["no_berth"], 0)

    def test_rows_and_split(self):
        for sid, r in self.res["sites"].items():
            for x in r["rows"]:
                self.assertIn(x["y"], (0, 1))
                self.assertTrue(0.0 <= x["p"] <= 1.0 and 0.0 <= x["p_any"] <= 1.0)
                self.assertIn(x["source"], ("est-2", "s1-recount"))
                self.assertEqual(x["oos"], (not r["calibrated"]) or x["date"] >= r["freeze_utc"])
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

    def test_published_files_are_consistent(self):
        """docs/data/sat/eval/presence.json and meta.json: the code's version and shape, and the table of meta.json is the table of
        presence.json. Not an exact match with a fresh evaluation: the daily run (sat/run.py save -> eval_presence.refresh)
        rewrites both from the stored files, so the committed copy may lag the data by a day without failing the collect workflows."""
        pub = json.loads((EVAL / "presence.json").read_text(encoding="utf-8"))
        self.assertEqual(pub["version"], ep.EVAL_VERSION)
        self.assertEqual(set(pub["sites"]), set(self.res["sites"]))
        for r in pub["sites"].values():
            for k in ("calibrated", "by_source", "pairing", "in_sample", "out_of_sample", "rows"):
                self.assertIn(k, r)
        meta = json.loads((EVAL / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["table"], ep.table(pub))
        self.assertEqual(meta["version"], ep.EVAL_VERSION)
        self.assertIn("attribution", meta)
        names = {r["site_id"]: r["name"] for r in meta["table"]}
        self.assertEqual(names["ZJG"], "湛江")
        self.assertEqual(names["LSN"], "大連・旅順")
        self.assertEqual(names["ALL"], "全港")

    def test_one_label_day_per_scene_and_sources(self):
        for sid, r in self.res["sites"].items():
            days_of = {}
            for x in r["rows"]:
                days_of.setdefault(x["s1"], set()).add(x["date"])
            for s1, days in days_of.items():
                self.assertEqual(len(days), 1, f"{sid} {s1} judged against {sorted(days)}")
            self.assertEqual(r["pairing"]["scenes"], len(days_of))
            self.assertEqual(sum(v["n"] for v in r["by_source"].values()), len(r["rows"]))
            if not r["calibrated"]:   # borrowed weights: nothing of this port is in sample
                self.assertEqual(r["in_sample"]["min50"]["n"], 0)
        zjg = self.res["sites"]["ZJG"]
        self.assertTrue(zjg["calibrated"])
        self.assertEqual(set(zjg["by_source"]), set(ep.SOURCES))
        self.assertGreater(zjg["by_source"]["s1-recount"]["n"], 0)   # the labelled days before 2026-10 have only the re-count

    def test_nearest_day_per_scene(self):
        rows = [{"s1": "a", "date": "2026-04-03", "dt_h": 30.0}, {"s1": "a", "date": "2026-04-06", "dt_h": -40.0},
                {"s1": "a", "date": "2026-04-03", "dt_h": 30.0}, {"s1": "b", "date": "2026-05-01", "dt_h": None}]
        keep, dropped = ep.nearest_day_per_scene(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual([(r["s1"], r["date"]) for r in keep], [("a", "2026-04-03"), ("a", "2026-04-03"), ("b", "2026-05-01")])

    def test_scene_features_by_mask_version(self):
        feats = {("t", "X-land-v1"): [{"n": 1}], ("t", "X-land-v2"): [{"n": 2}]}
        self.assertEqual(ep.scene_features(feats, {"acquired_utc": "t", "mask_version": "X-land-v2"}), [{"n": 2}])
        self.assertEqual(len(ep.scene_features(feats, {"acquired_utc": "t"})), 2)
        self.assertEqual(ep.scene_features(feats, {"acquired_utc": "t", "mask_version": "X-land-v3"}), [])

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
    50 m) on the labelled naval sections of the stored scenes of the last 30 days. Checked only when a rule version differs from
    the snapshot's (berths.json and its hash, a naval-1 class file, the quay version, a site's quay lines): then more than a
    fifth of the common cells changed, in all or in any site with at least 30 common cells, fails, and so does a site whose
    snapshot scenes are mostly missing from the stored files (a stale snapshot). The daily run refreshes the snapshot while
    the rules are unchanged; a PR that changes a rule refreshes it by hand (python -m sat.eval_presence --snapshot) once the
    change is understood, and writes the number of changed cells in the PR."""

    def test_presence_within_a_fifth_of_the_snapshot(self):
        p = EVAL / ep.SNAPSHOT_FILE
        if not p.exists():
            self.skipTest("no snapshot: python -m sat.eval_presence --snapshot")
        snap = json.loads(p.read_text(encoding="utf-8"))
        rules_now = ep.rule_versions()
        changed_rules = {k: (snap["rules"].get(k), rules_now.get(k)) for k in rules_now if snap["rules"].get(k) != rules_now.get(k)}
        if not changed_rules:
            self.skipTest("the rules are those of the snapshot: nothing to compare (the daily run keeps the snapshot current)")
        since = min((min(v) for v in snap["table"].values() if v), default=None)
        cur = ep.presence_table(OUT, since=since) if since else {}
        cmp = ep.compare_tables(snap["table"], cur)
        limit = snap.get("max_change", ep.SNAPSHOT_MAX_CHANGE)
        min_cells = snap.get("min_site_cells", ep.SNAPSHOT_MIN_SITE_CELLS)
        hint = f"変わった規則: {changed_rules}。意図した変更なら python -m sat.eval_presence --snapshot で写しを更新し、PR に升の数を書く"
        for sid, c in cmp["sites"].items():
            if c["scenes_old"]:
                self.assertGreaterEqual(c["scenes"], c["scenes_old"] / 2,
                                        f"{sid}: 写しの撮影 {c['scenes_old']} のうち {c['scenes']} しか保存に無い（写しが古い）。{hint}")
            if c["cells"] >= min_cells:
                self.assertLessEqual(c["share"], limit, f"{sid}: 在/不在が写しと {c['changed']}/{c['cells']} 升（{c['share']:.0%}）違う（上限 {limit:.0%}）。{hint}")
        self.assertGreater(cmp["cells"], 0, f"写しと共通の升が無い。{hint}")
        self.assertLessEqual(cmp["share"], limit,
                             f"全体で在/不在が写しと {cmp['changed']}/{cmp['cells']} 升（{cmp['share']:.0%}）違う（上限 {limit:.0%}）。港別: "
                             f"{ {k: v['share'] for k, v in cmp['sites'].items()} }。{hint}")

    def test_guard_fires_on_a_simulated_rule_change(self):
        """With Zhanjiang's unclassed lines read as naval, the table differs from the snapshot by far more than a fifth."""
        p = EVAL / ep.SNAPSHOT_FILE
        if not p.exists():
            self.skipTest("no snapshot")
        snap = json.loads(p.read_text(encoding="utf-8"))
        orig = berths.load_classes

        def fake(site_id, d=None):
            c = orig(site_id, d)
            return dict(c, default="naval") if c and site_id == "ZJG" else c
        berths.load_classes = fake
        try:
            cur = ep.presence_table(OUT, [s for s in ep.load_sites() if s["id"] == "ZJG"], since=min(snap["table"]["ZJG"]))
        finally:
            berths.load_classes = orig
        c = ep.compare_tables({"ZJG": snap["table"]["ZJG"]}, cur)["sites"]["ZJG"]
        self.assertGreater(c["share"], ep.SNAPSHOT_MAX_CHANGE)

    def test_refresh_keeps_a_snapshot_of_other_rules(self):
        """refresh() rewrites presence.json; the snapshot only when its rule versions are the working tree's."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "eval").mkdir()
            (out / "eval" / ep.SNAPSHOT_FILE).write_text(json.dumps({"rules": {"berths": "other"}, "table": {}}), encoding="utf-8")
            r = ep.refresh(out, [s for s in ep.load_sites() if s["id"] == "SLP"])
            self.assertEqual(r["snapshot"], "kept")
            self.assertEqual(json.loads((out / "eval" / ep.SNAPSHOT_FILE).read_text(encoding="utf-8"))["rules"], {"berths": "other"})
            self.assertTrue((out / "eval" / "presence.json").exists())
            self.assertTrue((out / "eval" / "meta.json").exists())
            (out / "eval" / ep.SNAPSHOT_FILE).unlink()
            self.assertEqual(ep.refresh(out, [s for s in ep.load_sites() if s["id"] == "SLP"])["snapshot"], "created")
            self.assertEqual(ep.refresh(out, [s for s in ep.load_sites() if s["id"] == "SLP"])["snapshot"], "updated")


if __name__ == "__main__":
    unittest.main()
