"""pres-1a (trial) and the change flag (sat/pres1.py, sat/pres1_calib.py).

Pure Python: these tests run in the collect workflows without numpy too. The main point is that no answer from the scene's day
or later enters a feature or the calibration (a leak would make the out-of-sample numbers look better than they are).
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sat import eval_presence as ep
from sat import pres1, pres1_calib as pc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"


class HistoryTest(unittest.TestCase):
    ANS = {"A": [("2026-01-01", 1), ("2026-02-01", 1), ("2026-03-01", 0), ("2026-03-05", 1), ("2026-04-01", 0)]}

    def test_only_days_strictly_before(self):
        h = pres1.history(self.ANS, "A", "2026-03-05T10:00:00Z")      # the scene's own day (03-05) is left out
        self.assertEqual(h["res_n"], 3)
        self.assertAlmostEqual(h["res_rate"], (2 + 1) / (3 + 2))
        self.assertEqual((h["prev_y"], h["prev_date"]), (0, "2026-03-01"))

    def test_future_answers_do_not_change_the_history(self):
        a = pres1.history(self.ANS, "A", "2026-03-02")
        fut = {"A": self.ANS["A"][:3] + [("2026-03-02", 0), ("2026-03-05", 0), ("2027-01-01", 1), ("2027-02-01", 1)]}
        self.assertEqual(a, pres1.history(fut, "A", "2026-03-02"))

    def test_no_answer(self):
        h = pres1.history(self.ANS, "B", "2026-03-02")
        self.assertEqual((h["res_rate"], h["res_n"], h["prev_y"]), (0.5, 0, None))
        self.assertIsNone(pres1.change_flag(1.0, h))
        h = pres1.history(self.ANS, "A", "2026-01-01")                 # nothing before the first answer
        self.assertEqual((h["res_n"], h["prev_y"]), (0, None))

    def test_cut_day_is_the_earlier_of_label_day_and_scene(self):
        self.assertEqual(pres1.cut_day("2026-03-05", "2026-03-03T10:00:00Z"), "2026-03-03")
        self.assertEqual(pres1.cut_day("2026-03-05", "2026-03-07T10:00:00Z"), "2026-03-05")
        self.assertEqual(pres1.cut_day(None, "2026-03-07T10:00:00Z"), "2026-03-07")

    def test_change_flag(self):
        h = {"prev_y": 1}
        self.assertEqual(pres1.change_flag(0.0, h), 1)
        self.assertEqual(pres1.change_flag(0.9, h), 0)
        self.assertEqual(pres1.change_flag(0.5, {"prev_y": 0}), 1)

    def test_answers_one_per_day_s2_first(self):
        s2 = [{"section_id": "A", "date": "2026-01-02", "y": 0}, {"section_id": "A", "date": "2026-01-05", "y": 1}]
        s1 = [{"section_id": "A", "s1": "2026-01-02T10:00:00Z", "y": 1}, {"section_id": "A", "s1": "2026-01-09T10:00:00Z", "y": 1}]
        with mock.patch.object(ep, "labels", return_value=s2), mock.patch.object(ep, "labels_s1", return_value=s1):
            a = pres1.answers(OUT, "X")
        self.assertEqual(a, {"A": [("2026-01-02", 0), ("2026-01-05", 1), ("2026-01-09", 1)]})


class AnnotateTest(unittest.TestCase):
    CAL = {"fit_before": "2026-03-01", "model": {"features": list(pres1.FEATURES), "mean": [0.3, 0.4, 2.0], "sd": [0.5, 0.2, 2.0],
                                                 "intercept": -0.5, "coef": [0.7, 1.5, 0.2]}}

    def test_row_features_ignore_own_day_and_later(self):
        ans = {"A": [("2026-01-01", 1), ("2026-02-01", 1), ("2026-03-10", 0), ("2026-04-01", 0)]}
        row = {"section_id": "A", "date": "2026-03-10", "s1": "2026-03-11T10:00:00Z", "p": 0.0, "y": 0}
        pres1.annotate([row], ans, self.CAL)
        self.assertEqual((row["res_n"], row["prev_y"], row["change_flag"]), (2, 1, 1))
        self.assertTrue(row["pres1_oos"])
        # the same row with the answers of 03-10 and later reversed: nothing changes
        row2 = {k: row[k] for k in ("section_id", "date", "s1", "p", "y")}
        pres1.annotate([row2], {"A": ans["A"][:2] + [("2026-03-10", 1), ("2026-04-01", 1), ("2026-05-01", 1)]}, self.CAL)
        self.assertEqual(row["p_pres1"], row2["p_pres1"])

    def test_without_calibration(self):
        row = {"section_id": "A", "date": "2026-03-10", "s1": "2026-03-11T10:00:00Z", "p": 1.0, "y": 1}
        pres1.annotate([row], {}, None)
        self.assertIsNone(row["p_pres1"])
        self.assertIsNone(row["change_flag"])

    def test_block_changed_cells_and_flag(self):
        rows = [  # (y, prev_y, p, p_pres1)
            {"y": 1, "prev_y": 0, "p": 1.0, "p_pres1": 0.2, "pres1_oos": True},    # changed, current right, pres1 wrong, flagged
            {"y": 0, "prev_y": 1, "p": 1.0, "p_pres1": 0.9, "pres1_oos": True},    # changed, both wrong, not flagged
            {"y": 1, "prev_y": 1, "p": 0.0, "p_pres1": 0.8, "pres1_oos": True},    # same, pres1 right, flagged (false alarm)
            {"y": 0, "prev_y": None, "p": 0.0, "p_pres1": 0.3, "pres1_oos": True},  # no previous answer
            {"y": 1, "prev_y": 1, "p": 1.0, "p_pres1": 0.9, "pres1_oos": False},   # training period: not reported
        ]
        for r in rows:
            r["change_flag"] = pres1.change_flag(r["p"], r)
        b = pres1.block(rows, ep.metrics)
        self.assertEqual(b["n"], 4)
        self.assertEqual((b["changed"]["n"], b["same"]["n"], b["no_prev_n"]), (2, 1, 1))
        self.assertEqual(b["changed"]["current"]["accuracy"], 0.5)
        self.assertEqual(b["changed"]["pres1"]["accuracy"], 0.0)
        cf = b["change_flag"]
        self.assertEqual((cf["cells"], cf["flags"], cf["true"], cf["changes"]), (3, 2, 1, 2))
        self.assertEqual((cf["precision"], cf["recall"]), (0.5, 0.5))


class FitTest(unittest.TestCase):
    def test_fit_reaches_the_penalised_optimum(self):
        """The gradient of sum log-loss + ||w||^2/(2C) (intercept unpenalised) is zero at the fitted coefficients."""
        X = [[float(i % 2), (i % 7) / 7.0, float(i % 5)] for i in range(60)]
        y = [1 if (x[0] + x[1] > 0.8) != (i % 9 == 0) else 0 for i, x in enumerate(X)]
        m = pres1.fit(X, y)
        Z = [[1.0] + [(v - mu) / s for v, mu, s in zip(x, m["mean"], m["sd"])] for x in X]
        w = [m["intercept"]] + m["coef"]
        g = [0.0] * 4
        for z, t in zip(Z, y):
            p = 1 / (1 + math.exp(-sum(a * b for a, b in zip(w, z))))
            for i in range(4):
                g[i] += (p - t) * z[i]
        for i in range(1, 4):
            g[i] += w[i] / m["c"]
        self.assertLess(max(abs(v) for v in g), 1e-6)
        self.assertAlmostEqual(pres1.predict(m, X[0]), 1 / (1 + math.exp(-sum(a * b for a, b in zip(w, Z[0])))))


def _fake_cells():
    """Two sites' evaluation rows and answers over 2025-11 .. 2026-08 (the answers are the rows' own truth)."""
    res, ans = {"sites": {}}, {}
    for sid in ("P1", "P2"):
        rows, a = [], {}
        for j, day in enumerate(["2025-11-05", "2025-12-10", "2026-01-15", "2026-02-20", "2026-04-02", "2026-05-10", "2026-06-20", "2026-08-01"]):
            for k in range(6):
                sec = f"{sid}:{k}"
                y = 1 if (k < 3) != (j == 4 and k == 0) else 0
                p = 1.0 if (y and k != 1) or (not y and k == 5) else 0.0
                rows.append({"section_id": sec, "date": day, "s1": day + "T10:00:00Z", "y": y, "p": p, "dt_h": 5.0})
                a.setdefault(sec, []).append((day, y))
        res["sites"][sid] = {"rows": rows}
        ans[sid] = a
    return res, ans


class CalibrationLeakTest(unittest.TestCase):
    def run_calib(self, res, ans, fit_before="2026-04-01"):
        with mock.patch.object(ep, "evaluate_one", return_value=res), mock.patch.object(pres1, "answers", side_effect=lambda out, sid: ans[sid]):
            return pc.calibrate(OUT, [], fit_before, boot=50)

    def test_answers_from_fit_before_on_do_not_move_the_model(self):
        res, ans = _fake_cells()
        a = self.run_calib(res, ans)
        self.assertLess(a["train_last_label_day"], "2026-04-01")
        self.assertLess(a["train_last_answer_used"], "2026-04-01")
        # reverse every answer and every truth from fit_before on: the coefficients stay the same, the holdout moves
        res2 = json.loads(json.dumps(res))
        for s in res2["sites"].values():
            for r in s["rows"]:
                if r["date"] >= "2026-04-01":
                    r["y"] = 1 - r["y"]
        ans2 = {sid: {sec: [(d, (1 - y) if d >= "2026-04-01" else y) for d, y in v] for sec, v in a_.items()} for sid, a_ in ans.items()}
        b = self.run_calib(res2, ans2)
        self.assertEqual(a["model"], b["model"])
        self.assertNotEqual(a["holdout"]["all"]["pres1"], b["holdout"]["all"]["pres1"])
        # forward windows: each one's model is the same too (fitted before its own start)
        for wa, wb in zip(a["forward"]["windows"], b["forward"]["windows"]):
            self.assertEqual((wa["coef"], wa["intercept"]), (wb["coef"], wb["intercept"]))

    def test_check_no_future_catches_a_leak(self):
        good = {"date": "2026-01-01", "s1": "2026-01-01T10:00:00Z", "cut": "2026-01-01", "last_answer_used": "2025-12-01"}
        pc.check_no_future([good], "2026-02-01")
        with self.assertRaises(AssertionError):
            pc.check_no_future([{**good, "date": "2026-02-01"}], "2026-02-01")
        with self.assertRaises(AssertionError):
            pc.check_no_future([{**good, "last_answer_used": "2026-01-01"}], "2026-02-01")   # the scene's own day

    def test_cells_history_before_the_scene(self):
        res, ans = _fake_cells()
        with mock.patch.object(ep, "evaluate_one", return_value=res), mock.patch.object(pres1, "answers", side_effect=lambda out, sid: ans[sid]):
            cs = pc.cells(OUT, [])
        for c in cs:
            if c["last_answer_used"] is not None:
                self.assertLess(c["last_answer_used"], c["cut"])
        first = [c for c in cs if c["date"] == "2025-11-05"]
        self.assertTrue(all(c["x"][2] == 0 and c["prev_y"] is None for c in first))


class PublishedTest(unittest.TestCase):
    def test_calibration_file(self):
        d = json.loads(pres1.CALIB_PATH.read_text(encoding="utf-8"))
        self.assertEqual(d["version"], pres1.VERSION)
        self.assertEqual(d["model"]["features"], list(pres1.FEATURES))
        self.assertLess(d["train_last_label_day"], d["fit_before"])
        self.assertLess(d["train_last_answer_used"], d["fit_before"])
        for k in ("all", "changed", "same"):
            self.assertIn("pres1", d["holdout"][k])
        self.assertIsNotNone(pres1.load_calib())

    def test_sites_start_off_or_trial(self):
        sites = ep.load_sites()
        for s in sites:
            self.assertIn(pres1.mode(s), ("off", "trial"), s["id"])
        self.assertEqual(pres1.mode(next(s for s in sites if s["id"] == "ZJG")), "off")
        self.assertEqual(pres1.mode({"id": "X"}), "off")
        with self.assertRaises(ValueError):
            pres1.mode({"id": "X", "quay": {"pres1": "yes"}})
        self.assertFalse((OUT / pres1.OUT_DIR / "ZJG.json").exists())

    def test_published_rows_carry_pres1(self):
        pub = json.loads((OUT / "eval" / "presence.json").read_text(encoding="utf-8"))
        rows = [r for s in pub["sites"].values() for r in s["rows"]]
        self.assertTrue(all("p_pres1" in r and "change_flag" in r for r in rows))
        self.assertIn("pres1", pub["total"])
        self.assertIn("changed", pub["total"]["pres1"])


class WriteTest(unittest.TestCase):
    def test_off_site_gets_no_file_and_loses_a_stale_one(self):
        cal = {"fit_before": "2026-06-01", "model": {"mean": [0, 0, 0], "sd": [1, 1, 1], "intercept": 0, "coef": [1, 1, 0]}}
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            self.assertEqual(pres1.write(out, [{"id": "ZJG", "quay": {"pres1": "off"}}], cal), {})
            self.assertFalse((out / pres1.OUT_DIR).exists())        # no empty directory (Zhanjiang's outputs unchanged)
            (out / pres1.OUT_DIR).mkdir()
            stale = out / pres1.OUT_DIR / "QGZ.json"
            stale.write_text("{}")
            pres1.write(out, [{"id": "QGZ", "quay": {"pres1": "off"}}], cal)
            self.assertFalse(stale.exists())
            # trial, but nothing stored for the site: no file
            with mock.patch.object(pres1, "scene_rows", return_value=[]):
                self.assertEqual(pres1.write(out, [{"id": "QGZ", "quay": {"pres1": "trial"}}], cal), {})
            with mock.patch.object(pres1, "scene_rows", return_value=[{"acquired_utc": "2026-11-01T10:00:00Z", "sections": []}]):
                self.assertEqual(pres1.write(out, [{"id": "QGZ", "quay": {"pres1": "trial"}}], cal), {"QGZ": 1})
            d = json.loads(stale.read_text(encoding="utf-8"))
            self.assertEqual((d["version"], d["mode"]), (pres1.VERSION, "trial"))

    def test_meta_ref_never_raises(self):
        with mock.patch.object(pres1, "settings", side_effect=OSError("x")):
            self.assertIn("status", pres1.meta_ref())


class NeverStopsTest(unittest.TestCase):
    """An error in pres-1 (a calibration file cut short, a bad answers file) never stops the evaluation nor the map's counts
    (quay_blind, quay_naval50) that eval_presence.refresh writes after it: the pres-1 fields are null and a warning is printed."""
    BROKEN = {"fit_before": "2026-06-01", "model": {"coef": [1.0, 2.0, 3.0]}}   # no mean / sd / intercept

    def test_broken_calibration_nulls_the_fields(self):
        slp = [s for s in ep.load_sites() if s["id"] == "SLP"]
        with mock.patch.object(pres1, "load_calib", return_value=self.BROKEN), mock.patch("sys.stderr"):
            res = ep.evaluate(OUT, slp)
        rows = res["sites"]["SLP"]["rows"]
        self.assertTrue(rows)
        self.assertTrue(all(r[k] is None for r in rows for k in pres1.PRES1_FIELDS))
        self.assertTrue(all(r["p"] is not None for r in rows))                 # the current judge is untouched
        self.assertEqual(res["sites"]["SLP"]["pres1"]["n"], 0)
        t = ep.table(res)
        self.assertIsNone(next(x for x in t if x["site_id"] == "SLP")["f1_pres1"])

    def test_block_error_and_columns(self):
        with mock.patch("sys.stderr"):
            b = pres1.safe_block([{"y": 1, "p": 1.0, "p_pres1": 0.9, "pres1_oos": True, "prev_y": 1}], lambda *a: 1 / 0)
        self.assertIn("error", b)
        cols = ep.pres1_columns(b)
        self.assertIsNone(cols["n_pres1"])
        self.assertIsNone(cols["change_flag_recall"])

    def test_refresh_goes_on_to_the_map_counts(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stderr"), \
                mock.patch.object(pres1, "block", side_effect=RuntimeError("block")), \
                mock.patch.object(pres1, "write", side_effect=ValueError("write")), \
                mock.patch.object(ep, "write_naval50", side_effect=lambda out, sites: calls.append("naval50") or 0), \
                mock.patch.object(ep, "write_blind", side_effect=lambda out, sites: calls.append("blind") or {}):
            r = ep.refresh(Path(d), [s for s in ep.load_sites() if s["id"] == "SLP"])
            self.assertTrue((Path(d) / "eval" / "presence.json").exists())
        self.assertEqual(sorted(calls), ["blind", "naval50"])
        self.assertEqual(r["pres1"], {})


class PrevAgeTest(unittest.TestCase):
    def test_days_since_previous_answer(self):
        self.assertEqual(pres1._days("2026-03-15", "2026-11-14T10:00:00Z"), 244)
        self.assertIsNone(pres1._days(None, "2026-11-14T10:00:00Z"))
        d = json.loads((OUT / pres1.OUT_DIR / "ZDH.json").read_text(encoding="utf-8"))
        c = d["scenes"][-1]["sections"]
        self.assertTrue(all("prev_age_days" in x for x in c))
        self.assertIn("prev_age_days", d["note"])

    def test_calibration_reports_ports_of_the_holdout(self):
        d = json.loads(pres1.CALIB_PATH.read_text(encoding="utf-8"))
        hp = d["holdout_by_port"]
        self.assertEqual(sum(v["cells"] for v in hp.values()), d["holdout_n"])
        self.assertEqual(sum(v["scenes"] for v in hp.values()), d["holdout_scenes"])
        self.assertIn("手法の選び方", d["note"])
        self.assertIn("limits", pres1.meta_ref())


if __name__ == "__main__":
    unittest.main()
