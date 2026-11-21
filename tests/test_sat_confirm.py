"""confirm-1 (sat/confirm.py): the rule, the states, the queue, the append-only log, merge, the section ledger, the audit.

Pure Python: these tests run in the collect workflows without numpy too. Everything is synthetic and written to a temporary
directory; the last class reads the repository's own label files (read only).
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from sat import confirm as cf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
NOW = "2026-11-20T08:00:00Z"


def cfg(**kw) -> dict:
    c = cf.load_config()
    c["auto_display"] = True          # the tests of the rule look at the states the rule gives; shadow has its own test
    c.update(kw)
    return c


def sec(i: str) -> dict:
    return {"id": i, "way": 1, "kind": "pier", "side": 1, "sector": "A", "length_m": 150,
            "line": [[120.0, 36.0], [120.001, 36.0]], "frame": [[120.0, 36.0], [120.001, 36.0], [120.001, 36.001], [120.0, 36.001]]}


def cell(site, section, date, p, **kw) -> dict:
    # n_answers: the answers the judge used. 99 by default, so that the smaller count is the one of the answer files (the
    # tests of the rule through the table); the tests of the judge's count set it
    return {"site_id": site, "section_id": section, "date": date, "acquired_utc": date + "T02:50:00Z",
            "p": "" if p is None else str(p), "model_version": "pres3o-test", "cloud_shadow": "0.0", "nodata": "0",
            "n_answers": "99", **kw}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def ledger(self, site="QDH", ids=("A", "B", "C")):
        return cf.sync_sections(self.out, site, [sec(i) for i in ids], "2026-10-01T00:00:00Z")

    def review_answers(self, site, rows):
        """The owner's earlier answers (the review set): rows of (section, date, y) with y 1 / 0 / None."""
        p = self.out / "labels" / f"{site}-s2-answers.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c", "verified_color", "verified_by", "verified_note"])
            for s, d, y in rows:
                w.writerow([site, s, d, "0" if y is None else "1", "" if y is None else y, "" if y is None else y, "", "owner", ""])

    def answers_file(self, rows, name="dl.csv"):
        """A file as the page downloads it: rows of (site, section, date, y)."""
        p = self.out / name
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cf.ANSWER_FIELDS)
            for site, s, d, y in rows:
                w.writerow([site, s, d, "0" if y is None else "1", "" if y is None else y, "owner", "", "2026-11-21T00:00:00Z", "q-test"])
        return p

    def state(self, site="QDH"):
        return cf.load_state(self.out, site)

    def files(self) -> dict:
        return {str(p.relative_to(self.out)): p.read_bytes() for p in sorted(self.out.rglob("*")) if p.is_file()}


H2 = {"n_answers": 2, "prev_answer_date": "2026-10-01", "prev_answer": 0, "prev_days": 30, "n_answers_judge": 2}


class RuleTest(unittest.TestCase):
    """§2.5: the edges of rule v1."""

    def test_config_is_v1(self):
        c = cf.load_config()
        self.assertEqual((c["p_low"], c["p_high"], c["min_answers"], c["max_prev_days"]), (0.03, 0.97, 2, 120))
        self.assertEqual(sorted(c["auto_sites"]), ["LSN", "QDH", "QGZ", "QJG", "SLP"])
        self.assertIs(c["auto_display"], False)          # starts as the shadow operation (§7.2)
        self.assertEqual((c["expire_days"], c["audit"]["rate"]), (28, 0.2))

    def test_threshold_edges(self):
        c = cfg()
        self.assertEqual(cf.decide("QDH", 0.03, True, H2, c), {"state": "auto", "value": 0, "reason": "confident"})
        self.assertEqual(cf.decide("QDH", 0.97, True, H2, c), {"state": "auto", "value": 1, "reason": "confident"})
        self.assertEqual(cf.decide("QDH", 0.0, True, H2, c)["value"], 0)
        self.assertEqual(cf.decide("QDH", 1.0, True, H2, c)["value"], 1)
        for p in (0.0300001, 0.031, 0.5, 0.969, 0.9699999):
            d = cf.decide("QDH", p, True, H2, c)
            self.assertEqual((d["state"], d["value"], d["reason"]), ("pending", None, "uncertain"), p)

    def test_number_of_answers(self):
        c = cfg()
        self.assertEqual(cf.decide("QDH", 0.01, True, {**H2, "n_answers": 2}, c)["state"], "auto")
        d = cf.decide("QDH", 0.01, True, {**H2, "n_answers": 1}, c)
        self.assertEqual((d["state"], d["reason"]), ("pending", "few_answers"))
        d = cf.decide("QDH", 0.01, True, {"n_answers": 0, "prev_answer_date": None, "prev_answer": None, "prev_days": None,
                                          "n_answers_judge": 0}, c)
        self.assertEqual((d["state"], d["reason"]), ("pending", "few_answers"))

    def test_the_smaller_of_the_two_counts(self):
        # B1: the answers in the files and the answers the judge used (n_answers of the score table): the smaller one
        c = cfg()
        for n_file, n_judge, state, reason in ((2, 2, "auto", "confident"), (3, 2, "auto", "confident"), (2, 9, "auto", "confident"),
                                               (1, 2, "pending", "few_answers"), (2, 1, "pending", "few_answers"),
                                               (5, 1, "pending", "few_answers"), (1, 1, "pending", "few_answers"),
                                               (2, 0, "pending", "few_answers"), (2, None, "pending", "answers_unknown"),
                                               (1, None, "pending", "few_answers")):
            d = cf.decide("QDH", 0.01, True, {**H2, "n_answers": n_file, "n_answers_judge": n_judge}, c)
            self.assertEqual((d["state"], d["reason"]), (state, reason), (n_file, n_judge))
        h = dict(H2); del h["n_answers_judge"]                        # no judge count at all: not known, never automatic
        self.assertEqual(cf.decide("QDH", 0.99, True, h, c)["reason"], "answers_unknown")
        # the reasons keep their order; 120 days still come from the files
        self.assertEqual(cf.decide("QDH", 0.5, True, {**H2, "n_answers_judge": None}, c)["reason"], "uncertain+answers_unknown")
        self.assertEqual(cf.decide("QDH", 0.99, True, {**H2, "n_answers_judge": 2, "prev_days": 121}, c)["reason"], "stale_answer")
        self.assertEqual([cf.judge_count(v) for v in ("2", " 3 ", "0", "", None, "1.0", "-1", "x", "²", 2)], [2, 3, 0, None, None, None, None, None, None, 2])

    def test_days_since_the_last_answer(self):
        c = cfg()
        self.assertEqual(cf.decide("QDH", 0.99, True, {**H2, "prev_days": 120}, c)["state"], "auto")
        d = cf.decide("QDH", 0.99, True, {**H2, "prev_days": 121}, c)
        self.assertEqual((d["state"], d["reason"]), ("pending", "stale_answer"))

    def test_ports_that_are_off(self):
        c = cfg()
        for site in ("SLP", "QGZ", "QJG", "QDH", "LSN"):
            self.assertEqual(cf.decide(site, 0.01, True, H2, c)["state"], "auto", site)
        for site in ("ZJG", "SYL", "ZDH", "XSN", "PDG"):
            d = cf.decide(site, 0.01, True, H2, c)
            self.assertEqual((d["state"], d["reason"]), ("pending", "site_off"), site)

    def test_no_image_and_no_score(self):
        c = cfg()
        self.assertEqual(cf.decide("QDH", 0.01, False, H2, c), {"state": "no_image", "value": None, "reason": "no_image"})
        d = cf.decide("QDH", None, True, H2, c)
        self.assertEqual((d["state"], d["reason"]), ("pending", "no_reference"))
        self.assertTrue(cf.is_clear({"cloud_shadow": "0.2", "nodata": "0"}, c))
        self.assertFalse(cf.is_clear({"cloud_shadow": "0.21", "nodata": "0"}, c))
        self.assertFalse(cf.is_clear({"cloud": "0.15", "shadow": "0.1", "nodata": "0"}, c))
        self.assertFalse(cf.is_clear({"cloud_shadow": "0", "nodata": "0.01"}, c))
        self.assertFalse(cf.is_clear({"status": "no_image"}, c))

    def test_history_is_strictly_before_and_skips_cannot_judge(self):
        a = {("A", "2026-09-01"): {"y": 1}, ("A", "2026-10-01"): {"y": None}, ("A", "2026-10-10"): {"y": 0},
             ("A", "2026-11-01"): {"y": 1}, ("B", "2026-10-20"): {"y": 1}}
        h = cf.history(a, "A", "2026-11-01")
        self.assertEqual(h, {"n_answers": 2, "prev_answer_date": "2026-10-10", "prev_answer": 0, "prev_days": 22})
        self.assertEqual(cf.history(a, "A", "2026-09-01")["n_answers"], 0)


class RunTest(Base):
    def setUp(self):
        super().setUp()
        self.ledger()
        self.review_answers("QDH", [("A", "2026-07-04", 0), ("A", "2026-07-24", 0),          # A: 2 answers, the last 2026-07-24
                                    ("B", "2026-07-24", 1),                                    # B: 1 answer
                                    ("C", "2026-07-04", 1), ("C", "2026-07-24", None), ("C", "2026-07-25", 1)])

    def test_edges_through_the_table(self):
        # 2026-07-24 + 120 days = 2026-11-21, + 121 = 2026-11-22
        r = cf.run([cell("QDH", "A", "2026-11-21", "0.03"), cell("QDH", "B", "2026-11-21", "0.01"),
                    cell("QDH", "C", "2026-11-21", "0.9700"), cell("QDH", "X", "2026-11-21", "0.01")],
                   self.out, cfg(), "2026-11-22T06:00:00Z")
        s = self.state()
        self.assertEqual((s[("A", "2026-11-21")]["state"], s[("A", "2026-11-21")]["value"], s[("A", "2026-11-21")]["source"]), ("auto", "0", "auto"))
        self.assertEqual((s[("B", "2026-11-21")]["state"], s[("B", "2026-11-21")]["reason"]), ("pending", "few_answers"))
        self.assertEqual((s[("C", "2026-11-21")]["state"], s[("C", "2026-11-21")]["value"]), ("auto", "1"))
        self.assertEqual(len(r["skipped"]), 1)                       # X is not in the ledger
        cf.run([cell("QDH", "A", "2026-11-22", "0.03"), cell("QDH", "A", "2026-11-23", "0.5")], self.out, cfg(), "2026-11-23T06:00:00Z")
        s = self.state()
        self.assertEqual((s[("A", "2026-11-22")]["state"], s[("A", "2026-11-22")]["reason"]), ("unconfirmed", "superseded"))
        self.assertEqual(s[("A", "2026-11-23")]["reason"], "uncertain+stale_answer")
        self.assertEqual(s[("A", "2026-11-21")]["state"], "auto")    # judged before, not in this table: untouched

    def test_the_judges_count_through_the_table(self):
        # B1: A has 2 answers in the files, B has 1. The table says how many the judge used
        c, day, now = cfg(), "2026-11-01", "2026-11-02T06:00:00Z"

        def judge(rows, header=True):
            with tempfile.TemporaryDirectory() as t:
                out = Path(t)
                cf.sync_sections(out, "QDH", [sec(i) for i in ("A", "B", "C")], "2026-10-01T00:00:00Z")
                (out / "labels").mkdir()
                (out / "labels" / "QDH-s2-answers.csv").write_bytes((self.out / "labels" / "QDH-s2-answers.csv").read_bytes())
                f = out / "scores.csv"
                fields = [k for k in cf.SCORE_FIELDS if header or k != "n_answers"]
                with f.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
                cf.run(cf.read_csv(f), out, c, now)                  # as from the command line: the table read from its file
                return {k[0]: (r["state"], r["reason"], r["n_answers"], r["n_answers_judge"]) for k, r in cf.load_state(out, "QDH").items()}

        s = judge([cell("QDH", "A", day, "0.01", n_answers="2"), cell("QDH", "B", day, "0.01", n_answers="2")])
        self.assertEqual(s["A"], ("auto", "confident", "2", "2"))                 # both 2
        self.assertEqual(s["B"], ("pending", "few_answers", "1", "2"))            # the files say 1
        s = judge([cell("QDH", "A", day, "0.01", n_answers="1"), cell("QDH", "C", day, "0.99", n_answers="3")])
        self.assertEqual(s["A"], ("pending", "few_answers", "2", "1"))            # the judge used 1 (an answer without a picture)
        self.assertEqual(s["C"], ("auto", "confident", "2", "3"))
        s = judge([cell("QDH", "A", day, "0.01", n_answers=""), cell("QDH", "C", day, "0.99", n_answers="")])
        self.assertEqual(s["A"], ("pending", "answers_unknown", "2", ""))         # an empty value: not known
        self.assertEqual(s["C"], ("pending", "answers_unknown", "2", ""))
        s = judge([cell("QDH", "A", day, "0.01"), cell("QDH", "C", day, "0.99")], header=False)
        self.assertEqual(s["A"], ("pending", "answers_unknown", "2", ""))         # no column: not known
        self.assertEqual(s["C"], ("pending", "answers_unknown", "2", ""))
        self.assertEqual(judge([]), {})                                           # an empty table: nothing is judged
        self.assertEqual(judge([], header=False), {})
        # the judge's count goes into the log with the judgement
        cf.run([cell("QDH", "A", day, "0.01", n_answers="1")], self.out, c, now)
        self.assertEqual([(r["n_answers"], r["n_answers_judge"], r["v1_state"]) for r in cf.load_log(self.out, "QDH")], [("2", "1", "pending")])

    def test_an_automatic_state_is_not_an_answer(self):
        c = cfg()
        cf.run([cell("QDH", "A", "2026-10-01", "0.01"), cell("QDH", "B", "2026-10-01", "0.99")], self.out, c, "2026-10-02T06:00:00Z")
        cf.run([cell("QDH", "A", "2026-10-06", "0.01"), cell("QDH", "B", "2026-10-06", "0.99")], self.out, c, "2026-10-07T06:00:00Z")
        cf.run([cell("QDH", "A", "2026-10-11", "0.01"), cell("QDH", "B", "2026-10-11", "0.99")], self.out, c, "2026-10-12T06:00:00Z")
        s = self.state()
        for d in ("2026-10-01", "2026-10-06", "2026-10-11"):
            a, b = s[("A", d)], s[("B", d)]
            self.assertEqual(a["state"], "auto")
            # the automatic cells before it changed neither the count nor the day of the last answer
            self.assertEqual((a["n_answers"], a["prev_answer_date"], a["prev_answer"]), ("2", "2026-07-24", "0"), d)
            # B has one answer of the owner: three confident days of the judge do not make a second one
            self.assertEqual((b["n_answers"], b["prev_answer_date"]), ("1", "2026-07-24"), d)
            self.assertIn(b["state"], ("pending", "unconfirmed"))
            self.assertNotEqual(b["state"], "auto")
        self.assertEqual(cf.owner_answers(self.out, "QDH").keys() & {("A", "2026-10-01"), ("A", "2026-10-06")}, set())
        # 120 days count from the owner's answer, not from the automatic state: 2026-07-24 + 121 days
        cf.run([cell("QDH", "A", "2026-11-22", "0.01")], self.out, c, "2026-11-23T06:00:00Z")
        self.assertEqual(self.state()[("A", "2026-11-22")]["reason"], "stale_answer")

    def test_cells_before_the_answered_day_do_not_change(self):
        c = cfg(latest_only=False)
        table = [cell("QDH", "B", d, "0.4") for d in ("2026-11-01", "2026-11-06", "2026-11-11")]
        cf.run(table, self.out, c, "2026-11-12T06:00:00Z")
        before = self.state()
        log_before = cf.load_log(self.out, "QDH")
        m = cf.merge(self.answers_file([("QDH", "B", "2026-11-06", 0)]), self.out, c, "2026-11-13T00:00:00Z")
        self.assertEqual((m["added"], m["rejected"]), (1, []))
        self.assertEqual(m["rejudge"], {"QDH": ["B@2026-11-11"]})    # only the later day, never the earlier one
        table[2]["p"] = "0.02"                                      # the judge, with the new answer in its history
        cf.run(table, self.out, c, "2026-11-13T06:00:00Z")
        s = self.state()
        self.assertEqual(s[("B", "2026-11-01")], before[("B", "2026-11-01")])
        self.assertEqual([r for r in cf.load_log(self.out, "QDH") if r["date"] == "2026-11-01"],
                         [r for r in log_before if r["date"] == "2026-11-01"])
        self.assertEqual((s[("B", "2026-11-06")]["state"], s[("B", "2026-11-06")]["value"], s[("B", "2026-11-06")]["source"]), ("human", "0", "owner"))
        later = s[("B", "2026-11-11")]
        self.assertEqual((later["n_answers"], later["prev_answer_date"], later["prev_answer"]), ("2", "2026-11-06", "0"))
        self.assertEqual((later["state"], later["value"]), ("auto", "0"))   # two answers now, the last 5 days before
        # the owner's cell is never judged again, whatever the table says
        table[1]["p"] = "0.99"
        cf.run(table, self.out, c, "2026-11-14T06:00:00Z")
        self.assertEqual(self.state()[("B", "2026-11-06")]["state"], "human")

    def test_an_answered_cell_is_not_judged_again(self):
        # C1 (§6.4-1): once the owner has answered a cell, a later run does not judge it again, whatever the table says: no
        # new log row (scoring stays on the row from before the answer) and the state stays the owner's
        c = cfg()
        cf.run([cell("QDH", "A", "2026-11-01", "0.5"), cell("QDH", "B", "2026-11-01", "0.4")], self.out, c, "2026-11-02T06:00:00Z")
        cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 1), ("QDH", "B", "2026-11-01", None)]), self.out, c, "2026-11-03T00:00:00Z")
        log = (self.out / "confirm" / "log" / "QDH" / "2026-11.csv").read_bytes()
        st = {k: dict(v) for k, v in self.state().items()}
        for n, p in enumerate(("0.01", "0.99", "", "0.5")):
            cf.run([cell("QDH", "A", "2026-11-01", p, model_version=f"pres3o-v{n}"),
                    cell("QDH", "B", "2026-11-01", p, model_version=f"pres3o-v{n}")], self.out, c, f"2026-11-{4 + n:02d}T06:00:00Z")
            self.assertEqual((self.out / "confirm" / "log" / "QDH" / "2026-11.csv").read_bytes(), log, p)
            self.assertEqual(self.state(), st, p)
        self.assertEqual([(r["p"], r["state"], r["model_version"], r["y"]) for r in cf.scored_cells(self.out, "QDH")],
                         [("0.5", "pending", "pres3o-test", 1)])
        # a day of the review set: judged once, before its answer, and never again
        cf.run([cell("QDH", "C", "2026-07-25", "0.2")], self.out, c, "2026-11-10T06:00:00Z")
        n = len(cf.load_log(self.out, "QDH"))
        cf.run([cell("QDH", "C", "2026-07-25", "0.9")], self.out, c, "2026-11-11T06:00:00Z")
        self.assertEqual(len(cf.load_log(self.out, "QDH")), n)
        self.assertEqual([r["p"] for r in cf.scored_cells(self.out, "QDH") if r["section_id"] == "C"], ["0.2"])

    def test_the_log_only_grows(self):
        c = cfg()
        logf = self.out / "confirm" / "log" / "QDH" / "2026-11.csv"
        seen = []

        def step():
            b = logf.read_bytes()
            for old in seen:
                self.assertTrue(b.startswith(old), "an earlier part of the log was rewritten")
            seen.append(b)

        cf.run([cell("QDH", "A", "2026-11-01", "0.5"), cell("QDH", "B", "2026-11-01", "0.5")], self.out, c, "2026-11-02T06:00:00Z"); step()
        n1 = len(cf.load_log(self.out, "QDH"))
        cf.run([cell("QDH", "A", "2026-11-01", "0.5"), cell("QDH", "B", "2026-11-01", "0.5")], self.out, c, "2026-11-03T06:00:00Z"); step()
        self.assertEqual(len(cf.load_log(self.out, "QDH")), n1)      # the same judgement again: nothing is added
        cf.run([cell("QDH", "A", "2026-11-01", "0.6")], self.out, c, "2026-11-04T06:00:00Z"); step()
        cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 1)]), self.out, c, "2026-11-05T00:00:00Z"); step()
        cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 0)]), self.out, c, "2026-11-06T00:00:00Z"); step()   # the owner changes the answer
        cf.run([cell("QDH", "A", "2026-11-06", "0.5")], self.out, c, "2026-11-07T06:00:00Z"); step()
        cf.run([], self.out, c, "2026-12-20T06:00:00Z"); step()      # expiry
        rows = [r for r in cf.load_log(self.out, "QDH") if (r["section_id"], r["date"]) == ("A", "2026-11-01")]
        self.assertEqual([(r["state"], r["value"], r["p"]) for r in rows],
                         [("pending", "", "0.5"), ("pending", "", "0.6"), ("human", "1", "0.6"), ("human", "0", "0.6")])
        self.assertEqual(self.state()[("A", "2026-11-01")]["value"], "0")       # the later answer is the state; the earlier one stays in the log
        self.assertEqual(self.state()[("A", "2026-11-06")]["state"], "unconfirmed")
        # scoring uses the last row from before the answer
        sc = [r for r in cf.scored_cells(self.out, "QDH") if r["date"] == "2026-11-01"]
        self.assertEqual([(r["p"], r["state"], r["y"]) for r in sc], [("0.6", "pending", 0)])
        with self.assertRaises(ValueError):                           # a log with another header is not silently rewritten
            p = self.out / "confirm" / "log" / "QDH" / "2026-12.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            cf.append_log(p, [{"site_id": "QDH"}])

    def test_a_reviewed_day_is_logged_as_judged_before_the_answer(self):
        cf.run([cell("QDH", "C", "2026-07-25", "0.2")], self.out, cfg(), "2026-11-02T06:00:00Z")
        rows = cf.load_log(self.out, "QDH")
        self.assertEqual([(r["state"], r["v1_state"], r["n_answers"]) for r in rows], [("pending", "pending", "1"), ("human", "pending", "1")])
        self.assertEqual(self.state()[("C", "2026-07-25")]["value"], "1")

    def test_merge_takes_only_cells_that_were_in_a_queue(self):
        c = cfg()
        cf.run([cell("QDH", "A", "2026-11-01", "0.5"), cell("QDH", "B", "2026-11-01", "0.5", cloud_shadow="0.9")], self.out, c, "2026-11-02T06:00:00Z")
        q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
        self.assertEqual([(x["site_id"], x["section_id"], x["date"]) for x in q["cells"]], [("QDH", "A", "2026-11-01")])
        f = self.answers_file([("QDH", "A", "2026-11-01", 1),
                               ("QDH", "B", "2026-11-01", 1),        # no image: was never shown
                               ("QDH", "A", "2026-10-30", 1),        # no such cell
                               ("QDH", "Z", "2026-11-01", 0),        # no such section
                               ("LSN", "A", "2026-11-01", 0)])       # a port without a queue
        with f.open("a", encoding="utf-8") as fh:
            fh.write("QDH,A,2026-11-01,1,2,owner,,,\n")              # not a valid answer (rejected; the valid row above stays)
        m = cf.merge(f, self.out, c, "2026-11-03T00:00:00Z")
        self.assertEqual(m["added"], 1)
        self.assertEqual(sorted(m["rejected"]), ["LSN:A@2026-11-01", "QDH:A@2026-10-30", "QDH:A@2026-11-01", "QDH:B@2026-11-01", "QDH:Z@2026-11-01"])
        rows = cf.read_csv(cf.confirm_answers_path(self.out, "QDH"))
        self.assertEqual([(r["section_id"], r["date"], r["verified"], r["verified_occupied"], r["verified_by"]) for r in rows],
                         [("A", "2026-11-01", "1", "1", "owner")])
        self.assertFalse(cf.confirm_answers_path(self.out, "LSN").exists())
        # the review-set file of the port is read only
        self.assertEqual(len(cf.read_csv(self.out / "labels" / "QDH-s2-answers.csv")), 6)
        # "cannot judge": the state is unsure, it is not an answer for the history, and the section comes round again
        cf.run([cell("QDH", "A", "2026-11-06", "0.5")], self.out, c, "2026-11-07T06:00:00Z")
        cf.merge(self.answers_file([("QDH", "A", "2026-11-06", None)]), self.out, c, "2026-11-08T00:00:00Z")
        cf.run([cell("QDH", "A", "2026-11-11", "0.5")], self.out, c, "2026-11-12T06:00:00Z")
        s = self.state()
        self.assertEqual((s[("A", "2026-11-06")]["state"], s[("A", "2026-11-06")]["value"]), ("unsure", ""))
        self.assertEqual((s[("A", "2026-11-11")]["n_answers"], s[("A", "2026-11-11")]["prev_answer_date"]), ("3", "2026-11-01"))

    def test_adding_a_section_leaves_the_cells_alone(self):
        c = cfg()
        cf.run([cell("QDH", s, "2026-11-01", p) for s, p in (("A", "0.01"), ("B", "0.5"), ("C", "0.99"))], self.out, c, "2026-11-02T06:00:00Z")
        keep = {k: v for k, v in self.files().items() if k.startswith(("confirm/state/", "confirm/log/"))}
        led = cf.sync_sections(self.out, "QDH", [sec(i) for i in ("A", "B", "C", "D")], "2026-11-03T00:00:00Z")
        self.assertEqual(led["sections_version"], 2)
        self.assertEqual({s["id"]: s["sections_version"] for s in led["sections"]}, {"A": 1, "B": 1, "C": 1, "D": 2})
        self.assertEqual({k: v for k, v in self.files().items() if k in keep}, keep)
        cf.run([cell("QDH", "D", "2026-11-01", "0.01")], self.out, c, "2026-11-03T06:00:00Z")
        now = self.files()
        for k, v in keep.items():
            if k.startswith("confirm/state/"):
                old = v.decode().splitlines()
                self.assertEqual([ln for ln in now[k].decode().splitlines() if not ln.startswith("QDH,D,")], old)
            else:
                self.assertTrue(now[k].startswith(v))
        d = self.state()[("D", "2026-11-01")]
        self.assertEqual((d["state"], d["reason"], d["sections_version"]), ("pending", "few_answers", "2"))   # a new section: everything is shown
        self.assertEqual(self.state()[("A", "2026-11-01")]["sections_version"], "1")
        # a line drawn again needs a new id; a retired id stays in the ledger and its cells stay
        moved = sec("A"); moved["line"] = [[120.0, 36.0], [120.002, 36.0]]
        with self.assertRaises(ValueError):
            cf.sync_sections(self.out, "QDH", [moved], "2026-11-04T00:00:00Z")
        led = cf.sync_sections(self.out, "QDH", [], "2026-11-04T00:00:00Z", retire=("B",))
        self.assertEqual([s["status"] for s in led["sections"]], ["active", "retired", "active", "active"])
        r = cf.run([cell("QDH", "B", "2026-11-06", "0.5")], self.out, c, "2026-11-07T06:00:00Z")
        self.assertEqual(len(r["skipped"]), 1)
        self.assertIn(("B", "2026-11-01"), self.state())

    def test_queue_newest_only_and_expiry(self):
        c = cfg()
        cf.run([cell("QDH", "B", "2026-11-01", "0.5"), cell("QDH", "B", "2026-11-06", "0.5"), cell("QDH", "A", "2026-11-01", "0.4")],
               self.out, c, "2026-11-07T06:00:00Z")
        s = self.state()
        self.assertEqual((s[("B", "2026-11-01")]["state"], s[("B", "2026-11-01")]["reason"]), ("unconfirmed", "superseded"))
        self.assertEqual(s[("B", "2026-11-06")]["state"], "pending")
        cf.run([], self.out, c, "2026-11-29T06:00:00Z")              # 28 days after 11-01: still waiting
        self.assertEqual(self.state()[("A", "2026-11-01")]["state"], "pending")
        cf.run([], self.out, c, "2026-11-30T06:00:00Z")              # 29 days
        s = self.state()
        self.assertEqual((s[("A", "2026-11-01")]["state"], s[("A", "2026-11-01")]["reason"]), ("unconfirmed", "expired"))
        q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
        self.assertEqual([(x["section_id"], x["date"]) for x in q["cells"]], [("B", "2026-11-06")])
        # an answer that comes after the expiry is still taken (the cell was in a queue)
        m = cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 1)]), self.out, c, "2026-12-01T00:00:00Z")
        self.assertEqual((m["added"], self.state()[("A", "2026-11-01")]["state"]), (1, "human"))

    def test_shadow_operation_shows_everything_and_records_the_rule(self):
        c = cf.load_config()                                         # auto_display false
        cf.run([cell("QDH", "A", "2026-11-01", "0.01"), cell("QDH", "C", "2026-11-01", "0.5")], self.out, c, "2026-11-02T06:00:00Z")
        s = self.state()
        self.assertEqual((s[("A", "2026-11-01")]["state"], s[("A", "2026-11-01")]["value"], s[("A", "2026-11-01")]["reason"]), ("pending", "", "shadow"))
        log = {r["section_id"]: r for r in cf.load_log(self.out, "QDH")}
        self.assertEqual((log["A"]["v1_state"], log["A"]["v1_value"], log["A"]["queued"], log["A"]["audit"]), ("auto", "0", "1", "0"))
        self.assertEqual((log["C"]["v1_state"], log["C"]["v1_value"]), ("pending", ""))
        cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 0), ("QDH", "C", "2026-11-01", 0)]), self.out, c, "2026-11-03T00:00:00Z")
        t = cf.summary(self.out, c, NOW)["parts"]["before_freeze"]["total"]
        self.assertEqual((t["would_be_auto"]["n"], t["would_be_auto"]["agree"], t["A_auto_accuracy"]["rate"]), (1, 1, 1.0))
        self.assertEqual((t["routed_judge_accuracy"]["n"], t["routed_judge_accuracy"]["agree"]), (1, 0))
        self.assertEqual(t["changed"], {"n": 1, "routed": 1})         # C: the last answer was "present", now "absent"; it was shown
        self.assertEqual(t["states"]["human"], 2)

    def test_run_writes_only_under_confirm(self):
        c = cfg()
        before = self.files()
        cf.run([cell("QDH", "A", "2026-11-01", "0.5")], self.out, c, "2026-11-02T06:00:00Z")
        cf.merge(self.answers_file([("QDH", "A", "2026-11-01", 1)], "dl.csv"), self.out, c, "2026-11-03T00:00:00Z")
        after = self.files()
        for k, v in before.items():
            self.assertEqual(after[k], v, k)
        new = sorted(k for k in after if k not in before and k != "dl.csv")
        self.assertTrue(all(k.startswith("confirm/") or k == "labels/QDH-confirm-answers.csv" for k in new), new)


class AuditTest(Base):
    """§7.3: the sample of the automatic cells."""
    DAYS = [f"2026-10-{d:02d}" for d in range(1, 29)]

    def make(self, out):
        secs = [f"S{i:02d}" for i in range(20)]
        cf.sync_sections(out, "QDH", [sec(i) for i in secs], "2026-09-01T00:00:00Z")
        p = out / "labels" / "QDH-s2-answers.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = ["site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note"]
        for s in secs:
            rows += [f"QDH,{s},2026-09-10,1,0,0,,owner,", f"QDH,{s},2026-09-20,1,0,0,,owner,"]
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return [cell("QDH", s, d, "0.01" if int(s[1:]) % 2 else "0.99") for s in secs for d in self.DAYS]

    def audited(self, out):
        return {(r["section_id"], r["date"]) for r in cf.load_log(out, "QDH") if r["audit"] == "1"}

    def test_the_same_cells_on_every_run(self):
        c = cfg()
        table = self.make(self.out)
        cf.run(table, self.out, c, "2026-10-29T06:00:00Z")
        a = self.audited(self.out)
        with tempfile.TemporaryDirectory() as t2:                      # from scratch, in another order, one day at a time
            out2 = Path(t2)
            self.make(out2)
            for d in reversed(self.DAYS):
                cf.run([r for r in reversed(table) if r["date"] == d], out2, c, "2026-10-29T06:00:00Z")
            self.assertEqual(self.audited(out2), a)
        cf.run(table, self.out, c, "2026-10-29T12:00:00Z")           # again over the same directory
        self.assertEqual(self.audited(self.out), a)
        # the rates: 20% of the absent side, 40% of the present side (560 cells, 280 each)
        absent = sum(1 for s, _ in a if int(s[1:]) % 2)
        present = len(a) - absent
        self.assertTrue(0.12 < absent / 280 < 0.28, absent)
        self.assertTrue(0.30 < present / 280 < 0.50, present)
        self.assertEqual(cf.audit_rate(0, 30, c), 0.2)
        self.assertEqual(cf.audit_rate(1, 30, c), 0.4)
        self.assertEqual(cf.audit_rate(0, 61, c), 0.4)
        self.assertEqual(cf.audit_rate(1, 61, c), 0.8)
        # another rule version picks another sample
        self.assertNotEqual({k for k in a if cf.audit_pick("QDH", k[0], k[1], 0, 30, {**c, "rule_version": "confirm-v2"})[0]}, a)

    def test_the_queue_does_not_tell(self):
        c = cfg()
        table = self.make(self.out)
        for r in table:
            if r["section_id"] == "S00":
                r["p"] = "0.5"
        cf.run(table, self.out, c, "2026-10-29T06:00:00Z")
        a = self.audited(self.out)
        text = (self.out / "confirm" / "queue.json").read_text(encoding="utf-8")
        q = json.loads(text)
        keys = {(x["section_id"], x["date"]) for x in q["cells"]}
        self.assertEqual(keys, a | {("S00", self.DAYS[-1])})         # the audit cells and the newest waiting cell of S00
        for x in q["cells"]:
            self.assertEqual(sorted(x), ["acquired_utc", "date", "image", "order", "prev", "section_id", "site_id"])
            self.assertEqual(sorted(x["prev"]), ["answer", "date", "image"])
        for word in ('"p"', "audit\"", "v1_", "0.99", "0.01", '"state"', '"reason"', "pres3o"):
            self.assertNotIn(word, text)
        self.assertEqual([x["order"] for x in q["cells"]], list(range(len(q["cells"]))))
        # the state files carry no audit mark either
        st = (self.out / "confirm" / "state" / "QDH" / "2026-10.csv").read_text(encoding="utf-8")
        self.assertNotIn("audit", st.splitlines()[0])
        # the answers of the audit: A is the weighted agreement; an audited cell becomes "human"
        ans = [("QDH", s, d, 0) for s, d in sorted(a)]               # the owner says "absent" everywhere: the present side is wrong
        m = cf.merge(self.answers_file(ans), self.out, c, "2026-10-30T00:00:00Z")
        self.assertEqual((m["added"], m["rejected"]), (len(a), []))
        t = cf.summary(self.out, c, NOW)["parts"]["before_freeze"]["total"]
        n_abs = sum(1 for s, _ in a if int(s[1:]) % 2)
        self.assertEqual((t["audit"]["n"], t["audit"]["agree"]), (len(a), n_abs))
        w_abs, w_pre = n_abs / 0.2, (len(a) - n_abs) / 0.4
        self.assertAlmostEqual(t["A_auto_accuracy"]["rate"], w_abs / (w_abs + w_pre), places=3)
        self.assertEqual(t["states"]["human"], len(a))
        self.assertLessEqual(t["A_auto_accuracy"]["ci95"][0], t["A_auto_accuracy"]["rate"])

    def test_order_changed_first_then_near_the_middle(self):
        c = cfg()
        cells = [{"key": ("QDH", "A", "d"), "state": "pending", "reason": "uncertain", "p": 0.2, "prev_answer": 0},
                 {"key": ("QDH", "B", "d"), "state": "pending", "reason": "uncertain", "p": 0.45, "prev_answer": 0},
                 {"key": ("QDH", "C", "d"), "state": "pending", "reason": "uncertain", "p": 0.8, "prev_answer": 0},
                 {"key": ("QDH", "D", "d"), "state": "pending", "reason": "few_answers", "p": 0.01, "prev_answer": None},
                 {"key": ("QDH", "E", "d"), "state": "pending", "reason": "stale_answer", "p": 0.01, "prev_answer": 0}]
        self.assertEqual([x["key"][1] for x in cf._order(cells, c)], ["C", "B", "A", "D", "E"])


class RepositoryTest(unittest.TestCase):
    """The label files of the repository as they are (read only)."""

    def test_the_owners_answers(self):
        n1 = n0 = 0
        for site in ("SYL", "SLP", "QGZ", "ZDH", "QJG", "QDH", "XSN", "LSN"):
            a = cf.owner_answers(OUT, site)
            n1 += sum(1 for v in a.values() if v["y"] is not None)
            n0 += sum(1 for v in a.values() if v["y"] is None)
        self.assertGreaterEqual(n1, 1450)                             # the design's count (2026-11-20); answers only grow
        self.assertGreaterEqual(n0, 36)
        z = cf.owner_answers(OUT, "ZJG")
        self.assertGreaterEqual(sum(1 for v in z.values() if v["y"] is not None), 81)

    def test_sections_from_the_review_sets(self):
        q = cf.review_sections(OUT, "QDH")
        self.assertEqual(len(q), 24)
        self.assertTrue(all(s["line"] and s["frame"] and s["id"] for s in q))
        z = cf.review_sections(OUT, "ZJG")
        self.assertEqual(len(z), 182)
        s = z[0]
        self.assertEqual(len(s["frame"]), 2 * len(s["line"]))
        # the far side of the frame is 120 m off the line
        (x0, y0), (x1, y1) = s["line"][0], s["frame"][-1]
        import math
        d = math.hypot((x1 - x0) * 111320 * math.cos(math.radians(y0)), (y1 - y0) * 110950)
        self.assertTrue(110 < d < 130, d)


if __name__ == "__main__":
    unittest.main()
