"""opt-fill-1 (sat/opt_fill.py): the radar-dark sections filled from the optical judge pres-3o as a trial series, and no count
changes (quay_blind's earlier columns, warship_blind, quay_naval50). Pure Python (no numpy), no network."""
from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from sat import confirm
from sat import eval_presence as ep
from sat import opt_fill as of

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
STATE_FIELDS = ["site_id", "section_id", "date", "acquired_utc", "state", "value", "source", "p", "model_version"]
ANS_FIELDS = ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c", "verified_color", "verified_by", "verified_note"]


def _write(path: Path, fields: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows([{k: r.get(k, "") for k in fields} for r in rows])
    path.write_text(f.getvalue(), encoding="utf-8")


class RuleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        (self.out / "confirm" / "sections").mkdir(parents=True)
        (self.out / "confirm" / "sections" / "TST.json").write_text(json.dumps({"sections": [
            {"id": s} for s in ("a", "b", "c", "d", "e", "res", "seen")] + [{"id": "gone", "status": "retired"}]}), encoding="utf-8")
        st = [{"site_id": "TST", "section_id": "a", "date": "2026-11-10", "state": "pending", "p": "0.95"},
              {"site_id": "TST", "section_id": "a", "date": "2026-11-12", "state": "pending", "p": "0.10"},      # tie: earlier day wins
              {"site_id": "TST", "section_id": "b", "date": "2026-11-11", "state": "no_image", "p": ""},         # no image: not used
              {"site_id": "TST", "section_id": "b", "date": "2026-11-15", "state": "pending", "p": "0.99"},       # 4 days: in the window
              {"site_id": "TST", "section_id": "c", "date": "2026-11-11", "state": "pending", "p": "0.20"},       # the owner says present
              {"site_id": "TST", "section_id": "d", "date": "2026-11-11", "state": "pending", "p": "0.97"},       # the owner cannot judge
              {"site_id": "TST", "section_id": "d", "date": "2026-11-16", "state": "pending", "p": "0.97"},       # 5 days: out
              {"site_id": "TST", "section_id": "e", "date": "2026-11-11", "state": "unconfirmed", "p": "0.899"},  # under the line
              {"site_id": "TST", "section_id": "res", "date": "2026-11-11", "state": "pending", "p": "0.99"},
              {"site_id": "TST", "section_id": "seen", "date": "2026-11-11", "state": "pending", "p": "0.99"},
              {"site_id": "TST", "section_id": "gone", "date": "2026-11-11", "state": "pending", "p": "0.99"}]
        _write(self.out / "confirm" / "state" / "TST" / "2026-11.csv", STATE_FIELDS, st)
        _write(self.out / "labels" / "TST-s2-answers.csv", ANS_FIELDS, [
            {"site_id": "TST", "section_id": "c", "date": "2026-11-11", "verified": "1", "verified_occupied": "1", "verified_by": "owner"},
            {"site_id": "TST", "section_id": "d", "date": "2026-11-11", "verified": "0", "verified_by": "owner"},
            {"site_id": "TST", "section_id": "e", "date": "2026-11-11", "verified": "1", "verified_occupied": "1", "verified_by": "ai"}])
        self.ctx = of.prepare(self.out, ["TST", "ZJG"])

    def test_zhanjiang_is_left_out(self):
        self.assertNotIn("ZJG", self.ctx)
        self.assertEqual(of.fill(self.ctx.get("ZJG"), "2026-11-11T10:00:00Z", ["a"], {}, []), of.EMPTY)

    def test_the_rule(self):
        dark = ["a", "b", "c", "d", "e", "res", "seen", "gone", "nowhere"]
        row = of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", dark, {"seen": 1, "a": 0}, ["res"])
        self.assertEqual(row["opt_fill_version"], of.VERSION)
        self.assertEqual(row["opt_fill_sections"], 5)               # a b c d e: not the seen, the resident, the retired, the unknown
        self.assertEqual(row["opt_fill_scored"], 4)                 # d: the owner could not judge, and the 16th is 5 days away
        self.assertEqual(row["opt_fill_present"], 3)                # a (0.95, the earlier of a tie), b (0.99, +4 d), c (owner 1)
        basis = row["opt_fill_basis"].split(";")
        self.assertEqual(basis[0], "a|2026-11-10|-1|p|0.950")
        self.assertEqual(basis[1], "b|2026-11-15|+4|p|0.990")
        self.assertEqual(basis[2], "c|2026-11-11|+0|owner|1")     # the owner's answer before the score
        self.assertEqual(basis[3], "e|2026-11-11|+0|p|0.899")     # an AI's answer is not an answer
        self.assertEqual(self.ctx["TST"]["rows"][-1]["opt_fill_present"], 3)

    def test_public_file(self):
        of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", ["a"], {}, [])
        c = of.write_public(self.out, self.ctx)
        self.assertEqual(c["TST"]["present_sum"], 1)
        doc = json.loads((self.out / of.OPT_FILL).read_text(encoding="utf-8"))
        self.assertEqual((doc["version"], doc["threshold"], doc["window_d"]), (of.VERSION, 0.9, 4))
        self.assertIn("Contains modified Copernicus Sentinel data", " ".join(doc["credit"]))
        self.assertEqual(set(doc["policy"]), {"submarine", "length"})
        self.assertTrue(doc["caveats"])
        t = (self.out / of.OPT_FILL).stat().st_mtime_ns
        of.write_public(self.out, self.ctx)
        self.assertEqual((self.out / of.OPT_FILL).stat().st_mtime_ns, t)    # nothing new: not rewritten

    def test_no_file_without_rows(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(of.write_public(Path(d), of.prepare(Path(d), ["ZJG"])), {})
            self.assertFalse((Path(d) / of.OPT_FILL).exists())


class CountsUnchangedTest(unittest.TestCase):
    """The columns before the trial's are the same with and without it, and the published files agree with the code."""

    def test_earlier_columns_do_not_move(self):
        self.assertEqual(ep.BLIND_COLUMNS[-len(of.COLUMNS):], of.COLUMNS)
        sites = [s for s in ep.load_sites() if s["id"] in ("XSN", "QDH", "QJG", "SLP")]
        with_fill = ep.blind_counts(OUT, sites)
        with mock.patch.object(of, "fill", side_effect=RuntimeError("x")):
            broken = ep.blind_counts(OUT, sites)
        without = ep.blind_counts(OUT, sites, {})
        self.assertEqual(sorted(with_fill), sorted(broken))
        for s in with_fill:
            strip = lambda rows: [{k: v for k, v in r.items() if k not in of.COLUMNS} for r in rows]
            self.assertEqual(strip(with_fill[s]), strip(broken[s]))
            self.assertEqual(strip(with_fill[s]), strip(without[s]))
            self.assertTrue(all(r[k] == "" for r in broken[s] + without[s] for k in of.COLUMNS))
            for r in with_fill[s]:
                self.assertEqual(r["opt_fill_version"], of.VERSION)
                self.assertLessEqual(r["opt_fill_present"], r["opt_fill_scored"])
                self.assertLessEqual(r["opt_fill_scored"], r["opt_fill_sections"])
                self.assertLessEqual(r["opt_fill_sections"], r["radar_dark_sections"] - r["radar_dark_seen"])

    def test_readers_agree_with_the_confirm_series(self):
        """opt_fill reads the confirm files with its own readers (the confirm code stays out of the daily collection): the same
        answers, state rows and active sections as sat.confirm gives for the published files."""
        for sid in confirm.known_sites(OUT):
            if sid in of.EXCLUDED:
                continue
            self.assertEqual(of.owner_answers(OUT, sid), {k: a["y"] for k, a in confirm.owner_answers(OUT, sid).items()}, sid)
            self.assertEqual(of.load_state(OUT, sid), confirm.load_state(OUT, sid), sid)
            self.assertEqual(of.active_sections(OUT, sid), set(confirm.active_sections(confirm.load_sections(OUT, sid))), sid)

    def test_zjg_has_no_trial(self):
        self.assertIn("ZJG", of.EXCLUDED)
        self.assertNotIn("ZJG", of.prepare(OUT, ["ZJG"]))
        self.assertFalse((OUT / ep.BLIND_DIR / "ZJG.csv").exists())

    def test_published_file(self):
        p = OUT / of.OPT_FILL
        if not p.exists():
            self.skipTest("opt_fill.json not written yet")
        doc = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], of.VERSION)
        self.assertNotIn("ZJG", doc["counts"])


class FailureStopsNothingTest(unittest.TestCase):
    def test_refresh_goes_on(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stderr"), mock.patch("builtins.print") as pr, \
                mock.patch.object(of, "prepare", side_effect=KeyError("broken")):
            out = Path(d) / "sat"
            shutil.copytree(OUT, out, ignore=shutil.ignore_patterns("images", "color", "s2", "chips", "change", "windows", "confirm"))
            sites = [s for s in ep.load_sites() if s["id"] in ("XSN", "QGZ")]
            r = ep.refresh(out, sites, now=datetime(2026, 11, 22, 6, 0, tzinfo=timezone.utc))
            self.assertIn("KeyError", r["opt_fill"]["error"])
            self.assertEqual(set(r["blind"]), {"XSN", "QGZ"})
            self.assertTrue(r["naval50"])
            printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
            self.assertIn("::warning title=sat opt_fill::", printed)
            with (out / ep.BLIND_DIR / "XSN.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows and all(r[k] == "" for r in rows for k in of.COLUMNS))
            self.assertTrue(all(r["warship_blind"] != "" for r in rows))


if __name__ == "__main__":
    unittest.main()
