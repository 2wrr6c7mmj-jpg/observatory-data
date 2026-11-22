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
        kept = self.ctx["TST"]["rows"]["2026-11-11T10:00:00Z"]
        self.assertEqual(kept["opt_fill_present"], 3)
        # split by where the present comes from: c (owner), a and b (p)
        self.assertEqual((kept["opt_fill_present_owner"], kept["opt_fill_present_p"]), (1, 2))
        self.assertTrue(kept["in_quay_blind"])
        of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", dark, {"seen": 1, "a": 0}, ["res"])     # made again: replaced, not added
        self.assertEqual(len(self.ctx["TST"]["rows"]), 1)

    def test_confirm_answers_win_over_s2_answers(self):
        """labels/<site>-confirm-answers.csv is read after -s2-answers.csv: its answer of the same cell wins (as sat.confirm)."""
        _write(self.out / "labels" / "TST-confirm-answers.csv", ANS_FIELDS, [
            {"site_id": "TST", "section_id": "c", "date": "2026-11-11", "verified": "1", "verified_occupied": "0", "verified_by": "owner"},
            {"site_id": "TST", "section_id": "d", "date": "2026-11-11", "verified": "1", "verified_occupied": "1", "verified_by": "owner"}])
        ans = of.owner_answers(self.out, "TST")
        self.assertEqual(ans[("c", "2026-11-11")], 0)     # s2 said present, confirm says absent
        self.assertEqual(ans[("d", "2026-11-11")], 1)     # s2 said cannot judge, confirm says present
        ctx = of.prepare(self.out, ["TST"])
        row = of.fill(ctx["TST"], "2026-11-11T10:00:00Z", ["c", "d"], {}, [])
        self.assertEqual(row["opt_fill_basis"], "c|2026-11-11|+0|owner|0;d|2026-11-11|+0|owner|1")
        self.assertEqual(row["opt_fill_present"], 1)

    def test_a_later_answer_changes_a_past_row(self):
        """The values are made again every day: an owner's answer that comes in later near a past scene changes its row."""
        acq = "2026-11-11T10:00:00Z"
        before = of.fill(of.prepare(self.out, ["TST"])["TST"], acq, ["a"], {}, [])
        self.assertEqual((before["opt_fill_basis"], before["opt_fill_present"]), ("a|2026-11-10|-1|p|0.950", 1))
        _write(self.out / "labels" / "TST-confirm-answers.csv", ANS_FIELDS, [
            {"site_id": "TST", "section_id": "a", "date": "2026-11-11", "verified": "1", "verified_occupied": "0", "verified_by": "owner"}])
        ctx = of.prepare(self.out, ["TST"])
        after = of.fill(ctx["TST"], acq, ["a"], {}, [])
        self.assertEqual((after["opt_fill_basis"], after["opt_fill_present"]), ("a|2026-11-11|+0|owner|0", 0))
        c = of.write_public(self.out, ctx)
        self.assertEqual((c["TST"]["present_owner_sum"], c["TST"]["present_p_sum"]), (0, 0))
        self.assertIn("過去の撮影の行", json.loads((self.out / of.OPT_FILL).read_text(encoding="utf-8"))["note"])

    def test_public_file(self):
        of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", ["a"], {}, [])
        c = of.write_public(self.out, self.ctx)
        self.assertEqual(c["TST"]["present_sum"], 1)
        self.assertEqual((c["TST"]["present_owner_sum"], c["TST"]["present_p_sum"]), (0, 1))
        doc = json.loads((self.out / of.OPT_FILL).read_text(encoding="utf-8"))
        self.assertIn("p 由来の升だけで採点", doc["note"])
        self.assertIn("(c) OpenStreetMap contributors（ODbL 1.0）", " ".join(doc["credit"]))
        self.assertIsNone(doc["error"])
        self.assertEqual(doc["counts"]["TST"]["rows"][0]["opt_fill_basis"], "a|2026-11-10|-1|p|0.950")
        self.assertEqual((doc["version"], doc["threshold"], doc["window_d"]), (of.VERSION, 0.9, 4))
        self.assertIn("Contains modified Copernicus Sentinel data", " ".join(doc["credit"]))
        self.assertEqual(set(doc["policy"]), {"submarine", "length"})
        self.assertTrue(doc["caveats"])
        t = (self.out / of.OPT_FILL).stat().st_mtime_ns
        of.write_public(self.out, self.ctx)
        self.assertEqual((self.out / of.OPT_FILL).stat().st_mtime_ns, t)    # nothing new: not rewritten

    def test_failure_leaves_no_old_counts(self):
        """A failure of the day (prepare, or the contexts lost) writes counts null and the reason over an earlier file; a failure
        of one scene of a site makes that site's values null; with no earlier file nothing is made."""
        of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", ["a"], {}, [])
        of.write_public(self.out, self.ctx)
        self.assertEqual(of.write_public(self.out, {}, "KeyError: 'x'"), {"error": "KeyError: 'x'"})
        doc = json.loads((self.out / of.OPT_FILL).read_text(encoding="utf-8"))
        self.assertIsNone(doc["counts"])
        self.assertEqual(doc["error"], "KeyError: 'x'")
        of.write_public(self.out, None)
        self.assertIsNone(json.loads((self.out / of.OPT_FILL).read_text(encoding="utf-8"))["counts"])
        self.ctx["TST"]["error"] = "RuntimeError: y"
        c = of.write_public(self.out, self.ctx)
        self.assertEqual((c["TST"]["error"], c["TST"]["present_sum"], c["TST"]["rows"]), ("RuntimeError: y", None, None))
        with tempfile.TemporaryDirectory() as d:
            of.write_public(Path(d), None, "x")
            self.assertFalse((Path(d) / of.OPT_FILL).exists())

    def test_written_atomically(self):
        of.fill(self.ctx["TST"], "2026-11-11T10:00:00Z", ["a"], {}, [])
        with mock.patch.object(of.atomic, "write_text") as w:
            of.write_public(self.out, self.ctx)
        w.assert_called_once()
        self.assertEqual(w.call_args.args[0], self.out / of.OPT_FILL)

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

    def test_osm_credit_of_the_marks(self):
        from sat import radar_dark
        p = OUT / radar_dark.PUBLIC
        if not p.exists():
            self.skipTest("radar_dark.json not written yet")
        self.assertIn("(c) OpenStreetMap contributors（ODbL 1.0）", " ".join(json.loads(p.read_text(encoding="utf-8"))["credit"]))

    def test_target_includes_submarines(self):
        """The decision (2026-11-22): the target is a ship of 50 m or more, submarines included (the text only: no rule hash)."""
        from sat import blind_prior as bp
        for doc in (json.loads((OUT / "eval" / "presence.json").read_text(encoding="utf-8")),
                    json.loads((OUT / "eval" / "meta.json").read_text(encoding="utf-8")), bp.build(OUT, "QGZ")):
            self.assertIn("50 m 以上の艦", doc["target"])
            self.assertIn("潜水艦を含む", doc["target"])
            self.assertNotIn("水上艦", doc["target"])

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

    def test_published_file_is_what_the_code_makes(self):
        """opt_fill.json against the code: every part but counts and error is what render() writes today, the sums are those of
        its rows, and the rows of the sites with a quay_blind file are that file's trial columns (both are written in the same
        run; the confirm files they read are published by another run, so the values themselves are not made again here)."""
        p = OUT / of.OPT_FILL
        if not p.exists():
            self.skipTest("opt_fill.json not written yet")
        doc = json.loads(p.read_text(encoding="utf-8"))
        want = json.loads(of.render({}, None)[1])
        self.assertEqual({k: v for k, v in doc.items() if k not in ("counts", "error")}, {k: v for k, v in want.items() if k not in ("counts", "error")})
        if doc["counts"] is None:
            self.assertTrue(doc["error"])
            return
        self.assertIsNone(doc["error"])
        for sid, c in doc["counts"].items():
            if c.get("error"):
                continue
            rows = c["rows"]
            self.assertEqual(c["scenes"], len(rows), sid)
            self.assertEqual(c["latest"], rows[-1], sid)
            self.assertEqual(c["present_sum"], sum(r["opt_fill_present"] for r in rows), sid)
            self.assertEqual(c["present_sum"], c["present_owner_sum"] + c["present_p_sum"], sid)
            for r in rows:
                self.assertEqual(r["opt_fill_present"], r["opt_fill_present_owner"] + r["opt_fill_present_p"], (sid, r["acquired_utc"]))
                kinds = [b.split("|")[3] for b in r["opt_fill_basis"].split(";") if b]
                self.assertEqual(len(kinds), r["opt_fill_scored"])
            f = OUT / ep.BLIND_DIR / f"{sid}.csv"
            self.assertEqual(c["in_quay_blind"], f.exists(), sid)
            if f.exists():
                with f.open(encoding="utf-8", newline="") as fh:
                    csv_rows = {r["acquired_utc"]: r for r in csv.DictReader(fh)}
                self.assertEqual(sorted(csv_rows), [r["acquired_utc"] for r in rows], sid)
                for r in rows:
                    self.assertEqual({k: csv_rows[r["acquired_utc"]][k] for k in of.COLUMNS}, {k: str(r[k]) if k != "opt_fill_version" else of.VERSION for k in of.COLUMNS}, sid)

    def test_lushun_is_in_the_file_alone(self):
        """A site without a resident table (no quay_blind file) gets the trial values in opt_fill.json alone, and no quay_blind row."""
        site = [s for s in ep.load_sites() if s["id"] == "LSN"]
        fill = of.prepare(OUT, ["LSN"])
        self.assertEqual(ep.blind_counts(OUT, site, fill), {})
        if not fill["LSN"]["rows"]:
            self.skipTest("no stored scene of Lushun with a mark in the confirm ledger")
        self.assertTrue(all(not r["in_quay_blind"] for r in fill["LSN"]["rows"].values()))


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
            doc = json.loads((out / of.OPT_FILL).read_text(encoding="utf-8"))     # the earlier file: its counts are not left standing
            self.assertIsNone(doc["counts"])
            self.assertIn("KeyError", doc["error"])

    def test_only_the_public_write_fails(self):
        """write_public alone fails inside refresh: quay_blind, quay_naval50 are written as always, the warning is printed, and one
        more write of counts null is tried (here it works: the earlier counts do not stand)."""
        real = of.write_public
        calls = []

        def flaky(out, ctx, error=None):
            calls.append(error)
            if len(calls) == 1:
                raise OSError("disk")
            return real(out, ctx, error)
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stderr"), mock.patch("builtins.print") as pr, \
                mock.patch.object(of, "write_public", side_effect=flaky):
            out = Path(d) / "sat"
            shutil.copytree(OUT, out, ignore=shutil.ignore_patterns("images", "color", "s2", "chips", "change", "windows"))
            before = {f.name: f.read_text(encoding="utf-8") for f in (out / ep.BLIND_DIR).glob("*.csv")}
            sites = [s for s in ep.load_sites() if s["id"] in ("XSN", "QGZ")]
            r = ep.refresh(out, sites, now=datetime(2026, 11, 22, 6, 0, tzinfo=timezone.utc))
            self.assertIn("OSError", r["opt_fill"]["error"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(set(r["blind"]), {"XSN", "QGZ"})
            self.assertTrue(r["naval50"])
            printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
            self.assertIn("::warning title=sat opt_fill::opt_fill.json の書き出しに失敗", printed)
            doc = json.loads((out / of.OPT_FILL).read_text(encoding="utf-8"))
            self.assertIsNone(doc["counts"])
            self.assertIn("OSError", doc["error"])
            for name in ("XSN.csv", "QGZ.csv"):     # the trial columns are made (the context did not fail); the file is whole
                with (out / ep.BLIND_DIR / name).open(encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(rows and all(x["opt_fill_version"] == of.VERSION for x in rows), name)
                self.assertEqual(len(rows) + 1, len(before[name].splitlines()), name)


class ByHandTest(unittest.TestCase):
    def test_main_rewrites_opt_fill(self):
        """python -m sat.eval_presence (by hand) makes quay_blind's trial columns and opt_fill.json from one context, as refresh."""
        seen = {}
        fake = {"truth": "s2", "other_truth": {"truth": "s1"}}
        with mock.patch.object(ep, "evaluate", return_value=fake), mock.patch.object(ep, "write"), mock.patch.object(ep, "table", return_value=[]), \
                mock.patch.object(ep, "write_naval50"), mock.patch.object(ep.pres1, "safe_write"), mock.patch("sys.stderr"), \
                mock.patch.object(ep, "opt_fill_prepare", return_value=({"X": {}}, None)), \
                mock.patch.object(ep, "write_blind", side_effect=lambda out, sites, fill: seen.setdefault("blind", fill) and {}), \
                mock.patch.object(ep, "opt_fill_publish", side_effect=lambda out, fill, err: seen.setdefault("public", (fill, err))):
            self.assertEqual(ep.main(["--out", str(OUT)]), 0)
        self.assertIs(seen["blind"], seen["public"][0])
        self.assertIsNone(seen["public"][1])


if __name__ == "__main__":
    unittest.main()
