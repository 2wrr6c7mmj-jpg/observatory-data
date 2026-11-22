"""dark-1 (sat/radar_dark.py): the sections the radar cannot judge, marked apart from 0 ships, and nothing counted changes.

The list is the one of the miss-1 design §5.2-1 / §7 PR-2: the hires-1 submarine berths (verdict U) and the sections where the
owner said "present" 3 times or more and the radar never did; the second reason holds on a day only from the answers of earlier
days. Pure Python: these tests run without numpy too.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sat import radar_dark as rd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
PUB = json.loads((OUT / rd.PUBLIC).read_text(encoding="utf-8"))


def hires_u():
    with (OUT / "labels" / "hires-berth-read.csv").open(encoding="utf-8", newline="") as f:
        return {(r["site_id"], r["section_id"]) for r in csv.DictReader(f) if r["verdict"] == "U"}


def file_marks():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")).get(rd.KEY) or [] for p in sorted(rd.CLASS_DIR.glob("*.json"))}


class ListTest(unittest.TestCase):
    def test_the_list_is_the_union_of_the_design(self):
        # hires-1's submarine berths, and over the evaluation cells of a section: the owner present >= 3, the radar present 0
        want = hires_u()
        for site, secs in PUB["evidence"].items():
            for sec, rows in secs.items():
                if sum(y for _, y, _ in rows) >= 3 and sum(s for _, _, s in rows) == 0:
                    want.add((site, sec))
        got = {(site, e["section"]) for site, es in file_marks().items() for e in es}
        self.assertEqual(got, want)
        self.assertEqual(len(hires_u()), 77)
        static12 = {(s, k) for s, secs in PUB["evidence"].items() for k, rows in secs.items()
                    if sum(y for _, y, _ in rows) >= 3 and sum(x for _, _, x in rows) == 0}
        self.assertEqual(len(static12), 12)                     # miss-1 §3.1: 12 sections (64 present cells)
        self.assertEqual(sum(sum(y for _, y, _ in PUB["evidence"][s][k]) for s, k in static12), 64)
        self.assertEqual(got - hires_u(), {("LSN", "995110510:L:00"), ("QDH", "69732009:L:37")})   # the two warship quays
        for site, es in file_marks().items():
            for e in es:
                self.assertIs(e["radar_dark"], True); self.assertTrue(e["basis"]); self.assertTrue(e["reasons"])
                self.assertEqual(e["reasons"], [r for r in rd.REASONS if r in e["reasons"]])
                self.assertEqual(rd.SUBMARINE in e["reasons"], (site, e["section"]) in hires_u())

    def test_the_files_agree(self):
        marks = rd.build_marks(rd._hires(), PUB["evidence"])
        self.assertEqual({k: v for k, v in file_marks().items() if v}, marks)
        self.assertEqual(PUB["sites"], marks)
        self.assertEqual(PUB["counts"], {s: len(v) for s, v in marks.items()})
        self.assertNotIn("ZJG", marks)                                     # Zhanjiang's files do not change
        self.assertNotIn(rd.KEY, json.loads((rd.CLASS_DIR / "ZJG.json").read_text(encoding="utf-8")))

    def test_the_evidence_is_the_evaluation_sheet(self):
        # the marks were built from eval/presence.json; the daily run rewrites it, so a drift is reported (rebuild), not failed:
        # the published counts never depend on the marks
        ev = rd.evidence_of(rd.load_presence(OUT))
        if ev != PUB["evidence"] or rd.current(OUT)[0] != PUB["sites"]:
            self.skipTest("eval/presence.json changed since the marks were built: run python -m sat.radar_dark build")
        self.assertEqual(rd.check(OUT), [])


class DateRuleTest(unittest.TestCase):
    """The second reason on a day d uses only the cells dated before d."""

    def brute(self, rows, d):
        before = [r for r in rows if r[0] < d]
        return sum(y for _, y, _ in before) >= rd.MIN_PRESENT and not any(s for _, _, s in before)

    def days(self, rows):
        from datetime import date, timedelta
        ds = set()
        for r in rows:
            x = date.fromisoformat(r[0])
            ds |= {str(x - timedelta(days=1)), str(x), str(x + timedelta(days=1))}
        return sorted(ds) + ["2099-01-01"]

    def test_every_mark_against_the_brute_force(self):
        n = 0
        for site, secs in PUB["evidence"].items():
            marks = {e["section"]: e for e in PUB["sites"].get(site, [])}
            for sec, rows in secs.items():
                e = marks.get(sec, {"reasons": []})
                for d in self.days(rows):
                    self.assertEqual(rd.OWNER_NONE in rd.is_dark(e, d), self.brute(rows, d), (site, sec, d)); n += 1
        self.assertGreater(n, 500)

    def test_a_later_answer_does_not_reach_back(self):
        rows = [["2026-01-01", 1, 0], ["2026-02-01", 1, 0], ["2026-03-01", 0, 0], ["2026-04-01", 1, 0]]
        e = {"reasons": [rd.OWNER_NONE], "owner": dict(zip(("since", "until"), rd.owner_window(rows)))}
        self.assertEqual(e["owner"], {"since": "2026-04-01", "until": None})
        self.assertEqual([bool(rd.is_dark(e, d)) for d in ("2026-04-01", "2026-04-02")], [False, True])   # the day of the 3rd: not yet
        later = rows + [["2026-05-01", 1, 1], ["2026-06-01", 1, 0]]       # the radar saw it on 05-01
        e2 = {"reasons": [rd.OWNER_NONE], "owner": dict(zip(("since", "until"), rd.owner_window(later)))}
        for d in ("2026-03-15", "2026-04-02", "2026-04-30", "2026-05-01"):
            self.assertEqual(rd.is_dark(e2, d), rd.is_dark(e, d), d)       # the days up to the radar's sighting: as before
        self.assertEqual(rd.is_dark(e2, "2026-05-02"), [])
        self.assertIsNone(rd.owner_window([["2026-01-01", 1, 1], ["2026-02-01", 1, 0], ["2026-03-01", 1, 0]]))   # seen before the 3rd
        self.assertIsNone(rd.owner_window([["2026-01-01", 1, 0], ["2026-02-01", 1, 0]]))
        self.assertEqual(rd.is_dark({"reasons": [rd.SUBMARINE]}, "2001-01-01"), [rd.SUBMARINE])          # the berth's trait: any day


class NothingCountedChangesTest(unittest.TestCase):
    def test_the_presence_rules_hash_the_class_files_without_the_marks(self):
        with tempfile.TemporaryDirectory() as d:
            cdir = Path(d) / "cls"; cdir.mkdir(); out = Path(d) / "out"; (out / "eval").mkdir(parents=True)
            shutil.copy(OUT / "eval" / "presence.json", out / "eval" / "presence.json")
            for s in ("ZJG", "XSN", "QDH"):
                p = rd.CLASS_DIR / f"{s}.json"
                doc = json.loads(p.read_text(encoding="utf-8")); doc.pop(rd.KEY, None)
                (cdir / f"{s}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            before = {p.stem: p.read_bytes() for p in cdir.glob("*.json")}
            m = rd.build_marks(rd._hires(), PUB["evidence"])
            rd.write({k: v for k, v in m.items() if k in before}, PUB["evidence"], out, cdir)
            for p in cdir.glob("*.json"):
                self.assertEqual(rd.class_bytes_without_marks(p.read_text(encoding="utf-8")), before[p.stem], p.stem)
            self.assertEqual((cdir / "ZJG.json").read_bytes(), before["ZJG"])
            self.assertNotEqual((cdir / "XSN.json").read_bytes(), before["XSN"])
            rd.write({}, PUB["evidence"], out, cdir)                                       # removed again: the bytes as before
            self.assertEqual({p.stem: p.read_bytes() for p in cdir.glob("*.json")}, before)

    def test_the_class_files_round_trip(self):
        # the hash rule above needs every class file to be json.dumps(indent=1, ensure_ascii=False) + "\n" with the marks last
        for p in sorted(rd.CLASS_DIR.glob("*.json")):
            t = p.read_text(encoding="utf-8"); doc = json.loads(t)
            self.assertEqual(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", t, p.name)
            if rd.KEY in doc:
                self.assertEqual(list(doc)[-1], rd.KEY, p.name)

    def test_the_lines_and_their_classes_are_those_without_the_marks(self):
        from sat import berths
        for p in sorted(rd.CLASS_DIR.glob("*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(rd.class_bytes_without_marks(p.read_text(encoding="utf-8")))["lines"], doc["lines"])
            self.assertEqual(berths.load_classes(p.stem)["lines"], doc["lines"])

    def test_quay_blind_gets_the_reason_columns_only(self):
        from sat import eval_presence as ep
        self.assertEqual(ep.BLIND_COLUMNS[-3:], ("radar_dark_sections", "radar_dark_seen", "radar_dark_reasons"))
        sites = [s for s in ep.load_sites() if s["id"] in ("XSN", "QDH", "QGZ")]
        with_marks = ep.blind_counts(OUT, sites)
        with mock.patch.object(rd, "marks", return_value={}):
            without = ep.blind_counts(OUT, sites)
        self.assertEqual(sorted(with_marks), sorted(without))
        new = ("radar_dark_sections", "radar_dark_seen", "radar_dark_reasons")
        for s in with_marks:
            self.assertEqual([{k: v for k, v in r.items() if k not in new} for r in with_marks[s]],
                             [{k: v for k, v in r.items() if k not in new} for r in without[s]])
            self.assertTrue(all(r["radar_dark_sections"] == 0 and r["radar_dark_reasons"] == "" for r in without[s]))
        xsn = with_marks["XSN"]
        self.assertTrue(all(r["radar_dark_sections"] > 0 and rd.SUBMARINE in r["radar_dark_reasons"] for r in xsn))
        self.assertTrue(all(r["radar_dark_seen"] <= r["radar_dark_sections"] for s in with_marks for r in with_marks[s]))
        self.assertTrue(all(r["radar_dark_sections"] == 0 for r in with_marks["QGZ"]))          # Guzhenkou has no mark


if __name__ == "__main__":
    unittest.main()
