import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from collectors import pla_tracker as P
from collectors import polite

HEADER = "date,aircraft_total,median_line_cross,cross_rate,aircraft_type,ships_total,activity_start,activity_end,special_event\n"
SAMPLE = HEADER + (
    "2026-10-28,6,,0,,7,,,\n"
    "2026-10-29,6,5,0.83,J-16,7,,,\n"
    "2026-10-30,6,3,0.5,,8,,,\n"
    "2026-10-31,12,10,0.83,J-10;J-16,9,2026-10-30 06:00,2026-10-31 06:00,\"Joint, combat readiness patrol\"\n"
)
MND = ("report_date,format,aircraft,median_line_or_adiz,plan_vessels\n"
       "2020-11-01,sw-adiz,,,\n"
       "2026-10-28,around-taiwan,6,,7\n"
       "2026-10-29,around-taiwan,6,5,7\n"
       "2026-10-30,around-taiwan,6,3,7\n")


class ParseTest(unittest.TestCase):
    def test_parse(self):
        rows, skipped = P.parse_csv("﻿" + SAMPLE)
        self.assertEqual(skipped, [])
        self.assertEqual(rows["2026-10-28"]["median_line_cross"], None)
        self.assertEqual(rows["2026-10-31"]["aircraft_total"], 12)
        self.assertEqual(rows["2026-10-31"]["special_event"], "Joint, combat readiness patrol")

    def test_bad_rows_are_skipped(self):
        rows, skipped = P.parse_csv(HEADER + "x,1,,,,,,,\n2026-01-01,abc,,,,,,,\n2026-01-02,999,,,,,,,\n2026-01-03,1.0,,,,2,,,\n")
        self.assertEqual(list(rows), ["2026-01-03"])
        self.assertEqual(len(skipped), 3)

    def test_missing_columns(self):
        with self.assertRaises(P.Stop):
            P.parse_csv("date,aircraft_total\n2026-01-01,1\n")


class MergeTest(unittest.TestCase):
    def test_add_revise_keep(self):
        old, _ = P.parse_csv(SAMPLE)
        old = {d: {**r, "retrieved_utc": "T0"} for d, r in old.items()}
        fresh, _ = P.parse_csv(SAMPLE.replace("2026-10-30,6,3,0.5,,8", "2026-10-30,6,3,0.5,,7")
                               .replace("2026-10-28,6,,0,,7,,,\n", "") + "2026-11-01,1,0,0,,2,,,\n")
        rows, ch = P.merge(old, fresh, "T1")
        self.assertEqual(ch["added"], 1)
        self.assertEqual(ch["revised_count"], 1)
        self.assertEqual(ch["revised"][0]["changed"], {"ships_total": [8, 7]})
        self.assertEqual(ch["kept_not_in_source"], ["2026-10-28"])
        self.assertEqual(rows["2026-10-29"]["retrieved_utc"], "T0")
        self.assertEqual(rows["2026-10-30"]["retrieved_utc"], "T1")
        self.assertIn("2026-10-28", rows)

    def test_short_csv_not_used(self):
        old = {f"2026-01-{i:02d}": {} for i in range(1, 11)}
        with self.assertRaises(P.Stop):
            P.merge(old, {"2026-01-01": {}}, "T")


class CompareTest(unittest.TestCase):
    def test_compare(self):
        rows, _ = P.parse_csv(SAMPLE)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "mnd.csv"
            p.write_text(MND, encoding="utf-8")
            mnd = P.read_mnd(p)
        self.assertNotIn("2020-11-01", mnd)
        c = P.compare(rows, mnd)
        self.assertEqual(c["overlap_days"], 3)
        self.assertEqual(c["fields"]["aircraft_total"]["rate"], 1.0)
        self.assertEqual(c["fields"]["ships_total"]["equal"], 2)
        self.assertEqual(c["fields"]["ships_total"]["examples_largest"], [{"date": "2026-10-30", "tracker": 8, "mnd": 7}])
        self.assertEqual(c["fields"]["median_line_cross"]["comparable"], 2)
        self.assertEqual(c["days_all_three_equal"], 2)
        self.assertEqual(c["offsets"]["0"]["equal"], 7)
        self.assertIn("一致", P.compare_text(c))


class RunTest(unittest.TestCase):
    def test_run_writes_files_once_fetched(self):
        def getter(url, ua, timeout, headers, max_bytes):
            if url.endswith("robots.txt"):
                return polite.Response(url, 200, {"content-type": "text/plain"}, b"User-agent: *\nAllow: /\n")
            return polite.Response(url, 200, {"content-type": "text/csv"}, SAMPLE.encode())
        polite._cache.clear()
        with tempfile.TemporaryDirectory() as t, mock.patch.object(polite, "raw_get", getter):
            out = Path(t) / "out"
            mnd = Path(t) / "mnd.csv"
            mnd.write_text(MND, encoding="utf-8")
            self.assertEqual(P.run(out, mnd_csv=mnd), 0)
            st = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["source"]["last_day"], "2026-10-31")
            self.assertEqual(st["last_run"]["added"], 4)
            self.assertIn("allowed", st["last_run"]["robots"])
            with open(out / "daily.csv", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("CC BY 4.0", meta["attribution"]["license"])
            self.assertIn("precedence", meta)
            # second run: nothing new
            self.assertEqual(P.run(out, mnd_csv=mnd), 0)
            st = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["last_run"]["added"], 0)
        polite._cache.clear()

    def test_disallowed_is_skipped(self):
        def getter(url, ua, timeout, headers, max_bytes):
            if url.endswith("robots.txt"):
                return polite.Response(url, 200, {"content-type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
            raise AssertionError("CSV must not be fetched")
        polite._cache.clear()
        with tempfile.TemporaryDirectory() as t, mock.patch.object(polite, "raw_get", getter):
            self.assertEqual(P.run(Path(t) / "out"), 0)
            self.assertFalse((Path(t) / "out" / "daily.csv").exists())
        polite._cache.clear()


if __name__ == "__main__":
    unittest.main()
