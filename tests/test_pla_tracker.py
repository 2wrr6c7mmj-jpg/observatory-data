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
MND = ("report_date,format,aircraft,median_line_or_adiz,plan_vessels,official_ships\n"
       "2020-11-01,sw-adiz,,,,\n"
       "2026-10-28,around-taiwan,6,,7,\n"
       "2026-10-29,around-taiwan,6,5,7,\n"
       "2026-10-30,around-taiwan,6,3,7,1\n")


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
        self.assertEqual(c["days_aircraft_and_ships_equal"], 3)
        self.assertEqual(c["fields"]["ships_total_vs_plan_plus_official"]["rate"], 1.0)
        self.assertEqual(c["fields"]["ships_total"]["by_month"], {"2026-10": "2/3"})
        self.assertEqual(c["offsets"]["0"]["equal"], 7)
        self.assertIn("一致", P.compare_text(c))


class RunTest(unittest.TestCase):
    def setUp(self):
        # Keep the collector's ::notice/::warning lines out of the CI annotations: a test that
        # prints "PLA Activity Tracker を使わない：... disallowed" reads like a real skip (2026-10-31).
        self.notices = []
        patcher = mock.patch.object(P, "notice", lambda msg, level="notice", title="pla-tracker":
                                    self.notices.append((level, title, msg)))
        patcher.start()
        self.addCleanup(patcher.stop)
        polite._cache.clear()
        self.addCleanup(polite._cache.clear)
        # never read the published docs/data/pla-tracker/status.json (its last_request_utc) in tests
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_patch = mock.patch.object(P, "OUT", Path(tmp.name) / "published")
        out_patch.start()
        self.addCleanup(out_patch.stop)

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
            self.assertEqual(st["last_run"]["access"], "robots")
            self.assertNotIn("permission", st["last_run"])
            self.assertTrue(st["last_request_utc"])
            with open(out / "daily.csv", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("CC BY 4.0", meta["attribution"]["license"])
            self.assertIn("Adam Pan", meta["attribution"]["author"])
            self.assertEqual(meta["attribution"]["url"], P.SITE_URL)
            self.assertIn("permissions.json", meta["access"])
            self.assertIn("公式の関与・承認を示すものではない", meta["attribution"]["disclaimer"])
            self.assertIn("official involvement or endorsement", meta["attribution"]["credit"])
            daily = json.loads((out / "daily.json").read_text(encoding="utf-8"))
            self.assertEqual(daily["attribution"], meta["attribution"])
            self.assertIn("precedence", meta)
            # second run on the next UTC day: nothing new
            from datetime import timedelta
            with mock.patch.object(P, "now_utc", return_value=P.now_utc() + timedelta(days=1)):
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
        # without a permission entry (the real permissions.json has one for pla-tracker)
        with tempfile.TemporaryDirectory() as t, mock.patch.object(polite, "raw_get", getter), \
                mock.patch.object(polite, "load_permissions", return_value={}):
            self.assertEqual(P.run(Path(t) / "out"), 0)
            self.assertFalse((Path(t) / "out" / "daily.csv").exists())
        polite._cache.clear()

    # ---- explicit permission (collectors/permissions.json)
    PERM = {"pla-tracker": {"granted_by": "maker", "received": "2026-10-31T19:20+09:00", "channel": "user report",
                            "scope": {"urls": [P.CSV_URL], "max_requests_per_day": 1}, "evidence": "pending",
                            "review_by": "2026-11-30"}}

    def disallow_getter(self):
        calls = []

        def getter(url, ua, timeout, headers, max_bytes):
            calls.append(url)
            if url.endswith("robots.txt"):
                return polite.Response(url, 200, {"content-type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
            return polite.Response(url, 200, {"content-type": "text/csv"}, SAMPLE.encode())
        getter.calls = calls
        return getter

    def run_with(self, out, perms, today, now, getter=None):
        g = getter or self.disallow_getter()
        polite._cache.clear()
        with mock.patch.object(polite, "raw_get", g), \
                mock.patch.object(polite, "load_permissions", return_value=perms), \
                mock.patch.object(polite, "_today_utc", return_value=today), \
                mock.patch.object(P, "now_utc", return_value=now), \
                mock.patch.object(P, "OUT", out):
            rc = P.run(out, mnd_csv=out / "none.csv")
        return rc, g.calls

    def test_permission_fetches_once_a_day_and_says_so(self):
        from datetime import date, datetime, timezone
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            now = datetime(2026, 11, 1, 6, 37, tzinfo=timezone.utc)
            rc, calls = self.run_with(out, self.PERM, date(2026, 11, 1), now)
            self.assertEqual(rc, 0)
            self.assertEqual(calls, ["https://pla-tracker.skyfaring.net/robots.txt", P.CSV_URL])
            st = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(st["last_run"]["result"], "ok")
            self.assertEqual(st["last_run"]["access"], "permission")
            self.assertEqual(st["last_run"]["permission"]["evidence"], "pending")
            self.assertEqual(st["last_run"]["permission"]["file"], "collectors/permissions.json")
            self.assertIn("overridden by explicit permission", st["last_run"]["robots"])
            self.assertEqual(st["last_request_utc"], "2026-11-01T06:37:00Z")
            self.assertTrue(any(title == "pla-tracker permission" for _, title, _ in self.notices))
            # the same UTC day again: no request at all, status.json untouched
            before = (out / "status.json").read_text(encoding="utf-8")
            rc, calls = self.run_with(out, self.PERM, date(2026, 11, 1), now.replace(hour=20))
            self.assertEqual((rc, calls), (0, []))
            self.assertEqual((out / "status.json").read_text(encoding="utf-8"), before)
            # the next day: one request again
            rc, calls = self.run_with(out, self.PERM, date(2026, 11, 2), now.replace(day=2))
            self.assertEqual(calls[-1], P.CSV_URL)

    def allow_getter(self):
        calls = []

        def getter(url, ua, timeout, headers, max_bytes):
            calls.append(url)
            if url.endswith("robots.txt"):
                return polite.Response(url, 200, {"content-type": "text/plain"}, b"User-agent: *\nAllow: /\n")
            return polite.Response(url, 200, {"content-type": "text/csv"}, SAMPLE.encode())
        getter.calls = calls
        return getter

    def test_once_a_day_also_when_robots_allows(self):
        from datetime import date, datetime, timezone
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            now = datetime(2026, 11, 1, 6, 37, tzinfo=timezone.utc)
            rc, calls = self.run_with(out, {}, date(2026, 11, 1), now, self.allow_getter())
            self.assertEqual(calls[-1], P.CSV_URL)
            st = json.loads((out / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((st["last_run"]["access"], st["last_request_utc"]), ("robots", "2026-11-01T06:37:00Z"))
            rc, calls = self.run_with(out, {}, date(2026, 11, 1), now.replace(hour=23), self.allow_getter())
            self.assertEqual((rc, calls), (0, []))
            rc, calls = self.run_with(out, {}, date(2026, 11, 2), now.replace(day=2, hour=0), self.allow_getter())
            self.assertEqual(calls[-1], P.CSV_URL)

    def test_probe_path_respects_the_published_status(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as t:
            published = Path(t) / "published"
            published.mkdir()
            (published / "status.json").write_text(json.dumps({"last_request_utc": "2026-11-01T06:37:00Z"}), encoding="utf-8")
            g = self.allow_getter()
            polite._cache.clear()
            with mock.patch.object(polite, "raw_get", g), mock.patch.object(P, "OUT", published), \
                    mock.patch.object(P, "now_utc", return_value=datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)):
                rc = P.main(["--out", str(Path(t) / "probe"), "--save-raw", str(Path(t) / "records.csv")])
            self.assertEqual((rc, g.calls), (0, []))
            self.assertFalse((Path(t) / "records.csv").exists())
            # the next day the probe fetches once
            g = self.allow_getter()
            polite._cache.clear()
            with mock.patch.object(polite, "raw_get", g), mock.patch.object(P, "OUT", published), \
                    mock.patch.object(P, "now_utc", return_value=datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)):
                rc = P.main(["--out", str(Path(t) / "probe"), "--save-raw", str(Path(t) / "records.csv")])
            self.assertEqual(g.calls.count(P.CSV_URL), 1)
            self.assertTrue((Path(t) / "records.csv").exists())

    def test_no_or_expired_permission_skips_without_fetching(self):
        from datetime import date, datetime, timezone
        for perms, today, why in [({}, date(2026, 11, 1), "no permission recorded"),
                                  (self.PERM, date(2026, 12, 1), "expired")]:
            with self.subTest(why=why), tempfile.TemporaryDirectory() as t:
                out = Path(t) / "out"
                rc, calls = self.run_with(out, perms, today, datetime(2026, 12, 1, tzinfo=timezone.utc))
                self.assertEqual(rc, 0)
                self.assertEqual(calls, ["https://pla-tracker.skyfaring.net/robots.txt"])
                self.assertFalse((out / "daily.csv").exists())
                self.assertTrue(any(why in m for _, _, m in self.notices))


if __name__ == "__main__":
    unittest.main()
