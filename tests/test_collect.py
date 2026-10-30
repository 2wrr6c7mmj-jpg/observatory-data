import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from collectors import taiwan_mnd

LIST = '<base href="/en/"><a href="news/plaact/2">b</a><a href="/en/news/plaact/1">a</a>'
REPORTS = {
    "/en/news/plaact/1": "<p>2026.10.28</p><p>4 sorties of PLA aircraft and 7 PLAN vessels were detected.</p>",
    "/en/news/plaact/2": "<p>2026.10.29</p><p>Download area</p>",
}


def fake_fetch(url):
    if "plaactlist" in url:
        return LIST if url.endswith("/1") else ""
    return REPORTS[url.replace(taiwan_mnd.BASE, "")]


class Collect(unittest.TestCase):
    def test_collect_writes_dataset_and_skips_known_reports(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(taiwan_mnd, "OUT", Path(d)), \
                mock.patch.object(taiwan_mnd, "fetch", side_effect=fake_fetch) as f:
            self.assertEqual(taiwan_mnd.collect(pages=3, stop_when_known=True), 0)
            daily = json.loads((Path(d) / "daily.json").read_text())
            self.assertEqual(len(daily["rows"]), 1)
            row = dict(zip(daily["fields"], daily["rows"][0]))
            self.assertEqual((row["report_date"], row["aircraft"], row["plan_vessels"]), ("2026-10-28", 4, 7))
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertEqual(status["unparsed_ids"], ["2"])
            self.assertIn("report_date,format,aircraft", (Path(d) / "daily.csv").read_text())
            calls = f.call_count
            taiwan_mnd.collect(pages=3, stop_when_known=True)
            self.assertEqual(f.call_count - calls, 1)  # only the list page; known reports are not refetched


class Daily(unittest.TestCase):
    def test_daytime_and_nighttime_are_added_and_corrections_replace(self):
        base = {"parsed": True, "format": "sw-adiz", "report_date": "2021-10-04"}
        reports = {
            "10": {**base, "period": "daytime", "adiz_aircraft": 52, "adiz_types": {"J-16": 34, "H-6": 18}, "source_url": "a"},
            "11": {**base, "period": "nighttime", "adiz_aircraft": 4, "adiz_types": {"J-16": 4}, "source_url": "b"},
            "12": {**base, "period": "nighttime", "adiz_aircraft": 4, "adiz_types": {"J-16": 4}, "source_url": "b2"},
            "20": {"parsed": True, "format": "around-taiwan", "report_date": "2023-01-01", "aircraft": 5, "source_url": "c"},
            "21": {"parsed": True, "format": "around-taiwan", "report_date": "2023-01-01", "aircraft": 6, "source_url": "d"},
        }
        rows = {r["report_date"]: r for r in taiwan_mnd.daily_rows(reports)}
        self.assertEqual(rows["2021-10-04"]["adiz_aircraft"], 56)
        self.assertEqual(rows["2021-10-04"]["adiz_types"], "H-6:18;J-16:38")
        self.assertEqual(rows["2021-10-04"]["source_url"], "a b2")
        self.assertEqual(rows["2023-01-01"]["aircraft"], 6)


def http_error(code, retry_after=None):
    import email.message
    import urllib.error
    h = email.message.Message()
    if retry_after is not None:
        h["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://x", code, "err", h, None)


class ListPageRetry(unittest.TestCase):
    def test_temporary_errors_are_retried_then_succeed(self):
        calls = {"n": 0}

        def flaky(url):
            if "plaactlist" in url and url.endswith("/1") and calls["n"] < 2:
                calls["n"] += 1
                raise http_error(503) if calls["n"] == 1 else http_error(429, retry_after=200)
            return fake_fetch(url)

        waits = []
        with mock.patch.object(taiwan_mnd, "fetch", side_effect=flaky), \
                mock.patch.object(taiwan_mnd, "_sleep", side_effect=waits.append), \
                mock.patch.object(taiwan_mnd, "notice"):
            links = taiwan_mnd.list_page(1)
        self.assertEqual(len(links), 2)
        self.assertEqual(waits, [taiwan_mnd.LIST_RETRY_WAITS[0], 200])

    def test_retry_after_is_clamped_between_default_and_cap(self):
        wait = taiwan_mnd.retry_wait
        self.assertEqual(wait(http_error(429, retry_after=1), 30), 30)
        self.assertEqual(wait(http_error(429, retry_after=90), 30), 90)
        self.assertEqual(wait(http_error(429, retry_after=99999), 30), taiwan_mnd.RETRY_AFTER_CAP)
        self.assertEqual(wait(http_error(503), 30), 30)

    def test_persistent_failure_ends_the_run_and_keeps_progress(self):
        def failing_page2(url):
            if "plaactlist" in url and url.endswith("/2"):
                raise http_error(502)
            return fake_fetch(url)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(taiwan_mnd, "OUT", Path(d)), \
                mock.patch.object(taiwan_mnd, "fetch", side_effect=failing_page2), \
                mock.patch.object(taiwan_mnd, "_sleep") as sl, \
                mock.patch.object(taiwan_mnd, "notice"), \
                mock.patch.object(taiwan_mnd, "warning") as warn:
            self.assertNotEqual(taiwan_mnd.collect(5, stop_when_known=False), 0)
            stored = json.loads((Path(d) / "reports.json").read_text(encoding="utf-8"))["reports"]
        self.assertEqual(set(stored), {"1", "2"})
        self.assertEqual(sl.call_count, len(taiwan_mnd.LIST_RETRY_WAITS))
        self.assertTrue(any("ending this run" in c.args[0] for c in warn.call_args_list))

    def test_other_errors_are_not_retried(self):
        with mock.patch.object(taiwan_mnd, "fetch", side_effect=http_error(403)), \
                mock.patch.object(taiwan_mnd, "_sleep") as sl:
            with self.assertRaises(Exception):
                taiwan_mnd.list_page(1)
        sl.assert_not_called()


class ReportFetchErrors(unittest.TestCase):
    """Run 40983532615 died on a bare TimeoutError from a report page."""

    def test_read_timeout_is_a_bare_timeout_error_and_counts_as_transient(self):
        import socket
        import threading
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        held = []
        t = threading.Thread(target=lambda: held.append(srv.accept()[0]), daemon=True)
        t.start()  # accepts the connection, never answers
        try:
            with mock.patch.object(taiwan_mnd, "FETCH_TIMEOUT", 0.3), mock.patch.object(taiwan_mnd, "PAUSE", 0):
                with self.assertRaises(TimeoutError) as cm:
                    taiwan_mnd.fetch(f"http://127.0.0.1:{srv.getsockname()[1]}/")
        finally:
            t.join(1)
            for c in held:
                c.close()
            srv.close()
        import urllib.error
        self.assertNotIsInstance(cm.exception, urllib.error.URLError)  # why `except URLError` missed it
        self.assertTrue(taiwan_mnd.transient(cm.exception))

    def test_dropped_connections_count_as_transient(self):
        import http.client
        for e in (http.client.RemoteDisconnected("x"), http.client.IncompleteRead(b""), ConnectionResetError()):
            self.assertTrue(taiwan_mnd.transient(e), e)
        self.assertFalse(taiwan_mnd.transient(ValueError()))

    def test_fetch_keeps_the_pause_after_a_failure(self):
        with mock.patch.object(taiwan_mnd.urllib.request, "urlopen", side_effect=TimeoutError("t")), \
                mock.patch.object(taiwan_mnd.time, "sleep") as sl:
            with self.assertRaises(TimeoutError):
                taiwan_mnd.fetch("https://x")
        sl.assert_called_once_with(taiwan_mnd.PAUSE)

    def run_collect(self, fetch, pages=1):
        waits = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(taiwan_mnd, "OUT", Path(d)), \
                mock.patch.object(taiwan_mnd, "fetch", side_effect=fetch), \
                mock.patch.object(taiwan_mnd, "_sleep", side_effect=waits.append), \
                mock.patch.object(taiwan_mnd, "notice"), \
                mock.patch.object(taiwan_mnd, "warning") as warn:
            code = taiwan_mnd.collect(pages, stop_when_known=False)
            stored = json.loads((Path(d) / "reports.json").read_text(encoding="utf-8"))["reports"]
        return code, set(stored), waits, warn

    def test_one_timeout_is_retried_once(self):
        calls = {"n": 0}

        def flaky(url):
            if url.endswith("/plaact/2") and calls["n"] == 0:
                calls["n"] += 1
                raise TimeoutError("The read operation timed out")
            return fake_fetch(url)

        code, stored, waits, _ = self.run_collect(flaky)
        self.assertEqual((code, stored, waits), (0, {"1", "2"}, [taiwan_mnd.REPORT_RETRY_WAIT]))

    def test_a_report_that_keeps_timing_out_is_skipped_and_the_run_goes_on(self):
        def stuck(url):
            if url.endswith("/plaact/2"):
                raise TimeoutError("The read operation timed out")
            return fake_fetch(url)

        code, stored, waits, _ = self.run_collect(stuck)
        self.assertEqual((code, stored), (0, {"1"}))
        self.assertEqual(len(waits), 1)  # one retry, not more

    def test_many_failures_in_a_row_end_the_run_and_keep_progress(self):
        pages = {f"/en/news/plaactlist/{p}": "".join(f'<a href="/en/news/plaact/{p * 10 + i}">r</a>' for i in range(5))
                 for p in (1, 2)}

        def site(url):
            path = url.replace(taiwan_mnd.BASE, "")
            if path in pages:
                return pages[path]
            if "plaactlist" in path:
                return ""
            if path.startswith("/en/news/plaact/2"):  # every report on page 2 times out
                raise TimeoutError("The read operation timed out")
            return REPORTS["/en/news/plaact/1"]

        code, stored, waits, warn = self.run_collect(site, pages=5)
        self.assertEqual(code, 75)
        self.assertEqual(stored, {str(10 + i) for i in range(5)})
        self.assertEqual(len(waits), taiwan_mnd.REPORT_FAILURE_LIMIT)
        self.assertTrue(any("ending this run" in c.args[0] for c in warn.call_args_list))

    def test_unexpected_error_still_saves_what_was_fetched(self):
        def broken(url):
            if url.endswith("/plaact/1"):
                raise RuntimeError("boom")
            return fake_fetch(url)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(taiwan_mnd, "OUT", Path(d)), \
                mock.patch.object(taiwan_mnd, "fetch", side_effect=broken):
            with self.assertRaises(RuntimeError):
                taiwan_mnd.collect(1, stop_when_known=False)
            stored = json.loads((Path(d) / "reports.json").read_text(encoding="utf-8"))["reports"]
        self.assertEqual(set(stored), {"2"})


if __name__ == "__main__":
    unittest.main()
