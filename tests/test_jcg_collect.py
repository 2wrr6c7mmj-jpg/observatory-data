import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from collectors import jcg_senkaku as j
from collectors import polite
from tests.jcg_fixtures import table

BASE = "https://www.kaiho.mlit.go.jp/mission/senkaku/"
PAGE = f"""<table><tr><td>令和8年</td><td><a href="{BASE}data_R8_9.pdf">9月</a>
<a href="data_R8_10.pdf">10月</a></td></tr></table>"""
SEPT = {d: (4, None) for d in range(1, 31)}
WORDS = {
    BASE + "data_R8_9.pdf": table(title="令和8年9月", as_of="( 9/30 現在)", vals=SEPT, last=30),
    BASE + "data_R8_10.pdf": table(),
}


class Server:
    def __init__(self, robots=(200, b"User-agent: *\nDisallow: /cgi/\n"), challenge=False):
        self.robots, self.challenge, self.calls = robots, challenge, []

    def __call__(self, url, ua, timeout, headers, max_bytes):
        self.calls.append((url, dict(headers or {})))
        assert "github.com/2wrr6c7mmj-jpg/observatory-data" in ua
        if url.endswith("/robots.txt"):
            return polite.Response(url, self.robots[0], {"content-type": "text/plain"}, self.robots[1])
        if self.challenge:
            return polite.Response(url, 503, {"content-type": "text/html"}, b"<title>Just a moment...</title>")
        if url == j.PAGE_URL:
            return polite.Response(url, 200, {"content-type": "text/html"}, PAGE.encode())
        if headers and headers.get("If-None-Match") == '"e1"':
            return polite.Response(url, 304, {}, b"")
        return polite.Response(url, 200, {"content-type": "application/pdf", "etag": '"e1"',
                                          "last-modified": "Wed, 28 Oct 2026 01:00:00 GMT"},
                               b"%PDF-" + url.encode())


def fake_words(body):
    return WORDS[body[5:].decode()]


class Collect(unittest.TestCase):
    def run_collect(self, server, d, today=datetime(2026, 10, 30, 3, tzinfo=timezone.utc)):
        polite._cache.clear()
        with mock.patch.object(j, "OUT", Path(d)), mock.patch.object(j, "_sleep"), \
                mock.patch.object(j, "notice"), mock.patch.object(j, "warning") as warn, \
                mock.patch.object(j, "now_utc", return_value=today), \
                mock.patch.object(j, "pdf_words", side_effect=fake_words), \
                mock.patch.object(polite, "raw_get", side_effect=server):
            code = j.collect(250)
        return code, warn

    def test_daily_rows_and_no_refetch(self):
        with tempfile.TemporaryDirectory() as d:
            s = Server()
            code, _ = self.run_collect(s, d)
            self.assertEqual(code, 0)
            daily = json.loads((Path(d) / "daily.json").read_text())
            rows = [dict(zip(daily["fields"], r)) for r in daily["rows"]]
            self.assertEqual(len(rows), 30 + 28)  # October up to the as-of date
            first = rows[0]
            self.assertEqual((first["date"], first["contiguous_zone"], first["territorial_sea"], first["check"]),
                             ("2026-09-01", 4, 0, "match"))
            self.assertEqual(first["window_start_utc"], "2026-08-31T15:00:00Z")
            self.assertEqual(first["source_last_modified_utc"], "2026-10-28T01:00:00Z")
            oct1 = next(r for r in rows if r["date"] == "2026-10-01")
            self.assertEqual((oct1["contiguous_zone"], oct1["territorial_sea"]), (8, 4))
            pdfs = json.loads((Path(d) / "pdfs.json").read_text())["pdfs"]
            self.assertEqual(len(pdfs[BASE + "data_R8_10.pdf"]["sha256"]), 64)
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertEqual(status["not_published"], {})
            self.assertIn("allowed", status["last_run"]["robots"])

            s2 = Server()
            self.run_collect(s2, d)
            fetched = [(u, h) for u, h in s2.calls if u.endswith(".pdf")]
            # September is complete and not asked for again; October is asked conditionally.
            self.assertEqual(fetched, [(BASE + "data_R8_10.pdf", {
                "If-None-Match": '"e1"', "If-Modified-Since": "Wed, 28 Oct 2026 01:00:00 GMT"})])
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertEqual(status["last_run"]["not_modified"], 1)

    def test_robots_disallow_skips_without_fetching(self):
        with tempfile.TemporaryDirectory() as d:
            s = Server(robots=(200, b"User-agent: *\nDisallow: /mission/\n"))
            code, warn = self.run_collect(s, d)
            self.assertEqual(code, 0)
            self.assertEqual([u for u, _ in s.calls], ["https://www.kaiho.mlit.go.jp/robots.txt"])
            self.assertIn("skipped", warn.call_args.args[0])
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertIn("disallows", status["last_run"]["result"])

    def test_unverifiable_robots_skips(self):
        with tempfile.TemporaryDirectory() as d:
            s = Server(robots=(503, b""))
            code, _ = self.run_collect(s, d)
            self.assertNotEqual(code, 0)
            self.assertEqual(len(s.calls), 1)

    def test_challenge_stops(self):
        with tempfile.TemporaryDirectory() as d:
            s = Server(robots=(404, b""), challenge=True)
            code, warn = self.run_collect(s, d)
            self.assertNotEqual(code, 0)
            self.assertEqual(len(s.calls), 2)  # robots.txt, then the page; nothing after the challenge
            self.assertIn("challenge", warn.call_args.args[0])

    def test_mismatching_month_is_not_published(self):
        bad = table(title="令和8年9月", as_of="( 9/30 現在)", vals=SEPT, last=30)
        bad = [w for w in bad if not (w[1] == "4" and w[4] == 95 + 5 * 14)]  # a value lost on day 5
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(WORDS, {BASE + "data_R8_9.pdf": bad}):
            self.run_collect(Server(), d)
            status = json.loads((Path(d) / "status.json").read_text())
            daily = json.loads((Path(d) / "daily.json").read_text())
        self.assertIn(BASE + "data_R8_9.pdf", status["not_published"])
        self.assertFalse(any(r[0].startswith("2026-09") for r in daily["rows"]))


if __name__ == "__main__":
    unittest.main()
