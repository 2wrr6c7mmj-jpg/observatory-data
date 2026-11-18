import http.client
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from collectors import ll2

RECORD = {"id": "abc", "name": "Long March 6A | Group 12", "status": {"abbrev": "Success"}, "net": "2026-10-28T03:15:00Z",
          "launch_service_provider": {"name": "CASC", "country_code": "CHN"},
          "rocket": {"configuration": {"full_name": "Long March 6A"}},
          "mission": {"type": "Communications", "orbit": {"abbrev": "LEO"}},
          "pad": {"name": "LC-9A", "latitude": "38.8", "longitude": "111.6", "location": {"name": "Taiyuan", "country_code": "CHN"}}}


class LL2(unittest.TestCase):
    def test_compact_record(self):
        r = ll2.compact(RECORD)
        self.assertEqual((r["ll2_id"], r["status"], r["rocket"], r["orbit"], r["lat"], r["country"]),
                         ("abc", "Success", "Long March 6A", "LEO", 38.8, "CHN"))

    def test_merge_updates_status_and_keeps_first_seen(self):
        s = ll2.merge({}, [{**ll2.compact(RECORD), "status": "Go"}], "2026-10-27T00:00:00Z")
        s = ll2.merge(s, [ll2.compact(RECORD)], "2026-10-28T12:00:00Z")
        r = s["launches"]["abc"]
        self.assertEqual((r["status"], r["first_seen_utc"]), ("Success", "2026-10-27T00:00:00Z"))


class Fetch(unittest.TestCase):
    def run_main(self, out, get):
        buf = io.StringIO()
        with mock.patch.object(ll2, "OUT", Path(out)), mock.patch.object(ll2, "get", side_effect=get), redirect_stdout(buf):
            code = ll2.main()
        return code, buf.getvalue()

    def test_failure_while_reading_keeps_the_other_request(self):
        for exc in (TimeoutError("The read operation timed out"), http.client.IncompleteRead(b"{\"res", 1000),
                    ConnectionResetError("reset"), ValueError("Expecting value")):
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory() as d:
                def get(url):
                    if "/previous/" in url:
                        raise exc
                    return {"results": [RECORD]}
                code, out = self.run_main(d, get)
                self.assertEqual(code, 0)
                stored = json.loads((Path(d) / "ll2.json").read_text(encoding="utf-8"))
                self.assertEqual(list(stored["launches"]), ["abc"])
                self.assertIn(type(exc).__name__, out)
                self.assertEqual(os.listdir(d), ["ll2.json"])  # written atomically, no .tmp left

    def test_all_requests_failing_keeps_the_stored_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ll2.json"
            path.write_text('{"launches": {"old": {"ll2_id": "old"}}}\n', encoding="utf-8")
            for bad in (TimeoutError("timed out"), {"detail": "throttled"}, [1, 2], {"results": None}):
                def get(url):
                    if isinstance(bad, Exception):
                        raise bad
                    return bad
                with self.subTest(bad=bad):
                    code, out = self.run_main(d, get)
                    self.assertEqual(code, 0)
                    self.assertEqual(path.read_text(encoding="utf-8"), '{"launches": {"old": {"ll2_id": "old"}}}\n')
                    self.assertIn("stored data kept", out)

    def test_incomplete_body_from_urlopen_is_caught(self):
        class Body:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                raise http.client.IncompleteRead(b'{"results": [', 5000)
        with mock.patch("urllib.request.urlopen", return_value=Body()):
            rows, errors = ll2.fetch_all(datetime(2026, 11, 1, tzinfo=timezone.utc))
        self.assertEqual((rows, len(errors)), ([], 2))
        self.assertIn("IncompleteRead", errors[0])


if __name__ == "__main__":
    unittest.main()
