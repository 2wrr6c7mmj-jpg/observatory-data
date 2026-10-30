import unittest

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


if __name__ == "__main__":
    unittest.main()
