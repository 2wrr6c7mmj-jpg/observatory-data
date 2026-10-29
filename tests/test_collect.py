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
            taiwan_mnd.collect(pages=3, stop_when_known=True)
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


if __name__ == "__main__":
    unittest.main()
