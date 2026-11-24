import unittest
from datetime import date

from exercises import forecast as F


def rows(name, starts):
    return [{"name": name, "start": s} for s in starts]


class Forecast(unittest.TestCase):
    def test_names_are_normalised(self):
        for t in ("Joint Sea-2025", "海上联合-2024", "Морское взаимодействие-2021"):
            self.assertEqual(F.series_of(t), "joint-sea")
        self.assertIsNone(F.series_of("日米共同訓練"))

    def test_one_occurrence_per_year_the_earliest(self):
        occ = F.occurrences(rows("Keen Sword 24", ["2024-10-23", "2024-11-01"]) + [{"name": "Keen Sword", "start": None}])
        self.assertEqual(occ["keen-sword"], {2024: date(2024, 10, 23)})

    def test_biennial_series_called_in_its_window_with_the_denominator(self):
        occ = F.occurrences(rows("Keen Sword", [f"{y}-10-{d}" for y, d in ((2016, 30), (2018, 29), (2020, 26), (2022, 28), (2024, 23))]))
        calls = F.forecast(occ, date(2026, 10, 1))
        self.assertEqual([c["series"] for c in calls], ["keen-sword"])
        c = calls[0]
        self.assertEqual((c["year"], c["period_years"]), (2026, 2))
        self.assertTrue(c["window_start"] <= "2026-10-23" <= c["window_end"])
        self.assertIn("隔年の機会6回のうち5回", c["basis"])
        self.assertEqual(F.forecast(occ, date(2027, 10, 1)), [], "not in the off year")

    def test_too_few_years_or_no_season_is_not_called(self):
        few = F.occurrences(rows("Ulchi Freedom Shield", ["2016-08-22", "2017-08-21", "2018-08-20", "2019-08-19"]))
        self.assertEqual(F.forecast(few, date(2026, 8, 1)), [])
        spread = F.occurrences(rows("Malabar", ["2018-06-08", "2019-09-26", "2020-11-03", "2021-08-23", "2022-11-08", "2023-08-11"]))
        self.assertEqual(F.forecast(spread, date(2026, 8, 1)), [])

    def test_a_call_is_scored_after_its_window(self):
        log = [{"series": "keen-sword", "year": 2026, "window_start": "2026-10-14", "window_end": "2026-11-08", "probability": 0.8, "outcome": None},
               {"series": "rimpac", "year": 2026, "window_start": "2026-06-01", "window_end": "2026-07-10", "probability": 0.9, "outcome": None}]
        occ = {"keen-sword": {2026: date(2026, 10, 20)}}
        F.score(log, occ, date(2026, 11, 24))
        self.assertEqual([f["outcome"] for f in log], [1, 0])
        b = F.brier(log)
        self.assertEqual(b["2026-11"], {"n": 1, "brier": 0.04})
        self.assertEqual(b["2026-07"], {"n": 1, "brier": 0.81})


if __name__ == "__main__":
    unittest.main()
