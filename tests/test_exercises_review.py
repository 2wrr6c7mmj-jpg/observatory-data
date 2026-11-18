"""Regression tests for the review of exercises/ (rules 7): wrong values that were published."""
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from collectors import polite
from exercises import dates, extract, merge, run

from tests.test_exercises import DOC, PLAIN, TEXT, answer

DAY = 86400000


def ms(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


class CountryWordsTest(unittest.TestCase):
    def test_ordinary_words_are_not_countries(self):
        self.assertEqual(extract.supported_countries(["米国", "中国", "ロシア"],
                                                     "Indonesia and Malaysia told us they plan to hold a drill"), [])
        for text in ("新型装備を披露した", "露出した", "計画を暴露", "露天で", "结露"):
            self.assertEqual(extract.supported_countries(["ロシア"], text), [], text)

    def test_places_named_after_a_country_are_not_the_country(self):
        cases = [("中国", "US and Philippine ships sailed in the South China Sea"),
                 ("中国", "drills in the East China Sea"),
                 ("中国", "中国地方の陸自駐屯地で訓練"),
                 ("フィリピン", "US carrier drills in the Philippine Sea"),
                 ("フィリピン", "フィリピン海で日米共同訓練"),
                 ("日本", "Russian ships in the Sea of Japan"),
                 ("日本", "中露の艦艇が日本海で演習"),
                 ("インド", "インド太平洋地域の安定"),
                 ("インド", "印度洋"),
                 ("台湾", "PLA ships transited the Taiwan Strait"),
                 ("モンゴル", "内モンゴル自治区で演習"),
                 ("モンゴル", "drills in Inner Mongolia"),
                 ("北朝鮮", "朝鲜半岛局势"),
                 ("フィリピン", "TOKYO (AFP) - Japan held a drill")]
        for country, text in cases:
            self.assertEqual(extract.supported_countries([country], text), [], text)

    def test_a_day_before_日米_is_not_japan(self):
        self.assertEqual(extract.supported_countries(["日本", "米国"], "11日米軍が発表した"), ["米国"])
        self.assertEqual(extract.supported_countries(["日本"], "本日米海軍と"), [])

    def test_abbreviations_count_in_capitals(self):
        self.assertEqual(extract.supported_countries(["米国", "中国", "韓国", "豪州"], "US, PLAN, ROK and ADF units"),
                         ["米国", "中国", "韓国", "豪州"])
        self.assertEqual(extract.supported_countries(["米国"], "U.S. Navy"), ["米国"])
        self.assertEqual(extract.supported_countries(["米国"], "the US-led drill"), ["米国"])
        self.assertEqual(extract.supported_countries(["韓国", "豪州"], "the rok and adf of it"), [])

    def test_names_of_countries_still_count(self):
        cases = [("ロシア", "露軍が演習"), ("ロシア", "中露の艦艇"), ("ロシア", "日露関係"), ("ロシア", "Russian Navy"),
                 ("ロシア", "中俄海军"), ("中国", "China and Russia"), ("中国", "the PLA Eastern Theater"),
                 ("中国", "中国海軍"), ("中国", "中俄海军"), ("日本", "日米共同訓練"), ("日本", "日本海上自卫队"),
                 ("米国", "米海軍"), ("米国", "American troops"), ("フィリピン", "Philippine Navy"),
                 ("フィリピン", "フィリピン海軍"), ("フィリピン", "日米比の共同訓練"), ("インド", "インド海軍"),
                 ("インド", "Indian Navy"), ("台湾", "台湾軍"), ("ベトナム", "中越两国"), ("モンゴル", "モンゴル軍"),
                 ("北朝鮮", "朝鲜人民军"), ("日本", "日本"), ("中国", "南シナ海で中国海警")]
        for country, text in cases:
            self.assertEqual(extract.supported_countries([country], text), [country], text)

    def test_every_table_name_names_itself(self):
        for c in extract.COUNTRY_WORDS:
            self.assertEqual(extract.canonical_country(c), c)
            if c == "タイ":   # bare タイ is a part of other words (ネクタイ); タイ軍・タイ王国 name it
                continue
            self.assertEqual(extract.supported_countries([c], c), [c], c)


class NumbersNotDatesTest(unittest.TestCase):
    def test_numbers_of_dates_are_not_counts(self):
        self.assertFalse(extract._number_in("11月12日に実施", 12))
        self.assertFalse(extract._number_in("11月12日に実施", 11))
        self.assertFalse(extract._number_in("令和8年11月", 8))
        self.assertFalse(extract._number_in("On Nov. 12, 2026, the drill began", 12))
        self.assertFalse(extract._number_in("On Nov. 12, 2026, the drill began", 2026))
        self.assertFalse(extract._number_in("held 3-5 May", 5))

    def test_counts_still_match(self):
        self.assertTrue(extract._number_in("11月12日に艦艇12隻", 12))
        self.assertTrue(extract._number_in("艦艇１６隻", 16))
        self.assertTrue(extract._number_in("about 2,000 personnel in 2026", 2000))
        self.assertTrue(extract._number_in("2026 troops took part", 2026))
        self.assertTrue(extract._number_in("12, 16 and 20 ships", 16))   # a list is not one number

    def test_validate_drops_a_count_that_is_a_day(self):
        v = extract.validate(answer(ships=12, aircraft=None, dateEvidence=None, place=None), PLAIN, "11月12日に訓練を実施", "2026-11-12")
        self.assertIsNone(v["ships"])
        v = extract.validate(answer(), DOC, TEXT, "2026-11-09")
        self.assertEqual((v["ships"], v["aircraft"]), (16, 2))

    def test_forces_numbers_are_not_dates(self):
        self.assertFalse(extract.numbers_supported("艦艇12隻", "11月12日に実施"))
        self.assertTrue(extract.numbers_supported("艦艇12隻（11月12日から）", "11月12日から艦艇12隻"))


class DatesWordsTest(unittest.TestCase):
    def test_length_in_days_is_not_the_last_day(self):
        self.assertEqual(dates.parse("11月3日から10日間", "2026-11-01"), ("2026-11-03", "2026-11-12", "day"))
        self.assertEqual(dates.parse("11月3日起为期10天", "2026-11-01"), ("2026-11-03", "2026-11-12", "day"))

    def test_month_names_are_whole_words(self):
        for words in ("involving 3 marines", "2 decks", "the 4 mayors", "5 junior officers"):
            self.assertIsNone(dates.parse(words, "2026-11-10"), words)
        self.assertEqual(dates.parse("early decisions on 5 May", "2026-11-10"), ("2026-05-05", "2026-05-05", "day"))
        self.assertEqual(dates.parse("early Dec.", "2026-11-10"), ("2026-12-01", "2026-12-10", "part"))
        self.assertEqual(dates.parse("Nov. 3-5", "2026-11-01"), ("2026-11-03", "2026-11-05", "day"))
        self.assertEqual(dates.parse("3 March", "2026-02-01"), ("2026-03-03", "2026-03-03", "day"))

    def test_relative_days_are_not_days(self):
        self.assertIsNone(dates.parse("3日前", "2026-11-10"))


def named(src, at, start=None, end=None, name="Keen Sword"):
    ans = {**answer(name=name), "start": start, "end": end or start}
    return merge.event_from({"url": "https://x/" + src, "source": src, "kind": "報道", "publishedAt": at}, ans)


class NamedMergeTest(unittest.TestCase):
    def test_next_years_undated_report_is_another_event(self):
        done = named("a", ms("2025-11-01"), "2025-11-10", "2025-11-20")
        later = named("b", ms("2026-10-20"))
        self.assertEqual(len(merge.merge_all([done, later])), 2)

    def test_undated_report_near_the_period_joins(self):
        dated = named("a", ms("2026-11-01"), "2026-11-10", "2026-11-20")
        for day in ("2026-09-20", "2026-11-15", "2026-12-15"):
            self.assertEqual(len(merge.merge_all([dated, named("b", ms(day))])), 1, day)
        self.assertEqual(len(merge.merge_all([dated, named("c", ms("2027-01-05"))])), 2)

    def test_two_undated_reports_need_close_reports(self):
        self.assertEqual(len(merge.merge_all([named("a", ms("2026-01-10")), named("b", ms("2026-02-01"))])), 1)
        self.assertEqual(len(merge.merge_all([named("a", ms("2025-11-10")), named("b", ms("2026-11-01"))])), 2)


def stored(url, name="Balikatan", start="2025-04-21"):
    return {"kind": "exercise", "name": name, "titleJa": name, "countries": ["米国", "フィリピン"], "region": "philippines",
            "place": None, "forces": None, "ships": None, "aircraft": None, "personnel": None, "content": None,
            "start": start, "end": start, "status": "completed", "dateText": None, "datePrecision": "day",
            "sources": [{"url": url, "title": name, "source": "usni", "sourceName": "USNI", "kind": "報道",
                         "publishedAt": ms(start)}],
            "firstReported": ms(start), "lastReported": ms(start), "official": False, "pubKnown": True}


class RunHistoryTest(unittest.TestCase):
    def run_main(self, d, docs=(), ask=None, fetch=None, robots=None):
        patches = [mock.patch.object(run.sources, "get_json", return_value={"records": [], "sources": []}),
                   mock.patch.object(run.sources, "official_candidates", return_value=list(docs)),
                   mock.patch.object(run, "ledger_events", return_value=[])]
        if ask is not None:
            patches.append(mock.patch.object(run.extract, "ask", side_effect=ask))
        if fetch is not None:
            patches.append(mock.patch.object(run, "fetch", side_effect=fetch))
        if robots is not None:
            patches.append(mock.patch.object(run.polite, "check_robots", return_value=robots))
        for p in patches:
            p.start()
        try:
            return run.main(["--out", d, "--minutes", "5"])
        finally:
            for p in patches:
                p.stop()

    def setup_files(self, d, rules, events_by_year, items=None):
        out = Path(d)
        (out / "events").mkdir(parents=True, exist_ok=True)
        for y, evs in events_by_year.items():
            (out / "events" / f"{y}.json").write_text(json.dumps({"year": y, "rules": rules, "events": evs}))
        (out / "analyses.json").write_text(json.dumps({"rules": rules, "items": items or {}}))
        return out

    def years(self, out):
        return {int(p.stem): json.loads(p.read_text())["events"] for p in (out / "events").glob("[0-9][0-9][0-9][0-9].json")}

    def test_change_of_rules_keeps_events_that_cannot_be_read_again(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.setup_files(d, run.RULES - 1, {2025: [stored("https://x/old")]})
            self.run_main(d)
            ys = self.years(out)
            self.assertIn(2025, ys)
            self.assertEqual([e["name"] for e in ys[2025]], ["Balikatan"])
            self.assertEqual(json.loads((out / "analyses.json").read_text())["rules"], run.RULES)

    def test_a_document_read_again_replaces_its_old_answer(self):
        doc = {"id": "d1", "url": "https://x/old", "title": "Balikatan drills", "source": "usni", "sourceName": "USNI",
               "kind": "報道", "publishedAt": ms("2026-11-10")}
        with tempfile.TemporaryDirectory() as d:
            out = self.setup_files(d, run.RULES - 1, {2025: [stored("https://x/old")]})   # a wrong year under old rules
            self.run_main(d, [doc], ask=lambda *a, **k: {"relevant": False}, fetch=lambda url: (200, b"<p>text</p>"))
            self.assertNotIn(2025, self.years(out))   # the new reading says not relevant: the old event goes

    def test_broken_files_stop_the_run(self):
        for broken in ("events/2025.json", "analyses.json"):
            with tempfile.TemporaryDirectory() as d:
                out = self.setup_files(d, run.RULES, {2025: [stored("https://x/old")]})
                (out / broken).write_text('{"events": [')
                with self.assertRaises(SystemExit):
                    self.run_main(d)
                self.assertEqual((out / broken).read_text(), '{"events": [')
                self.assertTrue((out / "events" / "2025.json").exists())

    def doc(self):
        return {"id": "d2", "url": "https://x/new", "title": "US, Japan drills near Okinawa", "source": "usni",
                "sourceName": "USNI", "kind": "報道", "publishedAt": ms("2026-11-10")}

    def test_a_body_that_could_not_be_fetched_is_fetched_again(self):
        asked = []

        def ask(*a, **k):
            asked.append(a)
            return {"relevant": False}

        def timeout(url):
            raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as d:
            out = self.setup_files(d, run.RULES, {})
            self.run_main(d, [self.doc()], ask=ask, fetch=timeout)
            self.assertEqual(asked, [])   # not read from the headline alone
            items = json.loads((out / "analyses.json").read_text())["items"]
            (entry,) = items.values()
            self.assertFalse(entry["ok"])
            self.assertEqual(entry["fetchTries"], 1)
            # a day later it is due again; after FETCH_TRIES tries the headline is read alone
            for i in range(2, run.FETCH_TRIES + 1):
                items = {k: {**v, "at": v["at"] - 2 * DAY} for k, v in items.items()}
                (out / "analyses.json").write_text(json.dumps({"rules": run.RULES, "items": items}))
                self.run_main(d, [self.doc()], ask=ask, fetch=lambda url: (503, b""))
                items = json.loads((out / "analyses.json").read_text())["items"]
            (entry,) = items.values()
            self.assertEqual(len(asked), 1)
            self.assertTrue(entry["ok"])
            self.assertEqual((entry["body"], entry["fetchTries"]), ("none", run.FETCH_TRIES))

    def test_robots_disallow_is_for_good(self):
        asked = []

        def blocked(url):
            raise polite.SourceBlocked("disallowed for this URL")

        with tempfile.TemporaryDirectory() as d:
            out = self.setup_files(d, run.RULES, {})
            self.run_main(d, [self.doc()], ask=lambda *a, **k: asked.append(a) or {"relevant": False}, fetch=blocked,
                          robots=polite.RobotsResult(False, True, "disallowed", by_rules=True))
            (entry,) = json.loads((out / "analyses.json").read_text())["items"].values()
            self.assertEqual(len(asked), 1)   # headline and summary only, once
            self.assertTrue(entry["ok"])
            self.assertNotIn("fetchTries", entry)

    def test_robots_that_cannot_be_checked_is_tried_again(self):
        def blocked(url):
            raise polite.SourceBlocked("robots.txt could not be read")

        with tempfile.TemporaryDirectory() as d:
            out = self.setup_files(d, run.RULES, {})
            self.run_main(d, [self.doc()], ask=lambda *a, **k: {"relevant": False}, fetch=blocked,
                          robots=polite.RobotsResult(False, False, "could not be read"))
            (entry,) = json.loads((out / "analyses.json").read_text())["items"].values()
            self.assertFalse(entry["ok"])


if __name__ == "__main__":
    unittest.main()
