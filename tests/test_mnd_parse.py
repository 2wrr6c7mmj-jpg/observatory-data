import unittest
from datetime import date, datetime, timezone

from collectors.mnd_parse import adiz_types, counts, parse_report, reparse, report_date, report_links

NOW = datetime(2026, 10, 30, 2, 0, tzinfo=timezone.utc)

EN_2025 = """<html><body><div class="date">2025.06.28</div><h2>PLA activities in the waters and airspace around Taiwan</h2>
<p>一、Date:<br>6 a.m. Jun. 27 (Fri.) to 6 a.m. Jun. 28 (Sat.) (UTC+8)</p>
<p>二、PLA activities:<br>8 sorties of PLA aircraft and 6 PLAN vessels operating around Taiwan were detected up until
6 a.m. (UTC+8) today. 6 out of 8 sorties crossed the median line and entered Taiwan's southwestern and eastern ADIZ.</p>
<script>var x = "99 sorties of PLA aircraft";</script></body></html>"""

EN_OFFICIAL = """<div>2026.01.26</div><p>5 PLA aircraft, 8 PLAN vessels and 3 official ships operating around Taiwan
were detected up until 6 a.m. (UTC+8) today.</p>"""

EN_2023 = """<p>PLA activities in the waters and airspace around Taiwan Dec. 25, 2023 09:00(UTC+8)</p>
<p>1.Date: 6 to 6 a.m. (UTC+8) Sunday to Monday, Dec. 24-25</p>
<p>2.PLA activities: 2 PLA aircraft and 4 PLAN vessels around Taiwan were detected by 6 a.m.(UTC+8) today.</p>"""

ZH_2023 = """<div>區域動態 112.12.25 發布單位:政治作戰局</div><p>中共解放軍臺海周邊海、空域動態 發布時間:112年12月25日0900時</p>
<p>一、日期:中華民國112年12月24日(星期日)0600時至12月25日(星期一)0600時止。</p>
<p>二、活動動態:迄0600時止,偵獲共機2架次、共艦4艘次,在臺海周邊活動。</p>
<p>三、上述期間共機無逾越海峽中線及西南空域,故無提供航跡圖。</p>"""

ZH_2026 = """<div>2026.07.31</div><p>共偵獲中共軍機27架次、共艦9艘及公務船2艘持續在台海周邊活動,
統計至今日上午6時止,共機27架次,其中22架次逾越海峽中線,進入北部、中部、西南及東部空域</p>"""


OLD_DAY = """<p>PLA Activities 2021.10.04 Issuing Authority：Political Warfare Bureau</p><p>Air activities in the southwestern ADIZ of R.O.C.</p>
<p>一、Date: October 4,2021（Monday daytime）</p><p>二、Aircraft Type: Thirty four J-16 Two SU-30 Two Y-8 ASW Two KJ-500 AEW&amp;C Twelve H-6</p>
<p>三、Activity Area: Flight paths as illustrated</p>"""
OLD_2020 = """<p>PLA Activities 2020.11.15 Issuing Authority：Administration Office</p><p>1、Date: Nov. 15, 2020（Sunday）</p>
<p>2、Aircraft Type: One Y-8 ASW One Y-8 RECCE</p><p>3、Activity Area: Flight path as illustrated</p>"""
EN_2024 = """<p>PLA Activities 2024.05.24</p><p>1.Date： 6 to 6 a.m. (UTC+8) Thursday to Friday, May 23-24</p>
<p>2.PLA activities： 49 PLA aircraft, 19 PLAN vessels, and 7 CCG vessels operating around Taiwan were detected up until 6 a.m. (UTC+8) today.
35 of the aircraft crossed the median line of the Taiwan Strait and entered Taiwan’s southwestern ADIZ.</p>"""
EN_2023_CROSS = """<p>PLA Activities 2023.09.18</p><p>1.Date： 6 to 6 a.m. (UTC+8) Sunday to Monday, Sep. 17-18</p><p>2.PLA activities： 103 PLA aircraft and 9 PLAN vessels
around Taiwan were detected by 6 a.m.(UTC+8) today. 40 of the detected aircraft had crossed the median line</p>"""


class Parse(unittest.TestCase):
    def test_2024_crossing_and_coast_guard(self):
        r = parse_report(EN_2024, "u", NOW)
        self.assertEqual((r["report_date"], r["aircraft"], r["median_line_or_adiz"], r["plan_vessels"], r["official_ships"]),
                         ("2024-05-24", 49, 35, 19, 7))
        self.assertEqual(counts(EN_2023_CROSS)["median_line_or_adiz"], 40)

    def test_old_southwestern_adiz_format(self):
        r = parse_report(OLD_DAY, "u", NOW)
        self.assertTrue(r["parsed"])
        self.assertEqual((r["format"], r["report_date"], r["period"]), ("sw-adiz", "2021-10-04", "daytime"))
        self.assertEqual(r["adiz_types"], {"J-16": 34, "SU-30": 2, "Y-8 ASW": 2, "KJ-500 AEW&C": 2, "H-6": 12})
        self.assertEqual(r["adiz_aircraft"], 52)
        self.assertIsNone(r["aircraft"])  # a different measure: not merged into the around-Taiwan total
        self.assertEqual(r["window_start_utc"], "2021-10-03T16:00Z")
        r = parse_report(OLD_2020, "u", NOW)
        self.assertEqual((r["report_date"], r["adiz_aircraft"], r["period"]), ("2020-11-15", 2, None))

    def test_number_word_joined_to_type_by_hyphen(self):
        # Report 79236 (2021-10-26): "One-Y-8 EW" is one Y-8 EW, not part of the "Y-8 RECCE" name.
        t = ("Aircraft Type: One Y-8 C3 One Y-8 RECCE One-Y-8 EW Two J-16 One Mi-17 Cargo "
             "One WZ-10 Attack 三、Activity Area")
        types = adiz_types(t)
        self.assertEqual(types, {"Y-8 C3": 1, "Y-8 RECCE": 1, "Y-8 EW": 1, "J-16": 2, "Mi-17 Cargo": 1, "WZ-10 Attack": 1})
        self.assertEqual(sum(types.values()), 7)
        # Digits after a hyphen stay part of the type name, and "Twenty-Four" is still 24.
        self.assertEqual(adiz_types("Aircraft Type: Twenty-Four J-16 Two-KJ-500 AEW&C Keywords"),
                         {"J-16": 24, "KJ-500 AEW&C": 2})
        # Brackets around each entry (2021-03..09) and a stray ")" (2022-05) are not part of the name.
        self.assertEqual(adiz_types("Aircraft Type: (Two J-10) (One Y-8 ASW) 三、Activity Area: x"),
                         {"J-10": 2, "Y-8 ASW": 1})
        self.assertEqual(adiz_types("Aircraft Type: (One Y-8 ASW) (One KJ-500 AEW&C) (Six J-16) Activity Area: x"),
                         {"Y-8 ASW": 1, "KJ-500 AEW&C": 1, "J-16": 6})
        self.assertEqual(adiz_types("Aircraft Type: One Y-8 EW Five J-16) Four J-11 Activities Area: x"),
                         {"Y-8 EW": 1, "J-16": 5, "J-11": 4})

    def test_english_current_format(self):
        r = parse_report(EN_2025, "https://www.mnd.gov.tw/en/news/plaact/84583", NOW)
        self.assertTrue(r["parsed"])
        self.assertEqual(r["report_date"], "2025-06-28")
        self.assertEqual((r["aircraft"], r["median_line_or_adiz"], r["plan_vessels"]), (8, 6, 6))
        self.assertIsNone(r["official_ships"])  # not stated → unknown, not zero
        self.assertEqual(r["window_start_utc"], "2025-06-26T22:00Z")
        self.assertEqual(r["window_end_utc"], "2025-06-27T22:00Z")

    def test_window_overrides_unrelated_page_date(self):
        page = ("<nav>2026.10.01 Latest news</nav><p>2026.10.29</p><p>1.Date: 6 a.m. Oct. 28 (Tue.) to 6 a.m. Oct. 29 (Wed.) (UTC+8)</p>"
                "<p>6 sorties of PLA aircraft and 7 PLAN vessels</p>")
        r = parse_report(page, "u", NOW)
        self.assertEqual((r["report_date"], r["date_from"]), ("2026-10-29", "window"))

    def test_window_across_new_year(self):
        page = "<p>2026.01.01</p><p>6 a.m. Dec. 31 (Wed.) to 6 a.m. Jan. 1 (Thu.)</p><p>3 PLA aircraft</p>"
        self.assertEqual(parse_report(page, "u", NOW)["report_date"], "2026-01-01")
        page = "<p>2025.12.31</p><p>Date: 6 to 6 a.m. Tuesday to Wednesday, Dec. 30-31</p><p>3 PLA aircraft</p>"
        self.assertEqual(parse_report(page, "u", NOW)["report_date"], "2025-12-31")

    def test_kept_text_reparses_offline(self):
        page = "<nav>Menu About MND</nav>" + EN_2024 + "<p>Keywords： Regional dynamics</p><footer>Privacy</footer>"
        r = parse_report(page, "u", NOW)
        self.assertTrue(r["main_text"].startswith("PLA Activities 2024.05.24"))
        self.assertNotIn("Privacy", r["main_text"])
        again = reparse({**r, "parser_version": 0})
        self.assertEqual({k: again[k] for k in ("report_date", "aircraft", "median_line_or_adiz", "official_ships")},
                         {k: r[k] for k in ("report_date", "aircraft", "median_line_or_adiz", "official_ships")})

    def test_scripts_are_ignored(self):
        self.assertEqual(parse_report(EN_2025, "u", NOW)["aircraft"], 8)

    def test_official_ships(self):
        c = counts(EN_OFFICIAL)
        self.assertEqual((c["aircraft"], c["plan_vessels"], c["official_ships"]), (5, 8, 3))
        self.assertEqual(report_date(EN_OFFICIAL), date(2026, 1, 26))

    def test_english_2023(self):
        r = parse_report(EN_2023, "u", NOW)
        self.assertEqual((r["report_date"], r["aircraft"], r["plan_vessels"]), ("2023-12-25", 2, 4))

    def test_chinese_roc_calendar_and_no_crossing(self):
        r = parse_report(ZH_2023, "u", NOW)
        self.assertEqual((r["report_date"], r["aircraft"], r["plan_vessels"]), ("2023-12-25", 2, 4))
        self.assertEqual(r["median_line_or_adiz"], 0)

    def test_chinese_current(self):
        c = counts(ZH_2026)
        self.assertEqual((c["aircraft"], c["median_line_or_adiz"], c["plan_vessels"], c["official_ships"]), (27, 22, 9, 2))

    def test_number_words(self):
        c = counts("eleven sorties of PLA aircraft and five PLAN vessels were detected")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (11, 5))

    def test_unreadable_page_is_kept_for_review(self):
        r = parse_report("<p>2025.01.08</p><p>Download area</p>", "u", NOW)
        self.assertFalse(r["parsed"])
        self.assertIn("Download area", r["main_text"])
        self.assertIsNone(r["aircraft"])

    def test_list_links(self):
        html = ('<a href="/en/news/plaact/84583">a</a><a href="/en/news/plaact/84583">dup</a>'
                '<a href="https://www.mnd.gov.tw/en/news/plaact/84560">b</a><a href="/en/Publication/1">x</a>'
                "<a class='x' href='news/plaact/84500'>c</a><a href=\"news/plaactlist/2\">next</a>")
        self.assertEqual(report_links(html), [("/en/news/plaact/84583", 84583),
                                              ("https://www.mnd.gov.tw/en/news/plaact/84560", 84560),
                                              ("news/plaact/84500", 84500)])


FOOTER = ("<p>Keywords PLA</p><p>Tel: 886-2-2311-6117 No. 409, Bei'an Rd., Zhongshan Dist., Taipei 104</p>"
          "<footer>Phone 886 2 8509 9999 409 1997</footer>")
AFC_2022 = ("<p>PLA Activities 2022.08.05 Issuing Authority：Air Force Command Headquarters Share "
            "Air activities of PLA aircraft around the R.O.C. airspace Date: August 5, 2022 (Friday) "
            "Aircraft Type: Seven J-10 Six J-11 Ten J-16 Twenty Four SU-30 One Y-8 EW One Y-8 ASW "
            "Activities Area: Flight paths as illustrated. Reactions: CAP aircraft tasked.</p>") + FOOTER
SW_2022 = ("<p>PLA Activities 2022.08.03 Issuing Authority：Air Force Command Headquarters Share "
           "Air activities in the southwestern ADIZ of R.O.C. Date: August 2, 2022 (Tuesday) "
           "Aircraft Type: EIGHT J-11 Ten J-16 One KJ-500 AEW&C Activities Area: Flight paths as illustrated.</p>") + FOOTER
NO_LABEL = ("<p>PLA Activities 2021.10.21 Issuing Authority：Political Warfare Bureau Share Date: October 21,2021"
            "（Thursday daytime） Two J-16 Two J-11 Activity Area: Flight paths as illustrated</p>")


class Fixes2026(unittest.TestCase):
    def test_footer_numbers_are_not_aircraft(self):
        r = parse_report(AFC_2022, "u", NOW)
        self.assertTrue(r["parsed"])
        self.assertEqual(r["adiz_aircraft"], 49)
        self.assertEqual(r["format"], "afc-around-roc")
        self.assertEqual(r["report_date"], "2022-08-05")
        self.assertNotIn("886", r["main_text"])
        r = parse_report(SW_2022, "u", NOW)
        self.assertEqual((r["format"], r["report_date"], r["adiz_aircraft"]), ("sw-adiz", "2022-08-02", 19))

    def test_air_force_report_without_heading_follows_its_series(self):
        page = AFC_2022.replace("Air activities of PLA aircraft around the R.O.C. airspace ", "")
        r = parse_report(page, "u", NOW)
        self.assertEqual((r["format"], r["format_basis"]), ("afc-around-roc", "series"))
        r = parse_report(NO_LABEL, "u", NOW)  # Political Warfare Bureau, 2021
        self.assertEqual((r["format"], r["format_basis"]), ("sw-adiz", "series"))

    def test_implausible_count_is_not_published(self):
        page = AFC_2022.replace("Twenty Four SU-30", "409 SU-30")
        r = parse_report(page, "u", NOW)
        self.assertFalse(r["parsed"])
        self.assertIn("implausible", r["parse_error"])

    def test_type_list_without_label(self):
        r = parse_report(NO_LABEL, "u", NOW)
        self.assertTrue(r["parsed"])
        self.assertEqual((r["format"], r["period"], r["adiz_types"]), ("sw-adiz", "daytime", {"J-16": 2, "J-11": 2}))

    def test_periods(self):
        for text, want in (("（Thursday morning）", "morning"), ("（Thursday afteroon）", "afternoon"),
                           ("(Sunday Night)", "nighttime"), ("（Tuesday）", None)):
            r = parse_report(NO_LABEL.replace("（Thursday daytime）", text), "u", NOW)
            self.assertEqual(r["period"], want, text)

    def test_no_aircraft_and_vessels_stated(self):
        c = counts("2.PLA activities： No PLA aircraft and PLAN vessel operating around Taiwan were detected up until 6 a.m.")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (0, 0))
        c = counts("2.PLA activities： No PLA aircraft and PLAN ship operating around Taiwan were detected")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (0, 0))

    def test_missing_space_and_plan_without_noun(self):
        c = counts("2.PLA activities： 17PLA aircraft and 3 PLAN vessels around Taiwan were detected. "
                   "9 of the detected aircraft had crossed the median line")
        self.assertEqual((c["aircraft"], c["plan_vessels"], c["median_line_or_adiz"]), (17, 3, 9))
        c = counts("2.PLA activities： 4 PLA aircraft and 10 PLAN around Taiwan were detected by 6 a.m.")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (4, 10))

    def test_all_aircraft_entered(self):
        c = counts("2.PLA activities： 3 PLA aircraft and 4 PLAN vessels around Taiwan were detected by 6 a.m. today. "
                   "The detected aircraft had entered Taiwan’s southwest and east ADIZ.")
        self.assertEqual(c["median_line_or_adiz"], 3)
        c = counts("2.PLA activities: 1 PLA aircraft and 9 PLAN vessels operating around Taiwan were detected. "
                   "The aircraft crossed the median line of the Taiwan Strait.")
        self.assertEqual(c["median_line_or_adiz"], 1)

    def test_only_ships_or_only_aircraft_listed(self):
        c = counts("2.PLA activities： 4 PLAN vessels around Taiwan were detected by 6 a.m.(UTC+8) today.")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (0, 4))
        c = counts("二、PLA activities： 8 PLAN ships and 1 official ship operating around Taiwan were detected as of 6 a.m.")
        self.assertEqual((c["aircraft"], c["plan_vessels"], c["official_ships"]), (0, 8, 1))
        c = counts("2.PLA activities： 5 PLA aircraft around Taiwan were detected by 6 a.m.(UTC+8) today.")
        self.assertEqual((c["aircraft"], c["plan_vessels"]), (5, 0))
        self.assertIsNone(c["official_ships"])  # not part of the rule: stays unknown


if __name__ == "__main__":
    unittest.main()
