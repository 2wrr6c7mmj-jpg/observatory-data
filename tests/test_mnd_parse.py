import unittest
from datetime import date, datetime, timezone

from collectors.mnd_parse import counts, parse_report, report_date, report_links

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


class Parse(unittest.TestCase):
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
        self.assertIn("Download area", r["text_excerpt"])
        self.assertIsNone(r["aircraft"])

    def test_list_links(self):
        html = ('<a href="/en/news/plaact/84583">a</a><a href="/en/news/plaact/84583">dup</a>'
                '<a href="https://www.mnd.gov.tw/en/news/plaact/84560">b</a><a href="/en/Publication/1">x</a>'
                "<a class='x' href='news/plaact/84500'>c</a><a href=\"news/plaactlist/2\">next</a>")
        self.assertEqual(report_links(html), [("/en/news/plaact/84583", 84583),
                                              ("https://www.mnd.gov.tw/en/news/plaact/84560", 84560),
                                              ("news/plaact/84500", 84500)])


if __name__ == "__main__":
    unittest.main()
