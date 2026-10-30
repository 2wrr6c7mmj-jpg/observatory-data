import unittest

from collectors.jcg_parse import check, decode_html, page_links, parse_words
import calendar
from pathlib import Path

from tests.jcg_fixtures import PAGE, from_layout, table, word

LAYOUTS = Path(__file__).resolve().parent / "fixtures" / "jcg"


def layout(name):
    return from_layout(next(LAYOUTS.glob(f"*_{name}.pdf.txt")).read_text(encoding="utf-8"))

URL = "https://www.kaiho.mlit.go.jp/mission/senkaku/senkaku.html"


class Page(unittest.TestCase):
    def test_links_carry_the_row_year_and_link_month(self):
        links = page_links(PAGE, URL)
        got = [(l["url"].rsplit("/", 1)[-1], l["year"], l["month"]) for l in links]
        self.assertEqual(got, [("data_h24_09.pdf", 2012, 9), ("data_h24_10.pdf", 2012, 10),
                               ("data_h30_06.pdf", 2018, 6), ("data_h31_04.pdf", 2019, 4),
                               ("data_R_05.pdf", 2019, 5), ("data_R8_10.pdf", 2026, 10), ("odd.pdf", 2026, None)])
        self.assertTrue(all(l["url"].startswith("https://www.kaiho.mlit.go.jp/mission/senkaku/") for l in links))


    def test_shift_jis_page(self):
        body = ('<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">' + PAGE).encode("cp932")
        links = page_links(decode_html(body, "text/html"), URL)
        self.assertEqual((links[0]["year"], links[0]["month"]), (2012, 9))
        self.assertEqual(decode_html("令和".encode(), "text/html; charset=UTF-8"), "令和")


class Pdf(unittest.TestCase):
    def test_table_totals_and_as_of(self):
        p = parse_words(table(), 2026, 10)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual(p["as_of"], "2026-10-28")
        self.assertEqual(p["title_year_month"], "2026-10")
        self.assertEqual(p["days"]["1"], {"cz": 8, "ts": 4, "cz_cell": "number", "ts_cell": "number"})
        self.assertEqual(p["days"]["2"]["ts"], None)
        self.assertEqual(p["days"]["30"]["cz_cell"], "blank")
        self.assertEqual(p["totals"], {"cz_days": 28, "cz_vessels": 116, "ts_days": 1, "ts_vessels": 4})
        self.assertEqual(check(p, 28), "match")

    def test_vessels_with_entry_count_in_brackets(self):
        # 2016-08: "2【3】※" = 2 vessels entered the territorial sea 3 times in all
        ws = table()
        for w in ws:
            if w[1] == "4" and abs(w[4] - 109) < 1 and w[2] > 200:
                w[1] = "4【7】"
        ws.append(word("※", 262, 109))
        p = parse_words(ws, 2026, 10)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual(p["days"]["1"], {"cz": 8, "ts": 4, "cz_cell": "number", "ts_cell": "number", "ts_entries": 7})
        self.assertNotIn("ts_entries", p["days"]["2"])
        self.assertEqual(check(p, 28), "match")

    def test_stored_2016_08_matches_monthly_totals(self):
        import json
        store = json.loads((Path(__file__).resolve().parent.parent / "docs" / "data" / "jcg-senkaku" / "pdfs.json")
                           .read_text(encoding="utf-8"))
        rec = store["pdfs"]["https://www.kaiho.mlit.go.jp/mission/senkaku/data_h28_08.pdf"]
        p = parse_words(rec["words"], 2016, 8)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual(p["totals"], {"cz_days": 25, "cz_vessels": 147, "ts_days": 6, "ts_vessels": 23})
        self.assertEqual(check(p, 31), "match")
        self.assertEqual({d: (r["ts"], r.get("ts_entries")) for d, r in p["days"].items() if r.get("ts_entries")},
                         {"5": (2, 3), "7": (6, 11), "8": (3, 4), "9": (4, 10)})

    def test_blank_days_inside_the_month_and_mismatch(self):
        vals = {d: (4, None) for d in range(1, 31) if d not in (22, 23)}
        p = parse_words(table(title="令和8年9月", as_of="( 9/30 現在)", vals=vals, last=30), 2026, 9)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual(sorted(p["days"], key=int)[-1], "30")
        self.assertEqual(check(p, 30), "match")
        p["days"]["5"]["cz"] = 3
        self.assertEqual(check(p, 30), "mismatch")

    def test_without_headers_values_are_taken_left_to_right(self):
        p = parse_words(table(headers=False), 2026, 10)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual((p["days"]["1"]["cz"], p["days"]["1"]["ts"]), (8, 4))

    def test_without_totals(self):
        p = parse_words(table(totals=False), 2026, 10)
        self.assertEqual(check(p, 28), "no_totals")

    def test_dash_is_an_explicit_zero(self):
        ws = table(vals={d: (4, None) for d in range(1, 29)}, totals=False)
        ws.append(word("-", 240, 95 + 14 * 3))
        p = parse_words(ws, 2026, 10)
        self.assertEqual((p["days"]["3"]["ts"], p["days"]["3"]["ts_cell"]), (0, "dash"))

    def test_problems_block_parsing(self):
        self.assertFalse(parse_words([], 2026, 10)["parsed"])
        ws = table(vals={1: (2, 4), **{d: (4, None) for d in range(2, 29)}})
        p = parse_words(ws, 2026, 10)
        self.assertFalse(p["parsed"])
        self.assertIn("territorial sea 4 > contiguous zone 2", " ".join(p["problems"]))
        p = parse_words([word("1", 100, 100), word("2", 100, 114)], 2026, 10)
        self.assertIn("day column not found", p["problems"][0])

    def test_days_beyond_month_end_are_dropped(self):
        p = parse_words(table(title="令和8年2月", as_of="( 2/28 現在)", vals={d: (4, None) for d in range(1, 29)}),
                        2026, 2)
        self.assertTrue(p["parsed"], p["problems"])
        self.assertEqual(len(p["days"]), 28)

    def test_as_of_in_january_for_december(self):
        p = parse_words(table(title="令和7年12月", as_of="( 1/1 現在)"), 2025, 12)
        self.assertEqual(p["as_of"], "2026-01-01")


class RealLayouts(unittest.TestCase):
    """Word positions printed by the probe run on GitHub Actions (2026-10-30), one PDF per year."""

    def test_every_sample_parses_and_matches_its_totals(self):
        files = sorted(LAYOUTS.glob("*.txt"))
        self.assertGreaterEqual(len(files), 19)
        for f in files:
            with self.subTest(f.name):
                p = parse_words(from_layout(f.read_text(encoding="utf-8")))
                self.assertTrue(p["parsed"], p["problems"])
                y, m = (int(x) for x in p["title_year_month"].split("-"))
                last = calendar.monthrange(y, m)[1]
                if p["as_of"] and p["as_of"][:7] == p["title_year_month"]:
                    last = int(p["as_of"][8:])
                self.assertIn(check(p, last), ("match", "match_days"))

    def test_two_tables_side_by_side(self):
        p = parse_words(layout("R8_9"))
        self.assertEqual(p["title_year_month"], "2026-09")
        self.assertEqual(p["as_of"], "2026-09-30")
        self.assertEqual(p["days"]["1"]["cz"], 8)
        self.assertEqual((p["days"]["10"]["cz"], p["days"]["10"]["ts"]), (4, 4))
        self.assertEqual((p["days"]["20"]["cz"], p["days"]["20"]["ts"]), (2, None))
        self.assertEqual(p["totals"], {"cz_days": 30, "cz_vessels": 120, "ts_days": 1, "ts_vessels": 4})

    def test_first_month_starts_on_the_14th_and_has_no_as_of(self):
        p = parse_words(layout("h24_09"))
        self.assertEqual((p["title_year_month"], p["as_of"]), ("2012-09", None))
        self.assertEqual(min(map(int, p["days"])), 14)
        self.assertEqual(p["days"]["18"], {"cz": 12, "ts": 3, "cz_cell": "number", "ts_cell": "number"})

    def test_unreadable_vessel_total(self):
        p = parse_words(layout("R8_10"))
        self.assertEqual(p["totals"]["cz_vessels"], None)
        self.assertEqual(check(p, 29), "match_days")
        p["days"]["2"]["cz"] = None
        self.assertEqual(check(p, 29), "mismatch")


if __name__ == "__main__":
    unittest.main()
