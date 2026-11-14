import unittest
from datetime import date

from exercises import extract, merge, regions, sources
from exercises.text import html_to_text

DOC = {"title": "令和８年度機雷戦訓練（日向灘）及び掃海特別訓練（日米共同訓練）について", "summary": ""}
TEXT = ("２ 期 間 令和８年１１月１６日（月）～１１月２６日（木） ３ 訓練海空域 日向灘 "
        "４ 参加予定部隊等 （１）海上自衛隊 艦艇１６隻（掃海母艦×１隻、掃海艦×３隻） 航空機２機（ＭＣＨ－１０１×２機）")


def answer(**kw):
    base = {"relevant": True, "kind": "exercise", "name": None, "titleJa": "日向灘で機雷戦訓練", "countries": ["日本", "米国"],
            "region": "kyushu", "place": "日向灘", "forces": "海自 艦艇16隻", "ships": 16, "aircraft": 2, "personnel": None,
            "content": "機雷敷設・掃海", "start": "2026-11-16", "end": "2026-11-26",
            "dateEvidence": "令和８年１１月１６日（月）～１１月２６日（木）", "status": "planned"}
    base.update(kw)
    return base


class ValidateTest(unittest.TestCase):
    def test_keeps_supported_fields(self):
        v = extract.validate(answer(), DOC, TEXT, "2026-11-09")
        self.assertTrue(v["relevant"])
        self.assertEqual((v["start"], v["end"]), ("2026-11-16", "2026-11-26"))
        self.assertEqual((v["ships"], v["aircraft"]), (16, 2))

    def test_dates_without_evidence_are_dropped(self):
        v = extract.validate(answer(dateEvidence="November 16 to 26"), DOC, TEXT, "2026-11-09")
        self.assertIsNone(v["start"])
        self.assertIsNone(v["end"])

    def test_counts_not_in_text_are_dropped(self):
        v = extract.validate(answer(ships=20, personnel=1200), DOC, TEXT, "2026-11-09")
        self.assertIsNone(v["ships"])
        self.assertIsNone(v["personnel"])

    def test_name_must_appear(self):
        self.assertIsNone(extract.validate(answer(name="Keen Sword 27"), DOC, TEXT, "2026-11-09")["name"])

    def test_not_relevant(self):
        self.assertEqual(extract.validate(answer(relevant=False), DOC, TEXT, "2026-11-09"), {"relevant": False})
        self.assertEqual(extract.validate(answer(kind="port_visit"), DOC, TEXT, "2026-11-09"), {"relevant": False})

    def test_unknown_region(self):
        self.assertEqual(extract.validate(answer(region="mars"), DOC, TEXT, "2026-11-09")["region"], "other")


def ev(src, kind="報道", at=1_794_000_000_000, **kw):
    doc = {"url": "https://x/" + src, "title": src, "source": src, "sourceName": src, "kind": kind, "publishedAt": at}
    return merge.event_from(doc, extract.validate(answer(**kw), DOC, TEXT + " Keen Sword 27 Balikatan", "2026-11-09"))


class MergeTest(unittest.TestCase):
    def test_same_name_merges(self):
        out = merge.merge_all([ev("a", name="Keen Sword 27"), ev("b", name="Keen Sword 27", start=None)])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["sources"]), 2)

    def test_different_names_stay_apart(self):
        self.assertEqual(len(merge.merge_all([ev("a", name="Keen Sword 27"), ev("b", name="Balikatan")])), 2)

    def test_generic_name_is_not_a_name(self):
        e = merge.event_from({"url": "u", "kind": "公式", "publishedAt": 1}, {**answer(name="日米共同訓練")})
        self.assertIsNone(e["name"])

    def test_unnamed_same_area_overlapping(self):
        self.assertEqual(len(merge.merge_all([ev("a"), ev("b")])), 1)

    def test_unnamed_other_area(self):
        self.assertEqual(len(merge.merge_all([ev("a"), ev("b", region="nansei")])), 2)

    def test_official_fields_win_and_confidence(self):
        out = merge.merge_all([ev("news", forces="艦艇10隻"), ev("msdf", kind="公式", forces="海自 艦艇16隻")])
        self.assertEqual(out[0]["forces"], "海自 艦艇16隻")
        self.assertEqual(merge.confidence(out[0]), "高")

    def test_phase(self):
        e = ev("a")
        self.assertEqual(merge.phase(e, date(2026, 11, 14)), "next")
        self.assertEqual(merge.phase(e, date(2026, 11, 20)), "now")
        self.assertEqual(merge.phase(e, date(2026, 11, 30)), "done")
        undated = ev("b", start=None, end=None, status="ongoing")
        self.assertEqual(merge.phase(undated, date(2026, 11, 30)), "now")

    def test_finish_has_region_name_and_stable_id(self):
        a = merge.finish(ev("a", name="Keen Sword 27"), date(2026, 11, 14))
        b = merge.finish(ev("b", name="Keen Sword 27"), date(2026, 11, 14))
        self.assertEqual(a["regionName"], "九州周辺")
        self.assertEqual(a["id"], b["id"])


class SourcesTest(unittest.TestCase):
    def test_candidate_filter(self):
        self.assertTrue(sources.is_candidate("US, Japan kick off Keen Sword drills near Okinawa"))
        self.assertTrue(sources.is_candidate("東部战区 台岛周边 联合演训"))
        self.assertFalse(sources.is_candidate("Airmen complete training at Dyess AFB, Texas"))
        self.assertFalse(sources.is_candidate("China and Japan hold trade talks"))

    def test_feed_candidates(self):
        feed = {"sources": [{"id": "usni", "kind": "専門報道", "name": "USNI"}],
                "records": [{"id": "1", "url": "https://n/1", "title": "PLA drills around Taiwan", "source": "usni",
                             "publishedAt": 2},
                            {"id": "2", "url": "https://n/2", "title": "Budget hearing", "source": "usni", "publishedAt": 3}]}
        out = sources.feed_candidates(feed)
        self.assertEqual([d["url"] for d in out], ["https://n/1"])
        self.assertEqual(out[0]["kind"], "専門報道")

    def test_list_links(self):
        html = ('<li>2026/11/09 <a href="/msdf/release/202611/20261109.pdf">機雷戦訓練（日向灘）について</a></li>'
                '<li><a href="/msdf/about/">about</a></li>')
        links = sources.list_links(html, "https://www.mod.go.jp/msdf/release/", sources.OFFICIAL[0]["href"])
        self.assertEqual(links, [("https://www.mod.go.jp/msdf/release/202611/20261109.pdf",
                                  "機雷戦訓練（日向灘）について", "2026-11-09")])


class TextTest(unittest.TestCase):
    def test_html_to_text(self):
        html = ("<html><script>var x=1</script><nav>menu menu menu menu menu menu</nav>"
                "<p>The exercise began on Monday off Okinawa with 12 ships.</p><p>short</p></html>")
        self.assertEqual(html_to_text(html), "The exercise began on Monday off Okinawa with 12 ships.")


class RegionsTest(unittest.TestCase):
    def test_ids_unique_and_other_last(self):
        self.assertEqual(len(regions.IDS), len(set(regions.IDS)))
        self.assertEqual(regions.IDS[-1], "other")


if __name__ == "__main__":
    unittest.main()
