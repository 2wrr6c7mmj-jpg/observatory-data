import datetime as dt
import unittest

from jso import build

D = dt.date.fromisoformat
EVENTS = [(D("2024-05-23"), D("2024-05-24"), "CN", "exercise", "Joint Sword-2024A", "taiwan_strait"),
          (D("2010-01-01"), D("2010-01-01"), "CN", "exercise", "far away", "taiwan_strait")]
ANY = lambda a, cat, reg: True


def p(sid, date, hulls, frm, to, nation="CN"):
    return dict(sighting_id=sid, date=date, hulls=hulls, waters_from=frm, waters_to=to, nation=nation, strait="miyako",
                strait_jp="宮古海峡", units="u", has_aor=0, has_cv=0, release_id=f"r{sid}", url=f"u{sid}")


class Context(unittest.TestCase):
    def test_only_events_within_the_window(self):
        self.assertEqual(build.nearest(D("2024-05-20"), EVENTS, ANY), (3, "Joint Sword-2024A"))
        self.assertEqual(build.nearest(D("2024-05-24"), EVENTS, ANY), (0, "Joint Sword-2024A"))
        self.assertEqual(build.nearest(D("2024-05-30"), EVENTS, ANY), (-6, "Joint Sword-2024A"))
        self.assertEqual(build.nearest(D("2024-08-01"), EVENTS, ANY), (None, None))


class HullPairs(unittest.TestCase):
    def test_return_through_another_strait(self):
        ps = [p(1, "2025-03-01", "101 102 103", "東シナ海", "太平洋"),
              p(2, "2025-03-01", "101 102 103", "東シナ海", "太平洋"),   # same-day duplicate is skipped
              p(3, "2025-03-09", "101 102", "太平洋(西)", "東シナ海")]
        pairs = build.hull_pairs(ps, [])
        self.assertEqual([(x["out_sighting"], x["in_sighting"], x["days_away"], x["method"]) for x in pairs],
                         [(1, 3, 8, "hull-match")])

    def test_no_pair_when_next_sighting_goes_further_or_nation_differs(self):
        ps = [p(1, "2025-03-01", "101", "東シナ海", "太平洋"), p(2, "2025-03-05", "101", "太平洋", "東シナ海", nation="RU"),
              p(3, "2025-04-01", "201", "日本海", "太平洋"), p(4, "2025-04-03", "201", "日本海", "太平洋")]
        self.assertEqual(build.hull_pairs(ps, []), [])

    def test_stated_pairs_take_precedence(self):
        ps = [p(1, "2025-03-01", "101", "東シナ海", "太平洋"), p(3, "2025-03-09", "101", "太平洋", "東シナ海")]
        self.assertEqual(build.hull_pairs(ps, [{"out_sighting": 1, "in_sighting": 3}]), [])

    def test_pair_stated_on_a_repeat_blocks_the_kept_row(self):
        # 9 was dropped as a repeat of 1; a pair stated on 9 must keep 1 out of hull matching.
        ps = [p(1, "2025-03-01", "101", "東シナ海", "太平洋"), p(3, "2025-03-09", "101", "太平洋", "東シナ海")]
        stated = [{"out_sighting": 7, "in_sighting": 9}]
        self.assertEqual(len(build.hull_pairs(ps, stated)), 1)
        self.assertEqual(build.hull_pairs(ps, stated, {9: 1}), [])


def row(sid, rid, date, strait, inferred=0, hulls="101", to="太平洋", dir="SE", pub="2025-03-01", nation="CN"):
    return dict(sighting_id=sid, release_id=rid, pub_date=pub, date=date, strait=strait, inferred=inferred, hulls=hulls,
                waters_to=to, dir=dir, nation=nation)


class Dedupe(unittest.TestCase):
    def test_inferred_row_near_a_stated_passage_in_the_same_release_is_dropped(self):
        rows = [row(1, "r1", "2025-03-01", "yonaguni_taiwan", inferred=1), row(2, "r1", "2025-03-02", "yonaguni_iriomote"),
                row(3, "r2", "2025-06-01", "miyako", inferred=1, hulls="999")]
        kept, alias, n = build.dedupe(rows)
        self.assertEqual([r["sighting_id"] for r in kept], [2, 3])
        self.assertEqual(n["inferred_near_stated"], 1)

    def test_inferred_row_far_from_stated_passages_is_kept(self):
        rows = [row(1, "r1", "2025-03-01", "yonaguni_taiwan", inferred=1), row(2, "r1", "2025-03-04", "yonaguni_iriomote"),
                row(3, "r2", "2025-03-01", "miyako")]  # another release does not count for A1
        kept, alias, n = build.dedupe(rows)
        self.assertEqual(sorted(r["sighting_id"] for r in kept), [1, 2, 3])
        self.assertEqual(n["inferred_near_stated"], 0)

    def test_repeat_in_a_later_release_keeps_the_earliest(self):
        rows = [row(5, "r9", "2025-03-02", "miyako", dir="E", pub="2025-03-20"),   # return release re-reports it
                row(4, "r8", "2025-03-01", "miyako", dir="SE", pub="2025-03-02"),
                row(6, "r9", "2025-03-01", "miyako", to="東シナ海", pub="2025-03-20")]  # other direction: kept
        kept, alias, n = build.dedupe(rows)
        self.assertEqual(sorted(r["sighting_id"] for r in kept), [4, 6])
        self.assertEqual(alias, {5: 4})

    def test_repeat_beyond_two_days_or_through_another_strait_is_kept(self):
        rows = [row(4, "r8", "2025-03-01", "miyako", pub="2025-03-02"), row(5, "r9", "2025-03-04", "miyako", pub="2025-03-20"),
                row(6, "r9", "2025-03-01", "osumi", pub="2025-03-20")]
        kept, alias, n = build.dedupe(rows)
        self.assertEqual(sorted(r["sighting_id"] for r in kept), [4, 5, 6])
        self.assertEqual(n["repeated_in_later_release"], 0)

    def test_joint_release_repeat_is_removed_but_cn_and_ru_hulls_never_match(self):
        rows = [row(1, "r1", "2025-03-01", "tsushima", hulls="101 172", pub="2025-03-02", nation="CN"),
                row(2, "r2", "2025-03-01", "tsushima", hulls="101 172 548", pub="2025-03-10", nation="CNRU"),
                row(3, "r3", "2025-05-01", "tsushima", hulls="548", pub="2025-05-02", nation="RU"),
                row(4, "r4", "2025-05-02", "tsushima", hulls="548", pub="2025-05-03", nation="CN")]
        kept, alias, n = build.dedupe(rows)
        self.assertEqual(sorted(r["sighting_id"] for r in kept), [1, 3, 4])
        self.assertEqual(alias, {2: 1})


class Stay(unittest.TestCase):
    def test_return_without_a_strait_is_not_a_stay(self):
        self.assertTrue(build.is_stay({"direction": "進出→帰投", "in_strait": "miyako"}))
        self.assertFalse(build.is_stay({"direction": "進出→帰投", "in_strait": None}))  # e.g. flight operations next day
        self.assertFalse(build.is_stay({"direction": "帰投→再進出", "in_strait": "miyako"}))
        self.assertFalse(build.is_stay({"direction": "不明", "in_strait": "miyako"}))


class Direction(unittest.TestCase):
    def test_outbound_leg_heading_home_is_a_return_first(self):
        self.assertEqual(build.direction("CN", "太平洋(西)"), "進出→帰投")
        self.assertEqual(build.direction("CN", "東シナ海"), "帰投→再進出")
        self.assertEqual(build.direction("RU", "日本海"), "帰投→再進出")
        self.assertEqual(build.direction("RU", "オホーツク海"), "進出→帰投")

    def test_no_destination_or_same_basin_is_unknown(self):
        self.assertEqual(build.direction("CN", None), "不明")
        self.assertEqual(build.direction("CN", "東シナ海", "東シナ海"), "不明")
        self.assertEqual(build.direction("CN", "太平洋(東)", "太平洋(西)"), "不明")
        self.assertEqual(build.direction("CN", "太平洋", "東シナ海"), "進出→帰投")


class ReferencePoints(unittest.TestCase):
    def test_shiretoko_is_not_mapped_to_soya(self):
        import ast
        from pathlib import Path
        src = (Path(build.__file__).parent / "pipeline" / "jsoparse.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "NEAR_STRAIT")
        near = ast.literal_eval(node.value)
        self.assertNotIn("知床岬", near)
        self.assertEqual(near["宗谷岬"], "soya")


if __name__ == "__main__":
    unittest.main()
