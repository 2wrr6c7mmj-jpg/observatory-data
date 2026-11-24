import unittest
from datetime import date

from exercises import board as B

TODAY = date(2026, 11, 24)


def notice(key, issuer, cc, cat, start, end, pts):
    return {"key": key, "id": key.split(":")[1], "issuer": {"name": issuer, "country": cc}, "category": cat,
            "valid": {"start_utc": start, "end_utc": end, "status": "active"},
            "area": {"shapes": [{"kind": "polygon", "points": pts}]}, "original": {"text": key}, "source": {}}


DL = [[38.9, 121.3], [38.8, 121.5], [38.7, 121.3]]


class Board(unittest.TestCase):
    def test_own_notices_grouped_others_not_listed(self):
        ns = [notice("maritime:LN1", "大连海事局", "CN", "military", "2026-11-23T00:00:00Z", "2026-11-26T00:00:00Z", DL),
              notice("maritime:LN2", "大连海事局", "CN", "firing", "2026-11-25T00:00:00Z", "2026-11-27T00:00:00Z", DL),
              notice("maritime:TW1", "漁業署", "TW", "firing", "2026-11-24T00:00:00Z", "2026-11-25T00:00:00Z", [[23, 120], [23.1, 120.1], [23, 120.2]]),
              notice("maritime:LN9", "大连海事局", "CN", "firing", "2027-02-01T00:00:00Z", "2027-02-02T00:00:00Z", DL)]
        b = B.build({"events": []}, {"items": ns}, {}, TODAY)
        self.assertEqual([x["kind"] for x in b["items"]], ["notice"])
        g = b["items"][0]
        self.assertEqual(g["name"], "射撃（大连海事局）")
        self.assertEqual([n["id"] for n in g["notices"]], ["LN1", "LN2"])
        self.assertEqual((g["start"], g["end"]), ("2026-11-23", "2026-11-27"))

    def test_events_in_window_with_nearby_notices_and_no_accidents_or_far_areas(self):
        ev = lambda i, **k: {"id": i, "name": i, "countries": ["中国"], "region": "yellow-sea", "lat": 38.5, "lon": 121.0,
                             "start": "2026-11-25", "end": "2026-11-27", **k}
        cur = {"events": [ev("a"), ev("b", start="2027-04-21", end="2027-04-21"), ev("c", start=None, end=None),
                          ev("d", region="hawaii"), ev("e", content="戦車砲弾が破裂し隊員4人が死傷")]}
        ns = [notice("maritime:LN1", "大连海事局", "CN", "firing", "2026-11-26T00:00:00Z", "2026-11-26T08:00:00Z", DL)]
        b = B.build(cur, {"items": ns}, {}, TODAY)
        evs = [x for x in b["items"] if x["kind"] == "event"]
        self.assertEqual([x["event_id"] for x in evs], ["a"])
        self.assertEqual([n["id"] for n in evs[0]["notices"]], ["LN1"])

    def test_an_exercise_under_way_with_no_end_stays_listed(self):
        ev = lambda i, **k: {"id": i, "name": i, "countries": ["中国"], "region": "taiwan-east", "lat": 23.5, "lon": 123.4,
                             "start": "2026-11-14", "end": None, **k}
        b = B.build({"events": [ev("now", phase="now"), ev("old", phase="done")]}, {"items": []}, {}, TODAY)
        evs = [x for x in b["items"] if x["kind"] == "event"]
        self.assertEqual([x["event_id"] for x in evs], ["now"])
        self.assertTrue(evs[0]["ongoing"])
        self.assertIsNone(evs[0]["end"])

    def test_forecasts_listed_with_their_basis(self):
        fc = {"calls": [{"series": "keen-sword", "year": 2026, "name": "キーン・ソード（日米）", "countries": "日本・米国", "area": "日本周辺",
                         "window_start": "2026-11-30", "window_end": "2026-12-20", "basis": "隔年の機会6回のうち5回", "probability": 0.83}]}
        b = B.build({"events": []}, {"items": []}, fc, TODAY)
        self.assertEqual(b["items"][0]["kind"], "forecast")
        self.assertEqual(b["items"][0]["countries"], ["日本", "米国"])
        self.assertEqual(b["counts"], {"event": 0, "notice": 0, "forecast": 1})


if __name__ == "__main__":
    unittest.main()
