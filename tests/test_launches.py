import math
import unittest

from notam import launches as L


def dest(lat, lon, az, km):
    d, la, lo, a = km / L.EARTH_KM, math.radians(lat), math.radians(lon), math.radians(az)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(a))
    lo2 = lo + math.atan2(math.sin(a) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return round(math.degrees(la2), 4), round(math.degrees(lo2), 4)


def notice(key, point, start="2026-11-01T01:00:00Z", end="2026-11-01T03:00:00Z"):
    return {"key": key, "start_utc": start, "end_utc": end, "shapes": [{"kind": "circle", "points": [list(point)], "radius": 20000}]}


class Inference(unittest.TestCase):
    def test_aligned_areas_point_back_to_the_launch_site(self):
        lat, lon = L.SITES["酒泉"][:2]
        areas = [dest(lat, lon, 150, km) for km in (500, 1300, 2400)]
        ev = L.events([notice(f"a{i}", p) for i, p in enumerate(areas)])
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((e["method"], e["site"]), ("aligned", "酒泉"))
        self.assertAlmostEqual(e["azimuth_deg"], 150, delta=1.5)
        self.assertAlmostEqual(e["inclination_deg"], 67.8, delta=1.5)

    def test_due_east_from_wenchang_gives_low_inclination(self):
        self.assertAlmostEqual(L.inclination(L.SITES["文昌"][0], 90), 19.6, delta=0.2)

    def test_separate_times_are_separate_events_and_single_area_is_weak(self):
        lat, lon = L.SITES["西昌"][:2]
        ev = L.events([notice("x", dest(lat, lon, 120, 400)),
                       notice("y", dest(lat, lon, 120, 900), "2026-11-05T01:00:00Z", "2026-11-05T02:00:00Z")])
        self.assertEqual([e["method"] for e in ev], ["nearest-only", "nearest-only"])
        self.assertEqual(ev[0]["site"], "西昌")

    def test_scattered_areas_are_not_forced_onto_a_site(self):
        ev = L.events([notice("p", (30.0, 125.0)), notice("q", (20.0, 135.0)), notice("r", (35.0, 140.0))])
        self.assertEqual(ev[0]["method"], "unaligned")

    def test_duplicate_places_count_once(self):
        lat, lon = L.SITES["太原"][:2]
        p = dest(lat, lon, 170, 800)
        ev = L.events([notice("aviation:1", p), notice("maritime:1", (p[0] + 0.05, p[1]))])
        self.assertEqual((len(ev[0]["areas"]), len(ev[0]["notices"])), (1, 2))

    def test_recorded_launch_matches_event_by_time_and_site(self):
        lat, lon = L.SITES["酒泉"][:2]
        ev = L.events([notice(f"a{i}", dest(lat, lon, 150, km)) for i, km in enumerate((500, 1300))])
        ll2 = [{"ll2_id": "x", "name": "Long March 2D | Yaogan", "status": "Success", "net_utc": "2026-11-01T02:10:00Z",
                "lat": 40.96, "lon": 100.30, "rocket": "Long March 2D", "orbit": "SSO", "pad": "LC-43/94"},
               {"ll2_id": "y", "name": "elsewhere", "net_utc": "2026-11-01T02:10:00Z", "lat": 28.25, "lon": 102.03},
               {"ll2_id": "z", "name": "Falcon 9", "net_utc": "2026-11-01T02:10:00Z", "lat": 28.5, "lon": -80.6}]
        rec = L.match_launches(ev, ll2)
        self.assertEqual([(r["ll2_id"], r["event_index"]) for r in rec], [("x", 0), ("y", None)])  # z is outside East Asia
        self.assertEqual(ev[0]["recorded_launches"][0]["name"], "Long March 2D | Yaogan")

    def test_history_keeps_first_seen(self):
        h = L.merge({}, {"retrieved_utc": "2026-11-01T00:00:00Z", "notices": [{"key": "k"}]})
        h = L.merge(h, {"retrieved_utc": "2026-11-02T00:00:00Z", "notices": [{"key": "k"}]})
        self.assertEqual((h["notices"]["k"]["first_seen_utc"], h["notices"]["k"]["last_seen_utc"]),
                         ("2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z"))


if __name__ == "__main__":
    unittest.main()
