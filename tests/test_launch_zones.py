import unittest
from datetime import datetime, timezone

from notam import launch_zones as Z
from notam import launches as L

REF = datetime(2026, 11, 11, 7, 0, tzinfo=timezone.utc)


class Kinds(unittest.TestCase):
    def test_languages(self):
        self.assertEqual(Z.kind_of("鲁航警993/26，黄海，预计有火箭残骸落在…"), "fall")
        self.assertEqual(Z.kind_of("SD907/26 YELLOW SEA, ROCKET DEBRIS FALL IN AREA"), "fall")
        self.assertEqual(Z.kind_of("NORTH PACIFIC.\nROCKET LAUNCHING.\n"), "rocket")
        self.assertEqual(Z.kind_of("E)SPACE LAUNCH VEHICLE WILL BE LAUNCHED AS FLW"), "rocket")
        self.assertEqual(Z.kind_of("ANTIBALLISTIC MISSILE … DESTRUCTION OF AN OBJECT"), "intercept")

    def test_not_launches(self):
        self.assertIsNone(Z.kind_of("E)FRNG:\nWEAPON: MISSILE\n(HOKKAIDODAI MANEUVERING AREA SHIMAMATSU)"))
        self.assertIsNone(Z.kind_of("E)ALTN RTE ARE ESTABLISHED DUE TO AN AEROSPACE FLIGHT ACTIVITY."))


class Periods(unittest.TestCase):
    def test_navarea(self):
        p = Z.text_periods("ROCKET LAUNCHING.\n130500Z TO 130900Z NOV. AREAS", REF)
        self.assertEqual([(a.isoformat(), b.isoformat()) for a, b in p],
                         [("2026-11-13T05:00:00+00:00", "2026-11-13T09:00:00+00:00")])

    def test_msa_english(self):
        p = Z.text_periods("SD907/26 YELLOW SEA，121400 UTC TO 122100UTC NOV , ROCKET DEBRIS", REF)
        self.assertEqual(p[0][0].hour, 14)
        self.assertEqual(p[0][1].hour, 21)

    def test_msa_chinese_is_beijing_time(self):
        p = Z.text_periods("鲁航警993/26，黄海，青岛，11月12日2200时至13日0500时，预计有火箭残骸", REF)
        self.assertEqual((p[0][0].day, p[0][0].hour, p[0][1].day, p[0][1].hour), (12, 14, 12, 21))

    def test_reserve_days_run_until_the_notice_cancels_itself(self):
        n = Z.normalise({"key": "maritime:NAVAREA XI 26-0486", "id": "NAVAREA XI 26-0486", "source": "maritime",
                         "first_seen_utc": "2026-10-30T00:00:00Z", "start_utc": None, "end_utc": None,
                         "text": "ROCKET LAUNCHING.\n040500Z TO 040900Z NOV,\nALTERNATE 0500Z TO 0900Z DAILY 02, 03\n"
                                 "AND 05 TO 30 NOV.\nCANCEL THIS MSG 301000Z NOV."})
        self.assertEqual((n["start_utc"], n["end_utc"], n["alternate"]), ("2026-11-04T05:00:00Z", "2026-11-30T10:00:00Z", True))

    def test_navarea_cancellation(self):
        gone = Z.cancelled_keys([{"key": "maritime:NAVAREA XI 26-0511", "id": "NAVAREA XI 26-0511",
                                  "text": "ROCKET LAUNCHING SUSPENDED.\nCANCEL 0508/26 AND THIS MSG."}])
        self.assertEqual(gone, {"maritime:NAVAREA XI 26-0508", "maritime:NAVAREA XI 26-0511"})


class Sites(unittest.TestCase):
    def test_stated_site(self):
        self.assertEqual(Z.stated_site("- LAUNCH SITE : 330842N1262718E"), [33.145, 126.455])

    def test_stated_site_names_the_known_site_and_gives_the_azimuth(self):
        out = L.stated_site_inference([33.145, 126.455], [(30.3, 128.5), (26.5, 132.0)])
        self.assertEqual((out["method"], out["site"]), ("stated", "済州沖（海上発射）"))
        self.assertAlmostEqual(out["azimuth_deg"], 145, delta=5)

    def test_ground_firing_is_not_a_launch_event(self):
        n = {"key": "aviation:P5732/26", "start_utc": "2026-11-30T22:00:00Z", "end_utc": "2026-12-02T08:00:00Z",
             "shapes": [{"kind": "polygon", "points": [[42.9, 141.5], [42.8, 141.5], [42.8, 141.4]]}], "kind": None}
        self.assertEqual(L.events([n]), [])

    def test_vehicle_names(self):
        self.assertEqual(Z.rocket_ja("Long March 8A"), "長征8A号")
        self.assertEqual(Z.rocket_ja("Long March 2F/G"), "長征2F/G号")


if __name__ == "__main__":
    unittest.main()


class Attempts(unittest.TestCase):
    N = {k: {"kind": v} for k, v in {"a": "rocket", "b": "rocket", "c": "fall", "i1": "intercept", "i2": "intercept",
                                       "r1": "related", "r2": "related"}.items()}

    def ev(self, s, e, notices, site=None, mission=None):
        return {"window_start_utc": s, "window_end_utc": e, "notices": notices, "areas": [], "site": site,
                "recorded_launches": [{"name": mission}] if mission else []}

    def test_attempts_of_one_launch_are_one_entry(self):
        out = Z.merge_attempts([self.ev("2026-11-10T00:00Z", "2026-11-30T09:34Z", ["a"], "済州沖（海上発射）", "Mir | Demo"),
                                self.ev("2026-11-11T05:00Z", "2026-11-30T09:34Z", ["b"], "済州沖（海上発射）"),
                                self.ev("2026-11-13T05:00Z", "2026-11-30T09:34Z", ["c"], "済州沖（海上発射）")], self.N)
        self.assertEqual([e["notices"] for e in out], [["a", "b", "c"]])
        self.assertEqual(out[0]["recorded_launches"][0]["name"], "Mir | Demo")

    def test_two_launches_from_one_site_stay_apart(self):
        out = Z.merge_attempts([self.ev("2026-11-20T11:00Z", "2026-11-23T14:00Z", ["a"], "文昌", "Long March 8A"),
                                self.ev("2026-11-22T07:00Z", "2026-11-27T10:00Z", ["c"], "文昌", "Long March 12")], self.N)
        self.assertEqual(len(out), 2)

    def test_notices_with_no_site(self):
        out = Z.merge_attempts([self.ev("2026-08-17T07:09Z", "2026-11-16T15:00Z", ["i1"]),
                                self.ev("2026-09-01T00:00Z", "2026-11-16T15:00Z", ["i2"]),
                                self.ev("2026-11-14T01:30Z", "2026-11-20T05:01Z", ["r1"]),
                                self.ev("2026-11-15T01:30Z", "2026-11-20T05:01Z", ["r2"])], self.N)
        self.assertEqual([e["notices"] for e in out], [["i1", "i2"], ["r1"], ["r2"]])


# ---------- launches first (2026-11-23) ----------
NOW = datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc)


def closure(nid, start, end, pts):
    return {"key": "aviation:" + nid, "id": nid, "source": "aviation", "first_seen_utc": "2026-11-21T00:00:00Z",
            "start_utc": start, "end_utc": end, "evidence": ["cn-closure"], "shapes": [{"kind": "polygon", "points": pts}],
            "text": f"{nid} NOTAMN Q)ZXXX/QRTCA/IV/BO/W/000/999/3812N11017E516 E) A TEMPORARY RESTRICTED AREA ESTABLISHED"}


JIUQUAN_CORRIDOR = [[41.398, 100.42], [39.843, 106.163], [37.204, 115.908], [35.912, 120.678], [35.013, 120.163],
                    [36.318, 115.424], [38.917, 105.994], [40.518, 100.159]]
YELLOW_SEA = [[36.211, 121.187], [35.09, 124.0], [32.918, 124.0], [34.499, 120.216]]
LL2 = [
    {"ll2_id": "sz24", "name": "Long March 2F/G | Shenzhou 24", "status": "Go", "net_utc": "2026-11-25T00:18:00Z",
     "rocket": "Long March 2F/G", "lat": 40.958, "lon": 100.291, "location": "Jiuquan", "pad": "SLS-1"},
    {"ll2_id": "tai", "name": "Long March 6A | X", "status": "Go", "net_utc": "2026-11-27T13:00:00Z",
     "rocket": "Long March 6A", "lat": 38.85, "lon": 111.61, "location": "Taiyuan"},
    {"ll2_id": "sea", "name": "Kinetica 1 | X", "status": "TBC", "net_utc": "2026-11-29T01:00:00Z",
     "rocket": "Kinetica 1", "lat": 19.61, "lon": 110.95, "location": "Wenchang"},
    {"ll2_id": "done", "name": "Long March 8A | X", "status": "Success", "net_utc": "2026-11-22T00:00:00Z",
     "rocket": "Long March 8A", "lat": 19.61, "lon": 110.95},
    {"ll2_id": "far", "name": "H3 | X", "status": "TBD", "net_utc": "2026-12-31T00:00:00Z", "rocket": "H3", "lat": 30.4, "lon": 130.97},
]


def launches_of(raw):
    notices = {n["key"]: Z.normalise(n) for n in raw}
    ev = L.events(list(notices.values()))
    return Z.build_launches(LL2, ev, notices, L.SITES, NOW)


class LaunchesFirst(unittest.TestCase):
    def test_china_closures_are_matched_to_the_launch_from_the_site_they_reach(self):
        out = launches_of([closure("A6094/26", "2026-11-25T00:08:00Z", "2026-11-25T00:48:00Z", JIUQUAN_CORRIDOR),
                           closure("A6095/26", "2026-11-25T00:08:00Z", "2026-11-25T00:50:00Z", YELLOW_SEA)])
        sz = next(x for x in out if x["id"] == "ll2:sz24")
        self.assertEqual(sz["site"]["name"], "酒泉")
        self.assertEqual([n["id"] for n in sz["notices"]], ["A6094/26", "A6095/26"])
        self.assertEqual({n["kind"] for n in sz["notices"]}, {"closure"})
        self.assertEqual(sz["estimated_window"], {"start_utc": "2026-11-25T00:08:00Z", "end_utc": "2026-11-25T00:48:00Z"})

    def test_inland_without_notice_is_left_out_sea_without_notice_waits_done_and_far_are_out(self):
        out = launches_of([])
        self.assertEqual([x["id"] for x in out], ["ll2:sea"])
        self.assertEqual(out[0]["notice_state"], "waiting")
        self.assertIsNone(out[0]["estimated_window"])

    def test_reserve_days_and_long_notices(self):
        n = {"start_utc": "2026-11-24T07:00:00Z", "end_utc": "2026-11-27T10:00:00Z",
             "periods": [{"start_utc": f"2026-11-{d}T07:00:00Z", "end_utc": f"2026-11-{d}T10:00:00Z"} for d in (24, 25, 26, 27)]}
        slot, reserve = Z.slot_for(n, datetime(2026, 11, 26, 9, 35, tzinfo=timezone.utc))
        self.assertEqual((slot[0].day, reserve), (26, 3))
        self.assertIsNone(Z.estimated_window([(datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 30, tzinfo=timezone.utc))]))

    def test_an_expired_notice_is_not_listed(self):
        out = launches_of([closure("A6107/26", "2026-11-27T12:53:00Z", "2026-11-27T13:15:00Z", [[38.9, 111.7], [38.0, 113.0], [37.8, 112.8]]),
                           closure("OLD", "2026-11-20T12:53:00Z", "2026-11-20T13:15:00Z", [[38.9, 111.7], [38.0, 113.0], [37.8, 112.8]])])
        tai = next(x for x in out if x["id"] == "ll2:tai")
        self.assertEqual([n["id"] for n in tai["notices"]], ["A6107/26"])
