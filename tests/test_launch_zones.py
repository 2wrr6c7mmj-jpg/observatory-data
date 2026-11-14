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
