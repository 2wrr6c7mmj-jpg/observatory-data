"""ocean/ais_eval.py: detection rate against AIS south of Hong Kong (standard library only; no network, no key)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ocean import ais_eval as ae

ROOT = Path(__file__).resolve().parents[1]
BBOX = [113.7, 21.5, 114.6, 22.3]
LON0, LAT0 = 114.15, 21.9
T0 = datetime(2026, 11, 4, 10, 32, 10, tzinfo=timezone.utc)   # a pass
KEY = "test-key-not-a-secret"


def product(start: datetime, rel=11, plat="D", direction="ASCENDING", lat=(21.0, 22.8), lon=(113.0, 115.5), secs=25):
    end = start + timedelta(seconds=secs)
    ring = [[lon[0], lat[0]], [lon[1], lat[0]], [lon[1], lat[1]], [lon[0], lat[1]], [lon[0], lat[0]]]
    name = (f"S1{plat}_IW_GRDH_1SDV_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}_000123_00ABCD_1234.SAFE")
    return {"Id": name, "Name": name,
            "ContentDate": {"Start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "End": end.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
            "PublicationDate": (end + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "GeoFootprint": {"type": "Polygon", "coordinates": [ring]},
            "Attributes": [{"Name": "relativeOrbitNumber", "Value": str(rel)},
                           {"Name": "orbitDirection", "Value": direction}]}


def ship(sid, lon, lat, sog, cog, length, t_list=(0, 20, 40, 60, 80)):
    return {"id": sid, "len": length, "wid": 20, "t": list(t_list), "lon": [lon] * len(t_list), "lat": [lat] * len(t_list),
            "sog": [sog] * len(t_list), "cog": [cog] * len(t_list)}


class SealTest(unittest.TestCase):
    def test_roundtrip_wrong_key_and_tamper(self):
        data = json.dumps({"a": list(range(500))}).encode()
        blob = ae.seal(data, KEY)
        self.assertNotIn(b'"a"', blob)
        self.assertEqual(ae.unseal(blob, KEY), data)
        with self.assertRaises(ValueError):
            ae.unseal(blob, KEY + "x")
        bad = bytearray(blob)
        bad[60] ^= 1
        with self.assertRaises(ValueError):
            ae.unseal(bytes(bad), KEY)
        with self.assertRaises(ValueError):
            ae.unseal(b"garbage", KEY)
        self.assertNotEqual(ae.seal(data, KEY), ae.seal(data, KEY))    # random salt and nonce
        with self.assertRaises(ValueError):
            ae.seal(data, "")


class GeometryTest(unittest.TestCase):
    def test_heading(self):
        self.assertAlmostEqual(ae.heading_deg("ascending", 22.0), 348.0, delta=2.0)
        self.assertAlmostEqual(ae.heading_deg("DESCENDING", 22.0), 192.0, delta=2.0)
        with self.assertRaises(ValueError):
            ae.heading_deg("x", 22.0)

    def test_heading_same_as_detect2(self):
        try:
            from sat import detect2
        except Exception:   # numpy not installed
            self.skipTest("sat.detect2 needs numpy")
        for d in ("ascending", "descending"):
            self.assertAlmostEqual(ae.heading_deg(d, 21.9), detect2.platform_heading(d, 21.9), places=6)

    def test_azimuth_shift_sign_and_size(self):
        head = ae.heading_deg("ascending", LAT0)
        # still ship or no course: no shift
        self.assertEqual(ae.azimuth_shift(LON0, LAT0, 0.2, 90, head)[2], 0.0)
        # moving along the heading (no line-of-sight component): no shift
        self.assertAlmostEqual(ae.azimuth_shift(LON0, LAT0, 10, head, head)[2], 0.0, delta=1e-6)
        # moving away from the radar (right-looking: heading + 90): shift = -(R/V) v_away, behind the platform
        sog = 10 / ae.KN
        lon, lat, sh = ae.azimuth_shift(LON0, LAT0, sog, (head + 90) % 360, head, rv_s=120)
        self.assertAlmostEqual(sh, -1200.0, delta=1.0)
        # the apparent position lies 1200 m from the true one, backwards along the heading
        self.assertAlmostEqual(ae.dist_m(LON0, LAT0, lon, lat), 1200.0, delta=5.0)
        ux, uy = math.sin(math.radians(head)), math.cos(math.radians(head))
        x, y = ae.xy(lon, lat, LON0, LAT0)
        self.assertLess(x * ux + y * uy, 0)
        # towards the radar: the opposite side
        _, _, sh2 = ae.azimuth_shift(LON0, LAT0, sog, (head - 90) % 360, head, rv_s=120)
        self.assertAlmostEqual(sh2, 1200.0, delta=1.0)

    def test_dead_reckon(self):
        lon, lat = ae.dead_reckon(LON0, LAT0, 10.0, 90.0, 360)      # 10 kn east for 6 min = 1852 m
        self.assertAlmostEqual(ae.dist_m(LON0, LAT0, lon, lat), 1852.0 * 0.514444 * 10 / 10 / 0.514444 * 0.99974, delta=5)
        self.assertGreater(lon, LON0)
        self.assertAlmostEqual(lat, LAT0, places=6)


class CatalogueTest(unittest.TestCase):
    def passes(self):
        # S1D/11 ascending seen on 2026-10-23 and 2026-10-11 (12 days apart); an unrelated orbit with a small share
        prods = [product(T0 - timedelta(days=12) - timedelta(seconds=25)), product(T0 - timedelta(days=24) - timedelta(seconds=25)),
                 product(T0 - timedelta(days=5), rel=105, plat="C", lat=(18.0, 19.0), lon=(112.0, 113.0))]
        return ae.past_passes(prods, BBOX, 0.3)

    def test_time_at_is_linear_and_northwards_later_for_ascending(self):
        sl = ae.slices_of([product(T0)])
        head = ae.heading_deg("ascending", LAT0)
        a = ae.time_at(sl, LON0, 21.6, head, LON0, LAT0)
        b = ae.time_at(sl, LON0, 22.2, head, LON0, LAT0)
        self.assertLess(a, b)
        self.assertTrue(T0 <= a <= b <= T0 + timedelta(seconds=25))
        self.assertAlmostEqual((b - a).total_seconds(), 25 * 66000 / (1.8 * 110574) * 1, delta=3.0)

    def test_past_passes_group_and_filter(self):
        ps = self.passes()
        self.assertEqual({(p["platform"], p["relative_orbit"]) for p in ps}, {("S1D", 11)})
        self.assertEqual(len(ps), 2)
        self.assertGreaterEqual(ps[0]["share"], 0.99)
        # two slices of one pass are one pass
        two = ae.past_passes([product(T0, lat=(21.0, 22.0)), product(T0 + timedelta(seconds=25), lat=(22.0, 22.8))], BBOX, 0.3)
        self.assertEqual(len(two), 1)

    def test_due_passes_predicted_from_the_latest_pass(self):
        ps = self.passes()
        pred = ps[-1]["t_center"] + timedelta(days=12)
        # 30 minutes before the window starts: due (window starts within 65 min)
        d = ae.due_passes(ps, pred - timedelta(minutes=20 + 30))
        self.assertEqual(len(d), 1)
        self.assertEqual((d[0]["platform"], d[0]["relative_orbit"], d[0]["predicted"]), ("S1D", 11, True))
        self.assertEqual(ae.parse_time(d[0]["start_utc"]), ae.parse_time(d[0]["t_center_utc"]) - timedelta(minutes=20))
        self.assertTrue(d[0]["name"].startswith("ais-S1D-11-"))
        # 2 hours before: not yet; 17 minutes after the centre (3 min left): too late; a day later: no
        self.assertEqual(ae.due_passes(ps, pred - timedelta(hours=2)), [])
        self.assertEqual(ae.due_passes(ps, pred + timedelta(minutes=17)), [])
        self.assertEqual(ae.due_passes(ps, pred + timedelta(days=1)), [])
        # inside the window: recorded with the time that is left
        self.assertEqual(len(ae.due_passes(ps, pred + timedelta(minutes=5))), 1)
        # twelve days later again (the following repeat)
        self.assertEqual(len(ae.due_passes(ps, pred + timedelta(days=12) - timedelta(minutes=30))), 1)

    def test_predict_with_fake_odata(self):
        class A:
            horizon_min = 65.0
        prods = [product(T0 - timedelta(days=12) - timedelta(seconds=25))]
        now = T0 - timedelta(days=12) + timedelta(days=12) - timedelta(minutes=40)
        out = ae.cmd_predict(A(), now=now, get=lambda url: {"value": prods}, log=lambda *_: None)
        self.assertEqual(len(out), 1)


class CollectorTest(unittest.TestCase):
    def msgs(self):
        pos = lambda mmsi, lat, lon, sog, cog: {"MessageType": "PositionReport", "MetaData": {"MMSI": mmsi, "ShipName": "NAME", "latitude": lat, "longitude": lon},
                                                "Message": {"PositionReport": {"Latitude": lat, "Longitude": lon, "Sog": sog, "Cog": cog}}}
        return [pos(412345678, 21.9, 114.1, 12.3, 45.0),
                pos(412345678, 21.9001, 114.1001, 12.3, 45.0),           # < 15 s later: thinned
                pos(412345678, 21.91, 114.11, 102.3, 360.0),             # not available
                pos(999999999, 30.0, 114.1, 5.0, 10.0),                  # outside the box
                {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 412345678, "ShipName": "SECRET NAME"},
                 "Message": {"ShipStaticData": {"Dimension": {"A": 200, "B": 30, "C": 15, "D": 15}, "CallSign": "XYZ", "Name": "SECRET NAME"}}},
                pos(477000001, 21.95, 114.2, 3.0, 90.0),
                {"MessageType": "StaticDataReport", "MetaData": {"MMSI": 477000001},
                 "Message": {"StaticDataReport": {"ReportB": {"Valid": True, "Dimension": {"A": 10, "B": 5, "C": 2, "D": 2}}}}},
                {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 477000002}, "Message": {"ShipStaticData": {"Dimension": {"A": 0, "B": 0, "C": 0, "D": 0}}}}]

    def test_anonymous_thinned_tracks(self):
        col = ae.Collector(BBOX, 1000.0)
        for k, m in enumerate(self.msgs()):
            col.feed(m, 1000.0 + [1, 5, 30, 40, 50, 55, 60, 70][k])
        res = col.result()
        self.assertEqual(len(res["ships"]), 2)       # the ship outside the box and the static-only (no length) are not kept
        s = res["ships"][0]
        self.assertEqual((s["len"], s["wid"], s["t"]), (230, 30, [1.0, 30.0]))
        self.assertEqual((s["sog"], s["cog"]), ([12.3, None], [45.0, None]))
        text = json.dumps(res)
        for secret in ("412345678", "SECRET", "XYZ", "477000001", "NAME"):
            self.assertNotIn(secret, text)
        self.assertEqual(res["ships"][1]["len"], 15)

    def test_subscription(self):
        sub = ae.subscription("K", BBOX)
        self.assertEqual(sub["BoundingBoxes"], [[[21.5, 113.7], [22.3, 114.6]]])     # [lat, lon] corners
        self.assertIn("ShipStaticData", sub["FilterMessageTypes"])


def cands():
    return [{"lon": 114.10, "lat": 21.80, "v2_match": True, "length": 210.0, "v2_length": 190.0},
            {"lon": 114.30, "lat": 21.70, "v2_match": False, "length": 90.0, "v2_length": None},
            {"lon": 114.45, "lat": 22.20, "v2_match": False, "length": 120.0, "v2_length": None}]   # far from every AIS ship


class EvaluateTest(unittest.TestCase):
    head = ae.heading_deg("ascending", LAT0)
    rects = [(113.7, 21.5, 114.6, 22.3)]

    @staticmethod
    def t_img(lon, lat):
        return 100.0

    def run_eval(self, ships, cs=None, obst=None):
        return ae.evaluate_pass({"ships": ships, "messages": 99}, cs if cs is not None else cands(), self.rects, obst, self.head,
                                self.t_img)

    def test_hits_by_band_detector_and_radius(self):
        ships = [ship(0, 114.10, 21.80, 0.0, None, 230),       # still, >= 100 m: hit by v1 and v2
                 ship(1, 114.30, 21.7015, 0.0, None, 70),      # still, 50-100 m: 167 m from a v1-only candidate
                 ship(2, 114.0, 22.0, 0.0, None, 250),         # no candidate: a miss
                 ship(3, 114.2, 21.6, 0.0, None, None),        # no length: not in n
                 ship(4, 114.5, 21.9, 0.0, None, 30)]          # < 50 m: not in the rates
        res, flags = self.run_eval(ships)
        d = res["detection"]
        self.assertEqual(d["v1"]["200"]["ge100"], {"n": 2, "hit": 1})
        self.assertEqual(d["v2"]["200"]["ge100"], {"n": 2, "hit": 1})
        self.assertEqual(d["v1"]["200"]["50_100"], {"n": 1, "hit": 1})
        self.assertEqual(d["v2"]["200"]["50_100"], {"n": 1, "hit": 0})      # the candidate has no v2 match
        self.assertEqual(res["ais"]["excluded"]["no_length"], 1)
        self.assertEqual(res["ais"]["ships_reference"], 4)
        # length differences: v1 +(-20) for the 230 m ship (210 - 230), v2 190 - 230 = -40
        self.assertEqual(res["length_diff_m"]["v1"]["ge100"], {"-20": 1})
        self.assertEqual(res["length_diff_m"]["v2"]["ge100"], {"-40": 1})
        # candidate flags: matched, matched, unmatched (an AIS ship within 10 km? no: 114.45,22.20 is > 10 km from all)
        self.assertEqual([f["ais_200"] for f in flags], ["matched", "matched", "no_ais_coverage"])
        self.assertEqual(flags[0]["ais_length_band"], "ge100")
        self.assertEqual(flags[1]["ais_length_band"], "50_100")
        self.assertEqual(res["candidates"]["by_ais"]["200"], {"matched": 2, "unmatched": 0, "no_ais_coverage": 1})

    def test_radius_300_and_unmatched_inside_coverage(self):
        # 250 m east of the candidate: a hit at 300 m only; the third candidate has an AIS ship 4 km away
        ships = [ship(0, 114.10 + 250 / ae.m_per_deg(21.8)[0], 21.80, 0.0, None, 150),
                 ship(1, 114.45, 22.16, 0.0, None, 80)]
        res, flags = self.run_eval(ships)
        self.assertEqual(res["detection"]["v1"]["200"]["ge100"], {"n": 1, "hit": 0})
        self.assertEqual(res["detection"]["v1"]["300"]["ge100"], {"n": 1, "hit": 1})
        self.assertEqual((flags[0]["ais_200"], flags[0]["ais_300"]), ("unmatched", "matched"))
        self.assertEqual(flags[2]["ais_300"], "unmatched")     # AIS is received nearby, but no ship there
        self.assertIsNone(flags[2]["ais_length_band"])

    def test_moving_ship_is_matched_at_its_azimuth_shifted_position(self):
        # a 200 m ship moving east at 10 kn: v_away > 0 (the radar looks east for an ascending pass)
        sog = 10.0
        s = ship(0, 114.10, 21.80, sog, 90.0, 200, t_list=(100,))
        lon, lat, shift = ae.azimuth_shift(114.10, 21.80, sog, 90.0, self.head)
        self.assertGreater(abs(shift), 400)
        c = [{"lon": lon, "lat": lat, "v2_match": False, "length": 180.0, "v2_length": None}]
        res, _ = self.run_eval([s], c)
        self.assertEqual(res["detection"]["v1"]["200"]["ge100"], {"n": 1, "hit": 1})
        # a candidate at the true position is a miss once the shift is applied
        c2 = [{"lon": 114.10, "lat": 21.80, "v2_match": False, "length": 180.0, "v2_length": None}]
        res2, _ = self.run_eval([s], c2)
        self.assertEqual(res2["detection"]["v1"]["300"]["ge100"], {"n": 1, "hit": 0})

    def test_dead_reckoning_to_the_imaging_time(self):
        # last report 60 s before the imaging time, 12 kn east: ~370 m further east
        sog = 12.0
        s = ship(0, 114.10, 21.80, sog, 90.0, 200, t_list=(40,))
        head_perp = ae.heading_deg("ascending", LAT0)
        lon, lat = ae.dead_reckon(114.10, 21.80, sog, 90.0, 60)
        alon, alat, _ = ae.azimuth_shift(lon, lat, sog, 90.0, head_perp)
        c = [{"lon": alon, "lat": alat, "v2_match": False, "length": 190.0, "v2_length": None}]
        res, _ = self.run_eval([s], c)
        self.assertEqual(res["detection"]["v1"]["200"]["ge100"], {"n": 1, "hit": 1})

    def test_exclusions(self):
        ships = [ship(0, 114.10, 21.80, 0.0, None, 200, t_list=(0, 1000)),      # nearest report 1000 - 100 s: kept?
                 ship(1, 114.12, 21.82, 0.0, None, 200, t_list=(2000,)),         # report 1900 s away: stale
                 ship(2, 114.14, 21.84, 8.0, None, 200),                          # moving, no course
                 ship(3, 113.60, 21.84, 0.0, None, 200),                          # outside the seen cells
                 ship(4, 114.176, 21.86, 0.0, None, 200)]                         # next to the coast
        osm = {"elements": [{"type": "way", "tags": {"natural": "coastline"},
                             "geometry": [{"lon": 114.17, "lat": 21.80}, {"lon": 114.17, "lat": 21.95}]}]}
        res, _ = self.run_eval(ships, cands(), ae.Obstacles(osm, LON0, LAT0))
        ex = res["ais"]["excluded"]
        self.assertEqual((ex["stale"], ex["motion_unknown"], ex["outside_seen"], ex["near_coast"]), (1, 1, 1, 1))
        self.assertEqual(res["ais"]["ships_reference"], 1)

    def test_no_ship_position_in_the_result(self):
        ships = [ship(0, 114.123456, 21.812345, 3.0, 20.0, 200)]
        res, flags = self.run_eval(ships)
        text = json.dumps(res)
        for v in ("114.123", "21.812", "mmsi", "MMSI"):
            self.assertNotIn(v, text)
        self.assertEqual(set(flags[0]), {"lon", "lat", "ais_200", "ais_300", "ais_length_band"})


class StatsTest(unittest.TestCase):
    def test_wilson_and_hist(self):
        self.assertIsNone(ae.wilson(0, 0))
        lo, hi = ae.wilson(8, 10)
        self.assertTrue(0.4 < lo < 0.8 < hi < 0.98)
        h = {}
        for v in (-95, -5, 0, 4, 12, 150, -300):
            ae.hist_add(h, v)
        self.assertEqual(sum(h.values()), 7)
        self.assertEqual(h["100"], 1)             # open end
        self.assertEqual(h["-110"], 1)
        self.assertIsNone(ae.hist_stats({"0": 4}))     # n < 5: no quantiles
        st = ae.hist_stats({"-10": 3, "0": 5, "10": 2})
        self.assertEqual((st["n"], st["median_m"]), (10, 5.0))

    def test_summarize(self):
        p = {"status": "evaluated", "candidates": {"n": 3, "by_ais": {"200": {"matched": 1, "unmatched": 1, "no_ais_coverage": 1},
                                                                       "300": {"matched": 2, "unmatched": 0, "no_ais_coverage": 1}}},
             "detection": {"v1": {"200": {"ge100": {"n": 10, "hit": 8}, "50_100": {"n": 4, "hit": 1}}}},
             "length_diff_m": {"v1": {"ge100": {"0": 6}}}}
        s = ae.summarize([p, {**p}, {"status": "not_evaluated"}])
        g = s["detection"]["v1"]["200"]["ge100"]
        self.assertEqual((g["n"], g["hit"], g["rate"]), (20, 16, 0.8))
        self.assertEqual(s["passes_evaluated"], 2)
        self.assertEqual(s["passes_not_evaluated"], 1)
        self.assertEqual(s["length_diff_m"]["v1"]["ge100"]["n"], 12)
        self.assertEqual(s["candidates"]["by_ais"]["300"]["matched"], 4)


class FileTest(unittest.TestCase):
    """evaluate_file on a sealed file, a fake monitor output and a fake OData."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mon, self.out, self.ais = self.root / "monitor", self.root / "out", self.root / "ais"
        for d in (self.mon / "candidates", self.mon / "coverage", self.ais):
            d.mkdir(parents=True)
        self.acq = T0 - timedelta(seconds=10)
        self.prod = product(self.acq)
        self.acq_s = self.acq.strftime("%Y-%m-%dT%H:%M:%SZ")
        row = {"area": "HKS", "acquired_utc": self.acq_s, "platform": "sentinel-1d", "relative_orbit": 11,
               "orbit_direction": "ascending", "status": "ok"}
        (self.mon / "passes.json").write_text(json.dumps({"passes": [row]}))
        day = self.acq_s[:10]
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [114.10, 21.80]},
                  "properties": {"area": "HKS", "acquired_utc": self.acq_s, "v2_match": True, "length_m_est_40m": 210.0,
                                 "v2_length_m": 190.0}},
                 {"type": "Feature", "geometry": {"type": "Point", "coordinates": [114.50, 22.20]},
                  "properties": {"area": "HKS", "acquired_utc": self.acq_s, "v2_match": False, "length_m_est_40m": 100.0}}]
        (self.mon / "candidates" / f"{day}.geojson").write_text(json.dumps({"features": feats}))
        ring = [[113.7, 21.5], [114.6, 21.5], [114.6, 22.3], [113.7, 22.3], [113.7, 21.5]]
        (self.mon / "coverage" / f"{day}.geojson").write_text(json.dumps({"features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
             "properties": {"area": "HKS", "acquired_utc": self.acq_s, "state": "seen"}}]}))
        self.get = lambda url: {"value": [self.prod]}

    def tearDown(self):
        self.tmp.cleanup()

    def sealed(self, name="ais-S1D-11-20261104T1032", t0=None, ships=None, auth="ok", key=KEY):
        t0 = t0 or (T0 - timedelta(minutes=20))
        payload = {"version": "ais-raw-1",
                   "pass": {"name": name, "platform": "S1D", "relative_orbit": 11, "direction": "ascending",
                            "t_center_utc": ae.iso(T0)},
                   "t0_utc": ae.iso(t0), "t1_utc": ae.iso(t0 + timedelta(minutes=40)), "reception": {"auth": auth},
                   "messages": 5, "ships": ships if ships is not None else
                   [ship(0, 114.10, 21.80, 0.0, None, 230, t_list=tuple(range(0, 2400, 20)))]}
        p = self.ais / (name + ".bin")
        p.write_bytes(ae.seal(json.dumps(payload).encode(), key))
        return p

    def test_evaluated_and_published_files(self):
        p = self.sealed()
        r = ae.evaluate_file(p, KEY, self.mon, self.out, get=self.get, log=lambda *_: None)
        self.assertEqual(r, "evaluated")
        passes = json.loads((self.out / "passes.json").read_text())["passes"]
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0]["status"], "evaluated")
        self.assertEqual(passes[0]["detection"]["v1"]["300"]["ge100"], {"n": 1, "hit": 1})
        summ = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summ["detection"]["v1"]["300"]["ge100"]["rate"], 1.0)
        cand = json.loads((self.out / "candidates" / f"{self.acq_s[:10]}.json").read_text())
        flags = cand["passes"][self.acq_s]
        self.assertEqual([f["ais_300"] for f in flags], ["matched", "no_ais_coverage"])
        self.assertTrue((self.out / "meta.json").exists())
        # no ship rows in the published files: only candidate flags carry coordinates (the monitor's own, public ones)
        for name in ("passes.json", "summary.json", "meta.json"):
            text = (self.out / name).read_text()
            self.assertNotIn('"lon"', text)
            self.assertNotIn('"lat"', text)
        # a second evaluation of the same pass does nothing new
        self.assertEqual(ae.evaluate_file(p, KEY, self.mon, self.out, get=self.get, log=lambda *_: None), "evaluated")
        self.assertEqual(len(json.loads((self.out / "passes.json").read_text())["passes"]), 1)

    def test_waits_when_the_monitor_has_not_published_the_pass(self):
        (self.mon / "passes.json").write_text(json.dumps({"passes": [
            {"area": "HKS", "acquired_utc": self.acq_s, "platform": "sentinel-1d", "relative_orbit": 11, "status": "deferred"}]}))
        self.assertEqual(ae.evaluate_file(self.sealed(), KEY, self.mon, self.out, get=self.get, log=lambda *_: None), "wait")
        self.assertFalse((self.out / "passes.json").exists())

    def test_window_miss_and_no_reception_are_recorded_not_evaluated(self):
        p = self.sealed(t0=T0 + timedelta(minutes=30))
        self.assertEqual(ae.evaluate_file(p, KEY, self.mon, self.out, get=self.get, log=lambda *_: None), "skip")
        st = json.loads((self.out / "passes.json").read_text())["passes"][0]
        self.assertEqual(st["status"], "not_evaluated")
        self.assertIn("窓の外", st["reason"])
        s = json.loads((self.out / "summary.json").read_text())
        self.assertEqual((s["passes_evaluated"], s["passes_not_evaluated"]), (0, 1))

    def test_no_auth_and_wrong_key(self):
        p = self.sealed(auth=None)
        self.assertEqual(ae.evaluate_file(p, KEY, self.mon, self.out, get=self.get, log=lambda *_: None), "skip")
        p2 = self.sealed(name="ais-S1D-11-20261104T1033")
        self.assertEqual(ae.evaluate_file(p2, "other", self.mon, self.out, get=self.get, log=lambda *_: None), "error")

    def test_cli_without_key_does_nothing(self):
        import os
        old = os.environ.pop("AISSTREAM_API_KEY", None)
        logs = []
        try:
            class A:
                out_dir = str(self.ais)
                horizon_min = 65.0
                skip = []
            self.assertEqual(ae.cmd_record_due(A(), now=T0, get=lambda u: (_ for _ in ()).throw(AssertionError("network")),
                                               log=logs.append), 0)
            self.assertTrue(any("AISSTREAM_API_KEY" in x and "::notice" in x for x in logs))

            class B:
                ais_dir, monitor, out = str(self.ais), str(self.mon), str(self.out)
            self.assertEqual(ae.cmd_evaluate(B(), log=logs.append), 0)
        finally:
            if old is not None:
                os.environ["AISSTREAM_API_KEY"] = old


class RepoTest(unittest.TestCase):
    def test_hks_in_the_monitor_config(self):
        cfg = json.loads((ROOT / "ocean" / "monitor.json").read_text(encoding="utf-8"))
        a = next(x for x in cfg["areas"] if x["id"] == "HKS")
        self.assertTrue(a["enabled"])
        self.assertEqual(a["bbox"], [113.7, 21.5, 114.6, 22.3])
        self.assertLessEqual(a["pu_month_cap"], 240)                      # the 4部 proposal
        self.assertLessEqual(sum(x["pu_month_cap"] for x in cfg["areas"] if x["enabled"]), 3500)   # OCEAN_CAP

    def test_workflows(self):
        rec = (ROOT / ".github/workflows/ais-record.yml").read_text(encoding="utf-8")
        ev = (ROOT / ".github/workflows/ais-eval.yml").read_text(encoding="utf-8")
        for t in (rec, ev):
            self.assertIn("secrets.AISSTREAM_API_KEY", t)
            self.assertNotIn("CDSE_SH_CLIENT", t)               # no Sentinel Hub (PU) access here
            self.assertNotIn("cloudflare", t.lower())
        self.assertIn("schedule:", rec)
        self.assertIn("cron:", rec)
        self.assertIn("contents: read", rec)                    # the recorder writes nothing to the repository
        self.assertNotIn("git push", rec)
        self.assertIn("workflow_run", ev)
        self.assertIn("docs/data/ocean/ais_eval", ev)
        # the evaluator publishes only below ais_eval/ and only after the tests
        self.assertIn("tests.test_ocean_ais_eval", ev)
        self.assertIn("steps.tests.outcome == 'success'", ev)
        self.assertGreaterEqual(ev.count("git pull -q --rebase"), 1)

    def test_published_state_has_no_raw_ais(self):
        d = ROOT / "docs" / "data" / "ocean" / "ais_eval"
        for f in d.rglob("*.json"):
            text = f.read_text(encoding="utf-8").lower()
            for w in ('"mmsi"', '"shipname"', '"callsign"', '"imo"', '"name"'):
                self.assertNotIn(w, text, f.name)


if __name__ == "__main__":
    unittest.main()
