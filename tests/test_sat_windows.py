"""win-1 (sat/windows.py): candidate windows only on sea-joined water along the berth lines."""
import json
import unittest

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import berths, run, windows
    from sat.grid import Grid

SITE = {"id": "TWN", "bbox": [110.0, 21.0, 110.01, 21.01], "pixel_m": 10,
        "windows": {"version": "win-1", "since_utc": "2026-11-04", "band_m": 80, "min_channel_m": 30}}


def parts():
    """~104 x 111 px: land in the west (cols < 30) with a pond; a quay along col 30; a long pier along row 40
    (cols 30-90); a 20 m slot between two short piers (rows 66-67 and 70-71, cols 30-50), open to the east; open water east."""
    g = Grid.for_site(SITE)
    h, w = g.height, g.width
    mask = np.zeros((h, w), np.uint8)
    mask[:, :30] = 1
    mask[10:16, 10:16] = 0                   # a pond on land
    mask[39:42, 30:90] = 2                   # the long pier
    mask[66:68, 30:50] = 2                   # two short piers 20 m apart: a slot between
    mask[70:72, 30:50] = 2
    lon = lambda c: g.west + c / g.width * (g.east - g.west)
    lat = lambda r: g.north - r / g.height * (g.north - g.south)
    lines = {"ways": [
        {"id": 1, "kind": "quay", "closed": False, "coords": [[lon(30), lat(1)], [lon(30), lat(h - 1)]]},
        {"id": 2, "kind": "pier", "closed": False, "coords": [[lon(30), lat(40.5)], [lon(90), lat(40.5)]]},
    ]}
    return g, mask, lines


@unittest.skipUnless(np is not None, "numpy not installed")
class WindowsTest(unittest.TestCase):
    def setUp(self):
        self.g, self.mask, self.lines = parts()
        self.bt = berths.Berths(SITE, self.lines, conf={})
        self.w = windows.Windows(SITE, self.g, self.mask, self.bt)

    def reason(self, r, c, base=None):
        return self.w.classify([{"row": r, "col": c}], base)[0]

    def test_reasons(self):
        self.assertEqual(self.reason(30.0, 33.0), "")            # alongside the quay
        self.assertEqual(self.reason(37.0, 70.0), "")            # alongside the pier
        self.assertEqual(self.reason(20.0, 20.0), "land")
        self.assertEqual(self.reason(12.0, 12.0), "pond")        # water not joined to the sea
        self.assertEqual(self.reason(69.0, 40.0), "enclosed")    # 20 m slot, under min_channel_m
        self.assertEqual(self.reason(90.0, 95.0), "anchorage")   # open water, far from any berth line: counted apart
        self.assertEqual(self.w.classify([{"row": 90.0, "col": 95.0, "zone": "quay"}])[0], "")   # berth-1 decides the band

    def test_stable_bright_baseline_pixel_is_land(self):
        base = {"vv_p10": np.full(self.mask.shape, -20.0), "vv_p90": np.full(self.mask.shape, -12.0)}
        base["vv_p10"][30, 33], base["vv_p90"][30, 33] = -3.0, -1.0       # a structure the mask misses
        self.assertEqual(self.reason(30.0, 33.0, base), "land")
        self.assertEqual(self.reason(31.0, 33.0, base), "")              # a normal berth pixel

    def test_regions_written_with_reasons(self):
        fc = self.w.regions()
        got = {f["properties"]["reason"] for f in fc["features"]}
        self.assertEqual(got, {"pond", "enclosed", "anchorage"})
        self.assertTrue(fc["has_berth_lines"])
        self.assertEqual(fc["version"], "win-1")
        for f in fc["features"]:
            ring = f["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            lo, la = f["properties"]["center"]
            self.assertTrue(110.0 <= lo <= 110.01 and 21.0 <= la <= 21.01)
        json.loads(windows.dumps(fc))

    def test_no_berth_lines_skips_the_band(self):
        w = windows.Windows(SITE, self.g, self.mask, berths.Berths(SITE, {"ways": []}, conf={}))
        self.assertFalse(w.has_lines)
        self.assertEqual(w.classify([{"row": 90.0, "col": 95.0, "zone": "harbour"}])[0], "")   # no anchorage split
        self.assertEqual(w.classify([{"row": 12.0, "col": 12.0}])[0], "pond")                 # steps 1-3 still apply
        self.assertEqual(w.regions()["counts_px"]["anchorage"], 0)

    def test_series_break_only_with_quay_lines(self):
        sites = [{**s, "enabled": True} for s in run.load_sites()[:1]]
        no_lines = {**sites[0], "id": "NOLINES"}
        br = [b for b in run.series_breaks(sites + [no_lines]) if b.get("kind") == "windows"]
        self.assertEqual([b["site_id"] for b in br], [sites[0]["id"]])

    def test_anchorage_frame_in_chips(self):
        from sat import chips
        f = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [110.005, 21.005]},
             "properties": {"n": 1, "zone": "harbour", "length_m": 40, "width_m": 20, "window": "anchorage"}}
        doc = chips.scene_doc(SITE, self.g, {"acquired_utc": "2026-11-05T10:00:00Z", "color_image": "x.webp"}, [f],
                              (10, 10), None, "TWN-land-v1")
        self.assertEqual(doc["candidates"][0]["window"], "anchorage")
        self.assertIn("錨泊", doc["candidates"][0]["window_label"])
        f["properties"].pop("window")
        doc = chips.scene_doc(SITE, self.g, {"acquired_utc": "2026-11-05T10:00:00Z", "color_image": "x.webp"}, [f],
                              (10, 10), None, "TWN-land-v1")
        self.assertNotIn("window", doc["candidates"][0])

    def test_active_from_since(self):
        self.assertFalse(windows.active(SITE, "2026-11-03T10:00:00Z"))
        self.assertTrue(windows.active(SITE, "2026-11-04T10:00:00Z"))
        self.assertFalse(windows.active({"id": "X"}, "2027-01-01T00:00:00Z"))

    def test_classify_windows_marks_and_counts(self):
        class Q:
            def windows(_self, naval=False):
                return self.w

            def naval(_self, acquired_utc):
                return False

            def baseline(_self, orbit):
                return None
        dets = [{"row": 30.0, "col": 33.0, "zone": "quay"}, {"row": 90.0, "col": 95.0, "zone": "harbour"},
                {"row": 12.0, "col": 12.0, "zone": "harbour"}]
        acq = {"acquired_utc": "2026-11-05T10:00:00Z", "relative_orbit": 157}
        res = run.classify_windows(Q(), SITE, acq, dets)
        self.assertEqual(res, {"land": 0, "pond": 1, "enclosed": 0, "anchorage": 1, "version": "win-1"})
        self.assertEqual([d["window_excluded"] for d in dets], ["", "", "pond"])
        self.assertEqual([d.get("window") for d in dets], [None, "anchorage", None])
        self.assertEqual([run.at_berth(d) for d in dets], [True, False, False])
        self.assertIsNone(run.classify_windows(Q(), SITE, {**acq, "acquired_utc": "2026-11-01T10:00:00Z"}, [{"row": 1, "col": 1}]))

    def test_every_site_has_a_rule(self):
        for s in run.load_sites():
            cfg = windows.config(s)
            self.assertIsNotNone(cfg, s["id"])
            self.assertEqual(cfg["version"], windows.VERSION)
            self.assertGreaterEqual(cfg["since_utc"], "2026-11-04")   # never before the decision (scenes not rewritten)


if __name__ == "__main__":
    unittest.main()
