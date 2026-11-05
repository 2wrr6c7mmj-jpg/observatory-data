"""MAW-land-v2: the land mask of a river site from the OSM water areas (sat/osm_mask.py, mask_water osm_polygons),
the baseline carried over from MAW-land-v1 without a fetch, and the switch without a recount (no CDSE request added)."""
from __future__ import annotations

import json
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from sat import baseline as fb
    from sat import osm_mask as om
    from sat import png, run
    from sat.grid import Grid


def way(coords, **tags):
    return {"type": "way", "tags": tags, "geometry": [{"lon": x, "lat": y} for x, y in coords]}


@unittest.skipUnless(np is not None, "numpy not installed")
class WaterPolygonTest(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(0.0, 0.0, 0.01, 0.01, 100, 100)

    def test_rings_are_joined_end_to_end(self):
        r = om._rings([[(0, 0), (1, 0)], [(1, 1), (1, 0)], [(0, 1), (1, 1)], [(0, 0), (0, 1)]])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], r[0][-1])
        self.assertEqual(len(r[0]), 5)
        self.assertEqual(len(om._rings([[(0, 0), (1, 0), (1, 1)]])[0]), 4)    # an open chain is closed

    def test_relation_with_a_hole_way_area_coastline_ignored_structure_kept(self):
        rel = {"type": "relation", "tags": {"type": "multipolygon", "natural": "water", "water": "river"}, "members": [
            {"type": "way", "role": "outer", "geometry": [{"lon": x, "lat": y} for x, y in [(-1, 0.002), (1, 0.002)]]},
            {"type": "way", "role": "outer", "geometry": [{"lon": x, "lat": y} for x, y in [(1, 0.002), (1, 0.006), (-1, 0.006)]]},
            {"type": "way", "role": "outer", "geometry": [{"lon": x, "lat": y} for x, y in [(-1, 0.006), (-1, 0.002)]]},
            {"type": "way", "role": "inner", "geometry": [{"lon": x, "lat": y} for x, y in
                                                         [(0.004, 0.003), (0.006, 0.003), (0.006, 0.005), (0.004, 0.005), (0.004, 0.003)]]}]}
        pond = way([(0.001, 0.008), (0.002, 0.008), (0.002, 0.009), (0.001, 0.009), (0.001, 0.008)], natural="water")
        coast = way([(0.0, 0.001), (0.01, 0.001)], natural="coastline")
        pier = way([(0.008, 0.0025), (0.008, 0.0035)], man_made="pier")
        osm = {"elements": [rel, pond, coast, pier]}
        m, info = om.build_mask(self.grid, osm, water_polygons=True)
        at = lambda lon, lat: m[tuple(int(v) for v in self.grid.to_pixel(lon, lat))]  # noqa: E731
        self.assertEqual(at(0.002, 0.004), om.WATER)          # the river
        self.assertEqual(at(0.005, 0.004), om.LAND)           # the island (inner ring)
        self.assertEqual(at(0.0015, 0.0085), om.WATER)        # a pond (closed way)
        self.assertEqual(at(0.005, 0.0005), om.LAND)          # south of the river: land, whatever the coastline says
        self.assertEqual(at(0.008, 0.003), om.STRUCTURE)      # a pier on the water
        self.assertEqual((info["water_mode"], info["water_areas"], info["water_inner_rings"]), ("osm_polygons", 2, 1))
        m0, info0 = om.build_mask(self.grid, osm)              # the coastline method ignores the water areas
        self.assertNotIn("water_mode", info0)

    def test_the_query_of_the_other_sites_is_unchanged(self):
        g = Grid(110.0, 21.0, 110.1, 21.1, 10, 10)
        self.assertEqual(om.overpass_query(g), '[out:json][timeout:90];(way["natural"="coastline"](21.0,110.0,21.1,110.1);'
                         'way["man_made"~"^(pier|breakwater|groyne|quay)$"](21.0,110.0,21.1,110.1);'
                         'way["bridge"]["highway"](21.0,110.0,21.1,110.1);way["bridge"]["railway"](21.0,110.0,21.1,110.1););out geom;')
        self.assertIn('relation["waterway"="riverbank"]', om.overpass_query(g, True))
        modes = {s["id"]: om.water_mode(s) for s in run.load_sites()}
        self.assertEqual({k for k, v in modes.items() if v != "coastline"}, {"MAW"})


@unittest.skipUnless(np is not None, "numpy not installed")
class MawV2Test(unittest.TestCase):
    def setUp(self):
        self.site = next(s for s in run.load_sites() if s["id"] == "MAW")
        self.grid = Grid.for_site(self.site)

    def test_the_mask_has_land_and_agrees_with_the_radar(self):
        """v1 had no land pixel; v2 agrees with the 24-scene VV p10 of ro142 (water dark: < -18 dB) on 95% or more."""
        self.assertEqual(run.mask_tag(self.site), "MAW-land-v2")
        m, meta = om.load_mask("MAW", 2)
        self.assertEqual((meta["water_mode"], meta["coastline_ways"]), ("osm_polygons", 0))
        self.assertEqual(m.shape, (self.grid.height, self.grid.width))
        self.assertGreater((m == om.LAND).mean(), 0.4)
        self.assertGreater((m == om.WATER).mean(), 0.35)
        v1, _ = om.load_mask("MAW", 1)
        self.assertEqual(int((v1 == om.LAND).sum()), 0)
        p10 = fb.dequantise(png.decode((fb.BASELINE_DIR / "MAW-land-v1" / "ro142-bl-1.vv_p10.png").read_bytes()))
        dark = p10 < -18
        ok = ((m == om.WATER) & dark) | ((m == om.LAND) & ~dark)
        self.assertGreater(ok.sum() / ((m != om.STRUCTURE)).sum(), 0.95)
        self.assertLess(((m == om.LAND) & dark).sum() / dark.sum(), 0.01)   # almost no dark water called land

    def test_the_baseline_is_the_v1_layers_on_the_new_mask(self):
        l1, m1 = fb.load({**self.site, "mask_version": 1}, 142)
        l2, m2 = fb.load(self.site, 142)
        mask, _ = om.load_mask("MAW", 2)
        keep = fb.keep_region(mask)
        for k in fb.LAYERS:
            a, b = l1[k], l2[k]
            both = keep & np.isfinite(a)
            self.assertTrue(np.array_equal(a[both], b[both]), k)
            self.assertFalse(np.isfinite(b[~keep]).any(), k)
        self.assertEqual((m2["mask_version"], m2["derived_from"]["mask_version"], m2["scenes"], m2["built_utc"]),
                         ("MAW-land-v2", "MAW-land-v1", m1["scenes"], m1["built_utc"]))
        self.assertLess(m2["stats"]["water_vv_p10_median_db"], -20)      # v1: -13.6 (the city counted as water)

    def test_no_scene_is_fetched_again(self):
        self.assertEqual(run.mask_since(self.site), "2026-11-06")
        self.assertIsNone(run.mask_since(next(s for s in run.load_sites() if s["id"] == "ZJG")))  # recount as before

        class S:
            rows = [{"site_id": "MAW", "acquired_utc": "2026-11-06T10:10:00Z", "mask_version": "MAW-land-v1"}]
            status = {"scenes": [{"site_id": "MAW", "acquired_utc": "2026-11-07T10:10:00Z", "mask_version": "MAW-land-v1",
                                  "result": "skipped_low_coverage"},
                                 {"site_id": "MAW", "acquired_utc": "2026-11-08T10:10:00Z", "mask_version": "MAW-land-v1",
                                  "result": "error"}]}
            done_any = run.Store.done_any
        acqs = [{"acquired_utc": t} for t in ("2026-11-05T10:10:00Z", "2026-11-06T10:10:00Z", "2026-11-07T10:10:00Z",
                                              "2026-11-08T10:10:00Z", "2026-11-09T10:10:00Z")]
        got = [a["acquired_utc"][:10] for a in run.rebuilt_mask_filter(self.site, S(), acqs, set())]
        self.assertEqual(got, ["2026-11-08", "2026-11-09"])     # before the date, counted or tried with v1: never again

    def test_series_break(self):
        br = [b for b in run.series_breaks(run.load_sites()) if b.get("site_id") == "MAW" and "kind" not in b]
        self.assertEqual([(b["from_mask_version"], b["to_mask_version"], b["date_utc"]) for b in br],
                         [("MAW-land-v1", "MAW-land-v2", "2026-11-06")])
        self.assertIn("取り直さない", br[0]["note"])


if __name__ == "__main__":
    unittest.main()
