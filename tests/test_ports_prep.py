import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import osm_mask
    from sat.grid import Grid
    from tools import ports_prep as pp


@unittest.skipUnless(np is not None, "numpy not installed")
class PortsPrepTest(unittest.TestCase):
    SITE = {"id": "TST", "bbox": [120.0, 30.0, 120.01, 30.01], "center_lat": 30.005, "center_lon": 120.005, "pixel_m": 10,
            "mask_version": 1, "enabled": True}

    def test_todo_is_the_sites_without_a_mask(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sites = [dict(self.SITE), {**self.SITE, "id": "OLD"}, {**self.SITE, "id": "OFF", "enabled": False}]
            osm_mask.mask_paths("OLD", 1, d)[0].write_bytes(b"x")
            self.assertEqual([s["id"] for s in pp.todo_sites(sites, [], d)], ["TST"])
            self.assertEqual([s["id"] for s in pp.todo_sites(sites, ["OLD", "OFF"], d)], ["OLD"])

    def test_one_sided_coastline_mask_asks_for_water_polygons(self):
        self.assertTrue(pp.needs_water_polygons({"coastline_ways": 0, "land_fraction": 0.0, "water_fraction": 0.96}))
        self.assertTrue(pp.needs_water_polygons({"coastline_ways": 2, "land_fraction": 0.99, "water_fraction": 0.005}))
        self.assertFalse(pp.needs_water_polygons({"coastline_ways": 3, "land_fraction": 0.6, "water_fraction": 0.38}))

    def test_river_port_is_built_again_from_water_areas(self):
        grid = Grid.for_site(self.SITE)
        w, s, e, n = grid.bbox
        # no coastline; a river (natural=water) over the southern half
        river = {"type": "way", "id": 1, "tags": {"natural": "water"},
                 "geometry": [{"lon": w, "lat": s}, {"lon": e, "lat": s}, {"lon": e, "lat": (s + n) / 2},
                              {"lon": w, "lat": (s + n) / 2}, {"lon": w, "lat": s}]}
        calls = []

        def fetch(g, water_polygons=False):
            calls.append(water_polygons)
            return {"elements": [river] if water_polygons else [], "osm3s": {"timestamp_osm_base": "2026-11-12T00:00:00Z"}}

        with tempfile.TemporaryDirectory() as d:
            r = pp.land_mask(self.SITE, Path(d), 0, fetch)
            self.assertEqual(calls, [False, True])
            self.assertTrue(r["switched_to_water_polygons"] and r["water_polygons"])
            self.assertAlmostEqual(r["water_fraction"], 0.5, delta=0.05)
            meta = json.loads(osm_mask.mask_paths("TST", 1, Path(d))[1].read_text(encoding="utf-8"))
            self.assertEqual(meta["water_mode"], "osm_polygons")
            self.assertEqual(meta["coastline_attempt"]["coastline_ways"], 0)


if __name__ == "__main__":
    unittest.main()
