"""berth-1 (2026-11-01): only the berth lines (piers, quays, hand-drawn lines, listed coastline sections)
count in the quay count, est-1, est-2 and the published chips; open-water candidates are kept."""
import json
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

from sat import berths

ROOT = Path(__file__).resolve().parent.parent
SITES = {s["id"]: s for s in json.loads((ROOT / "sat" / "sites.json").read_text(encoding="utf-8"))["sites"]}


def lines_of(site: dict) -> dict:
    lv = int(site.get("quay_lines_version", 1))
    return json.loads((ROOT / "sat" / "masks" / f"{site['id']}-quay-v{lv}.json").read_text(encoding="utf-8"))


class BerthsFileTest(unittest.TestCase):
    def test_entries_point_at_real_ways_and_sectors(self):
        conf = berths.load()
        self.assertEqual(conf["version"], berths.VERSION)
        for sid, sc in conf["sites"].items():
            self.assertIn(sid, SITES)
            kinds = {str(w["id"]): w["kind"] for w in lines_of(SITES[sid])["ways"]}
            secs = {x["id"] for x in SITES[sid].get("sectors") or []}
            for e in sc.get("coastline") or []:
                self.assertEqual(kinds.get(str(e["way"])), "coastline", (sid, e))
                self.assertTrue(e.get("sector") in secs or e.get("bbox"), (sid, e))
            for w in sc.get("exclude_ways") or []:
                self.assertIn(str(w), kinds, (sid, w))

    def test_zjg_rule(self):
        site = SITES["ZJG"]
        b = berths.Berths(site, lines_of(site))
        for w in lines_of(site)["ways"]:
            c = w["coords"]
            want = w["kind"] == "pier" and w["id"] != 971239820   # every coastline and the hard pier are out
            self.assertEqual(b.stretch_ok(w["kind"], w["id"], c), want, w["id"])
        self.assertEqual(b.counts()["coastline_sections"], 0)

    def test_listed_coastline_only_inside_its_sector(self):
        site = SITES["ZDH"]
        b = berths.Berths(site, lines_of(site))
        way = next(w for w in lines_of(site)["ways"] if w["id"] == 597183279)
        a1 = next(x for x in site["sectors"] if x["id"] == "A1")
        inside = [c for c in way["coords"] if berths._in_area(a1, c[0], c[1])]
        outside = [c for c in way["coords"] if not berths._in_area(a1, c[0], c[1])
                   and not any(berths._in_area(x, c[0], c[1]) for x in site["sectors"])]
        self.assertTrue(inside and outside)
        self.assertTrue(b.point_ok("coastline", 597183279, *inside[0]))
        self.assertFalse(b.point_ok("coastline", 597183279, *outside[0]))
        self.assertFalse(b.point_ok("coastline", 9977048, *inside[0]))   # another coastline: never


@unittest.skipUnless(np is not None, "numpy not installed")
class ClassifyTest(unittest.TestCase):
    LINES = {"ways": [
        {"id": 1, "kind": "pier", "coords": [[110.0, 21.0], [110.0, 21.002]]},
        {"id": 2, "kind": "coastline", "coords": [[110.01, 21.0], [110.01, 21.002]]},
    ]}

    def test_candidates(self):
        b = berths.Berths({"id": "TST"}, self.LINES)
        m = 111320.0 * np.cos(np.radians(21.001))
        pts = [(110.0 + 30 / m, 21.001),     # 30 m off the pier
               (110.01 - 30 / m, 21.001),    # 30 m off the coastline
               (110.005, 21.001)]            # ~520 m from both
        self.assertEqual(b.classify(pts), ["", "coastline", "off_berth"])
        self.assertEqual(berths.Berths({"id": "TST"}, {"ways": []}).classify(pts[:1]), ["off_berth"])

    def test_chips_leave_out_only_band_candidates(self):
        from sat import chips
        from sat.grid import Grid
        site = {"id": "TST", "bbox": [109.99, 20.99, 110.02, 21.01], "pixel_m": 10}
        g = Grid.for_site(site)
        f = lambda n, lon, zone: {"geometry": {"coordinates": [lon, 21.001]},
                                  "properties": {"n": n, "zone": zone, "length_m": 50, "width_m": 10}}
        feats = [f(1, 110.0003, "quay"), f(2, 110.0097, "quay"), f(3, 110.0097, "harbour")]
        b = berths.Berths(site, self.LINES)
        band = [x for x in feats if x["properties"]["zone"] == "quay"]
        why = b.classify([tuple(x["geometry"]["coordinates"]) for x in band])
        self.assertEqual(why, ["", "coastline"])
        keep = [x for x in feats if x not in band or not why[band.index(x)]]
        doc = chips.scene_doc(site, g, {"acquired_utc": "2026-11-01T00:00:00Z", "color_image": "c.webp"}, keep, (10, 10),
                              None, "TST-land-v1", {"off_berth": 0, "coastline": 1})
        self.assertEqual([c["n"] for c in doc["candidates"]], [1, 3])   # the open-water candidate stays
        self.assertEqual(doc["excluded_counts"], {"off_berth": 0, "coastline": 1})

    def test_series_break_and_meta(self):
        from sat import run
        sites = [s for s in SITES.values() if s.get("enabled")]
        br = [x for x in run.series_breaks(sites) if x.get("kind") == "berth_only"]
        self.assertEqual({x["site_id"] for x in br}, {s["id"] for s in sites})
        self.assertTrue(all("岸壁に限定" in x["note"] and x["date_utc"] == berths.SINCE_UTC for x in br))
        self.assertEqual(run.site_status(SITES["ZJG"])["quay"]["berths"]["berth_lines"], 12)


if __name__ == "__main__":
    unittest.main()
