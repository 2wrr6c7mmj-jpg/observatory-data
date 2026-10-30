import json
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone

from ocean import coverage_probe as cp

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def product(sat, mode, kind, start, abs_orbit, box, direction="ASCENDING", rel=69, pub_h=3.0, cog=False, ptype=None):
    w, s, e, n = box
    st = start.strftime("%Y%m%dT%H%M%S")
    sp = (start + timedelta(seconds=25)).strftime("%Y%m%dT%H%M%S")
    res = "H" if kind == "GRD" else "_"
    name = f"S1{sat}_{mode}_{kind}{res}_{'1' if kind == 'GRD' else '2'}SDV_{st}_{sp}_{abs_orbit:06d}_0ABCDE_1234" + ("_COG.SAFE" if cog else ".SAFE")
    attrs = [{"Name": "orbitDirection", "Value": direction}, {"Name": "relativeOrbitNumber", "Value": rel}]
    if ptype:
        attrs.append({"Name": "productType", "Value": ptype})
    return {
        "Id": name, "Name": name,
        "ContentDate": {"Start": start.strftime("%Y-%m-%dT%H:%M:%S.123456Z"), "End": start.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        "PublicationDate": (start + timedelta(hours=pub_h)).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
        "GeoFootprint": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        "Attributes": attrs,
    }


AREAS = {"X": {"name": "test", "bbox": [120.0, 22.0, 121.0, 23.0]}}


class CoverageTest(unittest.TestCase):
    def products(self):
        t = T0 + timedelta(hours=10, minutes=5)
        return [
            # One pass in two slices (same platform+orbit): the whole area, counted once per cell.
            product("A", "IW", "GRD", t, 60001, [119.0, 22.0, 122.0, 22.5], pub_h=2),
            product("A", "IW", "GRD", t + timedelta(seconds=25), 60001, [119.0, 22.4, 122.0, 23.5], pub_h=4),
            # Its COG copy: same slice, not counted again.
            product("A", "IW", "GRD", t, 60001, [119.0, 22.0, 122.0, 22.5], pub_h=6, cog=True),
            # A descending pass over the western half only.
            product("C", "IW", "GRD", T0 + timedelta(days=6, hours=22), 5001, [119.5, 21.5, 120.5, 23.5],
                    direction="DESCENDING", rel=105, pub_h=10),
            # Outside the area.
            product("C", "IW", "GRD", T0 + timedelta(days=7), 5002, [125.0, 25.0, 126.0, 26.0]),
            # Other modes.
            product("A", "WV", "OCN", T0 + timedelta(days=3), 60050, [120.2, 22.2, 120.4, 22.4]),
            product("A", "IW", "OCN", t, 60001, [119.0, 22.0, 122.0, 23.5], pub_h=1),
        ]

    def test_counts_passes_per_cell(self):
        res = cp.analyse(self.products(), T0, T0 + timedelta(days=60), AREAS)
        a = res["areas"]["X"]
        self.assertEqual((a["cells"]["nx"], a["cells"]["ny"]), (4, 4))
        self.assertEqual(a["passes_iw"], 2)
        # Western half (cell centres 120.125, 120.375) has both passes, eastern half one.
        for row in a["counts"]:
            self.assertEqual(row, [2, 2, 1, 1])
        self.assertEqual(a["counts_by_direction"]["descending"][0], [1, 1, 0, 0])
        self.assertEqual(a["cells"]["min"], 1)
        self.assertEqual(a["cells"]["max"], 2)
        self.assertEqual(a["passes_iw_by_direction"], {"ascending": 1, "descending": 1})
        ro = {(o["direction"], o["relative_orbit"]): o for o in a["orbits"]}
        self.assertEqual(ro[("ascending", 69)]["utc_first"], "10:05")
        self.assertEqual(ro[("descending", 105)]["share_of_cells_max"], 0.5)
        self.assertEqual(a["utc_hours"], {"10": {"ascending": 1, "descending": 0, "other": 0},
                                          "22": {"ascending": 0, "descending": 1, "other": 0}})
        self.assertEqual(a["other_modes"]["WV"]["passes"], 1)
        self.assertEqual(a["other_modes"]["WV"]["product_types"], ["WV_OCN__2S"])
        self.assertNotIn("EW", a["other_modes"])
        self.assertAlmostEqual(a["revisit_gap_h"]["median_h"], 6 * 24 + 22 - 10 - 5 / 60, places=1)

    def test_publication_lag_per_type(self):
        res = cp.analyse(self.products(), T0, T0 + timedelta(days=60), AREAS)
        lag = res["publication_lag"]
        self.assertEqual(lag["IW_GRDH_1S"]["n"], 4)
        self.assertAlmostEqual(lag["IW_GRDH_1S"]["median_h"], 3.5, places=2)  # 2, 3, 4, 10 h
        self.assertEqual(lag["IW_GRDH_1S-COG"]["n"], 1)
        self.assertIn("IW_OCN__2S", lag)

    def test_product_type_attribute_wins(self):
        p = product("A", "IW", "GRD", T0, 1, [0, 0, 1, 1], ptype="IW_GRDH_1S-COG", cog=True)
        self.assertEqual(cp.product_info(p)["product_type"], "IW_GRDH_1S-COG")

    def test_wkt_fallback_and_multipolygon(self):
        p = product("A", "IW", "GRD", T0, 1, [0, 0, 1, 1])
        del p["GeoFootprint"]
        p["Footprint"] = "geography'SRID=4326;MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((5 5,6 5,6 6,5 5)))'"
        rr = cp.product_info(p)["rings"]
        self.assertEqual(len(rr), 2)
        self.assertTrue(cp.inside(rr[0], 0.5, 0.5))
        self.assertFalse(cp.inside(rr[1], 0.5, 0.5))

    def test_fetch_pages_and_windows(self):
        calls = []
        many = [{"Id": str(i)} for i in range(cp.PAGE)]

        def get(url):
            calls.append(url)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            self.assertIn("Collection/Name eq 'SENTINEL-1'", q["$filter"][0])
            self.assertIn("OData.CSC.Intersects", q["$filter"][0])
            if len(calls) == 1:
                return {"value": many}
            if len(calls) == 2:
                self.assertEqual(q["$skip"], [str(cp.PAGE)])
                return {"value": [{"Id": "x"}, {"Id": "0"}]}
            return {"value": []}

        orig = cp.time.sleep
        cp.time.sleep = lambda s: None
        try:
            out, n = cp.fetch_products([0, 0, 1, 1], T0, T0 + timedelta(days=12), get=get, log=lambda *a: None)
        finally:
            cp.time.sleep = orig
        self.assertEqual(len(out), cp.PAGE + 1)  # duplicate Id dropped
        self.assertEqual(n, 4)  # 2 pages in the first window, then 2 more windows (5 + 5 + 2 days)

    def test_report_fits_annotations_and_json_is_compact(self):
        res = cp.analyse(self.products(), T0, T0 + timedelta(days=60), cp.AREAS)
        for k, v in cp.report(res).items():
            self.assertLess(len(v), 4000, k)
        self.assertEqual(res["areas"]["R"]["cells"]["nx"], 36)
        self.assertEqual(len(res["areas"]["R"]["counts"]), 24)
        s = cp.dumps(res)
        self.assertEqual(json.loads(s)["areas"]["T"]["counts"], res["areas"]["T"]["counts"])
        self.assertIn("[0,0,0,0", s)

    def test_parse_time_fractions(self):
        self.assertEqual(cp.parse_time("2026-09-01T10:05:00.1234567Z").microsecond, 123456)
        self.assertEqual(cp.parse_time("2026-09-01T10:05:00Z").hour, 10)


if __name__ == "__main__":
    unittest.main()
