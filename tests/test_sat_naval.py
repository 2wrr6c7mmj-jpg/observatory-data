"""naval-1: the class of every berth line (sat/berth_class/<site>.json) and the headline counts on the naval lines only."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from sat import berths

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parent.parent

SITE = {"id": "TNV", "bbox": [110.0, 21.0, 110.02, 21.02],
        "sectors": [{"id": "A1", "role": "naval", "bbox": [110.0, 21.0, 110.01, 21.02]}]}
LINES = {"ways": [
    {"id": 1, "kind": "pier", "coords": [[110.002, 21.010], [110.004, 21.010]]},       # naval
    {"id": 2, "kind": "pier", "coords": [[110.012, 21.010], [110.014, 21.010]]},       # civil
    {"id": 3, "kind": "pier", "coords": [[110.016, 21.015], [110.018, 21.015]]},       # unknown (no entry)
    {"id": 4, "kind": "coastline", "coords": [[110.000, 21.005], [110.020, 21.005]]},  # added from NAVAL_SINCE in A1
    {"id": -1, "kind": "pier", "coords": [[110.006, 21.012], [110.007, 21.013]]},      # drawn, excluded in berth-1
]}
CLASSES = {"site": "TNV", "default": "unknown", "lines": [
    {"way": 1, "kind": "pier", "class": "naval", "basis": ["test"]},
    {"way": 2, "kind": "pier", "class": "civil", "basis": ["test"]},
    {"way": 4, "kind": "coastline", "class": "naval", "added": True, "part": {"sector": "A1"}, "basis": ["test"]},
    {"way": -1, "kind": "pier", "class": "naval", "added": True, "basis": ["test"]},
]}
CONF = {"keep_m": 100.0, "own_tolerance_m": 30.0, "sites": {"TNV": {"coastline": [], "exclude_ways": [-1]}}}


class LineClassTest(unittest.TestCase):
    def test_classes_and_default(self):
        b = berths.Berths(SITE, LINES, CONF, classes=CLASSES)
        self.assertEqual(b.line_class(1, 110.003, 21.01), "naval")
        self.assertEqual(b.line_class(2, 110.013, 21.01), "civil")
        self.assertEqual(b.line_class(3, 110.017, 21.015), "unknown")
        self.assertEqual(b.stretch_class(2, LINES["ways"][1]["coords"]), "civil")

    def test_added_sections_only_in_the_naval_variant(self):
        b0 = berths.Berths(SITE, LINES, CONF, classes=CLASSES)
        b1 = berths.Berths(SITE, LINES, CONF, classes=CLASSES, naval=True)
        # berth-1 as before: the coastline is not a berth, the drawn pier is excluded
        self.assertFalse(b0.point_ok("coastline", 4, 110.005, 21.005))
        self.assertFalse(b0.point_ok("pier", -1, 110.0065, 21.0125))
        # naval-1: the added coastline inside its sector and the drawn pier are berths
        self.assertTrue(b1.point_ok("coastline", 4, 110.005, 21.005))
        self.assertFalse(b1.point_ok("coastline", 4, 110.015, 21.005))   # outside the part
        self.assertTrue(b1.point_ok("pier", -1, 110.0065, 21.0125))
        self.assertEqual(b1.line_class(4, 110.005, 21.005), "naval")

    def test_polygon_part_with_buffer_and_clip(self):
        ring = [[110.0, 21.0], [110.01, 21.0], [110.01, 21.01], [110.0, 21.01], [110.0, 21.0]]
        self.assertTrue(berths._in_area({"polygon": ring}, 110.005, 21.005))
        self.assertFalse(berths._in_area({"polygon": ring}, 110.0102, 21.005))
        self.assertTrue(berths._in_area({"polygon": ring, "buffer_m": 30}, 110.0102, 21.005))     # ~21 m outside
        self.assertFalse(berths._in_area({"polygon": ring, "buffer_m": 30, "clip": [110.0, 21.0, 110.004, 21.01]},
                                         110.005, 21.005))

    @unittest.skipUnless(np is not None, "numpy not installed")
    def test_nearest_class(self):
        b = berths.Berths(SITE, LINES, CONF, classes=CLASSES)
        got = b.nearest_class([(110.003, 21.0103), (110.013, 21.0103), (110.017, 21.0153), (110.01, 21.019)], 80.0)
        self.assertEqual(got, ["naval", "civil", "unknown", None])

    def test_feature_ok_by_date(self):
        b = berths.Berths(SITE, LINES, CONF, classes=CLASSES)
        c = LINES["ways"][1]["coords"]
        before = {"line_kind": "pier", "osm_way": 2, "acquired_utc": "2026-11-04T10:00:00Z"}
        after = {**before, "acquired_utc": berths.NAVAL_SINCE + "T10:00:00Z"}
        self.assertTrue(berths.feature_ok(b, before, c))      # past scenes: never rewritten
        self.assertFalse(berths.feature_ok(b, after, c))      # civil from NAVAL_SINCE
        nav = {"line_kind": "pier", "osm_way": 1, "acquired_utc": after["acquired_utc"]}
        self.assertTrue(berths.feature_ok(b, nav, LINES["ways"][0]["coords"]))


    def test_site_naval_since(self):
        """naval-gap-1: a site's naval_since (sat/berths.json) starts naval-1 earlier; the others keep NAVAL_SINCE."""
        conf = {**CONF, "sites": {"TNV": {"coastline": [], "exclude_ways": [-1], "naval_since": "2026-11-01"}}}
        self.assertEqual(berths.naval_since("TNV", conf), "2026-11-01")
        self.assertEqual(berths.naval_since("TNV", CONF), berths.NAVAL_SINCE)
        self.assertEqual(berths.naval_since("NOPE", conf), berths.NAVAL_SINCE)
        b = berths.Berths(SITE, LINES, conf, classes=CLASSES)
        civil = {"line_kind": "pier", "osm_way": 2, "acquired_utc": "2026-11-02T10:00:00Z"}
        self.assertFalse(berths.feature_ok(b, civil, LINES["ways"][1]["coords"]))   # naval-1 already on 11-02
        self.assertTrue(berths.feature_ok(b, {**civil, "acquired_utc": "2026-10-31T10:00:00Z"}, LINES["ways"][1]["coords"]))

    def test_gap_sites_in_berths_json(self):
        """The sites of the 2026-11-01..04 gap start naval-1 with berth-1; Zhanjiang (golden) and the others do not move."""
        for sid in ("LSN", "QDH", "QJG", "QGZ", "XSN", "ZDH", "SWS"):
            self.assertEqual(berths.naval_since(sid), berths.SINCE_UTC, sid)
            self.assertTrue(berths.naval_active(sid, "2026-11-02T09:56:32Z"), sid)
        for sid in ("ZJG", "SYL", "SLP", "STU", "XMN", "MAW"):
            self.assertEqual(berths.naval_since(sid), berths.NAVAL_SINCE, sid)
            self.assertFalse(berths.naval_active(sid, "2026-11-02T09:56:32Z"), sid)


@unittest.skipUnless(np is not None, "numpy not installed")
class NavalGapSeriesTest(unittest.TestCase):
    def test_series_break_dates_per_site(self):
        """meta.json: the naval series break of a gap site starts 2026-11-01 and says the gap scenes were counted again."""
        from sat import run as srun
        br = {b["site_id"]: b for b in srun.series_breaks(srun.load_sites()) if b.get("kind") == "naval"}
        self.assertEqual(br["LSN"]["date_utc"], "2026-11-01")
        self.assertIn("naval-gap-1", br["LSN"]["note"])
        self.assertEqual(br["STU"]["date_utc"], berths.NAVAL_SINCE)
        self.assertNotIn("naval-gap-1", br["STU"]["note"])
        self.assertNotIn("ZJG", {k for k, v in br.items() if v["date_utc"] != berths.NAVAL_SINCE})


@unittest.skipUnless(np is not None, "numpy not installed")
class ClassifyNavalTest(unittest.TestCase):
    def test_candidates_by_date(self):
        from sat import run
        b1 = berths.Berths(SITE, LINES, CONF, classes=CLASSES, naval=True)

        class Q:
            def __init__(self, on):
                self.on = on

            def berths(self, naval=False):
                return b1

            def naval(self, acquired_utc):
                return self.on

        def dets():
            return [{"zone": "quay", "lon": 110.003, "lat": 21.0103}, {"zone": "quay", "lon": 110.013, "lat": 21.0103},
                    {"zone": "harbour", "lon": 110.017, "lat": 21.0155}, {"zone": "harbour", "lon": 110.01, "lat": 21.019}]
        d = dets()
        self.assertIsNone(run.classify_naval(Q(False), SITE, {"acquired_utc": "2026-11-04T00:00:00Z"}, d))
        self.assertTrue(all(run.at_berth(x) for x in d))
        d = dets()
        n = run.classify_naval(Q(True), SITE, {"acquired_utc": "2026-11-05T00:00:00Z"}, d)
        self.assertEqual([x["berth_class"] for x in d], ["naval", "civil", "unknown", "unknown"])
        self.assertEqual([run.at_berth(x) for x in d], [True, False, False, False])
        self.assertEqual((n["quay_naval"], n["quay_civil"], n["harbour_unknown"]), (1, 1, 2))


@unittest.skipUnless(np is not None, "numpy not installed")   # sat.berth_class needs numpy
class ClassFilesTest(unittest.TestCase):
    """Every berth line of every enabled site has a class with its evidence; the published map matches the files."""

    def test_every_berth_line_has_a_class(self):
        from sat import berth_class, quay_lines
        from sat.run import load_sites
        for s in load_sites():
            if not s.get("enabled"):
                continue
            c = berths.load_classes(s["id"])
            self.assertIsNotNone(c, s["id"])
            lv = int(s.get("quay_lines_version", quay_lines.LINES_VERSION))
            self.assertEqual(c["lines_version"], f"{s['id']}-quay-v{lv}")
            lines = quay_lines.load(s["id"], lv)
            have = {(str(e["way"]), json.dumps(e.get("part"), sort_keys=True)) for e in c["lines"]}
            for it in berth_class.berth_items(s, lines):
                self.assertIn((str(it["way"]), json.dumps(it["part"], sort_keys=True)), have, (s["id"], it["way"]))
            ids = {str(w["id"]) for w in lines["ways"]}
            for e in c["lines"]:
                self.assertIn(e["class"], berths.CLASSES)
                self.assertTrue(e.get("basis"), (s["id"], e["way"]))
                self.assertIn(str(e["way"]), ids, (s["id"], e["way"]))   # a fixed entry of a line no longer drawn

    def test_geojson_is_up_to_date(self):
        from sat import berth_class
        from sat.run import load_sites
        want = berth_class.geojson([s for s in load_sites() if s.get("enabled")])
        have = json.loads((ROOT / "docs" / "data" / "sat" / "berth_class.geojson").read_text(encoding="utf-8"))
        self.assertEqual(have["features"], json.loads(json.dumps(want["features"])),
                         "python -m sat.berth_class geojson で作り直す")
        self.assertTrue(any("OpenStreetMap" in c for c in have["credit"]))
        self.assertTrue(any("溫約瑟" in c for c in have["credit"]))
        # naval-gap-1: the start of naval-1 per site (berths.json naval_since), on the file and on every line
        self.assertEqual(have["since_utc_by_site"], json.loads(json.dumps(want["since_utc_by_site"])))
        self.assertEqual(have["since_utc_by_site"]["LSN"], berths.naval_since("LSN"))
        self.assertTrue(all(f["properties"]["since_utc"] == have["since_utc_by_site"][f["properties"]["site_id"]]
                            for f in have["features"]))

    def test_lushun_east_basin_south_quay(self):
        """旅順の内港の東の泊地の南岸（2026-11-07 追加）：naval-1 の撮影でだけ着く線になり、区分は naval。"""
        from sat import quay_lines
        from sat.run import load_sites
        site = next(s for s in load_sites() if s["id"] == "LSN")
        lines = quay_lines.load("LSN", int(site.get("quay_lines_version", quay_lines.LINES_VERSION)))
        pt = (121.2595, 38.7992)   # 南岸の岸壁の上
        self.assertFalse(berths.Berths(site, lines, naval=False).point_ok("coastline", 696162852, *pt))
        b = berths.Berths(site, lines, naval=True)
        self.assertTrue(b.point_ok("coastline", 696162852, *pt))
        self.assertEqual(b.line_class(696162852, *pt), "naval")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(np is not None, "numpy not installed")
class NavalOnlyFramesTest(unittest.TestCase):
    """User's instruction 2026-11-05: the counts and the squares show the navy only. From NAVAL_SINCE the chips frame
    neither civil/unknown berth candidates nor the open-water anchorage; before it, the anchorage stays framed."""

    def test_anchorage_not_framed_on_naval_scenes(self):
        import io
        import tempfile
        from PIL import Image
        from sat import chips
        from sat.run import load_sites
        site = next(s for s in load_sites() if s["id"] == "SYL")
        lon = (site["bbox"][0] + site["bbox"][2]) / 2
        lat = (site["bbox"][1] + site["bbox"][3]) / 2
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows = []
            feats = []
            for t in ("2026-11-04T10:00:00Z", "2026-11-05T10:00:00Z"):
                img = f"color/SYL/{t[:10]}.webp"
                (out / img).parent.mkdir(parents=True, exist_ok=True)
                buf = io.BytesIO()
                Image.new("RGB", (20, 20)).save(buf, "WEBP")
                (out / img).write_bytes(buf.getvalue())
                rows.append({"site_id": "SYL", "acquired_utc": t, "color_image": img, "mask_version": "x"})
                base = {"acquired_utc": t, "site_id": "SYL", "mask_version": "x", "length_m": 100, "width_m": 20}
                feats += [
                    {"geometry": {"coordinates": [lon, lat]}, "properties": {**base, "n": 1, "zone": "quay", "berth_class": "naval", "naval_excluded": ""}},
                    {"geometry": {"coordinates": [lon, lat]}, "properties": {**base, "n": 2, "zone": "harbour", "window": "anchorage"}},
                    {"geometry": {"coordinates": [lon, lat]}, "properties": {**base, "n": 3, "zone": "quay", "berth_class": "civil", "naval_excluded": "civil"}},
                ]
            chips.update(out, [site], rows, lambda s, m: feats, lambda s: "x", lambda r: r["mask_version"])
            docs = {p.name[:10]: json.loads(p.read_text(encoding="utf-8")) for p in (out / "chips").rglob("SYL/*.json")}
            before, after = docs["2026-11-04"], docs["2026-11-05"]
            self.assertEqual([c["n"] for c in after["candidates"]], [1])
            self.assertEqual(after["excluded_counts"]["naval_anchorage"], 1)
            self.assertEqual(after["excluded_counts"]["naval_civil"], 1)
            self.assertIn(2, [c["n"] for c in before["candidates"]])   # before NAVAL_SINCE: unchanged
