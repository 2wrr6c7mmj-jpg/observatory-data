"""ocean-chips-1: the site's images of the open-sea areas (ocean/chips.py, made by ocean.monitor from the arrays already
fetched: no extra request). Checked against the reader of the site (observatory-watch lib/sat-chips.ts: parseChipIndex,
parseChipDoc, safePath), which is not changed."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
    import PIL  # noqa: F401
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from ocean import chips as oc
    from ocean import monitor as m
    from sat import chips as satchips
    from sat.grid import Grid
    from sat import cdse
    from ocean import wide
    from tests import test_ocean_monitor as tom
    from tests.test_ocean_monitor import NOW, box, product, small_config

CREDIT = "Contains modified Copernicus Sentinel data 2026"
SAFE = re.compile(r"^[A-Za-z0-9_\-./]+$")


def safe_path(p) -> bool:   # lib/sat-chips.ts safePath
    return (isinstance(p, str) and 0 < len(p) < 300 and not p.startswith("/") and ".." not in p and ":" not in p
            and "\\" not in p and bool(SAFE.match(p)))


def site_reads_doc(t: unittest.TestCase, doc: dict) -> None:
    """lib/sat-chips.ts parseChipDoc: the doc is read and no box is dropped."""
    ip, cp = doc["image_px"], doc["chip_px"]
    t.assertTrue(all(isinstance(x, int) and x > 0 for x in ip + cp))
    t.assertTrue(safe_path(doc["image"]))
    t.assertLessEqual(cp[0], ip[0])
    t.assertLessEqual(cp[1], ip[1])
    for c in doc["candidates"]:
        b = c["box_px"]
        t.assertTrue(0 <= b[0] < b[2] <= cp[0] and 0 <= b[1] < b[3] <= cp[1], c)
    t.assertEqual(doc["credit"], CREDIT)


def site_reads_index(t: unittest.TestCase, idx: dict) -> list:
    """lib/sat-chips.ts parseChipIndex / parseScene: the entries of ``ocean`` the site keeps."""
    t.assertIsInstance(idx["sites"], list)
    kept = []
    for e in idx["ocean"]:
        t.assertIsInstance(e["area_id"], str)
        for s in e["scenes"]:
            t.assertTrue(safe_path(s["chip"]) and safe_path(s["image"]), s)
            datetime.strptime(s["acquired_utc"], "%Y-%m-%dT%H:%M:%SZ")
        kept.append(e["area_id"])
    return kept


class FakeCells:
    def __init__(self, bbox, cell=0.1):
        self.w, self.s, self.e, self.n = bbox
        self.cell = cell
        self.nx, self.ny = int(round((self.e - self.w) / cell)), int(round((self.n - self.s) / cell))


def feature(lon, lat, cls, contrast, area="AAA", t="2026-11-01T10:00:00Z"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"area": area, "acquired_utc": t, "class": cls, "contrast_vv_db": contrast,
                           "length_m_est_40m": 160, "confidence_label": m.CLASS_LABEL[cls], "ais": "unmatched"}}


@unittest.skipUnless(np is not None, "numpy / Pillow not installed")
class ChipTest(unittest.TestCase):
    def setUp(self):
        self.ac = FakeCells([124.0, 25.0, 124.4, 25.6])
        self.acq = {"acquired_utc": "2026-11-01T10:00:00Z", "platform": "sentinel-1c", "orbit_direction": "ascending",
                    "relative_orbit": "171"}

    def make(self, sat: Path, n_a=12, n_b=3, t="2026-11-01T10:00:00Z", area_id="AAA"):
        canvas = oc.Canvas(self.ac)
        g = Grid(124.0, 25.0, 124.2, 25.6, 500, 1500)     # the western half only (the rest stays black)
        rng = np.random.default_rng(1)
        vv = 10 * np.log10(rng.exponential(10 ** (-1.8), (g.height, g.width)))
        vh = vv - 7.0
        vh[0, 0] = np.nan
        canvas.add(g, vv, vh, np.ones(vv.shape, bool))
        feats, crops = [], {}
        for k in range(n_a + n_b):
            lon, lat = 124.01 + 0.012 * k, 25.05 + 0.03 * k
            feats.append(feature(lon, lat, "A" if k < n_a else "B", 10 + k, area_id, t))
            crops[(lon, lat)] = oc.crop(vv, vh, g, lon, lat)
        acq = {**self.acq, "acquired_utc": t}
        return oc.write_pass(sat, {"id": area_id, "name": "テスト"}, acq, canvas, feats, crops, t), canvas

    def test_image_and_chip_file_read_by_the_site(self):
        with tempfile.TemporaryDirectory() as d:
            sat = Path(d)
            r, canvas = self.make(sat)
            doc = json.loads((sat / "chips/ocean/AAA/2026-11-01T10-00-00Z.json").read_text(encoding="utf-8"))
            site_reads_doc(self, doc)
            img = (sat / doc["image"]).read_bytes()
            self.assertEqual(list(satchips.webp_size(img)), doc["image_px"])
            self.assertLessEqual(len(img), oc.MAX_IMAGE_BYTES)
            self.assertLessEqual(max(doc["chip_px"]), oc.OVERVIEW_MAX)
            self.assertEqual(doc["chip_px"], [canvas.grid.width, canvas.grid.height])
            # A first (stronger contrast first), then B; ocean-chips-2: no zoom tiles in the overview, every
            # candidate has its own tile in the sheet (1.28 km, the 40 m crop x2 here: no 20 m cut-out)
            self.assertEqual([c["class"] for c in doc["candidates"]], ["A"] * 12 + ["B"] * 3)
            self.assertEqual([c["n"] for c in doc["candidates"]], list(range(1, 16)))
            self.assertEqual(doc["candidates"][0]["contrast_vv_db"], 21)
            self.assertEqual(doc["tiles"], [])
            self.assertEqual(doc["counts"], {"A": 12, "B": 3})
            sheet = (sat / doc["sheet"]).read_bytes()
            self.assertEqual(list(satchips.webp_size(sheet)), doc["sheet_px"])
            self.assertEqual(doc["sheet_px"], [15 * oc.SHEET_TILE, oc.SHEET_TILE])
            self.assertLessEqual(len(sheet), oc.MAX_SHEET_BYTES)
            xy = [c["tile"]["xy"] for c in doc["candidates"]]
            self.assertEqual(xy, [[k * oc.SHEET_TILE, 0] for k in range(15)])
            for c in doc["candidates"]:
                t = c["tile"]
                self.assertEqual((t["size"], t["res_m"]), (oc.SHEET_TILE, 40))
                x0, y0, x1, y1 = t["frame"]
                self.assertTrue(0 <= x0 < x1 <= oc.SHEET_TILE and 0 <= y0 < y1 <= oc.SHEET_TILE)
                self.assertAlmostEqual(x1 - x0, 160 / 20 + 3, places=3)   # 160 m at 20 m per tile px, +1.5 px each side
            # the 40 m length is not given as length_m (the site's text says 10 m pixels)
            self.assertTrue(all(c["length_m"] is None and c["length_m_est_40m"] == 160 for c in doc["candidates"]))
            self.assertTrue(all(c["credibility"] in ("A", "B") for c in doc["candidates"]))
            # the eastern half was not fetched: black
            from PIL import Image
            im = np.asarray(Image.open(sat / doc["image"]).convert("RGB"))
            W, H = doc["chip_px"]
            self.assertLess(im[: H, W - 5:W].mean(), 8)
            self.assertGreater(im[: H, :5].mean(), 20)
            self.assertEqual(r["tiles"], 15)

    def test_a_pass_in_parts_keeps_the_earlier_tiles(self):
        with tempfile.TemporaryDirectory() as d:
            sat = Path(d)
            t = "2026-11-01T10:00:00Z"
            canvas = oc.Canvas(self.ac)
            g = Grid(124.0, 25.0, 124.2, 25.6, 500, 1500)
            rng = np.random.default_rng(2)
            vv = 10 * np.log10(rng.exponential(10 ** (-1.8), (g.height, g.width)))
            vh = vv - 7.0
            canvas.add(g, vv, vh, np.ones(vv.shape, bool))
            pts = [(124.01 + 0.012 * k, 25.05 + 0.03 * k) for k in range(3)]
            feats = [feature(lon, lat, "A", 10 + k, "AAA", t) for k, (lon, lat) in enumerate(pts)]
            acq = {**self.acq, "acquired_utc": t}
            first = {p: oc.crop(vv, vh, g, *p) for p in pts[:2]}
            oc.write_pass(sat, {"id": "AAA", "name": "テスト"}, acq, canvas, feats[:2], first, t)
            later = {pts[2]: oc.crop(vv, vh, g, *pts[2])}       # the second part fetched only the third
            oc.write_pass(sat, {"id": "AAA", "name": "テスト"}, acq, canvas, feats, later, t)
            doc = json.loads((sat / "chips/ocean/AAA/2026-11-01T10-00-00Z.json").read_text(encoding="utf-8"))
            self.assertTrue(all(c.get("tile") for c in doc["candidates"]))
            self.assertEqual(doc["sheet_px"], [3 * oc.SHEET_TILE, oc.SHEET_TILE])
            self.assertEqual(doc["sheet_v"], t)

    def test_sheet_prefers_the_20m_cut_out(self):
        rgb40 = np.zeros((oc.TILE_NATIVE, oc.TILE_NATIVE, 3), np.uint8)
        rgb40[oc.TILE_NATIVE // 2, oc.TILE_NATIVE // 2] = 255
        t, res = oc.sheet_tile(rgb40, None)
        self.assertEqual((t.shape, res), ((oc.SHEET_TILE, oc.SHEET_TILE, 3), 40))
        self.assertEqual(int(t[oc.SHEET_TILE // 2, oc.SHEET_TILE // 2, 0]), 255)   # the candidate stays in the middle
        rgb20 = np.full((oc.SHEET_TILE, oc.SHEET_TILE, 3), 7, np.uint8)
        t, res = oc.sheet_tile(rgb40, rgb20)
        self.assertEqual((int(t[0, 0, 0]), res), (7, 20))
        self.assertIsNone(oc.sheet_tile(None, None))

    def test_keep_held_index_and_splice(self):
        with tempfile.TemporaryDirectory() as d:
            sat = Path(d)
            t0 = datetime(2026, 11, 1, 10, tzinfo=timezone.utc)
            for k in range(5):
                self.make(sat, n_a=1, n_b=0, t=(t0 + timedelta(days=k)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            self.make(sat, n_a=1, n_b=0, area_id="HLD")
            cfg = {"areas": [{"id": "AAA", "name": "A 域", "bbox": [124.0, 25.0, 124.4, 25.6], "priority": 1, "enabled": True},
                             {"id": "HLD", "name": "保留", "bbox": [124.0, 25.0, 124.4, 25.6], "priority": 2, "enabled": True,
                              "publish": "hold"}]}
            r = oc.refresh(sat, cfg)
            self.assertEqual((r["areas"], r["scenes"]), (1, 3))
            self.assertFalse((sat / "chips/ocean/HLD").exists())          # held: nothing of it stays
            self.assertEqual(len(list((sat / "chips/ocean/AAA").glob("*.webp"))), 6)   # overview and sheet
            self.assertEqual(len(list((sat / "chips/ocean/AAA").glob("*.json"))), 3)
            frag = json.loads((sat / "chips/ocean/index.json").read_text(encoding="utf-8"))
            self.assertEqual([s["acquired_utc"][:10] for s in frag["ocean"][0]["scenes"]],
                             ["2026-11-05", "2026-11-04", "2026-11-03"])
            self.assertEqual(frag["credit"], CREDIT)
            # splice: the ports' sites stay, ocean comes from the fragment
            (sat / "chips/index.json").write_text(json.dumps({"version": "chips-1", "sites": [
                {"site_id": "ZJG", "scenes": [{"acquired_utc": "2025-10-26T10:48:45Z"}]}], "ocean": [], "credit": "x"}),
                encoding="utf-8")
            self.assertTrue(oc.splice(sat))
            self.assertFalse(oc.splice(sat))
            idx = json.loads((sat / "chips/index.json").read_text(encoding="utf-8"))
            self.assertEqual(idx["sites"][0]["site_id"], "ZJG")
            self.assertEqual(site_reads_index(self, idx), ["AAA"])
            self.assertEqual(idx["credit"], CREDIT)
            # sat.chips.update (the port run) copies the same ocean list
            self.assertEqual(satchips.ocean_entries(sat), idx["ocean"])
            # unchanged: nothing rewritten
            before = {p: p.read_bytes() for p in sat.rglob("*") if p.is_file()}
            oc.refresh(sat, cfg)
            self.assertEqual(before, {p: p.read_bytes() for p in sat.rglob("*") if p.is_file()})

    def test_sat_chips_update_keeps_the_ocean_list(self):
        from tests import test_sat_chips as tsc
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "chips/ocean").mkdir(parents=True)
            entry = {"area_id": "AAA", "name": "x", "scenes": [{"acquired_utc": "2026-11-01T10:00:00Z", "chip": "chips/ocean/AAA/a.json",
                                                                "image": "chips/ocean/AAA/a.webp"}]}
            (out / "chips/ocean/index.json").write_text(json.dumps({"ocean": [entry]}), encoding="utf-8")
            case = tsc.ChipsTest()
            rows, feats = case.make(out, 2)
            case.call(out, rows, feats)
            idx = json.loads((out / "chips/index.json").read_text(encoding="utf-8"))
            self.assertEqual(idx["ocean"], [entry])


@unittest.skipUnless(np is not None, "numpy / Pillow not installed")
class MonitorChipsTest(unittest.TestCase):
    """ocean.monitor with --chips-out: the same requests as without; held areas get no image."""
    setUp, tearDown = tom.RunTest.setUp, tom.RunTest.tearDown
    go = tom.HoldRunTest.go
    _cfg = tom.HoldRunTest._cfg

    def test_images_need_no_extra_request_and_skip_held_areas(self):
        t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
        prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
        with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
            s0, c0 = self.go(Path(d0), prods, config=self._cfg(Path(d0), False))
            s1, c1 = self.go(Path(d1), prods, config=self._cfg(Path(d1), False), chips_out=Path(d1) / "sat")
            self.assertEqual(c0.calls, c1.calls)                               # no extra request
            self.assertAlmostEqual(c0.processing_units, c1.processing_units)   # no extra PU
            self.assertEqual(s1["chips_errors"], [])
            sat = Path(d1) / "sat"
            self.assertFalse((Path(d0) / "sat").exists())
            # the counts and candidates are the same with and without images
            for f in ("candidates/2026-10-30.geojson", "series.json"):
                a = json.loads((Path(d0) / "monitor" / f).read_text(encoding="utf-8"))
                b = json.loads((Path(d1) / "monitor" / f).read_text(encoding="utf-8"))
                a.pop("generated_utc", None), b.pop("generated_utc", None)
                self.assertEqual(a.get("features", a.get("rows")), b.get("features", b.get("rows")))
            frag = json.loads((sat / "chips/ocean/index.json").read_text(encoding="utf-8"))
            self.assertEqual({e["area_id"] for e in frag["ocean"]}, {"T", "M"})
            for e in frag["ocean"]:
                doc = json.loads((sat / e["scenes"][0]["chip"]).read_text(encoding="utf-8"))
                site_reads_doc(self, doc)
                pub = [f for f in json.loads((Path(d1) / "monitor/candidates/2026-10-30.geojson").read_text(encoding="utf-8"))["features"]
                       if f["properties"]["area"] == e["area_id"]]
                self.assertEqual(len(doc["candidates"]), len(pub))
            t_doc = json.loads((sat / frag["ocean"][0]["scenes"][0]["chip"]).read_text(encoding="utf-8"))
            self.assertEqual(t_doc["area_id"], "T")
            self.assertGreaterEqual(len(t_doc["candidates"]), 1)               # the strong candidate added at the centre
            self.assertIn("画像", m.report(s1))
            # T held (republish, no network): its images are deleted
            m.republish(Path(d1) / "monitor", m.load_config(self._cfg(Path(d1), True)), NOW + timedelta(hours=1), sat)
            self.assertFalse((sat / "chips/ocean/T").exists())
            frag = json.loads((sat / "chips/ocean/index.json").read_text(encoding="utf-8"))
            self.assertEqual({e["area_id"] for e in frag["ocean"]}, {"M"})

    def test_held_area_gets_no_image(self):
        t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
        prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            s, c = self.go(tmp, prods, config=self._cfg(tmp, True), chips_out=tmp / "sat")
            self.assertGreater(len(c.calls), 0)
            self.assertFalse((tmp / "sat/chips/ocean/T").exists())
            self.assertFalse((tmp / "sat/chips/ocean/held").exists())
            self.assertTrue((tmp / "sat/chips/ocean/M").exists())
            frag = (tmp / "sat/chips/ocean/index.json").read_text(encoding="utf-8")
            self.assertNotIn('"T"', frag)

    def test_pass_in_parts_keeps_the_earlier_part_of_the_image(self):
        from PIL import Image
        t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
        prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfgp = small_config(tmp, max_cells=1)
            one = cdse.estimate_pu(*wide.rect_px((124.0, 25.0, 124.1, 25.1)), 3)
            sat = tmp / "sat"

            def lit():
                doc = json.loads((sat / "chips/ocean/T/2026-10-30T09-52-08Z.json").read_text(encoding="utf-8"))
                im = np.asarray(Image.open(sat / doc["image"]).convert("RGB"))[:doc["chip_px"][1], :doc["chip_px"][0]]
                return float((im.max(axis=2) > 0).mean())
            self.go(tmp, prods, config=cfgp, max_pu=one * 6.5, chips_out=sat)
            first = lit()
            self.assertLess(first, 0.6)
            self.go(tmp, prods, config=cfgp, chips_out=sat)
            self.assertGreater(lit(), 0.9)          # (the 400 m test pixels leave a few overview pixels empty)

    def test_dry_run_writes_no_image(self):
        t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
        prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self.go(tmp, prods, config=small_config(tmp), chips_out=tmp / "sat", dry_run=True)
            self.assertFalse((tmp / "sat").exists())


if __name__ == "__main__":
    unittest.main()
