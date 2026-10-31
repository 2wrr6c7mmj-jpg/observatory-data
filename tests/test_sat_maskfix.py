"""ZDH の区画 A1 の多角形化と QGZ-land-v2（陸の修正）の準備（2部の検品 mask-check-qgz-zdh、2026-10-31）。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    from sat import osm_mask, quay, quay_lines, run
    from sat import sectors as fsectors
    from sat.grid import Grid
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
OLD_A1_BBOX = [122.0052, 29.9972, 122.0284, 30.0318]
NEW_A1 = [[122.0045, 29.9972], [122.0284, 29.9972], [122.0284, 30.0318], [122.0052, 30.0318], [122.0052, 30.0262], [122.0045, 30.0262]]
STRIP = (122.0045, 29.9972, 122.0052, 30.0262)  # the only area the polygon adds to the old bbox
NW_PIERS = (814801319, 814801322, 1566623245, 1566623249)  # the industrial port, north-west of A1 (B)


def _site(sid):
    return next(s for s in run.load_sites() if s["id"] == sid)


def _old_zdh(site):
    sc = [dict(x) for x in site["sectors"]]
    a1 = sc[0]
    a1.pop("polygon")
    a1["bbox"] = OLD_A1_BBOX
    return {**site, "sectors": sc}


@unittest.skipUnless(np is not None, "numpy not installed")
class ZdhSectorA1Test(unittest.TestCase):
    def test_polygon(self):
        z = _site("ZDH")
        a1 = z["sectors"][0]
        self.assertEqual((a1["id"], a1["role"], a1.get("bbox"), a1["polygon"]), ("A1", "naval", None, NEW_A1))
        self.assertEqual(fsectors.overlaps(z), [])
        self.assertEqual(fsectors.outside(z), [])
        self.assertEqual(z["mask_version"], 2)  # the mask and the baseline stay

    def test_raster_changes_only_in_strip(self):
        z = _site("ZDH")
        g = Grid.for_site(z)
        old = fsectors.sector_raster(g, run.site_sectors(_old_zdh(z)))
        new = fsectors.sector_raster(g, run.site_sectors(z))
        ch = old != new
        self.assertGreater(int(ch.sum()), 0)
        r, c = np.nonzero(ch)
        lon, lat = g.to_lonlat(r + 0.5, c + 0.5)
        w, s, e, n = STRIP
        self.assertTrue(((lon >= w) & (lon < e) & (lat >= s) & (lat < n)).all())
        self.assertEqual(set(zip(old[ch].tolist(), new[ch].tolist())), {(2, 0)})  # B -> A1

    def test_samples(self):
        """The only samples that change sector: 7 on the outer face of W1 (OSM 814801324, B -> A1).
        Each way near the strip is sampled alone (samples and their sides do not depend on the other
        lines; alone, a side is only kept more often), plus the industrial piers, which stay B."""
        z = _site("ZDH")
        g = Grid.for_site(z)
        mask, _ = run.load_mask("ZDH", run.mask_version(z))
        lines = quay_lines.load("ZDH", z["quay_lines_version"])
        w, s, e, n = STRIP
        pad = 0.002
        ways = [x for x in lines["ways"] if x["id"] in NW_PIERS or any(
            w - pad <= lo <= e + pad and s - pad <= la <= n + pad for lo, la in x["coords"])]
        self.assertEqual({x["id"] for x in ways} & set(NW_PIERS), set(NW_PIERS))
        old_s, new_s = run.site_sectors(_old_zdh(z)), run.site_sectors(z)
        changed, nw = [], {}
        for way in ways:
            geo = quay.Geometry(g, mask, {"ways": [way]})
            for p in geo.profiles:
                for i in np.flatnonzero(p["ok"]):
                    lo, la = float(p["lon"][i]), float(p["lat"][i])
                    a, b = fsectors.sector_of(old_s, lo, la), fsectors.sector_of(new_s, lo, la)
                    if a != b:
                        changed.append((p["way_id"], p["side"], a, b))
                    if p["way_id"] in NW_PIERS:
                        nw.setdefault(p["way_id"], set()).add(b)
        self.assertEqual(changed, [(814801324, 1, "B", "A1")] * 7)
        self.assertEqual(nw, {wid: {"B"} for wid in NW_PIERS})


@unittest.skipUnless(np is not None, "numpy not installed")
class QgzLandFixTest(unittest.TestCase):
    Q1 = [[119.9806, 35.7317], [119.9823, 35.7317], [119.9828, 35.7311], [119.9817, 35.7302], [119.9806, 35.7309]]

    def test_fix_file(self):
        self.assertEqual(osm_mask.load_land_fix("QGZ", 1), [])  # v1 unchanged
        fx = {f["id"]: f for f in osm_mask.load_land_fix("QGZ", 2)}
        self.assertEqual(set(fx), {"Q1", "Q5"})
        self.assertEqual(fx["Q1"]["polygon"][:-1], self.Q1)
        self.assertTrue(fx["Q5"]["whole_regions"])
        self.assertEqual(osm_mask.load_land_fix("ZJG", 9), [])

    def test_v2_is_v1_plus_fix(self):
        q = _site("QGZ")
        g = Grid.for_site(q)
        v1, m1 = run.load_mask("QGZ", 1)
        v2, m2 = run.load_mask("QGZ", 2)
        again, rep = osm_mask.apply_land_fix(g, v1, osm_mask.load_land_fix("QGZ", 2))
        self.assertTrue((again == v2).all())
        self.assertEqual(rep, [{"id": "Q1", "pixels": 169}, {"id": "Q5", "pixels": 90}])
        self.assertEqual(m2["land_fix"], rep)
        self.assertEqual((m2["base_version"], m2["osm_base"], m2["bbox"]), ("QGZ-land-v1", m1["osm_base"], m1["bbox"]))
        self.assertIn("ODbL", m2["license"])
        d = v1 != v2
        self.assertEqual(int(d.sum()), 259)
        self.assertEqual(set(zip(v1[d].tolist(), v2[d].tolist())), {(0, 1)})  # water -> land only
        r, c = np.nonzero(d)
        lon, lat = g.to_lonlat(r + 0.5, c + 0.5)
        q1 = fsectors._in_polygon(lon, lat, self.Q1)
        self.assertEqual(int(q1.sum()), 169)
        q5 = ~q1
        self.assertTrue(((lon[q5] >= 120.0092) & (lon[q5] <= 120.0100) & (lat[q5] >= 35.7316) & (lat[q5] <= 35.7337)).all())
        # Q1 lies in sector A2, Q5 in B
        secs = run.site_sectors(q)
        self.assertEqual({fsectors.sector_of(secs, float(a), float(b)) for a, b in zip(lon[q1], lat[q1])}, {"A2"})
        self.assertEqual({fsectors.sector_of(secs, float(a), float(b)) for a, b in zip(lon[q5], lat[q5])}, {"B"})

    def test_v2_not_used_yet(self):
        """QGZ stays on v1 until the baselines (ro69, ro171) are rebuilt on v2 (another PR, PU)."""
        q = _site("QGZ")
        self.assertEqual((q["mask_version"], run.mask_tag(q)), (1, "QGZ-land-v1"))
        self.assertFalse((ROOT / "sat" / "baselines" / "QGZ-land-v2").exists())
        hits = []
        for p in list((ROOT / "sat").rglob("*")) + list((ROOT / ".github").rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".yml", ".geojson") and p.name != "QGZ-land-v2.json":
                if "QGZ-land-v2" in p.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(str(p.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_build_and_cli(self):
        g = Grid(0.0, 0.0, 0.01, 0.01, 10, 10)
        m = np.zeros((10, 10), np.uint8)
        m[:, :3] = osm_mask.LAND
        m[0:2, 8:10] = osm_mask.LAND
        m[0, 9] = osm_mask.WATER  # a one-pixel pocket
        m[5, 5] = osm_mask.STRUCTURE
        poly = [[0.003, 0.003], [0.006, 0.003], [0.006, 0.006], [0.003, 0.006]]
        out, rep = osm_mask.apply_land_fix(g, m, [{"id": "a", "polygon": poly}])
        self.assertEqual(rep, [{"id": "a", "pixels": 8}])  # 3x3 minus the structure
        self.assertEqual(int(out[5, 5]), osm_mask.STRUCTURE)
        pocket = [[0.0091, 0.0091], [0.0099, 0.0091], [0.0099, 0.0099], [0.0091, 0.0099]]
        out, rep = osm_mask.apply_land_fix(g, m, [{"id": "p", "polygon": pocket, "whole_regions": True, "max_pixels": 5}])
        self.assertEqual((rep, int(out[0, 9])), ([{"id": "p", "pixels": 1}], osm_mask.LAND))
        sea = [[0.0051, 0.0051], [0.0059, 0.0051], [0.0059, 0.0059], [0.0051, 0.0059]]
        with self.assertRaises(ValueError):  # touches the sea
            osm_mask.apply_land_fix(g, m, [{"id": "s", "polygon": sea, "whole_regions": True, "max_pixels": 5}])
        # build_mask without fixes keeps its keys (no "land_fix")
        _, info = osm_mask.build_mask(g, {"elements": []})
        self.assertNotIn("land_fix", info)
        _, info = osm_mask.build_mask(g, {"elements": []}, [{"id": "a", "polygon": poly}])
        self.assertEqual(info["land_fix"], [{"id": "a", "pixels": 9}])
        # --from-version reproduces the committed v2 without Overpass
        q = _site("QGZ")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for name in ("QGZ-land-v1.png", "QGZ-land-v1.json", "QGZ-land-fix.geojson"):
                (td / name).write_bytes((osm_mask.MASK_DIR / name).read_bytes())
            meta = osm_mask.fix_from_version(q, Grid.for_site(q), 1, 2, td)
            self.assertEqual((td / "QGZ-land-v2.png").read_bytes(), (osm_mask.MASK_DIR / "QGZ-land-v2.png").read_bytes())
            self.assertEqual(meta["land_fix"], json.loads((osm_mask.MASK_DIR / "QGZ-land-v2.json").read_text(encoding="utf-8"))["land_fix"])


if __name__ == "__main__":
    unittest.main()
