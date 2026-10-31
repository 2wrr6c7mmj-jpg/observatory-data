"""Open-sea monitoring (ocean/monitor.py, ocean-monitor-1). No network: fake OData, Process API and Overpass;
the classes are checked on the October 40 m wide results (probe/ocean-cfar-wide-ops3, 2026-10-31T03:37Z)."""
from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
import urllib.parse
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from ocean import cfar_probe as cp
    from ocean import coverage_probe as covp
    from ocean import monitor as m
    from ocean import wide
    from sat.grid import Grid
    from tests.test_ocean_modes import Fake, Pacer, overpass

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "ocean_monitor" / "wide-2026-10-31T03-37-19Z.json"
NOW = datetime(2026, 10, 30, 19, 41, tzinfo=timezone.utc)
CREDIT = "Contains modified Copernicus Sentinel data 2026"


def product(sat: str, start: datetime, rel: int, direction: str, ring, absn: int = 5000, cog: bool = True) -> dict:
    s, e = start.strftime("%Y%m%dT%H%M%S"), (start + timedelta(seconds=25)).strftime("%Y%m%dT%H%M%S")
    name = f"S1{sat}_IW_GRDH_1SDV_{s}_{e}_{absn:06d}_0ABCDE_1234" + ("_COG.SAFE" if cog else ".SAFE")
    return {"Id": name, "Name": name, "ContentDate": {"Start": start.strftime("%Y-%m-%dT%H:%M:%S.000000Z")},
            "PublicationDate": (start + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "GeoFootprint": {"type": "Polygon", "coordinates": [ring]},
            "Attributes": [{"Name": "orbitDirection", "Value": direction}, {"Name": "relativeOrbitNumber", "Value": rel}]}


def box(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def odata(products: list[dict], calls: list | None = None):
    def get(url):
        if calls is not None:
            calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["$filter"][0]
        t0 = covp.parse_time(re.search(r"ge (\S+)", q).group(1))
        t1 = covp.parse_time(re.search(r"lt (\S+)", q).group(1))
        return {"value": [p for p in products if t0 <= covp.parse_time(p["ContentDate"]["Start"]) < t1]}
    return get


def small_config(tmp: Path, **kw) -> Path:
    cfg = m.load_config()
    cfg["areas"] = [
        {"id": "T", "name": "every pass", "bbox": [124.0, 25.0, 124.4, 25.4], "priority": 1, "mode": "every_pass",
         "enabled": True, "min_revisit_days": 4, "min_eff": 0.3, "pu_month_cap": 500, "orbits": "C171"},
        {"id": "M", "name": "monthly", "bbox": [124.4, 25.0, 124.8, 25.4], "priority": 3, "mode": "monthly",
         "enabled": True, "min_eff": 0.3, "pu_month_cap": 500, "orbits": "C171"},
        {"id": "X", "name": "disabled", "bbox": [124.0, 25.4, 124.4, 25.8], "priority": 2, "mode": "every_pass",
         "enabled": False, "min_eff": 0.3, "pu_month_cap": 0},
    ]
    cfg.update(kw)
    p = tmp / "monitor.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


def run_args(tmp: Path, config: Path, **kw) -> Namespace:
    base = dict(config=config, out=tmp / "monitor", report_dir=tmp / "report", usage=tmp / "usage.csv", sat_usage=[],
                other_usage=[], ports_usage=[], ocean_usage=[], budget_month="2026-10", ocean_cap=3500.0,
                global_cap=6000.0, global_margin=450.0, max_pu=400.0, max_minutes=10, lookback_days=None, dry_run=False)
    base.update(kw)
    return Namespace(**base)


class Client(Fake):
    def __init__(self):
        super().__init__([])


@unittest.skipUnless(np is not None, "numpy not installed")
class ConfigTest(unittest.TestCase):
    def test_plan_b_areas_and_caps(self):
        cfg = m.load_config()
        on = {a["id"]: a for a in m.areas_in_order(cfg)}
        self.assertEqual(set(on), {"SENKAKU", "YNG", "MYK", "TWE_S", "SAKI_S", "TSM", "OKI", "BASHI"})
        self.assertEqual({k for k, a in on.items() if a["mode"] == "monthly"}, {"OKI"})
        caps = {k: a["pu_month_cap"] for k, a in on.items()}
        self.assertEqual(caps, {"SENKAKU": 70, "YNG": 360, "MYK": 460, "TWE_S": 490, "SAKI_S": 400, "TSM": 480, "OKI": 760,
                                "BASHI": 300})
        self.assertEqual({k for k, a in on.items() if a.get("min_revisit_days") == 4}, {"YNG", "TWE_S", "TSM"})
        self.assertEqual([a["id"] for a in m.areas_in_order(cfg)][:3], ["SENKAKU", "YNG", "MYK"])
        off = {a["id"] for a in cfg["areas"] if not a.get("enabled")}
        self.assertEqual(off, {"HKS", "ZSN"})
        self.assertEqual((on["BASHI"]["priority"], on["BASHI"]["mode"]), (2, "every_pass"))
        self.assertEqual(on["TWE_S"].get("min_area_share", cfg["min_area_share"]), 0.3)
        self.assertLessEqual(sum(caps.values()), 3500)  # the areas' caps fit in OCEAN_CAP of plan B
        self.assertEqual(cfg["pixel_m"], 40)

    def test_priority_area_sea_is_not_counted_twice(self):
        cfg = m.load_config()
        cells = m.area_cells(cfg, m.areas_in_order(cfg))
        # MYK (priority 1) and OKI (3) overlap at 127.0-127.2E 25.6-26.2N: that sea is MYK's
        self.assertTrue(cells["MYK"].contains(127.1, 25.9))
        self.assertFalse(cells["OKI"].contains(127.1, 25.9))
        self.assertTrue(cells["OKI"].contains(127.3, 25.9))
        # TWE_S (listed before BASHI) keeps the overlap 121.5-122.0E 21.8-22.0N
        self.assertTrue(cells["TWE_S"].contains(121.7, 21.9))
        self.assertFalse(cells["BASHI"].contains(121.7, 21.9))
        self.assertTrue(cells["BASHI"].contains(121.0, 21.0))
        # Taiwan's west coast (Changhua, Yunlin) is land in the planning cells of TSM
        self.assertLess(cells["TSM"].sea_all[cells["TSM"].cell_of(120.35, 23.45)], 0.5)


@unittest.skipUnless(np is not None, "numpy not installed")
class CatalogueTest(unittest.TestCase):
    def test_odata_products_become_acquisitions(self):
        t = datetime(2026, 10, 27, 9, 52, 8, tzinfo=timezone.utc)
        prods = [product("C", t, 171, "ASCENDING", box(122, 23, 125, 25)),
                 product("C", t, 171, "ASCENDING", box(122, 23, 125, 25), cog=False),     # the same product (not COG)
                 product("C", t + timedelta(seconds=25), 171, "ASCENDING", box(122, 24.8, 125, 26.8)),  # next slice
                 product("D", t + timedelta(hours=11), 105, "DESCENDING", box(121, 21, 122.5, 25))]
        calls = []
        acqs, by_id, n = m.catalogue([121, 21, 126, 27], t - timedelta(days=1), t + timedelta(days=1),
                                     get=odata(prods, calls), log=lambda x: None, sleep=lambda x: None)
        self.assertEqual(len(acqs), 2)
        self.assertEqual(acqs[0]["platform"], "sentinel-1c")
        self.assertEqual(acqs[0]["relative_orbit"], 171)
        self.assertEqual(acqs[0]["orbit_direction"], "ascending")
        self.assertEqual(len(acqs[0]["ids"]), 2)
        self.assertEqual(acqs[1]["orbit_direction"], "descending")
        self.assertEqual(acqs[0]["published_utc"], "2026-10-27T11:52:33Z")
        self.assertTrue(all("catalogue.dataspace.copernicus.eu/odata" in u for u in calls))
        self.assertTrue(all("_IW_GRDH_1SDV_" in urllib.parse.unquote(u) for u in calls))


@unittest.skipUnless(np is not None, "numpy not installed")
class DecideTest(unittest.TestCase):
    def setUp(self):
        self.cfg = m.load_config()
        self.area = {"id": "T", "bbox": [124.0, 25.0, 124.4, 25.4], "mode": "every_pass", "min_revisit_days": 4}
        self.cells = m.AreaCells(self.area)
        self.acq = {"acquired_utc": "2026-10-20T09:52:08Z"}

    def test_low_share_and_revisit(self):
        full = np.ones((self.cells.ny, self.cells.nx))
        part = np.zeros_like(full)
        part[:, :1] = 1.0   # a quarter of the area
        self.assertEqual(m.decide(self.area, self.cfg, self.cells, self.acq, part, [])[0], "low_share")
        self.assertEqual(m.decide(self.area, self.cfg, self.cells, self.acq, full, [])[0], "take")
        hist = [{"acquired_utc": "2026-10-17T21:51:00Z", "status": "ok"}]
        self.assertEqual(m.decide(self.area, self.cfg, self.cells, self.acq, full, hist)[0], "revisit")
        hist = [{"acquired_utc": "2026-10-15T21:51:00Z", "status": "ok"}]
        self.assertEqual(m.decide(self.area, self.cfg, self.cells, self.acq, full, hist)[0], "take")
        hist = [{"acquired_utc": "2026-10-18T21:51:00Z", "status": "skipped_revisit"}]
        self.assertEqual(m.decide(self.area, self.cfg, self.cells, self.acq, full, hist)[0], "take")

    def test_monthly_takes_unseen_cells_only(self):
        area = {**self.area, "mode": "monthly"}
        full = np.ones((self.cells.ny, self.cells.nx))
        seen = [[124.0 + 0.1 * i, 25.0 + 0.1 * j, 1.0] for j in range(4) for i in range(4)]
        hist = [{"acquired_utc": "2026-10-05T09:28:00Z", "status": "ok", "observed_cells": seen}]
        self.assertEqual(m.decide(area, self.cfg, self.cells, self.acq, full, hist)[0], "seen_this_month")
        v, cov0, _ = m.decide(area, self.cfg, self.cells, self.acq, full, hist[:0])
        self.assertEqual(v, "take")
        old = [{**hist[0], "acquired_utc": "2026-09-28T09:28:00Z"}]  # last month's look does not count
        self.assertEqual(m.decide(area, self.cfg, self.cells, self.acq, full, old)[0], "take")
        half = [{**hist[0], "observed_cells": seen[:8]}]
        v, cov0, _ = m.decide(area, self.cfg, self.cells, self.acq, full, half)
        self.assertEqual(v, "take")
        plan = m.plan_rects(self.cells, {**area, "min_eff": 0.3}, full, cov0, 1e9)
        self.assertTrue(plan)
        for p in plan:
            self.assertGreaterEqual(p["bbox"][1], 25.2 - 1e-9)  # only the northern (unseen) half is fetched


@unittest.skipUnless(np is not None, "numpy not installed")
class OctoberClassesTest(unittest.TestCase):
    """The classes and flags on the 710 cfar-v1 candidates of the October wide run (40 m)."""

    def setUp(self):
        self.cfg = m.load_config()
        fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.results = fx["results"]
        self.cands = [dict(zip(fx["fields"], r)) for r in fx["candidates"]]
        areas = m.areas_in_order(self.cfg)
        cells = m.area_cells(self.cfg, areas)
        for c in self.cands:
            r = next((r for r in self.results if r["acquired_utc"] == c["acquired_utc"]
                      and r["bbox"][0] <= c["lon"] <= r["bbox"][2] and r["bbox"][1] <= c["lat"] <= r["bbox"][3]), None)
            c["sea_state"] = m.sea_state(r["vv_median_db"], self.cfg) if r else None
            c["area"] = next((a["id"] for a in areas if cells[a["id"]].contains(c["lon"], c["lat"])), None)
            c["ambiguity_suspect"] = False
        self.by = {}
        for c in self.cands:
            self.by.setdefault((c["acquired_utc"], c["area"]), []).append(c)
        for v in self.by.values():
            m.clusters([c for c in v if not (c["near_coast"] or c["static"])], 500, 5)
            for c in v:
                c.setdefault("cluster", None)

    def count(self, key, cls):
        return sum(1 for c in self.by[key] if m.classify(c)[0] == cls)

    def test_strong_wind_pass_is_not_counted(self):
        # 2026-10-22 descending C105 off eastern Taiwan: VV median -7.75 dB (v1 141, v2 5 in the design)
        key = ("2026-10-22T21:51:23Z", "TWE_S")
        self.assertEqual(self.count(key, "A"), 0)
        self.assertEqual(sum(1 for c in self.by[key] if m.classify(c) == ("ref", "sea_state")), 57)

    def test_headline_needs_v1_high_and_v2_A(self):
        key = ("2026-10-15T09:52:08Z", "YNG")
        self.assertEqual(self.count(key, "A"), 21)
        for c in self.by[key]:
            if m.classify(c)[0] == "A":
                self.assertEqual((c["confidence"], c["v2_confidence"]), ("high", "A"))
                self.assertFalse(c["near_coast"] or c["static"] or c["edge"])
        self.assertEqual(self.count(("2026-10-15T09:52:08Z", "SENKAKU"), "A"), 3)

    def test_coast_and_low_are_reference_only(self):
        n = sum(1 for c in self.cands if c["near_coast"])
        self.assertEqual(sum(1 for c in self.cands if m.classify(c) == ("ref", "near_coast")), n)
        for c in self.cands:
            if c["confidence"] == "low":
                self.assertEqual(m.classify(c)[0], "ref")

    def test_flags_take_a_candidate_out_of_the_headline(self):
        c = next(c for c in self.by[("2026-10-15T09:52:08Z", "YNG")] if m.classify(c)[0] == "A")
        for k, v in (("ambiguity_suspect", True), ("cluster", 1), ("edge", True)):
            self.assertEqual(m.classify({**c, k: v}), ("B", None))
        for k, v in (("static_learned", True), ("sea_state", "low"), ("sea_state", "high")):
            self.assertEqual(m.classify({**c, k: v})[0], "ref")


@unittest.skipUnless(np is not None, "numpy not installed")
class FlagsTest(unittest.TestCase):
    def test_clusters(self):
        base = (122.9, 29.6)
        pts = [{"lon": base[0] + 0.002 * k, "lat": base[1]} for k in range(6)]   # about 200 m apart, linked
        pts += [{"lon": 123.3, "lat": 29.9}, {"lon": 123.3, "lat": 29.9035}]       # a pair 390 m apart
        self.assertEqual(m.clusters(pts, 500, 5), 1)
        self.assertEqual([p["cluster"] for p in pts], [1] * 6 + [None, None])
        self.assertEqual(m.clusters(pts[:4], 500, 5), 0)

    def test_sea_state(self):
        cfg = m.load_config()
        self.assertEqual(m.sea_state(-33.0, cfg), "low")     # 10-19 C54
        self.assertEqual(m.sea_state(-7.95, cfg), "high")    # 10-22 C105
        self.assertEqual(m.sea_state(-16.4, cfg), "ok")
        self.assertIsNone(m.sea_state(float("nan"), cfg))

    def test_cell_stats(self):
        vv = np.full((40, 40), -18.0)
        vv[:20, :] = -30.0                           # northern half dark
        valid = np.ones_like(vv, bool)
        valid[:, 30:] = False
        share, med = m.cell_stats(vv, valid, valid, 2, 2)
        self.assertEqual(med[1, 0], -30.0)             # row 1 = north
        self.assertEqual(med[0, 0], -18.0)
        self.assertAlmostEqual(share[0, 1], 0.5)

    def test_ambiguity_from_a_bright_land_target(self):
        cfg = m.load_config()
        g = Grid(124.0, 25.0, 124.4, 25.4, 1000, 1000)          # about 40 m
        vv = np.full((1000, 1000), -20.0)
        land = np.zeros(vv.shape, bool)
        heading = 348.0                                       # ascending
        c = {"lon": 124.2, "lat": 25.2, "peak_vv_db": -8.0}
        far = {"lon": 124.3, "lat": 25.1, "peak_vv_db": -8.0}
        lia = np.full(vv.shape, 35.0)
        from sat import detect2 as v2
        d = v2.ambiguity_offset_m(35.0)
        ue, un = v2._unit(heading)
        py, px = g.pixel_m
        r0, c0 = (float(x) - 0.5 for x in g.to_pixel(c["lon"], c["lat"]))
        r, k = int(round(r0 - d * un / py)), int(round(c0 + d * ue / px))
        vv[r - 1:r + 2, k - 1:k + 2] = 12.0                    # a port 20 dB brighter at the ambiguity offset
        land[r - 1:r + 2, k - 1:k + 2] = True
        m.ambiguity_check([c, far], vv, lia, land, g, heading, cfg)
        self.assertTrue(c["ambiguity_suspect"])
        self.assertEqual(c["ambiguity_parent"], "land")
        self.assertFalse(far["ambiguity_suspect"])
        weak = {"lon": 124.2, "lat": 25.2, "peak_vv_db": 0.0}  # only 12 dB weaker: a ship of its own
        m.ambiguity_check([weak], vv, lia, land, g, heading, cfg)
        self.assertFalse(weak["ambiguity_suspect"])

    def test_learned_fixed_object(self):
        cfg = m.load_config()
        acq = {"acquired_utc": "2026-11-20T09:52:08Z", "platform": "sentinel-1c", "relative_orbit": 171}
        rect = [[123.0, 24.0, 124.0, 25.0]]
        hist = [{"acquired_utc": t, "platform": "sentinel-1c", "relative_orbit": 171, "status": "ok", "rects": rect}
                for t in ("2026-10-15T09:52:08Z", "2026-10-27T09:52:08Z", "2026-11-08T09:52:08Z")]
        hist.append({"acquired_utc": "2026-11-10T09:44:00Z", "platform": "sentinel-1c", "relative_orbit": 98,
                     "status": "ok", "rects": rect})
        pts = {"2026-10-15T09:52:08Z": [{"lon": 123.5, "lat": 24.5}], "2026-10-27T09:52:08Z": [],
               "2026-11-08T09:52:08Z": [{"lon": 123.5005, "lat": 24.5005}], "2026-11-10T09:44:00Z": [{"lon": 123.6, "lat": 24.6}]}
        learned = {"points": []}
        c = {"lon": 123.5, "lat": 24.5}
        other = {"lon": 123.6, "lat": 24.6}   # seen by another track only: not learned
        self.assertEqual(m.learn_static([c, other], acq, hist, lambda t: pts[t], learned, cfg), 1)
        self.assertTrue(c["static_learned"])
        self.assertNotIn("static_learned", other)
        self.assertEqual(learned["points"][0]["hits"], 3)
        self.assertEqual(learned["points"][0]["relative_orbit"], 171)


@unittest.skipUnless(np is not None, "numpy not installed")
class RunTest(unittest.TestCase):
    def setUp(self):
        self._saved = (wide.PIXEL_M, wide.MIN_LENGTH_M)
        wide.PIXEL_M = 400.0   # 0.1 deg = about 28 px: small images

    def tearDown(self):
        wide.PIXEL_M, wide.MIN_LENGTH_M = self._saved

    def go(self, tmp, prods, client=None, config=None, **kw):
        client = client or Client()
        a = run_args(tmp, config or small_config(tmp), **kw)
        s = m.run(a, client=client, odata_get=odata(prods), overpass=overpass, now=kw.pop("now", NOW), pacer=Pacer(),
                  sleep=lambda x: None, log=lambda x: None, clock=lambda: NOW)
        return s, client

    def test_run_publishes_and_skips_processed_passes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
            prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
            s, c = self.go(tmp, prods)
            self.assertGreater(len(c.calls), 0)
            out = tmp / "monitor"
            for f in ("meta.json", "areas.geojson", "passes.json", "series.csv", "series.json", "static_learned.json",
                      "candidates/2026-10-30.geojson", "coverage/2026-10-30.geojson", "osm/T.json", "osm/M.json"):
                self.assertTrue((out / f).exists(), f)
            passes = json.loads((out / "passes.json").read_text(encoding="utf-8"))["passes"]
            self.assertEqual({p["area"] for p in passes}, {"T", "M"})   # X is disabled
            for p in passes:
                self.assertEqual(p["status"], "ok")
                self.assertGreaterEqual(p["observed_share"], 0.9)
                self.assertIsInstance(p["counts"]["A"], int)
                self.assertIsNone(p["counts"]["A_ge120_confirmed"])
                self.assertEqual(p["platform"], "sentinel-1c")
                self.assertEqual(p["hours_to_processed"], 9.8)
            for f in ("meta.json", "passes.json", "series.json", "candidates/2026-10-30.geojson", "static_learned.json"):
                self.assertIn(CREDIT, json.loads((out / f).read_text(encoding="utf-8"))["attribution"], f)
            self.assertIn("OpenStreetMap", json.loads((out / "meta.json").read_text(encoding="utf-8"))["attribution"])
            self.assertIn("ODbL", json.loads((out / "osm/T.json").read_text(encoding="utf-8"))["license"])
            self.assertIn(CREDIT, (out / "series.csv").read_text(encoding="utf-8"))
            cands = json.loads((out / "candidates/2026-10-30.geojson").read_text(encoding="utf-8"))["features"]
            for f in cands:
                self.assertIn(f["properties"]["class"], ("A", "B"))
                self.assertFalse(f["properties"]["chip_checked"])
                self.assertEqual(f["properties"]["ais"], "unmatched")
            usage = list(csv.DictReader((tmp / "usage.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(usage), 1)
            self.assertEqual(usage[0]["probe_version"], "ocean-monitor-1")
            self.assertAlmostEqual(float(usage[0]["processing_units"]), c.processing_units, places=2)
            # the next day: the same pass is not fetched again, no usage row is added
            s2, c2 = self.go(tmp, prods)
            self.assertEqual(c2.calls, [])
            self.assertEqual(len(list(csv.DictReader((tmp / "usage.csv").read_text(encoding="utf-8").splitlines()))), 1)
            self.assertEqual(len(json.loads((out / "passes.json").read_text(encoding="utf-8"))["passes"]), 2)

    def test_revisit_and_monthly_rules(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            t = datetime(2026, 10, 28, 9, 52, 8, tzinfo=timezone.utc)
            prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5)),
                     product("D", t + timedelta(days=1, hours=12), 105, "DESCENDING", box(123.5, 24.5, 125.5, 26.5))]
            s, c = self.go(tmp, prods)
            verdicts = {(t["area"], t["acquired_utc"][:10]): t["verdict"] for t in s["tasks"]}
            self.assertEqual(verdicts[("T", "2026-10-29")], "revisit")          # 1.5 days after the first look
            self.assertEqual(verdicts[("M", "2026-10-29")], "seen_this_month")  # monthly: seen on 10-28
            passes = json.loads((tmp / "monitor" / "passes.json").read_text(encoding="utf-8"))["passes"]
            st = {(p["area"], p["acquired_utc"][:10]): p["status"] for p in passes}
            self.assertEqual(st[("T", "2026-10-29")], "skipped_revisit")
            self.assertNotIn(("M", "2026-10-29"), st)
            series = (tmp / "monitor" / "series.csv").read_text(encoding="utf-8")
            self.assertNotIn("skipped_revisit", series)

    def test_null_when_too_little_was_seen(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
            prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 124.25, 26.5))]  # 60% of T
            cfgp = small_config(tmp, count_min_share=0.9)
            self.go(tmp, prods, config=cfgp)
            p = next(p for p in json.loads((tmp / "monitor" / "passes.json").read_text(encoding="utf-8"))["passes"]
                     if p["area"] == "T")
            self.assertLess(p["observed_share"], 0.9)
            self.assertIsNone(p["counts"]["A"])
            self.assertIsNone(p["counts"]["density_A_per_1000km2"])
            row = next(r for r in csv.DictReader((tmp / "monitor" / "series.csv").read_text(encoding="utf-8").splitlines()) if r["area"] == "T")
            self.assertEqual(row["count_A"], "")

    def test_caps(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
            prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
            s, c = self.go(tmp, prods, max_pu=0.4)
            self.assertEqual(c.calls, [])
            self.assertTrue(all(x["verdict"] == "budget" for x in s["tasks"]))
            self.assertFalse((tmp / "usage.csv").exists())
            # a pass that hit the cap is tried again by the next run
            s2, c2 = self.go(tmp, prods)
            self.assertGreater(len(c2.calls), 0)
            # the area's month cap
            tmp2 = tmp / "b"
            tmp2.mkdir()
            cfgp = small_config(tmp2)
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            for a in cfg["areas"]:
                a["pu_month_cap"] = 0.5
            cfgp.write_text(json.dumps(cfg), encoding="utf-8")
            s3, c3 = self.go(tmp2, prods, config=cfgp)
            self.assertEqual(c3.calls, [])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            t = datetime(2026, 10, 30, 9, 52, 8, tzinfo=timezone.utc)
            prods = [product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5))]
            s, c = self.go(tmp, prods, dry_run=True)
            self.assertEqual(c.calls, [])
            self.assertEqual(c.requests, 0)
            self.assertFalse((tmp / "monitor").exists())
            self.assertFalse((tmp / "usage.csv").exists())
            self.assertTrue(all(x["verdict"] == "planned" for x in s["tasks"]))
            self.assertGreater(s["planned_pu"], 0)
            self.assertIn("dry-run", (tmp / "report" / "report.txt").read_text(encoding="utf-8"))

    def test_other_month_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            with self.assertRaises(cp.MonthStop):
                self.go(tmp, [], budget_month="2026-11")

    def test_no_new_pass_no_output_change(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            s, c = self.go(tmp, [])
            self.assertEqual(c.requests, 0)
            self.assertFalse((tmp / "usage.csv").exists())
            self.assertTrue((tmp / "monitor" / "meta.json").exists())   # first run: the static files
            before = (tmp / "monitor" / "meta.json").read_text(encoding="utf-8")
            self.go(tmp, [], now=NOW + timedelta(hours=1))
            self.assertEqual((tmp / "monitor" / "meta.json").read_text(encoding="utf-8"), before)


    def test_meta_notes_the_cover_of_east_of_taiwan(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            m.write_meta(out, m.load_config(), [], {"points": []}, NOW)
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            twe = next(a for a in meta["areas"] if a["id"] == "TWE_S")
            self.assertIn("D69・C/D105 の撮影が域の約3割未満しか覆わないため、月の観測は約1.8回", twe["note"])
            self.assertIn("BASHI", [a["id"] for a in meta["areas"]])

@unittest.skipUnless(np is not None, "numpy not installed")
class PlanMonthTest(unittest.TestCase):
    def test_plan_month_counts_passes_and_pu(self):
        cfg = json.loads(small_config(Path(tempfile.mkdtemp())).read_text(encoding="utf-8"))
        prods = []
        for k in range(3):
            t = datetime(2026, 10, 3, 9, 52, 8, tzinfo=timezone.utc) + timedelta(days=12 * k)
            prods.append(product("C", t, 171, "ASCENDING", box(123.5, 24.5, 125.5, 26.5)))
            prods.append(product("D", t + timedelta(days=2), 105, "DESCENDING", box(123.5, 24.5, 124.2, 26.5)))
        prods.append(product("C", datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc), 171, "ASCENDING", box(123, 24, 126, 27)))
        res = m.plan_month(cfg, "2026-10", get=odata(prods), log=lambda x: None, sleep=lambda x: None)
        t = res["areas"]["T"]
        self.assertEqual(t["passes_touching"], 6)
        self.assertEqual(t["taken"], 3)               # C171 every 12 days; D105 covers half and 2 days later
        self.assertEqual(t["skipped_revisit"], 3)
        self.assertGreater(t["est_pu_month"], 0)
        mo = res["areas"]["M"]
        self.assertEqual(mo["taken"], 1)                # monthly: once
        self.assertIn("X", res["areas"])               # disabled areas are planned too
        self.assertFalse(res["areas"]["X"]["enabled"])
        self.assertEqual(res["total_enabled"]["passes_taken"], 4)
        self.assertIn("PU 0", m.plan_report(res))


if __name__ == "__main__":
    unittest.main()
