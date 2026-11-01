"""全港の est-1・est-2（2026-11-01、ops3/est-all-ports）：4 港（SYL・SLP・QGZ・ZDH）は 2部の調査の重み、
残り 8 港は default-by-role（確度 4）、est-2 は湛江の較正を流用して calib_source を出す、入射角の範囲外は extrapolated。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

ROOT = Path(__file__).resolve().parents[1]
SAT = ROOT / "sat"
CLASSES = json.loads((SAT / "ship_classes.json").read_text(encoding="utf-8"))
ZONES = json.loads((SAT / "quay_zones.json").read_text(encoding="utf-8"))
SITES = {s["id"]: s for s in json.loads((SAT / "sites.json").read_text(encoding="utf-8"))["sites"]}

RESEARCHED = {"SYL": "south", "SLP": "south", "QGZ": "north", "ZDH": "east"}
DEFAULTS = {"QJG": "north", "QDH": "north", "LSN": "north", "XSN": "east", "SWS": "east", "XMN": "east",
            "STU": "east", "MAW": "east"}

if np is not None:
    from sat import quay_est, quay_est2


class ResearchedWeightsTest(unittest.TestCase):
    """調査_4港の艦種の重みと区画（SYL・SLP・QGZ・ZDH）§5.1〜5.3 の値（PR-1c・PR-3a で入れたもの）が崩れていない。"""

    def test_weights_have_own_basis(self):
        for sid, fleet in RESEARCHED.items():
            st = CLASSES["sites"][sid]
            self.assertEqual(st["fleet"], fleet, sid)
            self.assertEqual(st["status"], "provisional", sid)
            self.assertTrue(st["sources"] and all(x["url"].startswith("https://") for x in st["sources"]), sid)
            self.assertEqual(set(st["w"]), set(st["w_detail"]), sid)
            self.assertNotEqual(st["status"], "default-by-role")
            self.assertNotIn("conf_level", st, sid)

    def test_key_weights(self):
        w = {k: CLASSES["sites"][k]["w"] for k in RESEARCHED}
        self.assertEqual((w["SYL"]["039AB"], w["SYL"]["TUG"]), (1.0, 0.8))
        self.assertEqual((w["QGZ"]["001"], w["QGZ"]["055"]), (1.0, 0.9))
        self.assertEqual((w["ZDH"]["956"], w["ZDH"]["052C"]), (0.7, 0.8))
        self.assertEqual(w["SLP"]["094"] >= 0, True)
        for sid in ("QGZ", "ZDH"):
            self.assertEqual(CLASSES["sites"][sid]["version"], "est-1b-202610")

    def test_classes_exist_and_fleets_cover(self):
        ids = {c["id"] for c in CLASSES["classes"]}
        for need in ("001", "003", "956", "051C"):
            self.assertIn(need, ids)
        for f in ("south", "north", "east"):
            self.assertIn(f, CLASSES["fleets"])
        self.assertIn("001", CLASSES["fleets"]["north"])
        self.assertIn("956", CLASSES["fleets"]["east"])

    def test_pier_types(self):
        for t in ("carrier", "submarine", "large", "medium", "shore"):
            self.assertIn(t, ZONES["types"])
        self.assertEqual(ZONES["types"]["carrier"]["mult"]["X"], 5.0)
        self.assertEqual({ZONES["types"]["carrier"]["mult"][g] for g in "LMST"}, {0.3})
        self.assertEqual(ZONES["types"]["submarine"]["mult_kind"], {"submarine": 3.0})
        self.assertEqual(ZONES["types"]["submarine"]["mult_other"], 0.3)
        for sid in RESEARCHED:
            self.assertIn(sid, ZONES["sites"])
            for z in ZONES["sites"][sid]["zones"]:
                self.assertIn(z["type"], ZONES["types"], (sid, z["id"]))
        types = {z["type"] for sid in ("SYL", "SLP", "QGZ") for z in ZONES["sites"][sid]["zones"]}
        self.assertTrue({"carrier", "submarine"} <= types)


class DefaultByRoleTest(unittest.TestCase):
    def test_eight_ports(self):
        theatre = {k: v for k, v in (RESEARCHED | DEFAULTS).items() if k not in ("SYL", "SLP")}   # the 10 north / east ports
        n = {f: sum(1 for v in theatre.values() if v == f) for f in ("north", "east")}
        self.assertEqual(n, {"north": 4, "east": 6})
        kinds = {c["id"]: c.get("kind") for c in CLASSES["classes"]}
        for sid, fleet in DEFAULTS.items():
            st = CLASSES["sites"][sid]
            self.assertEqual((st["fleet"], st["status"], st["conf_level"], st["version"]),
                             (fleet, "default-by-role", 4, "est-1b-202611"), sid)
            self.assertEqual(st["sources"], [], sid)
            share = round(1 / n[fleet], 2)
            expect = {cid: share for cid in CLASSES["fleets"][fleet] if kinds.get(cid) != "carrier"}
            self.assertEqual(st["w"], expect, sid)
            self.assertNotIn("001", st["w"])
            self.assertEqual(SITES[sid]["quay"]["class_weights"], sid)
            for s in SITES[sid]["sectors"]:
                self.assertEqual(s["role"], "naval", sid)

    def test_zjg_entry_untouched(self):
        z = CLASSES["sites"]["ZJG"]
        self.assertNotIn("version", z)
        self.assertNotIn("conf_level", z)
        self.assertEqual(z["fleet"], "south")


@unittest.skipUnless(np is not None, "numpy not installed")
class PublishedInfoTest(unittest.TestCase):
    def test_weights_info(self):
        for sid in DEFAULTS:
            i = quay_est.site_weights_info(sid)
            self.assertTrue(i["own"], sid)
            self.assertEqual((i["status"], i["conf_level"], i["version"]), ("default-by-role", 4, "est-1b-202611"), sid)
            self.assertNotIn("001", i["classes"])
            m = quay_est.Model(site_id=sid, sector_roles={"A1": "naval"})
            self.assertEqual(m.class_site, sid)
            self.assertEqual(m.zone_of(sid, 1, "A1"), ("NV", "medium"))
            self.assertEqual(quay_est.site_class_tables()[sid]["conf_level"], 4)
        for sid in RESEARCHED:
            i = quay_est.site_weights_info(sid)
            self.assertEqual((i["own"], i["status"]), (True, "provisional"), sid)
            self.assertIsNone(i["conf_level"], sid)
        z = quay_est.site_weights_info("ZJG")
        self.assertEqual((z["status"], z["version"]), ("estimate", "est-1b"))

    def test_north_east_priors_differ(self):
        qjg = quay_est.Model(site_id="QJG", sector_roles={"A1": "naval"})
        xsn = quay_est.Model(site_id="XSN", sector_roles={"A1": "naval"})
        self.assertIn("039G", [c["id"] for c in qjg.classes])
        self.assertIn("956", [c["id"] for c in xsn.classes])
        self.assertNotIn("956", [c["id"] for c in qjg.classes])


@unittest.skipUnless(np is not None, "numpy not installed")
class Est2PerSiteTest(unittest.TestCase):
    def test_extrapolated(self):
        lo, hi = min(quay_est2.TOWARD_MU), max(quay_est2.TOWARD_MU)
        self.assertEqual((lo, hi), (32.0, 45.0))
        for inc, ext in ((None, False), (32.0, False), (40.0, False), (45.0, False), (31.9, True), (45.1, True), (20.0, True)):
            self.assertIs(quay_est2.incidence_extrapolated(inc), ext, inc)
        # out of range: stops at the end value
        self.assertEqual(quay_est2.toward_mu(20.0), quay_est2.TOWARD_MU[32.0])
        self.assertEqual(quay_est2.toward_mu(50.0), quay_est2.TOWARD_MU[45.0])

    def test_site_median(self):
        rows = [{"site_id": "XMN", "relative_orbit": "40", "incidence_deg": v} for v in ("30.0", "34.0", "50.0")]
        rows += [{"site_id": "XMN", "relative_orbit": "142", "incidence_deg": ""}]
        med = quay_est2.site_incidence_medians(rows)
        self.assertEqual(med, {("XMN", "40"): 34.0})

    def test_build_outputs_calib_source_and_extrapolated(self):
        sites = [{**SITES[s], "layout": "per_site"} for s in ("XMN", "QJG")]
        qrows = [
            {"site_id": "XMN", "acquired_utc": "2026-11-01T10:00:00Z", "relative_orbit": "40", "mask_version": "XMN-land-v1"},
            {"site_id": "XMN", "acquired_utc": "2026-11-02T10:00:00Z", "relative_orbit": "40", "mask_version": "XMN-land-v1"},
            {"site_id": "QJG", "acquired_utc": "2026-11-01T10:00:00Z", "relative_orbit": "171", "mask_version": "QJG-land-v1"},
        ]
        srows = [  # the second XMN scene has no angle: the site-orbit median (the first one's, 47.5, out of range)
            {"site_id": "XMN", "acquired_utc": "2026-11-01T10:00:00Z", "relative_orbit": "40", "incidence_deg": "47.5"},
            {"site_id": "QJG", "acquired_utc": "2026-11-01T10:00:00Z", "relative_orbit": "171", "incidence_deg": "38.0"},
        ]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            quay_est2.build(out, qrows, srows, pairs={}, sites=sites)
            self.assertFalse((out / "quay_est2.json").exists())   # per-site only: Zhanjiang's shared files are not written
            x = json.loads((out / "quay_est2" / "XMN.json").read_text(encoding="utf-8"))
            self.assertEqual(x["calib_source"], "ZJG" if quay_est2._P else "placeholder")
            self.assertIn("湛江の較正を流用", x["calib_note"])
            self.assertEqual(x["weights"]["status"], "default-by-role")
            self.assertEqual((x["weights"]["conf_level"], x["weights"]["version"]), (4, "est-1b-202611"))
            s1, s2 = x["scenes"]
            self.assertEqual((s1["incidence_deg"], s1["incidence_source"], s1["incidence_extrapolated"]), (47.5, "scene", True))
            self.assertEqual((s2["incidence_deg"], s2["incidence_source"], s2["incidence_extrapolated"]),
                             (47.5, "site_orbit_median", True))
            self.assertEqual(s1["calib_source"], x["calib_source"])
            with (out / "quay_est2" / "QJG.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual((r["weights_status"], r["weights_conf_level"], r["weights_version"]),
                             ("default-by-role", "4", "est-1b-202611"))
            self.assertEqual((r["incidence_deg"], r["incidence_extrapolated"], r["incidence_source"]), ("38.0", "False", "scene"))
            self.assertEqual(r["calib_source"], "ZJG" if quay_est2._P else "placeholder")


if __name__ == "__main__":
    unittest.main()
