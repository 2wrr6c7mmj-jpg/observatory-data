"""Multi-site PR-1: Zhanjiang's files stay byte for byte the same, N sectors (bbox / polygon, role),
per-site files, est-1 weights per site, est-2 falls back to Zhanjiang's calibration, priority / caps."""
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, cdse, png, quay_est, quay_est2, quay_lines, run, s2
    from sat import sectors as fsectors
    from sat.grid import Grid
    from tests import sat_zjg_golden
    from tests.test_sat_quay import FakeClient, scene_parts, ships

@unittest.skipUnless(np is not None, "numpy not installed")
class ZhanjiangUnchangedTest(unittest.TestCase):
    """The code before PR-1 (9e86565) and this code, on the same scenario and the same machine
    (tests/sat_zjg_golden.py): Zhanjiang's files must be byte for byte the same."""

    def test_every_zjg_file_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = sat_zjg_golden.compare(Path(d))
        self.assertEqual((old["rc"], new["rc"]), (0, 0))
        # the scenario reaches the quay count, est-1 and est-2 in both
        self.assertEqual(old["quay_results"], ["counted", "counted"])
        self.assertEqual(new["quay_results"], ["counted", "counted"])
        self.assertGreater(len(old["files"]), 20)
        changed = sorted(k for k in old["files"] if new["files"].get(k) != old["files"][k])
        if changed:  # visible in the Actions annotations
            print("::error::Zhanjiang outputs differ from 9e86565: " + ", ".join(changed))
        self.assertEqual(changed, [], "湛江の出力が変わった")
        self.assertEqual(sorted(set(new["files"]) - set(old["files"])), sorted(sat_zjg_golden.NEW_FILES))
        # meta.json: Zhanjiang's entry is the same but for the new role of each sector
        for sc in new["meta"]["site"]["sectors"]:
            sc.pop("role")
        self.assertEqual(new["meta"], old["meta"])
        # quay_sectors.csv: the old scenes from the wide columns, the two new scenes from quay-v1
        sectors = new["sectors"]
        fresh = [r for r in sectors if r["acquired_utc"] >= "2026-11"]
        self.assertEqual(sorted({(r["sector_id"], r["role"]) for r in fresh}), [("A", "naval"), ("B", "other")])
        self.assertEqual({r["source"] for r in fresh}, {"quay-v1"})
        self.assertEqual({r["est2_calib_source"] for r in sectors}, {"self"})
        past = [r for r in sectors if r["acquired_utc"] < "2026-11"]
        self.assertTrue(past and {r["source"] for r in past} == {"quay_counts（横持ちの列から）"})


@unittest.skipUnless(np is not None, "numpy not installed")
class SectorsTest(unittest.TestCase):
    SITE = {"id": "TSC", "bbox": [110.0, 21.0, 110.01, 21.01], "pixel_m": 10,
            "sectors": [{"id": "A1", "role": "naval", "polygon": [[110.0, 21.005], [110.005, 21.01], [110.0, 21.01]]},
                        {"id": "A2", "role": "other", "bbox": [110.006, 21.0, 110.01, 21.004]}],
            "sector_rest": {"id": "R", "name": "残り"}}

    def test_polygon_bbox_and_rest(self):
        secs = fsectors.site_sectors(self.SITE)
        self.assertEqual([s["id"] for s in secs], ["A1", "A2", "R"])
        self.assertEqual([s["role"] for s in secs], ["naval", "other", "other"])
        self.assertEqual(fsectors.sector_of(secs, 110.001, 21.009), "A1")
        self.assertEqual(fsectors.sector_of(secs, 110.004, 21.006), "R")  # below the diagonal
        self.assertEqual(fsectors.sector_of(secs, 110.008, 21.002), "A2")
        r = fsectors.sector_raster(Grid.for_site(self.SITE), secs)
        self.assertEqual(set(np.unique(r)), {0, 1, 2})
        self.assertEqual(fsectors.overlaps(self.SITE), [])
        self.assertEqual(fsectors.outside(self.SITE), [])

    def test_overlap_and_bad_role_are_errors(self):
        bad = json.loads(json.dumps(self.SITE))
        bad["sectors"][1]["bbox"] = [110.0, 21.006, 110.003, 21.01]
        self.assertTrue(fsectors.overlaps(bad))
        bad["sectors"][1]["role"] = "harbour"
        with self.assertRaises(ValueError):
            fsectors.site_sectors(bad)

    def test_zjg_sectors_as_before(self):
        z = next(s for s in run.load_sites() if s["id"] == "ZJG")
        secs = run.site_sectors(z)
        self.assertEqual([(s["id"], s["role"], s["bbox"]) for s in secs],
                         [("A", "naval", [110.43, 21.2, 110.455, 21.265]), ("B", "other", None)])
        self.assertEqual(run.wide_sectors(secs), secs)
        self.assertEqual(run.tuning_scenes(z), {"2026-10-02T10:48:45Z", "2026-10-03T10:41:00Z", "2026-10-08T10:49:01Z",
                                                "2026-10-14T10:48:45Z", "2026-10-15T10:41:00Z", "2026-10-26T10:48:45Z"})
        # the S2 cloud check: the naval sectors = Zhanjiang's sector A
        g = Grid.for_site(z)
        a = s2.sector_index(g, z)
        lon, lat = g.to_lonlat(np.arange(g.height)[:, None] + 0.5, np.arange(g.width)[None, :] + 0.5)
        lon, lat = np.broadcast_to(lon, a.shape), np.broadcast_to(lat, a.shape)
        b = [110.43, 21.2, 110.455, 21.265]
        self.assertTrue(np.array_equal(a, (lon >= b[0]) & (lon < b[2]) & (lat >= b[1]) & (lat < b[3])))

    def test_sites_json_is_consistent(self):
        sites = run.load_sites()
        self.assertEqual(sites[0]["id"], "ZJG")  # Zhanjiang stays first (processed first, as before)
        for s in sites:
            fsectors.site_sectors(s)  # raises on a bad role / id
            self.assertEqual(fsectors.overlaps(s), [], s["id"])
            self.assertEqual(fsectors.outside(s), [], s["id"])
            self.assertIn(s.get("publish", "trial"), ("trial", "public"))
            self.assertIn(s.get("kind", "naval"), ("naval", "yard", "island"))
            if s["id"] != "ZJG":  # new ports: trial page only, own folders, no PNG
                self.assertEqual(s["publish"], "trial")
                self.assertEqual(s["layout"], "per_site")
                self.assertFalse(s["png"])


@unittest.skipUnless(np is not None, "numpy not installed")
class ClassWeightsTest(unittest.TestCase):
    def test_zjg_weights_same_as_est1(self):
        """est-1b reproduces the flat est-1 table of Zhanjiang (count x w, same order)."""
        m0, m1 = quay_est.Model(), quay_est.Model(site_id="ZJG")
        self.assertEqual([c["id"] for c in m0.classes], [c["id"] for c in m1.classes])
        self.assertTrue(np.array_equal(m0.global_w, m1.global_w))
        old = {"055": 4, "052D": 5, "052DL": 5, "052C": 2, "054A": 8, "054A+": 7, "054B": 1, "072A": 5 * 0.8,
               "815A": 1, "056A": 20 * 0.6, "039AB": 5 * 0.2, "MCM": 6 * 0.6, "071": 5, "075": 2 * 0.4,
               "901": 1 * 0.3, "903": 2, "022": 20 * 0.1, "TUG": 15 * 0.6}
        w = np.array(list(old.values()))
        self.assertEqual([c["id"] for c in m0.classes], list(old))
        self.assertTrue(np.allclose(m0.global_w, w / w.sum()))
        self.assertEqual(quay_est.Model(site_id="XXX").class_site, "ZJG")  # no weights of its own: Zhanjiang's

    def test_role_fallback(self):
        # a site not in quay_zones.json: the zone from the sector's role (never silently shore for a naval sector)
        r = quay_est.Model(site_id="XXX", sector_roles={"N": "naval", "O": "other"})
        self.assertEqual(r.zone_of("XXX", 1, "N"), ("NV", "medium"))
        self.assertEqual(r.zone_of("XXX", 1, "O"), ("SH", "shore"))
        self.assertEqual(set(r.zone_ids("XXX")), {"SH", "NV"})
        self.assertEqual(quay_est.Model().zone_of("XXX", 1, "A"), ("SH", "shore"))  # no roles: as before

    def test_kind_multiplier(self):
        """A zone type with mult_kind (e.g. submarine) weights the classes of that kind."""
        zones = quay_est.load_zones()
        zones = {**zones, "types": {**zones["types"], "subtest": {"mult": {g: 1.0 for g in quay_est.GROUPS},
                                                                  "mult_kind": {"submarine": 3.0}, "mult_other": 0.3}}}
        m = quay_est.Model(zones=zones)
        sub = np.array([k == "submarine" for k in m.kinds])
        self.assertTrue(sub.any())
        self.assertGreater(m.zone_weights("subtest")[sub].sum(), m.zone_weights("large")[sub].sum())


@unittest.skipUnless(np is not None, "numpy not installed")
class Est2FallbackTest(unittest.TestCase):
    def test_calibration_read_order(self):
        self.assertEqual(quay_est2.calib_source("ZJG"), "self" if quay_est2._P else "placeholder")
        self.assertEqual(quay_est2.calib_source("SYL"), "ZJG" if quay_est2._P else "placeholder")
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(quay_est2, "CALIB_DIR", Path(d)):
                (Path(d) / "SYL-est2.json").write_text("{}")
                self.assertEqual(quay_est2.calib_source("SYL"), "self_pending")

    def test_place_from_role(self):
        self.assertEqual(quay_est2.place_of("A", "pier"), "A_pier")
        self.assertEqual(quay_est2.place_of("B", "pier"), "B")
        self.assertEqual(quay_est2.place_of("A2", "coastline", "naval"), "A_shore")
        self.assertEqual(quay_est2.place_of("A1", "pier", "other"), "B")


NEW = {"id": "TNS", "name": "試験港", "kind": "naval", "bbox": [110.0, 21.0, 110.005, 21.008], "pixel_m": 10,
       "mask_version": 1, "enabled": True, "daily": True, "priority": 2, "publish": "trial", "layout": "per_site",
       "png": False, "center_lat": 21.004,
       "sectors": [{"id": "N1", "name": "北の桟橋", "role": "naval",
                    "polygon": [[110.0, 21.004], [110.005, 21.004], [110.005, 21.008], [110.0, 21.008]]},
                   {"id": "N2", "name": "中ほど", "role": "naval", "bbox": [110.0, 21.0025, 110.005, 21.004]}],
       "sector_rest": {"id": "R", "name": "残り", "role": "other"},
       "baseline": {"orbits": [157], "min_scenes": 24}}


@unittest.skipUnless(np is not None, "numpy not installed")
class NewSiteRunTest(unittest.TestCase):
    def setup_dirs(self, d: Path, site: dict):
        g, mask, lines, cvv, cvh, base = scene_parts()
        assert (Grid.for_site(site).height, Grid.for_site(site).width) == mask.shape
        lines_dir, bl_dir = d / "masks", d / "baselines"
        lines_dir.mkdir()
        quay_lines.path(site["id"], 1, lines_dir).write_text(json.dumps({"version": f"{site['id']}-quay-v1", **lines}),
                                                              encoding="utf-8")
        layers = {"vv_p10": base["vv_p10"], "vv_p90": base["vv_p10"] + 2, "vh_p10": base["vh_p10"],
                  "n_valid": np.full(mask.shape, 30)}
        scenes = [{"acquired_utc": f"2026-09-{k + 1:02d}T10:48:45Z", "used": True} for k in range(30)]
        baseline.save(site, g, 157, layers, scenes, baseline.site_config(site), mask, bl_dir, "2026-10-30T00:00:00Z")
        vv, vh = ships(cvv, cvh)
        enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 4500.0), np.ones(vv.shape)], -1).astype(np.uint16)
        return mask, png.encode(arr, depth=16, filters=[1]), lines_dir, bl_dir

    def test_new_site_end_to_end(self):
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = self.setup_dirs(d, NEW)
            out = d / "out"
            store = run.Store(out, [NEW])
            cli = FakeClient(chip)
            s = run.process_site(cli, store, NEW, mask, now, None, 100.0, 12, log=lambda *a: None,
                                 lines_dir=lines_dir, baseline_dir=bl_dir)
            self.assertEqual(s["counted"], 2)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                        "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
            # per-site files, no PNG, no shared monthly files
            self.assertTrue((out / "detections" / "TNS" / "2026-10.geojson").exists())
            self.assertTrue((out / "quay" / "TNS" / "2026-10.geojson").exists())
            self.assertFalse((out / "detections" / "2026-10.geojson").exists())
            self.assertFalse((out / "quay" / "2026-10.geojson").exists())
            self.assertFalse((out / "images").exists())
            rows = run.read_csv(out / "ship_counts.csv")
            self.assertEqual({r["image"] for r in rows}, {""})
            self.assertTrue(all((out / r["color_image"]).exists() for r in rows))
            self.assertEqual({r["sector_a_harbour"] for r in rows}, {""})  # wide columns empty
            q = run.read_csv(out / "quay_counts.csv")
            self.assertEqual(len(q), 1)
            self.assertEqual((q[0]["sector_a_min"], q[0]["est_a_mode"]), ("", ""))
            # every sector in quay_sectors.csv, with est-1 and est-2 (Zhanjiang's calibration)
            sec = run.read_csv(out / "quay_sectors.csv")
            self.assertEqual([r["sector_id"] for r in sec], ["N1", "N2", "R"])
            self.assertEqual([r["role"] for r in sec], ["naval", "naval", "other"])
            tot = sum(int(r["occupied_min"]) for r in sec)
            self.assertEqual(tot, int(q[0]["occupied_min"]))
            self.assertTrue(all(r["est_mode"] != "" and r["est2_median"] != "" for r in sec))
            self.assertEqual({r["est2_calib_source"] for r in sec}, {quay_est2.calib_source("TNS")})
            # est-2 in its own files; the shared quay_est2.* are not written for a per-site-only run
            e2 = json.loads((out / "quay_est2" / "TNS.json").read_text())
            self.assertEqual(e2["calib_source"], "ZJG" if quay_est2._P else "placeholder")
            self.assertIn("湛江の較正を流用", e2["calib_note"] if quay_est2._P else "湛江の較正を流用")
            self.assertEqual(set(e2["scenes"][0]["by_sector"]), {"N1", "N2", "R"})
            self.assertFalse((out / "quay_est2.json").exists())
            # zone prior from the roles (TNS is not in quay_zones.json): the naval sectors are not shore
            zp = json.loads((out / "quay" / "TNS-zone-prior.json").read_text())
            self.assertEqual(set(zp["zones"]), {"SH", "NV"})
            # meta.json: the site's settings
            meta = json.loads((out / "meta.json").read_text())
            m = next(x for x in meta["sites"] if x["id"] == "TNS")
            self.assertEqual((m["publish"], m["layout"], m["png"]), ("trial", "per_site", False))
            self.assertNotIn("column", m["sectors"][0])
            self.assertTrue(m["quay"]["ready"] is False)  # its lines / baseline are in the test dirs only

    def test_zjg_and_new_site_share_the_store(self):
        """Zhanjiang's shared files keep only Zhanjiang; the new site writes to its folders."""
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        z = {**NEW, "id": "TZJ", "layout": "shared", "png": True, "sectors": [{"id": "A", "bbox": [110.0, 21.004, 110.005, 21.008]}],
             "sector_rest": {"id": "B", "name": "南"}}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, lines_dir, bl_dir = self.setup_dirs(d, NEW)
            quay_lines.path("TZJ", 1, lines_dir).write_text(quay_lines.path("TNS", 1, lines_dir).read_text())
            g = Grid.for_site(z)
            baseline.save(z, g, *self._layers(), baseline.site_config(z), mask, bl_dir, "2026-10-30T00:00:00Z")
            out = d / "out"
            store = run.Store(out, [z, NEW])
            for site in (z, NEW):
                run.process_site(FakeClient(chip), store, site, mask, now, None, 100.0, 12, log=lambda *a: None,
                                 lines_dir=lines_dir, baseline_dir=bl_dir)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": 6, "processing_units": 4.9,
                        "pu_source": "header", "scenes": 4})
            shared = json.loads((out / "quay" / "2026-10.geojson").read_text())
            self.assertEqual({f["properties"]["site_id"] for f in shared["features"]}, {"TZJ"})
            own = json.loads((out / "quay" / "TNS" / "2026-10.geojson").read_text())
            self.assertEqual({f["properties"]["site_id"] for f in own["features"]}, {"TNS"})
            e2 = run.read_csv(out / "quay_est2.csv")
            self.assertEqual({r["site_id"] for r in e2}, {"TZJ"})
            self.assertNotEqual(e2[0]["a_median"], "")
            self.assertEqual({r["site_id"] for r in run.read_csv(out / "quay_est2" / "TNS.csv")}, {"TNS"})

    def _layers(self):
        g, mask, lines, cvv, cvh, base = scene_parts()
        layers = {"vv_p10": base["vv_p10"], "vv_p90": base["vv_p10"] + 2, "vh_p10": base["vh_p10"],
                  "n_valid": np.full(mask.shape, 30)}
        scenes = [{"acquired_utc": f"2026-09-{k + 1:02d}T10:48:45Z", "used": True} for k in range(30)]
        return 157, layers, scenes


@unittest.skipUnless(np is not None, "numpy not installed")
class MainOrderTest(unittest.TestCase):
    def _main(self, sites, out, cli):
        g, mask, *_ = scene_parts()
        with mock.patch.object(run, "load_mask", return_value=(mask, {})), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.main(["--out", str(out)], client=cli, sites=sites)

    def _chip(self):
        g, mask, lines, cvv, cvh, base = scene_parts()
        vv, vh = ships(cvv, cvh)
        enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 4500.0), np.ones(vv.shape)], -1).astype(np.uint16)
        return png.encode(arr, depth=16, filters=[1])

    def test_priority_daily_cap_and_first_runs(self):
        base = {k: v for k, v in NEW.items() if k not in ("sectors", "sector_rest")}
        sites = [{**base, "id": "P3", "priority": 3}, {**base, "id": "P1", "priority": 1},
                 {**base, "id": "OFF", "daily": False}, {**base, "id": "P2", "priority": 2},
                 {**base, "id": "CAP", "priority": 1, "pu_month_cap": 0}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            rc = self._main(sites, out, FakeClient(self._chip()))
            self.assertEqual(rc, 0)
            st = json.loads((out / "status.json").read_text())
            order = [s["site_id"] for s in st["last_run"]["sites"]]
            self.assertEqual(order, ["P1", "CAP", "P2", "P3"])          # priority, then sites.json order; OFF not fetched
            by = {s["site_id"]: s for s in st["last_run"]["sites"]}
            self.assertTrue(by["CAP"].get("deferred_site_cap"))
            self.assertEqual(by["P1"]["counted"], 2)
            self.assertEqual(by["P2"]["counted"], 2)
            self.assertTrue(by["P3"].get("deferred_first_run"))       # at most 2 first backfills per run
            self.assertEqual({s["site_id"] for s in st["scenes"]}, {"P1", "P2"})
            # per-site PU from the scene records (usage.csv stays one row per run)
            month = st["last_run"]["run_utc"][:7]
            self.assertAlmostEqual(run.site_month_pu(run.Store(out, sites), "P1", month), 2 * 1.226, places=3)
            self.assertEqual(list(run.read_csv(out / "usage.csv")[0].keys()), run.USAGE_FIELDS)


@unittest.skipUnless(np is not None, "numpy not installed")
class SanyaSitesTest(unittest.TestCase):
    """楡林 SYL・亜龍湾 SLP（PR-1 の 2 つ目のコミット）：設定・マスク・仮置きの重みと桟橋の型。"""

    def sites(self):
        return {s["id"]: s for s in run.load_sites()}

    def test_settings(self):
        s = self.sites()
        for sid, mv in (("SYL", 2), ("SLP", 1)):
            x = s[sid]
            self.assertEqual((x["kind"], x["publish"], x["layout"], x["png"], x["mask_version"]), ("naval", "trial", "per_site", False, mv))
            self.assertFalse(x["daily"])  # settings only until the quay lines and the baseline are in
            self.assertEqual(x["baseline"], {"orbits": [157], "lookback_days": 365, "max_scenes": 30, "min_scenes": 24})
            self.assertTrue(x["pu_month_cap"] > 0)
            mask, info = run.load_mask(sid, mv)
            g = Grid.for_site(x)
            self.assertEqual(mask.shape, (g.height, g.width))
            self.assertEqual(info["bbox"], x["bbox"])
            meta = run.site_meta(x)
            self.assertFalse(meta["quay"]["ready"])
            self.assertEqual(meta["quay"]["est2_calib_source"], "ZJG" if quay_est2._P else "placeholder")
        self.assertNotIn("SYL", [y["id"] for y in run.load_sites() if run.runs_daily(y)])

    def test_weights_are_placeholders(self):
        t = quay_est.load_classes()
        for sid in ("SYL", "SLP"):
            st = t["sites"][sid]
            self.assertEqual(st["status"], "placeholder")
            self.assertIn("要出典", st["w_conf"])
            self.assertTrue(quay_est.site_weights_info(sid)["own"])
        slp = [c["id"] for c in quay_est.Model(site_id="SLP").classes]
        self.assertIn("094", slp)
        self.assertNotIn("022", slp)  # no weight: not in the prior
        self.assertNotIn("094", [c["id"] for c in quay_est.Model().classes])  # Zhanjiang unchanged

    def test_zone_types(self):
        m = quay_est.Model(site_id="SLP", sector_roles=fsectors.sector_roles(run.site_sectors(self.sites()["SLP"])))
        self.assertEqual(m.zone_of("SLP", 123, "A1"), ("SUB", "submarine"))
        self.assertEqual(m.zone_of("SLP", 123, "B"), ("SH", "shore"))
        sub = np.array([k == "submarine" for k in m.kinds])
        self.assertGreater(m.zone_weights("submarine")[sub].sum(), m.zone_weights("large")[sub].sum())
        y = quay_est.Model(site_id="SYL")
        self.assertEqual(y.zone_of("SYL", 1, "A2"), ("SN", "medium"))
        z = quay_est.load_zones()
        for t in ("carrier", "submarine"):
            self.assertIn("仮置き", z["types"][t]["conf"])


if __name__ == "__main__":
    unittest.main()
