import hashlib
import json
import math
import shutil
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
    from sat import quay_est2 as e2
    from sat import run

DATA = Path(__file__).resolve().parents[1] / "docs" / "data" / "sat"


def feat(n, way=218405149, side=1, sector="A", kind="pier", length=150, k=1, depth=40, face="toward", db=None,
         coords=None, t="2026-10-26T10:48:45Z", fv="feat-1"):
    coords = coords or [[110.44, 21.22 + i * 0.0001] for i in range(5)]
    return {"site_id": "ZJG", "acquired_utc": t, "n": n, "osm_way": way, "face": f"{way}{'+' if side > 0 else '-'}",
            "sector": sector, "line_kind": kind, "length_m": length, "min": k, "depth_m": depth, "face_look": face,
            "db_line_m": db, "coords": coords, "features_version": fv, "mask_version": "ZJG-land-v2"}


ROW = {"site_id": "ZJG", "acquired_utc": "2026-10-26T10:48:45Z", "relative_orbit": "157",
       "orbit_direction": "ascending", "mask_version": "ZJG-land-v2"}


@unittest.skipUnless(np is not None, "numpy not installed")
class PartsTest(unittest.TestCase):
    def test_depth_likelihood_orders_columns(self):
        for face in ("toward", "along", "away"):
            for d, c in ((35, 1), (60, 2), (95, 3)):
                lk, cl = e2.depth_like(d, face, 45.0, 120.0)
                self.assertEqual(int(np.argmax(lk)) + 1, c, (face, d))
                self.assertEqual(cl, "depth")
        # at the end of the profile: censored, favours the widest
        lk, cl = e2.depth_like(90, "toward", 45.0, 90.0)
        self.assertEqual(cl, "depth_censored")
        self.assertEqual(int(np.argmax(lk)), 2)
        # no depth or no face: no clue
        self.assertEqual(e2.depth_like(None, "toward", 45.0, 120.0)[1], "")
        self.assertEqual(e2.depth_like(50, "", 45.0, 120.0)[1], "")

    def test_toward_means_by_incidence(self):
        self.assertEqual(e2.toward_mu(45.0), (45.0, 65.0, 95.0))
        self.assertEqual(e2.toward_mu(32.0), (55.0, 75.0, 95.0))
        self.assertEqual(e2.toward_mu(38.5), (50.0, 70.0, 95.0))

    def test_db_line_recorded_only(self):
        # weight 0 until #7: even on a face toward the radar the likelihood is flat, but the clue is counted
        self.assertEqual(e2.DB_WEIGHT, 0.0)
        self.assertEqual(e2.db_like(42, "away")[1], "")
        for x in (7, 22, 42, 62):
            lk, cl = e2.db_like(x, "toward")
            self.assertEqual(cl, "db_line")
            np.testing.assert_allclose(lk, np.ones(3))
        r1 = e2.estimate_scene(ROW, [feat(1, depth=60, db=7)], e2.Site("ZJG", [], {}))
        r2 = e2.estimate_scene(ROW, [feat(1, depth=60, db=62)], e2.Site("ZJG", [], {}))
        self.assertEqual(r1["stretches"][0]["p_c"], r2["stretches"][0]["p_c"])
        self.assertEqual(r1["stretches"][0]["db_line_m"], 7)
        self.assertIn("db_line", r1["stretches"][0]["clues"])

    def test_kappa_and_colour(self):
        self.assertEqual(e2.kappa(None), 0.0)
        self.assertEqual(e2.kappa(-6), 0.85)
        self.assertEqual(e2.kappa(20), 0.6)
        self.assertEqual(e2.kappa(-40.5), 0.4)
        self.assertEqual(e2.kappa(79.4), 0.25)
        base = e2.warship_p("A_pier", 130, 0.0, 0.0)
        self.assertAlmostEqual(base, e2.warship_p("A_pier", 130, math.log(3), 0.0))  # kappa 0: LR has no effect
        self.assertGreater(e2.warship_p("A_pier", 130, math.log(3), 0.85), base)
        self.assertLess(e2.warship_p("A_pier", 130, math.log(0.2), 0.85), base)
        # place and length
        self.assertGreater(base, e2.warship_p("B", 130, 0.0, 0.0))
        self.assertGreater(base, e2.warship_p("A_pier", 40, 0.0, 0.0))
        self.assertAlmostEqual(e2.warship_p("A_pier", 400, 0.0, 0.0), 0.9, places=3)
        # 60 m or more (the headline) never above any length; a 43 m boat counts only in the breakdown
        for L in (40, 43, 70, 130):
            self.assertLessEqual(e2.warship_p("A_pier", L, 0.0, 0.0, min_bin=1), e2.warship_p("A_pier", L, 0.0, 0.0))
        self.assertLess(e2.warship_p("A_pier", 43, 0.0, 0.0, min_bin=1), 0.25)
        self.assertGreater(e2.warship_p("A_pier", 43, 0.0, 0.0), 0.35)
        self.assertAlmostEqual(e2.warship_p("A_pier", 400, 0.0, 0.0, min_bin=1), 0.9, places=3)

    def test_headline_and_breakdown(self):
        r = e2.estimate_scene(ROW, [feat(1, length=60, depth=40), feat(2, length=200, depth=40,
                              coords=[[110.45, 21.23 + i * 0.0001] for i in range(5)])], e2.Site("ZJG", [], {}))
        self.assertLessEqual(r["warship"]["mean"], r["warship_all"]["mean"])
        d = r["stretches"][0]
        self.assertLess(d["p_warship_60"], d["p_warship_all"])

    def test_stretch_distribution(self):
        pc = np.array([0.5, 0.3, 0.2])
        s, w = e2.stretch_dists(pc, 2, 0.5, 0.3)
        self.assertAlmostEqual(s.sum(), 1.0)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(s[0], 0.7)
        self.assertAlmostEqual(float((np.arange(len(s)) * s).sum()), 0.3 * 2 * (0.5 + 0.6 + 0.6))


@unittest.skipUnless(np is not None, "numpy not installed")
class SceneTest(unittest.TestCase):
    def site(self):
        return e2.Site("ZJG", [], {})

    def test_pair_and_smear(self):
        a = feat(1, depth=60, face="toward")
        # the opposite face of the same pier (away), occupied in the same scene: a smear with p 0.3
        b = feat(2, side=-1, depth=30, face="away", coords=[[110.4405, 21.22 + i * 0.0001] for i in range(5)])
        r = e2.estimate_scene(ROW, [a, b], self.site())
        st = {d["n"]: d for d in r["stretches"]}
        self.assertEqual(st[2]["p_real"], 0.3)
        self.assertIn("smear", st[2]["clues"])
        self.assertEqual(st[1]["p_real"], 1.0)
        # orbit pair: the partner's deeper view of the same face moves c up
        prow = dict(ROW, acquired_utc="2026-10-27T10:41:00Z", relative_orbit="84")
        p = feat(1, depth=95, face="toward", t=prow["acquired_utc"])
        r2 = e2.estimate_scene(ROW, [a], self.site(), (prow, [p]))
        r1 = e2.estimate_scene(ROW, [a], self.site())
        self.assertIn("pair", r2["stretches"][0]["clues"])
        self.assertGreater(r2["stretches"][0]["c_mean"], r1["stretches"][0]["c_mean"])

    def test_face_from_sections(self):
        coords = [[110.44, 21.22 + i * 0.0001] for i in range(5)]
        sec = {"section_id": "218405149:L:00", "way_id": 218405149, "side": 1, "normal_az_deg": 257.6, "coords": coords}
        site = e2.Site("ZJG", [sec], {"218405149:L:00": [("2026-10-26T03:00:00Z", "grey")]})
        f = feat(1, face=None)
        r = e2.estimate_scene(dict(ROW, look_az_deg="77.6"), [f], site)
        d = r["stretches"][0]
        self.assertEqual((d["face"], d["face_from"]), ("toward", "face_sections"))
        self.assertEqual(d["s2"]["colour"], "grey")
        self.assertEqual(d["s2"]["kappa"], 0.85)
        # an S2 day more than 96 h away is not used
        site2 = e2.Site("ZJG", [sec], {"218405149:L:00": [("2026-10-20T03:00:00Z", "grey")]})
        self.assertIsNone(e2.estimate_scene(ROW, [f], site2)["stretches"][0]["s2"])


def _digest(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@unittest.skipUnless(np is not None, "numpy not installed")
class Est1UnchangedTest(unittest.TestCase):
    def test_published_data_est1_unchanged(self):
        """est-2 writes its own files only: quay_counts.csv (est-1) and quay/*.geojson stay byte for byte."""
        if not (DATA / "quay_counts.csv").exists():
            self.skipTest("no published quay data")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for name in ("quay_counts.csv", "ship_counts.csv"):
                shutil.copy(DATA / name, out / name)
            shutil.copytree(DATA / "quay", out / "quay")
            if (DATA / "labels").exists():
                shutil.copytree(DATA / "labels", out / "labels")
            before = _digest([out / "quay_counts.csv", *sorted((out / "quay").iterdir())])
            res = e2.build(out)
            self.assertEqual(before, _digest([out / "quay_counts.csv", *sorted((out / "quay").iterdir())]))
            q = run.read_csv(out / "quay_counts.csv")
            rows = run.read_csv(out / "quay_est2.csv")
            self.assertEqual(len(rows), len(q))
            for a, b in zip(sorted(q, key=lambda r: r["acquired_utc"]), rows):
                self.assertEqual(a["est_warship_mode"], b["est1_warship_mode"])
                self.assertLessEqual(int(b["warship_p10"]), int(b["warship_median"]))
                self.assertLessEqual(int(b["warship_median"]), int(b["warship_p90"]))
                self.assertLessEqual(int(b["warship_median"]), int(b["warship_all_median"]))
            doc = json.loads((out / "quay_est2.json").read_text())
            self.assertEqual(doc["version"], "est-2")
            self.assertIn("較正前", doc["status"])
            self.assertEqual(len(res["scenes"]), len(q))

    def test_run_store_est1_same_with_and_without_est2(self):
        from tests.test_sat_quay import SITE, FakeClient, RunQuayTest
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        outs = {}
        for use in (False, True):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                mask, chip, lines_dir, bl_dir = RunQuayTest.setup_dirs(None, d)
                out = d / "out"
                store = run.Store(out, [SITE])
                cli = FakeClient(chip)
                ctx = mock.patch.object(run.quay_est2, "build") if not use else mock.patch.object(run, "SITES_FILE", run.SITES_FILE)
                with ctx:
                    run.process_site(cli, store, SITE, mask, now, None, 100.0, 12, log=lambda *a: None,
                                     lines_dir=lines_dir, baseline_dir=bl_dir)
                    store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests,
                                "processing_units": cli.processing_units, "pu_source": "header", "scenes": 2})
                outs[use] = {"q": (out / "quay_counts.csv").read_text(),
                             "g": (out / "quay" / "2026-10.geojson").read_text(),
                             "e2": (out / "quay_est2.csv").exists(),
                             "requests": cli.requests,
                             "meta": json.loads((out / "meta.json").read_text())["quay"].get("estimate2", {}).get("version")}
        self.assertEqual(outs[False]["q"], outs[True]["q"])
        self.assertEqual(outs[False]["g"], outs[True]["g"])
        self.assertEqual(outs[False]["requests"], outs[True]["requests"])  # est-2 fetches nothing
        self.assertFalse(outs[False]["e2"])
        self.assertTrue(outs[True]["e2"])
        self.assertEqual(outs[True]["meta"], "est-2")


@unittest.skipUnless(np is not None, "numpy not installed")
class LabelCheckTest(unittest.TestCase):
    def test_check_runs_on_published_labels(self):
        if not (DATA / "labels" / "ZJG-s2-s1-match.csv").exists():
            self.skipTest("no labels")
        r = e2.label_check(DATA)
        self.assertGreater(r["rows"], 0)
        self.assertEqual(r["est2_agree"]["all"]["n"], r["rows"])


if __name__ == "__main__":
    unittest.main()
