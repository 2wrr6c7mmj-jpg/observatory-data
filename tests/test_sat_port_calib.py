"""est-2 PR-2: a port's own calibration (shrunk toward Zhanjiang), the review set / answers flow, and the
out-of-sample evaluation. No network, no CDSE."""
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import eval_est2, quay_est2 as e2, quay_est2_calib as cal, s2_answers
    from sat.grid import Grid

ROOT = Path(__file__).resolve().parents[1]


def wcsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_port(out: Path, site="TNS", shift=-15.0, n_days=6, per_day=6, answer_c=1, noise=3.0):
    """Answers + S1 match of a fake port whose occupied depth is ``shift`` m off Zhanjiang's away-face mean."""
    mu = list(e2.DEPTH_MU["away"])
    ans, match = [], []
    rng = np.random.default_rng(1)
    for d in range(n_days):
        date = f"2026-0{d + 1}-10"
        for i in range(per_day):
            c = 1 + (i % 3) if answer_c is None else answer_c
            sid = f"{1000 + i}:L:00"
            ans.append({"site_id": site, "section_id": sid, "date": date, "verified": "1", "verified_occupied": "1", "verified_c": str(c),
                        "verified_color": "", "verified_by": "owner", "verified_note": ""})
            match.append({"site_id": site, "section_id": sid, "sector": "A", "kind": "pier", "s2_date": date, "face_look": "away",
                          "relative_orbit": "157", "s1_occupied": "1", "s1_depth_m": str(int(round(mu[c - 1] + shift + rng.normal(0, noise)))),
                          "s1_acquired_utc": f"{date}T10:48:00Z"})
    wcsv(out / "labels" / f"{site}-s2-answers.csv", ans, cal.ANSWER_FIELDS)
    wcsv(out / "labels" / f"{site}-s2-s1-match.csv", match, list(match[0]))
    wcsv(out / "ship_counts.csv", [{"site_id": site, "acquired_utc": m["s1_acquired_utc"], "incidence_deg": "36.0", "relative_orbit": "157"} for m in match],
         ["site_id", "acquired_utc", "incidence_deg", "relative_orbit"])
    return ans, match


@unittest.skipUnless(np is not None, "numpy not installed")
class PortFitTest(unittest.TestCase):
    def rows(self, shift, n_days=6, per_day=6):
        with tempfile.TemporaryDirectory() as d:
            make_port(Path(d), shift=shift, n_days=n_days, per_day=per_day)
            return cal.load_rows(Path(d), "TNS", None, lambda t, o: 36.0)[0]

    def test_shrinkage_formula(self):
        rows = self.rows(-15.0)
        self.assertEqual(len(rows), 36)
        base = cal.zjg_base()
        fit = cal.fit_port(rows, base)
        obs = float(np.mean([r["s1_depth"] for r in rows if r["v_c"] == "1" and r["group"] == "away"]))
        n = sum(1 for r in rows if r["v_c"] == "1" and r["group"] == "away")
        want = (n * obs + cal.K_PORT * base["depth_mu"]["away"][0]) / (n + cal.K_PORT)
        self.assertAlmostEqual(fit["depth_mu"]["away"][0], want, delta=0.06)
        self.assertLess(fit["depth_mu"]["away"][0], base["depth_mu"]["away"][0])       # moved toward the port's shallower depths
        self.assertGreater(fit["depth_mu"]["away"][0], obs)                              # but not all the way
        self.assertEqual(fit["depth_mu"]["along"], base["depth_mu"]["along"])           # no data: Zhanjiang's value
        self.assertLess(fit["depth_sd"], base["depth_sd"])                               # sd 3 m observed, shrunk with 10 df

    def test_no_shrink_beyond_zero_data(self):
        fit = cal.fit_port([], cal.zjg_base())
        self.assertEqual(fit["depth_mu"], cal.zjg_base()["depth_mu"])
        self.assertEqual(fit["depth_sd"], cal.zjg_base()["depth_sd"])

    def test_adopted_when_the_port_differs(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cd:
            out = Path(d)
            make_port(out, shift=-20.0, n_days=8, per_day=9, answer_c=None)
            with mock.patch.object(cal, "CALIB_DIR", Path(cd)):
                res = cal.write_port(out, "TNS")
                self.assertTrue((Path(cd) / "TNS-est2.json").exists())
            self.assertTrue(res["adopted"], res["evaluation"].get("reason"))
            ev = res["evaluation"]
            self.assertGreater(ev["own_cv"]["columns"]["log_score"], ev["zjg_borrowed"]["columns"]["log_score"])
            self.assertEqual(res["labels"]["verified"], 72)

    def test_not_adopted_when_the_port_matches_zhanjiang(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cd:
            out = Path(d)
            make_port(out, shift=0.0, n_days=8, per_day=9, answer_c=None, noise=12.0)
            with mock.patch.object(cal, "CALIB_DIR", Path(cd)):
                res = cal.write_port(out, "TNS")
            self.assertFalse(res["adopted"])
            self.assertIn("params", res)

    def test_too_few_answers_not_adopted(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cd:
            out = Path(d)
            make_port(out, shift=-25.0, n_days=2, per_day=2, answer_c=None)
            with mock.patch.object(cal, "CALIB_DIR", Path(cd)):
                res = cal.write_port(out, "TNS")
            self.assertFalse(res["adopted"])
            self.assertEqual(res["evaluation"]["reason"], "列数の答えが少ない")

    def test_no_answers_nothing_written(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as cd:
            with mock.patch.object(cal, "CALIB_DIR", Path(cd)):
                self.assertIsNone(cal.write_port(Path(d), "TNS"))
                self.assertEqual(list(Path(cd).iterdir()), [])


@unittest.skipUnless(np is not None, "numpy not installed")
class PortUseTest(unittest.TestCase):
    def test_source_and_labels(self):
        with tempfile.TemporaryDirectory() as cd:
            with mock.patch.object(e2, "CALIB_DIR", Path(cd)):
                self.assertEqual(e2.calib_info("SYL")["calib_label"], "湛江流用（未較正）")
                f = {"version": "x", "adopted": True, "labels": {"verified": 57}, "params": {"depth_sd": 9.0, "toward_delta": [-5, -5, -5]}}
                (Path(cd) / "SYL-est2.json").write_text(json.dumps(f))
                self.assertEqual(e2.calib_source("SYL"), "self")
                self.assertEqual(e2.calib_info("SYL"), {"calib_source": "self", "calib_n": 57, "calib_label": "較正済み（57 件）"})
                f["adopted"] = False
                (Path(cd) / "SYL-est2.json").write_text(json.dumps(f))
                self.assertEqual(e2.calib_source("SYL"), "ZJG")
                self.assertTrue(e2.calib_info("SYL")["calib_label"].startswith("湛江流用（未較正）"))
                self.assertEqual(e2.calib_source("SLP"), "ZJG")   # other ports unaffected

    def test_context_swaps_and_restores(self):
        with tempfile.TemporaryDirectory() as cd:
            with mock.patch.object(e2, "CALIB_DIR", Path(cd)):
                f = {"adopted": True, "labels": {"verified": 9},
                     "params": {"depth_sd": 7.0, "depth_mu": {"away": [20, 40, 60]}, "toward_delta": [-4.0, -4.0, -4.0], "smear_p_real": 0.5, "hard_ways": ["77"]}}
                (Path(cd) / "SYL-est2.json").write_text(json.dumps(f))
                before = (e2.DEPTH_SD_M, dict(e2.DEPTH_MU), e2.TOWARD_DELTA, e2.SMEAR_P_REAL, set(e2.HARD_WAYS), e2.depth_mu("toward", 36.0))
                with e2.site_calibration("SYL"):
                    self.assertEqual(e2.DEPTH_SD_M, 7.0)
                    self.assertEqual(e2.depth_mu("away", None), (20, 40, 60))
                    self.assertAlmostEqual(e2.depth_mu("toward", 36.0)[0], before[5][0] - 4.0)
                    self.assertEqual(e2.HARD_WAYS, {"77"})
                with e2.site_calibration("ZJG"), e2.site_calibration("SLP"):   # no own file / Zhanjiang: unchanged
                    self.assertEqual(e2.DEPTH_SD_M, before[0])
                after = (e2.DEPTH_SD_M, dict(e2.DEPTH_MU), e2.TOWARD_DELTA, e2.SMEAR_P_REAL, set(e2.HARD_WAYS), e2.depth_mu("toward", 36.0))
                self.assertEqual(before, after)


@unittest.skipUnless(np is not None, "numpy not installed")
class ReviewSetTest(unittest.TestCase):
    def test_image_grid_keeps_the_south_west_corner(self):
        g = Grid(109.665, 18.185, 109.715, 18.242, 529, 631)
        old = s2_answers.image_grid(g, 529, 498)
        self.assertEqual((old.west, old.south, old.width, old.height), (g.west, g.south, 529, 498))
        self.assertAlmostEqual(old.north, 18.23, places=2)
        with self.assertRaises(ValueError):
            s2_answers.image_grid(g, 600, 498)

    def test_days_need_an_s1_scene_within_four_days(self):
        s2 = [{"site_id": "T", "mask_version": "T-land-v1", "date": d, "acquired_utc": f"{d}T03:00:00Z", "result": r, "image": f"s2/T/v1/{d}.webp"}
              for d, r in (("2026-01-10", "clear"), ("2026-02-10", "clear"), ("2026-03-10", "cloudy"), ("2026-11-20", "clear"))]
        s1 = [{"acquired_utc": "2026-01-12T10:00:00Z", "relative_orbit": "157", "result": "counted"},
              {"acquired_utc": "2026-02-20T10:00:00Z", "relative_orbit": "157", "result": "counted"},
              {"acquired_utc": "2026-01-10T10:00:00Z", "relative_orbit": "99", "result": "counted"}]
        got = s2_answers.eligible_days(s2, s1, {"157"}, ["T-land-v1"], "2026-11-15")
        self.assertEqual([d["date"] for d in got], ["2026-01-10"])
        self.assertEqual(got[0]["s1_near"][0]["relative_orbit"], "157")
        oos = s2_answers.eligible_days(s2, [], {"157"}, ["T-land-v1"], "2026-11-15", oos=True)
        self.assertEqual([d["date"] for d in oos], ["2026-11-20"])

    def test_pick_days_spreads(self):
        days = [{"date": f"d{i:02d}"} for i in range(20)]
        got = [d["date"] for d in s2_answers.pick_days(days, 8)]
        self.assertEqual(len(got), 8)
        self.assertEqual((got[0], got[-1]), ("d00", "d19"))

    def test_pick_sections_round_robin_and_variable_first(self):
        secs = [{"section_id": f"{w}:L:{k:02d}", "way_id": w, "side": 1, "length_m": 150} for w in (1, 2) for k in range(6)]
        freq = {s["section_id"]: (1 if s["way_id"] == 1 else 2, 3) for s in secs}
        freq["2:L:05"] = (0, 3)
        got = s2_answers.pick_sections(secs, freq, 4)
        self.assertEqual(len(got), 4)
        self.assertEqual({s["way_id"] for s in got}, {1, 2})   # not all from one way
        self.assertNotIn("2:L:05", [s["section_id"] for s in got])

    def test_merge_validates_and_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rv = {"sections": [{"id": "1:L:00"}, {"id": "2:L:00"}], "days": [{"date": "2026-01-10"}]}
            (out / "labels" / "TNS").mkdir(parents=True)
            (out / "labels" / "TNS" / "review.json").write_text(json.dumps(rv))
            f = out / "dl.csv"
            wcsv(f, [{"site_id": "TNS", "section_id": "1:L:00", "date": "2026-01-10", "verified": "1", "verified_occupied": "1", "verified_c": "2"},
                     {"site_id": "TNS", "section_id": "2:L:00", "date": "2026-01-10", "verified": "0", "verified_occupied": "", "verified_c": ""},
                     {"site_id": "TNS", "section_id": "9:L:00", "date": "2026-01-10", "verified": "1", "verified_occupied": "0", "verified_c": "0"},
                     {"site_id": "TNS", "section_id": "1:L:00", "date": "2026-01-11", "verified": "1", "verified_occupied": "0", "verified_c": "0"},
                     {"site_id": "TNS", "section_id": "2:L:00", "date": "2026-01-10", "verified": "1", "verified_occupied": "1", "verified_c": "7"}],
                 ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c"])
            r = s2_answers.merge("TNS", f, out)
            self.assertEqual((r["added"], r["replaced"], len(r["rejected"])), (2, 0, 3))
            wcsv(f, [{"site_id": "TNS", "section_id": "1:L:00", "date": "2026-01-10", "verified": "1", "verified_occupied": "0", "verified_c": "0"}],
                 ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c"])
            r = s2_answers.merge("TNS", f, out)
            self.assertEqual((r["added"], r["replaced"], r["total"]), (0, 1, 2))
            rows = list(csv.DictReader(io.StringIO((out / "labels" / "TNS-s2-answers.csv").read_text(encoding="utf-8"))))
            self.assertEqual({(x["section_id"], x["verified_occupied"]) for x in rows}, {("1:L:00", "0"), ("2:L:00", "")})
            self.assertEqual({x["verified_by"] for x in rows}, {"owner"})

    def test_s1_plan_lists_scenes_and_pu(self):
        rv = {"site_id": "TNS", "days": [{"date": "2026-01-10", "s1_near": [{"acquired_utc": "2026-01-12T10:00:00Z", "relative_orbit": "157", "dt_h": 31.0}]},
                                          {"date": "2026-01-11", "s1_near": [{"acquired_utc": "2026-01-12T10:00:00Z", "relative_orbit": "157", "dt_h": 7.0}]},
                                          {"date": "2026-02-01", "s1_near": []}]}
        pl = s2_answers.s1_plan(rv, {}, Grid(110.0, 21.0, 110.05, 21.05, 500, 500), {"2026-01-12T10:00:00Z"})
        self.assertEqual(len(pl["scenes"]), 1)
        self.assertEqual(pl["scenes"][0]["days"], ["2026-01-10", "2026-01-11"])
        self.assertTrue(pl["scenes"][0]["stored_stretches"])
        self.assertGreater(pl["pu_total"], 0)
        self.assertEqual(pl["pu_if_stored_stretches_used"], 0)
        self.assertFalse(pl["fetched"])

    def test_committed_review_sets(self):
        for sid in ("SYL", "SLP", "QGZ", "ZDH"):
            p = ROOT / "docs" / "data" / "sat" / "labels" / sid / "review.json"
            if not p.exists():
                continue
            rv = json.loads(p.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(rv["days"]), s2_answers.MIN_DAYS)
            self.assertLessEqual(rv["cells"], 8 * 40)
            size = sum(f.stat().st_size for f in (p.parent).rglob("*") if f.is_file())
            self.assertLess(size, 5 * 1024 * 1024, f"{sid} over 5 MB")
            for d in rv["days"]:
                self.assertTrue((p.parent / d["image"]).exists())
            self.assertNotIn("@", p.read_text(encoding="utf-8").replace("@@", ""), "no contact in the data")

    def test_page_has_no_operator_identity(self):
        h = (ROOT / "docs" / "sat" / "label" / "index.html").read_text(encoding="utf-8")
        self.assertNotRegex(h, r"mailto:|@[a-z0-9-]+\.[a-z]{2,}")
        self.assertIn("review.json", h)
        self.assertIn("Copernicus Sentinel", h)


@unittest.skipUnless(np is not None, "numpy not installed")
class EvalTest(unittest.TestCase):
    def test_sample_zero(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "labels").mkdir()
            wcsv(out / "quay_counts.csv", [], ["site_id"])
            r = eval_est2.evaluate_site(out, "ZJG")
            self.assertEqual(r["status"], "標本 0")
            self.assertEqual((r["columns"]["n"], r["columns"]["share"], r["warship_interval"]["n"]), (0, None, 0))

    def test_committed_eval_is_sample_zero_or_consistent(self):
        p = ROOT / "docs" / "data" / "sat" / "eval_est2.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        self.assertIn("ZJG", d["sites"])
        meta = json.loads((ROOT / "docs" / "data" / "sat" / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["quay"]["estimate2_evaluation"]["file"], "eval_est2.json")

    def test_confusion_and_interval(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            coords = [[110.44, 21.22 + i * 0.0001] for i in range(5)]
            feat = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"site_id": "ZJG", "acquired_utc": "2026-10-28T10:48:45Z", "osm_way": 5, "face": "5+", "n": 1}}
            (out / "quay").mkdir()
            (out / "quay" / "2026-10.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}))
            st = {"p_real": 1.0, "p_c": [0.8, 0.1, 0.1]}
            sc = {"site_id": "ZJG", "acquired_utc": "2026-10-28T10:48:45Z", "warship": {"p10": 3, "median": 5, "p90": 8}, "stretches": [st]}
            (out / "quay_est2.json").write_text(json.dumps({"scenes": [sc]}))
            (out / "labels" / "ZJG").mkdir(parents=True)
            rv = {"sections": [{"id": "5:L:00", "way": 5, "side": 1, "line": coords}, {"id": "9:L:00", "way": 9, "side": 1, "line": coords}],
                  "days": [{"date": "2026-10-28", "acquired_utc": "2026-10-28T03:22:00Z"}]}
            (out / "labels" / "ZJG" / "review-oos.json").write_text(json.dumps(rv))
            wcsv(out / "labels" / "ZJG-oos-answers.csv",
                 [{"site_id": "ZJG", "section_id": "5:L:00", "date": "2026-10-28", "verified": "1", "verified_occupied": "1", "verified_c": "1"},
                  {"site_id": "ZJG", "section_id": "9:L:00", "date": "2026-10-28", "verified": "1", "verified_occupied": "0", "verified_c": "0"}],
                 ["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c"])
            wcsv(out / "labels" / "ZJG-oos-scene.csv", [{"site_id": "ZJG", "date": "2026-10-28", "warships_60": "9", "note": ""}],
                 ["site_id", "date", "warships_60", "note"])
            r = eval_est2.evaluate_site(out, "ZJG")
            self.assertEqual(r["columns"]["confusion"], {"0->0": 1, "1->1": 1})
            self.assertEqual(r["columns"]["share"], 1.0)
            self.assertEqual(r["occupied"]["agree"], 2)
            self.assertEqual(r["warship_interval"]["n"], 1)
            self.assertEqual(r["warship_interval"]["hit"], 0)     # 9 is outside 3-8
            self.assertLess(r["occupied"]["brier"], 0.01)


if __name__ == "__main__":
    unittest.main()
