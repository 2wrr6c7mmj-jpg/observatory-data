"""pres-3 (trial / research, sat/pres3/): synthetic data only, no network, no fetched windows.

What is guarded:
  1. the exemplar / history features of a cell use only the section's answers on days strictly before the cell's day
  2. the reference images (answer-free features, difference features) are made only from days strictly before the cell's day
  3. the evaluation functions (metrics, changed cells, folds, paired difference)
  4. the research code writes no published file other than its own two, is not wired into the daily collection, and the
     published sheet agrees with the published per-cell probabilities
The steps that need rasterio (fetch, build) and scikit-learn (the classifiers) are not run here; the Tests workflow installs
neither. Everything else needs numpy only and is skipped where numpy is missing (the collect workflows).
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

from sat.pres3 import common

if np is not None:
    from sat.pres3 import diffeat, evalcv, feats, final, model, pathfeat

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "docs" / "data" / "sat" / "eval"
SECTIONS = ("w1:L:00", "w1:L:01")


def synth(days=12, seed=0, port="AAA"):
    """One port, two sections, ``days`` clear days 10 days apart. Patches [N, 24, 41, 5] like sat.pres3.build makes."""
    rng = np.random.default_rng(seed)
    offs = np.arange(-80, 125, 5.0)
    dates = [f"2026-{1 + d // 3:02d}-{1 + 10 * (d % 3):02d}" for d in range(days)]
    C = []; P = []
    for d, date in enumerate(dates):
        for k, sid in enumerate(SECTIONS):
            y = int((d + k) % 3 == 0)
            p = 0.05 + 0.01 * rng.random((24, len(offs), 5))
            p[:, offs < 0] += 0.10                               # land
            if y: p[6:16, (offs >= 0) & (offs < 30)] += 0.08     # a ship at the quay
            C.append(dict(port=port, date=date, section=sid, kind="pier", sector="A", length_m=240, cloud=0.0, shadow=0.0,
                          nodata=0.0, y=y, eval=True, p_cur=0.5))
            P.append(p)
    return C, np.asarray(P, np.float32), offs, dates


def same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@unittest.skipUnless(np is not None, "numpy not installed")
class ReferenceFromEarlierDaysTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(warnings.catch_warnings()); warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices: no reference yet
        self.C, self.P, self.offs, self.dates = synth()
        self.cut = self.dates[8]
        self.early = np.array([c["date"] <= self.cut for c in self.C])
        self.P2 = self.P.copy()
        self.P2[~self.early] = np.random.default_rng(9).random(self.P2[~self.early].shape).astype(np.float32)   # later days differ

    def test_features_ignore_later_days(self):
        X, names, good = feats.features(self.P, self.offs, self.C)
        X2, names2, _ = feats.features(self.P2, self.offs, self.C)
        self.assertEqual(names, names2)
        self.assertTrue(same(X[self.early], X2[self.early]))
        self.assertFalse(same(X[~self.early], X2[~self.early]))
        # the reference is really in use on the compared cells (6 or more earlier clear days), not NaN everywhere
        late = np.array([c["date"] == self.cut for c in self.C])
        self.assertTrue((X[late, names.index("nref")] >= 6).all())
        self.assertTrue(np.isfinite(X[late][:, [names.index("vis_d50_mean_0"), names.index("ld_a04_area")]]).all())

    def test_features_use_no_day_of_the_cell_or_later_as_reference(self):
        X, names, _ = feats.features(self.P, self.offs, self.C)
        k = names.index("nref")
        for i, c in enumerate(self.C):       # every day is clear: the reference count is the number of earlier days
            self.assertEqual(X[i, k], self.dates.index(c["date"]))

    def test_features_do_depend_on_earlier_days(self):
        P3 = self.P.copy(); first = np.array([c["date"] == self.dates[0] for c in self.C]); P3[first] += 0.05 * np.random.default_rng(3).random(P3[first].shape).astype(np.float32)
        X, names, _ = feats.features(self.P, self.offs, self.C)
        X3, _, _ = feats.features(P3, self.offs, self.C)
        late = np.array([c["date"] == self.cut for c in self.C])
        self.assertFalse(same(X[late], X3[late]))

    def test_features_use_no_answer(self):
        C2 = [dict(c, y=None) for c in self.C]
        self.assertTrue(same(feats.features(self.P, self.offs, self.C)[0], feats.features(self.P, self.offs, C2)[0]))

    def test_difference_features_ignore_later_days_and_later_answers(self):
        F, names = diffeat.cell_feats(self.C, self.P, self.offs)
        C2 = [dict(c, y=(1 - c["y"] if c["date"] > self.cut else c["y"])) for c in self.C]
        F2, names2 = diffeat.cell_feats(C2, self.P2, self.offs)
        self.assertEqual(names, names2); self.assertEqual(len(names), 52)
        self.assertTrue(same(F[self.early], F2[self.early]))
        self.assertTrue(np.isfinite(F[self.early]).any())
        # the cell's own answer is not a feature either
        last = [i for i, c in enumerate(self.C) if c["date"] == self.cut]
        C3 = [dict(c) for c in self.C]
        for i in last: C3[i]["y"] = 1 - C3[i]["y"]
        self.assertTrue(same(F[last], diffeat.cell_feats(C3, self.P, self.offs)[0][last]))
        # an earlier answer is one (the exemplars change)
        C4 = [dict(c, y=(1 - c["y"] if c["date"] < self.cut else c["y"])) for c in self.C]
        self.assertFalse(same(F[last], diffeat.cell_feats(C4, self.P, self.offs)[0][last]))

    def test_path_features_ignore_later_days(self):          # tried, not adopted; the same rule
        F = pathfeat.cell_feats(self.C, self.P, self.offs); F2 = pathfeat.cell_feats(self.C, self.P2, self.offs)
        self.assertEqual(F.shape[1], len(pathfeat.NAMES))
        self.assertTrue(same(F[self.early], F2[self.early]))


@unittest.skipUnless(np is not None, "numpy not installed")
class HistoryFromEarlierAnswersTest(unittest.TestCase):
    def setUp(self):
        self.C, P, offs, self.dates = synth(days=9)
        rng = np.random.default_rng(1)
        self.X = rng.random((len(self.C), 5)).astype(np.float32); self.s = rng.normal(size=len(self.C))
        self.lab, self.y = model.labelled(self.C)
        self.phys = [1, 3]

    def adapt(self, y=None, use=None, lab=None):
        lab = self.lab if lab is None else lab
        return model.adapt_feats(self.C, self.X, self.y if y is None else y, lab, self.phys, self.s, set(lab) if use is None else use)

    def test_counts_and_previous_answer(self):
        A = self.adapt()
        for i, c in enumerate(self.C):
            past = [t for t in self.lab if self.C[t]["section"] == c["section"] and self.C[t]["date"] < c["date"]]
            n1 = sum(self.y[t] == 1 for t in past)
            self.assertEqual((A[i, 0], A[i, 1]), (n1, len(past) - n1))
            self.assertAlmostEqual(float(A[i, 2]), (n1 + 1) / (len(past) + 2), places=6)
            if past:
                prev = max(past, key=lambda t: self.C[t]["date"])
                self.assertEqual(A[i, 3], self.y[prev])
                self.assertEqual(A[i, 5], (dt.date.fromisoformat(c["date"]) - dt.date.fromisoformat(self.C[prev]["date"])).days)
            else:
                self.assertTrue(np.isnan(A[i, 3]) and np.isnan(A[i, 5]))

    def test_answers_of_the_day_and_later_do_not_change_the_features(self):
        A = self.adapt()
        for cut in self.dates[2:]:
            y2 = self.y.copy(); later = np.array([c["date"] >= cut for c in self.C]); y2[later] = 1 - y2[later]
            rows = np.array([c["date"] <= cut for c in self.C])        # the cut day itself: its own and same-day answers changed
            self.assertTrue(same(A[rows], self.adapt(y=y2)[rows]), cut)

    def test_an_earlier_answer_does_change_them(self):
        A = self.adapt(); y2 = self.y.copy(); y2[[i for i, c in enumerate(self.C) if c["date"] == self.dates[0]]] ^= 1
        rows = np.array([c["date"] == self.dates[4] for c in self.C])
        self.assertFalse(same(A[rows], self.adapt(y=y2)[rows]))

    def test_only_cells_in_use_are_exemplars(self):
        keep = {i for i in self.lab if self.C[i]["date"] != self.dates[1]}
        A = self.adapt(use=keep)
        i = next(k for k, c in enumerate(self.C) if c["date"] == self.dates[2] and c["section"] == SECTIONS[0])
        self.assertEqual(A[i, 0] + A[i, 1], 1)                          # of the two earlier answered days only one may be used
        self.assertEqual(self.adapt()[i, 0] + self.adapt()[i, 1], 2)

    def test_other_ports_and_sections_are_not_exemplars(self):
        C2, _, _, _ = synth(days=9, port="BBB")
        C = self.C + C2; X = np.vstack([self.X, self.X]); s = np.r_[self.s, self.s]
        lab, y = model.labelled(C)
        A = model.adapt_feats(C, X, y, lab, self.phys, s, set(lab))
        self.assertTrue(same(A[:len(self.C)], self.adapt()))

    def test_phys_columns(self):
        names = ["ld_a04_area", "ld_a02_area", "blob_vis_min_L", "x", "ld_nr_run10", "ld_nr_run20"]
        self.assertEqual(model.phys_cols(names), [0, 2, 4])


@unittest.skipUnless(np is not None, "numpy not installed")
class EvaluationTest(unittest.TestCase):
    def test_metrics(self):
        m = evalcv.metrics([1, 1, 0, 0, 1], [0.9, 0.4, 0.6, 0.1, 0.5])
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"], m["n"]), (2, 1, 1, 1, 5))
        self.assertAlmostEqual(m["acc"], 0.6); self.assertAlmostEqual(m["rec"], 2 / 3); self.assertAlmostEqual(m["pre"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3); self.assertAlmostEqual(m["brier"], (0.01 + 0.36 + 0.36 + 0.01 + 0.25) / 5)
        self.assertAlmostEqual(m["auc"], 4 / 6)                          # 6 pairs, two inverted
        self.assertAlmostEqual(evalcv.metrics([1, 0], [0.5, 0.5])["auc"], 0.5)      # ties count half
        m = evalcv.metrics([0, 0], [0.1, 0.2])
        self.assertTrue(math.isnan(m["auc"]) and math.isnan(m["rec"]) and math.isnan(m["pre"])); self.assertEqual(m["acc"], 1.0)

    def cells(self):
        rows = [("A", "s", "2025-11-01", 1), ("A", "s", "2025-12-10", 1), ("A", "s", "2026-03-05", 0), ("A", "t", "2026-03-05", 0),
                ("B", "s", "2026-01-01", 0), ("B", "s", "2026-07-01", 1), ("B", "s", "2026-06-01", None), ("B", "s", "2026-09-02", 1)]
        C = [dict(port=p, section=s, date=d, y=y) for p, s, d, y in rows]
        return C, [i for i, c in enumerate(C) if c["y"] is not None]

    def test_changed_flags(self):
        C, lab = self.cells()
        ch = evalcv.changed_flags(C, lab)
        self.assertEqual(ch, {0: -1, 1: 0, 2: 1, 3: -1, 4: -1, 5: 1, 7: 0})      # the same section id in another port is another section
        self.assertNotIn(6, ch)                                                    # no answer: neither changed nor a "previous answer"

    def test_folds(self):
        C, lab = self.cells()
        lopo = {p: (tr, te) for p, tr, te in evalcv.folds_lopo(C, lab)}
        self.assertEqual(lopo["A"], ([4, 5, 7], [0, 1, 2, 3])); self.assertEqual(lopo["B"], ([0, 1, 2, 3], [4, 5, 7]))
        fwd = list(evalcv.folds_fwd(C, lab))
        self.assertEqual([b for b, _, _ in fwd], evalcv.BOUNDS)
        seen = []
        for b, tr, te in fwd:
            self.assertTrue(all(C[i]["date"] < b for i in tr)); self.assertTrue(all(C[i]["date"] >= b for i in te))
            seen += te
        self.assertEqual(sorted(seen), [1, 2, 3, 4, 5, 7])                         # each cell after the first bound is tested once

    def test_bootstrap_and_paired_difference(self):
        C, lab = self.cells(); y = np.array([c["y"] if c["y"] is not None else -1 for c in C])
        p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.6, np.nan, 0.3]); q = np.array([0.1, 0.8, 0.9, 0.7, 0.1, 0.2, np.nan, 0.3])
        lo, hi = evalcv.boot_ci(C, lab, y[lab], p[lab], B=200)
        self.assertEqual((lo, hi), evalcv.boot_ci(C, lab, y[lab], p[lab], B=200))   # fixed seed
        self.assertTrue(0 <= lo <= 5 / 7 <= hi <= 1)
        r = final.acc_ci(C, y, lab + [6], p, B=100)                               # the cell without a probability is left out
        self.assertEqual((r["n"], r["tp"], r["fp"], r["fn"], r["tn"]), (7, 3, 1, 1, 2)); self.assertEqual(r["acc"], 0.714)
        d = final.diff_ci(C, y, lab + [6], p, q, B=100)
        self.assertEqual((d["n"], d["diff"]), (7, 0.429)); self.assertTrue(d["ci"][0] <= d["diff"] <= d["ci"][1])
        self.assertIsNone(final.acc_ci(C, y, [6], p))


class NoPublishedFileIsTouchedTest(unittest.TestCase):
    def test_working_directory_cannot_be_inside_the_published_tree(self):
        with self.assertRaises(SystemExit):
            common.work_dir(str(common.SAT / "eval"))
        with self.assertRaises(SystemExit):
            common.work_dir(str(common.SAT / "pres3-work" / "x"))
        self.assertFalse((common.SAT / "pres3-work").exists())

    def test_not_wired_into_the_daily_collection(self):
        for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")) + [p for p in sorted((ROOT / "sat").glob("*.py"))] + [ROOT / "sat" / "sites.json", ROOT / "sat" / "publish.sh"]:
            self.assertNotIn("pres3", f.read_text(encoding="utf-8"), f)

    def test_import_needs_no_third_party_package(self):
        # the collect workflows import every test module without numpy: the package and its helpers must import bare
        import sat.pres3  # noqa: F401
        self.assertEqual(sat.pres3.VERSION, "pres-3")
        self.assertIsNotNone(importlib.util.find_spec("sat.pres3.common"))

    @unittest.skipUnless(np is not None, "numpy not installed")
    def test_publish_writes_only_its_two_files(self):
        R = {"config": {"key": final.KEY, "note": final.NOTE}, "lopo": {"pres3": {"acc": 0.5, "precision": float("nan"), "ci": [0.1, float("nan")]}}, "fwd": {}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "eval"; out.mkdir()
            sheet = out / "presence.json"; sheet.write_bytes(EVAL.joinpath("presence.json").read_bytes()[:2000])
            other = out / "pres1-design.md"; other.write_text("x", encoding="utf-8")
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            src = Path(d) / "oos.csv"; src.write_text("site_id,date\nAAA,2026-01-01\n", encoding="utf-8")
            final.publish(R, src, out, "2026-11-19T00:00:00Z")
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["pres1-design.md", "pres3.json", "pres3_oos.csv", "presence.json"])
            for name, b in before.items(): self.assertEqual((out / name).read_bytes(), b, name)
            j = json.loads((out / "pres3.json").read_text(encoding="utf-8"), parse_constant=self.fail)     # strict JSON: no NaN
            self.assertEqual((j["version"], j["generated_utc"]), ("pres-3", "2026-11-19T00:00:00Z"))
            self.assertIsNone(j["lopo"]["pres3"]["precision"]); self.assertEqual(j["lopo"]["pres3"]["ci"], [0.1, None])
            self.assertEqual(j["lopo"]["pres3"]["acc"], 0.5)
            self.assertTrue(j["limits"] and j["sources"] and "Copernicus Sentinel" in j["sources"][0])
            self.assertEqual((out / "pres3_oos.csv").read_bytes(), src.read_bytes())


class PublishedSheetTest(unittest.TestCase):
    """docs/data/sat/eval/pres3.json and pres3_oos.csv agree with each other (pure Python)."""

    @classmethod
    def setUpClass(cls):
        cls.sheet = json.loads((EVAL / "pres3.json").read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        with open(EVAL / "pres3_oos.csv", newline="", encoding="utf-8") as fh:
            cls.rows = list(csv.DictReader(fh))

    def counts(self, col, pick=lambda r: True):
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for r in self.rows:
            if r["in_eval"] != "1" or r[col] == "" or not pick(r): continue
            h = float(r[col]) >= 0.5; y = r["y"] == "1"
            c["tp" if h and y else "fp" if h else "fn" if y else "tn"] += 1
        return c

    def check(self, block, col, pick=lambda r: True):
        c = self.counts(col, pick); n = sum(c.values())
        self.assertEqual({k: block[k] for k in c}, c); self.assertEqual(block["n"], n)
        self.assertAlmostEqual(block["acc"], (c["tp"] + c["tn"]) / n, delta=0.00051)

    def test_cells(self):
        self.assertEqual(len(self.rows), 1531)
        self.assertEqual(len({(r["site_id"], r["date"], r["section_id"]) for r in self.rows}), 1531)
        self.assertEqual(sum(r["in_eval"] == "1" for r in self.rows), 737)
        self.assertEqual(sorted({r["site_id"] for r in self.rows}), sorted(common.PORTS))
        self.assertTrue(all(r["p_pres3_lopo"] != "" and r["y"] in ("0", "1") for r in self.rows))      # no cell is left out
        self.assertTrue(all((r["p_pres3_fwd"] != "") == (r["date"] >= "2025-12-01") for r in self.rows))

    def test_sheet_matches_the_cells(self):
        for fold, col, hist in (("lopo", "p_pres3_lopo", "p_hist_lopo"), ("fwd", "p_pres3_fwd", "p_hist_fwd")):
            b = self.sheet[fold]
            self.check(b["pres3"], col)
            self.check(b["history_only"], hist, lambda r: r[col] != "")
            self.check(b["current_s1"], "p_current_s1", lambda r: r[col] != "")
            for name, flag in (("changed", "1"), ("same", "0"), ("no_prev", "-1")):
                self.check(b[name]["pres3"], col, lambda r: r["changed"] == flag)
            for port in common.PORTS:
                self.check(b["by_port"][port]["pres3"], col, lambda r: r["site_id"] == port)
        self.check(self.sheet["lopo"]["gen_no_answers"], "p_gen_lopo")

    def test_changed_column_follows_the_answers(self):
        prev = {}
        for r in sorted(self.rows, key=lambda r: r["date"]):
            k = (r["site_id"], r["section_id"])
            self.assertEqual(int(r["changed"]), -1 if k not in prev else int(r["y"] != prev[k]), r)
            prev[k] = r["y"]

    def test_headline_is_below_the_target_and_says_so(self):
        self.assertEqual(self.sheet["version"], "pres-3")
        self.assertLess(self.sheet["lopo"]["pres3"]["acc"], 0.95); self.assertLess(self.sheet["fwd"]["pres3"]["acc"], 0.95)
        self.assertIn("届いていない", self.sheet["target"])
        self.assertGreaterEqual(len(self.sheet["limits"]), 6)

    def test_other_evaluation_sheets_are_not_rewritten_by_this_package(self):
        src = "".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "sat" / "pres3").glob("*.py")))
        for name in ("presence_snapshot", "occ2.json", "meta.json", "quay_pres1", "ship_counts", "quay_counts"):
            self.assertNotIn(name, src)
        self.assertEqual(src.count("'presence.json'"), 1)                # read once (labels.lookup), never opened for writing
        self.assertIn("json.load(open(common.SAT / 'eval' / 'presence.json'))", src)


if __name__ == "__main__":
    unittest.main()
