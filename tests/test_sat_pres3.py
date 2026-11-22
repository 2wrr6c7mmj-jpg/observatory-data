"""pres-3 (trial / research, sat/pres3/): synthetic data only, no network, no fetched windows.

What is guarded:
  1. the exemplar / history features of a cell use only the section's answers on days strictly before the cell's day
  2. the reference images (answer-free features, difference features) are made only from days strictly before the cell's day
  3. the evaluation functions (metrics, changed cells, folds), that the classifiers are fitted only on the fold's training
     cells (a fake model in place of scikit-learn), that the intervals resample whole scenes and the difference is paired
  4. the research code writes no published file other than its own two, is not wired into the daily collection, and the
     published sheet agrees with the published per-cell probabilities
  5. confirm-1 (sat/pres3/triage.py): the rule v1 at its boundaries, and that the tables of the design
     (docs/data/sat/eval/confirm-design.md, section 2) come out of the published per-cell file
The steps that need rasterio (fetch, build) and scikit-learn (the classifiers) are not run here; the Tests workflow installs
neither. Everything else needs numpy only and is skipped where numpy is missing (the collect workflows).
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import math
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

from sat.pres3 import common

if np is not None:
    from sat.pres3 import diffeat, evalcv, feats, final, fuse, model, pathfeat, triage

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


class FakeModel:
    """Stands in for the gradient boosting (scikit-learn is not installed in the Tests workflow): records the rows given to
    fit (column 0 of the features is the cell's row number) and predicts 0.75 for the rows of predict_proba."""
    log = []

    def fit(self, Z, y, sample_weight=None):
        FakeModel.log.append(dict(rows=[int(v) for v in Z[:, 0]], y=list(y), w=None if sample_weight is None else list(sample_weight)))
        return self

    def predict_proba(self, Z):
        FakeModel.log[-1].setdefault("predicted", []).append([int(v) for v in Z[:, 0]])
        return np.column_stack([np.full(len(Z), 0.25), np.full(len(Z), 0.75)])


@unittest.skipUnless(np is not None, "numpy not installed")
class NoTestCellInTheFitTest(unittest.TestCase):
    """The coefficients: no cell of the fold's test set (and, out of fold, no cell of the scored port) is given to fit."""

    def setUp(self):
        FakeModel.log = []
        p = mock.patch.object(model, "hgb", lambda **kw: FakeModel()); p.start(); self.addCleanup(p.stop)
        self.C = synth(days=6, port="AAA")[0] + synth(days=6, port="BBB")[0] + synth(days=6, port="CCC")[0]
        self.C[5]["y"] = None                                           # a cell without an answer is never trained on
        self.lab, self.y = model.labelled(self.C)
        self.X = np.column_stack([np.arange(len(self.C)), np.ones(len(self.C))]).astype(np.float32)
        self.port = lambda rows: {self.C[i]["port"] for i in rows}

    def test_generic_scores_lopo(self):
        for name, tr, te in evalcv.folds_lopo(self.C, self.lab):
            FakeModel.log = []
            s = model.gen_scores(self.C, self.X, self.y, tr, te, [0, 1])
            self.assertEqual(len(FakeModel.log), 3)                      # two out-of-fold fits inside tr, then one on all of tr
            for f in FakeModel.log:
                self.assertFalse(set(f["rows"]) & set(te)); self.assertNotIn(name, self.port(f["rows"])); self.assertNotIn(5, f["rows"])
                self.assertEqual(f["y"], [self.y[i] for i in f["rows"]])
                for scored in f["predicted"]:                             # a fit never scores a port it was trained on
                    self.assertFalse(self.port(scored) & self.port(f["rows"]))
            self.assertEqual(sorted(FakeModel.log[-1]["rows"]), sorted(tr)); self.assertEqual(FakeModel.log[-1]["predicted"], [te])
            self.assertTrue(np.isfinite(s[self.lab]).all()); self.assertTrue(np.isnan(s[5]))

    def test_generic_scores_forward(self):
        folds = list(evalcv.folds_fwd(self.C, self.lab, bounds=["2026-01-21", "2026-02-11"]))
        self.assertEqual(len(folds), 2)
        for b, tr, te in folds:
            FakeModel.log = []
            model.gen_scores(self.C, self.X, self.y, tr, te, [0, 1])
            for f in FakeModel.log:
                self.assertTrue(all(self.C[i]["date"] < b for i in f["rows"])); self.assertFalse(set(f["rows"]) & set(te))

    def test_fusion_fits_only_on_the_training_cells(self):
        ch = evalcv.changed_flags(self.C, self.lab)
        for name, tr, te in evalcv.folds_lopo(self.C, self.lab):
            s = np.where(np.arange(len(self.C)) % 7 == 3, np.nan, 0.0)   # training cells without a generic score are dropped
            FakeModel.log = []; P = {}
            sets = {"hist": self.X, "opt": self.X, "opt+diff+S1": self.X}
            fuse.stack_fold(sets, self.y, s, ch, tr, te, P)
            self.assertEqual(sorted(P), [("hist", 1), ("opt", 1), ("opt+diff+S1", 1), ("opt+diff+S1", 3)])
            want = [i for i in tr if not np.isnan(s[i])]
            for f in FakeModel.log:
                self.assertEqual(f["rows"], want); self.assertFalse(set(f["rows"]) & set(te)); self.assertEqual(f["predicted"], [te])
                self.assertEqual(f["y"], [self.y[i] for i in want])
            self.assertEqual(FakeModel.log[0]["w"], [1.0] * len(want))
            self.assertEqual(FakeModel.log[-1]["w"], [3.0 if ch[i] == 1 else 1.0 for i in want])
            for p in P.values():                                          # a probability only on the fold's test cells
                self.assertTrue((p[te] == 0.75).all()); self.assertTrue(np.isnan(np.delete(p, te)).all())


@unittest.skipUnless(np is not None, "numpy not installed")
class ResamplingUnitTest(unittest.TestCase):
    """The intervals resample whole scenes (port x day), and the difference is paired on the same resample."""

    def cells(self):
        # 3 scenes of 8 cells; inside every scene half of the cells are right (p) -> any resample of whole scenes gives 0.5
        C = [dict(port="A", date=f"2026-01-0{d}", section=str(k), y=k % 2) for d in (1, 2, 3) for k in range(8)]
        y = np.array([c["y"] for c in C])
        right = np.array([k < 4 for d in range(3) for k in range(8)])
        p = np.where(right, y, 1 - y).astype(float)
        return C, y, p, right

    def test_scenes_are_the_unit(self):
        C, y, p, right = self.cells(); idx = list(range(len(C)))
        self.assertEqual(evalcv.boot_ci(C, idx, y, p, B=300), (0.5, 0.5))       # resampling cells would give a spread around 0.5
        # scenes that are wholly right or wholly wrong: only multiples of 1/3 can come out, and both ends are reached
        q = np.where(np.array([c["date"] == "2026-01-01" for c in C]), y, 1 - y).astype(float)
        lo, hi = evalcv.boot_ci(C, idx, y, q, B=300)
        self.assertEqual((lo, hi), (0.0, 1.0))
        r = final.acc_ci(C, y, idx, q, B=300)
        self.assertEqual((r["acc"], r["ci"]), (0.333, [0.0, 1.0]))

    def test_difference_is_paired(self):
        C, y, p, right = self.cells(); idx = list(range(len(C)))
        # q is right on the same cells as p except one fewer per scene: on every resample of scenes the difference is 1/8
        fewer = right & ~np.array([k == 0 for d in range(3) for k in range(8)])
        q = np.where(fewer, y, 1 - y).astype(float)
        d = final.diff_ci(C, y, idx, p, q, B=300)
        self.assertEqual((d["diff"], d["ci"]), (0.125, [0.125, 0.125]))           # unpaired resamples would not be constant
        # per-scene differences +1, 0, -1: the paired difference moves with the scenes drawn
        a = np.where(np.array([c["date"] != "2026-01-03" for c in C]), y, 1 - y).astype(float)
        b = np.where(np.array([c["date"] != "2026-01-01" for c in C]), y, 1 - y).astype(float)
        d = final.diff_ci(C, y, idx, a, b, B=300)
        self.assertEqual(d["diff"], 0.0); self.assertEqual(d["ci"], [-1.0, 1.0])

    def test_port_level_interval(self):
        CC = [dict(port=P, date=f"2026-01-{d:02d}", section="s", y=d % 2) for P in "AB" for d in range(1, 11)]
        yy = np.array([c["y"] for c in CC]); pp = np.where(np.array([c["port"] == "A" for c in CC]), yy, 1 - yy).astype(float)
        idx = list(range(len(CC)))                                                    # port A all right, port B all wrong
        self.assertEqual(final.port_ci(CC, yy, idx, pp, B=300), [0.0, 1.0])
        self.assertEqual(final.port_ci(CC, yy, idx, pp, pp, B=300), [0.0, 0.0])
        lo, hi = evalcv.boot_ci(CC, idx, yy, pp, B=300)                               # scene level: 20 scenes, narrower
        self.assertTrue(0.0 < lo < 0.5 < hi < 1.0)


class NoPublishedFileIsTouchedTest(unittest.TestCase):
    def test_working_directory_cannot_be_inside_the_repository(self):
        for inside in (common.SAT / "eval", common.SAT / "pres3-work" / "x", ROOT / "pres3-work", ROOT):
            with self.assertRaises(SystemExit):
                common.work_dir(str(inside))
        self.assertFalse((common.SAT / "pres3-work").exists()); self.assertFalse((ROOT / "pres3-work").exists())
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(common.work_dir(str(Path(d) / "w")), (Path(d) / "w").resolve())

    def test_files_made_from_another_cell_list_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cells = Path(d) / "cells_AAA.json"; cells.write_text('[{"y": 1}]', encoding="utf-8")
            k1 = common.cells_key(d, ["AAA"])
            self.assertEqual(k1, common.cells_key(d, ["AAA"]))
            cells.write_text('[{"y": 0}]', encoding="utf-8")                 # build was run again with other answers
            self.assertNotEqual(k1, common.cells_key(d, ["AAA"]))
            (Path(d) / "cells_BBB.json").write_text("[]", encoding="utf-8")
            self.assertNotEqual(common.cells_key(d, ["AAA", "BBB"]), common.cells_key(d, ["BBB", "AAA"]))

    # confirm-1 (design 設計書_確認回し §6): the files of the confirm series are meant to take the scores of pres-3. They are
    # not part of the daily collection (sat/run.py and what it calls, the existing workflows), which stays checked below.
    CONFIRM_SERIES = {"confirm.py"}

    def test_not_wired_into_the_daily_collection(self):
        # pres3-es-probe.yml (confirm-1, PR-0) is not part of the collection: it runs only on a push to probe/pres3-es-*,
        # reads two public endpoints and writes nothing (checked below). sat-confirm.yml (PR-5) is the confirm series' own
        # workflow (tests/test_sat_pres3_confirm_run.py): the only other one that may name pres3
        flows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        for f in [f for f in flows if f.name not in ("pres3-es-probe.yml", "sat-confirm.yml")] + [p for p in sorted((ROOT / "sat").glob("*.py")) if p.name not in self.CONFIRM_SERIES] + [ROOT / "sat" / "sites.json", ROOT / "sat" / "publish.sh"]:
            self.assertNotIn("pres3", f.read_text(encoding="utf-8"), f)
        # the exempted files stay outside the daily collection: nothing that is checked above refers to them
        for f in sorted((ROOT / "sat").glob("*.py")):
            if f.name not in self.CONFIRM_SERIES and not f.name.startswith("confirm"):
                for name in self.CONFIRM_SERIES:
                    self.assertNotRegex(f.read_text(encoding="utf-8"), r"\b(import|from)\b[^\n]*\b" + name[:-3] + r"\b", f)

    def test_the_probe_workflow_is_read_only(self):
        t = (ROOT / ".github" / "workflows" / "pres3-es-probe.yml").read_text(encoding="utf-8")
        self.assertIn("branches: ['probe/pres3-es-*']", t)
        self.assertIn("contents: read", t)
        for word in ("schedule", "workflow_dispatch", "pull_request", "contents: write", "git push", "secrets.", "CDSE"):
            self.assertNotIn(word, t.split("jobs:")[1] if word == "CDSE" else t, word)
        # it runs the probe and pip, nothing of the collection (sat.run, publish.sh, the other sat / pres3 modules)
        jobs = t.split("jobs:")[1]
        calls = re.findall(r"(?<![\w/.-])python3?(?:\.\d+)?\s+([^\n|;&)]*)", jobs)                  # every python command line
        self.assertEqual(sorted({re.match(r"\s*-m\s+([\w.]+)", c).group(1) if re.match(r"\s*-m\s+", c) else c.strip() for c in calls}),
                         ["pip", "sat.pres3.es_probe"], calls)
        for word in ("sat.run", "sat/run", "run.py", "publish.sh", "sat.confirm", "pres3.daily", "pres3.fetch", "docs", "git "):
            self.assertFalse(word in jobs, word)
        # and the probe script writes no file (and so nothing under docs/): it only reads sites.json and inputs/scenes.csv
        src = (ROOT / "sat" / "pres3" / "es_probe.py").read_text(encoding="utf-8")
        for word in ("docs", "common.SAT", ".write(", "write_text", "write_bytes", "savez", "json.dump(", "os.replace", "os.rename",
                     "mkdir", "shutil", "tofile", "to_csv", "NamedTemporaryFile"):
            self.assertFalse(word in src, word)
        opens = re.findall(r"(?<![\w.])open\(([^\n]*)", src)                                     # the built-in open (not rasterio.open)
        self.assertEqual(len(opens), 2, opens)
        for args in opens:
            self.assertNotRegex(args, r"""['"]\s*[rwax+bt]*[wax+][rwax+bt]*\s*['"]""", args)        # no write / append / create mode

    def test_probe_cog_path_from_the_scene_id(self):
        from sat.pres3 import es_probe
        self.assertEqual(es_probe.cog_href("S2C_51STV_20261111_0_L2A", "red"),
                         "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/51/S/TV/2026/11/S2C_51STV_20261111_0_L2A/B04.tif")
        self.assertTrue(es_probe.cog_href("S2A_49QDD_20230706_0_L2A", "scl").endswith("/49/Q/DD/2023/7/S2A_49QDD_20230706_0_L2A/SCL.tif"))
        self.assertRaises(ValueError, es_probe.cog_href, "S2A_MSIL2A_20230706", "red")

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

    def test_supplement_matches_the_cells(self):
        for fold, col in (("lopo", "p_pres3_lopo"), ("fwd", "p_pres3_fwd")):
            sup = self.sheet["supplement"][fold]
            self.check(sup["excl_zjg"]["pres3"], col, lambda r: r["site_id"] != "ZJG")
            self.check(sup["excl_zjg"]["current_s1"], "p_current_s1", lambda r, col=col: r["site_id"] != "ZJG" and r[col] != "")
            self.check(sup["zjg"]["pres3"], col, lambda r: r["site_id"] == "ZJG")
            self.assertEqual(sup["no_s1"]["n"], self.sheet[fold]["pres3"]["n"])
            late = sup["s1_after_s2_day"]
            self.assertEqual(sum(v[0] for v in late["by_port"].values()), late["cells"]); self.assertEqual(late["of"], self.sheet[fold]["pres3"]["n"])
            self.assertGreater(late["cells"], 0)            # the S1 columns do include scenes later than the S2 day: say so
            lo, hi = sup["port_bootstrap"]["pres3_ci"]; slo, shi = self.sheet[fold]["pres3"]["ci"]
            self.assertTrue(lo <= slo and shi <= hi)        # wider than the scene-level interval
        text = "".join(self.sheet["limits"])
        for word in ("湛江", "標本外でない", "dt_h > 0", "手に入らない", "74"):
            self.assertIn(word, text)

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
        self.assertGreaterEqual(len(self.sheet["limits"]), 12)

    def test_other_evaluation_sheets_are_not_rewritten_by_this_package(self):
        src = "".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "sat" / "pres3").glob("*.py")))
        for name in ("presence_snapshot", "occ2.json", "meta.json", "quay_pres1", "ship_counts", "quay_counts"):
            self.assertNotIn(name, src)
        self.assertEqual(src.count("'presence.json'"), 1)                # read once (labels.lookup), never opened for writing
        self.assertIn("json.load(open(common.SAT / 'eval' / 'presence.json'))", src)


@unittest.skipUnless(np is not None, "numpy not installed")
class TriageTest(unittest.TestCase):
    """confirm-1: which cells are settled automatically and which go to the owner (sat/pres3/triage.py)."""

    HEAD = "site_id,date,section_id,in_eval,y,changed,p_current_s1,p_gen_lopo,p_hist_lopo,p_pres3_lopo,p_hist_fwd,p_pres3_fwd,p_opt_lopo,p_opt_fwd"

    def cells(self, rows):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "oos.csv"
            f.write_text(self.HEAD + "\n" + "".join(f"{port},{date},{sec},1,{y},0,,,,0.5,,,{p},{p}\n" for port, date, sec, y, p in rows), encoding="utf-8")
            return triage.load(f)

    def test_rule_v1_boundaries(self):
        rows = [("QDH", "2026-01-01", "a", 0, 0.01), ("QDH", "2026-01-11", "a", 0, 0.01),    # 0 and 1 earlier answers: routed
                ("QDH", "2026-01-21", "a", 0, 0.03),                                          # 2 answers, p = 0.03 exactly: auto
                ("QDH", "2026-01-31", "a", 0, 0.030001),                                      # just inside the band: routed
                ("QDH", "2026-05-31", "a", 1, 0.97),                                          # 120 days after the previous answer: auto
                ("QDH", "2026-09-29", "a", 1, 0.99),                                          # 121 days: routed
                ("ZJG", "2026-01-01", "b", 0, 0.0), ("ZJG", "2026-01-02", "b", 0, 0.0), ("ZJG", "2026-01-03", "b", 0, 0.0),   # a port that is off
                ("QDH", "2026-01-21", "c", 1, 1.0)]                                           # another section: its own history
        D = self.cells(rows); a = triage.auto_v1(D, D["p_opt_lopo"])
        self.assertEqual(a.tolist(), [False, False, True, False, True, False, False, False, False, False])
        self.assertEqual(D["n_prev"].tolist(), [0, 1, 2, 3, 4, 5, 0, 1, 2, 0])
        self.assertEqual(D["age"][[2, 4, 5]].tolist(), [10, 120, 121])
        self.assertEqual(triage.auto_v1(D, D["p_opt_lopo"], ports=False).tolist()[6:9], [False, False, True])
        self.assertEqual(triage.auto_v1(D, D["p_opt_lopo"], answers=False, age=False).tolist()[:2], [True, True])

    def test_same_day_answers_are_not_earlier_answers(self):
        D = self.cells([("QDH", "2026-01-01", "a", 0, 0.0), ("QDH", "2026-01-01", "a", 0, 0.0), ("QDH", "2026-01-02", "a", 0, 0.0)])
        self.assertEqual(D["n_prev"].tolist(), [0, 0, 2])

    def test_line_counts_routed_cells_and_never_drops_one(self):
        D = self.cells([("QDH", f"2026-01-{d:02d}", "a", y, p) for d, y, p in ((1, 0, 0.0), (2, 0, 0.4), (3, 1, 0.99), (4, 0, 0.98), (5, 1, 0.2))])
        p = D["p_opt_lopo"]; r = triage.line(D, triage.eval_cells(D, p), triage.band(p, 0.03), p, B=50)
        self.assertEqual((r["n"], r["routed"], r["auto"], r["auto_errors"]), (5, 2, 3, 1))
        self.assertEqual((r["auto_acc"], r["routed_model_acc"], r["combined_if_owner_right"]), (0.667, 0.5, 0.8))
        self.assertEqual(len(r["auto_acc_ci"]), 2)

    def test_nested_choice_stops_at_the_first_threshold_that_fails(self):
        p = np.array([0.01, 0.025, 0.04, 0.5]); y = np.array([0, 1, 0, 1]); sel = np.ones(4, bool)
        self.assertEqual(triage.pick(p, y, sel, 0.97), 0.02)          # 0.03 lets the wrong cell in; 0.05 is not tried after it

    def test_design_numbers_from_the_published_cells(self):
        """The design's tables (confirm-design.md 2.3, 2.5, 2.6, 5) from docs/data/sat/eval/pres3_oos.csv. The numbers belong to
        the 1,531 cells of 2026-11-19: when the file is published again with more cells this check is skipped."""
        D = triage.load()
        if len(D["y"]) != 1531: self.skipTest("pres3_oos.csv is not the 2026-11-19 file the design was computed from")
        T = triage.tables(D, ci=False); pick = lambda r: (r["routed"], r["auto"], r["auto_acc"], r["auto_errors"], r["routed_model_acc"], r["combined_if_owner_right"])
        th = T["2.3 thresholds (pres-3o)"]
        self.assertEqual(pick(th["lopo"]["0.03"]), (442, 295, 0.986, 4, 0.792, 0.995))
        self.assertEqual(pick(th["lopo"]["0.02"]), (487, 250, 0.992, 2, 0.807, 0.997))
        self.assertEqual(pick(th["lopo"]["0.05"]), (373, 364, 0.967, 12, 0.775, 0.984))
        self.assertEqual(pick(th["fwd"]["0.03"]), (212, 304, 0.980, 6, 0.755, 0.988))
        self.assertEqual(pick(th["fwd"]["0.5"]), (0, 516, 0.888, 58, None, 0.888))
        ad = T["2.3 thresholds (adopted, with S1: an upper reference)"]
        self.assertEqual((ad["lopo"]["0.05"]["auto_acc"], ad["fwd"]["0.05"]["auto_acc"], ad["lopo"]["0.1"]["auto_acc"], ad["fwd"]["0.1"]["auto_acc"]), (0.986, 0.981, 0.958, 0.963))
        v = T["2.5 rule v1"]
        self.assertEqual(pick(v["lopo"]), (503, 234, 0.991, 2, 0.813, 0.997)); self.assertEqual(pick(v["fwd"]), (273, 243, 0.992, 2, 0.795, 0.996))
        self.assertEqual((v["lopo"]["routed_frac"], v["fwd"]["routed_frac"]), (0.682, 0.529))
        self.assertEqual([(v["lopo"]["changed"][k], v["fwd"]["changed"][k]) for k in ("n", "routed", "auto_errors")], [(96, 66), (93, 63), (2, 2)])
        self.assertEqual((v["lopo"]["same"]["auto"], v["lopo"]["same"]["auto_errors"], v["fwd"]["same"]["auto"], v["fwd"]["same"]["auto_errors"]), (231, 0, 240, 0))
        self.assertEqual({P: (r["auto"], r["auto_errors"]) for P, r in v["lopo"]["by_port"].items()},
                         {"ZJG": (0, 0), "SYL": (0, 0), "SLP": (65, 1), "QGZ": (23, 0), "ZDH": (0, 0), "QJG": (24, 0), "QDH": (49, 1), "XSN": (0, 0), "LSN": (73, 0)})
        self.assertEqual([(r["routed_frac"], r["auto_acc"]) for r in v["fwd"]["steps"].values()], [(0.411, 0.980), (0.434, 0.983), (0.504, 0.988), (0.529, 0.992)])
        for fold in ("lopo", "fwd"):        # no cell is left out: off ports and cells without history are routed, not dropped
            self.assertEqual(sum(r["n"] for r in v[fold]["by_port"].values()), v[fold]["n"])
            self.assertEqual(v[fold]["routed"] + v[fold]["auto"], T["cells"]["eval" if fold == "lopo" else "eval_fwd"])
        ch = T["2.4 changed cells at 0.05/0.95"]
        self.assertEqual((ch["lopo"]["auto_errors_on_changed"], ch["fwd"]["auto_errors_on_changed"]), ([11, 12], [10, 11]))
        n = T["2.6 nested selection"]
        self.assertEqual([(r["n"], r["routed_frac"], r["auto_acc"], r["auto_errors"]) for r in n[:4]],
                         [(737, 0.573, 0.965, 11), (329, 0.331, 0.964, 8), (195, 0.323, 0.985, 2), (172, 0.267, 0.960, 5)])
        self.assertEqual(T["5 generic judgement, confident band"], {"0.03": {"absent_n": 104, "absent_acc": 0.875, "present_n": 0}, "0.05": {"absent_n": 194, "absent_acc": 0.851, "present_n": 0}})

    def test_intervals_are_close_to_the_design(self):
        D = triage.load()
        if len(D["y"]) != 1531: self.skipTest("pres3_oos.csv is not the 2026-11-19 file the design was computed from")
        p = D["p_opt_lopo"]; r = triage.line(D, triage.eval_cells(D, p), triage.auto_v1(D, p), p, B=2000)
        for got, design in ((r["auto_acc_ci"], (0.979, 1.000)), (r["routed_ci"], (0.598, 0.775))):
            for g, d in zip(got, design): self.assertAlmostEqual(g, d, delta=0.011)

    def test_the_new_columns_do_not_touch_the_old_ones(self):
        self.assertEqual(final.OOS_COLUMNS[-2:], ["p_opt_lopo", "p_opt_fwd"])
        with open(EVAL / "pres3_oos.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0]), final.OOS_COLUMNS)
        self.assertTrue(all(r["p_opt_lopo"] != "" and (r["p_opt_fwd"] != "") == (r["p_pres3_fwd"] != "") for r in rows))
        self.assertTrue(all(len(r[c].split(".")[1]) == 4 for r in rows for c in final.OOS_COLUMNS[6:12] if r[c] != ""))     # the old columns keep 4 places

    OLD_COLUMNS_ROWS = 1531
    OLD_COLUMNS_SHA256 = "8e1858fe8447f5553f0476e81501634f0dd1bb2f6bc8b9641dd81812e1f87523"

    def test_the_old_columns_are_byte_for_byte_the_2026_11_19_file(self):
        """The first 12 columns (header + 1,531 rows) of pres3_oos.csv as published on 2026-11-19, before the two columns were
        added. A file with other rows (a row added or removed) FAILS here, it is not skipped: a new publication with more cells
        changes this on purpose, and then OLD_COLUMNS_ROWS and OLD_COLUMNS_SHA256 are replaced with the new file's."""
        with open(EVAL / "pres3_oos.csv", "rb") as fh:
            lines = fh.read().split(b"\n")
        self.assert_old_columns(lines)
        changed = list(lines); c = changed[1].split(b","); c[7] = b"%.4f" % (float(c[7]) + 0.0001); changed[1] = b",".join(c)
        self.assertNotEqual(self.old_columns_hash(changed), self.old_columns_hash(lines))       # one value moved by 0.0001 is seen
        # the check has teeth: a row added, a row removed, a value moved each fail it (a different file is not skipped)
        for bad in (lines[:-1] + [lines[-2], b""], lines[:1] + lines[2:], changed):
            with self.assertRaises(AssertionError): self.assert_old_columns(bad)
        tail = list(lines); c = tail[1].split(b","); c[12] = b"0.5"; tail[1] = b",".join(c)
        self.assertEqual(self.old_columns_hash(tail), self.old_columns_hash(lines))             # the new columns are not part of it

    def assert_old_columns(self, lines):
        self.assertEqual(len(lines) - 2, self.OLD_COLUMNS_ROWS, "pres3_oos.csv has other rows than the 2026-11-19 file: "
                         "if this is a new publication on purpose, replace OLD_COLUMNS_ROWS and OLD_COLUMNS_SHA256")
        self.assertEqual(self.old_columns_hash(lines), self.OLD_COLUMNS_SHA256)

    @staticmethod
    def old_columns_hash(lines):
        import hashlib
        return hashlib.sha256(b"".join(b",".join(l.split(b",")[:12]) + b"\n" for l in lines[:-1])).hexdigest()

    def test_publish_cells_only_leaves_the_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "eval"; out.mkdir(); (out / "pres3.json").write_text("{}", encoding="utf-8")
            src = Path(d) / "oos.csv"; src.write_text("site_id,date\nAAA,2026-01-01\n", encoding="utf-8")
            final.publish({}, src, out, "", cells_only=True)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["pres3.json", "pres3_oos.csv"])
            self.assertEqual((out / "pres3.json").read_text(encoding="utf-8"), "{}")
            self.assertEqual((out / "pres3_oos.csv").read_bytes(), src.read_bytes())

if __name__ == "__main__":
    unittest.main()
