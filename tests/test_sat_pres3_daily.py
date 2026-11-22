"""confirm-1 PR-2 (sat/pres3/daily.py): pres-3o one day at a time. Synthetic data only, no network, no fetched window.

What is guarded:
  1. the score of a day does not change when the images or the answers of LATER days (or the answers of the day itself)
     are changed, and does change with an earlier answer or an earlier image (so the check has teeth)
  2. scoring day by day from a store cut at each day gives the same numbers as scoring all the days at once
  3. the patch store is addressed by (port, section, date): the order of writing, a section added later or a record put
     again do not move the other records; no answer is stored in it
  4. the trees stored as arrays are predicted by numpy as scikit-learn predicts them (checked against scikit-learn where
     it is installed; the Tests workflow does not install it), and the model file is read back without pickle
  5. the features of the day's cells computed alone equal the rows of the research code's batch computation
  6. cells without an image get no score; the columns of the score table; nothing is written inside the repository
numpy only; skipped where numpy is missing (the collect workflows).
"""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat.pres3 import daily, feats, model

ROOT = Path(__file__).resolve().parent.parent
OFFS = None if np is None else np.arange(-80, 125, 5.0)
SECS = ("s:L:00", "s:L:01", "s:L:02")


def day_of(k):
    return f"2026-{1 + k // 28:02d}-{1 + k % 28:02d}"


def patch(rng, ship, cloud=False):
    """A quay: land (bright) up to 0 m, water after it; a ship is a bright block 5..40 m offshore over half the quay."""
    # (the shadow matters only to the fit test: without a dark pixel that comes and goes one feature is empty in every row,
    # and scikit-learn 1.9 cannot bin an empty column when sample weights are given)
    p = np.zeros((24, 41, 5), np.float32); p[:, OFFS < 0, :] = 0.20; p[:, OFFS >= 0, :] = 0.04
    if ship:
        p[4:16, (OFFS >= 5) & (OFFS <= 40), :] = 0.16
        p[4:16, (OFFS >= 45) & (OFFS <= 55), :] = 0.01            # and its shadow on the water, darker than the water
    if cloud: p += 0.3
    return p + rng.normal(0, 0.003, p.shape).astype(np.float32)


def world(n_days=14, seed=0, ships=None):
    """(cells, patches, answers): three sections on n_days days. ships: (section, day number) -> bool."""
    rng = np.random.default_rng(seed); C = []; P = []; ans = {}
    for k in range(n_days):
        for j, s in enumerate(SECS):
            here = ships[(s, k)] if ships and (s, k) in ships else (j == 0 and k % 3 == 0) or (j == 1)
            C.append(dict(port="AAA", date=day_of(k), section=s, kind="quay", sector="A", length_m=240, cloud=0.0, shadow=0.0, cirrus=0.0,
                          nodata=0.0, tile_cloud=1.0, scene=f"S2X_00AAA_{k:02d}", time=day_of(k) + "T02:00:00Z"))
            P.append(patch(rng, here)); ans[(s, day_of(k))] = int(here)
    return C, np.stack(P).astype(np.float16), ans


def tree(*nodes):
    """nodes: (value, feature, threshold, missing_go_to_left, left, right, is_leaf)."""
    return [np.array(n) for n in zip(*nodes)]


def trees(*ts, baseline=0.0):
    cols = [np.concatenate([t[k] for t in ts]) for k in range(7)]
    T = dict(zip(daily.NODE_FIELDS, cols)); T["start"] = np.cumsum([0] + [len(t[0]) for t in ts]); T["baseline"] = np.array([baseline])
    return T


def features(P, C, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # numpy's all-NaN slices (cells without a reference)
        return feats.features(np.asarray(P, np.float32), OFFS, C, **kw)


def toy_model(names):
    """gen: present when the brightest blob over the water is bright. stack: the generic score, pulled by the previous answer,
    by the share of earlier present answers and by the distance from the earlier absent days (all history features)."""
    f = names.index("blob_absd_max_W")                 # the zone 10..70 m offshore: water only
    gen = trees(tree((0, f, 0.05, 0, 1, 2, 0), (-2.0, 0, 0, 0, 0, 0, 1), (2.0, 0, 0, 0, 0, 0, 1)))
    stack = trees(tree((0, 4, 0.0, 0, 1, 2, 0), (-1.5, 0, 0, 0, 0, 0, 1), (1.5, 0, 0, 0, 0, 0, 1)),      # column 4: the generic score
                  tree((0, 3, 0.5, 1, 1, 2, 0), (-0.7, 0, 0, 0, 0, 0, 1), (0.7, 0, 0, 0, 0, 0, 1)),      # column 3: the previous answer
                  tree((0, 2, 0.5, 1, 1, 2, 0), (-0.3, 0, 0, 0, 0, 0, 1), (0.3, 0, 0, 0, 0, 0, 1)),      # column 2: (present + 1) / (n + 2)
                  tree((0, 5, 4.5, 1, 1, 2, 0), (0.05, 0, 0, 0, 0, 0, 1), (-0.05, 0, 0, 0, 0, 0, 1)),    # column 5: days since the previous answer
                  # the generic score minus its mean over the earlier ABSENT days (an exemplar feature)
                  tree((0, 6 + 1 + len(model.phys_cols(names)), -1.0, 0, 1, 2, 0), (-0.4, 0, 0, 0, 0, 0, 1), (0.4, 0, 0, 0, 0, 0, 1)))
    return dict(meta=dict(kind=daily.KIND, model_version="pres3o-test", cutoff="2026-01-01"), names=list(names), phys=model.phys_cols(names),
                gen=gen, stack=stack, train_scores={})


@unittest.skipUnless(np is not None, "numpy not installed")
class DailyScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C, cls.P, cls.ans = world()
        _, cls.names, _ = features(cls.P[:6], cls.C[:6])
        cls.B = toy_model(cls.names); cls.d = day_of(10)
        cls.base = daily.score_days("AAA", cls.C, cls.P, OFFS, cls.ans, cls.B, [cls.d])

    def score(self, C=None, P=None, ans=None, days=None, **kw):
        return daily.score_days("AAA", self.C if C is None else C, self.P if P is None else P, OFFS, self.ans if ans is None else ans, self.B,
                                days or [self.d], **kw)

    def test_the_toy_scores_use_the_image_and_the_history(self):
        p = {r["section_id"]: float(r["p"]) for r in self.base}
        self.assertEqual(len(self.base), 3); self.assertTrue(all(r["status"] == "" for r in self.base))
        self.assertGreater(p[SECS[1]], 0.85); self.assertLess(p[SECS[2]], 0.15)           # always present / always absent
        self.assertEqual([r["n_answers"] for r in self.base], [10, 10, 10])
        self.assertEqual({r["answers_through"] for r in self.base}, {day_of(9)})

    def test_later_images_and_answers_do_not_change_the_day(self):
        P = self.P.copy(); late = [i for i, c in enumerate(self.C) if c["date"] > self.d]
        P[late] = np.float16(0.5)                                                         # every later image replaced
        ans = {k: (1 - v if k[1] >= self.d else v) for k, v in self.ans.items()}         # the day's own and every later answer flipped
        self.assertEqual(self.score(P=P, ans=ans), self.base)
        cut = [i for i, c in enumerate(self.C) if c["date"] <= self.d]                    # and with the later days not in the store at all
        self.assertEqual(self.score(C=[self.C[i] for i in cut], P=self.P[cut]), self.base)
        ans = {k: v for k, v in self.ans.items() if k[1] < self.d}                        # and with no later answer given
        self.assertEqual(self.score(ans=ans), self.base)

    def test_an_earlier_answer_and_an_earlier_image_do_change_it(self):
        ans = dict(self.ans); ans[(SECS[2], day_of(9))] = 1                               # the previous answer of the absent section
        got = self.score(ans=ans); self.assertNotEqual(got[2]["p"], self.base[2]["p"]); self.assertEqual(got[:2], self.base[:2])   # its own section only
        P = self.P.copy(); early = [i for i, c in enumerate(self.C) if c["date"] < self.d and c["section"] == SECS[2] and c["date"] >= day_of(2)]
        P[early] = np.stack([patch(np.random.default_rng(9), True) for _ in early]).astype(np.float16)       # ships on its earlier absent days
        self.assertNotEqual(self.score(P=P)[2]["p"], self.base[2]["p"])

    def test_cannot_tell_and_answers_without_an_image_are_not_answers(self):
        ans = dict(self.ans); ans[(SECS[0], day_of(9))] = None; ans[(SECS[0], "2025-12-31")] = 1     # no patch on 2025-12-31
        r = self.score(ans=ans)[0]
        self.assertEqual((r["n_answers"], r["answers_through"]), (9, day_of(8)))

    def test_a_day_before_the_cutoff_is_marked_in_the_table(self):
        # the coefficients were fitted on the answers before the cutoff: a score of such a day is not out of sample
        self.assertEqual({r["before_cutoff"] for r in self.base}, {""})                   # cutoff 2026-01-01, the day 2026-01-11
        for cut, want in ((self.d, ""), (day_of(11), "1"), ("2027-01-01", "1")):
            B = dict(self.B, meta=dict(self.B["meta"], cutoff=cut))
            rows = daily.score_days("AAA", self.C, self.P, OFFS, self.ans, B, [self.d])
            self.assertEqual({r["before_cutoff"] for r in rows}, {want}, cut)
            self.assertEqual([{k: v for k, v in r.items() if k != "before_cutoff"} for r in rows],
                             [{k: v for k, v in r.items() if k != "before_cutoff"} for r in self.base])      # the score itself is the same
            with tempfile.TemporaryDirectory() as d:
                daily.write_scores(Path(d) / "s.csv", rows)
                with open(Path(d) / "s.csv", newline="", encoding="utf-8") as fh: self.assertEqual({r["before_cutoff"] for r in csv.DictReader(fh)}, {want})

    def test_n_answers_is_in_the_score_table(self):
        # an answer of a day without a patch is in the answer file but not used by the judge: n_answers does not count it
        ans = dict(self.ans); ans[(SECS[0], "2025-12-30")] = 1; ans[(SECS[0], "2025-12-31")] = 0
        rows = self.score(ans=ans)
        with tempfile.TemporaryDirectory() as d:
            daily.write_scores(Path(d) / "s.csv", rows)
            with open(Path(d) / "s.csv", newline="", encoding="utf-8") as fh: got = list(csv.DictReader(fh))
        self.assertEqual([r["n_answers"] for r in got], ["10", "10", "10"])
        rows = self.score(ans={k: v for k, v in self.ans.items() if not (k[0] == SECS[1] and k[1] >= day_of(4))})
        self.assertEqual([r["n_answers"] for r in rows], [10, 4, 10])

    def test_day_by_day_equals_all_at_once(self):
        days = [day_of(k) for k in range(7, 14)]; at_once = self.score(days=days); step = []
        for d in days:
            cut = [i for i, c in enumerate(self.C) if c["date"] <= d]
            step += self.score(C=[self.C[i] for i in cut], P=self.P[cut], ans={k: v for k, v in self.ans.items() if k[1] < d}, days=[d])
        self.assertEqual(step, at_once); self.assertEqual(len(at_once), 21)

    def test_a_training_cell_keeps_its_out_of_fold_score(self):
        B = dict(self.B, train_scores={f"AAA|{SECS[2]}|{day_of(k)}": 3.0 for k in range(9)})     # the earlier absent days scored 'present'
        got = daily.score_days("AAA", self.C, self.P, OFFS, self.ans, B, [self.d])
        self.assertNotEqual(got[2]["p"], self.base[2]["p"]); self.assertEqual(got[:2], self.base[:2])

    def test_no_image_no_score(self):
        C = [dict(c) for c in self.C]; i = next(k for k, c in enumerate(C) if c["date"] == self.d and c["section"] == SECS[0])
        C[i]["cloud"] = 0.15; C[i]["shadow"] = 0.06
        r = self.score(C=C)
        self.assertEqual((r[0]["p"], r[0]["status"], r[0]["cloud_shadow"], r[0]["nodata"], r[0]["n_reference_days"]), ("", "no_image", "0.2100", "0.0000", ""))
        self.assertEqual(r[0]["n_answers"], 10)                                           # the history is still reported
        self.assertEqual([x["p"] != "" for x in r], [False, True, True])
        C[i]["cloud"] = 0.0; C[i]["shadow"] = 0.0; C[i]["nodata"] = 0.01
        self.assertEqual(self.score(C=C)[0]["status"], "no_image")
        r = self.score(C=C, all_cells=True)[0]; self.assertNotEqual(r["p"], ""); self.assertEqual(r["status"], "no_image")

    def test_few_reference_days(self):
        r = self.score(days=[day_of(3)])
        self.assertTrue(all(x["p"] == "" and x["status"] == "" and x["n_reference_days"] == 3 for x in r))      # not enough input: no score, not "no image"
        self.assertTrue(all(x["p"] != "" and x["n_reference_days"] == 6 for x in self.score(days=[day_of(6)])))
        self.assertTrue(all(x["p"] != "" for x in self.score(days=[day_of(3)], all_cells=True)))

    def test_only_the_asked_sections(self):
        r = self.score(sections={SECS[1]}); self.assertEqual([x["section_id"] for x in r], [SECS[1]]); self.assertEqual(r[0], self.base[1])

    def test_another_model_is_refused(self):
        with self.assertRaises(SystemExit): daily.score_days("AAA", self.C, self.P, OFFS, self.ans, dict(self.B, names=self.names[:-1]), [self.d])

    def test_score_table(self):
        # the columns sat/confirm.py reads (its SCORE_FIELDS), in its order
        self.assertEqual(daily.SCORE_COLUMNS[:10], ["site_id", "section_id", "date", "acquired_utc", "p", "model_version", "cloud_shadow", "nodata", "status", "answers_through"])
        # then n_answers, always (not only with --diagnostics): the answers the judge used, which sat/confirm.py's rule reads
        self.assertEqual(daily.SCORE_COLUMNS[10:], ["n_answers", "before_cutoff"]); self.assertNotIn("n_answers", daily.DIAGNOSTIC_COLUMNS)
        self.assertEqual(list(self.base[0]), daily.SCORE_COLUMNS + daily.DIAGNOSTIC_COLUMNS)
        self.assertTrue(all(0 <= float(r["p"]) <= 1 and 0 <= float(r["cloud_shadow"]) <= 1 and float(r["nodata"]) == 0 for r in self.base))
        with tempfile.TemporaryDirectory() as d:
            daily.write_scores(Path(d) / "x" / "scores.csv", self.base)
            lines = (Path(d) / "x" / "scores.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(daily.SCORE_COLUMNS)); self.assertEqual(len(lines), 4)
            self.assertTrue(lines[1].startswith(f"AAA,{SECS[0]},{self.d},{self.d}T02:00:00Z,0.")); self.assertTrue(lines[1].endswith(f",pres3o-test,0.0000,0.0000,,{day_of(9)},10,"))
            daily.write_scores(Path(d) / "x" / "diag.csv", self.base, diagnostics=True)
            self.assertEqual((Path(d) / "x" / "diag.csv").read_text(encoding="utf-8").splitlines()[0], ",".join(daily.SCORE_COLUMNS + daily.DIAGNOSTIC_COLUMNS))
        with self.assertRaises(SystemExit): daily.write_scores(ROOT / "docs" / "data" / "sat" / "confirm" / "scores.csv", self.base)

    def test_features_of_the_needed_cells_equal_the_batch(self):
        P = self.P.astype(np.float32); X, names, good = features(P, self.C)
        rows = [i for i, c in enumerate(self.C) if c["section"] != SECS[1]]              # a subset of the sections, with the port's water colour
        only = {k for k, i in enumerate(rows) if self.C[i]["date"] in (self.d, day_of(8))}
        Xs, names2, _ = features(P[rows], [self.C[i] for i in rows], only=only, W=feats.water(self.P, OFFS, self.C)[rows], good=good[rows])
        self.assertEqual(names, names2)
        for k in only: np.testing.assert_array_equal(Xs[k], X[rows[k]])
        other = next(k for k in range(len(rows)) if k not in only and self.C[rows[k]]["date"] == day_of(12))
        self.assertTrue(np.isnan(Xs[other, names.index("ld_nr_area")]) and not np.isnan(X[rows[other], names.index("ld_nr_area")]))


@unittest.skipUnless(np is not None, "numpy not installed")
class PatchStoreTest(unittest.TestCase):
    def test_keyed_by_section_and_date(self):
        C, P, _ = world(n_days=40)
        with tempfile.TemporaryDirectory() as d:
            a = daily.Store(Path(d) / "a"); a.put("AAA", C, P)
            b = daily.Store(Path(d) / "b"); days = sorted({c["date"] for c in C}); rng = np.random.default_rng(1)
            for part in np.array_split(rng.permutation(len(days)), 7):            # other order, in pieces of whole days
                ix = [i for i, c in enumerate(C) if c["date"] in {days[k] for k in part}]
                ix = list(rng.permutation(ix))
                b.put("AAA", [dict(C[i], y=1, eval=True, p_cur=0.3) for i in ix], P[ix])
            Ca, Pa, offs = a.load("AAA"); Cb, Pb, _ = b.load("AAA")
            self.assertEqual(Ca, Cb); np.testing.assert_array_equal(Pa, Pb); np.testing.assert_array_equal(offs, OFFS)
            self.assertEqual([{k: c[k] for k in daily.CELL_KEYS} for c in Ca], sorted(C, key=lambda c: (c["date"], c["section"])))
            self.assertEqual(Pa.dtype, np.float16)
            self.assertEqual(set(Ca[0]), set(daily.CELL_KEYS) | {"water", "good"})       # no answer, no evaluation field in the store
            order = sorted(range(len(C)), key=lambda i: (C[i]["date"], C[i]["section"])); np.testing.assert_array_equal(Pa, P[order])
            self.assertEqual(len(a.dates("AAA")), 40); self.assertTrue(all(f.name.endswith(".c.npz") for f in a.files("AAA")))   # all clear
            Cu, Pu, _ = a.load("AAA", until=day_of(9)); self.assertEqual(max(c["date"] for c in Cu), day_of(9)); self.assertEqual(len(Cu), 30)
            self.assertRaises(ValueError, a.put, "BBB", C[:1], P[:1])
            self.assertRaises(ValueError, a.put, "AAA", C[:3], P[:3])                    # a day is written once
            z = np.load(a.files("AAA")[0], allow_pickle=False); self.assertEqual(sorted(z.files), ["cells", "patch", "shape", "version"])
            self.assertEqual(daily.Store(Path(d) / "none").load("AAA")[0], [])

    def test_the_patch_codec_is_lossless(self):
        rng = np.random.default_rng(0)
        for P in (rng.normal(0.1, 0.05, (7, 24, 41, 5)).astype(np.float16), np.zeros((1, 24, 41, 5), np.float16),
                  np.array(rng.integers(0, 65536, (3, 24, 41, 5)), np.uint16).view(np.float16)):     # every bit pattern, NaN and inf too
            got = daily.unpack_patch(np.frombuffer(daily.pack_patch(P), np.uint8), P.shape)
            np.testing.assert_array_equal(got.view(np.uint16), P.view(np.uint16))

    def test_thinning_keeps_the_scores(self):
        # cloudy cells on some days; the days before the window lose them (but keep the answered ones), and every score of a
        # day in the window is the same as from the full store: the water colour and the clear flag were stored with the day
        C, P, ans = world(n_days=60)
        rng = np.random.default_rng(5)
        for i in rng.choice(len(C), 40, replace=False): C[i]["cloud"] = 0.5
        for k in (5, 12): C = [dict(c, cloud=0.5) if c["date"] == day_of(k) and c["section"] != SECS[2] else c for c in C]   # cloudy days
        _, names, _ = features(P[:6], C[:6]); B = toy_model(names)
        with tempfile.TemporaryDirectory() as d:
            full = daily.Store(Path(d) / "full"); full.put("AAA", C, P)
            thin = daily.Store(Path(d) / "thin"); thin.put("AAA", C, P)
            ans = {k: v for k, v in ans.items() if k[1] in (day_of(3), day_of(20)) or k[1] >= "2026-03" or k == (SECS[2], day_of(5))}
            answered = set(ans)
            gone = thin.thin("AAA", "2026-03-05", answers=answered, months=1)
            self.assertTrue(gone and all(f.name[:7] < "2026-03" and f.name.endswith(".r.npz") for f in gone))
            self.assertFalse(any(f.name.endswith(".r.npz") and f.name[:7] < "2026-03" for f in thin.files("AAA")))
            Cf, Pf, _ = full.load("AAA"); Ct, Pt, _ = thin.load("AAA")
            self.assertLess(len(Ct), len(Cf))
            kept = {(c["section"], c["date"]) for c in Ct}
            self.assertTrue(answered <= kept)                                          # answered cells stay, clear or not
            self.assertTrue(all(c["good"] or (c["section"], c["date"]) in answered or c["date"] >= "2026-03" for c in Ct))
            days = [day_of(k) for k in range(56, 60)]
            want = daily.score_days("AAA", Cf, Pf, OFFS, ans, B, days)
            self.assertEqual(daily.score_days("AAA", Ct, Pt, OFFS, ans, B, days), want)
            self.assertTrue(any(r["p"] != "" for r in want))
            # without the stored flags, a thinned cloudy day would turn clear (its cloudy sections are gone): the check has teeth
            bare = [{k: c[k] for k in daily.CELL_KEYS} for c in Ct]
            g_bare = daily.day_inputs(Pt, OFFS, bare)[1]; W, g = daily.day_inputs(Pt, OFFS, Ct)
            self.assertTrue((g_bare & ~g).any())
            Wf, gf = daily.day_inputs(Pf, OFFS, Cf); at = {(c["section"], c["date"]): k for k, c in enumerate(Cf)}
            k = [at[(c["section"], c["date"])] for c in Ct]
            np.testing.assert_array_equal(W, Wf[k]); np.testing.assert_array_equal(g, gf[k])
            np.testing.assert_array_equal(Wf, feats.water(Pf, OFFS, Cf)); np.testing.assert_array_equal(gf, feats.good_day(Cf))
            self.assertEqual(thin.thin("AAA", "2026-03-05", answers=answered, months=1), [])           # a second time: nothing

    def test_store_and_model_must_be_outside_the_repository(self):
        for p in (ROOT / "docs" / "data" / "sat" / "confirm", ROOT / "x.npz"):
            with self.assertRaises(SystemExit): daily.outside_repo(p, "x")


@unittest.skipUnless(np is not None, "numpy not installed")
class ModelFileTest(unittest.TestCase):
    def test_numpy_walks_the_trees(self):
        # x0 <= 1 -> left (a leaf); else x1 <= 0 -> left leaf, else right leaf. Missing x0 goes right, missing x1 goes left.
        T = trees(tree((0, 0, 1.0, 0, 1, 2, 0), (10.0, 0, 0, 0, 0, 0, 1), (0, 1, 0.0, 1, 3, 4, 0), (20.0, 0, 0, 0, 0, 0, 1), (30.0, 0, 0, 0, 0, 0, 1)),
                  tree((0.5, 0, 0, 0, 0, 0, 1)), baseline=-1.0)
        X = np.array([[1.0, 5], [1.0000001, 0.0], [2, 0.1], [np.nan, np.nan], [np.nan, 3], [-np.inf, 0], [np.inf, np.inf]])
        np.testing.assert_allclose(daily.raw_predict(T, X), np.array([10, 20, 30, 20, 30, 10, 30]) - 0.5)
        np.testing.assert_allclose(daily.proba(T, X[:1]), 1 / (1 + np.exp(-9.5)))

    def test_model_file_round_trip_without_pickle(self):
        C, P, _ = world(n_days=2); _, names, _ = features(P, C); B = toy_model(names)
        B["train_scores"] = {"AAA|s:L:00|2026-01-01": -1.25, "AAA|s:L:01|2026-01-01": 0.5}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m" / "model.npz"; daily.save_model(f, B); np.load(f, allow_pickle=False)["meta"]; L = daily.load_model(f)
            self.assertEqual((L["meta"], L["names"], L["phys"], L["train_scores"]), (B["meta"], B["names"], B["phys"], B["train_scores"]))
            X = np.random.default_rng(0).normal(size=(50, len(names)))
            np.testing.assert_array_equal(daily.raw_predict(L["gen"], X), daily.raw_predict(B["gen"], X))
            np.testing.assert_array_equal(daily.raw_predict(L["stack"], X[:, :40]), daily.raw_predict(B["stack"], X[:, :40]))
            B["meta"]["kind"] = "other"; daily.save_model(f, B)
            with self.assertRaises(SystemExit): daily.load_model(f)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed (the Tests workflow does not install it)")
    def test_same_probabilities_as_scikit_learn(self):
        rng = np.random.default_rng(0); X = rng.normal(size=(400, 12)).astype(np.float32); X[rng.random(X.shape) < 0.15] = np.nan
        y = ((np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 3]) + rng.normal(0, 0.5, 400)) > 0).astype(int)
        for make in (lambda: model.gen_model(30), model.stack_model):
            m = make().fit(X[:300], y[:300], sample_weight=np.ones(300))
            np.testing.assert_allclose(daily.proba(daily.trees_of(m), X[300:]), m.predict_proba(X[300:])[:, 1], rtol=0, atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed (the Tests workflow does not install it)")
    def test_fit_uses_only_the_answers_before_the_cutoff(self):
        rng = np.random.default_rng(3); n = 30
        C, P, ans = world(n_days=n, ships={(s, k): bool(rng.random() < 0.5) for s in SECS for k in range(n)})     # ships come and go: no empty column
        cells = [dict(c, y=ans[(c["section"], c["date"])], port="AAA" if c["section"] != SECS[2] else "BBB") for c in C]   # two ports: out-of-fold over ports
        X, names, _ = features(P, cells); cut = day_of(22); msgs = []
        later = [dict(c, y=(1 - c["y"] if c["date"] >= cut else c["y"])) for c in cells]; X2 = X.copy(); X2[[c["date"] >= cut for c in cells]] = 7.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B1 = daily.fit(cells, X, names, cut, log=msgs.append)
            B2 = daily.fit(later, X2, names, cut, log=msgs.append)                            # the answers and the features from the cutoff on, changed
        for part in ("gen", "stack"):
            for k in daily.NODE_FIELDS: np.testing.assert_array_equal(B1[part][k], B2[part][k])
        self.assertEqual(B1["train_scores"], B2["train_scores"]); self.assertEqual(len(B1["train_scores"]), 66)
        self.assertEqual((B1["meta"]["n_train"], B1["meta"]["cutoff"], B1["meta"]["model_version"], B1["meta"]["last_answer_date"]), (66, cut, "pres3o-202601", day_of(21)))
        # and the fitted model scores a later day through the stored arrays
        A = dict(B1, meta=dict(B1["meta"]))
        with tempfile.TemporaryDirectory() as d:
            daily.save_model(Path(d) / "m.npz", A); L = daily.load_model(Path(d) / "m.npz")
        Cp = [c for c in cells if c["port"] == "AAA"]; Pp = P[[i for i, c in enumerate(cells) if c["port"] == "AAA"]]
        r = daily.score_days("AAA", Cp, Pp, OFFS, {k: v for k, v in ans.items() if k[0] != SECS[2]}, L, [day_of(25)])
        self.assertEqual(len(r), 2); self.assertTrue(all(0 < float(x["p"]) < 1 and x["model_version"] == "pres3o-202601" for x in r))
        self.assertTrue(B1["meta"]["scikit_learn"])


@unittest.skipUnless(np is not None, "numpy not installed")
class EmptyColumnTest(unittest.TestCase):
    """A4: scikit-learn 1.9 stops on a column missing in every training row when sample weights are given (the fusion)."""

    def test_fill_empty(self):
        X = np.array([[1.0, np.nan, np.nan], [2.0, np.nan, 3.0]])
        got = model.fill_empty(X); np.testing.assert_array_equal(got, [[1.0, 0.0, np.nan], [2.0, 0.0, 3.0]])
        self.assertTrue(np.isnan(X[0, 1]))                                    # the input is not changed
        Y = np.array([[1.0, np.nan], [np.nan, 2.0]]); self.assertIs(model.fill_empty(Y), Y)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed (the Tests workflow does not install it)")
    def test_fit_with_a_column_no_training_row_has(self):
        # every section present on every training day but the last: no training cell has an earlier ABSENT day, so the exemplar
        # columns "minus the mean of the earlier absent days" are missing in every row of the fusion's training set
        n = 30; cut = day_of(22)
        C, P, ans = world(n_days=n, ships={(s, k): (k != 21) if k < 22 else bool((k + j) % 2) for j, s in enumerate(SECS) for k in range(n)})
        cells = [dict(c, y=ans[(c["section"], c["date"])], port="AAA" if c["section"] != SECS[2] else "BBB") for c in C]
        X, names, _ = features(P, cells)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B = daily.fit(cells, X, names, cut, log=lambda m: None)
            self.assertEqual(B["meta"]["n_train"], 66)
            with mock.patch.object(model, "fill_empty", lambda X: X):         # without the fix, scikit-learn stops
                with self.assertRaises(ValueError): daily.fit(cells, X, names, cut, log=lambda m: None)


@unittest.skipUnless(np is not None, "numpy not installed")
class FetchChoiceTest(unittest.TestCase):
    def f(self, sid, day, cloud=10.0):
        return {"id": sid, "properties": {"datetime": day + "T02:57:55Z", "eo:cloud_cover": cloud}}

    def test_new_days_of_the_ports_tile(self):
        found = [self.f("S2B_51STV_20261116_0_L2A", "2026-11-16"), self.f("S2B_50SQE_20261116_0_L2A", "2026-11-16"),       # another tile
                 self.f("S2B_51STV_20261113_0_L2A", "2026-11-13"), self.f("S2B_51STV_20261113_1_L2A", "2026-11-13"),       # two of a day: the last
                 self.f("S2C_51STV_20261111_0_L2A", "2026-11-11"),                                                         # already in the store
                 self.f("S2C_51STV_20261108_0_L2A", "2026-11-08", 90.0)]                                                   # too cloudy a tile
        self.assertEqual([x["id"] for x in daily.pick_new(found, "51STV", {"2026-11-11"})], ["S2B_51STV_20261113_1_L2A", "S2B_51STV_20261116_0_L2A"])

    def test_the_tile_of_a_port_comes_from_the_scene_list(self):
        self.assertEqual(daily.port_tile("QDH"), "51STA"); self.assertEqual(daily.port_tile("ZJG"), "49QDD")
        self.assertEqual(daily.port_tile("QGZ"), "51STV")          # 50SQE until 2024, 51STV since: the recent one
        with self.assertRaises(SystemExit): daily.port_tile("NOPE")


@unittest.skipUnless(np is not None, "numpy not installed")
class OwnerAnswersTest(unittest.TestCase):
    HEAD = "site_id,section_id,date,verified,verified_occupied,verified_by,verified_note,answered_utc,queue_version\n"

    def test_the_two_answer_files(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d) / "labels"; lab.mkdir()
            (lab / "QDH-s2-answers.csv").write_text("site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note\n"
                                                    "QDH,a,2026-01-01,1,1,2,,owner,\nQDH,a,2026-01-02,0,,,,owner,\nQDH,b,2026-01-01,1,0,0,,owner,\n", encoding="utf-8")
            self.assertEqual(daily.owner_answers("QDH", d), {("a", "2026-01-01"): 1, ("b", "2026-01-01"): 0})
            (lab / "QDH-confirm-answers.csv").write_text(self.HEAD + "QDH,a,2026-02-01,1,0,owner,,2026-02-03T00:00:00Z,q1\n"       # a new answer
                                                         "QDH,a,2026-02-02,0,,owner,,2026-02-03T00:00:00Z,q1\n"                    # cannot tell
                                                         "QDH,b,2026-02-01,1,1,auto,,2026-02-03T00:00:00Z,q1\n"                    # not the owner: never an answer
                                                         "QDH,b,2026-01-01,0,,owner,,2026-02-03T00:00:00Z,q1\n", encoding="utf-8")  # withdraws an answer
            self.assertEqual(daily.owner_answers("QDH", d), {("a", "2026-01-01"): 1, ("a", "2026-02-01"): 0})
            self.assertEqual(daily.owner_answers("XXX", d), {})

    def test_only_the_owners_rows_of_the_first_file(self):
        # the first answer file (labels/<port>-s2-answers.csv; ZJG-s2-labels.csv) too: verified_by must be the owner
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d) / "labels"; lab.mkdir()
            (lab / "QDH-s2-answers.csv").write_text("site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note\n"
                                                    "QDH,a,2026-01-01,1,1,2,,owner,\nQDH,b,2026-01-01,1,0,0,,ai,\nQDH,c,2026-01-01,1,1,1,,,\n"
                                                    "QDH,d,2026-01-01,1,0,0,, owner ,\n", encoding="utf-8")
            self.assertEqual(daily.owner_answers("QDH", d), {("a", "2026-01-01"): 1, ("d", "2026-01-01"): 0})
            (lab / "ZJG-s2-labels.csv").write_text("site_id,section_id,date,verified,verified_occupied,verified_by\n"
                                                   "ZJG,a,2026-01-01,1,1,owner\nZJG,b,2026-01-01,1,0,auto\nZJG,c,2026-01-01,,,\n", encoding="utf-8")
            self.assertEqual(daily.owner_answers("ZJG", d), {("a", "2026-01-01"): 1})

    def test_the_repository_answers(self):
        a = daily.owner_answers("ZJG"); self.assertTrue(a and set(a.values()) <= {0, 1})
        self.assertTrue(daily.owner_answers("QDH"))


@unittest.skipUnless(np is not None, "numpy not installed")
class NotWiredTest(unittest.TestCase):
    def test_no_workflow_and_no_collector_runs_the_daily_scoring(self):
        # sat-confirm.yml (PR-5) runs it through sat.pres3.confirm_run, and is the only workflow that may
        flows = [f for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")) if f.name != "sat-confirm.yml"]
        self.assertGreater(len(flows), 20)
        for f in flows + sorted((ROOT / "sat").glob("*.py")):
            self.assertNotIn("pres3.daily", f.read_text(encoding="utf-8"), f)
            self.assertNotIn("confirm_run", f.read_text(encoding="utf-8"), f)


if __name__ == "__main__":
    unittest.main()
