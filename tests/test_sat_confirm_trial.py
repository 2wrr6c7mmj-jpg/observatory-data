"""The trial sections of the confirmation queue (sat/confirm_trial.py, sat/confirm_trial.json): the pier search's two lines
(QGZ-E-B, XSN-W-L) as a later sections_version of the ledger, their patches cut from a window widened for them, and nothing of
the existing sections moves (their ids, labels, states, patches, water colour, clear-day flags and scores).

The first classes are pure Python (the collect workflows run them without numpy); the patch tests need numpy, the cut of a
widened window rasterio (skipped where it is not installed: the Tests workflow does not install it).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from sat import confirm as cf
from sat import confirm_trial as ct

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
NOW = "2026-11-22T08:00:00Z"
SITES = {x["id"]: x for x in json.loads((ROOT / "sat" / "sites.json").read_text(encoding="utf-8"))["sites"]}


def review_ids(site):
    p = OUT / "labels" / site / "review.json"
    return [s["id"] for s in json.loads(p.read_text(encoding="utf-8"))["sections"]] if p.exists() else []


class TrialFileTest(unittest.TestCase):
    def test_the_file_is_up_to_date_and_valid(self):
        doc = ct.load()
        self.assertEqual(ct.check(doc, {s: review_ids(s) for s in doc["sections"]}), [])

    def test_the_two_candidates_of_the_report_only(self):
        doc = ct.load()
        self.assertEqual(sorted({s["candidate"] for s in doc["sources"]}), ["QGZ-E-B", "XSN-W-L"])
        self.assertEqual(sorted(doc["sections"]), ["QGZ", "XSN"])
        self.assertEqual(sorted(s["way"] for s in doc["sources"] if s["candidate"] == "QGZ-E-B"), [1564919265, 1564919266])
        for s in ct.sections("QGZ") + ct.sections("XSN"):
            self.assertIs(s["trial"], True); self.assertIs(s["counted"], False); self.assertEqual(s["sector"], ct.SECTOR)
        self.assertEqual({s["role"] for s in ct.sections("XSN")}, {"fitout"})
        self.assertEqual({s["role"] for s in ct.sections("QGZ")}, {"naval_empty"})
        self.assertEqual(ct.sections("ZJG"), []); self.assertEqual(ct.sections("HPU"), [])

    def test_the_window_is_widened_only_where_a_trial_line_lies(self):
        for sid, site in SITES.items():
            bb = ct.fetch_bbox(sid, site["bbox"])
            if sid in ("QGZ", "XSN"):
                e = ct.extent(sid)
                self.assertTrue(bb[0] <= e[0] and bb[1] <= e[1] and e[2] <= bb[2] and e[3] <= bb[3], sid)       # the frames are in it
                self.assertTrue(bb[0] <= site["bbox"][0] and bb[2] >= site["bbox"][2] and bb[1] <= site["bbox"][1] and bb[3] >= site["bbox"][3])
            else:
                self.assertEqual(bb, [float(x) for x in site["bbox"]], sid)
        self.assertGreater(ct.fetch_bbox("QGZ", SITES["QGZ"]["bbox"])[2], 120.016)           # Pier 9 is outside the site's bbox
        self.assertLess(ct.fetch_bbox("XSN", SITES["XSN"]["bbox"])[0], 121.681)              # 6 km west of the site's bbox

    def test_not_in_the_counted_sources(self):
        # the published counts read sat/berths.json, sat/berth_class/, the masks and the review sets: no trial id, no trial way
        ids = {s["id"] for s in ct.sections("QGZ") + ct.sections("XSN")}
        texts = [(ROOT / "sat" / "berths.json").read_text(encoding="utf-8")]
        texts += [p.read_text(encoding="utf-8") for p in sorted((ROOT / "sat" / "berth_class").glob("*.json"))]
        texts += [(OUT / "labels" / s / "review.json").read_text(encoding="utf-8") for s in ("QGZ", "XSN")]
        for t in texts:
            for w in ("1564919265", "1564919266", "XSN-W-L"):
                self.assertNotIn(w, t)
        for s in ("QGZ", "XSN"):
            self.assertFalse(ids & set(review_ids(s)))

    def test_pieces_and_frames(self):
        for s in ct.sections("QGZ") + ct.sections("XSN"):
            self.assertTrue(60 <= s["length_m"] <= ct.PIECE_MAX_M, s["id"])
            k = len(s["line"]); self.assertEqual(len(s["frame"]), 2 * k)
            self.assertEqual(s["frame"][:k], s["line"])
            self.assertEqual(s["side"], ct.SIDES[s["id"].split(":")[1]])
        # the two sides of a piece: the frames lie on opposite sides of the line
        a = next(s for s in ct.sections("QGZ") if s["id"] == "1564919265:L:00"); b = next(s for s in ct.sections("QGZ") if s["id"] == "1564919265:R:00")
        self.assertEqual(a["line"], b["line"])
        self.assertGreater(a["frame"][-1][0], a["line"][0][0]); self.assertLess(b["frame"][-1][0], b["line"][0][0])   # L east, R west (line to the north)


class LedgerTest(unittest.TestCase):
    """The trial sections enter a ledger made from the review set as sections_version 2, and nothing of the existing
    sections changes (ids, lines, labels, states); their cells all go to the owner."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        for s in ("QGZ", "XSN"):
            (self.out / "labels" / s).mkdir(parents=True)
            shutil.copy(OUT / "labels" / s / "review.json", self.out / "labels" / s / "review.json")
            shutil.copy(OUT / "labels" / f"{s}-s2-answers.csv", self.out / "labels" / f"{s}-s2-answers.csv")

    def scores(self, site, ids, day, p="0.01", n="5"):
        return [{"site_id": site, "section_id": i, "date": day, "acquired_utc": day + "T02:50:00Z", "p": p, "model_version": "pres3o-test",
                 "cloud_shadow": "0.0", "nodata": "0", "n_answers": n} for i in ids]

    def test_a_later_version_and_the_existing_sections_untouched(self):
        cfg = cf.load_config()
        for site in ("QGZ", "XSN"):
            v1 = cf.sync_sections(self.out, site, cf.review_sections(self.out, site), "2026-11-22T07:29:00Z")
            self.assertEqual(v1["sections_version"], 1)
            ids1 = [s["id"] for s in v1["sections"]]; labels1 = cf.section_labels(v1)
            cf.run(self.scores(site, ids1, "2026-11-20"), self.out, cfg, "2026-11-22T07:40:00Z")
            state1 = cf.load_state(self.out, site); log1 = cf.load_log(self.out, site)
            v2 = cf.sync_sections(self.out, site, ct.sections(site), "2026-11-23T07:29:00Z")
            self.assertEqual(v2["sections_version"], 2)
            self.assertEqual([s["id"] for s in v2["sections"]][:len(ids1)], ids1)
            self.assertEqual(v2["sections"][:len(ids1)], v1["sections"])                          # the existing entries as they were
            new = v2["sections"][len(ids1):]
            self.assertEqual([s["id"] for s in new], [s["id"] for s in ct.sections(site)])
            self.assertTrue(all(s["sections_version"] == 2 and s["since"] == "2026-11-23" and s["trial"] for s in new))
            labels2 = cf.section_labels(v2)
            self.assertEqual({k: labels2[k] for k in labels1}, labels1)                           # no existing label changes
            self.assertTrue(all(labels2[s["id"]][0].startswith("区画 試行・桟橋 ") for s in new))
            self.assertEqual(cf.sync_sections(self.out, site, ct.sections(site), "2026-11-24T07:29:00Z"), v2)   # again: nothing
            self.assertEqual(cf.load_state(self.out, site), state1)
            # the next run: the trial cells (no answer, no reference) go to the owner; the existing cells as before
            day = "2026-11-23"
            rows = self.scores(site, ids1, "2026-11-20") + self.scores(site, [s["id"] for s in new], day, p="", n="0")
            cf.run(rows, self.out, cfg, "2026-11-23T07:40:00Z")
            st = cf.load_state(self.out, site)
            self.assertEqual({k: v for k, v in st.items() if k in state1}, state1)
            self.assertEqual(cf.load_log(self.out, site)[:len(log1)], log1)
            for s in new:
                r = st[(s["id"], day)]
                self.assertEqual((r["state"], r["sections_version"]), ("pending", "2"))
                self.assertIn("few_answers", r["reason"]); self.assertIn("no_reference", r["reason"])
            q = json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))
            self.assertTrue({(site, s["id"], day) for s in new} <= {(c["site_id"], c["section_id"], c["date"]) for c in q["cells"]})
            lt = json.loads((self.out / "confirm" / "latest.json").read_text(encoding="utf-8"))
            rows_l = {r["id"]: r for r in lt["sites"][site]["sections"]}
            self.assertTrue(all(rows_l[s["id"]]["trial"] is True and rows_l[s["id"]]["role"] == s["role"] for s in new))
            self.assertTrue(all("trial" not in rows_l[i] for i in ids1))                      # the existing rows: the same keys
            summ = json.loads((self.out / "confirm" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summ["parts"]["after_freeze"]["by_site"][site]["new_sections"]["cells"], len(new))

    def test_the_daily_run_adds_them_to_a_ledger_it_makes_or_finds(self):
        src = (ROOT / "sat" / "pres3" / "confirm_run.py").read_text(encoding="utf-8")
        i = src.index("confirm.sync_sections(out, port, confirm.review_sections(out, port), nows)")
        j = src.index("bad = sync_trial(out, port, nows)")
        self.assertLess(i, j)            # the review set first (version 1 of a new ledger), then the trial sections (a later version)


@unittest.skipUnless(np is not None, "numpy not installed")
class PatchTest(unittest.TestCase):
    """The water colour, the clear-day flag and the scores of the existing sections are the same with the trial cells in the
    store."""

    def world(self):
        from tests.test_sat_pres3_daily import world
        C, P, ans = world(n_days=30)
        rng = np.random.default_rng(3)
        for i in rng.choice(len(C), 12, replace=False): C[i]["cloud"] = 0.5
        T = []; TP = []
        for c, p in zip(C, P):
            if c["section"] == "s:L:00":            # a trial section on every day: brighter water, sometimes cloudy
                T.append(dict(c, section="t:L:00", cloud=0.6 if c["date"].endswith("5") else 0.0)); TP.append(np.asarray(p, np.float32) + 0.2)
        return C, P, ans, T, np.asarray(TP, np.float16)

    def test_day_flags_of_the_existing_cells(self):
        from sat.pres3 import daily, feats
        C, P, _, T, TP = self.world()
        days = sorted({c["date"] for c in C})
        for d in days:
            ix = [i for i, c in enumerate(C) if c["date"] == d]; it = [i for i, c in enumerate(T) if c["date"] == d]
            Cd = [C[i] for i in ix] + [T[i] for i in it]; Pd = np.concatenate([P[ix], TP[it]])
            W, g = daily.day_flags(Pd, Cd, trial={"t:L:00"})
            W0, g0 = feats.water(P[ix], daily.OFFS, [C[i] for i in ix]), feats.good_day([C[i] for i in ix])
            np.testing.assert_array_equal(W[:len(ix)], W0); np.testing.assert_array_equal(g[:len(ix)], g0)
            np.testing.assert_array_equal(W[len(ix):], np.repeat(W0[:1], len(it), 0))
            Wall = feats.water(Pd, daily.OFFS, Cd)
            self.assertFalse(np.array_equal(Wall[:len(ix)], W0))        # without trial=: the trial cells would move the water colour
            np.testing.assert_array_equal(daily.day_flags(P[ix], [C[i] for i in ix])[0], W0)     # no trial: as before

    def test_scores_of_the_existing_sections_do_not_move(self):
        from sat.pres3 import daily
        from tests.test_sat_pres3_daily import OFFS, features, toy_model
        C, P, ans, T, TP = self.world()
        _, names, _ = features(P[:6], C[:6]); B = toy_model(names)
        with tempfile.TemporaryDirectory() as d:
            a = daily.Store(Path(d) / "a"); a.put("AAA", C, P)
            b = daily.Store(Path(d) / "b"); b.put("AAA", C + T, np.concatenate([P, TP]), trial=frozenset({"t:L:00"}))
            Ca, Pa, _ = a.load("AAA"); Cb, Pb, _ = b.load("AAA")
            keep = [k for k, c in enumerate(Cb) if c["section"] != "t:L:00"]
            self.assertEqual([Cb[k] for k in keep], Ca); np.testing.assert_array_equal(Pb[keep], Pa)
            days = sorted({c["date"] for c in C})[-8:]
            ra = daily.score_days("AAA", Ca, Pa, OFFS, ans, B, days)
            rb = daily.score_days("AAA", Cb, Pb, OFFS, ans, B, days)
            self.assertEqual([r for r in rb if r["section_id"] != "t:L:00"], ra)
            self.assertTrue(any(r["p"] != "" for r in ra))
            tr = [r for r in rb if r["section_id"] == "t:L:00"]
            self.assertEqual(len(tr), len(days)); self.assertTrue(all(r["n_answers"] == 0 for r in tr))
            # without trial= the water colour of the existing cells would move with the trial cells (the test has teeth)
            c = daily.Store(Path(d) / "c"); c.put("AAA", C + T, np.concatenate([P, TP]))
            Cc, _, _ = c.load("AAA")
            self.assertNotEqual([Cc[k] for k in keep], Ca)

    def test_the_frames_with_and_without_the_trial_sections(self):
        from sat.pres3 import labels
        for site in ("QGZ", "XSN", "QDH"):
            f0 = labels.frames(site); f1 = labels.frames(site, trial=True)
            self.assertEqual(set(f1) - set(f0), {s["id"] for s in ct.sections(site)})
            for k in f0:
                self.assertEqual(set(f0[k]), set(f1[k]) - {"trial"}); self.assertFalse(f1[k].get("trial"))
                np.testing.assert_array_equal(f0[k]["near"], f1[k]["near"]); np.testing.assert_array_equal(f0[k]["far"], f1[k]["far"])
            self.assertTrue(all(f1[k]["trial"] for k in set(f1) - set(f0)))

    def test_the_refit_learns_from_the_counted_sections_only(self):
        from sat.pres3 import confirm_run as cr, daily
        from tests.test_sat_pres3_daily import world
        C, P, ans = world(n_days=12)
        T = [dict(c, section=ct.sections("QGZ")[0]["id"]) for c in C if c["section"] == "s:L:00"]
        TP = P[[k for k, c in enumerate(C) if c["section"] == "s:L:00"]]
        C = [dict(c, port="QGZ") for c in C]; T = [dict(c, port="QGZ") for c in T]
        with tempfile.TemporaryDirectory() as d:
            st = daily.Store(Path(d) / "s"); st.put("QGZ", C + T, np.concatenate([P, TP]), trial=ct.trial_ids("QGZ"))
            sat = Path(d) / "sat"; (sat / "labels").mkdir(parents=True)
            with (sat / "labels" / "QGZ-s2-answers.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n"); w.writerow(["site_id", "section_id", "date", "verified", "verified_occupied", "verified_by"])
                for c in C + T: w.writerow(["QGZ", c["section"], c["date"], "1", "1", "owner"])
            got, X, _ = cr.store_cells(st, ["QGZ"], sat)
            self.assertEqual(len(got), len(C)); self.assertFalse({g["section"] for g in got} & ct.trial_ids("QGZ"))


@unittest.skipUnless(np is not None, "numpy not installed")
class BaseWindowTest(unittest.TestCase):
    def test_where_the_site_window_lies_in_the_wide_one(self):
        from sat.pres3 import fetch
        wide = (399960.0, 3950000.0, 412000.0, 3962020.0); base = (401000.0, 3951000.0, 405000.0, 3960000.0)
        bw = fetch.base_window(wide, base)
        self.assertEqual(bw, dict(x0=401000.0, y1=3960000.0, row=202, col=104, rows=900, cols=400))
        self.assertTrue(all(bw[k] % 2 == 0 for k in ("row", "col", "rows", "cols")))        # on the 20 m grid
        with self.assertRaises(ValueError): fetch.base_window(base, wide)
        self.assertEqual(fetch.snapped((401001.0, 3950999.0, 404999.5, 3960000.1)), (401000.0, 3950980.0, 405000.0, 3960020.0))

    @unittest.skipUnless(importlib.util.find_spec("rasterio"), "rasterio not installed (the Tests workflow does not install it)")
    def test_a_wide_window_cuts_the_same_patches(self):
        # a window of the site (as the research windows) and a wider one around it with meta['base']: the review sections'
        # cells are the same bytes; the trial sections are cut from the wide part
        from rasterio.warp import transform
        from sat.pres3 import build, fetch, labels
        site = SITES["QGZ"]; bb = site["bbox"]; wide_bb = ct.fetch_bbox("QGZ", bb); crs = "EPSG:32651"
        pad = fetch.PAD_DEG

        def box(b):
            xs, ys = transform("EPSG:4326", crs, [b[0] - pad, b[2] + pad, b[0] - pad, b[2] + pad], [b[1] - pad, b[1] - pad, b[3] + pad, b[3] + pad])
            return fetch.snapped((min(xs), min(ys), max(xs), max(ys)))
        bs = box(bb); wd = box(wide_bb); wd = (min(wd[0], bs[0]), min(wd[1], bs[1]), max(wd[2], bs[2]), max(wd[3], bs[3]))
        bw = fetch.base_window(wd, bs)
        H, Wd = int((wd[3] - wd[1]) / 10), int((wd[2] - wd[0]) / 10)
        rng = np.random.default_rng(0)
        full = {b: rng.integers(1, 3000, (H, Wd)).astype(np.uint16) for b in ("blue", "green", "red", "nir")}
        for b in ("swir16", "scl"):
            full[b] = (rng.integers(1, 3000, (H // 2, Wd // 2)) if b == "swir16" else rng.choice([4, 6, 8, 3], (H // 2, Wd // 2))).astype(np.uint16 if b == "swir16" else np.uint8)
        part = {b: (a[bw["row"]:bw["row"] + bw["rows"], bw["col"]:bw["col"] + bw["cols"]] if b in ("blue", "green", "red", "nir")
                    else a[bw["row"] // 2:(bw["row"] + bw["rows"]) // 2, bw["col"] // 2:(bw["col"] + bw["cols"]) // 2]) for b, a in full.items()}
        meta = dict(crs=crs, id="S2B_51STV_20261120_0_L2A", datetime="2026-11-20T02:57:55Z", cloud=3.0)
        with tempfile.TemporaryDirectory() as d:
            fw = Path(d) / "wide.npz"; fb = Path(d) / "base.npz"
            mw = dict(meta, x0=wd[0], y1=wd[3], base=bw); mb = dict(meta, x0=bs[0], y1=bs[3])
            np.savez(fw, meta=json.dumps(mw), **full); np.savez(fb, meta=json.dumps(mb), **part)
            s1 = labels.frames("QGZ", trial=True); trial = frozenset(k for k, s in s1.items() if s.get("trial")); s0 = labels.frames("QGZ")
            Cw, Pw = build.day_cells("QGZ", "2026-11-20", [(str(fw), json.loads(json.dumps(mw)))], s1, build.sample_points(s1), {}, trial=trial)
            Cb, Pb = build.day_cells("QGZ", "2026-11-20", [(str(fb), json.loads(json.dumps(mb)))], s0, build.sample_points(s0), {})
            iw = {c["section"]: k for k, c in enumerate(Cw)}
            for k, c in enumerate(Cb):
                self.assertEqual({f: v for f, v in Cw[iw[c["section"]]].items()}, c)
                np.testing.assert_array_equal(Pw[iw[c["section"]]].view(np.uint16), Pb[k].view(np.uint16))
            tc = [Cw[iw[i]] for i in trial]
            self.assertEqual(len(tc), len(trial)); self.assertTrue(all(c["nodata"] == 0 for c in tc))   # inside the wide arrays


class TrialNeverAutomaticTest(unittest.TestCase):
    """1部の審査（#170）：a trial section with 2 answers or more and a confident score is still shown to the owner (reason
    trial), in the shadow operation too (the log's v1), and is not in A."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        (self.out / "labels" / "QGZ").mkdir(parents=True)
        shutil.copy(OUT / "labels" / "QGZ" / "review.json", self.out / "labels" / "QGZ" / "review.json")
        self.t = ct.sections("QGZ")[0]["id"]; self.e = review_ids("QGZ")[0]
        rows = []
        for sec in (self.t, self.e):
            rows += [(sec, d, 0) for d in ("2026-11-01", "2026-11-05", "2026-11-09")] + [(sec, "2026-11-20", 0)]   # 3 earlier, and the day itself
        with (self.out / "labels" / "QGZ-s2-answers.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["site_id", "section_id", "date", "verified", "verified_occupied", "verified_c", "verified_color", "verified_by", "verified_note"])
            for sec, d, y in rows: w.writerow(["QGZ", sec, d, "1", y, y, "", "owner", ""])
        cf.sync_sections(self.out, "QGZ", cf.review_sections(self.out, "QGZ"), "2026-11-22T07:29:00Z")
        cf.sync_sections(self.out, "QGZ", ct.sections("QGZ"), "2026-11-22T07:29:00Z")

    def run_day(self, day, auto_display):
        cfg = {**cf.load_config(), "auto_display": auto_display}
        rows = [{"site_id": "QGZ", "section_id": sec, "date": day, "acquired_utc": day + "T02:50:00Z", "p": "0.01", "model_version": "pres3o-test",
                 "cloud_shadow": "0.0", "nodata": "0", "n_answers": "5"} for sec in (self.t, self.e)]
        cf.run(rows, self.out, cfg, day + "T07:40:00Z")
        return cfg

    def test_never_automatic(self):
        self.run_day("2026-11-21", True)
        st = cf.load_state(self.out, "QGZ")
        self.assertEqual(st[(self.e, "2026-11-21")]["state"], "auto")                 # the same answers and score: automatic
        r = st[(self.t, "2026-11-21")]
        self.assertEqual(r["state"], "pending"); self.assertEqual(r["reason"], "trial")
        log = {(x["section_id"], x["date"]): x for x in cf.load_log(self.out, "QGZ")}
        self.assertEqual(log[(self.t, "2026-11-21")]["v1_state"], "pending")
        self.assertIn((self.t, "2026-11-21"), {(c["section_id"], c["date"]) for c in json.loads((self.out / "confirm" / "queue.json").read_text(encoding="utf-8"))["cells"]})

    def test_not_in_A_in_the_shadow_operation(self):
        cfg = self.run_day("2026-11-20", False)                                       # the owner answered both cells of that day
        log = {(x["section_id"], x["date"]): x for x in cf.load_log(self.out, "QGZ")}
        self.assertEqual(log[(self.t, "2026-11-20")]["v1_state"], "pending"); self.assertEqual(log[(self.e, "2026-11-20")]["v1_state"], "auto")
        a = cf.summary(self.out, cfg, "2026-11-22T08:00:00Z")["parts"]["before_freeze"]["total"]["A_auto_accuracy"]
        self.assertEqual(a["n"], 1)                                                    # the existing section only
        # an old log row that would say automatic (before this rule) is kept out of A too
        p = cf.log_dir(self.out, "QGZ") / "2026-11.csv"
        rows = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
        for x in rows:
            if x["section_id"] == self.t: x["v1_state"], x["v1_value"] = "auto", "0"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cf.LOG_FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
        a = cf.summary(self.out, cfg, "2026-11-22T08:00:00Z")["parts"]["before_freeze"]["total"]["A_auto_accuracy"]
        self.assertEqual(a["n"], 1)


class RedrawnLineTest(unittest.TestCase):
    def test_a_redrawn_line_under_the_same_id_is_reported_and_skipped(self):
        doc = ct.load(); led_secs = json.loads(json.dumps(doc["sections"]["XSN"]))
        led_secs[0]["line"] = [[p[0] + 0.0003, p[1]] for p in led_secs[0]["line"]]          # drawn again, same id
        with tempfile.TemporaryDirectory() as d:
            ld = Path(d) / "sections"; ld.mkdir()
            (ld / "XSN.json").write_text(json.dumps({"sections": led_secs}), encoding="utf-8")
            bad = ct.check(doc, {}, ld)
            self.assertEqual(len(bad), 1); self.assertIn(led_secs[0]["id"], bad[0])
            self.assertEqual(ct.check(doc, {}, Path(d) / "none"), [])

    @unittest.skipUnless(np is not None, "numpy not installed")
    def test_the_daily_run_skips_only_that_section(self):
        from sat.pres3 import confirm_run as cr
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            first = [dict(s) for s in ct.sections("XSN")]; first[1] = dict(first[1], line=[[p[0] + 0.0003, p[1]] for p in first[1]["line"]])
            cf.sync_sections(out, "XSN", first, "2026-11-22T07:29:00Z")
            self.assertEqual(cr.sync_trial(out, "XSN", "2026-11-23T07:29:00Z"), [first[1]["id"]])   # no exception
            self.assertEqual(cr.sync_trial(out, "QGZ", "2026-11-23T07:29:00Z"), [])
            self.assertEqual(len(cf.load_sections(out, "QGZ")["sections"]), len(ct.sections("QGZ")))
            self.assertEqual(cr.redrawn_trial(out, "XSN"), [first[1]["id"]]); self.assertEqual(cr.redrawn_trial(out, "QGZ"), [])

    @unittest.skipUnless(np is not None, "numpy not installed")
    def test_a_redrawn_section_is_not_stored(self):
        # the fetch cuts no patch for it (no new line under the old id); the other sections as before
        from unittest import mock as m
        from sat.pres3 import build, confirm_run as cr, daily
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"; win = Path(d) / "win"; first = [dict(s) for s in ct.sections("XSN")]
            first[0] = dict(first[0], line=[[p[0] + 0.0003, p[1]] for p in first[0]["line"]])
            cf.sync_sections(out, "XSN", first, "2026-11-22T07:29:00Z")
            f = {"id": "S2B_51RUN_20261120_0_L2A", "properties": {"datetime": "2026-11-20T02:57:55Z", "eo:cloud_cover": 1.0}}

            def read(work, port, feat, bb, env_extra=None, **kw):
                (win / "data" / port).mkdir(parents=True, exist_ok=True)
                np.savez(win / "data" / port / f"{feat['id']}.npz", meta=json.dumps({"id": feat["id"]}))
            seen = []
            with m.patch("sat.pres3.fetch.search", return_value=[f]), m.patch.object(daily, "port_tile", return_value="51RUN"), \
                 m.patch.object(build, "day_cells", side_effect=lambda port, day, sc, secs, geo, old, trial=(): seen.append((set(secs), set(geo), set(trial)))), \
                 m.patch("time.sleep"):
                cr.fetch_port(daily.Store(Path(d) / "store"), win, "XSN", {"XSN": {"bbox": SITES["XSN"]["bbox"]}}, cr.Meter(), 10,
                              __import__("datetime").datetime(2026, 11, 22, tzinfo=__import__("datetime").timezone.utc), read=read, out=out)
            (secs, geo, trial), = seen
            self.assertNotIn(first[0]["id"], secs | geo | trial)
            self.assertEqual(trial, ct.trial_ids("XSN") - {first[0]["id"]})
            self.assertTrue(set(review_ids("XSN")) <= secs)


    @unittest.skipUnless(np is not None, "numpy not installed")
    def test_every_trial_section_redrawn_reads_the_site_window(self):
        # 1部の審査（#173）: XSN-W-L's 4 sections come from one line; drawn again, all 4 go: the window read is the site's bbox
        # (no widened window without base); with some left, the widened window comes with base
        from unittest import mock as m
        from sat.pres3 import build, confirm_run as cr, daily
        bb = [float(x) for x in SITES["XSN"]["bbox"]]
        for n_bad in (0, 1, 4):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out"; win = Path(d) / "win"; first = [dict(s) for s in ct.sections("XSN")]
                for k in range(n_bad):
                    first[k] = dict(first[k], line=[[p[0] + 0.0003, p[1]] for p in first[k]["line"]])
                cf.sync_sections(out, "XSN", first, "2026-11-22T07:29:00Z")
                f = {"id": "S2B_51RUN_20261120_0_L2A", "properties": {"datetime": "2026-11-20T02:57:55Z", "eo:cloud_cover": 1.0}}
                calls = []

                def read(work, port, feat, wide, env_extra=None, **kw):
                    calls.append((list(wide), kw.get("base")))
                    (win / "data" / port).mkdir(parents=True, exist_ok=True)
                    np.savez(win / "data" / port / f"{feat['id']}.npz", meta=json.dumps({"id": feat["id"]}))
                with m.patch("sat.pres3.fetch.search", return_value=[f]), m.patch.object(daily, "port_tile", return_value="51RUN"), \
                     m.patch.object(build, "day_cells", return_value=None), m.patch("time.sleep"):
                    cr.fetch_port(daily.Store(Path(d) / "store"), win, "XSN", {"XSN": {"bbox": bb}}, cr.Meter(), 10,
                                  __import__("datetime").datetime(2026, 11, 22, tzinfo=__import__("datetime").timezone.utc), read=read, out=out)
                (wide, base), = calls
                if n_bad == 4:
                    self.assertEqual((wide, base), (bb, None))
                else:
                    self.assertLess(wide[0], bb[0]); self.assertEqual(base, bb)


if __name__ == "__main__":
    unittest.main()
