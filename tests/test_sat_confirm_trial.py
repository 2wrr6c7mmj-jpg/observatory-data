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
        j = src.index("confirm.sync_sections(out, port, trial, nows)")
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


if __name__ == "__main__":
    unittest.main()
