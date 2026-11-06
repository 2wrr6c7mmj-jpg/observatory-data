"""fix-2 (2部の設計書 fixed_eval 4.2): the registry of confirmed fixed objects, the object decision, the point and field
flags and the counted anchors of naval_near (synthetic rasters and the committed registry files)."""
from __future__ import annotations

import json
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from sat import fixed_registry as freg

if np is not None:
    from sat import fixed, run

WP10 = -22.0
META = {"water_vv_p10_median_db": WP10}


def flat(base_db, rise_db, spread_db, h=20, w=20):
    p10 = np.full((h, w), WP10 + base_db)
    return np.full((h, w), WP10 + base_db + rise_db), {"vv_p10": p10, "vv_p90": p10 + spread_db, "vh_p10": p10 - 7}


def cand(lon, lat, length, zone="quay", **kw):
    d = {"rmin": 5, "rmax": 7, "cmin": 5, "cmax": 15, "pixels": 33, "vh_ratio_db": 2.0, "contrast_vv_db": 20.0,
         "lon": lon, "lat": lat, "length_m": length, "zone": zone}
    d.update(kw)
    return d


def entry(i, lon, lat, kind="pier_part", **kw):
    e = {"id": i, "lon": lon, "lat": lat, "radius_m": 30, "kind": kind, "status": "confirmed", "confirmed_by": "2部",
         "added_utc": "2026-11-06", "expires_utc": None,
         "evidence": {"imagery": [{"source": "Esri World Imagery", "viewed_utc": "2026-11-06"}], "length_m": 46}}
    e.update(kw)
    return e


class RegistryFileTest(unittest.TestCase):
    def test_committed_files_are_valid_and_their_geojson_is_current(self):
        sites = freg.sites_with_registry()
        self.assertIn("QGZ", sites)
        for sid in sites:
            with self.subTest(site=sid):
                reg = freg.load(sid)
                self.assertEqual(freg.check(reg, sid), [])
                p = freg.GEOJSON_DIR / f"{sid}.geojson"
                self.assertTrue(p.exists(), "python -m sat.fixed_registry geojson")
                self.assertEqual(json.loads(p.read_text(encoding="utf-8")), freg.geojson(reg))
        self.assertEqual(sorted(p.stem for p in freg.GEOJSON_DIR.glob("*.geojson")), sites)   # none without a registry

    def test_check_finds_the_problems(self):
        good = {"site": "X", "entries": [entry("X-1", 110.0, 21.0)]}
        self.assertEqual(freg.check(good, "X"), [])
        bad = {"site": "X", "entries": [
            entry("X-1", 110.0, 21.0, kind="crane"),
            entry("X-1", 110.0, 21.0, radius_m=150),
            entry("X-2", 110.0, 21.0, kind="gear"),                                      # gear without expiry
            entry("X-3", 110.0, 21.0, kind="raft", expires_utc="2027-06-01"),            # > 120 days
            entry("X-4", 110.0, 21.0, confirmed_by=""),
            entry("X-5", 110.0, 21.0, evidence={}),
        ]}
        errs = " ".join(freg.check(bad, "X"))
        for s in ("kind crane", "repeated", "radius_m 150", "X-2: gear needs", "X-3: expires_utc later", "X-4: confirmed",
                  "X-5: confirmed without evidence", "X-5: pier_part needs evidence.length_m"):
            self.assertIn(s, errs)

    def test_check_reports_a_broken_expires_utc_and_active_skips_it(self):
        """1部の審査 (4): a malformed expires_utc is an error (not an exception), for every kind; the entry is unused."""
        reg = {"site": "X", "entries": [
            entry("E1", 110.0, 21.0, expires_utc="2027-13-01"),
            entry("E2", 110.0, 21.0, kind="raft", expires_utc="soon"),
            entry("E3", 110.0, 21.0, kind="gear", expires_utc=20270101),
            entry("E4", 110.0, 21.0, added_utc="yesterday"),
            entry("OK", 110.0, 21.0, kind="raft", expires_utc="2027-01-01")]}
        errs = freg.check(reg, "X")
        for i in ("E1", "E2", "E3"):
            self.assertTrue(any(e.startswith(f"{i}: expires_utc") and "not a date" in e for e in errs), (i, errs))
        self.assertTrue(any(e.startswith("E4: added_utc") for e in errs))
        self.assertFalse(any(e.startswith("OK") for e in errs))
        self.assertEqual([e["id"] for e in freg.active(reg, "2026-12-01T10:00:00Z")], ["OK"])
        self.assertEqual(freg.check([], "X"), ["not a registry object with an entries list"])

    def test_check_command_reports_a_file_that_is_not_json(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as t:
            Path(t, "X.json").write_text("{not json", encoding="utf-8")
            Path(t, "Y.json").write_text(json.dumps({"site": "Y", "entries": [entry("Y-1", 110.0, 21.0,
                                                                                    expires_utc="2026-02-30")]}))
            out = io.StringIO()
            with mock.patch.object(freg, "REGISTRY_DIR", Path(t)), contextlib.redirect_stdout(out):
                rc = freg.main(["check"])
            self.assertEqual(rc, 1)
            res = json.loads(out.getvalue())
            self.assertTrue(res["X"][0].startswith("load:"))
            self.assertIn("Y-1: expires_utc", res["Y"][0])

    def test_active_expiry_match_and_hash(self):
        reg = {"site": "X", "entries": [
            entry("A", 110.0, 21.0), entry("B", 110.001, 21.0, kind="raft", expires_utc="2027-03-01"),
            entry("C", 110.002, 21.0, status="proposed", confirmed_by="")]}
        self.assertEqual([e["id"] for e in freg.active(reg, "2026-12-01T10:00:00Z")], ["A", "B"])
        self.assertEqual([e["id"] for e in freg.active(reg, "2027-03-01T10:00:00Z")], ["A"])
        act = freg.active(reg, "2026-12-01T10:00:00Z")
        self.assertEqual(freg.match(act, 110.0002, 21.0)["id"], "A")          # 21 m
        self.assertIsNone(freg.match(act, 110.0, 21.0004))                    # 44 m
        self.assertEqual(freg.entries_hash(act), freg.entries_hash(list(reversed(act))))
        self.assertNotEqual(freg.entries_hash(act), freg.entries_hash(act[:1]))
        self.assertEqual(freg.entries_hash([]), "")

    def test_review_lists_guarded_and_repeated_points(self):
        f = lambda t, lon, flag, **p: {"geometry": {"coordinates": [lon, 21.0]},  # noqa: E731
                                       "properties": {"site_id": "X", "acquired_utc": t, "fixed_flag": flag, **p}}
        r = freg.review([f("t1", 110.0, "point"), f("t2", 110.0001, "point"), f("t3", 110.01, "point"),
                         f("t1", 110.02, "guarded", fixed_guard="naval_long", berthed_long=True, length_m=120),
                         f("t2", 110.03, "", fixed_excluded="registry", registry_over_guard=True, fixed_registry_id="R",
                           fixed_guard="naval_near", length_m=46, rise_db=2.1),
                         f("t2", 110.04, "", fixed_excluded="registry", registry_over_guard=False)])
        self.assertEqual([(p["scenes"], p["status"]) for p in r["proposed_points"]], [(["t1", "t2"], "proposed")])
        self.assertEqual([(g["fixed_guard"], g["berthed_long"]) for g in r["guarded"]], [("naval_long", True)])
        self.assertEqual([(g["fixed_registry_id"], g["fixed_guard"], g["length_m"]) for g in r["registry_over_guard"]],
                         [("R", "naval_near", 46)])


@unittest.skipUnless(np is not None, "numpy not installed")
class DecideTest(unittest.TestCase):
    def run2(self, dets, vv, bl, ctx=None, registry=None, **kw):
        return fixed.classify(dets, vv, bl, META, None, version="fix-2", guard_ctx=ctx, registry=registry, **kw)

    def test_registry_drops_a_candidate_that_is_no_structure_by_the_baseline(self):
        vv, bl = flat(4.0, 1.0, 6.0)        # not a structure for fix-1 (base 4, spread 6)
        d = [cand(110.0, 21.0, 30)]
        c = self.run2(d, vv, bl, [{"berth_class": "civil"}], [entry("R1", 110.0001, 21.0, kind="buoy")], registry_hash="h")
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_registry_id"], d[0]["ship_conf"]), ("registry", "R1", None))
        self.assertEqual((c["registry"], c["structure"], c["registry_hash"], c["revision"]), (1, 0, "h", 2))
        self.assertFalse(run.at_berth(d[0]))

    def test_guard_and_the_confirmed_structure_kinds(self):
        """QGZ-001 (mask_error) on a naval line, 187 m: dropped when it did not rise, kept when something came today;
        a buoy entry never wins over the guard."""
        reg = [entry("M", 110.0, 21.0, kind="mask_error", evidence={"imagery": [{"source": "Esri"}], "length_m": 187}),
               entry("B", 110.01, 21.0, kind="buoy")]
        naval = [{"berth_class": "naval"}]
        vv, bl = flat(12.0, 2.8, 3.5)
        d = [cand(110.0, 21.0, 187)]
        c = self.run2(d, vv, bl, naval, reg)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"], d[0]["fixed_guard"], d[0]["fixed_registry_id"]),
                         ("registry", "", "naval_long", "M"))
        self.assertEqual(c["guarded"], 0)
        vv, bl = flat(12.0, 7.0, 3.5)       # a ship alongside today: rise 7 dB
        d = [cand(110.0, 21.0, 187)]
        self.run2(d, vv, bl, naval, reg)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"], d[0]["berthed_long"]), ("", "guarded", True))
        vv, bl = flat(12.0, 1.0, 2.0)
        d = [cand(110.01, 21.0, 120)]
        self.run2(d, vv, bl, naval, reg)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"], d[0]["ship_conf"]), ("", "guarded", "low"))

    def test_a_long_ship_alongside_a_registry_breakwater_is_kept(self):
        """1部の審査 (#77-1): QJG-002 and QJG-003 (the breakwater head, 48 m apart, radius 30 m). A warship moored there
        for months is bright in the baseline and does not rise today (rise < 3 dB), like the structure; it is kept by
        its length (over evidence.length_m 46 m + 10 m), and the 46 m piece of the structure itself is still dropped."""
        reg = freg.active(freg.load("QJG"), "2026-11-06T10:00:00Z")
        e2, e3 = (next(e for e in reg if e["id"] == i) for i in ("QJG-002", "QJG-003"))
        self.assertEqual(fixed.registry_len_limit(e2), 56.0)
        mid = ((e2["lon"] + e3["lon"]) / 2, (e2["lat"] + e3["lat"]) / 2)   # 24 m from each: in both circles
        naval = [{"berth_class": "naval"}]
        vv, bl = flat(16.0, 1.0, 2.5)        # S: bright when empty, nothing new today, never varies
        d = [cand(e2["lon"], e2["lat"], 46)]          # the structure alone: no guard (46 m), dropped as before
        c = self.run2(d, vv, bl, naval, reg)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_registry_id"], d[0]["registry_over_guard"]),
                         ("registry", "QJG-002", False))
        self.assertEqual(c["registry_over_guard"], 0)
        for (lon, lat), ln in ((mid, 120), ((e3["lon"], e3["lat"]), 60), ((e2["lon"], e2["lat"]), 95)):
            with self.subTest(length=ln, at=(lon, lat)):   # a ship alongside, in the circles, rise < 3 dB
                d = [cand(lon, lat, ln)]
                c = self.run2(d, vv, bl, naval, reg)
                self.assertIn(d[0]["fixed_registry_id"], ("QJG-002", "QJG-003"))
                self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"], d[0]["fixed_guard"], d[0]["berthed_long"],
                                  d[0]["ship_conf"], d[0]["registry_over_guard"]),
                                 ("", "guarded", "naval_long", True, "low", False))
                self.assertEqual((c["registry_over_guard"], c["guarded"], c["registry"]), (0, 1, 0))
        # the ship and a piece in the same scene (naval_near): the piece up to 56 m is dropped over the guard and counted
        # in registry_over_guard; a 57 m piece is kept with the ship
        for ln, want in ((46, ("registry", "", "naval_near", True)), (56, ("registry", "", "naval_near", True)),
                         (57, ("", "guarded", "naval_near", False))):
            with self.subTest(piece=ln):
                d = [cand(mid[0], mid[1], 120), cand(e3["lon"] + 0.0002, e3["lat"], ln)]   # the piece 18 m from QJG-003
                c = self.run2(d, vv, bl, naval * 2, reg)
                self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"], d[0]["fixed_guard"]),
                                 ("", "guarded", "naval_long"))
                self.assertEqual((d[1]["fixed_excluded"], d[1]["fixed_flag"], d[1]["fixed_guard"],
                                  d[1]["registry_over_guard"]), want)
                self.assertEqual((c["registry_over_guard"], c["guarded"]), (1, 1) if want[3] else (0, 2))
        # without a length in the evidence the guard always wins; off the guard the registry drops as before
        noln = [entry("N", 110.0, 21.0, evidence={"imagery": [{"source": "Esri"}]})]
        d = [cand(110.0, 21.0, 70)]
        self.run2(d, vv, bl, naval, noln)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_flag"]), ("", "guarded"))
        d = [cand(110.0, 21.0, 300)]
        c = self.run2(d, vv, bl, [{"berth_class": "civil"}], noln)
        self.assertEqual((d[0]["fixed_excluded"], d[0]["registry_over_guard"], c["registry_over_guard"]),
                         ("registry", False, 0))

    def test_naval_near_needs_a_counted_anchor(self):
        """1部の指摘 (2): the 60 m candidate that makes a piece guarded must itself be counted."""
        vv, bl = flat(20.0, 0.5, 2.0)
        mk = lambda **kw: [cand(110.0, 21.0, 120, **kw), cand(110.0005, 21.0, 40)]  # noqa: E731  52 m apart
        ctx = [{"berth_class": "naval"}, {"berth_class": "naval"}]
        d = mk()
        self.run2(d, vv, bl, ctx)
        self.assertEqual([x["fixed_guard"] for x in d], ["naval_long", "naval_near"])
        for kw in ({"window_excluded": "land"}, {"berth_excluded": "coastline"}):
            d = mk(**kw)
            self.run2(d, vv, bl, ctx)
            self.assertEqual((d[1]["fixed_guard"], d[1]["fixed_excluded"]), ("", "structure"))
        d = mk()                            # the anchor is dropped by a confirmed structure entry
        self.run2(d, vv, bl, ctx, [entry("M", 110.0, 21.0, kind="mask_error",
                                         evidence={"imagery": [{"source": "Esri"}], "length_m": 120})])
        self.assertEqual([x["fixed_excluded"] for x in d], ["registry", "structure"])

    def test_point_and_field_flags_do_not_change_the_counts(self):
        vv, bl = flat(6.0, 1.0, 3.0)        # base 6 < 8: no structure; weak and always there: a buoy
        d = [cand(110.0, 21.0, 30, zone="harbour"), cand(110.01, 21.0, 30, zone="quay"), cand(110.02, 21.0, 60, zone="harbour")]
        c = self.run2(d, vv, bl)
        self.assertEqual([x["fixed_flag"] for x in d], ["point", "", ""])
        self.assertEqual([x["fixed_excluded"] for x in d], ["", "", ""])
        self.assertEqual((c["point"], c["low"]), (1, 3))
        vv, bl = flat(0.0, 10.0, 3.0)       # new today (rise 10): no point; five small ones within 300 m: a field
        d = [cand(110.0 + 0.0005 * k, 21.0, 20, zone="harbour") for k in range(5)] + [cand(110.05, 21.0, 20, zone="harbour")]
        c = self.run2(d, vv, bl)
        self.assertEqual([x["fixed_flag"] for x in d], ["field"] * 5 + [""])
        self.assertEqual(c["field"], 5)
        self.assertTrue(all(run.at_berth(x) for x in d))

    def test_object_decision_for_merged_pieces(self):
        """Design 5: an object is dropped only when every piece is S or in the registry, and its merged length is
        what the guard sees (two 40 m pieces of a 100 m ship)."""
        vv, bl = flat(20.0, 0.5, 2.0)
        vs = fixed.smooth3_db(vv)
        naval = [{"berth_class": "naval"}] * 2
        for objs, lens, want in (([[0], [1]], None, ["structure", "structure"]),
                                 ([[0, 1]], [100.0], ["", ""])):
            d = [cand(110.0, 21.0, 40), cand(110.0004, 21.0, 40)]
            sc = fixed.score(d, vs, bl, WP10, None, None)
            self.assertFalse(any(fixed.bridge_ok(s) for s in sc))        # S never joins pieces
            fixed.decide(d, sc, naval, None, objs, lens)
            self.assertEqual([x["fixed_excluded"] for x in d], want)
        self.assertEqual(d[0]["fixed_flag"], "guarded")
        vv2, bl2 = flat(20.0, 8.0, 2.0)     # one piece rose today: the object is no structure
        d = [cand(110.0, 21.0, 40), cand(110.0004, 21.0, 40)]
        sc = fixed.score(d, vs, bl, WP10, None, None)
        sc[1] = fixed.score(d[1:], fixed.smooth3_db(vv2), bl2, WP10, None, None)[0]
        self.assertTrue(fixed.bridge_ok(sc[1]))
        fixed.decide(d, sc, [{"berth_class": "civil"}] * 2, None, [[0, 1]], [80.0])
        self.assertEqual([x["fixed_excluded"] for x in d], ["", ""])

    def test_the_first_registry_on_the_sample(self):
        """The committed QGZ / QJG entries on 2部's sample values (fixed_eval/out/classes.csv): the two structures of
        margin.csv and the two pieces left by naval_near are dropped; nothing else in those scenes is touched here."""
        cases = [("QGZ", 119.963946, 35.741315, 187, 12.0, 2.8, 3.5, "QGZ-001"),
                 ("QGZ", 119.96431, 35.741636, 38, 10.5, 0.1, 3.2, "QGZ-002"),
                 ("QJG", 120.59533, 36.103067, 102, 14.2, 2.0, 3.8, "QJG-001"),
                 ("QJG", 120.571333, 36.108815, 46, 15.8, 2.1, 2.9, "QJG-002"),
                 ("QJG", 120.571846, 36.108705, 46, 16.5, 0.7, 2.5, "QJG-003")]
        for sid, lon, lat, ln, base, rise, spread, rid in cases:
            with self.subTest(rid=rid):
                reg = freg.active(freg.load(sid), "2026-11-06T10:00:00Z")
                vv, bl = flat(base, rise, spread)
                d = [cand(lon, lat, ln)]
                self.run2(d, vv, bl, [{"berth_class": "naval"}], reg)
                self.assertEqual((d[0]["fixed_excluded"], d[0]["fixed_registry_id"]), ("registry", rid))

    def test_settings(self):
        st = fixed.settings({"QGZ": "abc"})
        self.assertEqual([v["version"] for v in st["versions"]], ["fix-2"])    # 1部の指摘 (5): no empty fix-1 period
        self.assertEqual(st["registry"]["hashes"], {"QGZ": "abc"})
        self.assertNotIn("errors", st["registry"])
        self.assertEqual(st["guard"]["registry_len"]["pad_m"], fixed.REGISTRY_LEN["pad_m"])
        self.assertEqual(fixed.settings({}, {"X": "bad"})["registry"]["errors"], {"X": "bad"})

    def test_a_broken_registry_file_does_not_stop_the_save(self):
        """1部の審査 (4): build_meta and the scene's decision record the error and go on."""
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as t:
            Path(t, "QGZ.json").write_text("{broken", encoding="utf-8")
            Path(t, "QJG.json").write_text((freg.REGISTRY_DIR / "QJG.json").read_text(encoding="utf-8"), encoding="utf-8")
            with mock.patch.object(freg, "REGISTRY_DIR", Path(t)):
                hashes, errors = run.registry_hashes()
                reg, h, err = run.site_registry("QGZ", "2026-11-06T10:00:00Z")
                reg2, h2, err2 = run.site_registry("QJG", "2026-11-06T10:00:00Z")
        self.assertEqual(set(hashes), {"QJG"})
        self.assertIn("QGZ", errors)
        self.assertEqual((reg, h), ([], ""))
        self.assertTrue(err.startswith("JSONDecodeError"))
        self.assertEqual((len(reg2), err2), (3, ""))
        st = fixed.settings(hashes, errors)
        self.assertIn("QGZ", st["registry"]["errors"])
        self.assertIn("bridge_ok", fixed.__dict__)

    def test_chips_grey_frame(self):
        from sat import chips as satchips
        from sat.grid import Grid
        g = Grid(110.0, 21.0, 110.04, 21.04, 400, 400)
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [110.02, 21.02]},
                  "properties": {"n": k + 1, "zone": "harbour", "length_m": 30, "width_m": 10, "fixed_version": "fix-2",
                                 "ship_conf": "low", "fixed_flag": flag, "berthed_long": flag == "guarded"}}
                 for k, flag in enumerate(("point", "guarded", ""))]
        doc = satchips.scene_doc({"id": "X"}, g, {"acquired_utc": "2026-11-06T10:00:00Z", "color_image": "c.webp"},
                                 feats, (400, 450), None, "X-land-v1")
        c = doc["candidates"]
        self.assertEqual([(x.get("frame"), x.get("berthed_long")) for x in c], [("grey", None), ("red", True), (None, None)])


if __name__ == "__main__":
    unittest.main()
