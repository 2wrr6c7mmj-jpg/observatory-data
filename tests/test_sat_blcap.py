"""bl-cap (trial, sat/blcap.py, sat/quay.cap_baseline): off by default, nothing published changes, no future answer in the mark."""
import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

from sat import blcap

ROOT = Path(__file__).resolve().parent.parent
HEAD = "site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note\n"


def write_answers(lab: Path, site: str, rows, name="{site}-s2-answers.csv", head=HEAD):
    lab.mkdir(parents=True, exist_ok=True)
    with (lab / name.format(site=site)).open("w", encoding="utf-8") as f:
        f.write(head)
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


class ResidentTest(unittest.TestCase):
    """The resident mark reads only the owner's answers dated before the scene's day."""

    def test_future_answers_are_never_read(self):
        ans = {"a": [("2026-01-01", 1), ("2026-01-05", 1), ("2026-01-09", 1), ("2026-01-10", 0), ("2026-02-01", 0),
                     ("2026-02-02", 0), ("2026-02-03", 0)]}
        m = blcap.resident(ans, "a", "2026-01-10T10:00:00Z")      # the scene's own day is not before it
        self.assertEqual((m["resident"], m["answers"], m["present"], m["last_answer"]), (True, 3, 3, "2026-01-09"))
        # answers after the scene (absent, many) cannot take the mark away, nor can they give it
        self.assertTrue(blcap.resident(ans, "a", "2026-01-10T00:00:00Z")["resident"])
        late = {"b": [("2026-03-01", 1), ("2026-03-02", 1), ("2026-03-03", 1)]}
        self.assertFalse(blcap.resident(late, "b", "2026-03-01T23:59:59Z")["resident"])
        self.assertEqual(blcap.resident(late, "b", "2026-03-01T23:59:59Z")["answers"], 0)
        self.assertEqual(blcap.resident_sections(late, "2026-03-03T01:00:00Z"), {})
        self.assertIn("b", blcap.resident_sections(late, "2026-03-04T01:00:00Z"))
        # the evaluation leaves out the cell's own answer
        self.assertFalse(blcap.resident(late, "b", "2026-03-04T01:00:00Z", exclude={"2026-03-03"})["resident"])

    def test_resident_rule(self):
        three_of_four = {"a": [("2026-01-01", 1), ("2026-01-02", 1), ("2026-01-03", 0), ("2026-01-04", 1)]}
        self.assertTrue(blcap.resident(three_of_four, "a", "2026-01-05")["resident"])          # 0.75
        self.assertFalse(blcap.resident(three_of_four, "a", "2026-01-04")["resident"])         # 2 of 3
        self.assertFalse(blcap.resident({"a": [("2026-01-01", 1), ("2026-01-02", 1)]}, "a", "2026-02-01")["resident"])

    def test_owner_answers_with_the_confirmation_queue(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Path(d) / "labels"
            write_answers(lab, "QDH", [("QDH", "a", "2026-01-01", 1, 1, "", "", "owner", ""),
                                       ("QDH", "a", "2026-01-02", 0, "", "", "", "owner", ""),      # cannot judge: no answer
                                       ("QDH", "a", "2026-01-03", 1, 0, "", "", "owner", ""),
                                       ("QDH", "b", "2026-01-01", 1, 1, "", "", "ai", "")])        # not the owner's
            ch = "site_id,section_id,date,verified,verified_occupied,verified_by,verified_note,answered_utc,queue_version\n"
            write_answers(lab, "QDH", [("QDH", "a", "2026-01-03", 1, 1, "owner", "", "2026-01-06T00:00:00Z", "q1"),  # later file wins
                                       ("QDH", "a", "2026-01-01", 0, "", "owner", "", "2026-01-06T00:00:00Z", "q1"),  # withdraws
                                       ("QDH", "a", "2026-01-04", 1, 1, "owner", "", "2026-01-06T00:00:00Z", "q1"),
                                       ("QDH", "a", "2026-01-05", 1, 1, "", "", "", "q1")],                       # not the owner's
                          name="{site}-confirm-answers.csv", head=ch)
            ans = blcap.owner_answers(Path(d), "QDH")
            self.assertEqual(ans, {"a": [("2026-01-03", 1), ("2026-01-04", 1)]})
            # the confirmation answers count toward the mark, before the scene's day only
            self.assertEqual(blcap.resident(ans, "a", "2026-01-04T09:00:00Z")["answers"], 1)

    def test_sites_modes(self):
        sites = json.loads((ROOT / "sat" / "sites.json").read_text(encoding="utf-8"))["sites"]
        modes = {s["id"]: blcap.mode(s) for s in sites}
        self.assertEqual(modes["ZJG"], "off")                                   # Zhanjiang off
        self.assertEqual(sorted(k for k, v in modes.items() if v == "trial"), ["QDH", "QGZ"])
        self.assertEqual(blcap.mode({"id": "X"}), "off")                         # the default
        with self.assertRaises(ValueError):
            blcap.mode({"id": "X", "quay": {"blcap": "on"}})

    def test_margin_is_the_designs(self):
        self.assertEqual((blcap.MARGIN_DB, blcap.QUANTILE, blcap.MIN_LENGTH_M), (3.0, 25.0, 50.0))
        self.assertEqual(blcap.ship_length(120, 2), 37.5)        # (120 - 20 - 25) / 2
        self.assertEqual(blcap.ship_length(0, 1), 0.0)
        self.assertEqual(blcap.ship_length(30, 1), 20.0)          # at least 20 m


if np is not None:
    from sat import baseline, cdse, png, quay, quay_lines, run, s2_labels
    from tests.test_sat_quay import SITE, FakeClient, scene_parts, ships


def burnt_base(mask, cvv, cvh):
    """The baseline with the pier's ship burnt in (it lay there in most baseline scenes)."""
    bvv, bvh = cvv.copy(), cvh.copy()
    bvv[27:29, 14:23] = 5.0
    bvh[27:29, 14:23] = -5.0
    return {"vv_p10": baseline.smooth_db(bvv), "vh_p10": baseline.smooth_db(bvh)}


@unittest.skipUnless(np is not None, "numpy not installed")
class CapTest(unittest.TestCase):
    def setUp(self):
        self.g, self.mask, self.lines, self.cvv, self.cvh, self.base = scene_parts()
        self.geo = quay.Geometry(self.g, self.mask, self.lines)

    def test_default_is_off_and_the_count_is_the_same(self):
        vv, vh = ships(self.cvv, self.cvh)
        self.assertIs(quay.cap_baseline(self.geo, self.base), self.base)
        a = quay.count(self.geo, vv, vh, self.base)
        b = quay.count(self.geo, vv, vh, quay.cap_baseline(self.geo, self.base))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))
        # a ceiling far above the baseline changes nothing either
        c = quay.count(self.geo, vv, vh, quay.cap_baseline(self.geo, self.base, 99.0))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(c, sort_keys=True, default=str))

    def test_cap_finds_the_burnt_in_ship(self):
        vv, vh = ships(self.cvv, self.cvh)
        base = burnt_base(self.mask, self.cvv, self.cvh)
        pier = lambda q: [s for s in q["segments"] if s["kind"] == "pier"]
        self.assertEqual(pier(quay.count(self.geo, vv, vh, base)), [])        # the published count misses it
        capped = quay.cap_baseline(self.geo, base, blcap.MARGIN_DB)
        self.assertFalse(np.shares_memory(capped["vv_p10"], base["vv_p10"]))  # the baseline itself is not changed
        self.assertGreater(float(np.nanmax(base["vv_p10"][27:29, 14:23])), float(np.nanmax(capped["vv_p10"][27:29, 14:23])))
        got = pier(quay.count(self.geo, vv, vh, capped))
        self.assertEqual(len(got), 1)
        self.assertGreaterEqual(blcap.ship_length(got[0]["length_m"], got[0]["min"]), 50)


class Frozen(datetime):
    """The wall clock of the run, frozen (the records carry the time they were fetched and written)."""
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 30, 5, 17, tzinfo=timezone.utc)


@unittest.skipUnless(np is not None, "numpy not installed")
class RunTrialTest(unittest.TestCase):
    """The daily run: with quay.blcap trial, only quay_blcap/<site>.json is added; every published file, the requests and
    the PU are the same as with off."""

    def setup_dirs(self, d: Path):
        g, mask, lines, cvv, cvh, _ = scene_parts()
        lines_dir, bl_dir = d / "masks", d / "baselines"
        lines_dir.mkdir()
        quay_lines.path("TQY", 1, lines_dir).write_text(json.dumps({"version": "TQY-quay-v1", **lines}), encoding="utf-8")
        base = burnt_base(mask, cvv, cvh)
        layers = {"vv_p10": base["vv_p10"], "vv_p90": base["vv_p10"] + 2, "vh_p10": base["vh_p10"], "n_valid": np.full(mask.shape, 30)}
        scenes = [{"acquired_utc": f"2026-09-{k + 1:02d}T10:48:45Z", "used": True} for k in range(30)]
        baseline.save(SITE, g, 157, layers, scenes, baseline.site_config(SITE), mask, bl_dir, "2026-10-30T00:00:00Z")
        vv, vh = ships(cvv, cvh)
        enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv), enc(vh), np.full(vv.shape, 4500.0), np.ones(vv.shape)], -1).astype(np.uint16)
        pier_secs = [sc["section_id"] for sc in s2_labels.sections(quay.Geometry(g, mask, lines)) if sc["kind"] == "pier"]
        return mask, png.encode(arr, depth=16, filters=[1]), lines_dir, bl_dir, pier_secs

    def run_once(self, d: Path, mode: str, answer_days):
        mask, chip, lines_dir, bl_dir, pier_secs = self.setup_dirs(d)
        out = d / "out"
        write_answers(out / "labels", "TQY", [("TQY", s, day, 1, 1, "", "", "owner", "") for s in pier_secs for day in answer_days])
        site = {**SITE, "quay": {"blcap": mode}}
        cli = FakeClient(chip)
        with mock.patch.object(run, "datetime", Frozen):
            store = run.Store(out, [site])
            run.process_site(cli, store, site, mask, datetime(2026, 10, 30, tzinfo=timezone.utc), None, 100.0, 12,
                             log=lambda *a: None, lines_dir=lines_dir, baseline_dir=bl_dir)
            store.save({"run_utc": "2026-10-30T05:17:00Z", "requests": cli.requests, "processing_units": cli.processing_units,
                        "pu_source": "header", "scenes": 2})
        files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*")) if p.is_file()}
        return out, files, (cli.requests, cli.processing_units, list(cli.processed)), pier_secs

    def test_trial_adds_only_its_file(self):
        before = ["2026-10-01", "2026-10-05", "2026-10-10"]
        with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
            _o, off, req_off, _ = self.run_once(Path(d0), "off", before)
            out, on, req_on, pier_secs = self.run_once(Path(d1), "trial", before)
            self.assertEqual(req_on, req_off)                              # no request, no PU of its own
            self.assertEqual(set(on) - set(off), {"quay_blcap/TQY.json"})
            self.assertEqual(set(off) - set(on), set())
            differ = sorted(k for k in set(on) & set(off) if on[k] != off[k])
            self.assertEqual(differ, [])
            for k in ("quay_counts.csv", "quay_est2.csv", "quay_est2.json", "quay_naval50.csv"):
                if k in off:
                    self.assertEqual(on[k], off[k], k)
            doc = json.loads((out / "quay_blcap" / "TQY.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["version"], blcap.VERSION)
            self.assertEqual(len(doc["scenes"]), 1)                        # the orbit-157 scene (84 has no baseline)
            sc = doc["scenes"][0]
            self.assertEqual(sc["resident_sections"], len(pier_secs))
            self.assertEqual(sc["now_present"], 0)                         # the burnt-in ship is missed as published
            self.assertEqual(sc["blcap_present"], 1)                       # and found with the ceiling
            self.assertTrue(any(s["differs"] for s in sc["sections"]))
            self.assertEqual(len(sc["pieces"]), 1)
            self.assertTrue(all(s["last_answer"] == "2026-10-10" for s in sc["sections"]))

    def test_answers_after_the_scene_give_no_mark(self):
        # the scene is 2026-10-20: answers of that day and later are never read
        with tempfile.TemporaryDirectory() as d:
            out, _f, _r, _ = self.run_once(Path(d), "trial", ["2026-10-20", "2026-10-21", "2026-10-25"])
            sc = json.loads((out / "quay_blcap" / "TQY.json").read_text(encoding="utf-8"))["scenes"][0]
            self.assertEqual((sc["resident_sections"], sc["sections"], sc["pieces"]), (0, [], []))


if __name__ == "__main__":
    unittest.main()
