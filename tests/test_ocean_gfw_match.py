"""ocean/gfw_match.py: marks of the open-sea candidates against GFW's detections (standard library only; no network, no token)."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ocean import gfw_match as gm

ACQ = "2026-10-31T10:08:57Z"
T = gm.parse_utc(ACQ)
BBOX = [119.5, 25.3, 121.0, 26.5]
AREAS = {"TSN": {"id": "TSN", "bbox": BBOX}}


def report_doc(cells, date="2026-10-31 10:00", key="public-global-sar-presence:v5.0"):
    return {"entries": [{key: [{"date": date, "detections": 1, "flag": "", "lat": la, "lon": lo, "vesselIDs": 1} for la, lo in cells]}], "total": 1}


def monitor_dir(root: Path, feats, day="2026-10-31"):
    d = root / "monitor" / "candidates"
    d.mkdir(parents=True)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
         "properties": {"area": area, "acquired_utc": acq, "class": cls}} for lon, lat, cls, area, acq in feats]}
    (d / f"{day}.geojson").write_text(json.dumps(fc), encoding="utf-8")
    return root / "monitor"


class Fake:
    def __init__(self, with_ais=(), without=(), fail=None):
        self.with_ais, self.without, self.fail, self.calls = with_ais, without, fail, []

    def __call__(self, bbox, days, matched):
        self.calls.append((tuple(bbox), days, matched))
        if self.fail:
            raise gm.GfwError(self.fail)
        return report_doc(self.with_ais if matched else self.without)


class GfwMatchTest(unittest.TestCase):
    def test_rows_and_cells(self):
        doc = report_doc([(25.5, 120.0)])
        self.assertEqual(len(gm.rows_of(doc)), 1)
        self.assertEqual(gm.rows_of({"entries": [{"x": None}]}), [])            # a dataset without data
        self.assertEqual(gm.cells(doc, gm.hours_of(T)), [(25.5, 120.0)])
        self.assertEqual(gm.cells(report_doc([(25.5, 120.0)], date="2026-10-31 21:00"), gm.hours_of(T)), [])   # another pass of the day
        self.assertEqual(gm.cells({"entries": [{"k": [{"date": "2026-10-31 10:00", "detections": 0, "lat": 1, "lon": 2}]}]}, gm.hours_of(T)), [])

    def test_hours_near_the_hour_and_date_range(self):
        self.assertEqual(gm.hours_of(gm.parse_utc("2026-10-31T10:30:00Z")), {"2026-10-31 10:00"})
        self.assertEqual(gm.hours_of(T), {"2026-10-31 09:00", "2026-10-31 10:00"})                 # 10:08: the hour before counts too
        self.assertEqual(gm.hours_of(gm.parse_utc("2026-10-31T09:55:00Z")), {"2026-10-31 09:00", "2026-10-31 10:00"})
        self.assertEqual(gm.date_range(T), "2026-10-31,2026-11-01")
        self.assertEqual(gm.date_range(gm.parse_utc("2026-10-31T23:58:00Z")), "2026-10-31,2026-11-02")
        self.assertEqual(gm.date_range(gm.parse_utc("2026-11-01T00:03:00Z")), "2026-10-31,2026-11-02")

    def test_classify(self):
        a, n = [(25.50, 120.00)], [(25.60, 120.10)]
        self.assertEqual(gm.classify(25.503, 120.003, a, n), "ais")         # ~0.45 km from the AIS cell
        self.assertEqual(gm.classify(25.602, 120.099, a, n), "no_ais")
        self.assertEqual(gm.classify(25.55, 120.05, a, n), "none")          # kilometres from both
        self.assertEqual(gm.classify(25.508, 120.0, a, n), "none")          # 0.88 km: outside the radius
        self.assertEqual(gm.classify(25.5, 120.0, [(25.505, 120.0)], [(25.502, 120.0)]), "no_ais")   # the nearer kind wins
        self.assertEqual(gm.classify(25.5, 120.0, [], []), "none")

    def test_due(self):
        self.assertFalse(gm.due(T, None, T + timedelta(days=4, hours=23)))
        self.assertTrue(gm.due(T, None, T + timedelta(days=5, hours=1)))
        done = {"status": "done"}
        self.assertTrue(gm.due(T, done, T + timedelta(days=8)))
        self.assertFalse(gm.due(T, done, T + timedelta(days=9, hours=1)))     # recheck days over
        self.assertTrue(gm.due(T, {"status": "waiting"}, T + timedelta(days=15)))
        self.assertFalse(gm.due(T, {"status": "waiting"}, T + timedelta(days=22)))

    def test_run_marks_counts_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mon = monitor_dir(root, [(120.000, 25.500, "A", "TSN", ACQ), (120.100, 25.600, "A", "TSN", ACQ), (120.300, 25.900, "A", "TSN", ACQ),
                                     (120.001, 25.501, "B", "TSN", ACQ), (114.0, 22.0, "A", "XXX", ACQ)])
            fake = Fake(with_ais=[(25.50, 120.00)], without=[(25.60, 120.10)])
            now = T + timedelta(days=6)
            s = gm.run(mon, root / "gfw", fake, now, AREAS, pause=lambda _: None)
            self.assertEqual((s["asked"], s["done"], s["waiting"], s["failed"]), (1, 1, 0, 0))   # the unknown area is not asked for
            self.assertEqual([c[2] for c in fake.calls], [True, False])
            self.assertEqual(fake.calls[0][:2], (tuple(BBOX), "2026-10-31,2026-11-01"))
            st = json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))
            e = st["passes"][0]
            self.assertEqual((e["status"], e["A_ais"], e["A_no_ais"], e["A_none"], e["B_ais"]), ("done", 1, 1, 1, 1))
            self.assertEqual(st["attribution"], gm.ATTRIBUTION)
            m = json.loads((root / "gfw" / "matches" / "2026-10-31.json").read_text(encoding="utf-8"))
            self.assertEqual(m["passes"][0]["candidates"][0], [120.0, 25.5, "A", "ais"])
            self.assertEqual(sorted(c[3] for c in m["passes"][0]["candidates"]), ["ais", "ais", "no_ais", "none"])
            rows = json.loads((root / "gfw" / "series.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(rows[0]["A_no_ais"], 1)
            self.assertIn("Powered by Global Fishing Watch", json.loads((root / "gfw" / "meta.json").read_text(encoding="utf-8"))["attribution"])
            # GFW's own rows and identifiers are not in what is written
            text = "".join(p.read_text(encoding="utf-8") for p in (root / "gfw").rglob("*.json"))
            self.assertNotIn("vesselIDs", text)
            self.assertNotIn("detections", text)
            # the recheck days over: not asked again, files unchanged
            fake2 = Fake()
            s2 = gm.run(mon, root / "gfw", fake2, T + timedelta(days=12), AREAS, pause=lambda _: None)
            self.assertEqual((s2["asked"], fake2.calls), (0, []))

    def test_waiting_then_done_and_never_taken_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mon = monitor_dir(root, [(120.0, 25.5, "A", "TSN", ACQ)])
            s = gm.run(mon, root / "gfw", Fake(), T + timedelta(days=5, hours=2), AREAS, pause=lambda _: None)
            self.assertEqual((s["done"], s["waiting"]), (0, 1))
            self.assertEqual(json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))["passes"][0]["status"], "waiting")
            self.assertFalse((root / "gfw" / "matches").exists())
            gm.run(mon, root / "gfw", Fake(with_ais=[(25.5, 120.0)]), T + timedelta(days=6), AREAS, pause=lambda _: None)
            e = json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))["passes"][0]
            self.assertEqual((e["status"], e["A_ais"]), ("done", 1))
            first = e["first_done_utc"]
            gm.run(mon, root / "gfw", Fake(), T + timedelta(days=7), AREAS, pause=lambda _: None)   # an empty answer later
            e = json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))["passes"][0]
            self.assertEqual((e["status"], e["A_ais"], e["first_done_utc"]), ("done", 1, first))

    def test_no_gfw_after_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mon = monitor_dir(root, [(120.0, 25.5, "A", "TSN", ACQ)])
            gm.run(mon, root / "gfw", Fake(), T + timedelta(days=20, hours=12), AREAS, pause=lambda _: None)
            self.assertEqual(json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))["passes"][0]["status"], "no_gfw")

    def test_errors_stop_and_keep_the_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feats = [(120.0, 25.5, "A", "TSN", f"2026-10-{d:02d}T10:08:57Z") for d in (25, 26, 27, 28, 29)]
            mon = monitor_dir(root, feats)
            fake = Fake(fail="HTTP 401")
            s = gm.run(mon, root / "gfw", fake, T + timedelta(days=6), AREAS, pause=lambda _: None)
            self.assertEqual((s["failed"], s["done"]), (3, 0))          # stops after three failures in a row from the start
            self.assertEqual(len(fake.calls), 3)
            self.assertEqual(json.loads((root / "gfw" / "passes.json").read_text(encoding="utf-8"))["passes"], [])

    def test_main_without_token(self):
        import contextlib
        import io
        import os
        old = os.environ.pop("GFW_API_TOKEN", None)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(gm.main([]), 0)
            self.assertIn("GFW_API_TOKEN", buf.getvalue())
        finally:
            if old is not None:
                os.environ["GFW_API_TOKEN"] = old


if __name__ == "__main__":
    unittest.main()
