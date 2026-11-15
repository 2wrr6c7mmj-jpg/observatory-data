"""sat/close_pairs: the S1 scenes paired with a clear S2 image within a day are found, kept and answered."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sat import close_pairs as cp


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def s2row(site, acq, result="clear", image="x.webp"):
    return {"site_id": site, "acquired_utc": acq, "result": result, "image": image}


def n50(site, acq, w=5, s=3):
    return {"site_id": site, "acquired_utc": acq, "naval50_version": "naval50-2", "stretches_50": s, "warship_50": w,
            "warship_est2": 2, "ships_all": 9, "min_length_m": "50"}


class ClosePairsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_within_a_day_only(self):
        rows = [n50("QGZ", "2026-11-01T10:00:00Z"), n50("QGZ", "2026-11-13T10:00:00Z")]
        s2 = {"QGZ": [("2026-11-01T03:00:00Z", "a"), ("2026-11-02T03:00:00Z", "b"), ("2026-11-05T03:00:00Z", "c")]}
        got = [(r["acquired_utc"], a, dt) for r, a, _, dt in cp.find_pairs(rows, s2)]
        self.assertEqual(got, [("2026-11-01T10:00:00Z", "2026-11-01T03:00:00Z", 7.0),
                               ("2026-11-01T10:00:00Z", "2026-11-02T03:00:00Z", -17.0)])

    def test_update_keeps_pairs_and_boxes_after_the_scene_leaves(self):
        write_csv(self.out / "s2_scenes.csv", [s2row("WHI", "2026-11-10T03:00:00Z"), s2row("WHI", "2026-11-12T03:00:00Z", "cloudy")])
        write_csv(self.out / "quay_blind" / "WHI.csv", [{"site_id": "WHI", "acquired_utc": "2026-11-09T10:00:00Z", "warship_blind": 1}])
        write_csv(self.out / "labels" / "ai-port-counts.csv",
                  [{"site_id": "WHI", "s2_utc": "2026-11-10T03:00:00Z", "ai_min": 0, "ai_max": 1, "verified_by": "ai"}])
        rows = [n50("WHI", "2026-11-09T10:00:00Z", w=3)]
        boxes = {"WHI": {"2026-11-09T10:00:00Z": [{"n": 1, "box_lonlat": [1, 2, 3, 4], "ships": 1.5, "length_m": 80}]}}
        t0 = datetime(2026, 11, 15, tzinfo=timezone.utc)
        self.assertEqual(cp.update(self.out, rows, boxes, now=t0), 1)
        r = cp._read(self.out / cp.PAIRS_FILE)[0]
        self.assertEqual((r["dt_h"], r["warship_50"], r["warship_blind"], r["card"]), ("-17", "3", "1", "4"))
        self.assertEqual((r["ans_min"], r["ans_max"], r["ans_by"], r["recorded_utc"]), ("0", "1", "ai", "2026-11-15T00:00:00Z"))
        # the next day the scene is no longer stored: the pair and its boxes stay, the date of recording too
        self.assertEqual(cp.update(self.out, [n50("WHI", "2026-11-21T10:00:00Z")], {}, now=datetime(2026, 11, 22, tzinfo=timezone.utc)), 1)
        r = cp._read(self.out / cp.PAIRS_FILE)[0]
        self.assertEqual((r["s1_utc"], r["card"], r["recorded_utc"]), ("2026-11-09T10:00:00Z", "4", "2026-11-15T00:00:00Z"))
        doc = json.loads((self.out / cp.PAIRS_DIR / "WHI.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["scenes"]["2026-11-09T10:00:00Z"]["boxes"][0]["ships"], 1.5)

    def test_no_pairs_no_file(self):
        write_csv(self.out / "s2_scenes.csv", [s2row("WHI", "2026-11-01T03:00:00Z")])
        self.assertEqual(cp.update(self.out, [n50("WHI", "2026-11-09T10:00:00Z")], {}), 0)
        self.assertFalse((self.out / cp.PAIRS_FILE).exists())


if __name__ == "__main__":
    unittest.main()
