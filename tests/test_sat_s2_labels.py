import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import baseline, cdse, png, quay, quay_lines, run, s2, s2_labels
    from sat.osm_mask import mask_paths
    from tests.test_sat_quay import SITE, scene_parts, ships
    from tests.test_sat_s2 import tar_of

# clear days (S2 at 03:22 UTC); the pier ship on some, two abreast on one; 10-28 is frozen
DAYS = ["2026-09-01", "2026-09-06", "2026-09-11", "2026-09-16", "2026-09-21", "2026-09-26", "2026-10-01",
        "2026-10-06", "2026-10-11", "2026-10-28"]
SHIP = {"2026-09-06": 1, "2026-09-16": 1, "2026-10-01": 2}


def s2_chip(mask, date):
    rng = np.random.default_rng(int(date[-2:]))
    h, w = mask.shape
    refl = np.zeros((h, w, 4), float)
    refl[:] = [500, 600, 400, 200]                  # water: B2 B3 B4 B8
    refl[mask == 1] = [1200, 1300, 1400, 2500]      # land
    refl[mask == 2] = [1500, 1500, 1500, 1500]      # pier deck
    refl += rng.normal(0, 15, refl.shape)
    n = SHIP.get(date, 0)
    if n:                                           # grey hull(s) north of the pier: rows 27-28 (and 25-26)
        refl[29 - 2 * n:29, 14:23] = [900, 920, 930, 950]
    scl = np.where(mask == 0, 6, 5).astype(np.uint8)
    return np.clip(refl, 1, 65535).astype(np.uint16), scl


class FakeClient:
    def __init__(self, mask, chip):
        self.mask, self.chip = mask, chip
        self.requests, self.processing_units, self.pu_reported = 0, 0.0, True
        self.calls = []

    def catalog(self, bbox, start, end, limit=100, collection="sentinel-1-grd"):
        self.requests += 1
        p = {"platform": "sentinel-1c", "sat:orbit_state": "ascending", "sar:instrument_mode": "IW",
             "sar:polarizations": ["VV", "VH"], "sat:relative_orbit": 157}
        return [{"id": d, "properties": {**p, "datetime": f"{d}T10:48:45Z"}}
                for d in ("2026-09-02", "2026-09-07", "2026-09-17", "2026-10-02", "2026-10-07", "2026-10-27")]

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 1.2
        self.calls.append(("s1", cdse.iso(t0)[:10]))
        return self.chip, 1.2

    def process_raw(self, payload, accept, input_bands):
        self.requests += 1
        self.processing_units += 1.5
        date = payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10]
        self.calls.append(("s2", date))
        refl, scl = s2_chip(self.mask, date)
        return tar_of({"refl.png": png.encode(refl, depth=16), "scl.png": png.encode(scl)}), 1.5


@unittest.skipUnless(np is not None, "numpy not installed")
class S2LabelsTest(unittest.TestCase):
    def setup(self, d: Path):
        g, mask, lines, cvv, cvh, base = scene_parts()
        mdir, bdir, out = d / "masks", d / "baselines", d / "out"
        mdir.mkdir()
        p_png, p_json = mask_paths(SITE["id"], 1, mdir)
        p_png.write_bytes(png.encode(mask))
        p_json.write_text("{}", encoding="utf-8")
        quay_lines.path("TQY", 1, mdir).write_text(json.dumps({"version": "TQY-quay-v1", **lines}), encoding="utf-8")
        layers = {"vv_p10": base["vv_p10"], "vv_p90": base["vv_p10"] + 2, "vh_p10": base["vh_p10"],
                  "n_valid": np.full(mask.shape, 30)}
        scenes = [{"acquired_utc": "2026-09-07T10:48:45Z", "used": True}]
        baseline.save(SITE, g, 157, layers, scenes, baseline.site_config(SITE), mask, bdir, "2026-10-30T00:00:00Z")
        vv, vh = ships(cvv, cvh)
        enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        chip = png.encode(np.stack([enc(vv), enc(vh), np.ones(vv.shape)], -1).astype(np.uint16), depth=16)
        rows = [{"site_id": "TQY", "mask_version": "TQY-land-v1", "date": dd, "acquired_utc": f"{dd}T03:22:10Z",
                 "result": "clear"} for dd in DAYS]
        s2.write_rows(out / "s2_scenes.csv", rows)
        return mask, chip, mdir, bdir, out

    def test_helpers(self):
        self.assertEqual(s2_labels.color_class(0.05, 0.2), "grey")
        self.assertEqual(s2_labels.color_class(0.4, 0.2), "colour")
        self.assertEqual(s2_labels.color_class(0.2, 1.2), "contrast")
        self.assertEqual(s2_labels.color_class(0.2, 0.2), "mixed")
        f = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]], bool)
        self.assertEqual(s2_labels.leading_depth(f, np.array([5.0, 10, 15, 20]), 5).tolist(), [12.5, 0.0, 22.5])
        acqs = [{"acquired_utc": "2026-09-02T10:48:45Z", "relative_orbit": 157},
                {"acquired_utc": "2026-09-01T10:41:00Z", "relative_orbit": 84},
                {"acquired_utc": "2026-10-27T10:41:00Z", "relative_orbit": 157}]
        t = cdse.parse_time("2026-09-01T03:22:00Z")
        self.assertEqual(s2_labels.nearest_s1(acqs, t, {"157"})["acquired_utc"], "2026-09-02T10:48:45Z")
        self.assertEqual(s2_labels.nearest_s1(acqs, t, {"157", "84"})["relative_orbit"], 84)
        self.assertIsNone(s2_labels.nearest_s1(acqs, cdse.parse_time("2026-10-26T03:22:00Z"), {"157"}))  # frozen

    def test_build_labels_and_match(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, mdir, bdir, out = self.setup(d)
            cli = FakeClient(mask, chip)
            raw = d / "raw"
            s = s2_labels.build(SITE, out, raw, cli, 100.0, mdir, bdir, log=lambda *a: None)
            self.assertEqual(s["s2_fetched"], 9)                   # 10-28 frozen
            self.assertNotIn(("s2", "2026-10-28"), cli.calls)
            self.assertNotIn(("s1", "2026-10-27"), [c[:2] for c in cli.calls])
            self.assertEqual(s["s1_fetched"], 5)                   # 09-11 shares 09-07, 09-21 shares 09-17, 10-11 shares 10-07; 09-26 none
            rows = list(csv.DictReader((out / "labels" / "TQY-s2-labels.csv").read_text(encoding="utf-8").splitlines()))
            secs = {r["section_id"] for r in rows}
            self.assertEqual(len(rows), 9 * len(secs))
            self.assertTrue(all(r["verified"] == "" and r["label_source"].startswith("auto") for r in rows))
            north = [sid for sid in secs if sid.startswith("2:")]
            occ = {(r["date"], r["section_id"]): r for r in rows if r["occupied"] == "1"}
            dates = sorted({k[0] for k in occ})
            self.assertEqual(dates, ["2026-09-06", "2026-09-16", "2026-10-01"])
            one = [r for (dd, sid), r in occ.items() if dd == "2026-09-06"]
            self.assertTrue(one and all(r["c"] == "1" and r["section_id"] in north for r in one))
            two = [r for (dd, sid), r in occ.items() if dd == "2026-10-01"]
            self.assertTrue(two and all(r["c"] == "2" for r in two))
            self.assertEqual(one[0]["color_class"], "grey")
            self.assertAlmostEqual(float(one[0]["med_nir"]), 0.095, delta=0.01)
            m = list(csv.DictReader((out / "labels" / "TQY-s2-s1-match.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(m), len(rows))
            r = [x for x in m if x["s2_date"] == "2026-09-06" and x["section_id"] in north and x["s2_occupied"] == "1"][0]
            self.assertEqual(r["s1_acquired_utc"], "2026-09-07T10:48:45Z")
            self.assertEqual(r["s1_in_sample"], "baseline")
            self.assertEqual(r["s1_occupied"], "1")                # the S1 chip has the pier ship
            self.assertEqual(r["agree"], "1")
            self.assertEqual(r["dt_h"], "31.4")
            usage_free = not (out / "usage.csv").exists()          # build() never writes usage (main does)
            self.assertTrue(usage_free)
            meta = json.loads((out / "labels" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["method"]["version"], "s2lab-1")
            # again from the saved chips: no request, same labels
            before = (out / "labels" / "TQY-s2-labels.csv").read_text(encoding="utf-8")
            s2_labels.build(SITE, out, raw, None, 100.0, mdir, bdir, log=lambda *a: None)
            self.assertEqual((out / "labels" / "TQY-s2-labels.csv").read_text(encoding="utf-8"), before)

    def test_pu_limit_defers(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, mdir, bdir, out = self.setup(d)
            cli = FakeClient(mask, chip)
            s = s2_labels.build(SITE, out, d / "raw", cli, 4.0, mdir, bdir, log=lambda *a: None)
            self.assertLessEqual(cli.processing_units, 4.0)
            self.assertTrue(s["deferred"])

    def test_main_records_usage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask, chip, mdir, bdir, out = self.setup(d)
            cli = FakeClient(mask, chip)
            orig = s2_labels.build
            s2_labels.build = lambda *a, **k: orig(*a[:4], max_pu=a[4] if len(a) > 4 else k.get("max_pu", 100.0),
                                                   mask_dir=mdir, baseline_dir=bdir, log=lambda *x: None,
                                                   budget=k.get("budget", 1e9))
            try:
                s2_labels.main(["--out", str(out), "--site", "TQY", "--raw", str(d / "raw"), "--max-pu", "50"],
                               client=cli, sites=[SITE])
            finally:
                s2_labels.build = orig
            u = run.read_csv(out / "usage.csv")
            self.assertEqual(len(u), 1)
            self.assertAlmostEqual(float(u[0]["processing_units"]), cli.processing_units, places=3)
            self.assertAlmostEqual(float(u[0]["s2_processing_units"]), 9 * 1.5, places=3)


if __name__ == "__main__":
    unittest.main()
