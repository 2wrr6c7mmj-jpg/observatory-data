"""sat/s1_recount.py: S1 only for the answered days, with its own 40 PU cap and the month's guard (no request here)."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cdse, png, s1_recount as r
    from sat.grid import Grid
    from sat.port_archive import LimitedClient

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
SITE = {"id": "TSR", "bbox": [110.0, 21.0, 110.01, 21.01], "pixel_m": 10}


def chip(h, w):
    enc = lambda x: np.clip(np.round((x + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
    vv = np.full((h, w), -20.0)
    vh = np.full((h, w), -26.0)
    return png.encode(np.stack([enc(vv), enc(vh), np.ones((h, w))], -1).astype(np.uint16), depth=16, filters=[1])


class Fake(LimitedClient if np is not None else object):   # no numpy in the collect workflows
    def __init__(self, cap, **kw):
        super().__init__(cap, 100, 1000.0, 1000, budget_month="2026-11", utcnow=lambda: NOW, sleep=lambda s: None,
                         client_id="x", client_secret="y")
        self.sent = []

    def catalog(self, bbox, start, end, collection=None):
        self.requests += 1
        t = start + (end - start) / 2
        return [{"id": "g", "properties": {"platform": "sentinel-1c", "sat:orbit_state": "ascending", "sar:instrument_mode": "IW",
                                           "sar:polarizations": ["VV", "VH"], "sat:relative_orbit": 157,
                                           "datetime": cdse.iso(t)}}]

    def _post_sh(self, url, payload, accept):
        est = self.next_est
        self.next_est, self.next_key = 0.0, None
        self._gate(est)
        self.requests += 1
        self.processing_units += est
        out = payload["output"]
        self.sent.append(out)
        return {"x-processingunits-spent": str(est)}, chip(out["height"], out["width"])


def write_plan(out: Path, sid: str, times):
    d = out / "labels" / sid
    d.mkdir(parents=True)
    (d / "s1_recount_plan.json").write_text(json.dumps({"scenes": [{"acquired_utc": t, "relative_orbit": "157"} for t in times]}))


@unittest.skipUnless(np is not None, "numpy not installed")
class RecountTest(unittest.TestCase):
    def run_main(self, d: Path, extra=(), client=None):
        args = ["--site", "TSR", "--out", str(d / "out"), "--raw", str(d / "raw"), "--budget-month", "2026-11",
                "--global-cap", "9000", "--global-margin", "450", "--ocean-cap", "7000",
                "--usage-out", str(d / "out" / "s1_recount_usage.csv"), *extra]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = r.main(args, client=client, sites=[SITE], utcnow=lambda: NOW)
        return rc, json.loads(buf.getvalue().strip().splitlines()[-1])

    def test_fetches_saves_and_records_usage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_plan(d / "out", "TSR", ["2026-10-14T10:48:16Z", "2026-10-26T10:48:16Z"])
            c = Fake(40.0)
            rc, s = self.run_main(d, client=c)
            self.assertEqual(rc, 0)
            self.assertEqual(len(s["sites"][0]["fetched"]), 2)
            g = Grid.for_site(SITE)
            vv, vh, meta = r.load(r.raw_path(d / "raw", "TSR", "2026-10-14T10:48:16Z"))
            self.assertEqual(vv.shape, (g.height, g.width))
            self.assertAlmostEqual(float(np.nanmedian(vv)), -20.0, delta=0.05)
            self.assertIn("Copernicus", meta["credit"])
            self.assertEqual(c.sent[0]["width"], g.width)
            rows = list(csv.DictReader((d / "out" / "s1_recount_usage.csv").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["probe_version"], r.PROBE_VERSION)
            # a second run fetches nothing (saved) and sends no request
            c2 = Fake(40.0)
            rc, s = self.run_main(d, client=c2)
            self.assertEqual((s["sites"][0]["fetched"], c2.requests), ([], 0))

    def test_total_cap_counts_earlier_runs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_plan(d / "out", "TSR", ["2026-10-14T10:48:16Z"])
            (d / "out" / "s1_recount_usage.csv").write_text(
                "run_utc,site_id,probe_version,requests,processing_units\n2026-11-03T01:00:00Z,SYL+SLP,s1-recount-1,10,39.99\n")
            rc, s = self.run_main(d, client=Fake(0.01))
            self.assertEqual(s["cap_this_run"], 0.01)
            self.assertEqual(s["sites"], [])
            self.assertIn("上限", s["stopped"])

    def test_constants_are_explicit(self):
        self.assertEqual(r.MAX_PU_TOTAL, 40.0)
        self.assertEqual(r.BUDGET_MONTH, "2026-11")

    def test_other_month_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_plan(d / "out", "TSR", ["2026-10-14T10:48:16Z"])
            args = ["--site", "TSR", "--out", str(d / "out"), "--raw", str(d / "raw"), "--budget-month", "2026-12",
                    "--global-cap", "9000", "--global-margin", "450", "--ocean-cap", "7000", "--usage-out", str(d / "u.csv")]
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(r.main(args, client=Fake(40.0), sites=[SITE], utcnow=lambda: NOW), 1)
            self.assertFalse((d / "raw").exists())

    def test_month_left(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sat = d / "sat.csv"
            sat.write_text("run_utc,requests,processing_units\n2026-11-02T01:00:00Z,10,100\n2026-10-30T01:00:00Z,10,999\n")
            oc = d / "other-docs_data_ocean_usage.csv--0.csv"
            oc.write_text("run_utc,site_id,probe_version,requests,processing_units\n2026-11-02T02:00:00Z,x,o,5,2000\n")
            pt = d / "ports-all--0.csv"
            pt.write_text("run_utc,site_id,archive_version,requests,processing_units\n2026-11-01T02:00:00Z,SYL,ports-1,5,250\n")
            m = r.month_left("2026-11", NOW, 9000, 450, 7000, [sat, oc], [pt])
            # 9000 - 450 - (100 + 2000) - (7000 - 2000) - 250
            self.assertEqual(m["left"], 1200.0)
            with self.assertRaises(ValueError):
                r.month_left("2026-12", NOW, 9000, 450, 7000, [], [])


if __name__ == "__main__":
    unittest.main()
