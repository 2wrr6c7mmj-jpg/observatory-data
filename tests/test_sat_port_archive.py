import contextlib
import io
import json
import tarfile
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import cdse, png, port_archive as pa
    from sat.grid import Grid

SITE = {"id": "TPA", "name": "試験港", "bbox": [120.0, 30.0, 120.005, 30.004], "pixel_m": 10, "mask_version": 1,
        "enabled": True, "priority": 1}
ROOT = Path(__file__).resolve().parent.parent


class Resp:
    def __init__(self, body, headers):
        self.status, self._body, self.headers = 200, body, headers

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def tar_of(parts):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in parts.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class Fake:
    """Token, catalog (S1: 4 acquisitions on 2 orbits; S2: 2 days) and process for one small site."""

    def __init__(self, grid, fail_429=0):
        self.grid = grid
        self.fail_429 = fail_429
        self.process_calls = 0

    def chip(self):
        h, w = self.grid.height, self.grid.width
        vv = np.full((h, w), -24.0)
        vh = np.full((h, w), -30.0)
        vv[:, : w // 4] = -5.0  # land on the west
        vh[:, : w // 4] = -12.0
        vv[20:24, 25:28] = 5.0  # a ship
        vh[20:24, 25:28] = -5.0
        enc = lambda d: np.clip(np.round((d + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535)
        arr = np.stack([enc(vv), enc(vh), np.full((h, w), 3500.0), np.ones((h, w))], axis=-1)
        return png.encode(arr.astype(np.uint16), depth=16)

    def __call__(self, req, timeout=None):
        url = req.full_url
        if url == cdse.TOKEN_URL:
            return Resp(json.dumps({"access_token": "t", "expires_in": 600}).encode(), {})
        body = json.loads(req.data)
        if url == cdse.CATALOG_URL:
            if body["collections"] == ["sentinel-2-l2a"]:
                feats = [{"id": d, "properties": {"datetime": d + "T03:00:00Z", "eo:cloud_cover": 5, "platform": "sentinel-2a"}}
                         for d in ("2026-06-01", "2026-09-01")]
            else:
                p = {"sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
                full = {"type": "Polygon", "coordinates": [[[119, 29], [121, 29], [121, 31], [119, 31], [119, 29]]]}
                # the swath edge: covers only the western strip (mostly land) of the site
                edge = {"type": "MultiPolygon", "coordinates": [[[[119, 29], [120.0015, 29], [120.0015, 31], [119, 31], [119, 29]]]]}
                feats = [{"id": f"s{i}", "geometry": g, "properties": {**p, "platform": "sentinel-1a", "datetime": t,
                                                                       "sat:orbit_state": o, "sat:relative_orbit": r}}
                         for i, (t, o, r, g) in enumerate([("2026-01-02T10:00:00Z", "ascending", 11, full),
                                                           ("2026-03-02T22:00:00Z", "descending", 40, full),
                                                           ("2026-05-02T10:00:00Z", "ascending", 11, full),
                                                           ("2026-06-02T10:00:00Z", "ascending", 157, edge),
                                                           ("2026-07-02T10:00:00Z", "ascending", 11, full)])]
            return Resp(json.dumps({"features": feats}).encode(), {})
        self.process_calls += 1
        if self.fail_429 > 0:
            self.fail_429 -= 1
            raise urllib.error.HTTPError(url, 429, "slow", {"Retry-After": "1"}, io.BytesIO(b"rate"))
        data = body["input"]["data"][0]
        out = body["output"]
        if data["type"] == "sentinel-1-grd":
            return Resp(self.chip(), {"x-processingunits-spent": "0.5"})
        h, w = out["height"], out["width"]
        scl = png.encode(np.full((h, w), 6, np.uint8))
        if len(out["responses"]) == 1:
            return Resp(scl, {"x-processingunits-spent": "0.01"})
        refl = png.encode(np.full((h, w, 4), 500, np.uint16), depth=16)
        return Resp(tar_of({"refl.png": refl, "scl.png": scl}), {"x-processingunits-spent": "0.2"})


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


@unittest.skipUnless(np is not None, "numpy not installed")
class PortsFileTest(unittest.TestCase):
    def test_ports_json(self):
        ports = pa.load_ports()
        ids = [p["id"] for p in ports]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("ZJG", ids)
        sites = {s["id"]: s for s in json.loads((ROOT / "sat" / "sites.json").read_text(encoding="utf-8"))["sites"]}
        # sites.json が唯一の設定（PR #30）。アーカイブ用に ports.json に残る港（SYL・SLP）は、日次の地点
        # （daily が false でないもの）であってはならず、範囲・中心・陸マスクの版は sites.json と同じであること。
        for i in set(ids) & set(sites):
            p, s = next(p for p in ports if p["id"] == i), sites[i]
            self.assertIs(s.get("daily", True), False, i)
            for k in ("bbox", "center_lat", "center_lon", "mask_version"):
                self.assertEqual(p.get(k), s.get(k), (i, k))
        for p in ports:
            g = Grid.for_site(p)
            self.assertTrue(2 <= g.width * 10 / 1000 <= 10 and 2 <= g.height * 10 / 1000 <= 10, p["id"])
            self.assertTrue(p["sources"] and all(s.get("url", "").startswith("https://") for s in p["sources"]), p["id"])
            w, s, e, n = p["bbox"]
            self.assertTrue(w <= p["center_lon"] <= e and s <= p["center_lat"] <= n, p["id"])


class BudgetArgsTest(unittest.TestCase):
    """plan has no default caps, and stops before any request outside the month the caps were set for."""

    def test_caps_are_required(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            for drop in ("--cap", "--global-cap", "--ocean-cap", "--budget-month"):
                args = {"--cap": "1", "--global-cap": "1", "--ocean-cap": "1", "--budget-month": "2026-10"}
                del args[drop]
                with self.assertRaises(SystemExit, msg=drop):
                    pa.main(["plan", "--out", str(Path(d) / "p.json")] + [x for kv in args.items() for x in kv])

    def test_other_month_stops(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()) as err, \
                mock.patch.object(pa.cdse, "Client", side_effect=AssertionError("no request")):
            rc = pa.main(["plan", "--out", str(Path(d) / "p.json"), "--cap", "4750", "--global-cap", "9995",
                          "--ocean-cap", "4600", "--budget-month", "1999-01"])
            self.assertEqual(rc, 2)
            self.assertFalse((Path(d) / "p.json").exists())
            self.assertIn("1999-01", err.getvalue())


    def test_run_and_baseline_need_the_month(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            d = Path(d)
            run = ["run", "--port", "SYL", "--cap", "1", "--raw-dir", str(d), "--tmp", str(d), "--mask-dir", str(d)]
            base = ["baseline", "--only", "SYL", "--cap", "1"]
            for argv in (run, base):
                with self.assertRaises(SystemExit, msg=argv[0]):
                    pa.main(argv)
            with mock.patch.object(pa, "LimitedClient", side_effect=AssertionError("no request")):
                for argv in (run, base):
                    self.assertEqual(pa.main(argv + ["--budget-month", "1999-01"]), 2, argv[0])


def fake_git(trees):
    """A git runner over {ref: {path: text}} for gather_usage."""
    def git(*args):
        if args[0] == "for-each-ref":
            return "\n".join(["origin/HEAD"] + list(trees)) + "\n"
        if args[0] == "ls-tree":
            return "\n".join(p for p in trees[args[3]] if p.startswith(args[4])) + "\n"
        if args[0] == "show":
            ref, path = args[1].split(":", 1)
            return trees[ref][path]
        raise AssertionError(args)
    return git


USAGE_HEAD = "run_utc,site_id,requests,processing_units\n"


@unittest.skipUnless(np is not None, "numpy not installed")
class PortsUsageTest(unittest.TestCase):
    """The archive's own usage is read from every branch (largest copy of each file), so a new probe branch cut
    from main does not start again from 0 PU."""

    def trees(self):
        syl_old = USAGE_HEAD + "2026-10-20T00:00:00Z,SYL,10,60\n"
        syl_new = syl_old + "2026-10-29T00:00:00Z,SYL,10,40\n2026-09-29T00:00:00Z,SYL,10,999\n"
        return {
            "origin/main": {"docs/data/sat/usage.csv": USAGE_HEAD + "2026-10-01T00:00:00Z,ZJG,5,7\n"},
            "origin/ops3/ports-archive": {"docs/data/sat/ports/SYL/usage.csv": syl_old,
                                          "docs/data/sat/ports_usage.csv": syl_old},
            "origin/probe/ports-archive-ops3": {"docs/data/sat/ports/SYL/usage.csv": syl_new,
                                                "docs/data/sat/ports/QGZ/usage.csv": USAGE_HEAD + "2026-10-29T00:00:00Z,QGZ,3,25\n",
                                                "docs/data/sat/ports/masks/x.json": "{}",
                                                "docs/data/sat/ports_usage.csv": syl_new,
                                                "docs/data/ocean/usage.csv": USAGE_HEAD + "2026-10-02T00:00:00Z,ocean,4,30\n"},
            "origin/ops/ocean-coverage-probe": {"docs/data/ocean/usage.csv": USAGE_HEAD + "2026-10-02T00:00:00Z,ocean,4,30\n2026-10-03T00:00:00Z,ocean,4,20\n"},
        }

    def test_gather_and_guard(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            n = pa.gather_usage(d / "other", d / "ports", git=fake_git(self.trees()), log=lambda s: None)
            self.assertEqual(n, {"other": 2, "ports": 3})
            self.assertEqual(sorted(p.name.split("--")[0] for p in (d / "ports").iterdir()),
                             ["ports-QGZ", "ports-SYL", "ports-SYL"])
            # a new probe branch cut from main: no ports usage in its own tree
            empty = d / "tree" / "docs" / "data" / "sat" / "ports" / "SYL" / "usage.csv"
            now = datetime(2026, 10, 31, tzinfo=timezone.utc)
            g = pa.guard("2026-10", now, 4750, 9995, 50, 4600, sorted((d / "other").iterdir()),
                         sorted((d / "ports").iterdir()) + [empty])
            self.assertEqual(g["done_pu"], 100 + 25)       # SYL: largest copy (60 + 40), not 60 + 100; QGZ 25
            self.assertEqual(g["done_req"], 20 + 3)
            self.assertEqual(g["other_pu"], 50)            # ocean: largest copy (30 + 20)
            self.assertEqual(g["reserve"], 4550)           # 4,600 - 50
            self.assertEqual(g["cap"], min(4750 - 125, 9995 - 50 - 50 - 4550 - 125))
            # the working tree's copy is the same file as the branch copies (largest counted, not summed)
            empty.parent.mkdir(parents=True)
            empty.write_text(USAGE_HEAD + "2026-10-20T00:00:00Z,SYL,10,60\n2026-10-30T00:00:00Z,SYL,10,70\n")
            g2 = pa.guard("2026-10", now, 4750, 9995, 50, 4600, [], sorted((d / "ports").iterdir()) + [empty])
            self.assertEqual(g2["done_pu"], 130 + 25)
            with self.assertRaises(ValueError):
                pa.guard("2026-09", now, 4750, 9995, 50, 4600, [], [])

    def test_plan_counts_other_branches(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            d = Path(d)
            pa.gather_usage(d / "other", d / "ports", git=fake_git(self.trees()), log=lambda s: None)
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            for f in list((d / "ports").iterdir()) + list((d / "other").iterdir()):  # move the rows to this month
                f.write_text(f.read_text().replace("2026-10-", month + "-"))
            rc = pa.main(["plan", "--out", str(d / "p.json"), "--cap", "4750", "--global-cap", "9995",
                          "--global-margin", "50", "--ocean-cap", "4600", "--budget-month", month, "--only", "NONE",
                          "--usage", *map(str, sorted((d / "other").iterdir())),
                          "--ports-usage", *map(str, sorted((d / "ports").iterdir()))])
            self.assertEqual(rc, 0)
            res = json.loads((d / "p.json").read_text())
            self.assertEqual(res["ports_done_pu"], 125)
            self.assertEqual(res["cap_pu"], min(4750 - 125, 9995 - 50 - 50 - 4550 - 125))
            self.assertEqual(res["catalog_requests"], 0)

    def test_credit(self):
        self.assertEqual(pa.credit(["2026", "2026"]), "Contains modified Copernicus Sentinel data 2026")
        self.assertEqual(pa.credit([2026, 2025]), "Contains modified Copernicus Sentinel data 2025–2026")
        from sat import port_locate as pl
        self.assertEqual(pl.overview_credit(datetime(2026, 2, 1, tzinfo=timezone.utc)),
                         "Contains modified Copernicus Sentinel data 2025–2026")
        self.assertEqual(pl.overview_credit(datetime(2026, 10, 31, tzinfo=timezone.utc)),
                         "Contains modified Copernicus Sentinel data 2026")

    def test_locate_goes_through_the_guard(self):
        from sat import port_locate as pl
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            d = Path(d)
            regions = d / "regions.json"
            regions.write_text(json.dumps({"regions": [{"id": "R1", "bbox": [120.0, 30.0, 120.1, 30.1]},
                                                       {"id": "R2", "bbox": [121.0, 30.0, 121.1, 30.1]}]}))
            guard = ["--cap", "4750", "--global-cap", "9995", "--ocean-cap", "4600"]
            for drop in ("--cap", "--global-cap", "--ocean-cap"):
                i = guard.index(drop)
                with self.assertRaises(SystemExit):
                    pl.main(["--regions", str(regions), "--out", str(d / "o"), "--budget-month", month] + guard[:i] + guard[i + 2:])
            with mock.patch.object(pl.cdse, "Client", side_effect=AssertionError("no request")), \
                    mock.patch.object(pa, "LimitedClient", side_effect=AssertionError("no request")):
                self.assertEqual(pl.main(["--regions", str(regions), "--out", str(d / "o"), "--budget-month", "1999-01"] + guard), 2)
            # the ports cap is used up on another branch: no overview request, locate.json says 0 PU
            used = d / "ports-SYL--0.csv"
            used.write_text(USAGE_HEAD + f"{month}-01T00:00:00Z,SYL,10,4750\n")
            fake = Fake(Grid.for_site(SITE))
            with mock.patch.object(pl, "fetch", return_value={}), mock.patch.object(pl.time, "sleep"), \
                    mock.patch.object(pl.cdse.urllib.request, "urlopen", fake):
                rc = pl.main(["--regions", str(regions), "--out", str(d / "o"), "--budget-month", month,
                              "--ports-usage", str(used)] + guard)
            self.assertEqual(rc, 0)
            self.assertEqual(fake.process_calls, 0)
            r = json.loads((d / "o" / "locate.json").read_text())
            self.assertEqual((r["pu"], r["requests"]), (0, 0))
            self.assertEqual([x["id"] for x in r["regions"]], ["R1"])  # stopped after the first refusal


@unittest.skipUnless(np is not None, "numpy not installed")
class PlanTest(unittest.TestCase):
    def test_allocate_priority_and_scale(self):
        est = [{"site_id": "A", "pu": 1000, "priority": 1, "s1_requests": 100, "s2_requests": 100},
               {"site_id": "B", "pu": 2000, "priority": 1, "s1_requests": 100, "s2_requests": 100},
               {"site_id": "C", "pu": 500, "priority": 2, "s1_requests": 10, "s2_requests": 10}]
        out = pa.allocate(est, 3000, 10000)
        self.assertEqual([e["site_id"] for e in out], ["A", "B"])
        self.assertLessEqual(sum(e["cap_pu"] for e in out), 3000)
        self.assertIn("skipped", est[2])
        out = pa.allocate([dict(e) for e in est], 10000, 10000)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0]["cap_pu"], 1250)

    def test_month_usage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.csv"
            p.write_text("run_utc,requests,processing_units\n2026-10-01T00:00:00Z,5,10.5\n2026-09-30T00:00:00Z,9,99\n")
            self.assertEqual(pa.month_usage([p, Path(d) / "none.csv"], "2026-10"), (10.5, 5))

    def test_representative(self):
        rows = [{"acquired_utc": f"2026-0{i}-01", "relative_orbit": o, "result": "counted"}
                for i, o in enumerate([11, 11, 11, 40, 11, 40], start=1)]
        rows.append({"acquired_utc": "2026-09-01", "relative_orbit": 99, "result": "skipped_low_coverage"})
        picks = pa.representative(rows)
        self.assertEqual(len(picks), 3)
        self.assertEqual(picks[:2], ["2026-03-01", "2026-06-01"])
        self.assertEqual(pa.representative(rows[:1]), ["2026-01-01"])


@unittest.skipUnless(np is not None, "numpy not installed")
class LimitTest(unittest.TestCase):
    def client(self, fake, cap=100.0, req=100, pu_min=1.0, req_min=100):
        clock = Clock()
        c = pa.LimitedClient(cap, req, pu_min, req_min, clock=clock, sleep=clock.sleep, client_id="i",
                             client_secret="s", opener=fake)
        return c, clock

    def test_rate_window_waits(self):
        g = Grid.for_site(SITE)
        c, clock = self.client(Fake(g), pu_min=1.0)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for _ in range(3):
            c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertAlmostEqual(c.processing_units, 1.5)
        self.assertTrue(clock.slept)  # the third request waited for the 60 s window

    def test_cap_stops_before_request(self):
        g = Grid.for_site(SITE)
        fake = Fake(g)
        c, _ = self.client(fake, cap=0.012)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with self.assertRaises(pa.BudgetStop):
            c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(fake.process_calls, 0)

    def test_request_cap(self):
        g = Grid.for_site(SITE)
        c, _ = self.client(Fake(g), req=1)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        c.catalog(g.bbox, t, t)
        with self.assertRaises(pa.BudgetStop):
            c.catalog(g.bbox, t, t)

    def test_429_retried(self):
        g = Grid.for_site(SITE)
        fake = Fake(g, fail_429=5)
        c, clock = self.client(fake, pu_min=100)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _data, pu = c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(pu, 0.5)
        self.assertGreaterEqual(c.retries_429, 1)

    def test_internal_resends_in_the_window(self):
        g = Grid.for_site(SITE)
        fake = Fake(g, fail_429=2)  # cdse.Client resends twice by itself, then succeeds
        c, _clock = self.client(fake, pu_min=100)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(c.requests, 3)
        self.assertEqual(len(c.window), 3)  # the two resends count in the request rate (0 PU)
        self.assertAlmostEqual(sum(p for _t, p in c.window), 0.5)

    def test_month_change_stops_before_next_request(self):
        g = Grid.for_site(SITE)
        fake = Fake(g)
        clock = Clock()
        months = iter([datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc), datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)])
        c = pa.LimitedClient(100.0, 100, 100, 100, clock=clock, sleep=clock.sleep, client_id="i", client_secret="s",
                             opener=fake, budget_month="2026-10", utcnow=lambda: next(months))
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        c.process(g.bbox, g.width, g.height, t, t, True)
        with self.assertRaises(pa.BudgetStop) as cm:
            c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertIn("2026-11", str(cm.exception))
        self.assertEqual(fake.process_calls, 1)

    def test_one_request_over_the_rate_stops(self):
        g = Grid.for_site(SITE)
        fake = Fake(g)
        c, clock = self.client(fake, pu_min=0.001)  # one S1 chip is estimated above 0.001 PU
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with self.assertRaises(pa.BudgetStop):
            c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(fake.process_calls, 0)
        self.assertEqual(clock.slept, [])

    def test_deadline_checked_while_waiting(self):
        g = Grid.for_site(SITE)
        fake = Fake(g)
        c, clock = self.client(fake, pu_min=1.0)
        c.deadline = 30.0
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for _ in range(2):
            c.process(g.bbox, g.width, g.height, t, t, True)
        with self.assertRaises(pa.BudgetStop):  # the window would free at 60 s, after the deadline
            c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(fake.process_calls, 2)
        self.assertLessEqual(clock.t, 61)


@unittest.skipUnless(np is not None, "numpy not installed")
class RunPortTest(unittest.TestCase):
    def test_run_port_end_to_end(self):
        g = Grid.for_site(SITE)
        clock = Clock()
        c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                             client_secret="s", opener=Fake(g))
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            from sat.osm_mask import save_mask
            save_mask(SITE, g, mask, {}, {}, 1, d / "masks")
            res = pa.run_port(c, SITE, d / "out", d / "raw", d / "tmp", d / "masks", now, log=lambda s: None)
            self.assertEqual(res["s1"]["counted"], 4)
            self.assertEqual(res["s1"]["footprint_skipped"], 1)
            self.assertEqual(len(res["raw_kept"]), 3)
            rows = pa.read_csv(d / "out" / "TPA" / "s1_scenes.csv")
            self.assertEqual([r["result"] for r in rows].count("skipped_footprint"), 1)
            rows = [r for r in rows if r["result"] == "counted"]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(int(r["candidates"]) >= 1 for r in rows))
            self.assertEqual(sum(bool(r["raw_chip"]) for r in rows), 3)
            self.assertTrue((d / "out" / "TPA" / rows[0]["color_image"]).exists())
            fc = json.loads((d / "out" / "TPA" / "detections" / "2026-01.geojson").read_text())
            self.assertTrue(fc["features"])
            self.assertTrue((d / "raw" / "TPA" / "README.json").exists())
            self.assertEqual(fc["attribution"], "Contains modified Copernicus Sentinel data 2026")
            self.assertEqual(json.loads((d / "raw" / "TPA" / "README.json").read_text())["credit"],
                             "Contains modified Copernicus Sentinel data 2026")
            s2rows = pa.read_csv(d / "out" / "TPA" / "s2_scenes.csv")
            self.assertEqual(sorted(r["result"] for r in s2rows), ["clear", "clear"])
            self.assertEqual(res["s2"]["clear"], 2)
            self.assertAlmostEqual(res["processing_units"], 4 * 0.5 + 2 * 0.21, places=3)
            # a second run fetches nothing new for S1 (and S2 days already recorded)
            c2 = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                                  client_secret="s", opener=Fake(g))
            res2 = pa.run_port(c2, SITE, d / "out", d / "raw", d / "tmp", d / "masks", now, log=lambda s: None)
            self.assertEqual(res2["s1"]["counted"], 0)
            self.assertEqual(res2["processing_units"], 0)
            self.assertIn("26-01:", pa.trend(d / "out" / "TPA"))
            # a new mask version re-fetches the 12 months; the v1 rows, detections and raw chips stay
            site2 = {**SITE, "mask_version": 2}
            save_mask(site2, g, mask, {}, {}, 2, d / "masks")
            c3 = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                                  client_secret="s", opener=Fake(g))
            res3 = pa.run_port(c3, site2, d / "out", d / "raw", d / "tmp2", d / "masks", now, log=lambda s: None)
            self.assertEqual(res3["s1"]["counted"], 4)
            self.assertEqual(res3["s2"]["clear"], 2)
            rows = pa.read_csv(d / "out" / "TPA" / "s1_scenes.csv")
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(r["raw_chip"].startswith("probe-raw/TPA/v2/") for r in rows), 3)
            self.assertEqual(sum(r["raw_chip"].startswith("probe-raw/TPA/2") for r in rows), 3)
            fc = json.loads((d / "out" / "TPA" / "detections" / "2026-01.geojson").read_text())
            self.assertEqual({f["properties"]["mask_version"] for f in fc["features"]}, {"TPA-land-v1", "TPA-land-v2"})
            self.assertTrue((d / "out" / "TPA" / "color" / "v2").is_dir())

    def test_plan_counts_only_new_work(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            from sat.osm_mask import save_mask
            save_mask(SITE, g, mask, {}, {}, 1, d / "masks")
            c = pa.cdse.Client("i", "s", opener=Fake(g))
            p = pa.plan(c, [SITE], now, 100, 1000, log=lambda s: None, out_root=d / "out", mask_dir=d / "masks")
            self.assertEqual((p["ports"][0]["s1_scenes"], p["ports"][0]["s1_footprint_skip"], p["ports"][0]["s2_days"]), (4, 1, 2))
            clock = Clock()
            lc = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                                  client_secret="s", opener=Fake(g))
            pa.run_port(lc, SITE, d / "out", d / "raw", d / "tmp", d / "masks", now, log=lambda s: None)
            p = pa.plan(c, [SITE], now, 100, 1000, log=lambda s: None, out_root=d / "out", mask_dir=d / "masks")
            self.assertEqual((p["ports"][0]["s1_scenes"], p["ports"][0]["s2_days"]), (0, 0))
            self.assertEqual(p["matrix"], [])

    def test_backfill_window_s1_only_and_order(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2027, 3, 1, tzinfo=timezone.utc)  # the fake scenes (2026-01..07) are 8-14 months back
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            from sat.osm_mask import save_mask
            save_mask(SITE, g, mask, {}, {}, 1, d / "masks")
            c = pa.cdse.Client("i", "s", opener=Fake(g))
            p = pa.plan(c, [SITE], now, 100, 1000, log=lambda s: None, out_root=d / "out", mask_dir=d / "masks",
                        days_from=730, days_to=365, s2_on=False, ordered=True)
            self.assertEqual(p["ports"][0]["s2_days"], 0)
            self.assertEqual(p["matrix"][0]["port"], "TPA")
            clock = Clock()
            lc = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                                  client_secret="s", opener=Fake(g))
            res = pa.run_port(lc, SITE, d / "out", d / "raw", d / "tmp", d / "masks", now, log=lambda s: None,
                              days_from=730, days_to=365, s2_on=False)
            self.assertEqual(res["s1"]["counted"], 4)
            self.assertEqual(res["s2"]["stopped"], "S2 なし（S1 だけの取得）")
            self.assertFalse((d / "out" / "TPA" / "s2_scenes.csv").exists())

    def test_allocate_ordered(self):
        est = [{"site_id": k, "pu": pu, "priority": 1, "s1_requests": 10, "s2_requests": 0}
               for k, pu in (("A", 400), ("B", 400), ("C", 400))]
        out = pa.allocate_ordered(est, 1000, 10000)
        self.assertEqual([(e["site_id"], e["cap_pu"]) for e in out], [("A", 500), ("B", 500)])
        self.assertIn("skipped", est[2])

    def test_baseline_port(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        clock = Clock()
        fake = Fake(g)
        c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                             client_secret="s", opener=fake)
        with tempfile.TemporaryDirectory() as d:
            s = pa.baseline_port(c, SITE, mask, Path(d), now, log=lambda t: None)
            self.assertEqual(s["footprint_dropped"], 1)  # the swath-edge scene is not fetched
            self.assertEqual(list(s["orbits"]), ["11"])   # orbit 40 has one scene only
            self.assertEqual(fake.process_calls, 3)
            self.assertFalse(s["orbits"]["11"]["enough"])
            self.assertTrue((Path(d) / "TPA-land-v1" / "ro11-bl-1.json").exists())
            self.assertTrue((Path(d) / "TPA-land-v1" / "ro11-bl-1.vv_p10.png").exists())

    def test_baseline_fill_orbits_and_period(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2027, 3, 1, tzinfo=timezone.utc)  # the fake scenes are 8-14 months back
        clock = Clock()
        fake = Fake(g)
        c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                             client_secret="s", opener=fake)
        with tempfile.TemporaryDirectory() as d:
            s = pa.baseline_port(c, SITE, mask, Path(d), now, log=lambda t: None, lookback=730, max_scenes=3,
                                 only_orbits=[11])
            self.assertEqual(fake.process_calls, 3)  # orbit 11 only (orbit 40 left out)
            meta = json.loads((Path(d) / "TPA-land-v1" / "ro11-bl-1.json").read_text())
            self.assertEqual(meta["period"]["lookback_days"], 730)
            self.assertEqual(meta["period"]["older_than_12_months"], 1)  # 2026-01 < 2026-03-01

    def test_baseline_orbit_max_and_top(self):
        g = Grid.for_site(SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        for kw, calls in (({"only_orbits": [11], "orbit_max": {11: 3}, "max_scenes": 2}, 3),
                          ({"top": 1, "max_scenes": 24}, 3)):
            clock = Clock()
            fake = Fake(g)
            c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                                 client_secret="s", opener=fake)
            with tempfile.TemporaryDirectory() as d:
                s = pa.baseline_port(c, SITE, mask, Path(d), now, log=lambda t: None, **kw)
                self.assertEqual(list(s["orbits"]), ["11"])
                self.assertEqual(fake.process_calls, calls)

    def test_footprint_share(self):
        g = Grid.for_site(SITE)
        water = np.ones((g.height, g.width), bool)
        half = {"type": "Polygon", "coordinates": [[[119, 29], [120.0025, 29], [120.0025, 31], [119, 31], [119, 29]]]}
        self.assertAlmostEqual(pa.footprint_share([half], g, water, step=1), 0.5, delta=0.05)
        self.assertIsNone(pa.footprint_share([], g, water))

    def test_cap_stops_s1_and_skips_s2(self):
        g = Grid.for_site(SITE)
        clock = Clock()
        c = pa.LimitedClient(1.2, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i",
                             client_secret="s", opener=Fake(g))
        now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            res = pa.run_port(c, SITE, d / "out", d / "raw", d / "tmp", d / "masks", now, log=lambda s: None)
            self.assertLessEqual(res["processing_units"], 1.2)
            self.assertTrue(res["s1"]["stopped"])
            self.assertEqual(res["s1"]["counted"], 2)  # no mask: images only (the edge scene is skipped on its footprint)
            self.assertEqual(pa.read_csv(d / "out" / "TPA" / "s1_scenes.csv")[0]["candidates"], "")


if __name__ == "__main__":
    unittest.main()
