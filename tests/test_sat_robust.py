"""Robustness of the daily run and the port archive (review of 2026-11-18): est-2 per mask version, PU records
that survive a failure, network errors retried and contained per site, atomic writes, the PU of failed scenes,
the quay refetch and est-1's prior on a refetched scene. No request leaves the machine (every client is a fake)."""
import contextlib
import http.client
import io
import json
import os
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

if np is None:  # the classes below subclass the numpy test fakes; skip the module where they cannot be imported
    raise unittest.SkipTest('numpy is not installed')
else:
    from sat import atomic, cdse, png, quay_est, run, s2
    from sat import port_archive as pa
    from sat.grid import Grid
    from tests.test_sat import SITE, FakeClient, FakeResp, RunTest, token_ok
    from tests.test_sat_port_archive import SITE as PA_SITE, Clock, Fake as PAFake
    from tests.test_sat_s2 import SITE as S2_SITE, FakeS2

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


class Later(datetime):
    """A later run (another month) for the retrieved_utc of the fetches."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2099, 1, 1, tzinfo=timezone.utc)
quiet = lambda *a, **k: None  # noqa: E731


def chip_and_land():
    return RunTest().chip()


class GarbageClient(FakeClient):
    """The process request succeeds (PU spent) but the chip cannot be read."""

    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processing_units += 4.883
        self.processed.append(t0)
        return b"not a png", 4.883


class NetDownClient(FakeClient):
    def process(self, bbox, w, h, t0, t1, with_lia=True):
        self.requests += 1
        self.processed.append(t0)
        raise cdse.NetworkError("CDSE（sh.dataspace.copernicus.eu）に接続できない（TimeoutError: timed out）")


# ---------- 3. network errors: CDSEError, a few retries, contained per site ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class NetworkRetryTest(unittest.TestCase):
    def client(self, opener):
        c = cdse.Client("id", "secret", opener=opener)
        c.waits = []
        c._wait = c.waits.append  # no real sleep
        return c

    def test_network_errors_become_cdse_errors_after_retries(self):
        t = datetime(2026, 10, 1, tzinfo=timezone.utc)
        for exc in (urllib.error.URLError("timed out"), TimeoutError("The read operation timed out"),
                    ConnectionResetError(104, "Connection reset by peer"),
                    http.client.RemoteDisconnected("Remote end closed connection without response"),
                    http.client.IncompleteRead(b"x")):
            calls = []

            def opener(req, timeout=None, exc=exc):
                calls.append(req.full_url)
                if req.full_url == cdse.TOKEN_URL:
                    return FakeResp(200, json.dumps({"access_token": "tok", "expires_in": 600}).encode(), {})
                raise exc
            c = self.client(opener)
            with self.assertRaises(cdse.NetworkError) as cm:
                c.catalog([0, 0, 1, 1], t, t)
            self.assertIsInstance(cm.exception, cdse.CDSEError)
            self.assertNotIsInstance(cm.exception, cdse.CredentialError)
            self.assertNotIn("secret", str(cm.exception))
            self.assertEqual(calls.count(cdse.CATALOG_URL), cdse.NET_ATTEMPTS, type(exc).__name__)
            self.assertEqual(c.requests, cdse.NET_ATTEMPTS)  # every send counts (request cap, 300 / min)
            self.assertEqual(c.waits, [cdse.backoff(i) for i in range(cdse.NET_ATTEMPTS - 1)])  # 5, 10, 20 s

    def test_backoff_keeps_far_below_the_request_rate(self):
        waits = [cdse.backoff(i) for i in range(cdse.NET_ATTEMPTS - 1)]
        self.assertEqual(waits, sorted(waits))
        self.assertGreaterEqual(min(waits), 1.0)  # at most one resend per second, 300 req / min never reached
        self.assertLessEqual(cdse.NET_ATTEMPTS, 4)

    def test_one_network_error_then_success(self):
        img = png.encode(np.ones((4, 4, 3), np.uint16) * 30000, depth=16)
        n = {"process": 0}

        def opener(req, timeout=None):
            if req.full_url == cdse.TOKEN_URL:
                return FakeResp(200, json.dumps({"access_token": "tok", "expires_in": 600}).encode(), {})
            n["process"] += 1
            if n["process"] == 1:
                raise ConnectionResetError(104, "reset")
            return FakeResp(200, img, {"x-processingunits-spent": "1.5"})
        c = self.client(opener)
        t = datetime(2026, 10, 1, tzinfo=timezone.utc)
        _data, pu = c.process([0, 0, 1, 1], 4, 4, t, t, with_lia=False)
        self.assertEqual(pu, 1.5)
        self.assertEqual((c.requests, c.network_retries, len(c.waits)), (2, 1, 1))

    def test_token_retries_429_5xx_and_network(self):
        seq = [lambda: (_ for _ in ()).throw(urllib.error.URLError("dns")), (503, b"busy", {}), (429, b"slow", {"Retry-After": "45"}),
               (200, json.dumps({"access_token": "tok", "expires_in": 600}).encode(), {})]
        calls = []

        def opener(req, timeout=None):
            item = seq[len(calls)]
            calls.append(req.full_url)
            if callable(item):
                item()
            status, body, headers = item
            if status != 200:
                raise urllib.error.HTTPError(req.full_url, status, "err", headers, io.BytesIO(body))
            return FakeResp(status, body, headers)
        c = self.client(opener)
        self.assertEqual(c.token(), "tok")
        self.assertEqual(len(calls), 4)
        self.assertEqual(c.waits, [cdse.backoff(0), cdse.backoff(1), 45.0])  # Retry-After honoured (when longer)

    def test_token_5xx_gives_up_with_cdse_error(self):
        def opener(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 502, "bad gateway", {}, io.BytesIO(b"x"))
        c = self.client(opener)
        with self.assertRaises(cdse.CDSEError) as cm:
            c.token()
        self.assertNotIsInstance(cm.exception, cdse.CredentialError)
        self.assertEqual(len(c.waits), cdse.NET_ATTEMPTS - 1)

    def test_invalid_client_is_not_retried(self):
        calls = []

        def opener(req, timeout=None):
            calls.append(1)
            raise urllib.error.HTTPError(req.full_url, 401, "err", {}, io.BytesIO(b'{"error":"invalid_client"}'))
        c = self.client(opener)
        with self.assertRaises(cdse.CredentialError):
            c.token()
        self.assertEqual(len(calls), 1)

    def test_limited_client_waits_on_its_own_clock(self):
        g = Grid.for_site(PA_SITE)
        clock = Clock()
        fake = PAFake(g)
        n = {"k": 0}

        def opener(req, timeout=None):
            if req.full_url == cdse.PROCESS_URL:
                n["k"] += 1
                if n["k"] == 1:
                    raise TimeoutError("timed out")
            return fake(req, timeout)
        c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i", client_secret="s",
                             opener=opener)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _d, pu = c.process(g.bbox, g.width, g.height, t, t, True)
        self.assertEqual(pu, 0.5)
        self.assertIn(cdse.backoff(0), clock.slept)
        self.assertEqual(c.requests, 2)
        self.assertEqual(len(c.window), 2)  # the resend is in the 60 s window

    def run_main(self, out, cli, sites, land):
        with mock.patch.object(run, "load_mask", return_value=(land.astype(np.uint8), {})), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.main(["--out", str(out)], client=cli, sites=sites)

    def test_one_site_timeout_does_not_stop_the_others_or_s2(self):
        chip, land = chip_and_land()
        bad = {**SITE, "id": "TBA", "layout": "per_site", "pu_month_cap": 100}
        good = {**SITE, "id": "TGO", "layout": "per_site", "pu_month_cap": 100}
        s2_called = []

        class Cli(FakeClient):
            calls = 0

            def catalog(self, bbox, start, end, collection=None):
                Cli.calls += 1
                if Cli.calls == 1:  # the first site's catalog: the network is gone (after the client's retries)
                    self.requests += cdse.NET_ATTEMPTS
                    raise cdse.NetworkError("CDSE（sh.dataspace.copernicus.eu）に接続できない（TimeoutError: timed out）")
                return super().catalog(bbox, start, end)

        def fake_s2(client, out, site, now, days, budget, log=print):
            s2_called.append(site["id"])
            return {"site_id": site["id"], "days": 0}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run.s2, "update_site", side_effect=fake_s2):
            out = Path(d)
            sites = [{**bad, "s2": {"enabled": True}}, {**good, "s2": {"enabled": True}}]
            rc = self.run_main(out, Cli(chip), sites, land)
            self.assertEqual(rc, 1)
            st = json.loads((out / "status.json").read_text())
            by = {s["site_id"]: s for s in st["last_run"]["sites"]}
            self.assertIn("TimeoutError", by["TBA"]["error"])
            self.assertEqual(by["TGO"]["counted"], 2)
            self.assertEqual(s2_called, ["TBA", "TGO"])
            usage = run.read_csv(out / "usage.csv")
            self.assertEqual(int(usage[0]["requests"]), cdse.NET_ATTEMPTS + 1 + 2)

    def test_bare_oserror_from_a_site_is_contained(self):
        chip, land = chip_and_land()

        class Cli(FakeClient):
            def catalog(self, bbox, start, end, collection=None):
                raise ConnectionResetError(104, "reset")  # a client that does not wrap its errors
        with tempfile.TemporaryDirectory() as d:
            rc = self.run_main(Path(d), Cli(chip), [SITE], land)
            self.assertEqual(rc, 1)
            st = json.loads((Path(d) / "status.json").read_text())
            self.assertIn("reset", st["last_run"]["sites"][0]["error"])

    def test_network_down_defers_the_rest_of_the_site(self):
        chip, land = chip_and_land()

        class Cli(NetDownClient):
            def catalog(self, bbox, start, end, collection=None):
                feats = super().catalog(bbox, start, end)
                p = feats[0]["properties"]
                return feats + [{"id": "c", "properties": {**p, "datetime": "2026-10-28T22:30:00Z"}}]
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d), [SITE])
            cli = Cli(chip)
            s = run.process_site(cli, store, SITE, land.astype(np.uint8), NOW, None, 100.0, 12, log=quiet)
            self.assertEqual(len(cli.processed), run.NET_FAILURES_PER_SITE)  # the third scene is not tried
            self.assertEqual((s["errors"], s["deferred_network"]), (2, 1))


# ---------- 5. the PU of a failed scene stays in its record ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class FailedScenePUTest(unittest.TestCase):
    def test_s1_decode_failure_keeps_pu_and_retry_keeps_both(self):
        chip, land = chip_and_land()
        site = {**SITE, "pu_month_cap": 100}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            store = run.Store(out, [site])
            run.process_site(GarbageClient(chip), store, site, land.astype(np.uint8), NOW, None, 100.0, 12, log=quiet)
            sc = store.status["scenes"]
            self.assertEqual([s["result"] for s in sc], ["error", "error"])
            self.assertEqual([s["processing_units"] for s in sc], [4.883, 4.883])
            month = sc[0]["retrieved_utc"][:7]
            self.assertAlmostEqual(run.site_month_pu(store, site["id"], month), 2 * 4.883, places=3)
            self.assertAlmostEqual(run.site_budget(store, site, month, 1000.0), 100 - 2 * 4.883, places=3)
            # the next run counts them; the PU of the failed fetches stay (earlier_fetches)
            run.process_site(FakeClient(chip), store, site, land.astype(np.uint8), NOW, None, 100.0, 12, log=quiet)
            sc = store.status["scenes"]
            self.assertEqual([s["result"] for s in sc], ["counted", "counted"])
            self.assertEqual([len(s["earlier_fetches"]) for s in sc], [1, 1])
            self.assertAlmostEqual(run.site_month_pu(store, site["id"], month), 4 * 4.883, places=3)

    def test_earlier_fetches_count_in_their_own_month(self):
        store = run.Store(Path(tempfile.mkdtemp()), [SITE])
        store.status["scenes"] = [{"site_id": "TST", "acquired_utc": "2026-10-31T22:30:00Z", "retrieved_utc": "2026-11-01T05:00:00Z",
                                   "processing_units": 2.0, "result": "counted",
                                   "earlier_fetches": [{"retrieved_utc": "2026-10-31T23:00:00Z", "processing_units": 3.0}]}]
        self.assertEqual(run.site_month_pu(store, "TST", "2026-10"), 3.0)
        self.assertEqual(run.site_month_pu(store, "TST", "2026-11"), 2.0)

    def test_s2_unreadable_answer_keeps_its_pu(self):
        class Broken(FakeS2):
            def process_raw(self, payload, accept, input_bands):
                data, pu = super().process_raw(payload, accept, input_bands)
                return (b"not a tar" if accept != "image/png" else data), pu
        cli = Broken({"2026-10-21": (5.0, 0.0)})
        with tempfile.TemporaryDirectory() as d:
            r = s2.update_site(cli, Path(d), S2_SITE, datetime(2026, 10, 30, 6, tzinfo=timezone.utc), 10, log=quiet)
            rows = s2.read_rows(Path(d) / "s2_scenes.csv")
            self.assertEqual(rows[0]["result"], "error")
            self.assertAlmostEqual(float(rows[0]["processing_units"]), 0.03 + 1.53, places=3)  # the 10 m answer too
            self.assertAlmostEqual(r["processing_units"], 1.56, places=3)

    def test_s2_rows_written_when_the_run_stops_midway(self):
        class Stops(FakeS2):
            def process_raw(self, payload, accept, input_bands):
                date = payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10]
                if date == "2026-10-23":
                    raise cdse.CredentialError("Sentinel Hub がアクセスを拒否（HTTP 403）")
                return super().process_raw(payload, accept, input_bands)
        cli = Stops({"2026-10-21": (5.0, 0.0), "2026-10-23": (5.0, 0.0)})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(cdse.CredentialError):
                s2.update_site(cli, Path(d), S2_SITE, datetime(2026, 10, 30, 6, tzinfo=timezone.utc), 10, log=quiet)
            rows = s2.read_rows(Path(d) / "s2_scenes.csv")
            self.assertEqual([(r["date"], r["result"]) for r in rows], [("2026-10-21", "clear")])


# ---------- 6. a failed quay refetch keeps the scene's record, and is tried a limited number of times ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class QuayRefetchTest(unittest.TestCase):
    def test_failed_refetch_keeps_the_record_and_gives_up(self):
        chip, land = chip_and_land()
        mask = land.astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            store = run.Store(out, [SITE])
            run.process_site(FakeClient(chip), store, SITE, mask, NOW, None, 100.0, 12, log=quiet)
            first = json.loads(json.dumps(store.status["scenes"]))
            self.assertEqual([s["result"] for s in first], ["counted", "counted"])
            month = first[0]["retrieved_utc"][:7]
            # a baseline of the orbit appears: the counted scenes are fetched again for their quay count, and fail
            with mock.patch.object(run.QuayContext, "baseline", lambda self, o: ({}, {})):
                for attempt in range(1, run.QUAY_REFETCH_MAX + 1):
                    cli = GarbageClient(chip)
                    s = run.process_site(cli, store, SITE, mask, NOW, None, 100.0, 12, log=quiet)
                    self.assertEqual(len(cli.processed), 2)
                    self.assertEqual(s["errors"], 2)
                    for old, now_ in zip(first, store.status["scenes"]):
                        for k in ("result", "processing_units", "coverage", "detector_info", "retrieved_utc"):
                            self.assertEqual(now_[k], old[k], k)
                        self.assertEqual(now_["quay_refetch_failures"], attempt)
                        self.assertEqual(len(now_["earlier_fetches"]), attempt)
                    self.assertAlmostEqual(run.site_month_pu(store, "TST", month), (2 + 2 * attempt) * 4.883, places=3)
                self.assertEqual({s["quay_result"] for s in store.status["scenes"]}, {"error"})
                # given up: never fetched again
                cli = FakeClient(chip)
                s = run.process_site(cli, store, SITE, mask, NOW, None, 100.0, 12, log=quiet)
                self.assertEqual((s["new"], cli.processed), (0, []))
            self.assertEqual(len(store.rows), 2)  # the ship rows of the first count stay

    def test_successful_refetch_keeps_the_first_pu(self):
        chip, land = chip_and_land()
        mask = land.astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d), [SITE])
            run.process_site(FakeClient(chip), store, SITE, mask, NOW, None, 100.0, 12, log=quiet)
            month = store.status["scenes"][0]["retrieved_utc"][:7]
            with mock.patch.object(run.QuayContext, "baseline", lambda self, o: ({}, {})), \
                    mock.patch.object(run, "count_quay", lambda store, q, site, acq, *a, **k: a[-2].update(quay_result="counted") or ""), \
                    mock.patch.object(run, "datetime", Later):
                run.process_site(FakeClient(chip), store, SITE, mask, NOW, None, 100.0, 12, log=quiet)
            self.assertEqual({s["quay_result"] for s in store.status["scenes"]}, {"counted"})
            self.assertAlmostEqual(run.site_month_pu(store, "TST", month), 2 * 4.883, places=3)  # the first fetches
            self.assertAlmostEqual(run.site_month_pu(store, "TST", "2099-01"), 2 * 4.883, places=3)


# ---------- 7. est-1 of a refetched scene: never its own prior ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class OwnPriorTest(unittest.TestCase):
    def prior(self):
        model = quay_est.Model()
        zp = quay_est.ZonePrior(model, "ZJG")
        zid = next(iter(zp.state["zones"]))
        return model, zp, zid

    def test_without_gives_the_prior_before_the_scene(self):
        model, zp, zid = self.prior()
        t = "2026-11-01T10:48:45Z"
        self.assertIs(zp.without(t), zp)
        zp.record(t, True)
        before = list(zp.alpha(zid))
        h = zp.digest()
        zp.update(t, {zid: [np.array([0, 0, 5.0, 0, 0, 0])]})
        self.assertNotEqual(zp.alpha(zid), before)
        p = zp.without(t)
        self.assertIsNot(p, zp)
        self.assertEqual(p.alpha(zid), before)
        self.assertEqual(p.digest(), h)
        self.assertNotEqual(zp.alpha(zid), before)  # the prior itself is unchanged
        # no history entry left: alpha0 (never the scene's own update)
        zp.state["history"] = []
        self.assertEqual(zp.without(t).alpha(zid), zp.state["zones"][zid]["alpha0"])

    def test_count_quay_estimates_a_refetched_scene_with_its_first_prior(self):
        model, _zp, zid = self.prior()
        site = {"id": "ZJG", "enabled": True, "center_lat": 21.2}
        acq = {"acquired_utc": "2026-11-01T10:48:45Z", "platform": "sentinel-1c", "orbit_direction": "ascending",
               "relative_orbit": 157}
        seg = {"way_id": 1, "sector": "A", "length_m": 120.0, "min": 1, "max": 1, "side": 1, "line": [[0, 0], [1, 1]],
               "kind": "quay", "lat": 21.2, "lon": 110.4}
        q = {"min": 1, "max": 1, "segments": [seg], "by_sector": {}, "rafted": 0, "coverage": 1.0, "excluded": {}}
        qctx = mock.Mock()
        qctx.lines, qctx.lines_version = {"ways": []}, 1
        qctx.baseline.return_value = ({"vv_p10": np.zeros((2, 2))}, {"relative_orbit": 157, "version": "b", "scenes": 30})
        qctx.berths.return_value.naval = False
        qctx.geometry.return_value.cfg = {"cluster_m": 1}
        qctx.model.return_value = model
        priors = []
        real = quay_est.estimate

        def spy(segments, site_id, model=None, prior=None, *a, **k):
            priors.append(prior.digest())
            return real([dict(s) for s in segments], site_id, model, prior, *a, **k)
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d), [site])
            with mock.patch.object(run.quay, "count", return_value=q), \
                    mock.patch.object(run.quay, "restrict", side_effect=lambda q, ok, m: {**q, "segments": [dict(s) for s in q["segments"]]}), \
                    mock.patch.object(run, "quay_features", return_value=[]), \
                    mock.patch.object(run.fcolor, "render_change", side_effect=ValueError("no image")), \
                    mock.patch.object(run.quay_est, "estimate", side_effect=spy):
                for _ in range(2):  # the first count, then the same scene fetched again
                    entry = {}
                    run.count_quay(store, qctx, site, acq, None, None, [], entry, log=quiet)
                    self.assertEqual(entry["quay_result"], "counted")
            zp = store.zone_priors["ZJG"]
            self.assertIn(acq["acquired_utc"], zp.state["applied"])
            self.assertEqual(priors[0], priors[1])  # the refetch used the prior from before its own update
            h = [x for x in zp.state["history"] if x["acquired_utc"] == acq["acquired_utc"]]
            self.assertEqual([x["prior_hash"] for x in h], [priors[0]])
            self.assertEqual(store.quay_rows[0]["est_prior_hash"], priors[0])


# ---------- 1. est-2 of the sector rows per mask version ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class SectorEst2Test(unittest.TestCase):
    def test_v1_and_v2_rows_of_one_scene_keep_their_own_est2(self):
        t = "2026-11-02T09:54:27Z"
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d), [SITE])
            store.sector_rows = [{"site_id": "TST", "acquired_utc": t, "mask_version": v, "sector_id": "A", "relative_orbit": 1}
                                 for v in ("TST-land-v1", "TST-land-v2")]
            st = lambda m: {"A": {"warship": {"median": m, "p10": m - 1, "p90": m + 1}}}  # noqa: E731
            est2 = {"scenes": [{"site_id": "TST", "acquired_utc": t, "mask_version": "TST-land-v1", "by_sector": st(3)},
                               {"site_id": "TST", "acquired_utc": t, "mask_version": "TST-land-v2", "by_sector": st(7)}]}
            store.save_sectors(est2)
            rows = {r["mask_version"]: r for r in run.read_csv(Path(d) / "quay_sectors.csv")}
            self.assertEqual((rows["TST-land-v1"]["est2_median"], rows["TST-land-v2"]["est2_median"]), ("3", "7"))
            self.assertEqual((rows["TST-land-v1"]["est2_p90"], rows["TST-land-v2"]["est2_p90"]), ("4", "8"))

    def test_rows_without_a_mask_version_are_v1(self):
        t = "2026-10-02T09:54:27Z"
        with tempfile.TemporaryDirectory() as d:
            store = run.Store(Path(d), [SITE])
            store.sector_rows = [{"site_id": "TST", "acquired_utc": t, "mask_version": "TST-land-v1", "sector_id": "A"}]
            est2 = {"scenes": [{"site_id": "TST", "acquired_utc": t, "mask_version": "TST-land-v2",
                                "by_sector": {"A": {"warship": {"median": 9, "p10": 8, "p90": 10}}}}]}
            store.save_sectors(est2)
            self.assertFalse((Path(d) / "quay_sectors.csv").exists())  # the v2 value never lands on the v1 row

    def test_build_gives_each_scene_its_mask_version(self):
        from sat import quay_est2
        rows = [{"site_id": "TST", "acquired_utc": "2026-11-02T09:54:27Z", "mask_version": m, "relative_orbit": "1"}
                for m in ("", "TST-land-v2")]
        with tempfile.TemporaryDirectory() as d:
            res = quay_est2.build(Path(d), rows, [], {}, sites=[{**SITE, "id": "TST", "layout": "per_site"}])
            self.assertEqual([x["mask_version"] for x in res["scenes"]], ["TST-land-v1", "TST-land-v2"])
            doc = json.loads((Path(d) / "quay_est2" / "TST.json").read_text())
            self.assertNotIn("mask_version", doc["scenes"][0])  # the published files keep their keys


# ---------- 4. atomic writes; usage.csv and status.json however the save ends ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class AtomicWriteTest(unittest.TestCase):
    def test_failed_write_leaves_the_old_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            atomic.write_csv(p, [{"a": 1}], ["a"])
            os.chmod(p, 0o644)
            with mock.patch.object(atomic.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic.write_csv(p, [{"a": 2}], ["a"])
            self.assertEqual(p.read_text(), "a\n1\n")
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ["x.csv"])  # no temporary file left
            atomic.write_csv(p, [{"a": 3, "b": None}], ["a", "b"])
            self.assertEqual(p.read_text(), "a,b\n3,\n")
            self.assertEqual(p.stat().st_mode & 0o777, 0o644)

    def test_same_bytes_as_before(self):
        rows = [{"a": "x,y", "b": None, "c": 1.5, "extra": 1}, {"a": "日本", "b": 'q"t'}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.csv"
            run.write_csv(p, rows, ["a", "b", "c"])
            buf = io.StringIO()
            import csv
            w = csv.DictWriter(buf, fieldnames=["a", "b", "c"], lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ["a", "b", "c"]})
            self.assertEqual(p.read_bytes(), buf.getvalue().encode("utf-8"))

    def test_pu_and_status_survive_a_failed_save(self):
        chip, land = chip_and_land()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            run.write_csv(out / "ship_counts.csv", [], run.CSV_FIELDS)
            before = (out / "ship_counts.csv").read_bytes()
            with mock.patch.object(run, "load_mask", return_value=(land.astype(np.uint8), {})), \
                    mock.patch.object(run.Store, "save_records", side_effect=OSError("disk full")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(OSError):
                    run.main(["--out", str(out)], client=FakeClient(chip), sites=[SITE])
            usage = run.read_csv(out / "usage.csv")
            self.assertAlmostEqual(float(usage[0]["processing_units"]), 2 * 4.883, places=3)
            st = json.loads((out / "status.json").read_text())
            self.assertIn("disk full", st["last_run"]["save_error"])
            self.assertEqual([s["processing_units"] for s in st["scenes"]], [4.883, 4.883])  # site_month_pu sees them
            self.assertEqual((out / "ship_counts.csv").read_bytes(), before)  # old file, not a half-written one


# ---------- 2. the port archive records its PU and scenes however the run ends ----------

@unittest.skipUnless(np is not None, "numpy not installed")
class PortArchivePUTest(unittest.TestCase):
    def setup(self, d, fail_on: int):
        g = Grid.for_site(PA_SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        from sat.osm_mask import save_mask
        save_mask(PA_SITE, g, mask, {}, {}, 1, d / "masks")
        (d / "ports.json").write_text(json.dumps({"sites": [PA_SITE]}), encoding="utf-8")
        fake = PAFake(g)
        n = {"k": 0}

        def opener(req, timeout=None):
            if req.full_url == cdse.PROCESS_URL:
                n["k"] += 1
                if n["k"] == fail_on:
                    raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, io.BytesIO(b"quota"))
            return fake(req, timeout)
        return opener

    def test_403_midway_keeps_scenes_and_usage(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            opener = self.setup(d, fail_on=3)
            clock = Clock()
            real = pa.LimitedClient
            make = lambda *a, **k: real(*a, **{**k, "client_id": "i", "client_secret": "s", "opener": opener,  # noqa: E731
                                               "clock": clock, "sleep": clock.sleep})
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            with mock.patch.object(pa, "LimitedClient", side_effect=make), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(cdse.CredentialError):
                    pa.main(["run", "--ports", str(d / "ports.json"), "--port", PA_SITE["id"], "--cap", "50",
                             "--budget-month", month, "--out", str(d / "out"), "--raw-dir", str(d / "raw"),
                             "--tmp", str(d / "tmp"), "--mask-dir", str(d / "masks")])
            rows = pa.read_csv(d / "out" / "TPA" / "s1_scenes.csv")
            self.assertEqual([r["result"] for r in rows if r["result"] == "counted"], ["counted", "counted"])
            self.assertTrue(list((d / "out" / "TPA" / "detections").glob("*.geojson")))
            usage = pa.read_csv(d / "out" / "TPA" / "usage.csv")
            self.assertEqual(len(usage), 1)
            self.assertAlmostEqual(float(usage[0]["processing_units"]), 2 * 0.5, places=3)
            self.assertEqual(usage[0]["site_id"], "TPA")

    def test_run_s1_saves_on_network_error_raised_through(self):
        g = Grid.for_site(PA_SITE)
        mask = np.zeros((g.height, g.width), np.uint8)
        mask[:, : g.width // 4] = 1
        clock = Clock()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            opener = self.setup(d, fail_on=99)
            c = pa.LimitedClient(50.0, 500, 1000, 1000, clock=clock, sleep=clock.sleep, client_id="i", client_secret="s",
                                 opener=opener)
            calls = {"n": 0}
            real = c.process

            def process(*a, **k):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise KeyboardInterrupt  # anything that escapes the per-scene handling
                return real(*a, **k)
            c.process = process
            with self.assertRaises(KeyboardInterrupt):
                pa.run_s1(c, PA_SITE, d / "o", d / "t", mask, NOW, log=quiet)
            rows = pa.read_csv(d / "o" / "s1_scenes.csv")
            self.assertEqual([r["result"] for r in rows], ["counted"])


if __name__ == "__main__":
    unittest.main()
