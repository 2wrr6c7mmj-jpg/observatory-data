"""ocean-verify-1 (the 20 m second look with the earlier passes), the wind-farm lattice and the cross-orbit learning
(ocean-monitor-3). Synthetic arrays only: no network."""
from __future__ import annotations

import math
import time
import unittest
from argparse import Namespace
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from ocean import monitor as m
    from ocean import verify as v
    from sat import cdse, png
    from sat.grid import m_per_deg

NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


def code(db):
    """dB -> the UINT16 code of sat.cdse (0 = no data)."""
    a = np.asarray(db, float)
    return np.where(np.isfinite(a), np.clip(np.round((a + cdse.DB_OFFSET) * cdse.DB_SCALE), 1, 65535), 0)


def scene(target_len_px=8, fixed=False, n_prior=6, seed=0):
    """(arrays for judge, PNG for the fake client): sea -22 dB with speckle, a target along the rows at the centre."""
    rng = np.random.default_rng(seed)
    s = v.SIZE_PX
    vv = -22 + 10 * np.log10(rng.exponential(1.0, (s, s)))
    vh = vv - 7
    vq = -24 + 10 * np.log10(rng.gamma(4, 0.25, (s, s)))
    c = s // 2
    rows = slice(c - target_len_px // 2, c - target_len_px // 2 + target_len_px)
    vv[rows, c:c + 2] = 2.0
    vh[rows, c:c + 2] = -8.0
    if fixed:
        vq[rows, c:c + 2] = 0.0
    n = np.full((s, s), float(n_prior))
    arr = {"vv": vv, "vh": vh, "vv_q": vq, "n_prior": n}
    img = np.stack([code(vv), code(vh), code(vq), n], axis=-1).astype(np.uint16)
    return arr, png.encode(img, depth=16)


@unittest.skipUnless(np is not None, "numpy not installed")
class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.st = v.settings(m.load_config())

    def test_a_ship_that_was_not_there_before(self):
        arr, _ = scene(target_len_px=8)
        r = v.judge(arr, self.st)
        self.assertEqual(r["verdict"], "transient")
        self.assertEqual(r["n_prior"], 6)
        self.assertLess(r["persist_db"], 4)
        self.assertAlmostEqual(r["length_m_20m"], 160, delta=1)     # 8 px x 20 m
        self.assertAlmostEqual(r["width_m_20m"], 40, delta=1)
        self.assertLess(min(r["heading_deg_20m"], 180 - r["heading_deg_20m"]), 1)   # along the rows: north-south

    def test_a_fixed_object_is_bright_in_the_earlier_passes(self):
        arr, _ = scene(fixed=True)
        r = v.judge(arr, self.st)
        self.assertEqual(r["verdict"], "fixed")
        self.assertGreater(r["persist_db"], 15)

    def test_too_few_earlier_passes(self):
        arr, _ = scene(n_prior=2)
        r = v.judge(arr, self.st)
        self.assertEqual((r["verdict"], r["reason"]), ("unknown", "few_prior_passes"))
        self.assertIsNone(r["persist_db"])
        self.assertIsNotNone(r["length_m_20m"])          # the size at 20 m is still measured

    def test_blank_cut_out(self):
        s = v.SIZE_PX
        nan = np.full((s, s), np.nan)
        r = v.judge({"vv": nan, "vh": nan, "vv_q": nan, "n_prior": np.zeros((s, s))}, self.st)
        self.assertEqual((r["verdict"], r["reason"]), ("unknown", "no_data"))

    def test_decode_round_trip_and_tile(self):
        arr, data = scene(fixed=True)
        d = v.decode(data)
        ok = arr["vv"] > -45          # the code's floor is -50 dB
        self.assertTrue(np.allclose(d["vv"][ok], arr["vv"][ok], atol=0.001))
        self.assertTrue(np.array_equal(d["n_prior"], arr["n_prior"]))
        self.assertEqual(v.judge(d, self.st)["verdict"], "fixed")
        rgb = v.tile_rgb(d)
        self.assertEqual((rgb.shape, rgb.dtype), ((v.SIZE_PX, v.SIZE_PX, 3), np.uint8))

    def test_request(self):
        acq = {"t_to": datetime(2026, 11, 3, 10, 9, tzinfo=timezone.utc)}
        g = v.tile_grid(120.0, 24.0)
        my, mx = m_per_deg(24.0)
        self.assertAlmostEqual((g.east - g.west) * mx, 1280, delta=1)
        self.assertAlmostEqual((g.north - g.south) * my, 1280, delta=1)
        p = v.payload(g, acq, self.st)
        self.assertEqual((p["output"]["width"], p["output"]["height"]), (64, 64))
        tr = p["input"]["data"][0]["dataFilter"]["timeRange"]
        self.assertEqual((tr["from"], tr["to"]), ("2026-09-16T10:09:00Z", "2026-11-03T10:09:00Z"))
        self.assertEqual(p["input"]["data"][0]["processing"]["upsampling"], "NEAREST")
        es = p["evalscript"]
        self.assertIn('mosaicking: "ORBIT"', es)
        self.assertIn("slice(0, 9)", es)                 # the pass and 8 earlier passes
        self.assertIn("0.3 * (prior.length - 1)", es)


class VerifyClient:
    """process_raw answers with the scene of each candidate (by its longitude)."""

    def __init__(self, scenes, pu=0.28, fail_at=None):
        self.scenes, self.pu, self.fail_at = scenes, pu, fail_at
        self.processing_units, self.requests, self.calls = 0.0, 0, []

    def process_raw(self, payload, accept, input_bands):
        self.requests += 1
        self.processing_units += self.pu
        w, s, e, n = payload["input"]["bounds"]["bbox"]
        self.calls.append(round((w + e) / 2, 3))
        if self.fail_at == len(self.calls):
            raise cdse.CDSEError("500")
        return self.scenes[round((w + e) / 2, 3)], self.pu


@unittest.skipUnless(np is not None, "numpy not installed")
class VerifyPassTest(unittest.TestCase):
    def setUp(self):
        self.cfg = m.load_config()
        self.area = {"id": "TSM", "pu_month_cap": 700}
        self.acq = {"acquired_utc": "2026-11-05T10:08:57Z", "platform": "sentinel-1d", "relative_orbit": 142,
                    "t_to": datetime(2026, 11, 5, 10, 10, tzinfo=timezone.utc)}
        self.row = {"processed_utc": "2026-11-05T12:00:00Z", "pu": 30.0}
        self.args = Namespace(dry_run=False, budget_month="2026-11")

    def cands(self):
        return [{"lon": 120.1, "lat": 24.0, "_class": "A", "contrast_vv_db": 12.0},     # a ship
                {"lon": 120.2, "lat": 24.0, "_class": "A", "contrast_vv_db": 14.0},     # a wind turbine
                {"lon": 120.3, "lat": 24.0, "_class": "A", "contrast_vv_db": 9.0},      # a small boat
                {"lon": 120.4, "lat": 24.0, "_class": "B", "contrast_vv_db": 20.0}]     # B: not verified

    def go(self, client, cands, learned=None, history=(), pu_left=100.0, chip=None):
        learned = learned if learned is not None else {"points": []}
        summary = {"errors": []}
        r = m._verify_pass(self.args, self.cfg, client, type("P", (), {"wait": lambda s, x: None, "spent": lambda s, x: None})(),
                           lambda x: None, lambda: NOW, {"pu_left": pu_left}, summary, self.area, self.acq, cands,
                           list(history), learned, time.monotonic() + 60, self.row, {"chip": chip}, lambda x: None)
        return r, learned, summary

    def scenes(self):
        return {120.1: scene(8)[1], 120.2: scene(8, fixed=True)[1], 120.3: scene(2)[1]}

    def test_verdicts_learning_and_pu(self):
        cs = self.cands()
        client = VerifyClient(self.scenes())
        chip = {"crops": {}}
        r, learned, summary = self.go(client, cs, chip=chip)
        self.assertEqual(client.calls, [120.2, 120.1, 120.3])          # strongest contrast first, B never
        self.assertEqual(r["requests"], 3)
        self.assertAlmostEqual(r["pu"], 0.84)
        self.assertAlmostEqual(self.row["pu"], 30.84)
        ship, turbine, boat, b = cs
        self.assertEqual(ship["verify"]["verdict"], "transient")
        self.assertTrue(turbine.get("static_temporal"))
        self.assertTrue(boat.get("small_20m_flag"))                   # 40 m long at 20 m: flagged ...
        self.assertNotIn("small_20m", boat)                           # ... but stays A until the review (small_demote)
        self.assertNotIn("verify", b)
        self.assertEqual(r["counts"]["fixed"], 1)
        self.assertEqual(learned["points"], [])                       # the second look is never learned
        self.assertEqual(set(chip["crops20"]), {(120.1, 24.0), (120.2, 24.0), (120.3, 24.0)})
        # the classes after the second look
        for c in cs:
            c.setdefault("cluster", None)
            c.update({"confidence": "high", "v2_confidence": "A"})
        self.assertEqual(m.classify(ship), ("A", None))
        self.assertEqual(m.classify(turbine), ("ref", "static_temporal"))
        self.assertEqual(m.classify(boat), ("A", None))
        boat["small_20m"] = True                                      # with small_demote
        self.assertEqual(m.classify(boat), ("B", None))
        props = m.verify_props(ship["verify"])
        self.assertEqual((props["verified"], props["verify_verdict"], props["length_m_20m"], props["small_20m"]),
                         (True, "transient", 160, False))
        self.assertTrue(m.verify_props(boat["verify"])["small_20m"])
        self.assertEqual(m.verify_props(None), {"verified": False})
        # a point learned by the cross-orbit rule (still active) is not fetched again; an expired one is
        pt = {"lon": 120.2, "lat": 24.0, "added_utc": "2026-11-01T10:00:00Z"}
        client2 = VerifyClient(self.scenes())
        self.go(client2, self.cands(), learned={"points": [pt]})
        self.assertNotIn(120.2, client2.calls)
        client3 = VerifyClient(self.scenes())
        self.go(client3, self.cands(), learned={"points": [{**pt, "added_utc": "2026-08-01T10:00:00Z"}]})
        self.assertIn(120.2, client3.calls)

    def test_caps(self):
        client = VerifyClient(self.scenes())
        r, _, _ = self.go(client, self.cands(), pu_left=0.5)          # the run's budget: one request fits
        self.assertEqual((r["requests"], r["counts"]["skipped_cap"]), (1, 2))
        hist = [{"area": "TSM", "processed_utc": "2026-11-04T12:00:00Z", "pu": 669.5}]
        client = VerifyClient(self.scenes())
        r, _, _ = self.go(client, self.cands(), history=hist)         # the area's month cap (700): 669.5 + 30 used
        self.assertEqual((r["requests"], r["counts"]["skipped_cap"]), (0, 3))
        self.args.dry_run = True
        client = VerifyClient(self.scenes())
        r, _, _ = self.go(client, self.cands())
        self.assertEqual(client.calls, [])

    def test_requests_left_month_stop_and_other_errors(self):
        client = VerifyClient(self.scenes())
        client.requests = 9
        bud_args = dict(pu_left=100.0)
        summary = {"errors": []}
        r = m._verify_pass(self.args, self.cfg, client, type("P", (), {"wait": lambda s, x: None, "spent": lambda s, x: None})(),
                           lambda x: None, lambda: NOW, {**bud_args, "requests_left": 11}, summary, self.area, self.acq,
                           self.cands(), [], {"points": []}, time.monotonic() + 60, self.row, {"chip": None}, lambda x: None)
        self.assertEqual((r["requests"], r["counts"]["skipped_cap"]), (2, 1))     # the month's requests: 11
        from ocean import cfar_probe as cp

        class Stops(VerifyClient):
            def process_raw(self, *a):
                raise cp.MonthStop("month")
        r, _, _ = self.go(Stops(self.scenes()), self.cands())
        self.assertEqual(r["counts"]["skipped_cap"], 3)

        class Garbage(VerifyClient):
            def process_raw(self, payload, accept, input_bands):
                self.requests += 1
                self.processing_units += self.pu
                return b"not a png", self.pu
        cs = self.cands()
        r, _, summary = self.go(Garbage(self.scenes()), cs)
        self.assertEqual((r["counts"]["errors"], r["requests"]), (3, 3))
        self.assertFalse(any("verify" in c for c in cs))

    def test_missing_pu_header_counts_the_planning_value(self):
        client = VerifyClient(self.scenes(), pu=0.016)
        client.pu_reported = False
        r, _, _ = self.go(client, self.cands())
        self.assertAlmostEqual(r["pu"], 0.9)                         # 3 x est_pu 0.3, not 3 x 0.016
        self.assertTrue(r.get("pu_estimated"))

    def test_a_failed_request_leaves_the_candidate(self):
        cs = self.cands()
        client = VerifyClient(self.scenes(), fail_at=1)
        r, _, summary = self.go(client, cs)
        self.assertEqual(r["counts"]["errors"], 1)
        self.assertEqual(r["requests"], 3)
        self.assertAlmostEqual(r["pu"], 0.84)                        # the failed request's PU are counted
        self.assertNotIn("verify", cs[1])
        self.assertEqual(summary["errors"][0]["stage"], "verify")

    def test_a_client_without_process_raw_is_skipped(self):
        r, _, _ = self.go(object(), self.cands())
        self.assertEqual(r["requests"], 0)


def lattice(n_side=4, spacing=720.0, jitter=0.0, lat=24.2, lon=119.8, seed=0):
    rng = np.random.default_rng(seed)
    my, mx = m_per_deg(lat)
    out = []
    for i in range(n_side):
        for j in range(n_side):
            dx, dy = rng.normal(0, jitter, 2) if jitter else (0.0, 0.0)
            out.append({"lon": lon + (i * spacing + dx) / mx, "lat": lat + (j * spacing + dy) / my})
    return out


@unittest.skipUnless(np is not None, "numpy not installed")
class GridTest(unittest.TestCase):
    def setUp(self):
        self.rule = m.load_config()["grid"]

    def test_a_wind_farm_is_flagged(self):
        cs = lattice()
        self.assertEqual(m.grid_fields(cs, self.rule), 16)
        self.assertTrue(all(c["static_grid"] for c in cs))

    def test_an_anchorage_or_a_fleet_is_not(self):
        cs = lattice(jitter=120.0, seed=3)     # ships at anchor: spacing varies by ~15%
        self.assertEqual(m.grid_fields(cs, self.rule), 0)
        rng = np.random.default_rng(5)
        scatter = [{"lon": 120 + rng.uniform(0, 0.05), "lat": 24 + rng.uniform(0, 0.05)} for _ in range(20)]
        self.assertEqual(m.grid_fields(scatter, self.rule), 0)

    def test_small_or_absent_rule(self):
        cs = lattice(n_side=2)
        self.assertEqual(m.grid_fields(cs, self.rule), 0)
        self.assertEqual(m.grid_fields(lattice(), None), 0)

    def test_grid_member_is_reference(self):
        c = {"static_grid": True, "confidence": "high", "v2_confidence": "A", "cluster": None}
        self.assertEqual(m.classify(c), ("ref", "static_grid"))


@unittest.skipUnless(np is not None, "numpy not installed")
class CrossOrbitLearningTest(unittest.TestCase):
    def test_looks_of_other_tracks_count_within_the_window(self):
        cfg = m.load_config()
        self.assertTrue(cfg["static_learned"]["any_orbit"])
        acq = {"acquired_utc": "2026-11-20T09:52:08Z", "platform": "sentinel-1c", "relative_orbit": 171}
        rect = [[123.0, 24.0, 124.0, 25.0]]

        def h(t, plat, ro):
            return {"acquired_utc": t, "platform": plat, "relative_orbit": ro, "status": "ok", "rects": rect}
        hist = [h("2026-09-01T09:52:08Z", "sentinel-1c", 171),     # older than 45 days: not a look
                h("2026-11-02T21:40:00Z", "sentinel-1d", 69), h("2026-11-14T09:44:00Z", "sentinel-1c", 98)]
        pts = {"2026-09-01T09:52:08Z": [{"lon": 123.5, "lat": 24.5}], "2026-11-02T21:40:00Z": [{"lon": 123.5, "lat": 24.5}],
               "2026-11-14T09:44:00Z": [{"lon": 123.5004, "lat": 24.4998}]}
        learned = {"points": []}
        c = {"lon": 123.5, "lat": 24.5}
        self.assertEqual(m.learn_static([c], acq, hist, lambda t: pts[t], learned, cfg), 1)
        self.assertEqual((learned["points"][0]["hits"], learned["points"][0]["kind"]), (3, "any_orbit"))
        learned = {"points": []}
        c = {"lon": 123.5, "lat": 24.5}
        self.assertEqual(m.learn_static([c], acq, hist[:2], lambda t: pts[t], learned, cfg), 0)
        # three hits within a week (a ship at anchor): not learned (min_span_days 14)
        week = [h("2026-11-14T21:40:00Z", "sentinel-1d", 69), h("2026-11-17T09:44:00Z", "sentinel-1c", 98)]
        wpts = {"2026-11-14T21:40:00Z": [{"lon": 123.5, "lat": 24.5}], "2026-11-17T09:44:00Z": [{"lon": 123.5, "lat": 24.5}]}
        learned = {"points": []}
        self.assertEqual(m.learn_static([{"lon": 123.5, "lat": 24.5}], acq, week, lambda t: wpts[t], learned, cfg), 0)

    def test_learned_points_expire(self):
        cfg = m.load_config()
        learned = {"points": [{"lon": 1, "lat": 1, "added_utc": "2026-10-01T00:00:00Z"},
                              {"lon": 2, "lat": 2, "added_utc": "2026-11-10T00:00:00Z"}]}
        self.assertEqual([p["lon"] for p in m.active_points(learned, "2026-11-20T00:00:00Z", cfg)], [2])


if __name__ == "__main__":
    unittest.main()
