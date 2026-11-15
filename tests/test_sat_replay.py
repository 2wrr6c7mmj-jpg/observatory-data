"""sat/replay: the saved chips go through the daily run's decoder, the targets are interleaved, the client prefers chips."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:   # the plain test job has no numpy: sat/replay.py and sat/run.py need it
    np = None

if np is not None:
    from sat import replay
    from sat.run import decode_chip


@unittest.skipUnless(np is not None, "numpy not installed")
class ReplayTest(unittest.TestCase):
    def test_encode_chip_round_trip(self):
        vv = np.array([[-12.5, np.nan], [-20.0, -3.25]])
        vh = np.array([[-19.0, -25.0], [np.nan, -9.5]])
        dv, dh, lia = replay.encode_chip(vv, vh), None, None
        v2, h2, lia = decode_chip(dv)
        self.assertIsNone(lia)
        self.assertTrue(np.isnan(v2[0, 1]) and np.isnan(v2[1, 0]))   # no data where either band has none
        self.assertAlmostEqual(v2[0, 0], -12.5, places=2)
        self.assertAlmostEqual(h2[1, 1], -9.5, places=2)

    def test_round_robin_chips_first_then_sites_in_turn(self):
        t = lambda n, chip: {"n": n, "chip": "c" if chip else None}
        order = replay.round_robin({"B": [t(1, False), t(2, True)], "A": [t(3, False), t(4, False)]})
        self.assertEqual([(s, x["n"]) for s, x in order], [("B", 2), ("A", 3), ("B", 1), ("A", 4)])

    def test_client_uses_a_chip_inside_the_window(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "QGZ" / "s1-2025-12-11T10-04-40Z.npz"
            p.parent.mkdir()
            np.savez_compressed(p, vv=np.full((2, 3), -10, np.float16), vh=np.full((2, 3), -16, np.float16))
            idx = replay.chip_index([Path(d)])
            t = datetime(2025, 12, 11, 10, 4, 40, tzinfo=timezone.utc)
            self.assertEqual(idx["QGZ"][0][0], t)

            class Real:
                def process(self, *a, **k):
                    return b"real", 1.5
            c = replay.ReplayClient(Real(), [], idx["QGZ"])
            data, pu = c.process([0, 0, 1, 1], 3, 2, t - timedelta(seconds=30), t + timedelta(seconds=30))
            self.assertEqual(pu, 0.0)
            self.assertEqual(c.used_chip, p)
            self.assertEqual(c.process([0, 0, 1, 1], 3, 2, t + timedelta(days=1), t + timedelta(days=1, minutes=1)), (b"real", 1.5))
            self.assertEqual(c.process([0, 0, 1, 1], 4, 2, t - timedelta(seconds=30), t + timedelta(seconds=30)), (b"real", 1.5))


if __name__ == "__main__":
    unittest.main()
