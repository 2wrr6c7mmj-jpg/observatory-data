import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

try:  # the satellite job installs these; the daily collection job does not
    import numpy as np
    import scipy  # noqa: F401
    import tifffile  # noqa: F401
    from PIL import Image  # noqa: F401
    from sat import detect, run, sentinel_hub as sh
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("satellite dependencies not installed")


def scene(seed=0):
    """400x400 px: land on the left 80 columns, speckled water, three ships and one at the quay."""
    rng = np.random.default_rng(seed)
    vv = rng.gamma(4.0, 0.004 / 4.0, (400, 400)).astype("float32")   # water ~ -24 dB
    vh = rng.gamma(4.0, 0.0008 / 4.0, (400, 400)).astype("float32")
    vv[:, :80] = rng.gamma(4.0, 0.2 / 4.0, (400, 80))                    # land ~ -7 dB
    vh[:, :80] = rng.gamma(4.0, 0.05 / 4.0, (400, 80))
    for r, c, length in [(100, 200, 15), (250, 300, 6), (330, 150, 3), (200, 84, 8)]:
        vv[r:r + 2, c:c + length] = 2.0
        vh[r:r + 2, c:c + length] = 0.3
    return vv, vh, np.ones_like(vv, dtype=bool)


class Detect(unittest.TestCase):
    def test_finds_the_ships_and_separates_the_quay_band(self):
        vv, vh, valid = scene()
        cands, masks = detect.detect(vv, vh, valid, 10.0)
        zones = sorted(c["zone"] for c in cands)
        self.assertEqual(zones, ["岸壁帯", "港内", "港内", "港内"])
        longest = max(cands, key=lambda c: c["pixels"])
        self.assertEqual(longest["length_class"], "100〜200m")
        self.assertTrue(masks["land"][:, :70].all())

    def test_plain_water_gives_no_candidates(self):
        rng = np.random.default_rng(3)
        vv = rng.gamma(4.0, 0.001, (300, 300)).astype("float32")
        vh = rng.gamma(4.0, 0.0002, (300, 300)).astype("float32")
        cands, _ = detect.detect(vv, vh, np.ones_like(vv, dtype=bool), 10.0)
        self.assertLessEqual(len(cands), 1)


class Auth(unittest.TestCase):
    def test_refused_client_gives_a_clear_error(self):
        err = urllib.error.HTTPError(sh.TOKEN_URL, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"invalid_client"}'))
        with mock.patch.object(sh, "_post", side_effect=err):
            with self.assertRaises(sh.AuthError) as cm:
                sh.token("id", "secret")
        self.assertIn("有効期限切れ", str(cm.exception))

    def test_missing_secrets(self):
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(sh.AuthError):
            sh.token()

    def test_main_reports_auth_error_with_exit_code_2(self):
        with mock.patch.object(run, "process", side_effect=sh.AuthError("x")), mock.patch("sys.argv", ["run"]):
            self.assertEqual(run.main(), 2)


class Run(unittest.TestCase):
    def test_end_to_end_with_a_fake_client_and_skips_processed_scenes(self):
        import tifffile
        vv, vh, valid = scene()
        site = run.SITES["sites"]["ZJG"]
        _, w, h = run.grid(site)
        big = np.zeros((h, w, 3), "float32")
        big[..., 0] = np.resize(vv, (h, w)); big[..., 1] = np.resize(vh, (h, w)); big[..., 2] = 1
        buf = io.BytesIO(); tifffile.imwrite(buf, big)

        class Fake:
            token = staticmethod(lambda: "t")
            search = staticmethod(lambda *a: [{"id": "S1A_X", "datetime": "2026-10-25T10:40:00Z", "platform": "sentinel-1a",
                                                "orbit_state": "ascending", "relative_orbit": 113, "polarizations": ["VV", "VH"]}])
            cutout = staticmethod(lambda *a: (buf.getvalue(), 4.5))

        with tempfile.TemporaryDirectory() as d, mock.patch.object(run, "OUT", Path(d)):
            run.process("ZJG", 12, 3, client=Fake)
            run.process("ZJG", 12, 3, client=Fake)       # second run: nothing new
            counts = run.read_csv(Path(d) / "ship_counts.csv")
            usage = run.read_csv(Path(d) / "usage.csv")
            self.assertEqual(len(counts), 1)
            self.assertGreater(int(counts[0]["harbour_candidates"]), 0)
            self.assertEqual([u["processing_units"] for u in usage if u["request"] == "process"], ["4.5"])
            fc = json.loads((Path(d) / "detections" / "2026-10.geojson").read_text(encoding="utf-8"))
            self.assertTrue(all(f["properties"]["status"] == "候補" for f in fc["features"]))
            self.assertTrue(list((Path(d) / "quicklook" / "ZJG").glob("*.png")))
