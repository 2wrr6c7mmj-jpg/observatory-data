"""raw-1: the daily run keeps the fetched chips (sat.run --keep-raw) and sat.raw_store pushes them to day branches."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

if np is not None:
    from sat import raw_store, run
    from tests import test_sat as ts  # (a module: its test classes are not collected again here)


def tree_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@unittest.skipUnless(np is not None, "numpy not installed")
class KeepRawTest(unittest.TestCase):
    def test_keep_raw_writes_chips_and_leaves_the_outputs_alone(self):
        rt = ts.RunTest()
        chip, land = rt.chip()
        with tempfile.TemporaryDirectory() as d:
            a, b, raw = Path(d) / "a", Path(d) / "b", Path(d) / "raw"
            from datetime import datetime, timezone
            from unittest import mock

            class Frozen(datetime):
                @classmethod
                def now(cls, tz=None):
                    return datetime(2026, 10, 30, 5, 17, tzinfo=timezone.utc)
            with mock.patch.object(run, "datetime", Frozen):
                self.assertEqual(rt.run_main(["--out", str(a)], ts.FakeClient(chip), land), 0)
                self.assertEqual(rt.run_main(["--out", str(b), "--keep-raw", str(raw)], ts.FakeClient(chip), land), 0)
            ha, hb = tree_hashes(a), tree_hashes(b)
            # every published file is byte for byte the same
            self.assertGreater(len(ha), 5)
            self.assertEqual(ha, hb)
            self.assertFalse(any("raw" in k for k in hb))
            metas = raw_store.scenes(raw)
            self.assertEqual(len(metas), 2)
            # the chip is the response as received, byte for byte
            self.assertEqual((metas[0].parent / json.loads(metas[0].read_text())["chip"]).read_bytes(), chip)
            vv, vh, lia, mland, meta = raw_store.load(metas[0])
            self.assertTrue(np.array_equal(mland, land))
            self.assertEqual(meta["site_id"], "TST")
            self.assertEqual(meta["detector_version"], "cfar-v1")
            self.assertIn("Contains modified Copernicus Sentinel data 2026", meta["attribution"])
            gj = json.loads((b / "detections" / "2026-10.geojson").read_text())
            self.assertEqual(sum(len(json.loads(m.read_text())["candidates"]) for m in metas), len(gj["features"]))
            self.assertTrue(all(r["same"] for r in raw_store.check(raw)))

    def test_a_failed_save_does_not_stop_the_count(self):
        from unittest import mock
        rt = ts.RunTest()
        chip, land = rt.chip()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with mock.patch.object(raw_store, "save_scene", side_effect=OSError("disk")):
                self.assertEqual(rt.run_main(["--out", str(out), "--keep-raw", str(Path(d) / "raw")], ts.FakeClient(chip), land), 0)
            self.assertEqual(len(run.read_csv(out / "ship_counts.csv")), 2)


@unittest.skipUnless(np is not None, "numpy not installed")
class ExpiredTest(unittest.TestCase):
    def test_keep_days(self):
        days = ["2026-10-20", "2026-10-22", "2026-10-23", "2026-11-06"]
        self.assertEqual(raw_store.expired(days, "2026-11-06", 14), ["2026-10-20", "2026-10-22"])
        self.assertEqual(raw_store.expired(days, "2026-11-06", 0), days[:3])


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout


@unittest.skipUnless(np is not None, "numpy not installed")
class PushTest(unittest.TestCase):
    def test_push_day_branches_and_delete_old(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            git(d, "init", "-q", "--bare", "remote.git")
            git(d, "clone", "-q", str(d / "remote.git"), "clone")
            c = d / "clone"
            for k, v in (("user.name", "t"), ("user.email", "t@example.com")):
                git(c, "config", k, v)
            (c / "f").write_text("main\n")
            git(c, "add", "f")
            git(c, "commit", "-q", "-m", "main")
            git(c, "push", "-q", "origin", "HEAD:refs/heads/main")
            git(c, "push", "-q", "origin", "HEAD:refs/heads/sat-raw/2026-10-01")   # expired
            git(c, "push", "-q", "origin", "HEAD:refs/heads/sat-raw/keep-me")      # not a day: never deleted
            head = git(c, "rev-parse", "HEAD")
            src = d / "src" / "ZJG"
            src.mkdir(parents=True)
            (src / "2026-11-05T10-48-45Z.png").write_bytes(b"\x89PNG fake")
            (src / "2026-11-05T10-48-45Z.json").write_text("{}")
            cwd = os.getcwd()
            os.chdir(c)
            try:
                s = raw_store.push(d / "src", today="2026-11-06", keep_days=14, log=lambda m: None)
                self.assertEqual(s["deleted"], ["2026-10-01"])
                self.assertEqual(s["kept"], ["2026-11-06"])
                (src / "2026-11-06T10-41-00Z.png").write_bytes(b"\x89PNG fake2")
                s2 = raw_store.push(d / "src", today="2026-11-06", keep_days=14, log=lambda m: None)
                got = raw_store.fetch(d / "dest")
            finally:
                os.chdir(cwd)
            heads = git(d / "remote.git", "for-each-ref", "--format=%(refname)", "refs/heads")
            self.assertNotIn("sat-raw/2026-10-01", heads)
            self.assertIn("sat-raw/keep-me", heads)
            self.assertIn("refs/heads/sat-raw/2026-11-06", heads)
            # the first commit of a day has no parent (no shared history); a second run adds on top
            first = git(d / "remote.git", "rev-list", "--parents", "sat-raw/2026-11-06").splitlines()
            self.assertEqual(len(first), 2)
            self.assertEqual(first[0].split()[1], s["commit"])
            self.assertEqual(len(first[1].split()), 1)
            self.assertEqual(first[0].split()[0], s2["commit"])
            self.assertEqual(git(d / "remote.git", "rev-parse", "main"), head)
            self.assertEqual(got, ["2026-11-06"])
            names = sorted(p.relative_to(d / "dest").as_posix() for p in (d / "dest").rglob("*") if p.is_file())
            self.assertEqual(names, ["2026-11-06/README.md", "2026-11-06/ZJG/2026-11-05T10-48-45Z.json",
                                     "2026-11-06/ZJG/2026-11-05T10-48-45Z.png", "2026-11-06/ZJG/2026-11-06T10-41-00Z.png"])
            self.assertIn("Contains modified Copernicus Sentinel data 2026", (d / "dest" / "2026-11-06" / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
