"""sat/publish.sh: the port run's push when ocean-monitor rewrote the shared chips/index.json in between."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401 - sat.chips (run as a subprocess below) needs it; the collect workflows have none
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "sat" / "publish.sh"
IDX = "docs/data/sat/chips/index.json"
FRAG = "docs/data/sat/chips/ocean/index.json"


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def port_site(site_id, acq):
    return {"site_id": site_id, "scenes": [{"acquired_utc": acq, "chip": f"chips/{site_id}/x.json"}]}


def ocean_area(area_id, acq):
    return {"area_id": area_id, "scenes": [{"acquired_utc": acq, "chip": f"chips/ocean/{area_id}/x.json"}]}


def index(sites, ocean):
    return {"version": "chips-1", "keep": 10, "updated_utc": None, "caption": "c", "credit": "x", "note": "n",
            "sites": sites, "ocean": ocean}


@unittest.skipUnless(shutil.which("git") and shutil.which("bash") and HAVE_NP, "git / bash / numpy not available")
class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "PUBLISH_SLEEP": "0",
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
                    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        self.remote = self.tmp / "remote.git"
        self.git(self.tmp, "init", "-q", "--bare", "-b", "main", str(self.remote))
        seed = self.clone("seed")
        self.write(seed, IDX, dump(index([port_site("ZJG", "2026-11-01T10:00:00Z")], [ocean_area("AAA", "2026-11-01T20:00:00Z")])))
        self.write(seed, FRAG, dump({"ocean": [ocean_area("AAA", "2026-11-01T20:00:00Z")]}))
        self.write(seed, "docs/data/sat/ship_counts.csv", "day\n1\n")
        self.commit_push(seed, "seed")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, cwd, *args, check=True):
        return subprocess.run(["git", *args], cwd=cwd, env=self.env, check=check, capture_output=True, text=True)

    def clone(self, name):
        d = self.tmp / name
        self.git(self.tmp, "clone", "-q", str(self.remote), str(d))
        self.git(d, "checkout", "-q", "-B", "main")
        return d

    def write(self, repo, rel, text):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def commit_push(self, repo, msg):
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", msg)
        self.git(repo, "push", "-q", "origin", "HEAD:main")

    def ocean_pushes(self, acq="2026-11-02T20:00:00Z"):
        """ocean-monitor: a new pass, its fragment and its splice of the shared index."""
        oc = self.clone("ocean")
        self.write(oc, FRAG, dump({"ocean": [ocean_area("AAA", acq)]}))
        idx = json.loads((oc / IDX).read_text(encoding="utf-8"))
        idx["ocean"] = [ocean_area("AAA", acq)]
        self.write(oc, IDX, dump(idx))
        self.commit_push(oc, "ocean")

    def publish(self, repo):
        return subprocess.run(["bash", str(SCRIPT), "port data"], cwd=repo, env=self.env, capture_output=True, text=True)

    def main_file(self, rel):
        return self.git(self.tmp, "--git-dir", str(self.remote), "show", f"main:{rel}").stdout

    def port_run(self, repo, touch_index=True):
        self.write(repo, "docs/data/sat/ship_counts.csv", "day\n1\n2\n")
        if touch_index:
            idx = json.loads((repo / IDX).read_text(encoding="utf-8"))
            idx["sites"] = [port_site("ZJG", "2026-11-02T10:00:00Z")]
            self.write(repo, IDX, dump(idx))

    def test_ocean_pushed_in_between(self):
        port = self.clone("port")
        self.port_run(port)
        self.ocean_pushes()
        r = self.publish(port)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        idx = json.loads(self.main_file(IDX))
        self.assertEqual(idx["sites"][0]["scenes"][0]["acquired_utc"], "2026-11-02T10:00:00Z")   # the port run's
        self.assertEqual(idx["ocean"][0]["scenes"][0]["acquired_utc"], "2026-11-02T20:00:00Z")   # ocean's, kept
        self.assertEqual(idx["credit"], "Contains modified Copernicus Sentinel data 2026")
        self.assertEqual(self.main_file("docs/data/sat/ship_counts.csv"), "day\n1\n2\n")
        self.assertEqual(self.git(port, "status", "--porcelain").stdout, "")

    def test_plain_rebase_would_conflict(self):
        """Documents the bug: without the splice the port's commit cannot be rebased onto ocean's."""
        port = self.clone("port")
        self.port_run(port)
        self.git(port, "add", "-A")
        self.git(port, "commit", "-q", "-m", "port")
        self.ocean_pushes()
        self.assertNotEqual(self.git(port, "pull", "-q", "--rebase", "origin", "main", check=False).returncode, 0)

    def test_index_untouched_by_the_run(self):
        port = self.clone("port")
        self.port_run(port, touch_index=False)
        self.ocean_pushes()
        r = self.publish(port)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        idx = json.loads(self.main_file(IDX))
        self.assertEqual(idx["ocean"][0]["scenes"][0]["acquired_utc"], "2026-11-02T20:00:00Z")
        self.assertEqual(idx["sites"][0]["scenes"][0]["acquired_utc"], "2026-11-01T10:00:00Z")
        log = self.git(self.tmp, "--git-dir", str(self.remote), "log", "--format=%s", "main").stdout.split("\n")
        self.assertEqual(log[0], "port data")   # no index commit of the port run

    def test_push_refused_once(self):
        """A refused push drops the index commit and makes it again on the next main."""
        hook = self.remote / "hooks" / "pre-receive"
        flag = self.tmp / "refused"
        hook.write_text(f"#!/bin/sh\nif [ ! -e '{flag}' ]; then touch '{flag}'; exit 1; fi\nexit 0\n", encoding="utf-8")
        hook.chmod(0o755)
        port = self.clone("port")
        self.port_run(port)
        r = self.publish(port)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(flag.exists())
        idx = json.loads(self.main_file(IDX))
        self.assertEqual(idx["sites"][0]["scenes"][0]["acquired_utc"], "2026-11-02T10:00:00Z")
        log = self.git(self.tmp, "--git-dir", str(self.remote), "log", "--format=%s", "main").stdout.split("\n")
        self.assertEqual(log[:2], ["データ更新: 衛星画像の索引（港） [sat-ships]", "port data"])

    def test_gives_up_with_an_error(self):
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
        port = self.clone("port")
        self.port_run(port)
        r = self.publish(port)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("::error title=sat-ships::push できなかった", r.stdout)


if __name__ == "__main__":
    unittest.main()
