"""tools/site_clock.sh：サイトの /api/status に upkeep 2 以上の印がある時だけ短い間隔で背景更新を進め、
印がない・読めない時は旧来の間隔（LEGACY_EVERY・LEGACY_STEPS）に戻して警告を出す。
手元の HTTP サーバーをサイトの代わりにし、間隔を秒に縮めて数える。標準ライブラリと bash・curl だけ。"""
import json
import os
import shutil
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Site:
    """A fake site: /api/status answers `status` (a dict, or an HTTP code), /api/maintenance answers idle."""

    def __init__(self, status):
        self.status = status
        self.maintenance = 0
        self.status_reads = 0
        self.dispatches = []          # (path, Authorization, body) of POSTs (the GitHub API's workflow_dispatch)
        self.dispatch_code = 204
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n).decode() if n else ""
                site.dispatches.append((self.path, self.headers.get("Authorization"), body))
                self.send_response(site.dispatch_code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                if self.path == "/api/status":
                    site.status_reads += 1
                    if isinstance(site.status, int):
                        return self.reply(site.status, {"error": "unavailable"})
                    return self.reply(200, site.status)
                if self.path == "/api/maintenance":
                    site.maintenance += 1
                    return self.reply(200, {"task": "idle"})
                self.reply(404, {})

            def reply(self, code, body):
                data = json.dumps(body, separators=(",", ":")).encode()  # as the site answers (site_upkeep.sh reads "task":"…")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ClockRun:
    """Runs tools/site_clock.sh against a fake site (and a fake GitHub API: the same server)."""

    def run_clock(self, status, seconds=7, extra=None, site=None):
        site = site or Site(status)
        try:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("UPKEEP_", "PUSH_TOKEN", "JSO_", "GITHUB_"))}
            env.update(extra or {})
            env.update(SITE=site.url, RUN_SECONDS=str(seconds), UPKEEP_EVERY="2", LEGACY_EVERY="3600",
                       TICK="1", TRANSLATE_EVERY="99999", MODEL="none",
                       NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            out = subprocess.run(["bash", str(ROOT / "tools/site_clock.sh")], env=env, cwd=ROOT,
                                 capture_output=True, text=True, timeout=60)
            return site, out
        finally:
            site.close()


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl"), "bash and curl are needed")
class SiteClockPace(ClockRun, unittest.TestCase):
    def test_marked_site_is_stepped_at_the_short_interval(self):
        site, out = self.run_clock({"upkeep": 2, "lastRunAt": None})
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertGreaterEqual(site.maintenance, 3)
        self.assertEqual(site.maintenance, site.status_reads)
        self.assertNotIn("::warning", out.stdout)

    def test_site_without_the_mark_falls_back_to_the_old_pace(self):
        site, out = self.run_clock({"lastRunAt": None})
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(site.maintenance, 1)          # one round, then LEGACY_EVERY later
        self.assertGreaterEqual(site.status_reads, 3)  # the mark is read before each round
        self.assertIn("::warning title=site clock::", out.stdout)

    def test_unreadable_status_falls_back_to_the_old_pace(self):
        for status in (503, {"upkeep": "2"}, {"upkeep": True}):
            site, out = self.run_clock(status, seconds=4)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(site.maintenance, 1, status)
            self.assertIn("::warning title=site clock::", out.stdout)



@unittest.skipUnless(shutil.which("bash") and shutil.which("curl"), "bash and curl are needed")
class SiteClockStartsJso(ClockRun, unittest.TestCase):
    """jso-update.yml is started from the clock (workflow_dispatch on main), at the start and then every JSO_*_EVERY."""

    def run_with_token(self, seconds, every, code=204):
        import tempfile
        site = Site({"upkeep": 2, "lastRunAt": None})
        site.dispatch_code = code
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "push"
            tok.write_text("t0ken")
            extra = dict(PUSH_TOKEN_FILE=str(tok), GITHUB_REPOSITORY="o/r", GITHUB_API_URL=site.url,
                         JSO_DAY_EVERY=str(every), JSO_NIGHT_EVERY=str(every))
            return self.run_clock(None, seconds=seconds, extra=extra, site=site)

    def test_started_at_once_and_then_at_the_interval(self):
        site, out = self.run_with_token(seconds=7, every=3)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        paths = [p for p, _, _ in site.dispatches]
        self.assertIn(len(paths), (2, 3), paths)      # t=0, 3, (6)
        self.assertEqual(set(paths), {"/repos/o/r/actions/workflows/jso-update.yml/dispatches"})
        self.assertEqual({a for _, a, _ in site.dispatches}, {"Bearer t0ken"})
        self.assertEqual({json.loads(b)["ref"] for _, _, b in site.dispatches}, {"main"})
        self.assertIn("jso-update started %d" % len(paths), out.stdout)
        self.assertNotIn("t0ken", out.stdout + out.stderr)

    def test_long_interval_starts_once(self):
        site, out = self.run_with_token(seconds=4, every=1800)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(len(site.dispatches), 1)

    def test_refused_start_is_a_warning_not_a_failure(self):
        site, out = self.run_with_token(seconds=3, every=1800, code=403)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("jso-update.yml を起動できなかった（HTTP 403）", out.stdout)

    def test_without_a_token_nothing_is_started(self):
        site, out = self.run_clock({"upkeep": 2, "lastRunAt": None}, seconds=3)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(site.dispatches, [])


if __name__ == "__main__":
    unittest.main()
