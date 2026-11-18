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
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

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


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl"), "bash and curl are needed")
class SiteClockPace(unittest.TestCase):
    def run_clock(self, status, seconds=7):
        site = Site(status)
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith(("UPKEEP_", "PUSH_TOKEN"))}
            env.update(SITE=site.url, RUN_SECONDS=str(seconds), UPKEEP_EVERY="2", LEGACY_EVERY="3600",
                       TICK="1", TRANSLATE_EVERY="99999", MODEL="none",
                       NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            out = subprocess.run(["bash", str(ROOT / "tools/site_clock.sh")], env=env, cwd=ROOT,
                                 capture_output=True, text=True, timeout=60)
            return site, out
        finally:
            site.close()

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


if __name__ == "__main__":
    unittest.main()
