"""confirm-1 PR-5: the daily run (sat/pres3/confirm_run.py) and its workflow (.github/workflows/sat-confirm.yml).

What is guarded (stubs only: no request leaves the machine):
  1. the limits of a run: no scene is fetched once the next one would pass the request or byte limit, and the run says so
  2. a refused port (HTTP 4xx from the search, the COG reads or a proxy) is skipped and reported; the other ports go on; the
     refusal is asked once
  3. the requests are counted (the search through urllib; the COG reads from libcurl's log)
  4. nothing is written when the run must not go on (no frozen_utc on main), and `paths` fails on any path outside
     docs/data/sat/confirm/ and labels/<site>-confirm-answers.csv
  5. the workflow: its cron clashes with no other workflow, its concurrency group is its own, data is published only after
     the tests passed and only from main, the path check runs before publishing, the re-fit is manual only, no CDSE, no
     secret, no force push, no deleted branch
  6. a re-fit learns from the store and the answer files only, from the answers before the cutoff
numpy only; skipped where numpy is missing (the collect workflows). The workflow checks need nothing.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # the collect workflows do not install numpy; Tests does
    np = None

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / ".github" / "workflows" / "sat-confirm.yml"

if np is not None:
    from sat.pres3 import confirm_run as cr, daily
    from tests.test_sat_pres3_daily import OFFS, SECS, day_of, features, world


def scene(sid, day):
    return {"id": sid, "properties": {"datetime": day + "T02:57:55Z", "eo:cloud_cover": 10.0}, "assets": {}}


class Search:
    """urllib.request.urlopen for the STAC search: one answer (a dict, or an exception) per call."""

    def __init__(self, *answers):
        self.answers = list(answers); self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1; a = self.answers.pop(0) if self.answers else {"features": [], "links": []}
        if isinstance(a, BaseException): raise a
        return io.BytesIO(json.dumps(a).encode())


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "x", {}, io.BytesIO(b""))


NOW = None if np is None else __import__("datetime").datetime(2026, 11, 21, 7, 29, tzinfo=__import__("datetime").timezone.utc)


@unittest.skipUnless(np is not None, "numpy not installed")
class FetchTest(unittest.TestCase):
    def run_fetch(self, search, read, ports=("QDH", "QGZ"), meter=None):
        with tempfile.TemporaryDirectory() as d, mock.patch("urllib.request.urlopen", search), mock.patch("time.sleep"):
            out = Path(d) / "out"; (out / "labels").mkdir(parents=True)
            rep = cr.daily_run(Path(d) / "store", Path(d) / "win", out, list(ports), Path(d) / "r.json", now=NOW, meter=meter or cr.Meter(),
                               fetch_read=read, log=lambda m: None)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["labels"])          # no model: nothing judged, nothing written
            return rep, json.loads((Path(d) / "r.json").read_text(encoding="utf-8"))

    def test_a_refused_port_is_skipped_and_the_others_go_on(self):
        calls = []

        def read(work, port, f, bb, env_extra=None):
            calls.append((port, f["id"]))
            if port == "QDH": raise Exception("HTTP response code: 403")           # what rasterio raises for GDAL's 403
            raise RuntimeError("stop here")                                        # another failure: reported as failed
        search = Search({"features": [scene("S2B_51STA_20261120_0_L2A", "2026-11-20"), scene("S2B_51STA_20261118_0_L2A", "2026-11-18")], "links": []},
                        {"features": [scene("S2C_51STV_20261119_0_L2A", "2026-11-19")], "links": []})
        rep, saved = self.run_fetch(search, read)
        self.assertEqual(rep, saved)
        self.assertIn("refused", rep["ports"]["QDH"]); self.assertIn("failed", rep["ports"]["QGZ"])
        self.assertEqual(calls, [("QDH", "S2B_51STA_20261118_0_L2A"), ("QGZ", "S2C_51STV_20261119_0_L2A")])   # once each: no retry
        self.assertEqual(rep["problems"][:2], ["QDH refused", "QGZ failed"]); self.assertIn("no model in the store", rep["problems"])
        self.assertEqual((rep["meter"]["search_requests"], search.calls), (2, 2))

    def test_a_refused_search(self):
        for err in (http_error(403), http_error(429), urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))):
            search = Search(err); read = mock.Mock()
            rep, _ = self.run_fetch(search, read, ports=("QDH",))
            self.assertIn("refused", rep["ports"]["QDH"], err); self.assertEqual(search.calls, 1); read.assert_not_called()
        search = Search(http_error(503), http_error(503), http_error(503), http_error(503)); rep, _ = self.run_fetch(search, mock.Mock(), ports=("QDH",))
        self.assertIn("failed", rep["ports"]["QDH"]); self.assertEqual(search.calls, 4)                 # a 5xx: the search's retries

    def test_the_limits_stop_the_fetching(self):
        read = mock.Mock(side_effect=AssertionError("no scene may be fetched"))
        search = Search({"features": [scene("S2B_51STA_20261120_0_L2A", "2026-11-20")], "links": []})
        rep, _ = self.run_fetch(search, read, ports=("QDH", "QGZ"), meter=cr.Meter(max_requests=10))
        read.assert_not_called(); self.assertEqual(rep["ports"]["QDH"]["left_for_the_limits"], 1)
        self.assertEqual(rep["ports"]["QGZ"]["skipped"], "limits"); self.assertTrue(any(p.startswith("limits:") for p in rep["problems"]))
        self.assertEqual(search.calls, 1)                                                     # QGZ is not even searched
        m = cr.Meter(max_requests=1000, max_bytes=40_000_000); self.assertTrue(m.room_for_a_scene())
        m.add(10, 15_000_000); self.assertFalse(m.room_for_a_scene()); self.assertIn("bytes", m.stopped)

    def test_the_cog_requests_are_counted_from_curl_s_log(self):
        log = "\n".join(["> HEAD /B04.tif HTTP/1.1", "< HTTP/1.1 200 OK", "< Content-Length: 99999999",
                         "> GET /B04.tif HTTP/1.1", "< HTTP/1.1 206 Partial Content", "< content-length: 16384",
                         "> CONNECT host:443 HTTP/1.1", "< HTTP/1.1 200 Connection established",
                         "> GET /B04.tif HTTP/1.1", "< HTTP/2 206", "< content-length: 1000", "> GET /B08.tif HTTP/1.1", "< HTTP/1.1 403 Forbidden",
                         "< Content-Length: 120"])
        from sat.pres3 import fetch
        self.assertEqual(fetch.curl_counts(log), {"requests": 4, "status": {"200": 1, "206": 2, "403": 1}, "bytes": 17384})

    def test_refusal(self):
        for e in (http_error(403), http_error(404), Exception("HTTP response code: 429"), urllib.error.URLError(OSError("Tunnel connection failed: 407 x"))):
            self.assertTrue(cr.refusal(e), e)
        for e in (http_error(503), Exception("CURL error: Failed to connect to x port 443 after 3 ms: Couldn't connect to server"), ValueError("x")):
            self.assertFalse(cr.refusal(e), e)


@unittest.skipUnless(np is not None, "numpy not installed")
class WriteTest(unittest.TestCase):
    def test_nothing_is_written_without_frozen_utc_on_main(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("sat.confirm.load_config", return_value={"frozen_utc": None}):
            out = Path(d) / "out"; out.mkdir(); search = Search()
            with mock.patch("urllib.request.urlopen", search):
                rep = cr.daily_run(Path(d) / "s", Path(d) / "w", out, ["QDH"], Path(d) / "r.json", now=NOW, require_frozen=True, log=lambda m: None)
            self.assertEqual(list(out.iterdir()), []); self.assertEqual(search.calls, 0)
            self.assertTrue(rep["problems"] and "frozen_utc" in rep["problems"][0])

    def test_paths_outside_the_two_places_fail(self):
        ok = (" M docs/data/sat/confirm/queue.json\n?? docs/data/sat/confirm/state/QDH/2026-11.csv\n"
              "?? docs/data/sat/confirm/img/QDH/2026-11-20.webp\n M docs/data/sat/labels/QDH-confirm-answers.csv\n")
        self.assertEqual(cr.unexpected(ok), [])
        for bad in (" M docs/data/sat/quay_counts.csv", "?? docs/data/sat/labels/QDH-s2-answers.csv", " M sat/confirm.json",
                    "R  docs/data/sat/confirm/a.json -> docs/data/sat/status.json", "?? docs/data/sat/confirm.json", " D docs/data/sat/eval/presence.json",
                    '?? "docs/data/sat/labels/QDH/x.csv"'):
            self.assertTrue(cr.unexpected(ok + bad + "\n"), bad)

    def test_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.json"
            f.write_text(json.dumps({"problems": []})); self.assertEqual(cr.main(["verdict", str(f)]), 0)
            f.write_text(json.dumps({"problems": ["QDH refused"]})); self.assertEqual(cr.main(["verdict", str(f)]), 1)

    def test_the_latest_model(self):
        with tempfile.TemporaryDirectory() as d:
            for v in ("202611", "202612", "202701"): (Path(d) / f"pres3o-{v}.npz").write_bytes(b"")
            (Path(d) / "check_2026-09-01.npz").write_bytes(b"")
            self.assertEqual(cr.latest_model(d, "2026-12-15").name, "pres3o-202612.npz")
            self.assertEqual(cr.latest_model(d, "2026-11-30").name, "pres3o-202611.npz")
            self.assertIsNone(cr.latest_model(d, "2026-10-31"))


@unittest.skipUnless(np is not None, "numpy not installed")
class RefitTest(unittest.TestCase):
    def test_the_training_cells_come_from_the_store_and_the_answer_files(self):
        C, P, ans = world(n_days=20)
        C = [dict(c, port="QDH") for c in C]
        with tempfile.TemporaryDirectory() as d:
            st = daily.Store(Path(d) / "s"); st.put("QDH", C, P)
            sat = Path(d) / "sat"; (sat / "labels").mkdir(parents=True)
            rows = [f"QDH,{s},{dd},1,{v},,,owner," for (s, dd), v in sorted(ans.items()) if dd < day_of(15)]
            rows.append(f"QDH,{SECS[0]},{day_of(16)},1,1,,,ai,")                                    # not the owner: not an answer
            (sat / "labels" / "QDH-s2-answers.csv").write_text("site_id,section_id,date,verified,verified_occupied,verified_c,verified_color,verified_by,verified_note\n"
                                                              + "\n".join(rows) + "\n", encoding="utf-8")
            got, X, names = cr.store_cells(st, ["QDH"], sat, cutoff=day_of(12))
            self.assertEqual({(c["section"], c["date"]) for c in got}, {k for k in ans if k[1] < day_of(12)})
            self.assertTrue(all(c["y"] == ans[(c["section"], c["date"])] for c in got))
            # the features of those cells are the rows of the batch computation over the whole store
            Xb, names_b, _ = features(P, C)
            at = {(c["section"], c["date"]): i for i, c in enumerate(C)}
            np.testing.assert_array_equal(X, Xb[[at[(c["section"], c["date"])] for c in got]]); self.assertEqual(names, names_b)


@unittest.skipUnless(np is not None and importlib.util.find_spec("rasterio") and importlib.util.find_spec("PIL"), "rasterio / Pillow not installed")
class PictureTest(unittest.TestCase):
    def test_a_picture_from_a_window_of_the_run(self):
        from rasterio.warp import transform
        sites = {x["id"]: x for x in json.load(open(ROOT / "sat" / "sites.json"))["sites"]}; bb = sites["QDH"]["bbox"]
        xs, ys = transform("EPSG:4326", "EPSG:32651", [bb[0] - 0.01, bb[2] + 0.01], [bb[1] - 0.01, bb[3] + 0.01])
        x0, y1 = xs[0] // 20 * 20, (ys[1] // 20 + 1) * 20; w = int((xs[1] - x0) // 10) + 2; h = int((y1 - ys[0]) // 10) + 2
        with tempfile.TemporaryDirectory() as d:
            win = Path(d) / "win" / "data" / "QDH"; win.mkdir(parents=True); out = Path(d) / "out"
            band = np.full((h, w), 800, np.uint16); band[: h // 2] = 2500
            np.savez(win / "S2B_51STA_20261120_0_L2A.npz", meta=json.dumps(dict(crs="EPSG:32651", x0=float(x0), y1=float(y1), id="S2B_51STA_20261120_0_L2A")),
                     blue=band, green=band, red=band)
            self.assertFalse(cr.draw(out, Path(d) / "win", "QDH", "2026-11-19", sites))                  # no window of that day
            self.assertTrue(cr.draw(out, Path(d) / "win", "QDH", "2026-11-20", sites))
            ix = json.loads((out / "confirm" / "img" / "QDH" / "index.json").read_text(encoding="utf-8"))
            g = ix["days"]["2026-11-20"]["grid"]
            self.assertEqual((g["west"], g["south"], g["east"], g["north"]), tuple(bb)); self.assertGreater(g["width"], 300)
            self.assertTrue((out / "confirm" / "img" / "QDH" / "2026-11-20.webp").stat().st_size > 1000)


class WorkflowTest(unittest.TestCase):
    T = FLOW.read_text(encoding="utf-8") if FLOW.exists() else ""

    @staticmethod
    def crons():
        out = {}
        for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            out[f.name] = re.findall(r"cron:\s*'([^']+)'", f.read_text(encoding="utf-8"))
        return out

    @staticmethod
    def expand(field, top):
        vals = set()
        for part in field.split(","):
            step = 1
            if "/" in part: part, step = part.split("/"); step = int(step)
            lo, hi = (0, top) if part == "*" else (tuple(int(x) for x in part.split("-")) if "-" in part else (int(part), int(part)))
            vals |= set(range(lo, hi + 1, step))
        return vals

    def test_the_time_clashes_with_no_other_workflow(self):
        c = self.crons(); mine = c.pop("sat-confirm.yml"); self.assertEqual(mine, ["29 7 * * *"])
        ours = {(h, m) for m in self.expand("29", 59) for h in self.expand("7", 23)}
        for name, cs in c.items():
            for cron in cs:
                mm, hh = cron.split()[:2]
                theirs = {(h, m) for m in self.expand(mm, 59) for h in self.expand(hh, 23)}
                self.assertFalse(ours & theirs, (name, cron))
                self.assertFalse({(h, m) for (h, m) in theirs if h == 7 and abs(m - 29) < 5}, (name, cron))      # nothing within 5 minutes

    def test_its_own_concurrency_group(self):
        mine = re.search(r"group:\s*(\S+)", self.T).group(1); self.assertTrue(mine.startswith("confirm-daily"))
        for f in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            if f.name != "sat-confirm.yml":
                self.assertNotIn("confirm-daily", f.read_text(encoding="utf-8"), f.name)

    def steps(self):
        parts = re.split(r"\n      - ", self.T.split("steps:", 1)[1])
        return {re.search(r"name:\s*(.+)", p).group(1).strip() if re.search(r"name:\s*(.+)", p) else p.split("\n")[0]: p for p in parts}

    def test_publishing_only_after_the_tests_and_from_main(self):
        s = self.steps(); names = list(s)
        self.assertIn("WRITE: ${{ github.event_name != 'push' && !inputs.probe && github.ref == 'refs/heads/main' }}", self.T)
        for n in ("Keep the new patches (branch pres3-patches)", "Publish docs/data/sat/confirm"):
            self.assertIn("steps.tests.outcome == 'success'", s[n]); self.assertIn("env.WRITE == 'true'", s[n])
            self.assertLess(names.index("Only docs/data/sat/confirm/ was written"), names.index(n))
        self.assertLess(names.index("Tests"), names.index("Daily run"))
        self.assertIn("python -m unittest discover -s tests -t .", s["Tests"])
        self.assertIn("python -m sat.pres3.confirm_run paths", s["Only docs/data/sat/confirm/ was written"])
        self.assertIn("git add -A docs/data/sat/confirm\n", s["Publish docs/data/sat/confirm"])
        self.assertEqual(len(re.findall(r"git pull -q --rebase origin main", self.T)), 1)
        self.assertIn("for i in 1 2 3 4", s["Publish docs/data/sat/confirm"])
        self.assertIn("--require-frozen", s["Daily run"])
        self.assertEqual(re.findall(r"git push[^\n]*", self.T), ["git push -q origin HEAD:refs/heads/pres3-patches || { echo \"::error title=sat-confirm store::pres3-patches に push できなかった（次の実行で取り直す）\"; exit 1; }",
                                                                  "git push -q origin HEAD:main; then exit 0; fi"])

    def test_no_force_no_deletion_no_secret_no_cdse(self):
        for word in ("--force", "push -f", "+HEAD", "--delete", "branch -D", "push origin :", "secrets.", "CDSE", "sat.run", "publish.sh", "reset --hard",
                     "filter-branch", "rebase -i"):
            self.assertFalse(word in (self.T.split("jobs:")[1] if word == "CDSE" else self.T), word)       # CDSE: in the comment only

    def test_the_refit_is_manual(self):
        on = self.T.split("permissions:")[0]
        self.assertIn("workflow_dispatch:", on); self.assertIn("refit:", on); self.assertEqual(on.count("cron:"), 1)
        s = self.steps(); self.assertIn("inputs.refit && env.WRITE == 'true'", s["Re-fit"])
        self.assertIn("scikit-learn==1.9.1", self.T)

    def test_the_probe_writes_nothing(self):
        self.assertIn("branches: ['probe/sat-confirm-*']", self.T)
        self.assertIn("PROBE: ${{ github.event_name == 'push' || inputs.probe }}", self.T)
        run = self.steps()["Daily run"]
        probe = run.split("elif")[0]
        self.assertIn('--out "$RUNNER_TEMP/out"', probe); self.assertIn("--ports QDH --max-scenes 1", probe)


if __name__ == "__main__":
    unittest.main()
