"""analysis.health（docs/data/health.json）の判定：OK／WARN／ALERT／HALT、ファイルが無い・壊れている場合、Actions の REST の偽の応答。
標準ライブラリだけ（収集のワークフローでも skip せずに動く）。"""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analysis import health as h

NOW = datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc)


def z(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(**kw) -> str:
    return z(NOW - timedelta(**kw))


def scene(acq: str, ro: int, mask="ZJG-land-v2"):
    return {"site_id": "ZJG", "acquired_utc": acq, "relative_orbit": ro, "platform": "sentinel-1c",
            "mask_version": mask, "coverage": 0.99, "result": "counted", "quay_result": "counted"}


# 湛江の撮影：ro157 は 6 日ごと（10-02・08・14・20・26）、ro84 は 12 日ごと（10-03・15・27）。すべて揃っている
FULL_SCENES = ([scene(f"2026-10-{d:02d}T10:48:45Z", 157) for d in (2, 8, 14, 20, 26)]
               + [scene(f"2026-10-{d:02d}T10:41:00Z", 84) for d in (3, 15, 27)])


def good_files() -> dict:
    """基準内の偽のデータ（docs/data からの相対パス → 中身）。"""
    return {
        "sat/status.json": {
            "updated_utc": ago(hours=20),
            "scenes": [dict(s, mask_version=None) for s in FULL_SCENES] + FULL_SCENES,
            "last_run": {"run_utc": ago(hours=20), "month_pu_before": 40.0, "month_pu_cap": 1000.0,
                         "processing_units": 10.0, "scenes": 6,
                         "sites": [{"site_id": "ZJG", "errors": 0, "budget_left_pu": 900.0}]},
        },
        "sat/meta.json": {"sites": [{"id": "ZJG", "name": "湛江", "mask_version": "ZJG-land-v2", "daily": True,
                                     "publish": "trial"}],
                          "attribution": {"credit": "Contains modified Copernicus Sentinel data 2026"}},
        "sat/usage.csv": ("run_utc,requests,processing_units\n"
                          f"{ago(days=2)},5,40.0\n{ago(hours=20)},7,10.0\n"),
        "sat/quay_counts.csv": ("site_id,acquired_utc,mask_version,est_version,est_mode,est_p10,est_p90\n"
                                "ZJG,2026-10-27T10:41:00Z,ZJG-land-v2,est-1,35,30,41\n"),
        "sat/quay_est2.csv": ("site_id,acquired_utc,mask_version,est2_version,warship_median,warship_p10,warship_p90\n"
                              "ZJG,2026-10-27T10:41:00Z,ZJG-land-v2,est-2,21,18,25\n"),
        "sat/CHANGES.md": "| 日付 | 版 | 内容 |\n| --- | --- | --- |\n| 2026-10-31 | est-2-cal2 | x |\n",
        "jcg-senkaku/status.json": {"last_day": "2026-10-30", "days": 10,
                                    "last_run": {"result": "ok", "finished_utc": ago(hours=10),
                                                 "robots": "x: allowed"}},
        "jcg-senkaku/daily.json": {"fields": ["date", "contiguous_zone", "territorial_sea", "check"],
                                   "rows": [["2026-10-30", 4, 0, "match"]]},
        "pla-tracker/status.json": {"last_success_utc": ago(hours=10), "last_day": "2026-10-31",
                                    "last_run": {"result": "ok", "robots": "https://x/robots.txt: allowed"},
                                    "source": {"http": 200}},
        "pla-tracker/daily.json": {"fields": ["date", "aircraft_total", "median_line_cross", "ships_total"],
                                   "rows": [["2026-10-31", 6, 3, 8]]},
        "taiwan-mnd/status.json": {"collection": "robots.txt が拒否しているため停止中", "last_day": "2026-10-30",
                                   "generated_utc": ago(days=1)},
        "jso/status.json": {"built_utc": ago(hours=3), "last_release": "2026-10-27", "passages": 1024},
        "launches/events.json": {"generated_utc": ago(hours=3), "events": [
            {"site": "内之浦", "window_start_utc": z(NOW + timedelta(days=1)), "window_end_utc": z(NOW + timedelta(days=2))},
            {"site": "古い", "window_start_utc": "2025-09-08T18:00Z", "window_end_utc": "2025-09-08T22:00Z"}]},
        "launches/notices.json": {"last_retrieved_utc": ago(hours=3), "notices": {
            "a": {"last_seen_utc": ago(hours=3), "end_utc": None},
            "b": {"last_seen_utc": ago(days=5), "end_utc": ago(days=4)}}},
        "launches/ll2.json": {"last_retrieved_utc": ago(hours=3)},
        "ocean/coverage_probe.json": {"generated_utc": ago(days=1)},
    }


CRONS = {"collect.yml": "23 1 * * *", "jcg-senkaku.yml": "43 3 * * *", "sat-ships.yml": "17 5 * * *",
         "pla-tracker.yml": "37 6 * * *"}


def run(event, conclusion, started, status="completed"):
    return {"id": 1, "event": event, "status": status, "conclusion": conclusion,
            "run_started_at": started, "created_at": started, "html_url": "https://example.invalid/run"}


def good_runs() -> dict:
    out = {f: [run("schedule", "success", ago(hours=10))] for f in CRONS}
    out["proto.yml"] = [run("workflow_run", "success", ago(hours=1))]
    return out


class Fixture:
    def __init__(self, files=None, runs=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.data = self.repo / "docs" / "data"
        for rel, body in (good_files() if files is None else files).items():
            p = self.data / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body if isinstance(body, str) else json.dumps(body, ensure_ascii=False), encoding="utf-8")
        (self.repo / "sat").mkdir()
        (self.repo / "sat" / "sites.json").write_text(json.dumps({"sites": [
            {"id": "ZJG", "daily": True, "publish": "trial"}, {"id": "SYL", "daily": False}]}), encoding="utf-8")
        wf = self.repo / ".github" / "workflows"
        wf.mkdir(parents=True)
        for f, c in CRONS.items():
            (wf / f).write_text(f"on:\n  schedule:\n    - cron: '{c}'\n", encoding="utf-8")
        self.runs = good_runs() if runs is None else runs

    def fetch(self, file):
        r = self.runs.get(file, [])
        if isinstance(r, Exception):
            raise r
        return copy.deepcopy(r)

    def build(self, fetch=True, now=NOW, prev=None):
        return h.build(now, self.data, self.repo, self.fetch if fetch else None, prev)

    def close(self):
        self.tmp.cleanup()


def src(health, sid):
    return next(s for s in health["sources"] if s["id"] == sid)


def codes(item):
    return {r["code"] for r in item["reasons"]}


def wf(health, file):
    return next(w for w in health["workflows"] if w["file"] == file)


class Base(unittest.TestCase):
    def make(self, files=None, runs=None, mutate=None):
        f = good_files() if files is None else files
        if mutate:
            mutate(f)
        fx = Fixture(f, runs)
        self.addCleanup(fx.close)
        return fx


class AllGoodTest(Base):
    def test_all_ok(self):
        out = self.make().build()
        self.assertEqual(out["state"], "ok")
        for s in out["sources"]:
            want = {"taiwan-mnd": "halt", "ocean": "info"}.get(s["id"], "ok")
            self.assertEqual(s["state"], want, (s["id"], s["reasons"]))
        for w in out["workflows"]:
            self.assertEqual(w["state"], "ok", (w["file"], w["reasons"]))
        self.assertEqual(out["summary"]["halt"], 1)
        self.assertEqual(out["thresholds"]["review_by"], "2026-12")
        self.assertTrue(out["thresholds"]["provisional"])
        self.assertEqual(out["sat"]["gaps"], [])
        self.assertEqual(out["sat"]["sites_configured_not_daily"], ["SYL"])
        self.assertEqual(out["sat"]["changes_latest"]["version"], "est-2-cal2")

    def test_latest_and_trial(self):
        out = self.make().build()
        zjg = out["latest"]["sat"][0]
        self.assertEqual((zjg["site_id"], zjg["publish"], zjg["trial"]), ("ZJG", "trial", True))
        self.assertEqual((zjg["est1"]["mode"], zjg["est1"]["p10"], zjg["est1"]["p90"]), (35, 30, 41))
        self.assertEqual(zjg["est2"]["warship_median"], 21)
        self.assertEqual(out["latest"]["jcg"]["contiguous_zone"], 4)
        self.assertEqual(out["latest"]["taiwan"]["ships_total"], 8)
        self.assertEqual(out["latest"]["launches"]["open_or_upcoming_windows"], 1)
        self.assertEqual(out["latest"]["launches"]["next_start"]["site"], "内之浦")
        self.assertEqual(out["latest"]["launches"]["active_notices"], 1)

    def test_current_mask_only(self):
        # 旧範囲（mask_version なし）の新しい行は使わない
        def m(f):
            f["sat/status.json"]["scenes"].append(dict(scene("2026-10-30T10:48:45Z", 157), mask_version="old"))
        out = self.make(mutate=m).build()
        self.assertEqual(src(out, "sat")["data_date"], "2026-10-27")

    def test_pu_consistent_has_no_note(self):
        pu = self.make().build()["sat"]["pu"]
        self.assertEqual(pu["usage_csv_pu"], 50.0)
        self.assertEqual(pu["difference_pu"], 0.0)
        self.assertNotIn("note", pu)


class SatTest(Base):
    def test_heartbeat(self):
        for hours_, want in ((29, "ok"), (40, "warn"), (60, "alert")):
            out = self.make(mutate=lambda f: f["sat/status.json"].update(updated_utc=ago(hours=hours_))).build()
            self.assertEqual(src(out, "sat")["state"], want, hours_)

    def test_scene_age(self):
        def only(day):
            def m(f):
                f["sat/status.json"]["scenes"] = [scene(f"2026-10-{day:02d}T10:48:45Z", 157)]
                f["sat/status.json"]["scenes"] += [scene(f"2026-10-{day - 6:02d}T10:48:45Z", 157)]
            return m
        # 撮影の抜け（軌道の期待）も出るので、scene_old だけを見る
        out = self.make(mutate=only(24)).build()
        self.assertNotIn("scene_old", codes(src(out, "sat")))
        out = self.make(mutate=only(22)).build()            # 9 日
        self.assertIn(("warn", "scene_old"), {(r["level"], r["code"]) for r in src(out, "sat")["reasons"]})
        out = self.make(mutate=only(16)).build()            # 15 日
        self.assertIn(("alert", "scene_old"), {(r["level"], r["code"]) for r in src(out, "sat")["reasons"]})
        self.assertEqual(src(out, "sat")["state"], "alert")

    def test_orbit_gaps_like_october(self):
        # 2部の指摘 (a)：10-20（ro157）と 10-27（ro84）が無い
        def m(f):
            sc = [s for s in FULL_SCENES if not s["acquired_utc"].startswith(("2026-10-20", "2026-10-27"))]
            f["sat/status.json"]["scenes"] = sc
        out = self.make(mutate=m).build()
        gaps = [(g["relative_orbit"], g["expected_utc"][:10], g["level"]) for g in out["sat"]["gaps"]]
        self.assertEqual(gaps, [(157, "2026-10-20", "warn"), (84, "2026-10-27", "warn")])
        self.assertEqual(src(out, "sat")["state"], "warn")
        self.assertIn("orbit_gap", codes(src(out, "sat")))

    def test_gap_not_due_yet(self):
        # 次の ro157（11-01）は期待日＋猶予の前なので抜けではない
        out = h.orbit_gaps(FULL_SCENES, {"157": 6}, datetime(2026, 11, 2, 12, tzinfo=timezone.utc), None)
        self.assertEqual(out, [])
        out = h.orbit_gaps(FULL_SCENES, {"157": 6}, datetime(2026, 11, 3, 4, tzinfo=timezone.utc), None)
        self.assertEqual([g["expected_utc"][:10] for g in out], ["2026-11-01"])

    def test_old_gap_is_info(self):
        def m(f):
            f["sat/status.json"]["scenes"] = [s for s in FULL_SCENES if not s["acquired_utc"].startswith("2026-10-08")]
        out = self.make(mutate=m).build()
        self.assertEqual([g["level"] for g in out["sat"]["gaps"]], ["info"])
        self.assertEqual(src(out, "sat")["state"], "ok")

    def test_errors_and_budget(self):
        def m(f):
            f["sat/status.json"]["last_run"]["sites"] = [{"site_id": "ZJG", "errors": 2, "budget_left_pu": 0}]
        s = src(self.make(mutate=m).build(), "sat")
        self.assertEqual(s["state"], "alert")
        self.assertTrue({"run_errors", "budget_exhausted"} <= codes(s))

    def test_pu_high(self):
        def m(f):
            f["sat/usage.csv"] = f"run_utc,requests,processing_units\n{ago(hours=30)},5,850\n"
        s = src(self.make(mutate=m).build(), "sat")
        self.assertIn("pu_high", codes(s))

    def test_pu_mismatch_note(self):
        # 2部の指摘 (b)：usage.csv と month_pu_before が合わない → 両方を出し、差を注記
        def m(f):
            f["sat/usage.csv"] += f"{ago(days=1)},84,28.138\n"
        pu = self.make(mutate=m).build()["sat"]["pu"]
        self.assertEqual(pu["status_month_pu_before"], 40.0)
        self.assertEqual(pu["usage_csv_pu_before_last_run"], 68.138)
        self.assertEqual(pu["difference_pu"], 28.138)
        self.assertIn("合わない", pu["note"])


class OtherSourcesTest(Base):
    def test_jcg(self):
        out = self.make(mutate=lambda f: f["jcg-senkaku/status.json"].update(last_day="2026-10-28")).build()
        self.assertEqual(src(out, "jcg-senkaku")["state"], "warn")      # JST 今日−3
        out = self.make(mutate=lambda f: f["jcg-senkaku/status.json"].update(last_day="2026-10-26")).build()
        self.assertEqual(src(out, "jcg-senkaku")["state"], "alert")     # −5
        out = self.make(mutate=lambda f: f["jcg-senkaku/daily.json"]["rows"][-1].__setitem__(3, "mismatch")).build()
        self.assertIn("check_mismatch", codes(src(out, "jcg-senkaku")))
        out = self.make(mutate=lambda f: f["jcg-senkaku/status.json"]["last_run"].update(result="error")).build()
        self.assertIn("last_run_not_ok", codes(src(out, "jcg-senkaku")))

    def test_pla(self):
        out = self.make(mutate=lambda f: f["pla-tracker/status.json"].update(last_day="2026-10-29")).build()
        self.assertEqual(src(out, "pla-tracker")["state"], "warn")      # 台湾 今日−2
        out = self.make(mutate=lambda f: f["pla-tracker/status.json"].update(last_day="2026-10-28")).build()
        self.assertEqual(src(out, "pla-tracker")["state"], "alert")
        out = self.make(mutate=lambda f: f["pla-tracker/status.json"].update(
            last_run={"result": "skipped", "reason": "x: HTTP 502"})).build()
        self.assertEqual(src(out, "pla-tracker")["state"], "warn")
        out = self.make(mutate=lambda f: f["pla-tracker/status.json"].update(
            last_run={"result": "skipped", "reason": "robots.txt disallows"})).build()
        self.assertEqual(src(out, "pla-tracker")["state"], "alert")

    def test_mnd_halt_and_active(self):
        out = self.make().build()
        self.assertEqual(src(out, "taiwan-mnd")["state"], "halt")
        self.assertEqual(src(out, "taiwan-mnd")["kind"], "halted")

        def active(f):
            f["taiwan-mnd/status.json"].update(collection="active", generated_utc=ago(hours=70))
        s = src(self.make(mutate=active).build(), "taiwan-mnd")
        self.assertEqual((s["kind"], s["state"]), ("daily", "alert"))

    def test_jso(self):
        for last, want in (("2026-10-21", "ok"), ("2026-10-16", "warn"), ("2026-10-01", "alert")):
            out = self.make(mutate=lambda f: f["jso/status.json"].update(last_release=last)).build()
            self.assertEqual(src(out, "jso")["state"], want, last)

    def test_launches_notam_watch_old(self):
        def m(f):
            f["launches/notices.json"]["notices"]["a"]["last_seen_utc"] = ago(hours=40)
        s = src(self.make(mutate=m).build(), "launches")
        self.assertEqual(s["state"], "warn")
        self.assertIn("notam_watch_old", codes(s))

    def test_proto_older_than_input(self):
        runs = good_runs()
        runs["proto.yml"] = [run("workflow_run", "success", ago(hours=5))]   # jso の再構築（3 h 前）より前
        self.assertEqual(src(self.make(runs=runs).build(), "proto")["state"], "warn")


class BrokenFilesTest(Base):
    def test_missing_files(self):
        out = self.make(files={}).build()      # docs/data が空
        for s in out["sources"]:
            if s["id"] in ("ocean", "proto"):
                continue
            self.assertEqual(s["state"], "warn", s["id"])
            self.assertIn("unreadable", codes(s), s["id"])
        self.assertEqual(out["latest"], {})

    def test_broken_json_and_csv(self):
        def m(f):
            f["sat/status.json"] = "{not json"
            f["jcg-senkaku/daily.json"] = "\x00\x01"
            f["sat/quay_counts.csv"] = b"\xff\xfe".decode("latin-1")
        out = self.make(mutate=m).build()
        self.assertEqual(src(out, "sat")["state"], "warn")
        self.assertIn("unreadable", codes(src(out, "sat")))
        jcg = src(out, "jcg-senkaku")
        self.assertIn("unreadable", codes(jcg))
        self.assertNotIn("jcg", out["latest"])

    def test_missing_detail_file_keeps_others(self):
        out = self.make(mutate=lambda f: f.pop("sat/quay_est2.csv")).build()
        self.assertIn("est1", out["latest"]["sat"][0])
        self.assertNotIn("est2", out["latest"]["sat"][0])


class WorkflowTest(Base):
    def test_failed_then_manual_success(self):
        runs = good_runs()
        runs["collect.yml"] = [run("schedule", "failure", ago(hours=3)),
                               run("workflow_dispatch", "success", ago(hours=1)),
                               run("schedule", "success", ago(hours=27))]
        w = wf(self.make(runs=runs).build(), "collect.yml")
        self.assertEqual(w["state"], "warn")
        self.assertEqual(w["consecutive_failures"], 1)
        self.assertIn("手動", w["reasons"][0]["text"])
        self.assertAlmostEqual(w["last_scheduled"]["delay_h"], 5.6, places=1)

    def test_two_failures_alert(self):
        runs = good_runs()
        runs["sat-ships.yml"] = [run("schedule", "failure", ago(hours=3)), run("schedule", "timed_out", ago(hours=27))]
        out = self.make(runs=runs).build()
        self.assertEqual(wf(out, "sat-ships.yml")["state"], "alert")
        self.assertEqual(out["state"], "alert")

    def test_in_progress_is_not_a_result(self):
        runs = good_runs()
        runs["jcg-senkaku.yml"] = [run("schedule", None, ago(minutes=5), status="in_progress"),
                                   run("schedule", "success", ago(hours=24))]
        self.assertEqual(wf(self.make(runs=runs).build(), "jcg-senkaku.yml")["state"], "ok")

    def test_never_and_stale(self):
        runs = good_runs()
        runs["pla-tracker.yml"] = [run("workflow_dispatch", "success", ago(hours=20))]
        runs["collect.yml"] = [run("schedule", "success", ago(hours=40))]
        runs["sat-ships.yml"] = [run("schedule", "success", ago(hours=60))]
        out = self.make(runs=runs).build()
        self.assertEqual(wf(out, "pla-tracker.yml")["state"], "warn")
        self.assertIn("never_scheduled", codes(wf(out, "pla-tracker.yml")))
        self.assertEqual(wf(out, "collect.yml")["state"], "warn")
        self.assertEqual(wf(out, "sat-ships.yml")["state"], "alert")

    def test_rest_error_and_no_token(self):
        runs = good_runs()
        runs["collect.yml"] = urllib.error.HTTPError("u", 403, "forbidden", {}, None)
        out = self.make(runs=runs).build()
        self.assertIn("rest_failed", codes(wf(out, "collect.yml")))
        out = self.make().build(fetch=False)
        self.assertEqual({w["state"] for w in out["workflows"]}, {"unknown"})
        self.assertEqual(out["summary"]["unknown"], len(h.WORKFLOWS))
        self.assertEqual(src(out, "proto")["state"], "ok")    # Actions を見ないときは判定しない

    def test_cron_nominal(self):
        started = datetime(2026, 10, 31, 7, 9, 45, tzinfo=timezone.utc)
        self.assertEqual(h.cron_nominal("23 1 * * *", started), datetime(2026, 10, 31, 1, 23, tzinfo=timezone.utc))
        early = datetime(2026, 10, 31, 0, 30, tzinfo=timezone.utc)
        self.assertEqual(h.cron_nominal("23 1 * * *", early), datetime(2026, 10, 30, 1, 23, tzinfo=timezone.utc))
        self.assertIsNone(h.cron_nominal("0 */6 * * *", started))


class OutputTest(Base):
    def test_annotations(self):
        runs = good_runs()
        runs["sat-ships.yml"] = [run("schedule", "failure", ago(hours=3)), run("schedule", "failure", ago(hours=27))]
        lines = h.annotations(self.make(runs=runs).build())
        self.assertTrue(any(ln.startswith("::error title=health sat-ships.yml") for ln in lines))
        self.assertTrue(lines[-1].startswith("::notice title=health summary::state=alert"))

    def test_main_writes_json(self):
        fx = self.make()
        out = fx.repo / "out" / "health.json"
        orig = (h.DATA, h.ROOT)
        h.DATA, h.ROOT = fx.data, fx.repo
        self.addCleanup(lambda: (setattr(h, "DATA", orig[0]), setattr(h, "ROOT", orig[1])))
        old = h.build

        def fixed(**kw):
            return old(NOW, fx.data, fx.repo, None)
        h.build = fixed
        self.addCleanup(setattr, h, "build", old)
        with redirect_stdout(io.StringIO()) as so, redirect_stderr(io.StringIO()):
            self.assertEqual(h.main(["--out", str(out), "--no-actions", "--annotate"]), 0)
        body = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(body["schema"], h.SCHEMA)
        self.assertEqual(body["generated_utc"], "2026-10-31T10:00:00Z")
        self.assertIn("::notice", so.getvalue())
        self.assertLess(len(out.read_bytes()), 20000)


class JcgRunResultTest(Base):
    def status(self, result, started="2026-10-31T09:00:00Z"):
        def m(f):
            f["jcg-senkaku/status.json"]["last_run"] = {"result": result, "started_utc": started,
                                                        "finished_utc": started, "robots": "x: allowed"}
        return m

    def test_blocked_is_alert(self):
        for result in ("skipped: robots.txt disallows", "skipped: robots.txt could not be checked",
                       "stopped: bot challenge page", "stopped: https://x: HTTP 403"):
            s = src(self.make(mutate=self.status(result)).build(), "jcg-senkaku")
            self.assertEqual(s["state"], "alert", result)
            self.assertIn("source_blocked", codes(s), result)

    def test_not_ok_twice_is_alert(self):
        fx = self.make(mutate=self.status("stopped: URLError timed out"))
        first = fx.build()
        s = src(first, "jcg-senkaku")
        self.assertEqual(s["values"]["not_ok_runs"], 1)
        self.assertIn(("warn", "last_run_not_ok"), {(r["level"], r["code"]) for r in s["reasons"]})
        # 同じ実行を見直しただけなら数を増やさない
        again = src(fx.build(prev=first), "jcg-senkaku")
        self.assertEqual(again["values"]["not_ok_runs"], 1)
        self.assertNotIn("alert", {r["level"] for r in again["reasons"]})
        # 次の実行も ok 以外 → 2 回で ALERT
        fx2 = self.make(mutate=self.status("stopped: URLError timed out", started="2026-11-01T09:00:00Z"))
        s2 = src(fx2.build(prev=first), "jcg-senkaku")
        self.assertEqual(s2["values"]["not_ok_runs"], 2)
        self.assertIn(("alert", "last_run_not_ok"), {(r["level"], r["code"]) for r in s2["reasons"]})

    def test_heartbeat_from_previous_ok(self):
        prev = {"sources": [{"id": "jcg-senkaku", "values": {"last_ok_utc": ago(hours=60), "not_ok_runs": 0,
                                                             "last_run_started_utc": ago(hours=60)}}]}
        s = src(self.make(mutate=self.status("stopped: HTTP 500")).build(prev=prev), "jcg-senkaku")
        self.assertEqual(s["heartbeat_utc"], ago(hours=60))
        self.assertIn("heartbeat_stopped", codes(s))
        self.assertEqual(s["state"], "alert")

    def test_ok_resets(self):
        prev = {"sources": [{"id": "jcg-senkaku", "values": {"not_ok_runs": 3, "last_run_started_utc": "x"}}]}
        s = src(self.make().build(prev=prev), "jcg-senkaku")
        self.assertEqual((s["state"], s["values"]["not_ok_runs"]), ("ok", 0))


class RobotsTextTest(Base):
    def test_pla_robots_texts(self):
        for text, want in (("https://x/robots.txt: HTTP 404, no robots.txt (all allowed)", "ok"),
                           ("https://x: allowed", "ok"),
                           ("https://x: disallowed for this URL", "warn")):
            out = self.make(mutate=lambda f: f["pla-tracker/status.json"]["last_run"].update(robots=text)).build()
            self.assertEqual(src(out, "pla-tracker")["state"], want, text)
        self.assertTrue(h.robots_allowed("https://www.kaiho.mlit.go.jp/robots.txt: HTTP 404, no robots.txt (all allowed)"))
        self.assertFalse(h.robots_allowed("x: disallowed for this URL"))


class BoundaryTest(Base):
    def test_heartbeat_boundaries(self):
        for hours_, want in ((30.0, "ok"), (30.1, "warn"), (54.0, "warn"), (54.1, "alert")):
            out = self.make(mutate=lambda f: f["sat/status.json"].update(updated_utc=ago(hours=hours_))).build()
            self.assertEqual(src(out, "sat")["state"], want, hours_)

    def test_workflow_boundaries(self):
        for hours_, want in ((30.0, "ok"), (54.0, "warn"), (54.1, "alert")):
            runs = good_runs()
            runs["collect.yml"] = [run("schedule", "success", ago(hours=hours_))]
            self.assertEqual(wf(self.make(runs=runs).build(), "collect.yml")["state"], want, hours_)

    def test_jst_day_change(self):
        fx = self.make(mutate=lambda f: f["jcg-senkaku/status.json"].update(last_day="2026-10-29"))
        before = datetime(2026, 10, 31, 14, 59, tzinfo=timezone.utc)   # JST 10-31 23:59 → 今日−2
        after = datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)     # JST 11-01 00:00 → 今日−3
        self.assertNotIn("data_old", codes(src(fx.build(now=before), "jcg-senkaku")))
        self.assertIn("data_old", codes(src(fx.build(now=after), "jcg-senkaku")))

    def test_taiwan_day_change(self):
        fx = self.make(mutate=lambda f: f["pla-tracker/status.json"].update(last_day="2026-10-30"))
        before = datetime(2026, 10, 31, 15, 59, tzinfo=timezone.utc)   # 台湾 10-31 23:59 → 今日−1
        after = datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc)     # 台湾 11-01 00:00 → 今日−2
        self.assertNotIn("data_old", codes(src(fx.build(now=before), "pla-tracker")))
        self.assertIn("data_old", codes(src(fx.build(now=after), "pla-tracker")))

    def test_cancelled_not_counted(self):
        runs = good_runs()
        runs["collect.yml"] = [run("schedule", "cancelled", ago(hours=3)), run("schedule", "success", ago(hours=27))]
        self.assertEqual(wf(self.make(runs=runs).build(), "collect.yml")["state"], "ok")
        runs["collect.yml"] = [run("schedule", "failure", ago(hours=3)), run("schedule", "cancelled", ago(hours=27)),
                               run("schedule", "success", ago(hours=51))]
        w = wf(self.make(runs=runs).build(), "collect.yml")
        self.assertEqual((w["state"], w["consecutive_failures"]), ("warn", 1))


class WriteOnlyOnChangeTest(Base):
    def test_should_write(self):
        fx = self.make()
        a = fx.build()
        b = fx.build(now=NOW + timedelta(hours=1))
        self.assertNotEqual(a["generated_utc"], b["generated_utc"])
        self.assertFalse(h.should_write(a, b))                          # 経過時間だけの違い
        self.assertTrue(h.should_write(None, b))
        c = fx.build(now=NOW + timedelta(hours=h.THRESHOLDS["health_refresh_h"]))
        self.assertTrue(h.should_write(a, c))                           # 一定時間ごとに書き直す（stale と区別）
        changed = copy.deepcopy(b)
        changed["latest"]["jcg"]["contiguous_zone"] = 5
        self.assertTrue(h.should_write(a, changed))
        self.assertEqual(a["stale_after_h"], 12)

    def test_main_skips_unchanged(self):
        fx = self.make()
        out = fx.repo / "out" / "health.json"
        old = h.build
        self.addCleanup(setattr, h, "build", old)
        for hours_ in (0, 1):
            h.build = lambda hours_=hours_, **kw: old(NOW + timedelta(hours=hours_), fx.data, fx.repo, None, kw.get("prev"))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                h.main(["--out", str(out), "--no-actions"])
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["generated_utc"], "2026-10-31T10:00:00Z")


if __name__ == "__main__":
    unittest.main()
