"""収集の健全性（docs/data/health.json）: 起動シーケンスと健全性の監視で共用する小さな集約ファイル。

設計：2部「起動シーケンス（実データの状態表示）と収集の健全性の基準」2026-10-31（§1 棚卸し・§2 鮮度の基準・§4 health.json）。

源ごとに 2 本の時計を分けて判定する（§2.1）:
  心拍（heartbeat）  … 最後に成功した実行の時刻。こちらの収集が動いているか。
  データの新しさ     … 最新の撮影日・公表日。源の側の休み・撮影の欠け・手動待ち。
状態は ok / warn / alert / halt（意図した停止。robots 等）/ info（一回きりの調査など、監視しない）。

あわせて GitHub Actions の REST（GITHUB_TOKEN、読み取り）で各ワークフローの最新の定期実行の結論を見る。
テストの失敗などで止まった実行はデータ側のファイルに何も書かないため、時刻の古さだけでは遅れて気付く（§0.2）。

標準ライブラリだけで動く（収集のワークフローと同じく追加の導入が要らないように）。書き出し先は docs/data/health.json だけ。

    python -m analysis.health                 # docs/data/health.json を作り直す（GITHUB_TOKEN があれば Actions も見る）
    python -m analysis.health --annotate      # あわせて WARN/ALERT を ::warning/::error の注釈で出す
    python -m analysis.health --no-actions    # Actions の REST を使わない
    python -m analysis.health --out PATH      # 書き出し先を変える（確かめる用）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
OUT = DATA / "health.json"
SCHEMA = 1

# ------------------------------------------------------------------ 基準（設定はここ 1 か所）
# 暫定値：2026-10 の実測（GitHub の schedule の遅れ約 6 h、統幕の公表間隔、Sentinel-1 の軌道）から 2部が決めた値（設計 §2）。
# 1 か月運用したら、遅れの実測（Actions の run_started_at − cron）と源の間隔で見直す。見直し時期：2026-12。
THRESHOLDS = {
    "version": "provisional-2026-10",
    "provisional": True,
    "review_by": "2026-12",
    "basis": "2部 設計『起動シーケンスと収集の健全性の基準』§2（2026-10-31）。2026-10 の実測からの暫定値",
    # 毎日の定期実行の心拍（最後に成功してから）。24 h ＋ schedule の遅れ約 6 h（§2.2）
    "heartbeat_h": {"warn": 30, "alert": 54},
    "sat": {
        "scene_days": {"warn": 8, "alert": 14},     # 最新撮影からの日数（UTC の暦日）。≤7 OK、8〜13 WARN、≥14 ALERT
        "pu_warn_fraction": 0.8,                    # 当月 PU（usage.csv）がコード上の上限のこの割合を超えたら WARN
        # 相対軌道ごとの期待される撮影間隔（日）。湛江：ro157 は S1C・S1D で 6 日ごと、ro84 は 12 日ごと（§0.3）
        "orbit_interval_days": {"ZJG": {"157": 6, "84": 12}},
        # 期待日からこの時間を過ぎても行が無ければ「撮影の抜け」：公開遅れ p90 約 9 h ＋次の定期実行（24 h ＋遅れ 6 h）
        "gap_grace_h": 39,
        "gap_match_h": 36,                          # 期待日の前後この時間内の撮影を同じ回とみなす
        "gap_warn_days": 14,                        # 期待日がこの日数以内の抜けは WARN、より古いものは記録だけ
    },
    # 尖閣：last_day（JST の日付）が今日（JST）から何日前か。≥ 今日−2 OK、−3〜−4 WARN、≤ −5 ALERT
    "jcg_lag_days": {"warn": 3, "alert": 5},
    # PLA Activity Tracker：last_day（台湾時間）。≥ 今日−1 OK、−2 WARN、≤ −3 ALERT
    "pla_lag_days": {"warn": 2, "alert": 3},
    # 統幕：最新の公表日から（JST）。≤10 日 OK（公表間隔の p90）、11〜24 WARN、>24 ALERT（p99 超。取り込み忘れの疑い）
    "jso_release_days": {"warn": 11, "alert": 25},
    # 発射：NOTAM Watch 側の notices[].last_seen_utc の最大からの時間。>30 h で WARN（NOTAM Watch が古い）
    "notam_seen_h": {"warn": 30},
    # Actions：定期実行の失敗が続いた回数
    "workflow_failures": {"warn": 1, "alert": 2},
}

JST = timezone(timedelta(hours=9))
TWT = timezone(timedelta(hours=8))

# 監視するワークフロー。cron はワークフローのファイルから読む（二重に持たない）
WORKFLOWS = [
    {"file": "collect.yml", "label": "台湾国防部・発射・統幕の再構築", "event": "schedule",
     "sources": ["taiwan-mnd", "launches", "jso"]},
    {"file": "jcg-senkaku.yml", "label": "海上保安庁 尖閣", "event": "schedule", "sources": ["jcg-senkaku"]},
    {"file": "sat-ships.yml", "label": "Sentinel-1 艦船候補", "event": "schedule", "sources": ["sat"]},
    {"file": "pla-tracker.yml", "label": "PLA Activity Tracker", "event": "schedule", "sources": ["pla-tracker"]},
    {"file": "proto.yml", "label": "分析の試作", "event": "workflow_run", "sources": ["proto"]},
]
FAILED = {"failure", "timed_out", "cancelled", "startup_failure", "action_required", "stale"}
RANK = {"ok": 0, "info": 0, "halt": 0, "warn": 1, "alert": 2}


# ------------------------------------------------------------------ 道具

def parse_utc(s) -> datetime | None:
    """'2026-10-30T11:28:44Z'・'2025-09-08T18:00Z'・'+00:00' 付きを UTC の datetime に。読めなければ None。"""
    if not s or not isinstance(s, str):
        return None
    t = s.strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def iso(d: datetime | None) -> str | None:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if d else None


def hours(now: datetime, d: datetime | None) -> float | None:
    return None if d is None else round((now - d).total_seconds() / 3600, 1)


def day_of(s) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def worst(states) -> str:
    out = "ok"
    for s in states:
        if RANK.get(s, 0) > RANK[out]:
            out = s
    return out


class Unreadable(Exception):
    """ファイルが無い・壊れている。"""


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise Unreadable(f"{rel(path)} が無い") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Unreadable(f"{rel(path)} が壊れている（{type(e).__name__}）") from None


def load_csv(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        raise Unreadable(f"{rel(path)} が無い") from None
    except (csv.Error, UnicodeDecodeError) as e:
        raise Unreadable(f"{rel(path)} が壊れている（{type(e).__name__}）") from None


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def num(x):
    """csv の文字列を数に（空・読めないものは None）。整数なら int。"""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(v) if v.is_integer() else v


class Source:
    """源 1 つ分の判定。reasons に (level, code, text) を足し、state は最悪のもの。"""

    def __init__(self, sid: str, name: str, kind: str, workflow: str | None = None):
        self.d = {"id": sid, "name": name, "kind": kind, "workflow": workflow,
                  "heartbeat_utc": None, "heartbeat_age_h": None,
                  "data_date": None, "data_date_tz": None, "data_age_days": None,
                  "state": "ok", "reasons": [], "values": {}}
        self.levels: list[str] = []

    def add(self, level: str, code: str, text: str):
        self.d["reasons"].append({"level": level, "code": code, "text": text})
        self.levels.append(level)

    def heartbeat(self, now: datetime, hb: datetime | None, what: str):
        self.d["heartbeat_utc"] = iso(hb)
        self.d["heartbeat_age_h"] = hours(now, hb)
        th = THRESHOLDS["heartbeat_h"]
        if hb is None:
            self.add("warn", "heartbeat_unknown", f"{what}の時刻が読めない")
            return
        age = (now - hb).total_seconds() / 3600
        if age > th["alert"]:
            self.add("alert", "heartbeat_stopped", f"{what}から {age:.0f} h（>{th['alert']} h：停止と判断）")
        elif age > th["warn"]:
            self.add("warn", "heartbeat_late", f"{what}から {age:.0f} h（>{th['warn']} h：遅延）")

    def data_date(self, d: date | None, tz: str, today: date):
        self.d["data_date"] = d.isoformat() if d else None
        self.d["data_date_tz"] = tz
        self.d["data_age_days"] = (today - d).days if d else None

    def done(self) -> dict:
        if self.d["kind"] == "halted":
            # 意図した停止。ファイルの読めない等の warn/alert があればそちらを出す
            self.d["state"] = worst(self.levels) if any(RANK.get(x, 0) for x in self.levels) else "halt"
        elif self.d["kind"] == "probe":
            self.d["state"] = worst(self.levels) if any(RANK.get(x, 0) for x in self.levels) else "info"
        else:
            self.d["state"] = worst(self.levels)
        return self.d


def unreadable(src: Source, e: Unreadable):
    # 読めなかった源は WARN（他の源は続ける。設計 §3）
    src.add("warn", "unreadable", str(e))


# ------------------------------------------------------------------ 源ごと

def sat_current_scenes(status: dict, mask_versions: dict) -> dict[str, list[dict]]:
    """現行 mask_version の scenes[]（同じ撮影に v1・v2 の行がある。§1.1 注意）を地点ごとに。"""
    out: dict[str, list[dict]] = {}
    for s in status.get("scenes") or []:
        sid = s.get("site_id")
        want = mask_versions.get(sid)
        if want is not None and s.get("mask_version") != want:
            continue
        if parse_utc(s.get("acquired_utc")) is None:
            continue
        out.setdefault(sid, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s["acquired_utc"])
    return out


def orbit_gaps(scenes: list[dict], intervals: dict, now: datetime, window_start: datetime | None) -> list[dict]:
    """相対軌道ごとの期待間隔に対する撮影の抜け（§2.4）。期待日 ＋ 猶予を過ぎて行が無いものを返す。"""
    th = THRESHOLDS["sat"]
    grace = timedelta(hours=th["gap_grace_h"])
    match = timedelta(hours=th["gap_match_h"])
    gaps = []
    for ro, step_days in sorted(intervals.items(), key=lambda kv: int(kv[0])):
        obs = sorted({parse_utc(s["acquired_utc"]) for s in scenes if str(s.get("relative_orbit")) == str(ro)})
        if not obs:
            gaps.append({"relative_orbit": int(ro), "expected_utc": None, "interval_days": step_days,
                         "note": "監視期間にこの軌道の撮影が 1 つも無い"})
            continue
        step = timedelta(days=step_days)
        anchor = obs[-1]
        start = window_start or obs[0]
        expected = []
        t = anchor
        while t - step >= start - match:
            t -= step
            expected.append(t)
        t = anchor
        while t + step + grace <= now:
            t += step
            expected.append(t)
        for e in sorted(expected):
            if not any(abs(o - e) <= match for o in obs):
                gaps.append({"relative_orbit": int(ro), "expected_utc": iso(e), "interval_days": step_days})
    return gaps


def check_sat(now: datetime, data: Path, repo: Path) -> tuple[dict, dict, dict]:
    """Sentinel-1（sat/）。戻り値：(源, 今日の要点, sat の詳細)。"""
    src = Source("sat", "Sentinel-1 艦船候補", "daily", "sat-ships.yml")
    latest: dict = {}
    detail: dict = {}
    th = THRESHOLDS["sat"]
    try:
        status = load_json(data / "sat" / "status.json")
    except Unreadable as e:
        unreadable(src, e)
        return src.done(), latest, detail
    src.heartbeat(now, parse_utc(status.get("updated_utc")), "最後の処理（updated_utc）")

    # 地点の設定：公開の meta.json（現行の mask_version・publish）と sat/sites.json（daily）
    sites_meta: dict[str, dict] = {}
    try:
        meta = load_json(data / "sat" / "meta.json")
        for s in meta.get("sites") or []:
            if isinstance(s, dict) and s.get("id"):
                sites_meta[s["id"]] = s
        detail["credit"] = (meta.get("attribution") or {}).get("credit")
        detail["land_mask_credit"] = (meta.get("attribution") or {}).get("land_mask")
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    sites_cfg: list[dict] = []
    try:
        sites_cfg = [s for s in (load_json(repo / "sat" / "sites.json").get("sites") or []) if isinstance(s, dict)]
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    daily = [s["id"] for s in sites_cfg if s.get("daily") and s.get("enabled", True)]
    if not daily:
        daily = [sid for sid, s in sites_meta.items() if s.get("daily")]
    detail["sites_daily"] = daily
    detail["sites_configured_not_daily"] = [s["id"] for s in sites_cfg if not s.get("daily")]
    mask_versions = {sid: s.get("mask_version") for sid, s in sites_meta.items() if s.get("mask_version")}
    scenes = sat_current_scenes(status, mask_versions)

    # データの新しさ：毎日取る地点の最新の撮影
    today = now.date()
    newest = None
    detail["latest_scene"] = {}
    detail["gaps"] = []
    all_acq = [parse_utc(s.get("acquired_utc")) for s in status.get("scenes") or []]
    all_acq = [a for a in all_acq if a]
    for sid in daily:
        rows = scenes.get(sid) or []
        if not rows:
            src.add("warn", "no_scene", f"{sid}：現行の範囲（{mask_versions.get(sid)}）の撮影の行が無い")
            continue
        last = rows[-1]
        acq = parse_utc(last["acquired_utc"])
        detail["latest_scene"][sid] = {k: last.get(k) for k in ("acquired_utc", "platform", "relative_orbit",
                                                                 "coverage", "result", "quay_result", "mask_version")}
        days = (today - acq.date()).days
        if days >= th["scene_days"]["alert"]:
            src.add("alert", "scene_old", f"{sid}：最新の撮影 {acq.date()} から {days} 日（≥{th['scene_days']['alert']} 日）")
        elif days >= th["scene_days"]["warn"]:
            src.add("warn", "scene_old", f"{sid}：最新の撮影 {acq.date()} から {days} 日（撮影の抜け）")
        if newest is None or acq > newest:
            newest = acq
        intervals = th["orbit_interval_days"].get(sid)
        if intervals:
            site_acq = [parse_utc(s.get("acquired_utc")) for s in status.get("scenes") or [] if s.get("site_id") == sid]
            site_acq = [a for a in site_acq if a]
            for g in sorted(orbit_gaps(rows, intervals, now, min(site_acq) if site_acq else None),
                            key=lambda g: g.get("expected_utc") or ""):
                g["site_id"] = sid
                e = parse_utc(g.get("expected_utc"))
                recent = e is None or (now - e) <= timedelta(days=th["gap_warn_days"])
                g["level"] = "warn" if recent else "info"
                detail["gaps"].append(g)
                if recent:
                    what = f"ro{g['relative_orbit']} {e.date().strftime('%m-%d')}" if e else f"ro{g['relative_orbit']}"
                    src.add("warn", "orbit_gap", f"{sid}：撮影の抜け {what}（同じ軌道 {g['interval_days']} 日ごとの期待に対して）")
    src.data_date(newest.date() if newest else None, "UTC", today)
    src.d["values"]["latest_acquired_utc"] = iso(newest)

    # 最後の実行：部分失敗・予算
    lr = status.get("last_run") or {}
    errs = 0
    for s in lr.get("sites") or []:
        e = s.get("errors") or 0
        errs += e if isinstance(e, (int, float)) else 0
        left = s.get("budget_left_pu")
        if isinstance(left, (int, float)) and left <= 0:
            src.add("alert", "budget_exhausted", f"{s.get('site_id')}：PU の残り {left}（上限に達した）")
    if errs:
        src.add("warn", "run_errors", f"最後の実行で誤り {errs} 件（部分失敗）")
    if lr.get("site_errors"):
        src.add("warn", "site_errors", f"地点の設定の誤り：{', '.join(map(str, lr['site_errors']))[:200]}")
    src.d["values"].update({"last_run_utc": lr.get("run_utc"), "last_run_errors": errs,
                            "last_run_scenes": lr.get("scenes")})

    # PU：usage.csv（main の実行の記録）と status.json の month_pu_before を両方出し、差を注記する（2部の指摘 (b)）
    month = now.strftime("%Y-%m")
    cap = lr.get("month_pu_cap")
    pu: dict = {"month": month, "code_cap_pu": cap}
    try:
        rows = load_csv(data / "sat" / "usage.csv")
        mine = [r for r in rows if str(r.get("run_utc") or "").startswith(month)]
        used = sum(num(r.get("processing_units")) or 0 for r in mine)
        pu["usage_csv_pu"] = round(used, 3)
        pu["usage_csv_requests"] = int(sum(num(r.get("requests")) or 0 for r in mine))
        pu["usage_csv_runs"] = len(mine)
        last_run = parse_utc(lr.get("run_utc"))
        before = lr.get("month_pu_before")
        if last_run and str(lr.get("run_utc")).startswith(month) and isinstance(before, (int, float)):
            csv_before = sum(num(r.get("processing_units")) or 0 for r in mine
                             if (parse_utc(r.get("run_utc")) or last_run) < last_run)
            pu["status_month_pu_before"] = before
            pu["status_last_run_utc"] = lr.get("run_utc")
            pu["usage_csv_pu_before_last_run"] = round(csv_before, 3)
            diff = round(csv_before - before, 3)
            pu["difference_pu"] = diff
            if abs(diff) > 0.01:
                pu["note"] = (f"usage.csv の当月の行のうち最後の実行（{lr.get('run_utc')}）より前の合計 {csv_before:.1f} PU と、"
                              f"status.json の month_pu_before {before} が {diff:+.1f} PU 合わない（2部の指摘、原因は未確認）。"
                              "使用量の表示には usage.csv の当月合計を使う。どちらも CDSE アカウント全体の残りではない")
        else:
            pu["status_month_pu_before"] = before if last_run and str(lr.get("run_utc")).startswith(month) else None
        if isinstance(cap, (int, float)) and cap > 0 and used > cap * th["pu_warn_fraction"]:
            src.add("warn", "pu_high", f"当月の PU {used:.1f}／上限 {cap:g}（{used / cap:.0%}）")
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    pu["account_note"] = "CDSE アカウント全体の残り（10,000 PU/月）は公開データからは分からない（probe ブランチの使用は usage.csv に入らない）"
    detail["pu"] = pu

    # 今日の要点：湛江などの数（publish が trial なら印を付ける。表示側で隠せるように）
    for sid in daily:
        publish = (sites_meta.get(sid) or {}).get("publish") or next(
            (s.get("publish") for s in sites_cfg if s.get("id") == sid), None)
        mv = mask_versions.get(sid)
        item = {"site_id": sid, "name": (sites_meta.get(sid) or {}).get("name"), "publish": publish,
                "trial": publish != "public", "mask_version": mv}
        try:
            q = [r for r in load_csv(data / "sat" / "quay_counts.csv")
                 if r.get("site_id") == sid and (mv is None or r.get("mask_version") == mv)]
            if q:
                r = q[-1]
                item["est1"] = {"acquired_utc": r.get("acquired_utc"), "version": r.get("est_version"),
                                "mode": num(r.get("est_mode")), "p10": num(r.get("est_p10")), "p90": num(r.get("est_p90")),
                                "warship_mode": num(r.get("est_warship_mode")), "warship_p10": num(r.get("est_warship_p10")),
                                "warship_p90": num(r.get("est_warship_p90")), "credibility": num(r.get("credibility")),
                                "in_sample": r.get("in_sample")}
        except Unreadable as e:
            src.add("warn", "unreadable", str(e))
        try:
            q = [r for r in load_csv(data / "sat" / "quay_est2.csv")
                 if r.get("site_id") == sid and (mv is None or r.get("mask_version") == mv)]
            if q:
                r = q[-1]
                item["est2"] = {"acquired_utc": r.get("acquired_utc"), "version": r.get("est2_version"),
                                "warship_median": num(r.get("warship_median")), "warship_p10": num(r.get("warship_p10")),
                                "warship_p90": num(r.get("warship_p90")), "ships_median": num(r.get("ships_median")),
                                "ships_p10": num(r.get("ships_p10")), "ships_p90": num(r.get("ships_p90")),
                                "in_sample": r.get("in_sample")}
        except Unreadable as e:
            src.add("warn", "unreadable", str(e))
        latest.setdefault("sat", []).append(item)

    detail["changes_latest"] = changes_latest(data / "sat" / "CHANGES.md")
    return src.done(), latest, detail


def changes_latest(path: Path) -> dict | None:
    """CHANGES.md の表の最終行（定義変更の最新）。日付と版の欄だけ。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|", ln)]
    if not rows:
        return None
    cells = [c.strip() for c in rows[-1].strip().strip("|").split("|")]
    what = cells[1] if len(cells) > 1 else ""
    return {"date": cells[0], "version": what[:160] + ("…" if len(what) > 160 else "")}


def check_jcg(now: datetime, data: Path) -> tuple[dict, dict]:
    src = Source("jcg-senkaku", "海上保安庁 尖閣周辺", "daily", "jcg-senkaku.yml")
    latest: dict = {}
    th = THRESHOLDS["jcg_lag_days"]
    try:
        st = load_json(data / "jcg-senkaku" / "status.json")
    except Unreadable as e:
        unreadable(src, e)
        return src.done(), latest
    lr = st.get("last_run") or {}
    if lr.get("result") == "ok":
        src.heartbeat(now, parse_utc(lr.get("finished_utc")), "最後に成功した取得")
    else:
        src.d["heartbeat_utc"] = None
        src.add("warn", "last_run_not_ok", f"最後の実行の結果 {lr.get('result')!r}（{lr.get('reason') or lr.get('error') or '理由の記録なし'}）")
    src.d["values"].update({"robots": lr.get("robots"), "days": st.get("days"), "fetched": lr.get("fetched"),
                            "not_modified": lr.get("not_modified")})
    today = now.astimezone(JST).date()
    last = day_of(st.get("last_day"))
    src.data_date(last, "JST", today)
    if last is None:
        src.add("warn", "data_date_unknown", "last_day が読めない")
    else:
        lag = (today - last).days
        if lag >= th["alert"]:
            src.add("alert", "data_old", f"最新の日 {last}（JST 今日−{lag}）")
        elif lag >= th["warn"]:
            src.add("warn", "data_old", f"最新の日 {last}（JST 今日−{lag}。週末・祝日の可能性）")
    # 最新の隻数は大きな daily.json にしかないので、ここで一度だけ読む（起動側は読まない）
    try:
        d = load_json(data / "jcg-senkaku" / "daily.json")
        fields = d.get("fields") or []
        row = (d.get("rows") or [None])[-1]
        if row is not None and fields:
            r = dict(zip(fields, row))
            latest["jcg"] = {"date": r.get("date"), "date_tz": "JST", "contiguous_zone": r.get("contiguous_zone"),
                             "territorial_sea": r.get("territorial_sea"), "ts_entries": r.get("ts_entries"),
                             "check": r.get("check"), "as_of": r.get("as_of"), "source_url": r.get("source_url"),
                             "note": "null は不明、0 は原文に明記"}
            if r.get("check") != "match":
                src.add("warn", "check_mismatch", f"{r.get('date')} の検算 {r.get('check')!r}（match でない）")
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    return src.done(), latest


def check_pla(now: datetime, data: Path) -> tuple[dict, dict]:
    src = Source("pla-tracker", "PLA Activity Tracker", "daily", "pla-tracker.yml")
    latest: dict = {}
    th = THRESHOLDS["pla_lag_days"]
    try:
        st = load_json(data / "pla-tracker" / "status.json")
    except Unreadable as e:
        unreadable(src, e)
        return src.done(), latest
    src.heartbeat(now, parse_utc(st.get("last_success_utc")), "最後に成功した取得")
    lr = st.get("last_run") or {}
    reason = str(lr.get("reason") or "")
    blocked = re.search(r"robots|disallow|403|challenge|チャレンジ", reason, re.I)
    if lr.get("result") == "skipped":
        if blocked:
            src.add("alert", "source_blocked", f"取得をやめた（収集停止）：{reason[:200]}")
        else:
            src.add("warn", "skipped", f"最後の実行で取得をやめた：{reason[:200]}")
    robots = lr.get("robots")
    if lr.get("result") == "ok" and robots and not str(robots).rstrip().endswith("allowed"):
        src.add("warn", "robots", f"robots：{robots}")
    http = (st.get("source") or {}).get("http")
    if isinstance(http, int) and http == 403:
        src.add("alert", "source_blocked", "取得先が HTTP 403")
    src.d["values"].update({"robots": robots, "days": st.get("days"), "last_run_result": lr.get("result"),
                            "revised_count": lr.get("revised_count")})
    today = now.astimezone(TWT).date()
    last = day_of(st.get("last_day"))
    src.data_date(last, "TWT", today)
    if last is None:
        src.add("warn", "data_date_unknown", "last_day が読めない")
    else:
        lag = (today - last).days
        if lag >= th["alert"]:
            src.add("alert", "data_old", f"最新の日 {last}（台湾 今日−{lag}）")
        elif lag >= th["warn"]:
            src.add("warn", "data_old", f"最新の日 {last}（台湾 今日−{lag}。元の欠けは年に数日ある）")
    try:
        d = load_json(data / "pla-tracker" / "daily.json")
        fields = d.get("fields") or []
        row = (d.get("rows") or [None])[-1]
        if row is not None and fields:
            r = dict(zip(fields, row))
            latest["taiwan"] = {"date": r.get("date"), "date_tz": "TWT", "aircraft_total": r.get("aircraft_total"),
                                "median_line_cross": r.get("median_line_cross"), "ships_total": r.get("ships_total"),
                                "source": "PLA Activity Tracker（元：台湾国防部）"}
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    return src.done(), latest


def check_mnd(now: datetime, data: Path) -> dict:
    try:
        st = load_json(data / "taiwan-mnd" / "status.json")
    except Unreadable as e:
        src = Source("taiwan-mnd", "台湾国防部", "daily", "collect.yml")
        unreadable(src, e)
        return src.done()
    collection = st.get("collection")
    halted = collection is not None and collection != "active"
    src = Source("taiwan-mnd", "台湾国防部", "halted" if halted else "daily", "collect.yml")
    src.data_date(day_of(st.get("last_day")), "TWT", now.astimezone(TWT).date())
    src.d["values"].update({"collection": collection, "reports": st.get("reports"),
                            "unparsed": len(st.get("unparsed_ids") or [])})
    if halted:
        # 停止中は status を書き直さないので、心拍は判定しない（記録だけ）
        src.d["heartbeat_utc"] = st.get("generated_utc")
        src.add("halt", "collection_halted", f"収集停止：{collection}")
    else:
        src.heartbeat(now, parse_utc(st.get("generated_utc")), "最後の取得")
    return src.done()


def check_jso(now: datetime, data: Path) -> dict:
    src = Source("jso", "統合幕僚監部（統幕）", "manual", "collect.yml")
    th = THRESHOLDS["jso_release_days"]
    try:
        st = load_json(data / "jso" / "status.json")
    except Unreadable as e:
        unreadable(src, e)
        return src.done()
    # 心拍は再構築（collect.yml 毎日）。データの新しさは公表日（手動取り込み）
    src.heartbeat(now, parse_utc(st.get("built_utc")), "最後の再構築（built_utc）")
    today = now.astimezone(JST).date()
    last = day_of(st.get("last_release"))
    src.data_date(last, "JST", today)
    src.d["values"].update({"passages": st.get("passages"), "releases": st.get("releases"),
                            "intake": "手動取り込み（mod.go.jp は自動取得できない）"})
    if last is None:
        src.add("warn", "data_date_unknown", "last_release が読めない")
    else:
        days = (today - last).days
        if days >= th["alert"]:
            src.add("alert", "release_old", f"最新の公表 {last} から {days} 日（公表間隔の p99 超：取り込み忘れの疑い）")
        elif days >= th["warn"]:
            src.add("warn", "release_old", f"最新の公表 {last} から {days} 日（手動取り込み待ちの可能性）")
    return src.done()


def check_launches(now: datetime, data: Path) -> tuple[dict, dict]:
    src = Source("launches", "発射の予告（NOTAM Watch・LL2）", "daily", "collect.yml")
    latest: dict = {}
    try:
        ev = load_json(data / "launches" / "events.json")
    except Unreadable as e:
        unreadable(src, e)
        return src.done(), latest
    src.heartbeat(now, parse_utc(ev.get("generated_utc")), "最後の作成（generated_utc）")
    windows = []
    for e in ev.get("events") or []:
        end = parse_utc(e.get("window_end_utc"))
        if end and end >= now:
            windows.append(e)
    windows.sort(key=lambda e: parse_utc(e.get("window_start_utc")) or now)
    upcoming = [e for e in windows if (parse_utc(e.get("window_start_utc")) or now) >= now]
    nxt = upcoming[0] if upcoming else None
    latest["launches"] = {"open_or_upcoming_windows": len(windows), "open_now": len(windows) - len(upcoming),
                          "next_start": ({k: nxt.get(k) for k in ("site", "country", "window_start_utc", "window_end_utc")}
                                         if nxt else None)}
    try:
        no = load_json(data / "launches" / "notices.json")
        notices = list((no.get("notices") or {}).values())
        seen = [parse_utc(n.get("last_seen_utc")) for n in notices if isinstance(n, dict)]
        seen = [s for s in seen if s]
        last_seen = max(seen) if seen else None
        active = [n for n in notices if isinstance(n, dict)
                  and (n.get("end_utc") is None or (parse_utc(n.get("end_utc")) or now) >= now)]
        latest["launches"]["active_notices"] = len(active)
        src.d["values"].update({"notices_last_retrieved_utc": no.get("last_retrieved_utc"),
                                "notam_watch_last_seen_utc": iso(last_seen)})
        src.data_date(last_seen.date() if last_seen else None, "UTC", now.date())
        lim = THRESHOLDS["notam_seen_h"]["warn"]
        if last_seen is None:
            src.add("warn", "notam_unknown", "NOTAM Watch 由来の告知の last_seen_utc が無い")
        elif (now - last_seen).total_seconds() / 3600 > lim:
            src.add("warn", "notam_watch_old",
                    f"NOTAM Watch 由来の最終確認 {iso(last_seen)}（{hours(now, last_seen):.0f} h 前、>{lim} h：NOTAM Watch 側が古い）")
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    try:
        src.d["values"]["ll2_last_retrieved_utc"] = load_json(data / "launches" / "ll2.json").get("last_retrieved_utc")
    except Unreadable as e:
        src.add("warn", "unreadable", str(e))
    return src.done(), latest


def check_proto(now: datetime, data: Path, workflows: list[dict], inputs_hb: list[datetime]) -> dict:
    """分析の試作。各 JSON は中身が変わらなければ書き直さない（analysis/proto.py）ので computed_utc は心拍に使えない。
    心拍は proto.yml の最後の成功（Actions REST）。入力（収集の心拍）より古ければ WARN。"""
    src = Source("proto", "分析の試作（proto）", "derived", "proto.yml")
    comp = []
    proto_dir = data.parent / "proto" / "data"
    for p in sorted(proto_dir.glob("*.json")) if proto_dir.is_dir() else []:
        try:
            comp.append(parse_utc(load_json(p).get("computed_utc")))
        except (Unreadable, AttributeError):
            src.add("warn", "unreadable", f"{rel(p)} が読めない")
    comp = [c for c in comp if c]
    src.d["values"]["files"] = len(comp)
    src.d["values"]["computed_utc_max"] = iso(max(comp)) if comp else None
    wf = next((w for w in workflows if w.get("file") == "proto.yml"), None)
    ok_at = parse_utc(wf.get("last_success_started_utc")) if wf else None
    newest_input = max(inputs_hb) if inputs_hb else None
    src.d["values"]["newest_input_utc"] = iso(newest_input)
    if wf is None or wf.get("state") == "unknown":
        src.d["heartbeat_utc"] = iso(max(comp)) if comp else None
        src.d["values"]["note"] = "Actions の記録が無いので判定しない（computed_utc は中身が変わったときだけ更新）"
        return src.done()
    src.d["heartbeat_utc"] = iso(ok_at)
    src.d["heartbeat_age_h"] = hours(now, ok_at)
    if ok_at is None:
        src.add("warn", "never_succeeded", "proto.yml の成功の記録が無い")
    elif newest_input and ok_at < newest_input:
        src.add("warn", "older_than_input", f"最後の成功 {iso(ok_at)} が入力の更新 {iso(newest_input)} より前")
    return src.done()


def check_ocean(now: datetime, data: Path) -> dict:
    src = Source("ocean", "Sentinel-1 カタログの公開遅れ調査", "probe", None)
    try:
        c = load_json(data / "ocean" / "coverage_probe.json")
    except Unreadable as e:
        src.add("info", "unreadable", str(e))
        return src.done()
    g = parse_utc(c.get("generated_utc"))
    src.d["heartbeat_utc"] = iso(g)
    src.d["values"]["note"] = "一回きりの調査。鮮度は監視しない"
    return src.done()


# ------------------------------------------------------------------ Actions（REST）

def workflow_cron(repo: Path, file: str) -> str | None:
    try:
        text = (repo / ".github" / "workflows" / file).read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    m = re.search(r"cron:\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else None


def cron_nominal(cron: str | None, started: datetime | None) -> datetime | None:
    """毎日の cron（'M H * * *'）について、started の直前の予定時刻。それ以外の形は None。"""
    if not cron or not started:
        return None
    parts = cron.split()
    if len(parts) != 5 or parts[2:] != ["*", "*", "*"] or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    t = started.replace(hour=int(parts[1]), minute=int(parts[0]), second=0, microsecond=0)
    return t if t <= started else t - timedelta(days=1)


def github_fetcher(token: str | None, repo_slug: str, api: str):
    """ワークフローのファイル名 → 最近の実行（main）。REST だけ（GraphQL は使わない）。"""
    def fetch(file: str) -> list[dict]:
        url = f"{api}/repos/{repo_slug}/actions/workflows/{file}/runs?branch=main&per_page=30"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "X-GitHub-Api-Version": "2022-11-28",
                                                   "User-Agent": "observatory-data-health"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        return body.get("workflow_runs") or []
    return fetch


def check_workflow(now: datetime, wf: dict, runs: list[dict], cron: str | None) -> dict:
    th = THRESHOLDS["heartbeat_h"]
    fth = THRESHOLDS["workflow_failures"]
    out = {"file": wf["file"], "label": wf["label"], "event": wf["event"], "cron_utc": cron,
           "sources": wf["sources"], "state": "ok", "reasons": []}
    levels = []

    def add(level, code, text):
        out["reasons"].append({"level": level, "code": code, "text": text})
        levels.append(level)

    runs = sorted((r for r in runs if isinstance(r, dict)), key=lambda r: r.get("created_at") or "", reverse=True)
    sched = [r for r in runs if r.get("event") == wf["event"]]
    done = [r for r in sched if r.get("status") == "completed" and r.get("conclusion") not in (None, "skipped", "neutral")]
    success_any = next((r for r in runs if r.get("conclusion") == "success"), None)
    out["last_success_started_utc"] = iso(parse_utc((success_any or {}).get("run_started_at")))
    out["last_success_event"] = (success_any or {}).get("event")
    last = sched[0] if sched else None
    if last:
        started = parse_utc(last.get("run_started_at")) or parse_utc(last.get("created_at"))
        out["last_scheduled"] = {"id": last.get("id"), "started_utc": iso(started), "status": last.get("status"),
                                 "conclusion": last.get("conclusion"), "html_url": last.get("html_url")}
        nominal = cron_nominal(cron, started) if wf["event"] == "schedule" else None
        if nominal:
            out["last_scheduled"]["delay_h"] = hours(started, nominal)
        age = hours(now, started)
        if age is not None and age > th["alert"]:
            add("alert", "not_started", f"{wf['event']} の実行が {age:.0f} h 起動していない（>{th['alert']} h）")
        elif age is not None and age > th["warn"]:
            add("warn", "not_started", f"{wf['event']} の実行が {age:.0f} h 起動していない（>{th['warn']} h）")
    else:
        out["last_scheduled"] = None
        add("warn", "never_scheduled", f"最近の記録に {wf['event']} の実行が無い（未実行）")
    fails = 0
    for r in done:
        if r.get("conclusion") in FAILED:
            fails += 1
        else:
            break
    out["consecutive_failures"] = fails
    if fails:
        r = done[0]
        when = iso(parse_utc(r.get("run_started_at")))
        recovered = success_any and (success_any.get("created_at") or "") > (r.get("created_at") or "")
        tail = "（その後の手動等の実行で成功）" if recovered else ""
        level = "alert" if fails >= fth["alert"] else "warn"
        add(level, "failed", f"{wf['event']} の実行 {when} が {r.get('conclusion')}"
                             + (f"（{fails} 回続けて）" if fails > 1 else "") + tail)
    out["state"] = worst(levels)
    return out


def check_workflows(now: datetime, repo: Path, fetch) -> list[dict]:
    out = []
    for wf in WORKFLOWS:
        cron = workflow_cron(repo, wf["file"])
        if fetch is None:
            out.append({"file": wf["file"], "label": wf["label"], "event": wf["event"], "cron_utc": cron,
                        "sources": wf["sources"], "state": "unknown",
                        "reasons": [{"level": "info", "code": "not_checked", "text": "Actions の REST を使わずに作った"}]})
            continue
        try:
            runs = fetch(wf["file"])
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            out.append({"file": wf["file"], "label": wf["label"], "event": wf["event"], "cron_utc": cron,
                        "sources": wf["sources"], "state": "warn",
                        "reasons": [{"level": "warn", "code": "rest_failed",
                                     "text": f"Actions の実行記録を取得できない（{type(e).__name__}: {str(e)[:120]}）"}]})
            continue
        out.append(check_workflow(now, wf, runs, cron))
    return out


# ------------------------------------------------------------------ 全体

def credits(data: Path, sat_detail: dict) -> list[str]:
    out = []
    if sat_detail.get("credit"):
        out.append(sat_detail["credit"])
    for path, keys in (("jcg-senkaku/meta.json", ("credit", "license")),
                       ("pla-tracker/meta.json", ("credit_ja", "license"))):
        try:
            a = load_json(data / path).get("attribution") or {}
            out.append("／".join(str(a[k]) for k in keys if a.get(k)))
        except Unreadable:
            pass
    if sat_detail.get("land_mask_credit"):
        out.append(sat_detail["land_mask_credit"])
    return [c for c in out if c]


def build(now: datetime | None = None, data: Path = DATA, repo: Path = ROOT, fetch=None) -> dict:
    now = now or datetime.now(timezone.utc)
    workflows = check_workflows(now, repo, fetch)
    sat, latest_sat, sat_detail = check_sat(now, data, repo)
    jcg, latest_jcg = check_jcg(now, data)
    pla, latest_pla = check_pla(now, data)
    mnd = check_mnd(now, data)
    jso = check_jso(now, data)
    launches, latest_launch = check_launches(now, data)
    inputs = [parse_utc(s.get("heartbeat_utc")) for s in (jcg, jso, launches)]
    proto = check_proto(now, data, workflows, [i for i in inputs if i])
    ocean = check_ocean(now, data)
    sources = [sat, jcg, pla, mnd, jso, launches, proto, ocean]
    states = [s["state"] for s in sources] + [w["state"] for w in workflows]
    summary = {k: states.count(k) for k in ("ok", "warn", "alert", "halt", "info", "unknown")}
    latest = {**latest_sat, **latest_jcg, **latest_pla, **latest_launch}
    sat_block = {k: sat_detail.get(k) for k in ("pu", "gaps", "latest_scene", "sites_daily",
                                                  "sites_configured_not_daily", "changes_latest")}
    return {
        "schema": SCHEMA,
        "generated_utc": iso(now),
        "state": worst(states),
        "summary": summary,
        "states_note": "ok＝基準内、warn＝遅延・抜け・不一致、alert＝停止と判断、halt＝意図した停止（robots 等）、info＝監視しない、unknown＝確かめていない",
        "thresholds": THRESHOLDS,
        "sources": sources,
        "latest": latest,
        "latest_note": "latest.sat[] の trial＝true は試行（publish: trial）。公開画面で数を出すかは表示側で決める",
        "workflows": workflows,
        "sat": sat_block,
        "credits": credits(data, sat_detail),
    }


def annotations(h: dict) -> list[str]:
    def esc(s: str) -> str:
        return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    lines = []
    for kind, items, key in (("source", h["sources"], "id"), ("workflow", h["workflows"], "file")):
        for it in items:
            for r in it.get("reasons") or []:
                cmd = {"alert": "error", "warn": "warning"}.get(r["level"])
                if cmd:
                    lines.append(f"::{cmd} title=health {it[key]} ({r['level'].upper()})::{esc(r['text'])}")
    s = h["summary"]
    lines.append(f"::notice title=health summary::state={h['state']} OK {s['ok']}・WARN {s['warn']}・ALERT {s['alert']}"
                 f"・HALT {s['halt']}・INFO {s['info']}・未確認 {s['unknown']}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--no-actions", action="store_true")
    a = ap.parse_args(argv)
    fetch = None
    if not a.no_actions:
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            fetch = github_fetcher(token, os.environ.get("GITHUB_REPOSITORY", "2wrr6c7mmj-jpg/observatory-data"),
                                   os.environ.get("GITHUB_API_URL", "https://api.github.com"))
        else:
            print("GITHUB_TOKEN が無いので Actions の実行記録は見ない", file=sys.stderr)
    h = build(fetch=fetch)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(h, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, a.out)
    if a.annotate:
        print("\n".join(annotations(h)))
    s = h["summary"]
    print(f"health: {h['state']} OK {s['ok']} WARN {s['warn']} ALERT {s['alert']} HALT {s['halt']} → {rel(a.out)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
