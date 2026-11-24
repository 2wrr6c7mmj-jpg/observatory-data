"""Recurring exercises expected within the next month (the 演習 screen's 見込み), and the record of how those calls turned out.

Owner, 2026-11-23: the exercise screen shows what is under way and what is expected within a month; the expectation comes
from the history of when and where each exercise was held. This file does the recurring, named exercises:

- The history: the site's exercise table (observatory-watch public/data/exercises.json, 1,671 rows from 2006, fetched from
  the published site), the hand-checked ledger (exercises/ledger.json) and the exercises collected here (events/*.json).
- Names are normalised by SERIES (Joint Sea = 海上聯合 = Морское взаимодействие …). One occurrence per series and year:
  the earliest dated start that year.
- A series is called only with MIN_YEARS years on record and a regular period (every year, or every second year).
- The window is the spread of the latest RECENT start days (25th to 75th percentile, widened by a week each side, at least
  three weeks; a series wider than MAX_WINDOW days is not called), placed in the next year the period gives. It is published when it meets the next FORECAST_DAYS days and the series has
  no start recorded in that window yet.
- The basis line gives the denominator: "records in N of the M years since the first one".
- Every call is kept (forecast-log.json) as issued, never edited. When its window has passed it is marked hit (a start
  of the series inside the window) or miss, and a Brier score is kept per month. The probability is the share of the
  years since the first record that had the exercise in its period.

Records are what was reported, not what was held: the table has far more Japanese and US records than Chinese, Russian
or North Korean ones (2026-11-24: Japan 1,053, US 642, China 267, Russia 226, North Korea 0). A missing year can be a
missing record. The basis line says how many years were counted.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://observatory-watch.pages.dev/data/exercises.json"
MIN_YEARS = 5
FORECAST_DAYS = 31
RECENT = 8          # the window comes from the latest eight occurrences
MAX_WINDOW = 60     # days; a wider spread is no season to call

# key, Japanese name, pattern over name / series / note, countries, usual area
SERIES: list[tuple[str, str, str, str, str]] = [
    ("joint-sea", "海上聯合（中露）", r"Joint Sea|海上联合|海上聯合|海上連合|Морское взаимодействие", "中国・ロシア", "日本海・東シナ海ほか"),
    ("northern-interaction", "北部・聯合（中露）", r"北部[・·]?联合|北部[・·]?連合|Northern[/ -]Interaction|Север[/ -]?Взаимодействие", "中国・ロシア", "日本海・オホーツク海"),
    ("keen-sword", "キーン・ソード（日米）", r"Keen Sword|キーン・?ソード", "日本・米国", "日本周辺"),
    ("keen-edge", "キーン・エッジ（日米）", r"Keen Edge|キーン・?エッジ", "日本・米国", "日本周辺（指揮所演習）"),
    ("yama-sakura", "ヤマサクラ（日米）", r"Yama ?Sakura|ヤマサクラ", "日本・米国", "日本（指揮所演習）"),
    ("orient-shield", "オリエント・シールド（日米）", r"Orient Shield|オリエント・?シールド", "日本・米国", "日本の演習場"),
    ("resolute-dragon", "レゾリュート・ドラゴン（日米）", r"Resolute Dragon|レゾリュート・?ドラゴン", "日本・米国", "南西諸島ほか"),
    ("freedom-shield", "フリーダム・シールド（米韓）", r"Freedom Shield|フリーダム・?シールド|自由の盾|Key Resolve", "米国・韓国", "韓国"),
    ("ulchi", "乙支フリーダム・シールド（米韓）", r"Ulchi|乙支", "米国・韓国", "韓国"),
    ("han-kuang", "漢光（台湾）", r"漢光|汉光|Han Kuang", "台湾", "台湾"),
    ("balikatan", "バリカタン（米比）", r"Balikatan|バリカタン", "米国・フィリピン", "フィリピン"),
    ("kamandag", "カマンダグ（米比）", r"KAMANDAG|カマンダグ", "米国・フィリピン", "フィリピン"),
    ("talisman-sabre", "タリスマン・セイバー（米豪）", r"Talisman Sabre|タリスマン", "米国・オーストラリアほか", "オーストラリア"),
    ("rimpac", "リムパック", r"RIMPAC|リムパック|環太平洋合同演習", "米国ほか多国間", "ハワイ周辺"),
    ("malabar", "マラバール（日米豪印）", r"Malabar|マラバール", "日本・米国・オーストラリア・インド", "インド太平洋"),
    ("pacific-vanguard", "パシフィック・ヴァンガード", r"Pacific Vanguard|パシフィック・?ヴァンガード", "日本・米国・韓国・オーストラリア", "西太平洋"),
    ("valiant-shield", "ヴァリアント・シールド（米）", r"Valiant Shield|ヴァリアント・?シールド|バリアント・?シールド", "米国ほか", "グアム・マリアナ"),
    ("cope-north", "コープ・ノース", r"Cope North|コープ・?ノース", "米国・日本・オーストラリア", "グアム"),
    ("peace-mission", "平和使命（上海協力機構）", r"Peace Mission|和平使命|平和の使命", "中国・ロシアほか", "中央アジア・ロシア・中国"),
    ("vostok", "ボストーク（ロシア）", r"\bVostok\b|ボストーク|Восток-", "ロシア", "ロシア東部"),
    ("ilex", "日米共同訓練（ILEX）", r"\bILEX\b", "日本・米国", "日本周辺"),
    ("sea-dragon", "シー・ドラゴン（対潜）", r"Sea Dragon|シー・?ドラゴン", "米国ほか多国間", "グアム周辺"),
    ("pacific-dragon", "パシフィック・ドラゴン（弾道ミサイル防衛）", r"Pacific Dragon|パシフィック・?ドラゴン", "米国・日本・韓国ほか", "ハワイ周辺"),
]
COMPILED = [(k, n, re.compile(p, re.I), c, a) for k, n, p, c, a in SERIES]


def series_of(text: str) -> str | None:
    for k, _, rx, _, _ in COMPILED:
        if rx.search(text or ""):
            return k
    return None


def _d(s) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def occurrences(rows: list[dict]) -> dict[str, dict[int, date]]:
    """series -> year -> the earliest dated start that year."""
    out: dict[str, dict[int, date]] = {}
    for r in rows:
        start = _d(r.get("start"))
        if start is None:
            continue
        text = " ".join(str(r.get(f) or "") for f in ("name", "nameEn", "series", "note"))
        k = series_of(text)
        if not k:
            continue
        y = out.setdefault(k, {})
        if start.year not in y or start < y[start.year]:
            y[start.year] = start
    return out


def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def period_of(years: list[int]) -> int | None:
    """1 (every year) or 2 (every second year) from the gaps between recorded years; None when irregular."""
    gaps = [b - a for a, b in zip(years, years[1:])]
    if not gaps:
        return None
    if sum(1 for g in gaps if g == 1) >= len(gaps) * 0.6:
        return 1
    if sum(1 for g in gaps if g % 2 == 0) >= len(gaps) * 0.8:
        return 2
    return None


def _q(vals: list[int], f: float) -> float:
    v = sorted(vals)
    i = (len(v) - 1) * f
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def forecast(occ: dict[str, dict[int, date]], today: date) -> list[dict]:
    out = []
    names = {k: (n, c, a) for k, n, _, c, a in COMPILED}
    for k, years_map in occ.items():
        years = sorted(years_map)
        if len(years) < MIN_YEARS:
            continue
        per = period_of(years)
        if per is None:
            continue
        doys = [_doy(years_map[y]) for y in years[-RECENT:]]     # the recent habit: dates drift over the years
        lo, hi = int(_q(doys, 0.25)) - 7, int(_q(doys, 0.75)) + 7
        if hi - lo < 21:
            mid = (hi + lo) // 2
            lo, hi = mid - 10, mid + 11
        if hi - lo > MAX_WINDOW:
            continue        # no settled season (マラバール 2015-2025: June to November): not called
        last = years[-1]
        for y in (today.year, today.year + 1):
            if per == 2 and (y - last) % 2:
                continue
            start = date(y, 1, 1) + timedelta(days=max(0, lo - 1))
            end = date(y, 1, 1) + timedelta(days=min(364, hi - 1))
            if end < today:
                continue
            if start > today + timedelta(days=FORECAST_DAYS):
                break
            if y in years_map and start <= years_map[y] <= end:
                break          # already recorded this time
            span = (today.year - years[0]) // per + 1
            name, countries, area = names[k]
            p = round(min(1.0, len(years) / span), 2)
            out.append({
                "series": k, "name": name, "countries": countries, "area": area, "year": y,
                "window_start": start.isoformat(), "window_end": end.isoformat(), "period_years": per,
                "probability": p, "years_recorded": len(years), "years_counted": span, "last_year": last,
                "basis": f"{years[0]}年以降の{'毎年' if per == 1 else '隔年'}の機会{span}回のうち{len(years)}回の記録。"
                         f"開始は例年{start.month}月{start.day}日〜{end.month}月{end.day}日、最近は{last}年",
            })
            break
    return sorted(out, key=lambda x: x["window_start"])


def score(log: list[dict], occ: dict[str, dict[int, date]], today: date) -> list[dict]:
    """Marks calls whose window has passed: hit when the series started inside the window that year."""
    for f in log:
        if f.get("outcome") is not None or _d(f["window_end"]) >= today:
            continue
        s = occ.get(f["series"], {}).get(f["year"])
        f["outcome"] = 1 if s and _d(f["window_start"]) <= s <= _d(f["window_end"]) else 0
        f["scored"] = today.isoformat()
    return log


def brier(log: list[dict]) -> dict:
    by: dict[str, list[float]] = {}
    for f in log:
        if f.get("outcome") is not None:
            by.setdefault(f["window_end"][:7], []).append((f["probability"] - f["outcome"]) ** 2)
    return {m: {"n": len(v), "brier": round(statistics.fmean(v), 3)} for m, v in sorted(by.items())}


def load_rows(out: Path, archive: list[dict]) -> list[dict]:
    rows = list(archive)
    ledger = json.loads((ROOT / "exercises" / "ledger.json").read_text())
    rows += [dict(e, series=e.get("name")) for e in ledger.get("events", [])]
    for f in sorted((out / "events").glob("*.json")) if (out / "events").exists() else []:
        rows += [dict(e, series=e.get("name")) for e in json.loads(f.read_text()).get("events", [])]
    return rows


def fetch_archive() -> list[dict]:
    from collectors import polite
    r = polite.get(ARCHIVE_URL, "observatory-data/1", timeout=60)
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}")
    return json.loads(r.body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "exercises"))
    ap.add_argument("--archive", help="a local copy of the site's exercises.json (tests, offline)")
    ap.add_argument("--today")
    a = ap.parse_args(argv)
    out = Path(a.out)
    today = date.fromisoformat(a.today) if a.today else datetime.now(timezone.utc).date()
    try:
        archive = json.loads(Path(a.archive).read_text()) if a.archive else fetch_archive()
    except Exception as e:     # the forecast is kept as it was; the rest of the run goes on
        print(f"::warning title=exercises forecast::archive not read ({e}); forecast unchanged")
        return 0
    occ = occurrences(load_rows(out, archive))
    calls = forecast(occ, today)
    log_path = out / "forecast-log.json"
    log = json.loads(log_path.read_text()).get("calls", []) if log_path.exists() else []
    seen = {(f["series"], f["year"]) for f in log}
    for c in calls:
        if (c["series"], c["year"]) not in seen:
            log.append({k: c[k] for k in ("series", "name", "year", "window_start", "window_end", "probability")} | {"issued": today.isoformat(), "outcome": None})
    log = score(log, occ, today)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forecast.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "today": today.isoformat(),
        "horizon_days": FORECAST_DAYS, "min_years": MIN_YEARS,
        "method": "演習の表・台帳・収集した演習から、名前を揃えた定例の演習ごとに、年ごとの最初の開始日を集める。"
                  f"{MIN_YEARS}年以上の記録があり、毎年か隔年の周期のものだけ。例年の開始の幅（25〜75%、前後1週）を次の機会に当て、"
                  f"今後{FORECAST_DAYS}日に入るものを出す。記録は報道されたもので、記録の無い年は実施が無かったとは限らない",
        "calls": calls, "score": brier(log),
    }, ensure_ascii=False, indent=1) + "\n")
    log_path.write_text(json.dumps({"note": "出した見込みを出したときのまま残す。期間が過ぎたら outcome（1=期間内に開始の記録、0=無し）", "calls": log}, ensure_ascii=False, indent=1) + "\n")
    print(f"::notice title=exercises forecast::{len(calls)} calls within {FORECAST_DAYS} days; log {len(log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
