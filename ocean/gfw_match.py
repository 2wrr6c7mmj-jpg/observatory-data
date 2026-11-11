"""Mark the open-sea candidates with what Global Fishing Watch (GFW) says about the same Sentinel-1 pass.

GFW detects vessels in the same Sentinel-1 scenes and matches each detection with AIS. Its 4Wings report gives
the detections on a 0.01 degree grid per hour, split by "matched with AIS" or not (about five days after the
pass). For every pass of the open-sea monitor (docs/data/ocean/monitor/candidates/) old enough to be in GFW,
each of our candidates gets one of:

    ais      a GFW detection matched with AIS lies within RADIUS_KM (a vessel that was transmitting: merchant
             ships, fishing vessels and the like)
    no_ais   a GFW detection not matched with AIS lies within RADIUS_KM (GFW saw a vessel there too, and no
             AIS track fits it)
    none     GFW has no detection there (a false alarm of ours, a fixed object, or a vessel GFW missed)

Only this mark, the candidate's position and class, and counts are written (docs/data/ocean/gfw/). GFW's own
rows are not stored or published, and vessel-level queries are not used. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "gfw-match-1"
URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-sar-presence:latest"
ATTRIBUTION = "Powered by Global Fishing Watch. https://globalfishingwatch.org"
SOURCE = ("AIS との一致：Global Fishing Watch の Sentinel-1 の船の検出（public-global-sar-presence、CC BY-NC 4.0）。"
          "候補：Contains modified Copernicus Sentinel data")
RADIUS_KM = 0.8        # a 0.01 degree cell is about 1.1 km x 1.0 km; our position against the cell centre
LAG_DAYS = 5           # GFW serves detections up to five days ago
RECHECK_DAYS = 4       # a pass is matched again each day for this long after it first becomes due (late arrivals)
GIVE_UP_DAYS = 21      # a pass GFW still has nothing for is left as it is after this
PAUSE_S = 1.0          # one report at a time per token


class GfwError(Exception):
    """The report did not come back as usable JSON (status and a short reason; never the token)."""


def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def region(bbox) -> dict:
    w, s, e, n = bbox
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}


def hours_of(acquired: datetime) -> set[str]:
    """The hour labels of GFW ("2026-10-31 10:00") a pass can fall under: its own, and the next or the last one near the hour."""
    return {(acquired + timedelta(minutes=m)).strftime("%Y-%m-%d %H:00") for m in (-10, 0, 10)}


def date_range(acquired: datetime) -> str:
    a, b = acquired - timedelta(minutes=10), acquired + timedelta(minutes=10)
    return f"{a:%Y-%m-%d},{b + timedelta(days=1):%Y-%m-%d}"


def rows_of(doc) -> list[dict]:
    """The rows of a report: entries is a list of {dataset: [rows]} (a dataset without data gives null)."""
    out: list[dict] = []
    for e in doc.get("entries", []) if isinstance(doc, dict) else []:
        if not isinstance(e, dict):
            continue
        for v in e.values():
            if isinstance(v, list):
                out += [r for r in v if isinstance(r, dict)]
    return out


def http_report(token: str, bbox, days: str, matched: bool, timeout: float = 150) -> dict:
    q = {"spatial-resolution": "HIGH", "temporal-resolution": "HOURLY", "group-by": "FLAG", "datasets[0]": DATASET,
         "date-range": days, "format": "JSON", "filters[0]": f"matched='{'true' if matched else 'false'}'"}
    req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(q), data=json.dumps({"geojson": region(bbox)}).encode(),
                                 method="POST", headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                                                         "User-Agent": "observatory-data"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("messages", [{}])[0].get("detail", "")
        except Exception:  # noqa: BLE001 - the body is only for the message
            detail = ""
        raise GfwError(f"HTTP {e.code} {str(detail)[:160]}".strip()) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise GfwError(type(e).__name__) from None


def cells(doc, hours: set[str]) -> list[tuple[float, float]]:
    """(lat, lon) of the cells with a detection in the pass's hour."""
    out = []
    for r in rows_of(doc):
        try:
            if str(r.get("date")) in hours and float(r.get("detections") or 0) > 0:
                out.append((float(r["lat"]), float(r["lon"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def km(lat1, lon1, lat2, lon2) -> float:
    return math.hypot((lat2 - lat1) * 110.57, (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2)))


def classify(lat: float, lon: float, with_ais, without_ais, radius_km: float = RADIUS_KM) -> str:
    """ais / no_ais / none for one candidate: the nearer of the two kinds of GFW cell within the radius."""
    da = min((km(lat, lon, a, b) for a, b in with_ais), default=math.inf)
    dn = min((km(lat, lon, a, b) for a, b in without_ais), default=math.inf)
    if min(da, dn) > radius_km:
        return "none"
    return "ais" if da <= dn else "no_ais"


def monitor_passes(monitor: Path) -> dict[tuple[str, str], list[tuple[float, float, str]]]:
    """(area, acquired_utc) -> [(lon, lat, class)] from the monitor's published candidates."""
    out: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for f in sorted((monitor / "candidates").glob("*.geojson")):
        for ft in json.loads(f.read_text(encoding="utf-8")).get("features", []):
            p, c = ft.get("properties") or {}, (ft.get("geometry") or {}).get("coordinates")
            if not (isinstance(c, list) and len(c) >= 2 and isinstance(p.get("area"), str) and isinstance(p.get("acquired_utc"), str)):
                continue
            out.setdefault((p["area"], p["acquired_utc"]), []).append((float(c[0]), float(c[1]), "A" if p.get("class") == "A" else "B"))
    return out


def due(acquired: datetime, entry: dict | None, now: datetime) -> bool:
    """Whether a pass is to be asked for now: old enough for GFW, and either not matched yet or still in its recheck days."""
    age = (now - acquired).total_seconds() / 86400
    if age < LAG_DAYS:
        return False
    if entry and entry.get("status") == "done":
        return age < LAG_DAYS + RECHECK_DAYS
    return age < GIVE_UP_DAYS


def match_pass(area: str, acquired_utc: str, cands, bbox, report, now: datetime) -> tuple[dict, list]:
    acquired = parse_utc(acquired_utc)
    hours, days = hours_of(acquired), date_range(acquired)
    with_ais = cells(report(bbox, days, True), hours)
    without_ais = cells(report(bbox, days, False), hours)
    entry = {"area": area, "acquired_utc": acquired_utc, "checked_utc": iso(now),
             "gfw_cells_ais": len(with_ais), "gfw_cells_no_ais": len(without_ais)}
    if not with_ais and not without_ais:
        # GFW has nothing for this hour and area yet (or never will): nothing is said about the candidates.
        entry["status"] = "waiting" if (now - acquired).total_seconds() / 86400 < GIVE_UP_DAYS - 1 else "no_gfw"
        return entry, []
    marks = [[round(lon, 5), round(lat, 5), cls, classify(lat, lon, with_ais, without_ais)] for lon, lat, cls in cands]
    entry["status"] = "done"
    for cls in ("A", "B"):
        for code in ("ais", "no_ais", "none"):
            entry[f"{cls}_{code}"] = sum(1 for m in marks if m[2] == cls and m[3] == code)
    return entry, marks


def meta() -> dict:
    return {
        "version": VERSION, "attribution": ATTRIBUTION, "source": SOURCE, "dataset": DATASET,
        "license": "Global Fishing Watch の API のデータは CC BY-NC 4.0（非商用）。公開物に「Powered by Global Fishing Watch.」を表示する",
        "marks": {
            "ais": f"候補から {RADIUS_KM} km 以内に、GFW が AIS と一致させた検出がある（AIS を出していた船）",
            "no_ais": f"候補から {RADIUS_KM} km 以内に、GFW が AIS と一致させられなかった検出がある（GFW も船と見たが、合う AIS の航跡がない）",
            "none": "GFW の検出がない（こちらの誤検出・固定物か、GFW の見逃し）",
        },
        "method": (f"外洋の定期監視（docs/data/ocean/monitor/candidates/）の撮影ごとに、域の範囲・撮影の時刻の GFW の検出（0.01° の格子、"
                   f"AIS 一致と不一致の別）を 4Wings report で取り、候補の位置を格子の中心と比べる。両方が {RADIUS_KM} km 以内なら近い方。"
                   f"GFW に出るのは撮影の約 {LAG_DAYS} 日後。出てから {RECHECK_DAYS} 日は毎日照合し直す。"
                   f"{GIVE_UP_DAYS} 日たっても GFW に検出がない撮影は no_gfw（印なし）"),
        "statuses": {"done": "照合した", "waiting": "GFW にまだその撮影の検出がない", "no_gfw": "GFW に検出がないまま期限を過ぎた（印なし）"},
        "limits": [
            "GFW の一致は確率による突き合わせ。AIS を出している船が no_ais になることも、その逆もある。",
            "格子は 0.01°（約 1 km）。船が混んだ所では隣の船の印が付くことがある。",
            "GFW の受信網が弱い海では、AIS を出していても no_ais になりやすい。",
            "no_ais は軍艦とは限らない（AIS を切った漁船・商船、AIS の義務がない船を含む）。艦種・国籍は分からない。",
            "GFW の行そのもの、船の識別子（MMSI・船名など）は保存も公開もしない。",
        ],
        "files": {"passes.json": "域×撮影ごとの照合の状態と件数", "series.json": "照合の済んだ撮影の件数の時系列",
                  "matches/YYYY-MM-DD.json": "撮影日（UTC）ごとの候補の印（[経度, 緯度, クラス, 印]）"},
    }


def run(monitor: Path, out: Path, report, now: datetime, areas: dict[str, dict], max_passes: int = 80, pause=time.sleep) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "passes.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"passes": []}
    entries = {(e["area"], e["acquired_utc"]): e for e in state.get("passes", [])}
    passes = monitor_passes(monitor)
    summary = {"asked": 0, "done": 0, "waiting": 0, "failed": 0, "errors": []}
    changed: dict[str, dict] = {}   # day -> {(area, acquired): marks}
    for (area, acq), cands in sorted(passes.items(), key=lambda kv: kv[0][1]):
        if summary["asked"] >= max_passes or area not in areas:
            continue
        if not due(parse_utc(acq), entries.get((area, acq)), now):
            continue
        summary["asked"] += 1
        try:
            entry, marks = match_pass(area, acq, cands, areas[area]["bbox"], report, now)
        except GfwError as e:
            summary["failed"] += 1
            summary["errors"].append(f"{area} {acq}: {e}")
            if len(summary["errors"]) >= 3 and summary["failed"] == summary["asked"]:
                break   # the service is down or the token is refused: stop asking
            continue
        finally:
            pause(PAUSE_S)
        old = entries.get((area, acq))
        if entry["status"] != "done" and old and old.get("status") == "done":
            continue   # a matched pass is never taken back by a later empty answer
        entry["first_done_utc"] = (old or {}).get("first_done_utc") or (entry["checked_utc"] if entry["status"] == "done" else None)
        entries[(area, acq)] = entry
        summary["done" if entry["status"] == "done" else "waiting"] += 1
        if entry["status"] == "done":
            changed.setdefault(acq[:10], {})[(area, acq)] = marks
    for day, by in changed.items():
        path = out / "matches" / f"{day}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"passes": []}
        keep = [p for p in doc.get("passes", []) if (p["area"], p["acquired_utc"]) not in by]
        new = [{"area": a, "acquired_utc": t, "candidates": m} for (a, t), m in by.items()]
        doc = {"version": VERSION, "attribution": ATTRIBUTION, "fields": ["lon", "lat", "class", "mark"],
               "passes": sorted(keep + new, key=lambda p: (p["acquired_utc"], p["area"]))}
        path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    ordered = sorted(entries.values(), key=lambda e: (e["acquired_utc"], e["area"]))
    state_path.write_text(json.dumps({"version": VERSION, "attribution": ATTRIBUTION, "passes": ordered}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    rows = [{k: e.get(k) for k in ("area", "acquired_utc", "A_ais", "A_no_ais", "A_none", "B_ais", "B_no_ais", "B_none", "gfw_cells_ais", "gfw_cells_no_ais", "checked_utc")}
            for e in ordered if e.get("status") == "done"]
    (out / "series.json").write_text(json.dumps({"version": VERSION, "attribution": ATTRIBUTION, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", default="docs/data/ocean/monitor")
    ap.add_argument("--out", default="docs/data/ocean/gfw")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "monitor.json"))
    ap.add_argument("--now")
    ap.add_argument("--max-passes", type=int, default=80)
    a = ap.parse_args(argv)
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if not token:
        print("::notice title=ocean-gfw::secret GFW_API_TOKEN が未登録（または空）。照合しない")
        return 0
    areas = {x["id"]: x for x in json.loads(Path(a.config).read_text(encoding="utf-8"))["areas"]}
    now = parse_utc(a.now) if a.now else datetime.now(timezone.utc)
    s = run(Path(a.monitor), Path(a.out), lambda bbox, days, matched: http_report(token, bbox, days, matched), now, areas, a.max_passes)
    print(f"::notice title=ocean-gfw::照合を試みた撮影 {s['asked']}（照合 {s['done']}・GFW 待ち {s['waiting']}・失敗 {s['failed']}）")
    for e in s["errors"][:5]:
        print(f"::warning title=ocean-gfw::{e}")
    return 1 if s["asked"] and s["failed"] == s["asked"] else 0


if __name__ == "__main__":
    sys.exit(main())
