"""Collect recorded and scheduled launches from Launch Library 2 (The Space Devs).

  python -m collectors.ll2

Two requests a day (recent previous launches, upcoming launches), well inside
the anonymous allowance of 15 requests per hour. Launches are kept in
docs/data/launches/ll2.json by Launch Library id; a later record of the same
launch (status changed from scheduled to success, a new time) replaces it.
A refused request (HTTP 429, e.g. a shared runner address) keeps the stored
data and is retried on the next run.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://ll.thespacedevs.com/2.2.0"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "launches"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data)"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read(20_000_000))


def compact(x: dict) -> dict:
    """The fields used here, from one Launch Library 2 launch record."""
    pad = x.get("pad") or {}
    mission = x.get("mission") or {}
    orbit = mission.get("orbit") or {}
    rocket = ((x.get("rocket") or {}).get("configuration") or {})
    lsp = x.get("launch_service_provider") or {}
    num = lambda v: float(v) if v not in (None, "") else None
    return {
        "ll2_id": x.get("id"), "name": x.get("name"),
        "status": (x.get("status") or {}).get("abbrev"),
        "net_utc": x.get("net"), "window_start_utc": x.get("window_start"), "window_end_utc": x.get("window_end"),
        "provider": lsp.get("name"), "provider_country": lsp.get("country_code"),
        "rocket": rocket.get("full_name") or rocket.get("name"),
        "mission_type": mission.get("type"), "orbit": orbit.get("abbrev"),
        "pad": pad.get("name"), "location": (pad.get("location") or {}).get("name"),
        "country": (pad.get("location") or {}).get("country_code") or pad.get("country_code"),
        "lat": num(pad.get("latitude")), "lon": num(pad.get("longitude")),
        "url": x.get("url"),
    }


def fetch_all(now: datetime) -> tuple[list[dict], list[str]]:
    since = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    urls = [f"{API}/launch/previous/?mode=normal&limit=100&net__gte={since}",
            f"{API}/launch/upcoming/?mode=normal&limit=100"]
    rows, errors = [], []
    for url in urls:
        try:
            rows += [compact(x) for x in get(url).get("results", [])]
        except (urllib.error.URLError, ValueError) as e:
            errors.append(f"{url.split('?')[0]}: {e}")
    return rows, errors


def merge(store: dict, rows: list[dict], now: str) -> dict:
    launches = store.setdefault("launches", {})
    for r in rows:
        if not r.get("ll2_id"):
            continue
        prev = launches.get(r["ll2_id"])
        launches[r["ll2_id"]] = {**r, "first_seen_utc": prev["first_seen_utc"] if prev else now, "last_seen_utc": now}
    store["last_retrieved_utc"] = now
    return store


def main() -> int:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = OUT / "ll2.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows, errors = fetch_all(now)
    if rows:
        store = merge(store, rows, stamp)
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for e in errors:
        print("::warning::Launch Library 2: " + e + " (stored data kept; retried next run)")
    print(f"::notice::Launch Library 2: {len(rows)} records, {len(store.get('launches', {}))} stored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
