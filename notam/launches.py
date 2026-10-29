"""Keep a history of launch-related NOTAMs / navigational warnings and infer
launch sites and azimuths from the hazard areas.

  python -m notam.launches EXTRACT.json

EXTRACT.json comes from notam/extract.mjs. NOTAM Watch publishes only the
notices that are currently valid, so this file keeps every launch notice seen
(docs/data/launches/notices.json) and rebuilds the events from that history.

Inference (events.json):
  1. Notices whose validity windows overlap and that start within 12 hours of
     each other, and whose areas lie within 4,000 km, form one event.
  2. For each known launch site, the bearings from the site to the area
     centres are computed. Areas downrange of one launch lie close to a
     single ray from the site, so the site with the smallest bearing spread
     (at most 8 degrees, at least two areas) is taken, and the mean bearing is
     the launch azimuth. With one area only, the nearest site is reported as
     a weak candidate.
  3. The orbit inclination implied by the azimuth: cos i = cos(latitude) x sin(azimuth).

These are estimates from the published hazard areas, not confirmation that a
launch took place: launches slip or are cancelled after notices are issued.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "launches"
EARTH_KM = 6371.0
SITES = {  # name: (lat, lon, country)
    "酒泉": (40.96, 100.29, "CN"), "太原": (38.85, 111.61, "CN"), "西昌": (28.25, 102.03, "CN"),
    "文昌": (19.61, 110.95, "CN"), "海陽（海上発射）": (35.0, 122.0, "CN"),
    "西海（東倉里）": (39.66, 124.71, "KP"), "平壌周辺": (39.03, 125.75, "KP"),
    "種子島": (30.40, 130.97, "JP"), "内之浦": (31.25, 131.08, "JP"), "羅老": (34.43, 127.53, "KR"),
    "ボストーチヌイ": (51.88, 128.33, "RU"),
}
MAX_SPREAD_DEG = 8.0
EAST_ASIA = (-10.0, 60.0, 90.0, 160.0)  # lat min, lat max, lon min, lon max


def parse_time(s: str | None) -> datetime | None:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if s else None


def centre(shapes: list[dict]) -> tuple[float, float] | None:
    pts = [p for s in shapes for p in (s["points"][:1] if s.get("kind") == "circle" else s["points"])]
    if not pts:
        return None
    # Average on the unit sphere so that areas across the antimeridian stay correct.
    x = sum(math.cos(math.radians(a)) * math.cos(math.radians(b)) for a, b in pts)
    y = sum(math.cos(math.radians(a)) * math.sin(math.radians(b)) for a, b in pts)
    z = sum(math.sin(math.radians(a)) for a, _ in pts)
    return (round(math.degrees(math.atan2(z, math.hypot(x, y))), 3), round(math.degrees(math.atan2(y, x)), 3))


def distance_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def bearing(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    y = math.sin(lo2 - lo1) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def circular_mean_and_spread(angles: list[float]) -> tuple[float, float]:
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    mean = (math.degrees(math.atan2(s, c)) + 360) % 360
    spread = max(abs((a - mean + 180) % 360 - 180) for a in angles) * 2
    return mean, spread


def inclination(site_lat: float, azimuth: float) -> float:
    v = math.cos(math.radians(site_lat)) * math.sin(math.radians(azimuth))
    return round(math.degrees(math.acos(max(-1.0, min(1.0, v)))), 1)


def infer_site(centres: list[tuple[float, float]]) -> dict:
    if not centres:
        return {"method": "none"}
    if len(centres) == 1:
        name, d = min(((n, distance_km(s[:2], centres[0])) for n, s in SITES.items()), key=lambda x: x[1])
        if d > 1500:
            return {"method": "none"}
        return {"method": "nearest-only", "site": name, "country": SITES[name][2], "distance_km": round(d)}
    best = None
    for name, (lat, lon, country) in SITES.items():
        ds = [distance_km((lat, lon), c) for c in centres]
        if min(ds) < 30 or max(ds) > 4500:
            continue
        mean, spread = circular_mean_and_spread([bearing((lat, lon), c) for c in centres])
        if spread <= MAX_SPREAD_DEG and (best is None or spread < best[1]):
            best = (name, spread, mean, country, lat)
    if best is None:
        return {"method": "unaligned"}
    name, spread, mean, country, lat = best
    return {"method": "aligned", "site": name, "country": country, "azimuth_deg": round(mean, 1),
            "bearing_spread_deg": round(spread, 1), "inclination_deg": inclination(lat, mean)}


def in_east_asia(c) -> bool:
    return EAST_ASIA[0] <= c[0] <= EAST_ASIA[1] and EAST_ASIA[2] <= c[1] <= EAST_ASIA[3]


def events(notices: list[dict]) -> list[dict]:
    items = []
    for n in notices:
        c = centre(n["shapes"])
        start, end = parse_time(n["start_utc"]), parse_time(n["end_utc"])
        if c and start and in_east_asia(c):
            items.append((start, end or start + timedelta(hours=6), c, n))
    items.sort(key=lambda x: x[0])
    groups: list[list] = []
    for it in items:
        for g in groups:
            g_start = min(x[0] for x in g)
            if abs((it[0] - g_start).total_seconds()) <= 12 * 3600 and any(it[0] <= x[1] and x[0] <= it[1] for x in g) \
                    and all(distance_km(it[2], x[2]) <= 4000 for x in g):
                g.append(it)
                break
        else:
            groups.append([it])
    out = []
    for g in groups:
        # Several notices may describe the same area (aviation + maritime); count each place once.
        centres = []
        for _, _, c, _ in g:
            if all(distance_km(c, k) > 20 for k in centres):
                centres.append(c)
        out.append({
            "window_start_utc": min(x[0] for x in g).strftime("%Y-%m-%dT%H:%MZ"),
            "window_end_utc": max(x[1] for x in g).strftime("%Y-%m-%dT%H:%MZ"),
            "areas": [list(c) for c in centres],
            "notices": [x[3]["key"] for x in g],
            **infer_site(centres),
        })
    return out


def merge(history: dict, extract: dict) -> dict:
    seen = history.setdefault("notices", {})
    now = extract["retrieved_utc"]
    for n in extract["notices"]:
        prev = seen.get(n["key"])
        seen[n["key"]] = {**n, "first_seen_utc": prev["first_seen_utc"] if prev else now, "last_seen_utc": now}
    history["last_snapshot"] = extract.get("snapshot")
    history["last_retrieved_utc"] = now
    return history


def main(path: str) -> int:
    extract = json.loads(Path(path).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    hist_path = OUT / "notices.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {}
    history = merge(history, extract)
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    ev = events(list(history["notices"].values()))
    (OUT / "events.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sites": {k: list(v) for k, v in SITES.items()}, "events": ev,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"::notice::launch notices stored {len(history['notices'])}, events {len(ev)} "
          f"({sum(1 for e in ev if e['method'] == 'aligned')} aligned)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
