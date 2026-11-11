"""Probe: what the Global Fishing Watch 4Wings report returns for Sentinel-1 SAR vessel detections.

Runs only on a probe branch. Asks for one area and one day in a few variants (matched to AIS or not,
different group-by values) and prints the status, the shape of the answer and a few rows. Vessel
identifiers (vessel id, MMSI, name, call sign, IMO) are never printed. Nothing is committed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-sar-presence:latest"
HIDE = {"vesselId", "vessel_id", "mmsi", "shipName", "callsign", "imo", "vesselIDs"}


def region(b):
    w, s, e, n = b
    return json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]})


def ask(token, bbox, days, extra, as_string):
    q = {"spatial-resolution": "HIGH", "temporal-resolution": "HOURLY", "datasets[0]": DATASET,
         "date-range": days, "format": "JSON"}
    q.update(extra)
    url = URL + "?" + urllib.parse.urlencode(q)
    g = region(bbox)
    body = json.dumps({"geojson": g if as_string else json.loads(g)}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json", "User-Agent": "observatory-data"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=150) as r:
            return r.status, r.read(), time.time() - t, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t, dict(e.headers)
    except Exception as e:  # noqa: BLE001 - a probe reports whatever happened
        return 0, repr(e).encode(), time.time() - t, {}


def rows_of(doc):
    out = []
    for e in doc.get("entries", []) if isinstance(doc, dict) else []:
        if isinstance(e, dict) and any(isinstance(v, list) for v in e.values()):
            for k, v in e.items():
                if isinstance(v, list):
                    out += [dict(x, _key=k) for x in v if isinstance(x, dict)]
        elif isinstance(e, dict):
            out.append(e)
    return out


def describe(name, status, raw, secs, headers):
    print(f"## {name}: HTTP {status}, {len(raw)} bytes, {secs:.1f} s, type {headers.get('Content-Type')}, datasets {headers.get('x-datasets')}")
    if raw[:2] == b"PK":
        print("   zip answer")
        return
    try:
        doc = json.loads(raw)
    except Exception:  # noqa: BLE001
        print("   not JSON:", raw[:300].decode("utf-8", "replace"))
        return
    if status != 200:
        print("   ", json.dumps(doc, ensure_ascii=False)[:500])
        return
    rows = rows_of(doc)
    keys = sorted({k for r in rows for k in r})
    print(f"   top keys {sorted(doc)[:8]}, total {doc.get('total')}, rows {len(rows)}, row keys {keys}")
    if not rows:
        print("   ", json.dumps(doc, ensure_ascii=False)[:300])
        return
    det = sum(float(r.get("detections") or 0) for r in rows)
    dates = sorted({str(r.get("date")) for r in rows})
    cells = {(r.get("lat"), r.get("lon")) for r in rows}
    print(f"   detections {det:g}, cells {len(cells)}, dates {dates[:6]}{'…' if len(dates) > 6 else ''}")
    for k in ("flag", "vesselType", "shiptype", "geartype", "_key"):
        vals = {}
        for r in rows:
            if k in r:
                vals[str(r[k])] = vals.get(str(r[k]), 0) + 1
        if vals:
            print(f"   {k}: {dict(sorted(vals.items(), key=lambda kv: -kv[1])[:8])}")
    for r in rows[:3]:
        print("   row", json.dumps({k: ("…" if k in HIDE and r[k] else r[k]) for k in r}, ensure_ascii=False)[:330])


def presence(token, bbox, days, hours):
    q = {"spatial-resolution": "HIGH", "temporal-resolution": "HOURLY", "group-by": "FLAG", "datasets[0]": "public-global-presence:latest",
         "date-range": days, "format": "JSON"}
    url = URL + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, data=json.dumps({"geojson": json.loads(region(bbox))}).encode(), method="POST", headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json", "User-Agent": "observatory-data/1"})
    try:
        with urllib.request.urlopen(req, timeout=170) as r:
            doc = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        print("presence failed", type(e).__name__, getattr(e, "code", ""))
        return None
    out = {}
    for r in rows_of(doc):
        d = str(r.get("date"))
        if d in hours:
            out.setdefault(d, []).append((float(r["lat"]), float(r["lon"]), str(r.get("flag") or "?")))
    return out


def main():
    """Check of the no_ais marks: is there an AIS position (GFW presence, same hour) beside a candidate GFW left unmatched?"""
    import math
    from datetime import datetime, timedelta, timezone
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    cfg = {a["id"]: a for a in json.load(open("ocean/monitor.json", encoding="utf-8"))["areas"]}
    def km(a, b, c, d):
        return math.hypot((c - a) * 110.57, (d - b) * 111.32 * math.cos(math.radians((a + c) / 2)))
    for area, day in (("OKI", "2026-11-05"), ("TWE_S", "2026-11-01"), ("TSS", "2026-11-03")):
        m = json.load(open(f"docs/data/ocean/gfw/matches/{day}.json", encoding="utf-8"))
        for ps in m["passes"]:
            if ps["area"] != area:
                continue
            t = datetime.strptime(ps["acquired_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            h0 = t.strftime("%Y-%m-%d %H:00")
            hours = {h0, (t - timedelta(hours=1)).strftime("%Y-%m-%d %H:00"), (t + timedelta(hours=1)).strftime("%Y-%m-%d %H:00")}
            nxt = (t + timedelta(days=1)).strftime("%Y-%m-%d")
            pr = presence(token, cfg[area]["bbox"], f"{day},{nxt}", hours)
            time.sleep(2)
            if pr is None:
                continue
            same = pr.get(h0, [])
            print(f"## {area} {ps['acquired_utc']}: presence cells in the pass's hour {len(same)}, hours {dict((k, len(v)) for k, v in sorted(pr.items()))}")
            for mark in ("no_ais", "ais", "none"):
                ds = []
                for lon, lat, cls, mk in ps["candidates"]:
                    if cls != "A" or mk != mark or not same:
                        continue
                    best = min(same, key=lambda c: km(lat, lon, c[0], c[1]))
                    ds.append((round(km(lat, lon, best[0], best[1]), 1), best[2], round(lat, 2), round(lon, 2)))
                ds.sort()
                n = len(ds)
                print(f"   {mark}: n {n}, within 1 km {sum(1 for d in ds if d[0] <= 1)}, 3 km {sum(1 for d in ds if d[0] <= 3)}, 10 km {sum(1 for d in ds if d[0] <= 10)}")
                if mark != "ais":
                    print("     ", [(d[0], d[1], d[2], d[3]) for d in ds][:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
