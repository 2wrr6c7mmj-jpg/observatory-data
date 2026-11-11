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


def main():
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if not token:
        print("GFW_API_TOKEN がない")
        return 1
    bbox, days = [119.5, 25.3, 121.0, 26.5], "2026-10-31,2026-11-01"   # TSN, the pass of 2026-10-31
    variants = [
        ("matched=false group FLAG", {"group-by": "FLAG", "filters[0]": "matched='false'"}, False),
        ("matched=true group FLAG", {"group-by": "FLAG", "filters[0]": "matched='true'"}, False),
        ("matched=false no group", {"filters[0]": "matched='false'"}, False),
        ("all no group", {}, False),
        ("matched=true group VESSEL_ID", {"group-by": "VESSEL_ID", "filters[0]": "matched='true'"}, False),
        ("matched=false LOW DAILY FLAG", {"group-by": "FLAG", "filters[0]": "matched='false'", "temporal-resolution": "DAILY", "spatial-resolution": "LOW"}, False),
        ("v4.0 matched=false FLAG", {"group-by": "FLAG", "filters[0]": "matched='false'", "datasets[0]": "public-global-sar-presence:v4.0"}, False),
        ("presence (AIS) all FLAG", {"group-by": "FLAG", "datasets[0]": "public-global-presence:latest"}, False),
    ]
    for name, extra, as_string in variants:
        describe(name, *ask(token, bbox, days, extra, as_string))
        time.sleep(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
