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


def cells(token, bbox, days, matched):
    status, raw, _, _ = ask(token, bbox, days, {"group-by": "FLAG", "filters[0]": f"matched='{matched}'"}, False)
    if status != 200:
        print("cells", matched, "HTTP", status, raw[:200])
        return []
    return [(float(r["lat"]), float(r["lon"]), str(r.get("date")), float(r.get("detections") or 0)) for r in rows_of(json.loads(raw)) if "lat" in r]


def main():
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if not token:
        print("GFW_API_TOKEN がない")
        return 1
    import collections
    import math
    for area, bbox, day, nxt in (("TSN", [119.5, 25.3, 121.0, 26.5], "2026-10-31", "2026-11-01"),
                                 ("TSM", [119.3, 23.9, 120.4, 26.2], "2026-10-31", "2026-11-01"),
                                 ("OKI", None, "2026-11-05", "2026-11-06"), ("SENKAKU", [123.3, 25.4, 124.0, 26.2], "2026-11-08", "2026-11-09")):
        cfg = {a["id"]: a for a in json.load(open("ocean/monitor.json", encoding="utf-8"))["areas"]}
        bbox = cfg[area]["bbox"]
        gj = json.load(open(f"docs/data/ocean/monitor/candidates/{day}.geojson", encoding="utf-8"))
        cand = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0], f["properties"]["class"], f["properties"]["acquired_utc"])
                for f in gj["features"] if f["properties"].get("area") == area]
        t, f_ = cells(token, bbox, f"{day},{nxt}", "true"), cells(token, bbox, f"{day},{nxt}", "false")
        time.sleep(2)
        hours = collections.Counter(c[2] for c in t + f_)
        print(f"## {area} {day}: ours {len(cand)} (A {sum(1 for c in cand if c[2] == 'A')}), acquired {sorted({c[3] for c in cand})}; GFW matched cells {len(t)} det {sum(c[3] for c in t):g}, unmatched cells {len(f_)} det {sum(c[3] for c in f_):g}; hours {dict(hours)}")
        allc = [(c[0], c[1], "T") for c in t] + [(c[0], c[1], "F") for c in f_]
        if not allc or not cand:
            continue
        off = collections.Counter()
        res = {}
        for name, (ola, olo) in (("centre", (0.0, 0.0)), ("corner", (0.005, 0.005))):
            n = collections.Counter()
            for la, lo, cl, _ in cand:
                best = min(allc, key=lambda c: math.hypot((c[0] + ola - la) * 110.6, (c[1] + olo - lo) * 111.3 * math.cos(math.radians(la))))
                d = math.hypot((best[0] + ola - la) * 110.6, (best[1] + olo - lo) * 111.3 * math.cos(math.radians(la)))
                if name == "centre" and d < 3:
                    off[(round((la - best[0]) / 0.0025) * 0.0025, round((lo - best[1]) / 0.0025) * 0.0025)] += 1
                for lim in (0.8, 1.2, 2.0):
                    if d <= lim:
                        n[(cl, lim, best[2])] += 1
            res[name] = dict(sorted(n.items()))
        print("   offsets ours-minus-cell (deg, 0.0025 bins):", sorted(off.items(), key=lambda kv: -kv[1])[:14])
        dl = [la - min(allc, key=lambda c: abs(c[0] - la) + abs(c[1] - lo))[0] for la, lo, _, _ in cand]
        print("   centre:", res["centre"])
        print("   corner:", res["corner"])
        # how many GFW cells have one of ours within 1.2 km (corner assumption)
        hit = collections.Counter()
        for c in allc:
            d = min(math.hypot((c[0] + 0.005 - la) * 110.6, (c[1] + 0.005 - lo) * 111.3 * math.cos(math.radians(la))) for la, lo, _, _ in cand)
            hit[(c[2], d <= 1.2)] += 1
        print("   GFW cells with one of ours within 1.2 km (corner):", dict(hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
