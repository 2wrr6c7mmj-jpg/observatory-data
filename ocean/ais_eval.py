"""Detection rate of the open-sea ship detectors (cfar-v1, cfar-v2) measured against AIS south of Hong Kong.

Design: cfar-v2 design doc 5.2 / 5.3 (PR4).  Standard library only (``websockets`` is imported when recording).

    python -m ocean.ais_eval record-due --out-dir DIR [--skip NAME ...]      # ais-record.yml (hourly)
    python -m ocean.ais_eval evaluate   --ais-dir DIR --monitor docs/data/ocean/monitor --out docs/data/ocean/ais_eval
    python -m ocean.ais_eval predict    [--horizon-min 65]                    # catalogue only, prints the due passes

Flow (no raw AIS is ever committed or published):

1. ``record-due`` (GitHub Actions, hourly): the CDSE OData catalogue (no login, 0 PU) gives the Sentinel-1 passes over the
   HKS box of ``ocean/monitor.json`` in the last 30 days; the next passes are predicted (same satellite and relative orbit
   repeat every 12 days).  For a pass due within the next hour, AIS is received from aisstream.io for +-20 minutes of the
   time the satellite is abeam the box, inside the HKS box only.  The messages are reduced in memory to anonymous tracks
   (no MMSI, no name, no call sign: a running number, the position reports thinned to 15 s, the ship length) and sealed
   (encrypt-then-MAC, key derived from the aisstream key) into a file that goes to a short-lived Actions artifact.
2. ``evaluate`` (after the daily "Ocean monitor" run): for a sealed file whose pass the monitor has published for HKS
   (``passes.json`` status ok/partial), the imaging time of each ship is taken from the product's start/end and footprint,
   the AIS position is extrapolated with SOG/COG to it, shifted by the azimuth displacement of a moving target
   (-(R/V)*v_away, along the heading), and compared with the monitor's candidates at 200 m and 300 m.  Only counts, rates
   and per-candidate "AIS matched / unmatched / no AIS coverage" with a length band are written.  The artifact is deleted
   by the workflow afterwards.

Published (``docs/data/ocean/ais_eval``): ``meta.json``, ``summary.json``, ``passes.json``, ``candidates/YYYY-MM-DD.json``.
Never published: positions of ships, MMSI, names, per-ship rows, positions of ships that were missed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import struct
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ocean import coverage_probe as covp

VERSION = "ais-eval-1"
URL = "wss://stream.aisstream.io/v0/stream"
CONFIG_FILE = Path(__file__).resolve().parent / "monitor.json"
AREA_ID = "HKS"
REPEAT_DAYS = 12.0                 # exact repeat cycle of one Sentinel-1 satellite
LOOKBACK_DAYS = 30
WINDOW_MIN = 20.0                  # +- minutes around the pass
START_BEFORE_MIN = 65.0            # a run records a pass whose window starts within this many minutes
MIN_REMAIN_MIN = 5.0               # ... and which still has at least this many minutes left
RV_S = 120.0                       # R/V of Sentinel-1 IW (slant range / platform speed): 107-133 s over the swath
KN = 0.514444
RADII_M = (200.0, 300.0)
BANDS = (("ge100", 100.0, 1e9), ("50_100", 50.0, 100.0))
THIN_S = 15.0
AGE_MOVING_S = 120.0
AGE_STILL_S = 600.0
STILL_KN = 0.5
COAST_BAND_M = 1000.0
STATIC_M = 300.0
COVER_RADIUS_M = 10000.0           # a candidate is "inside AIS coverage" when an AIS report lies within this distance
HIST_STEP = 10.0
HIST_LIM = 100.0
POS_TYPES = ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport")
MAGIC = b"AISE1"
PBKDF_ITER = 100_000
ATTRIBUTION = ("Contains modified Copernicus Sentinel data [{years}]. AIS: aisstream.io (reception results only: counts, "
               "rates and per-candidate match flags; no AIS ship position or identity is published).")


def load_config(path: Path = CONFIG_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hks_area(cfg: dict) -> dict:
    return next(a for a in cfg["areas"] if a["id"] == AREA_ID)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s: str) -> datetime:
    return covp.parse_time(s)


# ---------------------------------------------------------------- geometry (small areas: local equirectangular)

def m_per_deg(lat: float) -> tuple[float, float]:
    """(metres per degree of longitude, of latitude) at ``lat``."""
    return 111320.0 * math.cos(math.radians(lat)), 110574.0


def xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    mx, my = m_per_deg(lat0)
    return (lon - lon0) * mx, (lat - lat0) * my


def dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    x, y = xy(lon2, lat2, lon1, lat1)
    return math.hypot(x, y)


def heading_deg(direction: str, lat: float) -> float:
    """Nominal ground-track heading (deg from north) of Sentinel-1: about 348 (ascending) and 192 (descending) at 22N.
    Same model as ``sat.detect2.platform_heading`` (inclination 98.18 deg, Earth rotation subtracted)."""
    gm, re_, h = 3.986004418e14, 6371000.0, 698000.0
    phi = math.radians(lat)
    a = math.asin(max(-1.0, min(1.0, math.cos(math.radians(98.18)) / math.cos(phi))))
    d = (direction or "").lower()
    if d.startswith("d"):
        a = math.pi - a
    elif not d.startswith("a"):
        raise ValueError(direction)
    vs = math.sqrt(gm / (re_ + h))
    vg = vs * re_ / (re_ + h)
    east = vg * math.sin(a) - 7.2921159e-5 * re_ * math.cos(phi)
    north = vg * math.cos(a)
    return math.degrees(math.atan2(east, north)) % 360.0


def dead_reckon(lon: float, lat: float, sog_kn: float, cog_deg: float, dt_s: float) -> tuple[float, float]:
    d = sog_kn * KN * dt_s
    c = math.radians(cog_deg)
    mx, my = m_per_deg(lat)
    return lon + d * math.sin(c) / mx, lat + d * math.cos(c) / my


def azimuth_shift(lon: float, lat: float, sog_kn: float, cog_deg: float, head_deg: float, rv_s: float = RV_S):
    """Apparent position of a moving ship in the SAR image.

    The Doppler of the ship's line-of-sight motion places it along the platform track at
    ``-(R/V) * v_away`` (forward = platform heading; v_away = ship velocity along the look direction, positive away
    from the radar; Sentinel-1 looks to the right, heading + 90 deg).  Returns (lon, lat, shift_m)."""
    if sog_kn is None or sog_kn < STILL_KN or cog_deg is None:
        return lon, lat, 0.0
    v = sog_kn * KN
    ve, vn = v * math.sin(math.radians(cog_deg)), v * math.cos(math.radians(cog_deg))
    look = math.radians(head_deg + 90.0)
    v_away = ve * math.sin(look) + vn * math.cos(look)
    shift = -rv_s * v_away
    h = math.radians(head_deg)
    mx, my = m_per_deg(lat)
    return lon + shift * math.sin(h) / mx, lat + shift * math.cos(h) / my, shift


def in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def bbox_share(rings: list, bbox: list, n: int = 15) -> float:
    """Share of the box covered by the footprint rings (n x n sample points)."""
    w, s, e, nn = bbox
    hit = 0
    for j in range(n):
        for i in range(n):
            lon, lat = w + (i + 0.5) * (e - w) / n, s + (j + 0.5) * (nn - s) / n
            if any(in_ring(lon, lat, r) for r in rings):
                hit += 1
    return hit / (n * n)


# ---------------------------------------------------------------- catalogue: slices, passes, prediction

def odata_url(bbox: list, t0: datetime, t1: datetime, skip: int = 0) -> str:
    flt = (f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;{covp.wkt_box(bbox)}') "
           f"and ContentDate/Start ge {covp.iso_ms(t0)} and ContentDate/Start lt {covp.iso_ms(t1)} "
           f"and contains(Name,'_IW_GRDH_1SDV_')")
    q = {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": "1000"}
    if skip:
        q["$skip"] = str(skip)
    return covp.ODATA_URL + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def fetch_products(bbox: list, t0: datetime, t1: datetime, get=covp.http_json) -> list[dict]:
    out, seen, skip = [], set(), 0
    while True:
        page = get(odata_url(bbox, t0, t1, skip))
        items = page.get("value", [])
        for p in items:
            if p.get("Id") not in seen:
                seen.add(p.get("Id"))
                out.append(p)
        if len(items) < 1000:
            return out
        skip += 1000


def slices_of(products: list[dict]) -> list[dict]:
    """Slices (IW GRDH, one per product; the COG copy dropped): start, end, rings, platform, relative orbit, direction."""
    out, seen = [], set()
    for p in products:
        info = covp.product_info(p)
        if not info or info["mode"] != "IW" or info["kind"] != "GRD" or not info["start"] or not info["rings"]:
            continue
        key = (info["platform"], info["start"])
        if key in seen:
            continue
        seen.add(key)
        end = None
        cd = (p.get("ContentDate") or {}).get("End")
        if cd:
            end = parse_time(cd)
        else:
            m = covp.NAME_RE.match(p.get("Name", ""))
            if m:
                end = datetime.strptime(m["stop"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        if not end or end <= info["start"]:
            end = info["start"] + timedelta(seconds=25)
        out.append({"platform": info["platform"], "relative_orbit": info["relative_orbit"], "direction": info["direction"],
                    "start": info["start"], "end": end, "rings": info["rings"], "name": info["name"]})
    out.sort(key=lambda s: s["start"])
    return out


def time_at(sl: list[dict], lon: float, lat: float, head: float, lon0: float, lat0: float) -> datetime | None:
    """Time at which the platform is abeam (``lon``, ``lat``): linear in the along-track coordinate across the
    slice whose footprint extent (projected on the heading) holds the point (else the nearest slice)."""
    ux, uy = math.sin(math.radians(head)), math.cos(math.radians(head))
    px, py = xy(lon, lat, lon0, lat0)
    sp = px * ux + py * uy
    best = None
    for s in sl:
        proj = []
        for r in s["rings"]:
            for q in r:
                x, y = xy(q[0], q[1], lon0, lat0)
                proj.append(x * ux + y * uy)
        if not proj:
            continue
        s0, s1 = min(proj), max(proj)
        if s1 - s0 < 1.0:
            continue
        d = 0.0 if s0 <= sp <= s1 else min(abs(sp - s0), abs(sp - s1))
        if best is None or d < best[0]:
            best = (d, s["start"] + (s["end"] - s["start"]) * ((sp - s0) / (s1 - s0)))
    return best[1] if best else None


def group_passes(sl: list[dict], gap_s: float = 120.0) -> list[list[dict]]:
    """Consecutive slices of one satellite and relative orbit (start within ``gap_s`` of the previous end)."""
    groups: list[list[dict]] = []
    for s in sorted(sl, key=lambda x: x["start"]):
        for g in groups:
            last = g[-1]
            if (last["platform"], last["relative_orbit"]) == (s["platform"], s["relative_orbit"]) and \
                    0 <= (s["start"] - last["end"]).total_seconds() <= gap_s:
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def past_passes(products: list[dict], bbox: list, min_share: float) -> list[dict]:
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    out = []
    for g in group_passes(slices_of(products)):
        rings = [r for s in g for r in s["rings"]]
        share = bbox_share(rings, bbox)
        if share < min_share:
            continue
        head = heading_deg(g[0]["direction"] or "ascending", lat0)
        t = time_at(g, lon0, lat0, head, lon0, lat0)
        if t is None:
            continue
        out.append({"platform": g[0]["platform"], "relative_orbit": g[0]["relative_orbit"], "direction": g[0]["direction"],
                    "t_center": t, "share": round(share, 3), "slices": g})
    out.sort(key=lambda p: p["t_center"])
    return out


def due_passes(past: list[dict], now: datetime, start_before_min: float = START_BEFORE_MIN,
               window_min: float = WINDOW_MIN, min_remain_min: float = MIN_REMAIN_MIN) -> list[dict]:
    """Passes (predicted from the latest past pass of each satellite and relative orbit + k * 12 days) whose recording
    window [T-window, T+window] starts within ``start_before_min`` and has at least ``min_remain_min`` left."""
    latest: dict = {}
    for p in past:
        latest[(p["platform"], p["relative_orbit"])] = p
    out = []
    for (plat, rel), p in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        for k in range(0, 4):
            t = p["t_center"] + timedelta(days=REPEAT_DAYS * k)
            start, end = t - timedelta(minutes=window_min), t + timedelta(minutes=window_min)
            if end - timedelta(minutes=min_remain_min) <= now:
                continue
            if start > now + timedelta(minutes=start_before_min):
                continue
            out.append({"name": pass_name(plat, rel, t), "platform": plat, "relative_orbit": rel,
                        "direction": p["direction"], "t_center_utc": iso(t), "start_utc": iso(start), "end_utc": iso(end),
                        "predicted": k > 0, "based_on_utc": iso(p["t_center"]), "share": p["share"]})
    return out


def pass_name(platform: str, rel, t: datetime) -> str:
    return f"ais-{platform}-{rel}-{t:%Y%m%dT%H%M}"


# ---------------------------------------------------------------- sealing (no third-party package)

def _stream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hmac.new(key, nonce + struct.pack(">Q", i), hashlib.sha256).digest()
        i += 1
    return bytes(out[:n])


def _keys(secret: str, salt: bytes) -> tuple[bytes, bytes]:
    k = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), b"observatory-ais-eval-1" + salt, PBKDF_ITER, 64)
    return k[:32], k[32:]


def seal(data: bytes, secret: str, salt: bytes | None = None, nonce: bytes | None = None) -> bytes:
    """Encrypt-then-MAC (HMAC-SHA256 counter-mode keystream, HMAC-SHA256 tag); key from ``secret`` (PBKDF2)."""
    if not secret:
        raise ValueError("no key")
    salt = salt or os.urandom(16)
    nonce = nonce or os.urandom(16)
    ek, mk = _keys(secret, salt)
    ct = bytes(a ^ b for a, b in zip(data, _stream(ek, nonce, len(data))))
    head = MAGIC + salt + nonce
    return head + ct + hmac.new(mk, head + ct, hashlib.sha256).digest()


def unseal(blob: bytes, secret: str) -> bytes:
    if not secret or len(blob) < len(MAGIC) + 32 + 32 or not blob.startswith(MAGIC):
        raise ValueError("not a sealed file")
    salt, nonce = blob[5:21], blob[21:37]
    ct, tag = blob[37:-32], blob[-32:]
    ek, mk = _keys(secret, salt)
    if not hmac.compare_digest(tag, hmac.new(mk, blob[:-32], hashlib.sha256).digest()):
        raise ValueError("wrong key or damaged file")
    return bytes(a ^ b for a, b in zip(ct, _stream(ek, nonce, len(ct))))


# ---------------------------------------------------------------- AIS reception (in memory, anonymised)

class Collector:
    """Reduces aisstream messages to anonymous tracks.  Keeps no MMSI, name or call sign (a running number only)."""

    def __init__(self, bbox: list, t0: float):
        self.bbox, self.t0 = bbox, t0
        self.ids: dict = {}
        self.tracks: list[dict] = []
        self.messages = 0
        self.by_type: dict = {}

    def _ship(self, mmsi: int) -> dict:
        if mmsi not in self.ids:
            self.ids[mmsi] = len(self.tracks)
            self.tracks.append({"id": len(self.tracks), "len": None, "wid": None, "t": [], "lon": [], "lat": [],
                                "sog": [], "cog": []})
        return self.tracks[self.ids[mmsi]]

    @staticmethod
    def _dim(d: dict | None):
        if not d:
            return None, None
        try:
            a, b, c, e = (int(d.get(k) or 0) for k in ("A", "B", "C", "D"))
        except (TypeError, ValueError):
            return None, None
        return (a + b if a + b > 0 else None), (c + e if c + e > 0 else None)

    def feed(self, m: dict, t: float) -> None:
        mt = m.get("MessageType")
        if not mt or "error" in m:
            return
        self.by_type[mt] = self.by_type.get(mt, 0) + 1
        md = m.get("MetaData") or {}
        try:
            mmsi = int(md.get("MMSI"))
        except (TypeError, ValueError):
            return
        body = (m.get("Message") or {}).get(mt) or {}
        if mt in POS_TYPES:
            try:
                lat, lon = float(body.get("Latitude", md.get("latitude"))), float(body.get("Longitude", md.get("longitude")))
            except (TypeError, ValueError):
                return
            w, s, e, n = self.bbox
            if not (s <= lat <= n and w <= lon <= e) or abs(lat) > 90 or abs(lon) > 180:
                return
            self.messages += 1
            sh = self._ship(mmsi)
            if sh["t"] and t - self.t0 - sh["t"][-1] < THIN_S:
                return
            sog, cog = body.get("Sog"), body.get("Cog")
            sh["t"].append(round(t - self.t0, 1))
            sh["lon"].append(round(lon, 5))
            sh["lat"].append(round(lat, 5))
            sh["sog"].append(None if sog is None or sog >= 102.3 else round(float(sog), 1))
            sh["cog"].append(None if cog is None or cog >= 360 else round(float(cog), 1))
        elif mt == "ShipStaticData":
            ln, wd = self._dim(body.get("Dimension"))
            if ln:
                sh = self._ship(mmsi)
                sh["len"], sh["wid"] = ln, wd
        elif mt == "StaticDataReport":
            ln, wd = self._dim((body.get("ReportB") or {}).get("Dimension"))
            if ln:
                sh = self._ship(mmsi)
                sh["len"], sh["wid"] = ln, wd

    def result(self) -> dict:
        ships = [s for s in self.tracks if s["t"]]
        for i, s in enumerate(ships):
            s["id"] = i
        return {"ships": ships, "messages": self.messages, "by_type": self.by_type}


def subscription(key: str, bbox: list) -> dict:
    w, s, e, n = bbox
    return {"APIKey": key, "BoundingBoxes": [[[s, w], [n, e]]],
            "FilterMessageTypes": [*POS_TYPES, "ShipStaticData", "StaticDataReport"]}


def scrub(s: str, key: str) -> str:
    return s.replace(key, "***") if key else s


async def receive(key: str, bbox: list, t_start: float, t_end: float, col: Collector, log=print) -> dict:
    import websockets
    meta = {"connects": 0, "auth": None, "errors": []}
    while time.time() < t_end:
        meta["connects"] += 1
        try:
            async with websockets.connect(URL, open_timeout=20, max_size=2 ** 22) as ws:
                await ws.send(json.dumps(subscription(key, bbox)))
                while time.time() < t_end:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=max(1, min(30, t_end - time.time())))
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        continue
                    if m.get("MessageType") == "SubscriptionConfirmation":
                        meta["auth"] = "ok"
                    elif "error" in m:
                        meta["auth"] = "error"
                        meta["errors"].append(scrub(str(m["error"]), key)[:200])
                    col.feed(m, time.time())
        except Exception as e:  # noqa: BLE001
            msg = scrub(f"{type(e).__name__}: {e}", key)[:300]
            meta["errors"].append(msg)
            log(f"::warning title=ais-record::接続エラー: {msg}")
            if meta["connects"] >= 6:
                break
            await asyncio.sleep(min(30, 5 * meta["connects"]))
    return meta


def record_pass(due: dict, bbox: list, key: str, out_dir: Path, now_fn=time.time, recv=receive, log=print) -> Path | None:
    """Receive AIS for one due pass (waits until its window starts), seal the anonymous tracks into ``out_dir``."""
    start = parse_time(due["start_utc"]).timestamp()
    end = parse_time(due["end_utc"]).timestamp()
    t0 = max(start, now_fn())
    wait = t0 - now_fn()
    if wait > 0:
        log(f"{due['name']}: 窓の開始まで {wait / 60:.1f} 分待つ")
        time.sleep(wait)
    col = Collector(bbox, t0)
    meta = asyncio.run(recv(key, bbox, t0, end, col, log))
    res = col.result()
    payload = {"version": "ais-raw-1", "pass": due, "bbox": bbox, "t0_utc": iso(datetime.fromtimestamp(t0, timezone.utc)),
               "t1_utc": iso(datetime.fromtimestamp(end, timezone.utc)), "reception": meta,
               "messages": res["messages"], "by_type": res["by_type"], "ships": res["ships"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (due["name"] + ".bin")
    path.write_bytes(seal(json.dumps(payload, separators=(",", ":")).encode("utf-8"), key))
    log(f"{due['name']}: 受信 {res['messages']} 件、船 {len(res['ships'])} 隻、接続 {meta['connects']} 回、認証 {meta['auth']}、"
        f"封印したファイル {path.stat().st_size} バイト（MMSI・船名なし）")
    return path


# ---------------------------------------------------------------- evaluation

def band_of(length: float | None):
    if length is None:
        return None
    for name, lo, hi in BANDS:
        if lo <= length < hi:
            return name
    return "lt50"


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def hist_add(h: dict, v: float) -> None:
    v = max(-HIST_LIM - HIST_STEP, min(HIST_LIM + HIST_STEP, v))
    k = math.floor(v / HIST_STEP) * HIST_STEP
    k = min(k, HIST_LIM)
    h[str(int(k))] = h.get(str(int(k)), 0) + 1


def hist_stats(h: dict) -> dict | None:
    """Median and 10/90 % of the length difference (candidate - AIS, m) from a 10 m histogram (bin centres)."""
    n = sum(h.values())
    if n < 5:
        return None
    bins = sorted((int(k), v) for k, v in h.items())

    def q(p):
        acc = 0
        for k, v in bins:
            acc += v
            if acc >= p * n:
                return k + HIST_STEP / 2
        return bins[-1][0] + HIST_STEP / 2
    return {"n": n, "median_m": q(0.5), "p10_m": q(0.1), "p90_m": q(0.9),
            "note": "10 m bins (bin centre); the two end bins are open"}


def nearest_report(sh: dict, t: float):
    best = None
    for i, ti in enumerate(sh["t"]):
        d = abs(ti - t)
        if best is None or d < best[0]:
            best = (d, i)
    return best[1] if best else None


def seg_dist_m(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class Obstacles:
    """Coast and fixed objects of the OSM file of the monitor (``osm/HKS.json``): the same objects the monitor
    removes (coast band 1 km, fixed objects 300 m).  Approximate: distances to the OSM lines and nodes."""

    def __init__(self, osm: dict | None, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.coast: list = []     # segments
        self.static: list = []    # segments and points
        for e in (osm or {}).get("elements", []):
            tags = e.get("tags") or {}
            pts = [(p["lon"], p["lat"]) for p in e.get("geometry", [])] if e.get("geometry") else \
                ([(e["lon"], e["lat"])] if e.get("type") == "node" and "lat" in e else [])
            pts = [xy(a, b, lon0, lat0) for a, b in pts]
            if not pts:
                continue
            segs = list(zip(pts, pts[1:])) or [(pts[0], pts[0])]
            if tags.get("natural") in ("coastline",):
                self.coast += segs
            elif tags.get("natural") in ("water", "bay", "strait", "wetland"):
                continue
            else:
                self.static += segs

    @staticmethod
    def _near(segs: list, x: float, y: float, r: float) -> bool:
        for (ax, ay), (bx, by) in segs:
            if min(ax, bx) - r > x or max(ax, bx) + r < x or min(ay, by) - r > y or max(ay, by) + r < y:
                continue
            if seg_dist_m(x, y, ax, ay, bx, by) <= r:
                return True
        return False

    def reason(self, lon: float, lat: float) -> str | None:
        x, y = xy(lon, lat, self.lon0, self.lat0)
        if self._near(self.coast, x, y, COAST_BAND_M):
            return "near_coast"
        if self._near(self.static, x, y, STATIC_M):
            return "static"
        return None


def in_rects(rects: list, lon: float, lat: float) -> bool:
    return any(w <= lon <= e and s <= lat <= n for w, s, e, n in rects)


def feature_rect(f: dict):
    ring = f["geometry"]["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def evaluate_pass(ais: dict, cands: list[dict], seen_rects: list, obstacles: Obstacles | None, head: float,
                  t_img, rv_s: float = RV_S) -> tuple[dict, list[dict]]:
    """One pass.  ``cands``: [{lon, lat, v2_match, length, v2_length}] (the monitor's published candidates of HKS for this
    pass).  ``t_img(lon, lat)``: imaging time (seconds after the AIS t0) of a point.  Returns (pass result, per-candidate
    flags).  Positions of ships stay inside this function."""
    excl = {"no_length": 0, "stale": 0, "motion_unknown": 0, "outside_seen": 0, "near_coast": 0, "static": 0}
    allsh = []   # every ship with a usable position: apparent (azimuth-shifted) position, length, band
    for sh in ais.get("ships", []):
        i = nearest_report(sh, t_img(sh["lon"][0], sh["lat"][0]))
        if i is None:
            continue
        sog, cog = sh["sog"][i], sh["cog"][i]
        moving = sog is not None and sog >= STILL_KN
        if sog is None or (moving and cog is None):
            excl["motion_unknown"] += 1
            continue
        lon, lat = sh["lon"][i], sh["lat"][i]
        ti = t_img(lon, lat)
        for _ in range(2):   # the imaging time depends on where the ship is
            if moving:
                lon, lat = dead_reckon(sh["lon"][i], sh["lat"][i], sog, cog, ti - sh["t"][i])
            ti = t_img(lon, lat)
        if abs(sh["t"][i] - ti) > (AGE_MOVING_S if moving else AGE_STILL_S):
            excl["stale"] += 1
            continue
        alon, alat, shift = azimuth_shift(lon, lat, sog, cog, head, rv_s)
        allsh.append({"lon": alon, "lat": alat, "tlon": lon, "tlat": lat, "len": sh.get("len"), "shift": shift})
    ref = []
    for s in allsh:
        if not s["len"]:
            excl["no_length"] += 1
        elif not in_rects(seen_rects, s["tlon"], s["tlat"]):
            excl["outside_seen"] += 1
        else:
            why = obstacles.reason(s["tlon"], s["tlat"]) if obstacles else None
            if why:
                excl[why] += 1
            else:
                ref.append({**s, "len": float(s["len"]), "band": band_of(s["len"])})
    # detection rate: a reference ship is a hit when a candidate of the detector lies within r
    pools = {"v1": ("length", cands), "v2": ("v2_length", [c for c in cands if c.get("v2_match")])}
    det: dict = {}
    ldiff: dict = {}
    for dname, (lkey, pool) in pools.items():
        det[dname] = {}
        for r in RADII_M:
            cell = {b[0]: {"n": 0, "hit": 0} for b in BANDS}
            for s in ref:
                if s["band"] not in cell:
                    continue
                cell[s["band"]]["n"] += 1
                best = None
                for c in pool:
                    d = dist_m(s["lon"], s["lat"], c["lon"], c["lat"])
                    if d <= r and (best is None or d < best[0]):
                        best = (d, c)
                if best:
                    cell[s["band"]]["hit"] += 1
                    if r == RADII_M[-1] and best[1].get(lkey) is not None:
                        hist_add(ldiff.setdefault(dname, {}).setdefault(s["band"], {}), best[1][lkey] - s["len"])
            det[dname][str(int(r))] = cell
    flags = []
    cnt = {str(int(r)): {"matched": 0, "unmatched": 0, "no_ais_coverage": 0} for r in RADII_M}
    for c in cands:
        f = {"lon": c["lon"], "lat": c["lat"]}
        cover = any(dist_m(c["lon"], c["lat"], s["lon"], s["lat"]) <= COVER_RADIUS_M for s in allsh)
        for r in RADII_M:
            near = [s for s in allsh if dist_m(s["lon"], s["lat"], c["lon"], c["lat"]) <= r]
            st = "matched" if near else ("unmatched" if cover else "no_ais_coverage")
            f[f"ais_{int(r)}"] = st
            cnt[str(int(r))][st] += 1
            if r == RADII_M[-1]:
                lens = [s["len"] for s in near if s["len"]]
                f["ais_length_band"] = band_of(max(lens)) if lens else None
        flags.append(f)
    res = {"ais": {"ships_with_positions": len(ais.get("ships", [])), "ships_usable": len(allsh), "ships_reference": len(ref),
                   "excluded": excl, "messages": ais.get("messages")},
           "candidates": {"n": len(cands), "n_v2": len(pools["v2"][1]), "by_ais": cnt},
           "detection": det, "length_diff_m": ldiff, "rv_s": rv_s}
    return res, flags


def _json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def summarize(passes: list[dict]) -> dict:
    """Totals over the evaluated passes: detection rate per detector, radius and length band (n = reference ships)."""
    det: dict = {}
    hist: dict = {}
    cands = {"n": 0, "by_ais": {str(int(r)): {"matched": 0, "unmatched": 0, "no_ais_coverage": 0} for r in RADII_M}}
    used = [p for p in passes if p.get("status") == "evaluated"]
    for p in used:
        for dname, rr in p["detection"].items():
            for r, bands in rr.items():
                for b, v in bands.items():
                    t = det.setdefault(dname, {}).setdefault(r, {}).setdefault(b, {"n": 0, "hit": 0})
                    t["n"] += v["n"]
                    t["hit"] += v["hit"]
        for dname, bands in p.get("length_diff_m", {}).items():
            for b, h in bands.items():
                tgt = hist.setdefault(dname, {}).setdefault(b, {})
                for k, v in h.items():
                    tgt[k] = tgt.get(k, 0) + v
        cands["n"] += p["candidates"]["n"]
        for r, v in p["candidates"]["by_ais"].items():
            for k, x in v.items():
                cands["by_ais"][r][k] += x
    for dname, rr in det.items():
        for r, bands in rr.items():
            for b, t in bands.items():
                t["rate"] = round(t["hit"] / t["n"], 3) if t["n"] else None
                t["ci95"] = wilson(t["hit"], t["n"])
    ld = {d: {b: hist_stats(h) for b, h in bands.items()} for d, bands in hist.items()}
    return {"version": VERSION, "passes_evaluated": len(used),
            "passes_not_evaluated": len(passes) - len(used),
            "detection": det, "length_diff_m": ld, "candidates": cands,
            "display": {"min_n_to_show": 20,
                        "note": "show 'rate (n)' only when n >= min_n_to_show; the rate is for HKS at 40 m, ocean-monitor-1 "
                                "candidates, within the radius; ships not covered by the AIS receivers are not in n"},
            "updated_utc": iso(now_utc())}


def meta_doc() -> dict:
    return {
        "version": VERSION,
        "purpose": "Detection rate of cfar-v1 and cfar-v2 (the candidates of the open-sea monitor, ocean-monitor-1) measured "
                   "against AIS south of Hong Kong (area HKS of ocean/monitor.json), for the confidence shown on open-sea "
                   "candidates elsewhere.",
        "files": {"summary.json": "totals: detection rate per detector (v1, v2), match radius (200, 300 m) and AIS length band, "
                                  "with n (reference ships) and a 95 % Wilson interval; length-difference quantiles; "
                                  "counts of candidates by AIS match",
                  "passes.json": "one record per Sentinel-1 pass: counts only",
                  "candidates/YYYY-MM-DD.json": "per candidate of the monitor (join on acquired_utc + lon + lat): AIS matched / "
                                                "unmatched / no_ais_coverage at 200 and 300 m and the length band (ge100, "
                                                "50_100, lt50) of the longest AIS ship within 300 m"},
        "method": {
            "ais_source": "aisstream.io terrestrial AIS, received for +-20 minutes around each pass, inside the HKS box only",
            "time": "imaging time of each point from the product's start/end and footprint (linear along the track)",
            "extrapolation": "last report within 2 min (moving) or 10 min (SOG < 0.5 kn) extrapolated with SOG/COG to the imaging time",
            "azimuth_shift": "a moving ship appears displaced along the heading by -(R/V)*v_away (R/V = 120 s; 107-133 s over "
                             "the swath; v_away = velocity along the right-looking direction, positive away); the AIS position "
                             "is moved by it before matching",
            "radii_m": list(RADII_M),
            "reference_ships": "AIS ships with a known length inside the cells the monitor saw (state seen), not within 1 km "
                               "of the coast or 300 m of a fixed object of OSM (osm/HKS.json, approximate)",
            "bands": {"ge100": "AIS length >= 100 m", "50_100": "50 m <= length < 100 m"},
            "detectors": {"v1": "all published candidates (cfar-v1 medium/high, length estimate >= 80 m; v1 low and the coast "
                                "band, fixed objects and sea-state exclusions are not published, so they cannot hit)",
                          "v2": "published candidates with v2_match (a cfar-v2 detection within 100 m)"},
            "match": "a ship is a hit when at least one candidate of the detector lies within the radius (not one-to-one)",
            "candidate_status": "matched = an AIS ship within the radius; unmatched = none, but an AIS report within 10 km "
                                "(inside AIS coverage); no_ais_coverage = no AIS report within 10 km"},
        "limits": [
            "Terrestrial AIS only: ships far from the receivers (offshore) are missing, so the rate is for ships near the "
            "coast of Hong Kong and the Pearl River estuary, and 'unmatched' is not 'dark'.",
            "A hit does not prove the candidate is that ship (a 40 m pixel, 200-300 m radius, busy waters).",
            "Ships with no static data in the window have no length and are not in n.",
            "The v1 length estimate is limited to >= 80 m, so the 50-100 m band is expected to be low by design.",
        ],
        "rules": ["No AIS position, MMSI, name or call sign is published or committed; ship counts and rates only.",
                  "Per-candidate flags are published for the monitor's own candidates (already public); nothing is published "
                  "for ships that were not detected."],
        "attribution": ATTRIBUTION.format(years="2026"),
    }


def passes_state(out: Path) -> list[dict]:
    return _json(out / "passes.json", {"passes": []}).get("passes", [])


def merge_candidates(out: Path, day: str, acquired: str, flags: list[dict]) -> None:
    p = out / "candidates" / f"{day}.json"
    doc = _json(p, {"version": VERSION, "passes": {}})
    doc["passes"][acquired] = flags
    doc["passes"] = dict(sorted(doc["passes"].items()))
    _write(p, doc)


def load_pass_inputs(mon: Path, row: dict):
    """(candidates, seen rects) of the monitor for one pass row."""
    day = row["acquired_utc"][:10]
    cf = _json(mon / "candidates" / f"{day}.geojson", {"features": []})
    cands = []
    for f in cf.get("features", []):
        p = f["properties"]
        if p.get("area") == AREA_ID and p.get("acquired_utc") == row["acquired_utc"]:
            cands.append({"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1],
                          "v2_match": bool(p.get("v2_match")), "length": p.get("length_m_est_40m"),
                          "v2_length": p.get("v2_length_m")})
    cov = _json(mon / "coverage" / f"{day}.geojson", {"features": []})
    rects = [feature_rect(f) for f in cov.get("features", [])
             if f["properties"].get("area") == AREA_ID and f["properties"].get("acquired_utc") == row["acquired_utc"]
             and f["properties"].get("state") == "seen"]
    return cands, rects


def find_row(rows: list[dict], due: dict, tol_min: float = 15.0):
    t = parse_time(due["t_center_utc"])
    for r in rows:
        if r.get("area") != AREA_ID or r.get("status") not in ("ok", "partial"):
            continue
        if (r.get("platform") or "")[-1:].upper() != due["platform"][-1:].upper() or r.get("relative_orbit") != due["relative_orbit"]:
            continue
        if abs((parse_time(r["acquired_utc"]) - t).total_seconds()) <= tol_min * 60:
            return r
    return None


def evaluate_file(path: Path, key: str, mon: Path, out: Path, get=covp.http_json, log=print, rv_s: float = RV_S) -> str:
    """Evaluate one sealed AIS file.  Returns "evaluated" | "wait" (pass not published by the monitor yet) |
    "skip" (cannot be evaluated: record kept in passes.json with the reason) | "error"."""
    try:
        ais = json.loads(unseal(path.read_bytes(), key).decode("utf-8"))
    except (ValueError, OSError) as e:
        log(f"::warning title=ais-eval::{path.name}: 開けない（{e}）")
        return "error"
    due = ais["pass"]
    rows = _json(mon / "passes.json", {"passes": []}).get("passes", [])
    row = find_row(rows, due)
    if row is None:
        log(f"{due['name']}: 監視の passes.json にまだ（または ok/partial でない）。待つ")
        return "wait"
    state = passes_state(out)
    if any(p.get("acquired_utc") == row["acquired_utc"] for p in state):
        return "evaluated"
    cfg = load_config()
    bbox = hks_area(cfg)["bbox"]
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    acq = parse_time(row["acquired_utc"])
    prods = fetch_products(bbox, acq - timedelta(minutes=3), acq + timedelta(minutes=6), get=get)
    sl = [s for s in slices_of(prods) if s["platform"] == due["platform"] and s["relative_orbit"] == due["relative_orbit"]]
    base = {"area": AREA_ID, "acquired_utc": row["acquired_utc"], "platform": row.get("platform"),
            "relative_orbit": row.get("relative_orbit"), "orbit_direction": row.get("orbit_direction"),
            "recorded_window_utc": [ais["t0_utc"], ais["t1_utc"]], "evaluated_utc": iso(now_utc())}

    def record(status, reason):
        state.append({**base, "status": status, "reason": reason})
        state.sort(key=lambda p: p["acquired_utc"])
        _write(out / "passes.json", {"version": VERSION, "attribution": ATTRIBUTION.format(years=acq.year), "passes": state})
        _write(out / "summary.json", summarize(state))
        return "skip"
    if not sl:
        return record("not_evaluated", "OData でこの撮影の製品が見つからない（時刻を決められない）")
    head = heading_deg(row.get("orbit_direction") or due.get("direction") or "ascending", lat0)
    tc = time_at(sl, lon0, lat0, head, lon0, lat0)
    t0 = parse_time(ais["t0_utc"])
    t1 = parse_time(ais["t1_utc"])
    if tc is None or not (t0 + timedelta(minutes=1) <= tc <= t1 - timedelta(minutes=1)):
        return record("not_evaluated", "撮影の時刻が AIS の受信窓の外（予測のずれか受信の欠け）")
    if ais.get("reception", {}).get("auth") != "ok" or not ais.get("ships"):
        return record("not_evaluated", "AIS を受信できなかった（認証・接続・船なし）")
    cands, rects = load_pass_inputs(mon, row)
    osm = _json(mon / "osm" / f"{AREA_ID}.json", None)
    obst = Obstacles(osm, lon0, lat0) if osm else None

    def t_img(lon, lat):
        t = time_at(sl, lon, lat, head, lon0, lat0) or tc
        return (t - t0).total_seconds()
    res, flags = evaluate_pass(ais, cands, rects, obst, head, t_img, rv_s)
    state.append({**base, "status": "evaluated", "t_center_utc": iso(tc), "osm_obstacles": bool(osm), **res})
    state.sort(key=lambda p: p["acquired_utc"])
    _write(out / "passes.json", {"version": VERSION, "attribution": ATTRIBUTION.format(years=acq.year), "passes": state})
    merge_candidates(out, row["acquired_utc"][:10], row["acquired_utc"], flags)
    _write(out / "summary.json", summarize(state))
    _write(out / "meta.json", meta_doc())
    d1 = res["detection"]["v1"]["300"]
    log(f"{row['acquired_utc']}: 基準の船 {res['ais']['ships_reference']}、候補 {len(cands)}、"
        f"v1 300 m の検出 ≥100 m {d1['ge100']['hit']}/{d1['ge100']['n']}、50〜100 m {d1['50_100']['hit']}/{d1['50_100']['n']}")
    return "evaluated"


# ---------------------------------------------------------------- command line

def gh_output(name: str, value: str) -> None:
    p = os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def cmd_predict(a, now=None, get=covp.http_json, log=print) -> list[dict]:
    cfg = load_config()
    area = hks_area(cfg)
    now = now or now_utc()
    prods = fetch_products(area["bbox"], now - timedelta(days=LOOKBACK_DAYS), now, get=get)
    past = past_passes(prods, area["bbox"], float(area.get("min_area_share", cfg.get("min_area_share", 0.3))))
    due = due_passes(past, now, a.horizon_min)
    log(f"過去 {LOOKBACK_DAYS} 日の HKS の通過 {len(past)} 件（{', '.join(sorted({p['platform'] + '/' + str(p['relative_orbit']) for p in past}))}）、"
        f"これから受信する通過 {len(due)} 件")
    return due


def cmd_record_due(a, now=None, get=covp.http_json, log=print) -> int:
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    if not key:
        log("::notice title=ais-record::secret AISSTREAM_API_KEY が空か未登録。AIS は受信しない（何もせず成功で終わる）")
        gh_output("recorded", "")
        return 0
    due = [d for d in cmd_predict(a, now, get, log) if d["name"] not in set(a.skip or [])]
    if not due:
        log("::notice title=ais-record::今から受信する通過はない")
        gh_output("recorded", "")
        return 0
    cfg = load_config()
    bbox = hks_area(cfg)["bbox"]
    names = []
    for d in due:
        log(f"::notice title=ais-record::{d['name']} 撮影 {d['t_center_utc']}（{'予測' if d['predicted'] else '実績'}）、受信 {d['start_utc']}〜{d['end_utc']}")
        p = record_pass(d, bbox, key, Path(a.out_dir), log=log)
        if p:
            names.append(p.name)
    gh_output("recorded", " ".join(names))
    return 0


def cmd_evaluate(a, get=covp.http_json, log=print) -> int:
    key = os.environ.get("AISSTREAM_API_KEY", "").strip()
    results = {}
    if not key:
        log("::notice title=ais-eval::secret AISSTREAM_API_KEY が空か未登録。照合しない")
        return 0
    for p in sorted(Path(a.ais_dir).glob("*.bin")):
        try:
            results[p.name] = evaluate_file(p, key, Path(a.monitor), Path(a.out), get=get, log=log)
        except Exception as e:  # noqa: BLE001  (one bad file must not stop the others)
            results[p.name] = "error"
            log(f"::warning title=ais-eval::{p.name}: {type(e).__name__}: {str(e)[:200]}")
    done = [n for n, r in results.items() if r in ("evaluated", "skip")]
    gh_output("done", " ".join(done))
    log(f"::notice title=ais-eval::{json.dumps(results, ensure_ascii=False)}")
    if not (Path(a.out) / "meta.json").exists():
        _write(Path(a.out) / "meta.json", meta_doc())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("predict")
    p1.add_argument("--horizon-min", type=float, default=START_BEFORE_MIN)
    p2 = sub.add_parser("record-due")
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--horizon-min", type=float, default=START_BEFORE_MIN)
    p2.add_argument("--skip", nargs="*", default=[])
    p3 = sub.add_parser("evaluate")
    p3.add_argument("--ais-dir", required=True)
    p3.add_argument("--monitor", required=True)
    p3.add_argument("--out", required=True)
    p4 = sub.add_parser("init")
    p4.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        print(json.dumps(cmd_predict(a), ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "record-due":
        return cmd_record_due(a)
    if a.cmd == "evaluate":
        return cmd_evaluate(a)
    out = Path(a.out)
    _write(out / "meta.json", meta_doc())
    if not (out / "summary.json").exists():
        _write(out / "summary.json", summarize([]))
    if not (out / "passes.json").exists():
        _write(out / "passes.json", {"version": VERSION, "attribution": ATTRIBUTION.format(years="2026"), "passes": []})
    return 0


if __name__ == "__main__":
    sys.exit(main())
