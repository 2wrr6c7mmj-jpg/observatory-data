"""Copernicus Data Space Ecosystem: Sentinel Hub OAuth, Catalog and Process API.

Credentials come from the environment (CDSE_SH_CLIENT_ID, CDSE_SH_CLIENT_SECRET)
and are never printed, logged or written. Error messages say what went wrong
(missing, invalid/expired, no permission, quota) without echoing secrets.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_URL = "https://sh.dataspace.copernicus.eu"
CATALOG_URL = SH_URL + "/api/v1/catalog/1.0.0/search"
PROCESS_URL = SH_URL + "/api/v1/process"
USER_AGENT = "observatory-data/1"

# Encoding of the evalscript output (UINT16): dB = value / DB_SCALE - DB_OFFSET, 0 = no data.
DB_OFFSET = 50.0
DB_SCALE = 800.0
LIA_SCALE = 100.0

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [%(bands)s, "dataMask"],
          output: {bands: %(nout)d, sampleType: "UINT16"}};
}
function enc(x) {
  if (!(x > 0)) return 0;
  return Math.max(1, Math.min(65535, Math.round((10 * Math.log(x) / Math.LN10 + %(off)s) * %(scale)s)));
}
function evaluatePixel(s) {
  return [enc(s.VV), enc(s.VH)%(lia)s, s.dataMask];
}
"""


class CDSEError(RuntimeError):
    """A failure with a message that explains the cause (no secrets inside)."""


class CredentialError(CDSEError):
    pass


def evalscript(with_lia: bool) -> str:
    bands = '"VV", "VH", "localIncidenceAngle"' if with_lia else '"VV", "VH"'
    # VV/VH default to linear power (sigma0 with backCoeff SIGMA0_ELLIPSOID); the angle is in degrees.
    return EVALSCRIPT % {
        "bands": bands, "nout": 4 if with_lia else 3, "off": DB_OFFSET, "scale": DB_SCALE,
        "lia": ", Math.round(s.localIncidenceAngle * %s)" % LIA_SCALE if with_lia else "",
    }


def _http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120,
          opener=urllib.request.urlopen):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body


def _short(body: bytes, n: int = 300) -> str:
    text = body.decode("utf-8", "replace").replace("\n", " ")
    return text[:n]


class Client:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None, opener=urllib.request.urlopen):
        self._id = client_id if client_id is not None else os.environ.get("CDSE_SH_CLIENT_ID", "")
        self._secret = client_secret if client_secret is not None else os.environ.get("CDSE_SH_CLIENT_SECRET", "")
        self._opener = opener
        self._token = None
        self._token_exp = 0.0
        self.requests = 0  # Sentinel Hub requests (catalog + process), for the quota record
        self.processing_units = 0.0
        self.pu_reported = True  # False when any process response lacked the PU header
        self.last_headers: list[str] = []

    def token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        if not self._id or not self._secret:
            missing = [n for n, v in (("CDSE_SH_CLIENT_ID", self._id), ("CDSE_SH_CLIENT_SECRET", self._secret)) if not v]
            raise CredentialError(f"CDSE の資格情報が設定されていません（{', '.join(missing)} が空）。リポジトリの Secrets を確かめてください")
        data = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": self._id,
                                       "client_secret": self._secret}).encode()
        status, _h, body = _http(TOKEN_URL, data, {"Content-Type": "application/x-www-form-urlencoded"},
                                 timeout=60, opener=self._opener)
        if status != 200:
            try:
                err = json.loads(body)
            except ValueError:
                err = {}
            code = err.get("error", "")
            desc = err.get("error_description", "")
            if status in (400, 401) and code in ("invalid_client", "unauthorized_client", "invalid_grant"):
                raise CredentialError(
                    f"CDSE の認証に失敗（HTTP {status} {code}: {desc}）。OAuth クライアントが無効か期限切れです。"
                    "Sentinel Hub ダッシュボード（https://shapps.dataspace.copernicus.eu/dashboard/ → User settings → OAuth clients）で"
                    "クライアントの有効期限を確かめ、作り直したら Secrets の CDSE_SH_CLIENT_ID・CDSE_SH_CLIENT_SECRET を更新してください")
            raise CDSEError(f"CDSE のトークン取得に失敗（HTTP {status} {code} {desc}）".strip())
        tok = json.loads(body)
        self._token = tok["access_token"]
        self._token_exp = time.time() + float(tok.get("expires_in", 300))
        return self._token

    def _post_sh(self, url: str, payload: dict, accept: str) -> tuple[dict, bytes]:
        body = json.dumps(payload).encode()
        for attempt in range(4):
            headers = {"Authorization": "Bearer " + self.token(), "Content-Type": "application/json", "Accept": accept}
            self.requests += 1
            status, h, data = _http(url, body, headers, opener=self._opener)
            if status == 200:
                return h, data
            if status == 401 and attempt == 0:
                self._token = None  # token expired mid-run: get a new one once
                continue
            if status == 401:
                raise CredentialError(f"Sentinel Hub が認証を拒否（HTTP 401）。資格情報が無効か期限切れです: {_short(data)}")
            if status == 403:
                raise CredentialError(f"Sentinel Hub がアクセスを拒否（HTTP 403）。アカウントの権限か無料枠を確かめてください: {_short(data)}")
            if status == 429 or status >= 500:
                wait = float(h.get("retry-after") or 0) or 20 * (attempt + 1)
                if status == 429 and attempt == 3:
                    raise CDSEError(f"Sentinel Hub の利用制限（HTTP 429）。無料枠の処理単位かリクエスト数の上限の可能性: {_short(data)}")
                time.sleep(min(wait, 120))
                continue
            raise CDSEError(f"Sentinel Hub がリクエストを受け付けません（HTTP {status}）: {_short(data)}")
        raise CDSEError(f"Sentinel Hub への要求が繰り返し失敗しました（HTTP {status}）: {_short(data)}")

    def catalog(self, bbox: list[float], start: datetime, end: datetime, limit: int = 100,
                collection: str = "sentinel-1-grd") -> list[dict]:
        payload = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{iso(start)}/{iso(end)}",
            "limit": limit,
        }
        feats = []
        for _ in range(10):
            _h, data = self._post_sh(CATALOG_URL, payload, "application/geo+json, application/json")
            res = json.loads(data)
            feats.extend(res.get("features", []))
            nxt = (res.get("context") or {}).get("next")
            if not nxt:
                break
            payload["next"] = nxt
        return feats

    def process(self, bbox: list[float], width: int, height: int, t_from: datetime, t_to: datetime,
                with_lia: bool = True, upsampling: str = "NEAREST") -> tuple[bytes, float | None]:
        """``upsampling``: interpolation of the S1 GRD (Sentinel Hub docs: for Sentinel-1 it applies
        "regardless of the resolution" and ``downsampling`` is "Not used"). The daily run keeps NEAREST."""
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {"from": iso(t_from), "to": iso(t_to)},
                        "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH",
                        "mosaickingOrder": "mostRecent",
                    },
                    "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True,
                                   "demInstance": "COPERNICUS_30", "upsampling": upsampling, "downsampling": "NEAREST"},
                }],
            },
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
            "evalscript": evalscript(with_lia),
        }
        h, data = self._post_sh(PROCESS_URL, payload, "image/png")
        self.last_headers = sorted(h)  # names only, for the probe
        pu = h.get("x-processingunits-spent")
        try:
            pu = float(pu) if pu is not None else None
        except ValueError:
            pu = None
        if pu is None:
            self.pu_reported = False
            pu = estimate_pu(width, height, 3 if with_lia else 2)
        self.processing_units += pu
        return data, pu


    def process_raw(self, payload: dict, accept: str, input_bands: int) -> tuple[bytes, float]:
        """Any Process API request (Sentinel-2 in sat/s2.py); PU from the response header, else estimated
        (no orthorectification factor)."""
        h, data = self._post_sh(PROCESS_URL, payload, accept)
        self.last_headers = sorted(h)
        pu = h.get("x-processingunits-spent")
        try:
            pu = float(pu) if pu is not None else None
        except ValueError:
            pu = None
        if pu is None:
            self.pu_reported = False
            out = payload.get("output", {})
            pu = round(max(0.005, out.get("width", 0) * out.get("height", 0) / 512 ** 2 * max(input_bands / 3, 1)), 3)
        self.processing_units += pu
        return data, pu


def estimate_pu(width: int, height: int, input_bands: int) -> float:
    """Sentinel Hub PU rule: area/512^2 x max(bands/3, 1) x 2 for S1 orthorectification (min 0.01)."""
    return round(max(0.01, width * height / 512 ** 2 * max(input_bands / 3, 1) * 2), 3)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def relative_orbit(platform: str, absolute: int | None) -> int | None:
    """Relative orbit (track) from the absolute orbit number for Sentinel-1A/B."""
    if absolute is None:
        return None
    p = (platform or "").lower()
    if p.endswith("1a"):
        return (absolute - 73) % 175 + 1
    if p.endswith("1b"):
        return (absolute - 27) % 175 + 1
    return None


def acquisitions(features: list[dict], gap_s: float = 120) -> list[dict]:
    """Group catalog items (consecutive slices of one pass) into acquisitions.

    Only IW scenes with both VV and VH are kept. Returns dicts sorted by time.
    """
    items = []
    for f in features:
        p = f.get("properties", {})
        mode = p.get("sar:instrument_mode")
        pols = p.get("sar:polarizations") or []
        if mode and mode != "IW":
            continue
        if pols and not ({"VV", "VH"} <= set(pols)):
            continue
        t = parse_time(p["datetime"])
        absolute = p.get("sat:absolute_orbit")
        rel = p.get("sat:relative_orbit")
        if rel is None:
            rel = relative_orbit(p.get("platform", ""), absolute)
        items.append({
            "time": t,
            "platform": p.get("platform"),
            "orbit_direction": p.get("sat:orbit_state"),
            "absolute_orbit": absolute,
            "relative_orbit": rel,
            "ids": [f.get("id")],
        })
    items.sort(key=lambda x: x["time"])
    out = []
    for it in items:
        last = out[-1] if out else None
        if last and last["platform"] == it["platform"] and (it["time"] - last["t_end"]).total_seconds() <= gap_s:
            last["t_end"] = it["time"]
            last["ids"].extend(it["ids"])
            continue
        it["t_end"] = it["time"]
        out.append(it)
    for a in out:
        a["acquired_utc"] = iso(a["time"])
        a["t_from"] = a["time"] - timedelta(seconds=30)
        a["t_to"] = a["t_end"] + timedelta(seconds=30)
    return out
