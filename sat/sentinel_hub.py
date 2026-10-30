"""Copernicus Data Space Ecosystem: Sentinel Hub Catalog and Process APIs (A 案).

Only small Sentinel-1 GRD cut-outs (sigma0, orthorectified) are requested, so a
run stays tiny. Credentials: OAuth client CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET
(registered 2026-10-30, expires about one year later).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH = "https://sh.dataspace.copernicus.eu"
UA = "observatory-data/1 (+https://github.com/2wrr6c7mmj-jpg/observatory-data; public research)"


class AuthError(RuntimeError):
    """The OAuth client was refused: usually an expired or deleted client secret."""


def _post(url: str, body: bytes, headers: dict, timeout: int = 120):
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, **headers}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def token(client_id: str | None = None, client_secret: str | None = None) -> str:
    cid = client_id or os.environ.get("CDSE_SH_CLIENT_ID", "")
    sec = client_secret or os.environ.get("CDSE_SH_CLIENT_SECRET", "")
    if not cid or not sec:
        raise AuthError("CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET が設定されていません（リポジトリの Secrets を確認）")
    body = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": cid, "client_secret": sec}).encode()
    try:
        with _post(TOKEN_URL, body, {"Content-Type": "application/x-www-form-urlencoded"}, timeout=30) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        detail = e.read(2000).decode("utf-8", "replace")
        if e.code in (400, 401):
            raise AuthError(
                "CDSE の OAuth クライアントが拒否されました（HTTP %d）。有効期限切れ（登録 2026-10-30、約1年）か削除の"
                "可能性があります。CDSE のダッシュボード（https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings）"
                "の OAuth clients で新しいクライアントを作り、Secrets の CDSE_SH_CLIENT_ID と CDSE_SH_CLIENT_SECRET を"
                "差し替えてください。応答: %s" % (e.code, detail[:300])) from e
        raise


def bbox(site: dict) -> list[float]:
    """[west, south, east, north] in WGS84 around the site centre."""
    import math
    dlat = site["height_km"] / 2 / 110.574
    dlon = site["width_km"] / 2 / (111.320 * math.cos(math.radians(site["lat"])))
    return [round(site["lon"] - dlon, 5), round(site["lat"] - dlat, 5), round(site["lon"] + dlon, 5), round(site["lat"] + dlat, 5)]


def search(tok: str, box: list[float], start: str, end: str, limit: int = 20) -> list[dict]:
    """Sentinel-1 GRD IW scenes over box between start and end (ISO, UTC)."""
    body = json.dumps({"collections": ["sentinel-1-grd"], "bbox": box, "datetime": f"{start}/{end}", "limit": limit,
                       "filter": "sar:instrument_mode = 'IW'", "filter-lang": "cql2-text"}).encode()
    with _post(SH + "/api/v1/catalog/1.0.0/search", body,
               {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}) as r:
        feats = json.loads(r.read()).get("features", [])
    out = []
    for f in feats:
        p = f.get("properties", {})
        out.append({"id": f.get("id"), "datetime": p.get("datetime"), "platform": p.get("platform"),
                    "orbit_state": p.get("sat:orbit_state"), "relative_orbit": p.get("sat:relative_orbit"),
                    "polarizations": p.get("sar:polarizations")})
    return sorted(out, key=lambda s: s["datetime"] or "")


EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask"]}], output: {bands: 3, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask]; }
"""


def cutout(tok: str, box: list[float], when: str, width: int, height: int) -> tuple[bytes, float | None]:
    """sigma0 VV, VH (linear) and data mask for the scene at `when` (+-1 min). Returns (GeoTIFF, PU spent)."""
    import datetime as dt
    t = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    rng = {"from": (t - dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "to": (t + dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    body = json.dumps({
        "input": {"bounds": {"bbox": box, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-1-grd",
                            "dataFilter": {"timeRange": rng, "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH"},
                            "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True, "demInstance": "COPERNICUS_30",
                                           "upsampling": "NEAREST", "downsampling": "NEAREST"}}]},
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT}).encode()
    with _post(SH + "/api/v1/process", body, {"Authorization": "Bearer " + tok, "Content-Type": "application/json",
                                              "Accept": "image/tiff"}) as r:
        pu = r.headers.get("x-processingunits-spent")
        return r.read(), (float(pu) if pu not in (None, "") else None)
