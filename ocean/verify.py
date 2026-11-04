"""Second look at the open-sea A candidates (ocean-verify-1): a 20 m cut-out of the pass with the earlier passes.

One Process API request per candidate: 1.28 km square at 20 m (NEAREST), Sentinel-1 GRD IW, mosaicking ORBIT over
``lookback_days`` before the pass. The evalscript keeps the pass itself and up to ``max_prior`` earlier passes and
returns four bands per pixel (UINT16 PNG, the dB code of ``sat.cdse``):

    0  VV of the pass             (dB code; 0 = no data)
    1  VH of the pass
    2  VV low quantile (``q``) of the earlier passes at that pixel: what the spot looks like on most other days
    3  number of earlier passes with data at that pixel

What it decides (``judge``):

- persistence: the earlier passes at the candidate's peak (3x3 mean of band 2, the best of the 3x3 around the peak:
  20 m of misregistration) against the sea of the earlier passes. A wind turbine, a platform, a beacon or a moored
  hulk is bright on most days ("fixed"); a ship under way is not ("transient"). Too few earlier passes: "unknown".
- size at 20 m: the pixels joined to the peak above max(sea + 6 dB, peak - 12 dB); length and width along the axes.
- the image: the false colour of the pass at 20 m for the site (the photo the decision was made on).

Nothing is fetched for a candidate near a point already learned as fixed. Every request is counted in the caller's
PU; the per-pass and per-run caps are in ``monitor.json`` (``verify``).
"""
from __future__ import annotations

import math
import warnings
from datetime import timedelta

import numpy as np

from sat import cdse, color, png
from sat.grid import Grid, label, m_per_deg

VERSION = "ocean-verify-1"
SIZE_PX = 64            # 64 px x 20 m = 1.28 km square
PIXEL_M = 20.0
DEFAULTS = {
    "enabled": True,
    "max_per_pass": 15,      # A candidates verified per pass (strongest VV contrast first)
    "lookback_days": 48,
    "max_prior": 8,          # earlier passes kept by the evalscript (newest first)
    "quantile": 0.3,         # band 2: the 30th percentile of the earlier passes
    "min_prior": 3,          # fewer earlier passes with data at the peak: "unknown"
    "fixed_db": 8.0,         # persistence >= this over the sea of the earlier passes: fixed
    "transient_db": 4.0,     # <= this: transient (a ship that was not there)
    "peak_search_px": 4,     # the peak is searched within 80 m of the 40 m position
    "small_m": 50.0,         # shorter than this at 20 m: flagged (A -> B only with small_demote, after a review)
    "small_demote": False,
    "est_pu": 0.3,           # planning value per request (64 px, 2 bands, x2 orthorectification, <= 9 passes)
}
VERDICTS = ("fixed", "transient", "uncertain", "unknown")

EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask"]}],
          output: {bands: 4, sampleType: "UINT16"},
          mosaicking: "ORBIT"};
}
function preProcessScenes(collections) {
  collections.scenes.orbits.sort(function (a, b) { return new Date(b.dateFrom) - new Date(a.dateFrom); });
  collections.scenes.orbits = collections.scenes.orbits.slice(0, %(keep)d);
  return collections;
}
function enc(x) {
  if (!(x > 0)) return 0;
  return Math.max(1, Math.min(65535, Math.round((10 * Math.log(x) / Math.LN10 + %(off)s) * %(scale)s)));
}
function evaluatePixel(samples) {
  if (!samples.length || !samples[0].dataMask) return [0, 0, 0, 0];
  var prior = [];
  for (var i = 1; i < samples.length; i++) {
    if (samples[i].dataMask && samples[i].VV > 0) prior.push(samples[i].VV);
  }
  prior.sort(function (a, b) { return a - b; });
  var q = prior.length ? prior[Math.floor(%(q)s * (prior.length - 1))] : 0;
  return [enc(samples[0].VV), enc(samples[0].VH), enc(q), prior.length];
}
"""


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("verify") or {})}


def evalscript(keep: int, q: float) -> str:
    return EVALSCRIPT % {"keep": int(keep), "q": float(q), "off": cdse.DB_OFFSET, "scale": cdse.DB_SCALE}


def tile_grid(lon: float, lat: float) -> Grid:
    """1.28 km square at 20 m centred on the candidate."""
    my, mx = m_per_deg(lat)
    half = SIZE_PX * PIXEL_M / 2
    return Grid(lon - half / mx, lat - half / my, lon + half / mx, lat + half / my, SIZE_PX, SIZE_PX)


def payload(g: Grid, acq: dict, st: dict) -> dict:
    t_to = acq["t_to"]
    t_from = t_to - timedelta(days=float(st["lookback_days"]))
    return {
        "input": {
            "bounds": {"bbox": g.bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {"timeRange": {"from": cdse.iso(t_from), "to": cdse.iso(t_to)},
                               "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH"},
                "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True, "demInstance": "COPERNICUS_30",
                               "upsampling": "NEAREST", "downsampling": "NEAREST"},
            }],
        },
        "output": {"width": g.width, "height": g.height,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": evalscript(int(st["max_prior"]) + 1, float(st["quantile"])),
    }


def decode(data: bytes) -> dict:
    arr = png.decode(data).astype(np.float64)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise cdse.CDSEError(f"想定外の画像（形 {arr.shape}）")

    def db(ch):
        return np.where(ch > 0, ch / cdse.DB_SCALE - cdse.DB_OFFSET, np.nan)
    return {"vv": db(arr[..., 0]), "vh": db(arr[..., 1]), "vv_q": db(arr[..., 2]), "n_prior": arr[..., 3]}


def _mean3(a: np.ndarray) -> np.ndarray:
    """3x3 mean of finite values (NaN where none)."""
    p = np.pad(a, 1, constant_values=np.nan)
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _extent(rr: np.ndarray, cc: np.ndarray, px: float) -> tuple[float, float, float | None]:
    """Length, width (m, one pixel of footprint included) and the axis bearing (0-180 deg from north)."""
    if len(rr) < 2:
        return px, px, None
    y, x = -rr * px, cc * px
    pts = np.stack([x - x.mean(), y - y.mean()])
    vals, vecs = np.linalg.eigh(pts @ pts.T / len(rr))
    a, b = vecs[:, 1], vecs[:, 0]
    pa, pb = a @ pts, b @ pts
    bearing = math.degrees(math.atan2(a[0], a[1])) % 180.0
    return float(pa.max() - pa.min() + px), float(pb.max() - pb.min() + px), round(bearing, 1)


def judge(arr: dict, st: dict) -> dict:
    """The decision on one cut-out (see the module text). Never raises on a blank cut-out."""
    vv, vq, n = arr["vv"], arr["vv_q"], arr["n_prior"]
    h, w = vv.shape
    cy, cx = h // 2, w // 2
    out = {"version": VERSION, "verdict": "unknown", "n_prior": 0, "persist_db": None, "length_m_20m": None,
           "width_m_20m": None, "heading_deg_20m": None, "peak_vv_db_20m": None, "sea_vv_db_20m": None}
    if not np.isfinite(vv).any():
        out["reason"] = "no_data"
        return out
    k = int(st["peak_search_px"])
    win = vv[max(0, cy - k):cy + k + 1, max(0, cx - k):cx + k + 1]
    if not np.isfinite(win).any():
        out["reason"] = "no_data_at_candidate"
        return out
    j, i = np.unravel_index(np.nanargmax(win), win.shape)
    pr, pc = max(0, cy - k) + int(j), max(0, cx - k) + int(i)
    peak = float(vv[pr, pc])
    yy, xx = np.mgrid[0:h, 0:w]
    far = (np.hypot(yy - pr, xx - pc) > 10) & np.isfinite(vv)
    sea = float(np.nanmedian(vv[far])) if far.any() else float(np.nanmedian(vv))
    out.update(peak_vv_db_20m=round(peak, 1), sea_vv_db_20m=round(sea, 1))
    # the target at 20 m: joined to the peak, within 20 px
    thr = max(sea + 6.0, peak - 12.0)
    lab, _ = label(np.nan_to_num(vv, nan=-99.0) >= thr, connectivity=8)
    m = (lab == lab[pr, pc]) & (np.hypot(yy - pr, xx - pc) <= 20)
    rr, cc = np.nonzero(m)
    length, width, bearing = _extent(rr, cc, PIXEL_M)
    out.update(length_m_20m=round(length), width_m_20m=round(width), heading_deg_20m=bearing)
    # persistence in the earlier passes
    npk = int(np.nanmax(n[max(0, pr - 1):pr + 2, max(0, pc - 1):pc + 2]))
    out["n_prior"] = npk
    if npk < int(st["min_prior"]):
        out["reason"] = "few_prior_passes"
        return out
    q3 = _mean3(vq)
    farq = far & np.isfinite(vq)
    sea_q = float(np.nanmedian(vq[farq])) if farq.any() else float("nan")
    near_q = q3[max(0, pr - 1):pr + 2, max(0, pc - 1):pc + 2]
    if not (np.isfinite(sea_q) and np.isfinite(near_q).any()):
        out["reason"] = "no_prior_data"
        return out
    persist = float(np.nanmax(near_q)) - sea_q
    out["persist_db"] = round(persist, 1)
    out["verdict"] = ("fixed" if persist >= float(st["fixed_db"]) else
                      "transient" if persist <= float(st["transient_db"]) else "uncertain")
    return out


def tile_rgb(arr: dict) -> np.ndarray:
    """False colour of the pass at 20 m (the port images' colour; VH NaN with VV data = below the noise floor)."""
    vv, vh = arr["vv"], arr["vh"]
    vh = np.where(np.isfinite(vh), vh, np.where(np.isfinite(vv), color.VH_RANGE[0], np.nan))
    return color.false_color(vv, vh, filtered=True)


def select(cands: list[dict], learned_points: list[dict], st: dict, radius_m: float) -> list[dict]:
    """The A candidates to verify: strongest VV contrast first, none near a point already learned as fixed."""
    from ocean.monitor import near  # noqa: PLC0415 - avoid the import cycle at load time
    pool = [c for c in cands if c.get("_class") == "A"
            and not near(learned_points, c["lon"], c["lat"], radius_m)]
    pool.sort(key=lambda c: -(c.get("contrast_vv_db") or 0.0))
    return pool[:max(0, int(st["max_per_pass"]))]


def fetch(client, c: dict, acq: dict, st: dict) -> tuple[dict, float]:
    """(arrays, PU) of one candidate. Raises cdse.CDSEError on a failed request."""
    g = tile_grid(c["lon"], c["lat"])
    data, pu = client.process_raw(payload(g, acq, st), "image/png", 2)
    return decode(data), pu
