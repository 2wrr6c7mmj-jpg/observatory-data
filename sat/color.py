"""Display image of a chip: Sentinel-1 false colour (VV, VH, VV-VH) as WebP, legend below.

R = VV, G = VH, B = VV-VH (reversed: a small ratio is blue). The ranges are fixed
for all sites and all dates (no per-scene stretch), so colours can be compared
between scenes. A 3x3 Lee filter is applied for display only; detection uses the
raw sigma0. Candidate boxes are not burned in (the trial page draws them from the
GeoJSON).

    python -m sat.color RAW.png OUT_DIR [LABEL]   # re-render from a chip saved by a probe

Pillow is only needed here (WebP and the legend text).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

from .grid import box_sum

COLOR_VERSION = "s1fc-1"
# Fixed display ranges in dB (lo -> 0, hi -> 255). The ratio range is reversed.
VV_RANGE = (-22.0, 0.0)
VH_RANGE = (-28.0, -6.0)
RATIO_RANGE = (15.0, 3.0)
# out = n ** (1 / gamma) per channel (R, G, B). B 0.5 (2026-10-30, ZJG 2026-10-26 S1C asc RO157):
# open water sits near the VH noise floor, so its VV-VH (median 6.4 dB) made the whole bay
# saturated blue at gamma 1; 0.5 turns it navy and leaves land near neutral. Ranges unchanged.
GAMMA = (1.0, 1.0, 0.5)
LEE_HALF = 1                  # 3x3 window
LEE_ENL = 4.4                 # equivalent number of looks of Sentinel-1 IW GRDH
NODATA_RGB = (0, 0, 0)
WEBP_QUALITY = 75
WEBP_METHOD = 6
THUMB_MAX = 320               # thumbnail: fits in 320 x 320 (width 320 for a square chip)
THUMB_QUALITY = 70
LEGEND_MIN_WIDTH = 300
BG = (17, 17, 17)
FG = (235, 235, 235)


def lee(db: np.ndarray, half: int = LEE_HALF, enl: float = LEE_ENL) -> np.ndarray:
    """Lee filter of a dB image in linear power (NaN = no data, kept as NaN)."""
    ok = np.isfinite(db)
    z = np.where(ok, 10.0 ** (np.where(ok, db, 0.0) / 10.0), 0.0)
    n = box_sum(ok.astype(np.float64), half)
    nn = np.maximum(n, 1)
    m = box_sum(z, half) / nn
    v = np.maximum(box_sum(z * z, half) / nn - m * m, 0.0)
    cu2 = 1.0 / enl
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(v > 0, 1.0 - cu2 * m * m / v, 0.0)
    k = np.clip(k, 0.0, 1.0)
    x = m + k * (z - m)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(x)
    out[~ok | ~np.isfinite(out)] = np.nan
    return out


def scale(db: np.ndarray, rng: tuple[float, float], gamma: float = 1.0) -> np.ndarray:
    lo, hi = rng
    n = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        n = n ** (1.0 / gamma)
    return n


def false_color(vv_db: np.ndarray, vh_db: np.ndarray, filtered: bool = True) -> np.ndarray:
    """uint8 RGB (h, w, 3) with the fixed ranges; no data is NODATA_RGB."""
    if filtered:
        vv_db, vh_db = lee(vv_db), lee(vh_db)
    ok = np.isfinite(vv_db) & np.isfinite(vh_db)
    vv = np.where(ok, vv_db, VV_RANGE[0])
    vh = np.where(ok, vh_db, VH_RANGE[0])
    ch = [scale(vv, VV_RANGE, GAMMA[0]), scale(vh, VH_RANGE, GAMMA[1]), scale(vv - vh, RATIO_RANGE, GAMMA[2])]
    rgb = np.round(np.stack(ch, axis=-1) * 255).astype(np.uint8)
    rgb[~ok] = NODATA_RGB
    return rgb


def _font(size: int = 10):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size)
    except TypeError:  # older Pillow without sizes
        return ImageFont.load_default()


def _fmt(x: float) -> str:
    return f"{x:g}"  # ASCII minus: the default font has no U+2212


def legend(width: int, label: str, credit: str):
    """Legend image (Pillow) of the three channels, the label and the credit."""
    from PIL import Image, ImageDraw
    font = _font(10)
    rows = [("R  VV", VV_RANGE, 0), ("G  VH", VH_RANGE, 1), ("B  VV-VH", RATIO_RANGE, 2)]
    lh = 14
    h = 6 + lh * len(rows) + 4 + lh * 2 + 4
    im = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(im)
    x_name, x_lo = 4, 62
    x_bar0, x_bar1 = x_lo + 26, width - 50
    for i, (name, rng, c) in enumerate(rows):
        y = 6 + i * lh
        d.text((x_name, y), name, fill=FG, font=font)
        d.text((x_lo, y), _fmt(rng[0]), fill=FG, font=font)
        d.text((x_bar1 + 4, y), _fmt(rng[1]) + " dB", fill=FG, font=font)
        n = np.linspace(0.0, 1.0, max(2, x_bar1 - x_bar0)) ** (1.0 / GAMMA[c])
        bar = np.zeros((lh - 4, n.size, 3), np.uint8)
        bar[..., c] = np.round(n * 255).astype(np.uint8)
        im.paste(Image.fromarray(bar, "RGB"), (x_bar0, y + 1))
    y = 6 + lh * len(rows) + 2
    d.text((4, y), label, fill=FG, font=font)
    d.text((4, y + lh), credit, fill=FG, font=font)
    return im


def render(vv_db: np.ndarray, vh_db: np.ndarray, label: str, credit: str) -> tuple[bytes, bytes]:
    """(WebP with legend below, WebP thumbnail without legend)."""
    from PIL import Image
    rgb = false_color(vv_db, vh_db)
    chip = Image.fromarray(rgb, "RGB")
    w, h = chip.size
    lg = legend(max(w, LEGEND_MIN_WIDTH), label, credit)
    full = Image.new("RGB", (lg.width, h + lg.height), BG)
    full.paste(chip, (0, 0))
    full.paste(lg, (0, h))
    buf = io.BytesIO()
    full.save(buf, "WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
    th = chip.copy()
    th.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
    tbuf = io.BytesIO()
    th.save(tbuf, "WEBP", quality=THUMB_QUALITY, method=WEBP_METHOD)
    return buf.getvalue(), tbuf.getvalue()


CHANGE_VERSION = "s1chg-1"


def change_color(vv_now_db: np.ndarray, vv_base_db: np.ndarray) -> np.ndarray:
    """uint8 RGB: R = this scene's VV, G = B = the baseline VV (both 3x3 means, VV_RANGE).

    Brighter now than in the baseline (a ship arrived) is red; darker (gone) is cyan;
    unchanged is grey. Where the baseline has no data (inland), G = B = R (grey)."""
    from .baseline import smooth_db
    now = smooth_db(vv_now_db)
    base = np.where(np.isfinite(vv_base_db), vv_base_db, now)
    ok = np.isfinite(now)
    r = scale(np.where(ok, now, VV_RANGE[0]), VV_RANGE)
    g = scale(np.where(np.isfinite(base), base, VV_RANGE[0]), VV_RANGE)
    rgb = np.round(np.stack([r, g, g], axis=-1) * 255).astype(np.uint8)
    rgb[~ok] = NODATA_RGB
    return rgb


def change_legend(width: int, label: str, credit: str):
    from PIL import Image, ImageDraw
    font = _font(10)
    lh = 14
    h = 6 + lh * 3 + 4 + lh * 2 + 4
    im = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(im)
    x0, x1 = 88, width - 50
    n = np.linspace(0.0, 1.0, max(2, x1 - x0))
    for i, (name, ch) in enumerate((("R  VV now", (0,)), ("G,B VV base", (1, 2)))):
        y = 6 + i * lh
        d.text((4, y), name, fill=FG, font=font)
        bar = np.zeros((lh - 4, n.size, 3), np.uint8)
        for c in ch:
            bar[..., c] = np.round(n * 255).astype(np.uint8)
        im.paste(Image.fromarray(bar, "RGB"), (x0, y + 1))
        d.text((x0 - 26, y), _fmt(VV_RANGE[0]), fill=FG, font=font)
        d.text((x1 + 4, y), _fmt(VV_RANGE[1]) + " dB", fill=FG, font=font)
    y = 6 + 2 * lh
    d.rectangle((4, y + 2, 14, y + 11), fill=(255, 40, 40))
    d.text((18, y), "brighter now (arrived)", fill=FG, font=font)
    d.rectangle((150, y + 2, 160, y + 11), fill=(40, 220, 220))
    d.text((164, y), "darker now", fill=FG, font=font)
    y = 6 + lh * 3 + 2
    d.text((4, y), label, fill=FG, font=font)
    d.text((4, y + lh), credit, fill=FG, font=font)
    return im


def render_change(vv_now_db: np.ndarray, vv_base_db: np.ndarray, label: str, credit: str) -> tuple[bytes, bytes]:
    """(WebP of the change image with legend below, WebP thumbnail)."""
    from PIL import Image
    chip = Image.fromarray(change_color(vv_now_db, vv_base_db), "RGB")
    w, h = chip.size
    lg = change_legend(max(w, LEGEND_MIN_WIDTH), label, credit)
    full = Image.new("RGB", (lg.width, h + lg.height), BG)
    full.paste(chip, (0, 0))
    full.paste(lg, (0, h))
    buf = io.BytesIO()
    full.save(buf, "WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
    th = chip.copy()
    th.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
    tbuf = io.BytesIO()
    th.save(tbuf, "WEBP", quality=THUMB_QUALITY, method=WEBP_METHOD)
    return buf.getvalue(), tbuf.getvalue()


def change_settings() -> dict:
    return {
        "version": CHANGE_VERSION,
        "channels": {"R": "今回の VV", "G": "基準の VV（同じ相対軌道の 10 パーセンタイル）", "B": "G と同じ"},
        "range_db": list(VV_RANGE),
        "filter": "両方とも 3×3 の平均（線形の σ0）。基準と同じ処理",
        "reading": "赤＝今回の方が明るい（入港・新しい物）、水色＝今回の方が暗い（出港）、灰＝変化なし。"
                   "基準が 10 パーセンタイル（船のいない状態）なので水色は出にくく、全体がわずかに赤みを帯びる",
        "no_baseline": "基準のない画素（内陸）は今回の VV の灰",
        "format": f"WebP（品質 {WEBP_QUALITY}）、凡例は画像の下。縮小版は {THUMB_MAX}×{THUMB_MAX} に収める",
    }


def settings() -> dict:
    """Description of the display for meta.json."""
    return {
        "version": COLOR_VERSION,
        "channels": {"R": {"band": "VV", "range_db": list(VV_RANGE), "gamma": GAMMA[0]},
                     "G": {"band": "VH", "range_db": list(VH_RANGE), "gamma": GAMMA[1]},
                     "B": {"band": "VV-VH", "range_db": list(RATIO_RANGE), "gamma": GAMMA[2],
                           "note": "逆向き：比が小さいほど青"}},
        "fixed_ranges": "全地点・全日で固定（シーンごとに引き伸ばさない）",
        "filter": f"表示用だけ {2 * LEE_HALF + 1}×{2 * LEE_HALF + 1} の Lee フィルタ（ENL {LEE_ENL}、線形の σ0 で）。検出は生の σ0",
        "format": f"WebP（非可逆、品質 {WEBP_QUALITY}）、凡例は画像の下。縮小版は {THUMB_MAX}×{THUMB_MAX} に収める（品質 {THUMB_QUALITY}、凡例なし）",
        "no_data": "黒",
    }


def main(argv=None) -> int:
    """Re-render from a raw chip (the UINT16 PNG the Process API returns)."""
    from .run import decode_chip
    argv = argv if argv is not None else sys.argv[1:]
    raw, out = Path(argv[0]), Path(argv[1])
    label = argv[2] if len(argv) > 2 else raw.stem
    vv, vh, _ = decode_chip(raw.read_bytes())
    full, thumb = render(vv, vh, label, "Contains modified Copernicus Sentinel data")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{raw.stem}.webp").write_bytes(full)
    (out / f"{raw.stem}.thumb.webp").write_bytes(thumb)
    print(f"{raw.stem}: {len(full)} B, thumb {len(thumb)} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
