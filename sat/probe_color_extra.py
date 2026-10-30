"""Probe-only (not for the PR): stats of a raw chip and a compact 2x2 pack for offline tuning.

    python -m sat.probe_color_extra RAW_DIR DEST
"""
import json
import lzma
import sys
from pathlib import Path

import numpy as np

from sat import color
from sat.osm_mask import load_mask
from sat.run import decode_chip


def pct(a):
    a = a[np.isfinite(a)]
    if not a.size:
        return None
    return [round(float(x), 2) for x in np.percentile(a, [1, 5, 25, 50, 75, 95, 99])]


def main():
    raw_dir, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    raws = sorted(raw_dir.glob("*.png"))
    if not raws:
        (dest / "stats.txt").write_text("no raw chip\n")
        return
    raw = raws[-1]
    vv, vh, _ = decode_chip(raw.read_bytes())
    mask, _m = load_mask("ZJG", 2)
    land = mask > 0
    fv, fh = color.lee(vv), color.lee(vh)
    st = {"chip": raw.name, "shape": list(vv.shape), "pct": [1, 5, 25, 50, 75, 95, 99]}
    for name, a, b in (("raw", vv, vh), ("lee", fv, fh)):
        for zone, m in (("water", ~land), ("land", land)):
            st[f"{name}_{zone}_vv"] = pct(a[m])
            st[f"{name}_{zone}_vh"] = pct(b[m])
            st[f"{name}_{zone}_ratio"] = pct((a - b)[m])
    bright = (vv > -5) & ~land
    st["water_bright_px"] = int(bright.sum())
    st["water_bright_vh"] = pct(vh[bright])
    st["water_bright_ratio"] = pct((vv - vh)[bright])
    (dest / "stats.txt").write_text(json.dumps(st, separators=(",", ":")) + "\n")
    # 2x2 block mean in linear power, 0.25 dB steps (uint8: dB*4 + 200, 0 = no data)
    h, w = vv.shape
    h2, w2 = h // 2, w // 2
    pack = []
    for a in (vv, vh):
        lin = np.where(np.isfinite(a), 10 ** (a / 10), np.nan)[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2)
        m = np.nanmean(lin, axis=(1, 3))
        db = 10 * np.log10(m)
        q = np.where(np.isfinite(db), np.clip(np.round(db * 4 + 200), 1, 255), 0).astype(np.uint8)
        pack.append(q)
    hdr = np.array([h2, w2], dtype=np.uint16).tobytes()
    (dest / "pack.xz").write_bytes(lzma.compress(hdr + np.stack(pack).tobytes(), preset=9 | lzma.PRESET_EXTREME))
    print("pack", (dest / "pack.xz").stat().st_size)


if __name__ == "__main__":
    main()
