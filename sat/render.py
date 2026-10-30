"""Quick-look PNG of a chip: VV in grey, land-mask outline, candidate boxes."""
from __future__ import annotations

import numpy as np

from . import png
from .grid import dilate

GREYS = 240
RED, YELLOW, CYAN, NODATA, ORANGE = 240, 241, 242, 243, 244
PALETTE = [(i * 255 // (GREYS - 1),) * 3 for i in range(GREYS)] + [
    (255, 40, 40), (255, 220, 0), (0, 200, 255), (20, 20, 60), (255, 140, 0)]
VMIN, VMAX = -25.0, 5.0


def quicklook(vv_db: np.ndarray, mask: np.ndarray, dets: list[dict]) -> bytes:
    g = np.clip((np.nan_to_num(vv_db, nan=VMIN) - VMIN) / (VMAX - VMIN), 0, 1)
    img = np.round(g * (GREYS - 1)).astype(np.uint8)
    img[~np.isfinite(vv_db)] = NODATA
    land = mask > 0
    edge = dilate(land, 1) & ~land
    img[edge] = CYAN
    img[(mask == 2) & ~dilate(mask != 2, 1)] = ORANGE  # interior of structures
    h, w = img.shape
    for d in dets:
        colour = YELLOW if d["zone"] == "quay" else RED
        r0, r1 = max(0, d["rmin"] - 4), min(h - 1, d["rmax"] + 4)
        c0, c1 = max(0, d["cmin"] - 4), min(w - 1, d["cmax"] + 4)
        img[r0, c0:c1 + 1] = colour
        img[r1, c0:c1 + 1] = colour
        img[r0:r1 + 1, c0] = colour
        img[r0:r1 + 1, c1] = colour
    return png.encode(img, palette=PALETTE)
