"""Probe helper: shrink the newest quick-look to a small PNG and list its candidates.

    python -m sat.preview OUT_DIR DEST_DIR [FACTOR]

Writes DEST_DIR/preview.png (block-averaged grey, overlays kept), DEST_DIR/color_thumb.webp
(the false-colour thumbnail, if the scene has one) and DEST_DIR/candidates.txt. Used to read probe results through annotations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import png, render
from .run import read_csv


def shrink(index_img: np.ndarray, factor: int = 4, levels: int = 32) -> np.ndarray:
    h, w = index_img.shape
    h2, w2 = h // factor, w // factor
    blocks = index_img[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).swapaxes(1, 2).reshape(h2, w2, -1)
    grey = np.where(blocks < render.GREYS, blocks, 0).astype(float)
    n = (blocks < render.GREYS).sum(-1)
    mean = np.where(n > 0, grey.sum(-1) / np.maximum(n, 1), 0)
    step = render.GREYS / levels
    out = (np.floor(mean / step) * step).astype(np.uint8)
    for colour, need in ((render.CYAN, factor), (render.ORANGE, factor), (render.MAGENTA, 1), (render.YELLOW, 1), (render.RED, 1)):
        out[(blocks == colour).sum(-1) >= need] = colour
    out[(blocks == render.NODATA).sum(-1) > factor * factor // 2] = render.NODATA
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out, dest = Path(argv[0]), Path(argv[1])
    factor = int(argv[2]) if len(argv) > 2 else 4
    dest.mkdir(parents=True, exist_ok=True)
    rows = read_csv(out / "ship_counts.csv")
    if not rows:
        (dest / "candidates.txt").write_text("no counted scene\n", encoding="utf-8")
        return 0
    rows.sort(key=lambda r: int(r["harbour_candidates"] or 0) + int(r["quay_candidates"] or 0))
    row = sorted(rows, key=lambda r: r["acquired_utc"])[-1]
    img = png.decode((out / row["image"]).read_bytes())
    (dest / "preview.png").write_bytes(png.encode(shrink(img, factor), palette=render.PALETTE))
    if row.get("color_thumb") and (out / row["color_thumb"]).exists():
        (dest / "color_thumb.webp").write_bytes((out / row["color_thumb"]).read_bytes())
    lines = ["scene " + ",".join(f"{k}={row[k]}" for k in ("acquired_utc", "platform", "orbit_direction", "relative_orbit",
                                                         "incidence_deg", "harbour_candidates", "quay_candidates", "coverage"))]
    lines += ["all " + r["acquired_utc"] + f" {r['platform']} {r['orbit_direction']} ro={r['relative_orbit']} inc={r['incidence_deg']}"
              f" H={r['harbour_candidates']} Q={r['quay_candidates']}"
              f" A={r.get('sector_a_harbour', '')}+{r.get('sector_a_quay', '')} B={r.get('sector_b_harbour', '')}+{r.get('sector_b_quay', '')}"
              f" mask={r.get('mask_version', '')} cov={r['coverage']} pu={r['processing_units']}"
              for r in sorted(rows, key=lambda r: r["acquired_utc"])]
    gj = out / "detections" / f"{row['acquired_utc'][:7]}.geojson"
    if gj.exists():
        for f in json.loads(gj.read_text(encoding="utf-8"))["features"]:
            p = f["properties"]
            if p["acquired_utc"] == row["acquired_utc"] and p.get("mask_version", "") == row.get("mask_version", p.get("mask_version", "")):
                lon, lat = f["geometry"]["coordinates"]
                lines.append(f"{p['n']} {p['zone']}/{p.get('sector')} {lat:.5f},{lon:.5f} L={p['length_m']} W={p['width_m']} px={p['pixels']}"
                             f" peak={p['peak_vv_db']} c={p['contrast_vv_db']} vh={p['vh_ratio_db']}")
    (dest / "candidates.txt").write_text("\n".join(lines)[:3900] + "\n", encoding="utf-8")
    quay_report(out, dest)
    return 0


def quay_report(out: Path, dest: Path) -> None:
    """DEST/quay.txt (every scene's quay count, the newest scene's stretches), DEST/quay_segs.json (those
    stretches with every property) and DEST/change_thumb.webp."""
    rows = sorted(read_csv(out / "quay_counts.csv"), key=lambda r: r["acquired_utc"])
    if not rows:
        (dest / "quay.txt").write_text("no quay count\n", encoding="utf-8")
        return
    lines = [f"{r['acquired_utc']} {r['platform']} ro={r['relative_orbit']} base={r['baseline_version']} n={r['baseline_scenes']}"
             f" cred={r['credibility']} occ={r['occupied_min']}-{r['occupied_max']} A={r['sector_a_min']}-{r['sector_a_max']}"
             f" B={r['sector_b_min']}-{r['sector_b_max']} seg={r['segments']} raft={r['rafted']} cov={r['quay_coverage']}"
             + (f" est={r.get('est_mode')}[{r.get('est_p10')}-{r.get('est_p90')}] m={r.get('est_mean')} war={r.get('est_warship_mode')}"
                f" eA={r.get('est_a_mode')}[{r.get('est_a_p10')}-{r.get('est_a_p90')}] eB={r.get('est_b_mode')}[{r.get('est_b_p10')}-{r.get('est_b_p90')}]"
                f" cf={r.get('est_conflicts')}" if r.get("est_version") else "") for r in rows]
    row = rows[-1]
    if row.get("change_thumb") and (out / row["change_thumb"]).exists():
        (dest / "change_thumb.webp").write_bytes((out / row["change_thumb"]).read_bytes())
    gj = out / "quay" / f"{row['acquired_utc'][:7]}.geojson"
    segs = []
    if gj.exists():
        for f in json.loads(gj.read_text(encoding="utf-8"))["features"]:
            p = f["properties"]
            if p["acquired_utc"] == row["acquired_utc"]:
                segs.append({k: v for k, v in p.items() if k not in ("site_id", "acquired_utc", "status", "reliability",
                                                                      "credibility", "mask_version", "quay_version",
                                                                      "baseline_version", "in_sample", "est_version")})
                lon, lat = f["geometry"]["coordinates"][len(f["geometry"]["coordinates"]) // 2]
                lines.append(f"{p['n']} {p['sector']} {p['line_kind']} {lat:.5f},{lon:.5f} L={p['length_m']} est={p['est_length_m']}"
                             f" {p['min']}-{p['max']}{' raft' if p['rafted'] else ''} pk={p['peak_dvv_db']} cl={p.get('cluster')}")
    (dest / "quay.txt").write_text("\n".join(lines)[:3900] + "\n", encoding="utf-8")
    # the newest scene's stretches with every property (est-1 inputs), for re-computing offline
    (dest / "quay_segs.json").write_text(json.dumps({"acquired_utc": row["acquired_utc"], "segments": segs},
                                                    ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
