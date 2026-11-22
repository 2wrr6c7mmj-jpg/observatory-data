"""The pictures of the queue page (confirm-1, design §6.2): one natural-colour WebP per port and day.

    confirm/img/<site>/<date>.webp     the whole port, Sentinel-2 L2A natural colour (the same look as s2nc: R = B4, G = B3,
                                       B = B2, reflectance 0-0.3, gamma 1/2.2, the strip with the credit below)
    confirm/img/<site>/index.json      date -> file and grid (what sat/confirm.py puts into queue.json)

The bands come from whoever fetched the window (the daily judge; Element84 Earth Search / AWS Open Data, PU 0): this module
only draws, and it does not fetch anything. needed() says which days the queue wants a picture for (the days of its cells and
the days of the last answers that have no picture yet); prune() removes the pictures nothing refers to any more.
"""
from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path

from . import atomic, confirm

SOURCE = "Element84 Earth Search / AWS Open Data"


def img_dir(out: Path, site_id: str) -> Path:
    return Path(out) / confirm.DIR / "img" / site_id


def load_index(out: Path, site_id: str) -> dict:
    p = img_dir(out, site_id) / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"version": confirm.VERSION, "site_id": site_id, "days": {}}


def write_image(out: Path, site_id: str, date: str, refl, grid: dict, scl=None, name: str | None = None) -> Path:
    """Draw and store the picture of one port and day. refl: (h, w, >= 3) reflectance x 10000 in the order B2, B3, B4 (as
    sat/s2.py); grid: {"west", "south", "east", "north"} of the array in degrees (width and height are taken from it)."""
    import numpy as np
    from . import s2
    refl = np.asarray(refl)
    h, w = refl.shape[:2]
    for k in ("west", "south", "east", "north"):
        float(grid[k])
    rgb = s2.natural_color(refl, scl)
    label = f"{name or site_id}  Sentinel-2 L2A natural colour  {date}"
    webp, _thumb = s2.render(rgb, label, f"{s2.credit(date[:4])} ({SOURCE})")
    path = img_dir(out, site_id) / f"{date}.webp"
    atomic.write_bytes(path, webp)
    ix = load_index(out, site_id)
    ix["days"][date] = {"image": path.name, "grid": {"west": float(grid["west"]), "south": float(grid["south"]),
                                                      "east": float(grid["east"]), "north": float(grid["north"]), "width": w, "height": h}}
    ix["days"] = dict(sorted(ix["days"].items()))
    atomic.write_text(img_dir(out, site_id) / "index.json", json.dumps(ix, ensure_ascii=False, indent=1) + "\n")
    return path


def _referenced(queue: dict) -> set:
    """(site, day) of the queue's cells and of their last answers, and of the days held back for want of a picture."""
    ref = set()
    for c in queue.get("cells", []):
        ref.add((c["site_id"], c["date"]))
        if c.get("prev"):
            ref.add((c["site_id"], c["prev"]["date"]))
    for h in queue.get("held", []):
        ref.add((h["site_id"], h["date"]))
        ref |= {(h["site_id"], d) for d in h.get("prev_dates", [])}
    return ref


def needed(out: Path, queue: dict) -> list[tuple[str, str]]:
    """(site, day) the queue refers to and has no picture for (neither here nor in the review set of the port)."""
    have = set()
    for site_id in {s for s, _ in _referenced(queue)}:
        have |= {(site_id, d) for d in confirm._images(out, site_id)}
    return sorted(_referenced(queue) - have)


def prune(out: Path, queue: dict, today: str, keep_days: int = 90) -> list[str]:
    """Remove the pictures older than keep_days that the queue does not refer to (design §6.2). Returns what was removed."""
    ref = _referenced(queue)
    gone = []
    base = Path(out) / confirm.DIR / "img"
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir():
            continue
        ix = load_index(out, d.name)
        drop = [day for day in ix["days"] if (d.name, day) not in ref and (_date.fromisoformat(today[:10]) - _date.fromisoformat(day)).days > keep_days]
        for day in drop:
            f = d / ix["days"].pop(day)["image"]
            if f.exists():
                f.unlink()
            gone.append(f"{d.name}/{day}")
        if drop:
            atomic.write_text(d / "index.json", json.dumps(ix, ensure_ascii=False, indent=1) + "\n")
    return gone
