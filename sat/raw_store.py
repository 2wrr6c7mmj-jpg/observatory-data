"""raw-1: keep the Sentinel-1 chips of the daily run for a short time, on data branches (not on main).

The daily run (sat-ships, ``python -m sat.run --keep-raw DIR``) writes, for every counted scene of every site:

    DIR/<site>/<acquired stamp>.png    the CDSE Process API response as received (16-bit RGBA PNG: VV, VH,
                                       local incidence angle, data mask; dB = value / 800 - 50, 0 = no data)
    DIR/<site>/<acquired stamp>.json   grid (bbox, size, pixel_m), acquisition, mask version, PU, and the
                                       cfar-v1 result (detector info and candidates) to check a re-run against
    DIR/masks/<site>-land-v<N>.png     the land mask array the detector was given (uint8: 0 water, 1 land, 2 structure)

No request is added: these are the bytes the run fetched anyway (``client.process``) and decoded for the count.

``push`` then commits DIR as one commit with no parent (its own history) to the branch ``sat-raw/<run date UTC>``
(a second run on the same day adds a commit on top of that day's branch) and deletes the ``sat-raw/YYYY-MM-DD``
branches older than ``--keep-days`` days. Main and docs/ (Pages) are never touched, and no history grows without
end: a deleted day branch shares no object with the others, so its chips go away with it.

    python -m sat.raw_store push --src DIR [--prefix sat-raw] [--keep-days 14]   # in the workflow
    python -m sat.raw_store fetch --dest DIR [--prefix sat-raw]                  # on a work machine (git fetch)
    python -m sat.raw_store check DIR                                            # re-run cfar-v1, compare

Reading one scene in Python: ``vv, vh, lia, land, meta = sat.raw_store.load(Path("DIR/ZJG/2026-11-01T10-48-45Z.json"))``
(vv, vh in dB with NaN = no data; land = mask > 0, exactly what sat.run gives sat.detect.detect).

Contains modified Copernicus Sentinel data [year]; land masks (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import png

VERSION = "raw-1"
PREFIX = "sat-raw"
KEEP_DAYS = 14
DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def stamp(acquired_utc: str) -> str:
    return acquired_utc.replace(":", "-")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o).__name__)


def save_scene(root: Path, site: dict, grid, acq: dict, data: bytes, *, mask: np.ndarray, mask_tag: str,
               with_lia: bool, processing_units: float, retrieved_utc: str, coverage: float,
               detector_version: str, detector_params: dict, info: dict, dets: list[dict]) -> Path:
    """Write one scene (the response bytes, a JSON beside it, the mask once per mask version). Returns the JSON path."""
    sdir = root / site["id"]
    sdir.mkdir(parents=True, exist_ok=True)
    name = stamp(acq["acquired_utc"])
    (sdir / f"{name}.png").write_bytes(data)
    mdir = root / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / f"{mask_tag}.png"
    if not mpath.exists():
        mpath.write_bytes(png.encode(np.asarray(mask, dtype=np.uint8)))
    keep = ("row", "col", "rmin", "rmax", "cmin", "cmax", "pixels", "length_m", "width_m", "length_class", "zone",
            "peak_vv_db", "contrast_vv_db", "vh_ratio_db", "lon", "lat")
    meta = {
        "version": VERSION,
        "site_id": site["id"],
        "acquired_utc": acq["acquired_utc"],
        "platform": acq.get("platform"),
        "orbit_direction": acq.get("orbit_direction"),
        "relative_orbit": acq.get("relative_orbit"),
        "catalog_ids": acq.get("ids"),
        "retrieved_utc": retrieved_utc,
        "processing_units": processing_units,
        "chip": f"{name}.png",
        "encoding": {"format": "PNG 16-bit RGBA", "bands": ["VV", "VH", "localIncidenceAngle" if with_lia else None, "dataMask"],
                     "db": "value / 800 - 50 (0 = no data)", "lia_deg": "value / 100", "with_lia": with_lia},
        "bbox": list(grid.bbox), "width": grid.width, "height": grid.height,
        "pixel_m": [float(p) for p in grid.pixel_m],
        "mask": f"../masks/{mask_tag}.png", "mask_version": mask_tag,
        "coverage": coverage,
        "detector_version": detector_version, "detector_params": detector_params,
        "detector_info": info,
        "candidates": [{k: d.get(k) for k in keep} for d in dets],
        "attribution": f"Contains modified Copernicus Sentinel data {acq['acquired_utc'][:4]}; "
                       "land mask (c) OpenStreetMap contributors, ODbL 1.0",
    }
    p = sdir / f"{name}.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default) + "\n", encoding="utf-8")
    return p


def load(meta_path: Path):
    """(vv_db, vh_db, lia_deg or None, land bool, meta) of one saved scene, decoded as sat.run does."""
    from .run import decode_chip  # same decoding as the daily run (lazy: sat.run imports this module)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    base = Path(meta_path).parent
    vv, vh, lia = decode_chip((base / meta["chip"]).read_bytes())
    land = png.decode((base / meta["mask"]).read_bytes()) > 0
    return vv, vh, lia, land, meta


def scenes(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("*/*.json") if p.parent.name != "masks")


def check(root: Path) -> list[dict]:
    """Re-run cfar-v1 on every saved scene; compare with the saved candidates (row, col, length_m)."""
    from .detect import detect
    out = []
    for p in scenes(root):
        vv, vh, _lia, land, meta = load(p)
        dets, _info = detect(vv, vh, land, tuple(meta["pixel_m"]), meta.get("detector_params"))
        a = sorted((d["row"], d["col"], d["length_m"]) for d in dets)
        b = sorted((d["row"], d["col"], d["length_m"]) for d in meta["candidates"])
        out.append({"scene": f"{meta['site_id']} {meta['acquired_utc']}", "saved": len(b), "rerun": len(a), "same": a == b})
    return out


# ---- git (the workflow and a work machine) ----

def _git(*args: str, env: dict | None = None, input: bytes | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], capture_output=True, env={**os.environ, **(env or {})}, input=input)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace')[:400]}")
    return r.stdout.decode(errors="replace")


def remote_days(prefix: str = PREFIX, remote: str = "origin") -> dict[str, str]:
    """{YYYY-MM-DD: sha} of the remote's <prefix>/<day> branches (others under the prefix are ignored)."""
    out = {}
    for line in _git("ls-remote", "--heads", remote, f"refs/heads/{prefix}/*").splitlines():
        sha, ref = line.split("\t")
        m = DAY_RE.match(ref[len(f"refs/heads/{prefix}/"):])
        if m:
            out[m.group(1)] = sha
    return out


def expired(days: list[str], today: str, keep_days: int) -> list[str]:
    """Days strictly older than ``keep_days`` days before ``today`` (today and the keep_days days before it stay)."""
    cut = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    return sorted(d for d in days if d < cut)


def readme(src: Path, prefix: str, day: str, keep_days: int, note: str = "") -> str:
    years = sorted({p.stem[:4] for p in Path(src).glob("*/*.png") if p.parent.name != "masks"})
    yrs = "–".join([years[0], years[-1]] if len(years) > 1 else years) or day[:4]
    return (f"# {prefix}/{day}（{VERSION}）\n\n"
            f"{note}"
            "毎日の Sentinel-1 処理（sat-ships）が数えた撮影の切り出しを、取得したままの形で残したもの。"
            f"main には入れない。{keep_days} 日より古い日のブランチは毎日の実行で消す。\n\n"
            "- `<地点>/<撮影>.png`：CDSE Process API の応答そのもの（16 bit RGBA：VV・VH・局所入射角・データマスク。dB = 値/800 − 50、0 は値なし）\n"
            "- `<地点>/<撮影>.json`：範囲・画素の大きさ・軌道・PU・陸マスクの版と、cfar-v1 の結果（再実行の照合用）\n"
            "- `masks/<地点>-land-v<N>.png`：検出に渡した陸マスク（0 水、1 陸、2 構造物）\n\n"
            "読み方：`python -m sat.raw_store fetch --dest DIR` → `sat.raw_store.load(DIR/<日>/<地点>/<撮影>.json)`、"
            "`python -m sat.raw_store check DIR/<日>` で cfar-v1 を回し直して照合。\n\n"
            f"Contains modified Copernicus Sentinel data {yrs}. 陸マスク：(c) OpenStreetMap contributors, ODbL 1.0\n")


def push(src: Path, prefix: str = PREFIX, keep_days: int = KEEP_DAYS, today: str | None = None,
         remote: str = "origin", note: str = "", log=lambda m: print(m, file=sys.stderr)) -> dict:
    """Commit ``src`` to <prefix>/<today> (no parent, or on top of that day's branch), then delete expired days."""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days = remote_days(prefix, remote)
    summary = {"branch": f"{prefix}/{today}", "files": 0, "bytes": 0, "deleted": [], "kept": []}
    files = sorted(p for p in Path(src).rglob("*") if p.is_file()) if Path(src).exists() else []
    if files:
        with tempfile.TemporaryDirectory() as td:
            env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
            parent = None
            if today in days:
                _git("fetch", "-q", "--depth=1", remote, f"refs/heads/{prefix}/{today}")
                parent = _git("rev-parse", "FETCH_HEAD").strip()
                _git("read-tree", parent, env=env)
            readme_p = Path(td) / "README.md"
            readme_p.write_text(readme(src, prefix, today, keep_days, note), encoding="utf-8")
            for p in [*files, readme_p]:
                rel = "README.md" if p == readme_p else p.relative_to(src).as_posix()
                blob = _git("hash-object", "-w", str(p)).strip()
                _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
                summary["bytes"] += p.stat().st_size
            summary["files"] = len(files)
            tree = _git("write-tree", env=env).strip()
            msg = f"Sentinel-1 切り出し {today}（{VERSION}、{len(files)} ファイル）"
            sha = _git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg).strip()
            for i in range(3):  # only this workflow writes these branches (one concurrency group)
                r = subprocess.run(["git", "push", "-q", remote, f"{sha}:refs/heads/{prefix}/{today}"], capture_output=True)
                if r.returncode == 0:
                    break
                log(f"push に失敗（{i + 1} 回目）: {r.stderr.decode(errors='replace')[:200]}")
            else:
                raise RuntimeError(f"{prefix}/{today} を push できなかった")
            summary["commit"] = sha
            days[today] = sha
    old = expired(list(days), today, keep_days)
    if old:
        _git("push", "-q", remote, "--delete", *[f"refs/heads/{prefix}/{d}" for d in old])
    summary["deleted"] = old
    summary["kept"] = sorted(d for d in days if d not in old)
    return summary


def fetch(dest: Path, prefix: str = PREFIX, remote: str = "origin") -> list[str]:
    """Copy every kept day into dest/<day>/ (fetches only those branches; HEAD and the worktree are not touched)."""
    days = remote_days(prefix, remote)
    done = []
    for day in sorted(days):
        d = Path(dest) / day
        _git("fetch", "-q", remote, f"refs/heads/{prefix}/{day}")
        names = _git("ls-tree", "-r", "--name-only", "FETCH_HEAD").splitlines()
        for name in names:
            p = d / name
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(subprocess.run(["git", "cat-file", "blob", f"FETCH_HEAD:{name}"], capture_output=True, check=True).stdout)
        done.append(day)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    p.add_argument("--today", help="YYYY-MM-DD (default: today UTC)")
    p.add_argument("--note", default="")
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--prefix", default=PREFIX)
    c = sub.add_parser("check")
    c.add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "push":
        print(json.dumps(push(a.src, a.prefix, a.keep_days, a.today, note=a.note), ensure_ascii=False))
    elif a.cmd == "fetch":
        print(json.dumps(fetch(a.dest, a.prefix)))
    else:
        res = check(a.dir)
        for r in res:
            print(json.dumps(r, ensure_ascii=False))
        return 0 if res and all(r["same"] for r in res) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
