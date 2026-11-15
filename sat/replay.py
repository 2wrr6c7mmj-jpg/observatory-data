"""Close pairs from the past: older Sentinel-1 scenes counted again for the clear Sentinel-2 days (probe only).

User 2026-11-15: "can't we do the same [the close pairs, sat/close_pairs.py] from the past?" -> all ports, about 350 PU.
The daily run keeps est-2 only from 2026-10-02, so the pairs of earlier S2 images need their S1 scene again. For every clear
S2 image of a port (s2_scenes.csv and the days of the eye check, labels/<site>/review.json) that is not paired yet:

1. the catalog (+-CLOSE_PAIR_H around the S2 time; requests, no PU) gives the S1 acquisitions; the nearest within
   CLOSE_PAIR_H whose relative orbit has an empty-berth baseline (no quay count without it) is the target;
2. the target is counted by the daily run's own code (sat/run.process_site, one scene, ``only``), on a copy of the
   published directory (``--work``): the chips already fetched by the recount probes (probe-raw/s1-recount/<site>/*.npz,
   sat/s1_recount.py; no local incidence angle) are used as they are (0 PU), the others are fetched (VV, VH and the local
   incidence angle, as the daily run);
3. the work directory is saved (est-2, naval50: sat/run.Store.save), and the pairs of the counted scenes are added to the
   published close_pairs.csv / close_pairs/<site>.json with ``source`` replay (sat/close_pairs.update). Nothing else of
   the published directory is changed. The work directory's quay files of the counted scenes are kept under ``--keep``
   (probe branch only), so est-2 can be run on them again after a recalibration.

The land mask of today is used for every scene (a site's mask_since does not hold the scenes back here: the replay is not
the daily series). A scene inside the baseline's own window is counted like the others (its row says so: in_sample).

Budget, as sat/s1_recount.py: the PU of this job (``MAX_PU_TOTAL`` over every run, the rows of ``--usage-out`` and of its
copies among ``--usage``) and what is left of the month's own caps (s1_recount.month_left); ``--budget-month`` only.
Targets are taken round-robin over the sites (nearest in time first), so a cap reached early does not leave a port out.

    python -m sat.replay --site ZJG QGZ ... --work W --keep probe-raw/replay --chips C1 C2 --budget-month 2026-11 \
        --global-cap 9000 --global-margin 450 --ocean-cap 7000 --usage U/*.csv --ports-usage P/*.csv \
        --usage-out docs/data/sat/replay_usage.csv [--max-minutes 150]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse, close_pairs, png
from .grid import Grid

PROBE_VERSION = "replay-1"
MAX_PU_TOTAL = 350.0          # the owner's consent 2026-11-15 ("全港 約350PU")
BUDGET_MONTH = "2026-11"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "sat"
USAGE_FIELDS = ("run_utc", "site_id", "probe_version", "requests", "processing_units", "pu_source", "scenes", "cap_pu")
KEEP_FILES = ("quay_counts.csv", "quay", "quay_est2.csv", "quay_est2.json", "quay_est2", "quay_naval50.csv", "quay_naval50",
              "quay_blind", "ship_counts.csv", "status.json")
CHIP_MATCH_S = 120


def chip_index(dirs: list[Path]) -> dict:
    """{site_id: [(time, npz path), ...]} of the recount probes' chips (<dir>/<site>/s1-<stamp>.npz with its .json)."""
    res: dict = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*/s1-*.npz")):
            meta = p.with_suffix(".json")
            acq = None
            if meta.exists():
                acq = json.loads(meta.read_text(encoding="utf-8")).get("acquired_utc")
            if not acq:
                stamp = p.stem[3:]
                acq = stamp[:13] + ":" + stamp[14:16] + ":" + stamp[17:]
            res.setdefault(p.parent.name, []).append((cdse.parse_time(acq), p))
    return res


def encode_chip(vv: np.ndarray, vh: np.ndarray) -> bytes:
    """The Process API's UINT16 PNG (VV, VH, data mask; dB = v / DB_SCALE - DB_OFFSET, 0 = no data) of a saved chip,
    for sat/run.decode_chip (three channels: no local incidence angle)."""
    def enc(a):
        v = np.where(np.isfinite(a), np.round((a + cdse.DB_OFFSET) * cdse.DB_SCALE), 0)
        return np.clip(v, 0, 65535).astype(np.uint16)
    ok = np.isfinite(vv) & np.isfinite(vh)
    arr = np.stack([np.where(ok, np.maximum(enc(vv), 1), 0), np.where(ok, np.maximum(enc(vh), 1), 0),
                    np.where(ok, 65535, 0)], axis=-1).astype(np.uint16)
    return png.encode(arr, depth=16)


class ReplayClient:
    """The client process_site sees for one target: the catalog answer of the plan (no new request) and either a saved
    chip (0 PU) or the real request."""

    def __init__(self, real, feats: list[dict], chips: list[tuple]):
        self.real, self.feats, self.chips = real, feats, chips
        self.used_chip = None

    def catalog(self, bbox, t0, t1):
        return [f for f in self.feats if t0 <= cdse.parse_time(f["properties"]["datetime"]) <= t1]

    def process(self, bbox, width, height, t_from, t_to, with_lia=True, upsampling="NEAREST"):
        pad = timedelta(seconds=CHIP_MATCH_S)
        for t, p in self.chips:
            if t_from - pad <= t <= t_to + pad:
                z = np.load(p)
                vv, vh = z["vv"].astype(np.float64), z["vh"].astype(np.float64)
                if vv.shape != (height, width):
                    break   # another extent: fetch
                self.used_chip = p
                return encode_chip(vv, vh), 0.0
        return self.real.process(bbox, width, height, t_from, t_to, with_lia)


def s1_rows(out: Path) -> dict:
    """{site_id: set(acquired_utc)} of the S1 scenes in quay_naval50.csv."""
    res: dict = {}
    p = out / "quay_naval50.csv"
    if p.exists():
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                res.setdefault(r["site_id"], set()).add(r["acquired_utc"])
    return res


def plan_site(client, site: dict, qctx, out: Path, paired: set, done: set, chips: list, max_h: float, log=print) -> list[dict]:
    """The targets of one site: [{acq, feats, s2_utc, dt_h, chip}], nearest first, one per S1 acquisition."""
    grid = Grid.for_site(site)
    targets: dict = {}
    for acq2, _img in close_pairs.clear_s2(out).get(site["id"], []):
        if (site["id"], acq2) in paired:
            continue
        t2 = cdse.parse_time(acq2)
        feats = client.catalog(grid.bbox, t2 - timedelta(hours=max_h), t2 + timedelta(hours=max_h))
        acqs = [a for a in cdse.acquisitions(feats)
                if abs((a["time"] - t2).total_seconds()) <= max_h * 3600 and qctx.baseline(a["relative_orbit"]) is not None]
        if not acqs:
            continue
        a = min(acqs, key=lambda a: abs((a["time"] - t2).total_seconds()))
        if a["acquired_utc"] in done:
            continue   # counted already (the daily run, or an earlier replay): its pair is found by close_pairs
        dt = (a["time"] - t2).total_seconds() / 3600
        chip = next((p for t, p in chips if abs((t - a["time"]).total_seconds()) <= CHIP_MATCH_S), None)
        old = targets.get(a["acquired_utc"])
        if old is None or abs(dt) < abs(old["dt_h"]):
            targets[a["acquired_utc"]] = {"acq": a, "feats": feats, "s2_utc": acq2, "dt_h": round(dt, 1), "chip": chip}
    res = sorted(targets.values(), key=lambda x: (x["chip"] is None, abs(x["dt_h"])))
    log(f"{site['id']}: 組の候補 {len(res)}（手元の切り出し {sum(x['chip'] is not None for x in res)}）")
    return res


def round_robin(plans: dict) -> list[tuple[str, dict]]:
    """Interleave the sites' targets (the chips first: they cost nothing)."""
    order = []
    queues = {sid: list(v) for sid, v in plans.items()}
    for want_chip in (True, False):
        qs = {sid: [t for t in v if (t["chip"] is not None) == want_chip] for sid, v in queues.items()}
        while any(qs.values()):
            for sid in sorted(qs):
                if qs[sid]:
                    order.append((sid, qs[sid].pop(0)))
    return order


def keep_outputs(work: Path, keep: Path) -> None:
    keep.mkdir(parents=True, exist_ok=True)
    for name in KEEP_FILES:
        src = work / name
        dst = keep / name
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)


def replay(client, sites: list[dict], out: Path, work: Path, keep: Path | None, chip_dirs: list[Path], cap_pu: float,
           deadline: float | None = None, now: datetime | None = None, log=print, max_h: float = close_pairs.CLOSE_PAIR_H,
           mask_dir: Path | None = None, lines_dir: Path | None = None, baseline_dir: Path | None = None) -> dict:
    from . import run as srun
    from . import eval_presence
    now = now or datetime.now(timezone.utc)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(out, work)
    store = srun.Store(work, sites)
    chips = chip_index(chip_dirs)
    ledger = close_pairs._read(out / close_pairs.PAIRS_FILE)
    paired = {(r["site_id"], r["s2_utc"]) for r in ledger}
    live = s1_rows(out)
    summ = {"probe_version": PROBE_VERSION, "sites": {}, "counted": [], "skipped": [], "errors": []}
    ctx = {}
    plans = {}
    mask_dir = mask_dir or srun.MASK_DIR
    for site in sites:
        try:
            mask, _ = srun.load_mask(site["id"], srun.mask_version(site), mask_dir)
        except FileNotFoundError as e:
            summ["errors"].append(f"{site['id']}: {e}")
            continue
        grid = Grid.for_site(site)
        qctx = srun.QuayContext(site, grid, mask, lines_dir or srun.MASK_DIR, baseline_dir or srun.fbase.BASELINE_DIR)
        ctx[site["id"]] = (site, mask)
        done = live.get(site["id"], set()) | store.done(site["id"], srun.mask_tag(site))
        try:
            plans[site["id"]] = plan_site(client, site, qctx, out, paired, done, chips.get(site["id"], []), max_h, log)
        except cdse.CredentialError:
            raise
        except Exception as e:  # noqa: BLE001  a catalog failure leaves one site out
            summ["errors"].append(f"{site['id']} catalog: {str(e)[:200]}")
    summ["planned"] = {sid: len(v) for sid, v in plans.items()}
    # the replay is not the daily series: a rebuilt mask does not hold the older scenes back here
    keep_filter = srun.rebuilt_mask_filter
    srun.rebuilt_mask_filter = lambda site, store, acqs, done: acqs
    replayed = set()
    try:
        for sid, t in round_robin(plans):
            if deadline is not None and time.monotonic() > deadline:
                summ["stopped"] = "time"
                break
            left = cap_pu - float(getattr(client, "processing_units", 0.0))
            if t["chip"] is None and left < cdse.estimate_pu(Grid.for_site(ctx[sid][0]).width, Grid.for_site(ctx[sid][0]).height, 3):
                summ["skipped"].append(f"{sid} {t['acq']['acquired_utc']}: 上限")
                continue
            site, mask = ctx[sid]
            rc = ReplayClient(client, t["feats"], chips.get(sid, []))
            acq_t = t["acq"]["time"]
            try:
                budget = 1e9 if t["chip"] is not None else max(left, 1e-9)   # a saved chip costs nothing (the client caps the rest)
                s = srun.process_site(rc, store, site, mask, acq_t + timedelta(hours=1), 1, budget, 1,
                                      log=log, only=t["acq"]["acquired_utc"])
            except cdse.CredentialError:
                raise
            except Exception as e:  # noqa: BLE001  BudgetStop and the rest: stop here, keep what is counted
                summ["errors"].append(f"{sid} {t['acq']['acquired_utc']}: {type(e).__name__}: {str(e)[:200]}")
                if type(e).__name__ == "BudgetStop":
                    summ["stopped"] = str(e)[:200]
                    break
                continue
            if s.get("counted"):
                replayed.add((sid, t["acq"]["acquired_utc"]))
                summ["counted"].append({"site_id": sid, "s1_utc": t["acq"]["acquired_utc"], "s2_utc": t["s2_utc"],
                                        "dt_h": t["dt_h"], "chip": bool(rc.used_chip)})
            else:
                summ["skipped"].append(f"{sid} {t['acq']['acquired_utc']}: {json.dumps(s, ensure_ascii=False)[:160]}")
    finally:
        srun.rebuilt_mask_filter = keep_filter
    if replayed:
        store.save({"run_utc": cdse.iso(now), "sites": [], "replay": PROBE_VERSION})
        boxes: dict = {}
        rows = [r for r in eval_presence.naval50_counts(work, sites, boxes) if (r["site_id"], r["acquired_utc"]) in replayed]
        summ["pairs_kept"] = close_pairs.update(out, rows, boxes, now=now, source="replay", numbers_out=work)
        summ["pairs_replay"] = sum(1 for r in close_pairs._read(out / close_pairs.PAIRS_FILE) if r.get("source") == "replay")
        if keep is not None:
            keep_outputs(work, keep)
    summ["processing_units"] = round(float(getattr(client, "processing_units", 0.0)), 3)
    summ["requests"] = int(getattr(client, "requests", 0))
    return summ


def main(argv=None, client=None, sites=None, utcnow=lambda: datetime.now(timezone.utc)) -> int:
    from . import s1_recount
    from .port_archive import LimitedClient
    from .run import load_sites
    ap = argparse.ArgumentParser(description="close pairs from the past: older S1 scenes counted again (probe)")
    ap.add_argument("--site", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--keep", type=Path)
    ap.add_argument("--chips", type=Path, nargs="*", default=[])
    ap.add_argument("--budget-month", required=True)
    ap.add_argument("--global-cap", type=float, required=True)
    ap.add_argument("--global-margin", type=float, required=True)
    ap.add_argument("--ocean-cap", type=float, required=True)
    ap.add_argument("--usage", type=Path, nargs="*", default=[])
    ap.add_argument("--ports-usage", type=Path, nargs="*", default=[])
    ap.add_argument("--usage-out", type=Path, required=True)
    ap.add_argument("--max-minutes", type=float, default=150.0)
    ap.add_argument("--max-requests", type=int, default=1500)
    a = ap.parse_args(argv)
    now = utcnow()
    if a.budget_month != BUDGET_MONTH or now.strftime("%Y-%m") != a.budget_month:
        print(json.dumps({"error": f"上限は {BUDGET_MONTH} の値（指定 {a.budget_month}、今 {now:%Y-%m}）。取得しない"}, ensure_ascii=False))
        return 1
    usage = s1_recount.read_csv(a.usage_out)
    ml = s1_recount.month_left(a.budget_month, now, a.global_cap, a.global_margin, a.ocean_cap, a.usage, a.ports_usage)
    copies = [p for p in a.usage if "replay_usage" in p.name]
    spent_before = s1_recount.spent(usage, copies)
    left = round(min(ml["left"], MAX_PU_TOTAL - spent_before), 3)
    all_sites = {s["id"]: s for s in (sites if sites is not None else load_sites())}
    chosen = [all_sites[s] for s in a.site if s in all_sites]
    deadline = time.monotonic() + a.max_minutes * 60
    client = client if client is not None else LimitedClient(max(left, 0.0), a.max_requests, 80.0, 80, deadline=deadline,
                                                             budget_month=a.budget_month, utcnow=utcnow)
    summ = {"max_pu_total": MAX_PU_TOTAL, "spent_before": spent_before, "month": ml, "cap_this_run": left}
    rc = 0
    try:
        summ.update(replay(client, chosen, a.out, a.work, a.keep, a.chips, max(left, 0.0), deadline=deadline, now=now,
                           log=lambda m: print(m, file=sys.stderr)))
    except cdse.CredentialError as e:
        summ["error"] = str(e)[:300]
        rc = 1
    finally:
        if getattr(client, "requests", 0):
            usage.append({"run_utc": cdse.iso(now), "site_id": "+".join(a.site), "probe_version": PROBE_VERSION,
                          "requests": client.requests, "processing_units": round(client.processing_units, 3),
                          "pu_source": "header" if getattr(client, "pu_reported", True) else "estimate",
                          "scenes": len(summ.get("counted", [])), "cap_pu": left})
            a.usage_out.parent.mkdir(parents=True, exist_ok=True)
            with a.usage_out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=USAGE_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(usage)
    print(json.dumps(summ, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
