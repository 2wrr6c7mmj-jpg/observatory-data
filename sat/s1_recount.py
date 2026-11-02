"""Sentinel-1 again for the days of the owner's eye check (S1 only; probe; owner's consent 2026-11-03, about 37 PU).

The evaluation of the quay occupancy (sat/eval_occ.py: est-2, occ-1) and of the rows abreast (sat/eval_cols.py,
cols-1) needs the raw sigma0 of the Sentinel-1 scenes paired with the answered Sentinel-2 days. The list is
labels/<site>/s1_recount_plan.json (sat/s2_answers.py plan: one scene per answered day, of an orbit with a baseline,
within 4 days). Per scene: one catalog request (+-1 h) for the acquisition's time window, then one Process API
request on the site's grid (VV and VH, 10 m, no incidence angle: as cdse.estimate_pu(width, height, 2) in the plan).
The chip is saved as float16 dB (``--raw``: <raw>/<site>/s1-<stamp>.npz, NaN = no data) and nothing is published.

Budget, explicit here and checked before every request (port_archive.LimitedClient):

* ``MAX_PU_TOTAL`` (40 PU) for all runs of this job together: the PU of this job's earlier rows in ``--usage-out``
  are subtracted;
* what is left of the month's own caps (``month_left``, the same rule as port_archive.guard): the global cap
  (``--global-cap`` 9,000) minus its margin (450), the other usage of the month (main's sat usage and every
  branch's usage CSVs, ``--usage``; each record once), the port archive's usage (``--ports-usage``, one pool) and
  the reserve of the ocean (``--ocean-cap`` minus the ocean usage already in the files); and ``--budget-month``
  (2026-11): in any other UTC month nothing is sent;
* the scenes already saved in ``--raw`` are not fetched again.

The usage row (run_utc, site_id, probe_version, requests, processing_units) is appended to ``--usage-out``
(docs/data/sat/s1_recount_usage.csv on the probe branch), which the guard of the other probes gathers.

    python -m sat.s1_recount --site SYL SLP --raw probe-raw/s1-recount --budget-month 2026-11 \
        --global-cap 9000 --global-margin 450 --ocean-cap 7000 --usage U/*.csv --ports-usage P/*.csv \
        --usage-out docs/data/sat/s1_recount_usage.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import cdse
from .grid import Grid

PROBE_VERSION = "s1-recount-1"
MAX_PU_TOTAL = 40.0          # the owner's consent: about 37 PU (SYL 16.09 + SLP 20.38), with a little room
BUDGET_MONTH = "2026-11"     # the month the cap was set for
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
USAGE_FIELDS = ["run_utc", "site_id", "probe_version", "requests", "processing_units", "pu_source", "scenes", "cap_pu"]
CATALOG_PAD = timedelta(hours=1)


def plan_scenes(site_id: str, out: Path = OUT) -> list[dict]:
    p = out / "labels" / site_id / "s1_recount_plan.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("scenes", [])


def raw_path(raw: Path, site_id: str, acquired_utc: str) -> Path:
    return raw / site_id / f"s1-{acquired_utc.replace(':', '-')}.npz"


def save(path: Path, vv: np.ndarray, vh: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vv=vv.astype(np.float16), vh=vh.astype(np.float16))
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if not path.exists():
        return None
    z = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
    return z["vv"].astype(np.float64), z["vh"].astype(np.float64), meta


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def spent(usage: list[dict]) -> float:
    """PU of this job's earlier runs (rows of PROBE_VERSION)."""
    return round(sum(float(r.get("processing_units") or 0) for r in usage if r.get("probe_version") == PROBE_VERSION), 3)


def month_left(month: str, now: datetime, global_cap: float, global_margin: float, ocean_cap: float,
               usage: list[Path], ports_usage: list[Path]) -> dict:
    """PU left of the month's own caps for this job (port_archive.guard without the archive's own cap)."""
    from .port_archive import union_usage, usage_key
    if month != now.strftime("%Y-%m"):
        raise ValueError(f"上限は {month} の値（今は {now:%Y-%m}）")
    groups: dict = {}
    for u in usage:
        groups.setdefault(usage_key(Path(u)), []).append(Path(u))
    per = {k: union_usage(v, month) for k, v in groups.items()}
    other = sum(v[0] for v in per.values())
    ocean = sum(v[0] for k, v in per.items() if k.startswith("other-") and "ocean" in k)
    reserve = max(0.0, ocean_cap - ocean)
    ports, _ = union_usage([Path(p) for p in ports_usage], month)
    left = global_cap - global_margin - other - reserve - ports
    return {"left": round(max(0.0, left), 3), "other_pu": round(other, 1), "ocean_pu": round(ocean, 1),
            "reserve_pu": round(reserve, 1), "ports_pu": round(ports, 1), "global_cap": global_cap, "global_margin": global_margin}


def fetch_site(client, site: dict, raw: Path, out: Path = OUT, log=print) -> dict:
    """Fetch the plan's scenes of one site that are not in ``raw``. Stops (BudgetStop) at a cap."""
    from .run import decode_chip
    grid = Grid.for_site(site)
    res = {"site_id": site["id"], "planned": 0, "fetched": [], "had": [], "missing": [], "errors": []}
    for sc in plan_scenes(site["id"], out):
        res["planned"] += 1
        p = raw_path(raw, site["id"], sc["acquired_utc"])
        if p.exists():
            res["had"].append(sc["acquired_utc"])
            continue
        t = cdse.parse_time(sc["acquired_utc"])
        acqs = cdse.acquisitions(client.catalog(grid.bbox, t - CATALOG_PAD, t + CATALOG_PAD))
        acqs = [a for a in acqs if str(a["relative_orbit"]) == str(sc["relative_orbit"])
                and abs((a["time"] - t).total_seconds()) <= 120]
        if not acqs:
            res["missing"].append(sc["acquired_utc"])
            log(f"{site['id']} {sc['acquired_utc']}: カタログに無い")
            continue
        a = acqs[0]
        try:
            data, pu = client.process(grid.bbox, grid.width, grid.height, a["t_from"], a["t_to"], False)
            vv, vh, _ = decode_chip(data)
        except cdse.CredentialError:
            raise
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{sc['acquired_utc']}: {str(e)[:200]}")
            continue
        save(p, vv, vh, {"site_id": site["id"], "acquired_utc": a["acquired_utc"], "platform": a["platform"],
                         "orbit_direction": a["orbit_direction"], "relative_orbit": a["relative_orbit"],
                         "width": grid.width, "height": grid.height, "bbox": grid.bbox, "processing_units": pu,
                         "probe_version": PROBE_VERSION,
                         "credit": f"Contains modified Copernicus Sentinel data {a['acquired_utc'][:4]}"})
        res["fetched"].append(a["acquired_utc"])
        log(f"{site['id']} {a['acquired_utc']} 軌道{a['relative_orbit']}: {pu} PU")
    return res


def main(argv=None, client=None, sites=None, utcnow=lambda: datetime.now(timezone.utc)) -> int:
    from .port_archive import BudgetStop, LimitedClient
    from .run import load_sites
    ap = argparse.ArgumentParser(description="Sentinel-1 again for the answered days (S1 only, probe)")
    ap.add_argument("--site", nargs="+", default=["SYL", "SLP"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--budget-month", required=True)
    ap.add_argument("--global-cap", type=float, required=True)
    ap.add_argument("--global-margin", type=float, required=True)
    ap.add_argument("--ocean-cap", type=float, required=True)
    ap.add_argument("--usage", type=Path, nargs="*", default=[])
    ap.add_argument("--ports-usage", type=Path, nargs="*", default=[])
    ap.add_argument("--usage-out", type=Path, required=True)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    a = ap.parse_args(argv)
    now = utcnow()
    if a.budget_month != BUDGET_MONTH or now.strftime("%Y-%m") != a.budget_month:
        print(json.dumps({"error": f"上限は {BUDGET_MONTH} の値（指定 {a.budget_month}、今 {now:%Y-%m}）。取得しない"}, ensure_ascii=False))
        return 1
    usage = read_csv(a.usage_out)
    ml = month_left(a.budget_month, now, a.global_cap, a.global_margin, a.ocean_cap, a.usage, a.ports_usage)
    left = round(min(ml["left"], MAX_PU_TOTAL - spent(usage)), 3)
    summ = {"probe_version": PROBE_VERSION, "max_pu_total": MAX_PU_TOTAL, "spent_before": spent(usage), "month": ml,
            "cap_this_run": left, "sites": []}
    if left <= 0:
        summ["error"] = "上限に達しているため取得しない"
        print(json.dumps(summ, ensure_ascii=False))
        return 0
    all_sites = {s["id"]: s for s in (sites if sites is not None else load_sites())}
    client = client if client is not None else LimitedClient(left, 200, 80.0, 80, deadline=time.monotonic() + a.max_minutes * 60,
                                                             budget_month=a.budget_month, utcnow=utcnow)
    rc = 0
    try:
        for sid in a.site:
            try:
                summ["sites"].append(fetch_site(client, all_sites[sid], a.raw, a.out, log=lambda m: print(m, file=sys.stderr)))
            except BudgetStop as e:
                summ["stopped"] = str(e)
                break
    except cdse.CredentialError as e:
        summ["error"] = str(e)[:300]
        rc = 1
    finally:
        if client.requests:
            usage.append({"run_utc": cdse.iso(now), "site_id": "+".join(a.site), "probe_version": PROBE_VERSION,
                          "requests": client.requests, "processing_units": round(client.processing_units, 3),
                          "pu_source": "header" if getattr(client, "pu_reported", True) else "estimate",
                          "scenes": sum(len(s["fetched"]) for s in summ["sites"]), "cap_pu": left})
            a.usage_out.parent.mkdir(parents=True, exist_ok=True)
            with a.usage_out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=USAGE_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(usage)
    summ["processing_units"] = round(client.processing_units, 3)
    summ["requests"] = client.requests
    print(json.dumps(summ, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
