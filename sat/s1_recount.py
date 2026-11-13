"""Sentinel-1 again for the days of the owner's eye check (S1 only; probe; owner's consent 2026-11-03, about 37 PU).

The evaluation of the quay occupancy (sat/eval_occ.py: est-2, occ-1) and of the rows abreast (sat/eval_cols.py,
cols-1) needs the raw sigma0 of the Sentinel-1 scenes paired with the answered Sentinel-2 days. The list is
labels/<site>/s1_recount_plan.json (sat/s2_answers.py plan: one scene per answered day, of an orbit with a baseline,
within 4 days). Per scene: one catalog request (+-1 h) for the acquisition's time window, then one Process API
request on the site's grid (VV and VH, 10 m, no incidence angle: as cdse.estimate_pu(width, height, 2) in the plan).
The chip is saved as float16 dB (``--raw``: <raw>/<site>/s1-<stamp>.npz, NaN = no data) and nothing is published.

Budget, explicit here and checked before every request (port_archive.LimitedClient):

* ``MAX_PU_TOTAL`` (40 PU until 2026-11-13, 100 PU since) for all runs of this job together: the PU of this job's
  earlier rows in ``--usage-out`` are subtracted;
* what is left of the month's own caps (``month_left``, the same rule as port_archive.guard): the global cap
  (``--global-cap`` 9,000) minus its margin (450), the other usage of the month (main's sat usage and every
  branch's usage CSVs, ``--usage``; each record once), the port archive's usage (``--ports-usage``, one pool) and
  the reserve of the ocean (``--ocean-cap`` minus the ocean usage already in the files); and ``--budget-month``
  (2026-11): in any other UTC month nothing is sent;
* the scenes already saved in ``--raw`` are not fetched again, nor the scenes already in labels/<site>-s2-s1-match.csv
  (``matched``: fetched on an earlier probe branch and merged; the chips stay on that branch, so cut the next probe branch from
  it, probe/sat-s1-recount-ops4, when the chips are needed again);
* ``MAX_PU_TOTAL`` counts the rows of ``--usage-out`` and of every gathered copy of s1_recount_usage.csv among ``--usage``
  (each record once);
* ``--skip-est2``: the scenes est-2 already judged (quay_est2/<site>.json; sat/eval_presence.py prefers est-2 for them,
  so a re-count adds no cell to the presence evaluation) are not fetched;
* ``--max-scenes SITE=N``: at most N scenes of the site, the nearest to their label day (|dt_h| of the plan's per_day)
  first; the rest are listed under ``deferred`` for a later run.

Ops-4 (2026-11-07, the owner's priority: ports first, SYL -> SLP -> QGZ within the 40 PU): first run SYL 4 + SLP 7 (the
2026-10-14 scene of both is in est-2) + QGZ 4 of 8 = 15 scenes, 25.5 PU by the header (the plan's estimate, 38.3 PU, counts
3 bands; the request has 2, so a scene costs 2/3 of it); second run the other 4 QGZ scenes (about 5.1 PU).

Ops-5 (2026-11-13, the owner's consent "go ahead" to about 67 PU more): the owner answered the photo check of five more
ports (Dinghai, Lushun, Qingdao harbour, Jianggezhuang, Xiangshan; 39 answered days), whose Sentinel-1 scenes are not stored,
so the presence evaluation had almost no cell for them. The plans add up to 66.8 PU by the 3-band estimate (about 45 PU for
the 2 bands of the request). The cap of all runs together goes from 40 to 100 PU (about 30.6 PU spent by ops-4).

``--match`` (no CDSE, PU 0): labels/<site>-s2-s1-match.csv from the saved chips, in the shape of Zhanjiang's table
(sat/s2_labels.MATCH_FIELDS): one row per section the owner was asked about (labels/<site>/review*.json) x label day, with
the Sentinel-1 features of the day's paired scene (sat/s2_labels.s1_sections: s1_occupied, s1_occ_len_m, s1_stretch_len_m, depth, face) and
the owner's answer in the s2_* columns (s2_status "owner", s2_occupied = verified_occupied, s2_c = verified_c; blank when
not answered or "cannot judge"). ``agree`` compares the owner and S1 where the section's coverage is >= 0.5. A summary goes
to labels/<site>/s1_recount_match.json. sat/eval_presence.py reads the table for the scenes est-2 does not hold.

The usage row (run_utc, site_id, probe_version, requests, processing_units) is appended to ``--usage-out``
(docs/data/sat/s1_recount_usage.csv on the probe branch), which the guard of the other probes gathers.

    python -m sat.s1_recount --site SYL SLP QGZ --raw probe-raw/s1-recount --budget-month 2026-11 \
        --global-cap 9000 --global-margin 450 --ocean-cap 7000 --usage U/*.csv --ports-usage P/*.csv \
        --usage-out docs/data/sat/s1_recount_usage.csv --skip-est2 [--max-scenes QGZ=4]
    python -m sat.s1_recount --match --site SYL SLP QGZ --raw probe-raw/s1-recount      # the match tables, PU 0
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
RAW_SUBDIR = "probe-raw/s1-recount"   # where the chips are saved on the probe branch (--raw), read again by sat/s2_answers --s1
MAX_PU_TOTAL = 100.0         # the owner's consents: about 37 PU (2026-11-03, cap 40) + about 67 PU for five more ports (2026-11-13)
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


def plan_dt(plan: dict) -> dict:
    """{acquired_utc: smallest |dt_h| of its label days} from the plan's per_day (missing: no entry)."""
    res: dict = {}
    for d in plan.get("per_day", []):
        if d.get("s1") and d.get("dt_h") is not None:
            res[d["s1"]] = min(abs(float(d["dt_h"])), res.get(d["s1"], float("inf")))
    return res


def est2_acquired(out: Path, site_id: str) -> set:
    """The acquisitions est-2 already judged (quay_est2/<site>.json; Zhanjiang: quay_est2.json)."""
    p = out / ("quay_est2.json" if site_id == "ZJG" else f"quay_est2/{site_id}.json")
    if not p.exists():
        return set()
    return {sc["acquired_utc"] for sc in json.loads(p.read_text(encoding="utf-8")).get("scenes", []) if sc.get("site_id") == site_id}


def pick_scenes(plan: dict, skip: set = frozenset(), max_scenes: int | None = None) -> tuple[list[dict], list[str], list[str]]:
    """(scenes to fetch, skipped as in est-2, deferred beyond ``max_scenes``). The order: the nearest to its label day first
    (plan per_day dt_h; scenes without one keep the plan's order after them)."""
    dt = plan_dt(plan)
    scenes = [sc for _, _, sc in sorted((dt.get(sc["acquired_utc"], float("inf")), i, sc) for i, sc in enumerate(plan.get("scenes", [])))]
    skipped = [sc["acquired_utc"] for sc in scenes if sc["acquired_utc"] in skip]
    scenes = [sc for sc in scenes if sc["acquired_utc"] not in skip]
    deferred = []
    if max_scenes is not None and len(scenes) > max_scenes:
        deferred = [sc["acquired_utc"] for sc in scenes[max_scenes:]]
        scenes = scenes[:max_scenes]
    return scenes, skipped, deferred


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


def spent(usage: list[dict], copies: list[Path] = ()) -> float:
    """PU of this job's earlier runs (rows of PROBE_VERSION): the rows of ``--usage-out`` and of every copy of
    docs/data/sat/s1_recount_usage.csv gathered from the branches (``copies``: the other-*s1_recount* files of
    port_archive.gather_usage), each record once (a probe branch cut from main holds the rows main holds)."""
    from .port_archive import row_key
    seen = {row_key(r) for r in usage}
    rows = list(usage)
    for p in copies:
        for r in read_csv(Path(p)):
            if row_key(r) not in seen:
                seen.add(row_key(r))
                rows.append(r)
    return round(sum(float(r.get("processing_units") or 0) for r in rows if r.get("probe_version") == PROBE_VERSION), 3)


def recount_copies(usage: list[Path]) -> list[Path]:
    """The gathered copies of this job's usage among ``--usage`` (other-docs_data_sat_s1_recount_usage.csv--<n>.csv)."""
    from .port_archive import usage_key
    return [Path(u) for u in usage if usage_key(Path(u)).startswith("other-") and "s1_recount_usage" in usage_key(Path(u))]


def matched_acquired(out: Path, site_id: str) -> set:
    """The scenes whose features are already in labels/<site>-s2-s1-match.csv (s1_acquired_utc; on main after a merge):
    fetched before, so not fetched again on a probe branch that lacks the chips."""
    return {r["s1_acquired_utc"] for r in read_csv(out / "labels" / f"{site_id}-s2-s1-match.csv") if r.get("s1_acquired_utc")}


def month_left(month: str, now: datetime, global_cap: float, global_margin: float, ocean_cap: float,
               usage: list[Path], ports_usage: list[Path]) -> dict:
    """PU left of the month's own caps for this job (port_archive.guard without the archive's own cap)."""
    from .port_archive import is_ocean_key, union_usage, usage_key
    if month != now.strftime("%Y-%m"):
        raise ValueError(f"上限は {month} の値（今は {now:%Y-%m}）")
    groups: dict = {}
    for u in usage:
        groups.setdefault(usage_key(Path(u)), []).append(Path(u))
    per = {k: union_usage(v, month) for k, v in groups.items()}
    other = sum(v[0] for v in per.values())
    ocean = sum(v[0] for k, v in per.items() if is_ocean_key(k))
    reserve = max(0.0, ocean_cap - ocean)
    ports, _ = union_usage([Path(p) for p in ports_usage], month)
    left = global_cap - global_margin - other - reserve - ports
    return {"left": round(max(0.0, left), 3), "other_pu": round(other, 1), "ocean_pu": round(ocean, 1),
            "reserve_pu": round(reserve, 1), "ports_pu": round(ports, 1), "global_cap": global_cap, "global_margin": global_margin}


def fetch_site(client, site: dict, raw: Path, out: Path = OUT, log=print, skip_est2: bool = False,
               max_scenes: int | None = None) -> dict:
    """Fetch the plan's scenes of one site that are not in ``raw`` (nor, with ``skip_est2``, in est-2; at most
    ``max_scenes``, the nearest to their label day first). Stops (BudgetStop) at a cap."""
    from .run import decode_chip
    grid = Grid.for_site(site)
    p_plan = out / "labels" / site["id"] / "s1_recount_plan.json"
    plan = json.loads(p_plan.read_text(encoding="utf-8")) if p_plan.exists() else {}
    scenes, skipped, deferred = pick_scenes(plan, est2_acquired(out, site["id"]) if skip_est2 else frozenset(), max_scenes)
    res = {"site_id": site["id"], "planned": len(plan.get("scenes", [])), "fetched": [], "had": [], "matched": [], "missing": [],
           "errors": [], "skipped_est2": skipped, "deferred": deferred}
    matched = matched_acquired(out, site["id"])
    for sc in scenes:
        p = raw_path(raw, site["id"], sc["acquired_utc"])
        if p.exists():
            res["had"].append(sc["acquired_utc"])
            continue
        if sc["acquired_utc"] in matched:
            res["matched"].append(sc["acquired_utc"])
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


# ---------------------------------------------------------------- the match table (no CDSE)
MATCH_VERSION = "s1-match-3"   # s1-match-2 (s1-match-1 + s1_stretch_len_m) with the stretch assigned by s2_labels.STRETCH_ASSIGN (assign-2)


def review_days(out: Path, site_id: str) -> tuple[list[dict], set]:
    """(label days with their paired scene (s1_near[0]), ids of the sections the owner was asked about) of the review sets."""
    days, ids = {}, set()
    for name in ("review.json", "review-oos.json"):
        p = out / "labels" / site_id / name
        if not p.exists():
            continue
        rv = json.loads(p.read_text(encoding="utf-8"))
        ids |= {s["id"] for s in rv.get("sections", [])}
        for d in rv.get("days", []):
            near = d.get("s1_near") or []
            if near and d["date"] not in days:
                days[d["date"]] = {"date": d["date"], "acquired_utc": d.get("acquired_utc"), "s1": near[0]}
    return sorted(days.values(), key=lambda d: d["date"]), ids


def answers(out: Path, site_id: str) -> dict:
    """{(section_id, date): row} of the owner's answers (labels/<site>-s2-answers.csv, -oos-answers.csv)."""
    res = {}
    for name in (f"{site_id}-s2-answers.csv", f"{site_id}-oos-answers.csv"):
        for r in read_csv(out / "labels" / name):
            res.setdefault((r["section_id"], r["date"]), r)
    return res


def build_match(site: dict, raw: Path, out: Path = OUT, mask_dir=None, baseline_dir=None, log=print) -> dict:
    """labels/<site>-s2-s1-match.csv (sat/s2_labels.MATCH_FIELDS) from the saved chips: the review sections x the label days
    whose paired scene is in ``raw``; the owner's answer in the s2_* columns. Returns the summary (also written to
    labels/<site>/s1_recount_match.json)."""
    from . import baseline as fbase
    from . import quay, quay_lines, s2_labels
    from .osm_mask import MASK_DIR, load_mask
    from .run import in_sample, site_sectors, sector_of as _sector_of
    sid = site["id"]
    mask_dir = mask_dir or MASK_DIR
    baseline_dir = baseline_dir or fbase.BASELINE_DIR
    grid = Grid.for_site(site)
    mver = int(site.get("mask_version", 1))
    mask, _ = load_mask(sid, mver, mask_dir)
    lines = quay_lines.load(sid, int(site.get("quay_lines_version", quay_lines.LINES_VERSION)), mask_dir)
    geom = quay.Geometry(grid, mask, lines)
    sectors = site_sectors(site)
    days, ids = review_days(out, sid)
    secs = [sc for sc in s2_labels.sections(geom, lambda lo, la: _sector_of(sectors, lo, la)) if sc["section_id"] in ids]
    ans = answers(out, sid)
    bls = {str(o): fbase.load(site, o, baseline_dir) for o in fbase.site_config(site)["orbits"]}
    bls = {k: v for k, v in bls.items() if v is not None and (v[1].get("width"), v[1].get("height")) == (grid.width, grid.height)}
    lat = float(site.get("center_lat") or (grid.south + grid.north) / 2)
    summ = {"site_id": sid, "version": MATCH_VERSION, "sections": len(secs), "sections_unknown": sorted(ids - {sc["section_id"] for sc in secs}),
            "days": len(days), "days_matched": [], "days_without_chip": [], "scenes": [], "rows": 0, "answered": 0, "agree": 0,
            "owner_1_s1_0": 0, "owner_0_s1_1": 0, "no_baseline": []}
    feats: dict = {}
    rows = []
    for d in days:
        a = d["s1"]
        key = a["acquired_utc"]
        z = load(raw_path(raw, sid, key))
        if z is None:
            summ["days_without_chip"].append(d["date"])
            continue
        if key not in feats:
            if str(a["relative_orbit"]) not in bls:
                summ["no_baseline"].append(key)
                continue
            vv, vh, meta = z
            layers, bmeta = bls[str(a["relative_orbit"])]
            look = quay.look_azimuth(meta.get("orbit_direction") or a.get("orbit_direction"), lat)
            samples: list = []
            q = quay.count(geom, vv, vh, layers, look_az_deg=look, samples_out=samples)
            feats[key] = (s2_labels.s1_sections(secs, q, samples, geom.cfg["step_m"], geom.cfg["face_cos"]), look,
                          in_sample(site, key, bmeta))
            summ["scenes"].append(key)
        s1f, look, insample = feats[key]
        summ["days_matched"].append(d["date"])
        for sc in secs:
            f = s1f.get(sc["section_id"], {})
            r = ans.get((sc["section_id"], d["date"])) or {}
            verified = r.get("verified") == "1"
            m = {"site_id": sid, "section_id": sc["section_id"], "sector": sc["sector"], "kind": sc["kind"], "length_m": sc["length_m"],
                 "s2_date": d["date"], "s2_acquired_utc": d.get("acquired_utc"), "s1_acquired_utc": key,
                 "relative_orbit": a["relative_orbit"], "dt_h": a.get("dt_h"), "s1_in_sample": insample, "look_az_deg": look,
                 "s2_status": "owner" if r else "", "s2_occupied": r.get("verified_occupied") if verified else None,
                 "s2_depth_m": None, "s2_c": (r.get("verified_c") or None) if verified else None,
                 "s2_color_class": (r.get("verified_color") or None) if verified else None, **f}
            if verified and f.get("s1_coverage", 0) >= 0.5 and f.get("s1_occupied") is not None:
                y, s1 = int(r.get("verified_occupied") == "1"), int(f["s1_occupied"])
                m["agree"] = int(y == s1)
                summ["answered"] += 1
                summ["agree"] += m["agree"]
                if y and not s1:
                    summ["owner_1_s1_0"] += 1
                elif s1 and not y:
                    summ["owner_0_s1_1"] += 1
            rows.append(m)
    summ["rows"] = len(rows)
    summ["agree_rate"] = round(summ["agree"] / summ["answered"], 3) if summ["answered"] else None
    summ["note"] = ("s2_* の列は所有者の答え（labels/<港>-s2-answers.csv：s2_status owner、s2_occupied＝verified_occupied、s2_c＝verified_c。"
                    "未回答・判断できない（verified 0）は空欄）。S1 の列は sat/s2_labels.s1_sections（湛江の ZJG-s2-s1-match.csv と同じ）。"
                    "agree＝所有者と S1 の在/不在の一致（s1_coverage ≥ 0.5 の答えのある升）。評価票 eval/presence.json は est-2 に無い撮影でこの表を読む")
    summ["credit"] = "Contains modified Copernicus Sentinel data " + "–".join(sorted({k[:4] for k in summ["scenes"]})) if summ["scenes"] else None
    summ["attribution"] = "岸壁の線：(c) OpenStreetMap contributors（ODbL 1.0）"
    ldir = out / "labels"
    if rows:
        s2_labels._write(ldir / f"{sid}-s2-s1-match.csv", rows, s2_labels.PORT_MATCH_FIELDS)
    (ldir / sid).mkdir(parents=True, exist_ok=True)
    (ldir / sid / "s1_recount_match.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"{sid}: {len(summ['scenes'])} scenes, {len(summ['days_matched'])} days, {len(rows)} rows, "
        f"agree {summ['agree']}/{summ['answered']} (no chip: {summ['days_without_chip']})")
    return summ


def main(argv=None, client=None, sites=None, utcnow=lambda: datetime.now(timezone.utc)) -> int:
    from .port_archive import BudgetStop, LimitedClient
    from .run import load_sites
    ap = argparse.ArgumentParser(description="Sentinel-1 again for the answered days (S1 only, probe)")
    ap.add_argument("--site", nargs="+", default=["SYL", "SLP"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--match", action="store_true", help="build labels/<site>-s2-s1-match.csv from the saved chips (no CDSE)")
    ap.add_argument("--budget-month")
    ap.add_argument("--global-cap", type=float)
    ap.add_argument("--global-margin", type=float)
    ap.add_argument("--ocean-cap", type=float)
    ap.add_argument("--usage", type=Path, nargs="*", default=[])
    ap.add_argument("--ports-usage", type=Path, nargs="*", default=[])
    ap.add_argument("--usage-out", type=Path)
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--skip-est2", action="store_true", help="do not fetch the scenes est-2 already judged")
    ap.add_argument("--max-scenes", nargs="*", default=[], metavar="SITE=N",
                    help="at most N scenes of SITE, the nearest to their label day first")
    a = ap.parse_args(argv)
    if a.match:
        all_sites = {s["id"]: s for s in (sites if sites is not None else load_sites())}
        res = [build_match(all_sites[sid], a.raw, a.out, log=lambda m: print(m, file=sys.stderr)) for sid in a.site]
        print(json.dumps({"match": res}, ensure_ascii=False))
        return 0
    missing = [k for k in ("budget_month", "global_cap", "global_margin", "ocean_cap", "usage_out") if getattr(a, k) is None]
    if missing:
        ap.error("required without --match: " + ", ".join("--" + k.replace("_", "-") for k in missing))
    limits = {}
    for x in a.max_scenes:
        sid, eq, n = x.partition("=")
        if not sid or not eq or not n.isdigit():
            ap.error(f"--max-scenes: SITE=N の形で（{x!r}）")
        limits[sid] = int(n)
    now = utcnow()
    if a.budget_month != BUDGET_MONTH or now.strftime("%Y-%m") != a.budget_month:
        print(json.dumps({"error": f"上限は {BUDGET_MONTH} の値（指定 {a.budget_month}、今 {now:%Y-%m}）。取得しない"}, ensure_ascii=False))
        return 1
    usage = read_csv(a.usage_out)
    ml = month_left(a.budget_month, now, a.global_cap, a.global_margin, a.ocean_cap, a.usage, a.ports_usage)
    spent_before = spent(usage, recount_copies(a.usage))
    left = round(min(ml["left"], MAX_PU_TOTAL - spent_before), 3)
    summ = {"probe_version": PROBE_VERSION, "max_pu_total": MAX_PU_TOTAL, "spent_before": spent_before, "month": ml,
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
                summ["sites"].append(fetch_site(client, all_sites[sid], a.raw, a.out, log=lambda m: print(m, file=sys.stderr),
                                                skip_est2=a.skip_est2, max_scenes=limits.get(sid)))
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
