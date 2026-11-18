"""pres-1a calibration and its out-of-sample evaluation (sat/pres1.py; design docs/data/sat/eval/pres1-design.md §4.3).

The cells are the presence evaluation's cells with the S2 day's truth (sat/eval_presence.evaluate_one(..., "s2"): one Sentinel-1
scene x one eligible naval quay section, the owner's answer y and the current judge's p). Every cell's history (res_rate, res_n,
prev_y) comes from the owner's answers of the section on the days strictly before the cell's scene (sat/pres1.history), so no
answer of the scene's day or later enters a feature, in the training cells or in the test cells.

* forward validation (``forward``): the windows [2025-12-01, 2026-03-01), [2026-03-01, 2026-06-01), [2026-06-01, fit_before)... :
  each window's cells are judged by a model fitted on the cells of the label days before the window's start only.
* the model in use (``model``): fitted on the cells of the label days before ``fit_before`` only; the cells from ``fit_before``
  on are its out-of-sample test (``holdout``). The numbers of the training cells are not reported.

Both report pres-1a beside the current judge on the same cells: recall, precision, F1, Brier, AUC at p >= 0.5; the cells whose
answer changed from the section's previous answer (``changed``: where pres-1a is weak) and the others (``same``); the change
flag's hits; per port; 95% intervals by a bootstrap over scenes (the cells of one scene are not independent) for pres-1a and for
the difference to the current judge (paired, the same resample).

Pure Python (no numpy; PU 0). Refit monthly (move --fit-before to the first day of the month whose answers are frozen):

    python -m sat.pres1_calib [--fit-before 2026-06-01] [--out docs/data/sat] [--boot 2000]   # writes sat/pres1_calib.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import eval_presence as ep
from . import pres1

ROOT = Path(__file__).resolve().parent.parent
FIT_BEFORE = "2026-06-01"
WINDOWS = ("2025-12-01", "2026-03-01", "2026-06-01")
BOOT = 2000
SEED = 0


def cells(out: Path, sites: list[dict]) -> list[dict]:
    """The evaluation's s2 cells with their history: {site_id, section_id, date, s1, scene, y, p, dt_h, x, prev_y, cut}."""
    res = ep.evaluate_one(out, sites, "s2")
    rows = []
    for sid, s in sorted(res["sites"].items()):
        ans = pres1.answers(out, sid)
        for r in s["rows"]:
            cut = pres1.cut_day(r.get("date"), r.get("s1"))
            h = pres1.history(ans, r["section_id"], cut)
            used = [d for d, _ in ans.get(r["section_id"], []) if d < cut]
            rows.append({"site_id": sid, "section_id": r["section_id"], "date": (r.get("date") or r["s1"])[:10], "s1": r["s1"],
                         "scene": f"{sid}|{r['s1']}", "y": int(r["y"]), "p": float(r["p"]), "dt_h": r.get("dt_h"),
                         "x": pres1.vector(r["p"], h), "prev_y": h["prev_y"], "cut": cut,
                         "last_answer_used": max(used) if used else None})
    return rows


# ---------------------------------------------------------------- metrics
def _auc(y, p):
    pos = sorted(pp for yy, pp in zip(y, p) if yy)
    neg = sorted(pp for yy, pp in zip(y, p) if not yy)
    if not pos or not neg:
        return None
    import bisect
    s = 0.0
    for v in pos:
        lo, hi = bisect.bisect_left(neg, v), bisect.bisect_right(neg, v)
        s += lo + 0.5 * (hi - lo)
    return s / (len(pos) * len(neg))


def point(y: list[int], p: list[float]) -> dict:
    tp = sum(1 for a, b in zip(y, p) if a and b >= pres1.THRESHOLD)
    fp = sum(1 for a, b in zip(y, p) if not a and b >= pres1.THRESHOLD)
    fn = sum(1 for a, b in zip(y, p) if a and b < pres1.THRESHOLD)
    tn = len(y) - tp - fp - fn
    n = len(y)
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None, "accuracy": (tp + tn) / n if n else None,
            "brier": sum((b - a) ** 2 for a, b in zip(y, p)) / n if n else None, "auc": _auc(y, p)}


KEYS = ("recall", "precision", "f1", "accuracy", "brier", "auc")


def _pct(v: list[float], q: float):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    i = q * (len(v) - 1)
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def compare(cs: list[dict], pp: list[float], boot: int = BOOT, seed: int = SEED) -> dict:
    """pres-1a (pp) and the current judge on the cells cs: point metrics with 95% intervals (scene bootstrap) and the paired
    difference pres1 - current."""
    y = [c["y"] for c in cs]
    pc = [c["p"] for c in cs]
    a, b = point(y, pp), point(y, pc)
    out = {"n": len(cs), "scenes": len({c["scene"] for c in cs}), "base_rate": _r(sum(y) / len(y)) if y else None,
           "pres1": {k: _r(v) if isinstance(v, float) else v for k, v in a.items()},
           "current": {k: _r(v) if isinstance(v, float) else v for k, v in b.items()}}
    if not cs or boot <= 0:
        return out
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(cs):
        idx.setdefault(c["scene"], []).append(i)
    scenes = sorted(idx)
    rng = random.Random(seed)
    sa = {k: [] for k in KEYS}
    sd = {k: [] for k in KEYS}
    for _ in range(boot):
        ii = [i for s in (rng.choice(scenes) for _ in scenes) for i in idx[s]]
        yy = [y[i] for i in ii]
        ma, mb = point(yy, [pp[i] for i in ii]), point(yy, [pc[i] for i in ii])
        for k in KEYS:
            sa[k].append(ma[k])
            sd[k].append(None if ma[k] is None or mb[k] is None else ma[k] - mb[k])
    out["pres1_ci95"] = {k: [_r(_pct(sa[k], 0.025)), _r(_pct(sa[k], 0.975))] for k in KEYS}
    out["diff"] = {k: (None if a[k] is None or b[k] is None else _r(a[k] - b[k])) for k in KEYS}
    out["diff_ci95"] = {k: [_r(_pct(sd[k], 0.025)), _r(_pct(sd[k], 0.975))] for k in KEYS}
    return out


def flag_hits(cs: list[dict]) -> dict:
    f = [c for c in cs if c["prev_y"] is not None]
    flags = [c for c in f if int(c["p"] >= pres1.THRESHOLD) != c["prev_y"]]
    changes = [c for c in f if c["y"] != c["prev_y"]]
    hit = sum(1 for c in flags if c["y"] != c["prev_y"])
    return {"cells": len(f), "flags": len(flags), "true": hit, "changes": len(changes),
            "precision": _r(hit / len(flags)) if flags else None, "recall": _r(hit / len(changes)) if changes else None,
            "base_rate_change": _r(len(changes) / len(f)) if f else None}


def report(cs: list[dict], pp: list[float], boot: int) -> dict:
    """All cells, changed / same / no previous answer, near pairs, per port, and the change flag."""
    def sub(pred, b=boot):
        ii = [i for i, c in enumerate(cs) if pred(c)]
        return compare([cs[i] for i in ii], [pp[i] for i in ii], b)
    near = lambda c: c["dt_h"] is not None and abs(float(c["dt_h"])) <= ep.FAR_H  # noqa: E731
    return {"all": compare(cs, pp, boot),
            "changed": sub(lambda c: c["prev_y"] is not None and c["y"] != c["prev_y"]),
            "same": sub(lambda c: c["prev_y"] is not None and c["y"] == c["prev_y"]),
            "no_prev": sub(lambda c: c["prev_y"] is None, 0),
            "near": sub(near),
            "per_port": {s: sub(lambda c, s=s: c["site_id"] == s, 0) for s in sorted({c["site_id"] for c in cs})},
            "change_flag": flag_hits(cs)}


def check_no_future(train: list[dict], test_start: str) -> None:
    """The training cells and every answer in their features are from before test_start (raises otherwise)."""
    for c in train:
        if c["date"] >= test_start or c["s1"][:10] >= test_start:
            raise AssertionError(f"training cell from {c['date']} ({c['s1']}) is not before {test_start}")
        if c["last_answer_used"] is not None and (c["last_answer_used"] >= c["cut"] or c["last_answer_used"] >= test_start):
            raise AssertionError(f"answer of {c['last_answer_used']} used for a cell cut at {c['cut']}")


def fit_cells(train: list[dict]) -> dict:
    return pres1.fit([c["x"] for c in train], [c["y"] for c in train])


def forward(cs: list[dict], fit_before: str, boot: int) -> dict:
    """Each window judged by a model fitted on the cells before the window only; pooled and per window."""
    bounds = [w for w in WINDOWS if w < fit_before] + [fit_before]
    test, pred, wins = [], [], []
    for a, b in zip(bounds, bounds[1:] + ["9999-12-31"]):
        tr = [c for c in cs if c["date"] < a and c["s1"][:10] < a]
        te = [c for c in cs if a <= c["date"] < b]
        if not tr or not te:
            continue
        check_no_future(tr, a)
        m = fit_cells(tr)
        pp = [pres1.predict(m, c["x"]) for c in te]
        test += te
        pred += pp
        wins.append({"from": a, "to": None if b == "9999-12-31" else b, "train_n": len(tr), "test_n": len(te),
                     "coef": [_r(x, 4) for x in m["coef"]], "intercept": _r(m["intercept"], 4)})
    return {"windows": wins, **report(test, pred, boot)}


def calibrate(out: Path, sites: list[dict], fit_before: str = FIT_BEFORE, boot: int = BOOT, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cs = cells(out, sites)
    train = [c for c in cs if c["date"] < fit_before and c["s1"][:10] < fit_before]
    hold = [c for c in cs if c["date"] >= fit_before]
    check_no_future(train, fit_before)
    model = fit_cells(train)
    model = {**model, "mean": [_r(x, 6) for x in model["mean"]], "sd": [_r(x, 6) for x in model["sd"]],
             "coef": [_r(x, 6) for x in model["coef"]], "intercept": _r(model["intercept"], 6)}
    hp = [pres1.predict(model, c["x"]) for c in hold]
    return {"version": pres1.VERSION, "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "fit_before": fit_before,
            "truth": "s2", "cells": len(cs), "train_n": len(train), "holdout_n": len(hold),
            "train_last_label_day": max((c["date"] for c in train), default=None),
            "train_last_answer_used": max((c["last_answer_used"] for c in train if c["last_answer_used"]), default=None),
            "bootstrap": {"B": boot, "unit": "scene", "seed": SEED},
            "note": ("pres-1a の係数（model：fit_before より前のラベルの日の升だけで当てた。各升の res はその撮影の日より前の答えだけ）と標本外の評価。"
                     "holdout＝fit_before 以後の升（この係数の標本外）、forward＝前進検証（各窓をそれより前の升だけで当てた係数で判定）。"
                     "changed＝区間の答えが直前の答えから変わった升（pres-1a が現行の判定より弱い所）、same＝変わらなかった升。"
                     "*_ci95＝撮影を単位にしたブートストラップの 95% 区間、diff＝pres1 − current（同じ再標本の対）。閾値 0.5"),
            "model": model, "holdout": report(hold, hp, boot), "forward": forward(cs, fit_before, boot)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ep.OUT)
    ap.add_argument("--fit-before", default=FIT_BEFORE)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--write", type=Path, default=pres1.CALIB_PATH)
    a = ap.parse_args(argv)
    d = calibrate(a.out, ep.load_sites(), a.fit_before, a.boot)
    a.write.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in ("holdout", "forward"):
        r = d[name]
        print(name, json.dumps({k: {m: {kk: r[k][m].get(kk) for kk in ("f1", "brier", "auc", "accuracy")} for m in ("pres1", "current")}
                                for k in ("all", "changed", "same") if r[k]["n"]}, ensure_ascii=False), file=sys.stderr)
        print(name, "change_flag", r["change_flag"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
