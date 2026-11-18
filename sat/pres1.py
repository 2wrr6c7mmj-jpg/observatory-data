"""pres-1a (trial, a parallel series): the presence probability of a naval quay section with the section's own past answers,
and the change flag. Design: docs/data/sat/eval/pres1-design.md (2部, 2026-11-18).

The current judge (est-2 / s1-recount, sat/eval_presence: ``p``) is right about one time in four wrong, and most of its errors
repeat on the same sections. The owner's answers of the section's earlier days carry that: pres-1a is a logistic regression of
three variables,

    logit p_pres1 = b0 + b1 z(p_current) + b2 z(res_rate) + b3 z(res_n)

where res_rate = (k + 1) / (n + 2) and res_n = n are the owner's answers of the section on the days strictly before the scene
(k "present" among n; the S2 day's answers labels/<site>-s2-answers.csv, -oos-answers.csv, ZJG-s2-labels.csv, and the answers
at the S1 scene itself, labels/<site>-s1-answers.csv; one answer per day, the S2 answer first), and z() standardises with the
mean and the standard deviation of the training cells. A section without an earlier answer enters with res_rate 0.5, res_n 0.
The coefficients are fitted by sat/pres1_calib.py on the cells of the label days before ``fit_before`` only (with their own
earlier answers only) and stored in sat/pres1_calib.json; the cells from ``fit_before`` on are its out-of-sample evaluation.

pres-1a is good at "the same as before" and weak at a change: on the cells whose answer differs from the section's previous
answer it is right less often than the current judge (design §3.3). The change flag is therefore given apart:

    change_flag = 1 when (p_current >= 0.5) differs from the section's latest earlier answer, 0 when it agrees,
                  None without an earlier answer

It orders the owner's checks ("a sign of change, to be checked"); it is not a count.

Nothing published changes: the presence evaluation (sat/eval_presence, eval-presence-6) carries p_pres1 and change_flag
beside the current judge, and quay_pres1/<site>.json holds the newest scenes of the ports whose sites.json ``quay.pres1`` is
"trial" or "on" (default "off"; Zhanjiang is off and gets no file). No page reads quay_pres1 yet.

Pure Python (no numpy, no CDSE: PU 0). The daily run calls it through eval_presence.refresh.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
CALIB_PATH = Path(__file__).resolve().parent / "pres1_calib.json"
VERSION = "pres-1a"
FEATURES = ("p_current", "res_rate", "res_n")
MODES = ("off", "trial", "on")
THRESHOLD = 0.5
OUT_DIR = "quay_pres1"
KEEP_SCENES = 10           # quay_pres1/<site>.json: the newest scenes
L2_C = 0.3                 # inverse strength of the L2 penalty on the standardised coefficients (the design's fit)


# ---------------------------------------------------------------- settings
def mode(site: dict) -> str:
    """sites.json quay.pres1: off (default) / trial / on. Anything else is an error (a typo must not switch a port on)."""
    m = (site.get("quay") or {}).get("pres1") or "off"
    if m not in MODES:
        raise ValueError(f"{site.get('id')}: quay.pres1 は {MODES} のどれか（{m!r}）")
    return m


def load_calib(path: Path = CALIB_PATH) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if (d.get("model") or {}).get("coef") else None


# ---------------------------------------------------------------- the owner's answers of a section
def answers(out: Path, site_id: str) -> dict[str, list[tuple[str, int]]]:
    """{section_id: [(day, y), ...]} sorted by day: the S2 day's answers and the S1 check's answers (day = the S1 scene's date),
    verified 1 only; one answer per section and day (the S2 day's answer when both exist)."""
    from . import eval_presence as ep   # late: eval_presence imports this module
    by: dict[tuple[str, str], int] = {}
    for a in ep.labels_s1(out, site_id):
        by[(a["section_id"], a["s1"][:10])] = a["y"]
    for a in ep.labels(out, site_id):
        by[(a["section_id"], a["date"][:10])] = a["y"]
    res: dict[str, list[tuple[str, int]]] = {}
    for (sec, day), y in sorted(by.items()):
        res.setdefault(sec, []).append((day, y))
    return res


def history(ans: dict, section_id: str, before: str) -> dict:
    """The section's answers on the days strictly before ``before`` (YYYY-MM-DD...): res_rate, res_n, prev_y (the latest one)
    and prev_date. Nothing of the day ``before`` or later is used (no answer from the scene's own day or the future)."""
    cut = before[:10]
    past = [(d, y) for d, y in ans.get(section_id, []) if d < cut]
    n = len(past)
    k = sum(y for _, y in past)
    last = max(past) if past else None
    return {"res_rate": (k + 1) / (n + 2), "res_n": n, "prev_y": None if last is None else last[1],
            "prev_date": None if last is None else last[0]}


def cut_day(date: str | None, s1: str | None) -> str:
    """The first day whose answers are 'not before the scene': the earlier of the label day and the S1 scene's day."""
    days = [x[:10] for x in (date, s1) if x]
    return min(days)


def vector(p_current: float, h: dict) -> list[float]:
    return [float(p_current), float(h["res_rate"]), float(h["res_n"])]


# ---------------------------------------------------------------- the model (pure Python logistic regression)
def _sigmoid(t: float) -> float:
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (a small dense system)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            raise ValueError("singular system")
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for j in range(c, n + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(X: list[list[float]], y: list[int], c: float = L2_C, iters: int = 100) -> dict:
    """L2-penalised logistic regression on the standardised columns (Newton): minimises
    sum log-loss + ||w||^2 / (2c), the intercept unpenalised (the same objective as scikit-learn's LogisticRegression(C=c) after a
    StandardScaler). Returns {features, mean, sd, coef, intercept, n, n_present, c}."""
    if not X or len(X) != len(y):
        raise ValueError("no training cells")
    k = len(X[0])
    mean = [sum(r[j] for r in X) / len(X) for j in range(k)]
    sd = []
    for j in range(k):
        v = sum((r[j] - mean[j]) ** 2 for r in X) / len(X)
        sd.append(math.sqrt(v) if v > 1e-12 else 1.0)
    Z = [[1.0] + [(r[j] - mean[j]) / sd[j] for j in range(k)] for r in X]
    w = [0.0] * (k + 1)
    lam = 1.0 / c
    for _ in range(iters):
        g = [0.0] * (k + 1)
        h = [[0.0] * (k + 1) for _ in range(k + 1)]
        for z, t in zip(Z, y):
            p = _sigmoid(sum(a * b for a, b in zip(w, z)))
            d = p - t
            s = p * (1 - p)
            for i in range(k + 1):
                g[i] += d * z[i]
                for j in range(k + 1):
                    h[i][j] += s * z[i] * z[j]
        for i in range(1, k + 1):
            g[i] += lam * w[i]
            h[i][i] += lam
        step = _solve(h, g)
        w = [a - b for a, b in zip(w, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return {"features": list(FEATURES)[:k], "mean": mean, "sd": sd, "intercept": w[0], "coef": w[1:], "n": len(y),
            "n_present": int(sum(y)), "c": c}


def predict(model: dict, x: list[float]) -> float:
    t = model["intercept"] + sum(b * (v - m) / s for b, v, m, s in zip(model["coef"], x, model["mean"], model["sd"]))
    return _sigmoid(t)


def change_flag(p_current: float, h: dict) -> int | None:
    if h.get("prev_y") is None:
        return None
    return int(int(p_current >= THRESHOLD) != h["prev_y"])


# ---------------------------------------------------------------- the evaluation rows (sat/eval_presence)
def annotate(rows: list[dict], ans: dict, calib: dict | None) -> None:
    """Add to every evaluation row: res_rate, res_n, prev_y (the section's answers before the scene), change_flag, p_pres1 (None
    without a calibration) and pres1_oos (the label day is on or after the calibration's fit_before: out of sample for pres-1)."""
    model = (calib or {}).get("model")
    fb = (calib or {}).get("fit_before")
    for r in rows:
        h = history(ans, r["section_id"], cut_day(r.get("date"), r.get("s1")))
        r.update(res_rate=round(h["res_rate"], 3), res_n=h["res_n"], prev_y=h["prev_y"], change_flag=change_flag(r["p"], h),
                 p_pres1=None if not model else round(predict(model, vector(r["p"], h)), 3),
                 pres1_oos=None if not fb else (r.get("date") or r["s1"])[:10] >= fb)


def f1(m: dict):
    d = 2 * m["tp"] + m["fp"] + m["fn"]
    return round(2 * m["tp"] / d, 3) if d else None


def block(rows: list[dict], metrics) -> dict:
    """pres-1 beside the current judge on the out-of-sample cells of pres-1 (pres1_oos): all, the cells whose answer changed from
    the section's previous answer (``changed``), the cells where it did not (``same``), and the change flag's hits.
    ``metrics``: eval_presence.metrics (passed in: eval_presence imports this module)."""
    o = [r for r in rows if r.get("p_pres1") is not None and r.get("pres1_oos")]
    changed = [r for r in o if r.get("prev_y") is not None and r["y"] != r["prev_y"]]
    same = [r for r in o if r.get("prev_y") is not None and r["y"] == r["prev_y"]]

    def two(rs):
        a, b = metrics(rs, "p_pres1"), metrics(rs)
        a["f1"], b["f1"] = f1(a), f1(b)
        return {"pres1": a, "current": b}
    flagged = [r for r in o if r.get("change_flag") is not None]
    nf = sum(1 for r in flagged if r["change_flag"])
    hit = sum(1 for r in flagged if r["change_flag"] and r["y"] != r["prev_y"])
    return {"version": VERSION, "n": len(o), **two(o), "changed": {"n": len(changed), **two(changed)},
            "same": {"n": len(same), **two(same)}, "no_prev_n": sum(1 for r in o if r.get("prev_y") is None),
            "change_flag": {"cells": len(flagged), "flags": nf, "true": hit, "changes": len(changed),
                            "precision": round(hit / nf, 3) if nf else None,
                            "recall": round(hit / len(changed), 3) if changed else None}}


# ---------------------------------------------------------------- the newest scenes (quay_pres1/<site>.json)
def scene_rows(out: Path, site: dict, calib: dict, keep: int = KEEP_SCENES) -> list[dict]:
    """The newest ``keep`` stored est-2 scenes of the site: per eligible labelled section (naval, berth, >= 50 m) the current
    judge's p (as eval_presence: est-2 p_real of the section's stretch when 50 m or longer, else 0), the section's answers before
    the scene, p_pres1 and change_flag."""
    from . import berths, eval_presence as ep
    sid = site["id"]
    secs = {k: v for k, v in ep.site_sections(out, site).items() if v["eligible"]}
    est = ep.est2_scenes(out, sid)
    if not secs or not est:
        return []
    ans = answers(out, sid)
    b = berths.Berths(site, ep.load_lines(site), naval=True)
    feats = ep.quay_features(out, sid)
    lat0 = float((site.get("quay") or {}).get("est2_lat0") or site.get("center_lat") or 21.2)
    model = calib["model"]
    res = []
    for acq in sorted(est)[-keep:]:
        fs = [f for f in ep.scene_features(feats, est[acq]) if berths.feature_ok(b, f, f["coords"])]
        sts = est[acq].get("stretches") or []
        cells = []
        for sec_id in sorted(secs):
            st = ep.stretch_of_section(secs[sec_id], fs, sts, lat0, "assign-1" if sid in ep.ASSIGN_1_SITES else ep.STRETCH_ASSIGN) if fs else None
            p = ep.judge_est2(st, ep.MIN_LENGTH_M)
            h = history(ans, sec_id, acq)
            cells.append({"section_id": sec_id, "p_current": round(p, 3), "res_rate": round(h["res_rate"], 3), "res_n": h["res_n"],
                          "prev_y": h["prev_y"], "prev_date": h["prev_date"], "p_pres1": round(predict(model, vector(p, h)), 3),
                          "change_flag": change_flag(p, h)})
        res.append({"acquired_utc": acq, "relative_orbit": est[acq].get("relative_orbit"),
                    "present_pres1": sum(1 for c in cells if c["p_pres1"] >= THRESHOLD),
                    "present_current": sum(1 for c in cells if c["p_current"] >= THRESHOLD),
                    "change_flags": sum(1 for c in cells if c["change_flag"]), "sections": cells})
    return res


def site_doc(site: dict, m: str, calib: dict, scenes: list[dict]) -> dict:
    return {"site_id": site["id"], "version": VERSION, "mode": m, "since_utc": (site.get("quay") or {}).get("pres1_since_utc"),
            "calibration": {k: calib.get(k) for k in ("version", "fit_before", "generated_utc")},
            "note": ("試行（並行系列）。公開の画面は読まない。p_pres1＝在の確率（pres-1a：現行の判定 p_current と、区間の撮影の日より前の所有者の答えの割合 "
                     "res_rate＝(在＋1)／(答え＋2)・数 res_n のロジスティック回帰）。change_flag＝現行の判定（p_current ≥ 0.5）が区間の直前の答え prev_y と"
                     "食い違う（1）／同じ（0）／直前の答えなし（null）：目視の確認の順番付け用で、数には入れない。pres-1a は「前と同じ」に強く、"
                     "答えが前回から変わった升（入港・出港）では現行の判定より弱い（評価は eval/presence.json の pres1、sat/pres1_calib.json）"),
            "credit": "Contains modified Copernicus Sentinel data 2024, 2025, 2026",
            "attribution": "岸壁の線・区間：(c) OpenStreetMap contributors（ODbL 1.0）",
            "scenes": scenes}


def write(out: Path, sites: list[dict], calib: dict | None = None) -> dict:
    """quay_pres1/<site>.json for the sites with quay.pres1 trial or on (rewritten whole from the stored files: PU 0). A site that
    is off, or has no row, loses the file of an earlier run. Returns {site_id: number of scenes}. Nothing is written without a
    calibration, and a site without a file is not touched (no empty directory: Zhanjiang's outputs stay as they are)."""
    calib = calib if calib is not None else load_calib()
    res: dict = {}
    d = out / OUT_DIR
    for site in sites:
        m = mode(site)
        p = d / f"{site['id']}.json"
        scenes = scene_rows(out, site, calib) if (calib and m != "off") else []
        if not scenes:
            if p.exists():
                p.unlink()
            continue
        d.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(site_doc(site, m, calib, scenes), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        res[site["id"]] = len(scenes)
    return res


def settings(calib: dict | None = None) -> dict:
    """The definition for meta.json (sat/run.build_meta quay.presence_trial and eval/meta.json)."""
    calib = calib if calib is not None else load_calib()
    return {"version": VERSION, "status": "試行（並行系列）。公開の数・画面は変えない", "files": f"{OUT_DIR}/<地点>.json",
            "calibration": "sat/pres1_calib.json", "design": "docs/data/sat/eval/pres1-design.md",
            "p_pres1": ("在の確率。logit p＝b0＋b1·z(p_current)＋b2·z(res_rate)＋b3·z(res_n)（z＝学習の升の平均と標準偏差で標準化）。"
                        "p_current＝現行の判定（est-2／s1-recount）、res_rate＝(在＋1)／(答え＋2)、res_n＝答えの数：区間の、撮影の日（ラベルの日と S1 の日の早い方）"
                        "より前の日の所有者の答え（S2 の日の答えと S1 の撮影の答え、1 日 1 つ、両方あれば S2）。撮影の日以後の答えは使わない。答えの無い区間は 0.5・0"),
            "change_flag": "現行の判定（p_current ≥ 0.5）が区間の直前の答えと食い違うと 1、同じなら 0、直前の答えが無ければ null。変化の兆し（要確認）で、数には入れない",
            "fit": ("sat/pres1_calib.py：fit_before より前のラベルの日の升（s2 の正解）だけで当て、その升の res もそれぞれの撮影の日より前の答えだけ。"
                    "fit_before 以後の升を標本外として評価する（較正に使った升の指標は出さない）"),
            "sites": "sites.json の quay.pres1：off（既定）／trial（quay_pres1 と評価票の併記だけ）／on（未使用。切り替えの条件は設計書 §4.3）。湛江は off",
            "weakness": "答えが前回から変わった升（入港・出港）では現行の判定より当たらない。変化は change_flag で別に見る",
            "fit_before": (calib or {}).get("fit_before"), "model": (calib or {}).get("model")}


def meta_ref(out: Path | None = None) -> dict:
    """sat/meta.json quay.presence_trial (sat/run.build_meta): never raises."""
    try:
        s = settings()
        return {k: s[k] for k in ("version", "status", "files", "calibration", "design", "p_pres1", "change_flag", "sites", "weakness", "fit_before")}
    except Exception as e:  # noqa: BLE001
        return {"version": VERSION, "status": f"読めない（{type(e).__name__}）"}
