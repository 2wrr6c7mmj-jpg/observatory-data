"""blind-1: quays where a warship is nearly always present ("resident" sections), from the owner's answers.

The quay count compares a scene with the per-orbit baseline (sat/baseline.py: the 10th percentile of ~200 days as "the empty
quay"). A berth where a ship lies in nearly every scene has no empty baseline, so a ship there gives no change and is not
counted (the blind stretches; 28 of the 28+35 "present" cells of Guzhenkou were missed on 2026-11-10, 23 of them blind).

blind-1 does not trust the blind measure alone (a third of the blind cells of Guzhenkou hold no ship of 50 m: structures,
barges, floating things). It uses the owner's eye instead: a section where the owner answered "a surface ship of 50 m or more
is present" in at least RESIDENT_MIN of the Sentinel-1 scenes checked (labels/<site>-s1-answers.csv, the S1 check; at least
MIN_ANSWERS scenes) (a site without an S1 check: in the days of its S2 answers, labels/<site>-s2-answers.csv) is a resident section, and the judge's probability there is raised to the section's share of "present"
(Laplace-smoothed). The target is the warship whether or not it is in service (user's instruction 2026-11-11: laid-up and
immobile warships count; structures, barges and floating docks do not: that is what the owner's answer separates).

    p = max(p_current, p_blind),  p_blind = (k + 1) / (n + 2) when n >= MIN_ANSWERS and k / n >= RESIDENT_MIN, else 0

For the evaluation the scene being judged is left out of its own section's answers (leave one scene out).

    python -m sat.blind_prior QGZ [--out docs/data/sat]     # writes sat/blind_prior/QGZ.json (the table of the sections)

Pure Python (no numpy).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "sat"
PRIOR_DIR = Path(__file__).resolve().parent / "blind_prior"
VERSION = "blind-1"
MIN_ANSWERS = 3        # a section needs this many checked scenes (beside the one being judged)
RESIDENT_MIN = 0.75    # share of "present" among them for a resident section


S1_FILE = "{site}-s1-answers.csv"                       # the S1 check (the same scene): preferred
S2_FILES = ("{site}-s2-answers.csv", "{site}-oos-answers.csv")   # the S2 day's answers: used when the site has no S1 check


def _rows(p: Path):
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def answers_source(out: Path, site_id: str) -> str | None:
    """"s1" (labels/<site>-s1-answers.csv), "s2" (the S2 day's answers) or None: where the site's resident table comes from."""
    lab = Path(out) / "labels"
    if any(r.get("verified") == "1" and r.get("s1_acquired_utc") for r in _rows(lab / S1_FILE.format(site=site_id))):
        return "s1"
    if any(r.get("verified") == "1" and r.get("date") for n in S2_FILES for r in _rows(lab / n.format(site=site_id))):
        return "s2"
    return None


def answers(out: Path, site_id: str) -> dict[str, list[tuple[str, int]]]:
    """{section_id: [(key, 1 present / 0 absent), ...]} of the owner's answers; "cannot judge" (verified 0) left out. The key is the
    Sentinel-1 scene (acquired_utc) of the S1 check, or, for a site without one, the Sentinel-2 day (YYYY-MM-DD) of the S2 answers:
    whether a quay is resident does not depend on the hour, so the S2 day's answer serves as well."""
    lab = Path(out) / "labels"
    src = answers_source(out, site_id)
    seen: dict = {}
    if src == "s1":
        for r in _rows(lab / S1_FILE.format(site=site_id)):
            if r.get("site_id", site_id) != site_id or r.get("verified") != "1" or not r.get("s1_acquired_utc"):
                continue
            seen[(r["section_id"], r["s1_acquired_utc"])] = 1 if r.get("verified_occupied") == "1" else 0
    elif src == "s2":
        for n in S2_FILES:
            for r in _rows(lab / n.format(site=site_id)):
                if r.get("site_id", site_id) != site_id or r.get("verified") != "1" or not r.get("date"):
                    continue
                seen[(r["section_id"], r["date"])] = 1 if r.get("verified_occupied") == "1" else 0
    out_: dict[str, list[tuple[str, int]]] = {}
    for (sec, key), y in sorted(seen.items()):
        out_.setdefault(sec, []).append((key, y))
    return out_


def p_blind(ans: dict, section_id: str, exclude_s1=None) -> float:
    """The resident probability of a section (0 when it is not resident or has too few answers). ``exclude_s1``: the key of the
    scene being judged (its S1 time or its label day), or a set of such keys; its own answer is left out."""
    ex = set() if exclude_s1 is None else {exclude_s1} if isinstance(exclude_s1, str) else set(exclude_s1)
    ys = [y for key, y in ans.get(section_id, []) if key not in ex]
    n, k = len(ys), sum(ys)
    if n < MIN_ANSWERS or k / n < RESIDENT_MIN:
        return 0.0
    return round((k + 1) / (n + 2), 3)


def build(out: Path, site_id: str) -> dict:
    ans = answers(out, site_id)
    secs = {}
    for sec, rs in ans.items():
        n, k = len(rs), sum(y for _, y in rs)
        secs[sec] = {"present": k, "answers": n, "resident": bool(n >= MIN_ANSWERS and k / n >= RESIDENT_MIN), "p_blind": p_blind(ans, sec)}
    src = answers_source(out, site_id)
    return {"version": VERSION, "site_id": site_id, "answers": src,
            "source": (f"labels/{site_id}-s1-answers.csv（所有者の同日の答え、verified 1）" if src == "s1" else
                       f"labels/{site_id}-s2-answers.csv・-oos-answers.csv（所有者の S2 の日の答え、verified 1。同日の答えが無い港）"),
            "rule": f"答えが {MIN_ANSWERS} 撮影以上あり、在の割合が {RESIDENT_MIN:g} 以上の区間＝常在。p_blind＝(在＋1)／(答え＋2)、常在でなければ 0",
            "target": "軍艦（50 m 以上の水上艦）。稼働状態は問わない。構造物・艀・浮体は所有者の答えで不在",
            "scenes": sorted({key for rs in ans.values() for key, _ in rs}), "n_resident": sum(1 for s in secs.values() if s["resident"]),
            "sections": dict(sorted(secs.items()))}


def write(out: Path, site_id: str, d: Path = PRIOR_DIR) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{site_id}.json"
    p.write_text(json.dumps(build(out, site_id), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sites", nargs="+")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    for sid in a.sites:
        p = write(a.out, sid)
        doc = json.loads(p.read_text(encoding="utf-8"))
        print(f"{sid}: {len(doc['sections'])} sections, {doc['n_resident']} resident -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
