"""Take the owner's answers in, then make the public files that read them again (by hand, never in a workflow).

    python -m sat.confirm_merge --file confirm-answers-2026-11-27.csv

1. ``sat.confirm merge`` (labels/<site>-confirm-answers.csv, the state and the log of the confirm series).
2. The files of the daily collection that read those answers, made again the way the daily run makes them
   (sat/eval_presence.py refresh): ``opt_fill_prepare`` -> ``write_blind`` (docs/data/sat/quay_blind/<site>.csv) ->
   ``opt_fill_publish`` (docs/data/sat/opt_fill.json). An answer turns the source of a day of the opt-fill columns from the
   judge's p into the owner's answer; without this step the published rows differ from what the code gives for the same
   scenes and tests.test_sat_blind_prior.test_scene_counts_for_the_map fails on main (2026-11-23, PR #177).
3. summary.json and latest.json of the confirm series (the new answers in the numbers and the page).

Nothing else is written: the counts (quay_counts, est-2, naval50, eval/presence.json), blcap and Zhanjiang's files are not
touched (write_blind writes quay_blind only; opt_fill.json is the trial series beside it). Step 2 needs the stored scenes
of the repository (docs/data/sat/quay*/); it runs where the daily collection's files are, without numpy.

The daily collection (sat/run.py) does not import this module (tests/test_sat_confirm.py NotWiredTest): the answers come in
by hand, in a pull request, and the pull request carries the files made again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import confirm


def rebuild(out: Path = confirm.OUT) -> dict:
    """Step 2: quay_blind and opt_fill.json from the answers as they are now (as sat/eval_presence.py refresh makes them)."""
    from . import eval_presence as ep
    sites = ep.load_sites()
    fill, err = ep.opt_fill_prepare(Path(out), sites)
    blind = ep.write_blind(Path(out), sites, fill)
    fill_doc = ep.opt_fill_publish(Path(out), fill, err)
    errors = [e for e in (err, fill_doc.get("error")) if e]
    errors += [f"{k}: {v['error']}" for k, v in fill_doc.items() if isinstance(v, dict) and v.get("error")]
    res = {"quay_blind": blind, "opt_fill": {k: v.get("present_sum") for k, v in fill_doc.items() if isinstance(v, dict)}}
    if errors:          # the daily run goes on with empty columns; by hand the files are not to be committed: exit 1
        res["failed"] = errors
    return res


def merge_and_rebuild(file: Path, out: Path = confirm.OUT, cfg: dict | None = None, now: str | None = None,
                      rebuild_fn=None) -> dict:
    cfg, now = cfg or confirm.load_config(), now or confirm.utcnow()
    res = {"merge": confirm.merge(file, out, cfg, now)}
    m = res["merge"]
    if m["added"] or m["replaced"]:
        try:
            res["rebuild"] = (rebuild_fn or rebuild)(out)
        except Exception as e:                  # noqa: BLE001 (reported; exit 1)
            res["rebuild"] = {"failed": [f"{type(e).__name__}: {e}"[:500]]}
        summ = confirm.summary(out, cfg, now)
        confirm._write_json(Path(out) / confirm.DIR / "summary.json", summ)
        confirm._write_json(Path(out) / confirm.DIR / "latest.json", confirm.latest(out, cfg, now, summ))
    else:
        res["rebuild"] = "nothing new: no file made again"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the owner's answers in, then quay_blind / opt_fill.json / summary made again")
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=confirm.OUT)
    ap.add_argument("--config", type=Path, default=confirm.CONFIG_PATH)
    ap.add_argument("--now", help="UTC time (default: now)")
    a = ap.parse_args(argv)
    doc = merge_and_rebuild(a.file, a.out, confirm.load_config(a.config), a.now)
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    failed = isinstance(doc["rebuild"], dict) and doc["rebuild"].get("failed")
    if failed:
        print(f"::error title=confirm_merge::公開ファイルの作り直しに失敗（このまま commit しない）: {failed}", file=sys.stderr)
    return 1 if doc["merge"]["rejected"] or failed else 0


if __name__ == "__main__":
    sys.exit(main())
