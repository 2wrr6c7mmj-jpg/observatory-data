"""Paths and the command line shared by the pres-3 steps. No third-party imports here."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAT = REPO / "docs" / "data" / "sat"            # read only (labels, eval/presence.json)
INPUTS = Path(__file__).resolve().parent / "inputs"
PORTS = ["ZJG", "SYL", "SLP", "QGZ", "ZDH", "QJG", "QDH", "XSN", "LSN"]   # the 9 ports that have the owner's answers
ENV_WORK = "PRES3_WORK"


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--work", help=f"working directory for the fetched windows and every intermediate file "
                                   f"(default: ${ENV_WORK}). Several GB. Never inside the repository's docs/.")
    ap.add_argument("--ports", default=",".join(PORTS), help="comma separated site ids (default: the 9 ports)")
    return ap


def work_dir(arg: str | None = None) -> Path:
    p = arg or os.environ.get(ENV_WORK)
    if not p:
        raise SystemExit(f"give --work DIR or set {ENV_WORK} (the directory that holds data/<port>/*.npz)")
    w = Path(p).resolve()
    if SAT.resolve() in (w, *w.parents):
        raise SystemExit("the working directory must not be inside docs/data/sat (published files)")
    w.mkdir(parents=True, exist_ok=True)
    return w


def ports_of(arg: str) -> list[str]:
    return [p for p in arg.split(",") if p]
