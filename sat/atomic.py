"""Atomic file writes for the published records (CSV, JSON, images).

A record is written to a temporary file in the same directory and moved over the old one with ``os.replace``:
a reader (and the Publish step of the workflow, which commits whatever is in docs/data/sat even when the count
failed) sees either the old file or the new one, never a half-written file.
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path


def write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if path.exists():  # keep the mode of the file it replaces (mkstemp makes 0600)
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    write_bytes(path, text.encode(encoding))


def csv_text(rows: list[dict], fields: list[str]) -> str:
    """The CSV of the records (header, ``\\n`` line ends, None and missing as empty, other keys ignored)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return buf.getvalue()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    write_text(path, csv_text(rows, fields))
