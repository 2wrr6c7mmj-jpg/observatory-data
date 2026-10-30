"""Probe helper: put text or a small file into GitHub Actions annotations.

Job logs and artifacts live on blob storage that some readers cannot reach;
annotations are served by the REST API (check-runs/{id}/annotations).

    python -m sat.annotate text TITLE FILE        # tail of a text file
    python -m sat.annotate b64 TITLE FILE [CHUNK] # a binary file, base64 in parts
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path


def esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    kind, title, path = argv[:3]
    p = Path(path)
    if not p.exists():
        print(f"::warning title={title}::{path} がありません")
        return 0
    if kind == "text":
        text = p.read_text(encoding="utf-8", errors="replace")[-3900:]
        print(f"::notice title={title}::{esc(text)}")
        return 0
    # Annotation messages are cut at 4096 characters, and a step keeps at most
    # 10 annotations of each level, so parts cycle through notice/warning/error.
    chunk = int(argv[3]) if len(argv) > 3 else 4000
    data = base64.b64encode(p.read_bytes()).decode()
    parts = [data[i:i + chunk] for i in range(0, len(data), chunk)] or [""]
    for i, part in enumerate(parts, 1):
        level = ("notice", "warning", "error")[(i - 1) // 10 % 3]
        print(f"::{level} title={title} {i}/{len(parts)}::{part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
