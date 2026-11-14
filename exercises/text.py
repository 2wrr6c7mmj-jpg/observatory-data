"""Plain text from an article page or a PDF (no extra packages needed in tests)."""
import html
import re
import shutil
import subprocess
import tempfile

_DROP = re.compile(r"<(script|style|noscript|svg|nav|header|footer|form|aside)\b.*?</\1\s*>", re.S | re.I)
_BLOCK = re.compile(r"<(p|li|h[1-6]|td|th|blockquote|div|br)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(raw: str, limit: int = 6000) -> str:
    """Paragraph text of a page: drops scripts and navigation, keeps lines that
    read like sentences (or table cells of an announcement)."""
    s = _DROP.sub(" ", raw)
    s = _BLOCK.sub("\n", s)
    s = html.unescape(_TAG.sub(" ", s))
    lines = []
    for line in s.split("\n"):
        line = re.sub(r"[ \t　\xa0]+", " ", line).strip()
        if len(line) < 25 and not re.search(r"[0-9０-９].*[日月]|期間|場所|部隊|海域", line):
            continue
        if lines and line == lines[-1]:
            continue
        lines.append(line)
    out = "\n".join(lines)
    return out[:limit]


def pdf_to_text(data: bytes, limit: int = 6000) -> str:
    """pdftotext when the runner has it, else pypdf, else empty."""
    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(data)
            f.flush()
            try:
                out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", f.name, "-"],
                                     capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
                return _squeeze(out)[:limit]
            except Exception:  # noqa: BLE001 - fall through to pypdf
                pass
    try:
        import io
        from pypdf import PdfReader  # installed by the workflow, not needed by tests
        reader = PdfReader(io.BytesIO(data))
        out = "\n".join((p.extract_text() or "") for p in reader.pages[:6])
        return _squeeze(out)[:limit]
    except Exception:  # noqa: BLE001
        return ""


def _squeeze(s: str) -> str:
    s = re.sub(r"[ \t　]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()
