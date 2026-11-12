import io
import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_CHAR = 30_000
MAX_PAGES = 10
MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 MB = 2,097,152 bytes


class ResumeParseError(Exception):
    """Raised when we cannot turn an upload into usable text. The message is safe to show."""

def extract_text(filename: str, data: bytes) -> str:
    """Return clean plain text from a PDF, DOCX or TXT resume."""
    if len(data) > MAX_FILE_BYTES:
        raise ResumeParseError(
            f"This file is {len(data) / 1024 / 1024:.1f} MB. "
            f"Please upload a file under {MAX_FILE_BYTES // (1024 * 1024)} MB."
        )
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ResumeParseError("Please upload a PDF, DOCX or TXT file.")

    try:
        if ext == ".pdf":
            text = _pdf_text(data)
        elif ext == ".docx":
            text = _docx_text(data)
        else:
            text = data.decode("utf-8", errors="replace")
    except ResumeParseError:
        raise
    except Exception as e:
        raise ResumeParseError(
            "We could not read that file. Is it a valid, unlocked document?"
        ) from e
    text = clean_text(text)
    if len(text) < 50:  # far too short to be a real resume
        raise ResumeParseError(
            "We found almost no text in that file. If it is a scanned PDF, "
            "please upload a text-based PDF or a DOCX instead."
        )
    return text

def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    if pages > MAX_PAGES:
        raise ResumeParseError(
            "This resume is very long. Please upload a version under 10 pages."
        )
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def _docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:  # many resumes put skills in tables
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    text = "\n".join(parts)
    if len(text) > MAX_CHAR:
        raise ResumeParseError(
            "Too long resume file."
        )
    return text

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")  # Postgres text columns reject NUL bytes
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()