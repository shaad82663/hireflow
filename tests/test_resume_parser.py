import io

import pytest
from docx import Document

from hireflow.rag.chunking import chunk_text
from hireflow.services.resume_parser import ResumeParseError, clean_text, extract_text

RESUME = "Asha Rao\nBackend engineer with 5 years of Python, FastAPI and PostgreSQL experience."


def make_docx() -> bytes:
    doc = Document()
    doc.add_paragraph(RESUME)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Kubernetes, AWS"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_txt():
    assert extract_text("cv.TXT", RESUME.encode()) == RESUME


def test_docx_includes_tables():
    text = extract_text("cv.docx", make_docx())
    assert "FastAPI" in text
    assert "Skills | Kubernetes, AWS" in text


def test_rejects_unsupported_type():
    with pytest.raises(ResumeParseError, match="PDF, DOCX or TXT"):
        extract_text("cv.png", b"...")


def test_rejects_corrupt_pdf():
    with pytest.raises(ResumeParseError, match="could not read"):
        extract_text("cv.pdf", b"this is not really a pdf")


def test_rejects_almost_empty_text():
    with pytest.raises(ResumeParseError, match="scanned PDF"):
        extract_text("cv.txt", b"   hello   ")


def test_clean_text():
    assert clean_text("a\x00b   c\n\n\n\nd") == "ab c\n\nd"


def test_chunking_splits_long_text_with_overlap():
    text = "\n\n".join(
        f"Project {i}: built a FastAPI service handling payments. " * 4 for i in range(12)
    )
    chunks = chunk_text(text)
    assert len(chunks) > 3
    assert all(len(c) <= 800 for c in chunks)