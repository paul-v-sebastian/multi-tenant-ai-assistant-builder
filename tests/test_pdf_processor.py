from __future__ import annotations

import pymupdf
import pytest

from src.pdf_processor import PDFProcessingError, PageText, build_chunks, extract_pdf_pages


def build_pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for page_text in page_texts:
        page = document.new_page()
        if page_text:
            page.insert_text((72, 72), page_text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pdf_pages_returns_text_with_page_numbers():
    pdf_bytes = build_pdf_bytes(["First page text", "Second page text"])

    pages = extract_pdf_pages(pdf_bytes)

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].text == "First page text"
    assert pages[1].text == "Second page text"


def test_build_chunks_preserves_overlap():
    words = [f"word{i}" for i in range(1, 261)]
    pages = [PageText(page_number=1, text=" ".join(words))]

    chunks = build_chunks(pages, source="sample.pdf", chunk_size=200, overlap=40)

    assert len(chunks) == 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert len(first_words) == 200
    assert first_words[-40:] == second_words[:40]


def test_extract_pdf_pages_raises_for_empty_pdf():
    pdf_bytes = build_pdf_bytes([""])

    with pytest.raises(PDFProcessingError, match="no extractable text"):
        extract_pdf_pages(pdf_bytes)
