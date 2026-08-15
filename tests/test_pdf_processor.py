from __future__ import annotations

import pymupdf
import pytest
import tiktoken

from src.pdf_processor import PDFProcessingError, PageText, build_chunk_id, build_chunks, extract_pdf_pages


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
    words = [f"word{i}" for i in range(1, 401)]
    pages = [PageText(page_number=1, text=" ".join(words))]

    chunks = build_chunks(pages, source="sample.pdf", chunk_size=200, overlap=40)

    assert len(chunks) >= 2
    encoding = tiktoken.get_encoding("cl100k_base")
    first_tokens = encoding.encode(chunks[0].text)
    second_tokens = encoding.encode(chunks[1].text)
    assert len(first_tokens) <= 200
    assert first_tokens[-40:] == second_tokens[:40]
    assert chunks[0].chunk_id == "doc-0"
    assert chunks[1].chunk_id == "doc-1"


def test_build_chunk_id_matches_poc_format():
    assert build_chunk_id(7) == "doc-7"


def test_extract_pdf_pages_raises_for_empty_pdf():
    pdf_bytes = build_pdf_bytes([""])

    with pytest.raises(PDFProcessingError, match="no extractable text"):
        extract_pdf_pages(pdf_bytes)
