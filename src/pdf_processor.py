from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pymupdf


class PDFProcessingError(Exception):
    pass


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    page: int | None


def extract_pdf_pages(pdf_bytes: bytes) -> list[PageText]:
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError("The uploaded file is not a valid PDF.") from exc

    pages: list[PageText] = []
    try:
        for page_index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            normalized_text = normalize_whitespace(raw_text)
            if normalized_text:
                pages.append(PageText(page_number=page_index, text=normalized_text))
    finally:
        document.close()

    if not pages:
        raise PDFProcessingError("The PDF contains no extractable text.")
    return pages


def build_chunks(
    pages: list[PageText],
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words_with_pages: list[tuple[str, int]] = []
    for page in pages:
        words = page.text.split()
        words_with_pages.extend((word, page.page_number) for word in words)

    if not words_with_pages:
        raise PDFProcessingError("The PDF contains no extractable text.")

    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []
    for chunk_index, start in enumerate(range(0, len(words_with_pages), step)):
        end = min(start + chunk_size, len(words_with_pages))
        chunk_words = words_with_pages[start:end]
        if not chunk_words:
            continue
        text = " ".join(word for word, _ in chunk_words)
        page = chunk_words[0][1] if chunk_words else None
        chunk_id = build_chunk_id(source=source, chunk_index=chunk_index, page=page, text=text)
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                chunk_index=chunk_index,
                page=page,
            )
        )
        if end == len(words_with_pages):
            break

    return chunks


def build_chunk_id(source: str, chunk_index: int, page: int | None, text: str) -> str:
    payload = f"{source}|{chunk_index}|{page}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
