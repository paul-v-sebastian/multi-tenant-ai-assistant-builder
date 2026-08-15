from __future__ import annotations

import io
import re
from dataclasses import dataclass

import tiktoken
from PyPDF2 import PdfReader


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
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFProcessingError("The uploaded file is not a valid PDF.") from exc

    pages: list[PageText] = []
    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        normalized_text = normalize_whitespace(raw_text)
        if normalized_text:
            pages.append(PageText(page_number=page_index, text=normalized_text))

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

    full_text = "\n".join(page.text for page in pages if page.text)
    if not full_text:
        raise PDFProcessingError("The PDF contains no extractable text.")

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(full_text)
    if not tokens:
        raise PDFProcessingError("The PDF contains no extractable text.")

    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []
    for chunk_index, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            continue
        text = normalize_whitespace(encoding.decode(chunk_tokens))
        if not text:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=build_chunk_id(chunk_index),
                text=text,
                source=source,
                chunk_index=chunk_index,
                page=None,
            )
        )
        if start + chunk_size >= len(tokens):
            break

    return chunks


def build_chunk_id(chunk_index: int) -> str:
    return f"doc-{chunk_index}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
