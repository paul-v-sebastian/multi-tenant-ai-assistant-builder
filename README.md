# simple-rag-app

Simple Streamlit chat app with incremental PDF-based RAG on top of a preserved chat history.

## Overview

This project includes:

- Streamlit chat interface
- OpenAI-backed assistant responses
- Multi-turn conversation history in Streamlit session state
- PDF upload and text extraction with PyPDF2
- Token chunking with `tiktoken` `cl100k_base`
- OpenAI embeddings stored in Pinecone

## Project Structure

```text
app.py
src/
  config.py
  llm.py
  pdf_processor.py
  embeddings.py
  retrieval.py
  vector_store.py
tests/
  test_app_state.py
  test_pdf_processor.py
  test_retrieval.py
requirements.txt
.env.example
README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

Required:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY` for PDF indexing and retrieval

Optional:

- `PINECONE_INDEX_NAME` (default: `my-pdf-index`)

Current in-code defaults in `src/config.py`:

- chunk size: `200`
- chunk overlap: `40`
- top k: `3`
- min confidence score: `0.80`
- embedding model: `text-embedding-ada-002`
- chat model: `gpt-3.5-turbo`

## Running Locally

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest
```
