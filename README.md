# multi-tenant-ai-assistant-builder

Streamlit-based AI assistant builder that is evolving from a single-tenant PDF RAG demo into a multi-tenant assistant platform.

## Current Scope

The repository currently includes:

- A Streamlit chat experience with preserved conversation history
- PDF upload, extraction, chunking, embedding, and Pinecone-backed retrieval
- An Evals tab for CSV-driven ground-truth evaluation
- Langfuse tracing for upload and chat flows
- Thumbs-up / thumbs-down feedback on assistant responses
- Phase 1 tenant foundations:
  - tenant-aware session-state keys
  - tenant status lifecycle constants: `DRAFT -> INGESTED -> EVALUATED -> PUBLISHED`
  - tenant-first Pinecone namespace selection

The app is still in a transition state: core chat/evals flows work as before, while multi-tenant assistant-builder capabilities are being added in phases.

## Project Structure

```text
app.py
src/
  config.py
  embeddings.py
  llm.py
  pdf_processor.py
  retrieval.py
  tenants.py
  tracing.py
  vector_store.py
tests/
  test_app_state.py
  test_pdf_processor.py
  test_retrieval.py
  test_tenants.py
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
- `PINECONE_API_KEY`

Optional:

- `PINECONE_INDEX_NAME` (default: `my-pdf-index`)
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_HOST` (default: `https://cloud.langfuse.com`)

Current in-code defaults in `src/config.py`:

- chunk size: `200`
- chunk overlap: `40`
- top k: `3`
- min confidence score: `0.80`
- embedding model: `text-embedding-ada-002`
- chat model: `gpt-3.5-turbo`
- Pinecone cloud: `aws`
- Pinecone region: `us-east-1`

## Running Locally

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest
```
