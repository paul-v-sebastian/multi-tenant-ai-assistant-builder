# multi-tenant-ai-assistant-builder

AI Assistant Builder is a Streamlit app for creating, evaluating, and publishing multi-tenant AI assistants backed by PDF knowledge bases. Each tenant follows a structured lifecycle — DRAFT → INGESTED → EVALUATED → PUBLISHED — with isolated retrieval data in Pinecone, tenant metadata and persisted files in Supabase, and a shareable JWT-signed URL for published assistants.

## Product Overview

This project is an MVP for teams that want to turn static documents into customer- or tenant-specific AI assistants without building a custom retrieval pipeline per customer.

Core operator workflow:

1. Create or sign in to a tenant
2. Upload and index a PDF knowledge base
3. Test the assistant through the built-in chat UI
4. Tune retrieval settings
5. Run automated evaluations from a ground-truth CSV
6. Publish the assistant and share it through a signed URL

## Technology Overview

- **Frontend / orchestration:** Streamlit
- **LLM and embeddings:** OpenAI
- **Vector store:** Pinecone
- **Tenant data and file persistence:** Supabase
- **Tracing and feedback analytics:** Langfuse

## Architecture at a Glance

- Each tenant is assigned an isolated Pinecone namespace
- Tenant records, retrieval configuration, lifecycle state, and evaluation logs are stored in Supabase
- PDFs are persisted to Supabase Storage under `pdfs/<tenant_id>/<filename>`
- Evaluation CSVs are persisted to Supabase Storage under `eval-csvs/<tenant_id>/<filename>`
- Published assistants are exposed through HMAC-SHA256 signed JWT share links
- Retrieval and chat traces can be sent to Langfuse when tracing credentials are configured

## Features

### Chat
- Conversational interface with preserved message history
- RAG-powered answers with source citations when a PDF is indexed
- Retrieval debug panel showing threshold, precision, recall, and chunk scores
- 👍 / 👎 per-message feedback recorded as Langfuse scores
- Standalone shared chat experience for published assistants

### Knowledge Base
- Upload a PDF; pages are extracted, token-chunked with the vendored `cl100k_base` tokenizer (target chunk size 200, overlap 40), embedded with OpenAI, and upserted into Pinecone
- Each tenant gets its own Pinecone namespace for data isolation
- PDFs are persisted to Supabase Storage (`pdfs/<tenant_id>/<filename>`) when a tenant is authenticated

### Evals & Configs
- Upload a ground-truth CSV (`Query`, `Expected Response` columns) to run automated evaluation
- Each query is answered by the live retrieval pipeline and judged by `gpt-4o-mini` (score 1–5)
- Results shown in a table with average score; downloadable as `eval_report.csv`
- Eval logs are persisted to Supabase (`eval_logs` table) for authenticated tenants
- Retrieval settings (index name, top-k, min confidence) are configurable per tenant and saved to Supabase

### Tenant Management
- Create a new tenant with a name and bcrypt-hashed passkey; stored in Supabase with `DRAFT` status
- Sign in to an existing tenant by UUID + passkey; session state and retrieval config are restored
- Sidebar shows live Supabase connection status (grey / red / green) and the current tenant's lifecycle badge
- Tenant status advances automatically: `DRAFT → INGESTED` on PDF upload, `INGESTED → EVALUATED` after eval run, `EVALUATED → PUBLISHED` on publish

### Publishing & Sharing
- Publish button available after evaluation; generates an HMAC-SHA256 signed JWT share URL (`/?token=<jwt>`)
- Share token and `PUBLISHED` status are persisted to Supabase
- Shareable URL is displayed in the UI for copying

### Tracing (optional)
- Langfuse v4 tracing for upload (`pdf_upload_and_index` span) and chat (`chat_turn` span with retrieval metrics)

## Tenant Lifecycle

The assistant builder uses a linear tenant lifecycle:

- `DRAFT` — tenant exists but no indexed knowledge base yet
- `INGESTED` — a PDF has been uploaded and indexed
- `EVALUATED` — at least one evaluation run has completed
- `PUBLISHED` — the assistant has been published and given a share URL

Lifecycle transitions are enforced in application logic and persisted in Supabase.

## Project Structure

```text
app.py
src/
  config.py          — AppConfig dataclass and environment loader
  embeddings.py      — OpenAI embedding service
  llm.py             — OpenAI chat and judge service
  pdf_processor.py   — PDF extraction and token-aware chunking
  retrieval.py       — Retrieval result formatting
  share.py           — JWT share-token generation and verification
  storage.py         — Supabase Storage helpers (PDFs and eval CSVs)
  supabase_client.py — Supabase client singleton and tenant CRUD
  tenants.py         — Tenant status lifecycle and namespace utilities
  tracing.py         — Langfuse singleton
  vector_store.py    — Pinecone upsert and query
  db/
    migrations/
      001_initial_schema.sql          — tenants + eval_logs tables with RLS
      002_allow_draft_tenant_bootstrap.sql
tests/
  test_app_state.py
  test_db_migrations.py
  test_pdf_processor.py
  test_retrieval.py
  test_share.py
  test_storage.py
  test_supabase_client.py
  test_tenant_auth.py
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
# Edit .env with your credentials (see Environment Variables below)
```

## Environment Variables

Required:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `PINECONE_API_KEY` | Pinecone API key |

Optional for basic non-tenant local experimentation, but required for the full builder flow (tenant management, storage, evaluation persistence, publishing, and share links):

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role or anon key |
| `JWT_SECRET` | Secret for signing share tokens (required to publish) |
| `APP_BASE_URL` | Public app URL prepended to share tokens (e.g. `https://myapp.streamlit.app`) |

Optional — Pinecone:

| Variable | Default | Description |
|---|---|---|
| `PINECONE_INDEX_NAME` | `my-pdf-index` | Target Pinecone index |

Optional — Langfuse tracing:

| Variable | Default | Description |
|---|---|---|
| `LANGFUSE_SECRET_KEY` | | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | | Langfuse public key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse host |

In-code defaults (`src/config.py`):

| Setting | Default |
|---|---|
| Chunk size | 200 words |
| Chunk overlap | 40 words |
| Top-k | 3 |
| Min confidence score | 0.80 |
| Embedding model | `text-embedding-ada-002` |
| Chat model | `gpt-3.5-turbo` |
| Pinecone cloud | `aws` |
| Pinecone region | `us-east-1` |

## Current Scope and Limitations

- The knowledge base flow currently targets **PDF** ingestion
- Retrieval configuration is intentionally lightweight: index name, top-k, and minimum confidence
- Share links are JWT-signed, but the current implementation does not add token expiry
- The product includes evaluation and tracing, but not a full admin analytics dashboard

## Database Migrations

Run the SQL migrations in order against your Supabase project (SQL editor or `supabase db push`):

```
src/db/migrations/001_initial_schema.sql
src/db/migrations/002_allow_draft_tenant_bootstrap.sql
```

Migrations create the `tenants` and `eval_logs` tables with Row Level Security (RLS) policies that isolate each tenant's data.

## Running Locally

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest
```
