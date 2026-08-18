# multi-tenant-ai-assistant-builder

A Streamlit app for building and publishing multi-tenant AI assistants backed by PDF knowledge bases. Tenants follow a structured lifecycle — DRAFT → INGESTED → EVALUATED → PUBLISHED — and each gets an isolated Pinecone namespace, persisted files in Supabase Storage, and a shareable JWT-signed URL.

## Features

### Chat
- Conversational interface with preserved message history
- RAG-powered answers with source citations when a PDF is indexed
- Retrieval debug panel showing threshold, precision, recall, and chunk scores
- 👍 / 👎 per-message feedback recorded as Langfuse scores

### Knowledge Base
- Upload a PDF; pages are extracted, token-chunked (200-word chunks, 40-word overlap, `cl100k_base` tokenizer), embedded with OpenAI, and upserted into Pinecone
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

Optional — Supabase (required for tenant management, storage, and eval persistence):

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
