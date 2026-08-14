# simple-rag-app

Simple MVP for a PDF RAG chatbot built with Streamlit, OpenAI, and Pinecone.

## Overview

This project lets a user upload a PDF, extract and chunk its text, embed the chunks with OpenAI, store the vectors in Pinecone, and ask questions that are answered only from the uploaded document.

## Architecture

PDF  
→ Text Extraction  
→ Chunking  
→ Embeddings  
→ Pinecone  
→ Query Embedding  
→ Retrieval  
→ Context Filtering  
→ OpenAI LLM  
→ Answer + Citations

## Project Structure

```text
app.py
src/
  config.py
  embeddings.py
  llm.py
  pdf_processor.py
  retrieval.py
  vector_store.py
tests/
  test_pdf_processor.py
  test_retrieval.py
requirements.txt
.env.example
.gitignore
README.md
```

## Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Configure `.env`.
5. Run Streamlit.

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
- `PINECONE_INDEX_NAME` (default example: `pdf-rag-index`)

## Running Locally

```bash
streamlit run app.py
```

## CI/CD Workflows

This repository includes three lightweight GitHub Actions workflows:

- `build`: installs dependencies and runs `pytest`
- `deploy`: runs on pushes to `main` or manual dispatch and records that the repo is ready for Streamlit Community Cloud auto-deploy
- `health-check`: runs after the deploy workflow succeeds or by manual dispatch, then calls the deployed app URL and verifies the response contains `PDF RAG Chatbot`

### Streamlit Community Cloud Setup

1. In Streamlit Community Cloud, create an app connected to this repository.
2. Select the branch to deploy and use `app.py` as the entry point.
3. Add these app secrets in the Streamlit dashboard:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (optional, defaults to `pdf-rag-index`)

The app reads these values from local environment variables, `.env`, or Streamlit secrets.

### GitHub Repository Variable

Set this repository variable after your Community Cloud app has a public URL:

- `STREAMLIT_APP_URL`

This variable is used by the `health-check` workflow and does not need to be stored as a secret.

## RAG Configuration

- Chunk size: 200 words
- Chunk overlap: 40 words
- TOP_K: 3
- Confidence threshold: 0.80
- Embedding model: `text-embedding-3-small`
- LLM model: `gpt-4o-mini`

The sidebar lets you change the index name, `TOP_K`, and the minimum confidence score.

## Features

- PDF upload through Streamlit
- Page-aware text extraction
- Word-based chunking with overlap
- Deterministic chunk IDs
- Automatic Pinecone serverless index creation
- Retrieval threshold filtering
- Retrieval-score-based precision and recall proxies
- Multi-turn conversation stored in Streamlit session state
- Citation formatting with filename, page, and chunk index
- Refusal when the answer is not sufficiently supported by retrieved context

## Retrieval Metrics

The UI displays:

- Retrieved chunk count
- Relevant chunk count
- Threshold
- Average similarity score
- Recall proxy
- Individual similarity scores

These metrics are **retrieval-score-based proxies**, not true ML precision/recall metrics, because there is no labeled relevance dataset in this MVP.

## Known Limitations

- Precision and recall are retrieval-score-based proxies, not formal evaluation metrics.
- The app retrieves against the current uploaded PDF namespace in Pinecone for the active session.
- Citation placement is model-guided, so the app also shows the retrieved sources separately for transparency.
