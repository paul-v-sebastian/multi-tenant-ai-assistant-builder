# simple-rag-app

Simple MVP for a Streamlit chat app that wraps the OpenAI Chat Completions API and preserves conversation history.

## Overview

This project now focuses on a minimal chat experience with optional source-backed responses:

- Streamlit chat interface
- OpenAI-backed assistant responses
- Multi-turn conversation history in Streamlit session state
- Optional Pinecone retrieval with inline source inspection

## Project Structure

```text
app.py
src/
  config.py
  llm.py
tests/
  test_app_state.py
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

Optional:

- `OPENAI_MODEL` is controlled in code through `src/config.py` (default: `gpt-4o-mini`)
- `PINECONE_API_KEY` enables retrieved source chunks in assistant responses
- `PINECONE_INDEX_NAME` overrides the default `pdf-rag-index`

## Running Locally

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest
```
