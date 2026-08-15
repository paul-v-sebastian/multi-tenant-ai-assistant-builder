from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv

try:
    import streamlit as st
except Exception:  # pragma: no cover - defensive import for non-Streamlit contexts
    st = None


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "my-pdf-index"
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    top_k: int = 3
    min_confidence_score: float = 0.80
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    llm_model: str = "gpt-3.5-turbo"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    def with_overrides(self, **kwargs) -> "AppConfig":
        return replace(self, **kwargs)


def load_config() -> AppConfig:
    def read_secret(name: str, default: str = "") -> str:
        value = os.getenv(name)
        if value:
            return value
        if st is not None:
            try:
                secret_value = st.secrets.get(name)
            except Exception:
                secret_value = None
            if secret_value:
                return str(secret_value)
        return default

    return AppConfig(
        openai_api_key=read_secret("OPENAI_API_KEY"),
        pinecone_api_key=read_secret("PINECONE_API_KEY"),
        pinecone_index_name=read_secret("PINECONE_INDEX_NAME", "my-pdf-index"),
    )
