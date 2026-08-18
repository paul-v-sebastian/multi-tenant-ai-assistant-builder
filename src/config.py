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
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    supabase_url: str = ""
    supabase_key: str = ""
    jwt_secret: str = ""
    app_base_url: str = ""

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
        langfuse_secret_key=read_secret("LANGFUSE_SECRET_KEY"),
        langfuse_public_key=read_secret("LANGFUSE_PUBLIC_KEY"),
        langfuse_host=read_secret("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        supabase_url=read_secret("SUPABASE_URL"),
        supabase_key=read_secret("SUPABASE_KEY"),
        jwt_secret=read_secret("JWT_SECRET"),
        app_base_url=read_secret("APP_BASE_URL"),
    )
