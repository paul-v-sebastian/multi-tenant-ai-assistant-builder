from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "pdf-rag-index"
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    top_k: int = 3
    min_confidence_score: float = 0.80
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    llm_model: str = "gpt-4o-mini"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    def with_overrides(self, **kwargs) -> "AppConfig":
        return replace(self, **kwargs)


def load_config() -> AppConfig:
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "pdf-rag-index"),
    )

