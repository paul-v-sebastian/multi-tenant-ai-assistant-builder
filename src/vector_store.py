from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from src.pdf_processor import DocumentChunk
from src.retrieval import RetrievedChunk, calculate_retrieval_metrics, filter_matches_by_threshold


class VectorStoreError(Exception):
    pass


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str,
        region: str,
    ) -> None:
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self._ensure_index()
        self.index = self.client.Index(index_name)

    def _ensure_index(self) -> None:
        try:
            existing_indexes = set(self.client.list_indexes().names())
            if self.index_name in existing_indexes:
                return
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            for _ in range(30):
                description = self.client.describe_index(self.index_name)
                status = getattr(description, "status", {}) or {}
                ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
                if ready:
                    return
                time.sleep(1)
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise VectorStoreError(f"Pinecone index creation failure: {exc}") from exc
        raise VectorStoreError("Pinecone index creation failure: index was not ready in time.")

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        namespace: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Embedding count did not match chunk count.")
        records = [
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "source_filename": chunk.source,
                    "chunk_index": chunk.chunk_index,
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        try:
            self.index.upsert(vectors=records, namespace=namespace)
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise VectorStoreError(f"Failed to upsert vectors into Pinecone: {exc}") from exc

    def query(
        self,
        embedding: list[float],
        namespace: str,
        top_k: int,
        min_confidence_score: float,
    ) -> dict:
        try:
            response = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            )
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise VectorStoreError(f"Failed to query Pinecone: {exc}") from exc

        response_matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        matches = [
            RetrievedChunk(
                chunk_id=_get_match_value(match, "id", ""),
                text=_get_metadata_value(match, "text", ""),
                source=_get_metadata_value(
                    match,
                    "source_filename",
                    _get_metadata_value(match, "source", "unknown"),
                ),
                chunk_index=int(_get_metadata_value(match, "chunk_index", 0)),
                page=None,
                score=float(_get_match_value(match, "score", 0.0)),
            )
            for match in response_matches
        ]
        relevant_matches = filter_matches_by_threshold(matches, min_confidence_score)
        metrics = calculate_retrieval_metrics(matches, top_k, min_confidence_score)
        return {
            "matches": matches,
            "relevant_matches": relevant_matches,
            "relevant_count": len(relevant_matches),
            "metrics": metrics,
        }

    def describe_namespace(self, namespace: str) -> dict:
        try:
            stats = self.index.describe_index_stats()
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise VectorStoreError(f"Failed to describe Pinecone index: {exc}") from exc

        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
        namespace_stats = namespaces.get(namespace, {}) if isinstance(namespaces, dict) else getattr(namespaces, namespace, {})
        vector_count = (
            namespace_stats.get("vector_count", 0)
            if isinstance(namespace_stats, dict)
            else getattr(namespace_stats, "vector_count", 0)
        )
        total_vector_count = stats.get("total_vector_count", 0) if isinstance(stats, dict) else getattr(stats, "total_vector_count", 0)
        dimension = stats.get("dimension", self.dimension) if isinstance(stats, dict) else getattr(stats, "dimension", self.dimension)
        return {
            "namespace": namespace,
            "namespace_vector_count": int(vector_count),
            "total_vector_count": int(total_vector_count),
            "dimension": int(dimension),
        }


def _get_match_value(match, key: str, default):
    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


def _get_metadata_value(match, key: str, default):
    metadata = _get_match_value(match, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)
