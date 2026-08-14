from __future__ import annotations

from openai import OpenAI


class EmbeddingServiceError(Exception):
    pass


class EmbeddingService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise EmbeddingServiceError(f"Embedding API failure: {exc}") from exc
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        if not embeddings:
            raise EmbeddingServiceError("Embedding API failure: no embedding returned for the query.")
        return embeddings[0]

