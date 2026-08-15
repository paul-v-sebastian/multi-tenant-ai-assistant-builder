from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    page: int | None
    score: float


def filter_matches_by_threshold(
    matches: list[RetrievedChunk],
    min_confidence_score: float,
) -> list[RetrievedChunk]:
    return [match for match in matches if match.score >= min_confidence_score]


def calculate_retrieval_metrics(
    matches: list[RetrievedChunk],
    top_k: int,
    min_confidence_score: float,
) -> dict:
    relevant_matches = filter_matches_by_threshold(matches, min_confidence_score)
    average_score = (
        sum(match.score for match in relevant_matches) / len(relevant_matches)
        if relevant_matches
        else 0.0
    )
    return {
        "retrieved_count": len(matches),
        "relevant_count": len(relevant_matches),
        "threshold": min_confidence_score,
        "average_score": average_score,
        "precision": average_score,
        "recall": (len(relevant_matches) / top_k) if top_k else 0.0,
        "scores": [match.score for match in matches],
    }


def format_citation(chunk: RetrievedChunk) -> str:
    if chunk.page is not None:
        return f"[Source: {chunk.source}, page {chunk.page}, chunk {chunk.chunk_index}]"
    return f"[Source: {chunk.source}, chunk {chunk.chunk_index}]"


def format_metrics_for_display(metrics: dict) -> dict[str, str]:
    return {
        "Retrieved": str(metrics["retrieved_count"]),
        "Relevant": str(metrics["relevant_count"]),
        "Threshold": f'{metrics["threshold"]:.2f}',
        "Precision": f'{metrics["precision"]:.2f}',
        "Recall": f'{metrics["recall"]:.2f}',
        "Individual similarity scores": ", ".join(f"{score:.2f}" for score in metrics["scores"]) or "None",
    }
