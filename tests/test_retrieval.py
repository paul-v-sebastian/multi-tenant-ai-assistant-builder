from __future__ import annotations

from src.retrieval import RetrievedChunk, calculate_retrieval_metrics, filter_matches_by_threshold


def test_filter_matches_by_threshold():
    matches = [
        RetrievedChunk("1", "A", "doc.pdf", 0, 1, 0.91),
        RetrievedChunk("2", "B", "doc.pdf", 1, 1, 0.80),
        RetrievedChunk("3", "C", "doc.pdf", 2, 2, 0.42),
    ]

    filtered = filter_matches_by_threshold(matches, 0.80)

    assert [match.chunk_id for match in filtered] == ["1", "2"]


def test_calculate_retrieval_metrics_returns_score_based_proxies():
    matches = [
        RetrievedChunk("1", "A", "doc.pdf", 0, 1, 0.91),
        RetrievedChunk("2", "B", "doc.pdf", 1, 2, 0.84),
        RetrievedChunk("3", "C", "doc.pdf", 2, 3, 0.52),
    ]

    metrics = calculate_retrieval_metrics(matches, top_k=3, min_confidence_score=0.80)

    assert metrics["retrieved_count"] == 3
    assert metrics["relevant_count"] == 2
    assert metrics["average_score"] == (0.91 + 0.84) / 2
    assert metrics["precision"] == metrics["average_score"]
    assert metrics["recall"] == 2 / 3
    assert metrics["scores"] == [0.91, 0.84, 0.52]
