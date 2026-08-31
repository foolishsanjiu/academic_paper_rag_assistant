"""Rank-aware retrieval metrics for Chunk-level relevance judgments."""

from __future__ import annotations

from math import log2
from typing import Any, Iterable


DEFAULT_RANK_CUTOFFS = (5, 8, 10, 20)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k 必须是大于等于 1 的整数。")


def _relevance_map(qrels: list[dict[str, Any]]) -> dict[str, int]:
    relevance_by_chunk: dict[str, int] = {}
    for judgment in qrels:
        chunk_id = str(judgment.get("chunk_id", "")).strip()
        relevance = judgment.get("relevance")
        if not chunk_id:
            raise ValueError("qrel chunk_id 不能为空。")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, int)
            or relevance not in {0, 1, 2, 3}
        ):
            raise ValueError("qrel relevance 必须是 0、1、2 或 3。")
        if chunk_id in relevance_by_chunk:
            raise ValueError(f"qrels 包含重复 chunk_id：{chunk_id}")
        relevance_by_chunk[chunk_id] = relevance
    return relevance_by_chunk


def _deduplicate_ranking(retrieved_chunk_ids: Iterable[str]) -> list[str]:
    ranking: list[str] = []
    seen: set[str] = set()
    for raw_chunk_id in retrieved_chunk_ids:
        chunk_id = str(raw_chunk_id).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        ranking.append(chunk_id)
    return ranking


def recall_at_k(
    qrels: list[dict[str, Any]],
    retrieved_chunk_ids: Iterable[str],
    k: int,
) -> float | None:
    """Return the fraction of judged relevant Chunks retrieved in Top-k."""
    _validate_k(k)
    relevance_by_chunk = _relevance_map(qrels)
    relevant_ids = {
        chunk_id
        for chunk_id, relevance in relevance_by_chunk.items()
        if relevance > 0
    }
    if not relevant_ids:
        return None
    ranking = _deduplicate_ranking(retrieved_chunk_ids)[:k]
    return len(relevant_ids & set(ranking)) / len(relevant_ids)


def reciprocal_rank_at_k(
    qrels: list[dict[str, Any]],
    retrieved_chunk_ids: Iterable[str],
    k: int,
) -> float | None:
    """Return reciprocal rank of the first relevant Chunk in Top-k."""
    _validate_k(k)
    relevance_by_chunk = _relevance_map(qrels)
    if not any(relevance > 0 for relevance in relevance_by_chunk.values()):
        return None

    for rank, chunk_id in enumerate(
        _deduplicate_ranking(retrieved_chunk_ids)[:k],
        start=1,
    ):
        if relevance_by_chunk.get(chunk_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    qrels: list[dict[str, Any]],
    retrieved_chunk_ids: Iterable[str],
    k: int,
) -> float | None:
    """Return normalized discounted cumulative gain using grades 0..3."""
    _validate_k(k)
    relevance_by_chunk = _relevance_map(qrels)
    positive_grades = sorted(
        (
            relevance
            for relevance in relevance_by_chunk.values()
            if relevance > 0
        ),
        reverse=True,
    )
    if not positive_grades:
        return None

    ranking = _deduplicate_ranking(retrieved_chunk_ids)[:k]
    dcg = sum(
        ((2 ** relevance_by_chunk.get(chunk_id, 0)) - 1) / log2(rank + 1)
        for rank, chunk_id in enumerate(ranking, start=1)
    )
    ideal_dcg = sum(
        ((2**relevance) - 1) / log2(rank + 1)
        for rank, relevance in enumerate(positive_grades[:k], start=1)
    )
    return dcg / ideal_dcg


def calculate_ranked_retrieval_metrics(
    qrels: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    cutoffs: Iterable[int] = DEFAULT_RANK_CUTOFFS,
) -> dict[str, float | None]:
    """Calculate Recall, reciprocal rank and nDCG for several cutoffs."""
    chunk_ids = [str(source.get("chunk_id", "")) for source in sources]
    metrics: dict[str, float | None] = {}
    for k in cutoffs:
        _validate_k(k)
        metrics[f"recall@{k}"] = recall_at_k(qrels, chunk_ids, k)
        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(qrels, chunk_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(qrels, chunk_ids, k)
    return metrics


def mean_ranked_metrics(
    metrics_by_question: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Macro-average numeric rank metrics while ignoring None values."""
    values_by_name: dict[str, list[float]] = {}
    for metrics in metrics_by_question:
        for name, value in metrics.items():
            if not (
                name.startswith("recall@")
                or name.startswith("mrr@")
                or name.startswith("ndcg@")
            ):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values_by_name.setdefault(name, []).append(float(value))

    return {
        name: sum(values) / len(values)
        for name, values in sorted(values_by_name.items())
        if values
    }
