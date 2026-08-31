"""Dense + BM25 candidate fusion with deterministic reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from langchain_core.documents import Document

from config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_HYBRID_FUSION_K,
    DEFAULT_RRF_K,
)
from retriever import retrieve_with_scores, validate_query, validate_top_k
from sparse_retriever import SparseRetriever


@dataclass(frozen=True)
class FusedCandidate:
    """One unique Chunk with branch ranks and an RRF score."""

    document: Document
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_distance: float | None
    bm25_score: float | None


@dataclass(frozen=True)
class HybridSearchResult:
    """Fused results plus per-stage retrieval latency."""

    candidates: list[FusedCandidate]
    dense_latency_seconds: float
    sparse_latency_seconds: float
    fusion_latency_seconds: float
    total_latency_seconds: float


def _normalize_branch(
    results: list[tuple[Document, float]],
) -> dict[str, tuple[int, Document, float]]:
    """Keep the first occurrence of every valid persistent chunk_id."""
    normalized: dict[str, tuple[int, Document, float]] = {}
    for rank, (document, score) in enumerate(results, start=1):
        chunk_id = str(document.metadata.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Hybrid 候选缺少 chunk_id。")
        normalized.setdefault(chunk_id, (rank, document, float(score)))
    return normalized


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    sparse_results: list[tuple[Document, float]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    limit: int = DEFAULT_HYBRID_FUSION_K,
) -> list[FusedCandidate]:
    """Fuse branch rankings without normalizing incompatible raw scores."""
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0。")
    validate_top_k(limit)

    dense = _normalize_branch(dense_results)
    sparse = _normalize_branch(sparse_results)
    fused: list[FusedCandidate] = []

    for chunk_id in dense.keys() | sparse.keys():
        dense_item = dense.get(chunk_id)
        sparse_item = sparse.get(chunk_id)
        dense_rank = dense_item[0] if dense_item else None
        sparse_rank = sparse_item[0] if sparse_item else None
        document = dense_item[1] if dense_item else sparse_item[1]
        rrf_score = sum(
            1.0 / (rrf_k + rank)
            for rank in (dense_rank, sparse_rank)
            if rank is not None
        )
        fused.append(
            FusedCandidate(
                document=document,
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                dense_distance=dense_item[2] if dense_item else None,
                bm25_score=sparse_item[2] if sparse_item else None,
            )
        )

    missing_rank = len(dense) + len(sparse) + 1
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            min(
                rank
                for rank in (item.dense_rank, item.sparse_rank)
                if rank is not None
            ),
            item.dense_rank if item.dense_rank is not None else missing_rank,
            item.chunk_id,
        )
    )
    return fused[:limit]


def select_fused_candidates(
    candidates: list[FusedCandidate],
    *,
    top_k: int,
    max_chunks_per_file: int | None = None,
) -> list[FusedCandidate]:
    """Apply a source cap only after fusion, then take the final Top-k."""
    validate_top_k(top_k)
    if max_chunks_per_file is None:
        return candidates[:top_k]
    if max_chunks_per_file <= 0:
        raise ValueError("max_chunks_per_file 必须大于 0。")

    selected: list[FusedCandidate] = []
    file_counts: dict[str, int] = {}
    for candidate in candidates:
        file_name = str(
            candidate.document.metadata.get(
                "file_name",
                candidate.document.metadata.get("document_id", "unknown"),
            )
        )
        if file_counts.get(file_name, 0) >= max_chunks_per_file:
            continue
        selected.append(candidate)
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
        if len(selected) == top_k:
            break
    return selected


class HybridRetriever:
    """Retrieve unmodified branch candidates and fuse them using RRF."""

    def __init__(self, vector_store: Any, sparse_retriever: SparseRetriever):
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
        fusion_k: int = DEFAULT_HYBRID_FUSION_K,
        max_chunks_per_file: int | None = None,
    ) -> HybridSearchResult:
        cleaned_query = validate_query(query)
        validate_top_k(top_k)
        validate_top_k(candidate_k)
        validate_top_k(fusion_k)
        if candidate_k < top_k:
            raise ValueError("candidate_k 不能小于 top_k。")
        if fusion_k < top_k:
            raise ValueError("fusion_k 不能小于 top_k。")

        total_start = time.perf_counter()
        dense_start = time.perf_counter()
        dense_results = retrieve_with_scores(
            self.vector_store,
            cleaned_query,
            top_k=candidate_k,
        )
        dense_latency = time.perf_counter() - dense_start

        sparse_start = time.perf_counter()
        sparse_results = self.sparse_retriever.search(
            cleaned_query,
            top_k=candidate_k,
        )
        sparse_latency = time.perf_counter() - sparse_start

        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            rrf_k=rrf_k,
            limit=fusion_k,
        )
        selected = select_fused_candidates(
            fused,
            top_k=top_k,
            max_chunks_per_file=max_chunks_per_file,
        )
        fusion_latency = time.perf_counter() - fusion_start

        return HybridSearchResult(
            candidates=selected,
            dense_latency_seconds=dense_latency,
            sparse_latency_seconds=sparse_latency,
            fusion_latency_seconds=fusion_latency,
            total_latency_seconds=time.perf_counter() - total_start,
        )
