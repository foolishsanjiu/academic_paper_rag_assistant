"""Unified runtime retrieval pipeline for Dense, BM25, and Hybrid search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

from langchain_core.documents import Document

from config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_HYBRID_FUSION_K,
    DEFAULT_RRF_K,
)
from hybrid_retriever import HybridRetriever
from retriever import (
    RetrievalStrategy,
    cosine_distance_to_similarity,
    get_retrieval_options,
    retrieve_with_scores,
    validate_top_k,
)
from sparse_retriever import SparseRetriever


class RetrievalMethod(str, Enum):
    """Retrieval algorithms exposed by the application."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved Chunk with method-specific diagnostic scores."""

    document: Document
    chunk_id: str
    similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Selected Chunks plus the parameters and latency used to obtain them."""

    method: RetrievalMethod
    strategy: RetrievalStrategy
    chunks: list[RetrievedChunk]
    top_k: int
    candidate_k: int
    fusion_k: int | None
    rrf_k: int | None
    dense_latency_seconds: float
    sparse_latency_seconds: float
    fusion_latency_seconds: float
    total_latency_seconds: float


def _chunk_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError("检索候选缺少 chunk_id。")
    return chunk_id


def _select_sparse_results(
    results: list[tuple[Document, float]],
    *,
    top_k: int,
    max_chunks_per_file: int | None,
) -> list[tuple[int, Document, float]]:
    """Apply the strategy source cap while retaining original BM25 ranks."""
    if max_chunks_per_file is None:
        return [
            (rank, document, score)
            for rank, (document, score) in enumerate(results[:top_k], start=1)
        ]

    selected: list[tuple[int, Document, float]] = []
    file_counts: dict[str, int] = {}
    for rank, (document, score) in enumerate(results, start=1):
        file_name = str(
            document.metadata.get(
                "file_name",
                document.metadata.get("document_id", "unknown"),
            )
        )
        if file_counts.get(file_name, 0) >= max_chunks_per_file:
            continue
        selected.append((rank, document, score))
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
        if len(selected) == top_k:
            break
    return selected


class RetrievalPipeline:
    """Run one retrieval method under one independent retrieval strategy."""

    def __init__(
        self,
        vector_store: Any,
        sparse_retriever: SparseRetriever | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever

    def search(
        self,
        query: str,
        *,
        method: RetrievalMethod | str = RetrievalMethod.DENSE,
        strategy: RetrievalStrategy | str = RetrievalStrategy.FOCUSED,
        top_k: int | None = None,
        candidate_k: int | None = None,
        max_chunks_per_file: int | None = None,
    ) -> RetrievalResult:
        try:
            resolved_method = RetrievalMethod(method)
        except ValueError as error:
            allowed = ", ".join(item.value for item in RetrievalMethod)
            raise ValueError(
                f"不支持的检索方法：{method}。可选值：{allowed}"
            ) from error

        options = get_retrieval_options(strategy=strategy, top_k=top_k)
        if candidate_k is not None:
            validate_top_k(candidate_k)
        if max_chunks_per_file is not None and max_chunks_per_file <= 0:
            raise ValueError("max_chunks_per_file 必须大于 0。")
        resolved_cap = (
            max_chunks_per_file
            if max_chunks_per_file is not None
            else options.max_chunks_per_file
        )

        if resolved_method is RetrievalMethod.DENSE:
            return self._search_dense(
                query,
                options.strategy,
                options.top_k,
                candidate_k if candidate_k is not None else options.candidate_k,
                resolved_cap,
            )

        if self.sparse_retriever is None:
            raise RuntimeError(
                f"{resolved_method.value} 检索需要先加载 BM25 索引。"
            )

        if resolved_method is RetrievalMethod.BM25:
            return self._search_bm25(
                query,
                options.strategy,
                options.top_k,
                candidate_k if candidate_k is not None else options.candidate_k,
                resolved_cap,
            )

        return self._search_hybrid(
            query,
            options.strategy,
            options.top_k,
            (
                candidate_k
                if candidate_k is not None
                else DEFAULT_HYBRID_CANDIDATE_K
            ),
            resolved_cap,
        )

    def _search_dense(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int,
        candidate_k: int | None,
        max_chunks_per_file: int | None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        results = retrieve_with_scores(
            self.vector_store,
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            max_chunks_per_file=max_chunks_per_file,
        )
        dense_latency = time.perf_counter() - started
        chunks = [
            RetrievedChunk(
                document=document,
                chunk_id=_chunk_id(document),
                similarity=cosine_distance_to_similarity(distance),
                dense_rank=rank,
            )
            for rank, (document, distance) in enumerate(results, start=1)
        ]
        return RetrievalResult(
            method=RetrievalMethod.DENSE,
            strategy=strategy,
            chunks=chunks,
            top_k=top_k,
            candidate_k=candidate_k or top_k,
            fusion_k=None,
            rrf_k=None,
            dense_latency_seconds=dense_latency,
            sparse_latency_seconds=0.0,
            fusion_latency_seconds=0.0,
            total_latency_seconds=time.perf_counter() - started,
        )

    def _search_bm25(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int,
        candidate_k: int | None,
        max_chunks_per_file: int | None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        search_k = candidate_k if candidate_k is not None else top_k
        sparse_started = time.perf_counter()
        results = self.sparse_retriever.search(query, top_k=search_k)
        sparse_latency = time.perf_counter() - sparse_started
        selected = _select_sparse_results(
            results,
            top_k=top_k,
            max_chunks_per_file=max_chunks_per_file,
        )
        chunks = [
            RetrievedChunk(
                document=document,
                chunk_id=_chunk_id(document),
                bm25_score=score,
                sparse_rank=rank,
            )
            for rank, document, score in selected
        ]
        return RetrievalResult(
            method=RetrievalMethod.BM25,
            strategy=strategy,
            chunks=chunks,
            top_k=top_k,
            candidate_k=search_k,
            fusion_k=None,
            rrf_k=None,
            dense_latency_seconds=0.0,
            sparse_latency_seconds=sparse_latency,
            fusion_latency_seconds=0.0,
            total_latency_seconds=time.perf_counter() - started,
        )

    def _search_hybrid(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int,
        candidate_k: int,
        max_chunks_per_file: int | None,
    ) -> RetrievalResult:
        result = HybridRetriever(
            self.vector_store,
            self.sparse_retriever,
        ).search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=DEFAULT_RRF_K,
            fusion_k=DEFAULT_HYBRID_FUSION_K,
            max_chunks_per_file=max_chunks_per_file,
        )
        chunks = [
            RetrievedChunk(
                document=candidate.document,
                chunk_id=candidate.chunk_id,
                similarity=(
                    cosine_distance_to_similarity(candidate.dense_distance)
                    if candidate.dense_distance is not None
                    else None
                ),
                bm25_score=candidate.bm25_score,
                rrf_score=candidate.rrf_score,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
            )
            for candidate in result.candidates
        ]
        return RetrievalResult(
            method=RetrievalMethod.HYBRID,
            strategy=strategy,
            chunks=chunks,
            top_k=top_k,
            candidate_k=candidate_k,
            fusion_k=DEFAULT_HYBRID_FUSION_K,
            rrf_k=DEFAULT_RRF_K,
            dense_latency_seconds=result.dense_latency_seconds,
            sparse_latency_seconds=result.sparse_latency_seconds,
            fusion_latency_seconds=result.fusion_latency_seconds,
            total_latency_seconds=result.total_latency_seconds,
        )
