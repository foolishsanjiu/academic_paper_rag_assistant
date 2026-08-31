"""Cross-encoder scoring and deterministic candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol, Sequence

from langchain_core.documents import Document

from config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
)
from hybrid_retriever import FusedCandidate
from retriever import validate_query, validate_top_k


class Reranker(Protocol):
    """Minimal scoring contract used by the retrieval pipeline."""

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[float]: ...


@dataclass(frozen=True)
class RerankCandidate:
    """One retrieval candidate with diagnostics preserved for output."""

    document: Document
    chunk_id: str
    retrieval_rank: int
    dense_distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class RerankedCandidate:
    """A candidate plus its cross-encoder relevance score."""

    candidate: RerankCandidate
    rerank_score: float


@dataclass(frozen=True)
class RerankResult:
    """Final reranked candidates and model-scoring latency."""

    candidates: list[RerankedCandidate]
    latency_seconds: float


def _chunk_id(document: Document) -> str:
    chunk_id = str(document.metadata.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError("Reranker 候选缺少 chunk_id。")
    return chunk_id


def dense_rerank_candidates(
    results: list[tuple[Document, float]],
) -> list[RerankCandidate]:
    """Normalize Dense results and keep the first occurrence of each Chunk."""
    candidates: list[RerankCandidate] = []
    seen: set[str] = set()
    for rank, (document, distance) in enumerate(results, start=1):
        chunk_id = _chunk_id(document)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        candidates.append(
            RerankCandidate(
                document=document,
                chunk_id=chunk_id,
                retrieval_rank=rank,
                dense_distance=float(distance),
                dense_rank=rank,
            )
        )
    return candidates


def hybrid_rerank_candidates(
    candidates: list[FusedCandidate],
) -> list[RerankCandidate]:
    """Normalize fused candidates while preserving branch diagnostics."""
    return [
        RerankCandidate(
            document=item.document,
            chunk_id=item.chunk_id,
            retrieval_rank=rank,
            dense_distance=item.dense_distance,
            bm25_score=item.bm25_score,
            rrf_score=item.rrf_score,
            dense_rank=item.dense_rank,
            sparse_rank=item.sparse_rank,
        )
        for rank, item in enumerate(candidates, start=1)
    ]


def rerank_candidates(
    query: str,
    candidates: list[RerankCandidate],
    reranker: Reranker,
    *,
    top_k: int,
    max_chunks_per_file: int | None = None,
) -> RerankResult:
    """Score, stably sort, apply the source cap, and take final Top-k."""
    cleaned_query = validate_query(query)
    validate_top_k(top_k)
    if max_chunks_per_file is not None and max_chunks_per_file <= 0:
        raise ValueError("max_chunks_per_file 必须大于 0。")
    if not candidates:
        return RerankResult(candidates=[], latency_seconds=0.0)

    start = time.perf_counter()
    scores = reranker.score(
        cleaned_query,
        [candidate.document for candidate in candidates],
    )
    latency = time.perf_counter() - start
    if len(scores) != len(candidates):
        raise ValueError(
            "Reranker 返回分数数量与候选数量不一致："
            f"scores={len(scores)}, candidates={len(candidates)}"
        )
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Reranker 返回了非有限分数。")

    ranked = [
        RerankedCandidate(candidate=candidate, rerank_score=float(score))
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(
        key=lambda item: (
            -item.rerank_score,
            item.candidate.retrieval_rank,
            item.candidate.chunk_id,
        )
    )

    selected: list[RerankedCandidate] = []
    file_counts: dict[str, int] = {}
    for item in ranked:
        metadata = item.candidate.document.metadata
        file_name = str(
            metadata.get(
                "file_name",
                metadata.get("document_id", "unknown"),
            )
        )
        if (
            max_chunks_per_file is not None
            and file_counts.get(file_name, 0) >= max_chunks_per_file
        ):
            continue
        selected.append(item)
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
        if len(selected) == top_k:
            break

    return RerankResult(candidates=selected, latency_seconds=latency)


class TransformersCrossEncoderReranker:
    """Run a local sequence-classification reranker in deterministic batches."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        device: str = "cpu",
        normalize: bool = True,
        local_files_only: bool = True,
    ):
        validate_top_k(batch_size)
        validate_top_k(max_length)

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device)
        self.normalize = normalize
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[float]:
        cleaned_query = validate_query(query)
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            pairs = [
                [cleaned_query, document.page_content]
                for document in batch
            ]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {
                name: tensor.to(self.device)
                for name, tensor in inputs.items()
            }
            with self._torch.inference_mode():
                logits = self._model(
                    **inputs,
                    return_dict=True,
                ).logits.view(-1).float()
                if self.normalize:
                    logits = self._torch.sigmoid(logits)
            scores.extend(logits.cpu().tolist())
        return scores
