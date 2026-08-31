from __future__ import annotations

import unittest

from langchain_core.documents import Document

from evaluation.evaluate_retrieval import build_reranked_sources
from hybrid_retriever import FusedCandidate
from reranker import (
    RerankCandidate,
    RerankedCandidate,
    dense_rerank_candidates,
    hybrid_rerank_candidates,
    rerank_candidates,
)


def document(chunk_id: str, file_name: str = "paper.pdf") -> Document:
    return Document(
        page_content=chunk_id,
        metadata={"chunk_id": chunk_id, "file_name": file_name},
    )


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.queries: list[str] = []
        self.documents: list[Document] = []

    def score(self, query: str, documents: list[Document]) -> list[float]:
        self.queries.append(query)
        self.documents = documents
        return self.scores


class RerankerTests(unittest.TestCase):
    def test_ranks_by_score_and_preserves_diagnostics(self) -> None:
        candidates = dense_rerank_candidates(
            [(document("a"), 0.1), (document("b"), 0.2)]
        )

        result = rerank_candidates(
            "query",
            candidates,
            FakeReranker([0.2, 0.9]),
            top_k=2,
        )

        self.assertEqual(
            [item.candidate.chunk_id for item in result.candidates],
            ["b", "a"],
        )
        self.assertEqual(result.candidates[0].candidate.dense_distance, 0.2)
        self.assertEqual(result.candidates[0].rerank_score, 0.9)

    def test_equal_scores_keep_retrieval_order(self) -> None:
        candidates = [
            RerankCandidate(document("b"), "b", 1),
            RerankCandidate(document("a"), "a", 2),
        ]

        result = rerank_candidates(
            "query",
            candidates,
            FakeReranker([0.5, 0.5]),
            top_k=2,
        )

        self.assertEqual(
            [item.candidate.chunk_id for item in result.candidates],
            ["b", "a"],
        )

    def test_applies_source_cap_after_reranking(self) -> None:
        candidates = [
            RerankCandidate(document("a1", "a.pdf"), "a1", 1),
            RerankCandidate(document("a2", "a.pdf"), "a2", 2),
            RerankCandidate(document("b1", "b.pdf"), "b1", 3),
        ]

        result = rerank_candidates(
            "query",
            candidates,
            FakeReranker([0.9, 0.8, 0.7]),
            top_k=2,
            max_chunks_per_file=1,
        )

        self.assertEqual(
            [item.candidate.chunk_id for item in result.candidates],
            ["a1", "b1"],
        )

    def test_rejects_invalid_score_output(self) -> None:
        candidates = [RerankCandidate(document("a"), "a", 1)]
        with self.assertRaisesRegex(ValueError, "数量不一致"):
            rerank_candidates(
                "query",
                candidates,
                FakeReranker([]),
                top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "非有限"):
            rerank_candidates(
                "query",
                candidates,
                FakeReranker([float("nan")]),
                top_k=1,
            )

    def test_dense_normalization_deduplicates_and_requires_chunk_id(self) -> None:
        repeated = document("same")
        candidates = dense_rerank_candidates(
            [(repeated, 0.1), (repeated, 0.2)]
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].retrieval_rank, 1)
        with self.assertRaisesRegex(ValueError, "缺少 chunk_id"):
            dense_rerank_candidates(
                [(Document(page_content="missing"), 0.1)]
            )

    def test_hybrid_normalization_preserves_branch_scores(self) -> None:
        fused = FusedCandidate(
            document=document("shared"),
            chunk_id="shared",
            rrf_score=0.03,
            dense_rank=2,
            sparse_rank=1,
            dense_distance=0.2,
            bm25_score=4.0,
        )

        candidate = hybrid_rerank_candidates([fused])[0]

        self.assertEqual(candidate.retrieval_rank, 1)
        self.assertEqual(candidate.rrf_score, 0.03)
        self.assertEqual((candidate.dense_rank, candidate.sparse_rank), (2, 1))

    def test_serializes_final_score_and_original_rank(self) -> None:
        candidate = RerankCandidate(
            document("a"),
            "a",
            retrieval_rank=3,
            dense_distance=0.2,
            rrf_score=0.03,
        )

        source = build_reranked_sources(
            [RerankedCandidate(candidate, rerank_score=0.9)]
        )[0]

        self.assertEqual(source["rank"], 1)
        self.assertEqual(source["retrieval_rank"], 3)
        self.assertAlmostEqual(source["similarity"], 0.8)
        self.assertEqual(source["rrf_score"], 0.03)
        self.assertEqual(source["rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
