from __future__ import annotations

import unittest

from langchain_core.documents import Document

from evaluation.evaluate_retrieval import build_hybrid_sources
from hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    select_fused_candidates,
)


def document(chunk_id: str, file_name: str = "paper.pdf") -> Document:
    return Document(
        page_content=chunk_id,
        metadata={"chunk_id": chunk_id, "file_name": file_name},
    )


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def similarity_search_with_score(self, query, k):
        self.requested_k = k
        return self.results[:k]


class FakeSparseRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def search(self, query, top_k):
        self.requested_k = top_k
        return self.results[:top_k]


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_matches_hand_calculated_rrf_order_and_score(self) -> None:
        dense = [
            (document("a"), 0.1),
            (document("b"), 0.2),
            (document("c"), 0.3),
        ]
        sparse = [
            (document("b"), 9.0),
            (document("c"), 8.0),
            (document("d"), 7.0),
        ]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, limit=4)

        self.assertEqual([item.chunk_id for item in fused], ["b", "c", "a", "d"])
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 62 + 1 / 61)
        self.assertEqual((fused[0].dense_rank, fused[0].sparse_rank), (2, 1))

    def test_deduplicates_each_branch_by_first_rank(self) -> None:
        repeated = document("same")
        fused = reciprocal_rank_fusion(
            [(repeated, 0.1), (repeated, 0.2)],
            [(repeated, 5.0)],
            rrf_k=60,
            limit=5,
        )

        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].rrf_score, 2 / 61)

    def test_dense_rank_breaks_equal_single_branch_scores(self) -> None:
        fused = reciprocal_rank_fusion(
            [(document("dense"), 0.1)],
            [(document("sparse"), 5.0)],
            rrf_k=60,
            limit=2,
        )

        self.assertEqual([item.chunk_id for item in fused], ["dense", "sparse"])

    def test_rejects_missing_chunk_id_and_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少 chunk_id"):
            reciprocal_rank_fusion(
                [(Document(page_content="missing"), 0.1)],
                [],
            )
        with self.assertRaisesRegex(ValueError, "rrf_k"):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_source_cap_is_applied_after_fusion(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                (document("a1", "a.pdf"), 0.1),
                (document("a2", "a.pdf"), 0.2),
                (document("b1", "b.pdf"), 0.3),
            ],
            [],
            limit=3,
        )

        selected = select_fused_candidates(
            fused,
            top_k=2,
            max_chunks_per_file=1,
        )

        self.assertEqual([item.chunk_id for item in selected], ["a1", "b1"])

    def test_serializes_rrf_and_branch_diagnostics(self) -> None:
        fused = reciprocal_rank_fusion(
            [(document("shared"), 0.2)],
            [(document("shared"), 4.0)],
            limit=1,
        )

        source = build_hybrid_sources(fused)[0]

        self.assertAlmostEqual(source["similarity"], 0.8)
        self.assertEqual(source["bm25_score"], 4.0)
        self.assertEqual((source["dense_rank"], source["sparse_rank"]), (1, 1))
        self.assertGreater(source["rrf_score"], 0.0)


class HybridRetrieverTests(unittest.TestCase):
    def test_requests_full_candidates_and_handles_empty_sparse_branch(self) -> None:
        vector_store = FakeVectorStore(
            [(document("dense"), 0.1)]
        )
        sparse = FakeSparseRetriever([])
        retriever = HybridRetriever(vector_store, sparse)

        result = retriever.search(
            "纯中文查询",
            top_k=1,
            candidate_k=3,
            fusion_k=3,
        )

        self.assertEqual(vector_store.requested_k, 3)
        self.assertEqual(sparse.requested_k, 3)
        self.assertEqual(result.candidates[0].chunk_id, "dense")
        self.assertIsNone(result.candidates[0].sparse_rank)


if __name__ == "__main__":
    unittest.main()
