"""Tests for the runtime Dense / BM25 / Hybrid retrieval boundary."""

import unittest

from langchain_core.documents import Document

from retrieval_pipeline import RetrievalMethod, RetrievalPipeline
from retriever import RetrievalStrategy


def document(chunk_id: str, file_name: str = "paper.pdf") -> Document:
    return Document(
        page_content=chunk_id,
        metadata={
            "chunk_id": chunk_id,
            "file_name": file_name,
            "document_id": file_name,
        },
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


class RetrievalPipelineTests(unittest.TestCase):
    def test_dense_is_default_and_preserves_similarity(self) -> None:
        vector_store = FakeVectorStore([(document("dense"), 0.2)])

        result = RetrievalPipeline(vector_store).search("query", top_k=1)

        self.assertEqual(result.method, RetrievalMethod.DENSE)
        self.assertEqual(result.strategy, RetrievalStrategy.FOCUSED)
        self.assertEqual(vector_store.requested_k, 1)
        self.assertAlmostEqual(result.chunks[0].similarity, 0.8)
        self.assertEqual(result.chunks[0].dense_rank, 1)
        self.assertGreaterEqual(result.total_latency_seconds, 0.0)

    def test_bm25_multi_document_applies_source_cap_after_candidates(self) -> None:
        sparse = FakeSparseRetriever(
            [
                (document("a1", "a.pdf"), 9.0),
                (document("a2", "a.pdf"), 8.0),
                (document("b1", "b.pdf"), 7.0),
                (document("c1", "c.pdf"), 6.0),
            ]
        )
        pipeline = RetrievalPipeline(FakeVectorStore([]), sparse)

        result = pipeline.search(
            "query",
            method=RetrievalMethod.BM25,
            strategy=RetrievalStrategy.MULTI_DOCUMENT,
            top_k=3,
            max_chunks_per_file=1,
        )

        self.assertEqual(sparse.requested_k, 12)
        self.assertEqual(
            [chunk.chunk_id for chunk in result.chunks],
            ["a1", "b1", "c1"],
        )
        self.assertEqual([chunk.sparse_rank for chunk in result.chunks], [1, 3, 4])
        self.assertEqual(result.chunks[0].bm25_score, 9.0)

    def test_hybrid_uses_frozen_parameters_and_exposes_branch_scores(self) -> None:
        shared = document("shared")
        vector_store = FakeVectorStore([(shared, 0.1)])
        sparse = FakeSparseRetriever([(shared, 5.0)])

        result = RetrievalPipeline(vector_store, sparse).search(
            "query",
            method="hybrid",
            top_k=1,
        )

        self.assertEqual(vector_store.requested_k, 20)
        self.assertEqual(sparse.requested_k, 20)
        self.assertEqual((result.candidate_k, result.fusion_k, result.rrf_k), (20, 40, 60))
        self.assertAlmostEqual(result.chunks[0].similarity, 0.9)
        self.assertEqual(result.chunks[0].bm25_score, 5.0)
        self.assertGreater(result.chunks[0].rrf_score, 0.0)

    def test_sparse_methods_require_a_loaded_bm25_index(self) -> None:
        pipeline = RetrievalPipeline(FakeVectorStore([]))

        for method in (RetrievalMethod.BM25, RetrievalMethod.HYBRID):
            with self.subTest(method=method), self.assertRaisesRegex(
                RuntimeError,
                "BM25",
            ):
                pipeline.search("query", method=method)

    def test_invalid_method_lists_allowed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense, bm25, hybrid"):
            RetrievalPipeline(FakeVectorStore([])).search(
                "query",
                method="unsupported",
            )

    def test_invalid_optional_limits_fail_explicitly(self) -> None:
        pipeline = RetrievalPipeline(
            FakeVectorStore([]),
            FakeSparseRetriever([]),
        )

        with self.assertRaisesRegex(ValueError, "top_k"):
            pipeline.search("query", method="bm25", candidate_k=0)
        with self.assertRaisesRegex(ValueError, "max_chunks_per_file"):
            pipeline.search(
                "query",
                method="bm25",
                max_chunks_per_file=0,
            )


if __name__ == "__main__":
    unittest.main()
