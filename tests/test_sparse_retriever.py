from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.documents import Document

from evaluation.evaluate_retrieval import (
    build_sparse_sources,
    run_retrieval_experiment,
)
from sparse_retriever import (
    SparseRetriever,
    load_sparse_documents,
    tokenize_sparse,
)


def document(chunk_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "file_name": f"{chunk_id}.pdf"},
    )


class FakeVectorStore:
    def get(self, include):
        self.include = include
        return {
            "ids": ["chunk-b", "chunk-a"],
            "documents": ["beta", "alpha"],
            "metadatas": [
                {"chunk_id": "chunk-b"},
                {"chunk_id": "chunk-a"},
            ],
        }


class SparseTokenizerTests(unittest.TestCase):
    def test_normalizes_domain_terms_and_ignores_chinese(self) -> None:
        self.assertEqual(
            tokenize_sparse("ＳＡＲ DDPM G0 5.77 frequency-domain 中文"),
            ["sar", "ddpm", "g0", "5.77", "frequency-domain"],
        )


class SparseRetrieverTests(unittest.TestCase):
    def test_ranks_matching_document_first(self) -> None:
        retriever = SparseRetriever(
            [
                document("c", "ship detection"),
                document("a", "oil spill segmentation"),
                document("b", "target recognition"),
            ]
        )

        results = retriever.search("oil spill", top_k=2)

        self.assertEqual(results[0][0].metadata["chunk_id"], "a")
        self.assertGreater(results[0][1], 0.0)

    def test_ties_are_broken_by_chunk_id(self) -> None:
        retriever = SparseRetriever(
            [
                document("b", "needle"),
                document("a", "needle"),
                document("c", "gamma"),
                document("d", "delta"),
                document("e", "epsilon"),
            ]
        )

        results = retriever.search("needle", top_k=2)

        self.assertEqual(
            [item[0].metadata["chunk_id"] for item in results],
            ["a", "b"],
        )

    def test_empty_query_fails_but_unsupported_query_returns_empty(self) -> None:
        retriever = SparseRetriever([document("a", "alpha")])

        with self.assertRaisesRegex(ValueError, "不能为空"):
            retriever.search("  ", top_k=1)
        self.assertEqual(retriever.search("纯中文问题", top_k=1), [])
        self.assertEqual(retriever.search("unknown", top_k=1), [])

    def test_rejects_missing_or_duplicate_chunk_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少 chunk_id"):
            SparseRetriever([Document(page_content="alpha")])
        with self.assertRaisesRegex(ValueError, "重复 chunk_id"):
            SparseRetriever([document("a", "alpha"), document("a", "beta")])

    def test_loads_chroma_rows_with_persistent_ids(self) -> None:
        store = FakeVectorStore()

        rows = load_sparse_documents(store)

        self.assertEqual(store.include, ["documents", "metadatas"])
        self.assertEqual(
            [row.metadata["chunk_id"] for row in rows],
            ["chunk-b", "chunk-a"],
        )

    def test_serializes_bm25_score_without_fake_cosine_similarity(self) -> None:
        sources = build_sparse_sources(
            [(document("a", "evidence"), 2.5)]
        )

        self.assertIsNone(sources[0]["similarity"])
        self.assertEqual(sources[0]["bm25_score"], 2.5)
        self.assertEqual(sources[0]["chunk_id"], "a")

    def test_evaluation_rejects_unknown_method_before_loading_resources(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的检索方法"):
            run_retrieval_experiment(
                questions_path=Path("missing.json"),
                output_path=Path("unused.json"),
                top_k_values=[5],
                method="unknown",
            )


if __name__ == "__main__":
    unittest.main()
