"""Unit tests for similarity retrieval and source diversification."""

import unittest

from langchain_core.documents import Document

from retriever import (
    RetrievalStrategy,
    get_retrieval_options,
    retrieve_with_scores,
)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.requested_k = None

    def similarity_search_with_score(self, query, k):
        self.requested_k = k
        return self.results[:k]


def result(file_name: str, distance: float):
    return (
        Document(
            page_content=file_name,
            metadata={"file_name": file_name},
        ),
        distance,
    )


class DiverseRetrievalTests(unittest.TestCase):
    def test_plain_retrieval_keeps_similarity_order(self) -> None:
        store = FakeVectorStore(
            [result("a.pdf", 0.1), result("a.pdf", 0.2)]
        )

        retrieved = retrieve_with_scores(store, "query", top_k=1)

        self.assertEqual(store.requested_k, 1)
        self.assertEqual(retrieved[0][0].metadata["file_name"], "a.pdf")

    def test_retrieval_emits_start_and_completion_logs(self) -> None:
        store = FakeVectorStore([result("a.pdf", 0.1)])

        with self.assertLogs("retriever", level="INFO") as captured:
            retrieve_with_scores(store, "query", top_k=1)

        output = "\n".join(captured.output)
        self.assertIn("Retrieval started", output)
        self.assertIn("Retrieval completed", output)

    def test_diversification_caps_chunks_from_one_file(self) -> None:
        store = FakeVectorStore(
            [
                result("a.pdf", 0.1),
                result("a.pdf", 0.2),
                result("b.pdf", 0.3),
                result("c.pdf", 0.4),
            ]
        )

        retrieved = retrieve_with_scores(
            store,
            "query",
            top_k=3,
            candidate_k=4,
            max_chunks_per_file=1,
        )

        self.assertEqual(
            [item[0].metadata["file_name"] for item in retrieved],
            ["a.pdf", "b.pdf", "c.pdf"],
        )

    def test_invalid_diversification_parameters_fail(self) -> None:
        store = FakeVectorStore([])

        with self.assertRaises(ValueError):
            retrieve_with_scores(
                store,
                "query",
                top_k=5,
                candidate_k=4,
                max_chunks_per_file=2,
            )

        with self.assertRaises(ValueError):
            retrieve_with_scores(
                store,
                "query",
                top_k=5,
                max_chunks_per_file=0,
            )


class RetrievalStrategyTests(unittest.TestCase):
    def test_focused_strategy_preserves_plain_retrieval(self) -> None:
        options = get_retrieval_options(RetrievalStrategy.FOCUSED)

        self.assertEqual(options.top_k, 5)
        self.assertIsNone(options.candidate_k)
        self.assertIsNone(options.max_chunks_per_file)

    def test_multi_document_strategy_uses_experiment_winner(self) -> None:
        options = get_retrieval_options(
            RetrievalStrategy.MULTI_DOCUMENT
        )

        self.assertEqual(options.top_k, 8)
        self.assertEqual(options.candidate_k, 32)
        self.assertEqual(options.max_chunks_per_file, 3)

    def test_invalid_strategy_fails_with_allowed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "focused"):
            get_retrieval_options("unsupported")

    def test_top_k_rejects_boolean_and_non_integer_values(self) -> None:
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "整数",
            ):
                get_retrieval_options(top_k=value)


if __name__ == "__main__":
    unittest.main()
