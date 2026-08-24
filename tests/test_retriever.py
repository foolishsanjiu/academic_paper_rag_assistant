"""Unit tests for similarity retrieval and source diversification."""

import unittest

from langchain_core.documents import Document

from retriever import retrieve_with_scores


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


if __name__ == "__main__":
    unittest.main()
