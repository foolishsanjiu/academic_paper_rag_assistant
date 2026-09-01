"""Integration tests for RAGChain using the unified retrieval pipeline."""

import unittest

from langchain_core.documents import Document

from rag_chain import NO_ANSWER_MESSAGE, RAGChain
from retrieval_pipeline import RetrievalMethod


def document(chunk_id: str) -> Document:
    return Document(
        page_content="grounded evidence",
        metadata={
            "chunk_id": chunk_id,
            "file_name": "paper.pdf",
            "document_id": "paper",
            "document_type": "paper",
            "page_number": 3,
        },
    )


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, prompt, temperature=None):
        self.calls.append((prompt, temperature))
        return "answer [1]"


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, query, k):
        return self.results[:k]


class FakeSparseRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k):
        return self.results[:top_k]


class RAGChainRetrievalTests(unittest.TestCase):
    def test_existing_constructor_remains_dense_by_default(self) -> None:
        llm = FakeLLM()
        chain = RAGChain(llm, FakeVectorStore([(document("dense"), 0.2)]), top_k=1)

        result = chain.ask("question", temperature=0.0)

        self.assertEqual(result.answer, "answer [1]")
        self.assertEqual(result.retrieval_method, "dense")
        self.assertAlmostEqual(result.sources[0].similarity, 0.8)
        self.assertEqual(len(llm.calls), 1)

    def test_hybrid_response_contains_retrieval_diagnostics(self) -> None:
        shared = document("shared")
        chain = RAGChain(
            FakeLLM(),
            FakeVectorStore([(shared, 0.1)]),
            top_k=1,
            retrieval_method=RetrievalMethod.HYBRID,
            sparse_retriever=FakeSparseRetriever([(shared, 4.0)]),
        )

        result = chain.ask("question")

        self.assertEqual(result.retrieval_method, "hybrid")
        self.assertEqual(result.sources[0].bm25_score, 4.0)
        self.assertGreater(result.sources[0].rrf_score, 0.0)
        self.assertEqual(result.retrieval_diagnostics["candidate_k"], 20)
        self.assertIn("total_latency_seconds", result.retrieval_diagnostics)

    def test_empty_retrieval_refuses_without_generation(self) -> None:
        llm = FakeLLM()
        result = RAGChain(llm, FakeVectorStore([]), top_k=1).ask("question")

        self.assertEqual(result.answer, NO_ANSWER_MESSAGE)
        self.assertEqual(result.sources, [])
        self.assertEqual(llm.calls, [])
        self.assertIsNotNone(result.retrieval_diagnostics)


if __name__ == "__main__":
    unittest.main()
