import unittest

from evaluation.build_annotation_pool import (
    build_annotation_pool,
    iter_result_sets,
    normalize_index_rows,
)


QUESTIONS = [
    {
        "id": "fact_001",
        "question": "What is the method?",
        "expected_answer": "Expected answer.",
        "expected_key_points": ["key point"],
        "source_files": ["paper.pdf"],
        "source_pages": [3],
    }
]


class AnnotationPoolTests(unittest.TestCase):
    def test_iterates_retrieval_runs_with_distinct_origins(self) -> None:
        payload = {
            "runs": [
                {"top_k": 5, "results": []},
                {"top_k": 8, "results": []},
            ]
        }
        origins = [origin for origin, _ in iter_result_sets(payload, "run.json")]
        self.assertEqual(
            origins,
            ["run.json:top_k=5", "run.json:top_k=8"],
        )

    def test_merges_retrieval_and_expected_page_chunks(self) -> None:
        result_sets = [
            (
                "dense",
                [
                    {
                        "id": "fact_001",
                        "retrieved_sources": [
                            {
                                "chunk_id": "retrieved",
                                "file_name": "other.pdf",
                                "page_number": 1,
                                "text": "Retrieved text",
                            },
                            {
                                "chunk_id": "expected",
                                "file_name": "paper.pdf",
                                "page_number": 3,
                                "text": "Expected text",
                            },
                        ],
                    }
                ],
            )
        ]
        index_rows = {
            "expected": {
                "chunk_id": "expected",
                "file_name": "paper.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "text": "Expected text",
            },
            "source-page-only": {
                "chunk_id": "source-page-only",
                "file_name": "paper.pdf",
                "page_number": 3,
                "chunk_index": 1,
                "text": "Additional source page text",
            },
        }

        pool = build_annotation_pool(
            QUESTIONS,
            result_sets,
            candidate_depth=20,
            index_rows=index_rows,
        )

        question_pool = pool["questions"]["fact_001"]
        chunk_ids = {
            item["chunk_id"] for item in question_pool["candidates"]
        }
        self.assertEqual(
            chunk_ids,
            {"retrieved", "expected", "source-page-only"},
        )
        self.assertEqual(
            question_pool["pool_provenance"]["expected"],
            ["dense", "expected_source_page"],
        )

    def test_candidate_depth_limits_each_result_set(self) -> None:
        result_sets = [
            (
                "dense",
                [
                    {
                        "id": "fact_001",
                        "retrieved_sources": [
                            {"chunk_id": "first"},
                            {"chunk_id": "second"},
                        ],
                    }
                ],
            )
        ]
        pool = build_annotation_pool(
            QUESTIONS,
            result_sets,
            candidate_depth=1,
            include_source_pages=False,
        )
        candidates = pool["questions"]["fact_001"]["candidates"]
        self.assertEqual([item["chunk_id"] for item in candidates], ["first"])

    def test_rejects_misaligned_index_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "数量不一致"):
            normalize_index_rows(
                {
                    "ids": ["chunk-1"],
                    "documents": [],
                    "metadatas": [{}],
                }
            )


if __name__ == "__main__":
    unittest.main()
