from __future__ import annotations

import unittest

from langchain_core.documents import Document

from evaluation.propose_qrels import (
    build_search_text,
    build_source_search_text,
    propose_qrels,
)


class ProposeQrelsTests(unittest.TestCase):
    def test_builds_one_judgment_per_source_pair(self) -> None:
        questions = [
            {
                "id": "answerable",
                "type": "comparison",
                "expected_answer": "Expected answer",
                "expected_key_points": ["Point A"],
                "source_files": ["a.pdf", "b.pdf"],
                "source_pages": [1, 2],
            },
            {
                "id": "missing",
                "type": "no_answer",
                "expected_answer": "Not available",
                "expected_key_points": [],
                "source_files": [],
                "source_pages": [],
            },
        ]

        def search_page(query: str, file_name: str, page_number: int):
            return [
                (
                    Document(
                        page_content=f"Evidence from {file_name}",
                        metadata={
                            "chunk_id": f"{file_name}-{page_number}",
                            "file_name": file_name,
                            "page_number": page_number,
                        },
                    ),
                    0.8,
                )
            ]

        qrels, review = propose_qrels(questions, search_page)

        self.assertEqual(len(qrels["answerable"]), 2)
        self.assertEqual(qrels["missing"], [])
        self.assertEqual(
            review["questions"]["answerable"]["selections"][0]["score"],
            0.8,
        )

    def test_search_text_includes_answer_and_key_points(self) -> None:
        text = build_search_text(
            {
                "expected_answer": "Answer",
                "expected_key_points": ["First", "Second"],
            }
        )
        self.assertEqual(text, "Answer\nFirst\nSecond")

    def test_source_search_text_uses_matching_answer_clause(self) -> None:
        question = {
            "expected_answer": "First method does A. Second method does B.",
            "expected_key_points": ["First point", "Second point"],
            "source_files": ["a.pdf", "b.pdf"],
        }
        self.assertEqual(
            build_source_search_text(question, 1),
            "Second method does B.\nSecond point",
        )

    def test_fails_when_annotated_page_has_no_chunk(self) -> None:
        questions = [
            {
                "id": "answerable",
                "type": "fact",
                "expected_answer": "Expected answer",
                "expected_key_points": [],
                "source_files": ["a.pdf"],
                "source_pages": [1],
            }
        ]

        with self.assertRaisesRegex(ValueError, "没有可用 Chunk"):
            propose_qrels(questions, lambda query, file_name, page: [])


if __name__ == "__main__":
    unittest.main()
