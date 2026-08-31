from __future__ import annotations

import unittest

from evaluation.finalize_qrels import build_reviewed_qrels


class FinalizeQrelsTests(unittest.TestCase):
    def test_builds_graded_qrels_and_empty_no_answer(self) -> None:
        questions = [
            {
                "id": "answerable",
                "type": "fact",
                "source_files": ["paper.pdf"],
                "source_pages": [2],
            },
            {
                "id": "missing",
                "type": "no_answer",
                "source_files": [],
                "source_pages": [],
            },
        ]
        index_rows = {
            "direct": {"file_name": "paper.pdf", "page_number": 2},
            "support": {"file_name": "paper.pdf", "page_number": 2},
        }

        qrels = build_reviewed_qrels(
            questions,
            index_rows,
            {"answerable": [["direct", "support"]]},
        )

        self.assertEqual(
            [item["relevance"] for item in qrels["answerable"]], [3, 2]
        )
        self.assertEqual(qrels["missing"], [])

    def test_rejects_source_mismatch(self) -> None:
        questions = [
            {
                "id": "answerable",
                "type": "fact",
                "source_files": ["paper.pdf"],
                "source_pages": [2],
            }
        ]
        with self.assertRaisesRegex(ValueError, "来源不匹配"):
            build_reviewed_qrels(
                questions,
                {"wrong": {"file_name": "other.pdf", "page_number": 2}},
                {"answerable": [["wrong"]]},
            )


if __name__ == "__main__":
    unittest.main()
