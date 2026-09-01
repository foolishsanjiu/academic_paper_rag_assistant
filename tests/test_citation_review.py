"""Tests for deterministic human citation-review package construction."""

import unittest

from evaluation.build_citation_review import (
    build_citation_pairs,
    select_review_questions,
)


class CitationReviewTests(unittest.TestCase):
    def test_expands_one_claim_with_two_citations_into_two_pairs(self) -> None:
        pairs = build_citation_pairs(
            "A supported claim [1][2].",
            [
                {"chunk_id": "a", "text": "first"},
                {"chunk_id": "b", "text": "second"},
            ],
        )

        self.assertEqual([item["citation_rank"] for item in pairs], [1, 2])
        self.assertEqual([item["chunk_id"] for item in pairs], ["a", "b"])
        self.assertTrue(all(item["supported"] is None for item in pairs))

    def test_selection_contains_30_answerable_questions_and_all_chinese(self) -> None:
        questions = []
        counts = {
            ("fact", "en"): 15,
            ("fact", "zh"): 5,
            ("comparison", "en"): 9,
            ("comparison", "zh"): 1,
            ("cross_document", "en"): 8,
            ("cross_document", "zh"): 2,
            ("no_answer", "en"): 8,
            ("no_answer", "zh"): 2,
        }
        for (question_type, language), count in counts.items():
            for index in range(count):
                questions.append(
                    {
                        "id": f"{question_type}_{language}_{index}",
                        "split": "test",
                        "type": question_type,
                        "language": language,
                    }
                )

        selected = select_review_questions(questions)

        self.assertEqual(len(selected), 30)
        self.assertNotIn("no_answer", {item["type"] for item in selected})
        self.assertEqual(
            {item["id"] for item in selected if item["language"] == "zh"},
            {
                item["id"]
                for item in questions
                if item["language"] == "zh" and item["type"] != "no_answer"
            },
        )


if __name__ == "__main__":
    unittest.main()
