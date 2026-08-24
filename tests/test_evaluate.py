"""Unit tests for evaluation metrics that do not require model loading."""

import unittest

from evaluation.evaluate import (
    calculate_refusal_metrics,
    calculate_retrieval_metrics,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_requires_all_expected_files_for_all_hit(self) -> None:
        metrics = calculate_retrieval_metrics(
            ["a.pdf", "b.pdf"],
            [1, 2],
            [
                {
                    "file_name": "a.pdf",
                    "page_number": 1,
                }
            ],
        )

        self.assertTrue(metrics["retrieval_any_hit"])
        self.assertFalse(metrics["retrieval_all_hit"])
        self.assertTrue(metrics["page_any_hit"])
        self.assertFalse(metrics["page_all_hit"])

    def test_no_answer_has_no_retrieval_target(self) -> None:
        metrics = calculate_retrieval_metrics([], [], [])

        self.assertIsNone(metrics["retrieval_any_hit"])
        self.assertIsNone(metrics["page_all_hit"])


class RefusalMetricTests(unittest.TestCase):
    def test_expected_refusal_is_correct(self) -> None:
        metrics = calculate_refusal_metrics(
            "no_answer",
            "prefix STANDARD_REFUSAL suffix",
            "STANDARD_REFUSAL",
        )

        self.assertTrue(metrics["refused"])
        self.assertTrue(metrics["refusal_correct"])

    def test_answered_question_must_not_refuse(self) -> None:
        metrics = calculate_refusal_metrics(
            "fact",
            "A grounded answer.",
            "STANDARD_REFUSAL",
        )

        self.assertFalse(metrics["refused"])
        self.assertTrue(metrics["refusal_correct"])


if __name__ == "__main__":
    unittest.main()
