"""Unit tests for automatic citation correctness metrics."""

import unittest

from evaluation.citation_metrics import (
    calculate_citation_metrics,
    parse_citation_ranks,
)


SOURCES = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "x"}]
QRELS = [
    {"chunk_id": "a", "relevance": 3},
    {"chunk_id": "b", "relevance": 2},
    {"chunk_id": "x", "relevance": 0},
]


class CitationMetricTests(unittest.TestCase):
    def test_parses_adjacent_numeric_citations(self) -> None:
        ranks, valid = parse_citation_ranks("Claim [1][3].")

        self.assertEqual(ranks, [1, 3])
        self.assertTrue(valid)

    def test_rejects_malformed_and_unbalanced_brackets(self) -> None:
        for answer in ("Claim [one].", "Claim [1,2].", "Claim [1."):
            with self.subTest(answer=answer):
                self.assertFalse(parse_citation_ranks(answer)[1])

    def test_scores_qrel_precision_and_required_evidence_recall(self) -> None:
        metrics = calculate_citation_metrics(
            "Supported [1], irrelevant [3].",
            SOURCES,
            QRELS,
            "fact",
        )

        self.assertEqual(metrics["citation_qrel_precision"], 0.5)
        self.assertEqual(metrics["citation_qrel_recall"], 0.5)
        self.assertTrue(metrics["answer_has_citation"])

    def test_out_of_range_rank_fails_index_validity(self) -> None:
        metrics = calculate_citation_metrics("Claim [4].", SOURCES, QRELS, "fact")

        self.assertFalse(metrics["citation_index_validity"])
        self.assertFalse(metrics["answer_has_citation"])

    def test_no_answer_requires_no_citation_like_brackets(self) -> None:
        clean = calculate_citation_metrics("Insufficient evidence.", [], [], "no_answer")
        cited = calculate_citation_metrics("Insufficient [1].", [], [], "no_answer")

        self.assertTrue(clean["no_answer_has_no_spurious_citation"])
        self.assertFalse(cited["no_answer_has_no_spurious_citation"])
        self.assertIsNone(clean["answer_has_citation"])

    def test_missing_citation_is_valid_format_but_fails_coverage(self) -> None:
        metrics = calculate_citation_metrics("Uncited answer.", SOURCES, QRELS, "fact")

        self.assertTrue(metrics["citation_format_validity"])
        self.assertTrue(metrics["citation_index_validity"])
        self.assertFalse(metrics["answer_has_citation"])
        self.assertIsNone(metrics["citation_qrel_precision"])


if __name__ == "__main__":
    unittest.main()
