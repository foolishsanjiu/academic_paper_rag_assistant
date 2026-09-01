"""Tests for human citation-entailment metric aggregation."""

import unittest

from evaluation.evaluate_human_citations import (
    audit_label_quality,
    calculate_human_metrics,
)


def labeled_package() -> dict:
    questions = []
    for index in range(30):
        systems = {}
        for label in ("A", "C", "E"):
            systems[label] = {
                "answer_score": 2 if index else 1,
                "citation_pairs": [
                    {
                        "pair_id": "pair_01",
                        "claim": "claim",
                        "chunk_id": "chunk_1",
                        "supported": True,
                    },
                    {
                        "pair_id": "pair_02",
                        "claim": "claim",
                        "chunk_id": "chunk_2",
                        "supported": False if index else "uncertain",
                    },
                ],
            }
        questions.append({"id": f"q{index}", "systems": systems})
    return {
        "review_protocol": {"systems": ["A", "C", "E"]},
        "questions": questions,
    }


class HumanCitationEvaluationTests(unittest.TestCase):
    def test_calculates_answer_and_entailment_metrics(self) -> None:
        summary = calculate_human_metrics(labeled_package())

        self.assertEqual(summary["A"]["question_count"], 30)
        self.assertEqual(summary["A"]["citation_pair_count"], 60)
        self.assertEqual(summary["A"]["uncertain_citation_pair_count"], 1)
        self.assertAlmostEqual(summary["A"]["citation_correctness"], 30 / 59)
        self.assertEqual(summary["A"]["citation_correctness_lower_bound"], 0.5)
        self.assertAlmostEqual(
            summary["A"]["citation_correctness_upper_bound"], 31 / 60
        )
        self.assertEqual(summary["A"]["claim_citation_coverage"], 1.0)

    def test_audits_duplicate_conflicts_and_answer_score_comparisons(self) -> None:
        package = labeled_package()
        package["questions"][1]["systems"]["C"]["citation_pairs"][0][
            "supported"
        ] = False

        audit = audit_label_quality(package)

        self.assertEqual(audit["exact_duplicate_group_count"], 60)
        self.assertEqual(audit["decided_conflict_group_count"], 1)
        self.assertEqual(
            audit["system_summary"]["A"]["citation_label_distribution"],
            {"true": 30, "false": 29, "uncertain": 1},
        )
        self.assertEqual(
            audit["pairwise_answer_score_comparisons"]["E_vs_A"]["ties"],
            30,
        )
        self.assertEqual(
            audit["pairwise_answer_score_comparisons"]["E_vs_A"][
                "two_sided_sign_test_p_value"
            ],
            1.0,
        )

    def test_rejects_incomplete_answer_or_pair_labels(self) -> None:
        package = labeled_package()
        package["questions"][0]["systems"]["A"]["answer_score"] = None
        with self.assertRaisesRegex(ValueError, "answer_score"):
            calculate_human_metrics(package)

        package = labeled_package()
        package["questions"][0]["systems"]["A"]["citation_pairs"][0][
            "supported"
        ] = None
        with self.assertRaisesRegex(ValueError, "supported"):
            calculate_human_metrics(package)


if __name__ == "__main__":
    unittest.main()
