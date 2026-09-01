"""Tests for the paid-generation evaluation safety boundary."""

import unittest
from pathlib import Path

from evaluation.evaluate_generation import (
    build_context_from_sources,
    build_generation_summary,
    estimate_prompt_characters,
    run_generation_evaluation,
    validate_retrieval_payload,
)


class GenerationEvaluationTests(unittest.TestCase):
    def test_paid_api_requires_explicit_flag_before_loading_inputs(self) -> None:
        with self.assertRaisesRegex(PermissionError, "allow-paid-api"):
            run_generation_evaluation(
                questions_path=Path("missing-questions.json"),
                qrels_path=Path("missing-qrels.json"),
                retrieval_results_path=Path("missing-results.json"),
                output_path=Path("unused.json"),
            )

    def test_builds_numbered_context_from_frozen_sources(self) -> None:
        context = build_context_from_sources(
            [{
                "rank": 1,
                "file_name": "paper.pdf",
                "page_number": 3,
                "chunk_id": "chunk-1",
                "text": "evidence",
            }]
        )

        self.assertIn("[Source 1]", context)
        self.assertIn("Chunk ID: chunk-1", context)
        self.assertIn("evidence", context)

    def test_estimates_prompt_size_without_api_calls(self) -> None:
        estimate = estimate_prompt_characters(
            [{"id": "q1", "question": "Question?"}],
            {
                "results": [{
                    "id": "q1",
                    "retrieved_sources": [{
                        "rank": 1,
                        "file_name": "paper.pdf",
                        "page_number": 1,
                        "chunk_id": "chunk-1",
                        "text": "evidence",
                    }],
                }]
            },
        )

        self.assertEqual(estimate["prompt_count"], 1)
        self.assertGreater(estimate["total_characters"], len("evidence"))

    def test_validates_exact_split_and_question_order(self) -> None:
        questions = [{"id": "q1"}, {"id": "q2"}]
        payload = {
            "split": "test",
            "runs": [{
                "method": "hybrid",
                "results": [{"id": "q1"}, {"id": "q2"}],
            }],
        }

        run = validate_retrieval_payload(payload, questions, "test")

        self.assertEqual(run["method"], "hybrid")

    def test_rejects_non_ace_method_and_changed_order(self) -> None:
        questions = [{"id": "q1"}, {"id": "q2"}]
        with self.assertRaisesRegex(ValueError, "A、C、E"):
            validate_retrieval_payload(
                {
                    "split": "test",
                    "runs": [{"method": "bm25", "results": []}],
                },
                questions,
                "test",
            )
        with self.assertRaisesRegex(ValueError, "顺序"):
            validate_retrieval_payload(
                {
                    "split": "test",
                    "runs": [{
                        "method": "dense",
                        "results": [{"id": "q2"}, {"id": "q1"}],
                    }],
                },
                questions,
                "test",
            )

    def test_summary_separates_answerable_and_no_answer_metrics(self) -> None:
        summary = build_generation_summary(
            [
                {
                    "error": None,
                    "latency_seconds": 1.0,
                    "answer_score": None,
                    "metrics": {
                        "refusal_correct": True,
                        "citation_format_validity": True,
                        "citation_index_validity": True,
                        "answer_has_citation": True,
                        "no_answer_has_no_spurious_citation": None,
                        "citation_qrel_precision": 1.0,
                        "citation_qrel_recall": 0.5,
                    },
                },
                {
                    "error": None,
                    "latency_seconds": 3.0,
                    "answer_score": None,
                    "metrics": {
                        "refusal_correct": True,
                        "citation_format_validity": True,
                        "citation_index_validity": True,
                        "answer_has_citation": None,
                        "no_answer_has_no_spurious_citation": True,
                        "citation_qrel_precision": None,
                        "citation_qrel_recall": None,
                    },
                },
            ]
        )

        self.assertEqual(summary["answer_has_citation_rate"], 1.0)
        self.assertEqual(summary["no_answer_has_no_spurious_citation_rate"], 1.0)
        self.assertEqual(summary["average_generation_latency_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
