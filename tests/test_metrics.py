from math import log2
import unittest

from evaluation.metrics import (
    calculate_ranked_retrieval_metrics,
    mean_ranked_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


QRELS = [
    {"chunk_id": "a", "relevance": 3},
    {"chunk_id": "b", "relevance": 2},
    {"chunk_id": "c", "relevance": 1},
    {"chunk_id": "negative", "relevance": 0},
]


class RankedRetrievalMetricTests(unittest.TestCase):
    def test_recall_at_k_uses_all_positive_qrels(self) -> None:
        self.assertAlmostEqual(recall_at_k(QRELS, ["x", "a", "c"], 3), 2 / 3)

    def test_reciprocal_rank_uses_first_relevant_result(self) -> None:
        self.assertEqual(
            reciprocal_rank_at_k(QRELS, ["negative", "x", "b"], 3),
            1 / 3,
        )

    def test_ndcg_uses_graded_relevance(self) -> None:
        actual = ndcg_at_k(QRELS, ["x", "a", "c"], 3)
        dcg = 7 / log2(3) + 1 / log2(4)
        ideal_dcg = 7 + 3 / log2(3) + 1 / log2(4)
        self.assertAlmostEqual(actual, dcg / ideal_dcg)

    def test_ideal_ranking_has_ndcg_one(self) -> None:
        self.assertEqual(ndcg_at_k(QRELS, ["a", "b", "c"], 3), 1.0)

    def test_duplicate_retrieved_chunks_do_not_gain_extra_rank(self) -> None:
        qrels = [
            {"chunk_id": "a", "relevance": 1},
            {"chunk_id": "b", "relevance": 1},
        ]
        self.assertEqual(recall_at_k(qrels, ["a", "a", "b"], 2), 1.0)

    def test_no_answer_qrels_are_excluded(self) -> None:
        self.assertIsNone(recall_at_k([], ["a"], 5))
        self.assertIsNone(reciprocal_rank_at_k([], ["a"], 5))
        self.assertIsNone(ndcg_at_k([], ["a"], 5))

    def test_calculates_named_metrics_for_each_cutoff(self) -> None:
        metrics = calculate_ranked_retrieval_metrics(
            QRELS,
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
            cutoffs=(1, 2),
        )
        self.assertEqual(
            set(metrics),
            {
                "recall@1",
                "mrr@1",
                "ndcg@1",
                "recall@2",
                "mrr@2",
                "ndcg@2",
            },
        )

    def test_macro_average_ignores_none_and_non_rank_metrics(self) -> None:
        averaged = mean_ranked_metrics(
            [
                {"recall@5": 0.5, "mrr@5": None, "page_any_hit": True},
                {"recall@5": 1.0, "mrr@5": 0.25},
            ]
        )
        self.assertEqual(averaged, {"mrr@5": 0.25, "recall@5": 0.75})

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "k 必须"):
            recall_at_k(QRELS, ["a"], 0)


if __name__ == "__main__":
    unittest.main()
