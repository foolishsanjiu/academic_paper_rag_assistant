"""Tests for safe, resumable human-review label persistence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.review_state import (
    calculate_review_progress,
    initialize_label_file,
    save_label_file,
)


def review_package() -> dict:
    questions = []
    for index in range(30):
        systems = {}
        for label in ("A", "C", "E"):
            systems[label] = {
                "answer": f"answer {label}",
                "answer_score": None,
                "answer_notes": "",
                "citation_pairs": [
                    {
                        "pair_id": "pair_01",
                        "claim": "claim",
                        "citation_rank": 1,
                        "chunk_id": "chunk_1",
                        "file_name": "paper.pdf",
                        "page_number": 1,
                        "source_text": "source",
                        "supported": None,
                        "reviewer_notes": "",
                    }
                ],
            }
        questions.append(
            {
                "id": f"q{index:02d}",
                "type": "fact",
                "language": "en",
                "question": "question",
                "expected_answer": "expected",
                "systems": systems,
            }
        )
    return {
        "review_protocol": {"systems": ["A", "C", "E"]},
        "questions": questions,
    }


class ReviewStateTests(unittest.TestCase):
    def test_initializes_separate_labels_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.json"
            labels = Path(directory) / "labels.json"
            source.write_text(
                json.dumps(review_package(), ensure_ascii=False), encoding="utf-8"
            )
            source_bytes = source.read_bytes()

            loaded = initialize_label_file(source, labels)

            self.assertTrue(labels.exists())
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(loaded, review_package())

    def test_rejects_source_as_label_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.json"
            source.write_text(json.dumps(review_package()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不能使用同一路径"):
                initialize_label_file(source, source)

    def test_saves_labels_and_reports_resumable_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.json"
            labels = Path(directory) / "labels.json"
            source.write_text(json.dumps(review_package()), encoding="utf-8")
            package = initialize_label_file(source, labels)
            system = package["questions"][0]["systems"]["A"]
            system["answer_score"] = 2
            system["citation_pairs"][0]["supported"] = True

            save_label_file(source, labels, package)
            reloaded = initialize_label_file(source, labels)
            progress = calculate_review_progress(reloaded)

            self.assertEqual(progress["answer_completed"], 1)
            self.assertEqual(progress["answer_total"], 90)
            self.assertEqual(progress["citation_completed"], 1)
            self.assertEqual(progress["citation_total"], 90)
            self.assertEqual(progress["target_completed"], 1)

    def test_rejects_changes_to_read_only_content_and_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.json"
            labels = Path(directory) / "labels.json"
            source.write_text(json.dumps(review_package()), encoding="utf-8")

            changed = deepcopy(review_package())
            changed["questions"][0]["question"] = "changed"
            with self.assertRaisesRegex(ValueError, "只读内容"):
                save_label_file(source, labels, changed)

            invalid = review_package()
            invalid["questions"][0]["systems"]["A"]["answer_score"] = 3
            with self.assertRaisesRegex(ValueError, "answer_score"):
                save_label_file(source, labels, invalid)


if __name__ == "__main__":
    unittest.main()
