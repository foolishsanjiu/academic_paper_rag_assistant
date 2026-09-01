import json
from pathlib import Path
import tempfile
import unittest

from evaluation.dataset import load_qrels, load_questions, validate_qrels
from evaluation.validate_dataset import normalize_index_metadata


def question(
    question_id: str = "fact_001",
    question_type: str = "fact",
) -> dict:
    has_answer = question_type != "no_answer"
    return {
        "id": question_id,
        "type": question_type,
        "question": "What is the method?",
        "expected_answer": "The expected answer.",
        "source_files": ["paper.pdf"] if has_answer else [],
        "source_pages": [3] if has_answer else [],
    }


class QuestionDatasetTests(unittest.TestCase):
    def write_json(self, directory: str, name: str, payload) -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_legacy_question_receives_expanded_schema_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "questions.json", [question()])
            loaded = load_questions(path)

        self.assertEqual(loaded[0]["split"], "legacy")
        self.assertEqual(loaded[0]["language"], "en")
        self.assertEqual(loaded[0]["difficulty"], "unspecified")
        self.assertEqual(loaded[0]["tags"], [])
        self.assertEqual(loaded[0]["expected_key_points"], [])

    def test_rejects_answerable_question_without_source(self) -> None:
        item = question()
        item["source_files"] = []
        item["source_pages"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "questions.json", [item])
            with self.assertRaisesRegex(ValueError, "必须包含预期来源"):
                load_questions(path)

    def test_rejects_no_answer_question_with_source(self) -> None:
        item = question(question_type="no_answer")
        item["source_files"] = ["paper.pdf"]
        item["source_pages"] = [3]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_json(directory, "questions.json", [item])
            with self.assertRaisesRegex(ValueError, "不能包含预期来源"):
                load_questions(path)


class QrelsDatasetTests(unittest.TestCase):
    def test_validates_qrels_against_index_metadata(self) -> None:
        questions = [
            {
                **question(),
                "split": "dev",
                "language": "en",
                "difficulty": "medium",
                "tags": [],
                "expected_key_points": [],
            }
        ]
        qrels = {
            "fact_001": [
                {
                    "chunk_id": "doc::page_3::chunk_0",
                    "file_name": "paper.pdf",
                    "page_number": 3,
                    "relevance": 3,
                    "reason": "Direct evidence.",
                }
            ]
        }
        summary = validate_qrels(
            questions,
            qrels,
            index_metadata={
                "doc::page_3::chunk_0": {
                    "file_name": "paper.pdf",
                    "page_number": 3,
                }
            },
            require_complete=True,
        )
        self.assertEqual(summary["judgment_count"], 1)
        self.assertEqual(summary["relevant_judgment_count"], 1)

    def test_rejects_qrel_page_mismatch(self) -> None:
        questions = [question()]
        qrels = {
            "fact_001": [
                {
                    "chunk_id": "chunk-1",
                    "file_name": "paper.pdf",
                    "page_number": 3,
                    "relevance": 2,
                    "reason": "Supports one key point.",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "页码与索引不一致"):
            validate_qrels(
                questions,
                qrels,
                index_metadata={
                    "chunk-1": {
                        "file_name": "paper.pdf",
                        "page_number": 4,
                    }
                },
            )

    def test_load_qrels_rejects_duplicate_chunks(self) -> None:
        judgment = {
            "chunk_id": "chunk-1",
            "file_name": "paper.pdf",
            "page_number": 3,
            "relevance": 3,
            "reason": "Direct evidence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrels.json"
            path.write_text(
                json.dumps({"fact_001": [judgment, judgment]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复 qrel chunk_id"):
                load_qrels(path)

    def test_rejects_misaligned_index_metadata_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "数量不一致"):
            normalize_index_metadata(
                {"ids": ["chunk-1"], "metadatas": []}
            )


if __name__ == "__main__":
    unittest.main()
