"""Evaluation dataset and qrels loading with schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QUESTION_REQUIRED_FIELDS = {
    "id",
    "type",
    "question",
    "expected_answer",
    "source_files",
    "source_pages",
}
QUESTION_TYPES = {
    "fact",
    "comparison",
    "cross_document",
    "no_answer",
}
QUESTION_SPLITS = {"legacy", "dev", "test"}
QUESTION_DIFFICULTIES = {"easy", "medium", "hard", "unspecified"}
QREL_RELEVANCE_LEVELS = {0, 1, 2, 3}


def _require_non_empty_string(value: Any, field: str, item_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{item_id} 的 {field} 必须是非空字符串。")
    return value.strip()


def _require_string_list(value: Any, field: str, item_id: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise ValueError(f"{item_id} 的 {field} 必须是字符串数组。")
    return [item.strip() for item in value]


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load questions and add defaults for the expanded evaluation schema."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("评测集必须是非空 JSON 数组。")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_item in questions:
        if not isinstance(raw_item, dict):
            raise ValueError("评测题必须是 JSON object。")

        missing = QUESTION_REQUIRED_FIELDS - set(raw_item)
        item_label = str(raw_item.get("id", "<unknown>"))
        if missing:
            raise ValueError(
                f"问题缺少字段：{item_label} -> {sorted(missing)}"
            )

        item = dict(raw_item)
        question_id = _require_non_empty_string(item["id"], "id", item_label)
        if question_id in seen_ids:
            raise ValueError(f"问题 ID 重复：{question_id}")
        seen_ids.add(question_id)
        item["id"] = question_id

        question_type = _require_non_empty_string(
            item["type"], "type", question_id
        )
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"{question_id} 的 type 不受支持：{question_type}"
            )
        item["type"] = question_type

        item["question"] = _require_non_empty_string(
            item["question"], "question", question_id
        )
        item["expected_answer"] = _require_non_empty_string(
            item["expected_answer"], "expected_answer", question_id
        )
        item["source_files"] = _require_string_list(
            item["source_files"], "source_files", question_id
        )

        source_pages = item["source_pages"]
        if not isinstance(source_pages, list) or any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in source_pages
        ):
            raise ValueError(
                f"{question_id} 的 source_pages 必须是正整数数组。"
            )
        if len(item["source_files"]) != len(source_pages):
            raise ValueError(f"来源文件与页码数量不一致：{question_id}")

        if question_type == "no_answer" and (
            item["source_files"] or source_pages
        ):
            raise ValueError(f"无答案题不能包含预期来源：{question_id}")
        if question_type != "no_answer" and not item["source_files"]:
            raise ValueError(f"可回答题必须包含预期来源：{question_id}")

        split = str(item.get("split", "legacy")).strip()
        if split not in QUESTION_SPLITS:
            raise ValueError(f"{question_id} 的 split 不受支持：{split}")
        item["split"] = split

        language = str(item.get("language", "en")).strip()
        if not language:
            raise ValueError(f"{question_id} 的 language 不能为空。")
        item["language"] = language

        difficulty = str(item.get("difficulty", "unspecified")).strip()
        if difficulty not in QUESTION_DIFFICULTIES:
            raise ValueError(
                f"{question_id} 的 difficulty 不受支持：{difficulty}"
            )
        item["difficulty"] = difficulty
        item["tags"] = _require_string_list(
            item.get("tags", []), "tags", question_id
        )
        item["expected_key_points"] = _require_string_list(
            item.get("expected_key_points", []),
            "expected_key_points",
            question_id,
        )
        normalized.append(item)

    return normalized


def load_qrels(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and validate the shape of Chunk-level relevance judgments."""
    raw_qrels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_qrels, dict):
        raise ValueError("qrels 必须是以问题 ID 为键的 JSON object。")

    normalized: dict[str, list[dict[str, Any]]] = {}
    for raw_question_id, raw_judgments in raw_qrels.items():
        question_id = _require_non_empty_string(
            raw_question_id, "question_id", "qrels"
        )
        if not isinstance(raw_judgments, list):
            raise ValueError(f"{question_id} 的 qrels 必须是数组。")

        seen_chunk_ids: set[str] = set()
        judgments: list[dict[str, Any]] = []
        for index, raw_judgment in enumerate(raw_judgments):
            label = f"{question_id} qrel[{index}]"
            if not isinstance(raw_judgment, dict):
                raise ValueError(f"{label} 必须是 JSON object。")

            required = {
                "chunk_id",
                "file_name",
                "page_number",
                "relevance",
                "reason",
            }
            missing = required - set(raw_judgment)
            if missing:
                raise ValueError(f"{label} 缺少字段：{sorted(missing)}")

            judgment = dict(raw_judgment)
            chunk_id = _require_non_empty_string(
                judgment["chunk_id"], "chunk_id", label
            )
            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"{question_id} 包含重复 qrel chunk_id：{chunk_id}"
                )
            seen_chunk_ids.add(chunk_id)
            judgment["chunk_id"] = chunk_id
            judgment["file_name"] = _require_non_empty_string(
                judgment["file_name"], "file_name", label
            )

            page_number = judgment["page_number"]
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise ValueError(f"{label} 的 page_number 必须是正整数。")

            relevance = judgment["relevance"]
            if (
                isinstance(relevance, bool)
                or not isinstance(relevance, int)
                or relevance not in QREL_RELEVANCE_LEVELS
            ):
                raise ValueError(f"{label} 的 relevance 必须是 0、1、2 或 3。")

            judgment["reason"] = _require_non_empty_string(
                judgment["reason"], "reason", label
            )
            judgments.append(judgment)

        normalized[question_id] = judgments

    return normalized


def validate_qrels(
    questions: list[dict[str, Any]],
    qrels: dict[str, list[dict[str, Any]]],
    *,
    index_metadata: dict[str, dict[str, Any]] | None = None,
    require_complete: bool = False,
) -> dict[str, int]:
    """Cross-check qrels against questions and optional Chroma metadata."""
    questions_by_id = {item["id"]: item for item in questions}
    unknown_ids = sorted(set(qrels) - set(questions_by_id))
    if unknown_ids:
        raise ValueError(f"qrels 包含未知问题 ID：{unknown_ids}")

    if require_complete:
        missing_ids = sorted(set(questions_by_id) - set(qrels))
        if missing_ids:
            raise ValueError(f"以下问题缺少 qrels：{missing_ids}")

    judgment_count = 0
    relevant_judgment_count = 0
    for question_id, judgments in qrels.items():
        question = questions_by_id[question_id]
        relevant = [item for item in judgments if item["relevance"] > 0]
        if question["type"] == "no_answer" and judgments:
            raise ValueError(f"无答案题的 qrels 必须为空：{question_id}")
        if question["type"] != "no_answer" and not relevant:
            raise ValueError(
                f"可回答题至少需要一个 relevance > 0 的 qrel：{question_id}"
            )

        for judgment in judgments:
            chunk_id = judgment["chunk_id"]
            if index_metadata is not None:
                metadata = index_metadata.get(chunk_id)
                if metadata is None:
                    raise ValueError(f"qrel Chunk 不存在于索引：{chunk_id}")
                if str(metadata.get("file_name")) != judgment["file_name"]:
                    raise ValueError(f"qrel 文件名与索引不一致：{chunk_id}")
                try:
                    indexed_page = int(metadata.get("page_number"))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"索引中的 page_number 无效：{chunk_id}"
                    ) from None
                if indexed_page != judgment["page_number"]:
                    raise ValueError(f"qrel 页码与索引不一致：{chunk_id}")

        judgment_count += len(judgments)
        relevant_judgment_count += len(relevant)

    return {
        "question_count": len(questions),
        "question_with_qrels_count": len(qrels),
        "judgment_count": judgment_count,
        "relevant_judgment_count": relevant_judgment_count,
    }
