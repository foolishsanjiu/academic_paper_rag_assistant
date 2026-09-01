"""Safe persistence and progress helpers for human citation review."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from evaluation.evaluate import write_json_atomic


SYSTEMS = ("A", "C", "E")


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象。")
    return payload


def _is_supported_label(value: Any) -> bool:
    return value is None or isinstance(value, bool) or value == "uncertain"


def validate_review_package(package: dict[str, Any]) -> None:
    """Validate package structure and all mutable label values."""
    protocol = package.get("review_protocol")
    if not isinstance(protocol, dict) or protocol.get("systems") != list(SYSTEMS):
        raise ValueError("复核包必须按 A、C、E 顺序包含三个系统。")
    questions = package.get("questions")
    if not isinstance(questions, list) or len(questions) != 30:
        raise ValueError("复核包必须包含固定的 30 道题。")

    question_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("每道题必须是 JSON 对象。")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("每道题都必须有非空 id。")
        if question_id in question_ids:
            raise ValueError(f"题目 id 重复：{question_id}")
        question_ids.add(question_id)

        systems = question.get("systems")
        if not isinstance(systems, dict) or set(systems) != set(SYSTEMS):
            raise ValueError(f"{question_id} 必须同时包含 A、C、E。")
        for label in SYSTEMS:
            system = systems[label]
            if not isinstance(system, dict):
                raise ValueError(f"{question_id} {label} 必须是 JSON 对象。")
            score = system.get("answer_score")
            if (
                score is not None
                and (isinstance(score, bool) or score not in {0, 1, 2})
            ):
                raise ValueError(f"{question_id} {label} 的 answer_score 非法。")
            if not isinstance(system.get("answer_notes", ""), str):
                raise ValueError(f"{question_id} {label} 的 answer_notes 必须是文本。")

            pairs = system.get("citation_pairs")
            if not isinstance(pairs, list):
                raise ValueError(f"{question_id} {label} 的 citation_pairs 必须是列表。")
            pair_ids: set[str] = set()
            for pair in pairs:
                if not isinstance(pair, dict):
                    raise ValueError(f"{question_id} {label} 的引用对必须是 JSON 对象。")
                pair_id = pair.get("pair_id")
                if not isinstance(pair_id, str) or not pair_id:
                    raise ValueError(f"{question_id} {label} 存在无效 pair_id。")
                if pair_id in pair_ids:
                    raise ValueError(f"{question_id} {label} 的 pair_id 重复：{pair_id}")
                pair_ids.add(pair_id)
                if not _is_supported_label(pair.get("supported")):
                    raise ValueError(
                        f"{question_id} {label} {pair_id} 的 supported 非法。"
                    )
                if not isinstance(pair.get("reviewer_notes", ""), str):
                    raise ValueError(
                        f"{question_id} {label} {pair_id} 的 reviewer_notes 必须是文本。"
                    )


def _immutable_snapshot(package: dict[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(package)
    for question in snapshot["questions"]:
        for system in question["systems"].values():
            system.pop("answer_score", None)
            system.pop("answer_notes", None)
            for pair in system["citation_pairs"]:
                pair.pop("supported", None)
                pair.pop("reviewer_notes", None)
    return snapshot


def validate_compatible_labels(
    source: dict[str, Any], labels: dict[str, Any]
) -> None:
    """Ensure labels only differ from the source in human-editable fields."""
    validate_review_package(source)
    validate_review_package(labels)
    if _immutable_snapshot(source) != _immutable_snapshot(labels):
        raise ValueError("标签文件修改了问题、答案或引用来源等只读内容。")


def initialize_label_file(source_path: Path, labels_path: Path) -> dict[str, Any]:
    """Create a separate label file once, or load an existing valid checkpoint."""
    if _same_path(source_path, labels_path):
        raise ValueError("原始复核包与标签文件不能使用同一路径。")
    source = _read_json(source_path)
    validate_review_package(source)
    if labels_path.exists():
        labels = _read_json(labels_path)
        validate_compatible_labels(source, labels)
        return labels

    labels = deepcopy(source)
    write_json_atomic(labels_path, labels)
    return labels


def save_label_file(
    source_path: Path,
    labels_path: Path,
    labels: dict[str, Any],
) -> None:
    """Atomically save labels after checking source immutability and schema."""
    if _same_path(source_path, labels_path):
        raise ValueError("原始复核包与标签文件不能使用同一路径。")
    source = _read_json(source_path)
    validate_compatible_labels(source, labels)
    write_json_atomic(labels_path, labels)


def calculate_review_progress(package: dict[str, Any]) -> dict[str, int]:
    """Count completed answer and citation labels."""
    validate_review_package(package)
    answer_total = 0
    answer_completed = 0
    citation_total = 0
    citation_completed = 0
    target_total = 0
    target_completed = 0
    for question in package["questions"]:
        for label in SYSTEMS:
            target_total += 1
            system = question["systems"][label]
            answer_total += 1
            score_done = system["answer_score"] is not None
            answer_completed += int(score_done)
            pairs = system["citation_pairs"]
            citation_total += len(pairs)
            completed_pairs = sum(
                pair["supported"] is not None for pair in pairs
            )
            citation_completed += completed_pairs
            target_completed += int(score_done and completed_pairs == len(pairs))
    return {
        "answer_completed": answer_completed,
        "answer_total": answer_total,
        "citation_completed": citation_completed,
        "citation_total": citation_total,
        "target_completed": target_completed,
        "target_total": target_total,
    }


def target_is_complete(system: dict[str, Any]) -> bool:
    """Return whether one question-system target has every required label."""
    return system.get("answer_score") is not None and all(
        pair.get("supported") is not None
        for pair in system.get("citation_pairs", [])
    )
