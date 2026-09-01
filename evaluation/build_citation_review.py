"""Build a deterministic human citation-entailment review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.citation_metrics import parse_citation_ranks  # noqa: E402
from evaluation.dataset import load_questions  # noqa: E402
from evaluation.evaluate import write_json_atomic  # noqa: E402
from evaluation.evaluate_generation import file_sha256  # noqa: E402


SELECTION_SEED = "m7-citation-review-v1"
ENGLISH_QUOTAS = {
    "fact": 10,
    "comparison": 6,
    "cross_document": 6,
}
SENTENCE_PATTERN = re.compile(r"[^.!?。！？]+[.!?。！？]?")


def _stable_key(question_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}:{question_id}".encode("utf-8")
    ).hexdigest()


def select_review_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select 30 answerable test questions with all Chinese questions."""
    answerable = [
        item
        for item in questions
        if item["split"] == "test" and item["type"] != "no_answer"
    ]
    selected = [item for item in answerable if item["language"] == "zh"]
    for question_type, quota in ENGLISH_QUOTAS.items():
        candidates = sorted(
            (
                item
                for item in answerable
                if item["language"] == "en" and item["type"] == question_type
            ),
            key=lambda item: _stable_key(item["id"]),
        )
        if len(candidates) < quota:
            raise ValueError(f"{question_type} 英文题不足 {quota} 道。")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda item: item["id"])
    if len(selected) != 30:
        raise ValueError(f"人工复核题数必须为 30，实际为 {len(selected)}。")
    return selected


def build_citation_pairs(
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand every cited answer sentence into reviewable claim-source pairs."""
    pairs: list[dict[str, Any]] = []
    for sentence in SENTENCE_PATTERN.findall(answer):
        claim = sentence.strip()
        if not claim:
            continue
        ranks, _ = parse_citation_ranks(claim)
        for rank in dict.fromkeys(ranks):
            if not 1 <= rank <= len(sources):
                continue
            source = sources[rank - 1]
            pairs.append(
                {
                    "pair_id": f"pair_{len(pairs) + 1:02d}",
                    "claim": claim,
                    "citation_rank": rank,
                    "chunk_id": source.get("chunk_id"),
                    "file_name": source.get("file_name"),
                    "page_number": source.get("page_number"),
                    "source_text": source.get("text"),
                    "supported": None,
                    "reviewer_notes": "",
                }
            )
    return pairs


def build_review_package(
    questions: list[dict[str, Any]],
    generation_payloads: dict[str, dict[str, Any]],
    source_files: dict[str, Path],
) -> dict[str, Any]:
    """Build one package containing the same 30 questions for A/C/E."""
    selected = select_review_questions(questions)
    result_maps = {
        label: {item["id"]: item for item in payload["results"]}
        for label, payload in generation_payloads.items()
    }
    rows: list[dict[str, Any]] = []
    for question in selected:
        systems: dict[str, Any] = {}
        for label in sorted(generation_payloads):
            result = result_maps[label].get(question["id"])
            if result is None or result.get("error") is not None:
                raise ValueError(f"{label} 缺少可复核结果：{question['id']}")
            systems[label] = {
                "answer": result["answer"],
                "answer_score": None,
                "answer_notes": "",
                "citation_pairs": build_citation_pairs(
                    result["answer"],
                    result["retrieved_sources"],
                ),
            }
        rows.append(
            {
                "id": question["id"],
                "type": question["type"],
                "language": question["language"],
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "systems": systems,
            }
        )
    return {
        "review_protocol": {
            "selection_seed": SELECTION_SEED,
            "question_count": len(rows),
            "selection": "all 8 answerable Chinese test questions plus deterministic English quotas fact=10, comparison=6, cross_document=6",
            "systems": sorted(generation_payloads),
            "source_files": {
                label: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for label, path in sorted(source_files.items())
            },
            "supported_labels": {
                "true": "The cited Chunk supports the claim.",
                "false": "The cited Chunk does not support the claim.",
                "uncertain": "A domain reviewer cannot decide confidently.",
            },
            "answer_score_scale": {
                "0": "incorrect",
                "1": "partially correct",
                "2": "mostly or fully correct",
            },
        },
        "questions": rows,
    }


def parse_result_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result 必须使用 LABEL=PATH。")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--result 的 LABEL 和 PATH 不能为空。")
    return label.strip(), Path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic 30-question A/C/E review package."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions.json"),
    )
    parser.add_argument(
        "--result",
        action="append",
        type=parse_result_argument,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_files = dict(args.result)
    if set(source_files) != {"A", "C", "E"}:
        raise ValueError("人工复核包必须同时提供 A、C、E。")
    payloads = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in source_files.items()
    }
    package = build_review_package(
        load_questions(args.questions),
        payloads,
        source_files,
    )
    write_json_atomic(args.output, package)
    print(f"人工引用复核包已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
