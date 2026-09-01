"""Validate human review labels and calculate entailment metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import write_json_atomic  # noqa: E402
from evaluation.evaluate_generation import file_sha256  # noqa: E402


def calculate_human_metrics(
    package: dict[str, Any],
) -> dict[str, Any]:
    """Validate complete labels and summarize each retrieval configuration."""
    questions = package.get("questions")
    if not isinstance(questions, list) or len(questions) != 30:
        raise ValueError("人工标签必须包含固定的 30 道题。")
    systems = package.get("review_protocol", {}).get("systems")
    if systems != ["A", "C", "E"]:
        raise ValueError("人工标签必须同时包含 A、C、E。")

    summary: dict[str, Any] = {}
    for label in systems:
        answer_scores: list[int] = []
        support_labels: list[bool | str] = []
        claims: dict[tuple[str, str], list[bool | str]] = {}
        for question in questions:
            system = question.get("systems", {}).get(label)
            if not isinstance(system, dict):
                raise ValueError(f"{question.get('id')} 缺少系统 {label}。")
            score = system.get("answer_score")
            if isinstance(score, bool) or score not in {0, 1, 2}:
                raise ValueError(
                    f"{question['id']} {label} 的 answer_score 尚未完成。"
                )
            answer_scores.append(score)
            for pair in system.get("citation_pairs", []):
                supported = pair.get("supported")
                if not (
                    isinstance(supported, bool)
                    or supported == "uncertain"
                ):
                    raise ValueError(
                        f"{question['id']} {label} {pair.get('pair_id')} "
                        "的 supported 尚未完成。"
                    )
                support_labels.append(supported)
                claim_key = (question["id"], str(pair.get("claim", "")).strip())
                claims.setdefault(claim_key, []).append(supported)

        decided = [value for value in support_labels if isinstance(value, bool)]
        supported_count = sum(value is True for value in decided)
        score_counts = Counter(answer_scores)
        summary[label] = {
            "question_count": len(answer_scores),
            "mean_answer_score": sum(answer_scores) / len(answer_scores),
            "answer_score_distribution": {
                str(score): score_counts.get(score, 0) for score in (0, 1, 2)
            },
            "citation_pair_count": len(support_labels),
            "decided_citation_pair_count": len(decided),
            "uncertain_citation_pair_count": sum(
                value == "uncertain" for value in support_labels
            ),
            "citation_correctness": (
                supported_count / len(decided) if decided else None
            ),
            "claim_count": len(claims),
            "claim_citation_coverage": (
                sum(any(value is True for value in labels) for labels in claims.values())
                / len(claims)
                if claims
                else None
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate M7 human labels and summarize citation entailment."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = json.loads(args.labels.read_text(encoding="utf-8"))
    output = {
        "labels_path": str(args.labels.resolve()),
        "labels_sha256": file_sha256(args.labels),
        "summary": calculate_human_metrics(package),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"人工引用指标已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
