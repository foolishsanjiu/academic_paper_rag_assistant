"""Validate human review labels and calculate entailment metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
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
        uncertain_count = sum(value == "uncertain" for value in support_labels)
        score_counts = Counter(answer_scores)
        upper_covered_claims = sum(
            any(value is True or value == "uncertain" for value in labels)
            for labels in claims.values()
        )
        summary[label] = {
            "question_count": len(answer_scores),
            "mean_answer_score": sum(answer_scores) / len(answer_scores),
            "answer_score_distribution": {
                str(score): score_counts.get(score, 0) for score in (0, 1, 2)
            },
            "citation_pair_count": len(support_labels),
            "decided_citation_pair_count": len(decided),
            "supported_citation_pair_count": supported_count,
            "unsupported_citation_pair_count": sum(
                value is False for value in decided
            ),
            "uncertain_citation_pair_count": uncertain_count,
            "citation_correctness": (
                supported_count / len(decided) if decided else None
            ),
            "citation_correctness_lower_bound": (
                supported_count / len(support_labels)
                if support_labels
                else None
            ),
            "citation_correctness_upper_bound": (
                (supported_count + uncertain_count) / len(support_labels)
                if support_labels
                else None
            ),
            "claim_count": len(claims),
            "claim_citation_coverage": (
                sum(
                    any(value is True for value in labels)
                    for labels in claims.values()
                )
                / len(claims)
                if claims
                else None
            ),
            "claim_citation_coverage_upper_bound": (
                upper_covered_claims / len(claims) if claims else None
            ),
        }
    return summary


def _signature_id(question_id: str, claim: str, chunk_id: Any) -> str:
    normalized_claim = " ".join(claim.split())
    value = f"{question_id}\n{normalized_claim}\n{chunk_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _two_sided_sign_test(wins: int, losses: int) -> float:
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0
    tail = sum(
        math.comb(non_ties, value)
        for value in range(min(wins, losses) + 1)
    ) / (2**non_ties)
    return min(1.0, 2 * tail)


def audit_label_quality(package: dict[str, Any]) -> dict[str, Any]:
    """Audit label distributions and exact duplicate-pair consistency."""
    calculate_human_metrics(package)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    system_summary: dict[str, Any] = {}

    for label in package["review_protocol"]["systems"]:
        citation_counts: Counter[str] = Counter()
        answer_note_count = 0
        citation_note_count = 0
        for question in package["questions"]:
            system = question["systems"][label]
            answer_note_count += int(bool(system.get("answer_notes", "").strip()))
            for pair in system["citation_pairs"]:
                supported = pair["supported"]
                label_name = (
                    "true"
                    if supported is True
                    else "false"
                    if supported is False
                    else "uncertain"
                )
                citation_counts[label_name] += 1
                citation_note_count += int(
                    bool(pair.get("reviewer_notes", "").strip())
                )
                signature = _signature_id(
                    question["id"],
                    str(pair.get("claim", "")),
                    pair.get("chunk_id"),
                )
                groups[signature].append(
                    {
                        "question_id": question["id"],
                        "system": label,
                        "pair_id": pair.get("pair_id"),
                        "chunk_id": pair.get("chunk_id"),
                        "supported": supported,
                    }
                )
        total = sum(citation_counts.values())
        system_summary[label] = {
            "citation_label_distribution": {
                name: citation_counts.get(name, 0)
                for name in ("true", "false", "uncertain")
            },
            "uncertain_rate": (
                citation_counts.get("uncertain", 0) / total if total else None
            ),
            "answer_note_count": answer_note_count,
            "citation_note_count": citation_note_count,
        }

    repeated = {key: rows for key, rows in groups.items() if len(rows) > 1}
    mixed = {
        key: rows
        for key, rows in repeated.items()
        if len({row["supported"] for row in rows}) > 1
    }
    conflicts = {
        key: rows
        for key, rows in repeated.items()
        if {True, False}.issubset({row["supported"] for row in rows})
    }

    comparisons: dict[str, Any] = {}
    for left, right in (("C", "A"), ("E", "A"), ("E", "C")):
        differences = [
            question["systems"][left]["answer_score"]
            - question["systems"][right]["answer_score"]
            for question in package["questions"]
        ]
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        comparisons[f"{left}_vs_{right}"] = {
            "mean_score_difference": sum(differences) / len(differences),
            "wins": wins,
            "ties": sum(value == 0 for value in differences),
            "losses": losses,
            "two_sided_sign_test_p_value": _two_sided_sign_test(wins, losses),
        }

    return {
        "system_summary": system_summary,
        "exact_duplicate_group_count": len(repeated),
        "exact_duplicate_pair_count": sum(len(rows) for rows in repeated.values()),
        "mixed_label_duplicate_group_count": len(mixed),
        "decided_conflict_group_count": len(conflicts),
        "decided_conflicts": [
            {"signature_id": key, "occurrences": rows}
            for key, rows in sorted(conflicts.items())
        ],
        "pairwise_answer_score_comparisons": comparisons,
    }


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
        "quality_audit": audit_label_quality(package),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"人工引用指标已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
