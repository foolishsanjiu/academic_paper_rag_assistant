"""Evaluate intent routing against the curated router dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_router import route_query  # noqa: E402
from config import get_settings  # noqa: E402
from llm_client import LLMClient  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "tests" / "router_cases.json"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def evaluate_case(case: dict[str, Any], llm: LLMClient | None) -> dict[str, Any]:
    decision = route_query(case["query"], llm)
    predicted = {
        "intent": decision.intent.value,
        "strategy": (
            decision.retrieval_strategy.value
            if decision.retrieval_strategy
            else None
        ),
        "action": (decision.tool_args or {}).get("action"),
        "route_source": decision.route_source,
    }
    expected_fields = ["intent"]
    expected_fields.extend(
        field for field in ("strategy", "action") if field in case
    )
    correct = all(predicted[field] == case[field] for field in expected_fields)
    return {"query": case["query"], "expected": case, "predicted": predicted, "correct": correct}


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    *,
    use_llm: bool,
) -> dict[str, Any]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    llm = LLMClient(get_settings()) if use_llm else None
    results = [evaluate_case(case, llm) for case in cases]
    correct_count = sum(item["correct"] for item in results)
    sources = Counter(item["predicted"]["route_source"] for item in results)
    by_intent: dict[str, dict[str, int]] = {}
    for intent in sorted({case["intent"] for case in cases}):
        matching = [item for item in results if item["expected"]["intent"] == intent]
        by_intent[intent] = {
            "correct": sum(item["correct"] for item in matching),
            "total": len(matching),
        }

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": "llm_with_rule_fallback" if use_llm else "rule_fallback",
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "summary": {
            "correct": correct_count,
            "total": len(results),
            "accuracy": correct_count / len(results),
            "route_sources": dict(sources),
            "by_intent": by_intent,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or DEFAULT_RESULTS_DIRECTORY / (
        "router_llm.json" if args.use_llm else "router_rules.json"
    )
    report = run_evaluation(args.dataset, output, use_llm=args.use_llm)
    summary = report["summary"]
    print(
        f"Router accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.1%})"
    )
    print(f"Route sources: {summary['route_sources']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
