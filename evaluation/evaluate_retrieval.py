"""Compare retrieval parameters without making LLM API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import (  # noqa: E402
    calculate_retrieval_metrics,
    load_questions,
    write_json_atomic,
)
from index_manifest import (  # noqa: E402
    load_index_manifest,
    validate_index_manifest,
)
from rag_chain import build_sources  # noqa: E402
from retriever import retrieve_with_scores  # noqa: E402
from vector_store import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    create_embedding_model,
    get_vector_count,
    load_vector_store,
)


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_PATH = (
    Path(__file__).parent
    / "results"
    / "retrieval_top_k.json"
)


def rate(results: list[dict[str, Any]], field: str) -> float | None:
    values = [
        item["metrics"][field]
        for item in results
        if isinstance(item["metrics"].get(field), bool)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval coverage and source concentration."""
    return {
        "question_count": len(results),
        "scored_question_count": sum(
            bool(item["expected_source_files"])
            for item in results
        ),
        "retrieval_any_hit_rate": rate(results, "retrieval_any_hit"),
        "retrieval_all_hit_rate": rate(results, "retrieval_all_hit"),
        "page_any_hit_rate": rate(results, "page_any_hit"),
        "page_all_hit_rate": rate(results, "page_all_hit"),
        "average_unique_source_count": (
            sum(item["unique_source_count"] for item in results)
            / len(results)
        ),
        "average_largest_source_share": (
            sum(item["largest_source_share"] for item in results)
            / len(results)
        ),
        "average_latency_seconds": (
            sum(item["latency_seconds"] for item in results)
            / len(results)
        ),
    }


def run_retrieval_experiment(
    questions_path: Path,
    output_path: Path,
    top_k_values: list[int],
) -> dict[str, Any]:
    """Evaluate several Top-k values with one loaded embedding model."""
    if not top_k_values or any(value <= 0 for value in top_k_values):
        raise ValueError("所有 Top-k 值都必须大于 0。")

    questions = load_questions(questions_path)
    manifest = load_index_manifest()
    if manifest is None:
        raise FileNotFoundError("缺少 chroma_db/index_manifest.json。")

    embeddings = create_embedding_model(DEFAULT_EMBEDDING_MODEL)
    vector_store = load_vector_store(embeddings)
    vector_count = get_vector_count(vector_store)
    validate_index_manifest(
        vector_count=vector_count,
        manifest=manifest,
    )

    runs: list[dict[str, Any]] = []
    payload = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "questions_path": str(questions_path.resolve()),
        "index_manifest": manifest,
        "runs": runs,
    }

    for top_k in top_k_values:
        print(f"Evaluating Top-k={top_k}")
        results: list[dict[str, Any]] = []

        for index, item in enumerate(questions, start=1):
            start = time.perf_counter()
            retrieved = retrieve_with_scores(
                vector_store=vector_store,
                query=item["question"],
                top_k=top_k,
            )
            sources = [
                asdict(source)
                for source in build_sources(retrieved)
            ]
            file_counts = Counter(
                source["file_name"]
                for source in sources
            )
            unique_source_count = len(file_counts)
            largest_source_share = (
                max(file_counts.values()) / len(sources)
                if sources
                else 0.0
            )

            results.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item["question"],
                    "expected_source_files": item["source_files"],
                    "expected_source_pages": item["source_pages"],
                    "retrieved_sources": sources,
                    "unique_source_count": unique_source_count,
                    "largest_source_share": largest_source_share,
                    "metrics": calculate_retrieval_metrics(
                        item["source_files"],
                        item["source_pages"],
                        sources,
                    ),
                    "latency_seconds": round(
                        time.perf_counter() - start,
                        4,
                    ),
                }
            )
            print(f"  [{index}/{len(questions)}] {item['id']}")

        run = {
            "top_k": top_k,
            "summary": summarize(results),
            "results": results,
        }
        runs.append(run)
        write_json_atomic(output_path, payload)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Top-k retrieval without LLM generation.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 8, 10],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_retrieval_experiment(
        questions_path=args.questions,
        output_path=args.output,
        top_k_values=args.top_k,
    )
    for run in payload["runs"]:
        print(
            json.dumps(
                {
                    "top_k": run["top_k"],
                    **run["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"检索实验结果已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
