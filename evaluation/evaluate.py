"""Run a reproducible baseline evaluation for the academic-paper RAG."""

from __future__ import annotations

import argparse
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
DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "results" / "baseline.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the evaluation dataset."""
    questions = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(questions, list) or not questions:
        raise ValueError("评测集必须是非空 JSON 数组。")

    required_fields = {
        "id",
        "type",
        "question",
        "expected_answer",
        "source_files",
        "source_pages",
    }
    seen_ids: set[str] = set()

    for item in questions:
        missing = required_fields - set(item)
        if missing:
            raise ValueError(
                f"问题缺少字段：{item.get('id', '<unknown>')} -> "
                f"{sorted(missing)}"
            )

        question_id = str(item["id"])
        if question_id in seen_ids:
            raise ValueError(f"问题 ID 重复：{question_id}")
        seen_ids.add(question_id)

        if len(item["source_files"]) != len(item["source_pages"]):
            raise ValueError(
                f"来源文件与页码数量不一致：{question_id}"
            )

    return questions


def calculate_retrieval_metrics(
    expected_files: list[str],
    expected_pages: list[int],
    sources: list[dict[str, Any]],
) -> dict[str, bool | None]:
    """Calculate file and exact file-page retrieval hits."""
    if not expected_files:
        return {
            "retrieval_any_hit": None,
            "retrieval_all_hit": None,
            "page_any_hit": None,
            "page_all_hit": None,
        }

    retrieved_files = {
        str(source["file_name"])
        for source in sources
    }
    expected_file_set = set(expected_files)

    retrieved_pairs = {
        (
            str(source["file_name"]),
            str(source["page_number"]),
        )
        for source in sources
    }
    expected_pairs = {
        (file_name, str(page_number))
        for file_name, page_number in zip(
            expected_files,
            expected_pages,
        )
    }

    return {
        "retrieval_any_hit": bool(
            expected_file_set & retrieved_files
        ),
        "retrieval_all_hit": expected_file_set <= retrieved_files,
        "page_any_hit": bool(expected_pairs & retrieved_pairs),
        "page_all_hit": expected_pairs <= retrieved_pairs,
    }


def calculate_refusal_metrics(
    question_type: str,
    answer: str,
    refusal_message: str,
) -> dict[str, bool]:
    """Evaluate whether the answer refused when it should have."""
    expected_refusal = question_type == "no_answer"
    refused = refusal_message in answer

    return {
        "expected_refusal": expected_refusal,
        "refused": refused,
        "refusal_correct": refused == expected_refusal,
    }


def mean_boolean(
    results: list[dict[str, Any]],
    field: str,
) -> float | None:
    """Return a rate while ignoring values that are not booleans."""
    values = [
        item["metrics"][field]
        for item in results
        if isinstance(item.get("metrics", {}).get(field), bool)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate baseline metrics."""
    completed = [item for item in results if item["error"] is None]
    latencies = [item["latency_seconds"] for item in completed]

    return {
        "question_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "retrieval_any_hit_rate": mean_boolean(
            completed,
            "retrieval_any_hit",
        ),
        "retrieval_all_hit_rate": mean_boolean(
            completed,
            "retrieval_all_hit",
        ),
        "page_any_hit_rate": mean_boolean(
            completed,
            "page_any_hit",
        ),
        "page_all_hit_rate": mean_boolean(
            completed,
            "page_all_hit",
        ),
        "refusal_correct_rate": mean_boolean(
            completed,
            "refusal_correct",
        ),
        "average_latency_seconds": (
            sum(latencies) / len(latencies)
            if latencies
            else None
        ),
        "answer_score_scale": {
            "0": "incorrect",
            "1": "partially correct",
            "2": "mostly or fully correct",
        },
        "answer_scores_pending_manual_review": sum(
            item["answer_score"] is None
            for item in completed
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint without leaving a partial JSON result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def run_evaluation(
    questions_path: Path,
    output_path: Path,
    top_k: int,
    temperature: float,
    limit: int | None,
    candidate_k: int | None = None,
    max_chunks_per_file: int | None = None,
) -> dict[str, Any]:
    """Load project resources and evaluate every selected question."""
    from config import get_settings
    from index_manifest import (
        load_index_manifest,
        validate_index_manifest,
    )
    from llm_client import LLMClient
    from rag_chain import NO_ANSWER_MESSAGE, RAGChain
    from vector_store import (
        DEFAULT_COLLECTION_NAME,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_PERSIST_DIRECTORY,
        create_embedding_model,
        get_vector_count,
        load_vector_store,
    )

    questions = load_questions(questions_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0。")
        questions = questions[:limit]

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

    llm = LLMClient(get_settings())
    rag = RAGChain(
        llm=llm,
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=candidate_k,
        max_chunks_per_file=max_chunks_per_file,
    )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "run": {
            "started_at": started_at,
            "completed_at": None,
            "questions_path": str(questions_path.resolve()),
            "parameters": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "max_chunks_per_file": max_chunks_per_file,
                "temperature": temperature,
                "chunk_size": manifest["chunk_size"],
                "chunk_overlap": manifest["chunk_overlap"],
                "embedding_model": manifest["embedding_model"],
                "collection_name": DEFAULT_COLLECTION_NAME,
                "persist_directory": str(
                    DEFAULT_PERSIST_DIRECTORY.resolve()
                ),
                "vector_count": vector_count,
            },
            "index_manifest": manifest,
        },
        "summary": {},
        "results": results,
    }

    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['id']}: {item['question']}")
        start = time.perf_counter()
        result: dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_source_files": item["source_files"],
            "expected_source_pages": item["source_pages"],
            "answer": None,
            "retrieval_query": None,
            "retrieved_sources": [],
            "metrics": {},
            "answer_score": None,
            "latency_seconds": None,
            "error": None,
        }

        try:
            response = rag.ask(
                question=item["question"],
                temperature=temperature,
            )
            sources = [asdict(source) for source in response.sources]
            result["answer"] = response.answer
            result["retrieval_query"] = response.retrieval_query
            result["retrieved_sources"] = sources
            result["metrics"] = {
                **calculate_retrieval_metrics(
                    item["source_files"],
                    item["source_pages"],
                    sources,
                ),
                **calculate_refusal_metrics(
                    item["type"],
                    response.answer,
                    NO_ANSWER_MESSAGE,
                ),
            }
        except Exception as error:
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

        result["latency_seconds"] = round(
            time.perf_counter() - start,
            4,
        )
        results.append(result)
        payload["summary"] = build_summary(results)
        write_json_atomic(output_path, payload)

    payload["run"]["completed_at"] = (
        datetime.now().astimezone().isoformat(timespec="seconds")
    )
    payload["summary"] = build_summary(results)
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Academic Paper RAG baseline.",
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--max-chunks-per-file", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        top_k=args.top_k,
        temperature=args.temperature,
        limit=args.limit,
        candidate_k=args.candidate_k,
        max_chunks_per_file=args.max_chunks_per_file,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"评测结果已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
