"""Generate answers from frozen retrieval results and score citations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings  # noqa: E402
from evaluation.citation_metrics import calculate_citation_metrics  # noqa: E402
from evaluation.dataset import load_qrels, load_questions, validate_qrels  # noqa: E402
from evaluation.evaluate import (  # noqa: E402
    calculate_refusal_metrics,
    write_json_atomic,
)
from llm_client import LLMClient  # noqa: E402
from rag_chain import NO_ANSWER_MESSAGE, build_rag_prompt  # noqa: E402


ALLOWED_RETRIEVAL_METHODS = {"dense", "hybrid", "hybrid_rerank"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_context_from_sources(sources: list[dict[str, Any]]) -> str:
    """Build the same numbered context format used by ``RAGChain``."""
    parts: list[str] = []
    for position, source in enumerate(sources, start=1):
        rank = int(source.get("rank", position))
        parts.append(
            "\n".join(
                [
                    f"[Source {rank}]",
                    f"File: {source.get('file_name', 'unknown.pdf')}",
                    f"PDF Page: {source.get('page_number', 'unknown')}",
                    f"Chunk ID: {source.get('chunk_id', 'unknown')}",
                    "",
                    str(source.get("text", "")),
                ]
            )
        )
    return "\n\n".join(parts)


def estimate_prompt_characters(
    questions: list[dict[str, Any]],
    retrieval_run: dict[str, Any],
) -> dict[str, float | int]:
    """Estimate paid input size without calling an API."""
    questions_by_id = {item["id"]: item for item in questions}
    lengths = [
        len(
            build_rag_prompt(
                question=questions_by_id[item["id"]]["question"],
                context=build_context_from_sources(item["retrieved_sources"]),
            )
        )
        for item in retrieval_run["results"]
    ]
    return {
        "prompt_count": len(lengths),
        "total_characters": sum(lengths),
        "average_characters": sum(lengths) / len(lengths),
        "minimum_characters": min(lengths),
        "maximum_characters": max(lengths),
    }


def validate_retrieval_payload(
    payload: dict[str, Any],
    questions: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Validate that one frozen retrieval run exactly matches the split."""
    if payload.get("split") != split:
        raise ValueError(
            f"检索结果 split 不匹配：expected={split}, actual={payload.get('split')}"
        )
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError("端到端评测要求检索结果恰好包含一个 run。")
    run = runs[0]
    if run.get("method") not in ALLOWED_RETRIEVAL_METHODS:
        raise ValueError("端到端评测只允许冻结配置 A、C、E。")
    results = run.get("results")
    if not isinstance(results, list):
        raise ValueError("检索结果缺少 results。")
    expected_ids = [item["id"] for item in questions]
    actual_ids = [item.get("id") for item in results]
    if actual_ids != expected_ids:
        raise ValueError("检索结果题目 ID 或顺序与冻结问题集不一致。")
    return run


def _mean_boolean(results: list[dict[str, Any]], field: str) -> float | None:
    values = [
        item["metrics"][field]
        for item in results
        if isinstance(item.get("metrics", {}).get(field), bool)
    ]
    return sum(values) / len(values) if values else None


def _mean_number(results: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(item["metrics"][field])
        for item in results
        if (
            not isinstance(item.get("metrics", {}).get(field), bool)
            and isinstance(item.get("metrics", {}).get(field), (int, float))
        )
    ]
    return sum(values) / len(values) if values else None


def build_generation_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate generation, refusal, and automatic citation metrics."""
    completed = [item for item in results if item.get("error") is None]
    latencies = [float(item["latency_seconds"]) for item in completed]
    return {
        "question_count": len(results),
        "completed_count": len(completed),
        "error_count": len(results) - len(completed),
        "refusal_correct_rate": _mean_boolean(completed, "refusal_correct"),
        "citation_format_validity_rate": _mean_boolean(
            completed, "citation_format_validity"
        ),
        "citation_index_validity_rate": _mean_boolean(
            completed, "citation_index_validity"
        ),
        "answer_has_citation_rate": _mean_boolean(
            completed, "answer_has_citation"
        ),
        "no_answer_has_no_spurious_citation_rate": _mean_boolean(
            completed, "no_answer_has_no_spurious_citation"
        ),
        "citation_qrel_precision": _mean_number(
            completed, "citation_qrel_precision"
        ),
        "citation_qrel_recall": _mean_number(
            completed, "citation_qrel_recall"
        ),
        "average_generation_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "answer_scores_pending_manual_review": sum(
            item.get("answer_score") is None for item in completed
        ),
    }


def run_generation_evaluation(
    *,
    questions_path: Path,
    qrels_path: Path,
    retrieval_results_path: Path,
    output_path: Path,
    split: str = "test",
    temperature: float = 0.2,
    max_tokens: int = 800,
    thinking_mode: str = "disabled",
    allow_paid_api: bool = False,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Run one paid generation configuration with atomic checkpoints."""
    if not allow_paid_api:
        raise PermissionError(
            "端到端生成会调用付费 API；必须显式传入 --allow-paid-api。"
        )
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0。")
    if thinking_mode not in {"disabled", "enabled"}:
        raise ValueError("thinking_mode 只能是 disabled 或 enabled。")

    all_questions = load_questions(questions_path)
    qrels = load_qrels(qrels_path)
    validate_qrels(all_questions, qrels, require_complete=True)
    questions = [item for item in all_questions if item["split"] == split]
    retrieval_payload = json.loads(
        retrieval_results_path.read_text(encoding="utf-8")
    )
    retrieval_run = validate_retrieval_payload(
        retrieval_payload,
        questions,
        split,
    )

    settings = get_settings()
    llm = llm or LLMClient(settings)
    retrieval_hash = file_sha256(retrieval_results_path)
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "run": {
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "completed_at": None,
            "split": split,
            "questions_path": str(questions_path.resolve()),
            "questions_sha256": file_sha256(questions_path),
            "qrels_path": str(qrels_path.resolve()),
            "qrels_sha256": file_sha256(qrels_path),
            "retrieval_results_path": str(retrieval_results_path.resolve()),
            "retrieval_results_sha256": retrieval_hash,
            "retrieval_method": retrieval_run["method"],
            "retrieval_parameters": {
                key: retrieval_run.get(key)
                for key in (
                    "top_k",
                    "candidate_k",
                    "rrf_k",
                    "fusion_k",
                    "reranker_candidate_k",
                )
            },
            "model": settings.model,
            "base_url": settings.base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking_mode": thinking_mode,
            "attempted_api_calls": 0,
        },
        "summary": {},
        "results": results,
    }

    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_run = existing.get("run", {})
        if (
            existing_run.get("retrieval_results_sha256") != retrieval_hash
            or existing_run.get("model") != settings.model
            or existing_run.get("temperature") != temperature
            or existing_run.get("max_tokens") != max_tokens
            or existing_run.get("thinking_mode") != thinking_mode
        ):
            raise ValueError("已有生成结果与本次冻结输入或模型参数不一致。")
        if existing_run.get("completed_at"):
            return existing
        payload = existing
        results = payload["results"]

    completed_ids = {item["id"] for item in results}
    retrieved_by_id = {
        item["id"]: item for item in retrieval_run["results"]
    }
    for index, question in enumerate(questions, start=1):
        if question["id"] in completed_ids:
            continue
        print(f"[{index}/{len(questions)}] {question['id']}", flush=True)
        retrieved = retrieved_by_id[question["id"]]
        sources = retrieved["retrieved_sources"]
        prompt = build_rag_prompt(
            question=question["question"],
            context=build_context_from_sources(sources),
        )
        started = time.perf_counter()
        item: dict[str, Any] = {
            "id": question["id"],
            "type": question["type"],
            "language": question["language"],
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "answer": None,
            "retrieved_sources": sources,
            "metrics": {},
            "answer_score": None,
            "latency_seconds": None,
            "error": None,
        }
        payload["run"]["attempted_api_calls"] += 1
        try:
            answer = llm.chat(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking=thinking_mode == "enabled",
            )
            item["answer"] = answer
            item["metrics"] = {
                **calculate_refusal_metrics(
                    question["type"], answer, NO_ANSWER_MESSAGE
                ),
                **calculate_citation_metrics(
                    answer,
                    sources,
                    qrels[question["id"]],
                    question["type"],
                ),
            }
        except Exception as error:
            item["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        item["latency_seconds"] = round(time.perf_counter() - started, 4)
        results.append(item)
        payload["summary"] = build_generation_summary(results)
        write_json_atomic(output_path, payload)

    payload["run"]["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    payload["summary"] = build_generation_summary(results)
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answers from one frozen A/C/E retrieval result."
    )
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.json"))
    parser.add_argument("--qrels", type=Path, default=Path(__file__).with_name("qrels.json"))
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["legacy", "dev", "test"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--thinking-mode",
        choices=["disabled", "enabled"],
        default="disabled",
    )
    parser.add_argument("--allow-paid-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_generation_evaluation(
        questions_path=args.questions,
        qrels_path=args.qrels,
        retrieval_results_path=args.retrieval_results,
        output_path=args.output,
        split=args.split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode,
        allow_paid_api=args.allow_paid_api,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"生成评测结果已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
