"""Compare retrieval parameters without making LLM API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from itertools import product
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CHROMA_DIRECTORY,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_HYBRID_FUSION_K,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RRF_K,
)
from evaluation.evaluate import (  # noqa: E402
    calculate_retrieval_metrics,
    load_questions,
    write_json_atomic,
)
from evaluation.dataset import load_qrels, validate_qrels  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    calculate_ranked_retrieval_metrics,
    mean_ranked_metrics,
)
from index_manifest import (  # noqa: E402
    load_index_manifest,
    validate_index_manifest,
)
from hybrid_retriever import FusedCandidate, HybridRetriever  # noqa: E402
from rag_chain import build_sources  # noqa: E402
from reranker import (  # noqa: E402
    RerankedCandidate,
    TransformersCrossEncoderReranker,
    dense_rerank_candidates,
    hybrid_rerank_candidates,
    rerank_candidates,
)
from retriever import retrieve_with_scores  # noqa: E402
from sparse_retriever import build_sparse_retriever  # noqa: E402
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
DEFAULT_BM25_OUTPUT_PATH = (
    Path(__file__).parent
    / "results"
    / "retrieval_bm25.json"
)
DEFAULT_HYBRID_OUTPUT_PATH = (
    Path(__file__).parent
    / "results"
    / "retrieval_hybrid.json"
)
DEFAULT_DENSE_RERANK_OUTPUT_PATH = (
    Path(__file__).parent / "results" / "retrieval_dense_rerank.json"
)
DEFAULT_HYBRID_RERANK_OUTPUT_PATH = (
    Path(__file__).parent / "results" / "retrieval_hybrid_rerank.json"
)
RETRIEVAL_METHODS = {
    "dense",
    "bm25",
    "hybrid",
    "dense_rerank",
    "hybrid_rerank",
}


def build_sparse_sources(
    results: list[tuple[Any, float]],
) -> list[dict[str, Any]]:
    """Serialize BM25 results without calling the score cosine similarity."""
    sources: list[dict[str, Any]] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        sources.append(
            {
                "rank": rank,
                "file_name": str(metadata.get("file_name", "unknown.pdf")),
                "document_id": str(metadata.get("document_id", "unknown")),
                "document_type": str(metadata.get("document_type", "unknown")),
                "page_number": metadata.get("page_number", "unknown"),
                "chunk_id": str(metadata.get("chunk_id", "unknown")),
                "similarity": None,
                "bm25_score": score,
                "text": document.page_content,
            }
        )
    return sources


def build_hybrid_sources(
    candidates: list[FusedCandidate],
) -> list[dict[str, Any]]:
    """Serialize fused candidates with branch and RRF diagnostics."""
    sources: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate.document.metadata
        sources.append(
            {
                "rank": rank,
                "file_name": str(metadata.get("file_name", "unknown.pdf")),
                "document_id": str(metadata.get("document_id", "unknown")),
                "document_type": str(metadata.get("document_type", "unknown")),
                "page_number": metadata.get("page_number", "unknown"),
                "chunk_id": candidate.chunk_id,
                "similarity": (
                    1.0 - candidate.dense_distance
                    if candidate.dense_distance is not None
                    else None
                ),
                "bm25_score": candidate.bm25_score,
                "rrf_score": candidate.rrf_score,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "text": candidate.document.page_content,
            }
        )
    return sources


def build_reranked_sources(
    candidates: list[RerankedCandidate],
) -> list[dict[str, Any]]:
    """Serialize final reranked candidates with retrieval diagnostics."""
    sources: list[dict[str, Any]] = []
    for rank, item in enumerate(candidates, start=1):
        candidate = item.candidate
        metadata = candidate.document.metadata
        sources.append(
            {
                "rank": rank,
                "file_name": str(metadata.get("file_name", "unknown.pdf")),
                "document_id": str(metadata.get("document_id", "unknown")),
                "document_type": str(metadata.get("document_type", "unknown")),
                "page_number": metadata.get("page_number", "unknown"),
                "chunk_id": candidate.chunk_id,
                "similarity": (
                    1.0 - candidate.dense_distance
                    if candidate.dense_distance is not None
                    else None
                ),
                "bm25_score": candidate.bm25_score,
                "rrf_score": candidate.rrf_score,
                "rerank_score": item.rerank_score,
                "retrieval_rank": candidate.retrieval_rank,
                "dense_rank": candidate.dense_rank,
                "sparse_rank": candidate.sparse_rank,
                "text": candidate.document.page_content,
            }
        )
    return sources


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
        **mean_ranked_metrics(
            item.get("metrics", {})
            for item in results
        ),
    }


def run_retrieval_experiment(
    questions_path: Path,
    output_path: Path,
    top_k_values: list[int],
    candidate_multiplier: int = 4,
    max_chunks_per_file: int | None = None,
    qrels_path: Path | None = None,
    method: str = "dense",
    hybrid_candidate_k_values: list[int] | None = None,
    rrf_k_values: list[int] | None = None,
    fusion_k: int = DEFAULT_HYBRID_FUSION_K,
    split: str | None = None,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    reranker_candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
    reranker_device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate retrieval and optional reranking without LLM calls."""
    if not top_k_values or any(value <= 0 for value in top_k_values):
        raise ValueError("所有 Top-k 值都必须大于 0。")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier 必须大于 0。")
    if method not in RETRIEVAL_METHODS:
        raise ValueError(
            f"不支持的检索方法：{method}。可选值：{sorted(RETRIEVAL_METHODS)}"
        )
    if method == "bm25" and max_chunks_per_file is not None:
        raise ValueError("M3 的 BM25-only 路径暂不应用来源多样化。")
    if split is not None and split not in {"legacy", "dev", "test"}:
        raise ValueError(f"不支持的 split：{split}")
    validate_values = hybrid_candidate_k_values or [DEFAULT_HYBRID_CANDIDATE_K]
    if any(value <= 0 for value in validate_values):
        raise ValueError("所有 Hybrid candidate_k 都必须大于 0。")
    validate_rrf_values = rrf_k_values or [DEFAULT_RRF_K]
    if any(value <= 0 for value in validate_rrf_values):
        raise ValueError("所有 rrf_k 都必须大于 0。")
    if fusion_k <= 0:
        raise ValueError("fusion_k 必须大于 0。")
    if (
        method in {"hybrid", "hybrid_rerank"}
        and fusion_k < max(top_k_values)
    ):
        raise ValueError("fusion_k 不能小于最大 Top-k。")
    if reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size 必须大于 0。")
    if reranker_candidate_k <= 0:
        raise ValueError("reranker_candidate_k 必须大于 0。")
    if method in {"dense_rerank", "hybrid_rerank"}:
        if not reranker_model.strip():
            raise ValueError("Reranker 模型路径不能为空。")
        if reranker_candidate_k < max(top_k_values):
            raise ValueError("reranker_candidate_k 不能小于最大 Top-k。")

    questions = load_questions(questions_path)
    qrels = load_qrels(qrels_path) if qrels_path is not None else None
    if qrels is not None:
        validate_qrels(questions, qrels, require_complete=True)
    if split is not None:
        questions = [item for item in questions if item["split"] == split]
        if not questions:
            raise ValueError(f"split={split} 没有评测题。")
    manifest = load_index_manifest()
    if manifest is None:
        raise FileNotFoundError("缺少 chroma_db/index_manifest.json。")

    vector_store = None
    sparse_retriever = None
    hybrid_retriever = None
    cross_encoder = None
    if method in {"dense", "dense_rerank"}:
        embeddings = create_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector_store = load_vector_store(embeddings)
        vector_count = get_vector_count(vector_store)
        retrieval_identity: dict[str, Any] = {
            "method": method,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        }
    elif method == "bm25":
        from langchain_chroma import Chroma

        metadata_store = Chroma(
            collection_name=DEFAULT_COLLECTION_NAME,
            persist_directory=str(CHROMA_DIRECTORY),
            embedding_function=None,
        )
        sparse_retriever = build_sparse_retriever(metadata_store)
        vector_count = sparse_retriever.document_count
        retrieval_identity = sparse_retriever.identity()
    else:
        embeddings = create_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector_store = load_vector_store(embeddings)
        vector_count = get_vector_count(vector_store)
        sparse_retriever = build_sparse_retriever(vector_store)
        if sparse_retriever.document_count != vector_count:
            raise ValueError(
                "Dense 与 BM25 的 Chunk 数量不一致："
                f"dense={vector_count}, sparse={sparse_retriever.document_count}"
            )
        hybrid_retriever = HybridRetriever(vector_store, sparse_retriever)
        retrieval_identity = {
            "method": (
                "hybrid_rrf"
                if method == "hybrid"
                else "hybrid_rrf_rerank"
            ),
            "dense": {"embedding_model": DEFAULT_EMBEDDING_MODEL},
            "sparse": sparse_retriever.identity(),
        }
    validate_index_manifest(
        vector_count=vector_count,
        manifest=manifest,
    )
    if method in {"dense_rerank", "hybrid_rerank"}:
        cross_encoder = TransformersCrossEncoderReranker(
            reranker_model,
            batch_size=reranker_batch_size,
            device=reranker_device,
            local_files_only=True,
        )
        retrieval_identity["reranker"] = {
            "model": reranker_model,
            "batch_size": reranker_batch_size,
            "candidate_k": reranker_candidate_k,
            "device": reranker_device,
        }

    runs: list[dict[str, Any]] = []
    payload = {
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "questions_path": str(questions_path.resolve()),
        "qrels_path": (
            str(qrels_path.resolve())
            if qrels_path is not None
            else None
        ),
        "split": split,
        "index_manifest": manifest,
        "retrieval": retrieval_identity,
        "runs": runs,
    }

    configurations: list[tuple[int | None, int | None]] = (
        list(product(validate_values, validate_rrf_values))
        if method in {"hybrid", "hybrid_rerank"}
        else [(None, None)]
    )
    for resolved_candidate_k, resolved_rrf_k in configurations:
        for top_k in top_k_values:
            print(
                f"Evaluating method={method}, Top-k={top_k}, "
                f"candidate_k={resolved_candidate_k}, rrf_k={resolved_rrf_k}"
            )
            results: list[dict[str, Any]] = []

            for index, item in enumerate(questions, start=1):
                start = time.perf_counter()
                latency_breakdown = None
                if method == "dense":
                    if vector_store is None:
                        raise RuntimeError("Dense vector store 未初始化。")
                    retrieved = retrieve_with_scores(
                        vector_store=vector_store,
                        query=item["question"],
                        top_k=top_k,
                        candidate_k=(
                            top_k * candidate_multiplier
                            if max_chunks_per_file is not None
                            else None
                        ),
                        max_chunks_per_file=max_chunks_per_file,
                    )
                    sources = [
                        asdict(source)
                        for source in build_sources(retrieved)
                    ]
                elif method == "bm25":
                    if sparse_retriever is None:
                        raise RuntimeError("BM25 retriever 未初始化。")
                    retrieved = sparse_retriever.search(
                        item["question"],
                        top_k=top_k,
                    )
                    sources = build_sparse_sources(retrieved)
                elif method == "dense_rerank":
                    if vector_store is None or cross_encoder is None:
                        raise RuntimeError("Dense Rerank 资源未初始化。")
                    dense_start = time.perf_counter()
                    retrieved = retrieve_with_scores(
                        vector_store=vector_store,
                        query=item["question"],
                        top_k=reranker_candidate_k,
                    )
                    dense_latency = time.perf_counter() - dense_start
                    rerank_result = rerank_candidates(
                        item["question"],
                        dense_rerank_candidates(retrieved),
                        cross_encoder,
                        top_k=top_k,
                        max_chunks_per_file=max_chunks_per_file,
                    )
                    sources = build_reranked_sources(
                        rerank_result.candidates
                    )
                    latency_breakdown = {
                        "dense": round(dense_latency, 6),
                        "rerank": round(rerank_result.latency_seconds, 6),
                        "retrieval_total": round(
                            dense_latency + rerank_result.latency_seconds,
                            6,
                        ),
                    }
                elif method == "hybrid":
                    if hybrid_retriever is None:
                        raise RuntimeError("Hybrid retriever 未初始化。")
                    if resolved_candidate_k is None or resolved_rrf_k is None:
                        raise RuntimeError("Hybrid 参数未初始化。")
                    hybrid_result = hybrid_retriever.search(
                        item["question"],
                        top_k=top_k,
                        candidate_k=resolved_candidate_k,
                        rrf_k=resolved_rrf_k,
                        fusion_k=fusion_k,
                        max_chunks_per_file=max_chunks_per_file,
                    )
                    sources = build_hybrid_sources(hybrid_result.candidates)
                    latency_breakdown = {
                        "dense": round(
                            hybrid_result.dense_latency_seconds, 6
                        ),
                        "bm25": round(
                            hybrid_result.sparse_latency_seconds, 6
                        ),
                        "fusion": round(
                            hybrid_result.fusion_latency_seconds, 6
                        ),
                        "retrieval_total": round(
                            hybrid_result.total_latency_seconds, 6
                        ),
                    }
                else:
                    if hybrid_retriever is None or cross_encoder is None:
                        raise RuntimeError("Hybrid Rerank 资源未初始化。")
                    if resolved_candidate_k is None or resolved_rrf_k is None:
                        raise RuntimeError("Hybrid 参数未初始化。")
                    hybrid_result = hybrid_retriever.search(
                        item["question"],
                        top_k=fusion_k,
                        candidate_k=resolved_candidate_k,
                        rrf_k=resolved_rrf_k,
                        fusion_k=fusion_k,
                    )
                    rerank_input = hybrid_rerank_candidates(
                        hybrid_result.candidates[:reranker_candidate_k]
                    )
                    rerank_result = rerank_candidates(
                        item["question"],
                        rerank_input,
                        cross_encoder,
                        top_k=top_k,
                        max_chunks_per_file=max_chunks_per_file,
                    )
                    sources = build_reranked_sources(
                        rerank_result.candidates
                    )
                    latency_breakdown = {
                        "dense": round(
                            hybrid_result.dense_latency_seconds, 6
                        ),
                        "bm25": round(
                            hybrid_result.sparse_latency_seconds, 6
                        ),
                        "fusion": round(
                            hybrid_result.fusion_latency_seconds, 6
                        ),
                        "rerank": round(rerank_result.latency_seconds, 6),
                        "retrieval_total": round(
                            hybrid_result.total_latency_seconds
                            + rerank_result.latency_seconds,
                            6,
                        ),
                    }

                file_counts = Counter(
                    source["file_name"] for source in sources
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
                        "split": item["split"],
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
                        ) | (
                            calculate_ranked_retrieval_metrics(
                                qrels.get(item["id"], []),
                                sources,
                                cutoffs=(top_k,),
                            )
                            if qrels is not None
                            else {}
                        ),
                        "latency_breakdown_seconds": latency_breakdown,
                        "latency_seconds": round(
                            time.perf_counter() - start,
                            4,
                        ),
                    }
                )
                print(f"  [{index}/{len(questions)}] {item['id']}")

            run_candidate_k = (
                resolved_candidate_k
                if method in {"hybrid", "hybrid_rerank"}
                else (
                    reranker_candidate_k
                    if method == "dense_rerank"
                    else (
                        top_k * candidate_multiplier
                        if max_chunks_per_file is not None
                        else top_k
                    )
                )
            )
            run = {
                "method": method,
                "top_k": top_k,
                "candidate_k": run_candidate_k,
                "rrf_k": resolved_rrf_k,
                "fusion_k": (
                    fusion_k
                    if method in {"hybrid", "hybrid_rerank"}
                    else None
                ),
                "reranker_candidate_k": (
                    reranker_candidate_k
                    if method in {"dense_rerank", "hybrid_rerank"}
                    else None
                ),
                "max_chunks_per_file": max_chunks_per_file,
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--method",
        choices=sorted(RETRIEVAL_METHODS),
        default="dense",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 8, 10],
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        nargs="+",
        help="Hybrid branch candidate counts.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        nargs="+",
        help="Hybrid RRF constants.",
    )
    parser.add_argument(
        "--fusion-k",
        type=int,
        default=DEFAULT_HYBRID_FUSION_K,
    )
    parser.add_argument(
        "--split",
        choices=["legacy", "dev", "test"],
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANKER_BATCH_SIZE,
    )
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_K,
    )
    parser.add_argument(
        "--reranker-device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--max-chunks-per-file", type=int)
    parser.add_argument("--qrels", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_outputs = {
        "dense": DEFAULT_OUTPUT_PATH,
        "bm25": DEFAULT_BM25_OUTPUT_PATH,
        "hybrid": DEFAULT_HYBRID_OUTPUT_PATH,
        "dense_rerank": DEFAULT_DENSE_RERANK_OUTPUT_PATH,
        "hybrid_rerank": DEFAULT_HYBRID_RERANK_OUTPUT_PATH,
    }
    output_path = args.output or default_outputs[args.method]
    payload = run_retrieval_experiment(
        questions_path=args.questions,
        output_path=output_path,
        top_k_values=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        max_chunks_per_file=args.max_chunks_per_file,
        qrels_path=args.qrels,
        method=args.method,
        hybrid_candidate_k_values=args.candidate_k,
        rrf_k_values=args.rrf_k,
        fusion_k=args.fusion_k,
        split=args.split,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        reranker_candidate_k=args.reranker_candidate_k,
        reranker_device=args.reranker_device,
    )
    for run in payload["runs"]:
        print(
            json.dumps(
                {
                    "method": run["method"],
                    "top_k": run["top_k"],
                    "candidate_k": run["candidate_k"],
                    "rrf_k": run["rrf_k"],
                    **run["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"检索实验结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
