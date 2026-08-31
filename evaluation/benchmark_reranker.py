"""Benchmark a local cross-encoder on retrieval candidates without LLM calls."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate import write_json_atomic  # noqa: E402
from reranker import TransformersCrossEncoderReranker  # noqa: E402


DEFAULT_RESULTS_PATH = Path(__file__).parent / "results" / "m4_hybrid_dev_grid.json"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "results" / "m5_reranker_gate.json"


def process_memory() -> dict[str, float | None]:
    """Return current and peak working set on Windows using only stdlib."""
    if sys.platform != "win32":
        return {"working_set_mb": None, "peak_working_set_mb": None}

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    success = get_process_memory_info(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        raise OSError("无法读取进程内存。")
    return {
        "working_set_mb": round(counters.WorkingSetSize / 1024**2, 2),
        "peak_working_set_mb": round(
            counters.PeakWorkingSetSize / 1024**2,
            2,
        ),
    }


def load_pairs(path: Path, required_count: int) -> tuple[str, list[Document]]:
    """Load one query and deterministic passages from the M4 winner."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    matching_runs = [
        run
        for run in payload.get("runs", [])
        if run.get("top_k") == 10
        and run.get("candidate_k") == 20
        and run.get("rrf_k") == 60
    ]
    if len(matching_runs) != 1:
        raise ValueError("无法唯一定位 M4 Hybrid dev 胜出配置。")

    query = str(matching_runs[0]["results"][0]["question"])
    documents: list[Document] = []
    for result in matching_runs[0].get("results", []):
        for source in result.get("retrieved_sources", []):
            documents.append(
                Document(
                    page_content=str(source["text"]),
                    metadata={"chunk_id": source["chunk_id"]},
                )
            )
            if len(documents) == required_count:
                return query, documents
    raise ValueError(
        f"候选不足：required={required_count}, actual={len(documents)}"
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def run_benchmark(
    model_path: Path,
    results_path: Path,
    output_path: Path,
    pair_counts: list[int],
    batch_sizes: list[int],
    repeat: int,
) -> dict[str, Any]:
    if any(value <= 0 for value in pair_counts + batch_sizes) or repeat <= 0:
        raise ValueError("pair count、batch size 和 repeat 必须大于 0。")

    query, documents = load_pairs(results_path, max(pair_counts))
    memory_before = process_memory()
    load_start = time.perf_counter()
    reranker = TransformersCrossEncoderReranker(
        str(model_path),
        batch_size=batch_sizes[0],
        device="cpu",
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - load_start
    memory_after_load = process_memory()

    reranker.score(query, documents[:2])
    benchmarks: list[dict[str, Any]] = []
    for pair_count in pair_counts:
        for batch_size in batch_sizes:
            reranker.batch_size = batch_size
            start = time.perf_counter()
            scores = reranker.score(query, documents[:pair_count])
            benchmarks.append(
                {
                    "pair_count": pair_count,
                    "batch_size": batch_size,
                    "latency_seconds": round(time.perf_counter() - start, 6),
                    "score_count": len(scores),
                }
            )

    forty_results = [
        result for result in benchmarks if result["pair_count"] == max(pair_counts)
    ]
    best_batch_size = min(
        forty_results,
        key=lambda item: item["latency_seconds"],
    )["batch_size"]
    reranker.batch_size = best_batch_size
    repeated_latencies: list[float] = []
    repeated_scores: list[list[float]] = []
    for _ in range(repeat):
        start = time.perf_counter()
        repeated_scores.append(
            reranker.score(query, documents[: max(pair_counts)])
        )
        repeated_latencies.append(time.perf_counter() - start)

    reference = repeated_scores[0]
    max_score_delta = max(
        abs(score - reference[index])
        for scores in repeated_scores[1:]
        for index, score in enumerate(scores)
    ) if len(repeated_scores) > 1 else 0.0
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(model_path.resolve()),
        "results_path": str(results_path.resolve()),
        "device": "cpu",
        "load_seconds": round(load_seconds, 6),
        "memory_before": memory_before,
        "memory_after_load": memory_after_load,
        "memory_after_benchmark": process_memory(),
        "benchmarks": benchmarks,
        "selected_batch_size": best_batch_size,
        "repeat_count": repeat,
        "repeated_40_candidate_latency_seconds": {
            "p50": round(statistics.median(repeated_latencies), 6),
            "p95": round(percentile(repeated_latencies, 0.95), 6),
            "values": [round(value, 6) for value in repeated_latencies],
        },
        "max_repeated_score_delta": max_score_delta,
    }
    write_json_atomic(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pair-count", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        model_path=args.model,
        results_path=args.results,
        output_path=args.output,
        pair_counts=args.pair_count,
        batch_sizes=args.batch_size,
        repeat=args.repeat,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
