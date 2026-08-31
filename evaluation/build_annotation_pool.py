"""Build a blind, deduplicated Chunk pool for qrels annotation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIRECTORY, DEFAULT_COLLECTION_NAME  # noqa: E402
from evaluation.dataset import load_questions  # noqa: E402
from evaluation.evaluate import write_json_atomic  # noqa: E402


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "results" / "annotation_pool.json"


def load_index_rows() -> dict[str, dict[str, Any]]:
    """Read indexed text and metadata without loading the embedding model."""
    from langchain_chroma import Chroma

    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(f"Chroma 数据库不存在：{CHROMA_DIRECTORY}")

    store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=None,
    )
    raw = store.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    return {
        str(chunk_id): {
            "chunk_id": str(chunk_id),
            "file_name": str((metadata or {}).get("file_name", "")),
            "page_number": (metadata or {}).get("page_number"),
            "chunk_index": (metadata or {}).get("chunk_index"),
            "text": str(document or ""),
        }
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    }


def iter_result_sets(
    payload: dict[str, Any],
    source_name: str,
) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """Yield named result lists from retrieval-only or end-to-end outputs."""
    runs = payload.get("runs")
    if isinstance(runs, list):
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or not isinstance(run.get("results"), list):
                raise ValueError(f"无效 retrieval run：{source_name}[{index}]")
            parameters = []
            for key in ("method", "strategy", "top_k"):
                if run.get(key) is not None:
                    parameters.append(f"{key}={run[key]}")
            suffix = ",".join(parameters) or f"run={index}"
            yield f"{source_name}:{suffix}", run["results"]
        return

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"结果文件缺少 runs/results：{source_name}")
    yield source_name, results


def _normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    chunk_id = str(source.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError("检索候选缺少 chunk_id。")
    return {
        "chunk_id": chunk_id,
        "file_name": str(source.get("file_name", "")),
        "page_number": source.get("page_number"),
        "chunk_index": source.get("chunk_index"),
        "text": str(source.get("text", "")),
    }


def build_annotation_pool(
    questions: list[dict[str, Any]],
    result_sets: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    candidate_depth: int = 20,
    index_rows: dict[str, dict[str, Any]] | None = None,
    include_source_pages: bool = True,
) -> dict[str, Any]:
    """Merge retrieval pools and expected-page Chunks into blind candidates."""
    if candidate_depth < 1:
        raise ValueError("candidate_depth 必须大于等于 1。")

    questions_by_id = {item["id"]: item for item in questions}
    candidates: dict[str, dict[str, dict[str, Any]]] = {
        question_id: {} for question_id in questions_by_id
    }
    provenance: dict[str, dict[str, set[str]]] = {
        question_id: {} for question_id in questions_by_id
    }

    for origin, results in result_sets:
        for result in results:
            question_id = str(result.get("id", ""))
            if question_id not in questions_by_id:
                continue
            retrieved_sources = result.get("retrieved_sources") or []
            if not isinstance(retrieved_sources, list):
                raise ValueError(f"{question_id} 的 retrieved_sources 必须是数组。")
            for source in retrieved_sources[:candidate_depth]:
                normalized = _normalize_source(source)
                chunk_id = normalized["chunk_id"]
                candidates[question_id].setdefault(chunk_id, normalized)
                provenance[question_id].setdefault(chunk_id, set()).add(origin)

    if include_source_pages:
        if index_rows is None:
            raise ValueError("include_source_pages=True 时必须提供 index_rows。")
        rows = list(index_rows.values())
        for question_id, question in questions_by_id.items():
            expected_pairs = set(
                zip(question["source_files"], question["source_pages"])
            )
            for row in rows:
                pair = (row["file_name"], row["page_number"])
                if pair not in expected_pairs:
                    continue
                chunk_id = row["chunk_id"]
                candidates[question_id].setdefault(chunk_id, dict(row))
                provenance[question_id].setdefault(chunk_id, set()).add(
                    "expected_source_page"
                )

    question_pools: dict[str, Any] = {}
    for question_id, question in questions_by_id.items():
        blind_candidates = sorted(
            candidates[question_id].values(),
            key=lambda item: sha256(
                f"{question_id}\0{item['chunk_id']}".encode("utf-8")
            ).hexdigest(),
        )
        question_pools[question_id] = {
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "expected_key_points": question.get("expected_key_points", []),
            "candidates": blind_candidates,
            "pool_provenance": {
                chunk_id: sorted(origins)
                for chunk_id, origins in sorted(provenance[question_id].items())
            },
        }

    return {
        "schema_version": 1,
        "candidate_depth": candidate_depth,
        "question_count": len(questions),
        "questions": question_pools,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a blind pool for Chunk-level relevance annotation.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=[],
    )
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument(
        "--exclude-source-pages",
        action="store_true",
        help="Do not add every Chunk from the expected file/page pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    result_sets: list[tuple[str, list[dict[str, Any]]]] = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"结果文件必须是 JSON object：{path}")
        result_sets.extend(iter_result_sets(payload, path.name))

    include_source_pages = not args.exclude_source_pages
    index_rows = load_index_rows() if include_source_pages else None
    pool = build_annotation_pool(
        questions,
        result_sets,
        candidate_depth=args.candidate_depth,
        index_rows=index_rows,
        include_source_pages=include_source_pages,
    )
    write_json_atomic(args.output, pool)
    candidate_count = sum(
        len(item["candidates"])
        for item in pool["questions"].values()
    )
    print(
        json.dumps(
            {
                "question_count": pool["question_count"],
                "candidate_count": candidate_count,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
