"""Validate evaluation questions and optional Chunk-level qrels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIRECTORY, DEFAULT_COLLECTION_NAME  # noqa: E402
from evaluation.dataset import (  # noqa: E402
    load_qrels,
    load_questions,
    validate_qrels,
)


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_QRELS_PATH = Path(__file__).with_name("qrels.json")


def load_index_metadata() -> dict[str, dict[str, Any]]:
    """Read Chroma metadata without loading an embedding model."""
    from langchain_chroma import Chroma

    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(f"Chroma 数据库不存在：{CHROMA_DIRECTORY}")

    store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=None,
    )
    raw = store.get(include=["metadatas"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    return {
        str(chunk_id): dict(metadata or {})
        for chunk_id, metadata in zip(ids, metadatas)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate evaluation questions and qrels.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--require-qrels", action="store_true")
    parser.add_argument("--check-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    qrels_path = args.qrels
    if qrels_path is None and DEFAULT_QRELS_PATH.exists():
        qrels_path = DEFAULT_QRELS_PATH

    if qrels_path is None:
        if args.require_qrels:
            raise FileNotFoundError("未提供 qrels，且默认 qrels.json 不存在。")
        summary = {
            "question_count": len(questions),
            "question_with_qrels_count": 0,
            "judgment_count": 0,
            "relevant_judgment_count": 0,
            "index_checked": False,
        }
    else:
        qrels = load_qrels(qrels_path)
        index_metadata = load_index_metadata() if args.check_index else None
        summary = {
            **validate_qrels(
                questions,
                qrels,
                index_metadata=index_metadata,
                require_complete=args.require_qrels,
            ),
            "index_checked": index_metadata is not None,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Evaluation dataset validation passed.")


if __name__ == "__main__":
    main()
