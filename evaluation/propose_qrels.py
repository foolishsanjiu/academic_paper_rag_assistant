"""Propose one semantically matched qrel per annotated source page.

This utility creates a reviewable draft.  It does not replace human relevance
judgment: the selected Chunk text and score are written to a separate review
report so weak matches can be inspected before the draft is promoted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable


# Hugging Face reads these flags while its modules are imported, not when the
# model constructor is called.  Set them before importing vector_store.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from evaluation.dataset import load_questions  # noqa: E402
from evaluation.evaluate import write_json_atomic  # noqa: E402
from vector_store import create_embedding_model, load_vector_store  # noqa: E402


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "results" / "qrels_draft.json"
DEFAULT_REVIEW_PATH = Path(__file__).parent / "results" / "qrels_review.json"


SearchPage = Callable[[str, str, int], list[tuple[Any, float]]]


def build_search_text(question: dict[str, Any]) -> str:
    """Favor the expected evidence wording over surface query wording."""
    parts = [question["expected_answer"]]
    parts.extend(question.get("expected_key_points", []))
    return "\n".join(str(part) for part in parts if str(part).strip())


def build_source_search_text(
    question: dict[str, Any],
    source_index: int,
) -> str:
    """Extract the answer clause most likely to describe one source.

    Comparison and cross-document answers are authored in source order.  Using
    the whole combined answer for every source lets vocabulary from one paper
    pull another paper's selection toward an unrelated Chunk.
    """
    source_count = len(question["source_files"])
    if source_count == 1:
        return build_search_text(question)

    clauses = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？；;])\s*", question["expected_answer"])
        if item.strip()
    ]
    key_points = question.get("expected_key_points", [])
    parts: list[str] = []
    if len(clauses) >= source_count:
        parts.append(clauses[min(source_index, len(clauses) - 1)])
    else:
        parts.append(question["expected_answer"])
    if len(key_points) >= source_count:
        parts.append(str(key_points[min(source_index, len(key_points) - 1)]))
    return "\n".join(parts)


def propose_qrels(
    questions: list[dict[str, Any]],
    search_page: SearchPage,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select the closest Chunk for every declared file/page evidence pair."""
    qrels: dict[str, list[dict[str, Any]]] = {}
    review_questions: dict[str, Any] = {}

    for question in questions:
        question_id = question["id"]
        if question["type"] == "no_answer":
            qrels[question_id] = []
            review_questions[question_id] = {
                "type": "no_answer",
                "selections": [],
            }
            continue

        judgments: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for source_index, (file_name, page_number) in enumerate(
            zip(question["source_files"], question["source_pages"])
        ):
            search_text = build_source_search_text(question, source_index)
            matches = search_page(search_text, file_name, page_number)
            if not matches:
                raise ValueError(
                    f"{question_id} 的来源页没有可用 Chunk："
                    f"{file_name} page {page_number}"
                )

            document, score = matches[0]
            metadata = document.metadata
            chunk_id = str(metadata.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError(f"{question_id} 的候选 Chunk 缺少 chunk_id。")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            judgments.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": str(metadata.get("file_name", file_name)),
                    "page_number": int(metadata.get("page_number", page_number)),
                    "relevance": 3,
                    "reason": (
                        "Direct evidence selected from the annotated source "
                        "page for the expected answer."
                    ),
                }
            )
            selections.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "page_number": page_number,
                    "score": round(float(score), 6),
                    "search_text": search_text,
                    "text": document.page_content,
                }
            )

        qrels[question_id] = judgments
        review_questions[question_id] = {
            "type": question["type"],
            "expected_answer": question["expected_answer"],
            "expected_key_points": question.get("expected_key_points", []),
            "selections": selections,
        }

    review = {
        "schema_version": 1,
        "policy": "top semantic match per annotated source file/page",
        "question_count": len(questions),
        "questions": review_questions,
    }
    return qrels, review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a semantic qrels draft and a Chunk review report.",
    )
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--review-output", type=Path, default=DEFAULT_REVIEW_PATH
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    embeddings = create_embedding_model(DEFAULT_EMBEDDING_MODEL)
    store = load_vector_store(embeddings)

    def search_page(
        query: str,
        file_name: str,
        page_number: int,
    ) -> list[tuple[Any, float]]:
        return store.similarity_search_with_relevance_scores(
            query,
            k=3,
            filter={
                "$and": [
                    {"file_name": {"$eq": file_name}},
                    {"page_number": {"$eq": page_number}},
                ]
            },
        )

    qrels, review = propose_qrels(questions, search_page)
    write_json_atomic(args.output, qrels)
    write_json_atomic(args.review_output, review)
    selection_count = sum(len(items) for items in qrels.values())
    scores = [
        selection["score"]
        for item in review["questions"].values()
        for selection in item["selections"]
    ]
    print(
        json.dumps(
            {
                "question_count": len(questions),
                "selection_count": selection_count,
                "minimum_score": min(scores) if scores else None,
                "output": str(args.output.resolve()),
                "review_output": str(args.review_output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
