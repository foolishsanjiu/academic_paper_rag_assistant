"""Automatic citation metrics for grounded RAG answers."""

from __future__ import annotations

import re
from typing import Any


BRACKET_PATTERN = re.compile(r"\[([^\[\]]*)\]")
INTEGER_PATTERN = re.compile(r"\d+")


def parse_citation_ranks(answer: str) -> tuple[list[int], bool]:
    """Return numeric citation ranks and whether all brackets are ``[n]``."""
    text = str(answer)
    tokens = BRACKET_PATTERN.findall(text)
    remaining = BRACKET_PATTERN.sub("", text)
    format_valid = (
        "[" not in remaining
        and "]" not in remaining
        and all(INTEGER_PATTERN.fullmatch(token.strip()) for token in tokens)
    )
    ranks = [
        int(token.strip())
        for token in tokens
        if INTEGER_PATTERN.fullmatch(token.strip())
    ]
    return ranks, format_valid


def calculate_citation_metrics(
    answer: str,
    sources: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    question_type: str,
) -> dict[str, Any]:
    """Calculate format, index, qrel relevance, and coverage metrics."""
    ranks, format_valid = parse_citation_ranks(answer)
    index_valid = all(1 <= rank <= len(sources) for rank in ranks)

    valid_ranks: list[int] = []
    seen_ranks: set[int] = set()
    for rank in ranks:
        if 1 <= rank <= len(sources) and rank not in seen_ranks:
            seen_ranks.add(rank)
            valid_ranks.append(rank)

    cited_chunk_ids = {
        str(sources[rank - 1].get("chunk_id", "")).strip()
        for rank in valid_ranks
        if str(sources[rank - 1].get("chunk_id", "")).strip()
    }
    relevance_by_chunk = {
        str(item.get("chunk_id", "")).strip(): int(item["relevance"])
        for item in qrels
        if str(item.get("chunk_id", "")).strip()
    }
    relevant_cited = {
        chunk_id
        for chunk_id in cited_chunk_ids
        if relevance_by_chunk.get(chunk_id, 0) > 0
    }
    required_evidence = {
        chunk_id
        for chunk_id, relevance in relevance_by_chunk.items()
        if relevance >= 2
    }
    answerable = question_type != "no_answer"

    return {
        "citation_format_validity": format_valid,
        "citation_index_validity": index_valid,
        "citation_count": len(ranks),
        "valid_citation_count": len(valid_ranks),
        "cited_ranks": valid_ranks,
        "citation_qrel_precision": (
            len(relevant_cited) / len(cited_chunk_ids)
            if cited_chunk_ids
            else None
        ),
        "citation_qrel_recall": (
            len(required_evidence & cited_chunk_ids) / len(required_evidence)
            if required_evidence
            else None
        ),
        "answer_has_citation": bool(valid_ranks) if answerable else None,
        "no_answer_has_no_spurious_citation": (
            not ranks and "[" not in str(answer) and "]" not in str(answer)
            if not answerable
            else None
        ),
    }
