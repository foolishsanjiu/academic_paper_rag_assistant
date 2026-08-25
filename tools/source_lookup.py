"""Exact source lookup by paper metadata, PDF page, or Chunk ID."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf
from langchain_chroma import Chroma

from config import (
    CHROMA_DIRECTORY,
    DEFAULT_COLLECTION_NAME,
    PAPER_DIRECTORY,
)
from tools.paper_library import find_paper_path, validate_paper_name


logger = logging.getLogger(__name__)
TOOL_NAME = "source_lookup"
ACTION_NAME = "lookup_source"


def success(**data: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": TOOL_NAME,
        "action": ACTION_NAME,
        **data,
    }


def failure(code: str, message: str) -> dict[str, Any]:
    logger.warning("SourceLookupTool rejected request | code=%s", code)
    return {
        "ok": False,
        "tool": TOOL_NAME,
        "action": ACTION_NAME,
        "error": {
            "code": code,
            "message": message,
        },
    }


def load_metadata_store(
    persist_directory: Path,
    collection_name: str,
) -> Chroma:
    """Open Chroma for metadata reads without loading an embedding model."""
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma 数据库不存在：{persist_directory}"
        )

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=None,
    )


def normalize_chroma_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Chroma's column-oriented get result into rows."""
    ids = raw.get("ids") or []
    documents = raw.get("documents") or [None] * len(ids)
    metadatas = raw.get("metadatas") or [None] * len(ids)
    rows: list[dict[str, Any]] = []

    for item_id, text, metadata in zip(ids, documents, metadatas):
        normalized_metadata = metadata or {}
        rows.append(
            {
                "id": str(item_id),
                "text": str(text or ""),
                "metadata": normalized_metadata,
            }
        )

    return rows


def metadata_page_number(metadata: dict[str, Any]) -> int | None:
    try:
        return int(metadata["page_number"])
    except (KeyError, TypeError, ValueError):
        return None


def lookup_source(
    paper_name: str,
    page_number: int | None = None,
    chunk_id: str | None = None,
    *,
    paper_directory: Path = PAPER_DIRECTORY,
    persist_directory: Path = CHROMA_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_store: Any | None = None,
) -> dict[str, Any]:
    """Return exact indexed chunks for a paper page or persistent Chunk ID."""
    cleaned_name, name_error = validate_paper_name(paper_name)
    if name_error is not None:
        return failure(
            name_error["error"]["code"],
            name_error["error"]["message"],
        )

    if page_number is None and (chunk_id is None or not chunk_id.strip()):
        return failure(
            "missing_locator",
            "必须提供 page_number 或 chunk_id。",
        )

    if page_number is not None and (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
    ):
        return failure(
            "invalid_page_number",
            "page_number 必须是大于等于 1 的整数。",
        )

    cleaned_chunk_id = chunk_id.strip() if chunk_id else None
    assert cleaned_name is not None
    paper_path = find_paper_path(cleaned_name, paper_directory)
    if paper_path is None:
        return failure(
            "paper_not_found",
            f"知识库中不存在论文：{cleaned_name}",
        )

    try:
        with pymupdf.open(paper_path) as document:
            paper_page_count = document.page_count
    except (pymupdf.FileDataError, OSError):
        logger.exception(
            "SourceLookupTool could not read PDF | file=%s",
            paper_path.name,
        )
        return failure(
            "paper_unreadable",
            f"论文无法读取：{paper_path.name}",
        )

    if page_number is not None and page_number > paper_page_count:
        return failure(
            "page_out_of_range",
            f"页码超出范围：{paper_path.name} 共 {paper_page_count} 页。",
        )

    logger.info(
        "SourceLookupTool call started | file=%s | page=%s | "
        "has_chunk_id=%s",
        paper_path.name,
        page_number,
        cleaned_chunk_id is not None,
    )

    try:
        store = vector_store or load_metadata_store(
            persist_directory,
            collection_name,
        )
        if cleaned_chunk_id is not None:
            raw = store.get(
                ids=[cleaned_chunk_id],
                include=["documents", "metadatas"],
            )
        else:
            raw = store.get(
                where={"file_name": paper_path.name},
                include=["documents", "metadatas"],
            )
    except Exception:
        logger.exception(
            "SourceLookupTool index read failed | file=%s",
            paper_path.name,
        )
        return failure(
            "index_unavailable",
            "向量索引不可用或无法读取。",
        )

    rows = normalize_chroma_rows(raw)
    matched_rows: list[dict[str, Any]] = []

    for row in rows:
        metadata = row["metadata"]
        if str(metadata.get("file_name", "")).casefold() != (
            paper_path.name.casefold()
        ):
            continue
        if cleaned_chunk_id is not None and row["id"] != cleaned_chunk_id:
            continue
        if (
            page_number is not None
            and metadata_page_number(metadata) != page_number
        ):
            continue
        matched_rows.append(row)

    if not matched_rows:
        if cleaned_chunk_id is not None:
            return failure(
                "chunk_not_found",
                "指定 Chunk 不存在，或不属于该论文/页码。",
            )
        return failure(
            "source_not_indexed",
            f"索引中没有找到 {paper_path.name} 第 {page_number} 页的 Chunk。",
        )

    matched_rows.sort(
        key=lambda row: int(row["metadata"].get("chunk_index", 0))
    )
    chunks = [
        {
            "chunk_id": row["id"],
            "page_number": metadata_page_number(row["metadata"]),
            "page_label": row["metadata"].get("page_label"),
            "chunk_index": row["metadata"].get("chunk_index"),
            "text": row["text"],
        }
        for row in matched_rows
    ]

    result = success(
        paper_name=paper_path.name,
        paper_page_count=paper_page_count,
        requested_page_number=page_number,
        requested_chunk_id=cleaned_chunk_id,
        match_count=len(chunks),
        chunks=chunks,
    )
    logger.info(
        "SourceLookupTool call completed | file=%s | matches=%s",
        paper_path.name,
        len(chunks),
    )
    return result
