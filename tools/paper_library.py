"""Structured queries about the local academic-paper knowledge base."""

from __future__ import annotations

from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

from config import DEFAULT_COLLECTION_NAME, INDEX_MANIFEST_PATH, PAPER_DIRECTORY
from document_loader import compute_document_id
from index_manifest import load_index_manifest
from knowledge_base import get_paper_names, list_pdf_files


logger = logging.getLogger(__name__)
TOOL_NAME = "paper_library"


class PaperLibraryAction(str, Enum):
    """Actions supported by PaperLibraryTool."""

    PAPER_COUNT = "paper_count"
    PAPER_LIST = "paper_list"
    PAPER_EXISTS = "paper_exists"
    PAPER_INFO = "paper_info"
    INDEX_STATUS = "index_status"


def success(action: str, **data: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": TOOL_NAME,
        "action": action,
        **data,
    }


def failure(
    action: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    logger.warning(
        "PaperLibraryTool rejected request | action=%s | code=%s",
        action,
        code,
    )
    return {
        "ok": False,
        "tool": TOOL_NAME,
        "action": action,
        "error": {
            "code": code,
            "message": message,
        },
    }


def validate_paper_name(
    paper_name: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Validate a filename without allowing directory traversal."""
    if paper_name is None or not paper_name.strip():
        return None, failure(
            action="paper_name_validation",
            code="missing_paper_name",
            message="paper_name 不能为空。",
        )

    cleaned = paper_name.strip()
    if Path(cleaned).name != cleaned:
        return None, failure(
            action="paper_name_validation",
            code="invalid_paper_name",
            message="paper_name 只能是文件名，不能包含目录路径。",
        )

    return cleaned, None


def find_paper_path(
    paper_name: str,
    paper_directory: Path,
) -> Path | None:
    """Find a PDF by filename using case-insensitive matching."""
    name_key = paper_name.casefold()
    return next(
        (
            path
            for path in list_pdf_files(paper_directory)
            if path.name.casefold() == name_key
        ),
        None,
    )


def query_paper_library(
    action: str,
    paper_name: str | None = None,
    *,
    paper_directory: Path = PAPER_DIRECTORY,
    manifest_path: Path = INDEX_MANIFEST_PATH,
) -> dict[str, Any]:
    """Run one whitelisted query about papers or the active index."""
    try:
        resolved_action = PaperLibraryAction(action)
    except ValueError:
        allowed = [item.value for item in PaperLibraryAction]
        return failure(
            action=str(action),
            code="invalid_action",
            message=f"不支持的 action。可选值：{', '.join(allowed)}。",
        )

    action_value = resolved_action.value
    logger.info(
        "PaperLibraryTool call started | action=%s",
        action_value,
    )

    if resolved_action is PaperLibraryAction.PAPER_COUNT:
        result = success(
            action_value,
            paper_count=len(list_pdf_files(paper_directory)),
        )

    elif resolved_action is PaperLibraryAction.PAPER_LIST:
        papers = get_paper_names(paper_directory)
        result = success(
            action_value,
            paper_count=len(papers),
            papers=papers,
        )

    elif resolved_action in {
        PaperLibraryAction.PAPER_EXISTS,
        PaperLibraryAction.PAPER_INFO,
    }:
        cleaned_name, error = validate_paper_name(paper_name)
        if error is not None:
            error["action"] = action_value
            return error

        assert cleaned_name is not None
        paper_path = find_paper_path(cleaned_name, paper_directory)

        if resolved_action is PaperLibraryAction.PAPER_EXISTS:
            result = success(
                action_value,
                paper_name=(paper_path.name if paper_path else cleaned_name),
                exists=paper_path is not None,
            )
        elif paper_path is None:
            return failure(
                action=action_value,
                code="paper_not_found",
                message=f"知识库中不存在论文：{cleaned_name}",
            )
        else:
            try:
                with pymupdf.open(paper_path) as document:
                    page_count = document.page_count
            except (pymupdf.FileDataError, OSError) as error:
                logger.exception(
                    "PaperLibraryTool could not read PDF | file=%s",
                    paper_path.name,
                )
                return failure(
                    action=action_value,
                    code="paper_unreadable",
                    message=f"论文无法读取：{paper_path.name}",
                )

            result = success(
                action_value,
                paper={
                    "file_name": paper_path.name,
                    "document_id": compute_document_id(paper_path),
                    "page_count": page_count,
                    "size_bytes": paper_path.stat().st_size,
                },
            )

    else:
        try:
            manifest = load_index_manifest(manifest_path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            logger.exception(
                "PaperLibraryTool could not load index manifest | path=%s",
                manifest_path,
            )
            return failure(
                action=action_value,
                code="invalid_index_manifest",
                message="索引清单损坏或无法读取。",
            )

        current_paper_count = len(list_pdf_files(paper_directory))
        if manifest is None:
            result = success(
                action_value,
                indexed=False,
                current_paper_count=current_paper_count,
                collection_name=DEFAULT_COLLECTION_NAME,
            )
        else:
            result = success(
                action_value,
                indexed=True,
                current_paper_count=current_paper_count,
                paper_count=manifest.get("paper_count"),
                page_count=manifest.get("page_count"),
                chunk_count=manifest.get("chunk_count"),
                chunk_size=manifest.get("chunk_size"),
                chunk_overlap=manifest.get("chunk_overlap"),
                embedding_model=manifest.get("embedding_model"),
                built_at=manifest.get("built_at"),
                collection_name=DEFAULT_COLLECTION_NAME,
                paper_set_matches_index=(
                    current_paper_count == manifest.get("paper_count")
                ),
            )

    logger.info(
        "PaperLibraryTool call completed | action=%s | ok=%s",
        action_value,
        result["ok"],
    )
    return result
