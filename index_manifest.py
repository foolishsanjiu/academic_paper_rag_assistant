"""Persistent metadata describing the active Chroma index."""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from config import INDEX_MANIFEST_PATH

DEFAULT_MANIFEST_PATH = INDEX_MANIFEST_PATH

logger = logging.getLogger(__name__)


def write_index_manifest(
    *,
    paper_count: int,
    page_count: int,
    chunk_count: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    manifest_path: Path = (
        DEFAULT_MANIFEST_PATH
    ),
) -> None:
    """
    Save metadata describing how the current
    vector index was built.
    """
    manifest = {
        "version": 1,

        "built_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),

        "paper_count":
            paper_count,

        "page_count":
            page_count,

        "chunk_count":
            chunk_count,

        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "embedding_model":
            embedding_model,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        manifest_path
        .with_suffix(
            ".tmp"
        )
    )

    temp_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Replace only after the temporary
    # manifest is completely written.
    temp_path.replace(
        manifest_path
    )
    logger.info(
        "Index manifest written | path=%s | papers=%s | chunks=%s",
        manifest_path,
        paper_count,
        chunk_count,
    )


def load_index_manifest(
    manifest_path: Path = (
        DEFAULT_MANIFEST_PATH
    ),
) -> dict[str, Any] | None:
    """
    Load the current index manifest.
    """
    if not manifest_path.exists():
        return None

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def validate_index_manifest(
    *,
    vector_count: int,
    manifest: dict[str, Any],
    embedding_model: str | None = None,
) -> None:
    """
    Verify that Chroma record count matches
    the recorded Chunk count.
    """
    if (
        isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or vector_count < 0
    ):
        raise ValueError("vector_count 必须是非负整数。")

    expected = manifest.get("chunk_count")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 0
    ):
        raise ValueError("索引清单中的 chunk_count 必须是非负整数。")

    if vector_count != expected:

        raise ValueError(
            "向量数据库记录数与索引清单不一致："
            f"Chroma={vector_count}, "
            f"Manifest={expected}"
        )

    if embedding_model is not None:
        recorded_model = str(manifest.get("embedding_model", "")).strip()
        if recorded_model != embedding_model:
            raise ValueError(
                "索引清单中的 Embedding 模型与当前运行配置不一致："
                f"Manifest={recorded_model or '<missing>'}, "
                f"Runtime={embedding_model}"
            )

    logger.info(
        "Index manifest validated | vector_count=%s",
        vector_count,
    )
