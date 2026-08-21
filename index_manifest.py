"""Persistent metadata describing the active Chroma index."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "chroma_db"
    / "index_manifest.json"
)


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
) -> None:
    """
    Verify that Chroma record count matches
    the recorded Chunk count.
    """
    expected = int(
        manifest["chunk_count"]
    )

    if vector_count != expected:

        raise ValueError(
            "向量数据库记录数与索引清单不一致："
            f"Chroma={vector_count}, "
            f"Manifest={expected}"
        )