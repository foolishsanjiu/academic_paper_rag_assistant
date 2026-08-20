"""Validate page-level and chunk-level metadata."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from pathlib import Path

from document_loader import load_pdf_directory
from text_splitter import split_documents


PAGE_REQUIRED_KEYS = {
    "document_id",
    "document_type",
    "file_name",
    "page_index",
    "page_number",
    "page_label",
    "char_count",
    "is_empty",
    "parse_warnings",
}


CHUNK_REQUIRED_KEYS = (
    PAGE_REQUIRED_KEYS
    | {
        "chunk_id",
        "chunk_index",
        "chunk_char_count",
    }
)


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    paper_dir = (
        project_root
        / "data"
        / "papers"
    )

    # --------------------------------------------------
    # Page Documents
    # --------------------------------------------------

    page_documents = load_pdf_directory(
        paper_dir
    )

    print(
        f"Page Documents: "
        f"{len(page_documents)}"
    )

    for document in page_documents:

        missing = (
            PAGE_REQUIRED_KEYS
            - set(document.metadata)
        )

        if missing:
            raise ValueError(
                "Page Document metadata 缺失："
                f"{missing}"
            )

    print(
        "Page metadata validation: PASS"
    )

    # --------------------------------------------------
    # Chunk Documents
    # --------------------------------------------------

    chunks = split_documents(
        documents=page_documents,
        chunk_size=600,
        chunk_overlap=100,
    )

    print(
        f"Chunk Documents: "
        f"{len(chunks)}"
    )

    chunk_ids: list[str] = []

    for chunk in chunks:

        missing = (
            CHUNK_REQUIRED_KEYS
            - set(chunk.metadata)
        )

        if missing:
            raise ValueError(
                "Chunk metadata 缺失："
                f"{missing}"
            )

        chunk_ids.append(
            chunk.metadata["chunk_id"]
        )

    print(
        "Chunk metadata validation: PASS"
    )

    # --------------------------------------------------
    # Unique IDs
    # --------------------------------------------------

    unique_chunk_ids = set(
        chunk_ids
    )

    print(
        f"Chunk IDs: {len(chunk_ids)}"
    )

    print(
        f"Unique Chunk IDs: "
        f"{len(unique_chunk_ids)}"
    )

    if len(chunk_ids) != len(
        unique_chunk_ids
    ):
        raise ValueError(
            "检测到重复 chunk_id。"
        )

    print(
        "Chunk ID uniqueness: PASS"
    )

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------

    sample = chunks[0]

    print()
    print("=" * 70)
    print("Chunk Metadata Example")
    print("=" * 70)

    for key, value in (
        sample.metadata.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()