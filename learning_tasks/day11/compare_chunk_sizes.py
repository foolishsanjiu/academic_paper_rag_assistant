"""Compare different chunk sizes on academic PDF documents."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from document_loader import load_pdf_directory
from text_splitter import split_documents


CHUNK_SIZES = [
    300,
    600,
    1000,
]


def print_statistics(
    chunk_size: int,
    chunks: list,
) -> None:
    """
    Print chunk statistics for one configuration.
    """
    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    if not lengths:
        print(
            f"chunk_size={chunk_size}: "
            "没有生成 Chunk"
        )
        return

    print()
    print("=" * 70)

    print(
        f"chunk_size = {chunk_size}"
    )

    print("=" * 70)

    print(
        f"Chunk 数量：{len(chunks)}"
    )

    print(
        f"平均长度："
        f"{sum(lengths) / len(lengths):.1f}"
    )

    print(
        f"最短长度：{min(lengths)}"
    )

    print(
        f"最长长度：{max(lengths)}"
    )


def print_page_chunks(
    chunks: list,
    file_name: str,
    page_number: int,
    max_chunks: int = 4,
) -> None:
    """
    Display chunks from one selected PDF page.
    """
    selected_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk.metadata["file_name"]
            == file_name
            and
            chunk.metadata["page_number"]
            == page_number
        )
    ]

    print()
    print(
        f"示例页面："
        f"{file_name}, page {page_number}"
    )

    for chunk in selected_chunks[:max_chunks]:

        print()
        print("-" * 70)

        print(
            f"chunk_id: "
            f"{chunk.metadata['chunk_id']}"
        )

        print(
            f"字符数: "
            f"{len(chunk.page_content)}"
        )

        print("-" * 70)

        print(
            chunk.page_content
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

    documents = load_pdf_directory(
        paper_dir
    )

    print()
    print(
        f"原始 Page Documents："
        f"{len(documents)}"
    )

    for chunk_size in CHUNK_SIZES:

        # 暂时让 overlap ≈ chunk_size 的 15%～20%
        # 为了实验简单：
        if chunk_size == 300:
            overlap = 50

        elif chunk_size == 600:
            overlap = 100

        else:
            overlap = 150

        chunks = split_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        print_statistics(
            chunk_size=chunk_size,
            chunks=chunks,
        )


if __name__ == "__main__":
    main()