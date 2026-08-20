"""Text splitting utilities for academic paper documents."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create a recursive character text splitter.

    Args:
        chunk_size:
            Maximum target size of each chunk measured
            in characters.

        chunk_overlap:
            Target number of overlapping characters
            between neighboring chunks.

    Returns:
        Configured RecursiveCharacterTextSplitter.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size 必须大于 0。"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap 不能小于 0。"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须小于 chunk_size。"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        # 当前阶段按照字符计算长度。
        length_function=len,

        # 优先保持段落、换行和单词边界。
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],

        is_separator_regex=False,
    )


def split_page_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split one page-level Document into chunk Documents.

    The original page metadata is preserved, while
    chunk-specific metadata is added.

    Args:
        document:
            One page-level Document returned by
            document_loader.py.

        splitter:
            Configured text splitter.

    Returns:
        A list of chunk-level Documents.
    """
    text = document.page_content.strip()

    if not text:
        return []

    chunk_texts = splitter.split_text(text)

    chunk_documents: list[Document] = []

    file_name = document.metadata.get(
        "file_name",
        "unknown.pdf",
    )

    page_number = document.metadata.get(
        "page_number",
        -1,
    )

    document_id = document.metadata.get(
        "document_id",
        file_name,
    )

    for chunk_index, chunk_text in enumerate(
        chunk_texts
    ):
        # 复制原页面 metadata，
        # 防止直接修改父 Document 的字典。
        metadata = dict(document.metadata)

        chunk_id = (
            f"{document_id}"
            f"::page_{page_number}"
            f"::chunk_{chunk_index}"
        )

        metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_char_count": len(chunk_text),
            }
        )

        chunk_document = Document(
            page_content=chunk_text,
            metadata=metadata,
        )

        chunk_documents.append(
            chunk_document
        )

    return chunk_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split multiple page-level Documents into chunks.

    Args:
        documents:
            Documents returned by document_loader.py.

        chunk_size:
            Maximum target chunk size in characters.

        chunk_overlap:
            Character overlap between neighboring chunks.

    Returns:
        Chunk-level Documents.
    """
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for document in documents:
        chunks = split_page_document(
            document=document,
            splitter=splitter,
        )

        all_chunks.extend(chunks)

    return all_chunks


def print_chunk_summary(
    chunks: list[Document],
) -> None:
    """
    Print basic statistics about generated chunks.
    """
    if not chunks:
        print("没有生成任何 Chunk。")
        return

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print()
    print("=" * 70)
    print("Chunk 切分统计")
    print("=" * 70)

    print(
        f"Chunk 总数：{len(chunks)}"
    )

    print(
        f"最短 Chunk：{min(chunk_lengths)} 字符"
    )

    print(
        f"最长 Chunk：{max(chunk_lengths)} 字符"
    )

    print(
        "平均 Chunk："
        f"{sum(chunk_lengths) / len(chunk_lengths):.1f} 字符"
    )


def print_chunk_sample(
    chunk: Document,
) -> None:
    """
    Print one complete chunk example.
    """
    print()
    print("=" * 70)
    print("Chunk 示例")
    print("=" * 70)

    print()
    print("metadata:")

    for key, value in chunk.metadata.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("-" * 70)
    print("page_content:")
    print("-" * 70)

    print(chunk.page_content)


if __name__ == "__main__":
    # text_splitter.py 本身只负责文本切分。
    # 完整运行测试放在 compare_chunk_sizes.py 中。
    print(
        "请运行 "
        "learning_tasks/day11/compare_chunk_sizes.py "
        "测试文本切分效果。"
    )