"""Build and reload the academic-paper Chroma vector store."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from document_loader import load_pdf_directory
from text_splitter import split_documents
from vector_store import (
    create_embedding_model,
    build_vector_store,
    load_vector_store,
    get_vector_count,
)


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


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
    # 1. Load PDFs
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 1: 读取 PDF")
    print("=" * 70)

    page_documents = load_pdf_directory(
        paper_dir
    )

    print(
        f"Page Documents："
        f"{len(page_documents)}"
    )

    # --------------------------------------------------
    # 2. Split into chunks
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 2: 文本切分")
    print("=" * 70)

    chunks = split_documents(
        documents=page_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        f"Chunk Documents："
        f"{len(chunks)}"
    )

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 3: 加载 Embedding 模型")
    print("=" * 70)

    embeddings = create_embedding_model()

    # --------------------------------------------------
    # 4. Test one embedding vector
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 4: Embedding 测试")
    print("=" * 70)

    sample_text = (
        "Synthetic aperture radar image generation"
    )

    sample_vector = embeddings.embed_query(
        sample_text
    )

    print(
        f"测试文本：{sample_text}"
    )

    print(
        f"Embedding 维度："
        f"{len(sample_vector)}"
    )

    print(
        "前 5 个向量值："
    )

    print(
        sample_vector[:5]
    )

    # --------------------------------------------------
    # 5. Build Chroma
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 5: 创建 Chroma 数据库")
    print("=" * 70)

    vector_store = build_vector_store(
        documents=chunks,
        embeddings=embeddings,

        # Day 12 第一版采用全量重建，
        # 避免旧索引与新 Chunk 混在一起。
        reset=True,
    )

    print(
        "写入后的向量数量："
        f"{get_vector_count(vector_store)}"
    )

    # --------------------------------------------------
    # 6. Reload Chroma
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Step 6: 重新加载 Chroma")
    print("=" * 70)

    del vector_store

    reloaded_vector_store = (
        load_vector_store(
            embeddings=embeddings
        )
    )

    reloaded_count = (
        get_vector_count(
            reloaded_vector_store
        )
    )

    print(
        f"重新加载后的向量数量："
        f"{reloaded_count}"
    )

    # --------------------------------------------------
    # 7. Verify
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("Day 12 验收")
    print("=" * 70)

    print(
        f"Page Documents："
        f"{len(page_documents)}"
    )

    print(
        f"Chunk Documents："
        f"{len(chunks)}"
    )

    print(
        f"Chroma Records："
        f"{reloaded_count}"
    )

    if reloaded_count == len(chunks):
        print(
            "结果：向量数据库构建成功。"
        )

    else:
        print(
            "警告：Chunk 数量与 "
            "Chroma Records 数量不一致。"
        )


if __name__ == "__main__":
    main()