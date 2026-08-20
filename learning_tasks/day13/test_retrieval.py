"""Test semantic retrieval from the academic-paper vector store."""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from retriever import (
    cosine_distance_to_similarity,
    retrieve_with_scores,
)
from vector_store import (
    create_embedding_model,
    get_vector_count,
    load_vector_store,
)


def print_results(
    query: str,
    results: list,
) -> None:
    """
    Print retrieval results with metadata.
    """
    print()
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    for rank, (document, distance) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        similarity = (
            cosine_distance_to_similarity(
                distance
            )
        )

        print()
        print(
            f">>> Rank {rank}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Cosine similarity: "
            f"{similarity:.4f}"
        )

        print(
            "Document ID: "
            f"{metadata.get('document_id')}"
        )

        print(
            "Document type: "
            f"{metadata.get('document_type')}"
        )

        print(
            "File: "
            f"{metadata.get('file_name')}"
        )

        print(
            "PDF page: "
            f"{metadata.get('page_number')}"
        )

        print(
            "Chunk ID: "
            f"{metadata.get('chunk_id')}"
        )

        print(
            "Chunk index: "
            f"{metadata.get('chunk_index')}"
        )

        print("-" * 80)

        print(
            document.page_content
        )

        print(
            f"\n<<< End Rank {rank}"
        )


def main() -> None:
    print()
    print("=" * 80)
    print("Step 1: 加载 Embedding 模型")
    print("=" * 80)

    embeddings = (
        create_embedding_model()
    )

    print()
    print("=" * 80)
    print("Step 2: 加载 Chroma")
    print("=" * 80)

    vector_store = (
        load_vector_store(
            embeddings=embeddings
        )
    )

    vector_count = (
        get_vector_count(
            vector_store
        )
    )

    print(
        f"Chroma Records："
        f"{vector_count}"
    )

    print()
    print("=" * 80)
    print("Step 3: 输入检索问题")
    print("=" * 80)

    query = input(
        "请输入问题："
    ).strip()

    top_k_text = input(
        "请输入 Top-k（默认 5）："
    ).strip()

    if top_k_text:
        top_k = int(
            top_k_text
        )

    else:
        top_k = 5

    results = (
        retrieve_with_scores(
            vector_store=vector_store,
            query=query,
            top_k=top_k,
        )
    )

    print_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()