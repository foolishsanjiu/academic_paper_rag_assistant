"""Compare different Top-k retrieval settings."""

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
    load_vector_store,
)


TOP_K_VALUES = [
    1,
    3,
    5,
    10,
]


def main() -> None:
    embeddings = (
        create_embedding_model()
    )

    vector_store = (
        load_vector_store(
            embeddings=embeddings
        )
    )

    query = input(
        "请输入测试问题："
    ).strip()

    for top_k in TOP_K_VALUES:

        results = (
            retrieve_with_scores(
                vector_store=vector_store,
                query=query,
                top_k=top_k,
            )
        )

        print()
        print("=" * 80)
        print(
            f"Top-k = {top_k}"
        )
        print("=" * 80)

        for rank, (
            document,
            distance,
        ) in enumerate(
            results,
            start=1,
        ):
            metadata = (
                document.metadata
            )

            similarity = (
                cosine_distance_to_similarity(
                    distance
                )
            )

            print(
                f"{rank}. "
                f"similarity={similarity:.4f}"
            )

            print(
                "   file="
                f"{metadata.get('file_name')}"
            )

            print(
                "   page="
                f"{metadata.get('page_number')}"
            )

            print(
                "   chunk="
                f"{metadata.get('chunk_id')}"
            )

            preview = (
                document.page_content
                .replace(
                    "\n",
                    " ",
                )[:200]
            )

            print(
                f"   text={preview}..."
            )

            print()


if __name__ == "__main__":
    main()