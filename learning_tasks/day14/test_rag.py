"""Test the complete academic-paper RAG pipeline."""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from config import get_settings
from llm_client import LLMClient
from rag_chain import RAGChain
from vector_store import (
    create_embedding_model,
    load_vector_store,
)


def print_sources(
    response,
) -> None:
    """Print retrieved sources."""
    print()
    print("=" * 80)
    print("检索来源")
    print("=" * 80)

    for source in response.sources:

        print()
        print(
            f"[{source.rank}] "
            f"{source.file_name}"
        )

        print(
            f"PDF Page: "
            f"{source.page_number}"
        )

        print(
            f"Chunk: "
            f"{source.chunk_id}"
        )

        print(
            f"Similarity: "
            f"{source.similarity:.4f}"
        )

        preview = (
            source.text
            .replace(
                "\n",
                " ",
            )[:300]
        )

        print(
            f"Text: {preview}..."
        )


def main() -> None:

    # ----------------------------------------------
    # 1. LLM
    # ----------------------------------------------

    print()
    print("=" * 80)
    print("Step 1: 初始化 DeepSeek")
    print("=" * 80)

    settings = get_settings()

    llm = LLMClient(
        settings
    )

    # ----------------------------------------------
    # 2. Embedding
    # ----------------------------------------------

    print()
    print("=" * 80)
    print("Step 2: 加载 BGE-M3")
    print("=" * 80)

    embeddings = (
        create_embedding_model()
    )

    # ----------------------------------------------
    # 3. Vector Store
    # ----------------------------------------------

    print()
    print("=" * 80)
    print("Step 3: 加载 Chroma")
    print("=" * 80)

    vector_store = (
        load_vector_store(
            embeddings=embeddings
        )
    )

    # ----------------------------------------------
    # 4. RAG
    # ----------------------------------------------

    rag = RAGChain(
        llm=llm,
        vector_store=vector_store,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("Academic Paper RAG Assistant")
    print("输入 exit 结束。")
    print("=" * 80)

    chat_history: list[dict] = []

    while True:

        print()

        question = input(
            "User: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            response = rag.ask(
                question=question,
                chat_history=chat_history,
            )

            print()
            print(
                "Retrieval Query:"
            )

            print(
                response.retrieval_query
            )

            print()
            print("Assistant:")

            print(
                response.answer
            )

            print_sources(
                response
            )

            # 保存当前轮对话，
            # 下一轮用于 Query Rewrite。
            chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                }
            )

        except (
            ValueError,
            RuntimeError,
        ) as error:

            print(
                f"\n运行失败：{error}"
            )


if __name__ == "__main__":
    main()