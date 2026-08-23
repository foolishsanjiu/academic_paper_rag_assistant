"""Compare Top-k and Temperature settings in the RAG pipeline."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from datetime import datetime
import json
from pathlib import Path

from config import get_settings
from index_manifest import load_index_manifest
from llm_client import LLMClient
from rag_chain import RAGChain
from vector_store import (
    create_embedding_model,
    load_vector_store,
)


TEST_QUESTION = (
    "How does the azimuth-controllable generative adversarial network control the azimuth angle of generated SAR target images?"
)


EXPERIMENTS = [
    {
        "name": "top_k_1",
        "top_k": 1,
        "temperature": 0.2,
    },
    {
        "name": "top_k_3",
        "top_k": 3,
        "temperature": 0.2,
    },
    {
        "name": "top_k_5",
        "top_k": 5,
        "temperature": 0.2,
    },
    {
        "name": "top_k_10",
        "top_k": 10,
        "temperature": 0.2,
    },
    {
        "name": "temperature_0",
        "top_k": 5,
        "temperature": 0.0,
    },
    {
        "name": "temperature_07",
        "top_k": 5,
        "temperature": 0.7,
    },
]


def serialize_sources(
    sources,
) -> list[dict]:
    """Convert RAG sources into JSON-serializable dictionaries."""
    result = []

    for source in sources:

        result.append(
            {
                "rank": source.rank,
                "document_id": (
                    source.document_id
                ),
                "file_name": (
                    source.file_name
                ),
                "page_number": (
                    source.page_number
                ),
                "chunk_id": (
                    source.chunk_id
                ),
                "similarity": (
                    source.similarity
                ),
                "text": (
                    source.text
                ),
            }
        )

    return result


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    output_dir = (
        project_root
        / "learning_tasks"
        / "day18"
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load resources only once
    # --------------------------------------------------

    print(
        "正在初始化 LLM..."
    )

    settings = get_settings()

    llm = LLMClient(
        settings
    )

    print(
        "正在加载 Embedding 模型..."
    )

    embeddings = (
        create_embedding_model()
    )

    print(
        "正在加载 Chroma..."
    )

    vector_store = (
        load_vector_store(
            embeddings=embeddings
        )
    )

    index_manifest = (
        load_index_manifest()
    )

    results = []

    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------

    for experiment in EXPERIMENTS:

        name = experiment["name"]

        top_k = experiment[
            "top_k"
        ]

        temperature = experiment[
            "temperature"
        ]

        print()
        print("=" * 70)

        print(
            f"Experiment: {name}"
        )

        print(
            f"Top-k: {top_k}"
        )

        print(
            f"Temperature: {temperature}"
        )

        print("=" * 70)

        rag = RAGChain(
            llm=llm,
            vector_store=vector_store,
            top_k=top_k,
        )

        response = rag.ask(
            question=TEST_QUESTION,
            chat_history=None,
            temperature=temperature,
        )

        experiment_result = {
            "name": name,

            "question": (
                TEST_QUESTION
            ),

            "top_k": top_k,

            "temperature": (
                temperature
            ),

            "retrieval_query": (
                response.retrieval_query
            ),

            "answer": (
                response.answer
            ),

            "sources": (
                serialize_sources(
                    response.sources
                )
            ),
        }

        results.append(
            experiment_result
        )

        print()
        print(
            response.answer
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),

        "index_manifest": (
            index_manifest
        ),

        "experiments": (
            results
        ),
    }

    output_path = (
        output_dir
        / "parameter_comparison.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)

    print(
        "实验完成："
        f"{output_path}"
    )


if __name__ == "__main__":
    main()