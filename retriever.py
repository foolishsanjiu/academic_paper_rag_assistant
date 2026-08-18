"""Semantic retrieval utilities for academic paper chunks."""

from langchain_chroma import Chroma
from langchain_core.documents import Document


DEFAULT_TOP_K = 5


def validate_query(query: str) -> str:
    """
    Clean and validate a retrieval query.

    Args:
        query:
            User query.

    Returns:
        Cleaned query.

    Raises:
        ValueError:
            If the query is empty.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "检索问题不能为空。"
        )

    return cleaned_query


def validate_top_k(top_k: int) -> None:
    """
    Validate the Top-k retrieval parameter.
    """
    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )


def cosine_distance_to_similarity(
    distance: float,
) -> float:
    """
    Convert cosine distance to cosine similarity.

    Chroma cosine distance:

        distance = 1 - cosine_similarity

    Therefore:

        cosine_similarity = 1 - distance
    """
    return 1.0 - distance


def retrieve_with_scores(
    vector_store: Chroma,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Retrieve the most relevant chunks with distance scores.

    Args:
        vector_store:
            Loaded Chroma vector store.

        query:
            User question.

        top_k:
            Number of chunks to return.

    Returns:
        List of (Document, distance) tuples.
    """
    cleaned_query = validate_query(
        query
    )

    validate_top_k(
        top_k
    )

    results = (
        vector_store.similarity_search_with_score(
            query=cleaned_query,
            k=top_k,
        )
    )

    return results


def create_retriever(
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Convert Chroma into a LangChain Retriever.

    This will be useful when building rag_chain.py later.
    """
    validate_top_k(
        top_k
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
        },
    )