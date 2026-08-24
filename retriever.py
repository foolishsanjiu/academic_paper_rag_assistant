"""Semantic retrieval utilities for academic paper chunks."""

from dataclasses import dataclass
from enum import Enum
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    DEFAULT_TOP_K,
    MULTI_DOCUMENT_CANDIDATE_K,
    MULTI_DOCUMENT_MAX_CHUNKS_PER_FILE,
    MULTI_DOCUMENT_TOP_K,
)


logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Supported retrieval behavior for the RAG pipeline."""

    FOCUSED = "focused"
    MULTI_DOCUMENT = "multi_document"


@dataclass(frozen=True)
class RetrievalOptions:
    """Resolved low-level parameters for one retrieval strategy."""

    strategy: RetrievalStrategy
    top_k: int
    candidate_k: int | None
    max_chunks_per_file: int | None


def get_retrieval_options(
    strategy: RetrievalStrategy | str = RetrievalStrategy.FOCUSED,
    top_k: int | None = None,
) -> RetrievalOptions:
    """Resolve a named strategy into tested retrieval parameters."""
    try:
        resolved_strategy = RetrievalStrategy(strategy)
    except ValueError as error:
        allowed = ", ".join(item.value for item in RetrievalStrategy)
        raise ValueError(
            f"不支持的检索策略：{strategy}。可选值：{allowed}"
        ) from error

    if top_k is not None:
        validate_top_k(top_k)

    if resolved_strategy is RetrievalStrategy.FOCUSED:
        return RetrievalOptions(
            strategy=resolved_strategy,
            top_k=top_k or DEFAULT_TOP_K,
            candidate_k=None,
            max_chunks_per_file=None,
        )

    resolved_top_k = top_k or MULTI_DOCUMENT_TOP_K
    candidate_k = (
        MULTI_DOCUMENT_CANDIDATE_K
        if resolved_top_k == MULTI_DOCUMENT_TOP_K
        else resolved_top_k * 4
    )

    return RetrievalOptions(
        strategy=resolved_strategy,
        top_k=resolved_top_k,
        candidate_k=candidate_k,
        max_chunks_per_file=(
            MULTI_DOCUMENT_MAX_CHUNKS_PER_FILE
        ),
    )


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
    candidate_k: int | None = None,
    max_chunks_per_file: int | None = None,
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

        candidate_k:
            Number of similarity candidates considered before
            source diversification. Defaults to ``top_k * 4`` when
            ``max_chunks_per_file`` is set, otherwise ``top_k``.

        max_chunks_per_file:
            Optional cap on chunks contributed by one PDF.

    Returns:
        List of (Document, distance) tuples.
    """
    cleaned_query = validate_query(
        query
    )

    validate_top_k(
        top_k
    )

    if candidate_k is not None and candidate_k < top_k:
        raise ValueError(
            "candidate_k 不能小于 top_k。"
        )

    if (
        max_chunks_per_file is not None
        and max_chunks_per_file <= 0
    ):
        raise ValueError(
            "max_chunks_per_file 必须大于 0。"
        )

    search_k = candidate_k or (
        top_k * 4
        if max_chunks_per_file is not None
        else top_k
    )

    logger.info(
        "Retrieval started | query_length=%s | top_k=%s | "
        "candidate_k=%s | max_chunks_per_file=%s",
        len(cleaned_query),
        top_k,
        search_k,
        max_chunks_per_file,
    )

    results = (
        vector_store.similarity_search_with_score(
            query=cleaned_query,
            k=search_k,
        )
    )

    if max_chunks_per_file is None:
        selected = results[:top_k]
        logger.info(
            "Retrieval completed | candidates=%s | selected=%s",
            len(results),
            len(selected),
        )
        return selected

    selected: list[tuple[Document, float]] = []
    file_counts: dict[str, int] = {}

    for document, distance in results:
        file_name = str(
            document.metadata.get(
                "file_name",
                document.metadata.get(
                    "document_id",
                    "unknown",
                ),
            )
        )
        current_count = file_counts.get(file_name, 0)

        if current_count >= max_chunks_per_file:
            continue

        selected.append((document, distance))
        file_counts[file_name] = current_count + 1

        if len(selected) == top_k:
            break

    logger.info(
        "Diversified retrieval completed | candidates=%s | "
        "selected=%s | unique_files=%s",
        len(results),
        len(selected),
        len(file_counts),
    )
    return selected


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
