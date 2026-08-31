"""Deterministic in-memory BM25 retrieval over indexed paper Chunks."""

from __future__ import annotations

from importlib.metadata import version
import logging
import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_EPSILON,
    DEFAULT_BM25_K1,
)
from retriever import validate_query, validate_top_k


logger = logging.getLogger(__name__)

SPARSE_TOKENIZER_VERSION = "unicode_nfkc_ascii_v1"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize_sparse(text: str) -> list[str]:
    """Normalize text and extract explainable English/numeric terms."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return TOKEN_PATTERN.findall(normalized)


def load_sparse_documents(vector_store: Any) -> list[Document]:
    """Read all texts and metadata from Chroma without using embeddings."""
    raw = vector_store.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    texts = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    if not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError("Chroma 返回的 ID、文本和 metadata 数量不一致。")

    documents: list[Document] = []
    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        clean_chunk_id = str(chunk_id).strip()
        if not clean_chunk_id:
            raise ValueError("Chroma 中存在空 chunk_id。")
        clean_metadata = dict(metadata or {})
        metadata_chunk_id = str(
            clean_metadata.get("chunk_id", clean_chunk_id)
        ).strip()
        if metadata_chunk_id != clean_chunk_id:
            raise ValueError(
                f"Chroma ID 与 metadata chunk_id 不一致：{clean_chunk_id}"
            )
        clean_metadata["chunk_id"] = clean_chunk_id
        documents.append(
            Document(
                page_content=str(text or ""),
                metadata=clean_metadata,
                id=clean_chunk_id,
            )
        )
    return documents


class SparseRetriever:
    """BM25 index whose corpus position is fixed by sorted chunk_id."""

    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        epsilon: float = DEFAULT_BM25_EPSILON,
    ) -> None:
        if not documents:
            raise ValueError("无法用空文档集构建 BM25 索引。")

        rows: list[tuple[str, Document]] = []
        seen: set[str] = set()
        for document in documents:
            chunk_id = str(document.metadata.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError("BM25 文档缺少 chunk_id。")
            if chunk_id in seen:
                raise ValueError(f"BM25 文档包含重复 chunk_id：{chunk_id}")
            seen.add(chunk_id)
            rows.append((chunk_id, document))

        rows.sort(key=lambda item: item[0])
        self._chunk_ids = [chunk_id for chunk_id, _ in rows]
        self._documents = [document for _, document in rows]
        tokenized_corpus = [
            tokenize_sparse(document.page_content)
            for document in self._documents
        ]
        if not any(tokenized_corpus):
            raise ValueError("BM25 文档集不包含可索引词项。")
        self._model = BM25Okapi(
            tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Return positive-score results with a stable chunk_id tie-break."""
        cleaned_query = validate_query(query)
        validate_top_k(top_k)
        tokens = tokenize_sparse(cleaned_query)
        if not tokens:
            logger.info(
                "BM25 query has no supported lexical tokens | query_length=%s",
                len(cleaned_query),
            )
            return []

        scores = self._model.get_scores(tokens)
        ranked = sorted(
            (
                (document, float(score), chunk_id)
                for document, score, chunk_id in zip(
                    self._documents,
                    scores,
                    self._chunk_ids,
                )
                if float(score) > 0.0
            ),
            key=lambda item: (-item[1], item[2]),
        )
        return [
            (document, score)
            for document, score, _ in ranked[:top_k]
        ]

    def identity(self) -> dict[str, Any]:
        """Describe the runtime sparse index for experiment manifests."""
        return {
            "method": "bm25_okapi",
            "implementation": "rank-bm25",
            "implementation_version": version("rank-bm25"),
            "tokenizer_version": SPARSE_TOKENIZER_VERSION,
            "document_count": self.document_count,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
        }


def build_sparse_retriever(vector_store: Any) -> SparseRetriever:
    """Build a deterministic runtime BM25 index from a Chroma collection."""
    return SparseRetriever(load_sparse_documents(vector_store))
