"""Embedding model and Chroma vector-store utilities."""

from pathlib import Path
import shutil
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_COLLECTION_NAME = "academic_papers"

DEFAULT_PERSIST_DIRECTORY = (
    PROJECT_ROOT / "chroma_db"
)


def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Create the local embedding model.

    Args:
        model_name:
            Hugging Face model name.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    print(
        f"正在加载 Embedding 模型：{model_name}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embeddings


def sanitize_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """
    Convert metadata into simple values suitable for Chroma.

    None values and complex Python objects are converted
    into strings.

    Args:
        metadata:
            Metadata from a chunk Document.

    Returns:
        Sanitized metadata dictionary.
    """
    sanitized: dict[
        str,
        str | int | float | bool,
    ] = {}

    for key, value in metadata.items():

        if value is None:
            sanitized[key] = ""

        elif isinstance(
            value,
            (str, int, float, bool),
        ):
            sanitized[key] = value

        elif isinstance(value, list):
            sanitized[key] = " | ".join(
                str(item)
                for item in value
            )

        else:
            sanitized[key] = str(value)

    return sanitized


def prepare_documents_for_chroma(
    documents: list[Document],
) -> tuple[list[Document], list[str]]:
    """
    Prepare chunk Documents before inserting them into Chroma.

    Metadata is sanitized and chunk_id is used as
    the persistent vector-store ID.

    Args:
        documents:
            Chunk-level Documents.

    Returns:
        Tuple containing:
        - sanitized Documents
        - corresponding unique IDs
    """
    prepared_documents: list[Document] = []
    ids: list[str] = []

    for index, document in enumerate(documents):

        metadata = sanitize_metadata(
            document.metadata
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
                f"chunk_{index}",
            )
        )

        metadata["chunk_id"] = chunk_id

        prepared_document = Document(
            page_content=document.page_content,
            metadata=metadata,
            id=chunk_id,
        )

        prepared_documents.append(
            prepared_document
        )

        ids.append(chunk_id)

    if len(ids) != len(set(ids)):
        raise ValueError(
            "检测到重复的 chunk_id，"
            "无法安全写入 Chroma。"
        )

    return prepared_documents, ids


def build_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = False,
) -> Chroma:
    """
    Create a persistent Chroma vector database.

    Args:
        documents:
            Chunk-level Documents.

        embeddings:
            Embedding model.

        persist_directory:
            Directory used to store Chroma data.

        collection_name:
            Chroma collection name.

        reset:
            Delete the existing database before rebuilding.

    Returns:
        Chroma vector store.
    """
    if not documents:
        raise ValueError(
            "没有可写入向量数据库的 Document。"
        )

    if reset and persist_directory.exists():
        print(
            f"删除旧向量数据库："
            f"{persist_directory}"
        )

        shutil.rmtree(
            persist_directory
        )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_documents, ids = (
        prepare_documents_for_chroma(
            documents
        )
    )

    print(
        f"准备写入 {len(prepared_documents)} "
        "个 Chunk..."
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(
            persist_directory
        ),

        # 显式使用 cosine distance。
        collection_configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    vector_store.add_documents(
        documents=prepared_documents,
        ids=ids,
    )

    print(
        f"Chroma 数据库已保存至："
        f"{persist_directory}"
    )

    return vector_store


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    Reload an existing persistent Chroma database.

    Args:
        embeddings:
            The same embedding model used when indexing.

        persist_directory:
            Existing Chroma database directory.

        collection_name:
            Chroma collection name.

    Returns:
        Loaded Chroma vector store.
    """
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma 数据库不存在："
            f"{persist_directory}"
        )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(
            persist_directory
        ),
    )

    return vector_store


def get_vector_count(
    vector_store: Chroma,
) -> int:
    """
    Return the number of records stored in Chroma.
    """
    result = vector_store.get()

    return len(
        result.get("ids", [])
    )