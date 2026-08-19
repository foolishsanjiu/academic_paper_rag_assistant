"""Streamlit interface for the Academic Paper RAG Assistant."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config import get_settings
from document_loader import load_pdf_directory
from knowledge_base import (
    DEFAULT_PAPER_DIRECTORY,
    get_paper_count,
    get_paper_names,
)
from llm_client import LLMClient
from logging_config import setup_logging
from rag_chain import RAGChain
from text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents,
)
from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    build_vector_store,
    create_embedding_model,
    get_vector_count,
    load_vector_store,
)


# ============================================================
# Basic configuration
# ============================================================

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Academic Paper RAG Assistant",
    page_icon="📚",
    layout="centered",
)


WELCOME_MESSAGE = (
    "你好，我是 Academic Paper RAG Assistant。"
    "当前已经接入本地 SAR 论文知识库，"
    "可以基于论文内容进行检索增强问答，并显示论文来源和 PDF 页码。"
)

MAX_UPLOAD_SIZE_MB = 50


# ============================================================
# Session state
# ============================================================


def initialize_messages() -> None:
    """Initialize chat history for the current browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "is_welcome": True,
            }
        ]


def clear_messages() -> None:
    """Reset chat history while keeping the welcome message."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "is_welcome": True,
        }
    ]


def build_chat_history_for_rag() -> list[dict]:
    """
    Build clean conversation history for query rewriting.

    Welcome messages and error messages are excluded.
    """
    history: list[dict] = []

    for message in st.session_state.messages:
        if message.get("is_welcome"):
            continue

        if message.get("is_error"):
            continue

        role = message.get("role")
        content = message.get("content")

        if role in {"user", "assistant"} and content:
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return history


# ============================================================
# Knowledge-base upload / rebuild helpers
# ============================================================


def validate_uploaded_pdf(uploaded_file) -> tuple[bool, str]:
    """Validate extension, size, and basic PDF file signature."""
    file_name = uploaded_file.name

    if Path(file_name).suffix.lower() != ".pdf":
        return False, f"{file_name}: 不是 PDF 文件。"

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        return False, f"{file_name}: 文件为空。"

    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        return (
            False,
            f"{file_name}: 文件大小为 {file_size_mb:.1f} MB，"
            f"超过 {MAX_UPLOAD_SIZE_MB} MB 限制。",
        )

    # Most normal PDF files start with b"%PDF-".
    if not file_bytes.startswith(b"%PDF-"):
        return False, f"{file_name}: 文件头不是有效的 PDF 标识。"

    return True, ""


def save_uploaded_pdfs(
    uploaded_files,
    overwrite_existing: bool,
) -> tuple[list[str], list[str]]:
    """Save validated uploaded PDFs into the local paper directory."""
    saved_files: list[str] = []
    errors: list[str] = []

    DEFAULT_PAPER_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for uploaded_file in uploaded_files:
        is_valid, error_message = validate_uploaded_pdf(uploaded_file)

        if not is_valid:
            errors.append(error_message)
            continue

        destination = DEFAULT_PAPER_DIRECTORY / Path(uploaded_file.name).name

        if destination.exists() and not overwrite_existing:
            errors.append(
                f"{uploaded_file.name}: 知识库中已存在同名文件，"
                "未覆盖。"
            )
            continue

        try:
            destination.write_bytes(
                uploaded_file.getvalue()
            )
            saved_files.append(
                destination.name
            )

        except OSError as error:
            logger.exception(
                "保存上传 PDF 失败 | file=%s",
                uploaded_file.name,
            )
            errors.append(
                f"{uploaded_file.name}: 保存失败：{error}"
            )

    return saved_files, errors


def rebuild_knowledge_base(
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
    Reload all PDFs, split them with the selected chunk parameters,
    and rebuild the persistent Chroma vector store.

    Returns:
        Number of chunk records written into the vector store.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk Overlap 必须小于 Chunk Size。"
        )

    paper_count = get_paper_count()
    if paper_count == 0:
        raise ValueError(
            "论文目录中没有 PDF，无法重建向量库。"
        )

    page_documents = load_pdf_directory(
        DEFAULT_PAPER_DIRECTORY
    )

    if not page_documents:
        raise ValueError(
            "PDF 解析结果为空，无法重建向量库。"
        )

    chunk_documents = split_documents(
        page_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunk_documents:
        raise ValueError(
            "文本切分结果为空，无法重建向量库。"
        )

    embeddings = create_embedding_model()

    vector_store = build_vector_store(
        documents=chunk_documents,
        embeddings=embeddings,
        reset=True,
    )

    vector_count = get_vector_count(
        vector_store
    )

    # Force Streamlit to reopen the newly rebuilt Chroma store on rerun.
    load_rag_resources.clear()

    return vector_count


# ============================================================
# RAG resources
# ============================================================


@st.cache_resource(show_spinner=False)
def load_rag_resources():
    """
    Load expensive resources only once.

    Returns:
        settings:
            Application configuration.

        llm:
            DeepSeek-compatible LLM client.

        vector_store:
            Persistent Chroma vector database.

        vector_count:
            Number of indexed chunk records.
    """
    settings = get_settings()

    llm = LLMClient(
        settings
    )

    embeddings = create_embedding_model()

    vector_store = load_vector_store(
        embeddings=embeddings
    )

    vector_count = get_vector_count(
        vector_store
    )

    return (
        settings,
        llm,
        vector_store,
        vector_count,
    )


# ============================================================
# Source display
# ============================================================


def serialize_sources(
    sources,
) -> list[dict]:
    """
    Convert RAGSource objects into dictionaries that can be
    safely stored in Streamlit session_state.
    """
    serialized_sources: list[dict] = []

    for source in sources:
        serialized_sources.append(
            {
                "rank": source.rank,
                "file_name": source.file_name,
                "page_number": source.page_number,
                "chunk_id": source.chunk_id,
                "similarity": source.similarity,
                "text": source.text,
            }
        )

    return serialized_sources


def render_sources(
    sources: list[dict],
) -> None:
    """Display retrieved paper chunks below an assistant answer."""
    if not sources:
        return

    with st.expander(
        f"查看检索来源（{len(sources)}）"
    ):
        for source in sources:
            rank = source.get(
                "rank",
                "?",
            )

            file_name = source.get(
                "file_name",
                "unknown.pdf",
            )

            page_number = source.get(
                "page_number",
                "unknown",
            )

            chunk_id = source.get(
                "chunk_id",
                "unknown",
            )

            similarity = source.get(
                "similarity",
            )

            text = source.get(
                "text",
                "",
            )

            st.markdown(
                f"### [{rank}] {file_name}"
            )

            st.markdown(
                f"**PDF Page:** {page_number}"
            )

            if similarity is not None:
                st.caption(
                    f"Chunk ID: {chunk_id}  |  "
                    f"Cosine similarity: {similarity:.4f}"
                )
            else:
                st.caption(
                    f"Chunk ID: {chunk_id}"
                )

            st.text(
                text
            )

            st.divider()


def render_retrieval_query(
    retrieval_query: str | None,
) -> None:
    """Display the query actually used for vector retrieval."""
    if not retrieval_query:
        return

    with st.expander(
        "查看实际检索 Query"
    ):
        st.code(
            retrieval_query,
            language=None,
        )


# ============================================================
# Initialize application
# ============================================================


initialize_messages()


try:
    with st.spinner(
        "正在加载 BGE-M3 和论文向量数据库……"
    ):
        (
            settings,
            llm,
            vector_store,
            vector_count,
        ) = load_rag_resources()

except Exception as error:
    logger.exception(
        "RAG 资源初始化失败"
    )

    st.error(
        "RAG 系统初始化失败："
        f"{error}"
    )

    st.stop()


# ============================================================
# Main page
# ============================================================


st.title(
    "📚 Academic Paper RAG Assistant"
)

st.caption(
    "基于本地 SAR 论文知识库的检索增强问答系统"
)


# ============================================================
# Sidebar
# ============================================================


with st.sidebar:
    st.header(
        "RAG 设置"
    )

    # --------------------------------------------------------
    # Model / knowledge-base status
    # --------------------------------------------------------

    st.markdown(
        "**生成模型**"
    )
    st.code(
        settings.model,
        language=None,
    )

    st.markdown(
        "**Embedding 模型**"
    )
    st.code(
        DEFAULT_EMBEDDING_MODEL,
        language=None,
    )

    current_paper_count = get_paper_count()
    current_paper_names = get_paper_names()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "论文数",
            current_paper_count,
        )

    with col2:
        st.metric(
            "Chunk 数",
            vector_count,
        )

    with st.expander(
        "查看知识库论文"
    ):
        if current_paper_names:
            for index, paper_name in enumerate(
                current_paper_names,
                start=1,
            ):
                st.write(
                    f"{index}. {paper_name}"
                )
        else:
            st.caption(
                "当前论文目录为空。"
            )

    st.divider()

    # --------------------------------------------------------
    # Retrieval / generation parameters
    # --------------------------------------------------------

    st.subheader(
        "问答参数"
    )

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help=(
            "控制每次从 Chroma 中检索的相关论文 Chunk 数量。"
        ),
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help=(
            "控制回答随机性。学术 RAG 建议使用较低值，"
            "例如 0.0～0.2。"
        ),
    )

    st.caption(
        "Top-k 会立即影响下一次检索；Temperature 会立即影响下一次生成。"
    )

    st.divider()

    # --------------------------------------------------------
    # Index parameters
    # --------------------------------------------------------

    st.subheader(
        "索引参数"
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1200,
        value=DEFAULT_CHUNK_SIZE,
        step=100,
        help=(
            "文本切分长度。修改后必须重建向量库才会生效。"
        ),
    )

    max_overlap = min(
        300,
        chunk_size - 1,
    )

    default_overlap = min(
        DEFAULT_CHUNK_OVERLAP,
        max_overlap,
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=max_overlap,
        value=default_overlap,
        step=50,
        help=(
            "相邻 Chunk 的重叠长度。修改后必须重建向量库才会生效。"
        ),
    )

    st.caption(
        "当前索引参数只有在点击“重建向量库”后才会真正写入 Chroma。"
    )

    st.divider()

    # --------------------------------------------------------
    # PDF upload
    # --------------------------------------------------------

    st.subheader(
        "论文管理"
    )

    uploaded_files = st.file_uploader(
        "上传 PDF 论文",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            f"支持一次上传多篇 PDF；单个文件建议不超过 {MAX_UPLOAD_SIZE_MB} MB。"
        ),
    )

    overwrite_existing = st.checkbox(
        "允许覆盖同名 PDF",
        value=False,
    )

    if st.button(
        "保存 PDF 并重建向量库",
        use_container_width=True,
        type="primary",
    ):
        if not uploaded_files:
            st.warning(
                "请先选择至少一个 PDF 文件。"
            )
        else:
            try:
                saved_files, upload_errors = save_uploaded_pdfs(
                    uploaded_files=uploaded_files,
                    overwrite_existing=overwrite_existing,
                )

                for error_message in upload_errors:
                    st.warning(
                        error_message
                    )

                if not saved_files:
                    st.error(
                        "没有成功保存任何 PDF，因此未重建向量库。"
                    )
                else:
                    with st.spinner(
                        "正在解析 PDF、切分文本并重建 Chroma……"
                    ):
                        new_vector_count = rebuild_knowledge_base(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                    st.success(
                        "已保存 "
                        f"{len(saved_files)} 篇 PDF，"
                        f"并重建向量库，共 {new_vector_count} 个 Chunk。"
                    )

                    st.rerun()

            except Exception as error:
                logger.exception(
                    "上传 PDF 并重建知识库失败"
                )
                st.error(
                    "上传/重建失败："
                    f"{error}"
                )

    if st.button(
        "按当前 Chunk 参数重建向量库",
        use_container_width=True,
        help=(
            "不新增论文，只使用当前论文目录中的 PDF "
            "按新的 Chunk Size / Overlap 重新建立 Chroma。"
        ),
    ):
        try:
            with st.spinner(
                "正在重新解析论文并重建向量库……"
            ):
                new_vector_count = rebuild_knowledge_base(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            st.success(
                "向量库重建完成，"
                f"共 {new_vector_count} 个 Chunk。"
            )

            st.rerun()

        except Exception as error:
            logger.exception(
                "重建知识库失败"
            )
            st.error(
                "重建失败："
                f"{error}"
            )

    st.divider()

    # --------------------------------------------------------
    # Session / cache controls
    # --------------------------------------------------------

    if st.button(
        "清空对话",
        use_container_width=True,
    ):
        clear_messages()
        st.rerun()

    if st.button(
        "重新加载模型与向量库",
        use_container_width=True,
        help=(
            "清除 Streamlit 资源缓存，重新加载 LLM、"
            "Embedding 和 Chroma。"
        ),
    ):
        load_rag_resources.clear()
        st.rerun()

    st.divider()

    st.info(
        "当前回答仅允许依据本地论文知识库中的检索内容生成。"
        "知识库中没有足够信息时应明确拒绝回答，不编造论文内容。"
    )


# ============================================================
# Create RAG Chain
# ============================================================


rag = RAGChain(
    llm=llm,
    vector_store=vector_store,
    top_k=top_k,
)


# ============================================================
# Display previous messages
# ============================================================


for message in st.session_state.messages:
    role = message["role"]

    with st.chat_message(role):
        if message.get("is_error"):
            st.error(
                message["content"]
            )
        else:
            st.markdown(
                message["content"]
            )

        # Only RAG assistant messages contain these fields.
        if role == "assistant":
            render_retrieval_query(
                message.get(
                    "retrieval_query"
                )
            )

            render_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# ============================================================
# User input
# ============================================================


prompt = st.chat_input(
    (
        "向论文知识库提问，例如："
        "SARGAN 使用了什么数据集？"
    ),
    max_chars=2000,
)


if prompt:
    # --------------------------------------------------------
    # Important:
    #
    # Save the previous history BEFORE inserting the current
    # user question.
    #
    # rewrite_query() needs previous conversation + current
    # question separately.
    # --------------------------------------------------------

    chat_history = build_chat_history_for_rag()

    # --------------------------------------------------------
    # Display and save current user message
    # --------------------------------------------------------

    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            prompt
        )

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    try:
        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "正在检索论文并生成回答……"
            ):
                rag_response = rag.ask(
                    question=prompt,
                    chat_history=chat_history,
                    temperature=temperature,
                )

            # -----------------------------------------------
            # Answer
            # -----------------------------------------------

            st.markdown(
                rag_response.answer
            )

            # -----------------------------------------------
            # Retrieval query
            # -----------------------------------------------

            render_retrieval_query(
                rag_response.retrieval_query
            )

            # -----------------------------------------------
            # Sources
            # -----------------------------------------------

            serialized_sources = serialize_sources(
                rag_response.sources
            )

            render_sources(
                serialized_sources
            )

        # ----------------------------------------------------
        # Save assistant message into session_state
        # ----------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": rag_response.answer,
            "retrieval_query": rag_response.retrieval_query,
            "sources": serialized_sources,
        }

        st.session_state.messages.append(
            assistant_message
        )

    except (
        ValueError,
        RuntimeError,
    ) as error:
        logger.error(
            "RAG 调用失败 | error=%s",
            error,
        )

        error_message = (
            f"RAG 调用失败：{error}"
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_message
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "is_error": True,
            }
        )

    except Exception as error:
        logger.exception(
            "RAG 页面发生未预期异常"
        )

        error_message = (
            "RAG 系统运行时发生未预期错误："
            f"{error}"
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_message
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "is_error": True,
            }
        )