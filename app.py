"""Streamlit interface for the Academic Paper RAG Assistant."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from agent_response import format_tool_response, source_lookup_sources
from agent_router import Intent, dispatch_route, route_query
from config import (
    CHROMA_DIRECTORY,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    MAX_UPLOAD_SIZE_MB,
    get_settings,
)
from document_loader import load_pdf_directory
from knowledge_base import (
    DEFAULT_PAPER_DIRECTORY,
    get_paper_count,
    get_paper_names,
    save_uploaded_pdf
)
from llm_client import LLMClient
from logging_config import setup_logging
from rag_chain import RAGChain
from retrieval_pipeline import RetrievalMethod
from retriever import RetrievalStrategy
from sparse_retriever import build_sparse_retriever
from text_splitter import split_documents
from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    build_vector_store,
    create_embedding_model,
    get_vector_count,
    load_vector_store,
)
import re

from index_manifest import (
    load_index_manifest,
    validate_index_manifest,
    write_index_manifest,
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
    "可以回答论文内容、查询知识库状态、定位 PDF 原文，"
    "并显示论文来源和页码。"
)

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


def rebuild_knowledge_base(
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
    Reload all PDFs, split them with the selected chunk parameters,
    rebuild the persistent Chroma vector store,
    and update the index manifest.

    Returns:
        Number of chunk records written into the vector store.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk Overlap 必须小于 Chunk Size。"
        )

    logger.info(
        "Knowledge-base rebuild started | chunk_size=%s | "
        "chunk_overlap=%s",
        chunk_size,
        chunk_overlap,
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

    # --------------------------------------------------------
    # Verify rebuild result
    # --------------------------------------------------------

    if vector_count != len(chunk_documents):
        raise RuntimeError(
            "索引构建失败："
            "Chunk 数量与 Chroma 数量不一致。"
        )

    # --------------------------------------------------------
    # Update index manifest
    # --------------------------------------------------------

    write_index_manifest(
        paper_count=paper_count,
        page_count=len(
            page_documents
        ),
        chunk_count=len(
            chunk_documents
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=(
            DEFAULT_EMBEDDING_MODEL
        ),
    )

    # Force Streamlit to reopen the newly rebuilt
    # Chroma store and rebuild the runtime BM25 index on rerun.
    load_rag_resources.clear()
    load_sparse_resources.clear()

    logger.info(
        "Knowledge-base rebuild completed | papers=%s | pages=%s | "
        "chunks=%s",
        paper_count,
        len(page_documents),
        vector_count,
    )

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
    logger.info("RAG resource load started")
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

    manifest = load_index_manifest()
    if manifest is None:
        raise FileNotFoundError(
            "缺少 chroma_db/index_manifest.json。"
        )
    validate_index_manifest(
        vector_count=vector_count,
        manifest=manifest,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )

    logger.info(
        "RAG resource load completed | vector_count=%s",
        vector_count,
    )

    return (
        settings,
        llm,
        vector_store,
        vector_count,
    )


@st.cache_resource(show_spinner=False)
def load_sparse_resources():
    """Build and cache the runtime BM25 index from current Chroma text."""
    from langchain_chroma import Chroma

    metadata_store = Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIRECTORY),
        embedding_function=None,
    )
    return build_sparse_retriever(metadata_store)


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
                "document_id": source.document_id,
                "document_type": source.document_type,
                "file_name": source.file_name,
                "page_number": source.page_number,
                "chunk_id": source.chunk_id,
                "similarity": source.similarity,
                "bm25_score": source.bm25_score,
                "rrf_score": source.rrf_score,
                "dense_rank": source.dense_rank,
                "sparse_rank": source.sparse_rank,
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
            bm25_score = source.get("bm25_score")
            rrf_score = source.get("rrf_score")
            dense_rank = source.get("dense_rank")
            sparse_rank = source.get("sparse_rank")

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

            score_parts = [f"Chunk ID: {chunk_id}"]
            if rrf_score is not None:
                score_parts.append(f"RRF: {rrf_score:.6f}")
            if similarity is not None:
                score_parts.append(f"Cosine: {similarity:.4f}")
            if bm25_score is not None:
                score_parts.append(f"BM25: {bm25_score:.4f}")
            if dense_rank is not None:
                score_parts.append(f"Dense rank: {dense_rank}")
            if sparse_rank is not None:
                score_parts.append(f"BM25 rank: {sparse_rank}")
            st.caption("  |  ".join(score_parts))

            st.text(
                text
            )

            st.divider()

def extract_cited_ranks(
    answer: str,
) -> set[int]:
    """
    Extract citation numbers such as [1], [2], [3]
    from the generated RAG answer.
    """
    matches = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    return {
        int(match)
        for match in matches
    }

def get_cited_sources(
    answer: str,
    sources: list[dict],
) -> list[dict]:
    """
    Return only sources explicitly cited in the answer.
    """
    cited_ranks = extract_cited_ranks(
        answer
    )

    return [
        source
        for source in sources
        if source.get("rank")
        in cited_ranks
    ]

def render_citation_summary(
    answer: str,
    sources: list[dict],
) -> None:

    cited_sources = get_cited_sources(
        answer=answer,
        sources=sources,
    )

    if not cited_sources:
        return

    st.markdown(
        "**回答引用来源**"
    )

    for source in cited_sources:

        st.markdown(
            f"- [{source['rank']}] "
            f"`{source['file_name']}` "
            f"— PDF Page "
            f"{source['page_number']}"
        )

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

def render_rag_parameters(
    parameters: dict | None,
) -> None:
    """
    Display the parameters actually used
    for one RAG answer.
    """
    if not parameters:
        return

    with st.expander(
        "查看本轮 RAG 参数"
    ):

        st.write(
            "Top-k：",
            parameters.get(
                "top_k"
            ),
        )

        st.write(
            "Temperature：",
            parameters.get(
                "temperature"
            ),
        )

        st.write(
            "Chunk Size：",
            parameters.get(
                "chunk_size"
            ),
        )

        st.write(
            "Chunk Overlap：",
            parameters.get(
                "chunk_overlap"
            ),
        )

        st.write(
            "Embedding Model：",
            parameters.get(
                "embedding_model"
            ),
        )

        if parameters.get("retrieval_method"):
            st.write("Retrieval Method：", parameters["retrieval_method"])

        if parameters.get("retrieval_diagnostics"):
            st.write(
                "Retrieval Diagnostics：",
                parameters["retrieval_diagnostics"],
            )


def render_agent_route(message: dict) -> None:
    """Show the validated route used for an assistant response."""
    intent = message.get("intent")
    if not intent:
        return
    details = [f"Intent: {intent}"]
    if message.get("retrieval_strategy"):
        details.append(f"Strategy: {message['retrieval_strategy']}")
    if message.get("retrieval_method"):
        details.append(f"Method: {message['retrieval_method']}")
    if message.get("tool_name"):
        details.append(f"Tool: {message['tool_name']}")
    st.caption(" · ".join(details))

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

# ------------------------------------------------------------
# Load current index manifest
# ------------------------------------------------------------

index_manifest = load_index_manifest()

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

    retrieval_method = RetrievalMethod(
        st.selectbox(
            "检索方法",
            options=[item.value for item in RetrievalMethod],
            index=0,
            help=(
                "Dense 为默认基线；BM25 使用关键词匹配；"
                "Hybrid 使用 Dense + BM25 的 RRF 融合。"
            ),
        )
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

    # --------------------------------------------------------
    # Current actual index parameters
    # --------------------------------------------------------

    if index_manifest:
        st.markdown(
            "**当前索引实际参数**"
        )

        st.write(
            "Chunk Size：",
            index_manifest[
                "chunk_size"
            ],
        )

        st.write(
            "Chunk Overlap：",
            index_manifest[
                "chunk_overlap"
            ],
        )

        st.write(
            "Chunk 数量：",
            index_manifest[
                "chunk_count"
            ],
        )

        st.caption(
            "构建时间："
            f"{index_manifest['built_at']}"
        )

    else:
        st.warning(
            "暂未找到索引 Manifest，"
            "无法确定当前向量库的实际构建参数。"
        )

    # --------------------------------------------------------
    # Rebuild parameters
    # --------------------------------------------------------


    st.markdown(
        "**重建参数**"
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
            "相邻 Chunk 的重叠长度。"
            "修改后必须重建向量库才会生效。"
        ),
    )

    # --------------------------------------------------------
    # Warn if Slider parameters differ from current index
    # --------------------------------------------------------

    if index_manifest:

        index_chunk_size = (
            index_manifest[
                "chunk_size"
            ]
        )

        index_chunk_overlap = (
            index_manifest[
                "chunk_overlap"
            ]
        )

        if (
            chunk_size
            != index_chunk_size
            or
            chunk_overlap
            != index_chunk_overlap
        ):
            st.warning(
                "Chunk 参数已经修改，"
                "但当前 Chroma 仍使用旧参数。"
                "请重建知识库后再使新参数生效。"
            )

    st.caption(
        "以上 Slider 是下一次重建向量库时使用的参数，"
        "不会立即改变当前 Chroma 索引。"
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
            f"支持一次上传多篇 PDF；"
            f"单个文件不能超过 {MAX_UPLOAD_SIZE_MB} MB。"
        ),
    )

    rebuild_requested = st.button(
        "保存所选 PDF / 重建向量库",
        use_container_width=True,
        type="primary",
    )

    if rebuild_requested:
        successful_uploads = []
        failed_uploads = []

        # ----------------------------------------------------
        # Save and validate each PDF independently
        # ----------------------------------------------------

        for uploaded_file in uploaded_files or []:
            try:
                file_path, validation = save_uploaded_pdf(
                    file_name=uploaded_file.name,
                    data=uploaded_file.getvalue(),
                )

                successful_uploads.append(
                    {
                        "file_name": validation.file_name,
                        "document_id": validation.document_id,
                        "page_count": validation.page_count,
                        "text_char_count": validation.text_char_count,
                    }
                )

            except (
                ValueError,
                FileExistsError,
                OSError,
            ) as error:
                failed_uploads.append(
                    {
                        "file_name": uploaded_file.name,
                        "error": str(error),
                    }
                )

        # ----------------------------------------------------
        # Display upload results
        # ----------------------------------------------------

        for item in successful_uploads:
            st.success(
                "上传成功："
                f"{item['file_name']} "
                f"({item['page_count']} pages)"
            )

        for item in failed_uploads:
            st.error(
                f"{item['file_name']}："
                f"{item['error']}"
            )

        if not uploaded_files:
            st.info("未选择新 PDF，将使用当前论文目录重建向量库。")

        try:
            with st.spinner(
                "正在解析 PDF、切分文本并重建 Chroma……"
            ):
                new_vector_count = rebuild_knowledge_base(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            if successful_uploads:
                st.success(
                    f"已成功保存 {len(successful_uploads)} 篇 PDF，"
                    f"并重建向量库，共 {new_vector_count} 个 Chunk。"
                )
            else:
                st.success(
                    "已使用当前论文目录重建向量库，"
                    f"共 {new_vector_count} 个 Chunk。"
                )

            st.rerun()

        except Exception as error:
            logger.exception(
                "重建知识库失败"
            )

            failure_prefix = (
                "PDF 已保存，但向量库重建失败："
                if successful_uploads
                else "向量库重建失败："
            )
            st.error(
                f"{failure_prefix}{error}。"
                "可不选择新文件，直接再次点击重建按钮。"
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
            "Embedding、Chroma 和 BM25。"
        ),
    ):
        load_rag_resources.clear()
        load_sparse_resources.clear()
        st.rerun()

    st.divider()

    st.info(
        "当前回答仅允许依据本地论文知识库中的检索内容生成。"
        "知识库中没有足够信息时应明确拒绝回答，不编造论文内容。"
    )


# ============================================================
# Create RAG Chain
# ============================================================


sparse_retriever = None
if retrieval_method is not RetrievalMethod.DENSE:
    with st.spinner("正在加载 BM25 索引……"):
        sparse_retriever = load_sparse_resources()


focused_rag = RAGChain(
    llm=llm,
    vector_store=vector_store,
    top_k=top_k,
    retrieval_strategy=RetrievalStrategy.FOCUSED,
    retrieval_method=retrieval_method,
    sparse_retriever=sparse_retriever,
)

multi_document_rag = RAGChain(
    llm=llm,
    vector_store=vector_store,
    retrieval_strategy=RetrievalStrategy.MULTI_DOCUMENT,
    retrieval_method=retrieval_method,
    sparse_retriever=sparse_retriever,
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

        if role == "assistant":

            render_agent_route(message)

            sources = message.get(
                "sources",
                [],
            )

            # 先显示回答真正引用的来源
            render_citation_summary(
                answer=message["content"],
                sources=sources,
            )
            # 显示本轮实际使用的 RAG 参数
            render_rag_parameters(
            message.get(
                "rag_parameters"
            )
)

            # 再显示实际 Retrieval Query
            render_retrieval_query(
                message.get(
                    "retrieval_query"
                )
            )

            # 最后显示全部 Top-k
            render_sources(
                sources
            )


# ============================================================
# User input
# ============================================================


prompt = st.chat_input(
    (
        "提问论文内容、知识库状态或指定 PDF 页码原文"
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
    # Route and execute RAG / Tool
    # --------------------------------------------------------

    decision = None
    route_message = {}

    try:
        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "正在判断意图并执行论文助手……"
            ):
                decision = route_query(prompt, llm)
                route_message = {
                    "intent": decision.intent.value,
                    "retrieval_strategy": (
                        decision.retrieval_strategy.value
                        if decision.retrieval_strategy
                        else None
                    ),
                    "tool_name": decision.tool_name,
                    "retrieval_method": (
                        retrieval_method.value
                        if decision.intent is Intent.PAPER_QA
                        else None
                    ),
                }
                result = dispatch_route(
                    decision,
                    prompt,
                    rag_handlers={
                        RetrievalStrategy.FOCUSED: lambda question: focused_rag.ask(
                            question=question,
                            chat_history=chat_history,
                            temperature=temperature,
                        ),
                        RetrievalStrategy.MULTI_DOCUMENT: lambda question: multi_document_rag.ask(
                            question=question,
                            chat_history=chat_history,
                            temperature=temperature,
                        ),
                    },
                )

            if decision.intent is Intent.PAPER_QA:
                answer = result.answer
                retrieval_query = result.retrieval_query
                serialized_sources = serialize_sources(result.sources)
                is_error = False
            else:
                answer = format_tool_response(result)
                retrieval_query = None
                serialized_sources = source_lookup_sources(result)
                is_error = (
                    not result.get("ok", False)
                    and decision.intent is not Intent.OUT_OF_SCOPE
                )

            if is_error:
                st.error(answer)
            else:
                st.markdown(answer)

            render_agent_route(route_message)

            rag_parameters = ({
                "top_k": result.retrieval_diagnostics["top_k"],
                "temperature": temperature,
                "retrieval_method": result.retrieval_method,
                "retrieval_diagnostics": result.retrieval_diagnostics,
                "chunk_size": (
                    index_manifest.get("chunk_size")
                    if index_manifest else None
                ),
                "chunk_overlap": (
                    index_manifest.get("chunk_overlap")
                    if index_manifest else None
                ),
                "embedding_model": (
                    index_manifest.get("embedding_model")
                    if index_manifest else None
                ),
            } if decision.intent is Intent.PAPER_QA else None)

            render_rag_parameters(rag_parameters)

            # -----------------------------------------------
            # Retrieval query
            # -----------------------------------------------

            render_retrieval_query(
                retrieval_query
            )

            # -----------------------------------------------
            # Sources
            # -----------------------------------------------

            # 显示回答真正引用的论文和页码
            render_citation_summary(
                answer=answer,
                sources=serialized_sources,
            )


            render_sources(
                serialized_sources
            )

        # ----------------------------------------------------
        # Save assistant message into session_state
        # ----------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "retrieval_query": retrieval_query,
            "sources": serialized_sources,
            "is_error": is_error,
            **route_message,

            "rag_parameters": rag_parameters,
        }

        st.session_state.messages.append(
            assistant_message
        )

    except (
        ValueError,
        RuntimeError,
    ) as error:
        logger.error(
            "Agent 调用失败 | error=%s",
            error,
        )

        error_message = (
            f"Agent 调用失败：{error}"
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_message
            )
            render_agent_route(route_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "is_error": True,
                **route_message,
            }
        )

    except Exception as error:
        logger.exception(
            "Agent 页面发生未预期异常"
        )

        error_message = (
            "Agent 运行时发生未预期错误："
            f"{error}"
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_message
            )
            render_agent_route(route_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "is_error": True,
                **route_message,
            }
        )
