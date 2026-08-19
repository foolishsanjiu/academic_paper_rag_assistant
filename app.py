"""Streamlit interface for the Academic Paper RAG Assistant."""

import logging

import streamlit as st

from config import get_settings
from llm_client import LLMClient
from logging_config import setup_logging
from rag_chain import RAGChain
from vector_store import (
    DEFAULT_EMBEDDING_MODEL,
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


# ============================================================
# Session state
# ============================================================


def initialize_messages() -> None:
    """
    Initialize chat history for the current browser session.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "is_welcome": True,
            }
        ]


def clear_messages() -> None:
    """
    Reset chat history while keeping the welcome message.
    """
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

        if (
            role in {"user", "assistant"}
            and content
        ):
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return history


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

    embeddings = (
        create_embedding_model()
    )

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
    """
    Display retrieved paper chunks below an assistant answer.
    """
    if not sources:
        return

    with st.expander(
        f"查看检索来源（{len(sources)}）"
    ):

        for source in sources:

            rank = source.get(
                "rank",
                "?"
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
                    f"Cosine similarity: "
                    f"{similarity:.4f}"
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
    """
    Display the query actually used for vector retrieval.
    """
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

    st.markdown(
        "**向量数据库记录数**"
    )

    st.write(
        vector_count
    )

    st.divider()

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help=(
            "控制每次从 Chroma 中检索的 "
            "相关论文 Chunk 数量。"
        ),
    )

    st.caption(
        "当前默认 Chunk Size = 600，"
        "Chunk Overlap = 100。"
    )

    st.divider()

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
            "重新构建 Chroma 后，可以点击此按钮"
            "清除缓存并重新加载数据库。"
        ),
    ):

        load_rag_resources.clear()

        st.rerun()

    st.divider()

    st.info(
        "当前回答仅允许依据本地论文知识库中的"
        "检索内容生成。"
        "知识库中没有足够信息时应拒绝回答。"
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

    chat_history = (
        build_chat_history_for_rag()
    )

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

                rag_response = (
                    rag.ask(
                        question=prompt,
                        chat_history=chat_history,
                    )
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

            serialized_sources = (
                serialize_sources(
                    rag_response.sources
                )
            )

            render_sources(
                serialized_sources
            )

        # ----------------------------------------------------
        # Save assistant message into session_state
        #
        # This is necessary because Streamlit reruns app.py
        # after every new interaction.
        # ----------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": rag_response.answer,
            "retrieval_query": (
                rag_response.retrieval_query
            ),
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