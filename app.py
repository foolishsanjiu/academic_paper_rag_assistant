"""Streamlit interface for the academic paper assistant."""

import logging

import streamlit as st

from config import get_settings
from llm_client import LLMClient
from logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "你好，我是 Academic Paper RAG Assistant 的基础聊天版本。"
    "当前仅调用大语言模型，尚未连接论文知识库。"
)


def initialize_messages() -> None:
    """Initialize the chat history for the current browser session."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]


def clear_messages() -> None:
    """Reset the chat history."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


st.set_page_config(
    page_title="Academic Paper RAG Assistant",
    page_icon="📚",
    layout="centered",
)

initialize_messages()

st.title("📚 Academic Paper RAG Assistant")
st.caption("第 5 天版本：普通大模型聊天，尚未接入论文检索。")

try:
    settings = get_settings()
    llm = LLMClient(settings)

except ValueError as error:
    logger.error("配置加载失败 | message=%s", error)
    st.error(f"配置加载失败：{error}")
    st.stop()


with st.sidebar:
    st.header("运行设置")

    st.write(f"当前模型：`{settings.model}`")

    use_stream = st.toggle(
        "流式输出",
        value=True,
        help="开启后，模型回答会逐步显示。",
    )

    if st.button(
        "清空对话",
        use_container_width=True,
    ):
        clear_messages()
        st.rerun()

    st.divider()

    st.info(
        "当前版本没有读取 PDF，也没有执行向量检索。"
        "回答完全来自大模型自身。"
    )


# 每次 Streamlit 重新运行脚本时，重新显示已有历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "请输入问题，例如：什么是检索增强生成？",
    max_chars=2000,
)

if prompt:
    # 保存并显示用户消息
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            if use_stream:
                response = st.write_stream(
                    llm.stream_chat(prompt)
                )
            else:
                with st.spinner("模型正在生成回答……"):
                    response = llm.chat(prompt)

                st.markdown(response)

        # 模型调用成功后才保存回答
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    except (ValueError, RuntimeError) as error:
        logger.error(
            "聊天页面调用失败 | error=%s",
            error,
        )

        error_message = f"模型调用失败：{error}"

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
            }
        )