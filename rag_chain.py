"""Core retrieval-augmented generation pipeline."""

from dataclasses import dataclass
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm_client import LLMClient
from retriever import (
    RetrievalStrategy,
    cosine_distance_to_similarity,
    get_retrieval_options,
    retrieve_with_scores,
)


logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = (
    "根据当前论文知识库中的检索内容，"
    "无法可靠回答该问题。"
)


@dataclass(frozen=True)
class RAGSource:
    """One retrieved source used by the RAG pipeline."""

    rank: int
    document_id: str
    document_type: str
    file_name: str
    page_number: int | str
    chunk_id: str
    similarity: float
    text: str


@dataclass(frozen=True)
class RAGResponse:
    """Complete output returned by the RAG pipeline."""

    question: str
    retrieval_query: str
    answer: str
    sources: list[RAGSource]


def format_chat_history(
    chat_history: list[dict] | None,
    max_messages: int = 6,
) -> str:
    """
    Convert recent chat history into plain text.

    Only user and assistant messages are retained.
    """
    if not chat_history:
        return ""

    valid_messages = [
        message
        for message in chat_history
        if (
            message.get("role")
            in {"user", "assistant"}
            and message.get("content")
        )
    ]

    recent_messages = valid_messages[
        -max_messages:
    ]

    lines: list[str] = []

    for message in recent_messages:

        role = message["role"]

        if role == "user":
            role_name = "用户"
        else:
            role_name = "助手"

        content = str(
            message["content"]
        ).strip()

        lines.append(
            f"{role_name}：{content}"
        )

    return "\n".join(lines)


def rewrite_query(
    llm: LLMClient,
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a context-dependent question into a standalone
    retrieval query.

    If there is no chat history, the original question is
    returned directly.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "问题不能为空。"
        )

    history_text = format_chat_history(
        chat_history
    )

    if not history_text:
        return cleaned_question

    rewrite_prompt = f"""
你正在为一个学术论文检索系统改写查询。

请根据历史对话，将“当前问题”改写为一个能够脱离历史对话、
独立理解的检索问题。

要求：
1. 保留原问题的真实意图；
2. 补全“它”“该方法”“这个模型”等指代；
3. 不回答问题；
4. 不增加历史对话中没有出现的新事实；
5. 保留 SAR、GAN、RAG 等专业术语；
6. 只输出改写后的问题，不输出解释。

历史对话：
{history_text}

当前问题：
{cleaned_question}

改写后的独立问题：
""".strip()

    rewritten_query = llm.chat(
        rewrite_prompt
    ).strip()

    if not rewritten_query:
        return cleaned_question

    return rewritten_query


def build_sources(
    results: list[
        tuple[Document, float]
    ],
) -> list[RAGSource]:
    """
    Convert retrieval results into structured RAG sources.
    """
    sources: list[RAGSource] = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        source = RAGSource(
            rank=rank,

            file_name=str(
                metadata.get(
                    "file_name",
                    "unknown.pdf",
                )
            ),

            document_id=str(
                metadata.get(
                    "document_id",
                    "unknown",
                )
            ),

            document_type=str(
                metadata.get(
                    "document_type",
                    "unknown",
                )
            ),

            page_number=metadata.get(
                "page_number",
                "unknown",
            ),

            chunk_id=str(
                metadata.get(
                    "chunk_id",
                    "unknown",
                )
            ),

            similarity=(
                cosine_distance_to_similarity(
                    distance
                )
            ),

            text=document.page_content,
        )

        sources.append(
            source
        )

    return sources


def build_context(
    sources: list[RAGSource],
) -> str:
    """
    Convert retrieved sources into context for the LLM.
    """
    context_parts: list[str] = []

    for source in sources:

        context_part = f"""
[Source {source.rank}]
File: {source.file_name}
PDF Page: {source.page_number}
Chunk ID: {source.chunk_id}

{source.text}
""".strip()

        context_parts.append(
            context_part
        )

    return "\n\n".join(
        context_parts
    )


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a strict grounded RAG prompt.
    """
    return f"""
你是一个学术论文 RAG 问答助手。

你必须严格依据下面提供的“论文检索内容”回答问题。

规则：

1. 只能使用论文检索内容中的信息回答。
2. 不要使用你自己的外部知识补充缺失信息。
3. 如果检索内容不足以可靠回答问题，请明确回答：
   “{NO_ANSWER_MESSAGE}”
4. 不要编造论文名称、方法、参数、实验结果或结论。
5. 回答中的重要事实应使用 [1]、[2] 等形式标记来源。
6. [1] 对应 Source 1，[2] 对应 Source 2，以此类推。
7. 如果多个来源支持同一结论，可以写成 [1][3]。
8. 使用与用户问题相同的语言回答。
9. 回答应首先直接回答问题，再进行必要解释。

论文检索内容：

{context}

用户问题：

{question}

请基于上述论文内容回答：
""".strip()


class RAGChain:
    """Basic two-step academic-paper RAG pipeline."""

    def __init__(
        self,
        llm: LLMClient,
        vector_store: Chroma,
        top_k: int | None = None,
        retrieval_strategy: RetrievalStrategy | str = (
            RetrievalStrategy.FOCUSED
        ),
        candidate_k: int | None = None,
        max_chunks_per_file: int | None = None,
    ) -> None:

        options = get_retrieval_options(
            strategy=retrieval_strategy,
            top_k=top_k,
        )

        self.llm = llm
        self.vector_store = vector_store
        self.retrieval_strategy = options.strategy
        self.top_k = options.top_k
        self.candidate_k = (
            candidate_k
            if candidate_k is not None
            else options.candidate_k
        )
        self.max_chunks_per_file = (
            max_chunks_per_file
            if max_chunks_per_file is not None
            else options.max_chunks_per_file
        )

    def ask(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """
        Run the complete RAG pipeline.

        Steps:
        1. Rewrite multi-turn question if necessary.
        2. Retrieve Top-k chunks.
        3. Build context.
        4. Build grounded prompt.
        5. Call the LLM.
        """
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "问题不能为空。"
            )

        logger.info(
            "RAG request started | strategy=%s | top_k=%s | "
            "question_length=%s | history_messages=%s",
            self.retrieval_strategy.value,
            self.top_k,
            len(cleaned_question),
            len(chat_history or []),
        )

        # ------------------------------------------
        # Step 1: Query rewrite
        # ------------------------------------------

        retrieval_query = rewrite_query(
            llm=self.llm,
            question=cleaned_question,
            chat_history=chat_history,
        )

        # ------------------------------------------
        # Step 2: Retrieval
        # ------------------------------------------

        results = retrieve_with_scores(
            vector_store=self.vector_store,
            query=retrieval_query,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
            max_chunks_per_file=self.max_chunks_per_file,
        )

        if not results:
            logger.info(
                "RAG request refused without retrieval results | "
                "strategy=%s",
                self.retrieval_strategy.value,
            )
            return RAGResponse(
                question=cleaned_question,
                retrieval_query=retrieval_query,
                answer=NO_ANSWER_MESSAGE,
                sources=[],
            )

        # ------------------------------------------
        # Step 3: Context
        # ------------------------------------------

        sources = build_sources(
            results
        )

        context = build_context(
            sources
        )

        # ------------------------------------------
        # Step 4: Prompt
        # ------------------------------------------

        prompt = build_rag_prompt(
            question=cleaned_question,
            context=context,
        )

        # ------------------------------------------
        # Step 5: LLM generation
        # ------------------------------------------

        answer = self.llm.chat(
            prompt,
            temperature=temperature,
        )

        logger.info(
            "RAG request completed | strategy=%s | sources=%s | "
            "answer_length=%s",
            self.retrieval_strategy.value,
            len(sources),
            len(answer),
        )

        return RAGResponse(
            question=cleaned_question,
            retrieval_query=retrieval_query,
            answer=answer,
            sources=sources,
        )
