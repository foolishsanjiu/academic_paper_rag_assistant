"""Convert structured tool results into chat-ready text and sources."""

from __future__ import annotations

from typing import Any


def format_tool_response(result: dict[str, Any]) -> str:
    """Create a concise user-facing message from a tool result."""
    if not result.get("ok"):
        return result.get("error", {}).get(
            "message",
            "工具调用失败，请稍后重试。",
        )

    tool = result.get("tool")
    action = result.get("action")
    if tool == "paper_library":
        if action == "paper_count":
            return f"当前知识库共有 {result['paper_count']} 篇论文。"
        if action == "paper_list":
            papers = result.get("papers", [])
            listing = "\n".join(
                f"{index}. `{name}`"
                for index, name in enumerate(papers, start=1)
            )
            return f"当前知识库共有 {len(papers)} 篇论文：\n\n{listing}"
        if action == "paper_exists":
            state = "存在" if result.get("exists") else "不存在"
            return f"知识库中{state}论文 `{result.get('paper_name')}`。"
        if action == "paper_info":
            paper = result["paper"]
            return (
                f"论文：`{paper['file_name']}`\n\n"
                f"- PDF 页数：{paper['page_count']}\n"
                f"- 文件大小：{paper['size_bytes']} bytes\n"
                f"- Document ID：`{paper['document_id']}`"
            )
        if action == "index_status":
            if not result.get("indexed"):
                return "当前尚未构建向量索引。"
            return (
                "当前索引状态：\n\n"
                f"- 论文数：{result.get('paper_count')}\n"
                f"- PDF 页数：{result.get('page_count')}\n"
                f"- Chunk 数：{result.get('chunk_count')}\n"
                f"- Chunk Size：{result.get('chunk_size')}\n"
                f"- Chunk Overlap：{result.get('chunk_overlap')}\n"
                f"- Embedding：`{result.get('embedding_model')}`"
            )

    if tool == "source_lookup":
        return (
            f"已定位 `{result.get('paper_name')}` 中的原文，"
            f"共找到 {result.get('match_count', 0)} 个 Chunk。"
        )

    return "工具调用已完成。"


def source_lookup_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt exact-source chunks to the existing Streamlit source renderer."""
    if not result.get("ok") or result.get("tool") != "source_lookup":
        return []

    return [
        {
            "rank": rank,
            "document_id": None,
            "document_type": "pdf",
            "file_name": result.get("paper_name"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "similarity": None,
            "text": chunk.get("text", ""),
        }
        for rank, chunk in enumerate(result.get("chunks", []), start=1)
    ]
