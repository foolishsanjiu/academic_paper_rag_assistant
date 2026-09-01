"""Local Streamlit UI for M7 human answer and citation review."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_human_citations import calculate_human_metrics  # noqa: E402
from evaluation.review_state import (  # noqa: E402
    SYSTEMS,
    calculate_review_progress,
    initialize_label_file,
    save_label_file,
    target_is_complete,
)


SOURCE_PATH = PROJECT_ROOT / "evaluation/results/m7_citation_review_30.json"
LABELS_PATH = PROJECT_ROOT / "evaluation/results/m7_citation_labels_30.json"
SCORE_LABELS = {
    0: "0 — 错误或无根据",
    1: "1 — 部分正确",
    2: "2 — 基本或完全正确",
}
SUPPORT_LABELS: dict[bool | str, str] = {
    True: "支持",
    False: "不支持",
    "uncertain": "不确定",
}


def _next_target(
    package: dict[str, Any], question_id: str, label: str
) -> tuple[str, str]:
    targets = [
        (question["id"], system_label)
        for question in package["questions"]
        for system_label in SYSTEMS
    ]
    start = targets.index((question_id, label))
    for offset in range(1, len(targets) + 1):
        candidate = targets[(start + offset) % len(targets)]
        candidate_question = next(
            question
            for question in package["questions"]
            if question["id"] == candidate[0]
        )
        if not target_is_complete(candidate_question["systems"][candidate[1]]):
            return candidate
    return question_id, label


def _apply_pending_navigation() -> None:
    pending = st.session_state.pop("pending_target", None)
    if pending is not None:
        st.session_state["question_selector"] = pending[0]
        st.session_state["system_selector"] = pending[1]


def _show_sidebar(package: dict[str, Any]) -> tuple[dict[str, Any], str]:
    progress = calculate_review_progress(package)
    st.sidebar.header("复核进度")
    st.sidebar.progress(
        progress["target_completed"] / progress["target_total"],
        text=(
            f"完整页面 {progress['target_completed']} / "
            f"{progress['target_total']}"
        ),
    )
    st.sidebar.caption(
        f"回答评分：{progress['answer_completed']} / {progress['answer_total']}"
    )
    st.sidebar.caption(
        "引用判断："
        f"{progress['citation_completed']} / {progress['citation_total']}"
    )
    st.sidebar.divider()

    questions = package["questions"]
    question_by_id = {question["id"]: question for question in questions}

    def format_question(question_id: str) -> str:
        question = question_by_id[question_id]
        completed = sum(
            target_is_complete(question["systems"][label]) for label in SYSTEMS
        )
        return f"[{completed}/3] {question_id}"

    question_id = st.sidebar.selectbox(
        "问题",
        options=list(question_by_id),
        format_func=format_question,
        key="question_selector",
    )
    label = st.sidebar.selectbox(
        "检索配置",
        options=list(SYSTEMS),
        key="system_selector",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"原始包（只读）：{SOURCE_PATH.name}")
    st.sidebar.caption(f"标签文件：{LABELS_PATH.name}")
    return question_by_id[question_id], label


def _review_form(
    package: dict[str, Any], question: dict[str, Any], label: str
) -> None:
    system = question["systems"][label]
    target_key = f"{question['id']}_{label}"

    st.title("M7 人工引用复核")
    st.caption(
        f"{question['id']} · {question['type']} · {question['language']} · 系统 {label}"
    )
    st.subheader("问题")
    st.write(question["question"])
    with st.expander("参考答案", expanded=False):
        st.write(question["expected_answer"])
    st.subheader("模型回答")
    st.markdown(system["answer"])

    with st.form(f"review_{target_key}"):
        current_score = system["answer_score"]
        score = st.radio(
            "回答正确性",
            options=list(SCORE_LABELS),
            index=current_score if current_score is not None else None,
            format_func=SCORE_LABELS.get,
            horizontal=True,
        )
        answer_notes = st.text_area(
            "回答备注（可选）",
            value=system.get("answer_notes", ""),
            key=f"answer_notes_{target_key}",
        )

        st.subheader(f"逐引用判断（{len(system['citation_pairs'])} 对）")
        pair_values: list[tuple[dict[str, Any], bool | str | None, str]] = []
        for index, pair in enumerate(system["citation_pairs"], start=1):
            with st.container(border=True):
                st.markdown(
                    f"**{index}. {pair['pair_id']} · "
                    f"引用 [{pair['citation_rank']}]**"
                )
                st.markdown("**待核验事实句**")
                st.markdown(pair["claim"])
                st.markdown(
                    f"**来源**：{pair.get('file_name')} · 第 {pair.get('page_number')} 页 · "
                    f"`{pair.get('chunk_id')}`"
                )
                st.text(pair.get("source_text", ""))
                current_support = pair.get("supported")
                if current_support is True:
                    support_index = 0
                elif current_support is False:
                    support_index = 1
                elif current_support == "uncertain":
                    support_index = 2
                else:
                    support_index = None
                supported = st.radio(
                    "该 Chunk 是否支持事实句？",
                    options=list(SUPPORT_LABELS),
                    index=support_index,
                    format_func=SUPPORT_LABELS.get,
                    horizontal=True,
                    key=f"supported_{target_key}_{pair['pair_id']}",
                )
                reviewer_notes = st.text_area(
                    "引用备注（可选）",
                    value=pair.get("reviewer_notes", ""),
                    key=f"reviewer_notes_{target_key}_{pair['pair_id']}",
                )
                pair_values.append((pair, supported, reviewer_notes))

        save = st.form_submit_button("保存当前页", type="primary")
        save_and_next = st.form_submit_button("保存并前往下一未完成项")

    if save or save_and_next:
        system["answer_score"] = score
        system["answer_notes"] = answer_notes
        for pair, supported, reviewer_notes in pair_values:
            pair["supported"] = supported
            pair["reviewer_notes"] = reviewer_notes
        save_label_file(SOURCE_PATH, LABELS_PATH, package)
        if save_and_next:
            st.session_state["pending_target"] = _next_target(
                package, question["id"], label
            )
        st.session_state["save_message"] = (
            f"已保存 {question['id']} / {label}。"
        )
        st.rerun()


def _show_completion(package: dict[str, Any]) -> None:
    progress = calculate_review_progress(package)
    if progress["target_completed"] != progress["target_total"]:
        return
    st.success("90 个回答评分与 475 个引用判断已全部完成。")
    summary = calculate_human_metrics(package)
    st.subheader("人工评测预览")
    st.dataframe(
        [
            {
                "系统": label,
                "平均回答分": round(values["mean_answer_score"], 4),
                "引用正确性": round(values["citation_correctness"], 4)
                if values["citation_correctness"] is not None
                else None,
                "Claim 引用覆盖率": round(values["claim_citation_coverage"], 4)
                if values["claim_citation_coverage"] is not None
                else None,
                "不确定引用数": values["uncertain_citation_pair_count"],
            }
            for label, values in summary.items()
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "下载指标预览 JSON",
        data=json.dumps(summary, ensure_ascii=False, indent=2),
        file_name="m7_human_citation_metrics_preview.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="M7 人工引用复核", layout="wide")
    try:
        package = initialize_label_file(SOURCE_PATH, LABELS_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"无法加载复核文件：{exc}")
        st.stop()

    _apply_pending_navigation()
    message = st.session_state.pop("save_message", None)
    if message:
        st.toast(message)
    question, label = _show_sidebar(package)
    _review_form(package, question, label)
    _show_completion(package)


if __name__ == "__main__":
    main()
