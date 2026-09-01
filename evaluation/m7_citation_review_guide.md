# M7 30 题人工引用复核指南

## 复核文件

打开 Git 忽略目录中的：

`evaluation/results/m7_citation_review_30.json`

文件由 `build_citation_review.py` 从冻结 A/C/E 生成结果构建，SHA-256 为：

`d93c21b86dbe488e0e95e993b26a2c13ad30baff42add2b0f8271b8d9d3b0298`

不要修改问题、答案、来源文本、文件名、页码、Chunk ID 或 `pair_id`。只填写下面
列出的人工字段，并另存为 `evaluation/results/m7_citation_labels_30.json`，保留原始
复核包不变。

推荐使用本地标注页：

```bash
python -m streamlit run evaluation/review_app.py
```

页面首次启动时会从原始复核包创建独立标签文件
`evaluation/results/m7_citation_labels_30.json`。之后每次点击“保存当前页”或
“保存并前往下一未完成项”都会原子写入该标签文件，可关闭页面后继续；程序会在
保存前校验问题、答案和引用来源等只读内容没有变化，并禁止把标签写回原始包。

## 回答评分

对每个问题下 A、C、E 的 `answer_score` 填写：

- `0`：错误、无根据，或本可回答却错误拒答。
- `1`：部分正确，但缺少重要事实、比较对象或跨文档证据。
- `2`：基本或完全正确，重要结论与参考答案一致。

在 `answer_notes` 简要说明扣分原因。评分时可参考 `expected_answer`，但不得用外部
知识补充论文中不存在的信息。

## 引用蕴含判断

对每个 `citation_pairs` 项，只判断其中的 `source_text` 是否支持 `claim`：

- `true`：Chunk 明确支持该事实句的全部关键内容。
- `false`：Chunk 不支持、仅主题相关，或与事实句矛盾。
- `"uncertain"`：必须依赖图表、公式上下文或领域判断，当前无法可靠决定。

不要因为答案整体正确就自动把全部引用标为 `true`；一个句子引用多个 Chunk 时，
每个 Chunk 独立判断。必要时在 `reviewer_notes` 记录原因。

## 完成检查

- 30 个问题的 A/C/E 都填写 `answer_score`，共 90 个评分。
- 475 个引用对全部填写 `supported`。
- 不覆盖 `m7_citation_review_30.json`。
- 完成后运行后续汇总脚本计算 Citation correctness 和 Claim citation coverage。

完成标签后运行：

```bash
python evaluation/evaluate_human_citations.py \
  --labels evaluation/results/m7_citation_labels_30.json \
  --output evaluation/results/m7_human_citation_metrics.json
```

脚本会拒绝任何缺失的回答评分或引用判断，并分别汇总 A/C/E 的平均回答评分、
Citation correctness、Claim citation coverage 和 uncertain 数量。
