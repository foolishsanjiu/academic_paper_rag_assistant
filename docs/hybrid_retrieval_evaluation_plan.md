# Hybrid Retrieval、Reranker 与评测扩展实施计划

## 1. 目标与边界

本文档用于指导 Academic Paper RAG Assistant 的下一阶段开发，包含两项工作：

1. 增加 BM25 + Dense Hybrid Retrieval，并在候选结果上增加 Cross-Encoder Reranker。
2. 将现有 25 题评测集扩展到 100 题，增加 Recall@k、MRR@k、nDCG@k 和引用正确性评测。

本阶段保留以下现有设计：

- PDF 解析继续使用 PyMuPDF。
- Dense Embedding 继续使用 `BAAI/bge-m3`。
- Dense 向量继续保存在 Chroma。
- `focused` 和 `multi_document` 继续表示来源选择策略。
- Streamlit、Router、PaperLibraryTool 和 SourceLookupTool 不做无关重构。

本阶段不包含：

- 更换向量数据库。
- GraphRAG、知识图谱或多 Agent。
- OCR、图表理解或公式结构恢复。
- FastAPI、用户系统或远程部署。
- Reranker 微调。

计划阶段不下载模型、不重建索引、不运行付费 LLM 评测。任何实现任务都应单独执行、验证后再进入下一阶段。

## 2. 当前基线

当前可复现状态：

| 项目 | 当前值 |
|---|---:|
| PDF | 15 |
| PDF pages | 218 |
| Chunks | 2142 |
| Chunk size / overlap | 600 / 100 |
| Dense embedding | `BAAI/bge-m3` |
| Vector store | Chroma cosine |
| 评测题 | 25 |
| 自动化测试 | 47 |

现有检索配置：

| 策略 | Candidate k | Final k | 每篇论文上限 |
|---|---:|---:|---:|
| `focused` | 5 | 5 | 无 |
| `multi_document` | 32 | 8 | 3 |

现有评测只记录预期文件和页码是否命中，适合计算 any/all hit，但不足以严谨计算排序指标。MRR 和 nDCG 开发前必须先增加 Chunk 级相关性标注。

## 3. 总体技术路线

```text
                         ┌─ Dense / Chroma Top-N ─┐
用户查询 → Query rewrite ┤                         ├→ RRF → Candidate pool
                         └─ BM25 Top-N ───────────┘
                                                        ↓
                                              Cross-Encoder Reranker
                                                        ↓
                                        focused 或 multi_document 选取
                                                        ↓
                                          Final Top-k → RAG Prompt → LLM
```

候选结果必须始终保留：

- `chunk_id`
- `file_name`
- `page_number`
- `dense_rank` / `dense_score`
- `sparse_rank` / `sparse_score`
- `rrf_score`
- `rerank_score`
- 最终 `rank`

对外引用仍使用最终排序编号 `[1]`、`[2]`，不得把内部 RRF 或 Reranker 分数当作引用编号。

## 4. 配置维度设计

不要扩展现有 `RetrievalStrategy` 来同时表达算法和业务策略。增加独立枚举：

```python
class RetrievalMethod(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    DENSE_RERANK = "dense_rerank"
    HYBRID_RERANK = "hybrid_rerank"
```

两个维度的职责：

| 维度 | 可选值 | 负责内容 |
|---|---|---|
| `RetrievalMethod` | dense / bm25 / hybrid / dense_rerank / hybrid_rerank | 候选如何产生和排序 |
| `RetrievalStrategy` | focused / multi_document | 最终 Top-k 和单篇论文上限 |

这样可以分别回答：

- Hybrid 是否优于纯 Dense？
- Reranker 是否带来增益？
- 来源多样化是否仍然必要？

## 5. 模块和文件设计

### 5.1 新增文件

| 文件 | 职责 |
|---|---|
| `sparse_retriever.py` | 分词、BM25 内存索引、Sparse Top-N |
| `hybrid_retriever.py` | Dense/Sparse 结果标准化、去重、RRF 融合 |
| `reranker.py` | 加载 Cross-Encoder、批量打分和重新排序 |
| `retrieval_pipeline.py` | 按 `RetrievalMethod` 编排候选召回、融合、重排和来源限制 |
| `evaluation/metrics.py` | Recall、MRR、nDCG 和聚合逻辑 |
| `evaluation/validate_dataset.py` | 评测数据和 qrels 静态校验 |
| `evaluation/build_annotation_pool.py` | 汇总多个系统的 Top-N 候选，生成待标注池 |
| `evaluation/evaluate_ablation.py` | 统一执行检索消融实验 |
| `evaluation/evaluate_citations.py` | 解析引用并生成自动/人工引用评测结果 |
| `evaluation/qrels.json` | Chunk 级分级相关性标注 |
| `evaluation/citation_labels.json` | 人工引用蕴含判断，可逐步追加 |
| `tests/test_sparse_retriever.py` | BM25 和分词测试 |
| `tests/test_hybrid_retriever.py` | RRF、去重和确定性测试 |
| `tests/test_reranker.py` | 使用 Fake Reranker 验证批量重排 |
| `tests/test_metrics.py` | Recall、MRR、nDCG 边界测试 |
| `tests/test_citation_metrics.py` | 引用解析、越界和正确性测试 |

### 5.2 修改文件

| 文件 | 最小修改 |
|---|---|
| `config.py` | 增加 BM25、RRF、候选数、Reranker 模型和 batch size 默认值 |
| `requirements.txt` | 增加 BM25/Reranker 所需依赖并固定验证后的版本 |
| `retriever.py` | 保留兼容入口；共享现有来源多样化逻辑 |
| `rag_chain.py` | 依赖统一 retrieval pipeline，不在此实现具体算法 |
| `app.py` | 缓存 Sparse Index/Reranker；展示实际检索方法和耗时 |
| `evaluation/evaluate.py` | 使用新版数据结构和引用指标，兼容旧字段读取 |
| `evaluation/evaluate_retrieval.py` | 最终迁移到 `evaluate_ablation.py`，旧命令保留兼容层 |
| `index_manifest.py` | 记录 sparse tokenizer/version；不把 Reranker 写入索引身份 |
| `README.md` | 更新运行方式、实验结果和限制 |

不得为了新增功能重排 `app.py` 的无关 UI，也不得同时改写 Router 或 Tool Calling。

## 6. Sparse Retrieval 方案

### 6.1 第一版选择

第一版使用 Okapi BM25。当前只有 2142 个 Chunk，应用启动时从 Chroma `get()` 读取 Chunk 文本并构建内存 BM25 索引即可，无需先引入单独搜索服务。

建议使用经过兼容性验证后固定版本的 `rank-bm25`。安装前先执行依赖 dry-run，确认不会破坏当前固定环境。

### 6.2 分词

当前语料主要为英文，第一版分词规则保持可解释：

```text
Unicode normalize
→ casefold
→ 提取英文单词、数字和连字符术语
→ 保留 SAR、GAN、DDPM、G0、5.77 等领域 token
```

要求：

- 索引和查询必须调用同一 tokenizer。
- 空查询返回结构化错误。
- `chunk_id` 必须与 BM25 语料位置一一对应。
- 相同输入必须产生相同排序；同分时以 `chunk_id` 作为稳定 tie-breaker。
- 中文查询不额外分词；当前由 Dense 分支承担跨语言召回，后续单独评测。

### 6.3 生命周期

第一版不持久化 Python pickle，避免不安全反序列化和索引版本漂移：

1. Streamlit 启动时从当前 Chroma Collection 读取全部文档和 metadata。
2. 按 `chunk_id` 排序。
3. 构建 BM25 内存索引。
4. 缓存在 `st.cache_resource`。
5. 知识库重建完成后与 Dense 资源一起清除缓存。

如果后续规模超过约 5 万 Chunk，再单独评估持久化 Sparse Index 或外部搜索引擎；本阶段不提前设计。

## 7. Hybrid 融合方案

### 7.1 RRF

使用 Reciprocal Rank Fusion，避免直接归一化 Dense distance 和 BM25 score：

```text
RRF(d) = Σ 1 / (rrf_k + rank_i(d))
```

M4 开发集实验前的初始值：

- `rrf_k = 60`
- Dense candidate k = 32
- BM25 candidate k = 32
- 融合后最多保留 40 个唯一 Chunk
- 根据 `chunk_id` 去重

M4 在 `dev` 集比较 candidate k `{20, 32, 50}` 和 RRF k `{30, 60}` 后，
按最高 nDCG@10 冻结为 candidate k `20`、RRF k `60`、fusion k `40`。
完整结果见 `evaluation/m4_hybrid_rrf_dev_report.md`。

RRF 同分时按以下顺序稳定排序：

1. 更好的最小分支 rank
2. 更好的 Dense rank
3. `chunk_id` 字典序

不要在第一轮同时调权重、RRF k、候选数和 final k。先固定上述参数验证端到端正确性，再只在开发集做小范围参数实验。

### 7.2 与来源多样化的顺序

顺序固定为：

```text
候选召回 → RRF → Rerank（如启用）→ 每篇论文上限 → Final Top-k
```

来源上限不能在 Reranker 之前应用，否则可能提前删除真正相关证据。

## 8. Reranker 方案

### 8.1 模型选择

主选 `BAAI/bge-reranker-v2-m3`：与当前 BGE-M3 路线一致，支持中英文和多语言 Query-Passage 相关性打分。官方模型约 2.29 GB，因此实施前必须先完成资源门禁。

加载方式优先使用官方 `FlagEmbedding.FlagReranker`，统一调用：

```python
scores = reranker.compute_score(
    [[query, document.page_content] for document in candidates],
    normalize=True,
)
```

### 8.2 资源门禁

正式接入前仅用 20、40 个 Query-Passage pair 做一次本机 smoke benchmark，记录：

- 模型加载时间
- 内存或显存峰值
- batch=4/8/16 的耗时
- CPU 和 GPU 是否可用
- 40 candidates 的 P50/P95 重排延迟

进入下一步的条件：

- 无 OOM。
- 40 candidates 可以稳定完成。
- 相同输入分数稳定。
- 不升级或破坏现有 torch、transformers、sentence-transformers 环境。

如果门禁失败，停止 Reranker 集成并记录原因；允许评估 `BAAI/bge-reranker-base` 作为轻量替代，但不能静默替换，也不能把不同模型结果混入同一实验。

### 8.3 接口与测试

业务代码依赖协议而非具体模型：

```python
class Reranker(Protocol):
    def score(self, query: str, documents: list[Document]) -> list[float]: ...
```

单元测试使用 Fake Reranker，不加载真实模型。真实模型只进入带显式标记的 integration/smoke test，避免普通测试下载数 GB 权重。

## 9. 评测集扩展方案

### 9.1 数据集划分

总题数从 25 扩展到 100：

| Split | 数量 | 用途 |
|---|---:|---|
| `legacy` | 25 | 保留历史回归；已参与旧参数选择，不作为盲测 |
| `dev` | 25 | Hybrid/RRF/Reranker 参数选择 |
| `test` | 50 | 参数冻结后只运行一次的最终报告 |

不得使用 test 调整候选数、RRF k、Reranker、Top-k 或 prompt。

### 9.2 题型目标

100 题总分布：

| 类型 | 总数 | 当前 | 新增 |
|---|---:|---:|---:|
| 单论文事实 `fact` | 40 | 10 | 30 |
| 双论文比较 `comparison` | 20 | 5 | 15 |
| 多论文综合 `cross_document` | 20 | 5 | 15 |
| 无答案 `no_answer` | 20 | 5 | 15 |

额外用 tags 标识正交难度，不另建互斥题型：

- `lexical_mismatch`
- `acronym`
- `numeric_detail`
- `method_mechanism`
- `experiment_result`
- `hard_negative`
- `paraphrase`
- `zh_query_en_corpus`
- `multi_evidence`
- `wrong_premise`

最低覆盖要求：

- 每篇论文至少 4 道可回答题。
- 至少 15 道 `lexical_mismatch`。
- 至少 10 道中文问题检索英文语料。
- 至少 10 道包含数值或实验结果。
- 至少 10 道 hard negative。
- 无答案题中至少一半是领域内但语料无证据的问题，不能全部依赖未来年份或明显跨领域。

### 9.3 questions schema

保留旧字段并增加：

```json
{
  "id": "fact_026",
  "split": "dev",
  "type": "fact",
  "language": "en",
  "difficulty": "medium",
  "tags": ["lexical_mismatch", "method_mechanism"],
  "question": "...",
  "expected_answer": "...",
  "expected_key_points": ["...", "..."],
  "source_files": ["paper.pdf"],
  "source_pages": [3],
  "notes": "..."
}
```

`questions.json` 不直接重复保存大量 Chunk ID；Chunk 级相关性放入独立 `qrels.json`。

### 9.4 qrels schema

```json
{
  "fact_026": [
    {
      "chunk_id": "document-id::page_3::chunk_2",
      "file_name": "paper.pdf",
      "page_number": 3,
      "relevance": 3,
      "reason": "Directly states the requested mechanism."
    },
    {
      "chunk_id": "document-id::page_4::chunk_0",
      "file_name": "paper.pdf",
      "page_number": 4,
      "relevance": 1,
      "reason": "Background context only."
    }
  ]
}
```

相关性等级：

| relevance | 含义 |
|---:|---|
| 3 | 可以直接、完整支持问题的核心答案 |
| 2 | 支持一个必要子结论或关键答案点 |
| 1 | 主题相关但不足以独立回答 |
| 0 | 不相关或误导性 hard negative |

无答案题的 qrels 必须为空。

### 9.5 标注流程

1. 为新题填写预期答案、关键点、预期文件和页面。
2. 用 Dense、BM25、Hybrid 各取 Top-20，合并 `chunk_id` 并去重。
3. 加入指定来源页全部 Chunk，避免只标注系统已经召回的结果。
4. 隐藏候选来源系统及排序，人工按 0～3 标注。
5. 每题至少确认一个 relevance 2 或 3 的 Chunk；无答案题除外。
6. 对 dev/test 各随机抽 20% 做第二次复核，记录冲突并统一标准。
7. `validate_dataset.py` 检查 Chunk 是否存在、文件页码是否匹配、测试集是否意外为空。

注意：Pooling qrels 仍可能不完整。报告中应明确指标表示“相对于已标注证据池”的效果，不能声称覆盖论文中的全部可能证据。

## 10. 新检索指标定义

所有排序指标以 `chunk_id` 为评测单位。默认报告 `k ∈ {5, 8, 10, 20}`。

### 10.1 Recall@k

只把 relevance 大于 0 的 Chunk 视为相关：

```text
Recall@k = Top-k 中相关 Chunk 数 / qrels 中全部相关 Chunk 数
```

同时保留现有 file/page any/all hit，方便与旧报告连续比较。

### 10.2 MRR@k

第一个 relevance 大于 0 的结果排名为 `rank_first`：

```text
MRR@k = mean(1 / rank_first)
```

Top-k 无相关结果时该题为 0；无答案题不参与 Retrieval MRR 分母，另行计算拒答指标。

### 10.3 nDCG@k

使用 0～3 分级相关性：

```text
DCG@k = Σ (2^relevance_i - 1) / log2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

无相关 qrels 的题不进入 nDCG 分母。

### 10.4 聚合

必须同时输出：

- 全集 Macro Average。
- 按 split、type、language、difficulty、tag 分组。
- 题目级明细。
- P50/P95 检索延迟。
- 候选阶段和 Reranker 阶段独立耗时。
- 平均唯一论文数和单篇最大占比。

不得只报告最优单一数字。

## 11. 引用正确性评测

“引用格式合法”“引用来源相关”“引用真正支持句子”是三个不同问题，必须分开计算。

### 11.1 自动指标

对每个生成答案解析 `[n]`：

1. `citation_format_validity`：所有引用能解析为整数编号。
2. `citation_index_validity`：所有编号都在实际 sources 范围内。
3. `citation_qrel_precision`：被引用 Chunk 中 relevance 大于 0 的比例。
4. `citation_qrel_recall`：relevance 2/3 的标注证据中，被答案实际引用的比例。
5. `answer_has_citation`：可回答题是否至少包含一个有效引用。
6. `no_answer_has_no_spurious_citation`：拒答时是否没有伪造引用。

`citation_qrel_precision` 只能证明被引用 Chunk 对问题相关，不能证明它蕴含答案句子。

### 11.2 人工蕴含指标

从 test 中分层抽取至少 30 题，覆盖 fact、comparison、cross_document 和中英文问题。对每个“带引用的答案句子—引用 Chunk”判断：

- `entailed = 1`：Chunk 直接支持该句核心事实。
- `entailed = 0`：不支持、只主题相关、或与该句冲突。

计算：

```text
Citation correctness = supported citation pairs / all citation pairs
Claim citation coverage = 有至少一个支持性引用的可验证事实句 / 全部可验证事实句
```

人工标签保存在 `citation_labels.json`，不得覆盖原始生成结果。可以增加 LLM Judge 作为辅助，但不能用 LLM Judge 替代人工 test 结论。

## 12. 消融实验设计

### 12.1 系统配置

至少执行以下五组：

| ID | Dense | BM25 | RRF | Reranker | 目的 |
|---|---:|---:|---:|---:|---|
| A | ✓ |  |  |  | 当前 Dense baseline |
| B |  | ✓ |  |  | 观察词法检索单独能力 |
| C | ✓ | ✓ | ✓ |  | 测量 Hybrid 增益 |
| D | ✓ |  |  | ✓ | 隔离 Reranker 对 Dense 的贡献 |
| E | ✓ | ✓ | ✓ | ✓ | 完整方案 |

另外对 C/E 分别报告 `focused` 和 `multi_document`，但不要把所有排列组合都纳入首轮调参。

### 12.2 两阶段实验

阶段一：纯检索实验，不调用 LLM。

- 在 dev 上比较候选数 `{20, 32, 50}`。
- RRF k 只比较 `{30, 60}`。
- Reranker 输出候选数比较 `{20, 40}`。
- 选择规则优先 nDCG@10，其次 Recall@10、MRR@10，再看延迟。
- 参数选择完成后冻结配置。

阶段二：端到端生成实验。

- 只运行 A、C、E，避免无效 LLM 成本。
- temperature 固定为 0.2，并记录模型、endpoint 配置摘要和运行时间。
- 比较回答正确性、拒答正确性、引用指标和端到端延迟。
- 参数冻结后在 50 题 test 上运行一次。

### 12.3 计时规范

- 每个检索配置先预热 5 题，预热不计入结果。
- 纯检索实验重复 3 次，报告 P50/P95；排序指标应完全一致。
- 分开记录 dense、bm25、fusion、rerank、total latency。
- 记录运行设备，不在不同硬件的延迟之间直接比较。

## 13. 验收标准

### 13.1 正确性门禁

- 原有 47 个测试全部通过。
- 新增 Metrics/RRF/BM25/Reranker/Citation 单元测试全部通过。
- Dataset validator 无错误。
- 所有 qrels `chunk_id` 均存在于当前索引，且 metadata 匹配。
- 相同输入的检索排序具有确定性。
- Dense baseline 新脚本结果与旧脚本在相同 Top-k 下保持一致。

### 13.2 效果门禁

完整方案 E 相对 A 在冻结 test 上应满足：

- nDCG@10 相对提升至少 5%。
- Recall@10 不下降超过 2 个百分点。
- MRR@10 不下降超过 2 个百分点。
- `retrieval_all_hit_rate` 不低于当前同策略 baseline。
- 引用编号合法率为 100%。
- 可回答题至少一个有效引用的覆盖率为 100%。
- 拒答正确率下降不超过 2 个百分点。

性能预算：

- 不含模型首次加载时，E 的检索 P95 不高于 A 的 2.5 倍。
- 如果效果达标但延迟超限，不默认上线；保留 C 作为无 Reranker 降级方案。

如果 E 未达到门禁，不为“完成任务”而强行切换默认策略。保留实现和实验报告，默认继续使用证据最优的已验证配置。

## 14. 分阶段开发任务

### M0：冻结基线

工作：

- 保存当前 25 题 Dense baseline 的完整 JSON。
- 记录 Git commit、Manifest、依赖版本、硬件和测试结果。
- 确认 `DENSE + focused/multi_document` 结果可复现。

验证：47 tests 通过；旧评测命令可运行。

停止点：只冻结数据，不改检索实现。

### M1：评测 Schema 和 Metrics

工作：

- 增加 questions 新字段兼容读取。
- 实现 qrels loader 和 validator。
- 实现 Recall@k、MRR@k、nDCG@k。
- 增加手工可验证的小样本单元测试。

验证：用 3～5 条人工排序列表逐项核算公式。

停止点：指标完成，但不接 BM25/Reranker。

### M2：扩题和 qrels 标注

状态（2026-08-31）：已完成。评测集已冻结为 100 题，`qrels.json` 包含
213 条经来源页复核的 Chunk 级分级标注，并已通过当前 Chroma 索引校验。

工作：

- 新增 75 题。
- 生成匿名候选池。
- 完成 Chunk 级 0～3 标注。
- 固定 legacy/dev/test。

验证：数据 validator 通过；覆盖要求满足；抽样复核完成。

停止点：数据集冻结后再开发算法，避免边调算法边改 test。

### M3：BM25

状态（2026-08-31）：已完成。已实现确定性内存 BM25、Streamlit 资源缓存与
重建失效、BM25-only 评测路径，并完成配置 B 的 100 题离线实验。

工作：

- 实现 tokenizer、内存索引和 Sparse Top-N。
- 接入资源缓存和知识库重建失效逻辑。
- 增加 BM25-only 评测路径。

验证：排序确定、Chunk ID 正确、空输入和重复分数边界通过。

停止点：只运行配置 B，不实现融合。

### M4：Hybrid/RRF

状态（2026-09-01）：已完成。配置 C 已实现并仅在冻结 `dev` 集完成参数选择；
最终参数为 candidate k `20`、RRF k `60`、fusion k `40`。未读取 test 调参，
未调用 LLM，未修改应用默认检索路径。

工作：

- 标准化 Dense 和 Sparse 候选。
- 实现 chunk_id 去重和 RRF。
- 接入配置 C。

验证：固定假数据手算 RRF 通过；Hybrid 重复运行排序一致且无重复 Chunk；
Dense-only legacy Top-5 与 M0 排名逐题完全一致。Focused dev 上 Hybrid 相对
Dense 的 Recall@10、MRR@10、nDCG@10 分别提升 21.36%、6.25%、9.15%；
multi-document 的 nDCG@8 下降 5.38%，因此暂不切换线上默认值。

停止点：已完成 dev 检索实验；不调用 LLM，不实现 Reranker。

### M5：Reranker

状态（2026-09-01）：已完成。`BAAI/bge-reranker-v2-m3` 的 SHA-256、离线加载、
20/40 candidates CPU 基准和配置 D/E dev 消融均已完成。没有新增 Python 依赖，
没有读取 test 或调用 LLM。

工作：

- 先做真实模型资源门禁。
- 定义协议并实现 Transformers Cross-Encoder adapter。
- 支持批量评分、稳定 tie-break、Fake Reranker 测试。
- 接入配置 D/E。

验证：40 candidates 的 batch 8 P50/P95 为 10.30/10.63 秒，重复分数差为 0，
峰值工作集约 2.27 GB。D/E 均完成 25 题 dev 实验；E 相对 C 的 Recall@10
提升 7.20%，但 MRR@10 和 nDCG@10 分别下降 27.22% 和 8.09%。模型异常会
显式失败，不静默回退并污染实验。

停止点：只完成检索级消融；D/E 未通过默认配置选择门禁，不进入应用集成。

### M6：RAG 和 UI 集成

状态（2026-09-01）：已完成。新增统一 `RetrievalPipeline`，RAG 与 UI 现可在
Dense、BM25、Hybrid 之间切换；Dense 保持默认，BM25 仅在需要时加载并缓存。
Reranker 因 M5 未通过默认配置选择门禁，没有接入应用。

工作：

- RAGChain 改为调用 retrieval pipeline。
- UI 展示 method、strategy、各阶段耗时和最终分数。
- 保持旧消息渲染兼容。

验证：新增统一管线与 RAGChain 集成测试，覆盖三种检索方法、focused、
multi-document、空结果拒答和旧构造方式；既有 Router 数据集继续覆盖 Tool、
out-of-scope 与策略分发。完整测试集共 104 项，全部通过。

停止点：未读取 test、未调用 LLM 或付费 API；默认 method 仍为 Dense，等待 M7
冻结 test 评测结果后再决定是否切换。

### M7：消融和引用评测

状态（2026-09-01）：进行中。冻结 test 的 A～E 纯检索消融已完成并形成协议与
报告；E 通过效果门禁但 CPU P95 约为 A 的 120 倍，因此默认 Dense 不切换。
端到端 A/C/E 生成器、自动引用指标和显式付费开关已实现，等待付费调用授权。

工作：

- dev 选择参数并冻结。
- test 运行 A/B/C/D/E 纯检索实验。
- test 运行 A/C/E 端到端生成。
- 自动引用评测并完成人工 30 题复核。

验证：结果 JSON、汇总 Markdown、环境信息齐全，原始结果不可手改。

停止点：根据验收门禁决定默认配置。

### M8：交付

工作：

- 更新 README、architecture、evaluation README。
- 写消融报告、失败案例和最终选择理由。
- 增加复现命令。

验证：从干净终端按文档完成 tests、retrieval eval 和一个有限端到端 eval。

## 15. 推荐的后续开发节奏

建议按以下独立任务推进，每次只授权和完成一个任务：

1. **任务一：M0 + M1**——冻结 baseline，先把新指标做对。
2. **任务二：M2 数据扩展**——新增问题和 qrels，人工工作量最大。
3. **任务三：M3 BM25**——只实现 Sparse baseline。
4. **任务四：M4 Hybrid/RRF**——完成第一轮核心算法增益验证。
5. **任务五：M5 Reranker 门禁与接入**——会下载约 2.29 GB 模型，单独确认。
6. **任务六：M6 集成**——将胜出检索路径接入 RAG 和 UI。
7. **任务七：M7 + M8**——运行付费评测、人工引用复核和文档交付。

每个任务完成时应报告：

- 实际修改文件。
- 没有修改的范围。
- 测试和实验命令。
- 测试结果。
- 当前停止点。
- 下一候选任务；候选任务不代表已授权执行。

## 16. 风险和控制

| 风险 | 控制措施 |
|---|---|
| Reranker 模型过大或 CPU 延迟过高 | 先做资源门禁；保留 Hybrid 无 Reranker 降级方案 |
| 新依赖破坏现有环境 | dry-run、固定版本、完整回归测试 |
| qrels 只覆盖旧系统结果 | 多系统 pooling + 指定来源页全部 Chunk |
| 用 test 调参导致指标虚高 | legacy/dev/test 分离，test 参数冻结后只运行一次 |
| nDCG 被同页重复 Chunk 放大 | 以明确 Chunk qrels 为准，不把整页所有 Chunk 自动判为相关 |
| 引用相关但不支持具体句子 | 自动 qrel 指标和人工 entailment 分开报告 |
| Hybrid 增益来自更多候选而非算法 | A～E 保持可比候选预算并报告阶段参数 |
| 改动破坏现有 UI/Tool | 检索管线独立，保留兼容入口和 47 项回归测试 |

## 17. 首个可执行任务

下一步建议只执行 **M0 + M1：冻结基线并实现评测 Schema、Recall@k、MRR@k、nDCG@k**。

该任务不需要下载 Reranker，不需要运行付费 LLM，也不改变线上默认检索策略；它为后续所有消融实验提供可信基础。
