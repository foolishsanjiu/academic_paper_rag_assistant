# M4 Hybrid Retrieval + RRF 开发集实验报告

## 1. 范围与结论

本阶段实现配置 C（Dense + BM25 + RRF），只在冻结的 `dev` 集上选择参数，
不读取 `test` 集做调参，不调用 LLM，不接入线上默认检索，也不实现 Reranker。

最终冻结的 M4 参数为：

| 参数 | 取值 |
|---|---:|
| Dense candidate k | 20 |
| BM25 candidate k | 20 |
| RRF k | 60 |
| Fusion pool k | 40 |
| Focused final k | 10 |

选择依据是开发集最高的 nDCG@10，同时 MRR@10 也是本轮最高值。
`candidate_k=50` 的 Recall@10 略高，但排序质量更低、候选预算更大，因此不采用。

## 2. 实现

- Dense 与 BM25 分支分别召回候选，再按持久化 `chunk_id` 去重。
- 使用 `RRF(d) = sum(1 / (rrf_k + rank_i(d)))` 融合，不直接混合两种异构分数。
- 同分时依次比较最好分支 rank、Dense rank 和 `chunk_id`，保证稳定排序。
- 来源上限在融合后应用，避免提前删除可能相关的候选。
- 结果保留 Dense/BM25 rank、Dense distance、BM25 score、RRF score 和分阶段延迟。

## 3. 参数网格

评测集为 25 道 `dev` 问题，其中 20 道有答案题参与 Recall、MRR 和 nDCG 聚合。
所有组合使用同一索引、同一 qrels 和 `final_k=10`。

| Candidate k | RRF k | Recall@10 | MRR@10 | nDCG@10 | 全文件命中 | 全页命中 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 30 | 0.5208 | 0.4203 | 0.3889 | 0.80 | 0.75 |
| **20** | **60** | **0.5208** | **0.4228** | **0.3901** | **0.80** | **0.75** |
| 32 | 30 | 0.5125 | 0.4131 | 0.3783 | 0.75 | 0.70 |
| 32 | 60 | 0.5125 | 0.4172 | 0.3801 | 0.75 | 0.70 |
| 50 | 30 | 0.5250 | 0.4131 | 0.3808 | 0.70 | 0.65 |
| 50 | 60 | 0.5250 | 0.4172 | 0.3829 | 0.70 | 0.65 |

## 4. Focused 消融对比

| 配置 | Recall@10 | MRR@10 | nDCG@10 | 全文件命中 | 全页命中 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| A Dense | 0.4292 | 0.3979 | 0.3574 | 0.70 | 0.60 | 89.85 ms |
| B BM25 | 0.3500 | 0.2724 | 0.2403 | 0.75 | 0.65 | 4.33 ms |
| **C Hybrid/RRF** | **0.5208** | **0.4228** | **0.3901** | **0.80** | **0.75** | **89.32 ms** |

相对 Dense，Hybrid 的 Recall@10 提升 21.36%，MRR@10 提升 6.25%，
nDCG@10 提升 9.15%；全文件命中率提升 10 个百分点，全页命中率提升
15 个百分点。Hybrid 检索延迟 P50 为 89.50 ms，P95 为 96.76 ms；平均阶段耗时为
Dense 81.97 ms、BM25 7.03 ms、Fusion 0.17 ms。

## 5. Multi-document 验证

为检查来源上限与融合顺序，额外在 `final_k=8`、每篇论文最多 3 个 Chunk 下比较：

| 配置 | Recall@8 | MRR@8 | nDCG@8 | 全文件命中 | 任一页命中 | 全页命中 |
|---|---:|---:|---:|---:|---:|---:|
| Dense，candidate 32 | 0.3167 | 0.3833 | **0.3346** | 0.75 | **0.95** | 0.50 |
| Hybrid，candidate 20 | **0.3208** | **0.3917** | 0.3166 | **0.90** | 0.85 | **0.60** |

该结果是混合的：Hybrid 提升了 Recall、MRR、全文件和全页覆盖，但 nDCG@8
下降 5.38%，任一页命中率下降 10 个百分点。因此 M4 的结果不足以授权把 Hybrid
设为应用默认值；后续需由 Reranker 和冻结 test 集消融判断最终配置。

## 6. 正确性与兼容性

- 固定假数据的手算 RRF、分支去重、稳定 tie-break 和融合后来源限制均由单元测试覆盖。
- 相同查询重复执行的 Hybrid 排名完全一致，且单个排名内无重复 `chunk_id`。
- M4 修改后的 Dense `legacy` Top-5 排名与 M0 原始结果逐题、逐 rank 完全一致。
- 评测输出明确记录方法、split、candidate k、RRF k、fusion k 和阶段延迟。

## 7. 复现命令

```bash
python evaluation/evaluate_retrieval.py \
  --method hybrid \
  --split dev \
  --top-k 10 \
  --candidate-k 20 32 50 \
  --rrf-k 30 60 \
  --fusion-k 40 \
  --qrels evaluation/qrels.json \
  --output evaluation/results/m4_hybrid_dev_grid.json
```

Multi-document 验证：

```bash
python evaluation/evaluate_retrieval.py \
  --method hybrid \
  --split dev \
  --top-k 8 \
  --candidate-k 20 \
  --rrf-k 60 \
  --fusion-k 40 \
  --max-chunks-per-file 3 \
  --qrels evaluation/qrels.json \
  --output evaluation/results/m4_hybrid_dev_multi.json
```

原始 JSON 位于 Git 忽略的 `evaluation/results/`，避免提交论文片段；本报告保存可审计汇总。

## 8. 停止点

M4 到此停止。尚未下载或评测 Reranker，未运行冻结 test 集，未调用付费 LLM，
未修改 RAGChain、Streamlit 或应用默认检索路径。
