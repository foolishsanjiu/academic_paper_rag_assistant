# M5 Cross-Encoder Reranker 资源门禁与开发集消融

## 1. 范围与结论

本阶段验证 `BAAI/bge-reranker-v2-m3`，实现配置 D（Dense + Reranker）和
配置 E（Hybrid/RRF + Reranker）。只使用冻结的 `dev` 集和已有 qrels，未读取
`test` 调参，未调用 LLM，未修改 RAGChain、Streamlit 或线上默认检索。

结论：模型文件和推理能力可用，CPU 上可以稳定完成离线实验，但约 9～11 秒/题；
Reranker 虽提高 Recall，却显著降低 MRR，并使配置 E 的 nDCG 低于配置 C。
因此 M5 不选择 D/E，不切换应用默认值。

## 2. 模型与环境门禁

| 项目 | 结果 |
|---|---|
| 模型 | `BAAI/bge-reranker-v2-m3` |
| 本地路径 | `D:\Projects\LLM_RAG_Learning\references\models\bge-reranker-v2-m3` |
| 权重大小 | 2,271,071,852 bytes |
| SHA-256 | `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` |
| 推理后端 | 已安装的 Transformers + PyTorch |
| 设备 | CPU（当前 PyTorch 为 CPU build） |
| 新增依赖 | 无 |

短文本 sanity check 中，相关 SAR 段落 logit 为 6.7673，不相关烘焙段落为
-11.0375，相关段落排序在前。模型离线加载成功且没有网络请求。

没有安装 `FlagEmbedding`：pip dry-run 显示它会额外引入训练相关的
`datasets`、`peft`、`accelerate`、`ir-datasets` 等依赖，而官方模型可直接通过
项目现有的 `AutoModelForSequenceClassification` 推理。

## 3. 20/40 Candidates CPU 基准

基准输入来自 M4 Hybrid dev 胜出配置保存的真实论文 Chunk，最大长度为 512 tokens。

| Pair 数 | Batch 4 | Batch 8 | Batch 16 |
|---:|---:|---:|---:|
| 20 | 5.11 s | **4.99 s** | 5.17 s |
| 40 | 10.61 s | **10.20 s** | 11.31 s |

选择 batch size 8。对 40 candidates 重复运行 5 次：

- P50：10.30 s
- P95：10.63 s
- 最大重复分数差：0
- 峰值工作集：2,270.06 MB
- 无 OOM、无异常

该结果满足离线消融的稳定性门禁，但不满足当前应用的交互延迟要求。

## 4. 配置 D/E

| ID | 候选生成 | Reranker 输入 | Final k |
|---|---|---:|---:|
| D | Dense Top-40 | 40 | 10 |
| E | Dense Top-20 + BM25 Top-20 → RRF Top-40 | 最多 40 | 10 |

配置 E 的实际唯一候选数为 26～40，平均 32.96。Reranker 后才应用最终 Top-k；
`multi_document` 来源上限也只能在 Reranker 后应用。

## 5. A～E 开发集消融

25 道 dev 问题中，20 道有答案题参与 Recall、MRR 和 nDCG 聚合。

| 配置 | Recall@10 | MRR@10 | nDCG@10 | 全文件命中 | 全页命中 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| A Dense | 0.4292 | 0.3979 | 0.3574 | 0.70 | 0.60 | 0.09 s |
| B BM25 | 0.3500 | 0.2724 | 0.2403 | 0.75 | 0.65 | <0.01 s |
| **C Hybrid/RRF** | 0.5208 | **0.4228** | **0.3901** | **0.80** | **0.75** | 0.09 s |
| D Dense + Reranker | **0.5667** | 0.3102 | 0.3673 | 0.75 | 0.65 | 11.19 s |
| E Hybrid/RRF + Reranker | 0.5583 | 0.3077 | 0.3585 | 0.65 | 0.60 | 8.81 s |

相对 A，D 的 Recall@10 提升 32.04%，nDCG@10 提升 2.78%，但 MRR@10
下降 22.04%。相对 C，E 的 Recall@10 提升 7.20%，但 MRR@10 下降 27.22%，
nDCG@10 下降 8.09%，全文件和全页命中率均下降 15 个百分点。

这说明当前 Cross-Encoder 能从更深候选池保留更多相关 Chunk，但对 SAR 论文问题的
前排排序与现有 qrels 不够一致。不能只依据 Recall 宣称 Reranker 有效。

## 6. 实现与正确性

- `Reranker` 协议只暴露 Query-Documents 批量评分。
- 本地 Transformers 适配器使用 `model.eval()` 和 `torch.inference_mode()`。
- 分数相同时保留原检索顺序，再以 `chunk_id` 稳定排序。
- 分数数量不匹配、NaN/Inf 和缺失 `chunk_id` 会显式失败。
- Dense distance、BM25 score、RRF score、原检索 rank 和 Reranker score
  均保留在评测结果中。
- 不发生模型异常后的静默回退，避免污染 D/E 实验身份。

## 7. 复现命令

资源门禁：

```bash
python evaluation/benchmark_reranker.py \
  --model D:/Projects/LLM_RAG_Learning/references/models/bge-reranker-v2-m3 \
  --results evaluation/results/m4_hybrid_dev_grid.json \
  --pair-count 20 40 \
  --batch-size 4 8 16 \
  --repeat 5
```

配置 D：

```bash
python evaluation/evaluate_retrieval.py \
  --method dense_rerank --split dev --top-k 10 \
  --reranker-model D:/Projects/LLM_RAG_Learning/references/models/bge-reranker-v2-m3 \
  --reranker-candidate-k 40 --reranker-batch-size 8 \
  --qrels evaluation/qrels.json
```

配置 E：

```bash
python evaluation/evaluate_retrieval.py \
  --method hybrid_rerank --split dev --top-k 10 \
  --candidate-k 20 --rrf-k 60 --fusion-k 40 \
  --reranker-model D:/Projects/LLM_RAG_Learning/references/models/bge-reranker-v2-m3 \
  --reranker-candidate-k 40 --reranker-batch-size 8 \
  --qrels evaluation/qrels.json
```

## 8. 停止点

M5 到此停止。配置 C 仍是 dev 检索消融胜出配置，但应用默认检索仍未切换。
不运行冻结 test，不运行付费端到端生成，不进入 M6 UI/RAG 集成。
