# M7 冻结 test 检索消融执行协议

## 1. 执行边界

- 执行日期：2026-09-01
- 数据划分：`test`，固定 50 题
- Final k：10
- 指标：Recall@10、MRR@10、nDCG@10、文件/页命中率和检索延迟
- 本协议在读取 M7 test 结果前冻结；A～E 全部运行完成前不得根据中间结果改参。
- 本阶段不调用 LLM，不产生付费 API 调用。

## 2. 冻结输入

| 文件 | SHA-256 |
|---|---|
| `evaluation/questions.json` | `032de79c4e3ad89a142ea9b4973f9a8849ec20b95f3ad62ba28b1db3ae19e8bf` |
| `evaluation/qrels.json` | `56fe37075a1397ea8fec4d25d88a14730e2aad0480f9faa299cb7b25b471fcd9` |
| `chroma_db/index_manifest.json` | `67133c59b3aaa9ff961d49ceece428801a8e6d355598e04111358887ea415fc8` |

数据门禁：100 题、50 道 test、213 条相关性判断，且 qrels 已通过当前 Chroma
索引校验。

## 3. 冻结配置

| ID | Method | 冻结参数 | 输出 |
|---|---|---|---|
| A | Dense | Top-10 | `evaluation/results/m7_a_dense_test.json` |
| B | BM25 | Top-10 | `evaluation/results/m7_b_bm25_test.json` |
| C | Hybrid/RRF | branch candidate k 20，RRF k 60，fusion k 40，Top-10 | `evaluation/results/m7_c_hybrid_test.json` |
| D | Dense + Reranker | Dense candidate k 40，batch 8，CPU，Top-10 | `evaluation/results/m7_d_dense_rerank_test.json` |
| E | Hybrid/RRF + Reranker | C 的候选融合，Reranker candidate k 40，batch 8，CPU，Top-10 | `evaluation/results/m7_e_hybrid_rerank_test.json` |

Reranker 固定使用本地
`D:/Projects/LLM_RAG_Learning/references/models/bge-reranker-v2-m3`，权重
SHA-256 为
`d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`。

## 4. 结果约束

- 原始 JSON 由评测程序原子写入，不手工修改。
- A～E 必须使用同一 questions、qrels 和 Chroma manifest。
- test 结果只用于最终报告和默认方法门禁，不再用于调整候选数、RRF、Top-k、
  Reranker 或 Prompt。
- 结果汇总必须同时报告绝对指标、相对变化、延迟和失败案例，不能只选取有利指标。

## 5. 端到端生成协议修正记录

2026-09-01 首次配置 A 生成运行发出 50 次调用后，发现 DeepSeek V4 默认开启
high thinking：temperature 被忽略，且 15 题的 800-token 额度被推理耗尽，没有
最终 `content`。该运行保存在 `m7_a_generation_test.json`，标记为无效试运行，
不得并入 A/C/E 比较。

根据 DeepSeek 官方 thinking-mode 协议，正式 A/C/E 统一显式使用
`thinking=disabled`，继续固定 temperature 0.2、max tokens 800。正式输出使用
`m7_{a,c,e}_generation_nonthinking_test.json`，不得与无效试运行混合或覆盖。
