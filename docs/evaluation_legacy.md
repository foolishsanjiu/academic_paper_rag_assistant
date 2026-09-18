# 历史 25 题评测说明

本页保存项目早期 25 题评测的入口，避免它们与当前 100 题数据集和 M7 冻结 test
混为同一口径。历史报告保留原文，不回填新指标，也不据此修改最终实验结论。

## 为什么保留

早期实验记录了项目从基础 Dense Retrieval 到来源多样化策略的演进，可用于说明
参数选择和迭代过程。但这些实验早于 Chunk-level qrels，主要使用文件/页命中率和
人工评分，不能与 M7 的 Recall@10、MRR@10、nDCG@10 直接横向比较。

## 历史报告

| 报告 | 范围 | 使用方式 |
|---|---|---|
| [Baseline Review](../evaluation/baseline_review.md) | 原始 25 题 RAG baseline | 历史基线 |
| [Top-k Experiment](../evaluation/top_k_report.md) | 25 题 Top-k 检索比较 | 参数探索记录 |
| [Diversified Retrieval](../evaluation/diversified_retrieval_report.md) | 来源多样化检索 | 在线策略演进记录 |
| [Optimized RAG](../evaluation/optimized_rag_report.md) | 优化前后问答比较 | 历史生成评测 |
| [M0 Dense Baseline](../evaluation/m0_dense_baseline.md) | 扩展评测前的冻结 Dense 快照 | 新评测体系起点 |

## 当前主 Benchmark

对外结果应以参数冻结后的 50 题 test 为准：

- [M7 检索消融报告](../evaluation/m7_retrieval_test_report.md)
- [M7 生成与自动引用报告](../evaluation/m7_generation_test_report.md)
- [M7 最终消融报告](../evaluation/m7_final_ablation_report.md)

当前完整数据集包含 100 题和 213 条 Chunk-level graded qrels；其中 25 道 legacy
问题仅作为完整数据集的一部分保留，不作为新的盲测集。
