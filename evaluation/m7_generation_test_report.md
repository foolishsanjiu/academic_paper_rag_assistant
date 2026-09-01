# M7 冻结 test 端到端生成与自动引用评测

## 1. 范围与结论

本阶段使用同一冻结 50 题 test 集，对配置 A、C、E 的既有检索 JSON 各执行一次
端到端生成。模型固定为 `deepseek-v4-flash`，temperature 0.2，max tokens 800，
显式关闭 thinking。正式 150 次请求全部成功，没有补发或错误。

结论：E 的拒答正确率最高，但 A/C/E 都未通过引用格式 100% 和可回答题有效引用
覆盖 100% 的门禁；三组均出现大量可回答题错误拒答。结合 E 的 CPU 检索 P95
性能门禁失败，应用默认继续使用 Dense，不能宣称完整 Hybrid + Reranker 已达到
上线标准。

## 2. DeepSeek 默认 thinking 事故

首次 A 运行未显式设置 thinking。DeepSeek V4 默认开启 high thinking，并忽略
temperature；50 次请求中 15 次耗尽 800-token 额度后没有最终 `content`。该运行
保留为 `m7_a_generation_test.json`，不纳入任何正式指标。

根据官方协议修正为 `extra_body={"thinking":{"type":"disabled"}}` 后，重新从头
执行 A/C/E。无效试运行 50 次加正式运行 150 次，总调用数为 200，未超过追加授权
后的上限。事故、修正和输出隔离规则已写入 `m7_test_protocol.md`。

## 3. 自动指标

| 配置 | 完成/错误 | 拒答正确率 | 引用格式合法率 | 引用下标合法率 | 可回答题有引用 | no-answer 无伪引用 | qrel precision | qrel recall | 平均生成延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Dense | 50/0 | 0.60 | 0.90 | **1.00** | **0.975** | **0.70** | **0.250** | 0.414 | 2.03 s |
| C Hybrid/RRF | 50/0 | 0.60 | **0.96** | **1.00** | **0.975** | 0.50 | 0.205 | 0.367 | 2.08 s |
| E Hybrid/RRF + Reranker | 50/0 | **0.64** | **0.96** | **1.00** | 0.950 | 0.60 | 0.228 | **0.454** | 2.08 s |

说明：

- `citation_qrel_precision` 证明引用 Chunk 是否在分级 qrels 中相关，不能替代人工
  “该 Chunk 是否蕴含当前句子”的判断。
- 三组引用下标都合法，说明 `[n]` 没有引用不存在的 source；格式失败主要来自不
  符合冻结 `[1][3]` 规范的方括号写法。
- A/C 各有 20 道可回答题错误拒答，E 有 18 道；这解释了仅 0.60～0.64 的整体
  拒答正确率。三组都正确拒答了 10 道 no-answer，但部分拒答仍附带伪引用。
- A/C 各有 1 道可回答题完全没有有效引用，E 有 2 道，因此全部失败 100% 覆盖门禁。

## 4. 门禁结果

| 门禁 | E 结果 | 判断 |
|---|---:|---|
| 引用编号格式合法率 100% | 96% | **失败** |
| 可回答题至少一个有效引用 100% | 95% | **失败** |
| 拒答正确率相对 A 不下降超过 2 pp | 64% vs 60% | 通过 |
| E 检索 P95 不超过 A 的 2.5× | 约 120× | **失败** |
| 人工 Citation correctness / claim coverage | 等待人工标签 | 待定 |

即使后续人工蕴含得分较高，E 仍因自动引用覆盖和交互性能失败而不能默认上线。

## 5. 原始结果身份

| 文件 | SHA-256 |
|---|---|
| `m7_a_generation_nonthinking_test.json` | `4e72aac9ea741e37b82f65a00d36803ef755f957ed6fadf086eb3f45a78bde11` |
| `m7_c_generation_nonthinking_test.json` | `f319ddf856b638aab58eb7137a2542411ef178e6132f1c0f9edc47a35d2cf3dd` |
| `m7_e_generation_nonthinking_test.json` | `5628c73c746eb1f71f67e3d2aba4570d117664233307c61a0d2d8de9488cd34c` |
| `m7_citation_review_30.json` | `d93c21b86dbe488e0e95e993b26a2c13ad30baff42add2b0f8271b8d9d3b0298` |

原始文件位于 Git 忽略的 `evaluation/results/`，包含论文 Chunk 和模型回答，不手工
修改。正式 A/C/E 文件各记录 `attempted_api_calls=50`、`error_count=0`、
`thinking_mode=disabled`。

## 6. 30 题人工复核包

复核包固定选择 30 道可回答题：15 fact、7 comparison、8 cross-document；包含
全部 8 道中文可回答题。A/C/E 使用完全相同的问题集合，共需复核 475 个
“答案句—引用 Chunk”对：A 150、C 138、E 187。

人工需要填写：

- 每个系统回答的 `answer_score`：0/1/2。
- 每个引用对的 `supported`：`true`、`false` 或 `uncertain`。
- 必要时填写 `answer_notes` 和 `reviewer_notes`。

在人工标签完成前，不报告 Citation correctness 或 Claim citation coverage 的最终值。

## 7. 当前停止点

付费生成和自动引用评测已完成，不再调用 API。M7 仅剩人工复核标签和据此计算的
人工蕴含指标；该工作不能由同一模型代替人工审核。
