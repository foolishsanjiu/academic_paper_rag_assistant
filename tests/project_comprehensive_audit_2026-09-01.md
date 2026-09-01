# 项目全面正确性审计报告（2026-09-01）

## 1. 结论

本次检查覆盖 Streamlit 主入口、PDF 管理、索引 Manifest、Dense/BM25/Hybrid、
Reranker、RAG Chain、Tool、LLM Client、评测集、qrels、引用指标与人工复核工具。

共确认并修复 8 类问题。修复后全仓语法编译通过，137 项自动化测试全部通过，
真实本地索引、三种检索方法、Reranker 和 Streamlit 启动冒烟均通过。在“不重新调用
付费 LLM、不破坏冻结索引”的可验证范围内，没有剩余的阻塞级代码缺陷。

## 2. 已修复问题

| 级别 | 问题 | 影响 | 修复 |
|---|---|---|---|
| P1 | 主应用未验证 Chroma 数量和 Embedding 身份 | Manifest 漂移时可能用错误模型查询现有向量 | 启动时同时校验 Chunk 数量与 `BAAI/bge-m3` 身份；评测入口也执行相同门禁 |
| P1 | 上传 PDF 后若重建失败，必须继续上传新文件才能重试 | PDF 已落盘但索引无法通过 UI 恢复 | 重建按钮现在允许不选择新文件，直接使用当前论文目录重建 |
| P1 | 多个 Chroma 读取工具用 `zip()` 静默截断不等长列 | 损坏索引可能被误判为没有结果或缺少 qrel | SourceLookup、数据集验证和标注池构建均拒绝列错位、非法 metadata 和重复/空 ID |
| P2 | `stream_chat()` 忽略传入的 Temperature | 调用方配置不生效 | 将 Temperature 传入 OpenAI-compatible 流式请求，并增加请求参数测试 |
| P2 | `MAX_UPLOAD_SIZE_MB=50` 只显示在 UI，没有实际执行 | 大文件可绕过项目声明的限制 | PDF 解析前执行 50 MB 硬限制 |
| P2 | Top-k 接受 `True` 或浮点数 | 非法参数可能进入 Chroma 查询 | 统一要求大于 0 的非布尔整数 |
| P2 | `*.pdf` 在大小写敏感文件系统漏掉 `.PDF` | 合法上传文件可能不被统计或重建 | PDF 发现改为对扩展名做大小写无关比较 |
| P2 | 代码直接导入 `transformers`，但依赖未直接固定 | 间接依赖升级可能破坏 Reranker | 在 `requirements.txt` 固定已验证的 `transformers==5.15.0` |

## 3. 自动化验证

| 检查 | 结果 |
|---|---|
| `python -m compileall -q .` | 通过 |
| `python -m unittest discover -s tests -v` | **137/137 通过** |
| `python -m pip check` | 无依赖冲突 |
| `pip install --dry-run --no-index -r requirements.txt` | 所有固定版本均已满足 |
| 11 个评测 CLI 的 `--help` | 全部正常导入和退出 |
| `git diff --check` | 通过 |

新增回归覆盖：

- Manifest Chunk 数量、Embedding 模型和非法类型。
- 流式请求 Temperature 透传。
- 非整数/布尔 Top-k 拒绝。
- 50 MB 上传限制。
- 大写 `.PDF` 跨平台发现与解析。
- SourceLookup、qrels 索引校验和标注池的 Chroma 列错位拒绝。

## 4. 真实本地资源验证

全部模型调用均设置 Hugging Face/Transformers 离线模式，没有联网下载。

| 检查 | 结果 |
|---|---|
| 当前 PDF | 15 篇 |
| 当前 PDF 内容 ID 与索引 document ID | 15/15 完全一致，无缺失或孤儿文档 |
| Chroma 与 Manifest | 2,142/2,142 Chunk，一致 |
| Manifest Embedding | `BAAI/bge-m3`，与运行配置一致 |
| 扩展评测集 | 100 题，100 题均有 qrels 项 |
| qrels | 213 条，全部与当前索引 Chunk/文件/页码一致 |
| SourceLookup | 首篇论文第 1 页成功返回 12 个 Chunk |
| Dense/BM25/Hybrid | 同一真实查询均成功返回 Top-3 |
| 本地 BGE Reranker | 4 个真实 Chunk 成功重排为 Top-2，分数有限且顺序有效 |
| Streamlit 主应用 | 离线启动 `0 exception / 0 error` |

## 5. 边界与剩余风险

以下事项不是本次发现的未实现功能，但无法被离线测试证明为“永远正确”：

1. 本次没有发送新的 DeepSeek 请求，因此没有重新验证外部 API 的实时可用性、余额、
   限流和服务端协议变化；请求参数由 Mock 测试覆盖，历史正式生成结果保持不变。
2. 为保护冻结索引，本次没有点击真实重建按钮覆盖 Chroma。重建逻辑仍属于本地
   单用户流程，不提供数据库级事务或并发读写保证；若进程在 `reset_collection()`
   后异常退出，可能需要再次执行完整重建。
3. PDF 路线只支持可提取文本，不支持扫描件 OCR、图表视觉理解和公式结构恢复。
4. 人工引用正确性来自单名标注者，属于实验测量限制，不是代码验证结果。
5. Reranker 保留在离线评测路径，没有接入默认交互路径。这是因为 CPU P95 延迟门禁
   失败后的明确工程决策，不是遗漏实现。

## 6. 当前状态

当前项目的离线功能、数据身份和主要运行入口均已实际验证。已知剩余风险均已在上节
明确边界，没有发现会阻塞现有 Dense 默认应用、Hybrid/Reranker 离线实验或 M7
评测复现的错误。
