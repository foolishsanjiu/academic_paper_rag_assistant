# Final Acceptance Report

验收日期：2026-08-25

## 结论

Academic Paper RAG Assistant 第一阶段达到项目结束定义：RAG、引用、拒答、
评测、参数实验、两个领域 Tool、Intent Router、异常与 Fallback、Streamlit 和
项目文档均已实现并验证。

## RAG

- [x] PDF 可读取，并按一页一个 Document 解析
- [x] 文件名、PDF 页码、Document ID 和 Chunk ID metadata 可追溯
- [x] Chroma 持久化索引可加载
- [x] `focused` Top-k 检索可运行
- [x] `multi_document` 来源多样化检索可运行
- [x] LLM 回答基于检索上下文并显示引用
- [x] 无答案问题具有明确拒答行为
- [x] 多轮问题可执行 retrieval query rewrite

当前真实索引：

| 项目 | 值 |
|---|---:|
| PDF | 15 |
| PDF pages | 218 |
| Chunks | 2142 |
| Chunk Size | 600 |
| Chunk Overlap | 100 |
| Embedding | BAAI/bge-m3 |
| Collection | academic_papers |

索引中的论文数与当前 PDF 目录一致；随机选取首篇论文第 1 页执行
SourceLookupTool，成功返回 12 个 Chunk。

## Evaluation

- [x] 25 题专业 RAG 评测集
- [x] 10 道事实题
- [x] 5 道比较题
- [x] 5 道跨文档题
- [x] 5 道无答案题
- [x] Baseline 完整生成结果
- [x] Top-k 3/5/8/10 检索实验
- [x] 来源多样化与单论文 Chunk 上限实验
- [x] 人工答案评分与优化报告
- [x] 40 题 Router 评测集和真实 LLM 结果

最终跨文档配置将全部期望论文召回率从 65% 提升到 90%，人工答案评分从
39/50 提升到 43/50；代价是平均延迟从 4.12 秒增加到 6.78 秒。因此系统保留
双策略，由 Router 仅为比较与跨文档问题选择多文档策略。

## Tool Calling

- [x] PaperLibraryTool：5 个白名单 action
- [x] SourceLookupTool：PDF 页码与 Chunk ID 精确定位
- [x] 四类 Intent Router
- [x] LLM JSON structured output
- [x] 非法 JSON/API 异常的规则 Fallback
- [x] Tool 参数校验和结构化错误
- [x] Tool 名称与 action 白名单
- [x] 每次请求最多 3 次工具调用
- [x] Tool 未预期异常隔离
- [x] Prompt Injection 与 out-of-scope 拒答

## Engineering

- [x] 配置、日志、异常和核心模块职责分离
- [x] `.env.example` 提供配置模板
- [x] `.env`、PDF、Chroma 和日志不进入 Git
- [x] 依赖版本固定为通过验收的环境版本
- [x] README 包含安装、运行、评测、限制和 Acknowledgements
- [x] Mermaid 架构图和详细模块说明
- [x] Streamlit Agent 页面完成真实交互测试
- [x] 自动化测试全部通过

## 验证命令

```text
python -m unittest discover -s tests
python -m py_compile app.py agent_router.py agent_response.py
python -m pip install --dry-run -r requirements.txt
git diff --check
```

## 已知限制（不阻塞第一阶段验收）

1. 语料聚焦英文 SAR 图像生成，尚未评测中文论文和跨任务语料。
2. PDF 解析不包含扫描件 OCR、图表视觉理解和复杂公式结构恢复。
3. SourceLookupTool 第一版要求请求中明确给出 PDF 文件名。
4. Chroma 是本地单用户存储，索引重建不支持并发读写。
5. 人工评分集规模为 25 题，结论适用于当前语料，不代表通用学术搜索性能。
6. 项目没有用户系统、远程 API、容器化部署或多 Agent 编排；这些不属于本阶段范围。

## 验收后原则

停止增加新框架和与当前项目无关的功能。后续只根据演示、面试或真实用户反馈修复
缺陷，并可将中文语料或 SAR 检测/识别论文作为独立对照实验扩展。
