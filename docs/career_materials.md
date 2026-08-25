# Academic Paper RAG Assistant 求职与面试材料

本文内容基于项目当前代码和 2026-08-25 的最终验收结果。根据简历版面选择
合适长度，不需要把所有内容同时写进简历。

## 一、简历项目描述

### 项目名称

**Academic Paper RAG Assistant｜学术论文 RAG + Tool Calling 智能助手**

技术栈：Python、LangChain、Chroma、BGE-M3、OpenAI-compatible API、
PyMuPDF、Streamlit、Unittest

### 推荐四条

1. 基于 LangChain、BGE-M3、Chroma 与兼容 OpenAI API 的大模型构建学术论文
   RAG 系统，实现 PDF 逐页解析、Chunk 切分、向量检索、多轮查询改写、基于证据
   的回答及论文页码/Chunk 引用。
2. 面向 15 篇 GAN 与 Diffusion SAR 图像生成论文构建 25 题专业评测集，完成
   Top-k 与来源多样化实验；跨文档策略将全部期望论文召回率由 65% 提升至 90%，
   人工答案评分由 39/50 提升至 43/50。
3. 设计四分类 Intent Router，按请求动态选择 focused RAG、multi-document RAG、
   PaperLibraryTool 或 SourceLookupTool；40 条中英文路由评测中，规则与真实 LLM
   JSON 路由均达到 40/40。
4. 实现 Tool 白名单、参数校验、最多 3 次调用限制、异常隔离与规则 Fallback，
   并完成 Streamlit Agent 集成、结构化日志、依赖锁定及 47 项自动化测试。

### 一行精简版

构建面向 SAR 论文的 RAG + Tool Calling 助手，通过评测驱动的双检索策略将跨文档
全部期望论文召回率从 65% 提升至 90%，并实现可追溯引用、安全 Router 与
Streamlit 交互界面。

### 不应写入简历的表述

- “支持海量论文”——当前真实语料是 15 篇。
- “Router 准确率 100%”但不说明样本——应写“40 条项目测试集 40/40”。
- “完全消除幻觉”——项目只通过 grounded prompt 和拒答机制降低幻觉。
- “生产级高并发”——当前是本地单用户 Streamlit + Chroma。
- “从零原创所有代码”——项目参考过 Datawhale LLM-Universe，应如实说明二次开发。

## 二、30 秒项目介绍

我做了一个面向 SAR 图像生成论文的 RAG + Tool Calling 助手。它会把用户请求分成
论文问答、知识库查询、原文定位和范围外请求。论文问答根据问题选择普通 Top-k 或
跨文档多样化检索，另外两个 Tool 分别查询知识库状态和按 PDF 页码、Chunk 精确
返回原文。我用 15 篇论文构建了 25 题评测集，实验后把跨文档全部期望论文召回率
从 65% 提升到 90%，并加入引用、拒答、路由回退、工具白名单和 Streamlit 界面。

## 三、3 分钟项目介绍

### 1. 背景

这个项目解决的是专业论文问答中的两个问题：第一，大模型不能可靠记住具体论文；
第二，即使回答正确，用户也需要知道答案来自哪篇论文、哪一页。我的语料是 15 篇
英文 SAR 图像生成论文，覆盖 GAN 和 Diffusion 两类方法。

### 2. RAG 主链路

PDF 使用 PyMuPDF 按页解析，每页保留文件名、PDF 页码和 Document ID，再以
600 字符、100 字符重叠切成 Chunk，并生成稳定 Chunk ID。BGE-M3 负责向量化，
Chroma 负责本地持久化。查询时系统先做多轮问题改写，再召回相关 Chunk，把来源
编号和正文放进 grounded prompt，要求模型只能依据证据回答并用 `[1]` 形式引用。

### 3. 评测驱动的检索设计

我没有只凭直觉调 Top-k，而是建立了 25 题评测集，包括事实、比较、跨文档和
无答案题。普通 Top-k 5 对聚焦事实题更稳定，但全部期望论文召回率只有 65%。
针对跨论文问题，我先取 32 个候选，再限制每篇最多贡献 3 个 Chunk，最终返回 8
个 Chunk，全部期望论文召回率达到 90%，人工答案评分从 39/50 提升到 43/50。
由于延迟从 4.12 秒增加到 6.78 秒，而且个别事实题会退化，我没有全局替换检索器，
而是保留两种策略交给 Router 选择。

### 4. Agent 与 Tool Calling

Router 支持 `paper_qa`、`knowledge_base_query`、`source_lookup` 和
`out_of_scope`。它优先调用 LLM 输出 JSON，再把输出转换成枚举和白名单参数；
如果 JSON 非法或 API 失败，就使用确定性规则回退。PaperLibraryTool 不经过向量
检索，直接查询论文和索引状态；SourceLookupTool 使用 Chroma metadata 精确读取
已知页码或 Chunk，这与“不知道答案位置时”的语义 Retriever 是不同职责。

### 5. 工程化和结果

项目加入集中配置、索引 Manifest、日志、结构化异常、Tool 白名单、最多 3 次
调用限制和 Prompt Injection 拒答。最终 40 条 Router 样例中，规则和真实 LLM
路由均为 40/40；47 项自动化测试通过，并完成了 Streamlit 全流程页面测试。

## 四、10 分钟项目介绍提纲

建议配合 README 架构图和 Streamlit Demo，按以下顺序展开。

### 0:00–1:00：问题与目标

- 为什么选择专业论文，而不是通用聊天场景
- SAR 图像生成语料的边界：15 篇英文论文、GAN 与 Diffusion
- 目标：可回答、可拒答、可追溯、可评测、可调用领域 Tool

### 1:00–2:30：索引构建

- PyMuPDF 逐页解析，而不是整篇合成一个 Document
- metadata：file、page、document_id、chunk_id
- Chunk Size 600、Overlap 100
- BGE-M3 归一化向量与 Chroma 持久化
- Manifest 记录论文数、页数、Chunk 数和构建参数

### 2:30–4:00：RAG 执行

- 多轮历史如何用于 query rewrite
- Top-k 检索和 cosine distance 到 similarity 的转换
- Context 中如何标注来源
- grounded prompt、编号引用和无证据拒答
- UI 如何显示实际 retrieval query、参数与来源

### 4:00–5:30：评测与优化

- 25 题构成及 source file/page 标注
- Retrieval Hit、Page Hit、Refusal、人工 0/1/2 评分、Latency
- Top-k 3/5/8/10 实验
- 为什么仅增加 Top-k 不能解决来源集中
- 候选 32 + Top-k 8 + 每篇最多 3 Chunk 的选择依据
- 为什么最终保留双策略而不是全局使用最优平均指标

### 5:30–7:30：Router 与两个 Tool

- 四类 Intent 和 40 条路由测试集
- LLM JSON 优先、规则 Fallback
- 从不信任模型返回的函数名，只转换为内部白名单操作
- PaperLibraryTool 为什么查询文件和 Manifest
- SourceLookupTool 为什么使用 metadata 精确读取而不是 embedding
- Streamlit 如何统一呈现 RAGResponse 和 Tool JSON

### 7:30–8:30：安全与异常

- action、paper_name、page_number、chunk_id 校验
- 文件名路径穿越防护
- ToolCallBudget 最多 3 次
- Tool Exception 与非法返回格式隔离
- 不向 Agent 暴露 Shell 和文件删除能力
- Prompt Injection 路由为 out-of-scope

### 8:30–9:30：结果

- 15 篇、218 页、2142 Chunk
- 跨文档全部期望论文召回率：65% → 90%
- 人工答案评分：39/50 → 43/50
- 拒答行为正确率：84% → 96%
- Router：40/40；自动化测试：47 项
- 明确延迟代价：4.12 秒 → 6.78 秒

### 9:30–10:00：限制与下一步

- 当前结论只适用于项目语料和小规模评测集
- 不支持 OCR、图表视觉理解、复杂公式恢复和并发索引重建
- 可独立增加 3–5 篇中文论文评测跨语言检索
- 可增加 4–6 篇 SAR 检测/识别论文评测跨任务路由
- 优先根据真实用户和面试反馈改进，不继续无目标堆框架

## 五、12 个高频面试问题

### 1. 为什么使用 RAG，而不是把整篇 PDF 放进上下文？

整篇输入会让上下文长度、延迟和费用随论文数量快速增加，而且无关内容会稀释证据。
RAG 先检索少量相关 Chunk，再让模型回答，能够扩展到多篇论文并保留来源 metadata。
我的项目还需要跨论文比较，所以不能假设每次只输入一篇完整 PDF。代价是系统质量
依赖 Retriever，因此我为检索单独建立了可重复评测。

### 2. Chunk Size 和 Chunk Overlap 如何选择？

当前索引使用 600 字符和 100 字符重叠。较小 Chunk 定位更精确，但容易切断完整
论述；较大 Chunk 上下文完整，但每个召回单元包含更多无关信息。Overlap 用来保留
跨边界语义，但过大会增加重复向量和上下文冗余。这个参数属于索引构建参数，修改后
必须重建 Chroma，不能只改 UI Slider 就宣称生效。

### 3. Top-k 太大和太小分别有什么问题？

Top-k 太小容易遗漏跨论文问题中的第二、第三个来源；太大会增加延迟、上下文长度和
无关证据。实验中 Top-k 从 3 增至 8 提高了多论文覆盖，但 Top-k 10 没有继续提高
期望来源召回。因此我对 focused 使用 5，对 multi-document 使用 8，并结合来源
上限控制集中度。

### 4. Temperature 为什么不会改变 Retriever 的结果？

Retriever 使用 embedding 相似度和确定性的候选筛选，Temperature 是生成模型的
采样参数，只影响 query rewrite 或答案生成。相同 retrieval query 和索引下，修改
答案生成 Temperature 不会改变 Chroma 的距离结果；如果 query rewrite 输出发生
变化，才可能间接改变检索词，这需要区分直接作用和间接作用。

### 5. 为什么需要 metadata？

向量本身只能支持相似度搜索，不能可靠回答“来自哪篇、哪页、哪个 Chunk”。项目用
metadata 完成引用展示、评测命中计算、每篇论文 Chunk 上限、SourceLookupTool
精确查询和索引审计。没有 metadata，系统即使生成正确答案也缺少可追溯性。

### 6. 如何实现论文页码引用？

PDF Loader 按页创建 Document，并在切分时把一基 PDF 页码传递给每个 Chunk。
检索后系统把 Chunk 编号为 `[1]`、`[2]`，Prompt 要求答案使用同样编号引用；
Streamlit 再将编号映射回文件名、页码和 Chunk ID。需要注意这是 PDF 文件页码，
不一定等于论文印刷页码，因此 UI 明确标注 PDF Page。

### 7. 知识库没有答案时如何减少幻觉？

我使用三层约束：Prompt 明确要求只依据给定上下文；没有检索结果时直接返回统一拒答；
评测集中加入 5 道确认无答案的问题，测量拒答行为。优化策略的拒答正确率是 96%，
不是 100%，所以我会说“降低幻觉”，不会说“消除幻觉”。后续还可增加相关性阈值
或单独的证据充分性判断。

### 8. Chroma 保存的是什么？

Chroma 保存每个 Chunk 的 embedding、原文和 metadata，并使用稳定 Chunk ID 作为
记录 ID。项目还在独立 Manifest 中记录构建时间、论文数、页数、Chunk 参数和
embedding 模型，用于判断当前 PDF 集合与索引是否一致。LLM 的答案不保存在 Chroma。

### 9. PaperLibraryTool 为什么不用 RAG？

“有多少篇论文”“Chunk Size 是多少”属于确定性系统状态，答案来自文件目录和索引
Manifest。使用 RAG 会增加延迟，还可能根据论文正文生成错误数字。Tool 直接读取
结构化状态，结果更快、更准确，也更容易测试。

### 10. SourceLookupTool 和 Retriever 有什么区别？

Retriever 用于不知道答案在哪里的情况，根据 query embedding 做近似语义搜索。
SourceLookupTool 用于已经知道 PDF、页码或 Chunk 的情况，通过 metadata 和 ID
精确读取，不计算 query embedding。把两者分开能避免用户要求“第 3 页原文”时，
Retriever 却返回语义相似但页码不同的文本。

### 11. Intent Router 路由错误怎么办？

第一层是严格 JSON 字段和枚举校验，非法结果不会直接执行；第二层是 LLM API 或
解析失败时使用规则 Fallback；第三层是 Dispatcher 再校验 Tool 名称与参数。范围外
请求拒答，Tool 异常转换为结构化错误。项目用四类共 40 条样例评测路由，但小数据集
不代表线上绝对准确，因此页面会显示 Intent 和 Strategy，方便发现误路由。

### 12. 哪些来自 LLM-Universe，哪些是自己新增的？

项目最初参考 LLM-Universe 学习 PDF Loader、文本切分、embedding、向量数据库和
基础 RAG 思路。在此基础上，我围绕学术论文场景新增了 PDF 上传与质量校验、稳定
metadata/Chunk ID、索引 Manifest、页码引用、拒答评测、25 题专业数据集、Top-k
与来源多样化实验、双检索策略、两个领域 Tool、JSON Router、白名单与 Fallback、
结构化日志、47 项测试以及 Streamlit Agent 集成。我会把“参考学习”和“个人工程
实现”明确区分，而不是把项目包装成完全从零。

## 六、Demo 建议

推荐控制在 3–5 分钟，按下面顺序演示：

1. 展示侧栏的 15 篇论文、2142 Chunk 和索引参数。
2. 问一个单论文事实题，展示 `focused`、引用页码和实际 retrieval query。
3. 问一个跨论文比较题，展示 `multi_document` 和多个文件来源。
4. 问“当前知识库有多少篇论文”，展示 PaperLibraryTool 不走 Retriever。
5. 指定真实 PDF 第 1 页，展示 SourceLookupTool 返回原始 Chunk。
6. 输入删除文件或天气请求，展示 `out_of_scope` 安全拒答。
7. 最后展示评测报告中的 65% → 90% 及延迟代价，不只展示成功案例。

演示前应预加载 BGE-M3，并确认 `.env`、15 篇 PDF 与 Chroma 索引位于本机；屏幕
录制时不要打开 `.env` 或输出 API Key。
