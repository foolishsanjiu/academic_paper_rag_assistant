# Academic Paper RAG Assistant

基于大语言模型和检索增强生成技术构建的学术论文问答系统。

本项目参考 Datawhale LLM-Universe 教程进行学习和二次开发，目标是实现一个面向 SAR 学术论文的 RAG 检索问答应用。

## 项目目标

系统计划支持：

- PDF 学术论文读取与解析
- 文件名、页码和 Chunk ID 等元数据保留
- 论文文本切分
- Embedding 向量生成
- Chroma 向量数据库存储
- Top-k 相关段落检索
- 基于论文内容生成回答
- 显示论文名称和页码引用
- 文献中不存在答案时拒绝回答
- Streamlit 交互页面
- 简单参数评测与问题路由

## 当前进度

- [x] 配置 Conda Python 环境
- [x] 配置 VS Code Python 解释器
- [x] 创建独立项目目录
- [x] 编写 PDF 目录扫描脚本
- [x] 将论文文件信息保存为 JSON
- [x] 配置大模型 API
- [x] 实现流式与非流式调用
- [x] 创建 Streamlit 聊天页面
- [ ] 实现 PDF 文本解析
- [ ] 实现文本切分
- [ ] 创建向量数据库
- [ ] 完成 RAG 问答流程

## 当前目录结构

```text
academic-paper-rag/
├── learning_tasks/
│   └── day02/
│       └── scan_papers.py
├── data/
│   └── papers/
├── .gitignore
└── README.md

## 大模型 API 调用
项目当前通过 OpenAI 兼容客户端调用大模型 API，支持非流式和流式输出。
运行：
```powershell
python llm_client.py

## 启动聊天页面
确保已经在 `.env` 中配置模型 API，然后运行：
```powershell
python -m streamlit run app.py