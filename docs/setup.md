# Environment Setup

## 1. 创建环境

在项目根目录执行：

```bash
conda create -n academic-paper-rag python=3.10
conda activate academic-paper-rag
python -m pip install -r requirements.txt
```

## 2. 配置 API

将 `.env.example` 复制为 `.env`，填写兼容 OpenAI Chat Completions API 的
模型配置：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-endpoint/v1
LLM_MODEL=your_model_name
```

`LLM_BASE_URL` 可留空。先验证环境变量是否完整；此命令不调用 API：

```bash
python config.py
```

如需验证模型连接，可运行交互脚本；它会向配置的模型发送请求，可能产生费用：

```bash
python llm_client.py
```

## 3. 启动 Streamlit

```bash
python -m streamlit run app.py
```

BGE-M3 首次运行时会从 Hugging Face 下载。进入页面后上传 PDF，并点击
“保存所选 PDF / 重建向量库”。本地 PDF、Chroma 数据库、日志和 `.env` 不会提交
到 Git。

## 4. 验证安装

```bash
python -m unittest discover -s tests -v
```
