# Environment Setup
## 1. 克隆项目
git clone ...

## 2. 创建 Conda 环境
conda create -n xxx python=...

conda activate xxx

## 3. 安装依赖
python -m pip install -r requirements.txt

## 4. 配置 API
复制：
.env.example
创建：
.env
填写：
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

## 5. 测试配置
python config.py

## 6. 测试模型调用
python llm_client.py

## 7. 启动 Streamlit
python -m streamlit run app.py