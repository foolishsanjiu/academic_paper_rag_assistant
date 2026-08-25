# 全流程测试记录

测试日期：2026-08-25  
知识库状态：15 篇 PDF、218 页、2142 个 Chunk

## 自动化回归

- `python -m unittest discover -s tests`：通过。
- `python -m py_compile app.py agent_response.py agent_router.py`：通过。

## Streamlit 页面测试

使用真实本地 Streamlit 页面和浏览器交互完成以下场景：

| 场景 | 输入/操作 | 结果 |
|---|---|---|
| 首次页面加载 | 打开 `http://localhost:8502` | 成功显示 15 篇论文和 2142 个 Chunk |
| 未选择文件 | 点击“保存 PDF 并重建向量库” | 显示“请先选择至少一个 PDF 文件” |
| 未点击上传按钮 | 仅打开页面 | 不再错误显示“没有成功保存任何 PDF” |
| 知识库查询 | “当前知识库有多少篇论文？” | `paper_library` 返回 15，页面显示正确 Intent/Tool |
| 原文定位 | 指定真实 PDF 第 1 页 | `source_lookup` 返回 12 个 Chunk，可展开查看 |
| Prompt Injection | 要求执行 Shell 并删除文件 | 路由为 `out_of_scope`，未调用工具 |
| 跨论文问答 | 比较 ATGAN 与 DiffuSAR | 页面显示 `paper_qa / multi_document` |
| 生成异常 | 浏览器托管进程无法连接外部 API | 页面显示安全错误并保留 Intent/Strategy |

没有在测试中上传或覆盖现有 PDF，也没有重建当前向量库。

## 真实 RAG 链路测试

由于本地浏览器托管进程的网络权限会阻止外部 LLM API，生成链路使用具有
网络权限的同一项目解释器独立验证；使用的模型、Chroma 和代码路径与页面一致。

| 策略 | 问题 | 结果 |
|---|---|---|
| `focused` | `What architecture does ATGAN use?` | 成功生成 694 字符，返回 5 个来源 |
| `multi_document` | `Compare ATGAN and DiffuSAR approaches.` | 成功生成 648 字符，返回 8 个来源，覆盖 4 篇论文 |

## 本轮发现并修复

1. 上传按钮的条件分支错误，导致页面初始状态误报上传失败。
2. RAG 生成异常发生后，页面没有保存已经完成的 Router 决策。

修复后分别通过页面可见状态验证。

## 未执行的破坏性场景

- 未上传重复 PDF。
- 未触发真实索引重建。
- 未删除或替换 Chroma 数据。

这些流程已有参数校验与模块测试；对当前 15 篇论文库执行真实重建会覆盖现用索引，
不适合作为每次回归测试的一部分。
