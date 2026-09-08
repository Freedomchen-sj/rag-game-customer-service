# 游戏领域 RAG 智能客服系统（知更鸟 / GameRobin）

基于 LangChain + Chroma + 阿里云百炼构建的游戏领域多轮对话 RAG 智能客服，覆盖《王者荣耀》《英雄联盟》《无畏契约 VALORANT》《CS:GO/CS2》四类游戏的**游戏理解、英雄出装、玩法思路**问答，支持知识库管理、流式问答与会话记忆持久化。

#注意，由于代码里存在便捷封装函数，需要依赖社区包，所以运行时会爆红色警告，但是可以正常运行，后续需要换成官方包实现

## 功能特性

- **知识库管理端**：上传 TXT 知识文档自动入库；MD5 指纹台账去重，避免同一文档重复向量化；按内容长度阈值决定是否走递归切分，并注入来源、时间、操作者元数据
- **智能问答端**：检索增强生成（RAG），召回片段携带出处元数据注入提示词；Streamlit 打字机流式输出；按 session_id 隔离的多轮会话记忆，历史对话文件级持久化
- **双模型环境**：云端阿里云百炼（qwen3-max + text-embedding-v4）与本地 Ollama 均可运行
- **领域标签化语料**：知识文档采用「【游戏 | 主题】」条目级标签 + 一问一答 FAQ 结构，便于切分后每块自带上下文，降低跨领域误召回

## 技术栈

Python · LangChain（LCEL / RunnableWithMessageHistory）· Chroma · 阿里云百炼 DashScope · Streamlit · Ollama

## 项目结构

```
├── config_data.py          # 全局配置（切分参数、top-k、模型名、路径）
├── knowledge_base.py       # 知识库入库服务（MD5去重 + 递归切分 + 元数据）
├── vector_stores.py        # Chroma 向量库与检索器封装
├── rag.py                  # RAG 核心链（检索增强 + 多轮记忆）
├── file_history_store.py   # 会话历史持久化
├── app_qa.py               # 智能问答端（Streamlit）
├── app_file_uploader.py    # 知识库管理端（Streamlit）
└── data/游戏/              # 游戏领域知识文档（4 款游戏 / 11 篇）
```

## 快速开始

```bash
# 1. 配置百炼 API Key（环境变量）
set DASHSCOPE_API_KEY=你的Key

# 2. 安装依赖
pip install langchain langchain-community langchain-chroma streamlit

# 3. 启动知识库管理端，上传 data/游戏/ 下文档完成入库
streamlit run app_file_uploader.py

# 4. 启动智能问答端
streamlit run app_qa.py
```

## 检索效果评测

> 说明：本仓库沿用统一的检索评测方法论——自建问答测试集，对多种「切分粒度 × top-k」组合跑命中率与 MRR 评测，并对未命中样本做归因分析。游戏领域知识库的评测测试集与具体指标数据待补充（评测脚本与流程同服装版，可按需复现）。

评测通过一次性脚本完成（内存计算，不依赖持久化向量库），调整 `config_data.py` 中 `top_k` 与 `chunk_size` 可复现不同配置。

## 后续规划

- 扩充更多游戏品类（如原神、永劫无间等）的知识库
- 引入 BM25 与向量检索的多路召回融合，改善英雄名/装备名等专有名词的召回
- 基于 LangGraph 实现 ReAct Agent，支持战绩查询、版本更新查询等工具自主调度
