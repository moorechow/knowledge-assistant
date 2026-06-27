# 📚 知识库 AI 助手 (Knowledge Assistant) — 完整项目分析

> **版本**: v0.1.0 &emsp; **分析日期**: 2026-06-25

---

## 一、项目概述

这是一个基于 **RAG（检索增强生成）** 架构的个人知识库 AI 助手，具备 **CLI + Web 双入口**，使用 DeepSeek 大模型 + 本地 Embedding 实现知识管理、三层记忆和可插拔 Skill 系统。

- **LLM**: DeepSeek-Chat（OpenAI 兼容接口）
- **Embedding**: BAAI/bge-small-zh-v1.5（本地离线，512 维，中文优化）
- **向量库**: ChromaDB（嵌入式，零配置持久化）
- **前端**: Streamlit Web UI + CLI 交互式命令行

---

## 二、技术栈全景

| 层面 | 技术 | 说明 |
|------|------|------|
| **LLM** | DeepSeek-Chat (OpenAI 兼容) | 通过 `ChatOpenAI(base_url=...)` 接入 |
| **Embedding** | `BAAI/bge-small-zh-v1.5` | 512维，中文优化，本地运行无需 API |
| **向量数据库** | ChromaDB (嵌入式) | 零配置持久化存储 |
| **LLM 框架** | LangChain 0.3+ | `ChatOpenAI` / `Chroma` / `RecursiveTextSplitter` |
| **Web 前端** | Streamlit 1.35+ | 现代化聊天 UI（侧边栏 + 对话区） |
| **CLI 前端** | Python `argparse` + `asyncio` | 交互式命令行，流式输出 |
| **配置管理** | `.env` + `YAML` | 环境变量 + 文件双层配置 |
| **文档解析** | PyPDF / UnstructuredMarkdownLoader / Docx2txtLoader | PDF / Markdown / TXT / Word |
| **Embedding 库** | sentence-transformers 3.0+ | 通过 HF Mirror 下载模型 |
| **测试** | pytest 8.0 + pytest-asyncio | 记忆/会话单元测试 |

---

## 三、项目结构（逐层解剖）

```
knowledge-assistant/
│
├── .env                          # API Key 等敏感配置（DeepSeek）
├── .env.example                  # 配置模板
│
├── config/
│   └── settings.yaml             # 业务配置（LLM/记忆/知识库/重试参数）
│
├── .streamlit/
│   └── config.toml               # Streamlit 配置（端口8501，无头模式）
│
├── src/                          # === 核心源码 ===
│   ├── __init__.py
│   ├── config.py                 # 🔧 配置管理模块
│   ├── assistant.py              # 🧠 主控制器（整合所有模块）
│   ├── cli.py                    # 💻 CLI 入口
│   ├── web.py                    # 🌐 Web 入口
│   │
│   ├── core/                     # ⚙️ 核心引擎
│   │   ├── __init__.py
│   │   ├── engine.py             #   LLM 调用引擎（重试/流式/超时）
│   │   └── session.py            #   会话管理（历史窗口/上下文组装）
│   │
│   ├── memory/                   # 🧠 三层记忆系统
│   │   ├── __init__.py
│   │   ├── base.py               #   记忆基类 + MemoryEntry 数据类
│   │   ├── short_term.py         #   短期记忆（deque 滑动窗口）
│   │   ├── long_term.py          #   长期事实记忆（ChromaDB 向量化）
│   │   └── profile.py            #   用户画像（JSON 键值对持久化）
│   │
│   ├── knowledge/                # 📖 知识库模块
│   │   ├── __init__.py
│   │   ├── loader.py             #   文档加载（PDF/MD/TXT/DOCX）
│   │   ├── splitter.py           #   文本切分（递归字符分割）
│   │   ├── embeddings.py         #   本地 Embedding（sentence-transformers）
│   │   └── retriever.py          #   向量检索器（ChromaDB 封装）
│   │
│   └── skills/                   # 🔌 可插拔 Skill 系统
│       ├── __init__.py
│       ├── registry.py           #   Skill 注册中心 + BaseSkill + SkillResult
│       └── builtin/
│           ├── __init__.py
│           ├── summarize.py      #   文本摘要 Skill
│           └── search.py         #   网络搜索 Skill（占位）
│
├── tests/
│   ├── __init__.py
│   └── test_memory.py            # 记忆系统 + 会话管理单元测试
│
├── data/                         # 持久化数据目录
│   ├── chroma_db/                # ChromaDB 向量数据
│   └── profile.json              # 用户画像
│
├── requirements.txt              # 依赖清单
├── verify.py                     # LLM + Embedding 连通性验证脚本
└── venv/                         # Python 虚拟环境
```

---

## 四、核心模块详解

### 4.1 配置系统 — [config.py](src/config.py)

双层配置架构：

- **`.env`** → 敏感信息（API Key、Base URL、模型名）
- **`config/settings.yaml`** → 业务参数（温度、top-k、chunk 大小、重试策略）

通过 5 个 dataclass 组织配置：

| Dataclass | 关键字段 |
|-----------|---------|
| `LLMConfig` | `model`, `temperature`, `max_tokens`, `streaming` |
| `MemoryConfig` | `short_term_max_turns`, `long_term_top_k`, `profile_similarity_threshold` |
| `KnowledgeConfig` | `chunk_size`, `chunk_overlap`, `retrieval_top_k` |
| `RetryConfig` | `max_retries`, `initial_delay`, `backoff_factor` |
| `AppConfig` | 根配置，聚合上述子配置 + API Key / Base URL / 路径 |

使用模块级单例 `get_config()` 全局复用，第一次调用后缓存。

### 4.2 LLM 引擎 — [engine.py](src/core/engine.py)

| 特性 | 实现 |
|------|------|
| 连接方式 | `ChatOpenAI(base_url="https://api.deepseek.com")` |
| 角色映射 | `system→SystemMessage`, `user→HumanMessage`, `assistant→AIMessage` |
| 同步调用 | `chat()` — 指数退避重试（默认最多 3 次） |
| 流式调用 | `chat_stream()` — `AsyncIterator[str]`，逐 token 产出 |
| 超时控制 | `chat_with_timeout()` — 包装 `asyncio.wait_for` + `to_thread` |

重试策略：`delay = initial_delay * backoff_factor^attempt`（1s → 2s → 4s）

### 4.3 会话管理 — [session.py](src/core/session.py)

`SessionManager` 负责多轮对话状态维护：

- 维护 `Message(role, content, metadata)` 列表
- 滑动窗口：`max_turns=20`（默认），每轮 = user + assistant，超过则丢弃最早轮次
- `get_history_messages()` 返回 `[{role, content}]` 格式，直接传给 LLM API
- `clear()` — 清空历史但保留 system prompt
- `reset()` — 完全重置（可换新的 system prompt）

### 4.4 三层记忆系统

```
┌─────────────────────────────────────────────────────┐
│  短期记忆 (ShortTermMemory)                           │
│  文件: src/memory/short_term.py                       │
│  实现: collections.deque 滑动窗口                      │
│  用途: 最近 N 轮对话上下文                             │
│  检索: 时间倒序返回最近 N 条                           │
│  持久化: ❌ 进程重启丢失                               │
│  容量: max_entries=40（≈ 20 轮对话）                  │
├─────────────────────────────────────────────────────┤
│  长期记忆 (LongTermMemory)                            │
│  文件: src/memory/long_term.py                        │
│  实现: ChromaDB 向量存储 + Embedding 语义检索          │
│  用途: 关键事实、知识的持久化记忆                       │
│  检索: 语义相似度 top-k 查询                           │
│  持久化: ✅ 磁盘持久化到 data/chroma_db/              │
│  检索数: top_k=5（可配置）                             │
│  ⚠️ Embedding: 使用 OpenAI API（非本地模型）           │
├─────────────────────────────────────────────────────┤
│  用户画像 (ProfileMemory)                             │
│  文件: src/memory/profile.py                          │
│  实现: JSON 文件键值对存储                             │
│  用途: 用户偏好、特征（语言、技能等）                    │
│  检索: 简单关键词匹配（可升级为向量检索）                │
│  持久化: ✅ data/profile.json                         │
└─────────────────────────────────────────────────────┘
```

> ⚠️ **注意**：[long_term.py](src/memory/long_term.py) 中使用了 `OpenAIEmbeddings` 而非 `LocalEmbeddings`，这意味着长期记忆仍需调用 DeepSeek API 进行向量化，与知识库模块使用的本地 Embedding 不一致。建议统一使用本地模型。

### 4.5 知识库模块

数据流：`文档 → Loader → Splitter → LocalEmbeddings → ChromaDB`

| 子模块 | 文件 | 功能 |
|--------|------|------|
| 文档加载 | [loader.py](src/knowledge/loader.py) | `LOADER_REGISTRY` 映射 `.pdf/.txt/.md/.docx` 到对应 LangChain Loader；支持单文件和目录批量加载 |
| 文本切分 | [splitter.py](src/knowledge/splitter.py) | `RecursiveCharacterTextSplitter`，按 `\n\n → \n → 。→ ！→ ？→ ；→ 空格` 优先级切分，chunk_size=1000, overlap=200 |
| Embedding | [embeddings.py](src/knowledge/embeddings.py) | 封装 `SentenceTransformer`，通过 HF Mirror 下载模型，**免费离线**。支持 `embed_documents()` 和 `embed_query()` |
| 向量检索 | [retriever.py](src/knowledge/retriever.py) | 封装 ChromaDB `similarity_search_with_score`，返回 (内容, 元数据, 分数)；`search_as_context()` 格式化输出供 LLM 使用 |

**支持的文档格式：**

| 扩展名 | 加载器 | 说明 |
|--------|--------|------|
| `.pdf` | PyPDFLoader | PDF 文档 |
| `.txt` | TextLoader | 纯文本 |
| `.md` / `.markdown` | UnstructuredMarkdownLoader | Markdown |
| `.docx` | Docx2txtLoader | Word 文档 |

### 4.6 Skill 系统

可插拔架构，每个 Skill 继承 `BaseSkill`，支持导出为 OpenAI Function Calling 格式。

| Skill | 文件 | 状态 | 功能 |
|-------|------|------|------|
| `summarize` | [summarize.py](src/skills/builtin/summarize.py) | ✅ 完整 | 调用 LLM 生成摘要，支持无 LLM 时的降级截断 |
| `web_search` | [search.py](src/skills/builtin/search.py) | ⚠️ 占位 | 仅生成搜索 URL，未实现实际页面抓取 |

**核心类：**

- `SkillResult(success, data, error)` — 统一执行结果
- `BaseSkill` — 抽象基类，定义 `name/description/execute()`，提供 `to_tool_schema()` 导出 Function Calling 格式
- `SkillRegistry` — 注册中心，支持 register/unregister/get/execute/list_names/get_tool_schemas

### 4.7 主控制器 — [assistant.py](src/assistant.py)

`KnowledgeAssistant` 是**所有模块的编排者**，单次对话的完整流程：

```
用户输入
    │
    ├─ 1. 知识库检索 → 语义相关文档片段 (VectorRetriever.search_as_context)
    ├─ 2. 长期记忆检索 → 相关事实记忆 (LongTermMemory.query)
    ├─ 3. 读取用户画像 → 偏好/特征上下文 (ProfileMemory.to_context_text)
    │
    ├─ 4. 组装增强 System Prompt
    │      [用户画像] + [相关长期记忆] + [原始系统提示]
    │
    ├─ 5. 拼接对话历史 → 滑动窗口截断 (SessionManager.get_history_messages)
    │
    ├─ 6. 调用 LLM → 同步 chat() 或流式 chat_stream()
    │
    └─ 7. 记录回复 → ShortTermMemory + SessionManager
```

**暴露的管理接口：**

| 方法 | 功能 |
|------|------|
| `ingest_document(path)` | 导入单个文档到知识库 |
| `ingest_directory(path)` | 批量导入目录下文档 |
| `search_knowledge(query)` | 检索知识库上下文 |
| `remember_fact(content, importance)` | 手动存储长期事实记忆 |
| `remember_preference(key, value)` | 存储用户偏好 |
| `execute_skill(name, **kwargs)` | 按名执行 Skill |
| `clear_conversation()` | 清空当前对话，保留知识与记忆 |
| `reset()` | 完全重置 |
| `stats()` | 返回各部分统计信息 |

---

## 五、双入口对比

| 维度 | CLI ([cli.py](src/cli.py)) | Web ([web.py](src/web.py)) |
|------|---------------------------|---------------------------|
| 启动方式 | `python -m src.cli` | `streamlit run src/web.py` |
| 交互方式 | 终端输入 / 流式字符输出 | Streamlit 聊天 UI |
| 流式输出 | ✅ 真正的逐 token 打印 | ⚠️ 收集完成后一次性渲染 |
| 文档上传 | `/ingest <path>` 命令指定路径 | 侧边栏拖拽上传（支持多文件） |
| 记忆管理 | 无 GUI | 侧边栏表单（事实 + 偏好） |
| 会话管理 | `/clear`、`/stats`、`/skills` 命令 | 侧边栏按钮操作 + 实时统计面板 |
| 知识库状态 | 启动时打印统计 | 侧边栏指标卡片 |
| 适用场景 | 开发调试、极客日常使用 | 日常使用、演示展示 |

**CLI 支持的命令：**

| 命令 | 功能 |
|------|------|
| `quit` / `exit` | 退出 |
| `/help` | 显示帮助 |
| `/stats` | 查看各部分统计信息 |
| `/clear` | 清空当前对话 |
| `/skills` | 查看已注册 Skill 列表 |
| `/ingest <path>` | 导入文档/目录 |
| `/summarize <text>` | 调用摘要 Skill |

---

## 六、架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                         入口层                                   │
│                   CLI (cli.py)    Web (web.py)                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   KnowledgeAssistant    │  ← 主控制器
              │   (assistant.py)        │
              └────┬──────┬──────┬──────┘
                   │      │      │
     ┌─────────────▼┐ ┌───▼────┐ ┌▼───────────┐
     │   core/      │ │ memory/│ │ knowledge/  │
     │  ┌─────────┐ │ │        │ │             │
     │  │ Engine  │ │ │ Short  │ │ Loader      │
     │  │ (LLM)   │ │ │ Term   │ │ Splitter    │
     │  └─────────┘ │ │        │ │ Embeddings  │
     │  ┌─────────┐ │ │ Long   │ │ Retriever   │
     │  │ Session │ │ │ Term   │ │             │
     │  │ Manager │ │ │        │ └─────────────┘
     │  └─────────┘ │ │ Profile│
     └──────────────┘ └────────┘
                           │
              ┌────────────▼────────────┐
              │      skills/            │
              │  ┌──────────────────┐   │
              │  │ SkillRegistry    │   │
              │  │  ├─ summarize    │   │
              │  │  └─ web_search   │   │
              │  └──────────────────┘   │
              └─────────────────────────┘

     ┌──────────────────────────────────────┐
     │           持久化层                    │
     │  ChromaDB (data/chroma_db/)          │
     │  Profile JSON (data/profile.json)    │
     │  Settings YAML (config/settings.yaml)│
     │  .env (API Key)                      │
     └──────────────────────────────────────┘
```

---

## 七、数据流（单次对话）

```
  User Input: "Python 的装饰器怎么用？"
       │
       ▼
┌──────────────────┐
│ 1. 向量检索知识库  │──→ "[相关知识库内容]\n--- 片段1（来源: python_guide.md）---\n装饰器是..."
│   (VectorRetriever)│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 2. 检索长期记忆    │──→ [MemoryEntry("用户正在学习 Python 进阶", source="inferred")]
│   (LongTermMemory) │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 3. 读取用户画像    │──→ ["用户画像"]\n- preferred_language: 中文\n- 熟悉 Python、Docker
│   (ProfileMemory)  │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 4. 组装消息列表    │──→ [system: 画像+记忆+提示词, system: 知识库, user: 历史..., user: 当前输入]
│   (_build_context) │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 5. LLM 调用       │──→ "装饰器是 Python 中的语法糖，使用 @ 符号..." (流式逐 token)
│   (LLMEngine)     │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 6. 写回记忆       │──→ ShortTermMemory.add() + SessionManager.add_assistant_message()
└──────────────────┘
```

---

## 八、架构评价

### 亮点

| 维度 | 说明 |
|------|------|
| **分层清晰** | core / memory / knowledge / skills 四层职责分明，低耦合 |
| **配置灵活** | .env + YAML 双层配置，dataclass 类型安全，支持环境变量注入 |
| **记忆设计** | 三层记忆（短期/长期/画像）各有定位，符合认知架构 |
| **本地 Embedding** | 知识库使用本地模型，零 API 成本离线运行 |
| **Skill 可扩展** | 可插拔架构，支持导出 Function Calling schema，预留 LLM 自主决策能力 |
| **双入口** | CLI 适合开发调试，Web 适合日常使用 |
| **容错设计** | LLM 调用内置指数退避重试 + 超时控制 |
| **测试覆盖** | pytest 覆盖记忆系统和会话管理核心路径 |

### 待改进

| 问题 | 位置 | 严重程度 | 建议 |
|------|------|----------|------|
| **Embedding 不一致** | [long_term.py:31](src/memory/long_term.py#L31) | 中 | 长期记忆使用 `OpenAIEmbeddings`（需 API），应统一改用 `LocalEmbeddings` |
| **Web 非真流式** | [web.py:161-167](src/web.py#L161-L167) | 低 | `asyncio.run(_collect())` 收集完才渲染，应用 `st.write_stream()` 实现真流式 |
| **搜索 Skill 占位** | [search.py:30-37](src/skills/builtin/search.py#L30-L37) | 低 | 仅返回搜索 URL，未实现页面抓取和结果提取 |
| **画像检索低效** | [profile.py:67-83](src/memory/profile.py#L67-L83) | 低 | 关键词匹配 O(n)，大量画像时应升级为向量检索 |
| **无 Token 管理** | 全局 | 中 | 缺少 Token 计数和上下文窗口溢出保护 |
| **无错误中间层** | [assistant.py](src/assistant.py) | 中 | 异常直接向上抛出，缺少统一的错误降级策略 |

---

## 九、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key

# 3. 验证连通性
python verify.py

# 4a. 启动 CLI
python -m src.cli

# 4b. 启动 Web 界面
streamlit run src/web.py
```

---

## 十、依赖清单

```
# 核心框架
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0
langchain-chroma>=0.1.0

# 向量数据库
chromadb>=0.5.0

# 本地 Embedding（免费离线）
sentence-transformers>=3.0.0

# 文档解析
pypdf>=5.0.0
markdown>=3.7
python-docx>=1.1.0
beautifulsoup4>=4.12.0

# 数据存储
sqlalchemy>=2.0.0

# Web 前端
streamlit>=1.35.0

# 配置
PyYAML>=6.0
python-dotenv>=1.0.0

# 工具类
tiktoken>=0.7.0
tqdm>=4.66.0

# 测试
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0
```
