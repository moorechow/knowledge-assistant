# `src/` 源码深度分析

> 分析日期: 2026-06-25 &emsp; 文件数: 22 个 Python 文件

---

## 目录分层

```
src/
├── config.py              ← 配置层（基础）
├── core/                  ← 引擎层
├── memory/                ← 记忆层
├── knowledge/             ← 知识库层
├── skills/                ← 技能层
├── assistant.py           ← 编排层
├── cli.py / web.py        ← 入口层
```

**依赖方向：上层依赖下层，下层不感知上层**

```
cli.py / web.py
     ↓
assistant.py
     ↓
core/  +  memory/  +  knowledge/  +  skills/
     ↓
config.py
```

---

## 一、配置层 — [config.py](src/config.py)（127 行）

### 设计思路

采用 **双层配置 + 单例模式**。用 5 个 dataclass 组织配置项，避免散落的字典键值对。

### 核心代码逐段分析

#### Dataclass 定义（第 24-78 行）

```python
@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True
```

每个子配置都是独立的 `@dataclass`，好处：

- **类型安全**：IDE 能自动补全和类型检查
- **默认值内嵌**：不加载任何配置文件也能运行
- **易于测试**：可以直接 `LLMConfig(temperature=0.3)` 构造

5 个配置类的关系：

```
AppConfig (根)
  ├── openai_api_key, openai_api_base  ← 来自 .env
  ├── llm_model, embedding_model       ← 来自 .env
  ├── chroma_persist_dir, log_level    ← 来自 .env
  ├── llm: LLMConfig                   ← settings.yaml 覆盖
  ├── memory_config: MemoryConfig      ← settings.yaml 覆盖
  ├── knowledge_config: KnowledgeConfig← settings.yaml 覆盖
  └── retry_config: RetryConfig        ← settings.yaml 覆盖
```

#### 路径处理（第 74-78 行）

```python
def __post_init__(self):
    if not Path(self.chroma_persist_dir).is_absolute():
        self.chroma_persist_dir = str(
            PROJECT_ROOT / self.chroma_persist_dir
        )
```

`__post_init__` 是 dataclass 的特殊方法，在 `__init__` 之后自动调用。这里将相对路径 `./data/chroma_db` 转为基于项目根的绝对路径，避免运行时工作目录变化导致找不到数据库。

#### 配置加载函数（第 81-116 行）

```python
def load_config() -> AppConfig:
    # Step 1: 加载 .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Step 2: 用环境变量构造 AppConfig
    cfg = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        ...
    )

    # Step 3: 加载 YAML，逐段覆盖子配置
    yaml_cfg = _load_yaml_config(PROJECT_ROOT / "config" / "settings.yaml")

    if "llm" in yaml_cfg:
        for k, v in yaml_cfg["llm"].items():
            if hasattr(cfg.llm, k):
                setattr(cfg.llm, k, v)
    ...
```

**优先级链**：`代码默认值 → .env → settings.yaml`

**批量处理技巧**（第 106-114 行）：

```python
for section, target in [
    ("memory", cfg.memory_config),
    ("knowledge", cfg.knowledge_config),
    ("retry", cfg.retry_config),
]:
    if section in yaml_cfg:
        for k, v in yaml_cfg[section].items():
            if hasattr(target, k):
                setattr(target, k, v)
```

用元组列表驱动循环，避免为 memory / knowledge / retry 各写一遍相同的 `if + for + setattr`。但 `llm` 段因为 target 是 `cfg.llm` 而非 `cfg.llm_config`，所以单独处理。

#### 单例缓存（第 119-126 行）

```python
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

模块级变量 `_config` 作为缓存。`load_config()` 只执行一次（读取文件、解析 YAML），后续调用 `get_config()` 直接返回缓存。这是典型的**懒加载单例**模式。

> ⚠️ 注意：这不是线程安全的。如果多线程同时首次调用，`load_config()` 可能执行多次（虽然结果相同，不会出错）。对于 CLI/Streamlit 应用无影响。

---

## 二、核心引擎层 — `core/`

### 2.1 [engine.py](src/core/engine.py)（111 行）— LLM 调用引擎

#### 设计思路

封装 LangChain 的 `ChatOpenAI`，提供**重试、流式、超时**三种增强能力。只依赖 `config.py`，不依赖项目中其他模块。

#### 角色映射函数（第 18-23 行）

```python
def _map_role(role: str) -> type:
    return {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }.get(role, HumanMessage)
```

将字符串 role 映射为 LangChain 的消息类。字典查找 O(1)，`get(role, HumanMessage)` 保证未知 role 降级为 user 消息而非崩溃。这是一个纯函数，不依赖 `self`，所以定义为模块级函数而非方法。

#### 初始化（第 36-45 行）

```python
def __init__(self):
    cfg = get_config()
    self._llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_api_base,
    )
    self._retry_cfg = cfg.retry_config
```

关键点：`base_url=cfg.openai_api_base` 使得 `ChatOpenAI` 可以连接到任何 OpenAI 兼容服务（DeepSeek、Ollama、vLLM 等），不限于 OpenAI 官方 API。这是整个项目"模型可替换"的关键。

#### 同步调用 + 指数退避（第 47-72 行）

```python
def chat(self, messages: list[dict], **kwargs) -> str:
    langchain_messages = [
        _map_role(m["role"])(content=m["content"]) for m in messages
    ]

    last_error = None
    delay = self._retry_cfg.initial_delay

    for attempt in range(self._retry_cfg.max_retries + 1):
        try:
            response = self._llm.invoke(langchain_messages, **kwargs)
            return response.content
        except Exception as e:
            last_error = e
            if attempt < self._retry_cfg.max_retries:
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%s 秒后重试: %s",
                    attempt + 1, self._retry_cfg.max_retries, delay, e,
                )
                time.sleep(delay)
                delay *= self._retry_cfg.backoff_factor

    raise RuntimeError(
        f"LLM 调用失败，已重试 {self._retry_cfg.max_retries} 次"
    ) from last_error
```

重试时间线（max_retries=3, initial_delay=1.0, backoff_factor=2.0）：

```
attempt 0: 立即调用 → 失败 → sleep(1.0s)
attempt 1: 1.0s 后  → 失败 → sleep(2.0s)
attempt 2: 3.0s 后  → 失败 → sleep(4.0s)
attempt 3: 7.0s 后  → 失败 → 抛出异常
```

`from last_error` 保留了原始异常链，方便调试。

#### 异步流式调用（第 74-102 行）

```python
async def chat_stream(
    self, messages: list[dict], **kwargs
) -> AsyncIterator[str]:
    langchain_messages = [
        _map_role(m["role"])(content=m["content"]) for m in messages
    ]

    delay = self._retry_cfg.initial_delay

    for attempt in range(self._retry_cfg.max_retries + 1):
        try:
            async for chunk in self._llm.astream(
                langchain_messages, **kwargs
            ):
                if chunk.content:
                    yield chunk.content
            return     # ← 成功完成，退出重试循环
        except Exception as e:
            if attempt < self._retry_cfg.max_retries:
                logger.warning(
                    "流式调用失败（第 %d/%d 次），重试中: %s",
                    attempt + 1, self._retry_cfg.max_retries, e,
                )
                await asyncio.sleep(delay)
                delay *= self._retry_cfg.backoff_factor
            else:
                raise RuntimeError(
                    f"流式调用失败，已重试 {self._retry_cfg.max_retries} 次"
                ) from e
```

与同步版的关键差异：

- 使用 `self._llm.astream()` 异步流式方法
- `yield chunk.content` 逐 token 产出，调用方用 `async for` 消费
- `await asyncio.sleep(delay)` 而非 `time.sleep(delay)`——前者释放事件循环，后者会阻塞整个线程
- `if chunk.content:` 过滤空 chunk（某些模型会在流结束时发送空内容）

#### 超时控制（第 104-110 行）

```python
async def chat_with_timeout(
    self, messages: list[dict], timeout: float = 30.0, **kwargs
) -> str:
    return await asyncio.wait_for(
        asyncio.to_thread(self.chat, messages, **kwargs),
        timeout=timeout,
    )
```

`asyncio.to_thread()` 将同步的 `chat()` 放到线程池执行，`asyncio.wait_for()` 给它加上超时。这样同步调用也能享受异步超时控制。

### 2.2 [session.py](src/core/session.py)（90 行）— 会话管理

#### Message 数据类（第 10-14 行）

```python
@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str
    metadata: dict = field(default_factory=dict)
```

用 `field(default_factory=dict)` 而非 `= {}`，避免 Python 的可变默认参数陷阱——每个 Message 实例会得到自己的空 dict。

#### SessionManager（第 18-89 行）

核心数据结构是 `_history: list[Message]`，一个线性的消息时间线。

**添加消息**（第 42-55 行）：

```python
def add_user_message(self, content: str, metadata: dict | None = None):
    self._history.append(
        Message(role="user", content=content, metadata=metadata or {})
    )
```

`metadata or {}` 处理了 `None` 入参，避免后续代码访问 `.get()` 时崩溃。

**历史窗口截断**（第 57-68 行）——这是最核心的方法：

```python
def get_history_messages(self, max_turns: int | None = None) -> list[dict]:
    """返回适合 LLM API 的消息列表（含 system prompt）。"""
    limit = max_turns or self._max_turns
    recent = self._history[-limit * 2:]  # *2 因为每轮 = user + assistant

    messages: list[dict] = []
    if self._system_prompt:
        messages.append(
            {"role": "system", "content": self._system_prompt}
        )

    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})
    return messages
```

设计要点：

- 用 Python 切片 `[-limit*2:]` 取最后 N 条，O(k) 复杂度
- System prompt 永远在最前面，且不被窗口截断影响
- 返回 `list[dict]` 而非 `list[Message]`，因为下游（LLM API）只需要 dict 格式

**对话轮数计算**（第 77-80 行）：

```python
@property
def turn_count(self) -> int:
    """当前对话轮数。"""
    return len([m for m in self._history if m.role == "user"])
```

以 user 消息数计轮，而非 `len(self._history) // 2`。这样做更健壮：即使历史中出现连续 user 消息等异常情况，计数也不出错。

**clear vs reset**（第 82-89 行）：

```python
def clear(self):
    """清空历史，保留 system prompt。"""
    self._history.clear()

def reset(self, system_prompt: str = ""):
    """完全重置（同时更新 system prompt）。"""
    self._system_prompt = system_prompt
    self._history.clear()
```

语义区分：`clear()` = "换个话题"，`reset()` = "换个助手"。

---

## 三、记忆层 — `memory/`

### 3.1 [base.py](src/memory/base.py)（47 行）— 抽象基类

```python
@dataclass
class MemoryEntry:
    """一条记忆条目。"""
    content: str
    source: str = "conversation"   # conversation / explicit / inferred
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    importance: float = 0.5        # 0-1
    metadata: dict = field(default_factory=dict)
```

设计意图：

- `source` 区分记忆来源：对话自动记录 / 用户手动输入 / 系统推断
- `importance` 预留字段，当前未使用，未来可实现"重要记忆优先保留"
- `created_at` 时间戳用 ISO 格式字符串，方便 JSON 序列化

```python
class BaseMemory(ABC):
    """记忆模块抽象基类。"""

    @abstractmethod
    def add(self, entry: MemoryEntry): ...

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]: ...

    @abstractmethod
    def clear(self): ...

    @abstractmethod
    def count(self) -> int: ...
```

四个抽象方法构成统一的记忆接口，三个具体实现类（ShortTerm / LongTerm / Profile）必须实现全部四个。

### 3.2 [short_term.py](src/memory/short_term.py)（54 行）— 短期记忆

**数据结构**：`deque[MemoryEntry]`，两端队列，设 `maxlen`。

```python
def __init__(self, max_entries: int = 40):
    """
    Args:
        max_entries: 最大保留消息数（user + assistant 按条计）。
                     默认 40 条 ≈ 20 轮对话。
    """
    self._max_entries = max_entries
    self._buffer: deque[MemoryEntry] = deque(maxlen=max_entries)
```

Python 的 `deque(maxlen=N)` 自动实现滑动窗口：当队列满时，`append()` 会从左侧静默弹出最老元素。无需手动检查长度。

**查询策略**（第 30-34 行）：

```python
def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
    """短期记忆按时间倒序返回最近 N 条，不做语义检索。"""
    result = list(self._buffer)[-top_k:]
    result.reverse()
    return result
```

完全忽略 `query_text`！短期记忆不做语义检索，仅按时间倒序返回最近 N 条。这是合理的设计选择：

- 对话上下文具有强时间局部性（刚说的内容最相关）
- 在 CPU 上做语义检索反而增加延迟，收益不大
- 保持了实现简洁

`result.reverse()` 保证返回顺序为"从旧到新"（最近的最后），更符合人类阅读习惯。

**对话轮次批量添加**（第 25-28 行）：

```python
def add_conversation_turn(self, user_msg: str, assistant_msg: str):
    """一次添加一轮对话。"""
    self.add(MemoryEntry(content=user_msg, source="user"))
    self.add(MemoryEntry(content=assistant_msg, source="assistant"))
```

**上下文化**（第 46-53 行）：

```python
def to_context_text(self, max_turns: int = 10) -> str:
    """将最近 N 轮对话转为上下文字符串。"""
    recent = list(self._buffer)[-max_turns * 2:]
    lines = []
    for entry in recent:
        role = "用户" if entry.source == "user" else "助手"
        lines.append(f"{role}: {entry.content}")
    return "\n".join(lines)
```

将内部数据结构转为 LLM 可读的自然语言格式。中文化 role 标签让 LLM 更容易理解。

### 3.3 [long_term.py](src/memory/long_term.py)（120 行）— 长期记忆

#### 存储方案

直接使用 `chromadb.PersistentClient` + `OpenAIEmbeddings`，**没有**走 LangChain 的 `Chroma` 封装（与知识库的 `VectorRetriever` 不同）。

```python
def __init__(self, collection_name: str = "long_term_memory"):
    cfg = get_config()
    self._persist_dir = Path(cfg.chroma_persist_dir)
    self._persist_dir.mkdir(parents=True, exist_ok=True)

    self._client = chromadb.PersistentClient(
        path=str(self._persist_dir)
    )
    self._embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
        openai_api_base=cfg.openai_api_base,
    )
    self._collection = self._client.get_or_create_collection(
        name=collection_name
    )
```

> ⚠️ **关键问题**：这里用的是 `OpenAIEmbeddings`（调用远程 API），而知识库用的是 `LocalEmbeddings`（本地模型）。虽然都指向 `cfg.embedding_model`（`BAAI/bge-small-zh-v1.5`），但 DeepSeek 的 Embedding API 可能不支持这个模型名。应该统一改为 `LocalEmbeddings`。

#### 写入（第 41-57 行）

```python
def add(self, entry: MemoryEntry):
    """添加一条长期记忆（向量化后存储）。"""
    memory_id = str(uuid.uuid4())
    try:
        self._collection.add(
            ids=[memory_id],
            documents=[entry.content],
            metadatas=[{
                "source": entry.source,
                "created_at": entry.created_at,
                "importance": entry.importance,
                "extra": json.dumps(entry.metadata, ensure_ascii=False),
            }],
        )
    except Exception:
        logger.exception("写入长期记忆失败")
        raise
```

- `uuid.uuid4()` 生成全局唯一 ID，避免冲突
- `ensure_ascii=False` 保证中文元数据不被转义
- ChromaDB 的 `add` 接口要求所有参数都是 list（批量添加），单条也需要包在 `[]` 中

#### 检索（第 59-86 行）

```python
def query(
    self, query_text: str, top_k: int | None = None
) -> list[MemoryEntry]:
    """语义检索最相关的 top_k 条长期记忆。"""
    top_k = top_k or get_config().memory_config.long_term_top_k
    count = self.count()
    if count == 0:
        return []                      # 空库直接返回，避免无意义的查询

    results = self._collection.query(
        query_texts=[query_text],
        n_results=min(top_k, count),   # 防止请求超过实际存储量
    )
```

`min(top_k, count)` 是关键防护：如果只存了 2 条记忆却请求 5 条，ChromaDB 会报错。

**结果解包**（第 73-86 行）比较繁琐，因为 ChromaDB 返回的是嵌套结构：

```python
results["ids"][0]        # 所有返回的 ID
results["documents"][0]  # 所有返回的文档（可能为 None）
results["metadatas"][0]  # 所有返回的元数据（可能为 None）
```

需要处理 `None` 情况（用 `if results.get("documents")` 判断）。

#### 清空（第 95-103 行）

```python
def clear(self):
    """清空集合（删除并重建）。"""
    try:
        self._client.delete_collection(self._collection_name)
    except Exception:
        pass            # 集合不存在也不报错
    self._collection = self._client.create_collection(
        name=self._collection_name
    )
```

`delete_collection` + `create_collection` 的策略，比逐条删除 ID 高效得多。外层 `try/except: pass` 保证了 idempotent（重复清空不报错）。

### 3.4 [profile.py](src/memory/profile.py)（109 行）— 用户画像

#### 数据结构

画像底层是一个普通的 `dict[str, dict]`，键为简短标识符，值为内容字典。持久化到 `data/profile.json`。

#### 文件加载 + 容错（第 31-41 行）

```python
def _load(self):
    """从文件加载画像数据。"""
    if self._file_path.exists():
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("用户画像文件损坏，将重新创建")
            self._data = {}
    else:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
```

文件损坏不会导致程序崩溃，静默重建空画像。

#### 双写入接口（第 48-62 行）

```python
def add(self, entry: MemoryEntry):
    """将 MemoryEntry 存入画像（使用 content 的前缀作为 key）。"""
    key = entry.content[:60]
    self._data[key] = {
        "content": entry.content,
        "source": entry.source,
        "created_at": entry.created_at,
        "metadata": entry.metadata,
    }
    self._save()

def set(self, key: str, value: Any):
    """直接设置键值。"""
    self._data[key] = value
    self._save()
```

- `add()` — 兼容 `BaseMemory` 接口，自动从 content 截取前 60 字符做 key
- `set()` — 更友好的直接键值接口（`assistant.remember_preference(key, value)` 最终调用它）

每次写入后立即 `_save()`，保证数据不丢失。缺点是频繁写入时 I/O 开销大；对于画像这种低频操作完全可接受。

#### 查询（第 67-83 行）

```python
def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
    """
    简单关键词匹配检索画像。
    未来可升级为向量检索。
    """
    results = []
    query_lower = query_text.lower()
    for key, val in self._data.items():
        content = (
            val.get("content", "") if isinstance(val, dict) else str(val)
        )
        if query_lower in content.lower() or query_lower in key.lower():
            results.append(MemoryEntry(
                content=content,
                source=val.get("source", "profile") if isinstance(val, dict) else "profile",
                created_at=val.get("created_at", "") if isinstance(val, dict) else "",
                metadata=val.get("metadata", {}) if isinstance(val, dict) else {},
            ))
    return results[:top_k]
```

朴素的关键词匹配，O(n) 遍历。注释中明确标注"未来可升级为向量检索"，说明开发者知道这是临时方案。

#### 上下文化（第 100-108 行）

```python
def to_context_text(self) -> str:
    """将画像转为上下文字符串，注入 system prompt。"""
    if not self._data:
        return ""
    lines = ["[用户画像]"]
    for key, val in self._data.items():
        content = (
            val.get("content", str(val)) if isinstance(val, dict) else str(val)
        )
        lines.append(f"- {content}")
    return "\n".join(lines)
```

这个输出会被插入 system prompt，让 LLM "了解"用户偏好。例如：

```
[用户画像]
- preferred_language: 中文
- 熟悉 Python、Docker、Linux
```

---

## 四、知识库层 — `knowledge/`

### 4.1 [embeddings.py](src/knowledge/embeddings.py)（53 行）— 本地 Embedding

#### 镜像设置（第 13-14 行）

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

`setdefault` 只在不存在的 key 上设置值。意味着：

- 如果用户已在环境变量中设置了 `HF_ENDPOINT`，使用用户的值
- 如果没设置，默认使用 HF 镜像（国内加速下载）
- 必须在 `import sentence_transformers` **之前**执行，否则环境变量无效

后面的注释 `# noqa: E402` 是因为 import 不在文件顶部（必须先设置环境变量再导入），告诉 linter 忽略此警告。

#### LocalEmbeddings 类（第 24-52 行）

```python
class LocalEmbeddings(Embeddings):
    """
    基于 sentence-transformers 的本地 Embedding。

    使用示例:
        embeddings = LocalEmbeddings()
        vectors = embeddings.embed_documents(["文本 A", "文本 B"])
        query_vec = embeddings.embed_query("搜索关键词")
    """

    def __init__(self, model_name: str | None = None):
        cfg = get_config()
        self._model_name = model_name or cfg.embedding_model
        logger.info("加载本地 Embedding 模型: %s", self._model_name)
        logger.info("下载镜像: %s", os.environ.get("HF_ENDPOINT"))
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=True,
        )
```

继承 LangChain 的 `Embeddings` 抽象类，使其可以无缝接入 LangChain 生态（`Chroma(embedding_function=...)` 接受任何 `Embeddings` 子类）。

**两个关键方法**：

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    embeddings = self._model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

def embed_query(self, text: str) -> List[float]:
    embedding = self._model.encode(
        [text], normalize_embeddings=True
    )
    return embedding[0].tolist()
```

- `normalize_embeddings=True` — 将向量归一化为单位向量，使得余弦相似度等价于点积（更高效）
- `embed_documents` 返回 `List[List[float]]`（批量），`embed_query` 返回 `List[float]`（单条）
- `.tolist()` 将 numpy array 转为 Python list，方便序列化

### 4.2 [loader.py](src/knowledge/loader.py)（84 行）— 文档加载

#### 注册表模式（第 18-24 行）

```python
# 支持的文件扩展名到加载器的映射
LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}
```

用字典做扩展名到 Loader 的映射，添加新格式只需加一行，符合开闭原则。

#### 单文档加载（第 27-54 行）

```python
def load_document(file_path: str) -> list:
    """
    加载单个文档。
    Returns:
        LangChain Document 列表。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    loader_cls = LOADER_REGISTRY.get(ext)
    if not loader_cls:
        raise ValueError(
            f"不支持的文件类型: {ext}（支持: {list(LOADER_REGISTRY.keys())}）"
        )

    logger.info("正在加载文档: %s", path.name)
    loader = loader_cls(str(path))
    documents = loader.load()

    # 为每个文档添加来源元数据
    for doc in documents:
        if "source" not in doc.metadata:
            doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)
        doc.metadata["file_type"] = ext

    logger.info("文档加载完成: %d 页/段", len(documents))
    return documents
```

- `.suffix.lower()` 统一处理 `.PDF` / `.Pdf` 等大小写变体
- 为每个文档片段的元数据注入 `source` / `file_path` / `file_type`，供检索结果展示来源
- `if "source" not in doc.metadata` 只在 Loader 未设置 source 时补充，不覆盖已有值

#### 目录批量加载（第 57-84 行）

```python
def load_documents_from_directory(
    directory: str, recursive: bool = False
) -> list:
    """
    加载目录下所有支持的文档。
    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    all_docs = []
    pattern = "**/*" if recursive else "*"
    for file_path in dir_path.glob(pattern):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in LOADER_REGISTRY:
            continue          # 静默跳过不支持的类型
        try:
            docs = load_document(str(file_path))
            all_docs.extend(docs)
        except Exception:
            logger.exception("加载文件失败: %s", file_path.name)
                                # 单个文件失败不影响其他文件

    logger.info(
        "目录加载完成: %d 个文档, %d 个分段",
        len(set(d.metadata["source"] for d in all_docs)), len(all_docs),
    )
    return all_docs
```

设计考量：

- 默认非递归，需要递归时传 `recursive=True`
- 不支持的文件类型静默跳过（不报错），单个加载失败也继续（logging + 继续）
- 适用场景：用户指定一个大目录，程序尽力加载所有能识别的文档

### 4.3 [splitter.py](src/knowledge/splitter.py)（43 行）— 文本切分

```python
def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建文本切片器。"""
    cfg = get_config().knowledge_config
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or cfg.chunk_size,
        chunk_overlap=chunk_overlap or cfg.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            " ",
            "",
        ],
        keep_separator=True,
    )
```

**分隔符优先级**：从粗到细依次尝试

```
"\n\n"  →  段落边界（最优）
"\n"    →  行边界
"。"    →  中文句号
"！"    →  中文感叹号
"？"    →  中文问号
"；"    →  中文分号
" "     →  英文空格
""      →  字符级（最后手段）
```

`RecursiveCharacterTextSplitter` 的行为：先尝试用 `\n\n` 分割，如果分出的块超过 `chunk_size`，再用 `\n` 分，逐级下沉，直到字符级分割。这样能最大化保持语义完整性。

`keep_separator=True` 保留分隔符在切分后的文本中，避免句号丢失导致句子粘连。

### 4.4 [retriever.py](src/knowledge/retriever.py)（79 行）— 向量检索

#### 初始化（第 20-31 行）

```python
class VectorRetriever:
    """向量检索器，封装 ChromaDB 的存储和查询。"""

    def __init__(self, collection_name: str = "knowledge_base"):
        cfg = get_config()
        persist_dir = Path(cfg.chroma_persist_dir) / collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings = LocalEmbeddings()
        self._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
        )
        self._top_k = cfg.knowledge_config.retrieval_top_k
```

- 用 `collection_name` 做子目录，使得长期记忆（`long_term_memory`）和知识库（`knowledge_base`）的数据物理隔离
- `persist_dir.mkdir(parents=True, exist_ok=True)` 确保目录存在
- 使用 LangChain 的 `Chroma` 封装（与 `LongTermMemory` 直接使用 chromadb 客户端形成对比）

#### 批量添加（第 33-36 行）

```python
def add_documents(self, documents: list):
    logger.info("正在向量化 %d 个文档块...", len(documents))
    self._vector_store.add_documents(documents)
    logger.info("向量化完成，当前库总量: %d", self.count())
```

#### 检索方法（第 38-52 行）

```python
def search(
    self, query: str, top_k: int | None = None
) -> list[tuple[str, dict, float]]:
    k = top_k or self._top_k
    count = self.count()
    if count == 0:
        return []

    results = self._vector_store.similarity_search_with_score(
        query, k=min(k, count)
    )
    return [
        (doc.page_content, doc.metadata, score)
        for doc, score in results
    ]
```

- `similarity_search_with_score` 返回 `(Document, 距离分数)` 元组
- 分数越小表示越相似（ChromaDB 默认使用余弦距离 = 1 - 余弦相似度）

#### 上下文格式化（第 54-69 行）

```python
def search_as_context(
    self, query: str, top_k: int | None = None
) -> str:
    results = self.search(query, top_k)
    if not results:
        return ""

    lines = ["[相关知识库内容]"]
    for i, (content, meta, score) in enumerate(results, start=1):
        source = meta.get("source", "未知来源")
        lines.append(
            f"\n--- 片段 {i}（来源: {source}，相关度: {score:.3f}）---"
        )
        lines.append(content)

    return "\n".join(lines)
```

这是直接给 LLM 看的格式化文本，示例输出：

```
[相关知识库内容]

--- 片段 1（来源: python_guide.md，相关度: 0.152）---
装饰器是 Python 中的语法糖...

--- 片段 2（来源: notes.txt，相关度: 0.231）---
Python 装饰器本质上是一个接受函数...
```

---

## 五、技能层 — `skills/`

### 5.1 [registry.py](src/skills/registry.py)（103 行）— Skill 框架

#### 设计思路

实现了类似 OpenAI Function Calling 的插件架构。每个 Skill 描述自己的名称、功能和参数 schema，可以被 LLM 自主选择调用。

#### SkillResult（第 19-24 行）

```python
@dataclass
class SkillResult:
    """Skill 执行结果。"""
    success: bool
    data: Any = None
    error: str = ""
```

统一的返回值格式。无论 Skill 执行成功还是失败，调用方都拿到同一结构。失败时 `success=False` + `error` 含原因。

#### BaseSkill（第 27-54 行）

```python
class BaseSkill(ABC):
    """Skill 抽象基类。"""

    name: str = ""
    description: str = ""
    parameters_schema: dict = field(default_factory=dict)

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult: ...

    def to_tool_schema(self) -> dict:
        """转为 OpenAI Function Calling 格式的 tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
```

`to_tool_schema()` 输出格式直接兼容 OpenAI 的 tool calling API。存在 `parameters_schema` 为空的情况时用默认空 schema，保证始终合法。

#### SkillRegistry（第 56-103 行）

```python
class SkillRegistry:
    """Skill 注册中心。"""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册一个 Skill。"""
        if skill.name in self._skills:
            logger.warning("Skill %s 已注册，将被覆盖", skill.name)
        self._skills[skill.name] = skill
        logger.info("Skill 已注册: %s", skill.name)

    def unregister(self, name: str):
        """卸载 Skill。"""
        if name in self._skills:
            del self._skills[name]

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def execute(self, name: str, **kwargs) -> SkillResult:
        """按名称执行 Skill。"""
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(
                success=False,
                error=f"Skill 不存在: {name}（可用: {self.list_names()}）",
            )
        try:
            return skill.execute(**kwargs)
        except Exception as e:
            logger.exception("Skill %s 执行失败", name)
            return SkillResult(success=False, error=str(e))

    def get_tool_schemas(self) -> list[dict]:
        """获取所有 Skill 的 tool schema 列表（用于 Function Calling）。"""
        return [s.to_tool_schema() for s in self._skills.values()]
```

设计要点：

- `register` 允许覆盖已注册的 Skill（warning 但不停）
- `execute` 捕获所有异常，确保调用方不会因 Skill 内部错误而崩溃——Skill 永远是"安全失败"的
- `get_tool_schemas()` 批量导出所有 Skill schema，一次性传给 LLM 做 function calling 决策

### 5.2 [summarize.py](src/skills/builtin/summarize.py)（40 行）— 摘要 Skill

```python
class SummarizeSkill(BaseSkill):
    """文本摘要 Skill，需配合 LLM 引擎使用。"""

    name = "summarize"
    description = "对长文本进行摘要总结。输入 text 参数即可获得摘要。"

    def __init__(self, llm_engine=None):
        """
        Args:
            llm_engine: LLMEngine 实例，如果不传则返回原文前 200 字。
        """
        self._llm = llm_engine

    def execute(
        self, text: str = "", max_length: int = 200, **kwargs
    ) -> SkillResult:
        if not text:
            return SkillResult(success=False, error="待摘要文本不能为空")

        if self._llm:
            messages = [
                {
                    "role": "system",
                    "content": f"请对以下内容进行摘要，控制在 {max_length} 字以内。",
                },
                {"role": "user", "content": text},
            ]
            summary = self._llm.chat(messages)
            return SkillResult(success=True, data={"summary": summary})
        else:
            # 无 LLM 时简单截断
            summary = text[:max_length] + (
                "..." if len(text) > max_length else ""
            )
            return SkillResult(
                success=True,
                data={
                    "summary": summary,
                    "truncated": len(text) > max_length,
                },
            )
```

两种模式：

- **有 LLM** → 调用 LLM 做摘要（高质量）
- **无 LLM** → 简单截断前 200 字（降级方案）

这种设计让 Skill 在无 API 环境下也能提供基本功能，不会直接报错。

### 5.3 [search.py](src/skills/builtin/search.py)（39 行）— 搜索 Skill（占位）

```python
class WebSearchSkill(BaseSkill):
    """网络搜索 Skill（使用搜索引擎 URL 生成搜索链接）。"""

    name = "web_search"
    description = (
        "搜索互联网获取信息。当需要查找最新资讯、事实性知识时使用。"
    )

    def __init__(
        self, search_engine: str = "https://www.bing.com/search?q="
    ):
        self._search_engine = search_engine

    def execute(self, query: str = "", **kwargs) -> SkillResult:
        if not query:
            return SkillResult(success=False, error="搜索关键词不能为空")

        search_url = self._search_engine + quote(query)
        # 这里只返回搜索链接，实际抓取需要进一步实现
        return SkillResult(
            success=True,
            data={
                "query": query,
                "search_url": search_url,
                "tip": "可扩展为调用搜索 API 获取实际结果（如 SerpAPI / Bing API）",
            },
        )
```

只构造了搜索 URL，没有实际抓取。`tip` 字段直接告诉调用方"这是占位实现"，是诚实的工程做法。

---

## 六、编排层 — [assistant.py](src/assistant.py)（234 行）

### 初始化 — 组装全部模块（第 26-48 行）

```python
class KnowledgeAssistant:
    """知识库 AI 助手的主控制器。"""

    def __init__(self, system_prompt: str = ""):
        cfg = get_config()

        # 核心引擎
        self._engine = LLMEngine()
        self._session = SessionManager(
            system_prompt=system_prompt or self._default_system_prompt(),
            max_turns=cfg.memory_config.short_term_max_turns,
        )

        # 三层记忆
        self._short_term = ShortTermMemory(
            max_entries=cfg.memory_config.short_term_max_turns * 2
        )
        self._long_term = LongTermMemory()
        self._profile = ProfileMemory()

        # 知识库
        self._knowledge = VectorRetriever()

        # Skill 系统
        self._skills = SkillRegistry()
        self._skills.register(SummarizeSkill(llm_engine=self._engine))
```

这里体现了依赖注入的雏形：`SummarizeSkill(llm_engine=self._engine)` 把 LLM 引擎传入 Skill，使得 Skill 具备了调用 LLM 的能力。

### 系统提示词（第 52-60 行）

```python
def _default_system_prompt(self) -> str:
    return (
        "你是一个个人知识库 AI 助手。你可以：\n"
        "1. 基于用户的知识库文档回答问题\n"
        "2. 记住用户的关键偏好和事实\n"
        "3. 搜索互联网获取新信息\n"
        "4. 对长篇内容进行摘要\n\n"
        "请始终保持专业、准确的回答风格。"
    )
```

### 上下文组装 — 整个系统的核心（第 125-170 行）

```python
def _build_context_for_llm(self, user_input: str) -> list[dict]:
    """构建发给 LLM 的完整消息列表。"""

    # 1. 检索知识库
    knowledge_context = self._knowledge.search_as_context(user_input)

    # 2. 检索长期记忆
    long_term_entries = self._long_term.query(user_input)

    # 3. 用户画像
    profile_text = self._profile.to_context_text()

    # 4. 组装增强后的 system prompt
    enhanced_parts = []
    if profile_text:
        enhanced_parts.append(profile_text)
    if long_term_entries:
        lt_lines = ["[相关长期记忆]"]
        for entry in long_term_entries:
            lt_lines.append(f"- {entry.content}")
        enhanced_parts.append("\n".join(lt_lines))

    base_sp = self._session.system_prompt
    if enhanced_parts:
        full_sp = "\n\n".join(enhanced_parts) + "\n\n" + base_sp
    else:
        full_sp = base_sp

    # 5. 构建消息列表
    messages: list[dict] = [{"role": "system", "content": full_sp}]

    # 6. 知识库上下文
    if knowledge_context:
        messages.append({
            "role": "system",
            "content": knowledge_context,
        })

    # 7. 对话历史
    history = self._session.get_history_messages()
    messages.extend(history)

    # 8. 当前用户输入
    messages.append({"role": "user", "content": user_input})

    return messages
```

**最终发给 LLM 的消息结构：**

```
[0] system: "[用户画像]\n- preferred_language:...\n\n[相关长期记忆]\n- ...\n\n你是一个个人知识库 AI 助手..."

[1] system: "[相关知识库内容]\n--- 片段 1 ---\n..."

[2] user: "你好"                      ← 历史（window 截断后）
[3] assistant: "你好！有什么可以帮助你的？"
[4] user: "Python 装饰器怎么用？"      ← 当前输入
```

设计考量：

- 用户画像和长期记忆融合到第一个 system prompt 中（"你是谁 + 你知道用户什么"）
- 知识库内容作为独立的第二个 system 消息（"你有这些参考资料"）
- 让 LLM 能区分"关于用户的背景知识"和"可参考的文档资料"

### 对话方法（第 172-205 行）

```python
def chat(self, user_input: str) -> str:
    """同步对话。"""
    # 记录用户输入到短期记忆
    self._short_term.add_conversation_turn(user_input, "")
    self._session.add_user_message(user_input)

    # 构建上下文并调用 LLM
    messages = self._build_context_for_llm(user_input)
    response = self._engine.chat(messages)

    # 记录回复
    self._short_term.add(MemoryEntry(
        content=response, source="assistant"
    ))
    self._session.add_assistant_message(response)

    return response
```

注意执行顺序：**先在 session 中记录用户消息 → 再构建上下文 → 调用 LLM → 记录回复**。这个顺序保证了上下文构建时不包含当前轮的用户消息（避免重复，因为 `_build_context_for_llm` 最后会手动 append）。

异步流式版本 `chat_stream()` 逻辑相同，只是用 `async for` 逐 token 收集：

```python
async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
    """异步流式对话。"""
    self._short_term.add_conversation_turn(user_input, "")
    self._session.add_user_message(user_input)

    messages = self._build_context_for_llm(user_input)

    full_response = ""
    async for token in self._engine.chat_stream(messages):
        full_response += token
        yield token

    self._short_term.add(MemoryEntry(
        content=full_response, source="assistant"
    ))
    self._session.add_assistant_message(full_response)
```

### 会话管理接口（第 209-221 行）

```python
def clear_conversation(self):
    """清空当前对话，保留知识和记忆。"""
    self._session.clear()
    self._short_term.clear()

def reset(self, system_prompt: str = ""):
    """完全重置助手。"""
    self._session.reset(system_prompt)
    self._short_term.clear()
```

### 统计信息（第 225-233 行）

```python
def stats(self) -> dict:
    return {
        "short_term_entries": self._short_term.count(),
        "long_term_entries": self._long_term.count(),
        "profile_keys": self._profile.count(),
        "knowledge_chunks": self._knowledge.count(),
        "registered_skills": self._skills.count(),
        "session_turns": self._session.turn_count,
    }
```

---

## 七、入口层

### 7.1 [cli.py](src/cli.py)（162 行）— CLI 入口

#### Banner（第 30-38 行）

```python
def print_banner():
    print(r"""
  _   __                           __              __   __       _
 | | / /                          / /             / /  / _|     (_)
 | |/ /  ___  _   _ ___  _   __ / /__   ___     / /__| |_ ___   _  ___  _ __
 |    \ / _ \| | | / __|| | / // //_/  / _ \   / //_/|  _// _ \ | |/ _ \| '_ \
 | |\  \  __/| |_| \__ \| |/ // ,<    | (_) | / /  _| | | (_) || |  __/| | | |
 \_| \_/\___| \__,_|___/ \___//_/|_|   \___/  \/  (_)_| |_|\___/ |_|\___||_| |_|
    """)
```

#### 交互循环（第 41-118 行）

```python
async def interactive_mode(assistant):
    """交互式对话模式。"""
    print_banner()
    print("个人知识库 AI 助手  v0.1.0")
    print("输入 'quit' 或 'exit' 退出，输入 '/help' 查看命令")
    print(f"已加载知识库: {assistant.knowledge_count} 条")
    print(f"已注册 Skill: {assistant.skill_names}")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        elif user_input == "/help":
            print("""
命令列表:
  /stats    - 查看统计信息
  /clear    - 清空当前对话
  /ingest <path> - 导入文档/目录到知识库
  /skills   - 查看已注册的 Skill
  /summarize <text> - 对文本进行摘要
  /help     - 显示此帮助
            """)
            continue
        elif user_input == "/stats":
            stats = assistant.stats()
            print("\n[统计信息]")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue
        elif user_input == "/clear":
            assistant.clear_conversation()
            print("[对话已清空]")
            continue
        elif user_input == "/skills":
            print(f"\n已注册 Skill: {assistant.skill_names}")
            continue
        elif user_input.startswith("/ingest "):
            path = user_input[len("/ingest "):].strip()
            try:
                p = Path(path)
                if p.is_dir():
                    assistant.ingest_directory(str(p))
                else:
                    assistant.ingest_document(str(p))
                print(f"[已导入: {path}]")
            except Exception as e:
                print(f"[导入失败: {e}]")
            continue
        elif user_input.startswith("/summarize "):
            text = user_input[len("/summarize "):].strip()
            result = assistant.execute_skill("summarize", text=text)
            if result.success:
                print(f"\n[摘要]\n{result.data['summary']}")
            else:
                print(f"[摘要失败: {result.error}]")
            continue

        # 正常对话
        try:
            print("\n助手: ", end="", flush=True)
            async for token in assistant.chat_stream(user_input):
                print(token, end="", flush=True)
            print()
        except Exception as e:
            logger.exception("对话出错")
            print(f"\n[错误: {e}]")
```

命令分发用 `if/elif` 链而非字典映射，对于 6 个命令的场景代码可读性高于过度抽象。

`flush=True` 确保 token 立即显示在终端（不使用行缓冲），实现打字机效果。

#### main 入口（第 120-161 行）

```python
def main():
    parser = argparse.ArgumentParser(description="个人知识库 AI 助手")
    parser.add_argument(
        "--ingest",
        type=str,
        help="启动时导入指定文档或目录到知识库",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API Key（也可通过 OPENAI_API_KEY 环境变量设置）",
    )
    args = parser.parse_args()

    setup_logging()

    # 如果命令行传入了 API Key，设置到环境变量
    if args.api_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.api_key

    from src.assistant import KnowledgeAssistant

    assistant = KnowledgeAssistant()

    # 启动时导入知识
    if args.ingest:
        p = Path(args.ingest)
        try:
            if p.is_dir():
                assistant.ingest_directory(str(p))
            else:
                assistant.ingest_document(str(p))
        except Exception as e:
            logger.error("启动导入失败: %s", e)

    # 进入交互模式
    asyncio.run(interactive_mode(assistant))
```

支持启动参数：
- `python -m src.cli --api-key sk-xxx` — 命令行传入 API Key
- `python -m src.cli --ingest ./docs/` — 启动时自动导入目录

### 7.2 [web.py](src/web.py)（178 行）— Streamlit Web 入口

#### 页面配置（第 21-26 行）

```python
st.set_page_config(
    page_title="知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

#### 自定义样式（第 28-37 行）

```python
st.markdown("""
<style>
.stMainBlockContainer { max-width: 1200px; }
.stChatMessage { border-radius: 12px; }
.stSidebar .stMetric { font-size: 13px; }
.upload-section { border: 1px dashed #ccc; border-radius: 12px; padding: 16px; margin: 12px 0; }
.stats-label { font-size: 12px; color: #999; }
</style>
""", unsafe_allow_html=True)
```

#### 缓存单例（第 42-45 行）

```python
@st.cache_resource
def get_assistant() -> KnowledgeAssistant:
    """全局单例助手，序列化缓存避免重复初始化。"""
    return KnowledgeAssistant()
```

`st.cache_resource` 是 Streamlit 的资源缓存装饰器。每次用户交互都会重新执行脚本，但这个函数只执行一次——`KnowledgeAssistant` 在多次 rerun 之间保持是同一个实例。这对 LLM 引擎初始化（可能加载模型）和数据库连接尤为重要。

#### 侧边栏 — 三块功能区

**统计面板**（第 68-75 行）：

```python
stats = assistant.stats()
col1, col2 = st.columns(2)
with col1:
    st.metric("知识块", stats["knowledge_chunks"])
    st.metric("长期记忆", stats["long_term_entries"])
with col2:
    st.metric("对话轮次", stats["session_turns"])
    st.metric("已注册技能", stats["registered_skills"])
```

用 `st.metric` 展示实时统计，每次 rerun 刷新。

**文档上传**（第 80-105 行）：

```python
uploaded = st.file_uploader(
    "上传文档（PDF / Markdown / TXT / DOCX）",
    type=["pdf", "md", "txt", "docx"],
    accept_multiple_files=True,
    key="kb_uploader",
)
if uploaded:
    progress = st.progress(0, text="准备导入...")
    for i, file in enumerate(uploaded):
        progress.progress(
            (i + 1) / len(uploaded),
            text=f"导入中: {file.name}",
        )
        tmp_path = Path(tempfile.gettempdir()) / file.name
        with open(tmp_path, "wb") as f:
            f.write(file.getbuffer())
        try:
            assistant.ingest_document(str(tmp_path))
        except Exception as e:
            st.error(f"导入失败: {file.name} - {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()   # 用完即删
    progress.empty()
    st.success(f"已导入 {len(uploaded)} 个文件")
    st.rerun()
```

Streamlit 的 `file_uploader` 返回的是 `UploadedFile` 对象（内存中的字节流），而 `load_document()` 需要文件路径。因此需要：

1. 写入临时目录 `tempfile.gettempdir()`
2. 导入知识库
3. 立即删除临时文件

`progress.progress()` 按文件进度更新进度条，`progress.empty()` 在完成后移除进度条。

**记忆管理**（第 109-123 行）：

利用 Streamlit 的 `text_area`、`text_input` 和 `button` 组件，提供 GUI 方式来添加事实记忆和偏好。

**会话管理**（第 126-134 行）：

```python
with st.expander("⚙️ 会话管理", expanded=False):
    if st.button("清空当前对话", use_container_width=True):
        assistant.clear_conversation()
        st.session_state.messages = []
        st.rerun()
    if st.button("完全重置", use_container_width=True, type="secondary"):
        assistant.reset()
        st.session_state.messages = []
        st.rerun()
```

#### 对话区（第 142-177 行）

```python
# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题，基于知识库回答..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复（流式）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 收集流式 token
                async def _collect():
                    tokens = []
                    async for token in assistant.chat_stream(prompt):
                        tokens.append(token)
                    return "".join(tokens)

                response = asyncio.run(_collect())
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"出错了: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
```

`:=` 海象运算符同时赋值和判断，简洁。但流式处理的实现确实是先全部收集再渲染，因为 Streamlit 的渲染模型是"整个脚本跑完一次性更新 UI"。

---

## 八、模块间依赖关系图

```
                     ┌──────────┐
                     │ config.py│  ← 被所有模块依赖
                     └────┬─────┘
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ core/    │   │ memory/  │   │ knowledge/   │
    │ engine   │   │ base     │   │ embeddings   │
    │ session  │   │ short    │   │ loader       │
    └────┬─────┘   │ long     │   │ splitter     │
         │         │ profile  │   │ retriever    │
         │         └────┬─────┘   └──────┬───────┘
         │              │               │
         │         ┌────┴────┐          │
         │         │ skills/ │          │
         │         │ registry│          │
         │         │ builtin │          │
         │         └────┬────┘          │
         │              │               │
         └──────┬───────┴───────┬───────┘
                ▼               ▼
          ┌──────────────────────────┐
          │     assistant.py         │  ← 编排层
          └──────────┬───────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    ┌──────────┐         ┌──────────┐
    │  cli.py  │         │  web.py  │     ← 入口层
    └──────────┘         └──────────┘
```

---

## 九、单次对话完整数据流

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
│ 3. 读取用户画像    │──→ "[用户画像]\n- preferred_language: 中文\n- 熟悉 Python、Docker"
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

**最终发给 LLM 的消息结构：**

```
[0] system: "[用户画像]\n- preferred_language: 中文\n\n[相关长期记忆]\n- 用户正在学习 Python 进阶\n\n你是一个个人知识库 AI 助手。你可以：\n1. 基于用户的知识库文档回答问题\n..."

[1] system: "[相关知识库内容]\n\n--- 片段 1（来源: python_guide.md，相关度: 0.152）---\n装饰器是 Python 中的语法糖..."

[2] user: "你好"                      ← 历史（window 截断后）
[3] assistant: "你好！有什么可以帮助你的？"
[4] user: "Python 装饰器怎么用？"      ← 当前输入
```

---

## 十、设计模式总结

| 模式 | 位置 | 说明 |
|------|------|------|
| **单例** | `config.py:119-126` | `get_config()` 模块级缓存 |
| **抽象基类** | `memory/base.py:25`，`skills/registry.py:27` | `BaseMemory` / `BaseSkill` 定义统一接口 |
| **注册表** | `knowledge/loader.py:18`，`skills/registry.py:56` | Loader 映射 + Skill 注册中心 |
| **策略模式** | `memory/` | 三种记忆各有不同检索策略（时间/语义/关键词） |
| **模板方法** | `skills/registry.py:27-54` | `BaseSkill.execute()` 定义为抽象方法，子类实现 |
| **依赖注入** | `assistant.py:48` | `SummarizeSkill(llm_engine=self._engine)` |
| **门面模式** | `assistant.py` | `KnowledgeAssistant` 对外暴露简洁接口，隐藏内部复杂度 |
| **MVC 变体** | 整体架构 | assistant=Model, cli/web=View, `_build_context_for_llm`=Controller |

---

## 十一、潜在问题与改进建议

| 问题 | 位置 | 严重程度 | 建议 |
|------|------|----------|------|
| **Embedding 不一致** | `long_term.py:31` | 🔴 中 | 长期记忆使用 `OpenAIEmbeddings`（需 API），应统一改用 `LocalEmbeddings` |
| **Web 非真流式** | `web.py:161-167` | 🟡 低 | `asyncio.run(_collect())` 收集完才渲染，应用 `st.write_stream()` 实现真流式 |
| **搜索 Skill 占位** | `search.py:30-37` | 🟡 低 | 仅返回搜索 URL，未实现页面抓取和结果提取 |
| **画像检索低效** | `profile.py:67-83` | 🟡 低 | 关键词匹配 O(n)，大量画像时应升级为向量检索 |
| **无 Token 管理** | 全局 | 🔴 中 | 缺少 Token 计数和上下文窗口溢出保护 |
| **无错误中间层** | `assistant.py` | 🔴 中 | 异常直接向上抛出，缺少统一的错误降级策略 |
| **LongTermMemory 用 API Embedding** | `long_term.py:31-35` | 🟡 低 | 虽然模型名配置为本地模型，但底层调用的是 OpenAI API，实际不可用 |
| **config 单例非线程安全** | `config.py:119-126` | 🟢 极低 | 对 CLI/Streamlit 无实际影响 |
