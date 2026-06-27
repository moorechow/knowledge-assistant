"""
配置管理模块
从 .env + settings.yaml 加载配置。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True


@dataclass
class MemoryConfig:
    short_term_max_turns: int = 20
    long_term_top_k: int = 5
    profile_similarity_threshold: float = 0.85


@dataclass
class KnowledgeConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class AppConfig:
    # OpenAI 兼容 API 配置（DeepSeek）
    openai_api_key: str = ""
    openai_api_base: str = "https://api.deepseek.com"

    # 模型
    llm_model: str = "deepseek-chat"
    # 本地 Embedding 模型名称（HuggingFace 格式）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # 子配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge_config: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    # 路径
    chroma_persist_dir: str = "./data/chroma_db"
    log_level: str = "INFO"

    def __post_init__(self):
        if not Path(self.chroma_persist_dir).is_absolute():
            self.chroma_persist_dir = str(
                PROJECT_ROOT / self.chroma_persist_dir
            )


def load_config() -> AppConfig:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    cfg = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base=os.getenv(
            "OPENAI_API_BASE", "https://api.deepseek.com"
        ),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    yaml_cfg = _load_yaml_config(PROJECT_ROOT / "config" / "settings.yaml")

    if "llm" in yaml_cfg:
        for k, v in yaml_cfg["llm"].items():
            if hasattr(cfg.llm, k):
                setattr(cfg.llm, k, v)

    for section, target in [
        ("memory", cfg.memory_config),
        ("knowledge", cfg.knowledge_config),
        ("retry", cfg.retry_config),
    ]:
        if section in yaml_cfg:
            for k, v in yaml_cfg[section].items():
                if hasattr(target, k):
                    setattr(target, k, v)

    return cfg


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
