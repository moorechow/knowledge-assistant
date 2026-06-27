"""
记忆系统基类

本系统设计了三层记忆：
1. 短期记忆（ShortTermMemory）: 内存中的对话历史，滑动窗口。
2. 长期事实记忆（LongTermMemory）: 向量化存储的关键事实，持久化到 ChromaDB。
3. 用户画像记忆（ProfileMemory）: 键值对形式的用户偏好/特征，持久化到 JSON 文件。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """一条记忆条目。"""
    content: str
    source: str = "conversation"   # 来源：conversation / explicit / inferred
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5        # 重要性 0-1
    metadata: dict = field(default_factory=dict)


class BaseMemory(ABC):
    """记忆模块抽象基类。"""

    @abstractmethod
    def add(self, entry: MemoryEntry):
        """添加一条记忆。"""
        ...

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
        """根据查询文本检索相关记忆。"""
        ...

    @abstractmethod
    def clear(self):
        """清空所有记忆。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """当前记忆数量。"""
        ...
