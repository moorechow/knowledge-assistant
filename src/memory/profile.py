"""
用户画像记忆：键值对形式存储用户偏好 / 特征。
持久化到本地 JSON 文件，支持自动合并与去重。
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .base import BaseMemory, MemoryEntry

logger = logging.getLogger(__name__)


class ProfileMemory(BaseMemory):
    """用户画像记忆，以键值对形式存储。"""

    def __init__(self, file_path: str = "./data/profile.json"):
        """
        Args:
            file_path: JSON 文件路径，相对于项目根目录。
        """
        from src.config import PROJECT_ROOT
        self._file_path = Path(file_path)
        if not self._file_path.is_absolute():
            self._file_path = PROJECT_ROOT / self._file_path
        self._data: dict[str, dict] = {}
        self._load()

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

    def _save(self):
        """持久化到文件。"""
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

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

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        简单关键词匹配检索画像。
        未来可升级为向量检索。
        """
        results = []
        query_lower = query_text.lower()
        for key, val in self._data.items():
            content = val.get("content", "") if isinstance(val, dict) else str(val)
            if query_lower in content.lower() or query_lower in key.lower():
                results.append(MemoryEntry(
                    content=content,
                    source=val.get("source", "profile") if isinstance(val, dict) else "profile",
                    created_at=val.get("created_at", "") if isinstance(val, dict) else "",
                    metadata=val.get("metadata", {}) if isinstance(val, dict) else {},
                ))
        return results[:top_k]

    def remove(self, key: str):
        if key in self._data:
            del self._data[key]
            self._save()

    def clear(self):
        self._data.clear()
        self._save()

    def count(self) -> int:
        return len(self._data)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def to_context_text(self) -> str:
        """将画像转为上下文字符串，注入 system prompt。"""
        if not self._data:
            return ""
        lines = ["[用户画像]"]
        for key, val in self._data.items():
            content = val.get("content", str(val)) if isinstance(val, dict) else str(val)
            lines.append(f"- {content}")
        return "\n".join(lines)
