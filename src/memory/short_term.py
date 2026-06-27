"""
短期记忆：基于内存的对话历史管理。
"""

from collections import deque

from .base import BaseMemory, MemoryEntry


class ShortTermMemory(BaseMemory):
    """滑动窗口式短期记忆，仅保存在内存中。"""

    def __init__(self, max_entries: int = 40):
        """
        Args:
            max_entries: 最大保留消息数（user + assistant 按条计）。
                         默认 40 条 ≈ 20 轮对话。
        """
        self._max_entries = max_entries
        self._buffer: deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, entry: MemoryEntry):
        self._buffer.append(entry)

    def add_conversation_turn(self, user_msg: str, assistant_msg: str):
        """一次添加一轮对话。"""
        self.add(MemoryEntry(content=user_msg, source="user"))
        self.add(MemoryEntry(content=assistant_msg, source="assistant"))

    def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
        """短期记忆按时间倒序返回最近 N 条，不做语义检索。"""
        result = list(self._buffer)[-top_k:]
        result.reverse()
        return result

    def get_recent(self, n: int) -> list[MemoryEntry]:
        """获取最近 n 条记忆（按时间正序）。"""
        return list(self._buffer)[-n:]

    def clear(self):
        self._buffer.clear()

    def count(self) -> int:
        return len(self._buffer)

    def to_context_text(self, max_turns: int = 10) -> str:
        """将最近 N 轮对话转为上下文字符串。"""
        recent = list(self._buffer)[-max_turns * 2:]
        lines = []
        for entry in recent:
            role = "用户" if entry.source == "user" else "助手"
            lines.append(f"{role}: {entry.content}")
        return "\n".join(lines)
