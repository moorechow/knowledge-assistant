"""
会话管理模块
管理多轮对话的上下文组装、历史压缩和 Prompt 拼装。
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Message:
    """单条消息。"""
    role: str          # "system" | "user" | "assistant"
    content: str
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """管理单个会话的对话历史。"""

    def __init__(self, system_prompt: str = "", max_turns: int = 20):
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._history: list[Message] = []
        self._session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        self._session_id = value

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def update_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def add_user_message(self, content: str, metadata: dict | None = None):
        self._history.append(
            Message(role="user", content=content, metadata=metadata or {})
        )

    def add_assistant_message(self, content: str, metadata: dict | None = None):
        self._history.append(
            Message(role="assistant", content=content, metadata=metadata or {})
        )

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        self._history.append(
            Message(role=role, content=content, metadata=metadata or {})
        )

    def get_history_messages(self, max_turns: int | None = None) -> list[dict]:
        """返回适合 LLM API 的消息列表（含 system prompt）。
        
        注意：max_turns 不能超过 self._max_turns，取二者较小值为最终上限。
        """
        if max_turns is None:
            limit = self._max_turns
        else:
            limit = min(max_turns, self._max_turns)
        recent = self._history[-limit * 2:]  # *2 因为每轮 = user + assistant

        messages: list[dict] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def get_history_text(self) -> str:
        """将历史转换为纯文本，用于摘要等场景。"""
        lines = []
        for msg in self._history:
            lines.append(f"[{msg.role}]: {msg.content}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        """当前对话轮数。"""
        return len([m for m in self._history if m.role == "user"])

    def clear(self):
        """清空历史，保留 system prompt。"""
        self._history.clear()

    def reset(self, system_prompt: str = ""):
        """完全重置（同时更新 system prompt）。"""
        self._system_prompt = system_prompt
        self._history.clear()
