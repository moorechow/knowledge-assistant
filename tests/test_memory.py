"""
测试：记忆系统
"""

import os
import sys
import tempfile
from pathlib import Path

# 确保 src 在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestShortTermMemory:
    """短期记忆测试。"""

    def test_add_and_query(self):
        from src.memory.short_term import ShortTermMemory, MemoryEntry

        mem = ShortTermMemory(max_entries=10)
        mem.add(MemoryEntry(content="你好", source="user"))
        mem.add(MemoryEntry(content="你好！有什么可以帮助你的？", source="assistant"))

        assert mem.count() == 2

        results = mem.query("你好", top_k=1)
        assert len(results) == 1

    def test_conversation_turn(self):
        from src.memory.short_term import ShortTermMemory

        mem = ShortTermMemory(max_entries=10)
        mem.add_conversation_turn("今天天气怎么样？", "今天天气晴朗，适合出门。")

        assert mem.count() == 2
        assert "天气" in mem.to_context_text()

    def test_sliding_window(self):
        from src.memory.short_term import ShortTermMemory, MemoryEntry

        mem = ShortTermMemory(max_entries=5)
        for i in range(10):
            mem.add(MemoryEntry(content=f"消息 {i}", source="user"))

        assert mem.count() == 5  # 只保留最近 5 条
        entries = mem.get_recent(5)
        assert entries[0].content == "消息 5"

    def test_clear(self):
        from src.memory.short_term import ShortTermMemory, MemoryEntry

        mem = ShortTermMemory()
        mem.add(MemoryEntry(content="测试", source="user"))
        mem.clear()
        assert mem.count() == 0


class TestProfileMemory:
    """用户画像记忆测试。"""

    def test_set_and_get(self):
        from src.memory.profile import ProfileMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            mem = ProfileMemory(file_path=tmp_path)
            mem.set("preferred_language", "中文")
            assert mem.get("preferred_language") == "中文"
        finally:
            os.unlink(tmp_path)

    def test_query(self):
        from src.memory.profile import ProfileMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            mem = ProfileMemory(file_path=tmp_path)
            mem.set("skill_level", "熟悉 Python、Docker、Linux")
            mem.set("hobby", "喜欢骑行")

            results = mem.query("Python")
            assert len(results) >= 1
            assert any("Python" in r.content for r in results)
        finally:
            os.unlink(tmp_path)

    def test_clear(self):
        from src.memory.profile import ProfileMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            mem = ProfileMemory(file_path=tmp_path)
            mem.set("test_key", "test_value")
            assert mem.count() == 1
            mem.clear()
            assert mem.count() == 0
        finally:
            os.unlink(tmp_path)

    def test_persistence(self):
        from src.memory.profile import ProfileMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            mem1 = ProfileMemory(file_path=tmp_path)
            mem1.set("name", "张三")
            assert mem1.count() == 1

            # 重新加载，验证持久化
            mem2 = ProfileMemory(file_path=tmp_path)
            assert mem2.get("name") == "张三"
        finally:
            os.unlink(tmp_path)


class TestSessionManager:
    """会话管理测试。"""

    def test_message_flow(self):
        from src.core.session import SessionManager

        session = SessionManager(
            system_prompt="你是一个测试助手",
            max_turns=10,
        )

        session.add_user_message("你好")
        session.add_assistant_message("你好！有什么可以帮助你的？")

        assert session.turn_count == 1

        messages = session.get_history_messages()
        # system + user + assistant = 3
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_sliding_window(self):
        from src.core.session import SessionManager

        session = SessionManager(system_prompt="test", max_turns=3)

        for i in range(5):
            session.add_user_message(f"问题 {i}")
            session.add_assistant_message(f"回答 {i}")

        # 只保留最近 3 轮 = 6 条消息 + system = 7
        messages = session.get_history_messages()
        assert len(messages) == 7
        # 第一组应该是 system
        assert messages[0]["role"] == "system"
        # 最近一轮的 user 消息
        assert "问题 2" in messages[1]["content"]

    def test_clear_and_reset(self):
        from src.core.session import SessionManager

        session = SessionManager(system_prompt="v1")
        session.add_user_message("test")
        assert session.turn_count == 1

        session.clear()
        assert session.turn_count == 0
        assert session.system_prompt == "v1"  # clear 不改变 system prompt

        session.reset("v2")
        assert session.system_prompt == "v2"
        assert session.turn_count == 0
