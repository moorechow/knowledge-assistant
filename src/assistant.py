"""
知识库助手 - 主应用逻辑
整合 LLM 引擎 + 三层记忆 + 知识库检索 + Skill 系统。
"""

import logging
from typing import AsyncIterator, Optional

from src.config import get_config
from src.core import LLMEngine, SessionManager
from src.memory import (
    ShortTermMemory,
    LongTermMemory,
    ProfileMemory,
    MemoryEntry,
)
from src.knowledge import VectorRetriever, load_document, split_documents
from src.skills import SkillRegistry, SummarizeSkill

logger = logging.getLogger(__name__)


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

    # ---- 系统提示词 ----

    def _default_system_prompt(self) -> str:
        return (
            "你是一个个人知识库 AI 助手。你可以：\n"
            "1. 基于用户的知识库文档回答问题\n"
            "2. 记住用户的关键偏好和事实\n"
            "3. 搜索互联网获取新信息\n"
            "4. 对长篇内容进行摘要\n\n"
            "请始终保持专业、准确的回答风格。"
        )

    def update_system_prompt(self, prompt: str):
        self._session.update_system_prompt(prompt)

    # ---- 知识库管理 ----

    def ingest_document(self, file_path: str):
        """导入单个文档到知识库。"""
        docs = load_document(file_path)
        chunks = split_documents(docs)
        self._knowledge.add_documents(chunks)
        logger.info("知识库已导入: %s (%d 块)", file_path, len(chunks))

    def ingest_directory(self, directory: str):
        """导入目录下所有文档到知识库。"""
        from src.knowledge import load_documents_from_directory
        docs = load_documents_from_directory(directory)
        if not docs:
            logger.warning("目录 %s 中未找到支持的文档", directory)
            return
        chunks = split_documents(docs)
        self._knowledge.add_documents(chunks)

    def search_knowledge(self, query: str) -> str:
        """检索知识库上下文。"""
        return self._knowledge.search_as_context(query)

    @property
    def knowledge_count(self) -> int:
        return self._knowledge.count()

    # ---- 记忆管理 ----

    def remember_fact(self, content: str, importance: float = 0.5):
        """手动存储一条长期事实记忆。"""
        self._long_term.add(MemoryEntry(
            content=content,
            source="explicit",
            importance=importance,
        ))

    def remember_preference(self, key: str, value: str):
        """存储用户偏好。"""
        self._profile.set(key, value)

    def forget_fact(self, content: str):
        """删除匹配的长期记忆（通过清空重建简化实现）。"""
        logger.info("长期记忆删除功能通过清空重建实现")
        self._long_term.clear()

    # ---- Skill 管理 ----

    def register_skill(self, skill):
        self._skills.register(skill)

    def execute_skill(self, name: str, **kwargs):
        return self._skills.execute(name, **kwargs)

    @property
    def skill_names(self) -> list[str]:
        return self._skills.list_names()

    # ---- 对话核心 ----

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

    # ---- 会话管理 ----

    def clear_conversation(self):
        """清空当前对话，保留知识和记忆。"""
        self._session.clear()
        self._short_term.clear()

    def reset(self, system_prompt: str = ""):
        """完全重置助手。"""
        self._session.reset(system_prompt)
        self._short_term.clear()

    @property
    def session_turns(self) -> int:
        return self._session.turn_count

    # ---- 统计信息 ----

    def stats(self) -> dict:
        return {
            "short_term_entries": self._short_term.count(),
            "long_term_entries": self._long_term.count(),
            "profile_keys": self._profile.count(),
            "knowledge_chunks": self._knowledge.count(),
            "registered_skills": self._skills.count(),
            "session_turns": self._session.turn_count,
        }
