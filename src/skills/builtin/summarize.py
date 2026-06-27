"""
内置摘要 Skill
"""

import logging

from ..registry import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


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

    def execute(self, text: str = "", max_length: int = 200, **kwargs) -> SkillResult:
        if not text:
            return SkillResult(success=False, error="待摘要文本不能为空")

        if self._llm:
            messages = [
                {"role": "system", "content": f"请对以下内容进行摘要，控制在 {max_length} 字以内。"},
                {"role": "user", "content": text},
            ]
            summary = self._llm.chat(messages)
            return SkillResult(success=True, data={"summary": summary})
        else:
            # 无 LLM 时简单截断
            summary = text[:max_length] + ("..." if len(text) > max_length else "")
            return SkillResult(success=True, data={"summary": summary, "truncated": len(text) > max_length})
