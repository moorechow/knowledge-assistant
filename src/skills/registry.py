"""
Skill 注册中心 + 基类

Skill 是可插拔的功能模块，每个 Skill 提供：
- name: 唯一标识
- description: LLM 选择 Skill 时用的描述
- parameters_schema: 输入参数 schema
- execute: 执行逻辑
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Skill 执行结果。"""
    success: bool
    data: Any = None
    error: str = ""


class BaseSkill(ABC):
    """Skill 抽象基类。"""

    name: str = ""
    description: str = ""
    parameters_schema: dict = field(default_factory=dict)

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        ...

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

    def __repr__(self):
        return f"<Skill: {self.name}>"


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

    def list_skills(self) -> list[BaseSkill]:
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

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

    def count(self) -> int:
        return len(self._skills)
