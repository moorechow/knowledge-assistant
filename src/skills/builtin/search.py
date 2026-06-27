"""
内置搜索 Skill
支持网络搜索和本地知识库搜索。
"""

import logging
from urllib.parse import quote

from ..registry import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class WebSearchSkill(BaseSkill):
    """网络搜索 Skill（使用搜索引擎 URL 生成搜索链接）。"""

    name = "web_search"
    description = (
        "搜索互联网获取信息。当需要查找最新资讯、事实性知识时使用。"
    )

    def __init__(self, search_engine: str = "https://www.bing.com/search?q="):
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
