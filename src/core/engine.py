"""
LLM 引擎模块 — DeepSeek (OpenAI 兼容)
封装大模型调用：普通调用、流式输出、重试机制、超时控制。
"""

import asyncio
import logging
from typing import AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import get_config

logger = logging.getLogger(__name__)


def _map_role(role: str) -> type:
    return {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }.get(role, HumanMessage)


class LLMEngine:
    """
    LLM 调用引擎，基于 OpenAI 兼容接口连接 DeepSeek。

    特性:
    - 使用 ChatOpenAI 指定 base_url 连接 DeepSeek
    - 内置指数退避重试
    - 支持同步/异步/流式三种调用模式
    """

    def __init__(self):
        cfg = get_config()
        self._llm = ChatOpenAI(
            model=cfg.llm_model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_api_base,
        )
        self._retry_cfg = cfg.retry_config

    def chat(self, messages: list[dict], **kwargs) -> str:
        langchain_messages = [
            _map_role(m["role"])(content=m["content"]) for m in messages
        ]

        last_error = None
        delay = self._retry_cfg.initial_delay

        for attempt in range(self._retry_cfg.max_retries + 1):
            try:
                response = self._llm.invoke(langchain_messages, **kwargs)
                return response.content
            except Exception as e:
                last_error = e
                if attempt < self._retry_cfg.max_retries:
                    logger.warning(
                        "LLM 调用失败（第 %d/%d 次），%s 秒后重试: %s",
                        attempt + 1,
                        self._retry_cfg.max_retries,
                        delay,
                        e,
                    )
                    import time
                    time.sleep(delay)
                    delay *= self._retry_cfg.backoff_factor

        raise RuntimeError(
            f"LLM 调用失败，已重试 {self._retry_cfg.max_retries} 次") from last_error

    async def chat_stream(self, messages: list[dict],
                          **kwargs) -> AsyncIterator[str]:
        langchain_messages = [
            _map_role(m["role"])(content=m["content"]) for m in messages
        ]

        delay = self._retry_cfg.initial_delay

        for attempt in range(self._retry_cfg.max_retries + 1):
            try:
                async for chunk in self._llm.astream(langchain_messages,
                                                     **kwargs):
                    if chunk.content:
                        yield chunk.content
                return
            except Exception as e:
                if attempt < self._retry_cfg.max_retries:
                    logger.warning(
                        "流式调用失败（第 %d/%d 次），重试中: %s",
                        attempt + 1,
                        self._retry_cfg.max_retries,
                        e,
                    )
                    await asyncio.sleep(delay)
                    delay *= self._retry_cfg.backoff_factor
                else:
                    raise RuntimeError(
                        f"流式调用失败，已重试 {self._retry_cfg.max_retries} 次") from e

    async def chat_with_timeout(self,
                                messages: list[dict],
                                timeout: float = 30.0,
                                **kwargs) -> str:
        """同步 chat() 的超时包装版本，用于需要自动超时控制的场景。

        当前项目（v0.1.0）的 CLI/Web 入口均使用流式 chat_stream()，
        故本方法暂未被调用。预留用于以下场景：

        - 批量后台任务：同时对多篇文档调用 LLM，每篇设独立超时，
          配合 asyncio.gather() 并发处理，防止单个任务卡住所有任务。
        - Skill 执行：某个 Skill 内部需要调用 LLM（如翻译、分类），
          设定合理超时避免 Skill 无限等待导致用户请求挂起。
        - 定时任务 / 自动化脚本：无人值守的异步任务中，
          需要 timeout 作为兜底保护，替代人工 Ctrl+C 中断。
        - API 服务端：在 Web 请求处理中，如果后端对响应时间有 SLA 约束，
          可以用本方法确保 LLM 调用不会超过给定时限。

        实现：
        - asyncio.to_thread() 将同步 chat() 放入线程池，避免阻塞事件循环。
        - asyncio.wait_for() 对线程任务施加超时，超时抛 asyncio.TimeoutError。
        """
        return await asyncio.wait_for(
            asyncio.to_thread(self.chat, messages, **kwargs),
            timeout=timeout,
        )
