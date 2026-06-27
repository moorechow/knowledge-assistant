"""
长期事实记忆：基于 ChromaDB 的向量化存储与语义检索。
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.embeddings import Embeddings

from .base import BaseMemory, MemoryEntry
from src.config import get_config
from src.knowledge.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)


class LongTermMemory(BaseMemory):
    """基于 ChromaDB 的长期记忆，支持语义检索和持久化。"""

    def __init__(self, collection_name: str = "long_term_memory"):
        cfg = get_config()
        self._persist_dir = Path(cfg.chroma_persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir)
        )
        self._embeddings: Embeddings = LocalEmbeddings()
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self._collection_name = collection_name

    def add(self, entry: MemoryEntry):
        """添加一条长期记忆（向量化后存储）。"""
        memory_id = str(uuid.uuid4())
        try:
            self._collection.add(
                ids=[memory_id],
                documents=[entry.content],
                metadatas=[{
                    "source": entry.source,
                    "created_at": entry.created_at,
                    "importance": entry.importance,
                    "extra": json.dumps(entry.metadata, ensure_ascii=False),
                }],
            )
        except Exception:
            logger.exception("写入长期记忆失败")
            raise

    def query(
        self, query_text: str, top_k: int | None = None
    ) -> list[MemoryEntry]:
        """语义检索最相关的 top_k 条长期记忆。"""
        if top_k is None:
            top_k = get_config().memory_config.long_term_top_k
        else:
            top_k = min(top_k, get_config().memory_config.long_term_top_k)
        count = self.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
        )

        def _first_batch(key: str):
            """安全取出第一个查询的结果列表。"""
            batch = results.get(key)
            if batch and batch[0]:
                return batch[0]
            return []

        ids = _first_batch("ids")
        docs = _first_batch("documents")
        metas = _first_batch("metadatas")

        entries: list[MemoryEntry] = []
        for memory_id, doc, meta in zip(ids, docs, metas):
            if not isinstance(meta, dict):
                continue
            try:
                extra = json.loads(meta.get("extra", "{}"))
            except (json.JSONDecodeError, TypeError):
                extra = {}
            entries.append(MemoryEntry(
                content=doc or "",
                source=meta.get("source", "unknown"),
                created_at=meta.get("created_at", ""),
                importance=float(meta.get("importance", 0.5)),
                metadata=extra,
            ))
        return entries

    def delete(self, memory_id: str):
        """删除指定记忆。"""
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            logger.exception("删除长期记忆失败")

    def clear(self):
        """清空集合（删除并重建）。"""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name
        )

    def count(self) -> int:
        return self._collection.count()

    def get_all(self) -> list[dict]:
        """获取所有记忆（仅供调试，大量数据慎用）。"""
        results = self._collection.get()
        items = []
        if results["ids"]:
            for i, mid in enumerate(results["ids"]):
                items.append({
                    "id": mid,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return items
