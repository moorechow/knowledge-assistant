"""
向量检索引擎：基于 ChromaDB 的向量存储与语义检索。
使用本地 sentence-transformers 嵌入模型（免费、离线）。
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma

from src.config import get_config
from src.knowledge.embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器，封装 ChromaDB 的存储和查询。"""

    def __init__(self, collection_name: str = "knowledge_base"):
        cfg = get_config()
        persist_dir = Path(cfg.chroma_persist_dir) / collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings = LocalEmbeddings()
        self._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
        )
        self._top_k = cfg.knowledge_config.retrieval_top_k

    def add_documents(self, documents: list):
        logger.info("正在向量化 %d 个文档块...", len(documents))
        self._vector_store.add_documents(documents)
        logger.info("向量化完成，当前库总量: %d", self.count())

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[str, dict, float]]:
        if top_k is None:
            k = self._top_k
        else:
            k = min(top_k, self._top_k)
        count = self.count()
        if count == 0:
            return []

        results = self._vector_store.similarity_search_with_score(
            query, k=min(k, count)
        )
        return [
            (doc.page_content, doc.metadata, score)
            for doc, score in results
        ]

    def search_as_context(
        self, query: str, top_k: int | None = None
    ) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""

        lines = ["[相关知识库内容]"]
        for i, (content, meta, score) in enumerate(results, start=1):
            source = meta.get("source", "未知来源")
            lines.append(
                f"\n--- 片段 {i}（来源: {source}，相关度: {score:.3f}）---"
            )
            lines.append(content)

        return "\n".join(lines)

    def count(self) -> int:
        return len(self._vector_store.get()["ids"])

    def clear(self):
        ids = self._vector_store.get()["ids"]
        if ids:
            self._vector_store.delete(ids=ids)
        logger.info("向量库已清空")
