"""
本地 Embedding 适配器
使用 sentence-transformers 加载本地模型，无需调用外部 API，免费离线。

默认模型: BAAI/bge-small-zh-v1.5（512维，中文优化）
首次下载约 500MB，通过 hf-mirror.com 镜像加速。
"""

import os
import logging
from typing import List

# 必须在导入 sentence_transformers 之前设置，否则无效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_core.embeddings import Embeddings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """
    基于 sentence-transformers 的本地 Embedding。

    使用示例:
        embeddings = LocalEmbeddings()
        vectors = embeddings.embed_documents(["文本 A", "文本 B"])
        query_vec = embeddings.embed_query("搜索关键词")
    """

    def __init__(self, model_name: str | None = None):
        cfg = get_config()
        self._model_name = model_name if model_name is not None else cfg.embedding_model
        logger.info("加载本地 Embedding 模型: %s", self._model_name)
        logger.info("下载镜像: %s", os.environ.get("HF_ENDPOINT"))
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=True,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(
            [text], normalize_embeddings=True
        )
        return embedding[0].tolist()
