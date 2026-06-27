"""
文本切片器：将文档切分为适合向量化的文本块。
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config

logger = logging.getLogger(__name__)


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建文本切片器。"""
    cfg = get_config().knowledge_config
    size = chunk_size if chunk_size is not None else cfg.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            " ",
            "",
        ],
        keep_separator=True,
    )


def split_documents(documents: list) -> list:
    """将文档列表切分为文本块。"""
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)
    logger.info("文档切分完成: %d → %d 个块", len(documents), len(chunks))
    return chunks
