"""
文档加载器：支持 PDF、Markdown、TXT、Word 文件。
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)

logger = logging.getLogger(__name__)

# 支持的文件扩展名到加载器的映射
LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}


def load_document(file_path: str) -> list:
    """
    加载单个文档。
    Returns:
        LangChain Document 列表。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    loader_cls = LOADER_REGISTRY.get(ext)
    if not loader_cls:
        raise ValueError(f"不支持的文件类型: {ext}（支持: {list(LOADER_REGISTRY.keys())}）")

    logger.info("正在加载文档: %s", path.name)
    loader = loader_cls(str(path))
    documents = loader.load()

    # 为每个文档添加来源元数据
    for doc in documents:
        if "source" not in doc.metadata:
            doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)
        doc.metadata["file_type"] = ext

    logger.info("文档加载完成: %d 页/段", len(documents))
    return documents


def load_documents_from_directory(directory: str, recursive: bool = False) -> list:
    """
    加载目录下所有支持的文档。
    Args:
        directory: 目录路径。
        recursive: 是否递归子目录。
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")

    all_docs = []
    pattern = "**/*" if recursive else "*"
    for file_path in dir_path.glob(pattern):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in LOADER_REGISTRY:
            continue
        try:
            docs = load_document(str(file_path))
            all_docs.extend(docs)
        except (OSError, ValueError, RuntimeError):
            logger.exception("加载文件失败: %s", file_path.name)

    source_count = len({
        d.metadata.get("source", "unknown") for d in all_docs
    })
    logger.info(
        "目录加载完成: %d 个文档, %d 个分段", source_count, len(all_docs)
    )
    return all_docs
