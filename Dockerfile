# Python
FROM python:3.12-slim

LABEL description="Knowledge Assistant — AI Agent with RAG + Memory"
LABEL version="1.0"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载 Embedding 模型（避免运行时等待）
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')" \
    && echo ">>> Embedding model cached OK"

# 复制项目代码
COPY . .

# 创建运行时目录
RUN mkdir -p /app/data/chroma_db

# 暴露 Streamlit 端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/web.py", "--server.port=8501", "--server.address=0.0.0.0"]
