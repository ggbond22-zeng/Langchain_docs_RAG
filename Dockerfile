# 构建阶段：装依赖
FROM python:3.13-slim AS builder

WORKDIR /app

# 只复制依赖清单，先装依赖再复制源码，这样改代码不会让依赖层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 运行阶段：只带运行所需内容，镜像更小
FROM python:3.13-slim

WORKDIR /app

# 从构建阶段拷贝已装好的依赖
COPY --from=builder /install /usr/local

# 拷贝源码（.dockerignore 已排除 .env、缓存等）
COPY langchain_docs_mcp.py langchain_docs_ingest.py ./

# 以非 root 用户运行，降低容器逃逸风险
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# 远程托管时用 streamable-http；本地 stdio 模式不需要暴露端口
ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000

# 健康检查：直接探端口。不要用 HTTP GET /mcp —— 缺少 MCP 协议头时该端点会返回
# 4xx，导致健康检查误报失败。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)"

CMD ["python", "langchain_docs_mcp.py"]
