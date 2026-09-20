# LangChain 文档 MCP 检索服务

把 LangChain 官方文档变成一个 MCP server，让 Cline / Cursor / Claude Desktop 等任意 MCP 客户端
在回答 LangChain 相关问题时**先查官方文档再作答**，从根源上减少模型编造 API。

数据源是 LangChain 官方提供的 [`llms.txt`](https://docs.langchain.com/oss/python/langchain/llms.txt)
（官方专为 LLM 维护的文档索引），因此**不需要自己写爬虫，内容永远是最新的**。

## 特性

- **混合检索**：向量语义检索 + BM25 关键词检索，用 RRF 融合，再用交叉编码器精排
- **中文友好**：中文提问会被自动改写成多条英文查询去检索英文文档，精排阶段用原始中文保留真实意图
- **不编造**：精排分数低于阈值时明确返回「未检索到」，而不是硬凑一段答案
- **BYOK（自带 Key）**：远程托管时从请求头读取用户自己的 API Key，托管方零成本
- **配置全外部化**：所有可调项走环境变量，换模型、换数据源、换 Milvus 地址都不用改代码
- **不依赖 LangChain 框架**：直连 OpenAI 兼容接口，依赖少、启动快

## 工作原理

```
用户提问
   │
   ▼
① 查询改写    中文问题 → 3 条英文查询（LLM，失败则降级用原问题）
   │
   ▼
② 双路召回    向量路：Qwen3-Embedding-4B → Milvus COSINE TopK
   │          关键词路：BM25 内存索引 TopK
   ▼
③ RRF 融合    只按排名融合，规避余弦分与 BM25 分的量纲差异
   │
   ▼
④ 精排        bge-reranker-v2-m3 用「原始中文问题」逐条打分（跨语言匹配）
   │
   ▼
⑤ 阈值过滤    分数 ≥ MIN_SCORE 才保留；全不达标则明确告知未检索到
```

一次提问大约触发 **6 次上游 API 调用**（1 次查询改写 + 4 次向量化 + 1 次精排），
所以远程托管时建议用 BYOK 模式，避免额度被刷。

## 快速开始

### 0. 前置条件

- Python 3.10+
- 一个 **Milvus 2.6.x** 实例（本地或远程）
- 一个 OpenAI 兼容的 API Key（用于向量化 / 查询改写 / 精排，默认按硅基流动配置）

> **重要**：请务必使用 Milvus **2.6.x**。实测 Milvus **3.0.0** 在本地存储模式（`COMMON_STORAGETYPE=local`）
> 下存在路径拼接缺陷——数据写入 `data/json_stats/...`，读取却去找 `data/files/json_stats/...`，
> 导致 collection 永久卡在 `Loading, progress=50%` 且重启无效。降级到 2.6.x 即恢复正常。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，至少填好 OPENAI_API_KEY 和 MILVUS_URI
```

### 3. 入库（构建知识库）

```bash
python langchain_docs_ingest.py
```

该脚本会：拉取官方 `llms.txt` 清单 → 下载 79 篇 Markdown → 按段落分块（代码块整体保留）
→ 向量化 → 写入 Milvus。首次约需几分钟。

### 4. 接入 MCP 客户端

以 Cline 为例，编辑 `cline_mcp_settings.json`：

```json
{
  "mcpServers": {
    "langchain-docs": {
      "command": "python",
      "args": ["/绝对路径/langchain_docs_mcp.py"],
      "disabled": false,
      "autoApprove": ["search_docs", "get_document", "kb_status"]
    }
  }
}
```

Cursor 在 `~/.cursor/mcp.json`、Claude Desktop 在 `claude_desktop_config.json` 中配置，格式相同。

接入后建议先让模型调一次 `kb_status` 确认知识库就绪（首次调用需等待约十几秒预热）。

## 提供的工具

| 工具 | 参数 | 说明 |
|---|---|---|
| `search_docs` | `question`、`top_k`(默认 4) | 混合检索，返回带来源标题、原文链接、相关性分数的片段 |
| `get_document` | `title` | 按标题取回整篇文档正文，实时从官方站点拉取 `.md` |
| `kb_status` | — | 健康检查：chunk 数、Milvus 连接、模型、凭证来源 |

## 远程托管（可选）

把 MCP 服务部署到服务器，让其他人通过 URL 连接：

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 python langchain_docs_mcp.py
```

也可以用 Docker：

```bash
docker build -t langchain-docs-mcp .
docker run -p 8000:8000 \
  -e MILVUS_URI=http://你的milvus地址:19530 \
  -e OPENAI_API_KEY=你的key \
  langchain-docs-mcp
```

客户端连接时带上自己的 Key（**BYOK**），服务端会优先使用它：

```json
{
  "mcpServers": {
    "langchain-docs": {
      "url": "https://你的域名/mcp",
      "headers": {
        "X-Api-Key": "sk-用户自己的key"
      }
    }
  }
}
```

凭证解析优先级：请求头 `X-Api-Key` → 请求头 `Authorization: Bearer xxx` → 环境变量 `OPENAI_API_KEY`。
自定义网关可用请求头 `X-Api-Base` 覆盖 API 地址。

> **公开托管前请注意**：MCP 规范要求面向公众的远程服务使用 OAuth 2.1 + PKCE 做认证。
> 若只是小范围分享，静态 Bearer Token 或 BYOK 就够用；若要完全公开，建议再加一层
> 反向代理做限流（如 Caddy / Nginx），否则任何知道地址的人都能消耗你的资源。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | — | **必填**，上游 API Key（BYOK 模式下可省略） |
| `OPENAI_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址 |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `MILVUS_DB` | `langchain_docs_db` | 数据库名 |
| `MILVUS_COLLECTION` | `langchain_docs_llms_v1` | collection 名 |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B` | 向量模型（换模型需重新入库） |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V4-Flash` | 查询改写模型 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 精排模型 |
| `LLMS_INDEX_URLS` | LangChain Python 分区 | 入库用的文档清单，多个用逗号分隔 |
| `LLMS_INDEX_URL` | LangChain Python 分区 | `get_document` 用的文档索引 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | 分块参数（改动后需重新入库） |
| `UPSERT_BATCH` | `200` | Milvus 写入批大小，**不要调太大** |
| `QUERY_VARIANTS` | `3` | 改写出的英文查询条数 |
| `RECALL_K` | `8` | 每路召回条数 |
| `RRF_K` | `60` | RRF 平滑常数 |
| `RERANK_CANDIDATES` | `12` | 送入精排的候选数上限 |
| `MIN_SCORE` | `0.3` | 精排相关性阈值 |
| `MCP_TRANSPORT` | `stdio` | `stdio` / `streamable-http` / `sse` |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8000` | HTTP 模式监听地址 |

## 常见问题

**Q：collection 卡在 `Loading, progress=50%`？**
Milvus 版本问题，见上文「前置条件」。请确认服务端是 2.6.x，不要用 3.0.0。

**Q：入库时报 `received message larger than max (64MB)`？**
一次性写入的向量太多。调小 `UPSERT_BATCH`（默认 200 已足够安全）。

**Q：检索一直卡住不返回？**
历史版本曾在后台线程里惰性 `import pymilvus`，会与 asyncio 事件循环争用导入锁而死锁。
当前代码已把所有第三方依赖放到模块顶层导入，请勿改回函数内导入。

**Q：回答总是「未检索到」？**
先调 `kb_status` 确认 chunk 数不为 0；若知识库正常，可能是 `MIN_SCORE` 偏严，可适当调低。

**Q：想换更全的知识库？**
改 `LLMS_INDEX_URLS` 加入更多分区再重新入库。例如 Python 全量：
`https://docs.langchain.com/oss/python/llms.txt`（367 页）。

## 数据来源与许可

- 文档内容通过官方 `llms.txt` 索引实时获取，不在本仓库中再分发任何文档正文。
- 本仓库代码以 [MIT 许可证](LICENSE) 发布。
- LangChain 是 [MIT 许可](https://github.com/langchain-ai/langchain) 的开源项目；文档内容版权归 LangChain, Inc. 所有。

## 目录结构

```
.
├── langchain_docs_mcp.py       # MCP 服务：检索 + 工具暴露
├── langchain_docs_ingest.py    # 入库：llms.txt → 下载 → 分块 → 向量化 → Milvus
├── requirements.txt
├── .env.example
├── Dockerfile
├── LICENSE
└── README.md
```
