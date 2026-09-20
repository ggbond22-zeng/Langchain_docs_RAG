"""
LangChain 文档 MCP 检索服务
============================
把 LangChain 官方文档做成一个可开源分发的 MCP server：任何人 clone 后
填上自己的 API Key 就能用，无需改一行代码。

与同目录 my_mcp_rag_server.py 的区别（那份是课程练习版，本份面向开源分发）：
    1. 配置全部外部化：所有可调项走环境变量，.env 从「当前目录 / 脚本目录」查找，
       不再写死「上溯三级」这种只在本机成立的路径
    2. 数据源改为官方 llms.txt：不再依赖本地爬取的 langchain_docs/ 与 pages_index.json，
       get_document 直接按官方链接拉取最新 .md，天然免维护
    3. 支持 BYOK（Bring Your Own Key）：远程部署时从请求头取用户自己的 Key，
       本地 stdio 模式回退到环境变量。托管方零成本，用户自担用量
    4. 不依赖 LangChain 框架：embedding / LLM / rerank 直连 OpenAI 兼容接口

检索流水线（五步）：
    ① 查询改写：中文问题 → 3 条英文查询（LLM JSON 输出，失败降级用原问题）
    ② 双路召回：向量检索（Milvus COSINE）+ 关键词检索（BM25 内存索引）
    ③ RRF 融合：只看排名不看分数，规避两种分数的量纲差异
    ④ 精排：bge-reranker-v2-m3 用原始中文问题做跨语言打分
    ⑤ 阈值过滤：分数 < MIN_SCORE 视为不相关，全不达标时明确告知"未检索到"，不编造

运行方式：
    # stdio 模式（本地，由 Cline / Cursor / Claude Desktop 拉起）
    python langchain_docs_mcp.py

    # streamable-http 模式（远程托管）
    MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 python langchain_docs_mcp.py

调试方式：
    npx @modelcontextprotocol/inspector python langchain_docs_mcp.py

暴露的工具：
    search_docs(question, top_k)   检索文档片段（带来源标题 / 链接 / 相关性分数）
    get_document(title)            按标题取回整篇文档正文（实时拉官方 .md）
    kb_status()                    知识库健康检查
"""

# ---------- 第 0 部分：日志与协议通道隔离（必须最先执行） ----------
import os
import sys
import builtins
import logging

# Windows 下默认编码可能是 GBK，中文日志会乱码/报错，统一成 UTF-8
# 注意：只改编码，不改 sys.stdout 指向的控制台对象
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# stdio 模式下 stdout 只能跑 JSON-RPC 协议消息，任何 print 混进去都会让客户端解析失败。
# 正确做法是保留 sys.stdout 原对象（SDK 运行时读它的 buffer 发消息），只把 print 的
# 默认输出目标改成 stderr —— 千万不要写 sys.stdout = sys.stderr。
_original_print = builtins.print


def _print_stderr(*args, **kwargs):
    """包装 print：调用方没指定 file 时默认写 stderr，保护 stdout 协议流。"""
    kwargs.setdefault('file', sys.stderr)
    _original_print(*args, **kwargs)


builtins.print = _print_stderr

# 第三方库的日志统一压到 stderr，避免污染协议流
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(levelname)s %(name)s: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('pymilvus').setLevel(logging.WARNING)

# 关闭 LangSmith 追踪：本项目不用 LangChain，但若环境变量开着会带来额外网络开销
os.environ.setdefault('LANGSMITH_TRACING', 'false')
os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')

# ---------- 第 1 部分：依赖导入与配置 ----------
import json
import re
import threading
from pathlib import Path

# 第三方依赖一律在模块加载阶段（主线程）导入，不要在工具函数/后台线程里做惰性导入。
# 原因：pymilvus、numpy 这类含 C 扩展的库若首次导入发生在后台线程，会与主线程的
# asyncio 事件循环争用导入锁，实测卡死在 importlib 的 create_module 且永不返回
# （此时 MCP 握手成功、但任何检索调用都会一直挂住）。放主线程导入可彻底规避。
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

# .env 查找顺序：当前工作目录 → 脚本所在目录。用「就近查找」而非「上溯固定层数」，
# 这样无论别人把项目放在哪、以什么方式启动，都能找到配置。
for _candidate in (Path.cwd() / '.env', Path(__file__).parent / '.env'):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)   # 已存在的真实环境变量优先，便于容器注入
        break

# ---- 连接与知识库（必须与入库脚本 langchain_docs_ingest.py 保持一致）----
MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
DB_NAME = os.getenv('MILVUS_DB', 'langchain_docs_db')
COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'langchain_docs_llms_v1')

# ---- 模型 ----
EMBED_MODEL = os.getenv('EMBED_MODEL', 'Qwen/Qwen3-Embedding-4B')      # 输出 2560 维
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-ai/DeepSeek-V4-Flash')    # 仅用于查询改写
RERANK_MODEL = os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-v2-m3')    # 中英跨语言交叉编码器

# ---- 上游 API（BYOK 回退值）----
API_KEY_ENV = os.getenv('OPENAI_API_KEY', '')
API_BASE_ENV = os.getenv('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1').rstrip('/')

# ---- 官方文档索引（get_document 用；按官方约定，每个分区都有 llms.txt）----
LLMS_INDEX_URL = os.getenv(
    'LLMS_INDEX_URL',
    'https://docs.langchain.com/oss/python/langchain/llms.txt',
)

# ---- 检索参数 ----
QUERY_VARIANTS = int(os.getenv('QUERY_VARIANTS', '3'))        # 改写出的英文查询条数
RECALL_K = int(os.getenv('RECALL_K', '8'))                    # 每路召回的条数
RRF_K = int(os.getenv('RRF_K', '60'))                         # RRF 平滑常数（论文推荐值）
RERANK_CANDIDATES = int(os.getenv('RERANK_CANDIDATES', '12'))  # 送入 rerank 的候选数上限
MIN_SCORE = float(os.getenv('MIN_SCORE', '0.3'))              # rerank 相关性阈值
MAX_QUERY_ROWS = int(os.getenv('MAX_QUERY_ROWS', '16384'))    # Milvus query 结果窗口上限

# ---- 传输方式 ----
MCP_TRANSPORT = os.getenv('MCP_TRANSPORT', 'stdio')           # stdio | streamable-http | sse
MCP_HOST = os.getenv('MCP_HOST', '127.0.0.1')
MCP_PORT = int(os.getenv('MCP_PORT', '8000'))

HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '30'))         # 调用上游 API 的超时秒数

# ---------- 第 2 部分：凭证解析（BYOK） ----------
# 设计：远程部署时，用户在 MCP 客户端的配置里带上自己的 Key，服务端从请求头读取；
# 本地 stdio 模式没有 HTTP 请求，回退到环境变量。托管方因此不承担任何 API 费用。
#
# 为什么用 threading.local 而不是层层传参：FastMCP 会把同步工具函数整体丢进
# 一个工作线程执行（anyio.to_thread.run_sync），所以「进入工具函数时设置、函数内读取」
# 在同一线程内始终一致，既能保证正确性，又不必给每个内部函数都加一个 creds 参数。
_cred = threading.local()
_clients = {}
_clients_lock = threading.Lock()
_MAX_CACHED_CLIENTS = 16   # 上限，防止公网 BYOK 场景下 Key 种类过多导致内存无限增长


def _set_credentials(api_key: str, base_url: str) -> None:
    """在工具函数入口设置本次调用要用的凭证。"""
    _cred.api_key = api_key
    _cred.base_url = base_url


def _current_credentials() -> tuple:
    """读取本次调用要用的 (api_key, base_url)；未设置时回退到环境变量。"""
    return (
        getattr(_cred, 'api_key', None) or API_KEY_ENV,
        getattr(_cred, 'base_url', None) or API_BASE_ENV,
    )


def _resolve_credentials() -> tuple:
    """
    从当前请求解析凭证，实现 BYOK。优先级：
        1. 请求头 X-Api-Key
        2. 请求头 Authorization: Bearer <key>
        3. 环境变量 OPENAI_API_KEY（本地 stdio 模式）
    自定义 API 网关可用请求头 X-Api-Base 覆盖 base_url。
    """
    request = None
    try:
        # stdio 传输下 request 为 None；HTTP 传输下是 Starlette Request
        request = mcp.get_context().request_context.request
    except Exception:
        request = None

    if request is not None:
        headers = getattr(request, 'headers', None) or {}
        key = (headers.get('x-api-key') or '').strip()
        if not key:
            auth = (headers.get('authorization') or '').strip()
            if auth.lower().startswith('bearer '):
                key = auth[7:].strip()
        if key:
            base = (headers.get('x-api-base') or '').strip() or API_BASE_ENV
            return key, base.rstrip('/')

    return API_KEY_ENV, API_BASE_ENV


def _get_client(api_key: str, base_url: str) -> OpenAI:
    """按 (key, base) 复用 OpenAI 客户端，避免每次请求都重建连接池。"""
    cache_key = (api_key, base_url)
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is None:
            if len(_clients) >= _MAX_CACHED_CLIENTS:
                _clients.clear()   # 简单策略：满了整体清空，够用且不会无限增长
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=HTTP_TIMEOUT, max_retries=2)
            _clients[cache_key] = client
        return client


# ---------- 第 3 部分：知识库句柄（Milvus 连接 + BM25 索引） ----------
_STATE_LOCK = threading.Lock()
_STATE = None          # {'client', 'rows', 'bm25'}，None 表示尚未初始化
_STATE_ERROR = None    # 初始化失败信息，供 kb_status 展示


def _tokenize(text: str) -> list:
    """
    BM25 分词器：提取英文单词 / 数字 / 下划线，全部小写。
    例如 "Create an agent with create_agent()" → ['create', 'an', 'agent', 'with', 'create_agent']
    纯中文串会被整段丢弃（返回空列表），因此中文查询不送 BM25，只走向量路。
    """
    return re.findall(r'[a-z0-9_]+', text.lower())


def _embed_texts(texts: list) -> list:
    """批量向量化，返回与输入等长的向量列表。"""
    api_key, base_url = _current_credentials()
    resp = _get_client(api_key, base_url).embeddings.create(model=EMBED_MODEL, input=texts)
    # 按 index 排序，防止服务端乱序返回导致向量与文本错位
    items = sorted(resp.data, key=lambda d: d.index)
    return [item.embedding for item in items]


def _init_state() -> dict:
    """
    初始化知识库句柄：连 Milvus、拉全量 chunk、建 BM25 索引。
    约 7000 条量级，耗时十几秒，放后台线程预热。
    """
    client = MilvusClient(MILVUS_URI)
    client.use_database(DB_NAME)

    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter='id >= 0',
        output_fields=['text', 'source', 'url'],
        limit=MAX_QUERY_ROWS,
    )
    rows.sort(key=lambda r: r['id'])   # 查询返回顺序不保证，按 id 排好再建索引

    # BM25 索引：喂入每条 chunk 的分词结果，索引与 rows 顺序一一对应
    bm25 = BM25Okapi([_tokenize(r['text']) for r in rows])

    print(f'[rag] 知识库就绪：{len(rows)} 条 chunk，BM25 索引已建好')
    return {'client': client, 'rows': rows, 'bm25': bm25}


def _get_state() -> dict:
    """
    获取知识库句柄（线程安全单例）。
    预热线程与工具调用可能同时触发初始化，用锁保证只建一次；
    失败时缓存错误信息，避免每次调用都重连重试。
    """
    global _STATE, _STATE_ERROR
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            try:
                _STATE = _init_state()
                _STATE_ERROR = None
            except Exception as exc:
                _STATE_ERROR = f'{type(exc).__name__}: {exc}'
                raise
    return _STATE


# ---------- 第 4 部分：检索流水线 ----------
def _gen_queries(question: str) -> list:
    """
    查询改写：把中文问题改写成 N 条措辞各异的英文检索查询。
    改英文的原因：知识库是英文文档，且 BM25 是关键词匹配、不认中文；
    多条不同措辞并行召回取并集，能显著扩大召回面（RAG-Fusion 思想）。
    失败时降级为 [原问题]，由跨语言向量检索兜底。
    """
    api_key, base_url = _current_credentials()
    prompt = (
        '你是检索查询改写器。知识库是 LangChain 官方英文文档。\n'
        f'把下面的用户问题改写成 {QUERY_VARIANTS} 条措辞各异的英文搜索查询：\n'
        '- 第一条：紧扣原意直译\n'
        '- 第二条：补充同义的专业术语（如 agent / tool / middleware）\n'
        '- 第三条：面向 API 或函数名（如 create_agent、init_chat_model）\n'
        '只输出 JSON，格式为 {"queries": ["...", "...", "..."]}，不要输出其他内容。\n'
        f'用户问题：{question}'
    )
    try:
        resp = _get_client(api_key, base_url).chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            response_format={'type': 'json_object'},
        )
        content = resp.choices[0].message.content or ''
        # 容错：模型可能把 JSON 包在 ```json 代码块里
        match = re.search(r'\{.*\}', content, re.S)
        data = json.loads(match.group(0) if match else content)
        queries = [q for q in data.get('queries', []) if isinstance(q, str) and q.strip()]
        return queries[:QUERY_VARIANTS] or [question]
    except Exception as exc:
        print(f'[rag] 查询改写失败（降级为原问题）：{exc}')
        return [question]


def _vector_search(query: str, k: int) -> list:
    """向量检索：查询向量化后在 Milvus 取 COSINE 相似度最高的 k 条，返回 id 列表。"""
    state = _get_state()
    vector = _embed_texts([query])[0]
    result = state['client'].search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        limit=k,
        output_fields=['id'],
    )
    # 返回结构：[[{id, distance, entity}, ...]]，外层是批次，内层已按相似度降序
    return [hit['id'] for hit in result[0]]


def _bm25_search(query: str, k: int) -> list:
    """
    关键词检索：BM25 打分后取前 k 条 id。
    纯中文查询没有英文 token，直接返回空（此时靠向量路和其他英文改写查询补位）。
    """
    tokens = _tokenize(query)
    if not tokens:
        return []
    state = _get_state()
    scores = state['bm25'].get_scores(tokens)   # 与 rows 顺序一一对应
    # 只取分数 > 0 的，避免召回一堆零分词条
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [state['rows'][i]['id'] for i in top_idx if scores[i] > 0]


def _rrf_fuse(id_lists: list) -> list:
    """
    RRF（Reciprocal Rank Fusion）融合多路召回结果。
    每个文档的融合分 = Σ 1/(k + 排名)，只依赖排名，天然规避余弦分与 BM25 分量纲不同的问题。
    数值举例（k=60，两路）：向量路 [10,20,30]、BM25 路 [20,30,99]
        id=20 → 1/62 + 1/61 = 0.0325（两路都命中，排最前）
        id=30 → 1/63 + 1/62 = 0.0320
        id=10 → 1/61        = 0.0164
    返回融合分降序的 id 列表。
    """
    scores = {}
    for id_list in id_lists:
        for rank, doc_id in enumerate(id_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _rerank(question: str, candidates: list, top_n: int) -> list:
    """
    精排：交叉编码器对 (query, 文档) 逐条精细打分。
    传入的是用户的原始中文问题 —— 精排阶段的偏差会直接决定最终顺序，
    用原始问题能保留真实意图（bge-reranker-v2-m3 支持中英跨语言匹配）。
    调用失败时降级：沿用 RRF 融合顺序，分数置 None。
    """
    if not candidates:
        return []

    api_key, base_url = _current_credentials()
    try:
        resp = requests.post(
            f'{base_url}/rerank',
            headers={'Authorization': f'Bearer {api_key}'},
            json={
                'model': RERANK_MODEL,
                'query': question,
                'documents': [c['text'] for c in candidates],
                'top_n': top_n,
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()['results']
    except Exception as exc:
        print(f'[rag] Rerank 调用失败（降级沿用融合顺序）：{exc}')
        fallback = [dict(c) for c in candidates[:top_n]]
        for item in fallback:
            item['rerank_score'] = None
        return fallback

    ranked = []
    for item in results:
        # results 里的 index 是候选列表下标，用它回填分数
        doc = dict(candidates[item['index']])
        doc['rerank_score'] = item['relevance_score']
        ranked.append(doc)
    return ranked


def retrieve(question: str, top_k: int = 4) -> dict:
    """
    完整检索流水线：改写 → 双路召回 → RRF 融合 → 精排 → 阈值过滤。

    Args:
        question: 用户问题（中文或英文均可）
        top_k: 最终保留的片段数

    Returns:
        {'question', 'queries', 'docs': [带 rerank_score 的 chunk], 'filtered': 过滤掉的不相关条数}
    """
    # ① 查询改写（中文问题转英文），并追加原问题兜底
    variants = _gen_queries(question)
    all_queries = variants + [question]

    # ② 双路召回：每条查询各跑一次向量 + 一次 BM25
    id_lists = []
    for query in all_queries:
        id_lists.append(_vector_search(query, RECALL_K))
        id_lists.append(_bm25_search(query, RECALL_K))

    # ③ RRF 融合，截断后再送精排（控制 rerank 成本）
    merged_ids = _rrf_fuse(id_lists)[:RERANK_CANDIDATES]

    state = _get_state()
    row_map = {r['id']: r for r in state['rows']}
    candidates = [row_map[i] for i in merged_ids if i in row_map]

    # ④ 精排 ⑤ 阈值过滤：分数达标才保留
    ranked = _rerank(question, candidates, top_n=max(top_k, RERANK_CANDIDATES))
    good = [d for d in ranked if d.get('rerank_score') is None or d['rerank_score'] >= MIN_SCORE]

    return {
        'question': question,
        'queries': variants,
        'docs': good[:top_k],
        'filtered': len(ranked) - len(good),
    }


def _format_docs(result: dict) -> str:
    """把检索结果格式化成带来源的文本块，便于 LLM 直接引用出处。"""
    docs = result['docs']
    if not docs:
        return (
            f'未在 LangChain 官方文档知识库中检索到与「{result["question"]}」相关的内容。\n'
            '（召回了候选但相关性均低于阈值，或知识库未就绪）'
        )

    lines = [
        f'问题：{result["question"]}',
        f'英文改写查询：{"; ".join(result["queries"])}',
        f'命中片段：{len(docs)} 条' + (f'（另有 {result["filtered"]} 条低相关已过滤）' if result['filtered'] else ''),
        '',
    ]
    for i, doc in enumerate(docs, start=1):
        score = '未知' if doc.get('rerank_score') is None else f'{doc["rerank_score"]:.3f}'
        lines.append(f'[片段 {i}] 来源：{doc["source"]} ｜ 相关性：{score}')
        lines.append(f'原文链接：{doc["url"]}')
        lines.append(f'内容：{doc["text"]}')
        lines.append('')
    return '\n'.join(lines)


# ---------- 第 5 部分：官方文档索引与全文获取 ----------
# 官方为每个文档分区提供了 llms.txt（Markdown 链接清单），例如：
#     - [Agents](https://docs.langchain.com/oss/python/langchain/agents.md)
# 解析它即可拿到「标题 → 原始 .md 链接」的映射，用于 get_document。
_index_lock = threading.Lock()
_index_cache = None    # [(title, url), ...]


def _load_doc_index() -> list:
    """
    拉取并解析 llms.txt，返回 [(title, url), ...]，进程内缓存一次。
    失败时返回空列表（不影响 search_docs，只影响 get_document）。
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    with _index_lock:
        if _index_cache is not None:
            return _index_cache
        try:
            resp = requests.get(LLMS_INDEX_URL, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            # 只匹配形如 "- [标题](http...)" 的行，忽略说明性文字
            pairs = re.findall(r'^\s*-\s*\[([^\]]+)\]\((https?://[^)]+)\)', resp.text, re.M)
            _index_cache = pairs
            print(f'[rag] 已加载官方文档索引：{len(pairs)} 篇（{LLMS_INDEX_URL}）')
        except Exception as exc:
            print(f'[rag] 加载文档索引失败：{exc}')
            _index_cache = []
    return _index_cache


def _find_pages(title: str) -> list:
    """按标题定位文档：先精确匹配（不区分大小写），再退化为关键字包含匹配。"""
    key = title.strip().lower()
    pages = _load_doc_index()
    exact = [p for p in pages if key == p[0].lower() or key == p[1].rsplit('/', 1)[-1].lower()]
    if exact:
        return exact
    return [p for p in pages if key in p[0].lower() or key in p[1].lower()]


# ---------- 第 6 部分：MCP 服务 ----------
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('langchain-docs-rag', host=MCP_HOST, port=MCP_PORT)


@mcp.tool()
def search_docs(question: str, top_k: int = 4) -> str:
    """
    在 LangChain 官方文档知识库中做混合检索（语义 + 关键词 + 重排），返回带来源和链接的文档片段。
    凡是涉及 LangChain / LangGraph 的概念、API、用法问题，都应先调用本工具获取依据，再作答。

    Args:
        question: 检索问题，中文或英文均可（中文会被自动改写成英文查询提高召回）
        top_k: 返回的文档片段数量，默认 4，建议 3~6
    """
    _set_credentials(*_resolve_credentials())
    top_k = max(1, min(int(top_k), 10))    # 夹紧到合理区间，防止客户端传入异常值
    return _format_docs(retrieve(question, top_k=top_k))


@mcp.tool()
def get_document(title: str) -> str:
    """
    按文档标题取回整篇原始文档正文（Markdown），内容实时来自 LangChain 官方站点。
    当检索片段不足以回答问题、需要了解完整上下文时使用。标题可传模糊词（如 "middleware"）。

    Args:
        title: 文档标题或文件名关键词，例如 "Custom middleware"、"agents.md"、"retrieval"
    """
    pages = _find_pages(title)
    if not pages:
        samples = '、'.join(p[0] for p in _load_doc_index()[:10])
        return (
            f'未找到标题包含「{title}」的文档。\n'
            f'文档索引：{LLMS_INDEX_URL}\n'
            + (f'可供参考的部分标题：{samples} ...' if samples else '（文档索引加载失败，请检查网络）')
        )

    # 命中多篇时只返回最靠前的一篇，并列出其他候选，避免一次塞爆上下文
    page_title, page_url = pages[0]
    try:
        resp = requests.get(page_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        return f'拉取文档失败：{page_url}\n错误：{type(exc).__name__}: {exc}'

    header = f'# {page_title}\n原文链接：{page_url}\n'
    others = [p[0] for p in pages[1:6]]
    if others:
        header += f'（另有标题相近的文档：{"、".join(others)}）\n'
    return header + '\n' + body


@mcp.tool()
def kb_status() -> str:
    """
    检查知识库健康状态：Milvus 连接、chunk 数量、BM25 索引、所用模型与凭证来源。
    检索异常或结果为空时，先调用本工具确认知识库是否就绪。
    """
    # 同样要从当前请求解析凭证：否则会读到同线程上一次调用残留的 Key，导致来源显示错误
    _set_credentials(*_resolve_credentials())

    if _STATE is not None:
        state_line = f'状态：就绪\nchunk 数量：{len(_STATE["rows"])}'
    elif _STATE_ERROR:
        state_line = f'状态：初始化失败\n错误：{_STATE_ERROR}\n请确认 Milvus 已启动且 collection 处于 Loaded 状态。'
    else:
        state_line = '状态：尚未初始化（首次调用检索工具时会自动加载，约需十几秒）'

    # 说明当前凭证来自请求头（BYOK）还是环境变量，便于排查 401
    key, _ = _current_credentials()
    if key and key == API_KEY_ENV:
        cred_line = '凭证来源：环境变量 OPENAI_API_KEY'
    elif key:
        cred_line = '凭证来源：请求头（BYOK）'
    else:
        cred_line = '凭证来源：未配置（请设置 OPENAI_API_KEY，或在请求头携带 X-Api-Key）'

    return '\n'.join([
        state_line,
        f'Milvus：{MILVUS_URI} / {DB_NAME}.{COLLECTION_NAME}',
        f'向量模型：{EMBED_MODEL}',
        f'改写模型：{LLM_MODEL}',
        f'重排模型：{RERANK_MODEL}',
        f'相关性阈值：rerank ≥ {MIN_SCORE}',
        cred_line,
    ])


# ---------- 第 7 部分：启动 ----------
def _warmup():
    """后台预热：提前连 Milvus、拉全量 chunk、建 BM25 索引，让首次检索不必等待。"""
    try:
        _get_state()
    except Exception as exc:
        # 预热失败不阻断启动，真实错误会在调用工具时通过 kb_status / 异常暴露
        print(f'[rag] 预热失败（不影响启动）：{exc}')


if __name__ == '__main__':
    # 后台线程预热：MCP 握手不能被十几秒的索引构建拖慢（客户端通常有 30s 连接超时）
    threading.Thread(target=_warmup, daemon=True).start()

    if not API_KEY_ENV:
        print('[rag] 提示：未检测到 OPENAI_API_KEY。本地模式请在 .env 中配置，'
              '远程模式可由客户端在请求头携带 X-Api-Key（BYOK）。')

    print(f'[rag] 启动 MCP 服务，传输方式：{MCP_TRANSPORT}')
    if MCP_TRANSPORT == 'stdio':
        mcp.run()                                    # 进程 stdin/stdout 即与客户端的协议管道
    else:
        mcp.run(transport=MCP_TRANSPORT)             # streamable-http / sse，监听 MCP_HOST:MCP_PORT
