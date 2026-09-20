"""
LangChain 文档入库脚本
======================
从 LangChain 官方 llms.txt 拉取文档清单 → 下载每页 Markdown → 分块 →
向量化 → 写入 Milvus，供 langchain_docs_mcp.py 检索。

相比自己写爬虫，用官方 llms.txt 有三个好处：
    1. 官方维护的页面清单，永远是最新的，不用跟着站点改版改选择器
    2. 直接拿到每页的 .md 原始链接，跳过 HTML 解析，正文干净
    3. 不涉及文档内容的抓取与再分发争议

用法：
    # 1. 确认 Milvus 已就绪（默认 http://localhost:19530）
    # 2. 配置 .env 里的 OPENAI_API_KEY（向量化要调用 API）
    # 3. 执行入库
    python langchain_docs_ingest.py

    # 想连别的文档分区（逗号分隔多个 llms.txt）：
    LLMS_INDEX_URLS="https://docs.langchain.com/oss/python/langchain/llms.txt,https://docs.langchain.com/oss/python/langgraph/llms.txt" \
        python langchain_docs_ingest.py

可调参数（全部走环境变量，见 .env.example）：
    MILVUS_URI / MILVUS_DB / MILVUS_COLLECTION / EMBED_MODEL
    CHUNK_SIZE / CHUNK_OVERLAP / EMBED_BATCH / UPSERT_BATCH
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import DataType, MilvusClient

# .env 查找顺序：当前工作目录 → 脚本所在目录（与 MCP 服务保持一致的就近查找策略）
for _candidate in (Path.cwd() / '.env', Path(__file__).parent / '.env'):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

# ---- 配置 ----
MILVUS_URI = os.getenv('MILVUS_URI', 'http://localhost:19530')
DB_NAME = os.getenv('MILVUS_DB', 'langchain_docs_db')
COLLECTION_NAME = os.getenv('MILVUS_COLLECTION', 'langchain_docs_llms_v1')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'Qwen/Qwen3-Embedding-4B')
API_KEY = os.getenv('OPENAI_API_KEY', '')
API_BASE = os.getenv('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1').rstrip('/')

# 文档清单来源：默认只取 LangChain（Python）分区，共 79 页，规模适中
LLMS_INDEX_URLS = [
    u.strip() for u in os.getenv(
        'LLMS_INDEX_URLS',
        'https://docs.langchain.com/oss/python/langchain/llms.txt',
    ).split(',') if u.strip()
]

# 分块参数：与课程约定一致（按段落切、800 字符、重叠 100）
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))

# 批量参数：
#   EMBED_BATCH 受上游 API 单次请求条数限制
#   UPSERT_BATCH 必须分批——一次性写入上千条向量会触发 gRPC
#   "received message larger than max (64MB)" 错误（实测约 7000 条必炸）
EMBED_BATCH = int(os.getenv('EMBED_BATCH', '32'))
UPSERT_BATCH = int(os.getenv('UPSERT_BATCH', '200'))

DOWNLOAD_WORKERS = int(os.getenv('DOWNLOAD_WORKERS', '8'))
HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '30'))


# ---------- 第 1 部分：文档获取 ----------
def fetch_page_list() -> list:
    """拉取并合并所有 llms.txt，返回去重后的 [(title, url), ...]。"""
    seen = set()
    pages = []
    for index_url in LLMS_INDEX_URLS:
        print(f'[ingest] 读取文档清单：{index_url}')
        resp = requests.get(index_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        # 只匹配形如 "- [标题](http...)" 的行
        pairs = re.findall(r'^\s*-\s*\[([^\]]+)\]\((https?://[^)]+)\)', resp.text, re.M)
        for title, url in pairs:
            if url not in seen:
                seen.add(url)
                pages.append((title, url))
        print(f'[ingest]   本分区 {len(pairs)} 篇')
    print(f'[ingest] 去重后共 {len(pages)} 篇待下载')
    return pages


def download_pages(pages: list) -> list:
    """
    并发下载所有页面，返回 [(title, url, markdown), ...]。
    单页失败只跳过该页并告警，不中断整体流程。
    """
    results = []

    def _one(item):
        title, url = item
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return title, url, resp.text
        except Exception as exc:
            print(f'[ingest]   跳过（下载失败）{url} → {type(exc).__name__}: {exc}')
            return None

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for i, out in enumerate(pool.map(_one, pages), start=1):
            if out is not None:
                results.append(out)
            if i % 20 == 0 or i == len(pages):
                print(f'[ingest]   已下载 {i}/{len(pages)}')

    print(f'[ingest] 成功下载 {len(results)}/{len(pages)} 篇')
    return results


# ---------- 第 2 部分：分块 ----------
def _atomic_blocks(text: str) -> list:
    """
    把正文切成「原子块」：``` 围栏代码块整体保留，其余按空行分段。

    为什么要单独处理代码块：LangChain 文档大量内容是代码示例，
    若按空行无脑切分，一个示例会被切成两三段，检索到半截代码毫无用处。
    数值举例：一段 40 行的示例，若不保护会被切成 3 块；保护后是完整的 1 块。
    """
    blocks = []
    para = []
    code = None      # 非 None 表示正处于代码块中，内容累积在这里

    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            if code is None:
                if para:                          # 代码块开始前，先冲掉已累积的正文
                    blocks.append('\n'.join(para))
                    para = []
                code = [line]
            else:
                code.append(line)                 # 代码块结束
                blocks.append('\n'.join(code))
                code = None
        elif code is not None:
            code.append(line)
        elif line.strip():
            para.append(line)
        elif para:                                # 遇到空行 → 一个段落结束
            blocks.append('\n'.join(para))
            para = []

    if code is not None:                          # 兜底：未闭合的代码块
        blocks.append('\n'.join(code))
    if para:
        blocks.append('\n'.join(para))
    return blocks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list:
    """对超长单块做定长硬切（例如一个几百行的代码块）。"""
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    段落感知分块：把原子块贪心打包到 chunk_size 附近，块间用尾部 overlap 字符重叠。

    数值举例（chunk_size=800、overlap=100）：
        原子块 A=500字符、B=400字符
        → A 单独成块（500 ≤ 800）；放 A+B 会到 902 > 800，于是 B 另起一块，
          并把 A 末尾 100 字符接到 B 前面做重叠，避免跨块语义断裂。
    """
    chunks = []
    cur = ''

    for block in _atomic_blocks(text):
        # 单块本身就超长：先冲掉手上的，再对这块硬切
        if len(block) > chunk_size:
            if cur:
                chunks.append(cur)
                cur = ''
            chunks.extend(_hard_split(block, chunk_size, overlap))
            continue

        candidate = f'{cur}\n\n{block}' if cur else block
        if len(candidate) <= chunk_size:
            cur = candidate
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ''
            cur = f'{tail}\n\n{block}' if tail else block

    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


# ---------- 第 3 部分：向量化 ----------
def embed_all(texts: list, client: OpenAI) -> list:
    """分批向量化，返回与输入等长的向量列表。"""
    vectors = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # 按 index 排序，防止服务端乱序返回导致向量与文本错位
        vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        done = min(start + EMBED_BATCH, total)
        if done % (EMBED_BATCH * 10) == 0 or done == total:
            print(f'[ingest]   已向量化 {done}/{total}')
    return vectors


# ---------- 第 4 部分：写入 Milvus ----------
def write_to_milvus(records: list, dim: int) -> None:
    """
    重建 collection 并分批写入。
    records: [{'id', 'vector', 'text', 'source', 'url'}, ...]
    """
    client = MilvusClient(MILVUS_URI)

    # 数据库不存在则创建（幂等）
    if DB_NAME not in client.list_databases():
        client.create_database(DB_NAME)
    client.use_database(DB_NAME)

    # 先删后建，保证每次入库结果干净可复现
    if COLLECTION_NAME in client.list_collections():
        print(f'[ingest] 删除已存在的 collection：{COLLECTION_NAME}')
        client.drop_collection(COLLECTION_NAME)

    # text/source/url 用动态字段承载，省去逐字段声明（enable_dynamic_field=True）
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    # COSINE 是文本向量的标准选择：只看方向不看模长，不受文本长度影响
    index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='COSINE')

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f'[ingest] 已创建 collection：{COLLECTION_NAME}（dim={dim}，metric=COSINE）')

    # 分批写入：一次性写上千条会触发 gRPC 64MB 消息上限
    total = len(records)
    for start in range(0, total, UPSERT_BATCH):
        batch = records[start:start + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION_NAME, data=batch)
        print(f'[ingest]   已写入 {min(start + UPSERT_BATCH, total)}/{total}')

    client.load_collection(COLLECTION_NAME)

    # 用 count(*) 而不是 get_collection_stats：后者返回的是滞后/近似的统计值，
    # 实测刚写完 6758 条时它报 4800，会让人误以为写入丢失。count(*) 才是准的。
    counted = client.query(
        collection_name=COLLECTION_NAME,
        filter='id >= 0',
        output_fields=['count(*)'],
    )
    print(f'[ingest] 入库完成，collection 行数：{counted[0]["count(*)"]}')


def main() -> int:
    if not API_KEY:
        print('[ingest] 错误：未配置 OPENAI_API_KEY，请在 .env 中填写后重试。')
        return 1

    t0 = time.time()
    client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=HTTP_TIMEOUT, max_retries=2)

    # ① 拉清单 → ② 下载正文
    pages = fetch_page_list()
    if not pages:
        print('[ingest] 错误：文档清单为空，请检查 LLMS_INDEX_URLS 与网络。')
        return 1
    docs = download_pages(pages)
    if not docs:
        print('[ingest] 错误：所有页面均下载失败，请检查网络。')
        return 1

    # ③ 分块：chunk 顺序即 id 顺序，id 从 0 递增
    records = []
    for title, url, body in docs:
        for chunk in split_text(body):
            records.append({'id': len(records), 'text': chunk, 'source': title, 'url': url})
    print(f'[ingest] 分块完成：{len(docs)} 篇 → {len(records)} 条 chunk')

    # ④ 向量化：先探一次维度，这样换 embedding 模型也不用改代码
    dim = len(embed_all(['dimension probe'], client)[0])
    print(f'[ingest] 向量维度：{dim}（{EMBED_MODEL}）')
    vectors = embed_all([r['text'] for r in records], client)
    for record, vector in zip(records, vectors):
        record['vector'] = vector

    # ⑤ 写入 Milvus
    write_to_milvus(records, dim)

    print(f'[ingest] 全部完成，总耗时 {time.time() - t0:.1f} 秒')
    return 0


if __name__ == '__main__':
    sys.exit(main())
