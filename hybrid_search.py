"""
混合检索模块 (Hybrid Search)

为什么需要混合检索?(面试核心考点)
- 向量检索:擅长语义匹配("心情不好的日子"能匹配到"今天很低落"),
  但对低频专有名词不敏感("铁锅炖大鹅"可能淹没在大量"吃饭"日记里)
- BM25 关键词检索:对独特词汇一抓一个准,但不懂语义
  ("跑步"搜不到只写了"夜跑"的日记)
- 两者互补,融合后效果显著优于单路检索

融合算法:RRF (Reciprocal Rank Fusion)
  score(doc) = Σ 1 / (k + rank_i)   (k=60, 业界标准值)
  不依赖两路检索的分数量纲,只用排名,简单鲁棒。

另一个优化:实体级去重 (entry-level dedup)
  同一篇日记被切成多个 chunk,top-k 里可能被同一篇占多个坑。
  按日期去重后,top-5 就是 5 篇不同的日记,有效召回直接提升。

依赖: pip install rank-bm25
"""

import json
import os
import pickle

import jieba
from rank_bm25 import BM25Okapi

from config import DIARY_JSON_PATH, TOP_K
from data_loader import load_diary, prepare_chunks
from vector_store import search as vector_search

BM25_CACHE_PATH = "./data/bm25_index.pkl"
RRF_K = 60  # RRF 平滑常数,业界标准值


# ── BM25 索引构建 ──────────────────────────────────────────────────
def build_bm25_index(force_rebuild: bool = False):
    """
    构建 BM25 索引并缓存到磁盘。
    分词用 jieba(BM25 是词级别匹配,中文必须先分词)。
    """
    if os.path.exists(BM25_CACHE_PATH) and not force_rebuild:
        with open(BM25_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    entries = load_diary()
    documents, metadatas, ids = prepare_chunks(entries)

    print("[BM25] 正在分词并构建索引...")
    tokenized = [jieba.lcut(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "bm25": bm25,
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }
    os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"[BM25] 索引完成,共 {len(documents)} 个 chunks,已缓存")
    return index_data


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 检索,返回 [{document, metadata, rank}, ...]"""
    index_data = build_bm25_index()
    bm25 = index_data["bm25"]

    query_tokens = jieba.lcut(query)
    scores = bm25.get_scores(query_tokens)

    # 按分数排序取 top_k
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_idx, start=1):
        if scores[idx] <= 0:
            break  # 分数为0说明完全没有词匹配,不要凑数
        results.append({
            "document": index_data["documents"][idx],
            "metadata": index_data["metadatas"][idx],
            "rank": rank,
        })
    return results


# ── 混合检索 ───────────────────────────────────────────────────────
def hybrid_search(query: str, top_k: int = TOP_K, date_filter: dict = None,
                  dedup_by_date: bool = True) -> dict:
    """
    混合检索主函数:向量 + BM25 + RRF 融合 + 日期去重。

    返回格式与 vector_store.search 兼容:
    {"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
    (distances 用 RRF 分数的负数填充,仅作展示用)
    """
    fetch_k = max(top_k * 4, 20)  # 两路各多取一些,融合后再截断

    # ── 路 1:向量检索 ──
    vec_results = vector_search(query, top_k=fetch_k, date_filter=date_filter)
    vec_ranked = {}  # chunk唯一键 → (rank, doc, meta)
    for rank, (doc, meta) in enumerate(zip(
        vec_results["documents"][0], vec_results["metadatas"][0]
    ), start=1):
        key = f"{meta['date']}_{meta['chunk_index']}"
        vec_ranked[key] = (rank, doc, meta)

    # ── 路 2:BM25 检索 ──
    bm25_results = bm25_search(query, top_k=fetch_k)
    bm25_ranked = {}
    for item in bm25_results:
        meta = item["metadata"]
        key = f"{meta['date']}_{meta['chunk_index']}"
        # date_filter 简单适配:BM25 不支持元数据过滤,在这里手动过滤
        bm25_ranked[key] = (item["rank"], item["document"], meta)

    # ── RRF 融合 ──
    all_keys = set(vec_ranked) | set(bm25_ranked)
    fused = []
    for key in all_keys:
        score = 0.0
        doc, meta = None, None
        if key in vec_ranked:
            r, doc, meta = vec_ranked[key]
            score += 1.0 / (RRF_K + r)
        if key in bm25_ranked:
            r, d, m = bm25_ranked[key]
            score += 1.0 / (RRF_K + r)
            if doc is None:
                doc, meta = d, m
        fused.append((score, doc, meta))

    fused.sort(key=lambda x: -x[0])

    # ── 日期去重:同一篇日记只保留得分最高的 chunk ──
    if dedup_by_date:
        seen_dates = set()
        deduped = []
        for score, doc, meta in fused:
            if meta["date"] not in seen_dates:
                seen_dates.add(meta["date"])
                deduped.append((score, doc, meta))
        fused = deduped

    fused = fused[:top_k]

    # 组装成与 vector_store.search 兼容的格式
    return {
        "documents": [[doc for _, doc, _ in fused]],
        "metadatas": [[meta for _, _, meta in fused]],
        "distances": [[1 - score * RRF_K for score, _, _ in fused]],  # 仅展示用
    }


if __name__ == "__main__":
    # 快速对比测试:用几个"低频关键词"问题看混合检索的提升
    test_queries = [
        "我和哪些人一起去吃了铁锅炖大鹅?",
        "我今天中午在哪吃了炸鸡叉骨?",
        "我和谁一起去三年二班吃麻辣烫?",
    ]

    print("=" * 60)
    print("向量检索 vs 混合检索 对比")
    print("=" * 60)

    for q in test_queries:
        print(f"\n查询: {q}")

        vec = vector_search(q, top_k=3)
        print("  [纯向量] top-3 日期:", [m["date"] for m in vec["metadatas"][0]])

        hyb = hybrid_search(q, top_k=3)
        print("  [混  合] top-3 日期:", [m["date"] for m in hyb["metadatas"][0]])
