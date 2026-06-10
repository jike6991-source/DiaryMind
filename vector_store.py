"""
向量数据库模块
负责：加载 embedding 模型 → 写入/查询 ChromaDB
"""

import chromadb
from chromadb.utils import embedding_functions
from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


def get_embedding_function():
    """
    加载中文 embedding 模型（BGE-small-zh）。
    
    为什么选 BGE-small-zh？
    - 专门为中文优化，在 C-MTEB 榜单上排名靠前
    - 体积小（~90MB），本地 CPU 就能跑
    - 首次运行会自动从 HuggingFace 下载
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_or_create_collection():
    """获取或创建 ChromaDB collection"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )
    return collection


def index_documents(documents: list[str], metadatas: list[dict], ids: list[str]):
    """
    将 chunks 写入向量数据库。
    
    ChromaDB 会自动调用 embedding function 将文本转为向量，
    然后建立 HNSW 索引用于近似最近邻搜索。
    """
    collection = get_or_create_collection()
    
    # 检查是否已有数据
    existing = collection.count()
    if existing > 0:
        print(f"[向量库] 已有 {existing} 条数据，跳过写入（如需重建请删除 {CHROMA_PERSIST_DIR} 目录）")
        return collection
    
    # ChromaDB 单次最多处理约 5000 条，分批写入
    batch_size = 500
    total = len(documents)
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"[向量库] 已写入 {end}/{total} chunks")
    
    print(f"[向量库] 索引完成，共 {collection.count()} 条向量")
    return collection


def search(query: str, top_k: int = TOP_K, date_filter: dict = None) -> dict:
    """
    语义检索：输入自然语言问题，返回最相关的 chunks。
    
    参数：
        query: 用户的问题
        top_k: 返回结果数
        date_filter: 可选，按日期过滤，如 {"date": {"$gte": "2025-01"}}
    
    返回：
        ChromaDB 查询结果，包含 documents, metadatas, distances
    """
    collection = get_or_create_collection()
    
    query_params = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if date_filter:
        query_params["where"] = date_filter
    
    results = collection.query(**query_params)
    return results


if __name__ == "__main__":
    from data_loader import load_diary, prepare_chunks
    
    # 加载数据并建立索引
    entries = load_diary()
    docs, metas, ids = prepare_chunks(entries)
    collection = index_documents(docs, metas, ids)
    
    # 测试检索
    print("\n--- 测试检索 ---")
    test_queries = [
        "打篮球",
        "考试复习",
        "和朋友吃饭",
    ]
    for q in test_queries:
        results = search(q, top_k=3)
        print(f"\n查询: {q}")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            print(f"  [{i+1}] {meta['date']} (相似度: {1-dist:.3f})")
            print(f"      {doc[:80]}...")
