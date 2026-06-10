"""
数据加载与分块模块
负责：读取 JSON → 按字符数分块 → 为每个 chunk 附加元数据
"""

import json
from config import DIARY_JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_diary(path: str = DIARY_JSON_PATH) -> list[dict]:
    """读取结构化日记 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"[数据加载] 读入 {len(entries)} 篇日记")
    return entries


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本按字符数切分为多个 chunk，保留重叠区间以避免语义割裂。
    
    为什么不按句子切？
    - 日记是口语化长段落，一句话可能几百字，按句切粒度不均匀
    - 按固定窗口 + 重叠是 RAG 中最常用的 baseline 方案
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 尝试在标点处断句，避免把一个词劈成两半
        if end < len(text):
            # 从 chunk 末尾往前找最近的断句点
            for punct in ["。", "！", "？", "，", "\n", "；"]:
                last_punct = chunk.rfind(punct)
                if last_punct > chunk_size * 0.5:  # 至少保留一半长度
                    chunk = chunk[:last_punct + 1]
                    end = start + last_punct + 1
                    break
        
        chunks.append(chunk.strip())
        start = end - overlap  # 重叠
    
    return [c for c in chunks if c]  # 去空


def prepare_chunks(entries: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    将所有日记条目切分为 chunks，并生成对应的元数据和 ID。
    
    返回：
        documents: chunk 文本列表
        metadatas: 每个 chunk 的元数据（日期、标题、chunk序号）
        ids:       唯一 ID 列表
    """
    documents = []
    metadatas = []
    ids = []
    
    for entry in entries:
        date = entry["date"]
        title = entry.get("title", "")
        content = entry["content"]
        rating = entry.get("rating", "")
        
        # 在内容前加上日期和标题，让 embedding 能捕获时间信息
        full_text = f"日期：{date} {title}\n{content}"
        if rating:
            full_text += f"\n{rating}"
        
        chunks = chunk_text(full_text)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{date}_chunk{i}"
            documents.append(chunk)
            metadatas.append({
                "date": date,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            ids.append(chunk_id)
    
    print(f"[分块完成] {len(entries)} 篇日记 → {len(documents)} 个 chunks")
    print(f"[分块统计] 平均每篇 {len(documents)/len(entries):.1f} 个 chunks")
    
    return documents, metadatas, ids


if __name__ == "__main__":
    entries = load_diary()
    docs, metas, ids = prepare_chunks(entries)
    
    # 展示几个样本
    print("\n--- 样本 chunk ---")
    for i in range(min(3, len(docs))):
        print(f"\nID: {ids[i]}")
        print(f"Meta: {metas[i]}")
        print(f"Text: {docs[i][:150]}...")
