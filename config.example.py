"""
DiaryMind 配置模板
使用方法:复制本文件为 config.py,填入你的 API Key
  cp config.example.py config.py    (Windows: copy config.example.py config.py)

API Key 也可以通过环境变量 DIARYMIND_API_KEY 提供,优先级高于本文件。
"""

import os

# ── 嵌入模型(本地运行,无需 API)──────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512

# ── LLM 配置 ───────────────────────────────────────────────────────
LLM_PROVIDER = "deepseek"

LLM_CONFIG = {
    "deepseek": {
        "api_key": os.environ.get("DIARYMIND_API_KEY", "你的 DeepSeek API Key"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "zhipu": {
        "api_key": os.environ.get("DIARYMIND_API_KEY", "你的智谱 API Key"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
}

# ── 向量数据库 ─────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "diary_entries"

# ── RAG 参数 ───────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# ── 数据路径 ───────────────────────────────────────────────────────
DIARY_JSON_PATH = "./data/diary_structured.json"
