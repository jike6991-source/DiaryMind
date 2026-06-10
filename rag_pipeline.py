"""
RAG 问答管线
负责：接收用户问题 → 检索相关日记 → 构造 prompt → 调用 LLM 生成回答
"""

from openai import OpenAI
try:
    from hybrid_search import hybrid_search as search   # 优先用混合检索
except ImportError:
    from vector_store import search                      # 回退到纯向量
from config import LLM_PROVIDER, LLM_CONFIG, TOP_K


def get_llm_client() -> OpenAI:
    """创建 LLM 客户端（兼容 OpenAI 接口格式）"""
    cfg = LLM_CONFIG[LLM_PROVIDER]
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )


SYSTEM_PROMPT = """你是一个私人日记分析助手。用户会向你提问关于他过去生活的问题，
你需要根据提供的日记片段来回答。

规则：
1. 只根据提供的日记内容回答，不要编造不存在的事件
2. 如果日记中没有相关信息，坦诚说明"在你的日记中没有找到相关记录"
3. 回答时引用具体日期，让用户知道信息来源
4. 语气自然亲切，像朋友一样聊天
5. 如果涉及多个时间点，按时间顺序整理
"""


def build_context(query: str, top_k: int = TOP_K, date_filter: dict = None) -> str:
    """
    检索相关日记片段并拼接为上下文。
    
    这是 RAG 的核心环节：把向量检索结果格式化为 LLM 能理解的上下文。
    """
    results = search(query, top_k=top_k, date_filter=date_filter)
    
    if not results["documents"][0]:
        return ""
    
    context_parts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist
        context_parts.append(
            f"[{meta['date']} | 相似度 {similarity:.2f}]\n{doc}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def ask(query: str, top_k: int = TOP_K, date_filter: dict = None, stream: bool = False):
    """
    RAG 问答主函数。
    
    流程：query → 向量检索 → 拼接 context → LLM 生成
    
    参数：
        query: 用户问题
        top_k: 检索数量
        date_filter: 日期过滤
        stream: 是否流式输出
    
    返回：
        answer: LLM 的回答
        context: 检索到的日记片段（用于展示来源）
    """
    # Step 1: 检索
    context = build_context(query, top_k, date_filter)
    
    if not context:
        return "在你的日记中没有找到与这个问题相关的记录。", ""
    
    # Step 2: 构造 prompt
    user_message = f"""以下是从日记中检索到的相关片段：

{context}

---

用户的问题：{query}

请根据以上日记内容回答。"""

    # Step 3: 调用 LLM
    client = get_llm_client()
    cfg = LLM_CONFIG[LLM_PROVIDER]
    
    if stream:
        # 流式输出（用于 Streamlit）
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            stream=True,
        )
        return response, context  # 返回 stream 对象
    else:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        return answer, context


if __name__ == "__main__":
    print("DiaryMind RAG 测试")
    print("=" * 40)
    
    while True:
        query = input("\n请输入问题（输入 q 退出）: ")
        if query.lower() == "q":
            break
        
        answer, context = ask(query)
        print(f"\n📝 回答:\n{answer}")
        print(f"\n📎 来源片段数: {len(context.split('---')) if context else 0}")
