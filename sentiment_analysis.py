"""
情绪分析模块
对每篇日记调用 LLM 打情绪分数,结果缓存到本地 JSON。

设计要点(面试可讲):
1. 断点续传 —— 每完成一批就保存,中断后重跑会跳过已完成的
2. 并发控制 —— ThreadPoolExecutor 加速,但限制并发数避免 API 限流
3. 结构化输出 —— prompt 要求 LLM 只返回 JSON,便于解析
4. 异常重试 —— 单篇失败自动重试,最终失败的单独记录不阻塞整体

运行: python sentiment_analysis.py
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from config import LLM_PROVIDER, LLM_CONFIG, DIARY_JSON_PATH

# ── 配置 ───────────────────────────────────────────────────────────
SENTIMENT_OUTPUT_PATH = "./data/sentiment_scores.json"
MAX_WORKERS = 5        # 并发线程数(DeepSeek 限流较宽松,5 比较稳)
MAX_RETRIES = 3        # 单篇最大重试次数
SAVE_EVERY = 20        # 每完成多少篇保存一次进度

SENTIMENT_PROMPT = """你是一个日记情绪分析助手。请阅读以下日记,输出一个 JSON 对象,包含:

- "score": 整体情绪分数,整数 1-10(1=非常低落, 5=平静中性, 10=非常开心)
- "emotion": 主导情绪,从这些选项中选一个:["开心", "平静", "充实", "疲惫", "焦虑", "低落", "烦躁", "感动"]
- "reason": 一句话概括打分原因(不超过30字)

只输出 JSON,不要任何其他文字、不要 markdown 代码块。

日记内容:
{content}"""


def get_client() -> OpenAI:
    cfg = LLM_CONFIG[LLM_PROVIDER]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def parse_llm_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON(容错处理:去掉可能的代码块标记)"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        # 校验字段
        score = int(data.get("score", 0))
        if not (1 <= score <= 10):
            return None
        return {
            "score": score,
            "emotion": str(data.get("emotion", "")),
            "reason": str(data.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def score_one_entry(client: OpenAI, entry: dict) -> dict | None:
    """对单篇日记打分,带重试"""
    cfg = LLM_CONFIG[LLM_PROVIDER]
    # 日记太长就截断(情绪判断不需要全文,前1500字足够)
    content = entry["content"][:1500]
    prompt = SENTIMENT_PROMPT.format(content=content)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   # 打分任务要稳定,温度调低
                max_tokens=150,
            )
            result = parse_llm_json(response.choices[0].message.content)
            if result:
                return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s
            else:
                print(f"  ✗ {entry['date']} 失败: {e}")
    return None


def load_existing() -> dict:
    """读取已有的打分结果(断点续传)"""
    if os.path.exists(SENTIMENT_OUTPUT_PATH):
        with open(SENTIMENT_OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    os.makedirs(os.path.dirname(SENTIMENT_OUTPUT_PATH), exist_ok=True)
    with open(SENTIMENT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_batch_sentiment():
    """主流程:批量打分所有日记"""
    with open(DIARY_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = load_existing()
    todo = [e for e in entries if e["date"] not in results]

    print(f"[情绪分析] 总共 {len(entries)} 篇,已完成 {len(results)} 篇,待处理 {len(todo)} 篇")
    if not todo:
        print("[情绪分析] 全部完成,无需处理")
        return results

    client = get_client()
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {
            executor.submit(score_one_entry, client, e): e for e in todo
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            result = future.result()
            if result:
                results[entry["date"]] = result
                done_count += 1
                if done_count % SAVE_EVERY == 0:
                    save_results(results)
                    print(f"  ✓ 进度 {len(results)}/{len(entries)} (已保存)")

    save_results(results)
    print(f"[情绪分析] 完成!成功 {len(results)}/{len(entries)} 篇")
    print(f"[情绪分析] 结果已保存到 {SENTIMENT_OUTPUT_PATH}")

    # 简单统计
    scores = [r["score"] for r in results.values()]
    print(f"\n平均情绪分: {sum(scores)/len(scores):.2f}")
    from collections import Counter
    emotions = Counter(r["emotion"] for r in results.values())
    print("情绪分布:")
    for emo, cnt in emotions.most_common():
        print(f"  {emo}: {cnt} 篇 ({cnt/len(results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    run_batch_sentiment()
