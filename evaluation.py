"""
RAG 检索评测体系

回答一个核心问题:检索系统能否为一个问题找到正确的日记?

工作流程:
  1. python evaluation.py --generate   # LLM 半自动生成测试集(需人工审核)
  2. 人工审核 data/eval_testset.json,删掉质量差的问题
  3. python evaluation.py --run        # 跑评测,输出 Recall@k 和 MRR

指标说明(面试可讲):
- Recall@k: 前 k 个检索结果中包含正确日记的比例。
  例如 Recall@5 = 0.85 表示 85% 的问题在前5个结果里能找到正确日记。
- MRR (Mean Reciprocal Rank): 正确结果排名的倒数的平均值。
  正确答案排第1得1分,排第2得0.5分,排第5得0.2分,没找到得0分。
  MRR 衡量的是"正确答案排得靠不靠前",比 Recall 更细。

为什么用日期做 ground truth?
  问题由某篇日记生成,该日记的 date 就是标准出处。
  检索结果的 metadata 里有 date 字段,直接比对即可,无需人工标注。
"""

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from config import LLM_PROVIDER, LLM_CONFIG, DIARY_JSON_PATH
from vector_store import search

TESTSET_PATH = "./data/eval_testset.json"
REPORT_PATH = "./data/eval_report.json"
NUM_QUESTIONS = 50      # 生成的测试问题数量
MAX_WORKERS = 5
EVAL_TOP_K = 10         # 评测时检索的结果数(覆盖 Recall@1/3/5/10)

GENERATE_PROMPT = """你是一个测试集构造助手。请阅读以下日记,生成一个可以用来测试检索系统的问题。

要求:
1. 问题必须能用这篇日记的内容回答,且最好只有这篇日记能回答
2. 问题要具体(涉及具体的人物、事件、地点),不要太宽泛
3. 不要在问题中直接给出日期
4. 用第一人称提问(模拟日记主人问自己的日记)
5. 输出 JSON: {{"question": "问题", "answer": "简短的标准答案"}}

反面例子(太宽泛,很多日记都能回答):"我今天做了什么?"
正面例子(具体,指向性强):"我和谁一起去吃羊肉煲结果上成牛肉煲了?"

只输出 JSON,不要任何其他文字。

日记日期:{date}
日记内容:
{content}"""


def get_client() -> OpenAI:
    cfg = LLM_CONFIG[LLM_PROVIDER]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def parse_llm_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        q, a = data.get("question"), data.get("answer")
        if q and a:
            return {"question": str(q), "answer": str(a)}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def generate_one(client: OpenAI, entry: dict) -> dict | None:
    cfg = LLM_CONFIG[LLM_PROVIDER]
    prompt = GENERATE_PROMPT.format(date=entry["date"], content=entry["content"][:1500])
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            result = parse_llm_json(response.choices[0].message.content)
            if result:
                result["source_date"] = entry["date"]
                return result
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def run_generate():
    """生成测试集:随机抽日记,LLM 针对每篇生成一个问题"""
    with open(DIARY_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # 只从内容足够长的日记中抽样(太短的日记问题质量差)
    candidates = [e for e in entries if len(e["content"]) >= 200]
    random.seed(42)  # 固定种子保证可复现
    sampled = random.sample(candidates, min(NUM_QUESTIONS, len(candidates)))

    print(f"[测试集生成] 从 {len(candidates)} 篇日记中抽样 {len(sampled)} 篇")

    client = get_client()
    testset = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_one, client, e): e for e in sampled}
        for future in as_completed(futures):
            result = future.result()
            if result:
                testset.append(result)
                print(f"  ✓ [{result['source_date']}] {result['question']}")

    testset.sort(key=lambda x: x["source_date"])
    os.makedirs(os.path.dirname(TESTSET_PATH), exist_ok=True)
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 生成 {len(testset)} 个测试问题 → {TESTSET_PATH}")
    print("请人工审核该文件:删掉太宽泛或质量差的问题,然后运行:")
    print("  python evaluation.py --run")


def dedup_dates(dates: list[str]) -> list[str]:
    """保序去重:同一篇日记的多个chunk只算一次"""
    seen, out = set(), []
    for d in dates:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def evaluate_with(search_fn, testset, label: str, verbose: bool = True) -> dict:
    """用给定的检索函数跑一轮评测"""
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    failures = []

    for i, item in enumerate(testset):
        question = item["question"]
        gt_date = item["source_date"]

        results = search_fn(question, top_k=EVAL_TOP_K)
        retrieved_dates = dedup_dates([m["date"] for m in results["metadatas"][0]])

        rank = None
        for r, d in enumerate(retrieved_dates, start=1):
            if d == gt_date:
                rank = r
                break

        if rank is not None:
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / rank)
            status = f"✓ rank={rank}"
        else:
            reciprocal_ranks.append(0.0)
            failures.append(item)
            status = "✗ miss"

        if verbose:
            print(f"  [{label}] [{i+1:2d}/{len(testset)}] {status}  {question[:38]}")

    n = len(testset)
    return {
        "mode": label,
        "num_questions": n,
        "recall@1": hits_at[1] / n,
        "recall@3": hits_at[3] / n,
        "recall@5": hits_at[5] / n,
        "recall@10": hits_at[10] / n,
        "mrr": sum(reciprocal_ranks) / n,
        "failed_questions": [
            {"question": f["question"], "source_date": f["source_date"]}
            for f in failures
        ],
    }


def print_report(report: dict):
    print(f"  Recall@1:  {report['recall@1']*100:.1f}%")
    print(f"  Recall@3:  {report['recall@3']*100:.1f}%")
    print(f"  Recall@5:  {report['recall@5']*100:.1f}%")
    print(f"  Recall@10: {report['recall@10']*100:.1f}%")
    print(f"  MRR:       {report['mrr']:.3f}")


def run_evaluation(mode: str = "both"):
    """跑检索评测。mode: vector | hybrid | both"""
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)

    print(f"[评测] 共 {len(testset)} 个测试问题,检索 top-{EVAL_TOP_K},模式: {mode}\n")

    reports = {}

    if mode in ("vector", "both"):
        reports["vector"] = evaluate_with(
            lambda q, top_k: search(q, top_k=top_k), testset, "vector")

    if mode in ("hybrid", "both"):
        from hybrid_search import hybrid_search
        reports["hybrid"] = evaluate_with(
            lambda q, top_k: hybrid_search(q, top_k=top_k), testset, "hybrid")

    print(f"\n{'='*50}")
    print(f"评测报告 (n={len(testset)})")
    print(f"{'='*50}")
    for label, rep in reports.items():
        print(f"\n【{label}】")
        print_report(rep)

    if "vector" in reports and "hybrid" in reports:
        v, h = reports["vector"], reports["hybrid"]
        print(f"\n【提升对比 vector → hybrid】")
        for metric in ["recall@1", "recall@3", "recall@5", "recall@10"]:
            delta = (h[metric] - v[metric]) * 100
            print(f"  {metric}: {v[metric]*100:.1f}% → {h[metric]*100:.1f}%  ({delta:+.1f}pp)")
        print(f"  mrr: {v['mrr']:.3f} → {h['mrr']:.3f}")

        hyb_failures = reports["hybrid"]["failed_questions"]
        print(f"\n混合检索仍未命中 ({len(hyb_failures)} 个):")
        for f_item in hyb_failures[:10]:
            print(f"  ✗ [{f_item['source_date']}] {f_item['question']}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存 → {REPORT_PATH}")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        run_generate()
    elif "--run" in sys.argv:
        mode = "both"
        if "--mode" in sys.argv:
            mode = sys.argv[sys.argv.index("--mode") + 1]
        run_evaluation(mode)
    else:
        print("用法:")
        print("  python evaluation.py --generate              # 第一步:生成测试集")
        print("  python evaluation.py --run                   # 第二步:跑评测(默认对比两种模式)")
        print("  python evaluation.py --run --mode vector     # 只测纯向量")
        print("  python evaluation.py --run --mode hybrid     # 只测混合检索")
