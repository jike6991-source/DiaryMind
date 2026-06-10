"""
人物提取模块
对每篇日记调用 LLM 抽取出现的人物,结果缓存到本地。

设计要点(面试可讲):
1. 为什么用 LLM 而不是 jieba 词性标注?
   日记里人物称呼口语化(绰号、姓氏简称、外号),传统 NER 识别率低,
   LLM 能结合上下文判断"练""崔""老五"这种称呼是人物。
2. 别名问题:同一人可能被抽成"苟豪林"和"苟"。
   解决:抽取后生成候选别名映射文件 person_aliases.json,
   用户人工确认合并规则,程序按规则归一化。
3. 与情绪分析相同的工程模式:断点续传 + 并发 + 重试。

运行:
  python person_extraction.py            # 第一步:批量抽取
  (手动编辑 data/person_aliases.json 合并别名)
  python person_extraction.py --merge    # 第二步:应用别名合并,生成最终统计

"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from openai import OpenAI
from config import LLM_PROVIDER, LLM_CONFIG, DIARY_JSON_PATH

# ── 配置 ───────────────────────────────────────────────────────────
PERSONS_RAW_PATH = "./data/persons_raw.json"        # LLM 抽取的原始结果
ALIASES_PATH = "./data/person_aliases.json"          # 别名映射(用户可编辑)
PERSONS_FINAL_PATH = "./data/persons_final.json"     # 合并后的最终结果
MAX_WORKERS = 5
MAX_RETRIES = 3
SAVE_EVERY = 20
MIN_FREQ_FOR_GRAPH = 3   # 出现少于这个次数的人不进入统计(过滤一次性提及)

EXTRACT_PROMPT = """你是一个日记人物提取助手。请阅读以下日记,提取其中出现的人物。

规则:
1. 只提取真实出现/互动的人物(朋友、同学、家人、老师等)
2. 用日记中的称呼原文(如"苟豪林"、"老五"、"崔"都保留原样)
3. 不要提取:明星、网红、游戏角色、未出现只是被提到名字的公众人物
4. 不要提取"我"(日记作者本人)
5. 输出 JSON: {{"persons": ["称呼1", "称呼2"]}}
6. 如果没有人物,输出 {{"persons": []}}

只输出 JSON,不要任何其他文字。

日记内容:
{content}"""


def get_client() -> OpenAI:
    cfg = LLM_CONFIG[LLM_PROVIDER]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def parse_llm_json(text: str) -> list | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        persons = data.get("persons", None)
        if not isinstance(persons, list):
            return None
        # 清洗:去空、去重、限制长度(超过6个字的大概率不是称呼)
        cleaned = []
        for p in persons:
            p = str(p).strip()
            if p and len(p) <= 6 and p not in cleaned:
                cleaned.append(p)
        return cleaned
    except (json.JSONDecodeError, TypeError):
        return None


def extract_one(client: OpenAI, entry: dict) -> list | None:
    cfg = LLM_CONFIG[LLM_PROVIDER]
    content = entry["content"][:2000]
    prompt = EXTRACT_PROMPT.format(content=content)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            result = parse_llm_json(response.choices[0].message.content)
            if result is not None:
                return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ✗ {entry['date']} 失败: {e}")
    return None


def run_extraction():
    """第一步:批量抽取所有日记中的人物"""
    with open(DIARY_JSON_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = {}
    if os.path.exists(PERSONS_RAW_PATH):
        with open(PERSONS_RAW_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

    todo = [e for e in entries if e["date"] not in results]
    print(f"[人物提取] 总共 {len(entries)} 篇,已完成 {len(results)} 篇,待处理 {len(todo)} 篇")

    if todo:
        client = get_client()
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_one, client, e): e for e in todo}
            for future in as_completed(futures):
                entry = futures[future]
                result = future.result()
                if result is not None:
                    results[entry["date"]] = result
                    done += 1
                    if done % SAVE_EVERY == 0:
                        os.makedirs(os.path.dirname(PERSONS_RAW_PATH), exist_ok=True)
                        with open(PERSONS_RAW_PATH, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"  ✓ 进度 {len(results)}/{len(entries)} (已保存)")

        os.makedirs(os.path.dirname(PERSONS_RAW_PATH), exist_ok=True)
        with open(PERSONS_RAW_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[人物提取] 完成 {len(results)}/{len(entries)} 篇")

    # 生成别名候选文件
    generate_alias_candidates(results)


def generate_alias_candidates(results: dict):
    """
    自动发现可能的别名对:如果 A 是 B 的子串(如"苟"⊂"苟豪林"),则可能是同一人。
    生成候选映射文件供用户人工确认。
    """
    freq = Counter()
    for persons in results.values():
        freq.update(persons)

    names = list(freq.keys())
    candidates = {}

    for short, long in combinations(sorted(names, key=len), 2):
        if len(short) < len(long) and short in long:
            # 短名是长名的子串 → 候选合并:短名 → 长名
            candidates[short] = long

    # 如果已有别名文件,保留用户的修改
    existing = {}
    if os.path.exists(ALIASES_PATH):
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    merged = {**candidates, **existing}  # 用户的修改优先

    with open(ALIASES_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[别名候选] 已生成 {ALIASES_PATH}")
    print("请打开该文件人工检查:")
    print('  格式: {"短称呼": "归一化为的全名"}')
    print('  正确的映射保留,错误的删掉,也可以手动添加(如 {"老五": "王某某"})')
    print("  确认完成后运行: python person_extraction.py --merge")

    print(f"\n当前自动发现的候选 (共{len(candidates)}对):")
    for s, l in sorted(candidates.items(), key=lambda x: -freq[x[0]])[:20]:
        print(f"  {s} ({freq[s]}次) → {l} ({freq[l]}次)")


def run_merge():
    """第二步:应用别名映射,生成最终的人物统计"""
    with open(PERSONS_RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    aliases = {}
    if os.path.exists(ALIASES_PATH):
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            aliases = json.load(f)

    # 归一化:把每篇的人名按别名映射替换,去重
    normalized = {}
    for date, persons in raw.items():
        mapped = []
        for p in persons:
            canonical = aliases.get(p, p)
            if canonical not in mapped:
                mapped.append(canonical)
        normalized[date] = mapped

    # 统计
    freq = Counter()
    cooccur = Counter()  # (A, B) 共现次数
    person_dates = defaultdict(list)

    for date, persons in normalized.items():
        freq.update(persons)
        for p in persons:
            person_dates[p].append(date)
        for a, b in combinations(sorted(persons), 2):
            cooccur[(a, b)] += 1

    # 过滤低频人物
    keep = {p for p, c in freq.items() if c >= MIN_FREQ_FOR_GRAPH}
    print(f"[合并完成] 共 {len(freq)} 个人物,出现≥{MIN_FREQ_FOR_GRAPH}次的有 {len(keep)} 人")

    output = {
        "frequency": {p: c for p, c in freq.most_common() if p in keep},
        "cooccurrence": [
            {"a": a, "b": b, "count": c}
            for (a, b), c in cooccur.most_common()
            if a in keep and b in keep
        ],
        "person_dates": {p: sorted(ds) for p, ds in person_dates.items() if p in keep},
    }

    with open(PERSONS_FINAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[输出] {PERSONS_FINAL_PATH}")
    print(f"\nTop 15 人物:")
    for p, c in list(output["frequency"].items())[:15]:
        print(f"  {p}: {c} 次")
    print(f"\nTop 10 共现关系:")
    for item in output["cooccurrence"][:10]:
        print(f"  {item['a']} ↔ {item['b']}: {item['count']} 次")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        run_merge()
    else:
        run_extraction()
