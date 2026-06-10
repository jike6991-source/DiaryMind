"""
DiaryMind — Streamlit 交互界面(v3: 加入社交关系图谱)
运行方式:streamlit run app.py
"""

import streamlit as st
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter, defaultdict

from rag_pipeline import ask
from config import DIARY_JSON_PATH

SENTIMENT_PATH = "./data/sentiment_scores.json"
PERSONS_PATH = "./data/persons_final.json"

# ── 页面配置 ───────────────────────────────────────────────────────
st.set_page_config(page_title="DiaryMind", page_icon="📔", layout="wide")
st.title("📔 DiaryMind")
st.caption("基于 RAG 的个人日记智能分析系统")


@st.cache_data
def load_entries():
    with open(DIARY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


entries = load_entries()
sentiment = load_json_if_exists(SENTIMENT_PATH)
persons = load_json_if_exists(PERSONS_PATH)

# ── 侧边栏 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 数据概览")
    st.metric("日记总篇数", len(entries))
    st.metric("时间跨度", f"{entries[0]['date']} ~ {entries[-1]['date']}")
    total_chars = sum(len(e["content"]) for e in entries)
    st.metric("总字数", f"{total_chars:,}")
    if sentiment:
        avg_score = sum(r["score"] for r in sentiment.values()) / len(sentiment)
        st.metric("平均情绪分", f"{avg_score:.2f} / 10")
    if persons:
        st.metric("日记中的人物", f"{len(persons.get('frequency', {}))} 人")

    st.divider()
    st.subheader("月度日记量")
    monthly = Counter(e["date"][:7] for e in entries)
    months = sorted(monthly.keys())
    fig_monthly = go.Figure(go.Bar(
        x=months, y=[monthly[m] for m in months], marker_color="#5DCAA5",
    ))
    fig_monthly.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_tickangle=-45, xaxis_tickfont_size=10, yaxis_title="篇数",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.divider()
    st.subheader("日期过滤")
    use_filter = st.checkbox("限定日期范围")
    date_filter = None
    if use_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("起始", datetime(2024, 8, 26))
        with col2:
            end_date = st.date_input("截止", datetime(2026, 5, 20))
        date_filter = {
            "$and": [
                {"date": {"$gte": start_date.strftime("%Y-%m-%d")}},
                {"date": {"$lte": end_date.strftime("%Y-%m-%d")}},
            ]
        }

# ── 主界面 ─────────────────────────────────────────────────────────
tab_chat, tab_browse, tab_stats, tab_mood, tab_social = st.tabs(
    ["💬 智能问答", "📅 日记浏览", "📈 数据分析", "💭 情绪分析", "👥 社交图谱"]
)

# ── Tab 1: RAG 问答 ───────────────────────────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 查看来源日记片段"):
                    st.text(msg["sources"])

    if query := st.chat_input("问我关于日记的任何问题..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("正在检索日记并生成回答..."):
                answer, context = ask(query, date_filter=date_filter)
            st.markdown(answer)
            if context:
                with st.expander("📎 查看来源日记片段"):
                    st.text(context)
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": context,
        })

    st.divider()
    st.caption("试试这些问题:")
    cols = st.columns(3)
    quick_questions = [
        "我最常做的运动是什么?",
        "总结一下我2025年3月的生活",
        "我和朋友一起做过哪些有趣的事?",
    ]
    for col, q in zip(cols, quick_questions):
        if col.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ── Tab 2: 日记浏览 ───────────────────────────────────────────────
with tab_browse:
    all_months = sorted(set(e["date"][:7] for e in entries))
    selected_month = st.selectbox("选择月份", all_months, index=len(all_months) - 1)
    month_entries = [e for e in entries if e["date"].startswith(selected_month)]
    st.caption(f"共 {len(month_entries)} 篇")
    for e in month_entries:
        mood_tag = ""
        if e["date"] in sentiment:
            s = sentiment[e["date"]]
            mood_tag = f"　[{s['emotion']} {s['score']}分]"
        with st.expander(f"📅 {e['date']}　{e['title']}{mood_tag}"):
            st.write(e["content"])
            if e.get("rating"):
                st.info(e["rating"])

# ── Tab 3: 数据分析 ───────────────────────────────────────────────
with tab_stats:
    st.subheader("字数趋势")
    dates = [e["date"] for e in entries]
    lengths = [len(e["content"]) for e in entries]
    fig_len = px.scatter(x=dates, y=lengths, labels={"x": "日期", "y": "字数"}, opacity=0.6)
    fig_len.add_trace(go.Scatter(
        x=dates,
        y=[sum(lengths[max(0, i - 6):i + 1]) / min(7, i + 1) for i in range(len(lengths))],
        mode="lines", name="7日均值", line=dict(color="#D85A30", width=2),
    ))
    fig_len.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig_len, use_container_width=True)

    st.divider()
    st.subheader("高频词统计")
    st.caption("基于 jieba 分词的 Top 30 关键词")

    @st.cache_data
    def get_word_freq():
        import jieba
        stopwords = set("的了是在我有和就都也不要还这个人去到说会他她它们一个上下不是没有但是然后".split())
        stopwords.update(["emmm", "然后", "就是", "还是", "还有", "因为", "所以", "可以", "今天",
                          "一个", "什么", "觉得", "但是", "其实", "不过", "那个", "这个", "已经"])
        word_counts = Counter()
        for e in entries:
            words = jieba.lcut(e["content"])
            words = [w for w in words if len(w) >= 2 and w not in stopwords]
            word_counts.update(words)
        return word_counts.most_common(30)

    top_words = get_word_freq()
    words, counts = zip(*top_words)
    fig_words = go.Figure(go.Bar(
        x=list(reversed(counts)), y=list(reversed(words)),
        orientation="h", marker_color="#7F77DD",
    ))
    fig_words.update_layout(height=600, margin=dict(l=80, r=20, t=10, b=10))
    st.plotly_chart(fig_words, use_container_width=True)

# ── Tab 4: 情绪分析 ───────────────────────────────────────────────
with tab_mood:
    if not sentiment:
        st.warning("还没有情绪分析数据。请先运行: `python sentiment_analysis.py`")
    else:
        st.caption(f"已分析 {len(sentiment)} / {len(entries)} 篇日记")

        st.subheader("情绪时间线")
        s_dates = sorted(sentiment.keys())
        s_scores = [sentiment[d]["score"] for d in s_dates]
        rolling = [
            sum(s_scores[max(0, i - 6):i + 1]) / min(7, i + 1)
            for i in range(len(s_scores))
        ]

        fig_mood = go.Figure()
        fig_mood.add_trace(go.Scatter(
            x=s_dates, y=s_scores, mode="markers", name="单日情绪分",
            marker=dict(size=5, opacity=0.4, color=s_scores,
                        colorscale=[[0, "#D85A30"], [0.5, "#E0A815"], [1, "#1D9E75"]],
                        cmin=1, cmax=10),
            hovertemplate="%{x}<br>情绪分: %{y}<extra></extra>",
        ))
        fig_mood.add_trace(go.Scatter(
            x=s_dates, y=rolling, mode="lines", name="7日均值",
            line=dict(color="#4A7BD0", width=2.5),
        ))
        fig_mood.add_hline(y=5, line_dash="dash", line_color="gray",
                           annotation_text="中性线")
        fig_mood.update_layout(
            height=400, yaxis=dict(range=[0.5, 10.5], title="情绪分"),
            xaxis_title="日期", showlegend=True,
        )
        st.plotly_chart(fig_mood, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("情绪类型分布")
            emo_counts = Counter(r["emotion"] for r in sentiment.values())
            fig_pie = px.pie(
                values=list(emo_counts.values()),
                names=list(emo_counts.keys()),
                hole=0.45,
            )
            fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("月度平均情绪")
            month_scores = defaultdict(list)
            for d, r in sentiment.items():
                month_scores[d[:7]].append(r["score"])
            m_keys = sorted(month_scores.keys())
            m_avgs = [sum(month_scores[m]) / len(month_scores[m]) for m in m_keys]
            fig_month = go.Figure(go.Bar(
                x=m_keys, y=m_avgs,
                marker_color=["#1D9E75" if v >= 6 else "#E0A815" if v >= 5 else "#D85A30" for v in m_avgs],
            ))
            fig_month.update_layout(
                height=350, yaxis=dict(range=[0, 10], title="平均分"),
                xaxis_tickangle=-45, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_month, use_container_width=True)

        st.divider()
        st.subheader("情绪极值日")
        sorted_days = sorted(sentiment.items(), key=lambda x: x[1]["score"])
        entry_map = {e["date"]: e for e in entries}
        col_low, col_high = st.columns(2)
        with col_low:
            st.markdown("**😞 最低落的 5 天**")
            for d, r in sorted_days[:5]:
                title = entry_map.get(d, {}).get("title", "")
                st.markdown(f"- `{d}` ({r['score']}分) {title}  \n  *{r['reason']}*")
        with col_high:
            st.markdown("**😄 最开心的 5 天**")
            for d, r in sorted_days[-5:][::-1]:
                title = entry_map.get(d, {}).get("title", "")
                st.markdown(f"- `{d}` ({r['score']}分) {title}  \n  *{r['reason']}*")

# ── Tab 5: 社交图谱 ───────────────────────────────────────────────
with tab_social:
    if not persons:
        st.warning(
            "还没有人物数据。请按顺序运行:\n\n"
            "1. `python person_extraction.py` (LLM 批量抽取)\n"
            "2. 编辑 `data/person_aliases.json` 确认别名合并\n"
            "3. `python person_extraction.py --merge` (生成最终统计)"
        )
    else:
        freq = persons["frequency"]
        cooccur = persons["cooccurrence"]
        person_dates = persons["person_dates"]

        # ── 关系网络图 ──
        st.subheader("社交关系网络")
        st.caption("节点大小 = 出现频次,连线粗细 = 共同出现次数。可拖拽、缩放。")

        # 控制参数
        col_a, col_b = st.columns(2)
        with col_a:
            top_n = st.slider("显示人数 (按频次排序)", 5, min(50, len(freq)), min(20, len(freq)))
        with col_b:
            min_edge = st.slider("最小共现次数 (过滤弱关系)", 1, 10, 2)

        @st.cache_data
        def build_network_figure(top_n, min_edge):
            import networkx as nx

            top_persons = dict(list(freq.items())[:top_n])
            max_freq = max(top_persons.values())

            # 构建图:中心节点"我" + 人物节点
            G = nx.Graph()
            G.add_node("我")
            for p in top_persons:
                G.add_node(p)
                # 与"我"的边权 = 出现频次
                G.add_edge("我", p, weight=top_persons[p], kind="me")

            for item in cooccur:
                a, b, c = item["a"], item["b"], item["count"]
                if a in top_persons and b in top_persons and c >= min_edge:
                    G.add_edge(a, b, weight=c, kind="co")

            # spring 布局:边权越大离得越近
            pos = nx.spring_layout(G, k=0.9, weight="weight", seed=42)

            # 边轨迹(分两类:我-人物 灰色,人物-人物 绿色)
            edge_traces = []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                if d["kind"] == "me":
                    color = "rgba(180,180,180,0.35)"
                    width = 0.5 + 3 * (d["weight"] / max_freq)
                else:
                    color = "rgba(93,202,165,0.7)"
                    width = 0.8 + d["weight"] * 0.6
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="text",
                    text=f"{u} ↔ {v}: {d['weight']} 次",
                    showlegend=False,
                ))

            # 节点轨迹
            node_x, node_y, node_size, node_text, node_color = [], [], [], [], []
            for n in G.nodes():
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                if n == "我":
                    node_size.append(46)
                    node_color.append("#D85A30")
                    node_text.append("我")
                else:
                    c = top_persons[n]
                    node_size.append(16 + 30 * (c / max_freq))
                    node_color.append("#4A7BD0")
                    node_text.append(f"{n}<br>出现 {c} 次")

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=[n for n in G.nodes()],
                textposition="top center",
                textfont=dict(size=13),
                hovertext=node_text, hoverinfo="text",
                marker=dict(size=node_size, color=node_color,
                            line=dict(width=1.5, color="white")),
                showlegend=False,
            )

            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                height=600, showlegend=False,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="white",
                hovermode="closest",
            )
            return fig

        try:
            fig_net = build_network_figure(top_n, min_edge)
            st.plotly_chart(fig_net, use_container_width=True)
        except ImportError:
            st.error("缺少 networkx 库,请运行: `pip install networkx`")

        st.divider()

        # ── 人物频次榜 ──
        col_rank, col_timeline = st.columns([1, 2])

        with col_rank:
            st.subheader("人物出现榜")
            top15 = list(freq.items())[:15]
            names = [p for p, _ in top15]
            counts = [c for _, c in top15]
            fig_rank = go.Figure(go.Bar(
                x=list(reversed(counts)), y=list(reversed(names)),
                orientation="h", marker_color="#4A7BD0",
            ))
            fig_rank.update_layout(height=450, margin=dict(l=60, r=20, t=10, b=10))
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_timeline:
            st.subheader("友谊时间线")
            selected_person = st.selectbox("选择一个人", list(freq.keys()))
            p_dates = person_dates.get(selected_person, [])

            # 按月统计该人物出现次数
            p_monthly = Counter(d[:7] for d in p_dates)
            all_months_range = sorted(set(e["date"][:7] for e in entries))
            p_counts = [p_monthly.get(m, 0) for m in all_months_range]

            fig_person = go.Figure(go.Bar(
                x=all_months_range, y=p_counts, marker_color="#7F77DD",
            ))
            fig_person.update_layout(
                height=380, yaxis_title="出现次数",
                xaxis_tickangle=-45,
                title=f"「{selected_person}」在日记中的出现分布 (共 {len(p_dates)} 次)",
            )
            st.plotly_chart(fig_person, use_container_width=True)
