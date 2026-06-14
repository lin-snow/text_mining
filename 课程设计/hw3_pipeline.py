# -*- coding: utf-8 -*-
"""课程设计：基于 LDA 主题模型的中文科技 RSS 语料主题结构与演化挖掘。

复现论文方法：Blei, Ng, Jordan (2003) *Latent Dirichlet Allocation*, JMLR 3:993-1022.

端到端流水线：
  ①  载入作业 1 的 RSS 语料（含来源 feed 与发布时间 published）
  ②  载入作业 2 的定制化分词词典 + 合并停用词，jieba 分词
  ③  构建 gensim 词典与 BoW 语料，filter_extremes 去极端词
  ④  困惑度(perplexity) + 主题一致性(C_v) 扫描，选择主题数 K
  ⑤  以最优 K 训练最终 LDA，得到 文档-主题 θ 与 主题-词 φ
  ⑥  导出数据：选 K 曲线 / 主题词 / 文档主题 / 主题×时间 / 主题×来源 / 新词归属
  ⑦  可视化：选K曲线、主题词网格、主题河流图(随时间)、主题×来源热力图、
       主题间距离图(JS散度+经典MDS)、作业2新词的主题归属、主题词云网格

设计目标：方法严格对齐 LDA 原论文；数据与分词深度联动作业 1、作业 2；
全部图表 Matplotlib/WordCloud 离线生成。
"""
from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

warnings.filterwarnings("ignore")

# --------------------------- 路径 ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

HW1 = ROOT.parent / "作业 1"
HW2 = ROOT.parent / "作业 2"
CORPUS_CSV = HW1 / "data" / "rss_cleaned.csv"
STOPWORDS_PATH = HW1 / "data" / "stopwords.txt"
STOPWORDS_EXTRA = HW2 / "data" / "stopwords_extra.txt"
CUSTOM_DICT = HW2 / "data" / "custom_dict_final.txt"          # 作业 2 产出的定制词典
NEW_WORDS_CSV = HW2 / "data" / "new_words_evaluation.csv"      # 作业 2 挖出的新词
STOPWORDS_TASK = DATA_DIR / "stopwords_task.txt"              # 本作业(主题建模)的任务级停用词

# --------------------------- 超参 ---------------------------
K_RANGE = [4, 5, 6, 7, 8, 9, 10, 12]   # 候选主题数
LDA_PASSES = 12
LDA_ITERS = 400
RANDOM_STATE = 42
NO_BELOW = 4        # 词至少出现在 4 篇文档
NO_ABOVE = 0.45     # 词最多出现在 45% 文档（去功能性高频词）
TOPN_TERMS = 12     # 每个主题展示的词数

# 仅保留 中文/英文/数字/分隔点，长度 2~10
_CAND_RE = re.compile(r"^[一-鿿A-Za-z0-9·]{2,10}$")
_BAD_RE = re.compile(r"^(?:\d+|[A-Za-z]{1,2})$")


# =========================================================
# ① 数据载入
# =========================================================
@dataclass
class Doc:
    idx: int
    feed: str
    title: str
    text: str
    date: pd.Timestamp | None


def load_docs(csv_path: Path = CORPUS_CSV) -> list[Doc]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    docs: list[Doc] = []
    for i, row in df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        raw_date = str(row.get("published", "") or "").strip()
        date = None
        if raw_date:
            try:
                dt = parsedate_to_datetime(raw_date)
                # 保留发布地本地壁钟日期（RSS 多为 +0800），去掉 tz 信息
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                date = pd.Timestamp(dt)
            except Exception:
                date = None
        docs.append(
            Doc(
                idx=len(docs),
                feed=str(row.get("feed", "未知") or "未知").strip(),
                title=str(row.get("title", "") or "").strip(),
                text=text,
                date=date,
            )
        )
    return docs


def load_stopwords() -> set[str]:
    words: set[str] = set()
    for p in (STOPWORDS_PATH, STOPWORDS_EXTRA, STOPWORDS_TASK):
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                words.add(ln)
    return words


# =========================================================
# ② 分词（加载作业 2 定制词典）
# =========================================================
def build_tokenizer() -> jieba.Tokenizer:
    tk = jieba.Tokenizer()
    tk.initialize()
    if CUSTOM_DICT.exists():
        tk.load_userdict(str(CUSTOM_DICT))
    return tk


def good_token(w: str, stopwords: set[str]) -> bool:
    w = w.strip()
    if len(w) < 2 or w in stopwords:
        return False
    if not _CAND_RE.match(w):
        return False
    if _BAD_RE.match(w):
        return False
    return True


def tokenize(docs: list[Doc], tk: jieba.Tokenizer, stopwords: set[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for d in docs:
        out.append([w for w in tk.cut(d.text) if good_token(w, stopwords)])
    return out


# =========================================================
# ③ 词典 + BoW
# =========================================================
def build_corpus(tokens: list[list[str]]):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=20000)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    return dictionary, corpus


# =========================================================
# ④ 选择主题数 K：困惑度 + C_v 一致性
# =========================================================
def train_lda(corpus, dictionary, k: int, passes: int = LDA_PASSES) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=passes,
        iterations=LDA_ITERS,
        random_state=RANDOM_STATE,
        alpha="auto",
        eta="auto",
        eval_every=None,
    )


def scan_k(corpus, dictionary, tokens, k_range=K_RANGE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = train_lda(corpus, dictionary, k, passes=8)
        # gensim 的 log_perplexity 返回每词对数似然界(以 2 为底)，
        # 困惑度 = 2^(-bound)，越低越好
        bound = model.log_perplexity(corpus)
        perplexity = float(np.exp2(-bound))
        cm = CoherenceModel(
            model=model, texts=tokens, dictionary=dictionary,
            coherence="c_v", processes=1,
        )
        coh = float(cm.get_coherence())
        rows.append({"k": k, "coherence_cv": round(coh, 4),
                     "perplexity": round(perplexity, 2),
                     "log_perplexity": round(float(bound), 4)})
        print(f"[scan] K={k:2d}  C_v={coh:.4f}  perplexity={perplexity:.1f}")
    return pd.DataFrame(rows)


def choose_k(scan_df: pd.DataFrame) -> int:
    """主选 C_v 最大；若并列在 0.005 内取较小 K（奥卡姆剃刀，便于解释）。"""
    best = scan_df.sort_values("coherence_cv", ascending=False).iloc[0]
    best_coh = best["coherence_cv"]
    cand = scan_df[scan_df["coherence_cv"] >= best_coh - 0.005]
    return int(cand.sort_values("k").iloc[0]["k"])


# =========================================================
# ⑤ 最终模型 + θ / φ
# =========================================================
def doc_topic_matrix(model: LdaModel, corpus, k: int) -> np.ndarray:
    theta = np.zeros((len(corpus), k), dtype=np.float64)
    for i, bow in enumerate(corpus):
        for tid, p in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, tid] = p
    # 数值兜底归一化
    rs = theta.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return theta / rs


def auto_label(model: LdaModel, k: int, topn: int = 3) -> dict[str, str]:
    labels = {}
    for t in range(k):
        terms = [w for w, _ in model.show_topic(t, topn)]
        labels[str(t)] = "/".join(terms)
    return labels


def load_labels(k: int, model: LdaModel) -> dict[int, str]:
    """优先读取人工校准的 data/topic_labels.json；缺失则用前三词自动标签。"""
    path = DATA_DIR / "topic_labels.json"
    auto = auto_label(model, k)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if len(saved) == k:
                return {int(i): saved[str(i)] for i in range(k)}
        except Exception:
            pass
    # 写出自动标签草稿，供人工编辑
    path.write_text(json.dumps(auto, ensure_ascii=False, indent=2), encoding="utf-8")
    return {int(i): auto[str(i)] for i in range(k)}


# =========================================================
# ⑥ 导出数据
# =========================================================
def export_tables(model, corpus, dictionary, docs, theta, k, labels, scan_df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    scan_df.to_csv(DATA_DIR / "coherence_scan.csv", index=False, encoding="utf-8-sig")

    # 主题词
    rows = []
    for t in range(k):
        for rank, (w, wt) in enumerate(model.show_topic(t, TOPN_TERMS), 1):
            rows.append({"topic": t, "label": labels[t], "rank": rank,
                         "term": w, "weight": round(float(wt), 5)})
    pd.DataFrame(rows).to_csv(DATA_DIR / "topic_terms.csv", index=False, encoding="utf-8-sig")

    # 文档-主题
    drows = []
    for d, th in zip(docs, theta):
        dom = int(np.argmax(th))
        rec = {"doc_id": d.idx, "feed": d.feed,
               "date": d.date.strftime("%Y-%m-%d") if d.date is not None else "",
               "dominant_topic": dom, "dominant_label": labels[dom],
               "title": d.title}
        for t in range(k):
            rec[f"theta_{t}"] = round(float(th[t]), 4)
        drows.append(rec)
    pd.DataFrame(drows).to_csv(DATA_DIR / "doc_topic.csv", index=False, encoding="utf-8-sig")

    # 主题×来源（均值 θ）
    feeds = sorted({d.feed for d in docs})
    src = np.zeros((len(feeds), k))
    cnt = np.zeros(len(feeds))
    fidx = {f: i for i, f in enumerate(feeds)}
    for d, th in zip(docs, theta):
        src[fidx[d.feed]] += th
        cnt[fidx[d.feed]] += 1
    cnt[cnt == 0] = 1
    src = src / cnt[:, None]
    srows = []
    for f, i in fidx.items():
        for t in range(k):
            srows.append({"feed": f, "topic": t, "label": labels[t],
                          "mean_theta": round(float(src[i, t]), 4)})
    pd.DataFrame(srows).to_csv(DATA_DIR / "topic_source.csv", index=False, encoding="utf-8-sig")

    # 主题×时间（按自然周聚合，绝对主题质量 = Σθ）
    tdocs = [(d, th) for d, th in zip(docs, theta) if d.date is not None]
    trows = []
    if tdocs:
        for d, th in tdocs:
            wk = (d.date - pd.Timedelta(days=int(d.date.weekday()))).normalize()
            for t in range(k):
                trows.append({"week": wk.strftime("%Y-%m-%d"), "topic": t,
                              "label": labels[t], "mass": float(th[t])})
    tdf = pd.DataFrame(trows)
    if not tdf.empty:
        tdf = tdf.groupby(["week", "topic", "label"], as_index=False)["mass"].sum()
        tdf["mass"] = tdf["mass"].round(4)
    tdf.to_csv(DATA_DIR / "topic_over_time.csv", index=False, encoding="utf-8-sig")

    return feeds, src


def bridge_new_words(model, dictionary, k, labels) -> pd.DataFrame:
    """作业 2 挖出的新词，落到本作业哪个 LDA 主题（按 φ 中该词的主题归属）。"""
    if not NEW_WORDS_CSV.exists():
        return pd.DataFrame()
    nw = pd.read_csv(NEW_WORDS_CSV)
    topics = model.get_topics()  # (K, V) 行归一化的主题-词分布
    rows = []
    for _, r in nw.iterrows():
        w = str(r["word"]).strip()
        if w not in dictionary.token2id:
            continue
        col = dictionary.token2id[w]
        share = topics[:, col]
        if share.sum() <= 0:
            continue
        dom = int(np.argmax(share))
        rows.append({"word": w, "iter": int(r.get("iter", 0)),
                     "dominant_topic": dom, "dominant_label": labels[dom],
                     "topic_share": round(float(share[dom] / share.sum()), 4),
                     "hw2_score": float(r.get("score", 0.0))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(DATA_DIR / "newword_topic.csv", index=False, encoding="utf-8-sig")
    return df


# =========================================================
# 字体
# =========================================================
_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def chinese_font():
    from matplotlib import font_manager as fm
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return None


def font_path():
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


# K 个主题的稳定配色
def topic_colors(k: int):
    base = plt.get_cmap("tab10").colors + plt.get_cmap("tab20b").colors
    return [base[i % len(base)] for i in range(k)]


# =========================================================
# ⑦ 可视化
# =========================================================
def fig_scan(scan_df, best_k, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ks = scan_df["k"].tolist()
    ax1.plot(ks, scan_df["coherence_cv"], "o-", color="#2563eb", lw=2, label="主题一致性 C_v")
    ax1.set_xlabel("主题数 K", fontproperties=font, fontsize=12)
    ax1.set_ylabel("C_v 一致性（越高越好）", fontproperties=font, color="#2563eb", fontsize=12)
    ax1.tick_params(axis="y", colors="#2563eb")
    ax1.axvline(best_k, color="#16a34a", ls="--", lw=1.5)
    ax1.annotate(f"最优主题数 K = {best_k}",
                 xy=(best_k, scan_df.set_index("k").loc[best_k, "coherence_cv"]),
                 xytext=(-118, 6), textcoords="offset points", fontproperties=font,
                 color="#16a34a", fontsize=12, weight="bold")
    ax2 = ax1.twinx()
    ax2.plot(ks, scan_df["perplexity"], "s--", color="#dc2626", lw=1.6, label="困惑度 Perplexity")
    ax2.set_ylabel("困惑度（越低越好）", fontproperties=font, color="#dc2626", fontsize=12)
    ax2.tick_params(axis="y", colors="#dc2626")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, prop=font, loc="upper center")
    ax1.set_title("主题数 K 的模型选择：一致性 C_v 与困惑度", fontproperties=font, fontsize=14)
    ax1.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_topic_terms(model, k, labels, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    colors = topic_colors(k)
    ncol = 3
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for t in range(k):
        ax = axes[t]
        terms = model.show_topic(t, 10)
        words = [w for w, _ in terms][::-1]
        wts = [wt for _, wt in terms][::-1]
        ax.barh(range(len(words)), wts, color=colors[t])
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontproperties=font, fontsize=10)
        ax.set_title(f"主题 {t}：{labels[t]}", fontproperties=font, fontsize=11, color=colors[t])
        ax.tick_params(axis="x", labelsize=8)
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle("各主题的 Top-10 关键词及其权重（φ）", fontproperties=font, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_topic_timeline(out: Path, k, labels):
    """时间维度（诚实版）：
    (a) 全期按周文档量——揭示 RSS 近期偏置（样本集中于抓取窗口，本身是一个发现）；
    (b) 抓取窗口内每日主题构成（归一化占比）——即"最近一周科技舆论"的主题热度。
    避免在稀疏的早期周上夸大"长期演化"。
    """
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    dt = pd.read_csv(DATA_DIR / "doc_topic.csv")
    dt = dt[dt["date"].astype(str).str.len() > 0].copy()
    dt["date"] = pd.to_datetime(dt["date"], errors="coerce")
    dt = dt.dropna(subset=["date"])
    if dt.empty:
        print("无时间数据，跳过时间线图")
        return
    theta_cols = [f"theta_{t}" for t in range(k)]
    colors = topic_colors(k)

    # (a) 全期按周文档量
    dt["week"] = (dt["date"] - pd.to_timedelta(dt["date"].dt.weekday, unit="D")).dt.normalize()
    wk = dt.groupby("week").size()

    # (b) 抓取窗口：最末 6 天的每日主题占比
    last = dt["date"].max().normalize()
    win = dt[dt["date"] >= last - pd.Timedelta(days=5)].copy()
    win["day"] = win["date"].dt.normalize()
    days = sorted(win["day"].unique())
    share = np.zeros((k, len(days)))
    day_n = []
    for j, d in enumerate(days):
        sub = win[win["day"] == d]
        day_n.append(len(sub))
        mass = sub[theta_cols].sum().to_numpy(dtype=float)
        share[:, j] = mass / mass.sum() if mass.sum() > 0 else mass

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9.2), gridspec_kw={"height_ratios": [1, 1.6]})

    ax1.bar(range(len(wk)), wk.values, color="#64748b")
    ax1.set_xticks(range(len(wk)))
    ax1.set_xticklabels([d.strftime("%m-%d") for d in wk.index], fontsize=9)
    ax1.set_ylabel("周文档量", fontproperties=font, fontsize=11)
    ax1.set_title("(a) 全期按周文档量：RSS 仅保留近期条目，样本天然集中于抓取窗口（3 月底–4 月初）",
                  fontproperties=font, fontsize=12)
    for i, v in enumerate(wk.values):
        ax1.text(i, v + 1.5, str(int(v)), ha="center", fontsize=8)
    ax1.margins(x=0.01)

    x = np.arange(len(days))
    ax2.stackplot(x, share, colors=colors,
                  labels=[f"T{t} {labels[t]}" for t in range(k)], alpha=0.92)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{pd.Timestamp(d).strftime('%m-%d')}\n{n} 篇"
                         for d, n in zip(days, day_n)], fontproperties=font, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("当日主题占比", fontproperties=font, fontsize=11)
    ax2.set_xlim(0, len(days) - 1)
    ax2.set_title("(b) 抓取窗口内（2026-04-01 ~ 04-06）每日主题构成：本周科技舆论的主题热度",
                  fontproperties=font, fontsize=12)
    ax2.legend(prop=font, loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=5, fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("已保存 ->", out)


def fig_source_heatmap(feeds, src, k, labels, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(1.1 * k + 3.5, 0.55 * len(feeds) + 2.2))
    im = ax.imshow(src, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"T{t}\n{labels[t].split('/')[0]}" for t in range(k)],
                       fontproperties=font, fontsize=9)
    ax.set_yticks(range(len(feeds)))
    ax.set_yticklabels(feeds, fontproperties=font, fontsize=10)
    for i in range(len(feeds)):
        for t in range(k):
            v = src[i, t]
            ax.text(t, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v > src.max() * 0.55 else "#1f2937")
    ax.set_title("主题 × 来源媒体：各 RSS 源的平均主题占比 θ", fontproperties=font, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def _js_distance_matrix(P: np.ndarray) -> np.ndarray:
    """主题-词分布两两 Jensen-Shannon 距离（sqrt(JSD)，是合法度量）。"""
    K = P.shape[0]
    eps = 1e-12
    P = P + eps
    P = P / P.sum(axis=1, keepdims=True)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            m = 0.5 * (P[i] + P[j])
            kl_im = np.sum(P[i] * np.log2(P[i] / m))
            kl_jm = np.sum(P[j] * np.log2(P[j] / m))
            jsd = 0.5 * kl_im + 0.5 * kl_jm
            D[i, j] = D[j, i] = math.sqrt(max(jsd, 0.0))
    return D


def _classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][:dim], vecs[:, order][:, :dim]
    vals = np.clip(vals, 0, None)
    return vecs * np.sqrt(vals)


def fig_intertopic(model, theta, k, labels, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    P = model.get_topics()
    D = _js_distance_matrix(P)
    xy = _classical_mds(D, 2)
    prevalence = theta.mean(axis=0)
    prevalence = prevalence / prevalence.sum()
    colors = topic_colors(k)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    sizes = 600 + prevalence * 16000
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=[colors[t] for t in range(k)],
               alpha=0.55, edgecolors="white", linewidths=1.5, zorder=2)
    for t in range(k):
        ax.annotate(f"T{t} {labels[t].split('/')[0]}\n({prevalence[t]*100:.1f}%)",
                    (xy[t, 0], xy[t, 1]), ha="center", va="center",
                    fontproperties=font, fontsize=9, zorder=3)
    ax.axhline(0, color="#e5e7eb", lw=1, zorder=1)
    ax.axvline(0, color="#e5e7eb", lw=1, zorder=1)
    ax.set_xlabel("MDS 维度 1", fontproperties=font, fontsize=11)
    ax.set_ylabel("MDS 维度 2", fontproperties=font, fontsize=11)
    ax.set_title("主题间距离图（JS 散度 + 经典 MDS；气泡面积 ∝ 主题占比）",
                 fontproperties=font, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_newword_bridge(bridge_df, k, labels, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    if bridge_df is None or bridge_df.empty:
        print("无新词联动数据，跳过")
        return
    colors = topic_colors(k)
    counts = bridge_df["dominant_topic"].value_counts().to_dict()
    order = sorted(range(k), key=lambda t: -counts.get(t, 0))
    vals = [counts.get(t, 0) for t in order]
    examples = []
    for t in order:
        ws = bridge_df[bridge_df["dominant_topic"] == t].sort_values(
            "hw2_score", ascending=False)["word"].head(5).tolist()
        examples.append("、".join(ws))
    fig, ax = plt.subplots(figsize=(11, 0.7 * k + 2.6))
    y = range(len(order))
    ax.barh(list(y), vals, color=[colors[t] for t in order])
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"T{t} {labels[t].split('/')[0]}" for t in order], fontproperties=font)
    ax.invert_yaxis()
    for i, (t, v, ex) in enumerate(zip(order, vals, examples)):
        ax.text(v + 0.15, i, f"{v} 词｜{ex}", va="center", fontproperties=font, fontsize=9)
    ax.set_xlabel("作业 2 新词数（按主导主题归类）", fontproperties=font, fontsize=12)
    ax.set_title("作业 2 迭代挖出的新词在本作业 LDA 主题中的归属",
                 fontproperties=font, fontsize=13)
    ax.set_xlim(0, max(vals) + max(vals) * 0.9 + 2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_topic_wordclouds(model, k, labels, out: Path):
    from wordcloud import WordCloud
    fp = font_path()
    colors = topic_colors(k)
    ncol = 3
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.6 * nrow))
    axes = np.array(axes).reshape(-1)
    font = chinese_font()
    for t in range(k):
        freq = {w: float(wt) for w, wt in model.show_topic(t, 30)}
        rgb = tuple(int(c * 255) for c in colors[t][:3])

        def color_fn(*a, **kw):
            return rgb
        wc = WordCloud(font_path=fp, width=560, height=320, background_color="white",
                       prefer_horizontal=0.95, max_words=30,
                       color_func=color_fn).generate_from_frequencies(freq)
        axes[t].imshow(wc, interpolation="bilinear")
        axes[t].axis("off")
        axes[t].set_title(f"主题 {t}：{labels[t]}", fontproperties=font, fontsize=11, color=colors[t])
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle("各主题关键词词云（按 φ 权重）", fontproperties=font, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("已保存 ->", out)


# =========================================================
# 主流程
# =========================================================
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("① 载入语料 ...")
    docs = load_docs()
    stopwords = load_stopwords()
    dated = sum(1 for d in docs if d.date is not None)
    print(f"   文档={len(docs)}  含日期={dated}  停用词={len(stopwords)}")

    print("② 分词（加载作业 2 定制词典）...")
    tk = build_tokenizer()
    tokens = tokenize(docs, tk, stopwords)
    print(f"   总 token={sum(len(t) for t in tokens)}")

    print("③ 构建词典与 BoW ...")
    dictionary, corpus = build_corpus(tokens)
    print(f"   词表={len(dictionary)}  文档={len(corpus)}")

    print("④ 扫描主题数 K ...")
    scan_df = scan_k(corpus, dictionary, tokens)
    best_k = choose_k(scan_df)
    print(f"   选定 K={best_k}")

    print(f"⑤ 训练最终 LDA (K={best_k}) ...")
    model = train_lda(corpus, dictionary, best_k)
    model.save(str(DATA_DIR / "lda_final.model"))
    theta = doc_topic_matrix(model, corpus, best_k)
    labels = load_labels(best_k, model)
    print("   主题标签:", labels)

    print("⑥ 导出数据 ...")
    feeds, src = export_tables(model, corpus, dictionary, docs, theta, best_k, labels, scan_df)
    bridge = bridge_new_words(model, dictionary, best_k, labels)

    print("⑦ 出图 ...")
    fig_scan(scan_df, best_k, FIGURES_DIR / "k_selection.png")
    fig_topic_terms(model, best_k, labels, FIGURES_DIR / "topic_terms.png")
    fig_topic_timeline(FIGURES_DIR / "topic_timeline.png", best_k, labels)
    fig_source_heatmap(feeds, src, best_k, labels, FIGURES_DIR / "topic_source_heatmap.png")
    fig_intertopic(model, theta, best_k, labels, FIGURES_DIR / "intertopic_map.png")
    fig_newword_bridge(bridge, best_k, labels, FIGURES_DIR / "newword_bridge.png")
    fig_topic_wordclouds(model, best_k, labels, FIGURES_DIR / "topic_wordclouds.png")

    print("\n=== 完成 ===")
    print(f"主题数 K={best_k}；最佳 C_v={scan_df['coherence_cv'].max()}")
    print("主题标签文件：data/topic_labels.json（可人工校准后重跑出图）")


if __name__ == "__main__":
    main()
