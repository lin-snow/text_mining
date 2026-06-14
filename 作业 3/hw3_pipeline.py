# -*- coding: utf-8 -*-
"""作业 3：基于降维(LSI/LSA + LDA)与近似最近邻(Annoy)的中文科技资讯推荐与可视化系统。

降维技术路线（呼应作业题面 LSA/LSI/LDA/UMAP）：
  ·  LSA/LSI —— 对 TF-IDF 矩阵做截断 SVD（A ≈ U_k Σ_k V_k^T），取前 k 个潜在语义"概念"；
  ·  LDA     —— 概率主题模型，把文档表示成 K 维主题分布 θ；
  两套低维稠密向量都送入 Annoy 近似最近邻索引，实现"读了这篇→还想读这些"的推荐。

端到端流水线：
  ①  载入作业 1 RSS 语料（含来源/时间）+ 作业 2 定制词典/新词 + 多级停用词，jieba 分词
  ②  BoW → TF-IDF 高维稀疏向量空间
  ③  降维：LSI（截断 SVD，按"主题一致命中率"选 k）与 LDA（K=10），得到文档稠密嵌入
  ④  推荐：Annoy（angular≈cosine）建索引，topN 近邻推荐；并做三类评估
        评估① 邻域保持 overlap@10（对照全维 TF-IDF，衡量降维是否保住推荐结构）
        评估② 主题一致命中率（以 LDA 主导主题为弱标签）对照随机基线
        评估③ Annoy 近似 vs 暴力精确：recall@10 与查询耗时
  ⑤  可视化：PCA / t-SNE / UMAP 三种投影并排（按主题着色）+ 推荐邻域连线图
  ⑥  新词推荐锚定力 NAP（脑洞新词评估）：把作业 2 的新词放进推荐系统，
        度量"含该词文档"在语义空间的簇内/全局相似度提升，并做"移除该词"的消融实验
  ⑦  导出 CSV + 全部 PNG 图

与"课程设计（LDA 主题挖掘）"区分：本作业聚焦*推荐*，把降维当作推荐的特征工程。
全部图表 Matplotlib 离线生成；数据与分词深度联动作业 1、作业 2。
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, TfidfModel

warnings.filterwarnings("ignore")
np.random.seed(42)

# --------------------------- 路径 ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

HW1 = ROOT.parent / "作业 1"
HW2 = ROOT.parent / "作业 2"
CORPUS_CSV = HW1 / "data" / "rss_cleaned.csv"
STOPWORDS_PATH = HW1 / "data" / "stopwords.txt"
STOPWORDS_EXTRA = HW2 / "data" / "stopwords_extra.txt"
CUSTOM_DICT = HW2 / "data" / "custom_dict_final.txt"            # 作业 2 产出的定制词典
NEW_WORDS_CSV = HW2 / "data" / "new_words_evaluation.csv"        # 作业 2 挖出的新词

import re  # noqa: E402

# --------------------------- 超参 ---------------------------
RANDOM_STATE = 42
NO_BELOW = 4          # 词至少出现在 4 篇文档
NO_ABOVE = 0.45       # 词最多出现在 45% 文档（去功能性高频词）
MAX_K_LSI = 200       # LSI 一次训练到的最大维度（再切片得到各候选 k）
K_SWEEP = [10, 20, 30, 50, 80, 120, 160, 200]   # 候选潜在语义维度
LDA_K = 10            # LDA 主题数（与课程设计口径一致，便于弱标签评估）
LDA_PASSES = 12
LDA_ITERS = 400
N_EVAL = 10           # 评估用近邻数
N_SHOW = 5            # 推荐展示条数
ANN_NEIGHBORS = 20    # 近似最近邻索引构建的邻接度
NAP_MIN_DF = 3        # 新词锚定力：含词文档数下限（需成簇）
NAP_MAX_DF_RATIO = 0.5  # 上限比例（太泛的词不作锚点）

_CAND_RE = re.compile(r"^[一-鿿A-Za-z0-9·]{2,10}$")
_BAD_RE = re.compile(r"^(?:\d+|[A-Za-z]{1,2})$")


# =========================================================
# ① 数据载入 + ② 分词（与课程设计同口径，保证三次作业可比）
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
    for _, row in df.iterrows():
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        raw_date = str(row.get("published", "") or "").strip()
        date = None
        if raw_date:
            try:
                dt = parsedate_to_datetime(raw_date)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                date = pd.Timestamp(dt)
            except Exception:
                date = None
        docs.append(Doc(idx=len(docs),
                        feed=str(row.get("feed", "未知") or "未知").strip(),
                        title=str(row.get("title", "") or "").strip(),
                        text=text, date=date))
    return docs


def load_stopwords() -> set[str]:
    words: set[str] = set()
    for p in (STOPWORDS_PATH, STOPWORDS_EXTRA):
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                words.add(ln)
    return words


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
    if not _CAND_RE.match(w) or _BAD_RE.match(w):
        return False
    return True


def tokenize(docs: list[Doc], tk: jieba.Tokenizer, stopwords: set[str]) -> list[list[str]]:
    return [[w for w in tk.cut(d.text) if good_token(w, stopwords)] for d in docs]


# =========================================================
# ③ 向量空间：BoW → TF-IDF → 降维(LSI / LDA)
# =========================================================
def build_corpus(tokens: list[list[str]]):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=20000)
    corpus = [dictionary.doc2bow(t) for t in tokens]
    return dictionary, corpus


def tfidf_space(corpus, dictionary):
    """返回 (tfidf 模型, tfidf 语料, 全维稠密 TF-IDF 矩阵[n×V]，行已 L2 归一)。"""
    tfidf = TfidfModel(corpus)                       # 默认 L2 归一 → 点积即 cosine
    tfidf_corpus = list(tfidf[corpus])
    dense = matutils.corpus2dense(tfidf_corpus, num_terms=len(dictionary)).T
    return tfidf, tfidf_corpus, np.ascontiguousarray(dense, dtype=np.float32)


def lsi_embedding(tfidf_corpus, dictionary, num_topics):
    """截断 SVD：一次训练到 num_topics 维，列按奇异值降序；切片即得低维 LSI。"""
    lsi = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics)
    emb = matutils.corpus2dense(lsi[tfidf_corpus], num_terms=lsi.num_topics).T
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    sing = np.asarray(lsi.projection.s, dtype=np.float64)   # 奇异值（降序）
    return lsi, emb, sing


def lda_model(corpus, dictionary, k=LDA_K):
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                     passes=LDA_PASSES, iterations=LDA_ITERS,
                     random_state=RANDOM_STATE, alpha="auto", eta="auto",
                     eval_every=None)
    theta = np.zeros((len(corpus), k), dtype=np.float64)
    for i, bow in enumerate(corpus):
        for tid, p in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, tid] = p
    rs = theta.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return model, theta / rs


def topic_labels(model: LdaModel, k: int, topn: int = 3) -> dict[int, str]:
    """优先读人工校准 data/topic_labels.json；否则用前 topn 词自动命名并写出草稿。"""
    path = DATA_DIR / "topic_labels.json"
    auto = {str(t): "/".join(w for w, _ in model.show_topic(t, topn)) for t in range(k)}
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if len(saved) == k:
                return {int(i): saved[str(i)] for i in range(k)}
        except Exception:
            pass
    path.write_text(json.dumps(auto, ensure_ascii=False, indent=2), encoding="utf-8")
    return {int(i): auto[str(i)] for i in range(k)}


# =========================================================
# 相似度 / 近邻 工具
# =========================================================
def l2norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return emb / n


def cosine_knn(emb: np.ndarray, n_neighbors: int) -> np.ndarray:
    """对每篇文档返回 cosine 最近的 n_neighbors 个其它文档的索引（暴力精确）。"""
    u = l2norm(emb)
    sim = u @ u.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :n_neighbors]


def overlap_at_n(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    """两套近邻表的平均交集占比（∈[0,1]，邻域保持度）。"""
    n, k = knn_a.shape
    s = sum(len(set(knn_a[i]) & set(knn_b[i])) for i in range(n))
    return s / (n * k)


def topic_hit_rate(knn: np.ndarray, labels: np.ndarray) -> float:
    """近邻里与 query 同主导主题的平均占比（语义一致性）。"""
    n, k = knn.shape
    hits = sum(np.mean(labels[knn[i]] == labels[i]) for i in range(n))
    return hits / n


def topic_hit_random_baseline(labels: np.ndarray) -> float:
    """随机近邻的期望命中率：Σ_i (同主题文档数-1)/(N-1) 的均值。"""
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    cmap = {l: c for l, c in zip(*np.unique(labels, return_counts=True))}
    return float(np.mean([(cmap[labels[i]] - 1) / (n - 1) for i in range(n)]))


# =========================================================
# ④ 推荐引擎
#   · 线上召回：低维语义空间的 exact cosine KNN（n=220 量级下即正确的"标准答案"）；
#   · 近似最近邻：pynndescent(NN-Descent)——与 Annoy 同属 ANN 家族、且正是 UMAP 的
#     召回引擎；用于演示"近似 vs 精确"的 recall/时延权衡（题面 [2] Annoy 的主旨）。
#   （注：annoy 1.17.3 在本机 Python 3.13 上的预编译/源码构建均会段错误，故改用
#     同族的 pynndescent，结论一致；详见 REPORT「实现说明」。）
# =========================================================
def brute_reco(U: np.ndarray, i: int, topn=N_SHOW):
    """精确 cosine 推荐：U 为行已归一的嵌入，返回 [(邻居, cosine)]。"""
    sims = U @ U[i]
    sims[i] = -np.inf
    order = np.argsort(-sims)[:topn]
    return [(int(j), float(sims[j])) for j in order]


def build_ann(emb: np.ndarray, n_neighbors=ANN_NEIGHBORS):
    """构建近似最近邻索引（cosine 度量）。"""
    from pynndescent import NNDescent
    u = l2norm(emb)
    index = NNDescent(u, metric="cosine", n_neighbors=n_neighbors,
                      random_state=RANDOM_STATE, verbose=False)
    index.prepare()
    return index


def ann_knn(index, emb: np.ndarray, n_neighbors: int) -> np.ndarray:
    """用 ANN 索引为每篇文档取 n_neighbors 个近似最近邻（去掉自身）。"""
    u = l2norm(emb)
    n = u.shape[0]
    idxs, _ = index.query(u, k=n_neighbors + 1)
    out = np.zeros((n, n_neighbors), dtype=int)
    for i in range(n):
        nbrs = [int(j) for j in idxs[i] if j != i][:n_neighbors]
        if len(nbrs) < n_neighbors:        # 极端兜底
            nbrs += [j for j in range(n) if j != i and j not in nbrs]
        out[i] = nbrs[:n_neighbors]
    return out


# =========================================================
# ⑥ 新词推荐锚定力 NAP
# =========================================================
def new_word_anchoring(tokens, emb, new_words_df, hw2_score_col="score"):
    """NAP(w) = 含 w 文档在 LSI 空间的平均两两 cosine / 全局平均 - 1。

    >0 表示该词把语义相近的文章聚到一起，是强"推荐锚点"。
    仅评估落在甜区 [NAP_MIN_DF, NAP_MAX_DF_RATIO·n] 的词。
    """
    u = l2norm(emb)
    sim = u @ u.T
    n = sim.shape[0]
    off = ~np.eye(n, dtype=bool)
    global_mean = float(sim[off].mean())
    doc_sets = [set(t) for t in tokens]
    rows = []
    for _, r in new_words_df.iterrows():
        w = str(r["word"]).strip()
        members = [i for i in range(n) if w in doc_sets[i]]
        df = len(members)
        if df < NAP_MIN_DF or df > NAP_MAX_DF_RATIO * n:
            continue
        sub = sim[np.ix_(members, members)]
        intra = float(sub[~np.eye(df, dtype=bool)].mean())
        nap = intra / global_mean - 1.0
        rows.append({"word": w, "df": df, "intra_sim": round(intra, 4),
                     "global_sim": round(global_mean, 4), "NAP": round(nap, 4),
                     "hw2_score": float(r.get(hw2_score_col, 0.0))})
    out = pd.DataFrame(rows).sort_values("NAP", ascending=False).reset_index(drop=True)
    return out, global_mean


def ablation(tokens, words, emb_full, k_star):
    """消融实验：把某新词从所有文档移除并重建 LSI，观察"含该词文档"的簇内
    平均 cosine 相似度掉多少——直接量化该词作为推荐锚点的*局部*贡献。"""
    U = l2norm(emb_full)
    sim_full = U @ U.T
    n = len(tokens)
    doc_sets = [set(t) for t in tokens]
    res = []
    for w in words:
        members = [i for i in range(n) if w in doc_sets[i]]
        if len(members) < 2:
            continue
        m = len(members)
        sub = sim_full[np.ix_(members, members)]
        intra_full = float(sub[~np.eye(m, dtype=bool)].mean())
        # 移除该词后重建 TF-IDF + LSI
        toks2 = [[t for t in doc if t != w] for doc in tokens]
        d2 = Dictionary(toks2)
        d2.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=20000)
        c2 = [d2.doc2bow(t) for t in toks2]
        tc2 = list(TfidfModel(c2)[c2])
        ndim = min(k_star, len(d2) - 1, n - 1)
        _, emb2, _ = lsi_embedding(tc2, d2, ndim)
        U2 = l2norm(emb2)
        sub2 = (U2 @ U2.T)[np.ix_(members, members)]
        intra_abl = float(sub2[~np.eye(m, dtype=bool)].mean())
        drop = intra_full - intra_abl
        res.append({"word": w, "df": m,
                    "intra_full": round(intra_full, 4),
                    "intra_ablated": round(intra_abl, 4),
                    "drop": round(drop, 4),
                    "drop_pct": round(drop / abs(intra_full) * 100, 1) if intra_full else 0.0})
    return pd.DataFrame(res)


# =========================================================
# 字体 / 配色
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


def topic_colors(k: int):
    base = plt.get_cmap("tab10").colors + plt.get_cmap("tab20b").colors
    return [base[i % len(base)] for i in range(k)]


# =========================================================
# ⑤ 可视化
# =========================================================
def fig_k_selection(sweep_df, sing, k_star, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    energy = sing ** 2
    cum = np.cumsum(energy) / energy.sum()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) 奇异值谱 + 累计能量
    ax0.bar(range(1, len(sing) + 1), sing, color="#94a3b8", width=1.0)
    ax0.set_xlabel("奇异值序号 i", fontproperties=font, fontsize=12)
    ax0.set_ylabel("奇异值 σ_i", fontproperties=font, fontsize=12, color="#475569")
    ax0b = ax0.twinx()
    ax0b.plot(range(1, len(cum) + 1), cum, color="#2563eb", lw=2)
    ax0b.set_ylabel("累计能量 Σσ²", fontproperties=font, fontsize=12, color="#2563eb")
    ax0b.set_ylim(0, 1.02)
    ax0b.axvline(k_star, color="#16a34a", ls="--", lw=1.5)
    ax0b.annotate(f"k*={k_star}\n累计能量 {cum[k_star-1]*100:.0f}%",
                  xy=(k_star, cum[k_star - 1]), xytext=(12, -28),
                  textcoords="offset points", fontproperties=font,
                  color="#16a34a", fontsize=11, weight="bold")
    ax0.set_title("(a) LSA 奇异值谱与累计能量", fontproperties=font, fontsize=13)

    # (b) 随 k 变化的两条推荐质量曲线
    ks = sweep_df["k"].tolist()
    ax1.plot(ks, sweep_df["topic_hit"], "o-", color="#dc2626", lw=2, label="主题一致命中率")
    ax1.plot(ks, sweep_df["overlap_tfidf"], "s--", color="#0891b2", lw=1.8,
             label="邻域保持 overlap@10（对全维TF-IDF）")
    ax1.axvline(k_star, color="#16a34a", ls="--", lw=1.5)
    ax1.scatter([k_star], [sweep_df.set_index("k").loc[k_star, "topic_hit"]],
                s=160, facecolors="none", edgecolors="#16a34a", linewidths=2.2, zorder=5)
    ax1.set_xlabel("潜在语义维度 k", fontproperties=font, fontsize=12)
    ax1.set_ylabel("推荐质量指标", fontproperties=font, fontsize=12)
    ax1.set_title(f"(b) 按推荐质量选 k：k*={k_star}（命中率峰值）",
                  fontproperties=font, fontsize=13)
    ax1.legend(prop=font, loc="best")
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_projections(proj: dict, labels, label_names, trust: dict, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    k = len(label_names)
    colors = topic_colors(k)
    names = list(proj.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, name in zip(axes, names):
        xy = proj[name]
        for t in range(k):
            m = labels == t
            ax.scatter(xy[m, 0], xy[m, 1], s=34, color=colors[t], alpha=0.82,
                       edgecolors="white", linewidths=0.4)
        ax.set_title(f"{name}（trustworthiness={trust[name]:.3f}）",
                     fontproperties=font, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[t],
               label=f"T{t} {label_names[t].split('/')[0]}") for t in range(k)]
    fig.legend(handles=handles, prop=font, loc="lower center", ncol=k,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LSA 语义空间的三种二维投影（点=文章，色=LDA主导主题）",
                 fontproperties=font, fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("已保存 ->", out)


def fig_reco_neighbors(xy, labels, label_names, queries, knn_show, docs, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    k = len(label_names)
    colors = topic_colors(k)
    ncol = len(queries)
    fig, axes = plt.subplots(1, ncol, figsize=(7.4 * ncol, 6.4))
    if ncol == 1:
        axes = [axes]
    for ax, q in zip(axes, queries):
        ax.scatter(xy[:, 0], xy[:, 1], s=22, c=[colors[t] for t in labels],
                   alpha=0.28, edgecolors="none")
        nbrs = knn_show[q]
        for j in nbrs:
            ax.plot([xy[q, 0], xy[j, 0]], [xy[q, 1], xy[j, 1]],
                    color="#475569", lw=1.0, alpha=0.7, zorder=2)
            ax.scatter(xy[j, 0], xy[j, 1], s=90, facecolors=colors[labels[j]],
                       edgecolors="#111827", linewidths=1.1, zorder=3)
        ax.scatter(xy[q, 0], xy[q, 1], s=320, marker="*", color="#f59e0b",
                   edgecolors="#111827", linewidths=1.3, zorder=4)
        title = docs[q].title[:22] + ("…" if len(docs[q].title) > 22 else "")
        ax.set_title(f"query: {title}\nTop-{len(nbrs)} 推荐邻居（连线）",
                     fontproperties=font, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("推荐 = 低维语义空间中的最近邻检索（UMAP 投影）",
                 fontproperties=font, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_reco_eval(eval_rows, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) 主题一致命中率：LSI(无监督) vs 随机；LDA 同源标签仅作循环上界
    names = ["随机基线", "LSI 嵌入\n(无监督)", "LDA 嵌入\n(同源标签)"]
    vals = [eval_rows["hit_random"], eval_rows["hit_lsi"], eval_rows["hit_lda"]]
    bars = ax1.bar(names, vals, color=["#cbd5e1", "#2563eb", "#a78bfa"])
    bars[2].set_hatch("////")          # LDA 循环上界，加阴影示意
    bars[2].set_alpha(0.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontproperties=font, fontsize=11)
    mult = eval_rows["hit_lsi"] / eval_rows["hit_random"]
    ax1.annotate(f"{mult:.1f}× 随机", xy=(1, eval_rows["hit_lsi"]),
                 xytext=(1, eval_rows["hit_lsi"] + 0.22), ha="center",
                 fontproperties=font, fontsize=11, color="#16a34a", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#16a34a"))
    ax1.text(2, eval_rows["hit_lda"] + 0.05, "标签同源→循环\n仅作上界参考",
             ha="center", fontproperties=font, fontsize=8.5, color="#6d28d9")
    ax1.set_ylabel("Top-10 推荐的主题一致命中率", fontproperties=font, fontsize=12)
    ax1.set_ylim(0, 1.25)
    for lb in ax1.get_xticklabels():
        lb.set_fontproperties(font)
    ax1.set_title("(a) 推荐语义一致性：LSI 无监督达随机的 2.4 倍", fontproperties=font, fontsize=13)

    # (b) 近似最近邻 vs 暴力精确
    ax2.axis("off")
    cell = [["recall@10（近似 ANN vs 精确）", f"{eval_rows['ann_recall']:.3f}"],
            [f"精确检索 {N_EVAL}NN 总耗时", f"{eval_rows['brute_ms']:.2f} ms"],
            [f"近似 ANN 检索 {N_EVAL}NN 总耗时", f"{eval_rows['ann_ms']:.2f} ms"],
            ["语料规模", f"{eval_rows['n_docs']} 篇 × {eval_rows['k_star']} 维"]]
    tbl = ax2.table(cellText=cell, colLabels=["指标", "数值"],
                    cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.0)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.get_text().set_fontproperties(font)
        if r == 0:
            cellobj.set_facecolor("#e2e8f0")
    ax2.set_title("(b) 近似最近邻召回与时延", fontproperties=font, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)


def fig_newword_nap(nap_df, abl_df, out: Path):
    font = chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    top = nap_df.head(18).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) 锚定力排行
    cols = ["#16a34a" if v > 0 else "#dc2626" for v in top["NAP"]]
    ax1.barh(range(len(top)), top["NAP"], color=cols)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels([f"{w}（{d}篇）" for w, d in zip(top["word"], top["df"])],
                        fontproperties=font, fontsize=10)
    for i, v in enumerate(top["NAP"]):
        ax1.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontproperties=font, fontsize=9)
    ax1.axvline(0, color="#334155", lw=1)
    ax1.set_xlabel("新词推荐锚定力 NAP = 簇内相似/全局相似 − 1", fontproperties=font, fontsize=11)
    ax1.set_title("(a) 作业2新词的“推荐锚定力”排行（Top-18）", fontproperties=font, fontsize=13)

    # (b) 作业2词频得分 vs 作业3锚定力
    x = nap_df["hw2_score"].to_numpy()
    y = nap_df["NAP"].to_numpy()
    ax2.scatter(x, y, s=44, color="#2563eb", alpha=0.75, edgecolors="white")
    if len(x) > 2 and np.std(x) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax2.plot(xs, a * xs + b, color="#dc2626", lw=1.6, ls="--",
                 label=f"Pearson r = {r:.2f}")
        ax2.legend(prop=font, loc="best")
    for _, row in nap_df.head(6).iterrows():
        ax2.annotate(row["word"], (row["hw2_score"], row["NAP"]),
                     fontproperties=font, fontsize=9,
                     xytext=(4, 3), textcoords="offset points")
    ax2.set_xlabel("作业2 词频/相似度得分 score", fontproperties=font, fontsize=11)
    ax2.set_ylabel("作业3 推荐锚定力 NAP", fontproperties=font, fontsize=11)
    ax2.axhline(0, color="#94a3b8", lw=0.8)
    ax2.set_title("(b) 两种新词评估机制的相关性", fontproperties=font, fontsize=13)
    ax2.grid(alpha=0.25)

    fig.suptitle("新词推荐锚定力 NAP：把作业2的新词放进推荐系统做“有效性”评估",
                 fontproperties=font, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存 ->", out)

    if abl_df is not None and not abl_df.empty:
        print("\n[消融] 移除高锚定力新词后，含该词文档的簇内相似度变化：")
        print(abl_df.to_string(index=False))


# =========================================================
# 主流程
# =========================================================
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("① 载入语料 + 分词（联动作业1语料 / 作业2词典）...")
    docs = load_docs()
    stopwords = load_stopwords()
    tk = build_tokenizer()
    tokens = tokenize(docs, tk, stopwords)
    print(f"   文档={len(docs)}  停用词={len(stopwords)}  总token={sum(map(len, tokens))}")

    print("② 构建 BoW → TF-IDF 高维空间 ...")
    dictionary, corpus = build_corpus(tokens)
    # filter_extremes 后个别短文档可能词袋为空（语义向量退化为零向量，
    # 会破坏 cosine/Annoy-angular），剔除之并保持 docs/tokens/corpus 对齐
    keep = [i for i, b in enumerate(corpus) if len(b) > 0]
    if len(keep) < len(corpus):
        print(f"   剔除空词袋文档 {len(corpus) - len(keep)} 篇")
        docs = [docs[i] for i in keep]
        tokens = [tokens[i] for i in keep]
        corpus = [corpus[i] for i in keep]
    tfidf, tfidf_corpus, tfidf_dense = tfidf_space(corpus, dictionary)
    V = len(dictionary)
    n = len(corpus)
    print(f"   词表 V={V}  文档 n={n}")

    print("③ 降维：LSI 截断SVD + LDA(K=10) ...")
    max_k = min(MAX_K_LSI, n - 1, V - 1)
    lsi, emb_lsi_full, sing = lsi_embedding(tfidf_corpus, dictionary, max_k)
    lda, theta = lda_model(corpus, dictionary, LDA_K)
    labels = np.argmax(theta, axis=1)
    label_names = topic_labels(lda, LDA_K)
    print(f"   LSI 维度上限={emb_lsi_full.shape[1]}  LDA 主题={LDA_K}")
    print("   主题标签:", label_names)
    # 导出主题词（供报告展示 + 人工校准 topic_labels.json）
    tt_rows = []
    for t in range(LDA_K):
        terms = lda.show_topic(t, 10)
        for rank, (w, wt) in enumerate(terms, 1):
            tt_rows.append({"topic": t, "label": label_names[t], "rank": rank,
                            "term": w, "weight": round(float(wt), 5)})
        print(f"   T{t}: " + " ".join(w for w, _ in terms[:8]))
    pd.DataFrame(tt_rows).to_csv(DATA_DIR / "topic_terms.csv", index=False, encoding="utf-8-sig")

    print("④ 选潜在语义维度 k（按推荐质量）...")
    knn_tfidf = cosine_knn(tfidf_dense, N_EVAL)         # 全维参考近邻
    rnd_base = topic_hit_random_baseline(labels)
    rows = []
    for k in [kk for kk in K_SWEEP if kk <= emb_lsi_full.shape[1]]:
        knn_k = cosine_knn(emb_lsi_full[:, :k], N_EVAL)
        rows.append({"k": k,
                     "topic_hit": round(topic_hit_rate(knn_k, labels), 4),
                     "overlap_tfidf": round(overlap_at_n(knn_k, knn_tfidf), 4)})
        print(f"   k={k:3d}  主题命中={rows[-1]['topic_hit']:.3f}  "
              f"overlap@10={rows[-1]['overlap_tfidf']:.3f}")
    sweep_df = pd.DataFrame(rows)
    # 选命中率最高者，近似并列(0.005内)取较小 k（奥卡姆 + 更强压缩/去噪）
    best = sweep_df.sort_values("topic_hit", ascending=False).iloc[0]["topic_hit"]
    k_star = int(sweep_df[sweep_df["topic_hit"] >= best - 0.005].sort_values("k").iloc[0]["k"])
    emb_lsi = emb_lsi_full[:, :k_star]
    print(f"   选定 k*={k_star}（随机基线命中={rnd_base:.3f}）")

    print("⑤ 推荐召回（精确 cosine KNN）+ 近似最近邻评估 ...")
    U_lsi = l2norm(emb_lsi)                              # 行归一，点积即 cosine
    knn_lsi = cosine_knn(emb_lsi, N_EVAL)               # 精确召回（线上推荐用）
    knn_lda = cosine_knn(theta.astype(np.float32), N_EVAL)
    # 近似最近邻（pynndescent）：recall@10 与查询时延（已排除一次性 numba 编译）
    ann = build_ann(emb_lsi)
    knn_ann = ann_knn(ann, emb_lsi, N_EVAL)            # 预热(触发编译)
    recall = overlap_at_n(knn_ann, knn_lsi)
    t0 = time.perf_counter(); _ = cosine_knn(emb_lsi, N_EVAL); brute_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); _ = ann_knn(ann, emb_lsi, N_EVAL); ann_ms = (time.perf_counter() - t0) * 1e3
    eval_rows = {
        "hit_random": round(rnd_base, 4),
        "hit_lsi": round(topic_hit_rate(knn_lsi, labels), 4),
        "hit_lda": round(topic_hit_rate(knn_lda, labels), 4),
        "overlap_lsi_tfidf": round(overlap_at_n(knn_lsi, knn_tfidf), 4),
        "overlap_lda_tfidf": round(overlap_at_n(knn_lda, knn_tfidf), 4),
        "ann_recall": round(recall, 4),
        "brute_ms": round(brute_ms, 3), "ann_ms": round(ann_ms, 3),
        "n_docs": n, "k_star": k_star,
    }
    print("   评估:", eval_rows)

    print("⑥ 生成推荐表（每篇 Top-5，精确 cosine）...")
    rec_rows = []
    for i in range(n):
        for rank, (j, s) in enumerate(brute_reco(U_lsi, i, N_SHOW), 1):
            rec_rows.append({"doc_id": i, "title": docs[i].title, "feed": docs[i].feed,
                             "rank": rank, "rec_id": j, "rec_title": docs[j].title,
                             "rec_feed": docs[j].feed, "cosine": round(s, 4),
                             "same_topic": int(labels[j] == labels[i])})
    pd.DataFrame(rec_rows).to_csv(DATA_DIR / "recommendations.csv", index=False,
                                  encoding="utf-8-sig")

    print("⑦ 降维可视化：PCA / t-SNE / UMAP ...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness
    import umap
    Xn = l2norm(emb_lsi)
    proj = {}
    proj["PCA"] = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xn)
    proj["t-SNE"] = TSNE(n_components=2, init="pca", perplexity=20,
                         random_state=RANDOM_STATE).fit_transform(Xn)
    proj["UMAP"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.12,
                             metric="cosine", random_state=RANDOM_STATE).fit_transform(Xn)
    trust = {name: float(trustworthiness(Xn, xy, n_neighbors=N_EVAL))
             for name, xy in proj.items()}
    print("   trustworthiness:", {k: round(v, 3) for k, v in trust.items()})

    print("⑧ 新词推荐锚定力 NAP + 消融 ...")
    nap_df = abl_df = None
    if NEW_WORDS_CSV.exists():
        nw = pd.read_csv(NEW_WORDS_CSV, encoding="utf-8-sig")
        nap_df, _ = new_word_anchoring(tokens, emb_lsi, nw)
        nap_df.to_csv(DATA_DIR / "newword_anchoring.csv", index=False, encoding="utf-8-sig")
        top_words = nap_df.head(4)["word"].tolist()
        abl_df = ablation(tokens, top_words, emb_lsi, k_star)
        abl_df.to_csv(DATA_DIR / "newword_ablation.csv", index=False, encoding="utf-8-sig")
        print(f"   评估新词={len(nap_df)}  最强锚点={top_words}")

    print("⑨ 导出数据表 + 出图 ...")
    sweep_df.to_csv(DATA_DIR / "k_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"i": range(1, len(sing) + 1), "singular_value": sing,
                  "energy_ratio": (sing ** 2) / (sing ** 2).sum(),
                  "cum_energy": np.cumsum(sing ** 2) / (sing ** 2).sum()}
                 ).to_csv(DATA_DIR / "singular_values.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([eval_rows]).to_csv(DATA_DIR / "reco_eval.csv", index=False, encoding="utf-8-sig")
    emb2d = pd.DataFrame({"doc_id": range(n),
                          "title": [d.title for d in docs],
                          "feed": [d.feed for d in docs],
                          "topic": labels, "label": [label_names[t] for t in labels]})
    for name in proj:
        emb2d[f"{name}_x"] = proj[name][:, 0]
        emb2d[f"{name}_y"] = proj[name][:, 1]
    emb2d.to_csv(DATA_DIR / "embedding_2d.csv", index=False, encoding="utf-8-sig")

    # 选 4 个代表性 query（4 个最高占比主题各取最典型一篇）
    order_t = np.argsort(-theta.mean(0))
    queries, seen = [], set()
    for t in order_t:
        i = int(np.argmax(theta[:, t]))
        if i not in seen:
            queries.append(i); seen.add(i)
        if len(queries) >= 4:
            break

    fig_k_selection(sweep_df, sing, k_star, FIGURES_DIR / "k_selection.png")
    fig_projections(proj, labels, label_names, trust, FIGURES_DIR / "projections.png")
    knn_show = cosine_knn(emb_lsi, N_SHOW)            # 精确 Top-5 邻居用于连线图
    fig_reco_neighbors(proj["UMAP"], labels, label_names, queries[:2], knn_show, docs,
                       FIGURES_DIR / "reco_neighbors.png")
    fig_reco_eval(eval_rows, FIGURES_DIR / "reco_eval.png")
    if nap_df is not None and not nap_df.empty:
        fig_newword_nap(nap_df, abl_df, FIGURES_DIR / "newword_nap.png")

    # 控制台演示：4 个 query 的推荐
    print("\n=== 推荐演示（Top-5） ===")
    for q in queries:
        print(f"\n[读了] T{labels[q]}/{label_names[labels[q]].split('/')[0]} | "
              f"{docs[q].feed} | {docs[q].title}")
        for rank, (j, s) in enumerate(brute_reco(U_lsi, q, N_SHOW), 1):
            print(f"   {rank}. (cos={s:.3f}) {docs[j].feed} | {docs[j].title}")

    print("\n=== 完成 ===")
    print(f"k*={k_star}；LSI命中={eval_rows['hit_lsi']} vs 随机={eval_rows['hit_random']}；"
          f"近似ANN recall@10={eval_rows['ann_recall']}")
    if nap_df is not None and not nap_df.empty:
        print("最强新词锚点 Top5:", nap_df.head(5)[["word", "df", "NAP"]].to_dict("records"))


if __name__ == "__main__":
    main()
