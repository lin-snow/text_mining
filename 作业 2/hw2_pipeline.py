# -*- coding: utf-8 -*-
"""作业 2：基于科技 RSS 语料的定制化词典构建。

流水线：
1. 载入种子词典 + 通用词典
2. 读取作业 1 抓取的 RSS 语料
3. 使用当前词典进行 jieba 分词
4. 训练 Word2Vec 词向量
5. Annoy 索引词向量、对种子词做 KNN 搜索
6. 以 (similarity, TF, DF, 字面合法性) 评估新词候选
7. 高分候选入词典，回到第 3 步迭代
8. 输出词典增长、新词排行、分词对比等可视化
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Annoy 在当前环境 (Python 3.13 / macOS arm64) 触发 segfault，
# 退化为基于 numpy 的暴力 KNN；语料词表 ~4k，开销可忽略。
# 参考：https://github.com/spotify/annoy/issues  (与 Python 3.13 wheel 兼容性问题相关)
try:
    from annoy import AnnoyIndex  # noqa: F401  (留作未来环境恢复)
    _ANNOY_OK = False  # 在当前环境实测会段错误，强制走 brute-force
except Exception:  # pragma: no cover
    _ANNOY_OK = False

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

CORPUS_CSV = ROOT.parent / "作业 1" / "data" / "rss_cleaned.csv"
STOPWORDS_PATH = ROOT.parent / "作业 1" / "data" / "stopwords.txt"
STOPWORDS_EXTRA = DATA_DIR / "stopwords_extra.txt"
SEED_DICT_PATH = DATA_DIR / "seed_dict.txt"

ITERATIONS = 3
KNN_K = 12
EMBED_DIM = 80
MIN_TF = 4
MIN_DF = 2
MIN_SIM = 0.35
TOP_N_PER_ITER = [40, 25, 15]   # 每轮加入的新词上限：递减以收敛

# 仅允许包含中文/英文/数字的候选词，长度 2~10
_CAND_RE = re.compile(r"^[一-鿿A-Za-z0-9·]{2,10}$")
# 排除纯英文短串/全是数字
_BAD_RE = re.compile(r"^(?:\d+|[A-Za-z]{1,2})$")


# --------------------------- 数据载入 ---------------------------
def load_corpus(csv_path: Path = CORPUS_CSV) -> list[str]:
    df = pd.read_csv(csv_path)
    return [t for t in df["text"].dropna().astype(str).tolist() if t.strip()]


def load_stopwords(path: Path = STOPWORDS_PATH, extra: Path = STOPWORDS_EXTRA) -> set[str]:
    words: set[str] = set()
    for p in (path, extra):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                words.add(ln)
    return words


def parse_seed_dict(path: Path = SEED_DICT_PATH) -> list[str]:
    """从种子词典文件提取词条（忽略注释/空行）。"""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(ln.split()[0])
    return out


# --------------------------- 分词器 ---------------------------
def fresh_tokenizer(user_dict_path: Path | None = None) -> jieba.Tokenizer:
    """每轮重建分词器，避免状态污染。"""
    tk = jieba.Tokenizer()
    tk.initialize()
    if user_dict_path and user_dict_path.exists():
        tk.load_userdict(str(user_dict_path))
    return tk


def write_user_dict(words: dict[str, str], path: Path) -> Path:
    """把当前词典持久化为 jieba 兼容格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w, origin in words.items():
            # seed 词条频率给高一点，迭代加入的稍低
            freq = 1000 if origin == "seed" else 500
            f.write(f"{w} {freq}\n")
    return path


def good_token(w: str, stopwords: set[str]) -> bool:
    w = w.strip()
    if len(w) < 2 or w in stopwords:
        return False
    if not _CAND_RE.match(w):
        return False
    if _BAD_RE.match(w):
        return False
    return True


def segment_corpus(
    docs: list[str], tokenizer: jieba.Tokenizer, stopwords: set[str]
) -> list[list[str]]:
    out: list[list[str]] = []
    for d in docs:
        out.append([w for w in tokenizer.cut(d) if good_token(w, stopwords)])
    return out


# --------------------------- 词向量 + Annoy ---------------------------
def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    """小语料偏好 skip-gram + 较少 epoch，避免过拟合。"""
    return Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=5,
        min_count=3,
        sg=1,
        workers=2,
        epochs=12,
        seed=42,
    )


class KnnIndex:
    """空间索引抽象：优先 Annoy，否则使用 numpy 暴力余弦 KNN。"""

    def __init__(self, model: Word2Vec):
        self.vocab = list(model.wv.index_to_key)
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        vectors = np.asarray(model.wv.vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._unit = vectors / norms
        self._annoy = None
        if _ANNOY_OK:
            self._annoy = AnnoyIndex(model.wv.vector_size, "angular")
            for i, v in enumerate(vectors):
                self._annoy.add_item(i, v)
            self._annoy.build(30)

    def knn(self, word: str, k: int) -> list[tuple[str, float]]:
        """返回 [(neighbor, cosine_sim), ...]，按相似度降序，排除自身。"""
        if word not in self.word2id:
            return []
        qid = self.word2id[word]
        if self._annoy is not None:
            ids, dists = self._annoy.get_nns_by_item(
                qid, k + 1, include_distances=True
            )
            out = []
            for nid, d in zip(ids, dists):
                if nid == qid:
                    continue
                sim = max(0.0, 1.0 - (d ** 2) / 2.0)
                out.append((self.vocab[nid], sim))
                if len(out) >= k:
                    break
            return out
        sims = self._unit @ self._unit[qid]
        # argpartition 取 top-k+1，再排序，避免对全量排序
        top = np.argpartition(-sims, k + 1)[: k + 1]
        ordered = top[np.argsort(-sims[top])]
        out = []
        for nid in ordered:
            if nid == qid:
                continue
            out.append((self.vocab[int(nid)], float(sims[int(nid)])))
            if len(out) >= k:
                break
        return out


# --------------------------- 新词评估 ---------------------------
@dataclass
class Candidate:
    word: str
    sim: float = 0.0
    tf: int = 0
    df: int = 0
    parents: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        # 综合：与种子最相近度 × log(频次) × log(文档频次)
        return self.sim * math.log(1 + self.tf) * math.log(1 + self.df)


def discover_new_words(
    cur_dict: dict[str, str],
    sentences: list[list[str]],
    knn_index: KnnIndex,
    stopwords: set[str],
    iter_idx: int,
) -> list[Candidate]:
    tf = Counter(w for sent in sentences for w in sent)
    df = Counter()
    for sent in sentences:
        for w in set(sent):
            df[w] += 1

    cands: dict[str, Candidate] = {}
    for seed in cur_dict.keys():
        for cand, sim in knn_index.knn(seed, KNN_K):
            if cand in cur_dict or cand in stopwords:
                continue
            if not good_token(cand, stopwords):
                continue
            if sim < MIN_SIM:
                continue
            if tf[cand] < MIN_TF or df[cand] < MIN_DF:
                continue
            entry = cands.setdefault(cand, Candidate(word=cand, tf=tf[cand], df=df[cand]))
            if sim > entry.sim:
                entry.sim = sim
            if seed not in entry.parents:
                entry.parents.append(seed)

    ranked = sorted(cands.values(), key=lambda c: -c.score)
    limit = TOP_N_PER_ITER[min(iter_idx, len(TOP_N_PER_ITER) - 1)]
    return ranked[:limit]


# --------------------------- 主流程 ---------------------------
@dataclass
class IterReport:
    iter: int
    dict_size_before: int
    dict_size_after: int
    added: list[Candidate]
    tokens_total: int
    vocab_size: int


def run_pipeline(out_dir: Path = DATA_DIR) -> tuple[list[IterReport], dict[str, str], Word2Vec]:
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_corpus()
    stopwords = load_stopwords()
    seed_words = parse_seed_dict()
    cur_dict: dict[str, str] = {w: "seed" for w in seed_words}
    print(f"[init] 文档数={len(docs)} 种子词={len(cur_dict)} 停用词={len(stopwords)}")

    reports: list[IterReport] = []
    final_model: Word2Vec | None = None
    final_sentences: list[list[str]] = []

    for it in range(ITERATIONS):
        ud_path = out_dir / f"custom_dict_v{it}.txt"
        write_user_dict(cur_dict, ud_path)
        tokenizer = fresh_tokenizer(ud_path)
        sentences = segment_corpus(docs, tokenizer, stopwords)
        token_total = sum(len(s) for s in sentences)
        print(
            f"[iter {it+1}] 加载词典 {len(cur_dict)} 条 -> "
            f"分词产出 {token_total} 个 token"
        )

        model = train_word2vec(sentences)
        print(f"[iter {it+1}] Word2Vec 词表={len(model.wv.index_to_key)}")
        knn_index = KnnIndex(model)

        added = discover_new_words(cur_dict, sentences, knn_index, stopwords, it)
        size_before = len(cur_dict)
        for c in added:
            cur_dict[c.word] = f"iter{it+1}"
        size_after = len(cur_dict)
        print(f"[iter {it+1}] 新增 {len(added)} 词 -> 词典 {size_before}→{size_after}")

        reports.append(
            IterReport(
                iter=it + 1,
                dict_size_before=size_before,
                dict_size_after=size_after,
                added=added,
                tokens_total=token_total,
                vocab_size=len(model.wv.index_to_key),
            )
        )
        final_model = model
        final_sentences = sentences

    # 写出最终词典 + 新词明细
    final_path = out_dir / "custom_dict_final.txt"
    write_user_dict(cur_dict, final_path)

    new_rows = []
    for r in reports:
        for c in r.added:
            new_rows.append(
                {
                    "iter": r.iter,
                    "word": c.word,
                    "sim_max": round(c.sim, 4),
                    "tf": c.tf,
                    "df": c.df,
                    "score": round(c.score, 4),
                    "parents": "|".join(c.parents[:5]),
                }
            )
    pd.DataFrame(new_rows).to_csv(
        out_dir / "new_words_evaluation.csv", index=False, encoding="utf-8-sig"
    )

    # 保存 word2vec 与最终分词后的语料（注释行可选）
    final_model.save(str(out_dir / "word2vec_final.model"))
    with open(out_dir / "tokenized_final.jsonl", "w", encoding="utf-8") as f:
        for sent in final_sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")

    return reports, cur_dict, final_model


# --------------------------- 可视化 ---------------------------
def _chinese_font():
    from matplotlib import font_manager as fm

    for p in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return None


def plot_dict_growth(reports: list[IterReport], out: Path) -> Path:
    font = _chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    iters = [r.iter for r in reports]
    sizes_before = [r.dict_size_before for r in reports]
    sizes_after = [r.dict_size_after for r in reports]
    added = [len(r.added) for r in reports]

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    x = [i for i in iters]
    ax1.bar(
        [xi - width / 2 for xi in x], sizes_before,
        width=width, color="#94a3b8", label="迭代前规模"
    )
    ax1.bar(
        [xi + width / 2 for xi in x], sizes_after,
        width=width, color="#2563eb", label="迭代后规模"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"第 {i} 轮" for i in iters], fontproperties=font)
    ax1.set_ylabel("词典词条数", fontproperties=font)
    ax1.set_title("定制化词典规模随迭代轮次的变化", fontproperties=font)
    for xi, v in zip(x, sizes_after):
        ax1.text(xi + width / 2, v + 1, str(v), ha="center", fontproperties=font)

    ax2 = ax1.twinx()
    ax2.plot(x, added, color="#dc2626", marker="o", label="本轮新增词数")
    ax2.set_ylabel("本轮新增词数", fontproperties=font, color="#dc2626")
    for xi, v in zip(x, added):
        ax2.text(xi, v + 0.5, f"+{v}", color="#dc2626", ha="center", fontproperties=font)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, prop=font, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已保存 -> {out}")
    return out


def plot_new_words_score(reports: list[IterReport], out: Path, top_k: int = 20) -> Path:
    font = _chinese_font()
    plt.rcParams["axes.unicode_minus"] = False

    rows = []
    for r in reports:
        for c in r.added:
            rows.append((c.word, c.score, r.iter, c.sim, c.tf))
    rows.sort(key=lambda x: -x[1])
    rows = rows[:top_k]
    if not rows:
        raise RuntimeError("没有可绘制的新词")
    words = [r[0] for r in rows][::-1]
    scores = [r[1] for r in rows][::-1]
    iters = [r[2] for r in rows][::-1]
    colors = {1: "#2563eb", 2: "#0ea5e9", 3: "#22c55e"}

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(range(len(words)), scores, color=[colors.get(i, "#94a3b8") for i in iters])
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontproperties=font)
    ax.set_xlabel("综合得分 = max_sim × log(1+TF) × log(1+DF)", fontproperties=font)
    ax.set_title(f"挖掘到的新词 Top-{top_k}（按综合得分排序）", fontproperties=font)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()
    ]
    ax.legend(
        handles,
        [f"第 {k} 轮加入" for k in colors.keys()],
        prop=font, loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已保存 -> {out}")
    return out


def _split_sentences(doc: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。！？!?\.;；\n]", doc) if s.strip()]


def _pick_case_sentences(
    docs: list[str],
    extra_dict_words: list[str],
    n: int = 5,
) -> list[str]:
    """挑选那些 *因定制词典而出现切分差异* 的短句。"""
    pool: list[tuple[int, str]] = []
    for d in docs:
        for s in _split_sentences(d):
            if 15 <= len(s) <= 55:
                score = sum(s.count(w) for w in extra_dict_words)
                if score > 0:
                    pool.append((score, s))
    pool.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, s in pool:
        key = s[:20]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= n:
            break
    return out


def _token_width(tok: str) -> float:
    """每个字符的宽度估计（中文 ~ 2 单位，西文/数字 ~ 1 单位）。"""
    w = 0.0
    for ch in tok:
        w += 2.0 if "一" <= ch <= "鿿" else 1.0
    return w + 1.6  # 末尾加上分隔符宽度


def plot_segmentation_compare(
    docs: list[str],
    seed_dict_path: Path,
    final_dict_path: Path,
    extra_words: list[str],
    out: Path,
    sample: list[str] | None = None,
) -> Path:
    """对若干例句使用 v0 与最终词典分词，对比展示，新词高亮。"""
    font = _chinese_font()
    plt.rcParams["axes.unicode_minus"] = False

    tk0 = fresh_tokenizer(seed_dict_path)
    tk1 = fresh_tokenizer(final_dict_path)
    sample = sample or _pick_case_sentences(docs, extra_words, n=5)
    if not sample:
        sample = [d[:40] for d in docs[:3]]

    rows = []
    for sent in sample:
        a = list(tk0.cut(sent))
        b = list(tk1.cut(sent))
        rows.append((sent, a, b))

    extra_set = set(extra_words)

    # 计算所有行所需最大宽度（字符宽度单位）
    LABEL_W = 11.0
    max_w = 0.0
    for sent, a, b in rows:
        for toks in (a, b):
            w = LABEL_W + sum(_token_width(t) for t in toks)
            max_w = max(max_w, w)
        max_w = max(max_w, LABEL_W + _token_width(sent))
    fig_w = max(12.0, min(20.0, max_w * 0.16))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, 1.55 * n + 0.6), squeeze=False)

    def render_row(ax, label, tokens, *, highlight=False, label_color="#475569"):
        x = 0.0
        ax.text(
            x, 0.5, label, fontproperties=font, fontsize=10.5,
            color=label_color, weight="bold", va="center",
        )
        x = LABEL_W
        for i, tok in enumerate(tokens):
            if highlight and tok in extra_set:
                ax.text(
                    x, 0.5, tok, fontproperties=font, fontsize=11,
                    color="#1d4ed8", weight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#dbeafe", ec="#bfdbfe", lw=0.6),
                )
            else:
                ax.text(
                    x, 0.5, tok, fontproperties=font, fontsize=10.5,
                    color="#0f172a" if highlight else "#64748b", va="center",
                )
            x += _token_width(tok)
            if i < len(tokens) - 1:
                ax.text(
                    x - 1.0, 0.5, "·", fontproperties=font, fontsize=9,
                    color="#cbd5e1", va="center",
                )

    for ax_row, (sent, a, b) in zip(axes[:, 0], rows):
        ax_row.axis("off")
        ax_row.set_xlim(0, max_w + 2)
        ax_row.set_ylim(0, 3)
        # 三行：原句 / 种子 / 定制
        for ax_y, label, content, kw in (
            (2.3, "原句:", [sent], {"label_color": "#0f172a", "highlight": False}),
            (1.35, "种子词典:", a, {"label_color": "#475569", "highlight": False}),
            (0.4, "定制词典:", b, {"label_color": "#475569", "highlight": True}),
        ):
            # 用 sub-axes 一行简化：直接在同一 ax 上不同 y 渲染
            x = 0.0
            ax_row.text(
                x, ax_y, label, fontproperties=font, fontsize=10.5,
                color=kw["label_color"], weight="bold", va="center",
            )
            x = LABEL_W
            for i, tok in enumerate(content):
                if kw["highlight"] and tok in extra_set:
                    ax_row.text(
                        x, ax_y, tok, fontproperties=font, fontsize=11,
                        color="#1d4ed8", weight="bold", va="center",
                        bbox=dict(boxstyle="round,pad=0.16", fc="#dbeafe",
                                  ec="#93c5fd", lw=0.5),
                    )
                else:
                    color = "#0f172a" if ax_y > 2 else ("#0f172a" if kw["highlight"] else "#64748b")
                    ax_row.text(
                        x, ax_y, tok, fontproperties=font, fontsize=10.5,
                        color=color, va="center",
                    )
                x += _token_width(tok)
                if i < len(content) - 1 and ax_y < 2:
                    ax_row.text(
                        x - 1.0, ax_y, "·", fontproperties=font, fontsize=9,
                        color="#cbd5e1", va="center",
                    )

    fig.suptitle(
        "种子词典 vs 定制化词典：分词对比（蓝色高亮 = 定制词典识别出的新词）",
        fontproperties=font, fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已保存 -> {out}")
    return out


def plot_knn_neighbors(
    model: Word2Vec, seeds: list[str], out: Path, k: int = 8
) -> Path:
    font = _chinese_font()
    plt.rcParams["axes.unicode_minus"] = False

    seeds = [s for s in seeds if s in model.wv.key_to_index][:5]
    if not seeds:
        raise RuntimeError("种子词全部缺失向量")

    fig, axes = plt.subplots(1, len(seeds), figsize=(3.6 * len(seeds), 5.5))
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        neighbors = model.wv.most_similar(seed, topn=k)
        names = [n for n, _ in neighbors][::-1]
        sims = [s for _, s in neighbors][::-1]
        ax.barh(range(len(names)), sims, color="#0ea5e9")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontproperties=font, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"「{seed}」的 KNN", fontproperties=font, fontsize=11)
        ax.set_xlabel("cos sim", fontsize=9)
    fig.suptitle("种子词在 Word2Vec 空间中的近邻示例", fontproperties=font, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已保存 -> {out}")
    return out


def plot_new_words_wordcloud(reports: list[IterReport], out: Path) -> Path:
    from wordcloud import WordCloud

    font_path = None
    for p in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(p).exists():
            font_path = p
            break
    freq: dict[str, float] = {}
    for r in reports:
        for c in r.added:
            freq[c.word] = c.score
    if not freq:
        raise RuntimeError("没有新词可生成词云")
    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=600,
        background_color="white",
        max_words=200,
        prefer_horizontal=0.95,
        colormap="viridis",
    ).generate_from_frequencies(freq)
    wc.to_file(str(out))
    print(f"已保存 -> {out}")
    return out


def make_all_figures(
    reports: list[IterReport],
    cur_dict: dict[str, str],
    model: Word2Vec,
) -> dict[str, Path]:
    docs = load_corpus()
    stopwords = load_stopwords()
    paths = {}
    paths["growth"] = plot_dict_growth(reports, FIGURES_DIR / "iter_growth.png")
    paths["new_words"] = plot_new_words_score(reports, FIGURES_DIR / "new_words_top.png")
    extra_words = sorted(
        {c.word for r in reports for c in r.added}, key=len, reverse=True
    )
    paths["case"] = plot_segmentation_compare(
        docs,
        DATA_DIR / "custom_dict_v0.txt",
        DATA_DIR / "custom_dict_final.txt",
        extra_words,
        FIGURES_DIR / "case_compare.png",
    )
    seeds_for_knn = ["大模型", "开源", "内核", "GPU", "AI"]
    paths["knn"] = plot_knn_neighbors(model, seeds_for_knn, FIGURES_DIR / "knn_demo.png")
    paths["wc"] = plot_new_words_wordcloud(reports, FIGURES_DIR / "new_words_wordcloud.png")
    return paths


if __name__ == "__main__":
    reports, cur_dict, model = run_pipeline()
    make_all_figures(reports, cur_dict, model)
    print("\n=== 完成 ===")
    print(f"最终词典规模：{len(cur_dict)}")
