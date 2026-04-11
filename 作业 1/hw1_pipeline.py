# -*- coding: utf-8 -*-
"""作业1：科技类中文 RSS 采集、预处理与词频/词云可视化。"""
from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import certifi
import feedparser
import jieba
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RAW_JSONL = DATA_DIR / "rss_raw.jsonl"
CLEAN_CSV = DATA_DIR / "rss_cleaned.csv"
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"

# 公开 RSS：科技向中文博客/媒体（需在报告中注明出处与访问日期）
FEEDS: list[dict[str, str]] = [
    {"name": "Solidot", "url": "https://www.solidot.org/index.rss"},
    {"name": "爱范儿", "url": "https://www.ifanr.com/feed"},
    {"name": "少数派", "url": "https://sspai.com/feed"},
    {"name": "开源中国", "url": "https://www.oschina.net/news/rss"},
    {"name": "36氪", "url": "https://www.36kr.com/feed"},
    {"name": "机器之心", "url": "https://www.jiqizhixin.com/feed"},
    {"name": "雷锋网", "url": "https://www.leiphone.com/feed"},
    {"name": "阮一峰的网络日志", "url": "https://www.ruanyifeng.com/blog/atom.xml"},
    {"name": "异次元软件世界", "url": "https://feed.iplaysoft.com/"},
    {"name": "CNBeta", "url": "https://www.cnbeta.com.tw/backend.php"},
]

# 每个源最多取若干条，避免单一站点条目过多压过其他来源
MAX_ITEMS_PER_FEED = 45

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&lt;|&gt;|&amp;|&quot;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _entry_body_text(entry) -> str:
    """优先使用 Atom/RSS 的 content 全文，其次 summary/description，词频更稳。"""
    chunks: list[str] = []
    content = getattr(entry, "content", None)
    if content:
        for c in content:
            if isinstance(c, dict) and c.get("value"):
                chunks.append(c["value"])
    if chunks:
        merged = _strip_html(" ".join(chunks))
        if len(merged) >= 12:
            return merged
    raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    return _strip_html(raw)


def fetch_rss_bytes(url: str, timeout: int = 20) -> bytes:
    """使用 urllib 拉取 RSS（符合课程对 urllib 的引用场景）。"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def collect_feeds(
    out_jsonl: Path | None = None,
    out_csv: Path | None = None,
) -> list[dict]:
    """抓取配置的 RSS，写入 jsonl 与 csv。"""
    out_jsonl = out_jsonl or RAW_JSONL
    out_csv = out_csv or CLEAN_CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for feed in FEEDS:
        print(f"[info] 正在抓取: {feed['name']} -> {feed['url']}")
        try:
            content = fetch_rss_bytes(feed["url"])
        except (urllib.error.URLError, OSError) as e:
            print(f"[warn] 无法拉取 {feed['name']} {feed['url']}: {e}")
            continue
        parsed = feedparser.parse(content)
        entries = (getattr(parsed, "entries", []) or [])[:MAX_ITEMS_PER_FEED]
        print(f"[info] 抓取完成: {feed['name']}，获得 {len(entries)} 条")
        for entry in entries:
            title = _strip_html(getattr(entry, "title", "") or "")
            summary = _entry_body_text(entry)
            link = getattr(entry, "link", "") or ""
            published = (
                getattr(entry, "published", "")
                or getattr(entry, "updated", "")
                or getattr(entry, "published_parsed", "")
                or ""
            )
            text_full = f"{title} {summary}".strip()
            rows.append(
                {
                    "feed": feed["name"],
                    "title": title,
                    "summary": summary,
                    "text": text_full,
                    "link": link,
                    "published": str(published),
                }
            )

    # 按标题去重，减少同源通稿在多站转载导致的词频偏置
    seen_titles: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        key = r["title"].strip().lower()
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(r)
    rows = deduped

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已写入 {len(rows)} 条 -> {out_jsonl} , {out_csv}")
    return rows


def load_stopwords(path: Path | None = None) -> set[str]:
    path = path or STOPWORDS_PATH
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize_corpus(texts: list[str], stopwords: set[str]) -> list[str]:
    """结巴分词 + 停用词过滤 + 长度过滤。"""
    tokens: list[str] = []
    for t in texts:
        for w in jieba.cut(t):
            w = w.strip()
            if len(w) < 2:
                continue
            if w in stopwords:
                continue
            if re.match(r"^[0-9\s\W_]+$", w):
                continue
            tokens.append(w)
    return tokens


def word_frequencies(tokens: list[str], top_n: int = 50) -> list[tuple[str, int]]:
    return Counter(tokens).most_common(top_n)


def _pick_chinese_font_path() -> str | None:
    from matplotlib import font_manager

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    for f in font_manager.fontManager.ttflist:
        name = getattr(f, "name", "") or ""
        if any(
            x in name
            for x in (
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "FangSong",
                "KaiTi",
                "PingFang",
                "Heiti",
                "Songti",
                "Hiragino",
            )
        ):
            return f.fname
    return None


def plot_top_words(
    freq: list[tuple[str, int]],
    top_k: int = 20,
    out_path: Path | None = None,
) -> Path:
    """Matplotlib：词频 Top-K 条形图。"""
    from matplotlib import font_manager as fm

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (FIGURES_DIR / "top_words.png")
    top = freq[:top_k]
    if not top:
        raise ValueError("词频为空，无法绘图")

    words, counts = zip(*reversed(top))
    font_path = _pick_chinese_font_path()
    font_prop = fm.FontProperties(fname=font_path) if font_path else None
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), list(counts), color="#3b82f6")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(list(words), fontproperties=font_prop)
    ax.set_xlabel("频次", fontproperties=font_prop)
    ax.set_title("RSS 合并文本词频 Top-%d" % top_k, fontproperties=font_prop)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存条形图 -> {out_path}")
    return out_path


def plot_wordcloud(
    tokens: list[str],
    out_path: Path | None = None,
    width: int = 1200,
    height: int = 700,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (FIGURES_DIR / "wordcloud.png")
    text = " ".join(tokens)
    if not text.strip():
        raise ValueError("分词结果为空，无法生成词云")

    font = _pick_chinese_font_path()
    wc = WordCloud(
        font_path=font,
        width=width,
        height=height,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate(text)
    wc.to_file(str(out_path))
    print(f"已保存词云 -> {out_path}")
    return out_path


def run_visualization(
    jsonl_path: Path | None = None,
) -> tuple[Path, Path]:
    """从 jsonl 读取、分词、输出词云与条形图。"""
    jsonl_path = jsonl_path or RAW_JSONL
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    texts = [r.get("text", "") for r in rows]
    stopwords = load_stopwords()
    if not texts or not any(t.strip() for t in texts):
        raise FileNotFoundError(
            f"{jsonl_path} 无有效文本，请先成功运行 collect_feeds() 或检查网络/SSL。"
        )
    tokens = tokenize_corpus(texts, stopwords)
    freq = word_frequencies(tokens, top_n=80)
    if not tokens:
        raise ValueError("分词后无词项，请检查停用词表是否过宽或原始文本是否为空。")
    wc_path = plot_wordcloud(tokens)
    bar_path = plot_top_words(freq, top_k=20)
    return wc_path, bar_path


if __name__ == "__main__":
    print("[info] 开始执行 RSS 采集与可视化流水线...")
    collect_feeds()
    run_visualization()
    print("[info] 全部完成。")
