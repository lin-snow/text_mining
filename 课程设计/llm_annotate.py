# -*- coding: utf-8 -*-
"""AIGC 增强：用大语言模型（Claude）对作业 2 自动挖出的新词做语义画像与质量审计。

这是课程设计「玩 AIGC」环节的方法脚本，诚实记录新词语义标注是如何产生的：
把每个候选新词连同它在语料中被 LDA 归入的主题、该主题的高频共现词，作为上下文
输入 Claude，要求其输出结构化的语义画像（类别 / 是否真实体 / 是否隐喻 / 情感 /
语义域 / 释义）。LLM 在此充当“语义裁判”，提供 TF-IDF、Word2Vec-KNN、LDA 共现这
三类纯统计方法都给不出的*语义判断*。

运行方式：
  - 默认（离线）：直接使用仓库内缓存好的 data/newword_llm.json（由本脚本此前在线生成），
    课程设计主管线 hw3_pipeline.py 也只读取该缓存，保证答辩现场全程离线可复现。
  - 重新生成（在线）：设置环境变量 ANTHROPIC_API_KEY 后运行本脚本，将实时调用
    Claude 重新标注并覆盖 data/newword_llm.json。
      ANTHROPIC_API_KEY=sk-... "../作业 2/.venv/bin/python" llm_annotate.py --regenerate

设计要点：模型 claude-opus-4-8；温度 0 以求标注稳定；输出强制 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NEWWORD_TOPIC = DATA / "newword_topic.csv"      # 作业2新词 → LDA主题（hw3_pipeline 产出）
TOPIC_TERMS = DATA / "topic_terms.csv"          # 各主题 Top 词
OUT_JSON = DATA / "newword_llm.json"

MODEL = "claude-opus-4-8"
SYSTEM = (
    "你是中文科技新词分析助手。只依据词在 2026 年初中文科技舆论中的真实含义判断，"
    "不要迎合 LDA 的归类。"
)
PROMPT_TMPL = (
    "候选新词：「{word}」\n"
    "它被 LDA 归入主题：{lda_label}（该词在此主题的归属度 {share}）。\n"
    "该主题高频共现词：{neighbors}\n\n"
    "请输出 JSON，字段：\n"
    "  category：实体-模型/实体-产品/实体-公司/实体-平台/技术术语/通用词/隐喻梗\n"
    "  is_real_term：该词是否值得收入科技领域分词词典的专名或术语（true），"
    "还是被近邻误召回的通用词（false）\n"
    "  is_metaphor：是否为隐喻或网络梗\n"
    "  sentiment：正面/中性/负面\n"
    "  llm_domain：你独立判断的语义域\n"
    "  gloss：一句话释义\n"
    "只输出 JSON，不要解释。"
)


def load_context() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    nw = pd.read_csv(NEWWORD_TOPIC)
    terms = pd.read_csv(TOPIC_TERMS)
    neigh: dict[str, list[str]] = {}
    for label, grp in terms.groupby("label"):
        neigh[str(label)] = grp.sort_values("rank")["term"].head(10).tolist()
    return nw, neigh


def annotate_online(nw: pd.DataFrame, neigh: dict[str, list[str]]) -> list[dict]:
    try:
        import anthropic
    except ImportError:
        sys.exit("缺少 anthropic 库：请 `pip install anthropic` 后重试，或不带 "
                 "--regenerate 直接使用缓存 JSON。")
    client = anthropic.Anthropic()  # 读取 ANTHROPIC_API_KEY
    items: list[dict] = []
    for _, r in nw.iterrows():
        word = str(r["word"]).strip()
        label = str(r["dominant_label"])
        prompt = PROMPT_TMPL.format(
            word=word, lda_label=label, share=round(float(r["topic_share"]), 3),
            neighbors="、".join(neigh.get(label, [])),
        )
        msg = client.messages.create(
            model=MODEL, max_tokens=400, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip().strip("`")
        if text.startswith("json"):
            text = text[4:]
        obj = json.loads(text)
        obj["word"] = word
        items.append(obj)
        print(f"[llm] {word:8s} -> {obj.get('category')} / real={obj.get('is_real_term')}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="调用 Claude API 重新生成（需 ANTHROPIC_API_KEY）")
    args = ap.parse_args()

    if not args.regenerate:
        if OUT_JSON.exists():
            d = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            print(f"使用缓存：{OUT_JSON.name}（{len(d['items'])} 词，模型 "
                  f"{d['_meta']['model']}）。如需在线重生成请加 --regenerate。")
            return 0
        sys.exit("缓存 newword_llm.json 不存在，请加 --regenerate 在线生成。")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("未检测到 ANTHROPIC_API_KEY，无法在线生成。")
    if not NEWWORD_TOPIC.exists():
        sys.exit("缺少 data/newword_topic.csv，请先运行 hw3_pipeline.py。")

    nw, neigh = load_context()
    print(f"在线标注 {len(nw)} 个新词（模型 {MODEL}）...")
    items = annotate_online(nw, neigh)
    payload = {
        "_meta": {
            "model": MODEL,
            "task": "对作业2自动挖出的候选新词做语义画像与质量审计",
            "generated_by": "llm_annotate.py --regenerate",
            "prompt_template": PROMPT_TMPL,
            "fields": "category, is_real_term, is_metaphor, sentiment, llm_domain, gloss",
        },
        "items": items,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写出 ->", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
