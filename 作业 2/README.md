# 作业 2 · 定制化分词词典（科技中文）

> 课程：文本信息挖掘概论 ｜ 学生：林奕宏（3123004449）软件工程 2 班

## 一句话简介
基于作业 1 抓取的中文科技 RSS 语料，按 *jieba 分词 → Word2Vec 词向量 → KNN 近邻搜索 → TF/DF 评估 → 词典回灌* 的循环，3 轮迭代得到一份 188 条目的定制化分词词典（种子 108 + 新挖 80）。

## 项目结构

```
作业 2/
├── HOMEWORK.md                # 课程作业题面
├── README.md                  # 本文件
├── REPORT.md                  # 正式报告（参考毕业设计格式）
├── requirements.txt           # Python 依赖
├── hw2_pipeline.py            # 端到端流水线 & 可视化
├── data/
│   ├── seed_dict.txt          # 手工整理的 108 词种子词典
│   ├── stopwords_extra.txt    # 针对新词发现的扩展停用词
│   ├── custom_dict_v0..v2.txt # 每轮词典快照
│   ├── custom_dict_final.txt  # 最终词典（188 词）
│   ├── new_words_evaluation.csv  # 新词得分明细
│   ├── word2vec_final.model      # 最终词向量
│   └── tokenized_final.jsonl     # 最终分词结果
├── figures/
│   ├── iter_growth.png        # 词典规模随迭代变化
│   ├── new_words_top.png      # 新词 Top-20 综合得分排行
│   ├── knn_demo.png           # 种子词的 Word2Vec 近邻示例
│   ├── case_compare.png       # 种子 vs 定制：分词对比
│   └── new_words_wordcloud.png   # 新词词云
└── notebook/
    └── homework2.ipynb        # 交互式 notebook
```

## 复现步骤

```bash
cd "作业 2"
python -m venv .venv               # Python 3.13 已验证
.venv/bin/pip install -r requirements.txt
.venv/bin/python hw2_pipeline.py   # 跑完整流水线 + 出图
```

或交互式：
```bash
.venv/bin/jupyter notebook notebook/homework2.ipynb
```

## 关键产物

- `data/custom_dict_final.txt` — 188 个词条的最终词典，可直接用 `jieba.load_userdict` 加载
- `data/new_words_evaluation.csv` — 80 个新词的来源种子、相似度、TF/DF 与综合得分
- `figures/*.png` — 全部可视化图表

## 引用

详见 [`REPORT.md`](REPORT.md) 末尾的参考文献列表。
