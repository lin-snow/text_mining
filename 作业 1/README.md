# 文本信息挖掘概论 - 作业1：数据收集与预处理

**选题：** 科技类中文 RSS 新闻语料的收集与词频可视化  

**小组 / 姓名：** （请填写）

**要求说明：** [HOMEWORK.md](HOMEWORK.md)

**数据源：** 默认在 [`hw1_pipeline.py`](hw1_pipeline.py) 的 `FEEDS` 中配置 **10 个**中文科技向 RSS（含少数派、开源中国、36氪、机器之心、雷锋网、阮一峰博客、异次元等）；可按需增删。抓取时**每源最多 `MAX_ITEMS_PER_FEED` 条**，并按**标题去重**，正文优先取 Atom `content` 以提升词频信息量。详细列表见 [REPORT.md](REPORT.md)。

## 环境

```bash
cd "作业 1"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 若在运行中出现 **SSL 证书校验失败**，请确保已安装 `certifi`（已列入 `requirements.txt`）；本仓库 `hw1_pipeline.py` 已对 `urllib` 使用 `certifi.where()` 作为 CA 路径。

## 一键复现（采集 + 预处理 + 出图）

```bash
python hw1_pipeline.py
```

生成：

- `data/rss_raw.jsonl`、`data/rss_cleaned.csv`：抓取结果  
- `figures/wordcloud.png`：词云  
- `figures/top_words.png`：Matplotlib 词频 Top-20  

仅采集：

```bash
python scripts/collect_rss.py
```

## Jupyter Notebook（加分项）

```bash
cd notebook
jupyter notebook homework1.ipynb
```

或在项目根 `作业 1/` 下：

```bash
jupyter notebook notebook/homework1.ipynb
```

按单元格顺序运行即可。

## 报告

书面报告内容与结构见 [REPORT.md](REPORT.md)，可对照课程下发的 Word 模板誊写。

## 目录结构

| 路径 | 说明 |
|------|------|
| `hw1_pipeline.py` | 采集、预处理、可视化主逻辑 |
| `scripts/collect_rss.py` | 仅采集脚本 |
| `data/stopwords.txt` | 中文停用词表 |
| `notebook/homework1.ipynb` | 全流程笔记本 |
