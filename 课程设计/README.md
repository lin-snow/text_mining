# 课程设计：基于 LDA 主题模型的中文科技 RSS 语料主题挖掘

> 文本信息挖掘概论 · 课程设计　|　林奕宏 3123004449 软件工程 2 班

复现论文：**Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. JMLR**（被引 5 万+）。
将 LDA 应用于自建的中文科技 RSS 语料，挖掘主题结构、媒体偏好与时间维度上的主题热度，
并把前两次作业的产出深度联动起来。

## 三次作业的联动

| 作业 | 角色 | 复用产物 |
|---|---|---|
| 作业 1 | 数据采集 | `作业 1/data/rss_cleaned.csv`（221 篇科技 RSS，含来源/时间） |
| 作业 2 | 分词词典 | `作业 2/data/custom_dict_final.txt`（188 词）+ 80 个新词 |
| **作业 3（本设计）** | 主题建模 | LDA 主题、可视化、"新词→主题"归属分析 |

## 目录结构

```
课程设计/
├── hw3_pipeline.py        # 端到端 LDA 管线（分词→选K→训练→导出→出图）
├── build_docx.py          # 把 REPORT.md 渲染为可打印的 课程设计.docx
├── REPORT.md              # 正式报告（毕设格式，Markdown 源）
├── 课程设计.docx           # 提交用打印版本（由 build_docx.py 生成）
├── requirements.txt
├── data/                  # 主题、文档-主题、主题-来源、选K、新词归属等 CSV + 模型 + 标签/停用词
├── figures/               # 全部可视化 PNG（含报告所用公式图、流程图）
└── notebook/homework3.ipynb
```

## 复现

```bash
cd "课程设计"
# 复用作业 2 的虚拟环境（含 gensim/jieba/matplotlib/wordcloud）
"../作业 2/.venv/bin/python" hw3_pipeline.py     # 约 9 秒，产出 data/ 与 figures/
# 主题命名可在 data/topic_labels.json 校准后重跑
"../作业 2/.venv/bin/python" build_docx.py        # 生成 课程设计.docx（需 officecli）
```

## 核心结论

- 困惑度 + C_v 一致性选定 **K=10**（C_v 峰值 0.417）；
- 2026 年初中文科技舆论解构为开源小模型 / AI 编程助手 / 生成式 AI / 产业 AI / AI 芯片 / 自动驾驶 / 激光雷达 / 苹果生态等 10 个主题；
- 主题间距离图（JS 散度 + 经典 MDS）在第一主成分上分出"AI/软件 vs 硬件/汽车"两大阵营；
- 媒体主题偏好显著（阮一峰→AI 编程 59%、雷锋网→激光雷达 30%、少数派→苹果生态 36%）；
- 作业 2 的 80 个新词中 **79 个**被本设计主题成功吸纳，印证两次作业的语义一致性。
