# 文本信息挖掘概论 · 作业1：数据收集与预处理

**题目：科技类中文 RSS 新闻语料的收集与词频可视化**

---

## 摘要

本作业从多个公开的中文科技向 RSS 源（含科技媒体与个人博客类订阅）抓取条目；解析时**优先使用 Atom `content` 全文**，否则使用摘要/描述，并与标题合并为待挖掘文本。对抓取结果按**标题去重**、并对**每个源限制最大条数**，以减轻单一站点通稿对词频的支配。文本经清洗、结巴分词与停用词过滤后，用词云与 Matplotlib 条形图展示高频词，完成从「数据定位—查询下载—解析存储—预处理—可视化」的基本流水线。

**关键词：** RSS；文本预处理；jieba；词云；Matplotlib

---

## 一、选题与数据来源

- **选题说明：** 选用科技新闻类短文本，体裁统一、获取路径清晰，便于展示分词与词频统计效果。
- **数据获取方式：** 通过 **HTTP/HTTPS** 获取各站点提供的 **RSS 2.0** 订阅地址（非登录接口，属站点公开的聚合格式），使用 Python 标准库 **urllib** 发起请求，**feedparser** 解析 XML。
- **数据源（须在复现时核对可访问性，站点可能调整订阅地址）：**
  1. Solidot：`https://www.solidot.org/index.rss`
  2. 爱范儿：`https://www.ifanr.com/feed`
  3. 少数派：`https://sspai.com/feed`
  4. 开源中国：`https://www.oschina.net/news/rss`
  5. 36氪：`https://www.36kr.com/feed`
  6. 机器之心：`https://www.jiqizhixin.com/feed`
  7. 雷锋网：`https://www.leiphone.com/feed`
  8. 阮一峰的网络日志：`https://www.ruanyifeng.com/blog/atom.xml`
  9. 异次元软件世界：`https://feed.iplaysoft.com/`
  10. CNBeta：`https://www.cnbeta.com.tw/backend.php`

每条记录保存字段：来源名称、标题、正文片段（优先全文）、合并文本、链接、发布时间（若解析可得）。

---

## 二、方法与实现要点

### 2.1 收集与存储

1. 使用 `urllib.request.Request` 设置 **User-Agent**，避免部分站点对默认客户端返回错误。
2. 使用 **SSL** 上下文并指定 **certifi** 的根证书路径，避免在部分环境下出现证书校验失败。
3. 解析后对标题与正文做简单 **HTML 标签去除** 与空白规整；若条目提供 **Atom content**，优先采用其文本以获得更多实词。
4. 每个 RSS 源仅保留前 **N** 条（代码中 `MAX_ITEMS_PER_FEED`），并在全库按**标题去重**，减少转载通稿重复计数。
5. 本地存储：**JSON Lines**（`data/rss_raw.jsonl`）便于按行追加与流式读取；**CSV**（`data/rss_cleaned.csv`）便于表格查看与汇报。

### 2.2 预处理

1. **分词：** 采用 **jieba** 默认词典进行中文分词。
2. **停用词：** 使用项目内 `data/stopwords.txt`（通用虚词、标点名、部分无区分度词及 RSS 相关噪声词），过滤过短与纯符号片段。
3. **统计：** 对全部词项做词频计数，取 Top 词用于条形图与词云。

### 2.3 可视化

1. **词云：** **wordcloud**，背景白色，限制最大词数，关闭搭配短语以避免英文式二元搭配干扰中文展示。
2. **条形图：** **Matplotlib** 横向条形图展示 Top-20 词频；在 macOS 下通过系统中文字体文件路径设置 **FontProperties**，保证中文标签可显示。

---

## 三、结果说明

运行 `python hw1_pipeline.py` 或在 `notebook/homework1.ipynb` 中顺序执行后，将生成：

| 输出 | 说明 |
|------|------|
| `figures/wordcloud.png` | 词云图 |
| `figures/top_words.png` | 词频 Top-20 条形图 |
| `data/rss_raw.jsonl` / `data/rss_cleaned.csv` | 原始整理数据 |

（具体高频词会随抓取时刻的 RSS 内容变化，属正常现象。）

---

## 四、结论

本作业实现了面向公开 RSS 的科技中文短文本采集与基础预处理，并以词云与柱状图完成可视化展示。后续若扩展，可在同一流水线基础上增加去重策略、时间序列统计或主题建模等，但已超出本次作业范围。

---

## 参考文献

1. Python Software Foundation. *urllib — URL handling modules*. Python 3 documentation. https://docs.python.org/3/library/urllib.html Accessed 2026-04-06.  
2. feedparser contributors. *Universal Feed Parser*. https://feedparser.readthedocs.io/ Accessed 2026-04-06.  
3. Sun Junyi et al. *jieba 中文分词*. https://github.com/fxsjy/jieba Accessed 2026-04-06.  
4. Andreas Mueller. *wordcloud*. https://github.com/amueller/word_cloud Accessed 2026-04-06.  
5. Matplotlib development team. *Matplotlib documentation*. https://matplotlib.org/ Accessed 2026-04-06.  
6. Certifi Project. *Certifi: Python SSL Certificates*. https://github.com/certifi/python-certifi Accessed 2026-04-06.  
7. Solidot. RSS 订阅说明见站点页脚/订阅入口；本作业使用的订阅地址为 `https://www.solidot.org/index.rss`。Accessed 2026-04-06.  
8. 爱范儿. RSS：`https://www.ifanr.com/feed`。Accessed 2026-04-06.

（作业要求中的 Urllib、Scapy、Selenium 等为网络与自动化相关技术索引；本作业主要实践 **urllib** 与 HTML/XML 解析类库 **feedparser**，与课程说明一致。）
