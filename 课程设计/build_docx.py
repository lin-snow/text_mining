# -*- coding: utf-8 -*-
"""把 REPORT.md 渲染为可打印的课程设计 Word 文档（毕设格式）。

步骤：
  1. 用 matplotlib 渲染 3 张展示公式图 + 1 张流程图（离线，无需 pandoc/mermaid）；
  2. 解析 REPORT.md（标题/段落/表格/图片/公式/流程图/列表/引用）为 officecli batch 操作；
  3. 调用 officecli 生成 课程设计.docx，并 validate。

字体：正文 宋体 + Times New Roman，标题 黑体，行距 1.5，首行缩进 2 字符。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
REPORT = ROOT / "REPORT.md"
OUT = ROOT / "课程设计.docx"

CJK = "宋体"          # 正文中文字体（目标 Windows 渲染为 SimSun）
CJK_H = "黑体"        # 标题中文字体（SimHei）
LATIN = "Times New Roman"


def font(path_candidates):
    from matplotlib import font_manager as fm
    for p in path_candidates:
        if Path(p).exists():
            return fm.FontProperties(fname=p)
    return None


CN_FONT = font([
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
])


# =============== 1. 资源图：公式 + 流程图 ===============
EQUATIONS = [
    r"$p(\mathbf{W},\mathbf{Z},\theta,\varphi\,|\,\alpha,\beta)=\prod_{k=1}^{K}p(\varphi_k|\beta)\,\prod_{d=1}^{D}p(\theta_d|\alpha)\,\prod_{n=1}^{N_d}p(z_{d,n}|\theta_d)\,p(w_{d,n}|\varphi_{z_{d,n}})$",
    r"$\mathrm{Perplexity}=2^{-\frac{1}{N}\sum_{d}\log_2 p(\mathbf{w}_d)}$",
    r"$\mathrm{JSD}(p,q)=\frac{1}{2}\mathrm{KL}(p\|m)+\frac{1}{2}\mathrm{KL}(q\|m),\quad m=\frac{1}{2}(p+q)$",
]


def render_equations():
    paths = []
    for idx, eq in enumerate(EQUATIONS, 1):
        fig = plt.figure(figsize=(8.6, 0.95))
        fig.text(0.5, 0.5, eq, ha="center", va="center", fontsize=19)
        out = FIG / f"eq_{idx}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.18,
                    facecolor="white")
        plt.close(fig)
        paths.append(out)
    return paths


def render_flowchart():
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc, fs=11, ec="#334155"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.6,rounding_size=2",
                     fc=fc, ec=ec, lw=1.3, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontproperties=CN_FONT,
                fontsize=fs, zorder=3, color="#0f172a")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.6),
                    zorder=1)

    # 顶部三个输入
    box(20, 92, 26, 9, "作业1 RSS 语料\n221 篇·含时间/来源", "#dbeafe", 10)
    box(50, 92, 24, 9, "作业2 定制词典\n188 词", "#dcfce7", 10)
    box(80, 92, 24, 9, "三级停用词\n767 个", "#fef9c3", 10)
    # 主干
    box(50, 76, 40, 8, "② jieba 分词 + good_token 过滤", "#e0f2fe", 11)
    box(50, 63, 46, 8, "③ Dictionary + filter_extremes → BoW（1845 词）", "#e0f2fe", 11)
    box(50, 50, 44, 8, "④ 扫描 K∈{4..12}：困惑度 + C_v 一致性", "#e0f2fe", 11)
    box(50, 37, 40, 8, "选定 K=10（C_v 峰值 0.417）", "#fde68a", 11)
    box(50, 24, 44, 8, "⑤ 训练最终 LDA（在线变分贝叶斯）", "#e0f2fe", 11)
    box(50, 13, 46, 7, "θ 文档-主题分布 ／ φ 主题-词分布", "#c7d2fe", 11)

    for x in (20, 50, 80):
        arrow(x, 87.5, 50, 80.5)
    arrow(50, 72, 50, 67)
    arrow(50, 59, 50, 54)
    arrow(50, 46, 50, 41)
    arrow(50, 33, 50, 28)
    arrow(50, 20, 50, 16.5)

    # 底部输出
    outs = ["选K曲线/主题词/词云", "主题间距离图\n(JS+MDS)", "主题×来源\n热力图",
            "每日主题构成", "作业2新词\n→主题 联动"]
    xs = [10, 30, 50, 70, 90]
    for x, t in zip(xs, outs):
        box(x, 3.5, 18, 7, t, "#f1f5f9", 8.5)
        arrow(50, 9.5, x, 7.2)

    fig.tight_layout()
    out = FIG / "pipeline_flow.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =============== 2. Markdown → ops ===============
ops: list[dict] = []
_tbl = 0


def _p(text, **props):
    props = {k: v for k, v in props.items() if v is not None}
    props["text"] = text
    ops.append({"op": "add", "parent": "/body", "type": "paragraph", "props": props})


def body_para(text):
    _p(text, size="12pt", lineSpacing="1.5x", firstLineIndent="480",
       **{"font.ea": CJK, "font.latin": LATIN}, spaceAfter="4pt")


def list_para(text):
    _p(text, size="12pt", lineSpacing="1.5x", indent="480", hangingIndent="480",
       **{"font.ea": CJK, "font.latin": LATIN}, spaceAfter="2pt")


def code_para(text):
    _p(text if text else " ", size="9pt", **{"font.latin": "Courier New", "font.ea": CJK},
       color="334155", spaceAfter="0pt", lineSpacing="1.0x")


def title(text):
    _p(text, align="center", size="22pt", bold="true",
       **{"font.ea": CJK_H, "font.latin": LATIN}, spaceBefore="40pt", spaceAfter="6pt")


def subtitle(text):
    _p(text.lstrip("—- "), align="center", size="15pt", bold="true",
       **{"font.ea": CJK_H, "font.latin": LATIN}, spaceAfter="26pt")


CENTER_H1 = {"摘要", "目录"}


def heading(text, level):
    if level == 1:
        brk = "true" if text in ("摘要", "目录", "第一章 绪论") else None
        _p(text, style="Heading1", size="16pt", bold="true",
           align=("center" if text in CENTER_H1 else "left"),
           **{"font.ea": CJK_H, "font.latin": LATIN},
           spaceBefore="16pt", spaceAfter="8pt", pageBreakBefore=brk, keepNext="true")
    else:
        _p(text, style="Heading2", size="14pt", bold="true",
           **{"font.ea": CJK_H, "font.latin": LATIN},
           spaceBefore="12pt", spaceAfter="6pt", keepNext="true")


def caption(text):
    _p(text, align="center", italic="true", size="10.5pt", color="555555",
       **{"font.ea": CJK, "font.latin": LATIN}, spaceBefore="2pt", spaceAfter="12pt")


def figure(src: Path, cap: str | None, width=14.5):
    _p("", align="center", spaceBefore="6pt")
    ops.append({"op": "add", "parent": "/body/p[last()]", "type": "picture",
                "props": {"src": str(src), "width": f"{width}cm"}})
    if cap:
        caption(cap)


def equation(src: Path):
    _p("", align="center", spaceBefore="4pt", spaceAfter="4pt")
    ops.append({"op": "add", "parent": "/body/p[last()]", "type": "picture",
                "props": {"src": str(src), "width": "12cm"}})


def table(rows: list[list[str]], colw_spec: str | None = None):
    """建表 → 补行 → set 填格。

    注意：officecli `add table`(带 colWidths) 会自动创建 1 行 ncol 个空单元格，
    `add row` 同样按 grid 宽自动补空格——因此必须用 `set` 写文字，
    而不是再 `add cell`（后者会在自动空格之后追加，造成列数膨胀/阶梯错位）。
    """
    global _tbl
    _tbl += 1
    ncol = max(len(r) for r in rows)
    R = len(rows)
    if colw_spec is None:
        w = 9200 // ncol
        colw_spec = ",".join([str(w)] * ncol)
    ops.append({"op": "add", "parent": "/body", "type": "table",
                "props": {"colWidths": colw_spec, "align": "center", "layout": "fixed",
                          "border.all": "single;4;94a3b8"}})
    tp = f"/body/tbl[{_tbl}]"
    for _ in range(R - 1):          # add table 自带第 1 行，再补 R-1 行
        ops.append({"op": "add", "parent": tp, "type": "row", "props": {}})
    for r, row in enumerate(rows, 1):
        for c in range(1, ncol + 1):
            text = row[c - 1] if c - 1 < len(row) else " "
            cp = {"text": text, "size": "10.5pt", "lineSpacing": "1.15x"}
            if r == 1:
                cp["bold"] = "true"
            ops.append({"op": "set", "path": f"{tp}/tr[{r}]/tc[{c}]", "props": cp})


# ---- 行内 Markdown / LaTeX 清洗 ----
_LATEX = {
    r"\theta": "θ", r"\varphi": "φ", r"\phi": "φ", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\times": "×", r"\dots": "…", r"\ldots": "…", r"\cdot": "·",
    r"\mid": "|", r"\Vert": "‖", r"\sum": "Σ", r"\prod": "∏", r"\log": "log",
    r"\,": " ", r"\;": " ", r"\le": "≤", r"\ge": "≥", r"\approx": "≈",
}


def clean(t: str) -> str:
    # 行内代码
    t = re.sub(r"`([^`]*)`", r"\1", t)
    # 加粗 / 斜体标记
    t = t.replace("**", "")
    t = re.sub(r"(?<!\*)\*(?!\*)", "", t)

    def _math(m):
        s = m.group(1)
        s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        for k, v in _LATEX.items():
            s = s.replace(k, v)
        s = s.replace("{", "").replace("}", "").replace("\\", "")
        return s

    t = re.sub(r"\$([^$]*)\$", _math, t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def split_table_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return [clean(c) for c in cells]


def parse(md_lines, eq_imgs, flow_img):
    i = 0
    title_done = subtitle_done = False
    eq_i = 0
    fig_cap_done = False
    while i < len(md_lines):
        raw = md_lines[i].rstrip("\n")
        s = raw.strip()
        if s == "" or s == "---":
            i += 1
            continue

        # 代码围栏
        if s.startswith("```"):
            is_mermaid = "mermaid" in s
            j = i + 1
            buf = []
            while j < len(md_lines) and not md_lines[j].strip().startswith("```"):
                buf.append(md_lines[j].rstrip("\n"))
                j += 1
            if is_mermaid:
                figure(flow_img, "图 4-1　LDA 主题挖掘总体流程（数据与分词联动作业 1、作业 2）", 16)
            else:
                for b in buf:
                    code_para(b)
            i = j + 1
            continue

        # 展示公式
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            if eq_i < len(eq_imgs):
                equation(eq_imgs[eq_i])
                eq_i += 1
            i += 1
            continue

        # 标题
        if s.startswith("#"):
            h = len(s) - len(s.lstrip("#"))
            text = s[h:].strip()
            if not title_done:
                title(text); title_done = True; i += 1; continue
            if not subtitle_done:
                subtitle(text); subtitle_done = True; i += 1; continue
            heading(clean(text), 1 if h == 2 else 2)
            i += 1
            continue

        # 图片
        mm = re.match(r"!\[(.*?)\]\((.*?)\)", s)
        if mm:
            path = mm.group(2)
            src = (ROOT / path).resolve()
            # 紧随的斜体行作为图注
            cap = None
            k = i + 1
            while k < len(md_lines) and md_lines[k].strip() == "":
                k += 1
            if k < len(md_lines):
                cs = md_lines[k].strip()
                if cs.startswith("*") and cs.endswith("*") and not cs.startswith("**"):
                    cap = clean(cs.strip("*").strip())
            figure(src, cap)
            i = (k + 1) if cap else (i + 1)
            continue

        # 表格
        if s.startswith("|") and s.endswith("|"):
            block = []
            while i < len(md_lines) and md_lines[i].strip().startswith("|"):
                block.append(md_lines[i].strip())
                i += 1
            rows = []
            for bi, bl in enumerate(block):
                if re.match(r"^\|[\s:\-|]+\|$", bl):   # 分隔行
                    continue
                rows.append(split_table_row(bl))
            if rows:
                table(rows)
            continue

        # 引用块（说明）
        if s.startswith(">"):
            body_para(clean(s.lstrip(">").strip()))
            i += 1
            continue

        # 列表
        if re.match(r"^(\-|\*|\d+\.)\s+", s):
            marker = re.match(r"^(\-|\*|\d+\.)\s+", s).group(1)
            txt = re.sub(r"^(\-|\*|\d+\.)\s+", "", s)
            prefix = "• " if marker in ("-", "*") else (marker + " ")
            list_para(prefix + clean(txt))
            i += 1
            continue

        # 普通段落
        body_para(clean(s))
        i += 1


# =============== 3. 运行 ===============
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL:", " ".join(cmd[:4]), "...\n", r.stdout[-800:], r.stderr[-800:])
    return r


def main():
    print("渲染公式与流程图 ...")
    eq_imgs = render_equations()
    flow_img = render_flowchart()

    print("解析 REPORT.md ...")
    md = REPORT.read_text(encoding="utf-8").splitlines(keepends=True)
    parse(md, eq_imgs, flow_img)
    print(f"  生成 {len(ops)} 个 batch 操作")

    if OUT.exists():
        OUT.unlink()
    run(["officecli", "create", str(OUT)])
    # 文档默认字体：拉丁 Times New Roman + 东亚 宋体 + 12pt（单元格等继承）
    run(["officecli", "set", str(OUT), "/",
         "--prop", f"docDefaults.font={LATIN}",
         "--prop", f"docDefaults.font.eastAsia={CJK}",
         "--prop", "docDefaults.fontSize=12pt"])

    ops_file = ROOT / "data" / "_docx_ops.json"
    ops_file.write_text(json.dumps(ops, ensure_ascii=False), encoding="utf-8")
    print("执行 officecli batch ...")
    r = run(["officecli", "batch", str(OUT), "--input", str(ops_file), "--json"])
    try:
        summary = json.loads(r.stdout)["data"]["Summary"]
        print("  batch:", summary)
    except Exception:
        print(r.stdout[-500:])

    # 页脚页码
    run(["officecli", "add", str(OUT), "/", "--type", "footer",
         "--prop", "type=default", "--prop", "align=center",
         "--prop", "size=10pt", "--prop", "field=page"])
    print("validate ...")
    v = run(["officecli", "validate", str(OUT)])
    print(v.stdout[-300:])
    print("完成 ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
