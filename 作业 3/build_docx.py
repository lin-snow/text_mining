# -*- coding: utf-8 -*-
"""把 REPORT.md 渲染为可打印的作业 3 Word 文档（毕设格式）。

步骤：
  1. 用 matplotlib 渲染 3 张展示公式图 + 1 张总体流程图（离线，无需 pandoc/mermaid）；
  2. 解析 REPORT.md（标题/段落/表格/图片/公式/流程图/列表/引用）为 officecli batch 操作；
  3. 调用 officecli 生成 作业3.docx，并 validate。

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
OUT = ROOT / "作业3.docx"

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
# 顺序须与 REPORT.md 中 $$...$$ 展示公式块出现顺序一致（共 3 个）。
EQUATIONS = [
    r"$A \approx A_k = U_k \Sigma_k V_k^{\top}\qquad(\text{截断 SVD，秩-}k\text{ 最佳逼近})$",
    r"$\mathrm{sim}(d_i, d_j) = \dfrac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\|\,\|\mathbf{x}_j\|}\qquad(\text{cosine 相似度})$",
    r"$\mathrm{NAP}(w) = \dfrac{\overline{\mathrm{sim}}_{\mathrm{intra}}(D_w)}{\overline{\mathrm{sim}}_{\mathrm{global}}} - 1\qquad(\text{新词推荐锚定力})$",
]


def render_equations():
    paths = []
    for idx, eq in enumerate(EQUATIONS, 1):
        fig = plt.figure(figsize=(9.2, 1.0))
        fig.text(0.5, 0.5, eq, ha="center", va="center", fontsize=18)
        out = FIG / f"eq_{idx}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.18,
                    facecolor="white")
        plt.close(fig)
        paths.append(out)
    return paths


def render_flowchart():
    fig, ax = plt.subplots(figsize=(11, 8.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc, fs=10.5, ec="#334155"):
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
    box(20, 93, 26, 8, "作业1 RSS 语料\n221→220 篇·含来源/时间", "#dbeafe", 9.5)
    box(50, 93, 24, 8, "作业2 定制词典\n188 词", "#dcfce7", 9.5)
    box(80, 93, 24, 8, "三级停用词\n709 个", "#fef9c3", 9.5)
    # 主干
    box(50, 79, 42, 7.5, "jieba 分词 + good_token 过滤", "#e0f2fe", 11)
    box(50, 67, 48, 7.5, "Dictionary + filter_extremes → TF-IDF（1897 维）", "#e0f2fe", 10.5)
    # 两路降维
    box(27, 54, 34, 8, "LSA/LSI：截断 SVD\n→ k 维语义向量", "#ede9fe", 10)
    box(73, 54, 34, 8, "LDA：θ 文档-主题\n→ K=10 维", "#ede9fe", 10)
    box(27, 42, 34, 7, "按推荐质量选 k*=80", "#fde68a", 10.5)
    # 推荐
    box(50, 29, 58, 8, "推荐召回：精确 cosine KNN\n+ 近似最近邻（recall@10=0.995）", "#c7d2fe", 10.5)

    for x in (20, 50, 80):
        arrow(x, 89, 50, 83)
    arrow(50, 75.2, 50, 71)
    arrow(50, 63.2, 32, 58.2)
    arrow(50, 63.2, 68, 58.2)
    arrow(27, 50, 27, 45.5)
    arrow(27, 38.5, 45, 33.2)
    arrow(73, 50, 60, 33.2)

    # 底部输出
    outs = ["推荐表 /\n邻域连线图", "PCA·t-SNE·UMAP\n可视化", "新词锚定力 NAP\n+ 消融实验"]
    xs = [20, 50, 80]
    for x, t in zip(xs, outs):
        box(x, 9, 24, 8, t, "#f1f5f9", 9.5)
        arrow(50, 25, x, 13.2)

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
                "props": {"src": str(src), "width": "12.5cm"}})


def table(rows: list[list[str]], colw_spec: str | None = None):
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
    r"\gamma": "γ", r"\Sigma": "Σ", r"\sigma": "σ", r"\Delta": "Δ", r"\times": "×",
    r"\dots": "…", r"\ldots": "…", r"\cdots": "⋯", r"\cdot": "·", r"\top": "⊤",
    r"\in": "∈", r"\mid": "|", r"\Vert": "‖", r"\lVert": "‖", r"\rVert": "‖",
    r"\langle": "⟨", r"\rangle": "⟩", r"\sum": "Σ", r"\prod": "∏", r"\log": "log",
    r"\,": " ", r"\;": " ", r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\star": "*",
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
        s = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\overline\{([^}]*)\}", r"\1", s)
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
                figure(flow_img, "图 4-1　推荐系统总体流程（降维→召回→评估，联动作业 1、作业 2）", 16)
            else:
                for b in buf:
                    code_para(b)
            i = j + 1
            continue

        # 展示公式：支持独占行的 $$ 围栏（多行）与单行 $$...$$
        if s.startswith("$$"):
            if s == "$$":                       # 多行围栏：吃到配对的 $$
                j = i + 1
                while j < len(md_lines) and md_lines[j].strip() != "$$":
                    j += 1
                if eq_i < len(eq_imgs):
                    equation(eq_imgs[eq_i]); eq_i += 1
                i = j + 1
                continue
            if s.endswith("$$") and len(s) > 4:  # 单行 $$...$$
                if eq_i < len(eq_imgs):
                    equation(eq_imgs[eq_i]); eq_i += 1
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
            for bl in block:
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

    run(["officecli", "add", str(OUT), "/", "--type", "footer",
         "--prop", "type=default", "--prop", "align=center",
         "--prop", "size=10pt", "--prop", "field=page"])
    print("validate ...")
    v = run(["officecli", "validate", str(OUT)])
    print(v.stdout[-300:])
    print("完成 ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
