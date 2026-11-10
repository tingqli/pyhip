#!/usr/bin/env python3
"""Draws the figures of docs/moe_942_ep8/collectives_wiki.md.

Usage: python collectives.py <dir with NotoSansSC-{Regular,Medium,Bold}.otf from notofonts/noto-cjk Sans/SubsetOTF/SC>
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

for weight in ("Regular", "Medium", "Bold"):
    font_manager.fontManager.addfont(os.path.join(sys.argv[1], f"NotoSansSC-{weight}.otf"))
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["hatch.linewidth"] = 0.8
OUT = os.path.dirname(os.path.abspath(__file__))

C = ["#2E75B6", "#ED7D31", "#70AD47", "#7030A0"]  # data owned by GPU0..GPU3
DARK, SUB, EDGE, RED = "#262626", "#595959", "#BFBFBF", "#C00000"
P = 4


def slide(width, height):
    fig = plt.figure(figsize=(width / 10, height / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec="none", lw=1.2, r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def rect(ax, x, y, w, h, fc, ec="none", lw=0.0, z=3, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def text(ax, x, y, s, size=12, color=DARK, ha="center", va="center", weight="normal", z=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z, linespacing=1.35, fontweight=weight)


def arrow(ax, p, q, color=DARK, lw=1.6, rad=0.0, scale=14, z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def title(ax, height, main, sub):
    text(ax, 2, height - 3.0, main, size=20, ha="left", weight="bold")
    text(ax, 2, height - 6.6, sub, size=12, color=SUB, ha="left")


def cell(ax, x, y, w, h, kind, label="", color=None):
    """One chunk slot: 'empty', 'own' (one GPU's data) or 'sum' (all GPUs added)."""
    if kind == "empty":
        rect(ax, x, y, w, h, fc="white", ec=EDGE, lw=1.0, ls="--")
        return
    if kind == "own":
        rect(ax, x, y, w, h, fc=color, ec="white", lw=1.2)
    else:
        for i in range(P):
            rect(ax, x + i * w / P, y, w / P, h, fc=C[i])
        rect(ax, x, y, w, h, fc="none", ec="white", lw=1.2, z=4)
    text(ax, x + w / 2, y + h / 2, label, size=10.5, color="white", weight="bold")


def gpu(ax, x, top, g, slots, w=11.0, sh=3.4):
    """GPU g: a header and four chunk slots stacked downwards."""
    box(ax, x, top - 3.2, w, 3.2, fc=C[g], r=0.5)
    text(ax, x + w / 2, top - 1.6, f"GPU{g}", size=11, color="white", weight="bold")
    for j, (kind, label, color) in enumerate(slots):
        cell(ax, x + 0.4, top - 3.6 - (j + 1) * (sh + 0.3), w - 0.8, sh, kind, label, color)


def overview():
    W, H = 160.0, 116.0
    fig, ax = slide(W, H)
    title(ax, H, "四种集合通信：之前 → 之后（4 张卡为例）",
          "每张卡的数据切成 4 段；颜色 = 数据来自哪张卡；彩条 Σ = 4 张卡的同一段相加")
    text(ax, 55.5, 104.5, "之前", size=14, weight="bold", color=SUB)
    text(ax, 120.5, 104.5, "之后", size=14, weight="bold", color=SUB)
    own = lambda g, label: ("own", label, C[g])  # noqa: E731
    rows = [
        ("all-gather", "每卡交出自己的 1 块\n每卡最后都有全部 4 块\n每卡发 (p−1)/p × N",
         lambda g, j: own(j, f"块{j}") if j == g else ("empty", "", None),
         lambda g, j: own(j, f"块{j}")),
        ("reduce-scatter", "各卡的同一段相加\n每卡只留 1 段的总和\n每卡发 (p−1)/p × N",
         lambda g, j: own(g, f"段{j}"),
         lambda g, j: ("sum", f"Σ段{j}", None) if j == g else ("empty", "", None)),
        ("all-reduce", "各卡的同一段相加\n每卡都有全部总和\n每卡发 2(p−1)/p × N",
         lambda g, j: own(g, f"段{j}"),
         lambda g, j: ("sum", f"Σ段{j}", None)),
        ("all-to-all", "第 j 块发给 GPU j\n（段矩阵转置）\n每卡发 (p−1)/p × N",
         lambda g, j: own(g, f"{g}→{j}"),
         lambda g, j: own(j, f"{j}→{g}")),
    ]
    for r, (name, desc, before, after) in enumerate(rows):
        top = 102.0 - r * 25.0
        box(ax, 1.5, top - 21.0, 30.0, 21.0, fc="#F2F2F2", ec=EDGE, lw=1.0)
        text(ax, 16.5, top - 3.6, name, size=15, weight="bold")
        text(ax, 16.5, top - 12.8, desc, size=10.5, color=SUB)
        for g in range(P):
            gpu(ax, 34.0 + 11.5 * g, top, g, [before(g, j) for j in range(P)])
            gpu(ax, 99.0 + 11.5 * g, top, g, [after(g, j) for j in range(P)])
        arrow(ax, (82.5, top - 11.5), (96.0, top - 11.5), lw=2.4, scale=22)
    text(ax, 2, 3.0, "N = 每卡的完整数据量（all-gather 指收齐之后的大小）；每卡发出和收到的字节数相同。",
         size=11, color=SUB, ha="left")
    fig.savefig(os.path.join(OUT, "coll_overview.png"), dpi=110)


def ring_panel(ax, top, name, rule, kind):
    """Ring table: at step s GPU r sends segment (r - s) mod 4 (reduce-scatter) or (r - s + 1) mod 4 (all-gather)."""
    box(ax, 1.5, top - 33.0, 157.0, 33.0, fc="#FAFAFA", ec=EDGE, lw=1.0, r=1.2, z=1)
    text(ax, 3.5, top - 3.2, name, size=15, ha="left", weight="bold")
    text(ax, 3.5, top - 7.0, rule, size=11, ha="left", color=SUB)
    # Ring 0 -> 1 -> 2 -> 3 -> 0.
    pos = {0: (6.0, top - 17.0), 1: (22.0, top - 17.0), 2: (22.0, top - 28.0), 3: (6.0, top - 28.0)}
    for g, (x, y) in pos.items():
        box(ax, x, y, 9.0, 5.0, fc=C[g], r=0.6, z=3)
        text(ax, x + 4.5, y + 2.5, f"GPU{g}", size=11, color="white", weight="bold")
    arrow(ax, (15.3, top - 14.5), (21.7, top - 14.5))
    arrow(ax, (26.5, top - 17.3), (26.5, top - 22.7))
    arrow(ax, (21.7, top - 25.5), (15.3, top - 25.5))
    arrow(ax, (10.5, top - 22.7), (10.5, top - 17.3))
    # Step table.
    x0, cw, rh = 40.0, 26.3, 5.0
    heads = ["第 1 步", "第 2 步", "第 3 步", "结果"]
    for c, head in enumerate(heads):
        text(ax, x0 + 11.0 + c * cw + cw / 2, top - 10.3, head, size=11.5, weight="bold", color=SUB)
    shades = ["#DEEBF7", "#9DC3E6", "#5B9BD5"] if kind == "rs" else ["#E2F0D9", "#C5E0B4", "#A9D18E"]
    for g in range(P):
        y = top - 13.5 - (g + 1) * rh
        box(ax, x0, y + 0.3, 10.0, rh - 0.6, fc=C[g], r=0.5)
        text(ax, x0 + 5.0, y + rh / 2, f"GPU{g}", size=10.5, color="white", weight="bold")
        for s in range(1, P):
            chunk = (g - s) % P if kind == "rs" else (g - s + 1) % P
            label = f"发 段{chunk}（{s} 份的和）→ GPU{(g + 1) % P}" if kind == "rs" else f"发 段{chunk} → GPU{(g + 1) % P}"
            x = x0 + 11.0 + (s - 1) * cw
            rect(ax, x + 0.3, y + 0.3, cw - 0.6, rh - 0.6, fc=shades[s - 1])
            text(ax, x + cw / 2, y + rh / 2, label, size=10, color=DARK if s < 3 or kind != "rs" else "white")
        x = x0 + 11.0 + 3 * cw
        if kind == "rs":
            cell(ax, x + 0.3, y + 0.3, cw - 0.6, rh - 0.6, "sum", f"段{g} 的完整和")
        else:
            rect(ax, x + 0.3, y + 0.3, cw - 0.6, rh - 0.6, fc="#548235")
            text(ax, x + cw / 2, y + rh / 2, "段 0–3 全有", size=10.5, color="white", weight="bold")


def ring():
    W, H = 160.0, 84.0
    fig, ax = slide(W, H)
    title(ax, H, "环形（ring）算法：每步每卡只给下一张卡发 1 段",
          "4 张卡同时发，p − 1 = 3 步完成；每卡共发 3 × N/4 = (p−1)/p × N")
    ring_panel(ax, 75.0, "reduce-scatter", "收到的段加上自己那段，再传给下一张卡；3 步后每卡有 1 段的完整和", "rs")
    ring_panel(ax, 39.0, "all-gather", "收到的段直接存下，再传给下一张卡；3 步后每卡有全部 4 段", "ag")
    text(ax, 2, 2.6, "reduce-scatter 第 s 步 GPU r 发出段 (r − s) mod 4；all-gather 发出段 (r − s + 1) mod 4。"
                     "all-reduce = 先做上表，再做下表。", size=11, color=SUB, ha="left")
    fig.savefig(os.path.join(OUT, "coll_ring.png"), dpi=110)


def allreduce():
    W, H = 160.0, 44.0
    fig, ax = slide(W, H)
    title(ax, H, "all-reduce = reduce-scatter + all-gather",
          "每卡发送：(p−1)/p × N + (p−1)/p × N = 2(p−1)/p × N")
    stages = [
        ("开始：每卡有完整的部分和", lambda g, j: ("own", f"段{j}", C[g])),
        ("reduce-scatter 之后：每卡 1 段总和", lambda g, j: ("sum", f"Σ{j}", None) if j == g else ("empty", "", None)),
        ("all-gather 之后：每卡都有全部总和", lambda g, j: ("sum", f"Σ{j}", None)),
    ]
    xs = [2.0, 58.0, 114.0]
    for (label, spec), x in zip(stages, xs):
        text(ax, x + 22.0, 34.0, label, size=12, weight="bold")
        for g in range(P):
            gpu(ax, x + g * 11.2, 31.5, g, [spec(g, j) for j in range(P)], w=10.6, sh=3.6)
    arrow(ax, (47.5, 19.0), (57.0, 19.0), lw=2.4, scale=22)
    arrow(ax, (103.5, 19.0), (113.0, 19.0), lw=2.4, scale=22)
    text(ax, 52.3, 23.0, "RS", size=11, weight="bold", color=SUB)
    text(ax, 108.3, 23.0, "AG", size=11, weight="bold", color=SUB)
    box(ax, 2.0, 1.5, 156.0, 6.5, fc="#FFF2CC", ec="#BF9000", lw=1.0)
    text(ax, 80.0, 4.75, "例：p = 8，N = [32768, 4096] BF16 = 256 MiB → 每卡发 2 × 7/8 × 256 = 448 MiB（本文 EP8 模式 B 和 TP8 每层两次）",
         size=12)
    fig.savefig(os.path.join(OUT, "coll_allreduce.png"), dpi=110)


def alltoall():
    W, H = 160.0, 64.0
    fig, ax = slide(W, H)
    title(ax, H, "all-to-all：第 j 块发给 GPU j",
          "左：等大小（4 张卡）；右：MoE dispatch 是 all-to-all-v，每块大小由路由决定")
    cw, ch, x_before, x_after, top = 8.2, 4.6, 12.0, 60.0, 50.0
    for label, x0 in (("之前：行 = 持有的 GPU", x_before), ("之后：行 = 持有的 GPU", x_after)):
        text(ax, x0 + 2 * cw, top + 3.0, label, size=11.5, weight="bold", color=SUB)
    for r in range(P):
        y = top - (r + 1) * ch
        for x0 in (x_before, x_after):
            text(ax, x0 - 4.0, y + ch / 2, f"GPU{r}", size=10.5, color=C[r], weight="bold")
        for j in range(P):
            src, dst = (r, j)
            rect(ax, x_before + j * cw, y, cw, ch, fc=C[src], ec="white", lw=1.2)
            text(ax, x_before + j * cw + cw / 2, y + ch / 2, f"{src}→{dst}", size=10.5, color="white", weight="bold")
            src, dst = (j, r)
            rect(ax, x_after + j * cw, y, cw, ch, fc=C[src], ec="white", lw=1.2)
            text(ax, x_after + j * cw + cw / 2, y + ch / 2, f"{src}→{dst}", size=10.5, color="white", weight="bold")
    # Highlight 1->2 moving from GPU1's row to GPU2's row.
    rect(ax, x_before + 2 * cw, top - 2 * ch, cw, ch, fc="none", ec=RED, lw=2.6, z=5)
    rect(ax, x_after + 1 * cw, top - 3 * ch, cw, ch, fc="none", ec=RED, lw=2.6, z=5)
    text(ax, 46.0, 25.5, "红框：GPU1 的第 2 块（1→2）到了 GPU2，放在第 1 位", size=11, color=RED)
    text(ax, 46.0, 18.5, "所有块同时发：全互联上一步完成\n每卡发 3 块（自己那块不走链路）= (p−1)/p × N",
         size=11, color=SUB)
    # MoE dispatch from GPU0 (8 GPUs, all-to-all-v).
    sends = [2700, 2666, 2774, 2669, 2729, 2669, 2743, 2676]
    x0, scale = 112.0, 34.0 / 2800
    text(ax, x0 + 20.0, top + 3.0, "例：EP8 MORI dispatch，GPU0 发出", size=11.5, weight="bold", color=SUB)
    colors = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#843C0C", "#C55A11", "#ED7D31", "#F4B183"]
    for r, n in enumerate(sends):
        y = top - 1.0 - (r + 1) * 4.2
        text(ax, x0 - 1.0, y + 1.7, f"→GPU{r}", size=10.5, ha="right", color=DARK)
        rect(ax, x0, y + 0.3, n * scale, 2.8, fc=colors[r] if r else "#BFBFBF")
        text(ax, x0 + n * scale + 0.8, y + 1.7, f"{n}" + ("（本卡）" if r == 0 else ""), size=10, ha="left")
    text(ax, x0 + 20.0, 4.5, "4096 个 token 发出 21626 份：一个 token 发给\n它的 expert 所在的每张卡（平均 5.3 张）",
         size=10.5, color=SUB)
    fig.savefig(os.path.join(OUT, "coll_alltoall.png"), dpi=110)


def volume():
    W, H = 160.0, 46.0
    fig, ax = slide(W, H)
    title(ax, H, "对应到 MoE prefill：每卡每层发出的数据量（MiB）",
          "32768 token，H = 4096，attention DP2 × TP4；只计发送，不计发给自己的部分")
    kinds = {"RS": ("reduce-scatter", "#2E75B6"), "D": ("all-to-all dispatch", "#ED7D31"),
             "Cb": ("all-to-all combine", "#F4B183"), "AG": ("all-gather", "#70AD47"), "AR": ("all-reduce", "#7030A0")}
    rows = [("EP8 + MORI（模式 A）", [("RS", 96), ("D", 77.4), ("Cb", 148.2), ("AG", 49.5)]),
            ("EP8，不用 MORI（模式 B）", [("AR", 448), ("AR", 448)]),
            ("MoE TP8", [("AR", 448), ("AR", 448)])]
    x0, scale = 40.0, 105.0 / 896
    for i, (name, parts) in enumerate(rows):
        y = 28.0 - i * 8.0
        text(ax, x0 - 1.5, y + 2.5, name, size=12, ha="right", weight="bold")
        x = x0
        for key, value in parts:
            w = value * scale
            rect(ax, x, y, w, 5.0, fc=kinds[key][1], ec="white", lw=1.2)
            if w > 4:
                text(ax, x + w / 2, y + 2.5, f"{value:g}", size=10.5 if w > 9 else 9.5, color="white", weight="bold")
            x += w
        text(ax, x + 1.2, y + 2.5, f"合计 {sum(v for _, v in parts):.0f}", size=12, ha="left", weight="bold")
    x = 2.0
    for key in ("RS", "D", "Cb", "AG", "AR"):
        rect(ax, x, 3.0, 3.0, 2.2, fc=kinds[key][1])
        text(ax, x + 3.8, 4.1, kinds[key][0], size=10.5, ha="left", color=SUB)
        x += 31.0
    fig.savefig(os.path.join(OUT, "coll_moe_volume.png"), dpi=110)


overview()
ring()
allreduce()
alltoall()
volume()
