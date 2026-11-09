#!/usr/bin/env python3
"""Draws the EP8 mode A / mode B figures of docs/moe_ep8_prefill_wiki.md.

Usage: python ep8_modes.py <dir with NotoSansSC-{Regular,Medium,Bold}.otf from notofonts/noto-cjk Sans/SubsetOTF/SC>
Numbers are the wiki's scenario (32768 tokens, uniform top-8, seed 0, GPU0).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

for weight in ("Regular", "Medium", "Bold"):
    font_manager.fontManager.addfont(os.path.join(sys.argv[1], f"NotoSansSC-{weight}.otf"))
plt.rcParams["font.family"] = "Noto Sans SC"
OUT = os.path.dirname(os.path.abspath(__file__))

RECV = [2700, 2715, 2714, 2652, 2694, 2766, 2705, 2721]  # tokens GPU0 receives from GPU r
SEND = [2700, 2666, 2774, 2669, 2729, 2669, 2743, 2676]  # tokens GPU0 sends to GPU r
N, M, CAP = 21667, 131072, 16384

GPU = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#843C0C", "#C55A11", "#ED7D31", "#F4B183"]
ON = ["white", "white", "white", "#1F1F1F", "white", "white", "white", "#1F1F1F"]
DP_FILL, DP_EDGE = ["#EAF1FB", "#FDF1E7"], ["#2E75B6", "#C55A11"]
GRAY, DARK, SUB, RED = "#E7E6E6", "#262626", "#595959", "#C00000"
BX, BW = 23.0, 135.0  # x range of the content column


def slide(height):
    fig = plt.figure(figsize=(16, height / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc, ec="none", lw=1.2, r=0.8, z=2, ls="-"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    )


def rect(ax, x, y, w, h, fc, ec="none", lw=0.0, hatch=None, z=3, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=z, ls=ls))


def text(ax, x, y, s, size=12, color=DARK, ha="center", va="center", z=6, weight="normal", bg=None):
    bbox = dict(fc=bg, ec="none", pad=1.5, boxstyle="round,pad=0.25") if bg else None
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z, linespacing=1.35, fontweight=weight, bbox=bbox)


def arrow(ax, p, q, color=SUB, lw=1.4, rad=0.0, scale=12, z=4):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z)
    )


def title(ax, height, main, sub):
    text(ax, 2, height - 3.2, main, size=22, ha="left", weight="bold")
    text(ax, 2, height - 7.2, sub, size=12.5, color=SUB, ha="left")


def step(ax, y, h, head, sub):
    box(ax, 1.5, y, 19.5, h, fc="#F2F2F2", ec="#BFBFBF", lw=1.0)
    text(ax, 11.25, y + h - 2.6, head, size=14, weight="bold")
    text(ax, 11.25, y + (h - 4.2) / 2, sub, size=10.5, color=SUB)


def gpu_x(r):
    return 24.5 + 16 * r + (5.0 if r >= 4 else 0.0)


def gpu_row(ax, y, labels, frames, h=10.0, mini=False):
    for d in range(2):
        box(ax, BX + 69 * d, y - 1.5, 66, h + 5.2, fc=DP_FILL[d], ec=DP_EDGE[d], lw=1.2, r=1.2, z=1)
        text(ax, BX + 69 * d + 33, y + h + 1.85, frames[d], size=11.5, color=DP_EDGE[d], weight="medium")
    for r in range(8):
        x = gpu_x(r)
        box(ax, x, y, 14.5, h, fc=GPU[r], ec=RED if r == 0 else "none", lw=3.0, r=0.9)
        box(ax, x + 0.45, y + 0.45, 13.6, h - 3.3, fc="white", r=0.6, z=3)
        text(ax, x + 7.25, y + h - 1.45, f"GPU{r}", size=11.5, color=ON[r], weight="bold")
        if mini:
            text(ax, x + 7.25, y + 0.45 + (h - 3.3) * 0.64, labels[r], size=12.5, weight="medium")
            rect(ax, x + 1.4, y + 1.3, 5.85, 1.5, fc=GPU[2], z=4)
            rect(ax, x + 7.25, y + 1.3, 5.85, 1.5, fc=GPU[6], z=4)
        else:
            text(ax, x + 7.25, y + 0.45 + (h - 3.3) / 2, labels[r], size=13, weight="medium")


def flow(ax, y, h, items):
    """Boxes left to right inside the content column, joined by arrows."""
    gap = 4.5
    w = (BW - gap * (len(items) - 1)) / len(items)
    for i, (s, fc) in enumerate(items):
        x = BX + i * (w + gap)
        box(ax, x, y, w, h, fc=fc, ec="#A6A6A6", lw=1.0)
        text(ax, x + w / 2, y + h / 2, s, size=12)
        if i:
            arrow(ax, (x - gap + 0.4, y + h / 2), (x - 0.4, y + h / 2), color=DARK, lw=1.8, scale=16)


def mode_a():
    H = 100.0
    fig, ax = slide(H)
    title(ax, H, "模式 A：all2all（MORI）",
          "一步 prefill 共 32768 个 token（DP2 × attention TP4，EP8），均匀随机路由；以 GPU0 为例（红框）")

    step(ax, 71.5, 16, "① attention 之后", "DP 组内 4 卡\nreduce-scatter")
    gpu_row(ax, 73, ["4096 token"] * 8, ["DP0：attention 算 16384 个 token", "DP1：attention 算 16384 个 token"])

    # GPU0 dispatch input (MORI reads it in place) and the copies it sends to every GPU.
    y = 62.5
    step(ax, 49.5, 19.5, "② GPU0 发送", "发送输入就是\n本卡的 4096 个 token\nMORI 直接读取\n\n共发出 21626 份")
    rect(ax, BX, y, BW, 5, fc="white", ec="#A6A6A6", lw=1.2, z=2, ls="--")
    used = BW * 4096 / CAP
    rect(ax, BX, y, used, 5, fc=GPU[0])
    text(ax, BX + used / 2, y + 2.5, "发送输入：本卡 4096 个 token", size=12, color="white", weight="medium")
    text(ax, BX + used + (BW - used) / 2, y + 2.5, "容量上限 16384 行：每卡最多发这么多，决定接收 buffer = 8 × 16384（见 3.3 节）",
         size=11.5, color="#7F7F7F")
    for r in range(8):
        x = gpu_x(r)
        box(ax, x, 51, 14.5, 5, fc=GPU[r], r=0.6, z=3)
        text(ax, x + 7.25, 53.5, f"→GPU{r}  {SEND[r]}", size=11, color=ON[r])
        arrow(ax, (BX + used / 2, y), (x + 7.25, 56.1), color="#8C8C8C", lw=1.1, scale=10)
    text(ax, 158, 48.6, "每个 token 发给有它 expert 的每张卡（同一张卡只发一份），平均 5.3 份", size=11, color=SUB, ha="right")

    # GPU0 receive buffer, with the valid rows zoomed in.
    y, zy, zh = 38.5, 24.5, 6.5
    step(ax, 21.5, 24.5, "③ GPU0 接收", "MORI 接收 buffer\nM = 8 × 16384\n= 131072 行")
    rect(ax, BX, y, BW, 5, fc=GRAY, ec="#A6A6A6", lw=1.0, z=2)
    valid, x = BW * N / M, BX
    for r in range(8):
        rect(ax, x, y, BW * RECV[r] / M, 5, fc=GPU[r])
        x += BW * RECV[r] / M
    text(ax, BX + valid / 2 + 1, y + 6.5, "有效 21667 行（16.5%）", size=11.5, color=RED)
    text(ax, BX + valid + (BW - valid) / 2, y + 2.5, "无效尾部 109405 行（83.5%）：kernel 不读、不写", size=12, color="#7F7F7F")
    ax.add_patch(Polygon([(BX, y), (BX + valid, y), (BX + BW, zy + zh), (BX, zy + zh)], closed=True,
                         fc="#FBE5D6", ec="none", alpha=0.7, zorder=1))
    ax.plot([BX + valid, BX + BW], [y, zy + zh], ls="--", lw=1.0, color="#BFBFBF", zorder=2)
    x = BX
    for r in range(8):
        w = BW * RECV[r] / N
        rect(ax, x, zy, w, zh, fc=GPU[r], ec="white", lw=1.2)
        text(ax, x + w / 2, zy + zh / 2, f"来自 GPU{r}\n{RECV[r]}", size=10.5, color=ON[r])
        x += w
    text(ax, BX, zy - 2.3, "num_local_tokens = [21667]：只有这些行有效（上面放大的部分）；每个 token 在 GPU0 只出现一次",
         size=11, color=SUB, ha="left")

    step(ax, 4.5, 13, "④ MoE ⑤ combine", "只算本卡 32 个 expert")
    flow(ax, 5, 12, [("MoE 输入：21667 个 token\n带 32644 条本地路由", "#DEEBF7"),
                     ("moe_sorting：142 块 × 256 行\nGateUp → Down", "#DEEBF7"),
                     ("combine（BF16）：送回来源卡并相加\n每卡拿回自己的 4096 个 token", "#FBE5D6")])
    fig.savefig(os.path.join(OUT, "ep8_mode_a_all2all.png"), dpi=110)


def mode_b():
    H = 100.0
    fig, ax = slide(H)
    title(ax, H, "模式 B：EP（all-reduce 汇总全部 token）",
          "同一步 prefill：32768 个 token，均匀随机路由；以 GPU0 为例（红框）")

    step(ax, 71.5, 16, "① attention 之后", "8 卡 all-reduce\n= TP 求和\n+ 两个 DP 组汇总")
    gpu_row(ax, 73, ["32768 token"] * 8, ["DP0：attention 算 16384 个 token", "DP1：attention 算 16384 个 token"],
            mini=True)
    text(ax, 94, H - 7.2, "GPU 里的小条：每张卡都有两组的全部 token", size=11, color=SUB, ha="left")
    rect(ax, 127, H - 8.0, 3, 1.6, fc=GPU[2], z=4)
    text(ax, 130.8, H - 7.2, "DP0 的 16384", size=11, color=SUB, ha="left")
    rect(ax, 142, H - 8.0, 3, 1.6, fc=GPU[6], z=4)
    text(ax, 145.8, H - 7.2, "DP1 的 16384", size=11, color=SUB, ha="left")

    # GPU0 MoE input: no MORI, every row is a real token, a third of them without a local expert.
    y = 52
    step(ax, 47.5, 20, "② GPU0 MoE 输入", "没有 MORI\n没有发送/接收 buffer\nM = n = 32768")
    half = BW / 2
    rect(ax, BX, y + 8.5, half, 3, fc=GPU[2], ec="white", lw=1.0)
    rect(ax, BX + half, y + 8.5, half, 3, fc=GPU[6], ec="white", lw=1.0)
    text(ax, BX + half / 2, y + 10, "来自 DP0：16384 行", size=11, color="white")
    text(ax, BX + 1.5 * half, y + 10, "来自 DP1：16384 行", size=11, color="white")
    local = BW * N / 32768
    rect(ax, BX, y, local, 6.5, fc=GPU[0])
    rect(ax, BX + local, y, BW - local, 6.5, fc="#F2F2F2", ec="#A6A6A6", lw=0.8, hatch="///")
    text(ax, BX + local / 2, y + 3.25, "有本地路由：21667 个 token → 正常计算", size=12.5, color="white", weight="medium")
    text(ax, BX + local + (BW - local) / 2, y + 3.25, "没有本地路由：11101 个\n→ 输出 0", size=12, color=DARK, bg="white")
    text(ax, BX, y - 2.3, "两类 token 在行里是交错分布的，这里按数量画；不传 num_local_tokens，32768 行全部有效",
         size=11, color=SUB, ha="left")

    step(ax, 30.5, 13, "③ MoE", "只算本卡 32 个 expert")
    flow(ax, 31, 12, [("MoE 输入：32768 个 token\n其中 21667 个带 32644 条本地路由", "#DEEBF7"),
                      ("moe_sorting：142 块 × 256 行\nGateUp → Down（和模式 A 相同）", "#DEEBF7"),
                      ("输出 [32768, 4096] 部分和\n11101 行为 0", "#FBE5D6")])

    step(ax, 4.5, 20.5, "④ MoE 之后", "8 卡 all-reduce\n[32768, 4096]\n再 dp_scatter")
    gpu_row(ax, 6, ["16384 token"] * 8, ["DP0 的 4 张卡：取回 DP0 的 16384 行", "DP1 的 4 张卡：取回 DP1 的 16384 行"], h=9.0)
    fig.savefig(os.path.join(OUT, "ep8_mode_b_ep.png"), dpi=110)


def compare():
    H = 52.0
    fig, ax = slide(H)
    title(ax, H, "两种模式对比（GPU0，同一比例尺）", "A = all2all（MORI），B = EP（all-reduce）")
    x0, width = 40.0, 104.0

    def row(y, head, bars, unit):
        box(ax, 1.5, y - 1.0, 35.5, 9.0, fc="#F2F2F2", ec="#BFBFBF", lw=1.0)
        text(ax, 19.25, y + 3.5, head, size=13, weight="bold")
        for i, (tag, parts, label) in enumerate(bars):
            by = y + 4.6 - i * 4.2
            text(ax, x0 - 1.5, by + 1.5, tag, size=13, color=RED if tag == "A" else DP_EDGE[1], weight="bold")
            x = x0
            for value, fc, hatch in parts:
                w = width * value / unit
                rect(ax, x, by, w, 3.0, fc=fc, ec="#A6A6A6" if hatch or fc == GRAY else "white", lw=0.8, hatch=hatch)
                x += w
            text(ax, x + 1.2, by + 1.5, label, size=11.5, ha="left")

    row(33, "attention 之后每卡 token", [
        ("A", [(4096, GPU[1], None)], "4096"),
        ("B", [(32768, GPU[6], None)], "32768（两个 DP 组的全部 token）")], 32768 * 1.25)
    row(21.5, "GPU0 的 MoE 输入行数", [
        ("A", [(N, GPU[0], None), (M - N, GRAY, None)], "131072 行 buffer，只有 21667 行有效"),
        ("B", [(N, GPU[0], None), (32768 - N, "#F2F2F2", "///")], "32768 行全部有效，11101 行算出 0")], M * 1.25)
    row(10, "每卡每层传输", [
        ("A", [(371, GPU[1], None)], "≈ 371 MiB：只传有路由的 token，dispatch 用 FP8"),
        ("B", [(896, GPU[6], None)], "≈ 896 MiB：两次 8 卡 all-reduce [32768, 4096]")], 896 * 1.6)
    text(ax, 2, 4.2, "两种模式 GPU0 的 MoE 计算量相同：32644 条本地路由，moe_sorting 后 142 块 × 256 行。",
         size=12, color=SUB, ha="left")
    fig.savefig(os.path.join(OUT, "ep8_modes_compare.png"), dpi=110)


mode_a()
mode_b()
compare()
