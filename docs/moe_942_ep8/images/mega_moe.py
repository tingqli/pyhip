"""Figures for docs/moe_942_ep8/mega_moe_ep8_design.md.

    python docs/moe_942_ep8/images/mega_moe.py /tmp/fonts
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).parent
FONT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fonts")
for f in FONT_DIR.glob("NotoSansSC-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False

C = dict(comm="#4C9BE8", gemm1="#F5A623", gemm2="#E8743B", misc="#9B9B9B", wait="#E6E6E6", prep="#7ED321")


def bar(ax, y, x0, w, color, text, h=0.6, fc="white", fs=10):
    ax.barh(y, w, left=x0, height=h, color=color, edgecolor="white", linewidth=1.5)
    if text:
        ax.text(x0 + w / 2, y, text, ha="center", va="center", fontsize=fs, color=fc, fontweight="medium")


def timeline():
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=150)
    # Baseline: measured medians on 8x MI300X, 4096 tokens/rank (max over ranks).
    y = 2.2
    x = 0
    for w, c, t in [(18, C["misc"], ""), (330, C["comm"], "dispatch\n330"), (150, C["misc"], ""),
                    (1250, C["gemm1"], "GateUp ≈1250"), (800, C["gemm2"], "Down ≈800"), (100, C["wait"], ""),
                    (950, C["comm"], "combine ≈950（含等慢rank）")]:
        bar(ax, y, x, w, c, t, fc="black" if c == C["wait"] else "white")
        x += w
    ax.text(348 + 75, y + 0.42, "sort/转置/计数 ≈150", ha="center", fontsize=9)
    ax.text(x + 30, y, "≈3650 µs", va="center", fontsize=12, fontweight="bold")
    # Mega kernel: comm overlapped with GEMMs.
    y = 0.9
    bar(ax, y - 0.12, 0, 40, C["prep"], "", h=0.5)
    ax.text(20, y - 0.55, "prep", ha="center", fontsize=9)
    bar(ax, y + 0.36, 40, 400, C["comm"], "dispatch", h=0.3, fs=9)
    bar(ax, y - 0.12, 40, 1250, C["gemm1"], "GateUp（expert到齐即开始）", h=0.5)
    bar(ax, y - 0.12, 1290, 800, C["gemm2"], "Down", h=0.5)
    bar(ax, y + 0.36, 1290, 830, C["comm"], "combine（Down epilog写回源rank）", h=0.3, fs=9)
    ax.text(2160, y, "目标 ≈2100–2500 µs", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks([2.2, 0.9], ["当前\nMORI + FlyDSL MoE", "942 mega kernel"], fontsize=11)
    ax.set_xlim(0, 4100)
    ax.set_ylim(0.2, 2.8)
    ax.set_xlabel("时间 (µs)，一个MoE层，每卡4096 token")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(HERE / "mega_moe_timeline.png")


def roles():
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, color, text, fs=10, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=fc)

    def arrow(x0, y0, x1, y1, text="", color="#333333"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", fontsize=9, color=color)

    box(0.2, 3.9, 3.0, 1.0, C["prep"], "prep kernel（1个WG）\n按global expert计数排序路由\n交换计数矩阵 C[src][expert]", 9.5)
    box(0.2, 2.2, 3.0, 1.3, C["comm"], "producer WG（前P个）\n按 (expert, 目标rank) 顺序\n把FP8行+scale+元数据\nP2P写到目标rank的排序位置", 9.5)
    box(4.2, 2.2, 3.6, 1.3, C["gemm1"], "consumer WG：GateUp任务\n等 flag[src][e]==epoch（8个源）\n算完 → done[block]++", 9.5)
    box(8.7, 2.2, 4.1, 1.3, C["gemm2"], "consumer WG：Down任务\n等 done[block]==N tile数\nepilog：权重×y → 源rank的out\n(global_atomic_pk_add_bf16，P2P)", 9.5)
    arrow(3.2, 2.85, 4.2, 2.85, "flag")
    arrow(7.8, 2.85, 8.7, 2.85, "done计数")
    arrow(1.7, 3.9, 1.7, 3.5)
    box(4.2, 0.4, 8.6, 1.1, C["misc"], "任务队列（每rank一个原子计数）：GateUp(e0) … GateUp(e1) Down(e0) … 按expert交错，Down延后一个expert\n"
        "队列按顺序领取 → 等待的对象一定已被正在运行的WG持有 → 无死锁；网格=CU数，全部常驻", 9.5)
    box(9.6, 3.9, 3.2, 1.0, "#555555", "结束：每rank所有Down完成后\n给所有rank写done flag，\n等齐8个flag → out可用", 9.5)
    ax.text(0.2, 1.7, "一次调用的epoch号 +1：flag比较epoch，不需要清零", fontsize=9.5, color="#333333")
    fig.tight_layout()
    fig.savefig(HERE / "mega_moe_roles.png")


if __name__ == "__main__":
    timeline()
    roles()
