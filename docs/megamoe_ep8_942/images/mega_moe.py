"""Figures for docs/megamoe_ep8_942/mega_moe_ep8_design.md.

    python docs/megamoe_ep8_942/images/mega_moe.py /tmp/fonts
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


def flow():
    """One layer call and what every workgroup of the mega kernel does (the implemented step 3)."""
    fig, ax = plt.subplots(figsize=(16, 12.6), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12.6)
    ax.axis("off")
    dark = "#333333"

    def box(x, y, w, h, color, text, fs=9.5, fc="white", ec="none"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=fc, linespacing=1.35)

    def arrow(x0, y0, x1, y1, color=dark, style="-", lw=1.5):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style,
                                                           shrinkA=0, shrinkB=0))

    def path(points, color=dark, style="-", lw=1.5):
        """Polyline whose last segment ends in an arrow head."""
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=style)
        arrow(*points[-2], *points[-1], color=color, style=style, lw=lw)

    # One call, in stream order.
    ax.text(0.2, 12.35, "一次MoE层调用（同一个stream上依次执行）", fontsize=12, fontweight="bold", color=dark)
    calls = [
        (0.2, 2.3, C["misc"], "quant\naiter per-1x128 FP8"),
        (2.9, 3.0, C["prep"], "ep_prep（1个WG）\n清arrived计数、路由计数\nP2P交换计数矩阵，算expert起点"),
        (6.3, 3.3, C["prep"], "ep_plan（每个producer wave一个WG）\n每条路由的目标行和发送key，排序\n每个block：expert、期望路由数、gu_ids"),
        (10.0, 2.6, "#555555", "mega kernel\npersistent，304 WG × 512线程\n每CU 1个WG（64 KB LDS）"),
        (13.0, 2.8, C["prep"], "ep_finish（1个WG）\n8个rank互发done flag\n→ out可用"),
    ]
    for i, (x, w, color, text) in enumerate(calls):
        box(x, 11.0, w, 1.05, color, text, fs=9)
        if i:
            prev_x, prev_w = calls[i - 1][:2]
            arrow(prev_x + prev_w, 11.52, x, 11.52)
    arrow(11.3, 11.0, 11.3, 10.62)

    # The mega kernel, per workgroup.
    ax.add_patch(FancyBboxPatch((0.15, 0.2), 15.7, 10.4, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#F7F7F7",
                                ec="#BBBBBB", lw=1.2))
    ax.text(0.35, 10.3, "mega kernel：每个WG循环", fontsize=11, fontweight="bold", color=dark)
    left = [
        (8.45, 1.2, C["misc"], "领票（wave 0）\n先用普通load看8个队列计数\n本XCD的队列优先，lane 0原子加\nLDS广播 (queue, ticket)", "white"),
        (7.35, 0.7, "white", "ticket ≥ 0 ?\n否：所有队列已空，退出", dark),
        (6.25, 0.7, "white", "重读kernel参数\n（经不透明的kernarg指针）", dark),
        (4.65, 1.2, "white", "按队列内的ticket分派\nticket < 2 → producer\n其余按“配对 + lag”顺序\n→ GateUp或Down任务", dark),
    ]
    for i, (y, h, color, text, fc) in enumerate(left):
        box(0.4, y, 3.4, h, color, text, fs=9, fc=fc, ec="#888888" if color == "white" else "none")
        if i:
            arrow(2.1, left[i - 1][0], 2.1, y + h)
    ax.plot([3.8, 4.1, 4.1, 13.75], [5.25, 5.25, 10.1, 10.1], color=dark, lw=1.5)
    columns = [
        (4.4, 3.2, C["comm"], [
            "Producer（每个队列的前2张票）\nwave j 发送ep_plan排好的slice j",
            "每次8条路由：\n读x的FP8行 + scale + 路由权重",
            "P2P store到目标rank\nrecv_a / recv_scale / down_ids / recv_w\n（sc0 nt sc1，写穿透）",
            "s_waitcnt vmcnt(0)\n目标rank的 arrived[block] += 1\n（系统范围原子加）",
        ]),
        (8.3, 3.2, C["gemm1"], [
            "GateUp(b, tile)\n等 arrived[b] ≥ 期望路由数\n（spin + s_sleep，只invalidate L1）",
            "GEMM 256×256×4096\nFP8 128×128 block scale，LDS流水",
            "epilogue开头：预领下一张票\n（原子加，结果到下个任务前才读）",
            "SiLU·up → FP8 + 每128列scale\n写a2（sc1写穿透）\nvmcnt(0) + barrier → done[b] += 1",
        ]),
        (12.2, 3.1, C["gemm2"], [
            "Down(b, tile)\n等 done[b] ≥ 16（GateUp的tile数）",
            "GEMM 256×256×2048\nFP8 128×128 block scale，LDS流水",
            "epilogue开头：预领下一张票",
            "× 路由权重 → BF16\nglobal_atomic_pk_add_bf16\n加到源rank的 out[token]（P2P）",
        ]),
    ]
    tops = [8.7, 7.35, 6.0, 4.65]
    for x, w, color, steps in columns:
        arrow(x + w / 2, 10.1, x + w / 2, 9.65)
        for i, (y, text) in enumerate(zip(tops, steps)):
            prefetch = text.startswith("epilogue")
            box(x, y, w, 0.95, C["misc"] if prefetch else color, text, fs=8.8)
            if i:
                arrow(x + w / 2, tops[i - 1], x + w / 2, y + 0.95)
        arrow(x + w / 2, 4.65, x + w / 2, 4.15)
    # Who waits for whom.
    path([(7.6, 5.12), (7.95, 5.12), (7.95, 9.17), (8.3, 9.17)], color=C["comm"], style="--")
    ax.text(8.12, 7.1, "数据 + arrived[b]（来自8个源rank）", rotation=90, ha="center", va="center", fontsize=8, color=C["comm"])
    path([(11.5, 5.12), (11.85, 5.12), (11.85, 9.17), (12.2, 9.17)], color=C["gemm1"], style="--")
    ax.text(12.02, 7.1, "done[b]（本rank）", rotation=90, ha="center", va="center", fontsize=8, color=C["gemm1"])
    path([(15.3, 5.12), (15.55, 5.12), (15.55, 11.0)], color=C["comm"], style="--")
    ax.text(15.72, 7.1, "out上的原子加全部完成后由ep_finish确认", rotation=90, ha="center", va="center", fontsize=8,
            color=C["comm"])
    # Back to the next ticket.
    box(4.4, 3.55, 10.9, 0.6, "#555555", "任务结束 → 回到领票：直接用epilogue里预领到的票；票无效（本XCD队列已空）才去别的队列找", fs=9)
    path([(4.4, 3.85), (0.25, 3.85), (0.25, 9.05), (0.4, 9.05)], color=dark)

    # Task order in one queue.
    ax.text(0.4, 2.85, "一个队列（每个XCD一个，放一段连续的block）里的任务顺序：", fontsize=10, fontweight="bold", color=dark)
    cells = [("P", C["comm"], 0.5), ("P", C["comm"], 0.5), ("G 组0", C["gemm1"], 1.45), ("G 组1", C["gemm1"], 1.45),
             ("G 组2", C["gemm1"], 1.45), ("D 组0", C["gemm2"], 1.45), ("G 组3", C["gemm1"], 1.45),
             ("D 组1", C["gemm2"], 1.45), ("…", "#BBBBBB", 0.8), ("D 倒数第2组", C["gemm2"], 1.45),
             ("D 最后一组", C["gemm2"], 1.45)]
    x = 0.6
    for text, color, w in cells:
        box(x, 1.75, w - 0.08, 0.8, color, text, fs=9)
        x += w
    ax.text(0.6, 1.25, "组 = 队列里相邻的2个block；GateUp的32个任务按tile交替两个block（同一个expert时两块同时读同一个B tile，"
            "在L2里共享），Down的32个任务同理。", fontsize=9, color=dark)
    ax.text(0.6, 0.85, "Down比GateUp晚 lag = 2 组，combine流量分散在整个kernel里；producer不等任何东西。", fontsize=9, color=dark)
    ax.text(0.6, 0.45, "不会死锁：任务只等同一队列里更早领走的票或producer，持票的WG要么正在跑它，要么已在epilogue、不再等待。",
            fontsize=9, color=dark)
    fig.savefig(HERE / "mega_moe_flow.png", bbox_inches="tight")


if __name__ == "__main__":
    timeline()
    roles()
    flow()
