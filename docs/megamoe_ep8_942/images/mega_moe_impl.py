"""Figures for docs/megamoe_ep8_942/mega_moe_ep8_impl.md and mega_moe_ep8_att.md.

    python docs/megamoe_ep8_942/images/mega_moe_impl.py [FONT_DIR] [--from-trace TRACE.pt]

FONT_DIR holds NotoSansSC-*.otf (default /tmp/fonts). --from-trace refreshes data/trace_32k_balanced_bf16_rank0.json.gz
from `trace_ep8_mega.py --routing balanced --save TRACE.pt` (rank 0's tasks).
Example everywhere: MiMo-V2.6-Flash MoE layer, EP8, H 4096, I 2048, 256 experts (32 per rank), top-8, 4096 tokens per
rank (32k in total), balanced routing: every (source rank, expert) pair carries 128 routes, so every rank receives
32 x 1024 routes = 128 full blocks of 256 rows.
"""

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "trace_32k_balanced_bf16_rank0.json.gz"

NPES, E_LOCAL, NG, ND, QUEUES, QP, LAG, PAIR, CUS_PER_XCD = 8, 32, 16, 16, 8, 2, 2, 2, 38
BLOCKS = 128
MFMA_CYCLES = {"G": 32 * 4096, "D": 16 * 4096}  # per task: K blocks x 4096 SIMD cycles (2 waves x 128 MFMA x 16)
C = dict(P="#4C9BE8", G="#F5A623", D="#E8743B", wait="#D9D9D9", ticket="#6E6E6E", pro="#FBD38D", epi="#B9770E",
         dpro="#F4B183", depi="#9E3D16", idle="#FFFFFF", prep="#7ED321", dark="#333333")
GROUP_COLORS = ["#FFE0A3", "#FFD27A", "#C6E2FF", "#9CCBF7", "#D4EDC9", "#B3DEA0", "#F3D1E8", "#E7B4D6"]


def decode(queue, ticket, blocks=BLOCKS):
    """(kind, block, n tile) of a ticket of a queue: the mega kernel's run_task order (contiguous, paired, lag)."""
    if ticket < QP:
        return "P", None, None
    first = blocks * queue // QUEUES
    own = blocks * (queue + 1) // QUEUES - first
    groups = (own + PAIR - 1) // PAIR
    gate, down = PAIR * NG, PAIR * ND
    lead = min(groups, LAG)
    index = ticket - QP
    if index < gate * lead:
        kind, group, sub = "G", index // gate, index % gate
    else:
        mid = index - gate * lead
        unit, slot = divmod(mid, gate + down)
        if mid < (gate + down) * (groups - lead):
            kind, group, sub = ("G", lead + unit, slot) if slot < gate else ("D", unit, slot - gate)
        else:
            tail = mid - (gate + down) * (groups - lead)
            kind, group, sub = "D", groups - lead + tail // down, tail % down
    return kind, first + group * PAIR + sub % PAIR, sub // PAIR


def name(kind, block, tile):
    """e{expert}_blk{block within the expert}_N{tile} (balanced: 4 blocks per expert)."""
    return f"e{block // 4}_blk{block % 4}_N{tile}" if kind != "P" else "producer"


# ----------------------------------------------------------------------------------------------- trace data


def extract(trace_pt):
    import torch

    d = torch.load(trace_pt, weights_only=False)
    raw = d["traces"][0].view(-1, 8).tolist()
    qcap = d["traces"][0].numel() // 8 // 8
    experts = d["expert_ids"][0].tolist()
    assert len(experts) == BLOCKS and all(e == b // 4 for b, e in enumerate(experts)), "expects the balanced example"
    rows = [(i, r) for i, r in enumerate(raw) if r[3] & 0xFF in (1, 2, 3)]
    t0 = min(r[4] for _, r in rows)
    us = lambda t: round(((t - t0) & 0xFFFFFFFF) * 0.01, 2)
    tasks = []
    for i, r in rows:
        queue, ticket = divmod(i, qcap)
        kind, block, tile = decode(queue, ticket)
        assert {"P": 1, "G": 2, "D": 3}[kind] == r[3] & 0xFF, (queue, ticket)
        entry, start, waited, end = us(r[4]), us(r[0]), us(r[1]), us(r[2])
        if kind == "P":
            marks = [us(x) for x in r[5:8]]
        else:
            # Shader clocks wait end -> end (word 5), -> K-loop start (6), -> K-loop end (7).
            cyc, k0, k1 = (r[5] & 0xFFFFFFFF), (r[6] & 0xFFFFFFFF), (r[7] & 0xFFFFFFFF)
            marks = [round(waited + (end - waited) * k0 / cyc, 2), round(waited + (end - waited) * k1 / cyc, 2), cyc]
        tasks.append([kind, queue, ticket, block, tile, r[3] >> 24, (r[3] >> 8) & 0xFF, entry, start, waited, end, *marks])
    DATA.parent.mkdir(exist_ok=True)
    payload = dict(args=d["args"], stats=d["stats"], fields="kind queue ticket block tile xcc hw_id entry start waited end "
                   "[P: first three chunk ends | G/D: K-loop start, K-loop end, shader clocks wait->end]", tasks=tasks)
    with gzip.open(DATA, "wt") as f:
        json.dump(payload, f)
    print(f"wrote {DATA} ({len(tasks)} tasks)")


def load():
    with gzip.open(DATA, "rt") as f:
        payload = json.load(f)
    keys = "kind queue ticket block tile xcc hw entry start waited end".split()
    tasks = []
    for row in payload["tasks"]:
        t = dict(zip(keys, row[:11]))
        if t["kind"] == "P":
            t["chunks"] = row[11:14]
        else:
            t["k0"], t["k1"], t["cycles"] = row[11:14]
            t["mhz"] = t["cycles"] / max(t["end"] - t["waited"], 1e-3)
        t["cu"] = (t["xcc"], t["hw"])
        tasks.append(t)
    return payload, tasks


def budget(tasks):
    """Per-CU mean us of each use of the kernel span (rank 0, all 304 CUs)."""
    span = max(t["end"] for t in tasks)
    by_cu = defaultdict(list)
    for t in tasks:
        by_cu[t["cu"]].append(t)
    acc = defaultdict(float)
    for cu, ts in by_cu.items():
        ts.sort(key=lambda t: t["entry"])
        prev_end, prev_kind = 0.0, None
        for t in ts:
            gap = max(t["entry"] - prev_end, 0.0)
            ticket = t["start"] - t["entry"]
            # After a producer task the ticket barrier also waits for the producer's slower waves.
            acc["producer" if prev_kind == "P" else "ticket"] += gap + ticket
            if t["kind"] == "P":
                acc["producer"] += t["end"] - t["start"]
            else:
                acc["wait_" + t["kind"]] += t["waited"] - t["start"]
                mfma = MFMA_CYCLES[t["kind"]] / t["mhz"]
                acc["mfma"] += mfma
                acc["kloop_other"] += (t["k1"] - t["k0"]) - mfma
                acc["pro_epi"] += (t["k0"] - t["waited"]) + (t["end"] - t["k1"])
            prev_end, prev_kind = t["end"], t["kind"]
        acc["tail"] += span - prev_end
    n = len(by_cu)
    return span, n, {k: v / n for k, v in acc.items()}


# ----------------------------------------------------------------------------------------------- drawing helpers


def box(ax, x, y, w, h, color, text="", fs=9, fc="black", ec="none", lw=1.0, ha="center", round_=True, z=2, **kw):
    style = "round,pad=0.0,rounding_size=0.25" if round_ else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=color, ec=ec, lw=lw, zorder=z, **kw))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.4
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=fc, zorder=z + 1, linespacing=1.3)


def arrow(ax, p0, p1, color="#333333", lw=1.4, style="-|>", ls="-", rad=0.0, z=4, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def canvas(w, h, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, name_):
    fig.savefig(HERE / name_, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", HERE / name_)


# ----------------------------------------------------------------------------------------------- figures


def fig_partition():
    """Rank 0: blocks -> queues / XCDs -> CUs, the ticket order of a queue, and who waits for whom."""
    fig, ax = canvas(26, 19.5)
    dark = C["dark"]
    # (a) the 128 blocks of rank 0 by queue and position.
    ax.text(0, 99.2, "(a) rank 0 收到的 32768 条路由 = 32 个本地expert × 1024 行 = 128 个block（每个256行），按队列切成8段",
            fontsize=13, fontweight="bold", color=dark, va="top")
    x0, y0, cw, ch = 7.5, 59.0, 2.6, 4.1
    for p in range(16):
        ax.text(x0 + (p + 0.5) * cw, y0 + 8 * ch + 0.6, f"p{p}", ha="center", fontsize=8.5, color="#555555")
    for g in range(8):
        ax.text(x0 + (2 * g + 1) * cw, y0 + 8 * ch + 2.3, f"组 g{g}", ha="center", fontsize=9, color=dark)
        ax.plot([x0 + 2 * g * cw + 0.15, x0 + (2 * g + 2) * cw - 0.15], [y0 + 8 * ch + 1.6] * 2, color="#999999", lw=1)
    for q in range(8):
        y = y0 + (7 - q) * ch
        ax.text(x0 - 0.4, y + ch / 2, f"队列{q}\nXCD {q}", ha="right", va="center", fontsize=9, color=dark)
        for p in range(16):
            b = 16 * q + p
            box(ax, x0 + p * cw + 0.08, y + 0.1, cw - 0.16, ch - 0.2, GROUP_COLORS[(p // 2) % 8],
                f"e{b // 4}_blk{b % 4}\nb{b}", fs=7.2, round_=False)
    lines = [
        "• block b = 16·q + p（q = 队列/XCD，p = 队列内位置），占 recv_a 的行 [256b, 256b+256)；expert e 占 block 4e..4e+3，即行 [1024e, 1024e+1024)",
        "• 均匀路由：每个(源rank, expert)正好128条路由。源 s 的第 i 条路由放在 expert 内第 8i+s 行 → 每个block里8个源rank各32行",
        "• 组 g = 队列里相邻2个block（同一expert）：GateUp/Down 都按“blk_even N0, blk_odd N0, blk_even N1 …”交替，两个block同时读同一个权重tile",
        "• GateUp 任务 tile：256行 × N tile（128个gate列 + 对应128个up列，共256个W1行），K = H = 4096；N tile 0..15（I/128）",
        "• Down 任务 tile：256行 × 256个hidden列，K = I = 2048；N tile 0..15（H/256）。任务名 e?_blk?_N?，GateUp 记 G，Down 记 D",
    ]
    for i, line in enumerate(lines):
        ax.text(0.5, 57.4 - i * 1.55, line, fontsize=9.3, color=dark, va="top")
    # (b) GPU0's CUs.
    ax.text(51.5, 99.2, "(b) GPU0：304 个CU = 8 个XCD × 38 CU，每CU一个常驻WG（512线程，64 KB LDS）", fontsize=13,
            fontweight="bold", color=dark, va="top")
    bw, bh = 11.6, 14.0
    for x in range(8):
        bx, by = 52.0 + (x % 4) * (bw + 0.5), 80.5 - (x // 4) * (bh + 1.2)
        box(ax, bx, by, bw, bh, "#F7F7F7", ec="#BBBBBB")
        ax.text(bx + bw / 2, by + bh - 1.2, f"XCD {x} → 队列{x}", ha="center", fontsize=10, fontweight="bold", color=dark)
        ax.text(bx + bw / 2, by + bh - 2.9, f"e{4 * x}–e{4 * x + 3}，block {16 * x}–{16 * x + 15}", ha="center", fontsize=8.5,
                color="#555555")
        for c in range(CUS_PER_XCD):
            cx, cy = bx + 0.55 + (c % 10) * 1.06, by + bh - 5.4 - (c // 10) * 1.25
            producer = c < 2
            box(ax, cx, cy, 0.9, 1.0, C["P"] if producer else "#F5C27A", "P" if producer else "", fs=6.5, fc="white",
                round_=False)
        ax.text(bx + 0.55, by + 2.2, "2个P：先领到票0、1的WG做producer\n36个C：领GateUp/Down票", fontsize=8, color=dark,
                va="center")
    notes = [
        "• grid = 304 个WG（persistent）；硬件把WG轮流派到8个XCD；WG读 HW_REG_XCC_ID 得到自己的队列（home队列）",
        "• 每个队列 514 张票：2 张 producer + 256 个GateUp + 256 个Down；全卡 16 个producer WG（每个8 wave）",
        "• producer发完数据后也去领GateUp/Down票；home队列空了才去别的队列领（先普通load看8个计数，再原子加）",
        "• 一个WG同一时间只跑一个任务：8个wave一起算同一个tile，任务之间用 s_barrier + LDS 广播票号",
    ]
    for i, line in enumerate(notes):
        ax.text(52.0, 63.6 - i * 1.55, line, fontsize=9.3, color=dark, va="top")
    # (c) the ticket order of queue 0.
    ax.text(0, 42.0, "(c) 一个队列里的票顺序（队列0；lag = 2 组，pair = 2 个block；GateUp 先领2组，之后每组GateUp后面跟着晚2组的Down）",
            fontsize=13, fontweight="bold", color=dark, va="top")
    segs = [("P", None, 2)] + [("G", 0, 32), ("G", 1, 32)]
    for u in range(6):
        segs += [("G", u + 2, 32), ("D", u, 32)]
    segs += [("D", 6, 32), ("D", 7, 32)]
    x, y, scale = 0.5, 29.8, 98.6 / (2 * 4.0 + 16 * 32)
    ticket = 0
    seg_x = {}
    for kind, g, n in segs:
        w = (4.0 if kind == "P" else n) * scale * (1 if kind != "P" else 2)
        label = "P\nT0–1" if kind == "P" else f"{kind} g{g}\ne{g // 2}_blk{2 * (g % 2)}/{2 * (g % 2) + 1}\nT{ticket}–{ticket + n - 1}"
        box(ax, x, y, w - 0.12, 4.6, C[kind], label, fs=7.6, fc="white", round_=False)
        seg_x[(kind, g)] = (x, w)
        x += w
        ticket += n
    gx, gw = seg_x[("G", 0)]
    dx, dw = seg_x[("D", 0)]
    arrow(ax, (gx + gw / 2, y + 4.7), (dx + dw / 2, y + 4.7), color="#B03A2E", rad=-0.25, lw=1.6)
    ax.text(dx + dw + 1.0, y + 6.6, "Down(g0) 排在 GateUp(g2) 之后：比GateUp晚 lag = 2 组，combine流量分散在整个kernel里",
            ha="left", fontsize=9.3, color="#B03A2E")
    # Zoom into G g0 and D g0.
    zoom = [("G", t) for t in range(2, 8)] + [("…", None)] + [("G", t) for t in (32, 33)]
    zx, zy = 0.5, 21.0
    ax.text(zx, zy + 6.4, "放大：G g0 的32张票（同一组2个block按N tile交替）", fontsize=9.5, color=dark)
    for i, (kind, t) in enumerate(zoom):
        if kind == "…":
            ax.text(zx + i * 5.0 + 2.2, zy + 2.3, "…", fontsize=14, ha="center", va="center")
            continue
        k, b, n = decode(0, t)
        box(ax, zx + i * 5.0, zy, 4.8, 4.6, C["G"], f"T{t}\nG {name(k, b, n)}", fs=7.6, fc="white", round_=False)
    zoom = [("D", t) for t in range(98, 102)] + [("…", None)] + [("D", t) for t in (128, 129)]
    zx = 50.5
    ax.text(zx, zy + 6.4, "放大：D g0 的32张票（Down 的 N tile = 256个hidden列）", fontsize=9.5, color=dark)
    for i, (kind, t) in enumerate(zoom):
        if kind == "…":
            ax.text(zx + i * 6.6 + 3.0, zy + 2.3, "…", fontsize=14, ha="center", va="center")
            continue
        k, b, n = decode(0, t)
        box(ax, zx + i * 6.6, zy, 6.4, 4.6, C["D"], f"T{t}\nD {name(k, b, n)}", fs=7.6, fc="white", round_=False)
    # (d) who writes and who waits.
    ax.text(0, 17.6, "(d) 协同：谁写、谁等（rank 0 视角；所有等待都是 wave 0 自旋 + s_sleep，等到后整个WG过 barrier）", fontsize=13,
            fontweight="bold", color=dark, va="top")
    y = 2.6
    flow = [
        (0.5, 17.5, C["P"], "8个源rank的producer\n（每rank 16 WG × 8 wave）\n按 ep_plan 排好的顺序，每次8条路由：\n读x的FP8行+scale+权重 → P2P store"),
        (23.5, 17.0, C["G"], "GateUp 任务 G(b, n)\nN0 等 arrived[b] ≥ expected[b]，转置A scale\n其余tile等 scaled[b] ≥ 1\n256行 × 256个W1行，K=4096 → FP8 a2"),
        (46.0, 17.0, C["D"], "Down 任务 D(b, n)\n等 done[b] ≥ 16（16个GateUp tile）\n256行 × 256个hidden列，K=2048\n× 路由权重 → BF16"),
        (68.5, 13.5, "#555555", "源rank的 out[token]\n（BF16，对称堆，ep_prep前清零）"),
        (85.5, 14.0, C["prep"], "ep_finish（1 WG）\nFFLAG[s] = epoch 互发\n等齐8个 → out可用"),
    ]
    for x, w, color, text in flow:
        box(ax, x, y, w, 8.6, color, text, fs=9, fc="white")
    labels = [
        ((18.0, 6.9), (23.5, 6.9), "recv_a / recv_scale /\ndown_ids / recv_w 行\n+ arrived[b] += 1（每条路由）"),
        ((40.5, 6.9), (46.0, 6.9), "本地 a2 行 + a2_scale_t\n+ done[b] += 1（每个tile）"),
        ((63.0, 6.9), (68.5, 6.9), "global_atomic_pk_add_bf16\n（P2P，每行加一次）"),
        ((82.0, 6.9), (85.5, 6.9), ""),
    ]
    for p0, p1, text in labels:
        arrow(ax, p0, p1, lw=1.8)
        if text:
            ax.text((p0[0] + p1[0]) / 2, 11.6, text, ha="center", va="bottom", fontsize=8.3, color=dark)
    ax.text(0.5, 1.0, "领票：sched.ticket[q]（每个XCD队列一个计数），wave 0 的 lane 0 原子加；任务在 epilogue 开头就预领下一张票（结果到下个任务前才读）。"
            "任务只等更早的票或producer，所以不会死锁。", fontsize=9.3, color=dark)
    save(fig, "impl_partition.png")


SRC_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7"]


def ink(color):
    """Black or white text, whichever reads better on `color`."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "white"


def mega_row(counts, s, i):
    """Row of source s's i-th route within its expert, as ep_plan's mega_row computes it."""
    if i < min(counts):
        return i * NPES + s
    return sum(min(c, i + (t < s)) for t, c in enumerate(counts))


def expert_rows(counts):
    """(source, i) of an expert's rows in row order: round i takes route i of every source that still has one."""
    rows = [(s, i) for i in range(max(counts)) for s in range(NPES) if i < counts[s]]
    assert all(mega_row(counts, s, i) == r for r, (s, i) in enumerate(rows))
    return rows


def fig_expert_rows():
    """Rank 0's rows of one local expert: the 8 sources' routes interleaved round by round."""
    fig, ax = canvas(22, 15.5, ylim=(3, 100))
    dark, red = C["dark"], "#B03A2E"
    ax.text(0, 99.6, "接收端一个expert的行布局（rank 0 的本地expert e；源 s = 发出这条路由的rank，s0 是本卡自己）",
            fontsize=14, fontweight="bold", color=dark, va="top")
    for s in range(NPES):
        box(ax, 72.0 + s * 3.4, 97.3, 3.1, 2.0, SRC_COLORS[s], f"s{s}", fs=9, fc=ink(SRC_COLORS[s]), round_=False)
    # (a) The whole expert, one strip of 256 rows per block.
    ax.text(0, 95.4, "(a) 均匀路由：8 个源各 128 条 = 1024 行 = 4 个block。每个block画成一条：256 个格子 = 256 行，颜色 = 源"
            "（行号相对 expert 的起点 base_e）", fontsize=12, fontweight="bold", color=dark, va="top")
    x0, width, height = 13.0, 84.0, 1.7
    layouts = [
        (92.4, "mega kernel（ep_plan 的 mega_row）：源 s 的第 i 条放在第 8i+s 行 → 每个block里8个源各32条",
         [s for s, _ in expert_rows([128] * NPES)]),
        (81.8, "对照 step1（ep_dispatch 的 send_base）：每个源的128条连续放 → block 0 只有 s0、s1，要等这两个源各发完128条",
         [r // 128 for r in range(1024)]),
    ]
    cell = width / 256
    for top, title, sources in layouts:
        ax.text(x0, top, title, fontsize=10, color=dark, va="bottom")
        for blk in range(4):
            y = top - 0.3 - height - blk * 2.2
            ax.text(x0 - 0.5, y + height / 2, f"blk{blk}（b = 4e{f'+{blk}' if blk else ''}）行 {256 * blk}–{256 * blk + 255}",
                    ha="right", va="center", fontsize=8.5, color=dark)
            for k in range(256):
                ax.add_patch(Rectangle((x0 + k * cell, y), cell, height, fc=SRC_COLORS[sources[256 * blk + k]], ec="none",
                                       antialiased=False, zorder=2))
            ax.add_patch(Rectangle((x0, y), width, height, fc="none", ec="#555555", lw=0.6, zorder=3))
    # (b) Block 0 zoomed in: one line per round, one column per source.
    ax.text(0, 71.6, "(b) 放大 block 0：32 轮 × 8 个源。一行一行读（先 s0→s7，再下一轮）就是 recv_a 的行顺序", fontsize=12,
            fontweight="bold", color=dark, va="top")
    gx, cw = 9.5, 4.9
    for s in range(NPES):
        box(ax, gx + s * cw + 0.1, 66.4, cw - 0.2, 2.0, SRC_COLORS[s], f"rank {s} 写", fs=8.5, fc=ink(SRC_COLORS[s]), round_=False)
    for k, i in enumerate([0, 1, 2, 3, None, 30, 31]):
        y = 62.6 - k * 3.75
        if i is None:
            ax.text(gx + 4 * cw, y + 1.75, "…（第 4–29 轮同样，每轮 8 行）", ha="center", va="center", fontsize=10, color=dark)
            continue
        ax.text(gx - 0.5, y + 1.75, f"第{i}轮", ha="right", va="center", fontsize=9, color=dark)
        for s in range(NPES):
            box(ax, gx + s * cw + 0.1, y, cw - 0.2, 3.5, SRC_COLORS[s], f"行 {8 * i + s}\ns{s} 第{i}条", fs=7.8,
                fc=ink(SRC_COLORS[s]), round_=False)
    example = [
        "例：expert 内第 19 行 = 第 2 轮的 s3 = rank 3 发给 expert e 的第 2 条路由（rank 3 上的某个 token t，slot k）",
        "• recv_a[base_e+19] = rank 3 的 x_fp8[t]（4096 B）；recv_scale 同一行 = 它的 x_scale[t]（32 个 FP32）",
        "• down_ids[base_e+19] = (3 << 12 | t) | k << 24；recv_w[base_e+19] = topk_w[t][k]（rank 3 上的值）",
        "• GateUp 的结果写 a2 第 base_e+19 行；Down 乘上 recv_w 后加回 rank 3 的 out[t]",
        "• block 0 要 8 个源各写 32 行：arrived[4e] 加到 256 = expected[4e]，GateUp(4e, *) 才开始",
    ]
    box(ax, 0.5, 23.6, 48.5, 14.4, "#F4F4F4", ec="#CCCCCC")
    for j, line in enumerate(example):
        ax.text(1.3, 36.6 - j * 2.75, line, fontsize=9.3, color=dark, va="center", fontweight="bold" if j == 0 else "normal")
    # (c) Unequal counts: a source that ran out drops out of the later rounds.
    counts = [130, 124, 135, 128, 121, 140, 118, 126]
    n = sum(counts)
    ax.text(52.0, 71.6, "(c) 各源条数不同（随机路由）：用完的源在后面的轮次里跳过，往前压紧", fontsize=12, fontweight="bold",
            color=dark, va="top")
    ax.text(52.0, 68.6, f"例：s0..s7 发给这个expert的条数 c = {counts}，共 {n} 行 → 4 个block，最后 {1024 - n} 行是 padding",
            fontsize=9.3, color=dark, va="center")
    ax.text(52.0, 66.9, f"第 0–{min(counts) - 1} 轮 8 个源都有（fewest = {min(counts)}），和 (b) 一样是第 8i+s 行。"
            "下面是 block 3（行 768–1023）的末尾：", fontsize=9.3, color=dark, va="center")
    rounds = defaultdict(list)
    for r, (s, i) in enumerate(expert_rows(counts)):
        rounds[i].append((r, s))
    cx, cw2, step = 64.0, 3.4, 1.45
    for k, i in enumerate(list(range(116, max(counts))) + [None]):
        y = 64.6 - (k + 1) * step
        if i is None:
            ax.text(cx - 0.4, y + 0.67, f"padding  行{n}–1023", ha="right", va="center", fontsize=7.8, color=dark)
            for j in range(1024 - n):
                box(ax, cx + j * cw2 + 0.06, y, cw2 - 0.12, 1.33, "#BDBDBD", "pad", fs=7.3, round_=False)
            ax.text(cx + (1024 - n) * cw2 + 0.6, y + 0.67, f"← gu_ids = down_ids = −1；block 3 的 expected = {n - 768}",
                    va="center", fontsize=8.3, color=red)
            continue
        cells = rounds[i]
        span = f"{cells[0][0]}" if len(cells) == 1 else f"{cells[0][0]}–{cells[-1][0]}"
        ax.text(cx - 0.4, y + 0.67, f"第{i}轮  行{span}", ha="right", va="center", fontsize=7.8, color=dark)
        for j, (_, s) in enumerate(cells):
            box(ax, cx + j * cw2 + 0.06, y, cw2 - 0.12, 1.33, SRC_COLORS[s], f"s{s}", fs=7.3, fc=ink(SRC_COLORS[s]), round_=False)
        notes = [f"s{s} 用完（{counts[s]} 条）" for s in range(NPES) if counts[s] == i]
        notes += [f"s{s} 的最后一条（共 {counts[s]} 条）" for s in range(NPES) if counts[s] == i + 1 == max(counts)]
        if notes:
            ax.text(cx + len(cells) * cw2 + 0.6, y + 0.67, "← " + "，".join(notes), va="center", fontsize=8.3, color=red)
    for j, line in enumerate([
        "• 第 i 轮按 s0→s7 的顺序，只排还有第 i 条的源，行与行之间没有空洞：i ≥ fewest 时行号 = base_e + Σ_s' min(c_s', i + [s' < s])",
        "• 压紧只发生在最少的源用完以后（前 8·fewest 行仍是每个block各源32条）。block 数 = ceil(n/256)，n > 1024 时是 5 个block",
    ]):
        ax.text(52.0, 26.6 - j * 2.0, line, fontsize=8.8, color=dark, va="center")
    for j, line in enumerate([
        "• 行号由源 rank 在 ep_plan 里算（mega_row），要用 ep_prep 交换来的计数矩阵 C[s][g]（c_s = C[s][g]，fewest = min_s c_s）。"
        "接收端不再排序，recv_a 第 r 行就是 GateUp 的第 r 个 sorted 行。",
        "• “第 i 条” = route_idx：ep_prep 给每个 (源, expert) 的路由用原子计数编号，大致是 token 顺序但不保证；行里实际是哪个 token 记在 down_ids 里。",
        "   均匀路由的例子里，源 s 上选中同一个 expert 的 128 个 token 是 t0, t0+32, t0+64, …（slot 都相同，t0 和 slot 由源 s 的随机排列决定）。",
        "• 为什么交错：每个 block 都要 8 个源各 32 条才能开始 GateUp（arrived[b] = expected[b]）。ep_plan 按 block 在目标队列里的位置排发送顺序，",
        "   所以队列前面的 block 由 8 个源同时填；连续排布（step1）下 block 0 只靠 s0、s1，要等它们各发完 128 条，别的源先到的数据也用不上。",
        "• expert 的起点按 256 行对齐（base_e = Σ_{e'<e} ceil(n_e'/256)·256，均匀路由下 = 1024e），block 不跨 expert；padding 行 GateUp 读 0，"
        "Down 跳过。a2、a2_scale_t、down_ids、recv_w 都用同一个行号。",
    ]):
        ax.text(0.5, 19.6 - j * 2.6, line, fontsize=10, color=dark, va="center")
    save(fig, "impl_expert_rows.png")


# Model of one queue (XCD 0, 38 CUs) following the kernel's rules, with measured mean phase lengths (us, rank 0 trace)
# and data-arrival times per queue position (tile-0 waits of positions 0-3 and 5 in the trace, the rest interpolated
# up to the producers' end; later positions arrive before they are needed).
MODEL = dict(ticket=3.4, G=(5.3, 159.1, 15.6), D=(3.7, 74.3, 14.5), P=705.0,
             arrive=[89, 121, 164, 210, 280, 356, 400, 445, 490, 535, 580, 625, 665, 705, 705, 705])


def simulate_queue(queue=0):
    """Tasks of one queue on its XCD's 38 CUs: tickets in request order (first at start, then at each task's
    epilogue start, a producer's at its end); GateUp waits for its block's data, Down for its block's 16 GateUp tiles."""
    import heapq

    first = BLOCKS * queue // QUEUES
    total = QP + (NG + ND) * (BLOCKS // QUEUES)
    heap = [(0.5 + 0.03 * c, c, 0.5 + 0.03 * c) for c in range(CUS_PER_XCD)]
    heapq.heapify(heap)
    tasks, g_end = [], defaultdict(list)
    for ticket in range(total):
        request, cu, free = heapq.heappop(heap)
        kind, block, tile = decode(queue, ticket)
        start = free + MODEL["ticket"]
        t = dict(kind=kind, block=block, tile=tile, cu=cu, ticket=ticket, entry=free, start=start)
        if kind == "P":
            t.update(waited=start, k0=start, k1=start, end=start + MODEL["P"])
            heapq.heappush(heap, (t["end"], cu, t["end"]))
        else:
            pro, loop, epi = MODEL[kind]
            ready = MODEL["arrive"][block - first] if kind == "G" else max(g_end[block])
            t["waited"] = max(start, ready)
            t["k0"] = t["waited"] + pro
            t["k1"] = t["k0"] + loop
            t["end"] = t["k1"] + epi
            if kind == "G":
                g_end[block].append(t["end"])
            heapq.heappush(heap, (t["k1"], cu, t["end"]))
        tasks.append(t)
    return tasks


def draw_lanes(ax, lanes, x_max, label_min=60.0, fs=6.0):
    """lanes: [(label, [task dict with kind/block/tile/entry/start/waited/k0/k1/end])]; one row per CU."""
    for row, (label, ts) in enumerate(lanes):
        y = len(lanes) - 1 - row
        ax.text(-12, y, label, ha="right", va="center", fontsize=7.2)
        prev = None
        for t in ts:
            h = 0.78
            if t["start"] > t["entry"]:
                # After a producer task, the take-ticket barrier waits for the producer's slower waves.
                tail = prev is not None and prev["kind"] == "P"
                ax.barh(y, t["start"] - t["entry"], left=t["entry"], height=h, color="#A9CDF2" if tail else C["ticket"], lw=0,
                        hatch="\\\\" if tail else None, edgecolor=C["P"] if tail else None)
                if tail and t["start"] - t["entry"] > 80:
                    ax.text((t["entry"] + t["start"]) / 2, y, "producer其余wave", ha="center", va="center", fontsize=fs, color="#1F4E79")
            prev = t
            if t["kind"] == "P":
                ax.barh(y, t["end"] - t["start"], left=t["start"], height=h, color=C["P"], lw=0)
                ax.text((t["start"] + t["end"]) / 2, y, "producer（wave 0 发dispatch数据）", ha="center", va="center", fontsize=fs + 0.5,
                        color="white")
                continue
            if t["waited"] > t["start"]:
                ax.barh(y, t["waited"] - t["start"], left=t["start"], height=h, color=C["wait"], lw=0, hatch="////",
                        edgecolor="#9A9A9A")
            pro, loop, epi = (C["pro"], C["G"], C["epi"]) if t["kind"] == "G" else (C["dpro"], C["D"], C["depi"])
            ax.barh(y, t["k0"] - t["waited"], left=t["waited"], height=h, color=pro, lw=0)
            ax.barh(y, t["k1"] - t["k0"], left=t["k0"], height=h, color=loop, lw=0)
            ax.barh(y, t["end"] - t["k1"], left=t["k1"], height=h, color=epi, lw=0)
            ax.plot([t["end"]] * 2, [y - h / 2, y + h / 2], color="white", lw=0.6)
            if t["end"] - t["waited"] >= label_min:
                ax.text((t["k0"] + t["k1"]) / 2, y, f"{t['kind']} {name(t['kind'], t['block'], t['tile'])}", ha="center",
                        va="center", fontsize=fs, color="black" if t["kind"] == "G" else "white")
    ax.set_ylim(-0.8, len(lanes) - 0.2)
    ax.set_xlim(-2, x_max)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("kernel 内时间 (µs)", fontsize=10)


def lane_legend(ax, y=1.01):
    items = [(C["ticket"], "领票/分派"), (C["wait"], "等待（G等数据 / D等GateUp）"), (C["pro"], "GateUp prologue"), (C["G"], "GateUp K循环"),
             (C["epi"], "GateUp epilogue"), (C["dpro"], "Down prologue"), (C["D"], "Down K循环"), (C["depi"], "Down epilogue（含combine原子加）"),
             (C["P"], "producer"), ("#A9CDF2", "等producer其余wave")]
    handles = [Rectangle((0, 0), 1, 1, fc=c, hatch="////" if c == C["wait"] else None, ec="#9A9A9A" if c == C["wait"] else "none")
               for c, _ in items]
    ax.legend(handles, [t for _, t in items], loc="lower left", bbox_to_anchor=(0, y), ncol=10, fontsize=8.5, frameon=False,
              handlelength=1.6, columnspacing=1.2)


def fig_swimlane_model():
    tasks = simulate_queue(0)
    by_cu = defaultdict(list)
    for t in tasks:
        by_cu[t["cu"]].append(t)
    order = sorted(by_cu, key=lambda c: by_cu[c][0]["ticket"])
    lanes = [(f"CU{c:02d}  T{by_cu[c][0]['ticket']}", by_cu[c]) for c in order]
    span = max(t["end"] for t in tasks)
    fig, ax = plt.subplots(figsize=(28, 15), dpi=120)
    draw_lanes(ax, lanes, span + 20)
    lane_legend(ax)
    # Dependencies of the first pair group (e0_blk0 / e0_blk1): data arrival -> 16 GateUp tiles -> done -> Down.
    row_of = {c: len(order) - 1 - i for i, c in enumerate(order)}
    for block, color in ((0, "#1F5FBF"),):
        gs = [t for t in tasks if t["kind"] == "G" and t["block"] == block]
        ds = [t for t in tasks if t["kind"] == "D" and t["block"] == block]
        done = max(t["end"] for t in gs)
        ax.axvline(done, color=color, lw=1.0, ls="--", alpha=0.8)
        ax.text(done + 3, len(order) - 0.45, "done[e0_blk0] = 16", color=color, fontsize=8.5, va="bottom")
        last = max(gs, key=lambda t: t["end"])
        for d in ds[:2]:
            arrow(ax, (last["end"], row_of[last["cu"]]), (d["waited"], row_of[d["cu"]]), color=color, lw=1.2, rad=0.08, ms=9)
    ax.axvline(MODEL["arrive"][0], color="#2E7D32", lw=1.2, ls="--")
    ax.text(MODEL["arrive"][0] + 3, -0.75, "队列位置0的数据到齐（arrived[b] = 256）", color="#2E7D32", fontsize=9, va="bottom")
    ax.axvline(MODEL["arrive"][3], color="#2E7D32", lw=1.0, ls=":")
    ax.text(MODEL["arrive"][3] + 3, -0.75, "位置3到齐", color="#2E7D32", fontsize=9, va="bottom")
    ax.set_title("按代码逻辑模拟的泳道图：rank 0 的 XCD 0（队列0，38个CU，514张票）。每条泳道一个CU；输入用实测的平均阶段时长和数据到达时间；"
                 "箭头：e0_blk0 的16个GateUp tile都完成（done = 16）后，它的Down才能开始（实际要等队列把D g0的票发出来）", fontsize=12, loc="left", pad=28)
    fig.tight_layout()
    save(fig, "impl_swimlane_model.png")


def fig_swimlane_trace(tasks):
    xcd = [t for t in tasks if t["xcc"] == 0]
    by_cu = defaultdict(list)
    for t in xcd:
        by_cu[t["hw"]].append(t)
    for ts in by_cu.values():
        ts.sort(key=lambda t: t["entry"])
    order = sorted(by_cu, key=lambda hw: (hw >> 5, (hw >> 4) & 1, hw & 15))
    lanes = []
    for hw in order:
        lanes.append((f"SE{hw >> 5} CU{hw & 15:2d}", by_cu[hw]))
    span = max(t["end"] for t in tasks)
    fig, ax = plt.subplots(figsize=(28, 15), dpi=120)
    draw_lanes(ax, lanes, span + 20)
    lane_legend(ax)
    stolen = sum(t["queue"] != 0 for t in xcd)
    row_of = {}
    for i, hw in enumerate(order):
        for t in by_cu[hw]:
            row_of[id(t)] = len(order) - 1 - i
    for block, color in ((0, "#1F5FBF"),):
        gs = [t for t in tasks if t["kind"] == "G" and t["block"] == block]
        ds = [t for t in tasks if t["kind"] == "D" and t["block"] == block and t["xcc"] == 0]
        last = max(gs, key=lambda t: t["end"])
        ax.axvline(last["end"], color=color, lw=1.0, ls="--", alpha=0.8)
        ax.text(last["end"] + 3, len(order) - 0.45, "done[e0_blk0] = 16", color=color, fontsize=8.5, va="bottom")
        if last["xcc"] == 0:
            for d in ds[:2]:
                arrow(ax, (last["end"], row_of[id(last)]), (d["waited"], row_of[id(d)]), color=color, lw=1.2, rad=0.08, ms=9)
    first_ready = min(t["waited"] for t in xcd if t["kind"] == "G")
    ax.axvline(first_ready, color="#2E7D32", lw=1.2, ls="--")
    ax.text(first_ready + 3, -0.75, f"第一个GateUp的数据到齐（{first_ready:.0f} µs）", color="#2E7D32", fontsize=9, va="bottom")
    ax.set_title(f"实测trace：rank 0 的 XCD 0（38个CU，{len(xcd)}个任务，其中{stolen}个来自别的队列）。8卡、每卡4096 token、均匀路由、BF16 combine；"
                 "每条泳道一个CU，时间从rank 0第一个WG领票算起", fontsize=12, loc="left", pad=28)
    fig.tight_layout()
    save(fig, "impl_swimlane_trace.png")


BUDGET_LABELS = [("mfma", "MFMA（K循环里MFMA本身，按实测时钟）", "#2E7D32"), ("kloop_other", "K循环里的非MFMA时间", "#9CCC65"),
                 ("pro_epi", "任务 prologue + epilogue", "#F5A623"), ("wait_G", "GateUp等数据（dispatch）", "#BDBDBD"),
                 ("wait_D", "Down等GateUp", "#8D8D8D"), ("ticket", "领票/分派", "#5D5D5D"), ("producer", "producer", C["P"]),
                 ("tail", "收尾空闲", "#EEEEEE")]


def fig_budget(tasks):
    span, cus, b = budget(tasks)
    fig, ax = plt.subplots(figsize=(15, 4.8), dpi=150)
    y, x = 3.0, 0.0
    for key, label, color in BUDGET_LABELS:
        w = b.get(key, 0.0)
        ax.barh(y, w, left=x, height=0.62, color=color, edgecolor="white", lw=1)
        if w > 45:
            ax.text(x + w / 2, y, f"{w:.0f}\n{w / span * 100:.0f}%", ha="center", va="center", fontsize=8.6,
                    color="white" if color in ("#2E7D32", "#5D5D5D", C["P"], "#8D8D8D") else "black")
        x += w
    ax.text(x + 20, y, f"{span:.0f} µs（实测 span）", va="center", fontsize=10.5, fontweight="bold")
    small = "、".join(f"{label} {b.get(key, 0.0):.0f} µs" for key, label, _ in BUDGET_LABELS if b.get(key, 0.0) <= 45)
    ax.text(0, y - 0.48, "窄条：" + small, fontsize=9, va="center", color="#444444")
    mfma_ideal_cycles = (2048 * MFMA_CYCLES["G"] + 2048 * MFMA_CYCLES["D"]) / 304
    ax.barh(2.0, b["mfma"], left=0, height=0.5, color="#2E7D32")
    ax.text(b["mfma"] + 20, 2.0, f"{b['mfma']:.0f} µs：全部只做MFMA（功耗墙下实测时钟约1.0 GHz）", va="center", fontsize=10)
    ax.barh(1.2, mfma_ideal_cycles / 2100, left=0, height=0.5, color="#81C784")
    ax.text(mfma_ideal_cycles / 2100 + 20, 1.2, f"{mfma_ideal_cycles / 2100:.0f} µs：全部只做MFMA（2.1 GHz，不受功耗墙限制）", va="center",
            fontsize=10)
    ax.barh(0.4, 430, left=0, height=0.5, color=C["P"])
    ax.barh(0.4, 790, left=430, height=0.5, color="#1565C0")
    ax.text(215, 0.4, "dispatch ≈430", ha="center", va="center", fontsize=8.6, color="white")
    ax.text(825, 0.4, "combine（BF16远端原子加）≈790", ha="center", va="center", fontsize=8.6, color="white")
    ax.text(1240, 0.4, "通信单独跑约1220 µs（P2P实测带宽）；mega里露出来的约145 µs（profile：p2p − local）", va="center", fontsize=10)
    ax.set_yticks([3.0, 2.0, 1.2, 0.4], ["实测：rank 0 每个CU的平均时间", "MFMA下限（实测时钟）", "MFMA下限（2.1 GHz）", "通信量折算的时间"],
                  fontsize=10)
    ax.set_xlim(0, 2900)
    ax.set_ylim(-0.1, 3.75)
    ax.set_xlabel("µs（8卡共32k token，均匀路由，BF16 combine；rank 0 一次调用的trace）")
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="#BBBBBB") for _, _, c in BUDGET_LABELS]
    ax.legend(handles, [l for _, l, _ in BUDGET_LABELS], loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=9,
              frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "impl_budget.png")
    return span, cus, b


# ATT of the mega kernel (rank 0, 8 CUs with all 8 waves traced, two dispatches) and of the standalone GateUp (one
# GPU, same 128 blocks, A scales token-major as in the mega kernel vs transposed). Thousands of shader cycles per task.
ATT = dict(
    G=dict(ticket=(2.4, 8.2), dispatch=(3.4, 2.8), prologue=(4.8, 5.5), kloop=(165.6, 176.9), epilogue=(13.9, 13.6)),
    D=dict(ticket=(5.5, 6.4), dispatch=(2.8, 2.4), prologue=(5.0, 5.6), kloop=(74.7, 77.7), epilogue=(21.3, 18.4)),
    standalone=dict(transposed=(7.8, 139.8, 12.1), token_major=(8.1, 163.6, 12.0)),
    idle_G={"VMEM：等vmcnt / buffer_load发不出去": (14.0, 21.6), "LDS：等lgkmcnt / ds指令发不出去": (11.9, 15.5),
            "发射开销（VALU/SALU/MFMA之间的间隔）": (8.0, 8.2), "barrier释放延迟": (1.6, 1.6)},
    idle_D={"VMEM：等vmcnt / buffer_load发不出去": (5.5, 8.5), "LDS：等lgkmcnt / ds指令发不出去": (1.6, 1.5),
            "发射开销（VALU/SALU/MFMA之间的间隔）": (2.4, 2.5), "barrier释放延迟": (0.7, 0.7)},
)


def fig_att():
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.0), dpi=150, gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    rows = [("GateUp（mega）", "G", MFMA_CYCLES["G"]), ("Down（mega）", "D", MFMA_CYCLES["D"])]
    colors = dict(ticket="#5D5D5D", dispatch="#9E9E9E", prologue="#FBD38D", mfma="#2E7D32", kidle="#9CCC65", epilogue="#B9770E")
    labels = dict(ticket="领票", dispatch="分派（重读参数、解码）", prologue="prologue", mfma="K循环：MFMA忙", kidle="K循环：MFMA空闲",
                  epilogue="epilogue")
    ys = [3.0, 2.0]
    for y, (label, kind, mfma) in zip(ys, rows):
        v = {k: statistics.mean(x) for k, x in ATT[kind].items()}
        parts = [("ticket", v["ticket"]), ("dispatch", v["dispatch"]), ("prologue", v["prologue"]), ("mfma", mfma / 1e3),
                 ("kidle", v["kloop"] - mfma / 1e3), ("epilogue", v["epilogue"])]
        x = 0.0
        for key, w in parts:
            ax.barh(y, w, left=x, height=0.6, color=colors[key], edgecolor="white", lw=1)
            if w > 6:
                ax.text(x + w / 2, y, f"{w:.0f}k", ha="center", va="center", fontsize=8.5,
                        color="white" if key in ("ticket", "mfma") else "black")
            x += w
        ax.text(x + 2, y, f"{x:.0f}k（MFMA {mfma / 1e3 / x * 100:.0f}%）", va="center", fontsize=9.5)
    for y, (key, label) in zip((1.0, 0.2), (("transposed", "GateUp单独跑\nA scale转置存"), ("token_major", "GateUp单独跑\nA scale按token存"))):
        pro, loop, epi = ATT["standalone"][key]
        x = 0.0
        for k2, w in (("prologue", pro), ("mfma", MFMA_CYCLES["G"] / 1e3), ("kidle", loop - MFMA_CYCLES["G"] / 1e3), ("epilogue", epi)):
            ax.barh(y, w, left=x, height=0.6, color=colors[k2], edgecolor="white", lw=1)
            if w > 6:
                ax.text(x + w / 2, y, f"{w:.0f}k", ha="center", va="center", fontsize=8.5, color="white" if k2 == "mfma" else "black")
            x += w
        ax.text(x + 2, y, f"K循环 {loop:.0f}k（MFMA {MFMA_CYCLES['G'] / 1e3 / loop * 100:.0f}%）", va="center", fontsize=9.5)
    ax.set_yticks(ys + [1.0, 0.2], [r[0] for r in rows] + ["GateUp单独跑\n（A scale转置存）", "GateUp单独跑\n（A scale按token存）"], fontsize=9.5)
    ax.set_xlim(0, 255)
    ax.set_xlabel("每个任务的shader cycle（千），ATT")
    handles = [Rectangle((0, 0), 1, 1, fc=colors[k]) for k in colors]
    ax.legend(handles, [labels[k] for k in colors], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.8, frameon=False)
    ax.set_title("(a) 每个任务的时间分解（mega：rank 0 的8个CU全部8个wave，两次dispatch的平均）", fontsize=10.5, loc="left")
    ax = axes[1]
    cats = list(ATT["idle_G"])
    for i, (kind, data) in enumerate((("GateUp", ATT["idle_G"]), ("Down", ATT["idle_D"]))):
        x = 0.0
        y = 1 - i
        for j, cat in enumerate(cats):
            w = statistics.mean(data[cat])
            ax.barh(y, w, left=x, height=0.55, color=["#E57373", "#64B5F6", "#BDBDBD", "#FFD54F"][j], edgecolor="white")
            if w > 2.6:
                ax.text(x + w / 2, y, f"{w:.1f}k", ha="center", va="center", fontsize=8.5)
            x += w
        ax.text(x + 0.5, y, f"{x:.1f}k", va="center", fontsize=9.5)
    ax.set_yticks([1, 0], ["GateUp K循环", "Down K循环"], fontsize=9.5)
    ax.set_xlim(0, 52)
    ax.set_xlabel("每个任务K循环里SIMD上MFMA空闲的cycle（千）")
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in ["#E57373", "#64B5F6", "#BDBDBD", "#FFD54F"]]
    ax.legend(handles, cats, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("(b) K循环里MFMA空闲的原因（下一条MFMA的wave在等什么；等barrier时记到最后到达的wave上）", fontsize=10.5,
                 loc="left")
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "att_breakdown.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("font_dir", nargs="?", default="/tmp/fonts")
    parser.add_argument("--from-trace")
    args = parser.parse_args()
    for f in Path(args.font_dir).glob("NotoSansSC-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    if args.from_trace:
        extract(args.from_trace)
    _, tasks = load()
    random.seed(0)
    fig_partition()
    fig_expert_rows()
    fig_swimlane_model()
    fig_swimlane_trace(tasks)
    span, cus, b = fig_budget(tasks)
    print(f"budget: span {span:.0f} us over {cus} CUs: " + ", ".join(f"{k} {v:.1f}" for k, v in b.items()))
    fig_att()


if __name__ == "__main__":
    main()
