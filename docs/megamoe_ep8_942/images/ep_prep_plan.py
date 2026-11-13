"""Figures for docs/megamoe_ep8_942/ep_prep_plan.md (the ep_prep and ep_plan kernels).

    python docs/megamoe_ep8_942/images/ep_prep_plan.py [FONT_DIR] [--from-dir DIR]

FONT_DIR holds NotoSansSC-*.otf (default /tmp/fonts). --from-dir refreshes data/prep_plan.json from a measurement
directory laid out as in ep_prep_plan.md section 6: kt_rank{0..7}/kt_kernel_trace.csv and random/kt_rank{0..7}/ (every
rank under rocprofv3 --kernel-trace, balanced / random routing), phases_{balanced,random}.json (ep_prep phase
timestamps) and tables_{balanced,random}.pt (every rank's tables after a call).
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from mega_moe_impl import arrow, box, canvas, save  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "prep_plan.json"
KERNELS = (("dynamic_per_group_scaled_quant", "quant"), ("FillFunctor", "fill"), ("ep_prep", "ep_prep"),
           ("ep_plan", "ep_plan"), ("mega_kernel", "mega"))
PHASES = ("entry", "route_idx", "counts", "exchange", "tables", "exit")
PLAN_KEY_BASE = 3 * 256 + 8 + 8 * 9
DARK = "#333333"
COL = dict(quant="#9E9E9E", fill="#C9C9C9", plan="#F5A623", mega="#4C9BE8")
PREP_PARTS = (("编号 route_idx", "#2E7D32"), ("写计数 + fence", "#66BB6A"), ("发 CFLAG、等齐 8 个", "#E0E0E0"),
              ("读计数、建表", "#8E24AA"), ("key 直方图", "#CE93D8"))
# Section 5.1's example: 2 ranks x 2 local experts, top-2, 6 tokens per rank, blocks of 4 rows, 2 queues per rank; ep_plan
# with 2 workgroups (workgroup 1's atomic on key 0 lands first), 2 producers sending chunks of 2 routes.
EXAMPLE = dict(ids=[[0, 2, 1, 0, 3, 0, 2, 1, 0, 3, 2, 3], [1, 3, 0, 1, 2, 1, 3, 2, 1, 0, 3, 1]], npes=2, e_local=2,
               block_m=4, queues=2, s_cap=64)
EXAMPLE_FIRST = {0: [1, 0]}


# ----------------------------------------------------------------------------------------------- data


def kernel_timeline(directory):
    """Per rank, the median start / end (us) of each kernel of a call relative to rank 0's ep_prep start (calls 6..)."""
    calls = []
    for r in range(8):
        ks = []
        with open(directory / f"kt_rank{r}" / "kt_kernel_trace.csv") as f:
            for x in csv.DictReader(f):
                name = next((short for key, short in KERNELS if key in x["Kernel_Name"]), None)
                if name:
                    ks.append((name, int(x["Start_Timestamp"]), int(x["End_Timestamp"])))
        per_call = []
        for i, (name, s, e) in enumerate(ks):
            if name != "ep_prep":
                continue
            seq = {"ep_prep": (s, e)}
            for other in ks[max(i - 2, 0):i][::-1] + ks[i + 1:i + 3]:
                seq.setdefault(other[0], other[1:])
            per_call.append(seq)
        calls.append(per_call)
    n = min(len(c) for c in calls)
    out = {}
    for r in range(8):
        t = {}
        for name in ("quant", "fill", "ep_prep", "ep_plan", "mega"):
            pts = [(calls[r][c][name][0] - calls[0][c]["ep_prep"][0], calls[r][c][name][1] - calls[0][c]["ep_prep"][0])
                   for c in range(6, n) if name in calls[r][c]]
            t[name] = [round(statistics.median(p[0] for p in pts) / 1000, 2), round(statistics.median(p[1] for p in pts) / 1000, 2)]
        out[str(r)] = t
    return out


def phase_medians(path):
    d = json.loads(path.read_text())
    return {str(r): {k: round(statistics.median(c[k] for c in calls), 2) for k in PHASES} for r, calls in enumerate(d["ranks"])}


def send_list(path):
    """Rank 0's send list: routes per key, and for the first key the ep_plan workgroup of every entry (runs)."""
    import torch

    d = torch.load(path, weights_only=False)[0]
    route, info = d["send_route"], d["send_info"]
    n = route.numel()
    base = d["plan"][PLAN_KEY_BASE:PLAN_KEY_BASE + 64].tolist() + [n]
    sizes = [base[k + 1] - base[k] for k in range(64)]
    per = -(-n // 128)
    runs = []
    for wg in (route[base[0]:base[1]] // per).tolist():
        if runs and runs[-1][0] == wg:
            runs[-1][1] += 1
        else:
            runs.append([wg, 1])
    per_wg = {}
    for wg, n in runs:
        per_wg[wg] = per_wg.get(wg, 0) + n
    dest = ((info[:8] >> 17) >> 5).tolist()
    return dict(sizes=sizes, key0_runs=runs, key0_wgs=[len(per_wg), min(per_wg.values()), max(per_wg.values())],
                chunk0=dict(tokens=(route[:8] >> 3).tolist(), dest=dest), blocks=d["num_valid"] // 256,
                qstart=d["plan"][3 * 256 + 8:3 * 256 + 8 + 9].tolist())


def token_trace(path, token):
    """Rank 0's routes of one token: expert, route_idx, destination rank / row, send position (balanced dump)."""
    import torch

    d = torch.load(path, weights_only=False)[0]
    route, info = d["send_route"].tolist(), d["send_info"].tolist()
    where = {r: p for p, r in enumerate(route)}
    base = d["plan"][PLAN_KEY_BASE:PLAN_KEY_BASE + 64].tolist()
    rows = []
    for r in range(token * 8, token * 8 + 8):
        p = where[r]
        rows.append(dict(route=r, expert=int(d["ids"].view(-1)[r]), idx=int(d["route_idx"][r]), dest=info[p] >> 22,
                         row=info[p] & ((1 << 17) - 1), pos=p, key=max(k for k in range(64) if base[k] <= p)))
    return rows


def extract(directory):
    directory = Path(directory)
    data = dict(
        timeline=dict(balanced=kernel_timeline(directory), random=kernel_timeline(directory / "random")),
        phases={r: phase_medians(directory / f"phases_{r}.json") for r in ("balanced", "random")},
        send_list={r: send_list(directory / f"tables_{r}.pt") for r in ("balanced", "random")},
        token_trace=token_trace(directory / "tables_balanced.pt", 928),
    )
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(data))
    print("wrote", DATA)


# ----------------------------------------------------------------------------------------------- model


class Model:
    """ep_prep + ep_plan in Python, the kernels' constants as parameters: section 5.1's example runs it with small ones,
    check_model() with the real ones against the kernels' tables. ids[s]: rank s's route experts (route r = token *
    topk + slot); route_idx defaults to route order (the kernel's order within a batch varies)."""

    def __init__(self, ids, npes, e_local, block_m, queues, s_cap, route_idx=None, max_keys=64):
        self.ids, self.npes, self.e_local, self.block_m, self.queues = ids, npes, e_local, block_m, queues
        self.s_cap, self.max_keys = s_cap, max_keys
        e_global = npes * e_local
        if route_idx is None:
            route_idx = []
            for routes in ids:
                seen = [0] * e_global
                route_idx.append([])
                for g in routes:
                    route_idx[-1].append(seen[g])
                    seen[g] += 1
        self.route_idx = route_idx
        self.counts = [[routes.count(g) for g in range(e_global)] for routes in ids]
        self.total = [sum(c[g] for c in self.counts) for g in range(e_global)]
        self.fewest = [min(c[g] for c in self.counts) for g in range(e_global)]
        self.base, self.rows = [0] * e_global, [0] * npes
        for d in range(npes):
            for g in range(d * e_local, (d + 1) * e_local):
                self.base[g] = self.rows[d]
                self.rows[d] += -(-self.total[g] // block_m) * block_m
        self.qstart = [[min(rows, s_cap) // block_m * q // queues for q in range(queues + 1)] for rows in self.rows]

    def row(self, g, idx, me):
        if idx < self.fewest[g]:
            return self.base[g] + idx * self.npes + me
        return self.base[g] + sum(min(c[g], idx + (s < me)) for s, c in enumerate(self.counts))

    def key(self, g, row):
        if row >= self.s_cap:
            return self.max_keys - 1
        qs, block = self.qstart[g // self.e_local], row // self.block_m
        q = max(q for q in range(self.queues) if qs[q] <= block)
        return min(block - qs[q], self.max_keys - 1)

    def routes(self, me):
        """(route, expert, row, key) of each of rank me's routes."""
        out = []
        for r, g in enumerate(self.ids[me]):
            row = self.row(g, self.route_idx[me][r], me)
            out.append((r, g, row, self.key(g, row)))
        return out

    def key_base(self, me):
        hist = [0] * self.max_keys
        for *_, k in self.routes(me):
            hist[k] += 1
        return [sum(hist[:k]) for k in range(self.max_keys)]

    def send_list(self, me, wgs, first=None):
        """[(route, expert, row, key, workgroup)] in send order; first[k] lists the workgroups in the order their
        atomics on key k's cursor land (default: workgroup order)."""
        routes, base = self.routes(me), self.key_base(me)
        per = -(-len(routes) // wgs)
        out = [None] * len(routes)
        for k in range(self.max_keys):
            at = base[k]
            for j in (first or {}).get(k, range(wgs)):
                for x in routes[j * per:(j + 1) * per]:
                    if x[3] == k:
                        out[at] = (*x, j)
                        at += 1
        return out

    def blocks(self, me):
        """(local expert, expected rows) of each of rank me's sorted blocks."""
        out = []
        for b in range(min(self.rows[me], self.s_cap) // self.block_m):
            e = max(e for e in range(self.e_local) if self.base[me * self.e_local + e] <= b * self.block_m)
            g = me * self.e_local + e
            out.append((e, min(self.block_m, self.total[g] - (b * self.block_m - self.base[g]))))
        return out


def check_model(directory):
    """The model against every rank's kernel tables (tables_{balanced,random}.pt): plan, send_info of every route and
    its key range, block tables."""
    import torch

    for routing in ("balanced", "random"):
        dumps = torch.load(Path(directory) / f"tables_{routing}.pt", weights_only=False)
        m = Model([d["ids"].view(-1).tolist() for d in dumps], 8, 32, 256, 8, 73728,
                  route_idx=[d["route_idx"].tolist() for d in dumps])
        for me, d in enumerate(dumps):
            plan = d["plan"].tolist()
            assert d["counts"].tolist() == m.counts
            assert plan[:256] == m.total and plan[256:512] == m.fewest and plan[512:776] == m.base + m.rows
            assert plan[776:848] == sum(m.qstart, []) and plan[848:912] == m.key_base(me)
            base, routes = m.key_base(me) + [len(m.ids[me])], m.routes(me)
            for p, (r, info) in enumerate(zip(d["send_route"].tolist(), d["send_info"].tolist())):
                _, g, row, k = routes[r]
                assert info == (row | g << 17 if row < m.s_cap else -1) and base[k] <= p < base[k + 1], (me, p)
            blocks = m.blocks(me)
            assert d["expert_ids"][:len(blocks)].tolist() == [e for e, _ in blocks]
            assert d["sched"][16:16 + len(blocks)].tolist() == [n for _, n in blocks]
            gu = d["gu_ids"][:256 * len(blocks)].view(-1, 256)
            assert all(gu[b].tolist() == [256 * b + t if t < n else -1 for t in range(256)] for b, (_, n) in enumerate(blocks))
        print(f"model matches the kernels' tables: {routing}, 8 ranks")


# ----------------------------------------------------------------------------------------------- figures


def fig_timeline(data):
    """Measured: every rank's stream from quant to the mega kernel's start, ep_prep split into its phases."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.4), dpi=150, sharex=True)
    x0, x1 = -46, 74
    for ax, routing, title in zip(axes, ("balanced", "random"), ("(a) 均匀路由", "(b) 随机路由")):
        tl, ph = data["timeline"][routing], data["phases"][routing]
        exchanges = []
        for r in range(8):
            y, t, p = 7 - r, tl[str(r)], ph[str(r)]

            def bar(a, b, color, label="", fc="black"):
                a, b = max(a, x0), min(b, x1)
                ax.barh(y, b - a, left=a, height=0.62, color=color, edgecolor="white", lw=0.6)
                if label and b - a > 4.5:
                    ax.text((a + b) / 2, y, label, ha="center", va="center", fontsize=7.6, color=fc)

            bar(*t["quant"], COL["quant"], "quant", "white")
            bar(*t["fill"], COL["fill"], "out 清零")
            # ep_prep: the kernel's measured span, split by the phase timestamps anchored at its end (= exit).
            s, e = t["ep_prep"]
            exch = e - p["exit"]
            exchanges.append(exch)
            edges = [s, exch + p["route_idx"], exch + p["counts"], exch, exch + p["tables"], e]
            for (a, b), (label, color) in zip(zip(edges, edges[1:]), PREP_PARTS):
                bar(a, b, color, "", "white")
            bar(*t["ep_plan"], COL["plan"], "ep_plan")
            bar(t["mega"][0], x1, COL["mega"], "mega kernel →", "white")
        ex = statistics.median(exchanges)
        ax.axvline(ex, color="#B03A2E", ls="--", lw=1.1)
        ax.text(ex + 0.6, 7.62, "计数交换完成（8 个 rank 几乎同时）", color="#B03A2E", fontsize=8.6, va="bottom")
        d = {k: statistics.median(tl[str(r)][k][1] - tl[str(r)][k][0] for r in range(8)) for k in ("ep_prep", "ep_plan")}
        gap = statistics.median(tl[str(r)]["mega"][0] - tl[str(r)]["ep_plan"][1] for r in range(8))
        parts = [statistics.median(ph[str(r)][b] - ph[str(r)][a] for r in range(8))
                 for a, b in zip(PHASES[1:], PHASES[2:])]
        first = statistics.median(ph[str(r)]["route_idx"] - ph[str(r)]["entry"] for r in range(8))
        ax.set_title(f"{title}：ep_prep {d['ep_prep']:.1f} µs（编号 {first:.0f}，写计数 + fence {parts[0]:.0f}，发 CFLAG、等齐 "
                     f"{parts[1]:.0f}，建表 {parts[2]:.0f}，key 直方图 {parts[3]:.0f}），ep_plan {d['ep_plan']:.1f} µs，"
                     f"再过 {gap:.1f} µs mega kernel 开始（各 rank 中位数）", fontsize=10.5, loc="left")
        ax.set_yticks(range(8), [f"rank {7 - i}" for i in range(8)], fontsize=9)
        ax.set_ylim(-0.6, 8.1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].set_xlim(x0, x1)
    axes[1].set_xlabel("µs（相对 rank 0 的 ep_prep 开始；rocprofv3 kernel trace 的中位数，8 个 rank 用同一个时钟）", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in (COL["quant"], COL["fill"])] + \
        [Rectangle((0, 0), 1, 1, fc=c) for _, c in PREP_PARTS] + [Rectangle((0, 0), 1, 1, fc=COL[k]) for k in ("plan", "mega")]
    labels = ["quant（aiter）", "out 清零（BF16 combine）"] + [f"ep_prep：{l}" for l, _ in PREP_PARTS] + ["ep_plan", "mega kernel"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "prep_plan_timeline.png")


def lane(ax, y, h, label, color="#F4F4F4"):
    ax.add_patch(Rectangle((0, y), 100, h, fc=color, ec="#DDDDDD", lw=0.8, zorder=0))
    ax.text(0.6, y + h / 2, label, va="center", fontsize=12.5, fontweight="bold", color=DARK, linespacing=1.35)


def fig_prep_flow():
    """Conceptual swimlanes of ep_prep: one workgroup per rank, the count exchange through every rank's ws."""
    fig, ax = canvas(24, 10.5)
    green, light, purple, red, blue = "#2E7D32", "#66BB6A", "#8E24AA", "#B03A2E", "#1565C0"
    ax.text(0, 99.5, "ep_prep：每个 rank 1 个 WG（1024 线程），和其他 7 个 rank 交换计数（时间从左到右，不按比例）", fontsize=15,
            fontweight="bold", color=DARK, va="top")
    lane(ax, 60.0, 33.0, "rank r：\nep_prep")
    lane(ax, 40.0, 18.5, "rank r 的 ws\n（对称堆）", "#FBF6E9")
    lane(ax, 22.0, 16.5, "其他 7 个\nrank 的\nep_prep", "#EEF4FB")
    steps = [
        (10.5, 7.4, "① arrived[ ] = 0\nhist[ ] = 0", "#A5D6A7"),
        (17.9, 9.8, "② 每条路由：\nroute_idx[r] =\natomicAdd(\nhist[g])\n（LDS，32768 条）", green),
        (27.7, 9.8, "③ hist[0..255]\n写到 8 个 rank\n的 ws.C[r][ ]\n（relaxed\n系统 store）", light),
        (37.5, 6.6, "④ thread\nfence\n_system", light),
        (44.1, 9.0, "⑤ CFLAG[r]\n= epoch\n写到 8 个 rank\n（release）", light),
        (53.1, 7.8, "⑥ 等 8 个\nCFLAG[s]\n== epoch", "#BDBDBD"),
        (60.9, 10.0, "⑦ 读 ws 里的\n8×256 计数：\ntotal、fewest、\nexpert_base、\nsend_base", purple),
        (70.9, 9.6, "⑧ sched 清零；\nplan 表、\nqstart（每个\n队列的起始\nblock）", purple),
        (80.5, 9.8, "⑨ 每个 key 的\n路由数（按段\n算）→ key_base\n（前缀和）", "#CE93D8"),
        (90.3, 9.5, "⑩ num_valid、\nstatus\n（kernel 结束）", "#CE93D8"),
    ]
    for x, w, text, color in steps:
        box(ax, x, 62.0, w - 0.5, 29.0, color, text, fs=11.5, fc="white" if color in (green, light, purple) else DARK)
    data_box = dict(fs=11, ec="#E0C068")
    box(ax, 10.5, 42.0, 8.4, 14.5, "#FFF3D6", "arrived[b] = 0\n（b < s_cap/256）", **data_box)
    box(ax, 27.7, 42.0, 15.4, 14.5, "#FFF3D6", "C[8][256]：第 s 行\n由源 rank s 写\n（每次调用整行覆盖）", **data_box)
    box(ax, 44.1, 42.0, 16.3, 14.5, "#FFF3D6", "CFLAG[8]：存 epoch\n（每次调用 + 1，不清零）", **data_box)
    box(ax, 27.7, 24.0, 15.4, 12.5, "#DCE9F7", "它们的 ③：\nC[s][ ] 也写到 rank r", fs=11)
    box(ax, 44.1, 24.0, 16.3, 12.5, "#DCE9F7", "它们的 ⑤：\nCFLAG[s] 也写到 rank r", fs=11)
    arrow(ax, (14.7, 62.0), (14.7, 56.5), color=green)
    arrow(ax, (32.4, 62.0), (32.4, 56.5), color=green)
    arrow(ax, (48.4, 62.0), (48.4, 56.5), color=green)
    arrow(ax, (38.4, 36.5), (38.4, 42.0), color=blue)
    arrow(ax, (55.5, 36.5), (55.5, 42.0), color=blue)
    arrow(ax, (57.6, 56.5), (57.6, 62.0), color=red)
    ax.text(62.0, 49.2, "③ ⑤ 同样写到其他 7 个 rank 的 ws。\n⑥ 看到 8 个 CFLAG 都等于本次 epoch：8 行计数都已写到，\n⑦ 再读整个 C 矩阵（每个 rank 读到的矩阵相同）。",
            fontsize=11, color=red, va="center", linespacing=1.4)
    notes = [
        "• 跨 rank 只同步这一次：每个 rank 先写计数行（③），fence（④）等它们落到对端，再发 CFLAG（⑤）；看到 8 个 CFLAG（⑥）就能读整个矩阵。",
        "• ① 要在 ⑤ 之前：别的 rank 的 producer 往 rank r 的 arrived 上加，要等它自己的 ⑥ 看到 rank r 的 CFLAG，所以不会加在清零之前。",
        "• ⑦ 之后每个 rank 有同一个 8×256 矩阵，⑦–⑨ 的表（每个 expert 在目标 rank 上的起始行、各队列的起始 block、key）各 rank 算得一样。",
    ]
    for i, line in enumerate(notes):
        ax.text(10.5, 17.5 - i * 5.6, line, fontsize=11.5, color=DARK, va="top")
    save(fig, "prep_flow.png")


def fig_plan_flow():
    """Conceptual swimlanes of ep_plan: one workgroup's steps, the tables it reads and writes, and their readers."""
    fig, ax = canvas(24, 11)
    orange, brown, blue = "#E8743B", "#B9770E", "#1565C0"
    ax.text(0, 99.5, "ep_plan：128 个 WG × 256 线程，WG j 处理本卡路由 [256j, 256j + 256)（32768 条时），然后 mega kernel 用这些表",
            fontsize=15, fontweight="bold", color=DARK, va="top")
    lane(ax, 62.0, 31.5, "rank r：\nep_plan\nWG j")
    lane(ax, 44.0, 16.5, "显存里的表", "#FBF6E9")
    lane(ax, 13.0, 29.5, "rank r：\nmega\nkernel", "#EEF4FB")
    steps = [
        (9.5, 10.4, "⑪ plan 表 → LDS\n（976 个 int：\ntotal、fewest、\nbase、qstart、\nkey_base）", "#FFE0A3"),
        (19.9, 17.6, "⑫ 每个线程 1–4 条路由 r：\ng = topk_ids[r]\nrow = mega_row(g, route_idx[r])\ninfo = row | g << 17\n（超出 s_cap 为 −1）\nkey = row 所在 block 在\n目标队列里的位置", "#F5A623"),
        (37.5, 9.6, "⑬ rank_in_wg =\natomicAdd(\nkey_count[key])\n（LDS）", "#F5A623"),
        (47.1, 13.6, "⑭ 每个有路由的 key k：\nstart[k] = key_base[k]\n+ atomicAdd(\nkey_cursor[k],\nkey_count[k])\n（全局原子加，\n占一段位置）", orange),
        (60.7, 12.0, "⑮ at = start[key]\n+ rank_in_wg\nsend_route[at] = r\nsend_info[at] = info", orange),
        (72.7, 15.0, "⑯ 本卡 block\nb = j, j + 128, …：\nexpert_ids[b]、\nexpected[b]、gu_ids、\npadding 行的\ndown_ids = −1", brown),
    ]
    for x, w, text, color in steps:
        box(ax, x, 64.0, w - 0.5, 27.5, color, text, fs=11.5, fc="white" if color in (orange, brown) else DARK)
    ax.text(88.3, 77.7, "其他 127 个 WG\n同时做 ⑪–⑯。\n同一个 key 里\n谁先做 ⑭ 谁在前，\nkey 内的顺序\n每次可能不同", fontsize=11, color=DARK,
            va="center", linespacing=1.4)
    data_box = dict(fs=11, ec="#E0C068")
    box(ax, 9.5, 46.0, 9.9, 12.5, "#FFF3D6", "plan[976]\n（ep_prep 写）", **data_box)
    box(ax, 19.9, 46.0, 17.1, 12.5, "#FFF3D6", "topk_ids、route_idx、\nws.C（mega_row 读计数）", **data_box)
    box(ax, 47.1, 46.0, 13.1, 12.5, "#FFF3D6", "plan.key_cursor[64]\n（ep_prep 清零）", **data_box)
    box(ax, 60.7, 46.0, 11.5, 12.5, "#FFF3D6", "send_route、\nsend_info\n[本卡路由数]", **data_box)
    box(ax, 72.7, 46.0, 14.5, 12.5, "#FFF3D6", "expert_ids、\nsched.expected、\ngu_ids、down_ids", **data_box)
    arrow(ax, (14.4, 58.5), (14.4, 64.0), color=DARK)
    arrow(ax, (28.4, 58.5), (28.4, 64.0), color=DARK)
    arrow(ax, (53.6, 58.5), (53.6, 64.0), color=orange, style="<|-|>")
    arrow(ax, (66.4, 64.0), (66.4, 58.5), color=orange)
    arrow(ax, (80.0, 64.0), (80.0, 58.5), color=brown)
    consumers = [
        (9.5, 30.5, "领票：sched 的票计数、done、scaled 已由 ep_prep 清零；\n8 个队列的划分（qstart）和 block 数（num_valid）各 rank 一致", "#9E9E9E"),
        (40.5, 22.5, "producer（16 个 WG × 8 wave）：\nWG p 发第 p、p + 16、… 个 chunk（8 条），\n按 send_info 写到目标 rank 的那一行，\n再给那个 block 的 arrived 加 1", "#4C9BE8"),
        (63.5, 18.0, "GateUp(b, n)：expert_ids[b] 选 W1，\n等 arrived[b] ≥ expected[b]，\ngu_ids 区分有效行和 padding", "#F5A623"),
        (82.0, 18.0, "Down(b, n)：按 down_ids\n把每行送回源 rank 的 token，\npadding 行（−1）跳过", orange),
    ]
    for x, w, text, color in consumers:
        box(ax, x, 15.0, w - 0.5, 25.5, color, text, fs=11.5, fc="white")
    arrow(ax, (64.0, 46.0), (54.0, 40.5), color=blue)
    arrow(ax, (75.5, 46.0), (72.0, 40.5), color=brown)
    arrow(ax, (84.5, 46.0), (89.0, 40.5), color=brown)
    notes = [
        "• 行号在 ⑫ 由 mega_row 定死，key 内的顺序只影响发送先后，不影响数据落在哪一行：FP8 combine 的结果每次逐位相同。",
        "• ⑭ 每个 WG 对它有路由的每个 key 做一次全局原子加（最多 64 次）；⑯ 只写本卡的 block，和别的 rank 无关。",
    ]
    for i, line in enumerate(notes):
        ax.text(9.5, 10.5 - i * 5.0, line, fontsize=11.5, color=DARK, va="top")
    save(fig, "plan_flow.png")


def fig_send_list(data):
    """Rank 0's send list: key segments (both routings), the ep_plan workgroups inside key 0, and who sends which chunk."""
    fig, ax = canvas(24, 12.5)
    palette = ["#4C9BE8", "#9CCBF7"]
    ax.text(0, 99.4, "(a) send_route / send_info：rank 0 的 32768 条路由按 key（目标 block 在目标队列里的位置）排好", fontsize=12.5,
            fontweight="bold", color=DARK, va="top")
    for row, routing, label in ((0, "balanced", "均匀路由"), (1, "random", "随机路由")):
        sl = data["send_list"][routing]
        y = 86.0 - row * 10.5
        ax.text(0, y + 2.6, label, fontsize=10, color=DARK, va="center")
        x, scale = 9.0, 90.0 / sum(sl["sizes"])
        for k, n in enumerate(sl["sizes"]):
            if not n:
                continue
            w = n * scale
            ax.add_patch(Rectangle((x, y), w, 5.2, fc=palette[k % 2], ec="white", lw=0.5))
            if w > 3.2:
                ax.text(x + w / 2, y + 3.5, f"k{k}", ha="center", va="center", fontsize=8, color="white")
                ax.text(x + w / 2, y + 1.4, f"{n}", ha="center", va="center", fontsize=7, color="white")
            x += w
        qs = sl["qstart"]
        lo, hi = min(b - a for a, b in zip(qs, qs[1:])), max(b - a for a, b in zip(qs, qs[1:]))
        keys = sum(1 for n in sl["sizes"] if n)
        ax.text(9.0, y - 1.6, f"{keys} 个 key（k0..k{keys - 1}）：key k = 发往 8 个目标 rank 上每个队列第 k 个 block 的路由。"
                f"rank 0 作为目标有 {sl['blocks']} 个 block，每个队列 {lo if lo == hi else f'{lo}–{hi}'} 个",
                fontsize=9.5, color="#555555", va="center")
    bal, rnd = data["send_list"]["balanced"], data["send_list"]["random"]
    ax.text(0, 61.0, "(b) 放大均匀路由的 key 0（2048 条）：每段是一个 ep_plan WG 在 ⑭ 占到的位置，段的先后取决于原子加的先后",
            fontsize=12.5, fontweight="bold", color=DARK, va="top")
    x, scale = 9.0, 90.0 / sum(n for _, n in bal["key0_runs"])
    for i, (wg, n) in enumerate(bal["key0_runs"]):
        w = n * scale
        ax.add_patch(Rectangle((x, 50.5), w, 6.0, fc=["#FFD27A", "#F5A623"][i % 2], ec="white", lw=0.4))
        ax.text(x + w / 2, 53.5, f"{wg}", ha="center", va="center", fontsize=8.5, color=DARK)
        x += w
    count, few, most = bal["key0_wgs"]
    ax.text(9.0, 48.4, f"数字是 WG 号 j（处理 token 32j..32j + 31）。route_idx 大致按 token 顺序编号，均匀路由下 key 0（各队列第一个 block，"
            f"idx < 32）只来自 token 0..1023，即 {count} 个 WG，每个 {few} 条", fontsize=9.5, color="#555555", va="center")
    count, few, most = rnd["key0_wgs"]
    c0 = bal["chunk0"]
    ax.text(9.0, 45.6, f"随机路由的 key 0 有 {rnd['sizes'][0]} 条，来自 {count} 个 WG（每个 {few}–{most} 条）。均匀路由的第一个 chunk（前 8 条）："
            f"token {c0['tokens'][0]}–{c0['tokens'][-1]}，目标 rank {c0['dest']}", fontsize=9.5, color="#555555", va="center")
    ax.text(0, 40.0, "(c) producer 怎么取：chunk c（8 条）由 producer WG c mod 16 发，是它的第 c div 16 个；WG 的 8 个 wave 用 LDS 计数"
            "轮流领（均匀路由，key 0 = chunk 0..255）", fontsize=12.5, fontweight="bold", color=DARK, va="top")
    cw, ch = 5.4, 1.9
    for p in range(16):
        y = 33.0 - p * ch
        ax.text(8.6, y + ch / 2, f"WG {p}", ha="right", va="center", fontsize=7.6, color=DARK)
        for i in range(16):
            c = p + 16 * i
            color = "#4C9BE8" if i < 8 else "#9CCBF7"
            ax.add_patch(Rectangle((9.0 + i * cw, y + 0.1), cw - 0.15, ch - 0.2, fc=color, ec="none"))
            ax.text(9.0 + i * cw + cw / 2, y + ch / 2, f"c{c}", ha="center", va="center", fontsize=6.5,
                    color="white" if i < 8 else DARK)
    ax.text(9.0, 3.4, "深色：8 个 wave 各自先领的一个（i = 0..7，全卡 128 个 wave 一起发 chunk 0..127）；浅色：每个 wave 开始发时预领的下一个"
            "（i = 8..15）。两批发完 key 0 就发完：各目标队列第 0 个 block 在每个源发完约 2 个 chunk 后到齐。", fontsize=9.5,
            color=DARK, va="center")
    ax.text(9.0, 1.2, "之后 key 1 是 chunk 256..511，依此类推；发得慢的 wave 少领几个，WG 里 8 个 wave 差不多一起结束。", fontsize=9.5,
            color=DARK, va="center")
    save(fig, "prep_plan_sendlist.png")


def fig_example():
    """Section 5.1's small example end to end: inputs and numbering, the tables, the receive rows, rank 0's send list."""
    m = Model(**EXAMPLE)
    topk, chunk = 2, 2
    sends = m.send_list(0, 2, first=EXAMPLE_FIRST)
    expert_col = ["#FFD27A", "#F3D1E8", "#C6E2FF", "#D4EDC9"]
    source_col = ["#1565C0", "#E8743B"]
    key_col = ["#A5D6A7", "#E1BEE7"]
    fig, ax = canvas(24, 15.5)

    ax.text(0, 99.6, "(a) 输入和 ② 编号：r = 路由号（2t + slot），e = expert，# = route_idx", fontsize=13, fontweight="bold",
            color=DARK, va="top")
    for s in range(2):
        x0 = 1 + s * 23.5
        ax.text(x0 + 3.4, 93.6, f"rank {s} 的 6 个 token", fontsize=11.5, fontweight="bold", color=source_col[s], va="center")
        for t in range(6):
            y = 87.8 - t * 4.4
            ax.text(x0 + 2.8, y + 1.9, f"t{t}", ha="right", va="center", fontsize=10.5, color=DARK)
            for slot in range(topk):
                r = t * topk + slot
                g = m.ids[s][r]
                box(ax, x0 + 3.4 + slot * 9.4, y, 9.0, 3.8, expert_col[g], f"r{r}：e{g}  #{m.route_idx[s][r]}", fs=10.5,
                    round_=False)
    ax.text(0, 57.8, "③–⑧ 交换后两个 rank 算出同样的表（e0、e1 在 rank 0，e2、e3 在 rank 1）", fontsize=11.5, fontweight="bold",
            color=DARK, va="top")
    table = [("C[0][g]（rank 0 发的）", m.counts[0]), ("C[1][g]（rank 1 发的）", m.counts[1]), ("total", m.total),
             ("fewest", m.fewest), ("base（目标 rank 上的起始行）", m.base)]
    for i, (name, values) in enumerate(table):
        y = 50.5 - i * 3.6
        ax.text(1, y + 1.6, name, va="center", fontsize=10.5, color=DARK)
        for g, v in enumerate(values):
            box(ax, 22 + g * 6.0, y, 5.6, 3.2, expert_col[g] if i < 2 else "#F4F4F4", str(v), fs=10.5, round_=False)
    for g in range(4):
        ax.text(22 + g * 6.0 + 2.8, 55.2, f"e{g}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=DARK)
    ax.text(1, 31.6, f"每个 rank 的行数 {m.rows}，2 个队列：qstart = {m.qstart[0]}（block 0–1 是队列 0，2–3 是队列 1）",
            fontsize=10.5, color=DARK, va="center")

    ax.text(49, 99.6, "(b) 接收端的行（每格一行：源 rank、token、route_idx），block = 4 行", fontsize=13, fontweight="bold",
            color=DARK, va="top")
    cells = [[None] * 16 for _ in range(2)]
    for s in range(2):
        for r, g, row, _ in m.routes(s):
            cells[g // 2][row] = (s, r // topk, m.route_idx[s][r])
    cw = 3.0
    for d in range(2):
        y = 84.5 - d * 27.0
        ax.text(49, y + 8.6, f"rank {d}（e{2 * d}、e{2 * d + 1}）", fontsize=11.5, fontweight="bold", color=DARK, va="center")
        blocks = m.blocks(d)
        for b, (e, n) in enumerate(blocks):
            g = 2 * d + e
            x0 = 49 + b * (4 * cw + 0.8)
            for i in range(4):
                c = cells[d][4 * b + i]
                text, color = ("pad", "#E0E0E0") if c is None else (f"s{c[0]} t{c[1]}\n#{c[2]}", source_col[c[0]])
                box(ax, x0 + i * cw, y, cw - 0.2, 6.2, color, text, fs=9.5, fc=DARK if c is None else "white", round_=False)
                ax.text(x0 + i * cw + cw / 2, y + 6.9, str(4 * b + i), ha="center", va="center", fontsize=8.5, color="#777777")
            q = max(q for q in range(2) if m.qstart[d][q] <= b)
            ax.text(x0 + 2 * cw, y - 3.9, f"b{b}：e{g} 第 {(4 * b - m.base[g]) // 4 + 1} 块\n队列 {q} 第 {b - m.qstart[d][q]} 个"
                    f" → key {m.key(g, 4 * b)}\nexpected = {n}", ha="center", va="center", fontsize=9.5, color=DARK,
                    linespacing=1.3)
    ax.text(49, 47.0, "• e0：rank 0 发了 4 条、rank 1 发了 2 条（fewest = 2）。#0、#1 两轮按 2·idx + s 交错（行 0–3），"
            "rank 0 的 #2、#3 压紧到行 4、5\n  （不压紧就是 4、6，中间空一行）；e1 里 rank 1 的 #2–#4 同样压紧到行 12–14。",
            fontsize=10, color=DARK, va="top", linespacing=1.4)

    ax.text(0, 26.6, "(c) rank 0 的 send_route（ep_plan 2 个 WG：WG 0 管 r0–r5，WG 1 管 r6–r11；key 0 上 WG 1 的原子加先到）",
            fontsize=13, fontweight="bold", color=DARK, va="top")
    cw = 7.6
    for p, (r, g, row, k, j) in enumerate(sends):
        x0 = 1 + p * cw
        box(ax, x0, 9.5, cw - 0.3, 7.6, key_col[k], f"r{r}（t{r // topk}）\n→ rank {g // 2} 行 {row}", fs=10, round_=False)
        ax.text(x0 + cw / 2, 18.0, str(p), ha="center", va="center", fontsize=9, color="#777777")
    runs = []
    for p, (*_, k, j) in enumerate(sends):
        if runs and runs[-1][1:] == [k, j]:
            runs[-1][0].append(p)
        else:
            runs.append([[p], k, j])
    for ps, k, j in runs:
        a, b = 1 + ps[0] * cw, 1 + (ps[-1] + 1) * cw - 0.3
        ax.plot([a, a, b, b], [19.4, 20.2, 20.2, 19.4], color=DARK, lw=1.0)
        ax.text((a + b) / 2, 21.4, f"key {k}：WG {j}", ha="center", va="center", fontsize=10, color=DARK)
    for c in range(len(sends) // chunk):
        a, b = 1 + c * chunk * cw, 1 + (c + 1) * chunk * cw - 0.3
        ax.plot([a, a, b, b], [8.6, 7.8, 7.8, 8.6], color=DARK, lw=1.0)
        ax.text((a + b) / 2, 6.2, f"chunk {c} → producer {c % 2}", ha="center", va="center", fontsize=10, color=DARK)
    ax.text(1, 2.0, "key_base = [0, 8]：key 0 有 8 条（发往两个 rank 上各队列第 0 个 block），key 1 有 4 条。两个 producer 先一起发"
            " chunk 0–3（key 0），再发 key 1。", fontsize=10.5, color=DARK, va="center")
    save(fig, "prep_plan_example.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("font_dir", nargs="?", default="/tmp/fonts")
    parser.add_argument("--from-dir")
    parser.add_argument("--check-model", metavar="DIR", help="compare the model with the tables_*.pt dumps in DIR")
    args = parser.parse_args()
    for f in Path(args.font_dir).glob("NotoSansSC-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    if args.check_model:
        check_model(args.check_model)
    if args.from_dir:
        extract(args.from_dir)
    data = json.loads(DATA.read_text())
    fig_timeline(data)
    fig_prep_flow()
    fig_plan_flow()
    fig_send_list(data)
    fig_example()


if __name__ == "__main__":
    main()
