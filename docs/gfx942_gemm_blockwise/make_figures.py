"""Figures of docs/gfx942_gemm_blockwise/README.md.

The layout figures (C tile ownership, A/B LDS packets, scale mapping) follow the kernel source. The pipeline figures
(prefetch chart and swimlanes) are drawn from the emulation of the final ISA done by
tests/contrib/gemm/audit_gemm_2x4_lds_pipeline_gfx942.py.

usage: python make_figures.py --isa DIR     (DIR written by audit_gemm_2x4_lds_pipeline_gfx942.py --dump DIR)
       python make_figures.py --inline      (README_inline.md: README.md with the images embedded as data URIs)
"""

import argparse
import base64
import io
import os
import re
import sys
from collections import defaultdict

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "contrib", "gemm"))
import audit_gemm_2x4_lds_pipeline_gfx942 as audit  # noqa: E402

IMAGES = os.path.join(HERE, "images")
WAVE_COLORS = ["#4e79a7", "#76b7b2", "#59a14f", "#a0cbe8", "#f28e2b", "#ffbe7d", "#e15759", "#ff9d9a"]
GROUP_COLORS = ["#4e79a7", "#e15759"]
TILE_COLOR = {"A": "#1f77b4", "B": "#ff7f0e"}
KB_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#9c755f"]
# MI300 ds_read_b128 lane groups (archive/bank-conflict): each group of 8 lanes x 16 B must cover the 32 banks once.
B128_GROUPS = [[0, 1, 2, 3, 20, 21, 22, 23], [4, 5, 6, 7, 16, 17, 18, 19], [8, 9, 10, 11, 28, 29, 30, 31], [12, 13, 14, 15, 24, 25, 26, 27]]
B128_GROUPS += [[lane + 32 for lane in group] for group in B128_GROUPS]


def slot_of(row, chunk):
    """Physical 16-byte slot of logical K chunk `chunk` in LDS row `row` (the kernels' XOR swizzle)."""
    return (row ^ chunk) & 7


def owner(row, col):
    """C[row, col] of the 256x256 tile -> (wave, accumulator index)."""
    packet, rep_m, wave_m = row // 64, row % 64 // 32, row % 32 // 16
    side, rep_n, wave_n = col // 128, col % 128 // 64, col % 64 // 16
    return wave_m * 4 + wave_n, (packet * 2 + side) * 4 + rep_m * 2 + rep_n


def save(fig, name):
    fig.savefig(os.path.join(IMAGES, name), dpi=100)
    plt.close(fig)
    print("wrote", os.path.join(IMAGES, name))


def fig_c_tile():
    fig = plt.figure(figsize=(26, 13.5))
    ax = fig.add_axes([0.045, 0.07, 0.50, 0.84])
    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            wave, acc = owner(r, c)
            ax.add_patch(Rectangle((c, r), 16, 16, facecolor=WAVE_COLORS[wave], edgecolor="white", lw=0.8))
            ax.text(c + 8, r + 5.5, f"w{wave}", ha="center", va="center", fontsize=8, fontweight="bold")
            ax.text(c + 8, r + 11.5, f"acc{acc}", ha="center", va="center", fontsize=7)
    for y in range(0, 257, 64):
        ax.axhline(y, color="black", lw=2.4)
    for y in range(32, 256, 64):
        ax.axhline(y, color="black", lw=0.9, ls="--")
    ax.axvline(128, color="black", lw=2.8)
    for x in (64, 192):
        ax.axvline(x, color="black", lw=1.3, ls="--")
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_xticks(range(0, 257, 32))
    ax.set_yticks(range(0, 257, 32))
    ax.xaxis.tick_top()
    ax.set_xlabel("n = side*128 + rep_n*64 + wave_n*16 + (lane//16)*4 + v      (wave_n = wave % 4)", fontsize=11)
    ax.set_ylabel("m = packet*64 + rep_m*32 + wave_m*16 + lane%16      (wave_m = wave // 4)", fontsize=11)
    for p in range(4):
        ax.text(259, 64 * p + 32, f"A packet {p}\n= stage {p}\n(rep_m 0 | 1\nsplit at +32)", ha="left", va="center", fontsize=10.5, fontweight="bold")
    for s in range(2):
        ax.text(128 * s + 64, 262, f"side {s} = B packets {2 * s} (rep_n 0) and {2 * s + 1} (rep_n 1)", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.set_title("(a) 256x256 C tile of one workgroup: owner wave (w) and its accumulator (acc = (packet*2+side)*4 + rep_m*2 + rep_n, 4 FP32 VGPR)",
                 loc="left", fontsize=13, pad=30)
    fig.legend(handles=[Patch(color=WAVE_COLORS[w], label=f"wave {w} (group {w // 4}, wave_m {w // 4}, wave_n {w % 4})") for w in range(8)],
               loc="upper left", bbox_to_anchor=(0.60, 0.36), ncol=2, fontsize=10.5, frameon=False)

    az = fig.add_axes([0.66, 0.42, 0.32, 0.48])
    for m in range(16):
        for n in range(16):
            lane, v = (n // 4) * 16 + m, n % 4
            az.add_patch(Rectangle((n, m), 1, 1, facecolor=plt.cm.tab10(n // 4), alpha=0.3 + 0.17 * v, edgecolor="white"))
            az.text(n + 0.5, m + 0.5, f"L{lane}\nv{v}", fontsize=6.5, ha="center", va="center")
    az.set_xlim(0, 16)
    az.set_ylim(16, 0)
    az.set_xticks(range(16))
    az.set_yticks(range(16))
    az.xaxis.tick_top()
    az.set_xlabel("n within the 16x16 block", fontsize=10)
    az.set_ylabel("m within the 16x16 block", fontsize=10)
    az.set_title("(b) one v_mfma_f32_16x16x32_fp8_fp8 result: lane L, VGPR v holds C[m = L%16][n = 4*(L//16) + v]", loc="left", fontsize=12, pad=26)
    fig.text(0.60, 0.015, "\n".join([
        "Operands are swapped: srcA = B fragment (lane L: B row n0 + L%16,",
        "8 k bytes of k group L//16), srcB = A fragment (lane L: A row m0 + L%16,",
        "the same 8 k). D = srcA * srcB^T is D[n][m]: each lane owns 4 consecutive n",
        "of one row m, stored with one buffer_store_dwordx2 (4 BF16).",
        "",
        "Per wave: 4 packets x 2 sides x 2 rep_m x 2 rep_n = 32 accumulator tiles",
        "= 128 FP32 VGPR, 32 stores per lane. A K-block partial (4 chained MFMA = K128)",
        "is scaled by ScaleA[m, kb] * ScaleB[n/128, kb] and FMA-ed into acc.",
        "BF16 packing: pf1 truncates like PyHIP (+0x8000 denormal, v_perm),",
        "pf2 rounds to nearest even. Columns n >= N get voffset 0x7FFFFFFF",
        "(dropped by the buffer range check); rows m >= M are outside num_records.",
    ]), fontsize=12, va="bottom", family="monospace")
    save(fig, "c_tile_waves.png")


def _bank_panel(ax, title, rows_of_lane, chunk_of_lane):
    """Physical slot (= bank quad) hit by each lane of the 8 ds_read_b128 lane groups."""
    ok = True
    for g, group in enumerate(B128_GROUPS):
        slots = [slot_of(rows_of_lane(lane), chunk_of_lane(lane)) for lane in group]
        ok &= len(set(slots)) == 8
        for i, (lane, slot) in enumerate(zip(group, slots)):
            ax.add_patch(Rectangle((i, g), 1, 1, facecolor=plt.cm.tab10(slot % 10), alpha=0.55, edgecolor="white"))
            ax.text(i + 0.5, g + 0.5, f"L{lane}\n{slot}", ha="center", va="center", fontsize=6.5)
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)
    ax.set_xticks([])
    ax.set_yticks([g + 0.5 for g in range(8)], [f"group {g}" for g in range(8)], fontsize=8)
    ax.set_title(f"{title}: {'conflict free' if ok else 'BANK CONFLICT'}", fontsize=10, loc="left")
    return ok


def fig_lds(tile):
    """One packet of the A or B LDS tile: XOR-swizzled 16-byte slots, the waves writing them and the waves reading them."""
    fig = plt.figure(figsize=(26 if tile == "A" else 32, 14))
    ax0 = fig.add_axes([0.02, 0.08, 0.085 if tile == "A" else 0.07, 0.80])
    for i, (t, p) in enumerate((t, p) for t in "AB" for p in range(4)):
        ax0.add_patch(Rectangle((0, i * 8), 1, 8, facecolor=TILE_COLOR[t], alpha=0.2 + 0.15 * p, edgecolor="black", lw=2.5 if t == tile else 0.8))
        what = f"M rows {64 * p}..{64 * p + 63}\nread in stage {p}" if t == "A" else f"N rows {64 * p}..{64 * p + 63}\nside {p // 2}, rep_n {p % 2}"
        ax0.text(0.5, i * 8 + 4, f"{t}{p}  @ {i * 8192:#06x}\n{what}", ha="center", va="center", fontsize=9.5)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(64, 0)
    ax0.set_xticks([])
    ax0.set_yticks(range(0, 65, 8), [f"{k} KiB" for k in range(0, 65, 8)])
    ax0.set_title("(a) 64 KiB LDS", loc="left", fontsize=12)

    panels = [("plain", "writer wave, B[N,K] row-major" if tile == "B" else "writer wave (tid = 8*row + chunk)")]
    if tile == "B":
        panels.append(("preshuffle", "writer wave, preshuffled B"))
    width, step = (0.34, 0.40) if tile == "A" else (0.215, 0.27)
    for j, (mode, title) in enumerate(panels):
        ax = fig.add_axes([0.145 + j * step, 0.08, width, 0.80])
        for row in range(64):
            for slot in range(8):
                chunk = slot ^ (row & 7)
                if mode == "plain":
                    wave = (row * 8 + chunk) // 64
                else:  # wave w loads 16 rows (w // 2) x 4 chunks (w % 2) of 16 B per lane
                    wave = (row // 16) * 2 + chunk // 4
                ax.add_patch(Rectangle((slot, row), 1, 1, facecolor=WAVE_COLORS[wave], edgecolor="white", lw=0.5))
                ax.text(slot + 0.5, row + 0.5, f"k{chunk * 16}", ha="center", va="center", fontsize=6.5)
        ax.axhline(32, color="black", lw=2.5)
        ax.set_xlim(0, 8)
        ax.set_ylim(64, 0)
        ax.set_xticks([s + 0.5 for s in range(8)], [f"slot {s}\nB {16 * s}..\nbank {4 * s}-{4 * s + 3}" for s in range(8)], fontsize=8)
        ax.set_yticks(range(0, 64, 4))
        ax.set_ylabel(f"row in packet ({'m' if tile == 'A' else 'n'} - 64*packet)")
        ax.set_title(f"({'bc'[j]}) one {tile} packet, 64 rows x 128 B: {title}\n'k<x>' = K bytes x..x+15 in slot (row ^ chunk) & 7",
                     loc="left", fontsize=11)
        ax.text(8.15, 16, "rows 0-31:\nwaves 0-3\n(group 0)", fontsize=10, va="center", color=GROUP_COLORS[0], fontweight="bold")
        ax.text(8.15, 48, "rows 32-63:\nwaves 4-7\n(group 1)", fontsize=10, va="center", color=GROUP_COLORS[1], fontweight="bold")

    x0 = 0.145 + (len(panels) - 1) * step + width + 0.075
    ar = fig.add_axes([x0, 0.50, 0.97 - x0, 0.38])
    for row in range(64):
        if tile == "A":
            wave_m = row % 32 // 16
            ar.add_patch(Rectangle((0, row), 1, 1, facecolor=GROUP_COLORS[wave_m], alpha=0.75, edgecolor="white", lw=0.3))
            ar.text(1.1, row + 0.5, f"rep_m {row // 32}, lane%16 = {row % 16}" if row % 16 == 0 else "", fontsize=7.5, va="center")
        else:
            ar.add_patch(Rectangle((0, row), 1, 1, facecolor=WAVE_COLORS[row // 16], alpha=0.75, edgecolor="white", lw=0.3))
            ar.add_patch(Rectangle((1, row), 1, 1, facecolor=WAVE_COLORS[4 + row // 16], alpha=0.75, edgecolor="white", lw=0.3))
    ar.set_xlim(0, 6)
    ar.set_ylim(64, 0)
    ar.set_xticks([])
    ar.set_yticks(range(0, 65, 16))
    if tile == "A":
        ar.set_title("(d) readers: group 0 (wave_m 0) reads rows 0-15, 32-47; group 1 rows 16-31, 48-63.\n"
                     "All 4 waves of a group read the same rows (their B columns differ):\n"
                     "4 x ds_read_b128 per wave = 2 rep_m x 2 half_k, in stage p of the packet.", loc="left", fontsize=10.5)
        legend = [Patch(color=GROUP_COLORS[g], label=f"read by group {g} (waves {4 * g}-{4 * g + 3})") for g in range(2)]
    else:
        ar.set_title("(d) readers: wave w reads rows 16*(w%4)..+15 of every B packet\n"
                     "(left column group 0, right column group 1: both groups read all rows).\n"
                     "8 x ds_read_b128 per wave per K block = 2 sides x 2 rep_n x 2 half_k.", loc="left", fontsize=10.5)
        legend = [Patch(color=WAVE_COLORS[w], label=f"wave {w}") for w in range(8)]
    ar.legend(handles=legend, loc="lower right", fontsize=8.5, frameon=False)

    ab = fig.add_axes([x0, 0.08, (0.97 - x0) * 0.48, 0.32])
    _bank_panel(ab, "(e) ds_read_b128", lambda lane: lane % 16, lambda lane: lane // 16)
    aw = fig.add_axes([x0 + (0.97 - x0) * 0.52, 0.08, (0.97 - x0) * 0.48, 0.32])
    if tile == "A":
        _bank_panel(aw, "(f) ds_write_b128", lambda lane: lane // 8, lambda lane: lane % 8)
    else:  # preshuffled B, wave 0: lane -> row lane % 16, chunk lane // 16 (plain B writes like A)
        _bank_panel(aw, "(f) ds_write_b128 (preshuffled B)", lambda lane: lane % 16, lambda lane: lane // 16)
    fig.suptitle(f"{tile} tile in LDS (pf1 and pf2 share this layout): 4 packets x 64 rows x K128 FP8, one K block, written cooperatively "
                 f"by all 512 threads from VGPR staging.   (e)/(f): 16-B slot (= 4-bank quad) of each lane of the 8 MI300 ds_read_b128 lane "
                 f"groups; 8 distinct slots per group = no bank conflict (half_k 1 XORs the slot with 4)", fontsize=13, x=0.02, ha="left", y=0.975)
    save(fig, f"lds_{tile.lower()}_layout.png")


def fig_scales():
    """ScaleA / ScaleB: global layout, which lane loads what, and the register slots of pf1 and pf2 (no LDS)."""
    fig = plt.figure(figsize=(26, 14))
    ax = fig.add_axes([0.04, 0.10, 0.20, 0.78])
    for row in range(0, 256, 16):
        packet, rep_m, wave_m = row // 64, row % 64 // 32, row % 32 // 16
        for kb in range(4):
            ax.add_patch(Rectangle((kb, row), 1, 16, facecolor=GROUP_COLORS[wave_m], alpha=0.25 + 0.15 * packet, edgecolor="white"))
        ax.text(4.1, row + 8, f"p{packet} r{rep_m} g{wave_m}", fontsize=8, va="center")
    ax.set_xlim(0, 5.3)
    ax.set_ylim(256, 0)
    ax.set_xticks([k + 0.5 for k in range(4)], [f"kb {k}" for k in range(4)])
    ax.set_yticks(range(0, 257, 32))
    ax.set_ylabel("row m of the tile")
    ax.set_title("(a) ScaleA^T [K/128, M] slice: 256 rows x KB\nlane L of group g loads row\n64p + 32r + 16g + L%16 (p packet, r rep_m);\n"
                 "lanes L, L+16, L+32, L+48 load the same row", loc="left", fontsize=10.5)

    bx = fig.add_axes([0.28, 0.62, 0.16, 0.26])
    for side in range(2):
        for kb in range(4):
            bx.add_patch(Rectangle((kb, side), 1, 1, facecolor=TILE_COLOR["B"], alpha=0.3 + 0.3 * side, edgecolor="white"))
            bx.text(kb + 0.5, side + 0.5, f"[nb+{side}, {kb}]", ha="center", va="center", fontsize=9)
    bx.set_xlim(0, 4)
    bx.set_ylim(2, 0)
    bx.set_xticks([k + 0.5 for k in range(4)], [f"kb {k}" for k in range(4)])
    bx.set_yticks([0.5, 1.5], ["side 0", "side 1"])
    bx.set_title("(b) ScaleB [ceil(N/128), K/128]: nb = col_base/128;\nside 1 clamped to the last N block; uniform per wave", loc="left", fontsize=10.5)

    rows = [
        ("pf1", "scale_a[slot*2 + rep_m]  slot = packet % 2", "4 VGPR",
         ["s0: load sA(k, p1) -> slot 1", "s1: load sA(k, p2) -> slot 0", "s2: load sA(k, p3) -> slot 1", "s3: load sA(k+1, p0) -> slot 0"],
         "buffer_load_dword (VMEM), one stage ahead; stage p reads slot p%2"),
        ("pf1", "scale_b[slot*2 + side]  slot = K block % 2", "4 SGPR",
         ["s0 (C phase): s_buffer_load sB(k+1) -> slot (k+1)%2", "", "", ""],
         "inline-asm s_buffer_load_dword (SMEM, invisible to LLVM): retired by the next lgkmcnt(0)"),
        ("pf2", "scale_a[packet*2 + rep_m]", "8 VGPR",
         ["s0: load sA(k, p3) -> [6,7]", "s1: load sA(k+1, p0) -> [0,1]", "s2: load sA(k+1, p1) -> [2,3]", "s3: load sA(k+1, p2) -> [4,5]"],
         "buffer_load_dword; a slot is refilled for the next K block one stage after its use"),
        ("pf2", "scale_b[side], scale_b_next[side]", "2+2 VGPR",
         ["s0: load sB(k+1) -> scale_b_next", "", "", "end of block: scale_b = scale_b_next"],
         "uniform buffer_load_dword (VMEM, tracked by LLVM waits)"),
    ]
    cx = fig.add_axes([0.28, 0.06, 0.70, 0.44])
    cx.set_xlim(0, 10)
    cx.set_ylim(len(rows) * 2.2, -0.6)
    cx.axis("off")
    for i, (kernel, slots, size, stages, how) in enumerate(rows):
        y = i * 2.2
        cx.add_patch(Rectangle((0, y + 0.1), 2.3, 1.6, facecolor="#eeeeee", edgecolor="black"))
        cx.text(0.08, y + 0.55, f"{kernel}: {slots}\n({size})", fontsize=10, va="center")
        cx.text(0.08, y + 1.3, how.replace(": ", ":\n").replace("; ", ";\n"), fontsize=8, color="#444444", va="center")
        for s, text in enumerate(stages):
            cx.add_patch(Rectangle((2.45 + s * 1.9, y + 0.1), 1.8, 1.6, facecolor=GROUP_COLORS[0], alpha=0.12 if text else 0.03, edgecolor="#888888"))
            cx.text(2.5 + s * 1.9, y + 0.9, text.replace(" -> ", "\n-> "), fontsize=9, va="center")
            if i == 0:
                cx.text(2.45 + s * 1.9 + 0.9, -0.3, f"stage {s} of K block k", ha="center", fontsize=10, fontweight="bold")
    cx.set_title("(c) register slots per stage (both groups run the same code, group 1 one barrier later). "
                 "Stage s consumes scale_a of A packet s and scale_b of the block.", loc="left", fontsize=11)
    fig.text(0.50, 0.89, "Scales are not staged in LDS: 4 B per row and per N block per K block are loaded straight into registers.\n"
             "Per stage and side: one VALU product sA*sB per rep_m (partial_scale / pending_scale), then one FMA per accumulator element.",
             fontsize=12, va="top", wrap=True)
    fig.suptitle("ScaleA / ScaleB mapping and register slots", fontsize=14, x=0.02, ha="left", y=0.985)
    save(fig, "scale_layout.png")


# ---------------------------------------------------------------------------------------------------------------
# Pipeline figures from the emulated ISA.


class Run:
    """Events of wave 0 (group 0) and wave 4 (group 1, same SIMD) of one emulated workgroup."""

    def __init__(self, config, isa_dir):
        self.config = config
        self.res = audit.audit(config, audit.isa_path(isa_dir, config))
        assert not self.res.failed, f"{config} does not pass the audit"
        self.prog, self.emu = self.res.prog, self.res.emu
        self.trace = {0: self.emu.waves[0].trace, 1: self.emu.waves[4].trace}
        self.by_seq = {g: {ev.seq: ev for ev in tr} for g, tr in self.trace.items()}
        self.labels = {g: audit.epoch_labels(self.emu, self.prog, 4 * g) for g in (0, 1)}

    def kind(self, ev):
        return self.prog.instrs[ev.index].kind

    def ev_of(self, op):
        """Traced event of the same instruction instance as `op` (any wave of the group runs the same sequence)."""
        return self.by_seq[op.wave // 4].get(op.seq)

    def ready_ev(self, group, op):
        return self.by_seq[group].get(op.cseq)

    def lds_events(self, group, last_epoch):
        out = []
        for ev in self.trace[group]:
            if ev.epoch <= last_epoch and self.kind(ev) in ("ds_read", "ds_write"):
                out.append(ev)
        return out

    def binding(self):
        """Cross-group LDS dependencies with no spare barrier, as (kind, slot, producer event, consumer event)."""
        seen, out = set(), []
        for kind, region, packet, prod, cons in self.emu.lds.binding:
            src, dst = self.ev_of(prod), self.ev_of(cons)
            key = (kind, region, packet, prod.wave // 4, prod.seq, cons.wave // 4, cons.seq)
            if src is not None and dst is not None and key not in seen:
                seen.add(key)
                out.append((kind, f"{region}{packet}", prod.wave // 4, src, cons.wave // 4, dst))
        return out


def timeline(run, epochs):
    """Nominal issue times of both traced waves (one SIMD) and the barrier releases.

    The two waves share the SIMD's XDL pipe (16 cycles per 16x16x32 FP8 MFMA; the issuing wave moves on after 4) and
    VALU (4 cycles); other instructions cost 4 cycles (scalar 1, s_nop n+1). Memory latency is not modeled.
    """
    times, release, xdl, valu = {}, [0.0], 0.0, 0.0
    queues = {g: defaultdict(list) for g in run.trace}
    for g, tr in run.trace.items():
        for ev in tr:
            queues[g][ev.epoch].append(ev)
    for epoch in range(epochs):
        t, idx = {g: release[-1] for g in run.trace}, {g: 0 for g in run.trace}
        while any(idx[g] < len(queues[g][epoch]) for g in run.trace):
            g = min((k for k in run.trace if idx[k] < len(queues[k][epoch])), key=lambda k: t[k])
            ev = queues[g][epoch][idx[g]]
            idx[g] += 1
            ins = run.prog.instrs[ev.index]
            if ins.kind == "mfma":
                start = max(t[g], xdl)
                xdl, t[g] = start + 16, start + 4
                times[id(ev)] = (start, start + 16)
            elif ins.kind == "valu":
                start = max(t[g], valu)
                valu = t[g] = start + 4
                times[id(ev)] = (start, start + 4)
            else:
                cost = 1 + int(ins.ops[0].value) if ins.mnem == "s_nop" else (1 if ins.kind in ("salu", "branch", "waitcnt", "barrier", "end") else 4)
                times[id(ev)] = (t[g], t[g] + cost)
                t[g] += cost
        release.append(max(t.values()))
    return times, release


def short(label):
    phase, kb, stage = label
    return f"k{kb}.s{stage}\n{phase}" if phase in ("C", "M") else phase


def fig_prefetch(run, blocks=3):
    """Epoch-level life of every A/B packet version: global load -> VGPR staging -> ds_write -> reads of both groups."""
    last = 8 * blocks + 1
    fig = plt.figure(figsize=(28, 13.5))
    ax = fig.add_axes([0.075, 0.12, 0.905, 0.72])
    slots = [f"{t}{p}" for t in "AB" for p in range(4)]
    row = {s: i for i, s in enumerate(slots)}
    # VGPR staging of group 0: global load issue -> the ds_write that consumes it.
    for rec in audit.register_lifetimes(run.emu, run.prog, 0):
        if rec["class"] in ("A stage", "B stage") and rec["last"] is not None and rec["define"].epoch <= last:
            slot = rec["label"].split("(")[0]
            y = row[slot] - 0.36
            e0, e1 = rec["define"].epoch, rec["last"].epoch
            ax.add_patch(Rectangle((e0 + 0.85, y), e1 - e0 + 0.3, 0.24, facecolor="white", edgecolor=KB_COLORS[rec["kb"] % 5], hatch="////", lw=1.2))
            ax.text(e0 + 0.9, y + 0.12, f"load k{rec['kb']}", fontsize=8, va="center", color=KB_COLORS[rec["kb"] % 5])
    # LDS residency per version and the accesses of both groups.
    versions = defaultdict(lambda: {"w": [], "r": []})
    for g in (0, 1):
        for ev in run.lds_events(g, last):
            info = ev.op.info
            if info.get("kb"):
                versions[(f"{info['region']}{info['packet']}", info["kb"][0])]["w" if run.kind(ev) == "ds_write" else "r"].append((g, ev))
    for (slot, kb), acc in versions.items():
        y = row[slot]
        writes = [ev.epoch for _, ev in acc["w"]]
        ends = [ev.op.complete for _, ev in acc["r"]] or [max(writes)]
        if writes:
            x0, x1 = min(writes) + 0.85, min(max(ends), last + 0.4) + 1.15
            ax.add_patch(Rectangle((x0, y - 0.08), x1 - x0, 0.42, facecolor=KB_COLORS[kb % 5], alpha=0.22, edgecolor=KB_COLORS[kb % 5]))
            ax.text(x0 + 0.05, y + 0.26, f"{slot} holds k{kb}", fontsize=8.5, color=KB_COLORS[kb % 5], fontweight="bold")
        for kind, items in acc.items():
            for g, ev in items:
                marker = "v" if kind == "w" else "o"
                ax.plot(ev.epoch + (0.85 if g == 0 else 1.15), y + 0.12, marker=marker, ms=9,
                        mfc=GROUP_COLORS[g] if g == 0 else "white", mec=GROUP_COLORS[g], mew=1.6, ls="none")
    for kind, slot, pg, src, cg, dst in run.binding():
        if dst.epoch <= last:
            color = "#c0392b" if kind == "WAR" else "#1e8449"
            ax.annotate("", xy=(dst.epoch + 0.85, row[slot] + 0.12), xytext=(src.epoch + 1.15, row[slot] + 0.12),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, connectionstyle="arc3,rad=-0.35"))
    for e in range(last + 2):
        ax.axvline(e + 0.5, color="black", lw=0.6, ls=(0, (4, 3)))
    for g in (0, 1):
        for e in range(last + 1):
            phase = run.labels[g].get(e, ("idle", None, None))
            y = -2.25 + 0.8 * g
            ax.add_patch(Rectangle((e + 0.5, y), 1, 0.75, facecolor={"C": "#fde9e9", "M": "#e3f2e3"}.get(phase[0], "#eeeeee"), edgecolor="white"))
            ax.text(e + 1.0, y + 0.37, short(phase), ha="center", va="center", fontsize=7.5)
        ax.text(0.45, -1.87 + 0.8 * g, f"group {g}\n(waves {4 * g}-{4 * g + 3})", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0.5, last + 1.5)
    ax.set_ylim(len(slots) - 0.4, -2.35)
    ax.set_yticks(range(len(slots)), [f"LDS {s}" for s in slots], fontsize=11)
    ax.set_xticks([e + 1 for e in range(last + 1)], [str(e) for e in range(last + 1)])
    ax.set_xlabel("barrier epoch (interval between two s_barrier of the workgroup)", fontsize=11)
    variant = run.res.cfg["variant"]
    lead = "1 stage (loads of K block k+1 are written to LDS one stage later)" if variant == "pf1" else "2 stages (some loads run 2 K blocks ahead)"
    fig.suptitle(f"{variant}: how A/B are prefetched, epoch by epoch (prolog + K blocks 0..{blocks - 1}), load lead {lead}", fontsize=14, x=0.01, ha="left", y=0.975)
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="////", label="group-0 VGPR staging: global load issue -> ds_write"),
        Patch(facecolor="#888888", alpha=0.25, label="LDS version: first write -> last read retired"),
        Line2D([], [], ls="none", marker="v", ms=9, mfc=GROUP_COLORS[0], mec=GROUP_COLORS[0], label="ds_write group 0"),
        Line2D([], [], ls="none", marker="v", ms=9, mfc="white", mec=GROUP_COLORS[1], mew=1.6, label="ds_write group 1"),
        Line2D([], [], ls="none", marker="o", ms=9, mfc=GROUP_COLORS[0], mec=GROUP_COLORS[0], label="ds_read group 0"),
        Line2D([], [], ls="none", marker="o", ms=9, mfc="white", mec=GROUP_COLORS[1], mew=1.6, label="ds_read group 1"),
        Line2D([], [], color="#c0392b", lw=1.6, label="WAR with 0 spare barriers (group 1 read -> group 0 overwrite)"),
        Line2D([], [], color="#1e8449", lw=1.6, label="RAW with 0 spare barriers (group 1 write -> group 0 read)"),
    ], loc="lower left", bbox_to_anchor=(0.075, 0.005), ncol=4, fontsize=10, frameon=False)
    save(fig, f"prefetch_{variant}.png")


def pack(bars, gap):
    """Greedy rows so that bars sharing a row are at least `gap` apart; sets bar['row']."""
    ends = []
    for b in sorted(bars, key=lambda b: b["s"]):
        for r, end in enumerate(ends):
            if end + gap <= b["s"]:
                ends[r], b["row"] = b["e"], r
                break
        else:
            ends.append(b["e"])
            b["row"] = len(ends) - 1
    return max(len(ends), 1)


def slot_runs(infos):
    """'B0-3 x8 k1' style label of merged LDS accesses."""
    slots = sorted({(i["region"], i["packet"]) for i in infos})
    kbs = sorted({k for i in infos for k in i.get("kb", [])})
    packets = [p for _, p in slots]
    text = f"{slots[0][0]}{packets[0]}" + (f"-{packets[-1]}" if len(packets) > 1 else "")
    return text + (f" x{len(infos)}" if len(infos) > 1 else "") + (f" k{','.join(map(str, kbs))}" if kbs else "")


def fig_swimlane(run, blocks=3):
    emu, prog = run.emu, run.prog
    last = 8 * blocks
    times, release = timeline(run, last + 2)

    def t(ev):
        return times[id(ev)][0] if ev is not None and id(ev) in times else None

    def t_ready(group, op):
        return t(run.ready_ev(group, op))

    ev0 = [ev for ev in run.trace[0] if ev.epoch <= last]
    lanes = defaultdict(list)

    def bar(lane, s, e, label="", color="#888888", ready=None, row=None, hatch=None, alpha=0.85):
        e = s + 4 if e is None else e
        lanes[lane].append(dict(s=s, e=max(e, s + 4), ready=ready, label=label, color=color, row=row, hatch=hatch, alpha=alpha))

    scale_color = {"SA": "#2ca02c", "SB": "#9467bd"}
    waits = []
    for ev in ev0:
        kind, ins = run.kind(ev), prog.instrs[ev.index]
        if kind in ("vmem_load", "smem") and ev.op.info.get("tensor") in ("A", "B", "SA", "SB"):
            info = ev.op.info
            lane = "load" if info["tensor"] in ("A", "B") else "scale"
            color = TILE_COLOR.get(info["tensor"]) or scale_color[info["tensor"]]
            bar(lane, t(ev), t_ready(0, ev.op) or release[-1], audit.fmt_load(info) + (" (smem)" if kind == "smem" else ""), color)
        elif kind == "waitcnt":
            vm = sum(op.queue == "vm" for op in ev.retired)
            for row, name, count in ((0, "vmcnt", vm), (1, "lgkmcnt", len(ev.retired) - vm)):
                if name in ins.mods:
                    waits.append((t(ev), row, f"{name[:-3]}{ins.mods[name]}", count))
    merged = defaultdict(list)
    for ev in ev0:
        if run.kind(ev) in ("ds_read", "ds_write"):
            merged[(run.kind(ev), ev.epoch, ev.op.info["region"])].append(ev)
    for (kind, epoch, region), evs in merged.items():
        end = max(t_ready(0, ev.op) or t(ev) for ev in evs)
        bar("write" if kind == "ds_write" else "read", min(t(ev) for ev in evs), end, slot_runs([ev.op.info for ev in evs]), TILE_COLOR[region], row="AB".index(region))
    for kind in ("mfma", "valu"):
        run_evs = []
        for ev in [ev for ev in ev0 if run.kind(ev) == kind] + [None]:
            if ev is not None and run_evs and ev.epoch == run_evs[-1].epoch and t(ev) - times[id(run_evs[-1])][1] < 64:
                run_evs.append(ev)
                continue
            if run_evs:
                bar(kind, t(run_evs[0]), times[id(run_evs[-1])][1], str(len(run_evs)), "#e8605c" if kind == "mfma" else "#a8a8a8", row=0)
            run_evs = [ev] if ev is not None else []
    # Register resources of wave 0: VGPR staging per physical range, operand sets, scale slots, MFMA partial chains.
    stage_rows = {}
    operands = defaultdict(list)
    for rec in audit.register_lifetimes(emu, prog, 0):
        d, last_use = rec["define"], rec["last"]
        if d.epoch > last or last_use is None:
            continue
        ready = t(run.by_seq[0].get(rec["ready"][1]))
        cls = rec["class"]
        if cls in ("A stage", "B stage"):
            key = (cls, rec["regs"][0], rec["regs"][-1])
            stage_rows.setdefault(key, len([k for k in stage_rows if k[0] == cls]))
            bar(cls, t(d), t(last_use), rec["label"], TILE_COLOR[cls[0]], ready=ready, row=stage_rows[key])
        elif cls in ("A operand", "B operand"):
            operands[(cls, rec["kb"], d.epoch)].append((t(d), ready, t(last_use)))
        elif cls in ("scale_a", "scale_b"):
            bar(cls, t(d), t(last_use), rec["label"].replace("(", " ").replace(")", ""), scale_color["SA" if cls == "scale_a" else "SB"], ready=ready)
        elif cls == "partial":
            side = (d.info["n0"] - emu.col0) // 128
            bar("partial", t(d), t(last_use), "", ["#e8605c", "#8e44ad"][side], alpha=0.7)
    for (cls, kb, epoch), spans in operands.items():
        bar(cls, min(s for s, _, _ in spans), max(e for _, _, e in spans), f"{len(spans)} x b128 k{kb}", TILE_COLOR[cls[0]],
            ready=max(r for _, r, _ in spans if r is not None), row=0 if cls == "A operand" or not lanes[cls] else None)
    # LDS packets: versions with the accesses of both groups.
    versions = defaultdict(list)
    for g in (0, 1):
        for ev in run.lds_events(g, last + 1):
            if ev.op.info.get("kb"):
                versions[(f"{ev.op.info['region']}{ev.op.info['packet']}", ev.op.info["kb"][0])].append((g, ev))
    slots = [f"{a}{p}" for a in "AB" for p in range(4)]
    marks = []
    for (slot, kb), items in versions.items():
        starts = [t(ev) for _, ev in items if run.kind(ev) == "ds_write" and t(ev) is not None] or [t(ev) for _, ev in items if t(ev) is not None]
        ends = [t_ready(g, ev.op) or t(ev) for g, ev in items if run.kind(ev) == "ds_read" and t(ev) is not None] or starts
        if starts:
            bar("lds", min(starts), max(ends), f"k{kb}", KB_COLORS[kb % 5], row=slots.index(slot), alpha=0.28)
        for g, ev in items:
            if t(ev) is not None:
                marks.append((t(ev), slots.index(slot), "v" if run.kind(ev) == "ds_write" else "o", g))
    deps = [(kind, slot, t_ready(pg, src.op) or t(src), t(dst), dst.epoch) for kind, slot, pg, src, cg, dst in run.binding()
            if cg == 0 and dst.epoch <= last and t(dst) is not None]

    gap = (release[last + 1] - release[0]) / blocks / 120
    spec = [("sync", "s_barrier:\ncross-group\nLDS deps", 2), ("load", "buffer_load\nA / B (next K)", None), ("scale", "scale loads", None),
            ("wait", "s_waitcnt", 2), ("write", "ds_write A / B", 2), ("read", "ds_read A / B", 2), ("mfma", "MFMA", 1), ("valu", "VALU", 1),
            ("A stage", "VGPR: A staging", None), ("B stage", "VGPR: B staging", None), ("B operand", "VGPR: B operands", None),
            ("A operand", "VGPR: A operands", 1), ("scale_a", "VGPR: scale_a", None), ("scale_b", "SGPR/VGPR:\nscale_b", None),
            ("partial", "VGPR: MFMA\npartial chains", None), ("lds", "LDS packets", 8)]
    pitch = {"lds": 1.8}
    rows = {}
    for lane, _, n in spec:
        bars = lanes.get(lane, [])
        if n is None:
            fixed = [b for b in bars if b["row"] is not None]
            n = max([b["row"] + 1 for b in fixed] + [pack([b for b in bars if b["row"] is None], gap * 6) if any(b["row"] is None for b in bars) else 1])
        rows[lane] = n
    base, y = {}, 0.0
    for lane, _, _ in reversed(spec):
        base[lane] = y
        y += rows[lane] * pitch.get(lane, 1.0) + 0.9
    header = y

    def ly(lane, row):
        return base[lane] + (rows[lane] - 0.5 - row) * pitch.get(lane, 1.0)

    ticks = []
    for lane, name, _ in spec:
        if lane == "lds":
            ticks += [(ly(lane, r), f"LDS {slot}") for r, slot in enumerate(slots)]
        elif lane in ("A stage", "B stage"):
            ticks += [(ly(lane, r), f"{name}  v{key[1]}..v{key[2]}") for key, r in stage_rows.items() if key[0] == lane]
        else:
            ticks.append((base[lane] + rows[lane] * pitch.get(lane, 1.0) / 2 - 0.45, name))

    def draw(ax, lo, hi, e_lo, e_hi):
        for e in range(e_lo, e_hi + 1):
            phase = run.labels[0].get(e, ("idle", None, None))
            ax.add_patch(Rectangle((release[e], -0.5), release[e + 1] - release[e], header + 1.4,
                                   color={"C": "#fde9e9", "M": "#e3f2e3", "prolog": "#e8eef5"}.get(phase[0], "#f2f2f2"), zorder=0))
            mid = (release[e] + release[e + 1]) / 2
            ax.text(mid, header + 0.45, audit.label_text(phase), ha="center", va="center", fontsize=10, fontweight="bold" if phase[0] == "M" else "normal")
            ax.text(mid, header - 0.25, "G1: " + audit.label_text(run.labels[1].get(e, ("idle", None, None))), ha="center", va="center", fontsize=8, color="#666666")
        for b in range(e_lo + 1, e_hi + 2):
            ax.axvline(release[b], color="black", lw=1.0, ls=(0, (4, 3)), zorder=4)
        for lane, _, _ in spec:
            ax.axhline(base[lane] - 0.45, color="#b0b0b0", lw=0.7, zorder=0.6)
        for r in range(1, len(slots)):
            ax.axhline(ly("lds", r) + pitch["lds"] / 2, color="#e0e0e0", lw=0.5, zorder=0.6)
        for lane, bars in lanes.items():
            for b in bars:
                if b["e"] < lo or b["s"] > hi:
                    continue
                yy = ly(lane, b["row"])
                h = {"mfma": 0.62, "valu": 0.62, "lds": 0.9}.get(lane, 0.5)
                if b["ready"] is not None and b["ready"] > b["s"]:
                    ax.add_patch(Rectangle((b["s"], yy - h / 2), b["ready"] - b["s"], h, facecolor="white", edgecolor=b["color"], hatch="////", lw=0.8, zorder=2))
                    ax.add_patch(Rectangle((b["ready"], yy - h / 2), max(b["e"] - b["ready"], 2), h, facecolor=b["color"], alpha=b["alpha"], lw=0, zorder=2))
                else:
                    ax.add_patch(Rectangle((b["s"], yy - h / 2), b["e"] - b["s"], h, facecolor=b["color"], alpha=b["alpha"], lw=0, zorder=2))
                if b["label"]:
                    inside = lane in ("mfma", "valu")
                    dy = {"lds": 0.0}.get(lane, 0 if inside else 0.42)
                    ax.text(max(b["s"], lo) + (0 if inside else gap * 0.3) + ((b["e"] - b["s"]) / 2 if inside else 0), yy + dy,
                            b["label"], fontsize=7.5, ha="center" if inside else "left", va="center", clip_on=True, zorder=5,
                            color="white" if lane == "mfma" else "black")
        for x, row, name, count in waits:
            if lo <= x <= hi:
                yy = ly("wait", row)
                ax.plot([x, x], [yy - 0.35, yy + 0.35], color="#d62728" if row == 0 else "#1f4e9a", lw=1.4, alpha=1 if count else 0.3, zorder=3)
                ax.text(x + gap * 0.2, yy + 0.25, name, fontsize=7, color="#d62728" if row == 0 else "#1f4e9a", alpha=1 if count else 0.4, clip_on=True)
        for x, row, marker, g in marks:
            if lo <= x <= hi:
                ax.plot([x], [ly("lds", row)], marker=marker, ms=7, mfc=GROUP_COLORS[0] if g == 0 else "white", mec=GROUP_COLORS[g], mew=1.4, zorder=6)
        sync = defaultdict(set)
        for kind, slot, x0, x1, epoch in deps:
            color = "#c0392b" if kind == "WAR" else "#1e8449"
            if lo <= x1 <= hi:
                yy = ly("lds", slots.index(slot))
                ax.annotate("", xy=(x1, yy), xytext=(max(x0, lo), yy), arrowprops=dict(arrowstyle="->", color=color, lw=1.3), zorder=7)
                sync[(epoch, kind)].add(slot)
        for (epoch, kind), group in sync.items():
            ax.text(release[epoch] + gap * 0.3, ly("sync", 0 if kind == "WAR" else 1), f"{kind} " + " ".join(sorted(group)), fontsize=8.5,
                    color="#c0392b" if kind == "WAR" else "#1e8449", fontweight="bold", va="center", clip_on=True)

    panels = [("prolog + kiter 0 (K block 0)", 0, 8)] + [(f"kiter {b} (K block {b})", 8 * b + 1, 8 * b + 8) for b in range(1, blocks)]
    width = max(release[e1 + 1] - release[e0] for _, e0, e1 in panels)
    fig_w, left, unit = 32.0, 2.5, 0.2
    panel_h = (header + 1.6) * unit
    fig_h = 1.2 + len(panels) * (panel_h + 0.7) + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, (title, e0, e1) in enumerate(panels):
        lo, hi = release[e0], release[e1 + 1]
        ax = fig.add_axes([left / fig_w, (1.6 + (len(panels) - 1 - i) * (panel_h + 0.7)) / fig_h, (hi - lo) / width * (fig_w - left - 0.3) / fig_w, panel_h / fig_h])
        draw(ax, lo, hi, e0, e1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, header + 0.9)
        ax.set_yticks([y for y, _ in ticks], [name for _, name in ticks], fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    variant = run.res.cfg["variant"]
    fig.text(left / fig_w, 1 - 0.45 / fig_h, f"{variant}: swimlane of wave group 0 (waves 0-3; wave 0 on SIMD 0 shown, its SIMD partner wave 4 of group 1 shares the pipes), "
             f"from the emulated final ISA ({run.config})", fontsize=14, va="center")
    fig.legend(handles=[
        Patch(color=TILE_COLOR["A"], label="A"), Patch(color=TILE_COLOR["B"], label="B"), Patch(color=scale_color["SA"], label="scale_a"),
        Patch(color=scale_color["SB"], label="scale_b"), Patch(facecolor="white", edgecolor="black", hatch="////", label="in flight (issue -> retiring wait)"),
        Patch(color="#888888", label="resident (ready -> last use)"), Patch(color="#e8605c", label="MFMA run / side-0 partial"),
        Patch(color="#8e44ad", label="side-1 partial"), Line2D([], [], color="#d62728", lw=1.4, label="vmcnt wait (faded: retires nothing)"),
        Line2D([], [], color="#1f4e9a", lw=1.4, label="lgkmcnt wait"),
        Line2D([], [], ls="none", marker="v", ms=8, mfc=GROUP_COLORS[0], mec=GROUP_COLORS[0], label="LDS write G0 / hollow: G1"),
        Line2D([], [], ls="none", marker="o", ms=8, mfc=GROUP_COLORS[0], mec=GROUP_COLORS[0], label="LDS read G0 / hollow: G1"),
        Line2D([], [], color="#c0392b", lw=1.4, label="WAR, 0 spare barriers (G1 read -> G0 write)"),
        Line2D([], [], color="#1e8449", lw=1.4, label="RAW, 0 spare barriers (G1 write -> G0 read)"),
        Line2D([], [], color="black", ls=(0, (4, 3)), label="s_barrier (all 8 waves)"),
    ], loc="lower left", bbox_to_anchor=(left / fig_w, 0.55 / fig_h), ncol=8, fontsize=9.5, frameon=False)
    fig.text(left / fig_w, 0.15 / fig_h, "x: nominal issue cycles (16 per MFMA on the XDL pipe and 4 per VALU, both shared with wave 4; others 4, scalar 1; "
             "memory latency not modeled). Background: green = memory stage (M) of group 0, red = MFMA stage (C); 'G1:' shows what group 1 runs in the same "
             "interval. LDS rows: one bar per data K block from the first write to the last read retired, markers for both groups.", fontsize=10)
    save(fig, f"swimlane_{variant}.png")


INLINE_README = "README_inline.md"


def write_inline_readme():
    """README.md with images/*.png embedded as data URIs, for Markdown previews that cannot load relative image paths
    (e.g. a client that reads the document through the VS Code Agent Host). 256-color palettes keep it ~1.7 MB."""

    def embed(match):
        buf = io.BytesIO()
        image = Image.open(os.path.join(HERE, match.group(2))).convert("RGB")
        image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(buf, "PNG", optimize=True)
        return f"![{match.group(1)}](data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()})"

    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        title, _, body = f.read().partition("\n")
    body, count = re.subn(r"!\[([^\]]*)\]\((?:\./)?(images/[^)\s]+\.png)\)", embed, body)
    note = (f"> 本文件由 `python make_figures.py --inline` 从 [README.md](./README.md) 生成，{count} 张图以 data URI 内嵌，"
            "供无法读取相对路径图片的 Markdown 预览使用。请修改 README.md，不要修改本文件。\n")
    path = os.path.join(HERE, INLINE_README)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n{note}{body}")
    print("wrote", path, f"({count} images embedded, {os.path.getsize(path) // 1024} KiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isa", help="ISA dump directory of the audit tool; without it only the layout figures are drawn")
    parser.add_argument("--k", type=int, default=16384, help="K of the configs drawn (N=4096, preshuffled B)")
    parser.add_argument("--inline", action="store_true", help=f"only write {INLINE_README} from README.md and the current images")
    args = parser.parse_args()
    if args.inline:
        write_inline_readme()
        return
    os.makedirs(IMAGES, exist_ok=True)
    fig_c_tile()
    fig_lds("A")
    fig_lds("B")
    fig_scales()
    if args.isa:
        for variant in ("pf1", "pf2"):
            run = Run(f"{variant}_n4096_k{args.k}_pre1", args.isa)
            fig_prefetch(run)
            fig_swimlane(run)


if __name__ == "__main__":
    main()
