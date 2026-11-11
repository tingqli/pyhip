"""Figures of docs/design_moe_a8w8_blockscale_gfx942.md (written next to this file).

usage: python docs/design_moe_a8w8_blockscale_gfx942/make_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
WAVE = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5"]
A_C, B_C, S_C, O_C, G_C = "#a6c8ec", "#f9c58d", "#b7deb0", "#f4a6a7", "#e0e0e0"
M_C, C_C = "#d9ecd0", "#f6d0d3"
LGROUP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6"]
LGROUP2 = ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913"]
LGREEN = ["#c7e9c0", "#a1d99b", "#74c476", "#41ab5d"]
LPURPLE = ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba"]


def canvas(w, h, xlim, ylim, invert=False):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if invert:
        ax.invert_yaxis()
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc="white", ec="black", lw=0.8, fs=8, tc="black", hatch=None, weight="normal",
        ha="center", alpha=1.0, z=1, ls="-", tbg=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, hatch=hatch, alpha=alpha, zorder=z, ls=ls))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        kw = {"bbox": dict(facecolor=tbg, edgecolor="none", pad=1.2)} if tbg else {}
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc, weight=weight, zorder=z + 1, **kw)


def arrow(ax, p0, p1, color="black", lw=1.0, ls="-", rad=0.0, ms=10, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, fs=8, ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, **kw)


def lines(ax, x, y, rows, step, fs=8):
    for i, row in enumerate(rows):
        line, bold = row if isinstance(row, tuple) else (row, False)
        text(ax, x, y + step * i, line, fs=fs, weight="bold" if bold else "normal")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_dataflow():
    fig, ax = canvas(15.5, 7.0, (0, 23.2), (0, 10))
    box(ax, 4.1, 8.3, 4.7, 1.2, "W1 [E, 2I, H] FP8, shuffle_weight(16,16)\nw1_scale [E, 2I/128, H/128] FP32", fc=B_C)
    box(ax, 14.3, 8.3, 4.7, 1.2, "W2 [E, H, I] FP8, shuffle_weight(16,16)\nw2_scale [E, H/128, I/128] FP32", fc=B_C)
    box(ax, 0.0, 5.9, 3.5, 1.3, "a1 [T, H] FP8\na1_scale_t [H/128, T] FP32", fc=A_C)
    box(ax, 4.1, 5.6, 4.7, 1.9, "GateUp kernel\nSiLU(a1 W1_gate^T) * (a1 W1_up^T)\n-> FP8 per (row, 128 columns)", lw=1.8, fs=8.5)
    box(ax, 9.5, 5.9, 4.1, 1.3, "A2 [blocks*256, I] FP8 (sorted rows)\na2_scale_t [I/128, blocks*256] FP32", fc=A_C)
    box(ax, 14.3, 5.6, 4.7, 1.9, "Down kernel\nout[token] += w * (A2 row) W2^T\nBF16 atomics (single_store: store)", lw=1.8, fs=8.5)
    box(ax, 19.7, 5.9, 3.3, 1.3, "out [T, H] BF16", fc=O_C)
    box(ax, 0.0, 3.3, 3.5, 1.3, "topk_ids, topk_weights [T, topk]\nexpert_mask, num_local_tokens", fc=G_C)
    box(ax, 4.1, 3.3, 4.7, 1.3, "aiter moe_sorting(block_m = 256)", fc=G_C)
    box(ax, 9.5, 2.9, 4.1, 2.1, "sort table\nsorted_ids [cap] = slot<<24 | token\nsorted_weights [cap] FP32\nexpert_ids [blocks], num_valid_ids", fc=S_C)
    for p0, p1 in [((3.5, 6.55), (4.1, 6.55)), ((8.8, 6.55), (9.5, 6.55)), ((13.6, 6.55), (14.3, 6.55)),
                   ((19.0, 6.55), (19.7, 6.55)), ((6.45, 8.3), (6.45, 7.5)), ((16.65, 8.3), (16.65, 7.5)),
                   ((3.5, 3.95), (4.1, 3.95)), ((8.8, 3.95), (9.5, 3.95))]:
        arrow(ax, p0, p1)
    arrow(ax, (10.2, 5.0), (8.0, 5.6), rad=0.2)
    arrow(ax, (12.9, 5.0), (15.0, 5.6), rad=-0.2)
    box(ax, 0.0, 1.25, 23.0, 0.95, "aiter path  fused_moe_a8w8_blockscale_gfx942:   sort (empty moe_buf)  ->  moe_route_counts  ->  "
        "moe_zero_rows  ->  GateUp BM256 (bm128)  ->  Down BM256 (single_store, bm128)", fs=8.5, ha="left", ec="#555555")
    box(ax, 0.0, 0.1, 23.0, 0.95, "sort / mixed path  fused_moe_a8w8_blockscale_gfx942_sort, _mixed:   sort (moe_buf = out, zeroed)  ->  "
        "merged GateUp  ->  merged Down (atomics);   more than 2048 sort blocks: BM256 + bm128", fs=8.5, ha="left", ec="#555555")
    save(fig, "dataflow.png")


def fig_sort_table():
    fig, ax = canvas(14.5, 6.2, (0, 26), (-58, 330), invert=True)
    blocks = [(0, 3, 256), (1, 3, 70), (2, 7, 200), (3, 9, 128), (4, 12, 9)]
    width, step = 1.9, 2.5
    for i, (b, e, valid) in enumerate(blocks):
        x = 0.6 + i * step
        box(ax, x, 0, width, valid, fc=A_C)
        if valid < 256:
            box(ax, x, valid, width, 256 - valid, fc=G_C, hatch="//")
        label_y = valid / 2 if 40 <= valid and abs(valid / 2 - 128) > 20 else (64 if valid >= 40 else valid + 24)
        text(ax, x + width / 2, label_y, f"{valid}\nroutes", ha="center", fs=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0) if valid < 40 else None)
        if 256 - valid > 60:
            text(ax, x + width / 2, (valid + 256) / 2 + (18 if valid < 40 else 0), "padding", ha="center", fs=7,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        text(ax, x + width / 2, -12, f"block {b}\nexpert {e}", ha="center", fs=8)
        big = valid > 128
        box(ax, x - 0.1, 268, width + 0.2, 34, "route at row 128\n-> BM256 task" if big else "<= 128 routes\n-> bm128 / pair",
            fc="#c9e7c3" if big else "#fde3c8", fs=7)
    x = 0.6 + 5 * step
    box(ax, x, 0, width, 256, "spare\ncapacity\n(never\nread)", fc="#bdbdbd", hatch="xx", fs=8, tbg="white")
    text(ax, x + width / 2, -12, "block 5\n(sort spare)", ha="center", fs=8)
    ax.plot([0.3, x + width + 0.2], [128, 128], color="red", ls="--", lw=1.4, zorder=5)
    text(ax, x + width + 0.3, 128, "row 128 probe:\nsorted_ids[256 b + 128]", fs=8, color="red")
    ax.annotate("", xy=(0.6, -38), xytext=(0.6 + 4 * step + width, -38), arrowprops=dict(arrowstyle="<->", lw=1))
    text(ax, 0.6 + 2 * step + width / 2, -49, "num_valid_ids[0] = 5 x 256 rows  ->  valid_blocks = min(ceil(num_valid / 256), blocks) = 5",
         ha="center", fs=8)
    for r in (0, 64, 128, 192, 256):
        text(ax, 0.45, r, str(r), ha="right", fs=7)
    x0 = 19.0
    text(ax, x0, 10, "sorted ID (int32)", fs=9, weight="bold")
    box(ax, x0, 22, 1.8, 22, "31..24\nslot", fc="#fde3c8", fs=8)
    box(ax, x0 + 1.8, 22, 3.6, 22, "23..0\ntoken", fc=A_C, fs=8)
    lines(ax, x0, 60, [
        "route: token < T and slot < topk",
        "padding: anything else",
        "  (aiter writes (topk << 24) | T)",
        "",
        "per expert: its routes sorted by token,",
        "  padded to whole 256-row blocks:",
        "  routes are a prefix of each block",
        "expert_ids[b]: expert of block b",
        "sorted_weights[p]: weight of row p",
        "",
        "A2 row of sorted row p is p itself:",
        "  a2[256 b + r], a2_scale_t[:, 256 b + r]",
    ], 15)
    save(fig, "sort_table.png")


def fig_weight_layout():
    fig, ax = canvas(13.5, 7.0, (-2.4, 21.5), (-3.0, 13.6), invert=True)
    cw, rh = 1.25, 0.62
    text(ax, -2.4, -2.5, "W after shuffle_weight(16,16): one 16-row group x one K block (128 B of K per row) = 2 KiB contiguous",
         fs=9.5, weight="bold")
    for c in range(8):
        text(ax, (c + 0.5) * cw, -0.35, f"K {16 * c}-{16 * c + 15}", ha="center", fs=6.5)
    text(ax, 2 * cw, -1.3, "load j = 0: voffset 16 l", ha="center", fs=8, color="#08519c", weight="bold")
    text(ax, 6 * cw, -1.3, "load j = 1: voffset 16 l + 1024", ha="center", fs=8, color="#a63603", weight="bold")
    for r in range(16):
        text(ax, -0.1, (r + 0.5) * rh, f"row {r}", ha="right", fs=6.5)
        for c in range(8):
            j, g = divmod(c, 4)
            box(ax, c * cw, r * rh, cw, rh, f"l{16 * g + r}", fc=(LGROUP if j == 0 else LGROUP2)[g], fs=6, lw=0.4)
    text(ax, 0, 16 * rh + 0.45, "cell = lane that loads these 16 B (lane groups 0-3 shaded)", fs=7.5)
    x0 = 11.0
    lines(ax, x0, 0.1, [
        ("memory order inside the 2 KiB (byte offset):", True),
        "  offset = 256 c + 16 row + byte,  c = 16 B K chunk",
        "  (chunk outer, row inner)",
        "",
        ("one buffer_load_dwordx4 per lane:", True),
        "  64 lanes x 16 B = 1 KiB = K chunks 4j .. 4j+3 of",
        "  all 16 rows, fully coalesced",
        "  lane l, load j -> row l % 16,",
        "  K bytes 64 j + 16 (l // 16) ... +15",
        "",
        ("the 16 B of a lane = two MFMA operands:", True),
        "  h = 0: K bytes +0..+7,  h = 1: +8..+15",
        "  MFMA (j, h) of a K block: lane l holds",
        "  K 64 j + 16 (l // 16) + 8 h ... +7 of row l % 16",
        "",
        ("group g of rows n0 + 16 g .. +15 starts at", True),
        "  byte (n0 + 16 g) * K; K block k adds 2048 k",
        "  (K = reduction length: H for W1, I for W2)",
    ], 0.62, fs=8)
    y = 16 * rh + 1.3
    text(ax, 0, y, "memory strip (2 KiB): 8 segments of 256 B, each = 16 rows x 16 B of one K chunk", fs=7.5)
    for c in range(8):
        j, g = divmod(c, 4)
        box(ax, c * cw, y + 0.4, cw, 0.6, f"K{16 * c}+, r0-15", fc=(LGROUP if j == 0 else LGROUP2)[g], fs=5.8, lw=0.4)
        text(ax, c * cw, y + 1.3, f"{256 * c}", ha="center", fs=6.5)
    text(ax, 8 * cw, y + 1.3, "2048", ha="center", fs=6.5)
    save(fig, "weight_layout.png")


def fig_mfma():
    fig, ax = canvas(15.5, 7.4, (-1.5, 31.5), (-3.0, 13.6), invert=True)
    x1, y1, s = 0.0, -0.4, 0.62
    text(ax, -1.5, -2.6, "(a) MFMA result D: lane l holds token row", fs=9.5, weight="bold")
    text(ax, -1.5, -1.95, "    m = l % 16, columns n = 4 (l // 16) + e", fs=9.5, weight="bold")
    for n in range(16):
        text(ax, x1 + (n + 0.5) * s, y1 - 0.25, str(n), ha="center", fs=6)
    for m in range(16):
        text(ax, x1 - 0.1, y1 + (m + 0.5) * s, f"m{m}", ha="right", fs=6)
        for n in range(16):
            box(ax, x1 + n * s, y1 + m * s, s, s, str(m + 16 * (n // 4)), fc=LGROUP[n // 4], fs=5, lw=0.3)
    text(ax, x1, y1 + 16 * s + 0.45, "cell = lane holding D[m][n] (element e = n % 4)", fs=7.5)
    text(ax, x1, y1 + 16 * s + 1.05, "n: one 16-column N slice of the output (weight rows),", fs=7.5)
    text(ax, x1, y1 + 16 * s + 1.6, "m: one 16-row M slice (sorted token rows)", fs=7.5)
    x0 = 12.0
    text(ax, x0, -2.6, "(b) v_mfma_f32_16x16x32_fp8_fp8 operands: the weight is srcA, the activation srcB", fs=9.5,
         weight="bold")
    rows = [
        ("srcA = W: 16 out columns x 32 K", "lane l: 8 B of weight row l % 16"),
        ("srcB = A: 32 K x 16 token rows", "lane l: 8 B of token row l % 16"),
        ("MFMA (j, h), j, h in {0, 1}", "lane l: K 64 j + 16 (l // 16) + 8 h ... +7"),
        ("A and B use the same K map", "4 MFMAs (j, h) = all 128 K of a K block"),
        ("A from LDS: ds_read_b128", "row l % 16 of the slice, 16 B chunk 4 j + l // 16"),
    ]
    for i, (a, b) in enumerate(rows):
        box(ax, x0, -1.9 + 0.95 * i, 7.0, 0.85, a, fc=G_C, fs=7.6)
        box(ax, x0 + 7.0, -1.9 + 0.95 * i, 11.5, 0.85, b, fc="white", fs=7.6)
    y2 = 4.6
    text(ax, x0, y2 - 0.8, "(c) block-scale dequant of one 16x16 output block in K block k", fs=9.5, weight="bold")
    for i in range(4):
        box(ax, x0 + i * 2.4, y2, 2.0, 0.9, f"MFMA j{i // 2} h{i % 2}", fc=C_C, fs=7.6)
        arrow(ax, (x0 + i * 2.4 + 2.0, y2 + 0.45), (x0 + (i + 1) * 2.4, y2 + 0.45), ms=7)
    box(ax, x0 + 9.6, y2, 2.8, 0.9, "partial P (4 x f32)\nK 128 sum", fc="#fff2cc", fs=7.4)
    arrow(ax, (x0 + 12.4, y2 + 0.45), (x0 + 12.9, y2 + 0.45), ms=7)
    box(ax, x0 + 12.9, y2, 6.4, 0.9, "acc[e] = fma(P[e], sa * sb, acc[e])", fc=S_C, fs=7.6)
    lines(ax, x0, y2 + 1.5, [
        "first MFMA of the chain takes a zero accumulator (fresh K 128 partial sum)",
        "sa = A scale of the lane's token row in K block k (per lane);",
        "sb = W scale of the 128-column group in K block k (wave uniform, v_readlane)",
        "the 4 FMAs of P run interleaved with the MFMAs of the next chain: an MFMA result",
        "is read only after more MFMAs have issued (XDL latency) -> P is double-buffered",
        "accumulators stay FP32 across all K blocks; nothing is rounded before the epilogue",
    ], 0.62, fs=7.8)
    save(fig, "mfma.png")


def fig_gateup_tile():
    fig, ax = canvas(13, 9.6, (-34, 252), (-30, 262), invert=True)
    for rb in range(16):
        wm = rb % 2
        j = (rb // 4) * 2 + (rb % 4) // 2
        for cb in range(4):
            w = 4 * wm + cb
            ax.plot([cb * 32 + 16] * 2, [rb * 16, rb * 16 + 16], color="#777777", lw=0.4, ls=":", zorder=2)
            box(ax, cb * 32, rb * 16, 32, 16, f"w{w}  M{j}", fc=WAVE[w], fs=7.5, lw=0.5, tbg=WAVE[w])
    for p in range(5):
        ax.plot([-2, 130], [64 * p] * 2, color="black", lw=2.0)
    for p in range(4):
        text(ax, -3, 64 * p + 32, f"A packet {p}\nrows {64 * p}-{64 * p + 63}\nread at stage {p}", ha="right", fs=7.5)
    for cb in range(4):
        text(ax, cb * 32 + 16, -14, f"wave_n {cb}", ha="center", fs=8, weight="bold")
        text(ax, cb * 32 + 8, -4, "s0", ha="center", fs=6.5)
        text(ax, cb * 32 + 24, -4, "s1", ha="center", fs=6.5)
    text(ax, 64, -25, "output columns c = 0..127 of N tile n:  out = SiLU(gate c) * up c", ha="center", fs=8.5, weight="bold")
    lines(ax, 140, 6, [
        ("task = (sorted block b, N tile n), n = 0 .. I/128 - 1", True),
        "  rows: the 256 sorted rows of block b",
        "  W1 rows: gate n*128 + c, up I + n*128 + c",
        "",
        ("8 waves, 2 (M) x 4 (N):", True),
        "  wave = tid // 64",
        "  wave_m = tid // 256  (G0 = waves 0-3, G1 = 4-7)",
        "  wave_n = wave % 4",
        "",
        ("M slice j = 0..7 of a wave (16 rows):", True),
        "  rows 64 (j//2) + 32 (j%2) + 16 wave_m + [0,16)",
        "  A packet p holds M slices 2p, 2p+1 of both wave_m",
        "",
        ("N slices of a wave (16 columns each):", True),
        "  s0, s1: gate cols 32 wave_n + 16 s + [0,16)",
        "  s2, s3: up cols, same c as s0, s1",
        "  out col c = SiLU(acc[s]) * acc[s + 2]",
        "",
        ("accum[j][s]: 8 x 4 MFMA blocks x 4 f32", True),
        "  = 128 VGPR per lane",
        "cell 'w5 M2' = wave 5, its M slice 2",
        "dotted line: N slice boundary s0 | s1",
    ], 11.3, fs=8)
    save(fig, "gateup_tile.png")


def fig_gateup_b_lds():
    fig, ax = canvas(15, 9.4, (0, 30), (-1.5, 19.6), invert=True)
    text(ax, 0, -0.9, "(a) B (W1 rows of N tile n) -> LDS B packets -> waves (color = wave_n that uses the rows)", fs=9.5,
         weight="bold")
    for g in range(8):
        box(ax, 0, 0.3 + 0.55 * g, 4.6, 0.55, f"gate G{g}: rows n*128 + {16 * g}..+15", fc=WAVE[g // 2], fs=6.8, lw=0.4)
        box(ax, 0, 5.0 + 0.55 * g, 4.6, 0.55, f"up U{g}: rows I + n*128 + {16 * g}..+15", fc=WAVE[g // 2], fs=6.8, lw=0.4)
    px0, pw, ph = 6.6, 4.4, 0.75
    for q in range(4):
        x = px0 + q * (pw + 0.3)
        side = "G" if q < 2 else "U"
        text(ax, x + pw / 2, 0.05, f"LDS B packet {q}", ha="center", fs=8, weight="bold")
        for L in range(4):
            box(ax, x, 0.3 + L * ph, pw, ph, f"rows {16 * L}-{16 * L + 15}: {side}{2 * L + q % 2}\nload: w{2 * L} K0-63 | w{2 * L + 1} K64-127",
                fc=WAVE[L], fs=6.5, lw=1.6 if L == 0 else 0.5)
    for wn in range(4):
        box(ax, px0 + wn * 4.7, 5.0, 4.4, 1.1, f"wave_n {wn} (waves {wn}, {wn + 4})\ns0 G{2 * wn}, s1 G{2 * wn + 1}, s2 U{2 * wn}, s3 U{2 * wn + 1}\nout cols {32 * wn}-{32 * wn + 31}",
            fc=WAVE[wn], fs=6.8)
    lines(ax, px0, 6.6, [
        "slot L of packet q <- group 2L + q%2 of side q//2 (gate q<2, up q>=2), loaded by waves 2L, 2L+1",
        "wave_n reads N slice s from packet s, rows 16 wave_n .. +15 (bold: the four slots wave_n 0 reads)",
        "B load per wave w and K block k: 4 x buffer_load_dwordx4 (one per packet q), buffer base = W1[e] row n*128:",
        "  voffset = (side(q) I + 16 (2 (w//2) + q%2)) H + 1024 (w%2) + 16 lane,  soffset = 2048 k   (bytes)",
        "  LDS write: packet q, row 16 (w//2) + lane%16, 16 B chunk 4 (w%2) + lane//16  (K bytes in natural order)",
        "B read at stage 0 ('held'): 4 N slices x 2 K halves x ds_read_b128 = 32 VGPR, used by all 4 stages",
    ], 0.5, fs=7.5)
    y0 = 11.3
    text(ax, 0, y0 - 0.6, "(b) A packet p (64 sorted rows x 128 B of K block k): 512 threads, one 16 B load each", fs=9.5, weight="bold")
    cw, rh = 0.55, 0.11
    for r in range(64):
        for c in range(8):
            box(ax, c * cw, y0 + r * rh, cw, rh, "", fc=WAVE[r // 8], lw=0.15)
    for w in range(8):
        text(ax, 8 * cw + 0.15, y0 + (8 * w + 4) * rh, f"w{w}: rows {8 * w}-{8 * w + 7}", fs=6.5)
    for c in range(8):
        text(ax, (c + 0.5) * cw, y0 - 0.15, str(c), ha="center", fs=6)
    lines(ax, 6.6, y0 + 0.3, [
        "thread tid: row = tid // 8, 16 B chunk = tid % 8   (G0 writes rows 0-31, G1 rows 32-63)",
        "global byte: token(row) * H + 128 k + 16 chunk; token from sorted_ids[256 b + 64 p + row] (computed once per task)",
        "  invalid route -> voffset = OOB_OFFSET (0x7FFF0000): the buffer load returns 0, the row's MFMAs give 0",
        "LDS dword: packet p base + _lds_dwords(row, chunk) = 32 row + 4 ((row ^ chunk) & 7)",
        "A read of M slice j (stage j//2): row 32 (j%2) + 16 wave_m + lane%16, chunk 4 h + lane//16 (h = K half)",
        "A scales: one buffer_load_dword per M slice of the next stage, row of the lane (a1_scale_t[k][token])",
        "W scales: all K blocks of the task in one VGPR: lane side*KB + k (side 0 gate, 1 up) -> v_readlane per pair",
    ], 0.62, fs=7.8)
    save(fig, "gateup_b_lds.png")


def fig_lds_layout():
    fig, ax = canvas(16, 8.6, (-9, 70), (-2, 29.5), invert=True)
    text(ax, -9, -1.2, "(a) LDS of the BM256 / merged kernels: 64 KiB per workgroup (1 workgroup per CU), offsets in KiB", fs=9.5,
         weight="bold")

    def bar(y, label, parts):
        text(ax, -0.5, y + 0.6, label, ha="right", fs=8)
        for start, size, name, fc in parts:
            box(ax, start, y, size, 1.2, name, fc=fc, fs=6.8, lw=0.6)

    for k in range(0, 65, 8):
        text(ax, k, 0.3, str(k), ha="center", fs=7)
    bar(0.8, "BM256 K loop", [(8 * p, 8, f"A pkt {p}", A_C) for p in range(4)] + [(32 + 8 * q, 8, f"B pkt {q}", B_C) for q in range(4)])
    bar(2.4, "GateUp epilogue", [(0, 32, "FP8 tile 256 rows x 128 B (same swizzle)", O_C), (32, 4, "rowmax", S_C)])
    bar(4.0, "Down epilogue", [(0, 17, "8 waves x 16 rows x 34 dwords (transpose)", O_C)])
    bar(5.8, "scan (merged)", [(0, 1.2, "", S_C)])
    text(ax, 1.6, 6.4, "dwords 0-7: wave totals, dword 8: block id; a barrier frees them before the task", fs=7)
    bar(7.4, "pair GateUp K loop", [(0, 16, "A tile 128 rows x 128 B", A_C), (32, 4, "sa", S_C)])
    bar(9.0, "pair GateUp epilogue", [(0, 16, "FP8 tile, half 0", O_C), (16, 16, "FP8 tile, half 1", O_C), (32, 4, "rowmax", S_C)])
    bar(10.6, "pair Down K loop", [(0, 16, "A tile 128 rows x 128 B", A_C), (16, 0.5, "", S_C)])
    text(ax, 17, 11.2, "sa (128 f32)", fs=7)
    bar(12.2, "pair Down epilogue", [(0, 9, "8 x 16 x 18 dwords", O_C)])
    text(ax, 37, 3.0, "rowmax: [row][wave_n] f32 (4 KiB)", fs=7)
    text(ax, 37, 8.0, "sa: A scale per row (K loop),", fs=7)
    text(ax, 37, 8.6, "rowmax [row][wave] (epilogue)", fs=7)
    text(ax, -9, 14.6, "A pair task never stages B in LDS: every wave streams its own weight rows into VGPRs.", fs=8)
    x0, y0, s = 2, 18.4, 1.25
    text(ax, -9, 16.3, "(b) XOR swizzle of a 128 B row: logical 16 B chunk c of row r lives in physical slot (r ^ c) & 7", fs=9.5,
         weight="bold")
    colors = plt.cm.Set3.colors
    text(ax, x0 + 4 * s, y0 - 0.95, "logical chunk c", ha="center", fs=7.5)
    for c in range(8):
        text(ax, x0 + (c + 0.5) * s, y0 - 0.35, str(c), ha="center", fs=7.5)
    for r in range(8):
        text(ax, x0 - 0.2, y0 + (r + 0.5) * s, f"row%8 = {r}", ha="right", fs=7)
        for c in range(8):
            slot = (r ^ c) & 7
            box(ax, x0 + c * s, y0 + r * s, s, s, str(slot), fc=colors[slot], fs=7.5, lw=0.8 if c else 2.2)
    lines(ax, 15, y0 + 0.4, [
        "_lds_dwords(row, chunk) = 32 row + 4 ((row ^ chunk) & 7)   (a row = 128 B = all 32 banks)",
        "",
        "MFMA operand reads (ds_read_b128): lanes 0-15 read rows 0-15 of the slice at the same",
        "logical chunk (4 h + lane // 16). Any 8 consecutive rows then occupy 8 different 16 B slots",
        "(bold column: c = 0), i.e. all 32 banks once; without the XOR they would share one slot",
        "(4 banks). This is the access pattern the swizzle is designed for.",
        "",
        "Writes (global -> LDS copies) go through the same function, so inside a row the K bytes",
        "keep their order; only the 16 B slots are permuted.",
    ], 0.95, fs=8)
    save(fig, "lds_layout.png")


def lifecycle(ax, x0, y0, title, rows, cols, events, notes=()):
    """rows: labels; cols: labels; events: (row, col, kind, note) with kind in L/W/R."""
    cw, rh = 2.3, 0.75
    text(ax, x0, y0 - 1.3, title, fs=9.5, weight="bold")
    for c, label in enumerate(cols):
        box(ax, x0 + 3.2 + c * cw, y0 - 0.6, cw, 0.6, label, fc="#eeeeee" if c < len(cols) // 2 else "#dddddd", fs=7, lw=0.4)
    style = {"L": "#9ecae1", "W": "#a1d99b", "R": "#fc9272"}
    for r, label in enumerate(rows):
        box(ax, x0, y0 + r * rh, 3.2, rh, label, fc="white", fs=7, lw=0.4)
        for c in range(len(cols)):
            box(ax, x0 + 3.2 + c * cw, y0 + r * rh, cw, rh, "", fc="white", lw=0.3)
    by_row = {}
    for r, c, kind, note in events:
        box(ax, x0 + 3.2 + c * cw + 0.1, y0 + r * rh + 0.08, cw - 0.2, rh - 0.16, f"{kind} {note}", fc=style[kind], fs=6.3, lw=0.4)
        by_row.setdefault(r, []).append(c)
    for r, cs in by_row.items():
        cs = sorted(set(cs))
        for a, b in zip(cs, cs[1:]):
            arrow(ax, (x0 + 3.2 + a * cw + cw - 0.1, y0 + r * rh + rh / 2), (x0 + 3.2 + b * cw + 0.1, y0 + r * rh + rh / 2),
                  ms=6, lw=0.6, color="#555555")
    for i, line in enumerate(notes):
        text(ax, x0, y0 + len(rows) * rh + 0.45 + 0.5 * i, line, fs=7.3)


def fig_kloop_packets():
    fig, ax = canvas(16, 12.6, (-0.5, 23), (-3.4, 27.0), invert=True)
    box(ax, 0, -3.3, 1.2, 0.6, "L", fc="#9ecae1", fs=7)
    text(ax, 1.4, -3.0, "buffer_load from global into VGPR staging (h_stage / s_stage)", fs=7.5)
    box(ax, 9.6, -3.3, 1.2, 0.6, "W", fc="#a1d99b", fs=7)
    text(ax, 11.0, -3.0, "ds_write staging -> LDS packet", fs=7.5)
    box(ax, 16.0, -3.3, 1.2, 0.6, "R", fc="#fc9272", fs=7)
    text(ax, 17.4, -3.0, "ds_read_b128 LDS -> MFMA operands", fs=7.5)
    cols4 = [f"k: st {s}" for s in range(4)] + [f"k+1: st {s}" for s in range(4)]
    ev = []
    for p in range(4):
        ev += [(p, p, "L", f"-> s_stage[{p}]"), (p, p + 1 if p < 3 else 4, "W", ""), (p, 4 + p, "R", f"M{2 * p},M{2 * p + 1}")]
    for q, (lc, slot, wc) in enumerate([(0, 0, 1), (1, 1, 2), (2, 0, 3), (2, 1, 3)]):
        ev += [(4 + q, lc, "L", f"-> h_stage[{slot}]"), (4 + q, wc, "W", ""), (4 + q, 4, "R", "held")]
    lifecycle(ax, 0, 0, "(a) BM256 GateUp: life of the K block k+1 packets during K block k (A streamed, B held)",
              [f"A pkt {p} (rows {64 * p}-{64 * p + 63})" for p in range(4)] + [f"B pkt {q}" for q in range(4)], cols4, ev,
              ["scale_a: stage s < 3 loads the A scales of stage s+1's M slices (block k), stage 3 those of block k+1 stage 0.",
               "Stage s computes 4 pairs (M slice 2s + i%2, gate or up) x 8 MFMA. A packet is written to LDS in the stage after",
               "the one that read its K block k copy, and loaded one stage before that ('pf1' = prefetch distance 1 stage)."])
    cols2 = ["k: st 0", "k: st 1", "k+1: st 0", "k+1: st 1"]
    ev = [(0, 0, "L", "-> s_stage[0]"), (0, 1, "W", ""), (0, 2, "R", "M0,M1"), (1, 1, "L", "-> s_stage[1]"), (1, 2, "W", ""),
          (1, 3, "R", "M2,M3")]
    for q in range(4):
        ev += [(2 + q, 0, "L", f"-> h_stage[{q}]"), (2 + q, 1, "W", ""), (2 + q, 2, "R", "held")]
    lifecycle(ax, 0, 10.2, "(b) GateUp bm128 (block with <= 128 routes): 2 stages per K block, A packets 0-1 only",
              ["A pkt 0 (rows 0-63)", "A pkt 1 (rows 64-127)"] + [f"B pkt {q}" for q in range(4)], cols2, ev,
              ["The four B packets of K block k+1 are loaded together in stage 0 (h_stage has 4 slots) and written in stage 1;",
               "each wave computes M slices 0-3 only; rows 128-255 are neither loaded, multiplied nor written."])
    ev = []
    for q in range(4):
        ev += [(q, q, "L", f"-> s_stage[{q}]"), (q, q + 1 if q < 3 else 4, "W", ""), (q, 4 + q, "R", f"N{2 * q},N{2 * q + 1}")]
    for p, (lc, slot, wc) in enumerate([(0, 0, 1), (1, 1, 2), (2, 0, 3), (2, 1, 3)]):
        ev += [(4 + p, lc, "L", f"-> h_stage[{slot}]"), (4 + p, wc, "W", ""), (4 + p, 4, "R", "held")]
    lifecycle(ax, 0, 18.4, "(c) BM256 Down: the same schedule with the roles swapped (B streamed, A held)",
              [f"B pkt {q} (N slices {2 * q},{2 * q + 1})" for q in range(4)] + [f"A pkt {p} (rows {64 * p}-{64 * p + 63})" for p in range(4)],
              cols4, ev,
              ["scale_a: stage 0 loads the A scales of all held M slices of block k+1 into the other parity half of scale_a[8];",
               "the K loop is unrolled by two K blocks so that this parity is a compile-time constant.",
               "bm128 Down: only A packets 0 and 1 (loaded in stages 0, 1; written in stages 1, 2); 2 of the 4 pairs per stage.",
               "Last K block: k+1 is clamped to KB-1 (re-fetch) so every iteration runs the same code."])
    save(fig, "kloop_packets.png")


def fig_stagger():
    fig, ax = canvas(16, 5.6, (-3.2, 27), (-2.2, 10.2), invert=True)
    text(ax, -3.2, -1.6, "Wave groups G0 (waves 0-3) and G1 (waves 4-7): G1 runs one barrier behind, so on every SIMD the MFMA "
         "pipe alternates between a G0 and a G1 wave", fs=9.5, weight="bold")
    phases = ["M0", "C0", "M1", "C1", "M2", "C2", "M3", "C3", "M0'", "C0'"]
    w = 2.4
    for g, (y, label) in enumerate([(0, "G0"), (2.0, "G1")]):
        text(ax, -0.3, y + 0.6, label, ha="right", fs=10, weight="bold")
        off = 1 if g else 0
        if g:
            box(ax, 0, y, w, 1.2, "extra barrier\n(wait)", fc="#f0f0f0", fs=7)
        for i, ph in enumerate(phases[: len(phases) - off]):
            box(ax, (i + off) * w, y, w, 1.2, ph, fc=M_C if ph.startswith("M") else C_C, fs=9)
    for i in range(len(phases) + 1):
        ax.plot([i * w] * 2, [-0.3, 3.5], color="black", ls="--", lw=0.8)
    text(ax, 0, 4.2, "dashed: s_barrier instances; barrier i of G0 pairs with barrier i of G1 (both execute the same number). "
         "SIMD s runs wave s (G0) and wave s+4 (G1).", fs=8)
    rows = [
        ("Ms (memory phase of stage s, no MFMA):", M_C),
        ("  ds_write of the packets loaded one stage earlier -> ds_read_b128 of this stage's operands", None),
        ("  -> buffer_load of K block k+1 packets and scales -> s_waitcnt lgkmcnt(0) -> s_barrier", None),
        ("Cs (compute phase): s_setprio 3; 4 pairs x 8 MFMA, after each MFMA one FMA retires the previous pair; "
         "s_setprio 0; s_barrier", C_C),
        ("Prologue: K block 0 into LDS, then G1 executes one extra s_barrier.  After the loop G0 executes one extra s_barrier "
         "(both groups re-paired before the epilogue touches LDS).", None),
        ("LDS safety: a group reads a packet in its M phase; the other group overwrites that slot in a later M phase, after a "
         "barrier that the reader passed only after s_waitcnt lgkmcnt(0).", None),
    ]
    for i, (line, fc) in enumerate(rows):
        if fc:
            box(ax, -3.0, 5.0 + 0.85 * i, 0.5, 0.5, "", fc=fc)
        text(ax, -2.3 if fc else -3.0, 5.25 + 0.85 * i, line, fs=8)
    save(fig, "stagger.png")


def fig_down_tile():
    fig, ax = canvas(15, 9.0, (-40, 395), (-34, 262), invert=True)
    for c in range(0, 257, 16):
        ax.plot([c, c], [0, 256], color="#888888", lw=0.3, ls=":", zorder=2)
    for rb in range(16):
        wm, j = rb % 4, rb // 4
        for wn in range(2):
            w = 2 * wm + wn
            box(ax, wn * 128, rb * 16, 128, 16, f"w{w}  M{j}", fc=WAVE[w], fs=7.5, lw=0.5, tbg=WAVE[w], z=1)
    for c in range(0, 257, 32):
        ax.plot([c, c], [-6, 256], color="#333333", lw=0.8, zorder=1.5)
    ax.plot([128, 128], [-12, 256], color="black", lw=2.0, zorder=4)
    for p in range(5):
        ax.plot([-2, 258], [64 * p] * 2, color="black", lw=2.0, zorder=4)
    for p in range(4):
        text(ax, -3, 64 * p + 32, f"A pkt {p}\nrows {64 * p}-{64 * p + 63}\nheld", ha="right", fs=7.5)
    for wn in range(2):
        text(ax, wn * 128 + 64, -26, f"wave_n {wn}: hidden cols n*256 + {128 * wn} .. {128 * wn + 127}", ha="center", fs=8,
             weight="bold")
        for q in range(4):
            text(ax, wn * 128 + 32 * q + 16, -10, f"B pkt {q}", ha="center", fs=6.5)
            text(ax, wn * 128 + 32 * q + 16, -3, f"N{2 * q},N{2 * q + 1}", ha="center", fs=6)
    lines(ax, 268, 4, [
        ("task = (sorted block b, N tile n), n = 0 .. H/256 - 1", True),
        "  256 rows x 256 hidden columns, K = I",
        "",
        ("8 waves, 4 (M) x 2 (N):", True),
        "  wave_m = wave // 2, wave_n = wave % 2",
        "  M slice j (0..3) = A packet j:",
        "    rows 64 j + 16 wave_m + [0,16)",
        "  N slice s (0..7):",
        "    cols 128 wave_n + 16 s + [0,16)",
        "  accum[j][s] = 128 VGPR",
        "",
        ("A held: stage 0 reads all 4 M slices", True),
        "  (4 x 2 x ds_read_b128 = 32 VGPR)",
        ("B streamed: stage s reads N slices 2s, 2s+1", True),
        "  from LDS B packet s",
        "pair i of stage s: M slice i x N 2s, 2s+1",
        "",
        ("B packet q, LDS rows 16L..16L+15 <-", True),
        "  W2 rows n*256 + 16 ((L//2) 8 + 2q + L%2)",
        "  loaded by waves 2L (K 0-63), 2L+1 (K 64-127)",
        "wave_n reads packet s//2,",
        "  rows 32 wave_n + 16 (s%2) .. +15",
        ("W scale group: n*2 + wave_n (one per wave)", True),
    ], 11.2, fs=7.8)
    save(fig, "down_tile.png")


def fig_epilogues():
    fig, ax = canvas(16, 8.2, (-0.5, 33), (-1.5, 17.0), invert=True)
    text(ax, -0.5, -0.9, "(a) GateUp epilogue (BM256): SiLU, FP8 per (row, 128 columns), stored through LDS", fs=9.5, weight="bold")
    steps = [
        ("1  registers", "lane: row l%16 of M slice j,\ncols 4(l//16)+e of s0..s3\nv = SiLU(gate) * up: 8 values", A_C),
        ("2  row |max| in a wave", "max over the 8 values,\nds_bpermute from l^16, l^32\n-> 32 columns of the wave", S_C),
        ("3  LDS row max", "lanes 0-15 write\nrowmax[row][wave_n]; barrier;\namax = max of the 4 entries", S_C),
        ("4  quantize", "x * 240 / amax (0 if amax=0)\ncvt_pk_fp8_f32 x2 -> 1 dword\n= 4 FP8 columns", O_C),
        ("5  LDS FP8 tile", "row, 16 B chunk 2 wave_n + s,\ndword l//16; barrier", O_C),
        ("6  store", "tid: row 64 part + tid//8,\nchunk tid%8: a wave stores\n8 rows x 128 B per dwordx4", O_C),
    ]
    for i, (title, body, fc) in enumerate(steps):
        x = i * 5.5
        box(ax, x, 0, 5.0, 0.7, title, fc=fc, fs=8, weight="bold")
        box(ax, x, 0.7, 5.0, 1.9, body, fc="white", fs=7.3)
        if i < len(steps) - 1:
            arrow(ax, (x + 5.0, 1.3), (x + 5.5, 1.3), ms=8)
    text(ax, 0, 3.1, "A2 byte = (256 b + row) I + 128 n + 16 chunk;  scale amax / 240 -> a2_scale_t[n][256 b + row] "
         "(thread tid < tile rows writes row tid).  Padding rows are written too: FP8 zeros, scale 0.", fs=7.8)
    y0 = 4.2
    text(ax, 0, y0 - 0.2, "FP8 tile row (128 B = 8 chunks): which wave fills which chunk", fs=8)
    for c in range(8):
        wn, sl = divmod(c, 2)
        box(ax, c * 1.4, y0 + 0.2, 1.4, 0.7, f"wn{wn} s{sl}", fc=WAVE[wn], fs=7)
    text(ax, 11.8, y0 + 0.55, "chunk 2 wave_n + s holds output columns 32 wave_n + 16 s .. +15 (lane group g: dword g)", fs=7.5)
    y1 = 6.6
    text(ax, -0.5, y1, "(b) Down epilogue (BM256): weight, BF16, per-wave LDS transpose, 2 rows x 128 B per atomic", fs=9.5,
         weight="bold")
    x0, cw, rh = 0, 0.42, 0.42
    yb = y1 + 1.4
    text(ax, x0, yb - 0.45, "write: lane l -> row l%16, dwords 2(l//16) + 8 nl + {0,1}   (nl = N slice in the 64-column group)", fs=7.5)
    palettes = [LGROUP, LGROUP2, LGREEN, LPURPLE]
    for r in range(16):
        for d in range(32):
            nl, rest = divmod(d, 8)
            box(ax, x0 + d * cw, yb + r * rh, cw, rh, "", fc=palettes[nl][rest // 2], lw=0.15)
    for nl in range(4):
        text(ax, x0 + (8 * nl + 4) * cw, yb + 16 * rh + 0.3, f"N slice {nl}: 16 cols", ha="center", fs=7)
    ax.add_patch(Rectangle((x0, yb), 32 * cw, 2 * rh, fill=False, ec="red", lw=1.8, zorder=5))
    text(ax, x0 + 32 * cw + 0.2, yb + rh, "atomic i = 0: rows 0, 1", fs=7.5, color="red")
    text(ax, x0 + 32 * cw + 0.2, yb + 3 * rh, "(pitch 34 dwords:", fs=7.5)
    text(ax, x0 + 32 * cw + 0.2, yb + 4 * rh, " conflict-free both ways)", fs=7.5)
    lines(ax, 17.6, yb + 0.15, [
        "1  v = accum * routing weight -> BF16 RNE, 2 per dword",
        "2  ds_write_b64 per N slice into the wave's 16 x 32-dword tile",
        "3  read: lane l -> row 2i + l//32, dword l%32 (i = 0..7)",
        "4  atomic i: lanes 0-31 row 2i, lanes 32-63 row 2i+1,",
        "   global_atomic_pk_add_bf16 at out + token(row) * 2H",
        "   + 512 n + 256 wave_n + 128 c + 4 (l%32)   (c = 64-col group)",
        "5  token offset of row 2i + l//32: ds_bpermute from that lane",
        "   (lane r holds row r of the slice); invalid row: -1 -> skip",
        "6  single_store: bit 0 of the offset = token has 1 local route",
        "   -> global_store_dword instead of the atomic",
        "s_waitcnt vmcnt(0) first: LLVM does not count asm atomics",
    ], 0.62, fs=7.8)
    text(ax, 0, yb + 16 * rh + 1.0, "pair task Down: 16 rows x 16 dwords per group pair (pitch 18), atomic i: lanes 16 g .. +15 "
         "-> row 4i + g, 4 rows x 64 B", fs=7.8)
    save(fig, "epilogues.png")


def fig_bm128():
    fig, ax = canvas(16, 7.2, (-8, 430), (-50, 275), invert=True)
    box(ax, -6, -40, 96, 26, "workgroup task (b, n)", fc=G_C, fs=8)
    arrow(ax, (42, -14), (42, -2))
    box(ax, -6, -2, 96, 30, "probe = sorted_ids[256 b + 128]\n(one uniform buffer load)", fc=S_C, fs=8)
    arrow(ax, (20, 28), (20, 52))
    arrow(ax, (64, 28), (64, 52))
    text(ax, 14, 40, "route", ha="right", fs=8)
    text(ax, 70, 40, "padding", fs=8)
    box(ax, -6, 52, 50, 46, "big_body()\nBM256 pipeline\n+ asm '; bm256 task'", fc="#c9e7c3", fs=7.5)
    box(ax, 46, 52, 44, 46, "big_body(128)\n128-row pipeline\n+ asm '; bm128 task'", fc="#fde3c8", fs=7.5)
    lines(ax, -6, 115, ["routes are a prefix of", "the block: no route at", "row 128 -> at most 128",
                        "", "distinct last asm lines:", "SimplifyCFG must not", "sink the two pipelines'", "common tail (atomic base",
                        "phi -> memory fault)"], 13, fs=7.5)
    x0 = 110
    text(ax, x0 + 64, -30, "GateUp tile (bm128)", ha="center", fs=9, weight="bold")
    for rb in range(16):
        wm = rb % 2
        j = (rb // 4) * 2 + (rb % 4) // 2
        for cb in range(4):
            w = 4 * wm + cb
            if rb < 8:
                box(ax, x0 + cb * 32, rb * 16, 32, 16, f"w{w} M{j}", fc=WAVE[w], fs=6.5, lw=0.4)
            else:
                box(ax, x0 + cb * 32, rb * 16, 32, 16, "", fc="#eeeeee", lw=0.3, hatch="..")
    box(ax, x0 + 8, 168, 112, 50, "rows 128-255:\nno A load, no MFMA,\nnot written (Down never\nreads them)", fc="white", fs=7.5)
    text(ax, x0, 268, "2 stages / K block; M slices 0-3 per wave", fs=7.5)
    x1 = 270
    text(ax, x1 + 64, -30, "Down tile (bm128)", ha="center", fs=9, weight="bold")
    for rb in range(16):
        wm, j = rb % 4, rb // 4
        for wn in range(2):
            w = 2 * wm + wn
            if rb < 8:
                box(ax, x1 + wn * 64, rb * 16, 64, 16, f"w{w} M{j}", fc=WAVE[w], fs=6.5, lw=0.4)
            else:
                box(ax, x1 + wn * 64, rb * 16, 64, 16, "", fc="#eeeeee", lw=0.3, hatch="..")
    text(ax, x1 + 132, 4, "(256 columns drawn", fs=7)
    text(ax, x1 + 132, 14, " at half width)", fs=7)
    box(ax, x1 + 8, 168, 112, 50, "rows 128-255:\nA packets 2, 3 not loaded,\npairs of M slices 2, 3\nskipped", fc="white", fs=7.5)
    text(ax, x1, 268, "A pkts 0-1 held; 2 pairs per stage; 4 stages", fs=7.5)
    save(fig, "bm128.png")


def fig_xcd_bm256():
    fig, ax = canvas(14, 5.6, (-2.5, 19), (-2.4, 9.2), invert=True)
    text(ax, -2.5, -1.9, "BM256 grid: valid_blocks = 3, n_tiles = 16 -> 48 tasks, grid = 64 (capacity 4 blocks), "
         "chunk = ceil(48 / 8) = 6", fs=9.5, weight="bold")
    colors = ["#c6dbef", "#c7e9c0", "#dadaeb"]
    w, h = 2.0, 0.95
    for x in range(8):
        text(ax, x * w + w / 2, -0.5, f"XCD {x}", ha="center", fs=8.5, weight="bold")
    for slot in range(8):
        text(ax, -0.2, slot * h + h / 2, f"slot {slot}", ha="right", fs=8)
        for x in range(8):
            pid = 8 * slot + x
            if slot < 6:
                b, n = divmod(6 * x + slot, 16)
                box(ax, x * w, slot * h, w, h, f"pid {pid}\nb{b} n{n}", fc=colors[b], fs=7)
            else:
                box(ax, x * w, slot * h, w, h, f"pid {pid}\nidle", fc="#f4f4f4", fs=7, tc="#777777")
    text(ax, 0, 8.2, "workgroup pid runs on XCD pid % 8 (round robin);  task = (pid & 7) * chunk + (pid >> 3),  "
         "chunk = ceil(valid_blocks * n_tiles / 8);  block = task // n_tiles, n = task % n_tiles", fs=8)
    text(ax, 0, 8.8, "-> each XCD takes a contiguous task range: the N tiles of a block (same A rows) share that XCD's L2", fs=8)
    save(fig, "xcd_bm256.png")


def fig_merged_tasks():
    fig, ax = canvas(16, 10.6, (-3.5, 33), (-6.8, 16.6), invert=True)
    text(ax, -3.5, -6.3, "merged task list: example with valid_blocks = 6, n_tiles = 16 (8 pairs), sort capacity 8 blocks",
         fs=9.5, weight="bold")
    big = [True, False, True, False, False, True]
    for b, flag in enumerate(big):
        box(ax, b * 2.2, -5.3, 2.0, 0.9, f"b{b}: " + ("big" if flag else "small"), fc="#c6dbef" if flag else "#fdd0a2", fs=8)
    text(ax, 13.6, -4.85, "big = route at row 128 (one BM256 task per N tile), small = one pair task per 2 N tiles", fs=8)
    lines(ax, 0, -3.9, [
        "big_tasks = 3 x 16 = 48 -> big_chunk = ceil(48/8) = 6 slots per XCD;  small_tasks = 3 x 8 = 24 -> small_chunk = 3",
        "pid = 8 slot + xcd.  slot < big_chunk: big index = xcd*big_chunk + slot -> rank = index // 16 (-> block), "
        "tile = index % 16",
        "  else: small index = xcd*small_chunk + (slot - big_chunk) -> rank = index // 8 (-> block), pair = index % 8",
        "grid = 8 (ceil(16 * 8 / 8) + 1) = 136; slots >= ceil(16 * valid_blocks / 8) + 1 = 13 exit before the scan",
    ], 0.62, fs=8)
    bigs, smalls = [0, 2, 5], [1, 3, 4]
    w, h = 3.6, 0.95
    for x in range(8):
        text(ax, x * w + w / 2, -0.45, f"XCD {x}", ha="center", fs=8.5, weight="bold")
    big_c = ["#c6dbef", "#9ecae1", "#6baed6"]
    small_c = ["#fdd0a2", "#fdae6b", "#fd8d3c"]
    for slot in range(17):
        text(ax, -0.2, slot * h + h / 2, f"slot {slot}", ha="right", fs=7.5)
        for x in range(8):
            yy = slot * h
            if slot < 6:
                rank, tile = divmod(6 * x + slot, 16)
                box(ax, x * w, yy, w, h, f"BM256 b{bigs[rank]} n{tile}", fc=big_c[rank], fs=7)
            elif slot < 9:
                rank, pair = divmod(3 * x + slot - 6, 8)
                box(ax, x * w, yy, w, h, f"pair b{smalls[rank]} p{pair} (n{2 * pair},{2 * pair + 1})", fc=small_c[rank], fs=6.6)
            elif slot < 13:
                box(ax, x * w, yy, w, h, "scan, no task", fc="#eeeeee", fs=7, tc="#666666")
            else:
                box(ax, x * w, yy, w, h, "exit (no scan)", fc="white", fs=7, tc="#999999", ls=":")
    save(fig, "merged_tasks.png")


def fig_merged_scan():
    fig, ax = canvas(16, 8.0, (-1, 33), (-1.6, 16.0), invert=True)
    text(ax, -1, -1.1, "(a) the scan inside every in-range workgroup (512 threads, up to 2048 sort blocks)", fs=9.5, weight="bold")
    steps = [
        ("1  probe", "thread t: blocks 4t .. 4t+3\nflag = block < valid_blocks and\nsorted_ids[256 blk + 128] is a route\n"
                     "count = number of flags (0..4)"),
        ("2  wave prefix", "for bit in 0..2:\n  m = ballot(count bit set)\n  below += mbcnt(m) << bit\n  total += popcount(m) << bit"),
        ("3  workgroup", "lane 0: LDS[wave] = total; barrier\nbig_before = totals of lower waves\n+ below\n"
                         "big_count = sum of all totals"),
        ("4  my task", "big_chunk, small_chunk, slot, xcd\n-> big?, index, rank, tile / pair\n(formulas of the task-list figure)"),
        ("5  find block", "thread whose block i satisfies\nbig: flag and #bigs before = rank\nsmall: !flag and #smalls before = rank\n"
                          "-> LDS[8] = block; barrier"),
        ("6  run", "block = readfirstlane(LDS[8]);\nbarrier (all scan words read)\nbig: BM256 body (b, tile)\nsmall: pair body (b, pair)"),
    ]
    for i, (title, body) in enumerate(steps):
        x = (i % 3) * 10.8
        y = (i // 3) * 3.9
        box(ax, x, y, 10.0, 0.7, title, fc=S_C, fs=8.5, weight="bold")
        box(ax, x, y + 0.7, 10.0, 2.6, body, fc="white", fs=8)
        if i % 3 < 2:
            arrow(ax, (x + 10.0, y + 2.0), (x + 10.8, y + 2.0), ms=8)
    arrow(ax, (26.6, 3.3), (5.0, 3.9), rad=0.03, ms=8)
    y0 = 8.6
    text(ax, -1, y0 - 0.7, "(b) worked example (blocks of the task-list figure, all in wave 0)", fs=9.5, weight="bold")
    header = ["thread", "blocks", "flags (1 = big)", "count", "below"]
    widths = [3.0, 6.0, 7.0, 3.0, 3.0]
    rows = [["t0", "b0 b1 b2 b3", "1 0 1 0", "2", "0"], ["t1", "b4 b5 b6 b7", "0 1 - -   (b6, b7 >= valid)", "1", "2"],
            ["t2 ..", "b8 ..", "- (not probed)", "0", "3"]]
    xs = [0]
    for wdt in widths[:-1]:
        xs.append(xs[-1] + wdt)
    for c, (label, wdt) in enumerate(zip(header, widths)):
        box(ax, xs[c], y0, wdt, 0.7, label, fc=G_C, fs=8, weight="bold")
    for r, row in enumerate(rows):
        for c, (cell, wdt) in enumerate(zip(row, widths)):
            box(ax, xs[c], y0 + 0.7 * (r + 1), wdt, 0.7, cell, fc="white", fs=8)
    lines(ax, 0, y0 + 3.6, [
        "wave 0 total = 3, other waves 0  ->  big_count = 3, small = 6 - 3 = 3, big_chunk = 6, small_chunk = 3",
        "pid 21: xcd 5, slot 2 < 6 -> big index 5*6 + 2 = 32 -> rank 2, tile 0 -> t1, i = 1 (b5: 2 bigs before) -> BM256 (b5, n0)",
        "pid 59: xcd 3, slot 7 >= 6 -> small index 3*3 + 1 = 10 -> rank 1, pair 2 -> t0, i = 3 (b3: 3 - 2 = 1 small before)"
        " -> pair (b3, n4, n5)",
        "pid 75: xcd 3, slot 9 -> slot - big_chunk = 3 >= small_chunk -> inactive (scanned, no task)",
        "pid 120: slot 15 >= 13 -> skips the scan (in_range is false)",
    ], 0.75, fs=8)
    save(fig, "merged_scan.png")


def fig_pair_task():
    fig, ax = canvas(16, 11.8, (-30, 545), (-48, 338), invert=True)
    S = 5
    text(ax, -30, -40, "(a) GateUp pair task: one block with <= 128 routes (example: used rows 70 -> S = 5 slices) x N tiles "
         "2p and 2p+1", fs=9.5, weight="bold")
    for half in range(2):
        x0 = half * 132
        title = "N tile 2p  (half 0: waves 0-3)" if half == 0 else "N tile 2p+1  (half 1: waves 4-7)"
        text(ax, x0 + 64, -24, title, ha="center", fs=8.5, weight="bold")
        for tw in range(4):
            w = 4 * half + tw
            box(ax, x0 + 32 * tw, 0, 32, S * 16, f"w{w}\ncols\n{32 * tw}-{32 * tw + 31}", fc=WAVE[w], fs=7, lw=0.5)
            text(ax, x0 + 32 * tw + 16, -9, f"G{2 * tw},G{2 * tw + 1}", ha="center", fs=6)
            text(ax, x0 + 32 * tw + 16, -3, f"U{2 * tw},U{2 * tw + 1}", ha="center", fs=6)
        box(ax, x0, S * 16, 128, 128 - S * 16, "rows >= 16 S:\nno MFMA, not written\n(A copies of these rows\nare OOB loads: no traffic)",
            fc="#eeeeee", hatch="..", fs=7, tbg="white")
        for s in range(S + 1):
            ax.plot([x0, x0 + 128], [16 * s] * 2, color="#555555", lw=0.5, ls=":")
    for s in range(S):
        text(ax, -2, 16 * s + 8, f"slice {s}", ha="right", fs=7)
    y1 = 140
    text(ax, -30, y1, "(b) Down pair task: the same rows x N tiles 2p, 2p+1 of 256 hidden columns each", fs=9.5, weight="bold")
    for half in range(2):
        x0 = half * 132
        for tw in range(4):
            w = 4 * half + tw
            box(ax, x0 + 32 * tw, y1 + 10, 32, 22, f"w{w}: cols\n{64 * tw}-{64 * tw + 63}", fc=WAVE[w], fs=6.5, lw=0.5)
        text(ax, x0 + 64, y1 + 40, f"N tile 2p{'+1' if half else ''} (256 cols, drawn at half width)", ha="center", fs=7)
    lines(ax, 0, y1 + 58, [
        "wave tw of half h owns 4 weight groups (16 rows each) of N tile n = 2p + h:",
        "  GateUp: g = 0, 1: gate rows n*128 + 32 tw + 16 g;  g = 2, 3: up rows I + n*128 + 32 tw + 16 (g-2)",
        "  Down:   g = 0..3: W2 rows n*256 + 64 tw + 16 g",
        "every wave computes all S slices of its columns: accum[s][g] = 16 S VGPR (S = 8: 128)",
        "",
        "B: each wave streams its own groups global -> VGPR (no LDS): per K block 4 groups x 2 x buffer_load_dwordx4,",
        "   voffset 16 lane + 1024 j, soffset 2048 k + group base (row * K), nt; ring of 2 K blocks = 64 VGPR",
        "A: the rows go through LDS once for both halves: copies c = 0 .. ceil(S/4)-1 of 64 rows, all 512 threads,",
        "   thread tid: row 64 c + tid // 8, 16 B chunk tid % 8 (same as a BM256 A packet, same swizzle)",
        "A scale: thread tid loads row tid % 128 of K block k into LDS sa[row]",
        "W scales: one VGPR, GateUp lane side*KB + k, Down lane k",
        "",
        "used rows: valid = route(sorted_ids[256 b + lane]), route(... + 64 + lane); ballot -> highest valid row + 1",
        "S = ceil(used / 16): one compiled K loop + epilogue per S = 1..8 (if-chain on used rows);",
        "S >= 7 uses the light register layout (pair K-loop figure)",
    ], 9.4, fs=8)
    x1 = 290
    text(ax, x1, -24, "(c) LDS / registers of a pair task", fs=9, weight="bold")
    items = [
        ("LDS A tile [0, 16 KiB): up to 128 rows x 128 B, swizzled", A_C),
        ("LDS sa: 128 f32 A scales (Down at 16 KiB, GateUp at 32 KiB)", S_C),
        ("VGPR B ring: 2 K blocks x 4 groups x 2 x 4 dwords = 64", B_C),
        ("VGPR A staging: copies x 4 dwords per slot, 2 or 1 slots", A_C),
        ("VGPR accum: S x 4 groups x 4 f32 = 16 S", O_C),
        ("VGPR partial: 2 or 1 sets x 4 groups x 4 f32", "#fff2cc"),
        ("ISA: 250 VGPR for GateUp and Down (light layout at S = 8)", "white"),
    ]
    for i, (label, fc) in enumerate(items):
        box(ax, x1, -12 + 18 * i, 245, 15, label, fc=fc, fs=7.5, ha="left")
    save(fig, "pair_task.png")


def fig_pair_kloop():
    fig, ax = canvas(16, 8.6, (-1, 33), (-1.4, 16.4), invert=True)
    text(ax, -1, -0.9, "(a) pair task K loop (all 8 waves together; unrolled by the B ring of 2 K blocks)", fs=9.5, weight="bold")
    seq = [
        ("issue B(k+1)\n-> ring[(k+1)%2]", B_C, 3.2),
        ("slice 0: ds_read A\n16 MFMA + retires", C_C, 3.4),
        ("slice 1 ...", C_C, 2.2),
        ("slice S-1", C_C, 2.2),
        ("full: issue\nA(k+2) -> slot k%2", A_C, 3.3),
        ("lgkmcnt(0)\nbarrier", "#eeeeee", 2.1),
        ("write A(k+1)\nslot -> LDS tile, sa", "#a1d99b", 3.3),
        ("lgkmcnt(0)\nbarrier", "#eeeeee", 2.1),
        ("light: issue\nA(k+2) -> slot 0", A_C, 3.3),
    ]
    x = 0
    for label, fc, wdt in seq:
        box(ax, x, 0, wdt, 1.5, label, fc=fc, fs=7.2)
        x += wdt
    ax.plot([0, 0], [-0.3, 1.8], color="black", lw=1.5)
    text(ax, 0, 2.1, "compute phase of K block k", fs=8)
    text(ax, 14.3, 2.1, "memory phase (VMEM, LDS, barriers; no MFMA)", fs=8)
    rows = [
        ("B ring buf (k%2)", "B(k): read by the MFMAs of K block k"),
        ("B ring buf ((k+1)%2)", "B(k+1): in flight while K block k computes (no wait at issue)"),
        ("A staging", "full (S <= 6): 2 slots, A(k+1) arrived during K block k-1, A(k+2) issued before the barriers"),
        ("", "light (S >= 7): 1 slot, A(k+2) issued after the slot was written to LDS"),
        ("LDS A tile", "single buffer: A(k) until the first barrier, A(k+1) after the write"),
        ("waits", "s_waitcnt lgkmcnt(0) before each barrier: another wave overwrites the tile right after it"),
        ("prologue", "A(0) -> slot 0, B(0) -> buf 0, (full) A(1) -> slot 1; write A(0); barrier; (light) A(1) -> slot 0"),
    ]
    for i, (a, b) in enumerate(rows):
        text(ax, 0, 2.9 + 0.55 * i, a, fs=8, weight="bold")
        text(ax, 5.0, 2.9 + 0.55 * i, b, fs=8)
    y0 = 7.6
    text(ax, -1, y0 - 0.6, "(b) MFMA order of one slice (16 MFMA = 4 weight groups g x 4 K steps (j, h)) and when partials retire",
         fs=9.5, weight="bold")
    full = [(p // 8, (p // 4) % 2, p % 4) for p in range(16)]
    light = [(j, h, 2 * pr + mb) for j in range(2) for pr in range(2) for h in range(2) for mb in range(2)]
    full_ret = {p: [(1, p // 4, p % 4)] for p in range(16)}
    light_ret = {}
    for g in range(4):
        last = max(p for p, o in enumerate(light) if o[2] == g)
        for e in range(4):
            wp = last + 2 + e
            light_ret.setdefault(wp % 16, []).append((wp // 16, g, e))
    cw = 1.95
    for k, (name, order, ret) in enumerate([("full (S <= 6): two partial sets (slice parity)", full, full_ret),
                                            ("light (S >= 7): one partial set", light, light_ret)]):
        yy = y0 + k * 3.7
        text(ax, 0, yy, name, fs=8.5, weight="bold")
        for p, (j, h, g) in enumerate(order):
            box(ax, p * cw, yy + 0.4, cw, 0.8, f"#{p}\nj{j} h{h} g{g}", fc=WAVE[g], fs=6.6, lw=0.4)
            items = ret.get(p, [])
            label = "\n".join(f"{'prev ' if d else ''}g{rg}.e{e}" for d, rg, e in items)
            box(ax, p * cw, yy + 1.2, cw, 1.3, label, fc="#fff2cc" if items else "white", fs=6, lw=0.4)
        if k == 0:
            text(ax, 0, yy + 2.9, "row 2: partial elements retired right after that MFMA (acc += P * scale); 'prev' = chain of the "
                 "previous slice (for slice 0: the previous K block's last slice)", fs=7.3)
    text(ax, 0, y0 + 3.7 + 2.9, "light: element e of group g retires 2 + e MFMAs after the chain's last step, always before g's next "
         "chain starts -> one partial set is enough; the next slice's A is read half by half (after #7 and at the slice end)",
         fs=7.3)
    save(fig, "pair_kloop.png")


def main():
    for make in (fig_dataflow, fig_sort_table, fig_weight_layout, fig_mfma, fig_gateup_tile, fig_gateup_b_lds, fig_lds_layout,
                 fig_kloop_packets, fig_stagger, fig_down_tile, fig_epilogues, fig_bm128, fig_xcd_bm256, fig_merged_tasks,
                 fig_merged_scan, fig_pair_task, fig_pair_kloop):
        make()


if __name__ == "__main__":
    main()
