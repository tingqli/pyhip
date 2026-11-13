"""Figure for fp8_sweep_overlap_analysis.md section 4.3: per tokens/rank and routing, the MORI baseline, the step1 path
(communication in separate kernels) and the mega kernel, the last two split into the layer time with the xGMI traffic
sent to local scratch and the communication time left exposed.

    python docs/megamoe_ep8_942/images/overlap_sweep.py [FONT_DIR] [--from-dir /tmp/overlap]

--from-dir reads the logs and JSONs of a sweep (bench_*, mega_*, step1_* per routing and tokens/rank, see section 4.3)
into data/overlap_sweep.json.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "overlap_sweep.json"
TOKENS = (256, 512, 1024, 2048, 4096)
MODES = ("p2p", "local-dispatch", "local-combine", "local")


def profile(path):
    """Layer time and the slowest rank's stages per mode from profile_ep8_mega_overlap.py's summary."""
    text = path.read_text()
    out = {}
    for mode in MODES:
        m = re.search(rf"^{mode}\s+(\d+)(?:\s+(\d+))?.*slowest rank: (.*)$", text, re.M)
        out[mode] = dict(layer=float(m.group(1)), stages={k: float(v) for k, v in re.findall(r"(\w+) (\d+)", m.group(3))})
        trace = re.search(rf"^{mode}\s+\d+\s+(\d+) \|\s+(\d+)\s+(\d+) \|\s+([\d.]+)\s+(\d+)\s+(\d+) \| ([\d.]+) \|\s+(\d+)\s+(\d+)",
                          text, re.M)
        if trace:
            out[mode].update(zip(("span", "P_run", "P_end", "G_wait", "G_run", "D_run", "busy", "G_mhz", "D_mhz"),
                                 map(float, trace.groups())))
    return out


def extract(directory):
    directory = Path(directory)
    data = {}
    for routing in ("balanced", "random"):
        for tokens in TOKENS:
            tag = f"{routing}_{tokens}"
            bench = json.loads((directory / f"bench_{tag}.json").read_text())
            slowest = max(bench["baseline"], key=lambda r: r["total_us"])
            data[tag] = dict(
                mori={k: slowest[f"{k}_us"] for k in ("quant", "dispatch", "moe", "combine", "total")},
                mega_bench=max(r["total_us"] for r in bench["mega"]),
                step1=profile(directory / f"step1_{tag}.log"),
                mega=profile(directory / f"mega_{tag}.log"),
            )
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(data, indent=1))
    print("wrote", DATA)


def figure(data):
    blue, orange, red, gray = "#4C9BE8", "#F5A623", "#D0021B", "#9B9B9B"
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.2), dpi=150)
    for row, routing in enumerate(("balanced", "random")):
        for col, tokens in enumerate(TOKENS):
            ax, d = axes[row][col], data[f"{routing}_{tokens}"]
            mori = d["mori"]
            bottom = 0
            for key, color in (("quant", gray), ("dispatch", blue), ("moe", orange), ("combine", blue)):
                ax.bar(0, mori[key], bottom=bottom, width=0.62, color=color, edgecolor="white", lw=0.8)
                bottom += mori[key]
            top = {0: mori["total"]}
            for x, name, color in ((1, "step1", blue), (2, "mega", red)):
                p2p, local = d[name]["p2p"]["layer"], d[name]["local"]["layer"]
                ax.bar(x, local, width=0.62, color=orange, edgecolor="white", lw=0.8)
                ax.bar(x, p2p - local, bottom=local, width=0.62, color=color, edgecolor="white", lw=0.8)
                top[x] = p2p
            scale = max(top.values())
            for x, value in top.items():
                ax.text(x, value + scale * 0.015, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
            exposed = d["mega"]["p2p"]["layer"] - d["mega"]["local"]["layer"]
            serial = d["step1"]["p2p"]["layer"] - d["step1"]["local"]["layer"]
            # Rounds of one mode differ by 10-35 us: a few us either way is noise.
            label = f"露出 {exposed:.0f}\n({exposed / top[2]:.0%})" if exposed >= 10 else f"露出 ≈0\n({exposed:.0f}，噪声内)"
            ax.text(2, top[2] + scale * 0.09, label, ha="center", va="bottom", fontsize=9, color=red, fontweight="bold")
            ax.text(1, top[1] + scale * 0.09, f"通信 {serial:.0f}", ha="center", va="bottom", fontsize=9, color=blue)
            ax.set_ylim(0, scale * 1.3)
            ax.set_xticks([0, 1, 2], ["MORI\n基线", "step1\n（串行）", "mega"], fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{'均衡' if routing == 'balanced' else '随机'}路由，每卡 {tokens} token", fontsize=10.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[row][0].set_ylabel("一层的时间（µs，最慢的 rank）", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (gray, orange, blue, red)]
    labels = ["quant", "MoE 计算（MORI 的 MoE；step1 / mega：通信写到本卡 scratch 时的整层时间）",
              "通信（MORI 的 dispatch、combine；step1 里 p2p 比 local 多的时间）", "mega 里露出来的通信（p2p − local）"]
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "overlap_sweep.png")
    print("wrote", HERE / "overlap_sweep.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("font_dir", nargs="?", default="/tmp/fonts")
    parser.add_argument("--from-dir")
    args = parser.parse_args()
    for f in Path(args.font_dir).glob("NotoSansSC-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    if args.from_dir:
        extract(args.from_dir)
    figure(json.loads(DATA.read_text()))


if __name__ == "__main__":
    main()
