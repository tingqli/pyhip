#!/usr/bin/env python3
"""Render ATT slot0/slot1 function timelines for late-scale attention."""

import argparse
import json
from html import escape
from pathlib import Path

WIDTH = 1800
HEIGHT = 1320
LEFT = 255
RIGHT = 1715
CHART_WIDTH = RIGHT - LEFT
AXIS_MAX = 3400.0
BAR_HEIGHT = 62

COLORS = {
    "GEMM2": "#315c9b",
    "softmax": "#a45ab5",
    "GEMM1": "#167d75",
    "bridge": "#7f8791",
    "backedge": "#4d5660",
    "wait": "#d57a2a",
    "control": "#dce6ef",
    "late": "#e8f4ef",
    "grid": "#dfe4e8",
    "axis": "#3d4650",
    "text": "#20262d",
    "muted": "#66717d",
    "alert": "#b9445b",
}


def stage_color(stage):
    if stage.startswith("GEMM2"):
        return COLORS["GEMM2"]
    if stage.startswith("softmax"):
        return COLORS["softmax"]
    if stage.startswith("GEMM1"):
        return COLORS["GEMM1"]
    if "bridge" in stage:
        return COLORS["bridge"]
    return COLORS["backedge"]


def scale(value):
    return value / AXIS_MAX * CHART_WIDTH


class Svg:
    def __init__(self):
        self.lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
                f'viewBox="0 0 {WIDTH} {HEIGHT}" style="max-width:100%;height:auto;display:block" '
                'role="img" aria-labelledby="title desc">'
            ),
            '<title id="title">Attention ATT slot0/slot1主要功能与指令</title>',
            (
                '<desc id="desc">完整后移sum的control与late-scale版本，按ATT首批resident slot的'
                "平均cycle/tile展示GEMM2、softmax/K pipeline、GEMM1、bridge和backedge。</desc>"
            ),
            "<style>",
            (
                "text { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', "
                "'DejaVu Sans', sans-serif; fill: #20262d; }"
            ),
            ".title { font-size: 31px; font-weight: 700; }",
            ".subtitle { font-size: 16px; fill: #66717d; }",
            ".group { font-size: 19px; font-weight: 700; }",
            ".row { font-size: 17px; font-weight: 700; }",
            ".small { font-size: 13px; fill: #66717d; }",
            ".axis { font-size: 13px; fill: #66717d; }",
            ".segment { font-size: 13px; font-weight: 700; fill: #ffffff; }",
            ".segment-dark { font-size: 13px; font-weight: 700; fill: #20262d; }",
            ".callout { font-size: 14px; font-weight: 700; }",
            ".flow-head { font-size: 17px; font-weight: 700; }",
            ".flow { font-size: 14px; }",
            ".note { font-size: 14px; fill: #66717d; }",
            "</style>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def add(self, text):
        self.lines.append(text)

    def text(self, x, y, text, css="", anchor="start", fill=None):
        cls = f' class="{css}"' if css else ""
        fill_attr = f' fill="{fill}"' if fill else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{cls}{fill_attr}>'
            f"{escape(text)}</text>"
        )

    def rect(self, x, y, width, height, fill, stroke=None, radius=3, title=None):
        stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0):.2f}" '
            f'height="{height:.2f}" rx="{radius}" fill="{fill}"{stroke_attr}>'
        )
        if title:
            self.add(f"<title>{escape(title)}</title>")
        self.add("</rect>")

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
        )

    def polygon(self, points, fill):
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polygon points="{value}" fill="{fill}"/>')

    def finish(self):
        return "\n".join([*self.lines, "</svg>", ""])


def draw_axis(svg, top, bottom):
    for tick in range(0, 3401, 400):
        x = LEFT + scale(tick)
        svg.line(x, top, x, bottom, COLORS["grid"], dash="4 6")
        svg.text(x, top - 11, str(tick), "axis", anchor="middle")
    svg.line(LEFT, top, RIGHT, top, COLORS["axis"], width=1.4)
    svg.text(RIGHT, top - 35, "ATT cycle / BN32 tile", "axis", anchor="end")


def draw_lane(svg, y, group_name, slot_name, segments, expected_duration, late=False):
    svg.text(LEFT - 24, y + 24, slot_name, "row", anchor="end")
    svg.text(
        LEFT - 24, y + 48, f"duration {expected_duration:.1f}", "small", anchor="end"
    )
    cursor = LEFT
    for item in segments:
        stage = item["stage"]
        value = float(item["cycles"])
        width = scale(value)
        color = stage_color(stage)
        title = f"{group_name} {slot_name} · {stage}: {value:.3f} cycles/tile"
        svg.rect(cursor, y, width, BAR_HEIGHT, color, title=title)
        short = stage.replace("softmax/K-pipeline", "softmax/K").replace("phase64 ", "")
        if width >= 105:
            svg.text(cursor + width / 2, y + 27, short, "segment", anchor="middle")
            svg.text(
                cursor + width / 2, y + 48, f"{value:.1f}", "segment", anchor="middle"
            )
        elif width >= 55:
            svg.text(
                cursor + width / 2, y + 37, f"{value:.0f}", "segment", anchor="middle"
            )
        cursor += width
    svg.rect(LEFT, y, cursor - LEFT, BAR_HEIGHT, "none", stroke=COLORS["axis"])
    residual = expected_duration - sum(float(item["cycles"]) for item in segments)
    svg.text(cursor + 10, y + 27, f"Σ {expected_duration:.1f}", "callout")
    svg.text(cursor + 10, y + 48, f"边界/取整 {residual:+.1f}", "small")
    if late and slot_name == "slot1 (slow)":
        svg.rect(
            LEFT,
            y - 5,
            cursor - LEFT,
            BAR_HEIGHT + 10,
            "none",
            stroke=COLORS["alert"],
            radius=5,
        )


def draw_skew(svg, y, fast_duration, slow_duration, label):
    fast_x = LEFT + scale(fast_duration)
    slow_x = LEFT + scale(slow_duration)
    svg.line(fast_x, y, slow_x, y, COLORS["alert"], width=3)
    svg.line(fast_x, y - 7, fast_x, y + 7, COLORS["alert"], width=3)
    svg.line(slow_x, y - 7, slow_x, y + 7, COLORS["alert"], width=3)
    svg.text(
        (fast_x + slow_x) / 2,
        y - 10,
        label,
        "callout",
        anchor="middle",
        fill=COLORS["alert"],
    )


def draw_legend(svg, y):
    items = [
        ("GEMM2: Vᵀ @ Pᵀ", COLORS["GEMM2"]),
        ("softmax + K pipeline", COLORS["softmax"]),
        ("GEMM1: K @ Qᵀ", COLORS["GEMM1"]),
        ("bridge", COLORS["bridge"]),
        ("backedge", COLORS["backedge"]),
    ]
    x = LEFT
    for label, color in items:
        svg.rect(x, y - 15, 20, 20, color)
        svg.text(x + 29, y, label, "flow")
        x += 270


def draw_instruction_flow(svg, x, y, title, lines, color):
    width = 330
    height = 190
    svg.rect(x, y, width, height, "#f6f8fa", stroke="#cfd6dc", radius=5)
    svg.rect(x, y, width, 35, color, radius=5)
    svg.text(x + 14, y + 24, title, "flow-head", fill="#ffffff")
    for index, line in enumerate(lines):
        svg.text(x + 14, y + 64 + index * 25, f"• {line}", "flow")


def render(data):
    att = data["flydsl_deferred_sum_reduce_vpk_ablation"]["late_scale_att"]
    segments = att["slot_function_segments_cycles_per_tile"]
    durations = att["resident_slot_cycles_per_tile"]
    flow = att["major_instruction_flow_per_128_mfma_pair"]
    phase = att["slow_slot_phase_delta_cycles"]

    svg = Svg()
    svg.text(42, 52, "ATT resident slot0 / slot1：主要功能与机器指令", "title")
    svg.text(
        42,
        82,
        "gfx942 · H=1, M=N=40960, D=128 · 完整后移sum · 首批resident slots平均时间线",
        "subtitle",
    )
    draw_axis(svg, 135, 650)

    svg.text(42, 165, "Control · defer_all_inline", "group")
    draw_lane(
        svg,
        190,
        "control",
        "slot0 (fast)",
        segments["defer_all_inline"]["slot0"],
        durations["defer_all_inline"]["slot0_duration"],
    )
    draw_lane(
        svg,
        280,
        "control",
        "slot1 (slow)",
        segments["defer_all_inline"]["slot1"],
        durations["defer_all_inline"]["slot1_duration"],
    )
    draw_skew(
        svg,
        360,
        durations["defer_all_inline"]["slot0_duration"],
        durations["defer_all_inline"]["slot1_duration"],
        f"slot完成差 {durations['defer_all_inline']['completion_skew']:.1f}",
    )

    svg.text(42, 415, "Late-scale · defer_all_inline_late_scale", "group")
    draw_lane(
        svg,
        440,
        "late-scale",
        "slot0 (fast)",
        segments["defer_all_inline_late_scale"]["slot0"],
        durations["defer_all_inline_late_scale"]["slot0_duration"],
        late=True,
    )
    draw_lane(
        svg,
        530,
        "late-scale",
        "slot1 (slow)",
        segments["defer_all_inline_late_scale"]["slot1"],
        durations["defer_all_inline_late_scale"]["slot1_duration"],
        late=True,
    )
    draw_skew(
        svg,
        610,
        durations["defer_all_inline_late_scale"]["slot0_duration"],
        durations["defer_all_inline_late_scale"]["slot1_duration"],
        f"slot完成差 {durations['defer_all_inline_late_scale']['completion_skew']:.1f} (+144.8)",
    )
    draw_legend(svg, 675)

    svg.rect(42, 715, WIDTH - 84, 108, COLORS["late"], stroke="#b8d5c9", radius=5)
    svg.text(62, 748, "late-scale slot1 的关键等待放大", "flow-head")
    svg.text(
        62,
        779,
        (
            f"phase34 +{phase['phase34']:.1f} · phase42 +{phase['phase42']:.1f} · "
            f"phase106 +{phase['phase106']:.1f} · phase108 +{phase['phase108']:.1f} cycles"
        ),
        "callout",
        fill=COLORS["alert"],
    )
    svg.text(
        62,
        805,
        "softmax窗口缩短后，slow slot更早进入下一GEMM1，progressive K ds_read / lgkmcnt延迟暴露。",
        "note",
    )

    draw_instruction_flow(
        svg,
        42,
        850,
        "GEMM2-A / GEMM2-B",
        flow["GEMM2-A/B"],
        COLORS["GEMM2"],
    )
    draw_instruction_flow(
        svg,
        392,
        850,
        "softmax + K pipeline（共有）",
        flow["softmax/K-pipeline A/B shared"][:5],
        COLORS["softmax"],
    )
    draw_instruction_flow(
        svg,
        742,
        850,
        "late-scale score路径",
        flow["softmax late-scale score path"],
        "#7f4b8e",
    )
    draw_instruction_flow(
        svg,
        1092,
        850,
        "GEMM1-A / GEMM1-B",
        flow["GEMM1-A/B"],
        COLORS["GEMM1"],
    )
    draw_instruction_flow(
        svg,
        1442,
        850,
        "bridge / backedge",
        flow["bridge/backedge"],
        COLORS["bridge"],
    )

    svg.rect(42, 1070, WIDTH - 84, 190, "#f6f8fa", stroke="#cfd6dc", radius=5)
    svg.text(62, 1105, "如何读图", "flow-head")
    svg.text(
        62,
        1136,
        "1. 每条横条是同一slot平均处理一个BN32 tile的动态时间，不是静态指令条数。",
        "flow",
    )
    svg.text(
        62,
        1164,
        "2. 紫色softmax/K段同时包含max/DS归约、EXP、running-sum FMA、BF16 pack、future-K load与K LDS write。",
        "flow",
    )
    svg.text(
        62,
        1192,
        "3. 绿色GEMM1段包含32条MFMA及progressive lgkmcnt等待；late-scale slot1的主要回退落在这些等待点。",
        "flow",
    )
    svg.text(
        62,
        1220,
        "4. late-scale让slot0快116.7 cycles/tile，却让决定吞吐的slot1慢28.1，物理wall因此回退约0.84%。",
        "flow",
    )
    svg.text(
        62,
        1248,
        "数据源：四份单dispatch ATT；图中control/late-scale使用完整后移sum的16个物理SIMD。",
        "note",
    )
    return svg.finish()


def main():
    bundle_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=bundle_dir / "data/attn-jit-coissue-optimization-gfx942.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=bundle_dir / "images/attn-late-scale-slot-functions-gfx942.svg",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data))
    print(args.output)


if __name__ == "__main__":
    main()
