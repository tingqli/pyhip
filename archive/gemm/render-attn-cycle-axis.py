#!/usr/bin/env python3
"""Render the attention ATT cycle decomposition as a shared-axis SVG chart."""

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

WIDTH = 1600
HEIGHT = 1880
CHART_LEFT = 300
CHART_RIGHT = 1510
CHART_WIDTH = CHART_RIGHT - CHART_LEFT
BAR_HEIGHT = 50
AXIS_COLOR = "#3d4650"
GRID_COLOR = "#dfe4e8"
TEXT_COLOR = "#20262d"
MUTED_COLOR = "#66717d"

COLORS = {
    "perfect": "#dce8d8",
    "shadow_non_mfma": "#167d75",
    "shadow_mfma_only": "#315c9b",
    "shadow_no_issue": "#b9445b",
    "outside_non_mfma": "#7dbbb4",
    "outside_mfma_only": "#7f9cc2",
    "outside_no_issue": "#d57a2a",
    "shadow_alias": "#9aa1a9",
    "wall_boundary": "#343a40",
    "MFMA": "#315c9b",
    "TRANS": "#a45ab5",
    "LDS/SMEM-wait": "#d99c24",
    "VMEM-load": "#4b9ccb",
    "scheduler/ready": "#6c737d",
    "LDS/crosslane": "#158f75",
    "VALU": "#c45c8a",
    "barrier": "#c65d2e",
    "other": "#aab0b6",
}


@dataclass(frozen=True)
class Segment:
    label: str
    value: float
    color: str
    text_color: str = "#ffffff"
    short_label: str | None = None


class Svg:
    def __init__(self):
        self.lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
                f'viewBox="0 0 {WIDTH} {HEIGHT}" style="max-width:100%;height:auto;display:block" '
                'role="img" aria-labelledby="chart-title chart-desc">'
            ),
            '<title id="chart-title">Attention JIT cycle/tile 共享横轴分解</title>',
            (
                '<desc id="chart-desc">H=1, M=N=40960 的墙钟、ATT物理时间线、MFMA shadow和'
                "no-issue原因均绘制在同一个cycle/tile横轴上。</desc>"
            ),
            "<style>",
            (
                "text { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', "
                "'DejaVu Sans', sans-serif; fill: #20262d; }"
            ),
            ".title { font-size: 30px; font-weight: 700; }",
            ".subtitle { font-size: 17px; fill: #66717d; }",
            ".row-title { font-size: 18px; font-weight: 700; }",
            ".row-subtitle { font-size: 14px; fill: #66717d; }",
            ".axis { font-size: 13px; fill: #66717d; }",
            ".segment { font-size: 13px; font-weight: 700; }",
            ".total { font-size: 14px; font-weight: 700; }",
            ".legend { font-size: 14px; }",
            ".note { font-size: 14px; fill: #66717d; }",
            ".explain-title { font-size: 20px; font-weight: 700; }",
            ".explain-head { font-size: 16px; font-weight: 700; }",
            ".explain { font-size: 14px; }",
            ".example { font-size: 13px; font-weight: 700; }",
            "</style>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def add(self, text):
        self.lines.append(text)

    def text(self, x, y, text, css_class="", anchor="start", fill=None):
        fill_attr = f' fill="{fill}"' if fill else ""
        class_attr = f' class="{css_class}"' if css_class else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{class_attr}{fill_attr}>'
            f"{escape(text)}</text>"
        )

    def rect(self, x, y, width, height, color, title=None, stroke=None, radius=2):
        stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.add(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(width, 0):.3f}" height="{height:.3f}" '
            f'rx="{radius}" fill="{color}"{stroke_attr}>'
        )
        if title:
            self.add(f"<title>{escape(title)}</title>")
        self.add("</rect>")

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
        )

    def finish(self):
        return "\n".join([*self.lines, "</svg>", ""])


def require_close(name, actual, expected, tolerance=2e-5):
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise ValueError(f"{name} does not close: {actual} != {expected}")


def by_key(rows, *keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def compact_categories(rows, keep):
    values = {row["category"]: float(row["cycles"]) for row in rows}
    compacted = [(category, values.pop(category, 0.0)) for category in keep]
    compacted.append(("other", sum(values.values())))
    return [(category, value) for category, value in compacted if value > 0]


def scale(value, axis_max):
    return value / axis_max * CHART_WIDTH


def draw_axis(svg, axis_max, top, bottom):
    for tick in range(0, 1601, 200):
        x = CHART_LEFT + scale(tick, axis_max)
        svg.line(x, top, x, bottom, GRID_COLOR, dash="4 6")
        svg.text(x, top - 10, f"{tick}", "axis", anchor="middle")
    svg.line(CHART_LEFT, top, CHART_RIGHT, top, AXIS_COLOR, width=1.5)
    svg.text(CHART_RIGHT, top - 34, "cycle / BN32 tile", "axis", anchor="end")


def draw_bar(svg, axis_max, y, title, subtitle, segments, total, label_threshold=75):
    svg.text(CHART_LEFT - 24, y + 21, title, "row-title", anchor="end")
    svg.text(CHART_LEFT - 24, y + 43, subtitle, "row-subtitle", anchor="end")
    cursor = CHART_LEFT
    for segment in segments:
        width = scale(segment.value, axis_max)
        tooltip = f"{segment.label}: {segment.value:.3f} cycle/tile"
        svg.rect(cursor, y, width, BAR_HEIGHT, segment.color, title=tooltip)
        label = segment.short_label or segment.label
        if width >= label_threshold:
            svg.text(
                cursor + width / 2,
                y + 31,
                f"{label} {segment.value:.1f}",
                "segment",
                anchor="middle",
                fill=segment.text_color,
            )
        cursor += width
    outline_width = scale(total, axis_max)
    svg.rect(
        CHART_LEFT, y, outline_width, BAR_HEIGHT, "none", stroke=AXIS_COLOR, radius=2
    )
    svg.text(cursor + 9, y + 31, f"Σ {total:.3f}", "total")


def draw_legend(svg, x, y, items, columns=4, column_width=300):
    for index, (label, color) in enumerate(items):
        row = index // columns
        column = index % columns
        item_x = x + column * column_width
        item_y = y + row * 30
        svg.rect(item_x, item_y - 14, 18, 18, color, radius=2)
        svg.text(item_x + 27, item_y, label, "legend")


def draw_reading_guide(svg):
    panel_x = 44
    panel_y = 1130
    panel_width = WIDTH - 88
    panel_height = 710
    svg.rect(
        panel_x,
        panel_y,
        panel_width,
        panel_height,
        "#f6f8fa",
        stroke="#cfd6dc",
        radius=5,
    )
    svg.text(70, 1167, "如何读图：每一行、每种颜色和一个具体ATT示例", "explain-title")

    svg.text(70, 1202, "五条横条分别表示", "explain-head")
    svg.text(
        70,
        1230,
        "1  墙钟闭合：实测 = 完美模型 + shadow内/外归一残差 + ATT外边界。",
        "explain",
    )
    svg.text(
        70,
        1256,
        "2  ATT物理时间线：S=MFMA shadow内，O=shadow外；六种状态互斥。",
        "explain",
    )
    svg.text(
        70,
        1282,
        "3  MFMA逻辑shadow：64×12的理论容量；灰色alias不是独立物理机会。",
        "explain",
    )
    svg.text(
        70,
        1308,
        "4/5 no-issue原因：按active wave当时阻塞的PC归因，是raw物理cycle。",
        "explain",
    )

    svg.line(790, 1188, 790, 1322, "#cfd6dc")
    svg.text(820, 1202, "状态颜色的含义", "explain-head")
    svg.text(
        820,
        1230,
        "深/浅绿 non-MFMA issue：分别表示shadow内/外有非MFMA工作发射。",
        "explain",
    )
    svg.text(
        820, 1256, "深/浅蓝 MFMA-only：分别表示shadow内/外只有MFMA工作发射。", "explain"
    )
    svg.text(
        820,
        1282,
        "红/橙 no issue：分别表示shadow内/外整个物理SIMD没有指令发射。",
        "explain",
    )
    svg.text(
        820,
        1308,
        "灰色  alias：两个resident wave的逻辑shadow重叠，只算一次物理时间。",
        "explain",
    )

    svg.text(
        70,
        1354,
        "示例：两个MFMA如何形成hidden、MFMA-only、no-issue与alias",
        "explain-head",
    )
    example_left = 340
    slot_width = 170
    example_y = 1392
    ticks = [100, 104, 108, 112, 116, 120, 124]
    for index, tick in enumerate(ticks):
        x = example_left + index * slot_width
        svg.line(x, example_y - 10, x, example_y + 55, "#bec6cd")
        svg.text(x, example_y - 18, f"t={tick}", "axis", anchor="middle")

    example_segments = [
        Segment("MFMA0 issue", 4, COLORS["MFMA"], short_label="MFMA0 issue"),
        Segment(
            "S/non-MFMA: v_add",
            4,
            COLORS["shadow_non_mfma"],
            short_label="v_add → S/non-MFMA",
        ),
        Segment(
            "S/MFMA-only: MFMA1",
            4,
            COLORS["shadow_mfma_only"],
            short_label="MFMA1 → S/MFMA-only",
        ),
        Segment(
            "S/no-issue", 4, COLORS["shadow_no_issue"], short_label="idle → S/no-issue"
        ),
        Segment(
            "S/non-MFMA: v_exp",
            4,
            COLORS["shadow_non_mfma"],
            short_label="v_exp → S/non-MFMA",
        ),
        Segment(
            "S/no-issue", 4, COLORS["shadow_no_issue"], short_label="idle → S/no-issue"
        ),
    ]
    cursor = example_left
    for segment in example_segments:
        svg.rect(cursor, example_y, slot_width, 46, segment.color, title=segment.label)
        svg.text(
            cursor + slot_width / 2,
            example_y + 28,
            segment.short_label or segment.label,
            "example",
            anchor="middle",
            fill=segment.text_color,
        )
        cursor += slot_width

    shadow0_start = example_left + slot_width
    shadow0_end = example_left + 4 * slot_width
    shadow1_start = example_left + 3 * slot_width
    shadow1_end = example_left + 6 * slot_width
    bracket_y = 1460
    svg.line(
        shadow0_start,
        bracket_y,
        shadow0_end,
        bracket_y,
        COLORS["shadow_mfma_only"],
        width=3,
    )
    svg.line(
        shadow0_start,
        bracket_y - 6,
        shadow0_start,
        bracket_y + 6,
        COLORS["shadow_mfma_only"],
        width=3,
    )
    svg.line(
        shadow0_end,
        bracket_y - 6,
        shadow0_end,
        bracket_y + 6,
        COLORS["shadow_mfma_only"],
        width=3,
    )
    svg.text(
        (shadow0_start + shadow0_end) / 2,
        bracket_y + 21,
        "MFMA0 shadow [104,116)",
        "axis",
        anchor="middle",
    )

    bracket_y = 1500
    svg.line(
        shadow1_start,
        bracket_y,
        shadow1_end,
        bracket_y,
        COLORS["outside_mfma_only"],
        width=3,
    )
    svg.line(
        shadow1_start,
        bracket_y - 6,
        shadow1_start,
        bracket_y + 6,
        COLORS["outside_mfma_only"],
        width=3,
    )
    svg.line(
        shadow1_end,
        bracket_y - 6,
        shadow1_end,
        bracket_y + 6,
        COLORS["outside_mfma_only"],
        width=3,
    )
    svg.text(
        (shadow1_start + shadow1_end) / 2,
        bracket_y + 21,
        "MFMA1 shadow [112,124)",
        "axis",
        anchor="middle",
    )

    overlap_x = shadow1_start
    overlap_width = slot_width
    svg.rect(
        overlap_x,
        1534,
        overlap_width,
        24,
        COLORS["shadow_alias"],
        title="逻辑shadow重叠4 cycles",
    )
    svg.text(
        overlap_x + overlap_width / 2,
        1551,
        "alias 4 cycles",
        "example",
        anchor="middle",
        fill=TEXT_COLOR,
    )
    svg.text(
        overlap_x + overlap_width + 18,
        1551,
        "[112,116)同时属于两个逻辑shadow，但物理时间线只计算一次。",
        "explain",
    )

    svg.text(70, 1590, "阻塞归因示例", "explain-head")
    svg.text(
        210,
        1590,
        "若一个4-cycle no-issue区间内，wave0阻塞在MFMA、wave1阻塞在s_waitcnt lgkmcnt(0)，",
        "explain",
    )
    svg.text(
        210,
        1616,
        "则raw权重等分为MFMA 2 cycles + LDS/SMEM-wait 2 cycles；它不是两次墙钟时间。",
        "explain",
    )

    svg.line(70, 1646, 1530, 1646, "#cfd6dc")
    svg.text(70, 1680, "阻塞类别逐项含义", "explain-head")
    svg.text(
        70,
        1710,
        "MFMA：active wave停在MFMA PC，通常是accumulator RAW或MFMA pipeline依赖。",
        "explain",
    )
    svg.text(
        70,
        1736,
        "TRANS：停在v_exp_f32/v_rcp_f32，表示超越函数结果或TRANS pipeline依赖。",
        "explain",
    )
    svg.text(
        70,
        1762,
        "LDS wait：停在s_waitcnt lgkmcnt(...)；LDS/SMEM请求尚未满足。",
        "explain",
    )
    svg.text(
        70,
        1788,
        "VMEM load：停在buffer_load；global/L2/HBM请求仍处于ATT duration。",
        "explain",
    )
    svg.text(
        820,
        1710,
        "scheduler：上一条长duration已结束，但物理SIMD仍没有下一条issue。",
        "explain",
    )
    svg.text(
        820,
        1736,
        "LDS/crosslane：停在ds_read/write/swizzle/bpermute等DS指令。",
        "explain",
    )
    svg.text(
        820,
        1762,
        "VALU：停在普通vector ALU PC，通常是数据依赖或VALU pipeline空洞。",
        "explain",
    )
    svg.text(
        820,
        1788,
        "barrier/other：wave到达差；以及少量SALU、store、wait和trace edge。",
        "explain",
    )


def render(data):
    closure = data["closure_cycles_per_tile"]
    decomposition = data["mfma_shadow_decomposition"]
    summary = decomposition["summary_cycles_per_tile"]
    matrix = by_key(decomposition["timeline_matrix_cycles_per_tile"], "region", "state")
    no_issue = {row["region"]: row for row in decomposition["no_issue_regions"]}

    wall = float(closure["wall_reconstructed"])
    att_path = float(closure["att_critical_path"])
    perfect = float(closure["perfect_model_including_amortized_fixed_cost"])
    shadow_residual = float(
        no_issue["mfma_shadow"]["attributed_residual_cycles_per_tile"]
    )
    outside_residual = float(
        no_issue["outside_shadow"]["attributed_residual_cycles_per_tile"]
    )
    wall_boundary = float(closure["wall_outside_att"])
    axis_max = wall

    require_close(
        "wall", perfect + shadow_residual + outside_residual + wall_boundary, wall
    )
    require_close(
        "ATT matrix", sum(float(row["cycles"]) for row in matrix.values()), att_path
    )
    require_close(
        "physical no-issue",
        float(summary["no_issue_in_shadow"])
        + float(summary["no_issue_outside_shadow"]),
        float(summary["all_physical_no_issue"]),
    )
    require_close(
        "logical shadow",
        float(summary["overlapping_shadow_alias"])
        + float(summary["non_mfma_issue_in_shadow"])
        + float(summary["mfma_only_issue_in_shadow"])
        + float(summary["no_issue_in_shadow"]),
        float(summary["logical_shadow"]),
    )

    shadow_causes = compact_categories(
        no_issue["mfma_shadow"]["categories_cycles_per_tile"],
        [
            "MFMA",
            "scheduler/ready",
            "LDS/SMEM-wait",
            "TRANS",
            "VMEM-load",
            "LDS/crosslane",
            "barrier",
        ],
    )
    outside_causes = compact_categories(
        no_issue["outside_shadow"]["categories_cycles_per_tile"],
        [
            "TRANS",
            "LDS/SMEM-wait",
            "VMEM-load",
            "scheduler/ready",
            "VALU",
            "LDS/crosslane",
            "MFMA",
            "barrier",
        ],
    )

    svg = Svg()
    svg.text(44, 51, "Attention JIT 时间消耗：共享 cycle/tile 横轴", "title")
    svg.text(
        44,
        82,
        "gfx942 · H=1, M=N=40960, D=128 · 彩色段是聚合预算，不表示真实时间先后顺序",
        "subtitle",
    )
    draw_axis(svg, axis_max, 135, 890)

    draw_bar(
        svg,
        axis_max,
        180,
        "墙钟闭合",
        "实测 = perfect + 两类残差 + 边界",
        [
            Segment("完美模型", perfect, COLORS["perfect"], TEXT_COLOR, "perfect"),
            Segment(
                "shadow内残差",
                shadow_residual,
                COLORS["shadow_no_issue"],
                short_label="shadow residual",
            ),
            Segment(
                "shadow外残差",
                outside_residual,
                COLORS["outside_no_issue"],
                short_label="outside residual",
            ),
            Segment(
                "ATT外边界",
                wall_boundary,
                COLORS["wall_boundary"],
                short_label="boundary",
            ),
        ],
        wall,
        label_threshold=90,
    )

    draw_bar(
        svg,
        axis_max,
        315,
        "ATT物理时间线",
        "S=shadow内 / O=shadow外",
        [
            Segment(
                "shadow/non-MFMA",
                float(matrix["mfma_shadow", "non_mfma_issue"]["cycles"]),
                COLORS["shadow_non_mfma"],
                short_label="S:non-MFMA",
            ),
            Segment(
                "shadow/MFMA-only",
                float(matrix["mfma_shadow", "mfma_only_issue"]["cycles"]),
                COLORS["shadow_mfma_only"],
                short_label="S:MFMA",
            ),
            Segment(
                "shadow/no-issue",
                float(matrix["mfma_shadow", "no_issue"]["cycles"]),
                COLORS["shadow_no_issue"],
                short_label="S:idle",
            ),
            Segment(
                "outside/non-MFMA",
                float(matrix["outside_shadow", "non_mfma_issue"]["cycles"]),
                COLORS["outside_non_mfma"],
                TEXT_COLOR,
                "O:non-MFMA",
            ),
            Segment(
                "outside/MFMA-only",
                float(matrix["outside_shadow", "mfma_only_issue"]["cycles"]),
                COLORS["outside_mfma_only"],
                TEXT_COLOR,
                "O:MFMA",
            ),
            Segment(
                "outside/no-issue",
                float(matrix["outside_shadow", "no_issue"]["cycles"]),
                COLORS["outside_no_issue"],
                short_label="O:idle",
            ),
            Segment(
                "ATT外边界",
                wall_boundary,
                COLORS["wall_boundary"],
                short_label="boundary",
            ),
        ],
        wall,
        label_threshold=78,
    )

    draw_bar(
        svg,
        axis_max,
        450,
        "MFMA逻辑shadow",
        "64 × 12 = 768；alias不是独立物理机会",
        [
            Segment(
                "resident shadow重叠",
                float(summary["overlapping_shadow_alias"]),
                COLORS["shadow_alias"],
                TEXT_COLOR,
                "alias",
            ),
            Segment(
                "已隐藏non-MFMA",
                float(summary["non_mfma_issue_in_shadow"]),
                COLORS["shadow_non_mfma"],
                short_label="hidden",
            ),
            Segment(
                "MFMA-only",
                float(summary["mfma_only_issue_in_shadow"]),
                COLORS["shadow_mfma_only"],
                short_label="MFMA-only",
            ),
            Segment(
                "no-issue",
                float(summary["no_issue_in_shadow"]),
                COLORS["shadow_no_issue"],
                short_label="idle",
            ),
        ],
        float(summary["logical_shadow"]),
        label_threshold=66,
    )

    draw_bar(
        svg,
        axis_max,
        600,
        "shadow内 no-issue",
        "A ∩ B = 335.348；颜色=阻塞PC原因",
        [
            Segment(
                category,
                value,
                COLORS[category],
                short_label=category.replace("scheduler/ready", "scheduler"),
            )
            for category, value in shadow_causes
        ],
        float(summary["no_issue_in_shadow"]),
        label_threshold=68,
    )

    draw_bar(
        svg,
        axis_max,
        750,
        "shadow外 no-issue",
        "B \\ A = 399.370；颜色=阻塞PC原因",
        [
            Segment(
                category,
                value,
                COLORS[category],
                short_label=category.replace("scheduler/ready", "scheduler"),
            )
            for category, value in outside_causes
        ],
        float(summary["no_issue_outside_shadow"]),
        label_threshold=58,
    )

    svg.text(44, 924, "状态颜色", "row-title")
    draw_legend(
        svg,
        170,
        924,
        [
            ("shadow: non-MFMA issue", COLORS["shadow_non_mfma"]),
            ("shadow: MFMA-only", COLORS["shadow_mfma_only"]),
            ("shadow: no issue", COLORS["shadow_no_issue"]),
            ("outside: non-MFMA issue", COLORS["outside_non_mfma"]),
            ("outside: MFMA-only", COLORS["outside_mfma_only"]),
            ("outside: no issue", COLORS["outside_no_issue"]),
            ("resident shadow alias", COLORS["shadow_alias"]),
            ("ATT外墙钟边界", COLORS["wall_boundary"]),
        ],
        columns=4,
        column_width=330,
    )

    svg.text(44, 1007, "阻塞类别", "row-title")
    draw_legend(
        svg,
        170,
        1007,
        [
            ("MFMA", COLORS["MFMA"]),
            ("TRANS", COLORS["TRANS"]),
            ("LDS wait", COLORS["LDS/SMEM-wait"]),
            ("VMEM load", COLORS["VMEM-load"]),
            ("scheduler", COLORS["scheduler/ready"]),
            ("LDS/crosslane", COLORS["LDS/crosslane"]),
            ("VALU", COLORS["VALU"]),
            ("barrier", COLORS["barrier"]),
            ("other", COLORS["other"]),
        ],
        columns=5,
        column_width=255,
    )
    svg.text(
        44,
        1092,
        "注：no-issue原因按active wave阻塞PC等分归因；颜色面积用于比较规模，不代表单项可独立消除。",
        "note",
    )
    draw_reading_guide(svg)
    return svg.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/data/attn-jit-att-cycle-ledger-gfx942.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/attn-jit-cycle-axis-gfx942.svg"),
    )
    args = parser.parse_args()
    with args.input.open() as stream:
        data = json.load(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data))
    print(args.output)


if __name__ == "__main__":
    main()
