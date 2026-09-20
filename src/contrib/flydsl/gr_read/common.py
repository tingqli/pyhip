# SPDX-License-Identifier: MIT
"""Fixed dimensions, weight layouts, and batch dispatch rules for GRRead."""

C, H, R = 4, 2560, 320
K = C * H
BLOCK_M = 256

# gfx942/80CU正常输出起点的历史矩阵校准值，单位为每CTA轮的相对时间。
# N2/N4/N8均为1CTA/CU；用实际CU数计算整轮及尾轮，不在调用时做autotune。
# 已测16档选择：N8=1/2/12/16/32k；N4=4/24/64k；其余N2。
# 未测行数是容量模型的确定性插值，不宣称每一个行数都实测最优。
_ROUND_COST = ((2, 280), (4, 144), (8, 77))

# Down M64：8档同址Down-only校准得到单CTA轮相对成本，N2减少工作但翻倍CTA。
# 当前两版VGPR均限制为1CTA/CU；按实际M256 padding后的M64 CTA数估算尾轮。
# gfx942/80CU时1..2560行选N2，其余选N1；其他CU数量仍用同一确定性模型。
_DOWN_ROUND_COST = ((1, 140), (2, 105))


def validate_rows(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("expected GRRead rows to be a nonnegative integer")


def select_n_splits(rows, compute_units):
    """Select N2/N4/N8 from task rounds at construction without a manual kernel switch."""
    validate_rows(rows)
    if not isinstance(compute_units, int) or isinstance(compute_units, bool) or compute_units <= 0:
        raise ValueError("compute_units must be a positive integer")
    if rows == 0:
        return 8
    m_tiles = (rows + BLOCK_M - 1) // BLOCK_M
    return min(_ROUND_COST, key=lambda item: ((m_tiles * item[0] + compute_units - 1) // compute_units) * item[1])[0]


def select_down_n_splits(rows, compute_units):
    """Select Down N1/N2 by CTA rounds times relative cost, without runtime autotuning."""
    validate_rows(rows)
    if not isinstance(compute_units, int) or isinstance(compute_units, bool) or compute_units <= 0:
        raise ValueError("compute_units must be a positive integer")
    if rows == 0:
        return 1
    m_tiles = (rows + BLOCK_M - 1) // BLOCK_M * (BLOCK_M // 64)
    return min(_DOWN_ROUND_COST,
               key=lambda item: ((m_tiles * item[0] + compute_units - 1) // compute_units) * item[1])[0]


def preshuffle_weight(weight):
    """Convert BF16 [N,K] to [N/16,K/32,4,16,8] during weight preparation only."""
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)