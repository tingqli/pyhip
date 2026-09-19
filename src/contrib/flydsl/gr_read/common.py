# SPDX-License-Identifier: MIT
"""GRRead的固定维度、权重布局与batch调度规则。"""

C, H, R = 4, 2560, 320
K = C * H
MAX_ROWS = 65536
BLOCK_M = 256

# gfx942/80CU正常输出起点的历史矩阵校准值，单位为每CTA轮的相对时间。
# N2/N4/N8均为1CTA/CU；用实际CU数计算整轮及尾轮，不在调用时做autotune。
# 已测16档选择：N8=1/2/12/16/32k；N4=4/24/64k；其余N2。
# 未测行数是容量模型的确定性插值，不宣称每一个行数都实测最优。
_ROUND_COST = ((2, 280), (4, 144), (8, 77))


def validate_rows(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or not 0 <= rows <= MAX_ROWS:
        raise ValueError(f"GRRead supports 0..{MAX_ROWS} rows")


def select_n_splits(rows, compute_units):
    """构造时按任务尾轮选择N2/N4/N8；不暴露手动kernel开关。"""
    validate_rows(rows)
    if not isinstance(compute_units, int) or isinstance(compute_units, bool) or compute_units <= 0:
        raise ValueError("compute_units must be a positive integer")
    if rows == 0:
        return 8
    m_tiles = (rows + BLOCK_M - 1) // BLOCK_M
    return min(_ROUND_COST, key=lambda item: ((m_tiles * item[0] + compute_units - 1) // compute_units) * item[1])[0]


def preshuffle_weight(weight):
    """BF16 [N,K] -> [N/16,K/32,4,16,8]；仅权重准备时执行。"""
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)