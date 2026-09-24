# SPDX-License-Identifier: MIT
"""Fixed dimensions, weight layouts, and batch dispatch rules for GRRead."""

C, H, R = 4, 2560, 320
K = C * H
BLOCK_M = 256

# Up M256：gfx942/80CU正常输出起点的历史校准值，每CTA轮的相对成本。
# 64KiB LDS限制为1CTA/CU；计算整轮及尾轮，不在调用时做autotune。
# 原N2/N4/N8模型用于大batch和其它CU配置。
_ROUND_COST = ((2, 280), (4, 144), (8, 77))
# 80CU同址512/1024/2048/4096对照：N10约65、N20约40每轮。
# 仅小batch加入候选；中间行数按轮数插值，不宣称逐shape实测最优。
_SMALL_UP_ROUND_COST = _ROUND_COST + ((10, 65), (20, 40))

# Down M64回退路径：8档同址Down-only校准得到单CTA轮相对成本。
# 这两版VGPR均限制为1CTA/CU，不能把该驻留假设套到M32。
# M64模型在80CU时1..2560行选N2，其余选N1；小batch覆盖规则见select_down_config。
_DOWN_ROUND_COST = ((1, 140), (2, 105))


def validate_rows(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("expected GRRead rows to be a nonnegative integer")


def select_n_splits(rows, compute_units):
    """Select N40 for calibrated tiny batches, otherwise use the existing CTA-round model."""
    validate_rows(rows)
    if not isinstance(compute_units, int) or isinstance(compute_units, bool) or compute_units <= 0:
        raise ValueError("compute_units must be a positive integer")
    if rows == 0:
        return 8
    # 80CU的32..512行同址对照支持单H64组；1024开始回退，不推广到更大batch。
    if compute_units == 80 and rows <= 512:
        return 40
    m_tiles = (rows + BLOCK_M - 1) // BLOCK_M
    costs = _SMALL_UP_ROUND_COST if compute_units == 80 and rows <= 4096 else _ROUND_COST
    return min(costs, key=lambda item: ((m_tiles * item[0] + compute_units - 1) // compute_units) * item[1])[0]


def select_down_n_splits(rows, compute_units):
    """Select N1/N2 for the M64 fallback using the calibrated CTA-round model."""
    validate_rows(rows)
    if not isinstance(compute_units, int) or isinstance(compute_units, bool) or compute_units <= 0:
        raise ValueError("compute_units must be a positive integer")
    if rows == 0:
        return 1
    m_tiles = (rows + 63) // 64
    return min(_DOWN_ROUND_COST,
               key=lambda item: ((m_tiles * item[0] + compute_units - 1) // compute_units) * item[1])[0]


def select_down_config(rows, compute_units):
    """Return (block_m, num_waves, n_splits, block_k) without runtime autotuning."""
    n_splits = select_down_n_splits(rows, compute_units)
    if compute_units == 80 and 33 <= rows <= 512:
        if rows <= 128:
            return 16, 2, 10, 1024
        if rows <= 256:
            return 16, 2, 10, 512
        return 32, 4, 5, 512
    # 80CU同址32..2048的BK128优于正式基线；中间行数是插值，非逐shape最优保证。
    # 4096/8192保留M64的数据复用；其它CU仍用原M64/BK64模型。
    if compute_units == 80 and 0 < rows <= 2048:
        return 32, 4, 5, 128
    return 64, 4, n_splits, 64


def select_up_config(rows, compute_units):
    """Return (M tile, N splits), retaining the original M256 path outside 33..512."""
    n_splits = select_n_splits(rows, compute_units)
    if compute_units == 80 and 33 <= rows <= 256:
        return (64 if rows <= 128 else 128), n_splits
    return 256, n_splits


def select_prefill_config(rows, compute_units):
    """Return the complete, fixed prefill configuration used during preparation."""
    dm, dw, dn, dk = select_down_config(rows, compute_units)
    um, un = select_up_config(rows, compute_units)
    swizzle = (7 if rows <= 128 else 6) if compute_units == 80 and 33 <= rows <= 512 else 3
    return dm, dw, dn, dk, um, un, swizzle


def preshuffle_weight(weight):
    """Convert BF16 [N,K] to [N/16,K/32,4,16,8] during weight preparation only."""
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)


def prepare_weights(w_down, w_up):
    """Pack original BF16 matrices once; the result is shared by all T and both GR read backends."""
    import torch

    if w_down.shape != (R, K) or w_up.shape != (K, R):
        raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
    if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        raise ValueError("GR read weights must be BF16")
    if w_down.device != w_up.device:
        raise ValueError("weights must share a device")
    interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(
        1, 0, 2, 4, 3, 5, 6
    ).contiguous().reshape(K, R)
    return preshuffle_weight(w_down), preshuffle_weight(interleaved)
