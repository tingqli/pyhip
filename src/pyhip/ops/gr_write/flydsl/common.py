# SPDX-License-Identifier: MIT
"""Fixed dimensions, prepared weight layouts and launch rules for GR write."""

import torch

C, H = 4, 2560
K = C * H
EPS = 1e-6

# Gate: one workgroup = 4 K-slice waves of a 4-row quad. Each wave owns 2560 columns,
# moved as 512-column groups (one contiguous 1 KiB load per row) through LDS into
# MFMA 4x4x4 BF16 operands.
GATE_SLICES = 4
GATE_GROUP = 512
GATE_SLICE_COLUMNS = K // GATE_SLICES
GATE_GROUPS = GATE_SLICE_COLUMNS // GATE_GROUP
GATE_PREFETCH = 3

# Apply: one workgroup = 4 waves of one row, one wave per SIMD. Lane (wave w, lane t) owns 5
# chunks of 8 columns: chunk b < 4 is branch b, columns 512w + 8t; chunk 4 is branch w,
# columns 2048 + 8t. The per-branch RMSNorm sum therefore crosses waves.
APPLY_WAVES = 4
APPLY_CHUNKS = 5

# Persistent workgroups per CU, calibrated on MI308X / gfx942 / 80 CU.
GATE_WORKGROUPS_PER_CU = 2
APPLY_WORKGROUPS_PER_CU = 3


def validate_rows(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError('expected GR write rows to be a nonnegative integer')


def launch_grids(rows, compute_units):
    """Return (gate, apply) persistent grid sizes; both kernels take rows at runtime."""
    validate_rows(rows)
    quads = (rows + 3) // 4
    gate = max(1, min(GATE_WORKGROUPS_PER_CU * compute_units, quads))
    apply = max(1, min(APPLY_WORKGROUPS_PER_CU * compute_units, rows))
    return gate, apply


def workspace_words(rows):
    """FP32 words of the per-call workspace: gate values [rows, 4] plus the apply row counter."""
    return rows * C + 4


@torch.no_grad()
def prepare_weights(inject_weight, norm_weight):
    """Pack the GR write weights once per layer.

    inject_weight: BF16 [4, 10240] (the combine gate weight).
    norm_weight: BF16 [10240] (the next per-branch Gemma RMSNorm weight).

    Returns (packed_inject, packed_norm):
    - packed_inject: BF16 [10240 * 4], MFMA A-operand fragments ordered
      [slice][group][t][lane = 4 * block + c][8 columns], so each wave loads
      its fragments as contiguous 1 KiB rows.
    - packed_norm: FP32 [10240], (1 + w) computed in FP32 and ordered
      [wave][chunk][half][lane][4], matching the apply chunk ownership.
    """
    if not isinstance(inject_weight, torch.Tensor) or inject_weight.shape != (C, K) \
            or inject_weight.dtype != torch.bfloat16:
        raise ValueError('inject_weight must be a BF16 tensor with shape [4, 10240]')
    if not isinstance(norm_weight, torch.Tensor) or norm_weight.shape != (K,) \
            or norm_weight.dtype != torch.bfloat16:
        raise ValueError('norm_weight must be a BF16 tensor with shape [10240]')
    if inject_weight.device != norm_weight.device:
        raise ValueError('inject_weight and norm_weight must be on the same device')
    # column = s * 2560 + g * 512 + t * 128 + block * 8 + e
    packed_inject = inject_weight.view(C, GATE_SLICES, GATE_GROUPS, 4, 16, 8)
    packed_inject = packed_inject.permute(1, 2, 3, 4, 0, 5).contiguous().view(-1)
    # chunk b < 4: branch b, column 512 * wave + 8 * lane + 4 * half + e;
    # chunk 4: branch wave, column 2048 + 8 * lane + 4 * half + e.
    gain = (1.0 + norm_weight.float()).view(C, APPLY_CHUNKS, 64, 2, 4)
    main = gain[:, :APPLY_WAVES].permute(1, 0, 3, 2, 4)        # [wave][branch][half][lane][4]
    tail = gain[:, APPLY_WAVES].permute(0, 2, 1, 3).unsqueeze(1)  # [wave = branch][1][half][lane][4]
    packed_norm = torch.cat((main, tail), dim=1).contiguous().view(-1)
    return packed_inject, packed_norm
