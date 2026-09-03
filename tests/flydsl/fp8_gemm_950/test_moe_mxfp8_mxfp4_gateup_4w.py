"""Four-wave MXFP8 x MXFP4 MoE gate/up GEMM.

The kernel writes the unfused gate and up projections to
``[tokens, topk, 2 * intermediate_size]``.  This file starts with a focused
buffer-resource routing probe; the GEMM implementation below reuses the same
routing contract.
"""

import argparse
import csv
import os
import time

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, Float32, Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf


SORT_BLOCK_M = 256
TOKEN_MASK = 0xFFFFFF
A_INPUT_SCALE = 0.33
B_INPUT_SCALE = 0.2


def div_up(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def validate_case_parameters(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
) -> None:
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if intermediate_size <= 0 or intermediate_size % 128 != 0:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if hidden_size < 512 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be a multiple of 256 and at least 512")
    if not 0 < topk < 256:
        raise ValueError("topk must be in [1, 255]")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if topk > num_experts:
        raise ValueError("topk must not exceed num_experts")


def make_balanced_routing(
    tokens: int,
    topk: int,
    experts: int,
    model_dim: int,
    *,
    device: str,
):
    """Generate balanced top-k choices and sort them into expert-major blocks."""
    from aiter.fused_moe import moe_sorting

    assert 0 < tokens <= TOKEN_MASK
    assert 0 < topk < 256
    assert topk <= experts
    num_routes = tokens * topk
    repetitions = div_up(num_routes, experts)
    topk_ids_1d = torch.empty(
        (repetitions, experts), device=device, dtype=torch.int32
    )
    topk_ids_1d[:] = torch.randperm(experts, device=device, dtype=torch.int32)
    topk_ids = topk_ids_1d.reshape(-1)[:num_routes].reshape(tokens, topk)
    topk_weights = torch.randn(
        (tokens, topk), device=device, dtype=torch.float32
    )

    counts = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64), minlength=experts
    )
    assert int(counts.max() - counts.min()) <= 1
    (
        # Expert-major packed IDs: (topk slot << 24) | token ID, with padding. sorted_ids.numele() % SORT_BLOCK_M == 0
        sorted_ids,
        # Route weights reordered to match sorted_ids. sorted_weights.numele() % SORT_BLOCK_M == 0
        sorted_weights,
        # sorted_expert_ids.numele() = sorted_ids.numele() // SORT_BLOCK_M
        sorted_expert_ids,
        # [0]: route count after padding; [1]: original token count.
        num_valid_ids,
        # Internal workspace allocated by moe_sorting; unused here.
        _,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        SORT_BLOCK_M,
        None,
        None,
        0,
    )
    num_sorted = int(num_valid_ids[0].item())
    assert num_sorted % SORT_BLOCK_M == 0
    num_expert_blocks = num_sorted // SORT_BLOCK_M
    return (
        topk_ids,
        topk_weights,
        sorted_ids[:num_sorted].contiguous(),
        sorted_weights[:num_sorted].contiguous(),
        sorted_expert_ids[:num_expert_blocks].contiguous(),
        num_valid_ids,
    )


def _load_mx_helpers():
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    return per_1x32_mx_quant_hip, dtypes, e8m0_to_f32, mxfp4_to_f32


def _permute_scale(
    scale: torch.Tensor, *, padded_rows: int | None = None
) -> torch.Tensor:
    """Match the packed E8M0 scale layout consumed by the MFMA selectors."""
    scale_u8 = scale.view(torch.uint8)
    rows, groups = scale_u8.shape
    if padded_rows is None:
        padded_rows = div_up(rows, 256) * 256
    if padded_rows < rows or padded_rows % 256 != 0:
        raise ValueError(
            "padded_rows must cover all scale rows and be a multiple of 256"
        )
    if padded_rows != rows:
        padded = torch.full(
            (padded_rows, groups),
            127,
            device=scale.device,
            dtype=torch.uint8,
        )
        padded[:rows].copy_(scale_u8)
        scale_u8 = padded
        rows = padded_rows
    return (
        # 32是2x2 wave排列，两个wave cover 32行， 4 is rep, 
        scale_u8.view(rows // 128, 4, 32, groups)
        .permute(3, 0, 2, 1)
        .contiguous()
        .view(torch.int32)
    )


def _convert_aiter_moe_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert AIter's routed MX scale swizzle to this kernel's layout."""
    scale_u8 = scale.view(torch.uint8)
    rows, groups = scale_u8.shape
    if rows % 128 != 0 or groups % 8 != 0:
        raise ValueError(
            "AIter MoE scale requires rows divisible by 128 and groups by 8"
        )
    return (
        scale_u8.view(rows // 128, 4, groups // 8, 4, 16, 2, 2)
        .permute(2, 5, 3, 0, 6, 4, 1)
        .contiguous()
        .view(torch.int32)
    )


def prepare_moe_inputs(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
):
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort

    per_1x32_mx_quant_hip, dtypes, _, _ = _load_mx_helpers()
    (
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        expert_ids,
        num_valid_ids,
    ) = make_balanced_routing(
        tokens, topk, num_experts, hidden_size, device="cuda"
    )

    assert hidden_size % 32 == 0, f"hidden_size must be a multiple of 32, but got {hidden_size}"
    a_source = torch.randn(
        (tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    ) * A_INPUT_SCALE
    # Keep token-order scales for the dequantized reference.
    a_reference, scale_a_raw = per_1x32_mx_quant_hip(
        a_source,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    # Set up A scale routing data.
    # Production stage1 path: quantize A and emit routed, AIter-swizzled scales.
    # R: routing token ID as routing table. 
    # G: quantization groups. hidden_state // 32
    # [R, G] ->[R//32, 2r1, 16r0, G//8, 2g1, 4g0] would be permuted to 
    # [R//32, G//8, 4g0,16r0,2g1,2r1] 
    
    # scale_a_aiter is padded to align with the SORTING_BLOCK_M,
    a, scale_a_aiter = fused_dynamic_mxfp8_quant_moe_sort(
        a_source,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        topk=topk,
        block_size=SORT_BLOCK_M,
        sorted_weights=sorted_weights,
    )
    torch.testing.assert_close(a, a_reference, rtol=0, atol=0)
    del a_source

    output_size = 2 * intermediate_size
    scale_a_padded_rows = sorted_ids.numel()
    scale_b_rows = num_experts * output_size
    weight, scale_b_raw = per_1x32_mx_quant_hip(
        torch.randn(
            (scale_b_rows, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        ) * B_INPUT_SCALE,
        quant_dtype=dtypes.fp4x2,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    weight = weight.view(num_experts, output_size, hidden_size // 2)
    scale_b_raw = scale_b_raw.view(
        num_experts, output_size, hidden_size // 32
    )
    # scale B need to padding to align with BLOCK_N, because N is not the outter most dimension after B scale is permuted.
    scale_b_rows_per_expert = div_up(output_size, 256) * 256
    scale_b_padded_rows = num_experts * scale_b_rows_per_expert
    ## gaps with MOE in aiter.
    # For mxfp8 and mxfp4,  aiter fused MOE would sort the A scale as sorted_id table. [R, G], `R` means routed token ID. `G` means quantization groups. 
    # R would padded to 32 alignment, G would be padded to 8 aligned.
    # Also after A scale is sorted, [R, G] would be permuted    

    # [R, G] =  [R//32, 2r1, 16r0, G//8, 2g1, 4g0] would be permuted to 

    # [R//32, G//8, 4g0,16r0,2g1,2r1] 

    # current gemmA8w4 perfer the layout:


    # ```
    #         scale_u8.view(r // 128, 4r1, 32r0, groups).permute(3, 0, 2, 1)
    # ```

    # [R//128, 4r1, 32r0, G] -> [G, R//128, 32r0, 4r1]

    # 把 这个是fused_dynamic_mxfp8_quant_moe_sort的 preshuffle之后的scale转化成test_moe_mxfp8_mxfp4_gateup_4w.py里面的A

    # 1. 这个是fused_dynamic_mxfp8_quant_moe_sort scale的layout [R//32, G//8,4g0 ,16r0,2g1,2r1]  
    # view as  [R//128, 4r2, G//8,4g0 ,16r0,2g1,2r1],


    # 2. permute:
    # [R//128, 4r2, G//8,4g0 ,16r0,2g1,2r1] -[0, 1, 2, 3, 4, 5,6]    -> [G, R//128, 32r0, 4r1]

    # permute to 
    # []
    # [G//8, 4g0, 2g1, R//128, 2r1, 16r0, 4r2]


    # permute:  [2， 5， 3， 0， 6， 4， 1]
    # reshape:

    scale_a = _convert_aiter_moe_scale(scale_a_aiter)
    scale_b_padded = torch.full(
        (num_experts, scale_b_rows_per_expert, hidden_size // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    scale_b_padded[:, :output_size].copy_(scale_b_raw.view(torch.uint8))
    scale_b = _permute_scale(
        scale_b_padded.view(scale_b_padded_rows, hidden_size // 32)
    )
    assert a.view(torch.uint8).numel() == tokens * hidden_size
    assert weight.view(torch.uint8).numel() == (
        num_experts * output_size * hidden_size // 2
    )
    assert scale_a.view(torch.uint8).numel() == (
        scale_a_padded_rows * (hidden_size // 32)
    )
    assert scale_b.view(torch.uint8).numel() == (
        scale_b_padded_rows * (hidden_size // 32)
    )

    return {
        "a": a,
        "weight": weight,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "scale_a_raw": scale_a_raw,
        "scale_b_raw": scale_b_raw,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "scale_a_padded_rows": scale_a_padded_rows,
        "scale_b_rows_per_expert": scale_b_rows_per_expert,
        "scale_b_padded_rows": scale_b_padded_rows,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "expert_ids": expert_ids,
        "num_valid_ids": num_valid_ids,
    }


def moe_reference(
    inputs,
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
):
    _, _, e8m0_to_f32, mxfp4_to_f32 = _load_mx_helpers()
    scale_a = e8m0_to_f32(inputs["scale_a_raw"]).repeat_interleave(32, dim=1)
    a_dequant = inputs["a"].float() * scale_a
    output_size = 2 * intermediate_size
    reference = torch.zeros(
        (tokens, topk, output_size), device="cuda", dtype=torch.bfloat16
    )
    sorted_blocks = inputs["sorted_ids"].view(-1, SORT_BLOCK_M)
    for expert in range(num_experts):
        block_mask = inputs["expert_ids"] == expert
        routed = sorted_blocks[block_mask].reshape(-1)
        token_ids = (routed & TOKEN_MASK).to(torch.int64)
        slot_ids = ((routed >> 24) & 0xFF).to(torch.int64)
        valid = (token_ids < tokens) & (slot_ids < topk)
        token_ids = token_ids[valid]
        slot_ids = slot_ids[valid]
        weight_scale = e8m0_to_f32(
            inputs["scale_b_raw"][expert]
        ).repeat_interleave(32, dim=1)
        weight_dequant = mxfp4_to_f32(inputs["weight"][expert]) * weight_scale
        projected = a_dequant[token_ids] @ weight_dequant.t()
        reference[token_ids, slot_ids] = projected.to(torch.bfloat16)
    return reference


def encode_waitcnt_950(vmcnt: int = 63, expcnt: int = 7, lgkmcnt: int = 63) -> int:
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def waitvmcnt_barrier(vmcnt: int) -> None:
    rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
    rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
    rocdl.s_barrier()


def hot_loop_scheduler_mainloop(group_id, vmem_ops, dsrd_ops):
    total_mfmas = 16
    scale_sched_late = _env_flag("SCALE_SCHED_LATE", "1")
    scale_dsrd_pos = int(os.environ.get("SCALE_DSRD_POS", "13"))
    scale_vmem_pos = int(os.environ.get("SCALE_VMEM_POS", "7"))
    base_dsrd_ops = 8 if scale_sched_late and dsrd_ops == 9 else dsrd_ops
    base_vmem_ops = 4 if scale_sched_late and vmem_ops == 5 else vmem_ops
    prev_dsrd = 0
    prev_vmem = 0
    for i in range_constexpr(total_mfmas):
        cur_dsrd = ((i + 3) * base_dsrd_ops + total_mfmas - 1) // total_mfmas
        cur_dsrd = min(cur_dsrd, base_dsrd_ops)
        if const_expr(scale_sched_late and dsrd_ops == 9 and i >= scale_dsrd_pos):
            cur_dsrd += 1
        if const_expr(cur_dsrd > prev_dsrd):
            rocdl.sched_group_barrier(
                rocdl.mask_dsrd, cur_dsrd - prev_dsrd, group_id
            )
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
        cur_vmem = ((i + 1) * base_vmem_ops + total_mfmas - 1) // total_mfmas
        if const_expr(scale_sched_late and vmem_ops == 5 and i >= scale_vmem_pos):
            cur_vmem += 1
        if const_expr(cur_vmem > prev_vmem):
            rocdl.sched_group_barrier(
                rocdl.mask_vmem_rd, cur_vmem - prev_vmem, group_id
            )
        prev_dsrd = cur_dsrd
        prev_vmem = cur_vmem


def compile_moe_gateup_4w(
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    *,
    b_lds_swizzle: bool = False,
    xcd_swizzle: bool = False,
    group_size_m: int = 1,
):
    """Build the padding-LDS four-wave raw gate/up projection kernel."""
    block_m = SORT_BLOCK_M // 2
    block_n = 128
    block_k = 128
    output_size = 2 * intermediate_size
    scale_b_rows_per_expert = div_up(output_size, 256) * 256
    scale_b_padded_rows = num_experts * scale_b_rows_per_expert
    num_n_tiles = intermediate_size // block_n
    assert intermediate_size % block_n == 0
    assert hidden_size % block_k == 0
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")

    def _get_pids_950(pid, num_pid_m, grid_mn, num_xcds, group_m):
        num_pid_n = num_n_tiles
        if const_expr(num_xcds != 1):
            pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
            tall_xcds = grid_mn % num_xcds
            tall_xcds = (tall_xcds == 0).select(num_xcds, tall_xcds)
            xcd = pid % num_xcds
            local_pid = pid // num_xcds
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = (
                    tall_xcds * pids_per_xcd
                    + (xcd - tall_xcds) * (pids_per_xcd - 1)
                    + local_pid
                )
        if const_expr(group_m == 1):
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = group_m * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * group_m
            remaining_pid_m = num_pid_m - first_pid_m
            group_m_actual = (remaining_pid_m < group_m).select(
                remaining_pid_m, group_m
            )
            pid_m = first_pid_m + (
                (pid % num_pid_in_group) % group_m_actual
            )
            pid_n = (pid % num_pid_in_group) // group_m_actual
        return pid_m, pid_n

    get_pids_950 = ASTRewriter.transform(_get_pids_950)

    # activateion data type fp8
    element_type = Float8E4M3FN
    # weight type fp4
    weight_type = Float4E2M1FN

    # A8w4 with scale A采用的是单padding[1024， 32]
    a_group8 = 8 * block_k + 32
    a_group16 = 2 * a_group8
    a_lds_elems = (block_m // 8) * a_group8
    
    # Padding keeps each 2048-element block contiguous; swizzle instead XORs
    # 16-byte slots within each 8-row group to avoid B LDS bank conflicts.
    b_group16 = 16 * block_k + 64
    b_lds_elems = (
        block_n * block_k
        if b_lds_swizzle
        else (block_n // 16) * b_group16
    )

    @fx.struct
    class LDS:
        a_top0: fx.Array[element_type, a_lds_elems, 16]
        a_top1: fx.Array[element_type, a_lds_elems, 16]
        a_bottom0: fx.Array[element_type, a_lds_elems, 16]
        a_bottom1: fx.Array[element_type, a_lds_elems, 16]
        b_gate0: fx.Array[weight_type, b_lds_elems, 16]
        b_gate1: fx.Array[weight_type, b_lds_elems, 16]
        b_up0: fx.Array[weight_type, b_lds_elems, 16]
        b_up1: fx.Array[weight_type, b_lds_elems, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    
    
    # args = (
    #     inputs["a"].view(torch.int8).view(-1),
    #     inputs["weight"].view(torch.int8).view(-1),
    #     inputs["scale_a"],
    #     inputs["scale_b"],
    #     output.view(-1),
    #     inputs["sorted_ids"],
    #     inputs["expert_ids"],
    #     inputs["num_valid_ids"],
    #     tokens,
    #     inputs["expert_ids"].numel(),
    #     torch.cuda.current_stream(),
    # )
    
    def moe_gateup_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_c: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        num_tokens: int,
        num_expert_blocks: int,
    ):
        tid = fx.thread_idx.x
        num_expert_blocks_i32 = fx.Int32(num_expert_blocks)
        if const_expr(xcd_swizzle):
            expert_block, n_tile = get_pids_950(
                fx.block_idx.x,
                num_expert_blocks_i32,
                fx.grid_dim.x,
                8,
                group_size_m,
            )
        else:
            expert_block = fx.block_idx.x // num_n_tiles
            n_tile = fx.block_idx.x % num_n_tiles
        n_tile_i32 = fx.Int32(n_tile)
        expert_block_i32 = fx.Int32(expert_block)
        num_tokens_i32 = fx.Int32(num_tokens)

        a_tensor_bytes = num_tokens_i32 * fx.Int32(hidden_size)
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a,
            num_records_bytes=arith._to_raw(a_tensor_bytes),
        )
        c_rsrc = buffer_ops.create_buffer_resource(
            arg_c,
            num_records_bytes=arith._to_raw(
                num_tokens_i32 * fx.Int32(topk * output_size * 2)
            ),
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_ids,
            num_records_bytes=arith._to_raw(
                num_expert_blocks_i32 * fx.Int32(SORT_BLOCK_M * 4)
            ),
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            arg_expert_ids,
            num_records_bytes=arith._to_raw(num_expert_blocks_i32 * fx.Int32(4)),
        )
        valid_rsrc = buffer_ops.create_buffer_resource(
            arg_num_valid_ids,
            num_records_bytes=arith._to_raw(fx.Int32(4)),
        )
        expert_i32 = fx.Int32(
            buffer_ops.buffer_load(
                expert_rsrc, expert_block, vec_width=1, dtype=T.i32
            )
        )
        expert_i32 = fx.Int32(
            rocdl.readfirstlane(T.i32, arith._to_raw(expert_i32))
        )
        expert_byte_offset = (
            arith.index_cast(T.index, expert_i32)
            * arith.constant(output_size * hidden_size // 2, index=True)
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b,
            num_records_bytes=output_size * hidden_size // 2,
            base_byte_offset=expert_byte_offset,
        )
        scale_a_padded_rows = (
            num_expert_blocks_i32 * fx.Int32(SORT_BLOCK_M)
        )
        scale_a_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a,
            num_records_bytes=arith._to_raw(
                scale_a_padded_rows * fx.Int32(hidden_size // 32)
            ),
        )
        scale_b_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b,
            num_records_bytes=arith._to_raw(
                fx.Int32(scale_b_padded_rows * (hidden_size // 32))
            ),
        )

        num_valid_i32 = fx.Int32(
            buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )

        # mma_atom 生成的tile 可以用来slice A, B,
        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, weight_type, element_type)
        )
        scale_atoms = {
            (n0, m0): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    weight_type,
                    element_type,
                    opsel_a=n0,
                    opsel_b=m0,
                )
            )
            for n0 in range_constexpr(4)
            for m0 in range_constexpr(4)
        }
        #k_permutation决定的是同一行中的thread, 如何分配K，每条lane读32个K， 这32个K是否是连续的，spec的描述是不连续的。
        k_permutation = fx.make_layout(((16, 2), 4), ((1, 64), 16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((2, 2, 1), (1, 2, 0)),
            (None, None, k_permutation),
        )
        thread_mma = tiled_mma.thr_slice(tid)

        lds = fx.SharedAllocator().allocate(LDS).peek()
        write_layout_a = fx.make_layout(
            ((8, 2, block_m // 16), block_k),
            ((block_k, a_group8, a_group16), 1),
        )
        read_layout_a = fx.make_layout(
            ((2, block_m // 16, 8), (32, block_k // 32)),
            ((a_group8, a_group16, block_k), (1, 32)),
        )
        if const_expr(b_lds_swizzle):
            read_layout_b = fx.make_ordered_layout(
                (block_n, block_k), (1, 0)
            )
        else:
            read_layout_b = fx.make_layout(
                ((16, block_n // 16), block_k),
                ((block_k, b_group16), 1),
            )

        a_top_read = [
            fx.make_view(ptr, read_layout_a)
            for ptr in (lds.a_top0.ptr, lds.a_top1.ptr)
        ]
        a_bottom_read = [
            fx.make_view(ptr, read_layout_a)
            for ptr in (lds.a_bottom0.ptr, lds.a_bottom1.ptr)
        ]
        b_gate_read = [
            fx.make_view(ptr, read_layout_b)
            for ptr in (lds.b_gate0.ptr, lds.b_gate1.ptr)
        ]
        b_up_read = [
            fx.make_view(ptr, read_layout_b)
            for ptr in (lds.b_up0.ptr, lds.b_up1.ptr)
        ]

        copy_a_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
        copy_a = fx.make_tiled_copy_B(copy_a_atom, tiled_mma).get_slice(tid)
        a_top_source = [copy_a.partition_S(view) for view in a_top_read]
        a_bottom_source = [copy_a.partition_S(view) for view in a_bottom_read]
        frag_a_top = thread_mma.make_fragment_B(a_top_read[0])
        frag_a_bottom = thread_mma.make_fragment_B(a_bottom_read[0])
        frag_a_top_dest = copy_a.retile(frag_a_top)
        frag_a_bottom_dest = copy_a.retile(frag_a_bottom)

        frag_b_gate = fx.make_rmem_tensor(16, Int32)
        frag_b_up = fx.make_rmem_tensor(16, Int32)


        lane_id = tid % 64
        wave_id = tid // 64
        wave_id_uniform = fx.Int32(
            rocdl.readfirstlane(T.i32, arith._to_raw(wave_id))
        )
        wave_a_lds_base = (
            (wave_id_uniform % 2) * a_group8
            + (wave_id_uniform // 2) * a_group16
        )

        def lds_root(ptr):
            return _fly.extract_aligned_pointer_as_index(
                ir.Type.parse("!llvm.ptr<3>"),
                arith._to_raw(fx.make_view(ptr, fx.make_layout(1, 1))),
            )

        def make_a_dma_ptr(ptr, copy_round: int):
            return buffer_ops.get_element_ptr(
                lds_root(ptr),
                byte_offset=wave_a_lds_base + copy_round * 4 * a_group8,
                elem_type=T.i8,
            )

        a_top_dma_ptrs = [
            [make_a_dma_ptr(ptr, copy_round) for copy_round in range_constexpr(4)]
            for ptr in (lds.a_top0.ptr, lds.a_top1.ptr)
        ]
        a_bottom_dma_ptrs = [
            [make_a_dma_ptr(ptr, copy_round) for copy_round in range_constexpr(4)]
            for ptr in (lds.a_bottom0.ptr, lds.a_bottom1.ptr)
        ]
        mask24 = arith.constant(TOKEN_MASK, type=T.i32)

        def load_a_token_ids(row_half: int):
            token_ids = []
            for copy_round in range_constexpr(4):
                row_local = (
                    row_half * block_m
                    + wave_id_uniform
                    + (lane_id // 8) * 16
                    + copy_round * 4
                )
                sorted_row = (
                    expert_block_i32 * SORT_BLOCK_M + fx.Int32(row_local)
                )
                fused_id = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_row, vec_width=1, dtype=T.i32
                )
                token_ids.append(arith.andi(fused_id, mask24))
            return token_ids

        a_top_token_ids = load_a_token_ids(0)
        a_bottom_token_ids = load_a_token_ids(1)

        def raw_a_gather_g2s(kk, ptrs, token_ids):
            for copy_round in range_constexpr(4):
                global_byte = (
                    fx.Int32(token_ids[copy_round]) * hidden_size
                    + kk * block_k
                    + fx.Int32((lane_id % 8) * 16)
                )
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    ptrs[copy_round],
                    fx.Int32(16),
                    global_byte,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def raw_b_mxfp4_g2s(kk, ptr, row_tile):
            root = lds_root(ptr)
            for copy_round in range_constexpr(2):
                if const_expr(b_lds_swizzle):
                    physical_slot = tid + copy_round * 256
                    logical_slot = physical_slot ^ (
                        (physical_slot >> 3) & 1
                    )
                    row = (logical_slot // 32) * 8 + logical_slot % 8
                    col_byte = ((logical_slot % 32) // 8) * 16
                    lds_ptr = buffer_ops.get_element_ptr(
                        root,
                        byte_offset=(
                            wave_id_uniform * 64 + copy_round * 256
                        )
                        * 16,
                        elem_type=T.i8,
                    )
                else:
                    chunk = wave_id_uniform + copy_round * 4
                    row = chunk * 16 + lane_id // 4
                    col_byte = (lane_id % 4) * 16
                    lds_ptr = buffer_ops.get_element_ptr(
                        root,
                        byte_offset=chunk * (b_group16 // 2),
                        elem_type=T.i8,
                    )
                global_row = row_tile * block_n + fx.Int32(row)
                global_byte = (
                    global_row * (hidden_size // 2)
                    + kk * (block_k // 2)
                    + fx.Int32(col_byte)
                )
                rocdl.raw_ptr_buffer_load_lds(
                    b_rsrc,
                    lds_ptr,
                    fx.Int32(16),
                    global_byte,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                )

        scale_lane_id = tid % 64
        scale_wave_id = wave_id_uniform

        def load_scale_dword(rsrc, kk, row_tile, rows, is_a: bool):
            wave_half = scale_wave_id // 2 if is_a else scale_wave_id % 2
            scale_row = scale_lane_id % 16 + wave_half * 16
            scale_group = scale_lane_id // 16
            dword_offset = (
                kk * rows
                + fx.Int32(scale_group) * (rows // 4)
                + row_tile * 32
                + fx.Int32(scale_row)
            )
            return fx.Int32(
                buffer_ops.buffer_load(
                    rsrc, dword_offset, vec_width=1, dtype=T.i32
                )
            )

        def load_b_fragment(source, destination):
            wave_n = wave_id % 2
            values = []
            for n0 in range_constexpr(4):
                row = (n0 * 2 + wave_n) * 16 + lane_id % 16
                col_byte = (lane_id // 16) * 16
                if const_expr(b_lds_swizzle):
                    lds_byte = (
                        (row // 8) * (8 * block_k // 2)
                        + (row % 8) * 16
                        + (col_byte // 16) * (8 * 16)
                    )
                    lds_byte = lds_byte ^ (((lds_byte >> 7) & 1) << 4)
                else:
                    lds_byte = (
                        (row // 16) * (b_group16 // 2)
                        + (row % 16) * (block_k // 2)
                        + col_byte
                    )
                ptr = fx.add_offset(
                    fx.recast_iter(fx.Uint8, fx.get_iter(source)),
                    fx.make_int_tuple(lds_byte),
                )
                packed = (
                    fx.make_view(ptr, fx.make_layout(16, 1))
                    .load()
                    .bitcast(Int32)
                )
                for word in range_constexpr(4):
                    values.append(packed[word])
            destination.store(Vec.from_elements(values, Int32))

        def do_gemm(c_frag, b_frag, a_frag, scale_a_frag, scale_b_frag):
            c_value = c_frag.load().ir_value()
            b_value = vector.bitcast(T.vec(64, T.i8), b_frag.load().ir_value())
            a_value = vector.bitcast(T.vec(128, T.i8), a_frag.load().ir_value())
            scale_a = Vec(scale_a_frag.load())[0]
            scale_b = Vec(scale_b_frag.load())[0]
            for n0 in range_constexpr(4):
                for m0 in range_constexpr(4):
                    c_offset = (m0 * 4 + n0) * 4
                    c_sub = vector.extract_strided_slice(
                        T.vec(4, T.f32),
                        c_value,
                        offsets=[c_offset],
                        sizes=[4],
                        strides=[1],
                    )
                    b_sub = vector.extract_strided_slice(
                        T.vec(16, T.i8),
                        b_value,
                        offsets=[n0 * 16],
                        sizes=[16],
                        strides=[1],
                    )
                    b_sub = vector.bitcast(T.vec(4, T.i32), b_sub)
                    a_sub = vector.extract_strided_slice(
                        T.vec(32, T.i8),
                        a_value,
                        offsets=[m0 * 32],
                        sizes=[32],
                        strides=[1],
                    )
                    scaled_atom = fx.atom_set_value(
                        scale_atoms[(n0, m0)], "scale_a", scale_b
                    )
                    scaled_atom = fx.atom_set_value(
                        scaled_atom, "scale_b", scale_a
                    )
                    c_sub = _fly.mma_atom_call_ssa(
                        [T.vec(4, T.f32)], scaled_atom, b_sub, a_sub, c_sub
                    )
                    c_value = vector.insert_strided_slice(
                        c_sub, c_value, [c_offset], [1]
                    )
            c_frag.store(c_value)

        c_layout_tile = fx.make_rmem_tensor(
            fx.make_ordered_layout((block_n, block_m), (1, 0)), Float32
        )
        frag_c_tl = thread_mma.make_fragment_C(c_layout_tile)
        frag_c_tr = thread_mma.make_fragment_C(c_layout_tile)
        frag_c_bl = thread_mma.make_fragment_C(c_layout_tile)
        frag_c_br = thread_mma.make_fragment_C(c_layout_tile)

        gate_row_tile = n_tile_i32
        up_row_tile = (
            intermediate_size // block_n + n_tile_i32
        )
        expert_scale_row_tile = expert_i32 * (
            scale_b_rows_per_expert // block_n
        )
        gate_scale_row_tile = expert_scale_row_tile + n_tile_i32
        up_scale_row_tile = (
            expert_scale_row_tile
            + intermediate_size // block_n
            + n_tile_i32
        )
        a_top_scale_tile = expert_block_i32 * 2
        a_bottom_scale_tile = a_top_scale_tile + 1
        num_k_tiles = hidden_size // block_k
        assert num_k_tiles >= 4 and num_k_tiles % 2 == 0

        scale_a_top_frag = fx.make_rmem_tensor(1, Int32)
        scale_a_bottom_frag = fx.make_rmem_tensor(1, Int32)
        scale_b_gate_frag = fx.make_rmem_tensor(1, Int32)
        scale_b_up_frag = fx.make_rmem_tensor(1, Int32)
        scale_a_top_g2r = [
            fx.make_rmem_tensor(1, Int32) for _ in range_constexpr(2)
        ]
        scale_a_bottom_g2r = [
            fx.make_rmem_tensor(1, Int32) for _ in range_constexpr(2)
        ]
        scale_b_gate_g2r = [
            fx.make_rmem_tensor(1, Int32) for _ in range_constexpr(2)
        ]
        scale_b_up_g2r = [
            fx.make_rmem_tensor(1, Int32) for _ in range_constexpr(2)
        ]

        def load_scale_g2r(destination, rsrc, kk, row_tile, rows, is_a):
            destination.store(
                Vec.from_elements(
                    [load_scale_dword(rsrc, kk, row_tile, rows, is_a)],
                    Int32,
                )
            )

        def load_a(source, destination):
            fx.copy(copy_a_atom, source, destination)

        def load_b(source, destination):
            load_b_fragment(source, destination)

        def do_g2s(kk, buffer_index: int):
            ki = fx.Int32(kk)
            raw_b_mxfp4_g2s(
                ki,
                (lds.b_gate0.ptr, lds.b_gate1.ptr)[buffer_index],
                gate_row_tile,
            )
            rocdl.sched_barrier(0)
            load_scale_g2r(
                scale_b_gate_g2r[buffer_index],
                scale_b_rsrc,
                ki,
                gate_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            rocdl.sched_barrier(0)
            raw_a_gather_g2s(
                ki, a_top_dma_ptrs[buffer_index], a_top_token_ids
            )
            rocdl.sched_barrier(0)
            load_scale_g2r(
                scale_a_top_g2r[buffer_index],
                scale_a_rsrc,
                ki,
                a_top_scale_tile,
                scale_a_padded_rows,
                True,
            )
            rocdl.sched_barrier(0)
            raw_a_gather_g2s(
                ki, a_bottom_dma_ptrs[buffer_index], a_bottom_token_ids
            )
            rocdl.sched_barrier(0)
            load_scale_g2r(
                scale_a_bottom_g2r[buffer_index],
                scale_a_rsrc,
                ki,
                a_bottom_scale_tile,
                scale_a_padded_rows,
                True,
            )
            rocdl.sched_barrier(0)
            raw_b_mxfp4_g2s(
                ki,
                (lds.b_up0.ptr, lds.b_up1.ptr)[buffer_index],
                up_row_tile,
            )
            rocdl.sched_barrier(0)
            load_scale_g2r(
                scale_b_up_g2r[buffer_index],
                scale_b_rsrc,
                ki,
                up_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            rocdl.sched_barrier(0)

        a_vmem = (
            block_m * block_k * element_type.width // 8
        ) // (256 * 16)
        b_vmem = (
            block_n * block_k * weight_type.width // 8
        ) // (256 * 16)
        a_phase_vmem = a_vmem + 1
        b_phase_vmem = b_vmem + 1
        wait_ab = 2 * a_phase_vmem + 3 * b_phase_vmem
        wait_ba = 3 * a_phase_vmem + 2 * b_phase_vmem
        a_dsrd = frag_a_top.load().numel * element_type.width // 8 // 16
        b_dsrd = 4

        do_g2s(0, 0)
        do_g2s(1, 1)
        waitvmcnt_barrier(3 * (a_phase_vmem + b_phase_vmem))
        load_b(b_gate_read[0], frag_b_gate)
        load_a(a_top_source[0], frag_a_top_dest)
        scale_b_gate_frag.store(scale_b_gate_g2r[0].load())
        scale_a_top_frag.store(scale_a_top_g2r[0].load())
        rocdl.sched_barrier(0)

        frag_c_tl.fill(0)
        frag_c_tr.fill(0)
        frag_c_bl.fill(0)
        frag_c_br.fill(0)
        rocdl.sched_barrier(0)
        accumulators = [
            frag_c_tl.load(),
            frag_c_tr.load(),
            frag_c_bl.load(),
            frag_c_br.load(),
        ]

        for kk_index, states in range(0, num_k_tiles - 2, 2, init=accumulators):
            frag_c_tl.store(states[0])
            frag_c_tr.store(states[1])
            frag_c_bl.store(states[2])
            frag_c_br.store(states[3])
            kk = fx.Int32(kk_index)

            do_gemm(
                frag_c_tl,
                frag_b_gate,
                frag_a_top,
                scale_a_top_frag,
                scale_b_gate_frag,
            )
            waitvmcnt_barrier(wait_ab)
            load_a(a_bottom_source[0], frag_a_bottom_dest)
            scale_a_bottom_frag.store(scale_a_bottom_g2r[0].load())
            raw_b_mxfp4_g2s(kk + 2, lds.b_gate0.ptr, gate_row_tile)
            load_scale_g2r(
                scale_b_gate_g2r[0],
                scale_b_rsrc,
                kk + 2,
                gate_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            hot_loop_scheduler_mainloop(0, b_phase_vmem, a_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_bl,
                frag_b_gate,
                frag_a_bottom,
                scale_a_bottom_frag,
                scale_b_gate_frag,
            )
            waitvmcnt_barrier(wait_ab)
            load_b(b_up_read[0], frag_b_up)
            scale_b_up_frag.store(scale_b_up_g2r[0].load())
            raw_a_gather_g2s(kk + 2, a_top_dma_ptrs[0], a_top_token_ids)
            load_scale_g2r(
                scale_a_top_g2r[0],
                scale_a_rsrc,
                kk + 2,
                a_top_scale_tile,
                scale_a_padded_rows,
                True,
            )
            hot_loop_scheduler_mainloop(1, a_phase_vmem, b_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_tr,
                frag_b_up,
                frag_a_top,
                scale_a_top_frag,
                scale_b_up_frag,
            )
            waitvmcnt_barrier(wait_ba)
            load_b(b_gate_read[1], frag_b_gate)
            scale_b_gate_frag.store(scale_b_gate_g2r[1].load())
            raw_a_gather_g2s(
                kk + 2, a_bottom_dma_ptrs[0], a_bottom_token_ids
            )
            load_scale_g2r(
                scale_a_bottom_g2r[0],
                scale_a_rsrc,
                kk + 2,
                a_bottom_scale_tile,
                scale_a_padded_rows,
                True,
            )
            hot_loop_scheduler_mainloop(2, a_phase_vmem, b_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_br,
                frag_b_up,
                frag_a_bottom,
                scale_a_bottom_frag,
                scale_b_up_frag,
            )
            waitvmcnt_barrier(wait_ba)
            load_a(a_top_source[1], frag_a_top_dest)
            scale_a_top_frag.store(scale_a_top_g2r[1].load())
            raw_b_mxfp4_g2s(kk + 2, lds.b_up0.ptr, up_row_tile)
            load_scale_g2r(
                scale_b_up_g2r[0],
                scale_b_rsrc,
                kk + 2,
                up_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            hot_loop_scheduler_mainloop(3, b_phase_vmem, a_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_tl,
                frag_b_gate,
                frag_a_top,
                scale_a_top_frag,
                scale_b_gate_frag,
            )
            waitvmcnt_barrier(wait_ab)
            load_a(a_bottom_source[1], frag_a_bottom_dest)
            scale_a_bottom_frag.store(scale_a_bottom_g2r[1].load())
            raw_b_mxfp4_g2s(kk + 3, lds.b_gate1.ptr, gate_row_tile)
            load_scale_g2r(
                scale_b_gate_g2r[1],
                scale_b_rsrc,
                kk + 3,
                gate_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            hot_loop_scheduler_mainloop(4, b_phase_vmem, a_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_bl,
                frag_b_gate,
                frag_a_bottom,
                scale_a_bottom_frag,
                scale_b_gate_frag,
            )
            waitvmcnt_barrier(wait_ab)
            load_b(b_up_read[1], frag_b_up)
            scale_b_up_frag.store(scale_b_up_g2r[1].load())
            raw_a_gather_g2s(kk + 3, a_top_dma_ptrs[1], a_top_token_ids)
            load_scale_g2r(
                scale_a_top_g2r[1],
                scale_a_rsrc,
                kk + 3,
                a_top_scale_tile,
                scale_a_padded_rows,
                True,
            )
            hot_loop_scheduler_mainloop(5, a_phase_vmem, b_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_tr,
                frag_b_up,
                frag_a_top,
                scale_a_top_frag,
                scale_b_up_frag,
            )
            waitvmcnt_barrier(wait_ba)
            load_b(b_gate_read[0], frag_b_gate)
            scale_b_gate_frag.store(scale_b_gate_g2r[0].load())
            raw_a_gather_g2s(
                kk + 3, a_bottom_dma_ptrs[1], a_bottom_token_ids
            )
            load_scale_g2r(
                scale_a_bottom_g2r[1],
                scale_a_rsrc,
                kk + 3,
                a_bottom_scale_tile,
                scale_a_padded_rows,
                True,
            )
            hot_loop_scheduler_mainloop(6, a_phase_vmem, b_dsrd)
            rocdl.sched_barrier(0)

            do_gemm(
                frag_c_br,
                frag_b_up,
                frag_a_bottom,
                scale_a_bottom_frag,
                scale_b_up_frag,
            )
            waitvmcnt_barrier(wait_ba)
            load_a(a_top_source[0], frag_a_top_dest)
            scale_a_top_frag.store(scale_a_top_g2r[0].load())
            raw_b_mxfp4_g2s(kk + 3, lds.b_up1.ptr, up_row_tile)
            load_scale_g2r(
                scale_b_up_g2r[1],
                scale_b_rsrc,
                kk + 3,
                up_scale_row_tile,
                scale_b_padded_rows,
                False,
            )
            hot_loop_scheduler_mainloop(7, b_phase_vmem, a_dsrd)
            rocdl.sched_barrier(0)
            loop_results = yield [
                frag_c_tl.load(),
                frag_c_tr.load(),
                frag_c_bl.load(),
                frag_c_br.load(),
            ]

        frag_c_tl.store(loop_results[0])
        frag_c_tr.store(loop_results[1])
        frag_c_bl.store(loop_results[2])
        frag_c_br.store(loop_results[3])

        waitvmcnt_barrier(wait_ab)
        do_gemm(
            frag_c_tl,
            frag_b_gate,
            frag_a_top,
            scale_a_top_frag,
            scale_b_gate_frag,
        )
        load_a(a_bottom_source[0], frag_a_bottom_dest)
        scale_a_bottom_frag.store(scale_a_bottom_g2r[0].load())
        hot_loop_scheduler_mainloop(0, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(2 * (a_phase_vmem + b_phase_vmem))
        do_gemm(
            frag_c_bl,
            frag_b_gate,
            frag_a_bottom,
            scale_a_bottom_frag,
            scale_b_gate_frag,
        )
        load_b(b_up_read[0], frag_b_up)
        scale_b_up_frag.store(scale_b_up_g2r[0].load())
        hot_loop_scheduler_mainloop(1, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(2 * a_phase_vmem + b_phase_vmem)
        do_gemm(
            frag_c_tr,
            frag_b_up,
            frag_a_top,
            scale_a_top_frag,
            scale_b_up_frag,
        )
        load_b(b_gate_read[1], frag_b_gate)
        scale_b_gate_frag.store(scale_b_gate_g2r[1].load())
        hot_loop_scheduler_mainloop(2, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(a_phase_vmem + b_phase_vmem)
        do_gemm(
            frag_c_br,
            frag_b_up,
            frag_a_bottom,
            scale_a_bottom_frag,
            scale_b_up_frag,
        )
        load_a(a_top_source[1], frag_a_top_dest)
        scale_a_top_frag.store(scale_a_top_g2r[1].load())
        hot_loop_scheduler_mainloop(3, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(b_phase_vmem)
        do_gemm(
            frag_c_tl,
            frag_b_gate,
            frag_a_top,
            scale_a_top_frag,
            scale_b_gate_frag,
        )
        load_a(a_bottom_source[1], frag_a_bottom_dest)
        scale_a_bottom_frag.store(scale_a_bottom_g2r[1].load())
        hot_loop_scheduler_mainloop(4, 0, 8)
        rocdl.sched_barrier(0)

        waitvmcnt_barrier(0)
        do_gemm(
            frag_c_bl,
            frag_b_gate,
            frag_a_bottom,
            scale_a_bottom_frag,
            scale_b_gate_frag,
        )
        load_b(b_up_read[1], frag_b_up)
        scale_b_up_frag.store(scale_b_up_g2r[1].load())
        hot_loop_scheduler_mainloop(5, 0, 8)
        rocdl.sched_barrier(0)

        do_gemm(
            frag_c_tr,
            frag_b_up,
            frag_a_top,
            scale_a_top_frag,
            scale_b_up_frag,
        )
        hot_loop_scheduler_mainloop(6, 0, 0)
        rocdl.sched_barrier(0)
        do_gemm(
            frag_c_br,
            frag_b_up,
            frag_a_bottom,
            scale_a_bottom_frag,
            scale_b_up_frag,
        )
        hot_loop_scheduler_mainloop(7, 0, 0)
        rocdl.sched_barrier(0)

        pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
        lane_group = lane_id // 16
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        def store_quadrant(c_frag, row_quadrant: int, is_up: bool):
            for row_repeat in range_constexpr(4):
                for col_repeat in range_constexpr(0, 4, 2):
                    acc_a = Vec(c_frag[None, col_repeat, row_repeat].load())
                    acc_b = Vec(c_frag[None, col_repeat + 1, row_repeat].load())
                    d0_a = rocdl.cvt_pk_bf16_f32(acc_a[0], acc_a[1])
                    d1_a = rocdl.cvt_pk_bf16_f32(acc_a[2], acc_a[3])
                    d0_b = rocdl.cvt_pk_bf16_f32(acc_b[0], acc_b[1])
                    d1_b = rocdl.cvt_pk_bf16_f32(acc_b[2], acc_b[3])
                    swap0 = rocdl.permlane16_swap(
                        pair_type,
                        arith._to_raw(d0_a),
                        arith._to_raw(d0_b),
                        False,
                        False,
                    )
                    swap1 = rocdl.permlane16_swap(
                        pair_type,
                        arith._to_raw(d1_a),
                        arith._to_raw(d1_b),
                        False,
                        False,
                    )
                    packed = Vec.from_elements(
                        [
                            fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                        ],
                        Int32,
                    )
                    row_local = (
                        row_quadrant * block_m
                        + row_repeat * 32
                        + wave_m * 16
                        + lane_id % 16
                    )
                    sorted_row = (
                        expert_block_i32 * SORT_BLOCK_M + fx.Int32(row_local)
                    )
                    fused_id = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row, vec_width=1, dtype=T.i32
                    )
                    token_id = arith.andi(fused_id, mask24)
                    slot_id = arith.shrui(fused_id, arith.constant(24, type=T.i32))
                    token_valid = arith.cmpi(
                        CmpIPredicate.ult, token_id, num_tokens_i32
                    )
                    slot_valid = arith.cmpi(
                        CmpIPredicate.ult,
                        slot_id,
                        arith.constant(topk, type=T.i32),
                    )
                    sorted_valid = arith.cmpi(
                        CmpIPredicate.ult, sorted_row, num_valid_i32
                    )
                    store_valid = arith.andi(
                        sorted_valid, arith.andi(token_valid, slot_valid)
                    )
                    projection_base = (
                        intermediate_size if is_up else 0
                    ) + n_tile_i32 * block_n
                    col = (
                        projection_base
                        + col_repeat * 32
                        + fx.Int32((lane_group % 2) * 32)
                        + fx.Int32(wave_n * 16)
                        + fx.Int32((lane_group // 2) * 8)
                    )
                    output_element = (
                        (fx.Int32(token_id) * topk + fx.Int32(slot_id))
                        * output_size
                        + col
                    )
                    buffer_ops.buffer_store(
                        packed,
                        c_rsrc,
                        output_element * 2,
                        offset_is_bytes=True,
                        mask=store_valid,
                    )

        store_quadrant(frag_c_tl, 0, False)
        store_quadrant(frag_c_bl, 1, False)
        store_quadrant(frag_c_tr, 0, True)
        store_quadrant(frag_c_br, 1, True)



    @flyc.jit
    def launch_moe_gateup(
        a: fx.Tensor,
        b: fx.Tensor,
        scale_a: fx.Tensor,
        scale_b: fx.Tensor,
        c: fx.Tensor,
        sorted_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        num_tokens: int,
        num_expert_blocks: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        value_attrs = {
            "rocdl.waves_per_eu": 1,
            "passthrough": [["amdgpu-agpr-alloc", "256,256"]],
        }
        moe_gateup_kernel(
            a,
            b,
            scale_a,
            scale_b,
            c,
            sorted_ids,
            expert_ids,
            num_valid_ids,
            num_tokens,
            num_expert_blocks,
            value_attrs=value_attrs,
        ).launch(
            grid=(num_n_tiles * num_expert_blocks, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_moe_gateup.compile_hints["llvm_options"] = {
        "amdgpu-mfma-vgpr-form": False
    }
    return launch_moe_gateup


def run_accuracy_case(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    *,
    b_lds_swizzle: bool = False,
    xcd_swizzle: bool = False,
    group_size_m: int = 1,
    return_metrics: bool = False,
    top_error_threshold: float | None = None,
    top_error_count: int = 20,
) -> bool | dict[str, object]:
    validate_case_parameters(
        tokens, intermediate_size, hidden_size, topk, num_experts
    )
    inputs = prepare_moe_inputs(
        tokens, intermediate_size, hidden_size, topk, num_experts
    )
    output = torch.full(
        (tokens, topk, 2 * intermediate_size),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    args = (
        inputs["a"].view(torch.int8).view(-1),
        inputs["weight"].view(torch.int8),
        inputs["scale_a"],
        inputs["scale_b"],
        output.view(-1),
        inputs["sorted_ids"],
        inputs["expert_ids"],
        inputs["num_valid_ids"],
        tokens,
        inputs["expert_ids"].numel(),
        torch.cuda.current_stream(),
    )
    launcher = compile_moe_gateup_4w(
        intermediate_size,
        hidden_size,
        topk,
        num_experts,
        b_lds_swizzle=b_lds_swizzle,
        xcd_swizzle=xcd_swizzle,
        group_size_m=group_size_m,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    reference = moe_reference(
        inputs, tokens, intermediate_size, hidden_size, topk, num_experts
    )
    finite = bool(torch.isfinite(output).all())
    max_abs = (output.float() - reference.float()).abs().max().item()
    allclose = finite and torch.allclose(
        output, reference, rtol=0.02, atol=0.01
    )
    import pyhip

    diff = (
        pyhip.calc_diff(output.float(), reference, diff_thr=0.00001)
        if finite
        else float("inf")
    )
    correct = allclose and diff <= 0.00001
    print(
        f"accuracy: shape={tuple(output.shape)} experts={num_experts} "
        f"b_lds={'swizzle' if b_lds_swizzle else 'padding'} "
        f"xcd_swizzle={xcd_swizzle} group_size_m={group_size_m} "
        f"expert_blocks={inputs['expert_ids'].numel()} finite={finite} "
        f"allclose={allclose} max_abs={max_abs:.6g} diff={diff:.6g}"
    )
    if top_error_threshold is not None and max_abs >= top_error_threshold:
        abs_error = (output.float() - reference.float()).abs()
        flat_error = abs_error.view(-1)
        error_count = min(top_error_count, flat_error.numel())
        top_errors, top_indices = torch.topk(flat_error, error_count)
        output_flat = output.view(-1).float()
        reference_flat = reference.view(-1).float()
        print(
            f"top {error_count} absolute errors: tokens={tokens} "
            f"hidden={hidden_size} topk={topk} experts={num_experts} "
            f"intermediate={intermediate_size}"
        )
        for rank, (flat_index, absolute_error) in enumerate(
            zip(top_indices.tolist(), top_errors.tolist()), start=1
        ):
            token_index, remainder = divmod(
                flat_index, topk * 2 * intermediate_size
            )
            slot_index, output_index = divmod(
                remainder, 2 * intermediate_size
            )
            reference_value = reference_flat[flat_index].item()
            output_value = output_flat[flat_index].item()
            relative_error = (
                absolute_error / abs(reference_value)
                if reference_value != 0.0
                else (0.0 if absolute_error == 0.0 else float("inf"))
            )
            print(
                f"  {rank:2d}: index=({token_index},{slot_index},{output_index}) "
                f"ref={reference_value:.9g} out={output_value:.9g} "
                f"abs_error={absolute_error:.9g} "
                f"relative_error={relative_error:.9g}"
            )
    if return_metrics:
        return {
            "output_shape": str(tuple(output.shape)),
            "expert_blocks": inputs["expert_ids"].numel(),
            "finite": finite,
            "allclose": allclose,
            "max_abs": max_abs,
            "diff": diff,
            "correct": correct,
        }
    return correct


def run_accuracy_matrix(
    output_csv: str,
    *,
    b_lds_swizzle: bool = False,
    xcd_swizzle: bool = False,
    group_size_m: int = 1,
) -> None:
    fieldnames = [
        "tokens",
        "hidden_size",
        "topk",
        "num_experts",
        "intermediate_size",
        "output_shape",
        "expert_blocks",
        "finite",
        "allclose",
        "max_abs",
        "diff",
        "correct",
        "elapsed_s",
        "error",
    ]
    combinations = [
        (tokens, hidden_size, topk, num_experts, intermediate_size)
        for tokens in range(8192, 8210)
        for hidden_size in (6144, 4096)
        for topk in (5, 6, 7, 8)
        for num_experts in (384, 120)
        for intermediate_size in (256, 128, 64)
    ]
    failures = []
    with open(output_csv, "w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for case_index, case in enumerate(combinations, start=1):
            tokens, hidden_size, topk, num_experts, intermediate_size = case
            started = time.perf_counter()
            row = {
                "tokens": tokens,
                "hidden_size": hidden_size,
                "topk": topk,
                "num_experts": num_experts,
                "intermediate_size": intermediate_size,
                "error": "",
            }
            try:
                row.update(
                    run_accuracy_case(
                        tokens,
                        intermediate_size,
                        hidden_size,
                        topk,
                        num_experts,
                        b_lds_swizzle=b_lds_swizzle,
                        xcd_swizzle=xcd_swizzle,
                        group_size_m=group_size_m,
                        return_metrics=True,
                        top_error_threshold=8.0,
                    )
                )
            except Exception as error:
                row.update(
                    {
                        "correct": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            row["elapsed_s"] = time.perf_counter() - started
            writer.writerow(row)
            csv_file.flush()
            if not row["correct"]:
                failures.append(row.copy())
            print(
                f"matrix [{case_index}/{len(combinations)}] "
                f"tokens={tokens} hidden={hidden_size} topk={topk} "
                f"experts={num_experts} intermediate={intermediate_size} "
                f"correct={row['correct']} elapsed={row['elapsed_s']:.3f}s"
            )
    print(
        f"accuracy matrix: total={len(combinations)} "
        f"passed={len(combinations) - len(failures)} failed={len(failures)} "
        f"csv={output_csv}"
    )
    for failure in failures:
        print(f"FAILED: {failure}")


def _clone_benchmark_args(args, data_clones: int):
    arg_sets = [args]
    for _ in range(1, data_clones):
        arg_sets.append(
            tuple(
                value.clone() if isinstance(value, torch.Tensor) else value
                for value in args
            )
        )
    return arg_sets


def _benchmark_kernel(
    kernel,
    arg_sets,
    flops: int,
    rw_bytes: int,
    name: str,
    warmup: int,
    iterations: int,
):
    from pyhip import cudaPerf

    for iteration in range(warmup):
        kernel(*arg_sets[iteration % len(arg_sets)])
    torch.cuda.synchronize()
    samples = []
    for iteration in range(iterations):
        clone_index = (warmup + iteration) % len(arg_sets)
        with cudaPerf(
            flops,
            rw_bytes,
            name=f"{name}_{clone_index}",
            verbose=0,
        ) as perf:
            kernel(*arg_sets[clone_index])
        samples.append((perf.dt() * 1.0e3, perf.tflops(), perf.bw()))
    samples.sort(key=lambda sample: sample[0])
    median = samples[len(samples) // 2]
    return samples[0], median


def run_scale_padding_accuracy_case(m: int, n: int, k: int) -> bool:
    from test_mxfp8_gemm_4w import compile_gemm_fp8

    if m <= 0:
        raise ValueError("m must be positive")
    if n <= 0 or n % 8 != 0:
        raise ValueError("n must be a positive multiple of 8")
    if k < 512 or k % 256 != 0:
        raise ValueError("k must be a multiple of 256 and at least 512")

    per_1x32_mx_quant_hip, dtypes, e8m0_to_f32, mxfp4_to_f32 = (
        _load_mx_helpers()
    )
    a, scale_a_raw = per_1x32_mx_quant_hip(
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * A_INPUT_SCALE,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    b, scale_b_raw = per_1x32_mx_quant_hip(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * B_INPUT_SCALE,
        quant_dtype=dtypes.fp4x2,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )

    scale_a_padded_rows = div_up(m, 256) * 256
    scale_b_padded_rows = div_up(n, 256) * 256
    scale_a = _permute_scale(
        scale_a_raw, padded_rows=scale_a_padded_rows
    )
    scale_b = _permute_scale(
        scale_b_raw, padded_rows=scale_b_padded_rows
    )
    a_descriptor_bytes = m * k
    b_descriptor_bytes = n * k // 2
    scale_a_descriptor_bytes = scale_a_padded_rows * (k // 32)
    scale_b_descriptor_bytes = scale_b_padded_rows * (k // 32)
    assert a.view(torch.uint8).numel() == a_descriptor_bytes
    assert b.view(torch.uint8).numel() == b_descriptor_bytes
    assert scale_a.view(torch.uint8).numel() == scale_a_descriptor_bytes
    assert scale_b.view(torch.uint8).numel() == scale_b_descriptor_bytes

    output = torch.full(
        (m, n), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    args = (
        a.view(torch.int8).view(-1),
        b.view(torch.int8).view(-1),
        scale_a,
        scale_b,
        output.view(-1),
        m,
        torch.cuda.current_stream(),
    )
    launcher = compile_gemm_fp8(
        256,
        256,
        128,
        n,
        k,
        lds_swizzle=False,
        preshuffle_b=False,
        permlane_epilogue=True,
        store_overlap=False,
        with_scale=True,
        b_mxfp4=True,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    a_dequant = a.float() * e8m0_to_f32(scale_a_raw).repeat_interleave(32, dim=1)
    b_dequant = mxfp4_to_f32(b) * e8m0_to_f32(
        scale_b_raw
    ).repeat_interleave(32, dim=1)
    reference = (a_dequant @ b_dequant.t()).to(torch.bfloat16)
    finite = bool(torch.isfinite(output).all())
    allclose = finite and torch.allclose(
        output, reference, rtol=0.02, atol=0.01
    )
    import pyhip

    diff = (
        pyhip.calc_diff(output.float(), reference.float(), diff_thr=0.00001)
        if finite
        else float("inf")
    )
    correct = allclose and diff <= 0.00001
    print(
        f"scale padding accuracy: M={m} N={n} K={k} "
        f"A={m}x{k} B={n}x{k} "
        f"scale_a_rows={m}->{scale_a_padded_rows} "
        f"scale_b_rows={n}->{scale_b_padded_rows} "
        f"descriptor_bytes=(A={a_descriptor_bytes}, B={b_descriptor_bytes}, "
        f"scale_a={scale_a_descriptor_bytes}, scale_b={scale_b_descriptor_bytes}) "
        f"allclose={allclose} diff={diff:.6g}"
    )
    return correct


def run_scale_padding_accuracy() -> None:
    for m in (33, 127, 128, 129, 255, 256, 257):
        if not run_scale_padding_accuracy_case(m, 392, 512):
            raise SystemExit(f"scale padding accuracy failed for M={m}")


def run_benchmark(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    warmup: int,
    iterations: int,
    data_clones: int,
    *,
    b_lds_swizzle: bool = False,
    xcd_swizzle: bool = False,
    group_size_m: int = 1,
) -> None:
    validate_case_parameters(
        tokens, intermediate_size, hidden_size, topk, num_experts
    )
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if data_clones <= 0:
        raise ValueError("data_clones must be positive")
    inputs = prepare_moe_inputs(
        tokens, intermediate_size, hidden_size, topk, num_experts
    )
    output = torch.empty(
        (tokens, topk, 2 * intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    moe_args = (
        inputs["a"].view(torch.int8).view(-1),
        inputs["weight"].view(torch.int8),
        inputs["scale_a"],
        inputs["scale_b"],
        output.view(-1),
        inputs["sorted_ids"],
        inputs["expert_ids"],
        inputs["num_valid_ids"],
        tokens,
        inputs["expert_ids"].numel(),
        torch.cuda.current_stream(),
    )
    moe_launcher = compile_moe_gateup_4w(
        intermediate_size,
        hidden_size,
        topk,
        num_experts,
        b_lds_swizzle=b_lds_swizzle,
        xcd_swizzle=xcd_swizzle,
        group_size_m=group_size_m,
    )
    moe_arg_sets = _clone_benchmark_args(moe_args, data_clones)
    moe_rw_bytes = sum(
        value.numel() * value.element_size()
        for value in moe_args
        if isinstance(value, torch.Tensor)
    )
    del inputs, moe_args, output
    moe_kernel = flyc.compile[{"opt_level": 2}](moe_launcher, *moe_arg_sets[0])
    flops = 2 * tokens * topk * (2 * intermediate_size) * hidden_size
    moe_best, moe_median = _benchmark_kernel(
        moe_kernel,
        moe_arg_sets,
        flops,
        moe_rw_bytes,
        f"moe_gateup_xcd{int(xcd_swizzle)}_group{group_size_m}",
        warmup,
        iterations,
    )
    del moe_arg_sets, moe_kernel

    from test_mxfp8_gemm_4w import compile_gemm_fp8

    per_1x32_mx_quant_hip, dtypes, _, _ = _load_mx_helpers()
    gemm_m = tokens
    gemm_n = topk * 2 * intermediate_size
    gemm_a, gemm_scale_a_raw = per_1x32_mx_quant_hip(
        torch.randn(
            (gemm_m, hidden_size), device="cuda", dtype=torch.bfloat16
        ) * A_INPUT_SCALE,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    gemm_b, gemm_scale_b_raw = per_1x32_mx_quant_hip(
        torch.randn(
            (gemm_n, hidden_size), device="cuda", dtype=torch.bfloat16
        ) * B_INPUT_SCALE,
        quant_dtype=dtypes.fp4x2,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    gemm_output = torch.empty(
        (gemm_m, gemm_n), device="cuda", dtype=torch.bfloat16
    )
    gemm_scale_a_padded_rows = div_up(gemm_m, 256) * 256
    gemm_scale_b_padded_rows = div_up(gemm_n, 256) * 256
    gemm_scale_a = _permute_scale(
        gemm_scale_a_raw, padded_rows=gemm_scale_a_padded_rows
    )
    gemm_scale_b = _permute_scale(
        gemm_scale_b_raw, padded_rows=gemm_scale_b_padded_rows
    )
    assert gemm_a.view(torch.uint8).numel() == gemm_m * hidden_size
    assert gemm_b.view(torch.uint8).numel() == gemm_n * hidden_size // 2
    assert gemm_scale_a.view(torch.uint8).numel() == (
        gemm_scale_a_padded_rows * (hidden_size // 32)
    )
    assert gemm_scale_b.view(torch.uint8).numel() == (
        gemm_scale_b_padded_rows * (hidden_size // 32)
    )
    gemm_args = (
        gemm_a.view(torch.int8).view(-1),
        gemm_b.view(torch.int8).view(-1),
        gemm_scale_a,
        gemm_scale_b,
        gemm_output.view(-1),
        gemm_m,
        torch.cuda.current_stream(),
    )
    gemm_launcher = compile_gemm_fp8(
        256,
        256,
        128,
        gemm_n,
        hidden_size,
        lds_swizzle=False,
        preshuffle_b=False,
        permlane_epilogue=True,
        store_overlap=False,
        with_scale=True,
        b_mxfp4=True,
    )
    gemm_arg_sets = _clone_benchmark_args(gemm_args, data_clones)
    gemm_rw_bytes = sum(
        value.numel() * value.element_size()
        for value in gemm_args
        if isinstance(value, torch.Tensor)
    )
    del (
        gemm_args,
        gemm_a,
        gemm_b,
        gemm_scale_a_raw,
        gemm_scale_b_raw,
        gemm_scale_a,
        gemm_scale_b,
        gemm_output,
    )
    gemm_kernel = flyc.compile[{"opt_level": 2}](gemm_launcher, *gemm_arg_sets[0])
    gemm_best, gemm_median = _benchmark_kernel(
        gemm_kernel,
        gemm_arg_sets,
        flops,
        gemm_rw_bytes,
        "gemm",
        warmup,
        iterations,
    )

    print(
        f"benchmark: b_lds={'swizzle' if b_lds_swizzle else 'padding'} "
        f"xcd_swizzle={xcd_swizzle} group_size_m={group_size_m} "
        f"clones={data_clones} warmup={warmup} runs={iterations}"
    )
    print(
        f"moe:  best={moe_best[0]:.6f} ms median={moe_median[0]:.6f} ms "
        f"best={moe_best[1]:.2f} TFLOPS median={moe_median[1]:.2f} TFLOPS "
        f"best_bw={moe_best[2]:.2f} GB/s median_bw={moe_median[2]:.2f} GB/s"
    )
    print(
        f"gemm: best={gemm_best[0]:.6f} ms median={gemm_median[0]:.6f} ms "
        f"best={gemm_best[1]:.2f} TFLOPS median={gemm_median[1]:.2f} TFLOPS "
        f"best_bw={gemm_best[2]:.2f} GB/s median_bw={gemm_median[2]:.2f} GB/s"
    )
    print(
        f"gap: latency={moe_best[0] / gemm_best[0]:.3f}x "
        f"throughput={moe_best[1] / gemm_best[1]:.3%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-accuracy", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--accuracy-matrix", action="store_true")
    parser.add_argument(
        "--accuracy-csv",
        default="moe_mxfp8_mxfp4_gateup_4w_accuracy.csv",
    )
    parser.add_argument("--unaligned-accuracy", action="store_true")
    parser.add_argument("--scale-padding-accuracy", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--b-lds-swizzle", action="store_true")
    parser.add_argument("--xcd-swizzle", action="store_true")
    parser.add_argument("--group-size-m", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-clones", type=int, default=10)
    args = parser.parse_args()

    if args.small_accuracy:
        if not run_accuracy_case(
            96,
            128,
            512,
            2,
            3,
            b_lds_swizzle=args.b_lds_swizzle,
            xcd_swizzle=args.xcd_swizzle,
            group_size_m=args.group_size_m,
        ):
            raise SystemExit("small padded accuracy failed")
        return
    if args.accuracy:
        if not run_accuracy_case(
            args.tokens,
            args.intermediate_size,
            args.hidden_size,
            args.topk,
            args.num_experts,
            b_lds_swizzle=args.b_lds_swizzle,
            xcd_swizzle=args.xcd_swizzle,
            group_size_m=args.group_size_m,
        ):
            raise SystemExit("full accuracy failed")
        return
    if args.accuracy_matrix:
        run_accuracy_matrix(
            args.accuracy_csv,
            b_lds_swizzle=args.b_lds_swizzle,
            xcd_swizzle=args.xcd_swizzle,
            group_size_m=args.group_size_m,
        )
        return
    if args.unaligned_accuracy:
        if not run_accuracy_case(
            8193,
            512,
            6144,
            8,
            384,
            b_lds_swizzle=args.b_lds_swizzle,
            xcd_swizzle=args.xcd_swizzle,
            group_size_m=args.group_size_m,
        ):
            raise SystemExit("unaligned-token accuracy failed")
        return
    if args.scale_padding_accuracy:
        run_scale_padding_accuracy()
        return
    if args.benchmark:
        run_benchmark(
            args.tokens,
            args.intermediate_size,
            args.hidden_size,
            args.topk,
            args.num_experts,
            args.warmup,
            args.iterations,
            args.data_clones,
            b_lds_swizzle=args.b_lds_swizzle,
            xcd_swizzle=args.xcd_swizzle,
            group_size_m=args.group_size_m,
        )
        return
    parser.error(
        "select --routing-probe, --small-accuracy, --accuracy, "
        "--accuracy-matrix, --unaligned-accuracy, --scale-padding-accuracy, "
        "or --benchmark"
    )


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "MFMA_Scale requires gfx950"
    torch.manual_seed(0)
    main()
