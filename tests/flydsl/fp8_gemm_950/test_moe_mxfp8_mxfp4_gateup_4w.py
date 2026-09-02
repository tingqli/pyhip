"""Four-wave MXFP8 x MXFP4 MoE gate/up GEMM.

The kernel writes the unfused gate and up projections to
``[tokens, topk, 2 * intermediate_size]``.  This file starts with a focused
buffer-resource routing probe; the GEMM implementation below reuses the same
routing contract.
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, Float32, Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf


SORT_BLOCK_M = 256
TOKEN_MASK = 0xFFFFFF


def div_up(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


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


def prepare_moe_inputs(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
):
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
    )
    # a :[tokens, hidden_size]
    # scale_a_raw: [tokens, hidden_size // 32]
    a, scale_a_raw = per_1x32_mx_quant_hip(
        a_source,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    del a_source

    output_size = 2 * intermediate_size
    weight_storage = torch.empty(
        (num_experts, output_size, hidden_size // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    weight = weight_storage.view(dtypes.fp4x2)
    scale_b_raw = torch.empty(
        (num_experts, output_size, hidden_size // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    for expert in range(num_experts):
        source = torch.randn(
            (output_size, hidden_size), device="cuda", dtype=torch.bfloat16
        ) * 3.0
        quantized, scale = per_1x32_mx_quant_hip(
            source,
            quant_dtype=dtypes.fp4x2,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
        weight[expert].copy_(quantized)
        scale_b_raw[expert].copy_(scale.view(torch.uint8))

    decoded_tokens = (sorted_ids & TOKEN_MASK).to(torch.int64)
    decoded_slots = ((sorted_ids >> 24) & 0xFF).to(torch.int64)
    valid = (decoded_tokens < tokens) & (decoded_slots < topk)
    sorted_scale_a_raw = torch.full(
        (sorted_ids.numel(), hidden_size // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    sorted_scale_a_raw[valid] = scale_a_raw.view(torch.uint8)[decoded_tokens[valid]]

    scale_a_padded_rows = sorted_ids.numel()
    scale_b_rows = num_experts * output_size
    scale_b_padded_rows = div_up(scale_b_rows, 256) * 256
    scale_a = _permute_scale(
        sorted_scale_a_raw, padded_rows=scale_a_padded_rows
    )
    scale_b = _permute_scale(
        scale_b_raw.view(scale_b_rows, hidden_size // 32),
        padded_rows=scale_b_padded_rows,
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


def compile_routing_buffer_probe(k: int):
    """Compile a 16-byte indirect gather/scatter probe for the MoE row mapping."""
    assert k % 16 == 0

    @flyc.kernel(known_block_size=[256, 1, 1])
    def routing_buffer_probe(
        arg_a: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_out: fx.Tensor,
        num_tokens: int,
        num_sorted: int,
    ):
        linear = fx.block_idx.x * 256 + fx.thread_idx.x
        sorted_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_ids,
            num_records_bytes=arith._to_raw(fx.Int32(num_sorted * 4)),
        )
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a,
            num_records_bytes=arith._to_raw(fx.Int32(num_tokens * k)),
        )
        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out,
            num_records_bytes=arith._to_raw(fx.Int32(num_sorted * 16)),
        )

        packed_id = ArithValue(
            buffer_ops.buffer_load(sorted_rsrc, linear, vec_width=1, dtype=T.i32)
        )
        token_id = packed_id & arith.constant(TOKEN_MASK, type=T.i32)
        num_tokens_i32 = ArithValue(num_tokens)
        token_valid = arith.cmpi(
            CmpIPredicate.ult, token_id, num_tokens_i32
        )
        gathered = buffer_ops.buffer_load(
            a_rsrc,
            token_id * arith.constant(k // 4, type=T.i32),
            vec_width=4,
            dtype=T.i32,
        )
        buffer_ops.buffer_store(
            gathered,
            out_rsrc,
            linear * arith.constant(4, type=T.i32),
            mask=token_valid,
        )

    @flyc.jit
    def launch_probe(
        a: fx.Tensor,
        sorted_ids: fx.Tensor,
        out: fx.Tensor,
        num_tokens: int,
        num_sorted: int,
        stream: fx.Stream = fx.Stream(None),
    ):
        routing_buffer_probe(a, sorted_ids, out, num_tokens, num_sorted).launch(
            grid=(div_up(num_sorted, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_probe


def encode_waitcnt_950(vmcnt: int = 63, expcnt: int = 7, lgkmcnt: int = 63) -> int:
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def waitvmcnt_barrier(vmcnt: int) -> None:
    rocdl.s_waitcnt(encode_waitcnt_950(vmcnt=vmcnt))
    rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
    rocdl.s_barrier()


def compile_moe_gateup_4w(
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
):
    """Build the padding-LDS four-wave raw gate/up projection kernel."""
    block_m = SORT_BLOCK_M // 2
    block_n = 128
    block_k = 128
    output_size = 2 * intermediate_size
    scale_b_padded_rows = div_up(num_experts * output_size, 256) * 256
    assert intermediate_size % block_n == 0
    assert hidden_size % block_k == 0

    element_type = Float8E4M3FN
    weight_type = Float4E2M1FN
    a_group8 = 8 * block_k + 32
    a_group16 = 2 * a_group8
    a_lds_elems = (block_m // 8) * a_group8
    b_group16 = 16 * block_k + 64
    b_lds_elems = (block_n // 16) * b_group16

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
        n_tile = fx.block_idx.x
        expert_block = fx.block_idx.y
        n_tile_i32 = fx.Int32(n_tile)
        expert_block_i32 = fx.Int32(expert_block)
        num_tokens_i32 = fx.Int32(num_tokens)
        num_expert_blocks_i32 = fx.Int32(num_expert_blocks)

        a_tensor_bytes = num_tokens_i32 * fx.Int32(hidden_size)
        b_tensor_bytes = fx.Int32(
            num_experts * output_size * hidden_size // 2
        )
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a,
            num_records_bytes=arith._to_raw(a_tensor_bytes),
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b,
            num_records_bytes=arith._to_raw(b_tensor_bytes),
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

        expert_i32 = fx.Int32(
            buffer_ops.buffer_load(
                expert_rsrc, expert_block, vec_width=1, dtype=T.i32
            )
        )
        num_valid_i32 = fx.Int32(
            buffer_ops.buffer_load(valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )

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
        write_layout_b = fx.make_layout(
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
            fx.make_view(ptr, write_layout_b)
            for ptr in (lds.b_gate0.ptr, lds.b_gate1.ptr)
        ]
        b_up_read = [
            fx.make_view(ptr, write_layout_b)
            for ptr in (lds.b_up0.ptr, lds.b_up1.ptr)
        ]

        copy_a_atom = fx.make_copy_atom(fx.UniversalCopy128b(), element_type)
        copy_a = fx.make_tiled_copy_B(copy_a_atom, tiled_mma).get_slice(tid)
        a_top_source = [copy_a.partition_S(view) for view in a_top_read]
        a_bottom_source = [copy_a.partition_S(view) for view in a_bottom_read]
        frag_a_top = thread_mma.make_fragment_B(a_top_read[0])
        frag_a_bottom = thread_mma.make_fragment_B(a_bottom_read[0])
        frag_b_gate = fx.make_rmem_tensor(16, Int32)
        frag_b_up = fx.make_rmem_tensor(16, Int32)
        frag_a_top_dest = copy_a.retile(frag_a_top)
        frag_a_bottom_dest = copy_a.retile(frag_a_bottom)

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
                chunk = wave_id_uniform + copy_round * 4
                row = chunk * 16 + lane_id // 4
                col_byte = (lane_id % 4) * 16
                global_row = row_tile * block_n + fx.Int32(row)
                global_byte = (
                    global_row * (hidden_size // 2)
                    + kk * (block_k // 2)
                    + fx.Int32(col_byte)
                )
                rocdl.raw_ptr_buffer_load_lds(
                    b_rsrc,
                    buffer_ops.get_element_ptr(
                        root,
                        byte_offset=chunk * (b_group16 // 2),
                        elem_type=T.i8,
                    ),
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

        def do_gemm(c_frag, b_frag, a_frag, scale_a, scale_b):
            c_value = c_frag.load().ir_value()
            b_value = vector.bitcast(T.vec(64, T.i8), b_frag.load().ir_value())
            a_value = vector.bitcast(T.vec(128, T.i8), a_frag.load().ir_value())
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
        frag_c_tl.fill(0)
        frag_c_tr.fill(0)
        frag_c_bl.fill(0)
        frag_c_br.fill(0)
        accumulators = [
            frag_c_tl.load(),
            frag_c_tr.load(),
            frag_c_bl.load(),
            frag_c_br.load(),
        ]

        expert_row_tile = expert_i32 * (output_size // block_n)
        gate_row_tile = expert_row_tile + n_tile_i32
        up_row_tile = (
            expert_row_tile
            + intermediate_size // block_n
            + n_tile_i32
        )
        a_top_scale_tile = expert_block_i32 * 2
        a_bottom_scale_tile = a_top_scale_tile + 1
        num_k_tiles = hidden_size // block_k
        assert num_k_tiles >= 4 and num_k_tiles % 2 == 0
        operand_vmem_per_tile = 12

        def issue_g2s(kk, buffer_index: int):
            raw_b_mxfp4_g2s(
                kk,
                (lds.b_gate0.ptr, lds.b_gate1.ptr)[buffer_index],
                gate_row_tile,
            )
            raw_b_mxfp4_g2s(
                kk,
                (lds.b_up0.ptr, lds.b_up1.ptr)[buffer_index],
                up_row_tile,
            )
            raw_a_gather_g2s(
                kk, a_top_dma_ptrs[buffer_index], a_top_token_ids
            )
            raw_a_gather_g2s(
                kk, a_bottom_dma_ptrs[buffer_index], a_bottom_token_ids
            )

        def load_operands(buffer_index: int):
            fx.copy(
                copy_a_atom,
                a_top_source[buffer_index],
                frag_a_top_dest,
            )
            fx.copy(
                copy_a_atom,
                a_bottom_source[buffer_index],
                frag_a_bottom_dest,
            )
            load_b_fragment(b_gate_read[buffer_index], frag_b_gate)
            load_b_fragment(b_up_read[buffer_index], frag_b_up)
            rocdl.s_waitcnt(encode_waitcnt_950(lgkmcnt=0))
            rocdl.s_barrier()

        def load_scales(kk):
            scale_a_top = load_scale_dword(
                scale_a_rsrc,
                kk,
                a_top_scale_tile,
                num_expert_blocks_i32 * 256,
                True,
            )
            scale_a_bottom = load_scale_dword(
                scale_a_rsrc,
                kk,
                a_bottom_scale_tile,
                num_expert_blocks_i32 * 256,
                True,
            )
            scale_b_gate = load_scale_dword(
                scale_b_rsrc,
                kk,
                gate_row_tile,
                scale_b_padded_rows,
                False,
            )
            scale_b_up = load_scale_dword(
                scale_b_rsrc,
                kk,
                up_row_tile,
                scale_b_padded_rows,
                False,
            )
            return scale_a_top, scale_a_bottom, scale_b_gate, scale_b_up

        def compute_tile(scales):
            scale_a_top, scale_a_bottom, scale_b_gate, scale_b_up = scales
            do_gemm(
                frag_c_tl,
                frag_b_gate,
                frag_a_top,
                scale_a_top,
                scale_b_gate,
            )
            do_gemm(
                frag_c_bl,
                frag_b_gate,
                frag_a_bottom,
                scale_a_bottom,
                scale_b_gate,
            )
            do_gemm(
                frag_c_tr,
                frag_b_up,
                frag_a_top,
                scale_a_top,
                scale_b_up,
            )
            do_gemm(
                frag_c_br,
                frag_b_up,
                frag_a_bottom,
                scale_a_bottom,
                scale_b_up,
            )

        issue_g2s(fx.Int32(0), 0)
        issue_g2s(fx.Int32(1), 1)
        waitvmcnt_barrier(operand_vmem_per_tile)

        for kk_index, states in range(0, num_k_tiles - 2, 2, init=accumulators):
            frag_c_tl.store(states[0])
            frag_c_tr.store(states[1])
            frag_c_bl.store(states[2])
            frag_c_br.store(states[3])
            kk = fx.Int32(kk_index)

            load_operands(0)
            scales0 = load_scales(kk)
            issue_g2s(kk + 2, 0)
            waitvmcnt_barrier(operand_vmem_per_tile)
            compute_tile(scales0)

            load_operands(1)
            scales1 = load_scales(kk + 1)
            issue_g2s(kk + 3, 1)
            waitvmcnt_barrier(operand_vmem_per_tile)
            compute_tile(scales1)
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

        load_operands(0)
        tail_scales0 = load_scales(fx.Int32(num_k_tiles - 2))
        waitvmcnt_barrier(0)
        compute_tile(tail_scales0)

        load_operands(1)
        tail_scales1 = load_scales(fx.Int32(num_k_tiles - 1))
        waitvmcnt_barrier(0)
        compute_tile(tail_scales1)

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
            grid=(intermediate_size // block_n, num_expert_blocks, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_moe_gateup.compile_hints["llvm_options"] = {
        "amdgpu-mfma-vgpr-form": False
    }
    return launch_moe_gateup


def run_routing_probe() -> None:
    tokens, topk, experts, k = 512, 4, 8, 128
    _, _, sorted_ids, _, _, _ = make_balanced_routing(
        tokens, topk, experts, k, device="cuda"
    )
    padding = torch.full((SORT_BLOCK_M,), tokens * topk, device="cuda", dtype=torch.int32)
    routed_ids = torch.cat((sorted_ids, padding))
    a = torch.arange(tokens * k, device="cuda", dtype=torch.uint8).view(tokens, k)
    sentinel = 0xA5
    out = torch.full((routed_ids.numel(), 16), sentinel, device="cuda", dtype=torch.uint8)
    args = (
        a.view(torch.int8).view(-1),
        routed_ids,
        out.view(torch.int8).view(-1),
        tokens,
        routed_ids.numel(),
        torch.cuda.current_stream(),
    )
    launcher = compile_routing_buffer_probe(k)
    kernel = flyc.compile[{"opt_level": 2}](launcher, *args)
    kernel(*args)
    torch.cuda.synchronize()

    decoded = (sorted_ids & TOKEN_MASK).to(torch.int64)
    torch.testing.assert_close(out[: sorted_ids.numel()], a[decoded, :16], rtol=0, atol=0)
    assert torch.all(out[sorted_ids.numel() :] == sentinel)
    print("routing buffer probe: PASS")


def run_accuracy_case(
    tokens: int,
    intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
) -> bool:
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
        inputs["weight"].view(torch.int8).view(-1),
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
        intermediate_size, hidden_size, topk, num_experts
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
        f"expert_blocks={inputs['expert_ids'].numel()} finite={finite} "
        f"allclose={allclose} max_abs={max_abs:.6g} diff={diff:.6g}"
    )
    return correct


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
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.75,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    b, scale_b_raw = per_1x32_mx_quant_hip(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 3.0,
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
        inputs["weight"].view(torch.int8).view(-1),
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
        intermediate_size, hidden_size, topk, num_experts
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
        "moe_gateup",
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
        ),
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    gemm_b, gemm_scale_b_raw = per_1x32_mx_quant_hip(
        torch.randn(
            (gemm_n, hidden_size), device="cuda", dtype=torch.bfloat16
        ),
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
        f"benchmark: clones={data_clones} warmup={warmup} runs={iterations}"
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
    parser.add_argument("--routing-probe", action="store_true")
    parser.add_argument("--small-accuracy", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--unaligned-accuracy", action="store_true")
    parser.add_argument("--scale-padding-accuracy", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-clones", type=int, default=10)
    args = parser.parse_args()
    if args.routing_probe:
        run_routing_probe()
        return
    if args.small_accuracy:
        if not run_accuracy_case(96, 128, 512, 2, 3):
            raise SystemExit("small padded accuracy failed")
        return
    if args.accuracy:
        if not run_accuracy_case(
            args.tokens,
            args.intermediate_size,
            args.hidden_size,
            args.topk,
            args.num_experts,
        ):
            raise SystemExit("full accuracy failed")
        return
    if args.unaligned_accuracy:
        if not run_accuracy_case(8193, 512, 6144, 8, 384):
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
        )
        return
    parser.error(
        "select --routing-probe, --small-accuracy, --accuracy, "
        "--unaligned-accuracy, --scale-padding-accuracy, or --benchmark"
    )


if __name__ == "__main__":
    props = torch.cuda.get_device_properties()
    assert "950" in props.gcnArchName, "MFMA_Scale requires gfx950"
    torch.manual_seed(0)
    main()
