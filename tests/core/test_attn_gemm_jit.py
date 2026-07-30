"""PyHIP JIT 版融合 BF16 attention，以及同 wave 的 mt0/mt1 精确交织。

固定配置：gfx942、D=128、BM=128、BN=32。一个 256-thread workgroup 包含
4 个 wave，每个 wave 负责 32 行 query，即两个独立的16行 ``mt``。
机器码同时交织 ``softmax(mt0) -> GEMM1(mt1)`` 与
``softmax(mt1) -> GEMM2(mt0)``，并用独立MFMA/VMEM/VALU隐藏softmax归约等待。

最终40960性能口径：默认``production``为208.6--208.8T；独立
``setprio_best``为236.5--237.1T；``setprio_best_all_vgpr``静态加载同一
归档ISA的全VGPR/Fly-ABI变体，约236.6T。三者随机输入``rel_l2≈0.00319``。
"""

import math
import os
from pathlib import Path
from typing import Any, cast

import torch

import pyhip

BM = 128
BN = 32
D = 128
WAVES = 4
THREADS = WAVES * 64
PREPARE_MFMAS = 4
CENTER_MFMAS = 3
FINISH_MFMAS = 9
LOG2E = math.log2(math.e)
VOID_POINTER = "void*"


@pyhip.jit("-g")  # pyright: ignore[reportAttributeAccessIssue]
def attn_gemm_jit(
    J: pyhip.JIT,  # pyright: ignore[reportAttributeAccessIssue]
    M,
    N,
    query: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    key: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    value_shuffled: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    assert D == 128
    assert M % BM == 0
    assert N % BN == 0
    use_best_setprio = J.special_vars.get("attn_best_setprio", False)

    sizeof_bf16 = 2
    tile_bytes = BN * D * sizeof_bf16
    lane_id = J.gpr(J.threadIdx.x[0] & 63)
    lane_mod_16 = J.gpr(lane_id & 15)
    lane_div_16 = J.gpr(lane_id >> 4)
    xor32_byte_address = J.gpr("vu32", (lane_id ^ 32) * 4)
    xor48_byte_address = J.gpr("vu32", (lane_id ^ 48) * 4)
    wave_id = J.gpr("su32")
    J.v_readfirstlane_b32(wave_id, J.threadIdx.x[0] >> 6)

    query_row = J.gpr("su32", J.blockIdx.x[0] * BM + wave_id * 32)
    head_qo_bytes = J.blockIdx.y[0] * (M * D * sizeof_bf16)
    head_kv_bytes = J.blockIdx.y[0] * (N * D * sizeof_bf16)
    query[:] += head_qo_bytes + query_row * (D * sizeof_bf16)
    output[:] += head_qo_bytes + query_row * (D * sizeof_bf16)
    key[:] += head_kv_bytes
    value_shuffled[:] += head_kv_bytes

    query_buf = J.Buffer(query, 32 * D * sizeof_bf16)
    key_buf = J.Buffer(key, N * D * sizeof_bf16)
    value_buf = J.Buffer(value_shuffled, N * D * sizeof_bf16)
    output_buf = J.Buffer(output, 32 * D * sizeof_bf16)

    # Q:每个 lane 读取 16-byte；lane_mod_16 选择 query 行，lane_div_16 选择 D 分片。
    query_reg = J.gpr(2, 16, "au32")
    query_voffset = J.gpr(2, "vu32")
    query_voffset[0] = lane_mod_16 * (D * sizeof_bf16) + lane_div_16 * 16
    query_voffset[1] = query_voffset[0] + 16 * D * sizeof_bf16
    for mt in range(2):
        for k in range(D // 32):
            query_buf.load_dwordx4(
                query_reg[mt, 4 * k : 4 * k + 3],
                query_voffset[mt],
                0,
                offset12=k * 64,
            )

    # K由4 waves协作预取，经swizzled LDS双缓冲广播；V仍直接global读取。
    key_reg = J.gpr(8, 4, "vu32")
    value_reg = J.gpr(8, 4, "au32")
    value_voffset0 = J.gpr(lane_id * 16)
    value_voffset1 = J.gpr(value_voffset0 + 4096)
    key_prefetch = J.gpr(2, 2, 4, "vu32")
    key_copy_voffset0 = J.gpr("vu32", J.threadIdx.x[0] * 16)
    key_copy_voffset1 = J.gpr("vu32", key_copy_voffset0 + 4096)
    key_write_addr0 = J.gpr(
        "vu32", key_copy_voffset0 ^ ((key_copy_voffset0 & 0x380) >> 3)
    )
    key_write_addr1 = J.gpr("vu32", key_write_addr0 + 4096)
    key_read_base = J.gpr("vu32", lane_id * 16)
    key_read_base[0] = key_read_base ^ ((key_read_base & 0x380) >> 3)
    key_lds = J.alloc_lds(2 * tile_bytes)

    score = J.gpr(2, 2, 4, "vf32", align=4)
    out = J.gpr(2, D // 16, 4, "vf32", align=4)
    out[:] = 0.0

    running_max = J.gpr(2, "vf32")
    running_sum = J.gpr(2, "vf32")
    running_max[:] = torch.finfo(torch.float).min
    running_sum[:] = 0.0

    scale_log2 = J.gpr("vf32", (1.0 / math.sqrt(D)) * LOG2E)
    round_bias = J.gpr("vu32", 0x8000)
    one = J.gpr("vf32", 1.0)
    lazy_delta = J.gpr("vf32", 8.0 / ((1.0 / math.sqrt(D)) * LOG2E))
    pack_bf16 = J.get_sgpr_const(0x03_02_07_06)

    def load_value_generator(soffset):
        for index in range(4):
            value_buf.load_dwordx4(
                value_reg[index],
                value_voffset0,
                soffset,
                offset12=index * 1024,
            )
            yield 1
        for index in range(4, 8):
            value_buf.load_dwordx4(
                value_reg[index],
                value_voffset1,
                soffset,
                offset12=(index - 4) * 1024,
            )
            yield 1

    def prefetch_key_generator(bank, soffset):
        key_buf.load_dwordx4(key_prefetch[bank, 0], key_copy_voffset0, soffset)
        yield 1
        key_buf.load_dwordx4(key_prefetch[bank, 1], key_copy_voffset1, soffset)
        yield 1

    def prefetch_key(bank, soffset):
        J.emit(prefetch_key_generator(bank, soffset))

    def write_key_lds(bank, stage):
        stage_base = key_lds + stage * tile_bytes
        J.ds_write_b128(key_write_addr0 + stage_base, key_prefetch[bank, 0])
        J.ds_write_b128(key_write_addr1 + stage_base, key_prefetch[bank, 1])

    def write_key_lds_generator(bank):
        J.ds_write_b128(key_write_addr0, key_prefetch[bank, 0])
        yield 4
        J.ds_write_b128(key_write_addr1, key_prefetch[bank, 1])
        yield 4

    def read_key_generator(stage):
        stage_base = key_lds + stage * tile_bytes
        for index in range(8):
            J.ds_read_b128(
                key_reg[index],
                key_read_base,
                mod=f"offset:{stage_base + index * 1024}",
            )
            yield 4

    def gemm1_mt_generator(mt, progressive_key_wait=False):
        # 每个mt是完整16条GEMM1 MFMA链：K[32,128] @ Q[16,128]^T。
        waited_key = -1
        for k_half in range(D // 16):
            k_block = k_half // 2
            half = k_half & 1
            for n_block in range(BN // 16):
                key_index = 4 * n_block + k_block
                if progressive_key_wait and key_index > waited_key:
                    J.s_waitcnt(mod=f"lgkmcnt({7 - key_index})")
                    waited_key = key_index
                accumulator = 0 if k_half == 0 else score[mt, n_block]
                key_lo = half * 2
                query_lo = 4 * k_block + half * 2
                J.v_mfma_f32_16x16x16_bf16(
                    score[mt, n_block],
                    key_reg[key_index, key_lo : key_lo + 1],
                    query_reg[mt, query_lo : query_lo + 1],
                    accumulator,
                )
                yield 16

    def make_softmax_state():
        return {
            "tile_max": J.gpr("vf32"),
            "reduce_tmp": J.gpr(3, "vf32"),
            "threshold": J.gpr("vf32"),
            "new_max": J.gpr("vf32"),
            "correction_exp": J.gpr("vf32"),
            "correction": J.gpr("vf32"),
            "partial_sum": J.gpr("vf32"),
            "correction_pk": J.gpr(2, "vf32"),
        }

    def softmax_prepare_generator(
        mt,
        state,
        independent_loads=None,
        before_wait_work=None,
    ):
        tile_max = state["tile_max"]
        reduce_tmp = state["reduce_tmp"]
        J.v_max3_f32(
            tile_max,
            score[mt, 0, 0],
            score[mt, 0, 1],
            score[mt, 0, 2],
        )
        yield 5
        J.v_max3_f32(
            tile_max,
            tile_max,
            score[mt, 0, 3],
            score[mt, 1, 0],
        )
        yield 5
        J.v_max3_f32(
            tile_max,
            tile_max,
            score[mt, 1, 1],
            score[mt, 1, 2],
        )
        yield 5
        J.v_max_f32(tile_max, tile_max, score[mt, 1, 3])
        yield 4
        J.ds_swizzle_b32(reduce_tmp[0], tile_max, mod="offset:swizzle(SWAP,16)")
        J.ds_bpermute_b32(reduce_tmp[1], xor32_byte_address, tile_max)
        J.ds_bpermute_b32(reduce_tmp[2], xor48_byte_address, tile_max)
        yield 12
        J.v_add_f32(state["threshold"], running_max[mt], lazy_delta)
        yield 4
        if independent_loads is not None:
            for _ in range(2):
                load_cycles = next(independent_loads, None)
                if load_cycles is not None:
                    yield load_cycles
        pending_writes = 0
        if before_wait_work is not None:
            for _ in range(2):
                work_cycles = next(before_wait_work, None)
                if work_cycles is not None:
                    pending_writes += 1
                    yield work_cycles
        J.s_waitcnt(mod=f"lgkmcnt({pending_writes})")
        yield 4
        J.v_max3_f32(tile_max, tile_max, reduce_tmp[0], reduce_tmp[1])
        yield 4
        J.v_max_f32(tile_max, tile_max, reduce_tmp[2])
        yield 4
        J.SetMask("vcc", tile_max > state["threshold"])
        yield 4
        J.v_cndmask_b32_e32(state["new_max"], running_max[mt], tile_max, "vcc")
        yield 4

    def softmax_center_generator(mt, state):
        scaled_new_max = state["threshold"]
        J.v_mul_f32(scaled_new_max, state["new_max"], scale_log2)
        yield 4
        J.v_fma_f32(
            state["correction_exp"],
            running_max[mt],
            scale_log2,
            -scaled_new_max[0],
        )
        yield 5
        for n_block in range(BN // 16):
            for item in range(4):
                J.v_fma_f32(
                    score[mt, n_block, item],
                    score[mt, n_block, item],
                    scale_log2,
                    -scaled_new_max[0],
                )
                yield 5

    def softmax_exp(mt, state, correction_already_exp=False):
        if not correction_already_exp:
            J.v_exp_f32(state["correction_exp"], state["correction_exp"])
        for n_block in range(BN // 16):
            for item in range(4):
                J.v_exp_f32(
                    score[mt, n_block, item],
                    score[mt, n_block, item],
                )
        J.v_cndmask_b32_e32(state["correction"], one, state["correction_exp"], "vcc")

    def pack_probability_block(mt, n_block):
        for item in range(4):
            J.v_add_u32(
                score[mt, n_block, item],
                score[mt, n_block, item],
                round_bias,
            )
        J.v_perm_b32(
            score[mt, n_block, 0],
            score[mt, n_block, 0],
            score[mt, n_block, 1],
            pack_bf16,
        )
        J.v_perm_b32(
            score[mt, n_block, 1],
            score[mt, n_block, 2],
            score[mt, n_block, 3],
            pack_bf16,
        )

    def softmax_finish_generator(mt, state):
        partial_sum = state["partial_sum"]
        reduce_tmp = state["reduce_tmp"]
        J.v_add_f32(partial_sum, score[mt, 0, 0], score[mt, 0, 1])
        yield 4
        for n_block, item in (
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
        ):
            J.v_add_f32(partial_sum, partial_sum, score[mt, n_block, item])
            yield 4
        J.ds_swizzle_b32(
            reduce_tmp[0],
            partial_sum,
            mod="offset:swizzle(SWAP,16)",
        )
        J.ds_bpermute_b32(reduce_tmp[1], xor32_byte_address, partial_sum)
        J.ds_bpermute_b32(reduce_tmp[2], xor48_byte_address, partial_sum)
        yield 12
        J.v_mov_b32(running_max[mt], state["new_max"])
        yield 4
        state["correction_pk"][0] = state["correction"]
        yield 4
        state["correction_pk"][1] = state["correction"]
        yield 4
        # 一个n_block正好填充三路DS延迟；全部前移会延长sum关键路径。
        pack_probability_block(mt, 0)
        yield 48
        J.s_waitcnt(mod="lgkmcnt(0)")
        yield 4
        J.v_add_f32(partial_sum, partial_sum, reduce_tmp[0])
        yield 4
        J.v_add_f32(partial_sum, partial_sum, reduce_tmp[1])
        yield 4
        J.v_add_f32(partial_sum, partial_sum, reduce_tmp[2])
        yield 4
        J.v_fma_f32(
            running_sum[mt],
            running_sum[mt],
            state["correction"],
            partial_sum,
        )
        yield 5
        for n_block in range(1, BN // 16):
            for item in range(4):
                J.v_add_u32(
                    score[mt, n_block, item],
                    score[mt, n_block, item],
                    round_bias,
                )
                yield 4
            J.v_perm_b32(
                score[mt, n_block, 0],
                score[mt, n_block, 0],
                score[mt, n_block, 1],
                pack_bf16,
            )
            yield 4
            J.v_perm_b32(
                score[mt, n_block, 1],
                score[mt, n_block, 2],
                score[mt, n_block, 3],
                pack_bf16,
            )
            yield 4

    def softmax_rescale(mt, state):
        with J.ExecMask(state["correction"] < 1.0):
            for d_block in range(D // 16):
                J.v_pk_mul_f32(
                    out[mt, d_block, 0:1],
                    out[mt, d_block, 0:1],
                    state["correction_pk"],
                )
                J.v_pk_mul_f32(
                    out[mt, d_block, 2:3],
                    out[mt, d_block, 2:3],
                    state["correction_pk"],
                )

    def interleave_softmax0_gemm1(state, independent_loads=None):
        mfma = gemm1_mt_generator(1)
        prepare = softmax_prepare_generator(0, state, independent_loads)
        for _ in range(PREPARE_MFMAS):
            J.emit(mfma, 16)
            J.emit(prepare, 10)
        J.emit(prepare)

        center = softmax_center_generator(0, state)
        J.emit(mfma, 16)
        J.emit(center, 10)
        J.emit(mfma, 16)
        J.emit(center, 12)
        J.emit(mfma, 16)
        J.v_exp_f32(state["correction_exp"], state["correction_exp"])
        J.emit(center)
        softmax_exp(0, state, correction_already_exp=True)

        finish = softmax_finish_generator(0, state)
        for _ in range(FINISH_MFMAS):
            J.emit(mfma, 16)
            J.emit(finish, 12)
        J.emit(mfma)
        J.emit(finish)
        softmax_rescale(0, state)

    def interleave_softmax1_gemm2(state, prepare):
        mfma = gemm2_mt_generator(0)
        if use_best_setprio:
            mfma = priority_wrap_mfma_generator(mfma, priority_end=15)
        for _ in range(PREPARE_MFMAS):
            J.emit(mfma, 16)
            J.emit(prepare, 10)
        J.emit(prepare)

        center = softmax_center_generator(1, state)
        J.emit(mfma, 16)
        J.emit(center, 10)
        J.emit(mfma, 16)
        J.emit(center, 12)
        J.emit(mfma, 16)
        J.v_exp_f32(state["correction_exp"], state["correction_exp"])
        J.emit(center)
        softmax_exp(1, state, correction_already_exp=True)

        finish = softmax_finish_generator(1, state)
        for _ in range(FINISH_MFMAS):
            J.emit(mfma, 16)
            J.emit(finish, 12)
        J.emit(mfma)
        J.emit(finish)
        softmax_rescale(1, state)

    def gemm2_mt_generator(mt):
        # 每个mt是完整16条GEMM2 MFMA链：V[128,32] @ P[16,32]^T。
        for n_block in range(BN // 16):
            for d_block in range(D // 16):
                value_slice = (
                    value_reg[d_block, 0:1] if n_block == 0 else value_reg[d_block, 2:3]
                )
                J.v_mfma_f32_16x16x16_bf16(
                    out[mt, d_block],
                    value_slice,
                    score[mt, n_block, 0:1],
                    out[mt, d_block],
                )
                yield 16

    # prologue：K0写stage0，K1保持在预取bank1。
    key_tile1_soffset = J.gpr("su32", tile_bytes)
    prefetch_key(0, 0)
    J.s_waitcnt(mod="vmcnt(0)")
    write_key_lds(0, 0)
    prefetch_key(1, key_tile1_soffset)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_barrier()
    J.emit(read_key_generator(0))

    tile_count = N // BN
    pair_base = J.gpr("su32", 0)
    odd_value_soffset = J.gpr("su32", tile_bytes)
    even_next_key_soffset = J.gpr("su32", 2 * tile_bytes)
    odd_next_key_soffset = J.gpr("su32", 3 * tile_bytes)

    def priority_wrap_mfma_generator(
        mfma, priority_start=None, priority_end=None, count=16
    ):
        for index in range(count):
            cycles = next(mfma, None)
            if cycles is None:
                return
            if priority_start is not None and index == priority_start:
                J.s_setprio(1)
            if priority_end is not None and index == priority_end:
                J.s_setprio(0)
            yield cycles

    def compute_tile(
        write_bank, write_stage, next_bank, value_soffset, next_key_soffset
    ):
        value_loads = load_value_generator(value_soffset)
        mt0 = gemm1_mt_generator(0, progressive_key_wait=True)
        if use_best_setprio:
            mt0 = priority_wrap_mfma_generator(mt0, priority_start=7)
        # 每tile切换一次K写stage；两条地址XOR分散到GEMM1的空MFMA窗口。
        for group in range(8):
            J.emit(value_loads, 1)
            if group == 1:
                J.emit(mt0, 16)
                J.v_xor_b32(key_write_addr0, tile_bytes, key_write_addr0)
                J.emit(mt0, 16)
            elif group == 3:
                J.emit(mt0, 16)
                J.v_xor_b32(key_write_addr1, tile_bytes, key_write_addr1)
                J.emit(mt0, 16)
            elif group == 7:
                J.emit(mt0, 16)
                value_soffset[0] = value_soffset + 2 * tile_bytes
                J.emit(mt0, 16)
            else:
                J.emit(mt0, 32)
        J.emit(value_loads)
        J.emit(mt0)
        future_key_loads = prefetch_key_generator(next_bank, next_key_soffset)
        interleave_softmax0_gemm1(make_softmax_state(), future_key_loads)
        J.emit(future_key_loads)
        # pending K+1是最早2条VMEM；当前V随后8条，future K最后2条。
        key_writes = write_key_lds_generator(write_bank)
        softmax1_state = make_softmax_state()
        softmax1_prepare = softmax_prepare_generator(
            1,
            softmax1_state,
            before_wait_work=key_writes,
        )
        J.s_waitcnt(mod="vmcnt(10)")
        # score不依赖V；在更严格的vmcnt(2)前先做lane-local max以覆盖VMEM等待。
        J.emit(softmax1_prepare, 19)
        J.s_waitcnt(mod="vmcnt(2)")
        # 较新的K写不阻塞当前归约；lgkmcnt(1)只等待更早的DS reduce。
        interleave_softmax1_gemm2(
            softmax1_state,
            softmax1_prepare,
        )
        J.emit(key_writes)
        J.s_waitcnt(mod="lgkmcnt(0)")
        J.s_barrier()
        key_reads = read_key_generator(write_stage)
        mt1 = gemm2_mt_generator(1)
        # 下一K读与当前GEMM2独立；放到每组两条MFMA之间以填充第一条MFMA的shadow。
        for _ in range(8):
            J.emit(mt1, 16)
            J.emit(key_reads, 4)
            J.emit(mt1, 16)
        J.emit(key_reads)
        J.emit(mt1)

    # tile_count固定为正偶数；尾部条件分支比while少一条每轮无条件branch。
    J.Label("attn_pair_loop")
    # 偶tile用stage0；写K1到stage1，同时预取K2到bank0。
    compute_tile(1, 1, 0, pair_base, even_next_key_soffset)

    # 奇tile用stage1；末轮越界K预取由buffer descriptor返回0且不再消费。
    compute_tile(0, 0, 1, odd_value_soffset, odd_next_key_soffset)
    even_next_key_soffset[0] = even_next_key_soffset + 2 * tile_bytes
    odd_next_key_soffset[0] = odd_next_key_soffset + 2 * tile_bytes
    J.Jump("attn_pair_loop", pair_base < tile_count * tile_bytes)

    inverse_sum = J.gpr(2, "vf32")
    for mt in range(2):
        J.v_rcp_f32(inverse_sum[mt], running_sum[mt])
        for d_block in range(D // 16):
            for item in range(4):
                J.v_mul_f32(
                    out[mt, d_block, item],
                    out[mt, d_block, item],
                    inverse_sum[mt],
                )
                J.v_add_u32(out[mt, d_block, item], out[mt, d_block, item], round_bias)

    # 将MFMA的D×M寄存器布局转成O[M,D]；映射与test_pa_sv中的FP32写回一致。
    out_transposed = J.gpr(2, D // 16, 4, "vf32")
    for mt in range(2):
        for half in range(D // 64):
            J.transpose_per_lane(
                4,
                4,
                4,
                out[mt, half * 4 : half * 4 + 3],
                out_transposed[mt, half * 4 : half * 4 + 3],
            )

    output_voffset = J.gpr(
        lane_mod_16 * (D * sizeof_bf16) + lane_div_16 * (4 * sizeof_bf16)
    )
    for mt in range(2):
        mt_output_voffset = J.gpr("vu32", output_voffset + mt * 16 * D * sizeof_bf16)
        for half in range(D // 64):
            for component in range(4):
                packed = J.gpr(2, "vu32")
                J.v_perm_b32(
                    packed[0],
                    out_transposed[mt, half * 4 + 0, component],
                    out_transposed[mt, half * 4 + 1, component],
                    pack_bf16,
                )
                J.v_perm_b32(
                    packed[1],
                    out_transposed[mt, half * 4 + 2, component],
                    out_transposed[mt, half * 4 + 3, component],
                    pack_bf16,
                )
                logical_column = half * 64 + component * 16
                output_buf.store_dwordx2(
                    packed,
                    mt_output_voffset,
                    0,
                    offset12=logical_column * sizeof_bf16,
                )


@pyhip.jit("-g")  # pyright: ignore[reportAttributeAccessIssue]
def attn_gemm_jit_setprio_best(
    J: pyhip.JIT,  # pyright: ignore[reportAttributeAccessIssue]
    M,
    N,
    query: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    key: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    value_shuffled: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
    output: VOID_POINTER,  # pyright: ignore[reportInvalidTypeForm]
):
    J.special_vars["attn_best_setprio"] = True
    attn_gemm_jit.gen_func(J, M, N, query, key, value_shuffled, output)


def reference(query, key, value):
    scores = torch.einsum("hmd,hnd->hmn", query.float(), key.float()) / math.sqrt(D)
    probability = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    return torch.einsum("hmn,hnd->hmd", probability.float(), value.float()).to(
        torch.bfloat16
    )


def preshuffle_key(key):
    """把逻辑K[H,N,D]变成PA MFMA读取的每32-token物理布局。"""
    H, N, _ = key.shape
    tile_count = N // BN
    grouped = key.reshape(H, tile_count, 8, 4, D)
    inverse_order = (0, 2, 4, 6, 1, 3, 5, 7)
    grouped = grouped[:, :, inverse_order]
    return (
        grouped.reshape(H, tile_count, 2, 16, D)
        .reshape(H, tile_count, 2, 16, D // 8, 8)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )


def preshuffle_value(value):
    """把逻辑V[H,N,D]变成PA GEMM2读取的[D/16,4,16,8]布局。"""
    H, N, _ = value.shape
    tile_count = N // BN
    return (
        value.reshape(H, tile_count, 4, 8, D // 16, 16)
        .permute(0, 1, 4, 2, 5, 3)
        .contiguous()
    )


def run_case(H, M, N, check=True, benchmark=False):
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    if os.environ.get("INPUT") == "ones":
        query = torch.ones(H, M, D, dtype=torch.bfloat16)
        key = torch.ones(H, N, D, dtype=torch.bfloat16)
        value = torch.ones(H, N, D, dtype=torch.bfloat16)
    else:
        query = torch.randn(H, M, D, dtype=torch.bfloat16)
        key = torch.randn(H, N, D, dtype=torch.bfloat16)
        value = torch.randn(H, N, D, dtype=torch.bfloat16)
    key_shuffled = preshuffle_key(key)
    value_shuffled = preshuffle_value(value)
    output = torch.empty_like(query)
    kernel_name = os.environ.get("ATTN_JIT_KERNEL", "production")
    static_kernel = None
    if kernel_name == "production":
        kernel = cast(Any, attn_gemm_jit)
    elif kernel_name == "setprio_best":
        kernel = cast(Any, attn_gemm_jit_setprio_best)
    elif kernel_name == "setprio_best_all_vgpr":
        from pyhip.core.fly_isa_priority import (  # pyright: ignore[reportMissingImports]
            build_all_vgpr_jit_attention_kernel,
        )

        root = Path(__file__).resolve().parents[2]
        static_kernel, artifact = build_all_vgpr_jit_attention_kernel(
            root
            / "archive/gemm/attn-gemm-jit-setprio-best-gfx942-m40960-n40960-237p1t.s",
            root / ".cache/jit-attn-all-vgpr",
            m=M,
            n=N,
            h=H,
        )
        kernel = None
        print(
            f"[all-vgpr] assembly={artifact.assembly_path} code_object={artifact.code_object_path}"
        )
    else:
        raise ValueError(f"unsupported ATTN_JIT_KERNEL={kernel_name!r}")

    def launch(q, k, v, out):
        if static_kernel is not None:
            static_kernel(q, k, v, out)
            return
        assert kernel is not None
        kernel(
            [M // BM, H],
            [THREADS],
            M,
            N,
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
        )

    launch(query, key_shuffled, value_shuffled, output)
    torch.cuda.synchronize()
    if check:
        expected = reference(query, key, value)
        difference = (output.float() - expected.float()).abs()
        relative_l2 = difference.norm() / expected.float().norm().clamp_min(1e-6)
        print(
            f"[acc] max_abs={difference.max().item():.4f} "
            f"mean_abs={difference.mean().item():.5f} rel_l2={relative_l2.item():.5f}"
        )
        if os.environ.get("DEBUG_VALUES") == "1":
            print(
                "[values]",
                f"min={output.min().item():.4f}",
                f"max={output.max().item():.4f}",
                f"mean={output.float().mean().item():.4f}",
                f"row0={output[0, 0, :32].float().cpu()}",
            )
        assert relative_l2.item() <= 0.0035

    if benchmark:
        buffers = []
        for _ in range(10):
            q = torch.randn_like(query)
            k = torch.randn_like(key)
            v = torch.randn_like(value)
            buffers.append(
                (
                    q,
                    preshuffle_key(k),
                    preshuffle_value(v),
                    torch.empty_like(output),
                )
            )
        flops = H * 4 * M * N * D
        for index in range(10):
            launch(*buffers[index])
        torch.cuda.synchronize()
        samples = []
        for index in range(50):
            with pyhip.cudaPerf(  # pyright: ignore[reportAttributeAccessIssue]
                flops, name="attn_jit", verbose=0
            ) as perf:
                launch(*buffers[index % len(buffers)])
            samples.append((perf.dt() * 1e6, perf.tflops()))
        samples.sort()
        microseconds, tflops = samples[len(samples) // 2]
        print(
            f"[perf] kernel={kernel_name} "
            f"schedule={PREPARE_MFMAS}/{CENTER_MFMAS}/{FINISH_MFMAS} "
            "k_write=before-wait reduce=fanout "
            f"{microseconds:.1f} us {tflops:.1f} TFLOPS"
        )


def test_attention_jit():
    run_case(H=1, M=256, N=256)


if __name__ == "__main__":
    H = int(os.environ.get("H", "1"))
    MULT = int(os.environ.get("MULT", "2"))
    M = BM * MULT
    N = BM * MULT
    check_setting = os.environ.get("CHECK", "auto").lower()
    if check_setting == "auto":
        free_bytes, _ = torch.cuda.mem_get_info()
        reference_bytes = 3 * H * M * N * torch.float32.itemsize
        check = reference_bytes <= free_bytes
        if not check:
            print(
                f"[acc] skipped: estimated reference workspace {reference_bytes / 2**30:.1f} GiB "
                f"exceeds {free_bytes / 2**30:.1f} GiB free; set CHECK=1 to force"
            )
    elif check_setting in ("0", "false", "off"):
        check = False
    elif check_setting in ("1", "true", "on"):
        check = True
    else:
        raise ValueError(f"unsupported CHECK={check_setting!r}; expected auto, 0, or 1")
    run_case(
        H=H,
        M=M,
        N=N,
        check=check,
        benchmark=True,
    )
