# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8x1生产实现：公共入口/原语、BK128倍数、K192、K320。"""

from functools import cache, partial

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec, as_ir_value
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh
from .common import BufferTensor, LdsTensor, get_down_device_config


# ==================== 公共入口与8x1共享实现 ====================


def _build_moe_gemm2_8x1(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    _task_table=False,
    _store_cache=2,
):
    # Host参数校验；各K族只共享任务映射与启动，不改变流水事件。
    del E, activation, swiglu_limit
    assert stage == "down"
    assert alg == "prefill_1x4"
    assert down_path == "8x1"
    assert weight_dtype == "fp8"
    assert weight_quant_type in ("ptpc", "per_tensor")
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    ), f"unsupported 8x1 quant combo (weight={weight_quant_type}, act={act_quant_type})"
    assert not USE_ATOMIC_WRITE
    assert BLOCK_TILE_SIZE_M == 256
    assert BLOCK_TILE_SIZE_N == 128
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert METADATA_TILE_SIZE_M == BLOCK_TILE_SIZE_M
    assert K in (192, 256, 320, 384, 512, 640), "8x1仅支持K=192/256/320/384/512/640"
    if tile_k is None:
        tile_k = 192 if K in (192, 320) else 128
    assert tile_k == 128 or (K in (192, 320) and tile_k == 192), "8x1仅保留K192整块192、K320的128+192，其余K使用BK128"
    # K192整块192、K320固定128+192；兼容调用者统一传入tile_k=128。
    if K in (192, 320):
        tile_k = 192
        assert N > 0
    else:
        assert K % tile_k == 0
    assert N % BLOCK_TILE_SIZE_N == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)
    assert _store_cache in (0, 2), "store aux: 0=普通，2=SLC"

    BM = 256
    ops = fxh.FlyObjCache()
    topology, xcc_count = get_down_device_config()
    se_count = xcc_count * 4

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32,
    ):
        # 按XCC/SE/CU映射任务；有效性判断必须早于两组4-wave错相。
        tid = fx.Int32(gpu.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        group = fx.Int32(rocdl.readfirstlane(ir.IntegerType.get_signless(32), _raw(tid // 256)))
        group_tid = tid % 256
        valid = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        wg = fx.Int32(gpu.block_idx.y)
        e_idx = wg
        if const_expr(topology):
            tasks = fxh.div_up(fx.Uint32(valid), BM)
            per_se = tasks // se_count
            mapped = per_se * se_count
            u = fx.Uint32(wg)
            xcc, local = u & (xcc_count - 1), u >> 2
            se, within = local & 3, local >> 2
            cu, round_id = within % 5, within // 5
            short, long = per_se // 5, per_se % 5
            rank = cu * short + arith.select(cu < long, cu, long) + round_id
            logical = ((xcc + 2) & (xcc_count - 1)) * (per_se * 4) + se * per_se + rank
            e_idx = fx.Int32(arith.select(u < mapped, logical, u))

        if e_idx * BM < valid:
            # K为统一的编译期选择；大段emit保持模块级JIT，不嵌入kernel定义。
            if const_expr(K == 192):
                _emit_k192_body(
                    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
                    p_w_scale, p_a_scale, M, e_idx, tid, lane, wave, group,
                    N, TOPK, down_output_padding_bytes, weight_quant_type, act_quant_type,
                    _task_table, _store_cache, ops,
                )
            elif const_expr(K == 320):
                _emit_k320_body(
                    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
                    p_w_scale, p_a_scale, M, e_idx, tid, lane, wave, group, group_tid,
                    N, TOPK, down_output_padding_bytes, weight_quant_type, act_quant_type,
                    _task_table, _store_cache, ops,
                )
            else:
                _emit_k128n_body(
                    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids,
                    p_w_scale, p_a_scale, M, e_idx, tid, lane, wave, group, group_tid,
                    N, K, TOPK, tile_k, down_output_padding_bytes, weight_quant_type, act_quant_type,
                    _task_table, _store_cache, ops,
                )

    @flyc.jit
    def launch_prefill_8x1(
        p_input: fx.Pointer, p_weight: fx.Pointer, p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer, p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer, p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer, p_a_scale: fx.Pointer, M: fx.Int32, task_num: fx.Int32, stream: fx.Stream,
    ):
        # 统一清理编译期缓存，并保留原ABI、launch形状与编译选项。
        CompilationContext.get_current()
        ops.clear_all()
        kernel = moe_2stage_down_prefill_8x1(
            p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights,
            p_sorted_expert_ids, p_num_valid_ids, p_w_scale, p_a_scale, M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(512, 1, 1), stream=stream)

    launch_prefill_8x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_8x1


# ==================== 8x1共享硬件/数学原语 ====================

# g=VMEM，r=寄存器，s=LDS；read表示发起异步读取，等待由流水线另行安排。


def stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def priority(value):
    rocdl.sched_barrier(0)
    rocdl.s_setprio(value)
    rocdl.sched_barrier(0)


def cshuffle_plane_offset(row, group, pair):
    # BF16元素偏移；source-group低位移到2KiB plane，保持128bit写。
    return ((row & 1) * 8 + ((group & 2) ^ (row & 2)) * 8
            + ((pair & 1) ^ ((row >> 2) & 1)) * 32 + (pair >> 1) * 64
            + (row >> 3) * 128 + (row & 6) * 128 + (group & 1) * 1024)


def read_b_g2r_packet(weights, byte_offset, scalar_offset, words):
    source = BufferTensor(
        fx.make_view(fx.get_iter(weights), fx.make_layout(words * 4, 1))
    )
    fragment = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)

    values = source.load(voffset_bytes=fx.Int32(byte_offset), soffset_bytes=fx.Int32(scalar_offset)).bitcast(fx.Uint32)
    fragment.store(values)
    return fragment


def read_a_g2r(a, ids_lds, afragments, copy_atom, lane, wave, k_widths, k_offsets):
    # BM=256，每wave两行组，组间跨度128；每次读取16B后拆为两个K8。
    packed_inputs = [
        [[fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
          for _ in range_constexpr(width // 64)] for _ in range_constexpr(2)]
        for width in k_widths
    ]

    for ks in range_constexpr(len(k_widths)):
        for row in range_constexpr(2):
            encoded = ids_lds[wave * 16 + row * 128 + lane % 16].bitcast(fx.Uint32)
            for k64 in range_constexpr(k_widths[ks] // 64):
                offset = k_offsets[ks] + k64 * 64 + (lane // 16) * 16
                # source依赖刚读出的路由，动态子视图只能在访存原语内就地建立。
                source = fxh.atom_tensor(a, (encoded & 0xFFFFFF, encoded >> 24, offset), 128)
                packed = packed_inputs[ks][row][k64]
                fx.copy(copy_atom, source, packed)
                values = Vec(packed.load())
                for k8 in range_constexpr(2):
                    part = values.shuffle(values, list(range(k8 * 8, k8 * 8 + 8)))
                    afragments[ks][None, row, (k8, k64)].store(part)


def read_8x1_scale_g2r(n, pair, *, weight_quant_type, scale_buffer, mm, ops):
    if const_expr(weight_quant_type == "ptpc"):
        # fx.copy的soffset单位为元素，lowering再乘4转为原生字节偏移。
        tensor = fx.make_view(fx.get_iter(scale_buffer) + pair * 32, fx.make_layout((32, 256), (1, 0)))
        # 保持每pair两条dwordx4；copy32不会自动合并，会破坏VMEM等待账本。
        copy_atom = ops.get_buffer_copy_atom(fx.Float32, 128)
        fragment = mm.make_fragment_C(tensor)

        fx.copy(copy_atom, ops.get_tiled_mma_partition_S(mm, tensor, "C", copy_atom_bits=128),
                ops.get_tiled_mma_retile(mm, fragment, "C", copy_atom=copy_atom),
                soffset=fx.Int32(n * 128))
        return fragment
    else:
        return fx.Float32(1.0)


def pack_8x1_record(pair, scale, *, c, row_scale, weight_quant_type):
    weighted, rows = [], []
    for row in range_constexpr(2):
        for ng in range_constexpr(2 * pair, 2 * pair + 2):
            if const_expr(weight_quant_type == "ptpc"):
                weighted.append(fxh.eltwise_op("v_fma_f32", Vec(c[None, ng, row].load()),
                                              Vec(scale[None, ng % 2, row].load()), fx.Float32(0.0)))
            else:
                weighted.append(Vec(c[None, ng, row].load()))
            rows.append(Vec(row_scale[None, ng, row].load()))
    bias = as_ir_value(fx.Uint32(0x8000)).bitcast(fx.Float32.ir_type)
    scaled = [fxh.eltwise_op("llvm.fma.f32", weighted[index], rows[index], bias) for index in range_constexpr(4)]
    selector = fx.Uint32(0x07060302)
    records = [[], []]
    for index in range_constexpr(4):
        for element in range_constexpr(0, scaled[index].numel, 2):
            records[index // 2].append(llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [_raw(scaled[index][element + 1]), _raw(scaled[index][element]), _raw(selector)],
                "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True,
            ))
    return [Vec.from_elements(record, fx.Uint32).bitcast(fx.BFloat16) for record in records]


def shuffle_8x1_c_r2s2r(
    n, packed, row, half, *, scratch_view, scratch_base, lane, lane_group, wave, out, scratch_write, scratch_read,
):
    for local_pair in range_constexpr(2):
        pair = half * 2 + local_pair
        offset = scratch_base + cshuffle_plane_offset(lane % 16, lane_group, pair)
        destination = fx.make_view(fx.get_iter(scratch_view) + offset, fx.make_layout(8, 1))
        fragment = fx.make_fragment_like(destination)
        fragment.store(packed[pair][row])
        fx.copy(scratch_write, fragment, destination)
    fragments, destinations = [], []
    for oh in range_constexpr(2):
        atom_index = half * 8 + lane % 8
        ng = atom_index // 2
        offset = scratch_base + cshuffle_plane_offset(oh * 8 + lane // 8, 2 * (atom_index % 2), ng // 2) + (ng % 2) * 4
        pieces = []
        for source_group in range_constexpr(2):
            source = fx.make_view(fx.get_iter(scratch_view) + offset + source_group * 1024, fx.make_layout(4, 1))
            fragment = fx.make_fragment_like(source)
            fx.copy(scratch_read, source, fragment)
            pieces.append(fragment)
        # 让同一输出的两段8B合并为read2st64(offset1:4)，禁止跨oh配对。
        fx.rocdl.sched_barrier(0)
        fragments.append(pieces)
        out_row = wave * 16 + row * 128 + oh * 8 + lane // 8
        destinations.append((n, fx.make_view(fx.get_iter(out) + out.layout(atom_index * 8, out_row),
                                            fx.make_layout(8, 1))))
    return fragments, destinations


def store_8x1_c_r2g(fragments, destinations, lgkmcnt=0, *, store_atom):
    rocdl.s_waitcnt(lgkmcnt=lgkmcnt)
    for index in range_constexpr(len(fragments)):
        first, second = Vec(fragments[index][0].load()), Vec(fragments[index][1].load())
        result = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        result.store(first.shuffle(second, list(range(8))))
        output_n, destination = destinations[index]
        # 线程内输出地址不含N循环变量；统一N偏移由SGPR soffset提供。
        fx.copy(store_atom, result, destination, soffset=fx.Int32(output_n * 128))


def store_8x1_c_tile_r2g(n, packed, *, shuffle_c_r2s2r, store_c_r2g):
    for half in range_constexpr(2):
        for row in range_constexpr(2):
            fragments, destinations = shuffle_c_r2s2r(n, packed, row, half)
            store_c_r2g(fragments, destinations)


def clear_8x1_record(pair, *, c):
    for row in range_constexpr(2):
        for ng in range_constexpr(pair * 2, pair * 2 + 2):
            c[None, ng, row].fill(0)


def mfma_8x1_record(weight, ks, pair, *, k_widths, afragments, c, atom, weight_n_group_begin=0):
    for ki in range_constexpr(k_widths[ks] // 64):
        for ka in range_constexpr(2):
            for row in range_constexpr(2):
                for ng in range_constexpr(2):
                    weight_piece = weight[None, weight_n_group_begin + ng, (ka, ki)]
                    activation = afragments[ks][None, row, (ka, ki)]
                    fx.mma_atom_call(
                        atom, c[None, pair * 2 + ng, row], weight_piece, activation, c[None, pair * 2 + ng, row],
                    )


def schedule_k128_pack(*, weight_quant_type):
    # PTPC为40条VALU，标量scale融合后为24条；均只放进BK128长packet。
    for index in range_constexpr(16):
        rocdl.sched_group_barrier(0x8, 1, 0)
        if const_expr(weight_quant_type == "ptpc"):
            if const_expr(index < 13):
                rocdl.sched_group_barrier(0x2, 3, 0)
            elif const_expr(index == 13):
                rocdl.sched_group_barrier(0x2, 1, 0)
        else:
            rocdl.sched_group_barrier(0x2, 2 if index < 8 else 1, 0)
    rocdl.sched_barrier(0)


# ==================== 8x1共享账本与SSA ====================


def packing_events(k, stage, first=False, k_widths=None):
    """返回(packet, previous-N, super-record)，不是当前MFMA的输出分片。"""
    assert k != 192, "K192 uses the independent K192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule不支持该K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    ks = (k + 127) // 128
    if k == 320:
        if stage == 0 and not first:
            return ((0, True, 2), (1, True, 3))
        if stage == 2:
            return ((0, False, 0), (1, False, 1))
    else:
        if stage == 0 and not first:
            return ((0, True, 3),)
        if stage == ks - 1:
            return ((1, False, 0),)
        if stage == ks:
            return ((0, False, 1),)
        if stage == 2 * ks - 1:
            return ((1, False, 2),)
    return ()


def output_quarter(k, stage, k_widths=None):
    assert k != 192, "K192 uses the independent K192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule不支持该K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    return stage if stage < 4 else None


@cache
def vmem_wait_schedule(k, n_tiles, ptpc=True, k_widths=None):
    """普通buffer/global每条指令一个事件；wait后才发起下一条B读取。

    预算取所有即将消费的B/scale的年龄最小值。不能只计算B的年龄：
    K256某些packet用的是上一拍scale，会把9收紧到7。
    K192使用独立账本；K320必须显式传入唯一分块(128, 192)。
    """
    assert k in (256, 320, 384, 512, 640) and n_tiles >= 1, "shared 8x1 schedule不支持该K或N块数"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    widths = k_widths or (128,) * (k // 128)
    ks, events, requests, scales = len(widths), [], {}, {}

    def valid(q):
        # 请求编号q代表提交Q[q+1]；只裁掉最终消费者之后的请求。
        return 0 <= q < n_tiles * 2 * ks - 1

    def request(q):
        if valid(q):
            # 末块192每lane24B，实际16B+8B两条VMEM；以最后一条约束提交。
            target_k = (q % ks + 1) % ks
            events.extend([("B", q)] * (2 if widths[target_k] == 192 else 1))
            requests[q] = len(events) - 1

    request(0)
    request(1)
    result = []
    for q in range(n_tiles * 2 * ks):
        n, stage = divmod(q, 2 * ks)
        scale_count = 2 if ptpc and stage < 4 else 0
        events.extend([("scale", n, stage)] * scale_count)
        if scale_count:
            scales[n, stage] = len(events) - 1
        store_count = 2 if n > 0 and output_quarter(k, stage, k_widths) is not None else 0
        events.extend([("store", n, stage)] * store_count)
        required = [requests[q]] if valid(q) else []
        if ptpc:
            for _, previous, record in packing_events(k, stage, n == 0, k_widths):
                required.append(scales[n - int(previous), record])
        budget = min((len(events) - 1 - index for index in required), default=63)
        result.append(budget)
        request(q + 2)
    return tuple(result)


def save_8x1_state(b_prefetch, packed, scales, addresses, *, c, ptpc, first_unpacked, prepare_b_addresses):
    state = [b_prefetch[index].load() for index in range_constexpr(2)]
    for row in range_constexpr(2):
        for group in range_constexpr(2 * first_unpacked, 8):
            state.append(c[None, group, row].load())
    if const_expr(ptpc):
        for pair in range_constexpr(first_unpacked, 4):
            state.append(scales[pair].load())
    for pair in range_constexpr(first_unpacked):
        for row in range_constexpr(2):
            state.append(packed[pair][row])
    if const_expr(prepare_b_addresses is not None):
        state.extend(addresses)
    return state


def restore_8x1_state(state, *, c, ptpc, first_unpacked, prepare_b_addresses, b_carriers, scale_carriers):
    for index in range_constexpr(2):
        b_carriers[index].store(state[index])
    offset = 2
    scales = []
    for row in range_constexpr(2):
        for group in range_constexpr(2 * first_unpacked, 8):
            c[None, group, row].store(state[offset])
            offset += 1
    for pair in range_constexpr(first_unpacked, 4):
        if const_expr(ptpc):
            scale_carriers[pair - first_unpacked].store(state[offset])
            offset += 1
            scales.append(scale_carriers[pair - first_unpacked])
        else:
            scales.append(fx.Float32(1.0))
    packed = []
    for pair in range_constexpr(first_unpacked):
        packed.append([Vec(state[offset]), Vec(state[offset + 1])])
        offset += 2
    addresses = [fx.Int32(value) for value in state[-4:]] if const_expr(prepare_b_addresses is not None) else []
    return list(b_carriers), packed, scales, addresses


# ==================== K=256/384/512/640：BK128倍数、双槽权重流水 ====================


@flyc.jit
def _emit_k128n_body(
    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids, p_w_scale, p_a_scale,
    M, e_idx, tid, lane, wave, group, group_tid,
    N, K, TOPK, tile_k, down_output_padding_bytes, weight_quant_type, act_quant_type,
    _task_table, _store_cache, ops,
):
    # 配置与原语
    # 形状/量化配置是host静态值；任务映射与线程坐标由kernel原样传入。
    BM = 256
    BN = 128
    BK = tile_k
    NUM_WAVES = 8
    WAVE_M = BM // NUM_WAVES
    KS = K // BK
    NT = N // BN
    K_WIDTHS = (BK,) * KS
    K_OFFSETS = tuple(ks * BK for ks in range(KS))
    FIRST_UNPACKED = 3
    LOOP_END = max(2, NT - 1)
    WEIGHT_QUARTER_ATOMS = BN * BK // (16 * 4)
    WEIGHT_PREFETCH_SLOTS = 2
    OUTPUT_STORES_PER_WAVE = WAVE_M * BN * 2 // (64 * 16)
    CSHUFFLE_N_PAIRS = BN // 32
    STRIDE = N + down_output_padding_bytes // 2
    use_n_loop = NT >= 3
    CSHUFFLE_WAVES = 4

    assert WEIGHT_QUARTER_ATOMS == 256
    assert OUTPUT_STORES_PER_WAVE == 8

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    input_tensor = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False, num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    sorted_ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
            (BM,), fx.Int32,
        ),
        max_size=False, num_records_bytes=BM * 4,
    )
    sorted_weights = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
        (BM,), fx.Float32,
    )
    if const_expr(_task_table):
        expert_id = p_sorted_expert_ids[2 * e_idx + 1]
    else:
        expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]

    shared_allocator = fx.SharedAllocator()
    if const_expr(use_n_loop):
        weight_slots = shared_allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, 2 * BN * BK, 16])
        weight_storage_ptrs = [weight_slots.peek().ptr, weight_slots.peek().ptr + BN * BK]
    else:
        weight_ping_storage = shared_allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * BK, 16])
        weight_pong_storage = shared_allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * BK, 16])
        weight_storage_ptrs = [weight_ping_storage.peek().ptr, weight_pong_storage.peek().ptr]
    cshuffle_storage = shared_allocator.allocate(fx.Array[fx.BFloat16, CSHUFFLE_WAVES * 16 * BN, 16])
    sorted_lds = fx.make_view(
        fx.recast_iter(fx.Int32, cshuffle_storage.peek().ptr), fx.make_layout(BM, 1),
    )
    if tid < BM:
        sorted_lds[tid] = sorted_ids[tid]
    gpu.barrier()

    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, NUM_WAVES, 1))
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c_fake = fx.make_view(fx.get_iter(input_tensor), fx.make_ordered_layout((BN, BM), (0, 1)))
    frag_c = mm.make_fragment_C(c_fake)
    row_tensor = fx.make_view(fx.get_iter(sorted_weights), fx.make_layout((BN, BM), (0, 1)))
    frag_row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coord_tensor = fx.make_view(fx.get_iter(sorted_lds), fx.make_layout((BN, BM), (0, 1)))
        frag_coord = ops.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        a_scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        frag_a_scale = mm.make_fragment_C(coord_tensor)
        frag_a_scale_retile = ops.get_tiled_mma_retile(mm, frag_a_scale, "C", copy_atom=a_scale_copy)
        for dst, coord in fxh.all_elements(frag_a_scale_retile, frag_coord):
            sorted_id = coord[0].bitcast(fx.Uint32)
            source = fxh.atom_tensor(a_scale_tensor, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32)
            fx.copy(a_scale_copy, source, dst)
        frag_row_scale.store(frag_row_scale.load() * frag_a_scale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            scalar_weight_scale = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + expert_id, fx.make_layout(1, 1),
            )[0]
            frag_row_scale.store(frag_row_scale.load() * scalar_weight_scale)
    else:
        # 每专家一个weight scale、全局一个activation scale，提前融合。
        scalar_a_scale = fx.make_view(fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1))[0]
        scalar_w_scale = fx.make_view(
            fxh._as_ptr(p_w_scale, fx.Float32) + expert_id, fx.make_layout(1, 1),
        )[0]
        frag_row_scale.store(frag_row_scale.load() * (scalar_a_scale * scalar_w_scale))

    weight_base = (
        fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight)) + fx.Int64(expert_id) * N * K
    )
    weight_view = fx.make_view(weight_base, fx.make_layout(N * K, 1))
    weight_flat = fx.rocdl.make_buffer_tensor(weight_view, max_size=False, num_records_bytes=N * K)
    weight_packet = BufferTensor(fx.make_view(fx.get_iter(weight_flat), fx.make_layout(16, 1)))
    weight_staging = [
        fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
        for _ in range_constexpr(WEIGHT_PREFETCH_SLOTS)
    ]
    weight_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FNUZ)

    def weight_lds_quarter_view(pointer, n_quarter):
        return fx.make_view(
            pointer + n_quarter * (BN // 4) * BK,
            fx.make_layout(((16, BN // 64), (16, BK // 16)), ((16, 16 * BK), (1, 256))),
        )

    # 整half布局只在此建立；短N的首tile仍按原布局读取，不改为quarter拼接。
    lds_weight_halves = [
        [fx.make_view(
            storage + n_half * (BN // 2) * BK,
            fx.make_layout(((16, BN // 32), (16, BK // 16)), ((16, 16 * BK), (1, 256))),
        ) for n_half in range_constexpr(2)]
        for storage in weight_storage_ptrs
    ]
    lds_weight_quarters = [
        [weight_lds_quarter_view(storage, n_quarter) for n_quarter in range_constexpr(4)]
        for storage in weight_storage_ptrs
    ]

    quarter_atom_index = group * WEIGHT_QUARTER_ATOMS + group_tid
    quarter_n_group = quarter_atom_index // BK
    quarter_within_group = quarter_atom_index % BK
    quarter_k_group = quarter_within_group // 16
    quarter_n_inner = quarter_within_group % 16
    weight_quarter_lane_offset_bytes = (
        quarter_n_group * (16 * K) + quarter_k_group * 256 + quarter_n_inner * 16
    )

    def read_b_g2r(n, kb, half, *, prefetch_slot=0):
        core_base_bytes = (
            n * (BN * K)
            + kb * (BK // 16) * 256
            + half * (BN // 2) * K
        )

        weight_staging[prefetch_slot].store(weight_packet.load(
            voffset_bytes=fx.Int32(weight_quarter_lane_offset_bytes), soffset_bytes=fx.Int32(core_base_bytes),
        ))
        return weight_staging[prefetch_slot]

    def store_b_r2s(lds_slot, kb, half, *, prefetch_slot=0, fragment=None, address=None):
        if const_expr(fragment is not None):
            weight_staging[prefetch_slot].store(fragment.load())
        if const_expr(address is not None):
            destination = LdsTensor(fx.make_view(b_write_pointer, fx.make_layout(16, 1)))
            destination.store(weight_staging[prefetch_slot], address_bytes=address,
                              offset_bytes=half * (BN // 2) * BK, copy_atom=weight_store_atom)
        else:
            n_group = quarter_atom_index // BK
            within_group = quarter_atom_index % BK
            k_group = within_group // 16
            n_inner = within_group % 16
            lds_offset = half * (BN // 2) * BK + n_group * (16 * BK) + k_group * 256 + n_inner * 16
            destination = fx.make_view(
                (weight_storage_ptrs[0] + lds_slot * BN * BK if const_expr(use_n_loop)
                 else weight_storage_ptrs[lds_slot]) + lds_offset,
                fx.make_layout(16, 1),
            )
            fx.copy(weight_store_atom, weight_staging[prefetch_slot], destination)

    def shuffle_c_r2s2r(block_n, packed_super_records, row_pair, n_half):
        packed_records = [
            packed_super_records[n_half * (CSHUFFLE_N_PAIRS // 2) + local_n_pair][row_pair]
            for local_n_pair in range_constexpr(CSHUFFLE_N_PAIRS // 2)
        ]
        # 先将两个record按原plane布局写入LDS，再发起同一输出行的两段8B读取。
        for local_n_pair in range_constexpr(CSHUFFLE_N_PAIRS // 2):
            n_pair = n_half * (CSHUFFLE_N_PAIRS // 2) + local_n_pair
            lds_offset = wave_lds_base + cshuffle_plane_offset(lane_row, lane_group, n_pair)
            destination = fx.make_view(fx.get_iter(cshuffle_lds) + lds_offset, fx.make_layout(8, 1))
            fragment = fx.make_fragment_like(destination)
            fragment.store(packed_records[local_n_pair])
            fx.copy(cshuffle_write_atom, fragment, destination)

        output_fragments, destinations = [], []
        for output_row_half in range_constexpr(2):
            fragment_pair = []
            for source_group in range_constexpr(2):
                source = fx.make_view(
                    cshuffle_read_pointers[2 * n_half + output_row_half] + source_group * 1024, fx.make_layout(4, 1),
                )
                fragment = fx.make_fragment_like(source)
                fx.copy(cshuffle_read_atom, source, fragment)
                fragment_pair.append(fragment)
            # 只合并同一输出行的两段8B，避免跨行配对产生额外搬运。
            fx.rocdl.sched_barrier(0)
            output_fragments.append(fragment_pair)
            destination_index = n_half * 2 + output_row_half
            destinations.append((block_n, fx.make_view(
                fx.get_iter(output_tensor) + output_destination_offsets[row_pair][destination_index],
                fx.make_layout(8, 1))))
        return output_fragments, destinations

    def b_startup_coords(half_core):
        # 启动只查q=0/1以预取Q1/Q2；每N至少4拍，两者始终有消费者。
        target = half_core + 1
        target_n = target // (2 * KS)
        target_k = target % KS
        return (
            target_n, target_k, (target % (2 * KS)) // KS,
            (target_n * KS + target_k) & 1,
        )

    def prepare_b_addresses(n):
        # 两种相对槽的读/写地址在上一N的compute尾部准备，并跨回边携带。
        parity = (n * KS) & 1
        read_base = fx.Int32(fx.ptrtoint(b_read_pointer))
        write_base = fx.Int32(fx.ptrtoint(b_write_pointer))
        values = [base + fx.Int32(((parity + relative_slot) & 1) * BN * BK)
                  for base in (read_base, write_base) for relative_slot in range_constexpr(2)]

        return [fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32), [_raw(value)],
                    "", "=v,0", has_side_effects=True)) for value in values]

    def read_b_s2r(lds_slot, kb, half, *, packet, address=None):
        if const_expr(packet is None):
            # N0首q0及短N首tile余拍消费整half，保留其读取/寄存器布局。
            return ops.load_tiled_mma_fragA(mm, lds_weight_halves[lds_slot][half], copy_atom_bits=128)
        if const_expr(address is not None):
            source = LdsTensor(b_partition)
            fragment = mm.make_fragment_A(b_template)
            source.load(address_bytes=address, offset_bytes=(2 * half + packet) * (BN // 4) * BK,
                        into=ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy), copy_atom=b_copy)
            return fragment
        if const_expr(use_n_loop):
            view = weight_lds_quarter_view(weight_storage_ptrs[0] + lds_slot * BN * BK, 2 * half + packet)
        else:
            view = lds_weight_quarters[lds_slot][2 * half + packet]
        return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

    # 闭包绑定视图和驻留A；跨N变化的SSA仍由显式state传递。
    a_fragments = output_tensor = cshuffle_lds = None
    cshuffle_write_atom = cshuffle_read_atom = output_store_atom = None
    lane_group = lane_row = wave_lds_base = None
    output_destination_offsets = cshuffle_read_bases = cshuffle_read_pointers = weight_scale_buffer = None
    b_template = b_copy = b_partition = b_read_pointer = b_write_pointer = None

    def prepare_views():
        nonlocal output_tensor, cshuffle_lds, cshuffle_write_atom, cshuffle_read_atom, output_store_atom
        nonlocal lane_group, lane_row, wave_lds_base, output_destination_offsets, cshuffle_read_bases, weight_scale_buffer
        nonlocal b_template, b_copy, b_partition, b_read_pointer, b_write_pointer
        if const_expr(prepare_addresses is not None):
            # B模板、partition和基指针在此集中准备；首N完整地址仍在prologue之后pin。
            b_template = weight_lds_quarter_view(weight_storage_ptrs[0], 0)
            b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
            b_partition = ops.get_tiled_mma_partition_S(mm, b_template, "A", copy_atom_bits=128)
            b_read_pointer = fx.get_iter(b_partition)
            b_write_pointer = (weight_storage_ptrs[0]
                + (quarter_atom_index // BK) * (16 * BK)
                + ((quarter_atom_index % BK) // 16) * 256
                + (quarter_atom_index % 16) * 16)

        output_base = (
            fxh._as_ptr(p_output, fx.BFloat16)
            + (fx.Int64(row_begin) * STRIDE if const_expr(_task_table)
               else fx.Int64(e_idx) * BM * STRIDE)
        )
        output_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(output_base, fx.make_layout((N, BM), (1, STRIDE))),
            max_size=False, num_records_bytes=BM * STRIDE * 2,
        )
        cshuffle_lds = cshuffle_storage.peek().view(fx.make_layout(CSHUFFLE_WAVES * 16 * BN, 1))
        cshuffle_write_atom = ops.get_universal_copy_atom(fx.BFloat16, 128)
        cshuffle_read_atom = ops.get_universal_copy_atom(fx.BFloat16, 64)
        output_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16)
        lane_group = lane // 16
        lane_row = lane % 16
        local_wave = wave % CSHUFFLE_WAVES
        wave_lds_base = local_wave * (16 * BN)
        output_destination_offsets = []
        for row_pair in range_constexpr(WAVE_M // 16):
            row_pair_offsets = []
            for n_half in range_constexpr(2):
                for output_row_half in range_constexpr(2):
                    output_atom = n_half * 8 + lane % 8
                    output_row = (
                        wave * 16 + row_pair * (NUM_WAVES * 16) + output_row_half * 8 + lane // 8
                    )
                    row_pair_offsets.append(output_tensor.layout(output_atom * 8, output_row))
            output_destination_offsets.append(row_pair_offsets)

        # 这里只准备C读指针；带副作用的pin仍在A读取之后执行。
        cshuffle_read_bases = []
        for n_half in range_constexpr(2):
            for output_row_half in range_constexpr(2):
                output_atom = n_half * 8 + lane % 8
                n_group = output_atom // 2
                offset = (wave_lds_base + cshuffle_plane_offset(
                    output_row_half * 8 + lane // 8, (output_atom % 2) * 2, n_group // 2)
                    + (n_group % 2) * 4)
                cshuffle_read_bases.append(fx.get_iter(cshuffle_lds) + offset)

        if const_expr(weight_quant_type == "ptpc"):
            weight_scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert_id) * N, (N,), fx.Float32,
                ),
                max_size=False, num_records_bytes=N * 4,
            )

    def pin_c_read_addresses():
        nonlocal cshuffle_read_pointers
        cshuffle_read_pointers = []

        for index in range_constexpr(4):
            pointer = cshuffle_read_bases[index]
            address = fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32),
                [_raw(fx.Int32(fx.ptrtoint(pointer)))], "", "=v,0", has_side_effects=True))
            cshuffle_read_pointers.append(fx.inttoptr(pointer.type, address))

    def mma(weight, kb, record, *, full_half=False):
        return mfma_8x1_record(weight, kb, record, k_widths=K_WIDTHS,
                              afragments=a_fragments, c=frag_c, atom=mma_atom,
                              weight_n_group_begin=2 * (record % 2) if full_half else 0)

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(n, pair, weight_quant_type=weight_quant_type,
                                scale_buffer=weight_scale_buffer, mm=mm, ops=ops)

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(fragments, destinations, lgkmcnt=lgkmcnt, store_atom=output_store_atom)

    # 仅绑定已有Tensor和编译期回调；各K共用scale/pack/输出store，B布局仍各自专用。
    clear_super_record = partial(clear_8x1_record, c=frag_c)
    constrain_mfma_valu_packet = partial(schedule_k128_pack, weight_quant_type=weight_quant_type)
    pack = partial(pack_8x1_record, c=frag_c, row_scale=frag_row_scale, weight_quant_type=weight_quant_type)
    store_c_tile_r2g = partial(store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g)
    # KS=3/5时LDS槽随N翻转，跨回边携带完整地址；偶数KS和短N无需此回调。
    prepare_addresses = prepare_b_addresses if const_expr(use_n_loop and K in (384, 640)) else None
    save_state = partial(save_8x1_state, c=frag_c, ptpc=weight_quant_type == "ptpc",
                         first_unpacked=FIRST_UNPACKED, prepare_b_addresses=prepare_addresses)
    run_tile = partial(
        run_8x1_tile, k=K, n_tiles=NT, ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r, store_b_r2s=store_b_r2s, read_b_s2r=read_b_s2r,
        read_scale_g2r=read_scale_g2r, pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g,
        mma=mma, clear=clear_super_record, schedule_pack=constrain_mfma_valu_packet,
        prepare_b_addresses=prepare_addresses,
    )

    def prologue():
        nonlocal a_fragments
        prepare_views()
        input_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        a_fragments = [mm.make_fragment_B(fx.make_view(
            fx.get_iter(input_tensor), fx.make_layout((BM, width), (1, BM)),
        )) for width in K_WIDTHS]
        # B坐标(n,kb,half)：N128块号、K分块号、N64半区号(0=L/1=H)；slot另指缓冲槽。
        prefetch0_n, prefetch0_kb, prefetch0_half, prefetch0_lds_slot = b_startup_coords(0)
        prefetch1_n, prefetch1_kb, prefetch1_half, prefetch1_lds_slot = b_startup_coords(1)

        # 执行：首B→A→Q0落LDS→Q1/Q2；C地址pin保持在A之后。
        read_b_g2r(0, 0, 0, prefetch_slot=0)
        fx.rocdl.sched_barrier(0)
        read_a_g2r(input_tensor, sorted_lds, a_fragments, input_copy, lane, wave, K_WIDTHS, K_OFFSETS)
        pin_c_read_addresses()
        rocdl.s_waitcnt(vmcnt=4)
        store_b_r2s(0, 0, 0, prefetch_slot=0)
        fx.rocdl.sched_barrier(0)
        read_b_g2r(prefetch0_n, prefetch0_kb, prefetch0_half, prefetch_slot=0)
        read_b_g2r(prefetch1_n, prefetch1_kb, prefetch1_half, prefetch_slot=1)
        rocdl.s_waitcnt(vmcnt=1)
        # q0前即错相，必须先让双方写入的Q0在LDS中就绪。
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        frag_c.fill(0)
        return weight_staging

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # 载体仅在N1完成后创建，不提前延长跨N的SSA生命周期。
        b_carriers = [fx.make_fragment_like(b_prefetch[index]) for index in range_constexpr(2)]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [fx.make_fragment_like(scales[pair]) for pair in range_constexpr(FIRST_UNPACKED, 4)]
        restore_state = partial(
            restore_8x1_state, c=frag_c, ptpc=weight_quant_type == "ptpc",
            first_unpacked=FIRST_UNPACKED, prepare_b_addresses=prepare_addresses,
            b_carriers=b_carriers, scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline执行：prologue驻留A、LDS=Q0、P0/P1=Q1/Q2。
    b_prefetch = prologue()
    b_addresses = prepare_b_addresses(0) if const_expr(prepare_addresses is not None) else []

    # N0：先错相再从q0执行；已pack C0/C1/C2，FP32 C3留给下一N。
    # group1的等待与group0的q0 Memory末屏障配对。
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0, b_prefetch, previous_packed=[], previous_scales=[],
        first=True, last=NT == 1, addresses=b_addresses,
    )

    # N1：补pack并排空N0，交织生成N1；旧scale只传尚未pack的C3。
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1, b_prefetch, c_bf16, c_scales[FIRST_UNPACKED:],
            first=False, last=NT == 2, addresses=b_addresses,
        )

    # Loop：旧C退休/新C计算的固定1N回边；NT=1/2无SSA，NT=3保留zero-trip。
    if const_expr(NT >= 3):
        initial, restore_state = prepare_loop_state(b_prefetch, c_bf16, c_scales, b_addresses)
        ops.clear_all()
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(state)
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0, b_prefetch, previous_packed, previous_scales,
                first=False, last=False, addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        ops.clear_all()
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail：仍交织旧C/新C，仅裁掉无消费者的B；NT=1/2时为空。
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n, b_prefetch, c_bf16, previous_scales,
            first=False, last=n == NT - 1, addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue：先wait，再补pack C3，最后C store四个输出quarter并保留group0补偿。
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()


# 单拍时间线：K128n/K320共享


def run_8x1_tile(
    n, b_prefetch, previous_packed, previous_scales, *, first=False, last=False, addresses,
    k, n_tiles, ptpc, read_b_g2r, store_b_r2s, read_b_s2r, read_scale_g2r, pack, shuffle_c_r2s2r, store_c_r2g,
    mma, clear, schedule_pack, k_widths=None, prepare_b_addresses=None,
):
    """q = 2*KS*n + step仅作时间坐标；n可动态，step仍在编译期展开。

    所有N均从step0开始；first=True仅标记N0，错相已在首N调用前建立。
    previous_packed原地补全；
    FP32 c由clear/mma/pack闭包更新，不在返回值中。packed/scales属于当前N，
    scales返回完整四record；previous_scales仅含尾部，按record-FIRST_UNPACKED索引。
    K128n在step0补旧C3，KS-1/KS/末拍pack新C0/C1/C2；K320在step0补旧C2/C3，
    step2 pack新C0/C1。以下只执行packing_events这一份规则。
    """
    widths = k_widths or (128,) * (k // 128)
    KS = len(widths)
    budgets = vmem_wait_schedule(k, n_tiles, ptpc, k_widths)
    FIRST_UNPACKED = 2 if k == 320 else 3
    # 短N等待规则同时适用于N0/N1，与当前拍是否读取整half分开。
    short_n_k128 = k != 320 and n_tiles < 3

    def b_target(n, step, *, ahead):
        target = step + ahead
        target_n, target_kb = n + target // (2 * KS), target % KS
        return target_n, target_kb, (target % (2 * KS)) // KS, (target_n * KS + target_kb) & 1

    def has_b_target(step, *, ahead, last):
        return not last or step + ahead < 2 * KS

    packed, scales = [], []
    for step in range_constexpr(2 * KS):
        # kb是K分块索引：K128n每块128，K320依次为128/192；不是K偏移或缓冲槽。
        # half=0/1选当前N128的前/后64列；先遍历L的全部K块，再遍历H。
        kb, half, prefetch_slot = step % KS, step // KS, step & 1
        lds_slot = (n * KS + kb) & 1
        # BK128首q0总是读整N64；短N首tile余拍也如此，K320始终读两个N32。
        read_full_half = first and k != 320 and (step == 0 or short_n_k128)
        output = output_quarter(k, step, k_widths)
        has_output = not first and output is not None
        events = packing_events(k, step, first=first, k_widths=k_widths)

        # Memory：读Q[q]，交织回写上一N的C；P槽与LDS槽各自轮转。
        priority(0)
        if const_expr(step < 4):
            scales.append(read_scale_g2r(n, step))
        if const_expr(has_output):
            # output编码(row, half)，不是Compute中的N32 record。
            fragments, destinations = shuffle_c_r2s2r(n - 1, previous_packed, output % 2, output // 2)
        b_read_address = addresses[kb & 1] if const_expr(prepare_b_addresses is not None) else None
        if const_expr(read_full_half):
            # 整half与两个packet的读取互斥；两种片段各用对应的MFMA索引。
            b_full = read_b_s2r(lds_slot, kb, half, packet=None, address=b_read_address)
        else:
            b0 = read_b_s2r(lds_slot, kb, half, packet=0, address=b_read_address)
            if const_expr(has_output):
                # 等旧CShuffle读完成即可发store，当前B的读取仍可在途。
                store_c_r2g(fragments, destinations, lgkmcnt=widths[kb] // 32)
                fx.rocdl.sched_barrier(0)
            b1 = read_b_s2r(lds_slot, kb, half, packet=1, address=b_read_address)

        # wait保护即将提交的B及本拍pack的scale；N1独立，回边使用n>=2预算。
        budget_n = 0 if first else n_tiles - 1 if last else n if isinstance(n, int) else 2
        # 短N末拍仍由lowering在scale实际使用前插wait，不提前到Memory。
        if const_expr(has_b_target(step, ahead=1, last=last) or (ptpc and not short_n_k128 and events)):
            rocdl.s_waitcnt(vmcnt=budgets[budget_n * 2 * KS + step])
        if const_expr(has_b_target(step, ahead=1, last=last)):
            # B r→s，写入Q[q+1]：wait后才物化目标地址，消费P[prefetch_slot]。
            b_r2s_n, b_r2s_kb, b_r2s_half, b_r2s_lds_slot = b_target(n, step, ahead=1)
            if const_expr(prepare_b_addresses is not None):
                # L末K转同N的H/K0；奇数KS不能用当前LDS槽+1。
                relative_slot = (b_r2s_kb + ((step + 1) // (2 * KS)) * KS) & 1
                write_address = addresses[2 + relative_slot]
            else:
                write_address = None
            store_b_r2s(b_r2s_lds_slot, b_r2s_kb, b_r2s_half, prefetch_slot=prefetch_slot,
                        fragment=b_prefetch[prefetch_slot], address=write_address)
        if const_expr(has_b_target(step, ahead=3, last=last)):
            # prefetch Q[q+3]：提交后复用同一个P槽；目标仍在原sched屏障前物化。
            b_g2r_n, b_g2r_kb, b_g2r_half, b_g2r_lds_slot = b_target(n, step, ahead=3)
            fx.rocdl.sched_barrier(0)
            b_prefetch[prefetch_slot] = read_b_g2r(
                b_g2r_n, b_g2r_kb, b_g2r_half, prefetch_slot=prefetch_slot,
            )
        if const_expr(first and (step == 0 or (k == 320 and step == 1))):
            # q0已错相：Q1的跨组消费者需要双方LDS写完成；K320另保留首次H/K128交接。
            rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()

        # Compute：packet顺序不变；当前record的MFMA与旧/新代record的pack交织。
        priority(3)
        # packet=0/1选half内前/后32列，每个执行一组MFMA；record=0..3是N32输出分片号。
        for packet in range_constexpr(2):
            record = 2 * half + packet
            if const_expr(kb == 0):
                clear(record)
            fx.rocdl.sched_barrier(0)
            if const_expr(read_full_half):
                mma(b_full, kb, record, full_half=True)
            else:
                b_packet = b0 if const_expr(packet == 0) else b1
                mma(b_packet, kb, record)
            for pack_packet, from_previous_n, pack_record in events:
                if const_expr(packet == pack_packet):
                    if const_expr(from_previous_n):
                        previous_packed.append(pack(pack_record, previous_scales[pack_record - FIRST_UNPACKED]))
                    else:
                        packed.append(pack(pack_record, scales[pack_record]))
                    schedule_pack()
        if const_expr(prepare_b_addresses is not None and step == 2 * KS - 1 and not last):
            addresses = prepare_b_addresses(n + 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        priority(0)
        stage_end()
    return b_prefetch, packed, scales, addresses


# ==================== K=192：整块K192双槽、每半区48-MFMA/wave ====================


@cache
def k192_wait_schedule(n_tiles, ptpc):
    """消费FIFO的每half-B为16B+8B两条VMEM；同时保护B及跨半区pack的scale。"""
    sequence, requests, scales, budgets = 0, {}, {}, []

    def request_b_g2r(q):
        nonlocal sequence
        if q < 2 * n_tiles:
            sequence += 2
            requests[q] = sequence

    request_b_g2r(1)
    request_b_g2r(2)
    for n in range(n_tiles):
        for half in range(2):
            q = 2 * n + half
            if ptpc:
                for pair in range(2 * half, 2 * half + 2):
                    sequence += 2
                    scales[n, pair] = sequence
            if n > 0:
                sequence += 4
            required = [requests[q + 1]] if q + 1 < 2 * n_tiles else []
            if ptpc:
                if half == 0 and n > 0:
                    required.extend(scales[n - 1, pair] for pair in (2, 3))
                elif half == 1:
                    required.extend(scales[n, pair] for pair in (0, 1))
            budgets.append(min((sequence - event for event in required), default=63))
            request_b_g2r(q + 3)
    return tuple(budgets)


def save_k192_state(b_prefetch, packed, scales, addresses, *, c, ptpc):
    state = [part.load() for carry in b_prefetch for part in carry]
    for row in range_constexpr(2):
        for group in range_constexpr(4, 8):
            state.append(c[None, group, row].load())
    if const_expr(ptpc):
        state.extend(scales[pair].load() for pair in range_constexpr(2, 4))
    for pair in range_constexpr(2):
        state.extend(packed[pair][row] for row in range_constexpr(2))
    state.extend(addresses)
    return state


def restore_k192_state(state, *, c, ptpc, b_carriers, scale_carriers):
    for half in range_constexpr(2):
        for part in range_constexpr(2):
            b_carriers[half][part].store(state[half * 2 + part])
    offset, scales = 4, []
    for row in range_constexpr(2):
        for group in range_constexpr(4, 8):
            c[None, group, row].store(state[offset])
            offset += 1
    for pair in range_constexpr(2):
        if const_expr(ptpc):
            scale_carriers[pair].store(state[offset])
            scales.append(scale_carriers[pair])
            offset += 1
        else:
            scales.append(fx.Float32(1.0))
    packed = []
    for pair in range_constexpr(2):
        packed.append([Vec(state[offset]), Vec(state[offset + 1])])
        offset += 2
    return [list(carry) for carry in b_carriers], packed, scales, [fx.Int32(value) for value in state[-5:]]


@flyc.jit
def _emit_k192_body(
    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids, p_w_scale, p_a_scale,
    M, e_idx, tid, lane, wave, group, N, TOPK, padding, weight_quant_type, act_quant_type,
    _task_table, _store_cache, ops,
):
    # 配置与原语
    # 只推导host静态形状；任务映射与动态线程坐标由kernel原样传入。
    K, BM, BN = 192, 256, 128
    K_WIDTHS = (192,)
    K_OFFSETS = (0,)
    KS, NT = len(K_WIDTHS), N // BN
    STRIDE = N + padding // 2
    FIRST_UNPACKED = 2
    LOOP_END = max(2, NT - 2)

    def schedule_pack():
        # PTPC40条VALU、per-tensor24条；K192每packet有24MFMA。
        for index in range_constexpr(24):
            rocdl.sched_group_barrier(0x8, 1, 0)
            rocdl.sched_group_barrier(0x2, 2 if weight_quant_type == "ptpc" and index < 16 else 1, 0)
        rocdl.sched_barrier(0)

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    allocator = fx.SharedAllocator()
    bslots = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, 2 * BN * K, 16])
    bptr = bslots.peek().ptr  # 两槽仍连续分配；槽位偏移包含在预计算地址中。
    scratch = allocator.allocate(fx.Array[fx.BFloat16, 4 * 16 * BN, 16])
    scratch_view = scratch.peek().view(fx.make_layout(4 * 16 * BN, 1))
    ids_lds = fx.make_view(fx.recast_iter(fx.Int32, scratch.peek().ptr), fx.make_layout(BM, 1))
    ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM), (BM,), fx.Int32,
        ),
        max_size=False, num_records_bytes=BM * 4,
    )
    if tid < BM:
        ids_lds[tid] = ids[tid]
    gpu.barrier()

    a = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False, num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    expert = (p_sorted_expert_ids[2 * e_idx + 1] if const_expr(_task_table)
              else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx])
    weights = fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight)) + fx.Int64(expert) * N * K,
            fx.make_layout(N * K, 1),
        ), max_size=False, num_records_bytes=N * K,
    )
    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c = mm.make_fragment_C(fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1))))
    routing_scale = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM), (BM,), fx.Float32,
    )
    row_tensor = fx.make_view(fx.get_iter(routing_scale), fx.make_layout((BN, BM), (0, 1)))
    row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coords = ops.load_tiled_mma_fragC(
            mm, fx.make_view(fx.get_iter(ids_lds), fx.make_layout((BN, BM), (0, 1))), copy_atom_bits=32,
        )
        als = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        ascale = mm.make_fragment_C(row_tensor)
        for dst, coord in fxh.all_elements(ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy), coords):
            encoded = coord[0].bitcast(fx.Uint32)
            fx.copy(scale_copy, fxh.atom_tensor(als, (encoded & 0xFFFFFF, encoded >> 24), 32), dst)
        row_scale.store(row_scale.load() * ascale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            ws = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
            row_scale.store(row_scale.load() * ws)
    else:
        # 全局activation scale乘每专家weight scale，提前融合进路由scale。
        scalar_a = fx.make_view(fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1))[0]
        scalar_w = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
        row_scale.store(row_scale.load() * (scalar_a * scalar_w))

    read_b_packet_g2r = partial(read_b_g2r_packet, weights)
    # 闭包绑定视图和驻留A；跨N变化的SSA仍由显式state传递。
    afragments = b_template = b_copy = b_partition = b_write_templates = None
    out = store_atom = scratch_write = scratch_read = None
    scratch_base = lane_group = scale_buffer = None

    def read_b_g2r(n, half):
        base = n * BN * K + half * (BN // 2) * K

        return [read_b_packet_g2r(tid * 16, base, 4), read_b_packet_g2r(8192 + tid * 8, base, 2)]

    def prepare_b_addresses(n):
        read_address = fx.Int32(fx.ptrtoint(fx.get_iter(b_partition))) + fx.Int32((n & 1) * BN * K)
        # 读L时写同N的H；读H时写下一N的L。两份完整lane指针在此准备。
        write_base = fx.Int32(fx.ptrtoint(bptr))
        write_h = write_base + fx.Int32((n & 1) * BN * K + (BN // 2) * K)
        write_l = write_base + fx.Int32(((n + 1) & 1) * BN * K)
        values = [read_address, write_h + tid * 16, write_h + 8192 + tid * 8,
                  write_l + tid * 16, write_l + 8192 + tid * 8]

        # 空tied asm不增加ISA指令，但强制地址在compute尾部sched屏障前就绪。
        return [fx.Int32(llvm.inline_asm(ir.IntegerType.get_signless(32), [_raw(value)],
            "", "=v,0", has_side_effects=True)) for value in values]

    def store_b_r2s(half, fragments, *, addresses):
        for part in range_constexpr(2):
            words = 4 if part == 0 else 2
            destination = LdsTensor(b_write_templates[part])  # pyright: ignore[reportOptionalSubscript]
            destination.store(
                fragments[part], address_bytes=addresses[1 + (1 - half) * 2 + part],
                copy_atom=ops.get_universal_copy_atom(fx.Uint32, words * 32),
            )

    def read_b_s2r(half, *, packet, addresses):
        source = LdsTensor(b_partition)
        fragment = mm.make_fragment_A(b_template)

        source.load(
            address_bytes=addresses[0], offset_bytes=half * (BN // 2) * K + packet * (BN // 4) * K,
            into=ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy), copy_atom=b_copy,
        )
        return fragment

    def prepare_views():
        nonlocal b_template, b_copy, b_partition, b_write_templates
        nonlocal out, store_atom, scratch_write, scratch_read, scratch_base, lane_group, scale_buffer
        # partition只建立一次；完整lane地址在上一compute尾部准备，跨N携带。
        b_template = fx.make_view(bptr,
            fx.make_layout(((16, BN // 64), (16, K // 16)), ((16, 16 * K), (1, 256))))
        b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
        b_partition = ops.get_tiled_mma_partition_S(mm, b_template, "A", copy_atom_bits=128)
        b_write_ptr = fx.recast_iter(fx.Uint32, bptr)
        # 两个Uint32模板分别保留原b_write_ptr和b_write_ptr+2的pointer type/alignment。
        b_write_templates = [
            fx.make_view(b_write_ptr, fx.make_layout(4, 1)),
            fx.make_view(b_write_ptr + 2, fx.make_layout(2, 1)),
        ]

        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.BFloat16)
                + (fx.Int64(row_begin) * STRIDE if const_expr(_task_table) else fx.Int64(e_idx) * BM * STRIDE),
                fx.make_layout((N, BM), (1, STRIDE)),
            ),
            max_size=False, num_records_bytes=BM * STRIDE * 2,
        )
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16)
        scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
        scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
        scratch_base = (wave % 4) * 16 * BN
        lane_group = lane // 16

        if const_expr(weight_quant_type == "ptpc"):
            scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, (N,), fx.Float32),
                max_size=False, num_records_bytes=N * 4,
            )

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(
            n, pair, weight_quant_type=weight_quant_type,
            scale_buffer=scale_buffer if const_expr(weight_quant_type == "ptpc") else None, mm=mm, ops=ops,
        )

    def mma(weight, ks, pair):
        return mfma_8x1_record(weight, ks, pair, k_widths=K_WIDTHS, afragments=afragments, c=c, atom=atom)

    def shuffle_c_r2s2r(n, packed, row, half):
        return shuffle_8x1_c_r2s2r(
            n, packed, row, half, scratch_view=scratch_view, scratch_base=scratch_base, lane=lane,
            lane_group=lane_group, wave=wave, out=out, scratch_write=scratch_write, scratch_read=scratch_read,
        )

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(fragments, destinations, lgkmcnt=lgkmcnt, store_atom=store_atom)

    pack = partial(pack_8x1_record, c=c, row_scale=row_scale, weight_quant_type=weight_quant_type)
    store_c_tile_r2g = partial(store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g)
    clear = partial(clear_8x1_record, c=c)
    save_state = partial(save_k192_state, c=c, ptpc=weight_quant_type == "ptpc")
    run_tile = partial(
        run_k192_tile, n_tiles=NT, ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r, store_b_r2s=store_b_r2s, read_b_s2r=read_b_s2r,
        read_scale_g2r=read_scale_g2r, pack=pack, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g,
        mma=mma, clear=clear, schedule_pack=schedule_pack,
        prepare_b_addresses=prepare_b_addresses,
    )

    def prologue():
        nonlocal afragments
        prepare_views()
        acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        afragments = [mm.make_fragment_B(fx.make_view(
            fx.get_iter(a), fx.make_layout((BM, width), (1, BM)),
        )) for width in K_WIDTHS]
        q0_destinations, q0_copies = [], []
        for part in range_constexpr(2):
            words = 4 if part == 0 else 2
            offset = tid * 16 if part == 0 else 8192 + tid * 8
            q0_destinations.append(fx.make_view(
                fx.recast_iter(fx.Uint32, bptr) + offset // 4, fx.make_layout(words, 1),
            ))
            q0_copies.append(ops.get_universal_copy_atom(fx.Uint32, words * 32))

        # 执行：首B→A→Q0落LDS→Q1/Q2；Q0仍为16B+8B，A只读三个K64。
        first_b = read_b_g2r(0, 0)
        rocdl.sched_barrier(0)
        read_a_g2r(a, ids_lds, afragments, acopy, lane, wave, K_WIDTHS, K_OFFSETS)
        rocdl.s_waitcnt(vmcnt=0)
        for part in range_constexpr(2):
            fx.copy(q0_copies[part], first_b[part], q0_destinations[part])
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        c.fill(0)
        b_prefetch = [read_b_g2r(0, 1), None]  # P0=Q1=当前H。
        if const_expr(NT > 1):
            b_prefetch[1] = read_b_g2r(1, 0)  # P1=Q2=下一N的L。
        return b_prefetch

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # 仅在N1之后创建原回边carrier；顺序及初次save时机不变。
        b_carriers = [[fx.make_fragment_like(part) for part in carry] for carry in b_prefetch]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [fx.make_fragment_like(scales[pair]) for pair in range_constexpr(FIRST_UNPACKED, 4)]
        restore_state = partial(
            restore_k192_state, c=c, ptpc=weight_quant_type == "ptpc", b_carriers=b_carriers,
            scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline执行：prologue驻留A、LDS=Q0、P0/P1=Q1/Q2（无消费者不预取）。
    b_prefetch = prologue()
    b_addresses = prepare_b_addresses(0)

    # N0：先错相再从q0执行L/H两拍；已pack C0/C1，FP32 C2/C3留给下一N。
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0, b_prefetch, previous_packed=[], previous_scales=[],
        first=True, last=NT == 1, addresses=b_addresses,
    )

    # N1：补pack并排空N0，交织生成N1；首过渡预算只使用一次。
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1, b_prefetch, c_bf16, c_scales[FIRST_UNPACKED:],
            first=False, last=NT == 2, addresses=b_addresses,
        )

    # Loop：旧C退休/新C计算；动态n+2均有效，NT=4保留zero-trip回边。
    if const_expr(NT >= 4):
        initial, restore_state = prepare_loop_state(b_prefetch, c_bf16, c_scales, b_addresses)
        ops.clear_all()
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(state)
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0, b_prefetch, previous_packed, previous_scales,
                first=False, last=False, addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        ops.clear_all()
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail：末两N仍交织旧C/新C，逐half按实际消费者裁剪预取。
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n, b_prefetch, c_bf16, previous_scales,
            first=False, last=n == NT - 1, addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue：先wait，再补pack C2/C3，最后store末N并保留group0补偿。
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()


# 单拍时间线：K192独立的L/H两拍


def run_k192_tile(
    n, b_prefetch, previous_packed, previous_scales, *, first=False, last=False, addresses,
    n_tiles, ptpc, read_b_g2r, store_b_r2s, read_b_s2r, read_scale_g2r, pack, shuffle_c_r2s2r, store_c_r2g,
    mma, clear, schedule_pack, prepare_b_addresses,
):
    """q = 2*n + half仅作注释时间坐标；错相已在首N调用前建立，所有N均执行L/H两拍。

    两个P槽各含16B+8B两段；addresses含当前N读地址、当前N.H/下一N.L写地址。
    previous_packed原地补旧C2/C3，FP32 c由闭包更新；返回当前N的packed C0/C1、
    完整四record scales；非last时地址推进到下一N，previous_scales仅为旧scale2/scale3。
    最后两N独立裁剪，不能套共享BK128尾部。
    """
    budgets = k192_wait_schedule(n_tiles, ptpc)
    body_budgets = tuple(min((budgets[2 * n + half] for n in range(2, n_tiles - 2)), default=63)
                         for half in range(2))
    has_next = not last
    # 动态n只出现在末两N之前，不对运行时SSA求host布尔值。
    has_future = n + 2 < n_tiles if isinstance(n, int) else True
    packed, scales = [], []

    # K192只有一个K块(kb=0)；half=0/1依次处理当前N128的前/后64列。
    for half in range_constexpr(2):
        prefetch_slot = half
        # Memory：read Q[2*n+half]；逐packet交织上一N的CShuffle/store。
        priority(0)
        for pair in range_constexpr(2 * half, 2 * half + 2):
            scales.append(read_scale_g2r(n, pair))
        b_packets = []
        for packet in range_constexpr(2):
            if const_expr(not first):
                # C输出quarter编码(row=packet, half)，不是N32 pack_record。
                fragments, destinations = shuffle_c_r2s2r(n - 1, previous_packed, packet, half)
            b_packets.append(read_b_s2r(half, packet=packet, addresses=addresses))
            if const_expr(not first):
                # 本次C读比随后6条B ds_read更早；不等待整个B片段再发store。
                store_c_r2g(fragments, destinations, lgkmcnt=6)
                fx.rocdl.sched_barrier(0)
        budget = budgets[2 * n + half] if isinstance(n, int) else body_budgets[half]
        # 保护下一B提交和本拍pack的scale，后面的新预取不计入此wait。
        if const_expr(budget != 63):
            rocdl.s_waitcnt(vmcnt=budget)
        if const_expr(half == 0 or has_next):
            # B r→s，写入Q[q+1]：L拍写当前N.H，H拍写下一N.L。
            store_b_r2s(1 - half, b_prefetch[prefetch_slot], addresses=addresses)
        if const_expr((half == 0 and has_next) or (half == 1 and has_future)):
            # prefetch Q[q+3]：复用刚提交的P[half]，读取下一N.H/下下N.L。
            fx.rocdl.sched_barrier(0)
            b_prefetch[prefetch_slot] = read_b_g2r(n + 1 + half, 1 - half)
        if const_expr(first and half == 0):
            # q0前已错相；另一组进入H前必须看到双方写入的完整Q1。
            rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()

        # Compute：保留pack交织与下一N的5地址准备。
        priority(3)
        # packet=0/1选half内前/后32列，每个执行一组MFMA；record=0..3是N32输出分片号。
        for packet in range_constexpr(2):
            record = half * 2 + packet
            clear(record)
            fx.rocdl.sched_barrier(0)
            mma(b_packets[packet], 0, record)
            if const_expr(half == 0 and not first):
                # L拍的新C0/C1计算后补pack旧C2/C3，尾部scale按packet索引。
                pack_record = 2 + packet
                previous_packed.append(pack(pack_record, previous_scales[packet]))
                schedule_pack()
            elif const_expr(half == 1):
                # H拍的新C2/C3计算后pack本N已完成的C0/C1。
                pack_record = packet
                packed.append(pack(pack_record, scales[pack_record]))
                schedule_pack()
        if const_expr(half == 1 and has_next):
            addresses = prepare_b_addresses(n + 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        priority(0)
        stage_end()
    return b_prefetch, packed, scales, addresses


# ==================== K=320：128+192四阶段，不padding或增加MFMA ====================


@flyc.jit
def _emit_k320_body(
    p_input, p_weight, p_output, p_sorted_ids, p_sorted_weights, p_sorted_expert_ids, p_w_scale, p_a_scale,
    M, e_idx, tid, lane, wave, group, group_tid,
    N, TOPK, padding, weight_quant_type, act_quant_type, _task_table, _store_cache, ops,
):
    # 配置与原语
    # 静态形状与等待账本沿用原Python表达式；任务映射和线程坐标由kernel传入。
    K, BM, BN = 320, 256, 128
    K_WIDTHS = (128, 192)
    K_OFFSETS = (0, 128)
    KS, NT = len(K_WIDTHS), N // BN
    STRIDE = N + padding // 2
    FIRST_UNPACKED = 2
    LOOP_END = max(2, NT - 1)

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    allocator = fx.SharedAllocator()
    # KS=2时slot=(n*2+ks)&1=ks，固定16/24KiB非对称槽，总LDS56KiB。
    b0 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[0], 16])
    b1 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[1], 16])
    bptrs = [b0.peek().ptr, b1.peek().ptr]
    scratch = allocator.allocate(fx.Array[fx.BFloat16, 4 * 16 * BN, 16])
    scratch_view = scratch.peek().view(fx.make_layout(4 * 16 * BN, 1))
    ids_lds = fx.make_view(fx.recast_iter(fx.Int32, scratch.peek().ptr), fx.make_layout(BM, 1))
    ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
            (BM,), fx.Int32,
        ),
        max_size=False, num_records_bytes=BM * 4,
    )
    if tid < BM:
        ids_lds[tid] = ids[tid]
    gpu.barrier()

    a = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False, num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    expert = (p_sorted_expert_ids[2 * e_idx + 1] if const_expr(_task_table)
              else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx])
    weights = fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight)) + fx.Int64(expert) * N * K,
            fx.make_layout(N * K, 1),
        ), max_size=False, num_records_bytes=N * K,
    )
    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c = mm.make_fragment_C(fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1))))
    routing_scale = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
        (BM,), fx.Float32,
    )
    row_tensor = fx.make_view(fx.get_iter(routing_scale), fx.make_layout((BN, BM), (0, 1)))
    row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coords = ops.load_tiled_mma_fragC(
            mm, fx.make_view(fx.get_iter(ids_lds), fx.make_layout((BN, BM), (0, 1))), copy_atom_bits=32,
        )
        als = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False, num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        ascale = mm.make_fragment_C(row_tensor)
        for dst, coord in fxh.all_elements(ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy), coords):
            encoded = coord[0].bitcast(fx.Uint32)
            fx.copy(scale_copy, fxh.atom_tensor(als, (encoded & 0xFFFFFF, encoded >> 24), 32), dst)
        row_scale.store(row_scale.load() * ascale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            ws = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
            row_scale.store(row_scale.load() * ws)
    else:
        scalar_a = fx.make_view(fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1))[0]
        scalar_w = fx.make_view(fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1))[0]
        row_scale.store(row_scale.load() * (scalar_a * scalar_w))

    read_b_packet_g2r = partial(read_b_g2r_packet, weights)
    # 闭包绑定视图和驻留A；跨N变化的SSA仍由显式state传递。
    afragments = out = store_atom = None
    scratch_write = scratch_read = scratch_base = lane_group = scale_buffer = None

    def b_startup_coords(q):
        # 仅供启动Q1/Q2定位；K320每N四拍，不需要valid字段。
        target = q + 1
        n, ks = target // (2 * KS), target % KS
        return n, ks, (target % (2 * KS)) // KS, (n * KS + ks) & 1

    def b_offsets(ks):
        width = K_WIDTHS[ks]
        index = group * (BN * width // 64) + group_tid
        global_offset = (index // width) * (16 * K) + ((index % width) // 16) * 256 + (index % 16) * 16
        local_offset = (index // width) * (16 * width) + ((index % width) // 16) * 256 + (index % 16) * 16
        return global_offset, local_offset

    def tail192_b_offsets(part):
        # 连续LDS字节映回全K320 preshuffle；每16行只取K[128:320]。
        local_offset = tid * 16 if part == 0 else 8192 + tid * 8
        global_offset = (local_offset // (16 * 192)) * (16 * K) + local_offset % (16 * 192)
        return global_offset, local_offset

    # B回调直接使用共享tile签名；固定LDS槽由kb决定，P槽无需额外载体适配。
    def read_b_g2r(n, kb, half, *, prefetch_slot=0):
        if const_expr(K_WIDTHS[kb] == 192):
            scalar = n * BN * K + K_OFFSETS[kb] * 16 + half * (BN // 2) * K
            pieces = [read_b_packet_g2r(tail192_b_offsets(part)[0], scalar, 4 if part == 0 else 2)
                      for part in range_constexpr(2)]
            # 6个真实u32作为单一carry，重用通用Nloop；拼接不产生额外K加载/MFMA。
            fragment = fx.make_rmem_tensor(fx.make_layout(6, 1), fx.Uint32)
            fragment.store(Vec.from_elements(
                [Vec(pieces[0].load())[i] for i in range_constexpr(4)]
                + [Vec(pieces[1].load())[i] for i in range_constexpr(2)], fx.Uint32,
            ))
            return fragment
        # 每组256线程：BK128各16B；没有条件exec或冗余全宽加载。
        return read_b_packet_g2r(
            b_offsets(kb)[0], n * BN * K + K_OFFSETS[kb] * 16 + half * (BN // 2) * K, K_WIDTHS[kb] // 32,
        )

    def store_b_r2s(lds_slot, kb, half, *, prefetch_slot=0, fragment, address=None):
        assert address is None
        width = K_WIDTHS[kb]
        base = fx.recast_iter(fx.Uint32, bptrs[kb]) + half * (BN // 2) * (width // 4)

        if const_expr(width == 192):
            values = Vec(fragment.load())
            for part in range_constexpr(2):
                words, start = (4, 0) if part == 0 else (2, 4)
                piece = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)
                piece.store(values.shuffle(values, list(range(start, start + words))))
                destination = fx.make_view(base + tail192_b_offsets(part)[1] // 4, fx.make_layout(words, 1))
                fx.copy(ops.get_universal_copy_atom(fx.Uint32, words * 32), piece, destination)
        else:
            destination = fx.make_view(base + b_offsets(kb)[1] // 4, fx.make_layout(4, 1))
            fx.copy(ops.get_universal_copy_atom(fx.Uint32, 128), fragment, destination)

    def read_b_s2r(lds_slot, kb, half, *, packet, address=None):
        assert address is None
        width = K_WIDTHS[kb]
        # 固定槽宽分别128/192，half间距分别8192/12288B。
        view = fx.make_view(
            bptrs[kb] + half * (BN // 2) * width + packet * (BN // 4) * width,
            fx.make_layout(((16, BN // 64), (16, width // 16)), ((16, 16 * width), (1, 256))),
        )

        return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

    def prepare_views():
        nonlocal out, store_atom, scratch_write, scratch_read, scratch_base, lane_group, scale_buffer
        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.BFloat16)
                + (fx.Int64(row_begin) * STRIDE if const_expr(_task_table) else fx.Int64(e_idx) * BM * STRIDE),
                fx.make_layout((N, BM), (1, STRIDE)),
            ),
            max_size=False, num_records_bytes=BM * STRIDE * 2,
        )
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16)
        scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
        scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
        scratch_base = (wave % 4) * 16 * BN
        lane_group = lane // 16
        if const_expr(weight_quant_type == "ptpc"):
            scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, (N,), fx.Float32),
                max_size=False, num_records_bytes=N * 4,
            )

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(
            n, pair, weight_quant_type=weight_quant_type, scale_buffer=scale_buffer, mm=mm, ops=ops,
        )

    # K320只消费packet，shared的整half分支不可达，不接收full_half控制参数。
    def mma(weight, kb, record):
        return mfma_8x1_record(weight, kb, record, k_widths=K_WIDTHS, afragments=afragments, c=c, atom=atom)

    def shuffle_c_r2s2r(n, packed, row, half):
        return shuffle_8x1_c_r2s2r(
            n, packed, row, half, scratch_view=scratch_view, scratch_base=scratch_base,
            lane=lane, lane_group=lane_group, wave=wave, out=out,
            scratch_write=scratch_write, scratch_read=scratch_read,
        )

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(fragments, destinations, lgkmcnt=lgkmcnt, store_atom=store_atom)

    # 固定编译期绑定，不生成GPU操作。
    pack = partial(pack_8x1_record, c=c, row_scale=row_scale, weight_quant_type=weight_quant_type)
    store_c_tile_r2g = partial(store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g)
    clear = partial(clear_8x1_record, c=c)
    schedule_pack = partial(schedule_k128_pack, weight_quant_type=weight_quant_type)
    save_state = partial(
        save_8x1_state, c=c, ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED, prepare_b_addresses=None,
    )
    run_tile = partial(
        run_8x1_tile, k=320, n_tiles=NT, ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r, store_b_r2s=store_b_r2s, read_b_s2r=read_b_s2r, read_scale_g2r=read_scale_g2r, pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g, mma=mma, clear=clear,
        schedule_pack=schedule_pack, k_widths=K_WIDTHS, prepare_b_addresses=None,
    )

    def prologue():
        nonlocal afragments
        prepare_views()
        acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        afragments = [mm.make_fragment_B(fx.make_view(
            fx.get_iter(a), fx.make_layout((BM, width), (1, BM)),
        )) for width in K_WIDTHS]
        first_index = group * 256 + group_tid
        first_offset = ((first_index // K_WIDTHS[0]) * (16 * K)
                        + ((first_index % K_WIDTHS[0]) // 16) * 256 + (first_index % 16) * 16)
        prefetch0_n, prefetch0_kb, prefetch0_half, prefetch0_lds_slot = b_startup_coords(0)
        prefetch1_n, prefetch1_kb, prefetch1_half, prefetch1_lds_slot = b_startup_coords(1)

        # 执行：首B→A→Q0落LDS→Q1/Q2；A按128+192读取，不在prologue做MMA。
        first_b = read_b_packet_g2r(first_offset, 0, 4)
        rocdl.sched_barrier(0)
        read_a_g2r(a, ids_lds, afragments, acopy, lane, wave, K_WIDTHS, K_OFFSETS)
        rocdl.s_waitcnt(vmcnt=4)
        store_b_r2s(0, 0, 0, prefetch_slot=0, fragment=first_b)
        rocdl.sched_barrier(0)
        b_prefetch = [read_b_g2r(prefetch0_n, prefetch0_kb, prefetch0_half, prefetch_slot=0), None]
        b_prefetch[1] = read_b_g2r(prefetch1_n, prefetch1_kb, prefetch1_half, prefetch_slot=1)
        rocdl.s_waitcnt(vmcnt=1)
        # q0前即错相，必须先让双方写入的Q0在LDS中就绪。
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        c.fill(0)
        return b_prefetch

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # carrier仍在N1完成后创建，初始save顺序不变。
        b_carriers = [fx.make_fragment_like(b_prefetch[index]) for index in range_constexpr(2)]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [fx.make_fragment_like(scales[pair])
                              for pair in range_constexpr(FIRST_UNPACKED, 4)]
        restore_state = partial(
            restore_8x1_state, c=c, ptpc=weight_quant_type == "ptpc",
            first_unpacked=FIRST_UNPACKED, prepare_b_addresses=None,
            b_carriers=b_carriers, scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline执行：直接复用K128n外层之后的run_8x1_tile，不另包Memory/Compute。
    # prologue驻留A、LDS=Q0、P0/P1=Q1/Q2。
    b_prefetch = prologue()

    # N0：先错相再从q0执行；已pack C0/C1，FP32 C2/C3留给下一N。
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0, b_prefetch, previous_packed=[], previous_scales=[],
        first=True, last=NT == 1, addresses=[],
    )

    # N1：补pack并排空N0，交织生成N1；保留独立预算及旧scale尾部。
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1, b_prefetch, c_bf16, c_scales[FIRST_UNPACKED:],
            first=False, last=NT == 2, addresses=b_addresses,
        )

    # Loop：旧C退休/新C计算的固定1N回边；NT=3保留zero-trip。
    if const_expr(NT >= 3):
        initial, restore_state = prepare_loop_state(b_prefetch, c_bf16, c_scales, b_addresses)
        ops.clear_all()
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(state)
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0, b_prefetch, previous_packed, previous_scales,
                first=False, last=False, addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        ops.clear_all()
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail：仍交织旧C/新C，仅裁掉无消费者的B；NT=1/2时为空。
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n, b_prefetch, c_bf16, previous_scales,
            first=False, last=n == NT - 1, addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue：先wait，再用末tile完整scale补pack C2/C3，最后C store排空。
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()
