"""Block-native gfx942 sparse attention, without cross-query union construction.

Four query waves each own one query and its GQA heads for BN32.
V loads directly from global memory into registers for the transpose; Q/K
remain in registers. Only the output transpose uses LDS (32 KiB). Softmax
is updated once per BN tokens. Sparse addresses are full byte VOFFSETs.
Triton block sorting is charged to rebuild_plan; only active direct rows sort.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import numpy as np
import torch
import triton
import triton.language as tl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ..mha._common import (
    _buffer,
    _buffer_words,
    _exp,
    _maximum,
    _min,
    _pack_bf16,
    _pin,
    _read_address,
    _stage_end,
    _uniform,
    _wait,
)


def _load(resource, offset, lane):
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _qk(q, k):
    q, k = fx.Vector(q).bitcast(fx.Int16), fx.Vector(k).bitcast(fx.Int16)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for s in range(16):
        first = (s // 2) * 8 + (s % 2) * 4
        a = fx.Vector.from_elements([k[first + i] for i in range(4)], fx.Int16)
        b = fx.Vector.from_elements([q[first + i] for i in range(4)], fx.Int16)
        acc = fx.Vector(
            rocdl.mfma_f32_16x16x16bf16_1k(
                ir.VectorType.get([4], fx.Float32.ir_type),
                [a.ir_value(), b.ir_value(), acc.ir_value(), 0, 0, 0],
            )
        )
    return acc


def _pv(p, values, output, alpha):
    output = fx.Vector(output)
    acc = [
        fx.Vector.from_elements(
            [output[n * 4 + i] * alpha for i in range(4)], fx.Float32
        )
        for n in range(8)
    ]
    for n in range(8):
        selector = 0x03020706 if n % 2 else 0x01000504
        words = [
            fx.Int32(
                rocdl.perm_b32(
                    values[r * 4 + n // 2].ir_value(),
                    values[(r + 1) * 4 + n // 2].ir_value(),
                    fx.Int32(selector).ir_value(),
                )
            )
            for r in (0, 2)
        ]
        a = fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)
        acc[n] = fx.Vector(
            rocdl.mfma_f32_16x16x16bf16_1k(
                ir.VectorType.get([4], fx.Float32.ir_type),
                [a.ir_value(), p.ir_value(), acc[n].ir_value(), 0, 0, 0],
            )
        )
    return fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )


class DirectPlan(msgspec.Struct, frozen=True, kw_only=True):
    metadata: torch.Tensor
    query_tile: int
    num_tiles: int
    active: torch.Tensor
    query_tiles: torch.Tensor
    gated: bool
    source_blocks: torch.Tensor


@triton.jit
def direct_qsa_sort_blocks(
    Source,
    Destination,
    Meta,
    Active,
    QueryTiles,
    WAVES: tl.constexpr,
    GATED: tl.constexpr,
):
    task = tl.program_id(0)
    tile, local = task // WAVES, task % WAVES
    rows = tl.load(Meta + tile * 5 + 1)
    if local >= rows:
        return
    row = tl.load(Meta + tile * 5) + local
    if GATED:
        group = tl.load(QueryTiles + row)
        if tl.load(Active + group) != 0:
            return
    col = tl.arange(0, 512)
    blocks = tl.load(Source + row * 512 + col)
    ordered = tl.sort(tl.where(blocks >= 0, blocks, 2147483647), descending=False)
    tl.store(Destination + row * 512 + col, tl.where(ordered < 2147483647, ordered, -1))


def rebuild_plan(*, inputs, plan: DirectPlan):
    if plan.num_tiles:
        direct_qsa_sort_blocks[(plan.num_tiles * plan.query_tile,)](
            inputs.block_indices,
            plan.source_blocks,
            plan.metadata,
            plan.active,
            plan.query_tiles,
            plan.query_tile,
            plan.gated,
            num_warps=4,
        )


def prepare(
    *,
    inputs,
    skip_counts=None,
    union=None,
):
    waves = 4
    skips = (0,) * len(inputs.query_lens) if skip_counts is None else skip_counts
    metadata = []
    q0, k0 = 0, 0
    for q_len, prefix, skip in zip(
        inputs.query_lens, inputs.prefix_lens, skips
    ):
        for local in range(skip, q_len, waves):
            metadata.append(
                (
                    q0 + local,
                    min(waves, q_len - local),
                    k0,
                    prefix + q_len,
                    prefix + local,
                )
            )
        q0 += q_len
        k0 += q_len + prefix
    array = np.asarray(metadata, dtype=np.int32).reshape(-1, 5)
    return DirectPlan(
        metadata=torch.from_numpy(array).to(inputs.q.device),
        query_tile=waves,
        num_tiles=len(metadata),
        gated=union is not None,
        active=inputs.query_positions if union is None else union.active,
        query_tiles=inputs.query_sequence_ids if union is None else union.query_tiles,
        source_blocks=torch.empty_like(inputs.block_indices),
    )


def _source_offset(blocks, row, n, visible, extent, hk):
    complete = _min(visible // 4, fx.Int32(512))
    width = complete * 4 + visible % 4
    block_col = n // 4
    block = fx.Int32(blocks[row * 512 + _min(block_col, fx.Int32(511))])
    token = (n < complete * 4).select(
        block * 4 + n % 4, (visible // 4) * 4 + n - complete * 4
    )
    return (n < width).select(token * (hk * 256 * 2), fx.Int32(extent))


def _load_v_global(
    resource, blocks, row, tile, segment, half, lane, visible, extent, hk, bn
):
    first = tile * bn + segment * 16 + (lane >> 4) * 4
    complete = _min(visible // 4, fx.Int32(512))
    width = complete * 4 + visible % 4
    block = fx.Int32(blocks[row * 512 + _min(first // 4, fx.Int32(511))])
    block = (first < complete * 4).select(block, visible // 4)
    base = block * (4 * hk * 256 * 2) + (lane & 15) * 16 + half * 256
    parts = []
    for r in range(4):
        offset = (first + r < width).select(base + r * (hk * 256 * 2), fx.Int32(extent))
        parts.append(_buffer_words(resource, offset))
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _enabled(row, rows, first, query_tiles, active, gated, nq):
    valid = row < rows
    if gated:
        # Padded waves must not inspect a following dense row's sentinel tile -1.
        tile = fx.Int32(query_tiles[first + _min(row, rows - 1)])
        valid = valid & (fx.Int32(active[tile]) == 0)
    return valid


def _output(
    o0,
    o1,
    inv,
    buffer,
    shared,
    tid,
    h,
    g,
    rows,
    extent,
    threads,
    first,
    query_tiles,
    active,
    gated,
    nq,
):
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        acc = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [acc[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = _pack_bf16(values)
            col = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(shared + ((row * 256 + col) ^ ((row & 7) * 8)) * 2)
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [address.ir_value(), words.ir_value()],
                "ds_write_b128 $0, $1",
                "v,v,~{memory}",
                has_side_effects=True,
            )
    _wait(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + part * threads * 8
        parts.append(
            _read_address(shared + (element ^ (((element // 256) & 7) * 8)) * 2)
        )
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * threads * 8
        r, col = element // 256, element % 256
        query, head = r // 16, r % 16
        enabled = _enabled(query, rows, first, query_tiles, active, gated, nq) & (
            head < g
        )
        offset = enabled.select(
            query * h * 256 + head * 256 + col, fx.Int32(extent // 2)
        )
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        fx.copy(
            fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16),
            fragment,
            fx.make_view(fx.get_iter(buffer) + offset, fx.make_layout(8, 1)),
        )


@flyc.jit
def _body(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
):
    load, qk, pv = _load, _qk, _pv
    source_offset, load_v, output_fn, enabled = (
        _source_offset,
        _load_v_global,
        _output,
        _enabled,
    )
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 32768, 16])
        .peek()
        .view(fx.make_layout(32768, 1))
    )
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    task = fx.Int32(gpu.block_id("x"))
    tile, hkv = task // HK, task % HK
    first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    k0, kv_len = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 3])
    position0 = _uniform(META[tile * 5 + 4])
    valid_wave = enabled(wave, rows, first, QUERY_TILES, ACTIVE, GATED, NQ)
    visible = position0 + wave + 1
    count = _min(visible // 4, fx.Int32(512)) * 4 + visible % 4
    tiles = valid_wave.select((count + BN - 1) // BN, fx.Int32(0))
    qe = (NQ - first) * H * 256 * 2 - hkv * (H // HK) * 256 * 2
    qptr = fx.get_iter(Q) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    qr = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * 256, 1)), qe)
    qoffset = (valid_wave & ((lane & 15) < H // HK)).select(
        (wave * H + (lane & 15)) * 512, fx.Int32(qe)
    )
    q = load(qr, qoffset, lane)
    extent = (kv_len * HK - hkv) * 512
    kr = _buffer(
        fx.make_view(
            fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(NK * HK * 256, 1),
        ),
        extent,
    )
    vr = _buffer(
        fx.make_view(
            fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(NK * HK * 256, 1),
        ),
        extent,
    )
    row = _min(first + wave, fx.Int32(NQ - 1))
    blocks = rocdl.make_buffer_tensor(BLOCKS)
    maximum, total = fx.Float32(-1e30), fx.Float32(0.0)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    scale = fx.Float32(SCALE * math.log2(math.e))
    initial = source_offset(blocks, row, lane & 15, visible, extent, HK)
    initial = valid_wave.select(initial, fx.Int32(extent))
    kval = load(kr, initial, lane)
    _wait(vmcnt=0)
    rocdl.sched_barrier(0)
    for t in range(fx.Int32(0), tiles, fx.Int32(1)):
        kval = fx.Vector(kval)
        all_scores = []
        for segment in fx.range_constexpr(BN // 16):
            following = _min(t * BN + (segment + 1) * 16, ((count + 15) // 16 - 1) * 16)
            offset = source_offset(
                blocks, row, following + (lane & 15), visible, extent, HK
            )
            next_k = load(kr, offset, lane)
            scores = qk(q, kval)
            for i in fx.range_constexpr(4):
                col = t * BN + segment * 16 + (lane >> 4) * 4 + i
                all_scores.append(
                    (col < count).select(scores[i] * scale, fx.Float32(float("-inf")))
                )
            _wait(vmcnt=0)
            rocdl.sched_barrier(0)
            kval = next_k
        candidate = fx.Float32(-1e30)
        for score in all_scores:
            candidate = _maximum(candidate, score)
        candidate = _maximum(candidate, candidate.shuffle_xor(16, 64))
        candidate = _maximum(candidate, candidate.shuffle_xor(32, 64))
        updated = _maximum(maximum, candidate)
        alpha = _exp(maximum - updated)
        probabilities = fx.Vector.from_elements(
            [_exp(v - updated) for v in all_scores], fx.Float32
        )
        current = fx.Float32(0.0)
        for i in fx.range_constexpr(BN // 4):
            current = current + probabilities[i]
        current = current + current.shuffle_xor(16, 64)
        current = current + current.shuffle_xor(32, 64)
        total = total * alpha + current
        o0 = fx.Vector(o0) * alpha
        o1 = fx.Vector(o1) * alpha
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        values0 = load_v(vr, blocks, row, t, 0, 0, lane, visible, extent, HK, BN)
        values1 = load_v(vr, blocks, row, t, 0, 1, lane, visible, extent, HK, BN)
        _wait(vmcnt=0)
        rocdl.sched_barrier(0)
        for segment in fx.range_constexpr(BN // 16):
            p = _pack_bf16(
                fx.Vector.from_elements(
                    [probabilities[segment * 4 + i] for i in range(4)], fx.Float32
                )
            ).bitcast(fx.Int16)
            o0 = pv(p, values0, o0, fx.Float32(1.0))
            if fx.const_expr(segment + 1 < BN // 16):
                next_values0 = load_v(
                    vr, blocks, row, t, segment + 1, 0, lane, visible, extent, HK, BN
                )
            o1 = pv(p, values1, o1, fx.Float32(1.0))
            if fx.const_expr(segment + 1 < BN // 16):
                next_values1 = load_v(
                    vr, blocks, row, t, segment + 1, 1, lane, visible, extent, HK, BN
                )
                _wait(vmcnt=0)
                rocdl.sched_barrier(0)
                values0, values1 = next_values0, next_values1
        maximum = updated
    _stage_end()
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(O) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = rocdl.make_buffer_tensor(
        fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), num_records_bytes=qe
    )
    output_fn(
        o0,
        o1,
        inv,
        output,
        shared,
        tid,
        H,
        H // HK,
        rows,
        qe,
        THREADS,
        first,
        QUERY_TILES,
        ACTIVE,
        GATED,
        NQ,
    )


@flyc.kernel(name="direct_qsa_bf16_d256")
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
):
    body = _body
    if fx.const_expr(GATED):
        tile = fx.Int32(gpu.block_id("x")) // HK
        first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
        needed = fx.Int32(0)
        for local in fx.range_constexpr(THREADS // 64):
            row = first + _min(fx.Int32(local), rows - 1)
            group = _uniform(QUERY_TILES[row])
            needed = needed | (_uniform(ACTIVE[group]) == 0).select(
                fx.Int32(1), fx.Int32(0)
            )
        if needed != 0:
            body(
                Q,
                K,
                V,
                O,
                BLOCKS,
                META,
                ACTIVE,
                QUERY_TILES,
                H,
                HK,
                NQ,
                NK,
                BN,
                THREADS,
                TASKS,
                GATED,
                SCALE,
            )
    else:
        body(
            Q,
            K,
            V,
            O,
            BLOCKS,
            META,
            ACTIVE,
            QUERY_TILES,
            H,
            HK,
            NQ,
            NK,
            BN,
            THREADS,
            TASKS,
            GATED,
            SCALE,
        )


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    stream: fx.Stream,
):
    _kernel(
        Q,
        K,
        V,
        O,
        BLOCKS,
        META,
        ACTIVE,
        QUERY_TILES,
        H,
        HK,
        NQ,
        NK,
        BN,
        THREADS,
        TASKS,
        GATED,
        SCALE,
        value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
    ).launch(grid=(TASKS, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def run(*, inputs, prepared: DirectPlan, out: torch.Tensor):
    if prepared.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = (
        inputs.q.view(-1),
        inputs.k.view(-1),
        inputs.v.view(-1),
        out.view(-1),
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.query_tiles,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        inputs.k.shape[0],
        32,
        prepared.query_tile * 64,
        prepared.num_tiles * inputs.k.shape[1],
        prepared.gated,
        inputs.scale,
        stream,
    )
    key = (
        inputs.q.device,
        tuple(
            (
                (a.dtype, tuple(a.shape), tuple(a.stride()))
                if isinstance(a, torch.Tensor)
                else ("stream",) if isinstance(a, torch.cuda.Stream) else a
            )
            for a in args
        ),
    )
    with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm direct QSA before graph capture")
            _COMPILED[key] = flyc.compile(_launch, *args)
        else:
            compiled(*args)
