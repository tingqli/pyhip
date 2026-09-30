"""Exact short-context QSA over the original packed BF16 Q/K/V/O buffers.

For a request with M queries and prefix P, only the first
``t = min(M, max(0, limit - P))`` queries are dense. Their Q/O views have
t rows and their K/V views have P+t rows, so native bottom-right causality
keeps the absolute position P+j. Requests are launched separately: concatenated
eligible prefixes cannot describe gaps in the original packed Q/O storage.

The adjacent native linear kernel is used for complete BN64 KV tiles. Other
lengths use the local, nonpaged specialization below: every K/V byte offset
is in VOFFSET, with SOFFSET=0, including speculative K loads. Score masking
alone cannot make the native SOFFSET tail loads physically safe.

prepare() transfers only small cumulative-length metadata, once. run() creates
storage-sharing views, never packs tensors or allocates output, and never reads
device scalars on the host. It leaves all noneligible output rows untouched.
Inputs must obey the validated QSA contract (positions P+j and complete block
selection below 2052); no sparse indices or membership tables are consumed.
Reprepare after changing request lengths, prefixes, shapes, or device. Warm all
specializations before graph capture; same-layout replacement Q/K/V/O buffers
and in-place value updates are supported.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

from ..mha import _common as base
from ..mha import mha_pa_bf16_256_linear_942 as native
from ..mha._common import (
    _advance_max,
    _buffer,
    _join,
    _maximum,
    _min,
    _pin,
    _pin_i32,
    _rescale,
    _schedule,
    _stage_end,
    _uniform,
    _wait,
)

BM, BN, D, THREADS, LDS_BYTES = 128, 64, 256, 512, 65536
MAX_DENSE_VISIBLE = 2051


class DenseCall(msgspec.Struct, frozen=True, kw_only=True):
    request: int
    q_start: int
    q_count: int
    k_start: int
    kv_count: int
    cu_q: torch.Tensor
    cu_k: torch.Tensor


class DensePlan(msgspec.Struct, frozen=True, kw_only=True):
    calls: tuple[DenseCall, ...]
    query_counts: tuple[int, ...]
    query_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    device: torch.device
    num_cus: int
    limit: int


def _check_qkv(inputs) -> None:
    if inputs.q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("Dense QSA requires ROCm/gfx942")
    for tensor in (inputs.q, inputs.k, inputs.v):
        if (
            tensor.dtype != torch.bfloat16
            or tensor.device != inputs.q.device
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.ndim != 3
            or tensor.shape[-1] != D
        ):
            raise ValueError("Q/K/V must be contiguous BF16 [tokens, heads, 256]")
        if tensor.requires_grad:
            raise ValueError("Dense QSA is inference-only")
        if tensor.data_ptr() % 16 or tensor.numel() * tensor.element_size() >= 2**31:
            raise ValueError("Q/K/V require 16-byte alignment and byte spans < 2**31")
    if (
        inputs.k.shape != inputs.v.shape
        or inputs.q.shape[1] <= 0
        or inputs.k.shape[1] <= 0
        or inputs.q.shape[1] % inputs.k.shape[1]
    ):
        raise ValueError(
            "K/V must match and Q heads must be a positive multiple of KV heads"
        )
    if not math.isfinite(inputs.scale) or inputs.scale <= 0:
        raise ValueError("softmax scale must be finite and positive")


def prepare(inputs) -> DensePlan:
    """Prepare one direct-view call per eligible request, not per sparse tile.

    query_counts has one entry per original request, including zero-query and
    long-prefix requests. The parent must exclude exactly these leading rows
    from its sparse path. The selected dense boundary is fixed at2051.
    """
    limit = MAX_DENSE_VISIBLE
    _check_qkv(inputs)
    query_lens, prefix_lens = inputs.query_lens, inputs.prefix_lens
    if len(query_lens) != len(prefix_lens) or any(
        type(n) is not int or not 0 <= n < 2**31 for n in (*query_lens, *prefix_lens)
    ):
        raise ValueError(
            "Request lengths and prefixes must be matching nonnegative host integers"
        )
    q_bounds, k_bounds = [0], [0]
    for length, prefix in zip(query_lens, prefix_lens):
        q_bounds.append(q_bounds[-1] + length)
        k_bounds.append(k_bounds[-1] + prefix + length)
    if q_bounds[-1] != inputs.q.shape[0] or k_bounds[-1] != inputs.k.shape[0]:
        raise ValueError("Request lengths do not describe the packed Q/K/V buffers")
    for tensor, expected in ((inputs.cu_q, q_bounds), (inputs.cu_k, k_bounds)):
        if (
            tensor.dtype != torch.int32
            or tensor.device != inputs.q.device
            or tensor.shape != (len(expected),)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Packed CU metadata must be contiguous device int32 [requests+1]"
            )
        if tensor.cpu().tolist() != expected:
            raise ValueError(
                "Packed CU offsets disagree with request lengths and prefixes"
            )
    properties = torch.cuda.get_device_properties(inputs.q.device)
    if properties.gcnArchName.split(":", 1)[0] != "gfx942":
        raise ValueError("Dense QSA requires gfx942")
    calls, counts = [], []
    for request, (length, prefix) in enumerate(zip(query_lens, prefix_lens)):
        count = min(length, max(0, limit - prefix))
        counts.append(count)
        if count == 0:
            continue
        kv_count = prefix + count
        padded_kv = ((kv_count + BN - 1) // BN) * BN
        if padded_kv * inputs.k.shape[1] * D * 2 >= 2**31:
            raise ValueError("Padded KV byte offsets must fit signed int32")
        bounds = torch.tensor(
            ((0, count), (0, kv_count)), dtype=torch.int32, device=inputs.q.device
        )
        calls.append(
            DenseCall(
                request=request,
                q_start=q_bounds[request],
                q_count=count,
                k_start=k_bounds[request],
                kv_count=kv_count,
                cu_q=bounds[0],
                cu_k=bounds[1],
            )
        )
    return DensePlan(
        calls=tuple(calls),
        query_counts=tuple(counts),
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        q_shape=tuple(inputs.q.shape),
        k_shape=tuple(inputs.k.shape),
        device=inputs.q.device,
        num_cus=properties.multi_processor_count,
        limit=limit,
    )


def run(inputs, prepared: DensePlan, out: torch.Tensor) -> None:
    """Write only prepared eligible prefixes into caller-owned packed output."""
    _check_qkv(inputs)
    if (
        tuple(inputs.q.shape) != prepared.q_shape
        or tuple(inputs.k.shape) != prepared.k_shape
        or inputs.q.device != prepared.device
        or inputs.query_lens != prepared.query_lens
        or inputs.prefix_lens != prepared.prefix_lens
    ):
        raise ValueError("Request layout changed; prepare dense metadata again")
    if (
        out.shape != inputs.q.shape
        or out.dtype != inputs.q.dtype
        or out.device != inputs.q.device
        or out.layout != torch.strided
        or not out.is_contiguous()
        or out.data_ptr() % 16
        or out.requires_grad
    ):
        raise ValueError("out must be aligned, contiguous, inference-only, and match Q")
    out_begin, out_end = out.data_ptr(), out.data_ptr() + out.numel() * 2
    for tensor in (inputs.q, inputs.k, inputs.v):
        begin, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * 2
        if out_begin < end and begin < out_end:
            raise ValueError("out must not overlap Q/K/V")
    if not prepared.calls:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    for call in prepared.calls:
        q = inputs.q.narrow(0, call.q_start, call.q_count)
        k = inputs.k.narrow(0, call.k_start, call.kv_count)
        v = inputs.v.narrow(0, call.k_start, call.kv_count)
        output = out.narrow(0, call.q_start, call.q_count)
        if call.kv_count % BN == 0:
            native.run(
                q,
                k,
                v,
                call.cu_q,
                call.cu_k,
                call.q_count,
                call.kv_count,
                out=output,
                causal=True,
                softmax_scale=inputs.scale,
                stream=stream,
            )
        else:
            _run_bounded(q, k, v, output, call, inputs.scale, prepared.num_cus, stream)


def _bounded_dma(
    resource,
    storage,
    wave,
    lane,
    rows,
    tile,
    kv_len,
    hk,
    packet,
    is_v,
    extent,
    tail=False,
    read_address=None,
    read_immediate=0,
):
    token, channel = native._copy_coordinates(wave, packet)
    lane_bytes = lane * 4 if is_v else (lane ^ (native._k_phase(token) * 4)) * 4
    # rows is request-relative: the host view already includes the packed offset.
    origin = rows[0] * (hk * D * 2) + channel * 2
    row_bytes = fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(origin).ir_value()],
            f"s_add_u32 $0, $1, {packet * 4 * hk * D * 2}",
            "=s,s,~{scc}",
            has_side_effects=True,
        )
    )
    # Keep wave-uniform arithmetic scalar, but include it in the checked VOFFSET.
    offset = fx.Int32(row_bytes + lane_bytes)
    if tail:
        offset = (tile * BN + token < kv_len).select(offset, fx.Int32(extent))
    base_row, base_channel = native._copy_coordinates(wave, 0)
    destination = fx.Int32(base_row * 512 + base_channel * 2)
    immediate = (32768 if is_v else 0) + packet * 2048
    if read_address is not None:
        # Preserve the native M0 -> VMEM spacing and asynchronous DS-read waits.
        return fx.Vector(
            llvm.inline_asm(
                ir.VectorType.get([4], fx.Int32.ir_type),
                [
                    fx.Int32(read_address).ir_value(),
                    resource,
                    offset.ir_value(),
                    destination.ir_value(),
                ],
                f"s_add_u32 m0, $4, {immediate}\n"
                f"ds_read_b128 $0, $1 offset:{read_immediate}\n"
                "buffer_load_dword $3, $2, 0 offen lds",
                "=&v,v,s,v,s,~{m0},~{scc},~{memory}",
                has_side_effects=True,
            )
        )
    destination = fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [destination.ir_value()],
            f"s_add_u32 $0, $1, {immediate}",
            "=s,s,~{scc}",
            has_side_effects=True,
        )
    )
    rocdl.raw_ptr_buffer_load_lds(
        resource,
        fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(),
        offset.ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
    )


@flyc.jit
def _bounded_body(
    Q,
    K,
    V,
    O,
    CQ,
    CK,
    storage,
    head,
    qb,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    STAGGER: fx.Constexpr[bool],
):
    read_k, read_v, dma, pv = native._read_k, native._read_v, _bounded_dma, native._pv
    dma_offsets, v_operands = native._dma_offsets, native._v_operands
    qk, local_sum, local_max, cross = base._qk, base._sum, base._max, base._cross
    exps, pack, mask, center = base._exps, base._pack, base._mask, base._center
    rescale, advance_max, output_fn = _rescale, _advance_max, native._output
    q_len, kv_len = _uniform(CQ[1]), _uniform(CK[1])
    tid = fx.Int32(gpu.thread_id("x"))
    wave, lane = _uniform(tid >> 6), tid & 63
    q_start = qb * BM
    row = q_start + wave * 16 + (lane & 15)
    valid = _min(fx.Int32(BM), q_len - q_start)
    hkv = head // (H // HK)
    qptr = fx.get_iter(Q) + fx.Int64(q_start) * (H * D) + fx.Int64(head) * D
    q_extent = (valid * H - head) * D * 2
    gq = _buffer(fx.make_view(qptr, fx.make_layout(BM * H * D, 1)), q_extent)
    extent = (NK * HK - hkv) * D * 2
    gk = _buffer(
        fx.make_view(fx.get_iter(K) + hkv * D, fx.make_layout(NK * HK * D, 1)), extent
    )
    gv = _buffer(
        fx.make_view(fx.get_iter(V) + hkv * D, fx.make_layout(NK * HK * D, 1)), extent
    )
    q = base._q_fragment(gq, (wave * 16 + (lane & 15)) * H * D * 2, lane)
    scale = fx.Float32(SCALE * math.log2(math.e))
    tiles = (kv_len + BN - 1) // BN
    end = (q_start + valid + kv_len - q_len + BN - 1) // BN
    tiles = _min(tiles, (end > 0).select(end, fx.Int32(1)))
    last = tiles - 1
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    key_row = (lane & 3) + ((lane & 12) << 1)
    kr = tuple(
        _pin_i32(
            shared
            + native._k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32)
        )
        for n in range(2)
        for parity in range(2)
    )
    vr = _pin_i32(shared + native._v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    rows0 = wave & 3
    rows1 = _min(fx.Int32(1), last) * BN + (wave & 3)
    rows2 = _min(fx.Int32(2), last) * BN + (wave & 3)
    offsets = dma_offsets(rows0, 0)
    v_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(
            gk,
            storage,
            wave,
            lane,
            offsets,
            fx.Int32(0),
            kv_len,
            HK,
            packet,
            False,
            extent,
        )
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()
    k = read_k(kr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    lo = qk(q, k)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    _schedule(32, 3, 5)
    _stage_end()
    k = read_k(kr, 1)
    _wait(lgkmcnt=0)
    _stage_end()
    hi = qk(q, k)
    scores = mask(_join(lo, hi), fx.Int32(0), row, q_len, kv_len, True)
    maximum = local_max(scores)
    maximum = _maximum(maximum, maximum.shuffle_xor(16, 64))
    maximum = _maximum(maximum, maximum.shuffle_xor(32, 64))
    maximum = _maximum(maximum * scale, fx.Float32(-1.0e30)) + 1.0
    scores = center(scores, scale, maximum)
    row_sum = fx.Float32(0.0)
    _stage_end()
    offsets = dma_offsets(rows1, 0)
    current_offsets = offsets
    for packet in fx.range_constexpr(16):
        dma(
            gk,
            storage,
            wave,
            lane,
            offsets,
            _min(fx.Int32(1), last),
            kv_len,
            HK,
            packet,
            False,
            extent,
        )
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(
        previous,
        maximum,
        row_sum,
        o0,
        o1,
        previous_offsets,
        current_offsets,
        next_rows,
        t,
    ):
        previous, maximum, row_sum = (
            fx.Vector(previous),
            fx.Float32(maximum),
            fx.Float32(row_sum),
        )
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        previous_offsets = fx.Vector(previous_offsets)
        current_offsets, next_rows = fx.Vector(current_offsets), fx.Int32(next_rows)
        future_rows = _min(t + 2, last) * BN + (wave & 3)
        parts = []
        for n in fx.range_constexpr(2):
            for step in fx.range_constexpr(8):
                parts.append(
                    dma(
                        gv,
                        storage,
                        wave,
                        lane,
                        previous_offsets,
                        t - 1,
                        kv_len,
                        HK,
                        n * 8 + step,
                        True,
                        extent,
                        read_address=kr[n * 2 + step % 2],
                        read_immediate=(step // 2) * 128,
                    )
                )
        words = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        fragment = fx.make_rmem_tensor(
            fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16
        )
        fragment.store(words.bitcast(fx.BFloat16))
        k = fx.make_view(fx.get_iter(fragment), fx.make_layout((4, 2, 16), (1, 64, 4)))
        _stage_end()
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        lo = qk(q, k)
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        k = read_k(kr, 1, STAGGER)
        _wait(vmcnt=0)
        if fx.const_expr(STAGGER):
            _wait(lgkmcnt=8)
        _stage_end()
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        hi = qk(q, k)
        total, probabilities = local_sum(previous), pack(previous)
        offsets = dma_offsets(next_rows, 0)
        _schedule(32, 3, 2)
        _stage_end()
        sums = cross(total, cross_addresses)
        parts = []
        for block in fx.range_constexpr(2):
            for r in fx.range_constexpr(8):
                parts.append(
                    dma(
                        gk,
                        storage,
                        wave,
                        lane,
                        offsets,
                        _min(t + 1, last),
                        kv_len,
                        HK,
                        block * 8 + r,
                        False,
                        extent,
                        read_address=vr,
                        read_immediate=block * 16384 + r * 512,
                    )
                )
        v = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        prepared = v_operands(v, 0, True)
        _stage_end()
        current = mask(_join(lo, hi), t, row, q_len, kv_len, True)
        o0, row_sum, candidate = pv(
            probabilities,
            v,
            o0,
            True,
            prepared,
            summary_args=(current, total, sums, row_sum),
        )
        _schedule(32, 3, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        _wait(vmcnt=0)
        v = read_v(vr, 1)
        prepared = v_operands(v, 0, True)
        _stage_end()
        o1, current, new_max, ballot = pv(
            probabilities,
            v,
            o1,
            True,
            prepared,
            (current, candidate, maxima, scale, maximum),
        )
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        new_max = advance_max(maximum, new_max)
        _stage_end()
        return current, new_max, row_sum, o0, o1, current_offsets, offsets, future_rows

    # The native causal path uses two phases to limit SGPR live state.
    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t
        )
        scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, t + 1
        )
    if (tiles & 1) == 0:
        scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2 = phase(
            scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, rows2, last
        )
    offsets = fx.Vector(v_offsets)
    for packet in fx.range_constexpr(16):
        dma(
            gv,
            storage,
            wave,
            lane,
            offsets,
            last,
            kv_len,
            HK,
            packet,
            True,
            extent,
            True,
        )
    scores = exps(fx.Vector(scores))
    total = local_sum(scores)
    total = total + total.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    row_sum = fx.Float32(row_sum) + total
    probabilities = pack(scores)
    _wait(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()
    v = read_v(vr, 0)
    _wait(lgkmcnt=0)
    _stage_end()
    o0 = pv(probabilities, v, o0)
    _stage_end()
    v = read_v(vr, 1)
    _wait(lgkmcnt=0)
    _stage_end()
    o1 = pv(probabilities, v, o1)
    _stage_end()
    if fx.const_expr(not STAGGER):
        _stage_end()
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    optr = fx.get_iter(O) + fx.Int64(q_start) * (H * D) + fx.Int64(head) * D
    output = rocdl.make_buffer_tensor(
        fx.make_view(optr, fx.make_layout(BM * H * D, 1)), num_records_bytes=q_extent
    )
    output_fn(o0, o1, inv, output, storage, shared, _pin_i32(tid), H)


@flyc.kernel(name="dense_qsa_bf16_d256_bounded", known_block_size=[THREADS, 1, 1])
def _bounded_kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    CQ: fx.Tensor,
    CK: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    CUS: fx.Constexpr[int],
):
    body = _bounded_body
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    work = fx.Int32(gpu.block_id("x"))
    query_blocks = (NQ + BM - 1) // BM
    while work < H * query_blocks:
        head, qb = work // query_blocks, work % query_blocks
        group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
        if group != 0:
            body(Q, K, V, O, CQ, CK, storage, head, qb, H, HK, NK, SCALE, True)
        else:
            body(Q, K, V, O, CQ, CK, storage, head, qb, H, HK, NK, SCALE, False)
        work = work + CUS


@flyc.jit
def _bounded_launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    O: fx.Tensor,
    CQ: fx.Tensor,
    CK: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NK: fx.Constexpr[int],
    NQ: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    CUS: fx.Constexpr[int],
    stream: fx.Stream,
):
    tasks = H * ((NQ + BM - 1) // BM)
    _bounded_kernel(
        Q,
        K,
        V,
        O,
        CQ,
        CK,
        H,
        HK,
        NK,
        NQ,
        SCALE,
        CUS,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "passthrough": [["target-features", "-packed-fp32-ops"]],
        },
    ).launch(grid=(min(CUS, tasks), 1, 1), block=(THREADS, 1, 1), stream=stream)


_BOUNDED_COMPILED = {}


def _run_bounded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    call: DenseCall,
    scale: float,
    num_cus: int,
    stream: torch.cuda.Stream,
) -> None:
    args = (
        q.view(-1),
        k.view(-1),
        v.view(-1),
        out.view(-1),
        call.cu_q,
        call.cu_k,
        q.shape[1],
        k.shape[1],
        call.kv_count,
        call.q_count,
        float(scale),
        num_cus,
        stream,
    )
    key = (
        q.device,
        q.shape[1],
        k.shape[1],
        call.kv_count,
        call.q_count,
        float(scale),
        num_cus,
    )
    with torch.cuda.device(q.device), torch.cuda.stream(stream):
        compiled = _BOUNDED_COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm this bounded dense specialization before graph capture"
                )
            _BOUNDED_COMPILED[key] = flyc.compile(_bounded_launch, *args)
        else:
            compiled(*args)


__all__ = ["DenseCall", "DensePlan", "prepare", "run"]
