# SPDX-License-Identifier: MIT
"""GR write apply: out = bf16(r + a * y), normed = bf16((out * rstd) * (1 + w)) per branch.

One workgroup processes one row at a time with 4 waves, one per SIMD. Lane t of
wave w owns 5 chunks of 8 columns: chunk b < 4 is branch b at columns 512w + 8t,
chunk 4 is branch w at columns 2048 + 8t. Every load and store moves 1 KiB per
wave and all waves do the same work. Each wave reduces its four branch sums with
a transposed DPP butterfly, and the waves combine them through LDS at one
barrier per row.

Rows are claimed in order from an atomic counter (zeroed by the gate kernel) so
concurrently processed rows stay close in memory. Two register sets ping-pong:
row k+1 is loaded while row k is computed. A claim is issued before the
prefetch it accompanies and is consumed one row later, because waiting on an
atomic (vmcnt is in-order) also waits on every older load and store. The ten
row stores are issued together after the normalization, which keeps each
wave's memory traffic in one burst per row. Per-branch scales (gates, rstd)
are read into SGPRs with readlane: VGPR splats let the register allocator pair
them with an in-flight load register, which forced full vmcnt drains.

Three workgroups fit per CU only while the kernel stays at <= 168 VGPRs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import C, H, K, APPLY_CHUNKS, APPLY_WAVES
from .helpers import (DPP_XOR1, DPP_XOR2, HIGH_HALF, NT, bf16_pair, dpp, lane_reduce_tail, load, make_claim,
                      pack_high, pair, pk_fma, resource, rne_bits, scalar, store)

PARTIAL_WORDS = APPLY_WAVES * 4  # one slot: [wave][branch]
TAIL_COLUMN = APPLY_WAVES * 512  # first column of chunk 4


def make_apply(eps):
    threads = 64 * APPLY_WAVES
    eps = float(eps)

    @fx.struct
    class SharedStorage:
        partials: fx.Array[fx.Int32, 2 * PARTIAL_WORDS, 16]
        claims: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gr_write_apply(Y: fx.Tensor, R: fx.Tensor, A: fx.Tensor, G: fx.Tensor, OUT: fx.Tensor,
                       NORMED: fx.Tensor, rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = scalar(tid // 64)
        lane_bytes = lane * 16
        # Byte offsets of this lane's chunks inside a row of r / out / normed, and inside a row of y.
        chunk_bytes = [b * (H * 2) + wave * 1024 + lane_bytes for b in range_constexpr(C)]
        chunk_bytes.append(wave * (H * 2) + TAIL_COLUMN * 2 + lane_bytes)
        y_bytes = [wave * 1024 + lane_bytes, TAIL_COLUMN * 2 + lane_bytes]
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        partials = shared.partials.ptr
        claims = shared.claims.ptr
        workgroups = fx.Int32(fx.grid_dim.x)
        # Rows bid and bid + workgroups are static; dynamic claims start at 2 * workgroups.
        claim = make_claim(fx.Int64(fx.ptrtoint(fx.get_iter(A))) + fx.Int64(rows) * (C * 4))

        def claim_tid0():
            value = fx.Int32(0)
            if tid == 0:
                value = claim()
            return value

        # Prepared (1 + w) pairs for this lane's chunks, loaded first.
        gains = resource(G, K * 4)
        gain_words = [load(gains, (((wave * APPLY_CHUNKS + c) * 2 + h) * 64 + lane) * 16)
                      for c in range_constexpr(APPLY_CHUNKS) for h in range_constexpr(2)]
        gain = [[pair(gain_words[c * 2 + d // 2][(d % 2) * 2].bitcast(fx.Float32),
                      gain_words[c * 2 + d // 2][(d % 2) * 2 + 1].bitcast(fx.Float32))
                 for d in range_constexpr(4)] for c in range_constexpr(APPLY_CHUNKS)]

        def row_view(tensor, m, width):
            return fx.make_view(fx.get_iter(tensor) + fx.Int64(m) * width, fx.make_layout(width, 1))

        def chunk_values(value):
            """Uniform values of lanes 0..3 (branches) plus lane `wave` (chunk 4), read into SGPRs."""
            lanes = [fx.Int32(b) for b in range_constexpr(C)] + [wave]
            return [fx.Int32(rocdl.readlane(fx.Int32.ir_type, value.bitcast(fx.Int32).ir_value(), index.ir_value()))
                    .bitcast(fx.Float32) for index in lanes]

        def issue(m):
            valid = m < rows
            safe = valid.select(m, fx.Int32(0))
            a_bytes = scalar(valid.select(fx.Int32(C * 4), fx.Int32(0)))
            y_rsrc = resource(row_view(Y, safe, H), scalar(valid.select(fx.Int32(H * 2), fx.Int32(0))))
            r_rsrc = resource(row_view(R, safe, K), scalar(valid.select(fx.Int32(K * 2), fx.Int32(0))))
            # Default cache policy: the y tail read by all four waves then hits in L1 (NT measured slower).
            y = [load(y_rsrc, offset) for offset in y_bytes]
            residual = [load(r_rsrc, offset) for offset in chunk_bytes]
            gate = load(resource(row_view(A, safe, C), a_bytes), (lane & 3) * 4, words=1)  # a[m, lane & 3]
            return [gate] + y + residual

        def process(m, data, slot, pending_claim):
            gate_word, y, residual = data[0], data[1:3], data[3:]
            gates = chunk_values(gate_word)
            out_bytes = scalar((m < rows).select(fx.Int32(K * 2), fx.Int32(0)))
            out_rsrc = resource(row_view(OUT, m, K), out_bytes)
            normed_rsrc = resource(row_view(NORMED, m, K), out_bytes)
            y_pairs = [[bf16_pair(y[i][d]) for d in range_constexpr(4)] for i in range_constexpr(2)]
            out_words = []
            out_pairs = []
            sums = []
            for c in range_constexpr(APPLY_CHUNKS):
                scale = pair(gates[c], gates[c])
                y_chunk = y_pairs[0] if const_expr(c < C) else y_pairs[1]
                acc = pair(fx.Float32(0.0), fx.Float32(0.0))
                words = []
                for d in range_constexpr(4):
                    value = pk_fma(y_chunk[d], scale, bf16_pair(residual[c][d]))
                    lo_bits = rne_bits(fx.Float32(value[0]))
                    hi_bits = rne_bits(fx.Float32(value[1]))
                    words.append(pack_high(hi_bits, lo_bits))
                    rounded = pair((lo_bits & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32),
                                   (hi_bits & fx.Uint32(HIGH_HALF)).bitcast(fx.Float32))
                    out_pairs.append(rounded)
                    acc = pk_fma(rounded, rounded, acc)
                out_words.append(fx.Vector.from_elements(words, fx.Uint32).bitcast(fx.Int32))
                sums.append(fx.Float32(acc[0]) + fx.Float32(acc[1]))
            # Chunk 4 belongs to branch `wave`.
            sums = [sums[b] + (wave == b).select(sums[C], fx.Float32(0.0)) for b in range_constexpr(C)]
            # Transposed butterfly: afterwards every lane holds the wave sum of branch (lane & 3).
            bit0 = (lane & 1) != 0
            bit1 = (lane & 2) != 0
            low = bit0.select(sums[1], sums[0]) + dpp(bit0.select(sums[0], sums[1]), DPP_XOR1)
            high = bit0.select(sums[3], sums[2]) + dpp(bit0.select(sums[2], sums[3]), DPP_XOR1)
            total = bit1.select(high, low) + dpp(bit1.select(low, high), DPP_XOR2)
            total = lane_reduce_tail(total, lane)
            # Lanes sharing (lane & 3) store identical values; wave 0 publishes tid 0's claim.
            partial_view = fx.make_view(partials + (slot * PARTIAL_WORDS + wave * 4 + (lane & 3)), fx.make_layout(1, 1))
            partial_view[0] = total.bitcast(fx.Int32)
            claim_view = fx.make_view(claims + (slot + (wave != 0).select(fx.Int32(4), fx.Int32(0))),
                                      fx.make_layout(1, 1))
            claim_view[0] = scalar(pending_claim) + 2 * workgroups
            # Let at most the last 4 prefetch loads stay in flight across the barrier: throttling
            # every wave here measured faster than leaving loads and stores outstanding.
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=4)
            rocdl.sched_barrier(0)
            fx.barrier()
            published = scalar(fx.make_view(claims + slot, fx.make_layout(1, 1))[0])
            # Lane l sums branch (l & 3) over the waves; lanes 0..3 and lane `wave` then supply the scales.
            branch_sum = None
            for w in range_constexpr(APPLY_WAVES):
                wave_sum = fx.make_view(partials + (slot * PARTIAL_WORDS + w * 4 + (lane & 3)),
                                        fx.make_layout(1, 1))[0].bitcast(fx.Float32)
                branch_sum = wave_sum if branch_sum is None else branch_sum + wave_sum
            rstds = chunk_values(fx.Float32(rocdl.rsq(fx.Float32.ir_type, (branch_sum * (1.0 / H) + eps).ir_value())))
            normed_words = []
            for c in range_constexpr(APPLY_CHUNKS):
                rstd_pair = pair(rstds[c], rstds[c])
                words = []
                for d in range_constexpr(4):
                    value = (out_pairs[c * 4 + d] * rstd_pair) * gain[c][d]
                    words.append(pack_high(rne_bits(fx.Float32(value[1])), rne_bits(fx.Float32(value[0]))))
                normed_words.append(fx.Vector.from_elements(words, fx.Uint32).bitcast(fx.Int32))
            rocdl.sched_barrier(0)
            for c in range_constexpr(APPLY_CHUNKS):
                store(out_rsrc, chunk_bytes[c], out_words[c], aux=NT)
                store(normed_rsrc, chunk_bytes[c], normed_words[c], aux=NT)
            rocdl.sched_barrier(0)
            return published

        current = fx.Int32(fx.block_idx.x)
        following = current + workgroups
        pending = claim_tid0()
        first_row = issue(current)
        rocdl.sched_barrier(0)
        # Out-of-range stores give the loop header the same vmcnt history as the steady state
        # (the row stores after the carried loads), so LLVM does not wait on the prefetch there.
        dummy = resource(fx.make_view(fx.get_iter(OUT), fx.make_layout(H, 1)), fx.Int32(0))
        for j in range_constexpr(2 * APPLY_CHUNKS):
            store(dummy, lane_bytes + j * 16, fx.Vector.filled(4, 0, fx.Int32), aux=NT)
        rocdl.sched_barrier(0)
        state = [current, following, pending] + first_row
        while fx.Int32(state[0]) < rows:
            row_a = fx.Int32(state[0])
            row_b = fx.Int32(state[1])
            claim_a = fx.Int32(state[2])
            data_a = state[3:]
            rocdl.sched_barrier(0)
            claim_b = claim_tid0()
            rocdl.sched_barrier(0)
            data_b = issue(row_b)
            rocdl.sched_barrier(0)
            row_c = process(row_a, data_a, 0, claim_a)
            rocdl.sched_barrier(0)
            claim_c = claim_tid0()
            rocdl.sched_barrier(0)
            data_c = issue(row_c)
            rocdl.sched_barrier(0)
            # Runs unconditionally (masked when row_b >= rows) to keep one vmcnt history per loop path.
            row_d = process(row_b, data_b, 1, claim_b)
            rocdl.sched_barrier(0)
            state = [row_c, row_d, claim_c] + data_c

    @flyc.jit
    def launch(Y: fx.Tensor, R: fx.Tensor, A: fx.Tensor, G: fx.Tensor, OUT: fx.Tensor, NORMED: fx.Tensor,
               rows: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        gr_write_apply(Y, R, A, G, OUT, NORMED, rows).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch
