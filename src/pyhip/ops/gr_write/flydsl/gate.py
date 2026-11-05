# SPDX-License-Identifier: MIT
"""GR write gate: a[m, c] = 2 * sigmoid(dot(n[m], W[c]) / 4) with MFMA 4x4x4 BF16.

One workgroup holds the 4 K-slice waves of a 4-row quad. Each wave streams its
2560 columns as contiguous 1 KiB row loads, transposes every 512-column group
through a wave-private LDS tile and accumulates FP32 dot products with MFMA. A
transposed DPP butterfly reduces each wave's 16 dot products; the slice
partials meet in LDS and wave 0 writes the 16 gate values. That finish step of
quad q runs after the first group of the next quad, so its reduction and
barrier overlap the next quad's loads. Workgroups walk quads with a static
persistent stride. Workgroup 0 also zeroes the apply row counter stored after
the gate values.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import C, K, GATE_GROUP, GATE_GROUPS, GATE_PREFETCH, GATE_SLICE_COLUMNS, GATE_SLICES
from .helpers import DPP_ROR4, DPP_ROR8, NT, SWIZZLE_XOR16, dpp, load, resource, scalar, store

LOG2E = 1.4426950408889634
ROW_STRIDE = GATE_GROUP * 2 + 32  # bytes; the pad puts the 4 transposed rows on disjoint LDS banks
WAVE_LDS_WORDS = (3 * ROW_STRIDE + GATE_GROUP * 2) // 4
PARTIAL_WORDS = GATE_SLICES * 16
FINISH_GROUP = 0  # the previous quad is finished after this group of the current quad


def make_gate():
    threads = 64 * GATE_SLICES

    @fx.struct
    class SharedStorage:
        tiles: fx.Array[fx.Int32, GATE_SLICES * WAVE_LDS_WORDS, 16]
        partials: fx.Array[fx.Int32, 2 * PARTIAL_WORDS, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gr_write_gate(N: fx.Tensor, WP: fx.Tensor, A: fx.Tensor, rows: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = scalar(tid // 64)
        block, sub = lane // 4, lane % 4
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile = shared.tiles.ptr + wave * WAVE_LDS_WORDS
        partials = shared.partials.ptr
        f32x4 = ir.VectorType.get([4], fx.Float32.ir_type)
        no_permute = ir.Attribute.parse('#rocdl<mfma_perm_b none>')
        quads = (rows + 3) // 4
        workgroups = fx.Int32(fx.grid_dim.x)
        first = fx.Int32(fx.block_idx.x)

        # Zero the apply row counter (stored after the gate values) for the following kernel.
        counter_bytes = scalar((first == 0).select(fx.Int32(4), fx.Int32(0)))
        counter = resource(fx.make_view(fx.get_iter(A) + fx.Int64(rows) * C, fx.make_layout(4, 1)), counter_bytes)
        store(counter, (tid == 0).select(fx.Int32(0), fx.Int32(0x7FFFFFF0)), fx.Int32(0))

        # Prepared W fragments (MFMA A operand), loaded first so the loop never waits on them.
        weights = resource(WP)
        fragments = []
        for g in range_constexpr(GATE_GROUPS):
            for t in range_constexpr(4):
                words = load(weights, ((wave * GATE_GROUPS + g) * 4 + t) * 1024 + lane * 16)
                fragments.append(fx.Vector.from_elements([words[0], words[1]], fx.Int32).bitcast(fx.Int16))
                fragments.append(fx.Vector.from_elements([words[2], words[3]], fx.Int32).bitcast(fx.Int16))

        write_words = lane * 4
        read_words = (sub * ROW_STRIDE + 16 * block) // 4
        bit2 = (lane & 4) != 0
        bit3 = (lane & 8) != 0

        def quad_rows(q):
            valid = rows - q * 4
            valid = (valid < 4).select(valid, fx.Int32(4))
            return (valid > 0).select(valid, fx.Int32(0))

        def issue(q, g):
            column = wave * GATE_SLICE_COLUMNS + g * GATE_GROUP
            num_bytes = quad_rows(q) * (K * 2) - column * 2
            num_bytes = scalar((num_bytes > 0).select(num_bytes, fx.Int32(0)))
            source = resource(fx.make_view(fx.get_iter(N) + fx.Int64(q * 4) * K + column,
                                           fx.make_layout(4 * K, 1)), num_bytes)
            return [load(source, (r * K + 8 * lane) * 2, aux=NT) for r in range_constexpr(4)]

        def accumulate(g, data, acc):
            for r in range_constexpr(4):
                fx.make_view(tile + (r * (ROW_STRIDE // 4) + write_words), fx.make_layout(4, 1)).store(
                    fx.Vector(data[r]))
            operands = [fx.make_view(tile + (read_words + 64 * t), fx.make_layout(4, 1)).load()
                        for t in range_constexpr(4)]
            for t in range_constexpr(4):
                words = fx.Vector(operands[t])
                for h in range_constexpr(2):
                    b_operand = fx.Vector.from_elements([words[2 * h], words[2 * h + 1]], fx.Int32).bitcast(fx.Int16)
                    acc = fx.Vector(rocdl_raw.mfma_f32_4x4x4bf16_1k_(
                        f32x4, fragments[(g * 4 + t) * 2 + h].ir_value(), b_operand.ir_value(), acc.ir_value(),
                        0, 0, no_permute))
            return acc

        def finish(q, acc, slot):
            """Reduce, combine the slices and store the gates of quad q (masked when q < 0)."""
            # Lane 4*block + j holds D[c][row j] in VGPR c. Transposed butterfly over the block bits
            # 0 and 1 (lanes +4, +8), then lanes ^16 and ^32: lane l ends with row l & 3, branch (l >> 2) & 3.
            d = [fx.Float32(acc[c]) for c in range_constexpr(C)]
            low = bit2.select(d[1], d[0]) + dpp(bit2.select(d[0], d[1]), DPP_ROR4)
            high = bit2.select(d[3], d[2]) + dpp(bit2.select(d[2], d[3]), DPP_ROR4)
            total = bit3.select(high, low) + dpp(bit3.select(low, high), DPP_ROR8)
            swapped = rocdl.ds_swizzle(fx.Int32.ir_type, total.bitcast(fx.Int32).ir_value(),
                                       fx.Int32(SWIZZLE_XOR16).ir_value())
            total = total + fx.Int32(swapped).bitcast(fx.Float32)
            permuted = rocdl.ds_bpermute(fx.Int32.ir_type, ((lane ^ 32) * 4).ir_value(),
                                         total.bitcast(fx.Int32).ir_value())
            total = total + fx.Int32(permuted).bitcast(fx.Float32)
            partial_view = fx.make_view(partials + (slot * PARTIAL_WORDS + wave * 16 + (lane & 15)),
                                        fx.make_layout(1, 1))
            partial_view[0] = total.bitcast(fx.Int32)
            fx.barrier()
            # Lane l < 16 of wave 0 sums the slices of row l & 3, branch l >> 2 and stores 64 contiguous bytes.
            gate_sum = fx.Float32(0.0)
            for s in range_constexpr(GATE_SLICES):
                gate_sum = gate_sum + fx.make_view(partials + (slot * PARTIAL_WORDS + s * 16 + (lane & 15)),
                                                   fx.make_layout(1, 1))[0].bitcast(fx.Float32)
            exponent = fx.Float32(rocdl.exp2(fx.Float32.ir_type, (gate_sum * (-0.25 * LOG2E)).ir_value()))
            gate = fx.Float32(rocdl.rcp(fx.Float32.ir_type, (1.0 + exponent).ir_value())) * 2.0
            valid = (wave == 0) & (q >= 0)
            safe = (q >= 0).select(q, fx.Int32(0))
            num_bytes = scalar(valid.select(quad_rows(safe) * 16, fx.Int32(0)))
            target = resource(fx.make_view(fx.get_iter(A) + fx.Int64(safe * 4) * C, fx.make_layout(16, 1)), num_bytes)
            offset = ((lane & 3) * C + ((lane >> 2) & 3)) * 4
            store(target, (lane < 16).select(offset, fx.Int32(0x7FFFFFF0)), gate.bitcast(fx.Int32))

        first_groups = []
        for g in range_constexpr(GATE_PREFETCH):
            first_groups = first_groups + issue(first, g)
        rocdl.sched_barrier(0)
        # The loop-carried state ends with the next quad's prefetched groups, matching this prologue.
        state = [first, fx.Int32(0), fx.Int32(-1), fx.Vector.filled(4, 0.0, fx.Float32)] + first_groups
        while fx.Int32(state[0]) < quads:
            q = fx.Int32(state[0])
            iteration = fx.Int32(state[1])
            previous = fx.Int32(state[2])
            previous_acc = fx.Vector(state[3])
            q_next = q + workgroups
            buffers = [state[4 + 4 * i:8 + 4 * i] for i in range_constexpr(GATE_PREFETCH)]
            carry = []
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(GATE_GROUPS):
                if const_expr(g + GATE_PREFETCH < GATE_GROUPS):
                    buffers.append(issue(q, g + GATE_PREFETCH))
                else:
                    carry = carry + issue(q_next, g + GATE_PREFETCH - GATE_GROUPS)
                acc = accumulate(g, buffers[g], acc)
                if const_expr(g == FINISH_GROUP):
                    finish(previous, previous_acc, (iteration + 1) & 1)
            rocdl.sched_barrier(0)
            state = [q_next, iteration + 1, q, acc] + carry
        finish(fx.Int32(state[2]), fx.Vector(state[3]), (fx.Int32(state[1]) + 1) & 1)

    @flyc.jit
    def launch(N: fx.Tensor, WP: fx.Tensor, A: fx.Tensor, rows: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        gr_write_gate(N, WP, A, rows).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch
