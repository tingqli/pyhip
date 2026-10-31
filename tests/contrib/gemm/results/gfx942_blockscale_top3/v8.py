import pyhip

from .common.loaders import tb_swizzle

# (rep_n, k_step): accumulations of one partial are one MFMA apart, and each partial
# finishes early enough for one dequant FMA per later MFMA shadow.
MFMA_ORDER = [(n, k) for pair in (0, 2) for k in range(4) for n in (pair, pair + 1)]


@pyhip.jit(with_debug_log=False)
def gemm_a8w8_blockscale_8wave_gfx942(
    J,
    preshuffle_b,
    N,
    K,
    pA: "void*",
    pB: "void*",
    pC: "void*",
    pScaleA: "float*",
    pScaleB: "float*",
    M: "int",
):
    assert N % 16 == 0 and K >= 128 and K % 128 == 0
    tile_m, tile_n, tile_k = 256, 256, 128
    block_m, block_n = tb_swizzle(J, J.blockIdx.x, M, tile_m, tile_n, N, 8, 8)
    row_base = J.gpr("su32", block_m * tile_m)
    col_base = J.gpr("su32", block_n * tile_n)
    rows = J.gpr("su32")
    cols = J.gpr("su32")
    J.s_min_u32(rows, M - row_base, tile_m)
    J.s_min_u32(cols, N - col_base, tile_n)
    pA[:] += row_base * K
    pB[:] += col_base * K
    pC[:] += row_base * (N * 2)
    wave_m = J.gpr(J.warp_id[0])
    group = J.gpr("su32", J.warp_id[0] // 4)
    packet_n = 64
    packet_bytes = packet_n * tile_k
    lds_b = J.alloc_lds(4 * packet_bytes)
    staging_b = J.gpr(2, 4, "vu32")
    if preshuffle_b:
        lane_b = J.gpr(
            (wave_m // 2) * (16 * K) + (wave_m % 2) * 1024 + J.lane_id[0] * 16
        )
        write_row = J.gpr((wave_m // 2) * 16 + J.lane_id[0] % 16)
        write_col = J.gpr((wave_m % 2) * 4 + J.lane_id[0] // 16)
    else:
        lane_b = J.gpr((J.threadIdx.x // 8) * K + (J.threadIdx.x % 8) * 16)
        write_row = J.gpr(J.threadIdx.x // 8)
        write_col = J.gpr(J.threadIdx.x % 8)
    write_b = J.gpr(write_row * 128 + ((write_row ^ write_col) % 8) * 16)
    # Row-dependent offsets stay in voffset: gfx9 raw buffers do not range-check soffset.
    load_b = J.gpr(4, "vu32")
    for packet in range(4):
        load_b[packet] = lane_b + packet * (packet_n * K)
    buffer_a = J.Buffer(pA, rows * K)
    buffer_b = J.Buffer(pB, cols * K)
    buffer_c = J.Buffer(pC, rows * (N * 2))
    buffer_sa = J.Buffer(pScaleA, M * ((K // 128) * 4))
    scale_a = J.gpr(2, 2, "vf32")
    scale_b = J.gpr(2, 2, "sf32")
    local_row = J.gpr(wave_m * 16 + J.lane_id[0] % 16)
    last_row = J.gpr("su32", M - 1)
    load_scale_a = J.gpr(2, "vu32")
    for m_slice in range(2):
        load_scale_a[m_slice] = row_base + local_row + m_slice * 128
        # Rows past M are never stored; clamping keeps the K-block soffset in bounds.
        J.v_min_u32(load_scale_a[m_slice], last_row, load_scale_a[m_slice])
        load_scale_a[m_slice] = load_scale_a[m_slice] * 4
    scale_rows_b = (N + 127) // 128
    scale_offset_b = J.gpr(2, "su32")
    scale_offset_b[0] = (col_base // 128) * ((K // 128) * 4)
    scale_offset_b[1] = scale_offset_b[0] + (K // 128) * 4
    # The last N tile may have no second 128-column scale row.
    J.s_min_u32(scale_offset_b[1], scale_offset_b[1], (scale_rows_b - 1) * (K // 128) * 4)
    load_a = J.gpr(2, 2, "vu32")
    read_b = J.gpr(2, "vu32")
    for half_k in range(2):
        lane_col = J.lane_id[0] // 16 + half_k * 4
        b_row = J.lane_id[0] % 16
        for m_slice in range(2):
            load_a[m_slice, half_k] = (local_row + m_slice * 128) * K + lane_col * 16
        read_b[half_k] = b_row * 128 + ((b_row ^ lane_col) % 8) * 16
    operands_a = J.gpr(2, 2, 2, 4, "vu32")
    operands_b = J.gpr(4, 2, 4, "vu32")
    accum = J.gpr(2, 16, 4, "vf32")
    partial = J.gpr(4, 4, "vf32")
    partial_scale = J.gpr(2, "vf32")
    partial[...] = 0
    partial_scale[...] = 0
    accum[...] = 0
    k_offset = J.gpr("su32", 0)
    scale_offset = J.gpr("su32", 0)

    def prefetch_a(slot, ahead, part):
        if isinstance(part, str):
            m_slice = int(part[1])
            offset = scale_offset + M * 4 if ahead else scale_offset
            buffer_sa.load_dword(scale_a[slot, m_slice], load_scale_a[m_slice], offset)
            return
        m_slice, half_k = divmod(part, 2)
        offset = k_offset + 128 if ahead else k_offset
        buffer_a.load_dwordx4(operands_a[slot, m_slice, half_k], load_a[m_slice, half_k], offset)

    def prefetch_scale_b(slot, ahead):
        for col_index in range(2):
            offset = scale_offset_b[col_index] + k_offset // 32
            if ahead:
                offset = offset + 4
            J.s_load_dword(scale_b[slot, col_index], pScaleB, offset)

    def prefetch_b(slot, packet):
        target_k = k_offset + 128 if packet >= 4 else k_offset
        if preshuffle_b:
            target_k = target_k * 16
        buffer_b.load_dwordx4(staging_b[slot], load_b[packet % 4], target_k)

    k_blocks = K // 128
    total_packets = k_blocks * 4
    a_order = [0, 1, "s0", 2, 3, "s1"]
    a_stages = [a_order[0:2], a_order[2:4], a_order[4:6], []]

    def stage_tokens(q):
        block, local = divmod(q, 4)
        tokens = []
        if block + 1 < k_blocks:
            tokens += [("a", block + 1, part) for part in a_stages[local]]
        if q + 3 < total_packets:
            tokens.append(("b", q + 3))
        return tokens

    requests = [("b", 0)] + [("a", 0, part) for part in a_order] + [("b", 1), ("b", 2)]
    wait_budgets = []
    for q in range(total_packets):
        needed = [("b", q + 1)] if q + 1 < total_packets else []
        if q % 4 == 0 and q:
            needed += [("a", q // 4, part) for part in a_order]
        wait_budgets.append(
            len(requests) - 1 - max(requests.index(t) for t in needed) if needed else None
        )
        requests += stage_tokens(q)
    for q in range(4, total_packets - 4):
        assert wait_budgets[q] == wait_budgets[4 + (q // 4 - 1) % 2 * 4 + q % 4]

    def run_k(slot, budget_block):
        for packet in range(4):
            q = budget_block * 4 + packet
            if wait_budgets[q] is not None:
                J.s_waitcnt(mod=f"vmcnt({wait_budgets[q]})")
                J.ds_write_b128(
                    write_b, staging_b[packet % 2],
                    mod=f"offset:{lds_b + ((packet + 1) % 4) * packet_bytes}",
                )
            for token in stage_tokens(q):
                if token[0] == "b":
                    prefetch_b(packet % 2, packet + 3)
                else:
                    prefetch_a(1 - slot, 1, token[2])
            for rep_n in range(4):
                for half_k in range(2):
                    J.ds_read_b128(
                        operands_b[rep_n, half_k], read_b[half_k],
                        mod=f"offset:{lds_b + packet * packet_bytes + rep_n * 16 * 128}",
                    )
            J.s_waitcnt(mod="lgkmcnt(8)")
            J.s_barrier()
            J.s_setprio(3)
            J.s_waitcnt(mod="lgkmcnt(0)")
            # Issued after the wait so SMEM latency overlaps MFMA; consumed one K-block later.
            if packet == 1 and budget_block + 1 < k_blocks:
                prefetch_scale_b(1 - slot, 1)
            for m_slice in range(2):
                previous_packet = (packet - 1) % 4 if m_slice == 0 else packet
                for index, (rep_n, k_step) in enumerate(MFMA_ORDER):
                    half_k, word_pair = divmod(k_step, 2)
                    J.v_mfma_f32_16x16x32_fp8_fp8(
                        partial[rep_n],
                        operands_b[rep_n, half_k, word_pair * 2:word_pair * 2 + 1],
                        operands_a[slot, m_slice, half_k, word_pair * 2:word_pair * 2 + 1],
                        0 if k_step == 0 else partial[rep_n],
                    )
                    if index == 0:
                        J.v_mul_f32(
                            partial_scale[m_slice], scale_b[slot, packet // 2],
                            scale_a[slot, m_slice],
                        )
                    if index < 8:
                        dq_slice, dq_packet, dq_rep = 1 - m_slice, previous_packet, 2 + index // 4
                    else:
                        dq_slice, dq_packet, dq_rep = m_slice, packet, (index - 8) // 4
                    J.v_fmac_f32(
                        accum[dq_slice, dq_packet * 4 + dq_rep, index % 4],
                        partial[dq_rep, index % 4], partial_scale[dq_slice],
                    )
            J.s_setprio(0)
            J.s_barrier()
        k_offset[0] += 128
        scale_offset[0] += M * 4

    prefetch_b(0, 0)
    for part in a_order:
        prefetch_a(0, 0, part)
    prefetch_scale_b(0, 0)
    J.s_waitcnt(mod=f"vmcnt({len(a_order)})")
    J.ds_write_b128(write_b, staging_b[0], mod=f"offset:{lds_b}")
    prefetch_b(0, 1)
    prefetch_b(1, 2)
    J.s_waitcnt(mod="vmcnt(2) lgkmcnt(0)")
    J.s_barrier()
    with J.If(group[0] == 1):
        J.s_barrier()

    run_k(0, 0)
    if k_blocks > 1:
        middle_pairs = (k_blocks - 2) // 2
        if middle_pairs:
            with J.While(k_offset[0] < (1 + middle_pairs * 2) * 128):
                run_k(1, 1)
                run_k(0, 2)
        if (k_blocks - 2) % 2:
            run_k(1, 1)
        run_k((k_blocks - 1) % 2, k_blocks - 1)

    # The last slice's partials 2-3 have no later MFMA shadow; wait out the XDL latency.
    J.s_nop(15)
    for rep_n in (2, 3):
        for element in range(4):
            J.v_fmac_f32(
                accum[1, 12 + rep_n, element], partial[rep_n, element], partial_scale[1]
            )
    with J.If(group[0] == 0):
        J.s_barrier()

    packed = J.gpr(4, "vbf16x2")
    exchanged = J.gpr(4, "vu32")
    lane_group = J.gpr(J.lane_id[0] // 16)
    odd_group = J.gpr(lane_group & 1)
    exchange_lane = J.gpr((J.lane_id[0] ^ 16) * 4)
    output_col = J.gpr(col_base + ((lane_group & 1) * 2 + lane_group // 2) * 8)
    for m_slice in range(2):
        for rep_n in range(0, 16, 2):
            column = J.gpr(output_col + rep_n * 16)
            offset = J.gpr((local_row + m_slice * 128) * (N * 2) + column * 2)
            J.uni_cvt_pk_bf16_f32(packed[0], accum[m_slice, rep_n, 0], accum[m_slice, rep_n, 1])
            J.uni_cvt_pk_bf16_f32(packed[1], accum[m_slice, rep_n, 2], accum[m_slice, rep_n, 3])
            J.uni_cvt_pk_bf16_f32(packed[2], accum[m_slice, rep_n + 1, 0], accum[m_slice, rep_n + 1, 1])
            J.uni_cvt_pk_bf16_f32(packed[3], accum[m_slice, rep_n + 1, 2], accum[m_slice, rep_n + 1, 3])
            for word in range(4):
                J.ds_bpermute_b32(exchanged[word], exchange_lane, packed[word])
            J.s_waitcnt(mod="lgkmcnt(0)")
            J.v_cmp_ne_u32("vcc", odd_group, 0)
            for word in range(2):
                J.v_cndmask_b32(packed[word], packed[word], exchanged[word + 2], "vcc")
                J.v_cndmask_b32(packed[word + 2], exchanged[word], packed[word + 2], "vcc")
            with J.ExecMask(column < N):
                buffer_c.store_dwordx4(packed, offset, 0)