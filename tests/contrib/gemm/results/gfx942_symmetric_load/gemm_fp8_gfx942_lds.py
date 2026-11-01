import pyhip

from .common.loaders import tb_swizzle


@pyhip.jit(with_debug_log=False)
def gemm_a8w8_blockscale_8wave_gfx942(
    J, preshuffle_b, N, K, pA: "void*", pB: "void*", pC: "void*",
    pScaleA: "float*", pScaleB: "float*", M: "int",
):
    assert N % 16 == 0 and K >= 128 and K % 128 == 0
    block_m, block_n = tb_swizzle(J, J.blockIdx.x, M, 256, 256, N, 8, 8)
    row_base = J.gpr("su32", block_m * 256)
    col_base = J.gpr("su32", block_n * 256)
    rows = J.gpr("su32")
    cols = J.gpr("su32")
    J.s_min_u32(rows, M - row_base, 256)
    J.s_min_u32(cols, N - col_base, 256)
    pA[:] += row_base * K
    pB[:] += col_base * K
    pC[:] += row_base * (N * 2)
    buffer_a = J.Buffer(pA, rows * K)
    buffer_b = J.Buffer(pB, cols * K)
    buffer_c = J.Buffer(pC, rows * (N * 2))
    buffer_sa = J.Buffer(pScaleA, M * (K // 128) * 4)
    wave = J.gpr(J.warp_id[0])
    group = J.gpr("su32", J.warp_id[0] // 4)
    lds_a = J.alloc_lds(32768)
    lds_b = J.alloc_lds(32768)
    staging_a = J.gpr(4, 4, "vu32")
    staging_b = J.gpr(4, 4, "vu32")
    copy_row = J.gpr(J.threadIdx.x // 8)
    copy_col = J.gpr(J.threadIdx.x % 8)
    load_a = J.gpr(4, "vu32")
    load_b = J.gpr(4, "vu32")
    for packet in range(4):
        load_a[packet] = (copy_row + packet * 64) * K + copy_col * 16
        if preshuffle_b:
            load_b[packet] = (wave // 2 + packet * 4) * (16 * K) + (wave % 2) * 1024 + J.lane_id[0] * 16
        else:
            load_b[packet] = load_a[packet]
    write_a = J.gpr(copy_row * 128 + ((copy_row ^ copy_col) % 8) * 16)
    if preshuffle_b:
        b_row = J.gpr((wave // 2) * 16 + J.lane_id[0] % 16)
        b_col = J.gpr((wave % 2) * 4 + J.lane_id[0] // 16)
        write_b = J.gpr(b_row * 128 + ((b_row ^ b_col) % 8) * 16)
    else:
        write_b = write_a
    local_row = J.gpr(wave * 16 + J.lane_id[0] % 16)
    read_a = J.gpr(2, "vu32")
    read_b = J.gpr(2, "vu32")
    for half_k in range(2):
        lane_col = J.lane_id[0] // 16 + half_k * 4
        read_a[half_k] = local_row * 128 + ((local_row ^ lane_col) % 8) * 16
        b_row = J.lane_id[0] % 16
        read_b[half_k] = b_row * 128 + ((b_row ^ lane_col) % 8) * 16
    operands_a = J.gpr(2, 2, 4, "vu32")
    operands_b = J.gpr(4, 2, 4, "vu32")
    scale_a = J.gpr(2, 2, "vf32")
    scale_b = J.gpr(2, 2, "sf32")
    scale_a_address = J.gpr(2, "vu32")
    last_row = J.gpr("su32", M - 1)
    for m_slice in range(2):
        scale_a_address[m_slice] = row_base + local_row + m_slice * 128
        J.v_min_u32(scale_a_address[m_slice], last_row, scale_a_address[m_slice])
        scale_a_address[m_slice] = scale_a_address[m_slice] * 4
    scale_b_offset = J.gpr(2, "su32")
    scale_b_offset[0] = (col_base // 128) * (K // 128) * 4
    scale_b_offset[1] = scale_b_offset[0] + (K // 128) * 4
    J.s_min_u32(scale_b_offset[1], scale_b_offset[1], ((N - 1) // 128) * (K // 128) * 4)
    accum = J.gpr(2, 16, 4, "vf32")
    partial = J.gpr(4, 4, "vf32")
    partial_scale = J.gpr(2, "vf32")
    accum[...] = 0
    partial[...] = 0
    partial_scale[...] = 0
    k_offset = J.gpr("su32", 0)
    scale_offset = J.gpr("su32", 0)

    def prefetch(packet, ahead):
        offset = k_offset + 128 if ahead else k_offset
        buffer_a.load_dwordx4(staging_a[packet], load_a[packet], offset)
        b_offset = offset * 16 if preshuffle_b else offset
        buffer_b.load_dwordx4(staging_b[packet], load_b[packet], b_offset)

    def prefetch_scale_a(slot, m_slice, ahead):
        offset = scale_offset + M * 4 if ahead else scale_offset
        buffer_sa.load_dword(scale_a[slot, m_slice], scale_a_address[m_slice], offset)

    def prefetch_scale_b(slot, ahead):
        for col_index in range(2):
            offset = scale_b_offset[col_index] + k_offset // 32
            if ahead:
                offset = offset + 4
            J.s_load_dword(scale_b[slot, col_index], pScaleB, offset)

    def write_packet(packet, write_activation=True, write_weight=True):
        if write_activation:
            J.ds_write_b128(write_a, staging_a[packet], mod=f"offset:{lds_a + packet * 8192}")
        if write_weight:
            J.ds_write_b128(write_b, staging_b[packet], mod=f"offset:{lds_b + packet * 8192}")

    for packet in range(4):
        prefetch(packet, False)
    for m_slice in range(2):
        prefetch_scale_a(0, m_slice, False)
    prefetch_scale_b(0, False)
    J.s_waitcnt(mod="vmcnt(0)")
    for packet in range(4):
        write_packet(packet)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_barrier()
    with J.If(group[0] == 1):
        J.s_barrier()

    k_blocks = K // 128

    def run_k(slot, block_index):
        has_next = block_index + 1 < k_blocks
        for packet in range(4):
            if packet == 0:
                if block_index:
                    write_packet(3, write_activation=False)
                for m_slice in range(2):
                    for half_k in range(2):
                        J.ds_read_b128(operands_a[m_slice, half_k], read_a[half_k], mod=f"offset:{lds_a + m_slice * 16384}")
            if packet in (1, 2) and has_next:
                J.s_waitcnt(mod="vmcnt(2)")
                write_packet(packet - 1, write_weight=False)
            if packet == 3 and has_next:
                J.s_waitcnt(mod="vmcnt(0)")
                # The lagging group still needs B3; publish its replacement next K's packet 0.
                for next_packet in range(4):
                    write_packet(next_packet, write_activation=next_packet >= 2, write_weight=next_packet < 3)
            for rep_n in range(4):
                for half_k in range(2):
                    J.ds_read_b128(operands_b[rep_n, half_k], read_b[half_k], mod=f"offset:{lds_b + packet * 8192 + rep_n * 2048}")
                if has_next and packet < 3 and rep_n == 1:
                    prefetch(packet, True)
                    if packet < 2:
                        prefetch_scale_a(1 - slot, packet, True)
                    else:
                        prefetch(3, True)
            J.s_waitcnt(mod=f"lgkmcnt({0 if packet == 3 else 8})")
            J.s_barrier()
            J.s_setprio(3)
            J.s_waitcnt(mod="lgkmcnt(0)")
            if packet == 0 and has_next:
                prefetch_scale_b(1 - slot, True)
            for m_slice in range(2):
                previous_packet = (packet - 1) % 4 if m_slice == 0 else packet
                for pair_begin in (0, 2):
                    for k_step in range(4):
                        half_k, word_pair = divmod(k_step, 2)
                        for pair_lane in range(2):
                            rep_n = pair_begin + pair_lane
                            index = pair_begin * 4 + k_step * 2 + pair_lane
                            if preshuffle_b and m_slice == 1 and index == 15:
                                J.s_barrier()
                            J.v_mfma_f32_16x16x32_fp8_fp8(partial[rep_n], operands_b[rep_n, half_k, word_pair * 2:word_pair * 2 + 1], operands_a[m_slice, half_k, word_pair * 2:word_pair * 2 + 1], 0 if k_step == 0 else partial[rep_n])
                            if index == 0:
                                J.v_mul_f32(partial_scale[m_slice], scale_b[slot, packet // 2], scale_a[slot, m_slice])
                            if index < 8:
                                dq_slice, dq_packet, dq_rep = 1 - m_slice, previous_packet, 2 + index // 4
                            else:
                                dq_slice, dq_packet, dq_rep = m_slice, packet, (index - 8) // 4
                            J.v_fmac_f32(accum[dq_slice, dq_packet * 4 + dq_rep, index % 4], partial[dq_rep, index % 4], partial_scale[dq_slice])
            J.s_setprio(0)
            if not preshuffle_b:
                J.s_barrier()
        k_offset[0] += 128
        scale_offset[0] += M * 4

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

    J.s_nop(15)
    for rep_n in (2, 3):
        for element in range(4):
            J.v_fmac_f32(accum[1, 12 + rep_n, element], partial[rep_n, element], partial_scale[1])
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
            for neighbor in range(2):
                for pair in range(2):
                    J.uni_cvt_pk_bf16_f32(packed[neighbor * 2 + pair], accum[m_slice, rep_n + neighbor, pair * 2], accum[m_slice, rep_n + neighbor, pair * 2 + 1])
            for word in range(4):
                J.ds_bpermute_b32(exchanged[word], exchange_lane, packed[word])
            J.s_waitcnt(mod="lgkmcnt(0)")
            J.v_cmp_ne_u32("vcc", odd_group, 0)
            for word in range(2):
                J.v_cndmask_b32(packed[word], packed[word], exchanged[word + 2], "vcc")
                J.v_cndmask_b32(packed[word + 2], exchanged[word], packed[word + 2], "vcc")
            with J.ExecMask(column < N):
                buffer_c.store_dwordx4(packed, offset, 0)