import pyhip

from .common.loaders import tb_swizzle


@pyhip.jit(with_debug_log=False)
def gemm_a8w8_blockscale_8wave_gfx942_4x2_lds(
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
    """Use the gfx950 (N, M)=(4, 2) quadrant mapping with gfx942 LDS copies."""
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
    wave_m = J.gpr("su32", J.warp_id[0] // 4)
    wave_n = J.gpr("su32", J.warp_id[0] % 4)
    lds_a = J.alloc_lds(32768)
    lds_b = J.alloc_lds(32768)
    staging_a = J.gpr(4, 4, "vu32")
    staging_b = J.gpr(2, 4, "vu32")
    copy_row = J.gpr(J.threadIdx.x // 8)
    copy_col = J.gpr(J.threadIdx.x % 8)
    load_a = J.gpr(4, "vu32")
    load_b = J.gpr(4, "vu32")
    for packet in range(4):
        load_a[packet] = (copy_row + packet * 64) * K + copy_col * 16
        if preshuffle_b:
            load_b[packet] = (
                (J.warp_id[0] // 2 + packet * 4) * (16 * K)
                + (J.warp_id[0] % 2) * 1024
                + J.lane_id[0] * 16
            )
        else:
            load_b[packet] = load_a[packet]
    write_a = J.gpr(copy_row * 128 + ((copy_row ^ copy_col) % 8) * 16)
    if preshuffle_b:
        copy_b_row = J.gpr((J.warp_id[0] // 2) * 16 + J.lane_id[0] % 16)
        copy_b_col = J.gpr((J.warp_id[0] % 2) * 4 + J.lane_id[0] // 16)
        write_b = J.gpr(copy_b_row * 128 + ((copy_b_row ^ copy_b_col) % 8) * 16)
    else:
        write_b = write_a
    local_row = J.gpr(wave_m * 16 + J.lane_id[0] % 16)
    read_a = J.gpr(2, 2, "vu32")
    read_b = J.gpr(2, 2, "vu32")
    for half_k in range(2):
        lane_col = J.lane_id[0] // 16 + half_k * 4
        for rep_m in range(2):
            row = local_row + rep_m * 32
            read_a[rep_m, half_k] = row * 128 + ((row ^ lane_col) % 8) * 16
        for rep_n in range(2):
            column = wave_n * 16 + rep_n * 64 + J.lane_id[0] % 16
            read_b[rep_n, half_k] = column * 128 + ((column ^ lane_col) % 8) * 16
    operands_a = J.gpr(2, 2, 4, "vu32")
    operands_b = J.gpr(2, 2, 2, 4, "vu32")
    scale_a = J.gpr(2, 2, "vf32")
    scale_b = J.gpr(2, 2, "sf32")
    scale_a_address = J.gpr(2, "vu32")
    last_row = J.gpr("su32", M - 1)
    scale_b_offset = J.gpr(2, "su32")
    scale_b_offset[0] = (col_base // 128) * (K // 128) * 4
    scale_b_offset[1] = scale_b_offset[0] + (K // 128) * 4
    J.s_min_u32(scale_b_offset[1], scale_b_offset[1], ((N - 1) // 128) * (K // 128) * 4)
    accum = J.gpr(4, 2, 4, 4, "vf32")
    partial = J.gpr(4, 4, "vf32")
    partial_scale = J.gpr(2, "vf32")
    accum[...] = 0
    partial[...] = 0
    partial_scale[...] = 0
    k_offset = J.gpr("su32", 0)
    scale_offset = J.gpr("su32", 0)

    def prefetch_a(packet, ahead):
        offset = k_offset + 128 if ahead else k_offset
        buffer_a.load_dwordx4(staging_a[packet], load_a[packet], offset)

    def prefetch_b(packet, ahead):
        offset = k_offset + 128 if ahead else k_offset
        if preshuffle_b:
            offset = offset * 16
        buffer_b.load_dwordx4(staging_b[packet % 2], load_b[packet], offset)

    def prefetch_scale_a(slot, packet, ahead):
        for rep_m in range(2):
            scale_a_address[rep_m] = row_base + local_row + packet * 64 + rep_m * 32
            J.v_min_u32(scale_a_address[rep_m], last_row, scale_a_address[rep_m])
            scale_a_address[rep_m] = scale_a_address[rep_m] * 4
            offset = scale_offset + M * 4 if ahead else scale_offset
            buffer_sa.load_dword(scale_a[slot, rep_m], scale_a_address[rep_m], offset)

    def prefetch_scale_b(slot, ahead):
        for side in range(2):
            offset = scale_b_offset[side] + k_offset // 32
            if ahead:
                offset = offset + 4
            J.s_load_dword(scale_b[slot, side], pScaleB, offset)

    def write_a_packet(packet):
        J.ds_write_b128(
            write_a, staging_a[packet], mod=f"offset:{lds_a + packet * 8192}"
        )

    def write_b_packet(packet):
        J.ds_write_b128(
            write_b, staging_b[packet % 2], mod=f"offset:{lds_b + packet * 8192}"
        )

    for packet in range(4):
        prefetch_a(packet, False)
    for pair_begin in (0, 2):
        prefetch_b(pair_begin, False)
        prefetch_b(pair_begin + 1, False)
        J.s_waitcnt(mod="vmcnt(0)")
        write_b_packet(pair_begin)
        write_b_packet(pair_begin + 1)
        J.s_waitcnt(mod="lgkmcnt(0)")
    prefetch_scale_a(0, 0, False)
    prefetch_scale_b(0, False)
    J.s_waitcnt(mod="vmcnt(0)")
    for packet in range(4):
        write_a_packet(packet)
    J.s_waitcnt(mod="lgkmcnt(0)")
    J.s_barrier()
    with J.If(wave_m[0] == 1):
        J.s_barrier()

    k_blocks = K // 128

    def run_k(slot, block_index):
        has_next = block_index + 1 < k_blocks
        for packet in range(4):
            J.s_waitcnt(mod="vmcnt(0)")
            if packet == 0 and block_index:
                write_a_packet(3)
            if packet and has_next:
                write_a_packet(packet - 1)
                write_b_packet(packet - 1)
                if packet == 3:
                    write_b_packet(3)
            if packet == 0:
                for side in range(2):
                    for rep_n in range(2):
                        for half_k in range(2):
                            J.ds_read_b128(
                                operands_b[side, rep_n, half_k],
                                read_b[rep_n, half_k],
                                mod=f"offset:{lds_b + side * 16384}",
                            )
            for rep_m in range(2):
                for half_k in range(2):
                    J.ds_read_b128(
                        operands_a[rep_m, half_k],
                        read_a[rep_m, half_k],
                        mod=f"offset:{lds_a + packet * 8192}",
                    )
            if packet < 3:
                prefetch_scale_a(1 - packet % 2, packet + 1, False)
            elif has_next:
                prefetch_scale_a(0, 0, True)
            if has_next:
                prefetch_a(packet, True)
                if packet < 3:
                    prefetch_b(packet, True)
                if packet == 2:
                    prefetch_b(3, True)
            J.s_waitcnt(mod="lgkmcnt(0)")
            J.s_barrier()
            J.s_setprio(3)
            if packet == 0 and has_next:
                prefetch_scale_b(1 - slot, True)
            for side in range(2):
                previous_packet = (packet - 1) % 4 if side == 0 else packet
                for pair_begin in (0, 2):
                    for k_step in range(4):
                        half_k, word_pair = divmod(k_step, 2)
                        for pair_lane in range(2):
                            partial_index = pair_begin + pair_lane
                            index = pair_begin * 4 + k_step * 2 + pair_lane
                            J.v_mfma_f32_16x16x32_fp8_fp8(
                                partial[partial_index],
                                operands_b[
                                    side,
                                    pair_lane,
                                    half_k,
                                    word_pair * 2 : word_pair * 2 + 1,
                                ],
                                operands_a[
                                    pair_begin // 2,
                                    half_k,
                                    word_pair * 2 : word_pair * 2 + 1,
                                ],
                                0 if k_step == 0 else partial[partial_index],
                            )
                            if index in (0, 8):
                                J.v_mul_f32(
                                    partial_scale[index // 8],
                                    scale_a[packet % 2, index // 8],
                                    scale_b[slot, side],
                                )
                            if index < 8:
                                J.v_fmac_f32(
                                    accum[
                                        previous_packet,
                                        1 - side,
                                        2 + index // 4,
                                        index % 4,
                                    ],
                                    partial[2 + index // 4, index % 4],
                                    partial_scale[1],
                                )
                            else:
                                J.v_fmac_f32(
                                    accum[packet, side, (index - 8) // 4, index % 4],
                                    partial[(index - 8) // 4, index % 4],
                                    partial_scale[0],
                                )
            J.s_setprio(0)
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
    for partial_index in (2, 3):
        for element in range(4):
            J.v_fmac_f32(
                accum[3, 1, partial_index, element],
                partial[partial_index, element],
                partial_scale[1],
            )
    with J.If(wave_m[0] == 0):
        J.s_barrier()
    packed = J.gpr(2, "vbf16x2")
    output_col = J.gpr(col_base + wave_n * 16 + (J.lane_id[0] // 16) * 4)
    for packet in range(4):
        for side in range(2):
            for rep_m in range(2):
                for rep_n in range(2):
                    partial_index = rep_m * 2 + rep_n
                    column = J.gpr(output_col + side * 128 + rep_n * 64)
                    offset = J.gpr(
                        (local_row + packet * 64 + rep_m * 32) * (N * 2) + column * 2
                    )
                    for pair in range(2):
                        J.uni_cvt_pk_bf16_f32(
                            packed[pair],
                            accum[packet, side, partial_index, pair * 2],
                            accum[packet, side, partial_index, pair * 2 + 1],
                        )
                    with J.ExecMask(column < N):
                        buffer_c.store_dwordx2(packed, offset, 0)
