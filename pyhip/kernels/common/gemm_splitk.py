from pyhip import cudaPerf, jit, JIT
import torch

def div_up(x, y):
    return (x + y - 1) // y

# TODO: refine gemm interface
def gemm_splitk(J:JIT,
                weight_dtype,
                K,
                N,
                num_split_k,
                buff_a,
                buff_b,
                p_w_scale,
                voffset_a,
                voffset_b,
                voffset_scale,
                C_reg,
                soffset_a = 0,
                soffset_b = 0,
                BLOCK_TILE_SIZE_N = 32,
                BLOCK_TILE_SIZE_M = 16,
                ):
    assert BLOCK_TILE_SIZE_M % 16 == 0, f'BLOCK_TILE_SIZE_M must be multiple of 16, current {BLOCK_TILE_SIZE_M=}'
    assert BLOCK_TILE_SIZE_N % 32 == 0, f'BLOCK_TILE_SIZE_N must be multiple of 32, current {BLOCK_TILE_SIZE_N=}'
    sizeof_f32 = 4
    sizeof_bf16 = 2
    sizeof_w = sizeof_bf16 if weight_dtype == torch.bfloat16 else 1
    # 16 elements for a if fp8
    a_element_num_per_thread = 8 if weight_dtype == torch.bfloat16 else 16

    soffset_kb = J.gpr("su32")
    soffset_ka = J.gpr("su32")
    soffset_kb[0] = soffset_b
    soffset_ka[0] = soffset_a

    # num block in A vert direction
    A_vert = BLOCK_TILE_SIZE_M // 16
    # num block in B horz direction
    B_horz = BLOCK_TILE_SIZE_N // 16
    # there is 16 elements per weight read if fp8, so A should be double read
    A_rep = 1 if weight_dtype == torch.bfloat16 else 2
    # A_reg layout:
    # pinpong  index for vert direction                 index for different mem read(x16bytes)   index for different mfma   minimal for one mfma
    # pinpong  dword4x[?]                               dword4[?]                                dword2[?]                  dword[?](for mfma)
    A_reg = J.gpr(2, A_vert, A_rep, 2, 2, "abf16x2") # 8-bf16 == DWORDx4
    # B_reg layout:
    # bf16: pinpong  n(diff N)/index for different mem read(x16bytes)  index for different mfma  minimal for one mfma
    # fp8:  ..       ..                                                ..                        index for different mfma
    B_reg = J.gpr(2, B_horz, 2, 2, "vbf16x2") # 8-bf16 == DWORDx4

    if weight_dtype != torch.bfloat16:
        v_w_scale = J.gpr(B_horz, 2, 'vf32')
        for n in range(B_horz):
            J.global_load_dword(v_w_scale[n, 0], voffset_scale[n], p_w_scale)

    # ping pong register buffer id
    pp_reg_id = 0
    k_step_wg = num_split_k * 32 if weight_dtype == torch.bfloat16 else num_split_k * 64

    # (A0.B0.C0.D0.A1.B1.C1.D1)[3, 2, 7, 6] = (A1.B1.A0.B0)
    pattern_cvt_bf16 = J.gpr("su32")
    pattern_cvt_bf16[0] = 0x03_02_07_06
    k_max = div_up(K, k_step_wg)
    s_cvt_bf16_bias = J.gpr(1, "su32")
    s_cvt_bf16_bias[0] = 0x00008000

    def load_gen(pp_reg_id):
        for m in range(A_vert):
            buff_a.load_dwordx4(A_reg[pp_reg_id, m, 0], voffset_a[m], soffset_ka)
            if weight_dtype != torch.bfloat16:
                yield buff_a.load_dwordx4(A_reg[pp_reg_id, m, 1], voffset_a[m], soffset_ka + 16)
        for n in range(B_horz):
            yield buff_b.load_dwordx4(B_reg[pp_reg_id, n], voffset_b[n], soffset_kb)

        soffset_kb[0] = soffset_kb[0] + 16 * 64 * num_split_k
        soffset_ka[0] = soffset_ka[0] + a_element_num_per_thread * 4 * num_split_k * sizeof_bf16
        if weight_dtype == torch.bfloat16:
            J.s_waitcnt(mod=f"vmcnt({B_horz + A_vert})")
        else:
            J.s_waitcnt(mod=f"vmcnt({B_horz + A_vert * 2})")
    
    def mfma_gen(pp_reg_id):
        if weight_dtype != torch.bfloat16:
            # decompress
            v_w_f32 = J.gpr(2, 2, 'vf32', align=4)
            v_w_bf16 = J.gpr(B_horz, 2, 'vf32', align=4)
            for i in range(2):
                for j in range(2):
                    for n in range(B_horz):
                        J.v_cvt_pk_f32_fp8(v_w_f32[0], B_reg[pp_reg_id, n, i, j])
                        J.v_cvt_pk_f32_fp8_sdwa(v_w_f32[1], B_reg[pp_reg_id, n, i, j], mod='src0_sel:WORD_1')
                        J.v_pk_mul_f32(v_w_f32[0], v_w_f32[0], v_w_scale[n])
                        J.v_pk_mul_f32(v_w_f32[1], v_w_f32[1], v_w_scale[n])
                        J.v_add_u32(v_w_f32[0, 0], v_w_f32[0, 0], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[0, 1], v_w_f32[0, 1], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[1, 0], v_w_f32[1, 0], s_cvt_bf16_bias)
                        J.v_add_u32(v_w_f32[1, 1], v_w_f32[1, 1], s_cvt_bf16_bias)
                        J.v_perm_b32(v_w_bf16[n, 0], v_w_f32[0, 0], v_w_f32[0, 1], pattern_cvt_bf16)
                        J.v_perm_b32(v_w_bf16[n, 1], v_w_f32[1, 0], v_w_f32[1, 1], pattern_cvt_bf16)
                    # 2, A_rep, 2, 2
                    for n in range(B_horz):
                        for m in range(A_vert):
                            yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], v_w_bf16[n], A_reg[pp_reg_id, m, i, j], C_reg[n, m])
        else:
            for m in range(A_vert):
                for n in range(B_horz):
                    yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], B_reg[pp_reg_id, n, 0], A_reg[pp_reg_id, m, 0, 0], C_reg[n, m])
                    yield J.v_mfma_f32_16x16x16_bf16(C_reg[n, m], B_reg[pp_reg_id, n, 1], A_reg[pp_reg_id, m, 0, 1], C_reg[n, m])

    def loop(pp_reg_id):
        loader = load_gen(pp_reg_id)
        mfma = mfma_gen(1 - pp_reg_id)

        J.emitter()([loader])

        J.emitter()([mfma])

    # prolog
    loader = load_gen(0)
    J.emitter()([loader])
    if weight_dtype != torch.bfloat16:
        for n in range(B_horz):
            J.v_mov_b32(v_w_scale[n, 1], v_w_scale[n, 0])
    pp_reg_id = 1

    def tail(pp_reg_id):
        J.s_waitcnt(mod=f"vmcnt(0)")
        mfma = mfma_gen(pp_reg_id)
        J.emitter()([mfma])

    if isinstance(K, int):
        if weight_dtype == torch.bfloat16:
            assert K % (num_split_k * 32) == 0, f'K must be multiple of {num_split_k * 32}'
        else:
            # a wave needs at least 64 elements
            assert K % 64 == 0, 'K must be multiple of 64'
    
        for k in range(0, k_max - 1):
            loop(pp_reg_id)
            pp_reg_id ^= 1

        # tail
        tail(1 - pp_reg_id)

    else:
        cur_k = J.gpr("si32")
        cur_k[0] = 0
        # at least 2 blocks, TODO: add check?
        k_loop_cnt = K // k_step_wg - 2

        # unroll 2 times for pin/pong buffer, align #loop to even
        with J.While(cur_k[0] < k_loop_cnt):
            loop(1)
            loop(0)
            cur_k[0] += 2
        J.Jump("odd_k_block", cur_k[0] == k_loop_cnt + 1)
        # there are 2 blocks left
        loop(1)
        # tail
        tail(1)
        J.Jump("k_block_end")
        # 1 block + tail
        J.Label("odd_k_block")
        # tail
        tail(0)
        J.Label("k_block_end")
