
import pyhip
import aiter
import torch
#from fly_moe_down import compile_gemm
import flydsl.compiler as flyc
import flydsl.expr as fx
from pyhip.contrib.flydsl.moe_gemm_splitk import compile_gemm

def xcd_swizzle(J, blk1d, num_blocks, num_oc_blocks, NUM_XCD, NUM_CU_PER_XCD):
    NUM_CU = NUM_XCD * NUM_CU_PER_XCD
    num_groupped_blocks = num_blocks // NUM_CU * NUM_CU
    blk_m = J.gpr("su32")
    blk_n = J.gpr("su32")
    if 0 and num_oc_blocks == 16:
        # in unit of 4x8 [256x256] blocks
        with J.If(blk1d < num_groupped_blocks) as If:
            blk_base = (blk1d // NUM_CU) * NUM_CU
            cu_id = blk1d % NUM_CU
            xcd_id = cu_id % 8  # 0~8
            xcd_cu = cu_id // 8 # 0~31
            coord_n = (xcd_id % 2)*8 + (xcd_cu % 8)
            coord_m = (xcd_id // 2)*4 + (xcd_cu // 8)
            task_id = coord_m * num_oc_blocks + coord_n
            new_blk1d = blk_base + task_id
            blk_m[0] = new_blk1d // num_oc_blocks
            blk_n[0] = new_blk1d - blk_m * num_oc_blocks            
            If.Else()
            blk_m[0] = blk1d // num_oc_blocks
            blk_n[0] = blk1d - blk_m * num_oc_blocks
    elif 0 and num_oc_blocks <= 4:
        with J.If(blk1d < num_groupped_blocks) as If:
            blk_base = (blk1d // NUM_CU) * NUM_CU
            cu_id = blk1d - blk1d // NUM_CU * NUM_CU
            xcd_id = cu_id % NUM_XCD
            xcd_cu = cu_id // NUM_XCD
            task_id = xcd_id * NUM_CU_PER_XCD + xcd_cu
            new_blk1d = blk_base + task_id
            blk_m[0] = new_blk1d // num_oc_blocks
            blk_n[0] = new_blk1d - blk_m * num_oc_blocks

            If.Else()
            blk_m[0] = blk1d // num_oc_blocks
            blk_n[0] = blk1d - blk_m * num_oc_blocks
    else:
        blk_m[0] = blk1d // num_oc_blocks
        blk_n[0] = blk1d - blk_m * num_oc_blocks
    return blk_m, blk_n

@pyhip.jit()
def moe_2stage_down(J:pyhip.JIT,
                    weight_dtype,
                    TOPK, K, N,          # 8,  128, 2048
                    with_silu,           #
                    BLOCK_TILE_SIZE_M,   # 32
                    BLOCK_TILE_SIZE_N,   # 64
                    quant_type_w,
                    p_id:"void*",
                    p_input:"void*",     # [8192, 8, 128]
                    p_weight:"void*",    # [128, 2048, 128]
                    p_output:"void*",    # [8192, 2048]
                    p_sorted_ids:"void*",        # [69624]
                    p_sorted_weights:"float*",   # [69624]
                    p_sorted_expert_ids:"void*", # [2176]
                    p_num_valid_ids:"void*",     # [2]  value: [65536,  8192]
                    pt_scale: "float*",
                    pc_scale:"float*",
                    M:"int",
                    num_blocks:"int",
                    dyn,):
    if dyn:
        SUB_M = 64
        assert BLOCK_TILE_SIZE_M % SUB_M == 0
    else:
        SUB_M = BLOCK_TILE_SIZE_M
    sizeof_w = J.sizeof(weight_dtype)
    dtype_A = "bf16"
    is_fp8 = str(weight_dtype).startswith("torch.float8_e4m3")
    
    if is_fp8:
        fp8_ptpc = {"quant_type_w":quant_type_w}
        dtype_A = "fp8"
    else:
        fp8_ptpc = None

    max_id = J.gpr(1, 'su32')
    J.s_load_dword(max_id, p_num_valid_ids, 0)

    mfma_MN = 16
    mfma_K = (64//mfma_MN) * (J.sizeof_DW4//J.sizeof(dtype_A))
    num_mfma_m = J.div(SUB_M, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)

    A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")

    # collect token id
    row = J.lane_id % mfma_MN
    col = J.lane_id // mfma_MN
    M_TOPK = J.gpr(M[0] * TOPK)

    J.get_sgpr_const(0x8000)
    J.get_sgpr_const(0x3020706)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    def loop_body(idx, run_in_m_sub_tiles):
        e_idx = idx
        sub_m = 0
        #e_idx, sub_m = xcd_swizzle(J, idx, num_blocks, BLOCK_TILE_SIZE_M // SUB_M, 4, 20)
        #if not run_in_m_sub_tiles:
        #    sub_m = 0
        # invalid padding section
        J.Jump("continue_following", e_idx * BLOCK_TILE_SIZE_M < max_id)
        J.s_endpgm()
        J.Label("continue_following")
        # sub_m = J.blockIdx.x
        s_e_id = J.gpr(1, 'su32')
        J.s_load_dword(s_e_id, p_sorted_expert_ids, e_idx[0] * 4)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

        p_cur_sorted_ids = J.gpr(2, 'su32')
        p_cur_sorted_ids[:] = p_sorted_ids[:] + (e_idx * (BLOCK_TILE_SIZE_M * 4) + sub_m * (SUB_M * 4))
        p_cur_sorted_weights = J.gpr(2, 'su32')
        p_cur_sorted_weights[:] = p_sorted_weights[:] + (e_idx * (BLOCK_TILE_SIZE_M * 4) + sub_m * (SUB_M * 4))
        p_cur_weight = J.gpr(2, 'su32')
        p_cur_weight[:] = p_weight[:] + s_e_id * (N * K * sizeof_w)
        cur_pc_scale = J.gpr(2, 'su32')
        if quant_type_w == "QuantType.per_Tensor":
            cur_pc_scale[:] = pc_scale[:] + s_e_id * (1 * J.sizeof_DW) # per-tensor scales for weights
        else:
            cur_pc_scale[:] = pc_scale[:] + s_e_id * (N * J.sizeof_DW) # per-channel scales for weights

        v_sorted_id = J.gpr(num_mfma_m, 'vu32')
        for m in range(num_mfma_m):
            J.global_load_dword(v_sorted_id[m], row*J.sizeof_DW + m*mfma_MN*J.sizeof_DW, p_cur_sorted_ids)

        J.s_waitcnt(mod=f"vmcnt(0)")

        vaddr = J.gpr(num_mfma_m, "vu32")
        for m in range(num_mfma_m):
            vaddr[m] = (v_sorted_id[m] & 0xFFFFFF) * (TOPK * K * sizeof_w) + (v_sorted_id[m]>>24) *(K * sizeof_w) + col * J.sizeof_DW4

        if is_fp8:
            buff_sa = J.Buffer(pt_scale, M * TOPK * J.sizeof("fp32"))
            v_pt_scales = J.gpr(num_mfma_m, 'vf32')
            for m in range(num_mfma_m):
                voffset_sa = J.gpr("vu32", (v_sorted_id[m] & 0xFFFFFF) * (TOPK * J.sizeof_DW) + (v_sorted_id[m]>>24) * J.sizeof_DW)
                buff_sa.load_dword(v_pt_scales[m], voffset_sa, 0)
            fp8_ptpc["v_pt_scales"] = v_pt_scales

            if quant_type_w == "QuantType.per_Token":
                fp8_ptpc["pc_scale"] = cur_pc_scale

            if quant_type_w == "QuantType.per_Tensor":
                v_pc_scales = J.gpr('vf32')
                v_offset_zero = J.gpr('vu32', 0)
                J.global_load_dword(v_pc_scales[0], v_offset_zero, cur_pc_scale)
                fp8_ptpc["v_pc_scales"] = v_pc_scales

        buff_a = J.Buffer(p_input, M * TOPK * K * J.sizeof(dtype_A))
        for m in range(num_mfma_m):
            for k in range(num_mfma_k):
                buff_a.load_dwordx4(A[m,k], vaddr[m], 0, offset12=k*mfma_K*J.sizeof(dtype_A))

        v_sorted_weights = J.gpr('vf32')
        assert SUB_M <= 256
        with J.ExecMask(J.threadIdx.x < SUB_M):
            J.global_load_dword(v_sorted_weights, J.threadIdx.x * 4, p_cur_sorted_weights)

        J.s_waitcnt(mod=f"vmcnt(0)")

        lds_weights = J.alloc_lds(256*4)
        lds_token_ids = J.alloc_lds(256*4)

        with J.ExecMask(J.threadIdx.x < SUB_M):
            J.ds_write_b32(J.threadIdx.x * 4, v_sorted_weights, mod=f"offset:{lds_weights}")
        with J.ExecMask(J.threadIdx.x < 16):
            for m in range(num_mfma_m):
                J.ds_write_b32(J.lane_id * 4, (v_sorted_id[m] & 0xffffff) * TOPK + (v_sorted_id[m]>>24), mod=f"offset:{lds_token_ids + m*16*4}")

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        if run_in_m_sub_tiles:
            s_sorted_ids = J.gpr(1, 'su32')
            J.v_readfirstlane_b32(s_sorted_ids[0], v_sorted_id[0])
            with J.If((s_sorted_ids[0] >> 24) != TOPK):
                num_mfma_n = 1 if SUB_M > 64 else 2
                down_kernel(J, mfma_MN, num_mfma_n, SUB_M, N, K,
                            A, lds_token_ids, lds_weights,
                            p_cur_weight, p_output, M_TOPK, fp8_ptpc)
        else:
            # 0,1,2,3,4,5,6,7, num_mfma_m
            """
            VALID_TILE_SIZES = [s for s in [64,96,128] if s < BLOCK_TILE_SIZE_M]
            s_sorted_ids = J.gpr(len(VALID_TILE_SIZES), 'su32')
            for i, TILE_SIZE in enumerate(VALID_TILE_SIZES):
                J.v_readfirstlane_b32(s_sorted_ids[i], v_sorted_id[TILE_SIZE//mfma_MN])

            for i, TILE_SIZE in enumerate(VALID_TILE_SIZES):
                with J.If((s_sorted_ids[i] >> 24) == TOPK):
                    num_mfma_n = 1 if TILE_SIZE > 64 else 2
                    down_kernel(J, mfma_MN, num_mfma_n, TILE_SIZE, N, K,
                                A, lds_token_ids, lds_weights,
                                p_cur_weight, p_output, M_TOPK, fp8_ptpc)
                    J.s_endpgm()
            """
            num_mfma_n = 1 if BLOCK_TILE_SIZE_M > 64 else 2
            num_mfma_n = 1
            down_kernel(J, mfma_MN, num_mfma_n, BLOCK_TILE_SIZE_M, N, K,
                        A, lds_token_ids, lds_weights,
                        p_cur_weight, p_output, M_TOPK, fp8_ptpc)
    if dyn:
        with J.While():
            idx = J.gpr(1, 'su32')
            idx[0] = 0xffffffff
            v_idx = J.gpr(1, 'vu32')
            J.s_barrier()
            with J.If(J.warp_id[0] == 0):
                J.s_atomic_inc(idx, p_id, 0, mod='glc')
                J.s_waitcnt(mod=f"lgkmcnt(0)")
                v_idx[0] = idx[0]
                J.ds_write_b32(0 * row, v_idx, mod=f"offset:{0}")
                J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()
            J.ds_read_b32(v_idx, 0 * row, mod=f"offset:{0}")
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_readfirstlane_b32(idx, v_idx)
            J.s_barrier()

            loop_body(idx, True)
    else:
        loop_body(J.blockIdx.x, False)

def down_kernel(J, mfma_MN, num_mfma_n, BM, N, K,
                A, # A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
                lds_token_ids,  # 256 int32 token_id
                lds_weights,    # 256 fp32 weights 
                pB:"void*",
                pC:"void*",
                M,
                fp8_ptpc):
    num_warps = 4
    sizeof_w = J.sizeof_bf16 if fp8_ptpc is None else J.sizeof("fp8")
    # given DW4 lane size, how many bf16 items along K direction
    mfma_K = (64//mfma_MN) * (J.sizeof_DW4//sizeof_w) # each DW4 vgpr holds a mfma_K=32(bf16) or 64(fp8)

    # load A [BM x K] bf16 into AccGPRs 
    num_mfma_m = J.div(BM, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)  # K=96, num_mfma_k=3

    # 4 warps work in parallel along N dimension
    buff_b = J.Buffer(pB, N * K * sizeof_w)
    # ping-pong buffer
    B = J.gpr(2, num_mfma_n, num_mfma_k, 4, "vbf16x2") # 4 x (8,bf16) or (16,fp8)
    C = J.gpr(2, num_mfma_m, num_mfma_n, 4, "vf32")

    # prelog0, load Bn0
    # prelog1, load Bn1, compute Cn0
    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    voff_b = J.gpr(J.lane_id * J.sizeof_DW4 + J.gpr(J.warp_id * (mfma_MN * K * sizeof_w)))
    soff_b = J.gpr("su32")
    soff_b[0] = 0

    if fp8_ptpc is not None:
        if fp8_ptpc['quant_type_w'] == "QuantType.per_Token":
            vaddr_pc_scale = J.gpr("vu32", (J.lane_id // 16) * J.sizeof_DW4 + J.warp_id * (mfma_MN * J.sizeof_DW)) # point to the start of pc scales for this WG
            scales_pc = J.gpr(2, num_mfma_n, 4, "vf32")
            buff_sb = J.Buffer(fp8_ptpc["pc_scale"], N * J.sizeof_DW)

    def loadB_generator(index):
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                yield 1
                buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * sizeof_w
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * sizeof_w)
        if fp8_ptpc is not None and fp8_ptpc['quant_type_w'] == "QuantType.per_Token":
            # load scales_pc for next block, each MFMA 16x16 block-B needs 16 scales
            # fp8_ptpc["v_pt_scales"] = v_pt_scales
            # fp8_ptpc["pc_scale"] = pc_scale
            for n in range(num_mfma_n):
                yield 1
                buff_sb.load_dwordx4(scales_pc[index, n], vaddr_pc_scale, 0, offset12=n*num_warps*mfma_MN*J.sizeof_DW)
            vaddr_pc_scale[0] += num_warps * num_mfma_n * mfma_MN * J.sizeof_DW

    print(num_mfma_m, num_mfma_n, num_mfma_k)
    def mfma_generator(index):
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    Ci = 0 if k == 0 else C[index,m,n]
                    yield 16
                    if fp8_ptpc is not None:
                        J.v_mfma_f32_16x16x32_fp8_fp8(C[index,m,n], B[index,n,k,0:1], A[m,k,0:1], Ci)
                    else:
                        J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,0:1], A[m,k,0:1], Ci)
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    yield 16
                    if fp8_ptpc is not None:
                        J.v_mfma_f32_16x16x32_fp8_fp8(C[index,m,n], B[index,n,k,2:3], A[m,k,2:3], C[index,m,n])
                    else:
                        J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,2:3], A[m,k,2:3], C[index,m,n])
        if fp8_ptpc is not None:
            # dequantize C here since all computations over K have finished
            # scales_pt are resident, scales_pc are 
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    yield 16
                    C[index,m,n,0] *= fp8_ptpc["v_pt_scales"][m]
                    C[index,m,n,1] *= fp8_ptpc["v_pt_scales"][m]
                    C[index,m,n,2] *= fp8_ptpc["v_pt_scales"][m]
                    C[index,m,n,3] *= fp8_ptpc["v_pt_scales"][m]
                    yield 16
                    if fp8_ptpc['quant_type_w'] == "QuantType.per_Token":
                        C[index,m,n,0] *= scales_pc[index, n, 0]
                        C[index,m,n,1] *= scales_pc[index, n, 1]
                        C[index,m,n,2] *= scales_pc[index, n, 2]
                        C[index,m,n,3] *= scales_pc[index, n, 3]
                    if fp8_ptpc['quant_type_w'] == "QuantType.per_Tensor":
                        C[index,m,n,0] *= fp8_ptpc["v_pc_scales"][0]
                        C[index,m,n,1] *= fp8_ptpc["v_pc_scales"][0]
                        C[index,m,n,2] *= fp8_ptpc["v_pc_scales"][0]
                        C[index,m,n,3] *= fp8_ptpc["v_pc_scales"][0]

    # prelog0, load Bn0
    J.emit(loadB_generator(0))

    # prelog1, load Bn1, compute Cn0
    J.emit(loadB_generator(1))
    J.s_waitcnt(mod=f"vmcnt({num_mfma_n*num_mfma_k})")
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    J.emit(mfma_generator(0))

    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    s_cvt_bf16_bias = J.get_sgpr_const(0x00008000)

    vmem_lane_size = J.sizeof_DW4

    lds_padding = (4 if vmem_lane_size == J.sizeof_DW else 8) * J.sizeof_bf16 # to avoid bank-conflict
    lds_width = num_mfma_n * 4 * mfma_MN * J.sizeof_bf16
    lds_stride = lds_width + lds_padding

    # WG level write C into LDS
    row = J.threadIdx.x % mfma_MN
    col = J.threadIdx.x // mfma_MN
    voff_c_lds_w = J.gpr(row * lds_stride + col * (4 * J.sizeof_bf16))

    # WG level load C from LDS
    num_lanes_ldsr = J.div(lds_width, vmem_lane_size)
    assert num_lanes_ldsr <= 64, num_lanes_ldsr
    col = J.threadIdx.x % num_lanes_ldsr
    row = J.threadIdx.x // num_lanes_ldsr
    num_rows_per_load = J.div(256, num_lanes_ldsr)
    num_loads = J.div(num_mfma_m * mfma_MN, num_rows_per_load)
    voff_c_lds_r = J.gpr(row * lds_stride + col * (vmem_lane_size))
    vmem_stride = N * J.sizeof_bf16
    v_weights = J.gpr(num_mfma_m, 2, "vf32") # pkmul
    voff_vmem = J.gpr(num_loads, 2, "vu32")
    for m in range(num_mfma_m):
        J.ds_read_b32(v_weights[m,0], (m*mfma_MN + (J.lane_id % mfma_MN))*4, mod=f"offset:{lds_weights}")

    voff_vmem_row = J.gpr(num_loads, "vu32")
    for i in range(num_loads):
        J.ds_read_b32(voff_vmem_row[i], row * 4, mod=f"offset:{lds_token_ids + i * num_rows_per_load * 4}")

    J.s_waitcnt(mod="lgkmcnt(0)")

    lds = J.alloc_lds((num_mfma_m * mfma_MN) * (lds_stride))

    for m in range(num_mfma_m):
        v_weights[m,1] = v_weights[m,0]

    saddr_dummy = J.gpr(2, "su32", 0)
    voff_vmem_base = J.gpr(2, "vu32", pC[0], pC[1])
    J.v_lshl_add_u64(voff_vmem_base, J.gpr(2, "vu32", col * (vmem_lane_size), 0), 0, voff_vmem_base)
    for i in range(num_loads):
        #voff_vmem[i] = voff_vmem_row[i] * vmem_stride + col * (vmem_lane_size)
        J.v_mad_u64_u32(voff_vmem[i], saddr_dummy, voff_vmem_row[i], J.gpr("su32", vmem_stride), voff_vmem_base)

    temp_c = J.gpr(num_loads, vmem_lane_size//J.sizeof_DW, "vbf16x2")

    voff_vmem_step = J.gpr(2, "vu32", 4 * num_mfma_n * mfma_MN * J.sizeof_bf16, 0)

    def loop_body(ni):
        J.s_waitcnt(mod=f"vmcnt({num_loads})")

        mfma1 = mfma_generator((ni+1)&1)
        B_loader = loadB_generator(ni&1)

        #cvt_f32_to_pk_bf16(n&1)
        index = ni&1
        for m in range(num_mfma_m):
            J.emit(B_loader, 1)
            for n in range(num_mfma_n):
                J.v_pk_mul_f32(C[index,m,n,0:1], C[index,m,n,0:1], v_weights[m])
                J.v_pk_mul_f32(C[index,m,n,2:3], C[index,m,n,2:3], v_weights[m])
                J.v_add_u32(C[index,m,n,0], C[index,m,n,0], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,1], C[index,m,n,1], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,2], C[index,m,n,2], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,3], C[index,m,n,3], s_cvt_bf16_bias)
                J.pk_f32_to_bf16(C[index,m,n,0], C[index,m,n,0], C[index,m,n,1])
                J.pk_f32_to_bf16(C[index,m,n,1], C[index,m,n,2], C[index,m,n,3])
            #emit_mfma([mfma1], 16)

        # ds_write_C(ni&1)
        index = ni & 1
        for m in range(num_mfma_m):
            J.emit(B_loader, 1)
            for n in range(num_mfma_n):
                offset = lds + m*mfma_MN*lds_stride + n*(4*mfma_MN*J.sizeof_bf16)
                J.ds_write_b64(voff_c_lds_w, C[index, m, n, 0:1], mod=f"offset:{offset}")
                J.emit(mfma1, 16)
        J.emit(mfma1, 32)
        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()

        #ds_load_C()
        for i in range(num_loads):
            offset = lds + i * num_rows_per_load * lds_stride
            J.ds_read_b128(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
            J.emit(mfma1, 32)

        J.emit(B_loader)

        #atomic_pk_add_bf16()
        for i in range(num_loads):
            J.s_waitcnt(mod=f"lgkmcnt({min(15,num_loads - i - 1)})")
            with J.ExecMask(voff_vmem_row[i] < M[0], early_skip=False):
                J.global_store_dwordx4(voff_vmem[i], temp_c[i], "off", mod="nt sc1")      # this is fast:  (48us)
            J.emit(mfma1, 128)
            J.v_lshl_add_u64(voff_vmem[i], voff_vmem_step, 0, voff_vmem[i])

        J.emit(mfma1)

    loop_i = J.gpr("su32")
    loop_i[0] = 0
    loop_cnt = J.div(N, 4 * num_mfma_n * mfma_MN)

    J.s_waitcnt(mod=f"vmcnt(0)")
    with J.While(loop_i[0] < (loop_cnt//2)):
        loop_body(0)
        loop_body(1)
        loop_i[0] = loop_i[0] + 1
    if loop_cnt % 2:
        loop_body(0)

    J.free_lds(lds)

def test_down(pt_file):

    _, stream = pyhip.set_device(1)
    moe_down_data = torch.load(pt_file, torch.get_default_device())
    down_in = moe_down_data["down_in"]
    w2 = moe_down_data["w2"]
    gemm2_out = moe_down_data["gemm2_out"]
    sorted_ids = moe_down_data["sorted_ids"]
    sorted_weights = moe_down_data["sorted_weights"]
    sorted_expert_ids = moe_down_data["sorted_expert_ids"]
    num_valid_ids = moe_down_data["num_valid_ids"]
    w2_scale_arg = moe_down_data["w2_scale_arg"]
    a_scale = moe_down_data["a_scale"]
    B = moe_down_data["B"]
    grid = moe_down_data["grid"]
    N = moe_down_data["N"]
    K = moe_down_data["K"]
    weight_dtype = moe_down_data["weight_dtype"]
    weight_quant_type = moe_down_data["weight_quant_type"]

    print(w2.shape, w2.dtype)
    print(w2_scale_arg.shape, w2_scale_arg.dtype)

    if 0:
        w2_scale_arg = w2_scale_arg.view(64, 1, 1).expand(64, 4096, 1).clone()
        weight_quant_type = "ptpc"

    if "act_quant_type" in moe_down_data:
        act_quant_type = moe_down_data["act_quant_type"]
    else:
        act_quant_type = None

    TOPK = moe_down_data["TOPK"]
    BLOCK_TILE_SIZE_M = moe_down_data["BLOCK_TILE_SIZE_M"]
    BLOCK_TILE_SIZE_N = moe_down_data["BLOCK_TILE_SIZE_N"]
    stage = moe_down_data["stage"]
    alg = moe_down_data["alg"]
    E = moe_down_data["E"]
    USE_ATOMIC_WRITE = moe_down_data["USE_ATOMIC_WRITE"]

    print("==========", weight_quant_type, act_quant_type)

    moe = compile_gemm(
        N,
        K,
        weight_dtype,
        weight_quant_type,
        TOPK,
        BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N,
        stage=stage,
        alg="prefill_1x4",
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=None,
    )

    _TORCH_TO_FX = {
        torch.bfloat16: fx.BFloat16,
        torch.float32: fx.Float32,
        torch.int32: fx.Int32,
        torch.float8_e4m3fnuz: fx.Uint8,
        torch.float8_e4m3fn: fx.Uint8,
    }
    def _ptr(t):
        return flyc.from_c_void_p(_TORCH_TO_FX[t.dtype], t.data_ptr())

    if act_quant_type == "ptpc":
        down_in, a_scale = aiter.get_hip_quant(aiter.QuantType.per_Token)(
            down_in.view(B * TOPK, -1), quant_dtype=w2.dtype
        )
        a_scale = a_scale.to(torch.float32).contiguous()
    elif act_quant_type == "per_tensor":
        fmax = torch.finfo(w2.dtype).max
        a_scale = down_in.float().abs().amax() / fmax
        down_in = (down_in.float() / a_scale).clamp(-fmax, fmax).to(w2.dtype)
        a_scale = a_scale.reshape(1).to(torch.float32)
    else:
        # no quant
        pass
    down_out = torch.empty([B, TOPK, N], dtype=gemm2_out.dtype)

    args = (
        _ptr(down_in),
        _ptr(w2),
        _ptr(down_out),
        _ptr(sorted_ids),
        _ptr(sorted_weights),
        _ptr(sorted_expert_ids),
        _ptr(num_valid_ids),
        _ptr(w2_scale_arg),
        _ptr(a_scale),
        B,
        grid,
        stream,
    )
    compiled = flyc.compile(moe, *args)
    num_flops = sorted_expert_ids.numel() * (BLOCK_TILE_SIZE_M * N * K * 2)
    num_bytes = sorted_expert_ids.numel() * (
        N * K * w2.element_size()
        + BLOCK_TILE_SIZE_M * K * down_in.element_size()
        + BLOCK_TILE_SIZE_M * N * gemm2_out.element_size()
    )
    pyhip.run_perftest(
        compiled,
        *args,
        num_name=f"fly-moe-down-{B}-{weight_dtype}-{weight_quant_type}-{act_quant_type}",
        num_flops=num_flops,
        num_bytes=num_bytes,
        num_verbose=1,
    )
    cur_out = torch.sum(down_out, dim=1)
    diff = pyhip.calc_diff(cur_out, gemm2_out)
    print(diff)

    STAGE2_TILE_N = 64
    
    fp8_quant_type = aiter.QuantType.No
    fp8_quant_type = aiter.QuantType.per_Token
    if weight_quant_type == "per_tensor":
        fp8_quant_type = aiter.QuantType.per_Tensor

    id_buf2 = torch.zeros(64, dtype=torch.int32)
    grid_down = sorted_expert_ids.shape[0]
    assert grid == grid_down
    dyn_schedule = False
    down_out[...] = 0
    moe_2stage_down([grid], [256],
                w2.dtype, TOPK, K, N, False, BLOCK_TILE_SIZE_M, STAGE2_TILE_N, str(fp8_quant_type),
                id_buf2, down_in, w2, 
                down_out, #cur_out,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                a_scale,
                w2_scale_arg,
                B,
                sorted_expert_ids.shape[0],
                dyn_schedule)
    cur_out = torch.sum(down_out, dim=1)

    diff = pyhip.calc_diff(cur_out, gemm2_out)
    print(diff)

    pyhip.run_perftest(
        moe_2stage_down,
        [grid], [256],
        w2.dtype, TOPK, K, N, False, BLOCK_TILE_SIZE_M, STAGE2_TILE_N, str(fp8_quant_type),
        id_buf2, down_in, w2, 
        down_out, #cur_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a_scale,
        w2_scale_arg,
        B,
        sorted_expert_ids.shape[0],
        dyn_schedule,
        num_name=f"jit-moe-down-{B}-{weight_dtype}-{fp8_quant_type}-{fp8_quant_type}",
        num_flops=num_flops,
        num_bytes=num_bytes,
        num_verbose=1,
    )
test_down("moe_down_data_fp8_ptpc_ptpc.pt")
#test_down("moe_down_data_bf16_no_no_16k.pt")
#test_down("moe_down_data_bf16_no_no_64k.pt")
#test_down("moe_down_data_bf16_no_no_16k.pt")
#test_down("moe_down_data_bf16_no_no_64k.pt")
#test_down("moe_down_data_fp8_ptpc_ptpc_16k.pt")
#test_down("moe_down_data_fp8_per_tensor_ptpc.pt")