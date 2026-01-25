import pyhip
import pytest
import functools
import torch


torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)


def pre_shuffle(x, mfma_MN):
    M, K = x.shape
    K_bytes = K * x.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//x.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0

    x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    x = x.permute(0,2,3,1,4)
    return x.contiguous()

wg_M = 256
wg_N = 256

def get_loader(J, buff, use_pre_shuffle, nbM, nbK, stride_b, ibM0):
    num_warps = 4
    #stride_b = 0
    if use_pre_shuffle:
        stride_1kb = J.div(16*stride_b, 1024)
        warp_k = J.warp_id[0] % nbK
        warp_m = J.warp_id[0] // nbK
        vmem_warp_off = warp_m * (stride_1kb * 1024) + warp_k * 1024
        vmem_voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + vmem_warp_off)
        lds_warp_off = J.gpr("su32", warp_m * (nbK * 1024) + warp_k * 1024)

        voff = J.gpr(J.lane_id[0] * J.sizeof_DW4 + ibM0 * (nbK * 1024))
        voff2 = J.gpr("vu32", voff[0] + 64*1024)
        def ds_read_1kb(lds, vdst, m, k):
            offset = lds + m*(nbK * 1024) + k*1024
            if offset >= 64*1024:
                voffset = voff2
                offset -= 64*1024
            else:
                voffset = voff
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

        vm_load_cnt = len(range(J.div(nbM, num_warps//nbK)))
        # return a loader constructor which can emit
        def vm_load(lds_offset):
            J.s_mov_b32("m0", lds_warp_off + lds_offset)
            voff = J.gpr("vu32", vmem_voff[0])

            for m in range(J.div(nbM, num_warps//nbK)):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                J.s_addk_i32("m0", 256*J.sizeof_DW4)
                voff[0] += (num_warps//nbK)*(stride_1kb)*1024

            vmem_voff[0] += nbK * 1024
    else:
        if 0:
            # 通过下面的可视化得知swizzle可以解决读入数据使用mfma格式ds_read时潜在的bank-conflict问题
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_1=1)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=1)
            J.show_mfma_in_lds(mfma_MN=16, num_mfmas=2, swizzle_2=2)
            print(f"{wg_M=} {wg_K=} {nbM=} {nbK=}")
            assert 0
        # each wave load 8x128 bytes , 4 waves loads 32x128 bytes
        lds_stride_b = nbK * 4 * J.sizeof_DW4
        warp_m_off = J.warp_id[0] * 8

        def swizzle(row, col):
            #return col
            return (col ^ row) % 8
        col = J.threadIdx.x % 8
        row = J.threadIdx.x // 8
        swizzle_col = swizzle(row, col)
        vmem_voff = J.gpr(row * stride_b + swizzle_col * J.sizeof_DW4)
        lds_warp_off = J.gpr("su32", warp_m_off * lds_stride_b)

        vm_load_cnt = len(range(0, nbM * 16, 8*num_warps))

        def vm_load(lds_offset):
            J.s_mov_b32("m0", lds_warp_off + lds_offset)
            voff = J.gpr("vu32", vmem_voff[0])
            for m in range(0, nbM * 16, 8*num_warps):
                yield 1
                buff.load_dwordx4(None, voff, 0, offset12=0)
                J.s_addk_i32("m0", 256*J.sizeof_DW4)
                voff[0] += (8*num_warps) * stride_b
            vmem_voff[0] += nbK * 4 * J.sizeof_DW4

        col = J.lane_id // 16
        row = J.lane_id % 16
        swizzle_col = swizzle(row, col)
        voff = J.gpr(2, "vu32",
                     (row + ibM0*16) * lds_stride_b + swizzle(row, col) * J.sizeof_DW4,
                     (row + ibM0*16) * lds_stride_b + swizzle(row, col + 4) * J.sizeof_DW4)
        # ds_read_b128's offset is just 16bits 
        voff2 = J.gpr(2, "vu32", voff[0] + 64*1024, voff[1] + 64*1024)
        def ds_read_1kb(lds, vdst, m, k):
            offset = lds + m*16*lds_stride_b
            if offset >= 64*1024:
                voffset = voff2[k]
                offset -= 64*1024
            else:
                voffset = voff[k]
            J.ds_read_b128(vdst, voffset, mod=f"offset:{offset}")

    
    return vm_load, vm_load_cnt, ds_read_1kb


@pyhip.jit()
def gemm_a4w4_kernel(J, wg_M, wg_N, N, K, use_pre_shuffleA, use_pre_shuffleB,
                     pA:"void*", pAscale:"void*",
                     pB:"void*", pBscale:"void*",
                     pC:"void*", M:"int"):
    # K in unit of fp4x2 128 bytes
    wg_K = J.div(128, J.sizeof_fp4x2)

    A_dtype = "fp4x2"
    B_dtype = "fp4x2"
    C_dtype = "bf16"
    M01 = 8
    GroupNum = 8

    stride_k = K * J.sizeof_fp4x2

    blk_m, blk_n = J.tb_swizzle(J.blockIdx.x, M, wg_M, wg_N, N, M01, GroupNum)
    pA[:] += blk_m * (wg_M * stride_k)
    pB[:] += blk_n * (wg_N * stride_k)
    pC[:] += (blk_m * (wg_M * N * J.sizeof(C_dtype)) + blk_n * (wg_N * J.sizeof(C_dtype)))

    stride_scale = J.div(K, 128) * J.sizeof_DW * 64
    pAscale[:] += blk_m * J.div(wg_M, 32) * stride_scale
    pBscale[:] += blk_n * J.div(wg_N, 32) * stride_scale

    pAscale[:] += (J.warp_id[0] // 2) * (J.div(wg_M//2, 32) * stride_scale)
    pBscale[:] += (J.warp_id[0] % 2) * (J.div(wg_N//2, 32) * stride_scale)

    sbuff_a = J.Buffer(pAscale, J.div(wg_M//2, 32) * stride_scale)
    sbuff_b = J.Buffer(pBscale, J.div(wg_N//2, 32) * stride_scale)

    assert N % wg_N == 0
    num_warps = 4
    nbN = J.div(wg_N, 16)
    nbM = J.div(wg_M, 16)
    nbK = 2 # 2 MFMA 16x16 
    buff_a = J.Buffer(pA, wg_M * stride_k)
    buff_b = J.Buffer(pB, wg_N * stride_k)
    ldsA = [J.alloc_lds(nbM * nbK * 1024), J.alloc_lds(nbM * nbK * 1024)]
    ldsB = [J.alloc_lds(nbN * nbK * 1024), J.alloc_lds(nbN * nbK * 1024)]

    nrM = J.div(nbM, 2)
    nrN = J.div(nbN, 2)
    nrK = nbK

    warp_m = J.gpr((J.warp_id[0] // 2)*nrM)
    warp_n = J.gpr((J.warp_id[0] % 2)*nrN)

    vm_load_a, vm_load_cnt_a, ds_read_a = get_loader(J, buff_a, use_pre_shuffleA, nbM, nbK, stride_k, warp_m)
    vm_load_b, vm_load_cnt_b, ds_read_b = get_loader(J, buff_b, use_pre_shuffleB, nbN, nbK, stride_k, warp_n)

    print(f"============={nbM=}, {nbN=}, {nbK=} {nrM=} {nrN=} {nrK=}")

    mfma_A = J.gpr(2, nrM, 4, "vbf16x2")
    mfma_B = J.gpr(2, nrN, 4, "vbf16x2")
    mfma_C = J.gpr(nrM, nrN, 4, "af32")
    
    mfma_Ascale = J.gpr(2, J.div(nrM, 2), "vu32") # 4
    mfma_Bscale = J.gpr(2, J.div(nrN, 2), "vu32") # 4

    def mfma(reg_id, lds_id):
        # lds_id : scale register is grouped by lds_id
        # src0: Matrix A scale {OP_SEL_HI [0], OP_SEL[0]} defines which part of scale is used by the Matrix A of MFMA instruction.
        # src1: Matrix B scale {OP_SEL_HI [1], OP_SEL[1]} defines which part of scale is used by the Matrix B of MFMA instruction.
        for m in range(nrM):
            for n in range(nrN):
                sel_scale_B = (n & 1) + (reg_id & 1)*2
                sel_scale_A = (m & 1) + (reg_id & 1)*2
                mod = f"op_sel:[{sel_scale_B & 1}, {sel_scale_A & 1},0] op_sel_hi:[{sel_scale_B//2}, {sel_scale_A//2}, 0] cbsz:4 blgp:4"

                # J.v_mfma_f32_16x16x32_bf16(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n])
                J.v_mfma_scale_f32_16x16x128_f8f6f4(mfma_C[m,n], mfma_B[reg_id, n], mfma_A[reg_id, m], mfma_C[m,n],
                                                    mfma_Bscale[lds_id, n//2],
                                                    mfma_Ascale[lds_id, m//2],
                                                    mod=mod)
                yield 16

    J.emit(vm_load_a(ldsA[0]))
    J.emit(vm_load_b(ldsB[0]))

    # load scales
    vaddr_scale = J.gpr("vu32", J.lane_id[0] * J.sizeof_DW)

    def load_next_scales(index):
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrM, 2)):
            sbuff_a.load_dword(mfma_Ascale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale
            yield 1
        vaddr = J.gpr("vu32", vaddr_scale[0])
        for ii in range(J.div(nrN, 2)):
            sbuff_b.load_dword(mfma_Bscale[index & 1, ii], vaddr, 0)
            vaddr[0] += stride_scale
            yield 1
        vaddr_scale[0] += J.sizeof_DW * 64

    J.emit(load_next_scales(0))

    J.emit(vm_load_a(ldsA[1]))
    J.emit(vm_load_b(ldsB[1]))
    mfma_C[...] = 0

    '''
    ab0: mfma ab0 | ds_read ab1; wait_lgkmcnt(0), barrier; vm-load a01 | load mfma_Ascale[1]
    ab1: mfma ab1 | vm-load b01; wait_vmcnt, barrier, ds_read ab2      | load mfma_Bscale[1]

    ab2: mfma ab2 | ds_read ab3; wait_lgkmcnt(0), barrier; vm-load  a23 | load mfma_Ascale[0]
    ab3: mfma ab3 | vm-load b23;  wait_vmcnt, barrier, ds_read ab0      | load mfma_Bscale[0]
    '''
    J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a})")
    J.s_barrier()

    # ds_read ab0
    for m in range(nrM): ds_read_a(ldsA[0], mfma_A[0, m], m, 0)
    for n in range(nrN): ds_read_b(ldsB[0], mfma_B[0, n], n, 0)
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    def loop_body(lds_id):
        # mfma ab0
        mfma_ab0 = mfma(0, lds_id)

        load_s = load_next_scales(lds_id + 1)
        # ds_read ab1
        for m in range(nrM):
            J.emit(mfma_ab0, 16)
            ds_read_a(ldsA[lds_id], mfma_A[1, m], m, 1)
            J.emit(load_s, 1)
        for n in range(nrN):
            J.emit(mfma_ab0, 16)
            ds_read_b(ldsB[lds_id], mfma_B[1, n], n, 1)
            J.emit(load_s, 1)

        for ii in range(4):
            J.emit(mfma_ab0, 16)
            J.emit(load_s, 1)

        J.emit(load_s)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.emit(mfma_ab0, 16)
        J.s_barrier()

        mfma_ab1 = mfma(1, lds_id)

        # vm-load a01
        vm_load = vm_load_a(ldsA[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_a):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)
        J.emit(vm_load)

        # vm-load b01
        vm_load = vm_load_b(ldsB[lds_id])
        J.emit([mfma_ab0, mfma_ab1], 16)
        J.emit(vm_load, 1) # first emit produce preparing instructions
        J.emit([mfma_ab0, mfma_ab1], 16)
        for _ in range(vm_load_cnt_b - 4):
            J.emit(vm_load, 1)
            J.emit([mfma_ab0, mfma_ab1], 96)

        J.emit(mfma_ab0) # emit all MFMA using AB register0 (since ds_read will override it)

        # wait vm-load a23/b23 to finish
        J.s_waitcnt(mod=f"vmcnt({vm_load_cnt_b + vm_load_cnt_a - 4})")
        J.s_barrier()

        J.emit(vm_load, 1)
        # ds_read ab2
        for m in range(nrM):
            J.emit(mfma_ab1, 16)
            ds_read_a(ldsA[(lds_id + 1)&1], mfma_A[0, m], m, 0)

        J.emit(vm_load, 1)
        for n in range(nrN):
            J.emit(mfma_ab1, 16)
            ds_read_b(ldsB[(lds_id + 1)&1], mfma_B[0, n], n, 0)

        J.emit(vm_load, 1)
        J.emit(mfma_ab1, 96)
        J.emit(vm_load, 1)
        J.emit(mfma_ab1)
        J.s_waitcnt(mod=f"lgkmcnt(0)")

    if 0:
        for koff in range(J.div(K, wg_K)):
            loop_body(koff & 1)
    else:
        koff = J.gpr("su32", 0)
        loop_cnt = K // (2*wg_K)
        with J.While(koff[0] < loop_cnt):
            loop_body(0)
            loop_body(1)
            koff[0] += 1

        if K % (2*wg_K):
            loop_body(0)

    for lds in ldsA: J.free_lds(lds) 
    for lds in ldsB: J.free_lds(lds)

    stride_c = N * J.sizeof_bf16
    vdata = J.gpr(8, "vbf16x2")
    vbf16 = J.gpr(4, "vbf16x2")
    col = J.lane_id // 16
    swap_12_col = (col & 1) * 2 + (col >> 1)
    vaddr = J.gpr(((J.lane_id % 16) + warp_m * 16)*stride_c + swap_12_col * J.sizeof_DW4 + warp_n * 4 * J.sizeof_DW2)
    for m in range(nrM):
        for n in range(0, nrN, 2):
            J.v_accvgpr_read_b32(vdata[0], mfma_C[m,n,0])
            J.v_accvgpr_read_b32(vdata[1], mfma_C[m,n,1])
            J.uni_cvt_pk_bf16_f32(vbf16[0], vdata[0], vdata[1])

            J.v_accvgpr_read_b32(vdata[2], mfma_C[m,n,2])
            J.v_accvgpr_read_b32(vdata[3], mfma_C[m,n,3])
            J.uni_cvt_pk_bf16_f32(vbf16[1], vdata[2], vdata[3])

            J.v_accvgpr_read_b32(vdata[4], mfma_C[m,n+1,0])
            J.v_accvgpr_read_b32(vdata[5], mfma_C[m,n+1,1])
            J.uni_cvt_pk_bf16_f32(vbf16[2], vdata[4], vdata[5])

            J.v_accvgpr_read_b32(vdata[6], mfma_C[m,n+1,2])
            J.v_accvgpr_read_b32(vdata[7], mfma_C[m,n+1,3])
            J.uni_cvt_pk_bf16_f32(vbf16[3], vdata[6], vdata[7])

            #    a0    a1   a2   a3   | 01 23
            #    b0    b1   b2   b3   | 45 67
            #  v_permlane16_swap_b32(a, b)
            #    a0    b0   a2   b2   |
            #    a1    b1   a3   b3   |
            #
            # swap of row 1 & 2 are done by swapping lane-address 
            J.v_permlane16_swap_b32(vbf16[0], vbf16[2])
            J.v_permlane16_swap_b32(vbf16[1], vbf16[3])
            J.global_store_dwordx4(vaddr, vbf16, pC, mod=f"offset:{n*4*J.sizeof_DW2}")
        vaddr[0] += 16*stride_c



M,N,K = 24000,4096,8192

# https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a4w4_blockscale
# AITER_REBUILD=1

# mxfp4 
BUF_COPY = 32
weight_type = torch.float4_e2m1fn_x2
import aiter
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight
w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)

w_qt_scale_[...] = 1.0

w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
w_ref = (w_f32 * w_scale_f32).reshape(N, K)
assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
A = (torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16) + 1)*0.01
ref_out = A[0] @ w_ref.t()

from aiter import gemm_a4w4, per_1x32_f4_quant_hip
def _run_aiter(
                x: "Tensor",  # A:[M, K] bf16
                weight: "Tensor",  # B:[N, K/2] f4x2
                weight_scale: "Tensor",  # B_scale:[N, K/32] e8m0 paded
            ):
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    # use hip quant kernel for performance
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)

    #print(f"{x.shape}{x.dtype},  {x_q.shape}{x_q.dtype},    {x_s.shape}{x_s.dtype} ")
    #print(f"{weight.shape}{weight.dtype},   {weight_scale.shape}{weight_scale.dtype}")
    #assert 0
    # 32 alignment is enough for dim0 padding of output for
    # gemm_a4w4 kernel
    y = torch.empty(
        (M + 31) // 32 * 32,
        weight.shape[0],
        device=x_q.device,
        dtype=x.dtype,
    )

    gemm_a4w4(
        x_q, weight, x_s, weight_scale.view(x_s.dtype), y, bpreshuffle=True
    )
    return y[:M]

def _run_fp4gemm(
                x: "Tensor",  # A:[M, K] bf16
                weight: "Tensor",  # B:[N, K/2] f4x2
                weight_scale: "Tensor",  # B_scale:[N, K/32] e8m0 paded
            ):
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    # use hip quant kernel for performance
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)

    '''
        m, n = x.shape
        assert quant_dtype == dtypes.fp4x2
        assert n % 2 == 0    

        scale = (
            torch.empty(
                (
                    (m + 255) // 256 * 256,     # round-up (m) to 256
                    (n // 32 + 7) // 8 * 8,     # round-up (n//32) to 8
                ),
                dtype=torch.uint8,
                device=device,
            )
            # .fill_(0x7F)
            .view(dtypes.fp8_e8m0)
    '''

    # 32 alignment is enough for dim0 padding of output for
    # gemm_a4w4 kernel
    y = torch.empty(
        (M + 31) // 32 * 32,
        weight.shape[0],
        device=x_q.device,
        dtype=x.dtype,
    )

    blk_cnt = ((M + wg_M - 1) // wg_M) * ((N + wg_N - 1) // wg_N)

    gemm_a4w4_kernel(
        [blk_cnt], [256],
        wg_M, wg_N, N, K, False, True,
        x_q.data_ptr(), x_s.data_ptr(),
        weight.data_ptr(), weight_scale.data_ptr(),
        y.data_ptr(), M)

    return y[:M]

# https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/numeric.py#L5
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

y = _run_aiter(A[0], weight=w[0], weight_scale=w_scale[0])
y2 = _run_fp4gemm(A[0], weight=w[0], weight_scale=w_scale[0])
acc = []
acc.append("pass1" if calc_diff(y2, y) < 0.01 else "fail1")

print(f"{A[0].shape} {A[0].dtype} x  {w[0].shape}  {w[0].dtype} , {w_scale[0].shape} {w_scale[0].dtype} =>  {y.shape} {y.dtype}")


diff = calc_diff(ref_out, y)
diff2 = calc_diff(ref_out, y2)
print(ref_out)
print(y)
print(diff, diff2)
acc.append("pass2" if diff < 0.01 else "fail2")
acc.append("pass3" if diff2 < 0.01 else "fail3")



flops = 2 * M * N * K
mem_size = M * K * 2 + N * K * 0.5
with pyhip.torchPerf():
    di = 0
    for i in range(10):
        with pyhip.cudaPerf(flops, mem_size, name="aiter"):
            _run_aiter(A[di], weight=w[di], weight_scale=w_scale[di])
            di += 1
    for i in range(10):
        with pyhip.cudaPerf(flops, mem_size, name="fp4gemm"):
            _run_fp4gemm(A[di], weight=w[di], weight_scale=w_scale[di])
            di += 1

print(acc)
