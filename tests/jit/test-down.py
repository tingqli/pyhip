import pyhip
import torch

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

dtype = torch.float8_e5m2fnuz
dtype = torch.float8_e4m3fnuz
dtype = torch.bfloat16
o_dtype = torch.bfloat16

M = 64
N = 2048
K = 96
# since M & K are small, A can be resident in AccGPRs, 4 waves loads & compute along N dimension

# each wave do 32 x 32,
# 4 wave

A = torch.randn(M, K)
B = torch.randn(N, K)
A = A.to(dtype=dtype)
B = B.to(dtype=dtype)

ref = A.to(dtype=torch.float) @ B.t().to(dtype=torch.float)
ref = ref.to(dtype=o_dtype)
print(ref)

@pyhip.jit(with_debug_log=False)
def down_kernel(J:pyhip.JIT, mfma_MN, num_mfma_n, BM, N, K, pA:"void*", pB:"void*", pC:"void*"):
    sizeof_DW = 4
    sizeof_DW2 = sizeof_DW * 2
    sizeof_DW4 = sizeof_DW * 4
    sizeof_bf16 = 2

    pA[:] = pA[:] + J.blockIdx.x * (M*K*sizeof_bf16)
    pB[:] = pB[:] + J.blockIdx.x * (N*K*sizeof_bf16)
    pC[:] = pC[:] + J.blockIdx.x * (M*N*sizeof_bf16)

    # given DW4 lane size, how many bf16 items along K direction
    mfma_K = (64//mfma_MN) * (sizeof_DW4//sizeof_bf16) # mfma_K = 32

    # load A [BM x K] bf16 into AccGPRs 
    num_mfma_m = J.div(BM, mfma_MN)
    num_mfma_k = J.div(K, mfma_K)  # K=96, num_mfma_k=3

    A = J.gpr(num_mfma_m, num_mfma_k, 4, "abf16x2")
    buff_a = J.Buffer(pA, BM * K * sizeof_bf16)
    row = J.lane_id % mfma_MN
    col = J.lane_id // mfma_MN
    vaddr = J.gpr(row * (K*sizeof_bf16) + col * sizeof_DW4)
    soff = J.gpr("su32")
    soff[0] = 0
    for m in range(num_mfma_m):
        for k in range(num_mfma_k):
            buff_a.load_dwordx4(A[m,k], vaddr, soff, offset12=k*mfma_K*sizeof_bf16)
        soff[0] = soff[0] + mfma_MN * K * sizeof_bf16

    # 4 warps work in parallel along N dimension
    buff_b = J.Buffer(pB, N * K * sizeof_bf16)
    # ping-pong buffer
    B = J.gpr(2, num_mfma_n, num_mfma_k, 4, "abf16x2")
    C = J.gpr(2, num_mfma_m, num_mfma_n, 4, "vf32")
    
    loop_cnt_n = J.div(N, (4 * num_mfma_n * mfma_MN))

    # prelog0, load Bn0
    # prelog1, load Bn1, compute Cn0
    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    voff_b = J.gpr(J.lane_id * sizeof_DW4 + J.gpr(J.warp_id * (mfma_MN * K * sizeof_bf16)))
    soff_b = J.gpr("su32")
    soff_b[0] = 0
    def load_next_B(index):
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * sizeof_bf16
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * sizeof_bf16)

    def load_next_B_gen(index):
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                yield buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * sizeof_bf16
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * sizeof_bf16)


    def mfma_generator(index):
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    Ci = 0 if k == 0 else C[index,m,n]
                    yield J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,0:1], A[m,k,0:1], Ci)
        for k in range(num_mfma_k):
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    yield J.v_mfma_f32_16x16x16_bf16(C[index,m,n], B[index,n,k,2:3], A[m,k,2:3], C[index,m,n])

    mfma_cycles = 16 if mfma_MN == 16 else 32
    emit_mfma = J.emitter(mfma_cycles)
    emit_bload = J.emitter(1)

    # prelog0, load Bn0
    load_next_B(0)

    # prelog1, load Bn1, compute Cn0
    load_next_B(1)
    J.s_waitcnt(mod=f"vmcnt({num_mfma_n*num_mfma_k})")

    mfma0 = mfma_generator(0)
    emit_mfma([mfma0])

    # loop:    load Bn2, compute Cn1, store Cn0 to LDS & load Cn0 & store to HBM
    s_weight = J.gpr(2, "vf32")
    s_weight[0] = 1.0
    s_weight[1] = 1.0
    s_cvt_bf16_bias = J.gpr("su32")
    s_cvt_bf16_bias[0] = 0x00008000

    vmem_lane_size = sizeof_DW

    lds_padding = (4 if vmem_lane_size == sizeof_DW else 8) * sizeof_bf16 # to avoid bank-conflict
    lds_width = num_mfma_n * 4 * mfma_MN * sizeof_bf16
    lds_stride = lds_width + lds_padding
    lds = J.alloc_lds((num_mfma_m * mfma_MN) * (lds_stride))
    #lds = J.alloc_lds(64*1024)

    # WG level write C into LDS
    row = J.threadIdx.x % mfma_MN
    col = J.threadIdx.x // mfma_MN
    voff_c_lds_w = J.gpr(row * lds_stride + col * (4 * sizeof_bf16))
    def ds_write_C(index):
        for m in range(num_mfma_m):
            for n in range(num_mfma_n):
                offset = lds + m*mfma_MN*lds_stride + n*(4*mfma_MN*sizeof_bf16)
                # suppose 4 fp32 in C[0:3] was already converted into packed bf16 and stored in C[0:1]
                J.ds_write_b64(voff_c_lds_w, C[index, m, n, 0:1], mod=f"offset:{offset}")

    # WG level load C from LDS
    
    num_lanes_ldsr = J.div(lds_width, vmem_lane_size)
    col = J.threadIdx.x % num_lanes_ldsr
    row = J.threadIdx.x // num_lanes_ldsr
    num_rows_per_load = J.div(256, num_lanes_ldsr)
    num_loads = J.div(num_mfma_m * mfma_MN, num_rows_per_load)
    voff_c_lds_r = J.gpr(row * lds_stride + col * (vmem_lane_size))
    vmem_stride = N * sizeof_bf16
    voff_vmem = J.gpr(row * vmem_stride + col * (vmem_lane_size))
    temp_c = J.gpr(num_loads, vmem_lane_size//sizeof_DW, "vbf16x2")
    def ds_load_C():
        for i in range(num_loads):
            offset = i * num_rows_per_load * lds_stride
            J.ds_read_b32(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")

    def atomic_pk_add_bf16():
        voff = J.gpr("vu32")
        voff[0] = voff_vmem[0]
        for i in range(num_loads):
            J.global_atomic_pk_add_bf16(voff, temp_c[i], pC)
            #J.global_store_dword(voff, temp_c[i], pC)
            if i + 1 < num_loads:
                voff[0] = voff[0] + num_rows_per_load * vmem_stride
        return num_loads

    def cvt_f32_to_pk_bf16(index):
        for m in range(num_mfma_m):
            for n in range(num_mfma_n):
                #J.v_pk_mul_f32(C[index,m,n,0:1], C[index,m,n,0:1], s_weight)
                #J.v_pk_mul_f32(C[index,m,n,2:3], C[index,m,n,2:3], s_weight)
                J.v_add_u32(C[index,m,n,0], C[index,m,n,0], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,1], C[index,m,n,1], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,2], C[index,m,n,2], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,3], C[index,m,n,3], s_cvt_bf16_bias)
                J.pk_f32_to_bf16(C[index,m,n,0], C[index,m,n,0], C[index,m,n,1])
                J.pk_f32_to_bf16(C[index,m,n,1], C[index,m,n,2], C[index,m,n,3])

    # directly output to vmem
    vmem_atomic_lane_size = sizeof_DW2
    row = J.threadIdx.x % mfma_MN
    col = J.threadIdx.x // mfma_MN
    vmem_atomic_vaddr = J.gpr(row * vmem_stride + col * (vmem_atomic_lane_size))

    def loop_body(ni):
        #load_next_B(n&1)

        J.s_waitcnt(mod=f"vmcnt({num_loads})")
        mfma1 = mfma_generator((ni+1)&1)

        B_loader = load_next_B_gen(ni&1)
        
        #load_next_B(ni&1)

        """
        index = ni & 1
        for n in range(num_mfma_n):
            for k in range(num_mfma_k):
                buff_b.load_dwordx4(B[index,n,k], voff_b, soff_b)
                emit_mfma([mfma1], 32)
                soff_b[0] = soff_b[0] + mfma_MN * mfma_K * sizeof_bf16
            soff_b[0] = soff_b[0] + (3*num_mfma_k * mfma_MN * mfma_K * sizeof_bf16)
            #soff_b[0] = soff_b[0] + (-num_mfma_k * mfma_MN * mfma_K * sizeof_bf16)
        """

        #cvt_f32_to_pk_bf16(n&1)
        index = ni&1
        for m in range(num_mfma_m):
            emit_bload([B_loader], 1)
            for n in range(num_mfma_n):
                J.v_pk_mul_f32(C[index,m,n,0:1], C[index,m,n,0:1], s_weight)
                J.v_pk_mul_f32(C[index,m,n,2:3], C[index,m,n,2:3], s_weight)
                J.v_add_u32(C[index,m,n,0], C[index,m,n,0], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,1], C[index,m,n,1], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,2], C[index,m,n,2], s_cvt_bf16_bias)
                J.v_add_u32(C[index,m,n,3], C[index,m,n,3], s_cvt_bf16_bias)
                J.pk_f32_to_bf16(C[index,m,n,0], C[index,m,n,0], C[index,m,n,1])
                J.pk_f32_to_bf16(C[index,m,n,1], C[index,m,n,2], C[index,m,n,3])
            #emit_mfma([mfma1], 16)

        if 0:
            for m in range(num_mfma_m):
                for n in range(num_mfma_n):
                    offset = m * mfma_MN * vmem_stride + n * (4*mfma_MN * sizeof_bf16)
                    J.global_atomic_pk_add_bf16(vmem_atomic_vaddr[0] + offset, C[index,m,n,0], pC)
                    J.global_atomic_pk_add_bf16(vmem_atomic_vaddr[0] + offset + 2*sizeof_bf16, C[index,m,n,1], pC)
                    emit_mfma([mfma1], 64)
            vmem_atomic_vaddr[0] = vmem_atomic_vaddr[0] + (4 * num_mfma_n * mfma_MN * sizeof_bf16)
        else:
            # ds_write_C(ni&1)
            index = ni & 1
            for m in range(num_mfma_m):
                emit_bload([B_loader], 1)
                for n in range(num_mfma_n):
                    offset = lds + m*mfma_MN*lds_stride + n*(4*mfma_MN*sizeof_bf16)
                    # suppose 4 fp32 in C[0:3] was already converted into packed bf16 and stored in C[0:1]
                    J.ds_write_b64(voff_c_lds_w, C[index, m, n, 0:1], mod=f"offset:{offset}")
                    emit_mfma([mfma1], 16)
            emit_mfma([mfma1], 32)
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.s_barrier()

            #ds_load_C()
            for i in range(num_loads):
                offset = i * num_rows_per_load * lds_stride
                if vmem_lane_size == sizeof_DW4:
                    J.ds_read_b128(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
                else:
                    assert vmem_lane_size == sizeof_DW
                    J.ds_read_b32(temp_c[i], voff_c_lds_r, mod=f"offset:{offset}")
                emit_mfma([mfma1], 32)

            #atomic_pk_add_bf16()
            voff = J.gpr("vu32")
            voff[0] = voff_vmem[0]
            for i in range(num_loads):
                J.s_waitcnt(mod=f"lgkmcnt({min(15,num_loads - i - 1)})")
                
                if vmem_lane_size == sizeof_DW4:
                    J.global_store_dwordx4(voff, temp_c[i], pC)      # this is fast:  (48us)
                else:
                    assert vmem_lane_size == sizeof_DW
                    # the bigger the M is, the bigger the perf-diff is
                    #J.global_store_dword(voff, temp_c[i], pC)       # this is slightly slower than directly store  (49us)
                    J.global_atomic_pk_add_bf16(voff, temp_c[i], pC) # this is much slower than directly store      (60us)
                emit_mfma([mfma1], 32)

                if i + 1 < num_loads:
                    voff[0] = voff[0] + num_rows_per_load * vmem_stride

        emit_mfma([mfma1])
        emit_bload([B_loader])
        voff_vmem[0] = voff_vmem[0] + (4 * num_mfma_n * mfma_MN * sizeof_bf16)

    if 0:
        for n in range(J.div(N, 4 * num_mfma_n * mfma_MN)):
            loop_body(n)
        return
    loop_i = J.gpr("su32")
    loop_i[0] = 0
    loop_cnt = J.div(N, 8 * num_mfma_n * mfma_MN)
    J.s_waitcnt(mod=f"vmcnt(0)")
    with J.While(loop_i[0] < loop_cnt):
        loop_body(0)
        loop_body(1)
        loop_i[0] = loop_i[0] + 1

    #v_mfma_f32_32x32x16_bf8_bf8

def pre_shuffle(x, mfma_MN):
    M, K = x.shape
    K_bytes = K * x.itemsize
    sizeof_DWORDX4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DWORDX4//x.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DWORDX4
    assert K_bytes % mfma_K_bytes == 0

    x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    x = x.permute(0,2,3,1,4)
    return x.contiguous()

mfma_MN = 16
B2 = pre_shuffle(B, mfma_MN)
out = torch.zeros(M, N, dtype=o_dtype)

num_mfma_n = 2

down_kernel([1],[256], mfma_MN, num_mfma_n, M, N, K, A.data_ptr(), B2.data_ptr(), out.data_ptr())
print(ref)
print(out)
acc = True
if not torch.allclose(out, ref, atol=0.01, rtol=0.01):
    idx = torch.where(torch.abs(ref - out) > 0.03)
    if len(idx[0]):
        print(f'idx = {idx}\nref={ref[idx]}\ncur={out[idx]}\n{len(idx[0])}')
    acc = False

DATA_CLONES = 40
num_blocks = 80*2
A = torch.randn(num_blocks, M, K)
B = torch.randn(num_blocks, N, K)
A = A.to(dtype=dtype)
B = B.to(dtype=dtype)
out = torch.zeros(num_blocks, M, N, dtype=o_dtype)
As = [torch.clone(A) for _ in range(DATA_CLONES)]
Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
Cs = [torch.clone(out) for _ in range(DATA_CLONES)]

di = 0
for i in range(10):
    di = (di + 1)%DATA_CLONES
    with pyhip.cudaPerf((M*N*K*2*num_blocks), (num_blocks*(M+K)*N*2), name=f"down_kernel {di}") as p0:
        down_kernel([num_blocks],[256], mfma_MN, num_mfma_n, M, N, K, As[di].data_ptr(), Bs[di].data_ptr(), Cs[di].data_ptr())



