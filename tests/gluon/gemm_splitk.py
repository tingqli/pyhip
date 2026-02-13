import torch
import triton
import triton.language as tl
from functools import cache

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)

@cache
def gemm_splitk(
        BLOCK_TILE_SIZE_M: gl.constexpr,
        BLOCK_TILE_SIZE_N: gl.constexpr,
        num_warps: gl.constexpr = 4,
        ):
    BLOCK_TILE_SIZE_K: gl.constexpr = gl.constexpr(4 * 8 * num_warps)

    assert BLOCK_TILE_SIZE_M.bit_count() == 1, "BLOCK_TILE_SIZE_M must be power of 2"
    assert BLOCK_TILE_SIZE_N.bit_count() == 1, "BLOCK_TILE_SIZE_N must be power of 2"
    assert BLOCK_TILE_SIZE_M % 16 == 0, "BLOCK_TILE_SIZE_M must be multiple of 16"
    assert BLOCK_TILE_SIZE_N % 32 == 0, "BLOCK_TILE_SIZE_N must be multiple of 32"

    # input layout will be if using BlockedLayout(8 elements per thread):
    #   0  1  2  3
    #   4  5  6  7
    #   ...
    #   60 61 62 63
    # expected is:
    #   0 16 32 48
    #   1 17 33 49
    #   ...
    #   15 31 47 63
    # mem_a_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
    #                              threads_per_warp=[16, 4],
    #                              warps_per_cta=[1, num_warps],
    #                              order=[1, 0])
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    m = BLOCK_TILE_SIZE_M // 16
    bit_pos = 16
    while m // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        m = m // 2
    # thread:[1, 8(bf16)], lane: [16, 4], wave: [1, 4], rep thread: [M/16, 1]
    mem_a_layout: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                                warp_bases=[[0, 32], [0, 64]] if num_warps > 1 else [],
                                block_bases=[],
                                shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K])

    if num_warps == 1:
        # output layout
        reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16]]
        n = BLOCK_TILE_SIZE_N // 32
        bit_pos = 32
        while n // 2:
            reg_bases.append([0, bit_pos])
            bit_pos *= 2
            n = n // 2
        m = BLOCK_TILE_SIZE_M // 16
        bit_pos = 16
        while m // 2:
            reg_bases.append([bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        # thread:[1, 4(bf16)], lane: [16, 4], wave: [1, 1], rep thread: [M/16, N/32]
        mem_c_layout: gl.constexpr = gl.DistributedLinearLayout(
                                    reg_bases=reg_bases,
                                    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
                                    warp_bases=[],
                                    block_bases=[],
                                    shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N],)
        lds_layout_write = None
        lds_layout_read = None
    else:
        # lds layout
        # process 16x32 tile each
        # acc will be written to lds using the pyhical layout of [num_warps, BLOCK_TILE_SIZE_N // 32, BLOCK_TILE_SIZE_M, 32], 
        #   the column will apply the swizzle decribled in lds_layout_write(assume BLOCK_TILE_SIZE_M = 16, BLOCK_TILE_SIZE_N = 128):
        #       v_lshlrev_b32_e32 v38, 7, v2                    ; v2 = threadid, v38 = threadid * 128
        #       v_lshlrev_b32_e32 v39, 4, v0                    ; v0 = threadid, v39 = threadid * 16
        #       v_and_b32_e32 v38, 0x6780, v38                  ; v38 = threadid * 128 & 0x6780, which is (lane_id_mod_16 * 32 * size_f32) + (wave_id * 4 * 16 * 32 * size_f32)
        #                                                       ; `& 0x6780` will clear the bit position of `BLOCK_TILE_SIZE_N // 32` and `32`
        #       .loc	1 205 20                        ; gemm_splitk.py:205:20
        #       v_and_b32_e32 v39, 0x70, v39                    ; v39 = threadid * 16 & 0x70, -> threadid << 4 & 0x70, which is the (lane_id_mod_16 % 8) * 16 
        #       v_and_b32_e32 v0, 48, v0                        ; v0 = threadid, v0 = threadid & 48, which is the lane_id_div_16 * 16
        #       bitop3 `0x36` means:
        #         tmp |= ~a & ~b & c        ; cur+next lines is:
        #         tmp |= ~a & b & ~c        ; tmp |= ~a & (b ^ c)
        #         tmp |= a & ~b & ~c        ; cur+next lines is:
        #         tmp |= a & ~b & c         ; tmp |= a & ~b
        #       v_bitop3_b32 v0, v38, v0, v39 bitop3:0x36       ; col = v0 ^ v39, aka (lane_id_div_16 ^ (lane_id_mod_16 % 8)) * 16
        #       v_add_u32_e32 v38, 0, v0
        #       ds_write_b128 v38, v[6:9]
        lds_layout_write: gl.constexpr = gl.SwizzledSharedLayout(vec=4, per_phase=1, max_phase=16, order=[3, 2, 1, 0])
        reg_bases=[[0, 0, 0, 1], [1, 0, 0, 0], [2, 0, 0, 0]]
        n = BLOCK_TILE_SIZE_N // 32
        bit_pos = 1
        while n // 2:
            reg_bases.append([0, bit_pos, 0, 0])
            bit_pos *= 2
            n = n // 2
        m = BLOCK_TILE_SIZE_M // 16
        bit_pos = 16
        while m // 2:
            reg_bases.append([0, 0, bit_pos, 0])
            bit_pos *= 2
            m = m // 2
        # thread: [4wave, 2], lane: [4, 16], wave[4, 1], rep thread: [N/32, M/2]
        lds_layout_read: gl.constexpr = gl.DistributedLinearLayout(
                                    reg_bases=reg_bases,
                                    lane_bases=[[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16], [0, 0, 1, 0], [0, 0, 2, 0]],
                                    warp_bases=[[0, 0, 4, 0], [0, 0, 8, 0]],
                                    block_bases=[],
                                    shape=[num_warps, BLOCK_TILE_SIZE_N // 32, BLOCK_TILE_SIZE_M, 32])
        mem_c_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 2],
            threads_per_warp=[4, 16],
            warps_per_cta=[num_warps, 1],
            order=[1, 0],
        )

    @gluon.jit
    def kernel(
        p_input,            # bf16 [M, K]
        p_weight,           # bf16 [K/8 * 16 * 8, N/16]
        p_output,           # bf16 [M, N]
        M,
        weight_dtype: gl.constexpr,
        K: gl.constexpr,
        N: gl.constexpr,
        BLOCK_TILE_SIZE_M: gl.constexpr,
        BLOCK_TILE_SIZE_N: gl.constexpr,
        num_warps: gl.constexpr = 4,
    ):
        pid = gl.program_id(0)
        max_tile_n: gl.constexpr = N // BLOCK_TILE_SIZE_N
        tile_m = pid // max_tile_n
        tile_n = pid % max_tile_n
        num_pid = gl.num_programs(0)
        if weight_dtype == torch.bfloat16:
            mem_b_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 1],
                                        threads_per_warp=[64, 1],
                                        warps_per_cta=[num_warps, 1],
                                        order=[0, 1])
        mem_a_offset_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_a_layout))) % M
        mem_a_offsets = mem_a_offset_m[:, None] * K + \
                        gl.arange(0, BLOCK_TILE_SIZE_K, layout=gl.SliceLayout(0, mem_a_layout))[None, :]
        mem_b_offsets = tile_n * BLOCK_TILE_SIZE_N * K + gl.arange(0, BLOCK_TILE_SIZE_N // 16, layout=gl.SliceLayout(0, mem_b_layout))[None, :] * (K // 8 * 16 * 8) + \
                        gl.arange(0, BLOCK_TILE_SIZE_K * 16, layout=gl.SliceLayout(1, mem_b_layout))[:, None]
        c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                      instr_shape=[16, 16, 32],
                                                      transposed=True,
                                                      tiles_per_warp=[1, BLOCK_TILE_SIZE_M // 16, BLOCK_TILE_SIZE_N // 16],
                                                      warps_per_cta=[num_warps, 1, 1])
        a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
        b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offsets)
        b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offsets)
        acc = gl.zeros((num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N), dtype=gl.float32, layout=c_layout)
        s_vmem_read: gl.constexpr = 0x0020
        s_mfma: gl.constexpr = 0x0008
        s_ds_read: gl.constexpr = 0x0100
        s_ds_write: gl.constexpr = 0x0200

        for k_start in gl.static_range(0, K - BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_K):
            _amd_iglp_sched_barrier(0)
            a_offsets = mem_a_offsets + k_start + BLOCK_TILE_SIZE_K
            b_offsets = mem_b_offsets + (k_start // 8) * 16 * 8 + BLOCK_TILE_SIZE_K * 16
            next_a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
            if weight_dtype == torch.bfloat16:
                next_b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
            else:
                assert 0, "only bf16 weight is supported in this kernel"
            a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
            a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
            b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N)
            b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
            acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
            a = next_a
            b = next_b
            _amd_iglp_sched_group_barrier(s_vmem_read, BLOCK_TILE_SIZE_M // 16 + BLOCK_TILE_SIZE_N // 16, 0)
            _amd_iglp_sched_group_barrier(s_mfma, BLOCK_TILE_SIZE_M // 16 * BLOCK_TILE_SIZE_N // 16, 0)

        _amd_iglp_sched_barrier(0)
        # tail
        a = a.reshape(BLOCK_TILE_SIZE_M, num_warps, 4, 8).permute(1, 0, 2, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K // num_warps)
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        b = b.reshape(num_warps, 4, 16, 8, BLOCK_TILE_SIZE_N // 16).permute(0, 1, 3, 4, 2).reshape(num_warps, BLOCK_TILE_SIZE_K // num_warps, BLOCK_TILE_SIZE_N)
        b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)

        if num_warps == 1:
            out_offsets_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_c_layout))) % M
            out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_c_layout))) % N
            out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
            gl.amd.cdna3.buffer_store(gl.convert_layout(acc.reshape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N), mem_c_layout, assert_trivial=True).to(gl.bfloat16), p_output, out_offsets)
        else:
            # (num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N) --> (num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 32, 32) -->
            #   (num_warps, BLOCK_TILE_SIZE_N // 32, BLOCK_TILE_SIZE_M, 32)
            acc = acc.reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N // 32, 32).permute(0, 2, 1, 3)
            lds_c = gl.allocate_shared_memory(gl.float32, [num_warps, BLOCK_TILE_SIZE_N // 32, BLOCK_TILE_SIZE_M, 32], layout=lds_layout_write)
            lds_c.store(acc)
            acc_r = lds_c.load(layout=lds_layout_read).permute(0, 2, 1, 3).reshape(num_warps, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N)

            acc_r = acc_r.sum(axis=0).to(gl.bfloat16)

            out_offsets_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_c_layout))) % M
            out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_c_layout))) % N
            out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
            gl.amd.cdna3.buffer_store(gl.convert_layout(acc_r, mem_c_layout, assert_trivial=True), p_output, out_offsets)

    return kernel
#####################################################################
from pyhip import cudaPerf
from torch import Tensor
import pytest

def div_up(x, y):
    return (x + y - 1) // y
def _run_aiter(
                x: Tensor,  # A:[M, K] bf16
                weight: Tensor,  # B:[N, K/2] f4x2
                weight_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
            ):
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip, gemm_a16w16
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
    if x.dtype == torch.bfloat16:
        out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
        gemm_a16w16(x, weight, out, splitK=4, bpreshuffle=True)
        return out

    # use hip quant kernel for performance
    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)

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

def _run_batch(kernel_type, M=1, weight_type=torch.bfloat16, TILE_M=16, TILE_N=32, run_count=10, N=4096, K=4096):
    BUF_COPY = 32
    A = (torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16) + 1)*0.001
    from aiter.ops.shuffle import shuffle_weight
    if weight_type == torch.float4_e2m1fn_x2:
        import aiter
        from aiter.utility import fp4_utils
        w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
        w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
        w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K)
        assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
        w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.bfloat16:
        w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
        w_shuffled = shuffle_weight(w_).reshape(N // 16, -1)
        w = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w_scale = [None] * BUF_COPY
        w_ref = w_
    else:
        assert 0, f'Only fp4 weight is supported in this test, current weight_type={weight_type}'

    flops = 2 * M * N * K
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = M * K * 2 + N * K * ele_size

    def run(A, weight, weight_scale):
        M, K = A.shape
        N = w_ref.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)
        num_warps = 4
        if kernel_type == 'mxn_splitk_2s':
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N

            x = gemm_splitk(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, num_warps)[(div_up(N, BLOCK_TILE_SIZE_N) * div_up(M, BLOCK_TILE_SIZE_M),)](
                A,                # bf16 [M, K]
                weight.T,           # bf16 [K/8 * 16 * 8, N/16]
                gemm_out,         # bf16 [M, N]
                # weight_scale,
                M,
                weight_type,
                K,
                N,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                num_warps=num_warps
                )
            # print(x.asm['amdgcn'])
            # assert 0
        else:
            assert 0, f'not support kernel type "{kernel_type}"'
        return gemm_out

    tflops_res = []
    latencies = []
    bw = []
    if kernel_type == 'aiter':
        # aiter needs preshuffle weights
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                _run_aiter(x=A[i], weight=w[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())
    else:
        ref_out = A[0] @ w_ref.t()
        cur_out = run(A=A[0], weight=w[0], weight_scale=w_scale[0])
        i = 0
        for _ in range(run_count):
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                run(A=A[i], weight=w[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {M=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=}"
        else:
            print(f"{kernel_type}[{M=} {weight_type=}] acc OK")
    if run_count > 0:
        return {'flops': sum(tflops_res[1:])/len(tflops_res[1:]),              # tflops
                'latency': sum(latencies[1:])/len(latencies[1:]) * 1e6,        # us
                'bw': sum(bw[1:]) / len(bw[1:])}                               # GB/s

def is_arch_type(arch):
    props = torch.cuda.get_device_properties()
    return arch in props.gcnArchName

def get_fp8type():
    return torch.float8_e4m3fn if is_arch_type('950') else torch.float8_e4m3fnuz

def get_fp4type_if_valid():
    return torch.float4_e2m1fn_x2 if is_arch_type('950') else None

def entry_common(kernel_type, M, prec=[torch.bfloat16], TILE_M=32, TILE_N=64, N=4096, K=4096, run_count=10):
    perf = {}
    perf[kernel_type] = {}
    for weight_type in prec:
        if weight_type is None: continue
        perf_prec = {}
        for i in M:
            perf_prec[i] = _run_batch(kernel_type, M=i, weight_type=weight_type, TILE_M=TILE_M, TILE_N=TILE_N, run_count=run_count, N=N, K=K)
        perf[kernel_type][str(weight_type)] = perf_prec
    
    return perf

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def test_acc(TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    M = list(range(2, 64))
    # fix TILE_M=16, TILE_N=32
    M += list(range(128, 256))
    M += [i * 256 for i in range(1, 4)]
    M += [i * 2048 for i in range(1, 5)]
    M += list(range(2048 * 3, 2048 * 3 + 256))
    # TILE_M/N is configurable
    entry_common('mxn_splitk_2s', M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K, run_count=0)

def show_perf(perf, dict_tile_mn):
    print('\nsummary:')
    for kernel, vals in perf.items():
        for prec, vals_ in vals.items():
            for b, data in vals_.items():
                if kernel != 'aiter':
                    TILE_M, TILE_N = dict_tile_mn[f'{b}']
                    print(f'{kernel}[{prec:<4} B={b:<4}({TILE_M}x{TILE_N})]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')
                else:
                    print(f'{kernel}[{prec:<4} B={b:<4}]: {data["latency"]:5.0f} us, {data["bw"]:6.1f} GB/s, {data["flops"]:4.1f} tflops')

@pytest.mark.parametrize("M", [[1, 2, 4, 8, 12, 16, 32, 64]])
def test_perf(M, TILE_M=32, TILE_N=64, N=4096, K=4096):
    init_env()
    perf = {}
    #perf.update(entry_common('aiter', M, prec=[torch.bfloat16], N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N))
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', M=M, prec=[torch.bfloat16], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    return perf

def merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    #TILE_M = 16
    #TILE_N = 128
    # qwen3 235b/a22b qkv projection
    N, K = 9216, 4096
    # qwen3 235b/a22b qkv projection
    N, K = 4096*8*2, 8192 # 4096*8*2 /128 = 512
    Ms = [16, 32, 64, 128, 256]
    perf = {}
    dict_tile_mn = {}

    def get_tile_mn(M):
        num_CU = torch.cuda.get_device_properties().multi_processor_count
        solutions = []
        for tile_m in [16, 32, 64]:
            for tile_n in [32, 64, 128]:
                works = div_up(M, tile_m) * div_up(N, tile_n)
                if works >= num_CU:
                    round = works // num_CU
                    reminder = works % num_CU
                    solutions.append((round, reminder, tile_m, tile_n))
                else:
                    reminder = num_CU - works % num_CU
                    solutions.append((100000, reminder, tile_m, tile_n))
        # prefer less rounds; then less reminder
        TILE_M, TILE_N = sorted(solutions)[0][2:]
        return TILE_M, TILE_N

    TILE_M, TILE_N = get_tile_mn(64)
    for M in Ms:
        TILE_M, TILE_N = get_tile_mn(M)
        # if N == 9216 and K == 4096:
        #     if M in [16]:
        #         TILE_M = 16
        #         TILE_N = 64
        #     elif M in [32]:
        #         TILE_M = 32
        #         TILE_N = 64
        #     elif M in [64, 128]:
        #         TILE_M = 32
        #         TILE_N = 128

        print(f'final selected TILE_M={TILE_M}, TILE_N={TILE_N}')
        #TILE_M, TILE_N = 16, 64
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        #test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        perf = merge(perf, test_perf([M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)