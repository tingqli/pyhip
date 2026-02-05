import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)

@gluon.jit
def gemm_splitk(
        p_input,            # bf16 [M, K]
        p_weight,           # bf16 [K/8 * 16 * 8, N/16]
        p_output,           # bf16 [M, N]
        p_w_scale,
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
    BLOCK_TILE_SIZE_K: gl.constexpr = 4 * 8 * num_warps
    # blocklayout will be(8 elements per thread):
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
    if num_warps == 1:
        warp_bases:gl.constexpr = []
        if BLOCK_TILE_SIZE_M == 16:
            reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
        elif BLOCK_TILE_SIZE_M == 32:
            reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4], [16, 0]]
        else:
            gl.static_assert(False, "Unsupported BLOCK_TILE_SIZE_M")    
        gl.static_print('reg_bases:', reg_bases)
    else:
        warp_bases:gl.constexpr = [[0, 32], [0, 64]]
    mem_a_layout: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=reg_bases,
                                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                                warp_bases=warp_bases,
                                block_bases=[],
                                shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K])
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
                                                  warps_per_cta=[1, num_warps])
    a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
    b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    tl.static_print('b fma layout:', b_fma_layout, gl.to_linear_layout(b_fma_layout, [32, BLOCK_TILE_SIZE_N]))
    acc = gl.zeros((BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N), dtype=gl.float32, layout=c_layout)
    for k_start in gl.static_range(0, K, BLOCK_TILE_SIZE_K):
        a_offsets = mem_a_offsets + k_start
        b_offsets = mem_b_offsets + (k_start // 8) * 16 * 8
        a = gl.amd.cdna3.buffer_load(p_input, a_offsets)
        if weight_dtype == torch.bfloat16:
            b = gl.amd.cdna3.buffer_load(p_weight, b_offsets)
        else:
            assert 0, "only bf16 weight is supported in this kernel"
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        # tl.static_print('b shape:', b.shape, 'b layout', gl.to_linear_layout(mem_b_layout, b.shape))
        b = b.reshape(4, 16, 8, BLOCK_TILE_SIZE_N // 16).permute(0, 2, 3, 1).reshape(BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_N)
        b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)

    if num_warps == 1:
        out_warp_bases:gl.constexpr = []
        if BLOCK_TILE_SIZE_M == 16:
            if BLOCK_TILE_SIZE_N == 32:
                out_reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16]]
            elif BLOCK_TILE_SIZE_N == 64:
                out_reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16], [0, 32]]
            else:
                gl.static_assert(False, "Unsupported BLOCK_TILE_SIZE_N")    
        elif BLOCK_TILE_SIZE_M == 32:
            if BLOCK_TILE_SIZE_N == 32:
                out_reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16], [16, 0]]
            elif BLOCK_TILE_SIZE_N == 64:
                out_reg_bases: gl.constexpr = [[0, 1], [0, 2], [0, 16], [0, 32], [16, 0]]
            else:
                gl.static_assert(False, "Unsupported BLOCK_TILE_SIZE_N")
        else:
            gl.static_assert(False, "Unsupported BLOCK_TILE_SIZE_M")
        gl.static_print('reg_bases:', out_reg_bases)
    else:
        out_warp_bases:gl.constexpr = [[0, 16], [0, 32]]
    mem_c_layout: gl.constexpr = gl.DistributedLinearLayout(
                                 reg_bases=out_reg_bases,
                                 lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
                                 warp_bases=out_warp_bases,
                                 block_bases=[],
                                 shape=[BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N],)
    out_offsets_m = (tile_m * BLOCK_TILE_SIZE_M + gl.arange(0, BLOCK_TILE_SIZE_M, layout=gl.SliceLayout(1, mem_c_layout))) % M
    out_offsets_n = (tile_n * BLOCK_TILE_SIZE_N + gl.arange(0, BLOCK_TILE_SIZE_N, layout=gl.SliceLayout(0, mem_c_layout))) % N
    out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
    gl.amd.cdna3.buffer_store(gl.convert_layout(acc, mem_c_layout, assert_trivial=True).to(gl.bfloat16), p_output, out_offsets)

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
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip
    M = x.shape[0]
    N = weight.shape[0]
    K = weight.shape[1]
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
        if kernel_type == 'mxn_splitk_2s':
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N

            # gemm_splitk_wd([div_up(N, BLOCK_TILE_SIZE_N), div_up(M, BLOCK_TILE_SIZE_M)], [256],
            #                 weight.dtype, K, N, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
            #                 A.data_ptr(), weight.data_ptr(), gemm_out.data_ptr(), weight_scale.data_ptr() if weight_scale is not None else 0, M)
            x = gemm_splitk[(div_up(N, BLOCK_TILE_SIZE_N) * div_up(M, BLOCK_TILE_SIZE_M),)](
                A,                # bf16 [M, K]
                weight.T,           # bf16 [K/8 * 16 * 8, N/16]
                gemm_out,         # bf16 [M, N]
                weight_scale,
                M,
                weight_type,
                K,
                N,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                num_warps=1
                )
            #print(x.asm['amdgcn'])
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
    #perf.update(entry_common('aiter', M, prec=[get_fp4type_if_valid()], N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N))
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
    #N, K = 4096, 8192
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

        if (TILE_M, TILE_N) not in [(16, 32), (32, 32)]:
            # TODO
            TILE_M, TILE_N = 16, 32
        print(f'final selected TILE_M={TILE_M}, TILE_N={TILE_N}')
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        #test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        perf = merge(perf, test_perf([M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)