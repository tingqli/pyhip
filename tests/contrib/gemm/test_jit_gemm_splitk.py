import os
import random
from typing import Optional
from torch import Tensor

import pytest
os.environ['PYHIP_JIT_LOG'] = '0'
import torch
from pyhip import cudaPerf, torchPerf, calc_diff, div_up
from pyhip.contrib.jit_gemm_splitk import *

import aiter
from aiter.utility import fp4_utils
from aiter.ops.shuffle import shuffle_weight


#####################################################################
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
    from pyhip import cudaPerf
    BUF_COPY = 32
    A = (torch.randn([BUF_COPY, M, K], dtype=torch.bfloat16) + 1)*0.001
    if weight_type == torch.float4_e2m1fn_x2:
        import aiter
        from aiter.utility import fp4_utils
        from aiter.ops.shuffle import shuffle_weight
        w_ = torch.randint(-2, 3, [N, K], dtype=torch.bfloat16) / 2
        w_qt, w_qt_scale_ = aiter.get_torch_quant(aiter.QuantType.per_1x32)(w_, quant_dtype=weight_type)
        w_f32 = fp4_utils.mxfp4_to_f32(w_qt).to(dtype=torch.bfloat16).reshape(N, K // 32, 32)
        w_scale_f32 = fp4_utils.e8m0_to_f32(w_qt_scale_).to(dtype=torch.bfloat16).reshape(N, K // 32, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K)
        assert K % 256 == 0, f'e8m0_shuffle assume there will be 8 groups of 32 elements in K dim, current K={K} is not supported'
        w_qt_scale = fp4_utils.e8m0_shuffle(w_qt_scale_)
        w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz:
        from aiter.ops.shuffle import shuffle_weight
        # fp8 blockwise per 128x128 quantization
        def weight_per_128x128_quant(weight, quant_dtype):
            from aiter.ops.quant import pertoken_quant
            N_dim, K_dim = weight.shape
            assert N_dim % 128 == 0 and K_dim % 128 == 0, f"weight shape {weight.shape} is not aligned to 128 for per 128x128 quantization"
            
            weight_blocks = weight.view(
                N_dim // 128, 128, K_dim // 128, 128
            )  # [num_blocks_N, 128, num_blocks_K, 128]
            weight_blocks = weight_blocks.permute(
                0, 2, 1, 3
            ).contiguous()  # [num_blocks_N, num_blocks_K, 128, 128]
            weight_blocks = weight_blocks.view(
                -1, 128 * 128
            )  # [num_blocks, 128*128]
            weight_qt, weight_scale = pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                N_dim // 128, K_dim // 128, 128, 128
            )  # [num_blocks_N, num_blocks_K, 128, 128]
            weight_qt = weight_qt.permute(
                0, 2, 1, 3
            ).contiguous()  # [num_blocks_N, 128, num_blocks_K, 128]
            weight_qt = weight_qt.view(N_dim, K_dim)  # [N, K]
            weight_scale = weight_scale.view(
                N_dim // 128, K_dim // 128
            )  # [num_blocks_N, num_blocks_K]
            return weight_qt, weight_scale
        
        QUAN_BLOCK_SZ = 128
        assert N % QUAN_BLOCK_SZ == 0 and K % QUAN_BLOCK_SZ == 0, f"N and K must be multiples of {QUAN_BLOCK_SZ} for per block quantization"
        assert QUAN_BLOCK_SZ % TILE_N == 0, f"{QUAN_BLOCK_SZ=} must be multiples of {TILE_N=} for per block quantization"
        
        w_ = torch.randn([N, K], dtype=torch.bfloat16) / 10.0
        w_qt, w_qt_scale = weight_per_128x128_quant(w_, quant_dtype=weight_type)
        
        # Reconstruct w_ref for validation
        w_ref = (w_qt.to(dtype=torch.bfloat16).view(N // 128, 128, K // 128, 128) * w_qt_scale.view(N // 128, 1, K // 128, 1)).to(dtype=torch.bfloat16)
        w_ref = w_ref.view(N, K)
        w_qt = w_qt.view(N, K)
        w_qt_scale = w_qt_scale.view(N // 128, K // 128)
        w = [shuffle_weight(w_qt, layout=(16, 16)).clone() for _ in range(BUF_COPY)]
        w_scale = [w_qt_scale.clone() for _ in range(BUF_COPY)]
    elif weight_type == torch.bfloat16:
        from aiter.ops.shuffle import shuffle_weight
        w_ = torch.rand([N, K], dtype=torch.bfloat16) / 2.0
        w_ref = w_
        w_scale = [None] * BUF_COPY
        w = [shuffle_weight(w_, layout=(16, 16)).clone() for _ in range(BUF_COPY)]
    else:
        assert 0, f'Only fp4 ,fp8,bf16 weights are supported in this test, current weight_type={weight_type}'

    flops = 2 * M * N * K
    if weight_type == torch.bfloat16:
        ele_size = 2
    elif weight_type == torch.float4_e2m1fn_x2:
        ele_size = 0.5
    else:
        ele_size = 1
    mem_size = M * K * 2 + N * K * ele_size
    if weight_type == torch.float4_e2m1fn_x2:
        mem_size += (N * K // 32)  # scale for fp4
    elif wei_is_fp8(weight_type):
        mem_size += (N * K // 128 // 128)*4  # scale for fp8

    def run(A, weight, weight_scale):
        M, K = A.shape
        N = weight.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)
        if kernel_type == 'mxn_splitk_2s':
            if 1:
                BLOCK_TILE_SIZE_M = TILE_M
                BLOCK_TILE_SIZE_N = TILE_N

                gemm_splitk_wd([div_up(N, BLOCK_TILE_SIZE_N), div_up(M, BLOCK_TILE_SIZE_M)], [256],
                                weight.dtype, K, N, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                                A.data_ptr(), weight.data_ptr(), gemm_out.data_ptr(), weight_scale.data_ptr() if weight_scale is not None else 0, M)
            else:
                gemm_splitk_jit(A, weight, gemm_out, weight_scale, b_preshuffle=True)
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
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]", verbose=0) as p:
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
            with cudaPerf(flops, mem_size, name=f"{kernel_type}[{M=},{str(weight_type).split('.')[1]}]", verbose=0) as p:
                run(A=A[i], weight=w[i], weight_scale=w_scale[i])
            i = (i + 1) % BUF_COPY
            tflops_res.append(p.tflops())
            latencies.append(p.dt())
            bw.append(p.bw())

        diff = calc_diff(ref_out, cur_out)
        if diff > 0.02:
            #if not torch.allclose(ref_out, cur_out, rtol=0.1, atol=0.03):
            print(ref_out)
            print(cur_out)
            idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
            if len(idx[0]):
                print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
            assert 0, f"{kernel_type=}, {M=}, {weight_type=}, {TILE_M=}, {TILE_N=}, {run_count=} {diff=}"
        else:
            quantype=""
            print(f"{kernel_type}[{M=} {weight_type=}{quantype}] acc OK")
        
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

def wei_is_fp8(weight_type):
    return weight_type == torch.float8_e4m3fn or weight_type == torch.float8_e4m3fnuz

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
    entry_common('mxn_splitk_2s', M=M, prec=[get_fp4type_if_valid(),torch.bfloat16, get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K, run_count=0)

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
    perf.update(entry_common('aiter', M, prec=[get_fp4type_if_valid()], N=N, K=K, TILE_M=TILE_M, TILE_N=TILE_N))
    # TILE_M/N is configurable
    perf.update(entry_common('mxn_splitk_2s', M=M, prec=[get_fp4type_if_valid(),torch.bfloat16, get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
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
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256]
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
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        perf = merge(perf, test_perf([M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)