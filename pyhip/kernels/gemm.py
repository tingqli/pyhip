import os
import random
from typing import Optional
from torch import Tensor

import pytest
os.environ['PYHIP_JIT_LOG'] = '0'
from pyhip import cudaPerf, jit, JIT
import torch

from functools import cache
try:
    # work as package
    from .common.gemm import UGEMM
    from .common.gemm_splitk import gemm_splitk
except:
    from common.gemm import UGEMM
    from common.gemm_splitk import gemm_splitk

USE_FP4_SHUFFLE_WEIGHT = 1
#####################################################################
# kernel define
@cache
def get_lane_id(J):
    vgpr_lane_id = J.gpr(J.threadIdx.x[0] & 63)
    return vgpr_lane_id

@cache
def get_lane_id_div(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) // divisor)

@cache
def get_lane_id_mod(J, divisor):
    assert isinstance(divisor, int)
    return J.gpr(get_lane_id(J) % divisor)

def div_up(x, y):
    return (x + y - 1) // y

@jit(with_debug_log=False)
def gemm_splitk_wd(J:JIT,
                   weight_dtype,
                   K,            # compile-time args
                   N,            # compile-time args
                   BLOCK_TILE_SIZE_M,
                   BLOCK_TILE_SIZE_N,
                   p_input:"void*",
                   p_weight:"void*",
                   p_output:"void*",
                   p_w_scale:"float*",
                   M:"int",):
    assert weight_dtype == torch.float4_e2m1fn_x2
    assert BLOCK_TILE_SIZE_N % 32 == 0, f'due to scale is packed with [2*16, 8*32], current BLOCK_TILE_SIZE_N={BLOCK_TILE_SIZE_N} is not supported'
    assert K % 1024 == 0, f'will read (16*4)*4(wave) bytes once in main loop, aka 256*2=512 elements; to use packed scale in K dimension, will double read; current K={K} is not supported'
    assert BLOCK_TILE_SIZE_M % 16 == 0
    assert N % BLOCK_TILE_SIZE_N == 0

    num_split_k = 4

    def get_k_bytes(k_in_elements):
        if weight_dtype == torch.bfloat16:
            return k_in_elements * J.sizeof_bf16
        elif weight_dtype == torch.float4_e2m1fn_x2:
            return k_in_elements // 2
        elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
            return k_in_elements
    stride_A = K * J.sizeof_bf16
    stride_B = get_k_bytes(K)
    stride_C = N * J.sizeof_bf16

    A_vert = BLOCK_TILE_SIZE_M // 16
    B_horz = BLOCK_TILE_SIZE_N // 16

    C_reg = J.gpr(B_horz, A_vert, 4, "vf32")

    C_reg[:] = 0

    # one WG per CU,  4 waves split on K
    lane_mod_16 = get_lane_id_mod(J, 16)
    lane_div_16 = get_lane_id_div(J, 16)
    warp_id = J.warp_id
    J.debug_setup((J.blockIdx.x[0] == 0) & (J.blockIdx.y[0] == 0) & (warp_id[0] == 0))

    voffset_b = J.gpr(B_horz, 'vu32')
    if weight_dtype == torch.float4_e2m1fn_x2:
        # shffled
        voffset_b[0] = J.blockIdx.x * (BLOCK_TILE_SIZE_N * stride_B) + J.lane_id * 16 + J.warp_id * (16 * get_k_bytes(K) // 4)
        for m in range(1, B_horz):
            voffset_b[m] = voffset_b[0] + get_k_bytes(K) * 16 * m
    else:
        assert 0

    buff_a = J.Buffer(p_input, M * (K * J.sizeof_bf16))
    buff_c = J.Buffer(p_output, M * (N * J.sizeof_bf16))

    voffset_a = J.gpr(A_vert, 'vu32')
    # a elements number:
    # bf16: 8 elements
    # fp8: 16 elements
    # fp4: 32 elements
    if weight_dtype == torch.bfloat16:
        a_element_num_per_thread = 8
    elif weight_dtype == torch.float4_e2m1fn_x2:
        a_element_num_per_thread = 32
    elif weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e4m3fnuz:
        a_element_num_per_thread = 16
    else:
        raise ValueError(f"Unsupported weight dtype: {weight_dtype}")
    voffset_a[0] = J.gpr((lane_mod_16 + J.blockIdx.y * BLOCK_TILE_SIZE_M) * stride_A) + lane_div_16 * (a_element_num_per_thread * J.sizeof_bf16) + J.warp_id * (K // 4 * J.sizeof_bf16)
    for m in range(1, A_vert):
        voffset_a[m] = voffset_a[0] + 16 * stride_A * m

    voffset_scale = None
    if weight_dtype == torch.float4_e2m1fn_x2:
        # 32 rows shared in a uint32
        voffset_scale = J.gpr(B_horz // 2, 'vu32')
        # the group of fp4 is 32 elements, and scale will align to 8 groups
        k_scale_stride = div_up(div_up(K, 32), 8) * 8
        p_w_scale[:] += BLOCK_TILE_SIZE_N * k_scale_stride * J.blockIdx.x
        voffset_scale[0] = J.lane_id * J.sizeof_fp32 + J.warp_id * (k_scale_stride // 8 // 4 * 64 * J.sizeof_fp32)
        for m in range(1, B_horz // 2):
            voffset_scale[m] = voffset_scale[0] + 32 * k_scale_stride * m

    buff_b = J.Buffer(p_weight, N * get_k_bytes(K))

    gemm_splitk(J, weight_dtype, K, N, num_split_k,
                buff_a, buff_b, p_w_scale,
                voffset_a, voffset_b, voffset_scale, C_reg, BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M, USE_FP4_SHUFFLE_WEIGHT=True)

    # split K
    lds_buff = J.LDSTensor([4 * BLOCK_TILE_SIZE_M, 32], torch.float)
    # each 32 N as a group
    n_groups = BLOCK_TILE_SIZE_N // 32
    vrow = J.gpr(lane_div_16 + warp_id * 4)
    c_offset = J.gpr(1, 'vu32')
    m_block_stride = J.gpr(1, 'vu32')
    m_block_stride[0] =BLOCK_TILE_SIZE_M * stride_C
    c_offset[0] = J.threadIdx.x[0] // 16 * stride_C + lane_mod_16 * 4 + J.blockIdx.y * m_block_stride + J.blockIdx.x * (BLOCK_TILE_SIZE_N * J.sizeof_bf16)
    s_c_offset = J.gpr(1, 'su32')
    s_c_offset[0] = 0
    for n in range(n_groups):
        for m in range(A_vert):
            col = lane_div_16
            row = lane_mod_16
            col = (row ^ col) % 8
            lds_buff.write("b128", C_reg[2 * n + 0, m], lane_mod_16 + warp_id * 16 + m * 64, col * 4)
            col = lane_div_16 + 4
            col = (row ^ col) % 8
            lds_buff.write("b128", C_reg[2 * n + 1, m], lane_mod_16 + warp_id * 16 + m * 64, col * 4)

        J.s_waitcnt(mod=f"lgkmcnt(0)")
        J.s_barrier()
        # each wave reduce 4 rows once
        for m in range(A_vert):
            c_in_wave = J.gpr(4, 2, "vf32", align=2)
            # interleave
            col = lane_mod_16 // 2
            row = vrow
            col = (row ^ col) % 8
            lds_buff.read("b64", c_in_wave[0], vrow + (0 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[1], vrow + (1 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[2], vrow + (2 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)
            lds_buff.read("b64", c_in_wave[3], vrow + (3 * 16 + m * 64), col * 4 + (lane_mod_16 & 1) * 2)

            J.s_waitcnt(mod=f"lgkmcnt(2)")
            J.v_pk_add_f32(c_in_wave[0], c_in_wave[0], c_in_wave[1])
            J.s_waitcnt(mod=f"lgkmcnt(0)")
            J.v_pk_add_f32(c_in_wave[2], c_in_wave[2], c_in_wave[3])
            J.v_pk_add_f32(c_in_wave[0], c_in_wave[0], c_in_wave[2])

            tmp = J.gpr(1, 'vu32')
            tmp[0] = c_offset[0] + m * 16 * stride_C + n * 32 * J.sizeof_bf16
            J.v_cvt_pk_bf16_f32(c_in_wave[0, 0], c_in_wave[0, 0], c_in_wave[0, 1])
            J.debug_log(c_in_wave[0, 0], torch_dtype=torch.bfloat16, message=f'{m=}{n=}')
            buff_c.store_dword(c_in_wave[0, 0], tmp, s_c_offset)
        J.s_barrier()

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
        N = weight.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)
        if kernel_type == 'mxn_splitk_2s':
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N

            gemm_splitk_wd([div_up(N, BLOCK_TILE_SIZE_N), div_up(M, BLOCK_TILE_SIZE_M)], [256],
                            weight.dtype, K, N, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N,
                            A.data_ptr(), weight.data_ptr(), gemm_out.data_ptr(), weight_scale.data_ptr() if weight_scale is not None else 0, M)
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
    entry_common('mxn_splitk_2s', M=M, prec=[get_fp4type_if_valid()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K, run_count=0)

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
    perf.update(entry_common('mxn_splitk_2s', M=M, prec=[get_fp4type_if_valid()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
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

        print(f'final selected TILE_M={TILE_M}, TILE_N={TILE_N}')
        dict_tile_mn[f'{M}'] = (TILE_M, TILE_N)
        test_acc(TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K)
        perf = merge(perf, test_perf([M], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    show_perf(perf, dict_tile_mn)