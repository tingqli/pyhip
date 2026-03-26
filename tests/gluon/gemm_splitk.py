import torch
import triton
import triton.language as tl
from functools import cache

from pyhip.contrib.gluon.gemm_splitk import gemm_splitk_kernel

#####################################################################
from pyhip import cudaPerf
from common.utils import gen_timing
from torch import Tensor
import pytest

# bf16 support no shuffle, but performance is worse than shuffled
SHUFFLE = 1
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
    import aiter
    if weight_type == torch.float4_e2m1fn_x2:
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
        if SHUFFLE:
            w_shuffled = shuffle_weight(w_).reshape(N // 16, -1)
        else:
            w_shuffled = w_
        w = [w_shuffled.clone() for _ in range(BUF_COPY)]
        w_scale = [None] * BUF_COPY
        w_ref = w_
    elif weight_type == torch.float8_e4m3fn:
        w_qt = torch.randint(-2, 3, [N, K], dtype=torch.float32).to(weight_type)
        w_qt_scale = torch.randint(-2, 3, [N // 128, K // 128], dtype=torch.float32)
        # # TODO
        # w_qt_scale[:] = 1
        w_f32 = w_qt.to(dtype=torch.float32).reshape(N // 128, 128, K // 128, 128)
        w_scale_f32 = w_qt_scale.reshape(N // 128, 1, K // 128, 1)
        w_ref = (w_f32 * w_scale_f32).reshape(N, K).to(dtype=torch.bfloat16)
        if SHUFFLE:
            w = [shuffle_weight(w_qt) for _ in range(BUF_COPY)]
        else:
            w = [w_qt.clone() for _ in range(BUF_COPY)]
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

    def run(A, weight, weight_scale, p_debug_buf=None):
        M, K = A.shape
        N = w_ref.shape[0]
        gemm_out = torch.empty([M, N], dtype=A.dtype, device=A.device)
        num_warps = 4
        if kernel_type == 'mxn_splitk_2s':
            BLOCK_TILE_SIZE_M = TILE_M
            BLOCK_TILE_SIZE_N = TILE_N

            x = gemm_splitk_kernel[(div_up(N, BLOCK_TILE_SIZE_N) * div_up(M, BLOCK_TILE_SIZE_M),)](
                A,                # bf16 [M, K]
                weight.T,         # bf16 [K/8 * 16 * 8, N/16]
                gemm_out,         # bf16 [M, N]
                weight_scale,
                M,
                weight_type,
                K,
                N,
                BLOCK_TILE_SIZE_M,
                BLOCK_TILE_SIZE_N,
                p_debug_buf,
                SHUFFLE,
                enable_debug=p_debug_buf is not None,
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

        grids = div_up(N, TILE_N) * div_up(M, TILE_M)
        p_debug_buf = torch.zeros([grids * 9], dtype=torch.int64)
        for _ in range(2):
            with cudaPerf(flops, mem_size, name=f"D{kernel_type}[{M=},{str(weight_type).split('.')[1]}]") as p:
                run(A=A[i], weight=w[i], weight_scale=w_scale[i], p_debug_buf=p_debug_buf)
            i = (i + 1) % BUF_COPY
        gen_timing(p_debug_buf, TILE_N * K * 2, TILE_N * K * TILE_M * 2)
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
    # perf.update(entry_common('mxn_splitk_2s', M=M, prec=[torch.bfloat16, get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    # perf.update(entry_common('mxn_splitk_2s', M=M, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
    perf.update(entry_common('mxn_splitk_2s', M=M, prec=[get_fp8type()], TILE_M=TILE_M, TILE_N=TILE_N, N=N, K=K))
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
    if 1:
        # UP&GATE: [2*intemediate*experts//tp8, hidden_states],
        N, K = 256*512, 2048
    else:
        # DOWN:[hiddenstates*experts, intemediate//tp8],
        N, K = 4096*512, 128*4
    # Ms = [1,2, 4, 8, 16, 32, 64]
    Ms = [16]
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