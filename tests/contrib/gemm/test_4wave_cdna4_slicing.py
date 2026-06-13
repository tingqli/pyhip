import pyhip

from pyhip.contrib.gemm_4wave_slicing import *
from pyhip.contrib.gemm_cdna4 import *
import pytest
import functools
import torch

torch.set_printoptions(
    linewidth=3000,
    sci_mode=False,
    edgeitems=8,
)
torch.set_default_device("cuda")
torch.manual_seed(0)


def u64_diff(newer, older):
    return (int(newer) - int(older)) & ((1 << 64) - 1)


@pytest.mark.parametrize("M", [32, 256, 2400])
@pytest.mark.parametrize("N", [256, 256 * 6])
@pytest.mark.parametrize("K", [256])
def test_accuracy(M, N, K, use_pre_shuffle=0):
    wg_M = 256
    wg_N = 256
    blk_cnt = pyhip.div_up(M, wg_M) * pyhip.div_up(N, wg_N)

    A0 = torch.randn(M, K).to(dtype=torch.bfloat16)
    B0 = torch.randn(N, K).to(dtype=torch.bfloat16)
    ref_out = A0 @ B0.t()

    if use_pre_shuffle:
        A = pyhip.pre_shuffle(A0, 16)
        B = pyhip.pre_shuffle(B0, 16)
    else:
        A = A0
        B = B0

    cur_out = torch.ones(M, N, dtype=torch.bfloat16)
    timestamps = torch.zeros(4, dtype=torch.uint64)

    gemm_kernel_slicing(
        [blk_cnt],
        [4 * 64],
        wg_M,
        wg_N,
        N,
        K,
        use_pre_shuffle,
        blk_cnt,
        A.data_ptr(),
        B.data_ptr(),
        cur_out.data_ptr(),
        timestamps.data_ptr(),
        M,
    )

    # ref_out = ref_out[:130, 128:192]
    # dbg_out = cur_out[132:135, 128:192]
    # # print(f"--------------------------------------------------\n{dbg_out=}\n")
    # ref_out = ref_out[128:256, 128:256]
    # cur_out = cur_out[128:256, 128:256]
    # print(f"{ref_out=}\n{cur_out=}")
    if not torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01):
        # print(ref_out)
        # print(cur_out)
        print(ref_out[0].tolist())
        print(cur_out[0].tolist())
        idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
        if len(idx[0]):
            print(f"idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}")
        assert 0


def compare_perf(M, N, K, use_pre_shuffle=0):
    wg_M = 256
    wg_N = 256

    blk_cnt = (M // wg_M) * (N // wg_N)

    A0 = torch.randn(M, K).to(dtype=torch.bfloat16)
    B0 = torch.randn(N, K).to(dtype=torch.bfloat16)
    C0 = A0 @ B0.t()

    if use_pre_shuffle:
        A = pre_shuffle(A0, 16)
        B = pre_shuffle(B0, 16)
    else:
        A = A0
        B = B0

    C = torch.zeros(M, N, dtype=torch.bfloat16)
    timestamps = torch.zeros(4, dtype=torch.uint64)

    gemm_kernel_slicing(
        [blk_cnt],
        [4 * 64],
        wg_M,
        wg_N,
        N,
        K,
        use_pre_shuffle,
        blk_cnt,
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        timestamps.data_ptr(),
        M,
    )

    ref_out = C0
    cur_out = C
    acc = "pass"
    if not torch.allclose(ref_out, cur_out, rtol=0.01, atol=0.01):
        # print(f"================= ref_out : {ref_out.shape} ")
        # print(ref_out)
        # print(f"================= cur_out : {cur_out.shape} ")
        # print(cur_out)
        # idx = torch.where(torch.abs(ref_out - cur_out) > 0.03)
        # if len(idx[0]):
        #     print(f'idx = {idx}\nref={ref_out[idx]}\ncur={cur_out[idx]}\n{len(idx[0])}')
        # assert 0
        acc = "failed"

    DATA_CLONES = 40
    As = [torch.clone(A) for _ in range(DATA_CLONES)]
    Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
    Cs = [torch.clone(C) for _ in range(DATA_CLONES)]

    A0s = [torch.clone(A0) for _ in range(DATA_CLONES)]
    B0s = [torch.clone(B0) for _ in range(DATA_CLONES)]
    timestamps_list = [torch.zeros(4, dtype=torch.uint64) for _ in range(DATA_CLONES)]

    di = 0
    latency = []
    torch_latncy = []
    pre_cycles = []
    main_cycles = []
    epilog_cycles = []
    for i in range(30):
        di = (di + 1) % DATA_CLONES
        with pyhip.cudaPerf(
            M * N * K * 2, (M * K * 2 + K * N * 2), name=f"gemm_{di}"
        ) as p0:
            gemm_kernel_slicing(
                [blk_cnt],
                [4 * 64],
                wg_M,
                wg_N,
                N,
                K,
                use_pre_shuffle,
                blk_cnt,
                As[di].data_ptr(),
                Bs[di].data_ptr(),
                Cs[di].data_ptr(),
                timestamps_list[di].data_ptr(),
                M,
            )
        # torch.cuda.synchronize()
        t0, t1, t2, t3 = [x.item() for x in timestamps_list[di]]
        pre_cycles.append(u64_diff(t1, t0))
        main_cycles.append(u64_diff(t2, t1))
        epilog_cycles.append(u64_diff(t3, t2))
        latency.append(p0.dt_ms)
    for i in range(30):
        di = (di + 1) % DATA_CLONES
        with pyhip.cudaPerf(
            M * N * K * 2, (M * K * 2 + K * N * 2), name=f"torch_{di}"
        ) as p0:
            ref = torch.nn.functional.linear(A0s[di], B0s[di])
            # # gemm_kernel(
            #     [blk_cnt],
            #     [4 * 64],
            #     wg_M,
            #     wg_N,
            #     N,
            #     K,
            #     use_pre_shuffle,
            #     As[di].data_ptr(),
            #     Bs[di].data_ptr(),
            #     Cs[di].data_ptr(),
            #     M,
            # )
        torch_latncy.append(p0.dt_ms)
    latency.sort()
    torch_latncy.sort()
    print(f"ratio:{torch_latncy[0]/latency[0]}")

    print(f"{acc=}")

    print("\nPer-run phase ratios (pre/main/epilog):")
    for i, (pre, main, epilog) in enumerate(zip(pre_cycles, main_cycles, epilog_cycles), start=1):
        total = pre + main + epilog
        if total == 0:
            continue
        print(
            f"  Run {i:2d}: {100.0*pre/total:6.2f}% / {100.0*main/total:6.2f}% / {100.0*epilog/total:6.2f}%"
        )

    avg_pre = sum(pre_cycles) / len(pre_cycles)
    avg_main = sum(main_cycles) / len(main_cycles)
    avg_epilog = sum(epilog_cycles) / len(epilog_cycles)
    avg_total = avg_pre + avg_main + avg_epilog
    print("\nAverage phase cycles:")
    print(f"  pre:    {avg_pre:,.0f} ({100.0*avg_pre/avg_total:.2f}%)")
    print(f"  main:   {avg_main:,.0f} ({100.0*avg_main/avg_total:.2f}%)")
    print(f"  epilog: {avg_epilog:,.0f} ({100.0*avg_epilog/avg_total:.2f}%)")
    print(f"  total:  {avg_total:,.0f}")


if __name__ == "__main__":
    # test_accuracy(2400, 256*4, 256*6)
    # test_accuracy(256, 256, 256)
    compare_perf(M=4096, N=4096, K=8192)
