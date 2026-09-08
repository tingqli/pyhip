#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import pyhip

# PYTHONPATH=/tmp/triton-gfx950-v11-site:/mywork/pyhip LLVM_PASS_PLUGIN_PATH=/mywork/gfx950-gluon-tutorials/plugins/llir_scheduler/libLlirSched.so LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 TRITON_FORCE_MFMA_AGPR=1 TRITON_AMDGCNAS_PLUGIN=1 TRITON_CACHE_DIR=/tmp/a8w8_compare_cache python compare_a8w8_noscale_cold.py

PYHIP_ROOT = "/mywork/pyhip"
GLUON_ROOT = "/mywork/gfx950-gluon-tutorials"
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare FlyDSL E4M3 A8W8 with and without scale against " "Gluon E5M2 A8W8"
        )
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--clones", type=int, default=50)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case",
        choices=("both", "noscale", "scaled"),
        default="both",
        help="select which FlyDSL A8W8 variant to benchmark",
    )
    parser.add_argument(
        "--gluon", action="store_true", help="also benchmark Gluon E5M2 A8W8"
    )
    return parser.parse_args()


def load_gluon_kernel():
    if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

    import triton

    if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
        sys.path.insert(0, f"{GLUON_ROOT}/plugins/amdgcnas")
        import amdgcnas_plugin
        from triton import knobs

        knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

    sys.path.insert(0, f"{GLUON_ROOT}/kernels/gemm/utils")
    sys.path.insert(0, f"{GLUON_ROOT}/kernels/gemm/intra_wave/a8w8")
    from matmul_kernel import a8w8_kernel

    return triton, a8w8_kernel


def permute_scale(scale, rows):
    scale = scale.view(torch.uint8)
    groups = scale.shape[1]
    padded_rows = ((rows + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if rows != padded_rows:
        scale = torch.cat(
            (
                scale,
                torch.full(
                    (padded_rows - rows, groups),
                    127,
                    device="cuda",
                    dtype=torch.uint8,
                ),
            ),
            dim=0,
        )
    permuted = (
        scale.view(padded_rows // 128, 4, 32, groups)
        .permute(3, 0, 2, 1)
        .contiguous()
        .view(-1)
    )
    padding = torch.full((padded_rows * 4,), 127, device="cuda", dtype=torch.uint8)
    return torch.cat((permuted, padding)).view(torch.int32)


def make_fp8_clones(shape, clones, generator, label, include_gluon):
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    source = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    # source_head = source.view(-1)[:8].float().cpu().tolist()
    # source_sum = source.view(-1)[:4096].float().sum().item()

    e4m3_base = source.to(torch.float8_e4m3fn)
    e5m2_base = source.to(torch.float8_e5m2) if include_gluon else None
    scaled_base, raw_scale = per_1x32_mx_quant_hip(
        source.to(torch.bfloat16),
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    scale_base = permute_scale(raw_scale, shape[0])
    del source

    e4m3 = [e4m3_base.clone() for _ in range(clones)]
    e5m2 = [e5m2_base.clone() for _ in range(clones)] if include_gluon else None
    scaled = [scaled_base.clone() for _ in range(clones)]
    scales = [scale_base] * clones

    # print(
    #     f"SOURCE {label} seed_values={source_head} sum4096={source_sum:.9f} "
    #     f"e4m3_sum4096={e4m3_base.view(-1)[:4096].float().sum().item():.9f} "
    #     f"e5m2_sum4096={e5m2_base.view(-1)[:4096].float().sum().item():.9f}",
    #     flush=True,
    # )
    return e4m3, e5m2, scaled, scales


def make_shared_inputs(m, n, k, clones, seed, include_gluon):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    fly_a, gluon_a, scaled_a, scale_a = make_fp8_clones(
        (m, k), clones, generator, "A", include_gluon
    )
    fly_b, gluon_b, scaled_b, scale_b = make_fp8_clones(
        (n, k), clones, generator, "B", include_gluon
    )

    fly_c = [
        torch.empty((m * n,), device="cuda", dtype=torch.bfloat16)
        for _ in range(clones)
    ]
    scaled_c = [
        torch.empty((m * n,), device="cuda", dtype=torch.bfloat16)
        for _ in range(clones)
    ]
    gluon_c = (
        [torch.empty((m, n), device="cuda", dtype=torch.float16) for _ in range(clones)]
        if include_gluon
        else None
    )
    input_mib = clones * (m * k + n * k) / 1024**2
    print(
        f"INPUTS clones={clones} per_backend_input={input_mib:.0f} MiB "
        "source_dtype=float32 flydsl_dtype=float8_e4m3fn "
        f"gluon_dtype={'float8_e5m2' if include_gluon else 'disabled'} "
        "scaled_dtype=mxfp8_e4m3 "
        "scaled_layout=ab_padding scale_path=g2r",
        flush=True,
    )
    return (
        fly_a,
        fly_b,
        fly_c,
        scaled_a,
        scaled_b,
        scale_a,
        scale_b,
        scaled_c,
        gluon_a,
        gluon_b,
        gluon_c,
    )


def make_flydsl_launcher(
    m, n, k, inputs_a, inputs_b, outputs, scales_a=None, scales_b=None
):
    sys.path.insert(0, f"{PYHIP_ROOT}/tests/flydsl/fp8_gemm_950")
    import flydsl.compiler as flyc
    from test_mxfp8_gemm_4w import compile_gemm_fp8

    with_scale = scales_a is not None and scales_b is not None
    if with_scale:
        os.environ["SCALE_G2R"] = "1"
    else:
        empty_scale = torch.empty(1, device="cuda", dtype=torch.uint8)
        scales_a = [empty_scale] * len(inputs_a)
        scales_b = [empty_scale] * len(inputs_b)
    stream = torch.cuda.current_stream()
    arg_sets = [
        (
            inputs_a[index].view(torch.int8).view(-1),
            inputs_b[index].view(torch.int8).view(-1),
            scales_a[index],
            scales_b[index],
            outputs[index],
            m,
            stream,
        )
        for index in range(len(inputs_a))
    ]
    launcher = compile_gemm_fp8(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        n,
        k,
        lds_swizzle=False,
        b_lds_swizzle=None,
        preshuffle_b=False,
        permlane_epilogue=True,
        store_overlap=False,
        with_scale=with_scale,
        b_mxfp4=False,
    )
    kernel = flyc.compile[{"opt_level": 2}](launcher, *arg_sets[0])
    return lambda index: kernel(*arg_sets[index])


def make_gluon_launcher(m, n, k, inputs_a, inputs_b, outputs, triton, kernel):
    grid_mn = triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N)

    def launch(index):
        a = inputs_a[index]
        b = inputs_b[index].T
        c = outputs[index]
        kernel[(grid_mn, 1)](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GRID_MN=grid_mn,
            NUM_XCDS=8,
            GROUP_SIZE_M=4,
            num_warps=4,
            llvm_fn_attrs="amdgpu-agpr-alloc=256",
        )

    return launch


def benchmark(name, launch, clones, runs, flops, rw_bytes):
    for index in range(clones):
        launch(index)
    torch.cuda.synchronize()

    latencies_ms = []
    for iteration in range(runs):
        clone = (iteration + 1) % clones
        with pyhip.cudaPerf(flops, rw_bytes, name=f"{name}_{clone}", verbose=0) as perf:
            launch(clone)
        latencies_ms.append(perf.dt_ms)

    latencies_ms.sort()
    best_ms = min(latencies_ms)
    print(
        f"RESULT backend={name} clones={clones} runs={runs} "
        f"best_us={best_ms * 1e3:.3f} "
        f"best_tflops={flops / best_ms / 1e9:.3f}",
        flush=True,
    )


def main():
    args = parse_args()
    if args.clones < 1 or args.runs < 1:
        raise ValueError("--clones and --runs must be positive")
    if args.m % BLOCK_M or args.n % BLOCK_N or args.k % 256:
        raise ValueError("M/N must be multiples of 256 and K must be a multiple of 256")
    if args.gluon and os.environ.get("LLVM_PASS_PLUGIN_PATH"):
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

    inputs = make_shared_inputs(
        args.m, args.n, args.k, args.clones, args.seed, args.gluon
    )
    (
        fly_a,
        fly_b,
        fly_c,
        scaled_a,
        scaled_b,
        scale_a,
        scale_b,
        scaled_c,
        gluon_a,
        gluon_b,
        gluon_c,
    ) = inputs
    if args.case in ("both", "noscale"):
        flydsl_launch = make_flydsl_launcher(
            args.m, args.n, args.k, fly_a, fly_b, fly_c
        )
    if args.case in ("both", "scaled"):
        flydsl_scaled_launch = make_flydsl_launcher(
            args.m,
            args.n,
            args.k,
            scaled_a,
            scaled_b,
            scaled_c,
            scale_a,
            scale_b,
        )
    flops = 2 * args.m * args.n * args.k
    rw_bytes = args.m * args.k + args.n * args.k + 2 * args.m * args.n
    scaled_rw_bytes = rw_bytes + (args.m + args.n) * (args.k // 32)
    print(f"SHAPE M={args.m} N={args.n} K={args.k}")
    if args.case in ("both", "noscale"):
        benchmark("flydsl-e4m3", flydsl_launch, args.clones, args.runs, flops, rw_bytes)
    if args.case in ("both", "scaled"):
        benchmark(
            "flydsl-mxfp8-e4m3-scale-g2r-padding",
            flydsl_scaled_launch,
            args.clones,
            args.runs,
            flops,
            scaled_rw_bytes,
        )
    if args.gluon:
        triton, gluon_kernel = load_gluon_kernel()
        gluon_launch = make_gluon_launcher(
            args.m,
            args.n,
            args.k,
            gluon_a,
            gluon_b,
            gluon_c,
            triton,
            gluon_kernel,
        )
        benchmark("gluon-e5m2", gluon_launch, args.clones, args.runs, flops, rw_bytes)


if __name__ == "__main__":
    main()
