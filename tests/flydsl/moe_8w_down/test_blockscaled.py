# SPDX-License-Identifier: MIT
"""M256/M128 packed paths plus retained PyHIP/FlyDSL BN32/BN64 baselines."""

import argparse
from pathlib import Path
import sys

import aiter
import pytest
import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight

import pyhip
from pyhip.ops.moe.asm.moe_gemm_8wave import moe_gemm_8wave_down
from moe_8wave_down import flydsl_moe_gemm_8wave_down as legacy_flydsl_down
from moe_multistage_down import flydsl_moe_gemm_8wave_down
import moe_multistage_down_m128 as m128_kernel
from moe_multistage_pipeline import DownReduceWorkspace, compile_packed_down_reduce
from moe_multistage_reduce import make_moe_sum
from pyhip.ops.moe.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids


def make_m128_priority3_down(*, n, k=256, topk, num_experts):
    return m128_kernel.make_m128_down(n=n, k=k, topk=topk, num_experts=num_experts)


def make_m128_persistent_down(*, n, k=256, topk, num_experts):
    return m128_kernel.make_m128_down(n=n, k=k, topk=topk, num_experts=num_experts,
                                      persistent=True)


def make_m256_swizzle_down(*, n, k=256, topk, num_experts):
    return flydsl_moe_gemm_8wave_down(n=n, k=k, topk=topk, num_experts=num_experts,
                                     persistent=False)


CANDIDATES = {
    "pyhip": ("PyHIP", 64, None),
    "flydsl_bn32": ("FlyDSL BN32", 32, legacy_flydsl_down),
    "flydsl_bn64": ("FlyDSL BN64", 64, legacy_flydsl_down),
    "256x128 persist": ("256x128 persist", 128, flydsl_moe_gemm_8wave_down),
    "256x128": ("256x128", 128, make_m256_swizzle_down),
    "128x128": ("128x128", 128, make_m128_priority3_down),
    "128x128 persist": ("128x128 persist", 128, make_m128_persistent_down),
}
DEFAULT_CANDIDATES = tuple(CANDIDATES)
M128_CANDIDATES = ("128x128", "128x128 persist")
PACKED_CANDIDATES = ("256x128 persist", "256x128", *M128_CANDIDATES)
SORT_BLOCK_M = {key: 128 if key in M128_CANDIDATES else 256 for key in CANDIDATES}
ACTIVATION_QUANT = aiter.get_hip_quant(aiter.QuantType.per_1x128)


def select_candidates(n, candidates=None):
    if n <= 0 or n % 512:
        raise ValueError("N must be a positive multiple of 512")
    selected = list(DEFAULT_CANDIDATES if candidates is None else candidates)
    if not selected or any(key not in CANDIDATES for key in selected):
        raise ValueError("select at least one known candidate")
    for key in M128_CANDIDATES:
        if key in selected and n % 1024:
            reason = f"{key}: fixed OC8 requires N to be a positive multiple of 1024"
            if candidates is not None:
                raise ValueError(reason)
            print(f"SKIP {reason}; N={n}")
            selected.remove(key)
    return selected


def make_pyhip_down(*, n, k, topk, num_experts):
    """Existing PyHIP BN64/OC1 is a test baseline, not another shipped kernel."""
    def down(output, input_q, weight, input_scales, weight_scales,
             ids, routes, experts, valid, counter):
        counter.zero_()
        moe_gemm_8wave_down(
            [256], [512], output.numel() * output.element_size() > (1 << 32),
            "fp8", 256, 64, num_experts, n, k, 1, False, True, topk,
            ids.data_ptr(), routes.data_ptr(), experts.data_ptr(), valid.data_ptr(),
            weight.data_ptr(), weight_scales.data_ptr(), input_q.data_ptr(), input_scales.data_ptr(),
            output.data_ptr(), input_q.shape[0], counter,
        )
        return output
    down.config = {"block_m": 256, "block_n": 64, "num_oc_splits": 1,
                   "persistent_workgroups": 256, "output_layout": "routed"}
    return down


def with_torch_sum(down, workspace):
    """Retain routed baseline + preallocated TOPK sum on the shared storage."""
    def buffers(args):
        output, input_q, expert_ids = args[0], args[1], args[7]
        storage, _ = workspace.prepare(output, input_q, expert_ids)
        tokens, topk = input_q.shape[:2]
        return storage[:tokens * topk].view(tokens, topk, output.shape[1])

    def launch(*args):
        middle = buffers(args)
        down(middle, *args[1:])
        torch.sum(middle, dim=1, out=args[0])
        return args[0]

    def components(*args):
        middle = buffers(args)
        return {"gemm": lambda: down(middle, *args[1:]),
                "reduce": lambda: torch.sum(middle, dim=1, out=args[0])}

    launch.benchmark_components = components
    launch.poison_workspace = lambda *args: buffers(args).fill_(torch.nan)
    launch.config = {**getattr(down, "config", {}), "output_layout": "routed",
                     "reduction": "torch", "includes_inverse": False, "includes_reduce": True}
    return launch


def make_packed_pipeline(candidate, *, n, k=256, topk, num_experts, workspace=None):
    """Compose each local packed kernel with matching inverse/reducer strides."""
    assert candidate in PACKED_CANDIDATES
    down = CANDIDATES[candidate][2](n=n, k=k, topk=topk, num_experts=num_experts)
    return compile_packed_down_reduce(down, n=n, topk=topk, workspace=workspace)


def make_routing(tokens, topk, experts, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(tokens, experts, generator=generator, device="cuda", dtype=torch.float32)
    ids = scores.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)
    weights = torch.rand(tokens, topk, generator=generator, device="cuda", dtype=torch.float32)
    return ids, weights / weights.sum(dim=1, keepdim=True)


def torch_reference_down(input_q, input_scales_k_major, weight_q, weight_scales, topk_ids, topk_weights):
    """Independent K128 block-scale reference, routing multiply, then BF16."""
    tokens, topk, k = input_q.shape
    experts, n, _ = weight_q.shape
    rows = tokens * topk
    a = input_q.float().reshape(rows, k // 128, 128)
    scales = input_scales_k_major.view(k // 128, rows).t().float()
    output = torch.empty((rows, n), dtype=torch.bfloat16, device=input_q.device)
    expert_per_row, routing = topk_ids.reshape(-1), topk_weights.reshape(-1)
    for expert in range(experts):
        row_ids = torch.where(expert_per_row == expert)[0]
        if row_ids.numel() == 0:
            continue
        accum = torch.zeros((row_ids.numel(), n), dtype=torch.float32, device=input_q.device)
        for kb in range(k // 128):
            for bn in range(n // 128):
                w = weight_q[expert, bn * 128:(bn + 1) * 128, kb * 128:(kb + 1) * 128]
                partial = a[row_ids, kb] @ w.float().t()
                factor = scales[row_ids, kb, None] * weight_scales[expert, bn, kb]
                accum[:, bn * 128:(bn + 1) * 128] += partial * factor
        output[row_ids] = (accum * routing[row_ids, None]).to(torch.bfloat16)
    return output.view(tokens, topk, n)


def resort_inputs(inputs, assignments, routing, *, n, experts, sort_block_m):
    """Same semantic routes/A/B/scales; a separate native-sized sorted buffer."""
    assert sort_block_m in (128, 256)
    assert assignments.shape == routing.shape == inputs[0].shape[:2]
    ids, weights, eids, valid, _ = moe_sorting(
        assignments, routing, experts, n, torch.bfloat16, sort_block_m, None, None, 0)
    return (*inputs[:4], ids, weights, eids, valid, inputs[-1])


def make_case(tokens, n, experts, topk, seed, *, sort_block_m=256):
    """Real quantization/shuffle and explicit AITER sorting alignment."""
    assert n % 512 == 0 and 0 < topk <= min(experts, 255)
    assert sort_block_m in (128, 256)
    torch.manual_seed(seed)
    a_bf16 = torch.randn((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn((experts, n, 256), dtype=torch.bfloat16, device="cuda")
    a, sa = ACTIVATION_QUANT(a_bf16, quant_dtype=dtypes.fp8, transpose_scale=True)
    blocks = w_bf16.view(experts, n // 128, 128, 2, 128).permute(0, 1, 3, 2, 4).contiguous()
    qblocks, sb = aiter.pertoken_quant(blocks.view(experts, -1, 128 * 128), quant_dtype=dtypes.fp8)
    w = qblocks.view(experts, n // 128, 2, 128, 128).permute(0, 1, 3, 2, 4).contiguous().view(experts, n, 256)
    sb = sb.view(experts, n // 128, 2)
    assignments, routing = make_routing(tokens, topk, experts, seed + 1)
    ids, routes, eids, valid, _ = moe_sorting(assignments, routing, experts, n, torch.bfloat16, sort_block_m, None, None, 0)
    reference = torch_reference_down(a, sa, w, sb, assignments, routing)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    return (a, shuffle_weight(w, layout=(16, 16)), sa, sb, ids, routes, eids, valid, counter), reference


def error_stats(output, reference):
    # Chunking avoids an extra full-shape FP32 allocation in the large case.
    max_abs, total_abs, mismatches, nonfinite = 0.0, 0.0, 0, False
    dot, denominator = 0.0, 0.0
    for begin in range(0, output.shape[0], 256):
        actual, expected = output[begin:begin + 256].float(), reference[begin:begin + 256].float()
        error = (actual - expected).abs()
        max_abs = max(max_abs, error.max().item())
        total_abs += error.sum().item()
        mismatches += (error > 0.01 + 0.01 * expected.abs()).sum().item()
        nonfinite |= not torch.isfinite(actual).all().item()
        # Keep the original pyhip.calc_diff definition without allocating
        # two full default-shape float64 tensors. Reporting is outside timers.
        actual64, expected64 = actual.double(), expected.double()
        dot += (actual64 * expected64).sum().item()
        denominator += (actual64.square() + expected64.square()).sum().item()
    return {"status": "FAIL" if mismatches or nonfinite else "PASS", "mismatch_count": mismatches,
            "mismatches": f"{mismatches}/{output.numel()}", "max_abs": max_abs,
            "mean_abs": total_abs / output.numel(), "diff": 1 - 2 * dot / denominator if denominator else 0.0}


def unpack_routes(source, ids, valid, tokens, topk, *, sort_block_m=256):
    """Test-only Torch decoding, outside timers; never a production restore."""
    limit = int(valid[0].item())
    n = source.shape[1]
    assert sort_block_m in (128, 256) and limit % sort_block_m == 0
    sorted_values = source[:limit].view(-1, n // 64, sort_block_m, 64).permute(0, 2, 1, 3).reshape(limit, n)
    encoded = ids[:limit].to(torch.int64)
    token, slot = encoded & 0xFFFFFF, (encoded >> 24) & 0xFF
    live = (token < tokens) & (slot < topk)
    result = torch.empty((tokens, topk, n), dtype=source.dtype, device=source.device)
    result[token[live], slot[live]] = sorted_values[live]
    return result


def print_markdown_table(headers, rows):
    widths = [max(len(str(header)), *(len(str(row[i])) for row in rows)) for i, header in enumerate(headers)]
    def formatted(row):
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
    print(formatted(headers))
    print(formatted(["-" * width for width in widths]))
    for row in rows:
        print(formatted(row))


def performance(us, flops, padded_flops, nbytes, ideal_nbytes):
    return {"elapsed_us": us, "us": f"{us:.3f}" if us is not None else "N/A",
            **{name: f"{work / us / 1e6:.3f}" if us is not None else "N/A" for name, work in (
                ("effective_tflops", flops), ("padded_tflops", padded_flops),
                ("tb_per_s", nbytes), ("tb_per_s_ideal", ideal_nbytes))}}


def count_live_m_blocks(ids, valid_rows, tokens, topk, block_m):
    """Untimed work model: inspect every row, including signed encoded slots."""
    encoded = ids[:valid_rows].to(torch.int64)
    live = ((encoded & 0xFFFFFF) < tokens) & (((encoded >> 24) & 0xFF) < topk)
    return int(live.view(-1, block_m).any(dim=1).sum().item())


def run_test(tokens=16384, model_dim=6144, experts=384, topk=8, seed=1234,
             candidates=None, profile=False, reduce_output=True):
    """Compare packed candidates and retained baselines on shared storage.

    K256 is fixed; legacy baselines use OC1, M256 OC4, and M128 OC8.
    BF16 routes are checked before TOPK sum so rounding near
    cancellation is not confused with a reduction correctness regression.
    """
    assert torch.cuda.is_available() and torch.cuda.get_device_properties().gcnArchName.startswith("gfx950")
    candidates = select_candidates(model_dim, candidates)
    assert not profile or len(candidates) == 1
    args, reference_routes = make_case(tokens, model_dim, experts, topk, seed)
    a, b, eids = args[0], args[1], args[6]
    baseline = make_pyhip_down(n=model_dim, k=256, topk=topk, num_experts=experts)
    baseline_routes = torch.empty_like(reference_routes)
    baseline(baseline_routes, *args)
    torch.cuda.synchronize()
    baseline_error = error_stats(baseline_routes, reference_routes)
    assert baseline_error["status"] == "PASS", f"PyHIP down vs Torch: {baseline_error}"
    reference = baseline_routes.sum(dim=1) if reduce_output else reference_routes
    del baseline_routes
    output = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    workspace = DownReduceWorkspace()
    storage, _ = workspace.prepare(output, a, eids)
    routed = storage[:tokens * topk].view(tokens, topk, model_dim)
    sorted_inputs = {256: args}
    if any(key in M128_CANDIDATES for key in candidates):
        assignments, routing = make_routing(tokens, topk, experts, seed + 1)
        sorted_inputs[128] = resort_inputs(args, assignments, routing, n=model_dim, experts=experts, sort_block_m=128)
    flops = 2 * tokens * topk * model_dim * 256
    weight_bytes = model_dim * 256 * b.element_size()
    intermediate_bytes = tokens * topk * model_dim * 2
    other_bytes = a.numel() * a.element_size() + intermediate_bytes
    reduce_bytes = intermediate_bytes + output.numel() * 2 if reduce_output else 0
    results = []

    for key in candidates:
        name, block_n, factory = CANDIDATES[key]
        sort_block_m = SORT_BLOCK_M[key]
        args = sorted_inputs[sort_block_m]
        ids, eids, valid = args[4], args[6], args[7]
        inputs = (output, *args)
        packed = storage[:eids.numel() * sort_block_m]
        valid_rows = int(valid[0].item())
        valid_blocks = valid_rows // sort_block_m
        unique_experts = torch.unique(eids[:valid_blocks]).numel()
        down_ideal = other_bytes + unique_experts * weight_bytes
        packed_output = key in PACKED_CANDIDATES
        storage.fill_(torch.nan)
        output.fill_(torch.nan)
        if packed_output:
            down = factory(n=model_dim, k=256, topk=topk, num_experts=experts)
            pipeline = compile_packed_down_reduce(down, n=model_dim, topk=topk, workspace=workspace)
            pipeline.poison_workspace(*inputs)
            launch = (lambda: pipeline(*inputs)) if reduce_output else (lambda: down(packed, *args))
            config = pipeline.config if reduce_output else down.config
            components = pipeline.benchmark_components(*inputs) if reduce_output else {}
        else:
            routed_down = baseline if factory is None else factory(
                n=model_dim, k=256, topk=topk, num_experts=experts,
                block_m=256, block_n=block_n, num_oc_splits=1,
            )
            routed_down.config = {"block_m": 256, "block_n": block_n, "num_oc_splits": 1,
                                  "persistent_workgroups": 256, "output_layout": "routed"}
            routed_pipeline = with_torch_sum(routed_down, workspace)
            launch = (lambda: routed_pipeline(*inputs)) if reduce_output else (lambda: routed_down(routed, *args))
            config = routed_pipeline.config if reduce_output else routed_down.config
            components = routed_pipeline.benchmark_components(*inputs) if reduce_output else {}
        config = {**config, "sort_block_m": sort_block_m, "num_waves": config.get("num_waves", 8),
                  "persistent_workgroups": config.get("persistent_workgroups", 0)}
        work_tasks = (count_live_m_blocks(ids, valid_rows, tokens, topk, 128)
                      if key == "128x128" else valid_blocks)
        compute_rows = work_tasks * config["block_m"]
        padded_flops = 2 * compute_rows * model_dim * 256
        down_bytes = other_bytes + work_tasks * weight_bytes
        elapsed, times = None, {}
        inverse_bytes = (valid_rows + 2 * tokens * topk) * 4 if reduce_output and packed_output else 0
        total_bytes, ideal_bytes = down_bytes + reduce_bytes + inverse_bytes, down_ideal + reduce_bytes + inverse_bytes
        if profile:
            for _ in range(23):
                launch()
                torch.cuda.synchronize()
        else:
            _, elapsed = pyhip.run_perftest(launch, num_warmup=2, num_iters=10, num_copies=1,
                num_flops=padded_flops, num_bytes=total_bytes, num_verbose=1,
                num_name=key + ("_total" if reduce_output else "_down"),
                num_spec_tag=f"M={tokens * topk},N={model_dim},K=256")
            for label, component in components.items():
                _, times[label] = pyhip.run_perftest(component, num_warmup=2, num_iters=10,
                                                   num_copies=1, num_verbose=0, num_name=f"{key}_{label}")
            launch()
        torch.cuda.synchronize()
        if packed_output:
            decoded = unpack_routes(packed, ids, valid, tokens, topk, sort_block_m=sort_block_m)
            down_error = error_stats(decoded, reference_routes)
            assert down_error["status"] == "PASS", f"{key} packed down vs Torch: {down_error}"
        actual = output if reduce_output else decoded if packed_output else routed
        errors = error_stats(actual, reference)
        down_us = times.get("gemm") if reduce_output else elapsed
        reduce_us = times.get("reduce")
        stats = {"name": name, "candidate": key, "block_n": block_n, "config": config,
                 "valid_expert_blocks": valid_blocks, "valid_padded_rows": valid_rows,
                 "sort_block_m": sort_block_m, "unique_experts": unique_experts,
                 "work_tasks": work_tasks, "compute_rows": compute_rows,
                 "rw_bytes": total_bytes, "ideal_rw_bytes": ideal_bytes,
                 "down_rw_bytes": down_bytes, "down_ideal_rw_bytes": down_ideal,
                 "reduce_rw_bytes": reduce_bytes, "inverse_rw_bytes": inverse_bytes,
                 "measurement_scope": "down+reduce+inverse_if_needed" if reduce_output else "down",
                 "components_us": times, **errors,
                 **performance(elapsed, flops, padded_flops, total_bytes, ideal_bytes),
                 **{"down_" + name: value for name, value in performance(down_us, flops, padded_flops, down_bytes, down_ideal).items()},
                 "reduce_tb_per_s": f"{reduce_bytes / reduce_us / 1e6:.3f}" if reduce_us is not None else "N/A"}
        results.append(stats)
        if packed_output:
            del decoded

    props = torch.cuda.get_device_properties(a.device)
    print(f"\nShape: tokens={tokens}, TOPK={topk}, E={experts}, N={model_dim}, K=256; {props.name}, CUs={props.multi_processor_count}")
    print("M256: N128, OC4, persistent256, PF3, B SC1, packed NT output; custom reduce256/2048/NT.")
    if "128x128" in candidates:
        print("M128: 4-wave independent CTA, OC8, PF3/ring4, transpose width2, virtual-worker N phase, SC1/NT stores, Memory priority3, early B + spaced DMA.")
    if "256x128" in candidates:
        print("M256 nonpersistent: unchanged 8-wave pipeline, width8 valid-prefix transpose, counter preserved.")
    if "128x128 persist" in candidates:
        print("M128 persistent: 512 workers, width4/8 shards, paired B publication/ring4, cached scales, DPP masks, SC1/NT stores; last-exit queue reset included.")
    print("Reference: independent Torch block-scale routes; final sum uses validated PyHIP BF16 routes. rtol=atol=0.01.")
    print("Sorting: M128 uses128, M256/legacy use256; identical semantic routes, separate sorted metadata. Sorting is outside timing.")
    print("Timing: same A/B/scales/intermediate/final addresses, required counter/queue reset included, warmup2/iters10.")
    print("Down is independently timed; Total includes inverse rebuild and reduce, never a component sum.")
    print("Work/bytes: executed padded rows; A once + B per executed M-block (or once per unique expert, ideal) + valid BF16 writes; not HBM counters.")
    if not reduce_output:
        print("Down-only compares M256/M128 PACKED output with baseline ROUTED output after untimed Torch decoding.")
    if profile:
        print("Profile:23 direct launches, no performance timings.")
    rows = [[title, *(r["config"].get(field, "N/A") for r in results)] for title, field in (
        ("Block M", "block_m"), ("Sorting block M", "sort_block_m"), ("Block N", "block_n"), ("Waves per CTA", "num_waves"), ("OC splits", "num_oc_splits"),
        ("Persistent CTAs", "persistent_workgroups"), ("Down output layout", "output_layout"),
        ("Reduce implementation", "reduction"),
    )]
    for title, field in (("Status", "status"), ("Valid expert blocks (sorting M)", "valid_expert_blocks"),
                         ("Sorted padded rows", "valid_padded_rows"), ("Unique experts", "unique_experts"),
                         ("Executed M-blocks", "work_tasks"), ("Executed padded rows", "compute_rows"),
                         ("Down time (us)", "down_us"), ("Down effective TF/s", "down_effective_tflops"),
                         ("Down padded TF/s", "down_padded_tflops"), ("Down TB/s (B per M-block)", "down_tb_per_s"),
                         ("Down TB/s (B once/expert, ideal)", "down_tb_per_s_ideal")):
        rows.append([title, *(r[field] for r in results)])
    if reduce_output:
        for label, title in (("inverse", "Invert + fill time (us)"), ("reduce", "Reduce time (us)")):
            rows.append([title, *(f"{r['components_us'][label]:.3f}" if label in r["components_us"] else "N/A" for r in results)])
        for title, field in (("Reduce TB/s", "reduce_tb_per_s"), ("Total Time (us)", "us"),
                             ("Total effective TF/s", "effective_tflops"), ("Total padded TF/s", "padded_tflops"),
                             ("Total TB/s (B per M-block)", "tb_per_s"), ("Total TB/s (B once/expert, ideal)", "tb_per_s_ideal")):
            rows.append([title, *(r[field] for r in results)])
    by_name = {r["name"]: r for r in results}
    for baseline_name in ("PyHIP", "FlyDSL BN64"):
        for field, scope in (("down_elapsed_us", "Down"), ("elapsed_us", "Total")):
            if scope == "Total" and not reduce_output:
                continue
            base = by_name.get(baseline_name, {}).get(field)
            rows.append([f"{scope} speedup vs {baseline_name}", *(
                f"{base / r[field]:.3f}x" if base is not None and r[field] is not None else "N/A" for r in results)])
    rows.extend([["Max abs error", *(f"{r['max_abs']:.6g}" for r in results)],
                 ["Mean abs error", *(f"{r['mean_abs']:.6g}" for r in results)],
                 ["calc_diff", *(f"{r['diff']:.6g}" for r in results)],
                 ["Mismatches", *(r["mismatches"] for r in results)]])
    print_markdown_table(["Metric", *(r["name"] for r in results)], rows)
    assert all(r["status"] == "PASS" for r in results), "correctness failed"
    print("PASS: all executed kernels match the validated reference")
    return results


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 required")


@pytest.mark.parametrize("candidate,tokens,n,experts,topk", [("256x128 persist", *shape) for shape in (
    (65, 512, 4, 1), (257, 512, 4, 2), (513, 1024, 8, 8), (129, 6144, 4, 2),
    (257, 1536, 4, 2), (257, 2048, 4, 2), (257, 2560, 4, 2), (128, 512, 257, 1),
)] + [("128x128", 65, 1024, 4, 1), ("128x128", 513, 1024, 8, 8),
    ("128x128", 129, 6144, 4, 2), ("128x128 persist", 65, 1024, 4, 1),
    ("128x128 persist", 513, 1024, 8, 8), ("128x128 persist", 129, 6144, 4, 2),
    ("256x128", 257, 512, 4, 2), ("256x128", 129, 6144, 4, 2)])
def test_packed_pipeline(candidate, tokens, n, experts, topk):
    require_gpu()
    args, ref = make_case(tokens, n, experts, topk, 2026)
    baseline = torch.empty_like(ref)
    make_pyhip_down(n=n, k=256, topk=topk, num_experts=experts)(baseline, *args)
    assert error_stats(baseline, ref)["status"] == "PASS"
    expected = baseline.sum(dim=1)
    sort_block_m = SORT_BLOCK_M[candidate]
    if sort_block_m != 256:
        assignments, routing = make_routing(tokens, topk, experts, 2027)
        args = resort_inputs(args, assignments, routing, n=n, experts=experts, sort_block_m=sort_block_m)
    guard = torch.full((expected.numel() + 2 * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = guard[n:-n].view_as(expected)
    inputs = (output, *args)
    workspace = DownReduceWorkspace()
    storage, inverse = workspace.prepare(output, args[0], args[6], sort_block_m=sort_block_m)
    pointers = storage.data_ptr(), inverse.data_ptr()
    pipeline = make_packed_pipeline(candidate, n=n, topk=topk, num_experts=experts, workspace=workspace)
    assert pipeline.workspace is workspace
    args[-1].fill_(0x123456)
    def check():
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.isfinite(output).all() and torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
        assert (workspace.data.data_ptr(), workspace.inverse.data_ptr()) == pointers
        config = pipeline.config
        resets_counter = candidate == "256x128 persist" or config.get("counter_reset", False)
        expected_counter = (args[7][0].item() // sort_block_m * config["num_oc_splits"]
                    + config["persistent_workgroups"] if resets_counter else 0x123456)
        assert args[-1].item() == expected_counter
    pipeline.poison_workspace(*inputs)
    assert pipeline(*inputs) is output
    torch.cuda.synchronize()
    check()
    decoded = unpack_routes(pipeline.workspace.data, args[4], args[7], tokens, topk, sort_block_m=sort_block_m)
    assert error_stats(decoded, ref)["status"] == "PASS"
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        pipeline(*inputs)
    for _ in range(2):
        pipeline.poison_workspace(*inputs)
        output.fill_(torch.nan)
        args[-1].fill_(0x123456)
        capture.replay()
        torch.cuda.synchronize()
        check()
    old_ids, old_valid = args[4].clone(), args[7].clone()
    args[4].fill_((topk << 24) | tokens)
    pipeline.poison_workspace(*inputs)
    capture.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    args[7].zero_()
    capture.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    resets_counter = candidate == "256x128 persist" or pipeline.config.get("counter_reset", False)
    assert args[-1].item() == (pipeline.config["persistent_workgroups"] if resets_counter else 0x123456)
    args[4].copy_(old_ids)
    args[7].copy_(old_valid)
    pipeline.poison_workspace(*inputs)
    capture.replay()
    torch.cuda.synchronize()
    check()


@pytest.mark.parametrize("sort_block_m", [128, 256])
@pytest.mark.parametrize("n,topk", [(512, 1), (512, 8), (6144, 8)])
def test_packed_reducer(n, topk, sort_block_m):
    require_gpu()
    tokens = 19
    rows = (tokens * topk + sort_block_m - 1) // sort_block_m * sort_block_m
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn((tokens, topk, n), generator=generator, device="cuda").to(torch.bfloat16)
    permutation = torch.randperm(tokens * topk, generator=generator, device="cuda")
    sorted_values = torch.full((rows, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    sorted_values[:tokens * topk] = values.reshape(-1, n)[permutation]
    source = sorted_values.view(-1, sort_block_m, n // 64, 64).permute(0, 2, 1, 3).contiguous().view(rows, n)
    inverse = torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
    inverse.view(-1)[permutation] = torch.arange(tokens * topk, dtype=torch.int32, device="cuda")
    output = torch.full((tokens, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    reduce = make_moe_sum(n=n, topk=topk, sort_block_m=sort_block_m)
    reduce(output, source, inverse)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, values.sum(dim=1), rtol=0.01, atol=0.01)
    inverse.fill_(-1)
    reduce(output, source, inverse)
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("candidate", ["256x128 persist", "256x128", "flydsl_bn32", "flydsl_bn64", *M128_CANDIDATES])
def test_cross_k_fma_cancellation(candidate):
    require_gpu()
    tokens, topk, n = 3, 2, 1024 if candidate in M128_CANDIDATES else 512
    tolerance = 0.0 if candidate in M128_CANDIDATES else 0.01
    a = torch.zeros((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    a[..., 0] = a[..., 128] = 1
    w = torch.zeros((2, n, 256), dtype=torch.bfloat16, device="cuda")
    w[0, :, 0], w[0, :, 128], w[1, :, 0] = 16, -5, -4
    sa = torch.ones((2, tokens * topk), device="cuda")
    sb = torch.ones((2, n // 128, 2), device="cuda")
    sb[0, :, 1] = 2.3968749046325684
    sort_block_m = SORT_BLOCK_M[candidate]
    ids = torch.full((2 * sort_block_m,), (topk << 24) | tokens, dtype=torch.int32, device="cuda")
    routes = torch.zeros(2 * sort_block_m, device="cuda")
    for expert in range(2):
        ids[expert * sort_block_m:expert * sort_block_m + tokens] = torch.arange(tokens, dtype=torch.int32, device="cuda") | (expert << 24)
        routes[expert * sort_block_m:expert * sort_block_m + tokens] = 0.5
    output = torch.empty((tokens, n), dtype=torch.bfloat16, device="cuda")
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(16, 16)),
            sa, sb, ids, routes, torch.arange(2, dtype=torch.int32, device="cuda"),
            torch.tensor([2 * sort_block_m], dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda"))
    if candidate in PACKED_CANDIDATES:
        pipeline = make_packed_pipeline(candidate, n=n, topk=topk, num_experts=2)
    else:
        pipeline = with_torch_sum(legacy_flydsl_down(n=n, k=256, topk=topk, num_experts=2,
                                                    block_n=CANDIDATES[candidate][1]), DownReduceWorkspace())
    pipeline.poison_workspace(*args)
    pipeline(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=tolerance, atol=tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    pipeline.poison_workspace(*args)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 0.015625), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("persistent", [False, True])
def test_native_sorting_expert_boundaries(persistent):
    """Odd expert-block counts must not inherit a neighbouring M256 expert."""
    require_gpu()
    n, experts, topk, block_m = 1024, 4, 1, 128
    counts = torch.tensor([1, 65, 129, 257], device="cuda")
    assignments = torch.repeat_interleave(torch.arange(experts, device="cuda"), counts).to(torch.int32)[:, None]
    tokens = assignments.shape[0]
    routing = ((torch.arange(tokens, device="cuda", dtype=torch.float32) % 8 + 1) / 8)[:, None]
    a = torch.zeros((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    a[..., 0], a[..., 128] = 1, 2
    w = torch.zeros((experts, n, 256), dtype=torch.bfloat16, device="cuda")
    w[..., 0] = torch.arange(1, experts + 1, device="cuda")[:, None]
    w[..., 128] = -w[..., 0]
    sa = torch.ones((2, tokens), dtype=torch.float32, device="cuda")
    sb = torch.ones((experts, n // 128, 2), dtype=torch.float32, device="cuda")
    sb[..., 1] = 0.25
    ids, routes, eids, valid, _ = moe_sorting(assignments, routing, experts, n, torch.bfloat16, block_m, None, None, 0)
    padded_blocks = (counts + block_m - 1) // block_m
    assert int(valid[0]) == int(padded_blocks.sum()) * block_m
    assert int(valid[0]) % 256 != 0  # catches truncation in active_tasks/unpack
    torch.testing.assert_close(eids[:int(padded_blocks.sum())],
        torch.repeat_interleave(torch.arange(experts, dtype=torch.int32, device="cuda"), padded_blocks), rtol=0, atol=0)
    # A valid-looking capacity tail must still never enter the GEMM or inverse.
    ids[int(valid[0]):] = 0
    routes[int(valid[0]):] = 123
    eids[int(padded_blocks.sum()):] = experts - 1
    counter = torch.full((1,), 0x123456, dtype=torch.int32, device="cuda")
    args = (a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(16, 16)),
            sa, sb, ids, routes, eids, valid, counter)
    down = m128_kernel.make_m128_down(n=n, topk=topk, num_experts=experts, persistent=persistent)
    rows = eids.numel() * block_m
    guard = torch.full(((rows + 2) * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
    packed = guard[n:-n].view(rows, n)
    down(packed, *args)
    decoded = unpack_routes(packed, ids, valid, tokens, topk, sort_block_m=block_m)
    expected = ((assignments.float() + 1) * routing * 0.5).to(torch.bfloat16).view(tokens, topk, 1).expand(-1, -1, n)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    inverse = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
    invert_sorted_ids(topk)(ids, inverse, valid, ids.numel(), tokens)
    output = torch.empty((tokens, n), dtype=torch.bfloat16, device="cuda")
    make_moe_sum(n=n, topk=topk, sort_block_m=block_m)(output, packed, inverse)
    torch.testing.assert_close(output, expected[:, 0], rtol=0, atol=0)
    assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
    assert counter.item() == 0x123456
    assert down.config["sort_block_m"] == down.config["packed_rows"] == block_m


@pytest.mark.parametrize("n", [1024, 2048, 3072, 5120, 6144, 8192])
def test_m128_self_reset_lifecycle(n):
    """Private heads close every launch; only public buffers are poisoned."""
    require_gpu()
    down = make_m128_persistent_down(n=n, topk=2, num_experts=4)
    pointer = None
    # Reuse one callable across shapes, including more tasks than workers.
    for tokens in ((1, 4097, 129) if n == 1024 else (1, 257)):
        args, expected = make_case(tokens, n, 4, 2, 2070 + tokens, sort_block_m=128)
        rows = args[6].numel() * 128
        guard = torch.full(((rows + 2) * n,), torch.nan, dtype=torch.bfloat16, device="cuda")
        packed = guard[n:-n].view(rows, n)
        args[-1].fill_(0x123456)
        down(packed, *args)
        heads = down.workspace["heads"]
        if pointer is None:
            pointer = heads.data_ptr()

        def check(real=True):
            torch.cuda.synchronize()
            assert heads.data_ptr() == pointer and heads.count_nonzero().item() == 0
            assert args[-1].item() == 0x123456
            assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
            if real:
                actual = unpack_routes(packed, args[4], args[7], tokens, 2, sort_block_m=128)
                assert error_stats(actual, expected)["status"] == "PASS"
            else:
                assert torch.isnan(packed).all()

        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Two launches in one graph, without any host-side head reset.
            down(packed, *args)
            down(packed, *args)
        for _ in range(3):
            packed.fill_(torch.nan)
            args[-1].fill_(0x123456)
            graph.replay()
        check()
        old_ids, old_valid = args[4].clone(), args[7].clone()
        args[4].fill_((2 << 24) | tokens)
        packed.fill_(torch.nan)
        graph.replay()
        check(False)
        args[7].zero_()
        graph.replay()
        check(False)
        args[4].copy_(old_ids)
        args[7].copy_(old_valid)
        packed.fill_(torch.nan)
        graph.replay()
        check()
        assert down.config["queue_self_reset"]


def test_m128_self_reset_interleavings():
    """Last terminal claim implies no future claim, not a grid barrier."""
    import random

    for blocks in (0, 1, 7, 63, 64, 65, 129, 1157):
        for seed in range(8):
            rng = random.Random(seed)
            head, waiting, pending, seen = 0, list(range(64)), {}, []
            reset = False
            while waiting or pending:
                if waiting and (not pending or rng.randrange(2)):
                    assert not reset
                    worker = waiting.pop(rng.randrange(len(waiting)))
                    rank, head = head, head + 1
                    pending[worker] = rank
                else:
                    worker = rng.choice(list(pending))
                    rank = pending.pop(worker)
                    if rank < blocks:
                        seen.append(rank)
                        waiting.append(worker)
                    elif rank == blocks + 63:
                        assert not waiting and all(value >= blocks for value in pending.values())
                        head, reset = 0, True
            assert reset and head == 0 and sorted(seen) == list(range(blocks))


@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("reduce_output", [False, True])
def test_down_and_total_metrics(n, reduce_output, capsys):
    require_gpu()
    results = run_test(tokens=513, model_dim=n, experts=4, topk=2, reduce_output=reduce_output)
    printed = capsys.readouterr().out
    expected_candidates = [key for key in DEFAULT_CANDIDATES if n % 1024 == 0 or key not in M128_CANDIDATES]
    assert [r["candidate"] for r in results] == expected_candidates
    assert all(r["status"] == "PASS" for r in results)
    for row in results:
        assert row["elapsed_us"] > 0 and row["down_elapsed_us"] > 0
        assert row["unique_experts"] == 4 and row["valid_expert_blocks"] > 4
        assert row["rw_bytes"] == row["down_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["ideal_rw_bytes"] == row["down_ideal_rw_bytes"] + row["reduce_rw_bytes"] + row["inverse_rw_bytes"]
        assert row["down_effective_tflops"] == f"{2 * 513 * 2 * n * 256 / row['down_elapsed_us'] / 1e6:.3f}"
        assert row["compute_rows"] == row["work_tasks"] * row["config"]["block_m"] >= 513 * 2
        assert row["sort_block_m"] == SORT_BLOCK_M[row["candidate"]]
        assert row["valid_padded_rows"] == row["valid_expert_blocks"] * row["sort_block_m"]
        if row["candidate"] in M128_CANDIDATES:
            assert row["config"]["packed_rows"] == 128
            assert row["valid_padded_rows"] == row["compute_rows"]
        assert row["down_padded_tflops"] == f"{2 * row['compute_rows'] * n * 256 / row['down_elapsed_us'] / 1e6:.3f}"
        assert row["down_rw_bytes"] == 513 * 2 * 256 + row["work_tasks"] * n * 256 + 513 * 2 * n * 2
        if reduce_output:
            assert row["down_elapsed_us"] == row["components_us"]["gemm"]
            assert row["reduce_tb_per_s"] == f"{row['reduce_rw_bytes'] / row['components_us']['reduce'] / 1e6:.3f}"
            assert bool(row["inverse_rw_bytes"]) == (row["candidate"] in PACKED_CANDIDATES)
            assert "Total Time (us)" in printed
        else:
            assert row["elapsed_us"] == row["down_elapsed_us"] and row["reduce_rw_bytes"] == row["inverse_rw_bytes"] == 0
    assert results[-1]["config"]["output_layout"] == "packed"
    if n % 1024 == 0:
        independent = next(r for r in results if r["candidate"] == "128x128")
        config = independent["config"]
        assert all(config[key] == value for key, value in m128_kernel.M128_CONTINUOUS_VMEM_CONFIG.items())
        assert config["num_waves"] == 4 and not config["persistent"]
        assert independent["compute_rows"] <= next(r for r in results if r["candidate"] == "256x128 persist")["compute_rows"]
    else:
        assert "SKIP 128x128" in printed
    assert "FlyDSL BN32" in printed and "FlyDSL BN64" in printed and "speedup vs FlyDSL BN64" in printed
    assert "Mean abs error" in printed and "calc_diff" in printed
    for row in results[1:3]:
        assert row["config"]["output_layout"] == "routed"
        assert row["config"]["block_n"] == (32 if row["candidate"] == "flydsl_bn32" else 64)
        if reduce_output:
            assert row["config"]["reduction"] == "torch" and not row["config"]["includes_inverse"]


def test_m128_selection_and_import(capsys):
    assert select_candidates(6144) == list(DEFAULT_CANDIDATES)
    assert select_candidates(512) == [key for key in DEFAULT_CANDIDATES if key not in M128_CANDIDATES]
    printed = capsys.readouterr().out
    assert select_candidates(1024, PACKED_CANDIDATES) == list(PACKED_CANDIDATES)
    for candidate in M128_CANDIDATES:
        assert f"SKIP {candidate}:" in printed
        assert select_candidates(1024, [candidate]) == [candidate]
        with pytest.raises(ValueError, match="OC8"):
            select_candidates(512, [candidate])
    paths = list(sys.path)
    main_down = sys.modules["moe_multistage_down"]
    main_pipeline = sys.modules["moe_multistage_pipeline"]
    module = m128_kernel
    assert sys.path == paths
    assert Path(module.__file__).resolve() == Path(__file__).resolve().parent / "moe_multistage_down_m128.py"
    assert sys.modules["moe_multistage_down"] is main_down
    assert sys.modules["moe_multistage_pipeline"] is main_pipeline
    assert main_pipeline.make_moe_sum is make_moe_sum


def test_live_m_blocks_ignore_padding_and_capacity():
    ids = torch.full((640,), (130 << 24) | 3, dtype=torch.int64, device="cpu")
    ids[127], ids[255], ids[511] = (129 << 24) | 1, (128 << 24) | 2, 2
    ids[256], ids[383], ids[639] = 130 << 24, 3, 1
    ids = ids.to(torch.int32)
    assert count_live_m_blocks(ids, 512, tokens=3, topk=130, block_m=128) == 3
    assert count_live_m_blocks(ids, 0, tokens=3, topk=130, block_m=128) == 0


@pytest.mark.parametrize("kwargs", [{"n": 384}, {"n": 512, "k": 384},
    {"n": 512, "num_oc_splits": 1}, {"n": 512, "xcd_swizzle": True}])
def test_reject_nonwinner_configuration(kwargs):
    with pytest.raises((AssertionError, TypeError)):
        flydsl_moe_gemm_8wave_down(topk=1, num_experts=1, **kwargs)


@pytest.mark.parametrize("argv,scope", [([], True), (["--mode", "down"], False),
    (["--candidate", "flydsl_bn32", "flydsl_bn64"], True),
    (["--mode", "down", "--candidate", "flydsl_bn32", "flydsl_bn64"], False)]
    + [([*options, "--candidate", candidate], scope) for candidate in PACKED_CANDIDATES
       for options, scope in (([], True), (["--mode", "down"], False),
                              (["--profile"], False), (["--profile", "--mode", "down-reduce"], True))])
def test_cli_scope(monkeypatch, argv, scope):
    module = sys.modules[__name__]
    calls = []
    monkeypatch.setattr(sys, "argv", [__file__, *argv])
    monkeypatch.setattr(module, "run_test", lambda **kw: calls.append(kw))
    main()
    assert len(calls) == 1 and calls[0]["reduce_output"] is scope
    expected = argv[argv.index("--candidate") + 1:] if "--candidate" in argv else list(DEFAULT_CANDIDATES)
    assert calls[0]["candidates"] == expected and calls[0]["profile"] == ("--profile" in argv)


@pytest.mark.parametrize("candidate", M128_CANDIDATES)
def test_cli_rejects_unsupported_m128(monkeypatch, capsys, candidate):
    monkeypatch.setattr(sys, "argv", [__file__, "--model-dim", "512", "--candidate", candidate])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and "OC8" in capsys.readouterr().err


@pytest.mark.parametrize("candidate", ["256x128 persist", "256x128"])
def test_m256_packed_crosses_4gib(candidate):
    """Exercise live stores below/above 4 GiB, not just an oversized tail."""
    require_gpu()
    tokens, topk, n, experts, block_m = 2, 2, 6144, 4, 256
    block_bytes = block_m * n * 2
    crossing = (1 << 32) // block_bytes
    blocks, valid_rows = crossing + 3, (crossing + 2) * block_m
    rows = blocks * block_m
    assert crossing * block_bytes < (1 << 32) < (crossing + 1) * block_bytes
    locations = [[0, (crossing + 1) * block_m + 255],
                 [crossing * block_m + 127, (crossing + 1) * block_m + 128]]
    a = torch.zeros((tokens, topk, 256), dtype=torch.bfloat16, device="cuda")
    a[..., 0], a[..., 128] = 1, 2
    pattern = (torch.arange(n, device="cuda", dtype=torch.float32) % 16 + 1) / 8
    w = torch.zeros((experts, n, 256), dtype=torch.bfloat16, device="cuda")
    w[..., 0], w[..., 128] = pattern, -pattern
    sa = torch.ones((2, tokens * topk), device="cuda")
    sb = torch.ones((experts, n // 128, 2), device="cuda")
    sb[:, :, 0] = torch.arange(1, experts + 1, device="cuda")[:, None]
    sb[:, :, 1] = sb[:, :, 0] * 0.25
    routing = torch.tensor([[0.25, 0.75], [0.5, 0.5]], device="cuda")
    expected_routes = (pattern[None, None, :] * 0.5
                       * torch.tensor([1, 2], device="cuda")[None, :, None]
                       * routing[:, :, None]).to(torch.bfloat16)
    expected = expected_routes.sum(dim=1)
    sentinel = (topk << 24) | tokens
    ids = torch.full((rows,), sentinel, dtype=torch.int32, device="cuda")
    routes = torch.zeros(rows, device="cuda")
    eids = torch.zeros(blocks, dtype=torch.int32, device="cuda")
    eids[crossing + 1:] = 1
    for token in range(tokens):
        for slot in range(topk):
            ids[locations[token][slot]] = (slot << 24) | token
            routes[locations[token][slot]] = routing[token, slot]
    # Native metadata is tested separately. This sparse synthetic prefix places
    # real rows at the addressing boundary; its capacity tail is hostile.
    ids[valid_rows:], routes[valid_rows:] = 0, 123
    valid = torch.tensor([valid_rows], dtype=torch.int32, device="cuda")
    counter = torch.full((1,), 0x123456, dtype=torch.int32, device="cuda")
    source_guard = torch.full((rows + 2, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    output_guard = torch.full((tokens + 2, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    output, packed = output_guard[1:-1], source_guard[1:-1]
    workspace = DownReduceWorkspace()
    workspace.data = packed
    pipeline = make_packed_pipeline(candidate, n=n, topk=topk, num_experts=experts, workspace=workspace)
    args = (output, a.to(torch.float8_e4m3fn), shuffle_weight(w.to(torch.float8_e4m3fn), layout=(16, 16)),
            sa, sb, ids, routes, eids, valid, counter)
    physical = packed.view(blocks, n // 64, block_m, 64)
    expected_inverse = torch.tensor(locations, dtype=torch.int32, device="cuda")

    def check(live):
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected if live else torch.zeros_like(expected), rtol=0, atol=0)
        assert torch.isnan(source_guard[[0, -1]]).all() and torch.isnan(output_guard[[0, -1]]).all()
        assert torch.isnan(packed[valid_rows:]).all()
        assert workspace.data.data_ptr() == packed.data_ptr()
        expected_counter = int(valid[0].item()) // block_m * 4 + 256 if candidate == "256x128 persist" else 0x123456
        assert counter.item() == expected_counter
        if live:
            torch.testing.assert_close(workspace.inverse, expected_inverse, rtol=0, atol=0)
            for token in range(tokens):
                for slot in range(topk):
                    bm, row = divmod(locations[token][slot], block_m)
                    torch.testing.assert_close(physical[bm, :, row, :].reshape(n),
                                               expected_routes[token, slot], rtol=0, atol=0)
        else:
            assert (workspace.inverse == -1).all()
        for bm in (0, crossing, crossing + 1):
            keep = torch.ones(block_m, dtype=torch.bool, device="cuda")
            if live:
                for row in (location % block_m for route in locations for location in route
                            if location // block_m == bm):
                    keep[row] = False
            assert torch.isnan(physical[bm, :, keep, :]).all()

    pipeline(*args)
    check(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pipeline(*args)
    old_ids = ids.clone()
    for state in ("valid", "invalid", "empty", "restore"):
        if state == "invalid":
            ids[:valid_rows].fill_(sentinel)
        elif state == "empty":
            valid.zero_()
        elif state == "restore":
            ids.copy_(old_ids)
            valid.fill_(valid_rows)
        packed.fill_(torch.nan)
        output.fill_(torch.nan)
        counter.fill_(0x123456)
        graph.replay()
        check(state in ("valid", "restore"))


@pytest.mark.parametrize("sort_block_m", [128, 256])
def test_packed_reducer_crosses_4gib(sort_block_m):
    """Wide source gathers, partial column CTA, invalid indices and replay."""
    require_gpu()
    tokens, n, topk = 2, 2560, 8
    block_bytes = sort_block_m * n * 2
    crossing = (1 << 32) // block_bytes
    rows = (crossing + 2) * sort_block_m
    assert crossing * block_bytes < (1 << 32) < (crossing + 1) * block_bytes
    locations = [[0, crossing * sort_block_m + sort_block_m // 2, rows - 1,
                  -1, -2, rows, (1 << 31) - 1, -(1 << 31)],
                 [1, crossing * sort_block_m + 1, (crossing + 1) * sort_block_m,
                  -1, -2, rows + 1, (1 << 31) - 1, -(1 << 31)]]
    generator = torch.Generator(device="cuda").manual_seed(2091)
    values = torch.randn((tokens, topk, n), generator=generator, device="cuda").to(torch.bfloat16)
    values[:, 3:] = 0
    source_guard = torch.full((rows + 2, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    source = source_guard[1:-1]
    physical = source.view(-1, n // 64, sort_block_m, 64)
    for token in range(tokens):
        for slot in range(3):
            bm, row = divmod(locations[token][slot], sort_block_m)
            physical[bm, :, row, :].copy_(values[token, slot].view(n // 64, 64))
    expected = torch.zeros((tokens, n), device="cuda")
    for slot in range(topk):
        expected += values[:, slot].float()
    expected = expected.to(torch.bfloat16)
    inverse = torch.tensor(locations, dtype=torch.int32, device="cuda")
    output_guard = torch.full((tokens + 2, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = output_guard[1:-1]
    reduce = make_moe_sum(n=n, topk=topk, sort_block_m=sort_block_m)

    def check(live):
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected if live else torch.zeros_like(expected), rtol=0, atol=0)
        assert torch.isnan(source_guard[[0, -1]]).all() and torch.isnan(output_guard[[0, -1]]).all()

    reduce(output, source, inverse)
    check(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduce(output, source, inverse)
    original = inverse.clone()
    for live in (True, False, True):
        inverse.copy_(original) if live else inverse.fill_(-1)
        output.fill_(torch.nan)
        graph.replay()
        check(live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--model-dim", type=int, default=6144)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--candidate", choices=DEFAULT_CANDIDATES, nargs="+", default=None)
    parser.add_argument("--mode", choices=("down-reduce", "down"), default=None)
    parser.add_argument("--profile", action="store_true", help="23 direct launches; defaults to packed down-only")
    args = parser.parse_args()
    if args.profile and (args.candidate is None or len(args.candidate) != 1):
        parser.error("--profile requires exactly one --candidate")
    try:
        candidates = select_candidates(args.model_dim, args.candidate)
    except ValueError as error:
        parser.error(str(error))
    mode = args.mode or ("down" if args.profile else "down-reduce")
    run_test(tokens=args.tokens, model_dim=args.model_dim, experts=args.experts,
             topk=args.topk, seed=args.seed, candidates=candidates,
             profile=args.profile, reduce_output=mode == "down-reduce")


if __name__ == "__main__":
    main()