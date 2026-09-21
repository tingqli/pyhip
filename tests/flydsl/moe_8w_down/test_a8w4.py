# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4: Aiter, original eight-wave and optimized four-wave comparison.

Run directly for Markdown Down/Full tables; use pytest for regressions.
Native sorting per candidate, shared routed storage and preallocated sum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import (
    pick_flydsl_stage2_tile_k,
    resolve_flydsl_stage2_tile_k,
)
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f8_scale_f8_quant,
)

import pyhip
from a8w4_test_utils import (
    Q_TYPE,
    _inter_pad,
    _generate_a8w4_gui_data,
    error_stats,
    check_exact,
    print_markdown_table,
    sorted_metadata,
)
from moe_8wave_down_a8w4 import flydsl_moe_gemm_8wave_down_a8w4
from moe_4wave_down_a8w4 import flydsl_moe_gemm_4wave_down_a8w4, select_config
from moe_8wave_down_a8w4_optimized import flydsl_moe_gemm_8wave_down_a8w4_optimized
from moe_multistage_reduce import make_moe_sum

CANDIDATES = {
    'aiter': 'Aiter FlyDSL',
    '8wave': '8-wave original',
    '8wave_oc4': '8-wave OC4',
    '8wave_opt_control': '8-wave opt control',
    '8wave_opt_dpp': '8-wave DPP aux0',
    '8wave_opt_ntsc1': '8-wave DPP NT+SC1',
    '8wave_opt_dpp_overlap': '8-wave DPP overlap aux0',
    '8wave_optimized': '8-wave optimized',
    '8wave_opt_oc1': '8-wave optimized OC1',
    '8wave_opt_bn64': '8-wave optimized BN64',
    '4wave': '4-wave optimized',
    '4wave_bn64': '4-wave BN64',
    '4wave_bn128': '4-wave BN128',
    '4wave_sharded': '4-wave sharded',
    'opus_bf16': 'Opus BF16 routes',
    'opus_fp8': 'Opus FP8 routes',
}
DEFAULT_CANDIDATES = ('aiter', '8wave', '4wave_bn64', '4wave_bn128', '8wave_optimized')

# Each edge isolates one factor; the control includes Tensor DMA migration.
OPT8_OPTIONS = {
    '8wave_opt_control': dict(coalesce_output=False, output_cache_policy=0, compute_overlap=False),
    '8wave_opt_dpp': dict(coalesce_output=True, output_cache_policy=0, compute_overlap=False),
    '8wave_opt_ntsc1': dict(coalesce_output=True, output_cache_policy=18, compute_overlap=False),
    '8wave_opt_dpp_overlap': dict(coalesce_output=True, output_cache_policy=0, compute_overlap=True),
    '8wave_optimized': dict(coalesce_output=True, output_cache_policy=18, compute_overlap=True),
    '8wave_opt_oc1': dict(coalesce_output=True, output_cache_policy=18, compute_overlap=True, num_oc_splits=1),
    '8wave_opt_bn64': dict(coalesce_output=True, output_cache_policy=18, compute_overlap=False, block_n=64),
}

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() not in ("gfx950",),
    reason="gfx950 FlyDSL required",
)


def _stage1_tile_k(model_dim: int) -> int:
    return 512 if (model_dim % 512 == 0) else 256


def _check_close(ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05):
    stats = error_stats(out, ref, atol=atol, rtol=rtol, max_err_ratio=max_err_ratio)
    assert stats['status'] == 'PASS', f'{label}: {stats}'
    return stats['error_ratio']


@pytest.fixture(autouse=True)
def _a8w4_env():
    old_bound = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    old_aot = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    yield
    if old_bound is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old_bound
    if old_aot is None:
        os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    else:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = old_aot


def test_pick_flydsl_stage2_tile_k():
    assert pick_flydsl_stage2_tile_k(256) == 256
    assert pick_flydsl_stage2_tile_k(512) == 256
    assert pick_flydsl_stage2_tile_k(640) == 128
    assert pick_flydsl_stage2_tile_k(384) == 128
    assert pick_flydsl_stage2_tile_k(896) == 128
    assert pick_flydsl_stage2_tile_k(1024) == 256
    assert resolve_flydsl_stage2_tile_k(640, 256) == 128
    assert resolve_flydsl_stage2_tile_k(256, 256) == 256
    assert resolve_flydsl_stage2_tile_k(512, 128) == 128


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a8w4_gui(seed=1234):
    run_test(tokens=65, model_dim=3072, experts=4, topk=2, seed=seed, rounds=1, iters=2)


@pytest.mark.parametrize("inter_dim", [256, 384, 640])
@_SKIP_GFX950_FLYDSL
def test_flydsl_e2e_a8w4_gui(inter_dim):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, E, topk, block_m, seed = 16, 512, 8, 2, 32, 0
    data = _generate_a8w4_gui_data(token, model_dim, inter_dim, E, topk, block_m, seed=seed)
    stage1_out = flydsl_moe_stage1(
        a=data["a_q"], w1=data["w1_shuf"], sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"], num_valid_ids=data["num_valid_ids"],
        topk=topk, tile_m=32, tile_n=256, tile_k=_stage1_tile_k(model_dim),
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", act="swiglu", gate_mode="interleave",
        w1_scale=data["w1_scale_shuf"], a1_scale=data["a_scale_sort"],
        inter_dim_pad=data["inter_pad"], model_dim_pad=0)
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(stage1_out, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    a2_q = a2_q.view(token, topk, inter_dim)
    a2_scale_sort = mxfp4_moe_sort_fwd(a2_scale, sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"], token_num=token, cols=inter_dim)
    out = flydsl_moe_stage2(inter_states=a2_q, w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"], sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"], topk=topk, tile_m=32, tile_n=256, tile_k=256,
        a_dtype="fp8", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=data["w2_scale_shuf"], a2_scale=a2_scale_sort, sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"], model_dim_pad=0)
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, f"e2e_a8w4_gui_i{inter_dim}")


def build_local(key, *, tokens, n, k, topk, num_experts):
    shape = dict(n=n, k=k, topk=topk, num_experts=num_experts)
    if key in OPT8_OPTIONS:
        return flydsl_moe_gemm_8wave_down_a8w4_optimized(**shape, **OPT8_OPTIONS[key])
    if key.startswith('4wave'):
        config = select_config(tokens=tokens, n=n, k=k, topk=topk)
        if key == '4wave_bn64':
            config['block_n'] = 64
        elif key == '4wave_bn128':
            config['block_n'] = 128
        if key == '4wave_sharded':
            config['sharded'] = True
        return flydsl_moe_gemm_4wave_down_a8w4(**shape, **config)
    oc = (1 if tokens >= 16384 else 4) if key == '8wave' else 4
    kernel = flydsl_moe_gemm_8wave_down_a8w4(**shape, block_m=256, block_n=128, num_oc_splits=oc)
    kernel.config = dict(block_m=256, sort_block_m=256, block_n=128, num_waves=8, num_oc_splits=oc,
                         workers=256, counter_elements=1, output_layout='routed', row_store_bytes=64,
                         sharded=False, output_cache_policy=0)
    return kernel


def run_test(tokens=16384, model_dim=6144, inter_dim=256, experts=384, topk=8, seed=1234,
             candidates=None, profile=False, reduce_output=True, rounds=5, iters=20, graph_check=True,
             reducer='moe', compare_reduce=False):
    assert torch.cuda.is_available() and torch.cuda.get_device_properties().gcnArchName.startswith('gfx950')
    assert os.environ.get('PYHIP_FLYDSL_NOP_MFMA') != '1', 'correctness requires real MFMA'
    selected = list(DEFAULT_CANDIDATES if candidates is None else candidates)
    if not selected or len(set(selected)) != len(selected) or any(key not in CANDIDATES for key in selected):
        raise ValueError('select distinct known candidates')
    if profile and len(selected) != 1:
        raise ValueError('profiling requires one candidate')
    if any(key.startswith('opus') for key in selected) and (inter_dim != 384 or not reduce_output):
        raise ValueError('retained Opus instances are Full-only K384 comparisons')
    if rounds <= 0 or iters <= 0:
        raise ValueError('rounds and iters must be positive')
    if reducer not in ('moe', 'torch') or (compare_reduce and (profile or not reduce_output)):
        raise ValueError('compare_reduce requires timed down-reduce mode; reducer must be moe/torch')
    data = _generate_a8w4_gui_data(tokens, model_dim, inter_dim, experts, topk, 256,
                                   seed=seed, inter_pad_override=0)
    metadata = {256: (data['sorted_ids'], data['sorted_weights'], data['sorted_expert_ids'],
                      data['num_valid_ids'], data['a2_scale_sort'])}
    guard = torch.full((tokens * topk * model_dim + 2 * model_dim,), torch.nan, dtype=torch.bfloat16, device='cuda')
    middle = guard[model_dim:-model_dim].view(tokens, topk, model_dim)
    output = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device='cuda')
    moe_sum = make_moe_sum(n=model_dim, topk=topk, source_layout='routed') if reducer == 'moe' or compare_reduce else None

    def torch_reduce():
        torch.sum(middle, dim=1, out=output)

    def moe_reduce():
        moe_sum(output, middle)

    reducers = {'torch': torch_reduce, 'moe': moe_reduce}
    reduce_bytes = (middle.numel() + output.numel()) * 2
    counters = torch.zeros(256, dtype=torch.int32, device='cuda')
    shape = dict(tokens=tokens, n=model_dim, k=inter_dim, topk=topk, num_experts=experts)

    def inputs(bm, sharded=False):
        if bm not in metadata:
            metadata[bm] = sorted_metadata(data, bm, experts)
        ids, weights, eids, valid, scales = metadata[bm]
        return (middle, data['a2_q'], data['w2_shuf'], scales, data['w2_scale_shuf'],
                ids, weights, eids, valid, counters if sharded else counters[:1])

    reference_kernel = build_local('8wave_oc4', **shape)
    reference_kernel(*inputs(256))
    torch.cuda.synchronize()
    assert torch.isfinite(middle).all()
    reference_routes = middle.clone()
    reference_full = reference_routes.sum(dim=1)
    _check_close(data['ref_stage2'], reference_full, 'original baseline vs model')
    records, launches, reduction_launches = [], {}, {}
    for key in selected:
        config = dict(output_layout='routed', sharded=False, output_cache_policy='N/A',
                      num_oc_splits='N/A', row_store_bytes='N/A', workers='N/A', num_waves='N/A')
        full_only = key.startswith('opus')
        if key == 'aiter':
            from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
            args = inputs(64)
            config.update(block_m=64, block_n=256, sort_block_m=64)
            def down(args=args):
                return flydsl_moe_stage2(inter_states=args[1], w2=args[2], a2_scale=args[3], w2_scale=args[4],
                    sorted_token_ids=args[5], sorted_weights=args[6], sorted_expert_ids=args[7],
                    num_valid_ids=args[8], out=middle, topk=topk, tile_m=64, tile_n=256, tile_k=256,
                    a_dtype='fp8', b_dtype='fp4', out_dtype='bf16', mode='reduce', return_per_slot=True,
                    inter_dim_pad=0, model_dim_pad=0)
        elif full_only:
            from aiter.ops.opus import moe_stage2_a8w4 as opus
            from csrc.opus_moe.opus_moe_common import (
                OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL,
                OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072, require_opus_a8w4_stage2_instance)
            kid = OPUS_A8W4_KID_ROUTE_BF16_BM32_FULL_N7168_SMALL if key == 'opus_bf16' else OPUS_A8W4_KID_ROUTE_FP8_BM64_RBN3072
            instance, launch_config = require_opus_a8w4_stage2_instance(kid), opus.stage2_launch_config(kid)
            bm = 32 if key == 'opus_bf16' else 64
            args = inputs(bm)
            config.update(block_m=bm, sort_block_m=bm, block_n='N/A', output_layout='internal')
            def full(args=args, instance=instance, launch_config=launch_config, bm=bm):
                return opus.opus_a8w4_stage2_wrapper(inter_states=args[1], w1=None, w2=args[2],
                    a2_scale=args[3], w2_scale=args[4], sorted_token_ids=args[5], sorted_weights=args[6],
                    sorted_expert_ids=args[7], num_valid_ids=args[8], out=output, topk=topk,
                    launch=launch_config, kernelName=instance.name, block_m=bm, inter_dim_pad=0, model_dim_pad=0)
            down = None
        else:
            kernel = build_local(key, **shape)
            config = dict(kernel.config)
            args = inputs(config['block_m'], config['sharded'])
            def down(kernel=kernel, args=args):
                return kernel(*args)  # factory owns the single reset; no duplicate reset
        if not full_only:
            def full(down=down):
                down()
                reducers[reducer]()
                return output
            config['reduction'] = reducer
            if compare_reduce:
                def torch_full(down=down):
                    down()
                    torch_reduce()
                def moe_full(down=down):
                    down()
                    moe_reduce()
                reduction_launches[key] = {'torch': torch_full, 'moe': moe_full}
        exact = key.startswith(('4wave', '8wave'))
        allowance = 0.1 if key == 'opus_fp8' else 0.05

        def validate(exact=exact, full_only=full_only, allowance=allowance, key=key):
            torch.cuda.synchronize()
            assert torch.isnan(guard[:model_dim]).all() and torch.isnan(guard[-model_dim:]).all(), key
            if exact:
                check_exact(middle, reference_routes)
                check_exact(output, reference_full)
            elif not full_only:
                _check_close(reference_routes, middle, key + ' routes')
            _check_close(data['ref_stage2'], output, key + ' vs model', max_err_ratio=allowance)

        middle.fill_(torch.nan)
        output.fill_(torch.nan)
        full()
        validate()
        if graph_check:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                full()
                full()
            for _ in range(2):
                middle.fill_(torch.nan)
                output.fill_(torch.nan)
                counters.fill_(123)
                graph.replay()
                validate()
        if compare_reduce and not full_only:
            for kind, comparison in reduction_launches[key].items():
                comparison()
                expected_sum = middle.sum(dim=1)
                check_exact(output, expected_sum)
                if graph_check:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        comparison()
                        comparison()
                    output.fill_(torch.nan)
                    counters.fill_(123)
                    graph.replay()
                    check_exact(output, expected_sum)
        errors = error_stats(output, data['ref_stage2'], max_err_ratio=allowance)
        valid_rows = int(args[8][0].item())
        record = dict(candidate=key, name=CANDIDATES[key], config=config, exact_routes=exact,
                      graph_replay=graph_check, valid_padded_rows=valid_rows,
                      valid_expert_blocks=valid_rows // config['block_m'],
                      samples={'down': [], 'full': []}, **errors)
        records.append(record)
        if compare_reduce and not full_only:
            record['reduction_comparison'] = {kind: {'full_samples': [], 'reduce_samples': []} for kind in ('torch', 'moe')}
        launches[key] = down, full
        print(f'CHECK {key}: {errors["status"]}, exact_routes={exact}, graph={graph_check}', flush=True)

    for r in range(1 if profile else rounds):
        for record in (records if r % 2 == 0 else list(reversed(records))):
            key = record['candidate']
            down, full = launches[key]
            scopes = [('down', down)] if not reduce_output else [('down', down), ('full', full)]
            if profile:
                launch = full if reduce_output else down
                for _ in range(23):
                    launch()
                    torch.cuda.synchronize()
            else:
                if compare_reduce and down is not None:
                    _, us = pyhip.run_perftest(down, num_warmup=2, num_iters=iters, num_copies=1, num_verbose=0)
                    record['samples']['down'].append(us)
                    for kind in (('torch', 'moe') if r % 2 == 0 else ('moe', 'torch')):
                        entry = record['reduction_comparison'][kind]
                        _, us = pyhip.run_perftest(reduction_launches[key][kind], num_warmup=2,
                            num_iters=iters, num_copies=1, num_verbose=0)
                        entry['full_samples'].append(us)
                        if kind == reducer:
                            record['samples']['full'].append(us)
                        down()  # same producer before each warm consumer measurement
                        _, us = pyhip.run_perftest(reducers[kind], num_warmup=2, num_iters=iters,
                            num_copies=1, num_verbose=0)
                        entry['reduce_samples'].append(us)
                    continue
                for scope, launch in scopes:
                    if launch is not None:
                        _, us = pyhip.run_perftest(launch, num_warmup=2, num_iters=iters,
                                                   num_copies=1, num_verbose=0, num_name=f'{key}_{scope}')
                        record['samples'][scope].append(us)
    for record in records:
        record['median_us'] = {scope: statistics.median(samples) if samples else None
                               for scope, samples in record['samples'].items()}
        down, full = launches[record['candidate']]
        if 'reduction_comparison' in record:
            for entry in record['reduction_comparison'].values():
                entry['full_us'] = statistics.median(entry['full_samples'])
                entry['reduce_us'] = statistics.median(entry['reduce_samples'])
                entry['logical_tb_s'] = reduce_bytes / entry['reduce_us'] / 1e6
            record['warm_reduce_us'] = record['reduction_comparison'][reducer]['reduce_us']
        elif not profile and down is not None and reduce_output:
            down()
            _, us = pyhip.run_perftest(reducers[reducer],
                num_warmup=2, num_iters=iters, num_copies=1, num_verbose=0)
            record['warm_reduce_us'] = us
        else:
            record['warm_reduce_us'] = None
        record['reduce_bytes'] = reduce_bytes if down is not None and reduce_output else 0
        record['reduce_tb_s'] = reduce_bytes / record['warm_reduce_us'] / 1e6 if record['warm_reduce_us'] else None
        flops = 2 * tokens * topk * model_dim * inter_dim
        padded_flops = 2 * record['valid_padded_rows'] * model_dim * inter_dim
        record['effective_tflops'] = {s: flops / us / 1e6 if us else None for s, us in record['median_us'].items()}
        record['padded_tflops'] = {s: padded_flops / us / 1e6 if us else None for s, us in record['median_us'].items()}

    def number(value):
        return f'{value:.3f}' if value is not None else 'N/A'
    print(f'\nShape: tokens={tokens}, N={model_dim}, K={inter_dim}, E={experts}, TOPK={topk}; {torch.cuda.get_device_name()}')
    print('Same A/B, routed/final addresses; native sorting and quantization outside timing; reset included once.')
    print(f'Aiter uses return_per_slot=True; shared reducer={reducer}. Opus uses its own Full-only reducer.')
    print('Full is directly measured; warm reduce is supplementary, not the post-Down cache-state latency.')
    print('Local routes/Full: exact vs original. Model: atol1/rtol0.05, <=5% outliers (Opus FP8 <=10%).')
    rows = [[title, *(r['config'].get(field, 'N/A') for r in records)] for title, field in (
        ('Block M / sorting', 'block_m'), ('Block N', 'block_n'), ('Waves per CTA', 'num_waves'),
        ('OC splits', 'num_oc_splits'), ('Workers', 'workers'), ('Sharded queue', 'sharded'),
        ('Output layout', 'output_layout'), ('Bytes/row/store', 'row_store_bytes'), ('Output aux', 'output_cache_policy'))]
    rows.append(['Compute/pack overlap', *(r['config'].get('compute_overlap', False) for r in records)])
    for title, field in (('Status', 'status'), ('Exact routes', 'exact_routes'), ('Graph replay', 'graph_replay'),
                         ('Sorted padded rows', 'valid_padded_rows'), ('Valid expert blocks', 'valid_expert_blocks')):
        rows.append([title, *(r[field] for r in records)])
    for scope in ('down', 'full') if reduce_output else ('down',):
        rows.append([scope.title() + ' time (us)', *(number(r['median_us'][scope]) for r in records)])
        rows.append([scope.title() + ' effective TF/s', *(number(r['effective_tflops'][scope]) for r in records)])
        rows.append([scope.title() + ' padded TF/s', *(number(r['padded_tflops'][scope]) for r in records)])
        for base_key in ('aiter', '8wave', '8wave_oc4'):
            base = next((r['median_us'][scope] for r in records if r['candidate'] == base_key), None)
            if base is not None:
                rows.append([f'{scope.title()} speedup vs {base_key}', *(
                    f'{base / r["median_us"][scope]:.3f}x' if r['median_us'][scope] else 'N/A' for r in records)])
    if reduce_output:
        rows.append(['Warm reduce time (us)', *(number(r['warm_reduce_us']) for r in records)])
        rows.append(['Reduce logical TB/s', *(number(r['reduce_tb_s']) for r in records)])
    if compare_reduce:
        print('Paired reducer comparison: same addresses/Down; logical bytes=(routes+output)*2, NOT HBM counters.')
        for kind in ('torch', 'moe'):
            for title, field in (('Full us', 'full_us'), ('Reduce us', 'reduce_us'), ('Reduce TB/s', 'logical_tb_s')):
                rows.append([f'{kind} {title}', *(number(r.get('reduction_comparison', {}).get(kind, {}).get(field)) for r in records)])
    for title, field in (('Max abs error', 'max_abs'), ('Mean abs error', 'mean_abs'), ('calc_diff', 'diff'), ('Model outlier ratio', 'error_ratio')):
        rows.append([title, *(f'{r[field]:.6g}' for r in records)])
    print_markdown_table(['Metric', *(r['name'] for r in records)], rows)
    assert all(r['status'] == 'PASS' for r in records)
    return records


@pytest.mark.parametrize('k', [128, 256, 384, 512])
@pytest.mark.parametrize('block_n,sharded,policy', [(128, False, 0), (128, False, 18), (64, False, 18), (128, True, 18)])
@_SKIP_GFX950_FLYDSL
def test_4wave_boundaries(k, block_n, sharded, policy):
    """Exact original-kernel/graph checks; K512 is NOT a model-reference PASS."""
    tokens, n, topk, experts = 65, 2048, 2, 3
    data = _generate_a8w4_gui_data(tokens, n, k, experts, topk, 128, seed=43, inter_pad_override=0)
    ids, weights, eids, valid, scales = (data[key] for key in
        ('sorted_ids', 'sorted_weights', 'sorted_expert_ids', 'num_valid_ids', 'a2_scale_sort'))
    guard = torch.full((tokens * topk * n + 2 * n,), torch.nan, dtype=torch.bfloat16, device='cuda')
    middle = guard[n:-n].view(tokens, topk, n)
    output = torch.empty((tokens, n), dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(256 if sharded else 1, dtype=torch.int32, device='cuda')
    args = (middle, data['a2_q'], data['w2_shuf'], scales, data['w2_scale_shuf'], ids, weights, eids, valid, counter)
    shape = dict(n=n, k=k, topk=topk, num_experts=experts, block_n=block_n, sharded=sharded,
                 output_cache_policy=policy)
    parent = flydsl_moe_gemm_8wave_down_a8w4(n=n, k=k, topk=topk, num_experts=experts,
                                           block_m=128, block_n=block_n, num_oc_splits=4)
    parent(*args)
    torch.cuda.synchronize()
    assert torch.isfinite(middle).all()
    reference = middle.clone()
    kernel = flydsl_moe_gemm_4wave_down_a8w4(**shape)

    def full():
        kernel(*args)
        torch.sum(middle, dim=1, out=output)

    full()
    check_exact(middle, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full()
        full()
    old_ids, old_valid, old_weights, old_eids = ids.clone(), valid.clone(), weights.clone(), eids.clone()
    length = int(valid[0].item())
    encoded = old_ids[:length].to(torch.int64) & 0xFFFFFFFF
    token, slot = encoded & 0xFFFFFF, encoded >> 24
    active = (token < tokens) & (slot < topk)
    for state in ('normal', 'drop_even', 'drop_odd', 'bad_token', 'bad_slot', 'signed_slot', 'empty', 'zero_weights', 'restore'):
        ids.copy_(old_ids)
        valid.copy_(old_valid)
        weights.copy_(old_weights)
        ids[length:], eids[length // 128:] = 0, experts - 1
        expected = reference.clone()
        if state.startswith('drop_'):
            positions = torch.arange(length, device='cuda')
            drop = active & (positions % 2 == (0 if state == 'drop_even' else 1))
            ids[positions[drop]] = (topk << 24) | tokens
            expected[token[drop], slot[drop]] = torch.nan
        elif state in ('bad_token', 'bad_slot', 'signed_slot'):
            sentinel = tokens if state == 'bad_token' else (topk << 24) if state == 'bad_slot' else -16777216
            ids[:length] = sentinel
            expected.fill_(torch.nan)
        elif state == 'empty':
            valid.zero_()
            expected.fill_(torch.nan)
        elif state == 'zero_weights':
            weights.zero_()
            expected.zero_()
        middle.fill_(torch.nan)
        output.fill_(torch.nan)
        counter.fill_(123)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(middle, expected, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(output, expected.sum(dim=1), atol=0, rtol=0, equal_nan=True)
        assert torch.isnan(guard[:n]).all() and torch.isnan(guard[-n:]).all()
        tasks = int(valid[0].item()) // 128 * 4
        if sharded:
            for h in range(8):
                assert counter[h * 32].item() == max(0, (tasks - h + 7) // 8) + 64
            padding = counter.view(8, 32)[:, 1:]
            assert (padding == 0).all()
        else:
            assert counter.item() == tasks + 512
    ids.copy_(old_ids)
    valid.copy_(old_valid)
    weights.copy_(old_weights)
    eids.copy_(old_eids)


def test_4wave_config_and_errors():
    assert select_config(tokens=4096, n=6144, k=256, topk=8) == dict(block_n=128, output_cache_policy=0, sharded=False)
    assert select_config(tokens=8192, n=6144, k=256, topk=8) == dict(block_n=128, output_cache_policy=18, sharded=False)
    assert select_config(tokens=32768, n=6144, k=256, topk=8)['sharded']
    assert select_config(tokens=16384, n=6144, k=384, topk=8)['block_n'] == 64
    assert '4wave_bn64' in DEFAULT_CANDIDATES and '4wave_bn128' in DEFAULT_CANDIDATES
    for k in (256, 384):
        shape = dict(tokens=16384, n=6144, k=k, topk=8, num_experts=384)
        for key, bn in (('4wave_bn64', 64), ('4wave_bn128', 128)):
            config = build_local(key, **shape).config
            assert config['block_n'] == bn
            assert config['num_waves'] == 4 and config['output_cache_policy'] == 18
            assert not config['sharded']
    assert select_config(tokens=32768, n=3072, k=256, topk=2)['output_cache_policy'] == 0
    for kwargs in (dict(n=512, k=256), dict(n=6144, k=640), dict(n=1536, k=256), dict(n=6144, k=256, output_cache_policy=2)):
        with pytest.raises(ValueError):
            flydsl_moe_gemm_4wave_down_a8w4(topk=2, num_experts=4, **kwargs)
    with pytest.raises(AssertionError):
        _check_close(torch.zeros(2), torch.full((2,), torch.nan), 'must fail')


@_SKIP_GFX950_FLYDSL
def test_4wave_partial_sort_capacity():
    # Aiter capacity uses tokens*topk + E*BM - topk, which need not align to BM.
    data = _generate_a8w4_gui_data(64, 2048, 256, 3, 2, 128, seed=43, inter_pad_override=0)
    ids = data['sorted_ids']
    assert ids.numel() % 128 != 0
    middle = torch.full((64, 2, 2048), torch.nan, dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    kernel = flydsl_moe_gemm_4wave_down_a8w4(n=2048, k=256, topk=2, num_experts=3)
    kernel(middle, data['a2_q'], data['w2_shuf'], data['a2_scale_sort'], data['w2_scale_shuf'],
           ids, data['sorted_weights'], data['sorted_expert_ids'], data['num_valid_ids'], counter)
    torch.cuda.synchronize()
    assert torch.isfinite(middle).all()
    _check_close(data['ref_stage2'], middle.sum(dim=1), 'partial sorting capacity')


@_SKIP_GFX950_FLYDSL
def test_4wave_actual_isa():
    from a8w4_test_isa import audit
    data = _generate_a8w4_gui_data(65, 2048, 256, 3, 2, 128, seed=43, inter_pad_override=0)
    middle = torch.empty((65, 2, 2048), dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    reports = []
    for policy in (0, 18):
        kernel = flydsl_moe_gemm_4wave_down_a8w4(n=2048, k=256, topk=2, num_experts=3, output_cache_policy=policy)
        kernel(middle, data['a2_q'], data['w2_shuf'], data['a2_scale_sort'], data['w2_scale_shuf'],
               data['sorted_ids'], data['sorted_weights'], data['sorted_expert_ids'], data['num_valid_ids'], counter)
        torch.cuda.synchronize()
        reports.append(audit(kernel, policy))
    assert reports[0]['store_flags_removed_sha256'] == reports[1]['store_flags_removed_sha256']
    assert reports[0]['static_mfma_count'] > 0 and reports[0]['static_output_store_count'] > 0


@pytest.mark.parametrize('mode', ['down', 'down-reduce'])
def test_cli_scope(monkeypatch, mode):
    import sys
    calls = []
    monkeypatch.setattr(sys.modules[__name__], 'run_test', lambda **kwargs: calls.append(kwargs) or [])
    main(['--tokens', '65', '--candidate', '4wave', '--mode', mode])
    assert len(calls) == 1 and calls[0]['reduce_output'] == (mode == 'down-reduce')


@_SKIP_GFX950_FLYDSL
def test_reducer_comparison_metrics():
    results = run_test(tokens=65, model_dim=3072, experts=4, topk=2,
                       candidates=['4wave_bn64'], rounds=1, iters=2, compare_reduce=True)
    row = results[0]
    assert row['config']['reduction'] == 'moe'
    assert row['reduce_bytes'] == (65 * 2 * 3072 + 65 * 3072) * 2
    for kind in ('torch', 'moe'):
        comparison = row['reduction_comparison'][kind]
        assert comparison['full_us'] > 0 and comparison['reduce_us'] > 0
        assert comparison['logical_tb_s'] == row['reduce_bytes'] / comparison['reduce_us'] / 1e6
    assert row['median_us']['full'] == row['reduction_comparison']['moe']['full_us']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=[4096, 8192, 16384, 32768])
    parser.add_argument('--model-dim', '--n', dest='model_dim', type=int, default=6144)
    parser.add_argument('--inter-dim', '--k', dest='inter_dim', type=int, default=256)
    parser.add_argument('--experts', type=int, default=384)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--candidate', nargs='+', choices=CANDIDATES, default=list(DEFAULT_CANDIDATES))
    parser.add_argument('--mode', choices=('down', 'down-reduce'), default=None)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--reducer', choices=('moe', 'torch'), default='moe')
    parser.add_argument('--compare-reduce', action='store_true', help='Paired Torch vs make_moe_sum Full and warm-reduce timings')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    reduce_output = args.mode == 'down-reduce' or (args.mode is None and not args.profile)
    if args.profile and (len(args.tokens) != 1 or len(args.candidate) != 1):
        parser.error('--profile requires one token count and one candidate')
    if args.compare_reduce and (args.profile or not reduce_output):
        parser.error('--compare-reduce requires timed down-reduce mode')
    if any(k.startswith('opus') for k in args.candidate) and (args.inter_dim != 384 or not reduce_output):
        parser.error('Opus comparison requires K384 and --mode down-reduce')
    reports = []
    for tokens in args.tokens:
        results = run_test(tokens=tokens, model_dim=args.model_dim, inter_dim=args.inter_dim,
            experts=args.experts, topk=args.topk, seed=args.seed, candidates=args.candidate,
            profile=args.profile, reduce_output=reduce_output, rounds=args.rounds, iters=args.iters,
            reducer=args.reducer, compare_reduce=args.compare_reduce)
        reports.append(dict(tokens=tokens, n=args.model_dim, k=args.inter_dim, experts=args.experts,
                            topk=args.topk, seed=args.seed, results=results))
    if args.json:
        root = Path(__file__).parent
        files = ('test_a8w4.py', 'a8w4_test_utils.py', 'a8w4_test_isa.py',
                  'moe_4wave_down_a8w4.py', 'moe_8wave_down_a8w4.py', 'moe_8wave_down_a8w4_optimized.py',
                  'moe_8wave_down_utils.py', 'a8w4_store_policy.py',
                  'moe_multistage_reduce.py', 'moe_multistage_down.py')
        report = dict(cases=reports, rounds=args.rounds, iters=args.iters, torch=torch.__version__,
                      reducer=args.reducer, compare_reduce=args.compare_reduce,
                      device=torch.cuda.get_device_name(), clock_locked=False,
                      source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    return reports


if __name__ == '__main__':
    main()

