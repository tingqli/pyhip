# SPDX-License-Identifier: MIT
"""Eight-wave DPP/cache/pack combinations: exact routes, graph and actual ISA."""

import os
os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '0')

import pytest
import torch
from a8w4_test_utils import _generate_a8w4_gui_data, check_exact
from moe_8wave_down_a8w4 import flydsl_moe_gemm_8wave_down_a8w4
from moe_8wave_down_a8w4_optimized import flydsl_moe_gemm_8wave_down_a8w4_optimized
from a8w4_test_isa import inspect_isa


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties().gcnArchName.startswith('gfx950'):
        pytest.skip('gfx950 required')


@pytest.mark.parametrize('k', [128, 256, 384, 512])
@pytest.mark.parametrize('n,bn,oc,tokens,topk', [
    (1536, 128, 4, 65, 1),  # shortest three packets, no steady-state iterations
    (2048, 128, 4, 64, 2),  # partial capacity tail
    (3072, 64, 4, 65, 2),
    (3072, 128, 1, 257, 2),  # multiple M blocks, many ring reuses
    (1024, 128, 2, 65, 2),
    (3072, 128, 8, 65, 2),
])
def test_optimized_matrix(k, n, bn, oc, tokens, topk):
    require_gpu()
    experts = 3
    data = _generate_a8w4_gui_data(tokens, n, k, experts, topk, 256, seed=43, inter_pad_override=0)
    ids, weights, eids, valid, scales = (data[key] for key in
        ('sorted_ids', 'sorted_weights', 'sorted_expert_ids', 'num_valid_ids', 'a2_scale_sort'))
    guard = torch.full((tokens * topk * n + 2 * n,), torch.nan, dtype=torch.bfloat16, device='cuda')
    middle = guard[n:-n].view(tokens, topk, n)
    output = torch.empty((tokens, n), dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    args = (middle, data['a2_q'], data['w2_shuf'], scales, data['w2_scale_shuf'], ids, weights, eids, valid, counter)
    shape = dict(n=n, k=k, topk=topk, num_experts=experts, block_n=bn, num_oc_splits=oc)
    baseline = flydsl_moe_gemm_8wave_down_a8w4(**shape, block_m=256)
    baseline(*args)
    torch.cuda.synchronize()
    assert torch.isfinite(middle).all()
    reference = middle.clone()
    old_ids, old_weights, old_valid, old_eids = ids.clone(), weights.clone(), valid.clone(), eids.clone()
    length = int(valid[0].item())
    encoded = old_ids[:length].to(torch.int64) & 0xFFFFFFFF
    token, slot = encoded & 0xFFFFFF, encoded >> 24
    active = (token < tokens) & (slot < topk)
    isa_pairs = {}
    # Main DPP path gets the full boundary suite. Ablation combinations receive
    # a direct exact check in test_ablation_smoke, not another robustness matrix.
    for dpp, overlap, policy in ((True, bn == 128, 0), (True, bn == 128, 18)):
        kernel = flydsl_moe_gemm_8wave_down_a8w4_optimized(**shape, coalesce_output=dpp,
                     compute_overlap=overlap, output_cache_policy=policy)
        def full():
            kernel(*args)
            torch.sum(middle, dim=1, out=output)
        middle.fill_(torch.nan)
        full()
        check_exact(middle, reference)
        check_exact(output, reference.sum(dim=1))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full()
            full()
        for state in ('normal', 'drop_even', 'drop_odd', 'bad_token', 'bad_slot', 'signed_slot', 'empty', 'zero', 'restore'):
            ids.copy_(old_ids)
            weights.copy_(old_weights)
            valid.copy_(old_valid)
            ids[length:], eids[length // 256:] = 0, experts - 1
            expected = reference.clone()
            if state.startswith('drop_'):
                pos = torch.arange(length, device='cuda')
                dropped = active & (pos % 2 == (0 if state == 'drop_even' else 1))
                ids[pos[dropped]] = (topk << 24) | tokens
                expected[token[dropped], slot[dropped]] = torch.nan
            elif state in ('bad_token', 'bad_slot', 'signed_slot'):
                ids[:length] = tokens if state == 'bad_token' else topk << 24 if state == 'bad_slot' else -16777216
                expected.fill_(torch.nan)
            elif state == 'empty':
                valid.zero_()
                expected.fill_(torch.nan)
            elif state == 'zero':
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
            assert counter.item() == int(valid[0].item()) // 256 * oc + 256
        ids.copy_(old_ids)
        weights.copy_(old_weights)
        valid.copy_(old_valid)
        eids.copy_(old_eids)
        isa = inspect_isa(kernel, policy)
        assert bool(isa['dpp_instructions']) == dpp
        assert isa['dma_instructions'] > 0 and isa['static_mfma_count'] > 0
        if policy == 0:
            isa_pairs[dpp, overlap] = isa['store_flags_removed_sha256']
        else:
            assert isa_pairs[dpp, overlap] == isa['store_flags_removed_sha256']


def test_optimized_rejects_invalid_config():
    for options in (dict(block_n=64), dict(n=1024), dict(k=640), dict(output_cache_policy=2), dict(topk=128)):
        shape = dict(n=6144, k=256, topk=8, num_experts=384)
        shape.update(options)
        with pytest.raises(ValueError):
            flydsl_moe_gemm_8wave_down_a8w4_optimized(**shape)


def test_single_factor_options():
    from test_a8w4 import OPT8_OPTIONS
    parents = {
        '8wave_opt_dpp': '8wave_opt_control',
        '8wave_opt_ntsc1': '8wave_opt_dpp',
        '8wave_opt_dpp_overlap': '8wave_opt_dpp',
        '8wave_optimized': '8wave_opt_ntsc1',
        '8wave_opt_oc1': '8wave_optimized',
        '8wave_opt_bn64': '8wave_opt_ntsc1',
    }
    for name, parent in parents.items():
        a, b = OPT8_OPTIONS[name], OPT8_OPTIONS[parent]
        assert sum(a.get(key) != b.get(key) for key in a.keys() | b.keys()) == 1


@pytest.mark.parametrize('k', [128, 256, 384, 512])
def test_ablation_smoke(k):
    """Experimental switches: one exact output check, no graph/guard/ISA matrix."""
    import itertools
    require_gpu()
    data = _generate_a8w4_gui_data(65, 2048, k, 3, 2, 256, seed=43, inter_pad_override=0)
    output = torch.empty((65, 2, 2048), dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    args = (output, data['a2_q'], data['w2_shuf'], data['a2_scale_sort'], data['w2_scale_shuf'],
            data['sorted_ids'], data['sorted_weights'], data['sorted_expert_ids'], data['num_valid_ids'], counter)
    shape = dict(n=2048, k=k, topk=2, num_experts=3, num_oc_splits=4)
    flydsl_moe_gemm_8wave_down_a8w4(**shape, block_m=256)(*args)
    reference = output.clone()
    for dpp, overlap, policy in itertools.product((False, True), (False, True), (0, 18)):
        kernel = flydsl_moe_gemm_8wave_down_a8w4_optimized(**shape, coalesce_output=dpp,
                     compute_overlap=overlap, output_cache_policy=policy)
        output.fill_(torch.nan)
        kernel(*args)
        check_exact(output, reference)