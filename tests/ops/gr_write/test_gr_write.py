# SPDX-License-Identifier: MIT
"""GR write regression tests plus a small accuracy/timing CLI.

pytest only checks correctness; the CLI can add cudaPerf timing (see benchmarks/gr_write
for the full performance script).
"""
import argparse
import os
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))

from pyhip.testing.gr_write import (  # noqa: E402
    C, H, K, check_close, make_inputs, make_weights, reference,
)

ROW_SIZES = (1, 2, 3, 4, 5, 33, 129, 1000, 4097, 12000)


def _require_rocm():
    torch = pytest.importorskip('torch')
    pytest.importorskip('flydsl')
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip('ROCm GPU required')
    return torch


@pytest.fixture(scope='module')
def weights():
    _require_rocm()
    from pyhip.ops.gr_write import prepare_weights
    inject, norm = make_weights(seed=5)
    packed_inject, packed_norm = prepare_weights(inject, norm)
    return inject, norm, packed_inject, packed_norm


def test_prepare_weights_layout(weights):
    torch = _require_rocm()
    inject, norm, packed_inject, packed_norm = weights
    assert packed_inject.shape == (C * K,) and packed_inject.dtype == torch.bfloat16
    assert packed_norm.shape == (K,) and packed_norm.dtype == torch.float32
    # Fragment [slice s][group g][t][block][c][e] holds column s*2560 + g*512 + t*128 + block*8 + e of row c.
    fragments = packed_inject.view(4, 5, 4, 16, C, 8)
    for s, g, t, block, c, e in ((0, 0, 0, 0, 0, 0), (3, 4, 3, 15, 3, 7), (1, 2, 1, 9, 2, 5)):
        column = s * 2560 + g * 512 + t * 128 + block * 8 + e
        assert torch.equal(fragments[s, g, t, block, c, e], inject[c, column])
    # Gain [wave][chunk][half][lane][e]: chunk b < 4 holds 1 + w[b*2560 + 512*wave + 8*lane + 4*half + e],
    # chunk 4 holds 1 + w[wave*2560 + 2048 + 8*lane + 4*half + e].
    gains = packed_norm.view(4, 5, 2, 64, 4)
    for wave, chunk, half, lane, e in ((0, 0, 0, 0, 0), (3, 4, 1, 63, 3), (2, 1, 0, 17, 2), (1, 4, 0, 5, 1)):
        branch, base = (chunk, 512 * wave) if chunk < 4 else (wave, 2048)
        column = branch * H + base + 8 * lane + 4 * half + e
        assert gains[wave, chunk, half, lane, e].item() == 1.0 + norm[column].float().item()
    # Every gain appears exactly once.
    assert torch.equal(packed_norm.sort().values, (1.0 + norm.float()).sort().values)


@pytest.mark.parametrize('rows', ROW_SIZES)
def test_matches_reference(weights, rows):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write
    inject, norm, packed_inject, packed_norm = weights
    y, r, n = make_inputs(rows, seed=rows)
    copies = [t.clone() for t in (y, r, n)]
    with torch.inference_mode():
        out, normed = gr_write(y, r, n, packed_inject, packed_norm)
    torch.cuda.synchronize()
    for before, after in zip(copies, (y, r, n)):
        assert torch.equal(before, after), 'inputs must not change'
    out_ref, normed_ref = reference(y, r, n, inject, norm)
    check_close(out, normed, out_ref, normed_ref)


def test_output_buffers_guards_and_repeat(weights):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write
    inject, norm, packed_inject, packed_norm = weights
    rows = 777
    y, r, n = make_inputs(rows, seed=11)
    # Guard rows around both outputs catch out-of-range stores.
    out_storage = torch.full((rows + 2, K), 7.0, dtype=torch.bfloat16, device='cuda')
    normed_storage = torch.full((rows + 2, K), -3.0, dtype=torch.bfloat16, device='cuda')
    out, normed = out_storage[1:rows + 1], normed_storage[1:rows + 1]
    with torch.inference_mode():
        result = gr_write(y, r, n, packed_inject, packed_norm, out=out, normed=normed)
        assert result[0] is out and result[1] is normed
        first = (out.clone(), normed.clone())
        gr_write(y, r, n, packed_inject, packed_norm, out=out, normed=normed)
    torch.cuda.synchronize()
    for storage, value in ((out_storage, 7.0), (normed_storage, -3.0)):
        assert torch.all(storage[0] == value) and torch.all(storage[-1] == value)
    assert torch.equal(first[0], out) and torch.equal(first[1], normed), 'repeated calls must be bitwise stable'
    check_close(out, normed, *reference(y, r, n, inject, norm))


def test_eps_variant(weights):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write
    inject, norm, packed_inject, packed_norm = weights
    y, r, n = make_inputs(64, seed=3)
    y[:5].mul_(1e-3)  # rows with tiny branches make eps visible
    r[:5].mul_(1e-3)
    with torch.inference_mode():
        out, normed = gr_write(y, r, n, packed_inject, packed_norm, eps=1e-2)
    check_close(out, normed, *reference(y, r, n, inject, norm, eps=1e-2))


def test_empty_rows(weights):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write
    _, _, packed_inject, packed_norm = weights
    y, r, n = make_inputs(0)
    with torch.inference_mode():
        out, normed = gr_write(y, r, n, packed_inject, packed_norm)
    assert out.shape == (0, K) and normed.shape == (0, K)


def test_rejects_invalid_arguments(weights):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write, prepare_weights
    inject, norm, packed_inject, packed_norm = weights
    y, r, n = make_inputs(8)
    with torch.inference_mode():
        with pytest.raises(ValueError, match='block_output'):
            gr_write(y[:7], r, n, packed_inject, packed_norm)
        with pytest.raises(ValueError, match='normed_residual'):
            gr_write(y, r, n.float(), packed_inject, packed_norm)
        with pytest.raises(ValueError, match='residual'):
            gr_write(y, r.t(), n, packed_inject, packed_norm)
        with pytest.raises(ValueError, match='packed_norm'):
            gr_write(y, r, n, packed_inject, packed_norm.to(torch.bfloat16))
        with pytest.raises(ValueError, match='16-byte'):
            flat = torch.empty(8 * K + 1, dtype=torch.bfloat16, device='cuda')
            gr_write(y, flat[1:].view(8, K), n, packed_inject, packed_norm)
        with pytest.raises(ValueError, match='overlap'):
            gr_write(y, r, n, packed_inject, packed_norm, out=r)
        with pytest.raises(ValueError, match='overlap'):
            shared = torch.empty((8, K), dtype=torch.bfloat16, device='cuda')
            gr_write(y, r, n, packed_inject, packed_norm, out=shared, normed=shared)
        with pytest.raises(ValueError, match='eps'):
            gr_write(y, r, n, packed_inject, packed_norm, eps=0.0)
    with pytest.raises(ValueError, match='inference-only'):
        gr_write(y, r.requires_grad_(False), n, packed_inject, packed_norm,
                 out=torch.empty((8, K), dtype=torch.bfloat16, device='cuda', requires_grad=True))
    with pytest.raises(ValueError, match='inject_weight'):
        prepare_weights(inject.float(), norm)
    with pytest.raises(ValueError, match='norm_weight'):
        prepare_weights(inject, norm[:-1])


def test_graph_replay(weights):
    torch = _require_rocm()
    from pyhip.ops.gr_write import gr_write
    inject, norm, packed_inject, packed_norm = weights
    rows = 513
    y, r, n = make_inputs(rows, seed=21)
    with torch.inference_mode():
        gr_write(y, r, n, packed_inject, packed_norm)  # warm up the compiled call
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, normed = gr_write(y, r, n, packed_inject, packed_norm)
        for seed in (22, 23):
            fresh = make_inputs(rows, seed=seed)
            for dst, src in zip((y, r, n), fresh):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            check_close(out, normed, *reference(y, r, n, inject, norm))


def test_second_device():
    torch = _require_rocm()
    if torch.cuda.device_count() < 2:
        pytest.skip('needs two visible GPUs')
    from pyhip.ops.gr_write import gr_write, prepare_weights
    device = torch.device('cuda', 1)
    inject, norm = make_weights(seed=9, device=device)
    packed = prepare_weights(inject, norm)
    y, r, n = make_inputs(301, seed=9, device=device)
    with torch.cuda.device(0), torch.inference_mode():
        out, normed = gr_write(y, r, n, *packed)
        assert torch.cuda.current_device() == 0
    assert out.device == device and normed.device == device
    check_close(out, normed, *reference(y, r, n, inject, norm))


def _timing(rows, buffers, warmup, iters, seed):
    import torch
    from pyhip.ops.gr_write import gr_write, prepare_weights
    from pyhip.testing import cudaPerf
    inject, norm = make_weights(seed)
    packed = prepare_weights(inject, norm)
    sets = [make_inputs(rows, seed=seed + i) for i in range(buffers)]
    outs = [(torch.empty((rows, K), dtype=torch.bfloat16, device='cuda'),
             torch.empty((rows, K), dtype=torch.bfloat16, device='cuda')) for _ in range(buffers)]
    perf = cudaPerf(name=f'gr_write T={rows}', verbose=0)
    with torch.inference_mode():
        for index in range(warmup + iters):
            y, r, n = sets[index % buffers]
            out, normed = outs[index % buffers]
            with perf:
                gr_write(y, r, n, *packed, out=out, normed=normed)
    samples = sorted(perf.latencies[warmup:])
    return samples[len(samples) // 2] * 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=None, help='physical GPU index (sets HIP_VISIBLE_DEVICES)')
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 33, 4097, 12000])
    parser.add_argument('--check-only', action='store_true', help='skip timing')
    parser.add_argument('--buffers', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.gpu is not None:
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from pyhip.ops.gr_write import gr_write, prepare_weights
    inject, norm = make_weights(args.seed)
    packed = prepare_weights(inject, norm)
    for rows in args.rows:
        y, r, n = make_inputs(rows, seed=args.seed + rows)
        with torch.inference_mode():
            out, normed = gr_write(y, r, n, *packed)
        exact = check_close(out, normed, *reference(y, r, n, inject, norm))
        line = f'T={rows}: accuracy OK (exact out {exact[0]:.6f}, normed {exact[1]:.6f})'
        if not args.check_only:
            line += f', median {_timing(rows, args.buffers, args.warmup, args.iters, args.seed):.1f} us'
        print(line, flush=True)


if __name__ == '__main__':
    main()
