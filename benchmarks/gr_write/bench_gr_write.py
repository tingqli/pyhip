# SPDX-License-Identifier: MIT
"""GR write prefill timing: public gr_write() total, optional gate/apply breakdown and Torch baseline.

Protocol (same as benchmarks/gr_read prefill): eager cudaPerf, independent input/output
buffer sets rotated per call, warmup samples dropped, median reported.
"""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=None, help='physical GPU index (sets HIP_VISIBLE_DEVICES)')
    parser.add_argument('--rows', type=int, nargs='+', default=None, help='T values (default: 1K..32K incl. 12000)')
    parser.add_argument('--buffers', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--breakdown', action='store_true', help='also time the gate and apply kernels alone')
    parser.add_argument('--baseline', action='store_true', help='also time an unfused Torch eager implementation')
    parser.add_argument('--no-check', action='store_true', help='skip the accuracy check')
    return parser.parse_args(argv)


def torch_baseline(y, r, n, inject, norm, eps):
    import torch
    rows = r.shape[0]
    dot = torch.matmul(n, inject.T).float()
    gate = 2.0 * torch.sigmoid(dot / 4)
    out = (r.view(rows, 4, -1).float() + gate[:, :, None] * y.float()[:, None, :]).to(torch.bfloat16)
    of = out.float()
    normed = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + norm.float().view(4, -1))
    return out.view(rows, -1), normed.to(torch.bfloat16).view(rows, -1)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(REPO / 'src'))
    import torch
    from pyhip.ops.gr_write import gr_write, prepare_weights
    from pyhip.ops.gr_write.flydsl.apply import make_apply
    from pyhip.ops.gr_write.flydsl.common import EPS, K, launch_grids, workspace_words
    from pyhip.ops.gr_write.flydsl.gate import make_gate
    from pyhip.testing import cudaPerf
    from pyhip.testing.gr_write import DEFAULT_ROWS, bytes_moved, check_close, make_inputs, make_weights, reference

    props = torch.cuda.get_device_properties(0)
    print(f'{props.name} / {props.gcnArchName.split(":")[0]} / {props.multi_processor_count} CU; '
          f'{args.buffers} buffers, {args.warmup} warmup, {args.iters} samples (median)', flush=True)
    inject, norm = make_weights(args.seed)
    packed_inject, packed_norm = prepare_weights(inject, norm)
    gate_kernel, apply_kernel = make_gate(), make_apply(EPS)
    stream = torch.cuda.current_stream()

    def median_us(calls, prepares=None):
        perf = cudaPerf(name='gr_write', verbose=0)
        for index in range(args.warmup + args.iters):
            if prepares is not None:
                prepares[index % len(prepares)]()  # submitted before the timed region
            with perf:
                calls[index % len(calls)]()
        samples = sorted(perf.latencies[args.warmup:])
        return samples[len(samples) // 2] * 1e6

    header = ['T', 'Total us', 'TB/s']
    if args.breakdown:
        header += ['Gate us', 'Apply us']
    if args.baseline:
        header += ['Torch eager us', 'Speedup']
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---:|' * len(header)]
    row_list = args.rows or DEFAULT_ROWS
    # Allocate every buffer set once at the largest T and time row-prefix views. Re-allocating
    # per T changes the physical placement, which alone moved a plain HIP streaming read of the
    # same tensors from 4.6 to 3.7 TB/s on MI308X.
    capacity = max(row_list)
    full_sets = [make_inputs(capacity, seed=args.seed + 1000 * i) for i in range(args.buffers)]
    full_outs = [(torch.empty((capacity, K), dtype=torch.bfloat16, device='cuda'),
                  torch.empty((capacity, K), dtype=torch.bfloat16, device='cuda')) for _ in range(args.buffers)]
    full_works = [torch.empty(workspace_words(capacity), dtype=torch.float32, device='cuda')
                  for _ in range(args.buffers)]
    with torch.inference_mode():
        for rows in row_list:
            sets = [tuple(t[:rows] for t in s) for s in full_sets]
            outs = [tuple(t[:rows] for t in o) for o in full_outs]
            if not args.no_check:
                y, r, n = sets[0]
                gr_write(y, r, n, packed_inject, packed_norm, out=outs[0][0], normed=outs[0][1])
                check_close(*outs[0], *reference(y, r, n, inject, norm))
            calls = [lambda s=s, o=o: gr_write(*s, packed_inject, packed_norm, out=o[0], normed=o[1])
                     for s, o in zip(sets, outs)]
            total = median_us(calls)
            row = [str(rows), f'{total:.1f}', f'{bytes_moved(rows) / total / 1e6:.2f}']
            if args.breakdown:
                gate_grid, apply_grid = launch_grids(rows, props.multi_processor_count)
                works = [w[:workspace_words(rows)] for w in full_works]
                gates = [lambda s=s, w=w: gate_kernel(s[2], packed_inject, w, rows, gate_grid, stream)
                         for s, w in zip(sets, works)]
                for g in gates:
                    g()  # gate values for the apply-only samples
                # The apply row counter normally comes zeroed from the gate kernel.
                resets = [lambda w=w: w[rows * 4:].zero_() for w in works]
                applies = [lambda s=s, w=w, o=o: apply_kernel(s[0], s[1], w, packed_norm, o[0], o[1], rows,
                                                              apply_grid, stream)
                           for s, w, o in zip(sets, works, outs)]
                row += [f'{median_us(gates):.1f}', f'{median_us(applies, resets):.1f}']
            if args.baseline:
                eager = median_us([lambda s=s: torch_baseline(*s, inject, norm, EPS) for s in sets])
                row += [f'{eager:.1f}', f'{eager / total:.2f}x']
            lines.append('| ' + ' | '.join(row) + ' |')
            print(lines[-1], flush=True)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
