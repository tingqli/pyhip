# SPDX-License-Identifier: MIT
"""Historical A8W4 exploration: one exact check, then paired Down/Full timing.

Run from the parent directory with python -m experiments.bench. No pytest,
graph robustness matrix, guard poisoning, occupancy probe or automatic ISA audit.
Full intentionally keeps Torch sum (packed: inverse + sequential gather sum).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('FLYDSL_RUNTIME_ENABLE_CACHE', '0')

import torch
import pyhip
from a8w4_test_utils import _generate_a8w4_gui_data, check_exact, print_markdown_table, sorted_metadata
from moe_8wave_down_a8w4 import flydsl_moe_gemm_8wave_down_a8w4
from moe_multistage_reduce import make_moe_sum
from pyhip.contrib.flydsl.moe_gemm_2stage.moe_reduce import invert_sorted_ids
from .moe_8wave_down_a8w4_experiments import EXPERIMENTS, make_experiment
from .moe_a8w4_4wave_oc4 import make_oc4_4wave


# Only retained useful controls; low-value historical variants remain in archive.
OC4 = {
    'oc4_8w': None,
    'uniform512': {},
    'paired512': {'publication': 'paired'},
    'paired_bn64': {'publication': 'paired', 'block_n': 64},
    'sharded512': {'publication': 'paired', 'sharded': True},
}
SINGLE_DEFAULTS = ('control', 'oc4', 'compute_overlap', 'packed', 'packed_dpp')


def run(args):
    shape = dict(n=args.n, k=args.k, topk=args.topk, num_experts=args.experts)
    single = args.suite == 'single'
    choices = EXPERIMENTS if single else OC4
    names = list(args.variants or (SINGLE_DEFAULTS if single else OC4))
    if not names or any(name not in choices for name in names):
        raise ValueError(f'unknown variants; choices: {", ".join(choices)}')
    # Same-model single-factor comparisons keep their declared parents.
    if single:
        for name in list(names):
            parent = EXPERIMENTS[name][0]
            while parent is not None:
                if parent not in names:
                    names.insert(0, parent)
                parent = EXPERIMENTS[parent][0]
    elif 'oc4_8w' not in names:
        names.insert(0, 'oc4_8w')
    data = _generate_a8w4_gui_data(args.tokens, args.n, args.k, args.experts, args.topk,
                                   256, seed=args.seed, inter_pad_override=0)
    metadata = {256: (data['sorted_ids'], data['sorted_weights'], data['sorted_expert_ids'],
                      data['num_valid_ids'], data['a2_scale_sort'])}
    metadata[128] = sorted_metadata(data, 128, args.experts)
    capacity = max(eids.numel() * bm for bm, (_, _, eids, _, _) in metadata.items())
    storage = torch.empty((capacity, args.n), dtype=torch.bfloat16, device='cuda')
    routed = storage[:args.tokens * args.topk].view(args.tokens, args.topk, args.n)
    output = torch.empty((args.tokens, args.n), dtype=torch.bfloat16, device='cuda')
    counter = torch.zeros(256, dtype=torch.int32, device='cuda')
    inverse = torch.empty((args.tokens, args.topk), dtype=torch.int32, device='cuda')

    def inputs(bm, target, sharded=False):
        ids, routes, eids, valid, sa = metadata[bm]
        return (target, data['a2_q'], data['w2_shuf'], sa, data['w2_scale_shuf'],
                ids, routes, eids, valid, counter if sharded else counter[:1])

    baseline = flydsl_moe_gemm_8wave_down_a8w4(**shape, block_m=256, block_n=128,
                                              num_oc_splits=1 if single else 4)
    routed.fill_(torch.nan)
    baseline(*inputs(256, routed))
    torch.cuda.synchronize()
    assert torch.isfinite(routed).all()
    reference = routed.clone()
    expected = reference.sum(dim=1)
    launches, records = {}, {}
    for name in names:
        if single:
            options = EXPERIMENTS[name][2]
            bm, packed = options.get('block_m', 256), options.get('packed_output', False)
            settings = [(None, None)]
        else:
            options = OC4[name] or {}
            bm, packed = (256 if name == 'oc4_8w' else 128), False
            settings = [(None, None)] if name == 'oc4_8w' else (
                [(dpp, policy) for dpp in (False, True) for policy in (0, 18)]
                if args.suite == 'stores' else [(False, None)])
        ids, _, eids, valid, _ = metadata[bm]
        middle = storage[:eids.numel() * bm] if packed else routed
        kernel_args = inputs(bm, middle, options.get('sharded', False))
        for dpp, policy in settings:
            key = name if dpp is None or args.suite != 'stores' else f'{name}/{128 if dpp else 64}B/aux{policy}'
            kernel = make_experiment(name, **shape) if single else baseline if name == 'oc4_8w' else make_oc4_4wave(
                **shape, **options, coalesce_output=dpp, output_cache_policy=policy)
            reduce = make_moe_sum(n=args.n, topk=args.topk, sort_block_m=bm) if packed else None
            invert = invert_sorted_ids(args.topk) if packed else None

            def down(kernel=kernel, kernel_args=kernel_args):
                kernel(*kernel_args)

            def full(down=down, packed=packed, reduce=reduce, invert=invert, middle=middle, ids=ids, valid=valid):
                down()
                if packed:
                    inverse.fill_(-1)
                    invert(ids, inverse, valid, ids.numel(), args.tokens)
                    reduce(output, middle, inverse)
                else:
                    torch.sum(routed, dim=1, out=output)

            # One ordinary correctness check before any timing, never swallow failure.
            middle.fill_(torch.nan)
            full()
            torch.cuda.synchronize()
            if packed:
                physical = middle.view(-1, args.n // 64, bm, 64)
                length = int(valid[0].item())
                for begin in range(0, length, 256):
                    pos = torch.arange(begin, min(begin + 256, length), device='cuda')
                    encoded = ids[pos].to(torch.int64) & 0xFFFFFFFF
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    live = (token < args.tokens) & (slot < args.topk)
                    pos, token, slot = pos[live], token[live], slot[live]
                    check_exact(physical[pos // bm, :, pos % bm, :].reshape(-1, args.n), reference[token, slot])
                for begin in range(0, args.tokens, 64):
                    total = reference[begin:begin + 64, 0].float()
                    for slot in range(1, args.topk):
                        total += reference[begin:begin + 64, slot].float()
                    check_exact(output[begin:begin + 64], total.to(torch.bfloat16))
            else:
                check_exact(middle, reference)
                check_exact(output, expected)
            records[key] = dict(exact=True, options=options, dpp=dpp, policy=policy,
                                samples={'down': [], 'full': []})
            launches[key] = down, full
            print(f'CHECK {key}: exact PASS', flush=True)
    for r in range(args.rounds):
        keys = list(records)
        for key in keys if r % 2 == 0 else reversed(keys):
            for scope, launch in zip(('down', 'full'), launches[key]):
                _, us = pyhip.run_perftest(launch, num_warmup=2, num_iters=args.iters, num_copies=1, num_verbose=0)
                records[key]['samples'][scope].append(us)
    for entry in records.values():
        entry['median_us'] = {scope: statistics.median(values) for scope, values in entry['samples'].items()}
    print_markdown_table(['Experiment', 'Down us', 'Full us'], [
        [key, f'{entry["median_us"]["down"]:.3f}', f'{entry["median_us"]["full"]:.3f}'] for key, entry in records.items()])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('single', 'oc4', 'stores'), default='single')
    parser.add_argument('--variants', nargs='+')
    for name, default in (('tokens', 16384), ('n', 6144), ('k', 256), ('experts', 384),
                          ('topk', 8), ('seed', 1234), ('rounds', 5), ('iters', 20)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    assert args.rounds > 0 and args.iters > 0 and os.environ.get('PYHIP_FLYDSL_NOP_MFMA') != '1'
    records = run(args)
    if args.json:
        root = Path(__file__).resolve().parents[1]
        files = ['a8w4_test_utils.py', 'moe_8wave_down_a8w4.py', 'moe_8wave_down_utils.py',
                 'moe_multistage_down.py', 'moe_multistage_reduce.py', 'a8w4_store_policy.py',
                 'experiments/bench.py', 'experiments/moe_8wave_down_a8w4_experiments.py',
                 'experiments/moe_a8w4_4wave_oc4.py']
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(suite=args.suite,
            shape={key: getattr(args, key) for key in ('tokens', 'n', 'k', 'experts', 'topk', 'seed')},
            validation='one ordinary exact check; no robustness matrix', results=records,
            source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}), indent=2) + '\n')


if __name__ == '__main__':
    main()