"""Individual QSA indexer kernels: synthetic correctness and explicitly opted-in timing.

Default pytest and CLI execution check correctness only. Pytest timing requires
the perf marker selection plus QSA_REPLAY_OUTPUT; CLI timing requires --perf.
The unchanged shared benchmark owns timing, hardware gates and output receipts.
No captured input files, model service or plugin installation are required.
"""

import argparse
import os
from pathlib import Path
import sys

import pytest

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.ops.qsa._benchmark import DATA, measure_components, recording_matrix  # noqa: E402
from tests.ops.qsa._indexer import _gpu  # noqa: E402
from tests.ops.qsa._indexer_kernels import (  # noqa: E402
    DECODE_LABELS,
    PREFILL_LABELS,
    make_decode_case,
    make_prefill_case,
    prefill_labels,
)


# (sequence lengths, extend lengths). Every prefix is group-aligned. Keep rows
# small while crossing TB8, GB16, wave32, CTA128 and the 512-key selection edge.
PREFILL_CASES = (
    ((1,), (1,)), ((3,), (3,)), ((4,), (4,)), ((7,), (7,)), ((8,), (8,)), ((9,), (9,)),
    ((63,), (63,)), ((64,), (64,)), ((65,), (65,)),
    ((2047,), (3,)), ((2051,), (7,)), ((2052,), (8,)),
    ((2079,), (31,)), ((2080,), (32,)), ((2081,), (33,)),
    ((2175,), (127,)), ((2176,), (128,)), ((2177,), (129,)),
    ((2055, 13, 4101), (7, 9, 5)), ((2052, 16, 13), (4, 0, 9)),
)
# (token lengths, graph padding, context). These include compressed lengths
# 0/1, page boundaries, 511/512/513 and 1024/1025, with ragged graph rows up to 32.
DECODE_CASES = (
    ((1,), 0, 64), ((4,), 0, 64), ((5, 8, 3, 12), 1, 256),
    ((63, 64, 65, 68), 0, 256), ((2044, 2048, 2052, 2055), 2, 4096),
    ((4096, 4099, 4100), 1, 8192),
    (tuple(2052 + 4 * i for i in range(8)), 0, 4096),
    (tuple(2001 + 37 * i for i in range(29)), 3, 4096),
)
PREFILL_PERF_CASES = (((2055,), (7,)), ((2081,), (33,)), ((2177,), (129,)),
                      ((2055, 13, 4101), (7, 9, 5)))
DECODE_PERF_CASES = (((2052,), 0, 4096),
                     (tuple(2052 + 4 * i for i in range(8)), 0, 4096),
                     (tuple(2001 + 37 * i for i in range(29)), 3, 4096))


def _prefill_name(seq_lens, extend_lens):
    return f"prefill_s{'-'.join(map(str, seq_lens))}_e{'-'.join(map(str, extend_lens))}"


def _decode_name(lengths, padding, context):
    return f"decode_n{'-'.join(map(str, lengths))}_pad{padding}_ctx{context}"


def _check_case(case, scopes=None):
    labels = tuple(case.runs) if scopes is None else tuple(scopes)
    assert labels and set(labels) <= case.runs.keys()
    # Reverse the second pass to expose accidental dependencies between labels;
    # neither pass uses events, cudaPerf, graph replay or a performance gate.
    for order in (labels, labels[::-1]):
        for label in order:
            case.reset[label]()
            case.runs[label]()
            case.checks[label]()
    assert set(labels) <= case.metadata["resources"].keys()


@pytest.mark.parametrize("seq_lens,extend_lens", PREFILL_CASES,
                         ids=[_prefill_name(*spec) for spec in PREFILL_CASES])
def test_prefill_kernels(seq_lens, extend_lens):
    _check_case(make_prefill_case(seq_lens, extend_lens, _gpu()))


@pytest.mark.parametrize("lengths,padding,context", DECODE_CASES,
                         ids=[_decode_name(*spec) for spec in DECODE_CASES])
def test_decode_kernels(lengths, padding, context):
    _check_case(make_decode_case(lengths, _gpu(), padding=padding, context=context))


@pytest.mark.parametrize("pattern", ("projected", "equal", "repeated"))
@pytest.mark.parametrize("row0", (1, 3, 4))
def test_topk_ties_and_row_offset(pattern, row0):
    # A partial last CTA, nonzero absolute row ids and deliberately tied scores.
    # Only this label is launched; no prep/logits kernel output is a prerequisite.
    case = make_prefill_case((4101,), (129,), _gpu(), topk_pattern=pattern, row0=row0)
    _check_case(case, ("indexer_topk",))


@pytest.mark.parametrize("heads", (4, 8))
def test_decode_logits_shuffled_pages(heads):
    case = make_decode_case((2052, 65, 1, 4100), _gpu(), padding=1, context=8192,
                            heads=heads, shuffle_pages=True)
    _check_case(case, ("decode_logits",))


def _perf_folder(name):
    output = os.environ.get("QSA_REPLAY_OUTPUT")
    if not output:
        pytest.skip("Set QSA_REPLAY_OUTPUT under mytest/mydata to opt into kernel timing")
    return Path(output) / "indexer_kernels" / name


@pytest.mark.perf
@pytest.mark.parametrize("seq_lens,extend_lens", PREFILL_PERF_CASES,
                         ids=[_prefill_name(*spec) for spec in PREFILL_PERF_CASES])
def test_prefill_kernel_perf(seq_lens, extend_lens):
    folder = _perf_folder(_prefill_name(seq_lens, extend_lens))
    device = _gpu()
    measure_components(lambda buffer: make_prefill_case(seq_lens, extend_lens, device, seed=41 + buffer),
                       folder, device.index, buffers=10, warmup=2, samples=128)


@pytest.mark.perf
@pytest.mark.parametrize("lengths,padding,context", DECODE_PERF_CASES,
                         ids=[_decode_name(*spec) for spec in DECODE_PERF_CASES])
def test_decode_kernel_perf(lengths, padding, context):
    folder = _perf_folder(_decode_name(lengths, padding, context))
    device = _gpu()
    measure_components(lambda buffer: make_decode_case(lengths, device, padding=padding, context=context,
                                                      seed=73 + buffer),
                       folder, device.index, buffers=10, warmup=2, samples=128)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("QSA_REPLAY_GPU", "0")))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="correctness/resource checks only (default)")
    mode.add_argument("--perf", action="store_true", help="explicitly enable gated per-kernel GPU timing")
    parser.add_argument("--output", type=Path, default=os.environ.get("QSA_REPLAY_OUTPUT"),
                        help="new result subdirectories under mytest/mydata; defaults to QSA_REPLAY_OUTPUT")
    parser.add_argument("--buffers", type=int, default=10)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--scopes", nargs="+", choices=PREFILL_LABELS + DECODE_LABELS,
                        help="only these kernel labels; cases without a requested launch are omitted")
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output or QSA_REPLAY_OUTPUT is required")
    if args.gpu < 0 or args.buffers < 1 or args.samples < args.buffers:
        parser.error("Require gpu >= 0 and samples >= buffers >= 1")
    if args.scopes is not None and len(set(args.scopes)) != len(args.scopes):
        parser.error("--scopes must not contain duplicates")
    if not args.output.resolve().is_relative_to(DATA.resolve()):
        parser.error("Results must stay under mytest/mydata")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QSA_REPLAY_GPU"] = str(args.gpu)
    device = _gpu()
    jobs = []
    for seq_lens, extend_lens in PREFILL_PERF_CASES if args.perf else PREFILL_CASES:
        available = prefill_labels(seq_lens, extend_lens)
        scopes = available if args.scopes is None else tuple(s for s in args.scopes if s in available)
        if scopes:
            jobs.append((_prefill_name(seq_lens, extend_lens),
                         lambda buffer, seq=seq_lens, ext=extend_lens: make_prefill_case(seq, ext, device, seed=41 + buffer),
                         scopes))
    for lengths, padding, context in DECODE_PERF_CASES if args.perf else DECODE_CASES:
        scopes = DECODE_LABELS if args.scopes is None else tuple(s for s in args.scopes if s in DECODE_LABELS)
        if scopes:
            jobs.append((_decode_name(lengths, padding, context),
                         lambda buffer, n=lengths, p=padding, c=context: make_decode_case(
                             n, device, padding=p, context=c, seed=73 + buffer), scopes))
    with recording_matrix(args.output, [name for name, _, _ in jobs]) as reports:
        for name, make_case, scopes in jobs:
            reports[name] = measure_components(make_case, args.output / name, device.index,
                                               buffers=args.buffers, warmup=2, samples=args.samples,
                                               check_only=not args.perf, scopes=scopes)


if __name__ == "__main__":
    main()
