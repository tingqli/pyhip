#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only swizzle proof/model; block%XCD is an assumption, not a measurement."""
import argparse
from collections import Counter
import json


def transpose(v, tasks, width):
    chunk = tasks // width
    return v % width * chunk + v // width if v < chunk * width else v


def verify(tasks, width):
    assert tasks >= 0 and width > 0
    mapped = [transpose(v, tasks, width) for v in range(tasks)]
    assert sorted(mapped) == list(range(tasks))
    chunk = tasks // width
    for v, task in enumerate(mapped):
        inverse = task % chunk * width + task // chunk if task < chunk * width else task
        assert inverse == v
    assert all(transpose(v, tasks, width) == v for v in range(tasks, tasks + width + 1))


def self_test():
    cases = 0
    for tasks in (0, 1, 3, 4, 7, 8, 9, 31, 32, 33, 127, 129, 3072, 9256):
        for width in (1, 2, 4, 8, 16, 32):
            verify(tasks, width)
            cases += 1
    for packets in (2, 4, 6, 10, 12, 16):
        for phase in range(packets):
            order = [2 * ((q // 2 + phase) % (packets // 2)) + q % 2 for q in range(packets)]
            assert sorted(order) == list(range(packets))
            assert all(order[q + 1] == order[q] + 1 for q in range(0, packets, 2))
    print("CPU_SWIZZLE_SELF_TEST_PASS", cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=1157)
    parser.add_argument("--oc-splits", type=int, default=8)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--xcds", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.blocks < 0 or min(args.oc_splits, args.width, args.xcds) < 1:
        parser.error("blocks must be nonnegative; splits, width and xcds must be positive")
    tasks = args.blocks * args.oc_splits
    verify(tasks, args.width)
    domains = [set() for _ in range(args.blocks)]
    examples = []
    for v in range(tasks):
        task = transpose(v, tasks, args.width)
        block, split = divmod(task, args.oc_splits)
        domains[block].add(v % args.xcds)
        if v % args.xcds == 0 and len(examples) < 8:
            examples.append({"virtual": v, "task": task, "m_block": block, "oc": split})
    print(json.dumps({"tasks": tasks, "width": args.width,
                      "assumption": f"physical XCD = virtual block modulo {args.xcds}; NOT measured",
                      "a_domain_histogram": dict(sorted(Counter(map(len, domains)).items())),
                      "modeled_xcd0_first_tasks": examples}, indent=2))


if __name__ == "__main__":
    main()