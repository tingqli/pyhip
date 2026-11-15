"""L2 (TCC) / memory read counters of the EP gemms per W cache modifier (``bcm``: 2 = nt,
0 = cached), to explain the 20261115 timing results. Runs every variant ``--reps``
times, each after an L2 + MALL flush; under rocprofv3 (one pass per counter group):

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 rocprofv3 \
        --pmc TCC_HIT_sum TCC_MISS_sum TCC_READ_sum TCC_STREAMING_REQ_sum \
        --pmc TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_RDREQ_DRAM_sum \
        -d OUT -o c --output-format csv -- \
        python a8w4/experiments/20261115_ep_prio/counters.py 1024 --base m64
    python a8w4/experiments/20261115_ep_prio/counters.py --parse OUT

Variants (W loads gemm1 / gemm2): nt / nt (all configs before 20261115), nt / cached,
cached / cached; PF as the base config. The kernel names carry bcm, so the dispatches
of a variant are told apart by name.
"""
import argparse
import copy
import csv
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

H, I, E, EGL, K = 4096, 2048, 32, 256, 8
VARIANTS = {"nt_nt": (2, 2), "nt_c": (2, 0), "c_c": (0, 0)}


def run(tokens, base, reps):
    import torch

    from a8w4 import ref as R
    from a8w4.kernels import ep_gemm as EG

    W, _ = R.make_weights(E, H, I)
    flush = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    for T in tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        for b1, b2 in VARIANTS.values():
            c = copy.deepcopy(EG.CFGS[base])
            c["g1"]["bcm"], c["g2"]["bcm"] = b1, b2
            for _ in range(reps):
                flush.zero_()
                EG.ep_moe(x, W, tw, ti, e0=0, adt="fp8", cfg=c)
        torch.cuda.synchronize()


def parse(out_dir):
    """Mean per dispatch of every counter, per kernel name (gemm1 / gemm2 only)."""
    acc = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(out_dir, "**", "*counter_collection.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                name = r["Kernel_Name"]
                if "gemm" not in name:
                    continue
                key = (name, r["Dispatch_Id"])
                acc[name][r["Counter_Name"]].append((key, float(r["Counter_Value"])))
    for name in sorted(acc):
        short = re.sub(r"\(.*", "", name)[-90:]
        vals = {c: sum(v for _, v in xs) / len(xs) for c, xs in sorted(acc[name].items())}
        print(short)
        print("   " + "  ".join(f"{c} {v / 1e6:.2f}M" for c, v in vals.items()))
        hit, miss = vals.get("TCC_HIT_sum"), vals.get("TCC_MISS_sum")
        if hit is not None and miss:
            print(f"   L2 hit rate {hit / (hit + miss) * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int, nargs="*")
    ap.add_argument("--base", default="m64")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parse", default=None, help="rocprofv3 output dir")
    a = ap.parse_args()
    if a.parse:
        parse(a.parse)
    else:
        run(a.tokens, a.base, a.reps)


if __name__ == "__main__":
    main()
