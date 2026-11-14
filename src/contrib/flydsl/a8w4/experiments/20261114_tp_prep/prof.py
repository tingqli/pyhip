"""tp_prep alone (no GEMMs): kernel time, cold cache (flush, as kernel_times) and warm,
for aiter (moe_sorting + quantize_rows), v1 (tp_prep_v1.py) and v2 (tp_prep.py) with
optional parts left out (SKIP diagnostics: the outputs are then invalid).

    cd src/contrib/flydsl
    HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261114_tp_prep/prof.py 1 1024 8192 \
        --variants "aiter;v1;v2;v2-pad;v2-quant;v2-sort"

Variants: "aiter", "v1", "v2a" (tp_prep_v2a.py), "v2" (tp_prep.py = v2b) with optional
"r<N>" (N routes per sort WG), "n<N>" (>= N sort WGs), "u<N>" (histogram loads per
thread per round trip) and "-<part>" (left out: hist scatter pad sort quant), e.g.
"v2n8u4-pad".
"""
import argparse
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../20261114_r2_baseline"))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels.row_quant import quantize_rows  # noqa: E402
from a8w4.moe import _sort  # noqa: E402
import tp_prep as TPP  # noqa: E402
import tp_prep_v1 as TPP1  # noqa: E402
import tp_prep_v2a as TPP2A  # noqa: E402
from breakdown import kernel_times  # noqa: E402

SHAPE = (4096, 256, 256, 256, 8)
H, I, E, EGL, K = SHAPE


def make(v, x, tw, ti, bm):
    if v == "aiter":
        return lambda: (_sort(ti, tw, SHAPE, 0, bm), quantize_rows(x))
    if v == "v1":
        return lambda: TPP1.tp_prep(ti, tw, x, block_m=bm, num_experts=E)
    if v == "v2a":
        return lambda: TPP2A.tp_prep(ti, tw, x, block_m=bm, num_experts=E)
    m = re.fullmatch(r"v2(?:r(\d+))?(?:n(\d+))?(?:u(\d+))?((?:-[a-z]+)*)", v)
    assert m, v
    kw = dict(skip=",".join(m[4].split("-")[1:]))
    for key, g in (("rng", m[1]), ("nsmin", m[2]), ("hu", m[3])):
        if g:
            kw[key] = int(g)
    return lambda: TPP.tp_prep(ti, tw, x, block_m=bm, num_experts=E, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", type=int, nargs="+")
    ap.add_argument("--variants", default="aiter;v1;v2")
    ap.add_argument("--bm", type=int, default=0, help="0: the tuned config's block_m")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    variants = a.variants.split(";")
    for T in a.tokens:
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        bm = a.bm or (32 if T <= 16 else 64)
        cells = []
        for v in variants:
            fn = make(v, x, tw, ti, bm)
            cold = sum(kernel_times(fn, iters=a.iters).values())
            warm = R.time_cold(fn, iters=a.iters, flush=False)
            cells.append(f"{v} {cold:.1f}/{warm:.1f}")
        print(f"T={T:5d} bm={bm} kernel us cold/warm-e2e:  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
