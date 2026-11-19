"""Precision of MegaMoeLargeEpA8W4's x quant (qx 0 = x_quant per_token: one scale per row,
qx 1 = per_1x128: per (row, 128)) against the fp32 reference on the UNQUANTIZED x (x bf16 ->
f32; only the intermediate is quantized as the kernels do). Also with channel
outliers in x (``--outliers``: 1/64 of the channels scaled by 30, as LLM hidden states have).

    python a8w4/experiments/20261118_mega_large/precision_qx.py -t 2048 8192 [--outliers]
        [--act situv2]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from a8w4 import ref as R  # noqa: E402
from a8w4.bench import dist as D  # noqa: E402
from a8w4.bench.dist import E_GLOBAL, E_LOCAL, H, I, TOPK, WORLD  # noqa: E402


def worker(rank, port, points, outliers, act):
    D.init(rank, port)
    from a8w4.megamoe import MegaMoeLargeEpA8W4

    situ = R.SITU if act == "situv2" else None
    W, raw = R.make_weights(E_LOCAL, H, I, seed=rank)
    m = MegaMoeLargeEpA8W4(W, rank=rank, max_tokens=4096, topk=TOPK)
    for ns in points:
        n = ns[rank]
        x, tw, ti = R.make_routing(n, H, E_GLOBAL, TOPK, seed=7919 * rank + n)
        if outliers:
            g = torch.Generator(device="cuda").manual_seed(123)  # the same channels everywhere
            ch = torch.randperm(H, device="cuda", generator=g)[:H // 64]
            x = x.clone()
            x[:, ch] *= 30
        X, TW, TI = (D.gather_rows(t, ns) for t in (x.float(), tw, ti))
        ref = R.ref_moe(X, TW, TI, raw, rank * E_LOCAL, act, situ, quant_x=False)
        dist.all_reduce(ref)
        off = sum(ns[:rank])
        res = {}
        for qx, xq in ((0, "per_token"), (1, "per_1x128")):
            y = m(x, ti, tw, act=act, situ=situ, x_quant=xq, bcm1=0, bcm2=1)[:n].float()
            res[qx] = R.cos_diff(ref[off:off + n], y) if n else 0.0
        allr = [None] * WORLD
        dist.all_gather_object(allr, res)
        if rank == 0:
            print(f"{act} tokens/rank {ns[0]}{' outliers' if outliers else ''}: cos diff vs the "
                  f"fp32 reference on unquantized x, max over ranks: qx0 (per row) "
                  f"{max(r[0] for r in allr):.2e}, qx1 (per 128) {max(r[1] for r in allr):.2e}",
                  flush=True)
    D.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--outliers", action="store_true")
    ap.add_argument("--act", default="silu")
    a = ap.parse_args()
    D.spawn(worker, ([D.split_tokens(t) for t in a.tokens], a.outliers, a.act))
