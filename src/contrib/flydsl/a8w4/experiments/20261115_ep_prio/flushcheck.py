"""Which part of ``ref.time_cold`` lets cached W loads (bcm 0) look faster than in a
model? Protocols, same configs, interleaved (rotate.py showed the gap):

  cold     ref.time_cold: 512 MB read flush + 1 ms GPU sleep, the same W every call
  coldrot  same flush + sleep, but each call reads the next of 8 W copies (3.3 GB)
  wflush   512 MB write (zero_) + 512 MB read flush + sleep, the same W
  bigread  2 GB read flush + sleep, the same W
  rotate   8 W copies back to back, no flush (as consecutive layers)

    cd src/contrib/flydsl && HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_ep_prio/flushcheck.py
"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from a8w4 import ref as R  # noqa: E402
from a8w4.kernels import ep_gemm as EG  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from rotate import NEW, old, rotate_pass  # noqa: E402

H, I, E, EGL, K = 4096, 2048, 32, 256, 8
CASES = {16: ("t16", old("t16"), "t16c", NEW["t16c"]),
         256: ("s16 old", old("s16"), "s16 new", NEW["s16"]),
         1024: ("m64 old", old("m64"), "m64 new", NEW["m64"])}
BIG = {}


def timed(fn, pre):
    pre()
    torch.cuda._sleep(2_000_000)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3


def wflush():
    if "w" not in BIG:
        BIG["w"] = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    BIG["w"].zero_()
    R.flush_cache()


def bigread():
    if "r" not in BIG:
        BIG["r"] = torch.ones(512 << 20, dtype=torch.int32, device="cuda")  # 2 GB
    BIG["r"].amax()


def main():
    W0, _ = R.make_weights(E, H, I)
    Ws = [W0] + [{k: v.clone() if torch.is_tensor(v) else v for k, v in W0.items()}
                 for _ in range(7)]
    for T, (na, ca, nb, cb) in CASES.items():
        x, tw, ti = R.make_routing(T, H, EGL, K, seed=7 + T)
        res = {}
        for rnd in range(5):
            for n, c in ((na, ca), (nb, cb)) if rnd % 2 == 0 else ((nb, cb), (na, ca)):
                fW = lambda W, c=c: EG.ep_moe(x, W, tw, ti, e0=0, adt="fp8", cfg=c)  # noqa: E731
                if rnd == 0:
                    rotate_pass([lambda W=W: fW(W) for W in Ws])
                r = res.setdefault(n, {})
                r.setdefault("cold", []).extend(timed(lambda: fW(W0), R.flush_cache) for _ in range(8))
                r.setdefault("coldrot", []).extend(timed(lambda W=W: fW(W), R.flush_cache) for W in Ws)
                r.setdefault("wflush", []).extend(timed(lambda: fW(W0), wflush) for _ in range(8))
                r.setdefault("bigread", []).extend(timed(lambda: fW(W0), bigread) for _ in range(8))
                r.setdefault("rotate", []).extend(rotate_pass([lambda W=W: fW(W) for W in Ws]))
        print(f"== tokens {T}: median us ({nb} vs {na})")
        for p in res[na]:
            a, b = statistics.median(res[na][p]), statistics.median(res[nb][p])
            print(f"  {p:8s} {na} {a:8.1f} | {nb} {b:8.1f} ({(b / a - 1) * 100:+5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
