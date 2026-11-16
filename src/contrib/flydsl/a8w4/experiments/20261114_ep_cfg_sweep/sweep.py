"""EP gemm config sweep around l128/m64 (run from flydsl/: python -m a8w4.experiments.20261114_ep_cfg_sweep.sweep)."""
import copy, sys, torch
sys.path.insert(0, "/sgl-workspace/pyhip/src/contrib/flydsl")
from a8w4 import ref as R, prepare_weights
from a8w4.kernels import ep_gemm as EG

H, I, E, EG_, K = 4096, 2048, 32, 256, 8
W, raw = R.make_weights(E, H, I)

def var(base, g1=None, g2=None, **kw):
    c = copy.deepcopy(EG.CFGS[base])
    c["g1"].update(g1 or {}); c["g2"].update(g2 or {}); c.update(kw); return c

V = {
 "l128": var("l128"),
 "l128_g2nw4": var("l128", g2=dict(nw=4)),
 "l128_g2no128": var("l128", g2=dict(no=128)),
 "l128_g2arb64": var("l128", g2=dict(arb=64)),
 "l128_g2arb256": var("l128", g2=dict(arb=256)),
 "l128_g2sch1": var("l128", g2=dict(sch=1)),
 "l128_g2sch2": var("l128", g2=dict(sch=2)),
 "l128_g1sch2": var("l128", g1=dict(sch=2)),
 "l128_notr": var("l128", g1=dict(tr=False)),
 "l128_nosst": var("l128", sst=False),
 "m64": var("m64"),
 "m64_g2nw4": var("m64", g2=dict(nw=4)),
}
for T in [int(a) for a in sys.argv[1:]] or [2048, 12288]:
    x, tw, ti = R.make_routing(T, H, EG_, K, seed=7 + T)
    ref = None
    print(f"== tokens {T}")
    rows = {}
    for rnd in range(3):  # interleaved rounds
        for n, c in V.items():
            try:
                f = lambda: EG.ep_moe(x, W, tw, ti, e0=0, cfg=c)
                out = f()
                if ref is None:
                    ref = out.float()
                cos = R.cos_diff(out.float(), ref)
                rows.setdefault(n, []).append(R.time_cold(f, iters=10))
            except Exception as e:
                rows[n] = [float("nan")]; print(n, "FAILED", str(e)[:100]); 
    for n, t in rows.items():
        print(f"{n:16s} {min(t):9.1f} us")
