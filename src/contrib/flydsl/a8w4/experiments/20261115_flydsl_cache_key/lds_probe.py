"""What happens when a FlyDSL kernel's static LDS exceeds 64 KB on gfx942? (FlyDSL 0.3.2)

    cd src/contrib/flydsl && for n in 65536 65552; do FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d) \\
        HIP_VISIBLE_DEVICES=0 python a8w4/experiments/20261115_flydsl_cache_key/lds_probe.py $n; done

Prints "ran: <value>"; the kernel writes N + the LDS address when it runs, so "ran: 0" means
the launch failed WITHOUT a Python exception (see the stderr lines above it).
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu

from a8w4.experiments._launch import _run_compiled
from a8w4.kernels._mem import _pointer

N = int(sys.argv[1])


@fx.struct
class S:
    raw: fx.Array[fx.Uint8, N, 16]


@flyc.kernel(known_block_size=(64, 1, 1))
def lds_kernel(out: fx.Int64):
    lds = fx.SharedAllocator().allocate(S).peek().raw.ptr
    tid = fx.Int32(gpu.thread_id("x"))
    if tid == fx.Int32(0):
        fx.ptr_store(fx.Int32(fx.ptrtoint(lds)) + fx.Int32(N), _pointer(out, fx.Int32, 4))


@flyc.jit
def launch(out: fx.Int64, stream: fx.Stream):
    lds_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


out = torch.zeros(1, dtype=torch.int32, device="cuda")
try:
    _run_compiled(launch, out.data_ptr(), torch.cuda.current_stream())
    torch.cuda.synchronize()
    print("ran:", int(out.item()))
except Exception as e:  # noqa: BLE001
    print("ERROR", type(e).__name__, str(e)[:300])
