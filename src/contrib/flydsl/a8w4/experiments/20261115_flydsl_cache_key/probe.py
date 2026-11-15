"""Probe kernel of run_probe.py (copied to a temp dir there, next to helper_same.py and
other/helper_other.py). Prints the value the kernel wrote.

    python probe.py <lst[0]> <scalar> [tuple]
"""

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "other"))
from a8w4._vendor.a8w4_ops import _pointer  # noqa: E402
from a8w4._vendor.tensor_shim import _run_compiled  # noqa: E402
from helper_other import helper_other  # noqa: E402
from helper_same import helper_same  # noqa: E402

MODE = 1


def make(lst, scalar):
    @flyc.kernel(known_block_size=(64, 1, 1))
    def probe_kernel(out: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        v = fx.Int32(1000 * MODE + 100 * lst[0] + 10 * scalar + helper_other() + helper_same())
        if tid == fx.Int32(0):
            fx.ptr_store(v, _pointer(out, fx.Int32, 4))

    @flyc.jit
    def launch(out: fx.Int64, stream: fx.Stream):
        probe_kernel(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


if __name__ == "__main__":
    lst = (int(sys.argv[1]),) if len(sys.argv) > 3 else [int(sys.argv[1])]
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _run_compiled(make(lst, int(sys.argv[2])), out.data_ptr(), torch.cuda.current_stream())
    torch.cuda.synchronize()
    print(int(out.item()))
