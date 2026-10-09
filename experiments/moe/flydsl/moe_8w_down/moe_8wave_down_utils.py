# SPDX-License-Identifier: MIT
"""Round-trip test for the packaged global-to-LDS copy helper."""

import torch
import flydsl.expr as fx
from flydsl.expr import range_constexpr

import pyhip
from pyhip.ops.moe.flydsl.moe_8wave_down_utils import ROCDLBuffer


def test_buffer_load_lds():
    """Round-trip a non-aligned slice through LDS and compare exactly."""
    pyhip.set_device()

    num_threads = 512
    # Eight full rounds, one 16-byte tail atom, and three dword tail atoms.
    num_elements = num_threads * 16 * 2 + 4 + 3
    source_prefix = 1024

    @pyhip.fly(num_threads)
    def kernel(global_src, global_dst):
        tid = fx.thread_idx.x

        @fx.struct
        class SharedStorage:
            lds_tile: fx.Array[fx.Float32, num_elements, 16]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_tile = fx.make_view(
            lds.lds_tile.ptr,
            fx.make_layout(num_elements, 1),
        )

        buff = ROCDLBuffer(global_src)
        buff.load_async(
            fx.get_iter(global_src) + source_prefix,
            fx.get_iter(lds_tile),
            num_elements,
            num_threads,
        )

        fx.rocdl.asyncmark()
        fx.rocdl.wait_asyncmark(0)
        fx.barrier()

        num_store_rounds = (num_elements + num_threads - 1) // num_threads
        for copy_round in range_constexpr(num_store_rounds):
            index = tid + fx.Int32(copy_round * num_threads)
            if index < fx.Int32(num_elements):
                global_dst[index] = lds_tile[index]

    a = torch.randn(source_prefix + num_elements, dtype=torch.float32)
    b = torch.zeros(num_elements, dtype=torch.float32)
    kernel([1], a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(b, a[source_prefix:source_prefix + num_elements])


if __name__ == "__main__":
    test_buffer_load_lds()