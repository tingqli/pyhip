# SPDX-License-Identifier: MIT

import torch

if __name__ == "__main__":
    # disable cache at develop time
    import os
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

import pyhip
from pyhip.contrib.flydsl import helpers as fxh

if __name__ == "__main__":
    fxh.dump_ir(False)


class ROCDLBuffer:
    """Buffer-resource view for direct global-to-LDS copies of dwords."""

    def __init__(self, base_tensor):
        num_elements = fx.size(base_tensor).to_py_value()
        self.base_offset = fx.ptrtoint(fx.get_iter(base_tensor))
        self.rsrc = fx.buffer_ops.create_buffer_resource_from_addr(
            self.base_offset,
            num_records_bytes=num_elements * base_tensor.dtype.width // 8,
        )

    @flyc.jit
    def load_async(
        self,
        ptr_src: fx.Pointer,
        ptr_dst: fx.Pointer,
        num_dwords: int,
        num_threads: int,
    ):
        """Cooperatively copy contiguous dwords from global memory to LDS.

        ``ptr_src`` must point inside the tensor used to construct this object;
        its descriptor-relative byte offset is calculated automatically.
        ``ptr_dst`` must point to compact LDS storage. No LDS bank swizzle is
        applied, so preshuffled input is copied byte-for-byte.

        Every full round uses one 16-byte DMA per thread. A partial 16-byte
        round is predicated by thread ID, and up to three residual dwords are
        copied by the first lanes of wave 0. The caller owns async group commit,
        waiting, and the workgroup barrier.
        """
        assert num_dwords > 0
        assert num_threads > 0 and num_threads % 64 == 0

        atom_bytes = 16
        dword_bytes = 4
        num_bytes = num_dwords * dword_bytes
        num_atoms = num_bytes // atom_bytes
        num_rounds = num_atoms // num_threads
        tail_atoms = num_atoms % num_threads
        tail_dwords = (num_bytes % atom_bytes) // dword_bytes

        tid = fx.thread_idx.x
        wave = tid // fx.Int32(64)
        dst_base_ptr = fx.buffer_ops.create_llvm_ptr(
            fx.ptrtoint(ptr_dst),
            address_space=3,
        )
        zero_i32 = arith._to_raw(fx.Int32(0))
        wave_byte_offset = fx.rocdl.readfirstlane(
            T.i32,
            arith._to_raw(wave * fx.Int32(64 * atom_bytes)),
        )
        lds_ptr = fx.buffer_ops.get_element_ptr(
            dst_base_ptr,
            wave_byte_offset,
            0,
            T.i8,
        )
        src_base_byte_offset = fx.Int32(
            fx.ptrtoint(ptr_src) - self.base_offset
        )
        source_byte_offset = (
            src_base_byte_offset + tid * fx.Int32(atom_bytes)
        )

        # Full rounds: every thread copies one 16-byte atom.
        for copy_round in range_constexpr(num_rounds):
            rocdl.raw_ptr_buffer_load_async_lds(
                self.rsrc,
                lds_ptr,
                arith._to_raw(fx.Int32(atom_bytes)),
                arith._to_raw(source_byte_offset),
                zero_i32,
                zero_i32,
                zero_i32,
            )
            lds_ptr = fx.buffer_ops.get_element_ptr(
                lds_ptr,
                None,
                num_threads * atom_bytes,
                T.i8,
            )
            source_byte_offset += num_threads * atom_bytes

        # Partial 16-byte round.
        if fx.const_expr(tail_atoms != 0):
            if tid < fx.Int32(tail_atoms):
                rocdl.raw_ptr_buffer_load_async_lds(
                    self.rsrc,
                    lds_ptr,
                    arith._to_raw(fx.Int32(atom_bytes)),
                    arith._to_raw(source_byte_offset),
                    zero_i32,
                    zero_i32,
                    zero_i32,
                )

        # Up to three residual dwords after the final 16-byte atom.
        if fx.const_expr(tail_dwords != 0):
            dword_source_base = (
                src_base_byte_offset
                + fx.Int32((num_bytes // atom_bytes) * atom_bytes)
            )
            dword_lds_ptr = fx.buffer_ops.get_element_ptr(
                lds_ptr,
                None,
                tail_atoms * atom_bytes,
                T.i8,
            )
            if tid < fx.Int32(tail_dwords):
                rocdl.raw_ptr_buffer_load_async_lds(
                    self.rsrc,
                    dword_lds_ptr,
                    arith._to_raw(fx.Int32(dword_bytes)),
                    arith._to_raw(
                        dword_source_base + tid * fx.Int32(dword_bytes)
                    ),
                    zero_i32,
                    zero_i32,
                    zero_i32,
                )


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

    A = torch.randn(source_prefix + num_elements, dtype=torch.float32)
    B = torch.zeros(num_elements, dtype=torch.float32)
    kernel([1], A, B)
    torch.cuda.synchronize()
    torch.testing.assert_close(B, A[source_prefix : source_prefix + num_elements])

if __name__ == "__main__":
    test_buffer_load_lds()