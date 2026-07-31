
"""
这个小脚本证实使用32x32版本的MFMA时，C矩阵的行和可以通过一次shuffle_xor操作来计算。
而使用16x16版本的MFMA时，C矩阵的行和需要两次shuffle_xor操作来计算。
这可以进一步节省online-softmax中的计算开销。

例如:
 - 16x16版本，32x32的区域需要 4 个 16x16 MFMA_C, 使用 in-thread reduce 之后,
   一共有 2x16x4个值，需要4次shuffle_xor.
 - 32x32版本，32x32的区域只需要 1 个 32x32 MFMA_C, 使用 in-thread reduce 之后，
   一共有 2x32个值，单次shuffle_xor就可以完成行和的计算。

"""
import os

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm

import pyhip.contrib.flydsl.helpers as fxh

fxh.dump_ir(True)

import pyhip
pyhip.set_device()


@pyhip.fly
def test(A:fx.Tensor, B:fx.Tensor, C:fx.Tensor, C_rowsum:fx.Tensor=None):
    tid = fx.thread_idx.x
    flyobj = fxh.FlyObjCache()
    print(A, B, C)
    dtype = A.dtype
    tmma1 = flyobj.create_thr_mma(dtype, (1, 1, 1), 32)
    #tmma1 = flyobj.create_thr_mma(dtype, (1, 1, 1), (16,16,32,1))
    #tmma1 = flyobj.create_thr_mma(dtype, (1, 1, 1))
    print(f"tmma1.tile_size_mnk: {tmma1.tile_size_mnk}: make sure M/N/K is consistent!!!!")

    fragA = flyobj.load_tiled_mma_fragA(tmma1, A, [None, None], copy_atom_bits=128)
    fragB = flyobj.load_tiled_mma_fragB(tmma1, B, [None, None], copy_atom_bits=128)
    fragC = flyobj.load_tiled_mma_fragC(tmma1, C, [None, None], copy_atom_bits=32)
    fragC.fill(0.0)

    # B @ A -> C.T
    fx.gemm(tmma1, fragC, fragB, fragA, fragC)

    # store to C.T
    flyobj.store_tiled_mma_fragC(tmma1, fragC, fx.select(C, [1,0]), copy_atom_bits=128)

    # 32x32 MFMA input layout, only two lanes per row, one shuffle_xor is enough to reduce the row sum
    vrowsum = fragC.load().reduce("add")
    vrowsum = vrowsum + vrowsum.shuffle_xor(32, 64)
    C_rowsum[tid] = vrowsum

ab_dtype = torch.float8_e4m3fnuz
#ab_dtype = torch.bfloat16

A = torch.randn(32, 32, dtype=torch.float32).to(ab_dtype)
B = torch.randn(32, 32, dtype=torch.float32).to(ab_dtype)
C = torch.zeros(32, 32, dtype=torch.float32)
C_rowsum = torch.zeros(32, dtype=torch.float32)

test([1], [64], A, B, C, C_rowsum)

ref = A.float() @ B.float().T
ret = C
pyhip.allclose(ref, ret)

rowsum_ref = C.sum(1)
rowsum_ret = C_rowsum
print(rowsum_ref, rowsum_ret)

pyhip.allclose(rowsum_ref, rowsum_ret)