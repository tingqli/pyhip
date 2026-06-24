import os
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
import pyhip
import pyhip.contrib.flydsl.utils as fxu

#fxu.enable_dump_ir(True)
_, stream = pyhip.set_device()

@flyc.kernel
def mma_atom_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, mma_atom):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    atom_mnk = mma_atom.shape_mnk.to_py_value()
    print("mma_atom:", mma_atom)

    copy_atomA = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), A.dtype)
    copy_atomB = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), B.dtype)
    copy_atomC = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)

    copyA = fx.make_tiled_copy(copy_atomA, mma_atom.layout_A_tv, (atom_mnk[0], atom_mnk[2]))
    copyB = fx.make_tiled_copy(copy_atomB, mma_atom.layout_B_tv, (atom_mnk[1], atom_mnk[2]))
    copyC = fx.make_tiled_copy(copy_atomC, mma_atom.layout_C_tv, (atom_mnk[0], atom_mnk[1]))

    print(" mma_atom.layout_A_tv: ", mma_atom.layout_A_tv)
    print("======= partion of A: ", copyA.partition_S(A), (atom_mnk[0], atom_mnk[2]), A)
    # (T64,V2) => (16,8):(8,1)    m,n=8,1
    # ((16,4),(1,1),(1, 2)):
    #  ((m,n),(0,0),(0,4n))

    partA = copyA.get_slice(tid).partition_S(A)
    partB = copyB.get_slice(tid).partition_S(B)
    partC = copyC.get_slice(tid).partition_D(C)

    """
    Why we can directly feed tiled-copy frag into fx.gemm?

    tile-copy will simply stack 1 tile after another
        C: (one_tile, num_tiles_M, num_tiles_N)
        A: (one_tile, num_tiles_M, num_tiles_K)
        B: (one_tile, num_tiles_N, num_tiles_K)

    gemm expects input layouts
        C/D : (c-tile, loop_m, loop_n)    always rank-3
        A  : (a-tile, loop_m, [loop_k])  [.] means optional, if not present, means 1
        B  : (b-tile, loop_n, [loop_k])  [.] means optional, if not present, means 1

    So they are aligned in [mma_atom + tiled_copy] case.

    But when we want to use vector-loads in tiled-copy,
    we need to permute the K-dimension of each tile, which means we need to
    change the layout of the fragments feeding into gemm.
    """
    fragA = fx.make_fragment_like(partA)
    fragB = fx.make_fragment_like(partB)
    fragC = fx.make_fragment_like(partC)
    fragD = fx.make_fragment_like(partC)

    fx.copy(copy_atomA, partA, fragA)
    fx.copy(copy_atomB, partB, fragB)

    fragC.fill(0)

    print("partA:", partA, fragA)
    print("partB:", partB, fragB)
    print("partC:", partC, fragC)

    fxu.gemm(mma_atom, fragD, fragA, fragB, fragC)
    #fx.gemm(mma_atom, fragD, fragA, fragB, fragC)
    fx.copy(copy_atomC, fragD, partC)


@flyc.jit
def mma_atom(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    ATOM_M: fx.Constexpr[int],
    ATOM_N: fx.Constexpr[int],
    ATOM_K: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream):
    A = fxu.view_as(A, fx.make_layout((ATOM_M, K), (K, 1)))
    B = fxu.view_as(B, fx.make_layout((ATOM_N, K), (K, 1)))
    C = fxu.view_as(C, fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)))
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(ATOM_M, ATOM_N, ATOM_K, A.dtype))
    mma_atom_kernel(A, B, C, mma_atom).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)



"""
use b128/DW4 load
"""


@flyc.kernel
def mma_atom_kernel_load_dw4(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    mma_atom,
):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    atom_mnk = mma_atom.shape_mnk.to_py_value()
    copy_atomA = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), A.dtype)
    copy_atomB = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), B.dtype)
    copy_atomC = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), C.dtype)

    """
    mma_atom.layout_A_tv : ((16,4),1):((1,16),16)
    ((16,4),1):((1,16),16)  =>  (16,4):(1,16)  m,n=1,16

    ((16,4),1): 
     ((1,n),n)

    repeat value dimension (16,8):(1,16)  m,n=1,16
    value dimension must be continous, so we can use vector-loads to load 4 values at once

    ((16,  4),4):((1, 32),16)
    ((1, 4*n),n)
    """
    tv_layout_AB = fx.make_layout(((16,4),4),((1,64),16))
    tv_layout_mn = (atom_mnk[0], 4*4)

    copyA = fx.make_tiled_copy(copy_atomA, tv_layout_AB, (atom_mnk[0], 4*4))
    copyB = fx.make_tiled_copy(copy_atomB, tv_layout_AB, (atom_mnk[1], 4*4))
    copyC = fx.make_tiled_copy(copy_atomC, mma_atom.layout_C_tv, (atom_mnk[0], atom_mnk[1]))

    print(copy_atomA, copy_atomA.layout_ref_tv)   #Layout<(1,4):(1,1)>
    print(dir(copy_atomA))  # layout_src_tv_tiled tile_mn 

    print(" mma_atom.layout_A_tv: ", mma_atom.layout_A_tv)
    print("======= partion of A: ", copyA.partition_S(A), (atom_mnk[0], atom_mnk[2]), A)
    # (T64,V2) => (16,8):(8,1)    m,n=8,1
    # ((16,4),(1,1),(1, 2)):
    #  ((m,n),(0,0),(0,4n))

    partA = copyA.get_slice(tid).partition_S(A)
    partB = copyB.get_slice(tid).partition_S(B)
    partC = copyC.get_slice(tid).partition_D(C)

    fragA = fx.make_fragment_like(partA)
    fragB = fx.make_fragment_like(partB)
    fragC = fx.make_fragment_like(partC)
    fragD = fx.make_fragment_like(partC)

    fx.copy(copy_atomA, partA, fragA)
    fx.copy(copy_atomB, partB, fragB)

    fragC.fill(0)

    print("partA:", partA, fragA)
    print("partB:", partB, fragB)
    print("partC:", partC, fragC)

    """
    tile-copy will simply stack 1 tile after another
        C: (one_tile, num_tiles_M, num_tiles_N)
        A: (one_tile, num_tiles_M, num_tiles_K)
        B: (one_tile, num_tiles_N, num_tiles_K)

    gemm expects input layouts
        C/D : (c-tile, loop_m, loop_n)    always rank-3
        A  : (a-tile, loop_m, [loop_k])  [.] means optional, if not present, means 1
        B  : (b-tile, loop_n, [loop_k])  [.] means optional, if not present, means 1

    now the first-mode, `one_tile` has 4 values (DW4) per thread (because tiled-copy loads 16x16 in DW4)
    to feed into gemm, we need to move 4 values out-of the mode `one_tile` into mode `loop_k`

    this is `retile`, split one_tile (which is contiguous) into 4 values, and prepend it into loop_k
    """
    def retile(t):
        t = fx.flat_divide(t, (4,)) # (4, sub-tile, loop_m, loop_k)
        t = fx.select(t, (1,2,0,3)) # (sub-tile, loop_m, 4, loop_k)
        t = fx.group(t, 2, -1)      # (sub-tile, loop_m, (4,loop_k))
        return t
    
    fragA = retile(fragA)
    fragB = retile(fragB)
    print("fragA after retile:", fragA)

    #fxu.gemm(mma_atom, fragD, fragA, fragB, fragC)
    fx.gemm(mma_atom, fragD, fragA, fragB, fragC)
    fx.copy(copy_atomC, fragD, partC)


@flyc.jit
def mma_atom_load_dw4(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    ATOM_M: fx.Constexpr[int],
    ATOM_N: fx.Constexpr[int],
    ATOM_K: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream):
    A = fxu.view_as(A, fx.make_layout((ATOM_M, K), (K, 1)))
    B = fxu.view_as(B, fx.make_layout((ATOM_N, K), (K, 1)))
    C = fxu.view_as(C, fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)))
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(ATOM_M, ATOM_N, ATOM_K, A.dtype))
    mma_atom_kernel_load_dw4(A, B, C, mma_atom).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def test_mma_atom(func_jit, ATOM_M, ATOM_N, ATOM_K, dtype):
    M = ATOM_M
    N = ATOM_N
    K = ATOM_K * 4

    dtype= torch.float32
    #dtype= torch.bfloat16
    A = torch.randn(M, K, dtype=dtype).cuda()
    B = torch.randn(N, K, dtype=dtype).cuda()
    C = torch.zeros(M, N, dtype=dtype).cuda()

    func_jit(A, B, C, ATOM_M, ATOM_N, ATOM_K, K, stream=stream)

    R = A @ B.T
    is_correct = torch.allclose(C, R, atol=1e-5, rtol=1e-5)
    print("Result correct:", is_correct)
    if not is_correct:
        print("Max diff:", (C - R).abs().max().item())


# test_mma_atom(mma_atom, 16, 16, 4, torch.float32)
test_mma_atom(mma_atom_load_dw4, 16, 16, 4, torch.float32)

