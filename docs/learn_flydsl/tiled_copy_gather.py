import os
import flydsl
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly

import pyhip
import pyhip.contrib.flydsl.utils as fxu

# fxu.enable_dump_ir()

def test_tiled_gather_rows(M, N, tileM, tileN, num_waves):
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def kernel(A: fx.Tensor, B: fx.Tensor, sorted_row_idx: fx.Tensor):

        sorted_row_idx = fx.Tensor(
            fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(sorted_row_idx)), sorted_row_idx.layout)
        )
        
        bx = fx.block_idx.x
        by = fx.block_idx.y
        tid = fx.thread_idx.x

        assert A.dtype == B.dtype

        sorted_row_idx = sorted_row_idx[None, bx]   # Tensor<f32, global, tileM:1>
        B = B[None, (bx, by)]                       # Tensor<f32, global, (tileM,tileN):(?{i64},1)>

        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        VECT_WIDTH = fxu.div_e(copy_bits, A.dtype.width)

        # ======= coealescing TV-layout =======
        threads_n = tileN//VECT_WIDTH
        threads_m = num_threads//threads_n
        tv_tilemn = (threads_m, tileN)
        tv_layout = fxu.make_layout_nd(((threads_n, threads_m), VECT_WIDTH),
                                        tv_tilemn,
                                        (([0, VECT_WIDTH], [1, 0]), [0, 1]))        

        tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tv_tilemn)

        sorted_row_tile = fx.Tensor(fx.make_view(
                            fx.get_iter(sorted_row_idx),
                            fx.make_layout((tileM, tileN), (1, 0))))

        sorted_row_tile = tiled_copy.get_slice(tid).partition_S(sorted_row_tile)
        B = tiled_copy.get_slice(tid).partition_D(B)
        
        sorted_row_tile = fx.group(sorted_row_tile, 1, -1)
        B = fx.group(B, 1, -1)

        num_tv_tiles = fx.size(B.layout[1]).get_static_leaf_int

        # suppose Tile size is fixed & not too big
        # pre-load all row-indexes for current threads
        row_idx = [sorted_row_tile[0, i] for i in fx.range_constexpr(num_tv_tiles)]

        # according to tv-layout, all values of a thread appears within same row, so
        # we can transform the coordinate in unit of TV-tile
        for i in fx.range_constexpr(num_tv_tiles):
            off = fx.crd2idx((tid, 0), tv_layout)
            col = fx.get_scalar(off) // tv_tilemn[0]
            row = row_idx[i]
            coord = fx.make_coord(row, col + by* tileN)
            iter = fx.add_offset(fx.get_iter(A), fx.crd2idx(coord, A.layout))
            atom_A = fx.make_view(iter, B.layout[0])
            fx.copy(copy_atom, atom_A, B[None, i])

    @flyc.jit
    def test(A: fx.Tensor, B: fx.Tensor, sorted_row_idx: fx.Tensor, stream):

        assert A.stride[1].get_static_leaf_int == 1 and B.stride[1].get_static_leaf_int == 1, \
              "kernel assumes 2nd mode is contiguous"

        assert A.leading_dim == 1 and B.leading_dim == 1, \
              "kernel assumes 2nd mode is contiguous"
        B = fx.zipped_divide(B, fx.make_tile(tileM, tileN))
        sorted_row_idx = fx.zipped_divide(sorted_row_idx, fxu.make_tile(tileM))

        #print("before recast_iter: ", sorted_row_idx)
        #print("after recast_iter: ", sorted_row_idx)

        #print("========", sorted_row_idx)
        #print("========", B)

        grid_m = fx.get_scalar(B.shape[1][0])
        grid_n = fx.get_scalar(B.shape[1][1])

        # fx.printf("grid_m: {}, grid_n: {}", grid_m, grid_n)

        kernel(A, B, sorted_row_idx).launch(
            grid=(grid_m, grid_n, 1),
            block=(num_threads, 1, 1), stream=stream
        )

    _, stream = pyhip.set_device()
    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    sorted_row_idx = torch.randint(low=0, high=M, size=(M,), dtype=torch.int32, device="cuda")
    print("sorted_row_idx: ", sorted_row_idx)
    print(sorted_row_idx)
    test(A, B, sorted_row_idx, stream)
    assert torch.allclose(A[sorted_row_idx], B)

    _, us = pyhip.run_perftest(test, A, B, sorted_row_idx, stream, num_iters=10, num_warmup=2, num_bytes=M*N*4*2)
    assert torch.allclose(A[sorted_row_idx], B)

#test_tiled_gather_rows(128, 128, 128, 128, 4)
test_tiled_gather_rows(4096, 4096, 128, 128, 4)

