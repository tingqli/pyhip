import os
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly

import pyhip
import fx_utils as fxu

# fxu.enable_dump_ir(True)


# 描述 pipeline 时常常需要多个线程合作，按照 tile 访问数据
# 需要多线程协同工作的时候，只有确定好了每个线程每条指令要做的事情，才能生成这个指令
# 根本目的还是要生成我们要的指令序列，完成一个imperaive的命令, 本质上我们还是在做一个imperative的命令
# 但是每条这样的imperative指令都需要综合考虑：
#      所有的wave都会参与执行这条指令，因此需要先获取切分方案
#
# tv-layout描述的就是全部线程都执行一条 atom load 指令时，从work-group角度，所有的线程究竟完成了一个什么操作
# 答案是一个tile的数据的读入，并且读入的结果以什么方式分布在各个线程的寄存器中（由于有转置读取指令，不是线程
# 给什么地址最终就得到什么数据）。
# 换句话说，读入到fragment之后，这个fragment其实也有layout，所有线程的数据在寄存器中的分布，合起来才是完整的tile数据
# 一旦进入kernel之后，我们就不再关心外部tensor的layout了，我们只是假定外部的逻辑layout，例如MOE的专家权重矩阵完全可以preshuffle
# 如果要使用某个copy_atom, 我们就必须传入一个tensor切片表达源的layout, fx.copy本质上就是copy1d，因此可以说，layout系统的
# 最大受益者就是数据搬运，外存，寄存器，LDS之间交换数据。
# copy指令本质上可以理解为：把src按照copy_atom的src切片，dst按照copy_atom的dst切片，然后执行copy_atom完成复制
#
# 真的存在一种办法，可以把任意copy_atom的src/dst切片拼接成一个完整的tile,并且保证尽量coalescing吗？
# 这个方法要求不能对copy_atom的切片行为做任何假定，支持任意可能的切片方式，甚至是转置切片。
# 但是这真的是更好的设计方式吗，强制把layout engineering的复杂度提高的好处有那么大吗？
# 适度使用layout algebra, 点到即止，不过度设计，也许是更好的设计方式。
#

"""
 Layout Shape : (M, N, L, ...)
 Tiler Shape  : <TileM, TileN>
 logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
 zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
 tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
 flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
"""


@pytest.mark.parametrize("M, N, num_waves", [
    (4096, 4096, 2),
    (4096, 4096, 4),
    (4096, 4096, 8),
])
@pytest.mark.parametrize("tileM", [64, 128, 256])
@pytest.mark.parametrize("tileN", [64, 128, 256])
def test_tiled_copy_basic(M, N, tileM, tileN, num_waves):
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def kernel(A: fx.Tensor, B: fx.Tensor, expand_copy: fx.Constexpr[bool]):
        bx = fx.block_idx.x
        by = fx.block_idx.y
        tid = fx.thread_idx.x

        assert A.dtype == B.dtype

        A = A[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>
        B = B[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>

        copy_bits = 128

        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        vect_width = fxu.div_e(copy_bits, A.dtype.width)
        vect_width = 8
        print("vect_width: ", vect_width)
        tv_layout = fxu.get_coalescing_tv_layout(tileM, tileN, vect_width, num_threads)

        # tv_layout.show()
        tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout,
                                        fx.make_tile(tv_layout.tvM, tv_layout.tvN))

        part_A = tiled_copy.get_slice(tid).partition_S(A)
        part_B = tiled_copy.get_slice(tid).partition_D(B)

        print("part_A: ", part_A)
        if expand_copy:
            # fxu.recurisve_apply(lambda pA, pB: fx.copy(copy_atom, pA, pB), part_A, part_B)
            #=========================================================================
            # faster version
            frag = fx.make_fragment_like(part_A)
            fxu.recurisve_apply(lambda a, b: fx.copy(copy_atom, a, b), part_A, frag)
            fxu.recurisve_apply(lambda a, b: fx.copy(copy_atom, a, b), frag, part_B)
        else:
            # fx.copy(copy_atom, part_A, part_B)
            #=========================================================================
            # directly copy from global to global like above cannot hide load-latency
            # need to issue all copies from global to register w/o waitting each-other(no dependency),
            # then from register to global(wait before each store inserted by AMDGPU backend)
            #=========================================================================
            frag = fx.make_fragment_like(part_A)
            fx.copy(copy_atom, part_A, frag)
            fx.copy(copy_atom, frag, part_B)
    
    @flyc.jit
    def test(A: fx.Tensor, B: fx.Tensor, expand_copy: fx.Constexpr[bool],stream):
        # Host side can also use layout algera to determine how
        # to split task at block/WG level, and also canonicalize
        # the problem representation, so kernel can handle more general cases,
        # for example：
        #    - more flexible/relaxed A/B layout (including ranks, strides)
        # canonicalization:  kernel assumes 2D form, with 2nd mode contiguous
        assert A.leading_dim == 1 and B.leading_dim == 1, \
              "kernel assumes 2nd mode is contiguous"
        A = fx.zipped_divide(A, fx.make_tile(tileM, tileN))
        B = fx.zipped_divide(B, fx.make_tile(tileM, tileN))
        assert A.stride[0][1].get_static_leaf_int == 1 and B.stride[0][1].get_static_leaf_int == 1, \
              "kernel assumes 2nd mode is contiguous"

        grid_m = fx.get_scalar(A.shape[1][0])
        grid_n = fx.get_scalar(A.shape[1][1])

        kernel(A, B, expand_copy).launch(
            grid=(grid_m, grid_n, 1),
            block=(num_threads, 1, 1), stream=stream
        )

    _, stream = pyhip.set_device()
    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    test(A, B, False, stream)
    assert torch.allclose(A, B, atol=1e-5)
    B[...] = 0
    test(A, B, True, torch.cuda.Stream())
    assert torch.allclose(A, B, atol=1e-5)

    _, us = pyhip.run_perftest(test, A, B, False, stream, num_iters=10, num_warmup=2, num_bytes=M*N*4*2)
    assert torch.allclose(A, B, atol=1e-5)
    _, us = pyhip.run_perftest(test, A, B, True, stream, num_iters=10, num_warmup=2, num_bytes=M*N*4*2)
    assert torch.allclose(A, B, atol=1e-5)

#test_tiled_copy_basic(2*4096, 4096, 64, 128, 4)


"""
what happens inside partition_S/D ?
 - divide src tile into tv-layout tile
 - divide tv-layout tile into copy-atom tile
"""
def test_partion():
    @flyc.jit
    def partition():
        M, K = 512, 8192
        TILE_M, TILE_K = 128, 64
        bid = 2
        A = fx.make_layout((M,K), (K,1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(64), fx.BFloat16)
        #print("test_partion")

        blockA = fx.flat_divide(A, fx.make_tile(TILE_M, TILE_K))
        at_block = fx.slice(blockA, ((None, None, bid, None)))
        tv_layout = fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32))
        at_subtile = fx.tiled_divide(at_block, fx.make_tile(32, 64))
        print("at_subtile", at_subtile)
        atomNumThr, atomNumVal = copy_atom.layout_ref_tv.shape.to_py_value()
        atomTile = fx.make_tile(atomNumThr, atomNumVal)      # tv_layout = Layout<((8,8,4),8):((256,1,8),32)> Tile<[1|8]>
        atomLayoutTV = fx.zipped_divide(tv_layout, atomTile) #             Layout<((1,8),((8,32),1)):((0,32),((256,1),0))>
        refInv = fx.right_inverse(copy_atom.layout_ref_tv)
        ref2trg = fx.composition(refInv, copy_atom.layout_src_tv) # Layout<(1,8):(0,1)>
        print("ref2trg: ", ref2trg)
        copy_atom_tv = fx.composition(atomLayoutTV[0], ref2trg) #  Layout<(1,8):(0,32)> o Layout<(1,8):(0,1)> => Layout<(1,8):(0,32)>
        print("atomLayoutTV[0]: ", atomLayoutTV[0])
        print("copy_atom_tv: ", copy_atom_tv)

        thrval2mn = fxu.concat_modes((copy_atom_tv[0], atomLayoutTV[1][0]),
                                     (copy_atom_tv[1], atomLayoutTV[1][1]))
        print("thrval2mn: ", thrval2mn) # Layout<((1,(8,32)),(8,1)):((0,(256,1)),(32,0))>
        thrval2mn = fx.coalesce(thrval2mn, fx.make_int_tuple((1, fx.make_int_tuple((1,1))))) # Layout<((8,32),(8,1)):((256,1),(32,0))>
        print("thrval2mn coalesce : ", thrval2mn)

        thrval2mn_mode0 = fly.static(fly.TileType.get([thrval2mn.type,])) # make_tile has issue
        at_subtile_tv = fx.composition(at_subtile, thrval2mn_mode0) # Layout<(((8,32),(8,1)),4,1,128) : (((8,8192),(1,0)),262144,0,64)>
        print("at_subtile_tv", at_subtile_tv)
        at_subtile_thread = fx.slice(at_subtile_tv, ((1, None), None, None, None))
        print("at_subtile_thread", at_subtile_thread) # Layout<((8,1),4,1,128):((1,0),262144,0,64)>
        
        # first mode is (copy-atom-value, tv-layout-value), for example, tv-layout allocate 8 values per thread
        # copy-atom can only copy 4 values per thread, the first-mode will be (4,2)

    partition()

# test_partion()


def test_tiled_gather_rows(M, N, tileM, tileN, num_waves):
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def kernel(A: fx.Tensor, B: fx.Tensor, sorted_row_idx: fx.Tensor):
        bx = fx.block_idx.x
        by = fx.block_idx.y
        tid = fx.thread_idx.x

        assert A.dtype == B.dtype

        print(sorted_row_idx)
        sorted_row_idx = sorted_row_idx[None, bx] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>
        print(sorted_row_idx)

        B = B[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>

        copy_bits = 128

        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        vect_width = fxu.div_e(copy_bits, A.dtype.width)

        tv_layout = fxu.get_coalescing_tv_layout(tileM, tileN, vect_width, num_threads)

        # tv_layout.show()
        tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout,
                                        fx.make_tile(tv_layout.tvM, tv_layout.tvN))

        coord_tensor = fx.Tensor(fx.make_view(fx.make_int_tuple(0),
                                              fx.make_layout((tileM, tileN), (1, tileM))))

        part_crd = tiled_copy.get_slice(tid).partition_S(coord_tensor)
        part_B = tiled_copy.get_slice(tid).partition_D(B)

        # since make_tiled_copy() is applied to static layout, all partitions can be known at compile time, 
        # so we can collect all real row-numbers for each atom.
        row_idx = {}
        def collect_rows(pc, idx):
            m = fx.get_scalar(pc[0]) % tileM
            row_idx[idx] = sorted_row_idx[m]
        fxu.recurisve_apply(collect_rows, part_crd, idx=0)

        print("part_A: ", part_crd)
        def copy_atom_call(pc, pB, idx):
            nonlocal A
            # recover coordinate from coord_tensor
            # print("pc: ", pc, "pB: ", pB)
            m, n = fx.get_scalar(pc[0]) % tileM, fx.get_scalar(pc[0]) // tileM
            coord = fx.make_coord(row_idx[idx], n + by* tileN)
            iter = fx.add_offset(fx.get_iter(A), fx.crd2idx(coord, A.layout))
            # here we assume atom copy is 1d continous
            atom_A = fx.make_view(iter, copy_atom.layout_src_tv)
            fx.copy_atom_call(copy_atom, atom_A, pB)
            idx += 1

        # fxu.recurisve_apply(copy_atom_call, part_crd, part_B)
        # ==========================================================
        # faster version, first copy from global to register, then from register to global
        frag = fx.make_fragment_like(part_B)
        print(frag)
        assert 0
        fxu.recurisve_apply(copy_atom_call, part_crd, frag, idx=0)
        fxu.recurisve_apply(lambda a, b: fx.copy(copy_atom, a, b), frag, part_B)

    @flyc.jit
    def test(A: fx.Tensor, B: fx.Tensor, sorted_row_idx: fx.Tensor, stream):

        assert A.stride[1].get_static_leaf_int == 1 and B.stride[1].get_static_leaf_int == 1, \
              "kernel assumes 2nd mode is contiguous"

        assert A.leading_dim == 1 and B.leading_dim == 1, \
              "kernel assumes 2nd mode is contiguous"
        sorted_row_idx = fx.zipped_divide(sorted_row_idx, fx.make_tile(tileM))
        B = fx.zipped_divide(B, fx.make_tile(tileM, tileN))

        print("before recast_iter: ", sorted_row_idx)
        sorted_row_idx = fx.Tensor(
            fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(sorted_row_idx)), sorted_row_idx.layout)
        )
        print("after recast_iter: ", sorted_row_idx)

        print("========", sorted_row_idx)
        print("========", B)

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
    print("sorted_row_idx: ", sorted_row_idx.shape)
    print(sorted_row_idx)
    test(A, B, sorted_row_idx, stream)
    assert torch.allclose(A[sorted_row_idx], B)

    _, us = pyhip.run_perftest(test, A, B, sorted_row_idx, stream, num_iters=10, num_warmup=2, num_bytes=M*N*4*2)
    assert torch.allclose(A[sorted_row_idx], B)

#test_tiled_gather_rows(128, 128, 128, 128, 4)
test_tiled_gather_rows(4096, 4096, 128, 128, 4)
