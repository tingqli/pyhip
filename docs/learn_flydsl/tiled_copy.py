import os
import flydsl
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly

import pyhip
import pyhip.contrib.flydsl as fxu

if 1:
    make_tiled_copy = fxu.make_tiled_copy
    make_tiled_mma = fxu.make_tiled_mma
    make_tiled_copy_A = fxu.make_tiled_copy_A
    make_tiled_copy_B = fxu.make_tiled_copy_B
    make_tiled_copy_C = fxu.make_tiled_copy_C
else:
    make_tiled_copy = fx.make_tiled_copy
    make_tiled_mma = fx.make_tiled_mma
    make_tiled_copy_A = fx.make_tiled_copy_A
    make_tiled_copy_B = fx.make_tiled_copy_B
    make_tiled_copy_C = fx.make_tiled_copy_C    

#fxu.enable_dump_ir(True)


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
def test_tiled_copy_basic(M, N, tileM, tileN, num_waves, expand_copy):
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def cp_kernel_basic(A: fx.Tensor, B: fx.Tensor, expand_copy: fx.Constexpr[bool]):
        bx = fx.block_idx.x
        by = fx.block_idx.y
        tid = fx.thread_idx.x

        assert A.dtype == B.dtype

        A = A[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>
        B = B[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>

        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        VECT_WIDTH = fxu.div_e(copy_bits, A.dtype.width)
        print("VECT_WIDTH: ", VECT_WIDTH)

        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(num_threads, VECT_WIDTH, tileN)
        
        #tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tv_tilemn)
        tiled_copy = make_tiled_copy(copy_atom, tv_layout, tv_tilemn)
        part_A = tiled_copy.get_slice(tid).partition_S(A)
        part_B = tiled_copy.get_slice(tid).partition_D(B)

        if expand_copy:
            fxu.recurisve_apply(lambda pA, pB: fx.copy(copy_atom, pA, pB), part_A, part_B)
            #=========================================================================
            # faster version
            #frag = fx.make_fragment_like(part_A)
            #fxu.recurisve_apply(lambda a, b: fx.copy(copy_atom, a, b), part_A, frag)
            #fxu.recurisve_apply(lambda a, b: fx.copy(copy_atom, a, b), frag, part_B)
        else:
            fx.copy(copy_atom, part_A, part_B)
            #=========================================================================
            # directly copy from global to global like above cannot hide load-latency
            # need to issue all copies from global to register w/o waitting each-other(no dependency),
            # then from register to global(wait before each store inserted by AMDGPU backend)
            #=========================================================================
            #frag = fx.make_fragment_like(part_A)
            #print(">>>>>>>>>>>> ", frag)
            #fx.copy(copy_atom, part_A, frag)
            #fx.copy(copy_atom, frag, part_B)
    
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

        cp_kernel_basic(A, B, expand_copy).launch(
            grid=(grid_m, grid_n, 1),
            block=(num_threads, 1, 1), stream=stream
        )

    _, stream = pyhip.set_device()
    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    test(A, B, expand_copy, stream)
    assert torch.allclose(A, B, atol=1e-5)
    _, us = pyhip.run_perftest(test, A, B, expand_copy, stream, num_iters=10, num_warmup=2, num_bytes=M*N*4*2)
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

        # STEP.1 divide block/tile into tv-layout sub-tiles
        tv_layout = fx.make_layout(((8, 8, 4), 8), ((256, 1, 8), 32))
        at_subtile = fx.tiled_divide(at_block, fx.make_tile(32, 64))
        print("at_subtile", at_subtile)

        # STEP.2 divide tv-layout tile into copy-atom tile
        atomNumThr, atomNumVal = copy_atom.layout_ref_tv.shape.to_py_value()
        atomTile = fx.make_tile(atomNumThr, atomNumVal)      # tv_layout = Layout<((8,8,4),8):((256,1,8),32)> Tile<[1|8]>
        atomLayoutTV = fx.zipped_divide(tv_layout, atomTile) #             Layout<((1,8),((8,32),1)):((0,32),((256,1),0))>

        # STEP.3 view copy-atom sub-tile part (atomLayoutTV[0]) as layout_src_tv/layout_dst_tv
        refInv = fx.right_inverse(copy_atom.layout_ref_tv)
        ref2trg = fx.composition(refInv, copy_atom.layout_src_tv)
        print("ref2trg: ", ref2trg)                 # Layout<(1,4):(0,1)>

        copy_atom_tv = fx.composition(atomLayoutTV[0], ref2trg) #
        print("atomLayoutTV[0]: ", atomLayoutTV[0]) # Layout<(1,4):(0,32)>
        print("copy_atom_tv: ", copy_atom_tv)       # Layout<(1,4):(0,32)>

        # STEP.4 from (t1,v1),(t2,v2) -> (t1,t2),(v1,v2)
        thrval2mn = fxu.concat_modes((copy_atom_tv[0], atomLayoutTV[1][0]),
                                    (copy_atom_tv[1], atomLayoutTV[1][1]))

        print("thrval2mn: ", thrval2mn)           # Layout<((1,(8,32)),(4,2)):((0,(256,1)),(32,128))>
        thrval2mn = fx.coalesce(thrval2mn, fx.make_int_tuple((1, fx.make_int_tuple((1,1))))) # Layout<((8,32),(8,1)):((256,1),(32,0))>
        print("thrval2mn coalesce : ", thrval2mn) # Layout<((8,32),(4,2)):((256,1),(32,128))>

        at_subtile_tv = fx.composition(at_subtile, make_tile(thrval2mn))
        print("at_subtile_tv", at_subtile_tv)         # Layout<(((8,32),(4,2)),4,1,128):(((8,8192),(1,4)),262144,0,64)>
        at_subtile_thread = fx.slice(at_subtile_tv, ((1, None), None, None, None))
        print("at_subtile_thread", at_subtile_thread) # Layout<((4,2),4,1,128):((1,4),262144,0,64)>
        
        # first mode is (num_values_copyatom, num_values_tvlayout//num_values_copyatom),
        # for example, tv-layout allocate 8 values per thread copy-atom can only copy 4
        # values per thread, the first-mode will be (4,2)
        #
        # the rest modes (4,1,128) are tv-layout-tile repeats from tiled_divide(block, tile(tv_layout_m, tv_layout_n))

    partition()

# test_partion()


def test_custom_copy_basic(M, N, tileM, tileN, num_waves):
    num_threads = num_waves * 64

    @flyc.kernel(known_block_size=[num_threads, 1, 1]) # known_block_size at compile time
    def cp_kernel_custom(A: fx.Tensor, B: fx.Tensor):
        bx = fx.block_idx.x
        by = fx.block_idx.y
        tid = fx.thread_idx.x
        A = A[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>
        B = B[None, (bx, by)] # Tensor<f32, global, (tileM,tileN):(?{i64},1)>

        copy_bits = 128
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        VECT_WIDTH = fxu.div_e(copy_bits, A.dtype.width)

        # ======= coealescing TV-layout =======
        tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(num_threads, VECT_WIDTH*1, tileN)
        
        A = fxu.tv_partition(A, tv_layout, tv_tilemn)
        B = fxu.tv_partition(B, tv_layout, tv_tilemn)
        frag = fx.make_fragment_like(A[(tid, None), None])
        print(">>>>>>>>>>>>2 ", A[(tid, None), None], B[(tid, None), None], frag)

        print(A.layout)
        frags = []
        num_iters = fx.size(A.layout[1]).get_static_leaf_int
        for i in fx.range_constexpr(num_iters):
            part_A = A[(tid, None), i]
            part_B = B[(tid, None), i]
            frags.append(fx.make_fragment_like(part_A))
            fx.copy(copy_atom, part_A, frags[-1])
            fx.copy(copy_atom, frags[-1], part_B)
            
    @flyc.jit
    def test(A: fx.Tensor, B: fx.Tensor, stream):
        assert A.dtype == B.dtype
        assert A.leading_dim == 1 and B.leading_dim == 1, "kernel assumes 2nd mode is contiguous"
        A = fx.zipped_divide(A, fx.make_tile(tileM, tileN))
        B = fx.zipped_divide(B, fx.make_tile(tileM, tileN))
        assert A.stride[0][1].get_static_leaf_int == 1 and B.stride[0][1].get_static_leaf_int == 1, \
              "kernel assumes 2nd mode is contiguous"

        grid_m = fx.get_scalar(A.shape[1][0])
        grid_n = fx.get_scalar(A.shape[1][1])

        cp_kernel_custom(A, B).launch(
            grid=(grid_m, grid_n, 1),
            block=(num_threads, 1, 1), stream=stream
        )

    _, stream = pyhip.set_device()
    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    test(A, B, stream)
    assert torch.allclose(A, B, atol=1e-5)
    _, us = pyhip.run_perftest(test, A, B, stream, num_iters=10, num_warmup=2, num_verbose=0, num_name = "kerne_copy",
                               num_bytes=A.numel()*A.element_size()+B.numel()*B.element_size())
    assert torch.allclose(A, B, atol=1e-5)

    torch_copy = lambda A, B: B.copy_(A)
    _, us = pyhip.run_perftest(torch_copy, A, B, num_iters=10, num_warmup=2, num_verbose=0, num_name = "torch_copy",
                               num_bytes=A.numel()*A.element_size()+B.numel()*B.element_size())    

if 0:
    test_custom_copy_basic(2*4096, 4096, 64, 256, 4)
    test_tiled_copy_basic(2*4096, 4096, 64, 256, 4, True)
    test_tiled_copy_basic(2*4096, 4096, 64, 256, 4, False)
    test_tiled_copy_basic(1024, 256*256, 1, 256*256, 4, False)

# more waves/threads per work-group allows more coalescing
# but if tileN is bigger enough, 1 wave per WG is also good
if 0:
    test_custom_copy_basic(2*4096, 4096, 64, 256, 4)    # 2.8TB/s 
    test_custom_copy_basic(2*4096, 4096, 64, 256, 1)    # 2.9TB/s
    test_custom_copy_basic(2*4096, 4096, 1, 4096, 1)    # 3.1TB/s
    test_custom_copy_basic(2*4096, 4096, 1, 4096, 4)    # 3.3TB/s
    test_custom_copy_basic(2*4096, 4096, 1, 4096, 8)    # 3.1TB/s

test_custom_copy_basic(4*4096, 8192, 1, 8192, 4)    # 3.4TB/s
test_custom_copy_basic(1, 400*80*8192, 1, 8192, 8)  # 3.5TB/s

# Why we can't reach 4TB/s ?
# torch copy was even slower.
