# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl, vector

# shuffle_weight lives in the FlyDSL repo's tests/utils.py (not part of the installed package).
import sys as _sys
if "/mywork/FlyDSL" not in _sys.path:
    _sys.path.insert(0, "/mywork/FlyDSL")
from tests.utils import shuffle_weight
NUM_CU = 64
K_ITER = 64
M_REPEAT = 4
WAVE_NUM=4
M_BLOCK = WAVE_NUM * 8 * M_REPEAT
K_BLOCK= 64

M = NUM_CU * M_BLOCK
K = K_BLOCK * K_ITER


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def enable_dump_ir(enable_debug_info=True):
    if enable_debug_info:
        import flydsl
        from flydsl.utils.env import DebugEnvManager
        from flydsl._mlir import ir

        DebugEnvManager.enable_debug_info = enable_debug_info
        DebugEnvManager.dump_asm = True
        DebugEnvManager.dump_ir = True
        DebugEnvManager.dump_dir = "my_ir_dumps"
        ir._globals.register_traceback_file_inclusion(__file__)
        ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
        ir._globals.set_loc_tracebacks_frame_limit(40)
        ir._globals.set_loc_tracebacks_enabled(True)
        os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


LDS_SWIZZLE = _env_flag("SWIZZLE", "1")

@fx.struct
class SharedStorage:
    a0: fx.Array[BFloat16, M_BLOCK*K_BLOCK, 16]

    
@flyc.kernel
def async_copy_tile(
    A: fx.Tensor,
    B: fx.Tensor,
):
    print(f'###{LDS_SWIZZLE=}')

    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    # slicing A, B tensor
    bA = fx.flat_divide(A, (M_BLOCK, K_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, K_BLOCK))
    bA = bA[None, None, bid, None]
    bB = bB[None, None, bid, None]
    
    thr_layout = fx.make_layout((8*WAVE_NUM , 8), (8, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    dma_tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    dma_thr_copy = dma_tiled_copy.get_slice(tid)
    lds_thr_copy = dma_thr_copy

    lsd_copy_128b_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

    tiled_mma = fx.make_tiled_mma(
    fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float16)),
    fx.make_layout((2, 2, 1), (2, 1, 0)),
    fx.make_tile(None, None, fx.make_layout((8, 4, 2), (1, 8, 32))),)

    thr_copy_mma = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
    thr_mma = tiled_mma.thr_slice(tid)


    lds = fx.SharedAllocator().allocate(SharedStorage).peek()   
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, K_BLOCK), (1, 0))
    sA_wr = fx.make_view(fx.get_dyn_shared(fx.BFloat16), LDS_layout_A)
    sA_rd = sA_wr
    
    # partition the tensor into tiles for each thread
    g2s_src = dma_thr_copy.partition_S(bA)
    g2s_dest = dma_thr_copy.partition_D(sA_wr)

    s2g_src = thr_copy_mma.partition_S(sA_rd)
    s2g_dst = thr_copy_mma.partition_D(bB)
    # if using swizzle, apply the swizzle in the buffer_load_lds src address and LDS read src
    # only needs to update the g2s_src and s2g_src
    if LDS_SWIZZLE:
        #https://github.com/NVIDIA/cutlass/blob/b0e09d7cd371eded41f7c1e057caf1593c27ba55/include/cute/swizzle.hpp#L40
        # 
        # 同样的swizzle作用在不同尺寸的tensor shape上需要有相同的num_base和num_bits, num_shift由tensor的行维度的stride决定。
        num_base = 3
        num_bits = 3
        # num_shift = log 2 (stride of m) - num_base
        num_shift = K_BLOCK.bit_length() - 1 - num_base 
        SWIZZLE_LDS_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                LDS_layout_A,
            )
        sA_rd = fx.make_view(fx.get_iter(sA_wr), SWIZZLE_LDS_layout_A)
        s2g_src = thr_copy_mma.partition_S(sA_rd)
        num_shift = K.bit_length() - 1 - num_base 
        # num_shift = log 2 (stride of m) - num_base
        GA_SWIZZLE_LAYOUT = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, num_shift)),
                fx.get_layout(A),
            )
        gA_swizzle =fx.make_view(fx.get_iter(A), GA_SWIZZLE_LAYOUT)
        bA_swizzle = fx.flat_divide(gA_swizzle, (M_BLOCK, K_BLOCK))
        bA_swizzle = bA_swizzle[None, None, bid, None]
        g2s_src = dma_thr_copy.partition_S(bA_swizzle)

    mma_frag = thr_mma.make_fragment_A(sA_rd)
    s2g_frag = thr_copy_mma.retile(mma_frag)
    for block_idx in range(0, K_ITER):
        fx.copy(async_copy_atom, g2s_src[None, None, None, block_idx], g2s_dest)
        gpu.barrier()
        fx.copy(lsd_copy_128b_atom, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, block_idx])


@flyc.kernel
def async_copy_isa(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    B = fx.rocdl.make_buffer_tensor(B)
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()   
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, K_BLOCK), (1, 0))

    if LDS_SWIZZLE:
        LDS_layout_A = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((M_BLOCK, K_BLOCK), (1, 0)),
        )
    sA = fx.make_view(lds.a0.ptr, LDS_layout_A)
    bB = fx.flat_divide(B, (M_BLOCK, K_BLOCK))
    bB = bB[None, None, bid, None]

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    lsd_copy_128b_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    thr_layout = fx.make_layout((WAVE_NUM*8 , 8), (8, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))

    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)
    s2g_src = thr_copy.partition_S(sA)
    s2g_frag = fx.make_fragment_like(s2g_src)
    s2g_dst = thr_copy.partition_D(bB)

    elem_bytes = 2
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    gA_flat = fx.rocdl.make_buffer_tensor(
        (fx.make_view(fx.get_iter(A), fx.make_layout(65536 * K, 1))),
        max_size=False,
        num_records_bytes=fx.Int64(M) * fx.Int64(K) * fx.Int64(elem_bytes),
    )

    gA_div = fx.logical_divide(gA_flat, fx.make_layout(1, 1))
    print(f'#####{gA_div=}\n{gA_flat=}')
    sA_ptr = lds.a0.ptr


    bx_m = bid * M_BLOCK
    wave_id = tid // 64
    # 256*8. 256 threads load contineously
    load_elems = 16 // elem_bytes
    # each wave load 64*16 btyes.
    wave_stride = 64 * load_elems
    # num_a_loads = M_BLOCK // (WAVE_NUM*8)
    def dma_a_to_lds(blk_idx):
        wave_off = fx.rocdl.readfirstlane(fx.Int32.ir_type, wave_id * wave_stride)
        lds_ptr = fx.add_offset(sA_ptr, wave_off)
        base_n = blk_idx * K_BLOCK
        total_threads = WAVE_NUM * 64
        step_elems = total_threads * load_elems
        for i in range_constexpr(M_REPEAT):
            elem_pos =  i * total_threads * load_elems + tid * load_elems
            m = elem_pos // K_BLOCK
            n = elem_pos % K_BLOCK
            n_swz = n
            if LDS_SWIZZLE:
                # n_swz = n ^ ((m % k_blocks16_dma) * elems_per_16b)
                n_swz = ((n // 8) ^ (m % 8)) * 8
            # print(f'###############################################')
            offset = ((bx_m + m) * K + base_n + n_swz)
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(gA_div, (None, fx.Int32(offset)))
            fx.copy(dma_atom, src, dst)
            lds_ptr = fx.add_offset(lds_ptr, step_elems)

    for block_idx in range(0, K_ITER):
        dma_a_to_lds(block_idx)
        # rocdl.s_waitcnt(0)
        gpu.barrier()
        fx.copy(lsd_copy_128b_atom, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, block_idx])


PADDING_ELEMS = 16
PADDING_NUM = PADDING_ELEMS * 16

@fx.struct
class LDS_PADDING:
    a0: fx.Array[BFloat16, M_BLOCK*K_BLOCK+PADDING_NUM, 16]
@flyc.kernel
def async_copy_padding(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    lsd_copy_128b_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)


    #async copy使用的是 8x8 thread layout copy数据, 每一行的8个列线程去读取Atensor中一行的BK个连续元素，
    #padding为了避免LDS每一行都增加padding,所以在LDS 读取数据的时候会采用8个行threads把LDS分成8组的读法，从而导致buffer load也需要分成8组去读。
    #8个thread 行这里不会去读取A中BM连续的8行， paddding的copy中会将BM分成8组，每一组中包含连续的BM//8行数据，8组对应8x8thread mapping中的8行，
    #所以每个组内部的行数BM //8, 会被不同wave相同的行thread访问，BM //8 // WAVE_NUM 就是每个thread的buffer load LDS 指令个数，
    #BM =128是，每组16行，m0-m15中的每一行会被4个waves的第一行线程(thread0-thread7)访问，4个waves每组16行，所以buffer load指令是4条。
    #对应的8个thread 行会从对应的8个组中读取数据，所以这个的一个copy tile在实际的读取A tensor中是个sparse的布局，如何实现？？
    
    #fx.make_tiled_copy（暂时先把copy atom放在一边）代表的是一个workgroup 中所有的thread访问src和dest是 thread 与 logical m, logical n之间的关系，
    #logical m/n都是连续的，所以make tiled copy代表的是一个逻辑m,n上dense的 tile. tile大小用(tile_m, tile_n) 表达, 每个thread访问的logical_m, logical_n 通过value计算
    #thread_id -> 1D value(M major/column major) -> 2D index (logic m, logic n), logical_m = value % tile_m, logial_n = value // tile_m
    #所以tiled_copy就是一个逻辑上dense的块，与实际的内存布局可以解耦， 每一列的stride就是tile_m, 所以如果source和destination的线程的layout一致(比如都是8x8 coalescing的方式)
    #source和destination就可以使用同一个tiled_copy. 
    
    # （logical_m, logical_n) 是 tensor layout的coordinate输入，通过输入得出一个基于tensor的offset, 这里需要保证tensor layout的corrdinate排序与(logical_m, logical_n)
    #  这里不会改变实际的内存布局，只是把这个tensor用不同的layout 视角。
    #  假设(logical_m,logical_n) 可以分解为((logical_m0, logical_m2), logical_n),
    #  假设实际tensor在内存中的natural layout可以把m分解为sublayout((m0,m1,m2),n):((ms0,ms1,ms2),ns)
    #  根据tiled_copy中(logical_m0, logical_m2)的访问顺序，需要把tensor用相同的mode访问，m0,m2这两个mode位于m sublayout的变化快的维度
    #  ((m0,m1,m2),n):((ms0, ms1,ms2),ns) ->((m0,m2,m1),n):((ms0, ms2, ms1),ns)
    
    # 把上面的方法应用到目前的例子
    # A的 nature layout 是 （M，N）: (N, 1), 这里对m mode做一下拆解 , 对于每个sublayout, mode0是变化最快的。
    # ((M_BLOCK, M//M_BLOCK), N): 
    # ((N,       N*(M_BLOCK), 1) 
    # 继续拆解M_BLOCK, M_BLOCK 分成8组
    # ((M_BLOCK//8 , 8 ,       M // M_BLOCK), N) : 
    # ((N,         N*M_BLOCK//8, N * MBLOCK),  1)
    
    # 这里对于tile_copy的logical m, 8 是8个thread row访问的维度，所以在tiled_copy生成pdf文件中，8 是在logical m 里面的变化最快维度，
    # 目的试tensor A 实际的 m sublayout的被分解后的mode 排序与 tiled copy中逻辑m的mode 排序一致， logical m是按照(8 thread rows, WAVE_NUM) 排布，
    # 所以这里我们需要把m 里面的mode0 与 mode1交换一下，重新生成一个A tensor view.
    # ((8 ,            M_BLOCK//8 , M // M_BLOCK), N) : 
    # ((N * M_BLOCK//8, N,  N * MBLOCK), 1)
    # 这样就可以用这个copy tiled 去partition A tensor了。
    
    # wave0的thread0在tiled_copy中访问8行，实际的tensor layout对应m0, m16, m32, m48, m64, m80, m96, m112.
    ac_tile_mn = fx.make_tile(WAVE_NUM*8, 64)
    ac_tv_layout =  fx.make_layout(((8, 8, WAVE_NUM), 8), ((8*WAVE_NUM*8, 1, 8), WAVE_NUM*8))
    ac_tiled_copy = fx.make_tiled_copy(buffer_copy_atom, ac_tv_layout, ac_tile_mn)
    ac_thr = ac_tiled_copy.get_slice(tid)
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(((8, M_BLOCK//8, M//M_BLOCK), K), ((M_BLOCK//8*K, K, M_BLOCK*K), 1))))
    
    # slicing A, B tensor
    bA = fx.flat_divide(A, (M_BLOCK, K_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, K_BLOCK))
    bA = bA[None, None, bid, None]
    bB = bB[None, None, bid, None]
    
    # buffer load写入LDS的layout是自然的layout,
    # 没有padding的情况下，LDS的layout是 (M_BLOCK, K_BLOCK) : (K_BLOCK, 1), M_BLOCK这里是128
    # 拆解成 ((8, 16), K_BLOCK) : ((K_BLOCK, K_BLOCK*8), 1), 然后考虑padding,每8行之后做一个Padding,所以只需要把padding的stride加到`16`这个mode上
    lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()   
    lds_layout_wr =fx.make_layout(((8, 16), K_BLOCK), ((K_BLOCK, 8*K_BLOCK+PADDING_ELEMS), 1))
    
    # LDS的读取使用的是16x4 thread copy_tiled, 模拟MFMA读取pattern, 4个wave , 2x2方式读取，模拟MFMA wave layout，但是只是读取和写出，没有MFMA计算，
    # 所以这里的读取和写出都存在重复的操作
    # 对于tensorA读取以及tensorB写出， wave0/wave1读取写出相同的内容，wave2/wave3读取写出相同的内容
    # LDS 读取的tiled_copy与写入B的tiled_copy 一致，wave分成2x2,其中一个mode `2`的stride为0， 代表这两个wave读取写入相同的logical_m,logical_n
    # lds_rd_tv_layout =  fx.make_layout(((16, 4, 2, 2), 8), ((1, 16*WAVE_NUM//2*8, 0, 16), WAVE_NUM*16//2))
    # 同样需要考虑LDS的读取layout 
    # LDS自然的layout是 ((8, 16), K_BLOCK), ((K_BLOCK, 8*K_BLOCK+PADDING_ELEMS), 1)， N block拆解成32x2:
    # ((8, 16), (32, K_BLOCK//2), ((K_BLOCK, 8*K_BLOCK+PADDING_ELEMS), (1, 32))
    # read LDS tiled copy 中 logical_m，按照（16， WAVE_NUM）排布，所以把16 放在mode0, sA_rd的layout就是：
    # ((16,  8),                     (32, K_BLOCK//32))：
    # ((8*K_BLOCK+PADDING_ELEMS, K_BLOCK), (1, 32))
    
    # 4个WAVE的 thread0-thread3会读LDS中的连续2行，由于这2行时连续的，所以这里就不用在M_BLOCK//16里把2再拆分出来了，如果想要拆分也可以，就是
    # ((16,                               2， 4),  (32, K_BLOCK//32))：
    # ((8*K_BLOCK+PADDING_ELEMS, 64， 64*2), (1, 32))
    
    # 同样的道理，lds_rd_tiled 的32行m 
    # each wave use 16 * 4 MFMA threads layout
    lds_rd_tile_mn = fx.make_tile(WAVE_NUM*16//2, 32)
    lds_rd_tv_layout =  fx.make_layout(((16, 4, 2, 2), 8), ((1, 16*WAVE_NUM//2*8, 0, 16), WAVE_NUM*16//2))
    lds_rd_tiled = fx.make_tiled_copy(buffer_copy_atom, lds_rd_tv_layout, lds_rd_tile_mn)
    lds_rd_thread = lds_rd_tiled.get_slice(tid)
    # fx.utils.print_typst(lds_rd_tiled, file="lds_rd_tiled.typ")
    sA_wr = fx.make_view(lds.a0.ptr, lds_layout_wr)
    
    # padding LDS read layout:
    lds_layout_rd =fx.make_layout(((16, M_BLOCK//16), (32, K_BLOCK//32)), ((M_BLOCK//16*K_BLOCK+PADDING_ELEMS, 64), (1, 32)))
    # lds_layout_rd =fx.make_layout(((16, 2, M_BLOCK//32), (32, K_BLOCK//32)), ((M_BLOCK//16*K_BLOCK+PADDING_ELEMS, 64, 128), (1, 32)))

    sA_rd = fx.make_view(lds.a0.ptr, lds_layout_rd)
    ac_src = ac_thr.partition_S(bA)
    ac_dest = ac_thr.partition_D(sA_wr)
    
    s2g_src = lds_rd_thread.partition_S(sA_rd)
    # B的自然layout与LDS 读取的mode一致，不需要改变B的layout view。
    s2g_dest = lds_rd_thread.partition_D(bB)
    s2g_frag = fx.make_fragment_like(s2g_dest[None, None, None, 0])
    
    test_s2g_src = ac_thr.partition_S(sA_rd)
    test_dest = ac_thr.partition_D(bB)
    test_frag = fx.make_fragment_like(test_dest[None, None, None, 0])
    for block_idx in range(0, K_ITER):
        fx.copy(async_copy_atom, ac_src[None, None, None, block_idx], ac_dest)
        gpu.barrier()
        fx.copy(lsd_copy_128b_atom, s2g_src, s2g_frag)
        fx.copy(buffer_copy_atom, s2g_frag, s2g_dest[None, None, None, block_idx])

# 输入 A 是 host 端 preshuffle 后的数据（shuffle_weight(orig,(16,16))），布局按 (M,N)=(preshuffle-N, preshuffle-K)。
# global A 保持 plain (M,N)，per-tile 用 preshuffle 的 subB layout gather 逻辑 (n,k) tile；
# DMA copy tile / LDS wr==rd layout 完全按 preshuffle.txt；LDS 读取用 MMA copy-B tile（4 波平分 N，无 MMA 计算），
# 读出后直接 store 到输出 B。校验 输出 == 原始（未 shuffle）数据。

#################################preshuffle layout#################################p
# preshuffle B(N, K) layout is 

# (N//16, K // 32, 4k, 16n, 8k)
# (K*16,  512,     128, 8,   1)

# 假设BN=128， BK = 64 for bf16 type.

# 每一次DMA需要copy [BN, BK] data into LDS, 这些会均分给256个线程，每个线程copy 8个元素。我们看看[BN, BK]这些元素在layout中的stide


# (N//16, K // 32, 4k, 16n, 8k) ->
# (N//BN, BN // 16, K // BK,  BK//32, 4k, 16n, 8k) -> 
# (N//128, 8n,   K//64, 2k, 4k, 16n, 8k)  合并2k x4k ->

# (N//128, 8n,   K//64, 8k, 16n, 8k), strides:
# (128*K,  16*K,  1024, 128, 8, 1 )

# 这个stride我们先放在这里，后面再描述layout的时候，需要根据对应的mode做一下变化。

################################## copy tile#################################p

# 现在需要考虑的是的是copy tile，
# copy tile对应的大小是,4个wave, 256*8 = 2048个element.尽量的连续copy,

# 以[128n, 64k]作为一次fx.copy到DMA里的单位， 在(N//128, 8n, K//64, 8k, 16n, 8k), slice出[BN, BK] layout:
# (8n, 8k, 16n, 8k) , (16*K, 128, 8, 1) , tilecopy的尺寸是只256个thread copy的n, k的值，
# 可以看出有128 thread可以在物理连续上copy(4k, 16n, 8k) ,其余的128 thread需要在8n上分，

# (8n, 8k, 16n, 8k) -> (4n, 2n, 8k, 16n, 8k), 256个thread copy 的tile大小是(2n, 8k, 16n, 8k) 即（32n, 64k）

# 一个tile的layout是：
# (4n,  2n,  8k, 16n, 8k):
# (32*K,16*K, 128, 8, 1)

# 这个tile copy的thread value layout, 一共有256个thread, 按照尽量连续copy,我们先来排thread, value， 然后在根据是在M, K的sub mode维度计算stride.
# 256个threads, 8个value, 按照上面的layout, 连续thread访问来开始排放
# (256, 8) -> ((128, 2), 8) -> ((16, 8, 2) , 8value)  这里对应的 k, n mode,

#  ((16,   8,  2) , 8value) ， 对应的mode:
#  ((16n,  8k, 2n), 8k) 

# 根据mode就可以计算出stride,这个stride是N major， tile尺寸是(32n, 64k), 所以K维度的stride是32，
#  ((16,   8,   2) ,  8value) 
#  ((16n,  8k,  2n),  8k) stride是：
#  ((1,   8*32, 16), 32) 

# 所以tile thread value layout 是((16, 8, 2) , 8):((1, 256, 16), 32), tile is (32, 64)
##################################layout of subtensor#################################
# 现在需要描述一下sub B tensor的layout,之前的这个B preshuffle之后的layout 分解：

# (N//128, 8n,   K//64, 8k, 16n, 8k), strides:
# (128*K,  16*K,  1024, 128, 8, 1 )


# subB tensor 是（BN, BK, Kiter）->(128n, 64K, K//64)， 

# 分解BN 维度， tile Size是32n, 是否凑出32??好像不需要：
# 128n -> (16, 8):(8， 16*K)

# 分解BK维度：
# 64k -> (8, 8):(1, 128)

# K iter 维度：
# (K//64) : (1024)

# ((16, 8), (8, 8), K // 64):((8， 16*K), (1, 128), 1024)

##################################layout of LDS#################################p

#  LDS layout可以根据[BN, BK]在subtensor中的layout改动，

# SubB, [BN, BK]
#  (8n,  8k, 16n, 8k), strides:(16*K, 128, 8, 1 ), 8n mode stride只需要改成连续就可以了 ， (BN, BK)在LDS中连续，所以
#  LDS （8n, 8k, 16n, 8k）, (1024 ，128， 8， 1)

# 按照（N， K）就是(16, 8), (8, 8): (8, 1024), (1, 128)

@flyc.kernel
def async_copy_preshuffleB(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)

    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    lsd_copy_128b_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)

    # A 保持 plain (M,N)；flat_divide 取 tile 基址后，用 preshuffle subB layout 重新 view 以 gather 逻辑 (n,k)。
    # doc: subB = ((16, BN//16),(8, BK//8), Kiter):((8, 16*K),(1,128),1024)（BN=M_BLOCK, BK=K_BLOCK, K=N）。
    bA = fx.flat_divide(A, (M_BLOCK, K_BLOCK))[None, None, bid, None]  # (BN, BK, k) base
    bB = fx.flat_divide(B, (M_BLOCK, K_BLOCK))[None, None, bid, None]  # (BN, BK, k) plain out
    subB_layout = fx.make_layout(
        ((16, M_BLOCK//16), (8, K_BLOCK//8), K//K_BLOCK),
        ((8, 16*K), (1, 128), 1024),
    )
    bA = fx.Tensor(fx.make_view(fx.get_iter(bA), subB_layout))

    # copy tile（doc）：tv = ((16n,8k,2n),8k0):((1,256,16),32), tile (32,64) -> coalesced gather。
    b_tv_layout = fx.make_layout(((16, 8, 2), 8), ((1, 256, 16), 32))
    b_tiled_copy = fx.make_tiled_copy(buffer_copy_atom, b_tv_layout, fx.make_tile(32, 64))
    b_thr = b_tiled_copy.get_slice(tid)

    # LDS 布局（doc）：连续 preshuffle 序（无 pad/swizzle），wr==rd 保证写入/读取 (n,k) 一致。
    # ((16, BN//16),(8, BK//8)):((8,1024),(1,128))。
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    lds_b_layout = fx.make_layout(((16, M_BLOCK//16), (8, K_BLOCK//8)), ((8, 1024), (1, 128)))
    sA_wr = fx.make_view(lds.a0.ptr, lds_b_layout)
    sA_rd = fx.make_view(lds.a0.ptr, lds_b_layout)

    ac_src = b_thr.partition_S(bA)
    ac_dest = b_thr.partition_D(sA_wr)

    # LDS 读取用 MMA copy-B tile：wave layout = 4 波平分 N (1,4,1)（非 gemm 的 2x2），不做 MMA 计算。
    tiled_mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)),
        fx.make_layout((1, 4, 1), (0, 1, 0)),
    )
    ldsB_rd_thread = fx.make_tiled_copy_B(buffer_copy_atom, tiled_mma).get_slice(tid)
    s2g_src = ldsB_rd_thread.partition_S(sA_rd)
    s2g_dest = ldsB_rd_thread.partition_D(bB)
    s2g_frag = fx.make_fragment_like(s2g_dest[None, None, None, 0])

    for block_idx in range(0, K_ITER):
        fx.copy(async_copy_atom, ac_src[None, None, None, block_idx], ac_dest)
        gpu.barrier()
        fx.copy(lsd_copy_128b_atom, s2g_src, s2g_frag)
        fx.copy(buffer_copy_atom, s2g_frag, s2g_dest[None, None, None, block_idx])


@flyc.jit
def async_dma_copy_padding(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    async_copy_padding(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=(M_BLOCK*K_BLOCK+PADDING_NUM)*2, stream=stream)


@flyc.jit
def async_dma_copy_swizzle(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    async_copy_tile(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=M_BLOCK*K_BLOCK*2, stream=stream)


@flyc.jit
def async_dma_copy_preshuffleB(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    assert M_BLOCK == 128 and K_BLOCK == 64, "preshuffleB copy only support M_BLOCK=128, K_BLOCK=64"
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1))))
    async_copy_preshuffleB(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=M_BLOCK*K_BLOCK*2, stream=stream)
        

enable_dump_ir(False)
A = torch.arange(M * K, dtype=torch.bfloat16).reshape(M, K).cuda()
B = torch.zeros(M, K, dtype=torch.bfloat16).cuda()
C = torch.zeros(M, K, dtype=torch.bfloat16).cuda()
LDS_RESULT = A.clone().reshape(M//M_BLOCK, 8, M_BLOCK//8, K).permute(0, 2, 1, 3).contiguous().reshape(M, K)
print(f'{M=} {K=} {M_BLOCK=} {K_BLOCK=} {WAVE_NUM=}')
async_dma_copy_padding(A, B, stream=torch.cuda.Stream())
torch.cuda.synchronize()
async_dma_copy_swizzle(A, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()
is_correct = torch.allclose(A, B)
print("Padding Result correct:", is_correct)
is_correct = torch.allclose(A, C)
print("Swizzle Result correct:", is_correct)
if not is_correct:
    print(f'{A[0]=}')
    print(f'{B[0]=}')
    print(f'{A[1]=}')
    print(f'{B[1]=}')

# preshuffle B copy 验证：喂入 shuffle 后的数据，kernel 重建逻辑数据并 store，验证 == 原始。
_orig = torch.randn(M, K, dtype=torch.bfloat16).cuda()
_shuffled = shuffle_weight(_orig, layout=(16, 16))
_D = torch.zeros(M, K, dtype=torch.bfloat16).cuda()
async_dma_copy_preshuffleB(_shuffled, _D, stream=torch.cuda.Stream())
torch.cuda.synchronize()
_ok = torch.allclose(_orig, _D)
print("PreshuffleB Result correct:", _ok)
if not _ok:
    print(f'{_orig[0, :16]=}')
    print(f'{_D[0, :16]=}')
    print(f'mismatch count: {(_orig != _D).sum().item()} / {M*K}')
