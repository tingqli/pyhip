# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl, vector
from flydsl.expr import arith as fxa
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import scf as _mlir_scf
NUM_CU = 64
N_ITER = 64
M_REPEAT = 4
WAVE_NUM=4
M_BLOCK = WAVE_NUM * 8 * M_REPEAT
N_BLOCK= 64

M = NUM_CU * M_BLOCK
N = N_BLOCK * N_ITER


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
    a0: fx.Array[BFloat16, M_BLOCK*N_BLOCK, 16]

    
@flyc.kernel
def asycn_copy_tile(
    A: fx.Tensor,
    B: fx.Tensor,
):
    print(f'###{LDS_SWIZZLE=}')

    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    # slicing A, B tensor
    bA = fx.flat_divide(A, (M_BLOCK, N_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
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
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0))
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
        num_shift = N_BLOCK.bit_length() - 1 - num_base 
        SWIZZLE_LDS_layout_A = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, 3)),
                LDS_layout_A,
            )
        sA_rd = fx.make_view(fx.get_iter(sA_wr), SWIZZLE_LDS_layout_A)
        s2g_src = thr_copy_mma.partition_S(sA_rd)
        num_shift = N.bit_length() - 1 - num_base 
        # num_shift = log 2 (stride of m) - num_base
        GA_SWIZZLE_LAYOUT = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 3, num_shift)),
                fx.get_layout(A),
            )
        gA_swizzle =fx.make_view(fx.get_iter(A), GA_SWIZZLE_LAYOUT)
        bA_swizzle = fx.flat_divide(gA_swizzle, (M_BLOCK, N_BLOCK))
        bA_swizzle = bA_swizzle[None, None, bid, None]
        g2s_src = dma_thr_copy.partition_S(bA_swizzle)

    mma_frag = thr_mma.make_fragment_A(sA_rd)
    s2g_frag = thr_copy_mma.retile(mma_frag)
    for block_idx in range(0, N_ITER):
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
    LDS_layout_A =fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0))

    if LDS_SWIZZLE:
        LDS_layout_A = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 3, 3)),
            fx.make_ordered_layout((M_BLOCK, N_BLOCK), (1, 0)),
        )
    sA = fx.make_view(lds.a0.ptr, LDS_layout_A)
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
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
        (fx.make_view(fx.get_iter(A), fx.make_layout(65536 * N, 1))),
        max_size=False,
        num_records_bytes=fx.Int64(M) * fx.Int64(N) * fx.Int64(elem_bytes),
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
        base_n = blk_idx * N_BLOCK
        total_threads = WAVE_NUM * 64
        step_elems = total_threads * load_elems
        for i in range_constexpr(M_REPEAT):
            elem_pos =  i * total_threads * load_elems + tid * load_elems
            m = elem_pos // N_BLOCK
            n = elem_pos % N_BLOCK
            n_swz = n
            if LDS_SWIZZLE:
                # n_swz = n ^ ((m % k_blocks16_dma) * elems_per_16b)
                n_swz = ((n // 8) ^ (m % 8)) * 8
            # print(f'###############################################')
            offset = ((bx_m + m) * N + base_n + n_swz)
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(gA_div, (None, fx.Int32(offset)))
            fx.copy(dma_atom, src, dst)
            lds_ptr = fx.add_offset(lds_ptr, step_elems)

    for block_idx in range(0, N_ITER):
        dma_a_to_lds(block_idx)
        # rocdl.s_waitcnt(0)
        gpu.barrier()
        fx.copy(lsd_copy_128b_atom, s2g_src, s2g_frag)
        fx.copy(copy_atom, s2g_frag, s2g_dst[None, None, None, block_idx])


PADDING_ELEMS = 16
PADDING_NUM = PADDING_ELEMS * 16

@fx.struct
class LDS_PADDING:
    a0: fx.Array[BFloat16, M_BLOCK*N_BLOCK+PADDING_NUM, 16]
@flyc.kernel
def asycn_copy_padding(
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
    A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(((8, M_BLOCK//8, M//M_BLOCK), N), ((M_BLOCK//8*N, N, M_BLOCK*N), 1))))
    
    # slicing A, B tensor
    bA = fx.flat_divide(A, (M_BLOCK, N_BLOCK))
    bB = fx.flat_divide(B, (M_BLOCK, N_BLOCK))
    bA = bA[None, None, bid, None]
    bB = bB[None, None, bid, None]
    
    # buffer load写入LDS的layout是自然的layout,
    # 没有padding的情况下，LDS的layout是 (M_BLOCK, N_BLOCK) : (N_BLOCK, 1), M_BLOCK这里是128
    # 拆解成 ((8, 16), N_BLOCK) : ((N_BLOCK, N_BLOCK*8), 1), 然后考虑padding,每8行之后做一个Padding,所以只需要把padding的stride加到`16`这个mode上
    lds = fx.SharedAllocator().allocate(LDS_PADDING).peek()   
    lds_layout_wr =fx.make_layout(((8, 16), N_BLOCK), ((N_BLOCK, 8*N_BLOCK+PADDING_ELEMS), 1))
    
    # LDS的读取使用的是16x4 thread copy_tiled, 模拟MFMA读取pattern, 4个wave , 2x2方式读取，模拟MFMA wave layout，但是只是读取和写出，没有MFMA计算，
    # 所以这里的读取和写出都存在重复的操作
    # 对于tensorA读取以及tensorB写出， wave0/wave1读取写出相同的内容，wave2/wave3读取写出相同的内容
    # LDS 读取的tiled_copy与写入B的tiled_copy 一致，wave分成2x2,其中一个mode `2`的stride为0， 代表这两个wave读取写入相同的logical_m,logical_n
    # lds_rd_tv_layout =  fx.make_layout(((16, 4, 2, 2), 8), ((1, 16*WAVE_NUM//2*8, 0, 16), WAVE_NUM*16//2))
    # 同样需要考虑LDS的读取layout 
    # LDS自然的layout是 ((8, 16), N_BLOCK), ((N_BLOCK, 8*N_BLOCK+PADDING_ELEMS), 1)， N block拆解成32x2:
    # ((8, 16), (32, N_BLOCK//2), ((N_BLOCK, 8*N_BLOCK+PADDING_ELEMS), (1, 32))
    # read LDS tiled copy 中 logical_m，按照（16， WAVE_NUM）排布，所以把16 放在mode0, sA_rd的layout就是：
    # ((16,  8),                     (32, N_BLOCK//32))：
    # ((8*N_BLOCK+PADDING_ELEMS, N_BLOCK), (1, 32))
    
    # 4个WAVE的 thread0-thread3会读LDS中的连续2行，由于这2行时连续的，所以这里就不用在M_BLOCK//16里把2再拆分出来了，如果想要拆分也可以，就是
    # ((16,                               2， 4),  (32, N_BLOCK//32))：
    # ((8*N_BLOCK+PADDING_ELEMS, 64， 64*2), (1, 32))
    
    # 同样的道理，lds_rd_tiled 的32行m 
    # each wave use 16 * 4 MFMA threads layout
    lds_rd_tile_mn = fx.make_tile(WAVE_NUM*16//2, 32)
    lds_rd_tv_layout =  fx.make_layout(((16, 4, 2, 2), 8), ((1, 16*WAVE_NUM//2*8, 0, 16), WAVE_NUM*16//2))
    lds_rd_tiled = fx.make_tiled_copy(buffer_copy_atom, lds_rd_tv_layout, lds_rd_tile_mn)
    lds_rd_thread = lds_rd_tiled.get_slice(tid)
    # fx.utils.print_typst(lds_rd_tiled, file="lds_rd_tiled.typ")
    sA_wr = fx.make_view(lds.a0.ptr, lds_layout_wr)
    
    # padding LDS read layout:
    lds_layout_rd =fx.make_layout(((16, M_BLOCK//16), (32, N_BLOCK//32)), ((M_BLOCK//16*N_BLOCK+PADDING_ELEMS, 64), (1, 32)))
    # lds_layout_rd =fx.make_layout(((16, 2, M_BLOCK//32), (32, N_BLOCK//32)), ((M_BLOCK//16*N_BLOCK+PADDING_ELEMS, 64, 128), (1, 32)))

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
    for block_idx in range(0, N_ITER):
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
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, N), (N, 1))))
    asycn_copy_padding(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=(M_BLOCK*N_BLOCK+PADDING_NUM)*2, stream=stream)


@flyc.jit
def async_dma_copy_swizzle(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, N), (N, 1))))
    asycn_copy_tile(arg_a_2d, B).launch(grid=(NUM_CU, 1, 1), block=(WAVE_NUM*64, 1, 1), smem=M_BLOCK*N_BLOCK*2, stream=stream)


# =====================================================================
# wg_buffer_load_lds_fp32 : LDS 尺寸 == 实际 copy 尺寸(不 padding 到 128b)
# 把 nelems 个 fp32 分成三段(都用 buffer_load_lds, 每 lane 分别搬 4/1/1 个 fp32):
#   1) DWORDx4(128b=4fp32) 满轮 : load1 = nelems // 2048 * 2048
#   2) DWORD  (32b=1fp32) 满轮 : rest1 = nelems-load1; load2 = rest1 // 512 * 512
#   3) DWORD  条件 lane        : part3 = rest1-load2 (<512), 只放行 tid<part3 的 lane
# 前两段都是「满员整轮」(512 lane 全部 in-bounds, 无需 mask); 只有第三段用 exec-mask。
# 三段在 LDS 上依次拼接、恰好铺满 [0,nelems), 单一 buffer resource(num_records=nelems*4)即可。
# =====================================================================
WG_WAVES = 8                                # 8 waves
WG_BLK = WG_WAVES * 64                       # 512 threads
# 选一个能同时触发三段的尺寸: nelems % 2048 >= 512 才有第二段, 且有余数才有第三段。
WG_F32_NELEMS = 2048 + 512 + 100            # 2660: part1=1轮DWORDx4, part2=1轮DWORD, part3=100 lane

@fx.struct
class SharedStorageF32:
    a0: fx.Array[Float32, WG_F32_NELEMS, 16]


def wg_buffer_load_lds_fp32(sA_ptr, gA, nelems, num_waves=WG_WAVES):
    """fp32 三段式 global->LDS DMA(LDS 恰好 = nelems 个 fp32, 不 padding)。

      1) DWORDx4(128b=4fp32) 满轮 : load1 = nelems // 2048 * 2048
      2) DWORD  (32b=1fp32) 满轮 : load2 = (nelems-load1) // 512 * 512
      3) DWORD  条件 lane        : part3 = 剩余(<512), scf.if(tid<part3) 放行
    前两段满轮全 in-bounds(无 mask); 只第三段用 exec-mask。gA 单一 resource
    (num_records=nelems*4) 即可——全程无 lane 越界读/写 LDS。
    """
    tid = fx.thread_idx.x
    wave_id = tid // 64
    WARP = 64
    LOAD4 = 4
    total = num_waves * WARP           # 512
    step4 = total * LOAD4              # 2048

    n1 = nelems // step4               # 第一段 DWORDx4 满轮数
    load1 = n1 * step4
    rest1 = nelems - load1            # 0..2047
    n2 = rest1 // total               # 第二段 DWORD 满轮数
    load2 = n2 * total
    part3 = rest1 - load2            # 0..511

    # ---- 第一段: DWORDx4(128b=4fp32) 满轮, 写 LDS[0..load1-1] ----
    # 只有 n1>0(nelems>=2048) 时才需要, 否则连 setup 都不发射。
    if const_expr(n1 > 0):
        dma128 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        wave_off4 = fx.rocdl.readfirstlane(fx.Int32.ir_type, wave_id * (WARP * LOAD4))  # wave_id*256
        lds4 = fx.add_offset(sA_ptr, wave_off4)
        for i in range_constexpr(n1):
            chunk = i * total + tid                          # 第几个 128b chunk
            dst = fx.make_view(lds4, fx.make_layout(1, 1))
            src = fx.slice(gA, (None, fx.Int32(chunk * LOAD4)))
            fx.copy(dma128, src, dst)
            lds4 = fx.add_offset(lds4, step4)

    # ---- 第二/三段共享 DWORD prologue: 只有还有剩余(rest1>0) 才需要 ----
    if const_expr(rest1 > 0):
        dma32 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        wave_off1 = fx.rocdl.readfirstlane(fx.Int32.ir_type, wave_id * WARP)            # wave_id*64
        lds1 = fx.add_offset(sA_ptr, wave_off1)
        lds1 = fx.add_offset(lds1, load1)                    # 基址 = load1(+per-wave wave_off1)

        # ---- 第二段: DWORD(32b=1fp32) 满轮, 写 LDS[load1..load1+load2-1] ----
        # 只有 n2>0(rest1>=512) 时才走。
        if const_expr(n2 > 0):
            for j in range_constexpr(n2):
                idx = load1 + j * total + tid                # 源/目标元素下标
                dst = fx.make_view(lds1, fx.make_layout(1, 1))
                src = fx.slice(gA, (None, fx.Int32(idx)))
                fx.copy(dma32, src, dst)
                lds1 = fx.add_offset(lds1, total)            # 每轮前进 512 个 fp32

        # ---- 第三段: DWORD 条件 lane(tid<part3), 写 LDS[load1+load2..nelems-1] ----
        # part3 可能跨多个 wave, 所以 lds1 带 wave_off1(per-wave); exec-mask 保证只有
        # tid<part3 的 lane 写 LDS, 恰好铺到 nelems-1, LDS 精确不越界。
        if const_expr(part3 > 0):
            cond = fxa.cmpi(fxa.CmpIPredicate.ult, tid, fxa.constant(part3, type=T.i32))
            if_op = _mlir_scf.IfOp(cond, [], has_else=False)
            if len(if_op.regions[0].blocks) == 0:
                if_op.regions[0].blocks.append(*[])
            with _mlir_ir.InsertionPoint(if_op.regions[0].blocks[0]):
                idx = load1 + load2 + tid
                dst = fx.make_view(lds1, fx.make_layout(1, 1))  # lds1 已前进到 load1+load2 基址
                src = fx.slice(gA, (None, fx.Int32(idx)))
                fx.copy(dma32, src, dst)
                _mlir_scf.YieldOp([])


@flyc.kernel(known_block_size=[512, 1, 1])
def wg_load_lds_fp32_test(A: fx.Tensor, B: fx.Tensor):
    tid = fx.thread_idx.x

    # 单一 buffer resource: num_records = nelems*4 bytes(三段全程 in-bounds)
    gA = fx.rocdl.make_buffer_tensor(
        fx.make_view(fx.get_iter(A), fx.make_layout(1 << 20, 1)),
        max_size=False,
        num_records_bytes=fx.Int64(WG_F32_NELEMS) * fx.Int64(4),
    )
    gA = fx.logical_divide(gA, fx.make_layout(1, 1))

    lds = fx.SharedAllocator().allocate(SharedStorageF32).peek()

    wg_buffer_load_lds_fp32(lds.a0.ptr, gA, WG_F32_NELEMS, num_waves=WG_WAVES)
    gpu.barrier()

    # ---- LDS -> global 回写: per-thread 线性(每线程 1 个 fp32), 越界由 num_records 丢弃 ----
    Bt_full = fx.rocdl.make_buffer_tensor(
        fx.make_view(fx.get_iter(B), fx.make_layout(1 << 20, 1)),
        max_size=False,
        num_records_bytes=fx.Int64(WG_F32_NELEMS) * fx.Int64(4),
    )
    Bt_full = fx.logical_divide(Bt_full, fx.make_layout(1, 1))
    uni32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    bc32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    rb_rounds = (WG_F32_NELEMS + WG_BLK - 1) // WG_BLK
    for j in range_constexpr(rb_rounds):
        idx = j * WG_BLK + tid
        lv = fx.make_view(fx.add_offset(lds.a0.ptr, idx), fx.make_layout(1, 1))
        fr = fx.make_fragment_like(lv)
        fx.copy(uni32, lv, fr)                       # ds_read 1 个 fp32
        dst = fx.slice(Bt_full, (None, fx.Int32(idx)))
        fx.copy(bc32, fr, dst)                       # idx>=nelems 由 num_records 丢弃


@flyc.jit
def run_wg_load_lds_fp32(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    a1d = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(WG_F32_NELEMS, 1)))
    b1d = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout(WG_F32_NELEMS, 1)))
    wg_load_lds_fp32_test(a1d, b1d).launch(
        grid=(1, 1, 1), block=(WG_BLK, 1, 1), smem=WG_F32_NELEMS * 4, stream=stream
    )



enable_dump_ir(False)
A = torch.arange(M * N, dtype=torch.bfloat16).reshape(M, N).cuda()
B = torch.zeros(M, N, dtype=torch.bfloat16).cuda()
C = torch.zeros(M, N, dtype=torch.bfloat16).cuda()
LDS_RESULT = A.clone().reshape(M//M_BLOCK, 8, M_BLOCK//8, N).permute(0, 2, 1, 3).contiguous().reshape(M, N)
print(f'{M=} {N=} {M_BLOCK=} {N_BLOCK=} {WAVE_NUM=}')
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


# ---- test fp32 exact-LDS DW-tail path (LDS == 实际 copy 尺寸, tail 用 DWORD) ----
At4 = torch.arange(WG_F32_NELEMS, dtype=torch.float32).cuda()
Bt4 = torch.zeros(WG_F32_NELEMS, dtype=torch.float32).cuda()
run_wg_load_lds_fp32(At4, Bt4, stream=torch.cuda.Stream())
torch.cuda.synchronize()
ok4 = torch.allclose(At4, Bt4)
print(f"wg_buffer_load_lds_fp32 (nelems={WG_F32_NELEMS}) correct:", ok4)
if not ok4:
    diff = (At4 != Bt4).nonzero().flatten()
    print(f'first mismatch idx={diff[:16].tolist()}')
