import torch
import triton
import triton.language as tl
from functools import cache

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)
from pyhip import div_up
from pyhip.contrib.gluon.utils import read_cycle, read_realtime, get_cu_id

def init_env():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

def gemm_test(bpreshuffle = False, transposeB = False):
    assert not (bpreshuffle and transposeB), f'weight can not be preshuffled and transposed'
    from aiter.ops.shuffle import shuffle_weight

    mfaM = 16
    mfaN = 16
    mfaK = 32
    tileM = 2
    tileN = 2
    tileK = 10
    M = mfaM * tileM
    N = mfaN * tileN
    K = mfaK * tileK
    A = torch.randn([M, K], dtype=torch.bfloat16)*0.1
    B = torch.randn([N, K], dtype=torch.bfloat16)*0.1
    if bpreshuffle:
        weight = shuffle_weight(B)
    else:
        # for the B weight(N=16, K=32) torch tensor. B.t() and B would have the same buffer but with diffrent shape and stride description
        # B.shape = [16, 32] B.stride() = [32, 1], B.t().shape = [32, 16], B.t().stride() = [1, 32]. B and B.t() share same buffer layout, but shape
        # and stride are both permuted. Since the linear layout is also descriping the shape and stride, which is similiar with torch tensor shape and stride()
        weight = B.t() if transposeB else B
    C = torch.zeros([M, N], dtype=torch.float32)
    C_ref = A @ B.T

    gemm_test_kernel[(1,)](
                A,
                weight,
                C,
                K=K,
                N=N,
                tileM = tileM,
                tileN = tileN,
                preshffuleB = bpreshuffle,
                num_warps=1)
    assert torch.allclose(C_ref.to(dtype=torch.float32), C, rtol=0.1, atol=0.03)
    print(f"pass {M=} {N=} {K=} {bpreshuffle=}")


#linear layout， 根据reg_bases->lane_bases->warp_base顺序tile数据，在每一层里，描述
#哪个维度先被tile进GPU寄存器。本测试中B tensor是[N, K]的layout，无论传输kernel的是b 还是
# b.t(),都采用同样的linear layout, b和b.t()实际是物理内存的排布是不变的。
# 这里的表述是对于64个lane,先用16个lane 排布dimention[0],然后是dimmention[1].由于 b和b.t()
# 有相同的物理内存排布（[N,K]），可以使用相同的linear layout.


@gluon.constexpr_function
def get_mem_ab_layout(tilemn:gl.constexpr, tilek:gl.constexpr=1):
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]]
    warp_bases= []
    bit_pos = 32
    blkk: gl.constexpr = tilek *32
    while tilek // 2:
        reg_bases.append([0, bit_pos])
        bit_pos *= 2
        tilek = tilek // 2
    
    bit_pos = 16
    blkmn: gl.constexpr = tilemn *16
    while tilemn // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        tilemn = tilemn // 2
    mem_ab_layout: gl.constexpr = gl.DistributedLinearLayout(
                                        reg_bases=reg_bases,
                                        lane_bases=lane_bases,
                                        warp_bases=warp_bases,
                                        block_bases=[],
                                        shape=[blkmn, blkk])
    return mem_ab_layout

# [ n//16 ,K//32,4k,16n,8k,]
@gluon.constexpr_function
def get_mem_preshffule_layout():
    # 如何理解order,如果把order简单的理解成为行优先或者列优先,将对后面的reshape, permute操作很难理解。
    # blocked layout和linear layout的本质上都是在描述, reg, lane , wave与tensor的映射关系。 相当于pyhip里
    # 每个lane需要找到这条lane对应的m, n.
    # 在tensor的每个维度上都将按照reg-> lane -> wave的顺序d读取并堆积(tile)数据到GPU VGPRs，这里的order实际上在描述维度
    # 堆积的顺序（当然通常最先堆积的维度时物理排布连续的）。但是如果理解成tile的顺序，就把实际tensor的物理内存排布与
    # 映射关系解耦，比如说，这个表示每个thread先排8个dimension 1的数据排进8个寄存器（通常时连续寄存器）。至于这8个数据是否在外存连续，
    # 这里可以暂时不考虑。
    mem_b_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
                                            threads_per_warp=[1, 64],
                                            warps_per_cta=[1, 1],
                                            order=[1, 0])
    return mem_b_layout


@gluon.constexpr_function
def get_mem_c_layout(tileM:gl.constexpr, tileN:gl.constexpr):
    reg_bases = [[0, 1], [0, 2]]
    lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]]
    warp_bases= []
    blkM: gl.constexpr = tileM *16
    blkN: gl.constexpr = tileN *16
    # First tileN or first Tile M would be handled by LLVM when generating disasembly code.
    bit_pos = 16
    while tileN // 2:
        reg_bases.append([0, bit_pos])
        bit_pos *= 2
        tileN = tileN // 2

    bit_pos = 16
    while tileM // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        tileM = tileM // 2
    mem_c_layout: gl.constexpr = gl.DistributedLinearLayout(
                                    reg_bases=reg_bases,
                                    lane_bases=lane_bases,
                                    warp_bases=warp_bases,
                                    block_bases=[],
                                    shape=[blkM, blkN])
    
    return mem_c_layout

@gluon.jit
def gemm_test_kernel(
    p_input,            # bf16 [M, K]
    p_weight,           # bf16 [N,K]
    p_output,           # float32 [M, N],
    K:gl.constexpr,
    N:gl.constexpr,
    tileM:gl.constexpr,
    tileN:gl.constexpr,
    preshffuleB: gl.constexpr,
    num_warps: gl.constexpr = 1,
):
    BK : gl.constexpr = 32

    mem_a_layout: gl.constexpr = get_mem_ab_layout(tileM)
    if preshffuleB:
        mem_b_layout: gl.constexpr = get_mem_preshffule_layout()
    else:
        mem_b_layout: gl.constexpr = get_mem_ab_layout(tileN)

    #获得每个lane mapping的 m, k，并根据mk以及实际的layout计算offset, 从HBM中读取。
    mem_a_offset_m = gl.arange(0, 16*tileM, layout=gl.SliceLayout(1, mem_a_layout))
    mem_a_offset_k = gl.arange(0, BK, layout=gl.SliceLayout(0, mem_a_layout))
    mem_a_offset = mem_a_offset_m[:,None] * K + mem_a_offset_k[None,:]
    
    if preshffuleB:
        mem_b_offset_n = gl.arange(0, tileN, layout=gl.SliceLayout(1, mem_b_layout)) #[tileN]
        mem_b_offset_k = gl.arange(0, 64*8, layout=gl.SliceLayout(0, mem_b_layout))  # [64*8]              
        mem_b_offset = mem_b_offset_n[:, None] * K * 16 + mem_b_offset_k[None,:] # [tileN,64*8]
    else:
        mem_b_offset_n = gl.arange(0, 16*tileN, layout=gl.SliceLayout(1, mem_b_layout))
        mem_b_offset_k = gl.arange(0, BK, layout=gl.SliceLayout(0, mem_b_layout))
        mem_b_offset = mem_b_offset_n[:,None] * K + mem_b_offset_k[None,:]

    mem_c_layout: gl.constexpr = get_mem_c_layout(tileM=tileM, tileN=tileN)
    c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                            instr_shape=[16, 16, 32],
                                            transposed=True,
                                            tiles_per_warp=[tileM, tileN],
                                            warps_per_cta=[1, 1])
    a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
    b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    acc = gl.zeros((16*tileM, 16*tileN), dtype=gl.float32, layout=c_layout)
    
    # gl.static_print(">>>>>> fma_A ", gl.to_linear_layout(a_fma_layout, [16, 32]))
    # gl.static_print(">>>>>> fma_B ", gl.to_linear_layout(b_fma_layout, [32, 16]))
    # gl.static_print(">>>>>> fma_C ", gl.to_linear_layout(c_layout, [16, 16]))
    for k_start in gl.static_range(0, K, BK):
        # gl.static_print(">>>>>>?>>>>>>>>>>>")

        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offset)
        a = a.reshape(16*tileM, 4, 8).reshape(16*tileM, 32)
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offset)
        if preshffuleB:
            if k_start == 0:
                gl.static_print(">>>>>> layout0 ", b.type)
                #  <['2', '512'], bf16, BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[1, 64], warps_per_cta=[1, 1], order=[1, 0], cga_layout=[])>
                #  在第一个维度按lane顺序，每个tile 8. preshuffle的layout[tileN, 4k, 16n, 8k] squeeze成[tileN, 64x8]
                gl.static_print(">>>>>> layout0,+reshape ", b.reshape(tileN, 4, 16, 8).type)
                #  <['2', '4', '16', '8'], bf16, BlockedLayout(size_per_thread=[1, 1, 1, 8], threads_per_warp=[1, 4, 16, 1], warps_per_cta=[1, 1, 1, 1], order=[3, 2, 1, 0], cga_layout=[]
                # ['2', '4', '16', '8'] is [tileN, 4k, 16n, 8k],根据order的顺序是 先tile 8k, 然后16n, 然后4k, 然后tileN
                gl.static_print(">>>>>> layout0,+permute ", b.reshape(tileN, 4, 16, 8).permute(1, 3, 0, 2).type)
                #  <['4', '8', '2', '16'], bf16, BlockedLayout(size_per_thread=[1, 8, 1, 1], threads_per_warp=[4, 1, 1, 16], warps_per_cta=[1, 1, 1, 1], order=[1, 3, 0, 2], cga_layout=[])>
                # ['4', '8', '2', '16'] is [4k, 8k, tileN 16n],根据order的顺序[1, 3, 0, 2], 是 先tile 8k, 然后16n, 然后4k, 然后tileN
                gl.static_print(">>>>>> layout0 ", b.reshape(tileN, 4, 16, 8).permute(1, 3, 0, 2).reshape(32, 16*tileN).type)
                #  <['32', '32'], bf16, DistributedLinearLayout(reg_bases=[[1, 0], [2, 0], [4, 0], [0, 16]], lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp_bases=[], block_bases=[], shape=[32, 32])>
            
            #[tileN, 64lane*8k] -> [tileN, 4k, 16n, 8k] -> [4k, 8k, tileN, 16n,] -> [32k, tileN*16N]
            # The process is like the reverse process of preshuffle.
            b = b.reshape(tileN, 4, 16, 8).permute(1, 3, 0, 2).reshape(32, 16*tileN)
        else:
            b = b.permute(1, 0)
        b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
        mem_a_offset += BK
        if preshffuleB:
            mem_b_offset += BK*16
        else:
            mem_b_offset += BK
        
    out_offsets_m = (gl.arange(0, 16*tileM, layout=gl.SliceLayout(1, mem_c_layout)))
    out_offsets_n = (gl.arange(0, 16*tileN, layout=gl.SliceLayout(0, mem_c_layout)))
    out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
    gl.amd.cdna3.buffer_store(gl.convert_layout(acc.reshape(16*tileM, 16*tileN), mem_c_layout, assert_trivial=True), p_output, out_offsets)

init_env()
gemm_test(bpreshuffle=False, transposeB = True)
gemm_test(bpreshuffle=False, transposeB = False)
gemm_test(bpreshuffle=True)



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
@gluon.constexpr_function
def buffer_load_layout(k_lanes:gl.constexpr):
    mn_lanes: gl.constexpr = 64 // k_lanes
    load_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
                                            threads_per_warp=[mn_lanes, k_lanes],
                                            warps_per_cta=[1, 1],
                                            order=[1, 0])

    return load_layout

@gluon.constexpr_function
def get_lds_layout_write():
    lds_layout_write: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    return lds_layout_write

@gluon.constexpr_function
def get_lds_layout_read(mn:gl.constexpr, k:gl.constexpr):
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    ktile:gl.constexp = k // 32
    bit_pos:gl.constexp = 32
    while ktile // 2:
        reg_bases.append([0, bit_pos])
        bit_pos *= 2
        ktile = ktile // 2

    mtile:gl.constexp = mn // 16
    bit_pos:gl.constexp = 16
    while mtile // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        mtile = mtile // 2

    lds_layout_read: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 8],
            [0, 16],
        ],
        warp_bases=[],
        block_bases=[],
        shape=[
            mn,
            k,
        ],
    )
    return lds_layout_read


@gluon.jit
def gemm_test_kernel_lds(
    p_input,            # bf16 [M, K]
    p_weight,           # bf16 [N,K]
    p_output,           # float32 [M, N],
    K:gl.constexpr,
    N:gl.constexpr,
    tileM:gl.constexpr,
    tileN:gl.constexpr,
    preshffuleB: gl.constexpr,
    num_warps: gl.constexpr = 1,
):
    BK : gl.constexpr = 64
    tileK : gl.constexpr = BK // 32
    assert preshffuleB == False
    mem_a_layout: gl.constexpr = get_mem_ab_layout(tilemn=tileM, tilek=tileK)
    mem_b_layout: gl.constexpr = buffer_load_layout(k_lanes=8)

    #获得每个lane mapping的 m, k，并根据mk以及实际的layout计算offset, 从HBM中读取。
    mem_a_offset_m = gl.arange(0, 16*tileM, layout=gl.SliceLayout(1, mem_a_layout))
    mem_a_offset_k = gl.arange(0, BK, layout=gl.SliceLayout(0, mem_a_layout))
    mem_a_offset = mem_a_offset_m[:,None] * K + mem_a_offset_k[None,:]
    

    mem_b_offset_n = gl.arange(0, 16*tileN, layout=gl.SliceLayout(1, mem_b_layout))
    mem_b_offset_k = gl.arange(0, BK, layout=gl.SliceLayout(0, mem_b_layout))
    mem_b_offset = mem_b_offset_n[:,None] * K + mem_b_offset_k[None,:]

    mem_c_layout: gl.constexpr = get_mem_c_layout(tileM=tileM, tileN=tileN)
    c_layout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                            instr_shape=[16, 16, 32],
                                            transposed=True,
                                            tiles_per_warp=[tileM, tileN],
                                            warps_per_cta=[1, 1])
    a_fma_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, k_width=8)
    b_fma_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, k_width=8)
    acc = gl.zeros((16*tileM, 16*tileN), dtype=gl.float32, layout=c_layout)
    
    
    lds_layout_write: gl.constexpr = get_lds_layout_write()
    lds_layout_read: gl.constexpr = get_lds_layout_read(mn=16*tileN, k=BK)
    lds_b = gl.allocate_shared_memory(gl.bfloat16, [tileN*16, BK], layout=lds_layout_write,)

    for k_start in gl.static_range(0, K, BK):
        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offset)
        a = a.reshape(16*tileM, BK)
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offset)
        b = b.reshape(16*tileN, BK)
        lds_b.store(b)
        b = (
                lds_b.load(layout=lds_layout_read)
                .permute(1, 0)
                .reshape(BK,16*tileN)
            )

        b_fma = gl.convert_layout(b, b_fma_layout, assert_trivial=True)
        acc = gl.amd.cdna4.mfma(a_fma, b_fma, acc)
        mem_a_offset += BK
        mem_b_offset += BK
        
    out_offsets_m = (gl.arange(0, 16*tileM, layout=gl.SliceLayout(1, mem_c_layout)))
    out_offsets_n = (gl.arange(0, 16*tileN, layout=gl.SliceLayout(0, mem_c_layout)))
    out_offsets = out_offsets_m[:, None] * N + out_offsets_n[None, :]
    gl.amd.cdna3.buffer_store(gl.convert_layout(acc.reshape(16*tileM, 16*tileN), mem_c_layout, assert_trivial=True), p_output, out_offsets)

def gemm_test_lds():
    mfaM = 16
    mfaN = 16
    mfaK = 32
    tileM = 2
    tileN = 2
    tileK = 40
    M = mfaM * tileM
    N = mfaN * tileN
    K = mfaK * tileK
    A = torch.randn([M, K], dtype=torch.bfloat16)*0.1
    B = torch.randn([N, K], dtype=torch.bfloat16)*0.1


    weight =  B
    C = torch.zeros([M, N], dtype=torch.float32)
    C_ref = A @ B.T

    gemm_test_kernel_lds[(1,)](
                A,
                weight,
                C,
                K=K,
                N=N,
                tileM = tileM,
                tileN = tileN,
                preshffuleB = False,
                num_warps=1)
    assert torch.allclose(C_ref.to(dtype=torch.float32), C, rtol=0.1, atol=0.03)
    print(f"[LDS]pass {M=} {N=} {K=} ")

gemm_test_lds()