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

def gemm_test(bpreshuffle = False):
    from aiter.ops.shuffle import shuffle_weight

    mfaM = 16
    mfaN = 16
    mfaK = 32
    tileM = 2
    tileN = 2
    tileK = 8
    M = mfaM * tileM
    N = mfaN * tileN
    K = mfaK * tileK
    A = torch.randn([M, K], dtype=torch.bfloat16)*0.1
    B = torch.randn([N, K], dtype=torch.bfloat16)*0.1
    if bpreshuffle:
        weight = shuffle_weight(B)
    else:
        weight = B
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

@gluon.constexpr_function
def get_mem_ab_layout(tiles:gl.constexpr):
    reg_bases = [[0, 1], [0, 2], [0, 4]]
    lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]]
    warp_bases= []
    bit_pos = 16
    blk: gl.constexpr = tiles *16
    while tiles // 2:
        reg_bases.append([bit_pos, 0])
        bit_pos *= 2
        tiles = tiles // 2
    mem_ab_layout: gl.constexpr = gl.DistributedLinearLayout(
                                        reg_bases=reg_bases,
                                        lane_bases=lane_bases,
                                        warp_bases=warp_bases,
                                        block_bases=[],
                                        shape=[blk, 32])
    return mem_ab_layout

# [ n//16 ,K//32,4k,16n,8k,]
@gluon.constexpr_function
def get_mem_preshffule_layout():
    
    mem_b_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[8, 1],
                                            threads_per_warp=[64, 1],
                                            warps_per_cta=[1, 1],
                                            order=[0, 1])
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

    mem_a_offset_m = gl.arange(0, 16*tileM, layout=gl.SliceLayout(1, mem_a_layout))
    mem_a_offset_k = gl.arange(0, BK, layout=gl.SliceLayout(0, mem_a_layout))
    mem_a_offset = mem_a_offset_m[:,None] * K + mem_a_offset_k[None,:]
    
    if preshffuleB:
        mem_b_offset_n = gl.arange(0, tileN, layout=gl.SliceLayout(0, mem_b_layout))
        mem_b_offset_k = gl.arange(0, 64*8, layout=gl.SliceLayout(1, mem_b_layout))                
        mem_b_offset = mem_b_offset_n[None,:] * K * 16 + mem_b_offset_k[:,None]
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
    for k_start in gl.static_range(0, K, BK):
        a = gl.amd.cdna3.buffer_load(p_input, mem_a_offset)
        a = a.reshape(16*tileM, 4, 8).reshape(16*tileM, 32)
        a_fma = gl.convert_layout(a, a_fma_layout, assert_trivial=True)
        b = gl.amd.cdna3.buffer_load(p_weight, mem_b_offset)
        if preshffuleB:
            #[N//16, K//32, 4k, 16n, 8k]
            #[4k, 16N, 8k, N//16] -> [4k, 8k, N//16, 16N, ]
            b = b.reshape(4, 16, 8, tileN).permute(0, 2, 3, 1).reshape(32, 16*tileN)
        else:
            b = b.reshape(16*tileN, 4, 8).permute(1, 2, 0).reshape(32, 16*tileN)
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
gemm_test(bpreshuffle=False)
gemm_test(bpreshuffle=True)

