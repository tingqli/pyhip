import torch
import triton
import triton.language as tl
import pyhip

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)

@gluon.jit
def moe_gemm_final_reduce_bf16(
    in_ptr, out_ptr,
    num_tokens_wg, num_extra_tokens, num_tokens_total, 
    TOPK: gl.constexpr,
    OC: gl.constexpr,
    num_warps: gl.constexpr,):
    block_id = gl.program_id(0)
    block_num = gl.num_programs(0)

    if block_id < num_extra_tokens:
        num_tokens_wg += 1
        m_start = block_id * num_tokens_wg
    else:
        m_start = num_extra_tokens * (num_tokens_wg + 1) + (block_id - num_extra_tokens) * num_tokens_wg

    if m_start >= num_tokens_total:
        return
    in_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
                                 threads_per_warp=[1, 64],
                                 warps_per_cta=[num_warps, 1],
                                 order=[1, 0])
    off_m = m_start * TOPK + gl.arange(0, TOPK, layout=gl.SliceLayout(1, in_layout))
    off_k = gl.arange(0, 512, layout=gl.SliceLayout(0, in_layout))
    in_offsets = off_m[:, None] * OC + off_k[None, :]

    out_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8],
                                 threads_per_warp=[1, 64],
                                 warps_per_cta=[num_warps, 1],
                                 order=[1, 0])
    off_ok = gl.arange(0, 512, layout=gl.SliceLayout(0, out_layout))
    out_offsets = m_start * OC + off_ok[None, :]
    # gl.static_print('in_offsets shape:', in_offsets.shape)
    # gl.static_print('out_offsets shape:', out_offsets.shape)

    data = gl.amd.cdna3.buffer_load(in_ptr,
                                    in_offsets,
                                    )
    cur_in_offsets = in_offsets

    gl.static_assert(OC % 512 == 0, "OC must be multiple of 512")
    for j in range(0, num_tokens_wg - 1):
        for i in gl.static_range(0, OC, 512):
            if i == OC - 512:
                in_offsets = in_offsets + TOPK * OC
                cur_in_offsets = in_offsets
            else:
                cur_in_offsets += 512
            cur_out_offsets = out_offsets + i
            next_data = gl.amd.cdna3.buffer_load(in_ptr,
                                            cur_in_offsets,
                                            )
            data = gl.sum(data.to(gl.float32), axis=0, keep_dims=True).to(gl.bfloat16)
            # gl.static_print('data shape:', data.shape)
            # gl.static_print('out_offsets:', cur_out_offsets.shape)
            gl.amd.cdna3.buffer_store(data, out_ptr,
                                      cur_out_offsets)
            data = next_data

        out_offsets = out_offsets + OC

    if 1:
        for i in gl.static_range(0, OC, 512):
            cur_out_offsets = out_offsets + i
            if i == OC - 512:
                pass
            else:
                cur_in_offsets += 512
                next_data = gl.amd.cdna3.buffer_load(in_ptr,
                                                cur_in_offsets,
                                                )
            data = gl.sum(data.to(gl.float32), axis=0, keep_dims=True).to(gl.bfloat16)
            # gl.static_print('data shape:', data.shape)
            # gl.static_print('out_offsets:', cur_out_offsets.shape)
            gl.amd.cdna3.buffer_store(data, out_ptr,
                                      cur_out_offsets)
            data = next_data


def test():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    TOPK = 8
    OC = 4096
    num_tokens_total = 24000
    input = torch.randn(num_tokens_total, TOPK, OC, dtype=torch.bfloat16)
    output = torch.empty(num_tokens_total, OC, dtype=torch.bfloat16)
    num_CU = torch.cuda.get_device_properties().multi_processor_count
    num_WG = num_CU * 2
    
    num_tokens_wg = num_tokens_total // num_WG
    num_extra_tokens = num_tokens_total % num_WG
    '''
    num_big_wg = num_extra_tokens
    if wg_id < num_big_wg:
        tok0 = wg_id * (1 + num_tokens_wg) # need to do 1 more 
        tok1 = tok0 + (1 + num_tokens_wg)
    else:
        tok_base = num_big_wg * (1 + num_tokens_wg)
        tok0 = tok_base + (wg_id - num_big_wg) * num_tokens_wg
        tok1 = tok0 + num_tokens_wg
    '''

    print(num_WG, num_tokens_wg, num_extra_tokens, num_tokens_total)
    x=moe_gemm_final_reduce_bf16[(num_WG,)](
                               input,
                               output,
                               num_tokens_wg, num_extra_tokens, num_tokens_total, 
                               TOPK, OC,
                               num_warps=1)
    #print(x.asm['amdgcn'])
    
    ref = torch.zeros(num_tokens_total, OC, dtype=torch.float)
    for i in range(num_tokens_total):
        for t in range(TOPK):
            ref[i] += input[i, t]

    ref = ref.to(torch.bfloat16)
    for i in range(num_tokens_total):
        if not torch.allclose(ref[i], output[i]):
            print(i)
            # print(ref[i])
            # print(output[i])
            idx = torch.where(torch.abs(ref[i] - output[i]) > 0.05)
            print(f'{idx=}, {ref[i][idx]=}, {output[i][idx]=}')
            assert 0
    
    for _ in range(10):
        with pyhip.cuPerf(name="moe_gemm_final_reduce_bf16"):
            moe_gemm_final_reduce_bf16[(num_WG,)](
                                    input,
                                    output,
                                    num_tokens_wg, num_extra_tokens, num_tokens_total, 
                                    TOPK, OC,
                                    num_warps=1)
    # assert 0

    print('done')

test()