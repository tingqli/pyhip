import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )

from triton.experimental.gluon.language.amd.cdna3 import (
    sched_group_barrier as _amd_iglp_sched_group_barrier,
)

@gluon.jit
def softmax_4w_4x16(
    in_ptr, out_ptr,
    K,
    M_BLOCK: gl.constexpr,
    K_BLOCK: gl.constexpr,
    num_warps: gl.constexpr,):
    m_block = gl.program_id(0)
    gl.static_assert(M_BLOCK % (4 * num_warps) == 0, "M_BLOCK must be multiple of 4*num_warps")
    in_layout: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 4],
                                 threads_per_warp=[4, 16],
                                 warps_per_cta=[num_warps, 1],
                                 order=[1, 0])
    off_m = m_block * M_BLOCK + gl.arange(0, M_BLOCK, layout=gl.SliceLayout(1, in_layout))
    off_k = gl.arange(0, K_BLOCK, layout=gl.SliceLayout(0, in_layout))
    offsets = off_m[:, None] * K + off_k[None, :]
    for i in range(0, K, K_BLOCK):
        in_offsets = offsets + i
        out_offsets = offsets + i
        data = gl.amd.cdna3.buffer_load(in_ptr,
                                        in_offsets,
                                        )
        gl.static_print('data shape:', data.shape)
        max = gl.max(data, axis=1, keep_dims=True)
        data = gl.reshape(data - max, [data.shape[0] // 4, 4, K_BLOCK])
        #data = gl.convert_layout(data, layout=in_layout3d, assert_trivial=True)
        log2e:gl.constexpr = 1.4426950408889634
        data = gl.exp2(data * log2e)
        sum = gl.sum(data, axis=2, keep_dims=True)
        sum_inv = 1.0 / sum
        data = data * sum_inv
        data = gl.reshape(data, [data.shape[0] * 4, K_BLOCK])
        gl.amd.cdna3.buffer_store(data, out_ptr,
                                  out_offsets)


def test():
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.set_default_device('cuda')
    torch.manual_seed(0)

    M = 32
    K = 128
    M_BLOCK = 16
    K_BLOCK = 128
    input = torch.randn([M_BLOCK, K_BLOCK], dtype=torch.float32)
    output = torch.zeros([M_BLOCK, K_BLOCK], dtype=torch.float32)
    num_warps = 4
    x = softmax_4w_4x16[(M // (num_warps * 4),)](
        input, output, K,
        M_BLOCK,
        K_BLOCK,
        num_warps=num_warps,
        )
    #print(x.asm['amdgcn'])
    ref = torch.nn.functional.softmax(input, dim=1)
    torch.testing.assert_close(output, ref)

    # output[:] = 0
    # x = softmax_splitk[(M // 4,)](
    #     input, output, K,
    #     M_BLOCK,
    #     K_BLOCK,
    #     num_warps=num_warps,
    #     )
    # torch.testing.assert_close(output, ref)

    print('done')

test()