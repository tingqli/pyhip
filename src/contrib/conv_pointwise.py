import pyhip

__all__ = [
    "conv_pointwise", "conv_pointwise_jit"
]

import os
USE_GLUON = int(os.getenv("USE_GLUON", "0"))

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
def conv_pointwise(input, weight, bias,
                    stride=1,
                    padding=0, 
                    dilation=1,
                    groups=1,
                    use_gluon = None):
    assert stride == 1, stride
    assert padding == 0, padding
    assert dilation == 1, dilation
    assert str(input.dtype) == 'torch.bfloat16'

    global torch
    import torch
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    B = input.shape[0]
    C = input.shape[1]
    DHW = input.numel() // (B*C)

    assert C % groups == 0
    group_size = C//groups

    if use_gluon is None:
        use_gluon = USE_GLUON
    if use_gluon:
        grid = (B, C//group_size)
        conv_pointwise_gl[grid](group_size,
                                input,
                                weight,
                                output,
                                bias,
                                B, C, DHW, num_warps=4)
    else:
        conv_pointwise_jit([B, C//group_size],
                        [256], group_size,
                        input.data_ptr(),
                        weight.data_ptr(),
                        output.data_ptr(),
                        bias.data_ptr(),
                        B, C, DHW)
    return output

@pyhip.jit(with_debug_log=False)
def conv_pointwise_jit(J, group_size,
           input:"void*",
           weight:"void*",
           output:"void*",
           bias:"void*",
           B:"int",
           C:"int",
           DHW:"int",
           ):
    """
    input:  [B, C, D*H*W]
    weight: [C, group_size, 1, 1, 1]
    bias  : [C]
    output: [B, C, D*H*W]

    因为是point-wise, DHW可以展平为1D
    gridDims: [B, C//group_size]

    每个kernel完成 [1, group_size, D*H*W] 这么多输出点
    读入数据 [1, group_size, D*H*W] 这么多，

    假定 group_size 足够小，输入点沿着 DHW 维度直接读入到 group_size 个寄存器，
    然后这些寄存器使用fmac组合得到 group_size 个输出直接写出
    """
    blk_B = J.blockIdx.x
    blk_g = J.blockIdx.y

    offset = J.gpr("su32", (blk_B * C + blk_g * group_size) * DHW * J.sizeof_bf16)
    input += offset
    output += offset
    weight += (blk_g * group_size) * group_size * J.sizeof_bf16
    bias += (blk_g * group_size) * J.sizeof_bf16

    num_warps = 4
    num_threads = num_warps * 64
    # load group_size weight & bias
    # 
    B = J.gpr(group_size, group_size, "vf32", 0)
    vbias = J.gpr(group_size, "vf32", 0x00008000) # f32=>bf16 bias
    A = J.gpr(group_size, 4, "vf32")
    C = J.gpr(2, group_size, 4, "vf32")

    buff_a = J.Buffer(input, group_size * DHW * J.sizeof_bf16)
    buff_c = J.Buffer(output, group_size * DHW * J.sizeof_bf16)

    # load weights: broadcast to all spatial lanes
    for g0 in range(group_size):
        for g1 in range(group_size):
            vaddr = J.gpr("vu32", (g0*group_size + g1) * J.sizeof_bf16)
            J.global_load_short_d16_hi(B[g0, g1], vaddr, weight)

    # load bias: broadcast to all spatial lanes
    for g in range(group_size):
        vaddr = J.gpr("vu32", g * J.sizeof_bf16)
        J.global_load_short_d16_hi(vbias[g], vaddr, bias)

    voffset = J.gpr(group_size, "vu32")
    for g in range(group_size):
        voffset[g] = J.threadIdx.x * J.sizeof_DW4 + DHW * (g * J.sizeof_bf16)

    dhw0 = J.gpr("su32", J.warp_id*64*J.sizeof_DW4 // J.sizeof_bf16)
    with J.While(dhw0 < DHW):
        with J.ExecMask(voffset[0] < DHW*J.sizeof_bf16):
            for g in range(group_size):
                buff_a.load_dwordx4(A[g], voffset[g], 0)

            J.s_waitcnt(mod="vmcnt(0)")
            
            for g0 in range(group_size):
                C[0,g0,0] = vbias[g0]
                C[0,g0,1] = vbias[g0]
                C[0,g0,2] = vbias[g0]
                C[0,g0,3] = vbias[g0]
                for g1 in range(group_size):
                    J.v_fmac_f32(C[0,g0,0], A[g1,0] << 16, B[g0, g1])
                    J.v_fmac_f32(C[0,g0,1], A[g1,1] << 16, B[g0, g1])
                    J.v_fmac_f32(C[0,g0,2], A[g1,2] << 16, B[g0, g1])
                    J.v_fmac_f32(C[0,g0,3], A[g1,3] << 16, B[g0, g1])

            for g0 in range(group_size):
                C[1,g0,0] = vbias[g0]
                C[1,g0,1] = vbias[g0]
                C[1,g0,2] = vbias[g0]
                C[1,g0,3] = vbias[g0]
                for g1 in range(group_size):
                    J.v_fmac_f32(C[1,g0,0], A[g1,0] & 0xFFFF0000, B[g0, g1])
                    J.v_fmac_f32(C[1,g0,1], A[g1,1] & 0xFFFF0000, B[g0, g1])
                    J.v_fmac_f32(C[1,g0,2], A[g1,2] & 0xFFFF0000, B[g0, g1])
                    J.v_fmac_f32(C[1,g0,3], A[g1,3] & 0xFFFF0000, B[g0, g1])

            for g0 in range(group_size):
                J.v_perm_b32(C[0,g0,0], C[0,g0,0], C[1,g0,0], J.get_sgpr_const(0x03_02_07_06))
                J.v_perm_b32(C[0,g0,1], C[0,g0,1], C[1,g0,1], J.get_sgpr_const(0x03_02_07_06))
                J.v_perm_b32(C[0,g0,2], C[0,g0,2], C[1,g0,2], J.get_sgpr_const(0x03_02_07_06))
                J.v_perm_b32(C[0,g0,3], C[0,g0,3], C[1,g0,3], J.get_sgpr_const(0x03_02_07_06))

            for g0 in range(group_size):
                buff_c.store_dwordx4(C[0,g0], voffset[g0], 0)

        for g0 in range(group_size):
            voffset[g0] += num_threads * J.sizeof_DW4

        dhw0 += num_threads * J.sizeof_DW4 // J.sizeof_bf16

    pass

"""
Same idea : gluon version 
"""
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def conv_pointwise_gl(group_size:gl.constexpr,
            input_ptr,   # input:  [B, C, D*H*W] bf16
            weight_ptr,  # weight: [C, group_size, 1, 1, 1] bf16 = [num_groups, group_size, group_size, 1, 1, 1]
            output_ptr,  # output: [B, C, D*H*W] bf16
            bias_ptr,    # bias  : [C] bf16
            B, C, DHW):
        blk_B = gl.program_id(0)
        blk_g = gl.program_id(1)

        offset0 = (blk_B * C + blk_g * group_size) * DHW
        input_ptr += offset0
        output_ptr += offset0

        # load weights: same offsets for all threads/warps
        weight_layout:gl.constexpr = gl.BlockedLayout([1, group_size*group_size], [64, 1], [4, 1], order=[1, 0])
        weight_offsets = gl.arange(0, group_size*group_size, layout=gl.SliceLayout(0, weight_layout))
        weight0 = gl.amd.cdna3.buffer_load(weight_ptr, blk_g * group_size * group_size + weight_offsets)
        weight = weight0.reshape(group_size, group_size, 1)

        # load bias: same offsets for all threads/warps
        bias_layout:gl.constexpr = gl.BlockedLayout([1, group_size], [64, 1], [4, 1], order=[1, 0])
        bias_offsets = gl.arange(0, group_size, layout=gl.SliceLayout(0, bias_layout))
        bias = gl.amd.cdna3.buffer_load(bias_ptr, blk_g * group_size + bias_offsets)

        # DW4 ~ 8 bf16
        input_layout:gl.constexpr = gl.BlockedLayout([group_size, 8], [1, 64], [1, 4], order=[1, 0])
        indices_col = gl.arange(0, 4*64*8, gl.SliceLayout(0, input_layout))
        indices_row = gl.arange(0, group_size, gl.SliceLayout(1, input_layout))

        weight = weight.to(gl.float32)
        bias = bias.to(gl.float32)

        for i in range(0, DHW, 4*64*8):
            # [group_size, 8*64*4]
            offsets = i + indices_row[:,None] * DHW + indices_col[None, :]
            mask = (i + indices_col) < DHW
            input = gl.amd.cdna3.buffer_load(input_ptr, offsets, mask[None, :])
            input = input.to(gl.float32)

            # [gs,  gs]  =>  [gs,gs,1]
            # [gs, 256]  =>  [ 1,gs,256]
            #      => [gs, gs, 256]
            i0 = input.reshape(1, group_size, 4*64*8)
            w0 = gl.convert_layout(weight, i0.type.layout, assert_trivial=True)

            o0 = (i0 * w0).sum(axis=1)
            
            b0 = gl.convert_layout(bias.reshape(group_size, 1), o0.type.layout, assert_trivial=True)
            o0 += b0

            output = gl.convert_layout(o0.to(gl.bfloat16), input.type.layout, assert_trivial=True)
            gl.amd.cdna3.buffer_store(output, output_ptr, offsets, mask[None, :])


    __all__.append("conv_pointwise_gl")

except Exception as e:
    print(f"failed to add gluon version due to : ", e)
