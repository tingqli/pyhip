import pyhip

__all__ = [
    "conv_depthwise_3d", "conv_depthwise_3d_jit"
]

import os
USE_GLUON = int(os.getenv("USE_GLUON", "0"))

# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html
def conv_depthwise_3d(input, weight, bias,
                    stride,
                    padding, 
                    dilation,
                    groups,
                    use_gluon = None):
    for s in stride: assert s == 1, s
    for d in dilation: assert d == 1, d

    assert str(input.dtype) == 'torch.bfloat16'

    global torch
    import torch
    output = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    B, C_in, D, H, W = input.shape
    C_out, C_g, KD, KH, KW = weight.shape

    assert padding[0] == 0
    assert padding[1] == KH//2
    assert padding[2] == KW//2

    assert C_out == C_in
    assert C_g * groups == C_in
    assert C_g == 1, f"depthwise assumes C_g == 1 but got {C_g}"

    D_out = (D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) // stride[2] + 1

    output = torch.zeros(B, C_out, D_out, H_out, W_out, dtype=input.dtype, device=input.device)

    if use_gluon is None:
        use_gluon = USE_GLUON
    if use_gluon:
        grid = (B, C_out, D_out)
        conv_depthwise_3d_gl[grid](KD, KH, KW, H_out, W_out,
                                   input, weight, output, bias, B, C_out, D, D_out)
    else:
        conv_depthwise_3d_jit([B, C_out,D_out], [256],
                        KD, KH, KW, H_out, W_out,
                        input.data_ptr(),
                        weight.data_ptr(),
                        output.data_ptr(),
                        bias.data_ptr(),
                        B, C_out, D, D_out)
    return output

@pyhip.jit(with_debug_log=False)
def conv_depthwise_3d_jit(J, KD, KH, KW, H, W,
           input:"void*",
           weight:"void*",
           output:"void*",
           bias:"void*",
           B:"int",
           C:"int",
           D0:"int",
           D1:"int",
           ):
    """
    input:  [B, C, D0, H, W]
    weight: [C, 1, KD, KH, KW]
    bias  : [C]
    output: [B, C, D1, H, W]
    
    假设：
      1. 每个channel的权重足够小可以被寄存器保存得下
      2. B*C*D1 足够大可以产生足够多的任务数占满所有CU
      3. KD*16*16足够小可以在LDS放得下
    
    每个work-group负责计算 H*W 这么多输出数据点
    权重，外存读入LDS然后被各个warp再复制读取
    输入，外存读入LDS，然后从LDS中读入 KD*KH*KW 次 fmac得到结果
    """
    H0, W0 = H,W
    H1, W1 = H,W
    
    blk_B = J.blockIdx.x[0]
    blk_C = J.blockIdx.y[0]
    blk_D = J.blockIdx.z[0]

    blk_in = J.gpr("su32", (blk_B * C + blk_C) * D0 + blk_D)
    blk_out = J.gpr("su32", (blk_B * C + blk_C) * D1 + blk_D)

    # each WG handles W output elements
    input[:] += blk_in * H0 * W0 * J.sizeof_bf16
    output[:] += blk_out * H1 * W1 * J.sizeof_bf16
    weight[:] += blk_C * KD * KH * KW * J.sizeof_bf16
    bias[:] += blk_C * J.sizeof_bf16

    vbias = J.gpr("vf32", 0x00008000)
    J.global_load_short_d16_hi(vbias, J.gpr("vu32", 0), bias)

    buff_a = J.Buffer(input, KD * H0 * W0 * J.sizeof_bf16)
    buff_c = J.Buffer(output, H1 * W1 * J.sizeof_bf16)

    # wach WG compute all inner most dimensions : W <=> threadIdx
    # cooperatively load KD*KH*(W + KW//2 + KW//2) into LDS
    padded_H = KH//2 + H + KH//2
    padded_W = KW//2 + W + KW//2
    input_size = KD * padded_H * padded_W * J.sizeof_bf16
    weight_size = KD * KH * KW * J.sizeof_bf16
    lds_weight = J.alloc_lds(J.div_up(weight_size, 64)*64)
    lds_input = J.alloc_lds(input_size)

    num_warps = 4
    num_threads = num_warps * 64

    # fill the padding part in LDS [KD, padded_H, padded_W]
    vzeros = J.gpr(4, "vu32", 0)
    voff = J.gpr("vu32", lds_input + J.threadIdx.x[0]*J.sizeof_DW4)
    for i in range(0, input_size, num_threads*J.sizeof_DW4):
        J.ds_write_b128(voff, vzeros)
        voff[0] += num_threads*J.sizeof_DW4

    J.s_waitcnt(mod=f"lgkmcnt(0)")
    J.s_barrier()

    # load input [KD, H, W] into LDS [KD, padded_H, padded_W]
    for d in range(KD):
        d_off_lds = ((d * padded_H + KH//2) * padded_W + KW//2) * J.sizeof_bf16
        d_off_vmem = d*H*W*J.sizeof_bf16
        h_off_lds = J.warp_id[0] * (padded_W * J.sizeof_bf16)
        h_off_vmem = J.gpr("vu32", J.warp_id[0]) * (W * J.sizeof_bf16)

        m0_base = J.gpr("su32", lds_input + d_off_lds + h_off_lds)
        vm_base = J.gpr("vu32", d_off_vmem + h_off_vmem + J.lane_id[0] * J.sizeof_DW)
        h = J.gpr("su32", J.warp_id[0])
        with J.While(h[0] < H):
            J.s_mov_b32("m0", m0_base)
            for w in range(0, W, 128):
                with J.ExecMask(J.lane_id[0]*2 < (W-w)):
                    buff_a.load_dword(None, vm_base, 0, offset12=w*J.sizeof_bf16)

            m0_base[0] += num_warps * padded_W * J.sizeof_bf16
            vm_base[0] += num_warps * W * J.sizeof_bf16
            h[0] += num_warps

    J.s_waitcnt(mod=f"vmcnt(0)")
    J.s_barrier()

    J.wg_load_lds(lds_weight, weight, weight_size, num_warps = num_warps, wait_barrier = True)

    # load all weight into registers from LDS (broadcast to all lanes)
    vaddr = J.gpr("vu32", lds_weight)
    B = J.gpr(KD, KH, KW, "f32")
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                B[kd, kh, kw] = 0
                J.ds_read_u16_d16_hi(B[kd, kh, kw], vaddr, mod=f"offset:{((kd*KH + kh)*KW + kw) * J.sizeof_bf16}")
    J.s_waitcnt(mod=f"lgkmcnt(0)")

    A = J.gpr(KD, KH, J.div_up(KW,2), "f32", 0)
    conv_tasks = []
    for kd in range(KD):
        for kh in range(KH):
            conv_tasks.append((kd, kh))

    h = J.gpr("su32", J.warp_id[0])
    with J.While(h < H):
        for w in range(0, W, 128): 
            # process 1x64 outputs per warp
            vaddr = J.gpr("vu32", lds_input + \
                                    J.gpr("vu32", J.gpr("su32", h[0] * (padded_W * J.sizeof_bf16))) + \
                                    w * J.sizeof_bf16 + J.lane_id[0] * J.sizeof_u32)
            C = J.gpr(2, "f32", vbias) 
            for kd, kh in conv_tasks:
                for kw in range(0,KW,2):
                    J.ds_read_b32(A[kd, kh, kw//2], vaddr, mod=f"offset:{((kd*padded_H + kh)*padded_W + kw) * J.sizeof_bf16}")

            J.s_waitcnt(mod=f"lgkmcnt({0})")

            for kd, kh in conv_tasks:
                for kw in range(KW):
                    if kw & 1:
                        J.v_fmac_f32(C[0], A[kd, kh, kw//2] & 0xFFFF0000, B[kd, kh, kw])
                        J.v_fmac_f32(C[1], A[kd, kh, (kw+1)//2] << 16, B[kd, kh, kw])
                    else:
                        J.v_fmac_f32(C[0], A[kd, kh, kw//2] << 16, B[kd, kh, kw])
                        J.v_fmac_f32(C[1], A[kd, kh, kw//2] & 0xFFFF0000, B[kd, kh, kw])

            J.v_perm_b32(C[0], C[0], C[1], J.get_sgpr_const(0x03_02_07_06))

            vcol_addr = J.gpr("vu32", w * J.sizeof_bf16 + J.lane_id[0]*J.sizeof_u32)
            with J.ExecMask(vcol_addr < W * J.sizeof_bf16):
                J.global_store_dword(vcol_addr + J.gpr("su32", h[0] * (W*J.sizeof_bf16)), C[0], output)
        h[0] += num_warps

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

    @gluon.constexpr_function
    def next_power_of_2(n: int):
        """Return the smallest power of 2 greater than or equal to n"""
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        n += 1
        return n

    @gluon.jit
    def conv_depthwise_3d_gl(
            KD: gl.constexpr, KH: gl.constexpr, KW: gl.constexpr, H: gl.constexpr, W: gl.constexpr,
            input_ptr,  # [B, C, D0, H, W]
            weight_ptr, # [C, 1, KD, KH, KW]
            output_ptr, # [B, C, D1, H, W]
            bias_ptr,   # [C]
            B, C, D0, D1):
        # grid = (B, C_out, D_out)
        blk_B = gl.program_id(0)
        blk_C = gl.program_id(1)
        blk_D = gl.program_id(2)

        input_ptr += ((blk_B * C + blk_C) * D0 + blk_D) * (H * W)
        output_ptr += ((blk_B * C + blk_C) * D1 + blk_D) * (H * W)
        weight_ptr += blk_C * KD * KH * KW
        bias_ptr += blk_C

        padded_H: gl.constexpr = KH//2 + H + KH//2
        padded_W: gl.constexpr = KW//2 + W + KW//2

        KD2 : gl.constexpr = next_power_of_2(KD)
        padded_H2 : gl.constexpr = next_power_of_2(padded_H)
        padded_W2 : gl.constexpr = next_power_of_2(padded_W)

        gl.static_print(">>>>>>>>", KD2, padded_H2, padded_W2, KD2*padded_H2*padded_W2*2)

        smem_input_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0,1,2])
        smem_input = gl.allocate_shared_memory(gl.bfloat16, [KD2, padded_H2, padded_W2], layout=smem_input_layout)

        # clear paddings
        vzeros = gl.zeros([KD2, padded_H2, padded_W2], gl.bfloat16,
                            layout=gl.BlockedLayout([1,1,1],[1,1,64],[1,4,1],[0,1,2]))
        smem_input.store(vzeros)

        false:gl.constexpr = 0
        gl.static_assert(false, "impl is not finished yet")

        # load LDS
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        offsets_kd = gl.arange(0, XBLOCK, layout=layout)
        offsets_kh = gl.arange(0, XBLOCK, layout=layout)
        offsets_kw = gl.arange(0, XBLOCK, layout=layout)
        mask = offsets < xnumel

        smem_input.slice(start, length, dim=0)

        cp.async_copy_global_to_shared(smem_input, input_ptr + offsets, mask=mask)
        cp.commit_group()
        cp.wait_group(0)

        return
        """

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
        """

    __all__.append("conv_depthwise_3d_gl")

except Exception as e:
    print(f"failed to add gluon version due to : ", e)
