import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import argparse


import pyhip

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.manual_seed(0)

@pyhip.jit(with_debug_log=False)
def depthwise_conv3d_jit(J, KD, KH, KW, H, W,
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

    A = J.gpr(KD, KH, KW, "f32", 0)
    conv_tasks = []
    for kd in range(KD):
        for kh in range(KH):
            conv_tasks.append((kd, kh))

    h = J.gpr("su32", J.warp_id[0])
    with J.While(h < H):
        for w in range(0, W, 64):
            # process 1x64 outputs per warp
            vaddr = J.gpr("vu32", lds_input + \
                                    J.gpr("vu32", J.gpr("su32", h[0] * (padded_W * J.sizeof_bf16))) + \
                                    (w + J.lane_id[0]) * J.sizeof_bf16)
            C = J.gpr("f32", 0)
            for kd, kh in conv_tasks:
                for kw in range(KW):
                    J.ds_read_u16_d16_hi(A[kd, kh, kw], vaddr, mod=f"offset:{((kd*padded_H + kh)*padded_W + kw) * J.sizeof_bf16}")

            J.s_waitcnt(mod=f"lgkmcnt({0})")

            for ready_kd, ready_kh in conv_tasks:
                for kw in range(KW):
                    J.v_fmac_f32(C, A[ready_kd, ready_kh, kw], B[ready_kd, ready_kh, kw])

            C[0] += vbias

            vcol_addr = J.gpr("vu32", (w + J.lane_id[0]) * J.sizeof_bf16)
            with J.ExecMask(vcol_addr < W * J.sizeof_bf16):
                J.global_store_short_d16_hi(vcol_addr + J.gpr("su32", h[0] * (W*J.sizeof_bf16)), C, output)
        h[0] += num_warps


@pyhip.jit(with_debug_log=False)
def pointwise_conv3d_jit(J, group_size,
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
    blk_B = J.blockIdx.x[0]
    blk_g = J.blockIdx.y[0]

    offset = J.gpr("su32", (blk_B * C + blk_g * group_size) * DHW * J.sizeof_bf16)
    input[:] += offset[0]
    output[:] += offset[0]
    weight[:] += (blk_g * group_size) * group_size * J.sizeof_bf16
    bias[:] += (blk_g * group_size) * J.sizeof_bf16

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
        voffset[g] = J.threadIdx.x[0] * J.sizeof_DW4 + g * J.gpr("vu32", DHW) * J.sizeof_bf16

    dhw0 = J.gpr("su32", J.warp_id[0]*64*J.sizeof_DW4 // J.sizeof_bf16)
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

        dhw0[0] += num_threads * J.sizeof_DW4 // J.sizeof_bf16

    pass

def benchmark_op(op_func, op_name, iters, gflops, device):
    print(f"\n正在进行 {op_name} 预热/Tune...")
    
    # 统计预热/Tune 耗时
    os.putenv("PYTORCH_TUNABLEOP_TUNING", "1")
    warmup_start = time.time()
    for _ in range(10):
        _ = op_func()
    os.unsetenv("PYTORCH_TUNABLEOP_TUNING")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_end = time.time()
    warmup_time_ms = (warmup_end - warmup_start) * 1000
    print(f"预热/Tune 完成，耗时: {warmup_time_ms:.2f} ms")

    print(f"开始 {op_name} 性能测试 ({iters} 次迭代)...")
    start_time = time.time()
    
    for _ in range(iters):
        _ = op_func()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / iters * 1000
    tflops = (gflops / 1000.0) / (avg_time_ms / 1000.0) if avg_time_ms > 0 else 0
    
    return avg_time_ms, tflops, warmup_time_ms

def test_conv3d_benchmark(args):
    # 1. 准备数据
    if args.shape == "case1":
        # Case 1: [1, 64, 63, 45, 80] x [512, 64, 3, 3, 3]
        B, C_in, C_out, D, H, W = 1, 64, 512, 63, 45, 80
        kernel_size = (3, 3, 3)
        padding = (1, 1, 1)
        groups = 5
    elif args.shape == "case2":
        # Case 2: [1, 512, 61, 45, 80] x [2048, 512, 1, 1, 1]
        B, C_in, C_out, D, H, W = 1, 512, 2048, 61, 45, 80
        kernel_size = (1, 1, 1)
        padding = (0, 0, 0)
        groups = 1
    elif args.shape == "case3":
        # Case 3: [1, 512, 61, 45, 80] x [512, 1, 3, 5, 5], groups=512
        B, C_in, C_out, D, H, W = 1, 512, 512, 61, 45, 80
        # B, C_in, C_out, D, H, W = 1, 512, 512, 3, 5, 64
        #B, C_in, C_out, D, H, W = 1, 32, 32, 61, 45, 80
        kernel_size = (3, 5, 5)
        padding = (0, 2, 2)
        groups = C_out
    elif args.shape == "case4":
        # Case 4: [1, 2048, 61, 45, 80] x [2048, 512, 1, 1, 1], groups=4
        B, C_in, C_out, D, H, W = 1, 2048, 2048, 61, 45, 80
        kernel_size = (1, 1, 1)
        padding = (0, 0, 0)
        groups = 512
    else:
        raise ValueError(f"不支持的 shape 选项: {args.shape}")

    stride = (1, 1, 1)
    dilation = (1, 1, 1)

    input_dtype = torch.bfloat16
    #input_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- 正在初始化 Conv3d 数据 ({args.shape}) ---")
    print(f"Input Shape: [{B}, {C_in}, {D}, {H}, {W}]")
    print(f"Weight Shape: [{C_out}, {C_in // groups}, {kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}]")
    print(f"Groups: {groups}")
    print(f"Device: {device}, Dtype: {input_dtype}")
    
    # 初始化输入、权重和偏置
    input_tensor = torch.randn(B, C_in, D, H, W).to(dtype=input_dtype).to(device)
    weight_tensor = torch.randn(C_out, C_in // groups, *kernel_size).to(dtype=input_dtype).to(device)
    bias_tensor = torch.randn(C_out).to(dtype=input_dtype).to(device)

    # 内存格式处理
    # try:
    #     input_tensor = input_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     weight_tensor = weight_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     bias_tensor = bias_tensor.contiguous(memory_format=torch.channels_last_3d)
    #     print("使用 channels_last_3d 内存格式")
    # except:
    #     input_tensor = input_tensor.contiguous()
    #     print("使用默认内存格式 (NCHW)")
        
    # 计算输出尺寸
    D_out = (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
    print(f"输出尺寸: [{B}, {C_out}, {D_out}, {H_out}, {W_out}]")

    # 计算量 (GFLOPs)
    gflops = (2.0 * B * C_out * D_out * H_out * W_out * (C_in // groups) * kernel_size[0] * kernel_size[1] * kernel_size[2]) / 1e9
    print(f"理论计算量: {gflops:.4f} GFLOPs")


    print(weight_tensor.shape, weight_tensor.dtype)
    print(weight_tensor[0,0,...])
    print(weight_tensor[0,0,...].to(torch.float).sum().to(torch.bfloat16))

    # 定义测试操作
    def run_torch_conv3d():
        return F.conv3d(input_tensor, weight_tensor, 
                        bias=bias_tensor,
                        stride=stride, 
                        padding=padding, 
                        dilation=dilation,
                        groups=groups)

    def run_torch_conv1d():
        return F.conv1d(input_tensor.view(B, C_in, D*H*W), weight_tensor.view(C_out, C_in//groups, 1), 
                        bias=bias_tensor,
                        stride=stride[0], 
                        padding=padding[0], 
                        dilation=dilation[0],
                        groups=groups).view(B, C_out, D_out, H_out, W_out)

    ref = run_torch_conv3d()
    print(ref.shape, ref.dtype)

    def run_depthwise_conv3d_jit():
        output_tensor = torch.empty(B, C_out, D_out, H_out, W_out, dtype=input_dtype, device=device)
        depthwise_conv3d_jit(
                        [B,C_out,D_out],[256],
                         kernel_size[0], kernel_size[1], kernel_size[2], H_out, W_out,
                         input_tensor.data_ptr(),
                         weight_tensor.data_ptr(),
                         output_tensor.data_ptr(),
                         bias_tensor.data_ptr(),
                         B, C_out, D, D_out)
        return output_tensor

    def run_pointwise_conv3d_jit():
        output_tensor = torch.empty(B, C_out, D_out, H_out, W_out, dtype=input_dtype, device=device)
        group_size = C_in // groups
        pointwise_conv3d_jit(
                         [B, C_out//group_size],
                         [256], group_size,
                         input_tensor.data_ptr(),
                         weight_tensor.data_ptr(),
                         output_tensor.data_ptr(),
                         bias_tensor.data_ptr(),
                        B, C_out, D_out*H_out*W_out)
        return output_tensor

    # 2. 运行 Benchmark
    torch_ms, torch_tflops, torch_warmup_ms = benchmark_op(run_torch_conv3d, f"Standard PyTorch Conv3d ({args.shape})", args.iters, gflops, device)
    # 3. 汇总对比
    print(f"\n--- 性能对比汇总 ({args.shape}) ---")
    print(f"{'方法':<35} | {'平均耗时 (ms)':<15} | {'吞吐量 (TFLOPS)':<15} | {'预热/Tune (ms)':<15}")
    print("-" * 90)
    print(f"{'Standard PyTorch Conv3d':<35} | {torch_ms:>15.4f} | {torch_tflops:>15.2f} | {torch_warmup_ms:>15.2f}")

    run_conv = run_torch_conv3d
    if args.shape == "case3":
        run_conv = run_depthwise_conv3d_jit
    elif args.shape == "case4":
        run_conv = run_torch_conv1d
        run_conv = run_pointwise_conv3d_jit
    if run_conv != run_torch_conv3d:
        ret = run_conv()
        all_diff = pyhip.calc_diff(ref, ret)
        if all_diff > 0.001:
            for iib in range(B):
                for iic in range(C_out):
                    for iid in range(D_out):
                        iiref = ref[iib, iic, iid, ...]
                        iiret = ret[iib, iic, iid, ...]
                        passed = torch.allclose(iiref, iiret, atol=0.01, rtol=0.01)
                        diff = pyhip.calc_diff(iiref, iiret)
                        if not passed and diff > 0.001:
                            print(f"================ {B},{C_out},{D_out} : {iib}, {iic}, {iid}    {diff=}", )
                            print(ref.shape)
                            print(ret.shape)
                            print(iiref)
                            print(iiret)
                            print(iiret.view(-1)[:32].view(4,8))
                            assert 0

        opt_ms, opt_tflops, opt_warmup_ms = benchmark_op(run_conv, f"Standard PyTorch Conv3d ({args.shape})", args.iters, gflops, device)
        print(f"{'方法':<35} | {'平均耗时 (ms)':<15} | {'吞吐量 (TFLOPS)':<15} | {'预热/Tune (ms)':<15}")
        print(f"{'Standard PyTorch Conv3d':<35} | {torch_ms:>15.4f} | {torch_tflops:>15.2f} | {torch_warmup_ms:>15.2f}")
        print(f"{run_conv.__name__:<35} | {opt_ms:>15.4f} | {opt_tflops:>15.2f} | {opt_warmup_ms:>15.2f}")


    # 4. Profile (可选)
    if args.profile:
        print(f"\n开始 Torch Profile (Standard PyTorch Conv3d - {args.shape})...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/conv3d_{args.shape}_profile'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for _ in range(5):
                run_torch_conv3d()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        print(f"\nProfile 跟踪已保存到 ./log/conv3d_{args.shape}_profile")

if __name__ == "__main__":
    os.putenv("PYTORCH_TUNABLEOP_ENABLED", "1")

    parser = argparse.ArgumentParser(description="Conv3d Benchmark 脚本")
    parser.add_argument("--iters", type=int, default=10, help="迭代次数")
    parser.add_argument("--profile", action="store_true", help="是否启用 profile")
    parser.add_argument("--shape", type=str, default="case3", choices=["case1", "case2", "case3", "case4"], 
                        help="选择测试的 shape 选项: case1 ([1, 64, 63, 45, 80]), case2 ([1, 512, 61, 45, 80]), case3 ([1, 512, 61, 45, 80], groups=512), case4 ([1, 2048, 61, 45, 80], groups=4)")
    args = parser.parse_args()
    
    test_conv3d_benchmark(args)
