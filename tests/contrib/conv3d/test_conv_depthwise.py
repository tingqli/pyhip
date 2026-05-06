import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import argparse

import pyhip

from pyhip.contrib.conv_depthwise import *
from pyhip.contrib.conv_pointwise import *

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.manual_seed(0)

def benchmark_op(op_func, op_name, iters, gflops, device, mem_bytes=None):
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

    if mem_bytes is not None and avg_time_ms > 0:
        gbps = (mem_bytes / 1e9) / (avg_time_ms / 1000.0)
    else:
        gbps = float("nan")

    return avg_time_ms, tflops, warmup_time_ms, gbps

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

    input_dtype = torch.bfloat16 # or torch.float16
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

    # 读 input、weight、bias，写 output
    mem_bytes = (
        input_tensor.numel() + weight_tensor.numel() + B * C_out * D_out * H_out * W_out + bias_tensor.numel()
    ) * input_tensor.element_size()

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

    def run_conv_depthwise_3d_pyhip():
        return conv_depthwise_3d(input_tensor, weight_tensor, bias_tensor,
                                 stride, padding, dilation, groups=groups,
                                 method="hip")

    def run_pointwise_conv_jit():
        return conv_pointwise(input_tensor, weight_tensor, bias_tensor, groups=groups,
                              use_gluon=False)

    # 2. 运行 Benchmark
    torch_ms, torch_tflops, torch_warmup_ms, torch_gbps = benchmark_op(
        run_torch_conv3d, f"Standard PyTorch Conv3d ({args.shape})", args.iters, gflops, device, mem_bytes
    )
    # 3. 汇总对比
    print(f"\n--- 性能对比汇总 ({args.shape}) ---")
    hdr = f"{'方法':<35} | {'平均耗时 (ms)':<15} | {'吞吐量 (TFLOPS)':<15} | {'带宽 (GB/s)':<15} | {'预热/Tune (ms)':<15}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'Standard PyTorch Conv3d':<35} | {torch_ms:>15.4f} | {torch_tflops:>15.2f} | {torch_gbps:>15.2f} | {torch_warmup_ms:>15.2f}")

    run_conv = run_torch_conv3d
    if args.shape == "case3":
        run_conv = run_conv_depthwise_3d_pyhip
    elif args.shape == "case4":
        run_conv = run_torch_conv1d
        run_conv = run_pointwise_conv_jit
    if run_conv != run_torch_conv3d:
        ret = run_conv()
        all_diff = pyhip.calc_diff(ref, ret)
        if all_diff > 0.001:
            def check():
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
            check()

        opt_ms, opt_tflops, opt_warmup_ms, opt_gbps = benchmark_op(
            run_conv, f"Standard PyTorch Conv3d ({args.shape})", args.iters, gflops, device, mem_bytes
        )
        print(hdr)
        print(f"{'Standard PyTorch Conv3d':<35} | {torch_ms:>15.4f} | {torch_tflops:>15.2f} | {torch_gbps:>15.2f} | {torch_warmup_ms:>15.2f}")
        print(
            f"{run_conv.__name__:<35} | {opt_ms:>15.4f} | {opt_tflops:>15.2f} | {opt_gbps:>15.2f} | {opt_warmup_ms:>15.2f} | max_diff={all_diff:.6f}"
        )


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
