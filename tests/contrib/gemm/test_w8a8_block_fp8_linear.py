
import torch
import pyhip
from pyhip.contrib.w8a8_block_fp8_linear import *
from aiter import dtypes

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

def test(m, n, k):
    block_size = (128, 128)
    block_shape_n, block_shape_k = block_size
    output_dtype = dtypes.bf16
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    input = torch.rand((m, k), dtype=output_dtype, device="cuda") / 10
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    w = (weight_scale.view(scale_n, 1, scale_k, 1) * \
         weight.to(dtypes.fp32).view(scale_n, block_shape_n, scale_k, block_shape_k)).view(n, k).to(output_dtype)

    rw_bytes = weight.numel() * weight.itemsize + \
               weight_scale.numel() * weight_scale.itemsize + \
               input.numel() * input.itemsize

    ref = input @ w.t()
    ret, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            use_aiter = True,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"aiter {m},{n},{k}")
    print(f"{pyhip.calc_diff(ref, ret, diff_thr=0.01)=:.6f}")

    ret, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            use_aiter = False,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"pyhip {m},{n},{k}")
    print(f"{pyhip.calc_diff(ref, ret, diff_thr=0.01)=:.6f}")

if __name__ == "__main__":
    """
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 128]) torch.float8_e4m3fn torch.Size([4096, 128])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([2560, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 1024]) torch.float8_e4m3fn torch.Size([4096, 1024])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([256, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1536, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1024, 4096])
    """
    M = 16384
    test(M, 256, 4096)
    test(M, 1024, 4096)
    test(M, 1536, 4096)
    test(M, 2560, 4096)
    test(M, 4096, 1024)

    test(32, 1024, 4096)
