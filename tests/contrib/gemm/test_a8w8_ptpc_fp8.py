
import torch
import pyhip
from pyhip.contrib.w8a8_fp8_ptpc_linear import *
import aiter
from aiter import dtypes, get_hip_quant
from aiter.ops.shuffle import shuffle_weight

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

def test(m, n, k, b_preshuffle = False):
    dtype = torch.bfloat16
    input_as_fp8 = (torch.randn((m, k), dtype=dtype, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    weight_as_fp8 = (torch.rand((n, k), dtype=dtype, device="cuda") / 10.0).to(torch.float8_e4m3fn)
    qx, x_scale = aiter.pertoken_quant(input_as_fp8.to(torch.bfloat16), quant_dtype=aiter.dtypes.fp8, scale_dtype=dtypes.fp32)
    qwei, w_scale = aiter.pertoken_quant(weight_as_fp8.to(torch.bfloat16), quant_dtype=aiter.dtypes.fp8, scale_dtype=dtypes.fp32)

    rw_bytes = qwei.numel() * qwei.itemsize + \
               w_scale.numel() * w_scale.itemsize + \
               qx.numel() * qx.itemsize
    assert k % 128 == 0

    wei_fp32 = weight_as_fp8.t().to(torch.float32)
    input_fp32 = input_as_fp8.to(torch.float32)
    ref = (input_fp32 @ wei_fp32).to(torch.bfloat16)
    if 0:
        fake_x_scale_blkwise = torch.ones(x_scale.shape, device="cuda").to(torch.float32)
        fake_w_scale_blkwise = torch.ones(w_scale.shape, device="cuda").to(torch.float32)
        ret_jit, dt = pyhip.run_perftest(
            w8a8_ptpc_fp8_linear,
            input_as_fp8,
            weight_as_fp8,
            fake_x_scale_blkwise,
            fake_w_scale_blkwise,
            False,
            "jit",
            torch.bfloat16,
            None,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"jit {m},{n},{k}")
    else:
        ret_jit, dt = pyhip.run_perftest(
            w8a8_ptpc_fp8_linear,
            qx,
            qwei,
            x_scale,
            w_scale,
            False,
            "jit",
            torch.bfloat16,
            None,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"jit {m},{n},{k}")

    ret_aiter, dt = pyhip.run_perftest(
        w8a8_ptpc_fp8_linear,
        qx,
        qwei,
        x_scale,
        w_scale,
        False,
        "aiter",
        torch.bfloat16,
        None,
        num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"aiter {m},{n},{k}")

    ret_fly, dt = pyhip.run_perftest(
        w8a8_ptpc_fp8_linear,
        qx,
        qwei,
        x_scale,
        w_scale,
        False,
        "fly",
        torch.bfloat16,
        None,
        num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"fly {m},{n},{k}")
    
    print(f"{pyhip.calc_diff(ref, ret_jit, diff_thr=1000000)=:.6f}")
    print(f"{pyhip.calc_diff(ref, ret_aiter, diff_thr=0.01)=:.6f}")
    print(f"{pyhip.calc_diff(ref, ret_fly, diff_thr=0.01)=:.6f}")


if __name__ == "__main__":
    """
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 128]) torch.float8_e4m3fn torch.Size([4096, 128])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([2560, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 1024]) torch.float8_e4m3fn torch.Size([4096, 1024])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([256, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1536, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1024, 4096])
    """

    N = 4096
    K = 1536
    # token_list = [256]
    token_list = [6144, 12288, 24576, 49152, 73728, 98304, 196608, 294912,393216]
    # token_list = [6144, 12288]
    for M in token_list:
        print(f'--------------------------------------{M=}--------------------------------------------------')
        test(M, N, K, b_preshuffle = False)
    # special shape accuracy test



