
import torch
import pyhip
from pyhip.contrib.w8a8_block_fp8_linear import *
import aiter
from aiter import dtypes, get_hip_quant
from aiter.ops.shuffle import shuffle_weight

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)

def test(m, n, k, b_preshuffle = False):
    block_size = (128, 128)
    block_shape_n, block_shape_k = block_size
    output_dtype = dtypes.bf16
    scale_m = m
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    input = torch.rand((m, k), dtype=output_dtype, device="cuda") /10.0
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    w = (weight_scale.view(scale_n, 1, scale_k, 1) * \
         weight.to(dtypes.fp32).view(scale_n, block_shape_n, scale_k, block_shape_k)).view(n, k).to(output_dtype)

    if b_preshuffle:
        weight = shuffle_weight(weight, layout=(16, 16))

    rw_bytes = weight.numel() * weight.itemsize + \
               weight_scale.numel() * weight_scale.itemsize + \
               input.numel() * input.itemsize

    if 1:
        ref = input @ w.t()
    else:
        aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)
        q_input, x_scale = aiter_per1x128_quant(input, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)
        ref = q_input.to(output_dtype) @ w.t()

    w8a8_block_fp8_linear(input, weight, block_size, weight_scale, input_scale = None, bias = None, method = "jit")


    ret_aiter, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            method = "aiter",
            b_preshuffle = b_preshuffle,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"aiter {m},{n},{k}")

    ret_jit, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            method = "jit",
            b_preshuffle = b_preshuffle,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"  jit {m},{n},{k}")

    ret_gluon, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            method = "gluon",
            b_preshuffle = b_preshuffle,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f"gluon {m},{n},{k}")

    ret_auto, dt = pyhip.run_perftest(
            w8a8_block_fp8_linear,
            input,
            weight,
            block_size,
            weight_scale,
            input_scale = None,
            bias = None,
            method = "auto",
            b_preshuffle = b_preshuffle,
            num_flops=m*n*k*2, num_bytes=rw_bytes, num_spec_tag=f" auto {m},{n},{k}")

    print(f"{pyhip.calc_diff(ref, ret_aiter)=:.6f}")
    print(f"{pyhip.calc_diff(ref, ret_jit, diff_thr=0.01)=:.6f}")
    print(f"{pyhip.calc_diff(ref, ret_gluon, diff_thr=0.01)=:.6f}")
    print(f"{pyhip.calc_diff(ref, ret_auto, diff_thr=0.01)=:.6f}")

if __name__ == "__main__":
    """
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 128]) torch.float8_e4m3fn torch.Size([4096, 128])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([2560, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 1024]) torch.float8_e4m3fn torch.Size([4096, 1024])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([256, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1536, 4096])
    gemm_a8w8_blockscale:  torch.float8_e4m3fn torch.Size([16384, 4096]) torch.float8_e4m3fn torch.Size([1024, 4096])
    """

    # # special shape accuracy test
    # test(32, 4096, 256, b_preshuffle=True)
    # test(16, 16384, 4096*4, b_preshuffle=True)
    # test(255, 512, 256, b_preshuffle=False)
    # if 1:
    for m in [16,32,64,128,256]:
            test(m, 64*256, 4096, b_preshuffle=True)
    if 0:
        M = 16384
        M = 57
        test(M, 4096, 1024, b_preshuffle=True); assert 0

        test(M, 256, 4096)
        test(M, 1024, 4096)
        test(M, 1536, 4096)
        test(M, 2560, 4096)
        test(M, 4096, 1024)

