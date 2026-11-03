"""Correctness tests for the three FlyDSL gfx942 FP8 block-scale GEMM kernels (8x1, 4x2-LDS, 2x4)."""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import pyhip
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

SHAPES = [
    (256, 256, 128),
    (257, 272, 256),
    (17, 528, 384),
    (129, 272, 512),
    (255, 528, 640),
    (257, 272, 2048),
    (1, 4096, 128),
    (129, 4096, 256),
    (257, 4096, 384),
    (4097, 4096, 512),
    (255, 4096, 640),
    (257, 4096, 2048),
]


def _kernel(name):
    if name == "8x1":
        from pyhip.contrib.flydsl.gemm_a8w8_blockscale_gfx942_8x1 import gemm_a8w8_blockscale_gfx942_8x1

        return gemm_a8w8_blockscale_gfx942_8x1
    if name == "4x2-lds":
        from pyhip.contrib.flydsl.gemm_a8w8_blockscale_gfx942_4x2_lds import gemm_a8w8_blockscale_gfx942_4x2_lds

        return gemm_a8w8_blockscale_gfx942_4x2_lds
    from pyhip.contrib.flydsl.gemm_a8w8_blockscale_gfx942_2x4_lds import gemm_a8w8_blockscale_gfx942_2x4_lds

    return gemm_a8w8_blockscale_gfx942_2x4_lds


def run_torch(x, weight, x_scale, w_scale):
    m, k = x.shape
    n = weight.shape[0]
    x = (x.to(torch.float32).view(m, k // 128, 128) * x_scale.unsqueeze(-1)).view(m, k)
    w_scale = rearrange(
        w_scale.view(-1, 1).repeat(1, 128 * 128).view(w_scale.shape[0], k // 128, 128, 128),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )[:n, :k]
    return F.linear(x, weight.to(torch.float32) * w_scale).to(torch.bfloat16)


@pytest.mark.parametrize("kernel", ["8x1", "4x2-lds", "2x4"])
@pytest.mark.parametrize("preshuffle_b", [False, True])
@pytest.mark.parametrize("rows,cols,depth", SHAPES)
def test_gfx942_flydsl_blockscale(rows, cols, depth, preshuffle_b, kernel):
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
        pytest.skip("gfx942 kernels")
    gemm = _kernel(kernel)
    torch.manual_seed(0)
    x = (torch.randn((rows, depth), device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.randn((cols, depth), device="cuda") / 10).to(dtypes.fp8)
    scale_a = torch.rand((rows, depth // 128), device="cuda")
    scale_b = torch.rand((pyhip.div_up(cols, 128), depth // 128), device="cuda")
    reference = run_torch(x, weight, scale_a, scale_b)
    weight = shuffle_weight(weight, layout=(16, 16)) if preshuffle_b else weight
    scale_a = scale_a.T.contiguous()
    storage = torch.full((rows * cols + 128,), 123, dtype=torch.bfloat16, device="cuda")
    output = storage[64:-64].view(rows, cols)
    for _ in range(3):
        output.fill_(float("nan"))
        gemm(x, weight, output, scale_a, scale_b, preshuffle_b)
        assert checkAllclose(reference, output, msg=kernel, tol_err_ratio=0) == 0
        assert pyhip.calc_diff(reference, output) < 1e-5
        assert torch.all(storage[:64] == 123) and torch.all(storage[-64:] == 123)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, weight, output, scale_a, scale_b, preshuffle_b)
    output.fill_(float("nan"))
    graph.replay()
    assert checkAllclose(reference, output, msg=f"{kernel} graph", tol_err_ratio=0) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
