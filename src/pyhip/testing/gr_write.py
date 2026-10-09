# SPDX-License-Identifier: MIT
"""Shared GR write inputs, references and accuracy checks for tests and benchmarks.

GPU/compiler imports stay lazy; this module imports neither test scripts nor an
inference framework.
"""
import math

C, H = 4, 2560
K = C * H
EPS = 1e-6
CHECK_ROWS = 1024
# One BF16 ULP is 2**-8 relative; FMA vs. reference rounding and the gate's FP32
# accumulation order may move a rounding boundary by one ULP.
OUT_TOLERANCE = dict(rtol=2 ** -7, atol=1e-5)
NORMED_TOLERANCE = dict(rtol=2 ** -6, atol=1e-4)
MIN_EXACT_FRACTION = 0.999
DEFAULT_ROWS = (1024, 2048, 4096, 8192, 12000, 16384, 32768)


def bytes_moved(rows):
    """HBM bytes of one call: read y, r, n; write out, normed (weights and gate values excluded)."""
    return rows * (H * 2 + 4 * K * 2)


def make_weights(seed=0, device='cuda'):
    """Random BF16 inject weight [4, 10240] (dot ~ N(0, 1)) and norm weight [10240]."""
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)
    inject = (torch.randn((C, K), generator=generator, device=device) / math.sqrt(K)).to(torch.bfloat16)
    norm = (torch.randn((K,), generator=generator, device=device) * 0.1).to(torch.bfloat16)
    return inject, norm


def make_inputs(rows, seed=0, device='cuda'):
    """Random BF16 block_output [T, 2560], residual and normed_residual [T, 10240]."""
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)
    y = torch.randn((rows, H), generator=generator, device=device).to(torch.bfloat16)
    r = torch.randn((rows, K), generator=generator, device=device).to(torch.bfloat16)
    n = torch.randn((rows, K), generator=generator, device=device).to(torch.bfloat16)
    return y, r, n


def reference(block_output, residual, normed_residual, inject_weight, norm_weight, eps=EPS, chunk=CHECK_ROWS):
    """Reference (out, normed) with the kernel's rounding points.

    dot uses FP64 and is rounded to FP32; out = bf16(fp32(r + a * y)) emulates one
    FP32 FMA; normed = bf16(fp32(fp32(out * rstd) * fp32(1 + w))) per branch.
    """
    import torch
    rows = residual.shape[0]
    out = torch.empty((rows, K), dtype=torch.bfloat16, device=residual.device)
    normed = torch.empty_like(out)
    gain = (1.0 + norm_weight.float()).view(C, H)
    weight = inject_weight.double()
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        dot = (normed_residual[begin:end].double() @ weight.T).float()
        gate = 2.0 / (1.0 + torch.exp(-dot / C))
        y = block_output[begin:end].double()[:, None, :]
        r = residual[begin:end].double().view(end - begin, C, H)
        value = (r + gate.double()[:, :, None] * y).float().to(torch.bfloat16)
        of = value.float()
        rstd = torch.rsqrt((of.double().pow(2).sum(-1, keepdim=True) / H + eps)).float()
        out[begin:end] = value.view(end - begin, K)
        normed[begin:end] = ((of * rstd) * gain).to(torch.bfloat16).view(end - begin, K)
    return out, normed


def check_close(out, normed, out_ref, normed_ref, chunk=CHECK_ROWS):
    """Assert tolerance and exact-match fraction; return the exact fractions (out, normed)."""
    import torch
    rows = out.shape[0]
    exact = [0, 0]
    for begin in range(0, rows, chunk):
        end = min(rows, begin + chunk)
        for index, (actual, expected, tolerance) in enumerate((
                (out[begin:end], out_ref[begin:end], OUT_TOLERANCE),
                (normed[begin:end], normed_ref[begin:end], NORMED_TOLERANCE))):
            torch.testing.assert_close(actual.float(), expected.float(), **tolerance)
            exact[index] += (actual.view(torch.int16) == expected.view(torch.int16)).sum().item()
    total = max(1, rows * K)
    fractions = (exact[0] / total, exact[1] / total)
    if rows * K >= 4096:
        assert min(fractions) >= MIN_EXACT_FRACTION, f'exact-match fraction too low: {fractions}'
    return fractions
