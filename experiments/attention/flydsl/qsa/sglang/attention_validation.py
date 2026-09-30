"""Independent selected-token reference for opt-in service validation, not serving.

SGLang's legacy reference rounds Q * scale * log2(e) to BF16 before QK.
That changes the logits: QSA-T13's legacy result itself fails the original
.02/.02 tolerance against FP64. Keep Q/K unchanged here, scale FP32 QK,
and retain FP32 probabilities and PV. No QSA planner or attention helper is
used, and no runtime output is ever replaced by this reference.
"""

from itertools import accumulate

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_attention_fp32(
    Q, K, V, I, O, CQ, CK, LENS,
    H: tl.constexpr, HK: tl.constexpr, SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    request, hkv = tl.program_id(1) // HK, tl.program_id(1) % HK
    q0, q1 = tl.load(CQ + request), tl.load(CQ + request + 1)
    if row >= q1 - q0:
        return
    k0, length = tl.load(CK + request), tl.load(LENS + request)
    visible = length - (q1 - q0) + row + 1
    heads, dims, tokens = tl.arange(0, 16), tl.arange(0, 256), tl.arange(0, BN)
    query = tl.load(Q + ((q0 + row) * H + hkv * (H // HK) + heads[:, None]) * 256
                    + dims[None, :], mask=(heads < H // HK)[:, None], other=0.)
    maximum = tl.full((16,), -float('inf'), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 256), tl.float32)
    for start in range(0, tl.minimum(2051, visible), BN):
        selected = tl.load(I + (q0 + row) * 2051 + start + tokens,
                           mask=start + tokens < 2051, other=-1)
        valid = (selected >= 0) & (selected < visible)
        keys = tl.load(K + ((k0 + selected[None, :]) * HK + hkv) * 256 + dims[:, None],
                       mask=valid[None, :], other=0.)
        values = tl.load(V + ((k0 + selected[:, None]) * HK + hkv) * 256 + dims[None, :],
                         mask=valid[:, None], other=0.).to(tl.float32)
        scores = tl.dot(query, keys) * (SCALE * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        # Empty selected tiles (padding) must not produce -inf - -inf NaNs.
        next_max = tl.maximum(next_max, -1.0e30)
        alpha = tl.exp2(maximum - next_max)
        probabilities = tl.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities, values, accumulator * alpha[:, None], input_precision='ieee')
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        maximum = next_max
    tl.store(O + ((q0 + row) * H + hkv * (H // HK) + heads[:, None]) * 256 + dims[None, :],
             accumulator / denominator[:, None], mask=(heads < H // HK)[:, None])


def reference(q, k, v, indices, *, query_lens, prefix_lens, scale):
    """FP32 output for every query/head/channel, using only original inputs."""
    lengths = tuple(n + p for n, p in zip(query_lens, prefix_lens))
    if (len(query_lens) != len(prefix_lens) or sum(query_lens) != q.shape[0]
            or sum(lengths) != k.shape[0] or k.shape != v.shape
            or q.shape[1] % k.shape[1] or not 0 < q.shape[1] // k.shape[1] <= 16):
        raise ValueError('Invalid packed QSA validation layout')
    kw = dict(device=q.device, dtype=torch.int32)
    cq = torch.tensor(tuple(accumulate(query_lens, initial=0)), **kw)
    ck = torch.tensor(tuple(accumulate(lengths, initial=0)), **kw)
    lens = torch.tensor(lengths, **kw)
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    if q.shape[0]:
        _selected_attention_fp32[(max(query_lens), len(query_lens) * k.shape[1])](
            q, k, v, indices, output, cq, ck, lens, q.shape[1], k.shape[1], scale, 32,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    return output