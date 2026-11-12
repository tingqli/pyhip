# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of the a8w4 package (gfx942):

    pytest a8w4/tests/test_a8w4_moe.py -q          # from the directory holding a8w4/

* every kernel family vs the fp8-emulating fp32 reference, on the MiMo-V2.6-Flash
  TP8 and EP8 per-rank shapes (EP: rank 3, global expert ids);
* ``a8w4_moe`` with the tuned table, including token counts between tuned points;
* CUDA-graph capture: only graph-safe configs are picked, replay matches eager;
* SiTUv2 on every kernel family (TP and EP, incl. ``a8w4_moe_dyn``);
* ``a8w4_moe_dyn`` on an all-to-all receive buffer (device-side valid count, stale
  rows past it, bf16 or blockwise-fp8 x), incl. one CUDA graph replayed with
  different counts.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import (  # noqa: E402
    a8w4_moe,
    a8w4_moe_dyn,
    dyn_config,
    lookup_config,
    parse_config,
    run_config,
    supported,
)
from a8w4 import ref as R  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="a8w4 kernels need gfx942 (fp8 FNUZ MFMA)")

COS_TOL = 5e-3
# name: (H, I, E local, E global, topk, expert_offset)
SHAPES = {"tp8": (4096, 256, 256, 256, 8, 0), "ep8": (4096, 2048, 32, 256, 8, 96)}
CONFIGS = {
    "tp8": ["tp2k bm32 g1bcm1 g2base", "tp2k bm64 g1bcm2 g2nt4", "tp1k bcm2 ks1", "tp1kq bcm1"],
    "ep8": ["tp2k bm32 g1bcm1 g2base", "tp1k bcm1 ks4", "tp1k bcm2 ks8", "ep3k t16", "ep3k s16",
            "ep3k s32", "ep3k m64", "ep3k l128", "ep1k m512"],
}
_W = {}


def weights(case):
    if case not in _W:
        H, I, E, *_ = SHAPES[case]
        _W[case] = R.make_weights(E, H, I)
    return _W[case]


def data(case, tokens, act="silu", situ=None, seed=0):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, raw = weights(case)
    x, tw, ti = R.make_routing(tokens, H, EGl, K, seed=seed + tokens)
    return W, x, tw, ti, R.ref_moe(x, tw, ti, raw, e0, act, situ)


def test_weights_layout():
    W, _ = weights("ep8")
    n = W["w2_scale"].numel() - W["w2_exp"].numel() * 4
    assert W["w2_exp"].data_ptr() == W["w2_scale"][n:].data_ptr()
    assert W["w1_exp"].shape == W["w1"].shape[:2] and W["w2_exp"].shape == W["w2"].shape[:2]


@pytest.mark.parametrize("case,name", [(c, n) for c in SHAPES for n in CONFIGS[c]])
@pytest.mark.parametrize("tokens", [1, 77, 600])
def test_config_vs_ref(case, name, tokens):
    H, I, E, EGl, K, e0 = SHAPES[case]
    cfg = parse_config(name)
    if not supported(cfg, (H, I, E, EGl, K), tokens=tokens):
        pytest.skip("not supported at this token count")
    W, x, tw, ti, ref = data(case, tokens)
    out = run_config(cfg, x, W, tw, ti, expert_offset=e0, num_experts=EGl)
    assert out.shape == (tokens, H) and out.dtype == torch.bfloat16
    assert R.cos_diff(ref, out.float()) < COS_TOL


@pytest.mark.parametrize("case", list(SHAPES))
@pytest.mark.parametrize("tokens", [1, 5, 100, 1000, 5000])
def test_tuned_dispatch(case, tokens):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, x, tw, ti, ref = data(case, tokens, seed=3)
    out = a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl)
    assert R.cos_diff(ref, out.float()) < COS_TOL


@pytest.mark.parametrize("case,tokens", [("tp8", 64), ("tp8", 4096), ("ep8", 16), ("ep8", 256)])
def test_cuda_graph(case, tokens):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, x, tw, ti, ref = data(case, tokens, seed=5)
    cfg = lookup_config((H, I, E, EGl, K), tokens, capturing=True)
    assert cfg["kind"] != "tp1kq"
    fn = lambda: a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl)  # noqa: E731
    eager = fn()  # warm-up: compiles, allocates the kernels' workspaces
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    for _ in range(3):  # replays reuse the workspaces / counters
        g.replay()
        torch.cuda.synchronize()
        assert R.cos_diff(ref, out.float()) < COS_TOL
    assert R.cos_diff(eager.float(), out.float()) < 1e-5


@pytest.mark.parametrize("case,name", [(c, n) for c in SHAPES for n in CONFIGS[c]])
def test_situv2_config(case, name):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, x, tw, ti, ref = data(case, 300, act="situv2", situ=R.SITU)
    out = run_config(parse_config(name), x, W, tw, ti, expert_offset=e0, num_experts=EGl,
                     act="situv2", situ=R.SITU)
    assert R.cos_diff(ref, out.float()) < COS_TOL


@pytest.mark.parametrize("case,tokens", [("tp8", 100), ("ep8", 5), ("ep8", 3000)])
def test_situv2_dispatch(case, tokens):
    """The table picks the same kernels for SiTUv2 as for SiLU (EP shape: EP family)."""
    H, I, E, EGl, K, e0 = SHAPES[case]
    shape = (H, I, E, EGl, K)
    assert lookup_config(shape, tokens, act="situv2") == lookup_config(shape, tokens)
    W, x, tw, ti, ref = data(case, tokens, act="situv2", situ=R.SITU, seed=7)
    raw = weights(case)[1]
    no_clamp = dict(R.SITU, swiglu_limit=float("inf"))
    ref_no_clamp = R.ref_moe(x, tw, ti, raw, e0, "situv2", no_clamp)
    assert R.cos_diff(ref, ref_no_clamp) > 10 * COS_TOL  # the clamp is exercised
    out = a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl, act="situv2",
                   situ=R.SITU)
    assert R.cos_diff(ref, out.float()) < COS_TOL
    with pytest.raises(ValueError):  # betas must be > 0
        a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl, act="situv2",
                 situ=dict(R.SITU, situ_beta=0.0))


def recv_buffer(capacity, valid, seed, fp8):
    """An EP all-to-all receive buffer of the ep8 case: rows [0, valid) are tokens with
    >= 1 local route (global ids); the rows past ``valid`` are stale (NaN x, random
    ids) and must be ignored. -> (x, x_scale, topk_weights, topk_ids, reference x)."""
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    x, tw, ti = R.make_routing(capacity, H, EGl, K, seed=seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    forced = torch.randint(e0, e0 + E, (capacity,), device="cuda", generator=g)
    has_local = ((ti >= e0) & (ti < e0 + E)).any(-1)
    ti[:, 0] = torch.where(has_local, ti[:, 0], forced.to(torch.int32))
    ti[valid:] = torch.randint(0, EGl, ti[valid:].shape, device="cuda", generator=g,
                               dtype=torch.int32)
    x[valid:] = float("nan")
    if not fp8:
        return x, None, tw, ti, x
    xq, xs = R.quant_fp8_groups(x)
    x_ref = (xq.float().view(capacity, -1, 128) * xs.unsqueeze(-1)).view(capacity, H)
    return xq, xs, tw, ti, x_ref


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("valid", [0, 1, 37, 300])
def test_dyn_vs_ref(fp8, valid):
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    W, raw = weights("ep8")
    capacity = 512
    x, xs, tw, ti, x_ref = recv_buffer(capacity, valid, seed=valid, fp8=fp8)
    m_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
    out = a8w4_moe_dyn(x, W, tw, ti, m_valid, x_scale=xs, expert_offset=e0, num_experts=EGl,
                       est_tokens=max(valid, 1))
    assert out.shape == (capacity, H) and out.dtype == torch.bfloat16
    if valid:
        ref = R.ref_moe(x_ref[:valid], tw[:valid], ti[:valid], raw, e0, quant_x=not fp8)
        assert R.cos_diff(ref, out[:valid].float()) < COS_TOL


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("valid,capacity", [(37, 512), (1500, 2048)])  # s16 / m64 (TR)
def test_dyn_situv2(fp8, valid, capacity):
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    W, raw = weights("ep8")
    x, xs, tw, ti, x_ref = recv_buffer(capacity, valid, seed=valid, fp8=fp8)
    m_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
    out = a8w4_moe_dyn(x, W, tw, ti, m_valid, x_scale=xs, expert_offset=e0, num_experts=EGl,
                       act="situv2", situ=R.SITU, est_tokens=valid)
    ref = R.ref_moe(x_ref[:valid], tw[:valid], ti[:valid], raw, e0, "situv2", R.SITU,
                    quant_x=not fp8)
    assert R.cos_diff(ref, out[:valid].float()) < COS_TOL


def test_dyn_cuda_graph():
    """One capture serves any device-side valid count; the estimate (128) may be below
    it: the workgroups stride over the actual tiles."""
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    W, raw = weights("ep8")
    capacity = 512
    x, xs, tw, ti, _ = recv_buffer(capacity, capacity, seed=1, fp8=True)
    m_valid = torch.tensor([64], dtype=torch.int32, device="cuda")
    out = torch.empty((capacity, H), dtype=torch.bfloat16, device="cuda")
    run = lambda: a8w4_moe_dyn(x, W, tw, ti, m_valid, x_scale=xs, expert_offset=e0,  # noqa: E731
                               num_experts=EGl, est_tokens=128, out=out)
    run()  # warm-up: compiles, allocates the prep workspace
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for valid in (64, 1, 400, 0, 512):
        x2, xs2, tw2, ti2, x_ref = recv_buffer(capacity, valid, seed=10 + valid, fp8=True)
        for dst, src in ((x, x2), (xs, xs2), (tw, tw2), (ti, ti2)):
            dst.copy_(src)
        m_valid.fill_(valid)
        g.replay()
        torch.cuda.synchronize()
        if valid:
            ref = R.ref_moe(x_ref[:valid], tw2[:valid], ti2[:valid], raw, e0, quant_x=False)
            assert R.cos_diff(ref, out[:valid].float()) < COS_TOL, valid


def test_dyn_config():
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    for tokens in (1, 300, 3000, 40000):
        cfg = dyn_config((H, I, E, EGl, K), tokens)
        assert cfg["kind"] == "ep3k" and supported(cfg, (H, I, E, EGl, K))
    with pytest.raises(ValueError):  # TP8 shape: 256 local experts, no EP kernels
        dyn_config(SHAPES["tp8"][:5], 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))
