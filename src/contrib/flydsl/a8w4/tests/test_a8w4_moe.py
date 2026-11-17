# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of the a8w4 package (gfx942), besides every config vs the reference
(``test_a8w4_ops.py``, generated from the ops):

    pytest a8w4/tests/test_a8w4_moe.py -q          # from the directory holding a8w4/

* the references: ``decode_fd2`` / ``moe_ref`` / ``moe_dyn_ref`` on the prepared
  weights vs ``ref_moe`` on the mxfp4 tensors, the packed weight layout;
* ``a8w4_moe`` with the tuned table, including token counts between tuned points, on
  the MiMo-V2.6-Flash TP8 and EP8 per-rank shapes (EP: rank 3, global expert ids);
* ``tp_prep`` (the TP kinds' fused prep) vs aiter moe_sorting + quantize_rows;
* the table's act key (SiTUv2 rows, SiLU fallback, tables without the act column),
  SiTUv2 dispatch (the clamp is exercised, betas are checked);
* CUDA-graph capture: only graph-safe configs are picked, replay matches eager;
  ``a8w4_moe_dyn``: one graph replayed with different device-side valid counts;
* ``MOE_DYN.pick``: EP kernels only.
"""

import csv
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from a8w4 import a8w4_moe, a8w4_moe_dyn, prepare_weights, shuffle_mxfp4  # noqa: E402
from a8w4 import ref as R  # noqa: E402
from a8w4.moe import MOE, MOE_DYN, shape_vocab  # noqa: E402
from a8w4.ref import COS_TOL  # noqa: E402
from a8w4.tune import FIELDS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="a8w4 kernels need gfx942 (fp8 FNUZ MFMA)")

# name: (H, I, E local, E global, topk, expert_offset)
SHAPES = {k: R.SHAPES[k] for k in ("tp8", "ep8")}
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


def test_decode_fd2():
    """decode_fd2 of the prepared weights == the mxfp4 values, except the groups >= 12
    exponents below their channel's max: 0 (the fd2 form drops them, as the kernels)."""
    E, H, I = 2, 512, 256
    _, raw = R.make_weights(E, H, I, seed=4)
    for k in ("w1s", "w2s"):  # rows 0-7 of expert 1: groups 11 (kept), 12, 13 below the max
        s = raw[k].view(torch.uint8)[1, :8]
        top = s.max(dim=-1, keepdim=True).values
        s[:, -1:] = top
        s[:, :3] = top - torch.tensor([13, 12, 11], dtype=torch.uint8, device="cuda")
    W = prepare_weights(*shuffle_mxfp4(raw["w1q"], raw["w1s"], raw["w2q"], raw["w2s"]))
    for e in range(E):
        for i, got in zip((1, 2), R.decode_fd2(W, e)):
            q, s = raw[f"w{i}q"][e], raw[f"w{i}s"][e].view(torch.uint8)
            want = R._mxfp4_to_f32(q, s).view(*s.shape, 32)
            drop = s.int() <= s.int().amax(-1, keepdim=True) - 12
            assert int(drop.sum()) == (16 if e else 0)
            want[drop] = 0
            assert torch.equal(got, want.view(got.shape)), (e, i)


@pytest.mark.parametrize("case", list(SHAPES))
def test_moe_ref_prepared(case):
    """moe_ref on the prepared weights == ref_moe on the mxfp4 tensors, bit for bit (the
    random weights have no dropped groups)."""
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, raw = weights(case)
    x, tw, ti = R.make_routing(77, H, EGl, K, seed=2)
    assert torch.equal(R.moe_ref(x, W, tw, ti, expert_offset=e0, num_experts=EGl),
                       R.ref_moe(x, tw, ti, raw, e0))
    out = R.moe_ref(x, W, tw, ti, expert_offset=e0, act="situv2", situ=R.SITU)
    assert torch.equal(out, R.ref_moe(x, tw, ti, raw, e0, "situv2", R.SITU))


def test_moe_dyn_ref():
    """moe_dyn_ref on a receive buffer == ref_moe of its valid rows (an fp8 x dequantized
    as given), zeros past them."""
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    raw = weights("ep8")[1]
    for fp8 in (False, True):
        inp = R.moe_dyn_test(*SHAPES["ep8"], capacity=512, num_tokens=37, fp8=fp8, seed=3)
        out = R.moe_dyn_ref(**inp)
        assert out.shape == (512, H) and not out[37:].any()
        x = inp["x"][:37]
        if fp8:
            x = (x.float().view(37, -1, 128) * inp["x_scale"][:37].unsqueeze(-1)).view(37, H)
        ref = R.ref_moe(x, inp["topk_weights"][:37], inp["topk_ids"][:37], raw, e0,
                        quant_x=not fp8)
        assert torch.equal(out[:37], ref)


@pytest.mark.parametrize("case", list(SHAPES))
@pytest.mark.parametrize("tokens", [1, 5, 100, 1000, 5000])
def test_tuned_dispatch(case, tokens):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, x, tw, ti, ref = data(case, tokens, seed=3)
    out = a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl)
    assert R.cos_diff(ref, out.float()) < COS_TOL


@pytest.mark.parametrize("tokens", [1, 300, 5000])
@pytest.mark.parametrize("bm", [32, 64])
def test_tp_prep_vs_aiter(tokens, bm):
    """same rows per expert block (any order inside), bit-identical x8 / xs, out zeroed."""
    from a8w4.kernels.row_quant import quantize_rows
    from a8w4.kernels.tp_prep import tp_prep
    from a8w4.moe import _sort

    H, I, E, EGl, K, _ = SHAPES["tp8"]
    x, tw, ti = R.make_routing(tokens, H, EGl, K, seed=11 + tokens)
    sa, wa, ea, nva, _ = _sort(ti, tw, (H, I, E, EGl, K), 0, bm)
    junk = torch.full((tokens, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    del junk  # the allocator hands this block to tp_prep's out
    s, w, e, nv, x8, xs, out = tp_prep(ti, tw, x, block_m=bm, num_experts=E)
    assert nv.tolist() == nva.tolist()
    n = int(nv[0])

    def rows(sids, sw, seids):
        key = (seids[:n // bm].long().repeat_interleave(bm) << 32) | sids[:n].long()
        order = torch.argsort(key)
        return key[order], sw[:n][order]

    (k, wk), (ka, wka) = rows(s, w, e), rows(sa, wa, ea)
    assert torch.equal(k, ka) and torch.equal(wk, wka)
    xa8, xas = quantize_rows(x)
    assert torch.equal(x8.view(torch.uint8), xa8.view(torch.uint8)) and torch.equal(xs, xas)
    assert int(torch.count_nonzero(out)) == 0


@pytest.mark.parametrize("case,tokens", [("tp8", 64), ("tp8", 4096), ("ep8", 16), ("ep8", 256)])
def test_cuda_graph(case, tokens):
    H, I, E, EGl, K, e0 = SHAPES[case]
    W, x, tw, ti, ref = data(case, tokens, seed=5)
    cfg = MOE.pick(shape_vocab((H, I, E, EGl, K), tokens, "silu"), capturing=True)
    assert cfg["kind"] != "tp1kq"
    # eager, the config the capture picks (a8w4_moe may pick another: tp8 4096 = tp1kq);
    # warm-up: compiles it, allocates the kernels' workspaces
    eager = MOE.run(cfg, x, W, tw, ti, expert_offset=e0, num_experts=EGl)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = a8w4_moe(x, W, tw, ti, expert_offset=e0, num_experts=EGl)
    for _ in range(3):  # replays reuse the workspaces / counters
        g.replay()
        torch.cuda.synchronize()
        assert R.cos_diff(ref, out.float()) < COS_TOL
    # the TP kernels' atomic adds: two eager runs differ by ~5e-6
    assert R.cos_diff(eager.float(), out.float()) < 1e-5


@pytest.mark.parametrize("case,tokens", [("tp8", 100), ("ep8", 5), ("ep8", 3000)])
def test_situv2_dispatch(case, tokens):
    """SiTUv2 calls pick from the table's situv2 rows (EP shape: an EP kernel)."""
    H, I, E, EGl, K, e0 = SHAPES[case]
    shape = (H, I, E, EGl, K)
    kind = MOE.pick(shape_vocab(shape, tokens, "situv2"))["kind"]
    assert kind.startswith("ep" if case == "ep8" else "tp"), kind
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


def test_table_act(tmp_path):
    """The act column keys the table; an act without rows of its own uses the SiLU rows,
    and so do tables written before the column existed."""
    shape = SHAPES["ep8"][:5]
    row = dict(zip(("model_dim", "inter_dim", "experts", "experts_global", "topk"), shape),
               routing="uniform", num_tokens=64, cos="0")
    rows = [dict(row, act=act, config=name, us=us) for act, name, us in (
        ("silu", "ep3k t16", 10), ("silu", "ep3k s32", 11),
        ("situv2", "ep3k s32", 10), ("situv2", "ep3k t16", 11))]

    def write(path, fields, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return str(path)

    def pick(op, act, path):
        return op.pick(shape_vocab(shape, 64, act, op=op), table=path)

    both = write(tmp_path / "both.csv", FIELDS, rows)
    for op in (MOE, MOE_DYN):
        assert pick(op, "silu", both)["cfg"] == "t16"
        assert pick(op, "situv2", both)["cfg"] == "s32"
    silu = rows[:2]
    old = write(tmp_path / "old.csv", [f for f in FIELDS if f != "act"], silu)
    for path in (write(tmp_path / "silu.csv", FIELDS, silu), old):
        assert pick(MOE, "situv2", path)["cfg"] == "t16"


def test_dyn_cuda_graph():
    """One capture serves any device-side valid count; the estimate (128) may be below
    it: the workgroups stride over the actual tiles."""
    def buffer(valid, seed):  # a receive buffer of 512 rows, fp8 x
        return R.moe_dyn_test(*SHAPES["ep8"], capacity=512, num_tokens=valid, fp8=True,
                              seed=seed)

    buf = buffer(512, seed=1)
    del buf["est_tokens"]  # the capture's estimate: 128
    buf["m_valid"].fill_(64)
    out = torch.empty_like(buf["x"], dtype=torch.bfloat16)
    run = lambda: a8w4_moe_dyn(**buf, est_tokens=128, out=out)  # noqa: E731
    run()  # warm-up: compiles, allocates the prep workspace
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for valid in (64, 1, 400, 0, 512):
        new = buffer(valid, seed=10 + valid)
        for k in ("x", "x_scale", "topk_weights", "topk_ids", "m_valid"):
            buf[k].copy_(new[k])
        g.replay()
        torch.cuda.synchronize()
        R.moe_dyn_check(out, R.moe_dyn_ref(**new), m_valid=new["m_valid"])


def test_dyn_pick():
    H, I, E, EGl, K, e0 = SHAPES["ep8"]
    for tokens in (1, 300, 3000, 40000):
        vocab = shape_vocab((H, I, E, EGl, K), tokens, "silu", op=MOE_DYN)
        cfg = MOE_DYN.pick(vocab)
        assert cfg["kind"] == "ep3k" and MOE_DYN.why_not(cfg, vocab) is None
    with pytest.raises(ValueError):  # TP8 shape: 256 local experts, no EP kernels
        MOE_DYN.pick(shape_vocab(SHAPES["tp8"][:5], 16, "silu", op=MOE_DYN))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"] + sys.argv[1:]))
