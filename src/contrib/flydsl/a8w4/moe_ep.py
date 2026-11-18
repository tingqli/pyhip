# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Op ``moe_ep`` (``kernelkit``, registry ``_ops.ep_ops``): the expert-parallel MoE layer,
one call on every rank: send this rank's tokens to the ranks of their experts, run the
experts, add the routes back into this rank's output.

    comm = EpComm(rank=rank, world=world, max_tokens=4096)           # once, on every rank
    out = MOE_EP(x, W, topk_weights, topk_ids, comm=comm, tokens=T)   # on every rank

The interface is one rank's call: its tokens ``x`` bf16 [n, H] and ``topk_*`` (global
expert ids), its experts ``W`` (``prepare_weights``; rank r holds global experts
[r E, r E + E)), ``comm`` (``EpComm``: the rank's communication buffers), ``tokens`` (the
batch over all ranks, DP metadata: the same on every rank), ``act`` / ``situ``. The
problem: ``tokens`` split evenly over ``world`` ranks, ``experts_global`` experts, the
shape, routing and act.

Solvers, all ``@multi_gpu("world")`` (``python -m kernelkit tune / verify / ab / check
a8w4.tune:EP_TUNING ...`` start ``world`` ranks; ``mori_init``: MORI shmem):

  mega        ``megamoe.MegaMoeEpA8W4``: dispatch, gemm1 / gemm2 and combine in one
              persistent kernel; knobs ``producers``, ``lag``, ``bcm``; EP8 x 32 local
              experts only (other problems: n/a)
  mega_small  ``megamoe.MegaMoeSmallEpA8W4`` (after aiter's MegaMoEV2 small-input
              kernel): x quant, dispatch into fixed per-expert slots, gemm1 / gemm2 and
              the combine as ONE kernel launch; knob ``bm`` (m-tile rows); ``tokens`` <=
              ``EpComm.small_tmax`` (1024), EP8 x 32 local experts only
  mori        sglang's MORI path (``tests/test_a8w4_mori_ep8.py``): fp8 quant per (token,
              128), MORI dispatch, ``a8w4_moe_dyn`` ep3k (knob ``cfg``) on the whole
              receive buffer (the received count stays on the device, row estimate
              ``tokens``: sglang's CUDA-graph call), MORI combine
  mori_eager  the same with sglang's eager call: the received count read on the host, ep3k
              on those rows (``@no_capture``)

The other a8w4 kernels (tp1k, tp2k, ep1k) take bf16 rows with a host-known count (MORI
would send bf16, twice the bytes); on EP8 rows ep3k is faster than all of them at every
batch of ``configs/a8w4_moe_tuned.csv``.
"""

import functools

import torch
import torch.distributed as dist
from kernelkit import dim, knob, meta, multi_gpu, no_capture

from . import ref as R
from ._ops import ep_ops, ep_register
from .kernels import ep_mega as K
from .kernels._act import situ_args
from .kernels._protocol import QUEUES
from .moe import EP_CFGS, MOE_DYN, _ep_cached, ep_preset
from .weights import weight_shape

FP8 = torch.float8_e4m3fnuz
CAP = 4096  # tokens per rank the problems' EpComm holds at least
BCM_NT, BCM_BIG = (2, 2), (0, 1)  # (gemm1, gemm2) W cache modifiers MegaMoeEpA8W4 picks from


def mori_init():
    """A rank's MORI shmem over the default process group (the symmetric heap of the
    MORI op and of MegaMoeEpA8W4): the ``@multi_gpu`` init of the solvers."""
    import mori

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")


class EpComm:
    """One rank's communication state for ``moe_ep``: its ``rank`` of ``world``, the tokens
    per rank a call may bring (``max_tokens``), the most tokens over all ranks a
    ``mega_small`` call may bring (``small_tmax``) and each solver family's symmetric
    buffers, made by the first call that needs them (a collective: every rank makes the
    same calls in the same order)."""

    def __init__(self, *, rank, world, max_tokens, small_tmax=1024):
        self.rank, self.world, self.max_tokens = rank, world, max_tokens
        self.small_tmax = small_tmax
        self._mega = self._mori = self._small = None

    def mega(self, W, topk):
        """The ``MegaMoeEpA8W4`` (built for W's shape; the calls pass their own W)."""
        if self._mega is None:
            from .megamoe import MegaMoeEpA8W4

            self._mega = MegaMoeEpA8W4(W, rank=self.rank, max_tokens=self.max_tokens,
                                       topk=topk, world=self.world)
        return self._mega

    def small(self, W, topk):
        """The ``MegaMoeSmallEpA8W4`` (built for W's shape; the calls pass their own W)."""
        if self._small is None:
            from .megamoe import MegaMoeSmallEpA8W4

            self._small = MegaMoeSmallEpA8W4(W, rank=self.rank, tmax=self.small_tmax,
                                             topk=topk, world=self.world)
        return self._small

    def mori(self, W, topk):
        """(MORI op, fp8 quant per (token, 128)) as sglang builds them for one node with
        fp8 dispatch (``init_mori_op``)."""
        if self._mori is None:
            import mori
            from aiter import QuantType, get_hip_quant

            E, H, _ = weight_shape(W)
            op = mori.ops.EpDispatchCombineOp(mori.ops.EpDispatchCombineConfig(
                data_type=FP8, rank=self.rank, world_size=self.world, hidden_dim=H,
                scale_dim=H // 128, scale_type_size=4, max_token_type_size=2,  # bf16 model
                max_num_inp_token_per_rank=self.max_tokens, num_experts_per_rank=E,
                num_experts_per_token=topk, warp_num_per_block=16, block_num=80,
                kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
                gpu_per_node=self.world, rdma_block_num=0, num_qp_per_pe=2, quant_type="none"))
            self._mori = op, get_hip_quant(QuantType.per_1x128)
        return self._mori


def _gather_rows(t, ns):
    """Every rank's rows of ``t`` ([ns[rank], ...]), concatenated in rank order."""
    buf = t.new_zeros((max(max(ns), 1), *t.shape[1:]))
    buf[: t.shape[0]] = t
    parts = [torch.empty_like(buf) for _ in ns]
    dist.all_gather(parts, buf)
    return torch.cat([p[:n] for p, n in zip(parts, ns)])


@functools.lru_cache(maxsize=1)
def _weights(rank, experts, model_dim, inter_dim):
    return R.make_weights(experts, model_dim, inter_dim, seed=rank)[0]


@functools.lru_cache(maxsize=1)
def _comm(rank, world, cap):
    return EpComm(rank=rank, world=world, max_tokens=cap)


# ---- the problem ----------------------------------------------------------------------

@ep_register
def moe_ep_ref(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None):
    """fp32 reference of this rank's rows (collective: every rank runs it). x is quantized to
    fp8 per (row, 128 columns) as the solvers do; every rank evaluates its experts on all
    ranks' tokens, and the ranks' sums are added."""
    world, rank, n = dist.get_world_size(), dist.get_rank(), x.shape[0]
    xf = R.dequant_fp8(*R.quant_fp8_groups(x)) if n else x.float()
    ns = [None] * world
    dist.all_gather_object(ns, n)
    X, TW, TI = (_gather_rows(t, ns) for t in (xf, topk_weights, topk_ids))
    E = weight_shape(W)[0]
    out = R._moe_f32(X, TW, TI, functools.partial(R.decode_fd2, W), E, rank * E, act, situ,
                     quant_x=False)
    dist.all_reduce(out)
    off = sum(ns[:rank])
    return out[off:off + n]


@ep_register
@dim("act", R.ACTS, id="{}")
@dim("routing", R.ROUTINGS, id="{}")
def moe_ep_problem(tokens, model_dim=4096, inter_dim=2048, experts_global=256, world=8, topk=8,
                   routing="uniform", act="silu", *, seed=0):
    """This rank's inputs: a batch of ``tokens`` (``make_routing`` over the ``experts_global``
    experts, the same seed on every rank) split evenly over the ``world`` ranks as DP
    attention does, this rank's rows; its ``experts_global / world`` experts
    (``make_weights``, seed = rank) and its ``EpComm`` (>= ``CAP`` tokens per rank), both
    cached; SiTUv2 with ``R.SITU``."""
    rank = dist.get_rank()
    assert world == dist.get_world_size(), (world, dist.get_world_size())
    assert experts_global % world == 0, f"{experts_global} experts over {world} ranks"
    share = [tokens // world + (r < tokens % world) for r in range(world)]
    rows = slice(sum(share[:rank]), sum(share[:rank + 1]))
    x, tw, ti = R.make_routing(tokens, model_dim, experts_global, topk, seed=seed,
                               routing=routing)
    cap = 1 << (max(CAP, share[0]) - 1).bit_length()
    return dict(x=x[rows].contiguous(), W=_weights(rank, experts_global // world, model_dim,
                                                   inter_dim),
                topk_weights=tw[rows].contiguous(), topk_ids=ti[rows].contiguous(),
                comm=_comm(rank, world, cap), tokens=tokens, act=act,
                situ=R.SITU if act == "situv2" else None)


@ep_register
def moe_ep_check(out, expected, **_):
    """This rank's bf16 rows within ``R.COS_TOL`` -> {"cos": cos diff} (no collectives: a
    failure on any rank fails the config)."""
    return R.moe_check(out, expected)


@ep_register
def moe_ep_dims(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None):
    """The dims of a call, the same on every rank (also checks the arguments); not
    ``routing`` (a hint)."""
    E, H, I = weight_shape(W)
    M, topk = topk_ids.shape
    assert x.dtype == torch.bfloat16 and x.shape == (M, H) and M <= comm.max_tokens, (
        x.dtype, x.shape, comm.max_tokens)
    assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
    assert topk_weights.shape == (M, topk), topk_weights.shape
    return dict(tokens=tokens, model_dim=H, inter_dim=I, experts_global=E * comm.world,
                world=comm.world, topk=topk, act=act)


# ---- solver mega -------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _mega_variant(H, I, topk, token_shift, nblk, producers, lag, act, bcm):
    """(mega kernel, ep_plan slices, grid) of a MegaMoeEpA8W4 with these knobs."""
    slices = producers * 8  # ep_plan: one workgroup per producer wave, <= 1024 routes each
    assert producers % QUEUES == 0 and (topk << token_shift) <= slices * 1024, (
        f"producers {producers}: a multiple of {QUEUES}, >= {topk << token_shift} routes / 8192")
    kw = dict(H=H, I=I, TOPK=topk, token_shift=token_shift, producers=producers, lag=lag, ACT=act)
    if bcm != BCM_NT:
        kw.update(BCM1=bcm[0], BCM2=bcm[1])
    cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return K.ep_mega_kernel.get(**kw), slices, min(cus, K.max_tasks(nblk, H, I, producers))


@ep_register
@multi_gpu("world", init=mori_init)
@knob("producers", (8, 16, 32), id="p{}")
@knob("lag", (1, 2, 4), id="l{}")
@knob("bcm", {"bcm22": BCM_NT, "bcm01": BCM_BIG})
@meta(cached_loads=lambda bcm: bcm != BCM_NT)
def moe_ep_mega(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None,
                producers=16, lag=2, bcm=BCM_NT):
    """``comm.mega`` with the knobs (``W`` / ``situ`` / the knobs' kernel, slices and grid
    set on it for the call, as ``bench/dist.py`` does)."""
    mega = comm.mega(W, topk_ids.shape[1])
    launch, slices, grid = _mega_variant(mega.H, mega.I, mega.topk, mega.token_shift, mega.nblk,
                                         producers, lag, act, bcm)
    saved = mega.launch, mega.launch_big, mega.slices, mega.grid, mega.W, mega.situ
    mega.launch = mega.launch_big = launch
    mega.slices, mega.grid, mega.W, mega.situ = slices, grid, W, situ_args(**(situ or {}))
    try:
        return mega(x, topk_ids, topk_weights)
    finally:
        mega.launch, mega.launch_big, mega.slices, mega.grid, mega.W, mega.situ = saved


# ---- solver mega_small -------------------------------------------------------------------

@ep_register
@multi_gpu("world", init=mori_init)
@knob("bm", (32, 64), id="bm{}")
@meta(cached_loads=False)
def moe_ep_mega_small(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None,
                      bm=64, **over):
    """``comm.small``: one kernel per call, m-tiles of ``bm`` rows; ``over``: the other
    kernel knobs of ``megamoe.SMALL_KNOBS`` (experiments)."""
    small = comm.small(W, topk_ids.shape[1])
    assert tokens <= small.tmax, (
        f"mega_small: {tokens} tokens over all ranks > its fixed slots ({small.tmax} per expert)")
    return small(x, topk_ids, topk_weights, W=W, act=act, situ=situ, bm=bm, **over)


# ---- solvers mori, mori_eager ------------------------------------------------------------

def _mori(x, W, topk_weights, topk_ids, comm, act, situ, cfg, est):
    """fp8 quant -> MORI dispatch -> a8w4_moe_dyn ep3k ``cfg`` -> MORI combine (this rank's
    rows). ``est``: the row estimate of a call on the whole receive buffer, or None: read
    the received count on the host and run on those rows (est = count)."""
    op, quant = comm.mori(W, topk_ids.shape[1])
    E, H, _ = weight_shape(W)
    n = x.shape[0]
    if n:
        xq, xs = quant(x, quant_dtype=FP8)
    else:  # sglang skips the quant kernel on an empty batch
        xq, xs = x.to(FP8), x.new_empty((0, H // 128), dtype=torch.float32)
    rx, rtw, rxs, rti, count = op.dispatch(xq, topk_weights, xs, topk_ids)
    if est is None:
        rows = int(count.item())
        est = max(1, rows)
    else:
        rows, est = rx.shape[0], max(1, min(rx.shape[0], est))
    out = torch.empty((rx.shape[0], H), dtype=torch.bfloat16, device=x.device)
    if rows:
        MOE_DYN.run(dict(solver="ep3k", cfg=cfg), rx[:rows], W, rtw[:rows], rti[:rows], count,
                    x_scale=rxs[:rows], expert_offset=comm.rank * E, num_experts=E * comm.world,
                    act=act, situ=situ, est_tokens=est, out=out[:rows])
    return op.combine(out, None, topk_ids)[0][:n]


@ep_register
@multi_gpu("world", init=mori_init)
@knob("cfg", EP_CFGS, id="{}")
@meta(cached_loads=lambda cfg: _ep_cached(ep_preset(cfg)))
def moe_ep_mori(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None,
                cfg="l128"):
    """MORI dispatch / combine around ep3k ``cfg`` on the whole receive buffer (row
    estimate: ``tokens``); no host sync."""
    return _mori(x, W, topk_weights, topk_ids, comm, act, situ, cfg, tokens)


@ep_register
@multi_gpu("world", init=mori_init)
@knob("cfg", EP_CFGS, id="{}")
@no_capture("reads the received count on the host")
@meta(cached_loads=lambda cfg: _ep_cached(ep_preset(cfg)))
def moe_ep_mori_eager(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None,
                      cfg="l128"):
    """MORI dispatch / combine around ep3k ``cfg`` on the received rows (count read on the
    host)."""
    return _mori(x, W, topk_weights, topk_ids, comm, act, situ, cfg, None)


MOE_EP = ep_ops.moe_ep
