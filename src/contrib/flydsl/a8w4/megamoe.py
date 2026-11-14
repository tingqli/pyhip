# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""megamoe_ep_a8w4: the EP8 MoE layer (dispatch + a8w4 experts + combine) as one
persistent kernel, for large batches (prefill, >= 2048 tokens in total).

Replaces MORI dispatch -> ``a8w4_moe_dyn`` -> MORI combine. Per call, on one stream,
without host synchronization:

  ep_quant   x bf16 -> fp8 + one f32 scale per (token, 128 columns); out[:T] zeroed
  ep_prep    count this rank's routes per global expert, exchange the counts with all
             ranks, derive every expert's rows on its destination (1 workgroup)
  ep_plan    every route's destination row and send order; this rank's blocks
  mega       kernels/ep_mega.py: producers write each route's row straight into its
             destination, gemm1 / gemm2 tasks run on the blocks as their rows arrive,
             gemm2 adds every route into out[token] of its source rank (BF16 atomics)
  ep_finish  all ranks' adds are done -> out is complete

Needs: 8 ranks of one node (EP8, 32 local experts each), MORI shmem initialized
(``mori.shmem.shmem_torch_process_group_init``: the buffers peers write live in its
symmetric heap), and the whole GPU while it runs (one workgroup on every CU). All kernels
are FlyDSL (``kernels/ep_mega.py``, ``kernels/ep_mega_comm.py``).
"""

import torch

from ._vendor.tensor_shim import _run_compiled
from .kernels import ep_mega as K
from .kernels import ep_mega_comm as C
from .kernels import tp_gemm1 as G1


class MegaMoeEpA8W4:
    """One rank's EP8 MoE layer. ``W``: ``prepare_weights`` of the rank's 32 experts
    (global experts [32 rank, 32 rank + 32)); ``max_tokens``: tokens per rank and call.
    ``s_cap``: rows this rank can receive (default: twice the uniform-routing average
    plus one partial block per expert; routes past it are dropped, see ``rows_needed``).
    ``producers``: producer workgroups (a multiple of 8); ``lag``: gemm2 trails gemm1 by
    this many block pairs of a queue. ``act`` / ``situ``: as ``a8w4_moe``."""

    def __init__(self, W, *, rank, max_tokens, topk=8, world=8, s_cap=None, producers=16,
                 lag=2, act="silu", situ=None):
        import mori.shmem as ms
        from mori.shmem import mori_shmem_create_tensor

        E, N2, _ = W["w1"].shape
        assert world == K.NPES and E == K.E_LOCAL, "EP8 x 32 local experts"
        assert topk in (1, 2, 4, 8) and producers % K.QUEUES == 0
        self.W, self.rank, self.topk = W, rank, topk
        self.H, self.I = W["w2"].shape[1], N2 // 2
        self.producers, self.lag, self.act = producers, lag, act
        self.situ = G1.situ_args(**(situ or {}))
        self.token_shift = max(int(max_tokens - 1).bit_length(), 1)
        self.tcap = 1 << self.token_shift
        bm = K.BLOCK_M
        s_cap = s_cap or 2 * self.tcap * topk + K.E_LOCAL * bm
        self.s_cap = -(-s_cap // bm) * bm
        assert self.s_cap < 1 << K.ROW_BITS
        self.nblk = self.s_cap // bm
        H, I, S = self.H, self.I, self.s_cap
        dev = torch.device("cuda", torch.cuda.current_device())

        def sym(shape, dtype):
            t = mori_shmem_create_tensor(shape, dtype)
            t.zero_()
            return t

        self.recv_a = sym((S, H), torch.uint8)
        self.recv_scale = sym((S, H // 128), torch.float32)
        self.down_ids = sym((S,), torch.int32)
        self.recv_w = sym((S,), torch.float32)
        self.ws = sym((K.ARRIVED + self.nblk,), torch.int32)
        self.out = sym((self.tcap, H), torch.bfloat16)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.tables = {
            name: torch.tensor([ms.shmem_ptr_p2p(getattr(self, name).data_ptr(), rank, p)
                                for p in range(world)], dtype=torch.int64, device=dev)
            for name in ("recv_a", "recv_scale", "down_ids", "recv_w", "ws", "out")}
        i32 = dict(dtype=torch.int32, device=dev)
        self.route_idx = torch.empty(self.tcap * topk, **i32)
        self.send_route = torch.empty(self.tcap * topk, **i32)
        self.send_info = torch.empty(self.tcap * topk, **i32)
        self.num_valid = torch.zeros(1, **i32)
        self.status = torch.zeros(8, **i32)
        self.plan = torch.empty(C.PLAN_INTS, **i32)
        self.expert_ids = torch.zeros(self.nblk, **i32)
        self.sched = torch.zeros(K.SCHED_EXPECTED + 2 * self.nblk, **i32)
        # ep_plan: one workgroup per producer wave, each sorting up to 1024 routes
        self.slices = producers * 8
        assert self.tcap * topk <= self.slices * 1024, "too many routes per ep_plan slice"
        self.a2 = torch.empty((S, I), dtype=torch.uint8, device=dev)
        self.a2_scale = torch.empty((S, I // 128), dtype=torch.float32, device=dev)
        self.x8 = torch.empty((self.tcap, H), dtype=torch.uint8, device=dev)
        self.xs = torch.empty((self.tcap, H // 128), dtype=torch.float32, device=dev)
        self.grid = min(torch.cuda.get_device_properties(dev).multi_processor_count,
                        K.max_tasks(self.nblk, H, I, producers))
        self.launch = K.compile_ep_mega(H=H, I=I, TOPK=topk, token_shift=self.token_shift,
                                        producers=producers, lag=lag, ACT=act)
        self.quant_k, self.prep_k = C.compile_quant(H), C.compile_prep()
        self.plan_k, self.finish_k = C.compile_plan(), C.compile_finish()
        self.epoch = 0

    def __call__(self, x, topk_ids, topk_weights, x_scale=None, marks=None):
        """x: bf16 [T, H] (quantized here), or fp8 FNUZ [T, H] with ``x_scale`` f32
        [T, H/128]; topk_ids int32 [T, topk] (global ids), topk_weights f32 [T, topk].
        Returns bf16 [T, H], a view of the symmetric output buffer (valid until the next
        call). ``marks``: a list that gets (stage, CUDA event) after every stage (timing)."""

        def mark(name):
            if marks is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                marks.append((name, ev))

        mark("start")
        T, H = x.shape
        assert T <= self.tcap and H == self.H
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        topk_ids, topk_weights = topk_ids.contiguous(), topk_weights.contiguous()
        tab = self.tables
        stream = torch.cuda.current_stream()
        self.epoch += 1
        routes = T * self.topk
        quant = x_scale is None
        if quant:
            x8, xs = self.x8, self.xs
        else:
            assert x_scale.shape == (T, H // 128) and x_scale.dtype == torch.float32
            x8, xs = x.contiguous().view(torch.uint8), x_scale.contiguous()
        if T:
            waves = T * (H // 512)
            _run_compiled(self.quant_k, x.data_ptr() if quant else 0, x8.data_ptr(),
                          xs.data_ptr(), self.out.data_ptr(), T, int(quant), -(-waves // 4),
                          stream)
        mark("quant")
        _run_compiled(self.prep_k, topk_ids.data_ptr(), self.route_idx.data_ptr(),
                      self.ws.data_ptr(), tab["ws"].data_ptr(), self.num_valid.data_ptr(),
                      self.status.data_ptr(), self.sched.data_ptr(), self.plan.data_ptr(),
                      routes, self.rank, self.epoch, self.s_cap, stream)
        mark("prep")
        _run_compiled(self.plan_k, topk_ids.data_ptr(), self.route_idx.data_ptr(),
                      self.plan.data_ptr(), self.send_route.data_ptr(),
                      self.send_info.data_ptr(), self.down_ids.data_ptr(),
                      self.expert_ids.data_ptr(), self.sched.data_ptr(), routes, self.rank,
                      self.s_cap, self.slices, stream)
        mark("plan")
        W = self.W
        _run_compiled(
            self.launch, x8.data_ptr(), xs.data_ptr(), topk_weights.data_ptr(),
            self.send_route.data_ptr(), self.send_info.data_ptr(), tab["recv_a"].data_ptr(),
            tab["recv_scale"].data_ptr(), tab["down_ids"].data_ptr(), tab["recv_w"].data_ptr(),
            tab["ws"].data_ptr(), tab["out"].data_ptr(), self.recv_a.data_ptr(),
            self.recv_scale.data_ptr(), self.down_ids.data_ptr(), self.recv_w.data_ptr(),
            self.ws.data_ptr(), W["w1"].data_ptr(), W["w1_scale"].data_ptr(),
            W["w1_exp"].data_ptr(), W["w2"].data_ptr(), W["w2_scale"].data_ptr(),
            W["w2_exp"].data_ptr(), self.a2.data_ptr(), self.a2_scale.data_ptr(),
            self.expert_ids.data_ptr(), self.num_valid.data_ptr(), self.sched.data_ptr(),
            routes, T, self.rank, self.nblk, *self.situ, self.grid,
            stream)
        mark("mega")
        _run_compiled(self.finish_k, self.ws.data_ptr(), tab["ws"].data_ptr(), self.rank,
                      self.epoch, stream)
        mark("finish")
        return self.out[:T]

    def rows_needed(self):
        """Rows the last call needed on this rank (> s_cap: routes were dropped)."""
        return int(self.status[1].item())
