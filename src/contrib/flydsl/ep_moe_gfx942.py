"""gfx942 EP MoE with the all-to-all fused into the MoE kernels (MiMo-V2.6-Flash EP8: H 4096, I 2048, 256 experts,
top-8, FP8 128x128 block scales). See docs/megamoe_ep8_942/mega_moe_ep8_design.md.

No MORI and no sort. Per call, on one stream:
  ep_prep      count routes per global expert, exchange the count rows, derive every route's sorted row
  mode="mega" (ep_moe_mega_gfx942): one kernel dispatches, runs GateUp and Down interleaved, and adds every route
               with BF16 atomics into its source rank's out (P2P)
  mode="step1": the same as separate kernels: ep_dispatch, GateUp (token-major A scales), Down (remote atomics)
  ep_finish    all ranks' adds done -> out is complete
Buffers that peers write are MORI symmetric-heap tensors; the kernels get int64 tables of the peer addresses.
"""

import os

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

import pyhip

from .ep_moe_mega_gfx942 import ARRIVED, SCHED_EXPECTED, ep_mega
from .moe_down_a8w8_blockscale_gfx942_lds import moe_down_remote_a8w8_blockscale_gfx942_lds
from .moe_gateup_a8w8_blockscale_gfx942_lds import SORT_BLOCK_M, moe_gateup_a8w8_blockscale_gfx942_lds

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep_moe_comm_gfx942.cpp")
NPES, E_LOCAL = 8, 32
GATEUP_LAYOUT, DOWN_LAYOUT = "2x4_pf1", "4x2_pf1"


@pyhip.module(_SRC)
def ep_prep(topk_ids, num_routes, me, epoch, route_idx, ws, ws_peers, send_base, num_valid, s_cap, status, sched, plan): ...


@pyhip.module(_SRC)
def ep_plan(topk_ids, route_idx, num_routes, me, s_cap, plan, ws, send_route, send_info, gu_ids, down_ids, expert_ids,
            sched): ...


@pyhip.module(_SRC)
def ep_dispatch(x, x_scale, topk_ids, topk_w, route_idx, send_base, num_routes, topk, hidden, me, epoch, token_shift, s_cap,
                a_peers, s_peers, id_peers, w_peers, ws, ws_peers, status): ...


@pyhip.module(_SRC)
def ep_finish(me, epoch, ws, ws_peers): ...


def build_kernels():
    """Compile the HIP kernels once (call on one rank before the others load them)."""
    for k in (ep_prep, ep_plan, ep_dispatch, ep_finish):
        k.build()


class EPMoEGfx942:
    def __init__(self, rank, world, w1, w1_scale, w2, w2_scale, topk, max_tokens, s_cap=None, dispatch_wgs=64, mode="mega",
                 producers=16, lag=2, xcd_queues=True, relaxed=True, persistent=True, queue_order="paired", tasks_per_wg=1):
        assert world == NPES and w1.shape[0] == E_LOCAL, "kernels are specialized for EP8 x 32 local experts"
        assert mode in ("mega", "step1"), mode
        self.rank, self.world, self.topk = rank, world, topk
        self.mode, self.producers, self.lag, self.xcd_queues, self.relaxed = mode, producers, lag, xcd_queues, relaxed
        self.persistent, self.queue_order, self.tasks_per_wg = persistent, queue_order, tasks_per_wg
        self.w1, self.w1_scale, self.w2, self.w2_scale = w1, w1_scale, w2, w2_scale
        self.hidden, self.inter = w1.shape[2], w2.shape[2]
        self.token_shift = max(int(max_tokens - 1).bit_length(), 1)
        self.tcap = 1 << self.token_shift
        # Rows this rank can receive: twice the average route count, plus one partial block per expert.
        self.s_cap = s_cap or (2 * self.tcap * topk + E_LOCAL * SORT_BLOCK_M)
        assert self.s_cap % SORT_BLOCK_M == 0
        self.dispatch_wgs = dispatch_wgs
        dev = torch.device("cuda", torch.cuda.current_device())
        H, I, S = self.hidden, self.inter, self.s_cap

        def sym(shape, dtype):
            t = mori_shmem_create_tensor(shape, dtype)
            t.zero_()
            return t

        self.recv_a = sym((S, H), torch.uint8)
        self.recv_scale = sym((S, H // 128), torch.float32)
        self.down_ids = sym((S,), torch.int32)
        self.recv_w = sym((S,), torch.float32)
        self.ws = sym((ARRIVED + S // SORT_BLOCK_M,), torch.int32)
        self.out = sym((self.tcap, H), torch.bfloat16)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        def table(t):
            return torch.tensor([ms.shmem_ptr_p2p(t.data_ptr(), rank, p) for p in range(world)], dtype=torch.int64, device=dev)

        self.tables = {name: table(getattr(self, name)) for name in ("recv_a", "recv_scale", "down_ids", "recv_w", "ws", "out")}
        self.route_idx = torch.empty(self.tcap * topk, dtype=torch.int32, device=dev)
        self.send_base = torch.empty(NPES * E_LOCAL, dtype=torch.int32, device=dev)
        self.gu_ids = torch.empty(S, dtype=torch.int32, device=dev)
        self.expert_ids = torch.zeros(S // SORT_BLOCK_M, dtype=torch.int32, device=dev)
        self.num_valid = torch.zeros(1, dtype=torch.int32, device=dev)
        self.status = torch.zeros(8, dtype=torch.int32, device=dev)
        self.send_route = torch.empty(self.tcap * topk, dtype=torch.int32, device=dev)
        self.send_info = torch.empty(self.tcap * topk, dtype=torch.int32, device=dev)
        self.plan = torch.empty(3 * NPES * E_LOCAL + NPES + NPES * 9, dtype=torch.int32, device=dev)
        # ep_plan: one workgroup per producer wave, each sorting up to 1024 routes.
        self.slices = producers * 8
        assert self.tcap * topk <= self.slices * 1024, "too many routes per ep_plan slice"
        self.sched = torch.zeros(SCHED_EXPECTED + 2 * (S // SORT_BLOCK_M), dtype=torch.int32, device=dev)
        self.a2 = torch.empty((S, I), dtype=torch.uint8, device=dev)
        self.a2_scale_t = torch.empty((I // 128, S), dtype=torch.float32, device=dev)
        self.epoch = 0
        self.stage_events = None  # set to a list to record (name, event) after every stage
        self.trace = None  # set to an int32 tensor of trace_entries(self, producers) entries to trace the mega kernel

    def _mark(self, name):
        if self.stage_events is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.stage_events.append((name, ev))

    def forward(self, x_fp8, x_scale, topk_ids, topk_w):
        """x_fp8 [T, H] FP8 and x_scale [T, H/128] FP32 (per 1x128 quant), topk_ids [T, topk] global expert ids
        (int32), topk_w [T, topk] FP32. Returns out [T, H] BF16 (a view of the symmetric output buffer)."""
        T = x_fp8.shape[0]
        assert T <= self.tcap and topk_ids.dtype == torch.int32 and topk_w.dtype == torch.float32
        self.epoch += 1
        t = self.tables
        out = self.out[:T]
        self._mark("start")
        out.zero_()
        routes = T * self.topk
        ep_prep([1], [1024], topk_ids, routes, self.rank, self.epoch, self.route_idx, self.ws, t["ws"], self.send_base,
                self.num_valid, self.s_cap, self.status, self.sched, self.plan)
        ep_plan([self.slices], [256], topk_ids, self.route_idx, routes, self.rank, self.s_cap, self.plan, self.ws,
                self.send_route, self.send_info, self.gu_ids, self.down_ids, self.expert_ids, self.sched)
        self._mark("prep")
        if self.mode == "mega":
            ep_mega(self, x_fp8, x_scale, topk_w, self.producers, self.lag, self.xcd_queues, self.relaxed, self.persistent,
                    self.queue_order, self.tasks_per_wg, self.trace)
            self._mark("mega")
            ep_finish([1], [64], self.rank, self.epoch, self.ws, t["ws"])
            self._mark("finish")
            return out
        ep_dispatch([self.dispatch_wgs], [256], x_fp8, x_scale, topk_ids, topk_w, self.route_idx, self.send_base, routes,
                    self.topk, self.hidden, self.rank, self.epoch, self.token_shift, self.s_cap, t["recv_a"], t["recv_scale"],
                    t["down_ids"], t["recv_w"], self.ws, t["ws"], self.status)
        self._mark("dispatch")
        fp8 = x_fp8.dtype
        recv_a, recv_scale, out_table = self.recv_a, self.recv_scale, t["out"]
        debug = os.environ.get("EP_MOE_DEBUG", "")
        if "copy" in debug:  # diagnostic: GateUp on plain torch copies of the received rows
            recv_a, recv_scale = recv_a.clone(), recv_scale.clone()
            self._mark("debug_copy")
        if "local" in debug:  # diagnostic: Down adds into a local scratch instead of the source ranks
            if not hasattr(self, "_scratch"):
                self._scratch = torch.zeros_like(self.out)
                self._scratch_table = torch.full_like(t["out"], self._scratch.data_ptr())
            out_table = self._scratch_table
        moe_gateup_a8w8_blockscale_gfx942_lds(
            recv_a.view(fp8), self.w1, recv_scale, self.w1_scale, self.gu_ids, self.expert_ids, self.num_valid,
            self.a2.view(fp8), layout=GATEUP_LAYOUT, out_scale_t=self.a2_scale_t, topk=self.topk, a_scale_token_major=True,
        )
        self._mark("gateup")
        moe_down_remote_a8w8_blockscale_gfx942_lds(
            self.a2.view(fp8), self.w2, self.a2_scale_t, self.w2_scale, self.down_ids, self.recv_w, self.expert_ids,
            self.num_valid, out_table, self.token_shift, self.topk, layout=DOWN_LAYOUT,
        )
        self._mark("down")
        ep_finish([1], [64], self.rank, self.epoch, self.ws, t["ws"])
        self._mark("finish")
        return out

    def rows_needed(self):
        """Rows the last call needed on this rank (> s_cap means routes were dropped)."""
        return int(self.status[1].item())
