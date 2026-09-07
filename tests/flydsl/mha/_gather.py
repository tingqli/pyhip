"""Test-only complete paged BF16 KV gather; never used by production dispatch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv(K, V, SLOTS, LK, LV, HK: tl.constexpr, DQ: tl.constexpr,
               DV: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + token)
    page, offset = slot // PAGE, slot % PAGE
    d = tl.arange(0, BLOCK)
    # BF16 SHUFFLE-5D: K[P,H,D/8,S,8], V[P,H,S/8,D,8].
    k_offset = (((page * HK + head) * (DQ // 8) + d // 8) * PAGE + offset) * 8 + d % 8
    v_offset = (((page * HK + head) * (PAGE // 8) + offset // 8) * DV + d) * 8 + offset % 8
    k = tl.load(K + k_offset, d < DQ, other=0)
    v = tl.load(V + v_offset, d < DV, other=0)
    tl.store(LK + (token * HK + head) * DQ + d, k, d < DQ)
    tl.store(LV + (token * HK + head) * DV + d, v, d < DV)


def gather_kv_call(case):
    """Preallocate slot mapping/workspace, but read cache on EVERY call."""
    if case.q.dtype != torch.bfloat16 or case.q_offset or case.table_offset:
        raise ValueError("gather reference requires BF16 with no prefix offsets")
    slots, pos = [], 0
    for length in case.kv_lens:
        ids = case.page_order[pos:pos + (length + case.page - 1) // case.page]
        slots.extend(ids[i // case.page] * case.page + i % case.page for i in range(length))
        pos += len(ids)
    slot_tensor = torch.tensor(slots, device=case.q.device, dtype=torch.int64)
    k = torch.empty((len(slots), case.kv_heads, case.dq), device=case.q.device, dtype=torch.bfloat16)
    v = torch.empty((len(slots), case.kv_heads, case.dv), device=case.q.device, dtype=torch.bfloat16)

    def gather():
        if slots:
            _gather_kv[(len(slots), case.kv_heads)](
                case.k, case.v, slot_tensor, k, v, case.kv_heads, case.dq, case.dv,
                case.page, triton.next_power_of_2(max(case.dq, case.dv)), num_warps=4)
        return k, v
    return gather, (k, v)