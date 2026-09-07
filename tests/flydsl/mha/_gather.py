"""Test-only full-cache gather from SHUFFLE-5D to linear THD.

Ported from 23cc6d1e:tests/flydsl/pa_4wave/test_pa_prefill.py:111-192.
The original uses one Triton program per token, four warps, Dq block256 and
Dv block128. The second grid dimension here extends it to multiple KV heads.
Triton is imported only when this explicit comparison is requested.
"""

import functools

import torch

if __package__:
    from ._references import ReferenceUnavailable
else:
    from _references import ReferenceUnavailable


@functools.cache
def _gather_kernel():
    try:
        import triton
        import triton.language as tl
    except (ImportError, OSError) as exc:
        raise ReferenceUnavailable(f"Triton gather unavailable: {exc}") from exc

    @triton.jit
    def _gather_swa_kv_kernel(k_cache, v_cache, slot_ids, k_linear, v_linear,
                             PAGE_SIZE: tl.constexpr, HEAD_DIM_QK: tl.constexpr,
                             HEAD_DIM_V: tl.constexpr, KV_HEADS: tl.constexpr,
                             VECTOR_SIZE: tl.constexpr, BLOCK_QK: tl.constexpr,
                             BLOCK_V: tl.constexpr):
        token = tl.program_id(0)
        head = tl.program_id(1)
        slot = tl.load(slot_ids + token).to(tl.int64)
        page, page_offset = slot // PAGE_SIZE, slot % PAGE_SIZE

        dim = tl.arange(0, BLOCK_QK)
        k_offset = ((page * KV_HEADS + head) * HEAD_DIM_QK * PAGE_SIZE
                    + (dim // VECTOR_SIZE) * PAGE_SIZE * VECTOR_SIZE
                    + page_offset * VECTOR_SIZE + dim % VECTOR_SIZE)
        key = tl.load(k_cache + k_offset, mask=dim < HEAD_DIM_QK, other=0)
        tl.store(k_linear + (token.to(tl.int64) * KV_HEADS + head) * HEAD_DIM_QK + dim,
                 key, mask=dim < HEAD_DIM_QK)

        dim_v = tl.arange(0, BLOCK_V)
        v_offset = ((page * KV_HEADS + head) * HEAD_DIM_V * PAGE_SIZE
                    + (page_offset // VECTOR_SIZE) * HEAD_DIM_V * VECTOR_SIZE
                    + dim_v * VECTOR_SIZE + page_offset % VECTOR_SIZE)
        value = tl.load(v_cache + v_offset, mask=dim_v < HEAD_DIM_V, other=0)
        tl.store(v_linear + (token.to(tl.int64) * KV_HEADS + head) * HEAD_DIM_V + dim_v,
                 value, mask=dim_v < HEAD_DIM_V)

    return _gather_swa_kv_kernel


def gather_swa_kv_call(case):
    """Prepare slots/workspace once; each returned call really reads 5D KV.

    As in the original benchmark, page metadata is fixed for a comparison.
    Rebuild this callable after changing lengths/page mapping. Cache *values*
    are always read again; no logical-KV Python oracle runs inside timing.
    All logical tokens are gathered, not just the visible SWA suffix.
    """
    if case.q.dtype != torch.bfloat16 or case.k.dtype != torch.bfloat16 or case.v.dtype != torch.bfloat16:
        raise ReferenceUnavailable("SWA gather+linear comparison requires BF16 inputs")
    if case.page != 64 or case.dq not in (128, 192) or case.dv != 128:
        raise ReferenceUnavailable("original SWA gather comparison supports page64, D128/192, V128")
    if case.q_offset or case.table_offset:
        raise ReferenceUnavailable("SWA gather+linear comparison requires dense prefix metadata")
    if not case.k.is_contiguous() or not case.v.is_contiguous():
        raise ValueError("gather requires contiguous SHUFFLE-5D caches")
    kernel = _gather_kernel()
    slots = []
    begin = 0
    for length in case.kv_lens:
        tokens = torch.arange(length, device=case.q.device, dtype=torch.int64)
        page_ids = case.indices[begin + tokens // case.page].to(torch.int64)
        slots.append(page_ids * case.page + tokens % case.page)
        begin += (length + case.page - 1) // case.page
    # Match the original compact slot table. BF16 cache byte spans already
    # bound the legal slot count well below signed-int32 capacity.
    slot_ids = torch.cat(slots).to(torch.int32).contiguous() if slots else torch.empty(0, device=case.q.device, dtype=torch.int32)
    total_kv = sum(case.kv_lens)
    linear_k = torch.empty(total_kv, case.kv_heads, case.dq, device=case.q.device, dtype=torch.bfloat16)
    linear_v = torch.empty(total_kv, case.kv_heads, case.dv, device=case.q.device, dtype=torch.bfloat16)

    def gather():
        if total_kv:
            kernel[(total_kv, case.kv_heads)](case.k, case.v, slot_ids, linear_k, linear_v,
                PAGE_SIZE=case.page, HEAD_DIM_QK=case.dq, HEAD_DIM_V=case.dv, KV_HEADS=case.kv_heads,
                VECTOR_SIZE=8, BLOCK_QK=256, BLOCK_V=128, num_warps=4)
        return linear_k, linear_v

    return gather, (linear_k, linear_v)