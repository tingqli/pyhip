"""Exact dense/direct/local-union FlyDSL QSA dispatch."""

import math

import msgspec
import torch

from . import dense, direct, kernel
from .contract import AttentionInputs
from .plan import SparsePlan, allocate_plan
from .plan import rebuild_plan as rebuild_union_plan

NAME = "flydsl_gfx942_multiquery_multihead"


class DispatchPlan(msgspec.Struct, frozen=True, kw_only=True):
    dense: dense.DensePlan
    direct: direct.DirectPlan | None
    union: SparsePlan | None
    mode: str


def prepare(
    *,
    inputs: AttentionInputs,
    query_tile: int = 32,
    grid_multiplier: int = 2,
    max_union_inflation: float = 1.5,
    mode: str = "auto",
    dense_limit: int = 2051,
    block_n: int = 32,
    sort_blocks: bool = True,
) -> DispatchPlan:
    if (
        inputs.q.device.type != "cuda"
        or torch.version.hip is None
        or torch.cuda.get_device_properties(inputs.q.device).gcnArchName.split(":")[0]
        != "gfx942"
    ):
        raise ValueError("The FlyDSL QSA kernel requires gfx942")
    if (
        inputs.q.dtype != torch.bfloat16
        or inputs.k.dtype != inputs.q.dtype
        or inputs.v.dtype != inputs.q.dtype
    ):
        raise ValueError("Q/K/V must be BF16")
    if (
        inputs.q.shape[-1] != 256
        or inputs.model.indexer_compress_ratio != 4
        or inputs.model.block_topk != 512
    ):
        raise ValueError("Require D256, compression4, block_topk512")
    if grid_multiplier < 1:
        raise ValueError("grid_multiplier must be positive")
    if math.isnan(max_union_inflation) or max_union_inflation < 0:
        raise ValueError("max_union_inflation must be nonnegative")
    for tensor in (inputs.q, inputs.k, inputs.v):
        if not tensor.is_contiguous() or tensor.device != inputs.q.device:
            raise ValueError("Q/K/V must be contiguous on the same GPU")
    if inputs.k.shape != inputs.v.shape or inputs.k.shape[-1] != 256:
        raise ValueError("K/V shapes must match with D256")
    if inputs.q.shape[1] % inputs.k.shape[1]:
        raise ValueError("Q heads must be divisible by KV heads")
    if mode not in ("auto", "direct", "union"):
        raise ValueError("mode must be auto, direct or union")
    dense_plan = dense.prepare(inputs=inputs, limit=dense_limit)
    union_plan = None
    direct_plan = None
    if sum(dense_plan.query_counts) != inputs.q.shape[0]:
        if mode != "direct":
            union_plan = allocate_plan(
                inputs=inputs,
                query_tile=query_tile,
                grid_multiplier=grid_multiplier,
                max_union_inflation=(
                    float("inf") if mode == "union" else max_union_inflation
                ),
                skip_counts=dense_plan.query_counts,
            )
        if mode != "union":
            direct_plan = direct.prepare(
                inputs=inputs,
                block_n=block_n,
                skip_counts=dense_plan.query_counts,
                union=union_plan,
                sort_blocks=sort_blocks,
            )
    plan = DispatchPlan(
        dense=dense_plan, direct=direct_plan, union=union_plan, mode=mode
    )
    rebuild_plan(inputs=inputs, plan=plan)
    return plan


def rebuild_plan(*, inputs: AttentionInputs, plan: DispatchPlan) -> None:
    with torch.cuda.device(inputs.q.device):
        if plan.union is not None:
            rebuild_union_plan(inputs=inputs, plan=plan.union)
        if plan.direct is not None:
            direct.rebuild_plan(inputs=inputs, plan=plan.direct)


def run(*, inputs: AttentionInputs, prepared: object, out: torch.Tensor) -> None:
    assert isinstance(prepared, DispatchPlan)
    if (
        out.shape != inputs.q.shape
        or out.dtype != inputs.q.dtype
        or out.device != inputs.q.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous and match Q shape, dtype, device")
    dense.run(inputs=inputs, prepared=prepared.dense, out=out)
    if prepared.union is not None:
        kernel.run(inputs=inputs, plan=prepared.union, out=out)
    if prepared.direct is not None:
        direct.run(inputs=inputs, prepared=prepared.direct, out=out)
