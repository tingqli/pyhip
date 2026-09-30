# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""SGLang QSA input generation, adapted to package-relative imports."""

from __future__ import annotations

import numpy as np
import torch

from .contract import AttentionInputs, CaseSpec, ModelShape, load_model_shape


def default_spec(
    *,
    name: str,
    query_tokens: int = 12000,
    prefix_tokens: int = 12000,
    attention_tp: int = 1,
    selection: str = "independent",
    selection_group: int = 8,
    seed: int = 17,
) -> CaseSpec:
    if name not in ("no_prefix", "chunk_prefill"):
        raise ValueError(f"Unknown case: {name}")
    if selection not in ("independent", "shared", "recent"):
        raise ValueError(f"Unknown selection: {selection}")
    return CaseSpec(
        name=name,
        query_lens=(query_tokens,),
        prefix_lens=(0 if name == "no_prefix" else prefix_tokens,),
        attention_tp=attention_tp,
        selection=selection,
        selection_group=selection_group,
        seed=seed,
    )


def _validate_spec(*, spec: CaseSpec, model: ModelShape) -> None:
    if not spec.query_lens or len(spec.query_lens) != len(spec.prefix_lens):
        raise ValueError("query_lens and prefix_lens must have equal nonzero length")
    if min(spec.query_lens) < 0 or sum(spec.query_lens) <= 0:
        raise ValueError("Query lengths must be nonnegative with a positive sum")
    if min(spec.prefix_lens) < 0 or spec.selection_group <= 0:
        raise ValueError("Prefixes must be nonnegative and selection_group positive")
    if spec.name == "no_prefix" and any(spec.prefix_lens):
        raise ValueError("no_prefix cannot contain cached prefix tokens")
    if spec.attention_tp not in (1, 2, 4, 8):
        raise ValueError("This model's benchmark supports attention TP 1, 2, 4, 8")
    lengths = tuple(q + p for q, p in zip(spec.query_lens, spec.prefix_lens))
    if max(lengths) > model.max_position_embeddings:
        raise ValueError("Requested context exceeds the saved model configuration")


def _select_blocks(
    *,
    count: int,
    topk: int,
    selection: str,
    priority: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    width = min(count, topk)
    if count <= topk:
        return np.arange(count, dtype=np.int32)
    if selection == "recent":
        return np.arange(count - width, count, dtype=np.int32)
    if selection == "shared":
        assert priority is not None
        return priority[priority < count][:width]
    return rng.choice(count, size=width, replace=False).astype(np.int32)


def _make_indices(*, spec: CaseSpec, model: ModelShape) -> tuple[np.ndarray, ...]:
    rows = sum(spec.query_lens)
    ratio = model.indexer_compress_ratio
    blocks = np.full((rows, model.block_topk), -1, dtype=np.int32)
    indices = np.full((rows, model.final_topk), -1, dtype=np.int32)
    positions = np.empty(rows, dtype=np.int32)
    sequence_ids = np.empty(rows, dtype=np.int32)
    rng = np.random.default_rng(spec.seed)
    row = 0
    offsets = np.arange(ratio, dtype=np.int32)
    for sequence, (length, prefix) in enumerate(zip(spec.query_lens, spec.prefix_lens)):
        priority = None
        for local_row in range(length):
            if spec.selection == "shared" and local_row % spec.selection_group == 0:
                priority = rng.permutation((prefix + length) // ratio).astype(np.int32)
            visible = prefix + local_row + 1
            chosen = _select_blocks(
                count=visible // ratio,
                topk=model.block_topk,
                selection=spec.selection,
                priority=priority,
                rng=rng,
            )
            tokens = (chosen[:, None] * ratio + offsets).reshape(-1)
            tail = np.arange(visible // ratio * ratio, visible, dtype=np.int32)
            blocks[row, : chosen.size] = chosen
            indices[row, : tokens.size] = tokens
            indices[row, tokens.size : tokens.size + tail.size] = tail
            positions[row] = visible - 1
            sequence_ids[row] = sequence
            row += 1
    return blocks, indices, positions, sequence_ids


def make_inputs(*, spec: CaseSpec, device: str | torch.device) -> AttentionInputs:
    model = load_model_shape()
    _validate_spec(spec=spec, model=model)
    device = torch.device(device)
    query_heads = model.num_attention_heads // spec.attention_tp
    kv_heads = max(1, model.num_key_value_heads // spec.attention_tp)
    kv_lens = tuple(q + p for q, p in zip(spec.query_lens, spec.prefix_lens))
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    q = torch.randn(
        (sum(spec.query_lens), query_heads, model.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (sum(kv_lens), kv_heads, model.head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn(k.shape, generator=generator, dtype=k.dtype, device=device)
    blocks, indices, positions, sequence_ids = _make_indices(spec=spec, model=model)
    cu_q = np.array((0,) + spec.query_lens, dtype=np.int32).cumsum(dtype=np.int32)
    cu_k = np.array((0,) + kv_lens, dtype=np.int32).cumsum(dtype=np.int32)
    return AttentionInputs(
        spec=spec,
        model=model,
        q=q,
        k=k,
        v=v,
        indices=torch.from_numpy(indices).to(device),
        block_indices=torch.from_numpy(blocks).to(device),
        cu_q=torch.from_numpy(cu_q).to(device),
        cu_k=torch.from_numpy(cu_k).to(device),
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        query_positions=torch.from_numpy(positions).to(device),
        query_sequence_ids=torch.from_numpy(sequence_ids).to(device),
        max_seqlen_q=max(spec.query_lens),
        max_seqlen_k=max(kv_lens),
        scale=model.head_dim**-0.5,
    )


def validate_inputs(*, inputs: AttentionInputs) -> None:
    model = inputs.model
    rows = inputs.q.shape[0]
    assert inputs.q.dtype == inputs.k.dtype == inputs.v.dtype == torch.bfloat16
    assert inputs.q.shape[-1] == inputs.k.shape[-1] == model.head_dim
    assert inputs.k.shape == inputs.v.shape
    assert inputs.q.shape[1] % inputs.k.shape[1] == 0
    assert inputs.indices.shape == (rows, model.final_topk)
    assert inputs.block_indices.shape == (rows, model.block_topk)
    for tensor in (
        inputs.indices,
        inputs.block_indices,
        inputs.cu_q,
        inputs.cu_k,
        inputs.kv_lens,
        inputs.query_positions,
        inputs.query_sequence_ids,
    ):
        assert tensor.dtype == torch.int32 and tensor.is_contiguous()
        assert tensor.device == inputs.q.device
    cu_q = inputs.cu_q.cpu().numpy()
    cu_k = inputs.cu_k.cpu().numpy()
    assert cu_q[0] == cu_k[0] == 0
    assert cu_q[-1] == rows and cu_k[-1] == inputs.k.shape[0]
    np.testing.assert_array_equal(np.diff(cu_q), inputs.spec.query_lens)
    np.testing.assert_array_equal(np.diff(cu_k), inputs.kv_lens.cpu().numpy())
    _validate_selections(inputs=inputs)


def _validate_selections(*, inputs: AttentionInputs) -> None:
    ratio = inputs.model.indexer_compress_ratio
    blocks = inputs.block_indices.cpu().numpy()
    indices = inputs.indices.cpu().numpy()
    positions = inputs.query_positions.cpu().numpy()
    sequence_ids = inputs.query_sequence_ids.cpu().numpy()
    offsets = np.arange(ratio, dtype=np.int32)
    row = 0
    for sequence, (length, prefix) in enumerate(
        zip(inputs.spec.query_lens, inputs.spec.prefix_lens)
    ):
        for local_row in range(length):
            visible = prefix + local_row + 1
            count = min(visible // ratio, inputs.model.block_topk)
            chosen = blocks[row, :count]
            assert sequence_ids[row] == sequence and positions[row] == visible - 1
            assert np.all(chosen >= 0) and np.all(chosen < visible // ratio)
            assert np.unique(chosen).size == count
            assert np.all(blocks[row, count:] == -1)
            expected = np.concatenate(
                (
                    (chosen[:, None] * ratio + offsets).reshape(-1),
                    np.arange(visible // ratio * ratio, visible, dtype=np.int32),
                )
            )
            np.testing.assert_array_equal(indices[row, : expected.size], expected)
            assert np.all(indices[row, expected.size :] == -1)
            row += 1
