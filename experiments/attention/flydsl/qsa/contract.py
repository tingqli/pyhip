# Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""Portable packed sparse-attention contract, adapted from the SGLang QSA bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

import msgspec
import torch


class ModelShape(msgspec.Struct, frozen=True, kw_only=True):
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    max_position_embeddings: int

    @property
    def block_topk(self) -> int:
        return self.indexer_budget // self.indexer_compress_ratio

    @property
    def final_topk(self) -> int:
        return self.indexer_budget + self.indexer_compress_ratio - 1


def load_model_shape() -> ModelShape:
    path = Path(__file__).with_name("model_config.json")
    data = json.loads(path.read_text(encoding="utf-8"))["text_config"]
    return msgspec.convert(data, type=ModelShape)


class CaseSpec(msgspec.Struct, frozen=True, kw_only=True):
    name: Literal["no_prefix", "chunk_prefill"]
    query_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    attention_tp: int = 1
    selection: Literal["independent", "shared", "recent"] = "independent"
    selection_group: int = 8
    seed: int = 17


class AttentionInputs(msgspec.Struct, frozen=True, kw_only=True):
    spec: CaseSpec
    model: ModelShape
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    indices: torch.Tensor
    block_indices: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    kv_lens: torch.Tensor
    query_positions: torch.Tensor
    query_sequence_ids: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    scale: float


class Implementation(Protocol):
    NAME: str

    def prepare(self, *, inputs: AttentionInputs) -> object: ...

    def run(
        self,
        *,
        inputs: AttentionInputs,
        prepared: object,
        out: torch.Tensor,
    ) -> None: ...
