"""Small tensor fixtures and independent selected-token attention reference."""

import importlib
from itertools import accumulate
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pyhip.ops.qsa.flydsl.attention import attention
from pyhip.testing.qsa_reference import reference as full_reference

_runtime = importlib.import_module("pyhip.ops.qsa.flydsl.attention")
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "mytest/mydata"


def _metadata(q, k, v, indices, query_lens, prefix_lens, scale=0.0625):
    lengths = tuple(qn + pn for qn, pn in zip(query_lens, prefix_lens))
    kw = {"dtype": torch.int32, "device": q.device}
    return SimpleNamespace(q=q, k=k, v=v, indices=indices, query_lens=tuple(query_lens),
                           prefix_lens=tuple(prefix_lens), scale=scale,
                           cu_q=torch.tensor(tuple(accumulate(query_lens, initial=0)), **kw),
                           cu_k=torch.tensor(tuple(accumulate(lengths, initial=0)), **kw),
                           kv_lens=torch.tensor(lengths, **kw),
                           positions=torch.tensor([p + i for n, p in zip(query_lens, prefix_lens) for i in range(n)], **kw),
                           sequence_ids=torch.tensor([s for s, n in enumerate(query_lens) for _ in range(n)], **kw))


def _make_case(queries, prefixes, heads, device, seed=17, shared=False):
    rows, total = sum(queries), sum(queries) + sum(prefixes)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((rows, heads, 256), generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn((total, 1, 256), generator=generator, device=device, dtype=q.dtype)
    v = torch.randn(k.shape, generator=generator, device=device, dtype=q.dtype)
    indices = np.full((rows, 2051), -1, dtype=np.int32)
    rng, row = np.random.default_rng(seed), 0
    for count, prefix in zip(queries, prefixes):
        priority = None
        for local in range(count):
            visible = prefix + local + 1
            blocks = visible // 4
            if blocks <= 512:
                chosen = np.arange(blocks)
            elif shared:
                if priority is None or local % 32 == 0:
                    priority = rng.permutation((prefix + count) // 4)
                chosen = priority[priority < blocks][:512]
            else:
                chosen = rng.choice(blocks, 512, replace=False)
            tokens = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(blocks * 4, visible)))
            indices[row, :len(tokens)] = tokens
            row += 1
    return _metadata(q, k, v, torch.from_numpy(indices).to(device), queries, prefixes)


def _call(inputs, out=None):
    return attention(inputs.q, inputs.k, inputs.v, inputs.indices, query_lens=inputs.query_lens,
                     prefix_lens=inputs.prefix_lens, softmax_scale=inputs.scale, out=out)


def _base(inputs):
    return full_reference(inputs.q, inputs.k, inputs.v, inputs.indices,
                          query_lens=inputs.query_lens, prefix_lens=inputs.prefix_lens, scale=inputs.scale)


@torch.no_grad()
def reference(inputs, rows):
    outputs = []
    for row in rows:
        tokens = inputs.indices[row].long()
        tokens = tokens[tokens >= 0]
        sequence = int(inputs.sequence_ids[row])
        slots = tokens + int(inputs.cu_k[sequence])
        keys, values = inputs.k[slots].float(), inputs.v[slots].float()
        q = inputs.q[row].float().reshape(inputs.k.shape[1], -1, 256)
        scores = torch.einsum("ghd,kgd->ghk", q, keys) * inputs.scale
        outputs.append(torch.einsum("ghk,kgd->ghd", scores.softmax(-1), values).reshape(inputs.q.shape[1], 256))
    return torch.stack(outputs) if outputs else inputs.q.float()


def _gpu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm gfx942")
    torch.cuda.set_device(int(os.environ.get("QSA_REPLAY_GPU", "0")))
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("requires gfx942")
    return torch.device("cuda", torch.cuda.current_device())
