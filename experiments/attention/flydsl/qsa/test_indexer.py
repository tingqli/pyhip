"""Prefill QSA indexer: SGLang-eager replay of real captures, synthetic layouts and timing.

Pytest runs correctness only (-m perf times). The CLI replays captured inputs; base
is SGLang's own ``QSAIndexer.forward_cuda`` on this ROCm stack (Torch MQA fallback,
radix fast_topk, Triton expand), PyHIP is the plugin adapter plus ``indexer.py``,
either with SGLang's index_qk_proj GEMM (bit-exact prep; ``check``) or with the
plugin's default hipBLASLt projection (rounding-level changes; ``check_projection``).
The indexer weights and heads are replicated, so per-rank work is TP2/4/8-invariant.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "experiments.attention.flydsl.qsa"

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "mytest/mydata"
AITER_CONFIGS = ROOT / "mytest/sglang_tp2_base_20260925_01/aiter_configs_snapshot"
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("AITER_CONFIG_GEMM_BF16", str(AITER_CONFIGS / "bf16_tuned_gemm.csv"))

from . import indexer  # noqa: E402
from .sglang import plugin  # noqa: E402

REAL_INPUTS = DATA / "qsa_indexer_20260928_01/capture/inputs"
BENCHMARK_BUFFERS = 10
BENCHMARK_SAMPLES = 128
GATE_SETTLE_SECONDS = 3
RATIO, TOPK, WIDTH = 4, 512, 2051
# (sequence lengths, extend lengths): prefix = sequence - extend.
CASES = (
    ((1,), (1,)), ((7,), (7,)), ((1000,), (1000,)), ((2051,), (2051,)), ((2060,), (2060,)),
    ((5003,), (5003,)), ((3000, 777, 2100), (1976, 777, 2036)), ((20000,), (3616,)),
    ((33000,), (16616,)), ((70000,), (4096,)), ((131077, 3001), (4101, 3001)), ((262144,), (16384,)),
)


def _hash(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    return hashlib.sha256(raw).hexdigest()


def _gpu():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm gfx942")
    torch.cuda.set_device(int(os.environ.get("QSA_REPLAY_GPU", "0")))
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("requires gfx942")
    return torch.device("cuda", torch.cuda.current_device())


@contextlib.contextmanager
def _production_rope():
    # The server's attention backend is aiter, so MRoPE composes cos/sin with Triton.
    from sglang.srt.layers.rotary_embedding import mrope

    original = mrope.attention_backends
    mrope.attention_backends = lambda: ("aiter", "aiter")
    try:
        yield
    finally:
        mrope.attention_backends = original


def _module(weights, cache, layer, device, section=(11, 11, 10), interleaved=True):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rotary = MRotaryEmbedding(256, 64, 64, 10_000_000, True, torch.bfloat16,
                                  mrope_section=list(section), mrope_interleaved=interleaved)
        config = SimpleNamespace(indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048,
                                 indexer_compress_ratio=4, hidden_size=2560, rms_norm_eps=1e-6)
        module = QSAIndexer(config, layer_id=layer, rotary_emb=rotary)
    finally:
        torch.set_default_dtype(default)
    module = module.to(device)
    rotary.cos_sin_cache = cache.to(device)
    with torch.no_grad():
        module.index_qk_proj.weight.copy_(weights["index_qk_weight"])
        module.q_layernorm.weight.copy_(weights["q_norm_weight"])
        module.k_layernorm.weight.copy_(weights["k_norm_weight"])
    return module


def _pool(state, layer):
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.full_attention_layer_id_mapping = {layer: 0}
    pool.qsa_key_state_buffer_pool = [state["key_state"].clone()]
    pool.qsa_compressed_k_buffer_pool = [state["compressed"].clone()]
    pool.qsa_rope_position_buffer = state["rope_state"].clone()
    pool.qsa_compress_ratio, pool.qsa_index_head_dim, pool.qsa_index_kv_heads = RATIO, 128, 1
    pool.qsa_compressed_page_size, pool.qsa_block_topk, pool.qsa_token_topk = 16, TOPK, 2048
    return pool


def _metadata(case, pool):
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata

    return QSAIndexerMetadata(token_to_kv_pool=pool, compress_ratio=RATIO, block_topk=TOPK, **case.fields)


def _batch(case):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    prefixes = [s - e for s, e in zip(case.seq_lens, case.extend_lens)]
    return SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=case.logical,
                           seq_lens_cpu=torch.tensor(case.seq_lens), extend_seq_lens_cpu=list(case.extend_lens),
                           extend_prefix_lens_cpu=prefixes)


def _state(device, rows, slots, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    return dict(key_state=torch.randn((rows, 1, 128), generator=generator, device=device).to(torch.bfloat16),
                rope_state=torch.randint(0, 1 << 20, (rows, 3), generator=generator, device=device),
                compressed=torch.randn((slots, 1, 128), generator=generator, device=device).to(torch.bfloat16))


def load(path, device):
    value = torch.load(path, map_location="cpu", weights_only=True)
    meta, tensors = value["metadata"], value["tensors"]
    for name, tensor in tensors.items():
        assert _hash(tensor) == meta["tensor_metadata"][name]["sha256"], (path, name)
    tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    fields = {name.split(".", 1)[1]: tensor for name, tensor in tensors.items() if name.startswith("metadata.")}
    ring = (int(tensors["metadata.req_pool_indices"].max()) + 1) * RATIO
    slots = int(max(tensors["compressed_slots"].max(), tensors["metadata.write_locs"].max())) + 1
    return SimpleNamespace(
        name=path.stem, layer=meta["layer_id"], seq_lens=tuple(meta["seq_lens"]),
        extend_lens=tuple(meta["extend_seq_lens"]), hidden=tensors["hidden_states"],
        positions=tensors["positions"], logical=tensors["batch_positions"], fields=fields,
        module=_module(tensors, tensors["cos_sin_cache"], meta["layer_id"], device,
                       meta["mrope_section"], meta["mrope_interleaved"]),
        state=_state(device, ring, slots, 7), captured=tensors["output"],
        captured_state=(tensors["ring_rows"], tensors["ring_key_state"], tensors["ring_rope_positions"],
                        tensors["compressed_slots"], tensors["compressed_keys"]),
        capture=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_tp_rank=meta.get(
            "tp_rank"), metadata={k: v for k, v in meta.items() if k != "tensor_metadata"}))


def synthetic(seq_lens, extend_lens, device, seed=11):
    """Page-aligned request slots and SGLang's own extend write plan/ring/rope helpers."""
    from sglang.srt.layers.attention.qsa.metadata import build_pending_ring_slots, build_rope_position_matrix
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    generator = torch.Generator(device=device).manual_seed(seed)
    rows, batch = sum(extend_lens), len(seq_lens)
    prefixes = [s - e for s, e in zip(seq_lens, extend_lens)]
    width = -(-max(seq_lens) // 64) * 64
    starts = [64 * (1 + i) + i * width for i in range(batch)]
    table = torch.stack([torch.arange(s, s + width, device=device) for s in starts]).to(torch.int32)
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    logical = torch.cat([torch.arange(p, s, device=device) for p, s in zip(prefixes, seq_lens)])
    positions = logical[None].expand(3, -1).contiguous()
    extend = torch.tensor(extend_lens, device=device)
    prefix = torch.tensor(prefixes, device=device)
    write_locs, ends, sequences, members = QwenSparseAttnBackend._qsa_write_plan(
        token_slot_table=table, start_blocks=prefix // RATIO, end_blocks=lengths.long() // RATIO,
        capacity=rows // RATIO + batch, compress_ratio=RATIO, row_token_starts=torch.cumsum(extend, 0) - extend,
        prefix_lens=prefix)
    to_batch = torch.repeat_interleave(torch.arange(batch, device=device, dtype=torch.int32), extend)
    requests = torch.arange(1, batch + 1, device=device)
    fields = dict(sequence_lengths=lengths, token_to_batch_idx=to_batch, token_slot_table=table,
                  out_cache_loc=torch.zeros(rows, dtype=torch.int64, device=device), req_pool_indices=requests,
                  write_locs=write_locs, compress_group_positions=ends, compress_sequence_ids=sequences,
                  compress_member_rows=members,
                  pending_ring_slots=build_pending_ring_slots(token_to_batch_idx=to_batch, req_pool_indices=requests,
                                                              sequence_lengths=lengths, logical_positions=logical,
                                                              compress_ratio=RATIO, is_extend=True),
                  extend_rope_matrix=build_rope_position_matrix(positions, rows))
    weights = dict(index_qk_weight=torch.randn((640, 2560), generator=generator, device=device) / 50,
                   q_norm_weight=torch.randn(128, generator=generator, device=device) / 4,
                   k_norm_weight=torch.randn(128, generator=generator, device=device) / 4)
    cache = torch.randn((max(seq_lens) + 64, 64), generator=generator, device=device).to(torch.bfloat16)
    return SimpleNamespace(
        name=f"synthetic_s{'-'.join(map(str, seq_lens))}_e{'-'.join(map(str, extend_lens))}", layer=3,
        seq_lens=tuple(seq_lens), extend_lens=tuple(extend_lens),
        hidden=torch.randn((rows + 3, 2560), generator=generator, device=device).to(torch.bfloat16),
        positions=torch.cat((positions, positions[:, :3]), dim=1), logical=torch.cat((logical, logical[:3])),
        fields=fields, module=_module(weights, cache, 3, device),
        state=_state(device, (batch + 1) * RATIO, (starts[-1] + width) // RATIO + 1, seed), captured=None,
        captured_state=None, capture=None)


def base(case, pool):
    with _production_rope(), torch.no_grad():
        return case.module.forward_cuda(case.hidden, case.positions, _batch(case), _metadata(case, pool))


def pyhip(case, pool, intermediates=False, projection=None):
    """Plugin adapter + runtime; projection=None keeps SGLang's (bit-exact) index_qk_proj GEMM."""
    with torch.no_grad():
        metadata = _metadata(case, pool)
        inputs = plugin._indexer_inputs(case.module, case.hidden, case.positions, _batch(case), metadata, projection)
        assert inputs is not None, "adapter rejected an eligible extend call"
        result = indexer._prefill(inputs.pop("qk"), **inputs)
        return result if intermediates else result[0]


_plugin_state = None


def fast_projection(module, rows):
    """The plugin's default hipBLASLt index projection policy (None below its row threshold)."""
    global _plugin_state
    if _plugin_state is None:
        _plugin_state = plugin._State()
    return _plugin_state.projection(module, rows)


def _counts(case):
    lengths = [s for s, e in zip(case.seq_lens, case.extend_lens) for _ in range(e)]
    lengths = torch.tensor(lengths, device=case.logical.device)
    rows = sum(case.extend_lens)
    return torch.minimum((case.logical[:rows] + 1) // RATIO, lengths // RATIO), lengths


def _blocks(output, counts):
    first = output[:, :TOPK * RATIO:RATIO]
    live = torch.arange(TOPK, device=output.device)[None] < counts.clamp_max(TOPK)[:, None]
    return torch.where(live, first // RATIO, -1)


def _logits(case, pool, rows):
    """FP64 reference logits of the given rows over their request's compressed keys."""
    metadata = _metadata(case, pool)
    total = sum(case.extend_lens)
    with _production_rope(), torch.no_grad():
        q, _, _ = case.module.project_qk(case.hidden[:total], case.positions[:, :total])
        keys, starts, ends, _ = metadata.get_prefill_mqa_inputs(case.layer, case.logical[:total])
    q, keys = q[rows].double(), keys[:, 0].double()
    scores = torch.relu(torch.einsum("mhd,nd->mnh", q, keys)).sum(-1) / math.sqrt(128)
    return scores, starts[rows], ends[rows]


def _violation(case, pool, rows, blocks):
    """Worst FP64 (max unchosen - min chosen) / max|logit| over the given rows (pool: base, after its call)."""
    logits, starts, ends = _logits(case, pool, rows)
    worst = 0.0
    for local, row in enumerate(rows.tolist()):
        count = int(ends[local] - starts[local])
        values = logits[local, starts[local]:ends[local]]
        chosen = torch.zeros(count, dtype=torch.bool, device=values.device)
        chosen[blocks[row][blocks[row] >= 0].long()] = True
        gap = float(values[~chosen].max() - values[chosen].min()) if count > TOPK else 0.0
        worst = max(worst, gap / max(float(values.abs().max()), 1e-30))
    return worst


def check(case, *, compare_captured=True):
    """Compare token ABI, per-row block sets (tie-aware) and pool writes against SGLang eager."""
    from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices

    rows = sum(case.extend_lens)
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    expected = base(case, base_pool)
    actual, q, packed = pyhip(case, new_pool, intermediates=True)
    torch.cuda.synchronize()
    with _production_rope(), torch.no_grad():
        q_base, _, _ = case.module.project_qk(case.hidden[:rows], case.positions[:, :rows])
        keys, _, _, _ = _metadata(case, base_pool).get_prefill_mqa_inputs(case.layer, case.logical[:rows])
    assert torch.equal(q, q_base), "normed/rotated index Q must be bit-exact"
    assert torch.equal(packed[:keys.shape[0]], keys[:, 0]), "packed compressed keys must be bit-exact"
    assert actual.shape == expected.shape == (rows, WIDTH) and actual.dtype == torch.int32
    counts, lengths = _counts(case)
    blocks = _blocks(actual, counts)
    live = torch.arange(TOPK, device=blocks.device)[None] < counts.clamp_max(TOPK)[:, None]
    assert torch.equal(blocks >= 0, live) and bool((blocks < counts[:, None]).all()), "blocks must be causal"
    ordered = torch.where(live, blocks, 1 << 30).sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < 1 << 30)).any()), "duplicate blocks"
    torch.testing.assert_close(torch_expand_qsa_block_indices(blocks, case.logical[:rows], lengths, RATIO, 2048),
                               actual, rtol=0, atol=0)
    different = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1)
    report = dict(case=case.name, rows=rows, different_token_sets=int(different.sum()))
    if different.any():
        worst = _violation(case, base_pool, different.nonzero().flatten()[:256], blocks)
        report["worst_relative_boundary_violation"] = worst
        assert worst <= 1e-5, report
    for name, a, b in zip(("key_state", "rope_state", "compressed"),
                          plugin._indexer_state(_metadata(case, new_pool), case.layer),
                          plugin._indexer_state(_metadata(case, base_pool), case.layer)):
        assert torch.equal(a, b), name
    for name in ("key_state", "rope_state", "compressed"):
        old, dump = case.state[name], RATIO if name != "compressed" else 1
        changed = [(_buffer(pool, name) != old).flatten(1).any(1)[dump:] for pool in (base_pool, new_pool)]
        assert torch.equal(*changed), f"{name} write footprint differs"
    if compare_captured and case.captured is not None:
        ring, key_state, rope_state, slots, compressed = case.captured_state
        captured = (case.captured.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1)
        report["base_vs_captured_different_token_sets"] = int(captured.sum())
        report["pyhip_vs_captured_different_token_sets"] = int(
            (case.captured.sort(dim=1).values != actual.sort(dim=1).values).any(dim=1).sum())
        # Ring rows keep stale values unless this call leaves pending (incomplete-group) tokens.
        written = torch.isin(ring, case.fields["pending_ring_slots"][:rows]) & (ring >= RATIO)
        report["captured_pending_ring_rows"] = int(written.sum())
        assert torch.equal(new_pool.get_qsa_key_state_buffer(case.layer)[ring[written]], key_state[written])
        assert torch.equal(new_pool.qsa_rope_position_buffer[ring[written]], rope_state[written])
        assert torch.equal(new_pool.get_qsa_compressed_k_buffer(case.layer)[slots], compressed)
    first = actual.clone()
    again = pyhip(case, _pool(case.state, case.layer))
    assert torch.equal(again, first), "PyHIP indexer must be deterministic"
    return report


def check_projection(case):
    """hipBLASLt projection vs SGLang's GEMM: rounding-level qk, pool and near-tie selection changes only."""
    rows = sum(case.extend_lens)
    assert fast_projection(case.module, rows) is not None, "hipBLASLt projection not selected"
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    expected = base(case, base_pool)
    actual = pyhip(case, new_pool, projection=fast_projection)
    with torch.no_grad():
        exact = case.module.index_qk_proj(case.hidden[:rows])[0]
        fast = fast_projection(case.module, rows)(case.hidden[:rows])
    torch.cuda.synchronize()
    different = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1)
    # Both are BF16 roundings of FP32 sums in different orders: elements may differ by one output ulp,
    # or more near cancellation, but never by more than a BF16 ulp of the tensor's scale.
    error = float((fast.float() - exact.float()).abs().max())
    scale = float(exact.float().abs().max())
    report = dict(case=case.name, rows=rows, qk_mismatches=int((fast != exact).sum()), qk_max_abs_error=error,
                  qk_max_abs=scale, different_token_sets=int(different.sum()))
    counts, _ = _counts(case)
    blocks = _blocks(actual, counts)
    worst = _violation(case, base_pool, different.nonzero().flatten()[:256], blocks) if different.any() else 0.0
    report["worst_relative_boundary_violation"] = worst
    for name, a, b in zip(("key_state", "rope_state", "compressed"),
                          plugin._indexer_state(_metadata(case, new_pool), case.layer),
                          plugin._indexer_state(_metadata(case, base_pool), case.layer)):
        report[f"{name}_mismatches"] = int((a != b).sum())
    assert error <= scale * 2 ** -8 and report["qk_mismatches"] <= fast.numel() // 1000, report
    assert report["rope_state_mismatches"] == 0, report
    assert report["different_token_sets"] <= rows // 100 and worst <= 2e-3, report
    return report


def _buffer(pool, name):
    return {"key_state": pool.qsa_key_state_buffer_pool[0], "rope_state": pool.qsa_rope_position_buffer,
            "compressed": pool.qsa_compressed_k_buffer_pool[0]}[name]


def _real_files():
    folder = Path(os.environ.get("QSA_INDEXER_INPUT_DIR", REAL_INPUTS))
    if "QSA_INDEXER_INPUT_DIR" in os.environ and not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(folder.glob("indexer_tp*_layer*_m*.pt")) if folder.is_dir() else []


@pytest.mark.parametrize("seq_lens,extend_lens", CASES,
                         ids=[f"s{'-'.join(map(str, s))}_e{'-'.join(map(str, e))}" for s, e in CASES])
def test_synthetic(seq_lens, extend_lens):
    print(check(synthetic(seq_lens, extend_lens, _gpu())))


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_real_capture(path):
    print(check(load(path, _gpu())))


@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_real_capture_hipblaslt_projection(path):
    print(check_projection(load(path, _gpu())))


def test_ineligible_calls_keep_sglang():
    device = _gpu()
    case = synthetic((300,), (300,), device)
    batch = _batch(case)
    metadata = _metadata(case, _pool(case.state, case.layer))
    assert plugin._indexer_inputs(case.module, case.hidden, case.positions, batch, metadata) is not None
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    for change in (dict(forward_mode=ForwardMode.DECODE), dict(seq_lens_cpu=None),
                   dict(seq_lens_cpu=torch.tensor([70000]))):
        assert plugin._indexer_inputs(case.module, case.hidden, case.positions,
                                      SimpleNamespace(**{**vars(batch), **change}), metadata) is None
    fp32 = case.hidden.float()
    assert plugin._indexer_inputs(case.module, fp32, case.positions, batch, metadata) is None
    # 65536 compressed keys (262144 tokens, the model maximum) is the largest eligible request.
    for length, eligible in ((262144, True), (262148, False)):
        case = synthetic((length,), (4,), device)
        inputs = plugin._indexer_inputs(case.module, case.hidden, case.positions, _batch(case),
                                        _metadata(case, _pool(case.state, case.layer)))
        assert (inputs is not None) == eligible, length


def test_plugin_validation_then_fast_projection(monkeypatch):
    """PYHIP_QSA_VALIDATE checks a layout once with SGLang's GEMM; repeats use the hipBLASLt projection."""
    device = _gpu()
    case = synthetic((33000,), (16616,), device)
    monkeypatch.setenv("PYHIP_QSA_INDEXER", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    state = plugin._State()
    original = type(case.module).forward_cuda
    with _production_rope(), torch.no_grad():
        for _ in range(2):
            state.indexer(original, case.module, case.hidden, case.positions, _batch(case),
                          _metadata(case, _pool(case.state, case.layer)))
    torch.cuda.synchronize()
    (report,) = state.indexer_checks
    print(report)
    assert report["key_state_equal"] and report["rope_state_equal"] and report["compressed_equal"]
    assert report["worst_relative_boundary_violation"] <= 1e-5 and state.indexer_gemm is True


# (compressed lengths per decode row, page-table width in 16-key pages); 4096 pages is the
# server's graph width (context 262144). Length 0 rows are CUDA-graph padding rows.
DECODE_CASES = (
    ((0,), 4), ((1,), 4), ((17,), 4), ((511,), 64), ((512,), 64), ((513,), 64), ((3000,), 4096),
    ((5, 0, 777), 64), ((513, 2048, 1, 4096), 256), ((16384,) * 8, 1024),
    (tuple(3000 + 37 * i for i in range(32)), 4096), ((65536, 65535), 4096),
)


def decode_case(lengths, pages, device, heads=4, seed=5, pool=None):
    """Random compressed pool of ``pool`` pages, shuffled per-row page tables (stale ids past each length)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, total = len(lengths), pool or sum(-(-n // 16) for n in lengths) + 3
    assert total >= sum(-(-n // 16) for n in lengths)
    cache = torch.randn((total * 16, 1, 128), generator=generator, device=device).to(torch.bfloat16)
    order = torch.randperm(total, generator=generator, device=device).to(torch.int32)
    table = order[torch.randint(0, total, (rows, pages), generator=generator, device=device)]
    used = 0
    for row, length in enumerate(lengths):
        count = -(-length // 16)
        table[row, :count] = order[used:used + count]
        used += count
    q = torch.randn((rows, heads, 128), generator=generator, device=device).to(torch.bfloat16)
    q[:, 4:] = 0
    compressed = torch.tensor(lengths, dtype=torch.int32, device=device)
    sequences = compressed * 4 + torch.randint(0, 4, (rows,), generator=generator, device=device).to(torch.int32)
    return SimpleNamespace(q=q, cache=cache.view(-1, 16, 1, 128), table=table, lengths=compressed,
                           width=pages * 16, positions=sequences - 1, sequences=sequences,
                           module=SimpleNamespace(index_n_heads=4, index_head_dim=128, compress_ratio=4,
                                                  block_topk=TOPK, token_topk=2048, layer_id=3))


def _decode_args(case):
    return (case.module, case.q, case.cache, case.table, case.lengths, case.width, case.positions, case.sequences)


def decode_base(case):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    return QSAIndexer.select_decode_tokens(*_decode_args(case))


def decode_pyhip(case, state=None):
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    return (state or plugin._State()).decode(QSAIndexer.select_decode_tokens, *_decode_args(case))


def check_decode(case, actual=None, expected=None):
    """Paged logits vs FP64 and token sets vs SGLang's decode path (tie-aware); actual defaults to an eager call."""
    from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices

    from . import indexer_decode

    assert plugin._decode_eligible(*_decode_args(case)[:6])
    rows = case.q.shape[0]
    slots = (case.table.long()[:, :, None] * 16 + torch.arange(16, device=case.q.device)).flatten(1)
    exact = torch.relu(torch.einsum("rhd,rnd->rnh", case.q[:, :4].double(), case.cache.view(-1, 128)[slots].double()))
    exact = exact.sum(-1) / math.sqrt(128)
    logits = torch.empty((rows, case.width), dtype=torch.float32, device=case.q.device)
    indexer_decode.launch(case.q, case.cache, case.table, case.lengths, logits,
                          float(torch.tensor(1.0) / torch.tensor(math.sqrt(128))))
    expected = decode_base(case) if expected is None else expected
    actual = decode_pyhip(case) if actual is None else actual
    torch.cuda.synchronize()
    assert actual.shape == expected.shape == (rows, WIDTH) and actual.dtype == torch.int32
    counts = case.lengths.long()
    blocks = _blocks(actual, counts)
    torch.testing.assert_close(torch_expand_qsa_block_indices(blocks, case.positions, case.sequences, RATIO, 2048),
                               actual, rtol=0, atol=0)
    different = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1)
    report = dict(lengths=[int(n) for n in counts[:4]], rows=rows, width=case.width, heads=case.q.shape[1],
                  different_token_sets=int(different.sum()), max_logit_error=0.0, worst_boundary_violation=0.0)
    for row in range(rows):
        count = int(counts[row])
        tail = logits[row, count:-(-count // 16) * 16]
        assert bool(torch.isneginf(tail).all()), "partial-page keys past the length must be -inf"
        if count == 0:
            continue
        values, scale = exact[row, :count], max(float(exact[row, :count].abs().max()), 1e-30)
        report["max_logit_error"] = max(report["max_logit_error"],
                                        float((logits[row, :count].double() - values).abs().max()) / scale)
        if different[row]:
            chosen = torch.zeros(count, dtype=torch.bool, device=values.device)
            chosen[blocks[row][blocks[row] >= 0].long()] = True
            gap = float(values[~chosen].max() - values[chosen].min())
            report["worst_boundary_violation"] = max(report["worst_boundary_violation"], gap / scale)
    assert report["max_logit_error"] <= 1e-6 and report["worst_boundary_violation"] <= 1e-5, report
    return report


@pytest.mark.parametrize("heads", (4, 8))
@pytest.mark.parametrize("lengths,pages", DECODE_CASES,
                         ids=[f"n{'-'.join(map(str, n[:4]))}{'x%d' % len(n) if len(n) > 4 else ''}_p{p}"
                              for n, p in DECODE_CASES])
def test_decode(lengths, pages, heads):
    print(check_decode(decode_case(lengths, pages, _gpu(), heads=heads)))


def test_decode_graph_replay():
    """One capture serves every later length/page table written into its static buffers."""
    device = _gpu()
    case = decode_case((3000, 0, 700, 16384), 1024, device, pool=2100)
    state = plugin._State()
    decode_pyhip(case, state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode_pyhip(case, state)
    for lengths in ((3000, 0, 700, 16384), (1, 512, 513, 9999), (16384, 16383, 0, 4)):
        fresh = decode_case(lengths, 1024, device, seed=sum(lengths), pool=2100)
        for name in ("q", "cache", "table", "lengths", "positions", "sequences"):
            getattr(case, name).copy_(getattr(fresh, name))
        graph.replay()
        # Exact-zero ReLU ties (and near-zero sign flips) may change which tied block fast_topk keeps.
        print(check_decode(case, output.clone()))


def test_decode_ineligible_calls_keep_sglang():
    device = _gpu()
    case = decode_case((600,), 64, device)
    args = _decode_args(case)[:6]
    assert plugin._decode_eligible(*args)
    padded = torch.zeros((1, 8, 128), dtype=torch.bfloat16, device=device)
    changes = [dict(q=case.q.float()), dict(q=case.q[:, :, :64]), dict(q=case.q.repeat(1, 3, 1)[:, :6]),
               dict(q=padded[:, ::2]), dict(cache=case.cache.view(-1, 8, 1, 128)), dict(table=case.table.long()),
               dict(lengths=case.lengths.long()), dict(width=case.width - 16),
               dict(module=SimpleNamespace(**{**vars(case.module), "block_topk": 2048}))]
    names = ("module", "q", "cache", "table", "lengths", "width")
    for change in changes:
        values = dict(zip(names, args), **change)
        assert not plugin._decode_eligible(*(values[n] for n in names)), change.keys()


def decode_forward_case(lengths, device, *, padding=0, seed=17, capture=None, context=65536):
    """CUDA-graph decode rows over page-aligned request slots; graph metadata from SGLang's own refresh.

    Request r (1..R) owns raw slots [64 * (1 + (r - 1) * P), ...) of a P-page row; padding rows use
    SGLang's graph fill (request 0, length 1) and only touch the dump ring rows 0..3 / slot 0.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    requests, rows, pages = len(lengths), len(lengths) + padding, context // 64
    table = torch.zeros((requests + 1, context), dtype=torch.int32, device=device)
    for request in range(1, requests + 1):
        table[request] = 64 * (1 + (request - 1) * pages) + torch.arange(context, dtype=torch.int32, device=device)
    slots = -(-(64 * (1 + requests * pages) // RATIO) // 16) * 16
    if capture is None:
        weights = dict(index_qk_weight=torch.randn((640, 2560), generator=generator, device=device) / 50,
                       q_norm_weight=torch.randn(128, generator=generator, device=device) / 4,
                       k_norm_weight=torch.randn(128, generator=generator, device=device) / 4)
        cache = torch.randn((context + 64, 64), generator=generator, device=device).to(torch.bfloat16)
        module, name = _module(weights, cache, 3, device), "synthetic"
    else:
        module, name = load(capture, device).module, capture.stem
    sequences = torch.tensor(list(lengths) + [1] * padding, dtype=torch.int32, device=device)
    buffers = dict(sequence_lengths=sequences, token_to_batch_idx=torch.arange(rows, dtype=torch.int32, device=device),
                   token_slot_table=torch.zeros((rows, 1), dtype=torch.int32, device=device),
                   out_cache_loc=torch.zeros(rows, dtype=torch.int64, device=device),
                   req_pool_indices=torch.tensor(list(range(1, requests + 1)) + [0] * padding, dtype=torch.int32,
                                                 device=device),
                   graph_write_locs=torch.zeros(rows, dtype=torch.int32, device=device),
                   graph_compressed_page_table=torch.zeros((rows, pages), dtype=torch.int32, device=device),
                   graph_compressed_lengths=torch.zeros(rows, dtype=torch.int32, device=device),
                   graph_prefix_lengths=(sequences - 1).clamp_min(0),
                   decode_logical_positions=torch.zeros(rows, dtype=torch.int32, device=device),
                   pending_ring_slots=torch.zeros(rows, dtype=torch.int64, device=device),
                   graph_ring_group_locs=torch.zeros((rows, RATIO), dtype=torch.int32, device=device))
    case = SimpleNamespace(name=f"{name}_n{'-'.join(map(str, lengths))}_pad{padding}", layer=module.layer_id,
                           module=module, table=table, buffers=buffers, requests=requests, rows=rows,
                           state=_state(device, (requests + 1) * RATIO, slots, seed),
                           hidden=torch.empty((rows, 2560), dtype=torch.bfloat16, device=device),
                           positions=torch.empty((3, rows), dtype=torch.int64, device=device), generator=generator)
    # Ring slot p % 4 of request r last held position p; compression RoPE reads member 0's coordinates.
    rope = case.state["rope_state"]
    rope.zero_()
    for request, length in enumerate(lengths, start=1):
        for position in range(max(0, length - 1 - RATIO), length - 1):
            rope[request * RATIO + position % RATIO] = torch.tensor([position, position + 3, position + 7])
    decode_step(case, lengths)
    return case


def decode_step(case, lengths):
    """Write this step's lengths, hidden states and 3-axis positions into the static graph inputs."""
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    device = case.hidden.device
    case.buffers["sequence_lengths"][:case.requests].copy_(torch.tensor(lengths, dtype=torch.int32))
    case.buffers["graph_prefix_lengths"].copy_((case.buffers["sequence_lengths"] - 1).clamp_min(0))
    QwenSparseAttnBackend._update_qsa_cuda_graph_metadata(SimpleNamespace(req_to_token=case.table),
                                                          _decode_metadata(case, _pool(case.state, case.layer)),
                                                          case.buffers["req_pool_indices"])
    case.hidden.copy_(torch.randn(case.hidden.shape, generator=case.generator, device=device))
    logical = case.buffers["decode_logical_positions"].long()
    case.positions.copy_(torch.stack([logical, logical + 3, logical + 7]))


def _decode_metadata(case, pool):
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata

    return QSAIndexerMetadata(token_to_kv_pool=pool, compress_ratio=RATIO, block_topk=TOPK, is_cuda_graph=True,
                              **case.buffers)


def _decode_batch(case):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(forward_mode=ForwardMode.DECODE, positions=None, extend_seq_lens_cpu=None,
                           seq_lens_cpu=case.buffers["sequence_lengths"].cpu())


def check_decode_forward(case, base_pool, new_pool, actual, q, expected):
    """Bit-exact q/ring/compressed writes vs SGLang (dump ring rows 0..3 and slot 0 excluded: padding and
    non-boundary rows race there in both), then tie-aware selections."""
    rows = case.rows
    with _production_rope(), torch.no_grad():
        q_base = case.module.project_qk(case.hidden[:rows], case.positions[:, :rows])[0]
    torch.cuda.synchronize()
    assert torch.equal(q, q_base), "normed/rotated decode Q must be bit-exact"
    report = dict(case=case.name)
    for name, lo in (("key_state", RATIO), ("rope_state", RATIO), ("compressed", 1)):
        old, a, b = case.state[name][lo:], _buffer(new_pool, name)[lo:], _buffer(base_pool, name)[lo:]
        assert torch.equal(a, b), name
        report[f"{name}_rows_written"] = int((b != old).flatten(1).any(1).sum())
    meta = _decode_metadata(case, new_pool)
    cache, table, lengths, width = meta.get_decode_mqa_inputs(case.layer)
    view = SimpleNamespace(q=q, cache=cache, table=table, lengths=lengths, width=width, module=case.module,
                           positions=meta.decode_logical_positions, sequences=meta.get_seqlens_int32())
    report["selection"] = check_decode(view, actual, expected)
    return report


DECODE_FORWARD_CASES = (((1,), 0), ((4,), 0), ((5, 8, 3, 12), 1), ((12000, 11888, 11667, 11851), 0),
                        ((4001, 6, 1022, 2), 2), (tuple(3000 + 37 * i for i in range(29)), 3), ((65536,), 0))


@pytest.mark.parametrize("lengths,padding", DECODE_FORWARD_CASES,
                         ids=[f"n{'-'.join(map(str, n[:4]))}{'x%d' % len(n) if len(n) > 4 else ''}_pad{p}"
                              for n, p in DECODE_FORWARD_CASES])
def test_decode_forward(lengths, padding):
    device = _gpu()
    case = decode_forward_case(lengths, device, padding=padding, context=max(4096, -(-max(lengths) // 64) * 64 + 64))
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    batch = _decode_batch(case)
    with _production_rope(), torch.no_grad():
        expected = case.module.forward_cuda(case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
        inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                               _decode_metadata(case, new_pool))
        assert inputs is not None, "adapter rejected an eligible graph decode call"
        actual, q, _ = indexer._decode_forward(inputs.pop("qk"), **inputs)
    print(check_decode_forward(case, base_pool, new_pool, actual, q, expected))


@pytest.mark.parametrize("path", _real_files()[:2], ids=lambda p: p.stem)
def test_decode_forward_real_weights(path):
    device = _gpu()
    case = decode_forward_case((12000, 11888, 4, 7), device, padding=1, capture=path, context=16384)
    base_pool, new_pool = _pool(case.state, case.layer), _pool(case.state, case.layer)
    batch = _decode_batch(case)
    with _production_rope(), torch.no_grad():
        expected = case.module.forward_cuda(case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
        inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                               _decode_metadata(case, new_pool))
        actual, q, _ = indexer._decode_forward(inputs.pop("qk"), **inputs)
    print(check_decode_forward(case, base_pool, new_pool, actual, q, expected))


def test_decode_forward_graph_steps(monkeypatch):
    """Capture the hooked decode forward once, then replay 6 steps (crossing group boundaries) against
    SGLang's eager graph-metadata path on an identical pool history."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    lengths = [4001, 6, 1022, 2]
    case = decode_forward_case(tuple(lengths), device, padding=2, context=8192)
    base_pool, new_pool, warm_pool = (_pool(case.state, case.layer) for _ in range(3))
    state, batch, original = plugin._State(), _decode_batch(case), QSAIndexer.forward_cuda
    with _production_rope(), torch.no_grad():
        state.indexer(original, case.module, case.hidden, case.positions, batch, _decode_metadata(case, warm_pool))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = state.indexer(original, case.module, case.hidden, case.positions, batch,
                                   _decode_metadata(case, new_pool))
        for step in range(6):
            if step:
                lengths = [n + 1 for n in lengths]
                decode_step(case, lengths)
            graph.replay()
            actual = output.clone()
            expected = original(case.module, case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
            inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                                   _decode_metadata(case, warm_pool))
            q = indexer._decode_forward(inputs.pop("qk"), **inputs)[1]
            report = check_decode_forward(case, base_pool, new_pool, actual, q, expected)
            print(step, lengths, report)
            case.state = {name: _buffer(base_pool, name).clone() for name in ("key_state", "rope_state", "compressed")}


def test_decode_validation_graph(monkeypatch):
    """PYHIP_QSA_VALIDATE=1 captures SGLang's decode forward and the PyHIP one together; device counters
    cover every replay (6 steps crossing group boundaries) and the pool history still matches SGLang."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    lengths = [4001, 6, 1022, 2]
    case = decode_forward_case(tuple(lengths), device, padding=2, context=8192)
    base_pool, new_pool, warm_pool = (_pool(case.state, case.layer) for _ in range(3))
    state, batch, original = plugin._State(), _decode_batch(case), QSAIndexer.forward_cuda
    with _production_rope(), torch.no_grad():
        state.indexer(original, case.module, case.hidden, case.positions, batch, _decode_metadata(case, warm_pool))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph):
            output = state.indexer(original, case.module, case.hidden, case.positions, batch,
                                   _decode_metadata(case, new_pool))
        for step in range(6):
            if step:
                lengths = [n + 1 for n in lengths]
                decode_step(case, lengths)
            graph.replay()
            actual = output.clone()
            expected = original(case.module, case.hidden, case.positions, batch, _decode_metadata(case, base_pool))
            inputs = plugin._decode_forward_inputs(case.module, case.hidden, case.positions, batch,
                                                   _decode_metadata(case, warm_pool))
            q = indexer._decode_forward(inputs.pop("qk"), **inputs)[1]
            print(step, lengths, check_decode_forward(case, base_pool, new_pool, actual, q, expected))
            case.state = {name: _buffer(base_pool, name).clone() for name in ("key_state", "rope_state", "compressed")}
    counts = state.decode_summary()["forward:3"]
    print(counts)
    # 1 eager warmup + 6 replays, 4 real rows each; boundaries at lengths 8/1024/4 (step 2) and 4004 (step 3).
    assert (counts["calls"], counts["rows"], counts["compressed_rows"]) == (7, 28, 4)
    assert not any(counts[name] for name in plugin._DECODE_FAILURES)
    state.check_decode()


@pytest.mark.parametrize("fault", ("q", "compressed", "selection"))
def test_decode_validation_detects_faults(monkeypatch, fault):
    """Each class of PyHIP decode error is counted on device and fails the next prefill's host check."""
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_INDEXER_DECODE", "1")
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    case = decode_forward_case((4003, 1024, 2051), device, padding=1, context=8192)
    decode_forward = indexer._decode_forward

    def faulty(qk, **inputs):
        tokens, q, logits = decode_forward(qk, **inputs)
        if fault == "q":
            q = q.clone()
            q[0, 0, 0] += 1
        elif fault == "compressed":
            locs = inputs["write_locs"].long()
            inputs["compressed"][locs[locs != 0]] += 1
        else:
            # Swap row 0's first chosen block for its lowest-scoring unchosen block.
            chosen = set((tokens[0, :2048:4] // 4).tolist()) - {-1}
            low = min((b for b in range(int(inputs["lengths"][0])) if b not in chosen), key=lambda b: float(logits[0, b]))
            tokens = tokens.clone()
            tokens[0, :4] = torch.arange(4 * low, 4 * low + 4, dtype=tokens.dtype, device=tokens.device)
        return tokens, q, logits

    monkeypatch.setattr(indexer, "_decode_forward", faulty)
    state = plugin._State()
    with _production_rope(), torch.no_grad():
        state.indexer(QSAIndexer.forward_cuda, case.module, case.hidden, case.positions, _decode_batch(case),
                      _decode_metadata(case, _pool(case.state, case.layer)))
    counts = state.decode_summary()["forward:3"]
    print(fault, counts)
    field = dict(q="q_mismatch", compressed="compressed_mismatch", selection="selection_violations")[fault]
    assert counts[field] == 1 and counts["rows"] == 3 and counts["compressed_rows"] == 1
    assert not any(counts[name] for name in plugin._DECODE_FAILURES if name != field)
    prefill = synthetic((2051,), (2051,), device)
    with pytest.raises(AssertionError, match="decode differs from SGLang"), _production_rope(), torch.no_grad():
        state.indexer(type(prefill.module).forward_cuda, prefill.module, prefill.hidden, prefill.positions,
                      _batch(prefill), _metadata(prefill, _pool(prefill.state, prefill.layer)))


def test_decode_select_validation(monkeypatch):
    """The select-only hook validates against SGLang's selection too; the forward's reference call bypasses it."""
    device = _gpu()
    monkeypatch.setenv("PYHIP_QSA_VALIDATE", "1")
    case = decode_case((3000, 513, 17, 0), 256, device)
    state = plugin._State()
    print(check_decode(case, decode_pyhip(case, state)))
    counts = state.decode_summary()["select:3"]
    assert counts["calls"] == 1 and counts["rows"] == 4 and not any(counts[n] for n in plugin._DECODE_FAILURES)
    # Inside the validated forward's reference call the hook hands SGLang's result back unvalidated. (SGLang's own
    # selection is not repeatable on near-ties, so its set is not compared here.)
    state.decode_reference = True
    assert state.decode(lambda *args: "sglang", *_decode_args(case)) == "sglang"
    assert state.decode_summary()["select:3"]["calls"] == 1
    state.decode_reference = False
    decode_select = indexer._decode_select

    def reversed_ranking(*args):
        tokens, logits = decode_select(*args)
        return tokens, -logits

    monkeypatch.setattr(indexer, "_decode_select", reversed_ranking)
    decode_pyhip(case, state)
    counts = state.decode_summary()["select:3"]
    assert counts["calls"] == 2 and counts["selection_violations"] == 2
    with pytest.raises(AssertionError, match="decode differs from SGLang"):
        state.check_decode()


def _gate(folder, phase, gpu):
    from tests.ops.gr_read.test_gr_read import read_hardware, validate_hardware

    # amd-smi's use% trails recent activity; let this process's own warmup/timing drain first.
    torch.cuda.synchronize(gpu)
    time.sleep(GATE_SETTLE_SECONDS)
    snapshot = read_hardware(gpu, Path("/opt/rocm-7.14/bin/amd-smi"))
    props = torch.cuda.get_device_properties(gpu)
    pci = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    snapshot["runtime_pci"] = pci
    (folder / f"hardware_{phase}.json").write_text(json.dumps(snapshot, indent=2))
    validate_hardware(snapshot)
    assert pci.lower() == snapshot["card"]["PCI Bus"].lower()


def benchmark(case, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    """Time full ``forward_cuda`` (incl. index_qk_proj) for base and both PyHIP projections, AB/BA."""
    from pyhip.testing.misc import cudaPerf

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / "indexer.py", here / "indexer_logits.py", here / "indexer_topk.py", here / "test_indexer.py",
               here / "sglang/plugin.py", ROOT / "src/pyhip/testing/misc.py", ROOT / "tests/ops/gr_read/test_gr_read.py"]
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "pyhip_exact", "pyhip")
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, capture=case.capture,
                  gate_settle_seconds=GATE_SETTLE_SECONDS,
                  scope="base: SGLang QSAIndexer.forward_cuda; pyhip_exact: plugin adapter + indexer.py with "
                        "SGLang's index_qk_proj; pyhip: same with the plugin's default hipBLASLt projection; "
                        "fresh pool copy per buffer; outputs allocated by each implementation",
                  seq_lens=case.seq_lens, extend_lens=case.extend_lens, torch=torch.__version__,
                  hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    try:
        _gate(folder, "before", gpu)
        report["check"] = check(case)
        report["check_projection"] = check_projection(case)
        runs = dict(base=lambda value, pool: base(value, pool), pyhip_exact=lambda value, pool: pyhip(value, pool),
                    pyhip=lambda value, pool: pyhip(value, pool, projection=fast_projection))
        values = []
        for index in range(buffers):
            copy = SimpleNamespace(**vars(case))
            copy.hidden = case.hidden.clone()
            values.append((copy, {name: _pool(case.state, case.layer) for name in arms}))
        expected = {}
        for value, pools in values:
            for name in arms:
                result = runs[name](value, pools[name])
                if name != "base":
                    expected.setdefault(name, result)
                    assert torch.equal(result, expected[name])
            for _ in range(warmup):
                for name in arms:
                    runs[name](value, pools[name])
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            value, pools = values[index]
            for name in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    result = runs[name](value, pools[name])
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=name, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                if name != "base":
                    assert torch.equal(result, expected[name])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == name]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == name and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][name] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                           min_us=min(us), max_us=max(us),
                                           ratio_to_base=statistics.median(us) / base_us,
                                           paired_ratio_median=statistics.median(paired))
        assert hashes() == report["source_sha256"], "Source changed during measurement"
        for source in sources:
            target = folder / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _gate(folder, "after", gpu)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


@pytest.mark.perf
@pytest.mark.parametrize("path", _real_files(), ids=lambda p: p.stem)
def test_indexer_performance(path):
    device = _gpu()
    output = Path(os.environ["QSA_REPLAY_OUTPUT"])
    assert output.resolve().is_relative_to(DATA.resolve())
    assert benchmark(load(path, device), output / path.stem, device.index)["complete"]


# Formal decode shapes: decode rows (graph batch) x compressed keys per row, 4096-page (262144-token) tables.
DECODE_BENCH = tuple((rows, keys) for rows in (1, 8, 32) for keys in (3000, 16384, 65536))


def decode_benchmark(rows, keys, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2, samples=BENCHMARK_SAMPLES):
    """Time CUDA-graph replays of ``select_decode_tokens``: SGLang vs the PyHIP hook, AB/BA."""
    from pyhip.testing.misc import cudaPerf

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / "indexer.py", here / "indexer_decode.py", here / "indexer_logits.py", here / "indexer_topk.py",
               here / "test_indexer.py", here / "sglang/plugin.py", ROOT / "src/pyhip/testing/misc.py",
               ROOT / "tests/ops/gr_read/test_gr_read.py"]
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "pyhip")
    device = torch.device("cuda", gpu)
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, rows=rows, keys=keys,
                  table_pages=4096, gate_settle_seconds=GATE_SETTLE_SECONDS,
                  scope="one CUDA-graph replay of QSAIndexer.select_decode_tokens per arm: base = SGLang (Torch "
                        "paged MQA fallback over the full table width, fast_topk, Triton expand); pyhip = plugin "
                        "hook (FlyDSL paged logits over each row's length, same fast_topk and expand); one graph "
                        "per arm and buffer, independent random pools/tables/queries per buffer",
                  torch=torch.__version__, hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    try:
        _gate(folder, "before", gpu)
        cases = [decode_case((keys,) * rows, 4096, device, seed=1000 * index + rows) for index in range(buffers)]
        report["check"] = check_decode(cases[0])
        runs = dict(base=decode_base, pyhip=decode_pyhip)
        graphs, outputs, expected = {}, {}, {}
        for index, case in enumerate(cases):
            for name in arms:
                for _ in range(1 + warmup):
                    runs[name](case)
                torch.cuda.synchronize(gpu)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name, index] = runs[name](case)
                graph.replay()
                graphs[name, index] = graph
            torch.cuda.synchronize(gpu)
            for name in arms:
                expected[name, index] = outputs[name, index].sort(dim=1).values.clone()
            report.setdefault("base_vs_pyhip_different_token_set_rows", []).append(
                int((expected["base", index] != expected["pyhip", index]).any(dim=1).sum()))
        for _ in range(warmup):
            for key in graphs:
                graphs[key].replay()
        torch.cuda.synchronize(gpu)
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer_decode", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            for name in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    graphs[name, index].replay()
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=name, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                assert torch.equal(outputs[name, index].sort(dim=1).values, expected[name, index])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for name in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == name]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == name and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][name] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                           min_us=min(us), max_us=max(us),
                                           ratio_to_base=statistics.median(us) / base_us,
                                           paired_ratio_median=statistics.median(paired))
        assert hashes() == report["source_sha256"], "Source changed during measurement"
        for source in sources:
            target = folder / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _gate(folder, "after", gpu)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


# Formal decode-forward shapes: graph rows x sequence length (tokens); tables are 4096 pages wide like the
# 262144-token server graphs.
DECODE_FORWARD_BENCH = tuple((rows, length) for rows in (1, 8, 32) for length in (12000, 65536, 262144))


def decode_forward_benchmark(rows, length, folder, gpu, *, buffers=BENCHMARK_BUFFERS, warmup=2,
                             samples=BENCHMARK_SAMPLES):
    """Time CUDA-graph replays of one decode ``QSAIndexer.forward_cuda`` (hidden states in, token
    selections out): SGLang, SGLang with the stage-1 selection hook, and the PyHIP decode forward."""
    from unittest import mock

    from pyhip.testing.misc import cudaPerf
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    if buffers < 1 or samples < buffers or warmup < 0:
        raise ValueError("Require samples >= buffers >= 1 and warmup >= 0")
    if not folder.resolve().is_relative_to(DATA.resolve()):
        raise ValueError("Results must stay under mytest/mydata")
    folder.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / "indexer.py", here / "indexer_decode.py", here / "indexer_logits.py", here / "indexer_topk.py",
               here / "test_indexer.py", here / "sglang/plugin.py", ROOT / "src/pyhip/testing/misc.py",
               ROOT / "tests/ops/gr_read/test_gr_read.py"]
    hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arms = ("base", "select", "forward")
    device = torch.device("cuda", gpu)
    report = dict(complete=False, raw=[], buffers=buffers, warmup=warmup, samples=samples, rows=rows, length=length,
                  table_pages=4096, gate_settle_seconds=GATE_SETTLE_SECONDS,
                  scope="one CUDA-graph replay of a decode QSAIndexer.forward_cuda per arm (index_qk_proj GEMM, q "
                        "norm/RoPE, pending-ring store, fixed-shape compression, MQA, top-k, expand): base = SGLang "
                        "(unfused prep, Torch MQA fallback over the table width); select = SGLang with the stage-1 "
                        "select_decode_tokens hook; forward = PyHIP decode forward (1 prep kernel + stage-1 "
                        "selection). Graph metadata from SGLang's own refresh; one graph and KV pool per arm/buffer; "
                        "all graphs share one CUDA-graph memory pool like SGLang's runner, so every output is "
                        "checked right after its own replay",
                  torch=torch.__version__, hip=torch.version.hip, source_sha256=hashes(), gpu=gpu,
                  tp_note="indexer heads/weights are replicated: TP2/4/8 ranks run this identical workload",
                  packages={n: importlib.metadata.version(n) for n in ("triton", "flydsl", "sglang", "amd-aiter")})
    original_select = QSAIndexer.select_decode_tokens

    def hooked_select(module, *args):
        return plugin._around_decode(original_select, module, *args)

    try:
        _gate(folder, "before", gpu)
        # Graph replays read these inputs/pools by address: keep every buffer alive until timing ends.
        state, graphs, outputs, expected, checks, keep = plugin._State(), {}, {}, {}, [], []
        graph_pool = torch.cuda.graph_pool_handle()
        for index in range(buffers):
            case = decode_forward_case((length,) * rows, device, seed=1000 * index + rows, context=262144)
            batch, pools = _decode_batch(case), {arm: _pool(case.state, case.layer) for arm in arms}
            case.state, snapshots = None, {}
            runs = dict(
                base=lambda meta: QSAIndexer.forward_cuda(case.module, case.hidden, case.positions, batch, meta),
                select=lambda meta: QSAIndexer.forward_cuda(case.module, case.hidden, case.positions, batch, meta),
                forward=lambda meta: state.indexer(QSAIndexer.forward_cuda, case.module, case.hidden, case.positions,
                                                   batch, meta))
            for arm in arms:
                meta = _decode_metadata(case, pools[arm])
                with (mock.patch.object(QSAIndexer, "select_decode_tokens", hooked_select) if arm == "select"
                      else contextlib.nullcontext()), \
                        mock.patch.dict(os.environ, {"PYHIP_QSA_INDEXER_DECODE": "1" if arm == "forward" else "0"}), \
                        _production_rope(), torch.no_grad():
                    for _ in range(1 + warmup):
                        runs[arm](meta)
                    torch.cuda.synchronize(gpu)
                    graph = torch.cuda.CUDAGraph()
                    with model_capture_mode(), torch.cuda.graph(graph, pool=graph_pool):
                        outputs[arm, index] = runs[arm](meta)
                graph.replay()
                graphs[arm, index] = graph
                snapshots[arm] = outputs[arm, index].clone()
                expected[arm, index] = snapshots[arm].sort(dim=1).values
            torch.cuda.synchronize(gpu)
            with _production_rope(), torch.no_grad():
                q = case.module.project_qk(case.hidden[:case.rows], case.positions[:, :case.rows])[0]
            meta = _decode_metadata(case, pools["forward"])
            cache, table, lengths, width = meta.get_decode_mqa_inputs(case.layer)
            view = SimpleNamespace(q=q, cache=cache, table=table, lengths=lengths, width=width, module=case.module,
                                   positions=meta.decode_logical_positions, sequences=meta.get_seqlens_int32())
            check = dict(buffer=index, select=check_decode(view, snapshots["select"], snapshots["base"]),
                         forward=check_decode(view, snapshots["forward"], snapshots["base"]))
            for name, lo in (("key_state", RATIO), ("rope_state", RATIO), ("compressed", 1)):
                check[f"{name}_equal"] = all(torch.equal(_buffer(pools[arm], name)[lo:], _buffer(pools["base"], name)[lo:])
                                             for arm in arms)
                assert check[f"{name}_equal"], (index, name)
            checks.append(check)
            keep.append((case, batch, pools))
        report["checks"] = checks
        for _ in range(warmup):
            for key in graphs:
                graphs[key].replay()
        torch.cuda.synchronize(gpu)
        # Return this process's cached eager/setup blocks; the VRAM gate targets other tenants.
        torch.cuda.empty_cache()
        _gate(folder, "before_samples", gpu)
        timer = cudaPerf(name="indexer_decode_forward", verbose=0)
        if not timer.enable:
            raise RuntimeError("CUDAPERF disabled timing")
        for sample in range(samples):
            index = sample % buffers
            for arm in (arms if sample % 2 == 0 else arms[::-1]):
                with timer:
                    graphs[arm, index].replay()
                elapsed = timer.latencies[-1] * 1e6
                report["raw"].append(dict(scope=arm, sample=sample, buffer=index, us=elapsed))
                assert math.isfinite(elapsed) and elapsed > 0
                assert torch.equal(outputs[arm, index].sort(dim=1).values, expected[arm, index])
        report["summary"] = {}
        base_us = statistics.median(r["us"] for r in report["raw"] if r["scope"] == "base")
        for arm in arms:
            us = [r["us"] for r in report["raw"] if r["scope"] == arm]
            paired = [next(r["us"] for r in report["raw"] if r["scope"] == arm and r["sample"] == i)
                      / next(r["us"] for r in report["raw"] if r["scope"] == "base" and r["sample"] == i)
                      for i in range(samples)]
            report["summary"][arm] = dict(median_us=statistics.median(us), mean_us=statistics.fmean(us),
                                          min_us=min(us), max_us=max(us),
                                          ratio_to_base=statistics.median(us) / base_us,
                                          paired_ratio_median=statistics.median(paired))
        assert hashes() == report["source_sha256"], "Source changed during measurement"
        for source in sources:
            target = folder / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _gate(folder, "after", gpu)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inputs", nargs="*", type=Path)
    parser.add_argument("--buffers", type=int, default=BENCHMARK_BUFFERS)
    parser.add_argument("--samples", type=int, default=BENCHMARK_SAMPLES)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--decode", action="store_true", help="formal decode matrix (DECODE_BENCH) instead of prefill")
    parser.add_argument("--decode-forward", action="store_true",
                        help="formal decode forward_cuda matrix (DECODE_FORWARD_BENCH) instead of prefill")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.buffers < 1 or args.samples < args.buffers:
        parser.error("Require samples >= buffers >= 1")
    if any(os.environ.get(n) for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                                        "HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")):
        raise RuntimeError("Use unmasked physical GPU indices")
    assert args.output.resolve().is_relative_to(DATA.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    if args.decode or args.decode_forward:
        results = {}
        shapes = DECODE_BENCH if args.decode else DECODE_FORWARD_BENCH
        for rows, size in shapes:
            name = f"decode_r{rows}_k{size}" if args.decode else f"decode_forward_r{rows}_n{size}"
            run = decode_benchmark if args.decode else decode_forward_benchmark
            results[name] = run(rows, size, args.output / name, args.gpu, buffers=args.buffers,
                                samples=args.samples)["summary"]
            print(name, results[name], flush=True)
            torch.cuda.empty_cache()
        (args.output / "checks.json").write_text(json.dumps(dict(shapes=shapes, results=results,
                                                                 buffers=args.buffers, samples=args.samples),
                                                            indent=2))
        return
    paths = args.inputs if args.inputs is not None else _real_files()
    results = {}
    for path in paths:
        case = load(path, device)
        result = check(case) if args.check_only else benchmark(case, args.output / path.stem, args.gpu,
                                                               buffers=args.buffers, samples=args.samples)
        results[path.stem] = result if args.check_only else result["summary"]
        print(path.stem, results[path.stem], flush=True)
        del case
    (args.output / "checks.json").write_text(json.dumps(dict(inputs=[str(p) for p in paths], results=results,
                                                             buffers=args.buffers, samples=args.samples,
                                                             check_only=args.check_only), indent=2))


if __name__ == "__main__":
    main()
