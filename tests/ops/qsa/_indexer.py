"""Reusable QSA indexer inputs and SGLang/FP64 correctness references."""

import contextlib
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "mytest/mydata"
AITER_CONFIGS = ROOT / "mytest/sglang_tp2_base_20260925_01/aiter_configs_snapshot"
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("AITER_CONFIG_GEMM_BF16", str(AITER_CONFIGS / "bf16_tuned_gemm.csv"))

from pyhip.ops.qsa.flydsl import indexer  # noqa: E402
from experiments.attention.flydsl.qsa.sglang import plugin  # noqa: E402

REAL_INPUTS = DATA / "qsa_indexer_20260928_01/capture/inputs"
BENCHMARK_BUFFERS = 10
BENCHMARK_SAMPLES = 128
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

    from pyhip.ops.qsa.flydsl import indexer_decode

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


# Formal decode shapes: decode rows (graph batch) x compressed keys per row, 4096-page (262144-token) tables.
DECODE_BENCH = tuple((rows, keys) for rows in (1, 8, 32) for keys in (3000, 16384, 65536))


# Formal decode-forward shapes: graph rows x sequence length (tokens); tables are 4096 pages wide like the
# 262144-token server graphs.
DECODE_FORWARD_BENCH = tuple((rows, length) for rows in (1, 8, 32) for length in (12000, 65536, 262144))
