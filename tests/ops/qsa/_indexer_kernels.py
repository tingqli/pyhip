"""Synthetic, independently resettable cases for the actual QSA indexer kernels.

Only ``runs`` launch kernels under test: one Triton handle or FlyDSL launch per
label. SGLang projection/preparation and FP64 references run during case setup,
never in a timed region. No captured tensors or installed plugin hooks are needed.
"""

import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from tests.ops.qsa import _indexer as helpers

indexer = helpers.indexer
plugin = helpers.plugin
HEADS, DIM, RATIO, TOPK, WIDTH = 4, 128, helpers.RATIO, helpers.TOPK, helpers.WIDTH
SCALE = float(np.float32(1.0) / np.float32(math.sqrt(DIM)))
SENTINEL = -123
RESOURCE_FIELDS = ("private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")
PREFILL_LABELS = ("q_prep", "k_compress", "gather_prefix", "indexer_logits", "indexer_topk")
DECODE_LABELS = ("decode_prep", "decode_logits")


def prefill_labels(seq_lens, extend_lens):
    """Omit launches that the production prefill plan itself omits."""
    prefix = any(s - e >= RATIO for s, e in zip(seq_lens, extend_lens))
    logits = any(e > 0 and s // RATIO > TOPK for s, e in zip(seq_lens, extend_lens))
    return tuple(label for label in PREFILL_LABELS
                 if (label != "gather_prefix" or prefix) and (label != "indexer_logits" or logits))


def _case(source, inputs, kind, **metadata):
    return SimpleNamespace(
        device=source.hidden.device, source=source, inputs=inputs, runs={}, reset={}, checks={}, flops={},
        tensors={name: value for name, value in inputs.items() if isinstance(value, torch.Tensor) and value.numel()},
        compiled={}, audited={},
        metadata=dict(source="synthetic; no capture dependence", kind=kind, heads=HEADS, head_dim=DIM,
                      compress_ratio=RATIO, topk=TOPK, token_width=WIDTH,
                      qk_shape=list(inputs["qk"].shape), prep_dtype="bfloat16", logits_dtype="float32",
                      boundary="one prepared runtime kernel; projection/setup/reset/reference/checks excluded",
                      tp="replicated indexer at TP2/4/8; this is not a distributed or service test",
                      pool_exclusions="reserved ring rows 0..3 and compressed slot 0 (racing inert writes)",
                      resources={}, validation={}, **metadata),
    )


def _resources(compiled, symbol, flydsl):
    """Fail closed on the *used* compiled object, including missing resource fields."""
    if flydsl:
        text = compiled._keepalive.ir
        assert re.findall(r'#gpu\.kernel_metadata<"([^"]+)"', text) == [symbol], symbol
        evidence = dict(compiler="FlyDSL", kernel=symbol,
                        compiled_ir_sha256=hashlib.sha256(text.encode()).hexdigest())
        pattern = lambda field: rf"\b{field}\s*=\s*(\d+)"
    else:
        assert compiled.src.fn.fn.__name__ == symbol, (symbol, compiled.name)
        binary = compiled.asm["hsaco"]
        readelf = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/llvm-readelf"
        text = subprocess.run([str(readelf), "--notes", "-"], input=binary,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode()
        names = re.findall(r"(?m)^\s*\.name:\s+['\"]?([\w.$]+)", text)
        # Argument metadata may also carry .name; the three resource fields
        # below must each occur exactly once, so a multi-kernel ELF fails closed.
        assert compiled.name in names, (symbol, names)
        evidence = dict(compiler="Triton", kernel=compiled.name,
                        hsaco_sha256=hashlib.sha256(binary).hexdigest(),
                        # Fixed ELF LDS alone does not include Triton's dynamic LDS.
                        launch_shared_bytes=int(compiled.metadata.shared))
        pattern = lambda field: rf"\.{field}:\s+(\d+)"
    for field in RESOURCE_FIELDS:
        values = re.findall(pattern(field), text)
        assert len(values) == 1 and int(values[0]) == 0, (symbol, field, values)
        evidence[field] = int(values[0])
    return evidence


def _register(case, label, launch, reset, check, symbol, *, flydsl=None):
    def run():
        with torch.cuda.device(case.device):
            compiled = launch()
        case.compiled[label] = compiled if flydsl is None else flydsl._COMPILED[case.device]

    def check_actual():
        # Consume this invocation's output before any other component can overwrite it.
        result = check()
        compiled = case.compiled[label]
        assert compiled is not None, f"{label}: launch returned no compiled artifact"
        if case.audited.get(label) is not compiled:
            case.metadata["resources"][label] = _resources(compiled, symbol, flydsl is not None)
            case.audited[label] = compiled
        case.metadata["validation"][label] = result or {"bitexact": True}

    case.runs[label], case.reset[label], case.checks[label] = run, reset, check_actual


def _reference_prep(source, pool, *, decode=False):
    """Use SGLang's own prep chain, without running its selection/forward wrappers."""
    rows = source.rows if decode else sum(source.extend_lens)
    metadata = helpers._decode_metadata(source, pool) if decode else helpers._metadata(source, pool)
    logical = metadata.decode_logical_positions if decode else source.logical[:rows]
    positions = source.positions[:, :rows]
    with helpers._production_rope():
        q, token_k, stored = source.module.project_qk(
            source.hidden[:rows], positions, pool=pool, cache_loc=metadata.pending_ring_slots,
            q_heads_padded=8 if decode else None,
        )
        source.module.update_key_state_and_compress(
            token_k, logical, positions, metadata,
            state_slots=metadata.pending_ring_slots, state_stored=stored,
        )
    return q, metadata


def _assert_exact(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    # Byte equality also catches signed-zero differences in BF16 preparation.
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), label


def _assert_pool(inputs, expected, names):
    for name in names:
        lo = 1 if name == "compressed" else RATIO
        _assert_exact(inputs[name][lo:], helpers._buffer(expected, name)[lo:], name)


def _assert_guard(tensor):
    assert bool((tensor == SENTINEL).all()), "output guard overwritten"


def _assert_logits(actual, exact, defined):
    # Per-row scale, all defined elements, including noncausal values explicitly
    # produced by a prefill tile. Unwritten padding is not an output contract.
    delta = (actual.double() - exact).abs().masked_fill(~defined, 0)
    scale = exact.abs().masked_fill(~defined, 0).amax(1).clamp_min(1e-30)
    error = delta.amax(1) / scale
    assert bool(torch.isfinite(error).all()), "nonfinite defined logits"
    worst = float(error.max())
    assert worst <= 1e-6, {"max_relative_logit_error": worst}
    return dict(max_relative_logit_error=worst, tolerance=1e-6)


def _assert_topk(actual, positions, lengths, exact, *, equal_ties=False):
    from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices

    rows = positions.numel()
    assert actual.shape == (rows, WIDTH) and actual.dtype == torch.int32
    counts = torch.minimum((positions.long() + 1) // RATIO, lengths.long() // RATIO)
    blocks = helpers._blocks(actual, counts)
    live = torch.arange(TOPK, device=actual.device)[None] < counts.clamp_max(TOPK)[:, None]
    assert torch.equal(blocks >= 0, live), "missing or negative selected block"
    assert bool((blocks < counts[:, None]).all()), "noncausal selected block"
    ordered = torch.where(live, blocks, 1 << 30).sort(dim=1).values
    assert not bool(((ordered[:, 1:] == ordered[:, :-1]) & live[:, 1:]).any()), "duplicate block"
    # Reconstructing from the actual block ids checks every quartet, the 0..3
    # causal tail tokens, and every -1 padding entry, not just the selected sets.
    expanded = torch_expand_qsa_block_indices(blocks, positions, lengths, RATIO, TOPK * RATIO)
    assert torch.equal(actual, expanded), "token quartet/tail/padding ABI"
    if equal_ties:
        expected = torch.arange(TOPK, device=actual.device)[None].expand(rows, -1)
        assert torch.equal(ordered[live], expected[live]), "equal ties must keep lowest block ids"

    columns = torch.arange(exact.shape[1], device=actual.device)[None]
    causal = columns < counts[:, None]
    hits = torch.zeros(exact.shape, dtype=torch.int32, device=actual.device)
    hits.scatter_add_(1, blocks.clamp_min(0).long(), live.int())
    selected = hits > 0
    high = exact.masked_fill(~causal | selected, -math.inf).amax(1)
    low = exact.masked_fill(~selected, math.inf).amin(1)
    scale = exact.abs().masked_fill(~causal, 0).amax(1).clamp_min(1e-30)
    gap = ((high - low) / scale).masked_fill(counts <= TOPK, 0)
    assert bool(torch.isfinite(gap).all()), "nonfinite top-k boundary"
    worst = max(0.0, float(gap.max()))
    assert worst <= 1e-5, {"worst_relative_boundary_violation": worst}
    return dict(worst_relative_boundary_violation=worst, tolerance=1e-5, checked_rows=rows)


def _prefill_scores(q, packed, seq_lens, extend_lens, width):
    exact = torch.full((q.shape[0], width), math.nan, dtype=torch.float64, device=q.device)
    row, base = 0, 0
    for sequence, extend in zip(seq_lens, extend_lens):
        keys = sequence // RATIO
        if extend and keys:
            dots = torch.einsum("rhd,nd->rnh", q[row:row + extend].double(), packed[base:base + keys].double())
            exact[row:row + extend, :keys] = dots.relu().sum(-1) / math.sqrt(DIM)
        row, base = row + extend, base + keys
    return exact


@torch.no_grad()
def make_prefill_case(seq_lens, extend_lens, device, *, seed=41, topk_pattern="projected", row0=0):
    """Small single-chunk prefill: BF16 [M,640] -> Q/pool, packed K, logits, token ids.

    Prefixes must be group-aligned, like SGLang extend. A fresh runtime _Layout
    avoids sharing cached device metadata between the independent timing buffers.
    ``row0`` tests a suffix top-k launch without changing the production row ABI.
    """
    if (not seq_lens or len(seq_lens) != len(extend_lens) or not sum(extend_lens)
            or any(e < 0 or s < e or (s - e) % RATIO for s, e in zip(seq_lens, extend_lens))):
        raise ValueError("Require nonempty rows and group-aligned, nonnegative prefixes")
    if max(seq_lens) // RATIO > indexer.MAX_COMPRESSED_KEYS:
        raise ValueError("Compressed lengths exceed the runtime's uint16 top-k id limit")
    if topk_pattern not in ("projected", "equal", "repeated"):
        raise ValueError(topk_pattern)
    with torch.cuda.device(device):
        source = helpers.synthetic(seq_lens, extend_lens, device, seed=seed)
        pool, expected_pool = (helpers._pool(source.state, source.layer) for _ in range(2))
        inputs = plugin._indexer_inputs(source.module, source.hidden, source.positions, helpers._batch(source),
                                       helpers._metadata(source, pool))
        assert inputs is not None, "plugin rejected synthetic prefill metadata"
        q_ref, reference_metadata = _reference_prep(source, expected_pool)
        keys, _, _, _ = reference_metadata.get_prefill_mqa_inputs(source.layer, source.logical[:sum(extend_lens)])
        packed_ref = keys[:, 0]
        layout = indexer._Layout(tuple(seq_lens), tuple(extend_lens), source.hidden.device)
        assert len(layout.chunks) == 1, "individual-kernel cases must fit one production logits chunk"
        chunk = layout.chunks[0]
        rows, width, groups = layout.rows, chunk.width, inputs["write_locs"].numel()
        assert chunk.row0 == 0 and 0 <= row0 < rows and groups > 0
        case = _case(source, inputs, "prefill", seq_lens=list(seq_lens), extend_lens=list(extend_lens),
                     rows=rows, compressed_keys=[s // RATIO for s in seq_lens], groups=groups,
                     logits_shape=[rows, width], topk_pattern=topk_pattern, topk_row0=row0, seed=seed)

        # Guards follow, rather than precede, native allocations: timing output
        # pointers stay at their allocation base for the default row0=0 cases.
        q_storage = torch.empty((rows + 1, HEADS, DIM), dtype=torch.bfloat16, device=device)
        q = q_storage[:rows]
        packed_storage = torch.empty((max(layout.keys, 1) + 1, DIM), dtype=torch.bfloat16, device=device)
        packed = packed_storage[:-1]
        empty_packed = torch.full_like(packed_storage, SENTINEL)
        compressed_packed, full_packed = empty_packed.clone(), empty_packed.clone()
        full_packed[:layout.keys].copy_(packed_ref)
        base = 0
        for sequence, extend in zip(seq_lens, extend_lens):
            begin, end = base + (sequence - extend) // RATIO, base + sequence // RATIO
            compressed_packed[begin:end].copy_(packed_ref[begin:end])
            base = end
        valid = torch.empty(1, dtype=torch.int32, device=device)
        # The real top-k vector loads may read up to the next 512-value boundary.
        # Allocate the production safety pad, but never require its values.
        logits_storage = torch.empty(rows * width + indexer._PAD + 16, dtype=torch.float32, device=device)
        logits = logits_storage[:rows * width].view(rows, width)
        output_storage = torch.empty((rows + 1, WIDTH), dtype=torch.int32, device=device)
        output = output_storage[:rows]
        exact = _prefill_scores(q_ref, packed_ref, seq_lens, extend_lens, width)
        defined = torch.zeros((rows, width), dtype=torch.bool, device=device)
        work_items = chunk.items.cpu().tolist()
        for _, local, count, _, begin, end in work_items:
            defined[local:local + count, begin:end] = True
        counts = torch.minimum((layout.row_info[:, 0].long() + 1) // RATIO,
                               layout.row_info[:, 1].long() // RATIO)
        topk_exact = exact
        if topk_pattern == "equal":
            topk_exact = torch.zeros_like(exact)
        elif topk_pattern == "repeated":
            topk_exact = (torch.arange(width, device=device) % 7).double()[None].expand(rows, -1).clone()
        topk_seed = torch.full_like(logits_storage, math.nan)
        read = (torch.arange(width, device=device)[None] < counts[:, None]) & (counts[:, None] > TOPK)
        topk_seed[:rows * width].view(rows, width).copy_(topk_exact.masked_fill(~read, math.nan).float())

        def reset_q():
            for name in ("key_state", "rope_state"):
                inputs[name].copy_(source.state[name])
            q_storage.fill_(SENTINEL)
            valid.zero_()

        def run_q():
            return indexer._indexer_q_prep[(-(-rows // indexer._TB),)](
                inputs["qk"], q, inputs["q_weight"], inputs["key_state"], inputs["rope_state"],
                inputs["state_slots"], inputs["positions"], inputs["positions"].stride(0),
                inputs["cos_sin_cache"], inputs["axis_map"], valid, rows, DIM, inputs["q_eps"],
                TB=indexer._TB, H=HEADS, D=DIM, ROT=64,
                CACHE_STRIDE=inputs["cos_sin_cache"].stride(0), num_warps=4,
            )

        def check_q():
            _assert_exact(q, q_ref, "SGLang q prep must be bitexact")
            _assert_pool(inputs, expected_pool, ("key_state", "rope_state"))
            _assert_guard(q_storage[-1])
            assert int(valid.item()) == 1, "q prep did not initialize valid"

        def reset_compress():
            inputs["compressed"].copy_(source.state["compressed"])
            packed_storage.copy_(empty_packed)

        def run_compress():
            return indexer._indexer_k_compress[(-(-groups // indexer._GB),)](
                inputs["qk"], inputs["member_rows"], inputs["write_locs"], inputs["group_sequences"],
                inputs["group_ends"], inputs["rope_matrix"], inputs["cos_sin_cache"], inputs["axis_map"],
                inputs["k_weight"], inputs["compressed"], packed, layout.key_base, groups, DIM, inputs["k_eps"],
                GB=indexer._GB, H=HEADS, D=DIM, ROT=64, RATIO=RATIO,
                CACHE_STRIDE=inputs["cos_sin_cache"].stride(0), num_warps=4,
            )

        def check_compress():
            _assert_pool(inputs, expected_pool, ("compressed",))
            # Prefix destinations and guard were explicitly seeded, not assumed
            # to have any value in an uninitialized production allocation.
            _assert_exact(packed_storage, compressed_packed, "compressed packed writes/footprint")

        _register(case, "q_prep", run_q, reset_q, check_q, "_indexer_q_prep")
        _register(case, "k_compress", run_compress, reset_compress, check_compress, "_indexer_k_compress")

        if layout.pairs is not None:
            def reset_gather():
                inputs["compressed"].copy_(helpers._buffer(expected_pool, "compressed"))
                packed_storage.copy_(compressed_packed)

            def run_gather():
                return indexer._indexer_gather_prefix[(layout.pairs.shape[0],)](
                    inputs["compressed"], packed, inputs["token_slot_table"], layout.pairs,
                    inputs["token_slot_table"].stride(0), D=DIM, RATIO=RATIO, num_warps=1,
                )

            def check_gather():
                _assert_exact(packed_storage, full_packed, "SGLang packed prefix/untouched suffix")
                _assert_pool(inputs, expected_pool, ("compressed",))

            _register(case, "gather_prefix", run_gather, reset_gather, check_gather, "_indexer_gather_prefix")
            case.tensors["prefix_pairs"] = layout.pairs

        if work_items:
            def reset_logits():
                q_storage.fill_(SENTINEL)
                q.copy_(q_ref)
                packed_storage.copy_(full_packed)
                logits_storage.fill_(math.nan)
                logits_storage[-16:].fill_(SENTINEL)

            def run_logits():
                indexer.indexer_logits.launch(q, packed, logits, chunk.items, width, SCALE)

            def check_logits():
                report = _assert_logits(logits, exact, defined)
                _assert_guard(logits_storage[-16:])
                return report

            _register(case, "indexer_logits", run_logits, reset_logits, check_logits, "qsa_indexer_logits",
                      flydsl=indexer.indexer_logits)
            case.flops["indexer_logits"] = 2 * HEADS * DIM * sum(n * (end - start)
                                                              for _, _, n, _, start, end in work_items)
            case.tensors["logits_items"] = chunk.items

        previous = []

        def reset_topk():
            logits_storage.copy_(topk_seed)
            output_storage.fill_(SENTINEL)
            valid.fill_(1)

        def run_topk():
            indexer.indexer_topk.launch(logits[row0:], width, row0, rows - row0, inputs["logical_positions"],
                                       layout.row_info, output, valid)

        def check_topk():
            assert int(valid.item()) == 1, "top-k rejected matching logical positions"
            report = _assert_topk(output[row0:], layout.row_info[row0:, 0], layout.row_info[row0:, 1],
                                 topk_exact[row0:], equal_ties=topk_pattern == "equal")
            _assert_guard(output_storage[-1])
            _assert_guard(output[:row0])
            if previous:
                assert torch.equal(output, previous[0]), "top-k is not deterministic"
            else:
                previous.append(output.clone())
            return report

        _register(case, "indexer_topk", run_topk, reset_topk, check_topk, "qsa_indexer_topk",
                  flydsl=indexer.indexer_topk)
        case.tensors.update(q=q, packed=packed, logits=logits, tokens=output, valid=valid,
                            row_info=layout.row_info, key_base=layout.key_base, q_reference=q_ref,
                            compressed_reference=helpers._buffer(expected_pool, "compressed"))
        case.metadata.update(labels=list(case.runs), prefix_pairs=0 if layout.pairs is None else len(layout.pairs),
                             logits_items=work_items, q_shape=[rows, HEADS, DIM],
                             packed_shape=[layout.keys, DIM], tokens_shape=[rows, WIDTH],
                             flop_definition="2*4*128 per defined dot; excludes padded MFMA/activation/prep work",
                             logits_padding="only explicit item ranges checked; partial 4-key stores/other padding unspecified")
        case.owners = (source, pool, expected_pool, layout, q_storage, packed_storage, logits_storage, output_storage)
        assert tuple(case.runs) == prefill_labels(seq_lens, extend_lens)
        return case


@torch.no_grad()
def make_decode_case(lengths, device, *, padding=0, context=4096, heads=4, shuffle_pages=False, seed=73):
    """Graph-metadata decode prep and paged logits, executed eagerly and separately.

    ``lengths`` are token lengths, not compressed lengths. Zero-compressed and
    request-zero padding rows are included; graph capture itself is not tested.
    """
    if (not lengths or min(lengths) < 1 or context < max(lengths) or context % 64
            or padding < 0 or heads not in (4, 8)):
        raise ValueError("Require positive lengths, a sufficient 64-aligned context and 4/8 logit heads")
    with torch.cuda.device(device):
        source = helpers.decode_forward_case(lengths, device, padding=padding, seed=seed, context=context)
        if shuffle_pages:
            for request in range(1, source.requests + 1):
                pages = source.table[request].view(-1, 64)
                order = torch.randperm(pages.shape[0], generator=source.generator, device=device)
                source.table[request].copy_(pages[order].reshape(-1))
            helpers.decode_step(source, lengths)
        pool, expected_pool = (helpers._pool(source.state, source.layer) for _ in range(2))
        inputs = plugin._decode_forward_inputs(source.module, source.hidden, source.positions,
                                              helpers._decode_batch(source), helpers._decode_metadata(source, pool))
        assert inputs is not None, "plugin rejected synthetic graph-decode metadata"
        q_ref, _ = _reference_prep(source, expected_pool, decode=True)
        rows, width = source.rows, inputs["page_table"].shape[1] * 16
        counts = inputs["lengths"].cpu().tolist()
        assert counts == [n // RATIO for n in lengths] + [0] * padding
        # Stale-but-valid page ids after the live prefix must never be consumed.
        for row, count in enumerate(counts):
            inputs["page_table"][row, -(-count // 16):].fill_(inputs["cache"].shape[0] - 1)
        case = _case(source, inputs, "decode", seq_lens=list(lengths), padding=padding, rows=rows, context=context,
                     compressed_lengths=counts, logits_shape=[rows, width], logits_heads=heads,
                     shuffled_pages=shuffle_pages, seed=seed)
        q_storage = torch.empty((rows + 1, HEADS, DIM), dtype=torch.bfloat16, device=device)
        q = q_storage[:rows]
        logits_q = torch.zeros((rows, heads, DIM), dtype=torch.bfloat16, device=device)
        logits_storage = torch.empty((rows + 1, width), dtype=torch.float32, device=device)
        logits = logits_storage[:rows]
        exact = torch.full((rows, width), math.nan, dtype=torch.float64, device=device)
        compressed_ref = helpers._buffer(expected_pool, "compressed")
        for row, count in enumerate(counts):
            if count:
                page_ids = inputs["page_table"][row, :-(-count // 16)].long()
                slots = (page_ids[:, None] * 16 + torch.arange(16, device=device)).flatten()[:count]
                keys = compressed_ref.view(-1, DIM)[slots].double()
                exact[row, :count] = (q_ref[row].double() @ keys.T).relu().sum(0) / math.sqrt(DIM)
        columns = torch.arange(width, device=device)[None]
        defined = columns < inputs["lengths"][:, None]
        partial_page = (columns >= inputs["lengths"][:, None]) & (
            columns < ((inputs["lengths"][:, None] + 15) // 16) * 16)

        def reset_prep():
            for name in ("key_state", "rope_state", "compressed"):
                inputs[name].copy_(source.state[name])
            q_storage.fill_(SENTINEL)

        def run_prep():
            return indexer._indexer_decode_prep[(rows,)](
                inputs["qk"], q, inputs["q_weight"], inputs["k_weight"], inputs["key_state"], inputs["rope_state"],
                inputs["compressed"], inputs["state_slots"], inputs["group_locs"], inputs["write_locs"],
                inputs["positions"], inputs["positions"].stride(0), inputs["cos_sin_cache"], inputs["axis_map"],
                DIM, inputs["q_eps"], inputs["k_eps"], H=HEADS, D=DIM, ROT=64, RATIO=RATIO,
                CACHE_STRIDE=inputs["cos_sin_cache"].stride(0), num_warps=4,
            )

        def check_prep():
            _assert_exact(q, q_ref, "SGLang decode Q must be bitexact (including padding rows)")
            _assert_pool(inputs, expected_pool, ("key_state", "rope_state", "compressed"))
            _assert_guard(q_storage[-1])

        def reset_logits():
            inputs["compressed"].copy_(compressed_ref)
            logits_q[:, :HEADS].copy_(q_ref)
            logits_q[:, HEADS:].zero_()
            logits_storage.fill_(math.nan)
            logits_storage[-1].fill_(SENTINEL)

        def run_logits():
            indexer.indexer_decode.launch(logits_q, inputs["cache"], inputs["page_table"], inputs["lengths"],
                                         logits, SCALE)

        def check_logits():
            report = _assert_logits(logits, exact, defined)
            assert bool(torch.isneginf(logits[partial_page]).all()), "partial-page padding must be -inf"
            _assert_guard(logits_storage[-1])
            return report

        _register(case, "decode_prep", run_prep, reset_prep, check_prep, "_indexer_decode_prep")
        _register(case, "decode_logits", run_logits, reset_logits, check_logits, "qsa_indexer_decode_logits",
                  flydsl=indexer.indexer_decode)
        case.flops["decode_logits"] = 2 * HEADS * DIM * sum(counts)
        case.tensors.update(q=q, logits_q=logits_q, logits=logits, q_reference=q_ref,
                            compressed_reference=compressed_ref)
        case.metadata.update(labels=list(case.runs), q_shape=[rows, HEADS, DIM],
                     cache_shape=list(inputs["cache"].shape), page_table_shape=list(inputs["page_table"].shape),
                             flop_definition="2*4*128 per live compressed key; excludes padded heads/pages and prep",
                             logits_padding="-inf only from length to next page; later columns unspecified",
                             graph="canonical graph metadata, eager kernel launches only")
        case.owners = (source, pool, expected_pool, q_storage, logits_storage)
        return case
