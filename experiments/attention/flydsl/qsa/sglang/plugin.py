"""Temporary SGLang hook adapter; the QSA runtime itself has no SGLang dependency."""

from collections import Counter
from contextvars import ContextVar
from functools import partial
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil


_call = ContextVar("pyhip_qsa_attention_forward", default=None)
_state = None
_POLICY = "dense2051;packed=pad1.7;raw=rho4"
_ABI = {
    "sglang.srt.layers.attention.qwen_sparse_attn_backend": "c75dbabacea6e9012428e7dbe695fc49400406fed77272752dce82f9cd8b6113",
    "sglang.srt.layers.attention.qsa.kernel": "7e369f09293fb9b0872c21f0010247ec1e3a696b5ad4809f04d9a730b1031095",
    "sglang.srt.layers.attention.qsa.qsa_indexer": "37547d9535934961c233f0c5d6ac5a5b94a08ba325c573857c2aebd9b1593257",
}
# The indexer replacement reproduces these eager numerics and pool side effects.
_INDEXER_ABI = {
    "sglang.srt.layers.attention.qsa.metadata": "6486f5d5e10597b94320e16aff889f2381b0c361707ccad8f77430c4d7290e6e",
    "sglang.srt.layers.attention.qsa.mqa": "feee2b80ea3b3b59bdddecbd74eba6de8e363fadd0b9dc3732a1b3372a3baf31",
    "sglang.srt.mem_cache.qsa_kv_pool": "f52178080fa5487c75bfb7070067c33d8071fdce823a69a6316640393191ff7e",
    "sglang.srt.layers.rotary_embedding.mrope": "7e16f073fa9016df2e67f3fe0b34a30cbe01a017e8dbf81210a8c12b328ee6f2",
    "sglang.srt.layers.rotary_embedding.utils": "828a50285218e40c3429d2f4e75b1d051e3a89d831a7ab005e8c2609278aeddc",
    "sglang.srt.layers.layernorm": "fab9eabf943a38de36ab675d012d5deab477ec8eb8f816361f022a1dfd32dce6",
    "sglang.kernels.ops.layernorm.minimax_m3_rmsnorm": "e7e81662d1989eacd46b757b0240111ef9caae31a77fb3035ca2b9196307198d",
    "sglang.kernels.ops.elementwise.fast_topk": "65eeb1c14a111e651819bdac3ecca3a796f06feec4fb5c03f2ea08a5a65d29e4",
    "sglang.srt.models.qwen4_exp": "d3596d3700a394d8c5bfec776feed0d7ecd9f27e5cd2931aa9b030014632335b",
}
# hipBLASLt's default heuristic is slow for the [M, 2560] x [2560, 640] index projection once
# M > 5120 on MI308X (e.g. 474 us at M=12000); solution 90517 (MT128x256x64, ~60 us per 4096 rows)
# is used there. Its FP32 accumulation order differs from SGLang's GEMM, so
# PYHIP_QSA_INDEXER_GEMM=sglang, and indexer validation, keep SGLang's bit-exact projection.
_INDEXER_GEMM = (90517, 5121, "36a6269ab58c07b750d7231395b56b035d25f10c24a0d73db3379a1de6791891")
# Per (path, layer) device counters of the CUDA-graph-safe decode validation (PYHIP_QSA_VALIDATE=1).
_DECODE_COUNTS = ("calls", "rows", "q_mismatch", "ring_key_mismatch", "ring_rope_mismatch", "compressed_rows",
                  "compressed_mismatch", "different_token_sets", "selection_violations")
_DECODE_FAILURES = ("q_mismatch", "ring_key_mismatch", "ring_rope_mismatch", "compressed_mismatch",
                    "selection_violations")


def _eligible(backend, q, k, v, layer, batch, indices, kwargs):
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import get_tc_piecewise_forward_context

    if batch.forward_mode != ForwardMode.EXTEND or backend.runner is None or backend.runner.is_draft_worker:
        return False
    if (not q.is_cuda or torch.version.hip is None
            or not torch.cuda.get_device_properties(q.device).gcnArchName.startswith("gfx942")
            or get_is_capture_mode() or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing() or get_tc_piecewise_forward_context() is not None):
        return False
    profile, ps = backend.qsa_profile, backend.runner.ps
    if (profile is None or profile.variant != "compressed"
            or (profile.compress_ratio, profile.block_topk, profile.budget) != (4, 512, 2048)
            or ps.attn_cp_size != 1 or ps.attn_dcp_size != 1):
        return False
    if k is None or v is None or backend.runner.kv_cache_dtype != torch.bfloat16:
        return False
    if ((layer.tp_q_head_num not in (12, 6, 3)) or layer.tp_k_head_num != 1
            or layer.tp_v_head_num != 1 or layer.head_dim != 256 or layer.v_head_dim != 256
            or layer.logit_cap != 0 or layer.sliding_window_size != -1
            or layer.is_cross_attention or layer.pos_encoding_mode != "NONE"
            or any(value is not None for value in kwargs.values())):
        return False
    if any(t.dtype != torch.bfloat16 or t.device != q.device or t.requires_grad for t in (q, k, v)):
        return False
    if any(isinstance(t, torch.Tensor) and t.device.type != "cpu" for t in (batch.extend_seq_lens_cpu, batch.seq_lens_cpu)):
        return False
    return (indices is not None and indices.ndim == 2 and indices.shape[1] == 2051 and indices.shape[0] > 0
            and batch.extend_seq_lens_cpu is not None and batch.seq_lens_cpu is not None)


def _around_forward(original, backend, q, k, v, layer, batch, save_kv_cache=True, topk_indices=None, **kwargs):
    if not _eligible(backend, q, k, v, layer, batch, topk_indices, kwargs):
        return original(backend, q, k, v, layer, batch, save_kv_cache=save_kv_cache, topk_indices=topk_indices, **kwargs)
    queries = tuple(int(n) for n in batch.extend_seq_lens_cpu)
    sequences = tuple(int(n) for n in batch.seq_lens_cpu)
    if len(queries) != len(sequences) or sum(queries) != topk_indices.shape[0] or any(qn < 0 or sn < qn for qn, sn in zip(queries, sequences)):
        raise ValueError("Invalid host QSA request layout")
    token = _call.set((backend, layer, queries, tuple(s - q for s, q in zip(sequences, queries))))
    try:
        # Original code still owns KV writes/gather, valid-row trimming and padding.
        return original(backend, q, k, v, layer, batch, save_kv_cache=save_kv_cache, topk_indices=topk_indices, **kwargs)
    finally:
        _call.reset(token)


class _State:
    def __init__(self):
        self.runtime = importlib.import_module("..attention", __package__)
        self.profiling = False
        self.calls = Counter()
        self.checked = set()
        self.checks = []
        self.pending = {}
        self.dump_layers = {int(n) for n in os.environ.get("PYHIP_QSA_DUMP_LAYERS", "").split(",") if n}
        self.dump_rows = {int(n) for n in os.environ.get("PYHIP_QSA_DUMP_ROWS", "12000,11888").split(",") if n}
        self.report_dir = Path(os.environ["PYHIP_QSA_REPORT_DIR"]) if "PYHIP_QSA_REPORT_DIR" in os.environ else None
        self.indexer_dump_layers = {int(n) for n in os.environ.get("PYHIP_QSA_INDEXER_DUMP_LAYERS", "").split(",") if n}
        self.indexer_calls = Counter()
        self.indexer_pending = {}
        self.indexer_runtime = None
        self.indexer_gemm = None
        self.indexer_checked = set()
        self.indexer_checks = []
        self.decode_reference = False
        self.decode_checks = {}

    def execute(self, context, q, k, v, indices, scale, original, args):
        import torch

        backend, layer, queries, prefixes = context
        if (any(not t.is_contiguous() or t.data_ptr() % 16 or t.ndim != 3 or t.shape[-1] != 256 for t in (q, k, v))
                or indices.dtype != torch.int32 or not indices.is_contiguous()):
            return original(*args)
        output = self.runtime.attention(q, k, v, indices, query_lens=queries, prefix_lens=prefixes, softmax_scale=scale)
        key = (layer.layer_id, queries, prefixes)
        if self.profiling:
            self.calls[str(layer.layer_id)] += 1
            if layer.layer_id in self.dump_layers and sum(queries) in self.dump_rows and key not in self.pending:
                with torch.profiler.record_function("pyhip_qsa.capture_attention"):
                    # Host lengths + token indices fully determine replay metadata;
                    # do not couple the adapter to the private workspace cache.
                    tensors = {name: tensor.detach().clone() for name, tensor in (
                        ("q", q), ("k", k), ("v", v), ("indices", indices), ("output", output))}
                self.pending[key] = (dict(layer_id=layer.layer_id, tp_rank=backend.runner.ps.tp_rank,
                                          query_lens=queries, prefix_lens=prefixes, scale=scale), tensors)
        elif os.environ.get("PYHIP_QSA_VALIDATE", "0") == "1" and key not in self.checked:
            reference = importlib.import_module(".attention_validation", __package__).reference
            legacy = original(*args)
            expected = reference(q, k, v, indices, query_lens=queries, prefix_lens=prefixes, scale=scale)
            try:
                torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)
            except AssertionError as error:
                self.capture_validation_failure(context, q, k, v, indices, scale, output, expected, error,
                                                legacy_expected=legacy)
                raise
            legacy_error = None
            try:
                torch.testing.assert_close(output, legacy, rtol=0.02, atol=0.02)
            except AssertionError as error:
                # Retain the legacy comparison, but it is not a correctness
                # oracle: BF16 Q prescaling fails FP64 on QSA-T13's real input.
                legacy_error = str(error)
                logging.getLogger(__name__).warning(
                    "QSA attention layer=%s queries=%s prefixes=%s passed FP32 .02/.02; legacy SGLang differs: %s",
                    layer.layer_id, queries, prefixes, legacy_error)
            self.checked.add(key)
            self.checks.append(dict(layer_id=layer.layer_id, query_lens=queries, prefix_lens=prefixes,
                                    rows=output.shape[0], rtol=0.02, atol=0.02,
                                    reference="fp32_selected_tokens", elements=output.numel(),
                                    legacy_close=legacy_error is None, legacy_error=legacy_error))
        return output

    def capture_validation_failure(self, context, q, k, v, indices, scale, output, expected, error,
                                   *, legacy_expected=None):
        """Save the actual failed invocation before propagating the original assertion."""
        import torch

        backend, layer, queries, prefixes = context
        directory = os.environ.get("PYHIP_QSA_DUMP_DIR")
        if directory is None and self.report_dir is not None:
            directory = str(self.report_dir / "failures")
        if directory is None:
            logging.getLogger(__name__).error("QSA attention validation failed: layer=%s queries=%s prefixes=%s",
                                              layer.layer_id, queries, prefixes)
            return
        try:
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in (
                ("q", q), ("k", k), ("v", v), ("indices", indices), ("output", output), ("expected", expected))}
            if legacy_expected is not None:
                tensors["legacy_expected"] = legacy_expected.detach().cpu().contiguous()
            meta = dict(layer_id=layer.layer_id, tp_rank=backend.runner.ps.tp_rank,
                        query_lens=queries, prefix_lens=prefixes, scale=scale, rtol=0.02, atol=0.02,
                        error=str(error), validation_failed=True, reference="fp32_selected_tokens",
                        tensor_metadata={name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
                            for name, tensor in tensors.items()})
            layout = hashlib.sha256(json.dumps((queries, prefixes)).encode()).hexdigest()[:12]
            path = target / f"failure_tp{meta['tp_rank']}_layer{layer.layer_id}_m{sum(queries)}_{layout}_pid{os.getpid()}.pt"
            with path.open("xb") as stream:
                torch.save({"metadata": meta, "tensors": tensors}, stream)
            logging.getLogger(__name__).error("QSA attention failed validation input saved to %s", path)
        except Exception:
            # Diagnostic I/O must not replace or suppress the numerical failure.
            logging.getLogger(__name__).exception("Could not save QSA attention validation failure")

    def projection(self, module, rows):
        """Return the hipBLASLt index projection for this call, or None for SGLang's GEMM."""
        import torch

        solution, min_rows, kernel = _INDEXER_GEMM
        weight = module.index_qk_proj.weight
        if (os.environ.get("PYHIP_QSA_INDEXER_GEMM", "hipblaslt") != "hipblaslt" or rows < min_rows
                or module.index_qk_proj.bias is not None or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != (640, 2560) or not weight.is_contiguous()):
            return None
        import aiter
        from aiter.tuned_gemm import hipb_gemm

        if self.indexer_gemm is None:
            hipb_gemm(weight[:1], weight, solution)  # creates aiter's hipBLASLt handle
            name = aiter.getHipblasltKernelName(solution)
            self.indexer_gemm = hashlib.sha256(name.encode()).hexdigest() == kernel
            if not self.indexer_gemm:
                logging.getLogger(__name__).warning(
                    "PyHIP QSA indexer: hipBLASLt solution %d is %s..., using SGLang's GEMM", solution, name[:64])
        return partial(hipb_gemm, weights=weight, solidx=solution) if self.indexer_gemm else None

    def indexer(self, original, module, hidden_states, positions, forward_batch, metadata):
        import torch

        validate = os.environ.get("PYHIP_QSA_VALIDATE", "0") == "1"
        if os.environ.get("PYHIP_QSA_INDEXER_DECODE", "0") == "1":
            inputs = _decode_forward_inputs(module, hidden_states, positions, forward_batch, metadata)
            if inputs is not None:
                if self.indexer_runtime is None:
                    self.indexer_runtime = importlib.import_module("..indexer", __package__)
                if validate:
                    return self.validated_decode_forward(original, module, hidden_states, positions,
                                                         forward_batch, metadata, inputs)
                return self.indexer_runtime.decode_forward(inputs.pop("qk"), **inputs)
        if (validate and self.decode_checks and forward_batch.forward_mode.is_extend()
                and module.layer_id == min(layer for _, layer in self.decode_checks)
                and not torch.cuda.is_current_stream_capturing()):
            # Decode graphs cannot synchronize: their counters are read once per prefill forward.
            self.check_decode()
        enabled = os.environ.get("PYHIP_QSA_INDEXER", "0") == "1"
        lens = (forward_batch.seq_lens_cpu, forward_batch.extend_seq_lens_cpu)
        key = (module.layer_id, *(None if v is None else tuple(int(n) for n in v) for v in lens))
        # A validated call keeps SGLang's bit-exact projection so the comparison isolates the runtime.
        check = (enabled and not self.profiling and os.environ.get("PYHIP_QSA_VALIDATE", "0") == "1"
                 and key not in self.indexer_checked)
        inputs = _indexer_inputs(module, hidden_states, positions, forward_batch, metadata,
                                 None if check else self.projection) if enabled else None
        if inputs is None:
            output = original(module, hidden_states, positions, forward_batch, metadata)
        else:
            if self.indexer_runtime is None:
                self.indexer_runtime = importlib.import_module("..indexer", __package__)
            if check:
                expected = original(module, hidden_states, positions, forward_batch, metadata)
                state = _indexer_state(metadata, module.layer_id)
            output = self.indexer_runtime.prefill_indexer(inputs.pop("qk"), **inputs)
            if check:
                self.indexer_checks.append(_indexer_compare(module, hidden_states, positions, forward_batch,
                                                            metadata, key, output, expected, state,
                                                            _indexer_state(metadata, module.layer_id)))
                self.indexer_checked.add(key)
        if self.profiling:
            self.indexer_calls[str(module.layer_id)] += 1
            key = (module.layer_id, output.shape[0])
            if (module.layer_id in self.indexer_dump_layers and output.shape[0] in self.dump_rows
                    and key not in self.indexer_pending):
                with torch.profiler.record_function("pyhip_qsa.capture_indexer"):
                    self.indexer_pending[key] = _indexer_snapshot(module, hidden_states, positions,
                                                                  forward_batch, metadata, output)
        return output

    def finish(self, rank):
        import torch

        snapshots = []
        for (layer, queries, prefixes), (meta, tensors) in self.pending.items():
            directory = Path(os.environ["PYHIP_QSA_DUMP_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            tensors = {name: value.cpu() for name, value in tensors.items()}
            meta["tensor_metadata"] = {name: {"sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                                       for name, value in tensors.items()}
            layout = hashlib.sha256(json.dumps((queries, prefixes)).encode()).hexdigest()[:12]
            path = directory / f"tp{rank}_layer{layer}_m{sum(queries)}_{layout}.pt"
            with path.open("xb") as stream:
                torch.save({"metadata": meta, "tensors": tensors}, stream)
            snapshots.append(path.name)
        indexer_snapshots = []
        for (layer, rows), (meta, tensors) in self.indexer_pending.items():
            directory = Path(os.environ["PYHIP_QSA_DUMP_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            tensors = {name: value.cpu() for name, value in tensors.items()}
            meta["tensor_metadata"] = {name: {"sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                                              "shape": list(value.shape), "dtype": str(value.dtype)}
                                       for name, value in tensors.items()}
            path = directory / f"indexer_tp{rank}_layer{layer}_m{rows}.pt"
            with path.open("xb") as stream:
                torch.save({"metadata": meta, "tensors": tensors}, stream)
            indexer_snapshots.append(path.name)
        if not self.calls:
            raise RuntimeError("Profile captured zero QSA attention replacement calls")
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            with (self.report_dir / f"qsa_tp{rank}.json").open("x") as stream:
                json.dump({"calls_per_layer": dict(self.calls), "validation": self.checks,
                           "input_snapshots": snapshots, "policy": _POLICY,
                           "indexer_calls_per_layer": dict(self.indexer_calls),
                           "indexer_validation": self.indexer_checks,
                           "indexer_decode_validation": self.decode_summary(),
                           "indexer_snapshots": indexer_snapshots}, stream, indent=2)
        self.pending.clear()
        self.indexer_pending.clear()
        self.indexer_calls.clear()
        self.profiling = False

    def decode(self, original, module, q, cache, table, lengths, width, positions, sequences):
        if self.decode_reference or not _decode_eligible(module, q, cache, table, lengths, width):
            return original(module, q, cache, table, lengths, width, positions, sequences)
        if self.indexer_runtime is None:
            self.indexer_runtime = importlib.import_module("..indexer", __package__)
        if os.environ.get("PYHIP_QSA_VALIDATE", "0") != "1":
            return self.indexer_runtime.decode_indexer(q, cache, table, lengths, positions, sequences)
        import torch

        counts, gap = self.decode_counters("select", module.layer_id, q.device)
        expected = original(module, q, cache, table, lengths, width, positions, sequences)
        actual, logits = self.indexer_runtime._decode_select(q, cache, table, lengths, positions, sequences)
        real = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
        different, violations, worst = _selection_check(logits, lengths, actual, expected, real,
                                                        module.block_topk, module.compress_ratio)
        one, zero = (torch.full((), n, dtype=torch.int64, device=q.device) for n in (1, 0))
        counts += torch.stack((one, real.sum(), zero, zero, zero, zero, zero, different, violations))
        torch.maximum(gap, worst.reshape(1), out=gap)
        return actual

    def validated_decode_forward(self, original, module, hidden_states, positions, forward_batch, metadata, inputs):
        """PYHIP_QSA_VALIDATE=1 graph decode: SGLang's own forward runs first (selection hook bypassed) as
        reference for q, the pool writes and the selection; the PyHIP forward then rewrites the same rows.
        Differences accumulate in device counters, so the call stays CUDA-graph capturable."""
        import torch

        rows, ratio = inputs["qk"].shape[0], module.compress_ratio
        slots, locs, lengths = inputs["state_slots"], inputs["write_locs"].long(), inputs["lengths"]
        counts, gap = self.decode_counters("forward", module.layer_id, hidden_states.device)
        self.decode_reference = True
        try:
            expected = original(module, hidden_states, positions, forward_batch, metadata)
        finally:
            self.decode_reference = False
        buffers = (inputs["key_state"], inputs["rope_state"], inputs["compressed"])
        reference = (buffers[0][slots], buffers[1][slots], buffers[2][locs])
        q_reference = module.project_qk(hidden_states[:rows], positions[..., :rows])[0]
        actual, q, logits = self.indexer_runtime._decode_forward(inputs.pop("qk"), **inputs)
        # Graph padding rows use the reserved request 0 (ring rows < ratio); slot 0 is the dump row.
        real, boundary = slots >= ratio, locs != 0
        written = (buffers[0][slots], buffers[1][slots], buffers[2][locs])
        mismatch = [(a != b).flatten(1).any(1) & real for a, b in zip((q, *written), (q_reference, *reference))]
        mismatch[3] &= boundary
        different, violations, worst = _selection_check(logits, lengths, actual, expected, real,
                                                        module.block_topk, ratio)
        one = torch.ones((), dtype=torch.int64, device=q.device)
        counts += torch.stack((one, real.sum(), *(m.sum() for m in mismatch[:3]), (boundary & real).sum(),
                               mismatch[3].sum(), different, violations))
        torch.maximum(gap, worst.reshape(1), out=gap)
        return actual

    def decode_counters(self, kind, layer, device):
        import torch

        key = (kind, layer)
        if key not in self.decode_checks:
            # A capture-time allocation would be re-initialized by every replay.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PyHIP QSA decode validation needs an eager call before CUDA-graph capture")
            self.decode_checks[key] = (torch.zeros(len(_DECODE_COUNTS), dtype=torch.int64, device=device),
                                       torch.full((1,), -math.inf, dtype=torch.float32, device=device))
        return self.decode_checks[key]

    def decode_summary(self):
        """Per (path, layer) decode validation totals; reads the device counters (synchronizes)."""
        summary = {}
        for (kind, layer), (counts, gap) in sorted(self.decode_checks.items()):
            worst = float(gap.item())
            summary[f"{kind}:{layer}"] = dict(zip(_DECODE_COUNTS, counts.tolist()),
                                              max_relative_boundary_gap=worst if math.isfinite(worst) else None)
        return summary

    def check_decode(self):
        failed = {name: value for name, value in self.decode_summary().items()
                  if any(value[field] for field in _DECODE_FAILURES)}
        if failed:
            raise AssertionError(f"PyHIP QSA decode differs from SGLang: {failed}")


def _selection_check(logits, lengths, actual, expected, real, topk, ratio):
    """Compare two SGLang-ABI token selections on device (CUDA-graph safe) using the PyHIP logits.

    An expanded row lists the ``ratio`` tokens of every selected complete block, then (compacted right after
    them) the causal tail tokens at or past ``ratio * length``, then -1. Each selection's blocks must be a
    top-``topk`` set of the row's valid logits up to a relative near-tie of 1e-5 (the offline FP64
    criterion), with min(length, topk) complete blocks and identical tail tokens. Returns int64 counts of
    real rows with different token sets and with violations, and the largest relative boundary gap over
    real rows.
    """
    import torch

    rows, width = logits.shape
    valid = torch.arange(width, device=logits.device)[None, :] < lengths[:, None]
    values = logits.masked_fill(~valid, -math.inf)
    scale = logits.masked_fill(~valid, 0).abs().amax(1).clamp_min(1e-30)
    limit = lengths.long()[:, None] * ratio
    expect = lengths.long().clamp_max(topk)
    bad = torch.zeros(rows, dtype=torch.bool, device=logits.device)
    worst = torch.full((rows,), -math.inf, device=logits.device)
    tails = []
    for tokens in (actual, expected):
        tokens = tokens.long()
        block = (tokens >= 0) & (tokens < limit)
        hits = torch.zeros((rows, width), dtype=torch.int32, device=logits.device)
        hits.scatter_add_(1, torch.where(block, tokens // ratio, 0).clamp_max(width - 1), block.int())
        selected = hits > 0
        gap = (values.masked_fill(selected, -math.inf).amax(1) - values.masked_fill(~selected, math.inf).amin(1)) / scale
        bad |= ((selected & (hits != ratio)).any(1) | (selected.sum(1) != expect) | (gap > 1e-5))
        worst = torch.maximum(worst, gap)
        tails.append(torch.where(tokens >= limit, tokens, -1).sort(dim=1).values)
    bad |= (tails[0] != tails[1]).any(1)
    different = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(1)
    return (different & real).sum(), (bad & real).sum(), worst.masked_fill(~real, -math.inf).amax()


def _decode_eligible(module, q, cache, table, lengths, width):
    """Whether one ``select_decode_tokens`` call (eager or captured) fits the FlyDSL paged logits.

    Shapes, dtypes and layouts are static per CUDA graph, so the decision holds for every replay;
    lengths and page ids are only read on device.
    """
    import torch

    return (q.is_cuda and q.dtype == torch.bfloat16 and q.ndim == 3 and q.shape[1] in (4, 8) and q.shape[2] == 128
            and q.is_contiguous() and _decode_select_eligible(module, q.shape[0], q.device, cache, table, lengths, width))


def _decode_select_eligible(module, rows, device, cache, table, lengths, width):
    import torch

    if ((module.index_n_heads, module.index_head_dim, module.compress_ratio, module.block_topk,
         module.token_topk) != (4, 128, 4, 512, 2048)):
        return False
    if (torch.version.hip is None or not 0 < rows < 65536
            or not torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx942")):
        return False
    if (cache.dtype != torch.bfloat16 or cache.ndim != 4 or tuple(cache.shape[1:]) != (16, 1, 128)
            or not cache.is_contiguous() or cache.numel() * 2 >= 1 << 31):
        return False
    return (table.dtype == torch.int32 and table.ndim == 2 and table.shape[0] == rows and table.is_contiguous()
            and width == table.shape[1] * 16 and lengths.dtype == torch.int32 and lengths.ndim == 1
            and lengths.numel() == rows and lengths.is_contiguous()
            and all(t.device == device for t in (cache, table, lengths)))


def _decode_forward_inputs(module, hidden_states, positions, forward_batch, metadata):
    """Map one CUDA-graph decode ``forward_cuda`` (eager warmup or capture) onto the runtime, or None.

    Reproduces SGLang's unfused decode prep (BF16 cos/sin cache); eager decode, target-verify/draft
    modes and every other layout keep SGLang's path. Metadata values are only read on device.
    """
    import torch
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    graph = (metadata.graph_write_locs, metadata.graph_ring_group_locs, metadata.pending_ring_slots,
             metadata.decode_logical_positions, metadata.graph_compressed_page_table,
             metadata.graph_compressed_lengths)
    if forward_batch.forward_mode != ForwardMode.DECODE or not metadata.is_cuda_graph or any(t is None for t in graph):
        return None
    if ((module.index_n_heads, module.index_kv_heads, module.index_head_dim, module.compress_ratio,
         module.block_topk, module.token_topk) != (4, 1, 128, 4, 512, 2048)):
        return None
    rotary = module.rotary_emb
    if type(rotary) is MRotaryEmbedding:
        if rotary.mrope_interleaved_glm or len(rotary.mrope_section or ()) not in (0, 3):
            return None
    elif type(rotary) is not RotaryEmbedding:
        return None
    cache = rotary.cos_sin_cache
    rows = metadata.token_to_batch_idx.numel()
    # A BF16 cos/sin cache keeps SGLang off its fused prep/compress kernels (they need FP32).
    if (not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16 or torch.compiler.is_compiling()
            or not rotary.is_neox_style or rotary.rotary_dim != 64 or cache.dtype != torch.bfloat16
            or not cache.is_contiguous() or cache.device != hidden_states.device or cache.shape[1] != 64
            or not 0 < rows <= hidden_states.shape[0] or positions.dtype != torch.int64
            or positions.ndim not in (1, 2) or positions.shape[-1] < rows or positions.stride(-1) != 1
            or (positions.ndim == 2 and positions.shape[0] != 3)):
        return None
    pool, layer = metadata.token_to_kv_pool, module.layer_id
    key_state, rope_state = pool.get_qsa_key_state_buffer(layer), pool.qsa_rope_position_buffer
    compressed = pool.get_qsa_compressed_k_buffer(layer)
    slots, group_locs = metadata.pending_ring_slots[:rows], metadata.graph_ring_group_locs[:rows]
    write_locs = metadata.graph_write_locs[:rows]
    if (any(t.dtype != torch.bfloat16 or tuple(t.shape[1:]) != (1, 128) or not t.is_contiguous()
            for t in (key_state, compressed))
            or rope_state.dtype != torch.int64 or tuple(rope_state.shape[1:]) != (3,) or not rope_state.is_contiguous()
            or slots.dtype != torch.int64 or not slots.is_contiguous() or group_locs.dtype != torch.int32
            or tuple(group_locs.shape) != (rows, 4) or not group_locs.is_contiguous()
            or write_locs.dtype != torch.int32 or not write_locs.is_contiguous()):
        return None
    cache_view, table, lengths, width = metadata.get_decode_mqa_inputs(layer)
    if not _decode_select_eligible(module, rows, hidden_states.device, cache_view, table, lengths, width):
        return None
    qk = module.index_qk_proj(hidden_states[:rows])[0]
    return dict(qk=qk.contiguous(), positions=positions[:, :rows] if positions.ndim == 2 else positions[:rows],
                state_slots=slots, group_locs=group_locs, write_locs=write_locs, key_state=key_state,
                rope_state=rope_state, compressed=compressed, cos_sin_cache=cache,
                axis_map=module._rope_axis_map(hidden_states.device), q_weight=module.q_layernorm.weight.data,
                k_weight=module.k_layernorm.weight.data, q_eps=module.q_layernorm.variance_epsilon,
                k_eps=module.k_layernorm.variance_epsilon, cache=cache_view, page_table=table, lengths=lengths,
                query_positions=metadata.decode_logical_positions[:rows],
                sequence_lengths=metadata.get_seqlens_int32())


def _indexer_inputs(module, hidden_states, positions, forward_batch, metadata, projection=None):
    """Map one eager prefill indexer call onto the PyHIP runtime, or None to keep SGLang's path.

    projection(module, rows) may return a replacement for ``module.index_qk_proj``.
    """
    import torch
    from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    from ..indexer import MAX_COMPRESSED_KEYS

    if (forward_batch.forward_mode != ForwardMode.EXTEND or metadata.is_cuda_graph
            or metadata.compress_member_rows is None or metadata.write_locs is None
            or forward_batch.positions is None or forward_batch.seq_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None):
        return None
    if (not hidden_states.is_cuda or torch.version.hip is None or hidden_states.dtype != torch.bfloat16
            or not torch.cuda.get_device_properties(hidden_states.device).gcnArchName.startswith("gfx942")
            or get_is_capture_mode() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
        return None
    if ((module.index_n_heads, module.index_kv_heads, module.index_head_dim, module.compress_ratio,
         module.block_topk, module.token_topk) != (4, 1, 128, 4, 512, 2048)):
        return None
    rotary = module.rotary_emb
    if type(rotary) is MRotaryEmbedding:
        if rotary.mrope_interleaved_glm or len(rotary.mrope_section or ()) not in (0, 3):
            return None
    elif type(rotary) is not RotaryEmbedding:
        return None
    cache = rotary.cos_sin_cache
    seq_lens = tuple(int(n) for n in forward_batch.seq_lens_cpu)
    extend_lens = tuple(int(n) for n in forward_batch.extend_seq_lens_cpu)
    rows = sum(extend_lens)
    if (not rotary.is_neox_style or cache.device != hidden_states.device or cache.shape[1] != rotary.rotary_dim
            or rotary.rotary_dim != 64 or not cache.is_contiguous() or cache.dtype not in (torch.bfloat16, torch.float32)
            or rows == 0 or len(seq_lens) != len(extend_lens) or len(seq_lens) != metadata.sequence_lengths.numel()
            or rows != metadata.token_to_batch_idx.numel() or max(seq_lens) > cache.shape[0]
            or max(seq_lens) > metadata.token_slot_table.shape[1]
            or max(seq_lens) // module.compress_ratio > MAX_COMPRESSED_KEYS or hidden_states.shape[0] < rows
            or positions.shape[-1] < rows or forward_batch.positions.numel() < rows):
        return None
    logical = forward_batch.positions.flatten()[:rows]
    positions = positions[:, :rows] if positions.ndim == 2 else positions[:rows]
    slots = metadata.pending_ring_slots
    if slots is None:
        slots = module._pending_ring_slots(metadata, logical, True)
    rope = metadata.extend_rope_matrix
    if rope is None:
        rope = build_rope_position_matrix(positions, rows)
    project = projection(module, rows) if projection is not None else None
    qk = project(hidden_states[:rows]) if project is not None else module.index_qk_proj(hidden_states[:rows])[0]
    pool, layer = metadata.token_to_kv_pool, module.layer_id
    return dict(qk=qk.contiguous(), heads=4, positions=positions, logical_positions=logical.contiguous(),
                state_slots=slots[:rows].contiguous(), key_state=pool.get_qsa_key_state_buffer(layer),
                rope_state=pool.qsa_rope_position_buffer, write_locs=metadata.write_locs,
                member_rows=metadata.compress_member_rows, group_sequences=metadata.compress_sequence_ids,
                group_ends=metadata.compress_group_positions, rope_matrix=rope[:rows].contiguous(),
                compressed=pool.get_qsa_compressed_k_buffer(layer), token_slot_table=metadata.token_slot_table,
                cos_sin_cache=cache, axis_map=module._rope_axis_map(hidden_states.device),
                q_weight=module.q_layernorm.weight.data, k_weight=module.k_layernorm.weight.data,
                q_eps=module.q_layernorm.variance_epsilon, k_eps=module.k_layernorm.variance_epsilon,
                seq_lens=seq_lens, extend_lens=extend_lens)


def _indexer_state(metadata, layer):
    """Clone the pool rows an extend call may write (dump ring rows 0..ratio-1 are inert and skipped)."""
    import torch

    pool, ratio = metadata.token_to_kv_pool, metadata.compress_ratio
    lanes = torch.arange(ratio, device=metadata.write_locs.device)
    ring = (metadata.req_pool_indices.long()[:, None] * ratio + lanes).flatten()
    written = metadata.write_locs[metadata.write_locs != 0].long()
    return (pool.get_qsa_key_state_buffer(layer)[ring].clone(), pool.qsa_rope_position_buffer[ring].clone(),
            pool.get_qsa_compressed_k_buffer(layer)[written].clone())


def _indexer_compare(module, hidden_states, positions, forward_batch, metadata, key, actual, expected, before,
                     after):
    """Pool writes must match; rows whose token sets differ must be FP64 near-ties at the top-k boundary."""
    import torch

    rows, ratio, topk = actual.shape[0], module.compress_ratio, module.block_topk
    different = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1).nonzero().flatten()
    state = [bool(torch.equal(a, b)) for a, b in zip(before, after)]
    worst = 0.0
    if different.numel():
        logical = forward_batch.positions.flatten()[:rows]
        q, _, _ = module.project_qk(hidden_states[:rows], positions[..., :rows])
        keys, starts, ends, _ = metadata.get_prefill_mqa_inputs(module.layer_id, logical)
        for row in different[:256].tolist():
            start, end = int(starts[row]), int(ends[row])
            # Only the chosen blocks of rows with more than block_topk candidates may differ.
            if end - start <= topk or not torch.equal(actual[row, topk * ratio:], expected[row, topk * ratio:]):
                worst = math.inf
                continue
            values = torch.relu(q[row].double() @ keys[start:end, 0].double().T).sum(0) / math.sqrt(q.shape[-1])
            chosen = torch.zeros(end - start, dtype=torch.bool, device=values.device)
            chosen[actual[row, :topk * ratio:ratio].long() // ratio] = True
            gap = float(values[~chosen].max() - values[chosen].min()) / max(float(values.abs().max()), 1e-30)
            worst = max(worst, gap)
    result = dict(layer_id=module.layer_id, seq_lens=key[1], extend_lens=key[2], rows=rows,
                  different_token_sets=int(different.numel()), worst_relative_boundary_violation=worst,
                  key_state_equal=state[0], rope_state_equal=state[1], compressed_equal=state[2])
    if not all(state) or worst > 1e-5:
        raise AssertionError(f"PyHIP QSA indexer differs from SGLang: {result}")
    return result


def _indexer_snapshot(module, hidden_states, positions, forward_batch, metadata, output):
    """Clone one prefill indexer call: inputs, parameters, metadata and resulting pool rows."""
    import torch

    pool, layer, ratio = metadata.token_to_kv_pool, module.layer_id, module.compress_ratio
    tensors = {"hidden_states": hidden_states, "positions": positions, "batch_positions": forward_batch.positions,
               "output": output, "index_qk_weight": module.index_qk_proj.weight,
               "q_norm_weight": module.q_layernorm.weight, "k_norm_weight": module.k_layernorm.weight,
               "cos_sin_cache": module.rotary_emb.cos_sin_cache}
    fields = type(metadata).__struct_fields__
    tensors.update({"metadata." + name: getattr(metadata, name) for name in fields
                    if isinstance(getattr(metadata, name), torch.Tensor)})
    lanes = torch.arange(ratio, device=output.device)
    ring = torch.cat((lanes, (metadata.req_pool_indices.long()[:, None] * ratio + lanes).flatten()))
    lengths = [int(n) for n in forward_batch.seq_lens_cpu]
    slots = torch.cat([metadata.token_slot_table[s, : n // ratio * ratio : ratio].long() // ratio
                       for s, n in enumerate(lengths)])
    tensors.update(ring_rows=ring, ring_key_state=pool.get_qsa_key_state_buffer(layer)[ring],
                   ring_rope_positions=pool.qsa_rope_position_buffer[ring], compressed_slots=slots,
                   compressed_keys=pool.get_qsa_compressed_k_buffer(layer)[slots])
    rotary = module.rotary_emb
    meta = dict(layer_id=layer, rows=output.shape[0], forward_mode=str(forward_batch.forward_mode),
                seq_lens=lengths, extend_seq_lens=list(forward_batch.extend_seq_lens_cpu),
                extend_prefix_lens=list(forward_batch.extend_prefix_lens_cpu),
                index_n_heads=module.index_n_heads, index_head_dim=module.index_head_dim,
                compress_ratio=ratio, token_topk=module.token_topk, block_topk=module.block_topk,
                q_eps=module.q_layernorm.variance_epsilon, k_eps=module.k_layernorm.variance_epsilon,
                rotary_dim=rotary.rotary_dim, head_size=rotary.head_size, is_neox_style=rotary.is_neox_style,
                mrope_section=list(rotary.mrope_section), mrope_interleaved=rotary.mrope_interleaved,
                mrope_interleaved_glm=rotary.mrope_interleaved_glm, cos_sin_cache_dtype=str(rotary.cos_sin_cache.dtype),
                weight_type=type(module.index_qk_proj.weight.data).__name__,
                metadata_scalars={name: getattr(metadata, name) for name in fields
                                  if isinstance(getattr(metadata, name), (bool, int, float))})
    return meta, {name: value.detach().clone() for name, value in tensors.items()}


def _around_sparse(original, *args, chunk=False):
    global _state
    context = _call.get()
    if context is None:
        return original(*args)
    if _state is None:
        _state = _State()
    if chunk:
        q, k, v, indices, _, _, _, scale = args
    else:
        q, k, v, _, indices, _, scale = args
    return _state.execute(context, q, k, v, indices, scale, original, args)


def _around_indexer(original, module, hidden_states, positions, forward_batch, indexer_metadata):
    global _state
    if _state is None:
        _state = _State()
    return _state.indexer(original, module, hidden_states, positions, forward_batch, indexer_metadata)


def _around_decode(original, module, q, cache, table, lengths, width, positions, sequences):
    global _state
    if _state is None:
        _state = _State()
    return _state.decode(original, module, q, cache, table, lengths, width, positions, sequences)


def _profile_start(original, manager, *args, **kwargs):
    global _state
    result = original(manager, *args, **kwargs)
    if result is not None and result.success:
        if _state is None:
            _state = _State()
        _state.calls.clear()
        _state.indexer_calls.clear()
        _state.profiling = True
    return result


def _profile_stop(original, manager, *args, **kwargs):
    result = original(manager, *args, **kwargs)
    if result is not None and result.success and _state is not None and _state.profiling:
        _state.finish(manager.ps.tp_rank)
    return result


def register():
    if os.environ.get("PYHIP_QSA_PREFILL", "0") != "1":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for name, expected in _ABI.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"QSA plugin ABI mismatch: {name}")
    backend = "sglang.srt.layers.attention.qwen_sparse_attn_backend"
    profiler = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager"
    for name, hook in (
        (backend + ".QwenSparseAttnBackend.forward_extend", _around_forward),
        (backend + ".sparse_gqa_fwd_interface_triton", _around_sparse),
        (backend + ".sparse_gqa_fwd_interface_triton_ck", partial(_around_sparse, chunk=True)),
        (profiler + "._start_profile", _profile_start), (profiler + "._stop_profile", _profile_stop),
    ):
        HookRegistry.register(name, hook, HookType.AROUND)
    indexer = os.environ.get("PYHIP_QSA_INDEXER", "0")
    if indexer not in ("0", "1"):
        raise ValueError("PYHIP_QSA_INDEXER must be 0 or 1")
    decode = os.environ.get("PYHIP_QSA_INDEXER_DECODE", "0")
    if decode not in ("0", "select", "1"):
        raise ValueError("PYHIP_QSA_INDEXER_DECODE must be 0, select or 1")
    gemm = os.environ.get("PYHIP_QSA_INDEXER_GEMM", "hipblaslt")
    if gemm not in ("hipblaslt", "sglang"):
        raise ValueError("PYHIP_QSA_INDEXER_GEMM must be hipblaslt or sglang")
    if indexer == "1" or decode != "0" or os.environ.get("PYHIP_QSA_INDEXER_DUMP_LAYERS"):
        for name, expected in _INDEXER_ABI.items():
            spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"QSA indexer plugin ABI mismatch: {name}")
    if indexer == "1" or decode == "1" or os.environ.get("PYHIP_QSA_INDEXER_DUMP_LAYERS"):
        # Eager EXTEND (prefill indexer) and CUDA-graph DECODE (decode prep + selection) dispatch here.
        HookRegistry.register("sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.forward_cuda",
                              _around_indexer, HookType.AROUND)
    if decode != "0":
        # Remaining decode/target-verify selections, eager and inside CUDA graphs; FlyDSL/Triton compile
        # during SGLang's eager warmups before each capture.
        HookRegistry.register("sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens",
                              _around_decode, HookType.AROUND)
    logging.getLogger(__name__).warning("PyHIP QSA enabled: eager EXTEND, 3D only, %s; indexer=%s decode=%s gemm=%s",
                                        _POLICY, indexer, decode, gemm)


def build_target(target):
    """Build a local entry-point package without pip, wheels or source-tree imports."""
    source = Path(__file__).resolve().parent.parent
    root = source.parents[3]
    target = Path(os.path.abspath(target))
    if not target.resolve().is_relative_to((root / "mytest/mydata").resolve()):
        raise ValueError("Plugin targets must be new directories under mytest/mydata")
    target.mkdir(parents=True, exist_ok=False)
    package = target / "pyhip_qsa_runtime"
    files = {f"qsa/{name}.py": source / f"{name}.py"
             for name in ("attention", "attention_prepare", "attention_dense", "attention_direct",
                          "_attention_direct_packed", "attention_union", "indexer",
                          "indexer_logits", "indexer_topk", "indexer_decode")}
    files.update({f"mha/{name}.py": source.parent / "mha" / f"{name}.py"
                  for name in ("_common", "mha_pa_bf16_256_linear_942")})
    files["qsa/sglang/plugin.py"] = Path(__file__)
    files["qsa/sglang/attention_validation.py"] = Path(__file__).with_name("attention_validation.py")
    manifest = {}
    for relative, origin in files.items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        manifest[relative] = hashlib.sha256(origin.read_bytes()).hexdigest()
    for relative in ("", "qsa", "qsa/sglang", "mha"):
        (package / relative / "__init__.py").write_text('"""Source-hashed PyHIP runtime; imported lazily by the SGLang plugin."""\n')
    (package / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(root / "LICENSE", package / "LICENSE.pyhip")
    shutil.copyfile(Path("/opt/sglang/LICENSE"), package / "LICENSE.sglang")
    metadata = target / "pyhip_sglang_qsa_plugin-0.2.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: pyhip-sglang-qsa-plugin\nVersion: 0.2.0\n")
    (metadata / "entry_points.txt").write_text("[sglang.srt.plugins]\npyhip_flydsl_qsa = pyhip_qsa_runtime.qsa.sglang.plugin:register\n")
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the temporary QSA SGLang entry-point package")
    parser.add_argument("--build-target", type=Path, required=True)
    print(build_target(parser.parse_args().build_target))
