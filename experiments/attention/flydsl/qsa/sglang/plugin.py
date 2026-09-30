"""Temporary SGLang hook adapter; the QSA runtime itself has no SGLang dependency."""

from collections import Counter
from contextvars import ContextVar
from functools import partial
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil


_call = ContextVar("pyhip_qsa_forward", default=None)
_state = None
_ABI = {
    "sglang.srt.layers.attention.qwen_sparse_attn_backend": "c75dbabacea6e9012428e7dbe695fc49400406fed77272752dce82f9cd8b6113",
    "sglang.srt.layers.attention.qsa.kernel": "7e369f09293fb9b0872c21f0010247ec1e3a696b5ad4809f04d9a730b1031095",
    "sglang.srt.layers.attention.qsa.qsa_indexer": "37547d9535934961c233f0c5d6ac5a5b94a08ba325c573857c2aebd9b1593257",
}


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
        self.runtime = importlib.import_module("..qsa", __package__)
        self.profiling = False
        self.calls = Counter()
        self.checked = set()
        self.checks = []
        self.pending = {}
        self.dump_layers = {int(n) for n in os.environ.get("PYHIP_QSA_DUMP_LAYERS", "").split(",") if n}
        self.dump_rows = {int(n) for n in os.environ.get("PYHIP_QSA_DUMP_ROWS", "12000,11888").split(",") if n}
        self.report_dir = Path(os.environ["PYHIP_QSA_REPORT_DIR"]) if "PYHIP_QSA_REPORT_DIR" in os.environ else None

    def execute(self, context, q, k, v, indices, scale, original, args):
        import torch

        backend, layer, queries, prefixes = context
        if (any(not t.is_contiguous() or t.data_ptr() % 16 or t.ndim != 3 or t.shape[-1] != 256 for t in (q, k, v))
                or indices.dtype != torch.int32 or not indices.is_contiguous()):
            return original(*args)
        output = self.runtime.qsa(q, k, v, indices, query_lens=queries, prefix_lens=prefixes, softmax_scale=scale)
        key = (layer.layer_id, queries, prefixes)
        if self.profiling:
            self.calls[str(layer.layer_id)] += 1
            if layer.layer_id in self.dump_layers and sum(queries) in self.dump_rows and key not in self.pending:
                with torch.profiler.record_function("pyhip_qsa.capture_inputs"):
                    # Host lengths + token indices fully determine replay metadata;
                    # do not couple the adapter to the private workspace cache.
                    tensors = {name: tensor.detach().clone() for name, tensor in (
                        ("q", q), ("k", k), ("v", v), ("indices", indices), ("output", output))}
                self.pending[key] = (dict(layer_id=layer.layer_id, tp_rank=backend.runner.ps.tp_rank,
                                          query_lens=queries, prefix_lens=prefixes, scale=scale), tensors)
        elif os.environ.get("PYHIP_QSA_VALIDATE", "0") == "1" and key not in self.checked:
            expected = original(*args)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            self.checked.add(key)
            self.checks.append(dict(layer_id=layer.layer_id, query_lens=queries, prefix_lens=prefixes,
                                    rows=output.shape[0], rtol=0.02, atol=0.02))
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
        if not self.calls:
            raise RuntimeError("Profile captured zero QSA replacement calls")
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            with (self.report_dir / f"qsa_tp{rank}.json").open("x") as stream:
                json.dump({"calls_per_layer": dict(self.calls), "validation": self.checks,
                           "input_snapshots": snapshots, "policy": "auto4/dense2051/sortedBN32"}, stream, indent=2)
        self.pending.clear()
        self.profiling = False


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


def _profile_start(original, manager, *args, **kwargs):
    global _state
    result = original(manager, *args, **kwargs)
    if result is not None and result.success:
        if _state is None:
            _state = _State()
        _state.calls.clear()
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
    logging.getLogger(__name__).warning("PyHIP QSA enabled: eager EXTEND, auto4/dense2051/sortedBN32")


def build_target(target):
    """Build a local entry-point package without pip, wheels or source-tree imports."""
    source = Path(__file__).resolve().parent.parent
    root = source.parents[3]
    target = Path(os.path.abspath(target))
    if not target.resolve().is_relative_to((root / "mytest/mydata").resolve()):
        raise ValueError("Plugin targets must be new directories under mytest/mydata")
    target.mkdir(parents=True, exist_ok=False)
    package = target / "pyhip_qsa_runtime"
    files = {f"qsa/{name}.py": source / f"{name}.py" for name in ("qsa", "dense", "direct", "union")}
    files.update({f"mha/{name}.py": source.parent / "mha" / f"{name}.py"
                  for name in ("_common", "mha_pa_bf16_256_linear_942")})
    files["qsa/sglang/plugin.py"] = Path(__file__)
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
