"""Shared MHA test fixtures: paged/linear inputs, FP32 references and correctness checks.

Used by the unit tests next to this file and by benchmarks/mha. Collection and
listing never initialize a GPU; kernels are imported from pyhip.ops.mha.flydsl
on first use.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
from itertools import accumulate
import math
import os
import re
import sys
import tempfile
import warnings

import pytest
import torch


def gpu_arch():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


def require(*arches):
    """Select MHA_TEST_GPU (a visible device index, default 0); skip unless its arch is in `arches`."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires ROCm " + " or ".join(arches))
    torch.cuda.set_device(int(os.environ.get("MHA_TEST_GPU", "0")))
    if gpu_arch() not in arches:
        pytest.skip("requires " + " or ".join(arches))
    return torch.device("cuda", torch.cuda.current_device())


# ---- Paged kernels: backend contracts and workloads ----

@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    arch: str
    dtype: torch.dtype
    memory_mode: str = "lds"
    persistent: bool | None = None
    strided: bool = False
    empty_kv: bool = True
    causal_short_kv: bool = True

    def load(self):
        return importlib.import_module("pyhip.ops.mha.flydsl." + self.module)

    @property
    def available(self):
        return self.arch == gpu_arch() or self.arch == "both" and gpu_arch() in ("gfx942", "gfx950")

    @property
    def fp8(self):
        return self.dtype == torch.float8_e4m3fnuz


FP8 = Backend("fp8_942", "mha_pa_fp8_942", "gfx942", torch.float8_e4m3fnuz)
BF16_942 = Backend("bf16_942", "mha_pa_bf16_942", "gfx942", torch.bfloat16,
                   empty_kv=False, causal_short_kv=False)
BF16_942_GRID = Backend("bf16_942_grid", "mha_pa_bf16_942", "gfx942", torch.bfloat16,
                       persistent=False, empty_kv=False, causal_short_kv=False)
BF16_950 = Backend("bf16_950", "mha_pa_bf16_950", "gfx950", torch.bfloat16, strided=True)
BF16_950_PERSISTENT = Backend("bf16_950_persistent", "mha_pa_bf16_950", "gfx950", torch.bfloat16,
                              persistent=True, strided=True)
SWA = Backend("swa_bf16", "mha_pa_swa_bf16", "both", torch.bfloat16, strided=True)


@dataclass(frozen=True)
class Workload:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    dq: int = 192
    dv: int = 128
    heads: int = 16
    kv_heads: int = 1
    page: int = 64
    causal: bool = False
    window: int = -1
    sink: bool = False
    scale_mode: str = "per-token"
    seed: int = 20260905

    @property
    def flops(self):
        pairs = 0
        for q, k in zip(self.q_lens, self.kv_lens):
            if not self.causal:
                pairs += q * k
            else:
                for row in range(q):
                    diagonal = k - q + row
                    left = max(0, diagonal - self.window) if self.window >= 0 else 0
                    pairs += max(0, min(k, diagonal + 1) - left)
        return 2 * self.heads * pairs * (self.dq + self.dv)

    def unsupported(self, backend):
        if self.window >= 0 and not self.causal:
            return "windowed attention requires bottom-right causal mode"
        if backend.name == "swa_bf16":
            if self.window < 0 or not self.causal:
                return "single-wave backend requires causal SWA"
        elif self.window >= 0 or self.sink:
            if backend.arch != "gfx950":
                return "this full-MHA backend does not support SWA/sink"
        if backend.module != "mha_pa_bf16_942" and self.page != 64:
            return "backend supports page64 only"
        if backend.module != "mha_pa_bf16_942" and self.dv != 128:
            return "backend supports V128 only"
        if not backend.empty_kv and any(q > 0 and k == 0 for q, k in zip(self.q_lens, self.kv_lens)):
            return "BF16 gfx942 requires nonempty active KV"
        if self.causal and not backend.causal_short_kv and any(k < q for q, k in zip(self.q_lens, self.kv_lens)):
            return "BF16 gfx942 causal attention requires KV>=Q per sequence"
        return None

    def to_dict(self):
        return {**asdict(self), "effective_flops": self.flops}


# ---- Paged inputs (SHUFFLE-5D KV) and the independent FP32 oracle ----

def i32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def vectorize_kv(k_pages, v_pages):
    """[P,S,H,D] pages to SHUFFLE-5D: K[P,H,D/v,S,v], V[P,H,S/v,D,v] with v = 16 bytes."""
    pages, page, heads, dq = k_pages.shape
    dv = v_pages.shape[-1]
    vector = 16 // k_pages.element_size()
    k = k_pages.reshape(pages, page, heads, dq // vector, vector).permute(0, 2, 3, 1, 4).contiguous()
    v = v_pages.reshape(pages, page // vector, vector, heads, dv).permute(0, 3, 1, 4, 2).contiguous()
    return k, v


def quantize(values, dtype, *, per_token=False):
    scale = values.float().abs().amax(-1, keepdim=True) if per_token else values.float().abs().amax().reshape(1)
    scale = scale / torch.finfo(dtype).max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values.float() / scale).to(dtype), scale.float()


@dataclass
class Case:
    q: torch.Tensor
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cq: torch.Tensor
    ck: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    vs: torch.Tensor
    q_lens: tuple
    kv_lens: tuple
    page_order: list
    q_offset: int
    table_offset: int
    mode: str
    window_left: int
    sinks: torch.Tensor | None

    @property
    def heads(self):
        return self.q.shape[1]

    @property
    def kv_heads(self):
        return self.k_pages.shape[2]

    @property
    def dq(self):
        return self.q.shape[-1]

    @property
    def dv(self):
        return self.v_pages.shape[-1]

    @property
    def page(self):
        return self.k_pages.shape[1]

    def logical_kv(self):
        keys, values = [], []
        pos = self.table_offset
        k_pages, v_pages = self.k_pages.float(), self.v_pages.float()
        for length in self.kv_lens:
            count = (length + self.page - 1) // self.page
            ids = self.page_order[pos:pos + count]
            keys.append(k_pages[ids].reshape(-1, self.kv_heads, self.dq)[:length])
            values.append(v_pages[ids].reshape(-1, self.kv_heads, self.dv)[:length])
            pos += count
        return keys, values


def make_case(q_lens=(256,), kv_lens=(256,), *, dtype=torch.bfloat16, dq=192, dv=128,
              page=64, heads=4, kv_heads=1, mode="per-token", layout="contiguous",
              q_offset=0, table_offset=0, nonunit_scales=False, magnitude=1.0,
              window_left=-1, has_sink=False, poison_tail=True, reverse_pages=False,
              quantized=False, seed=20260906, source_dtype=torch.float32):
    """Random paged inputs; shuffled page table, optional prefix padding and NaN-poisoned page tails."""
    assert len(q_lens) == len(kv_lens)
    torch.manual_seed(seed)
    q_lens, kv_lens = tuple(q_lens), tuple(kv_lens)
    tokens = q_offset + sum(q_lens) + (7 if q_offset else 0)
    q = torch.randn(tokens, heads, dq, device="cuda", dtype=source_dtype) * magnitude
    counts = [(n + page - 1) // page for n in kv_lens]
    pages = max(1, sum(counts))
    k_pages = torch.randn(pages, page, kv_heads, dq, device="cuda", dtype=source_dtype) * magnitude
    v_pages = torch.randn(pages, page, kv_heads, dv, device="cuda", dtype=source_dtype)
    if quantized:
        q, qs = quantize(q, dtype, per_token=mode == "per-token")
        k_pages, ks = quantize(k_pages, dtype)
        v_pages, vs = quantize(v_pages, dtype)
    else:
        q, k_pages, v_pages = q.to(dtype), k_pages.to(dtype), v_pages.to(dtype)
        qs = torch.ones((tokens, heads, 1) if mode == "per-token" else (1,), device="cuda")
        ks, vs = torch.ones(1, device="cuda"), torch.ones(1, device="cuda")
    if nonunit_scales:
        qs.uniform_(0.5, 1.25)
        ks.fill_(0.75)
        vs.fill_(1.25)
    if layout == "padded":
        storage = torch.zeros(tokens, heads + 1, dq + 16, device="cuda", dtype=dtype)
        storage[:, :heads, :dq].copy_(q)
        q = storage[:, :heads, :dq]
    elif layout == "head-major":
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    elif layout != "contiguous":
        raise ValueError(layout)
    order = list(reversed(range(sum(counts)))) if reverse_pages else torch.randperm(sum(counts)).tolist()
    order = [0] * table_offset + order
    pos = table_offset
    for length, count in zip(kv_lens, counts):
        if length % page:
            k_pages[order[pos + count - 1], length % page:] = float("nan") if poison_tail else 0
            v_pages[order[pos + count - 1], length % page:] = float("nan") if poison_tail else 0
        pos += count
    k, v = vectorize_kv(k_pages, v_pages)
    return Case(q, k_pages, v_pages, k, v,
                i32(list(accumulate(q_lens, initial=q_offset))),
                i32(list(accumulate(kv_lens, initial=11 if table_offset else 0))),
                i32(list(accumulate(counts, initial=table_offset))), i32(order),
                i32([(n - 1) % page + 1 if n else 0 for n in kv_lens]), qs, ks, vs,
                q_lens, kv_lens, order, q_offset, table_offset, mode, window_left,
                torch.linspace(-1, 1, heads, device="cuda") if has_sink else None)


def torch_reference(case, causal, softmax_scale=None, *, return_lse=False):
    """Independent FP32 bottom-right attention; optional natural-log LSE."""
    output = torch.full((case.q.shape[0], case.heads, case.dv), float("nan"), device=case.q.device)
    lse = torch.full(case.q.shape[:2], float("nan"), device=case.q.device) if return_lse else None
    scale = case.dq**-0.5 if softmax_scale is None else softmax_scale
    keys, values = case.logical_kv()
    offset = case.q_offset
    for q_len, kv_len, k, v in zip(case.q_lens, case.kv_lens, keys, values):
        k = (k * case.ks).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        v = (v * case.vs).repeat_interleave(case.heads // case.kv_heads, 1).transpose(0, 1)
        for start in range(0, q_len, 256):
            end = min(start + 256, q_len)
            if kv_len == 0:
                output[offset + start:offset + end] = 0
                if return_lse:
                    lse[offset + start:offset + end] = -float("inf")
                continue
            q = case.q[offset + start:offset + end].float()
            qs = case.qs if case.qs.numel() == 1 else case.qs[offset + start:offset + end]
            first, last = 0, kv_len
            if case.window_left >= 0:
                first = max(0, kv_len - q_len + start - case.window_left)
                last = max(0, min(kv_len, kv_len - q_len + end))
            logits = ((q * qs).transpose(0, 1) @ k[:, first:last].transpose(1, 2)) * scale
            if causal:
                diagonal = kv_len - q_len + torch.arange(start, end, device=q.device)[:, None]
                cols = torch.arange(first, last, device=q.device)[None, :]
                mask = cols > diagonal
                if case.window_left >= 0:
                    mask |= cols < diagonal - case.window_left
                logits.masked_fill_(mask, -float("inf"))
            if case.sinks is not None:
                logits = torch.cat((logits, case.sinks[:, None, None].expand(-1, end - start, 1)), -1)
            if return_lse:
                lse[offset + start:offset + end] = logits.logsumexp(-1).transpose(0, 1)
            # Only rows without a visible key output 0; any other NaN must propagate.
            probabilities = logits.softmax(-1).masked_fill(torch.isneginf(logits).all(-1, keepdim=True), 0.0)
            if case.sinks is not None:
                probabilities = probabilities[..., :-1]
            output[offset + start:offset + end] = (probabilities @ v[:, first:last]).transpose(0, 1)
        offset += q_len
    return (output, lse) if return_lse else output


def output_buffer(case, layout="contiguous"):
    shape = (case.q.shape[0], case.heads, case.dv)
    if layout == "padded":
        backing = torch.full((shape[0], shape[1] + 1, shape[2] + 16), -123, device="cuda", dtype=torch.bfloat16)
        return backing[:, :shape[1], :shape[2]], backing
    if layout == "head-major":
        backing = torch.full((shape[1], shape[0], shape[2]), -123, device="cuda", dtype=torch.bfloat16)
        return backing.transpose(0, 1), backing
    out = torch.full(shape, -123, device="cuda", dtype=torch.bfloat16)
    return out, out


def make_call(case, backend, causal, *, out=None, **options):
    """A closure calling the backend's PagedAttention on `case`; runtime keywords override defaults."""
    if out is None:
        out = output_buffer(case)[0]
    if backend != SWA:
        options.setdefault("memory_mode", backend.memory_mode)
        options.setdefault("persistent", backend.persistent)
    kernel = backend.load().PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
        causal, case.mode, window_left=case.window_left, has_sink=case.sinks is not None, **options)

    def call(**runtime):
        kwargs = {"out": out, "lse": None, "sink_ptr": case.sinks,
                  "max_seqlen_q": max(case.q_lens, default=0), "max_seqlen_k": max(case.kv_lens, default=0)}
        kwargs.update(runtime)
        max_q, max_k = kwargs.pop("max_seqlen_q"), kwargs.pop("max_seqlen_k")
        return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                      max_q, max_k, causal, case.qs, case.ks, case.vs, case.last, **kwargs)
    return call, out, kernel


# ---- D256 linear/page1/page4 inputs and FP32 reference ----

def linear_inputs(device, *, qlens=(0, 7, 33, 65), klens=(3, 33, 65, 129), page=None, heads=12, kv_heads=1,
                  causal=True, persistent=True, scale=0.0625, magnitude=1.0, poison=True, reference=True,
                  seed=20260924):
    """flash_attn_varlen_func keyword arguments and the FP32 (O, natural-log LSE) reference.

    With `page`, logical KV rows live in reversed physical pages (NaN elsewhere when `poison`).
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((sum(qlens), heads, 256), generator=generator, device=device, dtype=torch.bfloat16)
    keys = [torch.randn((n, kv_heads, 256), generator=generator, device=device, dtype=q.dtype) for n in klens]
    values = [torch.randn(k.shape, generator=generator, device=device, dtype=q.dtype) for k in keys]
    if magnitude != 1.0:
        q.mul_(magnitude)
        for key in keys:
            key.mul_(magnitude)
    cq, ck = [0], [0]
    for n, k in zip(qlens, klens):
        cq.append(cq[-1] + n)
        ck.append(ck[-1] + k)
    table = None
    if page is None:
        k, v = torch.cat(keys), torch.cat(values)
    else:
        counts = [math.ceil(n / page) for n in klens]
        count = sum(counts)
        k = torch.full((count * page, kv_heads, 256), float("nan") if poison else 0.0, device=device, dtype=q.dtype)
        v = torch.full_like(k, float("nan") if poison else 0.0)
        table = torch.full((len(klens), max(counts) + 1), -1, device=device, dtype=torch.int32)
        cursor = 0
        for sequence, n in enumerate(klens):
            pages = torch.arange(count - cursor - 1, count - cursor - counts[sequence] - 1, -1, device=device)
            table[sequence, :counts[sequence]] = pages.to(torch.int32)
            rows = torch.arange(n, device=device)
            physical = pages[rows // page] * page + rows % page
            k[physical], v[physical] = keys[sequence], values[sequence]
            cursor += counts[sequence]
    arguments = dict(q=q, k=k, v=v,
                     cu_seqlens_q=torch.tensor(cq, dtype=torch.int32, device=device),
                     cu_seqlens_k=torch.tensor(ck, dtype=torch.int32, device=device),
                     max_seqlen_q=max(qlens), max_seqlen_k=max(klens),
                     block_table=table, page_size=page or 1, causal=causal,
                     persistent=persistent, softmax_scale=scale, return_lse=True)
    if not reference:
        return arguments, None
    return arguments, linear_reference(q, keys, values, cq, causal, scale)


@torch.no_grad()
def linear_reference(q, keys, values, cq, causal, scale, chunk=1024):
    """FP32 attention per sequence and query chunk (TF32 off): (O [T,H,256], LSE [T,H])."""
    heads, kv_heads = q.shape[1], keys[0].shape[1] if keys else 1
    output = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for sequence, (key, value) in enumerate(zip(keys, values)):
            n, length = cq[sequence + 1] - cq[sequence], key.shape[0]
            g = heads // kv_heads
            key = key.float().repeat_interleave(g, dim=1)
            value = value.float().repeat_interleave(g, dim=1)
            for begin in range(0, n, chunk):
                end = min(begin + chunk, n)
                rows = slice(cq[sequence] + begin, cq[sequence] + end)
                scores = torch.einsum("mhd,nhd->hmn", q[rows].float(), key) * scale
                if causal:
                    mask = (torch.arange(length, device=q.device)[None]
                            > torch.arange(begin, end, device=q.device)[:, None] + length - n)
                    scores.masked_fill_(mask[None], -float("inf"))
                lse[rows] = scores.logsumexp(-1).T
                output[rows] = torch.einsum("hmn,nhd->mhd", scores.softmax(-1), value)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return output, lse


# ---- Correctness checks shared by unit tests and benchmarks ----

def accuracy(actual, reference, label, tolerance=0.02):
    """acc = sum((ref-out)^2) / sum(ref^2+out^2) (pyhip.calc_diff); zero is best.

    The elementwise tolerance is checked too: a scalar average must not hide a few bad rows.
    """
    from pyhip import calc_diff
    value = float(calc_diff(reference, actual, diff_thr=-1))
    finite = math.isfinite(value) and bool(torch.isfinite(actual).all())
    if not finite:
        raise AssertionError(f"{label} acc={value}: nonfinite output")
    try:
        torch.testing.assert_close(actual.float(), reference.float(), rtol=tolerance, atol=tolerance)
    except AssertionError as exc:
        raise AssertionError(f"{label} acc={value:.8g} FAIL (rtol=atol={tolerance})\n{exc}") from exc
    return value


@contextmanager
def _profiler_diagnostics():
    """Drop only native profiler start/stop progress lines on stderr; replay everything on failure."""
    sys.stderr.flush()
    with os.fdopen(os.dup(2), "wb", buffering=0) as original, tempfile.TemporaryFile() as captured:
        failed = True
        try:
            os.dup2(captured.fileno(), 2)
            yield
            failed = False
        finally:
            sys.stderr.flush()
            os.dup2(original.fileno(), 2)
            captured.seek(0)
            for line in captured:
                progress = re.match(rb"^USDT:.*ActivityProfilerController\.cpp:\d+\]\s+profiler_(start|stop)\s*$", line)
                if failed or not progress:
                    original.write(line)


def dispatch_names(call):
    """GPU kernel names launched by one call (torch profiler, untimed)."""
    with _profiler_diagnostics(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Warning: Profiler clears events at the end of each cycle\.",
                                category=UserWarning, module=r"torch\.profiler\.profiler")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            call()
            torch.cuda.synchronize()
        return [e.name for e in profiler.events() if "CUDA" in str(e.device_type)]


def check_outputs(calls, references, *, label, tolerance=0.02):
    """Check every candidate on every buffer: accuracy, two bit-exact repeats, no aliasing, a dispatch.

    calls: name -> per-buffer callables returning the output; returns name -> result rows.
    """
    rows = {}
    for name, pool in calls.items():
        accs, outputs = [], []
        for index, (call, reference) in enumerate(zip(pool, references)):
            actual = call()
            accs.append(accuracy(actual, reference, f"{label}/{name}/buffer{index}", tolerance))
            outputs.append(actual.data_ptr())
            first = actual.clone()
            for _ in range(2):
                torch.testing.assert_close(call(), first, rtol=0, atol=0)
        if len(set(outputs)) != len(pool):
            raise AssertionError(f"{name}: output buffers alias")
        kernels = dispatch_names(pool[0])
        if not kernels and first.numel():
            raise AssertionError(f"{name}: expected a native attention dispatch")
        rows[name] = {"backend": name, "acc": max(accs), "acc_per_buffer": accs, "repeated_bit_exact": True,
                      "output_ptrs": outputs, "kernels": kernels}
    return rows


def paged_inputs(workload, backends, *, buffers=10, layout="contiguous", nonunit_scales=False,
                 softmax_scale=None, poison_tail=False):
    """`buffers` independent cases of a workload, FP32 references and per-backend callables."""
    supported = list(backends)
    if not supported:
        raise ValueError("a case must explicitly select at least one backend")
    for backend in supported:
        if not backend.available or workload.unsupported(backend):
            raise ValueError(f"{workload.name}/{backend.name}: "
                             f"{workload.unsupported(backend) or 'unsupported architecture'}")
    if len({b.dtype for b in supported}) != 1:
        raise ValueError("a case cannot mix BF16 and FP8")
    if layout != "contiguous" and any(not b.strided for b in supported):
        raise ValueError("the selected backend requires contiguous Q/O")
    w, backend = workload, supported[0]
    cases, references, calls = [], [], {}
    for index in range(buffers):
        case = make_case(w.q_lens, w.kv_lens, dtype=backend.dtype, dq=w.dq, dv=w.dv, page=w.page,
                         heads=w.heads, kv_heads=w.kv_heads, mode=w.scale_mode, layout=layout,
                         window_left=w.window, has_sink=w.sink, nonunit_scales=nonunit_scales,
                         poison_tail=poison_tail, quantized=backend.fp8, source_dtype=torch.bfloat16,
                         seed=w.seed + index)
        cases.append(case)
        references.append(torch_reference(case, w.causal, softmax_scale))
        for candidate in supported:
            out, _ = output_buffer(case, layout)
            call = make_call(case, candidate, w.causal, out=out)[0]
            calls.setdefault(candidate.name, []).append(lambda call=call: call(softmax_scale=softmax_scale))
    return cases, references, calls


@torch.inference_mode()
def check_paged(workload, backends, **options):
    """Correctness of each backend on `buffers` independent inputs (no timing)."""
    _, references, calls = paged_inputs(workload, backends, **options)
    tolerance = 0.1 if backends[0].fp8 else 0.02
    label = f"{workload.name} Q{workload.q_lens} KV{workload.kv_lens} W{workload.window}"
    return check_outputs(calls, references, label=label, tolerance=tolerance)
