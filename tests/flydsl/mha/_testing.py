"""Shared paged inputs, independent FP32 oracle and backend contracts.

No kernel imports or AITER initialization occur during test collection.
"""

from dataclasses import dataclass
import importlib
from itertools import accumulate
import math

import torch


def gpu_arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if torch.cuda.is_available() else "cpu"


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
    single_dispatch: bool = True

    def load(self):
        return importlib.import_module((__package__ + "." if __package__ else "") + self.module)

    @property
    def available(self):
        return self.arch == gpu_arch() or self.arch == "both" and gpu_arch() in ("gfx942", "gfx950")

    @property
    def fp8(self):
        return self.dtype == torch.float8_e4m3fnuz


FP8 = Backend("fp8_942", "mha_pa_fp8_942", "gfx942", torch.float8_e4m3fnuz)
FP8_REG = Backend("fp8_942_register", "mha_pa_fp8_942", "gfx942", torch.float8_e4m3fnuz, memory_mode="register")
BF16_942 = Backend("bf16_942", "mha_pa_bf16_942", "gfx942", torch.bfloat16,
                   empty_kv=False, causal_short_kv=False, single_dispatch=False)
BF16_950 = Backend("bf16_950", "mha_pa_bf16_950", "gfx950", torch.bfloat16, strided=True)
BF16_950_PERSISTENT = Backend("bf16_950_persistent", "mha_pa_bf16_950", "gfx950", torch.bfloat16,
                              persistent=True, strided=True)
SWA = Backend("swa_bf16", "mha_pa_swa_bf16", "both", torch.bfloat16, strided=True)
BACKENDS = (FP8, FP8_REG, BF16_942, BF16_950, BF16_950_PERSISTENT)
PRIMARY_BACKENDS = (FP8, BF16_942, BF16_950)


def i32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def vectorize_kv(k_pages, v_pages):
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

    def pack(self, *, copy=False):
        k, v = vectorize_kv(self.k_pages, self.v_pages)
        if copy:
            self.k.copy_(k)
            self.v.copy_(v)
        else:
            self.k, self.v = k, v

    def logical_kv(self):
        keys, values = [], []
        pos = self.table_offset
        for length in self.kv_lens:
            count = (length + self.page - 1) // self.page
            ids = self.page_order[pos:pos + count]
            # Cast before indexing: some PyTorch builds lack FP8 index kernels.
            keys.append(self.k_pages.float()[ids].reshape(-1, self.kv_heads, self.dq)[:length])
            values.append(self.v_pages.float()[ids].reshape(-1, self.kv_heads, self.dv)[:length])
            pos += count
        return keys, values


def make_case(q_lens=(256,), kv_lens=(256,), *, dtype=torch.bfloat16, dq=192, dv=128,
              page=64, heads=4, kv_heads=1, mode="per-token", layout="contiguous",
              q_offset=0, table_offset=0, nonunit_scales=False, magnitude=1.0,
              window_left=-1, has_sink=False, poison_tail=True, reverse_pages=False,
              quantized=False, seed=20260906, source_dtype=torch.float32, padding_before_quantization=False):
    assert len(q_lens) == len(kv_lens)
    torch.manual_seed(seed)
    q_lens, kv_lens = tuple(q_lens), tuple(kv_lens)
    tokens = q_offset + sum(q_lens) + (7 if q_offset else 0)
    q = torch.randn(tokens, heads, dq, device="cuda", dtype=source_dtype) * magnitude
    counts = [(n + page - 1) // page for n in kv_lens]
    pages = max(1, sum(counts))
    k_pages = torch.randn(pages, page, kv_heads, dq, device="cuda", dtype=source_dtype) * magnitude
    v_pages = torch.randn(pages, page, kv_heads, dv, device="cuda", dtype=source_dtype)
    order = None
    if padding_before_quantization:
        # The original README benchmark zeroed logical tail pages before
        # computing per-tensor FP8 maxima. Keep that input protocol explicit.
        order = list(reversed(range(sum(counts)))) if reverse_pages else torch.randperm(sum(counts)).tolist()
        order = [0] * table_offset + order
        pos = table_offset
        for length, count in zip(kv_lens, counts):
            if length % page:
                k_pages[order[pos + count - 1], length % page:] = 0
                v_pages[order[pos + count - 1], length % page:] = 0
            pos += count
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
    if order is None:
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


def torch_reference(case, causal, softmax_scale=None):
    """FP32 bottom-right attention; no tolerance/exception suppression."""
    output = torch.full((case.q.shape[0], case.heads, case.dv), float("nan"), device=case.q.device)
    lse = torch.full(case.q.shape[:2], float("nan"), device=case.q.device)
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
                lse[offset + start:offset + end] = case.sinks if case.sinks is not None else -float("inf")
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
            probabilities = logits.softmax(-1).nan_to_num(0.0)
            if case.sinks is not None:
                probabilities = probabilities[..., :-1]
            output[offset + start:offset + end] = (probabilities @ v[:, first:last]).transpose(0, 1)
            lse[offset + start:offset + end] = logits.logsumexp(-1).transpose(0, 1)
        offset += q_len
    return output, lse


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


def make_call(case, backend, causal, *, out=None, lse=None, **options):
    if out is None:
        out = output_buffer(case)[0]
    if backend != SWA:
        options.setdefault("memory_mode", backend.memory_mode)
        options.setdefault("persistent", backend.persistent)
    kernel = backend.load().PagedAttention(case.heads, case.kv_heads, case.dq, case.dv, case.page,
        causal, case.mode, window_left=case.window_left, has_sink=case.sinks is not None, **options)

    def call(**runtime):
        kwargs = {"out": out, "lse": lse, "sink_ptr": case.sinks,
                  "max_seqlen_q": max(case.q_lens, default=0), "max_seqlen_k": max(case.kv_lens, default=0)}
        kwargs.update(runtime)
        max_q, max_k = kwargs.pop("max_seqlen_q"), kwargs.pop("max_seqlen_k")
        return kernel(case.q, case.k, case.v, case.cq, case.ck, case.indptr, case.indices,
                      max_q, max_k, causal, case.qs, case.ks, case.vs, case.last, **kwargs)
    return call, out, kernel


def assert_close(case, backend, actual, lse, causal, *, softmax_scale=None, lse_atol=None):
    reference, ref_lse = torch_reference(case, causal, softmax_scale)
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    tolerance = 0.1 if backend.fp8 else 0.02
    torch.testing.assert_close(actual[begin:end].float(), reference[begin:end], rtol=tolerance, atol=tolerance)
    if lse is not None:
        atol = lse_atol if lse_atol is not None else (8e-4 if backend.fp8 else 5e-4)
        torch.testing.assert_close(lse[begin:end], ref_lse[begin:end], rtol=3e-4 if backend.fp8 else 2e-4, atol=atol)
    return reference, ref_lse


def assert_case(case, backend, causal, *, layout="contiguous", softmax_scale=None, repeats=3, with_lse=True, lse_atol=None, **options):
    out, backing = output_buffer(case, layout)
    lse = torch.full(case.q.shape[:2], -123.0, device="cuda", dtype=torch.float32) if with_lse else None
    call, _, kernel = make_call(case, backend, causal, out=out, lse=lse, **options)
    first = None
    for _ in range(repeats):
        result = call(return_lse=with_lse, softmax_scale=softmax_scale)
        if with_lse:
            assert result[0] is out and result[1] is lse
        else:
            assert result is out
        if first is None:
            first = out.clone(), None if lse is None else lse.clone()
        else:
            torch.testing.assert_close(out, first[0], rtol=0, atol=0)
            if lse is not None:
                torch.testing.assert_close(lse, first[1], rtol=0, atol=0)
    assert_close(case, backend, out, lse, causal, softmax_scale=softmax_scale, lse_atol=lse_atol)
    begin, end = case.q_offset, case.q_offset + sum(case.q_lens)
    assert (out[:begin] == -123).all() and (out[end:] == -123).all()
    if lse is not None:
        assert (lse[:begin] == -123).all() and (lse[end:] == -123).all()
    if layout == "padded":
        assert (backing[:, case.heads:] == -123).all() and (backing[:, :case.heads, case.dv:] == -123).all()
    return out, lse


def dispatch_names(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        call()
        torch.cuda.synchronize()
    return [e.name for e in profiler.events() if "CUDA" in str(e.device_type)]


def effective_flops(case, causal):
    pairs = 0
    for q, k in zip(case.q_lens, case.kv_lens):
        if not causal:
            pairs += q * k
        else:
            pairs += sum(max(0, min(k, k - q + r + 1) - (max(0, k - q + r - case.window_left) if case.window_left >= 0 else 0)) for r in range(q))
    return 2 * case.heads * pairs * (case.dq + case.dv)