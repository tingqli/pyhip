import math
import os
import statistics
from dataclasses import dataclass

import pytest
import torch

import pyhip
from pa_prefill_4wave import MHA


FP8_DTYPE = torch.float8_e4m3fnuz


def pertoken_quant(x, scale=None, quant_dtype=FP8_DTYPE):
    x_f32 = x.float()
    if scale is None:
        scale = x_f32.abs().amax(dim=-1, keepdim=True) / torch.finfo(
            quant_dtype
        ).max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x_f32 / scale).to(quant_dtype), scale.float()


def per_tensor_quant(x, scale=None, quant_dtype=FP8_DTYPE):
    x_f32 = x.float()
    if scale is None:
        scale = x_f32.abs().max() / torch.finfo(quant_dtype).max
    return (x_f32 / scale).to(quant_dtype), scale.reshape(1).float()


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="requires gfx942",
)


def vectorize_kv_cache(k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size):
    vector_size = 16 // k_cache.element_size()
    k_cache = (
        k_cache.contiguous()
        .view(-1, page_size, num_kv_heads, head_dim_qk // vector_size, vector_size)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.contiguous()
        .view(-1, page_size // vector_size, vector_size, num_kv_heads, head_dim_v)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    page_size: int = 32
    quant_dtype: torch.dtype = FP8_DTYPE


MIMO_TP8 = ModelConfig("MiMo_TP8", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128)
MIMO_BF16 = ModelConfig(
    "MiMo_BF16", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
BF16_REF = ModelConfig(
    "BF16_REF", num_qo_heads=1, num_kv_heads=1, head_dim_qk=128, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
H3_BF16 = ModelConfig(
    "MiniMax_H3", num_qo_heads=14, num_kv_heads=14, head_dim_qk=128, head_dim_v=128,
    quant_dtype=torch.bfloat16,
)
H3_SEGMENTS = (63225, 7)


def attention_flops(segments, num_heads, head_dim_qk, head_dim_v):
    return sum(
        2 * length * length * (head_dim_qk + head_dim_v) * num_heads
        for length in segments
    )


def run_formal_benchmark(
    kernel,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_indptr,
    kv_page_indices,
    qo_len,
    kv_len,
    causal,
    q_descale,
    k_descale,
    v_descale,
    kv_last_page_lens,
    output,
    flops,
    name="pa_prefill_4wave",
):
    num_buffers = 10
    num_warmup = 10
    num_samples = 50
    q_buffers = [q.clone() for _ in range(num_buffers)]
    k_buffers = [k.clone() for _ in range(num_buffers)]
    v_buffers = [v.clone() for _ in range(num_buffers)]
    q_descale_buffers = [q_descale.clone() for _ in range(num_buffers)]
    output_buffers = [torch.empty_like(output) for _ in range(num_buffers)]

    def launch(buffer_index):
        kernel(
            q_buffers[buffer_index], k_buffers[buffer_index], v_buffers[buffer_index],
            cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
            max_seqlen_q=qo_len, max_seqlen_k=kv_len, causal=causal,
            q_descale=q_descale_buffers[buffer_index], k_descale=k_descale, v_descale=v_descale,
            kv_last_page_lens=kv_last_page_lens, out=output_buffers[buffer_index],
        )

    for iteration in range(num_warmup):
        launch(iteration % num_buffers)
    torch.cuda.synchronize()

    samples_us = []
    for iteration in range(num_samples):
        with pyhip.cudaPerf(flops=flops, name=name, verbose=0) as perf:
            launch(iteration % num_buffers)
        samples_us.append(perf.dt() * 1e6)

    samples_us.sort()
    median_us = samples_us[num_samples // 2]
    median_tflops = flops * 1e-6 / median_us
    print(
        f"[formal:{name}] median={median_us:.3f} us tflops={median_tflops:.3f} "
        f"min={samples_us[0]:.3f} us max={samples_us[-1]:.3f} us"
    )
    return median_us, median_tflops


def make_h3_inputs(quant_dtype=torch.bfloat16):
    """Build the real H3 varlen pack in the paged-KV ABI used by this kernel."""
    assert quant_dtype in (torch.bfloat16, FP8_DTYPE)
    generator = torch.Generator(device="cuda").manual_seed(1101)
    segments = H3_SEGMENTS
    num_qo_heads = H3_BF16.num_qo_heads
    num_kv_heads = H3_BF16.num_kv_heads
    head_dim = H3_BF16.head_dim_qk
    page_size = H3_BF16.page_size
    total_tokens = sum(segments)

    shape = (total_tokens, num_qo_heads, head_dim)
    q_bf16, k_packed, v_packed = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )

    pages_per_sequence = [(length + page_size - 1) // page_size for length in segments]
    num_pages = sum(pages_per_sequence)
    k_pages = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_pages = torch.zeros_like(k_pages)
    page_base = 0
    token_base = 0
    for length, page_count in zip(segments, pages_per_sequence):
        padded_length = page_count * page_size
        k_pages[page_base : page_base + page_count].view(padded_length, num_kv_heads, head_dim)[:length].copy_(
            k_packed[token_base : token_base + length]
        )
        v_pages[page_base : page_base + page_count].view(padded_length, num_kv_heads, head_dim)[:length].copy_(
            v_packed[token_base : token_base + length]
        )
        page_base += page_count
        token_base += length

    if quant_dtype == torch.bfloat16:
        q_input, k_input, v_input = q_bf16, k_pages, v_pages
        q_descale = torch.ones(total_tokens, num_qo_heads, 1, device="cuda", dtype=torch.float32)
        k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    else:
        q_input, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_input, k_descale = per_tensor_quant(k_pages, quant_dtype=quant_dtype)
        v_input, v_descale = per_tensor_quant(v_pages, quant_dtype=quant_dtype)

    k_input, v_input = vectorize_kv_cache(
        k_input, v_input, num_kv_heads, head_dim, head_dim, page_size
    )
    kv_page_indices = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    )
    kv_indptr = torch.tensor(
        [0, *torch.tensor(pages_per_sequence).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    kv_last_page_lens = torch.tensor(
        [(length - 1) % page_size + 1 for length in segments], device="cuda", dtype=torch.int32
    )
    output = torch.empty_like(q_bf16)
    kernel = MHA(num_qo_heads, num_kv_heads, head_dim, head_dim, page_size, False)

    def launch():
        kernel(
            q_input, k_input, v_input, cu_seqlens, cu_seqlens, kv_indptr, kv_page_indices,
            max_seqlen_q=max(segments), max_seqlen_k=max(segments), causal=False,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            kv_last_page_lens=kv_last_page_lens, out=output,
        )

    return q_bf16, k_packed, v_packed, cu_seqlens, output, launch


def run_h3_benchmark(dtype="bf16"):
    """Run the real MiniMax-H3 varlen pack with the AITER benchmark protocol."""
    quant_dtype = torch.bfloat16 if dtype == "bf16" else FP8_DTYPE
    q, _, _, _, output, launch = make_h3_inputs(quant_dtype)
    segments = H3_SEGMENTS
    num_qo_heads = H3_BF16.num_qo_heads
    head_dim = H3_BF16.head_dim_qk

    for _ in range(3):
        launch()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        samples_ms.append(start.elapsed_time(stop))

    flops = attention_flops(segments, num_qo_heads, head_dim, head_dim)
    assert flops == 28_653_368_031_232
    median_ms = statistics.median(samples_ms)
    tflops = flops / 1e9 / median_ms
    print(
        f"[h3] dtype={dtype} segments={segments} heads={num_qo_heads} dim={head_dim} "
        f"flops={flops / 1e12:.6f} TFLOP"
    )
    print(
        f"[h3:{dtype}:4wave] median={median_ms:.3f} ms min={min(samples_ms):.3f} ms "
        f"max={max(samples_ms):.3f} ms tflops={tflops:.3f}"
    )
    print("[h3:protocol] warmup=3 samples=10 timing=CUDA-event aggregation=statistics.median")
    print("[h3:formula] FLOPs=sum(4 * S_i^2 * head_dim * heads); TFLOPS=FLOPs/(median_ms*1e9)")
    assert torch.isfinite(output).all()
    return median_ms, tflops


def run_pa_prefill(model_config, batch_size, qo_len, kv_len, causal, num_iters=10):
    torch.manual_seed(20260730)
    num_qo_heads = model_config.num_qo_heads
    num_kv_heads = model_config.num_kv_heads
    head_dim_qk = model_config.head_dim_qk
    head_dim_v = model_config.head_dim_v
    page_size = model_config.page_size
    quant_dtype = model_config.quant_dtype

    pages_per_sequence = math.ceil(kv_len / page_size)
    num_pages = batch_size * pages_per_sequence
    q_bf16 = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim_qk, device="cuda", dtype=torch.bfloat16
    )
    k_bf16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim_qk, device="cuda", dtype=torch.bfloat16
    )
    v_bf16 = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim_v, device="cuda", dtype=torch.bfloat16
    )
    if quant_dtype == torch.bfloat16:
        q_input, k_input, v_input = q_bf16, k_bf16, v_bf16
        q_descale = torch.ones((batch_size * qo_len, num_qo_heads, 1), device="cuda", dtype=torch.float32)
        k_descale = torch.ones(1, device="cuda", dtype=torch.float32)
        v_descale = torch.ones(1, device="cuda", dtype=torch.float32)
    else:
        q_input, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_input, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
        v_input, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)

    page_table = torch.arange(num_pages, dtype=torch.int32).view(batch_size, pages_per_sequence)
    page_table = page_table.flip(1).contiguous()
    kv_page_indices = page_table.flatten().to("cuda")
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * qo_len, qo_len, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device="cuda"
    )
    kv_indptr = torch.arange(
        0, (batch_size + 1) * pages_per_sequence, pages_per_sequence, dtype=torch.int32, device="cuda"
    )
    kv_last_page_lens = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )
    k_vectorized, v_vectorized = vectorize_kv_cache(
        k_input, v_input, num_kv_heads, head_dim_qk, head_dim_v, page_size
    )
    output = torch.full(
        (batch_size * qo_len, num_qo_heads, head_dim_v), float("nan"), device="cuda", dtype=torch.bfloat16
    )

    kernel = MHA(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, causal)
    flops = (
        batch_size
        * num_qo_heads
        * (qo_len * kv_len * head_dim_qk + qo_len * kv_len * head_dim_v)
        * 2
    ) // (2 if causal else 1)
    dtype_name = "bf16" if quant_dtype == torch.bfloat16 else "fp8"
    print(f"[case] dtype={dtype_name} batch={batch_size} qo={qo_len} kv={kv_len} causal={causal}")
    pyhip.run_perftest(
        kernel, q_input, k_vectorized, v_vectorized, cu_seqlens_q, cu_seqlens_k,
        kv_indptr, kv_page_indices,
        max_seqlen_q=qo_len, max_seqlen_k=kv_len, causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens, out=output,
        num_iters=num_iters, num_verbose=1, num_flops=flops,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()

    reference = None
    diff = None
    if os.environ.get("PA_SKIP_REFERENCE") != "1":
        q_reference = q_input.float() * q_descale
        references = []
        for batch_index in range(batch_size):
            pages = page_table[batch_index].long().to("cuda")
            k_reference = (
                k_input[pages].reshape(-1, num_kv_heads, head_dim_qk)[:kv_len].float()
                * k_descale
            )
            v_reference = (
                v_input[pages].reshape(-1, num_kv_heads, head_dim_v)[:kv_len].float()
                * v_descale
            )
            repeat = num_qo_heads // num_kv_heads
            k_reference = k_reference.repeat_interleave(repeat, dim=1)
            v_reference = v_reference.repeat_interleave(repeat, dim=1)
            q_batch = q_reference[batch_index * qo_len : (batch_index + 1) * qo_len]
            rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
            columns = torch.arange(kv_len, device="cuda").unsqueeze(0)
            causal_mask = columns <= (kv_len - qo_len + rows) if causal else None
            references.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_batch.transpose(0, 1).unsqueeze(0),
                    k_reference.transpose(0, 1).unsqueeze(0),
                    v_reference.transpose(0, 1).unsqueeze(0),
                    attn_mask=causal_mask,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )

        reference = torch.cat(references, dim=0)
        pyhip.allclose(output.float(), reference.float(), rtol=0.1, atol=0.1)
        diff = pyhip.calc_diff(output.float(), reference.float())
        print(f"[accuracy] diff={diff:.8f}")
        assert diff < 0.001, f"big diff: {diff}"
    else:
        print("[accuracy] reference skipped")

    if os.environ.get("PA_FORMAL_BENCH") == "1":
        run_formal_benchmark(
            kernel, q_input, k_vectorized, v_vectorized, cu_seqlens_q, cu_seqlens_k,
            kv_indptr, kv_page_indices,
            qo_len, kv_len, causal, q_descale, k_descale, v_descale, kv_last_page_lens, output, flops,
            name="4wave",
        )
    return diff


@pytest.mark.parametrize(
    ("model_config", "batch_size", "qo_len", "kv_len", "causal"),
    [
        (BF16_REF, 2, 129, 83, False),
        (BF16_REF, 1, 129, 129, True),
        (MIMO_TP8, 2, 128, 83, False),
    ],
)
def test_accuracy(model_config, batch_size, qo_len, kv_len, causal):
    diff = run_pa_prefill(
        model_config,
        batch_size,
        qo_len,
        kv_len,
        causal,
        num_iters=1,
    )
    assert diff is not None and diff < 0.001


def main():
    cases = {
        "short": [(1, 128, 3, False)],
        "tails": [
            (1, 128, 3, False),
            (1, 128, 13, False),
            (1, 128, 23, False),
            (1, 128, 53, False),
            (1, 128, 83, False),
        ],
        "noncausal": [(1, 256 * 40, 256 * 10 + 23, False)],
        "batch": [(4, 256 * 40, 256 * 10, False)],
        "causal": [(1, 32768, 32768, True)],
        "bf16_ref_short": [(1, 128, 83, False)],
        "bf16_ref": [(1, 40960, 40960, False)],
    }
    selected = os.environ.get("PA_CASE", "all")
    dtype = os.environ.get("PA_DTYPE", "bf16" if selected == "h3" else "fp8")
    if dtype not in ("fp8", "bf16"):
        raise ValueError(f"unknown PA_DTYPE={dtype!r}; expected 'fp8' or 'bf16'")
    if selected == "h3":
        run_h3_benchmark(dtype)
        return
    if selected == "all":
        selected_cases = [
            *cases["tails"],
            *cases["noncausal"],
            *cases["batch"],
            *cases["causal"],
        ]
    else:
        if selected not in cases:
            raise ValueError(
                f"unknown PA_CASE={selected!r}; expected one of {sorted(cases)} or 'all'"
            )
        selected_cases = cases[selected]

    num_iters = int(os.environ.get("PA_NUM_ITERS", "10"))
    if selected in ("bf16_ref_short", "bf16_ref"):
        model_config = BF16_REF
    else:
        model_config = MIMO_BF16 if dtype == "bf16" else MIMO_TP8
    for batch_size, qo_len, kv_len, causal in selected_cases:
        run_pa_prefill(model_config, batch_size, qo_len, kv_len, causal, num_iters=num_iters)


if __name__ == "__main__":
    main()
