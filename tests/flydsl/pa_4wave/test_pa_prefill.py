import math
import os
from dataclasses import dataclass

import torch

from aiter import dtypes, per_tensor_quant, pertoken_quant

import pyhip
from pa_prefill_4wave import MHA


def vectorize_kv_cache(
    k_cache,
    v_cache,
    num_kv_heads,
    head_dim_qk,
    head_dim_v,
    page_size,
):
    vector_size = 16 // k_cache.element_size()
    k_cache = (
        k_cache.contiguous()
        .view(
            -1,
            page_size,
            num_kv_heads,
            head_dim_qk // vector_size,
            vector_size,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.contiguous()
        .view(
            -1,
            page_size // vector_size,
            vector_size,
            num_kv_heads,
            head_dim_v,
        )
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
    quant_dtype = dtypes.fp8


MIMO_TP8 = ModelConfig(
    "MiMo_TP8",
    num_qo_heads=16,
    num_kv_heads=1,
    head_dim_qk=192,
    head_dim_v=128,
)


def run_formal_benchmark(
    kernel,
    q,
    k,
    v,
    cu_seqlens_q,
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
            q_buffers[buffer_index],
            k_buffers[buffer_index],
            v_buffers[buffer_index],
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q=qo_len,
            max_seqlen_k=kv_len,
            causal=causal,
            q_descale=q_descale_buffers[buffer_index],
            k_descale=k_descale,
            v_descale=v_descale,
            kv_last_page_lens=kv_last_page_lens,
            out=output_buffers[buffer_index],
        )

    for iteration in range(num_warmup):
        launch(iteration % num_buffers)
    torch.cuda.synchronize()

    samples_us = []
    for iteration in range(num_samples):
        with pyhip.cudaPerf(flops=flops, name="pa_prefill_4wave", verbose=0) as perf:
            launch(iteration % num_buffers)
        samples_us.append(perf.dt() * 1e6)

    samples_us.sort()
    median_us = samples_us[num_samples // 2]
    median_tflops = flops * 1e-6 / median_us
    print(
        f"[formal] median={median_us:.3f} us tflops={median_tflops:.3f} "
        f"min={samples_us[0]:.3f} us max={samples_us[-1]:.3f} us"
    )
    return median_us, median_tflops


def run_pa_prefill(
    model_config,
    batch_size,
    qo_len,
    kv_len,
    causal,
    num_iters=10,
):
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
        batch_size * qo_len,
        num_qo_heads,
        head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_bf16 = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_bf16 = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
    k_fp8, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
    v_fp8, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)

    page_table = torch.arange(num_pages, dtype=torch.int32).view(
        batch_size,
        pages_per_sequence,
    )
    page_table = page_table.flip(1).contiguous()
    kv_page_indices = torch.nn.functional.pad(
        page_table.flatten(),
        (0, 256),
        value=0,
    ).to("cuda")
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * qo_len,
        qo_len,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_sequence,
        pages_per_sequence,
        dtype=torch.int32,
        device="cuda",
    )
    kv_last_page_lens = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    k_vectorized, v_vectorized = vectorize_kv_cache(
        k_fp8,
        v_fp8,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        page_size,
    )
    output = torch.full(
        (batch_size * qo_len, num_qo_heads, head_dim_v),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )

    kernel = MHA(
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        page_size,
        causal,
    )
    flops = (
        batch_size
        * num_qo_heads
        * (qo_len * kv_len * head_dim_qk + qo_len * kv_len * head_dim_v)
        * 2
    ) // (2 if causal else 1)
    print(f"[case] batch={batch_size} qo={qo_len} kv={kv_len} causal={causal}")
    pyhip.run_perftest(
        kernel,
        q_fp8,
        k_vectorized,
        v_vectorized,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens,
        out=output,
        num_iters=num_iters,
        num_verbose=1,
        num_flops=flops,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()

    q_reference = q_fp8.float() * q_descale
    references = []
    for batch_index in range(batch_size):
        pages = page_table[batch_index].long().to("cuda")
        k_reference = (
            k_fp8[pages].reshape(-1, num_kv_heads, head_dim_qk)[:kv_len].float()
            * k_descale
        )
        v_reference = (
            v_fp8[pages].reshape(-1, num_kv_heads, head_dim_v)[:kv_len].float()
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

    if os.environ.get("PA_FORMAL_BENCH") == "1":
        run_formal_benchmark(
            kernel,
            q_fp8,
            k_vectorized,
            v_vectorized,
            cu_seqlens_q,
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
        )
    return diff


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
    }
    selected = os.environ.get("PA_CASE", "all")
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
    for batch_size, qo_len, kv_len, causal in selected_cases:
        run_pa_prefill(
            MIMO_TP8,
            batch_size,
            qo_len,
            kv_len,
            causal,
            num_iters=num_iters,
        )


if __name__ == "__main__":
    main()
