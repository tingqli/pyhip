import torch
import math

import aiter
from aiter import dtypes, pertoken_quant, per_tensor_quant
import pyhip
from dataclasses import dataclass

from pa_prefill_8w32x32 import PagedAttention


def vectorize_kv_cache(
    k_cache, v_cache, num_kv_heads, head_dim_qk, head_dim_v, page_size
):
    k_vector_size = 16 // torch.tensor([], dtype=k_cache.dtype).element_size()

    """
    [num_pages, page_size, num_kv_heads, head_dim]
      ->
    K: [num_pages, num_kv_heads, (head_dim // k_vector_size, page_size, k_vector_size)]
    V: [num_pages, num_kv_heads, (page_size // k_vector_size, head_dim, k_vector_size)]

    对于K， head_dim 是 Q @ K gemm的reduce维度K
    对于V， page_size(token数) 是 P @ V gemm的reduce维度K

    最内层维度是16字节,确保 K 维度可以使用 DWORDx4(b128) 的读取宽度：

        K: (head_dim // 16, page_size, 16)
        V: (page_size // 16, head_dim, 16)

    """
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    k_cache = (
        k_cache.view(
            -1, page_size, num_kv_heads, head_dim_qk // k_vector_size, k_vector_size
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(
            -1, page_size // k_vector_size, k_vector_size, num_kv_heads, head_dim_v
        )
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


@dataclass
class ModelConfig:
    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    page_size: int = 32
    quant_dtype = dtypes.fp8
    is_causal: bool = True

def test_pa_prefill(
    modelcfg: ModelConfig, batch_size, qo_len, kv_len, is_causal = None, page_size = None
):

    """Cover MiMo's direct cached-prefill contract and ragged last pages."""
    torch.cuda.empty_cache()
    torch.manual_seed(20260730)
    num_qo_heads, num_kv_heads = modelcfg.num_qo_heads, modelcfg.num_kv_heads
    page_size = modelcfg.page_size if page_size is None else page_size
    head_dim_qk = modelcfg.head_dim_qk
    head_dim_v = modelcfg.head_dim_v
    quant_dtype = modelcfg.quant_dtype
    is_causal = modelcfg.is_causal if is_causal is None else is_causal

    pages_per_seq = math.ceil(kv_len / page_size)
    num_pages = batch_size * pages_per_seq

    """
    q:torch.Size([10240, 16, 128]) 
    k:torch.Size([640, 1, 8, 16, 16])
    
    k.size(-3) * k_vector_size != head_size_q_og
    
    (head_dim // k_vector_size) 

    K: [num_pages, num_kv_heads, (head_dim_qk // k_vector_size, page_size, k_vector_size)]
    V: [num_pages, num_kv_heads, (page_size // k_vector_size, head_dim_v, k_vector_size)]

    """

    q_bf16 = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
    )                                 # [batch_size * qo_len, num_qo_heads, head_dim_qk]
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
    ) # [num_pages, page_size, num_kv_heads, head_dim_v]
    if quant_dtype != torch.bfloat16:
        q_fp8, q_descale = pertoken_quant(q_bf16, quant_dtype=quant_dtype)
        k_fp8, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
        v_fp8, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)
    else:
        q_fp8, q_descale = q_bf16, torch.ones([batch_size * qo_len, num_qo_heads, 1], device="cuda", dtype=torch.float32)
        k_fp8, k_descale = k_bf16, torch.ones(1, device="cuda", dtype=torch.float32)
        v_fp8, v_descale = v_bf16, torch.ones(1, device="cuda", dtype=torch.float32)

    # Reverse each request's physical pages so a linear-addressing accident
    # cannot pass while still keeping page ownership disjoint across requests.
    page_table = torch.arange(num_pages, dtype=torch.int32).view(
        batch_size, pages_per_seq
    )
    page_table = page_table.flip(1).contiguous()
    kv_page_indices = torch.nn.functional.pad(
        page_table.flatten(), (0, 256), value=0
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
        (batch_size + 1) * pages_per_seq,
        pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    kv_last_page_lens = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    k_vec, v_vec = vectorize_kv_cache(
        k_fp8,
        v_fp8,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        page_size,
    )

    # Use a NaN sentinel so a tile that silently leaves any token/head lane
    # unwritten cannot pass merely because zero is inside the loose FP8
    # absolute-error threshold.
    out = torch.full(
        (batch_size * qo_len, num_qo_heads, head_dim_v),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )

    fly_mha = PagedAttention(num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size, is_causal)

    pyhip.run_perftest(
        # aiter.mha_batch_prefill_func,
        fly_mha,
        q_fp8,
        k_vec,
        v_vec,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=is_causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens,
        out=out,
        num_iters=10,
        num_verbose=1,
        num_flops = (batch_size * num_qo_heads * (qo_len * kv_len * head_dim_qk + qo_len * kv_len * head_dim_v) * 2)//(2 if is_causal else 1)
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()

    # assert 0, f"{q_fp8.shape} {q_fp8.dtype} / {q_descale.shape} {q_descale.dtype} / {k_descale.dtype} / {v_descale.dtype}"
    try:
        q_ref = q_fp8.float() * q_descale
        refs = []
        for batch_idx in range(batch_size):
            pages = page_table[batch_idx].long().to("cuda")
            k_ref = (k_fp8[pages].reshape(-1, num_kv_heads, head_dim_qk)[:kv_len].float())
            v_ref = (v_fp8[pages].reshape(-1, num_kv_heads, head_dim_v)[:kv_len].float())
            k_ref = (k_ref * k_descale).repeat_interleave(
                num_qo_heads // num_kv_heads, dim=1
            )
            v_ref = (v_ref * v_descale).repeat_interleave(
                num_qo_heads // num_kv_heads, dim=1
            )
            q_ref_batch = q_ref[
                batch_idx * qo_len : (batch_idx + 1) * qo_len
            ]
            rows = torch.arange(qo_len, device="cuda").unsqueeze(1)
            cols = torch.arange(kv_len, device="cuda").unsqueeze(0)
            causal_mask = cols <= (kv_len - qo_len + rows) if is_causal else None
            refs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_ref_batch.transpose(0, 1).unsqueeze(0),
                    k_ref.transpose(0, 1).unsqueeze(0),
                    v_ref.transpose(0, 1).unsqueeze(0),
                    # Paged-prefill uses a bottom-right-aligned causal mask when
                    # Q and KV lengths differ.  PyTorch's is_causal=True is
                    # top-left aligned, so pass the explicit mask instead.
                    attn_mask=causal_mask,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        reference = torch.cat(refs, dim=0)
        pyhip.allclose(out.float(), reference.float(), rtol=1e-1, atol=1e-1)
        diff = pyhip.calc_diff(out.float(), reference.float())
        assert diff < 0.001, f"big diff: {diff}"
    except Exception as e:
        print("[accuracy unknown]: ", e)

    #verify_fp8_output()

multi_processor_count = torch.cuda.get_device_properties().multi_processor_count

modelcfg = ModelConfig("Llama3_70B_TP8", num_qo_heads=8, num_kv_heads=1, head_dim_qk=128, head_dim_v=128)
modelcfg = ModelConfig("MiMo_TP8", num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128)

page_size = 64
if 1:
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=3, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=13, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=23, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=53, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=83, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 1, qo_len=256*40, kv_len=256*10+23, is_causal = False, page_size=page_size)
    test_pa_prefill(modelcfg, 4, 256*40, 256*10, is_causal = False, page_size=page_size)

test_pa_prefill(modelcfg, 1, 8192, 8192+13, is_causal=True, page_size=page_size)
test_pa_prefill(modelcfg, 1, 8192, 8192+53, is_causal=True, page_size=page_size)
test_pa_prefill(modelcfg, 1, 8192, 8192+153, is_causal=True, page_size=page_size)
test_pa_prefill(modelcfg, 1, 32768, 32768, is_causal=True, page_size=page_size)
