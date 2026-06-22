"""
Test that the recurrent (decode / single-token) gated delta rule produces the
same output as the chunked prefill reference when run token-by-token over the
full sequence.

The recurrent function is:
    transformers/.../modeling_qwen3_next.py::torch_recurrent_gated_delta_rule

The prefill reference is:
    test_gdn_prefill.py::torch_chunk_gated_delta_rule

Both are pure-PyTorch; no Triton kernels involved.

Run:
    python test_gdn_decode.py
"""

import torch
import torch.nn.functional as F

CHUNK_SIZE = 64


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Token-by-token recurrent decode (from modeling_qwen3_next.py L454)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                     dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=CHUNK_SIZE,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Chunked prefill reference (from transformers modeling_qwen3_next.py L373)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    B, H, L, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - L % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    Lp = L + pad

    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(B, H, Dk, Dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, Lp // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :L]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def make_inputs(B, T, H, K, V, dtype, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=gen)
    beta = torch.randn(B, T, H, device=device, dtype=dtype, generator=gen).sigmoid()
    g_logits = torch.randn(B, T, H, device=device, dtype=torch.float32, generator=gen)
    g = F.logsigmoid(g_logits)

    h0 = torch.randn(B, H, K, V, device=device, dtype=dtype, generator=gen) * 0.01

    return q, k, v, g, beta, h0


def run_case(B, T, H, K, V, dtype=torch.float32, seed=0, label="",
             rtol=1e-4, atol=1e-4):
    device = torch.device("cuda", 0)
    q, k, v, g, beta, h0 = make_inputs(B, T, H, K, V, dtype, device, seed)

    o_decode, s_decode = torch_recurrent_gated_delta_rule(
        query=q, key=k, value=v, g=g, beta=beta,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    o_prefill, s_prefill = torch_chunk_gated_delta_rule(
        query=q, key=k, value=v, g=g, beta=beta,
        chunk_size=CHUNK_SIZE,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    header = f"[{label}] B={B} T={T} H={H} K={K} V={V} dtype={dtype}"
    print(header)

    of = o_decode.float()
    rf = o_prefill.float()
    diff = (of - rf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / rf.abs().clamp_min(1e-8)).max().item()
    cos = F.cosine_similarity(of.flatten(), rf.flatten(), dim=0).item()
    print(f"  output : max|d|={max_abs:.3e}  mean|d|={mean_abs:.3e}  "
          f"max_rel={max_rel:.3e}  cos={cos:.6f}")

    sf = s_decode.float()
    srf = s_prefill.float()
    sdiff = (sf - srf).abs()
    s_max_abs = sdiff.max().item()
    s_mean_abs = sdiff.mean().item()
    print(f"  state  : max|d|={s_max_abs:.3e}  mean|d|={s_mean_abs:.3e}")

    torch.testing.assert_close(of, rf, rtol=rtol, atol=atol)
    torch.testing.assert_close(sf, srf, rtol=rtol, atol=atol)
    print("  -> OK\n")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    run_case(B=1, T=128,  H=1, K=16,  V=16,  label="small")
    run_case(B=1, T=512,  H=8, K=128, V=128, seed=42, label="qwen3.5_moe_like")
    run_case(B=2, T=256,  H=4, K=64,  V=128, label="K!=V")

    print("All cases passed: decode ≡ prefill")


if __name__ == "__main__":
    main()
