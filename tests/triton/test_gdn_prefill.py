"""
Port of ``chunk_gated_delta_rule_fwd`` (Triton orchestrator for the chunk-wise
gated delta rule used by Qwen3.5-MoE / Qwen3-Next linear attention) into
``pyhip/tests/gluon``.

Source: ``sglang/srt/layers/attention/fla/chunk.py::chunk_gated_delta_rule_fwd``
(adapted from flash-linear-attention).

The Triton sub-kernels themselves live in sglang (cumsum / chunk_fwd_intra /
chunk_delta_h / chunk_fwd_o). They are imported as-is; this file re-defines the
orchestrator inline and exercises it with synthetic tensors whose shapes match
the parameter spec documented in the source docstring.

Run:
    python test_chunk_gated_delta_rule_fwd.py
"""

from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton kernel imports (the actual JIT'd kernels live in sglang).
# ---------------------------------------------------------------------------
from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum

# Chunk length C used throughout the chunk-wise derivation.
CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Ported orchestrator (verbatim from sglang chunk.py, minus the SUPPRESS_LEVEL
# branch so the return shape is fixed).
# ---------------------------------------------------------------------------
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,  # the state shape should be (B, H, Dv, Dk)
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
):
    r"""Chunk-wise gated delta rule forward.

    With  alpha_t = exp(g_t), gamma_j = prod_{r<=j} alpha_r,
    Gamma_{i,j} = gamma_i/gamma_j for j<=i  (per chunk [t] of length C):

        L         = strictLower( diag(beta) ( Gamma * K K^T ) )         (Eq. L)
        U~_{[t]}  = (I + L)^{-1} diag(beta) V                            (Eq. U~)
        W<-_{[t]} = (I + L)^{-1} diag(beta) diag(gamma_i) K              (Eq. W)
        DeltaV    = U~_{[t]} - W<-_{[t]} S_{[t]}^T                       (Eq. DV)
        S_{[t+1]} = gamma_C * S_{[t]} + K->_{[t]}^T DeltaV               (Eq. S)
        O_{[t]}   = Q<-_{[t]} S_{[t]}^T + (Q K^T * Gamma_{[t]}) DeltaV   (Eq. O)

    Pipeline:
        1. chunk_local_cumsum                -> G_i  (so gamma_i = exp(G_i))
        2. chunk_gated_delta_rule_fwd_intra  -> A=(I+L)^{-1}, w=W<-, u=U~
        3. chunk_gated_delta_rule_fwd_h      -> v_new=DeltaV, h[t]=S_{[t]}
        4. chunk_fwd_o                       -> O_{[t]}
    """
    # Step 1: cumulative log-gates per chunk -> G_i.
    # parallel on [[B, H, chunks]]，每个CU负责C个元素的cumulative sum
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    # Step 2: fused kkt + solve_tril + recompute_w_u -> A, u, w.

    # chunk_gated_delta_rule_fwd_kkt_solve_kernel()
    # k_beta = k * beta
    # L = k_beta@k^T * decay_mask
    # L是下三角阵， 求（I  + L ）的逆，  (I + L)^（-1）
    # [B, Hv, N] parallel，每个CU负责 [64, Dk]@[Dk, 64], inverse the [64, 64] triangular matrix

    # recompute_w_u_fwd() — [B, H, N] parallel. Each chunk: [64, 64]@[64, Dv] → Ũ, [64, 64]@[64, Dk] → W←
    # U~= atten @ (beta * V)
    # W← = atten@ (beta* gamma_i* K)
    # `w`, [ B, H, N, 64, Dk]
    # 'u',  [B, Hv, N, 64, Dv]
    # both can parallel on [B, Hv, N]
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # Step 3: chunk-sequential memory recurrence -> v_new/deltaV 是所有的trunk的state[B, H, chunks, 64, Dv], h是所有trunk的state, shape [B, H, chunks, Dv, Dk]
    # compute the h_all, v_new_all for every chunk.
    # For each trunk:
    # v_new = U~ - W← @ h_in,           [64,Dv] - [64,Dk]@[Dk,Dv]
    # K-> = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]), i 是trunk index ,
    # h_in-> = h_in * gamma_C
    # h_out = h_in-> + K->^T@v_new     [Dk,Dv] + [Dk,64]@[64, Dv]
    # parallel on [B, Hv, V/Dv]. each WG needs to iterate all the chunks.
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # Step 4: per-chunk Ot.
    # [B, H, chunks] parallel, so need every chunk sstate(h_all) and every chunk new value to compute o in parallel
    # Ox = Q<- @ S[t], Q-< = (q * g.exp()), q的每个token与对应token的g相乘。[64, Dk]@[Dk, Dv] = [64, Dv]
    # Oy = (q @ k^T * decay_mask) @ v_new_all,   [64, Dk]@[Dk, 64]@[64, Dv]
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return g, o, A, w, h, v_new


# ---------------------------------------------------------------------------
# Tensor synthesis matching the documented parameter shapes
# (head_first=False, equal-length batch):
#
#   q              : [B, T, H, K]    bf16/fp16
#   k              : [B, T, H, K]    bf16/fp16    (L2-normalized along K)
#   v              : [B, T, H, V]    bf16/fp16
#   g              : [B, T, H]       fp32 log-decays
#   beta           : [B, T, H]       bf16/fp16    (in (0,1) via sigmoid)
#   initial_state  : [N, H, V, K]    bf16/fp16    (N == B for equal-length)
#   initial_state_indices : [N]      int64
#   cu_seqlens     : None for equal-length; else [N+1] int64 with B==1
#   scale          : float, defaults to 1/sqrt(K)
#
# T must be a multiple of CHUNK_SIZE (=64) for the chunked pipeline.
# ---------------------------------------------------------------------------
def make_inputs(B, T, H, K, V, dtype, device, seed=0):
    """Synthesize a valid (q, k, v, g, beta, h0, idx) tuple."""
    gen = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, generator=gen)
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(dtype)

    v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=gen)

    # beta in (0,1): sigmoid of a normal sample.
    beta = torch.randn(B, T, H, device=device, dtype=dtype, generator=gen).sigmoid()

    # g is the per-step log-decay; keep it negative-ish via logsigmoid.
    g_logits = torch.randn(B, T, H, device=device, dtype=torch.float32, generator=gen)
    g = torch.nn.functional.logsigmoid(g_logits)

    # Initial recurrent state S_0 of shape [B, H, V, K].
    h0 = torch.randn(B, H, V, K, device=device, dtype=dtype, generator=gen) * 0.01
    idx = torch.arange(B, device=device, dtype=torch.long)

    return q, k, v, g, beta, h0, idx


# ---------------------------------------------------------------------------
# PyTorch reference (lifted from transformers Qwen3.5-MoE
# ``torch_chunk_gated_delta_rule``, with the unused ``query_orig`` shadow
# removed). Operates entirely in fp32 internally and returns ``[B, L, H, V]``
# matching the sglang ``o`` output.
# ---------------------------------------------------------------------------
def _l2norm(x, dim=-1, eps=1e-6):
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=CHUNK_SIZE,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Reference impl (see modeling_qwen3_5_moe.torch_chunk_gated_delta_rule).
    Arguments:
        query:              : [B, L, H, K]   bf16/fp16
        key:                : [B, L, H, K]    bf16/fp16
        value:              : [B, L, H, V]    bf16/fp16
        g                   : [B, L, H]       fp32 log-decays
        beta                : [B, L, H]       bf16/fp16    (in (0,1) via sigmoid)
        initial_state       : [B, H, K, V]    bf16/fp16    (N == B for equal-length)
    Returns:
        Ot:        [B, L, H, Dv]
        last_recurrent_state: [B, H, Dk, Dv] or None
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)

    # transpose from [B, L , H, ...] to [B, H, L, ...] and make contiguous for the reference.
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
    # v_beta = diag(β)@V
    v_beta = value * beta.unsqueeze(-1)
    # k_beta = diag(β)@K
    k_beta = key * beta.unsqueeze(-1)

    # reorder from [[B, H, T, ...] to [B, H, chunk_num, chunk_size, ...]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    # mask是上三角矩阵，包含对角线
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    # g shape : [B, H, chunk_num, chunk_size],
    # 假设 chunk_size=4: 原始 g = [g0, g1, g2, g3]，衰减系数的对数。
    # 经过 G = g.cumsum(-1) 之后:  G = [g0, g0+g1, g0+g1+g2, g0+g1+g2+g3],
    # G[:]       ： 当前chunk从第一个token到每个token的累计衰减。
    # G[-1]      ： 当前chunk从第一个token到最后一个token累计衰减。
    # G[-1] -G[:]： 当前chunk从每个token到最后一个token的衰减。
    # 用decay_mask可以更加灵活的表达衰减的关系：
    # decay[i,j] = G_i - G_j  (G0 在差分中被消掉, 所以下三角里只剩 g1..g3 的累加):
    # decay = [[ 0,         -g1,        -(g1+g2),    -(g1+g2+g3)],
    #        [ g1,         0,         -g2,         -(g2+g3)   ],
    #        [ g1+g2,      g2,         0,          -g3        ],
    #        [ g1+g2+g3,   g2+g3,      g3,          0         ]]
    #
    # decay.tril() 清零严格上三角, 保留下三角和对角线:
    # decay.tril() = [[ 0,         0,      0,    0 ],
    #               [ g1,        0,      0,    0 ],
    #               [ g1+g2,     g2,     0,    0 ],
    #               [ g1+g2+g3,  g2+g3,  g3,   0 ]]
    #
    # 含义:decay_mask{i,j}= decay.tril()[i,j].exp().tril() . 当i > j是， 每个trunk里面第j个token到第i个token一共衰减了多少。
    # 对应spec: gamma_i = G_i.exp()?
    # 就是下三角的 decay_mask{i,j} = gamma_i/gamma_j (对角线=1).
    g = g.cumsum(dim=-1)
    # decay_mask下三角矩阵[64, 64]，对角线都为1. g = strict_low_triangular (exp(g.cumsum(-1)))
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    #  atten被mask成下三角矩阵，对角线为0，k_beta@key.transpose(-1, -2)输出[64, 64]矩阵，
    # 这个矩阵与decay_mask逐元素相乘，得到的attn是下三角矩阵（包含对角线），再mask掉上三角（包含对角线）得到严格下三角矩阵, 对角线为0
    # 这里attn is -L,
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    # attn is (I + L)^{-1}，shape是【64， 64】，严格下三角矩阵逐行求逆。
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # U~= (1+L)^-1 @ diag(beta) @ V
    # U~/U_Tilde shape还是[B, H, chunk_num, 64, V]
    U_Tilde = attn @ v_beta
    # W<- = (I + L)^{-1}@diag(beta)@diag(gamma_i)@K, k_beta[B, H, chunk_num, 64, K], g.unsqueeze(-1) [B, H, chunk_num, 64, 1]
    # W_Arrow： [B, H, chunk_num, 64, K]
    # W_Arrow is W<-
    W_Arrow = attn @ (k_beta * g.exp().unsqueeze(-1))

    # state = (B, H, Dk, Dv)
    last_recurrent_state = (
        torch.zeros(B, H, Dk, Dv, dtype=U_Tilde.dtype, device=U_Tilde.device)
        if initial_state is None
        else initial_state.to(U_Tilde)
    )
    Ot = torch.zeros_like(U_Tilde)
    # 上三角矩阵，对角线为0
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # deltaV= U~ - W<- @ h_in
    # h_out = h_in * gamma_C + K<-^T @ deltaV
    # 这里更新state是逐chunk更新的。
    for i in range(0, Lp // chunk_size):
        # v_i 这里是U~
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], U_Tilde[:, :, i]
        # Q @ Kt * Gamma,  *是 elementwise, 这里是为后面Oy做准备。
        # attn代表的token里面第i个元素和第j个元素的关系，自然要与衰减矩阵相关联。
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        # v_i是U~, v_prime是W<- @ S[t], deltaV= U~ - W<- @ S[t]
        # W<- @ S[t], W_Arrow： [B, H, chunk_num, 64, Dk],  state = (B, H, Dk, Dv)
        v_prime = W_Arrow[:, :, i] @ last_recurrent_state
        # U~ - W<- @ S[t]
        delta_V = v_i - v_prime

        # 上面的是为了计算O的准备工作，现在开始计算O
        # Ox = Q<- @ S[t], Q-< = (q_i * g[:, :, i, :, None].exp())
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        # Oy = Q @ Kt * Gamma @ DeltaV
        # O = Ox + Oy
        Ot[:, :, i] = attn_inter + attn @ delta_V
        # 更新状态 S[t+1]
        # g shape [B, H, chunk_num, chunk_size],
        last_recurrent_state = (
            # g[:, :, i, -1] 是当前chunk从第一个token到最后一个token的衰减（gamma_C），
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            # (g[:, :, i, -1, None] - g[:, :, i]) 是当前chunk每个位置到最后一个位置的衰减（gamma_C/gamma_i），
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ delta_V
        )

    if not output_final_state:
        last_recurrent_state = None
    Ot = Ot.reshape(Ot.shape[0], Ot.shape[1], -1, Ot.shape[-1])
    Ot = Ot[:, :, :L]
    Ot = Ot.transpose(1, 2).contiguous().to(initial_dtype)
    return Ot, last_recurrent_state


def run_case(
    B, T, H, K, V, dtype=torch.bfloat16, seed=0, label="", rtol=1e-2, atol=2e-2
):
    """Build inputs for one config, run kernel + reference, print + compare."""
    assert T % CHUNK_SIZE == 0, f"T={T} must be a multiple of CHUNK_SIZE={CHUNK_SIZE}"
    device = torch.device("cuda", 0)

    q, k, v, g, beta, _h0, idx = make_inputs(B, T, H, K, V, dtype, device, seed)
    scale = K**-0.5
    NT = T // CHUNK_SIZE

    if 0:
        # Use ZERO initial state to keep layout-agnostic comparison
        # (sglang uses [B,H,V,K]; reference uses [B,H,K,V] -- avoid mismatch).
        h0_sglang = torch.zeros(B, H, V, K, device=device, dtype=dtype)

    # use none zero states, sglang triton kernel current support [B,H,Dv,Dk] layout states.
    h0_sglang = torch.rand(B, H, V, K, device=device, dtype=dtype)
    # torch reference kernel current support [B,H,Dk,Dv] layout states
    h0_trans = h0_sglang.transpose(-1, -2).contiguous()
    # ----- Triton path (sglang) -----
    g_out, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0_sglang,
        initial_state_indices=idx,
        cu_seqlens=None,
        chunk_indices=None,
    )

    # ----- PyTorch reference -----
    o_ref, _ = torch_chunk_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta,
        chunk_size=CHUNK_SIZE,
        initial_state=h0_trans,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )

    header = (
        f"[case{(' ' + label) if label else ''}] "
        f"B={B} T={T} H={H} K={K} V={V} dtype={dtype} NT={NT}"
    )
    print(header)
    print(
        f"  shapes  : o={tuple(o.shape)}  o_ref={tuple(o_ref.shape)}  "
        f"w={tuple(w.shape)}  v_new={tuple(v_new.shape)}  h={tuple(h.shape)}"
    )

    # Sanity / shape checks.
    assert g_out.shape == (B, T, H)
    assert o.shape == (B, T, H, V) == o_ref.shape
    assert A.shape[:3] == (B, T, H) and A.shape[-1] == CHUNK_SIZE
    assert w.shape == (B, T, H, K)
    assert v_new.shape == (B, T, H, V)
    assert h.dim() == 5 and h.shape[0] == B and h.shape[1] == NT and h.shape[2] == H
    assert sorted(h.shape[-2:]) == sorted((K, V))
    for name, t in [("o", o), ("A", A), ("w", w), ("v_new", v_new), ("h", h)]:
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"
    assert torch.isfinite(o_ref).all(), "o_ref contains non-finite values"

    # Numerical comparison (cast both to fp32 for stable diff stats).
    of = o.float()
    rf = o_ref.float()
    diff = (of - rf).abs()
    denom = rf.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / denom).max().item()
    cos = F.cosine_similarity(of.flatten(), rf.flatten(), dim=0).item()
    print(
        f"  vs ref  : max|d|={max_abs:.3e}  mean|d|={mean_abs:.3e}  "
        f"max_rel={max_rel:.3e}  cos={cos:.6f}  "
        f"o.std={of.std().item():.3e}  ref.std={rf.std().item():.3e}"
    )
    torch.testing.assert_close(of, rf, rtol=rtol, atol=atol)
    print("  -> OK\n")


# Limitation: Q, K, V would have same head number in the test.
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    # A few shape cases (small enough to run quickly).
    run_case(B=1, T=128, H=1, K=16, V=16, label="small")

    # run_case(B=1, T=1024, H=8, K=128, V=128, label="medium")
    # run_case(B=2, T=512, H=4, K=64, V=128, label="K!=V")

    run_case(B=1, T=512, H=8, K=128, V=128, seed=42, label="qwen3.5_moe_like")

    print("All cases passed.")


if __name__ == "__main__":
    main()
