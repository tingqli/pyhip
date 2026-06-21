"""
Verify that the one-solve and two-solve approaches for computing W^← (k_cumdecay)
produce identical results.

Two-solve (paper / GatedDeltaNet repo):
    W^← = D_γ  @  (I + L^(1))^{-1}  @  D_β  @  K

One-solve (Transformers):
    W^← = (I + L^(γ))^{-1}  @  D_γ  @  D_β  @  K

where L^(γ) = D_γ  L^(1)  D_γ^{-1}   (diagonal similarity).
"""

import torch
import torch.nn.functional as F


def forward_sub_inv(L_strict_lower):
    """Compute (I + L)^{-1} via forward substitution, where L is strict lower triangular."""
    C = L_strict_lower.shape[-1]
    attn = -L_strict_lower.clone()
    for i in range(2, C):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(C, dtype=attn.dtype, device=attn.device)
    return attn


def two_solve(K, beta, gamma, chunk_size):
    """Paper / GatedDeltaNet repo: two separate (I+L)^{-1} solves."""
    B, H, N, C, Dk = K.shape
    k_beta = K * beta.unsqueeze(-1)  # D_β @ K

    # --- Solve 1: with decay → ũ (not tested here, same in both) ---

    # --- Solve 2: without decay → W ---
    # L^(1)_{ij} = -β_i (k_i^T k_j),  strict lower
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=K.device), diagonal=0)
    L1 = (k_beta @ K.transpose(-1, -2)).masked_fill(mask, 0)  # strict lower, no decay
    inv1 = forward_sub_inv(L1)        # (I + L^(1))^{-1}
    W = inv1 @ k_beta                 # (I + L^(1))^{-1} @ D_β @ K

    # W^← = D_γ @ W
    W_left = W * gamma.unsqueeze(-1)  # multiply each row r by γ_r
    return W_left


def one_solve(K, beta, gamma, gamma_matrix, chunk_size):
    """Transformers: one (I+L)^{-1} solve, decay absorbed into input."""
    B, H, N, C, Dk = K.shape
    k_beta = K * beta.unsqueeze(-1)

    # L^(γ)_{ij} = -β_i (k_i^T k_j) * γ_i/γ_j,  strict lower
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=K.device), diagonal=0)
    Lg = (k_beta @ K.transpose(-1, -2) * gamma_matrix).masked_fill(mask, 0)
    inv_g = forward_sub_inv(Lg)       # (I + L^(γ))^{-1}

    # W^← = (I + L^(γ))^{-1} @ D_γ @ D_β @ K
    W_left = inv_g @ (k_beta * gamma.unsqueeze(-1))
    return W_left


def run_case(B, T, H, K_dim, V_dim, chunk_size=64, seed=42, label=""):
    torch.manual_seed(seed)
    device = "cuda"

    K = torch.randn(B, H, T, K_dim, device=device, dtype=torch.float32)
    K = F.normalize(K, dim=-1)  # L2-normed keys (like real model)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=torch.float32))
    g_raw = -F.softplus(torch.randn(B, H, T, device=device, dtype=torch.float32))

    # Pad to multiple of chunk_size
    pad = (chunk_size - T % chunk_size) % chunk_size
    K = F.pad(K, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g_raw = F.pad(g_raw, (0, pad))
    T_padded = T + pad

    # Reshape to chunks
    K_c = K.reshape(B, H, -1, chunk_size, K_dim)
    beta_c = beta.reshape(B, H, -1, chunk_size)
    g_c = g_raw.reshape(B, H, -1, chunk_size)

    # cumsum → γ_r = exp(g_1 + ... + g_r)
    g_cumsum = g_c.cumsum(dim=-1)
    gamma = g_cumsum.exp()  # [B, H, N, C]

    # Γ_{ij} = γ_i / γ_j = exp(g_cumsum_i - g_cumsum_j), lower triangular
    gamma_matrix = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp().tril()

    w_two = two_solve(K_c, beta_c, gamma, chunk_size)
    w_one = one_solve(K_c, beta_c, gamma, gamma_matrix, chunk_size)

    abs_diff = (w_two - w_one).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (w_two.abs() + 1e-12)).max().item()

    status = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"[{status}] {label:20s}  B={B} T={T} H={H} K={K_dim} V={V_dim}  "
          f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  rel_diff={rel_diff:.2e}")
    assert max_diff < 1e-4, f"FAILED: max_diff={max_diff}"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    run_case(B=1, T=128,  H=1, K_dim=16,  V_dim=16,  label="small")
    run_case(B=1, T=512,  H=8, K_dim=128, V_dim=128, seed=42, label="qwen3.5_moe_like")
    run_case(B=2, T=512,  H=4, K_dim=64,  V_dim=128, label="K!=V")
    run_case(B=1, T=1024, H=8, K_dim=128, V_dim=128, label="medium")

    print("\nAll cases passed: one-solve ≡ two-solve for W^←")


if __name__ == "__main__":
    main()
