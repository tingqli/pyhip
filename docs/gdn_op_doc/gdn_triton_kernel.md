# Implementation Notes / Naming Map (Math ↔ Code ↔ Shape)

`[B, Hv, T]` prefilled split by chunk, becomes `[B, Hv, N, C]`

---

### `chunk_local_cumsum()` — parallel on B, Hv, N，每个CU负责C个元素的cumulative sum

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $\gamma_j$ | `exp(g.cumsum(-1))` | `[B, Hv, N, C]` | per-chunk cumulative sum decay |

---

### `chunk_gated_delta_rule_fwd_kkt_solve_kernel()` — `[B, Hv, N]` parallel，每个CU负责 `[C, Dk]@[Dk, C]`, inverse the `[C, C]` triangular matrix

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $\Gamma$ | `decay_mask` | `[B, Hv, N, C, C]` | lower triangular decay matrix |
| $\text{diag}(\beta) K$ | `k_beta` | `[B, Hv, N, C, Dk]` | beta-scaled keys, input to W/L |
| $L$ | `-attn` (before loop) | `[B, Hv, N, C, C]` | $L = \text{strictLower}(k\_beta @ K^T \odot \Gamma)$ |
| $(I + L)^{-1}$ | `attn` (after loop) | `[B, Hv, N, C, C]` | triangular inverse via forward-sub loop |

---

### `recompute_w_u_fwd()` — `[B, H, N]` parallel. Each chunk: `[C, C]@[C, Dv]` → Ũ, `[C, C]@[C, Dk]` → W←

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $\text{diag}(\beta) V$ | `v_beta` | `[B, Hv, N, C, Dv]` | beta-scaled values, input to Ũ |
| $\tilde{U}$ | `value` after `attn@v_beta` | `[B, Hv, N, C, Dv]` | $\tilde{U}=(I+L)^{-1} \text{diag}(\beta) V$ = `attn@v_beta` |
| $W^\leftarrow$ | `k_cumdecay` | `[B, Hv, N, C, Dk]` | $W^\leftarrow=\text{diag}(\gamma)(I+L)^{-1}\text{diag}(\beta)K$ = `attn@(k_beta*gamma)`, rescale W |

---

### `chunk_gated_delta_rule_fwd_kernel_h_blockdim64()` — `[B, Hv, Dv/BV]` parallel

SSM state would have chunk-wise store → `[B, Hv, N, Dk, Dv]` to ensure O can be calculated in parallel between chunks.

In one WG, each chunk: `[C, Dk]@[Dk, Bv]` = `[C, Bv]` (ΔV), `[Dk, C]@[C, Bv]` = `[Dk, Bv]` ($S_{[t+1]}$). N chunks in sequence.

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $S_{[t]}$ | `last_recurrent_state` | `[B, Hv, Dk, Dv]` | recurrent memory |
| $\Delta V = \tilde{U} - W^\leftarrow S$ | `v_new` | `[B, Hv, N, C, Dv]` | $\Delta V = \tilde{U} - W^\leftarrow @ S_{[t]}$. Need $S_{[t]}$, sequentially updated per chunk |

**State Update:**

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $K^\rightarrow_{[t]} = \text{diag}(\gamma_C/\gamma_i) K$ | `k_i * exp(G_C - G_i)` | `[C, Dk]` | rescale K to chunk-end decay |
| ${K^\rightarrow}^T @ \Delta V$ | `(k_i*exp(G_C-G_i))^T @ v_new` | `[B, Hv, Dk, Dv]` | state update term |
| $S_{[t+1]} = S^\rightarrow_{[t]} + {K^\rightarrow}^T @ \Delta V$ | | `[B, Hv, Dk, Dv]` | need $S_{[t]}$, sequentially updated per chunk |

---

### `chunk_fwd_o()` — `[B, Hv, N]` parallel. Each WG: `[C, Dk]@[Dk, Dv]` = `[C, Dv]` and `[C, Dk]@[Dk, C]@[C, Dv]` = `[C, Dv]`

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $O_x = Q^\leftarrow @ S_{[t]}$ | `attn_inter` | `[B, Hv, N, C, Dv]` | inter-chunk readout. Need $S_{[t]}$, sequentially updated per chunk |
| $O_y = (Q @ K^T \odot \Gamma) \Delta V$ | `attn @ v_new` | `[B, Hv, N, C, Dv]` | intra-chunk readout. Need ΔV, which needs $S_{[t]}$ |
| $O = O_x + O_y$ | `o_chunk` | `[B, Hv, N, C, Dv]` | chunk output = inter + intra |
