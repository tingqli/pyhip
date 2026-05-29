# Chunk-wise Gated Delta Rule (Linear Attention)

> Math reference: `gated_delta_net_theory.md`, "Chunk形式" / "chunk wise形式推导"

## Token-level (Recurrent) View

Shapes: $S_t \in \mathbb{R}^{D_k \times D_v}$, $k_t \in \mathbb{R}^{D_k}$, $v_t \in \mathbb{R}^{D_v}$, $q_t \in \mathbb{R}^{D_k}$, $o_t \in \mathbb{R}^{D_v}$; scalars $\alpha_t = \exp(g_t)$, $\beta_t$.

$`S_t = S_{t-1} \bigl(\alpha_t (I - \beta_t k_t k_t^T)\bigr) + \beta_t v_t k_t^T \tag{Eq. GDN-mem}`$


$`= \alpha_t S_{t-1} + \Delta v_t \, k_t^T, \quad \Delta v_t = \beta_t (v_t - \alpha_t S_{t-1} k_t)`$


$`o_t = S_t \, q_t \tag{Eq. GDN-out}`$

## Chunk-level View

Per chunk $[t]$ of length $C$, indices $i, j \in 1..C$, with:

- $\gamma_j = \prod_{r=1}^{j} \alpha_{tC+r}$ — per-chunk cumulative decay
- $\Gamma_{i,j} = \gamma_i / \gamma_j$ for $j \le i$, else $0$ — lower triangular decay matrix
- $S_{[t]} \in \mathbb{R}^{D_k \times D_v}$ — chunk-start memory
- $K_{[t]} \in \mathbb{R}^{C \times D_k}$, $Q_{[t]} \in \mathbb{R}^{C \times D_k}$, $V_{[t]} \in \mathbb{R}^{C \times D_v}$ — per-chunk stacks

### Core Chunk Equations

| Quantity | Formula | Shape |
|----------|---------|-------|
| $L$ | $\text{strictLower}\bigl(\text{diag}(\beta)\,(\Gamma \odot K K^T)\bigr)$ | $[C, C]$ |
| $\tilde{U}_{[t]}$ | $(I + L)^{-1}\,\text{diag}(\beta)\,V$ | $[C, D_v]$ |
| $W_{[t]}$ | $(I + L)^{-1}\,\text{diag}(\beta)\,K$ | $[C, D_k]$ |
| $W^\leftarrow_{[t]}$ | $(I + L)^{-1}\,\text{diag}(\gamma_i)\,\text{diag}(\beta)\,K$ | $[C, D_k]$ |
| $K^\rightarrow_{[t]}$ | $\text{diag}(\gamma_C / \gamma_i)\,K_{[t]}$ | $[C, D_k]$ |
| $Q^\leftarrow_{[t]}$ | $\text{diag}(\gamma_i)\,Q_{[t]}$ | $[C, D_k]$ |
| $S^\rightarrow_{[t]}$ | $\gamma_C \cdot S_{[t]}$ | $[D_k, D_v]$ |
| $\Delta V_{[t]}$ | $\tilde{U}_{[t]} - W^\leftarrow_{[t]}\,S_{[t]}^T$ | $[C, D_v]$ |

### State Update

$`S_{[t+1]} = S^\rightarrow_{[t]} + {K^\rightarrow_{[t]}}^T \,\Delta V_{[t]} \tag{Eq. S-update}`$


$`= [D_k, D_v] + [D_k, C] \times [C, D_v]`$

### Output

$`O_{[t]} = Q^\leftarrow_{[t]}\,S_{[t]}^T + \bigl(Q_{[t]}\,K_{[t]}^T \odot \Gamma_{[t]}\bigr)\,\Delta V_{[t]} \tag{Eq. O}`$


$`= \underbrace{[C, D_k] \times [D_k, D_v]}_{\text{inter-chunk}} + \underbrace{([C, C] \odot [C, C]) \times [C, D_v]}_{\text{intra-chunk}}`$

## Implementation Notes / Naming Map

### Math ↔ Code ↔ Shape

| Math | Code | Shape | Notes |
|------|------|-------|-------|
| $\gamma_j$ | `exp(g.cumsum(-1))` | `[B, Hv, N, C]` | per-chunk cumulative decay |
| $\Gamma$ | `decay_mask` | `[B, Hv, N, C, C]` | lower triangular decay matrix |
| $\text{diag}(\beta) @ K$ | `k_beta` | `[B, Hv, N, C, Dk]` | beta-scaled keys, input to $W$/$L$ |
| $L$ | `-attn` (before loop) | `[B, Hv, N, C, C]` | $L = \text{strictLower}(k\_beta @ K^T \odot \Gamma)$ |
| $(I + L)^{-1}$ | `attn` (after loop) | `[B, Hv, N, C, C]` | triangular inverse via forward-sub loop |
| $\text{diag}(\beta) @ V$ | `v_beta` | `[B, Hv, N, C, Dv]` | beta-scaled values, input to $\tilde{U}$ |
| $\tilde{U}$ | `value` after `attn @ v_beta` | `[B, Hv, N, C, Dv]` | $\tilde{U} = (I+L)^{-1} @ \text{diag}(\beta) @ V$ |
| $W^\leftarrow$ | `k_cumdecay` | `[B, Hv, N, C, Dk]` | $W^\leftarrow = \text{diag}(\gamma) @ (I+L)^{-1} @ \text{diag}(\beta) @ K$ |
| $S_{[t]}$ | `last_recurrent_state` | `[B, Hv, Dk, Dv]` | recurrent memory |
| $\Delta V$ | `v_new` | `[B, Hv, C, Dv]` | $\Delta V = \tilde{U} - W^\leftarrow @ S_{[t]}$ |

### Output Computation Path

| Term | Code | Shape | Notes |
|------|------|-------|-------|
| $O_x = Q^\leftarrow @ S_{[t]}$ | `attn_inter` | `[B, Hv, C, Dv]` | inter-chunk readout; needs $S_{[t]}$, sequentially updated per chunk |
| $O_y = (Q @ K^T \odot \Gamma) @ \Delta V$ | `attn @ v_new` | `[B, Hv, C, Dv]` | intra-chunk readout; needs $\Delta V$, which needs $S_{[t]}$ |
| $O = O_x + O_y$ | `o_chunk` | `[B, Hv, C, Dv]` | chunk output = inter + intra |

### State Update Path

| Term | Code | Shape | Notes |
|------|------|-------|-------|
| $K^\rightarrow$ | `k_i * exp(G_C - G_i)` | `[C, Dk]` | rescale $K$ to chunk-end decay |
| ${K^\rightarrow}^T @ \Delta V$ | `(k_i * exp(G_C - G_i))^T @ v_new` | `[B, Hv, Dk, Dv]` | state update term |
| $S_{[t+1]}$ | $S^\rightarrow_{[t]} + {K^\rightarrow}^T @ \Delta V$ | `[B, Hv, Dk, Dv]` | needs $S_{[t]}$, sequentially updated |

## Complexity

Running the token-level recurrence is $O(L)$ sequential. The chunked form rewrites it into a sequence of chunk-wise matrix ops (intra-chunk + inter-chunk), so each chunk uses $O(C^2)$ sequential work ($C$ = `chunk_size`) but all batches/heads/chunk-internal dims run in parallel on GPU.

## Context Parallelism — Zigzag 变体

> 实现：`chunk_cp_zigzag.py`，辅助函数在 `cp/utils.py`

### 核心思想

GDN 的 state recurrence $S_{[t+1]} = M_t \cdot S_{[t]} + b_t$ 是关于 $S_{[t]}$ 的**仿射变换**。每个 segment 可以先假设 $h_0 = 0$ 独立计算出 $(b, M)$，再通过 all-gather 收集所有 segment 的仿射对，链式合并得到真实初始 state，最后用正确的 $h_0$ 重跑一次。

### Zigzag 分配规则

以 32 chunks × 64 tokens = 2048 tokens、4 GPUs (`cp_size=4`) 为例。

每张卡持有 2 个 segment（前半 seg0 + 后半 seg1），共 $`2 \times \text{cp\_size} = 8`$ 个 segment 组成因果链：

```
全局因果顺序 (chunk 0 → 31):

Rank 0 seg0: chunk  0,  1,  2,  3   (token   0–255)
Rank 1 seg0: chunk  4,  5,  6,  7   (token 256–511)
Rank 2 seg0: chunk  8,  9, 10, 11   (token 512–767)
Rank 3 seg0: chunk 12, 13, 14, 15   (token 768–1023)
─────────── 分界线 ───────────
Rank 3 seg1: chunk 16, 17, 18, 19   (token 1024–1279)
Rank 2 seg1: chunk 20, 21, 22, 23   (token 1280–1535)
Rank 1 seg1: chunk 24, 25, 26, 27   (token 1536–1791)
Rank 0 seg1: chunk 28, 29, 30, 31   (token 1792–2047)
```

**前半正序 rank 0→3，后半反序 rank 3→0**——这就是 "zigzag"。

视觉表示：

```
Token 序列:  0 ────────────────────────────── 2047
             ├── seg0 ──┤                ├── seg1 ──┤
Rank 0:      ████                                ████
Rank 1:          ████                        ████
Rank 2:              ████                ████
Rank 3:                  ████        ████
                         正序 →  ← 反序
```

### 为什么要 Zigzag

每个 segment 需要 merge 其因果位置之前所有 segment 的仿射变换。

```python
# cp/utils.py
def causal_positions(rank, cp_size):
    seg0_pos = rank                      # 正序: 0,1,2,3
    seg1_pos = 2 * cp_size - 1 - rank    # 反序: 7,6,5,4
```

| GPU | seg0 因果位置 | seg1 因果位置 | 需 merge 步数 (seg0 + seg1) |
|-----|:---:|:---:|:---:|
| Rank 0 | 0 | 7 | 0 + 7 = **7** |
| Rank 1 | 1 | 6 | 1 + 6 = **7** |
| Rank 2 | 2 | 5 | 2 + 5 = **7** |
| Rank 3 | 3 | 4 | 3 + 4 = **7** |

每张卡总 merge 步数 = $`2 \times \text{cp\_size} - 1`$，完全均匀。若按顺序切分（非 zigzag），rank 3 需 merge 7 步而 rank 0 只需 0 步，极不均衡。

### All-Gather 布局 vs 因果顺序

NCCL `all_gather` 产出的物理布局：

```
gathered[0] = rank0_seg0    gathered[1] = rank0_seg1
gathered[2] = rank1_seg0    gathered[3] = rank1_seg1
gathered[4] = rank2_seg0    gathered[5] = rank2_seg1
gathered[6] = rank3_seg0    gathered[7] = rank3_seg1
```

`zigzag_causal_order(4)` 返回的映射 `[0, 2, 4, 6, 7, 5, 3, 1]`：

| 因果位置 | → gathered 行 | 含义 |
|:---:|:---:|------|
| 0 | 0 | rank0\_seg0 |
| 1 | 2 | rank1\_seg0 |
| 2 | 4 | rank2\_seg0 |
| 3 | 6 | rank3\_seg0 |
| 4 | 7 | rank3\_seg1 |
| 5 | 5 | rank2\_seg1 |
| 6 | 3 | rank1\_seg1 |
| 7 | 1 | rank0\_seg1 |

`cp_merge_kernel` 用此查找表从 `gathered` 中按因果顺序读取 $(M, b)$，**不需要物理重排内存**。

### 四阶段流水线

| Phase | 计算 | 通信 | 复杂度 |
|:---:|------|------|--------|
| 1 | 每 rank 本地算 $(b, M)$ | 无 | $`O(T / \text{cp\_size})`$ |
| 2 | 无 | All-Gather $(b, M)$ | $`O(K^2)`$ per head — **很小** |
| 3 | 链式仿射合并求真实 $h_0$ | 无 | $`O(\text{cp\_size} \cdot K^2)`$ per head |
| 4 | 用正确 $h_0$ 重跑 `fwd_h` + `fwd_o` | 无 | $`O(T / \text{cp\_size})`$ |

通信量只有 $K \times (V + K)$ per head per segment，比 all-gather QKV 小几个数量级。代价是计算量约 2×（Phase 1 + Phase 4 各跑一遍 recurrence）。
