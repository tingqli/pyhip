# Gated Delta Rule: KKT Solve — 一次求解 vs 两次求解

## 问题背景

对比两个实现，发现在 chunk 内计算 $(I+L)^{-1}$ 时存在显著差异：

| 实现 | 文件 | KKT 求解次数 |
|------|------|:---:|
| 原始论文 / GatedDeltaNet repo | `GatedDeltaNet/.../chunk.py` L662-673 | **2 次** |
| HuggingFace Transformers (Qwen3) | `transformers/.../modeling_qwen3_next.py` L417-424 | **1 次** |

两者最终数值等价，但推导路径和计算方式不同。

---

## 论文的推导：需要两个不同的矩阵

回顾论文 Section 2.2 和 3.3 的推导。DeltaNet 的 WY representation 产生两个量：

### 量 1: $\tilde{U}$ — 用于修正 value

论文 Eq. (5) + Eq. (7)，对 Gated Delta Rule 扩展后：

$$\tilde{U}_{[t]} = T^{(\gamma)} \cdot \text{diag}(\beta) \cdot V$$

其中 $T^{(\gamma)}$ 是带 decay 的三角逆：

$$T^{(\gamma)} = \left[ I + \text{strictLower}\left(\text{diag}(\beta) \cdot (\Gamma \odot KK^T)\right) \right]^{-1}$$

这里 $\Gamma_{ij} = \gamma_i / \gamma_j$ 是 decay matrix。

### 量 2: $W$ — 用于修正 key（state update 的系数）

论文 Eq. (4) + Eq. (7)，对 **原始 DeltaNet**（无 gating）：

$$W_{[t]} = T^{(1)} \cdot \text{diag}(\beta) \cdot K$$

其中 $T^{(1)}$ 是 **不带 decay** 的三角逆：

$$T^{(1)} = \left[ I + \text{strictLower}\left(\text{diag}(\beta) \cdot KK^T\right) \right]^{-1}$$

关键点：**$\tilde{U}$ 和 $W$ 用的三角逆矩阵不同**：
- $\tilde{U}$ 的三角逆 $T^{(\gamma)}$ 包含 $\Gamma$ decay
- $W$ 的三角逆 $T^{(1)}$ 不包含 decay

### 为什么 $W$ 不带 decay？

回到论文的递推。$W$ 来源于 $P_r$ (Eq. 4)，它描述的是 state transition 矩阵的累积效应：

$$P^r_{[t]} = \prod_{i=1}^{r} \left(\alpha_i(I - \beta_i k_i k_i^T)\right)$$

展开后用 WY representation：

$$P^r_{[t]} = I - \sum_{i=1}^{r} w_i k_i^T$$

$$w_r = \beta_r \left( k_r - \sum_{i=1}^{r-1} w_i (k_i^T k_r) \right)$$

当扩展到 Gated Delta Rule 时，论文 Section 3.3 给出了 $G^r$ 的修正（即 $\tilde{U}$ 的来源），它在 WY 内部引入了 $\gamma_i/\gamma_j$ 的 decay factor。但对 $W$ 本身的定义，DeltaNet 原始论文中 $W$ 来自 $P$ 矩阵，$P$ 在 gated 版本中变成了 $F = \gamma \cdot P$，真正进入 state update 的是 $W^{\leftarrow} = \text{diag}(\gamma) \cdot W$，decay 是在外面乘的。

也就是说论文的路径是：
1. 先算不带 decay 的 $W = T^{(1)} \cdot \text{diag}(\beta) \cdot K$
2. 再乘 decay 得到 $W^{\leftarrow} = \text{diag}(\gamma) \cdot W$

---

## GatedDeltaNet repo 的实现（两次求解）

```python
# chunk.py L662-673

# 第 1 次求解：带 decay 的 T^(γ)，用于 u
attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
for i in range(1, chunk_size):
    attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
k_cumsum = attn @ v              # ← 这是 ũ = T^(γ) @ diag(β) @ V

# 第 2 次求解：不带 decay 的 T^(1)，用于 w
attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)   # 注意：没有 * L_mask
for i in range(1, chunk_size):
    attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
w = k_cumdecay = attn @ k_beta   # ← 这是 W = T^(1) @ diag(β) @ K
```

然后在 chunk loop 中：

```python
# L681: W^← = W * exp(decay)，decay 在外面乘
v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
```

严格遵循论文：先用不带 decay 的逆算 $W$，再乘 $\text{diag}(\gamma)$ 得 $W^{\leftarrow}$。

---

## Transformers 的实现（一次求解）

```python
# modeling_qwen3_next.py L414-424

# 只做 1 次求解：带 decay 的 T^(γ)
decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

value = attn @ v_beta                                # ← ũ = T^(γ) @ diag(β) @ V  ✓ 同上
k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1)) # ← 重用 T^(γ)，但输入不同！
```

### `k_cumdecay`（L424）的数学含义

`k_cumdecay` 就是论文的 $W^{\leftarrow}_{[t]}$，shape 为 $[B, H, N, C, D_k]$，其中 $N$ 是 chunk 数。

逐行展开，`k_cumdecay` 的第 $r$ 行（chunk 内第 $r$ 个 token）是向量 $w^{\leftarrow}_r \in \mathbb{R}^{D_k}$：

$$w^{\leftarrow}_r = \gamma_r \cdot w_r$$

其中 $w_r$ 是论文 Eq. (4) 的 WY 系数，$\gamma_r = \exp(g_1 + g_2 + ... + g_r)$ 是 cumulative decay。

$W^{\leftarrow}$ 的物理含义：**把 chunk 开头的旧 state $S_{[t]}$ 对当前 chunk 内每个 token 的"旧值贡献"提取出来**。具体地，对 chunk $[t]$ 内的第 $r$ 个 token：

$$\text{旧值贡献}_r = w^{\leftarrow}_r \cdot S_{[t]}^T \in \mathbb{R}^{D_v}$$

它告诉你：如果只看 chunk 开头的 state $S_{[t]}$，经过前 $r$ 个 token 的 delta rule 更新和 decay 衰减后，对第 $r$ 个 token 位置残留的 key 方向投影是多少。

代码对应：

```python
# L424: 计算 W^← = (I+L^(γ))^{-1} @ diag(γ) @ diag(β) @ K
#   attn       = (I+L^(γ))^{-1}               shape [B,H,N, C,C]
#   k_beta     = diag(β) @ K                   shape [B,H,N, C,Dk]
#   g.exp()    = diag(γ)  （按行缩放）           shape [B,H,N, C]
k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
#            [C,C] @ [C,Dk] → [C,Dk]
```

### `v_prime`（L437）的数学含义

在 chunk loop 中，`v_prime` 对应论文的 $W^{\leftarrow}_{[t]} S_{[t]}^T$，shape 为 $[B, H, C, D_v]$。

$$V'_{[t]} = W^{\leftarrow}_{[t]} \, S_{[t]}^T \in \mathbb{R}^{C \times D_v}$$

物理含义：**chunk 开始时旧 state 对当前 chunk 每个 token 的 value 贡献**（考虑了 decay + delta rule 的累积擦写效应）。

接下来 `v_new = v_i - v_prime`，也就是论文的 $\Delta V$：

$$\Delta V_{[t]} = \tilde{U}_{[t]} - W^{\leftarrow}_{[t]} S_{[t]}^T$$

$\tilde{U}$ 是"当前 chunk 内新写入的 value"，$W^{\leftarrow} S^T$ 是"旧 state 残留的 value"，两者相减得到 **净增量**——这个 chunk 真正对 state 的新贡献。

代码对应：

```python
# L437-438
v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
#         [C, Dk]              @ [Dk, Dv]  → [C, Dv]
#         W^←_{[t]}             S_{[t]}^T
v_new = v_i - v_prime
#       ũ_{[t]}  -  W^←_{[t]} @ S_{[t]}^T  =  ΔV_{[t]}
```

### 论文 repo 对比

论文 repo 的 `v_prime` 多了一步手动乘 decay：

```python
# GatedDeltaNet repo L681
v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
#          W_{[t]}（不带decay） * diag(γ)                         @ S^T
#          ─────────────────────────────────
#                     = W^←_{[t]}
```

因为 repo 的 `k_cumdecay` 是 $W$（不带 decay），需要额外乘 $\exp(\text{decay})$ 得到 $W^{\leftarrow}$。

Transformers 的 `k_cumdecay` 已经是 $W^{\leftarrow}$（带 decay），所以直接用。

---

## 等价性证明

### 符号约定

- 所有矩阵乘法写作 **并列**（如 $AB$），不用 $\cdot$
- $\odot$ 表示逐元素乘（Hadamard product）
- $C$ 是 chunk size，以下所有矩阵都是 $C \times C$ 或 $C \times D$
- $D_\gamma = \text{diag}(\gamma_1, \gamma_2, ..., \gamma_C)$ 是对角矩阵，$D_\gamma^{-1} = \text{diag}(1/\gamma_1, ..., 1/\gamma_C)$
- $D_\beta = \text{diag}(\beta_1, ..., \beta_C)$

### 要证的目标

两个实现对 $W^{\leftarrow}$（`k_cumdecay`）的计算方式不同，我们要证它们相等：

$$\boxed{ D_\gamma \underbrace{(I + L^{(1)})^{-1}}_{\text{不带decay的逆}} D_\beta K \;=\; \underbrace{(I + L^{(\gamma)})^{-1}}_{\text{带decay的逆}} D_\gamma D_\beta K }$$

$$\text{（论文：两次求解）} \qquad\qquad\qquad\qquad \text{（Transformers：一次求解）}$$

左边是论文的做法：先用不带 decay 的逆算 $W = (I+L^{(1)})^{-1} D_\beta K$，再乘 $D_\gamma$。

右边是 Transformers 的做法：把 $D_\gamma$ 吸收进输入，直接用带 decay 的逆一步算完。

### 两个 $L$ 矩阵的定义

不带 decay 的 $L$：

$$L^{(1)}_{ij} = \begin{cases} -\beta_i (k_i^T k_j) & \text{if } i > j \\ 0 & \text{otherwise} \end{cases}$$

带 decay 的 $L$：

$$L^{(\gamma)}_{ij} = \begin{cases} -\beta_i (k_i^T k_j) \cdot \dfrac{\gamma_i}{\gamma_j} & \text{if } i > j \\ 0 & \text{otherwise} \end{cases}$$

逐元素比较可得：

$$L^{(\gamma)}_{ij} = \frac{\gamma_i}{\gamma_j} L^{(1)}_{ij}$$

### Step 1: $L^{(\gamma)}$ 和 $L^{(1)}$ 的对角相似关系

上面的逐元素关系等价于矩阵关系：

$$L^{(\gamma)} = D_\gamma \, L^{(1)} \, D_\gamma^{-1}$$

验证：$(D_\gamma \, L^{(1)} \, D_\gamma^{-1})_{ij} = \gamma_i \cdot L^{(1)}_{ij} \cdot \frac{1}{\gamma_j} = \frac{\gamma_i}{\gamma_j} L^{(1)}_{ij} = L^{(\gamma)}_{ij}$ ✓

### Step 2: 逆矩阵的相似关系

由 Step 1：

$$I + L^{(\gamma)} = D_\gamma (I + L^{(1)}) D_\gamma^{-1}$$

（因为 $D_\gamma I D_\gamma^{-1} = I$）

对两边取逆：

$$(I + L^{(\gamma)})^{-1} = \left(D_\gamma (I + L^{(1)}) D_\gamma^{-1}\right)^{-1} = D_\gamma (I + L^{(1)})^{-1} D_\gamma^{-1}$$

（矩阵逆的性质：$(ABC)^{-1} = C^{-1} B^{-1} A^{-1}$）

### Step 3: 代入目标式右边

$$\text{右边} = (I + L^{(\gamma)})^{-1} \, D_\gamma D_\beta K$$

用 Step 2 替换：

$$= D_\gamma (I + L^{(1)})^{-1} D_\gamma^{-1} \, D_\gamma D_\beta K$$

$D_\gamma^{-1}$ 和 $D_\gamma$ 相消：

$$= D_\gamma (I + L^{(1)})^{-1} D_\beta K = \text{左边} \quad \blacksquare$$

### 对应到代码

| 数学 | 论文 repo 代码 | Transformers 代码 |
|------|------|------|
| $(I+L^{(1)})^{-1}$ | 第 2 次前向替换（不带 `L_mask`） | 不需要 |
| $(I+L^{(\gamma)})^{-1}$ | 第 1 次前向替换（带 `L_mask`） | 唯一的前向替换（带 `decay_mask`） |
| $D_\beta K$ | `k_beta` | `k_beta` |
| $D_\gamma D_\beta K$ | 不需要 | `k_beta * g.exp().unsqueeze(-1)` |
| $W^{\leftarrow}$ | `(attn @ k_beta) * exp(decay)` | `attn @ (k_beta * exp(g))` |

一句话：**decay 要么乘在逆矩阵外面（论文），要么吸收进输入里面（Transformers），等价是因为对角相似变换**。

---

## 总结对比

|  | 论文 / GatedDeltaNet repo | Transformers |
|--|:---:|:---:|
| 三角逆求解次数 | 2 | 1 |
| $\tilde{U}$ 的计算 | $(I+L^{(\gamma)})^{-1} D_\beta V$ | 相同 |
| $W^{\leftarrow}$ 的计算 | $D_\gamma (I+L^{(1)})^{-1} D_\beta K$ | $(I+L^{(\gamma)})^{-1} D_\gamma D_\beta K$ |
| 数学等价性 | ✓ | ✓ |
| 前向替换循环 | 2 × chunk_size 次 | 1 × chunk_size 次 |
| 核心 trick | 无 | 利用 $D_\gamma(I+L^{(1)})^{-1}D_\gamma^{-1} = (I+L^{(\gamma)})^{-1}$ 的对角相似变换 |

### Transformers 的优势

- 只需 **1 次** 前向替换（最昂贵的串行部分），计算量约减半
- 少一次 $O(C^2 D_k)$ 的矩阵乘
- 对 Triton kernel 实现而言，省掉第二次 $(I+L)^{-1}$ 意味着更少的寄存器压力和更简洁的 kernel

### 为什么论文用两次？

论文的推导是从 WY representation 的原始递推公式 (Eq. 4-5) 直接推出的。$P$ 和 $G$（对应 $W$ 和 $\tilde{U}$）分别来自不同的展开路径，所以自然得到两个不同的三角系统。论文在 Section 3.3 对 $\tilde{U}$ 做了 Gated 扩展（引入 $\Gamma$），但 $W$ 继承自 DeltaNet 原始公式，保持不带 decay 的形式。

Transformers 的实现者发现了 decay 矩阵和无 decay 矩阵之间的 **对角相似变换** 关系，从而将两次求解统一成一次。这是一个纯代数层面的优化，不改变任何语义。

---

## 数值验证

验证脚本：[`verify_kkt_one_vs_two.py`](verify_kkt_one_vs_two.py)

分别用 `two_solve`（论文做法：不带 decay 的逆 + 外乘 $D_\gamma$）和 `one_solve`（Transformers 做法：带 decay 的逆，输入吸收 $D_\gamma$）计算 $W^{\leftarrow}$，比较结果。

```
[PASS] small                 B=1 T=128 H=1 K=16 V=16    max_diff=1.86e-09  mean_diff=1.69e-11  rel_diff=3.56e-05
[PASS] qwen3.5_moe_like      B=1 T=512 H=8 K=128 V=128  max_diff=7.45e-09  mean_diff=1.09e-11  rel_diff=2.06e-03
[PASS] K!=V                  B=2 T=512 H=4 K=64 V=128   max_diff=7.45e-09  mean_diff=1.64e-11  rel_diff=2.17e-03
[PASS] medium                B=1 T=1024 H=8 K=128 V=128  max_diff=1.49e-08  mean_diff=9.32e-12  rel_diff=6.71e-03
```

所有 case 在 float32 下 max absolute diff < 1.5e-8，确认两种做法数值等价。
