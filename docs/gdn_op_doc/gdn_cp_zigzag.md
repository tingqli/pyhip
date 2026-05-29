# CP Zigzag 中 (b, M) 的理解笔记

> 运行例子：**512 tokens，2 GPU，zigzag context parallelism**

---

## 1. 问题背景

GDN（Gated Delta Net）的 chunk 递推是一个**严格因果**的过程：

$$h_{t+1} = \gamma_t \cdot h_t + K_t^T (u_t - W_t \cdot h_t)$$

后面 chunk 的 state 依赖前面 chunk 的 state。当把序列切分到多个 GPU 上时，每个 rank 想"独立算自己的 token"就需要一个真实的初始 state $h_0$ —— 而这个 $h_0$ 又依赖前面 rank 跑完的结果。

**朴素方案**：rank 之间 P2P 串行传递 state，GPU 大部分时间在等。GPU 越多，串行链越长，加速比越差。详见 [§2.5](#25-朴素方案的工作流程)。

**CP zigzag 方案**：用 $(b, M)$ 把"一个 segment 的完整递推"压缩成一个仿射变换。$(b, M)$ 不依赖 $h_0$，所以可以**各 rank 并行**计算；然后通过一次 all_gather + 廉价的矩阵串联，反推出每个 segment 真正的 $h_0$。

---

## 2. 512 tokens / 2 GPU 的 segment 布局

zigzag 把序列切成 $`2 \cdot \text{cp\_size} = 4`$ 个 segment，每段 $512 / 4 = 128$ token：

```
全局 token:  [0──127 | 128──255 | 256──383 | 384──511]
                ^ 实际是 [0:128) [128:256) [256:384) [384:512)

因果链:   seg0[0:128)  →  seg1[128:256)  →  seg2[256:384)  →  seg3[384:512)
```

**Zigzag 分布**（每个 rank 拿一前一后两段，保证负载均衡）：

| | seg0 | seg1 | seg2 | seg3 |
|---|:---:|:---:|:---:|:---:|
| 持有者 | Rank 0 | Rank 1 | Rank 1 | Rank 0 |

```
Rank 0 持有: seg0 [0:128)   + seg3 [384:512)   ← 物理内存上拼成 256 token
Rank 1 持有: seg1 [128:256) + seg2 [256:384)
```

每个 rank 本地有 256 token，但**这两段在因果链上不相邻**，必须当成两个独立 sub-sequence 处理（用 `seg_cu` 控制）。

### 2.5 朴素方案的工作流程

不引入 $(b, M)$ 时，唯一办法是**沿因果链 P2P 串行**：每个 rank 等前面 rank 算完再开干。

```
时间 ─────────────────────────────────────────────────────────────────→

Rank 0:  [fwd_h seg0]──send h₁──►  [............空闲............]  ──recv h₃──► [fwd_h seg3]
Rank 1:  [....空闲....]──recv h₁──► [fwd_h seg1][fwd_h seg2] ──send h₃──► [....空闲....]
```

具体步骤：

1. **Rank 0** 用 $h_g$ 跑 `fwd_h` 处理 seg0 → 得到 $h_1$
2. **Rank 0 → Rank 1** P2P 发送 $h_1$
3. **Rank 1** 用 $h_1$ 跑 `fwd_h` 处理 seg1 → 得到 $h_2$
4. **Rank 1** 用 $h_2$ 跑 `fwd_h` 处理 seg2 → 得到 $h_3$
5. **Rank 1 → Rank 0** P2P 发送 $h_3$
6. **Rank 0** 用 $h_3$ 跑 `fwd_h` 处理 seg3 → 得到 $h_4$

**问题**：

- **GPU 利用率低**：任意时刻只有一个 rank 在算，另一个在等 → ~50% 利用率
- **总时间 = 4 × T_seg + 2 次 P2P**，和单卡相比没有加速
- **扩展性差**：cp_size 翻倍 → segment 数翻倍 → 串行链变长，墙钟时间不变
- **通信量**：每次 P2P 发的是完整 state $[H, K, V]$，虽然不算大，但**通信和计算无法重叠**（下一步必须等通信结束）

朴素方案的根本问题是：因果递推有**串行依赖**，物理切分不能改变这个依赖。CP zigzag 的关键洞察是——**依赖可以"提前算好系数再合并"**，把串行变并行。

---

## 3. 核心数学：$`(b, M)`$ 是什么

### 3.1 chunk 递推可写成关于 $h$ 的仿射

$$h_{t+1} = \underbrace{(\gamma_t I - K_t^T W_t)}_{A_t} \cdot h_t + \underbrace{K_t^T u_t}_{c_t}$$

$A_t, c_t$ 只依赖**本地** token 数据 $(k, w, u, g)$，**不依赖 $h_t$**。

### 3.2 一个 segment 内 $NT$ 个 chunk 串起来

把 $NT$ 个 chunk 的仿射连乘展开：

$$h_{\text{out}} = \underbrace{\left(\prod_{t=NT-1}^{0} A_t\right)}_{M} \cdot h_{\text{in}} + \underbrace{\sum_{t=0}^{NT-1}\left(\prod_{s=t+1}^{NT-1} A_s\right) c_t}_{b}$$

即：

$$\boxed{\;h_{\text{out}} = M \cdot h_{\text{in}} + b\;}$$

这是**精确等式**，不是近似。无论 $h_{\text{in}}$ 取何值，跑完整个 segment 的真实结果都等于 $M h_{\text{in}} + b$。

### 3.3 两个边界情况就能反解出 $(b, M)$

| 设定 | 得到 | 实现 |
|---|---|---|
| $h_0 = 0$ | $h_{\text{out}} = b$ | `compute_br` |
| $h_0 = I$，去掉 $c_t$（即 $u=0$） | $h_{\text{out}} = M$ | `compute_M_total` |

两个 kernel 的差异：

```python
# compute_br: h0=0, 含 u
b_h = 0
for t in range(NT):
    b_v = u_t - W_t @ b_h
    b_h = γ_t * b_h + K_t^T @ b_v
return b_h          # 这就是 b

# compute_M_total: δh=I, 无 u, 注意减号
δh = I
for t in range(NT):
    wdh = W_t @ δh
    δh = γ_t * δh - K_t^T @ wdh    # = A_t @ δh
return δh           # 这就是 M
```

### 3.4 关键性质

> $(b, M)$ 完全由本地 $(k, w, u, g)$ 决定，**和 $h_0$ 无关**，所以各 rank 可以并行计算。

---

## 4. 形状（以 512 tokens / 2 GPU / H heads / K=128 / V=128 为例）

每个 rank 持有 2 个 segment，每段 128 token。设 batch 里有 $N$ 条序列（这里 $N = 1$）：

| 张量 | shape | 含义 |
|---|---|---|
| 单 segment 单 head 的 $b$ | $[K, V] = [128, 128]$ | $h_0=0$ 时的最终 state |
| 单 segment 单 head 的 $M$ | $[K, K] = [128, 128]$ | $h_0$ 的线性映射矩阵 |
| `compute_br` 输出 | $[2N, H, K, V] = [2, H, 128, 128]$ | 本 rank 2 段 |
| `compute_M_total` 输出 | $[2N, H, K, K] = [2, H, 128, 128]$ | 本 rank 2 段 |
| all_gather 后 `gathered` | $`[2\cdot\text{cp\_size}, N, H, K, V{+}K] = [4, 1, H, 128, 256]`$ | 全部 4 段的 (b, M) |

> 注意：$(b, M)$ 的大小**与 token 数无关**——segment 越长，计算越久，但输出形状不变。这是通信开销恒定的根源。

---

## 5. 4 个 segment 的真实 $h_0$ 怎么算

定义全局初始 state（用户传入）为 $h_g$。沿因果链串联 $(b, M)$：

| segment | 真实 $h_0$ |
|---|---|
| seg0 | $h_g$ |
| seg1 | $M_0 \cdot h_g + b_0$ |
| seg2 | $M_1 \cdot (M_0 h_g + b_0) + b_1$ |
| seg3 | $M_2 \cdot h_0^{\text{seg2}} + b_2$ |

每个 $h_0$ 都是真实的、和单卡 `fwd_h` 跑到该位置时的中间 state **数学等价**。

### 各 rank 的视角

```
Rank 0:  cp_merge 算 h0_seg0 (apply 0 个仿射) = h_g
         cp_merge 算 h0_seg3 (apply 3 个仿射)

Rank 1:  cp_merge 算 h0_seg1 (apply 1 个仿射)
         cp_merge 算 h0_seg2 (apply 2 个仿射)
```

### 5.1 `cp_merge` 是怎么工作的

`cp_merge` 的核心循环：从 `gathered`（all_gather 后的 $(b, M)$ 缓冲区）里**按因果链顺序**取出前 `num_ranks` 个仿射对，依次作用到 $h_0$ 上。

```python
b_h = h_global    # 或 0（如果没传 initial_state）
for r in range(num_ranks):
    r_actual = causal_order[r]              # 因果位置 → gathered 行号
    b_b = gathered[r_actual, ..., :V]       # 这一段的 b   [K, V]
    b_m = gathered[r_actual, ..., V:]       # 这一段的 M   [K, K]
    b_h = b_m @ b_h + b_b                   # 应用一次仿射
return b_h
```

#### 参数 `num_ranks` 控制串多少个仿射

`num_ranks` 决定**应用几个仿射**，也就是"目标 segment 在因果链上的位置"：

| 调用 | `num_ranks` | 含义 | 结果 |
|---|:---:|---|---|
| `cp_merge(..., 0, ...)` | 0 | 不串任何仿射 | $h_g$（seg0 的 $h_0$） |
| `cp_merge(..., 1, ...)` | 1 | 串 1 个：seg0 的 $(b_0, M_0)$ | seg1 的 $h_0 = M_0 h_g + b_0$ |
| `cp_merge(..., 2, ...)` | 2 | 串 2 个：seg0, seg1 | seg2 的 $h_0$ |
| `cp_merge(..., 3, ...)` | 3 | 串 3 个：seg0, seg1, seg2 | seg3 的 $h_0$ |
| `cp_merge(..., 4, ...)` | 4 | 串全部 4 个 | 整个序列末尾的 `final_state` |

#### 实际调用

```python
seg0_pos, seg1_pos = causal_positions(rank, cp_size)
# Rank 0: seg0_pos=0, seg1_pos=3
# Rank 1: seg0_pos=1, seg1_pos=2

h0_seg0 = cp_merge(gathered, h_g, seg0_pos, ...)   # 本 rank 第 1 段的 h0
h0_seg1 = cp_merge(gathered, h_g, seg1_pos, ...)   # 本 rank 第 2 段的 h0

# 可选：算最终 state
final_state = cp_merge(gathered, h_g, 2*cp_size, ...)
```

#### `causal_order` 的作用

`gathered` 的物理布局是 NCCL all_gather 顺序：`[r0_seg0, r0_seg1, r1_seg0, r1_seg1, ...]`，**不是因果顺序**。`causal_order` 是一张查找表，让 kernel 内部按因果顺序遍历 `gathered` 行，避免物理重排那个大 buffer（在大 $N$ 下能省下数百 MB）。

cp_size=2 时 `causal_order = [0, 2, 3, 1]`，意思是因果位置 0/1/2/3 分别对应 `gathered` 的第 0/2/3/1 行。

---

## 6. 完整 4 阶段流程

```
Phase 1 (并行)        各 rank 用本地 (k,w,u,g) 算 (b₀,b₁), (M₀,M₁)
                     ↓
Phase 2 (通信)        all_gather: 拿到全部 4 段的 (b, M)
                     ↓
Phase 3 (并行,轻量)   cp_merge 按因果链串联仿射 → 每段的真实 h₀
                     ↓
Phase 4 (并行,重)     fwd_h(initial_state=h₀) → 每 chunk 的 h, v_new
                     fwd_o(q, k, v_new, h, g) → 本 rank token 的输出 o
```

### Phase 4 的 `fwd_h` 做什么

**不是只算 v_new！** 它和单卡 `fwd_h` 一模一样：

1. 加载真实 $h_0$
2. 逐 chunk 跑递推
3. **每个 chunk 开头把 $h_t$ 存到 HBM**（给 `fwd_o` 用）
4. 算 v_new 也存到 HBM
5. 更新累加器到下一个 chunk

512 token / 2 GPU 时，每个 rank 的 `fwd_h` 输出：

| 张量 | shape |
|---|---|
| `h_all` | $[1, 4, H, K, V]$（4 = 2 段 × 每段 128/64 = 2 个 chunk） |
| `v_new_all` | $[1, 256, H, V]$（本 rank 256 个 token） |

---

## 7. 时间开销对比

设单卡处理 128 token 的 `fwd_h` 时间为 $T$。

| 方案 | 墙钟时间 | GPU 利用率 |
|---|---|---|
| 朴素 P2P 串行 | $4T$ + 2 次通信 | ~50%（两 rank 总有一个在等） |
| CP zigzag | $2 \times 2T$（每 rank 跑 2 段 × 2 遍） | ~100% |

2 GPU 时优势不明显，但**当 cp_size 变大时差异巨大**：

| cp_size | 朴素串行 | CP zigzag |
|---|---|---|
| 2 | $4T$ | $4T$ |
| 4 | $8T$ | $4T$ |
| 8 | $16T$ | $4T$ |

CP zigzag 的时间**只取决于每 rank 本地 segment 数**（固定为 2），与 cp_size 无关。

---

## 8. 通信开销

每个 rank all_gather 的数据量（H=16, K=128, V=128, fp32, $N=1$）：

$$2\;\text{segs} \times 1 \times 16 \times 128 \times (128 + 128) \times 4\,\text{B} \approx 1\,\text{MB}$$

**与 token 数无关**。相比 RingAttention 那种全 QKV all-gather（随 token 线性增长），CP zigzag 的通信是常数。

---

## 9. 一句话总结

> $(b, M)$ 是把一个 segment 的完整递推**精确压缩成的仿射变换** $h_{\text{out}} = M h_{\text{in}} + b$。它不依赖 $h_0$，所以可以各 rank 并行预算；通过 all_gather + 串联，反推出每段真实的 $h_0$；然后每个 rank 用正确的 $h_0$ 跑一遍完整的 `fwd_h`，得到所有 chunk 的 state 和 v_new。

代价：2× `fwd_h` 计算量。  
收益：消除了跨 rank 的串行依赖，通信量从 $O(T)$ 降到 $O(1)$。
