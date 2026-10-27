# MHA实现与优化记录

本目录是FlyDSL多头注意力（MHA）kernel：gfx942 BF16/FP8分页注意力、gfx942 BF16 D256线性varlen注意力、
gfx950 BF16分页注意力（含SWA/sink）以及gfx942/gfx950单wave SWA。本页记录实现、流水线、优化方法与历史；
使用方法和当前性能见[benchmarks/mha/readme.md](../../../../../benchmarks/mha/readme.md)，单元测试在
[tests/ops/mha](../../../../../tests/ops/mha)。QSA的union/direct kernel复用这里的D256线性流水线和`_common.py`。

## 0. 记号与命名

### 0.1 数据搬运

沿用MoE kernel（[gemm2_8x1.py](../../moe/flydsl/moe_gemm_2stage/gemm2_8x1.py)）的命名：g=global，s=LDS（shared），r=寄存器。

| 名字 | 含义 | 例子 |
|---|---|---|
| `read_x_g2r` | global → 寄存器（buffer_load到VGPR，或s_load到SGPR） | `_read_q_g2r`、`_read_page_g2r`、`_read_kv_g2r` |
| `read_x_g2s` | global直接写LDS（`buffer_load_dword ... lds`，DMA） | `_read_packet_g2s`、`_read_k_packet_g2s`、`_dma_g2s` |
| `store_x_r2s` | 寄存器 → LDS | `_store_kv_r2s`、`_store_tile_r2s` |
| `read_x_s2r` | LDS → 寄存器 | `_read_k_s2r`、`_read_v_s2r` |
| `store_x_r2g` | 寄存器 → global | `_store_o_tile_r2g`、`_store_o_r2g` |
| `_shuffle_o_r2s2r` | 经LDS重排输出（r2s后s2r） | D256 linear的O |
| `_read_k_s2r_v_g2s` | 交织两件事：读K（s2r）与DMA V（g2s）逐条交替发出 | D256 paged |

read类helper只**发射**指令、返回异步结果；所有`s_waitcnt`都由kernel主体在阶段边界显式放置（`_wait`）。

### 0.2 流水线

- tile t：第t个BN=64 key的KV块；phase t：主循环第t次迭代。phase t做tile t的QK与行最大值，
  同时**落后一个tile**做tile t−1的exp、行和与PV（软件流水）。
- stage：8-wave kernel把phase切成交替的memory stage（VMEM/DS/DMA）和compute stage（MFMA/VALU），
  BF16/FP8 BM256为S0–S7（偶数memory，奇数compute），D256为4个step，每个step = 一个memory stage + 一个compute stage。
- STAGGER：wave 4–7比wave 0–3晚一个stage（prologue多过一个barrier、epilogue少过一个），
  于是一组做MFMA时另一组在搬运数据，两组共用`s_barrier`划分的阶段。
- slot：LDS中的K/V缓冲；`cur = t & 1`、`prev = (t − 1) & 1`是双缓冲槽。
- 主体结构统一为`prologue()` → `phase(*state, t, ...)`循环 → `epilogue(state)`：
  state是按固定顺序传递的元组（循环携带值），phase的docstring说明每个分量的tile编号，
  主体docstring给出逐stage表（预取几个tile、哪个编号的数据在哪个stage发起/落地/被读）。
- softmax统一用log2单位（scale已含log2e）。序列b的第i个query、head h（一行）对第j个key：

  ```text
  q   = Q[cu_seqlens_q[b] + i, h]               # BF16/FP8原值（未乘descale），k_j、v_j同
  k_j = K[序列b第j个key, h // (H/HK)]，v_j同理   # 分页：页kv_page_indices[kv_indptr[b] + j/page]的第j%page个
  S_j = q · k_j                                 # MFMA，FP32累加
  S_j = −inf：j ≥ kv_len，或causal且j > kv_len − q_len + i   # 因果对齐右下角；gfx950/SWA另有滑窗与sink
  scale = FP32(FP32(softmax_scale·log2e)·k_descale·q_descale)  # q_descale为per-tensor或per-token
  S'_j = S_j·scale − m，P_j = exp2(S'_j)        # = e^(x_j)·2^(−m)，x_j = softmax_scale·(q·q_descale)·(k_j·k_descale)
  l   = Σ_j P_j                                 # FP32的P
  O   = Σ_j P̃_j·v_j                             # P̃：P转BF16（BF16 kernel）或FP8（FP8 kernel），FP32累加
  out = O / l · v_descale = Σ_j softmax(x)_j·(v_j·v_descale)，LSE = (m + log2 l)·ln2
  m换成m'时：c = exp2(m − m')，O ← c·O，l ← c·l
  ```

  m不影响结果（分子分母同乘2^(−m)），只决定P的范围：m取舍入后`S·scale`的行最大值附近，
  只在新行最大值超过m一个阈值时才前移（lazy rescale），从而大部分tile不需要缩放O和l。
  D256与SWA先舍入`S·scale`再减m（与aiter union相同）；gfx942 BF16 BM256、FP8与gfx950用一条FMA，
  某行|m| ≥ 2^22后再把S'舍入到ulp(m)的整数倍，结果与先舍入相同（见第3.3、6节）。QSA union复用D256的helper，同样先舍入。

### 0.3 主体与预取距离一览

| kernel | 主体 | 页号 | K | V | 计算 |
|---|---|---|---|---|---|
| BF16 D128/D192 V128无LSE | [_body_v128](mha_pa_bf16_942.py#L363) | t+2（S0 s_load） | t+1：S4 g2r，S6 r2s（单槽） | t：S0 g2r（64-bit GLOBAL），S2 r2s到`[cur]` | QK(t)，exp/PV(t−1) |
| BF16 LSE或V192 | [_body_wide](mha_pa_bf16_942.py#L649) | t+2 | t+1：S0 g2r，S4/S6 r2s | t：S0 g2r；环形S2发布，单槽下一phase S0发布 | 同上 |
| FP8 | [_body](mha_pa_fp8_942.py#L367) | t+3 | t+2：S0 g2r，S4 r2s（双槽） | t：S0 g2r，S2 r2s（双槽） | 同上 |
| BF16 D256 paged | [_body](mha_pa_bf16_256_paged_942.py#L182) | t+2（step0） | t+1：step2 DMA，step3落地 | t−1：step0 DMA，step1落地 | 同上 |
| BF16 D256 linear | [_body](mha_pa_bf16_256_linear_942.py#L280) | 行地址t+2（step0 g2r） | t+1：step2 DMA | t−1：step0 DMA | 同上 |
| gfx950 BF16 | [_attention_body](mha_pa_bf16_950.py#L309) | t+2（S1） | t+2：S4/S6 DMA（双槽） | t：S0或S2 DMA（双槽） | 同上 |
| SWA QT16 | [_swa_kernel](mha_pa_swa_bf16.py#L245) | 下一tile | 下一tile：g2r（寄存器，无LDS） | 下一tile：g2r | 当前tile |
| SWA QT32 | [_swa32_kernel](mha_pa_swa_bf16.py#L345) | 当前tile | 当前tile g2r后等待 | 与QK/softmax重叠 | 当前tile |

## 1. 文件与调用链

| 文件 | 入口 | 架构/类型 | 维度、页与功能 | LDS |
|---|---|---|---|---|
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `PagedAttention`、`flash_attn_varlen_func` | gfx942 BF16 | Dq/Dv∈{128,192}或Dq=Dv=256；page 32/64/128；causal、LSE；persistent/普通网格 | 49,920（D128）/58,240（D192） |
| [mha_pa_bf16_256_paged_942.py](mha_pa_bf16_256_paged_942.py) | 由`PagedAttention`在D256时分派 | gfx942 BF16 | D256 BM128/BN64，K/V DMA | 65,536 |
| [mha_pa_bf16_256_linear_942.py](mha_pa_bf16_256_linear_942.py) | `run`（由`flash_attn_varlen_func`调用；QSA复用helper） | gfx942 BF16 | 线性THD K/V、page1/page4；causal、LSE | 65,536 |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `PagedAttention` | gfx942 FP8 e4m3fnuz | Dq∈{128,192}、Dv=128、page64；per-token/per-tensor Q scale | 33,280（D128）/41,600（D192） |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `PagedAttention` | gfx950 BF16 | Dq∈{128,192}、Dv=128、page64；SWA、sink、LSE；strided Q/O | 99,840（D128）/149,760（D192） |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `PagedAttention` | gfx942/gfx950 BF16 | causal SWA单wave，Dq∈{128,192}、Dv=128、page64；sink、LSE | 0 |
| [_common.py](_common.py) | 共用helper | gfx942 | 也被QSA import | — |
| [_dsl.py](_dsl.py) | `select`、`rmem`、`resource`、`wait` | — | gfx950/SWA的小型layout/buffer适配 | — |

分页调用链（以gfx942 BF16为例）：

```text
PagedAttention(...)                    # functools.cache工厂：校验静态配置，返回_PagedAttention
  └─ _PagedAttention.__call__(...)     # host校验dtype/形状/连续性/2 GiB跨度/scale；不读GPU metadata
       └─ flyc.compile(_launch_attention, *args)   # 按(device, 参数签名)缓存；首调编译并执行
            └─ _attention_kernel_942    # persistent：min(CU, 任务数)个CTA grid-stride；否则grid=(H, B, QB)
                 └─ _work(head, batch, qb)          # 读CQ/KI/LAST得到q_len、kv_len、页表视图
                      └─ _body_v128 / _body_wide(STAGGER=wave>=4)
                           prologue → phase(t=1..tiles-1) → epilogue → O/l → store_o_r2g (+LSE)
```

头数、维度、page、causal、scale、LSE、persistent、CU数以及`B`、`MAX_Q`、物理页数`NP`等形状参数是编译期常量
（各kernel略有不同）；不同取值会编译新的specialization（见使用文档的集成注意事项）。

## 2. 数据布局

- **SHUFFLE-5D分页KV**（全部分页kernel）：v = 16字节/元素宽（BF16为8，FP8为16）。
  K：`[P, HK, Dq/v, page, v]`，V：`[P, HK, page/v, Dv, v]`。一个页内同一组v维（K）或同一组v个token（V）连续，
  一次16字节读即得到MFMA操作数片段。由普通`[P, page, HK, D]`页转换见
  [tests/ops/mha/_mha.py](../../../../../tests/ops/mha/_mha.py)的`vectorize_kv`。
- **metadata**：`cu_seqlens_q[B+1]`（Q前缀和），`kv_indptr[B+1]`（每个序列在`kv_page_indices`中的页区间），
  `kv_page_indices`（物理页号），`kv_last_page_lens[B]`（最后一页有效token数，1..page）。
  KV长度 = `(页数−1)·page + last`。metadata全部留在GPU上，kernel不回读host，热路径无同步、可被graph捕获。
- **scale**：FP32；Q可为per-tensor `[1]`或per-token `[T,H,1]`（FP8在量化域计算，BF16通常全1），K/V各一个标量。
  `softmax_scale`默认`Dq^-0.5`。
- **输出**：O为BF16 `[T,H,Dv]`；可选LSE为FP32 `[T,H]`、自然对数。gfx950与SWA允许空KV序列（输出0；gfx950的LSE为sink或`-inf`），
  gfx942 BF16要求活动序列KV非空，causal时每个序列KV≥Q。
- **D256线性API**：Q/K/V连续BF16 `[tokens, heads, 256]`；`block_table[B, max_pages]`时K/V是物理行
  `[P·page, HK, 256]`，page∈{1,4}，页号可任意置换。

## 3. gfx942 BF16 D128/D192（[mha_pa_bf16_942.py](mha_pa_bf16_942.py)）

### 3.1 任务与调度

- 一个任务 = (head, batch, 256行query块)；512线程=8 wave，每wave 32行，QK/PV用`v_mfma_f32_32x32x8_bf16`。
- `persistent=None/True`（默认）：`min(CU, 任务数)`个常驻CTA按grid-stride取任务，任务序号按head最快、
  再batch、再query块展开；无原子计数器、无workspace、无初始化dispatch，多stream和graph之间没有共享可变状态。
  `persistent=False`：普通网格`(H, B, QB)`。两者共用同一个主体。
- `waves_per_eu=2`、关闭packed-fp32（`-packed-fp32-ops`）是固定编译属性；后者经`llvm.target_features`设置
  （`passthrough`形式会在lowering中丢失，此前实际未生效，ISA里仍有`v_pk_mul_f32`）。

### 3.2 LDS布局

K/V都以“8维×64 key = 1024字节 + 16字节padding”的1040字节chunk存放（`PADDED_CHUNK`），padding错开bank。
V128无LSE热路径为1个K槽（`Dq/8`个chunk）+ 2个V槽（每槽16 chunk，`V_SLOT=16,640`）：D128 49,920、D192 58,240字节；
LDS（64 KiB/CU）和VGPR（每SIMD 2个wave）都只容纳一个8-wave CTA。`_body_wide`在D128/V128（带LSE）时保留V双槽，
其余情况（V192，或D192带LSE）用单V槽。

### 3.3 流水线（`_body_v128`）

```text
stage  phase t; cur = t & 1, prev = (t - 1) & 1
S0     s2r K(t) keys 0-31; s_load pages(t+2); g2r V(t)
S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-23
S2     V(t) lands; r2s V(t) -> [cur]; s2r K(t) keys 32-63
S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, lane row sum, P to BF16
S4     exchange row sums; s2r V(t-1) dims 0-63 [prev]; g2r K(t+1)
S5     PV: O dims 0-63 += P V      softmax: l += row sum; mask S(t), lane row max
S6     exchange row maxima; s2r V(t-1) dims 64-127 [prev]; K(t+1) lands; r2s K(t+1)
S7     PV: O dims 64-127 += P V    softmax: new m, S'(t); rescale O and l
```

- K只有一个槽：K(t+1)在S4发起global读、S6写LDS——此时两组wave都已读完K(t)。V用双槽环：V(t)在S0读、S2写入`[cur]`，
  下一phase的S4/S6从`[prev]`读。
- 32×32 MFMA的结果一行分布在lane的两半（lane和lane±32），行和/行最大值先lane内归约，再在memory stage用
  `ds_bpermute`交换（`_cross32_async`），与同stage的独立DS读共用一次等待，不在compute stage插入shuffle。
- exp按24+8分到S1/S3两次QK旁边（`_schedule`用`sched_group_barrier`把MFMA与VALU/TRANS交错），
  掩码和新m的计算放在S5/S7的PV旁。
- D128的S3原来是最长的stage：8个exp、31个行和加法、约80条BF16转换VALU挤在16个MFMA之间，ATT中领先组S3与落后组S2
  同时运行的slot中位数752周期（其余slot约520–650）。D128只在S3做lane行和的第一层（16个成对和），S5完成其余4层，
  S6与行最大值一起交换，S7在求新m之前加到l上（`_next_max`的高精度判据要用含P(t−1)的l）；加法与顺序不变，
  输出逐位一致。该slot降到556周期，VGPR +16（无spill，见3.4）。D192这样做会溢出到scratch，保持上表。
- D128主循环的stage从8字节对齐处开始（`_stage_end_aligned`：barrier后`_align8`，即`.p2align 3`，至多一条`s_nop 0`）。
  gfx942按32字节块取指：8字节VALU/MFMA组成的密集段若从地址≡4 (mod 8)开始，每第4条跨两个块，ATT中wave在它之前
  空等4周期，S3每次多约86周期。不对齐时每个循环副本的奇偶由它前面的全部代码决定：同一循环代码只在kernel入口加
  几个`s_nop`，D128就在4.50–4.66 ms之间变化（每个跨块的副本约1.6%）；对齐后只由stage内部决定，D128的S3都不跨块。
  D192的S3对齐后反而跨块（+1.5%），所以D192与`_body_wide`仍用`_stage_end`，ISA与改动前相同。
  两项合计（GPU7交替A/B中位数，相对改动前）：D128 per-token −5.7%、per-tensor −5.7%、causal −6.5%，
  高精度模式D128 −5.6%、causal −5.2%；[164800, 7, 128]持平（−0.1%，持续负载时功耗接近650 W上限）。
- m从首个行最大值+1开始，只有后续行最大值超过m+7才前移到新最大值+1，并按ballot只在有行需要时缩放O和l（`_rescale`）。
- import前设置`AITER_FLYDSL_MHA_EXACT_MAX=1`开启高精度模式（`_next_max`，`_body_v128`与`_body_wide`）：除超过m+7外，
  新行最大值的P = 2^(max − m)超过约l/8（l已含P(t−1)）时也前移，整个wave取max(m, 行最大值+1)：该key的P恰为1/2
  （BF16可精确表示），除ascending外误差与AITER相差不超过1%（3.6）。主循环每phase约多4条VALU，GPU7交替A/B中位数：
  D128 +0.7%、causal −0.5%、D192 +1.5%、LSE +0.6%、V192 +0.3%、[164800, 7, 128] +0.2%。8ac1c31c里m取行最大值本身，
  wave内任一行前移就进冷分支，stage barrier又让整个CTA等最慢的wave：D128 +7.6%、causal +5.5%、D192 +8.6%、LSE +6.3%、
  V192 +3.2%、[164800, 7, 128] +0.2%。默认关闭时主循环与改动前逐条相同，输出逐位一致。FP8 kernel也读取该变量（第6节）。
- S'(t)由S7一条FMA `S·scale − m`得到。FMA用精确乘积而m是舍入值，S'(max)偏离0最多ulp(m)/2，logit近1e11时
  exp2上溢或下溢。因此m前移时`_rescale`的分支顺带检查|m| ≥ 2^22；一旦成立（sticky，ballot掩码随state传递），
  另一个分支把S'(t)舍入到ulp(m)的整数倍（`_reround`：`(x + c) − c`，`c = 1.5·2^exponent(m)`），等于先舍入`S·scale`
  再减m（至多差一个ulp(m)）；prologue的S'(0)同理。热路径只多一次标量比较与跳转，正常输入输出与改动前逐位一致。
  直接改成先舍入再减m（每个分数多一条VALU）实测D128 −4.4%、D192 −3.5%、D128带LSE −2.7%；
  两个分支合并为一个时V192（已溢出到scratch）因多一次dwordx4溢出慢5%。
- D128每次循环展开两个phase，使`cur/prev`成为常量；D192每次一个phase，槽号由t计算。
- P与O都用`_pack_bf16`转BF16（软件RNE，NaN不规范化）；O按`O/l·v_descale`转换，由`_store_o_tile_r2g`写回。
  RNE比`+0x8000`（舍入到最近、平局远离零，RTA）每个P多一条VALU，落在VALU已满的S3：相对RTA，D128慢3.8%、
  causal 4.4%、LSE 4.9%、V192 3.5%、D192 0.6%、[164800, 7, 128] 1.7%。import前设置环境变量
  `AITER_FLYDSL_MHA_P_BF16_RTA=1`可让本kernel的P改用RTA（主循环与改动前逐条相同，输出逐位一致）；两种舍入只在恰好的平局上不同。

`_body_wide`（LSE或V192）用协作拷贝（cooperative g2r→r2s），行和/行最大值在compute stage用lane shuffle归约，
其余阶段角色相同；S6读V的64–Dv维，S7完成其余PV。

### 3.4 资源（MI308X，FlyDSL 0.3.4.1，ISA实测；2048²，D128 full为H8/HK8，其余H16/HK1）

| specialization | VGPR | SGPR | LDS | scratch/spill |
|---|---:|---:|---:|---:|
| full D128 persistent / grid | 246 / 244 | 72 / 58 | 49,920 | 0 |
| full D192 persistent / grid | 254 / 252 | 75 / 60 | 58,240 | 0 |
| causal D128 persistent / grid | 251 / 246 | 75 / 62 | 49,920 | 0 |
| causal D192 persistent / grid | 256 / 254 | 80 / 63 | 58,240 | 0 |

### 3.5 D128伪代码与精度损失（`_body_v128`，Dq = Dv = 128，无LSE）

按一行query写（实际每wave 32行，一行的64个key由lane r与r+32各持32个）；log2单位。注释中`[E*]`后是该步引入的误差范围：
u = 2^-24为FP32单位舍入，n为KV tile数（[164800, 7, 128]时n = 2575），“典型”按舍入误差随机估计；推导见下表。
未标注的步骤是精确的（`−inf`遮挡、越界V行置0、lane间交换、BF16输入搬运）。D192只是QK的K维为192，且行和在S3算完、S4交换、S5相加（3.3）。

```text
scale = FP32(FP32(softmax_scale·log2e) · k_descale · q_descale)   # [E1] scale相对 ≤ 3u ≈ 1.8e-7
q = Q[row, 0:128]                         # BF16，整个KV循环常驻寄存器
# prologue：tile 0
S = q · K(0)ᵀ                             # 32×32×8 BF16 MFMA；[E2] |ΔS| ≲ 16u·Σ|q_i·k_i|（D192为24u）
S = mask(S)                               # 越界/因果遮挡置−inf
m = max(rowmax(S)·scale, FLOOR) + 1       # [E3] 无（softmax对m平移不变）
S' = fma(S, scale, −m)                    # [E4] P相对 ≤ ln2·ulp(S')/2 ≤ 2.7e-6（|S'| < 128）
if 有行|m| ≥ 2^22: S' = reround(S', m)    # 冷分支，此后sticky；[E4] |ΔS'| ≤ ulp(m)/2（≥ 0.25），同FP32参考
l = 0; O = 0
for t = 1 .. tiles−1:                     # D128每趟展开两个phase；S0–S7见3.3
    P = exp2(S'(t−1))                     # S1算24个、S3算8个；[E5] P相对 ≤ 0.783 ulp ≈ 9.3e-8
    l_lane = Σ P(本lane 32个)              # S3第一层、S5其余4层的成对求和树；[E6] 相对 ≤ 5u ≈ 3e-7
    P16 = BF16(P)                         # S3，RNE（RTA开关见3.3）；[E7] 每个P相对 ≤ 2^-8 ≈ 3.9e-3
    S = q · K(t)ᵀ                         # S1、S3；[E2] 同上
    l += l_lane + l_lane(另一半lane)        # S6交换，S7相加；[E6] l相对 ≲ (n+6)u ≈ 1.5e-4，典型 √n·u ≈ 3e-6
    O[0:64]   += P16 · V(t−1)[:, 0:64]    # S5；[E8] |ΔO| ≲ 8n·u·ΣP16·|V|（8n·u ≈ 1.2e-3，典型 √(8n)·u ≈ 9e-6）
    O[64:128] += P16 · V(t−1)[:, 64:128]  # S7；[E8] 同上
    S = mask(S); cand = rowmax(S)·scale   # S5、S7；[E3] 无
    m' = cand + 1 if cand > m + 7 else m  # [E3] 无
    S' = fma(S, scale, −m')               # [E4] 同上
    if 有行m' ≠ m:                        # ballot，冷分支
        c = exp2(m − m'); O ·= c; l ·= c  # [E9] c相对ε ≤ 2.8e-6，即前移前的P统一乘(1+ε)；O·c、l·c各 ≤ u
        有行|m'| ≥ 2^22 → 此后S'都reround
    m = m'
# epilogue：tile last
P = exp2(S'(last)); l += ΣP(两半lane); P16 = BF16(P)   # [E5][E6][E7] 同上
V(last)中超出kv_len的行置0; O += P16 · V(last)          # [E8] 同上
inv = (l > 0 ? 1/l : 0) · v_descale       # IEEE除法；[E10] 相对 ≤ 2u ≈ 1.2e-7
out = BF16(O · inv)                       # RNE（_pack_bf16）；[E11] 相对 ≤ 2^-8 + u ≈ 3.9e-3
# [E6/E7] O用P16、l用P：out整行乘(1+δ̄)，δ̄ = ΣP·δ/ΣP（P16 = P·(1+δ)），|δ̄| ≤ 2^-8；
#         权重分散时δ̄ ≈ 0，单个key主导时δ̄ ≈ 该key的δ
```

| 标记 | 运算 | 推导与说明 |
|---|---|---|
| E1 | scale | softmax_scale·log2e转FP32、乘k_descale、乘q_descale各舍入一次（descale为1时精确）。同一行共用scale，只改变同行key之间的相对权重：P相对 ≈ ln2·\|ΔS'\|·3u（ΔS'为该key与行最大值之差），可忽略 |
| E2 | QK MFMA | BF16乘积在FP32中精确；按每条MFMA对累加器舍入一次估计，Dq = 128时每个S经16条MFMA链式累加（D192为24条）。对P的影响为相对 ≈ ln2·scale·\|ΔS\| |
| E3 | m（行最大值·scale、+1、比较m+7） | softmax对m平移不变，m取什么值都不引入误差，只决定E4中S'的大小 |
| E4 | `S' = fma(S, scale, −m)` | 乘积精确、只在最后舍入一次，P相对 = ln2·\|ΔS'\|。m是舍入值而乘积精确，\|logit\|很大时S'(max)偏离−1最多ulp(m)/2（exp2溢出，见3.3）；\|m\| ≥ 2^22后`_reround`对齐“先舍入S·scale再减m”的参考，误差就是FP32表示S·scale本身的误差 |
| E5 | `v_exp_f32` | 硬件近似：[−126, 8]全部FP32输入实测最大0.783 ulp（正确舍入为0.5）；S' < −126时P < 2^-126，贡献可忽略 |
| E6 | l的FP32求和 | 一个P经过的舍入：lane内5层成对树、两半相加、之后每个tile的累加，共 ≤ n + 6次。**l累加的是未舍入的FP32 P，而PV用的是BF16的P16（E7）**，后果见伪代码最后两行 |
| E7 | P→BF16 | kernel内部最大的单项误差；RNE与RTA只在恰好的平局上不同，10个精度用例结果在三位有效数字上相同 |
| E8 | PV MFMA | P16·V乘积精确；每个tile 8条MFMA（64个key，每条8个）链式累加到O，链长8n，估计方式同E2 |
| E9 | 重缩放 | c = exp2(m − m')：减法误差 ≤ ulp(m − m')/2（\|m − m'\| < 128时对c为相对 ≤ 2.7e-6；更远时c ≤ 2^-128，旧部分可忽略），再加exp2的0.783 ulp。O和l乘同一个c，但之后的tile按m'累加，所以c的误差ε等于把前移前所有key的权重统一乘(1+ε)，输出相对误差 ≲ \|ε\|；m只在超过m+7时前移，次数少 |
| E10 | 1/l、·v_descale | 除法正确舍入（`v_div_scale/v_rcp/v_div_fmas/v_div_fixup`），再一次乘法（v_descale为1时精确） |
| E11 | O·inv → BF16 | 输出格式本身的误差，即下表的“理想” |

[164800, 7, 128]、BF16、全可见、P为RNE时的实测（512个采样行对FP64参考的相对L2误差）：

| 输入 | 本kernel | AITER | 理想（FP64结果直接RNE到BF16，只有E11） |
|---|---:|---:|---:|
| normal（随机Q/K/V） | 2.35e-3 | 2.33e-3 | 1.66e-3 |
| peaked（每行一个突出的key） | 2.01e-3 | 1.69e-3 | 1.65e-3 |
| sinks（32个随机位置的key logit大、V×10） | 2.28e-3 | 1.81e-3 | 1.65e-3 |
| ascending（logit随位置增大） | 2.30e-3 | 2.01e-3 | 1.66e-3 |

normal比理想多出的误差主要来自E7，AITER同样把P舍入为BF16，量级相同。peaked/sinks/ascending比AITER大，来自E6与E7的不一致：
本kernel的m懒更新（只在超过m+7时前移），主导key的P尾数任意，其BF16舍入误差直接留在O/l里；
AITER的m每次都取精确行最大值，主导key的P恰为1（BF16可精确表示）。`AITER_FLYDSL_MHA_EXACT_MAX=1`让占比大的新最大值
也前移，除ascending外误差与AITER相差不超过1%（3.6）。

### 3.6 高精度模式的D128伪代码（`AITER_FLYDSL_MHA_EXACT_MAX=1`）

import前设置`AITER_FLYDSL_MHA_EXACT_MAX=1`后，`_body_v128`与`_body_wide`的`_next_max`除m+7外还看新行最大值在行和中的占比，
其余代码不变；FP8见第6节，其他kernel（D256、gfx950、SWA）不受影响。前移时最大key的S' = −1 + r，
|r| ≤ ulp(S_max·scale)/2（FMA的精确乘积减去舍入后的m），P = 2^(−1+r)：|S·scale| < 128时BF16舍入后就是1/2。
占比w小于约l/8的新最大值不前移，其舍入只贡献约w·2^-8。性能代价见3.3。记号同3.5，`★`标出与3.5不同的地方。

```text
scale = FP32(FP32(softmax_scale·log2e) · k_descale · q_descale)   # [E1] scale相对 ≤ 3u ≈ 1.8e-7
q = Q[row, 0:128]                         # BF16，整个KV循环常驻寄存器
# prologue：tile 0
S = q · K(0)ᵀ                             # 32×32×8 BF16 MFMA；[E2] |ΔS| ≲ 16u·Σ|q_i·k_i|（D192为24u）
S = mask(S)                               # 越界/因果遮挡置−inf
m = max(rowmax(S)·scale, FLOOR) + 1       # [E3] 无
S' = fma(S, scale, −m)                    # [E4] P相对 ≤ ln2·ulp(S')/2 ≤ 2.7e-6（|S'| < 128）
if 有行|m| ≥ 2^22: S' = reround(S', m)    # 冷分支，此后sticky；[E4] |ΔS'| ≤ ulp(m)/2（≥ 0.25），同FP32参考
l = 0; O = 0
for t = 1 .. tiles−1:                     # D128每趟展开两个phase；S0–S7见3.3
    P = exp2(S'(t−1))                     # S1算24个、S3算8个；[E5] P相对 ≤ 0.783 ulp ≈ 9.3e-8
    l_lane = Σ P(本lane 32个)              # S3第一层、S5其余4层的成对求和树；[E6] 相对 ≤ 5u ≈ 3e-7
    P16 = BF16(P)                         # S3，RNE；[E7] 每个P相对 ≤ 2^-8 ≈ 3.9e-3；★前移时的最大key ≤ ln2·|r|
    S = q · K(t)ᵀ                         # S1、S3；[E2] 同上
    l += l_lane + l_lane(另一半lane)        # S6交换，S7相加；[E6] l相对 ≲ (n+6)u ≈ 1.5e-4，典型 √n·u ≈ 3e-6
    O[0:64]   += P16 · V(t−1)[:, 0:64]    # S5；[E8] |ΔO| ≲ 8n·u·ΣP16·|V|（8n·u ≈ 1.2e-3，典型 √(8n)·u ≈ 9e-6）
    O[64:128] += P16 · V(t−1)[:, 64:128]  # S7；[E8] 同上
    S = mask(S); cand = rowmax(S)·scale   # S5、S7；[E3] 无
    move = cand − m > med3(e − 3, −1, 7)  # ★l ∈ [2^(e−1), 2^e)：超过m+7，或新最大值的P超过约l/8（_next_max）
    m' = max(m, cand + 1) if 有行move else m   # ★整个wave一起前移；[E3] 无
    S' = fma(S, scale, −m')               # [E4] 同上
    if 有行m' ≠ m:                        # ballot，冷分支；★占比大的新最大值也进入，随机输入很少
        c = exp2(m − m'); O ·= c; l ·= c  # [E9] c相对ε ≤ 2.8e-6，即前移前的P统一乘(1+ε)；O·c、l·c各 ≤ u
        有行|m'| ≥ 2^22 → 此后S'都reround
    m = m'
# epilogue：tile last
P = exp2(S'(last)); l += ΣP(两半lane); P16 = BF16(P)   # [E5][E6][E7] 同上
V(last)中超出kv_len的行置0; O += P16 · V(last)          # [E8] 同上
inv = (l > 0 ? 1/l : 0) · v_descale       # IEEE除法；[E10] 相对 ≤ 2u ≈ 1.2e-7
out = BF16(O · inv)                       # RNE（_pack_bf16）；[E11] 相对 ≤ 2^-8 + u ≈ 3.9e-3
# [E6/E7] ★O用P16、l用P，但前移时最大key的δ ≈ 0：一个key主导时δ̄ → 0；占比小于约l/8的新最大值
#         不前移，ascending（每个tile都有新最大值、单个占比小）的残余比m取行最大值时大
```

[164800, 7, 128]、BF16、全可见、P为RNE的实测（同3.5，512个采样行对FP64参考的相对L2误差）：

| 输入 | lazy（默认） | 高精度模式 | AITER | 理想 |
|---|---:|---:|---:|---:|
| normal | 2.35e-3 | 2.35e-3 | 2.33e-3 | 1.66e-3 |
| peaked | 2.01e-3 | 1.69e-3 | 1.69e-3 | 1.65e-3 |
| offset（logit约724 ± 11） | 2.03e-3 | 1.70e-3 | 1.70e-3 | 1.65e-3 |
| sinks | 2.28e-3 | 1.82e-3 | 1.81e-3 | 1.65e-3 |
| ascending | 2.30e-3 | 2.13e-3 | 2.01e-3 | 1.66e-3 |

其余输入（flat、zero_q、v_outliers、self_similar）三种实现相差不超过0.01e-3；所有logit相等且约5e10时（精确规则路径）
本kernel两种模式都是1.69e-3，AITER的输出错误（相对误差1）。高精度模式下仍高于理想的部分来自E7（未前移key的P16，
normal 2.35e-3 ≈ √2 × 1.66e-3，即E7与E11各占一半方差）和E6/E7的残余（ascending）。8ac1c31c里m取行最大值本身，
normal 2.33e-3、sinks 1.81e-3、ascending 2.02e-3，其余与高精度模式相同。短KV的随机输入上前几个key的权重较大，
逐个精确更有用：2048²×H8的calc_diff为lazy 2.73e-6、高精度模式2.70e-6、m取行最大值本身2.52e-6。

## 4. gfx942 BF16 D256 paged（[mha_pa_bf16_256_paged_942.py](mha_pa_bf16_256_paged_942.py)）

`PagedAttention(Dq=Dv=256)`自动分派到这里（`bf16_backend`以`-pages1-lateS41`结尾）。

- BM128/BN64，8 wave×16行，`v_mfma_f32_16x16x16_bf16`；Q一次读入寄存器（`_read_q_g2r`）并在全部tile复用。
- LDS只有一个V tile（32 KiB）和一个K tile（32 KiB，每wave的K行间距`K_PITCH=1024`），是两个独立的LDS分配：
  K/V的读与DMA因此有各自的alias scope，LLVM不会为另一块的DMA在读之前插`vmcnt`等待。V在地址0，asm算出的V DMA目的地址直接作M0。
- K/V都用direct-to-LDS DMA：`buffer_load_dword ... lds`每lane一dword、每个packet写2 KiB，一个tile 16个packet。
  V的每个tile按其所在页的完整64-bit地址重建buffer描述符（`_v_tile_resources`；page32时一个tile跨两页，用两个描述符），
  buffer指令只携带tile内相对偏移，页号×页跨度不在32-bit里计算。
- “交织”：`_read_k_s2r_v_g2s`把V(t−1)的每个DMA packet与K(t)的一次`ds_read_b128`交替发出；
  `_read_v_s2r_k_g2s`把K(t+1)的DMA与V(t−1)的读交替发出。DS读为VMEM提供M0→VMEM的必要间隔，省去nop。
  LDS读用`_load_lds`（LLVM可见的volatile load），LLVM据此补足MFMA之后的hazard等待（第13节）。

```text
step  memory stage                                compute stage
0     s_load pages(t+2); g2s V(t-1) with s2r      QK: S(t) keys 0-31
      K(t) keys 0-31 between its packets          softmax: P(t-1) = exp2(S'(t-1) - m)
1     s2r K(t) keys 32-63; V(t-1) lands           QK: S(t) keys 32-63
                                                  softmax: row sum of P(t-1), P to BF16
2     exchange row sums; g2s K(t+1) with s2r      PV: O dims 0-127 += P(t-1) V(t-1)
      V(t-1) dims 0-127 between packets           softmax: l += row sum; mask S(t), max
3     exchange row maxima; K(t+1) lands;          PV: O dims 128-255 += P(t-1) V(t-1)
      s2r V(t-1) dims 128-255                     softmax: m, S'(t); rescale O and l
```

当前默认“v73”组合：Q留在寄存器、K/V提前DMA、逐条交织、页表提前两tile且页号ready合并、
S4的LGKM等待移到barrier之后、首个消费者之前。page32/64/128、persistent/普通网格与LSE共用该路径。
资源：full persistent VGPR224/SGPR89、grid VGPR218/SGPR72，LDS 65,536，零scratch/spill。

## 5. gfx942 BF16 D256 linear（[mha_pa_bf16_256_linear_942.py](mha_pa_bf16_256_linear_942.py)）

公开入口为[flash_attn_varlen_func](mha_pa_bf16_942.py#L1200)：连续THD输入、无KV格式转换，热调用只有一个attention kernel。

### 5.1 布局与搬运

- LDS：K tile 32 KiB + 行主序V tile 32 KiB。K在**DMA源地址**做XOR swizzle（`_k_phase`），
  使后续`ds_read_b128`无bank冲突；V按行主序到达LDS，读出后用与CK `transpose_vectors`相同的2×2 BF16
  字节置换（`v_perm_b32`选择子`0x01000504/0x03020706`）在寄存器内转置（`_v_operands`）。
- PV的N坐标被置换，使每个lane转置连续的8×8块；输出经`_shuffle_o_r2s2r`（`ds_write_b128` + XOR行swizzle）
  恢复为普通`[T,H,D]`后由`_store_o_r2g`做16字节写回。
- DMA源行由`_read_kv_rows_g2r`计算：无页表时为线性行号；page1/page4时每个wave用一次向量页表读得到物理页，
  再用`readlane`取出每个packet的SOFFSET（`_dma_offsets`），没有发散的标量waterfall。
- `_read_packet_g2s`把SOFFSET/M0的标量计算、一次可选`ds_read_b128`和DMA放在同一段asm（同一leaf），
  避免独立asm结果之间的hazard nop；`_k_reads`/`_v_reads`是K/V读计划（slot、地址、立即数，按发射顺序）。
  其余K/V读（`_read_k_s2r`/`_read_v_s2r`）和O shuffle的读用`_load_address`（LLVM可见，第13节）。

### 5.2 流水线（`_body`）

```text
step  memory stage                                compute stage
0     g2r rows(t+2); g2s V(t-1) with s2r K(t)     QK: S(t) keys 0-31
      keys 0-31 between its packets               softmax: P(t-1) = exp2(S'(t-1) - m)
1     s2r K(t) keys 32-63; V(t-1) lands           QK: S(t) keys 32-63
                                                  softmax: row sum of P(t-1), P to BF16
2     exchange row sums; g2s K(t+1) with s2r      mask S(t); PV: O dims 0-127 += P V
      V(t-1) dims 0-127 between packets           softmax: l += row sum, lane max of S(t)
3     exchange row maxima; K(t+1) lands;          PV: O dims 128-255 += P(t-1) V(t-1)
      s2r V(t-1) dims 128-255                     softmax: m, S'(t); rescale O and l
```

- STAGGER（落后）组在step1先读K(t)的低128维（`lower_first`），并在barrier前用`lgkmcnt(8)`退休这8个读：
  领先组随后的K(t+1) DMA先覆盖K的D[0:128]，其余读只涉及D[128:256]。step3只读V的高128维，
  领先组下一step0的V DMA只写低半，落后组的DMA在本组step3退休之后才发起。
- softmax的VALU折叠进PV：`_pv(summary_args=...)`在最后一步顺带完成`l += 行和`和S(t)的lane最大值，
  `_pv(center_args=...)`在第一步算出新m，随后四步逐段完成`S' = S·scale`（固定在本stage），
  下一phase的step0做`exp2(S' − m)`，最后按ballot lazy rescale。QSA union复用同一个`_pv`。
- 主循环：noncausal且无LSE时每次展开4个phase（摊销SOFFSET队列移动）；causal或LSE每次2个phase
  （4个会让标量寄存器溢出）。
- 静态B1 noncausal且最后一tile不超过32个key时，最后一tile走NARROW phase（无keys 32–63），
  epilogue只drain第一个32-token V平面。B1非分页、noncausal、无LSE时Q/K长度作为编译期常量（`STATIC_Q/STATIC_K`）。
- persistent且B1非分页causal时，任务按“最长query块优先、蛇形分配到CTA”排序（[_linear_256_kernel](mha_pa_bf16_256_linear_942.py#L575)），
  平衡因果负载而不改变tile内算术。

### 5.3 API行为

- host对`cu_seqlens_q/k`与`block_table`做一次完整校验（`.cpu()`），结果按tensor身份+版本号缓存（`_validate_metadata`）；
  原地修改metadata会使缓存失效并重新校验。graph capture中遇到未预热/已修改的metadata直接报错。
- 不支持的参数（dropout、window/sink、alibi、bias、soft cap、page_size∉{1,4}、num_waves≠8、autograd）抛`NotImplementedError`；
  输出与Q/K/V/metadata/LSE重叠、`softmax_scale`非正有限值、长度超过上界等抛`ValueError`。
- 资源：full persistent VGPR228/SGPR52；causal B1 VGPR232/SGPR63；page1 VGPR242/SGPR106；page4 VGPR247/SGPR68；
  均LDS 65,536、零scratch/spill（单元测试在模块结束时检查本次编译的全部specialization）。
  LDS读改为LLVM可见后（第13节），causal/page1/page4/ragged的VGPR又多3–7个，仍无spill。

## 6. gfx942 FP8（[mha_pa_fp8_942.py](mha_pa_fp8_942.py)）

BF16 BM256版本的蓝本：同样的8 wave、8 stage、STAGGER与“QK(t)+PV(t−1)”结构，QK/PV为32×32×16 FP8 MFMA。

```text
stage  phase t; cur = t & 1, prev = (t - 1) & 1
S0     s2r K(t) keys 0-31 [cur]; s_load page(t+3); g2r V(t), K(t+2)
S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-15
S2     s2r K(t) keys 32-63 [cur]; V(t) lands; r2s V(t) -> [cur]
S3     QK: S(t) keys 32-63         softmax: P(t-1) values 16-31
S4     s2r V(t-1) dims 0-63 [prev]; K(t+2) lands; r2s K(t+2) -> [cur]; mask S(t)
S5     PV: O dims 0-63 += P V      softmax: P(t-1) to FP8, row max of S(t), row-sum adds
S6     s2r V(t-1) dims 64-127 [prev]; new m
S7     PV: O dims 64-127 += P V    softmax: l += row sum, S'(t); rescale O and l
```

- K与V各两个LDS槽，K比QK领先两个tile、页号领先三个tile；全部为寄存器中转的协作拷贝（g2r→r2s），
  不用direct-to-LDS DMA、LDS转置指令或原生FP32→BF16转换。LDS：D128 33,280、D192 41,600字节（16维×64 key的1040字节chunk）。
- P量化为FP8参与PV；m的前移阈值为6，保证FP8-FNUZ下P有限（P ≤ 64）。输出`O/l·v_descale`。
- **高精度模式**：import前设置`AITER_FLYDSL_MHA_EXACT_MAX=1`（与BF16共用；`_first_max`/`_next_max`）。lazy时m可比行最大值
  低至多6，主导key的P尾数任意，其FP8舍入（至多6.25%）因l累加FP32的P而留在O/l里（精度见本节末）。开启后，
  新行最大值的P = 2^(max − m)超过约l/8时也前移（`max − m + 3 > med3(e, 5, 9)`，e = frexp_exp(l)，l ∈ [2^(e−1), 2^e)；
  tile 1的l还没有P(0)，下界取为9），前移时m取行最大值减2：占比较大的key的P恰为4（FP8可精确表示），
  P ≤ 64不变。占比w小于约1/8的key的舍入只贡献约w·6.25%，不值得为它重缩放。758a1856在每个新最大值都前移
  （m取行最大值减7）：wave内任一行前移就进冷分支，stage barrier又让整个CTA等最慢的wave，短KV上几乎每个tile都缩放，
  D128 +17.5%、causal +12.2%、D192 +20.4%、LSE +17.2%、per-tensor D128 +23.3%、[164800, 7, 128] +2.0%。现在主循环
  每phase约多4条VALU、VGPR +2，GPU7交替A/B中位数：D128（16384²×H7）+0.2%、causal +0.7%、D192 +1.4%、LSE −0.1%、
  per-tensor D128（10240×2560）+1.1%、[164800, 7, 128] +0.7%。默认关闭时主循环与改动前逐条相同，输出逐位一致。
- **S'的中心化**：S7用一条FMA `S·scale − m`，精确规则与BF16 BM256相同（第3.3节）。|m| ≥ 2^24时FMA误差可使
  FP8的P超过240，logit近1e11时exp2上溢或下溢。直接先舍入再减m每个分数多一条VALU，实测FP8慢7%
  （GPU7 16384²×H7：414.6→385.5T）。FP8把lazy rescale与精确规则合并为一个分支并加`llvm.expect`冷路径提示
  （`_unlikely`）：不加提示慢1%，用BF16的两个分支慢0.6%；BF16则相反（带LSE时加提示慢1.2%）。
  输出与改动前逐位一致。
- 只保留LDS模式（旧register模式已删除）；不支持persistent、SWA和sink。
- **compute stage受VALU发射口限制**（ATT实测：`v_exp_f32`占16周期、`v_cvt_pk_fp8_f32`占8、VALU与MFMA发射各4；
  同一累加器的32×32 MFMA链每环约36周期）。FP8每个compute stage只有8个MFMA（256周期），softmax VALU必须均摊：
  - S1/S3：每个QK依赖链环节放2个exp（16+16+4≈36），QK与exp同时饱和。
  - S5/S7：`_pv_lo_with_max`/`_pv_hi_with_sum`用`sched_barrier`按程序顺序发射PV MFMA（两个累加器交替），
    下一个K-step的FP8转换跟在第一个MFMA后，MAX3树跟在第二个后，xor32 `ds_bpermute`提前一个K-step发出；
    行和拆成S5的14个加法和S7的17个加法（`_S5_SUM_ADDS`），S7每个MFMA后7条VALU（最后一个后只剩1条），
    再做S'(t)的32个FMA。ATT（一个CU，D128）中S5所在slot约368周期（16个`v_cvt_pk_fp8_f32`各占8），
    S1/S3为296，S7（对齐后，见下）约304。原来S5做16个、S7每个MFMA后6条；挪走2个加法后S5与S7所在slot各短8–12周期，
    GPU7交替A/B各形状快0.6–0.9%；挪4–6个相近，挪8个快0.3–0.8%，16个全挪+1.4%（S7变成发射口受限），
    挪进S6（memory stage）+1.2%。
  - memory stage只放短VALU（S4的mask、S6的新m）：同SIMD上另一组的compute wave占满发射口时，memory wave的VALU会被饿死（实测停顿130周期）。
  - S7取指对齐：S7的`ds_bpermute`与其后的32个FMA、PV MFMA是一段8字节指令的密集段，它前面4字节的
    `v_add_f32`/`s_nop`个数决定这段是否从≡4 (mod 8)开始、每第4条跨32字节取指块（3.3）。`_align8`在
    `ds_bpermute`前对齐：HEAD各build里不少S7每次有8–9条跨块，对齐后为0，ISA只多填充的`s_nop`。
    GPU7交替A/B（四轮）：D128 −0.8%到−1.0%、LSE −0.65%到−0.8%、D192与causal −0.15%到−0.5%、
    per-tensor与[164800, 7, 128]持平；高精度模式0到−0.7%。S5的密集段短且`v_cvt_pk_fp8_f32`占8周期，跨块不暴露：
    在S5每组VALU前也对齐，多数形状反而慢0.2–0.7%。
  - per-tensor时scale用`readfirstlane`放进SGPR，S'(t)的FMA直接读它（`_center`），每条FMA少读一个VGPR；
    ATT中S7所在slot再短约16–20周期，per-tensor D128（16384²×H7）再快0.65–1.0%（两轮）。per-token的scale按行不同，代码不变。
  - 行和用inline asm标量`v_add_f32`：LLVM SLP会把它合成`v_pk_add_f32`，而packed FP32在SIMD有MFMA执行时停顿（实测176周期）；
    同理S'(t)不用`v_pk_fma_f32`（A/B慢5%）。各加法/最大值的结合顺序与改动前相同，输出与改动前逐位一致。

[164800, 7, 128]、全可见、Q/K/V按per-tensor amax量化为FP8（输入同3.5）的实测（512个采样行对反量化输入FP64参考的相对L2误差）：

| 输入 | lazy（默认） | 高精度模式 | AITER FP8 | 理想 |
|---|---:|---:|---:|---:|
| normal | 2.60e-2 | 2.60e-2 | 2.63e-2 | 1.66e-3 |
| peaked | 1.69e-2 | 5.98e-3 | 6.06e-3 | 1.65e-3 |
| offset | 1.39e-2 | 6.50e-3 | 6.63e-3 | 1.67e-3 |
| ascending | 2.23e-2 | 1.31e-2 | 1.33e-2 | 1.66e-3 |
| sinks | 2.25e-2 | 1.29e-2 | 1.29e-2 | 1.66e-3 |
| self_similar（K = Q + 噪声） | 6.24e-3 | 4.60e-3 | 6.24e-3 | 1.66e-3 |

normal类输入的误差主要是P的FP8舍入（3位尾数，RMS约2.6%），三种实现量级相同。v_outliers、zero_q不差于AITER；
flat不前移，与lazy相同（2.26e-2，AITER 2.21e-2）。所有logit相等且约5e10时两种模式都是1.67e-3，AITER FP8输出NaN。
per-token Q scale的结果相差不超过3%。758a1856（每个新最大值都前移、m取行最大值减7）在这些输入上与本模式相差
不超过3%，只有self_similar更好（1.76e-3，减7让小P少下溢）。短KV的随机输入上前几个key的权重较大，
逐个精确更有用：2048²×H8的calc_diff为lazy 3.44e-4、本模式3.37e-4、m取行最大值本身（AITER的做法）2.95e-4。
相对未量化的原始输入，误差另含输入量化（normal为5.3e-2）。

## 7. gfx950 BF16（[mha_pa_bf16_950.py](mha_pa_bf16_950.py)）

由OPUS派生，直接读取page64 SHUFFLE-5D缓存，无gather、无线性KV workspace、无辅助dispatch。

- BM256，8 wave，K/V各两个槽并全部用direct-to-LDS DMA（`_dma_g2s`）；Q先DMA到V区域再读入寄存器，
  因此V区按Q的大小分配：D128 99,840、D192 149,760字节。
- 阶段表见[_attention_body](mha_pa_bf16_950.py#L309)：页号t+2在S1 s_load；K(t+2)在S4/S6 DMA进`[cur]`
  （STAGGER组把第一个D64 chunk放S4、其余放S6）；V(t)在S0（STAGGER）或S2 DMA，下一phase读取。m前移阈值为8。
- causal、无窗口且任务数≥512时启用MERGE：一个CTA先正向做第qb块，再**反向**扫描镜像块`QB−1−qb`，
  把最短和最长的因果query块配对以平衡负载并复用L2（[_attention_sequence](mha_pa_bf16_950.py#L664)）。
- B1、任务数≥1024且窗口较窄（D192≤128、D128≤64）时启用PRUNE：只为可见的tile做wave级QK/PV。
- SWA：`window_left≥0`要求bottom-right causal；sink是每个query头一个值为0的虚拟key，只加到`l`。
- `persistent=None/False`为单次普通网格；`persistent=True`使用每(device, stream, grid)缓存的8字节调度头
  （原子ticket + 完成计数，最后一个CTA复位），graph重放需在各自捕获stream上预热。
- O写回`_store_o_r2g`：FlyDSL的`rocdl.cvt_pk_bf16_f32`是inline asm，LLVM不知道它刚写了VGPR，`permlane32_swap`读它前
  只隔1个等待状态（规则要求2个），两者之间显式`s_nop 1`（第13节）。
- 本机为gfx942：gfx950 kernel在本次重构中只做了交叉编译（`COMPILE_ONLY=1 ARCH=gfx950`）的ISA对比，未在gfx950上执行。

## 8. 单wave SWA（[mha_pa_swa_bf16.py](mha_pa_swa_bf16.py)）

- 每个CTA 1个wave（64线程），负责16行（QT16）或32行（QT32，两个16行MFMA半区）query，只访问这些行可见的KV tile并集。
- K/V从page64 SHUFFLE-5D缓存直接读到寄存器（g2r）：无gather、无LDS、无跨wave barrier、无persistent队列。
- 默认`window_left≤16`用QT16/BN16，否则QT32/BN32；可显式`query_tile∈{16,32}`、`block_n∈{16,32,64}`。
- QT16：K/V预取距离一个tile（循环前读第一个tile，计算tile后立即发起下一tile的g2r）。
  QT32：每tile先读K并等待，V的读与QK和两个半区的online softmax重叠，再等V做PV。
- gfx950使用K32 MFMA；gfx942把每个K32拆成两个原生K16。

## 9. 共用helper（[_common.py](_common.py)）

1. 标量/向量工具与同步：`_MAX_FLOOR`（最小有限FP32，作行最大值下限，不截断任何有限logit）、
   `_softmax_scale`（host校验：张量/bool抛`TypeError`，`softmax_scale·log2e`须是正的有限FP32）、
   `_uniform`（readfirstlane）、`_pin`/`_pin_i32`（空asm绑定值，阻止重算与跨阶段移动）、
   `_stage_end`（`sched_barrier`+`s_barrier`+`sched_barrier`，不是内存等待）、`_align8`（`.p2align 3`，发射口受限的
   8字节指令密集段不跨32字节取指块，见3.3）、`_wait`（显式`s_waitcnt`编码）、
   `_schedule`/`_schedule_exps`（`sched_group_barrier`交错MFMA与VALU/TRANS；后者每步先减m再exp）、`_read_page_g2r`/`_page_ready`
   （s_load页号；等待与第一次读取放在同一段asm，避免RA在等待前复制尚未到达的SGPR）、`_buffer`/`_buffer_words`、
   `_read_address`（asm `ds_read_b128`）与`_load_address`/`_load_lds`（同一读取的volatile `generic_load`，LLVM据此补MFMA
   hazard等待；`_load_lds`按LDS分配取偏移并带其alias scope；D256使用，见第13节）、`_rescale`、
   `_pack_bf16`（P与O转BF16：`bits + 0x7FFF + lsb`的软件RNE，
   每对一条`v_perm_b32`拼高半；不规范化NaN，gfx950用硬件`v_cvt_pk_bf16_f32`）。
2. BM256 32×32 softmax：`_row_max_local`/`_row_max`、`_row_sum_local`/`_row_sum`、`_center_part`（FMA `S·scale − m`）、
   `_reround`/`_reround_if`（把S'舍入到ulp(m)的整数倍，|m| ≥ 2^22后的精确规则）、`_exp_part`（`exp2(S')`）、
   `_maximum`（asm `v_max_f32`，对LLVM不透明）、
   `_mfma_hazard_wait`（MFMA结果被inline asm读取前`s_nop 10`：gfx942 32×32 MFMA需要NumPasses(8)+3个等待周期，
   LLVM hazard recognizer不保护inline asm消费者）。
3. D256几何与16×16 MFMA片段：`_mma`、`_read_q_g2r`、`_qk`、`_pv`、`_sum`、`_max`（用`llvm.maxnum`让MFMA→max依赖对LLVM可见；
   曾经的不透明max树在grid重复逐位检查中失败）、`_maxnum`（两数的`maxnumf` nnan，即裸`v_max_f32`）、
   `_cross`（`ds_bpermute`交换）、`_scale`、`_exps`（`exp2(S' − m)`）、`_mask`、`_v_tail`。

**FlyDSL磁盘缓存注意**：缓存key只扫描函数自身`co_names`引用的helper（及闭包标量），不扫描嵌套闭包里引用的helper。
因此每个含闭包的主体开头都有`_helpers = (...)`元组，列出闭包用到的helper，使修改它们会改变缓存key。
新增闭包helper时同步更新该元组；排查“改了代码但结果不变”时先清空缓存或设`FLYDSL_RUNTIME_ENABLE_CACHE=0`。

## 10. 本次迁移与重构（2026-10-22）

- 由`experiments/attention/flydsl/mha`迁入本目录（`git mv`保留历史）；实验目录的桥接文件、README和
  `test_mha_pa.py`已删除，测试拆为[tests/ops/mha](../../../../../tests/ops/mha)单元测试与
  [benchmarks/mha](../../../../../benchmarks/mha)集成测试；原amd-smi选卡/空闲门禁按QSA的协议移除。
- 统一命名（部分）：

| 旧名 | 新名 |
|---|---|
| `_common._prefetch_page` / `_q_fragment` | `_read_page_g2r` / `_read_q_g2r` |
| `_common._read_k` / `_read_v` / `_pages`（仅paged D256用） | 移入paged D256：`_read_k_s2r` / `_read_v_s2r` / `_read_tile_pages_g2r` |
| linear `_source_rows` / `_dma` / `_read_k` / `_read_v` / `_output` | `_read_kv_rows_g2r` / `_read_packet_g2s` / `_read_k_s2r` / `_read_v_s2r` / `_shuffle_o_r2s2r`+`_store_o_r2g` |
| paged D256 `_dma_k` / `_dma_v` / `_read_k_vdma` / `_read_v_dma` | `_read_k_packet_g2s` / `_read_v_packet_g2s` / `_read_k_s2r_v_g2s` / `_read_v_s2r_k_g2s` |
| FP8 `_cooperative_load` / `_cooperative_store` / `_k_fragment` / `_v_fragment` | `_read_kv_g2r` / `_store_kv_r2s` / `_read_k_s2r` / `_read_v_s2r` |
| BF16 `_load_k` / `_load_v` / `_body` | `_read_k_g2r` / `_read_v_g2r` / `_body_v128` + `_body_wide` |
| gfx950 `_dma` / `_read_qk` / `_read_v` | `_dma_g2s` / `_read_qk_s2r` / `_read_v_s2r` |
| SWA `_page` / `_q_load` / `_k_load` / `_v_load` | `_read_page_g2r` / `_read_q_g2r` / `_read_k_g2r` / `_read_v_g2r` |

- 各8-wave主体改为`prologue/phase/epilogue`结构，docstring给出逐stage表，每个stage前注释标出memory/compute、
  对应tile编号与槽位；重复的helper收敛到`_common.py`，`_dsl.py`只保留仍在使用的4个函数。
- **机器码一致性**：用frozen原始源码与新源码在同一环境、关闭磁盘缓存分别编译，逐个比较code object的反汇编。
  FP8 16/16、D256 linear 42/42、性能shape 48/48、SWA 28/28、QSA 32/32完全一致；BF16 117/120、gfx950 78/80一致，
  其余5个（均为LSE或d128 causal persistent的掩码/地址整数运算）只是整数加减的结合顺序不同，指令数与资源相同。
- 功能：80个单元测试通过；QSA 81项通过（另2项为环境相关的trap stderr文本检查，迁移前后相同）。

## 11. 优化历史与未采纳方案

以下数字来自当时的测量（MI308X，10 buffer×5轮的中位数，低负载启动、非独占），只说明方向；当前性能见使用文档。
完整原始记录在git历史中的`experiments/attention/flydsl/mha/README*.md`（如`git show edde17b:experiments/attention/flydsl/mha/README.308.md`）。

- **2026-09-08/09 BF16重写与阶段分离**：BF16以FP8为蓝本重写为8-wave/8-stage、BM256/BN64，普通网格与persistent共用主体；
  随后把VMEM/DS集中到S0/2/4/6、MFMA/VALU集中到S1/3/5/7，exp按24+8分到两次QK旁，cross32归约移入memory stage，
  V在S0读/S2发布、下一块K在S4读/S6发布。12个persistent shape延迟下降1.9%–12.2%（如full D192 1254.9→1105.7µs），
  D128/D192 MHA long（persistent与grid）为264.1–267.2T。
- **D192对齐**：三个MHA形状各加D192/V128配对，D192吞吐与D128持平或略高（Dq192使QK+PV算量增加25%）。
- **2026-09-23 D256 paged**：新增BM128/BN64 M16内核（Full约190T）。DMA探索中，K early DMA+与S4读交织、V DMA、
  页表提前/ready合并、S4 LGKM后移逐步叠加，组合为默认的v73（相对v72同buffer配对约+1.6%）。
  未采纳：M32 P-exchange（LDS交换P与行统计，133T，远低于同轮190T，已删除）；CK S-shuffle实例（29–36T，仅对照）；
  v98（S0领先组晚等待、PV分批、K部分WAR退休、S7重叠；探索约221T、配对改善约3.9%，但频率波动下全样本中位数未优于v73，未设为默认）。
- **2026-09-24 D256 linear**：K/V直接DMA到LDS、V在寄存器转置，支持page1/page4与任意页表，热调用无KV转换。
- **2026-09-25 linear Full达到220T**：非分页地址leaf合并、B1静态长度、最后N32 tile/drain裁剪、仅静态分支的可见概率pack。
  B1/Q10240/KV2583/H24/HK2 noncausal persistent：3000.1→2932.2µs（216.7→221.7T），VGPR235→228、SGPR62→52。
- **2026-09-26 B1 causal配平**：最长query块优先+蛇形CTA分配。H12/HK1：Q=KV2048 292.5→161.6µs，8192 2475.3→1838.5µs。
- **inline asm hazard修复**：`_max`改为`llvm.maxnum`、`_mfma_hazard_wait`在inline asm读取MFMA结果前补足间隔；
  `_page_ready`把`s_waitcnt`与首次读取放进同一段asm。
- **2026-10-22 FP8 VALU重排**（见第6节）：ATT显示原S1（24个exp）与S3（8个exp＋行和＋FP8转换）受发射口限制，
  memory stage空等，SIMD MFMA busy 65.1%；重排后75.8%，每tile周期3044→约2590，输出逐位不变。
  GPU7同址：16384²×H7 D128 360.6→414.9T，D192 429→443T，causal D128/D192 +14%；FP8 benchmark全部case +3.3%–14.3%。
  长序列持续负载（[164800,7,128]、50次背靠背）382.7→405.9T，受功耗限制（约630 W，时钟1.78→1.71 GHz），
  同条件hipBLASLt FP8 GEMM 16384³为399.2T。未采纳：exp放进memory stage（−8%，发射口争用且QK依赖链变长）、
  `s_setprio`提高compute wave优先级（更慢）、`v_pk_fma_f32`中心化（−5%）、D192把FP8转换移入S3（−15%）。
- **2026-10-27 BF16 D128 S3配平与取指对齐**（对照gfx950-gluon-tutorials的attention，见3.3）：采纳S3行和后移
  （单独D128 −4.3%）与D128主循环stage 8字节对齐（再−1.2%，并消除kernel入口代码长度带来的约±1.6%布局抖动）。
  未采纳：memory stage开头加`s_nop`节拍（S0/S2/S4/S6各1–16个等待周期，都在±0.2%内）；per-tensor时把scale放SGPR
  作中心化FMA的操作数（未对齐时随布局−1.5%到+1.7%，对齐后在±0.3%内，差别来自取指而非SGPR）；每条max3/FMA/
  `v_perm`前也`.p2align 3`（跨块从每副本约28条降到0–2条，但比只在stage开头对齐慢0.2–0.8%）；D192也对齐（+1.5%）；
  配平后S7的中心化FMA段错位（每phase约11条跨块），在第一条FMA前对齐反而慢0到0.5%：BF16的S7有16个MFMA，
  受MFMA限制，取指气泡被掩盖。
- **2026-10-27 FP8**（同一组思路，见第6节）：采纳S7取指对齐、行和2个加法从S5挪到S7、per-tensor scale放SGPR。
  合计（GPU7交替A/B，相对改动前）：D128 −1.85%、per-tensor D128 −1.65%、causal −1.0%、D192 −1.2%、LSE −1.9%、
  [164800, 7, 128] −0.7%；高精度模式−0.45%到−1.1%。未采纳：S2开头加`s_nop 3`（默认模式−0.8%到+0.2%，
  高精度模式−0.3%到+0.9%）；行和挪进S6或16个全挪（+1.2%、+1.4%）；S5也对齐（多数慢0.2–0.7%）。FP8的softmax
  VALU已按发射口均摊（10-22），四个compute stage都受发射口或MFMA限制，配平只剩S5与S7之间的少量余地。

## 12. 与aiter PR #6576 union评审意见对齐（2026-10-22）

逐条核对[aiter PR #6576](https://github.com/ROCm/aiter/pull/6576)（QSA union kernel）的评审意见，本目录中适用的修复如下。
QSA只随共用helper改动（第2、3行）；其余同类问题（行最大值下限、`softmax_scale`校验、K/V说明、测试）留待独立PR。

| 评审意见 | 本仓库的处理 |
|---|---|
| 行最大值下限−1e30截断更小的有限logit | 全部MHA kernel改用`_MAX_FLOOR`（最小有限FP32）；SWA只需这一项 |
| FMA `S·scale − m`与舍入后的m差至多ulp(m)/2，logit近1e11时exp2上溢/下溢 | D256 paged/linear与aiter相同：先舍入`S·scale`再减m（`_scale`、`_exps`、`_schedule_exps`、`_pv`），复用这些helper的QSA union随之改变；gfx942 BF16 BM256、FP8与gfx950保留FMA，|m| ≥ 2^22后用`_reround`（第3.3、6节） |
| `passthrough`形式的target-features在lowering中丢失 | BF16 BM256、D256 paged/linear与QSA union改用`llvm.target_features`，FP8补上同一属性。QSA union若只换helper，新增的标量乘、减会被合成`v_pk_*`，kernel慢约8.5% |
| host的`softmax_scale`校验 | `_softmax_scale`：张量/bool抛`TypeError`；`softmax_scale·log2e`须是正的有限FP32，否则`ValueError` |
| 非有限K/V经共享的V污染被掩码的行 | 各入口docstring写明KV长度内的K/V须有限 |
| 测试参考把NaN一并置0 | `_mha.torch_reference`只把没有可见key的行置0，其余NaN照常传播 |
| 测试跨stream同步（`wait_stream`） | 已有；weakref arena不适用（MHA无arena） |

另修复gfx950 persistent的mailbox竞争：没有barrier的任务（如空KV）可能让tid 0在其他wave读到上一个ticket前改写它，
`_next_ticket`先做一次`_stage_end()`。

新增测试：paged/linear的`test_extreme_logits`（logit近1e11与低于−1e30，O应为1）、`test_logits_cross_exact_threshold`
（前两个tile |m| < 2^22，第三个tile把m抬到约2^28（FP8）或2^36（BF16），覆盖规则中途生效）、`softmax_scale`的
TypeError/ValueError用例。改动前的代码在除SWA近1e11外的全部用例上失败（输出0、inf或NaN）。

性能（MI308X，基线为改动前副本，交替A/B中位数，新/旧耗时；BF16 BM256、FP8、SWA输出与改动前逐位一致）：

| kernel | 结果 |
|---|---|
| BF16 BM256（GPU6） | D128 −1.1%、D128 causal +0.2%、D192 +0.5%、D128带LSE +0.8%、V192 +0.5% |
| FP8（GPU7） | D128 −0.1%、D128 causal −2.1%、D192 +0.3% |
| D256 paged / linear full / linear causal（GPU7） | −0.1% / +0.3% / +0.3% |
| SWA（GPU7） | −0.1% |
| QSA M12000 TP2/4/8（GPU6，顺序运行） | union kernel −1.8%/−2.0%/−2.1%，整体qsa −0.6%/−1.1%/−2.0% |

gfx950只做了交叉编译（80/80）：热路径的`v_fma_f32`/`v_mul_f32`/`v_exp_f32`条数与VGPR和改动前完全相同，
新增的`v_add_f32`/`v_sub_f32`都在`_reround`冷分支里，未在gfx950上运行和计时。

## 13. inline asm的hazard审计与修复（2026-10-27）

LLVM的hazard recognizer不分析inline asm里的指令：MFMA之后由asm读写相关VGPR、asm写VGPR之后紧跟permlane，
都可能少插等待状态。aiter PR #6576的评审也要求union kernel去掉这类asm。

- **审计**：离线编译72个specialization（gfx942默认36个、高精度模式15个、P转BF16 RTA 9个，gfx950 12个；不初始化GPU），
  按LLVM自身的GFX940/GFX950等待规则检查反汇编的每条控制流边；D256另用LDS仿真器跑38个动态场景，检查DMA与DS读写的先后和M0。
  脚本未入库：`mytest/mydata/mha_hazards_20261027/capture.py`与`mytest/mydata/mha_hazards_recheck_20261027/`。
- **发现**：63处等待不足，全部紧挨inline asm。
  - D256 gfx942 42处（linear 24、paged 18）：asm `v_max_f32`（`_maximum`）或asm `ds_read_b128`（`_read_address`）在4-pass
    `v_mfma_f32_16x16x16_bf16`之后过早改写该MFMA仍要读的C（26处）或仍要写的结果（16处）寄存器。
  - gfx950 21处：`_store_o_r2g`中FlyDSL的`rocdl.cvt_pk_bf16_f32`是inline asm，`permlane32_swap`读它的结果前只隔1个等待状态（要求2个）。
  - D256这些点用匹配的探针和现有测试都没有在硬件上复现，gfx950只交叉编译；但同类规则在gfx942上可以触发：
    FP8 32×32×16 MFMA之后立即用`v_mov`改写其C的最后一个VGPR，结果出错，补足7个等待状态后正常。
- **修复**（只改D256与gfx950）：
  - D256的LDS读改为LLVM可见的volatile `generic_load`（linear用`_load_address`，paged用`_load_lds`），仍是同样的
    `ds_read_b128`、同样的发射顺序，LLVM据此补足MFMA之后的等待；跨lane行最大值用`_maxnum`（`maxnumf` nnan，即裸`v_max_f32`）。
  - paged参照PR #6576的union把K/V拆成两个LDS分配。只有一个数组、指针来自`inttoptr`时，LLVM分不清K读与V DMA，
    在每个`ds_read`前插`vmcnt(0)`（62处），paged慢5.4倍；拆开后各有alias scope。V放在地址0，asm算出的V DMA目的地址直接作M0
    （V在0x8000时每个packet多一条`s_add_i32 m0`，共64条）。
  - paged step0：`_page_ready`的asm等待已经退休了K(t)的读，但LLVM看不到，会在之后的MFMA前逐个补`lgkmcnt`（每个循环副本15个），
    所以在它之前显式`_wait(lgkmcnt=0)`。
  - gfx950：`cvt_pk_bf16_f32`与`permlane32_swap`之间`sched_barrier`、`s_nop 1`、`sched_barrier`。
  - BF16 BM256与FP8保留asm `_maximum`：它们在asm读MFMA结果前已有`_mfma_hazard_wait`，没有违例；全局换成`maxnumf`后
    FP8 D128与BF16 D192慢1.1–1.2%（代码布局变化）。
- **正确性**：72个specialization全部满足规则；除D256的9个和gfx950的12个外，二进制与改动前逐字节相同，
  gfx950只在O写回多了`s_nop`。LDS仿真38个场景（228万条动态指令、31.3万次LDS事件）无竞争、无未写先读、无未知地址。
  复用linear读取helper的QSA union同样没有违例，等待数与VGPR不变。`tests/ops/mha`默认与高精度模式均117 passed、7 skipped；
  `tests/ops/qsa` 81 passed，另2项trap stderr文本检查在改动前同样失败。另有424个配置（BF16 BM256 96个×默认/高精度/RTA，
  FP8 32个×默认/高精度，D256 paged 24个、linear 48个；覆盖D128/D192/V192、page 1/4/32/64/128、causal、LSE、persistent/grid、
  per-token/per-tensor、GQA、变长、decode、空KV与Q>KV）与FP32参考的误差都在阈值内，3次重复逐位一致；
  D256的72个配置各重复30次（同一GPU上6个进程并发）也逐位一致。
- **精度**：上述424个配置和性能测试的68个大shape（O与LSE）都与父提交0cff467逐位一致；
  集成测试全部64个结果的逐buffer精度也与父提交相同。第3.5、3.6、6节的误差分析不变。
- **代码与资源**：paged每个specialization少37–40条指令，`s_waitcnt`只多显式的2个，LDS读前的`vmcnt`等待数不变，VGPR不变或少1–2个；
  linear的causal/page1/page4/ragged VGPR多3–7个，static/narrow不变或少2个，均无spill。
- **性能**：两个进程在同一GPU上交替计时（[使用文档第3节](../../../../../benchmarks/mha/readme.md)）：D256 paged 17个case
  在−0.6%到+0.6%之间（中位+0.2%），linear 13个case在−1.3%到+0.7%之间（中位0.0%），机器码未变的对照组在±0.1%以内。
  gfx950未计时。
- **剩余inline asm**：D256仍有linear中DMA与`ds_read_b128`同leaf的`_read_packet_g2s`、`_cross`（`ds_bpermute`）、
  O shuffle的`ds_write`、`_page_ready`的等待等。当前没有违例，但结果取决于寄存器分配；改动这些kernel后应重跑审计。
