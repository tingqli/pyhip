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
| `_read_k_s2r_v_g2s` | 一段asm里交织两件事：读K（s2r），同时DMA V（g2s） | D256 paged |

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
- softmax统一用log2单位：`S' = S·scale·log2e − m`，`P = exp2(S')`，`l = ΣP`，`O = ΣP·V`，输出`O / l`；
  m只在新行最大值超过m一个阈值时才前移（lazy rescale），从而大部分tile不需要缩放O和l。

### 0.3 主体与预取距离一览

| kernel | 主体 | 页号 | K | V | 计算 |
|---|---|---|---|---|---|
| BF16 D128/D192 V128无LSE | [_body_v128](mha_pa_bf16_942.py#L343) | t+2（S0 s_load） | t+1：S4 g2r，S6 r2s（单槽） | t：S0 g2r（64-bit GLOBAL），S2 r2s到`[cur]` | QK(t)，exp/PV(t−1) |
| BF16 LSE或V192 | [_body_wide](mha_pa_bf16_942.py#L623) | t+2 | t+1：S0 g2r，S4/S6 r2s | t：S0 g2r；环形S2发布，单槽下一phase S0发布 | 同上 |
| FP8 | [_body](mha_pa_fp8_942.py#L219) | t+3 | t+2：S0 g2r，S4 r2s（双槽） | t：S0 g2r，S2 r2s（双槽） | 同上 |
| BF16 D256 paged | [_body](mha_pa_bf16_256_paged_942.py#L182) | t+2（step0） | t+1：step2 DMA，step3落地 | t−1：step0 DMA，step1落地 | 同上 |
| BF16 D256 linear | [_body](mha_pa_bf16_256_linear_942.py#L281) | 行地址t+2（step0 g2r） | t+1：step2 DMA | t−1：step0 DMA | 同上 |
| gfx950 BF16 | [_attention_body](mha_pa_bf16_950.py#L299) | t+2（S1） | t+2：S4/S6 DMA（双槽） | t：S0或S2 DMA（双槽） | 同上 |
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
- `waves_per_eu=2`、关闭packed-fp32（`-packed-fp32-ops`）是固定编译属性。

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
- m从首个行最大值+1开始，只有后续行最大值超过m+7才前移到新最大值+1，并按ballot只在有行需要时缩放O和l（`_rescale`）。
- D128每次循环展开两个phase，使`cur/prev`成为常量；D192每次一个phase，槽号由t计算。
- O按`O/l·v_descale`转BF16（软件RNE，`_pack_bf16`），由`_store_o_tile_r2g`写回。

`_body_wide`（LSE或V192）用协作拷贝（cooperative g2r→r2s），行和/行最大值在compute stage用lane shuffle归约，
其余阶段角色相同；S6读V的64–Dv维，S7完成其余PV。

### 3.4 资源（MI308X，FlyDSL 0.3.4.1，ISA实测）

| specialization | VGPR | SGPR | LDS | scratch/spill |
|---|---:|---:|---:|---:|
| full D128 persistent / grid | 228 / 226 | 72 / 54 | 49,920 | 0 |
| full D192 persistent / grid | 254 / 256 | 75 / 56 | 58,240 | 0 |
| causal D128 persistent / grid | 232 / 228 | 74 / 58 | 49,920 | 0 |
| causal D192 persistent / grid | 256 / 254 | 77 / 60 | 58,240 | 0 |

## 4. gfx942 BF16 D256 paged（[mha_pa_bf16_256_paged_942.py](mha_pa_bf16_256_paged_942.py)）

`PagedAttention(Dq=Dv=256)`自动分派到这里（`bf16_backend`以`-pages1-lateS41`结尾）。

- BM128/BN64，8 wave×16行，`v_mfma_f32_16x16x16_bf16`；Q一次读入寄存器（`_read_q_g2r`）并在全部tile复用。
- LDS只有一个K tile（32 KiB，每wave的K行间距`K_PITCH=1024`）和一个V tile（32 KiB）。
- K/V都用direct-to-LDS DMA：`buffer_load_dword ... lds`每lane一dword、每个packet写2 KiB，一个tile 16个packet。
  V的每个tile按其所在页的完整64-bit地址重建buffer描述符（`_v_tile_resources`；page32时一个tile跨两页，用两个描述符），
  buffer指令只携带tile内相对偏移，页号×页跨度不在32-bit里计算。
- “交织”：`_read_k_s2r_v_g2s`在同一段asm里把V(t−1)的每个DMA packet与K(t)的一次`ds_read_b128`配对；
  `_read_v_s2r_k_g2s`把K(t+1)的DMA与V(t−1)的读配对。DS读为VMEM提供M0→VMEM的必要间隔，省去nop。

```text
step  memory stage                                compute stage
0     s_load pages(t+2); g2s V(t-1) with s2r      QK: S(t) keys 0-31
      K(t) keys 0-31 between its packets          softmax: P(t-1) = exp2(S'(t-1))
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

公开入口为[flash_attn_varlen_func](mha_pa_bf16_942.py#L1164)：连续THD输入、无KV格式转换，热调用只有一个attention kernel。

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

### 5.2 流水线（`_body`）

```text
step  memory stage                                compute stage
0     g2r rows(t+2); g2s V(t-1) with s2r K(t)     QK: S(t) keys 0-31
      keys 0-31 between its packets               softmax: P(t-1) = exp2(S'(t-1))
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
  `_pv(center_args=...)`在第一步算出新m，随后四步逐段完成`S' = S·scale − m`，最后按ballot lazy rescale。
- 主循环：noncausal且无LSE时每次展开4个phase（摊销SOFFSET队列移动）；causal或LSE每次2个phase
  （4个会让标量寄存器溢出）。
- 静态B1 noncausal且最后一tile不超过32个key时，最后一tile走NARROW phase（无keys 32–63），
  epilogue只drain第一个32-token V平面。B1非分页、noncausal、无LSE时Q/K长度作为编译期常量（`STATIC_Q/STATIC_K`）。
- persistent且B1非分页causal时，任务按“最长query块优先、蛇形分配到CTA”排序（[_linear_256_kernel](mha_pa_bf16_256_linear_942.py#L576)），
  平衡因果负载而不改变tile内算术。

### 5.3 API行为

- host对`cu_seqlens_q/k`与`block_table`做一次完整校验（`.cpu()`），结果按tensor身份+版本号缓存（`_validate_metadata`）；
  原地修改metadata会使缓存失效并重新校验。graph capture中遇到未预热/已修改的metadata直接报错。
- 不支持的参数（dropout、window/sink、alibi、bias、soft cap、page_size∉{1,4}、num_waves≠8、autograd）抛`NotImplementedError`；
  输出与Q/K/V/metadata/LSE重叠、`softmax_scale`非正有限值、长度超过上界等抛`ValueError`。
- 资源：full persistent VGPR228/SGPR52；causal B1 VGPR232/SGPR63；page1 VGPR242/SGPR106；page4 VGPR247/SGPR68；
  均LDS 65,536、零scratch/spill（单元测试在模块结束时检查本次编译的全部specialization）。

## 6. gfx942 FP8（[mha_pa_fp8_942.py](mha_pa_fp8_942.py)）

BF16 BM256版本的蓝本：同样的8 wave、8 stage、STAGGER与“QK(t)+PV(t−1)”结构，QK/PV为32×32×16 FP8 MFMA。

```text
stage  phase t; cur = t & 1, prev = (t - 1) & 1
S0     s2r K(t) keys 0-31 [cur]; s_load page(t+3); g2r V(t), K(t+2)
S1     QK: S(t) keys 0-31          softmax: P(t-1) = exp2(S'(t-1)), values 0-23
S2     s2r K(t) keys 32-63 [cur]; V(t) lands; r2s V(t) -> [cur]
S3     QK: S(t) keys 32-63         softmax: P(t-1) values 24-31, l += row sum, P to FP8
S4     s2r V(t-1) dims 0-63 [prev]; K(t+2) lands; r2s K(t+2) -> [cur]
S5     PV: O dims 0-63 += P V      softmax: mask S(t), new m, S'(t) values 0..split
S6     s2r V(t-1) dims 64-127 [prev]
S7     PV: O dims 64-127 += P V    softmax: S'(t) values split..31; rescale O and l
```

- K与V各两个LDS槽，K比QK领先两个tile、页号领先三个tile；全部为寄存器中转的协作拷贝（g2r→r2s），
  不用direct-to-LDS DMA、LDS转置指令或原生FP32→BF16转换。LDS：D128 33,280、D192 41,600字节（16维×64 key的1040字节chunk）。
- P量化为FP8参与PV；m的前移阈值为6，保证FP8-FNUZ下P有限。输出`O/l·v_descale`。
- 只保留LDS模式（旧register模式已删除）；不支持persistent、SWA和sink。

## 7. gfx950 BF16（[mha_pa_bf16_950.py](mha_pa_bf16_950.py)）

由OPUS派生，直接读取page64 SHUFFLE-5D缓存，无gather、无线性KV workspace、无辅助dispatch。

- BM256，8 wave，K/V各两个槽并全部用direct-to-LDS DMA（`_dma_g2s`）；Q先DMA到V区域再读入寄存器，
  因此V区按Q的大小分配：D128 99,840、D192 149,760字节。
- 阶段表见[_attention_body](mha_pa_bf16_950.py#L299)：页号t+2在S1 s_load；K(t+2)在S4/S6 DMA进`[cur]`
  （STAGGER组把第一个D64 chunk放S4、其余放S6）；V(t)在S0（STAGGER）或S2 DMA，下一phase读取。m前移阈值为8。
- causal、无窗口且任务数≥512时启用MERGE：一个CTA先正向做第qb块，再**反向**扫描镜像块`QB−1−qb`，
  把最短和最长的因果query块配对以平衡负载并复用L2（[_attention_sequence](mha_pa_bf16_950.py#L643)）。
- B1、任务数≥1024且窗口较窄（D192≤128、D128≤64）时启用PRUNE：只为可见的tile做wave级QK/PV。
- SWA：`window_left≥0`要求bottom-right causal；sink是每个query头一个值为0的虚拟key，只加到`l`。
- `persistent=None/False`为单次普通网格；`persistent=True`使用每(device, stream, grid)缓存的8字节调度头
  （原子ticket + 完成计数，最后一个CTA复位），graph重放需在各自捕获stream上预热。
- 本机为gfx942：gfx950 kernel在本次重构中只做了交叉编译（`COMPILE_ONLY=1 ARCH=gfx950`）的ISA对比，未在gfx950上执行。

## 8. 单wave SWA（[mha_pa_swa_bf16.py](mha_pa_swa_bf16.py)）

- 每个CTA 1个wave（64线程），负责16行（QT16）或32行（QT32，两个16行MFMA半区）query，只访问这些行可见的KV tile并集。
- K/V从page64 SHUFFLE-5D缓存直接读到寄存器（g2r）：无gather、无LDS、无跨wave barrier、无persistent队列。
- 默认`window_left≤16`用QT16/BN16，否则QT32/BN32；可显式`query_tile∈{16,32}`、`block_n∈{16,32,64}`。
- QT16：K/V预取距离一个tile（循环前读第一个tile，计算tile后立即发起下一tile的g2r）。
  QT32：每tile先读K并等待，V的读与QK和两个半区的online softmax重叠，再等V做PV。
- gfx950使用K32 MFMA；gfx942把每个K32拆成两个原生K16。

## 9. 共用helper（[_common.py](_common.py)）

1. 标量/向量工具与同步：`_uniform`（readfirstlane）、`_pin`/`_pin_i32`（空asm绑定值，阻止重算与跨阶段移动）、
   `_stage_end`（`sched_barrier`+`s_barrier`+`sched_barrier`，不是内存等待）、`_wait`（显式`s_waitcnt`编码）、
   `_schedule`（`sched_group_barrier`交错MFMA与VALU/TRANS）、`_read_page_g2r`/`_page_ready`
   （s_load页号；等待与第一次读取放在同一段asm，避免RA在等待前复制尚未到达的SGPR）、`_buffer`/`_buffer_words`、
   `_read_address`（`ds_read_b128`）、`_rescale`。
2. BM256 32×32 softmax：`_row_max_local`/`_row_max`、`_row_sum_local`/`_row_sum`、`_center_part`、`_exp_part`、
   `_mfma_hazard_wait`（MFMA结果被inline asm读取前`s_nop 10`：gfx942 32×32 MFMA需要NumPasses(8)+3个等待周期，
   LLVM hazard recognizer不保护inline asm消费者）。
3. D256几何与16×16 MFMA片段：`_mma`、`_read_q_g2r`、`_qk`、`_pv`、`_sum`、`_max`（用`llvm.maxnum`让MFMA→max依赖对LLVM可见；
   曾经的不透明max树在grid重复逐位检查中失败）、`_cross`（`ds_bpermute`交换）、`_center`、`_exps`、`_pack`、`_mask`、`_v_tail`。

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
