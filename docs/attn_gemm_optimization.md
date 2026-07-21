# 融合 Attention 双 GEMM 优化记录(FlyDSL, bf16, gfx942/MI3xx)

对应脚本:[`tests/flydsl/test_attn_gemm.py`](../tests/flydsl/test_attn_gemm.py)

## 1. 问题定义

无 softmax 的两段链式 GEMM(数学上等价于一次 attention 前向):

```
gemm1: S = Q[M,D] @ K[N,D]^T   -> [M, N]      # pv
gemm2: O = S[M,N] @ V[N,D]     -> [M, D]      # v
```

- 规模:`M = N = 8192`,`D = 128`,精度 bf16。
- 融合实现:一个 kernel 内沿 KV 维度循环,逐 tile 先算 `S`、再累加 `O`,中间结果 `S`(即 `pv`,完整为 `[8192,8192]`)**不落全局显存**。
- 参考:torch 两次 bf16 matmul(内部 rocBLAS)。

FLOPs = `2·M·N·D`(gemm1) + `2·M·N·D`(gemm2) = `4·M·N·D` ≈ 34.4 GFLOP。

## 2. 性能总览(MI308X / gfx942, HIP_VISIBLE_DEVICES=2)

| 版本 | 方法要点 | 延迟 | TFLOPS | VGPR | rel_l2 |
|---|---|---:|---:|---:|---:|
| torch 参考 | 两次独立 bf16 matmul(rocBLAS) | 312 µs | 110.0 | — | — |
| **v1 baseline** | 融合;`S` 经 LDS 中转;`V^T` 非合并全局加载 | 853 µs | 40.3 | — | 1.3e-4 |
| **v2 register-resident S**(BN=128) | `K@Q^T` register trick;4-wave M-split;K/V 协作载入 LDS | 915 µs | 37.6 | 326 | 1.3e-4 |
| **v3 低 VGPR**(BN=64) | v2 + `sched_barrier` 错峰 + 协作 frag 复用 + BN=64 | 882 µs | 38.9 | **202** | 1.3e-4 |
| **v4 k_perm(K 128-bit)** | v3 + `tmma1` 加 `k_perm=(4,4,2):(1,8,4)` → K 的 ds_read 从 128×16bit 变 16×128bit | 476 µs | 72.1 | 196 | 1.3e-4 |
| path-b V 128-bit(S 经 LDS,已被 v5 取代) | v4 去 register trick,`S^T` 经 LDS 中转 → GEMM2 用 `k_perm`+paged V | 500 µs | 68.8 | 196 | 1.3e-4 |
| **v5 register trick + V 全 128-bit** | v4 + `tmma1` 的 **M 维(Nk)也加 perm_M=k_perm** → C 累加器每 lane 8 连续 Nk,`select` register trick 仍成立 → GEMM2 用 `k_perm`+paged V（直读 global）,**保留 S 在寄存器**且 K/Q/A/V 读**全 128-bit** | **374 µs** | **91.7** | 200 | 1.3e-4 |
| **v6 V 协作→LDS** | v5 + V 由「直读 global」改为「协作加载到 v_lds（保留 v-连续 paged）→ 从 LDS 读」；V 全局读 4×广播→1×协作（`buffer_load` 28→16, `ds_read` 16→32） | 382 µs | 90.0 | 200 | 1.3e-4 |
| **v7 廉价 f32→bf16（当前主文件）** | v6 + 把 register trick 与 O epilogue 的 `.to(fx.BFloat16)`（RNE+NaN）换成 `_cvt_f32_to_bf16`（add-0x8000 舍入+截断）→ 热循环每 KV-tile 省 ~96 条 `v_bfe/v_cndmask` | **354 µs** | **97.1** | 200 | 1.5e-4 |

> **v4 是关键提速(1.85x)**:`k_perm` 让每 lane 沿 K(=D)取 8 个连续 bf16(FrgV=4 + cntK=2),`ds_read_b128` 替掉大量窄读。参考 `src/contrib/flydsl/moe_gemm_splitk.py::_make_1x4_tiled_mma`。

> **v5 是本轮最快版本（相对 v4 再 1.27x,达 rocBLAS 的 83%）**:给 GEMM1 的 **M 维(Nk)也加 perm_M**,使 C 累加器 `frag_St` 每 lane 直接持 **8 连续 Nk**,物理上匹配 GEMM2 的 `k_perm` A;`fx.select([0,2,1])` register trick **仍成立**（与 k_perm A 形状全等 `[4,2,(2,2)]`,详见 §8）,于是 GEMM2 可用 `k_perm`+paged V 拿到 V 128-bit,**同时保留 S 常驻寄存器**（无 LDS 中转）。关键教训:早前以为「带 perm_M 的 select 会 poison」是**误判**——poison 实际来自 paged V 读（未用 buffer-tile 模板 `v_fake`）,select 本身可编译。

> **v7 是当前主文件**（`tests/flydsl/test_attn_gemm.py`）:MFMA 的 f32 累加器转 bf16 时,`.to(fx.BFloat16)` 在 gfx942 上展开为逐值 RNE 舍入+NaN 处理（`v_bfe`/`v_add3`/`v_cmp_u`/`v_cndmask` 每值 ~5 条）。register trick 在**热循环**里每 KV-tile 要转 32 个 S 值,O epilogue 转 64 个。改用 `_cvt_f32_to_bf16`（移植自 `moe_gemm_splitk.py`：`((f32.bitcast(u32)+0x8000)>>16).to(u16).bitcast(bf16)`,round-half-up,无 NaN 处理）,每值仅 add+shift ~2 条 → 热循环省 ~96 条 VALU/KV-tile,**90.0 → 97.1 TFLOPS**（达 rocBLAS 的 89%）。精度从 1.3e-4 微升到 1.5e-4（round-half-up vs RNE）,仍 bf16 级别。注:这一优化与 V 路径正交,若用在 v5 直读版上估计可再高几个点。

> 精度均为 bf16 级别一致(`rel_l2 ≈ 1.3e-4`;`max_abs≈16` 是 `O` 量级达 ±千级时的 bf16 舍入,`mean_abs≈0.0036`)。

> 计时:kernel 用 `flyc.compile` 预建 CallState **快速派发**(~6µs/次,而非 `JitFunction.__call__` 的 ~140µs;参考 `tests/contrib/moe/test_moe.py`),再用 `torch.cuda.Event` 计 GPU 时间。本 kernel 为 GPU-bound,host 派发开销基本被 GPU 执行掩盖,故 fly 数值不随派发方式变化;但小 batch / launch-bound 场景该开销显著,`flyc.compile` 编译缓存是必要的。

## 3. 优化步骤

### v1 — baseline 融合(S 经 LDS 中转)

- 布局:一个 block 处理 `BM=128` 行 query,4 wave 排成 `2×2`(`tiled_mma (2,2,1)`),`BN=64`。
- gemm1 `S = Q@K^T`(`fx.gemm` 语义 `C=A@B^T`)在 f32 累加器里。
- **S 的搬运**:f32 累加器 → bf16 → 写入 `S_lds[BM,BN]` →(barrier)→ 作为 gemm2 的 A 操作数从 LDS 读回。
  - 关键坑:bf16 的 **C 累加器存储必须用 16-bit copy atom**(`UniversalCopy16b`)。32b 会把每 lane 的 2 个 bf16 打包到目标里不连续的位置 → 隔列错位。
- **V 的搬运**:以转置视图 `V^T=[D,N]:(1,D)` 直接作 gemm2 的 B 操作数(`A@B^T = S@V`)。
  - 关键坑:转置视图的收缩维在内存里 stride=D,**必须用 16-bit copy atom**;32b 向量化会读连续内存的错元素。此路径非合并、较慢。
- 结果:**40.3 TFLOPS**。瓶颈是每次迭代的 16-bit LDS/global 操作、2 barrier、小 tile、无流水线。V 只有 2MB 被 L2 缓存,故非合并访问被部分隐藏。

> 期间试过「把 V 合并加载到 LDS 再转置读」:38.5 TFLOPS,**反而更慢**(额外 LDS 写 + barrier 抵消了合并收益),已回退。说明 v1 的瓶颈不在 V 全局合并。

### v2 — register-resident S(本次)

目标:**避免 gemm1 的输出 `S` 进入 LDS**,把 `S` 保持在寄存器里直接喂给 gemm2,为后续流水线优化扫清障碍。

**布局(4-wave M-split)**

- 一个 workgroup = 256 线程 = 4 wave;4 个 wave 沿 **Q 的 M 方向均分**,每 wave 负责 **32 行 query**,贯穿两个 GEMM。
- gemm1 处理 `128×128`,每 wave `32×128`;gemm2 处理 `128×128`,每 wave `32×128`,最终每 wave 输出 32 行。

**register trick(核心)**

MFMA `16×16×16` 的 **C 累加器布局** 与 **A 操作数布局** 互为转置(同一 lane 的 4 个值:C 是「固定列、沿行」,A 是「固定行、沿 K」)。因此:

- gemm1 改算 **`S^T = K @ Q^T`**(把 `K` 当 A、`Q` 当 B),其 C 累加器 `S^T[Nk,M]` 的布局,正好等于 gemm2 需要的 A 操作数 `S[M,Nk]` 的布局(逐 atom 一致)。
- 但**多 atom** 时 C 的 rep 维顺序是 `(V, Nk_rep, M_rep)`,A 需要 `(V, M_rep, Nk_rep)`,两者交换了。用 `fx.select` 交换即可对齐:

```python
tmma1 = fx.make_tiled_mma(mma, fx.make_layout((1, WAVES, 1), (1, 1, 0)))  # gemm1: wave 分 query-M(=N 维)
tmma2 = fx.make_tiled_mma(mma, fx.make_layout((WAVES, 1, 1), (1, WAVES, 0)))  # gemm2: wave 分 query-M(=M 维)

frag_St = thr1.make_fragment_C(fx.make_rmem_tensor(fx.make_layout((BN, BM), (BM, 1)), fx.Float32))   # C = S^T
frag_Sa = thr2.make_fragment_A(fx.make_rmem_tensor(fx.make_layout((BM, BN), (BN, 1)), fx.BFloat16))  # A = S

fx.gemm(mma, frag_St, frag_K, frag_Q, frag_St)                      # S^T = K @ Q^T
frag_Sa.store(fx.select(frag_St, [0, 2, 1]).load().to(fx.BFloat16))  # C -> A(重排 rep 维 + f32->bf16)
fx.gemm(mma, frag_O, frag_Sa, frag_V, frag_O)                       # O += S @ V
```

- `tmma1` 与 `tmma2` 都把 wave `w` 分配到 query 行 `[w*32, w*32+32)`,保证 register 复用时数据归属一致。
- `make_rmem_tensor` 直接作 `make_fragment_C/A` 的形状+dtype 模板(无需全局 `S` 张量)。
- 验证:单 atom 无需 select 即对;多 atom 加 select 后 `O_rel` 从 0.99 → 0.0017(见 `/tmp/dbg_regtrick2.py`)。

**K/V 搬运**

- `K`、`V` 各占一块 LDS(`[BN,D]` 各 32KB,合计 64KB);`S` 不占 LDS。
- 每次迭代:4 wave **协作合并**(128b)把 `K`、`V` 从 global 载入 LDS →(barrier)→ 从 LDS 读到 frag。
  - gemm1 的 A(`K`)在 4 个 N-wave 间广播:各 wave 从 LDS 读全部 `Nk`。
  - gemm2 的 B(`V^T`)在 4 个 M-wave 间广播:各 wave 从 LDS 转置读全部 `[D,Nk]`(16b strided)。
- `Q` 是 gemm1 的 B 操作数,按 query-M 分给各 wave 的 32 行,**循环前只载入一次**。

**结果:37.6 TFLOPS**(BN 扫描:BN=128→37.6,BN=64→36.5,BN=32→34.4,故取 128)。

## 4. 分析:为什么 v2 暂未提速

- register trick 本身**不直接提速**——它是把 `S` 移出 LDS、腾出寄存器与同步预算,**为流水线做准备**。
- 当前 v2 仍是**同步**结构(每次迭代:载入→barrier→gemm1→gemm2→barrier),载入延迟没有被计算掩盖,和 v1 一样受 barrier/占用率制约。
- v2 的额外开销:4-wave M-split 使 `K`、`V` 在 wave 间**广播读**(4×);`K`、`V` 双 LDS = 64KB → 1 wg/CU 占用率。
- 实测**降低 LDS 反而更慢**(BN=64/32),说明占用率不是主瓶颈;瓶颈是每迭代固定开销(2 barrier + 协作载入 + 广播读),小 BN 只是把它摊到更多次迭代。
- 距离 rocBLAS(110 TFLOPS)的差距来自缺少软件流水线、指令调度、更宽的 LDS 访问等生产级优化。

## 5. VGPR 优化(326 → 202,<256 = 2 waves/SIMD)

用 `FLYDSL_DUMP_IR=1` 得到 `dump/attn_kernel_0/21_final_isa.s` 里的 `.vgpr_count`。v2(BN=128)= **326 VGPR**(超 256,只 1 wave/SIMD)。

**每步寄存器分析**(per-lane,wave64;bf16=2/reg,f32=1/reg):

| fragment | 形状(per-wave) | 元素/lane | VGPR |
|---|---|---|---:|
| frag_K(gemm1 A,广播全 Nk×D) | `[Nk, D]` | Nk·D/64 bf16 | **Nk·D/128** |
| frag_V(gemm2 B,广播全 D×Nk) | `[D, Nk]` | D·Nk/64 bf16 | **Nk·D/128** |
| frag_St(gemm1 C,f32) | `[Nk, 32]` | Nk·32/64 f32 | Nk/2 |
| frag_O(gemm2 C,f32,常在 AGPR) | `[32, D]` | 32·D/64 f32 | D/2 |
| frag_Q(gemm1 B) | `[32, D]` | 32·D/64 bf16 | D/4 |
| frag_Sa(gemm2 A) | `[32, Nk]` | 32·Nk/64 bf16 | Nk/4 |
| frag_ld(协作载入) | `[BN,D]/256` | BN·D/256 bf16 | BN·D/512 |
| register-trick 临时 | `select(frag_St).load()` | Nk·32/64 f32 | Nk/2 |

**根因**:操作数 `frag_K`、`frag_V` 各 = `Nk·D/128` VGPR。`Nk=D=128` 时**各 128 VGPR**;register trick 夹在 gemm1/gemm2 之间,使两者难以完全错峰 → 光操作数就逼近 256,加上 f32 累加器 `frag_St`(64)与 trick 临时向量(64)→ 326。操作数**随 BN 线性缩放**,是 128×128 的硬下限。

**降低手段**(累计 326 → 202):

1. **`fx.rocdl.sched_barrier(0)` 错峰**(gemm1 之后、读 `frag_V` 之前;协作载入之后):`v_lds` 在 barrier 前就绪,编译器会**提前预取** `frag_V`,使其在 gemm1 期间就与 `frag_K` 同时存活(各 128)。sched_barrier 阻止提前 → 330 → 306。
2. **协作载入复用一块 `frag_ld`**(K 先写 LDS 再复用装 V)+ 及时释放 → 小幅下降。
3. **BN=64**:把 `frag_K`、`frag_V` 各减半到 64 VGPR —— 决定性的一步。峰值(gemm1):`frag_K`(64)+`frag_St`(32)+`frag_Q`(32)+`frag_O`(64,AGPR)≈ 192,加临时 → **202 VGPR,AGPR=0,无 spill**。

> `amdgpu-mfma-vgpr-form` 与 `rocdl.waves_per_eu=2` 两个 hint 实测**无效**(编译器不肯为 128×128 挤到 ≤256,宁可留 1 wave/EU 也不 spill)。真正有效的是减少**实际数据量**(BN)+ 错峰(sched_barrier)。BN=64 同时略微**提速**(37.6→38.9 TFLOPS),因 sched_barrier 也改善了调度。

## 6. 后续优化方向(v2 已为其铺路)

1. **软件流水线 / double-buffer**:`S` 已在寄存器,可在算当前 tile 的 gemm1/gemm2 时,预取**下一** tile 的 `K/V` 到另一块 LDS,掩盖载入延迟(这是 register-resident S 的主要收益点)。
2. **减少 broadcast**:调整 wave 布局或用更宽的 LDS 读,降低 `K/V` 的 4× 重复读。
3. **V 转置读优化**:对 `v_lds` 加 swizzle 或用 `ds_read_tr`(gfx950)消除转置读的 bank 冲突,替掉 16b strided。(注:直接把 V 做成 128-bit 已在 §8 验证**反而更慢**,因需放弃 register trick。)
4. **指令调度**:`rocdl.sched_barrier` / `s_setprio` 等交织 MFMA 与访存(参考 `preshuffle_gemm`)。

## 7. 复现

```bash
cd tests/flydsl
HIP_VISIBLE_DEVICES=2 python3 test_attn_gemm.py     # 8192, BM=128 BN=64, 打印精度 + 性能
# 查看寄存器数(VGPR/AGPR/spill):
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dump HIP_VISIBLE_DEVICES=2 python3 test_attn_gemm.py
grep -E '\.vgpr_count|\.agpr_count|vgpr_spill' dump/attn_kernel_0/*_final_isa.s
```

## 8. v5:register trick + V 全 128-bit(perm_M 让 C 累加器直接 8 连续 Nk)

**动机**:v4 里 K/Q 已 128-bit,但 V 仍是 16-bit 转置读。要让 V 也 128-bit,需给 `tmma2` 加 `k_perm`
(把 V 预 shuffle 成 paged 布局 `[N//8, D, 8]`,每 lane 沿 Nk 取 8 连续 → `buffer_load_dwordx4`)。
但 `tmma2` 加 `k_perm` 后,GEMM2 的 A 操作数 `frag_Sa` 每 lane 需 **8 连续 Nk**,而 register trick 复用的
GEMM1 C 累加器 `frag_St` 默认每 lane 只有 4 连续 Nk。

**关键洞察(v5)**:给 GEMM1 的 **M 维(Nk)也加 `perm_M=k_perm`**,让 C 累加器 `frag_St` 每 lane 直接
持 **8 连续 Nk**。此时:
- `frag_St = ((4,1),(2,2),2)`,`fx.select(frag_St,[0,2,1]) = ((4,1),2,(2,2))`,形状 `[4,2,(2,2)]`;
- `frag_Sa`(k_perm A)`= (4,2,(2,2)):(1,16,(4,8))`,形状 `[4,2,(2,2)]`。
- 两者**形状全等**(判据是形状全等,不是布局完全相同),`frag_Sa.store(select(frag_St,[0,2,1]).load())`
  逐元素对齐 → **register trick 仍成立**,S 不入 LDS。

于是 GEMM2 用 `k_perm` + paged V 直读 global,K/Q/A/V 读**全部 128-bit**,且 **S 常驻寄存器**。
**374 µs / 91.7 TFLOPS**(相对 v4 再 1.27x,达 rocBLAS 的 83%),VGPR=200。

**关键教训:早前判定「带 perm_M 的 select register trick 会 `ub.poison`」是误判**。用 `COMPILE_ONLY`
最小复现证明:`perm_M`+`k_perm` 下 `frag_Sa.store(select(frag_St,[0,2,1]).load())` **可编译**。真正的
poison 来自 **paged V 读**(`make_fragment_B(vt_tile)` 用了 paged 嵌套 layout 且无 buffer-tile 模板)。

**paged V 的两个必修坑**(与被取代的 path-b 相同):
1. **`make_fragment_B` 的模板要用 buffer-tile**(`v_fake = flat_divide(make_buffer_tensor(...), tile)`),
   不能用 paged 嵌套 layout 直接 `make_fragment_B`,否则 register SSA 提升报 `ub.poison`。读取源才用 paged 分区。
2. **paged layout 列主序**:Nk 子模 `(v, nb_local)` 中 **`v` 必须在前(inner, stride 1)**,即
   `(8, NB):(1, D*8)`;写反(`(NB,8):(D*8,1)`)则逻辑 nk 0..7 跨页读错元素,`rel_l2≈1.3`。
   `flat_divide` 对嵌套维会拆散页,需**手动构造** 3 维 tile `[D,(8,NB),N//BN]` 并 `[None,None,kv]` 索引。

**被取代的中间探索 path-b**(S 经 LDS 中转,不用 perm_M):也能让 K/Q/A/V 全 128-bit 且正确,但
**68.8 TFLOPS < v4 的 72.1** —— 放弃 register trick、让 `S^T` 经 LDS 中转的开销(写+读+序列化)超过了
V 128-bit 的收益。v5 用 perm_M 保住了 register trick,才真正把 V 128-bit 变成净收益。
(path-b 另一教训:C→LDS 后各 wave 读写自己的 Mq-slice,是 same-wave,无需 `gpu.barrier()`。)

## 9. v8:软件流水线(预取 K/V,藏全局 load 延迟)

**动机**:每个 KV-tile 的 K/V 全局 load(coop `buffer_load_dwordx4`)与后面的 `frag->LDS->frag->GEMM`
串行,全局延迟暴露。软件流水线把**下一轮**的全局 load 提前发起,藏在**当前**轮的 GEMM 后面。

**结构**(`range(..., init=...)` 的 loop-carried 双缓冲,见 `.claude/skills/prefetch-data-load`):
- **prologue**:先发起 K(0) 的协作加载(`global->frag`,异步)。
- **循环体**(处理 tile i):
  1. 发起 V(i) 协作加载(异步,与下面 K->LDS + GEMM1 重叠);
  2. 上一轮预取的 K frag -> `k_lds` -> `frag_K` -> **GEMM1**;register trick;
  3. **在 S*V 之前**发起 K(i+1) 协作加载(异步,与 V->LDS + GEMM2 重叠);
  4. V frag -> `v_lds` -> `frag_V` -> **GEMM2**。
- K 的协作数据作 **loop-carried 值**(SSA phi)在轮间传递 —— 这本身就是 ping-pong 双缓冲;
  末轮 `kv_i+1` 越界,buffer_load 返回 0 且不被消费,**无需 epilogue**。

**结果(M=N=20480,此尺寸下融合已超 rocBLAS 122 TFLOPS)**:

| 版本 | 延迟 | TFLOPS | VGPR |
|---|---:|---:|---:|
| v7(无流水线) | 1450 µs | 148.1 | 200 |
| **v8 软件流水线** | **1313 µs** | **163.6** | **184** |

**+10.5%,且 VGPR 反降到 184**(K/V 各只用 1 块暂存 + 1 块预取,而非旧的共用 + 立即消费)。

**关键教训:手动"循环展开 2 次 + 显式 A/B ping-pong 寄存器"反而更慢**(155.3 TFLOPS,VGPR 涨到
234):展开使两个子迭代的临时量(`frag_K/St/Sa/V` ×2)+ 4 块 ping-pong 缓冲同时存活,VGPR 压力抵消了
调度收益。`range(init=...)` 的 loop-carried phi **已经**是双缓冲,不必手动展开。

## 10. v9–v12:进一步优化(M=N=20480,cudaPerf 多 buffer 轮换计时)

从 v8 起改用 `cudaPerf`(`pyhip` 包)+ 多 buffer 轮换计时:每次计时读**不同**显存(L2 冷),测真实
HBM 而非 L2 命中的虚高;`cudaPerf` 进入时先 `torch.cuda._sleep` 掩盖 host 派发,再用 CUDA event 计
kernel 时间。基准 torch(rocBLAS 两次 matmul)= 122 TFLOPS。

| 版本 | 方法要点 | TFLOPS | VGPR | rel_l2 |
|---|---|---:|---:|---:|
| v8 软件流水线 | 见 §9 | 163.6 | 184 | 1.5e-4 |
| **v9 LDS swizzle** | `k_lds/v_lds` 加 `SwizzleType.get(3,3,3)` 去 bank 冲突 | **168.6** | 184 | 1.9e-4 |
| **v10 转置 GEMM2 + 64-bit O 写** | GEMM2 交换 A/B 算 `O^T`,O 存转置视图 → C 累加器 4/lane 沿 D 连续 → `buffer_store_dwordx2` | **170.5** | 184 | 1.9e-4 |
| **v11 V 直读(跳过 LDS)** | V 用 paged global buffer 视图直接 `buffer_load` 到 `frag_V`,去掉 V 的 coop+LDS 写+barrier+读 | **183.0** | 178 | 1.9e-4 |
| **v12 BN=32 + K LDS 双缓冲** | BN 64→32 腾 VGPR;K LDS ping-pong 双缓冲,write+barrier 与 GEMM2 重叠 | **229.8** | 190 | 1.9e-4 |

> **v8 → v12 累计 +40%,达 rocBLAS 的 1.9×。全程(v4 72.1 → v12 229.8)约 3.2×。**

### v9 — LDS swizzle(bank 去冲突)

`k_lds`、`v_lds`(及 V 的 paged 读视图 `v_lds_T`)都套上 `swz = fx.SwizzleType.get(3, 3, 3)`(bf16;
参数是 `(mask, base, shift)`,`swz(x)=x ^ ((x & ((2^mask-1)<<(base+shift))) >> shift)`,period =
`2^(mask+base+shift)` = 512 元素),经 `make_view(ptr, make_composed_layout(fx.static(swz), 布局))`。
**写视图与读视图必须用同一个 swz** —— swz 只是对线性字节偏移做置换,两视图对同一物理字节算出相同 flat
offset,故 `swz(k)` 一致、数据正确;且 swz 保留低位,128-bit `ds_read/ds_write` 不受影响。→ **+3%**。

- **swizzle 参数扫描**:`(3,3,3)` 实测最优(170.5 版基线上 168.6);`(3,4,3)` 持平但把 K 读变成
  `ds_read_b64`(64-bit)无收益;**弱 swizzle `(1,3,3)` 暴跌到 94.5**(period 缩到 256B,bank 冲突暴增)。
- **`ds_read2st64_b64` 现象**:K 读(`k_lds`)加 swizzle 后从 `ds_read_b128` 变成 `ds_read2st64_b64`
  (2 个隔 1024B=1 period 的 b64)。**这不是性能损失**:`ds_read2st64_b64` 仍是 128-bit 吞吐,去掉
  k_lds swizzle 反而 164.1 < 170.5(bank 去冲突收益 > 读指令形式变化)。V 的 paged 读(内层 8-v 落在
  base=3 保护的低 3 bit)保持 `ds_read_b128`。**结论:swizzle 的 bank 去冲突远比 ds_read 指令形式重要。**

### v10 — 转置 GEMM2 + 64-bit O 写

**动机**:`cp_oc` 想用 64-bit 写 O,但 MFMA_16×16×16 的 C 累加器每 lane 存 **4 个连续 M 行**、固定
N(=D)列;在 `O[M,D]`(D 连续)里这 4 个值按 D 步长跨开,不连续 → 直接 64-bit 写**错误**(rel_l2 0.87)。

**解法**:GEMM2 计算 `O^T = V^T @ S^T`(**交换 A/B 操作数**)→ C 累加器变成 `O^T[D, Mq]`,4/lane 沿
`M'=D` 排列,在 `O[M,D]` 里正好**连续** → `cp_oc` 用 `BufferCopy64b`,16 条 `buffer_store_dwordx2`
替掉 64 条 `buffer_store_short`。附带发现:转置后 `tmma1`/`tmma2` 同 wave 布局 `(1,WAVES,1)`+同
`k_perm`,GEMM1 的 C(=S^T)直接就是 GEMM2 的 B(=S^T),register trick 的 `select([0,2,1])` 仍成立。
`make_fragment_B` 模板取 `[N,K]=[Mq,Nk]=[BM,BN]`(取错成 `[BN,BM]` 会触发 `Mismatch in loop_n`)。
compute-bound 下收益小(+1.1%),但免费且正确。

### v11 — V 直读(跳过 LDS)

V(GEMM2 的 A 操作数 `V^T[D,Nk]`)改为**直接从 global** `buffer_load` 到 `frag_V`,不经 LDS:
```python
v_g = fx.rocdl.make_buffer_tensor(
    fx.make_view(fx.get_iter(V_), fx.make_layout((D, (8, NB), N // BN), (8, (1, D * 8), BN * D)))
)
# 循环内: fx.copy(cp_vg, tcV.partition_S(v_g[None, None, kv_i]), tcV.retile(frag_V))
```
去掉 V 的整条 LDS 路径(coop 加载 + LDS 写 8.1% + barrier 2% + LDS 读 4.9%,ATT 里约 15%)→ **+7.3%**。

- **为什么 V 能直读、K 不能**:V 是 GEMM2 的 A 操作数,每 wave 读**自己**的 `V^T[D,Nk]`,且 V 常驻 L2;
  K 是 GEMM1 的 A 操作数,**在 4 个 N-wave 间广播**(每 wave 要全部 Nk×D)。K 直读 = 每 wave 各读一遍
  → **4× 冗余 + 合并度差**,实测全直读(K+V)只有 49.7(BN=64)/ 69.3(BN=32)TFLOPS。
  **广播操作数(K)必须走 LDS**(协作合并读一次 + LDS 广播)。
- **编译器坑**:`perm_M`(GEMM1 的 A 在 M=Nk 维的 k_perm)+ 直接 `buffer_load` 触发 LLVM `Invalid cast`。
  更一般的规则:**`BufferCopyNb` 宽度要匹配每 lane 连续元素数**(k_perm→8→128b;无 perm→4→64b;
  128b 灌进 4 元素/lane 的 frag 会崩)。

### v12 — BN=32 + K LDS 双缓冲(关键突破,229.8 TFLOPS)

**ATT 定位(v11)**:#1 停顿是 GEMM1 等 K 的 LDS 读(`s_waitcnt lgkmcnt`)= **54.9%** —— K 走
`coop → LDS 写 → barrier → LDS 读 → GEMM1 MFMA` 串行,读延迟暴露。

**K LDS 双缓冲**:开 2 块 `k_lds`,本轮读 `k_lds[kv%2]`(上轮写好、barrier 已过),同时把 K(kv+1) 写到
`k_lds[(kv+1)%2]`,barrier 放到 GEMM2 之后 → **write+barrier 移出 GEMM1 关键路径**,与 GEMM2 重叠。
staged 视图 `k_lds2 = make_view(ptr, make_composed_layout(swz, make_layout((2,BN,D),(BN*D,D,1))))`,
运行时索引 `k_lds2[stage, None, None]`(`k_lds2[stage]` 不行,返回非 Value);stage 偏移 `BN*D` 是
swizzle period 整数倍 → 两 buffer swz 一致。

**关键:先砍 VGPR 再上双缓冲。** 双缓冲的运行时 `kv%2` 让 LDS 读地址变动态,吃寄存器:

| 配置 | VGPR | 占用率 | TFLOPS |
|---|---:|---|---:|
| v11(BN=64 单缓冲) | 178 | 2 waves/SIMD | 183 |
| BN=64 双缓冲 | 265 | **1 wave/SIMD** | 115 ❌ |
| BN=32 单缓冲 | 152 | 3 waves/SIMD | 184 |
| **BN=32 + 双缓冲** | **190** | **2 waves/SIMD** | **229.8** ✅ |

BN=64 双缓冲(265 VGPR)掉到 1 wave 反而暴跌;**BN=32 把 K 操作数减半(178→152)腾出预算**,双缓冲塞进去
后仍保持 2 waves(190),既隐藏了 K 读停顿又不掉占用率 → **+25%**。ATT 复测:总停顿 82.5%→66.4%,
#1 的 54.9% K 读停顿被消到 1.6%,瓶颈变均衡(MFMA 38% / LDS 19% / VMEM 17% / barrier 14%)。

## 11. 失败的尝试与核心教训(VGPR → 占用率悬崖)

v12 之后瓶颈已均衡、无单一大头,且 kernel **死死卡在 2 waves/SIMD 的 VGPR 预算(190/256)**。所有
**增加寄存器的重构一律回退**:

| 尝试 | VGPR | 占用率 | TFLOPS | 回退原因 |
|---|---:|---|---:|---|
| unroll-by-2(v8 期) | 234 | — | 155(<163.6) | 展开使临时量翻倍 |
| V ping-pong @BN64 | 344 | 1 wave | 132 | 掉占用率 |
| K 双缓冲 @BN64 | 265 | 1 wave | 115 | 掉占用率 |
| V 预取 @BN32 | 222 | **2 waves** | 210.8 | **不掉占用率也慢** |
| 2-stage 计算流水 @BN32 | 200 | **2 waves** | 208.5 | **不掉占用率也慢** |
| hot_loop_scheduler(手写调度) | 190 | 2 waves | 185.6 | 手写不如编译器 |
| K 不经 LDS(全直读) | — | — | 69.3 | 广播 K 4× 冗余 |

**教训**:
1. **`reg-adding` 流水线要先造 VGPR 余量再上**:v12 唯一成功,靠先 BN=32 砍 VGPR 保证双缓冲后 ≥2 waves;
   BN=64 无余量(178)所以双缓冲失败。
2. **即使不掉占用率也可能变慢**:V 预取(222)、2-stage 流水(200)都仍是 2 waves、无 spill,但多携带
   loop-carried 状态**挤占了编译器交错 MFMA 的自由寄存器** → 变慢。
3. **广播操作数必须走 LDS**:K 直读永远赢不了 coop+LDS 广播。
4. **手写指令调度不如编译器**:`sched_group_barrier` 硬排 GEMM1-then-GEMM2 比编译器默认交错差(185.6)。
5. **`ds_read2st64_b64` / 宽读指令形式** 不是性能关键;bank 去冲突、占用率、隐藏关键路径延迟才是。

**天花板**:v12 = 229.8 TFLOPS 是当前结构在 2-wave VGPR 预算下的平台。再往上需要**根本上更省寄存器的
结构**(如 FP8 K/V 减半带宽+VGPR)或换 MMA 指令(32×32×8),属大改、非增量。




