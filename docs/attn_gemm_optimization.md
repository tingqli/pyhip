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

> **后续被 §12 推翻**:v12 不是终点。找到一个"省寄存器的等价变换"(把 perm_M 挪到全局 K)腾出预算后,
> 就能继续上流水(K-prefetch),v13 达到 235 TFLOPS。见 §12。

## 12. v13:突破天花板 —— perm_M 挪到全局 K + K-prefetch(235 TFLOPS)

§11 判断 v12(229.8)是 2-wave VGPR 预算下的平台,任何"加寄存器的流水"都回退。**但按「先腾寄存器、再上流水」
这条主线,v13 打破了天花板:同尺寸 M=N=20480 下 235.8 TFLOPS(1.92x,> v12 的 229.8);放大到 M=N=40960
达 266.0 TFLOPS(2.24x rocBLAS 119)。rel_l2=0.00021,VGPR 204,2 waves/SIMD。** 三步缺一不可,
`KLDS=gpermswz` 现为默认;`KLDS=swz` 走 v12 路径(此结构下 20480=168 / 40960=219.6,见下)。

### v13a — perm_M 从 MMA 挪到全局 K(gpermswz):直接看是负优化,实为腾寄存器

v12 的 perm_M(GEMM1 在 M=Nk 维加 k_perm)让 C 累加器每 lane 8 连续 Nk;它同时占着 MMA 的寄存器/寻址。
v13a 把这个 Nk 重排**预先做进全局 K**:coop 读源按**正向** k_perm `make_layout((4,4,2):(D,8D,4D))` 重排,
GEMM1 的 MMA **去掉 perm_M**(`make_tile(None, None, k_perm)`)。plain 读入 LDS 后,不加 perm_M 也能得到
**同样的 8-连续-Nk C 布局**(N 方向已在全局 shuffle)→ register trick 与 GEMM2 **完全不变**,正确。

- **单独看是负优化:197 < 230**。原因:(a) 全局 K 的 coop 读从连续变成嵌套 (4,4,2) 散读、合并度差;
  (b) 去 perm_M 后 LDS 读退回 `ds_read_b128`(swizzle 只能部分去冲突),不如 v12 的 `ds_read2st64_b64`。
- **但它腾出了 MMA 的寄存器预算**——这是后两步的关键铺垫。
- 正确性坑:方向必须是**正向 k_perm**(用逆 `(4,2,4)` → rel_l2≈1.2);且**必须同时去掉 perm_M**
  (只挪全局不去 perm_M = 双重 shuffle,也是 1.2)。

### v13b — 循环展开 2 次

每轮处理 2 个 kv,LDS stage 的 rd/wr 变编译期常量(0/1),消掉运行时 `kv%2`。展开本身不是收益点
(单独在 v12 上展开 = 175 回退,VGPR 190→254;在 gpermswz 上 = 194),它是为 K-prefetch 铺路——
静态 stage 让预取地址简单、`frag_ldK`/`frag_ldK_next` 做 coop ping-pong。

### v13c — K 读做成 prefetch(关键收益)

v12 的 K 读是 `coop → LDS 写 → barrier → LDS 读 → GEMM1` 串行,LDS 读延迟暴露在 GEMM1 前(v11 ATT 里这是
#1 停顿 54.9%,v12 用双缓冲把 write+barrier 移出关键路径,但**读**仍在 GEMM1 前)。v13c 把 `LDS 读 → frag_K`
从 GEMM1 **之前**移到 GEMM2+barrier **之后**:读 `k_lds[wr]`(刚写好且 barrier 过的 buffer)供**下一步**的
GEMM1;prologue 先预取第一个 `frag_K`;**`frag_K` 变成第 3 个 loop-carried 值**(`init=[frag_O, frag_ldK,
frag_K]`)。GEMM1 于是直接用已就绪的 `frag_K`,LDS 读延迟被上一步的 GEMM2 掩盖。→ **197 → 235.8**。

### 核心洞见:先砍寄存器,再上流水

- K-prefetch 单独加在 **swz** 上 = **168**(回退):swz 的 perm_M 还占着寄存器,多携带 `frag_K` 撑爆调度。
- **gpermswz 去了 perm_M → 216 VGPR 仍 2 waves → 装得下 `frag_K` + 流水 → 235.8。**
- 所以**把 perm_M 挪到全局的真正价值不是直接提速**(那步 197 反而更慢),而是**腾出寄存器,让下游的
  K-prefetch 流水成为可能**。这和 §11 里 v11→v12(先 BN=32 砍 VGPR 才装得下 K 双缓冲)是**同一条主线**:
  在 2-wave VGPR 预算下,任何"加寄存器的流水"都必须先在别处**省出等量寄存器**。
- **修正 §11 的"天花板":** v12 不是终点。只要能找到"省寄存器的等价变换",就能继续叠流水。下一步方向:
  把全局 K 的嵌套散读变回连续合并(host 侧物理预 shuffle K),或对 V 也做同样的 prefetch。

| 版本 | 方法 | TFLOPS | VGPR |
|---|---|---:|---:|
| v12 | swz(perm_M 在 MMA + K LDS 双缓冲,干净 v12) | 229.8 | 190 |
| v13a | + perm_M 挪全局 K(gpermswz) | 197 | — |
| v13b | + 展开 2 次 | 194 | — |
| **v13c** | **+ K-prefetch(= 当前默认 `gpermswz`)** | **235.8** | 216 |
| 参照 | 当前文件 `KLDS=swz`(v13 结构但 perm_M 在 MMA) | 168 | — |

> 上表 TFLOPS 为空闲 GPU 复测中位数(cudaPerf 多 buffer 轮换,**M=N=20480**);torch(rocBLAS)= 122.7 → **1.92x**。
> **放大到 M=N=40960(当前 main 默认):gpermswz = 266.0 TFLOPS(2.24x rocBLAS 119),swz = 219.6,rel_l2=0.00021**
> ——更大问题摊薄 host/prologue 开销;`hot_loop_scheduler` 手工微调(is_first_gemm 先排 8×(vmem+3mfma),else 用 mfma(7))
> 把 gpermswz 从初版 248.7 逐步抬到 266.0(VGPR 216→204),sched 指令数按 tile 尺寸(BN·D)自动算。
> 注意 `KLDS=swz`(20480=168 / 40960=219.6)因 perm_M+frag_K 撑爆寄存器而慢于 gpermswz;干净 v12 的 swz 才是 229.8。


## 13. 完整 MHA:multi-head + flash softmax

在 v13(266 TFLOPS 无 softmax 双 GEMM)之上加完整的多头注意力(non-causal)。分两阶段落地:

### 阶段 A — multi-head 布线

- `build(..., H)` 加 head 维;`launch` grid `(M//BM, H, 1)`,`h = block_idx.y`。
- 每 head 的 Q/K/V/O 按 head 偏移基址:**用 `fx.make_view(fx.get_iter(X_) + off, layout)`**(`qo_off=h*M*D`,
  `kv_off=h*N*D`)。**坑:`fx.Tensor(view)[h]` 返回标量元素,不是 head 子视图**——必须用 iter 偏移。
- `_make_ktiles(..., koff)` / `v_g` / `v_fake` 内部都 `get_iter(X_) + koff`。
- 验证:8 头无 softmax vs `(Q@K^T)@V` 逐头参考,rel_l2 = **0.00012**。

### 阶段 B — 在线 flash softmax

GEMM1 出 `frag_St`(= S^T[Nk, Mq],C 累加器 3 mode `((4 Nk),(2 Nk-tile),(2 Mq-tile))`)后,register trick 前插在线 softmax:

```python
for mt in range_constexpr(2):                 # 按 Mq-tile 切片(mode2)
    v    = frag_St[None, None, mt].load() * sm_scale   # 该 Mq-tile 的 8 个 Nk
    tmax = v.reduce("max")                     # lane 内 8 Nk 求 max(v_max3 链)
    for sh in (16, 32):                        # 跨 lane:合并 l//16 的 4 行组
        tmax = _maxnumf(tmax, tmax.shuffle_xor(sh, 64))
    nm   = _maxnumf(m[mt], tmax)               # 新 running max
    corr = _exp2_amdgcn((m[mt] - nm) * LOG2E)  # 旧统计的校正因子
    p    = _exp2_vec_amdgcn((v - nm) * LOG2E)  # P^T 片段(exp2 用 amdgcn 单 v_exp_f32)
    ts   = p.reduce("add") + 跨lane(16,32)     # tile 内 P 求和
    l[mt] = l[mt]*corr + ts;  m[mt] = nm
    frag_St[None,None,mt].store(p)             # 就地覆盖 frag_St = P,复用原 register trick
frag_O[None,None,mt] *= corr                    # GEMM2 累加前 rescale 旧 O
# 循环携带 m0,m1,l0,l1;epilogue: frag_O[None,None,mt] *= 1/l_final[mt]
```

- **`m`/`l` 循环携带**(每 lane 2 个 Mq-tile 标量,冗余存 4 行组);`frag_O` 与 `frag_St` 共享 Mq=col 结构,
  所以 correction 按 Mq-tile 正确广播到 `frag_O` 的 D 元素。
- **exp2 用 `llvm.amdgcn.exp2.f32`**(单 `v_exp_f32`,省 OCML 的 `v_ldexp`);指数 ≤ 0 在 fast-range,安全。
  相比 `.exp2()`(OCML):2048 尺寸 98.9 → 119.6 TFLOPS(+21%),精度不变。
- 精度:rel_l2 = **0.00311**(bf16 attention 正常范围),多尺寸一致。

### 性能(空闲 GPU,cudaPerf CUDA-event 计时)

| 配置 | per-head M=N | TFLOPS | 备注 |
|---|---|---:|---|
| H=8 | 8192 | 150 | |
| **H=8** | **16384** | **~164** | **最佳(3 次 156/164/164)** |
| H=4 | 20480 | 162 | |
| H=1 | 40960 | 160 | 单头,直接对比无 softmax 266 |
| H=8 | 20480 | 143 | 超长跑降频 |

- 好尺寸 plateau **160–164 TFLOPS**;vs 无 softmax 266@40960 → softmax 约 **−38%**。
- VGPR **204 → 247**(m/l 携带 + exp 中间量),但 `512/247 = 2 waves/SIMD` **occupancy 仍保住**,无 spill,LDS 16KB。

### 跨-lane reduce 的硬件约束(为何仍是 ds_swizzle/ds_bpermute)

softmax 的跨-lane 归约是**列向的跨-row all-reduce**:`frag_St` 的 C 布局里 Nk = `4*(l//16)+reg`(行组),
Mq = `l%16`(列)。over Nk 归约 = 合并 4 个行组(l//16),即 `xor 16` + `xor 32`。当前分别降为
`ds_swizzle_b32 SWAP,16`(xor16,32-lane 组内)和 `ds_bpermute_b32`(xor32,跨 32-lane)。

**gfx942(CDNA3)无法用 DPP 消掉它们**:
- `permlane16` / `v_permlanex16` / `row_xmask` DPP 都是 `isGFX10Plus`(RDNA)特性,CDNA3 没有。
- DPP row 操作只在 **16-lane 行内**(offset 1/2/4/8);`row_bcast:15/31` 是**单 lane 广播**,会混掉不同 Mq 列,
  不能做列向归约。
- 因此 gfx942 上寄存器级的列向跨-row 归约无 DPP 通路;LDS(ds_swizzle/ds_bpermute)或 readlane/writelane 是仅有选项。

**要彻底去掉 ds 归约,只能让归约变成行内(intra-row)**:即 GEMM1 改算 S[Mq, Nk](Nk 成为列 `l%16`)→ over Nk
= 行内 DPP(xor 1/2/4/8)。但这要求 GEMM2 也转置成 O[Mq,D](A=S、B=V),连带改 V 的 B-operand 布局与 O 的存出
(丢掉当前 O^T 的 64-bit 连续写)。属整核重构,权衡:去掉热循环里每步的 ds 归约 vs. V/O 通路重写 + 存出可能变散。

**但实测证明不值得**:临时去掉两句 `shuffle_xor`(perf probe,结果故意算错只测速),H=8/8192 同会话同 GPU 对比:

| 版本 | rel_l2 | TFLOPS |
|---|---|---:|
| baseline(有 ds 归约) | 0.00311(正确) | 150.5 |
| probe(去掉 ds 归约) | 错(仅测速) | 151.1 |

**差 +0.4%** ——`ds_swizzle`/`ds_bpermute` 已被 MFMA 流水完全掩盖,**不是瓶颈**;而且 gfx942 上 DPP 也做不了列向跨-row
归约(`dpp_xor_f32` 只有 1/2/4/8,够不到 xor16/32)。所以 DPP 重构对本核**无收益**,不做。
softmax 的 −38% 开销来自别处:VGPR 204→247(挤压调度)、`v_exp_f32`、以及在线 softmax 的 `v_pk_mul`/`v_sub`/`v_add`
(缩放/correction/rescale)占用 VALU 发射槽 —— 优化要往**压 VGPR / 减 VALU / 调度**方向,而非跨-lane 归约。

### VALU 优化 1:LOG2E 折进缩放(+6.6%)

既然 softmax 是 VALU-bound,先砍 `* LOG2E`。指数 `(S·sm_scale − m)·LOG2E = S·(sm_scale·LOG2E) − m·LOG2E`。
定义 `sm_scale_log2 = sm_scale * LOG2E`,开头就用它缩放(`v = S * sm_scale_log2`,把 `m` 也带进 log2 域):

```python
sm_scale_log2 = float(sm_scale * LOG2E)     # build 内
corr[mt] = _exp2_amdgcn(m_in[mt] - nm)      # 免标量 *LOG2E
p        = _exp2_vec_amdgcn(v - nm)          # 免逐元素(8 Nk)*LOG2E
```

`m` 只用于差值(corr/p)、`l` 是 exp 之和、`O/=l` 不变,放 log2 域完全等价。消掉每 kv-step 的 8 个
`v_pk_mul`(向量 *LOG2E)+ 2 个标量 *LOG2E。**H=8/8192:150.5 → 160.4 TFLOPS(+6.6%),rel_l2 不变 0.00311**
(3× 稳定);单头 40960 = 160 → 164.4。对比去 ds 的 +0.4%,坐实了“softmax 是 VALU-bound”。

> 注:large-size(16384/20480)在共享机上连续跑会热降频(同一配置多次跑递减),短跑的 8192(~1.7ms)才稳定可信。

### 回退的尝试(均为负优化)

下面两个"减 VALU/VGPR"的直觉改动**实测变慢**,已回退(请勿重试):
- **合并 O-rescale 进 softmax 循环**(`frag_O *= corr` 从独立循环挪到主循环内、corr 即用即弃):同 GPU
  156.0→152.4(−2~5%)。指令数不变,纯因重排——**独立的 rescale 循环反而调度更好**(编译器把 32 个
  `v_pk_mul` 批量与 MFMA 重叠;交织进 exp/reduce 依赖链反而上关键路径)。
- **预缩放 Q**(`frag_Q *= sm_scale_log2` 一次,免循环内 `v=S*scale`):158.5(−)且 rel_l2 0.00311→0.00358
  (bf16 乘常数掉精度)。且初始 `v=S*scale` 那批 mul 也被 MFMA 掩盖,去掉不帮忙。

**经验**:此核对指令顺序极敏感、已被编译器排得很好;只有**减关键路径上的 VALU**(如 LOG2E 折入 exp 前的乘法)才有效,
而重排/搬动那些已被 MFMA 掩盖的 VALU 反而打乱调度。VGPR 247 仍 2 waves(512/247),降到 ≤170 才能 3 waves,softmax 微调达不到。

### Profile 总结 + FA3/scheduler 结论

ISA 指令统计(LOG2E-fold 版,VGPR 238;每轮循环 = 2 kv-step):

| 指令 | 数 | 归属 |
|---|---:|---|
| `v_mfma_16x16x16_bf16` | 128 | 计算(有效功) |
| `v_pk_mul_f32` | 99 | **O-rescale ~64** + scale ~16 |
| `v_perm_b32` | 48 | register trick + O 存出 |
| `v_exp_f32` | 36 | softmax exp |
| `v_sub_f32` | 36 | v−nm, m−nm |
| `ds_swizzle+ds_bpermute` | 16 | 跨-lane 归约(已确认被 MFMA 掩盖) |

- **VALU-bound**:128 MFMA vs ~300 VALU/轮;最大 VALU 是 **O-rescale(64 v_pk_mul)**,其次 exp(36),二者均为 online softmax 固有,难降。
- **hot_loop_scheduler 无效(实测)**:`SCHED=0`(编译器默认)vs `SCHED=1`(手工)= 156.5 完全相同。无 softmax 时手工调度把 GEMM 从初值抬到 266;softmax 后瓶颈变 VALU 吞吐,mfma 调度 hint 不再起作用。
- **FA3 不适用**:FA3 核心(异步/warp-specialization/wgmma/FP8)是 Hopper 专属,CDNA3 无对应硬件;可迁移的"softmax-VALU 与 MFMA overlap"被 register-trick(GEMM1→softmax→GEMM2 硬依赖)+ `gpu.barrier`(串行化 kv-step)挡住,且本核 mfma 不是瓶颈(overlap GEMM 无益);FA3-FP8 只降 mfma、反而更 VALU-bound。
- **结论**:softmax 核 156~164 TFLOPS 已接近此在线-softmax 公式在 CDNA3 的 **VALU roofline**;本轮净胜 = LOG2E 折叠(+6.6%)。唯一(高风险)剩余杠杆:用 LDS 承载 S 替 register-trick,卸掉 48 perm+32 add 的 VALU(代价:加 LDS 流量,收益不确定)。

### 编译时 softmax 开关

`build(..., softmax=True)`(环境变量 `SOFTMAX`,默认 1)。关闭时退化为纯双 GEMM(`S@V`,无 softmax/scale),用于对照无 softmax 基线。

- **必须用 `if const_expr(softmax):`**,不能用裸 `if softmax:`——FlyDSL `@flyc.kernel` 里裸 `if` 被当运行时分支,内部赋值不传播(报 `UnboundLocalError: m0`)。共 5 处:kv_step 的 softmax 块、`loop_init`、state 解包、`yield_vals`、epilogue;loop-carried/yield/epilogue 全部按 `softmax` 条件化(关闭时只 carry `[frag_O, frag_ldK, frag_K]`,不带 `m/l`)。
- 关闭时是纯编译期分支,**零** softmax 指令,精确回到 266 基线路径。
- 验证:`SOFTMAX=1` @8192 = 160.6(rel_l2 0.00311);`SOFTMAX=0` @8192 = 250.1 / **@40960 = 265.2**(rel_l2 0.00023)。

同尺寸(H=1, M=N=40960)直接对比 softmax 开销:

| 模式 | rel_l2 | TFLOPS | 相对 |
|---|---|---:|---|
| `SOFTMAX=0`(纯双 GEMM) | 0.00023 | 265.2 | 基线 |
| `SOFTMAX=1`(flash softmax) | 0.00317 | 164.1 | **−38%** |

softmax 开销 −38%,与 profile 结论一致(VALU-bound,O-rescale + exp 主导,已接近 CDNA3 上此在线-softmax 公式的 VALU roofline)。

## 14. SOFTMAX=1 后续实验:条件 rescale、lazy rebase、two-pass、BN/pipeline

基准配置:gfx942,`H=8,M=N=8192,D=128,BN=32`,空闲 GPU,cudaPerf 多 buffer 中位数。下表模式均为
历史实验结果;实验结束后已删除失败分支,当前代码在 `SOFTMAX=1` 时**只保留最快的 lazy Δ=8**,
`SOFTMAX=0` 仍是纯双 GEMM 对照。

| 模式 | 方法 | TFLOPS | 相对 online | rel_l2 |
|---|---|---:|---:|---:|
| online | 原在线 softmax | 150.6 | 基线 | 0.00311 |
| branch | 仅 `tmax>m` 时计算 correction 并 rescale O | 158.4 | **+5.2%** | 0.00311 |
| **lazy(Δ=8)** | 仅 `tmax>reference+8` 时 rebase | **166.9** | **+10.8%** | 0.00316 |
| twopass | 第一遍全局 max,第二遍固定 max 做 exp/sum/PV | 93.0 | -38.2% | 0.00314 |
| pair | 两个物理 BN32 score 合并一次 64-key lazy 更新 | 161.2 | +7.0% | 0.00316 |
| lazy,BN=64,K-prefetch=0 | 更新次数减半,关闭跨迭代 K prefetch | 128.8 | -14.5% | 0.00316 |
| lazy,BN=64,K-prefetch=1 | 更新次数减半,保留 K prefetch | 128.0 | -15.0% | 0.00316 |

### 方案 1:条件 correction/O-rescale(+5.2%)

标准 online 更新中,只有新 tile max 超过 running max 时 `corr=exp2(m-nm)<1`;否则 `corr=1`,整块
`frag_O*=corr` 是无效工作。`branch` 用运行时 `scf.if` 跳过 correction 和 O-rescale。结果从 150.6 提升到
158.4 TFLOPS,精度不变。分支条件在一个 wave 的 16 个 query row 间并不一致,只要 wave 内任一活动 lane 需要
rebase,对应 VALU 仍会执行,因此收益有限。

### 方案 2:lazy rebase(+10.8%,当前最佳)

online softmax 不要求参考指数每轮都等于已见最大值。保持参考值 `r`,始终累计
`p=2^(score-r)`,`l=sum(p)`,`O=sum(p*V)`,最终 `O/l` 与标准 softmax 相同。只有
`tmax>r+LAZY_DELTA` 时才把 `l/O` 乘 correction 并更新 `r`;默认实验阈值为 log2 域 8。

| LAZY_DELTA | TFLOPS | rel_l2 |
|---:|---:|---:|
| 2 | 162.6 | 0.00315 |
| 4 | 165.4 | 0.00316 |
| **8** | **166.9** | 0.00316 |
| 16 | 166.8 | 0.00316 |

阈值 8 后已进入平台。`H=1,M=N=40960` 长序列复测:online 169.7 → lazy 181.2 TFLOPS(+6.8%),
`rel_l2 0.00317→0.00319`。全 1、单调上升 score、大幅随机 logits 三类边界输入均 finite;
对应 rel_l2 为 0、0.00331、0.00131。

PMC(`H=8,M=N=8192`)证实收益来自跳过 O-rescale,不是 occupancy:

| 每 wave PMC | online | lazy Δ=8 | 变化 |
|---|---:|---:|---:|
| `SQ_INSTS_VALU` | 53829 | 46567 | -13.5% |
| `SQ_INSTS_VALU_MUL_F32` | 12450 | 2370 | **-81.0%** |
| `SQ_INSTS_VALU_TRANS_F32` | 4610 | 4610 | 不变 |
| `MfmaUtil` | 48.0% | 50.0% | +2.0pp |
| `ValuPipeIssueUtil` | 49.5% | 46.1% | -3.4pp |
| VGPR | 240 | 240 | 不变,2 waves/SIMD,无 spill |

注意 correction `v_exp_f32` 虽位于分支内,LLVM 在无副作用条件下会投机执行,所以 transcendental 计数不降;
真正减少的是条件块内约 81% 的 O-rescale MUL。若要继续优化,应集中在禁止/延迟 correction exp 的投机执行,
或减少每 tile 固有的 16 个 probability exp,而不是再调 O-rescale 指令顺序。

已尝试把 correction exp 改成 `has_side_effects=True` 的 inline `v_exp_f32`,强制它留在 rebase 分支内;
精度不变,但 `H=8,M=N=8192` 从 166.9 降到 160.1 TFLOPS。side effect 阻断了有益的跨分支调度,
收益小于调度损失,已回退到纯 intrinsic。

### 方案 3:精确 two-pass(失败)

第一遍只执行 QK+max,重置 K 双缓冲 pipeline 后,第二遍用固定全局 max 执行 QK+exp/sum/PV。它彻底去掉
online correction/O-rescale,但多做一遍 QK,并重复 K 全局/LDS 流水。实测仅 93.0 TFLOPS;额外 MFMA 与访存远大于
省下的 VALU,不采用。

### 方案 4:BN/pipeline(失败)

- `BN=32,K_PREFETCH=0`:lazy 166.9 → 161.7,说明现有 K-prefetch 仍有约 3.2% 收益。
- `BN=64`:softmax 统计更新次数减半,但 K/V/S fragment 翻倍、live range 变长;无论 K-prefetch 开关都约
  128 TFLOPS,远慢于 BN=32。
- `pair`:保持 K/V 物理 tile 为 BN32,在寄存器同时保留两块 score,合并一次 64-key max/sum/correction,
  再执行两次 PV。精度正确、VGPR=230、2 waves/SIMD、无 spill,但只有 161.2 TFLOPS,低于普通 lazy 的 166.9。
  原因不是 occupancy,而是执行顺序被拉成长依赖链 `2×GEMM1 → softmax → 2×GEMM2`,破坏原先逐 tile 的
  GEMM/访存交错。既然纯寄存器 pair 已回退,再增加约 16KB score LDS 写读和同步没有继续实现价值。

当前最快实现复现:

```bash
cd tests/flydsl
HIP_VISIBLE_DEVICES=2 H=8 MULT=64 SOFTMAX=1 python3 test_attn_gemm.py
```

## 15. Fastest-only PMC + MFMA/VALU co-issue 复测

代码收敛为 `BN=32 + K-prefetch + lazy rebase Δ=8` 后重新采集。非 profiler cudaPerf 基线为
**1654 us / 166.1–166.3 TFLOPS / rel_l2=0.00316**;ISA 为 240 VGPR、2 waves/SIMD、16KB LDS、无 spill。
PMC 会把绝对时间扰动到约 2.38ms,所以下表只使用硬件计数及同类比值。

### PMC 结果

配置:`H=8,M=N=8192,D=128`;每组最多 4 个 counter,分别采集并按 dispatch 取中位数。

| 指标 | 结果 |
|---|---:|
| instructions / wave | 63,417 |
| MFMA / wave | 16,384 |
| non-MFMA VALU / wave | 30,183 |
| F32 ADD / wave | 9,600 |
| F32 MUL / wave | 2,370 |
| F32 TRANS(`v_exp`) / wave | 4,610 |
| `ValuPipeIssueUtil` | 46.42% |
| `MfmaUtil` | 50.37% |
| `OccupancyPercent` | 21.23% |
| active cycles / wave | 73,067 (33.06%) |
| dependency wait / wave | 30,047 (13.60%) |
| **issue wait / wave** | **117,902 (53.35%)** |
| issued IPC (`SQ_INSTS/SQ_ACTIVE_INST_ANY`) | 0.868 |

结论:lazy 已大幅消掉 O-rescale MUL,剩余大头不是 HBM/L2,而是 scheduler issue 空洞。超过一半 wave 周期是
`SQ_WAIT_INST_ANY`;依赖等待只有 13.6%。VALU 与 MFMA 利用率都约 50%,说明两条独立 pipeline 理论上有重叠空间,
但单独把两个百分比相加不能得到交集。

### co-issue 的定义与实测

ROCm 7.2 的 gfx950 配置提供 `VALU Co-Issue Efficiency`,公式依赖 `SQ_ACTIVE_INST_VALU2`;gfx942 不暴露
`SQ_ACTIVE_INST_VALU2`,且 gfx942 profile 配置明确删除该指标。因此 **gfx942 无法用 PMC 精确报告官方 VALU
co-issue**。这里增加一个 ATT 时间线指标,专门回答本 kernel 的 MFMA/VALU overlap:

1. 按物理 SIMD 合并其所有 raw wave issue 时间线(`seX_smY_*`)。
2. 每条 `v_mfma_f32_16x16x16_bf16` 建立 `[issue, issue+16)` busy window并取并集。
3. 普通 VALU/EXP 每次 issue 视为 4-cycle slot,计算它与 MFMA busy-window 并集的交集。
4. 对 16 个被跟踪 SIMD 取中位数。

ATT(dispatch 86,完整 128 workgroup grid)结果:

| 自定义 co-issue 指标 | 结果 |
|---|---:|
| **MFMA busy cycles covered by VALU issue** | **13.29%** |
| VALU instructions issued during MFMA busy | 26.29% |
| VMEM instructions issued during MFMA busy | 26.77% |
| LDS instructions issued during MFMA busy | 53.00% |

因此 K-LDS pipeline 的 overlap 已较好,真正不足的是 softmax VALU/EXP 与 MFMA overlap。ATT 同时显示总 stall
48.63M/80.56M=60.4%;其中 MFMA dependency 49.6%,softmax packed add/mul 链是主要 non-MFMA stall。

后续使用固定 ISA 微基准进一步区分了“时间线上落入 MFMA busy window”和“硬件允许 co-issue”:
gfx942 不提供 `SQ_VALU_MFMA_COEXEC_CYCLES`;LLVM gfx94x 将 TRANS(`v_exp_f32`/`v_rcp_f32`)和 packed FP32
add/mul 标为 never-coissue。实测 `v_pk_mul_f32`/`v_pk_add_f32` 在 gap0 额外阻塞约 20 clocks/MFMA,
gap>=8 才恢复;普通 `v_add_f32`/`v_perm_b32` 无额外 penalty。完整方法与脚本见
[`docs/mfma-valu-coissue.md`](mfma-valu-coissue.md)。因此 ATT 的 13.29% 是 issue 时间线 overlap 指标,
不能替代缺失的硬件 co-exec counter。

### co-issue 调度实验(均已回退)

| 尝试 | TFLOPS | co-issue/结论 |
|---|---:|---|
| fastest lazy baseline | **166.2** | MFMA-cycle VALU coverage 13.285% |
| `sched_barrier(0x2)`允许 VALU 跨 GEMM1 边界 | 164.6 | live range/调度变差 |
| `sched_barrier(0x402)`允许 VALU+EXP 跨边界 | 164.0 | 进一步回退 |
| branchless lazy + 显式 VMEM/MFMA/VALU/EXP groups | 150.7 | 固定分组破坏原 VMEM/LDS/MFMA 交错;VGPR 236 |
| softmax `s_setprio(1)` | 165.1 | 抢占 MFMA issue机会 |
| GEMM1 `s_setprio(1)` | 153.7 | softmax/访存阶段饥饿 |
| 奇数 workgroup `s_sleep(1..15)`一次性错相 | 166.2–166.3 | sleep15 coverage 13.310%,与 13.285% 基线等价 |

纯调度 hint 无法跨越 `GEMM1 -> softmax -> GEMM2` 的真实数据依赖;强制排序还会损失编译器已形成的 VMEM/LDS
流水。故最终代码不保留任何上述 hint/stagger,只保留原始 fastest 调度。

### 下一步改进方案

1. **已完成:GEMM1 拆分两个独立 Mq accumulator group。** 结果见 §16。拆分本身小幅提升到 166.7 TFLOPS;
  但把 `softmax(mt0)` 插进两个 GEMM1 group 之间会回退到 161.8,没有形成有效 co-issue。
2. **优先后续:真正的双 wave-specialization workgroup。** 把单 workgroup 从 4 waves 扩为 8 waves,两组 4-wave
  pipeline 用 workgroup 内条件 barrier 打开/关闭一拍相位差,类似 gfx950 dual-wave SWP。当前两个驻留 wave
  来自不同 workgroup,block 奇偶 sleep 无法稳定配对。风险是 Q/K/V/O fragment 或 LDS 翻倍、workgroup 数减半;
  必须保持每组独立 K stage,否则 barrier 会重新锁步。
3. **低风险但收益有限:减少剩余 VALU/EXP,而非继续加 scheduler hint。** correction exp 会被 LLVM 投机执行;
  side-effecting inline exp 已测 160.1 TFLOPS。可尝试针对 `v_exp_f32` 的近似多项式/分段近似,但需保持
  `rel_l2 <=0.0035` 并覆盖大 logits/长序列。每 tile 固有的 16 个 probability exp 才是主要目标。
4. **不建议:**继续调 `sched_barrier` mask、固定 group 比例、`s_setprio`、workgroup 启动 sleep、DPP reduce、
  L2/HBM。这些方向已被 PMC/ATT 或直接性能实验否定。

## 16. GEMM1 按 Mq accumulator group 拆分

### 实现

GEMM1 fragment 的逻辑结构为:

```text
frag_St: [value, m_rep=2, n_rep=2]
frag_K : [value, m_rep=2, k_rep=8]
frag_Q : [value, n_rep=2, k_rep=8]
```

其中 `n_rep` 就是 softmax 使用的两个 `mt`(每个对应 16 个 query row)。原来的完整调用:

```python
fx.gemm(mma, frag_St, frag_K, frag_Q, frag_St)
```

改为显式原子 MFMA:

```python
def gemm1_mt(mt):
    for k in range_constexpr(D // 16):
        for m in range_constexpr(BN // 16):
            acc = frag_St[None, m, mt]
            fx.mma_atom_call(
                mma,
                acc,
                frag_K[None, m, k],
                frag_Q[None, mt, k],
                acc,
            )

for mt in range_constexpr(2):
    gemm1_mt(mt)
```

这与 `ExpandGemmOpLowering` 最终发出的 `MmaAtomCall` 数完全相同,但把两个 `n/mt` accumulator group 明确分开。
最终顺序为 `mt -> k -> m`:固定一个 16-row query group,每个 K-step 在两个独立 `m` accumulator 之间交替。
MFMA 总数、VALU/EXP 数、VGPR 和数学结果均不变。

### 性能与资源

| 配置 | 拆分前 | 拆分后 |
|---|---:|---:|
| `H=8,M=N=8192` | 1653.8us / 166.2T | **1648.8us / 166.7T** |
| `H=1,M=N=40960` | 4740.0us / 181.2T | **4690.7us / 183.1T** |
| `SOFTMAX=0,H=8,M=N=8192` | 1131.0us / 243.1T | **1126.3us / 244.1T** |
| rel_l2(softmax) | 0.00316 | 0.00316 |
| VGPR / occupancy | 240 / 2 waves | 240 / 2 waves |
| LDS / spill | 16KB / 0 | 16KB / 0 |

`8192²×8 heads` 连续三次均为 1648.8us/166.7T,不是单次噪声。长序列提升更明显(+1.0%)。

### PMC/ATT 对比

| 指标 | 拆分前 | 拆分后 |
|---|---:|---:|
| instructions / wave | 63,417 | 63,161 |
| VALU / wave | 46,567 | 46,567 |
| MFMA / wave | 16,384 | 16,384 |
| wave cycles | 221,010 | **219,926(-0.49%)** |
| dependency wait | 13.60% | **13.04%** |
| issue wait | 53.35% | 53.75% |
| `ValuPipeIssueUtil` | 46.42% | 46.62% |
| `MfmaUtil` | 50.37% | 50.59% |
| ATT total stall | 48.63M / 60.4% | **46.52M / 60.2%** |
| MFMA-cycle VALU coverage | 13.285% | 13.393% |

收益主要来自 GEMM1 MFMA accumulator-chain 顺序和 dependency wait 下降,而不是显著提高 VALU/MFMA co-issue。
LDS/SMEM wait 也从 4.57M 降到 3.78M,但 barrier stall 从 1.75M 升到 2.05M,整体仍为净收益。

### 回退的交错尝试

- `GEMM1(mt0) -> softmax(mt0) -> GEMM1(mt1) -> softmax(mt1)`:精度正确、VGPR 240,但仅 161.8T。
  `mt0` softmax 的长 reduce/EXP 依赖链延迟了第二组 MFMA;机器调度器没有把 mt1 MFMA 提前到有效窗口。
- 拆分后允许 VALU+EXP 跨 `sched_barrier(0x402)`:162.7T。仍会破坏原 VMEM/MFMA 排程。
- `mt->m->k` 与 `mt->k->m` 三次中位数只差约 0.05%;最终保留已完成 PMC/ATT 验证、在两个 m accumulator
  之间按 K-step 交替的 `mt->k->m`。

## 17. 根据 co-issue profile 融合 running-sum correction

### 热点定位

最终 ISA 共有 114 条静态 `v_pk_mul_f32`,但不能按静态数量全部 scalarize。结合 source-mapped ATT:

| 源码位置 | 静态 packed MUL | 动态特征 | 处理决定 |
|---|---:|---|---|
| `l_out = l_in * corr + ts` | 2 | `exec=26,624`,stall 约 1.52M,并夹在 GEMM2 MFMA 之间 | 优先优化 |
| score scaling `S * sm_scale_log2` | 16 | 紧跟 GEMM1 尾部 | 只做 A/B 探针 |
| 条件 O-rescale / epilogue | 96 | 大多由 lazy 分支跳过或远离关键 MFMA shadow | 保持不动 |

gfx942 固定周期微基准已经证明 packed FP32 MUL 是 never-coissue,而 scalar `v_mul_f32`/`v_fma_f32`
可与 MFMA 共发。但“指令类别可共发”不等于“强制改写必然更快”:inline asm 还会改变后端可见性、寄存器分配和
机器调度。

### 失败的直接 scalar MUL

先用 LLVM inline asm 强制 `v_mul_f32` 做了两个最小实验:

| 变体 | ISA / 资源 | `H=8,M=N=8192` | 结论 |
|---|---|---:|---|
| 原拆分基线 | 114 pk-mul + 2 scalar-mul,240 VGPR | 1645.8–1648.6us / 166.7–167.0T | 基线 |
| 只 scalarize `l_in * corr` | 111 pk-mul + 8 scalar-mul,236 VGPR | 1689.4–1689.5us / 162.7T | **-2.5%,回退** |
| 只 scalarize score scaling | 98 pk-mul + 34 scalar-mul,240 VGPR | 1657.1–1658.3us / 165.8–165.9T | **-0.7%,回退** |

两者均无 spill且精度不变。第一版即使降低 VGPR仍明显变慢,说明 opaque inline asm 阻碍后端重排的损失大于
消除 packed issue block 的收益;第二版把 16 条 packed MUL 拆成 32 条 scalar MUL,issue 数增加也没有被
最后一段 GEMM1 shadow 完全吸收。因此两种 inline-asm 改写都已回退。

### 保留方案:scalar FMA 融合

running-sum 更新本来是:

```python
l_out[mt] = l_in[mt] * corr[mt] + ts
```

改为 FlyDSL 标准 math op:

```python
l_out[mt] = fx.fma(l_in[mt], corr[mt], ts)
```

这不是 opaque inline asm;LLVM 能看到 `math.fma` 并把每个 packed lane 生成为独立 `v_fma_f32`。最终 ISA 中
目标 2 条 `v_pk_mul_f32` 消失,新增 4 条 `v_fma_f32`,且它们实际穿插在 GEMM2 MFMA 之间。总资源仍为
240 VGPR、2 waves/SIMD、16KB LDS、零 spill。浮点语义从 `mul` 后 `add` 的两次舍入变为 FMA 的一次舍入,
并非位级等价;小尺寸与标准 `8192²×8 heads` 精度回归均通过,`rel_l2` 保持 0.00315–0.00316。

| 指标 | 拆分基线 | FMA correction | 变化 |
|---|---:|---:|---:|
| `H=8,M=N=8192` 三次 | 1645.8–1648.6us / 166.7–167.0T | **1612.1–1612.7us / 170.5T** | **约 +2.2%** |
| 最终复验 | - | **1611.5us / 170.6T** | - |
| rel_l2 | 0.00316 | 0.00316 | 不变 |
| VGPR / spill | 240 / 0 | 240 / 0 | 不变 |

PMC 使用 `H=8,M=N=8192`,512 workgroup × 4 waves = 2048 waves,每组最多 4 个 counter并按 dispatch
取中位数:

| 每 wave PMC | 拆分基线 | FMA correction | 变化 |
|---|---:|---:|---:|
| `SQ_INSTS` | 63,161 | **62,905** | -256 |
| `SQ_INSTS_VALU` | 46,567 | **46,311** | -256 |
| `SQ_INSTS_MFMA` | 16,384 | 16,384 | 0 |
| MFMA busy cycles/MFMA | 16.0 | 16.0 | 0 |
| F32 ADD | 9,600 | **9,088** | -512 |
| F32 MUL | 2,370 | **2,114** | -256 |
| F32 FMA | 12 | **524** | +512 |
| F32 TRANS | 4,610 | 4,610 | 0 |

每 wave 正好有 256 次 correction 更新。原版本每次用一条 packed MUL处理两个 lane,再用两条 scalar ADD;
新版本改为两条 scalar FMA,所以动态变化严格为 `MUL -256,ADD -512,FMA +512`,总 VALU 减少 256。
MFMA 数、16-cycle busy 和 TRANS 均不变,排除了少算矩阵工作或减少 EXP 的可能。收益来自同时减少依赖链指令数、
消除 hot-loop packed FP32 issue block,并保留 LLVM 对 scalar FMA 的调度自由。

## 18. FMA 版本重新 PMC 与后续机会

### 采集配置与归一修正

配置为 gfx942、`H=8,M=N=8192,D=128,BN=32,SOFTMAX=1`;非 profiler 基线稳定在
**1612.0–1612.6us / 170.5 TFLOPS / rel_l2=0.00316**。每个 PMC pass 最多 4 个 counter,
每组均得到 61 个 `attn_kernel` dispatch并取中位数。原始 CSV 与汇总保存在:

```text
/tmp/attn-fma-reprofile-{sched,inst,f32,meminst,util,l2,tcp,active,waitlds,int}/
/tmp/attn-nosm-reprofile-sched/
/tmp/attn-fma-reprofile-summary.json
```

本轮额外采集 `SQ_WAVES=2048`,确认 workload 是 512 workgroup × 4 waves。§15/§16 早期表格的
active/wait/wave-cycle **绝对值误按 512 workgroup 归一**,因此放大了 4 倍;百分比和前后比值不受影响。
本节统一按真实的 2048 waves 归一。

### 调度周期分解

四个原始 counter 完整闭合到 100%:

| 每 wave 周期 | FMA softmax | 占比 | `SOFTMAX=0` | 占比 |
|---|---:|---:|---:|---:|
| `SQ_WAVE_CYCLES` | **219,197** | 100% | 146,576 | 100% |
| `SQ_ACTIVE_INST_ANY` | 72,800 | 33.21% | 31,851 | 21.73% |
| `SQ_WAIT_ANY`(依赖等待) | 28,139 | 12.84% | 14,961 | 10.21% |
| **`SQ_WAIT_INST_ANY`(issue wait)** | **118,258** | **53.95%** | 99,758 | 68.06% |

softmax 使 wave cycles 增加 72,621(+49.5%):active +40,949、dependency wait +13,179、issue wait
+18,500。虽然 issue-wait **占比**低于无 softmax,它的绝对值仍增加,且在当前版本中继续占总周期 53.95%。
这说明主瓶颈不是单条依赖 latency,而是 2 waves/SIMD 下缺少足够的独立可发射工作。

### 指令、pipeline 与存储层次

| 每 wave PMC | 当前值 |
|---|---:|
| `SQ_INSTS` | 62,905 |
| `SQ_INSTS_MFMA` | 16,384 |
| `SQ_INSTS_VALU` | 46,311 |
| non-MFMA VALU | **29,927** |
| MFMA busy cycles/MFMA | 16.0 |
| F32 ADD / MUL / FMA / TRANS | 9,088 / 2,114 / 524 / **4,610** |
| INT32 / branch | **4,677** / 641 |
| LDS / VMEM read / VMEM write / SMEM | 4,618 / 2,572 / 16 / 4 |
| `ValuPipeIssueUtil` / `MfmaUtil` | 46.60% / 50.78% |
| `OccupancyPercent` | 21.23% |

`SQ_ACTIVE_INST_VALU/LDS/VMEM` 分别为 60,141 / 7,728 / 2,588 cycles/wave。VALU 明显大于数据搬运,
与 non-MFMA VALU 数量一致。LDS 专项 counter进一步排除了 LDS 主瓶颈:

| LDS wait | 周期/wave | 占总周期 | 占依赖等待 |
|---|---:|---:|---:|
| `SQ_WAIT_INST_LDS` | 2,097 | **0.96%** | 7.45% |

L2/TCP 结果也不支持继续优先调 HBM/L2:

| 指标 | 结果 |
|---|---:|
| `TCC_HIT/(HIT+MISS)` | **89.25%** |
| `TCC_MISS/TCC_READ` | 11.08% |
| L2 write占 read+write | 3.01% |
| TCP→TCC read request / TCP cache access | 0.194 |
| TCP pending-stall cycles / cache access | 0.285 |

cache/TCP pass 对 wall time 的 profiler 扰动比 SQ pass 更大,上述值只用于同组比例,不比较不同 pass 的绝对时间。

### 按 PMC 验证的局部优化(均回退)

1. **Q 预缩放:**把 `sm_scale_log2` 折入只加载一次的 BF16 `frag_Q`,希望消除循环内约 2,048 条
  F32 MUL/wave。额外 BF16 量化使小尺寸 `rel_l2=0.00356>0.0035`,立即回退。
2. **score MUL+SUB → vector FMA:**静态 `v_pk_mul_f32 112→98`,`v_sub_f32 36→0`,但 LLVM 生成
  16 条 never-coissue `v_pk_fma_f32`;三次仅 **161.7T**,回退。
3. **延后 score scaling:**先归约未缩放 max,再执行原 packed MUL,试图让 MUL 离开 GEMM1 shadow。
  它拉长 GEMM1→EXP 关键路径并增加 max 指令,三次仅 **158.9–159.0T**,回退。
4. **BF16 直接截断:**希望消除当前 helper 的逐值 `+0x8000`;小尺寸 `rel_l2=0.00648`,回退。

gfx942 没有 `v_cvt_pk_bf16_f32`/`v_cvt_bf16_f32`:ROCm assembler 会拒绝,LLVM 的 gfx942
`bf16-conversions.ll` 也使用软件序列。当前 round-half-up helper 已被后端优化为每值一条 `v_add_u32 0x8000`
+每对一条 `v_perm_b32`,没有额外 shift;不能套用 gfx950 的硬件 packed conversion。

### 后续机会排序

1. **结构性增加独立 wave 工作(优先):**当前 53.95% issue wait说明继续重排同一 softmax 依赖链收益有限。
  最值得验证的是 8-wave workgroup 内两套 4-wave pipeline显式错相,让一组 softmax/EXP时另一组发 MFMA/VMEM;
  这比跨 workgroup `s_sleep` 更能保证同一 CU 内配对。代价是 K stage/LDS ownership和 barrier必须完全独立。
2. **VGPR 降档需要大改,不是微调:**LLVM gfx942 模型为 512 VGPR/SIMD、8-VGPR分配粒度。当前 240
  VGPR静态 occupancy 为 2 waves/SIMD;3 waves要求 **≤168 VGPR**,即至少减少 72。236/192/176 VGPR仍只有
  2 waves。可研究每 wave 只负责 16 query rows(减半 `frag_Q/frag_St/frag_O`)的 8-wave tile,但 K/V广播
  和访存重复可能抵消收益,必须先做资源/流量模型。
3. **减少 probability EXP:**`v_exp_f32` 为 4,610/wave且 TRANS不能与 MFMA共发。correction EXP的条件执行
  已因调度损失回退;剩余可研究有误差上界的 `exp2` 分段近似,门槛仍为 `rel_l2<=0.0035`,并需覆盖全1、
  单调 logits和长序列。多项式若需要过多 scalar VALU,可能只把 TRANS瓶颈换成更长 issue 链。
4. **不再优先:**LDS swizzle/prefetch、L2/HBM、packed score FMA、Q BF16预缩放、BF16 truncate、
  scheduler mask/setprio/sleep。PMC 或直接 A/B 已否定这些局部方向。

## 19. softmax(mt0)/GEMM1(mt1) 细粒度交织验证

### 依赖结论

用户提出的依赖判断成立:

```text
GEMM1(mt0) -> softmax(mt0) -> GEMM2(mt0)
GEMM1(mt1) -> softmax(mt1) -> GEMM2(mt1)
```

两条链之间没有交叉 RAW依赖,因此理论上存在两个同wave窗口:

1. `softmax(mt0) ↔ GEMM1(mt1)`;
2. `softmax(mt1) ↔ GEMM2(mt0)`。

但“数据独立”只说明允许重排,不保证能免费共发。gfx942的
`v_mfma_f32_16x16x16_bf16` shadow为16 cycles;普通scalar VALU可共发,而packed FP32、TRANS/EXP和MFMA
本身是never-coissue。line 303的lazy条件在最终ISA中被if-convert为`v_cmp_gt_f32 + v_cndmask_b32`,没有形成
控制流basic-block边界;假设去掉该条件只会删除比较/选择和投机correction EXP,不会解除mt0/mt1之间的依赖。

### 单mt指令预算与理论遮盖上限

当前每个mt有16条GEMM1 MFMA,softmax主体的静态工作量约为:

| 阶段 | 典型指令 | 可与MFMA共发 |
|---|---|---|
| score scale | 8条`v_pk_mul_f32` | **否** |
| local max | 1 `v_max_f32` + 3 `v_max3_f32` | 是 |
| wave max | 2 DS shuffle/read + 4 scalar max | scalar max可,DS另管线 |
| lazy选择 | cmp/cndmask;无条件假设下可删除 | 是 |
| center | 8条`v_sub_f32` | 是 |
| probability | 8条`v_exp_f32` | **否** |
| local/wave sum | 7 scalar add + 2 DS shuffle/read + 4 scalar add | scalar add可 |
| running sum | 2条scalar FMA | 是 |
| correction/O-rescale | 1 correction EXP + 条件packed MUL | **否** |

在无line303条件的理想模型里,可作为MFMA shadow候选的scalar VALU约为
`4 local-max + 4 wave-max + 8 center + 11 sum + 2 running-FMA = 29条/mt`;另有4条DS归约。
8条packed scale、8条probability EXP以及correction/O-rescale不能由MFMA shadow遮盖。

为测量容量,额外运行了固定32-cycle period、每条MFMA后0–4条独立FMAC的gfx942微基准:

| scalar VALU/MFMA | 增量 cycles/MFMA | 纯串行成本 | 净遮盖 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 6.0 | 4 | 该固定period下反而+2 |
| 2 | 11.0 | 8 | 该固定period下反而+3 |
| 3 | 16.0 | 12 | 该固定period下反而+4 |
| 4 | 20.0 | 16 | 该固定period下反而+4 |

因此不能把16-cycle shadow简单解释为“每条MFMA免费塞4条VALU”。在该一wave probe中,即使普通VALU被判为
`coissue-capable`,加入它仍增加固定period总周期;收益来自相比never-coissue指令更少的阻塞,而不是零成本。
16条mt1 MFMA最多容纳约32条实用scalar候选(每MFMA 2条后边际收益很小),数量上刚好覆盖29条,
但理论时间收益上限很低,而且依赖链只能分阶段释放这些候选。

### 窗口1:`softmax(mt0) ↔ GEMM1(mt1)`

共测试两版:

1. **自动sched-group:**源码为`GEMM1(mt0)->softmax_prepare0->GEMM1(mt1)`,在后16条MFMA声明
   `2 MFMA + 2 VALU + 2 MFMA`。最终ISA没有把mt0 max拉入mt1区域,只重排了原VMEM/MFMA;VGPR 240→232,
   说明完整vector scale/DS依赖链没有给scheduler足够可选scalar节点。
2. **显式staged:**把mt0 local max、shuffle16/max、shuffle32/max拆开,在每阶段之间发4条mt1 MFMA,
   并使用LLVM官方alternating group pattern。最终ISA确认在16条mt1 MFMA中实际插入12条scalar max和2条DS
   shuffle,EXP/packed scale未进入shadow;VGPR=240、无spill、`rel_l2=0.00316`。

真正交织已经发生,但标准workload三次只有 **157.8–157.9T**,相对170.5T基线回退7.4%。原因是12条
scalar VALU的潜在遮盖远小于拆断连续mt1 MFMA链、增加wait和延长GEMM1→EXP关键路径的损失。

### 窗口2:`softmax(mt1) ↔ GEMM2(mt0)`

先建立只拆GEMM2、不交织的直接基线:

| 版本 | VGPR | 性能 |
|---|---:|---:|
| 原完整GEMM2 | 240 | **170.5T** |
| `gemm2_mt(0);gemm2_mt(1)` | 228 | 165.2–165.3T |

仅拆分GEMM2就损失约3.1%,说明原`fx.gemm`的atom顺序和机器调度已有重要ILP。随后把GEMM2(mt0)的16条MFMA
按`4/2/2/4/4`拆分,分别与softmax1的local max、两级wave max、center和sum/running-FMA阶段交织;
line303条件保留,EXP与packed指令不进入MFMA组。最终精度正确、VGPR降到222、无spill,但三次仅
**157.2–157.3T**。

同counter PMC证明调度确实减少了issue空洞,但转换成更大的依赖等待:

| cycles/wave | 170.5T基线 | 窗口2细粒度 | 变化 |
|---|---:|---:|---:|
| total | 219,197 | 232,827 | +13,630(+6.2%) |
| active | 72,800 | 74,082 | +1,282 |
| dependency wait | 28,139 | **44,023** | **+15,884(+56.5%)** |
| issue wait | 118,258 | **114,719** | -3,539(-3.0%) |

即:共发机会真实存在,issue wait也确实下降,但拆MFMA accumulator链带来的dependency wait增长更大。

### LLVM attention IGLP策略

还测试了无需手拆GEMM链的LLVM `rocdl.iglp.opt`:

| 策略 | 方法 | EXP到前一MFMA中位距离 | VGPR | 性能 |
|---|---|---:|---:|---:|
| baseline | 原scheduler | 122条指令 | 240 | **170.5T** |
| IGLP 3 | 简单`TRANS↔MFMA`一对一 | 49(最小2) | **280** | 105.7T |
| IGLP 2 | attention TRANS/MFMA+前驱分析 | 51(最小2) | **262** | 112.5–112.6T |

两种策略都真正把EXP靠近MFMA,但扩张live range并越过256-VGPR阈值,occupancy从2 waves/SIMD降到1,
因此严重回退。IGLP不能与同region的`sched_barrier/group`混用,实验中已关闭原hint后单独测试。

### 最终结论

- **合理性:**依赖分析正确,两个交织窗口都存在。
- **可遮盖量:**数量上最多约29条scalar VALU/mt可成为候选,但packed scale、8条EXP和条件packed rescale不能
  遮盖;微基准显示scalar插入也不是零成本。
- **实测:**手工两窗口和LLVM IGLP都回退。前者破坏MFMA链ILP并增加dependency wait,后者扩大live range导致
  occupancy降档。
- **softmax(mt1)的合适机会:**仍是GEMM2(mt0),但必须在**不拆MFMA accumulator链且不扩张到>256 VGPR**的
  条件下由更强的后端全局scheduler实现;当前FlyDSL/LLVM hint均未满足。更现实的方向仍是跨wave/双pipeline
  隐藏softmax,让一个wave的完整MFMA链与另一个wave的softmax并行。





