# gfx950 4-Wave MXFP8/Hybrid 优化状态（2026-08-30）

## 当前状态

目标 kernel：`test_mxfp8_gemm_4w.py`，固定 `M=N=8192`，`TILE_M=TILE_N=256`、`BLOCK_K=128`、4 waves/CTA。

今天保留的优化：

1. **Scale direct G2R（默认开启）**
   - `SCALE_G2R=1` 时，A/B 的 packed E8M0 scale 通过 `buffer_load_dword` 直接从 global memory 进入 VGPR，并采用两级寄存器 ping-pong。
   - MXFP8 A8W8 和 Hybrid（A=MXFP8、B=MXFP4）的 A/B scale 都不经过 LDS。
   - 可设置 `SCALE_G2R=0` 回退到原 global -> LDS -> VGPR scale 路径。
   - 主 A/B 矩阵数据仍保持 global -> LDS -> VGPR，不受此优化影响。

2. **Scaled 专用调度（默认开启）**
   - `SCALE_SCHED_LATE=1`。
   - 默认位置：`SCALE_DSRD_POS=13`、`SCALE_VMEM_POS=7`。
   - G2R 路径的 scheduler 不再计入额外 scale `ds_read_b32`。

3. **Scale LDS 无冲突布局作为 fallback 保留**
   - 每个 wave 使用独立的 64-dword LDS 区域，lane 覆盖全部 64 个 LDS banks。
   - 已验证 `SQ_LDS_BANK_CONFLICT: 512 -> 0`。
   - 注意：默认 G2R 路径不会读写这些 scale LDS 字段，但当前 `LDS` struct 仍为 fallback 分配它们；后续可考虑按编译路径移除这部分 LDS 占用。

4. **Hybrid MXFP4 B 数据搬运**
   - B 数据保持 global -> LDS -> VGPR。
   - 每个 2048-element block 使用 full-wave DMA，B phase 的数据 VMEM 数由四次 subgroup load 降为两次 full-wave load。

## Layout 设计

### 公共分块与 operand 方向

- `TILE_M=TILE_N=256`，CTA 内为 4 waves；每个 quadrant 使用 `BLOCK_M=BLOCK_N=BLOCK_K=128`。
- 逻辑 A 进入 MFMA operand B，逻辑 B 进入 MFMA operand A。
- A/B 数据采用 LDS ping-pong；每个时刻只有一份 active register fragment。
- non-scale MXFP8 使用 `k_perm=(32,4):(1,32)`；scaled MXFP8 与 Hybrid 使用 `k_perm=((16,2),4):((1,64),16)`。因此 scaled MXFP8/Hybrid A 与纯 non-scale MXFP8 的 padding stride 不完全相同。

### MXFP8 padding

A 和 B 都使用相同的 128-bit DMA mapping：

```python
dma_tv = ((8, 8, 4), 16):((16 * 32, 1, 8), 32)
dma_tile = (32, 128)
```

每个线程在四个 copy rounds 中各搬运一个 16-byte chunk。LDS write/read layout 为：

```python
wr = ((8, 2, 8), 128):((128, group8, 2112), 1)
rd = ((2, 8, 8), (32, 4)):((group8, 2112, 128), (1, 32))
```

其中：

| 路径 | `group8` | 设计 |
|---|---:|---|
| 纯 non-scale MXFP8 | 1040 | 前 8 rows 的 1024 elements 后偏移 16；16-row group stride 为 2112 |
| scaled MXFP8 | 1056 | 每 8 rows 使用 1056-element stride，匹配 scaled MFMA K permutation |
| Hybrid A | 1056 | 与 scaled MXFP8 A 相同 |

`rd` 对 `wr` 做与 MFMA lane/K 消费顺序匹配的 permutation；当前 counter 中 MXFP8 padding 的 A/B 均为 0 bank conflict。

### Hybrid MXFP4 B padding

Hybrid 的 A 使用上一节 `group8=1056` 的 FP8 padding；B 使用独立设计：

```python
b_group16 = 16 * 128 + 64  # 2112 logical FP4 elements
wr_b = rd_b = ((16, 8), 128):((128, 2112), 1)
```

- 每个 16-row block 含 2048 个有效 FP4 elements，之后 padding 64 个 logical FP4 elements（32 bytes）。
- 每个 wave 一条 full-wave raw DMA 搬运一个完整的 2048-element block；两个 copy rounds 覆盖 8 个 block/128 rows。
- lane mapping 为 `row=chunk*16+lane//4`、`col_byte=(lane%4)*16`。
- 该设计以 `2048` 个连续 elements 的整块 DMA 为优先，接受少量 B bank conflict；64/128/256-element padding 的 conflict 数相同，最终选择 64 以降低 LDS 占用并获得最佳性能。

### Swizzle：不同数据路径使用不同设计

Swizzle 不是 A/B 或 MXFP8/Hybrid 共用一套参数。当前实现如下：

| 路径 | FlyDSL spec | LDS write | LDS read/S2R |
|---|---|---|---|
| MXFP8 A/B | `S<3,4,4>` | ordered `(128,128):(128,1)` | composed swizzle + 专用 conflict-free S2R TV |
| Hybrid A（FP8） | `S<3,4,3>` | ordered `(128,128):(128,1)` | composed swizzle，经 `make_tiled_copy_B` 读取 |
| Hybrid B（FP4） | `S<1,5,2>` | ordered `(128,128):(128,1)` | 一位 16-byte slot XOR + 显式 S2R |

MXFP8 的专用 S2R TV 为：

```python
copy_a_tv = ((16,4,2,2),(16,2)):((1,512,0,16),(32,2048))
copy_b_tv = ((16,4,2,2),(16,2)):((1,512,16,0),(32,2048))
operand_tile = (32,128)
```

其中 A/B 的两个 wave strides 交换，是因为逻辑 A/B 分别进入 MFMA operand B/A。

Hybrid B 不能直接复用 MXFP8 的 `S<3,4,4>`。它按 16-byte slot 工作：

```python
physical_slot = tid + copy_round * 256
logical_slot = physical_slot ^ ((physical_slot >> 3) & 1)
```

G2S 根据 `logical_slot` 找到逻辑 row/column，但写入线性的 `physical_slot*16` LDS 地址；S2R 使用同一 XOR（byte-address bit 4 由 bit 7 控制）恢复 MFMA 需要的 FP4 数据。该 `S<1,5,2>` 设计与 Hybrid A 的 `S<3,4,3>` 独立，两者共同实现 Hybrid swizzle 0 bank conflict。

### Swizzle 的 global-load 路径

- 当 K 是 2 的幂时，在 global source tensor 上组合相同 swizzle，并用 `stride_shift=log2(K)-log2(128)` 调整 row stride，然后通过 tiled `fx.copy` 写入 ordered LDS。
- 当 K 不是 2 的幂时，不能依赖全局 layout 的 bit-shift 等价关系；当前实现改用 raw G2S，在每个 128-K tile 内反解 logical/physical slot，并用实际 K 计算 global row stride。
- Hybrid FP4 B 始终使用其专用 raw G2S；因此 `K=10240/12288/14336` 等非 2 次幂 K 也支持 swizzle。

### Scale layout

Host 将 E8M0 scale 从 `[rows, K/32]` 重排为 `[K/128, rows/128, 32, 4]` 的连续存储，并以四个 E8M0 bytes 为一个 `i32`。kernel 中对应的 i32 view 为：

```python
shape   = ((32, 8), (rows/128, K/128))
strides = ((1, rows/4), (32, rows))
```

默认 direct G2R 中，每 lane 加载一个 packed scale dword：

```python
scale_row   = lane % 16 + wave_half * 16
scale_group = lane // 16
dword_offset = kk*rows + scale_group*(rows/4) + row_tile*32 + scale_row
```

其中 A 使用 `wave_half=wave//2`，B 使用 `wave_half=wave%2`，与 2x2 wave/quadrant 方向匹配。两个 K tiles 使用两套 register fragments ping-pong；MXFP8 和 Hybrid 的 A/B scale 均复用此布局。

`SCALE_G2R=0` fallback 才会使用 scale LDS：每个 scale buffer 为 256 dwords，每 wave 独占连续 64 dwords，使 64 lanes 分别访问 64 banks。

## LDS Bank Conflict 状态

使用当前源码重新采集 counter，测试口径为 `M=N=K=512`、non-scale，并只统计 `gemm_kernel_0`。结果确认：**只有 Hybrid MXFP4 padding 的 B 数据路径存在 bank conflict**。

| 数据类型 | LDS 布局 | `SQ_LDS_BANK_CONFLICT` | `SQ_LDS_ADDR_CONFLICT` | `SQ_LDS_IDX_ACTIVE` | 状态 |
|---|---|---:|---:|---:|---|
| MXFP8 | Padding（默认） | 0 | 0 | 8192 | 无冲突 |
| MXFP8 | Swizzle `S<3,4,4>` | 0 | 0 | 8192 | 无冲突 |
| Hybrid MXFP4 | Padding（默认，B 每 2048 elements 后 pad 64） | 2048 | 0 | 8192 | 仅 B 数据有 bank conflict |
| Hybrid MXFP4 | Swizzle（A `S<3,4,3>`，B `S<1,5,2>`） | 0 | 0 | 6144 | 无冲突 |

说明：

- MXFP8 padding 的 A/B 数据均无 bank conflict。
- Hybrid padding 中，A 的 padding 路径无冲突；counter 中的 `2048` 来自 MXFP4 B。该布局有意保留部分冲突，以保证每个 2048-element B block 可通过一次 full-wave DMA 搬入 LDS。64/128/256-element padding 的冲突数相同，最终选择 64-element padding，因为 LDS 占用最小且 `8192^3` 性能最好。
- Hybrid swizzle 通过 B 的 16-byte slot XOR 和匹配的显式 S2R 消除 B conflict。
- 默认 `SCALE_G2R=1` 时，MXFP8/Hybrid 的 A/B scale 不访问 LDS，因此上述结果描述的是主 A/B 数据路径。
- `SCALE_G2R=0` fallback 中的 scale LDS 使用 wave-private 64-dword 区域，scale 路径自身也已验证为 0 bank conflict；Hybrid padding 的整体 kernel 仍会保留 B 数据产生的 conflict。

## 根因与效果

硬件 profile 表明，原 scaled 路径的主要损失不是 MFMA 吞吐或 occupancy，而是额外 scale VMEM、`ds_read_b32` 及其 wait/issue 放大：

- scaled/non-scale 的 MFMA 指令数及 MFMA busy cycles 相同。
- direct G2R 消除了 scale LDS round trip。
- K=512 的 G2R ISA：
  - `256` 条 MFMA
  - `16` 条 scale `buffer_load_dword`
  - `0` 条 scale `ds_read_b32`
  - `0` 条 scratch load/store
  - `VGPR=512`、`SGPR=96`、`accum_offset=256`

在 `K=8192` 的 MXFP8 正式测试中，scaled best 从约 `391.3 us` 降至 `380.8 us`，相对 non-scale 的 latency gap 为 `4.85%`。

## Accuracy

以下 20 组完整 `8192 x 8192` 输出均通过 `calc_diff <= 1e-5`：

- 类型：MXFP8、Hybrid MXFP4
- 模式：non-scale、with-scale
- K：`8192`、`10240`、`12288`、`14336`、`16384`

观测到的最大 `calc_diff`：

- MXFP8：约 `2.59e-08`
- Hybrid MXFP4 with-scale：约 `7.58e-09`

原请求中的 `10236/12280/14324/16368` 不满足 kernel 的 `K % 256 == 0` 约束，因此测试使用相邻合法值 `10240/12288/14336/16384`。

## Performance

测试口径：MI355X/gfx950，`M=N=8192`，50 份数据轮转、50 次 event 计时，取 best；with-scale 默认使用 direct G2R。

### MXFP8

| K | Non-scale us | Non-scale TFLOPS | With-scale us | With-scale TFLOPS | Latency gap |
|---:|---:|---:|---:|---:|---:|
| 8192 | 363.2 | 3027.60 | 380.8 | 2887.35 | 4.85% |
| 10240 | 447.0 | 3074.39 | 474.3 | 2897.57 | 6.11% |
| 12288 | 529.5 | 3114.85 | 559.2 | 2949.10 | 5.61% |
| 14336 | 598.4 | 3215.45 | 646.8 | 2974.84 | 8.09% |
| 16384 | 683.0 | 3219.63 | 729.9 | 3012.83 | 6.87% |

### Hybrid MXFP4

| K | Non-scale us | Non-scale TFLOPS | With-scale us | With-scale TFLOPS | Latency gap |
|---:|---:|---:|---:|---:|---:|
| 8192 | 344.6 | 3191.03 | 372.9 | 2948.67 | 8.21% |
| 10240 | 425.8 | 3227.75 | 462.7 | 2970.21 | 8.67% |
| 12288 | 502.1 | 3284.58 | 545.2 | 3024.82 | 8.58% |
| 14336 | 580.6 | 3314.04 | 633.9 | 3035.28 | 9.18% |
| 16384 | 655.7 | 3353.57 | 715.7 | 3072.43 | 9.15% |

## 后续方向

- Hybrid with-scale 的 gap 仍为约 `8.2%-9.2%`，高于 MXFP8 的约 `4.9%-8.1%`，应优先 profile Hybrid 的 VMEM wait、MXFP4 B S2R 和 scale load co-execution。
- 根据 `scale_g2r` 编译常量裁剪 scale LDS 字段及 LDS fallback setup，确认是否改善资源占用或调度。
- 当前 VGPR 已到 `512`，继续增加 scale prefetch depth 前必须检查 spill 与 occupancy。
