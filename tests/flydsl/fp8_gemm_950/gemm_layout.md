# 4 wave bf16(without preshuffleB):
```
TILE_M/TILE_N=256, BK=64.
slicing之后，BM/BN=128, BK=64
```
### AC A/B tile copy:
```
BK = 64, 8x8 thread mapping:
32 rows, 64 columns,
```
### AC A/B tv layout:
```
((8, 8, 4), elements_per_128b),
((elements_per_128b x 32, 1, 8), 32)
```
### AC Padding read layout:
```
sub A/B natural layout: (128, 64), (K, 1),主要针对m分组，128个被均分为8组，每一组是一个row lane
((16, 8), 64, K//64), ((K, 16xK), 1, 64), 根据访问的thread mapping顺序permute为：
((8, 16), 64, K//64), ((16xK， K), 1, 64)
```
### AC padding write LDS layout:
```
LDS layout:(128, 64), (64, 1) without padding, 每8行padding 16个元素，
LDS layout : ((8, 16), 64), ((64, 8x64+16), 1)
```
### AC swizzle read layout:
```
bA_t layout (128, 64, K//64), (K, 1, 64)
_num_shift = K.bit_length() - 1 - 3
_sw = fx.static(fx.SwizzleType.get(3, 3, _num_shift))
bA_t = fx.Tensor(fx.make_view(fx.get_iter(bA_t), fx.make_composed_layout(_sw, fx.get_layout(bA_t))))
```
### AC swizzle write layout:
```
swizzle已经在AC的source上apply,这里不需要变化。
LDS layout:(128, 64), (64, 1), 
```
### S2R read LDS layout padding:
```
在AC 写LDSlayout 基础上做一个permute.
LDS layout : ((8, 16), 64), ((64, 8x64+16), 1) permute to:
((16, 8), 64), ((8x64+16, 64), 1)
``` 
### S2R read LDS layout swizzle:
LDS layout:(128, 64), (64, 1) apply swizzle.
lds_layout_rd = fx.make_composed_layout(
    fx.static(fx.SwizzleType.get(3, 3, 3)),
    (128, 64), (64, 1),
)

# B preshuffle bf16:
参考 后面的 Preshuffle Layout bf16


# 4wave FP8(without preshuffleB):
```
TILE_M/TILE_N=256, BK=128.
slicing之后，BM/BN=128, BK=128
```
### AC A/B tile copy:
```
BK = 128, 8x8 thread mapping:
32 rows, 128 columns,
```
### AC A/B tv layout:
```
((8, 8, 4), elements_per_128b),
((elements_per_128b x 32, 1, 8), 32)
```
### AC Padding read layout:
```
BK是128
sub A/B natural layout: (128, 128), (K, 1),主要针对m分组，128=16x8,
((16, 8), BK, K//BK), ((K, 16xK), 1, BK), 根据访问的thread mapping顺序permute为：
((8, 16), BK, K//BK), ((16xK， K), 1, BK)
```
### AC padding write LDS layout:
```
LDS layout:(128, 128), (128) without padding, 每8行padding 16个元素，每16行再padding 32个元素
BK=128
LDS layout : ((8, 2，8), BK), ((BK, 8xBK+16，  （8xBK+16)x2+32), 1)
```
### AC swizzle read layout:
```
_nb = 4  # fp8 128-bit = 16 elem = 2^4
_swg = fx.static(fx.SwizzleType.get(3, _nb, K.bit_length() - 1 - _nb))
```
### AC swizzle write layout:
```
swizzle已经在AC的source上apply,这里不需要变化。
LDS layout:(128, 128), (128, 1)
```
### S2R read LDS layout padding:
```
在AC 写LDSlayout 基础上做一个permute.
LDS layout : LDS layout : ((8, 2，8), BK), ((BK, 8xBK+16，  (8xBK+16)x2+32), 1)
permute to:
 ((2，8, 8), BK), ((8xBK+16, BK, (8xBK+16)x2+32), 1)
 ```
### S2R read LDS layout swizzle:
```
swizzle已经在AC的source上apply,这里不需要变化。
LDS layout:(128, 128), (128, 1),
```
## AC A/B(preshuffle):
```
_sw = fx.static(fx.SwizzleType.get(3, 4, BLOCK_K.bit_length() - 1 - 4))
```
# 8wave FP8(without preshuffleB):

```
8 wave需要更改的是tile copy和thread mapping,
_a_dma_tv = fx.make_layout(
    ((8, 8, 8), elements_per_128b),
    ((elements_per_128b * 64, 1, 8), 64),
)
fx.make_tile(64, BLOCK_K))

layout 相关的可以与mxfp8 4 wave一致。
```


# Preshuffle Layout bf16

## 基本Layout

preshuffle B(N, K) layout is

```
(N//16, K // 32, 4k, 16n, 8k)
(K*16,  512,     128, 8,   1)
```

假设 `BN=128`, `BK = 64` for bf16 type。

## DMA Copy Layout

每一次DMA需要copy `[BN, BK]` data into LDS，这些会均分给256个线程，每个线程copy 8个元素。我们看看 `[BN, BK]` 这些元素在layout中的stride

```
(N//16, K // 32, 4k, 16n, 8k) ->
(N//BN, BN // 16, K // BK, BK//32, 4k, 16n, 8k) -> 
(N//128, 8n, K//64, 2k, 4k, 16n, 8k)  合并2k x4k ->

(N//128, 8n, K//64, 8k, 16n, 8k), strides:
(128*K,  16*K,  1024, 128, 8, 1)
```

这个stride我们先放在这里，后面再描述layout的时候，需要根据对应的mode做一下变化。

## Copy Tile

现在需要考虑的是的是copy tile，copy tile对应的大小是，4个wave，256*8 = 2048个element。尽量的连续copy。

以 `[128n, 64k]` 作为一次fx.copy到DMA里的单位，在 `(N//128, 8n, K//64, 8k, 16n, 8k)`, slice出 `[BN, BK]` layout:

```
(8n, 8k, 16n, 8k), (16*K, 128, 8, 1)
```

tilecopy的尺寸是只256个thread copy的n, k的值，可以看出有128 thread可以在物理连续上copy `(4k, 16n, 8k)`，其余的128 thread需要在8n上分。

```
(8n, 8k, 16n, 8k) -> (4n, 2n, 8k, 16n, 8k)
256个thread copy 的tile大小是 (2n, 8k, 16n, 8k) 即 (32n, 64k)
```

### Tile Layout

一个tile的layout是：

```
(4n,  2n,  8k, 16n, 8k):
(32*K,16*K, 128, 8, 1)
```

### Thread Value Layout

这个tile copy的thread value layout，一共有256个thread，按照尽量连续copy，我们先来排thread, value，然后在根据是在M, K的sub mode维度计算stride。

256个threads, 8个value，按照上面的layout，连续thread访问来开始排放：

```
(256, 8) -> ((128, 2), 8) -> ((16, 8, 2), 8value)  这里对应的 k, n mode

((16,   8,  2), 8value)，对应的mode:
((16n,  8k, 2n), 8k)
```

根据mode就可以计算出stride，这个stride是N major，tile尺寸是 `(32n, 64k)`，所以K维度的stride是32：

```
((16,   8,   2),  8value) 
((16n,  8k,  2n),  8k) stride是：
((1,   8*32, 16), 32)
```

所以tile thread value layout 是 `((16, 8, 2), 8):((1, 256, 16), 32)`, tile is `(32, 64)`

## B Layout

现在需要描述一下sub B tensor的layout，之前的这个B preshuffle之后的layout分解：

```
(N//128, 8n,   K//64, 8k, 16n, 8k), strides:
(128*K,  16*K,  1024, 128, 8, 1)
```

subB tensor 是 `(BN, BK, Kiter) -> (128n, 64K, K//64)`

### BN维度分解

分解BN维度，tile Size是32n，是否凑出32？好像不需要：

```
128n -> (16, 8):(8， 16*K)
```

### BK维度分解

分解BK维度：

```
64k -> (8, 8):(1, 128)
```

### Kiter维度

```
(K//64) : (1024)

((16, 8), (8, 8), K // 64):((8， 16*K), (1, 128), 1024)
```

## LDS Layout

LDS layout可以根据 `[BN, BK]` 在subtensor中的layout改动。

### SubB, [BN, BK]

```
(8n,  8k, 16n, 8k), strides:(16*K, 128, 8, 1)
```

8n mode stride只需要改成连续就可以了，`(BN, BK)` 在LDS中连续，所以：

```
LDS (8n, 8k, 16n, 8k), (1024, 128, 8, 1)

按照 (N, K) 就是 (16, 8), (8, 8): (8, 1024), (1, 128)
```



# MXFP4 scale layout in aiter 
[sm, g]  g=k//32
scale = scale.view(sm // 32, 2SM, 16sm, g // 8, 2G, 4g)
scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
(sm//32, g//8, 4g, 16sm, 2G, 2SM)

16x128//32 //64 = 1 , e8m0

f per_1x32_mx_quant_hip(
- ``dtypes.fp8_e8m0``: e8m0 byte scale ``(m, ceil(n/32))``
(or padded ``(pad256(m), pad8(ceil(n/32)))`` when
``shuffle=True``)

scale的对齐：

如果shuffle = False, 只会把k 基于group size(32)向上对其。
如果shuffle = True, 目前会把m/n 与256 对其 ， K会256对其。

shuffle等于True, E8M0 scale 会以 [sm//32, g//8, 4g, 16sm, 2G, 2SM]
2G is specially for MXFP4, BK = 256, MFMA_16_16_128_FP4, 2 BK instruction.
```

scale layout  [m, k//32]->[m, groups],

[m//32, 2M, 16m, g//8, 2G, 4g] permute

[m//32, g//8, 4g, 16m, 2G, 2M] 

```
# MXFP8 scale
##  padding:
假设M/N已经padding到256的的倍数, permute the scale_a/scale_b

## permute scale:
from [M, K//32] to coascling access, 128 bytes contineous for coascling access.

```
A scale [M, K//32] -> [M ,  G] -> [M//128, 4M, 32m, G]  tranpose to (G, M//128, 32m , 4M) : permute(3, 0, 2, 1)

A scale  layout after permute: ((4M, 32m, M//128), G), ((1, 4, 128), M)
```

## async copy to LDS
从global memory copy到 LDS的过程，128m 内部的permute对外可以不可见，所以这里直接把permute之后的 sA tensor 看作(M, G), （1， M)
sA view as: (M, G), （1， M）
Partition by : (128, 4) -> 

sA view as (M, G), (1, M), uint8
bsA_t = fx.flat_divide(sA, (128, 4))[None, None, bid_x * 2 + 0, none] shape  
bsA_t shape (128, 4, 1, G//4)

### limitation: read scale from global memory related, each lane 16 bit has some limitation.
实际需要的layout:
bsA_t read layout ((128, 4,  G//4), (1, M, 4*M))
tv_layout = ((64, 4), 2), ((2, 128), 1)


每条lane 读取 128*4//256=2 bytes. 128个bytes是连续的
问题：BUFFER_LOAD_{ubyte, sbyte, ushort, sshort, dword, dwordX3, dwordX4, format_x}. , 
     但type ushort的时候，存入的LDS ushort 会被存成dword.

### g2s copy WA
WA: 多读数据scale into LDS, [128, 4] -> [128, 8], 下一个BK, 但是不使用。

```
async_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
tilemn = (128, 8), m=128, n = 8
```

### LDS allocation
LDS buffer size for A/B 128*8 bytes = 1kB  ，有用的只有512 bytes.
总共需要 8KB, 8个LDS buffer entry for A, B scale ping-pong

### g2s: bsA_t read 
```
tilemn = (128, 8), m=128, n = 8
bsA_t layout view:  ((128, 4,  G//4), (1, M, 4*M)）  ,u8
read_tv_layout = ((32, 8), 4), ((4, 128), 1) , 每条lane读4个m, 32条lane128个m, 
```
### g2s: LDS write layout, 顺序排，理论上后一半的LDS是无用数据

```
LDS write layout: ((128,  8), (1, 128)） u8
tilemn = (128, 8), m=128, n = 8
wrlds_tv_layout = ((256), 4), (4, 1)
```

## s2r:

### LDS read layout, LDS 实际的物理排布， ( 8g, 32m,4M) U8 -> (( 8g, 32m)) int32 as ， 
我们只需要前一半的lds（4g, 32m）, LDS tensor view是原来的一半了。

```
i32_iter = fx.recast_iter(i32, fx.get_iter(lds_wr))
lds_rd = fx.Tensor(fx.make_view(i32_iter, fx.make_layout((32, 4), (1, 32))))

bsA_t LDS read layout: ((32,  4), (1, 32)） u32
tilemn = (32, 4), m=32, n = 4, u32

tv layout和实际2x2 wave相关， A， B 不同，MMA atom里面wave0, wave1是M方向排， wave0, wave2是N方向，
但是我们的bA用实际上用了mfma B 的tile, bB 用了 mfma A 的tile
wave0, wave1是N方向排， wave0, wave2是M方向，

scale A, tv_layout = ((16， 4， 2， 2), 1), ((1, 32, 0, 16), 1), or 

scale B,tv_layout = ((16， 4， 2， 2), 1), ((1, 32, 16, 0), 1),
```


# 4 wave mxfp4, hybrid
```
TILE_M/TILE_N=256, BK=128.
slicing之后，BM/BN=128, BK=128
```
### AC B tile copy:
```
BK = 128, 16x4 thread mapping, 32 mxfp4 elements per lane,128bit copy atom
64 rows, 128 columns,
```
### AC B tv layout:
```
((4, 16, 4), elements_per_128b),
((elements_per_128b x 64, 1, 16), 64)
```

### AC Padding read layout:
```
BK是128
sub A/B natural layout: (128, 128), (K, 1),主要针对m分组，128=16x8,
((16, 8), BK, K//BK), ((K, 16xK), 1, BK), 根据访问的thread mapping顺序permute为：
((8, 16), BK, K//BK), ((16xK， K), 1, BK)

((8, 2, 8), BK, K//BK), ((16xK， K, 2K), 1, BK)

```

### LDS for B per slice
B LDS layout:(128, 128) mxfp4 = 128*64 bytes , without considering padding.
### AC padding write LDS layout:
```
BK=128，每16行（2048个logical MXFP4）padding 128个logical MXFP4（64B）。
LDS layout : ((16, 8), BK), ((BK, 16xBK+128), 1)
每个16-row group占1088B：1024B packed MXFP4数据 + 64B padding。
```

### S2R read LDS layout padding:
```
S2R与写入使用相同padding布局；packed byte地址为：
`(row // 16) * 1088 + (row % 16) * 64 + col_byte`。

每个wave写入一个完整16-row group，因此G2S使用direct 128-bit
`raw_ptr_buffer_load_lds`，无需register-staged LDS write。

1024^3 non-scale实测整个kernel：`SQ_LDS_BANK_CONFLICT=49152`、
`SQ_LDS_IDX_ACTIVE=98304`、`SQ_LDS_ADDR_CONFLICT=0`。
 ```

### 2048+128 FP4 padding performance

Protocol: MI355X/gfx950, `M=N=8192`, `with_scale=True`, non-swizzle,
32 rotating datasets, 50 timings, best latency. Regression is
`Hybrid/MXFP8 - 1`; a positive value means Hybrid is faster.

| K | MXFP8 | Hybrid MXFP4 | Hybrid/MXFP8 | Regression |
|---:|---:|---:|---:|---:|
| 8192 | 2723.42 TFLOPS | 2866.87 TFLOPS | 105.27% | +5.27% |
| 10240 | 2755.81 TFLOPS | 2892.21 TFLOPS | 104.95% | +4.95% |
| 12288 | 2791.75 TFLOPS | 2946.14 TFLOPS | 105.53% | +5.53% |
| 14336 | 2834.27 TFLOPS | 2988.70 TFLOPS | 105.45% | +5.45% |

### Historical 1024+64 FP4 padding performance

Protocol: MI355X/gfx950, `M=N=8192`, `with_scale=True`, non-swizzle,
32 rotating datasets, 50 timings, best latency.

| K | MXFP8 | Hybrid MXFP4 | Hybrid/MXFP8 |
|---:|---:|---:|---:|
| 8192 | 2728.02 TFLOPS | 1703.39 TFLOPS | 62.44% |
| 10240 | 2755.36 TFLOPS | 1722.53 TFLOPS | 62.52% |
| 12288 | 2794.20 TFLOPS | 1742.44 TFLOPS | 62.36% |
| 14336 | 2831.60 TFLOPS | 1750.92 TFLOPS | 61.84% |



# Non-scale FP8 optimization log

Protocol: 8192^3, float32 randn input, 32 rotating clones, 50 runs,
pyhip.cudaPerf best-of-run. Gluon reference: 3132.84 TFLOPS.

## 0. Tiled-copy baseline
- Method: BufferCopyLDS128b tiled copy.
- Correctness: pass.
- ISA: mainloop has 36 v_add and 32 v_readfirstlane.
- Perf: 2938.90 TFLOPS; gap to Gluon 6.19%.

## 1. Raw G2S DMA
- Method: raw_ptr_buffer_load_lds with precomputed uniform LDS m0 pointers.
- Correctness: non-scale and scale 512^3 pass.
- ISA: mainloop v_add 36 -> 0, v_readfirstlane 32 -> 0; 42 scalar adds remain.
- Perf: 3001.5 TFLOPS repeatable best, +2.13%; gap to Gluon 4.19%.
- ATT: mainloop 4706.7 -> 4461.2 cycles.

## 2. Precomputed global voffsets
- Method: move row/wave/copy-round offsets outside the mainloop.
- Correctness: non-scale 512^3 pass.
- ISA: mainloop 323 -> 285 instructions; scalar adds 42 -> 4.
- Perf: 3006.73 TFLOPS, +0.17% over step 1 and +2.31% over baseline;
     gap to Gluon 4.02%.

## 3. Mainloop scheduling
- DSRD two slots earlier: correctness pass, 3035.62 TFLOPS, +0.96%.
- DSRD three slots earlier: correctness pass, 3024.27 TFLOPS; rejected.
- Direct Gluon local ordering: correctness pass, 2939.53 TFLOPS; rejected.

## 4. Remove precomputed global voffsets
- Method: compute row/wave/copy-round offsets inside each raw_g2s call; keep the
     DSRD two-slot schedule.
- Correctness: non-scale 512^3 pass, diff 2.23e-8.
- Perf: 3025.60 TFLOPS, -0.33% versus 3035.62 with precomputed voffsets;
     final-window average 3002.22 TFLOPS.