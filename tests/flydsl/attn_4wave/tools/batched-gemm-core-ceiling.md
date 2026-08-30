# Batched GEMM core ceiling

[`probe-batched-gemm-core-ceiling.py`](probe-batched-gemm-core-ceiling.py)用于测量gfx94x上
均衡FP8 batched GEMM的core co-issue ceiling。A/B为FP8，D为BF16；结果用于预测候选
MoE down tile的核心吞吐上界，不计算正确GEMM结果。

本工具从`8c1a86965b2a65b69036291f9b95533044c2d81f`中的同名探针移植。原文件SHA256为
`e393589fa1f49a0ede20ccd5df0f3aff2ad8fab7ed7a9fa9917dc59fba56bbcf`。正式测量版本SHA256为
`baca74ae95a564f98b14cfadd3f7f75665a3a7d5795d3362f109c4d6b3fe22a2`。测量后执行
Black/Ruff机械格式化，并把内聚occupancy helper的返回值对齐来源语义
`min(requested, achievable)`；本轮七个配置的`requested == achievable`，结果不受该修正
影响。最终版本SHA256为`80da30297540083b75eaafa22347ceb8a5379a3274c93a0c9e02065f7d952299`。

本次只移植该探针：未移植原提交中的生产profile、TODO或wave-stage工具，也未修改
`src/core/asmjit.py`。当前文件内聚了原探针依赖的GPU状态、统计和occupancy helper，
并只对自身JIT compile key做哈希缩短。

## 模型边界

- 每个batch元素具有相同的`M/N/K`；
- 每个WG处理一个`BM x (BN * NT/WG)`输出tile组；
- `waves_m * waves_n`必须为4、8或16；
- 每wave覆盖`BM/waves_m x BN/waves_n`；
- A使用`--a-in-reg`时只预留完整K维寄存器，不读取A buffer；
- B始终由VMEM读取，D在全部K tile后写出；
- VMEM结果、MFMA operand和D payload彼此独立，不存在真实RAW链；
- MFMA C输入固定为0，只保留指定数量的4-AGPR write-only目标；
- LDS只用于限制occupancy，最终ISA包含任何`ds_*`都会失败；
- 不包含scale、metadata、activation、reduction和真实epilogue。

因此该结果是core co-issue ceiling，不是正确kernel可以直接达到的性能下界。

## 工作量推导

```text
waves/WG = waves_m * waves_n
wave_M = BM / waves_m
wave_N = BN / waves_n

M_tiles = ceil(M / BM)
N_tiles = ceil(N / BN)
N_tile_groups = ceil(N_tiles / NT/WG)
K_tiles = ceil(K / BK)
workgroups = batch * M_tiles * N_tile_groups

MFMA/wave/K = (wave_M / 16) * (wave_N / 16) * (BK / 32)
A bytes/wave/K = wave_M * BK
B bytes/wave/K = wave_N * BK
D bytes/wave = wave_M * wave_N * 2
```

`useful_tflops`使用原始`batch*M*N*K`，`executed_tflops`使用向BM/BN/BK补齐后的工作量。

## 运行方法

先运行不访问GPU的派生自测：

```bash
PYTHONPATH=src:. python3 \
  tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py self-test
```

正式预测使用GPU4、10套B/D地址、40次round-robin warmup和50个CUDA-event样本。以Hy3
为例：

```bash
HIP_VISIBLE_DEVICES=4 \
PYTHONPATH=/opt/aiter:/usr/local/lib/python3.10/dist-packages:\
/root/workspace/luocheng/FlyDSL/build-fly/python_packages:\
/root/workspace/luocheng/FlyDSL/python:src \
PYHIP_JIT_LOG=0 python3 \
  tests/flydsl/attn_4wave/tools/probe-batched-gemm-core-ceiling.py bench \
  --physical-device 4 --device 0 \
  --batch 193 --m 1528 --n 4096 --k 192 \
  --bm 64 --bn 512 --bk 64 \
  --waves-m 1 --waves-n 8 --n-tiles-per-wg 8 \
  --waves-per-simd 4 --accumulator-destinations 1 --a-in-reg \
  --grid-order batch_m_n --schedule 2stage_0 --cache-policy temporal \
  --buffer-copies 10 --warmups 40 --samples 50 \
  --launches-per-sample 1 --sample-sync end \
  --json /tmp/batched-gemm-ceiling-probe-hy3.json
```

工具要求初始GPU处于`auto`且空闲，设置1800MHz performance determinism、PTL
`Enabled / VECTOR,F8`并检查650W power cap；结束后恢复原performance level和PTL。

生产实测采用同样的10-buffer、40-warmup、50-sample、`sample-sync=end`协议，每个event
只包围当前生产down dispatch。为遵守“只移植该工具”，生产profile没有加入仓库；本轮
临时harness为`/tmp/profile-batched-gemm-production-current.py`，SHA256为
`f8ba0964023d2e27beb50a83250772bdd7b9055d5efdc06ec6ac316c12d8fc3e`。

## 2026-08-27配置

当前生产源码为`168808caeacf7e0d7cb336df25554a0bf778d6dc`。每个ceiling WG处理完整N维，
即`NT/WG=ceil(N/BN)`且`N_tile_groups=1`。

| Case | ceiling `B x M x N x K` | `BM x BN x BK` | `WM x WN` (`W/WG`) | `W/SIMD` | `NT/WG` | ISA；LDS | ceiling WG | 生产active/launched WG |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Hy3 K=192 | `193x1528x4096x192` | `64x512x64` | `1x8` (8) | 4 | 8 | 92V+4A；32KiB | 4,632 | 4,632/4,801 |
| Qwen3.5 397B K=512 | `512x640x4096x512` | `64x256x128` | `1x4` (4) | 2 | 16 | 204V+4A；32KiB | 5,120 | 5,120/5,632 |
| Qwen3.5 397B K=256 | `512x640x4096x256` | `64x256x128` | `1x4` (4) | 2 | 16 | 140V+4A；32KiB | 5,120 | 5,120/5,632 |
| Qwen3.5 35B K=512 | `256x1024x2048x512` | `64x256x128` | `1x4` (4) | 2 | 8 | 204V+4A；32KiB | 4,096 | 4,096/4,352 |
| Qwen3.5 35B K=256 | `256x1024x2048x256` | `64x256x128` | `1x4` (4) | 2 | 8 | 140V+4A；32KiB | 4,096 | 4,096/4,352 |
| Xiaomi K=256 | `384x683x6144x256` | `64x256x128` | `1x4` (4) | 2 | 24 | 140V+4A；32KiB | 4,224 | 4,224/4,480 |
| H3 K=384 | `128x1024x6144x384` | `128x256x128` | `2x4` (8) | 2 | 24 | 172V+4A；64KiB | 1,024 | 1,024/1,152 |

Hy3和Xiaomi的均衡M是`B*TopK/E`的整数近似，ceiling useful/executed效率分别为
99.48%和97.02%；其余case为100%。七个最终ISA均为0 scratch、无`ds_*`，请求的
waves/SIMD与HIP occupancy一致。

## 预测与实测

两侧均独立运行，表中差值不是配对置信区间：

```text
差值 = ceiling - 生产
达到率 = 生产 / ceiling
```

| Case | 生产 ms / useful TFLOPS `[P25--P75]` | ceiling ms / useful TFLOPS `[P25--P75]` | ceiling - 生产 | 达到率 |
| --- | ---: | ---: | ---: | ---: |
| Hy3 K=192 | 1.3747 / 337.43 `[334.42--353.63]` | 1.2659 / 366.42 `[365.78--367.85]` | +28.99T / +8.59% | 92.09% |
| Qwen3.5 397B K=512 | 3.4214 / 401.71 `[400.25--402.76]` | 2.7981 / 491.19 `[490.94--491.68]` | +89.49T / +22.28% | 81.78% |
| Qwen3.5 397B K=256 | 1.8616 / 369.14 `[360.19--372.05]` | 1.6991 / 404.45 `[394.15--405.37]` | +35.31T / +9.57% | 91.27% |
| Qwen3.5 35B K=512 | 1.4113 / 389.54 `[388.78--391.27]` | 1.1284 / 487.21 `[486.40--488.41]` | +97.66T / +25.07% | 79.95% |
| Qwen3.5 35B K=256 | 0.7665 / 358.62 `[352.33--359.46]` | 0.6738 / 407.96 `[407.01--410.09]` | +49.34T / +13.76% | 87.91% |
| Xiaomi K=256 | 2.2746 / 362.53 `[361.38--363.79]` | 2.1124 / 390.57 `[389.52--391.95]` | +28.03T / +7.73% | 92.82% |
| H3 K=384 | 1.5777 / 392.01 `[390.50--395.17]` | 1.4609 / 423.34 `[422.91--424.88]` | +31.33T / +7.99% | 92.60% |

七份ceiling JSON的排序清单SHA256为
`1c3f5cd5b5c4b4f4d9b51f5852972d12ed7c82b0e6253c84a5705507fdc59f00`；七份生产
JSON的排序清单SHA256为
`478e08d30d7f729f5c7d4a0143e1950953558afa9a3932ef03067eaef8a318c7`。

## Wave、调度与W/SIMD扫描

在上述固定tile和full-N配置上，扫描Q1/Q2/Q4和W4/W8。W4测试
`2stage_0/2stage_prio/interleave`和Q1/Q2/Q4；W8测试四种调度和Q2/Q4。
`2stage_barrier`要求至少8 waves/WG，因此W4不适用。Hy3同时测试W8 `1x8`和新增的
W8 `2x4`；其余case测试W8 `2x4`。每项沿用10-buffer、40 warmup、50 sample和
`sample-sync=end`；扫描harness SHA256为
`73ed9fb6ad25761028042761aaa4935ba82467d35d705383f699e676c9b9a031`。

表中数值为useful TFLOPS中位数`[P25--P75]`；`R(resource)`表示请求Q因最终
VGPR/AGPR或workgroup资源被拒绝，`n/a`表示该W/WG不能形成对应整数驻留WG。七个case
共127项，其中97项实际执行，30项按资源约束拒绝。七份扫描JSON的排序清单SHA256为
`62e64333694bf8f5ce46e649e23c3c060c360e42a1863050a313c79efe9973f8`。

| Case | 生产布局匹配ceiling | 全扫描最优ceiling | 同轮提升 | 生产TFLOPS | 生产/扫描最优 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hy3 K=192 | W8 `1x8` / `2stage_0` / Q4：367.09 | W4 `1x4` / `2stage_0` / Q4：381.95 | +4.05% | 337.43 | 88.34% |
| Qwen3.5 397B K=512 | W4 `1x4` / `2stage_0` / Q2：491.21 | W4 `1x4` / `2stage_0` / Q2：491.21 | +0.00% | 401.71 | 81.78% |
| Qwen3.5 397B K=256 | W4 `1x4` / `2stage_0` / Q2：394.54 | W4 `1x4` / `2stage_0` / Q2：394.54 | +0.00% | 369.14 | 93.56% |
| Qwen3.5 35B K=512 | W4 `1x4` / `2stage_0` / Q2：487.09 | W4 `1x4` / `2stage_0` / Q2：487.09 | +0.00% | 389.54 | 79.97% |
| Qwen3.5 35B K=256 | W4 `1x4` / `2stage_0` / Q2：408.23 | W4 `1x4` / `2stage_0` / Q2：408.23 | +0.00% | 358.62 | 87.85% |
| Xiaomi K=256 | W4 `1x4` / `2stage_0` / Q2：390.11 | W4 `1x4` / `2stage_0` / Q2：390.11 | +0.00% | 362.53 | 92.93% |
| H3 K=384 | W8 `2x4` / `2stage_0` / Q2：424.12 | W4 `2x2` / `2stage_0` / Q2：493.62 | +16.39% | 392.01 | 79.42% |

精简矩阵中，所有非Hy3 case的最优Q都是Q2，Hy3为Q4。所有case的全局最优调度都是
`2stage_0`；`2stage_prio`接近但没有胜出，`interleave`普遍更慢，`2stage_barrier`
在W8上明显回退。Hy3新增W8 `2x4`的最佳结果为311.69T，显著低于W8 `1x8`的
367.13T；W4 ceiling相对生产匹配W8 `1x8`高4.05%。H3的W4 `2x2`相对生产匹配W8
`2x4`高16.39%，仍是最值得实现正确kernel候选的布局变化。

### Hy3 K=192

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 287.90 `[287.14--290.20]` | 366.30 `[365.76--369.60]` | **381.95 `[380.46--386.30]`** |
| W4 (`1x4`) | `2stage_prio` | 287.64 `[287.12--289.53]` | 366.16 `[365.14--369.63]` | 381.36 `[380.06--388.06]` |
| W4 (`1x4`) | `interleave` | 269.23 `[268.37--271.45]` | 336.14 `[335.59--339.47]` | 358.35 `[357.39--362.23]` |
| W8 (`1x8`) | `2stage_0` | n/a | 307.73 `[307.23--310.83]` | 367.09 `[366.50--370.82]` |
| W8 (`1x8`) | `2stage_prio` | n/a | 307.77 `[307.42--310.88]` | 367.13 `[366.59--371.01]` |
| W8 (`1x8`) | `2stage_barrier` | n/a | 253.81 `[253.01--255.08]` | 318.26 `[317.19--321.05]` |
| W8 (`1x8`) | `interleave` | n/a | 284.50 `[284.00--287.40]` | 340.44 `[340.07--344.79]` |
| W8 (`2x4`) | `2stage_0` | n/a | 280.17 `[279.84--282.19]` | 311.60 `[311.08--313.87]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 280.01 `[279.68--282.11]` | 311.69 `[311.28--314.17]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 235.13 `[234.60--236.31]` | 283.06 `[282.44--285.15]` |
| W8 (`2x4`) | `interleave` | n/a | 260.08 `[259.62--262.33]` | 289.66 `[289.21--292.54]` |

最优：W4 (`1x4`) / `2stage_0` / Q4，381.95 TFLOPS `[380.46--386.30]`。

### Qwen3.5 397B K=512

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 346.40 `[346.18--346.74]` | **491.21 `[491.01--491.57]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 345.71 `[345.46--346.17]` | 490.32 `[489.98--490.85]` | R(resource) |
| W4 (`1x4`) | `interleave` | 330.09 `[329.53--331.07]` | 469.62 `[469.22--470.07]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 339.04 `[338.64--339.61]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 338.56 `[338.13--339.15]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 268.39 `[267.38--268.90]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 319.59 `[319.12--320.16]` | R(resource) |

最优：W4 (`1x4`) / `2stage_0` / Q2，491.21 TFLOPS `[491.01--491.57]`。

### Qwen3.5 397B K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 267.39 `[266.21--269.35]` | **394.54 `[394.27--397.42]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 266.64 `[265.50--267.95]` | 394.20 `[393.59--397.03]` | R(resource) |
| W4 (`1x4`) | `interleave` | 231.92 `[231.18--233.79]` | 347.52 `[347.09--350.54]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 268.19 `[267.70--269.77]` | 317.85 `[317.59--319.60]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 265.81 `[265.33--267.82]` | 317.86 `[317.57--319.78]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 185.09 `[184.34--186.21]` | 264.39 `[263.05--266.19]` |
| W8 (`2x4`) | `interleave` | n/a | 251.77 `[251.07--254.00]` | 294.78 `[294.24--297.14]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，394.54 TFLOPS `[394.27--397.42]`。

### Qwen3.5 35B K=512

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 341.37 `[340.79--342.70]` | **487.09 `[486.32--488.94]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 341.00 `[340.35--341.96]` | 486.23 `[485.45--487.30]` | R(resource) |
| W4 (`1x4`) | `interleave` | 324.17 `[323.57--325.17]` | 467.25 `[466.33--468.83]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 337.02 `[335.97--338.21]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 336.36 `[335.74--338.21]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 266.62 `[265.58--268.39]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 314.90 `[313.72--316.27]` | R(resource) |

最优：W4 (`1x4`) / `2stage_0` / Q2，487.09 TFLOPS `[486.32--488.94]`。

### Qwen3.5 35B K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 263.47 `[262.57--265.53]` | **408.23 `[407.31--410.93]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 262.72 `[262.03--263.76]` | 407.72 `[406.97--410.90]` | R(resource) |
| W4 (`1x4`) | `interleave` | 232.67 `[231.89--234.67]` | 352.82 `[351.92--355.50]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 265.44 `[264.77--267.15]` | 336.10 `[335.46--337.71]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 264.66 `[264.02--266.84]` | 336.36 `[335.53--338.23]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 186.96 `[186.21--188.25]` | 266.25 `[264.97--268.44]` |
| W8 (`2x4`) | `interleave` | n/a | 254.48 `[253.83--255.58]` | 306.71 `[305.62--309.69]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，408.23 TFLOPS `[407.31--410.93]`。

### Xiaomi K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 280.03 `[279.59--280.60]` | **390.11 `[389.62--390.90]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 279.32 `[278.95--279.68]` | 389.91 `[389.38--390.28]` | R(resource) |
| W4 (`1x4`) | `interleave` | 243.59 `[243.22--244.41]` | 343.94 `[343.69--344.64]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 273.43 `[273.25--273.94]` | 310.08 `[309.84--310.40]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 273.43 `[273.22--273.71]` | 310.02 `[309.82--310.38]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 195.49 `[195.08--195.98]` | 273.18 `[272.87--273.61]` |
| W8 (`2x4`) | `interleave` | n/a | 258.35 `[258.07--258.81]` | 288.44 `[288.19--288.94]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，390.11 TFLOPS `[389.62--390.90]`。

### H3 K=384

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`2x2`) | `2stage_0` | 381.94 `[380.90--386.96]` | **493.62 `[492.76--499.38]`** | R(resource) |
| W4 (`2x2`) | `2stage_prio` | 381.04 `[380.27--386.35]` | 493.45 `[492.89--499.84]` | R(resource) |
| W4 (`2x2`) | `interleave` | 351.85 `[350.67--355.83]` | 454.21 `[453.08--464.25]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 424.12 `[423.10--430.11]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 423.20 `[422.43--429.30]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 327.10 `[325.50--330.97]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 402.00 `[401.01--407.67]` | R(resource) |

最优：W4 (`2x2`) / `2stage_0` / Q2，493.62 TFLOPS `[492.76--499.38]`。

## 结论

生产达到ceiling的范围为79.95%--92.82%。Hy3、Xiaomi和H3均超过92%；两个Qwen K=256
路径分别达到91.27%和87.91%。两个K=512 `default`路径只有81.78%和79.95%，且处于
204V+4A资源档位，是后续优先优化对象。

ceiling省略了真实kernel的VMEM到MFMA RAW、LDS搬运、scale、metadata和epilogue，因此
吞吐差距本身不能定位具体瓶颈；归因仍需PMC或ATT。完整MoE路径和最终path选择见
[MAIN_MERGE_PERFORMANCE_REPORT.md](../../../contrib/moe/MAIN_MERGE_PERFORMANCE_REPORT.md)。
