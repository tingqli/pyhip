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

2026-08-31新增默认关闭的`--b-load-cooperation`，用于模拟沿M分工的多个wave协作加载
同一个B tile；默认值1保持上述正式版本的行为。协作加载扩展版本SHA256为
`f277294969e170a47b81ecb331ed3eca6f2f8e7cbd3f21b2e3d151f6b7664721`。

本次只移植该探针：未移植原提交中的生产profile、TODO或wave-stage工具，也未修改
`src/core/asmjit.py`。当前文件内聚了原探针依赖的GPU状态、统计和occupancy helper，
并只对自身JIT compile key做哈希缩短。

## 模型边界

- 每个batch元素具有相同的`M/N/K`；
- 每个WG处理一个`BM x (BN * NT/WG)`输出tile组；
- `waves_m * waves_n`必须为4、8或16；
- 每wave覆盖`BM/waves_m x BN/waves_n`；
- `--b-load-cooperation C`要求`C`整除`waves_m`。同一`wave_n`下的C个`wave_m`
  各自读取连续`1/C` B tile，合起来只加载一份B；
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
B bytes/wave/K = wave_N * BK / B_load_cooperation
D bytes/wave = wave_M * wave_N * 2
```

`useful_tflops`使用原始`batch*M*N*K`，`executed_tflops`使用向BM/BN/BK补齐后的工作量。

带宽只使用`rocprofv3`的PMC实测值，不再用VMEM指令数或模型有效字节推导：

- `FETCH_SIZE`和`WRITE_SIZE`不能在同一个硬件pass采集，因此每个候选分别运行读、写
  两轮；每轮使用10套B/D地址、40次warmup和10次正式sample；
- CSV中的`Counter_Value`按KiB解释，时间戳单位为ns。每个正式dispatch的带宽为
  `Counter_Value * 1024 / (End_Timestamp - Start_Timestamp)`，单位为GB/s；
- 候选的读、写带宽分别取10个正式dispatch带宽的中位数，`PMC总GB/s = 读中位数 +
  写中位数`。读写来自独立pass，不能理解成同一dispatch上的配对样本；
- 只使用dispatch映射中`phase == "sample"`的记录，不把40次warmup混入统计。

本轮共覆盖97个成功候选；每个counter有4,850条连续dispatch，其中970条为正式样本。
FETCH/WRITE的候选、phase、repetition和buffer映射逐项一致。14份映射JSON、14份PMC
CSV、7份无插桩扫描JSON和PMC驱动的排序输入清单SHA256为
`b0c3b16ace678096fd129feeca32c69de66c2dc80df9bf25d779ad338d094e15`；保留全部
原始样本的聚合JSON为`/tmp/batched-gemm-ceiling-pmc-r2.json`，SHA256为
`0186891ce71acf9393884644e1441b3107b988c891a9dc7dbe766d8a2434d488`。

PMC实测使用以下命令形态；`FETCH_SIZE`和`WRITE_SIZE`分别执行，批量驱动只位于
`/tmp`，没有作为第二个工具移植到仓库：

```bash
HIP_VISIBLE_DEVICES=4 \
PYTHONPATH=/opt/aiter:/usr/local/lib/python3.10/dist-packages:src \
PYHIP_JIT_LOG=0 rocprofv3 \
  --pmc FETCH_SIZE \
  --kernel-include-regex 'batched_gemm_core_ceiling' \
  -f csv -d /tmp/batched-gemm-pmc-fetch-hy3-r2 -o pmc -- \
  python3 /tmp/profile_batched_gemm_ceiling_pmc.py \
    --counter FETCH_SIZE --case hy3 \
    --physical-device 4 --device 0 \
    --buffer-copies 10 --warmups 40 --samples 10 \
    --output /tmp/batched-gemm-pmc-fetch-hy3-r2.json
```

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

上述要求适用于2026-08-27的无插桩时间/TFLOPS正式测量。本轮PMC采集仍固定GPU4、
1800MHz performance determinism和650W，但PTL setter返回成功后立即回读仍为
`Disabled / N/A`，因此不声称PMC轮运行在`Enabled / VECTOR,F8`。PMC GB/s的分子和
时间均来自同一个PMC pass，不与2026-08-27的无插桩时间交叉计算。

生产实测采用同样的10-buffer、40-warmup、50-sample、`sample-sync=end`协议，每个event
只包围当前生产down dispatch。为遵守“只移植该工具”，生产profile没有加入仓库；本轮
临时harness为`/tmp/profile-batched-gemm-production-current.py`，SHA256为
`f8ba0964023d2e27beb50a83250772bdd7b9055d5efdc06ec6ac316c12d8fc3e`。

生产侧表格保留2026-08-27的无插桩时间和TFLOPS，但不再列模型推导带宽。
该日使用的系统FlyDSL安装已在2026-08-31被覆盖；历史生产源码在新编译器下触发
`fly.mma.make_fragment`区域支配错误，旧配套环境又缺少`buffer_ops` API。因此本轮没有
得到可与原生产结果绑定的PMC样本，宁可留空，也不把理论字节冒充PMC实测。

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

表中的ceiling `ms/TFLOPS`来自2026-08-27的50个无插桩样本；PMC读/写/总来自
2026-08-31同一候选配置的独立双pass重跑。两组数据不共享样本，PMC带宽也没有使用表中
所列的无插桩`ms`作分母。

| Case | 生产 ms / useful TFLOPS `[TFLOPS P25--P75]` | ceiling ms / useful TFLOPS / PMC读/写/总GB/s `[TFLOPS P25--P75]` | ceiling - 生产 | 达到率 |
| --- | ---: | ---: | ---: | ---: |
| Hy3 K=192 | 1.3747 / 337.43 `[334.42--353.63T]` | 1.2659 / 366.42 / 493.9/1805.8/2299.7 `[365.78--367.85T]` | +28.99T / +8.59% | 92.09% |
| Qwen3.5 397B K=512 | 3.4214 / 401.71 `[400.25--402.76T]` | 2.7981 / 491.19 / 1396.1/818.3/2214.4 `[490.94--491.68T]` | +89.49T / +22.28% | 81.78% |
| Qwen3.5 397B K=256 | 1.8616 / 369.14 `[360.19--372.05T]` | 1.6991 / 404.45 / 1199.3/1408.5/2607.7 `[394.15--405.37T]` | +35.31T / +9.57% | 91.27% |
| Qwen3.5 35B K=512 | 1.4113 / 389.54 `[388.78--391.27T]` | 1.1284 / 487.21 / 805.3/804.0/1609.2 `[486.40--488.41T]` | +97.66T / +25.07% | 79.95% |
| Qwen3.5 35B K=256 | 0.7665 / 358.62 `[352.33--359.46T]` | 0.6738 / 407.96 / 701.2/1385.5/2086.7 `[407.01--410.09T]` | +49.34T / +13.76% | 87.91% |
| Xiaomi K=256 | 2.2746 / 362.53 `[361.38--363.79T]` | 2.1124 / 390.57 / 1248.8/1408.3/2657.2 `[389.52--391.95T]` | +28.03T / +7.73% | 92.82% |
| H3 K=384 | 1.5777 / 392.01 `[390.50--395.17T]` | 1.4609 / 423.34 / 802.9/1050.3/1853.2 `[422.91--424.88T]` | +31.33T / +7.99% | 92.60% |

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

详细表中每个实际执行单元为`useful TFLOPS / PMC总GB/s [TFLOPS P25--P75]`；
`R(resource)`表示请求Q因最终VGPR/AGPR或workgroup资源被拒绝，`n/a`表示该W/WG
不能形成对应整数驻留WG。七个case共127项，其中97项实际执行，30项按资源约束拒绝。
每个PMC总带宽均来自对应候选独立FETCH/WRITE pass的正式样本中位数之和。七份无插桩
扫描JSON的排序清单SHA256为
`62e64333694bf8f5ce46e649e23c3c060c360e42a1863050a313c79efe9973f8`。

| Case | 生产布局匹配ceiling TFLOPS / PMC读/写/总GB/s | 全扫描最优ceiling TFLOPS / PMC读/写/总GB/s | 同轮提升 | 生产TFLOPS | 生产/扫描最优 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hy3 K=192 | W8 `1x8` / `2stage_0` / Q4：367.09 / 493.9/1805.8/2299.7 | W4 `1x4` / `2stage_0` / Q4：381.95 / 897.7/1881.0/2778.7 | +4.05% | 337.43 | 88.34% |
| Qwen3.5 397B K=512 | W4 `1x4` / `2stage_0` / Q2：491.21 / 1396.1/818.3/2214.4 | W4 `1x4` / `2stage_0` / Q2：491.21 / 1396.1/818.3/2214.4 | +0.00% | 401.71 | 81.78% |
| Qwen3.5 397B K=256 | W4 `1x4` / `2stage_0` / Q2：394.54 / 1199.3/1408.5/2607.7 | W4 `1x4` / `2stage_0` / Q2：394.54 / 1199.3/1408.5/2607.7 | +0.00% | 369.14 | 93.56% |
| Qwen3.5 35B K=512 | W4 `1x4` / `2stage_0` / Q2：487.09 / 805.3/804.0/1609.2 | W4 `1x4` / `2stage_0` / Q2：487.09 / 805.3/804.0/1609.2 | +0.00% | 389.54 | 79.97% |
| Qwen3.5 35B K=256 | W4 `1x4` / `2stage_0` / Q2：408.23 / 701.2/1385.5/2086.7 | W4 `1x4` / `2stage_0` / Q2：408.23 / 701.2/1385.5/2086.7 | +0.00% | 358.62 | 87.85% |
| Xiaomi K=256 | W4 `1x4` / `2stage_0` / Q2：390.11 / 1248.8/1408.3/2657.2 | W4 `1x4` / `2stage_0` / Q2：390.11 / 1248.8/1408.3/2657.2 | +0.00% | 362.53 | 92.93% |
| H3 K=384 | W8 `2x4` / `2stage_0` / Q2：424.12 / 802.9/1050.3/1853.2 | W4 `2x2` / `2stage_0` / Q2：493.62 / 1097.0/1037.6/2134.5 | +16.39% | 392.01 | 79.42% |

精简矩阵中，所有非Hy3 case的最优Q都是Q2，Hy3为Q4。所有case的全局最优调度都是
`2stage_0`；`2stage_prio`接近但没有胜出，`interleave`普遍更慢，`2stage_barrier`
在W8上明显回退。Hy3新增W8 `2x4`的最佳结果为311.69T，显著低于W8 `1x8`的
367.13T；W4 ceiling相对生产匹配W8 `1x8`高4.05%。H3的W4 `2x2`相对生产匹配W8
`2x4`高16.39%，仍是最值得实现正确kernel候选的布局变化。

### Hy3 K=192

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 287.90 / 1667.7 `[287.14--290.20T]` | 366.30 / 2216.4 `[365.76--369.60T]` | **381.95 / 2778.7 `[380.46--386.30T]`** |
| W4 (`1x4`) | `2stage_prio` | 287.64 / 1662.8 `[287.12--289.53T]` | 366.16 / 2214.3 `[365.14--369.63T]` | 381.36 / 2773.8 `[380.06--388.06T]` |
| W4 (`1x4`) | `interleave` | 269.23 / 1593.3 `[268.37--271.45T]` | 336.14 / 2095.7 `[335.59--339.47T]` | 358.35 / 2493.8 `[357.39--362.23T]` |
| W8 (`1x8`) | `2stage_0` | n/a | 307.73 / 1981.6 `[307.23--310.83T]` | 367.09 / 2299.7 `[366.50--370.82T]` |
| W8 (`1x8`) | `2stage_prio` | n/a | 307.77 / 1976.9 `[307.42--310.88T]` | 367.13 / 2296.8 `[366.59--371.01T]` |
| W8 (`1x8`) | `2stage_barrier` | n/a | 253.81 / 1439.9 `[253.01--255.08T]` | 318.26 / 1846.8 `[317.19--321.05T]` |
| W8 (`1x8`) | `interleave` | n/a | 284.50 / 1826.9 `[284.00--287.40T]` | 340.44 / 2184.1 `[340.07--344.79T]` |
| W8 (`2x4`) | `2stage_0` | n/a | 280.17 / 1655.1 `[279.84--282.19T]` | 311.60 / 1926.8 `[311.08--313.87T]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 280.01 / 1655.0 `[279.68--282.11T]` | 311.69 / 1925.5 `[311.28--314.17T]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 235.13 / 1283.7 `[234.60--236.31T]` | 283.06 / 1614.4 `[282.44--285.15T]` |
| W8 (`2x4`) | `interleave` | n/a | 260.08 / 1605.6 `[259.62--262.33T]` | 289.66 / 1852.5 `[289.21--292.54T]` |

最优：W4 (`1x4`) / `2stage_0` / Q4，381.95 TFLOPS / 2778.7 PMC总GB/s
`[380.46--386.30T]`。

### Qwen3.5 397B K=512

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 346.40 / 1540.5 `[346.18--346.74T]` | **491.21 / 2214.4 `[491.01--491.57T]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 345.71 / 1535.3 `[345.46--346.17T]` | 490.32 / 2206.0 `[489.98--490.85T]` | R(resource) |
| W4 (`1x4`) | `interleave` | 330.09 / 1471.0 `[329.53--331.07T]` | 469.62 / 2175.7 `[469.22--470.07T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 339.04 / 1634.0 `[338.64--339.61T]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 338.56 / 1626.1 `[338.13--339.15T]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 268.39 / 1188.5 `[267.38--268.90T]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 319.59 / 1530.7 `[319.12--320.16T]` | R(resource) |

最优：W4 (`1x4`) / `2stage_0` / Q2，491.21 TFLOPS / 2214.4 PMC总GB/s
`[491.01--491.57T]`。

### Qwen3.5 397B K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 267.39 / 1662.7 `[266.21--269.35T]` | **394.54 / 2607.7 `[394.27--397.42T]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 266.64 / 1660.7 `[265.50--267.95T]` | 394.20 / 2608.6 `[393.59--397.03T]` | R(resource) |
| W4 (`1x4`) | `interleave` | 231.92 / 1498.8 `[231.18--233.79T]` | 347.52 / 2345.0 `[347.09--350.54T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 268.19 / 1810.2 `[267.70--269.77T]` | 317.85 / 3196.0 `[317.59--319.60T]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 265.81 / 1794.8 `[265.33--267.82T]` | 317.86 / 3188.6 `[317.57--319.78T]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 185.09 / 1189.3 `[184.34--186.21T]` | 264.39 / 1702.6 `[263.05--266.19T]` |
| W8 (`2x4`) | `interleave` | n/a | 251.77 / 1685.4 `[251.07--254.00T]` | 294.78 / 2587.5 `[294.24--297.14T]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，394.54 TFLOPS / 2607.7 PMC总GB/s
`[394.27--397.42T]`。

### Qwen3.5 35B K=512

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 341.37 / 1151.4 `[340.79--342.70T]` | **487.09 / 1609.2 `[486.32--488.94T]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 341.00 / 1149.4 `[340.35--341.96T]` | 486.23 / 1604.6 `[485.45--487.30T]` | R(resource) |
| W4 (`1x4`) | `interleave` | 324.17 / 1097.3 `[323.57--325.17T]` | 467.25 / 1572.5 `[466.33--468.83T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 337.02 / 1176.0 `[335.97--338.21T]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 336.36 / 1173.3 `[335.74--338.21T]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 266.62 / 899.2 `[265.58--268.39T]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 314.90 / 1140.2 `[313.72--316.27T]` | R(resource) |

最优：W4 (`1x4`) / `2stage_0` / Q2，487.09 TFLOPS / 1609.2 PMC总GB/s
`[486.32--488.94T]`。

### Qwen3.5 35B K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 263.47 / 1329.5 `[262.57--265.53T]` | **408.23 / 2086.7 `[407.31--410.93T]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 262.72 / 1325.6 `[262.03--263.76T]` | 407.72 / 2081.3 `[406.97--410.90T]` | R(resource) |
| W4 (`1x4`) | `interleave` | 232.67 / 1206.6 `[231.89--234.67T]` | 352.82 / 1901.7 `[351.92--355.50T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 265.44 / 1390.7 `[264.77--267.15T]` | 336.10 / 1745.9 `[335.46--337.71T]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 264.66 / 1388.6 `[264.02--266.84T]` | 336.36 / 1744.9 `[335.53--338.23T]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 186.96 / 973.4 `[186.21--188.25T]` | 266.25 / 1406.7 `[264.97--268.44T]` |
| W8 (`2x4`) | `interleave` | n/a | 254.48 / 1346.1 `[253.83--255.58T]` | 306.71 / 1638.5 `[305.62--309.69T]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，408.23 TFLOPS / 2086.7 PMC总GB/s
`[407.31--410.93T]`。

#### K=256固定开销消融

针对K=256相对K=512的吞吐差距，在W4 `1x4` / Q2 / `2stage_0`基线上测试三项
改动：

1. `--cross-n-prefetch --cross-n-spread-stores`：跨N预取，并把前一个N tile的8条
  store分散到后一个tile的MFMA序列中；
2. `--cross-n-prefetch`：跨N预取但保持8条store成组发射，消除中间N tile结束后的
  逐tile完全排空；
3. `--bk 256`：把两个BK128 K tile合并为一个BK256 tile，减少K循环和wait边界。

每项和基线均独立运行两轮，第二轮使用反向顺序。每轮沿用GPU4、1800MHz、650W、PTL
`Enabled / VECTOR,F8`、10-buffer、40 warmup、50 CUDA-event sample和
`sample-sync=end`。下表的时间/TFLOPS是两轮中位数的中位数：

| 方案 | 两轮TFLOPS | 汇总ms / TFLOPS | 相对基线延迟 | 相对基线TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| 基线BK128 | 408.230 / 408.241 | 0.673332 / 408.235 | 0.00% | 0.00% |
| 1. 跨N预取+分散写回 | 383.660 / 384.389 | 0.715783 / 384.025 | +6.30% | -5.93% |
| 2. 仅跨N预取 | 404.243 / 403.874 | 0.680293 / 404.059 | +1.03% | -1.02% |
| 3. BK256 | 399.833 / 400.520 | 0.686892 / 400.177 | +2.01% | -1.97% |

三项均未提升，且两轮方向一致。静态ISA中的MFMA/load/store数均保持
`1024/128/64`；基线、方案1、方案2、方案3的`s_waitcnt`数分别为
`25/18/18/17`。减少wait指令没有减少暴露延迟：

- 方案1每16条MFMA插入1条store，持续打断MFMA发射，并让读写VMEM长期同时在途；
- 方案2把下一个N tile的load前移，但成组store仍存在；额外VMEM重叠没有覆盖其成本；
- 方案3在每个N tile开头一次发射16条load后立即`vmcnt(0)`，失去BK128两个K tile之间
  的load/MFMA流水，且SGPR从36增加到44。

因此当前探针中不能通过这三种直接变换回收K=256的固定开销，不应据此修改生产kernel。
八份正式JSON的排序清单SHA256为
`e4a9c6b1af92085b6effb1b50b6fe3658c3ab7ed677a03e4965599cbf3f5bad6`。

### 4-wave协作B加载估算

该形态对应`moe_gemm_down_tp.py`中的设计：`BM=256`由4个wave沿M按`4x1`切分，每个
wave负责`64x64`输出；4个wave协作把一份`BN64 x BK128` B tile加载到LDS。ceiling
探针忽略后续LDS写入、barrier和每个wave从LDS读回B的开销，只保留每wave四分之一的
B VMEM读取，因此结果仍是core co-issue上界。

固定配置为`BM x BN x BK = 256x64x128`、`WM x WN = 4x1`、Q2、`NT/WG=8`、
`--b-load-cooperation 4 --a-in-reg --schedule 2stage_0`。其他case的B/N/K沿用原表，
M按`M_new = M_old * 256 / BM_old`缩放，以保持原ceiling的M tile数；原BM64的case
放大4倍，原BM128的H3放大2倍。这与指定Qwen3.5 397B K=256的`640 -> 2560`一致。

指定配置`512x2560x4096x256`的派生量为：

```text
wave_M x wave_N = 64 x 64
M_tiles = 10, N_tiles = 64, N_tile_groups = 8, K_tiles = 2
workgroups = 512 * 10 * 8 = 40,960
MFMA/wave/BK = 64
B load/wave/BK = (64 * 128 / 4) / 1024 = 2
D store/wave = 64 * 64 * 2 / 1024 = 8
dynamic MFMA/B-load/D-store = 167,772,160 / 5,242,880 / 10,485,760
```

#### 测量协议

- AMD Instinct MI308X / gfx942 / 80 CU，GPU0，1800MHz performance determinism，
  650W，PTL `Enabled / VECTOR,F8`；
- TFLOPS：10套B/D地址、40 warmup、50 CUDA-event sample、`sample-sync=end`；
- 带宽：独立`FETCH_SIZE`和`WRITE_SIZE` pass；每轮10套地址、40 warmup、10正式
  sample；读/写各取样本GB/s中位数，总带宽为二者之和；
- 7个case共350条dispatch/counter，其中70条正式样本/counter；FETCH/WRITE映射
  逐项一致，PMC写字节逐case精确等于D存储量。

#### 结果

`useful/executed TFLOPS`分别使用原始shape和BM/BN/BK补齐shape。Hy3的K=192被BK128
补齐到256，因此useful效率只有74.61%；Xiaomi的M=2732补齐到2816，效率为97.02%。

| Case | `B x M x N x K` | WG | useful效率 | ms / useful / executed TFLOPS `[useful P25--P75]` | ISA | PMC读/写/总GB/s | 相对旧ceiling useful TFLOPS |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Hy3 K=192 | `193x6112x4096x192` | 37,056 | 74.61% | 5.1326 / 361.49 / 484.51 `[360.31--362.81T]` | 92V+4A | 43.8/1612.3/1656.2 | -1.35% |
| Qwen3.5 397B K=512 | `512x2560x4096x512` | 40,960 | 100.00% | 13.4223 / 409.58 / 409.58 `[409.08--410.16T]` | 156V+4A | 86.7/866.5/953.2 | -16.61% |
| Qwen3.5 397B K=256 | `512x2560x4096x256` | 40,960 | 100.00% | 6.0693 / 452.90 / 452.90 `[452.21--453.38T]` | 92V+4A | 78.4/1568.6/1647.0 | +11.98% |
| Qwen3.5 35B K=512 | `256x4096x2048x512` | 16,384 | 100.00% | 3.9463 / 557.24 / 557.24 `[556.77--558.48T]` | 156V+4A | 65.0/872.3/937.3 | +14.37% |
| Qwen3.5 35B K=256 | `256x4096x2048x256` | 16,384 | 100.00% | 2.3241 / 473.10 / 473.10 `[470.42--476.59T]` | 92V+4A | 52.1/1590.5/1642.6 | +15.97% |
| Xiaomi K=256 | `384x2732x6144x256` | 50,688 | 97.02% | 7.1781 / 459.75 / 473.89 `[458.04--461.36T]` | 92V+4A | 95.9/1598.1/1694.0 | +17.71% |
| H3 K=384 | `128x2048x6144x384` | 12,288 | 100.00% | 2.3393 / 528.78 / 528.78 `[528.28--531.68T]` | 124V+4A | 149.8/1125.0/1274.8 | +24.90% |

指定Qwen3.5 397B K=256的估算上界为`452.90 TFLOPS`，PMC物理带宽为
`78.4/1568.6/1647.0 GB/s`（读/写/总）。该shape的输出达到10GiB，写带宽占主导；
协作加载把动态B读取降到原始4x1重复加载的四分之一，但不减少D写回。

K=256形态为`452.90--473.10T`（排除有M padding的Xiaomi则两项Qwen）；K=512在
Qwen397为409.58T、Qwen35为557.24T，说明总WG数量和尾部波次对该无RAW ceiling影响
明显。H3达到528.78T。相对旧ceiling列同时改变了
shape、tile和B读取语义，只用于形态对照，不能解释成单个生产优化的收益。

7份TFLOPS JSON的排序清单SHA256为
`e9e9e0c8487a4ef657c88a88c9334eb77da4cfa924f1912aed41cd97d728f8ad`；探针、临时
PMC驱动、7份TFLOPS JSON、14份映射JSON和14份PMC CSV的排序输入清单SHA256为
`424d96d0431b015f0935f8b28bdc546e16b76a58791454b57894f1bd0120b5f1`。保留全部PMC
原始样本的聚合结果为`/tmp/batched-gemm-coop4-results.json`，SHA256为
`b26b7dd39b9e3e22c049331ef757ca3819ecd3cde6957402cf5719209b13b69b`。

### Xiaomi K=256

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`1x4`) | `2stage_0` | 280.03 / 1757.2 `[279.59--280.60T]` | **390.11 / 2657.2 `[389.62--390.90T]`** | R(resource) |
| W4 (`1x4`) | `2stage_prio` | 279.32 / 1756.7 `[278.95--279.68T]` | 389.91 / 2652.5 `[389.38--390.28T]` | R(resource) |
| W4 (`1x4`) | `interleave` | 243.59 / 1592.1 `[243.22--244.41T]` | 343.94 / 2392.0 `[343.69--344.64T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 273.43 / 1955.3 `[273.25--273.94T]` | 310.08 / 3409.0 `[309.84--310.40T]` |
| W8 (`2x4`) | `2stage_prio` | n/a | 273.43 / 1929.3 `[273.22--273.71T]` | 310.02 / 3408.2 `[309.82--310.38T]` |
| W8 (`2x4`) | `2stage_barrier` | n/a | 195.49 / 1284.1 `[195.08--195.98T]` | 273.18 / 1745.2 `[272.87--273.61T]` |
| W8 (`2x4`) | `interleave` | n/a | 258.35 / 1698.5 `[258.07--258.81T]` | 288.44 / 2948.7 `[288.19--288.94T]` |

最优：W4 (`1x4`) / `2stage_0` / Q2，390.11 TFLOPS / 2657.2 PMC总GB/s
`[389.62--390.90T]`。

### H3 K=384

| W/WG (`WMxWN`) | Schedule | Q1 | Q2 | Q4 |
| --- | --- | ---: | ---: | ---: |
| W4 (`2x2`) | `2stage_0` | 381.94 / 1448.7 `[380.90--386.96T]` | **493.62 / 2134.5 `[492.76--499.38T]`** | R(resource) |
| W4 (`2x2`) | `2stage_prio` | 381.04 / 1447.6 `[380.27--386.35T]` | 493.45 / 2109.0 `[492.89--499.84T]` | R(resource) |
| W4 (`2x2`) | `interleave` | 351.85 / 1370.1 `[350.67--355.83T]` | 454.21 / 2055.5 `[453.08--464.25T]` | R(resource) |
| W8 (`2x4`) | `2stage_0` | n/a | 424.12 / 1853.2 `[423.10--430.11T]` | R(resource) |
| W8 (`2x4`) | `2stage_prio` | n/a | 423.20 / 1847.9 `[422.43--429.30T]` | R(resource) |
| W8 (`2x4`) | `2stage_barrier` | n/a | 327.10 / 1240.3 `[325.50--330.97T]` | R(resource) |
| W8 (`2x4`) | `interleave` | n/a | 402.00 / 2332.7 `[401.01--407.67T]` | R(resource) |

最优：W4 (`2x2`) / `2stage_0` / Q2，493.62 TFLOPS / 2134.5 PMC总GB/s
`[492.76--499.38T]`。

## 结论

生产达到ceiling的范围为79.95%--92.82%。Hy3、Xiaomi和H3均超过92%；两个Qwen K=256
路径分别达到91.27%和87.91%。两个K=512 `default`路径只有81.78%和79.95%，且处于
204V+4A资源档位，是后续优先优化对象。

对应生产布局的七个正式ceiling点为`1609.2--2657.2 PMC总GB/s`；97个wave/调度/Q
扫描点为`899.2--3409.0 PMC总GB/s`。这些值来自PMC报告的物理fetch/write流量及各自
kernel时间，不是动态VMEM指令流量。生产侧因原编译环境已被覆盖，本轮没有可比PMC值。

ceiling省略了真实kernel的VMEM到MFMA RAW、LDS搬运、scale、metadata和epilogue，因此
吞吐差距本身不能定位具体瓶颈；归因仍需PMC或ATT。完整MoE路径和最终path选择见
[MAIN_MERGE_PERFORMANCE_REPORT.md](../../../contrib/moe/MAIN_MERGE_PERFORMANCE_REPORT.md)。
