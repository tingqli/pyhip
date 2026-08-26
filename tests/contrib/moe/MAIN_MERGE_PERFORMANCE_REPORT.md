# MoE down 合并最终性能报告（2026-08-26）

## 范围与最终状态

本次合并基于 `hy3-single-n512-handoff@3591fd0`。公开兼容入口保留在
`src/contrib/flydsl/moe_gemm_splitk.py`；实现参考FlyDSL上游`kernels/moe`的分层，
拆分到`moe_gemm_2stage/`：`gemm.py`负责stage分发，`gemm1.py`负责gate-up，
`gemm2.py`按路径分发到`gemm2_{default,1x4,1x8,2x4}.py`，其余公共能力位于
`{layout_helpers,moe_reduce,quant,common}.py`。新子包与FlyDSL上游一致，只公开
`compile_moe_gemm1`、`compile_moe_gemm2`和`compile_moe_reduction`三项keyword-only
缓存API；原`moe_gemm_splitk.py`保留兼容入口。重构前算法基线源码
SHA256为`efbf7b566968f2ce69696e05d4334370521454a63185185b32632624144d459b`；
三条专用down路径均与该基线逐bit一致。本轮正式矩阵直接绑定以下重构后源码：

| 文件 | SHA256 |
| --- | --- |
| `moe_gemm_splitk.py` | `92912d1d5c26470ec839fb8ae953d8691c9f2a2a142c3a2d9353ce33553da5e3` |
| `moe_gemm_2stage/__init__.py` | `7377ded7c4849dc903e4b5e18dcd0d61c071cfa14ae9ef2cfc173d9e93e1ae6b` |
| `moe_gemm_2stage/common.py` | `85e2153dce37859947577223959f7958b78e8147441e53d3ed4f4314d5fe4f0a` |
| `moe_gemm_2stage/gemm.py` | `1e83f80c82a5cff6accbdb80d3126072ebe8a6e4f10439f949e02367823b405c` |
| `moe_gemm_2stage/gemm1.py` | `4af72f253fe0bc806c4233a22745bbf3bfdc401269e0bbc743ead24cb542ccc4` |
| `moe_gemm_2stage/gemm2.py` | `e295f027eef3f0d2698b05920c572d2df3984742278aa55c899da6e908b8d341` |
| `moe_gemm_2stage/gemm2_1x4.py` | `075f8c5e2cb6223c2ebeeb6e77806f0d04c18b7c63063cfd88d6e478bf7e021a` |
| `moe_gemm_2stage/gemm2_1x8.py` | `42076f1e9eb937947feccb449e17c2c4c559aed21f8eb3f99acb44e074bf3869` |
| `moe_gemm_2stage/gemm2_2x4.py` | `659dd2b0dd3cd1fc08e7498c10e292d64ea721fcdfadeb00083eff68107cc971` |
| `moe_gemm_2stage/gemm2_default.py` | `afa57a758e552e91ea3a20da091f7d309f63559cb39cd477c44c5dedcb67ee6b` |
| `moe_gemm_2stage/layout_helpers.py` | `a72dc9ac9caa74b54d6027418ad5b9e68833cafda5a7141d2767d9b74018a42c` |
| `moe_gemm_2stage/moe_reduce.py` | `743dfe9dc1c4e1bade0a3771ff375ed183f5539090f713bc9aad112674136aa2` |
| `moe_gemm_2stage/quant.py` | `6876e2299806e2a0b9615bde122c3a9dcb6583b82bb4019532e7185aa0a55698` |

当前公开 `down_path` 及固定拓扑为：

| `down_path` | Kernel | 拓扑 |
| --- | --- | --- |
| `1x4_64x256` | `moe_2stage_down_prefill_1x4_64x256` | BM64、BN256、4 waves |
| `2x4` | `moe_2stage_down_prefill_2x4` | BM128、BN256、两个独立4-wave子组 |
| `1x8` | `moe_2stage_down_prefill_1x8` | BM64、BN512、8 waves |
| `default` | 原有路径 | Qwen K=512生产配置保持不变 |

旧路径名不保留兼容别名。`TILE_M_DOWN`控制sorting与down任务覆盖范围，
`TILE_M_GATEUP`控制gateup任务大小；只有`2x4`允许down BM128与gateup BM64组合，
由kernel直接消费原始M128 metadata并展开两个gateup任务。

## 生产配置

性能case均显式指定tile、路径和padding，不调用自动down selector。

| Case | B / Hidden / Inter-TP / E / TopK | Quant | `TILE_M_DOWN / TILE_M_GATEUP / TILE_N` | `down_path` | Padding |
| --- | --- | --- | --- | --- | ---: |
| Hy3 | 32768 / 4096 / 192 / 193 / 9 | per-tensor | 64 / 64 / 128 | `1x8` | 0B |
| Qwen3.5 397B K=512 | 32768 / 4096 / 512 / 512 / 10 | PTPC | 64 / 64 / 256 | `default` | 默认 |
| Qwen3.5 397B K=256 | 32768 / 4096 / 256 / 512 / 10 | PTPC | 64 / 64 / 256 | `1x4_64x256` | 128B |
| Qwen3.5 35B K=512 | 32768 / 2048 / 512 / 256 / 8 | PTPC | 64 / 64 / 256 | `default` | 默认 |
| Qwen3.5 35B K=256 | 32768 / 2048 / 256 / 256 / 8 | PTPC | 64 / 64 / 256 | `1x4_64x256` | 128B |
| Xiaomi | 32768 / 6144 / 256 / 384 / 8 | PTPC | 64 / 64 / 256 | `1x4_64x256` | 128B |
| H3 | 32768 / 6144 / 384 / 128 / 4 | PTPC | 128 / 64 / 256 | `2x4` | 0B |

`Inter-TP`为tensor-parallel切分后的K维；`TILE_N`是host/gateup配置，专用down
kernel的固定BN见上一节。两个Qwen3.5 35B case使用TP1，其余均使用TP8。

## 测试协议

- GPU：AMD Instinct MI308X / gfx942，80 CU。
- 1800MHz performance determinism；PTL `Enabled / VECTOR,F8`；650W power cap。
- 10组buffer轮换，避免固定地址偏置；同进程平衡ABBA顺序。
- 最终path矩阵使用ABBA12；接近边界的case升至ABBA48。每轮先对同一版本的两个
  样本求均值，再计算同轮提升率，最后取各轮提升率中位数。
- ABBA48中候选相对当前或统一路径的配对提升率IQR跨0时视为持平，不因微小绝对
  中位数差异切换路径。
- phase包括`down`、`down + sorted_sum`（Combined）和完整链（Full）；完整链内包含gateup。
- `提升率 = 1 - candidate / control`；正值表示candidate更快，负值表示回退。
- 每轮检查reduced输出、finite、inactive tail及padding；测试后恢复performance
  level、PTL与NUMA状态。

## 1K-32K最终path矩阵

下表是七个正式case的最终选择，全部使用上节源码重新执行。主矩阵使用ABBA12；
Qwen 397B K=512 1K、Xiaomi 4K和H3 8K再以ABBA48复核，并以长采样结果覆盖表中
对应行。
ABBA48中Qwen 397B 1K与Xiaomi 4K的候选对比IQR均跨0，按保守tie-break分别选择
`default`和同模型其余Batch统一使用的`1x4_64x256`；H3 8K的`2x4`相对`1x4`
Down/Combined提升率IQR均大于0，保留`2x4`。

`Down`列为“绝对中位延迟ms / 有效TFLOPS / 相对同轮default的配对提升率”；
`Full`列为“绝对中位延迟ms / 配对提升率”；`Combined`为down与`sorted_sum`的绝对
中位延迟。完整链包括sorting、两次activation quant、gateup、down、invert和
`sorted_sum`。Down有效FLOPs按`2 * B * TopK * Hidden * Inter-TP`计算；default相对
自身的提升率固定为0。

| Case | Batch | Path | Down ms / TFLOPS / 提升率 | Combined ms | Full ms / 提升率 |
| --- | ---: | --- | ---: | ---: | ---: |
| Hy3 K=192 | 1K | `1x8` | 0.1185 / 122.3 / 12.9% | 0.1534 | 0.4020 / 1.2% |
|  | 2K | `1x8` | 0.1794 / 161.6 / 15.5% | 0.2392 | 0.5773 / 4.0% |
|  | 4K | `1x8` | 0.2354 / 246.3 / 20.5% | 0.3381 | 0.8314 / 7.3% |
|  | 8K | `1x8` | 0.4052 / 286.2 / 20.5% | 0.5894 | 1.4312 / 7.0% |
|  | 16K | `1x8` | 0.7219 / 321.3 / 20.1% | 1.0935 | 2.6358 / 7.6% |
|  | 32K | `1x8` | 1.3530 / 342.8 / 19.7% | 2.0617 | 5.1662 / 6.9% |
| Qwen3.5 397B K=512 | 1K | `default` | 0.5410 / 79.4 / 0.0% | 0.5758 | 1.3339 / 0.0% |
|  | 2K | `default` | 0.5361 / 160.2 / 0.0% | 0.5989 | 1.4366 / 0.0% |
|  | 4K | `default` | 0.8130 / 211.3 / 0.0% | 0.9362 | 2.3437 / 0.0% |
|  | 8K | `default` | 1.1845 / 290.1 / 0.0% | 1.3943 | 3.4159 / 0.0% |
|  | 16K | `default` | 1.8200 / 377.6 / 0.0% | 2.2202 | 5.6234 / 0.0% |
|  | 32K | `default` | 3.3758 / 407.1 / 0.0% | 4.2248 | 10.9481 / 0.0% |
| Qwen3.5 397B K=256 | 1K | `1x4_64x256` | 0.2793 / 76.9 / 23.5% | 0.3136 | 0.6917 / 11.6% |
|  | 2K | `1x4_64x256` | 0.2836 / 151.4 / 23.5% | 0.3483 | 0.7830 / 10.2% |
|  | 4K | `1x4_64x256` | 0.4594 / 187.0 / 27.0% | 0.5942 | 1.3665 / 10.4% |
|  | 8K | `1x4_64x256` | 0.6508 / 264.0 / 26.6% | 0.9124 | 2.0510 / 9.4% |
|  | 16K | `1x4_64x256` | 1.0118 / 339.6 / 25.5% | 1.4830 | 3.4875 / 8.5% |
|  | 32K | `1x4_64x256` | 1.8516 / 371.1 / 26.1% | 2.7932 | 6.6118 / 8.7% |
| Qwen3.5 35B K=512 | 1K | `default` | 0.1466 / 117.2 / 0.0% | 0.1597 | 0.3768 / 0.0% |
|  | 2K | `default` | 0.1498 / 229.4 / 0.0% | 0.1715 | 0.4384 / 0.0% |
|  | 4K | `default` | 0.2300 / 298.8 / 0.0% | 0.2729 | 0.7233 / 0.0% |
|  | 8K | `default` | 0.4020 / 341.9 / 0.0% | 0.4847 | 1.3132 / 0.0% |
|  | 16K | `default` | 0.7310 / 376.0 / 0.0% | 0.9042 | 2.4840 / 0.0% |
|  | 32K | `default` | 1.3789 / 398.7 / 0.0% | 1.7301 | 4.8838 / 0.0% |
| Qwen3.5 35B K=256 | 1K | `1x4_64x256` | 0.0808 / 106.3 / 24.1% | 0.0960 | 0.2329 / 12.0% |
|  | 2K | `1x4_64x256` | 0.0835 / 205.6 / 22.9% | 0.1099 | 0.2784 / 5.8% |
|  | 4K | `1x4_64x256` | 0.1378 / 249.3 / 24.1% | 0.1867 | 0.4564 / 6.7% |
|  | 8K | `1x4_64x256` | 0.2276 / 302.0 / 23.6% | 0.3206 | 0.8196 / 6.8% |
|  | 16K | `1x4_64x256` | 0.4136 / 332.3 / 20.9% | 0.5916 | 1.5567 / 7.4% |
|  | 32K | `1x4_64x256` | 0.7815 / 351.7 / 20.5% | 1.1115 | 2.9706 / 7.5% |
| Xiaomi K=256 | 1K | `1x4_64x256` | 0.3048 / 84.5 / 26.9% | 0.3469 | 0.7627 / 12.6% |
|  | 2K | `1x4_64x256` | 0.3047 / 169.2 / 26.6% | 0.3801 | 0.8717 / 11.4% |
|  | 4K | `1x4_64x256` | 0.5133 / 200.8 / 29.2% | 0.6582 | 1.5206 / 12.3% |
|  | 8K | `1x4_64x256` | 0.7328 / 281.3 / 27.1% | 1.0068 | 2.2770 / 10.9% |
|  | 16K | `1x4_64x256` | 1.2865 / 320.5 / 28.1% | 1.8323 | 4.2370 / 10.9% |
|  | 32K | `1x4_64x256` | 2.2809 / 361.5 / 26.2% | 3.2848 | 7.7683 / 9.6% |
| H3 K=384 | 1K | `1x4_64x256` | 0.1623 / 119.1 / 15.4% | 0.1871 | 0.4116 / 6.4% |
|  | 2K | `1x4_64x256` | 0.1643 / 235.2 / 12.8% | 0.2041 | 0.4916 / 4.6% |
|  | 4K | `1x4_64x256` | 0.3018 / 256.2 / 8.1% | 0.3851 | 0.8622 / 3.2% |
|  | 8K | `2x4` | 0.5164 / 299.4 / 5.3% | 0.6645 | 1.5583 / 1.5% |
|  | 16K | `2x4` | 0.8649 / 357.5 / 7.6% | 1.1458 | 2.7934 / 2.5% |
|  | 32K | `2x4` | 1.5674 / 394.6 / 9.1% | 2.1196 | 5.3204 / 4.3% |

最终选择因此不是单一全局path：Hy3全Batch选`1x8`，Xiaomi全Batch选
`1x4_64x256`；H3在1K-4K选`1x4_64x256`、8K-32K选`2x4`；两个Qwen K=512
case保持`default`，两个Qwen K=256 case全Batch选`1x4_64x256`。Qwen3.5 397B
K=256的Down提升率为23.5%-27.0%，Full提升率为8.5%-11.6%；Qwen3.5 35B K=256
的Down提升率为20.5%-24.1%，Full提升率为5.8%-12.0%。H3 8K的`2x4` Down和
Full提升率分别为`5.3%`和`1.5%`。

## 专用down路径运行时拓扑映射

`1x4_64x256`、`2x4`和`1x8`共享`_map_down_task`，不包含Batch、E、N、K或TopK
专用映射常量。host编译时通过当前device name识别MI308；只有MI308启用4 XCC、
每XCC 4 SE、每SE 5 CU的topology特化。generic连续分段数也由设备决定：MI308为
4 XCC，非MI308为8 XCD；非MI308不生成topology `gpu.func`，只保留generic kernel。

MI308 topology kernel从sorting有效行数运行时计算：

```text
valid_tasks     = ceil(sorting_valid_rows / task_rows)
tasks_per_se    = floor(valid_tasks / 16)
mapped_tasks    = tasks_per_se * 16
short_cu_tasks  = floor(tasks_per_se / 5)
long_cu_count   = tasks_per_se % 5
```

其中`1x4_64x256`和`1x8`的`task_rows=64`，`2x4`的`task_rows=128`。
完整映射依次完成XCC连续分段、XCC内SE分段及每SE的5-CU ragged列转置；不能均分
到16个SE的尾部保持identity。generic映射按设备做4-way或8-way连续分段，不能完整
均分的尾部保持identity。两种映射对任意非负任务数都是双射。

`1x4_64x256`固定使用generic。MI308上的`2x4`要求精确padded task数至少为80；
`1x8`要求精确padded task数位于闭区间`[160, 2880]`，即每CU `[2, 36]`个任务。
host使用`task_num`及`M * TopK`做保守预选，最终由`_map_down_task`读取
`p_num_valid_ids[0]`，按对应`task_rows`计算sorting/expert padding后的精确任务数；
device端结果是topology/generic选择的权威门禁。

### 1x4_64x256 topology对比

control为当前重构package的生产generic映射，源码集合SHA256为
`f37254683b2cc4778b5628bf8dcf20ceeb437ce038dac136066fdd4a339a1888`；candidate集合
SHA256为`1e60d160f847b3dc677ea4137e07c704d2e5bba24122d2e590c6ce438efee94a`，唯一差异是
`gemm2_1x4.py`将`_map_down_task(..., False, ...)`改为`True`，该文件的control/candidate
SHA256分别为`075f8c5e2cb6223c2ebeeb6e77806f0d04c18b7c63063cfd88d6e478bf7e021a`和
`6434b6267d0538df36cdd298421172f3690f4f6e69ac60eeab50d4914852436f`。

四个生产配置均使用10组buffer和ABBA12。表格每格依次为
“Down提升率 [IQR] / Combined提升率 [IQR]”；正值表示topology更快，负值表示回退。

| Batch | Xiaomi K=256 | Qwen 397B K=256 | Qwen 35B K=256 | H3 K=384 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | +1.4% [-1.4%, +1.9%] / +0.7% [-1.5%, +1.5%] | +4.4% [+3.6%, +5.5%] / +6.5% [-4.5%, +21.4%] | +4.0% [-3.7%, +7.7%] / +3.0% [-3.2%, +4.6%] | -2.1% [-5.1%, -0.3%] / -1.6% [-3.6%, -0.8%] |
| 2K | -0.0% [-1.6%, +1.4%] / +0.8% [-1.3%, +1.5%] | +4.6% [+4.4%, +5.5%] / +4.0% [+3.5%, +4.5%] | +2.1% [-4.3%, +8.5%] / +1.8% [-2.7%, +4.6%] | -1.7% [-4.5%, -0.1%] / -0.8% [-1.5%, -0.3%] |
| 4K | -1.2% [-1.6%, +0.3%] / -0.9% [-1.8%, +0.2%] | +0.0% [-1.5%, +3.4%] / +0.2% [-1.2%, +2.1%] | +1.8% [+0.4%, +3.2%] / +1.5% [-0.1%, +2.9%] | +0.2% [-0.3%, +0.6%] / +0.2% [-0.3%, +0.4%] |
| 8K | -1.3% [-2.7%, +0.4%] / -0.9% [-2.2%, +0.7%] | +1.0% [-1.5%, +4.1%] / +0.8% [-0.5%, +1.6%] | +1.3% [-0.1%, +2.2%] / +0.9% [+0.0%, +1.1%] | +0.2% [-0.2%, +0.4%] / +0.2% [-0.2%, +0.4%] |
| 16K | -1.2% [-2.7%, +1.1%] / -1.2% [-2.5%, +0.9%] | +2.6% [+1.7%, +3.9%] / +1.8% [+0.0%, +2.2%] | +3.2% [+1.9%, +4.4%] / +1.5% [-2.6%, +4.4%] | -1.1% [-1.6%, -0.6%] / -1.4% [-1.8%, -0.6%] |
| 32K | +0.2% [-0.4%, +0.5%] / -0.2% [-0.3%, +0.1%] | -0.0% [-0.6%, +1.2%] / -0.6% [-0.7%, -0.0%] | +0.6% [+0.3%, +1.3%] / +0.6% [+0.3%, +1.0%] | -1.3% [-1.5%, -1.0%] / -1.2% [-1.3%, -1.0%] |

topology存在局部收益，但不能作为`1x4_64x256`的统一策略。相同的2,048个最小活跃
任务下，Qwen 35B 16K的Down提升3.2%，Xiaomi 16K回退1.2%，H3 32K回退1.3%；
收益显然不只由任务密度决定。Qwen 397B在1K/2K稳定提升约4%-5%，但其余Batch大多
中性；为避免引入模型或shape专用门禁，生产路径继续固定使用generic映射。

当前生产实例资源如下，均为0 scratch：

| 路径 | 指令数 | VGPR / SGPR / LDS |
| --- | ---: | --- |
| `1x4_64x256` | 930 | 250 / 96 / 25,600B |
| `2x4` generic | 1,092 | 256 / 96 / 65,536B |
| `2x4` topology | 1,119 | 256 / 96 / 65,536B |
| `1x8` generic | 490 | 128 / 96 / 28,672B |
| `1x8` topology | 521 | 128 / 96 / 28,672B |

资源使用当前重构源码和`COMPILE_ONLY=1` fresh dump；三份最终ISA SHA256依次为
`8a4515d8e7a5545e284780657d1fde8460fd0cdfa5be549cb141a0db66c56aa0`（1x4）、
`435dbb14bd0ab5601594e9536cf730512f229d3b9c0cd83711b88af94639f1fc`（2x4）和
`bb507d0b1df5856df45745aba516e9b0917e9f48657c9e17abda986bccba6525`（1x8）。
指令数按每个函数体内非标签、非directive的机器指令行统计；`_0`为先生成的generic，
`_1`为topology。


## 正确性与结论

- 全部最终path矩阵的输出均finite；Down最大relative-L2为
  `2.065696389763616e-05`，inactive tail及padding均保持未写，完整链结果一致。
- generic/topology映射整数模型对`valid_tasks=0..10000`穷举无重复或漏项；不能完整
  分段的尾部保持identity。
- `1x4_64x256` topology在Xiaomi、两个Qwen K=256和H3共24组A/B中均逐bit一致、
  `rel_l2=0`、finite、tail clean且padding clean；性能收益依赖模型shape，未进入生产。
- `2x4`与`1x8`本轮均完成fresh compile；正式矩阵实际执行了两者的
  generic/topology运行时门禁，所有结果finite且满足relative-L2阈值。
- MI308精确门禁覆盖padding跨界：`1x8`的原始/精确任务数`100/160`进入topology，
  `2800/2881`在topology kernel内回退generic；两者均与强制4-XCD generic逐bit一致。
  专项JSON SHA256为`edde066551505877e1d70713e72eb89fa9cb466e48db898605a3469e4c4b776f`。
- 模拟非MI308配置下，`2x4`和`1x8`的8-XCD generic均与MI308 topology逐bit一致，
  `rel_l2=0`且finite；运行时专项JSON SHA256为
  `b8d2aecc66f59ff2912c201a519b690e05ecbcfdc227ab208d8635500030c104`。fresh IR各只包含一个
  generic `gpu.func`，且确认使用8-XCD分段；本轮fresh ISA SHA256分别为
  `68311777e8d8fcede30a8507d793a518563692cf10428217bee0feed2916a6a8`和
  `68c20585ded3ee2b92cd0e2a717a7a7c0bedcf6d16c2980f561e7fa5a9aeac14`。
  device name配置单测3组全部通过。
- `1x8`当前源码六点Down、Combined及Full矩阵均重新执行，输出finite、tail clean；
  相对default的最大relative-L2包含在上述全矩阵上界内。
- 本轮Hy3 metadata显示`1x8`精确有效行数从12,352增至296,448：1K-16K的精确任务数
  位于`[160, 2880]`并进入topology，32K为4,632个任务，在kernel内回退generic。
- fresh dump确认`1x8`包含490/521指令的generic/topology两个`gpu.func`，资源均为
  128 VGPR、96 SGPR、28,672B LDS和0 scratch；`2x4`两个特化资源均为256 VGPR、
  96 SGPR、65,536B LDS和0 scratch，对应1,092/1,119条指令；`1x4_64x256`为
  930条指令、250 VGPR、96 SGPR、25,600B LDS和0 scratch。
- 最终生产选择见1K-32K矩阵；MI308的`2x4`和`1x8`保留topology特化与运行时门禁，
  其余设备只生成8-XCD generic；`1x4_64x256`始终使用设备相关generic swizzle。
  Qwen K=512维持`default`，K=256选
  `1x4_64x256`。
- 性能实验结束后，GPU恢复为空闲，performance level、PTL和NUMA恢复原状态。
