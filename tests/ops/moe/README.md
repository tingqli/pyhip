# 固定 MoE kernel 的 pytest

[test_moe.py](test_moe.py) 测试显式指定的 FlyDSL/ASM 完整 MoE 路径，
取代旧的 `entry_common()`、私有 buffer 池、后端别名和手动主循环。
每个 batch、dtype、布局是独立 pytest node；不读取 autotune winner，
不调用 `_configs()`，失败时不会悄悄改选其它实现。

[test_tuned_moe.py](test_tuned_moe.py) 保留调优策略、缓存、benchmark、
特殊数值和 graph/多设备边界回归。固定 kernel 的覆盖不随候选策略变化。

## 依赖接口

当前基准为 FlyDSL 0.3.0 和 Aiter `cdfd46fcd`，直接使用 SiTUv2、原生 buffer
描述符、命名 waitcnt 和 autotune artifact，不再提供旧版接口回退。
为兼容尚无 `output` 参数的 Aiter，保留 `moe_sorting` / `fused_moe` 的输出清零/复制适配。
`stage2_scatter` 和三项激活量化参数按原生签名转发；旧版不支持的 `None` 默认值会被移除，
显式非 `None` 值仍由原生 Aiter 报错，不会被静默忽略。
硬件架构限制与软件版本无关，继续保留对应跳过规则。

## 运行与选择

在仓库根目录使用默认 ROCm Python：

```bash
# 只收集，不初始化 GPU；查看可选 node id
python3 -m pytest tests/ops/moe/test_moe.py --collect-only -q

# 指定 ASM split-K 的一个精度和实际 batch
python3 -m pytest 'tests/ops/moe/test_moe.py::test_asm_splitk[ptpc-m17]' -q

# 固定 FlyDSL sorted decode / direct，或特定 Down 路径
python3 -m pytest tests/ops/moe/test_moe.py -k 'test_fly_decode or test_fly_direct' -q
python3 -m pytest tests/ops/moe/test_moe.py -k 'test_fly_down_paths and compact and i320' -q

# 性能单独选择；默认不运行 perf
python3 -m pytest tests/ops/moe/test_moe.py -m perf -k 'test_force_batch1_path_perf and bf16' -s
python3 -m pytest tests/ops/moe/test_moe.py -m perf \
  -k 'test_model_prefill_perf and qwen35_35B_k256 and compact and m1024' -s \
  -o junit_family=xunit1 --junitxml=/tmp/moe-kernel-perf.xml

# 全部固定 kernel 正确性用例，包括完整 batch sweep
python3 -m pytest tests/ops/moe/test_moe.py -q
```

ASM split-K 保留原脚本的实际 batch 集合：2–63、128–255、256/512/768、
2048/4096/6144/8192、6144–6399，去掉重复的6144。它们现在分别显示结果，
因此默认用例数量比旧脚本的单 node 内循环多。可用 node id 或 `-k` 做定向验证。
当前精度矩阵是 BF16、PTPC、MXFP4；block-scale 的 A1×128 激活语义由
`test_jit_blockscale` / `test_fly_blockscale` 验证。公共调优同时允许仅量化权重的 block `jit_splitk`，
但这不改变本文件固定 split-K 测试的精度矩阵，也不把两种数值流程视为等价。
没有新增 pytest 命令行配置框架；新增 tile 时直接增加普通参数化用例。

### jit_blockscale：小 shape 回归与大 shape 优化

两个测试函数复用普通 helper `_run_jit_blockscale`，不混合 shape 参数或使用参数级标记：

- `test_jit_blockscale`：小 shape 回归，不加 `perf` 标记。
- `test_jit_blockscale_perf`：大 shape 优化，函数上加 `@pytest.mark.perf`。

仓库默认的 `-m "not perf"` 会排除大 shape，不准备其输入，也不运行 kernel。

| 选择 | H / I_tp / E / topk | M |
|---|---|---|
| 默认小 shape | 1024 / 256 / 8 / 4 | 1、17、257 |
| `-m perf` 大 shape | 4096 / 256 / 512 / 10 | 8192、16384 |

大 shape 复用 `qwen35_397B_k256` 的维度，但本测试始终使用 FP8 block-scale
量化，不使用模型预设的 PTPC。两组都固定执行所选 tile / Down 路径，检查
正确性并打印耗时、TFLOPS 和 diff；没有 autotune 或 fallback。

```bash
# 普通回归：只运行小 shape
python3 -m pytest tests/ops/moe/test_moe.py::test_jit_blockscale -s

# 优化阶段：显式选择大 shape，不运行小 shape
python3 -m pytest tests/ops/moe/test_moe.py::test_jit_blockscale_perf -m perf -s

# 大 shape 固定为 M256/N64 persistent Down；选择其中一个 batch
python3 -m pytest tests/ops/moe/test_moe.py::test_jit_blockscale_perf -m perf \
  -k m8192 -s
```

### fly_blockscale：FlyDSL GateUp + M128／8wave Down

新增固定配置 `fly_blockscale`，不替换原 `jit_blockscale`：

- gfx950、BF16 输入、FP8 W128×128 / A1×128、separated SiLU，无 clamp。
- 两份权重必须预先使用 Aiter 16×16 shuffle；不在 runtime 改排权重。
- sort-M256、GateUp N256（融合后输出 128 列）。`default/persistent` 使用 Down M128/N128，I_tp、H 均为正的256倍数。
  `num_oc_splits` 在编译期拆分 H，每份须为正的 128 倍数。
  调优搜索 OC1/2/4/8 的合法配置和 independent/persistent 两种调度；固定用例还覆盖 OC3。
- Stage2 固定 `is_sorted_bm_256=True`，但输出保持 packed128。公开 `block_size_M=256`
  指排序粒度，不能因为 Down 计算 M128 就改为128。
- 使用真实排序容量启动；无效 block 在读取 expert ID 前跳过，不通过 `.item()` 截取。
  激活 scale 保持 K-block-major；两阶段之间仍调用 Aiter 量化，最后重建 inverse 并调用
  custom packed reduce，不使用 ASM Down 或 `torch.sum`。kernel 均位于 src 模块。
- Persistent 的 8 个 shard 与 OC 数分离；每次调用/graph 独占256个int32 queue words。
  各 shard 按实际任务数自复位，不共享实验入口的缓存 heads。
- Down K256保留原流水；K>256以固定双缓冲逐K256计算，逐K128 scale做FP32累加，
  全部K完成后才乘routing并转BF16。A scale物理布局为`[K/128,tokens*topk]`，
  B scale为`[E,H/128,K/128]`。不按K扩大LDS，也不在BF16中合并部分结果。
- 新 `down_path="8wave"` 使用 src 的 M256 routed Down，I_tp 支持正的128倍数，H仍为256倍数。
  调优搜索 N32/N64、OC1/2/4/8；每份 N 须为128倍数且至少三个 N tile。
  K 全驻留寄存器／四槽LDS，工厂按实际N/K检查160KiB LDS上限，不声称任意大K可用。
  N64在K640、N32在K1280起必定超限；固定测试还覆盖可用的N128 tile。
  输出为`[tokens,topk,H]`，最后预分配`torch.sum`，不使用 packed inverse/reduce。
  每次调用独占一个int32 counter，每次launch清零，清零包含在graph与计时中。

`test_fly_blockscale` 覆盖小 batch、随机/均衡/集中路由、长 K 和 XCD batch；
`test_fly_blockscale_k_multiples` 额外覆盖Down K512/768/1024/1792/4096的完整MoE及graph。
`test_fly_blockscale_8wave` 覆盖I128/256/384/512/640/768/896/1024/1152，
N32/64/128、OC1/2/3/4/8、小batch和tail；相同helper的
`test_fly_blockscale_8wave_perf` 单独测M8192/16384、I256/384/640完整路径。
`test_fly_blockscale_special_values` 覆盖零输入/scale/gate/up 和最后一个 K 分块。
`test_fly_blockscale_stage1_capacity_graph` 检查被投毒的容量尾部、valid0 和输出 guard；
`test_fly_blockscale_stream_graph` 检查非默认 stream、同一 graph 内双调用以及
输入/路由改变后的 replay；`test_fly_blockscale_concurrent_graphs` 检查两份 graph
同时 replay 时的队列隔离。候选约束、旧配置拒绝、实际调用 src Down/custom reduce
以及跨 token 数的调优缓存另在
[test_tuned_moe.py](test_tuned_moe.py) 中验证。
8wave同样覆盖非默认stream、双launch、独立graph；I128/384/640的调优缓存测试
必须实际选择`8wave`，K640必须选择N32。超LDS和不足三个N tile的配置在prune中拒绝。
block模式新增`block_down_8wave_revision=1`以重新比较新候选，其他量化模式key不变。

实验[Down/Reduce回归](../../../experiments/moe/flydsl/moe_8w_down/test_blockscaled.py)
还覆盖K1280/2048、两种sorting、奇偶K256循环次数、最后一个K128 scale，以及跨K256
相消的逐值精确结果；每次graph replay检查队列复位与输出guard。非法K（非正整数或
非256倍数）在M128工厂拒绝；8wave工厂要求正128倍数，另有TOPK130 signed slot、
容量尾部投毒、全padding／valid0／恢复、counter复位和输出guard回归。
M128另有K256缓存B scale导致的LDS边界测试；实验入口覆盖N512跳过BN64/OC4，
并检查报告保留PyHIP实际OC1及FlyDSL工厂配置。
K256快路径与原暂存源码的编译体保持不变。
`test_m128_k256_compiler_specialization`额外断言统一编译入口的K256返回原N流水的
同一个launcher，避免只验数值而遗漏流水退化；K2560另有回归，防止错误命中K256特化。

```bash
# 正确性、容量边界和 graph；默认排除 perf
python3 -m pytest tests/ops/moe/test_moe.py tests/ops/moe/test_tuned_moe.py \
  -k fly_blockscale -q

# 完整 MoE 计时，不是裸 GEMM1：与原 jit_blockscale_perf 使用相同模型维度
python3 -m pytest tests/ops/moe/test_moe.py::test_fly_blockscale_perf -m perf -s
```

新候选沿用独立 Torch 参考和 `calc_diff <= 0.02`；可编译、通过精度检查或被枚举，
都不代表一定比原候选更快。调优比较包含 sorting、两次量化、两个 GEMM，以及
所选Down路径的inverse/custom reduce或Torch sum。

### BF16 8-wave 与 A4W4 完整路径

- `test_jit_8wave` / `test_jit_8wave_persistent` 固定运行 BF16 SiLU，覆盖 raw/shuffled、
  M128/256、Down N128/256，以及 persistent K128/256、OC split1/2/4。
- `test_jit_mxfp4` 覆盖通用 `moe_gemm_mxfp4` Gate/Up 和专用
  `moe_gemm_mxfp4_gateup_4wave`，共用通用 Down；H 按1024对齐时使用 JIT
  `moe_gemm_final_reduce_bf16`，否则仅最后一步改用 `torch.sum`，不更换 GEMM。
  覆盖 H256/512/1536 的 fallback 与 H1024/2048 的 JIT reduce，
  M1/17/513 同时覆盖空 worker 和不均匀 token 分配；两种路径都检查 graph replay。
  测试的是完整 A4W4 MoE，不再保留逐阶段调试脚本或单独的 final-reduce 调试入口。
- `test_jit_8wave_perf` / `test_jit_mxfp4_perf` 分别是显式 `perf` 大 shape 入口；
  不通过参数级标记混合正确性组和性能组。

```bash
python3 -m pytest tests/ops/moe/test_moe.py -k 'test_jit_8wave or test_jit_mxfp4' -q
python3 -m pytest tests/ops/moe/test_moe.py::test_jit_8wave_perf -m perf -s
python3 -m pytest tests/ops/moe/test_moe.py::test_jit_mxfp4_perf -m perf -s
```

固定 A4W4 测试显式使用 `torch_reference(..., mxfp4_activations=True)`，
用 Torch 量化/反量化模拟两次激活量化，并保持 `calc_diff <= 0.02`。
这只是独立校验，不是运行候选或 fallback；公共 API 调优及 Aiter 对照仍使用默认
A16W4 参考，量化误差超过原门槛的 A4W4 候选仍被排除。

## 共享工具与测试边界

[pyhip.testing.moe](../../../src/pyhip/testing/moe.py) 提供：

- `prepare_moe()`：与 benchmark 相同的模型维度、生成器、量化和 shuffle；
  两份权重各自携带 `is_shuffled`，不做后端专属 padding。
- `torch_reference()`：独立 Torch 参考；不调用被测 GPU kernel，也不将 Aiter
  当作正确答案。完整 MoE 保持 finite、output identity 和 `calc_diff <= 0.02`。
- `make_moe_runner(config)`：只固定静态配置，tensor 每次从调用参数取得；
  不调优、不 fallback，不持有输入/指针，所以 buffer 轮换仍然有效。
- `check_output()` / `measure_moe()`：共享正确性检查与 `run_perftest` 计时，
  检查每个实际输出副本。benchmark 仍只负责 Aiter 与 tuned API 的对照，
  没有增加指定 kernel 的 CLI。

MXFP4 pytest 额外保留旧测试的 expert/row/K-group 幅度变化，覆盖非均匀
E8M0 scale；这不改变 benchmark 的默认输入。RTA/RTE 保持 kernel 的当前设置，
完整 MoE 参考和容差不随舍入模式放宽。

固定路径计时包含 sorting、quant、Gate/Up、Down、inverse/sum、内部 workspace
和 Python launch；**不是单个裸 Gate/Up 或 Down 的设备时间**。`perf` 用例先
编译和检查，再以2个副本、2次 warmup、10次 sample 测量；输出完整配置、
中位时延、有效 TFLOPS、diff，并通过 `record_property` 保存到 JUnit。
输出 JUnit 时使用 `-o junit_family=xunit1`，避免 xunit2 对 testcase 属性的兼容警告。
测试关闭 FlyDSL 磁盘编译缓存以验证当前源码，进程内编译对象与 fast launcher
仍会复用；autotune 配置缓存另有回归。不会删除用户缓存目录。
有效 FLOPs 为 `6*M*topk*H*I_tp`，不含无效 padding。没有自动清缓存、设备状态
门槛或性能通过阈值；计时是当前运行结果，不保证跨设备可比较。

## 旧用例迁移

| 旧路径 | 固定配置 / pytest |
|---|---|
| `16x32_2s_b1` | `jit_batch1` / `test_asm_batch1` |
| `16x32_2s_b` | `jit_batch` / `test_asm_batch`；loop-N 单独测试 |
| `mxn_splitk_2s` | `jit_splitk` / `test_asm_splitk` |
| `mxn_splitk_1s` | `jit_1stage` / `test_asm_one_stage` |
| BF16 SiLU 8-wave | `jit_8wave` / `test_jit_8wave`、`test_jit_8wave_persistent` |
| FP8 block-scale 8-wave | `jit_blockscale` / `test_jit_blockscale` |
| MXFP4 通用/专用 Gate/Up | `jit_mxfp4` / `jit_mxfp4_4wave` / `test_jit_mxfp4` |
| BF16 GELU | `jit_gelu` / `test_jit_gelu`；不保留 MXFP4 GELU Torch fallback |
| `fly_splitk_2s` 的 batch/启发式分支 | `fly_decode` / `fly_prefill`，按 direct、sorted、prefill 明确拆开 |
| SiTUv2 / SwiGLU / MXFP4 布局 | 保留专门的参数化用例，不在一个 test 中改写 dtype 列表 |
| 8x1 六种 K 和两种量化 | `test_fly_down_paths`，扩展为四条 Down 路径 |
| 手动模型性能循环 | `test_model_prefill_perf`；Aiter 对照仍使用原 benchmark |

旧 FP8 split-K 的 `block` 参数曾遗漏传给执行函数，实际重复测试 PTPC；
当前 block-scale 测试明确生成对应权重，并执行两次激活量化的 8-wave 路径。
旧性能阶段还可能使用未 shuffle 的副本；
新测量复制同一组准备好的 tensor 并校验各副本，不保留这些错误行为。
历史运行记录保持不变；不把新计时与旧脚本的不同边界直接比较。