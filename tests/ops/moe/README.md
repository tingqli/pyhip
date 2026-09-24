# 固定 MoE kernel 的 pytest

[test_moe.py](test_moe.py) 测试显式指定的 FlyDSL/ASM 完整 MoE 路径，
取代旧的 `entry_common()`、私有 buffer 池、后端别名和手动主循环。
每个 batch、dtype、布局是独立 pytest node；不读取 autotune winner，
不调用 `_configs()`，失败时不会悄悄改选其它实现。

[test_tuned_moe.py](test_tuned_moe.py) 保留调优策略、缓存、benchmark、
特殊数值和 graph/多设备边界回归。固定 kernel 的覆盖不随候选策略变化。

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
`test_jit_blockscale` 验证。公共调优同时允许仅量化权重的 block `jit_splitk`，
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