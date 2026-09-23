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
没有新增 pytest 命令行配置框架；新增 tile 时直接增加普通参数化用例。

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
| `fly_splitk_2s` 的 batch/启发式分支 | `fly_decode` / `fly_prefill`，按 direct、sorted、prefill 明确拆开 |
| SiTUv2 / SwiGLU / MXFP4 布局 | 保留专门的参数化用例，不在一个 test 中改写 dtype 列表 |
| 8x1 六种 K 和两种量化 | `test_fly_down_paths`，扩展为四条 Down 路径 |
| 手动模型性能循环 | `test_model_prefill_perf`；Aiter 对照仍使用原 benchmark |

旧 FP8 split-K 的 `block` 参数曾遗漏传给执行函数，实际重复测试 PTPC；
新用例明确生成 block-scale 权重。旧性能阶段还可能使用未 shuffle 的副本；
新测量复制同一组准备好的 tensor 并校验各副本，不保留这些错误行为。
历史运行记录保持不变；不把新计时与旧脚本的不同边界直接比较。