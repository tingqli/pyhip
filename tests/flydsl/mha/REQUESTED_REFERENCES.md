# 用户指定的BF16 / FP8性能参考（2026-09-07）

本页是**最新性能验收要求**，替代此前MI325报告使用的性能比较对象。
不覆盖或重标旧JSON，也不修改固定原版保留/逐位测试的历史来源。

## 比较对象与门槛

| dtype | 当前被测实现 | 用户指定参考 | 同机延迟门槛 |
|---|---|---|---|
| BF16 | [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | [test_attn_8wave_32x32_lkgv.py](../test_attn_8wave_32x32_lkgv.py) 的dense `MHA` | “基本一致”暂明确为 `current/reference <= 1.05`；5%是工具显式默认，非用户给出的精确百分比 |
| FP8 | [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | [pa_prefill_8w32x32.py](../pa_8wave/pa_prefill_8w32x32.py) 的FP8 `PagedAttention` 分支 | `current/reference <= 1.00`，不得更慢 |

两者使用本机**MI325X/gfx942/304CU、相同编译器/输入/硬件策略**，没有PTL设置，
不采用MI308X历史400T或其它绝对数字作门槛。必须先通过独立FP32检查、各自重复逐位检查，
再比较相同计时口径。BF16与参考的lazy-softmax阈值不同，不要求跨算法逐位相同，也不放宽现有精度。

源码SHA256：

- BF16参考：`c0880420cd10a797c59087d4f73e942aa237020c3f0793b2fa0bcd9a6ac0776a`。
- FP8参考：`620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849`。
- [compare_requested_references.py](compare_requested_references.py) 每次核对参考文件SHA，
  同时记录FP8驱动、计时器、helpers和当前kernel SHA；参考或候选变动不能继续用旧标签判pass。

## 对齐输入，不偷换shape

### BF16 dense参考

原 `main()` 默认 `H=8、D=128、M=N=256×CU`。因此MI325默认是：

**B1 / Hq=Hkv=8 / Dq=Dv=128 / Q=KV=77824 / full noncausal**。

参考Q/K为head-major dense，V预shuffle为 `[H,N/8,D,8]`，无GQA/causal/per-token descale接口。
当前版使用同一逻辑Q/K/V转成paged形式（默认page32），page order为identity、descales为1。
所有transpose/shuffle/vectorize在计时外；当前paged的counter分配/初始化仍在计时内，不隐藏辅助工作。

另外两个明确扩展对照为Q10240/KV2560、Hq=Hkv=8、D128、page32/64，用于隔离paging开销。
**不拿H16/HK1、D192/V128或不对齐的KV2583冒充dense参考支持的相同工作负载**。
默认77824不能因为内存/耗时大就静默缩短；较短case不替代默认case验收。

### FP8 paged参考

实现文件本身没有性能main；配套
[test_pa_prefill.py](../pa_8wave/test_pa_prefill.py) 的最后有效赋值是 `per-tensor`，main为：

**B1 / H16 / HK1 / D192 / V128 / Q=KV32768 / page64 / causal / per-tensor**。

两边均输入实际BF16随机数经FNUZ量化得到的Q/K/V，共享相同q/k/v descale、反序页表和预分配输出。
不会拿BF16分支替代FP8，也不再使用先前nested native-BN64源码作参考。
补充D128/H8与D192/H16的Q10240/KV2560 NC × 两种Q scale，以及causal32K/per-token，
共6个FP8 case。当前FP8的page64/V128约束保留，参考更多page/Dv配置不伪装成当前支持。

## 计时与TFLOPS

这是**同机配对协议**，不是不加说明地“原脚本逐字复刻”：

- 保留两边相同的原 `pyhip.cudaPerf` event interval。其GPU delay在start event之前，不计入attention时间。
- BF16：10套独立随机输入、2次warmup；每轮10样本，延迟取排序后的第6个（upper median）。
  原脚本分别排序µs和TFLOPS，两个中间项并非严格倒数；本工具TFLOPS统一从所报告延迟计算。
- FP8：遵循 `run_perftest` 的2次warmup/10样本、最多12份同内容独立buffer及4e9字节复制上限，
  每轮用均值。输入量化及buffer复制不计时。
- 两边逐样本交替先后顺序；重复5轮后取轮次统计中位数。该5轮配对扩展明确标
  `matched_pair_protocol=true`、`reference_standalone_protocol_exact=false`。
- gate基于配对的event总延迟（含所有辅助dispatch/间隙）；不会以当前attention-only对参考端到端时间。
- 报告包含双方µs/TFLOPS、全部raw samples、完整shape/协议、source和逐case延迟比。
  causal FLOPs计精确可见三角形，不采用原驱动简单“full FLOPs除2”的近似。
- 性能通过仅指**选中的case集合**，不是所有形状/架构普遍通过。GPU进程隔离检查仍不等于外部独占预约。

## 已完成与阻塞

- 新对照入口、CPU-only计划、hash/gate/计时统计/adapter/失败闭锁测试已实现。
- 最新计划在
  [plan-final.json](results/requested_references_20260907T062426Z/plan-final.json)，9个case，
  `records=[]`、`acceptance_passed=false`，不是实测结果。
- [CPU契约记录](results/requested_references_20260907T062426Z/contracts-final.xml)
  **54项通过**，为离线工具验证，不算kernel通过或性能达标。
- 新健康预检再次在amd-smi读取超时，
  [health-preflight.json](results/requested_references_20260907T062426Z/health-preflight.json)保留具体错误；
  **未导入torch/FlyDSL，未运行任何参考kernel**，预检自己的读取进程已清理。
- 两个生产kernel及两个参考文件都没有修改。旧MI325 BF16 384T/386T等数字不是这里的dense参考验收，
  FP8也没有新结果。因此**BF16基本一致、FP8不低于参考目前均未证明**。

原生执行入口支持 `--backend bf16` / `--backend fp8`、`--case` glob；纯计划要求显式
`--compute-units 304 --list-cases`。`--preflight-only`只做限时只读检查，不会进入native。
需管理员确认设备/驱动健康后，才执行完整对照并据结果优化。未获新授权不reset、不commit、不push。