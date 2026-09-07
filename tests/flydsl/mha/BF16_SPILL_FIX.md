# gfx942 BF16：消除大量scratch spill（2026-09-07）

> **性能复测后的重要更新**：第一版零scratch并不等于更快，实测D128曾回退约30%。已恢复
> lazy rescale及D128 PV/LDS重叠；[overlap-v5原版对照](results/bf16_overlap_v5_performance.json)
> 的D128 page64基本持平（+0.135%），D192 page64延迟降低15.44%。D192 page32仍慢10.95%，
> 不隐瞒该未解决项。以下均为最终overlap-v5版本，源码SHA以状态manifest为准。
> 本次已实际执行低负载性能诊断，PTL始终Disabled，**不是原文250T/410T验收**。

## 已完成的修复与证据

生产实现仍是 [mha_pa_bf16_942.py](mha_pa_bf16_942.py) 的**8-wave / BM256 / BN32 / persistent**，
K经LDS、V在寄存器；未换4-wave、未调用其他attention fallback，原round-half-up BF16约定不变。

| page64 / V128 / NC / 无LSE | 修改前D128 | 修改后D128 | 修改前D192 | 修改后D192 |
|---|---:|---:|---:|---:|
| VGPR | 256 | **246** | 256 | **252** |
| SGPR（metadata） | 106 | 106 | 106 | 106 |
| AGPR | 0 | 0 | 0 | 0 |
| Private bytes/thread | 104 | **0** | 176 | **0** |
| VGPR spill count | 25 | **0** | 43 | **0** |
| SGPR spill count | 2 | **0** | 76 | **0** |
| LDS bytes | 16384 | 16384 | 24576 | 24576 |

这不是只降低默认形状的metadata：已在CPU上对
**D128/192 × V128/192 × page32/64/128 × C/NC × LSE开关 × 两种Q scale模式 = 96项**
生成真实gfx942 ISA。全部 **Private=0、VGPR spill=0、scratch指令=0、AGPR=0**。
84项也没有SGPR spill；余下12项为NC/V128/LSE，每项有**2个SGPR到VGPR lane的转存**，
不是scratch访问，不隐瞒或称为所有寄存器完全零spill。

证据：

- [修改前默认资源](results/bf16_spill_baseline_compile.json)：CPU metadata-only编译准确复现此前GPU编译的104/176B。
- [最终per-token 48项](results/bf16_overlap_v5_full_compile.json)、
  [最终per-tensor 48项](results/bf16_overlap_v5_scalar_compile.json)，带源码SHA及96份保留ISA索引。
- [最终整合BF16功能选择](results/bf16_spill_v5_functional.xml)：**205 passed / 46 skipped / 0 failed**。
  跳过是其他backend专属能力、非连续布局及本进程缺AITER C++ runtime；未放宽数值门槛。
- 其中**24项原版逐位对照**全部通过，覆盖D/V/page/CNC；
  同输入含ragged和非单位scale，原版从origin/main固定SHA加载，独立FP32+重复逐位检查均通过。
- [CPU回归](results/bf16_spill_v5_cpu.xml)：**66 passed**，包括原55项性能契约，以及新增编译诊断、
  streamed-K顺序、监测触发/策略隔离和最终资源哈希检查。

205项由原BF16选择170项、24项逐位对照和11项spill工具CPU检查组成；另列66项CPU回归
与之有11项重叠，不把重复执行叠加成唯一用例数。此前中间版本的测试保留为历史。

## 已执行的低负载性能诊断

amd-smi监测确实观察到0–1%并自动启动了多次复测。当前PTL **Disabled**，sglang服务仍驻留，
故所有数字标记为**非独占诊断**；同进程原版/当前版、相同输入和原计时协议，输出逐位一致。

| workload | 原8-wave µs | 当前µs | 原TFLOPS | 当前TFLOPS | 延迟变化 |
|---|---:|---:|---:|---:|---:|
| D128/H16/Q10240/KV2583/page64 NC，profiler | 1881.097 | 1883.645 | 115.187 | 115.031 | +0.135%（基本持平） |
| D192，同上 | 2825.789 | 2389.412 | 95.848 | 113.353 | **−15.44%** |
| D128/H1/Q=KV40960/page32 NC，10buffer events | 6947.947 | 6891.747 | 123.633 | 124.641 | −0.81% |
| D192/H16/Q10240/KV2583/page32 NC，10buffer events | 2210.269 | 2452.229 | 122.540 | 110.449 | **+10.95%，仍待优化** |

原始记录：[results/bf16_overlap_v5_performance.json](results/bf16_overlap_v5_performance.json)。
page64为5轮profiler IQR均值的中位数；两个historical case严格沿用50个event样本中位数。

FP8也已在低负载队列执行：**1438.105µs /186.659T**，PTL Disabled；
[原始记录](results/low_load_spill_recheck_2/fp8_410t_gate.json)的gate正确标unmatched，
不是原PTL Enabled/F8下400T验收。没有为追数字干扰其他任务或修改硬件策略。

第一版无条件rescale/广泛调度fence导致D128约30%回退，已撤销；D192/page32的padding LDS、
address reuse和PV依赖等候选实测无改善，未合入。**本次解决了大量scratch spill，不宣称
所有shape性能都已恢复，也不宣称250T/410T复现成功。**

## 原因与改动

1. **uniform metadata未明确表达**：persistent ticket、query prefix、KV页表/尾长虽各lane相同，
   编译器仍使query bounds变成vector，从而给每个Q/O buffer load生成descriptor waterfall。
   用`readfirstlane`只标记真正uniform的metadata；Q per-token scale保持vector。
2. **大量只在后段使用的地址与mask被提前计算**：C-shuffle八个wave mask、两套swizzled LDS
   atom地址、V地址和tail mask跨整段attention存活。局部带side-effect的identity使其在使用点
   生成，随后`&511`恢复512线程的已知范围，避免产生通用有符号除法。
3. **P@V与下一组完整K过度重叠**：仅在D192/page64及page128需要的路径增加编译调度边界，
  D128/page32,64保留原重叠。D192/page32通过Q/scale地址在work-item入口重算，不强制串行。
  不删硬件等待/barrier。
4. **可选V192/LSE的额外寄存器压力**：V192按原32×32×8 MFMA reduction顺序逐16列加载K，
   不把全部K与96个O accumulator同时常驻；V192和D192+LSE把V load推迟到softmax之后。
   默认V128无LSE保留QK与V-load重叠。
5. **保留lazy rescale的性能**：多数路径只在`corr<1`时乘O；仅D192/page128使用每轮乘`corr`
  以避免该specialization溢出，`corr=1`时乘法精确。24项原版逐位对照验证数值不变。
  K/V退休值显式结束存活。
6. **mailbox地址与mask**：只在fetch-ticket位置计算，避免长期额外64-bit指针和mask对占用SGPR。

无全局FlyDSL monkeypatch、没有改时钟/功率/NUMA、没有增加LDS大小或AGPR需求。
局部调度改变会影响吞吐；**零scratch不自动证明速度更快**，上表包含实际收益和回退。

## GPU利用率监测与性能队列

用户最新要求是“观察gpu利用率，降下来之后立刻性能测试；解决BF16 spill无需等待gpu空闲”，
已取代上一轮暂不跑GPU的限制。

[watch_performance.py](watch_performance.py) 读取amd-smi原生metric流（2秒/次），连续两次≤1%
即触发串行 [recheck_performance.py](recheck_performance.py)：**先BF16原版/修复版，再FP8目标**。
最终监测已触发并结束：[results/gpu0_watch_overlap_v5.json](results/gpu0_watch_overlap_v5.json)。
没有留下无限期GPU benchmark或监测进程。

- 如果低利用率时sglang仍驻留：**只测当前PTL的非独占诊断**，明确标`isolated=false`、
  `diagnostic_only=true`，baseline checks为unmatched；绝不在其他任务驻留时修改PTL。
- `--when-low`始终当前PTL诊断，即使某一进程快照为空也不自动升级权限；曾观察到sglang重启
  在两个快照之间创建新进程。显式独占队列才按此前授权临时使用VECTOR,BF16/F8并finally恢复。
- 不终止用户或其他人的GPU进程，不并发多个性能候选，不将短暂0%等同独占空闲。
- 原文250T属于4-wave长序列输入，不将当前8-wave实测直接判为250T复现。
- 性能队列输出独立保存，不覆盖上一轮数据；已完成的诊断与尚未完成的独占验收明确分开。

## 复现方式

无须空闲GPU即可执行的CPU资源审计（生成ISA但不初始化GPU）：

```bash
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
FLYDSL_COMPILE_ARCH=gfx942 FLYDSL_COMPILE_ONLY=1 \
python tests/flydsl/mha/compile_bf16_942.py --dq 128 192 --dv 128 192 \
  --page 32 64 128 --causal 0 1 --lse 0 1 --require-no-scratch --retain-isa \
  --dump-root /tmp/mha_bf16_reaudit --output /tmp/mha_bf16_reaudit.json
```

`--mode per-tensor`覆盖scalar Q scale。编译器现在同时检查native stderr，
**即使生成了ISA且进程exit0，出现LLVM error也拒绝**。

中间AGPR强制约束、padding LDS、降低occupancy、缩页ID队列等候选无改善或不合法，均已丢弃。
[无效AGPR实验](results/bf16_spill_accumulator.json)明确标invalid/rejected，其零spill数字不是有效结果。