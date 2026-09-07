# 原始性能基线：口径修正与待复测计划

> 最新用户要求已恢复GPU监测及低负载测试；BF16942大量scratch spill已修复并验证。
> 见 [BF16_SPILL_FIX.md](BF16_SPILL_FIX.md)。以下CPU-only暂停状态保留为上一阶段历史，
> 不再是当前执行限制。绝对250T/410T门槛仍按原文匹配，不把诊断数据当复现。

## 早期CPU-only阶段快照（2026-09-07，已被后续阶段取代）

**以下段落描述暂停GPU时的历史状态，不是当前结论。** 当前已完成BF16 spill修复和低负载
诊断、SWA gather+linear对照；换机测试方法和剩余事项见 [CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md)。
仍未证明原文250T/410T匹配条件下的独占复现。

已按原分支文档修正性能矩阵、输入/计时协议、基线匹配及验收检查。准备执行FP8复现时，
保护检查发现GPU0由新启动的sglang占用（约149GB显存、100%利用率；8张卡均忙），
在任何PTL修改或attention launch之前终止。用户随后明确选择“先完善测试和文档，暂不跑GPU”。
因此本次只做CPU测试和规划，不把共享GPU干扰数据、旧结果或交叉编译当作复现成功。

四个生产kernel本次**未修改**，不在无法验证时改调度或换成其他wave数冒充修复。
之前的865通过功能回归属于上一版本；本次新增性能工具的GPU路径仍待空闲设备验证。
CPU-only回归已通过112项（55项新增性能契约 + 57项现有纯CPU辅助测试）；其中计时、原版
对照和PTL操作均为模拟，不是新GPU测量。

## 原文究竟测了什么

| 原文数字 | 实际kernel / GPU | 精确输入 | 计时与硬件前提 |
|---|---|---|---|
| **413.984T / 648.421µs** | native **8-wave FP8** / MI308X gfx942 | B1,H16,HK1,D192,V128,Q10240,KV2560,page64,NC,无LSE | FlyDSL0.2.2，PTL **Enabled/VECTOR,F8**；1200共同预热，5轮，每轮20warmup/100 profiler events，IQR均值后取中位数 |
| **250.952T / 3422.933µs** | 历史 **4-wave BF16 D128 static** / MI308X gfx942 | B1,H1,HK1,D128,V128,Q=KV40960,page32,NC | 2026-08-10，10套buffer、10warmup、50 CUDA-event样本中位数；原表未记录PTL/精确FlyDSL版本，不能擅自填入 |
| **204.653T / 1323.445µs** | 历史4-wave BF16 D192 static / MI308X gfx942 | H16,HK1,D192,V128,Q10240,KV2583,page32,NC | 同一历史event协议，不是当前8-wave D192 page64 |
| **253.215T / 85.471µs** | 单wave BF16 SWA / **MI350X gfx950** | B1,H16,HK1,D192,V128,Q16384,KV131072,page64,W128,sink | 100共同预热，5轮20/100 profiler；PyTorch2.9.1/ROCm7.2，不能用gfx942数字直接验收 |

来源：

- [FP8原生报告的可移植摘录](results/historical/fp8_942_native_baseline.md)，
  当前源码与原测量内核同源；原报告明确区分PTL Disabled与Enabled。
- `23cc6d1e` 的4-wave README历史章节：2026-08-10性能矩阵、BF16与H3段。
  原表250.952T这一行是4-wave；不能拿当前8-wave的D192/NC短KV形状去“复现250T”。
- `23cc6d1e` 的单waveREADME及已保留的
  [历史原始结果](results/historical/gfx950_swa_original.json)。
- 文档/源码SHA与当前待复测状态见
  [results/performance_followup_status.json](results/performance_followup_status.json)。

### 对上次结论的更正

上次“12组原版/重构版延迟相当”只说明 **FlyDSL0.3.1 + PTL Disabled** 下两者相当，
没有说明已复现文档中的绝对TFLOPS。两者同样慢不能替代原基线验收。
原 FP8 410T对应的硬件策略未启用，且测试Q scale从原per-token改为了per-tensor；
BF16还遗漏了H1长序列、page32和10buffer event协议。

当时可确认的修复是**测试口径及验收流程**；后续spill修复与测量见单独报告。PTL/编译器/寄存器分配分别造成多少绝对性能差距，
必须在空闲GPU上做控制实验后才能下结论，不能直接把差距全部归因于PTL或编译器。

## 新的44项命名workload矩阵

实现见 [_perf_cases.py](_perf_cases.py)，完整CPU计划见
[results/documented_performance_plan.json](results/documented_performance_plan.json)。

- 原4/8-wave统一文档的**全部22项workload**：D128/192 NC、causal32K，SWA KV32K/64K/128K，
  page32 NC/C，page32/64的batch4 D192、batch4 D128 H1、single-head40K、single-head causal32K、
  H3 ragged `(63225,7)` / H14 / HK14。
- 原单wave文档新增18项：D128/192 × W0/16/64/512/1024，以及Q256/2048/4096/65536。
- native FP8 410T专用workload：原per-token scale、native-cast输入、1200warmup协议。
- 3项历史event协议workload：BF16 250T形状、BF16 D192、H3；十套独立buffer真实轮换。
- `--tiles`可额外扫描单wave的全部QT16/32 × BN16/32/64。

原文主矩阵输入先直接生成BF16、将逻辑尾页清零，再对FP8计算Q逐token/head scale及K/V
逐tensor scale；native FP8目标使用FP32随机数直接cast为FNUZ、unit scales。两类输入明确分开，
matrix runner和原版对照共用生成函数，避免对照脚本漏掉量化；CPU按原文逻辑逐项比对。

所有backend逐项记录“不支持”原因；不把page32自动换成page64，不把Q10240自动缩小到KV2560，
也不把未覆盖的gfx950/SWA用例静默消失。长序列和H3对**全部输出**执行分块FP32 reference，
不会为了赶性能省掉正确性。

## 使用方式（空闲GPU后再执行性能命令）

CPU-only计划/测试：

```bash
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all --list-cases
python tests/flydsl/mha/test_mha_pa.py --mode performance --list-cases --case 'full_*' 'batch4_*' 'h3_*'
python -m pytest tests/flydsl/mha/test_perf_cases.py -q
```

统一文档矩阵是性能默认值；可显式选择较小的`--matrix quick`或`--matrix custom`。
documented模式的`--q --kv --dq --dv --page --heads --batch --window`为**过滤器**，不重写原case。
自定义例子：

```bash
python tests/flydsl/mha/test_mha_pa.py --mode performance --matrix custom \
  --backend bf16_942 --dq 128 --heads 1 --q 40960 --kv 40960 --page 32 --causal 0
```

用户已允许的临时GPU0 PTL实验，代码仍要求显式`--ptl`，永不由kernel自动设置。
最新利用率监测要求已取代暂停指令；有其他worker驻留时只能当前PTL诊断，不切换策略。

```bash
# 410T原形状；原文验收门限是400T，413.984T是历史实测值
HIP_VISIBLE_DEVICES=0 python tests/flydsl/mha/test_mha_pa.py --mode performance \
  --backend fp8_942 --case fp8_native_410t --ptl VECTOR,F8 --require-baseline \
  --output tests/flydsl/mha/results/fp8_documented_recheck.json

# 原版8-wave与当前8-wave相同输入/环境；BF16另列branch 4-wave comparator
HIP_VISIBLE_DEVICES=0 python tests/flydsl/mha/reproduce_baselines.py --backend fp8 \
  --ptl VECTOR,F8 --output tests/flydsl/mha/results/fp8_original_vs_current.json
HIP_VISIBLE_DEVICES=0 python tests/flydsl/mha/reproduce_baselines.py --backend bf16 --four-wave \
  --ptl VECTOR,BF16 --output tests/flydsl/mha/results/bf16_original_vs_current.json

# 不改变硬件策略；输出仍逐项记录policy和基线是否可比较
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all
```

- `_hardware`在PTL切换前、`_runner`在case分配/计时前后检查其他进程；有sglang等进程即拒绝。
  不kill进程，不改时钟、功耗、NUMA或其他GPU。检查是进程快照，不是独占资源预留；正式验收
  仍需要调度层保证实验期间不启动其他任务。
- PTL只允许从Disabled开始，finally恢复并读取确认，单独记录before/during/after与所有命令。
  已用CPU模拟成功、数值异常、format失败及enable部分失败时恢复；权限失败且状态没变时
  不重试sudo。非正常进程终止/机器故障无法依靠Python finally保证恢复。
- `--allow-contention`只用于诊断；与`--require-baseline`互斥，绝不能产生基线通过结论。
- FP8 **≥400T**只在backend/形状/MI308X/PTL/protocol匹配时通过。当前0.3.1是否恢复该吞吐
  与“原0.2.2环境逐项一致”分开记录；unknown historical compiler不伪装为匹配。
  `exact_environment_match`仅核对原文已记录的GPU/策略/编译器/PyTorch/HIP字段及实验口径，
  不能据此推断原文未采样的实际时钟相等。缺历史环境字段不会生成精确匹配结论。
- `--require-baseline`还要求有明确数值门槛；仅有历史实测值、没有门槛的SWA/BF16记录只能
  比较和报告，不能把`measured`当成`passed`。条件/协议不匹配时在输入分配及计时前拒绝。
- BF16 250T被标为historical 4-wave comparator，而非当前8-wave的验收数据。当前8-wave
  同形状的吞吐必须实测；是否需要修内核再根据同策略original8/current8/branch4对照判断。
- event interval包含辅助GPU工作及dispatch间隙，使用独立`event_interval_us`字段；
  `attention_us`、`total_gpu_us`均为null，不把interval标成attention-only或GPU kernel总和。
  两种协议都保留原始样本，不混取最快值。
- `--warmup/--rounds/--iterations/--timer/--buffers`可覆盖协议，但记录为unmatched，不能过
  原文严格验收。输出带`complete=false`直到整组成功，提前失败保留状态。
- 原版/当前版对照共用输入和严格FP32/逐位检查。原main的compiled cache只按dtype/device索引，
  因而在workload边界清空临时原版factory cache，**不在buffer轮换或计时区间清空**。
  `--backend swa`原版对照只接受gfx950，记录`original_1wave`；gfx942诊断用matrix runner，
  不跳过原版却标成对照成功。可选branch4同时记录batch1 static或多batch dynamic，仍不是
  已归档的August binary，不能借同为4-wave的名称伪造精确基线匹配。

## 完成与待办

本次55项CPU回归见 [results/performance_contracts_cpu.xml](results/performance_contracts_cpu.xml)：
覆盖workload完整性、精确FLOP、基线误配、没有GPU访问的list-cases、十buffer轮换、
输入生成、原版cache边界、计时标签/门槛失败报告，以及模拟成功/异常时的PTL恢复。
另重跑57项现有纯CPU oracle/地址/参考异常/runner检查，全部通过，见
[results/performance_helpers_cpu.xml](results/performance_helpers_cpu.xml)。

正式跨机器/独占验收仍待：

1. 同一个MI308X上对FP8原0.2.2/新0.3.1、原/现源码，在PTL相同条件下测TFLOPS。
2. BF16按250T实际H1/D128/page32/event协议，另测原8-wave与当前8-wave；不以4-wave替换生产实现。
3. 完整扩展性能矩阵、受影响功能回归；如确有codegen/调度回退，先保存ISA证据，再改内核。
4. 写入新的native结果与gate结论。**已执行的低负载诊断不等于以上正式验收，不宣称绝对性能已恢复。**