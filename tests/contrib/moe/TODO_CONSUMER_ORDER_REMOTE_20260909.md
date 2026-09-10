# TODO：消费顺序流水的换机性能确认

交接日期：2026-09-09。本文为新建的独立交接上下文，不修改已有TODO、设计报告或历史收据。

> **接手先看：代码及完整功能验收已经完成；现在只剩五K性能对照及结果追加。**
> 旧机器的性能计划停在硬件门禁，尚未启动任何一项测量。不要重新优化kernel、重跑完整path矩阵，或把“功能通过”写成“性能不变”。

## 1. 当前状态与唯一剩余任务

| 项目 | 状态 |
|---|---|
| 六K统一消费FIFO | 已完成，K192/256/320/384/512/640 |
| 展开、1N/2N、rolling/pure、ordinary/compact满块 | 已实现并纳入功能验收 |
| GPU随机/graph功能 | **180配置、3600次检查全部通过**，新旧active输出逐bit相同 |
| 独立源码/ELF/ISA/MFMA/VMEM审计 | **360份产物、648个kernel通过，全部零spill** |
| CPU实现与证据回归 | 1344＋175＝1519项通过 |
| 性能计划新增回归 | 55项通过；包含上述部分测试，不再加到1519中去重前的总数 |
| K256性能 | 已有24轮单点，IQR跨零，未确认稳定收益 |
| K192/320/384/512/640性能 | **0/5已测**，已冻结10份旧/新ELF及可执行计划 |
| 后台任务 | 交接时没有本轮验收或性能worker；不存在自动等空闲的守护进程 |

当前权威入口：

- [优化记录](OPTIMIZATION_LOG.md#L186)：按时间追加的经过与结论。
- [最终功能审计](results/all_consumer_order_20260909/final_audited.json#L1)：当前22项生产依赖身份、180配置及全部产物证据。
- [完整GPU原始运行](results/all_consumer_order_20260909/correctness_handoff/result.json#L1)：最终完成收据，不能误用早期失败运行。
- [五K性能计划](results/all_consumer_order_20260909/performance/consumer_suite.json#L1)：`status=blocked_by_hardware`、5个计划点、10份ELF、`jobs=[]`、`attempts=[]`。

### 接手执行清单

- [ ] 确定目标机器、GPU型号/架构、GPU编号、工程真实绝对路径及软件环境。
- [ ] 完整搬运工作树和证据依赖；核验第3节SHA，不能只拉Git提交。
- [ ] 处理第5节的换机限制：路径、8卡门禁、固定GPU7、AMD SMI/ROCm路径。
- [ ] 在目标机建立**新的性能输出目录**，保留原机计划与历史结果。
- [ ] 先完成只读身份/ISA预检；在目标机验证旧/新输出，再计时。
- [ ] 依次测K192/320/384/512/640，各24轮；满足补测条件的点独立测48轮。
- [ ] 从raw复算时延、有效TFLOPS、配对收益/IQR，核验进出硬件及恢复状态。
- [ ] 仅在[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)末尾追加目标机器结果，更新本TODO的完成状态。

**不在本计划内**：其它shape、per_tensor性能、compact/Full性能、PMC/ATT、42点path矩阵、selector修改、自动回退kernel。发现问题先保留证据并说明，不自行扩大范围。

## 2. 必须保留的代码上下文

### 2.1 统一调度

消费序列为先L半区的全部K块，再H半区的全部K块。一个逻辑拍是一个memory＋compute对，不是一个cycle。

```text
Q[q] = 当前要计算的N64 × BK分块
本拍：DS read Q[q] → wait → DS write Q[q+1] → VMEM load Q[q+3]
启动：LDS只预填Q0；P0=Q1、P1=Q2（超尾目标不发请求）
中转：两个P槽，issue到commit相隔两拍；不等到MFMA消费后才释放P
```

- 通用K256/384/512/640使用BK128；K192唯一整BK192；K320唯一128＋192。**总K128支持已经删除，不要恢复其分支。**
- 通用LDS槽为`(n*KS+k_stage)&1`；K384/640的奇数KS在L末K→同N H/K0时，写槽不能盲用“当前槽＋1”。
- K320保留16/24KiB非对称B槽；共享循环只需最后N裁剪，不再保留旧倒数第二N特判。
- K192每N仅两拍，`Q[q+3]`可跨两个N；仍保留末两N边界处理。地址carry有5项：当前读地址、当前H两段写地址、下一N L两段写地址。
- B中转仍双缓冲；CShuffle、scale打包与MFMA累加数学未改变。一次WG处理完整N，不仅一个N128。
- `vmcnt`含普通VMEM load及store，也受scale消费者约束。**wait之后才issue的请求不能计入该wait预算。**逻辑“提前3拍”不等于3拍实际stall或3拍完整延迟隐藏。

### 2.2 K320首次交接修复不能丢

移除整K0预填后，首次H/K128写入的跨wave交接出现过实际随机错误：首N最后一个N32损坏。第一次全量运行在K320第二seed报`rel_l2=0.00525167`，并非容差问题。

修复：在首次q1的B提交/预取之后、交接barrier之前加入`lgkmcnt(0)`：

- [共享循环修复](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py#L83)：`k == 320 and first and stage == 1`。
- [展开路径修复](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k320.py#L456)：`q == 1`。
- 仅启动交接增加等待，稳态未增加`lgkmcnt(0)`。
- [三版原ELF诊断](results/all_consumer_order_20260909/k320_diagnose_handoff/result.json#L1)：每版120次direct/graph，基线0失败、未修候选46失败、修复版0失败。之后最终180配置全部通过。

**不要测错版本**：性能计划中的candidate已包含修复；早期offline候选及[首次失败运行](results/all_consumer_order_20260909/correctness_unchanged/result.json#L1)不是最终候选。单纯token模型只能证明命令/占用关系，不能取代异步LDS的GPU验证。

## 3. 工作树与身份锚点

源机器：`hjbog-srdc-52.amd.com`，Ubuntu 22.04.5容器。分支`luocheng/moe-down-8wave-8x1`。

交接时HEAD：`bf44679342a429d89231386fbc22741f84db3620`，标题`[fly][moe] opt vmcnt/ds_read_b64+valu`。

**该提交不包含当前全部改动**：5个生产文件仍有未提交修改，大量测试脚本和结果未被Git跟踪。仅`git clone/checkout`、仅`git diff`或仅拷贝10份ELF都不足以完整交接。已有[TODO.md](TODO.md)有用户自己的修改，不要覆盖、回退或代替用户提交。

### 当前生产文件SHA256

| 文件 | SHA256 |
|---|---|
| [通用builder](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1) | `991d91a2f00710c481c1ba3c17517e666c79a4fb03938e8edc40f98c94ca7d0c` |
| [共享循环](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py#L1) | `caa4c8b35ffd17ca1cee02b522824405214c9f69510d7a57c65318ddeeb6012b` |
| [共享等待账本](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_schedule.py#L1) | `55333e262ee7786654d35aa0df738adce01710b72384f23fd2e6f12451eaa65c` |
| [K192](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k192.py#L1) | `628cfb24e62bc9e963387b4a0eaa114347b60134ac422e302b60e4c7c80c47e0` |
| [K320](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k320.py#L1) | `19279a835d323abb4f30de4aa97999121ed9caa93b803f476b86dc52dfda9041` |

其它生产依赖必须一并匹配[最终审计的sources](results/all_consumer_order_20260909/final_audited.json#L12)，不能只检查上述5项。

### 关键证据与驱动SHA256

| 项目 | SHA256 |
|---|---|
| [最终审计](results/all_consumer_order_20260909/final_audited.json#L1) | `e9f38ce8df44f7ca84e921706da28852630e4f411bee200471597ca69945e53a` |
| [完整GPU运行](results/all_consumer_order_20260909/correctness_handoff/result.json#L1) | `89604c2460c2c4863bccf49ad3dfa5782d30271b5302dac482bcafbf43b8645a` |
| [原机性能计划](results/all_consumer_order_20260909/performance/consumer_suite.json#L1) | `3acb382115ba03283dd25bb6d63638b63b5290338c8c78003a9dbd630eaeda52` |
| [benchmark_8x1_optimizations_quick.py](benchmark_8x1_optimizations_quick.py) | `f83b9c6cc86993717aad47ca60600c46e99158d33f07afa8ddb84ea126e7d32c` |
| [run_8x1_performance_suite.py](run_8x1_performance_suite.py) | `7e60cd73bde24c89fa6b45e0371774e759ec206138eee34994607e8d250b3099` |
| [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)，交接前26847字节 | `5e66327f8c3e2557e1c2ac60946c93bbf3b90d5c82e8e660dce8d5906c6ab977` |
| [TODO.md](TODO.md)，用户原有25637字节 | `db805110a87758518033650c26ab8d45bd7748a4999b8096c32ad9b345e37051` |

以上是交接时身份，不应在重新迁移路径或改变驱动后伪称同一SHA。性能计划会拒绝已变化的驱动/生产源；目标机适配应生成新计划、冻结新驱动，并保留源证据不变。

## 4. 搬运内容：不要漏掉未跟踪文件和证据链

推荐搬运当前**完整工作树副本**及所需结果树，原机器保留不动；不要使用会删除目标已有资料的同步选项。Python虚拟环境建议在目标机重建并确认可用，不以搬运虚拟环境代替依赖检查。

必须有：

1. 当前生产包、公共helpers/兼容facade，以及本目录的测试、审计、测量脚本和[硬件管理入口](../../flydsl/attn_4wave/tools/run_moe_8x1.py#L1)。依赖不止两个主脚本。
2. [原机性能计划](results/all_consumer_order_20260909/performance/consumer_suite.json#L1)所在的完整性能子树：10份ELF、封装收据、源码快照、驱动快照。
3. [最终审计](results/all_consumer_order_20260909/final_audited.json#L1)、[完整GPU运行](results/all_consumer_order_20260909/correctness_handoff/result.json#L1)，以及该运行的baseline/candidate完整包快照、实际manifest、压缩launcher状态和IR/ISA目录。
4. [上一轮完整基线运行](results/k256_consumer_order_20260909/correctness/result.json#L1)及其引用的冻结依赖，用于追溯最终运行的`baseline_override`。重跑完整审计/历史CPU测试时，还需要对应历史结果，不能只搬新ELF。
5. [K320修复诊断](results/all_consumer_order_20260909/k320_diagnose_handoff/result.json#L1)、CPU收据、[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)和本TODO。

搬运量参考：最终审计所在的本轮结果树交接时约**12GB**；K256上一轮结果树约**1.3GB**，二者不是包含所有历史测试依赖的体积。可以做精简包，但必须沿JSON里的`file`、`manifest`、`sources`、`prior_gpu_manifest`及实际代码读取路径求依赖闭包并逐SHA核验，不能凭目录名称猜最小集合。

**10份ELF已从最终GPU验收manifest的真实launcher提取，性能不需要重新编译Down kernel。** K512曾因离线重编译交换一条`v_add3_u32`的加数顺序而不符合严格ISA身份，因此最终采用原始ELF；不要拿“数学等价/指令数相同”当作同一测量产物。

## 5. 换机前置条件与当前脚本的限制

### 5.1 推荐硬件和软件

- 原环境为8张MI308X、gfx942；设备API报告每GPU 80 CU。优先使用同款、同拓扑机器。
- ELF验收环境：`gfx942:sramecc+:xnack-`。不要将已有ELF直接用于gfx950/其它架构；仅“也是gfx942”也不证明拓扑一致。换GPU型号/拓扑时先复核固定工作组映射，必要时在新目录重编并重验旧/新两版。
- 原解释器Python 3.10.12；FlyDSL 0.3.2；Torch `2.9.1+rocm7.2.0.git7e1940d4`；HIP `7.2.26015-fc0010cf6a`。
- [pyproject.toml](../../../pyproject.toml)没有声明全部GPU实验依赖；安装pyhip不等于已经装好Torch/ROCm/FlyDSL/AITER。
- 原环境另依赖AITER、FlyDSL的兼容`buffer_ops`。其实际哈希在manifest中，缺失时先修复环境，不改kernel迁就错误安装。

以下是**当前脚本真正使用的环境路径**，需要目标机存在，或先明确适配；不是已经可移植的自动发现逻辑：

```text
原工程真实根目录       /root/workspace/luocheng/pyhip
原解释器               /root/workspace/luocheng/pyhip/.venv/bin/python
原AITER                /opt/aiter
原系统Python包         /usr/local/lib/python3.10/dist-packages
FlyDSL候选搜索路径     /root/workspace/luocheng/FlyDSL/build-fly/python_packages
                       /root/workspace/luocheng/FlyDSL/python
LLVM工具               /opt/rocm-7.2.0/lib/llvm/bin/llvm-mc
                       /opt/rocm-7.2.0/lib/llvm/bin/llvm-objdump
HIP动态库              /opt/rocm/lib/libamdhip64.so
AMD SMI根目录          /tmp/amd-smi-lib-26.2.2-rocm-7.2.3/opt/rocm-7.2.3
主机NUMA设置           /proc/sys/kernel/numa_balancing
```

### 5.2 路径不是只改一个ROOT变量

原收据、manifest、压缩launcher状态和性能计划包含**绝对路径**，SHA及缓存身份也可能包含路径。

- **最少改动方案**：在新机器提供同样的真实绝对目录布局。普通软链接可能被`Path.resolve()`解析回不同目录，不能假定软链接即可满足身份；可采用合适的实际目录或挂载布局。
- 若根目录必须不同：先实现明确的只读路径映射/迁移适配，保留原JSON/IR/ELF与SHA，建立带来源映射的新本机收据及计划。**当前尚无通用路径迁移实现，此项是条件性TODO。**
- 禁止对全部JSON/MLIR简单搜索替换路径后，仍把它们称为原始证据；不要通过关闭SHA检查让测量继续。

### 5.3 GPU选择和门禁尚未完全参数化

- [套件入口](run_8x1_performance_suite.py#L1)目前固定传`--gpu 7`和`gpu_environment(7)`，套件本身**没有`--gpu`参数**。仅设置父进程`HIP_VISIBLE_DEVICES`不会改变这一选择。
- [单点入口](benchmark_8x1_optimizations_quick.py#L1)有`--gpu`和`--amdsmi-root`；套件目前没有转发这些可配置项。
- [全机负载检查](k192_cshuffle_workflow.py#L140)及[硬件审计](summarize_todo_routing.py#L18)要求**8张卡**，所有卡busy≤5%、VRAM≤20%；不是只检查选中的卡。
- 当前managed模式只接管初始auto/650W的空闲GPU，使用PTL **Enabled / VECTOR,F8**、1800MHz determinism、NUMA off，结束恢复原状态。
- 如果目标机不是8卡、没有GPU7、AMD SMI路径不同，先参数化运行器和对应审计/测试，在**新输出目录**建立目标机计划；不能只绕过一个assert或把性能测量改走“不改变硬件的功能模式”。
- NUMA设置是主机级别，测试前要有可控执行窗口和相应权限；权限不足就停止，由用户在机器侧处理。不能把密码或凭据写进本TODO/收据。
- 门禁失败保存状态后停止，不轮询等待、不杀其它用户任务、不筛除长尾来伪造“空闲”。

## 6. 目标机执行步骤

以下命令**仅适用于已满足同绝对路径、同软件布局、8卡/GPU7条件的目标机**。路径或硬件不同，先完成第5节适配；本交接没有在目标机实际执行或迁移文件。

### 6.1 建立独立目标机计划（不启动GPU）

先记录目标机hostname、GPU型号/PCI或UUID、驱动/ROCm/Torch/FlyDSL版本、选卡、PTL/时钟/功率/NUMA初态。记录放入目标机的新运行产物，不改原机验收收据。

```bash
cd /root/workspace/luocheng/pyhip
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
VALIDATION="$ROOT/tests/contrib/moe/results/all_consumer_order_20260909/final_audited.json"
OUT="$ROOT/tests/contrib/moe/results/all_consumer_order_20260909/performance_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)"
export PYTHONPATH="$ROOT/src:$ROOT/tests/contrib/moe:/opt/aiter:/usr/local/lib/python3.10/dist-packages"

# OUT必须尚不存在；不要先mkdir，也不要复用原机performance目录。
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$PY" \
  "$ROOT/tests/contrib/moe/run_8x1_performance_suite.py" \
  --consumer-order --prepare-only --validation "$VALIDATION" --output "$OUT"
```

- 必须带`--consumer-order`；不带会走历史六K×两量化×两Batch×24/48的三版大套件，不是本任务。
- 首次准备会提取并核验10份已验收ELF及保存驱动；不编译Down、不加载GPU。已有计划的`--prepare-only`只确认计划/source/driver身份，不代表硬件已空闲。
- `OUT`保存下来，之后同机器续跑继续使用同一值，不重新生成另一时间戳目录来丢失已测进度。
- 不从临时目录运行heredoc导入测试模块：原机临时目录曾有同名旧模块造成遮蔽。优先以上述绝对脚本入口、正确工作目录执行。

### 6.2 先核验目标机正确性，再运行性能

旧机器的完整180配置不用无条件重跑。软件/型号/路径一致时，至少检查所选10份ELF在目标机的随机权重direct/graph与输出哨兵；可复用[diagnose_consumer_order.py](diagnose_consumer_order.py)，其结果在独立新目录，不改原验收身份。每K的baseline/candidate封装收据路径从新计划的`records`读取，显式匹配GPU编号。

注意：诊断器`status=complete`只表示诊断执行完，必须同时确认`all_passed=true`、所有`mismatches=0`、有限误差<0.005和输出合同通过。性能入口自身另有测量前后每buffer的FP32/逐bit检查，但快速性能输入为随机A/全1W，不代替随机W检查。

一切前置条件通过后：

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$PY" \
  "$ROOT/tests/contrib/moe/run_8x1_performance_suite.py" \
  --consumer-order --validation "$VALIDATION" --output "$OUT"
```

该命令先复核source/manifest/ELF，再检查全机门禁，通过后才启动目标GPU和managed设置。结束后以新计划的`status`、`jobs`、`attempts`、`confirmed`及原始测量收据为准。

### 6.3 固定测量协议和完成标准

| 参数 | 值 |
|---|---|
| K | 192、320、384、512、640 |
| N / Batch / TopK / Experts | 2048 / 32768 / 8 / 256 |
| 量化 / 路径 / 输出padding | PTPC / ordinary Down-only / 128字节 |
| 循环与输出 | 默认1N、rolling、原输出store策略 |
| Buffer / 顺序 | 10个buffer；同地址ABBA/BAAB交替，无gateup预热 |
| 首轮 | 每点24轮＝96个GPU event；每版48个event样本 |
| 复核 | 配对收益Q1≤0：跨零或回退均独立补48轮；不与24轮混池 |
| 工作量 | $F=2\times Batch\times TopK\times N\times K$ |
| 有效TFLOPS | $F/(ms\times10^9)$，不是ATT模型TFLOPS |

五个初始点共480个event；若全部需要补48轮，合计最多10份运行、1440个event，不包含功能检查调用。

每点最终至少报告：目标机身份、baseline/candidate源和ELF、各自中位ms/有效TFLOPS、逐轮配对收益中位数/IQR、wins、实际轮数、前后数值检查、PTL/功率/时钟/NUMA与恢复状态。

- 配对收益由每轮两次同版本样本的均值比计算，不是两个独立中位时延的比值。
- **IQR跨零仅表示未确认稳定差异，不证明性能等价。** 如要“±某阈值内无回退”的等价结论，需要另行约定阈值及统计方法。
- `audit_run(..., pair=True)`会复算raw统计、同buffer顺序、工作量及硬件记录；每份正式记录必须通过。
- 同点48轮存在时选48整点作为最终结论，不挑24/48里更好看的结果。不同机器的absolute time不直接相减，旧/新必须在同一目标机同一协议内配对。
- 如prepare/执行/审计失败：保留原输出、日志与attempt；先定位失败。驱动改变则不能假装原计划仍绑定同一驱动，建立新计划并保留来源。成功但尚未登记的产物先审计，不盲目重复计时。
- [硬件管理入口](../../flydsl/attn_4wave/tools/run_moe_8x1.py#L74)应在异常时也恢复；恢复失败就停止，不继续下一个点。

## 7. 已完成结果，避免重做或误读

### 功能精度

| K | 最大rel_l2 |
|---:|---:|
| 192 | 0.003864611266180873 |
| 256 | 0.0034054694697260857 |
| 320 | 0.0035468696150928736 |
| 384 | 0.003433756297454238 |
| 512 | 0.0034018217120319605 |
| 640 | 0.0035654171369969845 |

当前N2048/PTPC默认1N编译资源，均零spill、稳态memory VALU为0：

| K | VGPR | 总LDS KiB | 稳态vmcnt |
|---:|---:|---:|---|
| 192 | 214 | 64 | 14/14 |
| 256 | 176 | 48 | 7/7/7/7 |
| 320 | 214 | 56 | 7/10/7/10 |
| 384 | 202 | 48 | 5/9/9/9/5/1 |
| 512 | 220 | 48 | 5/9/9/9/5/1/1/1 |
| 640 | 244 | 48 | 5/9/9/9/5/1/1/1/1/1 |

这些是结构事实，不是性能结论。K192新增地址carry、K320首次交接等待的实际代价都要通过上述五点实测确认。

### K256历史单点仅作上下文

[原始K256性能](results/k256_consumer_order_20260909/ptpc_abba24.json#L1)：同N2048/PTPC/Batch32K协议，修改前0.736963ms／372.99有效TFLOPS，修改后0.735643ms／373.66有效TFLOPS；$F=274877906944$。配对收益+0.343%，IQR[−0.114%, +1.157%]，17/24轮更快。

该IQR跨零，未确认稳定收益；本次全分支推广没有改变默认K256/PTPC的实际指令，因此未加入重复测量。**不能将这一个点外推其余K、其它机器或完整MoE链路。**

## 8. 交付约束

- 中文回复和记录；先功能再性能，小步修改后重测。
- 新机器结果只追加到[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)，本TODO用于交接/勾选，不再新建分散报告。
- 不修改已有[TODO.md](TODO.md)、旧报告、原始失败收据、source/ISA/ELF快照或历史缓存。
- 不自动stage/commit/push、回退kernel或改线上selector；用户自行处理版本管理。
- 没有目标机器信息、传输凭据或已建立远程执行连接；**本文只准备上下文与操作计划，本轮未实际传输、未在新机器测量。**