# FlyDSL MoE 两阶段：性能、测试与 8x1 流水线

本文是MoE当前唯一维护的说明文档，独立说明支持范围、路径选择、测试方法、六K完整流水及重要优化。默认目标为 **MI308X / gfx942、FP8 E4M3FNUZ A/W、BF16输出**，不将8x1结果外推其它dtype或硬件。

**版本边界：**2026-09-10仅保留Down四path：`default`、`1x4_64x256`、`8x1`、`8x1_compact`；`1x8/2x4/4x1`已删除，旧path明确报错、不静默回退。当前推荐是测试预设，不是生产自动selector。性能来自9/9矩阵及9/10两点独立确认，设计／资源来自9/9原pack消费FIFO版本；本次完成保持执行顺序的可读性重构，未新增GPU性能数据。

原始文档字节已冻结在[迁移准备清单](../../../../tests/contrib/moe/results/readability_docs_20260910/prepared.json)，SHA256：`cff0e61c660d25dac2f62215d2ab9dce75f0f7011c1f2d68b50d561f23d6454a`。历史日期、JSON、ISA、ELF和source身份不重标；链接只供复核，正文无需借助其它文档理解。

## 目录

- [一、性能与测试](#performance)：[支持与shape](#scope) · [测试方法](#testing) · [42点主表](#matrix42) · [66行历史落选](#historical-nonselection) · [删除与独立确认](#migration-confirmation) · [验证边界](#validation-boundaries)
- [二、流水线设计](#pipeline)：[符号](#notation) · [K256](#pipeline-k256) · [K192](#pipeline-k192) · [K320](#pipeline-k320) · [K384](#pipeline-k384) · [K512](#pipeline-k512) · [K640](#pipeline-k640) · [非PTPC与尾部](#pipeline-boundaries) · [资源](#resources)
- [三、重要优化](#optimizations)

<a id="performance"></a>

## 一、性能与测试

<a id="scope"></a>

### 1.1 支持范围与七组shape

| 当前path | 本文的布局／约束 |
|---|---|
| `default` | 矩阵BM64，BN沿用case原配置，默认padding；该builder的其它算法／格式不在8x1性能结论内。 |
| `1x4_64x256` | 简称1x4，BM64/BN256；非Hy3为padding128B。Hy3测过padding0和128B；4K当前选后者，称1x4-P128／direct_m64。 |
| `8x1` | BM256/BN128，512线程、8 waves，M256 sorting metadata，非atomic输出，`alg="prefill_1x4"`。 |
| `8x1_compact` | M64 metadata/BN128接口，构表→M256 full→M64 tail三个kernel，同stream、无host count读回；阈值0.6，Down计时包含全部三个kernel。 |

8x1/compact只支持 **K=192/256/320/384/512/640**，正N为128倍数；weight/activation支持 `ptpc/ptpc`、`per_tensor/per_tensor`、`per_tensor/ptpc`，activation未指定则随weight。padding为0/32/64/128B，BF16行stride=`N+padding/2`，sum须匹配；compact要求`0<E≤2048`。

**K192整192，K320为128+192，其余BK128。** K192/K320兼容`tile_k=None/128/192`但都转入真实192实现，不是三种算法。8x1的K128、BK64、pure/rolling环境开关、legacy wait及固定`_block_k`实验参数已删；其它Down算法的K128支持不受影响。builder默认`_n_loop=1`、`_store_cache=2`，0/2 N展开是显式选项，不是自动selector。源码：[四path分发](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2.py)、[8x1](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py)、[compact](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py)。

每case的Batch为1024/2048/4096/8192/16384/32768（1K–32K），七组共42点。

| Case | Hidden N | Inter/TP K | E | TopK | weight/activation量化 | gateup BN |
|---|---:|---:|---:|---:|---|---:|
| Hy3 K192 | 4096 | 192 | 193 | 9 | per_tensor/per_tensor | 128 |
| Qwen397 K512 | 4096 | 512 | 512 | 10 | ptpc/ptpc | 256 |
| Qwen397 K256 | 4096 | 256 | 512 | 10 | ptpc/ptpc | 256 |
| Qwen35 K512 | 2048 | 512 | 256 | 8 | ptpc/ptpc | 256 |
| Qwen35 K256 | 2048 | 256 | 256 | 8 | ptpc/ptpc | 256 |
| Xiaomi K256 | 6144 | 256 | 384 | 8 | ptpc/ptpc | 256 |
| H3 K384 | 6144 | 384 | 128 | 4 | ptpc/ptpc | 256 |

9/9历史候选还含1x8（BM64/BN512，仅Hy3双per-tensor）、2x4（BM128/BN256，五个非K512 case）、4x1-M128/M256（BN64，另有Qwen35 K256 M256-P128）。这些只解释历史，不能传给当前dispatcher。2x4的K512当时activation+CShuffle=80KiB超过64KiB guard，是当时实现限制，不是跨硬件的算法不支持结论。

<a id="testing"></a>

### 1.2 功能与性能测试方法

按 **CPU契约→fresh offline→GPU random→performance** 推进；CPU不证明GPU数值，离线编译不是运行，性能全1权重检查不替代随机W。

随机Down参考使用实际量化后的A/W转FP32做矩阵乘，再乘对应expert/channel weight scale、activation行scale和routing weight。`rel_l2`为 $\lVert actual-reference\rVert_2/\lVert reference\rVert_2$，专项Down要求小于0.005，并检查有效输出finite、padding及inactive区域的NaN哨兵未被覆盖；graph重放还须污染并重建任务workspace。公共整链另与其参考输出比较，不将Down单算子阈值冒称所有整链的统一契约。

| 层次 | 入口及检查 |
|---|---|
| CPU契约 | [共享schedule](../../../../tests/contrib/moe/test_all_8x1_schedule.py)、[K192](../../../../tests/contrib/moe/test_k192_bk192.py)、[首过渡/尾部/状态](../../../../tests/contrib/moe/test_nloop_transition.py)、[删path](../../../../tests/contrib/moe/test_remove_down_paths.py)：实际AST、B/scale消费者、wait、pack和状态；冻结指纹不是新运行。 |
| fresh offline | [本轮重构编译入口](../../../../tests/contrib/moe/check_moe_readability.py)：绑定本轮完整before/after源，COMPILE_ONLY/gfx942、4XCC/80CU显式输入，GPU/ExecutionEngine入口拒绝；保存source/IR/实际ISA/ELF/SHA，查零private/spill/scratch。不是读取GPU属性，不能仅据编译通过宣称功能正确。该入口限定本轮四源改动，后续新实验须重新建立匹配身份。 |
| ordinary随机 | [Nloop驱动](../../../../tests/contrib/moe/check_all_8x1_nloop.py)：随机FP8 A/W、routing、非恒定scale、FP32参考、rel_l2<0.005、finite及padding/inactive哨兵；六K、三量化、unroll0/1/2、task-table。**会编译并运行GPU**；旧CLI硬锁HIP_VISIBLE_DEVICES=7，不能直接当GPU4命令或伪造编号绕过。 |
| compact随机/边界 | [Down测试](../../../../tests/contrib/moe/test_compact_m64_down.py)：六K、mixed、N128/多tile、四padding、graph重放污染workspace、>2GiB真实权重偏移、FP32参考。必须核对均衡后的有效full/tail，而非仅有full launch：新增`test_compact_post_balance_full_coverage`以CU与expert数的公倍数构造full及full+tail，并断言准确counts；本次未运行这个GPU节点。[任务表exact cover](../../../../tests/contrib/moe/test_compact_m64_tasks.py)的GPU node不是CPU测试，即使文件含CPU reference。 |
| 公共整链 | [公共测试](../../../../tests/contrib/moe/test_moe.py)的test_acc_fly_splitk_2s_down_8x1：Batch33/N512/E8/TopK4/TP1、六K×两量化、run_count=0；compact整链也调用公共入口。小N Down不等于整链支持，整链沿用N为512倍数。 |

**以下命令从仓库根执行，本次未运行。** 每次新目录，不覆盖历史或失败记录。[pytest配置](../../../../pytest.ini)为`python_files=*.py`，必须显式file/node，不对目录盲跑。重构后契约若拒绝，应查真实改变，不能删断言或修改旧SHA。

```bash
set -euo pipefail
REPO="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/tests/contrib/moe:/opt/aiter:/usr/local/lib/python3.10/dist-packages"
RUN="$REPO/tests/contrib/moe/results/manual_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$RUN" ]] && mkdir -p "$RUN"
unset MOE_PREFILL_TILE_K MOE_8X1_ROLLING_EPILOGUE
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
	.venv/bin/python -m pytest -q -p no:cacheprovider \
	tests/contrib/moe/test_all_8x1_schedule.py \
	tests/contrib/moe/test_k192_bk192.py \
	tests/contrib/moe/test_nloop_transition.py --junitxml="$RUN/cpu.xml"
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 COMPILE_ONLY=1 \
ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942 \
	.venv/bin/python tests/contrib/moe/check_moe_readability.py \
	--version after \
	--k 256 --n 2048 --topk 8 --quant ptpc --n-loop 1 --padding 128 \
	--output "$RUN/offline_k256_ptpc"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
HIP_VISIBLE_DEVICES=4 \
	.venv/bin/python -m pytest -q -p no:cacheprovider \
	tests/contrib/moe/test_moe.py::test_acc_fly_splitk_2s_down_8x1 \
	tests/contrib/moe/test_compact_m64_down.py::test_compact_down_reference \
	tests/contrib/moe/test_compact_m64_down.py::test_compact_mixed_quant_all_full \
	tests/contrib/moe/test_compact_m64_down.py::test_compact_post_balance_full_coverage \
	tests/contrib/moe/test_compact_m64_down.py::test_compact_padding_multitile_and_graph \
	tests/contrib/moe/test_compact_m64_down.py::test_compact_end_to_end \
	--junitxml="$RUN/gpu.xml"
```

GPU测试先确认所选卡可用；仅设置一层HIP设备过滤，进程内物理GPU4重编号为cuda:0，不叠加ROCR筛选。上例单配置offline+公共pytest**不自动产生两版30配置随机/graph门禁**。旧小Batch的“all_full”测试名指均衡前几何，80CU/阈值0.6下可能全拆为tail；新增节点使full数为`lcm(CU,8)`，余数为0，MI308X/80CU的Batch5120/5121分别保留80个full及80full+4tail。完整ordinary Nloop随机应使用与目标卡/当前源码匹配的驱动，旧GPU7 guard不可绕过。[冻结版本runtime](../../../../tests/contrib/moe/k192_cshuffle_runtime.py)说明source/launcher绑定，但旧候选SHA/参数不是现行接口。

#### 计时与统计

- Down：ordinary一个kernel，compact构表/full/tail全计；矩阵event外先公共default gateup刷新上下文。
- Combined：同event内Down+sorted_sum，gateup仍在外。Full：sorting、两次量化、gateup、down、invert、sum，中间张量在调用内创建，不是全地址固定/graph计时。
- 三phase不能相减为独立sum/gateup耗时，也不能各选最优拼一个实现。编译/审计/父墙钟不当kernel时延。

$$
F_D=F_{Combined}=2\times B\times TopK\times N\times K,\quad
F_{Full}=3F_D,\quad T_{effective}=F/(t_{ms}\times10^9).
$$

所有格为 **ms / wall-time有效TFLOPS**，按useful而非padding执行行计F；Full的3倍仅计三路GEMM，其它算子耗时仍在event内。ATT union×roof只能称“模型TFLOPS”，不混用。

性能参考：只选**GPU4–7空闲目标卡**，9/9与9/10用GPU4/MI308X/gfx942/80CU；核验PCI/ID，busy≤5%、VRAM≤20%。其它卡只观察，非全机空闲，进出快照非连续监控。**NUMA保持1，不改主机设置；PTL实核Enabled / VECTOR,F8，请求1800MHz determinism/650W**，不把请求当持续实测频率。10-buffer、warmup、同半轮同输入/权重，两版ABBA/BAAB、矩阵轮转正序+反序，每轮两个sample先均值再配对；保留raw/IQR，不删长尾。微改动先两版单shape Down-only，复用SHA已验ELF；不自动扩大compact/Full/42点。快测无矩阵gateup priming，两协议绝对ms不混池。结束先drain stream、后检、卸载kernel，再恢复原auto/PTL并核验NUMA/设备。实现：[目标卡托管](../../../../tests/contrib/moe/k192_cshuffle_workflow.py)、[两版快测](../../../../tests/contrib/moe/benchmark_8x1_optimizations_quick.py)。不要复用旧全机NUMA关闭协议。

复现上述参考环境前还须只读确认NUMA初态为1；否则停止，不改主机设置。目标卡托管只保证保持初态，并不会强制把它设为1。

以下模板**仅在已有当前源码完整30配置随机/graph验证及兼容schema两版产物后**使用；三收据变量须真实存在。驱动严格核验，不能以旧JSON改SHA满足。

```bash
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
HIP_VISIBLE_DEVICES=4 \
PYTHONPATH="$REPO/src:$REPO/tests/contrib/moe:/opt/aiter:/usr/local/lib/python3.10/dist-packages" \
	.venv/bin/python tests/contrib/moe/benchmark_8x1_optimizations_quick.py \
	--k 256 --quant ptpc --batch 32768 --rounds 24 --buffers 10 \
	--gpu 4 --hardware-scope target \
	--baseline "${BASELINE_RECEIPT:?需已核验基线}" \
	--candidate "${CANDIDATE_RECEIPT:?需已核验候选}" \
	--validation "${VALIDATION_RECEIPT:?需当前源码30配置随机及graph审计}" \
	--output "$RUN/down_k256_pair.json"
```

9/9矩阵：旧9/8只给42个incumbent标签，不复用绝对ms。三phase配对改善**Q1均>0**才支配旧推荐，竞争候选互比，否则保守保留，推荐不必是各phase绝对median最小。全候选24轮后一次冻结5点独立48轮，涉及1x8/2x4的原推荐/初选也确认；48整点替代24，不混池/追加方向性抽样。**IQR是样本分位，不是置信区间；跨0不证明等价。** 同轮$r=mean(t_{candidate,2})/mean(t_{control,2})$：改善$1-r$，时延增幅$r-1$，吞吐损失$1-1/r$；配对median不等于绝对median之比。[旧矩阵门禁](../../../../tests/contrib/moe/matrix_retest_support.py)拒绝新源码/模块集合是保护，不得绕过；后续须先有匹配当前源码的新验证。

<a id="matrix42"></a>

### 1.3 42点当前推荐与当时测量：混合日期展示

[迁移标签](../../../../tests/contrib/moe/results/remove_down_paths_20260910/migrated_choices.json)：compact14点、1x4 18点、ordinary8x1 5点、default5点。40点沿用9/9选择和当时ms/T；仅Hy3 4K为direct_m64/1x4-P128、8K为compact，采用9/10确认的replacement_ms/T。Qwen35 K256/32K当前是8x1/padding128B，不再旧手工4x1。

**这是两个日期独立runs的并列展示，不是9/10新统一42点测量，不算跨日期加速、不把40个旧值重标为删除后性能。** 9/9共9份run、55,584个event（全候选24轮+5点独立48轮）；两点9/10确认另计。[旧矩阵汇总](../../../../tests/contrib/moe/results/final_matrix_20260909/matrix_summary.json)和[两点确认](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/audited.json)各保留原身份。

| Case / Batch | 当前推荐 | Down ms / T | Combined ms / T | Full ms / T | 9/9来源／轮数 | 9/10来源／轮数 |
|---|---|---|---|---|---|---|
| Hy3 K192 / 1K | 8x1_compact | 0.106360 / 136.29 | 0.135881 / 106.68 | 0.385041 / 112.94 | [48](../../../../tests/contrib/moe/results/final_matrix_20260909/hy3_abba48.json) | — |
| Hy3 K192 / 2K | 8x1_compact | 0.164841 / 175.87 | 0.224160 / 129.33 | 0.561322 / 154.94 | [48](../../../../tests/contrib/moe/results/final_matrix_20260909/hy3_abba48.json) | — |
| Hy3 K192 / 4K | 1x4-P128 | 0.232161 / 249.75 | 0.340062 / 170.50 | 0.824363 / 211.01 | — | [独立48](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/audited.json) |
| Hy3 K192 / 8K | 8x1_compact | 0.363761 / 318.79 | 0.579722 / 200.03 | 1.428066 / 243.61 | — | [独立48](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/audited.json) |
| Hy3 K192 / 16K | 8x1 | 0.607302 / 381.90 | 1.022984 / 226.72 | 2.574450 / 270.27 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/hy3_abba24.json) | — |
| Hy3 K192 / 32K | 8x1 | 1.097104 / 422.80 | 1.933428 / 239.91 | 5.125241 / 271.51 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/hy3_abba24.json) | — |
| Qwen397 K512 / 1K | 1x4 | 0.488781 / 87.87 | 0.521222 / 82.40 | 1.222564 / 105.39 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K512 / 2K | 1x4 | 0.489762 / 175.39 | 0.549662 / 156.28 | 1.321525 / 195.00 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K512 / 4K | default | 0.813683 / 211.14 | 0.924584 / 185.81 | 2.311949 / 222.93 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K512 / 8K | default | 1.154524 / 297.61 | 1.357945 / 253.03 | 3.361913 / 306.61 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K512 / 16K | 8x1_compact | 1.569807 / 437.76 | 2.010748 / 341.76 | 5.517061 / 373.67 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K512 / 32K | 8x1_compact | 3.097813 / 443.66 | 4.034175 / 340.69 | 10.886242 / 378.75 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_abba24.json) | — |
| Qwen397 K256 / 1K | 1x4 | 0.281141 / 76.38 | 0.315721 / 68.02 | 0.694423 / 92.77 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen397 K256 / 2K | 1x4 | 0.281941 / 152.34 | 0.339981 / 126.33 | 0.778163 / 165.58 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen397 K256 / 4K | 1x4 | 0.480922 / 178.61 | 0.603542 / 142.33 | 1.354925 / 190.19 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen397 K256 / 8K | 1x4 | 0.670983 / 256.04 | 0.899683 / 190.95 | 2.045528 / 251.96 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen397 K256 / 16K | 8x1_compact | 0.967524 / 355.13 | 1.439686 / 238.66 | 3.398293 / 303.33 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen397 K256 / 32K | 8x1_compact | 1.816547 / 378.30 | 2.742090 / 250.61 | 6.670746 / 309.05 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen397_k256_abba24.json) | — |
| Qwen35 K512 / 1K | default | 0.147881 / 116.17 | 0.160400 / 107.11 | 0.379221 / 135.91 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K512 / 2K | default | 0.151261 / 227.16 | 0.171660 / 200.16 | 0.436382 / 236.21 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K512 / 4K | default | 0.234341 / 293.25 | 0.274501 / 250.34 | 0.720803 / 286.01 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K512 / 8K | 8x1_compact | 0.323941 / 424.27 | 0.415321 / 330.92 | 1.239924 / 332.53 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K512 / 16K | 8x1_compact | 0.636443 / 431.90 | 0.807683 / 340.33 | 2.378649 / 346.68 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K512 / 32K | 8x1 | 1.198304 / 458.78 | 1.517466 / 362.29 | 4.639077 / 355.52 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_abba24.json) | — |
| Qwen35 K256 / 1K | 1x4 | 0.082181 / 104.53 | 0.097640 / 87.98 | 0.231481 / 111.33 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Qwen35 K256 / 2K | 1x4 | 0.085141 / 201.78 | 0.110900 / 154.91 | 0.277381 / 185.81 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Qwen35 K256 / 4K | 1x4 | 0.137441 / 250.00 | 0.185761 / 184.97 | 0.459261 / 224.45 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Qwen35 K256 / 8K | 8x1_compact | 0.202081 / 340.06 | 0.293261 / 234.33 | 0.789243 / 261.21 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Qwen35 K256 / 16K | 8x1_compact | 0.373301 / 368.17 | 0.544542 / 252.39 | 1.499326 / 275.00 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Qwen35 K256 / 32K | 8x1 | 0.711962 / 386.08 | 1.030364 / 266.78 | 2.906831 / 283.69 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/qwen35_k256_abba24.json) | — |
| Xiaomi K256 / 1K | 1x4 | 0.295601 / 87.18 | 0.337341 / 76.39 | 0.743303 / 104.01 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba24.json) | — |
| Xiaomi K256 / 2K | 1x4 | 0.294841 / 174.80 | 0.363582 / 141.76 | 0.850843 / 181.72 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba24.json) | — |
| Xiaomi K256 / 4K | 1x4 | 0.528742 / 194.95 | 0.661822 / 155.75 | 1.492765 / 207.16 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba24.json) | — |
| Xiaomi K256 / 8K | 1x4 | 0.748223 / 275.53 | 1.005484 / 205.03 | 2.255988 / 274.15 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba24.json) | — |
| Xiaomi K256 / 16K | 8x1_compact | 1.288145 / 320.09 | 1.799987 / 229.07 | 4.192895 / 295.01 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba24.json) | — |
| Xiaomi K256 / 32K | 1x4 | 2.306848 / 357.47 | 3.297132 / 250.11 | 7.853290 / 315.01 | [48](../../../../tests/contrib/moe/results/final_matrix_20260909/xiaomi_abba48.json) | — |
| H3 K384 / 1K | 1x4 | 0.157641 / 122.60 | 0.178621 / 108.20 | 0.415881 / 139.42 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |
| H3 K384 / 2K | 1x4 | 0.159800 / 241.89 | 0.198661 / 194.58 | 0.480002 / 241.59 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |
| H3 K384 / 4K | 1x4 | 0.302721 / 255.38 | 0.375281 / 206.00 | 0.844144 / 274.75 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |
| H3 K384 / 8K | 8x1 | 0.392921 / 393.51 | 0.534342 / 289.36 | 1.431425 / 324.05 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |
| H3 K384 / 16K | 8x1_compact | 0.672242 / 460.01 | 0.950464 / 325.35 | 2.615649 / 354.68 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |
| H3 K384 / 32K | 8x1_compact | 1.327125 / 466.03 | 1.878647 / 329.21 | 5.184999 / 357.84 | [24](../../../../tests/contrib/moe/results/final_matrix_20260909/h3_abba24.json) | — |

Hy3 1K/2K虽选compact，**full数为0，数学由M64 tail完成**，不是8x1满块吞吐获胜。

<a id="historical-nonselection"></a>

### 1.4 2026-09-09完整历史落选账本（66行）

> **66行保留9/9原值。Hy3 4K/8K当时对照1x8，现在已不存在；两点当前迁移看独立确认。这里不是删除后或重构后的全矩阵重评。** “现选/旧推荐”均为9/9当时标签。D/C/F=Down/Combined/Full，ms/T为有效吞吐；改善正值更快，IQR为配对Q1/Q3。行比是执行填充行/当时所选执行行，F/T是compact full/tail任务数（均衡routing几何，不是实测驻留）。24点两者均落选。padding与慢Down相关但无单因素归因；full=0仍三个launch。未采PMC/ATT，不猜bank/cache/occupancy。证据：[66行原始JSON](../../../../tests/contrib/moe/results/final_matrix_20260909/nonselection.json)。

| 点 | 未选候选 vs 所选 | 候选D/C/F ms / T | 相对所选改善 [IQR]% | 工作量 | 未选原因 |
| --- | --- | --- | --- | --- | --- |
| Hy3 K192 / 1K | 8x1 vs 8x1_compact | Down 0.237680 / 60.99<br>Combined 0.270541 / 53.58<br>Full 0.694143 / 62.65 | Down -123.854% [-125.117, -121.763]<br>Combined -99.492% [-100.507, -98.194]<br>Full -80.928% [-85.434, -76.843] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 2K | 8x1 vs 8x1_compact | Down 0.237721 / 121.95<br>Combined 0.297541 / 97.44<br>Full 0.751403 / 115.75 | Down -44.460% [-45.041, -43.582]<br>Combined -32.639% [-33.521, -32.226]<br>Full -34.058% [-36.056, -31.003] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 4K | 8x1 vs 1x8 | Down 0.238281 / 243.33<br>Combined 0.349842 / 165.74<br>Full 0.886363 / 196.25 | Down -0.651% [-1.648, -0.064]<br>Combined -3.916% [-4.608, -3.444]<br>Full -6.317% [-8.138, -4.762] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 4K | 8x1_compact vs 1x8 | Down 0.239841 / 241.75<br>Combined 0.348082 / 166.58<br>Full 0.838423 / 207.47 | Down -0.790% [-1.112, -0.236]<br>Combined -3.297% [-3.644, -2.857]<br>Full -0.634% [-2.500, +0.774] | 行比1.000；F/T=0/579 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 8K | 8x1 vs 1x8 | Down 0.423022 / 274.13<br>Combined 0.641023 / 180.90<br>Full 1.559246 / 223.12 | Down -4.378% [-4.678, -4.003]<br>Combined -7.907% [-8.383, -7.604]<br>Full -8.884% [-9.932, -8.219] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 8K | 8x1_compact vs 1x8 | Down 0.374462 / 309.68<br>Combined 0.592202 / 195.82<br>Full 1.439025 / 241.76 | Down +7.222% [+6.589, +7.708]<br>Combined +0.005% [-0.482, +0.374]<br>Full -0.217% [-1.324, +0.473] | 行比1.000；F/T=160/518 | 未达三phase共同改善门槛；对现选路径Q1≤0：Combined/Full；对旧推荐未过：Combined/Full |
| Hy3 K192 / 16K | 8x1_compact vs 8x1 | Down 0.655583 / 353.77<br>Combined 1.076024 / 215.54<br>Full 2.635690 / 263.99 | Down -8.540% [-8.913, -8.110]<br>Combined -5.077% [-5.301, -4.894]<br>Full -2.437% [-3.183, -1.930] | 行比1.000；F/T=560/76 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Hy3 K192 / 32K | 8x1_compact vs 8x1 | Down 1.155505 / 401.43<br>Combined 1.995348 / 232.47<br>Full 5.190741 / 268.09 | Down -5.678% [-6.140, -5.428]<br>Combined -3.225% [-3.482, -2.895]<br>Full -1.322% [-1.604, -0.983] | 行比1.000；F/T=1120/152 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 1K | 8x1 vs 1x4 | Down 1.164325 / 36.89<br>Combined 1.204105 / 35.67<br>Full 3.413273 / 37.75 | Down -151.226% [-155.063, -148.379]<br>Combined -142.406% [-146.331, -137.615]<br>Full -181.918% [-188.237, -176.145] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 1K | 8x1_compact vs 1x4 | Down 0.496522 / 86.50<br>Combined 0.529762 / 81.07<br>Full 1.229064 / 104.84 | Down -1.356% [-2.068, -0.181]<br>Combined -1.015% [-2.342, -0.081]<br>Full -0.543% [-1.311, +0.354] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 2K | 8x1 vs 1x4 | Down 1.175384 / 73.08<br>Combined 1.243485 / 69.08<br>Full 3.517614 / 73.26 | Down -150.686% [-153.738, -148.302]<br>Combined -136.456% [-141.666, -131.599]<br>Full -172.528% [-176.042, -165.517] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 2K | 8x1_compact vs 1x4 | Down 0.498502 / 172.32<br>Combined 0.561742 / 152.92<br>Full 1.328866 / 193.92 | Down -1.526% [-2.221, -0.948]<br>Combined -1.596% [-2.724, +0.209]<br>Full -0.404% [-1.484, +0.621] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 4K | 8x1 vs default | Down 1.188665 / 144.53<br>Combined 1.310905 / 131.05<br>Full 3.727535 / 138.27 | Down -48.814% [-49.814, -47.645]<br>Combined -44.631% [-45.645, -43.850]<br>Full -61.546% [-62.693, -60.209] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 4K | 8x1_compact vs default | Down 0.852463 / 201.53<br>Combined 0.981484 / 175.04<br>Full 2.375409 / 216.97 | Down -5.253% [-6.015, -4.520]<br>Combined -5.848% [-6.363, -5.389]<br>Full -1.855% [-2.342, -1.450] | 行比1.000；F/T=0/1024 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 8K | 8x1 vs default | Down 1.182025 / 290.69<br>Combined 1.413866 / 243.02<br>Full 3.951435 / 260.87 | Down -3.225% [-3.565, -2.594]<br>Combined -4.667% [-4.926, -4.326]<br>Full -16.959% [-17.630, -16.364] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 8K | 8x1_compact vs default | Down 1.231064 / 279.11<br>Combined 1.455046 / 236.14<br>Full 3.496974 / 294.77 | Down -6.845% [-7.605, -6.263]<br>Combined -7.981% [-8.259, -7.214]<br>Full -3.568% [-4.238, -2.779] | 行比1.000；F/T=0/1536 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 16K | 8x1 vs 8x1_compact | Down 2.204528 / 311.72<br>Combined 2.662730 / 258.08<br>Full 7.684910 / 268.26 | Down -40.647% [-40.764, -40.282]<br>Combined -31.887% [-32.212, -31.526]<br>Full -38.882% [-39.394, -38.490] | 行比1.600 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K512 / 32K | 8x1 vs 8x1_compact | Down 3.454893 / 397.81<br>Combined 4.385457 / 313.40<br>Full 12.203328 / 337.87 | Down -11.420% [-11.835, -11.267]<br>Combined -8.717% [-8.981, -8.195]<br>Full -11.890% [-12.109, -11.619] | 行比1.200 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 1K | 8x1 vs 1x4 | Down 0.717763 / 29.92<br>Combined 0.757523 / 28.35<br>Full 1.853927 / 34.75 | Down -154.549% [-156.110, -153.667]<br>Combined -139.632% [-140.897, -138.075]<br>Full -170.449% [-174.840, -167.442] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 1K | 8x1_compact vs 1x4 | Down 0.289841 / 74.09<br>Combined 0.324421 / 66.19<br>Full 0.700282 / 92.00 | Down -2.801% [-3.281, -2.443]<br>Combined -2.835% [-3.117, -2.352]<br>Full -1.009% [-1.787, -0.662] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 2K | 8x1 vs 1x4 | Down 0.722023 / 59.49<br>Combined 0.789643 / 54.39<br>Full 1.947187 / 66.17 | Down -155.309% [-157.538, -154.143]<br>Combined -132.009% [-134.285, -131.055]<br>Full -153.001% [-154.160, -148.546] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 2K | 8x1_compact vs 1x4 | Down 0.290561 / 147.82<br>Combined 0.349142 / 123.01<br>Full 0.784063 / 164.34 | Down -3.221% [-3.562, -2.727]<br>Combined -2.423% [-2.810, -1.969]<br>Full -1.132% [-1.324, -0.801] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 4K | 8x1 vs 1x4 | Down 0.724143 / 118.62<br>Combined 0.855323 / 100.43<br>Full 2.149428 / 119.89 | Down -51.320% [-52.161, -49.673]<br>Combined -42.246% [-43.761, -40.037]<br>Full -58.592% [-61.267, -57.259] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 4K | 8x1_compact vs 1x4 | Down 0.491682 / 174.71<br>Combined 0.614363 / 139.82<br>Full 1.360185 / 189.46 | Down -1.938% [-2.914, -1.483]<br>Combined -1.898% [-3.154, -0.973]<br>Full -0.592% [-1.162, +0.296] | 行比1.000；F/T=0/1024 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 8K | 8x1 vs 1x4 | Down 0.722503 / 237.78<br>Combined 0.965744 / 177.89<br>Full 2.403909 / 214.40 | Down -7.630% [-8.593, -7.071]<br>Combined -7.207% [-7.998, -6.484]<br>Full -17.157% [-18.995, -15.554] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 8K | 8x1_compact vs 1x4 | Down 0.684502 / 250.98<br>Combined 0.913564 / 188.05<br>Full 2.069568 / 249.04 | Down -1.748% [-2.074, -1.275]<br>Combined -1.482% [-2.606, -1.120]<br>Full -0.303% [-2.353, +0.350] | 行比1.000；F/T=0/1536 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 16K | 8x1 vs 8x1_compact | Down 1.347185 / 255.05<br>Combined 1.829927 / 187.77<br>Full 4.558038 / 226.15 | Down -38.890% [-39.401, -38.455]<br>Combined -26.862% [-27.278, -26.305]<br>Full -34.169% [-34.892, -33.347] | 行比1.600 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen397 K256 / 32K | 8x1 vs 8x1_compact | Down 2.003628 / 342.98<br>Combined 2.928071 / 234.69<br>Full 7.338089 / 280.94 | Down -9.714% [-10.069, -9.499]<br>Combined -6.562% [-6.944, -6.322]<br>Full -10.112% [-10.410, -9.797] | 行比1.200 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 1K | 8x1 vs default | Down 0.344961 / 49.80<br>Combined 0.360261 / 47.69<br>Full 0.997203 / 51.68 | Down -135.759% [-136.958, -132.727]<br>Combined -126.191% [-127.369, -124.378]<br>Full -163.698% [-165.674, -162.026] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 1K | 8x1_compact vs default | Down 0.157761 / 108.90<br>Combined 0.172580 / 99.55<br>Full 0.390421 / 132.01 | Down -7.620% [-8.200, -6.255]<br>Combined -8.001% [-8.714, -6.920]<br>Full -3.281% [-3.801, -2.588] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 2K | 8x1 vs default | Down 0.350601 / 98.00<br>Combined 0.376301 / 91.31<br>Full 1.050524 / 98.12 | Down -132.032% [-133.987, -131.115]<br>Combined -119.234% [-121.169, -118.440]<br>Full -141.150% [-141.729, -140.369] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 2K | 8x1_compact vs default | Down 0.159400 / 215.56<br>Combined 0.184720 / 186.01<br>Full 0.448061 / 230.06 | Down -5.533% [-6.212, -4.347]<br>Combined -7.746% [-8.648, -6.946]<br>Full -2.879% [-3.429, -2.448] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 4K | 8x1 vs default | Down 0.357261 / 192.35<br>Combined 0.404601 / 169.84<br>Full 1.117544 / 184.47 | Down -52.673% [-54.354, -51.771]<br>Combined -47.714% [-49.311, -46.714]<br>Full -55.157% [-56.137, -54.674] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 4K | 8x1_compact vs default | Down 0.252821 / 271.81<br>Combined 0.300141 / 228.96<br>Full 0.742682 / 277.59 | Down -7.920% [-8.366, -7.362]<br>Combined -9.244% [-9.825, -8.800]<br>Full -3.257% [-3.618, -2.635] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 8K | 8x1 vs 8x1_compact | Down 0.371681 / 369.78<br>Combined 0.462782 / 296.98<br>Full 1.287825 / 320.17 | Down -14.688% [-14.830, -14.592]<br>Combined -11.339% [-11.450, -11.245]<br>Full -3.900% [-4.014, -3.746] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 16K | 8x1 vs 8x1_compact | Down 0.647283 / 424.66<br>Combined 0.818163 / 335.97<br>Full 2.384229 / 345.87 | Down -1.675% [-1.741, -1.614]<br>Combined -1.313% [-1.405, -1.220]<br>Full -0.272% [-0.347, -0.178] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K512 / 32K | 8x1_compact vs 8x1 | Down 1.221484 / 450.07<br>Combined 1.541366 / 356.67<br>Full 4.660798 / 353.86 | Down -1.929% [-1.960, -1.906]<br>Combined -1.558% [-1.612, -1.510]<br>Full -0.483% [-0.557, -0.427] | 行比1.000；F/T=1024/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 1K | 8x1 vs 1x4 | Down 0.209561 / 40.99<br>Combined 0.224861 / 38.20<br>Full 0.556402 / 46.32 | Down -155.299% [-156.259, -154.635]<br>Combined -130.972% [-132.296, -129.474]<br>Full -141.205% [-142.453, -110.202] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 1K | 8x1_compact vs 1x4 | Down 0.089021 / 96.49<br>Combined 0.104621 / 82.11<br>Full 0.237321 / 108.59 | Down -8.406% [-8.836, -7.925]<br>Combined -7.673% [-8.182, -6.582]<br>Full -2.654% [-3.561, +7.060] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 2K | 8x1 vs 1x4 | Down 0.214641 / 80.04<br>Combined 0.240161 / 71.53<br>Full 0.604562 / 85.25 | Down -152.375% [-153.056, -151.736]<br>Combined -116.457% [-117.343, -115.611]<br>Full -117.776% [-118.729, -111.402] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 2K | 8x1_compact vs 1x4 | Down 0.092120 / 186.49<br>Combined 0.117860 / 145.76<br>Full 0.284101 / 181.41 | Down -8.240% [-8.720, -7.570]<br>Combined -6.292% [-6.687, -5.884]<br>Full -2.457% [-2.627, -0.783] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 4K | 8x1 vs 1x4 | Down 0.216761 / 158.51<br>Combined 0.264881 / 129.72<br>Full 0.665902 / 154.80 | Down -58.280% [-59.112, -57.244]<br>Combined -42.929% [-44.135, -41.900]<br>Full -44.900% [-45.548, -44.665] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 4K | 8x1_compact vs 1x4 | Down 0.144920 / 237.09<br>Combined 0.193040 / 177.99<br>Full 0.467041 / 220.71 | Down -5.295% [-5.805, -4.957]<br>Combined -4.175% [-4.612, -3.546]<br>Full -1.673% [-2.034, -1.389] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 8K | 8x1 vs 8x1_compact | Down 0.223880 / 306.95<br>Combined 0.314342 / 218.61<br>Full 0.811483 / 254.05 | Down -9.829% [-10.293, -9.524]<br>Combined -7.070% [-7.287, -6.582]<br>Full -2.751% [-2.896, -2.504] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 16K | 8x1 vs 8x1_compact | Down 0.386142 / 355.93<br>Combined 0.557702 / 246.44<br>Full 1.513326 / 272.46 | Down -3.585% [-3.913, -3.438]<br>Combined -2.538% [-2.747, -2.407]<br>Full -0.903% [-1.130, -0.733] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Qwen35 K256 / 32K | 8x1_compact vs 8x1 | Down 0.732343 / 375.34<br>Combined 1.051824 / 261.33<br>Full 2.921811 / 282.23 | Down -2.745% [-3.043, -2.452]<br>Combined -1.873% [-2.117, -1.608]<br>Full -0.671% [-0.886, -0.392] | 行比1.000；F/T=1024/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 1K | 8x1 vs 1x4 | Down 0.765603 / 33.66<br>Combined 0.807883 / 31.90<br>Full 2.023068 / 38.21 | Down -159.925% [-161.318, -158.129]<br>Combined -140.847% [-141.915, -139.688]<br>Full -175.506% [-179.222, -168.960] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 1K | 8x1_compact vs 1x4 | Down 0.303021 / 85.04<br>Combined 0.344902 / 74.72<br>Full 0.749923 / 103.09 | Down -2.612% [-2.906, -2.258]<br>Combined -2.588% [-3.252, -2.198]<br>Full -0.944% [-2.087, +0.005] | 行比1.000；F/T=0/384 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 2K | 8x1 vs 1x4 | Down 0.760523 / 67.77<br>Combined 0.837723 / 61.52<br>Full 2.145188 / 72.08 | Down -159.256% [-161.003, -156.658]<br>Combined -131.096% [-133.262, -128.267]<br>Full -152.804% [-155.656, -149.366] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 2K | 8x1_compact vs 1x4 | Down 0.303262 / 169.95<br>Combined 0.374381 / 137.67<br>Full 0.858443 / 180.12 | Down -2.526% [-2.881, -2.228]<br>Combined -2.503% [-2.842, -1.849]<br>Full -0.993% [-1.277, -0.536] | 行比1.000；F/T=0/384 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 4K | 8x1 vs 1x4 | Down 0.769423 / 133.97<br>Combined 0.908964 / 113.40<br>Full 2.345249 / 131.86 | Down -46.367% [-47.647, -44.537]<br>Combined -37.726% [-39.498, -36.423]<br>Full -56.448% [-58.552, -54.601] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 4K | 8x1_compact vs 1x4 | Down 0.538602 / 191.38<br>Combined 0.670543 / 153.72<br>Full 1.500906 / 206.03 | Down -1.773% [-2.149, -1.177]<br>Combined -1.402% [-2.295, -0.884]<br>Full +0.260% [-1.140, +1.303] | 行比1.000；F/T=0/768 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 8K | 8x1 vs 1x4 | Down 0.765323 / 269.37<br>Combined 1.027884 / 200.57<br>Full 2.587950 / 238.98 | Down -2.491% [-3.564, -1.766]<br>Combined -2.101% [-2.507, -1.486]<br>Full -15.251% [-16.167, -14.355] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 8K | 8x1_compact vs 1x4 | Down 0.760603 / 271.05<br>Combined 1.019624 / 202.19<br>Full 2.274948 / 271.86 | Down -1.308% [-2.183, -0.935]<br>Combined -1.246% [-2.316, -0.910]<br>Full -1.004% [-1.309, -0.327] | 行比1.000；F/T=0/1152 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 16K | 8x1 vs 8x1_compact | Down 1.465505 / 281.35<br>Combined 1.986607 / 207.55<br>Full 4.976279 / 248.57 | Down -13.994% [-14.264, -13.698]<br>Combined -10.390% [-10.606, -10.075]<br>Full -18.518% [-18.832, -18.021] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| Xiaomi K256 / 32K | 8x1 vs 1x4 | Down 2.212468 / 372.72<br>Combined 3.209952 / 256.90<br>Full 7.934249 / 311.80 | Down +3.519% [+3.201, +3.692]<br>Combined +2.221% [+2.064, +2.486]<br>Full -1.221% [-1.658, -0.830] | 行比1.091 | Down有收益，但Full配对回退；对现选路径Q1≤0：Full；对旧推荐未过：Full |
| Xiaomi K256 / 32K | 8x1_compact vs 1x4 | Down 2.249128 / 366.65<br>Combined 3.242412 / 254.33<br>Full 7.840829 / 315.52 | Down +2.258% [+2.078, +2.538]<br>Combined +1.517% [+1.378, +1.641]<br>Full +0.156% [-0.398, +0.698] | 行比1.000；F/T=768/1152 | 未达三phase共同改善门槛；对现选路径Q1≤0：Full；对旧推荐未过：Full |
| H3 K384 / 1K | 8x1 vs 1x4 | Down 0.382241 / 50.56<br>Combined 0.403102 / 47.95<br>Full 1.046144 / 55.42 | Down -142.880% [-144.527, -141.826]<br>Combined -125.819% [-128.231, -125.092]<br>Full -153.803% [-158.802, -150.735] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 1K | 8x1_compact vs 1x4 | Down 0.165601 / 116.71<br>Combined 0.186641 / 103.55<br>Full 0.423581 / 136.89 | Down -4.949% [-5.430, -4.697]<br>Combined -4.453% [-4.870, -4.105]<br>Full -2.311% [-2.874, -1.299] | 行比1.000；F/T=0/128 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 2K | 8x1 vs 1x4 | Down 0.384822 / 100.45<br>Combined 0.424281 / 91.11<br>Full 1.099944 / 105.43 | Down -141.498% [-143.151, -140.180]<br>Combined -114.612% [-116.928, -113.545]<br>Full -129.886% [-133.699, -127.520] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 2K | 8x1_compact vs 1x4 | Down 0.168201 / 229.81<br>Combined 0.206981 / 186.75<br>Full 0.488182 / 237.54 | Down -5.147% [-5.385, -4.870]<br>Combined -4.063% [-4.489, -3.813]<br>Full -1.677% [-2.486, -0.944] | 行比1.000；F/T=0/128 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 4K | 8x1 vs 1x4 | Down 0.387241 / 199.64<br>Combined 0.459622 / 168.20<br>Full 1.194744 / 194.12 | Down -28.485% [-28.865, -27.762]<br>Combined -22.996% [-23.672, -22.665]<br>Full -41.545% [-42.372, -40.235] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 4K | 8x1_compact vs 1x4 | Down 0.313421 / 246.66<br>Combined 0.385861 / 200.36<br>Full 0.855024 / 271.25 | Down -3.630% [-3.880, -3.425]<br>Combined -3.020% [-3.149, -2.801]<br>Full -1.209% [-1.916, -0.929] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 8K | 8x1_compact vs 8x1 | Down 0.400122 / 386.43<br>Combined 0.541862 / 285.35<br>Full 1.431445 / 324.05 | Down -1.886% [-1.997, -1.671]<br>Combined -0.978% [-1.352, -0.767]<br>Full -0.292% [-0.585, +0.084] | 行比1.000；F/T=128/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 16K | 8x1 vs 8x1_compact | Down 0.780203 / 396.36<br>Combined 1.058244 / 292.22<br>Full 2.739850 / 338.60 | Down -15.967% [-16.031, -15.890]<br>Combined -11.783% [-12.129, -11.283]<br>Full -4.131% [-4.834, -3.512] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |
| H3 K384 / 32K | 8x1 vs 8x1_compact | Down 1.362625 / 453.89<br>Combined 1.912447 / 323.39<br>Full 5.235179 / 354.41 | Down -2.633% [-2.704, -2.587]<br>Combined -1.863% [-1.984, -1.772]<br>Full -0.205% [-0.774, +0.201] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对旧推荐未过：Down/Combined/Full |

<a id="migration-confirmation"></a>

### 1.5 删除损失与两点独立确认

9/9[删除反事实](../../../../tests/contrib/moe/results/final_matrix_20260909/path_removal.json)按真实path过滤：原选存活则保持，策略损失严格0；被删则从剩余三phase非支配集中选Full绝对median最小者，同值按label。仅删1x8/同时删1x8+2x4都只影响Hy3 4K和8K，其余40点保留（仅删1x8时36点无该候选，同时删时12点无两者）；仅删2x4无改选，42点标签保留，其中12个K512点原本不含2x4。随后删4x1也无改选，因为当前42点没有选中它。标签策略损失0不是测得wall time零差异，也不是这些路径所有shape永远无益。

当时强制重选存在同样本挑最优偏差，**不等于删除后重测**。4K非支配1x4-P128与compact的Full分别0.832323 / 208.99、0.838423 / 207.47，按规则选前者；ordinary为0.886363 / 196.25。8K选compact，1x4-P128 Full为1.468506 / 236.90，ordinary为1.559246 / 223.12。保留两受影响点全部原值，不重复40行零策略损失：

| 9/9替代 | Phase | 原1x8 ms / T → 替代 ms / T | 时延增幅 [IQR]% | 吞吐损失 [IQR]% |
|---|---|---|---|---|
| Hy3 4K → 1x4-P128 | Down | 0.237281 / 244.36 → 0.231561 / 250.40 | -2.691 [-3.180, -2.346] | -2.765 [-3.285, -2.403] |
| Hy3 4K → 1x4-P128 | Combined | 0.335621 / 172.76 → 0.339221 / 170.93 | +0.882 [+0.451, +1.159] | +0.875 [+0.449, +1.145] |
| Hy3 4K → 1x4-P128 | Full | 0.833103 / 208.79 → 0.832323 / 208.99 | -0.211 [-1.637, +0.917] | -0.211 [-1.665, +0.909] |
| Hy3 8K → compact | Down | 0.406102 / 285.55 → 0.374462 / 309.68 | -7.222 [-7.708, -6.589] | -7.784 [-8.352, -7.054] |
| Hy3 8K → compact | Combined | 0.594262 / 195.14 → 0.592202 / 195.82 | -0.005 [-0.374, +0.482] | -0.005 [-0.376, +0.480] |
| Hy3 8K → compact | Full | 1.433525 / 242.68 → 1.439025 / 241.76 | +0.217 [-0.473, +1.324] | +0.217 [-0.475, +1.307] |

单phase包络另外两项非零：Hy3 4K Combined从1x8→1x4-P128，绝对median增幅+1.073%，配对+0.882% [+0.451,+1.159]；8K Full从1x8→compact，绝对median+0.384%，配对+0.217% [-0.473,+1.324]。仅删1x8与同时删两支分别重复这两项，其余最优label仍在、包络损失0；不能把独立phase包络拼成最终推荐。

#### 2026-09-10：预先固定替代的独立48轮

4K→direct_m64/1x4-P128，8K→compact，保持**删除前冻结源码和原9候选顺序**，仅分析两个预先固定的替代。不是删除后或可读性重构后计时，不是新42点，没有重新选路、没有与9/9拼96轮、没有删样本或按方向追加。证据：[计划](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/plan.json)、[raw](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/hy3_abba48.json)、[审计](../../../../tests/contrib/moe/results/hy3_path_removal_confirm_20260910/audited.json)。

| 9/10替代 | Phase | 同run原1x8 ms / T → 替代 ms / T | 时延增幅 [IQR]% | 吞吐损失 [IQR]% | 替代快/慢轮 |
|---|---|---|---|---|---|
| Hy3 4K → 1x4-P128 | Down | 0.238521 / 243.09 → 0.232161 / 249.75 | -2.771 [-3.197, -2.283] | -2.850 [-3.303, -2.336] | 48 / 0 |
| Hy3 4K → 1x4-P128 | Combined | 0.339081 / 171.00 → 0.340062 / 170.50 | +0.289 [-0.022, +0.856] | +0.289 [-0.022, +0.849] | 12 / 36 |
| Hy3 4K → 1x4-P128 | Full | 0.826463 / 210.47 → 0.824363 / 211.01 | -0.402 [-2.131, +1.273] | -0.404 [-2.177, +1.257] | 28 / 20 |
| Hy3 8K → compact | Down | 0.396321 / 292.60 → 0.363761 / 318.79 | -8.194 [-8.574, -7.756] | -8.926 [-9.378, -8.408] | 48 / 0 |
| Hy3 8K → compact | Combined | 0.584522 / 198.39 → 0.579722 / 200.03 | -0.696 [-1.037, -0.171] | -0.701 [-1.048, -0.171] | 41 / 7 |
| Hy3 8K → compact | Full | 1.428106 / 243.60 → 1.428066 / 243.61 | +0.126 [-0.781, +0.757] | +0.126 [-0.787, +0.751] | 23 / 25 |

Down两点均稳定收益，8K Combined也收益；4K Combined及两点Full IQR跨零，**只表示本轮未观察稳定损失，不是等价证明**。8K Full绝对median略低、配对增幅为正是不同统计量，不矛盾；9/9的4K Combined稳定小损失仍保留。

确认共5,184全候选event/1,152目标pair event、2,592逐轮检查+36初始检查，最大相对default rel_l2=0；12artifact/16kernel与前run ISA/ELF/资源精确一致。全1性能W不等于新随机W验证。GPU4 PCI `0001:0B:00.0`、ID `0xbe022834ccc51849`，PTL Enabled/VECTOR,F8、请求1800MHz/650W、NUMA1；退出auto、PTL Disabled/N/A、busy0/VRAM0、NUMA1。其它卡有负载非全机空闲，105.369885秒父墙钟不是kernel时延。

<a id="validation-boundaries"></a>

### 1.6 验证边界

- 9/9矩阵71artifact/93kernel零private/spill/scratch；fresh逐轮检查27,792次，最大相对default rel_l2=2.06569639e-05，初始另计；仅K192/256/384/512，**不外推K320/K640**。全1W、单seed均衡routing不能替代生产trace或随机压力。
- 既有原FIFO随机180配置/3,600检查；[cleanup审计](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/final_audited.json)28组ELF/ISA/host严格比较、40paired kernel、CPU1,625通过，new_gpu_checks=0，不是新跑180配置或任意shape证明。
- [删1x8/2x4交付](../../../../tests/contrib/moe/results/remove_down_paths_20260910/delivery.json)：CPU990通过，6离线compact对/18paired kernel，无新GPU/性能；误收GPU的31项不可用记录未当通过。[删4x1交付](../../../../tests/contrib/moe/results/remove_4x1_20260910/delivery.json)：CPU994、0改选，无新编译/GPU/性能。
- 历史20模块/22源是各run身份，不是当前计数；任务表并入compact后，现行源码集合为18项。未新增ATT/PMC或occupancy/cache/bank计数；零spill、功能通过及IQR跨零都不是性能保证。

<a id="pipeline"></a>

## 二、流水线设计

记录 **2026-09-09原pack消费FIFO、PTPC、默认1N循环**。这组合启动、B周转、C打包/回写和两组wave错相，不是几十级独立流水。**逐stage表左Memory、右紧接的Compute；每行是单wave一次逻辑配对，不是一个cycle或全WG同步同行。** 可读性重构保留这些事件/依赖；新的代码身份与旧数值证据分开核验。

<a id="notation"></a>

### 2.1 符号与读表方法

| 符号 | 含义 |
|---|---|
| N/n/T | N输出总列数，n为N128块编号，T=N/128；标题N=0/1指n=0/1，不是总列数。首轮/过渡表假设有足够后续块。 |
| L/H | 当前块前/后N64，列0–63/64–127。 |
| K0/K1 | K256的两个K128，坐标0–127/128–255；其它K按实际分块。 |
| s/q | K256 s=0..3、q=4n+s；一般s=0..2KS−1、q=2KS·n+s，前KS拍L后KS拍H。 |
| A | prologue gather到VGPR的activation，沿完整N复用。 |
| B[n,L/H,Kk] | N64×真实归约块的权重，各wave协作搬运后按MFMA布局读。 |
| Q[q] | 当前Compute消费B。K256 Q[4n..4n+3]为L/K0、L/K1、H/K0、H/K1。 |
| P0/P1 | 每wave负责搬运的B片段VGPR槽，load可在途，提交须wait；用P[q&1]，提交后同槽发新预取。不是Compute packet或LDS槽号。 |
| j=0..3 | N128内四N32，列0–31/32–63/64–95/96–127。 |
| C[n,j] | FP32累加结果，当前半区全部K完成才数学完成。 |
| C_bf16[n,j] | 同编号C缩放/pack后的BF16，留VGPR待CShuffle，不再是FP32累加器。 |
| r0/r1 | 每wave两个实际不相邻的M16条带；M维，不是packet或新旧C双缓冲。 |
| packet0/1 | 当前N64的两个N32，每个都含r0/r1；BK128每packet16 MFMA/wave、整192为24，每Compute两个packet，不是两wave组。 |
| S[n,j] | weight scale；PTPC每份2条VMEM，activation scale×routing weight已融合行scale。 |
| pack | 完成C结合weight/行scale生成BF16；**pack≠global store**。 |
| C_bf16[n,0/1].r0 | C0/C1各自r0合成M16×N64；2/3和r1同理，斜杠非除法。 |
| CShuffle | VGPR→LDS→VGPR→Global，计算布局重排为合并store布局。 |
| load/store×2 | 每wave两条VMEM，不是两个元素；LDS不计vmcnt。 |
| vmcnt(x)/lgkmcnt(x) | 等计数≤x，不是等x条完成；前者普通VMEM load/store依赖，后者此处LDS等，越小越严格。 |
| KS | K192/256/320/384/512/640分别1/2/2/3/4/5；K320不是ceil(320/128)=3。 |

沿三条数据路径读表：

1. **B供数：Global → P → LDS → Breg。** 正常读`Q[q]`、提交`Q[q+1]`、预取`Q[q+3]`，`Q[q+2]`在另一P槽；启动建立LDS中的Q0及P0/P1中的Q1/Q2种子。
2. **C计算与打包：FP32 → pack → BF16。** K256当前Compute使用上一Memory的scale；同一表行新load供后续Compute。`C[n,3]`到下一N的**Compute 0**才pack，不是更新下一N的`C[n+1,0]`。
3. **C回写：上一N的BF16在当前Memory回写，当前FP32继续计算。** 旧FP32被pack消费后才clear；旧BF16被CShuffle写消费后才复用，不能整轮开始就覆盖全部C。不同列处理不同代数据，不是一份数据一拍走完全程。

group0为wave0–3，group1为wave4–7；两组以scalar条件和barrier错相，一组memory时另一组可compute，不保证每cycle完美重叠。所有WG-uniform退出在错相前，尾部平衡barrier。K256的“上一拍scale”不套其它K：K512/K640保留多拍，K192/K320同Compute可pack两份。B的P槽和LDS槽不同；K192 P=16B+8B，K320 P0/P1=24B/16B每lane。K192深稳态另需`n+2<T`；短N裁剪，不能套表。wait按真实指令事件，不算wait后才issue的B，编译器可合并冗余wait。

| K | 归约块坐标 | 每N M/C对 | 每packet MFMA | 每N MFMA/wave | 跨N FP32 | 跨N BF16 | LDS |
|---|---|---:|---|---:|---|---|---|
| 192 | K0=0–191 | 2 | 24 | 96 | C2/C3 | C0/C1 | 64KiB |
| 256 | K0=0–127；K1=128–255 | 4 | 16 | 128 | C3 | C0/C1/C2 | 48KiB |
| 320 | K0=0–127；K1=128–319 | 4 | K0=16；K1=24 | 160 | C2/C3 | C0/C1 | 56KiB |
| 384 | K0=0–127；K1=128–255；K2=256–383 | 6 | 16 | 192 | C3 | C0/C1/C2 | 48KiB |
| 512 | K0=0–127；K1=128–255；K2=256–383；K3=384–511 | 8 | 16 | 256 | C3 | C0/C1/C2 | 48KiB |
| 640 | K0=0–127；K1=128–255；K2=256–383；K3=384–511；K4=512–639 | 10 | 16 | 320 | C3 | C0/C1/C2 | 48KiB |

共同WG=M256×N128/8 waves，A跨N驻留。固定原pack，不恢复Memory-pack/delayed/formal：后移曾延长FP32/scale活跃期，K256/PTPC VGPR176→202、K320 214→232且无稳定收益证据。K192/K320原本两FP32+两BF16，不是后移新加。

<a id="pipeline-k256"></a>

### 2.2 K256：Prologue、N=0、N=1、稳态与尾部

K256/PTPC/default1N，每N四M/C对，Compute两个16-MFMA packet；不是旧整N128/64-MFMA大stage账本。

#### Prologue（尚无正常MFMA packet）

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | 准备sorted IDs、expert ID；加载并融合routing weight与activation scale | 必要的元数据／LDS同步 | 尚未计算C |
| 首B与A | 先发出`B[0,L,K0] → P0`，再gather两个K128块的A到VGPR | A跨后续N块复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的`P0` | `vmcnt(4)` → `P0 → LDS B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,H,K0] → P1` | `vmcnt(1)` → barrier | 初始FP32累加区清零；**没有旧`C_bf16`** |

进入N0时LDS只有首B，P0/P1承载后两B预取。

#### N=0：无旧C回写

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(2)`** | `P0 → B[0,L,K1]` | `B[0,H,K1] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | `vmcnt(3)` | `P1 → B[0,H,K0]` | `B[1,L,K0] → P1` | **Compute 1** | Packet 0：`C[0,0]`累加K1，完成<br>Packet 1：`C[0,1]`累加K1，完成 | `C_bf16[0,0]`<br>与packet 1交织 | `S[0,0]` |
| **Memory 2** | `S[0,2]` | **无** | `B[0,H,K0]` | `vmcnt(3)` | `P0 → B[0,H,K1]` | `B[1,L,K1] → P0` | **Compute 2** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,1]`<br>与packet 0交织 | `S[0,1]` |
| **Memory 3** | `S[0,3]` | **无** | `B[0,H,K1]` | `vmcnt(3)` | `P1 → B[1,L,K0]` | `B[1,H,K0] → P1` | **Compute 3** | Packet 0：`C[0,2]`累加K1，完成<br>Packet 1：`C[0,3]`累加K1，完成 | `C_bf16[0,2]`<br>与packet 1交织 | `S[0,2]` |

N0结束保留BF16 C0/C1/C2，FP32 C3与S3。**首Memory0独立剥离实际vmcnt(2)，不可用通用账本3覆盖。**

#### N=1：首次回写过渡

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]` | `C_bf16[0,0/1].r0` | `B[1,L,K0]` | **`vmcnt(5)`** | `P0 → B[1,L,K1]` | `B[1,H,K1] → P0` | **Compute 0** | Packet 0：`C[1,0]`清零＋K0贡献<br>Packet 1：`C[1,1]`清零＋K0贡献 | **`C_bf16[0,3]`**<br>与packet 0交织 | **`S[0,3]`** |
| **Memory 1** | `S[1,1]` | `C_bf16[0,0/1].r1` | `B[1,L,K1]` | `vmcnt(7)` | `P1 → B[1,H,K0]` | `B[2,L,K0] → P1` | **Compute 1** | Packet 0：`C[1,0]`累加K1，完成<br>Packet 1：`C[1,1]`累加K1，完成 | `C_bf16[1,0]`<br>与packet 1交织 | `S[1,0]` |
| **Memory 2** | `S[1,2]` | `C_bf16[0,2/3].r0` | `B[1,H,K0]` | `vmcnt(7)` | `P0 → B[1,H,K1]` | `B[2,L,K1] → P0` | **Compute 2** | Packet 0：`C[1,2]`清零＋K0贡献<br>Packet 1：`C[1,3]`清零＋K0贡献 | `C_bf16[1,1]`<br>与packet 0交织 | `S[1,1]` |
| **Memory 3** | `S[1,3]` | `C_bf16[0,2/3].r1` | `B[1,H,K1]` | `vmcnt(7)` | `P1 → B[2,L,K0]` | `B[2,H,K0] → P1` | **Compute 3** | Packet 0：`C[1,2]`累加K1，完成<br>Packet 1：`C[1,3]`累加K1，完成 | `C_bf16[1,2]`<br>与packet 1交织 | `S[1,2]` |

首5保护旧S[0,3]，N0 Memory3无2条输出store，比深稳态少2个年轻请求。

#### N≥2：完整稳态，未排空

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]` | `C_bf16[n−1,0/1].r0` | `B[n,L,K0]` | `vmcnt(7)` | `P0 → B[n,L,K1]` | `B[n,H,K1] → P0` | **Compute 0** | Packet 0：`C[n,0]`清零＋K0贡献<br>Packet 1：`C[n,1]`清零＋K0贡献 | **`C_bf16[n−1,3]`**<br>与packet 0交织 | **`S[n−1,3]`** |
| **Memory 1** | `S[n,1]` | `C_bf16[n−1,0/1].r1` | `B[n,L,K1]` | `vmcnt(7)` | `P1 → B[n,H,K0]` | `B[n+1,L,K0] → P1` | **Compute 1** | Packet 0：`C[n,0]`累加K1，完成<br>Packet 1：`C[n,1]`累加K1，完成 | `C_bf16[n,0]`<br>与packet 1交织 | `S[n,0]` |
| **Memory 2** | `S[n,2]` | `C_bf16[n−1,2/3].r0` | `B[n,H,K0]` | `vmcnt(7)` | `P0 → B[n,H,K1]` | `B[n+1,L,K1] → P0` | **Compute 2** | Packet 0：`C[n,2]`清零＋K0贡献<br>Packet 1：`C[n,3]`清零＋K0贡献 | `C_bf16[n,1]`<br>与packet 0交织 | `S[n,1]` |
| **Memory 3** | `S[n,3]` | `C_bf16[n−1,2/3].r1` | `B[n,H,K1]` | `vmcnt(7)` | `P1 → B[n+1,L,K0]` | `B[n+1,H,K0] → P1` | **Compute 3** | Packet 0：`C[n,2]`累加K1，完成<br>Packet 1：`C[n,3]`累加K1，完成 | `C_bf16[n,2]`<br>与packet 1交织 | `S[n,2]` |

#### 顺序与尾部

Pack列是同编号C→缩放/pack→BF16，不是store。Memory按功能列，非严格左到右：**scale→CShuffle写/读→B packet0读→lgkmcnt(4)→BF16 store×2→B packet1读→vmcnt→B提交→B预取**。首N无CShuffle/store，首M0另lgkmcnt(0)再交接。两4-wave组错相，非WG同步逐行。

足够长PTPC最后N wait=**7/7/6/6**，末拍无下一B仍保护当前C2 pack；其后vmcnt(0)补C3，按L/H、r0/r1回写末N全部BF16、drain依赖、平衡barrier。短N按实际请求裁剪，不当所有ISA逐条模板。源码：[主builder](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py)、[循环](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py)、[账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_schedule.py)。

<a id="pipeline-k192"></a>

### 2.3 K192：Prologue、N=0、N=1、稳态与尾部

**整BK192、每N两拍**，Q[2n]=L/K0，Q[2n+1]=H/K0。每份N64×K192 B每lane16B+8B两VMEM，不补K256；每Compute两个24-MFMA packet，共48 MFMA/wave。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | 准备sorted IDs、expert与行scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | 发出`B[0,L,K0]`的16B＋8B加载；gather完整K192的A | 只加载三个K64范围，不加载不存在的第四段 | A留在VGPR跨N复用 |
| 首B落LDS | 使用启动临时B片段 | **`vmcnt(0)`** → 两段写入LDS槽0的L → **`lgkmcnt(0)`** → barrier | FP32累加区清零 |
| 两个预取种子 | `B[0,H,K0] → P0`；`B[1,L,K0] → P1`，各load ×2 | **此处不额外套用K256的`vmcnt(1)`种子等待**；由首Memory保护提交 | 无旧C或旧BF16 |

两个完整N128×K192 LDS槽48KiB+CShuffle16KiB=64KiB；当前N槽n&1，L/H同槽两个半区，下N另一槽。

#### N=0

| **Memory Stage** | **Scale load ×4** | **CShuffle → C_bf16 store ×4** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]`、`S[0,1]`，各load ×2 | **无** | `B[0,L,K0]` | **`vmcnt(6)`** | `P0 → B[0,H,K0]` | `B[1,H,K0] → P0`，load ×2 | **Compute 0** | Packet 0：`C[0,0]`清零＋K0，完成<br>Packet 1：`C[0,1]`清零＋K0，完成 | **无** | — |
| **Memory 1** | `S[0,2]`、`S[0,3]`，各load ×2 | **无** | `B[0,H,K0]` | **`vmcnt(6)`** | `P1 → B[1,L,K0]` | `B[2,L,K0] → P1`，load ×2 | **Compute 1** | Packet 0：`C[0,2]`清零＋K0，完成<br>Packet 1：`C[0,3]`清零＋K0，完成 | `C_bf16[0,0]`与packet 0交织<br>`C_bf16[0,1]`与packet 1交织 | `S[0,0]`、`S[0,1]` |

N0结束：BF16 C0/C1，FP32 C2/C3及S2/S3。

#### N=1

| **Memory Stage** | **Scale load ×4** | **CShuffle → C_bf16 store ×4** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]`、`S[1,1]` | `C_bf16[0,0/1].r0`与`.r1`，各store ×2 | `B[1,L,K0]` | **`vmcnt(10)`** | `P0 → B[1,H,K0]` | `B[2,H,K0] → P0`，load ×2 | **Compute 0** | Packet 0：`C[1,0]`清零＋K0，完成<br>Packet 1：`C[1,1]`清零＋K0，完成 | **`C_bf16[0,2]`**与packet 0交织<br>**`C_bf16[0,3]`**与packet 1交织 | **`S[0,2]`、`S[0,3]`** |
| **Memory 1** | `S[1,2]`、`S[1,3]` | `C_bf16[0,2/3].r0`与`.r1`，各store ×2 | `B[1,H,K0]` | **`vmcnt(14)`** | `P1 → B[2,L,K0]` | `B[3,L,K0] → P1`，load ×2 | **Compute 1** | Packet 0：`C[1,2]`清零＋K0，完成<br>Packet 1：`C[1,3]`清零＋K0，完成 | `C_bf16[1,0]`与packet 0交织<br>`C_bf16[1,1]`与packet 1交织 | `S[1,0]`、`S[1,1]` |

首Memory0的10保护旧S2/S3，N0 Memory1没有4store；Memory1已有完整14。

#### N≥2，且n+2<T：完整稳态

| **Memory Stage** | **Scale load ×4** | **CShuffle → C_bf16 store ×4** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]`、`S[n,1]` | `C_bf16[n−1,0/1].r0`与`.r1`，各store ×2 | `B[n,L,K0]` | **`vmcnt(14)`** | `P0 → B[n,H,K0]` | `B[n+1,H,K0] → P0`，load ×2 | **Compute 0** | Packet 0：`C[n,0]`清零＋K0，完成<br>Packet 1：`C[n,1]`清零＋K0，完成 | **`C_bf16[n−1,2]`**与packet 0交织<br>**`C_bf16[n−1,3]`**与packet 1交织 | **`S[n−1,2]`、`S[n−1,3]`** |
| **Memory 1** | `S[n,2]`、`S[n,3]` | `C_bf16[n−1,2/3].r0`与`.r1`，各store ×2 | `B[n,H,K0]` | **`vmcnt(14)`** | `P1 → B[n+1,L,K0]` | `B[n+2,L,K0] → P1`，load ×2 | **Compute 1** | Packet 0：`C[n,2]`清零＋K0，完成<br>Packet 1：`C[n,3]`清零＋K0，完成 | `C_bf16[n,0]`与packet 0交织<br>`C_bf16[n,1]`与packet 1交织 | `S[n,0]`、`S[n,1]` |

#### 回写／地址／尾部

1. r0/r1两小步骤各CShuffle写/读→对应B packet读→**lgkmcnt(6)**→store×2，再VM wait/提交/预取。每B packet6条128-bit LDS读，不套4。
2. 上N Compute1尾prepare_b_addresses预备下N**5个完整lane地址**：当前读、H写16B/8B、下L写16B/8B，跨回边携带；两组在首Compute0后额外barrier错相。
3. 始终跨N两BF16 C0/C1+两FP32 C2/C3及scale，下N Compute0 pack旧C2/C3，Memory1写完才清零高半累加器。
4. 最后两N独立裁剪：倒数第二N不发不存在的n+2/L；最后N不提交/预取越界。足够长PTPC末N**12/12**，后vmcnt(0)补C2/C3、L/H与r0/r1全部回写。
5. T=1无P1下N种子；T<4无动态回边，6/6、10/14、14/14表不机械套单N。源码：[K192原语/流程/账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k192.py)。

<a id="pipeline-k320"></a>

### 2.4 K320：Prologue、N=0、N=1、稳态与尾部

每N四拍**L/K128、L/K192、H/K128、H/K192**，Compute为32/48/32/48 MFMA/wave。K0每lane16B/load×1，K1每lane24B/load×2。槽0 N128×128=16KiB，槽1 N128×192=24KiB，CShuffle16KiB，共56KiB；L/H间距8192/12288B，槽不随N交换。P0=K192、P1=K128，不是LDS编号。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert及行scale准备 | 元数据／LDS同步 | 尚未计算C |
| 首B与A | 首B只发`B[0,L,K0]`，load ×1；A按128＋192 gather到VGPR | 不重复预填H/K0，不padding归约维 | 尚未计算C |
| 首B落LDS | 使用启动首B片段 | `vmcnt(4)` → LDS槽0的L/K0 | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`，load ×2；`B[0,H,K0] → P1`，load ×1 | `vmcnt(1)` → barrier | FP32累加区清零；无旧BF16 |

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,H,K1] → P0`，load ×2 | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | **`vmcnt(6)`** | `P1 → B[0,H,K0]` | `B[1,L,K0] → P1`，load ×1 | **Compute 1** | Packet 0：`C[0,0]`累加K1，完成<br>Packet 1：`C[0,1]`累加K1，完成 | **无** | — |
| **Memory 2** | `S[0,2]` | **无** | `B[0,H,K0]` | **`vmcnt(3)`** | `P0 → B[0,H,K1]` | `B[1,L,K1] → P0`，load ×2 | **Compute 2** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,0]`与packet 0交织<br>`C_bf16[0,1]`与packet 1交织 | `S[0,0]`、`S[0,1]` |
| **Memory 3** | `S[0,3]` | **无** | `B[0,H,K1]` | **`vmcnt(6)`** | `P1 → B[1,L,K0]` | `B[1,H,K0] → P1`，load ×1 | **Compute 3** | Packet 0：`C[0,2]`累加K1，完成<br>Packet 1：`C[0,3]`累加K1，完成 | **无** | — |

N0结束BF16 C0/C1、FP32 C2/C3及S2/S3。首M0实际3非BK128的2，首M1提交H/K128后**交接barrier前lgkmcnt(0)**。

#### N=1

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]` | `C_bf16[0,0/1].r0` | `B[1,L,K0]` | **`vmcnt(5)`** | `P0 → B[1,L,K1]` | `B[1,H,K1] → P0`，load ×2 | **Compute 0** | Packet 0：`C[1,0]`清零＋K0贡献<br>Packet 1：`C[1,1]`清零＋K0贡献 | **`C_bf16[0,2]`**与packet 0交织<br>**`C_bf16[0,3]`**与packet 1交织 | **`S[0,2]`、`S[0,3]`** |
| **Memory 1** | `S[1,1]` | `C_bf16[0,0/1].r1` | `B[1,L,K1]` | **`vmcnt(10)`** | `P1 → B[1,H,K0]` | `B[2,L,K0] → P1`，load ×1 | **Compute 1** | Packet 0：`C[1,0]`累加K1，完成<br>Packet 1：`C[1,1]`累加K1，完成 | **无** | — |
| **Memory 2** | `S[1,2]` | `C_bf16[0,2/3].r0` | `B[1,H,K0]` | **`vmcnt(7)`** | `P0 → B[1,H,K1]` | `B[2,L,K1] → P0`，load ×2 | **Compute 2** | Packet 0：`C[1,2]`清零＋K0贡献<br>Packet 1：`C[1,3]`清零＋K0贡献 | `C_bf16[1,0]`与packet 0交织<br>`C_bf16[1,1]`与packet 1交织 | `S[1,0]`、`S[1,1]` |
| **Memory 3** | `S[1,3]` | `C_bf16[0,2/3].r1` | `B[1,H,K1]` | **`vmcnt(10)`** | `P1 → B[2,L,K0]` | `B[2,H,K0] → P1`，load ×1 | **Compute 3** | Packet 0：`C[1,2]`累加K1，完成<br>Packet 1：`C[1,3]`累加K1，完成 | **无** | — |

M0保护上一N S2/S3，首N末拍无store所以5而深稳态7；BK192双VMEM改变年龄，不能写7/7/7/7。

#### N≥2：完整稳态

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]` | `C_bf16[n−1,0/1].r0` | `B[n,L,K0]` | **`vmcnt(7)`** | `P0 → B[n,L,K1]` | `B[n,H,K1] → P0`，load ×2 | **Compute 0** | Packet 0：`C[n,0]`清零＋K0贡献<br>Packet 1：`C[n,1]`清零＋K0贡献 | **`C_bf16[n−1,2]`**与packet 0交织<br>**`C_bf16[n−1,3]`**与packet 1交织 | **`S[n−1,2]`、`S[n−1,3]`** |
| **Memory 1** | `S[n,1]` | `C_bf16[n−1,0/1].r1` | `B[n,L,K1]` | **`vmcnt(10)`** | `P1 → B[n,H,K0]` | `B[n+1,L,K0] → P1`，load ×1 | **Compute 1** | Packet 0：`C[n,0]`累加K1，完成<br>Packet 1：`C[n,1]`累加K1，完成 | **无** | — |
| **Memory 2** | `S[n,2]` | `C_bf16[n−1,2/3].r0` | `B[n,H,K0]` | **`vmcnt(7)`** | `P0 → B[n,H,K1]` | `B[n+1,L,K1] → P0`，load ×2 | **Compute 2** | Packet 0：`C[n,2]`清零＋K0贡献<br>Packet 1：`C[n,3]`清零＋K0贡献 | `C_bf16[n,0]`与packet 0交织<br>`C_bf16[n,1]`与packet 1交织 | `S[n,0]`、`S[n,1]` |
| **Memory 3** | `S[n,3]` | `C_bf16[n−1,2/3].r1` | `B[n,H,K1]` | **`vmcnt(10)`** | `P1 → B[n+1,L,K0]` | `B[n+1,H,K0] → P1`，load ×1 | **Compute 3** | Packet 0：`C[n,2]`累加K1，完成<br>Packet 1：`C[n,3]`累加K1，完成 | **无** | — |

#### 交接与尾部

只有Compute0/2各pack两份，与packet0/1交织，模板仍是BK128 16-MFMA，不移到BK192拍。正常Memory同K256，B packet0后store前**K0 lgkmcnt4、K1 lgkmcnt6**；首M0和首M1交接前另lgkmcnt0，不能依赖晚Compute尾wait。足够长末N前3拍**7/10/6**，末M3无B提交/pack消费者，账本63且不发VM wait；后vmcnt0补C2/C3全部回写。共享循环独立N0/N1/末N，短N裁去越界。源码：[K320](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k320.py)。

<a id="pipeline-k384"></a>

### 2.5 K384：Prologue、N=0、N=1、稳态与尾部

3个K128，每N六拍Q[6n..6n+5]=L/K0、L/K1、L/K2、H/K0、H/K1、H/K2。每Compute两个16-MFMA packet，共32MFMA/wave，B预取每份load×1。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert、routing与activation scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | `B[0,L,K0] → P0`；gather K0/K1/K2的A到VGPR | A跨N复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的P0 | `vmcnt(4)` → LDS槽0的`B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；**`B[0,L,K2] → P1`** | `vmcnt(1)` → barrier | FP32累加区清零；无旧BF16 |

首M0读Q0、提交Q1、发Q3=H/K0；**Q2是L/K2**。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(2)`** | `P0 → B[0,L,K1]` | `B[0,H,K0] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | `vmcnt(5)` | `P1 → B[0,L,K2]` | `B[0,H,K1] → P1` | **Compute 1** | Packet 0：`C[0,0]`累加K1<br>Packet 1：`C[0,1]`累加K1 | **无** | — |
| **Memory 2** | `S[0,2]` | **无** | `B[0,L,K2]` | `vmcnt(5)` | `P0 → B[0,H,K0]` | `B[0,H,K2] → P0` | **Compute 2** | Packet 0：`C[0,0]`累加K2，完成<br>Packet 1：`C[0,1]`累加K2，完成 | `C_bf16[0,0]`<br>与packet 1交织 | `S[0,0]` |
| **Memory 3** | `S[0,3]` | **无** | `B[0,H,K0]` | `vmcnt(5)` | `P1 → B[0,H,K1]` | `B[1,L,K0] → P1` | **Compute 3** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,1]`<br>与packet 0交织 | `S[0,1]` |
| **Memory 4** | **无** | **无** | `B[0,H,K1]` | `vmcnt(3)` | `P0 → B[0,H,K2]` | `B[1,L,K1] → P0` | **Compute 4** | Packet 0：`C[0,2]`累加K1<br>Packet 1：`C[0,3]`累加K1 | **无** | — |
| **Memory 5** | **无** | **无** | `B[0,H,K2]` | `vmcnt(1)` | `P1 → B[1,L,K0]` | `B[1,L,K2] → P1` | **Compute 5** | Packet 0：`C[0,2]`累加K2，完成<br>Packet 1：`C[0,3]`累加K2，完成 | `C_bf16[0,2]`<br>与packet 1交织 | `S[0,2]` |

N0结束BF16 C0/C1/C2，FP32 C3及S3。

#### N=1

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]` | `C_bf16[0,0/1].r0` | `B[1,L,K0]` | **`vmcnt(5)`** | `P0 → B[1,L,K1]` | `B[1,H,K0] → P0` | **Compute 0** | Packet 0：`C[1,0]`清零＋K0贡献<br>Packet 1：`C[1,1]`清零＋K0贡献 | **`C_bf16[0,3]`**<br>与packet 0交织 | **`S[0,3]`** |
| **Memory 1** | `S[1,1]` | `C_bf16[0,0/1].r1` | `B[1,L,K1]` | `vmcnt(9)` | `P1 → B[1,L,K2]` | `B[1,H,K1] → P1` | **Compute 1** | Packet 0：`C[1,0]`累加K1<br>Packet 1：`C[1,1]`累加K1 | **无** | — |
| **Memory 2** | `S[1,2]` | `C_bf16[0,2/3].r0` | `B[1,L,K2]` | `vmcnt(9)` | `P0 → B[1,H,K0]` | `B[1,H,K2] → P0` | **Compute 2** | Packet 0：`C[1,0]`累加K2，完成<br>Packet 1：`C[1,1]`累加K2，完成 | `C_bf16[1,0]`<br>与packet 1交织 | `S[1,0]` |
| **Memory 3** | `S[1,3]` | `C_bf16[0,2/3].r1` | `B[1,H,K0]` | `vmcnt(9)` | `P1 → B[1,H,K1]` | `B[2,L,K0] → P1` | **Compute 3** | Packet 0：`C[1,2]`清零＋K0贡献<br>Packet 1：`C[1,3]`清零＋K0贡献 | `C_bf16[1,1]`<br>与packet 0交织 | `S[1,1]` |
| **Memory 4** | **无** | **无** | `B[1,H,K1]` | `vmcnt(5)` | `P0 → B[1,H,K2]` | `B[2,L,K1] → P0` | **Compute 4** | Packet 0：`C[1,2]`累加K1<br>Packet 1：`C[1,3]`累加K1 | **无** | — |
| **Memory 5** | **无** | **无** | `B[1,H,K2]` | `vmcnt(1)` | `P1 → B[2,L,K0]` | `B[2,L,K2] → P1` | **Compute 5** | Packet 0：`C[1,2]`累加K2，完成<br>Packet 1：`C[1,3]`累加K2，完成 | `C_bf16[1,2]`<br>与packet 1交织 | `S[1,2]` |

N1等待已同深稳态，不照搬K256首拍5/深7。

#### N≥2：完整稳态

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]` | `C_bf16[n−1,0/1].r0` | `B[n,L,K0]` | **`vmcnt(5)`** | `P0 → B[n,L,K1]` | `B[n,H,K0] → P0` | **Compute 0** | Packet 0：`C[n,0]`清零＋K0贡献<br>Packet 1：`C[n,1]`清零＋K0贡献 | **`C_bf16[n−1,3]`**<br>与packet 0交织 | **`S[n−1,3]`** |
| **Memory 1** | `S[n,1]` | `C_bf16[n−1,0/1].r1` | `B[n,L,K1]` | `vmcnt(9)` | `P1 → B[n,L,K2]` | `B[n,H,K1] → P1` | **Compute 1** | Packet 0：`C[n,0]`累加K1<br>Packet 1：`C[n,1]`累加K1 | **无** | — |
| **Memory 2** | `S[n,2]` | `C_bf16[n−1,2/3].r0` | `B[n,L,K2]` | `vmcnt(9)` | `P0 → B[n,H,K0]` | `B[n,H,K2] → P0` | **Compute 2** | Packet 0：`C[n,0]`累加K2，完成<br>Packet 1：`C[n,1]`累加K2，完成 | `C_bf16[n,0]`<br>与packet 1交织 | `S[n,0]` |
| **Memory 3** | `S[n,3]` | `C_bf16[n−1,2/3].r1` | `B[n,H,K0]` | `vmcnt(9)` | `P1 → B[n,H,K1]` | `B[n+1,L,K0] → P1` | **Compute 3** | Packet 0：`C[n,2]`清零＋K0贡献<br>Packet 1：`C[n,3]`清零＋K0贡献 | `C_bf16[n,1]`<br>与packet 0交织 | `S[n,1]` |
| **Memory 4** | **无** | **无** | `B[n,H,K1]` | `vmcnt(5)` | `P0 → B[n,H,K2]` | `B[n+1,L,K1] → P0` | **Compute 4** | Packet 0：`C[n,2]`累加K1<br>Packet 1：`C[n,3]`累加K1 | **无** | — |
| **Memory 5** | **无** | **无** | `B[n,H,K2]` | `vmcnt(1)` | `P1 → B[n+1,L,K0]` | `B[n+1,L,K2] → P1` | **Compute 5** | Packet 0：`C[n,2]`累加K2，完成<br>Packet 1：`C[n,3]`累加K2，完成 | `C_bf16[n,2]`<br>与packet 1交织 | `S[n,2]` |

#### 奇数KS、scale与尾部

LDS槽`(3n+k_stage)&1`随N翻转，P仍按`s&1`选择；L/K2和H/K0可在同槽不同N64区域。上一N的Compute 5准备2读+2写共**4个完整lane地址**，包含实际partition，跨回边携带。Scale只在Memory 0–3加载；Compute 0/2/3/5分别pack旧C3、当前C0/C1/C2。S0用于Compute 2，S2用于Compute 5，S3留到下一N，不能逐拍替换。Memory 4/5无scale/store；回写在B packet0读后用`lgkmcnt(4)`，VM等待同时保护B提交和本拍scale。足够长PTPC末N为**5/9/9/9/4/7**，末拍无下一B仍有pack；后`vmcnt(0)`只补C3并回写末N，短N可合并冗余等待，不能当统一ISA模板。

<a id="pipeline-k512"></a>

### 2.6 K512：Prologue、N=0、N=1、稳态与尾部

4个K128，每N八拍，先L/K0..K3再H/K0..K3；每Compute两个16-MFMA packet，非一次完整K512，每N256MFMA/wave，B每份load×1。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert、routing与activation scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | `B[0,L,K0] → P0`；gather K0/K1/K2/K3的A到VGPR | A跨N复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的P0 | `vmcnt(4)` → LDS槽0的`B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,L,K2] → P1` | `vmcnt(1)` → barrier | FP32累加区清零；无旧BF16 |

首M0预取Q3=L/K3，H/K0到M1才预取。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(2)`** | `P0 → B[0,L,K1]` | `B[0,L,K3] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | `vmcnt(5)` | `P1 → B[0,L,K2]` | `B[0,H,K0] → P1` | **Compute 1** | Packet 0：`C[0,0]`累加K1<br>Packet 1：`C[0,1]`累加K1 | **无** | — |
| **Memory 2** | `S[0,2]` | **无** | `B[0,L,K2]` | `vmcnt(5)` | `P0 → B[0,L,K3]` | `B[0,H,K1] → P0` | **Compute 2** | Packet 0：`C[0,0]`累加K2<br>Packet 1：`C[0,1]`累加K2 | **无** | — |
| **Memory 3** | `S[0,3]` | **无** | `B[0,L,K3]` | `vmcnt(5)` | `P1 → B[0,H,K0]` | `B[0,H,K2] → P1` | **Compute 3** | Packet 0：`C[0,0]`累加K3，完成<br>Packet 1：`C[0,1]`累加K3，完成 | `C_bf16[0,0]`<br>与packet 1交织 | `S[0,0]` |
| **Memory 4** | **无** | **无** | `B[0,H,K0]` | `vmcnt(3)` | `P0 → B[0,H,K1]` | `B[0,H,K3] → P0` | **Compute 4** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,1]`<br>与packet 0交织 | `S[0,1]` |
| **Memory 5** | **无** | **无** | `B[0,H,K1]` | `vmcnt(1)` | `P1 → B[0,H,K2]` | `B[1,L,K0] → P1` | **Compute 5** | Packet 0：`C[0,2]`累加K1<br>Packet 1：`C[0,3]`累加K1 | **无** | — |
| **Memory 6** | **无** | **无** | `B[0,H,K2]` | `vmcnt(1)` | `P0 → B[0,H,K3]` | `B[1,L,K1] → P0` | **Compute 6** | Packet 0：`C[0,2]`累加K2<br>Packet 1：`C[0,3]`累加K2 | **无** | — |
| **Memory 7** | **无** | **无** | `B[0,H,K3]` | `vmcnt(1)` | `P1 → B[1,L,K0]` | `B[1,L,K2] → P1` | **Compute 7** | Packet 0：`C[0,2]`累加K3，完成<br>Packet 1：`C[0,3]`累加K3，完成 | `C_bf16[0,2]`<br>与packet 1交织 | `S[0,2]` |

N0结束BF16 C0/C1/C2，FP32 C3及S3。

#### N=1

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]` | `C_bf16[0,0/1].r0` | `B[1,L,K0]` | **`vmcnt(5)`** | `P0 → B[1,L,K1]` | `B[1,L,K3] → P0` | **Compute 0** | Packet 0：`C[1,0]`清零＋K0贡献<br>Packet 1：`C[1,1]`清零＋K0贡献 | **`C_bf16[0,3]`**<br>与packet 0交织 | **`S[0,3]`** |
| **Memory 1** | `S[1,1]` | `C_bf16[0,0/1].r1` | `B[1,L,K1]` | `vmcnt(9)` | `P1 → B[1,L,K2]` | `B[1,H,K0] → P1` | **Compute 1** | Packet 0：`C[1,0]`累加K1<br>Packet 1：`C[1,1]`累加K1 | **无** | — |
| **Memory 2** | `S[1,2]` | `C_bf16[0,2/3].r0` | `B[1,L,K2]` | `vmcnt(9)` | `P0 → B[1,L,K3]` | `B[1,H,K1] → P0` | **Compute 2** | Packet 0：`C[1,0]`累加K2<br>Packet 1：`C[1,1]`累加K2 | **无** | — |
| **Memory 3** | `S[1,3]` | `C_bf16[0,2/3].r1` | `B[1,L,K3]` | `vmcnt(9)` | `P1 → B[1,H,K0]` | `B[1,H,K2] → P1` | **Compute 3** | Packet 0：`C[1,0]`累加K3，完成<br>Packet 1：`C[1,1]`累加K3，完成 | `C_bf16[1,0]`<br>与packet 1交织 | `S[1,0]` |
| **Memory 4** | **无** | **无** | `B[1,H,K0]` | `vmcnt(5)` | `P0 → B[1,H,K1]` | `B[1,H,K3] → P0` | **Compute 4** | Packet 0：`C[1,2]`清零＋K0贡献<br>Packet 1：`C[1,3]`清零＋K0贡献 | `C_bf16[1,1]`<br>与packet 0交织 | `S[1,1]` |
| **Memory 5** | **无** | **无** | `B[1,H,K1]` | `vmcnt(1)` | `P1 → B[1,H,K2]` | `B[2,L,K0] → P1` | **Compute 5** | Packet 0：`C[1,2]`累加K1<br>Packet 1：`C[1,3]`累加K1 | **无** | — |
| **Memory 6** | **无** | **无** | `B[1,H,K2]` | `vmcnt(1)` | `P0 → B[1,H,K3]` | `B[2,L,K1] → P0` | **Compute 6** | Packet 0：`C[1,2]`累加K2<br>Packet 1：`C[1,3]`累加K2 | **无** | — |
| **Memory 7** | **无** | **无** | `B[1,H,K3]` | `vmcnt(1)` | `P1 → B[2,L,K0]` | `B[2,L,K2] → P1` | **Compute 7** | Packet 0：`C[1,2]`累加K3，完成<br>Packet 1：`C[1,3]`累加K3，完成 | `C_bf16[1,2]`<br>与packet 1交织 | `S[1,2]` |

#### N≥2：完整稳态

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]` | `C_bf16[n−1,0/1].r0` | `B[n,L,K0]` | **`vmcnt(5)`** | `P0 → B[n,L,K1]` | `B[n,L,K3] → P0` | **Compute 0** | Packet 0：`C[n,0]`清零＋K0贡献<br>Packet 1：`C[n,1]`清零＋K0贡献 | **`C_bf16[n−1,3]`**<br>与packet 0交织 | **`S[n−1,3]`** |
| **Memory 1** | `S[n,1]` | `C_bf16[n−1,0/1].r1` | `B[n,L,K1]` | `vmcnt(9)` | `P1 → B[n,L,K2]` | `B[n,H,K0] → P1` | **Compute 1** | Packet 0：`C[n,0]`累加K1<br>Packet 1：`C[n,1]`累加K1 | **无** | — |
| **Memory 2** | `S[n,2]` | `C_bf16[n−1,2/3].r0` | `B[n,L,K2]` | `vmcnt(9)` | `P0 → B[n,L,K3]` | `B[n,H,K1] → P0` | **Compute 2** | Packet 0：`C[n,0]`累加K2<br>Packet 1：`C[n,1]`累加K2 | **无** | — |
| **Memory 3** | `S[n,3]` | `C_bf16[n−1,2/3].r1` | `B[n,L,K3]` | `vmcnt(9)` | `P1 → B[n,H,K0]` | `B[n,H,K2] → P1` | **Compute 3** | Packet 0：`C[n,0]`累加K3，完成<br>Packet 1：`C[n,1]`累加K3，完成 | `C_bf16[n,0]`<br>与packet 1交织 | `S[n,0]` |
| **Memory 4** | **无** | **无** | `B[n,H,K0]` | `vmcnt(5)` | `P0 → B[n,H,K1]` | `B[n,H,K3] → P0` | **Compute 4** | Packet 0：`C[n,2]`清零＋K0贡献<br>Packet 1：`C[n,3]`清零＋K0贡献 | `C_bf16[n,1]`<br>与packet 0交织 | `S[n,1]` |
| **Memory 5** | **无** | **无** | `B[n,H,K1]` | `vmcnt(1)` | `P1 → B[n,H,K2]` | `B[n+1,L,K0] → P1` | **Compute 5** | Packet 0：`C[n,2]`累加K1<br>Packet 1：`C[n,3]`累加K1 | **无** | — |
| **Memory 6** | **无** | **无** | `B[n,H,K2]` | `vmcnt(1)` | `P0 → B[n,H,K3]` | `B[n+1,L,K1] → P0` | **Compute 6** | Packet 0：`C[n,2]`累加K2<br>Packet 1：`C[n,3]`累加K2 | **无** | — |
| **Memory 7** | **无** | **无** | `B[n,H,K3]` | `vmcnt(1)` | `P1 → B[n+1,L,K0]` | `B[n+1,L,K2] → P1` | **Compute 7** | Packet 0：`C[n,2]`累加K3，完成<br>Packet 1：`C[n,3]`累加K3，完成 | `C_bf16[n,2]`<br>与packet 1交织 | `S[n,2]` |

#### 八拍与尾部

LDS槽`(4n+k_stage)&1=k_stage&1`不随N翻转；两个16KiB B槽+CShuffle16KiB=48KiB，P每拍轮换，不是四个B常驻。N0 wait为**2/5/5/5/3/1/1/1**，N1/steady为**5/9/9/9/5/1/1/1**；只有前4个Memory有scale/store，不恢复legacy +4。Compute 0 pack旧C3，Compute 3/4/7分别pack当前C0/C1/C2；C0到K3归约完成后才在独立packet1间隙pack，没有减少贡献。足够长PTPC末N为**5/9/9/9/5/1/0/9**，末拍仍保护C2的scale消费者；最后三个Memory无越界Q[q+3]，后`vmcnt(0)`补C3并完整回写。

<a id="pipeline-k640"></a>

### 2.7 K640：Prologue、N=0、N=1、稳态与尾部

5个K128、每N十拍，先L/K0..K4再H/K0..K4。Compute32MFMA/wave，每N320；**stage4仍L、stage5才H**，scale/旧回写已在前4拍安排。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert、routing与activation scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | `B[0,L,K0] → P0`；gather K0..K4的A到VGPR | 五段A跨N复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的P0 | `vmcnt(4)` → LDS槽0的`B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,L,K2] → P1` | `vmcnt(1)` → barrier | FP32累加区清零；无旧BF16 |

M0发Q3=L/K3，M1发Q4=L/K4，M2才H/K0。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(2)`** | `P0 → B[0,L,K1]` | `B[0,L,K3] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | `vmcnt(5)` | `P1 → B[0,L,K2]` | `B[0,L,K4] → P1` | **Compute 1** | Packet 0：`C[0,0]`累加K1<br>Packet 1：`C[0,1]`累加K1 | **无** | — |
| **Memory 2** | `S[0,2]` | **无** | `B[0,L,K2]` | `vmcnt(5)` | `P0 → B[0,L,K3]` | `B[0,H,K0] → P0` | **Compute 2** | Packet 0：`C[0,0]`累加K2<br>Packet 1：`C[0,1]`累加K2 | **无** | — |
| **Memory 3** | `S[0,3]` | **无** | `B[0,L,K3]` | `vmcnt(5)` | `P1 → B[0,L,K4]` | `B[0,H,K1] → P1` | **Compute 3** | Packet 0：`C[0,0]`累加K3<br>Packet 1：`C[0,1]`累加K3 | **无** | — |
| **Memory 4** | **无** | **无** | `B[0,L,K4]` | `vmcnt(3)` | `P0 → B[0,H,K0]` | `B[0,H,K2] → P0` | **Compute 4** | Packet 0：`C[0,0]`累加K4，完成<br>Packet 1：`C[0,1]`累加K4，完成 | `C_bf16[0,0]`<br>与packet 1交织 | `S[0,0]` |
| **Memory 5** | **无** | **无** | `B[0,H,K0]` | `vmcnt(1)` | `P1 → B[0,H,K1]` | `B[0,H,K3] → P1` | **Compute 5** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,1]`<br>与packet 0交织 | `S[0,1]` |
| **Memory 6** | **无** | **无** | `B[0,H,K1]` | `vmcnt(1)` | `P0 → B[0,H,K2]` | `B[0,H,K4] → P0` | **Compute 6** | Packet 0：`C[0,2]`累加K1<br>Packet 1：`C[0,3]`累加K1 | **无** | — |
| **Memory 7** | **无** | **无** | `B[0,H,K2]` | `vmcnt(1)` | `P1 → B[0,H,K3]` | `B[1,L,K0] → P1` | **Compute 7** | Packet 0：`C[0,2]`累加K2<br>Packet 1：`C[0,3]`累加K2 | **无** | — |
| **Memory 8** | **无** | **无** | `B[0,H,K3]` | `vmcnt(1)` | `P0 → B[0,H,K4]` | `B[1,L,K1] → P0` | **Compute 8** | Packet 0：`C[0,2]`累加K3<br>Packet 1：`C[0,3]`累加K3 | **无** | — |
| **Memory 9** | **无** | **无** | `B[0,H,K4]` | `vmcnt(1)` | `P1 → B[1,L,K0]` | `B[1,L,K2] → P1` | **Compute 9** | Packet 0：`C[0,2]`累加K4，完成<br>Packet 1：`C[0,3]`累加K4，完成 | `C_bf16[0,2]`<br>与packet 1交织 | `S[0,2]` |

N0结束BF16 C0/C1/C2、FP32 C3与S3。

#### N=1

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[1,0]` | `C_bf16[0,0/1].r0` | `B[1,L,K0]` | **`vmcnt(5)`** | `P0 → B[1,L,K1]` | `B[1,L,K3] → P0` | **Compute 0** | Packet 0：`C[1,0]`清零＋K0贡献<br>Packet 1：`C[1,1]`清零＋K0贡献 | **`C_bf16[0,3]`**<br>与packet 0交织 | **`S[0,3]`** |
| **Memory 1** | `S[1,1]` | `C_bf16[0,0/1].r1` | `B[1,L,K1]` | `vmcnt(9)` | `P1 → B[1,L,K2]` | `B[1,L,K4] → P1` | **Compute 1** | Packet 0：`C[1,0]`累加K1<br>Packet 1：`C[1,1]`累加K1 | **无** | — |
| **Memory 2** | `S[1,2]` | `C_bf16[0,2/3].r0` | `B[1,L,K2]` | `vmcnt(9)` | `P0 → B[1,L,K3]` | `B[1,H,K0] → P0` | **Compute 2** | Packet 0：`C[1,0]`累加K2<br>Packet 1：`C[1,1]`累加K2 | **无** | — |
| **Memory 3** | `S[1,3]` | `C_bf16[0,2/3].r1` | `B[1,L,K3]` | `vmcnt(9)` | `P1 → B[1,L,K4]` | `B[1,H,K1] → P1` | **Compute 3** | Packet 0：`C[1,0]`累加K3<br>Packet 1：`C[1,1]`累加K3 | **无** | — |
| **Memory 4** | **无** | **无** | `B[1,L,K4]` | `vmcnt(5)` | `P0 → B[1,H,K0]` | `B[1,H,K2] → P0` | **Compute 4** | Packet 0：`C[1,0]`累加K4，完成<br>Packet 1：`C[1,1]`累加K4，完成 | `C_bf16[1,0]`<br>与packet 1交织 | `S[1,0]` |
| **Memory 5** | **无** | **无** | `B[1,H,K0]` | `vmcnt(1)` | `P1 → B[1,H,K1]` | `B[1,H,K3] → P1` | **Compute 5** | Packet 0：`C[1,2]`清零＋K0贡献<br>Packet 1：`C[1,3]`清零＋K0贡献 | `C_bf16[1,1]`<br>与packet 0交织 | `S[1,1]` |
| **Memory 6** | **无** | **无** | `B[1,H,K1]` | `vmcnt(1)` | `P0 → B[1,H,K2]` | `B[1,H,K4] → P0` | **Compute 6** | Packet 0：`C[1,2]`累加K1<br>Packet 1：`C[1,3]`累加K1 | **无** | — |
| **Memory 7** | **无** | **无** | `B[1,H,K2]` | `vmcnt(1)` | `P1 → B[1,H,K3]` | `B[2,L,K0] → P1` | **Compute 7** | Packet 0：`C[1,2]`累加K2<br>Packet 1：`C[1,3]`累加K2 | **无** | — |
| **Memory 8** | **无** | **无** | `B[1,H,K3]` | `vmcnt(1)` | `P0 → B[1,H,K4]` | `B[2,L,K1] → P0` | **Compute 8** | Packet 0：`C[1,2]`累加K3<br>Packet 1：`C[1,3]`累加K3 | **无** | — |
| **Memory 9** | **无** | **无** | `B[1,H,K4]` | `vmcnt(1)` | `P1 → B[2,L,K0]` | `B[2,L,K2] → P1` | **Compute 9** | Packet 0：`C[1,2]`累加K4，完成<br>Packet 1：`C[1,3]`累加K4，完成 | `C_bf16[1,2]`<br>与packet 1交织 | `S[1,2]` |

#### N≥2：完整稳态

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[n,0]` | `C_bf16[n−1,0/1].r0` | `B[n,L,K0]` | **`vmcnt(5)`** | `P0 → B[n,L,K1]` | `B[n,L,K3] → P0` | **Compute 0** | Packet 0：`C[n,0]`清零＋K0贡献<br>Packet 1：`C[n,1]`清零＋K0贡献 | **`C_bf16[n−1,3]`**<br>与packet 0交织 | **`S[n−1,3]`** |
| **Memory 1** | `S[n,1]` | `C_bf16[n−1,0/1].r1` | `B[n,L,K1]` | `vmcnt(9)` | `P1 → B[n,L,K2]` | `B[n,L,K4] → P1` | **Compute 1** | Packet 0：`C[n,0]`累加K1<br>Packet 1：`C[n,1]`累加K1 | **无** | — |
| **Memory 2** | `S[n,2]` | `C_bf16[n−1,2/3].r0` | `B[n,L,K2]` | `vmcnt(9)` | `P0 → B[n,L,K3]` | `B[n,H,K0] → P0` | **Compute 2** | Packet 0：`C[n,0]`累加K2<br>Packet 1：`C[n,1]`累加K2 | **无** | — |
| **Memory 3** | `S[n,3]` | `C_bf16[n−1,2/3].r1` | `B[n,L,K3]` | `vmcnt(9)` | `P1 → B[n,L,K4]` | `B[n,H,K1] → P1` | **Compute 3** | Packet 0：`C[n,0]`累加K3<br>Packet 1：`C[n,1]`累加K3 | **无** | — |
| **Memory 4** | **无** | **无** | `B[n,L,K4]` | `vmcnt(5)` | `P0 → B[n,H,K0]` | `B[n,H,K2] → P0` | **Compute 4** | Packet 0：`C[n,0]`累加K4，完成<br>Packet 1：`C[n,1]`累加K4，完成 | `C_bf16[n,0]`<br>与packet 1交织 | `S[n,0]` |
| **Memory 5** | **无** | **无** | `B[n,H,K0]` | `vmcnt(1)` | `P1 → B[n,H,K1]` | `B[n,H,K3] → P1` | **Compute 5** | Packet 0：`C[n,2]`清零＋K0贡献<br>Packet 1：`C[n,3]`清零＋K0贡献 | `C_bf16[n,1]`<br>与packet 0交织 | `S[n,1]` |
| **Memory 6** | **无** | **无** | `B[n,H,K1]` | `vmcnt(1)` | `P0 → B[n,H,K2]` | `B[n,H,K4] → P0` | **Compute 6** | Packet 0：`C[n,2]`累加K1<br>Packet 1：`C[n,3]`累加K1 | **无** | — |
| **Memory 7** | **无** | **无** | `B[n,H,K2]` | `vmcnt(1)` | `P1 → B[n,H,K3]` | `B[n+1,L,K0] → P1` | **Compute 7** | Packet 0：`C[n,2]`累加K2<br>Packet 1：`C[n,3]`累加K2 | **无** | — |
| **Memory 8** | **无** | **无** | `B[n,H,K3]` | `vmcnt(1)` | `P0 → B[n,H,K4]` | `B[n+1,L,K1] → P0` | **Compute 8** | Packet 0：`C[n,2]`累加K3<br>Packet 1：`C[n,3]`累加K3 | **无** | — |
| **Memory 9** | **无** | **无** | `B[n,H,K4]` | `vmcnt(1)` | `P1 → B[n+1,L,K0]` | `B[n+1,L,K2] → P1` | **Compute 9** | Packet 0：`C[n,2]`累加K4，完成<br>Packet 1：`C[n,3]`累加K4，完成 | `C_bf16[n,2]`<br>与packet 1交织 | `S[n,2]` |

#### 十拍与尾部

LDS槽`(5n+k_stage)&1`随N翻转，上一N的Compute 9准备4个读/写lane地址跨回边；L/K4与H/K0可在同槽不同半区，不按`s&1`猜LDS。Scale/旧store仅在Memory 0–3；Compute 0 pack旧C3，Compute 4/5/9分别pack当前C0/C1/C2；**Compute 4没有新scale却使用保留的S0**。N0 wait为**2/5/5/5/3/1/1/1/1/1**，N1/steady为**5/9/9/9/5/1/1/1/1/1**；后六拍仍有B/MFMA。足够长PTPC末N为**5/9/9/9/5/1/1/1/0/11**，末拍C2仍有scale消费者，后`vmcnt(0)`补C3完整回写；11不推广到稳态或其它K。

K384/512/640共用[主原语](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py)、[流程](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py)、[事件账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_schedule.py)。

<a id="pipeline-boundaries"></a>

### 2.8 非PTPC、短N及compact边界

非PTPC权重（per_tensor/per_tensor或per_tensor/ptpc）**pack事件、B FIFO、C回写结构不变**，只是weight scalar在prologue融合行scale，不发逐N S[n,j] VMEM。每份pack由40降24 VALU，不少MFMA。

| K分支 | per-tensor权重：N=0实际逻辑wait | N=1过渡wait | 深稳态wait | 最后N账本（足够长循环） |
|---|---|---|---|---|
| K192 | `2/2` | `6/10` | `10/10` | `8/63` |
| K256 | `0/1/1/1` | `3/5/5/5` | `5/5/5/5` | `5/5/4/63` |
| K320 | `1/2/1/2` | `3/6/5/6` | `5/6/5/6` | `5/6/4/63` |
| K384 | `0/1/1/1/1/1` | `3/5/5/5/3/1` | `3/5/5/5/3/1` | `3/5/5/5/2/63` |
| K512 | `0/1/1/1/1/1/1/1` | `3/5/5/5/3/1/1/1` | `3/5/5/5/3/1/1/1` | `3/5/5/5/3/1/0/63` |
| K640 | `0/1/1/1/1/1/1/1/1/1` | `3/5/5/5/3/1/1/1/1/1` | `3/5/5/5/3/1/1/1/1/1` | `3/5/5/5/3/1/1/1/0/63` |

63是无当前VM消费者的哨兵，不是等63条、不能省LDS；是否发VM wait看实际guard，最后仍vmcnt0。单/双N按真实T生成，不能截深循环末N阈值。

| 路径 | 动态循环前 | 短N／尾部 | 最终补pack |
|---|---|---|---|
| K192独立 | N0/N1独立，按下N/下下N裁剪 | 最后两N独立；T<4展开；n_loop0全展开，1/2为每回边N数 | C2/C3 |
| K256/320/384/512/640共享 | 独立首M0/C0，再N0和完整N1 | 最后N独立；普通builder T<3不用共享回边；n_loop0实际展开，1/2保持pack/FIFO | K320 C2/C3，其余C3 |

只有q+1<2KS·T才提交、q+3<2KS·T才预取；取消越界B不取消当前pack消费者。展开路径可能已被更早严格wait保护，不能强塞共享尾拍冗余wait，错相/退出barrier仍须平衡。

compact M256 full沿用上述同K流水，从M64 metadata任务表读取`[physical_row_begin, expert_id]`及count，保持physical行编号，不改pack/K宽度。**构表→full→M64 tail**同stream，无host count读回；N%256=0时tail1x4，否则default，K192/K320 full的192归约不改变tail既有BK128。见[组合launcher与任务表](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py)。

<a id="resources"></a>

### 2.9 2026-09-09原pack资源记录

**实际归档产物，非delayed/formal、非活跃槽模型、非本次重构fresh编译。** 源码/ISA/ELF身份见各证据，保留不同descriptor字段原值。

| 项目 | 统计口径 |
|---|---|
| 环境 | gfx942/MI308X、FlyDSL0.3.2，离线拓扑输入4XCC/80CU；不由理论峰值推资源。 |
| ordinary | M256×N128、512 threads/8 waves、n_loop1、store_cache2、padding128B、非task-table，K192/K320真实归约。 |
| PTPC | weight/activation均PTPC，N2048/TopK8；不是块编号N0/N1。 |
| per-tensor | 双per_tensor，N4096/TopK9；与PTPC的N/TopK不同，不仅量化影响资源。 |
| VGPR/SGPR | ELF metadata vgpr_count/sgpr_count；每lane32-bit VGPR槽、wave共享SGPR，非WG合计或stage瞬时live值。 |
| next_free | ISA descriptor字段，与metadata分列；ordinary next_free_sgpr全部96，不是metadata全96。 |
| LDS | group_segment_fixed_size，每WG字节，1KiB=1024B。 |
| 静态ISA | 完整kernel机器指令，含prologue/回边/drain，非源码行或动态计数。 |
| 回边ISA | end−begin+1含回跳，默认一次一个N128，非完整kernel条数。 |
| spill/scratch | ordinary及compact均VGPR/SGPR spill=0、private=0、scratch指令0、metadata AGPR=0。 |

#### ordinary PTPC／N2048／TopK8

| K分支 | VGPR（metadata） | SGPR（metadata） | `next_free_vgpr` | LDS KiB | 静态ISA条数 | 回边ISA条数 | 资源证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| K192 | 214 | 36 | 214 | 64 | 2062 | 365 | [K192 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k192_ptpc/result.json) |
| K256 | 176 | 38 | 176 | 48 | 1989 | 433 | [K256 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k256_ptpc/result.json) |
| K320 | 214 | 37 | 214 | 56 | 2191 | 473 | [K320 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k320_ptpc/result.json) |
| K384 | 202 | 38 | 202 | 48 | 2447 | 545 | [K384 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k384_ptpc/result.json) |
| K512 | 220 | 38 | 220 | 48 | 2866 | 646 | [K512 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k512_ptpc/result.json) |
| K640 | 244 | 38 | 244 | 48 | 3331 | 762 | [K640 PTPC](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k640_ptpc/result.json) |

#### ordinary per-tensor／N4096／TopK9

| K分支 | VGPR（metadata） | SGPR（metadata） | `next_free_vgpr` | LDS KiB | 静态ISA条数 | 回边ISA条数 | 资源证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| K192 | 176 | 34 | 176 | 64 | 1675 | 293 | [K192 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k192_per_tensor/result.json) |
| K256 | **160** | 34 | **169** | 48 | 1657 | 354 | [K256 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k256_per_tensor/result.json) |
| K320 | 192 | 38 | 192 | 56 | 1854 | 396 | [K320 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k320_per_tensor/result.json) |
| K384 | 176 | 34 | 176 | 48 | 2128 | 472 | [K384 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k384_per_tensor/result.json) |
| K512 | 188 | 34 | 188 | 48 | 2546 | 571 | [K512 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k512_per_tensor/result.json) |
| K640 | 212 | 34 | 212 | 48 | 3021 | 688 | [K640 per-tensor](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/initial_k640_per_tensor/result.json) |

K256标量metadata VGPR160/next_free169两者保留，不凭差值猜spill/驻留。mixed的activation gather/行scale仍可不同，不套per-tensor资源。

| K | B LDS | CShuffle LDS | 合计 |
|---|---|---:|---:|
| K192 | 2×128×192=49152B | 16384B | 65536B/64KiB |
| K320 | 16384+24576=40960B | 16384B | 57344B/56KiB |
| K256/K384/K512/K640 | 2×128×128=32768B | 16384B | 49152B/48KiB |

sorted IDs启动复用既有LDS，不另驻buffer；增K不把全部B放LDS，仍两槽，完整A常驻VGPR。跨N C份数之外还有A/B/scale/地址等，不能据此线性推VGPR。12ordinary归档默认steady Memory均无VALU，不是无寄存器live或无wait；[原FIFO等价审计](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/final_audited.json)绑定版本。

#### compact full/tail分列（12个kernel），构表另计

顺序三个launch，不能将资源相加当一WG。共同gfx942、TopK4/E256、padding32B、阈值0.6；mixed=per_tensor weight+PTPC activation，u=n_loop、cache=store_cache，配置不同不是单因素比较。

| K/配置 | u/cache | kernel | metadata VGPR | metadata SGPR | LDS KiB | 静态ISA | 证据 |
|---|---|---|---:|---:|---:|---:|---|
| K192/N768/PTPC | 1/2 | M256 full | 214 | 34 | 64 | 2061 | [K192](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k192_n768_ptpc_u1_after/result.json) |
| K192/N768/PTPC | 1/2 | M64 tail 1x4 | 190 | 31 | 21 | 648 | [K192](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k192_n768_ptpc_u1_after/result.json) |
| K256/N640/mixed | 2/0 | M256 full | 154 | 34 | 48 | 2016 | [K256](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k256_n640_mixed_u2_after/result.json) |
| K256/N640/mixed | 2/0 | M64 tail default | 128 | 36 | 16 | 632 | [K256](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k256_n640_mixed_u2_after/result.json) |
| K320/N768/mixed | 2/0 | M256 full | 188 | 35 | 56 | 2637 | [K320](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k320_n768_mixed_u2_after/result.json) |
| K320/N768/mixed | 2/0 | M64 tail 1x4 | 158 | 31 | 28 | 719 | [K320](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k320_n768_mixed_u2_after/result.json) |
| K384/N640/per-tensor | 0/2 | M256 full | 174 | 34 | 48 | 2554 | [K384](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k384_n640_per_tensor_u0_after/result.json) |
| K384/N640/per-tensor | 0/2 | M64 tail default | 176 | 36 | 24 | 781 | [K384](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k384_n640_per_tensor_u0_after/result.json) |
| K512/N768/per-tensor | 1/2 | M256 full | 188 | 34 | 48 | 2547 | [K512](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k512_n768_per_tensor_u1_after/result.json) |
| K512/N768/per-tensor | 1/2 | M64 tail 1x4 | 204 | 31 | 40 | 852 | [K512](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k512_n768_per_tensor_u1_after/result.json) |
| K640/N768/mixed | 2/2 | M256 full | 216 | 31 | 48 | 4378 | [K640](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k640_n768_mixed_u2_after/result.json) |
| K640/N768/mixed | 2/2 | M64 tail 1x4 | 212 | 31 | 48 | 1077 | [K640](../../../../tests/contrib/moe/results/cleanup_8x1_20260909/compact/k640_n768_mixed_u2_after/result.json) |

六组构表相同：**20VGPR/28SGPR/4KiB LDS/544静态ISA**、256threads，next_free_sgpr22非metadata28。full512threads、tail256threads，full/tail next_free_sgpr96，全零private/spill/scratch。compact VGPR字段：K256 full metadata/next_free=154/169；K320 tail158/169；K512/640 tail204/257、212/257。next_free257不直接等于每lane可分配257或spill，须看完整descriptor/架构。

这些24行ordinary/full/tail+构表是特定源码/编译器/shape/选项事实，不是任意N/mixed/n_loop0/2/compact统一上限，不能只据表推occupancy/MFMA busy/性能；本次无新测量。

<a id="optimizations"></a>

## 三、重要优化

### 3.1 确实现有的机制

| 优化 | 当前机制和限制 |
|---|---|
| 8-wave反相 | 512线程/两4-wave组，scalar条件及barrier一代位移，memory优先级0/compute3；一组VMEM/LDS/CShuffle、另一组MFMA+原pack。16KiB scratch错相复用，不能半组提前退出或宣称每cycle完全重叠。 |
| 两P槽消费FIFO | 启动LDS Q0、P中Q1/Q2；正常读Q[q]、提交Q[q+1]、发Q[q+3]，Q[q+2]在另P，提前跨计算窗。wait含已发store，同时约束B/scale消费者，非统一放宽9。 |
| A驻留/真实K | 每wave两M16完整A只gather一次跨N；行scale提前融合。K192整192、K320 128+192避免补零和小stage，BK128多块仍两B LDS槽，不按K线性推寄存器。 |
| 原pack/scalar FMA/perm | 每N四N32 super-record各含r0/r1，在独立MFMA packet交织pack，不保留两套完整FP32 C。PTPC每份16weight FMA+16行scale/bias FMA+8perm=40VALU，scalar权重先融合后16FMA+8perm=24；用v_fma_f32/v_perm_b32，禁packed-fp32-ops。pack不是store，VMEM/DS/store不混入MFMA train；模板的最多3条独立VALU不是FP8实测时序保证。 |
| CShuffle plane xor/read2 | 128bit写，source-group低位移到相隔2KiB plane，row/pair xor分bank；仅配同一输出行两8B读，让后端read2st64_b64避免跨行v_mov。B packet0先读→partial LDS wait→store→packet1读，BK128用4、192用6。模型payload、ISA、PMC不同，源码不保证所有shape零bank。 |
| Nloop1 | 默认每回边一个N128，剥离N0/N1/尾部，回边只steady；保存B carry、未pack C/scale、BF16和必要地址。0为全展开、2两N；少展开体积不自动等于快。 |
| 地址提前准备 | 均匀N/K/wave偏移走scalar buffer offset，lane和C读指针提前，K192跨N5完整地址、K384/640四地址；12归档ordinary steady Memory无VALU，不意味Compute无地址或任意配置相同。copy offset须区分元素/字节。 |
| compact平衡 | 每expert单连续run，顺序不必排序；保留physical行。原full F、实际CU U、r=F mod U，仅0<5r<3U把全局full后缀r个M256拆4r个M64，r0不拆；counts为full×256、tail×64，容量包含拆分，device guard、无host读回。launch开销可能抵消padding收益，逐shape选。 |
| cache/64-thread sum | builder缓存包含设备/静态参数，task缓存也含CU/阈值。store_cache2为aux（源码SLC），ISA修饰以产物为准不猜命中。sorted_sum每token64线程，先TopK位置、128bit BF16读、FP32按TopK累加转BF16，stride含padding；inverse仅valid前缀并查token/topk边界，避免未写区域。 |

不把旧pure/K128/BK64、Memory-pack/delayed/formal或direct-to-LDS实验当当前实现，早期实验原字节仍归档；收益仅认对应run，不以指令数变少代替wall-time证据。









