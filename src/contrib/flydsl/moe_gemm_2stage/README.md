# FlyDSL MoE 两阶段：性能、测试与 8x1 流水线

本文是MoE当前唯一维护的说明文档，独立说明支持范围、路径选择、测试方法、六K完整流水及重要优化。默认目标为 **MI308X / gfx942、FP8 E4M3FNUZ A/W、BF16输出**，不将8x1结果外推其它dtype或硬件。

**版本边界：**2026-09-11仅保留Down四path：`default`、`1x4_64x256`、`8x1`、`8x1_compact`；`1x8/2x4/4x1`已删除，旧path明确报错、不静默回退。生产集合为 **13源（11个包内模块＋2个依赖）**，8x1为单文件、固定1N循环，首拍前建立错相并由共享/独立tile从step0开始，保留原pack消费FIFO。本文推荐来自离线测试选择，不是生产自动selector。

**测量边界：**[1.3的42点性能](#matrix42)、[1.4的64行落选](#historical-nonselection)、[2.9的30行资源](#resources)均已使用2026-09-11最终代码重新测试，主文件SHA256为`ef53d88c19ebedbbc955c2a7c8642206800992c7cdb2e93266967996bdd8875a`。首q0归入主循环、删除首tile薄封装，B模板/partition移入准备区后才冻结本轮源码；不是对旧测量换日期或换SHA。

[本轮准备清单](../../../../tests/contrib/moe/results/readme_latest_20260911/prepared.json)SHA256：`f6058b2b10b331730f7ff3266a7900decff282db5c72039b4377d797b529b89c`，冻结完整13源、改前README及原报告矩阵。正文以本轮结果替换旧表，不保留重复旧性能表；历史JSON、ISA、ELF、source和失败收据仍不改写、不重标。1.5以原README所用的merged矩阵为比较基线，固定同标签比较，不回退到更早报告。[独立复审收据](../../../../tests/contrib/moe/results/readme_latest_20260911/audited.json)绑定所有新raw、实际产物与旧比较来源。

## 目录

- [一、性能与测试](#performance)：[支持与shape](#scope) · [测试方法](#testing) · [最新42点主表](#matrix42) · [最新64行落选](#historical-nonselection) · [与原报告同标签比较](#migration-confirmation) · [验证边界](#validation-boundaries)
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

**K192整192，K320为128+192，其余BK128。** K192/K320兼容`tile_k=None/128/192`但都转入真实192实现，不是三种算法。8x1的K128、BK64、pure/rolling环境开关、legacy wait及固定`_block_k`实验参数已删；其它Down算法的K128支持不受影响。现行builder固定每回边处理1个N，已删除仅用于测试/调优的`_n_loop/unroll_n`参数及0/2测试维度；小N的必要展开仍保留。`_store_cache`默认2，不是自动selector。源码：[四path分发](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2.py)、[8x1](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py)、[compact](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py)。

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

已删除的1x8/2x4/4x1只保留历史产物，不参与本轮候选或当前dispatcher；其历史支持限制不外推为其它硬件或算法的限制。

<a id="testing"></a>

### 1.2 功能与性能测试方法

按 **CPU契约→fresh offline→GPU random→performance** 推进；CPU不证明GPU数值，离线编译不是运行，性能全1权重检查不替代随机W。

随机Down参考使用实际量化后的A/W转FP32做矩阵乘，再乘对应expert/channel weight scale、activation行scale和routing weight。`rel_l2`为 $\lVert actual-reference\rVert_2/\lVert reference\rVert_2$，专项Down要求小于0.005，并检查有效输出finite、padding及inactive区域的NaN哨兵未被覆盖；graph重放还须污染并重建任务workspace。公共整链另与其参考输出比较，不将Down单算子阈值冒称所有整链的统一契约。

**以下均为最终源码的新验收。** 源、编译器、输入配置、实际ISA/ELF和收据分别保存，不拿旧通过计数替代当前结果。

| 层次 | 入口及检查 |
|---|---|
| CPU契约 | [共享schedule](../../../../tests/contrib/moe/test_all_8x1_schedule.py)、[首拍与准备移位](../../../../tests/contrib/moe/test_startup_8x1.py)、[方向命名](../../../../tests/contrib/moe/test_8x1_direction_names.py)的[169项检查](../../../../tests/contrib/moe/results/startup_8x1_20260911/cpu_views.xml)通过：实际AST、全部K贡献、FP32/BF16生命周期、四scale与输出、短N/地址策略；不是全仓测试计数。 |
| fresh offline | [单配置编译入口](../../../../tests/contrib/moe/check_startup_8x1.py)与[最终资源编排](../../../../tests/contrib/moe/run_readme_latest.py)使用COMPILE_ONLY/gfx942、4XCC/80CU显式输入，拒绝GPU/runtime/ExecutionEngine。12ordinary＋6compact配置共30个kernel，18次编译与自重汇编全部通过，102个实际steady Memory段零VALU，见[资源收据](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/result.json)。这是最终after资源验收，不是旧/新ISA相同证明。 |
| ordinary/compact随机 | [GPU入口](../../../../tests/contrib/moe/check_startup_8x1.py)完成六K、短N/长N、三量化与compact，共48配置。每配置2个seed×2版本×（2次direct＋3次graph）=20检查，总960；[完整收据](../../../../tests/contrib/moe/results/readme_latest_20260911/gpu/result.json)为`complete=true`，最大rel_l2=0.003345513716340065，新旧输出bit-identical。正确性不调整PTL、频率、功耗或NUMA设置；随机W验收独立于性能全1W。 |
| 公共整链/额外边界 | [公共测试](../../../../tests/contrib/moe/test_moe.py)、[compact专项](../../../../tests/contrib/moe/test_compact_m64_down.py)及[任务表exact cover](../../../../tests/contrib/moe/test_compact_m64_tasks.py)通过[本轮入口](../../../../tests/contrib/moe/run_readme_latest.py)完成147项、0失败/错误/跳过，见[收据](../../../../tests/contrib/moe/results/readme_latest_20260911/public/result.json)。含ordinary/compact整链、超过2GiB真实权重偏移、`test_compact_post_balance_full_coverage`的实际full/full+tail及graph。含任务表CPU节点，147不是147个GPU配置，不与960次检查相加；整链仍使用N为512倍数。 |

**以下为仓库根目录下的独立复现模板。** 本次实际结果以链接收据为准，不覆盖成功、失败或历史记录，不重新运行准备步骤或修改冻结SHA绕过门禁。[文档合同](../../../../tests/contrib/moe/test_moe_readme.py)逐格重算数字、IQR和几何。[pytest配置](../../../../pytest.ini)为`python_files=*.py`，必须显式file/node，不对目录盲跑；若源码、驱动或依赖身份变化，应建立新的配套验证。

```bash
set -euo pipefail
REPO="$PWD"
PY="$REPO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/tests/contrib/moe:$REPO/tests/flydsl/attn_4wave/tools:/opt/aiter:/usr/local/lib/python3.10/dist-packages"
RUN="$REPO/tests/contrib/moe/results/manual_latest_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$RUN" ]] && mkdir -p "$RUN"
unset MOE_PREFILL_TILE_K MOE_8X1_ROLLING_EPILOGUE
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
	"$PY" -m pytest -q -p no:cacheprovider \
	tests/contrib/moe/test_all_8x1_schedule.py tests/contrib/moe/test_startup_8x1.py \
	tests/contrib/moe/test_8x1_direction_names.py \
	tests/contrib/moe/test_moe_readme.py --junitxml="$RUN/cpu.xml"
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 COMPILE_ONLY=1 \
ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942 \
	"$PY" tests/contrib/moe/check_startup_8x1.py \
	--compile --version after \
	--k 256 --n 128 --topk 4 --quant ptpc --padding 128 \
	--store-cache 2 --output "$RUN/offline_k256_n128_ptpc"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
HIP_VISIBLE_DEVICES=4 \
	"$PY" tests/contrib/moe/check_startup_8x1.py \
	--gpu 4 --output "$RUN/gpu"
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
	"$PY" tests/contrib/moe/run_readme_latest.py --resources --output "$RUN/resources"
HIP_VISIBLE_DEVICES=4 \
	"$PY" tests/contrib/moe/run_readme_latest.py --public --gpu 4 --output "$RUN/public"
HIP_VISIBLE_DEVICES=4 \
	"$PY" tests/contrib/moe/run_readme_latest.py --matrix --gpu 4 \
	--validation "$RUN/gpu/result.json" --output "$RUN/matrix"
```

单配置offline只编译一个after；`--resources`生成完整18配置。GPU48配置覆盖每K的N128/256两种权重量化、N384 mixed、N512/896 PTPC及一组compact；不能等同旧108配置集合。本轮矩阵门禁直接核验当前48配置/960检查的raw、来源和执行ELF，不改写旧108配置门禁或历史收据。仅设置一层HIP过滤，物理GPU4在进程内为cuda:0。native cache启用，复用也必须匹配源/环境/配置身份。

#### 本轮计时与统计

- Down：ordinary一个kernel，compact构表/full/tail全计；矩阵每次fresh测量在event外先公共default gateup priming刷新上下文。
- Combined：同event内Down+sorted_sum，gateup仍在外。Full：sorting、两次量化、gateup、down、invert、sum，中间张量在调用内创建，不是全地址固定/graph计时。
- 三phase不能相减为独立sum/gateup耗时，也不能各选最优拼一个实现；不做event扣除。编译/审计/父墙钟不当kernel时延。

$$
F_D=F_{Combined}=2\times B\times TopK\times N\times K,\quad
F_{Full}=3F_D,\quad T_{effective}=F/(t_{ms}\times10^9).
$$

所有格为 **ms / wall-time有效TFLOPS**，按useful而非padding执行行计F；Full的3倍仅计三路GEMM，其它算子耗时仍在event内。ATT union×roof只能称“模型TFLOPS”，不混用。

本轮性能使用**GPU4 / MI308X / gfx942 / 80CU**，PCI `0001:0B:00.0`、ID `0xbe022834ccc51849`。目标门禁busy≤5%、VRAM≤20%；其它卡负载另记录，不宣称全机空闲或连续监控。**NUMA保持原值，不修改主机设置；PTL实核Enabled / VECTOR,F8，请求1800MHz determinism/650W**，不把请求当持续实测频率。随机GPU批次的NUMA为0，之后外部改为1，公共测试及性能入场记录为1；各批次内部均保持其入场值，不能把批次间外部变化归为kernel修改。

**10-buffer**轮换、warmup，同半轮候选/对照使用相同输入与权重地址，ABBA/BAAB配合候选轮转正序＋反序；每轮两个sample先取均值，再算配对比值，保留raw与长尾。结束drain stream、后检和卸载kernel，再恢复原auto/PTL。9份run退出快照全部为auto、PTL Disabled/N/A、NUMA1、busy0、VRAM0%；这是进出快照而非连续空闲证明。[目标卡托管](../../../../tests/contrib/moe/k192_cshuffle_workflow.py)和[本轮矩阵入口](../../../../tests/contrib/moe/run_readme_latest.py)记录实际协议及身份。本次全矩阵是用户明确授权的README更新；无gateup priming的快测绝对ms不得与此矩阵混池。

本轮矩阵仅将原README推荐**标签**作为incumbent，旧绝对ms不参与选择。三phase配对改善**Q1均>0**才支配原推荐，竞争候选再互比，否则保守保留，推荐不必是各phase绝对median最小。先完成七case全候选24轮，再由预定规则一次冻结Hy3 1K/2K及Xiaomi 32K三个点的独立48轮确认；48整点替代24，不拼72轮、不按方向追加抽样。**IQR是样本分位，不是置信区间；跨0不证明等价。** 同轮$r=mean(t_{candidate,2})/mean(t_{control,2})$：改善$1-r$，时延增幅$r-1$，吞吐损失$1-1/r$；配对median不等于绝对median之比。证据：[计划](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/plan.json)、[一次冻结的确认计划](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/confirmation_plan.json)。

<a id="matrix42"></a>

### 1.3 最新源码42点推荐

全部来自[本轮矩阵汇总](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/matrix_summary.json)，SHA256：`354485c8ca8a0f03a9927fba7195bface213cab1b114655da434ceab3d0d5e79`。七case×六Batch共42点；9份run，3点采用独立48轮，其余采用24轮；均使用同一最终13源，不混入旧run时延。

推荐计数：**8x1_compact 15**、**1x4-P128 1**、**8x1 5**、**1x4 16**、**default 5**。相对原报告改选0点；选择依据同run三phase配对门槛，不是每phase各取最快。

| Case / Batch | 当前推荐 | Down ms / T | Combined ms / T | Full ms / T | 轮数/来源 |
|---|---|---|---|---|---|
| Hy3 K192 / 1K | 8x1_compact | 0.102980 / 140.76 | 0.130680 / 110.92 | 0.387022 / 112.36 | [48](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba48.json) |
| Hy3 K192 / 2K | 8x1_compact | 0.158821 / 182.54 | 0.216781 / 133.73 | 0.567542 / 153.25 | [48](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba48.json) |
| Hy3 K192 / 4K | 1x4-P128 | 0.233501 / 248.32 | 0.340861 / 170.10 | 0.812823 / 214.00 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba24.json) |
| Hy3 K192 / 8K | 8x1_compact | 0.373901 / 310.15 | 0.583243 / 198.83 | 1.421985 / 244.65 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba24.json) |
| Hy3 K192 / 16K | 8x1 | 0.600002 / 386.55 | 1.011924 / 229.20 | 2.569129 / 270.83 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba24.json) |
| Hy3 K192 / 32K | 8x1 | 1.073644 / 432.04 | 1.905907 / 243.38 | 5.064599 / 274.76 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/hy3_abba24.json) |
| Qwen397 K512 / 1K | 1x4 | 0.489461 / 87.75 | 0.523922 / 81.98 | 1.220564 / 105.57 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K512 / 2K | 1x4 | 0.501642 / 171.24 | 0.566882 / 151.53 | 1.302045 / 197.92 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K512 / 4K | default | 0.797663 / 215.38 | 0.916923 / 187.36 | 2.316088 / 222.53 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K512 / 8K | default | 1.134124 / 302.96 | 1.336725 / 257.04 | 3.352952 / 307.43 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K512 / 16K | 8x1_compact | 1.573106 / 436.84 | 2.032807 / 338.05 | 5.481740 / 376.08 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K512 / 32K | 8x1_compact | 3.092511 / 444.43 | 3.995975 / 343.94 | 10.805479 / 381.58 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_abba24.json) |
| Qwen397 K256 / 1K | 1x4 | 0.274841 / 78.14 | 0.305721 / 70.24 | 0.694722 / 92.73 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen397 K256 / 2K | 1x4 | 0.279461 / 153.69 | 0.339741 / 126.42 | 0.769283 / 167.49 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen397 K256 / 4K | 1x4 | 0.464561 / 184.90 | 0.587822 / 146.13 | 1.361745 / 189.24 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen397 K256 / 8K | 1x4 | 0.662843 / 259.18 | 0.893083 / 192.37 | 2.063607 / 249.75 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen397 K256 / 16K | 8x1_compact | 0.935823 / 367.16 | 1.395945 / 246.14 | 3.403053 / 302.90 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen397 K256 / 32K | 8x1_compact | 1.809067 / 379.86 | 2.707330 / 253.83 | 6.614224 / 311.69 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen397_k256_abba24.json) |
| Qwen35 K512 / 1K | default | 0.142820 / 120.29 | 0.155200 / 110.69 | 0.380121 / 135.59 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K512 / 2K | default | 0.151360 / 227.01 | 0.171921 / 199.86 | 0.433442 / 237.82 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K512 / 4K | default | 0.227801 / 301.66 | 0.269441 / 255.04 | 0.713722 / 288.85 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K512 / 8K | 8x1_compact | 0.321761 / 427.15 | 0.412822 / 332.93 | 1.240305 / 332.43 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K512 / 16K | 8x1_compact | 0.632502 / 434.59 | 0.798823 / 344.10 | 2.388748 / 345.22 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K512 / 32K | 8x1 | 1.191864 / 461.26 | 1.510925 / 363.85 | 4.625277 / 356.58 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_abba24.json) |
| Qwen35 K256 / 1K | 1x4 | 0.079841 / 107.59 | 0.094240 / 91.15 | 0.227821 / 113.11 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Qwen35 K256 / 2K | 1x4 | 0.083601 / 205.50 | 0.109520 / 156.86 | 0.276721 / 186.25 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Qwen35 K256 / 4K | 1x4 | 0.133460 / 257.45 | 0.179980 / 190.91 | 0.453602 / 227.25 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Qwen35 K256 / 8K | 8x1_compact | 0.204360 / 336.27 | 0.292221 / 235.16 | 0.792263 / 260.21 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Qwen35 K256 / 16K | 8x1_compact | 0.371241 / 370.21 | 0.540942 / 254.07 | 1.498266 / 275.20 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Qwen35 K256 / 32K | 8x1 | 0.704522 / 390.16 | 1.036744 / 265.14 | 2.906570 / 283.71 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/qwen35_k256_abba24.json) |
| Xiaomi K256 / 1K | 1x4 | 0.290201 / 88.80 | 0.330721 / 77.92 | 0.741162 / 104.31 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba24.json) |
| Xiaomi K256 / 2K | 1x4 | 0.298081 / 172.90 | 0.371021 / 138.91 | 0.835943 / 184.96 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba24.json) |
| Xiaomi K256 / 4K | 1x4 | 0.515502 / 199.96 | 0.643602 / 160.16 | 1.482406 / 208.61 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba24.json) |
| Xiaomi K256 / 8K | 1x4 | 0.743623 / 277.24 | 0.999103 / 206.34 | 2.276509 / 271.68 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba24.json) |
| Xiaomi K256 / 16K | 8x1_compact | 1.250284 / 329.78 | 1.759287 / 234.37 | 4.156636 / 297.58 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba24.json) |
| Xiaomi K256 / 32K | 8x1_compact | 2.191669 / 376.26 | 3.183593 / 259.03 | 7.753770 / 319.06 | [48](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/xiaomi_abba48.json) |
| H3 K384 / 1K | 1x4 | 0.155360 / 124.40 | 0.175720 / 109.99 | 0.413102 / 140.36 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |
| H3 K384 / 2K | 1x4 | 0.158961 / 243.17 | 0.200481 / 192.81 | 0.476282 / 243.48 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |
| H3 K384 / 4K | 1x4 | 0.298881 / 258.66 | 0.374561 / 206.40 | 0.853404 / 271.77 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |
| H3 K384 / 8K | 8x1 | 0.396842 / 389.62 | 0.544062 / 284.19 | 1.448026 / 320.34 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |
| H3 K384 / 16K | 8x1_compact | 0.676602 / 457.05 | 0.956883 / 323.17 | 2.635871 / 351.96 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |
| H3 K384 / 32K | 8x1_compact | 1.339425 / 461.75 | 1.895908 / 326.22 | 5.172501 / 358.71 | [24](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/h3_abba24.json) |

Hy3 1K/2K的compact均衡后full为0，仍有构表、空full、tail三个launch；不能称为8x1满块吞吐胜出。全部性能W为全1，随机W由独立48配置验收覆盖。

<a id="historical-nonselection"></a>

### 1.4 最新未选ordinary8x1/compact账本（64行）

逐点列出[本轮矩阵汇总](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/matrix_summary.json)中未选的ordinary8x1与compact，证据和轮数与1.3同点一致。行数按实际推荐重新生成，不固定为旧报告64行。

D/C/F=Down/Combined/Full，ms/T为wall-time有效TFLOPS；配对改善正值更快，显示median [Q1,Q3]%三位。**IQR不是置信区间，跨0不证明等价**。行比为候选/所选执行填充行；F/T为均衡routing的CPU full/tail几何，非实测驻留。full=0仍三个launch；不以padding或零spill直接解释全部性能，不推测未采集的bank/cache/occupancy。

| 点 | 未选候选 vs 所选 | 候选D/C/F ms / T | 相对所选配对改善 median [Q1,Q3]% | 工作量 | 未选原因 |
|---|---|---|---|---|---|
| Hy3 K192 / 1K | 8x1 vs 8x1_compact | Down 0.225421 / 64.30<br>Combined 0.258241 / 56.13<br>Full 0.691882 / 62.85 | Down -116.475% [-117.513, -115.430]<br>Combined -95.204% [-96.095, -94.423]<br>Full -78.826% [-86.891, -67.907] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 2K | 8x1 vs 8x1_compact | Down 0.229481 / 126.33<br>Combined 0.287901 / 100.70<br>Full 0.751642 / 115.71 | Down -45.261% [-46.970, -43.620]<br>Combined -33.273% [-34.214, -31.833]<br>Full -32.955% [-35.719, -29.654] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 4K | 8x1 vs 1x4-P128 | Down 0.238621 / 242.99<br>Combined 0.347682 / 166.77<br>Full 0.879003 / 197.89 | Down -2.161% [-2.436, -1.906]<br>Combined -2.071% [-2.362, -1.533]<br>Full -7.391% [-10.449, -4.225] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 4K | 8x1_compact vs 1x4-P128 | Down 0.242181 / 239.42<br>Combined 0.349141 / 166.07<br>Full 0.830083 / 209.55 | Down -3.565% [-3.951, -3.392]<br>Combined -2.411% [-2.607, -2.096]<br>Full -1.354% [-3.081, +0.682] | 行比1.000；F/T=0/579 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 8K | 8x1 vs 8x1_compact | Down 0.417201 / 277.96<br>Combined 0.623503 / 185.99<br>Full 1.561346 / 222.82 | Down -12.050% [-12.490, -10.744]<br>Combined -7.512% [-8.020, -7.052]<br>Full -9.925% [-12.172, -8.904] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 16K | 8x1_compact vs 8x1 | Down 0.646422 / 358.79<br>Combined 1.059584 / 218.89<br>Full 2.616171 / 265.96 | Down -7.863% [-8.475, -7.386]<br>Combined -4.753% [-5.140, -4.417]<br>Full -2.127% [-2.845, -1.066] | 行比1.000；F/T=560/76 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Hy3 K192 / 32K | 8x1_compact vs 8x1 | Down 1.132664 / 409.53<br>Combined 1.967207 / 235.79<br>Full 5.128999 / 271.31 | Down -5.304% [-5.506, -5.140]<br>Combined -2.958% [-3.152, -2.693]<br>Full -1.145% [-1.602, -0.664] | 行比1.000；F/T=1120/152 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 1K | 8x1 vs 1x4 | Down 1.270585 / 33.80<br>Combined 1.306425 / 32.88<br>Full 3.503993 / 36.77 | Down -160.271% [-163.146, -158.375]<br>Combined -150.979% [-153.893, -148.325]<br>Full -183.195% [-189.047, -176.392] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 1K | 8x1_compact vs 1x4 | Down 0.497862 / 86.27<br>Combined 0.528362 / 81.29<br>Full 1.229445 / 104.80 | Down -1.382% [-2.507, -0.387]<br>Combined -1.289% [-2.342, -0.085]<br>Full -0.648% [-5.546, +2.815] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 2K | 8x1 vs 1x4 | Down 1.277805 / 67.22<br>Combined 1.370005 / 62.70<br>Full 3.619213 / 71.20 | Down -156.837% [-159.269, -152.326]<br>Combined -143.014% [-146.218, -137.024]<br>Full -175.321% [-178.020, -166.483] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 2K | 8x1_compact vs 1x4 | Down 0.506641 / 169.55<br>Combined 0.571723 / 150.25<br>Full 1.324585 / 194.55 | Down -1.428% [-2.548, -0.232]<br>Combined -1.844% [-3.376, -0.009]<br>Full -0.551% [-5.295, +2.228] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 4K | 8x1 vs default | Down 1.233424 / 139.29<br>Combined 1.364985 / 125.86<br>Full 3.777354 / 136.44 | Down -54.112% [-56.172, -51.803]<br>Combined -49.029% [-50.591, -47.105]<br>Full -63.202% [-64.557, -60.153] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 4K | 8x1_compact vs default | Down 0.852823 / 201.45<br>Combined 0.985464 / 174.33<br>Full 2.377009 / 216.83 | Down -6.680% [-7.426, -5.663]<br>Combined -6.658% [-7.535, -6.071]<br>Full -1.867% [-4.986, +0.142] | 行比1.000；F/T=0/1024 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 8K | 8x1 vs default | Down 1.178244 / 291.62<br>Combined 1.418745 / 242.18<br>Full 3.960675 / 260.26 | Down -4.188% [-4.515, -3.651]<br>Combined -5.839% [-6.301, -5.197]<br>Full -17.562% [-19.450, -16.797] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 8K | 8x1_compact vs default | Down 1.220425 / 281.54<br>Combined 1.449506 / 237.04<br>Full 3.500813 / 294.44 | Down -7.822% [-8.210, -7.338]<br>Combined -8.284% [-8.708, -8.064]<br>Full -3.632% [-4.706, -2.832] | 行比1.000；F/T=0/1536 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 16K | 8x1 vs 8x1_compact | Down 2.199808 / 312.39<br>Combined 2.655470 / 258.78<br>Full 7.645788 / 269.64 | Down -39.862% [-40.070, -39.697]<br>Combined -30.979% [-31.416, -30.582]<br>Full -39.429% [-39.781, -38.721] | 行比1.600 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K512 / 32K | 8x1 vs 8x1_compact | Down 3.440392 / 399.49<br>Combined 4.348596 / 316.05<br>Full 12.097664 / 340.82 | Down -11.135% [-11.468, -10.940]<br>Combined -8.736% [-9.076, -8.489]<br>Full -11.726% [-12.071, -11.522] | 行比1.200 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 1K | 8x1 vs 1x4 | Down 0.703862 / 30.51<br>Combined 0.736283 / 29.17<br>Full 1.899046 / 33.92 | Down -155.245% [-156.280, -153.571]<br>Combined -139.597% [-141.837, -138.487]<br>Full -171.469% [-183.659, -144.806] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 1K | 8x1_compact vs 1x4 | Down 0.284062 / 75.60<br>Combined 0.314861 / 68.20<br>Full 0.699663 / 92.08 | Down -3.239% [-3.615, -2.639]<br>Combined -2.637% [-3.115, -2.260]<br>Full -0.988% [-10.531, +7.083] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 2K | 8x1 vs 1x4 | Down 0.704683 / 60.95<br>Combined 0.779903 / 55.07<br>Full 1.989347 / 64.77 | Down -152.922% [-155.011, -151.175]<br>Combined -128.501% [-129.286, -127.584]<br>Full -156.461% [-159.948, -133.269] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 2K | 8x1_compact vs 1x4 | Down 0.288341 / 148.95<br>Combined 0.349421 / 122.92<br>Full 0.776982 / 165.83 | Down -3.254% [-3.761, -2.785]<br>Combined -2.385% [-2.724, -2.054]<br>Full -0.978% [-9.598, +8.779] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 4K | 8x1 vs 1x4 | Down 0.712262 / 120.60<br>Combined 0.845903 / 101.55<br>Full 2.168667 / 118.83 | Down -52.372% [-54.237, -51.262]<br>Combined -44.242% [-44.976, -42.606]<br>Full -59.514% [-62.009, -52.940] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 4K | 8x1_compact vs 1x4 | Down 0.477022 / 180.07<br>Combined 0.598342 / 143.56<br>Full 1.378665 / 186.92 | Down -2.545% [-3.082, -1.756]<br>Combined -1.813% [-2.403, -0.732]<br>Full -0.757% [-5.144, +2.678] | 行比1.000；F/T=0/1024 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 8K | 8x1 vs 1x4 | Down 0.720403 / 238.48<br>Combined 0.961864 / 178.61<br>Full 2.431769 / 211.94 | Down -8.224% [-9.039, -7.545]<br>Combined -7.446% [-7.794, -6.475]<br>Full -18.117% [-18.644, -15.311] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 8K | 8x1_compact vs 1x4 | Down 0.671162 / 255.97<br>Combined 0.905963 / 189.63<br>Full 2.070607 / 248.91 | Down -1.807% [-2.293, -1.192]<br>Combined -1.453% [-2.299, -0.795]<br>Full -0.638% [-2.007, +1.129] | 行比1.000；F/T=0/1536 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 16K | 8x1 vs 8x1_compact | Down 1.329225 / 258.49<br>Combined 1.804866 / 190.37<br>Full 4.568836 / 225.61 | Down -40.478% [-41.271, -40.089]<br>Combined -28.778% [-29.804, -27.903]<br>Full -34.048% [-34.543, -33.631] | 行比1.600 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen397 K256 / 32K | 8x1 vs 8x1_compact | Down 1.989268 / 345.45<br>Combined 2.897850 / 237.14<br>Full 7.321487 / 281.58 | Down -9.663% [-10.062, -9.447]<br>Combined -6.554% [-6.736, -6.208]<br>Full -10.743% [-11.135, -10.220] | 行比1.200 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 1K | 8x1 vs default | Down 0.343581 / 50.00<br>Combined 0.358721 / 47.89<br>Full 0.993524 / 51.88 | Down -140.743% [-141.417, -139.923]<br>Combined -131.338% [-132.261, -130.701]<br>Full -161.668% [-179.247, -123.618] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 1K | 8x1_compact vs default | Down 0.155041 / 110.81<br>Combined 0.169360 / 101.44<br>Full 0.390821 / 131.88 | Down -8.577% [-9.093, -8.041]<br>Combined -9.120% [-9.915, -8.312]<br>Full -2.858% [-20.158, +11.889] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 2K | 8x1 vs default | Down 0.349601 / 98.28<br>Combined 0.376481 / 91.27<br>Full 1.046724 / 98.48 | Down -131.019% [-132.346, -130.725]<br>Combined -119.294% [-120.054, -118.580]<br>Full -141.912% [-155.957, -112.304] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 2K | 8x1_compact vs default | Down 0.159101 / 215.96<br>Combined 0.185821 / 184.91<br>Full 0.444922 / 231.68 | Down -4.936% [-5.747, -4.581]<br>Combined -7.926% [-9.198, -6.979]<br>Full -2.570% [-17.654, +10.110] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 4K | 8x1 vs default | Down 0.355381 / 193.37<br>Combined 0.408541 / 168.21<br>Full 1.112804 / 185.26 | Down -56.034% [-56.472, -55.644]<br>Combined -51.993% [-52.553, -51.228]<br>Full -55.921% [-64.380, -44.293] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 4K | 8x1_compact vs default | Down 0.247341 / 277.83<br>Combined 0.294201 / 233.58<br>Full 0.738223 / 279.26 | Down -8.475% [-9.028, -7.911]<br>Combined -9.360% [-9.697, -8.924]<br>Full -3.431% [-11.388, +4.205] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 8K | 8x1 vs 8x1_compact | Down 0.370061 / 371.40<br>Combined 0.461061 / 298.09<br>Full 1.281384 / 321.77 | Down -14.979% [-15.103, -14.837]<br>Combined -11.549% [-11.676, -11.450]<br>Full -3.866% [-7.358, +1.257] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 16K | 8x1 vs 8x1_compact | Down 0.644163 / 426.72<br>Combined 0.810543 / 339.13<br>Full 2.383849 / 345.93 | Down -1.834% [-1.934, -1.706]<br>Combined -1.473% [-1.572, -1.405]<br>Full -0.408% [-0.956, +1.954] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K512 / 32K | 8x1_compact vs 8x1 | Down 1.210925 / 454.00<br>Combined 1.530285 / 359.25<br>Full 4.645437 / 355.03 | Down -1.604% [-1.650, -1.532]<br>Combined -1.288% [-1.327, -1.232]<br>Full -0.469% [-1.157, +0.354] | 行比1.000；F/T=1024/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 1K | 8x1 vs 1x4 | Down 0.205161 / 41.87<br>Combined 0.221181 / 38.84<br>Full 0.632603 / 40.74 | Down -156.712% [-158.288, -155.291]<br>Combined -134.317% [-134.919, -133.785]<br>Full -128.250% [-164.031, -111.394] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 1K | 8x1_compact vs 1x4 | Down 0.086780 / 98.98<br>Combined 0.101201 / 84.88<br>Full 0.233761 / 110.24 | Down -8.785% [-9.343, -8.126]<br>Combined -7.385% [-7.900, -7.109]<br>Full +2.875% [-21.535, +23.506] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 2K | 8x1 vs 1x4 | Down 0.210261 / 81.71<br>Combined 0.237321 / 72.39<br>Full 0.616562 / 83.59 | Down -151.648% [-152.460, -149.963]<br>Combined -117.605% [-118.473, -115.795]<br>Full -115.291% [-141.768, -81.638] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 2K | 8x1_compact vs 1x4 | Down 0.090260 / 190.34<br>Combined 0.116301 / 147.72<br>Full 0.281942 / 182.80 | Down -8.316% [-8.726, -7.843]<br>Combined -6.530% [-6.857, -6.003]<br>Full -0.163% [-21.328, +17.750] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 4K | 8x1 vs 1x4 | Down 0.213420 / 161.00<br>Combined 0.259681 / 132.32<br>Full 0.663903 / 155.26 | Down -59.398% [-60.272, -58.362]<br>Combined -44.240% [-46.069, -43.365]<br>Full -46.432% [-59.570, -30.228] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 4K | 8x1_compact vs 1x4 | Down 0.140540 / 244.48<br>Combined 0.187200 / 183.55<br>Full 0.460841 / 223.68 | Down -5.246% [-5.738, -4.864]<br>Combined -4.037% [-4.469, -3.545]<br>Full -1.228% [-15.298, +9.686] | 行比1.000；F/T=0/512 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 8K | 8x1 vs 8x1_compact | Down 0.222641 / 308.66<br>Combined 0.312201 / 220.11<br>Full 0.810743 / 254.28 | Down -9.122% [-9.666, -8.716]<br>Combined -6.618% [-6.979, -6.222]<br>Full -2.502% [-8.983, +3.751] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 16K | 8x1 vs 8x1_compact | Down 0.383981 / 357.93<br>Combined 0.554062 / 248.06<br>Full 1.509286 / 273.19 | Down -3.475% [-3.618, -3.305]<br>Combined -2.284% [-2.574, -2.136]<br>Full -0.855% [-3.138, +1.569] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Qwen35 K256 / 32K | 8x1_compact vs 8x1 | Down 0.727663 / 377.75<br>Combined 1.055404 / 260.45<br>Full 2.906471 / 283.72 | Down -3.088% [-3.485, -2.608]<br>Combined -2.173% [-2.472, -1.798]<br>Full -0.070% [-0.926, +0.702] | 行比1.000；F/T=1024/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 1K | 8x1 vs 1x4 | Down 0.749663 / 34.38<br>Combined 0.788843 / 32.67<br>Full 2.071227 / 37.33 | Down -158.385% [-160.865, -156.779]<br>Combined -137.928% [-140.291, -136.318]<br>Full -178.114% [-188.251, -156.002] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 1K | 8x1_compact vs 1x4 | Down 0.296761 / 86.84<br>Combined 0.335021 / 76.92<br>Full 0.751263 / 102.91 | Down -2.801% [-3.132, -2.339]<br>Combined -2.044% [-2.521, -1.765]<br>Full -1.109% [-9.816, +7.132] | 行比1.000；F/T=0/384 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 2K | 8x1 vs 1x4 | Down 0.747882 / 68.91<br>Combined 0.825143 / 62.46<br>Full 2.152988 / 71.82 | Down -152.214% [-154.071, -150.667]<br>Combined -122.878% [-124.452, -121.320]<br>Full -155.607% [-162.687, -139.889] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 2K | 8x1_compact vs 1x4 | Down 0.306761 / 168.01<br>Combined 0.380682 / 135.39<br>Full 0.850143 / 181.87 | Down -2.623% [-3.043, -2.450]<br>Combined -2.208% [-2.662, -1.876]<br>Full -1.169% [-8.454, +5.355] | 行比1.000；F/T=0/384 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 4K | 8x1 vs 1x4 | Down 0.760843 / 135.48<br>Combined 0.896303 / 115.00<br>Full 2.351928 / 131.48 | Down -46.843% [-48.225, -45.291]<br>Combined -38.276% [-39.314, -36.679]<br>Full -58.398% [-59.338, -52.162] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 4K | 8x1_compact vs 1x4 | Down 0.521602 / 197.62<br>Combined 0.652502 / 157.98<br>Full 1.497746 / 206.47 | Down -1.479% [-2.185, -0.976]<br>Combined -1.735% [-2.219, -0.970]<br>Full -0.827% [-4.332, +2.275] | 行比1.000；F/T=0/768 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 8K | 8x1 vs 1x4 | Down 0.767083 / 268.76<br>Combined 1.027124 / 200.71<br>Full 2.607129 / 237.22 | Down -3.364% [-3.647, -2.543]<br>Combined -2.827% [-3.391, -1.925]<br>Full -15.060% [-15.702, -13.367] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 8K | 8x1_compact vs 1x4 | Down 0.751303 / 274.40<br>Combined 1.007864 / 204.55<br>Full 2.284148 / 270.77 | Down -1.264% [-1.884, -0.300]<br>Combined -1.124% [-1.732, +0.177]<br>Full -0.301% [-1.753, +0.562] | 行比1.000；F/T=0/1152 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 16K | 8x1 vs 8x1_compact | Down 1.442885 / 285.76<br>Combined 1.974228 / 208.85<br>Full 4.979059 / 248.43 | Down -14.697% [-15.589, -14.048]<br>Combined -11.970% [-13.315, -10.571]<br>Full -19.279% [-19.932, -18.957] | 行比1.333 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| Xiaomi K256 / 32K | 8x1 vs 8x1_compact | Down 2.182009 / 377.92<br>Combined 3.175532 / 259.68<br>Full 7.910131 / 312.75 | Down +0.989% [+0.625, +1.329]<br>Combined +0.605% [+0.343, +0.817]<br>Full -1.829% [-2.330, -1.480] | 行比1.091 | Down有收益，但Full配对回退；对现选路径Q1≤0：Full；对原推荐未过：Full |
| H3 K384 / 1K | 8x1 vs 1x4 | Down 0.385742 / 50.10<br>Combined 0.407621 / 47.41<br>Full 1.052804 / 55.07 | Down -148.951% [-150.221, -148.291]<br>Combined -131.925% [-133.476, -130.925]<br>Full -154.516% [-172.297, -123.359] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 1K | 8x1_compact vs 1x4 | Down 0.163380 / 118.30<br>Combined 0.183801 / 105.15<br>Full 0.420261 / 137.97 | Down -5.133% [-5.594, -4.562]<br>Combined -4.418% [-4.799, -4.014]<br>Full -2.169% [-17.164, +11.018] | 行比1.000；F/T=0/128 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 2K | 8x1 vs 1x4 | Down 0.388701 / 99.45<br>Combined 0.430741 / 89.74<br>Full 1.105145 / 104.93 | Down -145.221% [-147.220, -144.477]<br>Combined -115.915% [-118.255, -114.581]<br>Full -132.388% [-147.149, -108.911] | 行比4.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 2K | 8x1_compact vs 1x4 | Down 0.166961 / 231.52<br>Combined 0.208661 / 185.25<br>Full 0.484582 / 239.31 | Down -4.972% [-5.562, -4.665]<br>Combined -4.080% [-4.597, -3.353]<br>Full -1.834% [-14.345, +9.444] | 行比1.000；F/T=0/128 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 4K | 8x1 vs 1x4 | Down 0.390522 / 197.96<br>Combined 0.466542 / 165.71<br>Full 1.207405 / 192.09 | Down -30.639% [-31.063, -29.966]<br>Combined -24.763% [-25.507, -23.996]<br>Full -41.766% [-47.893, -33.830] | 行比2.000 | Down配对较慢；执行填充行更多；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 4K | 8x1_compact vs 1x4 | Down 0.310502 / 248.98<br>Combined 0.384662 / 200.98<br>Full 0.858443 / 270.17 | Down -3.840% [-4.198, -3.573]<br>Combined -2.999% [-3.532, -2.590]<br>Full -0.967% [-7.117, +4.733] | 行比1.000；F/T=0/256 | Down配对较慢；满块为0，仍走构表/空full/tail；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 8K | 8x1_compact vs 8x1 | Down 0.404481 / 382.26<br>Combined 0.553102 / 279.55<br>Full 1.452205 / 319.42 | Down -1.886% [-2.054, -1.448]<br>Combined -1.379% [-2.024, -1.089]<br>Full -0.981% [-2.005, +1.976] | 行比1.000；F/T=128/0 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 16K | 8x1 vs 8x1_compact | Down 0.787003 / 392.93<br>Combined 1.071124 / 288.70<br>Full 2.755811 / 336.64 | Down -16.280% [-16.377, -16.031]<br>Combined -11.950% [-12.367, -11.501]<br>Full -4.328% [-5.458, -3.408] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |
| H3 K384 / 32K | 8x1 vs 8x1_compact | Down 1.375586 / 449.61<br>Combined 1.932027 / 320.12<br>Full 5.219501 / 355.48 | Down -2.710% [-2.749, -2.655]<br>Combined -1.898% [-1.998, -1.851]<br>Full -0.556% [-1.445, +0.210] | 行比1.000 | Down配对较慢；对现选路径Q1≤0：Down/Combined/Full；对原推荐未过：Down/Combined/Full |

<a id="migration-confirmation"></a>

### 1.5 与原报告同标签性能比较

比较[原README矩阵](../../../../tests/contrib/moe/results/merged_8x1_20260910/matrix/matrix_summary.json)与[最新矩阵](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/matrix_summary.json)：每点固定原推荐同标签，计算$(t_{new}/t_{old}-1)\times100\%$，负值表示最新时延更短；即使改选，也不换分子标签。这是**跨run非配对描述，不是隔离源码因果实验**；旧样本不参与当前选择，也不与新样本拼池。

| Phase | 42点时延变化的median | 最小值 | 最大值 | 绝对变化≤3% | 绝对变化≤5% |
|---|---:|---:|---:|---:|---:|
| Down | -0.110% | -2.430% | +0.840% | 42/42 | 42/42 |
| Combined | -0.102% | -3.986% | +1.030% | 41/42 | 42/42 |
| Full | -0.112% | -1.288% | +0.942% | 42/42 | 42/42 |

为避免原推荐多为非8x1而掩盖分支变化，另固定ordinary8x1、compact标签各比较42点：

| 固定标签 / Phase | 时延变化median | 最小值 | 最大值 | 绝对变化≤5% |
|---|---:|---:|---:|---:|
| 8x1 / Down | -0.365% | -2.068% | +1.145% | 42/42 |
| 8x1 / Combined | -0.368% | -3.986% | +1.303% | 42/42 |
| 8x1 / Full | -0.132% | -1.381% | +1.629% | 42/42 |
| 8x1_compact / Down | -0.241% | -2.430% | +0.840% | 42/42 |
| 8x1_compact / Combined | -0.078% | -3.787% | +0.728% | 42/42 |
| 8x1_compact / Full | -0.133% | -1.133% | +1.369% | 42/42 |

原报告与本次性能run均记录NUMA=1及PTL Enabled/VECTOR,F8、1800MHz请求、650W。本次较早随机GPU验收NUMA=0，外部在公共/性能测试前改回1；测试本身均保留入场值。相同硬件设置仍不能消除跨run频率、缓存和系统噪声，因此只描述复测差异，不作等价检验或纯代码加速归因。

**结论：与原报告基本一致。** 42点推荐标签全部相同；原推荐同标签的Down/Full全部在±3%内，Combined为41/42在±3%内、全部在±5%内。唯一超过3%的点为Qwen35 K256/32K Combined降低3.986%（1.079783 ms / 254.57 T → 1.036744 ms / 265.14 T）。固定ordinary8x1和compact标签的全部三phase也均在±5%内；这是观察上的一致性，不是统计等价或隔离源码因果证明。

<a id="validation-boundaries"></a>

### 1.6 最新验证与证据边界

各批次的candidate/after均绑定本文顶部的最终13源；计数只报告实际完成项，不与旧批次相加。

| 验证层 | 本轮实际结果与范围 | 冻结证据 |
|---|---|---|
| CPU实现 | 169项通过、0失败/错误/跳过；检查主流水、全部K贡献、pack/状态、短N读取、薄封装内联和B准备移位。此前168项为移位前，不再冒称最新结果。 | [169项XML](../../../../tests/contrib/moe/results/startup_8x1_20260911/cpu_views.xml) |
| GPU随机Down | 48配置、960检查，2seed/2版本/2direct＋3graph；max rel_l2=0.003345513716340065<0.005，新旧bit-identical，finite/padding/inactive与compact workspace投毒重建通过；96artifact。GPU4前后auto/PTL Disabled/N/A/650W/NUMA0，未改设置。 | [完整GPU收据](../../../../tests/contrib/moe/results/readme_latest_20260911/gpu/result.json)，SHA256 `c276bd39a7cacc25e20b544ad15887f1dedb7ac11adccbb5ec5561de694e8d6f` |
| 公共整链/边界 | 147项通过、0失败/错误/跳过；覆盖ordinary/compact整链、实际full/tail、graph和超过2GiB权重偏移，含任务表CPU节点。GPU4前后auto/PTL Disabled/N/A/650W/NUMA1。 | [公共回归](../../../../tests/contrib/moe/results/readme_latest_20260911/public/result.json)、[147项XML](../../../../tests/contrib/moe/results/readme_latest_20260911/public/pytest.xml) |
| fresh offline | 18份after配置、30个kernel全部编译与自重汇编通过；102个实际steady Memory段零VALU，private/Vspill/Sspill/scratch均为0；不是与旧源严格相等声明。 | [资源收据](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/result.json)，SHA256 `c24716a4a29df9a5df4431f0b6ad55864ed46f9e8e167c9cb2388e766ecc9daa` |
| 性能矩阵 | 42点、9份run、3点独立48轮确认；29,088个event、14,544次逐轮检查、376次初始检查；43artifact/57kernel，max rel_l2 vs default=2.065696389763616e-05。全1性能W，所有run绑定相同13源/驱动/设备/设置。 | [矩阵汇总](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/matrix_summary.json)、[执行收据](../../../../tests/contrib/moe/results/readme_latest_20260911/matrix/suite.json) |

[独立复审](../../../../tests/contrib/moe/results/readme_latest_20260911/audited.json)SHA256 `e03f92aebe6c9006d3691ca0a311c26f99229a40dabcdc6a9d798ffa571c7dd4`，从每份raw重新计算统计、候选选择和落选账本，并复核18份资源及实际launcher/ELF/ISA。全部42标签保持原推荐；性能一致性见1.5。原报告及更早失败/成功收据保留原字节，既有ISA差异不会被本次验收追溯改成相同。

性能矩阵只含K192/256/384/512；**K320/K640有当前随机验收和离线资源，没有本七case性能数据**。全1W/均衡routing不能代表随机W或生产trace；Full矩阵检查与公共pytest分别记录，不累加为统一配置数。公共入口从/tmp先加载真实editable包与buffer兼容API，避免仓库同名namespace遮蔽；不为通过测试改公共test_moe或kernel。

没有新增ATT/PMC、occupancy、cache或bank实测。零spill、功能通过、IQR跨0与静态指令数都不是性能等价证明；硬件仅有进出快照，不宣称全过程空闲。[最终CPU验收](../../../../tests/contrib/moe/results/readme_latest_20260911/cpu_final_initial.xml)为**209项通过、0失败/错误/跳过**，包含169项实现合同和40项README数据/链接/历史身份/负例检查；与前面定向结果重叠，不重复相加。

<a id="pipeline"></a>

## 二、流水线设计

以下设计按 **2026-09-11最终源码：首拍前错相、首N从step0进入主循环、原pack消费FIFO、PTPC、固定1N循环** 核对。三个`first_tile()`薄封装已删除，布局/partition集中在`prepare_views()`；主流程直接显示prologue、首N地址、错相屏障、N0/N1、回边、尾部和epilogue。**逐stage表左Memory、右紧接的Compute；每行是单wave一次逻辑配对，不是一个cycle或全WG同步同行。** 首拍改序包含真实等待变化，不声称只是改名或ISA完全相同；MFMA贡献、pack时点和跨N状态保持。

<a id="notation"></a>

### 2.1 符号与读表方法

| 符号 | 含义 |
|---|---|
| N/n/T | N输出总列数，n为N128块编号，T=N/128；标题N=0/1指n=0/1，不是总列数。首轮/过渡表假设有足够后续块。 |
| L/H | 当前块前/后N64，列0–63/64–127。 |
| K0/K1 | K256的两个K128，坐标0–127/128–255；其它K按实际分块。 |
| step/s/q | 时间索引s与`step`同义：`step=0..2KS−1`、`q=2KS·n+step`；前KS拍L后KS拍H。K256每N四拍、q=4n+step。g/r/s访存命名中的s另指LDS。 |
| kb/half | `kb=step%KS`是K分块索引，不是K偏移或槽号；`half=step//KS`为N64半区0=L、1=H。 |
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
2. **C计算与打包：FP32 → pack → BF16。** MFMA的`record=2*half+packet`是当前N正在累加的N32；共享tile中`packing_events()`的`pack_record/from_previous_n`指定已完成的另一份C及其N归属，**不是该MFMA packet的当前输出record**；K192由独立两拍tile指定pack目标。K256当前Compute的pack使用上一Memory的scale；同一表行新load供后续Compute。`C[n,3]`到下一N的**Compute 0**才pack，不是更新下一N的`C[n+1,0]`。所有MFMA仍为FP8输入、FP32累加，BF16只是pack后的待回写结果。
3. **C回写：上一N的BF16在当前Memory回写，当前FP32继续计算。** 旧FP32被pack消费后才clear；旧BF16被CShuffle写消费后才复用，不能整轮开始就覆盖全部C。不同列处理不同代数据，不是一份数据一拍走完全程。

#### 代码组织与预计算地址

当前生产源码身份集合为 **13项：11个包内Python模块＋[helpers依赖](../../../../src/contrib/flydsl/helpers.py#L1)和[splitk依赖](../../../../src/contrib/flydsl/moe_gemm_splitk.py#L1)**。旧BK128n/K192/K320三个分K模块已物理删除；8x1实现集中于一个主文件，不再转发到三个独立builder。

| 主文件区域 | 当前符号、职责与执行位置 |
|---|---|
| 公共入口 | `_build_moe_gemm2_8x1`做[统一校验](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L24)；[唯一kernel](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L84)统一任务映射、错相前的有效性guard和编译期K分派；[唯一launcher](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L136)保持原ABI、grid `(1, task_num, 1)`、block `(512, 1, 1)`及编译选项。ordinary不是三个kernel依次launch。 |
| 8x1共享原语 | [屏障](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L161)/[priority](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L167)、[A gather](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L191)、[scale加载](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L214)、[pack](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L230)、[CShuffle](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L254)、[输出store](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L284)、[MFMA](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L308)及[BK128交织模板](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L320)均归主文件，不再放在common。 |
| 8x1共享账本/tile/SSA | [pack事件](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L337)、[输出quarter](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L360)、[VM事件账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L368)、`run_8x1_tile`的[单tile实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L911)及[状态保存](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L412)/[恢复](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L428)供BK128族和K320复用。所有N从step0进入同一实现，scale列表在tile内初始化。 |
| BK128：`_emit_k128n_body` | [实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L456)覆盖K256/384/512/640；[连续执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L854)显示prologue、首N地址、q0前错相、N0/N1、固定1N回边、末N及补pack C3。 |
| K192：`_emit_k192_body` | [实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1100)保留整BK192；[执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1337)、[独立VM账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1031)及`run_k192_tile`的[两拍实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1393)仍剥离最后两N，使用独立[状态保存](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1063)/[恢复](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1076)，不是共享BK128尾部。 |
| K320：`_emit_k320_body` | [实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1476)固定128＋192；[执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1727)直接q0前错相，再调用共享tile处理完整首N及后续N，不另写首拍算法。 |

[common](../../../../src/contrib/flydsl/moe_gemm_2stage/common.py#L14)现在只保留通用设备/Host接口、[布局reexport](../../../../src/contrib/flydsl/moe_gemm_2stage/common.py#L58)、偏移检查及`BufferTensor`/`LdsTensor`，不承载8x1专属原语、账本或tile，也不反向导入8x1实现。

各body按“**配置/原语准备区 → 连续Pipeline执行区**”阅读，末尾依次 **prologue → 首N地址 → group1错相屏障 → N0 → N1 → loop/末N → epilogue**。主流程不穿插函数定义、布局构造或`partial`绑定；SSA载体只由`prepare_loop_state()`在N1完成后、确实建立回边时创建。

- 三族`prologue()`均先调用`prepare_views()`集中准备视图，再执行**首B → gather A → 首B落LDS/预取Q1/Q2**。BK128的[B模板/partition及输出视图](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L709)也在准备区；[prologue](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L812)中C读地址pin仍在A读取之后，首N完整B地址仍在prologue返回后pin。
- BK128的[B g→r](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L600)直接返回P槽，[MFMA回调](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L782)按整half/packet片段选择N组；CShuffle写/读合在[输出重排](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L632)，随后显式C r→g。
- 三族均在首N调用前执行`if group == 1: stage_end()`；与group0首Memory末屏障配对，从q0开始错相。`first_tile()`已删除，K192两拍tile也不再接收`group`或保留内部运行时错相判断。
- K320的[prologue](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1683)同样先准备再访存；[B g→r](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1582)/[B r→s](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1599)直接使用共享tile签名，K192段仍为16B＋8B，不新增padding或归约贡献。

启动坐标为`(n, kb, half, lds_slot)`，只为Q1/Q2查询q=0/1；主循环中`b_target(..., ahead=1/3)`分别给出写LDS/新预取目标。共享`widths`缺省为`(128,) * (k // 128)`，K320显式`(128, 192)`，K192独立整192。`short_n_k128`只描述BK128族T<3的等待模式，适用于N0/N1；`read_full_half`则是**当前拍**的B读取方式：BK128首N的q0总读N64，短N首tile余拍也读N64，其余读两个N32。整half读取与两个packet互斥，MFMA仍按两个N32执行；K320始终读两个packet。K192的回调为`read_b_s2r(half, *, packet, addresses)`、`store_b_r2s(half, fragments, *, addresses)`，保留两LDS槽和5地址。

现行循环**固定每回边1个N128**；历史0/2展开调优参数已删除，不是可传的生产选项，历史收据不重标。小N仍保留必要展开，具体零迭代回边见2.8。没有新增Context/Workflow，也不把每条硬件指令封装成独立阶段。

`prepare_b_addresses=None`是**真实生产路径，而非失效回调**：[BK128回调选择](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L800)在K256/K512中因KS=2/4为偶数，槽`(n*KS+kb)&1`不随N翻转；K320的KS=2且16/24KiB两槽固定。K384/K640仅在T≥3时提供4地址回调，T=1/2已按静态N定址，无需跨动态回边携带地址。因此None分支仍执行正常B读/写，不能删除。K192按`n&1`换槽，独立[5地址准备](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1196)在短N也保留，只是不建立动态回边SSA。

wait使用`rocdl.s_waitcnt(lgkmcnt=...)`/`rocdl.s_waitcnt(vmcnt=...)`。q0前错相要求prologue的Q0写入在`lgkmcnt(0)`和双方barrier后完成；首q0 Memory末也显式`lgkmcnt(0)`保护下一拍的跨组Q1读取，K320首M1另保留同样交接保护。共享tile的[短N末拍等待guard](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L973-L975)保留lowering在scale使用前补wait的行为，不前移到Memory。

公共访存的[BufferTensor](../../../../src/contrib/flydsl/moe_gemm_2stage/common.py#L109)/[LdsTensor](../../../../src/contrib/flydsl/moe_gemm_2stage/common.py#L145)不携带额外动态Python字段；Tensor/view描述数据dtype、packet或partition，预计算地址仍是显式SSA值：

- `BufferTensor.load(voffset_bytes=..., soffset_bytes=...)`接收两路**完整32位字节偏移**，相对descriptor根；显式偏移替代、不是累加slice隐含偏移，调用者须已包含slice位置。`soffset`必须wave-uniform，接口不把divergent值伪装成SGPR。只接受静态连续一维32/64/128bit packet，不暗中拆访存；K192/K320仍明确16B＋8B两请求。
- `LdsTensor.load(address_bytes=..., offset_bytes=..., into=..., copy_atom=...)`及`store(...)`接收含allocation/lane/slot的完整LDS字节地址；`offset_bytes`仅为编译期、按元素对齐的固定增量。复杂MMA partition必须保留原layout、copy atom和目标retile，不根据numel改成连续load。完整地址在原Compute尾准备/pin后跨回边传入，`load/store`内部不准备动态偏移、不pin、不插wait。
- 无显式offset的buffer `.load()`仍是普通view语义，**不保证Memory零VALU或两路地址分离**；静态rmem `.load()`不是VMEM。输出与scale继续沿用原`fx.copy`，其`soffset`按**元素**计量，与上面的字节接口不同。

当前[共享调度合同](../../../../tests/contrib/moe/test_all_8x1_schedule.py)、[首拍/内联/准备移位合同](../../../../tests/contrib/moe/test_startup_8x1.py)和[命名合同](../../../../tests/contrib/moe/test_8x1_direction_names.py)直接执行真实tile或对冻结expected做定点结构投影，live不归一化。覆盖每份K贡献仅一次、FP32完成后pack、pack后才覆盖、四scale及四份输出、短N全half/packet和回边地址策略；这些CPU检查不替代GPU异步正确性。最新数值与静态证据见1.6/2.9，旧1129/108配置等仅保留在历史收据中，不作为当前SHA验收。

group0为wave0–3，group1为wave4–7；两组以scalar条件和barrier错相，一组memory时另一组可compute，不保证每cycle完美重叠。所有WG-uniform退出在错相前，尾部平衡barrier。K256的“上一拍scale”不套其它K：K512/K640保留多拍，K192/K320同Compute可pack两份。B的P槽和LDS槽不同；K192 P=16B+8B，K320 P0/P1=24B/16B每lane。K192深稳态另需`n+2<T`；短N裁剪，不能套表。wait按真实指令事件，不算wait后才issue的B，编译器可合并冗余wait。

| K | 归约块坐标 | 每N M/C对 | 每packet MFMA | 每N MFMA/wave | 跨N FP32 | 跨N BF16 | LDS |
|---|---|---:|---|---:|---|---|---|
| 192 | K0=0–191 | 2 | 24 | 96 | C2/C3 | C0/C1 | 64KiB |
| 256 | K0=0–127；K1=128–255 | 4 | 16 | 128 | C3 | C0/C1/C2 | 48KiB |
| 320 | K0=0–127；K1=128–319 | 4 | K0=16；K1=24 | 160 | C2/C3 | C0/C1 | 56KiB |
| 384 | K0=0–127；K1=128–255；K2=256–383 | 6 | 16 | 192 | C3 | C0/C1/C2 | 48KiB |
| 512 | K0=0–127；K1=128–255；K2=256–383；K3=384–511 | 8 | 16 | 256 | C3 | C0/C1/C2 | 48KiB |
| 640 | K0=0–127；K1=128–255；K2=256–383；K3=384–511；K4=512–639 | 10 | 16 | 320 | C3 | C0/C1/C2 | 48KiB |

共同WG=M256×N128/8 waves，A跨N驻留。跨N状态固定为**K192/K320：BF16 C0/C1＋FP32 C2/C3；其余四K：BF16 C0/C1/C2＋FP32 C3**，仍未pack的PTPC scale随对应FP32记录携带。固定原pack，不恢复Memory-pack/delayed/formal：历史后移实验曾延长FP32/scale活跃期，K256/PTPC VGPR176→202、K320 214→232且无稳定收益证据；这不是本次合并的新资源测量，也不由C份数推算VGPR。

<a id="pipeline-k256"></a>

### 2.2 K256：Prologue、N=0、N=1、稳态与尾部

K256/PTPC/固定1N，每N四M/C对，Compute两个16-MFMA packet；不是旧整N128/64-MFMA大stage账本。

#### Prologue（尚无正常MFMA packet）

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | 准备sorted IDs、expert ID；加载并融合routing weight与activation scale | 必要的元数据／LDS同步 | 尚未计算C |
| 首B与A | 先发出`B[0,L,K0] → P0`，再gather两个K128块的A到VGPR | A跨后续N块复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的`P0` | `vmcnt(4)` → `P0 → LDS B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,H,K0] → P1` | `vmcnt(1)` → `lgkmcnt(0)` → barrier | 初始FP32累加区清零；**没有旧`C_bf16`** |

进入N0时LDS只有首B，P0/P1承载后两B预取。

#### N=0：无旧C回写

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,H,K1] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | `vmcnt(3)` | `P1 → B[0,H,K0]` | `B[1,L,K0] → P1` | **Compute 1** | Packet 0：`C[0,0]`累加K1，完成<br>Packet 1：`C[0,1]`累加K1，完成 | `C_bf16[0,0]`<br>与packet 1交织 | `S[0,0]` |
| **Memory 2** | `S[0,2]` | **无** | `B[0,H,K0]` | `vmcnt(3)` | `P0 → B[0,H,K1]` | `B[1,L,K1] → P0` | **Compute 2** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,1]`<br>与packet 0交织 | `S[0,1]` |
| **Memory 3** | `S[0,3]` | **无** | `B[0,H,K1]` | `vmcnt(3)` | `P1 → B[1,L,K0]` | `B[1,H,K0] → P1` | **Compute 3** | Packet 0：`C[0,2]`累加K1，完成<br>Packet 1：`C[0,3]`累加K1，完成 | `C_bf16[0,2]`<br>与packet 1交织 | `S[0,2]` |

N0结束保留BF16 C0/C1/C2，FP32 C3与S3。**首Memory0已纳入共享循环，使用账本vmcnt(3)，不再单独硬编码2。** 首N之前已错相，首M0交接另以lgkmcnt(0)保护Q1。

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

足够长PTPC最后N wait=**7/7/6/6**，末拍无下一B仍保护当前C2 pack；其后vmcnt(0)补C3，按L/H、r0/r1回写末N全部BF16、drain依赖、平衡barrier。短N按实际请求裁剪，不当所有ISA逐条模板。源码：[公共入口](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L24)、[BK128执行流程](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L854)、[共享tile](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L911)与[VM账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L368)。

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
2. 上N Compute1尾prepare_b_addresses预备下N**5个完整lane地址**：当前读、H写16B/8B、下L写16B/8B，跨回边携带；group1在首Memory0之前额外barrier建立错相。首M0末lgkmcnt(0)保证双方Q1写完成后再进入H。
3. 始终跨N两BF16 C0/C1+两FP32 C2/C3及scale，下N Compute0 pack旧C2/C3，Memory1写完才清零高半累加器。
4. 最后两N独立裁剪：倒数第二N不发不存在的n+2/L；最后N不提交/预取越界。足够长PTPC末N**12/12**，后vmcnt(0)补C2/C3、L/H与r0/r1全部回写。
5. T=1无P1下N种子；T<4无动态回边，6/6、10/14、14/14表不机械套单N。源码：[K192 body](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1100)、[执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1337)、[独立VM账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1031)及[两拍tile](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1393)。

<a id="pipeline-k320"></a>

### 2.4 K320：Prologue、N=0、N=1、稳态与尾部

每N四拍**L/K128、L/K192、H/K128、H/K192**，Compute为32/48/32/48 MFMA/wave。K0每lane16B/load×1，K1每lane24B/load×2。槽0 N128×128=16KiB，槽1 N128×192=24KiB，CShuffle16KiB，共56KiB；L/H间距8192/12288B，槽不随N交换。P0=K192、P1=K128，不是LDS编号。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert及行scale准备 | 元数据／LDS同步 | 尚未计算C |
| 首B与A | 首B只发`B[0,L,K0]`，load ×1；A按128＋192 gather到VGPR | 不重复预填H/K0，不padding归约维 | 尚未计算C |
| 首B落LDS | 使用启动首B片段 | `vmcnt(4)` → LDS槽0的L/K0 | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`，load ×2；`B[0,H,K0] → P1`，load ×1 | `vmcnt(1)` → `lgkmcnt(0)` → barrier | FP32累加区清零；无旧BF16 |

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,H,K1] → P0`，load ×2 | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
| **Memory 1** | `S[0,1]` | **无** | `B[0,L,K1]` | **`vmcnt(6)`** | `P1 → B[0,H,K0]` | `B[1,L,K0] → P1`，load ×1 | **Compute 1** | Packet 0：`C[0,0]`累加K1，完成<br>Packet 1：`C[0,1]`累加K1，完成 | **无** | — |
| **Memory 2** | `S[0,2]` | **无** | `B[0,H,K0]` | **`vmcnt(3)`** | `P0 → B[0,H,K1]` | `B[1,L,K1] → P0`，load ×2 | **Compute 2** | Packet 0：`C[0,2]`清零＋K0贡献<br>Packet 1：`C[0,3]`清零＋K0贡献 | `C_bf16[0,0]`与packet 0交织<br>`C_bf16[0,1]`与packet 1交织 | `S[0,0]`、`S[0,1]` |
| **Memory 3** | `S[0,3]` | **无** | `B[0,H,K1]` | **`vmcnt(6)`** | `P1 → B[1,L,K0]` | `B[1,H,K0] → P1`，load ×1 | **Compute 3** | Packet 0：`C[0,2]`累加K1，完成<br>Packet 1：`C[0,3]`累加K1，完成 | **无** | — |

N0结束BF16 C0/C1、FP32 C2/C3及S2/S3。首M0与BK128族均使用账本3；首M0写Q1、首M1写H/K128后均在**交接barrier前lgkmcnt(0)**。

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

只有Compute0/2各pack两份，与packet0/1交织，模板仍是BK128 16-MFMA，不移到BK192拍。正常Memory同K256，B packet0后store前**K0 lgkmcnt4、K1 lgkmcnt6**；首M0和首M1交接前另lgkmcnt0，不能依赖晚Compute尾wait。足够长末N前3拍**7/10/6**，末M3无B提交/pack消费者，账本63且不发VM wait；后vmcnt0补C2/C3全部回写。共享循环独立N0/N1/末N，短N裁去越界。源码：[K320 body](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1476)、[执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1727)及[共享pack事件](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L337)。

<a id="pipeline-k384"></a>

### 2.5 K384：Prologue、N=0、N=1、稳态与尾部

3个K128，每N六拍Q[6n..6n+5]=L/K0、L/K1、L/K2、H/K0、H/K1、H/K2。每Compute两个16-MFMA packet，共32MFMA/wave，B预取每份load×1。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert、routing与activation scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | `B[0,L,K0] → P0`；gather K0/K1/K2的A到VGPR | A跨N复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的P0 | `vmcnt(4)` → LDS槽0的`B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；**`B[0,L,K2] → P1`** | `vmcnt(1)` → `lgkmcnt(0)` → barrier | FP32累加区清零；无旧BF16 |

首M0读Q0、提交Q1、发Q3=H/K0；**Q2是L/K2**。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,H,K0] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
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
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,L,K2] → P1` | `vmcnt(1)` → `lgkmcnt(0)` → barrier | FP32累加区清零；无旧BF16 |

首M0预取Q3=L/K3，H/K0到M1才预取。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,L,K3] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
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

LDS槽`(4n+kb)&1=kb&1`不随N翻转；两个16KiB B槽+CShuffle16KiB=48KiB，P每拍轮换，不是四个B常驻。N0 wait为**3/5/5/5/3/1/1/1**，N1/steady为**5/9/9/9/5/1/1/1**；只有前4个Memory有scale/store，不恢复legacy +4。Compute 0 pack旧C3，Compute 3/4/7分别pack当前C0/C1/C2；C0到K3归约完成后才在独立packet1间隙pack，没有减少贡献。足够长PTPC末N为**5/9/9/9/5/1/0/9**，末拍仍保护C2的scale消费者；最后三个Memory无越界Q[q+3]，后`vmcnt(0)`补C3并完整回写。

<a id="pipeline-k640"></a>

### 2.7 K640：Prologue、N=0、N=1、稳态与尾部

5个K128、每N十拍，先L/K0..K4再H/K0..K4。Compute32MFMA/wave，每N320；**stage4仍L、stage5才H**，scale/旧回写已在前4拍安排。

#### Prologue

| 启动部分 | Memory侧：加载／准备 | Memory侧：等待与提交 | Compute侧／C状态 |
|---|---|---|---|
| 元数据与行scale | sorted IDs、expert、routing与activation scale | 元数据／LDS同步 | 尚未计算C |
| 首B与A | `B[0,L,K0] → P0`；gather K0..K4的A到VGPR | 五段A跨N复用 | 尚未计算C |
| 首B落LDS | 使用刚加载的P0 | `vmcnt(4)` → LDS槽0的`B[0,L,K0]` | 尚未计算C |
| 两个预取种子 | `B[0,L,K1] → P0`；`B[0,L,K2] → P1` | `vmcnt(1)` → `lgkmcnt(0)` → barrier | FP32累加区清零；无旧BF16 |

M0发Q3=L/K3，M1发Q4=L/K4，M2才H/K0。

#### N=0

| **Memory Stage** | **Scale load ×2** | **CShuffle → C_bf16 store ×2** | **B：LDS→VGPR** | **VM等待** | **B：VGPR→LDS提交** | **B：Global→VGPR预取** | **Compute Stage** | **FP32累加** | **Pack生成的BF16结果** | **Pack使用的scale** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Memory 0** | `S[0,0]` | **无** | `B[0,L,K0]` | **`vmcnt(3)`** | `P0 → B[0,L,K1]` | `B[0,L,K3] → P0` | **Compute 0** | Packet 0：`C[0,0]`从0计算K0贡献<br>Packet 1：`C[0,1]`从0计算K0贡献 | **无** | — |
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

LDS槽`(5n+kb)&1`随N翻转，上一N的Compute 9准备4个读/写lane地址跨回边；L/K4与H/K0可在同槽不同半区，不按`step&1`猜LDS。Scale/旧store仅在Memory 0–3；Compute 0 pack旧C3，Compute 4/5/9分别pack当前C0/C1/C2；**Compute 4没有新scale却使用保留的S0**。N0 wait为**3/5/5/5/3/1/1/1/1/1**，N1/steady为**5/9/9/9/5/1/1/1/1/1**；后六拍仍有B/MFMA。足够长PTPC末N为**5/9/9/9/5/1/1/1/0/11**，末拍C2仍有scale消费者，后`vmcnt(0)`补C3完整回写；11不推广到稳态或其它K。

K384/512/640共用[BK128 body](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L456)及其[执行区](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L854)、[共享tile](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L911)、[VM事件账本](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L368)；这些8x1专属实现已不在common。

<a id="pipeline-boundaries"></a>

### 2.8 非PTPC、短N及compact边界

非PTPC权重（per_tensor/per_tensor或per_tensor/ptpc）**pack事件、B FIFO、C回写结构不变**，只是weight scalar在prologue融合行scale，不发逐N S[n,j] VMEM。每份pack由40降24 VALU，不少MFMA。

| K分支 | per-tensor权重：N=0实际逻辑wait | N=1过渡wait | 深稳态wait | 最后N账本（足够长循环） |
|---|---|---|---|---|
| K192 | `2/2` | `6/10` | `10/10` | `8/63` |
| K256 | `1/1/1/1` | `3/5/5/5` | `5/5/5/5` | `5/5/4/63` |
| K320 | `1/2/1/2` | `3/6/5/6` | `5/6/5/6` | `5/6/4/63` |
| K384 | `1/1/1/1/1/1` | `3/5/5/5/3/1` | `3/5/5/5/3/1` | `3/5/5/5/2/63` |
| K512 | `1/1/1/1/1/1/1/1` | `3/5/5/5/3/1/1/1` | `3/5/5/5/3/1/1/1` | `3/5/5/5/3/1/0/63` |
| K640 | `1/1/1/1/1/1/1/1/1/1` | `3/5/5/5/3/1/1/1/1/1` | `3/5/5/5/3/1/1/1/1/1` | `3/5/5/5/3/1/1/1/0/63` |

63是无当前VM消费者的哨兵，不是等63条、不能省LDS；是否发VM wait看实际guard，最后仍vmcnt0。单/双N按真实T生成，不能截深循环末N阈值。

| 路径 | 动态循环前 | 短N／尾部 | 最终补pack |
|---|---|---|---|
| K192独立 | N0/N1独立，按下N/下下N裁剪 | 最后两N独立；T<4必要展开，稳态固定每回边1个N | C2/C3 |
| K256/320/384/512/640共享 | q0前错相，N0从step0进入共享tile，再完整N1 | 最后N独立；T<3不用共享回边，稳态固定每回边1个N并保持pack/FIFO | K320 C2/C3，其余C3 |

共享族在T≥3时仍按`range(2, T-1, 1)`建立回边SSA，**T=3是零迭代回边**；K192在T≥4时按`range(2, T-2, 1)`建立，**T=4同样零迭代**。不能把“没有实际循环迭代”误写成这两种边界也不创建载体。BK128的T=1/2首tile余拍仍保留整half读取/MFMA布局，K320始终用quarter片段；它们是实际短N路径，不是已删除的0/2展开调优维度。

只有q+1<2KS·T才写下一B、q+3<2KS·T才预取；取消越界B不取消当前pack消费者。展开路径可能已被更早严格wait保护，不能强塞尾拍冗余wait，错相/退出barrier仍须平衡。以上102行逐stage和24行prologue按实际两个VM账本及最终源核对：四个BK128族首M0由2改3，五个非K192种子交接增加lgkmcnt(0)，非PTPC对应四个首M0由0改1；其余表内B/scale/pack/输出事件保持。等待表是逻辑消费者预算，具体ISA可能合并或补充等待，不保证逐条相同。

compact M256 full沿用上述同K流水，从M64 metadata任务表读取`[physical_row_begin, expert_id]`及count，保持physical行编号，不改pack/K宽度。**构表→full→M64 tail**同stream，无host count读回；N%256=0时tail1x4，否则default，K192/K320 full的192归约不改变tail既有BK128。见[组合launcher](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py#L68)与[任务表构建](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_compact.py#L132)。

<a id="resources"></a>

### 2.9 最终源码的fresh offline资源

以下30行全部来自[最终源码18配置离线结果](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/result.json)，SHA256：`c24716a4a29df9a5df4431f0b6ad55864ed46f9e8e167c9cb2388e766ecc9daa`。12个ordinary、6个compact full、6个tail及6个构表kernel均由最新13源重新编译，18份构建全部通过实际ELF、文本自重汇编、资源、launcher和VMEM依赖检查；102个实际稳态Memory段零VALU。每行链接本轮after收据，未用旧数值冒充新测；这不是旧/新ISA等价报告，也不是瞬时活跃寄存器、occupancy或性能测量。

| 项目 | 统计口径 |
|---|---|
| 环境 | gfx942/MI308X、FlyDSL0.3.2，离线拓扑输入4XCC/80CU；不由理论峰值推资源。 |
| ordinary | M256×N128、512 threads/8 waves、固定n_loop1、store_cache2、padding128B、非task-table，K192/K320真实归约。 |
| PTPC | weight/activation均PTPC，N2048/TopK8；不是块编号N0/N1。 |
| per-tensor | 双per_tensor，N4096/TopK9；与PTPC的N/TopK不同，不仅量化影响资源。 |
| metadata VGPR/SGPR/AGPR | ELF metadata的三个count；VGPR为每lane32-bit槽、SGPR为wave共享，非WG合计或stage瞬时live值。AGPR的0也来自各kernel。 |
| next_free与accum_offset | descriptor原字段，与metadata分列，不相互替换；next_free_sgpr在ordinary/full/tail为96、task为22，不代表metadata SGPR为这些值。 |
| LDS | group_segment_fixed_size，每WG字节，1KiB=1024B。 |
| 静态ISA/MFMA | 完整kernel机器指令数及其中MFMA条数，含prologue/回边/drain，非源码行或动态执行次数。 |
| 回边ISA | 对ordinary/full的每段回边求end−begin+1再相加，含回跳；当前每回边一个N128。task/tail不套用full的loop审计，填“—”，不是测得0。 |
| private/Vspill/Sspill/scratch | private字节、metadata VGPR/SGPR spill count、scratch指令数；逐kernel实读为0，不从其它组外推。 |

#### ordinary PTPC／N2048／TopK8

| K/配置 | store_cache | kernel | metadata VGPR/SGPR/AGPR | next_free_vgpr/sgpr | accum_offset | LDS KiB | private/Vspill/Sspill/scratch | 静态ISA/MFMA | 回边ISA | 资源证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| K192/N2048/ptpc | 2 | ordinary | 214/36/0 | 214/96 | 216 | 64 | 0/0/0/0 | 2062/480 | 366 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k192_n2048_ptpc/result.json) |
| K256/N2048/ptpc | 2 | ordinary | 184/38/0 | 184/96 | 184 | 48 | 0/0/0/0 | 1993/512 | 433 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k256_n2048_ptpc/result.json) |
| K320/N2048/ptpc | 2 | ordinary | 216/36/0 | 216/96 | 216 | 56 | 0/0/0/0 | 2191/640 | 473 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k320_n2048_ptpc/result.json) |
| K384/N2048/ptpc | 2 | ordinary | 202/38/0 | 202/96 | 204 | 48 | 0/0/0/0 | 2451/768 | 545 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k384_n2048_ptpc/result.json) |
| K512/N2048/ptpc | 2 | ordinary | 220/38/0 | 220/96 | 220 | 48 | 0/0/0/0 | 2870/1024 | 646 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k512_n2048_ptpc/result.json) |
| K640/N2048/ptpc | 2 | ordinary | 244/38/0 | 244/96 | 244 | 48 | 0/0/0/0 | 3333/1280 | 762 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k640_n2048_ptpc/result.json) |

#### ordinary per-tensor／N4096／TopK9

| K/配置 | store_cache | kernel | metadata VGPR/SGPR/AGPR | next_free_vgpr/sgpr | accum_offset | LDS KiB | private/Vspill/Sspill/scratch | 静态ISA/MFMA | 回边ISA | 资源证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| K192/N4096/per_tensor | 2 | ordinary | 176/34/0 | 176/96 | 176 | 64 | 0/0/0/0 | 1670/480 | 293 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k192_n4096_per_tensor/result.json) |
| K256/N4096/per_tensor | 2 | ordinary | 160/34/0 | 169/96 | 160 | 48 | 0/0/0/0 | 1658/512 | 354 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k256_n4096_per_tensor/result.json) |
| K320/N4096/per_tensor | 2 | ordinary | 192/38/0 | 192/96 | 192 | 56 | 0/0/0/0 | 1848/640 | 396 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k320_n4096_per_tensor/result.json) |
| K384/N4096/per_tensor | 2 | ordinary | 180/34/0 | 180/96 | 180 | 48 | 0/0/0/0 | 2127/768 | 472 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k384_n4096_per_tensor/result.json) |
| K512/N4096/per_tensor | 2 | ordinary | 192/34/0 | 192/96 | 192 | 48 | 0/0/0/0 | 2544/1024 | 571 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k512_n4096_per_tensor/result.json) |
| K640/N4096/per_tensor | 2 | ordinary | 216/34/0 | 216/96 | 216 | 48 | 0/0/0/0 | 3020/1280 | 688 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k640_n4096_per_tensor/result.json) |

K256标量metadata VGPR160/next_free169两者保留，不凭差值猜spill/驻留。mixed的activation gather/行scale仍可不同，不套per-tensor资源。

| K | B LDS | CShuffle LDS | 合计 |
|---|---|---:|---:|
| K192 | 2×128×192=49152B | 16384B | 65536B/64KiB |
| K320 | 16384+24576=40960B | 16384B | 57344B/56KiB |
| K256/K384/K512/K640 | 2×128×128=32768B | 16384B | 49152B/48KiB |

sorted IDs启动复用既有LDS，不另驻buffer；增K不把全部B放LDS，仍两槽，完整A常驻VGPR。跨N C份数之外还有A/B/scale/地址等，不能据此线性推VGPR。最新离线102个实际steady Memory段均零VALU，不是无寄存器live或无wait。相对原报告，同形状ordinary PTPC的K256 VGPR176→184、K320214→216；per_tensor的K384176→180、K512188→192、K640212→216，其余ordinary VGPR不变。所有LDS与零spill结论保持，但不由寄存器增减直接推性能。

#### compact full/tail分列（12个kernel）

共同gfx942、TopK4/E256、padding32B、阈值0.6，全部固定n_loop1；mixed=per_tensor weight＋PTPC activation，store_cache按配置逐行列出。顺序构表→full→tail三个launch，不能将三者资源相加当一WG；配置不同也不是单因素比较。full为512threads，tail为256threads。

| K/配置 | store_cache | kernel | metadata VGPR/SGPR/AGPR | next_free_vgpr/sgpr | accum_offset | LDS KiB | private/Vspill/Sspill/scratch | 静态ISA/MFMA | 回边ISA | 资源证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| K192/N768/ptpc | 2 | M256 full | 214/34/0 | 214/96 | 216 | 64 | 0/0/0/0 | 2061/480 | 366 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k192_n768_ptpc_compact/result.json) |
| K192/N768/ptpc | 2 | M64 tail 1x4 | 190/31/0 | 190/96 | 192 | 21 | 0/0/0/0 | 648/96 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k192_n768_ptpc_compact/result.json) |
| K256/N640/mixed | 0 | M256 full | 160/34/0 | 169/96 | 160 | 48 | 0/0/0/0 | 1673/512 | 354 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k256_n640_mixed_compact/result.json) |
| K256/N640/mixed | 0 | M64 tail default | 128/36/0 | 128/96 | 128 | 16 | 0/0/0/0 | 632/128 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k256_n640_mixed_compact/result.json) |
| K320/N768/mixed | 0 | M256 full | 192/35/0 | 192/96 | 192 | 56 | 0/0/0/0 | 1861/640 | 396 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k320_n768_mixed_compact/result.json) |
| K320/N768/mixed | 0 | M64 tail 1x4 | 158/31/0 | 169/96 | 160 | 28 | 0/0/0/0 | 719/160 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k320_n768_mixed_compact/result.json) |
| K384/N640/per_tensor | 2 | M256 full | 186/34/0 | 186/96 | 188 | 48 | 0/0/0/0 | 2129/768 | 472 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k384_n640_per_tensor_compact/result.json) |
| K384/N640/per_tensor | 2 | M64 tail default | 176/36/0 | 176/96 | 176 | 24 | 0/0/0/0 | 781/192 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k384_n640_per_tensor_compact/result.json) |
| K512/N768/per_tensor | 2 | M256 full | 188/34/0 | 188/96 | 188 | 48 | 0/0/0/0 | 2547/1024 | 571 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k512_n768_per_tensor_compact/result.json) |
| K512/N768/per_tensor | 2 | M64 tail 1x4 | 204/31/0 | 257/96 | 204 | 40 | 0/0/0/0 | 852/256 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k512_n768_per_tensor_compact/result.json) |
| K640/N768/mixed | 2 | M256 full | 212/31/0 | 212/96 | 212 | 48 | 0/0/0/0 | 3031/1280 | 688 | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k640_n768_mixed_compact/result.json) |
| K640/N768/mixed | 2 | M64 tail 1x4 | 212/31/0 | 257/96 | 212 | 48 | 0/0/0/0 | 1077/320 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k640_n768_mixed_compact/result.json) |

#### compact构表按六配置分别列出（6个kernel）

构表均256threads；store_cache列仅标识所属编译配置，不据此宣称构表使用full的输出store修饰。六份资源虽数值相同，仍各自列出实际after证据。

| K/配置 | store_cache | kernel | metadata VGPR/SGPR/AGPR | next_free_vgpr/sgpr | accum_offset | LDS KiB | private/Vspill/Sspill/scratch | 静态ISA/MFMA | 回边ISA | 资源证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| K192/N768/ptpc | 2 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k192_n768_ptpc_compact/result.json) |
| K256/N640/mixed | 0 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k256_n640_mixed_compact/result.json) |
| K320/N768/mixed | 0 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k320_n768_mixed_compact/result.json) |
| K384/N640/per_tensor | 2 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k384_n640_per_tensor_compact/result.json) |
| K512/N768/per_tensor | 2 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k512_n768_per_tensor_compact/result.json) |
| K640/N768/mixed | 2 | 构表 task | 20/28/0 | 20/22 | 20 | 4 | 0/0/0/0 | 544/0 | — | [after](../../../../tests/contrib/moe/results/readme_latest_20260911/resources/k640_n768_mixed_compact/result.json) |

compact K256 full的metadata/next_free_vgpr为160/169，K320 tail为158/169，K512/640 tail为204/257、212/257，均原样保留。next_free257不直接等于每lane可分配257或spill，须结合完整descriptor/架构，不能凭字段差值推驻留。30行仅是所列源码/编译器/shape/配置的静态事实，不是任意N、mixed或compact的资源上限，也不据此推occupancy、MFMA busy或wall-time性能。

<a id="optimizations"></a>

## 三、重要优化

### 3.1 确实现有的机制

| 优化 | 当前机制和限制 |
|---|---|
| 8-wave反相 | 512线程/两4-wave组，首q0前group1额外barrier与group0首Memory末屏障配对，memory优先级0/compute3；一组VMEM/LDS/CShuffle、另一组MFMA+原pack。Q0预先lgkmcnt0就绪，16KiB scratch错相复用，不能半组提前退出或宣称每cycle完全重叠。 |
| 两P槽消费FIFO | 启动LDS Q0、P中Q1/Q2；正常读Q[q]、提交Q[q+1]、发Q[q+3]，Q[q+2]在另P，提前跨计算窗。wait含已发store，同时约束B/scale消费者，非统一放宽9。 |
| A驻留/真实K | 每wave两M16完整A只gather一次跨N；行scale提前融合。K192整192、K320 128+192避免补零和小stage，BK128多块仍两B LDS槽，不按K线性推寄存器。 |
| 原pack/scalar FMA/perm | 每N四N32 super-record各含r0/r1，已完成record的pack与另一record的MFMA packet交织，不把pack目标当本packet累加目标。跨N时K192/K320保留BF16 C0/C1＋FP32 C2/C3，其余四K保留BF16 C0/C1/C2＋FP32 C3，不保留两套完整FP32 C。PTPC每份16weight FMA+16行scale/bias FMA+8perm=40VALU，scalar权重先融合后16FMA+8perm=24；用v_fma_f32/v_perm_b32，禁packed-fp32-ops。pack时点保持，首拍等待变化见二章；pack不是store，VMEM/DS/store不混入MFMA train，模板的最多3条独立VALU不是FP8实测时序保证。 |
| CShuffle plane xor/read2 | 128bit写，source-group低位移到相隔2KiB plane，row/pair xor分bank；仅配同一输出行两8B读，让后端read2st64_b64避免跨行v_mov。B packet0先读→partial LDS wait→store→packet1读，BK128用4、192用6。模型payload、ISA、PMC不同，源码不保证所有shape零bank。 |
| Nloop1（固定） | 现行每回边固定一个N128，剥离N0/N1/尾部，回边只steady；保存B carry、未pack C/scale、BF16和必要地址。旧0（全展开）/2（每回边两N）只解释历史调优收据，相关参数已从生产接口删除、不能再传；短N必要展开仍保留，较小代码体积不自动等于更快。 |
| 地址提前准备 | 均匀N/K/wave偏移走scalar buffer offset，lane和C读指针提前；K192保留5完整地址，K384/640仅T≥3时携带4地址，偶数KS及共享短N允许None回调，仍正常访存。B模板/partition集中prepare_views，完整地址在首N前/上一N Compute尾pin；最新18配置的102个steady Memory段无VALU，不意味Compute无地址或任意配置相同。copy offset须区分元素/字节。 |
| compact平衡 | 每expert单连续run，顺序不必排序；保留physical行。原full F、实际CU U、r=F mod U，仅0<5r<3U把全局full后缀r个M256拆4r个M64，r0不拆；counts为full×256、tail×64，容量包含拆分，device guard、无host读回。launch开销可能抵消padding收益，逐shape选。 |
| cache/64-thread sum | builder缓存包含设备/静态参数，task缓存也含CU/阈值。store_cache2为aux（源码SLC），ISA修饰以产物为准不猜命中。sorted_sum每token64线程，先TopK位置、128bit BF16读、FP32按TopK累加转BF16，stride含padding；inverse仅valid前缀并查token/topk边界，避免未写区域。 |

不把旧pure/K128/BK64、Memory-pack/delayed/formal或direct-to-LDS实验当当前实现，早期实验原字节仍归档；收益仅认对应run，不以指令数变少代替wall-time证据。









