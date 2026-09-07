# Paged MHA / 单 wave SWA

**换机器前先读：[CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md)**。包含离线Git历史迁移、环境搭建、
分架构测试顺序、原始计时协议、待测/待修事项和开发规则。

> **最新：BF16/gfx942 spill修复**。96个D/V/page/CNC/LSE/scale编译组合全部零scratch/零VGPR
> spill；默认D128/D192也零SGPR spill，VGPR为246/252。最终整合回归205通过（含24项原版逐位对照）。
> 低负载监测已触发BF16/FP8实测：D192/page64延迟−15.44%，D128/page64基本持平；
> D192/page32仍+10.95%，不宣称全面恢复。PTL Disabled诊断不冒充独占原文基线。
> 详见 [BF16_SPILL_FIX.md](BF16_SPILL_FIX.md)、[最新状态](results/bf16_spill_status.json)。

> **2026-09-07 性能口径修正**：下方26组PTL Disabled结果不是原文250T/410T验收。
> 已补原文44项命名workload、精确timer/输入协议、PTL/硬件基线检查和无干扰保护。
> 上一轮复测因sglang占满GPU而在PTL修改前终止。最新要求已恢复监测/测试；**尚未证明绝对吞吐恢复**。
> 详见 [PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md) 和
> [上一轮待复测状态](results/performance_followup_status.json)。上一轮112项CPU通过；最新BF16修改见上方。

四个显式 backend，共用测试、FP32 reference、性能和资源审计入口。重构基于
**`origin/main` (`ebc5334`)作为重构起点**；指定分支来源固定为 `lc/luocheng/mha_swa`
的 `23cc6d1e`。保留原计算流水和数值约定，不通过其他 attention backend 掩盖失败。

## 文件与验证结论

| 文件 | 实现 |
|---|---|
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | native gfx942 FNUZ FP8，8-wave，BM256/BN64；LDS / register 两条原路径 |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | origin/main BF16 prefill，8-wave persistent，BM256/BN32，K经LDS、V留寄存器 |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | native gfx950 BF16，8-wave，BM256/BN64；保留可选persistent、SWA/sink |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | **一个wave64/CTA**，QT16/32，BN16/32/64；gfx950原路径 + gfx942原生适配 |
| [test_mha_pa.py](test_mha_pa.py)、[test_mha_pa_swa.py](test_mha_pa_swa.py) | 参数化功能测试及统一CLI |
| [_testing.py](_testing.py)、[_references.py](_references.py)、[_runner.py](_runner.py) | 数据/独立oracle、显式AITER、计时/资源协议 |
| [_dsl.py](_dsl.py) | 局部layout/descriptor/wait兼容函数，不修改FlyDSL全局API |
| [validate_preservation.py](validate_preservation.py) | hash验证的原Git源码，同输入ABBA / gfx950双版本交叉编译 |
| [_perf_cases.py](_perf_cases.py)、[test_perf_cases.py](test_perf_cases.py) | 原文44项性能case、基线与CPU回归 |
| [reproduce_baselines.py](reproduce_baselines.py)、[_hardware.py](_hardware.py) | 原版/当前版复现，显式可恢复PTL与空闲检查 |
| [compile_bf16_942.py](compile_bf16_942.py)、[test_bf16_spills.py](test_bf16_spills.py) | metadata-only资源编译、spill回归及原版逐位对照 |
| [watch_performance.py](watch_performance.py)、[recheck_performance.py](recheck_performance.py) | 低利用率触发的串行性能队列；驻留worker仅诊断 |
| [summarize_results.py](summarize_results.py) | 合并资源、保留ISA与历史来源、汇总JUnit和性能 |

本机 MI308X / gfx942 **上一轮重构版本**验证（不是本次扩展性能工具的GPU验收）：

- **完整统一功能回归：865 passed / 1298 skipped / 0 failed**，2163项，1733.02秒。
  PA为403通过/869跳过，SWA为462通过/429跳过；其中36项实际AITER CK参考检查通过。
- 跳过原因：1047项需要gfx950硬件，44项AITER 5D无匹配kernel，其余207项为不适用能力
  或比较组合。没有用 `xfail`、`equal_nan` 或吞掉异常来计为成功。
- **12组原版/重构版 native 对照输出逐位一致**。5轮ABBA attention延迟变化
  **−0.431%～+0.110%**；所有组原版前后采样漂移最大0.249%。仅说明当时相同新环境/Disabled
  策略下的相对性能，没有证明文档绝对吞吐已复现。
- **旧26组冒烟性能矩阵**完成；每个计时候选先经过FP32和重复一致性检查。不是新44项文档矩阵。
- **88组资源审计**（gfx942与gfx950目标）；另40组gfx950原版/重构版双编译完成。
- **没有gfx950硬件**：其功能case已整理、可收集，原生正确性和性能没有重测。

证据：[results/functional_final.log](results/functional_final.log)、
[results/functional_final.xml](results/functional_final.xml)、[results/summary.json](results/summary.json)。
逐项来源与覆盖对应见 [COVERAGE.md](COVERAGE.md)。较早的失败/局部运行日志保留为排障记录，
不是最终回归结论。
早期重构收尾的BF16942仅做空白/注释清理，已校验当时测量版本的SHA及可执行AST一致，并补跑原始case；
记录见 [results/final_cleanup.json](results/final_cleanup.json)。
后续spill修复是语义/调度修改，当前证据见上方spill报告，不由该旧cleanup记录覆盖。

## 公共接口

三个full MHA工厂的参数顺序、默认值一致：

```python
kernel = module.PagedAttention(
    num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, page_size,
    is_causal, quant_query_mode="per-token", key_layout="vectorized",
    window_left=-1, has_sink=False, memory_mode="lds", persistent=None,
)

out = kernel(
    Q, K, V, cu_seqlens_q, cu_seqlens_k, kv_indptr, kv_page_indices,
    max_seqlen_q, max_seqlen_k, causal, q_descale, k_descale, v_descale,
    kv_last_page_lens, out=None, sink_ptr=None, stream=None,
    return_lse=False, lse=None, softmax_scale=None,
)
```

其中 `memory_mode`、`persistent` 以及调用端的 `return_lse`、`lse`、`softmax_scale`
是keyword-only参数。

SWA工厂保留独立调优参数 `block_n`、`query_tile`，默认 `is_causal=True`、
`window_left=128`；**调用接口与full MHA一致**。W<=16默认QT16/BN16，其余默认QT32/BN32。
QT32仍是单wave内的两个16行子tile，不是两个wave。

- `persistent=None` 表示原实现默认：FP8为False，BF16942为True，BF16950为False。
  不支持的组合显式报错，不自动更换实现。
- `return_lse=True` 返回 `(out, lse)`；否则只返回 `out`。即使不返回LSE，传入的 `lse=`
  仍会被写入。O为BF16 `[Tq,Hq,Dv]`，LSE为连续FP32 `[Tq,Hq]`，使用自然对数。
- `out=`、`lse=`允许调用方预分配。流必须与输入在同一设备；先预热再捕获graph。
  输出不可与输入或另一输出重叠。
- K/V使用SHUFFLE-5D：设 `X=16/element_size`，K为`[Np,Hkv,Dq/X,page,X]`，
  V为`[Np,Hkv,page/X,Dv,X]`。Q为`[Tq,Hq,Dq]`；要求Hq是Hkv的正整数倍。
- metadata为同GPU连续int32；页表可随机、重复、跨请求共享物理页。
  KV长度由`kv_indptr`和`kv_last_page_lens`给出；`cu_seqlens_k`可为None，不参与5D查页。
- Q descale为scalar或per-token/head；K/V descale各一个FP32 scalar。均须有限且严格为正。
  BF16942的per-tensor模式要求scalar Q scale；统一使用per-token模式可接受两种形状。
- metadata值与host最大长度须一致并在界内；热路径不拷回CPU验证值。物理缓存字节跨度受
  signed-int32 addressing限制。不是完整防御性API，调用方负责合法索引、scale和非重叠存储。

### 能力范围

| 能力 | FP8 942 | BF16 942 | BF16 950 | 单wave BF16 SWA |
|---|---|---|---|---|
| Q/K/V类型 | E4M3FNUZ | BF16 | BF16 | BF16 |
| 架构 | gfx942 | gfx942 | gfx950 | gfx942 / gfx950 |
| Dq / Dv | 128,192 / 128 | 128,192 / 128,192 | 128,192 / 128 | 128,192 / 128 |
| page size | 64 | 32,64,128 | 64 | 64 |
| C / NC full | 是 | 是 | 是 | 仅causal SWA |
| active空KV / causal KV<Q | 是 | **不支持** | 是 | 是 |
| 非连续Q/O | 不支持 | 不支持 | head dim连续、非重叠 | head dim连续、非重叠 |
| LSE / custom scale | 是 | 是 | 是 | 是 |
| SWA / sink | 不支持 | 不支持 | 保留，两种调度均有case | 是 |
| persistent | 不支持 | 始终开启 | 可选 | 不支持 |
| 热路径辅助GPU工作 | 无 | 每次counter分配/清零/seed | 无；预热时分配header | 无 |

BF16942保留原有效输入限制：**每个active sequence必须KV>0；causal时每个sequence须KV>=Q**，
不只是全batch最大值。该实现的尾页测试使用原来的**零padding契约**；其他三个实现检查
NaN-poison尾页。BF16942的BF16概率/输出转换沿用原`round-half-up` helper，不悄悄替换为RNE。

SWA闭区间：`0 <= (KV-Q+query_row)-key <= window_left`，W128最多129个key。
sink是每head一个未乘QK scale的自然logit，对应零value虚拟key，只加入一次分母；允许
有限值和−inf，不支持NaN/+inf。空KV/全mask无有效sink时O=0、LSE=−inf。

BF16950 persistent每device/stream/grid复用8-byte header，最后一个CTA自动归位。
分别预热的capture stream才有独立header；共享同一capture/header的graph不能并发replay。
BF16942使用独立的per-call counter，seed已改为device fill以支持graph，不改变ticket算法。

## 一次运行全部功能或性能

从仓库根目录运行，Python环境须包含ROCm PyTorch、pytest、pyhip及**FlyDSL 0.3.1**：

```bash
# full MHA + SWA，所有可用backend，gfx950不存在时明确skip
python tests/flydsl/mha/test_mha_pa.py --mode functional --suite all

# 原始文档44项workload；不支持的backend组合显式报告（包括长序列/H3）
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all

# CPU-only列举case，不查询/执行GPU；quick仅用于旧的小矩阵冒烟
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all --list-cases
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all --matrix quick

# 单独SWA，增加auto之外的6种QT/BN候选
python tests/flydsl/mha/test_mha_pa_swa.py --mode performance --tiles

# 原分支gather+linear口径：Q16K，KV32K/64K/128K，D128/192，共六组
python tests/flydsl/mha/test_mha_pa_swa.py --mode performance --gather-linear --aiter required

# 单个backend/形状；--pytest-args之后的参数原样交给pytest
python tests/flydsl/mha/test_mha_pa.py --mode functional --backend bf16_942 --pytest-args -x --tb=short
python tests/flydsl/mha/test_mha_pa.py --mode performance --backend fp8_942 --case fp8_native_410t

# 资源审计；交叉编译只生成ISA，不执行其他架构代码
python tests/flydsl/mha/test_mha_pa.py --mode audit --suite all --cross-compile

# 同策略原版对照 / gfx950双版本编译 / 保存证据汇总
python tests/flydsl/mha/validate_preservation.py --mode native
python tests/flydsl/mha/validate_preservation.py --mode cross-compile
# summarize_results.py仅用于早期历史汇总，依赖旧临时dump；新机参阅CONTEXT_HANDOFF
```

`--mode all` 顺序执行functional、audit、performance；若指定 `--output`，资源和性能
分别写入带`.resources` / `.performance`后缀的结果，互不覆盖。`--aiter off`仅关闭可选
性能reference，不关闭FP32正确性门槛；`--aiter required`要求所有适用的性能参考可用。
功能模式的AITER检查为pytest用例，可用 `--pytest-args -k ...`筛选。
`--strict-resources`额外要求所有specialization零scratch/spill；**部分BF16942 LSE仍有2个SGPR
lane转存，且950 persistent可能有scratch，不应期待该选项全通过**。原版对照需要本地Git仍能读取上述固定提交。

性能默认使用documented矩阵；`--case`接受名称/glob。包含D128/192 full/causal32K、batch4、
H1长序列、H3和SWA KV/window/query sweeps。documented模式下shape参数只过滤原case；
自定义输入须加`--matrix custom`。**不再自动缩小Q或更换page**。原形状/历史计时定义见
[PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)。

### 本次隔离环境

- MI308X gfx942，80CU；Python3.11.11，PyTorch2.12.1+rocm7.2，HIP7.2.53211，FlyDSL0.3.1。
- 用户原Python环境未升级；授权的新环境位于`/tmp/pyhip-mha-py311`，复用现有torch/pytest，
  pyhip绑定到本仓库src。临时环境可能被系统清理，长期使用应在自己的venv安装匹配版本。
- AITER checkout `83faabaa4bf077713c0c71546a8935313c01640b`，无源码修改。
  当前系统C++ runtime太旧，共享AITER模块又是Python3.10；本次仅在临时目录提供libstdc++，
  用`AITER_JIT_DIR`建立独立Python3.11模块，未覆盖共享cache。
- 本机复现AITER环境前缀（所有设置仅影响该进程）：

```bash
PATH=/raid/users/xisun/luocheng/.env/bin:/tmp/pyhip-mha-py311/bin:$PATH \
LD_LIBRARY_PATH=/tmp/mha_cpp_runtime/lib \
AITER_JIT_DIR=/tmp/mha_aiter_py311 GPU_ARCHS=gfx942 MAX_JOBS=4 \
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
/tmp/pyhip-mha-py311/bin/python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all
```

以下旧数据使用 **PTL Disabled、原有auto-DPM**。新CLI仅在显式`--ptl`时执行临时policy
实验并finally恢复；当前GPU忙碌，尚未实际切换。kernel不会自行修改硬件。

## 上一轮PTL Disabled实测（非文档基线验收）

协议：预分配O、无LSE，先完整chunked FP32 reference及3次逐位重复；共同预热1200次，
每轮20 warmup/100 iterations，5轮交替正反顺序。GPU event去第一条后计算1.5IQR过滤均值，
最后取5轮中位数。保存原始event时间、保留索引、dispatch名、输入参数和source hash。

`attention_us`为主kernel时间；`total_gpu_us`另计counter初始化等辅助GPU工作，**并非CPU
端到端wall time**。TFLOPS基于真正可见的attention FLOPs，不把causal/SWA掩掉的区域计入有效工作。

### gfx942 full MHA：Q10240/KV2560，noncausal

| backend | Dq | attention µs | total GPU µs | 有效TFLOPS | AITER CK linear µs |
|---|---:|---:|---:|---:|---:|
| FP8 LDS | 128 | 1103.834 | 1103.834 | 194.548 | 不适用BF16 comparator |
| FP8 LDS | 192 | 1232.845 | 1232.845 | 217.737 | 不适用BF16 comparator |
| FP8 register | 128 | 1477.585 | 1477.585 | 145.337 | 不适用BF16 comparator |
| FP8 register | 192 | 1623.020 | 1623.020 | 165.393 | 不适用BF16 comparator |
| BF16 original pipeline | 128 | 1843.585 | 1855.731 | 116.484 | 1889.654 |
| BF16 original pipeline | 192 | 2759.029 | 2771.184 | 97.293 | 2189.358 |

### gfx942 单wave SWA：Q16384/KV131072/W128/sink

| Dq | QT/BN | 本实现 µs | 有效TFLOPS | AITER CK linear µs |
|---:|---|---:|---:|---:|
| 128 | 32/32 | 316.947 | 54.628 | 610.255 |
| 192 | 32/32 | 364.109 | 59.440 | 686.759 |

完整26组含causal、KV2583和每轮数据：[results/performance_all.json](results/performance_all.json)。
这是新gfx942 SWA移植的数据，没有旧gfx942 SWA性能基线；不把不同GPU的数字当移植加速比。

**AITER比较含义：**

- `aiter_5d`使用完全相同5D缓存/metadata，当前page64 BF16及FNUZ FP8均返回
  **`no matching kernel found`**，没有可报告的5D时间；原错误保存在每条记录中。
- `aiter_ck_linear_prepared`显式调用CK `mha_varlen_fwd`，提前重建相同逻辑KV，
  gather/转换在计时之外；不是5D端到端时间，也不是缺失5D路径的fallback。
- 不使用当前AITER public linear router：它先导入与FlyDSL0.3.1不兼容的辅助模块。
  950的OPUS比较也使用显式入口，缺包/缺符号会报告不可用，不把其他kernel标成OPUS。
- 所有已计时reference都必须通过同样的FP32正确性门槛。缺依赖不影响主kernel计时，
  **数值失败仍终止该运行**。

### SWA：gather + linear 对照

[test_mha_pa_swa.py](test_mha_pa_swa.py) 新增 `--gather-linear`，参考固定原分支 `23cc6d1e`
的 `test_swa_aiter_production_performance` 与 `benchmark_comparison_candidates`，不替换生产SWA。

| 输出候选 | 实际计时范围 |
|---|---|
| `swa_direct` | 单wave直接读SHUFFLE-5D cache |
| `gather` | 原式Triton gather，完整逻辑KV → 预分配THD K/V |
| `aiter_ck_linear_prepared` | 已gather好的K/V，只运行CK `mha_varlen_fwd` |
| `gather_aiter_ck_linear` | **每次重新gather，再运行同一CK linear**，一对GPU event包住整个调用 |

- 默认Q16384、KV32768/65536/131072、H16/HK1、W128/sink、D128/192，共六组。
  可用`--case`选择现有文档SWA case，或`--matrix custom --q ... --kv ...`自定义。
  `--list-cases --gather-linear`只输出CPU计划；[已保存计划](results/swa_gather_linear_plan.json)。
- 计时沿用原gather对照的**20 warmup、100样本、5轮**，每轮/每样本轮换候选顺序，先取样本
  中位数再取轮次中位数。它与普通`--mode performance`的profiler/IQR口径不同；该模式拒绝
  `--timer profiler`、多buffer和`--require-baseline`，不会把新event耗时当原profiler验收数据。
- `event_interval_us`包含对应调用的GPU工作及dispatch间隙；**不把gather-only与linear-only
  两个独立计时相加冒充总耗时**。`tflops`对gather-only为null，其余按相同有效attention FLOPs计算。
  输出保留每轮全部样本、dispatch列表、linear workspace字节数，以及相对gather+linear的加速比。
- [_gather.py](_gather.py) 保留原来每token一个program、4warps、QK block256/V block128的
  gather方式，并扩展KV-head维度；从实际5D cache读取，**不只取SWA可见后缀**。slot mapping、
  workspace及输出在计时前准备；该benchmark固定metadata，每次重新读缓存内容。
- 对照共用统一case输入，不宣称随机张量、环境或耗时与原历史运行逐项相同。原128K
  `mha_batch_prefill` linear路径曾fault，本次使用原文推荐的varlen路径；在当前AITER中显式调用
  CK入口，避开会导入不兼容FlyDSL模块的public router。无静默后端替换。
- 默认`--aiter auto`可报告依赖不可用并保留direct测量，但`comparison_complete=false`且无加速比；
  `--aiter required`保证缺参考立即失败。所有实际运行的路径先过严格FP32/重复逐位检查。
- 仍遵守GPU隔离/PTL规则；`--allow-contention`只用于非独占诊断，不代表正式硬件基线。

本机已在GPU低利用率触发后跑完六组对照；MI308X/gfx942、PTL Disabled，**非独占诊断**。
以下均为GPU event interval，不与上方旧profiler attention-only数字直接作比。每组所有路径
均通过FP32和重复逐位检查，完整记录含12,000个原始候选样本：
[results/swa_gather_linear_performance.json](results/swa_gather_linear_performance.json)。

| Dq | KV | direct SWA µs | gather µs | prepared CK linear µs | gather+CK linear µs | direct相对总路径加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32768 | 337.521 | 144.230 | 687.002 | 837.364 | 2.481× |
| 128 | 65536 | 338.761 | 256.071 | 647.162 | 920.309 | 2.717× |
| 128 | 131072 | 322.381 | 461.492 | 657.183 | 1118.650 | 3.470× |
| 192 | 32768 | 370.541 | 132.241 | 794.683 | 928.844 | 2.507× |
| 192 | 65536 | 370.601 | 270.931 | 733.463 | 1015.049 | 2.739× |
| 192 | 131072 | 372.561 | 471.952 | 748.883 | 1232.895 | 3.309× |

本次只修改测试/比较工具，四个生产kernel均未改。CPU契约回归67通过，native gather4通过，
gather+linear及原有CK回归40通过；gfx950和缺失5D参考明确跳过，不计为通过。

### 重构前后同环境ABBA

原源码从固定Git提交读取并校验SHA256，使用同一输入、设备、FlyDSL0.3.1、PTL Disabled。
NC形状Q10240/KV2560；C形状Q=KV1024；均H16/HK1/V128/page64。

| backend | Dq | NC延迟变化 | C延迟变化 | 输出 |
|---|---:|---:|---:|---|
| FP8 LDS | 128 | +0.015% | −0.045% | bit-exact |
| FP8 LDS | 192 | +0.001% | +0.039% | bit-exact |
| FP8 register | 128 | +0.001% | +0.110% | bit-exact |
| FP8 register | 192 | +0.021% | −0.369% | bit-exact |
| BF16 942 | 128 | −0.032% | −0.431% | bit-exact |
| BF16 942 | 192 | −0.034% | −0.223% | bit-exact |

变化是每组5轮paired ratio的中位数，不是分别取最快值。小差值不作为优化收益宣传。
原始ABBA样本、total GPU差值：[results/preservation_942.json](results/preservation_942.json)。

### gfx950：仅保留历史数字，不冒充本次验证

以下来自指定分支保存的MI350X测量，**没有在本次机器重测**。不同PyTorch/compiler/PTL状态，
不得和上面gfx942数值做直接比值。

| 原阶段/场景 | D128 µs | D192 µs | 原始证据 |
|---|---:|---:|---|
| 8-wave tile-refactor阶段，Q10240/KV2583 NC | 258.133 | 296.550 | [results/historical/gfx950_pa_tile_refactor.json](results/historical/gfx950_pa_tile_refactor.json) |
| 同阶段8-wave，Q=KV32768 C | 4448.558 | 5282.314 | 同上；不是当前persistent性能 |
| 分支最终单wave，Q16384/KV131072/W128/sink | 72.385 | 85.471 | [results/historical/gfx950_swa_original.json](results/historical/gfx950_swa_original.json) |
| 同次分支最终8-wave static，SWA同形状 | 100.603 | 113.225 | 同上，源hash与请求来源匹配 |

本次40组gfx950双版本编译：VGPR+AGPR、LDS、scratch、VGPR spill数均与原版一致；部分
SGPR数/编译器wait不同，**不声称ISA逐条一致或原生性能已相当**。
原始记录见 [results/preservation_950_compile.json](results/preservation_950_compile.json)，
80份双版本ISA的本地副本索引见 [results/preservation_950_retained.json](results/preservation_950_retained.json)。

## 资源

下表BF16942是修改前数据，保留用于对照；**当前资源以[spill修复报告](BF16_SPILL_FIX.md)为准**。

FlyDSL0.3.1，V128/page64，无LSE；full表为noncausal；SWA为W128/QT32/BN32/sink。
寄存器取ISA metadata，非profiler推算。Private为每thread scratch bytes，spill是编译器计数。

| backend / target | Dq | VGPR | AGPR | SGPR | LDS bytes | Private bytes | V/S spill |
|---|---:|---:|---:|---:|---:|---:|---|
| FP8 LDS /942 | 128 | 200 | 0 | 56 | 33280 | 0 | 0/0 |
| FP8 LDS /942 | 192 | 220 | 0 | 56 | 41600 | 0 | 0/0 |
| FP8 register /942 | 128 | 224 | 0 | 54 | 0 | 0 | 0/0 |
| FP8 register /942 | 192 | 244 | 0 | 55 | 0 | 0 | 0/0 |
| BF16 /942 | 128 | 256 | 0 | 106 | 16384 | 104 | 25/2 |
| BF16 /942 | 192 | 256 | 0 | 106 | 24576 | 176 | 43/76 |
| BF16 static /950 | 128 | 226 | 0 | 52 | 99840 | 0 | 0/0 |
| BF16 static /950 | 192 | 256 | 0 | 57 | 149760 | 0 | 0/0 |
| BF16 persistent /950 | 128 | 232 | 0 | 80 | 99844 | 0 | 0/0 |
| BF16 persistent /950 | 192 | 256 | 0 | 86 | 149764 | 12 | 2/0 |
| 单wave /942 | 128 | 214 | 0 | 36 | 0 | 0 | 0/0 |
| 单wave /942 | 192 | 244 | 0 | 38 | 0 | 0 | 0/0 |
| 单wave /950 | 128 | 213 | 0 | 32 | 0 | 0 | 0/0 |
| 单wave /950 | 192 | 238 | 0 | 36 | 0 | 0 | 0/0 |

- BF16942原pipeline本就spill；独立原版D128编译同为Private104/Vspill25/Sspill2。
  保留该流水不等于零spill，不能套用FP8的零spill结论。
- SWA所有48种已编译组合LDS/scratch/spill均0，但部分有AGPR。比如942 D192 QT32/BN64
  有LSE时raw `vgpr_count=387`，实际 **256 VGPR + 131 AGPR**；默认D192 QT32/BN32
  有LSE也使用4AGPR。`零spill`不等于`零AGPR`。
- gfx950这些数据都是compile-only。88组完整metadata、opcode计数、来源、ISA hash及保存的
  汇编链接见 [results/resources_all.json](results/resources_all.json)。

## 流水伪代码

以下表达源码依赖与调度意图；不是声称编译器在所有case都按相同cycle发射。

### FP8 gfx942：BM256/BN64，8 stages

```text
CTA = 8 wave64，两个4-wave group错开一个stage
prologue:
    load Q；SMEM预取page0/1/2
    cooperative VMEM → packed VGPR → K双槽/V双槽LDS（register模式直接load）
    QK(page0) low/high → mask → initial maximum，O=0，sum=0
    发布K(page2)，prime page3标量lookahead

for KV tile t (两phase ping/pong展开):
    S0: K(t).lo LDS read first；request page(t+3)
        issue V(t) then K(t+2) VMEM；sole final LGKM wait；原stage barrier
    S1: QK(t).lo + exp(previous)[0:24]
    S2: K(t).hi read；wait只覆盖较旧V；publish V(t)
    S3: QK(t).hi + remaining exp；row sum；pack FP8 P(t-1)
    S4: V(t-1).lo read；wait K(t+2)并复用已退休K槽
    S5: PV low + current row max/center；lazy slack=6 log2单位
        noncausal主循环的max shuffle与wait之间保留两条独立PV MFMA调度提示
    S6: V(t-1).hi read，保留必要wait/barrier
    S7: PV high + remaining center；必要时同时rescale O/sum

drain last tile：mask/exp/sum/PV，清零无效V字节以避免0*NaN
normalize O，RNE软件BF16 pack
LDS模式复用32KiB为两次64列C-shuffle；register模式直接store
optional LSE = ln(sum) + maximum * ln(2)
```

FP8使用native32×32×16 MFMA。S0额外page-ID lookahead和S5调度提示来自原优化源码，
本次没有改数值表达式、stage barrier或phase顺序。

### BF16 gfx950：direct buffer→LDS，BM256/BN64

```text
prologue:
    Q DMA到与V alias的LDS区；读Q进寄存器后才复用为V
    K双槽 / V双槽，scalar page-ID lookahead
    QK(page0) low/high，初始化max/sum/可选sink
loop:
    S0/S1: K(t).lo read → QK low，与exp(t-1)交织
    S2/S3: K(t).hi read → QK high；rolling VM/LGKM wait；sum、BF16 P
    S4/S5: V(t-1).lo read + K(t+2) DMA → PV low + max/center
    S6/S7: V(t-1).hi read → PV high + center/rescale，lazy slack=8
drain: remaining exp/PV；reverse和epilogue都保留NaN V-tail mask
normalize；native BF16 pack/permlane；b128 global stores（没有C-shuffle）

static: large full-causal grid配对首尾query block，镜像部分reverse遍历KV
persistent: ticket映射到相同任务；最后一个CTA复位stream-local header
SWA: 只遍历query tile可见KV并集；满足原gate才跳过完全不可见wave的MFMA
```

使用32×32×16 BF16 MFMA。原gate为B1、任务数>=1024，D128 W<=64 / D192 W<=128；
SWA不做full-causal首尾配对。所有路径保留原同步边界，不用减少barrier来实现重构。

### BF16 gfx942：原BN32 persistent prefill

```text
each persistent CTA:
    ticket → (batch, head, BM256 query block)
    load Q，构造BN32 K-row permutation，K/O共用LDS union
    prime kv_step(-3),(-2),(-1)，K双buffer与page-ID lookahead
    for full-valid page pairs, then masked page pairs:
        publish下一K到LDS，prefetch未来K到packed registers
        QK当前tile（native32×32×8 BF16）；并行发起V→register loads
        保留vmcnt/lgkmcnt、barrier及priority切换
        online softmax：max超过old+7时改为row_max+1；rescale sum/O
        原round-half-up概率转BF16
        PV + 读取下一K fragment；交换LDS slot
    跨lane合并denominator；normalize；optional LSE
    原round-half-up输出转BF16；8个wave轮流复用O-LDS做C-shuffle
    fetch下一个ticket，直到该batch/head范围结束
```

没有改成另一种BF16 kernel；扩展的是host接口、正确的编译cache key、LSE和graph-safe seed。
per-call counter初始化仍计入total GPU时间。

### 单wave SWA：gfx942新增K16 / gfx950原K32

```text
CTA=64 threads；query_tile=16或32（同一wave两个16行subtile）
first = align_down(max(KV-Q+query_start-window, 0), BN)
end = max(0, min(KV, KV-Q+query_start+valid_query_rows))
load Q；max/sum/O初始化，四个lane group各持有1/4份sink denominator
for tile in [first,end), step BN:
    SMEM取page ID；packed K/V直接VMEM→register，无LDS缓存
    QK for one/two query subtiles；仅边界tile逐元素mask
    row max（942 ordered XOR16再XOR32；950 native permlane swaps）
    lazy slack=8；exp2、lane-local sum、P→BF16；需要时rescale O/sum
    清零无效V halfword；PV，两subtile共享V
epilogue: 最后一次跨lane sum，normalize、BF16 O与optional LSE
```

gfx950 QK使用16×16×32，PV在BN16用K16、其余K32；gfx942全部用native16×16×16
BF16-1k，将原K32 fragment按相同Q/K、P/V排列拆成两次K16，没有跨lane重排或多wavefallback。
QT16 gfx942对packed V加入局部tied-asm lifetime约束，修复编译后概率转换覆盖活跃V的情况；
没有改概率转换公式。核心60配置、完整矩阵、长形状均已验证。

## 保留的限制

- gfx950原生执行等待对应硬件；compile-only与历史数据不替代它。
- BF16942的原spill、输入范围、zero-padding契约保留，不在本次重构中另行优化。
- 单wave主攻窄SWA，不保证任意长窗口最优；不会偷偷换成8-wave。
- AITER page64 5D缺实例明确记录。输出shape/type检查不代替调用方对GPU metadata值的保证。
- 此目录之外原kernel源文件不需要修改；没有commit、stage或push。
- BF16942 spill已有原生功能/逐位验证；绝对TFLOPS仍须匹配原文策略的独占测试，低负载诊断不算验收。