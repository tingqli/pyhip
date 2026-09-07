# 测试来源与覆盖对应

> **当前MI325执行证据**见[MI325_VALIDATION.md](MI325_VALIDATION.md)：初始完整979通过/1310跳过，
> 后增stress14通过/4跳过（49,152次graph内调用）及纯离线报告回归。历史计数不与新计数相加。
> 四个生产kernel未变；gfx950缺硬件，FP8控制实验后剩余GPU性能因设备健康阻塞未完成。

最新BF16942 spill修复：96项CPU-only真实ISA编译、最终整合205通过（含24项原版逐位对照）、
66项CPU契约回归（有11项重叠），详见 [BF16_SPILL_FIX.md](BF16_SPILL_FIX.md)。

性能覆盖已单独扩展为原文44项workload及独立协议，见
[PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)、[_perf_cases.py](_perf_cases.py)；
本次GPU忙碌，只有CPU计划/回归，不能把上一轮数值测试和26组冒烟性能当作扩展矩阵已执行。

## 本次性能工具CPU回归

[test_perf_cases.py](test_perf_cases.py) 的55项回归隐藏GPU，并阻止真实GPU初始化和子进程调用；
计时与硬件操作只使用CPU模拟。包括：

- 原22项矩阵、18项SWA扩展、FP8 native目标和3项historical event workload，共44项。
- 原文FLOPs、输入生成/量化顺序、10套buffer轮换、effective protocol序列化。
- shape/backend/GPU/PTL/protocol不匹配、无数值门槛、无穷/零耗时、未隔离不能通过验收。
- 原版对照共享生成协议、每workload清空原BF16 factory cache、SWA必须有原gfx950。
- 错误保留`complete=false`，event interval不伪装成attention-only。
- PTL成功/数值异常/format失败/enable部分失败后的恢复；权限失败不重试、忙GPU不设置。

另重跑57项现有纯CPU oracle/地址/参考异常/runner测试，见
[results/performance_helpers_cpu.xml](results/performance_helpers_cpu.xml)。本次共112项CPU通过，
均不计为生产kernel的GPU正确性/性能验证。

内核来源固定在 [validate_preservation.py](validate_preservation.py) 的 `SOURCES`，包含提交、
Git 内路径和 SHA256；不依赖当前其他 attention 目录，也不把未执行的 gfx950 用例计为通过。

## 来源

| 新模块 | 原版本 |
|---|---|
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `lc/luocheng/mha_swa`，`23cc6d1e`，native gfx942 优化版本 |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `lc/luocheng/mha_swa`，`23cc6d1e`，direct-paged 8-wave |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | **`origin/main`，`ebc5334`**，原 BN32 persistent prefill 的 BF16 分支 |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `lc/luocheng/mha_swa`，`23cc6d1e`，单 wave SWA；新增 gfx942 |

旧本地 main 指向的 `8534483` 已保存在 `refactor-backup/main-lc-8534483`，当前 main 跟踪
`origin/main`。原 FP8 的 85-case native suite 与外层 942 suite 来自切换前保存的测试快照。

## Full PA：共享测试而非拼接旧测试文件

[test_mha_pa.py](test_mha_pa.py) 共用 [_testing.py](_testing.py) 的 `Case`、数据生成、
FP32 oracle、重复执行和 guard 检查。`Backend` 参数选择实际实现；不支持的能力显式跳过。

| 原测试组 | 新测试 | 保留的关键维度 |
|---|---|---|
| native FP8 `test_page_tails`、950 `test_page_parity_and_tails` | `test_page_parity_and_tails` | D128/192、C/NC、16 种 KV 尾长；FP8 LDS/register；950 static/persistent |
| 外层942 `test_fp8_page64_matches_reference` / poisoned tails | `test_original_quantized_fp8` | Q/KV=37/64、129/192、65/128、256/256；另有 Q65/KV65,79,95,127；真实量化、反序页 |
| 原 main `test_bf16_accuracy` | `test_original_bf16_cases` | 三个原始 D/page/Q/KV/C 组合，另增 LSE 开关 |
| 原 main 的 page/scale/layout 参数 | `test_bf16_942_original_extended_pages` | BF16 D128/192、V128/192、page32/64/128、两种 Q scale、C/NC |
| ragged GQA / strides / offsets | `test_ragged_gqa_offsets_and_scales` | 多序列、空请求、头分组、前缀偏移、非单位 scale；原942 BF16限制单独标明 |
| descales / lazy max / custom scale | `test_descales_large_logits_and_softmax_scale` | magnitude4、两种 Q scale、D128/192、C/NC、显式 softmax_scale |
| exact values / C-shuffle / BF16 ties / FNUZ safe range | `test_exact_zero_logits_value_and_lse`、`test_exact_output_pattern_and_guards`、`test_fp8_bf16_rounding_and_lazy_range` | 精确值与 LSE；逐位比较，不以宽松误差替代 |
| LDS/register comparison | `test_fp8_memory_modes_are_bit_exact` | D128 NC、D192 NC/C，Q1025/KV777、GQA/非单位scale |
| target shape / determinism | `test_target_shape_and_repeated_no_lse` | Q10240/KV2560,2583；BF16942 causal 改用合法 Q<=KV；全帧 FP32 检查 |
| live pages / page-table updates / physical aliases | `test_live_page_and_cache_mutations`、`test_shared_physical_pages` | 同一 callable 读更新内容；跨序列重复物理页；检查编译缓存不随内容变化 |
| poisoned V tail / forward and reverse merge | `test_runtime_lengths_poisoned_tail` | LSE开关；length=1,64,65,128,193,321,320；950 C 使用 Q7937 触发 merge |
| scalar Q scale + runtime aliases + graph | `test_fp8_scalar_scale_runtime_aliases_and_graph` | 原 Q513、H6/HK2；[1,1]、[3,0]、[4,1,4,2] 等页映射、两条 stream |
| empty requests / allocation / graphs | `test_empty_requests`、`test_stream_graph_and_allocation`、`test_warmed_dispatch_contract` | 空Q/空KV能力约束；预热后的单dispatch；BF16942显式计入counter初始化 |
| packed addressing / LDS banks | `test_fp8_packed_address_invariants`、`test_cshuffle_identity_and_banks` | CPU byte identity、bank映射、odd-row word pair反转 |

**明确裁剪**：原 main 的 `test_accuracy` 是 FP8 模型矩阵，并非另外36个BF16测试。
第4项需求允许只保留该实现的 BF16 分支，所以不要求它继续接受 FP8；保留三个 BF16 原始
case，并将 page/mode/V192 参数在 BF16 下扩展测试。所有 full PA 均不支持 linear cache。

## gfx950 保留扩展

以下用例仍在 [test_mha_pa.py](test_mha_pa.py)，`TestGfx950Extensions` 只参数化实际的
static/persistent 两个950 backend。**不是用单 wave SWA 的测试代替8-wave实现的测试。**

| 原测试组 | 新测试/测试族 |
|---|---|
| direct LDS rings / default no-LSE determinism | `test_original_dispatch_and_sink_logits`、`test_default_no_lse`：原 Q256/KV256 10次，Q257/KV256,321 11次；full/SWA/sink |
| causal head-tail merge / ragged empty-allmasked | `test_gfx950_causal_merge`、`test_merge_ragged_empty_rows`：Q7937/8193，batch Q33/4097/65 |
| sliding boundaries / sink denominator | `test_window_boundaries`、`test_sink_denominator`：W0,1,63,64,65,127,128,129,512；sink −80/0/80；非单位scale |
| full attention sink, large logits | `test_gfx950_full_sink`、`test_original_dispatch_and_sink_logits` |
| SWA ragged strides / persistent ragged empty requests | `test_ragged_scheduler_modes`：原短/长两组 ragged，padded/head-major，C/NC、W−1/0/128、sink、scale |
| more tasks than CUs / counter reuse | `test_persistent_ticket_reuse`：Q12289/KV901，C/NC/W128；12次，counter回到[grid,0]，禁止预热后分配 |
| live query mapping / empty grid | `test_persistent_runtime_query_mapping`：固定host bounds，GPU前缀多次改变，包括全空 |
| persistent stream isolation / repeated graphs | `test_stream_isolation_and_repeated_graphs`：Q4097，双stream独立counter，每graph两次调用、8轮replay |
| SWA runtime length/sinks | `test_runtime_lengths_and_sinks`：2049→257→128→0→193，sink含−inf |
| excluded prefix / exact inclusive window | `test_excluded_prefix_and_pruned_grid`、`test_exact_window_and_sink`：无效页ID、W+1个key、单份sink |
| pruning threshold | `test_pruning_thresholds`：Q16128/16129/16385，D128 W64/65、D192 W128/129 |
| sink buffer / factory scope | `test_sink_buffer_contract`、`test_gfx950_factory_contract`、`test_invalid_runtime_buffers` |
| original AITER 5D / OPUS references | `test_aiter_reference`、`test_aiter_window_reference`、`test_explicit_opus_comparison`、`test_opus_rejects_unsupported_semantics` |

## 单 wave SWA

[test_mha_pa_swa.py](test_mha_pa_swa.py) 保留原始全部数值矩阵和形状，在gfx942/gfx950两种
架构分别收集，不导入旧测试，不调用其他wave数的实现：

- D128/192 × QT16/32 × BN16/32/64 × 原始5种形状，共60种核心配置/架构。
- 原16种window × sink开关、18种KV length、padded/head-major、scale、extreme sink。
- 原精确 inclusive window、页/缓存变更、排除前缀、空Q、双stream/graph、无workspace单launch。
- 原CPU K/V地址覆盖与query/output/window模型；新增gfx942 K16 atom选择、CPU oracle edge cases。
- Q16384/KV131072/W128 的 auto + 6种显式tile都作为正确性用例，不只测时间。
- 新增 runtime query mapping、物理页别名、storage offset、严格invalid-buffer提前失败。
- AITER 5D和**显式CK linear**参考分别标记；后者先gather再计时，不代表原5D缓存性能。

### 新增gather+linear性能口径

[test_mha_pa_swa.py](test_mha_pa_swa.py) 的 `--gather-linear` 对应原分支 `23cc6d1e` 的
`test_swa_aiter_production_performance`（Q16K、KV32K/64K/128K），并扩展到D128/D192。
分别计direct、完整KV gather、prepared CK linear及gather+CK linear，采用原20/100/5
轮换GPU-event协议。生产单wave实现仍无gather/workspace/fallback。

新增检查包括：
- `test_gather_full_cache_and_live_values`：单/多KV-head、ragged/空请求、NaN尾页、缓存变更、
	gather workspace复用；检查完整逻辑KV而非SWA后缀。
- `test_gather_linear_comparison_matches_reference`：原prefix/extend样例及ragged样例，D128/192，
	W0/W128、sink开关，FP32与direct输出检查，以及总路径dispatch数验证。
- CPU验证每次总路径必有gather、已提供linear workspace不隐式重建、计时包含两段与gap、
	真实数值错误不吞掉、可选参考不可用标记、六组CLI计划和不兼容计时选项拒绝。

证据：[CPU回归](results/swa_gather_linear_cpu.xml)、[native gather](results/swa_gather_native.xml)、
[native gather/CK及现有参考回归](results/swa_gather_linear_native.xml)。gfx950因无硬件明确跳过。
分别为67通过、4通过/4跳过、40通过/104跳过；其中40通过包含新增8项gather+linear数值检查。
六组完整20/100/5性能对照已执行（PTL Disabled、非独占诊断），见
[results/swa_gather_linear_performance.json](results/swa_gather_linear_performance.json)。

## 正确性标准

- 独立chunked **FP32** bottom-right oracle，不把另一FlyDSL实现当唯一参考。
- FP8 O `rtol=atol=0.1`；BF16 O `0.02`；LSE沿用原严格阈值。
- 输出/LSE重复执行须逐位相等，前后guard和padded storage不得被覆盖。
- 可选reference只有缺依赖/缺匹配kernel会跳过；数值错误、异常launch仍失败。
- gfx950必须有原生硬件才运行数值测试。交叉编译只证明目标ISA可生成，**不计为功能测试通过**。