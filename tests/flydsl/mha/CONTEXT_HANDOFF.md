# MHA跨机器测试与开发上下文

**MI325最新完整上下文已汇总到 [README.325.md](README.325.md)**，包括环境、测试/TFLOPS、
新性能参考、阻塞事实和不commit的完整迁移步骤；本页仍保留MI308历史来源与契约。

更新：**2026-09-07**。这份文档可独立交给下一台机器/下一位开发者，不需要读取聊天记录。
本次交付只做**本地Git提交，不推送**；用目标机的`git rev-parse HEAD`记录实际交付commit。

> **最新性能参考已由用户重新指定**：BF16以dense
> [test_attn_8wave_32x32_lkgv.py](../test_attn_8wave_32x32_lkgv.py)基本持平为目标；FP8以
> [pa_prefill_8w32x32.py](../pa_8wave/pa_prefill_8w32x32.py)的FP8分支为参考且不得更慢。
> 详见[REQUESTED_REFERENCES.md](REQUESTED_REFERENCES.md)。旧MI308/MI325数字不自动满足新验收。

> **当前目标机补充**：MI325X / gfx942 / **304 CU**，不是下文历史源机的 MI308X / 80 CU。
> MI325X 的 PTL 为 **N/A（不支持）**；本轮不做PTL设置，不套用MI308X的400T gate。
> 新机环境、独立证据、原版比较和剩余阻塞统一记在
> [MI325_VALIDATION.md](MI325_VALIDATION.md)，下文源机历史报告不覆盖、不改标签。
> **本轮交付指令更新**：用户在本机明确选择“只保留改动，不提交”；因此本轮不commit、不push。
> **MI325结果/阻塞**：初始完整native979通过/1310跳过，新增stress14通过/4跳过；BF16四项
> 同机对照及96项304CTA资源完成。FP8控制实验后GPU状态读取超时，后续GPU测试暂停；未reset。
> TFLOPS及未完成清单见MI325报告，不能将本轮宣称为全矩阵完成。

## 0. 先读这几条

1. 这是4个显式kernel的统一测试目录，不是自动fallback框架：FP8/gfx942、BF16/gfx942、
   BF16/gfx950、BF16单wave SWA/gfx942+gfx950。
2. 本文原始源机仅有MI308X/gfx942；当前目标机MI325X也为gfx942。
  **gfx950原生功能/性能仍未验收**，交叉编译不等于原生通过。
3. BF16/gfx942大量scratch spill已消除，但**D192/page32仍有约10.95%性能回退**。
   D192/page64低负载诊断改善15.44%，D128/page64基本持平。不要宣称所有shape已恢复。
4. FP8历史413.984T依赖MI308X和PTL Enabled/VECTOR,F8；当前改版尚未完成该条件下的独占验收。
   BF16历史250.952T是**4-wave H1/D128/Q=KV40960/page32**，不是当前8-wave D192短KV形状。
5. 最近性能结果均明确标为**PTL Disabled、非独占低负载诊断**；不能当正式250T/410T复现。
6. 新机先复测现有源码，后改代码；新结果使用独立输出目录，不覆盖历史JSON、JUnit或ISA。

配套文档：[接口与结果](README.md)、[覆盖对应](COVERAGE.md)、
[基线口径](PERFORMANCE_BASELINES.md)、[BF16 spill开发记录](BF16_SPILL_FIX.md)。
旧文档内的“暂停GPU/尚未测试/未提交”可能描述较早阶段；**当前任务状态以本文和对应运行的SHA为准**。

## 1. 代码、Git来源和迁移

### 1.1 仓库与版本

- 命令工作目录是Git仓库根（原机器为工作区内的pyhip子目录），不是仅进入MHA测试子目录。
- 重构起点：`origin/main = ebc533488b3d6a55e1dd386da2cc8c04293432ab`。
- 原优化分支：`lc/luocheng/mha_swa = 23cc6d1e95b1611493e21232bef5d9962b7b73c9`。
- 此次本地提交在上述main起点之上，**不再意味着本地main与origin/main相同**。
- 原版对照由 [validate_preservation.py](validate_preservation.py) 使用`git show`读取并校验源码SHA。
  **只拷贝新目录或只带一个patch不足以运行原版对照**；还需要pyhip核心/helpers及两个固定Git对象。

| 当前文件 | 原始来源/功能 |
|---|---|
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | 指定分支nested native gfx942 FP8，8-wave，BM256/BN64，LDS/register |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | **origin/main** prefill，8-wave persistent，BM256/BN32，现已做spill修复 |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | 指定分支direct-paged 8-wave BF16，保留static/persistent与SWA/sink |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | 指定分支真单wave64，原gfx950加原生gfx942适配 |

不要把指定分支里更长的另一个prefill文件当成BF16/gfx942的origin/main基线。

### 1.2 不推送也能携带完整历史

以下是**提交之后、由使用者在两台机器之间执行**的离线迁移方法，不会访问远端或推送。
源机保留的本地分支名若变化，先检查再替换命令中的名字。

```bash
# 源机，Git根目录；同时携带新main与固定原优化分支
git bundle create /tmp/pyhip-mha-handoff.bundle main lc/luocheng/mha_swa
git bundle verify /tmp/pyhip-mha-handoff.bundle
```

把生成的bundle传到目标机（传输方式由使用者选择），在目标机新目录克隆，避免覆盖旧工作树：

```bash
git clone -b main /path/to/pyhip-mha-handoff.bundle pyhip-mha-validation
cd pyhip-mha-validation
git switch -c mha-machine-validation
git rev-parse HEAD
git cat-file -e ebc533488b3d6a55e1dd386da2cc8c04293432ab^{commit}
git cat-file -e 23cc6d1e95b1611493e21232bef5d9962b7b73c9^{commit}
```

固定来源对象缺失时，不要跳过SHA校验或换用别的源码冒充原版。可先运行不依赖原版的测试，
但须将原版对照标为blocked。不要因为需要迁移就执行`git push`。

### 1.3 本地文件与可移植证据

- 提交包含本目录源代码、文档、JSON/JUnit/log及保留的ISA；排除Python/pytest缓存和临时环境。
- 本目录的 [.gitignore](.gitignore) 仅对results中的保留汇编放行，避免被仓库全局`*.s`规则漏掉。
- 旧JSON的`isa`可能是源机临时路径：BF16新报告看`retained_isa`（相对Git根）；
  [资源汇总](results/resources_all.json)、[950保留ISA索引](results/preservation_950_retained.json)
  中的保留路径相对本MHA目录。保留SHA，不全局替换历史JSON路径。
- 旧临时venv、AITER缓存、临时C++库、原机器ATT/UI大目录均不属于提交。
- FP8原始独立报告在旧未跟踪实验目录，已将验收关键内容保存为
  [可移植基线摘录](results/historical/fp8_942_native_baseline.md)，不依赖旧ATT/UI链接。

## 2. 新机环境

### 2.1 已验证的源机组合

| 组件 | 源机验证值 / 新机要求 |
|---|---|
| OS/GPU | Linux；MI308X gfx942，80 CU；另需gfx950机器补原生测试 |
| Python | **3.11.11**；统一工具使用3.11功能，不只看项目最低Python声明 |
| PyTorch/HIP | 2.12.1+rocm7.2 / 7.2.53211；目标机使用与本机ROCm匹配的构建并记录版本 |
| FlyDSL | **0.3.1**；原FP8历史为0.2.2，两者不是相同编译器验收 |
| Triton | 3.7.1，gather比较需要；使用目标ROCm环境对应构建，不随意换成CUDA wheel |
| pytest | 源机9.1.1；需pytest与filelock/numpy |
| AITER | 可选，对照时需要；源机checkout `83faabaa4bf077713c0c71546a8935313c01640b` |
| 工具 | ROCm工具链、amd-smi；目前隔离检查默认使用`/opt/rocm/bin/amd-smi` |

不要复制源机的临时解释器路径或site-packages软链接。建议在目标机自己的venv/容器中安装，
保留其现有环境；若继承系统ROCm PyTorch，仍需验证实际import路径。

示例（先确保所选基础Python确实拥有适配ROCm的PyTorch/Triton，否则先按目标机规范安装它们）：

```bash
python3.11 -m venv --system-site-packages "$HOME/.venvs/pyhip-mha"
source "$HOME/.venvs/pyhip-mha/bin/activate"
python -m pip install 'setuptools>=77' 'setuptools-scm>=8' 'flydsl==0.3.1' pytest filelock numpy
python -m pip install -e . --no-deps
```

`setuptools>=77`用于识别项目的现代license配置，**不要为绕过旧setuptools去修改项目元数据**。
不在目标机器盲目执行`pip install torch`替换已有ROCm构建。

```bash
python - <<'PY'
import sys, importlib.metadata, torch, pyhip
print('python:', sys.executable, sys.version)
print('pyhip:', pyhip.__file__)
print('torch:', torch.__version__, 'HIP:', torch.version.hip)
print('flydsl:', importlib.metadata.version('flydsl'))
import triton
print('triton:', triton.__version__, triton.__file__)
assert torch.version.hip, '需要ROCm PyTorch，不是CPU/CUDA构建'
PY
```

`pyhip.__file__`应绑定到当前克隆的src目录，而不是另一个旧checkout。源机曾出现editable绑定错仓库。
Triton可能可import但没有同名distribution metadata；先检查模块路径/版本，不因此反复重装。

### 2.2 AITER与C++ runtime

- 普通kernel正确性不依赖AITER；gather+linear完整比较必须有AITER CK、Triton。
- 在目标机按AITER自己的构建步骤安装对应checkout，不把源机Python3.10/3.11的二进制缓存混用。
- 设置独立`AITER_JIT_DIR`，`GPU_ARCHS`与目标GPU一致，限制`MAX_JOBS`以免影响共享CPU。
- 源机使用临时libstdc++解决`GLIBCXX_3.4.29`缺失；目标机先检查自身库，**不要直接覆盖系统库**。
- public linear router曾无条件导入不兼容的FlyDSL buffer_ops；本目录显式调用CK
  `mha_varlen_fwd`，不是静默换后端。Pylance的可选AITER导入警告不等于原生运行一定失败。
- page64 AITER direct-5D缺instance会明确报告`no matching kernel found`；不以linear耗时填入5D栏目。
- 原linear batch-prefill在128K有已知fault，gather比较使用原文推荐的CK varlen，不再冒险执行该路径。

## 3. 测试前的硬件与输出隔离

所有后续示例从Git根执行；`OUT`必须是本轮独立目录，路径保持绝对，避免旧source import切换cwd影响输出。

```bash
OUT="$PWD/tests/flydsl/mha/results/newmachine_$(hostname -s)_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH
export AITER_JIT_DIR="$OUT/aiter-jit"
export GPU_ARCHS=gfx942    # gfx950机器改成gfx950
export MAX_JOBS=4
git rev-parse HEAD > "$OUT/commit.txt"
git status --short > "$OUT/worktree.txt"
/opt/rocm/bin/amd-smi static --gpu 0 --limit --json > "$OUT/limits-before.json"
/opt/rocm/bin/amd-smi metric --gpu 0 --usage --mem-usage --json > "$OUT/usage-before.json"
/opt/rocm/bin/amd-smi process --gpu 0 --json > "$OUT/processes-before.json"
```

设备不是物理GPU0时，应正确设置可见设备并使用对应BDF检查；**当前PTL实验脚本仅授权/实现了物理GPU0**。
新机器需重新取得管理员/使用者的策略授权，旧机授权不自动扩展到新机。

- 正式性能需要调度层保留独占GPU。短暂0%可能是服务重启；显存驻留worker不能视为已退出。
- 不杀别人进程，不改时钟/功率/NUMA/其他卡。不要照抄AITER warning里的sudo系统设置建议。
- `--ptl current`默认不设置硬件；显式VECTOR,F8/BF16仅支持gfx942 GPU0、初始Disabled，finally恢复并留记录。
- 需要权限的设置使用非交互方式；权限不足时停止，不反复试提权。不将密码/令牌写入配置或日志。
- `--allow-contention`只产生诊断，不能与`--require-baseline`同时使用；不能与PTL实验一起绕过隔离。
- 监测器只有触发前采样，不保证之后整轮绝对空闲；要声称独占性能需额外资源预约/监控证据。

## 4. 建议执行顺序

### 4.1 CPU契约与case计划（先做，不需要GPU）

```bash
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python -m pytest tests/flydsl/mha/test_perf_cases.py tests/flydsl/mha/test_bf16_spills.py \
  tests/flydsl/mha/test_mha_pa_swa.py -q \
  -k 'test_perf_cases or (test_bf16_spills and not bit_exact_original) or (gather_linear and not comparison_matches_reference)' \
  --junitxml="$OUT/cpu.xml"

HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all --list-cases \
  --output "$OUT/documented-plan.json"

HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python tests/flydsl/mha/test_mha_pa_swa.py --mode performance --gather-linear --list-cases \
  --output "$OUT/gather-linear-plan.json"

HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
python -m pytest tests/flydsl/mha --collect-only -q
```

交付时完整收集2289项；不同平台的skip数不同，**收集数/skip不能当功能通过**。
普通文档性能矩阵44项，gather+linear默认6项。`--case`支持glob；documented模式的shape参数
是过滤器，不会自动缩Q/换page。真正自定义用`--matrix custom --q ... --kv ...`。

### 4.2 原生正确性（gfx942或gfx950）

先按目标架构跑小范围，再完整跑；以下都是功能测试，没有性能门槛：

```bash
# gfx942 BF16（含24项固定原版逐位对照）
python tests/flydsl/mha/test_mha_pa.py --mode functional --backend bf16_942 \
  --pytest-args -x --tb=short --junitxml="$OUT/bf16-942.xml"

# 单wave SWA（fixture自动选择当前native架构）
python tests/flydsl/mha/test_mha_pa_swa.py --mode functional \
  --pytest-args -x --tb=short --junitxml="$OUT/swa.xml"

# 完整最终源码回归；另一个GPU架构明确skip
python tests/flydsl/mha/test_mha_pa.py --mode functional --suite all \
  --pytest-args --tb=short --junitxml="$OUT/functional-all.xml"
```

gfx950专用优先选择`--backend bf16_950 bf16_950_persistent`；还要单独跑SWA，不只测static full。
`--pytest-args`必须放最后。功能模式的`--aiter required`**不负责**强制pytest参考存在；
检查JUnit中的AITER skip。要确认新gather路径真正执行，可先筛选
`gather_full_cache_and_live_values or gather_linear_comparison_matches_reference`。

### 4.3 BF16/gfx942资源审计（真正CPU-only，共96项）

```bash
for scale_mode in per-token per-tensor; do
  HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  FLYDSL_COMPILE_ARCH=gfx942 FLYDSL_COMPILE_ONLY=1 \
  python tests/flydsl/mha/compile_bf16_942.py --dq 128 192 --dv 128 192 \
    --page 32 64 128 --causal 0 1 --lse 0 1 --mode "$scale_mode" \
    --require-no-scratch --retain-isa \
    --dump-root "$OUT/dump-bf16-$scale_mode" --output "$OUT/resources-bf16-$scale_mode.json"
done
```

该工具用meta tensor、固定80-CTA元数据，不初始化GPU；生成真实gfx942 ISA但不执行。
它准确复现了原机器默认104/176B scratch。新的编译器构建仍需记录，不能仅凭相同版本字符串保证ISA一致。
`--require-no-scratch`检查Private/VGPR spill/scratch指令，不把合法的少量SGPR lane转存隐藏掉。

其它目标的统一审计：

```bash
python tests/flydsl/mha/test_mha_pa.py --mode audit --suite all --cross-compile \
  --dump-root "$OUT/audit" --output "$OUT/resources-all.json"
```

注意：这个**统一audit仍会分配本机GPU tensor**，只是目标kernel不执行；不等同上面的纯CPU工具。
`--strict-resources`还要求零SGPR spill，当前部分BF16 LSE及950 persistent不满足，失败不能通过改统计消除。

### 4.4 常规性能（先正确性、后计时）

```bash
# gfx942，FP8实际历史shape：先当前策略记录，不宣称已达400T
python tests/flydsl/mha/test_mha_pa.py --mode performance --backend fp8_942 \
  --case fp8_native_410t --aiter off --output "$OUT/fp8-current-policy.json"

# BF16原8-wave与当前版本；默认含page32两个historical event case及page64 D128/D192
python tests/flydsl/mha/reproduce_baselines.py --backend bf16 --ptl current \
  --output "$OUT/bf16-original-current.json"

# 单独排查当前已知page32回退，保持原10-buffer event协议
python tests/flydsl/mha/reproduce_baselines.py --backend bf16 --ptl current \
  --case historical_bf16_d192_events --output "$OUT/bf16-page32.json"

# 全44项、所有native可用backend；含长序列/H3，耗时和内存较大
python tests/flydsl/mha/test_mha_pa.py --mode performance --suite all \
  --output "$OUT/performance-all.json"
```

`--aiter auto`仅允许缺依赖/缺instance报告不可用；实际数值异常仍失败。
只有当确实准备好所有适用参考时才用`--aiter required`；当前page64 5D可能导致它按设计失败。
输出`complete=true`仅说明已处理完选中矩阵，仍须检查`records`非空和`unavailable`，不能把全跳过当验收。

### 4.5 SWA gather+linear性能（新增入口）

```bash
python tests/flydsl/mha/test_mha_pa_swa.py --mode performance --gather-linear \
  --aiter required --output "$OUT/swa-gather-linear.json"
```

默认6项：Q16K/H16/HK1/W128/sink、KV32K/64K/128K × D128/192。分别测`direct`、
`gather`、prepared CK linear、**每次gather+CK linear**，对完整逻辑KV gather，不剪掉SWA前缀。
slot mapping和预分配workspace在计时外，gather读取缓存内容在总路径计时内。
一对GPU event包住整个gather+linear，不能相加两个独立均值或拿attention-only冒充总时间。
`comparison_complete`也必须为true；该模式不支持profiler、多个buffer或baseline gate。

### 4.6 gfx950原版对照与PTL实验

```bash
# 只能在原生gfx950跑：原版单wave vs 当前单wave
python tests/flydsl/mha/reproduce_baselines.py --backend swa --ptl current \
  --output "$OUT/swa950-original-current.json"

# 交叉目标仅比较生成ISA，不代表native性能
python tests/flydsl/mha/validate_preservation.py --mode cross-compile \
  --dump-root "$OUT/preservation950" --output "$OUT/preservation950-compile.json"
```

如下FP8 gate命令**只在新机另获授权、MI308X/gfx942物理GPU0独占、PTL初始Disabled时执行**：

```bash
python tests/flydsl/mha/test_mha_pa.py --mode performance --backend fp8_942 \
  --case fp8_native_410t --ptl VECTOR,F8 --require-baseline --aiter off \
  --output "$OUT/fp8-authorized-ptl.json"
```

hardware sidecar必须显示恢复成功。不同GPU型号，即使同为gfx942，也不能关闭型号检查冒充MI308X验收。
`recheck_performance.py`不加`--when-low`会尝试BF16/F8 PTL实验，**不适合作为陌生机器默认入口**。
若只是共享机器低负载诊断：watcher配合`--allow-resident-workers`触发的子命令必须保持
`--ptl current --allow-contention`；`--when-low`队列始终诊断，空进程快照也不升级为PTL设置。

## 5. 计时/精度规则：四种口径不能混

| 模式 | 样本/聚合 | 应读取的时间 |
|---|---|---|
| 普通profiler | 一般100共同warmup、5轮、每轮20/100；丢首样本、1.5IQR均值后取轮次中位数 | `attention_us`主kernel；`total_gpu_us`含辅助GPU工作，不含完整CPU wall time |
| native FP8目标 | 同上但1200共同warmup | 必须与原per-token/native-cast/无LSE匹配 |
| historical BF16 events | 10独立buffer、10warmup、50样本中位数；H3为3warmup/10samples | `event_interval_us`包含辅助工作与dispatch间隙 |
| SWA gather+linear | 每轮20warmup、100samples、5rounds；候选每样本轮换；双层中位数 | 总路径一对event；单gather不报attention TFLOPS |

所有路径先验证全部输出的分块FP32参考，再检查多次重复逐位相等。FP8 O `rtol=atol=0.1`，
BF16 O `0.02`，LSE使用现有更严格阈值；不得改成`equal_nan`、放宽误差或吞掉launch异常。
TFLOPS只计真正可见的QK/PV工作：$2H_q\sum_b N_{visible,b}(D_{qk}+D_v)$；
W128包含当前key，最多129个key。sink是单份分母logit，不是额外key/value向量。

`--warmup/--rounds/--iterations/--timer/--buffers`覆盖原协议会被记录为unmatched，调参结果不算原文严格复现。
数值、输入、随机种子、page order、Q scale模式、LSE、预分配、buffer轮换、GPU/PTL/编译器必须一起记录。

## 6. 当前可靠证据与待测清单

### 6.1 已验证（不是要求新机达到完全相同用例计数）

| 范围 | 交付证据/边界 |
|---|---|
| 最新BF16/gfx942 | [205通过/46跳过](results/bf16_spill_v5_functional.xml)，其中24项固定原版逐位对照；[状态与SHA](results/bf16_spill_status.json) |
| BF16资源 | [per-token](results/bf16_overlap_v5_full_compile.json)、[per-tensor](results/bf16_overlap_v5_scalar_compile.json)，96项无scratch/VGPR spill/AGPR；12个NC/V128/LSE有2个SGPR lane转存 |
| BF16低负载对照 | [4项原版/当前性能](results/bf16_overlap_v5_performance.json)：D192/page64 −15.44%；D128/page64 +0.135%；H1长序列 −0.81%；D192/page32 **+10.95%** |
| gather+linear | [67项CPU](results/swa_gather_linear_cpu.xml)、[gather原生4通过](results/swa_gather_native.xml)、[CK/gather及既有参考40通过](results/swa_gather_linear_native.xml) |
| gather性能 | [6项实际结果](results/swa_gather_linear_performance.json)，4路径、12,000个原始样本；PTL Disabled/非独占，不与gfx950历史数字直接比 |
| 较早完整重构回归 | [865通过/1298跳过](results/functional_final.xml)，属于spill/gather之前的历史快照，不代表当前全量最终源码已重跑 |
| gfx950 | 88项混合目标资源矩阵及40组双版本ISA仅compile；native未在本机运行 |

四个kernel交付SHA256（更改源码之后必须重新验证，不可修改旧报告的SHA假装相符）：

| 文件 | SHA256 |
|---|---|
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `c08b2cdbc14e09ae88f965b1c0191d59fd03a8b1ca4a64eae56016b4853c016f` |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `ea5f74190b437221ab632895d9ef287f076238b13c962c22ee1e20655d2a5edd` |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `ac99ff25c7d42ac787836d384e83c60eb8284fac219201c3c125753158099384` |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `28c5ac6fdd73f3f6026742de256e940ac08d8652c0a74d42308ee69df13592ed` |

### 6.2 待测/待修（按优先级）

- [ ] **新机最终源码全量native回归**，记录各平台skip原因，不将2289项收集当全部通过。
- [ ] **gfx950 native验收**：BF16 static/persistent、causal merge、SWA/sink/pruning、stream/graph、
  单wave全部QT/BN配置；再跑原版单wave和gather+linear对照。
- [ ] **gfx942 FP8绝对基线**：MI308X独占、PTL相同、原/当前代码、0.2.2/0.3.1分开环境控制实验。
  413.984T是历史值、400T是明确门槛；其它卡只建立新基线，别绕开匹配。
- [ ] **BF16/gfx942 D192/page32 +10.95%回退**：固定H16/Q10240/KV2583、10buffer event协议，
  比较原8-wave与当前8-wave，单变量检查地址重算、LDS/MFMA重叠及编译器schedule。
- [ ] **BF16 250T归属复现**：若需要，显式`reproduce_baselines.py --backend bf16 --four-wave`。
  该branch4 comparator不是已归档August binary，原PTL/compiler未知，不能替换当前生产kernel或声称精确匹配。
- [ ] **44项性能矩阵独占复测**：补长序列、batch4、H1、H3 ragged、SWA window/query scans，
  特别是H3/10buffer可能较耗内存与时间，分组运行但不缩短输入伪造原case。
- [ ] **AITER/OPUS在目标机重新确认**：有instance必须真正检查数值；缺instance单列，不填其它后端的时间。
- [ ] **残余SGPR lane转存及目标编译器资源差异**：当前只有少量LSE转存，零scratch不等于所有寄存器零spill。
- [ ] 多stream/graph长时间回放、metadata变更、边界/空请求、异常shape在目标架构补压力测试。

## 7. 开发方法与不能破坏的契约

1. **先保存基线**：Git commit+dirty diff、源码SHA、环境、设备/PTL、原始时间样本、ISA/resource。
   原版与候选必须同输入/同编译器/同策略交替测；新机默认只读旧证据。
2. **先重现再改**：数值问题用小case/FP32/逐位定位；spill先用meta编译审计，不必等待GPU。
   不因`get_errors`无错或LLVM exit0就认定编译合法，必须检查AST、真实诊断、ISA和测试。
3. **每轮单变量**：先小数值门槛→资源矩阵→原版逐位对照→完整native→原计时protocol。
   没有时钟/PTL控制证据时不宣传微小差值；保留失败/拒绝候选的原因。
4. **避免重复失败方向**：全路径无条件O rescale和宽调度fence曾使D128慢约30%；已恢复lazy
   rescale与适用路径重叠。D192/page32 padding LDS、原地址reuse、PV假依赖等未改善，不要只凭零spill合入。
5. **ISA规则**：gfx942不能用gfx950的direct buffer→LDS DMA、permlane swap或BF16 native pack；
   模式编译/真实设备要分开。raw `vgpr_count`可能含AGPR，须分别记录AGPR、Private、V/S spill。
6. **BF16/gfx942**：维持8wave/BM256/BN32、per-call独立counter及graph-safe `[:1].fill_` seed；
   active KV>0、causal每个sequence KV>=Q、零padding契约；概率/O的round-half-up不可悄悄改RNE。
   `_uniform`只能用于实际uniform metadata，不能把per-token Q scale广播；不要删同步来追性能。
7. **SWA**：真64线程单wave、无生产gather/workspace/多wave fallback；W128是闭区间129key；
   sink只加一次，masked/empty保持O=0及原LSE语义。gfx942 packed-V生命周期约束不可随意删。
8. **gather比较**：只在测试工具内，完整KV、固定slot mapping、预分配workspace；每次总路径
   确实执行gather+CK。计时用单个interval，单gather没有attention TFLOPS；不引入到生产dispatch。
9. **原版加载/缓存**：保持固定Git源码hash，临时cwd隔离原main import时的dump副作用。
   原BF16 factory按dtype/device缓存，跨workload需清cache，不能在计时/轮换buffer中清。
10. **禁止虚假通过**：不放宽精度、不吞异常、不把unsupported/compile-only当pass、不换kernel来源；
    不能给不相同环境/协议的结果改标签来满足400T。

[summarize_results.py](summarize_results.py) 是**早期历史汇总器**：依赖旧临时ISA路径及cleanup断言，
当前源码已语义更新，不能在新机直接重跑来生成“当前验证总结”。新报告直接使用各CLI的输出，
若要改汇总器，先明确各历史版本与新版本的来源映射，不能覆盖历史摘要假装新结果。

## 8. 完成新机测试后的交回内容

- 本轮commit、dirty diff、Python/torch/HIP/FlyDSL/Triton/AITER版本与实际import路径。
- GPU型号/架构/CU、设备映射、PTL/功率/时钟状态；独占预约或争用说明、任何授权设置及恢复记录。
- 选中case的完整输入与protocol、全部raw samples、正确性/JUnit、unavailable与skip原因。
- 新旧ISA/resource及SHA、保留汇编路径；哪部分只是compile、哪部分真实native执行。
- 明确回答：D192/page32是否消除回退、gfx950是否完成、FP8是否在匹配条件达到400T、gather总路径是否真的包含两段。
- 写明未测/失败/剩余风险；提交前检查范围。**未获得新的明确指令时不要推送。**