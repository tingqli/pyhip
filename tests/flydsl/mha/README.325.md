# MI325X MHA 完整交接：环境、结果、性能参考与跨机复测

更新：**2026-09-07 UTC**。本文件汇总截至此次交接的全部有效开发/测试上下文，
可独立交给下一台机器或下一位开发者，不需要读取聊天记录。
正文包含重建与继续测试步骤；链接指向需一起迁移的源码、原始样本、JUnit、ISA和版本清单。
**所有命令示例从 Git 仓库根目录执行，不是在本文件所在目录执行。**

阅读顺序：先看第0节状态与第6节最新验收要求；迁移看第10节，环境看第3节，
目标机执行看第7节。第4/5节是已完成结果，第8节是阻塞事实，第11节是剩余任务。

## 0. 下一位开发者先读

1. 当前测试机是 **MI325X / gfx942 / 304 CU / 256 GiB HBM**，不是源机的MI308X / 80 CU。
   两者同属gfx942，但不能按304/80换算性能，更不能混用硬件/编译器/计时协议的验收标签。
2. MI325X的PTL读取为 **N/A，不支持**。本轮没有设置PTL、时钟、功率或NUMA，没有reset GPU。
   不要在新机照抄MI308X的VECTOR,F8/BF16设置。
3. **最新用户性能要求**：BF16以 [dense LKG/V-global参考](../test_attn_8wave_32x32_lkgv.py)
   为基准、性能基本一致；FP8以 [BN32 paged prefill的FP8分支](../pa_8wave/pa_prefill_8w32x32.py)
   为基准、不得慢于它。两项最新门槛目前**均未经过native性能验收**。
4. 已完成：初始完整回归 **979 passed / 1310 skipped / 0 failed**；新增压力测试
   **14 passed / 4 skipped**；BF16 **96项零scratch/VGPR spill**；4个旧BF16原/当前性能对照。
   这些是不同运行，部分覆盖重叠，不能相加成新的唯一通过数。
5. **最后保存的GPU健康检查仍超时**。FP8旧对照实验无计时样本，随后amd-smi读取阻塞。
   原因未定位，不声称已证明是kernel、编译器或304CU硬件问题。下一台机器先独立确认健康。
6. 本轮四个生产kernel和两个新指定参考文件**没有修改**。新增的是环境、审计/对照工具、
   测试、文档和证据；不在无法验证时改调度、放宽精度或换fallback。
7. 用户已明确选择 **“只保留改动，不提交”**。这是最新指令，覆盖早期交接的本地commit要求。
   **不commit、不push、不擅自配置Git身份**，除非用户另行明确要求。
8. 当前工作树含大量**未跟踪的新工具、文档、结果**。只带commit、普通git diff或旧patch会丢失内容；
   按第10节迁移完整工作树和Git对象。不要覆盖旧结果或源机环境。

### 当前状态一览

| 项目 | 当前结论 |
|---|---|
| MI325环境 | 已重建；实际import路径、ROCm GPU smoke通过，有完整版本记录 |
| 现有kernel native回归 | 初始2289项：979通过、1310跳过；不是2289通过 |
| 长graph/多stream压力 | 14通过、4个gfx950跳过；49,152次graph内attention调用 |
| BF16/gfx942资源 | 96项无scratch/VGPR spill/AGPR；12项各有2个SGPR lane转存 |
| BF16旧原8wave对照 | 四项已测，µs/TFLOPS见第5节，不是最新dense参考的验收 |
| BF16最新dense参考 | CPU计划/契约完成，native性能未测；“基本一致”尚未证明 |
| FP8最新BN32参考 | CPU计划/契约完成，native性能未测；“不得更慢”尚未证明 |
| FP8旧对照 | 停滞后终止，`complete=false, records=[]`；没有FP8 TFLOPS |
| gather+linear | 完整KV gather及CK native正确性完成；MI325六项长shape计时未完成 |
| 全44项性能矩阵 | 计划完整，未全量计时 |
| gfx950 / OPUS | 本机无gfx950，native blocked；新跨目标审计也未完成 |
| GPU恢复 | 本轮进程已清理；未执行reset；需管理员确认设备/驱动健康 |

## 1. 仓库、代码来源与不能混淆的三个“参考”

### 1.1 Git与工作树

| 项目 | 值 |
|---|---|
| 源机工作目录 | /home/cheluo/pyhip |
| 当前分支 | `tmp-main` |
| 当前HEAD / 开始测试HEAD | `f06791491c3b157793500252a0127d8f45c7776a`，本轮未提交，未变化 |
| 重构起点 / 固定main源对象 | `ebc533488b3d6a55e1dd386da2cc8c04293432ab` |
| 固定优化分支源对象 | `23cc6d1e95b1611493e21232bef5d9962b7b73c9` |
| 当前包含优化源对象的ref | `refs/remotes/origin/luocheng/mha_swa`；不要假设旧本地分支名仍存在 |

两个固定源commit都已在本地核验存在。[validate_preservation.py](validate_preservation.py)
使用`git show`及源码SHA加载原实现；只复制MHA目录会缺Git对象、pyhip核心及helpers。
远端`origin/main`的当前位置不是证据，**固定commit+源码SHA才是**。

### 1.2 四个显式生产实现

本目录不是自动fallback框架，backend选择必须显式：

| 当前文件 | 路径与约束 |
|---|---|
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | gfx942 FNUZ FP8；8wave/BM256/BN64；LDS与register两种显式路径；page64/V128 |
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | 固定main BF16分支演进；8wave/BM256/BN32/persistent；K经LDS、V在寄存器；Dq/Dv128或192、page32/64/128 |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | gfx950 8wave BF16；static/persistent；保留causal merge、SWA/sink |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | 真单wave64；gfx942和gfx950；QT16/32、BN16/32/64；生产路径无gather/workspace/fallback |

`fp8_942_register`不是另一份源文件；`bf16_950_persistent`不是另一个架构。
gfx950源码能导入或能交叉编译，不代表在MI325上可以native执行。

### 1.3 源码SHA256（本轮全部未改）

| 文件 | SHA256 |
|---|---|
| [mha_pa_bf16_942.py](mha_pa_bf16_942.py) | `c08b2cdbc14e09ae88f965b1c0191d59fd03a8b1ca4a64eae56016b4853c016f` |
| [mha_pa_fp8_942.py](mha_pa_fp8_942.py) | `ea5f74190b437221ab632895d9ef287f076238b13c962c22ee1e20655d2a5edd` |
| [mha_pa_bf16_950.py](mha_pa_bf16_950.py) | `ac99ff25c7d42ac787836d384e83c60eb8284fac219201c3c125753158099384` |
| [mha_pa_swa_bf16.py](mha_pa_swa_bf16.py) | `28c5ac6fdd73f3f6026742de256e940ac08d8652c0a74d42308ee69df13592ed` |
| [BF16新指定dense参考](../test_attn_8wave_32x32_lkgv.py) | `c0880420cd10a797c59087d4f73e942aa237020c3f0793b2fa0bcd9a6ac0776a` |
| [FP8新指定BN32参考](../pa_8wave/pa_prefill_8w32x32.py) | `620209a023ccb5ea566489774d19edae880dfbcee298613233ed6f01f3b59849` |

必须区分：

1. **原版保留/逐位测试来源**：固定Git对象，不能随新性能需求改SHA。
2. **旧MI325性能对照**：当前BF16 vs 固定main的原8wave **BF16分支**，结果在第5节。
3. **最新性能要求**：BF16改为dense LKG/V-global参考；FP8改为该BN32 prefill的 **FP8分支**。
   新参考见第6节，不能将第2项结果重新贴标签当第3项通过。

旧FP8停滞实验使用的是固定优化分支里的nested native-BN64 FP8原版，不是后来指定的BN32 FP8参考。
源码选择详见 [validate_preservation.py](validate_preservation.py) 与
[compare_requested_references.py](compare_requested_references.py)，不要按同名文件猜来源。

## 2. MI325机器与硬件策略

已验证机器为 `asrock-1w300-e4-1e`，Linux x86_64，内核
`5.18.2-mi300-build-140423-ubuntu-22.04+`。当前只暴露一张卡：

| 项目 | 实测值/注意事项 |
|---|---|
| GPU名称 | AMD Instinct MI325X |
| 架构 | `gfx942:sramecc+:xnack-` |
| CU | **304**；源MI308X仅80 |
| GPU映射 | 逻辑0 → 物理0 → BDF `0000:83:00.0` |
| HBM | torch报告274,861,129,728 bytes / 262128 MiB，约256 GiB |
| ROCm系统工具链 | 7.2.3；amd-smi 26.2.2+c2d9476115 |
| clang | AMD clang 22.0.0git，roc-7.2.3；FlyDSL另带自己的LLVM/MLIR二进制 |
| PTL | `ptl_state=N/A, ptl_format=N/A`，不是Disabled |
| 功率限制 | static报告1000W；仅记录，未修改 |
| CPU | 工程样片 `AMD Eng Sample: 100-000000894-03`，96核/192线程 |
| CPU/NUMA | 未设置亲和性；观察到CPU允许0–191、memory node 0，不能据此改系统配置 |

源机MI308X的FP8历史 **413.984T** 依赖其PTL Enabled/VECTOR,F8和FlyDSL0.2.2；
历史 **400T gate不适用于MI325X**。BF16历史250.952T属于4wave/H1/D128/Q=KV40960/page32，
也不是当前8wave或最新dense参考的验收。

硬件记录：[static-before.json](results/newmachine_mi325_20260907T035439Z/static-before.json)、
[metrics-before.json](results/newmachine_mi325_20260907T035439Z/metrics-before.json)。
两秒级[metrics](results/newmachine_mi325_20260907T035439Z/metrics-watch.csv)和
[process](results/newmachine_mi325_20260907T035439Z/processes-watch.csv)监测不是从任务第一秒开始，
且不能证明外部调度层预约了独占GPU；报告中的`isolated=true`只说明对应CLI执行了进程隔离检查。

## 3. 所需环境与重建方法

### 3.1 已验证的主环境

| 组件 | 实测版本 |
|---|---|
| Python | **3.11.11**；统一工具依赖3.11功能，不能只看项目最低3.8声明 |
| torch | **2.12.1+rocm7.2**，官方ROCm wheel |
| torch HIP runtime | **7.2.53211**；与系统ROCm7.2.3的版本字符串不完全相同 |
| FlyDSL | **0.3.1** |
| Triton | **triton-rocm 3.7.1**；不要装CUDA wheel替换 |
| pytest / NumPy | 9.1.1 / 2.4.6 |
| setuptools / setuptools-scm | 78.1.0 / 10.2.3；现代license配置要求setuptools>=77 |
| filelock / packaging | 3.32.3 / 26.3 |
| ninja / cmake | 1.13.2 / 4.4.3 |
| pybind11 / pandas | 3.1.0 / 3.0.5 |
| psutil / einops | 7.2.2 / 0.8.2 |
| pyhip | editable，实际路径为当前工作树src；版本号随dirty树变化，commit+SHA更重要 |
| C++ ABI | 系统libstdc++已有GLIBCXX_3.4.30，本机不需要临时替换系统库 |

源机实际布局如下，**仅用于辨认，不要跨机复制解释器或site-packages软链接**：

```text
$HOME/.venvs/pyhip-mha-mi325/              主Python3.11/FlyDSL0.3.1环境
$HOME/.venvs/pyhip-mha-mi325-flydsl022/    独立FlyDSL0.2.2控制环境
$HOME/.local/src/aiter-mha-mi325/         AITER固定源checkout
$HOME/.cache/pyhip-mha-mi325/aiter-jit/    本机独立Python3.11/gfx942 JIT缓存
<仓库>/.venv/                            原Python3.10环境，未替换/删除
```

原环境安装工具曾误把依赖装入旧3.10 venv，虽然提示成功，实际3.11缺FlyDSL/NumPy。
随后使用显式目标解释器安装并核验imports。**不要只相信编辑器选择、安装成功信息或PATH**。
[工作区解释器设置](../../../.vscode/settings.json)目前指向用户目录中的新3.11环境；
迁移后路径不同必须重新选择，不能靠修改`extraPaths`掩盖装错环境。

完整记录：[实际import与GPU smoke](results/newmachine_mi325_20260907T035439Z/environment-verified.json)、
[最终distribution/双编译器二进制指纹](results/newmachine_mi325_20260907T035439Z/environment-final-inventory.json)。
FlyDSL0.3.1与0.2.2内嵌的LLVM/MLIR不同，仅版本字符串相同也不能保证ISA相同；必要时比较二进制SHA。

### 3.2 在新机建立独立环境

前提：新机已由管理员安装与目标GPU匹配的ROCm/驱动，具有相应设备访问权限。
以下示例供**目标机**使用，本次编写文档时未执行。已有同名venv时换一个新名字，不覆盖原环境。
`uv`可用时按下例安装用户级Python；没有`uv`可按其官方安装说明准备，或用已存在的Python3.11创建venv。

```bash
# 在目标Git仓库根；uv需已安装，环境目录需尚未存在。
ENV="$HOME/.venvs/pyhip-mha-mi325"
uv python install 3.11.11
uv venv --python 3.11.11 --seed "$ENV"
PY="$ENV/bin/python"

# 独立环境明确使用ROCm wheel；不要盲目pip install torch替换系统环境。
uv pip install --python "$PY" 'torch==2.12.1+rocm7.2' \
  --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install --python "$PY" \
  'setuptools==78.1.0' 'setuptools-scm==10.2.3' 'flydsl==0.3.1' \
  'pytest==9.1.1' 'numpy==2.4.6' 'filelock==3.32.3' 'packaging==26.3' \
  'ninja==1.13.2' 'cmake==4.4.3' 'pybind11==3.1.0' 'pandas==3.0.5' \
  'psutil==7.2.2' 'einops==0.8.2'
uv pip install --python "$PY" -e . --no-deps
```

已有目标ROCm torch环境也可用新venv继承system-site-packages，但必须记录实际版本和路径；
不能一边继承另一checkout的editable pyhip，一边声称当前源码被测。
不为兼容旧setuptools去修改 [pyproject.toml](../../../pyproject.toml) 的license元数据。
Triton可能模块可import但没有同名`triton` distribution；优先检查模块版本/路径，别因此反复重装。

**先执行第7节的纯stdlib健康预检，再做任何torch/FlyDSL import或AITER构建**。
健康通过后，在目标仓库根验证：

```bash
"$PY" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys, torch, triton, flydsl, pyhip
print('python:', sys.executable, sys.version)
print('torch:', torch.__version__, torch.__file__, 'HIP:', torch.version.hip)
print('flydsl:', md.version('flydsl'), flydsl.__file__)
print('triton:', triton.__version__, triton.__file__)
print('pyhip:', pyhip.__file__)
assert sys.version_info[:2] == (3, 11)
assert torch.version.hip and torch.cuda.is_available()
assert Path(pyhip.__file__).resolve() == (Path.cwd() / 'src/__init__.py').resolve()
prop = torch.cuda.get_device_properties(0)
print('gpu:', prop.name, prop.gcnArchName, 'CU:', prop.multi_processor_count)
# 若目标就是MI325X，应同时匹配304CU；其它设备必须建立另一个基线。
assert 'MI325' in prop.name and prop.gcnArchName.startswith('gfx942')
assert prop.multi_processor_count == 304
x = torch.arange(16, device='cuda')
assert torch.equal(x + 1, torch.arange(1, 17, device='cuda'))
torch.cuda.synchronize()
PY
```

若发生卡住/超时，停止后续操作并记录阶段；不要开启无限重试。第8节解释为什么隐藏GPU也未必避免runtime import阻塞。

### 3.3 AITER：可选参考，不是kernel运行依赖

普通FP32正确性和最新两个用户指定参考的对照**不需要AITER**。
gather+CK linear性能比较、AITER专门用例才需要：

- 本机AITER：`bde46043bcf08e41ac40395a18369ab6309153ca`，安装版本`0.1.21.dev11+gbde46043b`。
- CK子模块：`15e12dd7f25ee583617c78f66cb502ff9916585f`。
- 旧MI308交接写的`83faabaa4bf077713c0c71546a8935313c01640b`无法从公开上游取到：
  Git返回`not our ref`，提交网页404（API查询亦未解析）。不冒充取到了该旧源码。
- 本机明确选择了main升级FlyDSL0.3.2之前的固定提交，源码自己要求FlyDSL0.3.1；
  AITER源码无修改，但**不是旧机AITER精确复现**。
- `AITER_USE_SYSTEM_TRITON=1`保留ROCm torch配套的Triton；`PREBUILD_KERNELS=0`避免全量预编译，
  CK仍启用，不是`AITER_TRITON_ONLY=1`。实际首次调用会JIT，可能花数十分钟。

在已通过GPU健康/import检查的新机上：

```bash
AITER="$HOME/.local/src/aiter-mha-mi325"
# 目标目录应为新的空位置；已有checkout先检查，不覆盖。
git clone --no-checkout https://github.com/ROCm/aiter.git "$AITER"
git -C "$AITER" checkout --detach bde46043bcf08e41ac40395a18369ab6309153ca
git -C "$AITER" submodule update --init --recursive --jobs 4
export GPU_ARCHS=gfx942 MAX_JOBS=4
export AITER_JIT_DIR="$HOME/.cache/pyhip-mha-mi325/aiter-jit-newrun"
AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
  uv pip install --python "$PY" --no-deps --no-build-isolation -e "$AITER"
git -C "$AITER" rev-parse HEAD
git -C "$AITER" submodule status
git -C "$AITER" diff
```

安装后再次确认torch/Triton/FlyDSL没有被替换。不要混用Python3.10/3.11二进制cache、其他机器的AITER JIT产物。
本目录对linear显式调用CK `mha_varlen_fwd`，避开会导入不兼容FlyDSL buffer_ops的public router。
page64 direct-5D在本机缺instance会报告`no matching kernel found`；该栏目不能填linear时间。
128K路径采用CK varlen，不冒险重跑历史已知fault的linear batch-prefill路径。
不要照抄AITER警告中的sudo、时钟、NUMA、功率设置建议。

### 3.4 FlyDSL0.2.2控制环境

旧FP8绝对基线用0.2.2，主环境用0.3.1。本机另建0.2.2环境，保持Python3.11.11、
torch2.12.1+rocm7.2、HIP7.2.53211、triton-rocm3.7.1相同；只验证了imports，**没有性能数据**。
记录见 [environment-flydsl022.json](results/newmachine_mi325_20260907T035439Z/environment-flydsl022.json)。
需要控制实验时新建另一venv，不原地升级/降级主环境，也不要把要求FlyDSL0.3.1的AITER装进0.2.2控制环境。
不同编译器的结果单独输出，同一轮原/当前必须使用同一编译器。

## 4. 已完成的正确性、CPU与资源验证

### 4.1 JUnit事实（不相加）

以下结果均来自源机MI325的保存记录；新机器应重新测，不以相同收集数代替功能通过。

| 运行 | 通过 | 跳过 | 失败/错误 | 范围与证据 |
|---|---:|---:|---:|---|
| 初始CPU契约 | 78 | 0 | 0 | [cpu-initial.xml](results/newmachine_mi325_20260907T035439Z/cpu-initial.xml) |
| MI325新增工具CPU | 88 | 0 | 0 | [cpu-mi325.xml](results/newmachine_mi325_20260907T035439Z/cpu-mi325.xml) |
| BF16选定native | 205 | 46 | 0 | 含24项固定原版逐位对照；[bf16-942-initial.xml](results/newmachine_mi325_20260907T035439Z/bf16-942-initial.xml) |
| 初始完整native | **979** | **1310** | **0** | 2289项，3511.97秒，含AITER首次构建；[functional-all-initial.xml](results/newmachine_mi325_20260907T035439Z/functional-all-initial.xml) |
| 新gather/CK native子集 | 12 | 12 | 0 | 4项完整gather、8项gather+CK；另外12项需要gfx950；[gather-native-initial.xml](results/newmachine_mi325_20260907T035439Z/gather-native-initial.xml) |
| 新增graph/CU压力 | **14** | **4** | **0** | [stress-native-v2.xml](results/newmachine_mi325_20260907T035439Z/stress-native-v2.xml) |
| 旧报告导出/清单最终CPU | 15 | 0 | 0 | [offline-final-v2.xml](results/newmachine_mi325_20260907T035439Z/offline-final-v2.xml) |
| 最新指定参考+导出+清单CPU | **54** | **0** | **0** | 不导入ROCm；[contracts-final.xml](results/requested_references_20260907T062426Z/contracts-final.xml) |

初始完整1310个skip中1059个是架构不匹配：gfx950 SWA409、BF16 static327、persistent323。
其余是后端专属能力、unsupported布局、缺AITER匹配instance等，完整消息在JUnit，不能全部简写成“缺GPU”。

完整回归属于新增工具前的快照；stress和新的CPU工具后来独立测试通过，
**最终扩展CLI没有再做一次合并native回归**。中间的2326项收集记录也只是当时快照，
后续增加了测试；不要把任一旧收集数字写成最终全量通过数。

压力测试在2个独立stream上，每graph3次调用，每轮256次replay，4轮live metadata/cache变化；
每个native backend/dimension组合6144次attention，共8组=49,152次。
覆盖query重新分配、页映射、scale/cache内容变化、全空query、输出guard/LSE、编译cache稳定性，
另有6项BF16 page/D配置检查counter长度等于实际CU+1。
首次stress失败是PyTorch不支持FP8 `mul_cuda`的数据准备问题，已改为FP32计算再cast/copy回原tensor，
保持graph地址不变；[失败记录](results/newmachine_mi325_20260907T035439Z/stress-native.xml)仍保留，不删失败假装首轮全通过。

### 4.2 BF16资源与ISA

完整矩阵：**Dq128/192 × Dv128/192 × page32/64/128 × C/NC × LSE开关 × 两种Q scale = 96项**。
使用meta tensor和304 CTA / 305个int32 counter，不初始化GPU、不执行kernel。

- 全部Private=0、VGPR spill=0、scratch指令=0、AGPR=0。
- 84项SGPR spill也为0；12个NC/V128/LSE specialization各有2个SGPR→VGPR lane转存。
  这不是scratch，也不能称为所有寄存器完全零spill。
- 同编译器80/304 CTA的4项控制（D128/192×page32/64、V128、NC、无LSE），资源与指令流都相同。
- 所有96项304CTA资源计数与旧80CTA资源报告一致；runtime生产counter已读取真实CU，并非固定80。

| NC/V128/无LSE | 原版Private B/thread | 当前Private | 当前VGPR | 当前SGPR | LDS bytes |
|---|---:|---:|---:|---:|---:|
| D128/page32 | 56 | 0 | 234 | 106 | 16384 |
| D128/page64 | 104 | 0 | 246 | 106 | 16384 |
| D192/page32 | 120 | 0 | 256 | 106 | 24576 |
| D192/page64 | 176 | 0 | 252 | 106 | 24576 |

证据：[per-token 48项](results/newmachine_mi325_20260907T035439Z/resources-bf16-304-per-token.json)、
[per-tensor 48项](results/newmachine_mi325_20260907T035439Z/resources-bf16-304-per-tensor.json)、
[80CTA四项控制](results/newmachine_mi325_20260907T035439Z/resources-bf16-metadata80.json)、
[固定原版304CTA四项](results/newmachine_mi325_20260907T035439Z/resources-bf16-original-304.json)。
总计**104份保留ISA**，均已重新核验SHA。

原版4项通过读取其固定源码中的launch closure、meta参数进行编译，结果与当前分开；
没有执行原版meta kernel，也没有改原版源码。后续若要自动化该原版审计，应先复核相同参数契约。
原始IR大目录在源机临时空间，未作为必需交付；保留汇编是可移植证据。

## 5. MI325已有性能结果（旧参考，不代表最新要求达标）

这些数据比较**当前8wave BF16与固定main原8wave BF16**，并非dense LKG/V-global参考。
都是B1、HK1、Dv128、noncausal、per-token、无LSE、BF16-source输入；同一进程/编译器、
相同输入先独立FP32检查，再原/当前逐位相同和重复逐位检查。

| 配置 | 计时 | 原8wave µs | 当前 µs | 原TFLOPS | 当前TFLOPS | 当前延迟变化 |
|---|---|---:|---:|---:|---:|---:|
| D128/H16/Q10240/KV2583/page64 | profiler attention | 598.492 | 563.731 | 362.039 | **384.364** | −5.81% |
| D192/H16/Q10240/KV2583/page64 | profiler attention | 816.882 | 701.094 | 331.562 | **386.321** | −14.17% |
| D128/H1/Q=KV40960/page32 | 10buffer events | 2539.917 | 2450.320 | 338.197 | **350.564** | −3.53% |
| D192/H16/Q10240/KV2583/page32 | 10buffer events | 812.853 | 775.335 | 333.206 | **349.329** | −4.62% |

另有更早一次page32诊断：原816.714µs/331.630T，当前793.714µs/341.240T，延迟−2.82%。
两轮均没有在MI325上重现源MI308X的+10.95%回退，但**生产源码没改，不能宣称MI308X回退已经修复**。
没有外部独占预约、固定时钟/PTL控制证据；上述相对对照不升级成历史绝对基线验收。

原始结果：[四项同机对照](results/newmachine_mi325_20260907T035439Z/bf16-original-current.json)、
[首轮page32](results/newmachine_mi325_20260907T035439Z/bf16-page32-initial.json)。
含重复在内共10条candidate记录的[Markdown表](results/newmachine_mi325_20260907T035439Z/performance-summary.md)、
[CSV](results/newmachine_mi325_20260907T035439Z/performance-summary.csv)、
[JSON](results/newmachine_mi325_20260907T035439Z/performance-summary.json)包含双方TFLOPS，不只是当前值。

**本机没有可验收的FP8性能数据，也没有六项gather+linear的MI325计时结果。**
旧FP8报告为空且未完成；既有SWA/gather历史数字来自MI308/其它架构，不能填到MI325栏目。

## 6. 最新用户性能参考与九项对照计划

用户要求：“BF16以指定dense参考基本一致；FP8以指定prefill参考，不低于它的性能”。
入口是 [compare_requested_references.py](compare_requested_references.py)，而不是旧
[reproduce_baselines.py](reproduce_baselines.py)。详细单独说明见 [REQUESTED_REFERENCES.md](REQUESTED_REFERENCES.md)。

### 6.1 相对门槛

- BF16：工具暂将“基本一致”明确为 `current_us / reference_us <= 1.05`。
  **5%是助手为可执行验收设置的显式默认，不是用户给出的精确百分比**，由
  `--bf16-max-regression-pct`记录；最终需要与使用者确认，不能为通过测试偷偷调大。
- FP8：`current_us / reference_us <= 1.00`，没有允许回退百分比选项。
- 只有同机native、独立正确性、隔离、源码匹配、相同输入/计时均满足才可pass。
  计划、preflight成功、非native、未测、参考hash变化、numerical failure都不能判性能通过。
- gate只覆盖选中case；只跑短shape不代表默认长shape或所有支持形状通过。

### 6.2 默认输入与布局

**BF16**参考原main为H8、D128、M=N=256×CU，因此MI325默认 **Q=KV=77824、Hq=Hkv=8、Dq=Dv=128**。
它是full noncausal dense MHA，无GQA/causal/scales/异Dv接口。Q/K为head-major，
V预shuffle为`[H,N/8,D,8]`；当前版用相同逻辑值转换到paged形式，默认page32、identity页表、unit descales。
不能拿H16/HK1、D192/V128或KV2583冒充该dense参考的匹配shape。
参考import调用`pyhip.set_device()`，会选设备、设置默认device/seed；必须先限制可见物理GPU0，不能CPU直接import。

**FP8**参考实现没有performance main；配套 [test_pa_prefill.py](../pa_8wave/test_pa_prefill.py)
最后有效赋值是per-tensor（覆盖上面的per-token），默认 **B1/H16/HK1/D192/V128/Q=KV32768/page64/causal**。
输入是实际BF16随机数经FNUZ量化，双方使用同一Q/K/V/descales、反序页表、预分配输出。
该factory不接受旧native-BN64参考的`memory_mode`关键字；新工具使用专用adapter。
源码import会启用dump，工具隔离临时cwd并使双方在相同debug模式编译，不修改参考文件。

### 6.3 当前计划（全部尚未native计时）

| case | Q / KV | H / HK | Dq / Dv | page | C/NC | Q scale | buffer |
|---|---|---|---|---:|---|---|---:|
| `bf16_lkgv_native_default` | 77824 / 77824 | 8 / 8 | 128 / 128 | 32 | NC | unit scalar | 10 |
| `bf16_lkgv_q10240_k2560_p32` | 10240 / 2560 | 8 / 8 | 128 / 128 | 32 | NC | unit scalar | 10 |
| `bf16_lkgv_q10240_k2560_p64` | 10240 / 2560 | 8 / 8 | 128 / 128 | 64 | NC | unit scalar | 10 |
| `fp8_prefill_main` | 32768 / 32768 | 16 / 1 | 192 / 128 | 64 | C | per-tensor | 12 |
| `fp8_prefill_full_d128_per-token` | 10240 / 2560 | 8 / 1 | 128 / 128 | 64 | NC | per-token | 12 |
| `fp8_prefill_full_d128_per-tensor` | 10240 / 2560 | 8 / 1 | 128 / 128 | 64 | NC | per-tensor | 12 |
| `fp8_prefill_full_d192_per-token` | 10240 / 2560 | 16 / 1 | 192 / 128 | 64 | NC | per-token | 12 |
| `fp8_prefill_full_d192_per-tensor` | 10240 / 2560 | 16 / 1 | 192 / 128 | 64 | NC | per-tensor | 12 |
| `fp8_prefill_causal_d192_per-token` | 32768 / 32768 | 16 / 1 | 192 / 128 | 64 | C | per-token | 12 |

[最终CPU计划](results/requested_references_20260907T062426Z/plan-final.json)记录完整输入和FLOPs，
`records=[], acceptance_passed=false`。默认BF1677824单次attention为24,807,731,101,696 FLOPs；
多buffer、双实现、chunked FP32检查耗时和内存大，不能因为慢就静默缩短或少验buffer。

### 6.4 新旧计时口径

| 模式 | 协议/统计 | 使用时间 |
|---|---|---|
| 新BF16配对 | 原cudaPerf、10独立随机buffer、2warmup、每轮10样本取排序第6个；5轮后中位数 | 单event interval，含当前counter等辅助工作 |
| 新FP8配对 | cudaPerf、2warmup/10样本均值；按run_perftest复制规则最多12buffer、4e9字节上限；5轮后中位数 | 单event interval，含全部dispatch/gap |
| 旧普通profiler | 通常100共同warmup；5轮20warmup/100样本；丢首、1.5IQR均值、轮次中位数 | `attention_us`主kernel；另列`total_gpu_us` |
| 旧native FP8目标 | 1200共同warmup，其余同profiler；native-cast/per-token/无LSE | 旧MI308协议，不是新FP8参考 |
| 旧historical BF16 events | 10buffer、10warmup、50样本中位数；H3为3warmup/10样本 | `event_interval_us` |
| SWA gather+linear | 每轮20warmup/100samples，5轮；候选逐样本轮换，双层中位数 | 总路径一对event，不能相加两个独立均值 |

新比较双方逐样本交替先后顺序、同编译器同输入。5轮是配对扩展，明确标
`matched_pair_protocol=true`、`reference_standalone_protocol_exact=false`，不是原脚本逐字复刻。
cudaPerf自带的GPU delay在start event之前；不修改它或把delay算进attention FLOPs。
BF16原脚本分别排序µs与TFLOPS，两项中间值未必严格互逆；新工具统一由所报告延迟推导TFLOPS。
跨算法不要求原/当前逐位相同，但双方必须各自通过独立FP32和多次自身逐位重复。

原dense脚本的accuracy部分主要打印误差，关键allclose断言被注释；原FP8测试驱动也有
捕获异常后打印`accuracy unknown`的逻辑。**原脚本退出正常不等于正确性验收通过**。
新比较用本目录独立chunked FP32检查每套buffer，不继承这些吞异常行为。
长BF16默认shape若用原脚本一次物化完整`H×Q×KV` FP32 logits，单份就约180.5 GiB，
加softmax和其他buffer可能超内存；不要为绕开它减少测试规模或跳过正确性，应保持分块参考。

所有表必须同时给双方 **µs、TFLOPS、source/protocol、raw samples**：

$$
F=2H_q\sum_b N_{\mathrm{visible},b}(D_{qk}+D_v),\qquad
\mathrm{TFLOPS}=\frac{F}{t_{\mu s}\,10^6}.
$$

causal精确计可见三角形，不简单把full FLOPs除2；SWA W128最多129key，sink不是额外key/value。
单gather没有attention TFLOPS，填N/A而不是0；gather+linear以确实包含两段的总interval计算。
不能拿attention-only和event端到端混比，也不能给compile-only、missing instance填吞吐。

## 7. 新机继续测试的操作顺序

本节是后续操作者执行清单；写本文时只核验了磁盘证据，没有重新访问源机GPU。
先保留新机干净基线，再动代码。任何一步错误/超时都应停止相关队列，保留不完整报告。

### 7.1 独立输出与进程环境

```bash
# Git根；PY指向第3节创建的目标机解释器。
set -euo pipefail
PY="$HOME/.venvs/pyhip-mha-mi325/bin/python"
OUT="$PWD/tests/flydsl/mha/results/mi325_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$OUT"
export PY OUT
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0 GPU_ARCHS=gfx942 MAX_JOBS=4
export AITER_JIT_DIR="$HOME/.cache/pyhip-mha-mi325/aiter-jit-$(date -u +%Y%m%dT%H%M%SZ)"
export PATH="$(dirname "$PY"):$PATH"
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF
git rev-parse HEAD > "$OUT/commit-before.txt"
git status --short > "$OUT/worktree-before.txt"
git diff --binary HEAD > "$OUT/dirty-before.patch"
sha256sum tests/flydsl/mha/mha_pa_*.py > "$OUT/kernel-before.sha256"
```

**OUT必须绝对且每轮唯一**，参考import会临时切换cwd。新终端/后台任务可能丢失shell变量，
每次启动长命令先核对PY/OUT及可见设备，不假设继承上一次终端。
新比较入口目前要求三个visible变量都显式为0；非物理0映射必须先实现/验证BDF映射，不能移除防护绕过。

### 7.2 不导入ROCm的CPU阶段

以下三组CPU测试和新计划入口不依赖torch/FlyDSL import，可在健康尚不明时运行：

```bash
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  "$PY" -m pytest tests/flydsl/mha/test_requested_references.py \
  tests/flydsl/mha/test_export_performance.py tests/flydsl/mha/test_validation_manifest.py \
  -q --junitxml="$OUT/offline-contracts.xml"

HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  "$PY" tests/flydsl/mha/compare_requested_references.py --list-cases \
  --compute-units 304 --output "$OUT/requested-plan.json"

"$PY" tests/flydsl/mha/compare_requested_references.py --preflight-only \
  --output "$OUT/health-preflight.json"
```

preflight对amd-smi每次读取最多等待10秒；超时保存错误、清理自身reader、**不进入native import**。
健康读取成功也不算正确性或性能pass。返回失败时不得继续下面的GPU步骤。
性能仍需新机器的独占预约/进程检查，显存驻留worker不因0%利用率就当不存在。

健康通过后再做第3节实际import/GPU smoke，记录static/metric/process及环境路径。
普通amd-smi读取也建议设置外部timeout，避免一次监测变成无限阻塞。

### 7.3 当前源码先正确性，再最新参考对照

下列命令仅在健康新机执行，每个GPU workload串行。`--pytest-args`必须最后：

```bash
"$PY" tests/flydsl/mha/test_mha_pa.py --mode functional --backend bf16_942 \
  --pytest-args -x --tb=short --junitxml="$OUT/bf16-native.xml"

"$PY" tests/flydsl/mha/test_mha_pa.py --mode functional --backend fp8_942 fp8_942_register \
  --pytest-args -x --tb=short --junitxml="$OUT/fp8-native.xml"

"$PY" tests/flydsl/mha/test_mha_pa_swa.py --mode functional \
  --pytest-args -x --tb=short --junitxml="$OUT/swa-native.xml"

"$PY" -m pytest tests/flydsl/mha/test_mha_stress.py -q -x --tb=short \
  --junitxml="$OUT/stress-native.xml"

# 最终合并源码的全量回归；gfx950及unsupported按实际原因skip。
"$PY" tests/flydsl/mha/test_mha_pa.py --mode functional --suite all \
  --pytest-args --tb=short --junitxml="$OUT/functional-all.xml"
```

没有AITER时其功能用例可能skip；功能模式的`--aiter required/off`不负责强制/关闭pytest参考。
检查JUnit具体原因，不能把缺参考的skip说成AITER已验收。

先短case验证新adapter；短case不是最终默认shape的替代：

```bash
"$PY" tests/flydsl/mha/compare_requested_references.py --backend bf16 \
  --compute-units 304 --case bf16_lkgv_q10240_k2560_p32 \
  --output "$OUT/bf16-requested-short.json"

"$PY" tests/flydsl/mha/compare_requested_references.py --backend fp8 \
  --compute-units 304 --case fp8_prefill_full_d128_per-tensor \
  --output "$OUT/fp8-requested-short.json"

# 正式选中全部BF16三项与FP8六项；原/当前同机交替，分别留报告。
"$PY" tests/flydsl/mha/compare_requested_references.py --backend bf16 \
  --compute-units 304 --bf16-max-regression-pct 5 \
  --output "$OUT/bf16-requested-all.json"

"$PY" tests/flydsl/mha/compare_requested_references.py --backend fp8 \
  --compute-units 304 --output "$OUT/fp8-requested-all.json"
```

新native对照路径**尚未实机验证**，第一次在健康目标机跑时需确认dense/paged adapter、
大shape资源需求、buffer生命周期、参考源码import副作用和真实编译诊断。
输出`progress`记录reference加载、buffer准备、correctness候选、measure阶段。
只有初始health读取有工具内timeout，**native import/编译/launch/后续隔离读取没有全程看门狗**；
操作者应按shape预留时间并外设有界执行/监控。超时不得当skip通过，停止自己的任务，不能自动reset或改别人进程。

检查`complete`、非空`records`、每项`requested_reference_check`及`acceptance_passed`。
性能回退可以是`complete=true, acceptance_passed=false`；这表示测完但没达标，不应丢弃报告。

### 7.4 96项BF16资源复审

meta编译不launch kernel，但导入torch/FlyDSL可能探测损坏驱动，**仍放在健康检查之后**：

```bash
for mode in per-token per-tensor; do
  HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  FLYDSL_COMPILE_ONLY=1 FLYDSL_COMPILE_ARCH=gfx942 \
  "$PY" tests/flydsl/mha/compile_bf16_942.py \
    --dq 128 192 --dv 128 192 --page 32 64 128 --causal 0 1 --lse 0 1 \
    --mode "$mode" --workgroups 304 --require-no-scratch --retain-isa \
    --dump-root "$OUT/dump-bf16-$mode" --output "$OUT/resources-bf16-$mode.json"
done
```

该工具默认仍80CTA用于历史控制，MI325必须显式304，不根据被隐藏的GPU猜CU。
LLVM即使exit0并生成ISA也可能有register allocation错误，须检查native stderr与ISA；
不要仅凭“无编辑器错误”或Private=0合入。`--strict-resources`还要求零SGPR spill，
已知12项LSE及部分950 persistent不满足，不能通过删统计解决。

### 7.5 其他矩阵与gather（新相对门槛优先）

普通documented矩阵44项，gather默认6项；二者不是新参考9项计划。
documented的shape参数是过滤器，`--case`支持glob；自定义必须`--matrix custom`，不会自动缩Q换page。
H3是Q/KV `(63225,7)`、H=HK14，10buffer historical版本可能较重；保留完整输入、分组串行，不缩小冒充原case。

```bash
"$PY" tests/flydsl/mha/test_mha_pa.py --mode performance --suite all \
  --aiter auto --output "$OUT/performance44.json"

"$PY" tests/flydsl/mha/test_mha_pa_swa.py --mode performance --gather-linear \
  --aiter required --output "$OUT/gather-linear6.json"
```

gather六项是Q16384/H16/HK1/W128/sink，KV32768/65536/131072×D128/192；
比较direct、完整KV gather、prepared CK linear、每次gather+CK。
slot mapping与workspace分配在计时外，**每次总路径仍实际gather完整逻辑KV**，不裁掉SWA前缀。
检查`comparison_complete=true`且有全部6项；单interval包住两次launch和gap。
该模式不支持profiler、多buffer或历史baseline gate。

`--aiter auto`仅把缺依赖/缺instance记unavailable，数值或launch错误仍失败；
`--aiter required`在page64 direct-5D缺instance时按设计可能失败。
不要跑默认 [recheck_performance.py](recheck_performance.py) 队列：不加`--when-low`可能尝试PTL，
不适合MI325。`--allow-contention`只是诊断，不允许借此通过新性能门槛。

### 7.6 汇总并保留证据

```bash
# 只传实际有计时记录的JSON；计划/全空结果不能导出为吞吐通过。
"$PY" tests/flydsl/mha/export_performance.py \
  "$OUT/bf16-requested-all.json" "$OUT/fp8-requested-all.json" \
  --output "$OUT/throughput-summary"

# 完成全部写入后再生成新的清单，已存在路径会拒绝覆盖。
"$PY" tests/flydsl/mha/validation_manifest.py \
  --results "$OUT" --output "$OUT/manifest.json"
```

导出工具只读输入，保留source hash、partial/unavailable、每个candidate正确的gate，不给参考backend冒贴当前版pass。
[summarize_results.py](summarize_results.py)是早期历史汇总器，依赖旧临时ISA路径和旧cleanup断言，
不能直接拿来生成当前验证总结。

## 8. GPU停滞经过与排障边界

这是已观察到的时间线，不是已证明的根因：

1. 2026-09-07 05:22:29 UTC启动旧 `reproduce_baselines.py --backend fp8 --ptl current`。
   超过15分钟没有`BASELINE_ROUND`；报告保存环境但`records=[], complete=false`。
2. 本轮benchmark约200% CPU；新的amd-smi读取进程处于D状态，等待点
   `amddrm_sched_entity_flush`。没有足够日志区分原版/当前版、首次launch、correctness或warmup。
3. 核对PID/命令后仅终止本轮自己的benchmark，并停止本轮metric/process监测及阻塞reader。
   后接的BF16 `--four-wave` comparator未启动，不是已测失败。
4. 停止后单次15秒健康读取仍exit124；后来12秒读取和新工具10秒preflight也超时。
   最新保存的 [health-preflight.json](results/requested_references_20260907T062426Z/health-preflight.json)
   时间为06:34:09 UTC，发生在native import之前；自身reader已清理。
5. 一个隐藏可见设备的meta跨目标审计原型也在导入阶段进入driver wait，未生成ISA/报告；
   已停止并删除未验证原型，保留 [日志](results/newmachine_mi325_20260907T035439Z/preservation950-meta.log)。
6. `dmesg`权限不足，journal没有可访问的kernel条目，未尝试提权。没有GPU reset、PTL/clock/power修改。
   本轮创建的GPU任务/监测器最后检查均无残留；用户后来自行开的监控不是本轮清理对象。

**不应据此宣称**“304CU引发barrier死锁”“scalar cache饱和”“一定是某版kernel导致”。
设备/驱动/运行时/编译器/kernel之间的因果未确定，恢复后需小输入、阶段记录和单变量诊断。
新机器必须独立检查健康，不自动继承旧机器故障结论；本次写文档没有证明源机已经恢复。

ROCm启动还曾打印`Invalid processor info`（工程样片CPU），保留了stderr；当时native测试通过，
该警告与后续GPU阻塞之间无已验证因果关系，不能屏蔽后就当问题解决。

完整状态：[run-status.json](results/newmachine_mi325_20260907T035439Z/run-status.json)、
[旧FP8未完成报告](results/newmachine_mi325_20260907T035439Z/fp8-original-current-flydsl031.json)。

## 9. 开发时必须保留的契约

- **先重现再改**：保存commit/dirty diff、source SHA、环境、GPU策略、raw samples、ISA。
  小正确性→资源矩阵→固定原版对照→完整native→原计时协议。不得用零scratch自动推断性能更快。
- **精度不放宽**：BF16 O `rtol=atol=0.02`；FP8 O `0.1`；LSE沿用现有更严格阈值。
  不使用`equal_nan`吞掉错误，不吞launch异常。检查全部输出和guard，不只抽样。
- **BF16/gfx942**：保持8wave/BM256/BN32，per-call独立counter和graph-safe `[:1].fill_` seed。
  active KV>0，causal各sequence KV>=Q；零padding；概率/O round-half-up不能悄改RNE。
  `_uniform`只能标记真正uniform metadata，不能广播per-token Q scale。不要删wait/barrier追性能。
- **SWA**：64线程单wave，无生产gather/workspace/多wave fallback；W128含当前key共最多129key，
  sink分母仅加一次；masked/empty输出与LSE语义不变；gfx942 packed-V生命周期约束不能随意删。
- **gfx942 ISA**：不能用gfx950 direct buffer→LDS DMA、permlane swap或native BF16 pack。
  raw VGPR可能包含AGPR，分别记录VGPR/AGPR、Private、V/S spill，少量SGPR lane转存不隐藏。
- **缓存**：固定原BF16/BN32 factory的compiled cache不完整包含shape，跨workload清factory，
  但不能在计时或buffer轮换内清；参考import有dump/设备side-effect，保持临时cwd隔离。
- **已拒绝方向**：旧MI308 spill开发中无条件O rescale/过宽schedule fence曾让D128慢约30%，
  已恢复lazy rescale；D192/page32 LDS padding、原地址reuse、PV假依赖未改善，勿无证据重复合入。
  非法AGPR分配即使有ISA/exit0也被拒绝。细节见 [BF16_SPILL_FIX.md](BF16_SPILL_FIX.md)。
- **计时公平**：参考和候选同输入、同GPU/compiler/policy交替测；shape、page order、seed、quantization、
  Q scale、LSE、预分配、buffers、warmups、统计和辅助工作必须写清。不同路径/计时协议不能互填数据。

## 10. 不commit也能完整迁移

### 10.1 必须带走什么

- **完整pyhip工作树和Git对象**：core/helpers、四个kernel、最新工具/测试、本文和现有文档。
- 两个最新指定参考文件及FP8驱动，不能只拷贝本MHA子目录。
- 本机两轮独立证据目录：
  [MI325初始环境/结果](results/newmachine_mi325_20260907T035439Z/run-status.json)（目录约29MB），
  [最新参考计划/预检](results/requested_references_20260907T062426Z/manifest.json)（目录约120KB，后续大小以实际为准）。
- MHA原有历史results、保留ISA及 [本目录忽略规则](.gitignore)，该规则放行results内的`.s`。
  不要用“按Git忽略规则排除全部内容”的同步方法误丢汇编证据。
- AITER记录的commit/CK SHA，若目标无法联网还需独立带走相应源码/轮子；**不需要复制JIT二进制cache**。

不携带venv、Python/pytest cache、临时IR大目录、AITER JIT缓存或其他人的GPU进程状态。
**旧delivery-source.patch生成于新参考工具之前，不能单独恢复当前工作树**。
普通`git diff HEAD`也不包含未跟踪文件；bundle仅含Git对象，不含未提交改动。

### 10.2 推荐：整个仓库目录快照（保留.git）

以下在停止**本轮自己的写入任务**后，由操作者执行。归档放在仓库外；保留Git历史，
不需要commit/push，不使用`--exclude-vcs`或`--exclude-vcs-ignores`。

```bash
# 源机Git根
ROOT="$(git rev-parse --show-toplevel)"
NAME="$(basename "$ROOT")"
ARCHIVE="$HOME/pyhip-mi325-handoff-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -C "$(dirname "$ROOT")" \
  --exclude="$NAME/.venv" --exclude="$NAME/my_ir_dumps" \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.cache' \
  --exclude='*.egg-info' --exclude='.rocprofv3' \
  -czf "$ARCHIVE" "$NAME"
sha256sum "$ARCHIVE"
tar -tzf "$ARCHIVE" | grep -E 'README\.325\.md|compare_requested_references\.py|\.git/HEAD'
```

若使用Git worktree、alternates或partial clone，`.git`可能指向外部对象；先确认对象独立完整。
也可另建`git bundle --all`并verify，但仍需**当前工作树overlay**覆盖全部修改/未跟踪文件。
本机固定优化对象在remote-tracking ref中，不应照抄旧“只打包main和某个本地分支”的命令。

传输方式由使用者选择。本次仅编写交接文档，**没有实际创建/传输归档，也没有执行上述安装/GPU命令**。

目标机解压到**新目录**，不覆盖旧checkout：

```bash
# 先按源机输出核验归档SHA；再解压到新位置。
mkdir -p "$HOME/mi325-validation-next"
tar -xzf /path/to/pyhip-mi325-handoff.tar.gz -C "$HOME/mi325-validation-next"
cd "$HOME/mi325-validation-next/pyhip"  # 若源仓库目录名不同，相应调整
git rev-parse HEAD
git status --short
git cat-file -e 'ebc533488b3d6a55e1dd386da2cc8c04293432ab^{commit}'
git cat-file -e '23cc6d1e95b1611493e21232bef5d9962b7b73c9^{commit}'
sha256sum tests/flydsl/mha/mha_pa_*.py \
  tests/flydsl/test_attn_8wave_32x32_lkgv.py \
  tests/flydsl/pa_8wave/pa_prefill_8w32x32.py
```

固定对象/参考SHA缺失时标blocked，不能关SHA检查或换其它源码冒充。先跑不依赖这些对象的离线检查即可。

### 10.3 迁移后的证据路径与快照

旧JSON里的`isa`/`retained_isa`可能是源机绝对路径，**不要全局改写历史JSON**。
优先使用 [delivery-manifest.json](results/newmachine_mi325_20260907T035439Z/delivery-manifest.json)
里的相对`artifacts`/`isa`索引核验真正带走的文件；新版结果使用目标机自己的输出位置。
旧manifest的`source_files_sha256`是生成时快照，后来的文档/工具有合法改动，不能强行宣称全都匹配当前树。
最新参考的 [manifest.json](results/requested_references_20260907T062426Z/manifest.json)同理。
不要在目标机对旧报告直接运行会依赖源机绝对ISA路径的汇总器。

只验证迁移证据本身（不导入torch，不把源码快照重标为当前）：

```bash
"$PY" - <<'PY'
import hashlib, json
from pathlib import Path
base = Path('tests/flydsl/mha/results')
for folder, manifest in [
    ('newmachine_mi325_20260907T035439Z', 'delivery-manifest.json'),
    ('requested_references_20260907T062426Z', 'manifest.json'),
]:
    directory = base / folder
    data = json.loads((directory / manifest).read_text())
    for item in data['artifacts']:
        path = directory / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
    for item in data.get('isa', []):
        path = directory / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['isa_sha256'], path
    print(folder, 'artifact hashes verified; historical source snapshot not relabelled')
PY
```

## 11. 待办优先级与新机交回内容

### 优先级

- [ ] 新机确认MI325/gfx942/304CU、设备映射与运行时健康；保留只读PTL N/A/功率/时钟信息。
- [ ] 重建独立3.11环境，核验ROCm torch/Triton/FlyDSL和当前pyhip实际路径；需要时再装固定AITER。
- [ ] 原封不动先跑当前源码最终合并native回归、graph/metadata/边界压力，逐类记录skip原因。
- [ ] **优先完成最新BF16 dense三项与FP8 BN32六项参考对照**；双方µs/TFLOPS/ratio/raw samples齐全。
- [ ] 若新门槛未达标，在同机同编译器下逐项优化；小正确性、ISA资源、原版保留测试、全量native重跑。
- [ ] 补全44项普通性能和六项gather+linear长shape；不可缩短H3/长Q或漏掉总路径gather。
- [ ] 确认目标机AITER/CK可用性，missing instance单列；OPUS/gfx950须另有真实gfx950设备。
- [ ] 需要时分环境做FlyDSL0.2.2/0.3.1控制，或单独旧4wave来源调查；都不替代最新门槛。
- [ ] gfx950 static/persistent、causal merge、SWA/sink/pruning、QT/BN/stream/graph原生验收仍须在gfx950完成。
- [ ] 新结果用独立目录生成manifest和源patch/快照；遵守不commit/no-push指令。

### 交回报告必须回答

1. 本次实际commit、dirty范围、当前kernel/reference/helper SHA分别是什么？
2. Python/torch/HIP/FlyDSL/Triton/AITER/CK版本和实际import路径、GPU/CU/BDF/策略是否匹配？
3. 哪些是native通过、哪些只是CPU/compile、哪些skip/failed/blocked？最终CLI是否真的全量重跑？
4. BF16相对新dense参考是否在明确允许范围内？FP8是否不慢于指定BN32 FP8参考？不能用旧384T/386T替答。
5. D192/page32在本机是否有回退；如果没有，也不能推断MI308X已修复。
6. gather+linear的单interval是否真的包括每次gather+CK；完整KV、workspace/slot preparation是否在规定位置？
7. 原始时间样本、每条TFLOPS、protocol、ISA资源与SHA是否可移植复核？是否有完整独占/争用说明？
8. 是否发生任何异常/超时/授权设置，如何清理，硬件健康是否由真实证据确认？

## 12. 文件导航

- 本文件：当前MI325完整交接入口；[README.md](README.md)为通用接口目录。
- [CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md)：早期MI308来源、固定历史与完整开发契约，顶部有后续更新。
- [MI325_VALIDATION.md](MI325_VALIDATION.md)：本机环境/已完成运行和故障时间线。
- [REQUESTED_REFERENCES.md](REQUESTED_REFERENCES.md)：最新用户指定的相对性能参考。
- [COVERAGE.md](COVERAGE.md)、[PERFORMANCE_BASELINES.md](PERFORMANCE_BASELINES.md)、
  [BF16_SPILL_FIX.md](BF16_SPILL_FIX.md)：覆盖映射、旧性能口径和spill开发记录，不当作当前全量验收。
- [test_mha_pa.py](test_mha_pa.py)、[test_mha_pa_swa.py](test_mha_pa_swa.py)、
  [test_mha_stress.py](test_mha_stress.py)：native与压力测试。
- [compile_bf16_942.py](compile_bf16_942.py)、[test_bf16_spills.py](test_bf16_spills.py)：304CTA meta资源与固定原版回归。
- [compare_requested_references.py](compare_requested_references.py)、
  [test_requested_references.py](test_requested_references.py)：最新9项参考计划/相对gate/限时preflight；native路径待验证。
- [export_performance.py](export_performance.py)、[validation_manifest.py](validation_manifest.py)：纯离线吞吐和证据导出。

**本轮结论：环境、现有正确性、BF16资源与部分旧性能已有可核验证据；最新BF16/FP8性能目标未证明。
所有未完成事项保持未完成，不以换机、汇总或文档改名改变测试结论。**