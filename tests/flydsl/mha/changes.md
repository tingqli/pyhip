# 跨机器修改记录

后续涉及接口、测试口径、kernel或依赖的改动都在此追加：日期/起点SHA、修改范围、旧行为→新行为、
必须重测的场景及实际结果。不同机器保留各自结果，不覆盖历史JSON/ISA，也不将compile-only视为native通过。

**当前缓存注意事项（2026-09-08）**：已启用编译缓存，在排查问题时需要检查缓存是否出现问题。
已删除MHA私有持久缓存管理和额外开关，使用FlyDSL原生默认；下文旧私有缓存/关闭开关的记录仅为历史，不再适用。

## 2026-09-07：10-buffer测试、SWA gather+AITER、移除FP8 register

起点：`483cad8ebdf9ebeb7418871a3624b404f2790509`及本机已有未提交参考/空闲等待适配。
状态：**MI308完整重测完成**。不commit、不push。gfx950仍需在对应机器重测。

| 修改 | 旧行为 | 新行为 / 其他机器需关注 |
|---|---|---|
| `--buffers` | 单输入集 | 默认10，独立随机输入seed+i、独立Q/K/V/metadata/scale、每候选每buffer独立O；JSON记录地址与seed |
| `--run-count` | 5个单buffer event | 5轮，每轮遍历全部buffer；默认每候选50个event样本，最终取所有样本中位数 |
| `--repeat` | 同一buffer重复 | event内每次调用继续轮换buffer，按调用数归一化；不同候选使用相同buffer索引序列 |
| 正确性 | 每候选1个输出 | 每buffer独立FP32 O参考、有限值/容差检查及两次额外逐位重复；汇总acc取各buffer最大值 |
| SWA AITER参考 | 仅prepared linear CK | 保留`aiter`，增加`aiter_gather`：每次完整KV gather+CK，单event总时间；不裁掉SWA前缀 |
| gather workspace | 无 | 每buffer独立slot mapping与linear KV，预分配不计时，但每次缓存读取/gather均计时 |
| 带宽 | 逻辑Q/K/V/O | 保持已有定义；`aiter_gather`另加完整KV读+写，按其总event时间计算；不是实测HBM带宽 |
| FP8 register | 可选低性能分支 | 移除backend、全局K/V加载/输出及备用归约分支；仅LDS。`memory_mode='register'`明确拒绝，不静默fallback |
| 文档 | 单buffer数据及零散链接 | [README.308.md](README.308.md)各场景给复现命令；新数据全部10-buffer重测，旧单buffer报告只读保留 |

### 修改文件

- [test_mha_pa.py](test_mha_pa.py)：唯一入口、多buffer测量、同文件契约与native边界检查。
- [_references.py](_references.py)、[_gather.py](_gather.py)：测试专用完整gather+CK，不进入生产dispatch。
- [_testing.py](_testing.py)、[mha_pa_fp8_942.py](mha_pa_fp8_942.py)：移除FP8 register；LDS算术、同步和调度保持原顺序。
- BF16/gfx942、BF16/gfx950、单waveSWA三个生产kernel不修改；LSE仍不纳入本轮测试。

### 跨机必测

1. 主入口pytest：buffer轮换/索引覆盖/去别名、FP8两种scale与C/NC尾页、SWA gather乱序/空KV/live cache。
2. 原各组：BF16 basic六shape、FP8 basic四shape（仅LDS+BN32参考）、BF16 dense三shape、FP8 scalar三shape。
3. SWA Q16384/KV32768、65536、131072 × D128/192：direct、prepared AITER、每次gather+AITER均先验精度再测。
4. BF16 Q65/KV129、W0+sink SWA小shape；后续已移除独立repeat3专项，不再列为跨机必测。
5. gfx950另机跑static/persistent+SWA；本机MI308只将架构不匹配记skip，不宣称已验gfx950。

统一PTL策略以机器授权为准。本轮MI308保持Enabled/VECTOR,F8，无硬件setter/reset；
`--wait-idle`无截止时间等待无其他进程的稳定空闲窗口。不因10-buffer增加内存就静默缩小shape或buffer数。

### MI308实测结果

- 主入口pytest：16通过/2个gfx950跳过。其中2项CPU契约、14项native边界/gather；不是16项全GPU。
- 14份性能报告全部complete：68条候选、3400个event区间、3700次候选调用；10buffer全部独立、每轮覆盖0–9。
- 每候选10个buffer均通过FP32 O、有限值及两次额外逐位重复；repeat3原生验证了跨buffer9回绕到0/1。
- SWA新增参考每次执行完整KV gather+CK，实际2个dispatch；每buffer独立K/V workspace，字节分子额外计完整KV读写。
- FP8 LDS七项均快于指定BN32；BF16 dense三项仍慢6.30%–11.37%，不能把accuracy passed称为性能达标。
- 结果目录：[mi308_multibuffer_20260907T135348Z](results/mi308_multibuffer_20260907T135348Z)。
	各场景的复现命令和重测表见[README.308.md](README.308.md)第6章。
- 旧单buffer证据未修改；BF16/gfx942、BF16/gfx950、SWA三个kernel SHA未变。
	FP8删除register的LDS源级结构与旧LDS specialization核对一致，并通过本轮原生功能与性能重测。
- 本机约13GiB可用磁盘，显存约192GiB；其他机器需关注10组输入与参考workspace占用，不做自动降规模。

## 2026-09-07：删除CPU测试工具契约，澄清repeat3

- 按用户要求删除`test_multibuffer_rotation_contract`和`test_multibuffer_defaults_and_removed_register`，
	同步移除README中的CPU测试复现命令。主入口现仅收集16项GPU用例，MI308适用14项，gfx950 persistent2项跳过。
- 只删除测试，不删除生产/计时入口的参数校验、buffer独立性断言、逐buffer正确性或register拒绝逻辑。
- `--repeat 3`原生专项保留：一个event内连续调用同一候选3次并轮换buffer，按3次归一化；
	3条汇总结果对应direct、prepared AITER、gather+AITER，不是3次重复各出一条。
- 不改kernel或性能协议；旧JUnit、68条性能结果及原始样本保留，旧“16通过”不改写为新计数。
- 删除后回归：**14通过、2个gfx950跳过**。运行前GPU存在其他进程，持续等待至3次稳定空闲后执行，未终止他人任务。
	新证据：[mi308_gpu_tests_20260907T144448Z](results/mi308_gpu_tests_20260907T144448Z)；
	独立保存JUnit，不重新计时未变的性能场景。

	## 2026-09-07：解决重复测试耗时（起点4de089a）

	原因均有实测证据，不把GPU等待归咎于kernel：

	1. amd-smi Python子进程stdout被管道缓冲，3次空闲采样迟到约56–66秒；已设子进程PYTHONUNBUFFERED=1，
		不改5秒间隔/3次稳定空闲规则。GPU空闲时wall从66.27秒降至10.60秒，约1.1秒采样送达延迟。
	2. 原FLYDSL_RUNTIME_ENABLE_CACHE=0关闭跨进程磁盘缓存，首次编译在FP8场景占14.15/21.35秒。
		[_runner.py](_runner.py)新增用户私有、源码/编译器/架构指纹隔离缓存，CLI与pytest默认开启。
		`--no-compile-cache`或`PYHIP_MHA_COMPILE_CACHE=0`禁用；只缓存编译产物，不缓存正确性。
	3. [_testing.py](_testing.py)的FP32 oracle新增with_lse选项（保持原默认True），主测试显式False；
		不再执行无用LSE归约，ragged逻辑KV不再每sequence重新cast整缓存；12种dtype/scale/window对照原O逐位相同。
	4. [test_mha_pa.py](test_mha_pa.py)增加阶段wall_time_s、buffer准备/验证/轮次进度；默认精简日志，
		--verbose-runs恢复全部打印，JSON仍保留所有样本。显式环境查询15秒超时，避免单次读取无限卡住。

	验证：完整pytest冷79.85秒→热7.16秒（两次14通过/2gfx950跳过）；FP8+BN32 cProfile总时长
	修改前21.35秒→热缓存6.996秒（仍10buffer、每候选50样本、逐buffer精度和重复检查）；
	SWA长shape direct/prepared/gather三路径冷/热均通过，gather仍每次真实执行。
	证据：[latency_analysis_20260907T145541Z](results/latency_analysis_20260907T145541Z)。
	这些是测试运行速度，不是kernel TFLOPS提升；所有生产kernel未修改，旧性能报告不覆盖、不重标。
	首次编译/新shape、输入准备、完整FP32参考、真正GPU争用仍需时间。没有减少默认buffer/rounds或放宽精度。

## 2026-09-07：移除repeat3专项，区分pytest与smoke计数

- 删除README当前清单中的repeat3专项行、解释段和`bench repeat3 ... --repeat 3`复现命令。
	该专项原本是独立CLI调用，不是pytest函数，代码中没有需要删除的`test_repeat3`。
- 保留通用`--repeat`参数（默认1）及底层轮换逻辑；只移除专项，不改kernel、buffer数、精度或常规计时协议。
- 当前保留场景为65条候选/3250个event；旧14份报告中的repeat3三条结果及150个event只作历史保留，
	不修改旧JSON/manifest或把旧68条/3400个event记录重标为新运行。
- README 6.4拆成独立两类：pytest为16个参数化用例（MI308 14通过/2跳过）；
	`smoke-bf16`为D2×候选2=4条Full BF16结果，`smoke-swa-w0`为D2×候选3=6条BF16 SWA结果。
	smoke不是pytest的子项，命令名只是输出标签。
- 本次仅改文档和复现清单，无需重跑未变的GPU代码；验证删除后的命令清单和历史证据完整性。

## 2026-09-07：寻找空闲GPU＋单进程整轮重测

- `--wait-idle`不再只等物理GPU0：与`--gpu auto`一样扫描所有合格GPU；可用`--gpu-pool`限定授权设备池，
	`--gpu N`固定物理SMI卡。后台监测一次读取全部候选利用率，每轮一次读取全部候选进程，避免逐卡启动SMI。
- `--required-ptl VECTOR,F8`仅筛选现有Enabled/VECTOR,F8策略，不设置硬件；架构不匹配提前排除。
	全部忙才无限等待，仍需每卡连续3个quiet样本，无驻留进程。选卡前后进程及策略复查，不修改其他用户任务。
- sysfs unique_id构建ROCr UUID，只向HIP暴露该卡为逻辑0；实际torch属性BDF必须匹配，否则失败。
	所有结果记录physical_gpu/BDF/UUID/选择时间，旧物理卡结果不冒充新卡。用户级文件锁防止自己的并发测试选同一卡。
- pytest支持`PYHIP_MHA_GPU=auto`、`PYHIP_MHA_REQUIRED_PTL=VECTOR,F8`，同一测试进程先选卡再初始化GPU。
- `--retest-all`同进程/同设备重测13组保留场景，复用imports/编译缓存，减少重复选卡；每组仍有独立10buffer/正确性/样本。
	一旦数值/资源/占用错误即停止并保留partial清单，不缩shape、不迁移半轮到另一个GPU。
- 真实GPU1（候选池1–7）FP8小shape映射/正确性通过；两次整轮自动选GPU0，分别选择11.92/11.94秒、执行215.78/119.98秒。
	**各13组65候选3250样本均完成且输出正确**；pytest冷93.31秒/热18.63秒（包含选卡），两次14通过/2跳过。
- 验证所有10buffer地址独立、每轮覆盖、gather20workspace地址及双dispatch、PTL与真实BDF一致。
	生产kernel未修改；BF16相对门槛未全达标，不把正确性通过变成性能全达标。
- 证据：[mi308_autogpu_20260907T152916Z](results/mi308_autogpu_20260907T152916Z)。
	其他机器需确认SMI的完整设备可见性、sysfs unique_id与ROCr UUID；调度器限卡必须显式传入对应候选池。

## 2026-09-07：小shape仅测试我们的kernel

- `--retest-all`中`smoke-bf16`和`smoke-swa-w0`显式使用`--aiter off --requested-reference off`；
  前者只运行BF16 8wave，后者只运行W0+sink单wave SWA，各D128/192两条结果。
- README 6.4同步精简复现命令与候选说明；仍有10个独立buffer、每候选50个event、FP32 O和重复逐位检查。
- 不改通用CLI参考默认值，也不改§6.1–6.3长shape的AITER、指定BF16/FP8参考及SWA gather+AITER对照。
- 整轮仍13组，候选由65条减为59条，默认event样本由3250减为2950；旧报告只读保留。
- 本次只需重测两个受影响小shape并校验整轮清单，其余kernel、参数与计时协议不变。
- 已按`retest_all`实际参数重测：两个场景各D128/192共4条自有kernel结果、200个event样本全部通过，
	保留10buffer/FP32 O/重复逐位检查。验证时将外部参考工厂替换为一旦调用即报错，确认均未调用且未导入AITER。
	证据：[mi308_own_smoke_20260907T154525Z](results/mi308_own_smoke_20260907T154525Z)。

## 2026-09-08：全部重测，保存每项耗时

- 已完整重跑当前13组/26个计时shape/59候选，2950个event样本全部通过；GPU pytest重新执行16项，14通过/2个gfx950跳过。
- 成功性能进程176.680秒（选卡+激活12.099秒、suite161.694秒），pytest进程94.450秒（suite93.282秒，含选卡）；
	两次成功进程合计271.130秒，不包含之前排障/中断尝试。缓存开启但仅部分命中，不称纯冷或全热。
- [本轮完整报告](results/mi308_full_timed_20260907T230053Z/SUMMARY.md)保存13组wall time、26个shape阶段、
	59候选event中位数和16项pytest setup/call/teardown时间，并附原始JSON/JUnit/日志、CSV及只读汇总审计脚本。
- 首次尝试发现选卡解析缺陷：AMD SMI对每GPU分别取`int(time.time())`，同一轮跨秒合法，不能要求全卡时间戳相等。
	`_hardware`改为按请求顺序组成完整扫描、每卡时间戳递增；跨秒可通过，缺失/重复/陈旧/截断/半轮表头仍失败关闭。
	原生选卡及完整GPU复测通过；三次稳定空闲、进程检查、PTL筛选/BDF核验不变。其他平台须确认SMI按请求顺序输出。
- 第一次选卡失败7.64秒及一次终端操作造成的中断报告均保留且不计为完整通过；成功目录从头重跑，不拼接旧数据。
- 未更改生产kernel、event计时范围、10buffer/FP32/逐位检查，也未恢复CPU测试或已移除的repeat3/register场景。

## 2026-09-08：删除冗余缓存管理，恢复FlyDSL原生默认

- 已检查安装的FlyDSL 0.3.1：原生默认开启内存/磁盘编译缓存，管理源码/编译器/目标/参数及选项key、进程锁与原子写入。
	MHA私有指纹目录重复了已有机制；正常运行无需关闭编译缓存或再实现一层私有持久缓存。
- 删除`configure_compile_cache`、私有目录/指纹/初始化逻辑、`--compile-cache`/`--no-compile-cache`、
	`PYHIP_MHA_COMPILE_CACHE`及对应报告字段/初始化计时；pytest fixture仅保留GPU选择，阶段wall time与源码SHA仍保留。
- 删除gfx942 BF16、指定BF16/FP8参考导入时的缓存禁用设置；FP8参考自动`dump_ir`也会间接禁用缓存，一并删除。
	未改kernel运算/调度及正常`functools.cache`、JIT复用；两条参考的固定源码SHA随这些导入设置变更更新，旧报告SHA不改写。
- README与平台文档加入：**已启用编译缓存，在排查问题时需要检查缓存是否出现问题**；
	检查实际版本、缓存命中/失效、旧shell的FlyDSL环境覆盖、升级后兼容性。正常不关缓存、不跳过正确性。
- 旧私有缓存文件及历史实测证据原样保留，不再被当前框架选择，也不强制删除用户磁盘内容。
- 原生验证：两个独立进程分别完成13组59候选/2950样本，全部通过；第二轮FlyDSL原生get **29次命中、0次未命中**，
	两轮环境默认开启且目录未变，参考导入不再覆盖缓存。完整性能进程225.280→134.240秒（含选卡/启动）。
- GPU pytest两轮均14通过/2个gfx950跳过，suite91.041→18.834秒；10buffer、全部FP32/逐位与样本、gather均复核。
	[证据与逐项耗时](results/mi308_native_cache_20260907T234026Z/SUMMARY.md)。gfx950原生默认路径仍需跨机验证。

## 2026-09-08：删除README第9、10章，再次全部计时

- 删除[README.308.md](README.308.md)第9、10章及对应交叉引用；保留第7章原生缓存排障注意事项。历史结果文件不删除。
- 使用原生默认缓存重新完整运行13组/26个shape/59候选，2950个event全部通过；GPU pytest重新运行，14通过/2个gfx950跳过。
- 性能进程134.600秒（含选卡启动；suite119.724秒、选卡+激活11.925秒），pytest进程19.920秒（suite18.769秒，含选卡setup）。
	两个成功进程合计154.520秒，不是包含整理文档的会话总耗时。
- [独立计时报告](results/mi308_retest_timed_20260907T235346Z/SUMMARY.md)记录逐组wall time、26shape阶段、59候选GPU event与16项pytest耗时，附CSV/审计/原始日志。
	不修改kernel、计时协议或缓存逻辑，不清理/关闭/切换缓存，不在README重新增加第9、10章。

## 2026-09-08：删除独立gather测试

- 删除`test_full_gather_live_cache`及D128/192两个参数化用例；不删除FP8 LDS边界测试。
- 主入口pytest由16项减为14项，MI308重新运行**12通过、2个gfx950 persistent跳过**，suite耗时18.32秒。
	[JUnit](results/mi308_remove_gather_test_20260908T001506Z/tests.xml)、[日志与逐项耗时](results/mi308_remove_gather_test_20260908T001506Z/pytest.log)。
- 保留测试参考使用的gather实现、`aiter_gather`路径、运行时完整KV校验及双dispatch检查；59候选/2950样本的性能矩阵不变，本次未重测该未修改矩阵。
- README更新当前pytest范围，旧测试日志、结果与历史计数原样保留；不修改kernel、缓存或正确性协议。

## 2026-09-08：简化pytest用例函数名

- `test_page_boundaries` → `test_bf16_mha`；`test_fp8_lds_page_boundaries` → `test_fp8_mha`；
	`test_swa_page_boundaries` → `test_swa`，直接突出BF16 MHA、FP8 MHA与SWA。
- 仅重命名函数，参数化、用例内容及架构跳过条件不变；平台文档同步当前名称，历史JUnit/日志保留旧名称。
- 已验证pytest收集：`test_bf16_mha` 4项、`test_fp8_mha` 8项、`test_swa` 2项，共14项；本次仅命名修改，未重复执行GPU测试。

## 2026-09-08：性能选择改为三个显式集合，删除BF16/FP8外部指定参考

- 增加`BF16_MHA_PERF_CASES`、`FP8_MHA_PERF_CASES`、`SWA_PERF_CASES`，对应
	`test_perf_bf16_mha`、`test_perf_fp8_mha`、`test_perf_swa`；CLI与pytest共享集合和三类候选规则。
- CLI改为`--suite all|bf16-mha|fp8-mha|swa`、`--case ID ...`及GPU-free `--list`；
	删除混合backend/preset、shape猜测、参考auto/on/off探测和旧`retest_all`拼接CLI再解析的流程。
	自定义输入直接编辑显式Workload；保留公共buffers/rounds/warmup/repeat、设备选择及报告参数。
- BF16小shape仅自有，其余+AITER；原三个H8/HK8、P32/P64的shape保留，指定dense适配删除。
	后续按用户要求删除FP8指定BN32适配、动态加载与SHA依赖；FP8七个场景仅自有LDS，不导入外部FP8参考。
	独立FP32 O仍用于全部buffer正确性检查，不属于被删除的性能对照。
- SWA小shape仅自有，长shape+preparedCK+每次完整gatherCK；原basic中KV128K重复执行移到SWA集合并显式命名。
	声明的AITER缺失或失败时直接报错，不再静默减少对照候选。
- 最终gfx942集合：BF16 9case/16候选，FP8 7case/7候选，SWA 10case/26候选，合计**26case/49候选/2450样本**。
	与旧59候选多重集合逐项核对，仅去掉3条BF16及7条FP8指定参考，全部输入/seed及重复执行保留。
- 默认pytest仅14个功能参数项，26个性能项默认取消选择；`--mha-perf -m mha_perf`显式启用性能，
	可用`--mha-output`保存每case报告。无CPU工具测试，无独立gather测试。
- 新CLI和性能pytest分别完整通过26case/49候选/2450样本；功能pytest12通过/2个gfx950 persistent跳过。
	CLI进程107.62秒（选卡+激活11.930秒，执行93.170秒）；功能pytest suite18.051秒/进程19.10秒；
	性能pytest suite191.063秒/进程192.55秒。不同报告/环境查询开销不能当作kernel性能差异。
	[完整验证、逐项计时与审计](results/mi308_three_suites_no_fp8ref_20260908T011740Z/SUMMARY.md)。
- 中间仅删除BF16 dense的56候选结果保留在[历史报告](results/mi308_three_suites_20260908T004045Z/all.json)，不重标为49候选。
	原参考源文件作为独立实验保留，不删除历史文件。
- 四个生产kernel、公共输入/oracle、gather实现和硬件选择SHA均未改变；原生编译缓存保持开启。
	gfx950 Full/Causal static/persistent和SWA static/persistent/单wave的显式规则保留并静态核对，尚未做新gfx950原生验证。

## 2026-09-08：文档命令单行化，说明case筛选

- 三份入口/平台README中的测试命令改为单行，环境准备仍独立列出；不改变命令参数或测试逻辑。
- 明确`--suite`选类别，`--case`可选地筛选预定义场景；省略则运行所选集合全部场景，不改变shape或参考候选。
	增加`bf16-full-d128`单场景及`--list`查询示例，说明精确ID、重复限制、集合顺序及与pytest选择的区别。

## 2026-09-08：删除正常测试流程打印，保留性能与错误诊断

- 删除PREPARED/VALIDATED/MEASURED/WALL_TIME、环境起始行、AITER entry/dispatch、选卡采样/选中和架构skip流程打印；
	成功accuracy不再额外打印。性能汇总表及`--verbose-runs`逐样本acc/时间/TFLOPS/带宽保持原样，`--list`/`--help`不变。
- 不修改计算、计时循环、buffer/精度协议或选卡规则；详细阶段wall time、dispatch、skip原因和设备信息仍写JSON，选卡采样仍写idle JSONL。
- AITER通过正式日志级别默认ERROR（显式环境覆盖仍可用于调试），不全局重定向测试stdout。
	ROCm Kineto的USDT进度级别高于ERROR，实测ERROR级别仍打印启动/停止；仅在未计时的dispatch检查期间过滤该精确模式。
	单周期profiler提示单独过滤，其他异常警告/错误保留；dispatch检查抛错时回放完整原生stderr。无需关闭profiling或缓存。
- 注入验证确认数值错误/NaN仍失败、stderr错误仍可见、失败原生输出完整回放；完整CLI26case/49候选/2450样本通过，stdout只有性能表。
	verbose单BF16场景100条性能样本保留；功能pytest即使`-s`也无框架流程打印，12通过/2跳过（18.074秒）。
	[独立结果记录](results/mi308_quiet_20260908T020218Z/SUMMARY.md)。正常运行中的一条ROCTracer重复flow警告未隐藏，不当作流程日志删除。

## 2026-09-08：完整性能数据前置，明确SWA重复执行

- 核对D128/D192两对KV128K用例：除ID外，全部Workload字段、seed、smoke/架构限制、候选与检查相同；repeat标签不增加独立shape覆盖。
	本次仅解释并标注重复，未删除原先明确保留的两个用例；当前26case/49候选/2450样本不变。
- MI308文档将完整性能数据提前到第2节，按BF16 16条、FP8 7条、SWA 26条列出全部49行，包含acc/延迟/TFLOPS/逻辑GB/s及重复项。
	数值全部来自原第8节同一份已完成CLI JSON，保留三位小数；本次不重测、不拼接轮次、不删慢值。
- 更新后续章节编号，保留单行命令和缓存/输出注意事项；不新增MI308文档引用，删除以外部结果链接代替数据的示例/引用段。
	原始报告文件及历史证据不修改。

## 2026-09-08：删除重复SWA用例，核对配置与gather依赖

- 删除`swa-kv131072-repeat-d128/d192`两个重复参数项；SWA为8case/20候选，gfx942全集24case/43候选/2150样本，独立shape覆盖不变。
- `conftest`注册pytest性能开关、输出选项及marker并保证默认仅功能测试；gather实现仍供SWA的`aiter_gather`使用。两者有明确依赖，本次不直接删除。
- 完整CLI及性能pytest均通过24case/43候选/2150样本；合并pytest36通过/2个gfx950 persistent跳过，默认/显式收集分别14/24项。
	CLI进程103.86秒（选卡11.958秒，执行89.258秒），合并pytest suite111.290秒/进程112.73秒；[独立记录](results/mi308_dedup_20260908T022007Z/SUMMARY.md)。
- MI308第2节保持完整43行新性能数据并更新所有当前数量，无新增MI308引用；旧原始报告保留，gfx950只核对规则不冒充原生实测。

## 2026-09-08：全部测试辅助代码合入主文件

- 参数集合、Backend/Case、输入/量化/FP32 O、gather/AITER、只读GPU选择、环境/报告与计时全部合入主测试文件并按职责分区。
	删除6个旧辅助模块和本目录conftest，共7个文件；4个生产kernel及共用DSL适配的SHA不变，包初始化保留。
- 删除未使用的旧单卡等待、Case.pack、BACKENDS/single_dispatch、历史量化前padding、未使用LSE及unchecked测试分支。
	真gather、AITER声明参考、FP32 O、10buffer/5轮、逐位检查、逻辑字节模型及静默流程全部保留。
- pytest无需插件：默认函数`__test__`不收集性能项，`PYHIP_MHA_PERF=1`显式启用，标准`-k test_perf_`仅选性能；
	`PYHIP_MHA_OUTPUT`替代旧输出选项，原自定义pytest开关/marker删除。CLI参数不变，文档命令保持单行。
- 合并前后12种dtype/scale/window输入与FP32 O逐位一致；参数集合、核心搬迁函数AST和生产kernel/DSL SHA已核对。
- 单文件CLI与性能pytest各自24case/43候选/2150样本全部通过；功能pytest12通过/2个gfx950跳过。
	CLI进程103.90秒，功能pytest18.146秒（进程19.27秒），性能pytest110.409秒（进程111.91秒）。
	[验证记录](results/mi308_singlefile_20260908T023547Z/SUMMARY.md)。
- MI308第2节更新完整43行本轮CLI数据，无新增引用链接；删除对已移除辅助文件的活动文档引用。
	历史报告/脚本仅保留原证据，不为旧导入接口留下无用兼容壳；gfx950原生测试仍待对应机器完成。