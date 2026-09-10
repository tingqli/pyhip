# MoE down优化TODO

## 当前优先级更新（2026-09-07）

- [x] **当前源码1K–32K最终path矩阵重测**：七case×六Batch全部候选24轮，13个晋级/边界点独立48轮，
	共13份运行/71712个event，全部逐轮合同与入场/设置后/退出门禁通过。新推荐普通8x1 6点、compact 12点、
	1x4 17点、1x8 2点、default 5点，相对9/6改选8点；Hy3直接M64/pad128纳入但Full仍未晋级。
	[当前42点矩阵](MAIN_MERGE_PERFORMANCE_REPORT.md#1k-32k最终path矩阵) · [独立审计](results/final_matrix_20260907/matrix_summary.json)。
	生产sum仍64、自动selector/生产预设不变；原9/6及9/5矩阵保留历史，已有暂存改动未触碰。

- [x] **按要求删除K192/K320旧8x1分支并导出七K新ATT**：先完成四shape普通/compact的24＋48轮完整链，
	全部新旧配对三phase正IQR；K192仅整192，K320仅128+192，入口默认已切，参数128仅兼容别名。
	59个普通构建＋28个compact随机/graph、208CPU及79API/整链通过；七份fresh ATT UI已复制根目录并逐hash核验。
	[删除与七UI验收](BK192_PROMOTION_20260907.md) · [最终汇总](results/bk192_promotion_20260907/final_audited.json)。

- [x] **优先K192与K320的普通Down性能数据已完成**：分别比较K192的128+64/192、K320的128+128+64/128+192；
	先测N2048/PTPC、N4096/per-tensor及B16K/32K的Down配对，24轮起步、边界48轮，之后根据结果决定Combined/Full或fresh ATT。
	固定10-buffer、padding128/nt及相同任务策略，使用PTL Enabled/VECTOR,F8；每次性能入场要求全机busy≤5%、VRAM≤20%，结束恢复硬件原状态。
	先核验正确性和源码/配置/ELF/ISA身份，优先复用匹配产物；同时报告时延和有效TFLOPS（F=2×B×TopK×N×K），不以busy代替墙钟性能。
- 完成上述优先测试后，再按下节的**CU拆分性能 → 零满块直接M64对照 → 真实分布selector → sum线程实验**顺序继续。
- 最新续跑：routing独立48轮7份/10点全部通过门禁，Qwen35三分布补测完成；
	Hy3 skew 8K、Qwen397三分布16K、Qwen35 uniform 16K的compact对全部候选三phase共同胜，其他5点仍有边界。
	旧Qwen35负载风险整份隔离不变。sum线程独立24＋48轮四点完成：256线程sum/Combined均正，Hy3 Full正、Qwen35 Full跨零；
	生产sum64默认和自动selector未改，所有接管运行配置恢复。

[本轮性能报告](TODO_PERFORMANCE_20260907.md) · [独立复算汇总](results/todo_20260907/performance_summary.json)。

[零满块与routing续跑报告](ZERO_M64_ROUTING_20260907.md) · [分布证据与风险隔离](results/todo_20260907/routing_summary.json)。

[sum线程实验报告](SUM_THREADS_20260907.md) · [6912样本独立审计](results/todo_20260907/sum_threads/audited.json)。

### 本轮执行进度：普通Down/CU、routing48、sum线程指定范围完成

- [x] 新增[双K性能入口](benchmark_tail192.py)，复用零JIT ELF调用和逐轮正确性，固定10-buffer/24→48轮及独立PTL接管/恢复。
- [x] 重建当前源码身份不匹配的两份K192原分块控制，各5次随机检查通过，ISA与原控制严格相同；其余六份当前身份产物直接复用。
- [x] 四组N2048/PTPC、N4096/per-tensor的八份产物全部通过源码/快照/ELF/ISA和资源核验；10项CPU准备/拒绝合同测试通过。
- [x] 四组B16K/32K完成24＋48轮：K192配对时延降低10.028%～11.843%，K320为7.366%～8.723%，8点IQR均正；
  K320/N2048/32K确认47/48胜，其余48/48，不删除负向轮次。共2304个event样本，旧拒绝记录保留。
- [x] K192/K320新分块的普通/compact Combined/Full对照及七K fresh ATT已完成；8份17280event全部三phase正IQR，旧分支已删除，新实现成为唯一默认。
- [x] 35项CPU性能/门禁/路径/统计测试通过；已测driver快照冻结，源码/ELF/ISA/全部样本和硬件状态独立复算通过。

[准备与阻塞清单](results/todo_20260907/tail192/preflight.json) · [首组门禁记录](results/todo_20260907/tail192/k192_n4096_abba24.json)。

## 后续执行顺序（2026-09-07）

在上述K192/K320普通Down优先测试完成后，按以下顺序推进；勾选项表示所列测试范围已完成，
不表示所有测试点均通过性能晋级门槛。未勾选项仍是待办。

1. [x] **CU拆分性能（MI308X当前范围完成）**：在空闲GPU、PTL Enabled/VECTOR,F8下，比较“旧策略＋旧capacity / 旧策略＋新capacity / CU拆分＋新capacity”，
	同时测Down、Combined、Full，先ABBA24、边界ABBA48。优先Hy3 8K、Qwen397两种K的16K，
	已覆盖零满块、全满及不触发拆分的9点；Qwen397两K16K三phase正IQR，Full分别+1.956%/+1.068%。
  Hy3 8K与Qwen35 K512 16K仅Down/Combined确认，Full仍跨零；阈值不调整，304 CU硬件另需实测。
2. [x] **零满块直接M64同配置对照（测试完成，不代表Full晋级）**：Hy3 1K/2K，直接`1x4_64x256`＋padding128B＋nt store，
	与compact固定相同BM64 metadata、sorting/gateup/量化、输入buffer及`sorted_sum()`，
	分离构表、空满块launch和descriptor寻址的额外成本。此前独立1x4为padding0，不能据其差异判断此优化。
	此项先做独立性能对照，不为识别`F=0`增加GPU count回读。
	48轮：Down配对降低6.971%/4.635%，Combined降低5.001%/3.404%，均48/48胜；Full两点IQR跨零，不自动切换。
	Down绝对中位差约7.32µs是合计路径开销，构表/空full/descriptor各自的独立拆分计时仍待做。
3. [ ] **真实分布selector**：依据实际每expert行数、满块/尾块比例、CU余数与构表/launch成本，
	用真实或偏斜routing性能验证直接M64、普通8x1、compact的选择规则，不只使用单调Batch阈值。
	需设计无额外GPU→CPU同步的决策方式；仅GPU guard跳过数学仍有launch开销，不等价于绕过compact。
	候选必须满足Down、Combined、Full的配对IQR门槛，不能预设多数点选compact。
	已完成Hy3 1K/2K/8K、Qwen397 K512 16K的balanced/uniform/Zipf-skew合成分布24轮，共12个可用点；
	已完成Hy3 skew 8K、Qwen397三分布及Qwen35三分布8K/16K的7份独立48轮，共10点11808个event；
	前四点与Qwen35 uniform 16K的compact对全部候选三phase共同胜，其余5点Full或default竞争仍有边界。
	Qwen35旧balanced风险与uniform拒绝保留，不与新合格样本拼接。不是生产routing trace，不宣称在线selector已实现。
	待多seed/生产trace及host现成hint或device侧无CPU同步接入；计时外`.cpu()`统计仅作证据，不能搬进在线决策。
4. [x] **sum线程实验（指定两宽度/四点测试完成，生产默认未切）**：比较`sorted_sum()`的64/128/256线程按N分工，优先N4096/TopK9和N2048/TopK8，
	保持每元素FP32 TopK累加顺序；固定stride、loc、地址和前驱缓存条件，同时验证独立sum、Combined、Full。
	测试候选仅在tests，生产64指令/资源严格对照、随机/graph通过；24＋48轮/10-buffer/四phase共6912event，34CPU通过。
	256相对64确认sum降低8.622%～14.658%、Combined降低1.794%～5.496%；Hy3 16K/32K Full正IQR，Qwen35 Full跨零。
	[结果与限制](SUM_THREADS_20260907.md)：不直接泛化到compact/既有最佳路径，生产64不变；后续接入需实际路径与更多routing验证。

## K192整块BK192（2026-09-07，独立于上述四步待办）

- [x] 显式`tile_k=192`，BM256/BN128/8wave，真实两拍48+48 MFMA/wave，不padding K；默认128+64不变。
- [x] native16B+8B预取、双B槽48KiB＋CShuffle16KiB，首尾/1N/2N循环、scale/output退休与等待账本重新实现。
- [x] 26个普通BK192构建86次随机检查；14个compact配置70次候选检查（含42次graph重放），真实满尾覆盖。
- [x] 实测N2048/PTPC 210 VGPR、Hy3 N4096/per-tensor 176 VGPR，LDS65536B且零spill；默认和tasks/tail ISA隔离验证。
- [x] 114项CPU/API/调度回归，两个生产宽度的零JIT ELF复用正确性通过。
- [x] **BK192普通Down性能测试**：空闲GPU/PTL Enabled/VECTOR,F8下，固定10-buffer、padding128/nt，
	对Hy3 N4096/per-tensor及N2048/PTPC的B16K/32K做24→48轮Down配对，再决定Combined/Full或fresh ATT；
	复用身份匹配的已编译产物，同时报告时延和有效TFLOPS（F=2×B×TopK×N×192）。
	首轮普通Down四点确认，配对降低10.028%～11.843%；后续普通/compact Combined/Full及fresh ATT也通过，128+64已删除，整192成为默认。先前拒绝记录保留。

[BK192实现验收](K192_BK192_IMPLEMENTATION.md) · [实现前资源估算](K192_BK192_RESOURCE_ESTIMATE.md)。

## K320末组合并为192（2026-09-07）

- [x] 可行性分析并实现显式`tile_k=192`：128+128+64→128+192，默认不变；MFMA32/48/32/48，总160/wave/N。
- [x] KS=2固定16/24KiB非对称B槽＋16KiB CShuffle，总56KiB、余8KiB；native16B+8B，正确映射K320 preshuffle尾192。
- [x] 复用通用1N/2N循环，扩展四拍退休/native VMEM账本、首末边界与compact满块支持；tail继续BK128。
- [x] 32个候选104次随机检查、14个compact配置70次候选检查（42次graph）、600次候选稳定性零失败；190项CPU通过。
- [x] 生产1N实测N2048/PTPC 214 VGPR、N4096/per-tensor 196 VGPR，56KiB/零spill；共享分支40个数值回归通过，另1个旧基线异常单列。
- [x] **K320合并尾块普通Down性能测试**：空闲GPU/PTL环境固定10-buffer、padding128/nt及CU策略，
	比较128+128+64与128+192的Down 24→48轮，之后决定Combined/Full/ATT；同时报告时延与有效TFLOPS（F=2×B×TopK×N×320）。
	首轮普通Down四点配对降低7.366%～8.723%；后续普通/compact Combined/Full及fresh ATT通过，128+128+64已删除，128+192成为默认。
- [ ] **旧K320/per-tensor/rolling历史异常根因调查（对应分支已删除，非新默认已知失败）**：N640/B257原分块曾rel_l2=0.00511390；
	失败ISA/.text/metadata与历史完全相同，未修改历史ELF也复现2/600次异常。新候选600次无失败，不等于已定位/修复旧问题；
	保留失败张量及全部记录，核查MFMA→VALU、LDS/VMEM同步等，不能归咎忙卡或放宽阈值。

[可行性与实现验收](K320_TAIL192_IMPLEMENTATION.md) · [证据汇总](results/k320_tail192/summary.json)。

## K512原型清理（2026-09-06）

- [x] 删除没有生产调用的K512专用N循环原型，保留已在使用的[通用实现](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py)，不改生产流水。
- [x] K512测试改为验证通用`position()`/`valid()`及LDS槽位；历史汇总严格校验原源码快照，不读取当前文件冒充旧身份。
- [x] 原型源码、ISA、ELF与旧结果保留；[清理后归档复核](results/k512_nloop/cleanup_reaudit.json)完成4个构建、5个边界、9份性能/拒绝记录。
- [x] 95项CPU回归通过（0.45秒）；生产主builder、通用循环及等待账本哈希不变，未运行GPU测试、未改动原暂存区。

## 最新：CU感知compact末轮拆分（2026-09-06）

- [x] 解释旧30个A类：compact已消除BM256额外padding/gateup metadata，24点零满块，5点此前已选中compact。
- [x] API取得CU数；当有效BM256任务余数满足`0 < 5*r < 3*CU`，将全局full后缀拆成4倍M64，默认阈值0.6。
- [x] 扩大tail容量、保护旧workspace、设备隔离缓存；74项任务表与七K14shape随机旧/新对照通过。
- [x] 本轮完成MI308X旧策略/仅扩capacity/新拆分的Down、Combined、Full及9点控制；见[性能报告](TODO_PERFORMANCE_20260907.md)。
	Qwen397两K16K三phase确认，Hy3/Qwen35的Full未达稳定门槛；没有把历史拒绝/旧42点矩阵重标为新测量。

[实现与A类原因](COMPACT_CU_BALANCE_REPORT.md)。下方42点历史矩阵未重标为新任务策略的性能。

## 当前轮：所有8x1默认1N循环（2026-09-06）

- [x] 七K、三量化、rolling/pure及compact满块默认1N循环；小N首尾剥离，原全展开保留对照。
- [x] 七K逐消费者VMEM账本及真实ISA跨迭代审计；K192/256不能无条件改9，pure不套rolling预算。
- [x] 42个随机配置、28个小N边界和14个compact随机/graph组合通过；同N896冷编译对照与跨进程产物复用。
- [x] Hy3 16K/32K、Qwen35 K512 8K输出padding0/32/64/128B单变量：128B三phase均胜其它padding。
- [x] Hy3固定128B比较普通/nt store：保持nt；只改变真实ISA的nt标志。
- [x] Qwen35固定128B store比较、完整七case×六Batch×全部候选矩阵及17点48轮边界复核。
	普通8x1选10/42、compact选5/42；其中Hy3 1K/2K是零满块tail数学，不宣称多数compact。
- [x] fresh七K ATT复制根目录并逐hash核验；原先根UI已为空，原始results归档保留。
- [x] 只分析`sorted_sum()`：同代码/地址/loc指针与准备长度，证明stride与前驱缓存影响；未改算法/线程数。
- [x] 按完整链三phase配对IQR逐点决定路径，列出42点普通8x1/compact的比较与不选原因，不强制多数compact。

完整证据见[本轮报告](ALL_8X1_NLOOP_REPORT.md)。以下为前轮历史TODO及验收状态，不冒充本轮结果。
最终[交付核验](results/all_8x1_nloop/final_delivery.json)与87项CPU回归通过；GPU状态恢复，未提交/push。

## K512快速测试与N循环评估（2026-09-06）

- [x] 微小调度改动先测普通Down-only两版本，不默认重跑compact/Full/七K全集。
	[快速入口](benchmark_k512_quick.py)复用已生成的原始ELF并校验.text差异，零FlyDSL JIT；
	10-buffer同地址配对、ABBA/BAAB交替。四点12轮约18.4秒、40轮约21.4秒（包含初始化与硬件恢复）。
- [x] stage4 1→5快速确认：N4096两档本轮约+0.68%，N2048 IQR跨零；不替代Full收益判定。
	详见[快速测试与循环方案](../../flydsl/attn_4wave/tools/stall_analysis.md#k512-quick-and-n-loop)。
- [x] 评估N完全展开的成本：N2048/N4096原始IR约12.37/24.64 MB，.text为79,296/155,968B；
	动态N循环有现成1x4参考。K512每N4个K块和8个stage均为偶数，LDS/staging奇偶相位不随N改变。
- [x] 仅为K512建立独立原型：首N＋末N/drain单独处理，中间N用运行时循环，内部8 stage和MFMA保持展开。
	将两份B carry、跨N packed输出、未退休SR3及其scale改为固定形状loop-carried SSA；不清空整个C覆盖SR3。
- [x] 对比每次循环1个N tile与2个N tile，不与stage4/缓存策略等改动混测；其它K保持现有版本。
- [x] 记录冷编译时间、IR/.text体积、VGPR/SGPR/LDS/spill以及同buffer Down配对时延，确认循环未被后端重新全展开。
	N2048冷编译210.916秒→8.612秒；1N/2N四组40轮Down确认约+0.8%–1.4%，0 spill，首尾/descriptor随机检查通过。
	[原型验收](../../flydsl/attn_4wave/tools/stall_analysis.md#k512-nloop-prototype)；stage4=5为正式默认，N循环仍显式`_n_loop=1/2`，默认0不变。
- [x] 后续按要求将七K循环晋级默认，compact/Full与fresh七K ATT已补；此前K512-only原型仍保留历史记录。

## 当前轮：建议1＋2

- [x] 采集当前8x1七K（128/192/256/320/384/512/640）fresh ATT，复制UI至仓库根目录。
- [x] GPU端从紧凑BM64 metadata生成BM256满块与BM64尾块任务表；禁止跨expert合并。
- [x] 显式compact down路径使用任务表、输出仍保持紧凑physical-row编号。
- [x] gateup保持BM64 metadata，不跟随down的BM256填充；不读取GPU任务数到CPU。
- [x] 校验任务覆盖恰好一次、空expert、全尾/全满、边界与随机routing；跑七K/量化正确性。
	175项通过：compact112项＋旧8x1/API63项；另含图重放、E2048、2GiB/4GiB权重偏移边界。
- [x] 检查资源/spill、Full含任务表构建开销的配对性能；不改自动selector。
	七K与397B资源均0 spill；16组旧ISA一致；13点ABBA24＋8点ABBA48确认。
	结果与限制见[compact验收](COMPACT_M64_REPORT.md)，仅397B K512/32K稳定优于历史最佳完整链。

## 建议3＋4（本轮推进3，4只分析原因）

- [x] 针对Hy3 16K/32K、Qwen35 K512 8K，单变量比较输出padding 0/32/64/128B。
- [x] 固定最佳stride后比较普通/非临时store缓存策略；同时验证Down、Combined、Full。
	Hy3保持nt；Qwen35 nt Down小胜但Combined小退、Full持平，普通store未满足全phase晋级条件，默认不变。
- [ ] 对sorted_sum比较64/128/256线程按N分工，保持每元素FP32 TopK累加顺序（用户本轮要求先分析，不实施修改）。
- [ ] 优先覆盖N4096/TopK9与N2048/TopK8；不直接引入跨expert atomic输出归约。

## 后续轮：建议5（本轮不实施）

- [ ] 依据各expert实际行数、满块/尾块比例与任务表成本设计选择策略，不使用单调Batch阈值。
- [ ] 小Batch/尾块占比高时保留小块路径，并纳入真实分布selector；Qwen35 K256 8K的普通8x1已通过48轮晋级，待做的是自动选路接入与真实routing验证。
- [ ] 候选仅在Down、Combined、Full的配对IQR均改善时晋级；边界使用ABBA48。

历史结果与分析见[MAIN_MERGE_PERFORMANCE_REPORT.md](MAIN_MERGE_PERFORMANCE_REPORT.md)。