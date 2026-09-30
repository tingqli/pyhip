# QSA实现与优化记录

本目录是gfx942 QSA attention与indexer的生产实现。使用方法、测试命令和最终有效性能统一见[benchmark说明](../../../../../benchmarks/qsa/readme.md)。本文件保存实现概要，并按时间追加优化历史；不再维护实验目录下的QSA文档。

## 文档约定

- [benchmarks/qsa/readme.md](../../../../../benchmarks/qsa/readme.md)只记录当前使用说明、最终有效性能及必要的适用范围/验收状态，不记录优化过程、中间版本或失败性能表。
- 后续优化、回退、失败、未采用方案和补录历史，只追加到本文件；每项写清改动、结果、测量范围和证据。
- 原始数据、trace、冻结源码及失败收据保留在mytest/mydata，不覆盖、不改写。功能通过、单kernel性能、完整调用和服务结果分别表述。

## 实现概要

| 文件 | 职责 |
|---|---|
| [attention.py](attention.py) | 公共API、输入检查、按layout/stream管理私有工作区、分流调用 |
| [attention_prepare.py](attention_prepare.py) | 四token块恢复/校验、union构表、精确mask、任务排序及错误检查 |
| [attention_dense.py](attention_dense.py) | 每请求完整因果前缀、任务配平和精确物理尾部边界 |
| [attention_union.py](attention_union.py) | BM128/BN64 union计算、公共块免mask、分组wave流水 |
| [attention_direct.py](attention_direct.py) | Direct计划与raw四wave fallback |
| [attention_direct_packed.py](attention_direct_packed.py) | 每次四token KV预排、单query wave direct |
| [indexer.py](indexer.py) | Prefill/decode prep、缓存状态、RoPE、压缩及选块入口 |
| [indexer_logits.py](indexer_logits.py) / [indexer_topk.py](indexer_topk.py) | Prefill MFMA logits、top-512完整块选择与token展开 |
| [indexer_decode.py](indexer_decode.py) | 按实际长度读取分页压缩K的decode logits |

当前attention只接受3D BF16 Q/O `[M,H,256]`、K/V `[N,HK,256]`及int32 `[M,2051]`选择。保留完整唯一四token块、0–3个因果尾和-1填充，local heads共享选择，`H/HK <= 16`。Indexer为独立的D128投影、4Q/1K头、ratio4/top512，最多65536压缩key。Decode选块仍复用SGLang的fast_topk和展开叶子内核。

Dense覆盖每请求前`min(query_len, max(0, 2051-prefix))`行。单请求物理KV四token对齐且PK+PV合计不超过64MiB时使用packed direct，否则raw fallback；每次调用或graph重放刷新pack。Packed路由比较填充工作量`160*ceil(U/16) <= 17*sum(ceil(selected_tokens/32))`，raw/ragged保留`U*rows <= 4*sum(selected_blocks_with_tail)`。H12通常BQ10，H3为BQ32；H6默认BQ16，大单请求无prefix/HK1且稀疏行数足够时使用均衡BQ21。

Graph要求同stream/layout先预热，共享工作区不得并发重放。工作区仍按实际长度分配；编译缓存按head/tile/scale等配置复用，精确长度、task/grid和容量为运行时参数。首次配置初始化有限的packed/raw与尾部变体，零任务launcher不执行GPU kernel；全新编译配置仍可能JIT。

## 优化历史摘要

以下从原实验目录的README和opt日志提炼，数据仅代表各自冻结版本与当时协议，不能当作当前版本的统一性能表。早期50samples、32buffers、插件开关和旧入口均为历史，不是当前使用说明。

删除旧目录前的完整原文只读保留：[README快照](../../../../../mytest/mydata/qsa_docs_layout_20260930_01/README.md.snapshot)、[优化长日志快照](../../../../../mytest/mydata/qsa_docs_layout_20260930_01/opt.md.snapshot)、[哈希清单](../../../../../mytest/mydata/qsa_docs_layout_20260930_01/archive.json)。后续只向本文件追加，不继续编辑快照。

### 2026-09-25：单接口、三路分流与首次接入

收敛为一个attention调用，内部管理metadata/scratch，保留dense因果前缀、跨query union和block-native direct。Direct取消V的LDS往返，复用排序与地址；边界修复覆盖ragged、空段、尾块、无效CTA与graph更新。早期插件完成真实TP2接入和输入捕获，后续已被直接包依赖替代。[历史模型证据](../../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/README.md)中的累计GPU时间不是TTFT或端到端吞吐。

### 2026-09-25至26：Union与dense布局

Union采用BQ10、公共块免mask、N64成本排序、蛇形任务分配及4+4wave流水。真实两层prepared union约4085.8/3843.8 -> 2790.7/2699.7微秒，填充210T目标通过，有效100T未通过；完整调用另测，不与prepared范围混用。[证据](../../../../../mytest/mydata/qsa_union_210t_20260925_01/README.md)。

Dense保留因果任务配平、对齐DMA与有界尾部。相同2048/2051行前缀优化后约161/179微秒；长输入取消dense虽有局部收益，但短前缀仍受益，未全局删除dense。[对齐研究](../../../../../mytest/mydata/qsa_mha_parity_20260926_01/README.md)。

### 2026-09-26：Direct流水、类型化属性与KV pack

Raw direct采用相邻4lane合并读取同token连续64B、消费端K转置、64-block索引缓存和K/V流水复用，M2048低重合场景约1428 -> 763微秒。16x4免转置方案虽正确且零spill，正式慢79–89%，未采用；保留提前发V、将K等待移到下一消费者等改进。[流水证据](../../../../../mytest/mydata/qsa_direct_pipeline_20260926_01/README.md)、[未采用布局](../../../../../mytest/mydata/qsa_direct_16x4_pipeline_20260926_01/README.md)。

发现通用passthrough属性被lowering丢弃，改为类型化`llvm.target_features`后才真正移除packed FP32指令，direct改善约2.6–3.0%。随后四token KV预排消除K数据转置/PV字节重排，单query wave与并行softmax使真实prepared direct约3035 -> 2500微秒，计时包含每次pack；该轮两层达到约100有效T。[属性证据](../../../../../mytest/mydata/qsa_direct_packed_20260926_01/README.md)、[pack证据](../../../../../mytest/mydata/qsa_direct_100t_20260926_01/README.md)。

### 2026-09-26至27：路由、scratch与未达目标

Packed分流由rho4改为union填充工作量不超过direct的1.7倍；raw/ragged保留rho4。16例完整调用中8个真实输入改善1.50–13.02%；1.65/1.75候选慢例保留，未宣称独立泛化验证。[路由证据](../../../../../mytest/mydata/qsa_route_20260926_01/README.md)。

PK+PV预算限制为64MiB/工作区，按实际CPU shape计算；超限直接raw，不按模型最大上下文分配、不回读GPU活跃计数。Graph生命周期与scratch地址稳定性纳入回归。Packed与raw完整分支对比计入pack成本，并区分同路由和原生路由。[预算证据](../../../../../mytest/mydata/qsa_pack_limit_20260927_01)、[分支对照](../../../../../mytest/mydata/qsa_pack_vs_raw_20260927_01/README.md)。

Direct的160填充T尝试未达到目标，MFMA/VALU交织的收益被VMEM等待抵消，短M反而更慢，未采用。32buffer试验出现两版共同慢阶段，原因未确定，未挑选快段或归因于空闲频率；后续固定10buffers/128samples。[160T尝试](../../../../../mytest/mydata/qsa_direct_160t_20260927_01/README.md)、[32buffer试验](../../../../../mytest/mydata/qsa_buffers32_20260927_01/README.md)。

### 2026-09-27至28：5D研究与撤销

曾研究原生SHUFFLE-5D及SGLang接入。Vvec8以8token为内层，与QSA四token选择存在额外地址/操作数重排成本；page-size=4并非旧ABI的简单参数修改。部分dense达到linear水平，union/direct未全面达到3D表现，按用户要求撤销QSA 5D。保留3D、packed/raw及预算；SGLang既有MHA 5D和通用缓存写入不改。独立.venv中的[3D恢复验收](../../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/analysis.json)不覆盖当前未支持的5D接口。

### 2026-09-28：零spill与准备链

修复compact及Torch bool归约的private/spill问题，全链检查包含验证kernel，不只检查主attention。A3先把准备8个launch减到4个，完整调用12 -> 8，但融合compact/mask局部变慢；继续采用单wave四元组恢复、等价min/max排序、有序scatter和四片mask并入排序/校验。恢复约159 -> 93微秒，真实prep降低39.97–45.15%、full降低3.37–8.65%；保持原精度与三项零资源。[零spill](../../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_analysis.json)、[准备链结果](../../../../../mytest/mydata/qsa_prepare_latency_20260928_01/analysis.json)。

实际TP2系统中attention优化使TTFT中位约1037 -> 956ms；生成文本并非全模型bitexact，profile累计GPU时间不能替代HTTP时延。[历史服务证据](../../../../../mytest/mydata/qsa_system_20260928_01/final_analysis.json)。

### 2026-09-28至29：Prefill indexer与长上下文

融合q norm/RoPE、ring写入、四token压缩，采用MFMA ReLU logits与每wave top-512，将107个kernel和3次host同步缩为投影GEMM加5个kernel。历史含投影回放约6.3ms -> 0.50ms（替代hipBLASLt投影）或0.78ms（SGLang投影），不是当前投影后benchmark口径。当前直接接入保留原SGLang投影，不迁入旧插件的替代GEMM开关。[历史服务结果](../../../../../mytest/mydata/qsa_indexer_system_20260928_01)。

压缩key上限从16384扩到65536，覆盖262144 token；长上下文top-k/GEMM分别测量，不用12k加速比外推。Logits/top-k由HIP改写为FlyDSL，18例对旧HIP逐bit一致，12k约改善2–3.5%，长形状变化较小。[长上下文](../../../../../mytest/mydata/qsa_indexer_long_20260928_01)、[FlyDSL改写](../../../../../mytest/mydata/qsa_indexer_flydsl_20260929_01)。

### 2026-09-29：Decode indexer与H6分组

第一阶段只按实际压缩长度读取分页K，避免按graph全宽gather，保留SGLang fast_topk/expand。第二阶段以一个Triton kernel完成完整decode prep（原43个kernel），包含norm/RoPE、pending ring和组边界压缩，写入逐bit一致。历史含GEMM的batch1/32、12k完整decode约221.7/2022.7 -> 31.2/43.3微秒，实际TP2 C1 ITL约13.404 -> 10.852ms；这些是当时插件版本的数据。[服务证据](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_03)、[开校验验收](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_04)。

H6大单请求P0/HK1采用均衡BQ21，使M128有效槽位96 -> 126，避免小尾组触发direct；其余路径保持原布局。历史L3 M12000完整调用约2.285 -> 1.877ms。后续18输入xTP2/4/8矩阵完成；TP4/8为local-head派生单卡重放，不是多卡服务。[矩阵证据](../../../../../mytest/mydata/qsa_tp_kernels_20260929_02/analysis.json)。

### 2026-09-29：精度参考与直接接入

T13的混合prefill差异离线逐bit复现，FP64表明原生BF16 Q预缩放参考自身超出`.02/.02`，QSA符合阈值。改用独立FP32全元素参考而不替换实际输出、不放宽精度；真实TP2 256请求及原失败布局通过。[T13闭环](../../../../../mytest/mydata/qsa_t13_20260929_02/delivery.json)。

SGLang随后直接依赖安装的PyHIP，移除临时插件和实验转发，测试/benchmark分离。独立验证又发现原生fast_topk将阈值桶截断到4096候选；只在桶溢出时用全行ordered-FP32 radix精确选择，普通路径和32KiB scratch不变。13行真实decode及集中分数回归通过。初次直接接入中位时延改善，但新shape JIT使整轮吞吐回退约40%，该历史失败保留。[接入与top-k证据](../../../../../mytest/mydata/qsa_native_integration_20260929_01/delivery.json)。

### 2026-09-30：JIT长尾修复

精确query/KV长度、task/grid和mask容量曾进入编译特化，附近新长度触发约13–16秒编译。改为运行时参数，并取消Triton不必要的长度特化；先消除重复长度编译，再以零任务launcher初始化有限的packed/raw、union尾部和dense对齐变体，移除布局首遇长尾。不改路由、算术、工作区和buffer边界，不增加外部预热请求。

相同32请求、12000输入/350输出、C1下，旧PyHIP TP2/TP4吞吐35.136/39.758 -> 74.312/81.370 token/s，p99 TTFT16.707/14.100 -> 0.901/0.700s。116项测试加6子测试、70真实验证请求、33个旧新同head bitexact回放和50份零private/spill产物通过。首次新head/tile/scale等配置仍需初始化，不声称机器码相同或饱和吞吐。[完整证据](../../../../../mytest/mydata/qsa_jit_latency_20260929_01/delivery.json)。

### 2026-09-30：084fd8f逐kernel回归未完成

对`084fd8f2882c914b08febe81124bbf7dbd549b7f`新增显式pytest perf矩阵，捕获生产HIP节点，采用10buffers/2warmup/128samples、AB/BA及固定5%回退门槛。11项不计时旧新对照和34项基础单测通过；decode沿用ABI、FP64边界1e-5和状态校验，不要求返回顺序固定。

第一项TP2/H12、M12000结束门禁use6%超过5%，2304条raw无效，后10项停止。仅作诊断的整段变化+0.37%，compact +17.34%、order/mask +5.76%、pack +20.37%；不能据此确认无回退或确定回退。未重采、未修改生产kernel。报告schema错误修复后只重算独立汇总，原失败不覆盖。[阻塞收据](../../../../../mytest/mydata/qsa_refactor_084fd8f_20260930_01/delivery.json)、[无效场次逐kernel诊断](../../../../../mytest/mydata/qsa_refactor_084fd8f_20260930_01/formal_01_recovered/kernels.txt)。

## 尚未验收

- 对084fd8f的完整逐kernel无回退结论，以及当前版本TP8/indexer单卡有效性能，仍需合格环境完成；不能用服务吞吐代替。
- 自动union/direct分流不是所有输入的全局最优；强制分支、完整准备成本和实际路由须同输入比较。
- Direct 160填充T、普遍有效100T、5D支持、非gfx942平台和模型质量基准均未作为当前功能或成果承诺。
- 历史共同慢阶段、对齐敏感性及个别布局风险保留；不得通过挑快样本、放宽精度或硬件门禁清除失败。

### 2026-09-30：1562154版本并发矩阵与kernel验收补测

用户确认执行084fd8f逐kernel对照及原生/PyHIP的TP2/TP4、C1/C2/C4服务矩阵。新研究目录为[qsa_benchmark_matrix_20260930_01](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01)，当前PyHIP HEAD为1562154。用户更新后的三个attention文件与上一轮服务测量字节不同，重新记录当前源码身份，不将旧成绩改标为新版本。本轮未修改生产内核或SGLang源码。

测试辅助存在迁移后的失效引用：补齐refactor测试使用的历史源码加载/HIP graph节点辅助，保留现有component计时；GRRead地址与硬件读取已搬到benchmark模块，QSA改为引用新位置。QSA仍独立严格检查use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8和入口/采样前/结束三次门禁，没有采用GRRead的PTL警告或仅入口策略。两个整体benchmark的地址导入同步修复。

首次入口因缺少tensor_address失败，0样本，收据保留。修复后首次正式kernel场次TP2/H12、M12000完成2304条raw，但结束GPU2 use6%/VRAM2%使全部raw无效，后10项停止；未重采、未换卡、未放宽阈值。仅供诊断的full变化+0.54%，compact +17.98%、order/mask +5.23%、pack +19.55%，不能据此确认无回退或确定回退。见[原始结果](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/kernels_v2/attention_packed_tp2/result.json)、[逐kernel诊断表](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/kernels_v2/kernels.txt)。

独立服务数值验收先于服务计时完成：[TP2](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/validate_tp2/correctness.json)、[TP4](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/validate_tp4/correctness.json)各35请求，TEST=1，含24k分块及32并发，原精度判据不变。新的[服务驱动](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/service.py)只复用旧研究helper，旧脚本和证据不改写；每个TP/实现服务顺序运行C1/C2/C4、各32x12000输入/350输出，固定seed42和原smoke/11888/12000预热。三档共享该服务的warm cache，profile另测，不称为交错同址或饱和吞吐。

服务矩阵最终完成12场、384请求，原生/PyHIP TP2吞吐C1/C2/C4为60.963/94.387/119.721 -> 74.321/118.421/158.153 token/s（+21.91%/+25.46%/+32.10%）；TP4为65.554/104.219/145.951 -> 81.556/133.284/191.552（+24.41%/+27.89%/+31.24%）。TTFT/ITL中位数改善，但TP2/C1的p99为1064.433 -> 1066.610ms，不能宣称全部指标变快。各组配置/源码一致、请求数和全部门禁通过，完整表见[benchmark最终结果](../../../../../benchmarks/qsa/readme.md#3-最终服务性能对比)及[审计JSON](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/analysis/summary.json)。

指定profile脚本另完成原生/PyHIP TP2/TP4共12trace，当前每个PyHIP rank有48次union/48次prefill prep和240次decode prep/logits，哈希全部核对。当前34项基础测试通过；新服务驱动另做CPU-only成功/C2失败生命周期检查，确认失败停止后续并清理自建进程。用户更新后的README含旧命令，已按实际CLI修正，保留API示例；编辑器缓冲曾短暂晚于磁盘同步，最终以实际磁盘链接/数字校验为准。服务测量成功不改变本轮2304条kernel样本无效的结论，整体任务状态仍明确保留kernel性能缺口。

### 2026-09-30：默认逐kernel输出与整轮门禁

用户撤销逐kernel旧新对照，删除独立test_refactor及其pytest参数、历史commit加载器；旧084fd8f对照不再是当前验收目标，原数据仍只读保留。两个正式benchmark默认输出整体与当前逐kernel耗时，默认10buffers/2warmup/128samples，省略output时自动建新目录；check-only继续不计时。Indexer默认覆盖prefill、decode和decode-forward，README不再依赖未提交研究脚本或本地报告。

GPU门禁按新要求移到整个性能任务开始与结束，各一次；不在case、kernel或采样循环里检查，阈值仍为use5%/VRAM20%/PTL Enabled,VECTOR,F8。矩阵结束失败时顶层summary将所有case标为无效，raw继续保留。三个CPU回归覆盖双检查、check-only零检查及退出失败保留样本。小输入attention和三种indexer模式均通过实际kernel捕获/重放；生产内核与cudaPerf没有修改。

默认attention M12000、TP2/4/8三个case全部执行，自动生成24行kernel耗时、2688条整体scope样本和逐kernel raw，但整轮结束快照use6%/VRAM3%未通过，全部数值不作最终有效性能；indexer正式性能未继续采样，也未重试attention。此轮确实只有两个GPU快照，见[矩阵状态](../../../../../mytest/mydata/qsa_default_bench_20260930_01/attention/matrix_status.json)、[功能检查](../../../../../mytest/mydata/qsa_default_bench_20260930_01/implementation_checks.json)。

服务C8补测与旧C1/C2/C4证据分开保存。实现哈希与此前70请求精度验收相同；四种TP/实现配置各按C1/C2/C4/C8运行，前3档保留以对齐缓存历史。整个补测任务只做入口/清理后两次硬件检查，内部不重复检查；旧研究驱动仅复用、不改写，README使用已跟踪launcher及`python -m sglang.bench_serving`复现，不依赖这些研究文件。

C8补测完成四服务、512总请求（其中C8为128），整轮两次GPU门禁、所有请求量、源码一致性、配置及进程清理通过。TP2原生/PyHIP为161.384/218.584 token/s（+35.44%），TP4为195.100/259.011（+32.76%）；C8 TTFT中位数TP2为7389.196/5752.741ms、TP4为5519.791/4410.691ms。第3节保留此前C1/C2/C4并注明C8独立补测，未覆盖旧结果。[C8审计](../../../../../mytest/mydata/qsa_default_bench_20260930_01/c8_summary.json)记录service_complete=true。

另采的12份profile哈希通过，但TP2 rank1的decode_prep只有239条，预期240；其它5个PyHIP rank符合48次attention/prefill prep与240次decode prep/logits。该差异保留在[独立profile异常](../../../../../mytest/mydata/qsa_default_bench_20260930_01/profile_count_discrepancies.json)，原因未确定，没有补造事件或重采。审计整体complete=false/profile_counts_complete=false，不宣称所有profile计数完整；普通服务时延在profile之外计时并单独验收。

### 2026-09-30：kernel性能改为仅入口检查并完成默认测量

按用户新要求，kernel/算子benchmark矩阵只在整轮开始检查一次GPU，删除结束检查；入口use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8阈值不变。check-only零检查，入口失败零采样、case失败保留之前样本。四项CPU回归通过。该变更只适用于kernel/算子benchmark，既有服务测量协议和原始收据不追溯改写，生产kernel与cudaPerf均未改变。

新目录[qsa_kernel_entry_only_20260930_01](../../../../../mytest/mydata/qsa_kernel_entry_only_20260930_01)从当前默认入口重新测量，不复用此前结束门禁失败的raw。GPU2/a4上attention和indexer两个命令各入口use0%/VRAM0%、PTL合格；每命令恰好一个hardware_before快照。3个attention TP形状与5个indexer case全部完成，43行kernel、5504条kernel raw、3328条整体raw按128samples重算中位数；实际输出、原精度、10buffer及源码哈希检查通过。[审计](../../../../../mytest/mydata/qsa_kernel_entry_only_20260930_01/audit.json)、[kernel CSV](../../../../../mytest/mydata/qsa_kernel_entry_only_20260930_01/kernels.csv)。

当前auto QSA M12000/H12/H6/H3为2617.234/2005.471/1451.548微秒；prefill indexer M12000为289.641微秒；decode select B1/B32为22.360/30.080，forward为26.080/35.680（投影后口径）。完整逐kernel数值发布在benchmark第2节，不做旧commit对照，不把单节点图中位数相加；TP8的pack/direct是没有实际direct行的gated空分支开销。旧门禁失败仍保留其当时结论，本轮采用经用户明确授权的新入口唯一策略，不声称测量期间持续验证GPU空闲。

### 2026-09-30：删除kernel性能入口门禁及设备屏蔽限制

按用户后续要求，attention/indexer benchmark删除GPU/CU屏蔽环境变量拦截，以及共用入口的利用率、显存、PTL和PCI一致性检查；删除专用门禁函数、参数、状态字段和测试。`--gpu`直接使用当前进程可见设备编号，不再调用硬件查询工具或生成硬件快照。矩阵记录器仅负责结果和异常汇总，正常执行、check-only及case失败保留已完成数据的三项CPU回归通过。

生产kernel、原cudaPerf、采样数和正确性校验未修改；本次没有重新采集GPU性能。benchmark保留已有最终性能并注明采集时的硬件条件，不追溯改写历史快照、失败结论或服务测量协议。

### 2026-09-30：decode indexer移除SGLang依赖

按用户要求，删除decode对SGLang fast_topk和expand_qsa_block_indices的导入及零起点缓存。Prefill与decode共享FlyDSL的精确top-512和token展开核心；decode专用入口直接读取每行compressed长度、query位置和sequence长度，不构造host layout。保留四token块、0到3个因果尾token及-1填充的2051宽度ABI，等分值按已有prefill规则优先小block id。Logits底层额外分配512个FP32元素，覆盖最后一行按512值批量读取的尾部空间，选择逻辑屏蔽有效长度外候选。

18项indexer GPU回归通过，覆盖prefill、decode、decode-forward、短行/零长度、511/512/513边界、紧凑33页和65536-key上界，以及原地更新metadata的CUDA graph重放；测试禁止导入SGLang。加上3项矩阵记录回归共21项通过，原独立FP64边界与逐bit状态检查不变。

使用默认indexer benchmark重新采集5个case，进程级禁止SGLang/AITER导入，实际加载列表为空。GPU0为MI308X/gfx942、PCI 0000:0a:00.0；遵循当前无硬件门禁策略，未读取利用率、显存或PTL。原cudaPerf、10buffers、2warmup、128samples，15行kernel结果、1920条kernel raw、640条整体raw及源码哈希全部核对。证据：[执行](../../../../../mytest/mydata/qsa_no_sglang_20260930T065525_3523362/execution.json)、[审计](../../../../../mytest/mydata/qsa_no_sglang_20260930T065525_3523362/audit.json)、[kernel CSV](../../../../../mytest/mydata/qsa_no_sglang_20260930T065525_3523362/indexer/kernels.csv)。

当前prefill整体290.202微秒；decode select B1/B32为23.600/31.520，forward为27.100/37.120。Decode select是logits与融合top-k/展开两个kernel，forward加一个prep kernel。不同GPU/场次不作旧新性能归因，未筛除样本或改计时器。Benchmark README更新当前indexer绝对性能并去掉SGLANG_USE_AITER；attention和服务旧证据保留，服务表明确未针对本次实现重测。未修改SGLang源码或提交Git改动。

### 2026-09-30：独立decode top-k的SGLang服务复测及后置TP2 profile

按用户要求重新测量当前集成，PyHIP HEAD a04b9ba887aae237e16380c82730af96e6983848、SGLang HEAD c3c4bc86f5c4feee62ca3c76d1b6e893825c45de。生产源码未改动；新驱动仅复用历史服务启动、预热、负载和自有进程清理逻辑。先分别开启TEST=1完成TP2/TP4各35请求验收，覆盖24k分块、32并发与短请求；然后TEST=0依次测原生/PyHIP的TP2/TP4，每服务C1/C2/C4/C8、每档32请求、名义12000输入/350输出，共512个性能请求。

服务seed42，chunk16384、最大请求32、decode graph最大batch32、radix关闭、CPU线程4；实际mem_fraction_static为TP2 0.8075、TP4 0.7225，原生/PyHIP配置逐项一致。整轮服务任务仅开始和全部退出后两次硬件门禁通过，内部没有重复状态检查；不声称采样期间持续空闲。六个自有进程组已退出，9080端口释放，源文件与冻结副本哈希一致，两个仓库HEAD/index在本轮前后保持一致；没有Git或硬件写操作。

当前原生/PyHIP吞吐token/s及增幅：TP2 C1 61.050/74.000（+21.21%），C2 94.837/117.893（+24.31%），C4 119.760/157.979（+31.91%），C8 161.427/217.999（+35.04%）；TP4 C1 65.545/81.372（+24.15%），C2 104.110/133.524（+28.25%），C4 145.832/191.569（+31.36%），C8 194.949/258.577（+32.64%）。TTFT/ITL完整表更新benchmark第3节，TP2/C1的TTFT p99为1053.092/1080.852ms，仍不宣称全部延迟改善。原bench_serving聚合JSON和日志原样保留，没有逐token时延数组，不做样本过滤或重算不存在的原始分位数。

四个性能服务另外采集12份rank trace，确认PyHIP各rank出现qsa_indexer_decode_topk而原生不存在；普通性能计时与profile分离。本轮[整轮记录](../../../../../mytest/mydata/qsa_service_retest_20260930_01/campaign.json)、[审计及最终表](../../../../../mytest/mydata/qsa_service_retest_20260930_01/summary.json)、[服务CSV](../../../../../mytest/mydata/qsa_service_retest_20260930_01/service.csv)。此前C1/C2/C4及C8补测、失败记录和profile计数差异不改写，本轮替换最终服务表，不追溯修正旧结论。

根据后续要求，待完整512请求数据审计完成后再单独启动当前PyHIP TP2服务，以profile脚本缺省C1、4请求、12000输入/5输出采集调用栈和形状；两个rank各记录240次当前decode top-k调用，trace哈希及清理检查通过，独立入口/出口门禁通过，自有进程组3621705已退出。[后置profile状态](../../../../../mytest/mydata/qsa_service_retest_20260930_01/tp2_profile_after/status.json)、[TP0 trace](../../../../../mytest/mydata/qsa_service_retest_20260930_01/tp2_profile_after/profiles/1790757175.007742/1790757175.0096133-TP-0.trace.json.gz)、[TP1 trace](../../../../../mytest/mydata/qsa_service_retest_20260930_01/tp2_profile_after/profiles/1790757175.007742/1790757175.0096133-TP-1.trace.json.gz)。这些profile请求不计入服务吞吐表；本轮验收不是GSM8K或生成文本逐bit一致性结论。