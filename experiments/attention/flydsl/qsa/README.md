# gfx942 QSA：attention 与 indexer

**2026-09-29包迁移：** 计算实现已移至[已安装QSA包](../../../../src/pyhip/ops/qsa/flydsl)，共用MHA依赖移至[已安装MHA包](../../../../src/pyhip/ops/mha/flydsl)。本目录运行时文件仅为临时转发，不再含kernel正文；整体测试为[benchmarks/qsa](../../../../benchmarks/qsa/readme.md)，单kernel测试在[tests/ops/qsa](../../../../tests/ops/qsa)。以下优化历史保留。SGLang子目录本轮逐字不改，旧说明中的测试相对链接和“target无需安装PyHIP”仅适用于迁移前；新target依赖同版本PyHIP，见[过渡集成说明](../../../../benchmarks/qsa/readme.md#5-临时sglang接入与迁移范围)。

> 当前默认：**10个独立输入/输出buffer、每实现128samples、2warmup**。
> 后续优化记录、补录历史及TP2/4/8统一性能/query比例/逐kernel占比只追加到[opt.md](opt.md)，不新增Markdown报告。

**2026-09-29命名整理：** `qsa`保留为总包，attention专属实现统一为[attention.py](attention.py)、[attention_prepare.py](attention_prepare.py)、[attention_dense.py](attention_dense.py)、[attention_union.py](attention_union.py)、[attention_direct.py](attention_direct.py)与[_attention_direct_packed.py](_attention_direct_packed.py)，测试入口为[test_attention.py](test_attention.py)。公开函数改为`attention()`；`indexer`及其文件/接口不变。旧源码名和内核名仅在历史数据/日志中保留；当前名称映射见[命名记录](opt.md#qsa-attention-names)。这是命名重构，不是数值或性能优化。

## 当前状态：仅3D，5D支持已撤销

**2026-09-29 QSA-T13已闭环，attention输出不改。** 原C32的layer47同一元素误差已捕获且离线逐bit复现；FP64证明原SGLang的BF16 Q预缩放参考自身超出`.02/.02`，QSA输出符合该容差。插件改为全部元素对独立FP32 selected-token参考检查，原阈值、失败抛错和旧SGLang差异记录均保留，不返回参考输出。81项测试、真实输入和独立包TP2/4/8通过；用户授权续测后实际TP2/C1+C32 **256/256请求成功**，两rank各77布局×12层=924项全量校验零失败，并实际再次经过原失败布局。旧门禁失败保留；六个attention文件及indexer不改。详见[复现与根因](opt.md#qsa-t13-resolution)、[真实服务闭环](opt.md#qsa-t13-service-closure)。验证开启的时延不是生产性能，未测真实TP4/8服务或模型质量。

**2026-09-29 H6布局：** 大单请求、无prefix、HK1的H6 prefill采用均衡BQ21，M128有效槽位96→126，避免小尾组触发direct。其它TP2/8、prefix/ragged及小batch保持原路径；pad1.7和全部attention设备计算不变。同场L3 M12000完整调用2.285→1.877ms（−17.82%）；67项正常测试、独立包12例通过。此前门禁停止记录保留；用户授权续测后，当前18输入×TP2/4/8的分支/完整调用/准备/独立pack-direct和18份profile已全部完成，普通71040条＋组件29568条raw，见[完整审计](../../../../mytest/mydata/qsa_tp_kernels_20260929_02/analysis.json)。8个真实capture的自动qsa：TP4比TP2快31.06%～37.45%，TP8快47.33%～50.61%；这些是派生local-head回放，不是多卡服务。完整已有表格及[当前逐kernel伪代码](opt.md#qsa-current-pseudocode)见[opt.md](opt.md)，自动路由慢例和QSA-T13真实混合prefill容差问题未隐去。

2026-09-28按用户要求撤销本轮QSA SHUFFLE-5D及SGLang接入，恢复T01之前的3D实现。公共API不再接受`page_table`或5D K/V；dense/union、3D packed direct、raw fallback、64MiB pack预算与pad1.7分流全部保留。SGLang既有MHA 5D、通用缓存分配器和cache writer未改。

原因是现有BF16 Vvec8物理布局以8token为内层，而QSA `compress_ratio=4`按四token选块。二者可正确适配但有非连续载荷、额外operand重排和地址成本；S4不能直接套用Vvec8 ABI。Vvec4/S4只是新布局研究原型，未集成或部署。

撤销前最后同场两层TP2测试：3D union为4.191/4.026ms、direct为2.474/2.472ms；旧5D S64为4.678/4.500ms和3.923/3.889ms。每版10buffers/128samples，包含每次pack/地址准备；这是prepared分支，不是full或新环境复测。全部性能、限制、失败、原始数据和归档源码保留于[opt.md](opt.md)，旧5D包不可用于当前回退后的SGLang。

**后续默认使用PyHIP的.venv Python环境。** 2026-09-28已独立安装与镜像一致的ROCm运行依赖，39,890个安装载荷文件哈希核对一致，仍关闭system-site-packages；SGLang/AITER可导入。原回退轮缺依赖的限制已解除，旧阻塞记录不改写。

此前[恢复3D全面复测](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/analysis.json)完成QSA48＋MHA28正确性及54＋12组性能，发现Triton compact及Torch验证的全链资源缺口；原失败和全部历史结果保留于[opt.md](opt.md)。TP4/8仍是TP2 capture派生local-head，不是多卡服务。

此前[全链零spill修复](../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_analysis.json)已消除compact和Torch bool reduce的12B private，保留其53组有效性能及全部失败历史。最新准备重构继续保持三项零资源，不重新引入Torch临时归约。

此前A3已将准备8个launch→4个、真实完整调用12→8，但compact+mask融合局部变慢，原9组有效性能和门禁停止记录保留于[上轮审计](../../../../mytest/mydata/qsa_prepare_20260928_01/analysis.json)。

**最新：单wave恢复＋分片mask，仍保持4个准备/8个真实full launch。** [attention_prepare.py](attention_prepare.py)改为四元组加载、等价min/max排序和有序scatter；compact只构表/清理，mask分4片并入排序/校验launch。恢复从约159降到93µs；两层TP2/4/8的prep降低39.97%～45.15%、full降低3.37%～8.65%。不是仅靠任务调度，也没有增加launch。

65项正常测试、54组378个实际dispatch及独立包48dispatch全链三零。新36组正式普通性能18432raw全部完成，最大full中位回退仅0.291%；6组逐kernel6144raw及6份交错profile完成，见[内核审计](../../../../mytest/mydata/qsa_prepare_latency_20260928_01/analysis.json)和[opt.md](opt.md)。ELF固定LDS=0不等于总LDS=0，动态LDS已补核；三项private/spill仍0。没有回填旧停止矩阵、未处理MHA8192慢阶段。

**实际TP2系统测试已完成。** 原Qwen3.8配置、GPU0/1、同10条约12k prompt、并发1/输出5，各128个无profiler请求：原生SGLang→当前QSA的TTFT中位1037.031→955.764ms（−7.84%），请求时延1090.343→1009.214ms（−7.44%），ITL基本不变。另双rank profile各48次QSA累计GPU约467→133ms；它不是TTFT或普通吞吐分母。两组顺序启动非交错，114/128配对文本相同、两条prompt在原生/当前内部也不稳定；144项attention原容差检查通过，但不是模型质量认证。见[系统审计](../../../../mytest/mydata/qsa_system_20260928_01/final_analysis.json)。测试服务已清理，未测真实TP4/8系统或饱和吞吐。

**最新：prefill indexer替换（issue #47/#53）。** [indexer.py](indexer.py)＋[indexer_logits.py](indexer_logits.py)＋[indexer_topk.py](indexer_topk.py)在`PYHIP_QSA_INDEXER=1`时接管eager EXTEND的`QSAIndexer.forward_cuda`：107个kernel、3次host同步→GEMM＋5个kernel、无同步。GPU2真实8个capture（10buffers/128samples/AB-BA）完整调用6302–6510µs→默认503–513µs（12.5–12.8×，hipBLASLt投影）或782–784µs（8.1–8.3×，SGLang GEMM、prep逐bit）。实际TP2系统（顺序服务，各128个无profiler请求）TTFT中位955.13→879.42/881.26ms（−7.9%/−7.7%，两次独立服务），exact为879.75/888.39ms；服务间波动2–9ms，大于default/exact每请求约3ms的GPU差。trace中每rank每调用5502→525µs。TP间复制，TP2/4/8每rank indexer工作相同（TP4/8非实测服务）。正确性、GEMM取舍、各kernel占比与系统结果见[opt.md](opt.md)。

**最新：decode indexer第二阶段（2026-09-29）。** `PYHIP_QSA_INDEXER_DECODE=1`时，由插件接管CUDA graph decode的整段`QSAIndexer.forward_cuda`：先跑SGLang自己的GEMM，再由一个Triton kernel `_indexer_decode_prep`完成q norm/RoPE、pending ring写入、组边界均值压缩及写入，最后接第一阶段的分页logits、fast_topk和expand。

- kernel数：SGLang每层43个prep kernel，现在是1个。
- 正确性：q、ring、rope和压缩写入与SGLang逐bit一致（合成、真实权重、6步graph重放）。
- 回退：eager decode仍走SGLang加`select` hook。
- 服务内校验：`PYHIP_QSA_VALIDATE=1`现在也在每个decode graph步内对照SGLang（设备计数器，下一次prefill检查）。真实TP2验收（decode=1、VALIDATE=1，256请求加profile）全部通过：每rank 12层×10300次调用、25644个真实行，q/pool写入不一致和选择违例全为0；attention与prefill indexer各336项通过。开校验时的计时不是性能。见[分析](../../../../mytest/mydata/qsa_indexer_decode_system_20260929_04/analysis_075332.json)和[opt.md](opt.md#qsa-decode-stage2-validate)。

GPU3正式图重放（一层整段decode `forward_cuda`，10buffers/128samples/AB-BA），中位µs：

| 形状 | SGLang | 第一阶段 | 第二阶段 |
|---|---|---|---|
| bs1×12000 token | 221.7 | 128.8 | 31.2 |
| bs8×12000 token | 672.7 | 189.5 | 35.7 |
| bs32×12000 token | 2022.7 | 212.9 | 43.3 |
| bs32×262144 token | 2074.5 | 421.2 | 252.0 |

90组buffer的token集合与SGLang相同，pool写入逐bit相同。indexer在TP间复制，TP2/4/8每rank工作相同（TP4/8为推导）。

实际TP2三臂系统测试（顺序服务，12k prompt，VALIDATE=0）：

| 指标 | SGLang | 第一阶段 | 第二阶段 |
|---|---|---|---|
| C1 ITL中位 | 13.404ms | 12.253ms | 10.852ms（−19.0%） |
| C32（实际31路decode）ITL中位 | 43.644ms | 23.443ms | 21.186ms（−51.5%） |

- TTFT不变。
- profile中每步indexer 3.91→0.39ms，每步kernel合计15.96→12.40ms。
- 各臂自身稳定的fixture，输出三臂相同；其余fixture在同一臂内也不稳定。
- 详情与三次尝试记录见[opt.md](opt.md)。

**decode indexer第一阶段（2026-09-29）。** [indexer_decode.py](indexer_decode.py)的FlyDSL分页logits在`PYHIP_QSA_INDEXER_DECODE=select`时（第一阶段记录中的`=1`）经`QSAIndexer.select_decode_tokens`接管decode（eager与CUDA graph都接）。每行只读自己的`compressed_lengths`个key，不再按graph全宽65536 gather；SGLang的fast_topk和expand保持不变。原来每层16个kernel，现为3个。GPU3正式图重放（10buffers/128samples/AB-BA）：bs1/8/32×3000 key从110.3/473.6/1727.6µs降到22.4/24.9/31.7µs，65536 key从155.9/546.7/1888.9µs降到71.0/113.0/250.2µs，90组buffer的token集合都与SGLang相同。实际TP2系统（顺序服务，12k prompt）：C1的ITL中位13.440→12.209ms（−9.2%），31路并发decode为43.261→23.491ms（−45.7%）；trace中每层decode MQA 106.7→4.5µs。当时记录“剩余每层约85µs的decode prep/压缩（19个小kernel）”，是只算了压缩那一半；完整的SGLang decode prep为每层约190µs、43个kernel，已由第二阶段替换。首次系统运行时attention的`PYHIP_QSA_VALIDATE`在一个C32混合prefill布局超出.02容差（1/50233344个元素）并中止服务，详情见[opt.md](opt.md)。

## 2026-09-27：独立buffer增至32，重新完成TP2/4/8矩阵（历史）

本历史轮曾把benchmark默认改为32独立buffers、每实现128samples（每buffer4次）；最新要求已恢复10个buffer，保留128samples。CLI支持`--buffers`/`--samples`覆盖；warmup仍2。TP2/4/8与derived数据标记不变。
重新执行baseline/精确冻结phasepg8候选30例，7680raw和90次门禁全部完成。真实direct仍约135–138填充T，**160T未达到**；短M候选慢3.64%–4.16%，不采用，运行时源码不改。
TP4真实full出现随轮次从约2.45ms增至约3.6ms的共同变慢，全部样本保留，最终中位约3.44ms；路由/内核/输出相同，原因尚未确定，不只取快段或归因于buffer数/空闲频率。
[32buffer结果、逐轮分布与身份审计](../../../../mytest/mydata/qsa_buffers32_20260927_01/README.md)。

## 2026-09-27：统一TP2/4/8、scratch契约与160填充T尝试

正常真实回放、perf和CLI现默认同时覆盖TP2/4/8的H12/H6/H3；H6/H3是TP2 capture的local-head派生重放，**不是新多卡TP4/8采集**。44项正常测试通过，perf收集24项。
新增scratch回归验证按实际CPU KV shape分配、热调用/graph地址稳定、没有Tensor `.cpu()`/`.item()`/`.tolist()`/布尔回读或重新分配PK/PV。
KV pack额外容量为`1024*N*HK`字节：N12000/HK1为12.288MB，不按模型max262144预留；全union时只跳过搬运、缓冲仍保留。普通workspace最多8项，captured workspace另受graph生命周期约束。

已完成30例正式base/交织候选对照（TP2/4/8×两真实层/高低重合/短M×direct/full），3000raw全保留。**160填充T未达**：真实主例direct当前约135/137/138填充T，交织候选长形状相近、短M慢3.5%–4.2%，不采用；runtime kernel/分流不变。
匹配ATT证实MFMA/VALU交织出现，但VMEM等待增加抵消收益。详细证据和scope见[本轮报告](../../../../mytest/mydata/qsa_direct_160t_20260927_01/README.md)。

## 2026-09-27：当前pack / 无pack分支对照（源码未改）

同输入、同selected query的8例正式对照，当前packed direct **含每次pack**比raw4wave分支中位时延低16.80%–20.53%；原50sample/10buffer，2000正式raw全部保留。
真实L3：3163.336→2513.993µs（中位数比-20.53%，配对口径-17.15%）；L47：3051.676→2510.314µs（-17.74%）。本轮packed为**99.665/99.811有效T，未过100T**，不重标前轮成绩。
同路由完整QSA L3/L47快14.72%/12.22%，低重合H12/H6快14.94%/15.65%；全union高重合整体相近。M64仍有分支收益，但H12原生rho4尾4行走union造成的额外退化必须与pack收益区分。
这是两个完整分支（布局/CTA几何/流水线）的对比，不是只改pack的单因素消融；实际ELF及raw4wave metadata一致性已核验。生产代码和现行1.7路由未改。
[完整表格、同路由/原生路由区别和审计](../../../../mytest/mydata/qsa_pack_vs_raw_20260927_01/README.md)。

## 2026-09-26：Union/direct重新分流（当前策略）

以当前100T direct为基础，将packed场景旧rho4替换为**实际填充工作量比较：union≤1.7×direct时选union**。按query组的实际行数与BN64/BN32/M128/M16计算，包含因果尾；不改变选中token。
实际DirectPlan有packed KV才启用，raw/ragged保持旧rho4；dense2051、query分组、attention kernel、公共API不变，无新增kernel或CPU active回读。

完整公开`qsa()`（含恢复/校验/构表/pack/全部attention）最终16例正式，每例10buffer/2warmup/50sample：

| 真实TP0输入 | 原版→新版µs | 耗时下降 |
|---|---:|---:|
| L3 M12000 | 3338.018→**2903.916** | **13.005%** |
| L47 M12000 | 3247.318→**2993.316** | **7.822%** |
| L3 M11888 | 2882.395→**2839.096** | **1.502%** |
| L47 M11888 | 3135.257→**2881.116** | **8.106%** |

TP1也改善，8份真实输入全部快1.50%–13.02%；H12的75%共享块用例快37.05%，其余合成端点变化≤0.22%。27项正常回归通过。
rho1.75虽在两主例更快，却使H12高重合退化；首版1.65正式在M11888/L3小退化，失败证据保留后重新校准为1.7。留出输入参与了反馈调参，不称独立泛化证明。
最终attention整ELF不变，仅compact planner机器码变更；其它planner.text不变但debug位置可改变整ELF。没有新SGLang部署/全模型profile。
[完整选择公式、16例正式、失败与审计](../../../../mytest/mydata/qsa_route_20260926_01/README.md)。

## 2026-09-26：Direct有效100T（内核不变，分流记录为历史）

真实Layer3/47的prepared direct达到 **100.130/100.299有效TFLOPS**，计时包含**每次KV预排+attention两个GPU kernel**。
单纯DS分散/交织只带来小幅收益；最终在4-token块内预排KV，使原生MFMA读取无需K数据DS/PV字节置换，配合单query wave、索引预取和并行softmax归约。

| 精确生产direct | 原版 → 新版 µs | 耗时下降 | 新有效 / 填充 TFLOPS |
|---|---:|---:|---:|
| Layer3 sparse9949 | 3039.856 → **2502.334** | **17.68%** | **100.130 / 134.972** |
| Layer47 sparse9949 | 3035.136 → **2498.114** | **17.69%** | **100.299 / 135.200** |
| 低重合H12 | 712.364 → **588.963** | **17.32%** | 87.573 / 118.046 |
| 低重合H6 | 707.684 → **584.843** | **17.36%** | 44.095 / 118.878 |

10buffer/2warmup/50sample正式；低重合H12/H6未达到100有效T，不以填充T代替。最初99.965T失败原样保留，最终实质改进版和精确生产版分别通过真实两层门槛。
只对单请求且物理KV长度4对齐、安全byte-span启用预排，其它形状保留raw-KV路径；scratch每次刷新，包括graph。选择/RNE/rho4/dense2051不变，原26项回归通过。
完整公开QSA低重合H12/H6约快15.51%/15.93%；真实两层仍direct0，整体相近。所有长尾保留。
最终Layer3双kernel ELF **28fd66d3…**，pack26VGPR/0LDS，attention250VGPR/8KiBLDS，均零spill；新ATT与此ELF匹配。
[完整实验、失败记录、正式结果与ATT路径](../../../../mytest/mydata/qsa_direct_100t_20260926_01/README.md)。

## 2026-09-26：真正关闭packed FP32（历史轮）

原`passthrough/target-features`写法在当前GPU→ROCDL转换中丢失，原Layer3 ELF仍有64条`v_pk_mul_f32`；删除该属性ELF也完全相同。
现用`llvm.target_features`类型化属性真正关闭，packed FP32指令64→0，原BF16 RNE/数学/布局/流水线/分流不变。

| prepared direct | 原版 → 关闭 µs | 耗时下降 | 关闭后有效 / 填充 TFLOPS |
|---|---:|---:|---:|
| Layer3稀疏9949行 | 3119.916 → **3038.456** | **2.61%** | 82.462 / 111.157 |
| Layer47稀疏9949行 | 3116.857 → **3033.736** | **2.67%** | 82.591 / 111.330 |
| 低重合H12 | 733.884 → **713.244** | **2.81%** | 72.314 / 97.477 |
| 低重合H6 | 729.464 → **707.904** | **2.96%** | 36.430 / 98.212 |

10buffers/2warmup/50samples，400正式raw全部保留、门禁通过；精确生产版四例ELF/.text等于已测关闭版，原25项QSA回归通过。
当前248VGPR/42SGPR/32KiBLDS/spill0，Layer3 ELF **e27713d7…**。本轮没有新ATT/PMC或完整公开QSA计时；**旧800d13cf…的ATT不是当前版本**。
[本轮属性传递证据、正式结果与审计](../../../../mytest/mydata/qsa_direct_packed_20260926_01/README.md)。

## 2026-09-26：16×4 K尝试与提前V流水（历史轮）

已按建议实际测试16×4、lane沿16增长、16B/lane并省K转置：正确/零spill，但正式比原合并K布局慢79–89%，**不采用该布局**。
保留胜出部分：QK前发V0两半，K等待移至下一QK消费者，PV1不再被循环尾zero-wait提前阻塞；K仍为相邻4lane连续64B＋逆转置。

| prepared direct | 入口 → 最终 µs | 耗时下降 | 最终有效 / 填充 TFLOPS |
|---|---:|---:|---:|
| Layer3稀疏9949行 | 3185.997 → **3119.496** | **2.09%** | 80.320 / 108.269 |
| Layer47稀疏9949行 | 3187.277 → **3115.037** | **2.27%** | 80.435 / 108.424 |
| 低重合H12 | 762.704 → **733.004** | **3.89%** | 70.364 / 94.849 |
| 低重合H6 | 758.444 → **728.604** | **3.93%** | 35.395 / 95.422 |

精确生产版10buffers/2warmup/50samples，各门禁通过。完整低重合QSA H12/H6为826.625/817.584µs，比同场入口快3.55%/3.41%；真实两层仍不走direct，完整约相同。
完整调用58ms/36ms离群值全保留，不筛快样本。最终246VGPR/42SGPR/32KiBLDS/spill0，25普通测试通过。
ATT证实：16×4虽去64条K数据DS/BN32，VMEM指令区间却升至72.36%；最终合并流水MFMA36.37→39.14%，主要V/K等待中位降至4cycles，非HBM流量结论。
[本轮候选、正式与ATT完整报告](../../../../mytest/mydata/qsa_direct_16x4_pipeline_20260926_01/README.md)。API/选择/RNE/rho4/dense2051不变。

## 2026-09-26：完整正式矩阵与重合度 / 三分支对比（历史轮）

**16例、56组合、2800raw全部通过**：每实现10独立buffer/2warmup/50sample，GPU2/a4、原cudaPerf、48次门禁全部通过，旧失败轮保持原状态。
本轮不改kernel与rho4/dense2051生产策略。低重合H12/H6完整QSA从1523.549/1512.828降至856.445/843.364µs，分别-43.79%/-44.25%。

- 同QKV独立选块：direct759.644/755.444µs，union2125.952/1898.950µs，direct快2.80×/2.51×。
- shared高重合：direct742.124/738.224µs，union405.602/214.561µs，union快1.83×/3.44×。
- 真实两层高重合query[2060,4060)：union约423–426µs，direct约710–712µs；
  低重合[10000,12000)：direct736.044/740.023µs，union817.565/773.504µs，direct节省9.97%/4.33%。
  窗口仅按原indices选取、不按时延挑选；当前rho4仍全部送union，有重新标定gate的空间，但尚未改阈值。
- 整个真实9949稀疏行：direct3195.357/3190.357µs，union2787.815/2694.094µs；**整体union仍优**。
  union有效89.876/93.003T、填充213.370/210.471T；direct有效78.413/78.536T、填充105.699/105.864T。
- dense等价前缀：H12/2048 dense161–162µs优于union171–172µs；H12/2051 union约176µs略优于dense179–180µs；H6两长度仍dense优。
  direct约392–398µs，不适合替换这些前缀。稀疏长区dense不等价，未拿无mask全量attention代替。

[完整表格、图与独立审计](../../../../mytest/mydata/qsa_formal_matrix_20260926_01/README.md)。生产源码/完整用户index未改，无新SGLang部署或全模型profile。

## 2026-09-26：Direct K 合并读取与流水改进（历史轮）

相对上一轮已优化direct，保留相邻4lane读取同token连续64B、QK消费端转置、64-block索引缓存和PV分批K预取。
M2048/P30000/HK1，10buffers/2warmup/每版每例50samples：

| 用例 | 入口 → 最终 µs | 耗时下降 | 最终有效 / 填充 TFLOPS |
|---|---:|---:|---:|
| H12 | 1427.627 → **763.204** | **46.54%** | **67.580 / 91.096** |
| H6 | 1424.308 → **758.704** | **46.73%** | **33.990 / 91.636** |

两主例各自门禁通过，200raw全保留；随后real3采样前GPU use6%失败，停止、不重试，**整个正式矩阵未完成**。
完整QSA只有2buffer/4sample探索：低重叠约快39–40%；真实两层direct0，整体相近。没有完整调用正式50次或新模型profile。
25普通测试通过（新增缓存边界/HK2/NaN尾）；两合成与两真实forced-direct相对入口均bitexact。
ATT内部MFMA模型占比15.60→34.54%，K/V向量load数不变；不能解释为HBM流量减少。分流/选择/精度不变。
[本轮报告、全部候选与审计](../../../../mytest/mydata/qsa_direct_pipeline_20260926_01/README.md)。

## 2026-09-26：Linear causal 与 Direct / 准备优化（历史轮）

Linear B1非分页causal已迁入dense任务配平：Q2048/8192正式292.521/2475.312→161.641/1838.490µs；8192为224.297有效/227.774填充T。
Direct保留K地址复用V、成对QK与跨PV的K预取；共享恢复校验已有的排序结果，删去第二次direct排序及独立block buffer。
公开indices仍可无序且只读，私有block scratch每次重建；dense2051/rho4分流不变。25项QSA+28项MHA通过。
Direct计算候选探索约9.5–9.7%改善，但**最终正式轮采样前门禁失败，0raw，未重试**；不可把探索或较早d07组合重标为最终正式性能。
[完整解释、工作量/ATT、候选及审计](../../../../mytest/mydata/qsa_linear_direct_20260926_01/README.md)。

## 2026-09-26：块图与 Dense/MHA 对照（历史轮）

[真实两层块图及优化报告](../../../../mytest/mydata/qsa_mha_parity_20260926_01/README.md)含选中块/union额外计算/尾块与N64虚槽，附可缩放浏览页。
同语义causal Q2048/2051正式对照：优化dense分别160.96–162.04 /179.20–179.64µs，原生MHA约293.4 /225.3µs；union约171–175µs。
仅保留dense因果任务配平、对齐原生DMA和有界地址叶子融合；union新候选未胜出，已恢复。
**未宣称短因果前缀达到220T；未全局删除dense。** 完整12k不拆dense仍略快，但短正式矩阵结束门禁失败且2048本体dense胜出，生产分流不变。
24项正常回归通过。direct性能分析按用户要求仅作TODO，等待确认后继续。

## 2026-09-26：Dense 分支评估

“并集/M128填充2.6–2.8倍”对应旧BQ8；当前BQ10在两层真实稀疏9949行上的倍率为**2.3740/2.2631**，不是全12k工作量。
同一真实2051行因果前缀：union填充28.790G、dense30.702G FLOPs；正式10buffers/50samples，union约175µs、dense约237µs。
完整12k取消dense前缀分流为3.301/3.202ms，同场保留为3.434/3.339ms，下降3.89%/4.11%。
短请求探索仍显示dense有收益；正式短矩阵在结束门禁失败，已保留失败raw、不重采。
**本轮不全局删除dense、不改生产路由或推定切换阈值。** 无dense实验路径14项、生产QSA21项通过；只修复shared测试数据跨2051初始化并补回归。
[倍率公式、同输入比较、失败门禁与结论](../../../../mytest/mydata/qsa_dense_branch_20260926_01/README.md)。

## 2026-09-26：Union 性能更新

真实TP0、Layer3/47、M12000/P0、Q12/KV1/D256，预分配union本体：

| 输入 | 原版 → 最终 µs | 最终有效 TFLOPS | 最终填充 TFLOPS |
|---|---:|---:|---:|
| Layer3 | 4085.762 → 2790.715 | 89.783 | **213.148** |
| Layer47 | 3843.800 → 2699.675 | 92.810 | **210.036** |

原`cudaPerf`、10独立buffer、每版50样本、AB/BA，全量200raw含慢尾；时延下降31.70%/29.77%。
**填充≥210T通过，有效≥100T未通过。** 两版分别核算真实M/N几何，不能用原版更大的填充分子给新版换算。
含恢复/plan/新排序的完整QSA另测为3.432/3.337ms（同场原版4.706/4.372ms），不是TTFT。
88项BF16 MHA＋QSA功能回归通过，原精度不变；新ATT核验了S0 mask预取、S3 bit-select与4+4wave交织。
[完整数据、ATT、失败候选和限制](../../../../mytest/mydata/qsa_union_210t_20260925_01/README.md#L1)。

## 一个接口

从本包导入 `attention`，调用 `attention(q, k, v, indices, *, query_lens=None, prefix_lens=None, softmax_scale=None, out=None)`。
不需要构造contract、prepare/rebuild对象或管理scratch。

- Q为BF16 `[M,H,256]`，K/V为BF16 `[N,HK,256]`，均连续且16-byte对齐，每个buffer字节数小于2GiB；仅3D，推理专用。
- `H % HK == 0` 且 `H/HK <= 16`。主要验证TP2/4/8对应Q12/6/3、KV1；本函数不进行TP通信。
- `indices`为连续device int32 `[M,2051]`，每query独立选择，local heads共享；token ID在所属请求内编号。
- query位置$p$：`min((p+1)//4,512)`个不重复完整4-token块，接0–3个causal tail token，再以`-1`填充。
  完整块可无序；精确保留原选择，不近似合并各query的可见集合。
- host `query_lens`/`prefix_lens`描述packed请求。单请求默认`(M,)`/`(N-M,)`；多请求省略prefix表示全0。
  空query段仍计入KV偏移。
- scale默认`1/16`；可指定有限正数。`out`可复用，不得与Q/K/V、indices重叠。
- 首调用分配/校验元数据并JIT；热调用使用private per-stream缓存，但每次按当前indices重建选择。
  graph须在**同一stream**预热；捕获用workspace在进程生命周期内保留，普通缓存最多8项。
  调用者负责跨stream输入依赖；共享workspace的graph不得并发重放，也不得与使用该workspace的eager调用并发。
  当前FlyDSL同一进程只支持一个GPU；TP服务每rank独立进程。
- Direct的PK+PV合计最多**64MiB/工作区**；按当前K/V shape及dtype在CPU计算字节数，超过即不分配pack scratch并走原raw fallback。
  等于上限仍可pack；这是单工作区两份额外KV的上限，不是进程总scratch预算。完整调用和各kernel伪代码见[opt.md](opt.md)。

## 仅保留胜出路径

| 文件 | 职责 |
|---|---|
| [attention.py](attention.py) | 唯一公共attention函数、host校验、private per-stream workspace与分流调用 |
| [attention_prepare.py](attention_prepare.py) | 单wave恢复/校验＋有序scatter、compact＋暂存清理、分片精确mask＋稳定排序/错误归约及私有计划 |
| [attention_dense.py](attention_dense.py) | 每请求前`min(M,max(0,2051-prefix))`行；原tensor视图、因果任务配平，64对齐用native DMA，否则全VOFFSET有界DMA |
| [attention_union.py](attention_union.py) | BM128/BN64 union attention kernel，消费已准备mask/order；common前缀免mask |
| [attention_direct.py](attention_direct.py) | direct计划/分派与raw-KV fallback；非预排形状保留四wave合并K/消费端转置路径 |
| [_attention_direct_packed.py](_attention_direct_packed.py) | 单请求4-token对齐且PK+PV≤64MiB的每次预排+单query wave attention；无K数据转置，私有scratch随图重放刷新 |
| [整体attention测试](../../../../benchmarks/qsa/test_attention.py#L1) | 整体正常测试与性能；输入/FP32参考和单kernel检查位于[tests/ops/qsa](../../../../tests/ops/qsa) |
| [indexer.py](indexer.py) | prefill indexer：Triton q_prep/k_compress（逐bit）、布局与两个FlyDSL kernel launch；decode入口`decode_indexer`（分页logits＋SGLang fast_topk/expand）与`decode_forward`（Triton `_indexer_decode_prep`一次完成q norm/RoPE、ring写入和组边界压缩，逐bit）；无SGLang顶层依赖 |
| [indexer_logits.py](indexer_logits.py) / [indexer_topk.py](indexer_topk.py) | FlyDSL MFMA relu-logits；每wave一行的top-512块选择及2051 token展开（2026-09-29由HIP改写，逐bit一致） |
| [indexer_decode.py](indexer_decode.py) | FlyDSL decode分页logits：静态grid、设备端长度，合并1 KiB load经每wave 4 KiB LDS转置，MFMA 16×16×16以头为行 |
| [整体indexer测试](../../../../benchmarks/qsa/test_indexer.py#L1) | indexer真实capture/合成重放、插件校验流程与三臂性能；decode合成、graph重放、回退与`--decode`/`--decode-forward`性能 |

当前auto：packed direct可用时，以$W_U=128\cdot64\lceil U/16\rceil$、$W_D=16\cdot32\sum_i\lceil n_i/32\rceil$比较，$10W_U\le17W_D$时用union，否则direct；$n_i$是实际选中token数。
raw/ragged仍按$U r \le 4\sum_i s_i$选择，$s_i$为query包含尾块的block数。requested BQ32按M128容量限制，TP2/4/8为**10/16/32**；
G12的grid上限为CU，其它G为`CU×2`。每次重建后按N64成本排序、蛇形分配固定worker，排序按最多4096任务分chunk；没有atomic队列或CPU回读。
没有公开mode/rho/BN/BQ等调参接口。内部依赖只有[共用helper](../mha/_common.py)与[linear D256 kernel](../mha/mha_pa_bf16_256_linear_942.py)，不依赖SGLang。

## 必要测试

整体attention入口为[benchmark](../../../../benchmarks/qsa/test_attention.py#L1)，indexer入口为[benchmark](../../../../benchmarks/qsa/test_indexer.py#L1)，单kernel检查已拆至[tests/ops/qsa](../../../../tests/ops/qsa)。当前操作方式与完整数量以[新说明](../../../../benchmarks/qsa/readme.md)为准；下方65项等数字对应历史轮次。`QSA_REPLAY_GPU`选卡，`QSA_INDEXER_INPUT_DIR`覆盖indexer capture目录。

- 普通pytest：13种正常形状（dense边界、TP2/4/8、ragged/空段、NaN尾部、alias、rescale、选择更新和graph），
  8份真实layer/rank/shape输入各跑TP2/4/8 local-head的3D重放、原3D SGLang backend流程、union排序、dense任务/graph、direct共享排序及packed KV刷新/NaN尾/动态graph、分流整数cutoff/HK2/raw/ragged和scratch无回读复用，另含CPU预算边界及TP2/4/8的64MiB边界fallback/graph。异常输入恢复oracle覆盖正/反序launch，共65项，保留逐bit mask、消费后scratch归零和64bit排序fallback；原`rtol=atol=.02`不变。最新.venv实际65通过、24 perf deselected、0skip，见[JUnit](../../../../mytest/mydata/qsa_prepare_latency_20260928_01/validation/tests.xml)。资源fixture同时检查FlyDSL及准备Triton实际ELF三字段。
- 真实输入默认使用已有8份采集，可用`QSA_REAL_INPUT_DIR`覆盖。
  显式目录不存在/无数据时报错；无本地数据的环境仍可运行合成功能测试。
- 性能需显式`-m perf`并设置`QSA_REPLAY_OUTPUT`到新的mytest/mydata子目录；24项（8capture×TP2/4/8），每项原cudaPerf、10独立buffer、2warmup、每实现128sample、AB/BA交错。
  `QSA_REPLAY_REQUIRE_HALF=1`要求完整QSA不超过base一半。
- 直接运行同文件CLI：`--output`必需；`--inputs`指定采集文件，`--gpu`选卡，`--tp-sizes`默认2/4/8，`--buffers`默认10、`--samples`默认128且不得小于buffer数，`--check-only`仅正确性，`--require-half`启用半base门槛。结果记录source_tp_size/derived_head_slice，不将派生重放写成真实多卡capture。

每case默认`base`/`attention`两个scope、共256raw；attention包括恢复/校验、rebuild和dispatch，预分配输出，不计首次JIT、indexer或KV gather。新结果的summary/raw标签为`attention`，历史`qsa`标签不回写。
当前入口不再读取5D capture；历史5D研究应使用当时冻结源码与环境，不能直接套用当前3D入口。
10组Q/K/V/indices/O独立分配；private scratch由单接口在同一stream复用，不等于10份PK/PV或强制cache flush。
当前membership暂存分配时为零，构表消费后归零；不得把它当永久membership结果读取。prepared-only测试/研究入口是`attention_prepare.allocate_plan/rebuild_plan`，历史脚本需使用当时冻结版本，不保留旧模块兼容shim。

后续优化及统一性能表集中追加到[opt.md](opt.md)，不再新建Markdown报告；上方各32buffer章节仅保留其历史含义。
检查**实际计时输出**、重复输入bitexact、真实地址、use≤5%/VRAM≤20%及PTL Enabled/VECTOR,F8；失败保留全部raw、不轮询、不改硬件。
不是TTFT或端到端性能声明。历史50/70raw报告保持其原口径。

## 临时SGLang与历史证据

SGLang仅在[sglang/README.md](sglang/README.md)所述临时目录：冻结baseline合成一个文件，插件一个文件，模型profile/输入采集一个脚本。
原分散的回放、summary/audit、构建包装和插件测试入口已删除；一次性核验证据放mytest/mydata，不作为永久工具维护。

- [本次整理验证](../../../../mytest/mydata/qsa_consolidation_20260925_01/README.md)：72项正常回归、13个ELF逐位一致；新性能在采样前门禁失败，无有效计时，整模型未重跑。
- [历史真实输入正式性能](../../../../mytest/mydata/qsa_real_study_20260925/README.md)：旧接口auto4完整3.82–4.60ms、base9.71–9.82ms；不是本次重测结果。
- [历史模型profile](../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/README.md)：0.1.4、48calls/rank三段216.74/216.49ms；非端到端2×。
- [opt.md](opt.md)按轮次追加记录；旧数据、trace、冻结包与收据不改写。
