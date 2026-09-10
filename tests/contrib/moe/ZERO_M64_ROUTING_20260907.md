# 零满块M64与routing分布对照（2026-09-07）

本轮从[TODO](TODO.md)的零满块直接M64对照继续，不重复此前已经完成的K192/K320普通Down和CU拆分性能。

## 结论

- **Hy3 1K/2K零满块同配置对照已完成48轮。** 直接M64相对compact的Down配对时延降低6.971%/4.635%，
  Combined降低5.001%/3.404%，均48/48胜；**Full的IQR均跨零，不能据此自动切路径**。
- **routing首测取得12个通过入场/退出负载检查的点**：Hy3 1K/2K/8K、Qwen397 K512 16K，
  各覆盖确定性均衡、随机均匀和偏斜分布。它们是合成routing，不是生产trace。
- 后续独立48轮已完成**7份/10点**：Hy3 skew 8K、Qwen397三分布16K、Qwen35 uniform 16K的compact
  对全部候选三phase IQR均正，其余5个Qwen35点仍有Full或default竞争边界。**详见第7节，不是在线selector已实现。**
- Qwen35均衡运行退出时检测到busy18%、VRAM46%，**整份两点结果隔离为负载风险，不参与结论**；
  此前均匀分布入场拒绝、偏斜未启动。后续三分布的新48轮均通过门禁，**不改写旧风险状态、不拼接样本**。
- 生产kernel、默认分块、CU阈值、自动selector及`sorted_sum()`均未修改；随后完成独立的
  [sum线程测试](SUM_THREADS_20260907.md)，候选只在tests，生产64默认不变。所有已接管运行的硬件配置恢复原状态。

证据：[routing_summary.json](results/todo_20260907/routing_summary.json)，
由[独立审计](summarize_todo_routing.py)逐原始样本、源/快照、编译manifest、ISA和工作量复算。
原始负载风险和门禁拒绝文件没有删除、过滤或改写为通过。

## 1. 协议与控制变量

设备：MI308X/gfx942，物理GPU7，API可用CU数80。正式测量要求全机busy≤5%、VRAM≤20%，
目标auto/650W入场，PTL **Enabled / VECTOR,F8**、1800MHz determinism、NUMA0；结束恢复auto、650W、
PTL Disabled/N/A、NUMA1。检查在入场、设置后及退出进行，**不构成全程连续外部负载监控**。

- 10-buffer，每轮候选顺序轮转后正序＋反序；每半轮使用相同buffer索引。
- **Down/Combined**：相同输入、权重、物理输出base；同BM64的直接M64与compact共享metadata及loc指针、同一个sum callable。
  公共gateup在event外prime，逐轮event外fresh检查reference、finite和padding/inactive哨兵。
- **Full**：同半轮复用相同输入/权重buffer，包含sorting、量化、gateup、down、invert、sum。
  原harness仍在每次调用内分配中间输出和loc，**不是所有Full中间指针固定相同地址**；没有使用CUDA graph计时。
- 直接M64与compact固定BM64 metadata、padding128B、nt store、FP8、原`tile_k=128`；不混入新BK192的收益。
  普通8x1使用BM256；Hy3另外保留原1x8/padding0作为独立既有候选。
- 每个运行保存实际driver快照、完整源集合及schema2编译身份；合成routing改变输入内容，不改变相同shape的机器码。
- 所有测量保留原始样本，不将不同运行/不同候选集合/不同routing的时延混为同一次配对。

$$
F_{Down}=F_{Combined}=2B\cdot TopK\cdot N\cdot K,\qquad
F_{Full}=3F_{Down},\qquad
T_{effective}=F/(t_{ms}\times10^9).
$$

本文TFLOPS全部是实际event时延对应的有效工作量，不是ATT模型TFLOPS。
配对改善按每轮同版本两个样本的均值计算，再取中位和IQR，不一定等于两列绝对中位时延之比。

## 2. 零满块直接M64：48轮

Hy3配置N4096、K192、E193、TopK9、per-tensor。逐expert counts确认：

| Batch | useful rows | M64执行行 | full / tail任务 | 普通BM256执行行 |
| ---: | ---: | ---: | --- | ---: |
| 1K | 9216 | 12352 | 0 / 193 | 49408 |
| 2K | 18432 | 24704 | 0 / 386 | 49408 |

此处full为拆分前/后的真实几何计数，均为0；不是拿非零分配capacity冒充满块数学。
直接M64不再构表或启动空full kernel，其余sorting/gateup/quant/sum配置一致。

每个时延后同时列有效TFLOPS：

| Batch | Phase | compact ms / TFLOPS | 直接M64 ms / TFLOPS | 直接M64配对时延降低 [Q1,Q3] | 胜场 |
| --- | --- | ---: | ---: | --- | ---: |
| 1K | Down | 0.104200 / 139.11 | 0.096880 / 149.62 | +6.971% [6.586,7.464] | 48/48 |
| 1K | Combined | 0.133081 / 108.92 | 0.126160 / 114.90 | +5.001% [4.705,5.511] | 48/48 |
| 1K | Full | 0.387021 / 112.36 | 0.381381 / 114.02 | +3.053% **[-17.463,18.087]** | 27/48 |
| 2K | Down | 0.159520 / 181.74 | 0.152201 / 190.48 | +4.635% [4.366,4.992] | 48/48 |
| 2K | Combined | 0.218461 / 132.71 | 0.211241 / 137.24 | +3.404% [3.039,3.775] | 48/48 |
| 2K | Full | 0.560962 / 155.04 | 0.557202 / 156.09 | -0.293% **[-10.401,11.905]** | 22/48 |

两点Down绝对中位差都约**7.32µs**，是这次匹配对照中构表、空full launch、descriptor尾路径等差异的**合计观测**，
没有把它拆成各组件的独立成本；也不能用Combined−Down两组中位数推断独立sum时延。
Full离散较大，两点均未达三phase共同晋级门槛。不能仅因为绝对中位数稍低而把1K/2K自动切到直接M64。

原始记录：[hy3_confirm48_retry1.json](results/todo_20260907/zero_m64/hy3_confirm48_retry1.json)。
此前[入场拒绝](results/todo_20260907/zero_m64/hy3_confirm48.json)保留，未与本次样本拼接。

## 3. 同Batch、不同routing的实际几何

三种routing定义：

- `balanced`：route slots对E取余，确定性round-robin。
- `uniform`：独立随机score取top-k，无放回均匀分布。
- `skew`：Gumbel top-k加Zipf权重（指数1.2），无放回偏斜分布。

每个Batch固定seed，10个buffer使用相同routing内容。完整expert counts保存在原JSON；
它们是**计时前、仅用于验证的GPU→CPU统计**，没有进入被测选路或Full event，不能当成免费的在线host hint。

| Case / Batch | 分布 | 拆分前F/T | 拆分后F/T | M64执行行 | BM256执行行 |
| --- | --- | --- | --- | ---: | ---: |
| Hy3 1K | balanced | 0/193 | 0/193 | 12352 | 49408 |
| Hy3 1K | uniform | 0/194 | 0/194 | 12416 | 49408 |
| Hy3 1K | skew | 17/213 | **0/281** | 17984 | 52992 |
| Hy3 2K | balanced | 0/386 | 0/386 | 24704 | 49408 |
| Hy3 2K | uniform | 0/387 | 0/387 | 24768 | 49408 |
| Hy3 2K | skew | 41/239 | **0/403** | 25792 | 58368 |
| Hy3 8K | balanced | 193/386 | 160/518 | 74112 | 98816 |
| Hy3 8K | uniform | 193/476 | 160/608 | 79872 | 98816 |
| Hy3 8K | skew | 229/326 | **229/326** | 79488 | 102144 |
| Qwen397 K512 16K | balanced | 512/512 | 480/640 | 163840 | 262144 |
| Qwen397 K512 16K | uniform | 512/753 | 480/881 | 179264 | 262144 |
| Qwen397 K512 16K | skew | 510/749 | 480/869 | 178496 | 251136 |

Hy3偏斜1K/2K虽然形成17/41个full，仍全部被现有严格`0<5*r<3*CU`规则拆回M64；只看拆分前F是否非零会误判。
Hy3偏斜8K的余数为69/80，不触发拆分，与同Batch均衡/均匀的160个保留full明显不同。
因此Batch本身不能描述满尾、M填充或CU末轮成本。

## 4. routing性能首测：24轮（历史首测状态，确认见第7节）

本节只列通过入场/设置后/退出负载检查的运行。比较compact相对**同padding128的直接M64**，
不是相对旧padding0的独立1x4；完整default/普通8x1/Hy3 1x8数据都在原JSON。

| Case / 分布 | Phase | 直接M64 ms / TFLOPS | compact ms / TFLOPS | compact配对时延降低 [Q1,Q3] |
| --- | --- | ---: | ---: | --- |
| Hy3 8K / balanced | Down | 0.397241 / 291.92 | 0.391301 / 296.36 | +1.612% [1.167,2.858] |
| Hy3 8K / balanced | Combined | 0.608562 / 190.55 | 0.602682 / 192.41 | +1.045% [0.562,1.515] |
| Hy3 8K / balanced | Full | 1.454125 / 239.25 | 1.443226 / 241.05 | +0.922% **[-0.573,2.284]** |
| Hy3 8K / uniform | Down | 0.423862 / 273.59 | 0.418621 / 277.01 | +1.288% [0.907,2.056] |
| Hy3 8K / uniform | Combined | 0.627442 / 184.82 | 0.621922 / 186.46 | +0.911% [0.369,1.340] |
| Hy3 8K / uniform | Full | 1.492066 / 233.16 | 1.481906 / 234.76 | +0.398% **[-0.625,2.012]** |
| Hy3 8K / skew | Down | 0.466961 / 248.34 | 0.443022 / 261.76 | +5.039% [4.688,5.248] |
| Hy3 8K / skew | Combined | 0.658962 / 175.98 | 0.635883 / 182.37 | +3.601% [3.353,3.767] |
| Hy3 8K / skew | Full | 1.543686 / 225.36 | 1.513826 / 229.81 | +1.896% [0.251,2.720] |
| Qwen397 K512 16K / balanced | Down | 1.979887 / 347.09 | 1.564986 / 439.11 | +21.211% [21.069,21.400] |
| Qwen397 K512 16K / balanced | Combined | 2.440450 / 281.59 | 2.031308 / 338.30 | +17.162% [16.904,17.261] |
| Qwen397 K512 16K / balanced | Full | 5.886063 / 350.25 | 5.438341 / 379.08 | +7.664% [6.954,8.020] |
| Qwen397 K512 16K / uniform | Down | 2.121148 / 323.97 | 1.771666 / 387.88 | +16.823% [16.700,16.923] |
| Qwen397 K512 16K / uniform | Combined | 2.557129 / 268.74 | 2.210788 / 310.84 | +13.822% [13.689,13.928] |
| Qwen397 K512 16K / uniform | Full | 6.334183 / 325.47 | 5.938922 / 347.13 | +6.376% [5.841,6.655] |
| Qwen397 K512 16K / skew | Down | 2.156568 / 318.65 | 1.729487 / 397.34 | +19.942% [19.858,20.028] |
| Qwen397 K512 16K / skew | Combined | 2.580130 / 266.34 | 2.153168 / 319.16 | +16.676% [16.614,16.782] |
| Qwen397 K512 16K / skew | Full | 6.215343 / 331.69 | 5.757081 / 358.10 | +7.447% [6.548,7.911] |

原始记录：
[Hy3 balanced](results/todo_20260907/routing/hy3_balanced_abba24.json)、
[uniform](results/todo_20260907/routing/hy3_uniform_abba24.json)、
[skew](results/todo_20260907/routing/hy3_skew_abba24.json)；
[Qwen397 balanced](results/todo_20260907/routing/qwen397_balanced_abba24.json)、
[uniform](results/todo_20260907/routing/qwen397_uniform_abba24.json)、
[skew](results/todo_20260907/routing/qwen397_skew_abba24.json)。

### 离线选路证据，而非生产selector

[evidence_frontier()](summarize_todo_routing.py)只根据三phase配对Q1是否大于0构造优胜关系：
候选需要对所有其他已测路径都有三phase正IQR，才列为“待确认共同优胜者”。负载风险运行不参与。

- Hy3 1K/2K三种routing均没有三phase共同优胜者，不能套`F=0 => direct_m64`宣称已验证Full晋级。
- Hy3 8K balanced/uniform没有共同优胜者；skew的compact有初步共同优势，但Full Q1较窄，必须48轮确认。
- Qwen397 K512 16K三种分布compact均为共同优胜候选，下一步独立48轮确认，不与此前CU策略对照拼接。
- 本轮只是单seed的合成分布首测，未测试真实生产路由trace或跨seed推广，不能发布泛化的分布阈值。

现有[compile_moe_gemm2()](../../../src/contrib/flydsl/moe_gemm_2stage/gemm2.py)按显式`down_path`分发，
没有接收在线expert histogram；[测试配置选择](test_moe.py)也只是将路径名映射到tile参数。
后续接入应区分两种情况：

1. 调用方本来已持有host routing统计/模型hint：可利用它评估满尾、M64/BM256 padding及CU余数，不新增GPU同步。
2. routing统计仅在GPU：只能设计device侧决策/guard等方案，或保守保持显式路径；不能先`.cpu()`读取counts再称无同步selector。

GPU guard能避免数学执行，却不消除host launch成本；它与完全绕过构表/空full launch不是同一优化。
本轮未新增host hint参数，也未修改自动selector或默认路径。

## 5. 负载风险与下一步（首测结束时点，后续进展见第7节）

| 运行 | 状态 | 处理 |
| --- | --- | --- |
| 零满块Hy3 48轮，两点 | 入场/设置后/退出合格 | 保留全部样本，Down/Combined确认，Full不判晋级 |
| Hy3三分布、Qwen397三分布24轮，共12点 | 入场/设置后/退出合格 | 首测证据，待独立48轮及真实分布推广 |
| [Qwen35 balanced 24轮](results/todo_20260907/routing/qwen35_balanced_abba24.json)，8K/16K | 退出busy18%、VRAM46% | **整份2点/1152个event隔离**，不猜外载何时开始，不只删慢样本 |
| [Qwen35 uniform](results/todo_20260907/routing/qwen35_uniform_abba24.json) | 入场busy15%～22%、VRAM约46.5% | worker未启动、0样本、未修改硬件设置 |
| Qwen35 skew | 未启动 | 不越过门禁继续 |

优先恢复：Qwen35 balanced清洁重测 → uniform/skew → Hy3 skew 8K和Qwen397三分布48轮确认。
Full仍跨零的零满块场景保持待定；若要细分7.32µs总开销，需独立测构表/空full/descriptor成本，当前未完成。
随后再设计真实routing selector，按TODO继续sum线程实验；K192/K320新分块Combined/Full及旧K320随机异常也仍是待办。

## 6. 首测验收与限制

- 7份通过负载门槛的运行：零满块2点＋routing12点，**9936个event**逐原始数据复算。
- Qwen35风险运行的1152个event完整保留、整份排除；一份新入场拒绝0样本。
- 所有完成运行的逐轮正确性/存储合同通过，最大相对default误差0.00015139；
  性能输入仍为全1权重，不能替代随机权重或生产路由正确性覆盖。
- schema2源/快照/launcher配置与缓存key、ISA/ELF、零spill、8x1动态MFMA/VMEM消费者及真实nt store已复核。
- **33项CPU合同、负载风险隔离、三phase筛选负例与harness回归通过**（0.16s），
  [JUnit](results/todo_20260907/routing_tests.xml)。
- 无生产kernel改动、无自动selector、无sum线程改动、无fresh ATT；不把前轮或风险数据重标为本轮确认。

## 7. 后续：Qwen35补测及routing独立48轮确认

7份新运行均status complete，入场/设置后/退出检查通过，退出目标busy/VRAM均0，硬件设置全部恢复。
10个点、**11808个event**由同一[审计器](summarize_todo_routing.py)独立复核，
包括源/driver快照、schema2身份、ISA/ELF/零spill、8x1动态MFMA/VMEM、nt store、逐expert几何及原始配对统计。
[确认轮审计](results/todo_20260907/routing_confirm/audited.json)独立保存，不覆盖首测汇总。

### Qwen35的routing几何

N2048/K512/E256/TopK8/PTPC，仍使用原BK128，不混入sum线程候选（本节sum全部为生产64）。

| Batch / 分布 | 拆分前F/T | 拆分后F/T | M64执行行 | BM256执行行 |
| --- | --- | --- | ---: | ---: |
| 8K / balanced | 256/0 | 240/64 | 65536 | 65536 |
| 16K / balanced | 512/0 | 480/128 | 131072 | 131072 |
| 8K / uniform | 256/119 | 240/183 | 73152 | 96000 |
| 16K / uniform | 512/129 | 480/257 | 139328 | 163840 |
| 8K / skew | 192/373 | 160/501 | 73024 | 110080 |
| 16K / skew | 429/452 | 400/568 | 138752 | 166656 |

Hy3 skew 8K与Qwen397三分布16K的几何与第3节24轮相同，seed/生成规则未改。
Qwen35均衡虽然M64/M256执行行相同，CU拆分后仍有tail；不能把它当成纯full无tail对照。

### Full确认：compact相对直接M64

所有10点compact对直接M64的Down、Combined和Full IQR均正，但**不等于对其它候选也赢**。
表中只展示Full时延及对应有效TFLOPS，三phase全部候选配对保存在审计及原JSON。
有效工作量仍为第1节的$F_{Full}=6B\,TopK\,NK$，不是ATT模型TFLOPS。

| Case / Batch / 分布 | 直接M64 ms / TFLOPS | compact ms / TFLOPS | compact配对降低 [Q1,Q3] | 对全部候选三phase共同胜？ |
| --- | ---: | ---: | --- | --- |
| Hy3 8K / skew | 1.520447 / 228.81 | 1.498506 / 232.16 | +1.732% [0.707,2.503] | 是 |
| Qwen397 16K / balanced | 5.900547 / 349.39 | 5.454064 / 377.99 | +7.527% [6.867,7.851] | 是 |
| Qwen397 16K / uniform | 6.326047 / 325.89 | 5.942106 / 346.95 | +6.352% [5.584,6.670] | 是 |
| Qwen397 16K / skew | 6.204067 / 332.30 | 5.744004 / 358.91 | +7.434% [6.806,7.775] | 是 |
| Qwen35 8K / balanced | 1.370627 / 300.82 | 1.248346 / 330.29 | +8.850% [5.456,11.965] | 否：与ordinary Full边界 |
| Qwen35 16K / balanced | 2.581833 / 319.40 | 2.371072 / 347.79 | +8.245% [6.865,9.238] | 否：与ordinary Full边界 |
| Qwen35 8K / uniform | 1.469428 / 280.60 | 1.364026 / 302.28 | +7.128% [3.831,10.096] | 否：与default Full边界 |
| Qwen35 16K / uniform | 2.724313 / 302.69 | 2.529592 / 325.99 | +7.151% [6.000,8.124] | 是 |
| Qwen35 8K / skew | 1.486947 / 277.29 | 1.416826 / 291.01 | +4.758% [1.878,7.320] | 否：default的Down更快 |
| Qwen35 16K / skew | 2.762973 / 298.46 | 2.590552 / 318.32 | +6.245% [5.164,7.218] | 否：与default Full边界 |

边界明确保留，不能只看上表对直接M64的正收益：

- Qwen35 balanced 8K/16K，compact对ordinary Full为+2.610% **[-1.165,6.150]**、+0.289% **[-0.949,1.850]**。
- Qwen35 uniform 8K，compact对default Full为+3.055% **[-0.421,6.177]**。
- Qwen35 skew 8K，compact对default Down为**-0.460% [-0.743,-0.224]**，Combined/Full也跨零。
- Qwen35 skew 16K，compact对default Full为+0.746% **[-0.433,1.808]**。
- Qwen35 uniform 16K，compact对default Full为+2.126% [1.042,3.341]，对ordinary为+8.422% [7.260,9.472]，三phase共同通过。

原始48轮：
[Hy3 skew](results/todo_20260907/routing_confirm/hy3_skew_confirm48.json)；
Qwen397 [balanced](results/todo_20260907/routing_confirm/qwen397_balanced_confirm48.json)、
[uniform](results/todo_20260907/routing_confirm/qwen397_uniform_confirm48.json)、
[skew](results/todo_20260907/routing_confirm/qwen397_skew_confirm48.json)；
Qwen35 [balanced](results/todo_20260907/routing_confirm/qwen35_balanced_confirm48.json)、
[uniform](results/todo_20260907/routing_confirm/qwen35_uniform_confirm48.json)、
[skew](results/todo_20260907/routing_confirm/qwen35_skew_confirm48.json)。

### 当前判断与下一步

5/10点有compact共同优势。Hy3 skew与Qwen397三分布已有24＋48一致证据；Qwen35三分布为新合格48轮，
没有把旧风险24轮当作独立支持。仍是**单seed合成分布**，不是生产trace或已泛化的选择规则。
现阶段保留显式路径；在线接入仍需host已有统计hint或device侧设计，禁止新增GPU counts回读CPU来伪装无同步selector。

随后已推进[sum线程实验](SUM_THREADS_20260907.md)，不改生产默认，不重跑已完成的零满块对照。
真实routing推广/自动接入、零满块合计开销分解、K192/K320新分块Combined/Full及旧K320随机稳定性仍待完成。