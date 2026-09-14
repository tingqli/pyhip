# MoE down：相对PyHIP的优化与四个固定case

本文对应2026-09-14清理后的主代码。以测试中的 **PyHIP FP8 down＋Torch TOPK sum** 为基线，先列公共增量，再分别说明 **256x128 persist、256x128、128x128、128x128 persist**。历史筛选数字不作为每项优化的独立收益证明。

## 1. 名称、任务粒度与基线

[case入口和工厂](test_blockscaled.py#L25-L51)保留七项；旧FlyDSL BN32/BN64仍作对照，本文重点说明四个packed case。

`Mx128`中的M是计算/排序行块；128是该实现的N128组织单位，**不是一次任务只算128列**。B的实际DMA和单个Compute以 **N64×K256** packet进行。一次逻辑任务覆盖`M × (N / OC分片数)`，内部循环多个packet。

| case | M／sort粒度 | waves/CTA | OC分片数S | 一次逻辑任务的N范围 | 调度 | task swizzle | 输出策略 |
|---|---:|---:|---:|---|---|---|---|
| PyHIP | 256 | 8 | 1 | 全N，内部按N64循环 | 256个persistent CTA，全局atomic | 无启用的task swizzle | routed，默认store策略 |
| **256x128 persist** | 256 | 8 | 4 | N/4 | 256个persistent CTA，全局atomic | 无 | packed，NT／aux2 |
| **256x128** | 256 | 8 | 4 | N/4 | 每task独立CTA | **width8** | packed，NT／aux2 |
| **128x128** | 128 | 4 | 8 | N/8 | 每task独立CTA | **width2** | packed，SC1/NT／aux18 |
| **128x128 persist** | 128 | 4 | 8 | N/8 | 512个persistent CTA，8个逻辑分片队列 | **width4** | packed，SC1/NT／aux18 |

以上CTA数量是launch/work分配方式，不等于同时驻留数量。硬件为gfx950／MI350X，256 CUs、8 XCDs；`width2/4/8`是逻辑置换宽度，不是使用的物理XCD个数。

### PyHIP已经有的机制——不能算成新增

以[真正的基线调用](test_blockscaled.py#L71-L86)为准，进入的是[PyHIP down](../../../src/contrib/moe_gemm_8wave.py#L781-L864)，不是同文件的gate/up函数。

- 全局`atomicAdd(1)`领任务，LDS向整个CTA广播，任务内复用A寄存器片段。
- FP8原生MFMA16×16×128，K128独立scale，FP32乘法/FMA后routing乘法。
- B权重pre-shuffle、直接global→LDS、四个B槽，以及8 waves分成4＋4错相执行Memory/Compute。
- BF16 conversion和`permlane16_swap`在寄存器中完成；**原基线也没有C LDS shuffle**。

代码：[A加载与四槽B DMA](../../../src/contrib/moe_gemm_8wave.py#L869-L947)、[FP8 MFMA/反量化/打包](../../../src/contrib/moe_gemm_8wave.py#L969-L1077)、[错相流水](../../../src/contrib/moe_gemm_8wave.py#L1121-L1225)。默认`M256/8 waves`每wave是32行，`nrM=2`；不能照搬源码中遗留的`nrM=4`注释。

本测试的PyHIP输出为`[tokens, TOPK, N]` routed BF16，最终使用[预分配Torch sum](test_blockscaled.py#L89-L112)。源码里另有手写reduce函数，但**本基线没有调用它**。

## 2. 四个case共享的增量优化

### 2.1 OC拆分：缩短单任务，增加可调度任务

PyHIP基线OC1，一个claim计算整个N。新M256为OC4，新M128为OC8，多个任务分别计算同一M块的不同输出列，增加调度粒度。

默认tokens16384／TOPK8／E384／N6144／K256：

| 路径 | valid排序行 | 有效M块 | 逻辑任务数 | 每task列数 | 每task N64 packet数 |
|---|---:|---:|---:|---:|---:|
| PyHIP／sort256／OC1 | 196608 | 768 | 768 | 6144 | 96 |
| M256／sort256／OC4 | 196608 | 768 | 3072 | 1536 | 24 |
| M128／sort128／OC8 | 148096 | 1157 | 9256 | 768 | 12 |

收益意图是更细的负载平衡和N维并行；代价是更多task启动/领号/metadata，以及A按OC重复加载。不同OC读不同B列，合起来才是一份该M块所需的B，**不要再把B字节总量乘OC**。代码：[M256划分](moe_multistage_down.py#L118-L143)、[M128划分](moe_multistage_down_m128.py#L34-L55)。

### 2.2 把“反量化＋上一结果打包”穿插在MFMA之间

PyHIP已有延迟反量化队列，但其BF16/routing打包位于当前MFMA批次之后。新流水把已完成的另一半/前一packet的routing乘法、BF16 conversion、permlane/DPP拆成小操作，穿插到当前MFMA间；旧C的store继续延迟到后续阶段。

- 每Compute仍是16条MFMA；不是更换为低精度近似或减少所需dot。
- 以`pending/dequant`短队列延迟消费MFMA结果，用独立操作填依赖间隔；源码`remaining=7`是调度预算，不是所有ISA间隔都严格7条VALU的保证。
- 地址预先准备，热点片段局部pin到VGPR；uniform scale经标量广播，控制长生命周期值，避免意外spill。

代码：[M256 compute_stage](moe_multistage_down.py#L340-L418)、[M128 compute](moe_multistage_down_m128.py#L263-L339)、[寄存器/地址helper](moe_multistage_down.py#L39-L62)。M128与M256实际store时机不同，见后文各case。

### 2.3 从routed scatter改为packed合并写，再由专用reduce消费

PyHIP沿原token/slot写routed行；同wave的相邻排序行可能落在不相邻的token地址。新实现先写：

`[sorted_M_block, global_N64, row_in_M_block, col64]`。

在原BF16/permlane基础上，增加[行低位与列位的DPP交换](moe_multistage_down.py#L64-L115)，让相应lane的16B store组成连续128B地址片段；**不是一条线程指令写128B**，也不需要C进LDS。M256的[packed地址](moe_multistage_down.py#L256-L266)与M128的[地址/目标行掩码](moe_multistage_down_m128.py#L177-L214)都必须和重排匹配。

新的消费者直接从packed布局gather：

1. `inverse.fill_(-1)`并重建`inverse[token,slot]=sorted_position`，只扫描valid前缀。
2. 256线程、2048列/CTA，提前读TOPK位置和片段，FP32顺序累加后转BF16。
3. 没有额外的“先restore成routed再sum”中间步骤；但inverse和gather本身有成本，必须计入Full。

代码：[inverse有效前缀](../../../src/contrib/flydsl/moe_gemm_2stage/moe_reduce.py#L110-L136)、[packed reducer](moe_multistage_reduce.py#L20-L59)、[共享pipeline](moe_multistage_pipeline.py#L34-L82)。

若排序位置为l、输出列为j、Block M为B，则packed元素索引为：

$$
E(l,j)=\lfloor l/B\rfloor BN+\lfloor j/64\rfloor B64+(l\bmod B)64+(j\bmod64).
$$

实际分配是`[capacity_blocks*B,N]`，OC不增加行数。二维shape只是存储容器，不能直接当普通行主序结果。更大的padding分配容量不等于有效输出写流量。

### 2.4 明确B/C的cache policy，保持原数值顺序

四个新case的B DMA均为aux16／SC1；M256输出aux2／NT，M128输出aux18／SC1+NT。专用reduce中间读取aux2／NT，最终输出aux0。PyHIP基线的[Buffer load默认未加SC1](../../../src/core/asmjit.py#L672-L684)，[store调用没有NT扩展标志](../../../src/contrib/moe_gemm_8wave.py#L1099-L1110)。这些是当前实测选择，不是通用“开NT就快”的规则。

四个case仍保持：K128 partial分别缩放→第二项FP32 FMA→routing乘法→每route BF16 RNE→TOPK FP32 sum→最终BF16。没有把scale/routing重结合，也没有在BF16舍入前跨route求和。代码：[M256显式FMA](moe_multistage_down.py#L377-L383)、[M128显式FMA](moe_multistage_down_m128.py#L295-L302)、[严格相消回归](test_blockscaled.py#L515-L555)。

## 3. 256x128 persist

在PyHIP的M256、8-wave、全局persistent基础上，采用公共的OC4、packed/DPP、缓存策略和compute/pack交错。

- **每N128组织成两个N64 Memory/Compute对**，四个16KiB B槽、PF3；prologue先发至多3包，依赖允许时提前补q+3。8 waves共享同一份B，各自计算32行，4＋4组错相沿用基线思想。
- **上一N128输出均摊到下一tile两个Memory阶段**，每个阶段发一半；Compute计算当前半同时打包已完成的另一半。最后两个结果由epilogue排空。
- **任务不做swizzle**。所有resident CTA争同一全局counter，claim解码`m=t//4, oc=t%4`；谁先完成谁取下一个task，不能承诺相邻同expert任务总留在同一XCD。
- 调用者counter每次清零，计入Down；每任务结束的DMA/LDS等待与错相barrier保留。

代码：[全局领取](moe_multistage_down.py#L188-L211)、[DMA/scale](moe_multistage_down.py#L278-L316)、[Memory store与compute/pack衔接](moe_multistage_down.py#L420-L448)、[首尾错相及排空](moe_multistage_down.py#L474-L511)、[counter reset](moe_multistage_down.py#L547-L556)。

## 4. 256x128

与256x128 persist使用**同一M256算术、4＋4错相、PF3和packed epilogue**，不同之处只在任务调度：

- 每个逻辑task由一个独立CTA处理，不执行atomic领号，不修改counter。
- 按有效任务前缀做**width8转置**；capacity尾部仍无效，不足8个task的最后一段保留identity。
- 在已观测的`XCD=blockIdx.x%8`放置关系下，把原本随相邻block轮流跨XCD的任务，变成每XCD消费一段连续逻辑`(m,oc)`，有利于同M的A及同expert的B列片段复用。
- 代价是每个task都需要硬件CTA接纳；也可能存在资源/SE回压。更少HBM读取不自动意味着优于persistent。

代码：[独立width8映射](moe_multistage_down.py#L202-L211)、[独立退出/launch grid](moe_multistage_down.py#L509-L525)、[case工厂](test_blockscaled.py#L34-L36)。当前`persistent=False`固定选择此路径，已无额外`xcd_swizzle`开关。

## 5. 128x128

相对PyHIP进一步把M256/8 waves缩成 **M128/4 waves**，OC4再细分为OC8；每wave仍32行，不是将单wave行片段减半。

- **原生sort128减少padding计算**。默认case有效计算行从sort256的196608降至148096；与此同时M块数和task数增加，B的跨M读取与metadata开销不能忽略。不能用旧sort256跑M128后声称这是原生收益。
- **独立CTA＋width2 task swizzle**，不是早期实验的width4，也没有persistent queue。整个CTA无有效route时可跳过计算，但valid前缀中的expert仍必须合法。
- **N128成对旋转遍历起点**，phase取原始block ID的`worker//8`；不是取swizzle后的task。B、scale、packed store使用同一旋转后的索引。
- **提前3包B，PF3逐包发布**；每次Memory把q+3的4次DMA补充分散到16次LDS读之间。Memory priority3、Compute priority0。
- **Compute内分散store**：在q的Compute中写q−2、打包q−1、计算q，保持最多几批结果的明确生存期。不同于M256主要在下一Memory发旧C store。

代码：[shape和width选择](moe_multistage_down_m128.py#L34-L55)、[任务转置/空块检查](moe_multistage_down_m128.py#L91-L131)、[N phase与prologue](moe_multistage_down_m128.py#L139-L165)、[PF3/refill](moe_multistage_down_m128.py#L221-L261)、[Compute store/pack](moe_multistage_down_m128.py#L263-L339)。

## 6. 128x128 persist

保留M128/4-wave/OC8、sort128和上述packed算术，但为跨任务复用采用独立测得的调度与流水组合，**不等于只给128x128打开一个while循环**。

- **8个逻辑分片、512 workers、每次领1个task**。worker的home shard为`worker%8`，该shard的rank r映射成virtual task `v=8*r+home`，再做**width4 swizzle**。这保留模8归属，同时允许同shard的worker动态分工。不是work stealing，也不是直接以真实XCC作为队列下标。
- **N phase取resident worker**的`worker//8`，不取rank；每任务局部重新物化phase，避免衍生N地址被提到persistent外层造成长SGPR生存期。
- **两包B一次发布，四包启动**。偶数q等待/同步当前pair、退休上一pair的读，再往退休槽补下一pair；奇数q不重复publication barrier。仍保留逐包LDS读完成等待，最后不越界refill。
- **两半N128复用B scale**，输出目标valid位用DPP从已加载的行mask精确交换；省去重复ID读取。移除整CTA空route ballot，所有行的输入/输出掩码仍在；valid前缀之外不执行计算，valid0只做终止claim并退出。
- **末退出worker自复位**。每shard64 workers且每worker只做一次越界claim；当返回rank=`T/8+63`时，所有worker都已领取终止rank，不能再访问该head，于是置0。首次分配置零，此后reset在Down内部完成，不再单独launch清零kernel。

代码：[单claim站点、分片和width4](moe_multistage_down_m128.py#L72-L98)、[N相位生存期](moe_multistage_down_m128.py#L139-L150)、[DPP mask](moe_multistage_down_m128.py#L187-L208)、[pair同步/refill/scale](moe_multistage_down_m128.py#L221-L261)、[末尾priority与自复位](moe_multistage_down_m128.py#L361-L387)。

head间距为128B，目的在于隔离队列热点；不是声称所有GPU的L2 line均为128B。公开counter保持原值；私有head不能由调用者修改。缓存callable**不能跨stream并发**；换reduce workspace不能隔离同一个down queue。[状态拥有者与launch](moe_multistage_down_m128.py#L400-L436)、[生命周期和CPU交错证明测试](test_blockscaled.py#L602-L681)。

## 7. Swizzle的完整含义

### 7.1 三种“重排”不能混为一谈

| 重排 | 改变什么 | 不改变什么 |
|---|---|---|
| `shuffle_weight(layout=(16,16))` | B物理存储，适配MFMA lane读取 | 不是任务到XCD的映射；基线已使用 |
| task swizzle | virtual task到`(m,oc)`的置换 | 不搬A/B/C，不改变排序内容或结果位置 |
| M128 packet N phase | 一个task内访问N128对的顺序 | 不改变task集合；B/scale/C要使用同一packet索引 |

### 7.2 有效前缀转置及尾部

记L为`valid_ids[0]`（padded排序有效前缀，**不是实际route数**），B为Block M，S为OC分片数，$T=(L/B)S$。独立CTA的$v=blockIdx.x$；分片persistent的$v=8r+h$。

对转置宽度w，令$c=\lfloor T/w\rfloor$：

$$
t=F_w(v)=\begin{cases}
(v\bmod w)c+\lfloor v/w\rfloor,&v<wc,\\
v,&v\ge wc,
\end{cases}\qquad m=\lfloor t/S\rfloor,\quad oc=t\bmod S.
$$

可整分区是w×c矩阵转置，剩余尾部identity，保持不重不漏；capacity之外的task不会被“卷回”有效域。T=0、T<w也正确。M128的T为8的倍数，width2/4无需有效task余数尾部，但**转置段边界仍可能切开同一个M的OC集合**。

默认shape下：

- M256：$T=3072,w=8,c=384$。
- M128独立：$T=9256,w=2,c=4628$。
- M128 persistent：$T=9256,w=4,c=2314$。

### 7.3 为什么可能改善局部性

先**假设已在该launch验证**物理XCD满足$x=v\bmod8$，写$v=8j+x$。三个置换为：

$$
\begin{aligned}
F_8(8j+x)&=384x+j &&(\text{M256}),\\
F_2(8j+x)&=4628(x\bmod2)+4j+\lfloor x/2\rfloor &&(\text{M128独立}),\\
F_4(8j+x)&=2314(x\bmod4)+2j+\lfloor x/4\rfloor &&(\text{M128分片persistent}).
\end{aligned}
$$

以XCD0为例，j=0…3的逻辑任务如下；这只是映射，不是执行时间线：

| 映射 | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|
| M128/OC8 identity反例 | (M0,OC0) | (M1,OC0) | (M2,OC0) | (M3,OC0) |
| M256/OC4 width8 | (M0,OC0) | (M0,OC1) | (M0,OC2) | (M0,OC3) |
| M128/OC8 width2 | (M0,OC0) | (M0,OC4) | (M1,OC0) | (M1,OC4) |
| M128/OC8 width4 | (M0,OC0) | (M0,OC2) | (M0,OC4) | (M0,OC6) |

identity行是解释OC8置换的反例，**不是PyHIP OC1全局队列**。swizzle让同M的不同OC更集中在一组复用域，同时沿M分段：减少A跨XCD复制的机会；相邻M若同expert，也可能缩短B列片段复用距离。

在上述假设下，CPU枚举的A消费域数为：

| 配置 | 每M块涉及XCD数的直方图（模型） |
|---|---|
| M128/OC8 identity | 1157块×8域 |
| M128 width2 | 1156块×4域，1个边界块×8域 |
| M128 width4 | 1154块×2域，3个边界块×4域 |
| M256 width8 | 768块×1域 |

[CPU双射/相位检查脚本](../../../.github/skills/gpu-task-swizzle/scripts/check_swizzle.py)可复算以上例子；它没有运行GPU，**模型不是实测缓存命中率**。历史width4真实归属支持其中部分关系，[原生sorting及放置证据](../attn_4wave/tools/se-dispatch.md#L276-L319)有明确版本。当前width2不能直接引用旧width4彩图作为自己的测量。

更小消费域不一定更快：OC与M粒度、每域B工作集、请求发射相位、LDS/寄存器资源和硬件接纳都影响性能。当前独立选择width2，persistent选择width4，不是“宽度越大越好”。同一B`(expert,OC)`即使identity也可能只落一个XCD，不能笼统声称swizzle必然减少B的8份复制。

### 7.4 为什么全局persistent不能直接套相同结论

独立CTA中virtual ID和worker ID相同；全局atomic persistent中rank由完成顺序决定，物理worker与rank没有固定模8关系。仅对全局rank做$F_w$，不保证原独立grid的A/B消费域。

M128通过home shard $h=worker\bmod8$、$v=8r+h$保证$v\bmod8=h$。即使物理放置改变，逻辑覆盖仍正确；只有实际worker的XCD与h相合时，才能解释为保持了XCD affinity。当前方案不读取XCC强行绑定CTA；它是在可验证放置关系之上的调度优化。

### 7.5 N相位独立于task swizzle

M128每个OC有$Q=N/(8\times64)$个N64 packet，Q为偶数。令$r=\lfloor worker/8\rfloor$：

$$p(q)=2\big((\lfloor q/2\rfloor+r)\bmod(Q/2)\big)+(q\bmod2).$$

例如N6144时Q=12、r=1，访问顺序是2,3,4,5,6,7,8,9,10,11,0,1。每个N128的两半仍相邻，便于scale复用与pair publication；B读取用p，scale用`p//2`，C存回p。M256两case按N顺序遍历，没有这层worker N旋转。

代码：[统一packet_index和B DMA](moe_multistage_down_m128.py#L146-L160)、[scale索引](moe_multistage_down_m128.py#L247-L253)、[C地址](moe_multistage_down_m128.py#L211-L214)。目标是错开请求/计算相位，但不是自动消除所有VMEM空档。

### 7.6 不沿用错误的SE解释

旧sort256跑M128的`384/384/384/5`非空块周期，会与SE分派相位产生明显别名；原生sort128默认已是`290/289/289/289`。不能把当前收益继续全部归于修复旧padding偏斜，也不能把SE接纳回压说成“四SE必须全部完成一轮”的barrier。[区别与实测反例](../attn_4wave/tools/se-dispatch.md#L186-L292)。

## 8. 使用、验证与证据边界

- 模型输入固定K256、FP8 E4M3FN；M256要求N为512倍数，M128要求N为1024倍数。A scale物理K-major；B scale为`[experts,N/128,2]`；默认真实量化/排序/shuffle见[make_case](test_blockscaled.py#L163-L180)。
- [工厂](test_blockscaled.py#L25-L51)返回down callable；[compile_packed_down_reduce](moe_multistage_pipeline.py#L34-L82)接收已选down，自动使用128/256 stride。返回的十tensor接口最终输出`[tokens,N]` BF16；单独down输出packed，勿重复TOPK sum。
- `--candidate`使用本文四个名称，有空格的case参数需作为一个shell参数传递；`--mode down`只测Down，默认测完整流程；`--profile`单候选23次直接launch。参见[CLI](test_blockscaled.py#L793-L817)。
- 普通计时保留原warmup2/iters10/copies1、同址正反对照；Down包含需要的counter/queue重置，Full直接测down＋inverse清空/重建＋reduce。量化/排序/参考/解码/投毒在计时外。[计时与检查](test_blockscaled.py#L311-L352)。
- 最近代码清理回归为[57项测试通过](analysis/cleanup_20260914/tests_final.log)。四条packed路径清理后两轮Down/Full为：M256 persist515.746/898.656、M256独立513.949/901.661、M128独立474.221/865.687、M128 persist482.697/877.908μs；[原始前后对照](analysis/cleanup_20260914/paired_final.json)仍使用改名前case键。它用于检查清理退化，不是每个新增优化相对PyHIP的消融实验。
- [最近M128 persistent PMC](analysis/pmc_down_20260914/README.md)：清理前同算法快照，DRAM读＋写4.839891TB/s，分母为同次PMC dispatch时间；[最近ATT](analysis/att_down_20260914/README.md)同样是清理前来源。不能将其hash或带宽重标为当前文件，也不能将ATT时长用于PMC分母。
- 本次仅重写文档和提炼skills，没有修改GPU kernel、计时器或重跑GPU。旧基线保留为 [moe_8wave_down.py](moe_8wave_down.py)、[moe_8wave_down_utils.py](moe_8wave_down_utils.py)；[历史索引](try/readme.md)与已有analysis产物只读保留。重写前README已在[上轮清理源码归档](analysis/cleanup_20260914/after_sources.tar.gz)中保存。

## 9. 可复用skills

新建的[仓库skills目录与索引](../../../.github/skills/README.md)按需加载，使用标准Agent Skills格式；每篇有实施步骤、必要前提、反例/验证与当前代码链接。

| Skill | 可复用优化 |
|---|---|
| [task swizzle](../../../.github/skills/gpu-task-swizzle/SKILL.md) | 有效前缀双射、OC拆分、XCD消费域、N相位；附纯CPU检查脚本 |
| [MFMA流水](../../../.github/skills/flydsl-mfma-pipeline/SKILL.md) | 延迟退休、compute/pack重叠、LDS ring/pair发布、生存期与编译器调度 |
| [packed epilogue](../../../.github/skills/moe-packed-epilogue/SKILL.md) | 寄存器DPP合并写、native sorting、inverse/gather reduce、精确舍入 |
| [persistent队列](../../../.github/skills/gpu-persistent-work-queues/SKILL.md) | 分片保持归属、单claim站点、可证明的末退出worker自复位与graph生命周期 |