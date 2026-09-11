# MoE Down ATT与Stall分析方法

## 目标

本手册把一次stall观察变成可复验的优化闭环：

```text
正确性 -> ISA/资源 -> clean ABBA -> fresh ATT
         -> physical SIMD union -> exclusive owner -> 单变量候选 -> 重新闭环
```

ATT用于解释“时间在哪里、候选为何生效”，不能替代墙钟。只有目标stall下降、physical MFMA union busy改善且同进程ABBA墙钟同时改善，候选才可晋级。

> **当前8x1入口**：K192默认128+64，K320默认128+128+64；只接受`tile_k=None/128`。
> 3×64/5×64已从当前实现删除，历史对照须加载旧源码快照。默认切换不代表K192达到80%；
> 其已测PTPC busy仍为69.280%。通用K128 rolling遗留及恒真条件也已清理；新增per-tensor支持，
> 但未采新量化ATT。删除与回归见[清理验收](#bk64-cleanup)，最新完整链选择见
> [1K--32K性能报告](../../../contrib/moe/MAIN_MERGE_PERFORMANCE_REPORT.md)。

> 2026-09-06七K当前PTPC baseline已重新采集并复制到仓库根，逐文件校验与入口见
> [compact验收/UI清单](../../../contrib/moe/COMPACT_M64_REPORT.md#七k-baseline-ui)。
> 本轮随后新增显式`8x1_compact`，这些UI是改动前旧8x1 trace，不用于解释新descriptor/构表的stall；
> 未将单wave raw stall换算成physical idle，也未用baseline union×roof冒充compact墙钟TFLOPS。

> 2026-09-06随后将K512/PTPC跨N stage1–3的`vmcnt(5)`选择性放宽到9。
> [本轮生产源码验证与性能](#k512-vmcnt9-production)已完成：正确性通过，但独立配对复测没有一致提速，
> 部分配置小幅回退；当前保留用户要求的代码改动，不将其晋级为已验证性能优化。
> 随后按要求完成[普通K512旧5/新9 fresh ATT](#k512-vmcnt9-fresh-att)及仓库根UI副本；
> 该次busy为91.106%→90.128%，观察到VMEM等待向后续`vmcnt(1)`集中，不代替配对墙钟结果。
> 最新已将[stage4的1→5](#k512-stage4-vmcnt5)保留为正式默认，并完成[K512 N循环原型](#k512-nloop-prototype)：
> 同环境N2048冷编译210.916秒→8.612秒，Down-only确认正向；N循环仍需显式`_n_loop=1/2`，默认0不变。
> 上述为K512原型阶段。最新按要求[所有七K默认1N循环及全链验收](../../../contrib/moe/ALL_8X1_NLOOP_REPORT.md)已完成：
> 全42点＋17点ABBA48、stride/store单变量与sum归因、七K fresh根UI；默认现为1，不用旧ATT解释新循环。

## 四条不可违反的口径

### 1. Successful issue不是first attempt

gfx9 ATT动态记录为：

```text
[first_attempt, category, stall, duration, pc_index]
```

真实成功发射时刻是：

$$
t_{issue}=t_{first\_attempt}+stall
$$

MFMA execution window从`t_issue`开始，gfx942本项目按16 cycles建模。若把`first_attempt`当issue，会把阻塞期误画成执行期，夸大跨wave overlap并高估MFMA busy。

### 2. 指令映射必须使用`code[pc_index]`

`pc_index`已经是`code.json`中该动态事件的索引。使用`code[pc_index - 1]`会把事件整体错配到前一条ISA，进而颠倒load、wait、MFMA和store归因。每次更换rocprof版本后都应抽样检查动态事件的opcode与源码行。

### 3. 单wave stall不能直接相加

同一physical SIMD上的resident waves共享MFMA执行资源。一条wave的`stall:MFMA`可能被peer的16-cycle MFMA execution完全覆盖；这种stall对physical墙钟没有直接缺口。

必须按以下key合并resident waves：

```text
(shader_engine, cu, simd)
```

并计算MFMA execution window并集：

$$
U_{MFMA}=\frac{|\bigcup_i W_i^{MFMA}|}{T_{steady}}
$$

只有并集之外的physical idle才需要归因。slot数必须从trace动态识别；本项目已有2、3、4 resident-wave案例，不能硬编码为2。

### 4. Physical SIMD账本不覆盖跨CU dispatch-tail

ATT解释一个被采样physical SIMD内部的resident-wave overlap，但整卡墙钟还受workgroup调度粒度和各CU任务尾部影响。目标CU可能恰好落在轻载组，因此“采样CU的MFMA union更高”不能推出整个dispatch更快。

对80 CU和`T`个逻辑M64任务，当前4-wave physical与8-wave paired路径的critical waves/SIMD分别是：

$$
W_{physical}=\left\lceil\frac{T}{80}\right\rceil,\qquad
W_{paired}=2\left\lceil\frac{T}{160}\right\rceil
$$

当`T mod 160`落在`1..80`时，paired会多支付一个wave batch。实测`T=1024`时paired采样CU的steady MFMA union为90.73%，高于physical的86.29%，但整卡combined ratio仍为`1.01946`；将任务平衡为`T=1280`后，paired combined ratio转为`0.97949`。因此每次还必须报告全SE/CU的captured-wave或WG分布、critical-wave公式和整卡ABBA。Physical owner账本解释局部气泡，不能替代dispatch-tail模型。

## 七层分层报告

报告必须按下面七层自外向内展开。每层有自己的分母，不能把不同层的百分比相加。
前两层回答“工作是否均匀地送到硬件”，第3--5层回答“一批resident waves的生命周期
花在哪里”，第6层才回答“稳态MFMA空槽由什么占用”。

| 层级 | 分析对象 | 分母 | 核心输出 | 何时继续下钻 |
| ---: | --- | --- | --- | --- |
| 1 | 全设备CU | 静态active tile/WG分配容量 | `Z_CU`、`I_CU`、tasks/CU | `I_CU`显著时先修grid/mapping |
| 2 | 每CU内SIMD | 静态wave或resident-batch分配容量 | `Z_SIMD`、`I_SIMD`、waves/SIMD | 不均衡显著时先修wave映射 |
| 3 | 单SIMD prologue | active-batch lifecycle | cycles/batch、占比、分位数 | 首MFMA过晚时分析初始化关键路径 |
| 4 | 单SIMD epilogue | active-batch lifecycle | cycles/batch、占比、分位数 | 末MFMA后尾巴长时分析store/drain |
| 5 | 单SIMD steady | active-batch lifecycle | steady占比、内部N窗口覆盖率 | 只有steady足够长才进入第6层 |
| 6 | steady MFMA union | 选定的内部steady窗口 | busy/idle及七类互斥stall | 按最大可控类别选择一个实验 |
| 7 | 决策闭环 | 同shape的control/candidate | 原因转移、ABBA24、结论 | 三者同向才保留候选 |

第1--2层是由任务数、tile划分和硬件资源静态求出的容量口径，不依赖ATT时间戳；
第3--5层是生命周期口径，第6层是内部steady窗口口径。
只有第3--5层内部、以及第6层的`MFMA busy + 七类stall`各自要求加和闭合。

### 1. CU任务不均衡

该层必须在读取ATT前静态完成。先由shape和tile计算逻辑任务数，再由launch grid计算
启动的workgroup数。例如MoE down中：

$$
T_M=\left\lceil\frac{valid\_rows}{BM}\right\rceil,
\qquad
T_N=\left\lceil\frac{N}{N_{per\ WG}}\right\rceil,
\qquad
T=T_M T_N.
$$

若kernel在WG内部遍历完整N，则$T_N=1$。launch出来但被uniform early-exit的WG不计入
$T$，应静态报告$G_{launch}-T$及其占launch grid的比例。

根据kernel的确定性task-to-CU映射，直接计算每个CU的$n_c$。均匀商余分配时：

$$
q=\left\lfloor\frac{T}{C}\right\rfloor,
\qquad r=T\bmod C,
$$

即$r$个CU各$q+1$个任务，其余$C-r$个CU各$q$个任务。若代码有XCC/SE/CU重排，按
源码映射逐个枚举task，而不是假设硬件自然分配。

令$T$为active workgroup数，$C$为物理CU数，$n_c$为CU $c$承担的active workgroup数：

$$
n_{max}=\max_c n_c,
\qquad
I_{CU}=1-\frac{\sum_c n_c}{C\,n_{max}}.
$$

同时单列完全没有active workgroup的CU比例：

$$
Z_{CU}=\frac{|\{c:n_c=0\}|}{C}.
$$

`I_CU`是相对critical CU的静态容量损失；均匀商余分配可直接写成：

$$
I_{CU}=1-\frac{T}{C\lceil T/C\rceil}.
$$

若任务等成本，对应的理想到实际critical-path膨胀为$n_{max}/(T/C)-1$。必须报告：

- launch WG、active WG、uniform early-exit WG；
- CU数、零任务CU数；
- 静态`tasks/CU`直方图、$Z_{CU}$和$I_{CU}$；
- 使用的task-to-CU映射公式或枚举程序。

ATT在本层只做sanity check，例如确认被采样CU的wave数量与静态预测一致；不能用只采样
CU1的ATT反推全部CU分布。任务成本不等时，静态计数仍必须报告，但它只是容量模型；
另用全设备timeline报告每CU工作时长偏差，不能替换或混入$I_{CU}$。

### 2. CU内SIMD wave不均衡

该层同样必须静态完成。令每个WG包含$W$个wave，每个CU有$S$个SIMD；由kernel的
wave-to-SIMD规则计算一个WG对各SIMD的贡献$a_s$。对CU $c$：

$$
w_{c,s}=n_c a_s.
$$

若$W$能整除$S$且wave均匀映射，则$a_s=W/S$；例如8-wave WG和4个SIMD时，每个WG
静态贡献2 waves/SIMD。若不能整除或起始SIMD会旋转，必须按tile和映射规则枚举每个WG，
不能用ATT反推。wave-count不均衡为：

$$
I_{SIMD}=1-\frac{\sum_{c,s}w_{c,s}}
{\sum_c S\max_s w_{c,s}}.
$$

同时报告零active-wave SIMD比例：

$$
Z_{SIMD}=\frac{|\{(c,s):w_{c,s}=0\}|}{C\,S}.
$$

该式包含零wave SIMD，并按每个CU自己的critical SIMD归一。还要单列resident batch之间
的静态不均衡。令$R$为资源决定的resident waves/SIMD，静态batch数为：

$$
b_{c,s}=\left\lceil\frac{w_{c,s}}{R}\right\rceil,
\qquad
I_{SIMD,batch}=1-\frac{\sum_{c,s}b_{c,s}}
{\sum_c S\max_s b_{c,s}}.
$$

报告静态`waves/SIMD`和`resident batches/SIMD`直方图、零wave SIMD数、
$I_{SIMD}$与$I_{SIMD,batch}$。实际wave duration不同不会改变静态分配；若需要观察
相邻resident batch之间的动态供给空洞，将其作为第3--5层的补充指标$I_{gap}$报告：

$$
I_{gap}=\frac{\sum \text{inter-batch gap cycles}}
{\sum(\text{batch lifetime}+\text{inter-batch gap})}.
$$

ATT在第2层也只用于核对采样CU是否符合静态`waves/SIMD`结果，不能作为主要计算来源。

### 3. 单SIMD prologue

先在每个physical SIMD上按resident slot和生命周期重建wave batch。对batch $b$定义：

$$
t_0=\min_i t_{begin,i},\qquad
t_1=\min_i t_{first\ MFMA,i}.
$$

prologue为$P_b=t_1-t_0$。报告$\sum P_b/\sum L_b$、cycles/batch及
min/p50/p95/max。这里是“physical SIMD首次有MFMA前”的时间，不是把各wave prologue
相加。

### 4. 单SIMD epilogue

令$t_2=\max_i(t_{last\ MFMA\ issue,i}+16)$，$t_3=\max_i t_{end,i}$，则：

$$
E_b=t_3-t_2.
$$

同样报告$\sum E_b/\sum L_b$及分布。它只包含最后一条MFMA结束后的物理尾部；当前N
内部的CShuffle/store空洞属于steady中的`other/structural tail`，不能重复计入epilogue。

### 5. 单SIMD steady

resident batch的物理生命周期和steady span定义为：

$$
L_b=t_3-t_0,
\qquad
S_b=t_2-t_1,
\qquad
P_b+S_b+E_b=L_b.
$$

报告$P/S/E$三项占比，必须精确闭合到100%。随后为第6层选取内部稳定N窗口；通常去掉
首尾N块，但必须由timeline确认，而不是固定照抄`N2..N13`。同时报告该窗口覆盖完整
steady span的比例。

### 6. Steady physical MFMA-union与stall构成

ATT以4 cycles为最小时间粒度。先在每条MFMA的successful issue后标记16-cycle执行窗，
再合并同一SIMD所有resident wave：

$$
B(t)=\bigvee_i MFMA_i(t).
$$

`B(t)=1`记MFMA busy；只有`B(t)=0`才分类。为避免任意16-cycle对齐切断MFMA窗口，先按
4-cycle tick精确累计，最后除以16，报告为“16-cycle等效槽”；因此汇总值允许是小数。

单条ATT记录中的`stall`只是该wave从attempt到successful issue的原始等待，不能直接
当成physical stall。指令$r$对physical账本的贡献必须先与steady union idle求交：

$$
E_r=\left|[t_{attempt,r},t_{issue,r})\cap T_{steady}\cap
\overline{\bigcup_i W_i^{MFMA}}\right|.
$$

例如某条`ds_read2st64_b64`为`attempt=98540, issue=98580`，原始stall是40 cycles；
但peer在98548开始执行MFMA，且98540--98544也已被前一条MFMA覆盖。真正未被MFMA union
覆盖的只有`[98544,98548)`，因此该实例对physical `LDS issue`只贡献4 cycles，而不是
40 cycles。热点PC必须累计这种交集后的贡献，禁止按原始`record.stall`排序后直接解释
为物理损失。

当一个idle tick上多个wave有不同状态时，**整段4 cycles只归给以下最高优先级类别**，
不再在多个owner间等分：

| 优先级 | 主类别 | 包含内容 | 必须附带的子分解 |
| ---: | --- | --- | --- |
| 1 | VMEM issue | VMEM正常发射或发射前stall | service / issue-stall；load / store |
| 2 | VMEM wait | `s_waitcnt vmcnt(...)`；mixed wait也在此归类 | wait阈值、PC、producer距离 |
| 3 | LDS issue | DS/LDS正常发射或发射前stall | service / issue-stall；read / write |
| 4 | LDS wait | `s_waitcnt lgkmcnt(...)` | wait阈值、PC、producer距离 |
| 5 | VALU execution | 成功发射的VALU/TRANS | opcode、PC、所在phase |
| 6 | barrier | `s_barrier`发射或等待 | barrier代次、两wave phase |
| 7 | other | 以下所有剩余状态 | 必须继续展开，不能只报other |

主类别选定后，只在命中该主类别的wave之间等分4 cycles，用于生成该类别内部的
service/stall、load/store、opcode、PC和phase子表；未命中主类别的wave不参与子表。
因此每个子表之和必须等于对应主类别，而七个主类别之和必须等于union idle。

实现时可直接使用以下伪代码，避免把wave级stall求和：

```text
for each (SE, CU, SIMD):
  discover resident slots from trace
  reconstruct resident-wave batches
  for each 4-cycle tick in the selected internal steady window:
    if any resident wave has a 16-cycle MFMA execution window at tick:
      mfma_busy += 4
      continue
    categories = classify_each_active_wave(tick)
    owner = first_present(categories, PRIORITY_ORDER)
    stall[owner] += 4
    split 4 cycles among waves matching owner for opcode/PC/phase details
```

`other`至少拆成：structural tail、VALU dependency、SALU/control、MFMA unavailable、
scheduler ready、SMEM/其他service和无法解释的residual。每个主类别报告：

```text
cycles
16-cycle equivalent slots = cycles / 16
share of steady stall = cycles / union_idle_cycles
share of steady = cycles / steady_cycles
```

主表必须满足：

$$
U_{MFMA}+E_{VMEM\ issue}+E_{VMEM\ wait}+E_{LDS\ issue}
+E_{LDS\ wait}+E_{VALU}+E_{barrier}+E_{other}=100\%.
$$

每次报告必须执行四个断言：

```text
prologue + steady + epilogue == active-batch lifecycle
MFMA busy + MFMA idle == selected steady window
sum(seven exclusive categories) == MFMA idle
sum(subcategories of category r) == category r
```

第6层中的“stall”泛指没有MFMA执行的slot。VMEM/LDS正常issue和VALU execution是必要
服务成本，不是硬件阻塞，因此必须在子分解中与真正的issue stall分开。

### 7. 一页报告与快速判瓶颈

固定按以下顺序输出，不要先展示几十个opcode：

```text
1. CU: active/zero CU，tasks/CU分布，I_CU
2. SIMD: waves/SIMD、batches/SIMD分布，Z_SIMD、I_SIMD、I_SIMD,batch
3. prologue: cycles/batch，占active-batch lifecycle；补充动态inter-batch gap
4. epilogue: cycles/batch，占active-batch lifecycle
5. steady: cycles/batch，占active-batch lifecycle，内部N窗口覆盖率
6. MFMA union: busy/idle cycles、百分比、16-cycle等效槽
7. steady stall: 七类互斥主表；最大两项各给top opcode/PC/phase
8. witness: top joint state与all-waves-same，仅作定位
9. control -> candidate原因转移表
10. clean ABBA24与结论
```

快速决策顺序：

| 最大损失层/类别 | 首先检查 | 首选单变量实验 |
| --- | --- | --- |
| CU不均衡 | active WG数、mapping、zero-task CU | 改tile/grid/task mapping |
| SIMD不均衡 | waves/SIMD、batches/SIMD、resident容量 | 改waves/WG或SIMD映射 |
| inter-batch gap | 相邻batch的end到begin | 检查调度供给或slot replacement |
| prologue | 首个MFMA前的A/B/metadata关键路径 | 提前首批load、合并初始化wait |
| epilogue | 最后MFMA后的CShuffle/store | 分片退休、提前最后store |
| VMEM issue | service与issue-stall、load/store同相 | 错相请求或role/slot priority |
| VMEM wait | load到consumer距离 | 增加预取深度 |
| LDS issue | read/write同相、bank PMC | 拆分DS burst或调整地址/相位 |
| LDS wait | read到consumer距离 | 前移read并插入独立工作 |
| VALU execution | top VALU PC和peer phase | 将短生命周期VALU移入peer MFMA窗 |
| barrier | 两边到达phase、生产消费关系 | 先平衡前置工作；确认安全前不删barrier |
| structural tail | 最后ready MFMA位置 | 跨N overlap或分片退休 |
| VALU dependency | producer/consumer和VGPR生命周期 | 改依赖链，而非仅移动整组VALU |

选择候选时只处理占steady总周期最大的可控项。若该项下降但MFMA union或墙钟没有改善，
说明气泡转移，不算成功。

## 当前权威案例：8x1 K256

本节是当前仓库唯一的MoE 8x1 K256 ATT结果入口。历史候选、旧owner账本和旧脚本结果
不再作为依据。

### 输入身份

- raw trace：`ui_output_moe_8x1_k256_current_dispatch_16/`
- kernel：`moe_2stage_down_prefill_8x1_0`，GPU7，dispatch 16
- 采样：CU1、4个SE、每SE 4个SIMD；408条active wave、128条uniform early-exit wave
- shape：B32768、TOPK8、E256、N2048、K256、BM256、BN128、8 waves/WG
- geometry：16个N block，每N 4个core，每core 32条MFMA，每wave共2048条MFMA
- 源码SHA256：`85a13a748104baae3b9fd73f3936ca611a1e978e4d0e4ece1fe11a7e86de4d9b`
- ISA SHA256：`739617672892eb8035e997d493298e5617c0e1ddfa5f4350011f7aee2777c7df`
- `code.json` SHA256：`e87e0f766ac594dbcbfba27f49f6eb0155fa2232fdded806a56695d84c09560d`

单次ATT dispatch为0.717243ms，资源为60 regular + 132 accum VGPR、112 trace SGPR、
48KiB LDS、0 scratch。该单次时延只标识trace，不替代clean ABBA墙钟结果。

### 唯一复算命令

分析器与本文同目录：[analyze_mfma_stall.py](analyze_mfma_stall.py)。从仓库根目录运行：

```bash
python tests/flydsl/attn_4wave/tools/analyze_mfma_stall.py \
  ui_output_moe_8x1_k256_current_dispatch_16 \
  --n-blocks 16 --cores-per-n 4 --mfma-per-core 32 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 \
  --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 1 \
  --record-attempt 98540 --record-pc-index 1213 \
  --json ui_output_moe_8x1_k256_current_dispatch_16/mfma_stall_report.json
```

脚本只依赖Python标准库和项目已有的NumPy，不依赖`/tmp`文件。它在一个进程中完成：

1. 解析`code.json`和所有wave JSON，校验每条active wave的MFMA数。
2. 静态计算第1--2层，按resident slot重建第3--5层生命周期。
3. 按successful issue和16-cycle MFMA窗计算第6层physical union。
4. 按固定优先级生成七类互斥账本及PC/opcode/phase子表。
5. 可选重算一个32-MFMA stage及一条动态记录的physical贡献。

权威机器可读输出为`mfma_stall_report.json`；本文是唯一人读报告。重新运行后必须得到
下面的关键数值，否则先停止解释并核对trace、geometry和源码hash。

### 七层结果

第1层：launch 1280 WG，其中1024个active、256个uniform early-exit。80个CU中64个各
13个active WG、16个各12个，无零任务CU：

$$
I_{CU}=1-\frac{1024}{80\times13}=1.538\%,
\qquad I_{critical}=1.5625\%.
$$

第2层：每WG的8个wave均匀落到4个SIMD。256个SIMD各26 waves，64个SIMD各24 waves，
对应13或12个resident batch；$I_{SIMD}=I_{SIMD,batch}=Z_{SIMD}=0$。

204个采样resident batch的生命周期为：

| 阶段 | cycles/batch | p50 | p95 | lifecycle占比 |
| --- | ---: | ---: | ---: | ---: |
| prologue | 16,237.35 | 14,082 | 26,475.4 | 16.690% |
| steady | 78,885.82 | 76,628 | 98,764 | 81.083% |
| epilogue | 2,166.55 | 2,152 | 2,375.4 | 2.227% |

三项闭合为19,847,104 cycles。inter-batch gap共99,452 cycles，平均529 cycles，占观察
horizon的0.499%。N2--N13内部窗口为11,909,224 cycles，覆盖完整steady的74.004%。

| 类别 | cycles | 16-cycle等效槽 | idle占比 | steady占比 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 9,922,560 | 620,160.00 | - | 83.318% |
| VMEM issue | 189,156 | 11,822.25 | 9.521% | 1.588% |
| VMEM wait | 354,564 | 22,160.25 | 17.847% | 2.977% |
| LDS issue | 696,084 | 43,505.25 | 35.038% | 5.845% |
| LDS wait | 98,420 | 6,151.25 | 4.954% | 0.826% |
| VALU execution | 276,624 | 17,289.00 | 13.924% | 2.323% |
| barrier | 289,212 | 18,075.75 | 14.558% | 2.428% |
| other | 82,604 | 5,162.75 | 4.158% | 0.694% |

MFMA idle为1,986,664 cycles，七个类别精确闭合。LDS issue进一步拆成352,436 cycles
issue-stall和343,648 cycles正常服务；VMEM issue拆成104,616 cycles issue-stall和
84,540 cycles正常服务。`other`由60,430 SALU/control、20,120 structural tail、
1,250 scheduler ready和804 MFMA unavailable cycles组成。

### 具体32-MFMA stage

选取`se0_sm0_sl0_wv0.json`中的N2/core1，即slot0的MFMA ordinal 288--319：

- physical位置：SE0/CU1/SIMD0/slot0；partner为slot1。
- 动态窗口：`[99052,99648)`，共596 cycles。
- 首条MFMA：PC index 1239，ISA第1632行，源码`gemm2_8x1.py:1024`。
- 末条MFMA：PC index 1316，ISA第1757行，源码`gemm2_8x1.py:1024`。
- 32条MFMA的单wave raw stall之和为196 cycles；它不是physical idle。

| physical union状态 | cycles | 16-cycle等效槽 | stage占比 |
| --- | ---: | ---: | ---: |
| MFMA busy | 512 | 32.00 | 85.906% |
| VMEM issue | 8 | 0.50 | 1.342% |
| VMEM wait | 4 | 0.25 | 0.671% |
| LDS issue | 20 | 1.25 | 3.356% |
| LDS wait | 4 | 0.25 | 0.671% |
| VALU execution | 44 | 2.75 | 7.383% |
| barrier | 4 | 0.25 | 0.671% |

该stage的84-cycle idle由脚本逐tick唯一归类。20-cycle LDS issue全部是正常DS服务，
此stage内没有LDS issue-stall。主要位置是：

- VMEM issue：`[99324,99328)`、`[99360,99364)`；
- LDS issue：`[99364,99380)`、`[99384,99388)`；
- VMEM wait：`[99380,99384)`；LDS wait：`[99512,99516)`；
- barrier：`[99548,99552)`；其余44 cycles为零散VALU服务。

### 98540记录的交集反例

同一wave的PC index 1213为`ds_read2st64_b64 v[4:7], v49 offset1:4`，源码
`gemm2_8x1.py:895`。它在N2的core0->core1边界执行，raw stall区间为
`[98540,98580)`，共40 cycles。前一条MFMA覆盖98540--98544，peer下一条MFMA从98548
开始，因此：

$$
[98540,98580)\cap T_{steady}\cap\overline{\bigcup_iW_i^{MFMA}}
=[98544,98548).
$$

该记录对physical union idle和exclusive LDS owner都只贡献4 cycles。它不属于上面的
`[99052,99648)` stage窗口，不能把40 cycles加进stage或全局physical账本。

### 第7层决策闭环

当前K256 ISA包含4+4 B-read与prologue额外half-B两级carry。两级carry相对4+4基线的
10-buffer clean primed ABBA24为：

```text
4+4 baseline: 0.774583 ms / 354.87 useful TFLOPS
two-level carry: 0.751384 ms / 365.83 useful TFLOPS
candidate/control: 0.96702, IQR [0.96372, 0.96969], 24/24 wins
```

有效工作量为$2\times32768\times8\times2048\times256$
$=274,877,906,944$ FLOP。两级carry的K256 ISA SHA256与当前多K源码的K256 ISA相同，
均为`73961767...`；正确性、资源、墙钟和ATT方向均已闭合。当前trace的83.318% union
低于同ISA另一轮采样的85.241%，说明ATT采样存在波动；它不推翻24轮墙钟结论。

### Witness不是可加速预算

`joint state`和`all_waves_same_reason`只用于定位同时发生的状态，不能再加到主账本。
两者必须在上述七类映射之后统计；若需要展示原始`stall:DS-read`等细粒度状态，只能作为
对应主类别的子表，不能与七类主表混算。
Oracle recoverable假设目标blocker可立即替换为ready MFMA，只是上界见证，不是性能
预测。每次候选必须给出control到candidate的原因转移，确认不是把气泡搬到另一类别。

## K512的vmcnt(5)正确性核验（2026-09-06）

这里指8x1 **PTPC＋rolling**跨N内存stage中的`vmcnt(5)`，不是LDS的`lgkmcnt(5)`，
也不是K=512时所有等待都固定为5。compact的满块复用同一逻辑。
计算位置见[VMEM额度与B提交](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1386-L1424)。

**5 = 2条本stage weight-scale load + 2条上一N输出store + 1条已发出的next-pending B load。**
两个scale在当前ISA为`global_load_dwordx4`，两个输出为`buffer_store_dwordx4 ... nt`，
后一份B为`buffer_load_dwordx4`，每条指令各记一个VMEM事件，不按4个dword或64个lane计数。
`frag_weight_quarters`来自LDS，仅影响lgkmcnt；wait之后才发出的future-pending也不在这5条里。

gfx942普通buffer/global的load **和store**均计入vmcnt；不能套用gfx10+把store归入独立vscnt的规则。
[LLVM gfx942内存模型](https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942)规定这些普通VMEM
对同一wave按发射顺序报告完成；泛型FLAT访问LDS可能不满足此顺序，本kernel的该区间没有FLAT。
`s_waitcnt vmcnt(5)`的含义是**等到未完成数≤5**，不是“等待5条完成”，也不保证最后5条已完成。
当前要写LDS的B load位于最后5条之前，所以它及更早请求均已完成；下一份B/scale/output仍可在途。

每个N128的八core是`K0L,K1L,K2L,K3L,K0H,K1H,K2H,K3H`。
用`P(q)`表示第q个half-core内存stage要提交的B请求；它一般在q−2发出，两份staging轮换。
中间N的前四stage可简写为：

```text
// 较早stage中已经发出
staging[q & 1]       = VMEM_LOAD(P(q))       // 本stage必须完成的B
staging[(q+1) & 1]   = VMEM_LOAD(P(q+1))     // 可以继续在途

// 当前stage；scale属于当前N，输出属于上一N
scale[q % 8]        = VMEM_LOAD × 2
old_output          = LDS_CSHUFFLE_AND_READ()
WAIT_LDS_FOR(old_output)                    // lgkmcnt，不计入下面5
VMEM_STORE(old_output) × 2
current_B_registers  = LDS_READ(current_B)   // 不计入vmcnt

WAIT_VMCNT(2 + 2 + 1)                       // 只保证较早的P(q)等已完成
LDS_WRITE(next_B_slot, staging[q & 1])
staging[q & 1] = VMEM_LOAD(P(q+2))           // 消费旧staging后才复用
STAGE_BARRIER_AND_LDS_DEPENDENCY_WAITS()
MFMA(current_B_registers, resident_A) × 32
```

不是所有位置恰好只有5条比P(q)更新的请求：还可能有前一stage的scale/store。
实编译N2048中58处等待的producer之后分别有5条（15处）或9条（43处）VMEM。
因此5是允许保留的**最年轻尾部额度**；9条情形还会顺带等待前一stage的4条请求，仍正确。

**后续更正：这里证明了5安全，不代表5是最宽松的安全阈值。** 用户指出应把上一stage的4条
scale/store一并计入producer年龄；选择性5→9已在下节完成实际ISA与三版本数值验证。

### 首尾边界

| 情形 | 前四stage中的等待额度 |
| --- | --- |
| 首N且仍有后续N | 无上一N输出，常规stage1–3为2 scale＋1 B＝3；stage0由prologue单独处理 |
| 中间N，PTPC rolling | 2 scale＋2 store＋1 B＝5 |
| 中间N，per-tensor权重 | 无逐N scale，2 store＋1 B＝3 |
| 最后N，PTPC rolling | stage0/1为5；stage2无next-pending降4；stage3无pending，不发这次wait/commit |
| 后四stage | 不发该scale/rolling store，一般仅保留1份后继B；drain另有vmcnt(0) |

所以N2048（16个N tile）实际有$14\times4+2=58$处`vmcnt(5)`；N4096为122处，单N128没有。
若将所有位置无条件硬编码成5，或删掉scale/store后仍保留5，原依赖证明就不再成立。

### 已执行的验证

- [静态审计](../../../contrib/moe/results/k512_vmcnt5/isa_audit_v2.json)按寄存器定义追踪普通8x1、compact满块：
  每份N2048 ISA均128条Bload→LDS-write依赖全部被wait覆盖，58处vmcnt(5)的最后五条均为上述1＋2＋2。
- [专项脚本](../../../contrib/moe/check_k512_vmcnt.py)未修改生产kernel，编译当前版本与仅将VM字段5改为0的保守版本。
  四份实际ISA逐指令对照证明：除了`vmcnt(5)→vmcnt(0)`外完全一致，寄存器/数学指令不变。
- [数值结果](../../../contrib/moe/results/k512_vmcnt5/gpu_check/result.json)：K512、E8/TopK4、随机routing/随机FP8权重，
  N/B/padding为128/1/0、256/257/32、2048/769/64、4096/769/128，各版本各重复10次，
  全部finite、padding/inactive区域NaN未写，最终两版本输出逐bit相同。
  FP32参考rel_l2分别0.00357072、0.00330640、0.00330669、0.00330878，均<0.005。
- 这确认的是当前gfx942指令顺序和等待额度的正确性，不宣称形式化覆盖所有未来编译器/ISA。
  改动load/store数量、指令合并或移植其它架构后必须重算并查实际ISA；本轮没有性能采样或kernel修改。

### 用户提出4＋5＝9：跨stage年龄验证（2026-09-06）

**结论：对跨N rolling区间的stage1–3成立，原vmcnt(5)多等待了前一stage的4条VMEM；
stage0不成立，不能把全部5无条件替换成9。**

令`P(q)`为q拍要提交的B，它在q−2拍末发出，且在q−1拍不能覆盖或消费错误的staging：

```text
stage q-2: scale/store；wait(P(q-2))；commit(P(q-2))；issue P(q)
stage q-1: scale×2；store×2；wait(P(q-1))；commit(P(q-1))；issue P(q+1)
stage q  : scale×2；store×2；wait vmcnt(9)；commit(P(q))；issue P(q+2)

P(q)之后、当前wait之前：
    [上一拍 scale×2, store×2, P(q+1)×1] + [本拍 scale×2, store×2]
    = 5 + 4 = 9
```

对中间N这段，q−1拍本身的wait不计数，也不会生成新的VMEM；它至多使部分请求提前完成，
不破坏按发射顺序确定的9条后继额度。若P(q)仍未完成，则完成序前缀不能越过它，
P(q)及其9条后继会使计数至少为10，因此`vmcnt(9)`足以保证P(q)完成。
旧`vmcnt(5)`则还会要求完成其中最早的4条——即上一拍的scale×2/store×2。

物理上[weight_staging](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L354-L365)
是两份VGPR staging，`q&1`轮换；“q−2发出、q提交”的三拍跨度，不应误写成三个独立的
在途B global-load槽。当前计算使用的B已经从LDS读入寄存器，和这两份carry不是同一层。
9的依据是**目标producer之后的VMEM序列**，不是仅凭FIFO槽数推算。

| 当前位置（PTPC rolling，n>0） | 上一拍scale/store | 后继B | 当前scale/store | 仅针对本次B提交的年龄额度 | 本轮实验 |
| --- | ---: | ---: | ---: | ---: | --- |
| stage0，前一拍为stage7 | 0 | 1 | 4 | 5 | 保留5 |
| stage1–3，且has_next_pending | 4 | 1 | 4 | 9 | 5→9 |
| 最后N stage2，无has_next_pending | 4 | 0 | 4 | 8 | 保留原4，未放宽到8 |
| 最后N stage3，无has_pending | — | — | 4 | 无该commit | 不加wait |

此表仅解释B提交依赖；其它值的首次使用仍需实际ISA里的相应等待，不能把所有vmcnt一并放宽。
尤其N边界只存在5条后继，将stage0改9会失去对目标B完成的保证，CPU反例已覆盖，未运行不安全GPU版本。

**实测证据：** [选择性9三版本结果](../../../contrib/moe/results/k512_vmcnt9/gpu_check/result.json)。
保留生产源码不动，仅在测试进程编译时按展开位置调整VM字段，比较原版、5→0、选择性5→9。

| N / B / paddingB | 原vmcnt(5)数 | 实验保留5 / 改9 | FP32参考rel_l2 |
| --- | ---: | --- | ---: |
| 128 / 1 / 0 | 0 | 0 / 0 | 0.00357072 |
| 256 / 257 / 32 | 2 | 1 / 1 | 0.00330640 |
| 2048 / 769 / 64 | 58 | 15 / 43 | 0.00330669 |
| 4096 / 769 / 128 | 122 | 31 / 91 | 0.00330878 |

- K512、E8、TopK4、PTPC、随机routing及随机FP8输入/权重；四shape×三版本×20次重复，共240次。
  0/9版每次都与原版输出逐bit相同，所有版本rel_l2<0.005、finite、padding/inactive NaN合同通过。
- 实际9版ISA逐指令等于原版仅替换上述位置的`vmcnt(5)→vmcnt(9)`，寄存器/操作数/数学指令不变，
  无编译器暗中追加更紧wait。新ISA全部Bload→LDS-write依赖通过年龄审计。
- [CPU单测](../../../contrib/moe/test_check_k512_vmcnt.py)6项通过，包含9条后继的正例、N边界不足9条的反例。
- 因此原“5正确”的数值判断不变，但它不是这些稳态位置的最大安全额度；用户的4＋5计数成立。
  本轮只验证，没有替换生产kernel、采集新ATT或宣称性能提升。

<a id="k512-vmcnt9-production"></a>

### 生产源码选择性5→9：正确性通过，性能没有一致获益（2026-09-06）

本节是上一节实验之后、用户要求“修改kernel，测试性能”的独立验收。
当前[生产等待分支](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1405-L1416)
只在K512、PTPC、rolling、`block_n>0`、stage1–3且`has_next_pending`时增加4条年龄额度。
stage0仍5；最后N stage2仍4、stage3无该commit；pure、其它K和per-tensor不改。
普通8x1及compact满块复用该分支；构表、tail、cache policy、padding、sorted_sum和selector均不改。

- 修改前generic源码SHA256：`f9f8daea1e7ac09b06f7894ee0b9a199dac063d81fb44e00a736f9f834d60081`。
- 当前generic源码SHA256：`e56c70400345a4cb24a75ef153786eff4ae25b55ec6b44cbc6c171cc20a96410`。
- [机器可读汇总](../../../contrib/moe/results/k512_vmcnt9_production/summary.json)记录各轮源码/ISA哈希、
  资源、原始样本复算、硬件状态及非目标回归；[专用汇总脚本](../../../contrib/moe/summarize_k512_vmcnt9.py)
  不修改任何输入JSON，不将历史5版结果重标为当前9版。

#### 正确性、基线身份与资源

[生产三版本专项](../../../contrib/moe/results/k512_vmcnt9_production/correctness/result.json)：
四shape×旧5/保守0/原生9×20次，共**240次**，全部输出逐bit一致，随机FP32参考
`rel_l2=0.003307–0.003571<0.005`，finite及padding/inactive NaN合同通过。
覆盖N128/256/2048/4096；原生9实际ISA等于上一节实验9，重建旧5实际ISA等于修改前原版，
不是仅凭源码推测对照相同。

- **120项回归通过，929.66秒**：compact down/完整链100项、普通8x1七K×两量化14项、等待账本CPU6项。
  本轮未重复声称覆盖此前全部175项；测试入口为[compact测试](../../../contrib/moe/test_compact_m64_down.py)、
  [普通8x1测试](../../../contrib/moe/test_moe.py)、[等待账本测试](../../../contrib/moe/test_check_k512_vmcnt.py)。
- 性能harness另增加[4项ISA分组正反例](../../../contrib/moe/test_benchmark_k512_vmcnt9.py)，全部通过。
  首次冒烟错误地要求整份compact汇编没有9，误计了**1x4尾kernel原有的5处编译器`vmcnt(9)`**；
  已修为仅对full8x1计数/替换，tasks/tail逐指令原样比较。
  [失败冒烟](../../../contrib/moe/results/k512_vmcnt9_production/qwen35_smoke.json)无性能样本，
  [修复后冒烟](../../../contrib/moe/results/k512_vmcnt9_production/qwen35_smoke_v2.json)通过，不作正式收益依据。
- 最终等待/性能统计/任务几何/ISA解析的**22项CPU回归通过，0.13秒**；其中6项与上面的120项重叠，
  不简单相加。全部结果及哈希独立复算一致，新文档链接和48组时延/吞吐值自动核对通过。

| 生产shape | 普通 / compact full指令数 | full保留5 / 改9 | VGPR / next-free SGPR | LDS / private | MFMA/wave |
| --- | ---: | ---: | --- | --- | ---: |
| N2048/E256/TopK8 | 10877 / 10881 | 15 / 43 | 218 / 96 | 48KiB / 0 | 4096 |
| N4096/E512/TopK10 | 21451 / 21455 | 31 / 91 | 220 / 96 | 48KiB / 0 | 8192 |

五个资源字段、MFMA数量和寄存器操作数全部不变，只有full内预定43/91处5→9。
compact构表、full、tail逐kernel均0 private、0 VGPR spill、0 SGPR spill；tail原有5处9不变。
四份same-shape普通/compact的旧5重建均与修改前真实ISA逐指令相等，不与TopK4的专项dump混比。

另fresh编译并执行以下**5条非目标路径**，与各自旧ISA的指令及五个资源字段完全相同：

| 路径（均N2048/E256/TopK8） | 指令数 | VGPR | LDS / private |
| --- | ---: | ---: | --- |
| K256 PTPC rolling | 7437 | 186 | 48KiB / 0 |
| K384 PTPC rolling | 9167 | 206 | 48KiB / 0 |
| K640 PTPC rolling | 12647 | 238 | 48KiB / 0 |
| K512 PTPC pure | 10798 | 210 | 48KiB / 0 |
| K512 per-tensor rolling | 9738 | 206 | 48KiB / 0 |

这些是ISA/finite检查，不把它们称为新性能测量。首次非目标批次在K256完成后因全机busy6%
拒绝继续，未放宽门禁；CPU整理后重新入场门禁为0%，才执行其余四条。

#### 正式协议与硬件状态

[性能harness](../../../contrib/moe/benchmark_k512_vmcnt9.py)每次同进程测
`default/old_8x1/new_8x1/old_compact/new_compact`五个标签：每版本独立launcher，缓存关闭，
旧版仅在编译阶段把目标VM字段9还原5，计时循环没有patch操作。
两case均PTPC、BN128、padding128B：35B为N2048/E256/TopK8，397B为N4096/E512/TopK10。
Batch分别16384/32768；每case做首轮48轮和完整独立确认48轮，**四份结果不拼接、不挑最快轮**。

- 10-buffer、label顺序循环轮换，forward＋reverse平衡；每path/phase有96个绝对样本、48个配对值。
  每轮先求同版本两个样本均值，再算$g=1-\bar t_9/\bar t_5$，报告$g$中位数/IQR/胜轮。
  IQR是样本分布，不是置信区间，不能用绝对中位时延的比值代替配对统计。
- Down/Combined前统一default gateup在event外预热。Down计入compact构表＋full＋tail，
  Combined另含sum；Full包含sorting、两次量化、gateup、down、invert、sum。
  计时为GPU event包围host提交的真实路径，包含其timeline内的提交空隙；不是静态ISA周期估算。
- 所有轮次和长尾保留，event外逐轮校验共**5760条path/phase检查**，均finite、存储合同通过，
  与default的`rel_l2=0`。性能输入为round-robin routing、全1权重，不把这替代上面的随机权重正确性。
- compact在35B的16K/32K分别512/1024 full、0 tail；397B分别512 full＋512 tail、
  1024 full＋1024 tail。5→9不改变这些任务数，也不改变有效FLOP。

所有吞吐均为**墙钟有效TFLOPS**，不是padded或ATT模型吞吐：

$$
F=2B\cdot TopK\cdot N\cdot512,\qquad
T_{Down/Combined}=\frac{F}{t_{ms}\,10^9},\qquad
T_{Full}=\frac{3F}{t_{ms}\,10^9}.
$$

GPU7固定1800MHz determinism、650W、PTL **Enabled / VECTOR,F8**、NUMA off。
每次入场检查全机busy≤5%、VRAM≤20%；下表百分比为快照，不能代表全程负载监控。

| 轮次 | 全机before / managed最高busy | 目标initial / managed busy | 退出busy | 状态 |
| --- | --- | --- | ---: | --- |
| 35B首轮 | 1% / 0% | 0% / 0% | 14% | exit-load-risk，保留但不作clean定量结论 |
| 397B首轮 | 0% / 0% | 0% / 0% | 2% | 入场/退出检查通过 |
| 35B确认 | 5% / 0% | 0% / 0% | 0% | 入场/退出检查通过 |
| 397B确认 | 0% / 0% | 1% / 0% | 0% | 入场/退出检查通过 |

全部恢复auto、650W、PTL Disabled/N/A、NUMA balancing=1；风险轮和失败冒烟也完成恢复。

#### 独立确认轮：绝对中位时延与有效吞吐

每个单元格均为**旧5 ms / TFLOPS → 新9 ms / TFLOPS**。

| Case / Batch / 路径 | Down | Combined | Full |
| --- | --- | --- | --- |
| 35B / 16K / 普通8x1 | 0.656303 / 418.83 → 0.658123 / 417.67 | 0.824723 / 333.30 → 0.827863 / 332.03 | 2.403749 / 343.06 → 2.405090 / 342.87 |
| 35B / 16K / compact | 0.672123 / 408.97 → 0.671503 / 409.35 | 0.842144 / 326.40 → 0.840503 / 327.04 | 2.420350 / 340.71 → 2.418110 / 341.02 |
| 35B / 32K / 普通8x1 | 1.208705 / 454.83 → 1.211905 / 453.63 | 1.527447 / 359.92 → 1.531366 / 359.00 | 4.639639 / 355.47 → 4.643719 / 355.16 |
| 35B / 32K / compact | 1.232325 / 446.11 → 1.231525 / 446.40 | 1.552366 / 354.14 → 1.551806 / 354.27 | 4.666259 / 353.45 → 4.663419 / 353.66 |
| 397B / 16K / 普通8x1 | 2.237529 / 307.12 → 2.231829 / 307.91 | 2.690731 / 255.39 → 2.686810 / 255.77 | 7.669611 / 268.80 → 7.659671 / 269.15 |
| 397B / 16K / compact | 1.726107 / 398.12 → 1.732807 / 396.58 | 2.168028 / 316.97 → 2.176389 / 315.75 | 5.605323 / 367.79 → 5.617283 / 367.01 |
| 397B / 32K / 普通8x1 | 3.484594 / 394.42 → 3.477754 / 395.19 | 4.399098 / 312.43 → 4.398677 / 312.46 | 12.134809 / 339.78 → 12.136848 / 339.72 |
| 397B / 32K / compact | 3.153112 / 435.88 → 3.154253 / 435.73 | 4.052696 / 339.13 → 4.053597 / 339.05 | 10.907124 / 378.03 → 10.913724 / 377.80 |

对应配对统计单元格为**提升中位数 [Q1,Q3]；新9胜轮/48**，负数表示回退：

| Case / Batch / 路径 | Down | Combined | Full |
| --- | --- | --- | --- |
| 35B / 16K / 普通8x1 | -0.265% [-0.402%,-0.195%]；0/48 | -0.131% [-0.378%,+0.245%]；21/48 | -0.116% [-0.239%,+0.028%]；14/48 |
| 35B / 16K / compact | +0.080% [-0.042%,+0.243%]；34/48 | +0.167% [-0.008%,+0.500%]；35/48 | +0.022% [-0.078%,+0.148%]；30/48 |
| 35B / 32K / 普通8x1 | -0.262% [-0.334%,-0.215%]；0/48 | -0.212% [-0.348%,-0.007%]；11/48 | -0.088% [-0.225%,-0.013%]；11/48 |
| 35B / 32K / compact | +0.054% [+0.013%,+0.118%]；38/48 | +0.060% [-0.078%,+0.265%]；31/48 | +0.055% [-0.026%,+0.167%]；34/48 |
| 397B / 16K / 普通8x1 | +0.298% [+0.143%,+0.457%]；44/48 | +0.255% [+0.042%,+0.430%]；37/48 | +0.059% [-0.234%,+0.332%]；30/48 |
| 397B / 16K / compact | -0.430% [-0.591%,-0.129%]；8/48 | -0.347% [-0.655%,+0.125%]；13/48 | -0.236% [-0.587%,+0.094%]；13/48 |
| 397B / 32K / 普通8x1 | +0.238% [-0.112%,+0.388%]；34/48 | -0.132% [-0.589%,+0.305%]；21/48 | +0.060% [-0.212%,+0.199%]；26/48 |
| 397B / 32K / compact | -0.026% [-0.250%,+0.207%]；22/48 | +0.012% [-0.284%,+0.292%]；24/48 | -0.064% [-0.214%,+0.197%]；20/48 |

确认轮原始证据：[35B确认48轮](../../../contrib/moe/results/k512_vmcnt9_production/qwen35_confirm_abba48.json)、
[397B确认48轮](../../../contrib/moe/results/k512_vmcnt9_production/qwen397_confirm_abba48.json)。

#### 首轮与确认轮不一致处、结论边界

| 配置 | 首轮Down配对提升 | 确认Down配对提升 | 结论 |
| --- | ---: | ---: | --- |
| 35B普通16K / 32K | +0.273% / +0.321%（退出风险） | -0.265% / -0.262% | 方向翻转，不能宣称普通35B提速 |
| 35B compact16K / 32K | +0.161% / +0.091% | +0.080% / +0.054% | 微小改善，Full IQR仍跨0 |
| 397B普通16K / 32K | +0.280% / +0.119% | +0.298% / +0.238% | 16K Down/Combined同向；32K及两档Full未稳定改善 |
| 397B compact16K / 32K | -0.315% / +0.068% | -0.430% / -0.026% | 16K Down回退重现；32K持平 |

首轮原始证据：[35B首轮48轮](../../../contrib/moe/results/k512_vmcnt9_production/qwen35_abba48.json)、
[397B首轮48轮](../../../contrib/moe/results/k512_vmcnt9_production/qwen397_abba48.json)。
35B确认普通两档Down均0/48胜，不把它们用首轮的正收益替换；397B compact16K两次Down IQR
均为负，也不隐藏。确认轮35B/32K普通Full有小幅负IQR，但不能将一次结果扩大为全shape稳定回退。

**结论：9是已验证安全的更宽等待阈值，不等于普遍更快。** 本轮没有任何配置在两次测量中
同时证明Down/Combined/Full稳定改善。按用户要求，当前工作区保留选择性9的kernel改动，
但它**未达到本手册性能优化晋级门槛**；不修改自动selector或历史42点选择矩阵，未提交代码。
上述性能轮次结束时尚未采新ATT，因此当时没有声称VMEM physical owner下降或MFMA union提高，
也未拿旧5版UI解释新9。后续按用户要求新增的同shape fresh ATT见下一节，历史性能JSON保持不变。

<a id="k512-vmcnt9-fresh-att"></a>

### K512旧5/新9 fresh ATT与完整UI（2026-09-06）

**本轮只采普通8x1：K512、N2048、B32768、E256、TopK8、PTPC/PTPC、BM256/BN128、padding128B。**
按新9→旧5的顺序分别重新采集，两个版本都不是前面七K baseline UI；没有改kernel、selector或等待阈值。
compact和N4096不在本次ATT范围内，不能用这些结果解释397B compact的回退。

#### 输入身份、产物与硬件协议

| 版本 | 仓库根完整UI入口 | 原始UI文件 / 完整wave | 原始dispatch |
| --- | --- | --- | --- |
| 旧5 | [旧5文件清单](../../../../ui_output_moe_8x1_k512_vmcnt_old5_20260906_dispatch_22/filenames.json) | 429 / 416 | Agent26036 / dispatch22 |
| 原生9 | [新9文件清单](../../../../ui_output_moe_8x1_k512_vmcnt_new9_20260906_dispatch_22/filenames.json) | 548 / 536 | Agent10309 / dispatch22 |

完整raw UI、源码快照、code和wave文件逐文件复制并校验SHA256；根目录副本另附hardware、workload、
kernel trace、capture清单和七层analysis。code表10878行含一行kernel标题，**真实ISA均10877条指令**。
旧5的额外wrapper源码快照和两次early-exit采样数不同，因此文件数不同，不表示trace缺失。

- [采集器](../../../contrib/moe/capture_k512_vmcnt_att.py)复用现有down-only工作负载：新9原生编译，
  旧5仅在测试进程编译阶段将43处VM字段9还原5。两份源码快照的生产SHA均为
  `e56c70400345a4cb24a75ef153786eff4ae25b55ec6b44cbc6c171cc20a96410`；
  **不能只凭源码快照区分5/9**，必须同时看workload里的variant/编译改写记录和实际ISA。
- 两份debug ATT ISA分别与上节同shape已测旧/新性能ISA逐指令、五个资源字段相同；彼此只差43处5→9。
  旧版58处5、0处9；新版15处5、43处9。每wave4096 MFMA，218 VGPR、96 next-free SGPR、48KiB LDS、
  0 private/spill不变；各128条Bload→LDS-write依赖通过等待年龄审计。
- decoder的code表逐条核对寄存器和操作数；四个branch由ISA label与decoder真实PC及相对立即数核验目标，
  不直接忽略分支差异。所有wave均`num_insts==num_stitched`，每条active wave严格4096 MFMA。
- `[2]`选择第二次匹配调用；kernel trace实际两次为dispatch21/22，UI确实对应22，非先假定该编号。
  CU1、shader-engine mask0xf、SIMD mask0xf、ATT buffer0x60000000；每份记录完整采集命令。
- GPU7：1800MHz determinism、650W、PTL **Enabled / VECTOR,F8**、NUMA off。
  新9全机before/managed最高busy均0%；成功旧5为0%/1%；两次退出busy均0%，均恢复auto、
  PTL Disabled/N/A、NUMA=1。只有入场/退出快照，不能声称全程没有外部负载。
- 旧5首次尝试在设置硬件后的全机门禁遇到最高17% busy，未启动rocprof采样；
  [拒绝记录](../../../contrib/moe/results/k512_vmcnt9_production/att_old5/hardware.json)及恢复状态保留。
  CPU复算新9后重新核验门禁，使用全新retry目录成功采集，没有放宽阈值。

身份与逐文件校验见[旧5采集清单](../../../contrib/moe/results/k512_vmcnt9_production/att_old5_retry1/capture.json)、
[新9采集清单](../../../contrib/moe/results/k512_vmcnt9_production/att_new9/capture.json)。
旧/新debug ISA SHA分别为`3716bfb8dc98afb22ebec7150284d5c7d9c146d5a993ee542eb4f766742dd00a`、
`2183527d36bbba537094de4f73f157496efa2766a64b4790e766be55bae5582b`；
code SHA分别为`dea37ed24bd97137006de4a936020b43000a625f6df7954bb59efdf5fbcf11f1`、
`7cffd502f3569ab1f5d77c39fbd8bf6d0882509fc5fed8e45e81d3fce3ff46cb`。

#### 第1–2层：静态分配相同，采样数不同

两版均launch1280/active1024/early-exit256 WG，80 CU中64个各13任务、16个各12任务，
无零任务CU；$I_{CU}=1.53846\%$，critical-path膨胀1.5625%。每WG8 waves均匀落到4 SIMD；
全设备256 SIMD各26 waves、64 SIMD各24 waves，分别13/12个resident batch，
零wave比例、CU内wave/batch不均衡均0。这是静态容量计算，不从只采样CU1的trace反推全卡。

旧5采样416 active wave、0 early-exit、208 resident batch；16个physical SIMD各26 active wave。
新9采样408 active＋128 early-exit wave、204 batch；12个SIMD各26、4个各24 active wave。
按真实生命周期重叠检查，两份16个SIMD均恰有**2条最大同时active wave、slot0/1**。
early-exit不参与active生命周期或steady预算；两版总cycles不能直接相减，以下按batch归一。

#### 第3–5层：生命周期与内部窗口

| 阶段 | 旧5 cycles/batch / lifecycle占比 | 新9 cycles/batch / lifecycle占比 |
| --- | --- | --- |
| prologue | 19546.85 / 11.731% | 21682.39 / 12.743% |
| steady | 144581.33 / 86.771% | 146076.25 / 85.854% |
| epilogue | 2495.81 / 1.498% | 2386.31 / 1.403% |

旧/新active生命周期分别34,657,788 / 34,709,572 cycles，prologue＋steady＋epilogue精确闭合。
inter-batch gap平均318.90→426.19 cycles，占horizon0.176%→0.230%。
仍使用**N2–N13**相同内部窗口，覆盖完整steady的74.241%→74.279%；不缩窗口、不增MFMA。
prologue的变化也记录，不把不同采样/缓存上下文下的全部变化单因果归为等待改动。

#### 第6层：physical SIMD MFMA并集与互斥owner

使用成功issue后的16-cycle MFMA执行窗，按同physical SIMD所有resident waves取并集，
仅在并集外按4-cycle tick分配七类唯一owner。下表是cycles/batch及steady百分比，不是单wave raw stall之和。

| 类别 | 旧5 cycles/batch | 新9 cycles/batch | 旧5 steady | 新9 steady | 变化/百分点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MFMA busy | 97792.00 | 97792.00 | **91.106%** | **90.128%** | -0.978 |
| VMEM issue | 808.58 | 1078.88 | 0.753% | 0.994% | +0.241 |
| VMEM wait | 860.12 | 1252.18 | 0.801% | 1.154% | +0.353 |
| LDS issue | 3509.73 | 3508.00 | 3.270% | 3.233% | -0.037 |
| LDS wait | 372.10 | 357.22 | 0.347% | 0.329% | -0.017 |
| VALU execution | 1297.12 | 1250.53 | 1.208% | 1.153% | -0.056 |
| barrier | 1895.10 | 2456.53 | 1.766% | 2.264% | +0.498 |
| other | 803.67 | 808.02 | 0.749% | 0.745% | -0.004 |

每batch内部MFMA busy均97,792 cycles，idle由9546.40→10711.35，窗口107338.40→108503.35。
旧版原始窗口22,326,388=busy20,340,736＋idle1,985,652；新版22,134,684=19,949,568＋2,185,116。
七类与每类details/opcodes/phases/sources四个子表均精确闭合，cycles/16的等效槽见原始JSON。

最大两类仍为LDS issue/barrier。LDS issue分别包含1770.90→1751.73 cycles/batch的issue-stall，
以及1738.83→1756.27的正常服务，不把全部DS服务称为bank conflict；两版top opcode均`ds_read_b128`、
top phase均tail。barrier为`s_barrier`、主要phase为tail，不能据此删除生产消费同步。
VMEM issue的issue-stall为444.54→699.29、正常服务364.04→379.59 cycles/batch；
`other`完整拆为SALU/control677.24→675.96、scheduler ready30.82→31.38、structural tail95.62→100.68。

#### 第7层：等待暴露后移，而非消失

VMEM completion owner按实际等待阈值重新聚合，全部数值先与physical MFMA union idle求交：

| 阈值owner | 旧5 cycles/batch | 新9 cycles/batch |
| --- | ---: | ---: |
| `vmcnt(5)` | 710.25 | 81.94 |
| `vmcnt(9)` | 0 | 31.67 |
| `vmcnt(1)` | 149.87 | **1138.57** |
| 合计 | 860.12 | 1252.18 |

这两份trace中，新9前几个stage的等待确实减少，但后续仍保留的严格`vmcnt(1)`暴露更多：
VMEM-wait的`core3→4` phase由141.73→**1056.96 cycles/batch**，占新版该类84.410%。
最大joint见证由旧版`LDS issue@core0→1 + VALU execution@core0`（steady0.483%）
变为新版`VMEM wait@core3→4 + barrier@core3→4`（0.603%）。joint只用于定位，不能再加到主预算。
旧5编译wrapper使其wait源码行指向采集器；比较依据是核对过的实际指令/PC及阈值，不按源码行号字符串配对。

统一抽取的SE0/CU1/SIMD0/slot0、N2/core1的32-MFMA compute窗口两版均恰512 cycles、100% busy；
它只证明该短compute片段连续执行，**不能代替整段steady的91.106%/90.128%**，也不包含之前的memory等待。

两次单ATT dispatch旧/新为**1.208125ms / 455.05有效TFLOPS → 1.231125ms / 446.55有效TFLOPS**，
仅标识trace。工作量仍$F=549,755,813,888$ FLOP，$TFLOPS=F/(t_{ms}\times10^9)$，不是union×roof模型吞吐。
本采集是单buffer down-only，和此前10-buffer、gateup预热的Down/Combined/Full缓存上下文不同；
**每版一次fresh ATT不替代两次ABBA48，也不足以证明所有shape的因果或稳定回退**。
可确认的是当前采样中等待暴露后移、physical union没有提高，与“9安全但未证明一致提速”不矛盾。
本轮没有进一步修改等待、barrier、K或任务布局。

#### 复算与审计入口

七层实现仍唯一使用[analyze_mfma_stall.py](analyze_mfma_stall.py)，参数保持
16个N、8 core/N、32 MFMA/core、N2–N13、1280/1024 WG、80 CU、8 waves/WG、4 SIMD/CU、2 resident waves。
采集器保存rocprof命令和驱动快照，拒绝覆盖已有输出目录；旧5的wrapper语义与43处恢复记录随产物归档。

- [旧5七层账本](../../../contrib/moe/results/k512_vmcnt9_production/att_old5_retry1/analysis.json)
  与[新9七层账本](../../../contrib/moe/results/k512_vmcnt9_production/att_new9/analysis.json)。
- [完整fresh对照审计](../../../contrib/moe/results/k512_vmcnt9_production/att_comparison.json)由
  [ATT汇总脚本](../../../contrib/moe/summarize_k512_vmcnt_att.py)复核原件/副本全部hash、真实resident重叠及四类闭合。
- [decoder映射3项CPU测试](../../../contrib/moe/test_capture_k512_vmcnt_att.py)、等待分组4项、分析器14项，
  合计**21 passed，0.23秒**；分支错误目标与错误指令均有拒绝反例。

<a id="k512-stage4-vmcnt5"></a>

### K512继续放宽stage4：1→5（2026-09-06）

用户随后要求“放宽stage4”。本轮只扩展[前一拍请求年龄额度](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1405-L1416)
到K512/PTPC、rolling、`block_n>0`、stage4且`has_next_pending`；stage1–3仍9，stage0仍5，
首N、pure、per-tensor、其它K及stage5–7的等待不变。普通8x1和compact满块共同生效。

旧控制是上一轮**已经stage1–3=9、但stage4=1**的真实源码快照，SHA256为`e56c7040…`，
不是更早的全部stage1–3=5版本；当前源码SHA256为
`e55a9e331f30e92555c2f417268ca168b27a0a12eb44ba46fb73d58e23d02c98`。
所有本轮产物另存新目录；此前旧5/新9的性能、ATT和SHA保持历史身份，不能重标为当前stage4版本。

#### 为什么stage4可以加4，但stage5不行

stage4本拍无scale/store，旧局部公式仅保留一条后继B而算出1，却遗漏目标B发出之后的前拍4条请求。
令`P4`是stage4要提交到LDS的B（随后K1H使用，不是当前K0H已经读取的B）：

| 发射位置 | 请求序列 |
| --- | --- |
| stage2末 | `P4` |
| stage3 | scale×2、store×2、`P5` |
| stage4 | `wait vmcnt(5)`；commit `P4`；再发`P6` |

`P4`之后、wait之前有恰好5条年轻请求，所以5保证`P4`完成，允许stage3请求和`P5`继续在途。
旧1还会强制完成前拍4条scale/store。stage5提交的`P5`之后却只有`P6`一条，**stage5–7不能一并改5**。
末N stage4仍有`P4/P5`和stage3的4条请求，因此包含在放宽范围；单N没有任何改动。

只验证B依赖不足：stage4打包SR1使用的是更早stage1的scale；stage3加载的SR3 scale尚不消费，
后续stage5的原`vmcnt(1)`会保证它完成，末N还保留drain `vmcnt(0)`。
[新审计](../../../contrib/moe/check_k512_stage4.py)除B→LDS外，还沿实际指令检查**全部普通VMEM load结果的读取及寄存器重定义**。
当前模型限gfx942普通buffer/global、直线展开active主体，不声称覆盖任意FLAT/未知CFG。

#### 已完成的专项验证

[随机FP32参考及旧源码对照](../../../contrib/moe/results/k512_stage4_vmcnt5/correctness/result.json)：
四shape×两版本×20重复，共**160次**，新旧每次逐bit一致；finite、padding/inactive NaN合同全部通过。

| N / Batch / paddingB | 实际1→5处数 | Bload→LDS依赖数 | VMEM load寄存器依赖检查数 | 最大rel_l2 |
| --- | ---: | ---: | ---: | ---: |
| 128 / 1 / 0 | 0 | 8 | 706 | 0.00357072 |
| 256 / 257 / 32 | 1 | 16 | 1378 | 0.00330640 |
| 2048 / 769 / 64 | 15 | 128 | 10783 | 0.00330669 |
| 4096 / 769 / 128 | 31 | 256 | 21536 | 0.00330878 |

四份旧控制实际ISA与上一轮原生9一致；新ISA只在`N1..Nlast`的stage4改变1→5，所有寄存器、
其它等待、MFMA、B预取和barrier指令不变，五个资源字段相同、0 private。
[8项CPU测试](../../../contrib/moe/test_check_k512_stage4.py)包括首末N、scale提前使用/寄存器提前覆盖反例、
stage5误放宽反例、实际旧/新guard穷举及compact真实factory的旧/新全局builder替换。
compact控制使用私有模块复用同一构表/tail，只替换full builder；不改全局dispatcher，不共享新旧JIT对象。

**当前进度：160次专项及8项CPU检查已通过；35B完整对照和两case的快速Down-only已完成。**
七K/compact回归运行38分56秒时已通过78项，后续仍在FlyDSL编译；收到用户明确的性能测试请求后，
主动中止该回归以避免资源争用，保留中断日志，**不是120项全部通过**。没有数值失败记录，
但未完成部分仍属未验证，精确状态见[回归记录](../../../contrib/moe/results/k512_stage4_vmcnt5/regression_status.json)。
397B完整矩阵随后按用户“快速测试”要求在编译new_compact时中止、无完整性能样本；
普通旧/新ELF已经生成，直接复用做下节快速对照。不拿上一节stage4仍为1的ATT解释当前ISA。

#### 性能对照协议

[stage4专用harness](../../../contrib/moe/benchmark_k512_stage4.py)只对比stage4=1/5，两侧stage1–3均为9。
新旧普通8x1和compact使用独立launcher；compact仅在私有模块替换full builder，同一套构表和tail保持不变。
旧侧实际ISA必须与上轮stage4=1的生产shape ISA完全相同，新侧仅允许15/31处1→5，不能把上轮收益混入本轮。

固定PTPC、padding128B、10-buffer、48轮平衡forward＋reverse，每path/phase96个绝对样本。
35B为N2048/E256/TopK8，397B为N4096/E512/TopK10，均测B16384/32768。
Down、Combined（Down＋sum）、Full分别计时；compact全部阶段均包含GPU构表＋full＋tail，
Full另含sorting、两量化、gateup、invert、sum。每轮event外重新检查finite/数值/padding/inactive tail，
所有样本和长尾保留，不拼接独立轮次。有效工作量与统计公式为：

$$
F=2B\cdot TopK\cdot N\cdot512,\quad
TFLOPS_{Down/Combined}=\frac{F}{t_{ms}10^9},\quad
TFLOPS_{Full}=\frac{3F}{t_{ms}10^9},\quad
g_r=1-\frac{\operatorname{mean}(t_{5,r,1},t_{5,r,2})}{\operatorname{mean}(t_{1,r,1},t_{1,r,2})}.
$$

GPU7入场仍要求全机busy≤5%/VRAM≤20%，1800MHz determinism、650W、PTL Enabled/VECTOR,F8、NUMA off，
结束恢复原auto/PTL Disabled/N/A/NUMA=1。两轮两buffer的[冒烟](../../../contrib/moe/results/k512_stage4_vmcnt5/qwen35_smoke.json)
功能/ISA检查通过，但退出busy7%，仅保留功能证据，不作性能收益；正式48轮使用新文件独立门禁。
[汇总器](../../../contrib/moe/summarize_k512_stage4.py)拒绝缺失phase/path/样本/逐轮合同或非法时延，
[完整性反例测试](../../../contrib/moe/test_summarize_k512_stage4.py)及其它CPU审计合计31项通过（含前述8项），0.19秒。

<a id="k512-quick-and-n-loop"></a>

#### 快速确认：直接复用原始ELF，不再重复JIT

用户要求先快速测试，再考虑是否取消N完全展开。已停止剩余compact/Full测试，不继续重复完整矩阵，
也没有再采ATT或更改生产kernel。

[快速入口](../../../contrib/moe/benchmark_k512_quick.py)从已经生成的binary MLIR提取原始ELF，
**不重新编译、不重新汇编、不修改机器码**。两份ELF的.text长度相同，只有N2048的15个、N4096的31个
等待立即数字节由1变5；对应实际ISA逐指令审计同时通过。经HIP直接加载该kernel，参数为9个pointer＋M，
ABI大小76B，grid与原launcher相同；真实随机正确性仍由前述160次专项覆盖。

普通Down-only，10个输入/权重/输出buffer，每轮两个buffer，旧新使用**相同地址**；ABBA和BAAB逐轮交替。
12轮为每版本24个样本，40轮确认每版本80个样本、每个buffer每版恰好8次，外侧顺序20/20均衡。
预热和输出检查在计时外；计时前后所有buffer均finite、padding/inactive合同通过、输出逐bit相等。
性能数据仍为round-robin routing和全1权重，不把此比较替代独立随机FP32参考。

| 成功轮次 | 全部墙钟（含初始化/门禁/恢复） | ELF提取与ISA校验 | GPU库初始化 | 四点计时循环合计 |
| --- | ---: | ---: | ---: | ---: |
| 12轮快速 | 18.374秒 | 0.825秒 | 8.193秒 | 0.379秒 |
| 40轮确认 | 21.411秒 | 0.823秒 | 9.225秒 | 1.267秒 |

这里“计时循环合计”是host包围四点计时循环的耗时，包含event和同步管理；不是纯kernel时延之和。
这与之前几分钟到几十分钟的Python冷JIT不同，减少轮数并不是主要提速来源，**避免重复构建IR才是**。
不修改FlyDSL源码位置追踪配置来隐性改变生成代码；仍保留完整源码与机器码身份。

40轮确认结果如下，单元格为**旧stage4=1 → 新stage4=5，ms / 有效TFLOPS**；
有效工作量$F=2B\cdot TopK\cdot N\cdot512$，TFLOPS=$F/(t_{ms}10^9)$。

| Case / Batch | 旧1 | 新5 | 配对提升中位数 [IQR] | 新5胜轮 |
| --- | --- | --- | --- | --- |
| 35B / 16K | 0.655222 / 419.52 | 0.653522 / 420.61 | +0.432% [-0.422%,+1.128%] | 20/40 |
| 35B / 32K | 1.230522 / 446.77 | 1.227102 / 448.01 | +0.287% [-0.420%,+0.941%] | 24/40 |
| 397B / 16K | 2.356185 / 291.66 | 2.339924 / 293.68 | +0.679% [+0.206%,+1.418%] | 32/40 |
| 397B / 32K | 3.627667 / 378.86 | 3.608387 / 380.89 | +0.682% [+0.099%,+0.934%] | 31/40 |

12轮首测四点配对中位为+0.225%、+0.266%、+0.717%、-0.099%，IQR均跨零；因此追加40轮确认，
两轮独立保存、不拼接、不删除长尾。**本轮N4096有小幅正向证据，N2048仍不能确认稳定提升**。
快速测试没有gateup预热，采用直接HIP launcher，不能将其绝对时延与之前Full harness混比，
更不表示compact/完整链获得同幅收益。该阶段仅评估N循环；随后按用户要求实现的显式原型见下节，
生产默认展开方式仍不变。

证据：[12轮快速结果](../../../contrib/moe/results/k512_stage4_vmcnt5/quick_abba12_retry1.json)、
[40轮确认](../../../contrib/moe/results/k512_stage4_vmcnt5/quick_confirm_abba40.json)。
首次快速入场最高busy12%被拒绝、未产生样本；成功12轮入场/managed全机0%，退出busy4%/VRAM3%；
40轮入场0%、managed1%、退出0%/VRAM0%，均核验PTL Enabled/VECTOR,F8及1800MHz determinism，
最后恢复auto/650W/PTL Disabled/N/A/NUMA=1。完整397B中止和失败入场记录均保留，不标成已测通过。

此前已完成的[35B完整48轮](../../../contrib/moe/results/k512_stage4_vmcnt5/qwen35_abba48.json)也保留：
普通16K/32K Down配对+0.221%/+0.087%，compact +0.295%/+0.160%，四个Full IQR均跨零。
控制源码原先引用仓库根UI副本；用户清理根UI后改用原始ATT归档中的同SHA源码，不恢复已删除目录，
不改写既有JSON；汇总审计显式记录这个同内容来源替换。

#### 是否不完全展开N：建议独立原型，暂不直接替换

实际产物验证了展开成本随N显著增长：

| 普通K512 | N tile数 | 静态MFMA / ISA条数 | 原始IR字节 | ELF .text字节 |
| --- | ---: | ---: | ---: | ---: |
| N2048 | 16 | 4096 / 10877 | 12,365,497 | 79,296 |
| N4096 | 32 | 8192 / 21451 | 24,639,810 | 155,968 |

冷编译现场栈在`capture_user_location()`→`run_super_record_mfma()`→IR emit，CPU占一个核，
尚未进入测量循环。原始IR/.text约翻倍是实测事实，但不能把它换算成循环版固定倍数的编译提速预测。

**推荐仅K512先尝试“外层N运行时循环，内层8 stage/MFMA继续完全展开”**：

1. 首N与末N/drain剥离，循环只覆盖中间N，保留已核验的每stage VMEM额度与barrier次数。
2. 循环入口使用固定形状状态：两份B VGPR carry、上一N的已打包输出、尚未退休SR3及scale。
  当前Python列表和`None`不是运行时loop-carried SSA，不能只把`range_constexpr`改为`range`；
  更不能每N统一清空C而覆盖下一N stage0仍需打包的旧SR3。
3. K512每N4个K块、8个half-core，`slot=(4*n+k)&1=k&1`、`staging=(8*n+s)&1=s&1`，
  奇偶相位不随N变化，单N循环体可行；其它K不沿用这个证明。需要时再比较每次循环2个N tile。
4. 地址从N常量偏移改为uniform循环地址更新，loop-carried寄存器可能增加move/SGPR或改变VGPR分配；
  必须核验后端是否重新展开、机器码是否实际缩小、0 spill及Down时延是否不退化。
5. 单独测冷编译时间、IR/.text体积、资源与两版Down-only；只有这些通过才扩展compact/Full，
  不把N循环与stage4、padding/cache policy等多个改动混为一个实验。

已有[1x4的运行时N循环](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_1x4.py#L859-L920)
使用`range(..., init=loop_state)`和`yield next_state`，可参考其状态传递方式，不能照搬不同B/输出时序。
上述原型及验收项已记入[TODO](../../../contrib/moe/TODO.md)；后续实现与实测如下，默认仍保留全展开。

<a id="k512-nloop-prototype"></a>

### K512 N循环原型：保留stage4正式改动，冷编译显著缩短（2026-09-06）

本节是K512-only原型的历史记录。生产现已使用[通用N循环](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_nloop.py)，
旧专用实现已从生产目录清理；下面源码链接指向原实验快照，不以通用版本替代旧ISA身份。

用户要求“1合并入正式代码，尝试2，改善测试时间”。**stage4选择性1→5已作为正式默认保留**，
没有增加控制它的实验开关；stage1–3仍9、stage0/首N/后续stage及其它K/量化不变。
本轮新增的N循环为独立显式实验，不是将此前5→9或1→5重复算作收益，也没有提交/push代码。

#### 实现与默认隔离

- [当时的主builder快照](../../../contrib/moe/results/k512_nloop/final_n2048_u1/sources/gemm2_8x1.py)新增私有`_n_loop=0`参数。
  0沿用正式全展开，1/2分别每次循环处理1/2个N tile；只接受K512、PTPC、rolling。
  N128/N256回退原实现；不支持的K/量化在提前分流前显式拒绝，不静默忽略参数。
  公开`down_path`/自动selector/compact默认均未改变。
- [独立N循环原型快照](../../../contrib/moe/results/k512_nloop/final_n2048_u1/sources/gemm2_8x1_k512_nloop.py)
  保留主kernel的q0 prologue；首N补q1..q7、中间N使用运行时循环、末N单独处理，最终drain仍由主kernel执行。
  内部8 stage、MFMA packet、B预取、scale/pack/输出store顺序继续静态展开，stage4额度仍为5。
- 循环状态固定13项：2份B VGPR carry、SR3的4个fragment、SR3 scale、SR0..2的6个packed vector。
  只保存尚未退休的SR3，不在新N入口清空整个C；通过`range(..., init=state)`/`yield`传递SSA。
  循环region进入/退出时清理`FlyObjCache`的IR对象缓存，避免跨region引用失去dominance。
- K512的LDS slot与VGPR staging奇偶相位不随N改变；末N沿用原有效pending/next/future判定，
  2N展开的奇数余块和零次中间循环均单独覆盖。运行时N地址更新引入少量SALU和寄存器搬移，资源实际检查见下表。

当时实验源码SHA256：主builder为`7b272b87ba53d9f138b50974ef31fa9b37f4017470852d75c7b22212d0f7bf55`，
N循环模块为`a31f1b2f66af0337205cc5cd15195b70abc9e78200de115fe2639307ff0baaeb`。
正式默认`_n_loop=0`的10877条N2048 ISA与已验收stage4=5版本逐指令、资源完全相同。
每个最终候选归档源码副本/ISA/ELF及SHA；早期探索JSON保留原身份，不改写旧哈希以适配新源码。

#### 冷编译、机器码与动态工作量

使用[小范围编译/参考脚本](../../../contrib/moe/check_k512_nloop.py)，同一Python3.10/FlyDSL0.3.2/
gfx942环境，缓存关闭，IR/ASM dump策略一致；时间包括构建IR及编译，不含初始化、参考计算或性能采样。
只对N2048重测一次全展开冷编译；N4096控制复用已有正确性和ISA核验的ELF，避免再支付长编译。

| N / 方式 | 冷编译/秒 | 原始IR/字节 | .text/字节 | 静态ISA / MFMA | VGPR |
| --- | ---: | ---: | ---: | --- | ---: |
| 2048 / 全展开 | 210.916 | 12,365,497 | 79,296 | 10877 / 4096 | 218 |
| 2048 / 1N循环 | **8.612** | 2,628,825 | 17,216 | 2300 / 768 | 224 |
| 2048 / 2N循环 | 9.393 | 3,403,711 | 22,016 | 2951 / 1024 | 222 |
| 4096 / 全展开 | 本轮未重测 | 24,639,810 | 155,968 | 21451 / 8192 | 220 |
| 4096 / 1N循环 | **8.548** | 2,629,443 | 17,280 | 2300 / 768 | 218 |
| 4096 / 2N循环 | 9.464 | 3,404,329 | 22,080 | 2953 / 1024 | 222 |

N2048的1N循环冷编译缩短**95.917%（约24.49倍）**，原始IR缩小78.741%、机器码缩小78.289%。
这是同环境单次冷编译对照，不当成统计置信区间；未改变FlyDSL源码位置追踪配置以人为缩短耗时。
两种粒度在N4096仍保持近似相同编译规模，说明不再随N线性复制主体。
所有最终候选均48KiB LDS、96 next-free SGPR、0 private、0 VGPR/SGPR spill；没有改变LDS限制的驻留档位。

不能把静态768/1024条MFMA当作减少了数学工作。四个最终候选ISA均保留一个真实回边：

| N / 粒度 | 循环trip数 | 每次迭代MFMA | 首尾静态部分＋循环后的动态MFMA |
| --- | ---: | ---: | ---: |
| 2048 / 1N | 14 | 256 | 4096 |
| 2048 / 2N | 7 | 512 | 4096 |
| 4096 / 1N | 30 | 256 | 8192 |
| 4096 / 2N | 15 | 512 | 8192 |

按实际回边展开后重新检查全部普通VMEM load的使用和重定义，跨迭代依赖全部被wait覆盖。
没有K-padding、额外无效MFMA、LDS容量增加或将N分成多个GPU dispatch。

#### 正确性和边界

最终4个生产宽度候选各做3次随机FP8输入/权重、随机routing的FP32参考：N2048/TopK8、
N4096/TopK10最大rel_l2分别0.003307889/0.003308873，均<0.005；finite、padding和inactive NaN合同通过。
另5组边界各3次通过：N128/1N/padding0、N256/2N/padding32、N384/2N/padding64（中间循环零迭代）、
N640/2N/padding128（奇数余块）、N640/1N/padding32的descriptor满块入口。
最后一项只证明满块寻址可用，**不声称已验收compact构表＋tail＋Full链**。

[CPU循环测试](../../../contrib/moe/test_k512_nloop.py)覆盖N逐tile恰好一次、末N预取边界、动态B位置与原公式一致、
正式stage4默认及不支持参数拒绝。沿用之前stage4的160次专项和78项后中断的广泛回归记录，
不把本次针对性检查伪装成重跑七K/Full全集。

#### 40轮Down-only确认（每次仅18–21秒）

[零JIT性能入口](../../../contrib/moe/benchmark_k512_nloop.py)直接加载已有ELF。
旧控制为正式全展开stage4=5，新候选为1N/2N循环；每组10-buffer、同地址、交替ABBA/BAAB、40轮，
每版80个样本且每个buffer各8次。所有样本、长尾保留，每组独立不拼接；计时前后各buffer输出逐bit一致。
Round-robin routing、全1权重的性能输入不替代上述随机正确性。

下表为**旧全展开ms / 有效TFLOPS → 循环版ms / 有效TFLOPS**，工作量仍
$F=2B\cdot TopK\cdot N\cdot512$，有效TFLOPS=$F/(t_{ms}10^9)$。

| N / 粒度 / Batch | 旧 → 新 | 配对提升中位数 [IQR] | 新版胜轮 |
| --- | --- | --- | --- |
| 2048 / 1N / 16K | 0.654723 / 419.84 → 0.645283 / 425.98 | +1.352% [+0.796%,+1.882%] | 39/40 |
| 2048 / 1N / 32K | 1.225026 / 448.77 → 1.211607 / 453.74 | +1.218% [+0.621%,+1.791%] | 38/40 |
| 2048 / 2N / 16K | 0.653583 / 420.57 → 0.643803 / 426.96 | +1.427% [+0.700%,+2.044%] | 40/40 |
| 2048 / 2N / 32K | 1.222748 / 449.61 → 1.209987 / 454.35 | +1.092% [+0.445%,+1.801%] | 37/40 |
| 4096 / 1N / 16K | 2.324853 / 295.59 → 2.298753 / 298.94 | +0.995% [+0.659%,+1.507%] | 39/40 |
| 4096 / 1N / 32K | 3.591480 / 382.68 → 3.549600 / 387.20 | +1.140% [+0.724%,+1.798%] | 37/40 |
| 4096 / 2N / 16K | 2.319255 / 296.30 → 2.299632 / 298.83 | +0.783% [+0.216%,+1.380%] | 33/40 |
| 4096 / 2N / 32K | 3.589000 / 382.94 → 3.553720 / 386.75 | +1.143% [+0.734%,+1.662%] | 37/40 |

四份确认：[2048/1N](../../../contrib/moe/results/k512_nloop/n2048_u1_abba40.json)、
[2048/2N](../../../contrib/moe/results/k512_nloop/n2048_u2_abba40.json)、
[4096/1N](../../../contrib/moe/results/k512_nloop/n4096_u1_abba40_retry1.json)、
[4096/2N](../../../contrib/moe/results/k512_nloop/n4096_u2_abba40.json)。
对应最初32K的12轮短测中位提升为0.842%/1.366%/1.170%/0.952%，全部原始样本保留于汇总。
40轮每组总墙钟18.66/18.20/20.52/19.15秒，包含GPU初始化和硬件接管/恢复，不含已经完成的冷编译。

GPU7均核验PTL Enabled/VECTOR,F8、1800MHz determinism、650W和NUMA off，结束恢复原状态。
4096/1N第一次40轮入场最高busy8%被拒绝，无样本；CPU依赖审计后全新retry成功。
四份确认退出busy为0/1/0/3%，VRAM为0/0/6/5%，均保留实际值；门禁与恢复快照不代表全程负载监控。

#### 结论和交付范围

1. stage4=5正式默认保留，默认ISA不变；没有commit/push或改变自动selector。
2. N循环原型显著改善冷编译时间，已测Down-only没有回退，四组40轮的全部IQR为正。
   1N与2N分别对正式控制测试，**不能依据不同进程绝对时延宣称谁更快**；后续优先较小的1N循环，
   只是减少代码和编译成本的选择，不是已经证明的性能赢家。
3. 原型保留私有`_n_loop=1/2`显式使用，默认仍0；目前不将Down收益当成compact/Full或fresh ATT验收。
   测试工作流采用“一次编译并归档→后续直接ELF配对”，不再为每次采样重复冷JIT。

所有源码、边界、资源、回边工作量与配对统计由[汇总器](../../../contrib/moe/summarize_k512_nloop.py)
重算，机器证据见[最终验收汇总](../../../contrib/moe/results/k512_nloop/summary.json)。
最终39项CPU回归通过（0.24秒）；4个最终构建、5个边界构建、9份性能/拒绝记录全部独立复算，
新文档链接及40轮时延/吞吐表格自动校验通过。最终只读核验GPU7 busy0、auto/650W、
PTL Disabled/N/A、NUMA balancing=1；没有遗留后台测试任务。

## K512补测：pure epilogue基线（2026-09-05）

本节保留优化前的pure基线，当时没有为K512启用rolling特化；源码SHA256为
`67cefe2a8d665934bc88f3dd6159d171dccb3b9582da3463195907e8266ccfe0`。
shape固定为B32768、TOPK8、E256、N2048、K512、BM256、BN128、padding128B。
GPU7：1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off；采集和性能
测试结束后均恢复auto、PTL Disabled、NUMA on。

### 证据与复现参数

- [随机torch-reference结果](../../../contrib/moe/results/multik_busy80/current_k512_check.json#L1)：
  B257、E8、TOPK4、N2048，relative-L2 `0.0033062608`、finite；与原版control逐bit一致。
- [10-buffer ABBA24](../../../contrib/moe/results/multik_busy80/current_k512_abba24.json#L1)：
  每版本48个绝对样本；当前中位数 **1.609066 ms / 341.66有效TFLOPS**，
  P25--P75为1.607676--1.613416 ms。control为1.613846 ms / 340.65有效TFLOPS；
  配对ratio为0.99719、IQR [0.99578, 0.99964]、20/24胜，微小差异不作为rolling优化收益。
- [七层机器账本](../../../contrib/moe/results/multik_busy80/current_k512/analysis.json#L1)；
  raw trace为同目录的`ui_output_agent_42544_dispatch_22`，416条完整active wave、
  16个physical SIMD、208个resident batch，采样中没有early-exit wave。
- [采集命令及硬件状态](../../../contrib/moe/results/multik_busy80/current_k512/hardware.json#L1)。

沿用唯一分析器，geometry参数为`--n-blocks 16 --cores-per-n 8 --mfma-per-core 32`，
每wave4096条MFMA；窗口仍为`--first-n 2 --last-n-exclusive 14`。静态参数与K256相同：
launch1280/active1024 WG、80 CU、8 waves/WG、4 SIMD/CU、2 resident waves/SIMD。
有效工作量为$F=2\times32768\times8\times2048\times512=549,755,813,888$ FLOP；
有效TFLOPS使用$F/(t_{ms}\times10^9)$计算，不是ATT union乘roof。

### 七层结果与结论

静态CU容量损失1.538%、CU内SIMD不均衡0%，均无零任务单元。生命周期为
prologue **8.955%**、steady **89.156%**、epilogue **1.889%**；inter-batch gap占horizon
0.181%。N2--N13窗口覆盖完整steady的75.005%。

| 类别 | cycles | 占内部steady |
| --- | ---: | ---: |
| MFMA busy | 20,340,736 | **65.832%** |
| VMEM issue | 489,348 | 1.584% |
| VMEM wait | 1,749,736 | 5.663% |
| LDS issue | 2,269,924 | 7.346% |
| LDS wait | 861,288 | 2.788% |
| VALU execution | 3,340,956 | 10.813% |
| barrier | 1,685,268 | 5.454% |
| other | 160,836 | 0.521% |

总窗口30,898,092 cycles，MFMA idle10,557,356 cycles；主账本闭合。
LDS issue中的927,764 cycles为issue-stall、1,342,160为正常服务，不能全部解释成bank conflict。
最大joint见证为`VALU execution@tail + barrier@tail`（5.618%）和
`VALU execution@tail + barrier@core0->1`（5.195%），说明N尾部的集中pack/scale/store
仍在暴露；不能只因K增大就推断MFMA busy会自动超过80%。

资源：编译器210 VGPR、96 next-free SGPR、48KiB LDS、0 scratch；ATT报告
84 regular + 132 accum VGPR、112 trace SGPR，口径不同，不能直接相加后当编译器VGPR数。
目标dispatch22的单次ATT时延为1.603207 ms / 342.91有效TFLOPS，不替代ABBA24。
**这份pure基线未达到80%；下面的K512专用rolling结果是独立实现和重新采集的证据。**

## K512专用rolling：90.734% busy，时延下降25.3%（2026-09-05）

本次只优化K512，未改变K128/K192/K256/K384的调度。固定上述shape及N2--N13统计窗口，
最终clean ATT的physical SIMD MFMA union为 **90.734%**；16个采样SIMD分别为
**90.630%--90.902%**，全部超过80%。没有K-padding、额外MFMA或缩窄N窗口。

### 独立退休时序与依赖

在[该轮主builder快照](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/source_2_gemm2_8x1.py#L77)中增加
`DEDICATED_K512`，只对K512/BK128启用rolling；新增独立的
[`pack_k512_rolling()`](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/source_2_gemm2_8x1.py#L1221)。
八个core仍依次为`K0L/K1L/K2L/K3L/K0H/K1H/K2H/K3H`。

| compute位置 | 正在计算的分片 | 交织打包的已完成分片 |
| --- | --- | --- |
| 当前N，stage3 / `K3L`，packet1 | SR1的最后K贡献 | 当前SR0，已在packet0完成 |
| 当前N，stage4 / `K0H`，packet0 | SR2的首个K贡献 | 当前SR1，已在stage3完成 |
| 当前N，stage7 / `K3H`，packet1 | SR3的最后K贡献 | 当前SR2，已在packet0完成 |
| 下一N，stage0 / `K0L`，packet0 | 新SR0的首个K贡献 | 上一N的SR3 |

打包目标**有意不同于正在执行MFMA的分片**，这样才能交织独立的VALU与MFMA；不能将
打包参数改成当前`n_pair`。每份打包仍是40条VALU，使用既有scheduler group约束，
不增加硬件barrier。Scale在前四stage各加载一个quarter；下一N前四个memory stage
按`(N半区,row-pair)=(0,0)/(0,1)/(1,0)/(1,1)`写出旧N。SR3在stage0 compute完成打包，
早于stage2的高半区store；最后N沿用独立drain。B两级carry、ping/pong和4+4 read不改。

### Clean ABBA24与工作量

[最终10-buffer ABBA24](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_abba24_clean.json#L1)
使用固定的[67cefe2a pure快照](../../../contrib/moe/results/multik_busy80/current_k512/ui_output_agent_42544_dispatch_22/source_2_gemm2_8x1.py#L1)
作为control，而不是更早的85a13a版本。每版本48个样本，24个paired ratio；所有样本保留。

| 版本 | 中位时延 / ms | P25--P75 / ms | 有效TFLOPS |
| --- | ---: | ---: | ---: |
| pure control | 1.616006 | 1.614816--1.617306 | 340.19 |
| K512 rolling | **1.207905** | 1.205775--1.208965 | **455.13** |

paired candidate/control为 **0.746945**，IQR **[0.745240, 0.748239]**，**24/24胜**。
按绝对中位数，时延下降25.254%，有效吞吐提高33.786%。工作量不变：

$$
F=2\times32768\times8\times2048\times512=549,755,813,888\ \mathrm{FLOP},
\qquad \mathrm{TFLOPS}_{effective}=\frac{F}{t_{ms}\times10^9}.
$$

GPU7固定1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off；正式ABBA
和最终ATT的全机before/managed快照均为0% busy，结束恢复auto、PTL Disabled/N/A、NUMA on。
运行器现在保存全机快照，并对目标GPU的initial/managed状态再次检查，防止状态查询间
负载变化漏过门禁。

[首轮ABBA24](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_abba24.json#L1)
有明显长尾，未删除或筛选样本；它和
[首份91.253% ATT](../../../contrib/moe/results/multik_busy80/rolling_k512_v1/analysis.json#L1)
均保留作探索证据。首份ATT的initial查询出现10% busy，不作为clean终验；正式结论使用
下面的全新trace，不将两份trace拼接或取最高值。

### 七层结果：从pure到rolling

最终[机器账本](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/analysis.json#L1)与
[采集命令及硬件状态](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/hardware.json#L1)
对应dispatch22。完整采样776条wave，其中392条active、384条uniform early-exit；
16个physical SIMD、196个active resident batch，每个SIMD的实际最大驻留均为2。
12个采样SIMD各24条active wave，4个各26条；early-exit不进入active生命周期与steady。

**第1--2层**：静态分配与pure相同，launch1280/active1024/early-exit256 WG；
80 CU中64个各13任务、16个各12任务，零任务CU为0，$I_{CU}=1.53846\%$。
全设备320 SIMD中256个各26条active wave、64个各24条，分别13或12个resident batch；
CU内wave/batch不均衡和零任务SIMD均为0。上面的ATT采样数量只作sanity check，
不能拿采样中12/13任务CU的比例反推整卡分配。

**第3--5层**：active生命周期共32,641,612 cycles，三段相加闭合。

| 阶段 | pure cycles/batch | rolling cycles/batch | rolling lifecycle占比 |
| --- | ---: | ---: | ---: |
| prologue | 19,893.06 | 18,963.76 | 11.387% |
| steady | 198,051.67 | 144,960.88 | 87.043% |
| epilogue | 4,196.52 | 2,614.20 | 1.570% |

prologue百分比增大是总生命周期缩短的结果，绝对cycles/batch没有回退。
inter-batch gap为205.91 cycles/gap，占horizon0.113%；内部N2--N13覆盖完整steady的
74.350%（pure为75.005%），两者使用同一窗口规则。

**第6层**：内部窗口21,124,596 cycles；busy19,167,232、idle1,957,364，主账本闭合。

| 类别 | rolling cycles | pure steady占比 | rolling steady占比 | 变化 / 百分点 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 19,167,232 | 65.832% | **90.734%** | **+24.903** |
| VMEM issue | 165,164 | 1.584% | 0.782% | -0.802 |
| VMEM wait | 93,124 | 5.663% | 0.441% | -5.222 |
| LDS issue | 732,008 | 7.346% | 3.465% | -3.881 |
| LDS wait | 78,348 | 2.788% | 0.371% | -2.417 |
| VALU execution | 261,132 | 10.813% | 1.236% | -9.577 |
| barrier | 465,008 | 5.454% | 2.201% | -3.253 |
| other | 162,580 | 0.521% | 0.770% | +0.249 |

pure采样208个batch，rolling采样196个，不能直接拿两份总cycles相减当收益。
按batch归一，内部busy均为 **97,792 cycles/batch**；idle由 **50,756.52降至9,986.55**，
下降80.325%；窗口由148,548.52降至107,778.55 cycles/batch。相同busy工作量下空洞减少，
而不是通过额外计算增加分子。

**第7层原因转移**：原tail的集中scale/pack/CShuffle被分散。VALU占比减少9.577个百分点，
VMEM/LDS completion wait及barrier也下降；`other`仅增加0.249个百分点，不能把它省略。
剩余LDS issue为375,256 cycles issue-stall加356,752 cycles正常service，不能都称为bank
conflict。最大opcode为`ds_read_b128`（LDS issue的54.618%）；tail份额由92.476%降至23.347%。
剩余barrier的主要phase仍是tail，需要保留生产消费同步，不能凭占比直接删除。

原最大的joint见证`VALU execution@tail + barrier@tail`占steady5.618%；rolling最大的
joint变为`LDS issue@core0->1 + VALU execution@core0`，仅0.483%。这些只是定位见证，
不重复加入七类预算。原因转移、union busy和clean ABBA24三者同向，K512候选保留。

### 具体stage：N2 / K3L

[具体stage账本](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/analysis.json#L58)
选择SE0/CU1/SIMD0/slot0的第一条active wave，N2/core3，MFMA ordinal608--639。
first/last PC index为1832/1908，均对应
[`fx.mma_atom_call()`](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/source_2_gemm2_8x1.py#L1079)。
时间窗 **[109464,110036)**，共572 cycles；physical MFMA busy512 cycles，**89.510%**。
最后一条MFMA成功issue为110020，执行窗到110036；不能用raw attempt110008作为执行起点。

其60-cycle idle为VMEM issue4、LDS issue28、LDS wait8、VALU20；其余类别为0。
LDS issue进一步分为12-cycle issue-stall和16-cycle service。所有数值先与peer MFMA union
空洞求交，不是将该wave的raw stall相加。

### 正确性与ISA资源

以下随机权重/路由测试均通过独立torch FP32参考，与pure逐bit一致，relative-L2为
0.003295--0.003368；padding与uniform early-exit尾部的NaN哨兵均未被覆盖：

- 单N/drain：
  [N128/B1/padding0](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n128_b1_p0_r1.json#L1)、
  [N128/B33/padding128](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n128_b33_p128_r1.json#L1)。
- 跨N：
  [N256/padding32](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n256_b257_p32_r1.json#L1)、
  [N256/padding64](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n256_b257_p64_r1.json#L1)、
  [N2048/padding128](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n2048_b257_p128_r1.json#L1)，均B257/E8/TOPK4。
- 回退开关：
  [`MOE_8X1_ROLLING_EPILOGUE=0`](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_contract_n256_b257_p128_r0.json#L1)
  保留pure行为。正式[五K端到端回归](../../../contrib/moe/test_moe.py#L1359)为 **5 passed**。

每wave仍4096条MFMA，静态`s_barrier`仍261条。编译器VGPR为210→218，next-free SGPR96，
LDS仍48KiB，private/scratch为0；ATT口径为92 regular + 132 accum VGPR、112 SGPR。
实际驻留仍2 waves/SIMD，没有跨资源档位。debug/非debug最终ISA的10,877条指令逐行一致，
因此本次用来解释ATT的指令序列与非debug性能版本一致。

- [候选源码快照](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/source_2_gemm2_8x1.py#L1)
  SHA256：`d4353fc8dd5c6b325d72a507263c1bacec5a027fc8760fce722b3a2e46d7e13f`。
- 最终debug ISA SHA256：`61673bc7f57c3fe9be001a5eeb8401ef614778b83de6fac5fb27b44148aab667`。
- [解码code表](../../../contrib/moe/results/multik_busy80/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/code.json#L1)
  SHA256：`e0507629b81ad65d20ca9946b3ee536284e91bff121bc26917230096ea727380`。

单次ATT dispatch22为1.201605 ms / 457.52有效TFLOPS，仅用于标识trace；正式性能仍采用
ABBA24的1.207905 ms / 455.13有效TFLOPS，不使用union乘roof充当墙钟吞吐。

### K512复现入口

工作负载、随机reference、ABBA和fresh ATT统一使用同目录的
[run_moe_8x1.py](run_moe_8x1.py)，后处理只用[analyze_mfma_stall.py](analyze_mfma_stall.py)。
以下命令使用固定源码快照；本机GPU Python工作目录设为`/tmp`，避免仓库顶层同名namespace
遮蔽editable安装。管理父进程不要加载外部AITER sitecustomize shim，运行器会在子工作负载
内部处理AITER入口。ATT输出目录必须全新；空闲门禁失败时停止，不放宽阈值。

```bash
REPO=/root/workspace/luocheng/pyhip
PY="$REPO/.venv/bin/python"
TOOLS="$REPO/tests/flydsl/attn_4wave/tools"
RESULTS="$REPO/tests/contrib/moe/results/multik_busy80"
CONTROL="$RESULTS/current_k512/ui_output_agent_42544_dispatch_22/source_2_gemm2_8x1.py"
CANDIDATE="$RESULTS/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22/source_2_gemm2_8x1.py"
cd /tmp
export HIP_VISIBLE_DEVICES=7 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYHIP_JIT_LOG=0
export MOE_8X1_ROLLING_EPILOGUE=1
export PYTHONPATH="$REPO/src:/opt/aiter:/usr/local/lib/python3.10/dist-packages:/root/workspace/luocheng/FlyDSL/build-fly/python_packages:/root/workspace/luocheng/FlyDSL/python"

"$PY" "$TOOLS/run_moe_8x1.py" --mode check --k 512 --n 2048 \
  --batch 257 --experts 8 --topk 4 --random-routing \
  --source "$CANDIDATE" --control "$CONTROL" --output /tmp/moe8x1-k512-recheck.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode bench --k 512 --rounds 24 --managed \
  --source "$CANDIDATE" --control "$CONTROL" --output /tmp/moe8x1-k512-rebench.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode att --k 512 --managed \
  --source "$CANDIDATE" --output /tmp/moe8x1-k512-recapture

# CPU复算本文已保存的最终trace；不需要GPU或重新采集。
env -u PYTHONPATH "$PY" "$TOOLS/analyze_mfma_stall.py" \
  "$RESULTS/rolling_k512_v1_clean/ui_output_agent_53149_dispatch_22" \
  --n-blocks 16 --cores-per-n 8 --mfma-per-core 32 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 3 \
  --json /tmp/moe8x1-k512-reanalysis.json
```

## K320：128+128+64混合分块（2026-09-05）

**结论：混合分块保留，K320默认采用128+128+64；旧5×64已从当前入口删除，只保留历史源码快照作对照。**
独立实现位于[gemm2_8x1_k320.py](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k320.py)，
[通用入口](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L70)只做编译期分流，
不再把混合宽度条件堆进已验证的K512调度。

固定shape仍为B32768、TOPK8、E256、N2048、BM256、BN128、padding128B、8 waves/WG。
最终ATT busy为 **81.522%**，首轮为81.398%；最终16个采样SIMD为
**81.124%--81.846%**，全部超过80%。窗口仍为N2--N13，MFMA execution window仍为16 cycles。

### 分块、输入流水与退休时序

| core | K区间 | N半区 | MFMA条数 |
| ---: | --- | --- | ---: |
| 0 | [0,128) | L | 32 |
| 1 | [128,256) | L | 32 |
| 2 | [256,320) | L | 16 |
| 3 | [0,128) | H | 32 |
| 4 | [128,256) | H | 32 |
| 5 | [256,320) | H | 16 |

每N共160条、每wave共2560条MFMA，与5×64相同；不是将320补齐到384。
A按三个真实区间gather。B保留两级carry：BK128时每线程加载16B，BK64尾段每线程加载8B，
两个4-wave组均全员参与，不经过旧BK64路径的半组条件exec和FP8逐字节物化。
穷举字节映射验证，两组每次合计恰好覆盖N64×BK的8192B或4096B，各字节一次。

LDS按最大BK128分配，两个N半区的起点固定相距8192B；尾段在各自半区内紧凑放置。
这点用于防止不同K宽度的下一次commit覆盖尚未消费的另一半B，不能直接用尾段64去改变
两个半区的固定间距。单N情况下不发送不存在的下一N预取。

40条VALU的输出打包仅穿插到BK128的16-MFMA packet，不占用BK64的8-MFMA packet：

| 计算位置（均为packet0） | 打包对象 |
| --- | --- |
| 当前N/core3，K0H128 | 当前SR0，低半区已在core2完成 |
| 当前N/core4，K1H128 | 当前SR1 |
| 下一N/core0，K0L128 | 上一N的SR2 |
| 下一N/core1，K1L128 | 上一N的SR3 |

下一N的memory core0/1/3/4分别写出旧N的四个`N64 × row-pair`块，避开短BK64 core。
SR2/SR3在core0/1打包，早于core3/4的高N半区store；最后N单独drain这两份。
打包的是独立分片而非当前MFMA目标，不能把两者的`n_pair`改成相同。

### 正式性能：两种对照分别报告

两组均为10 rotating buffers、24轮ABBA/BAAB，每版本48个样本；不混合两组绝对时延：

| 对照 | control ms / 有效TFLOPS | mixed ms / 有效TFLOPS | paired ratio / IQR | 胜出 |
| --- | --- | --- | --- | --- |
| [已调优5×64单store](../../../contrib/moe/results/multik_busy80/mixed_k320_final_vs_single64_abba24.json#L1) | 0.986064 / 348.45 | **0.894944 / 383.93** | **0.908141 / [0.903181,0.913842]** | 24/24 |
| [原始5×64 pure](../../../contrib/moe/results/multik_busy80/mixed_k320_final_vs_pure_abba24.json#L1) | 1.964728 / 174.88 | **0.892584 / 384.95** | **0.453930 / [0.452500,0.454688]** | 24/24 |

相对已调优5×64单store，绝对中位时延下降 **9.241%**、有效吞吐提高10.182%；相对原始pure，
时延下降54.570%，但后者包含输入staging与rolling等全部改进，不能全部归因于合并K块。
前一组mixed的P25--P75为0.885034--0.900214 ms，control为0.981634--0.991324 ms。

$$
F=2\times32768\times8\times2048\times320=343,597,383,680\ \mathrm{FLOP},
\qquad \mathrm{TFLOPS}_{effective}=F/(t_{ms}\times10^9).
$$

性能和ATT均使用GPU7、1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off，
全机busy≤5%/VRAM≤20%的入场门禁不放宽。拒绝的尝试未产生性能样本。
[最终采集状态](../../../contrib/moe/results/multik_busy80/mixed_k320_final/hardware.json#L1)
记录before全机0%、managed全机0--1%；恢复查询busy为11%，不是全程负载监控数据，保留原值。
两轮ATT都报告，不选择最高一轮来隐瞒波动；硬件配置均恢复auto、PTL Disabled/N/A、NUMA on。

### 七层ATT与原因转移

证据为[最终账本](../../../contrib/moe/results/multik_busy80/mixed_k320_final/analysis.json#L1)及
[首轮账本](../../../contrib/moe/results/multik_busy80/mixed_k320_v1/analysis.json#L1)。
最终trace为416条完整active wave、16个physical SIMD、208个resident batch，无early-exit
样本；各SIMD均26条active wave，实际最大驻留为2。首轮408条active、128条early-exit，
204个batch，不能把两轮总cycles直接相减。

**第1--2层**：静态分配不变，1280 launch / 1024 active / 256 early-exit WG；
64个CU各13个active WG、16个各12个，$I_{CU}=1.53846\%$、critical inflation1.5625%。
256个SIMD各26条wave、64个各24条，对应13或12个resident batch；零任务和CU内不均衡均为0。
采样CU分布只作sanity check，不当作整卡统计。

**第3--5层**：最终active lifecycle为24,113,608 cycles，分段闭合：

| 阶段 | cycles/batch | lifecycle占比 |
| --- | ---: | ---: |
| prologue | 12,307.58 | 10.616% |
| steady | 100,471.15 | 86.665% |
| epilogue | 3,152.08 | 2.719% |

inter-batch gap为693.56 cycles/gap，占horizon0.549%；内部N2--N13覆盖完整steady的74.388%。
短core与长core使用真实长度标记，不将64尾段假装成32条MFMA。

**第6层**：窗口15,545,592 cycles = busy12,673,024 + idle2,872,568。

| 类别 | mixed cycles | 原始5×64 pure | 协作64位预取v3 | 128+128+64 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 12,673,024 | 34.956% | 74.801% | **81.522%** |
| VMEM issue | 347,956 | 2.466% | 7.456% | 2.238% |
| VMEM wait | 91,108 | 20.119% | 1.374% | 0.586% |
| LDS issue | 1,310,788 | 5.441% | 5.989% | 8.432% |
| LDS wait | 75,808 | 2.123% | 1.534% | 0.488% |
| VALU execution | 294,876 | 10.946% | 1.784% | 1.897% |
| barrier | 632,596 | 23.394% | 5.888% | 4.069% |
| other | 119,436 | 0.555% | 1.176% | 0.768% |

pure与v3证据分别为[原始账本](../../../contrib/moe/results/multik_busy80/base_k320/analysis.json#L1)、
[协作预取账本](../../../contrib/moe/results/multik_busy80/coop64_k320_v3/analysis.json#L1)。
v3仍是连续store版本，不是上面ABBA24的单store控制版本；没有给未采集的单store版本
套用v3的busy。探索过程的rolling-only为36.859%、dword staging为65.625%，两者未达标。

按batch归一，pure的窗口/busy/idle为175,029.73 / 61,184 / 113,845.73 cycles，mixed为
74,738.42 / 60,928 / 13,810.42。两个resident wave的内部N窗口交集端点随core宽度变化，
所以内部busy分子略有不同；没有修改交集规则或偷偷放大分子，每条完整wave仍严格2560 MFMA。

**第7层**：输入物化等待和barrier大幅下降，长core能覆盖更多输出工作；相对v3，VMEM issue
下降5.217个百分点，但LDS issue上升2.443个百分点，剩余瓶颈已转为LDS，不能只报下降项。
LDS issue包含665,028 cycles issue-stall与645,760 cycles正常service；最大opcode为
`ds_read_b128`（该类47.704%），最大phase为`core2->3`（27.281%）。最大joint见证为
`LDS issue@core2->3 + barrier@core2->3`，占steady1.481%，不能再加到主预算。

### 64尾段实例、正确性与资源

[尾段实例](../../../contrib/moe/results/multik_busy80/mixed_k320_final/analysis.json#L66)：
SE0/CU1/SIMD0/slot0，N2/core2，ordinal384--399，PC index1438--1456。
对应[`mma()`历史调用](../../../contrib/moe/results/multik_busy80/mixed_k320_final/ui_output_agent_9579_dispatch_22/source_2_gemm2_8x1_k320.py#L294)，窗口
**[96820,97076)**，恰好16条MFMA、256 cycles，physical busy256、idle0。
该wave这些MFMA的raw stall合计168 cycles，但被union覆盖，不能算成168-cycle physical idle。
这个100%短stage仅是尾段正确分段的见证，整体达标使用完整N2--N13的81.522%。

- 7组独立随机FP32参考检查：N128/B1/padding0、N128/B33/padding128、N256/B257/padding32/64、
  N2048/B257/padding128，以及N128/N256的rolling禁用；relative-L2为0.003162--0.003322，
  与原始5×64 pure逐bit一致，padding和inactive tail哨兵未写。
  [生产宽度结果](../../../contrib/moe/results/multik_busy80/mixed_k320_final_contract_n2048_b257_p128_r1.json#L1)。
- 默认dispatcher的[七K端到端回归](../../../contrib/moe/test_moe.py#L1359)为 **7 passed**；
  分析器[10项CPU测试](test_analyze_mfma_stall.py)通过，旧K512 JSON复算逐字节相同。
- 混合版为208 VGPR、96 next-free SGPR、48KiB LDS、0 private/scratch；ATT为80 regular +
  128 accum VGPR、112 SGPR，实际2 waves/SIMD。5×64单store为198 VGPR、32KiB LDS。
- 每wave MFMA仍2560；静态`s_barrier`由325降至197。最终TOPK8非debug与ATT的8465条
  指令完全一致；不能拿TOPK4随机检查的ISA做生产TOPK8的逐指令比较。
- K320定版时重新实编译K512与K640 pure，分别10877/12537条指令，与当时各自验证过的生产ISA完全一致。

该轮身份记录：入口源码SHA256为`6509d1a9eddb2b11d249798723d9d7606a3f61c534c89097fbe7e1b96bf6c6a9`；
[独立builder快照](../../../contrib/moe/results/multik_busy80/mixed_k320_final/ui_output_agent_9579_dispatch_22/source_2_gemm2_8x1_k320.py#L1)
为`e17234c0bd6787eadf6063acb6e676b7912b7a4215ef3436526ea6a003c9ac97`；最终debug ISA为
`42914cbfc402406db626a92a771a4597448cab81d29b4c4f642585ef12666b01`；
[code表](../../../contrib/moe/results/multik_busy80/mixed_k320_final/ui_output_agent_9579_dispatch_22/code.json#L1)
为`7ca13bb790ec0555c46e9c7ec586cce7669261a880ff7fddda00d8ca10bc3ea5`。
运行器同时记录入口与独立builder哈希，不能只绑定入口源码。

最终单次ATT dispatch22为0.845723 ms / 406.28有效TFLOPS，只标识trace，不替代ABBA24的
0.8926--0.8949 ms，也不将ATT union乘roof当成有效吞吐。

### 混合geometry复现

沿用K512节的环境变量；K320显式使用`--tile-k 128`，控制版本独立设置`--control-tile-k 64`。
控制必须使用下面的历史5×64单store快照，不能再把当前入口当作BK64控制，也不能用更早的
协作预取v3冒充单store版本。重新运行前记录入口与独立builder哈希；K320独立builder未改，
生产ISA已复验相同。采集新trace仍须全新输出目录，不能覆盖已有证据。

```bash
CONTROL320="$RESULTS/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py"
export MOE_8X1_ROLLING_EPILOGUE=1
"$PY" "$TOOLS/run_moe_8x1.py" --mode bench --k 320 \
  --tile-k 128 --control-tile-k 64 --rounds 24 --managed \
  --control "$CONTROL320" \
  --output /tmp/moe8x1-k320-mixed-rebench.json

PYTHONPATH=/usr/local/lib/python3.10/dist-packages "$PY" "$TOOLS/analyze_mfma_stall.py" \
  "$RESULTS/mixed_k320_final/ui_output_agent_9579_dispatch_22" \
  --n-blocks 16 --cores-per-n 6 --core-mfma-counts 32 32 16 32 32 16 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 2 \
  --json /tmp/moe8x1-k320-mixed-reanalysis.json
```

## K640：寄存器与LDS资源基线（2026-09-05）

**本节保留优化前8x1/BK128 pure的资源基线；后续rolling终验见下一节。**
K640为5×128、每N十个32-MFMA core，每wave5120条MFMA，仍是BM256/BN128/512线程。

| 配置 | 编译器VGPR | next-free SGPR | LDS | private/scratch | 实际waves/SIMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| K512 rolling（已验证） | 218 | 96 | 48KiB | 0 | 2 |
| K640 pure | **224** | **96** | **48KiB** | **0** | **2** |
| K320混合分块 | 208 | 96 | 48KiB | 0 | 2 |

LDS由两个最大B tile和CShuffle组成：

$$
LDS=2\times128\times128\times1+4\times16\times128\times2=49,152\ \mathrm{bytes}.
$$

sorted IDs重用CShuffle，不额外占1KiB。K总量从512到640增加的是循环次数和常驻A，
不是B槽尺寸。硬件LDS容量为64KiB，当前还剩16KiB；但两个48KiB WG放不进一个CU，
所以LDS只允许1 WG/CU，即8 waves/CU、2 waves/SIMD。剩余16KiB不表示还能再驻留一个WG。

A按当前gather映射每增加128个K逻辑上增加16个32位寄存器/线程；总寄存器变化仍取决于
编译器复用及临时变量，不能仅线性外推。K640的224 VGPR、0 scratch和实测2条resident wave
证明这份pure基线容纳得下；加入rolling或更多carry后仍须重新检查分配粒度和spill。
ATT报告96 regular + 128 accum VGPR、112 SGPR，与编译器字段分别列出，不混用口径。

证据：[随机参考/输出契约](../../../contrib/moe/results/multik_busy80/base_k640_check.json#L1)
relative-L2为0.003312115；[资源与七层账本](../../../contrib/moe/results/multik_busy80/base_k640/analysis.json#L1)
有416条完整active wave、208个batch，16个采样SIMD均核验最大驻留2。
[采集状态](../../../contrib/moe/results/multik_busy80/base_k640/hardware.json#L1)记录相同PTL/DPM协议。
K320定版时重编译的12537条K640生产指令与该pure基线相同；当前rolling的资源另列于下节。

这份K640 pure的N2--N13 MFMA busy为 **70.579%**，未达到80%；资源足够不是busy达标的保证。
单次ATT为1.854688 ms / 370.52有效TFLOPS，工作量为
$2\times32768\times8\times2048\times640=687,194,767,360$ FLOP；该次只做资源审计，
时延仅作trace身份，不作性能验收。后续正式ABBA24见下节。

## K640 rolling终验：91.515% busy，时延下降20.5%（2026-09-05）

**保留K640专用rolling，默认启用；`MOE_8X1_ROLLING_EPILOGUE=0`保留pure回退。**
本次只改变K640的输出退休，B两级carry、ping/pong、4+4 B-read及其他K的时序不改。
固定B32768、TOPK8、E256、N2048、K640、BM256、BN128、padding128B、8 waves/WG。

### 十core退休时序

[`DEDICATED_K640`](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py#L85)
选择独立的[`pack_k640_rolling()`](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py#L1329)，
不把K640硬编码塞进K512函数。每N顺序仍为`K0L..K4L → K0H..K4H`，每core32条MFMA。

| compute位置 | 正在计算 | 交织打包 |
| --- | --- | --- |
| 当前N/core4，K4L，packet1 | SR1的最后K贡献 | 已在packet0完成的当前SR0 |
| 当前N/core5，K0H，packet0 | SR2的首个K贡献 | 已完成的当前SR1 |
| 当前N/core9，K4H，packet1 | SR3的最后K贡献 | 已在packet0完成的当前SR2 |
| 下一N/core0，K0L，packet0 | 新SR0的首个K贡献 | 上一N的SR3 |

每份40条VALU仅与独立分片的16-MFMA packet交织。Scale在前四stage各加载一个quarter，
下一N前四个memory stage依次写出旧N的`(N半区,row-pair)=(0,0)/(0,1)/(1,0)/(1,1)`。
旧SR3在core0 compute打包，早于core2 memory的高半区store；最后N由drain补齐。
没有改变数学工作量，也没有添加无效MFMA或缩窄统计窗口。

### 正式ABBA24：保留全部样本

[正式复测](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_abba24_clean.json#L1)
使用10 rotating buffers、24轮ABBA/BAAB；每版本48个样本、24个paired ratio。
control固定为[7abaacbd pure快照](../../../contrib/moe/results/multik_busy80/base_k640/ui_output_agent_49139_dispatch_22/source_2_gemm2_8x1.py#L1)。

| 版本 | 中位时延 / ms | P25--P75 / ms | 有效TFLOPS |
| --- | ---: | ---: | ---: |
| pure control | 1.869247 | 1.868198--1.870678 | 367.63 |
| K640 rolling | **1.485366** | 1.484686--1.486786 | **462.64** |

paired ratio **0.794607**，IQR **[0.793885,0.795694]**，**23/24轮胜出**。
按绝对中位数，时延下降 **20.537%**、有效吞吐提高25.844%。工作量为：

$$
F=2\times32768\times8\times2048\times640=687,194,767,360\ \mathrm{FLOP},
\qquad \mathrm{TFLOPS}_{effective}=\frac{F}{t_{ms}\times10^9}.
$$

候选第2个绝对样本为 **2.330639 ms**，已计入全部统计，导致第1轮paired ratio为1.020485，
因此不能报告24/24胜。其余候选样本在1.480406--1.490086 ms；没有剔除异常值。
[首轮ABBA24](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_abba24.json#L1)
有更多长尾，也完整保留（中位1.485426 ms / 462.62有效TFLOPS、ratio0.795365），不与复测拼接。
“clean”表示通过入场门禁，不表示全程无系统噪声或所有样本都等于中位数。

GPU7使用1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off。
正式复测before全机0% busy、managed全机1%，符合busy≤5%/VRAM≤20%门禁；
被门禁拒绝的尝试未产生性能样本。结束恢复auto、PTL Disabled/N/A、NUMA on。

### 七层ATT与原因转移

最终[机器账本](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/analysis.json#L1)和
[采集状态/命令](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/hardware.json#L1)
对应fresh dispatch22。536条wave全部完整，其中408条active、128条uniform early-exit；
16个physical SIMD、204个active resident batch。各SIMD的实际最大active驻留均为2，
busy范围 **91.425%--91.574%**。12个采样SIMD各26条active wave，4个各24条。

**第1--2层**：静态分配不变，launch1280/active1024/early-exit256 WG；
80 CU中64个各13任务、16个各12任务，$I_{CU}=1.53846\%$，critical inflation1.5625%。
256个SIMD各26条wave、64个各24条，对应13或12个resident batch；零任务CU/SIMD与CU内
wave/batch不均衡均为0。采样数量只作sanity check，不替代整卡静态分配。

**第3--5层**：active lifecycle为41,856,836 cycles，三段相加闭合。

| 阶段 | pure cycles/batch | rolling cycles/batch | rolling lifecycle占比 |
| --- | ---: | ---: | ---: |
| prologue | 22,675.15 | 22,656.53 | 11.042% |
| steady | 230,539.04 | 180,026.88 | 87.741% |
| epilogue | 4,152.54 | 2,497.16 | 1.217% |

prologue占比增加来自总生命周期缩短，绝对时间没有回退。Inter-batch gap为233.43 cycles/gap，
占horizon0.105%。内部N2--N13覆盖完整steady的74.275%（pure为75.206%），窗口规则不变。

**第6层**：内部窗口27,277,712 cycles = busy24,963,072 + idle2,314,640。

| 类别 | rolling cycles | pure steady占比 | rolling steady占比 | 变化 / 百分点 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 24,963,072 | 70.579% | **91.515%** | **+20.936** |
| VMEM issue | 169,252 | 1.324% | 0.620% | -0.703 |
| VMEM wait | 226,296 | 4.660% | 0.830% | -3.831 |
| LDS issue | 706,264 | 6.429% | 2.589% | -3.840 |
| LDS wait | 79,376 | 2.406% | 0.291% | -2.115 |
| VALU execution | 275,504 | 9.263% | 1.010% | -8.253 |
| barrier | 654,072 | 4.782% | 2.398% | -2.384 |
| other | 203,876 | 0.556% | 0.747% | +0.191 |

pure采样208个batch，rolling采样204个，绝对cycles比较必须先归一。内部busy均为
**122,368 cycles/batch**；idle从 **51,010.31降至11,346.27**，下降 **77.757%**。
内部窗口从173,378.31降至133,714.27 cycles/batch，同一busy工作量下空洞减少。

**第7层**：集中在N尾部的scale/pack/CShuffle被分散，VALU、VMEM/LDS completion wait和
barrier均下降；`other`增加0.191个百分点也列入账本，不隐去暴露转移。
剩余LDS issue为354,884 cycles issue-stall与351,380 cycles正常service，不能都称为bank conflict。
LDS issue的tail份额由91.133%降至25.955%，最大opcode变为`ds_read_b128`（该类55.147%）。
原最大joint见证`VALU execution@tail + barrier@tail`占steady4.821%；rolling最大joint变为
`barrier@core8->9 + other@core9`，仅0.469%。Joint只是定位见证，不能再加到互斥预算。
原因转移、physical union与正式ABBA方向一致，候选保留。

### 具体stage：N2/K4L

[具体stage账本](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/analysis.json#L58)
为SE0/CU1/SIMD0/slot0，N2/core4，ordinal768--799，first/last PC index2117/2195，
对应[`fx.mma_atom_call()`](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py#L1124)。
窗口 **[125128,125692)** 共564 cycles，physical busy512 cycles，**90.780%**；
idle52 cycles分为VMEM issue4、VMEM wait32、LDS issue16，后者全部是service。
最后一条MFMA从successful issue125676执行到125692，不能从attempt125664开始计忙。
该wave的32条MFMA raw stall合计196 cycles，不能将它当作physical idle；上面的52 cycles
来自同SIMD resident-wave MFMA并集之外的互斥分类。

### 正确性、资源和源码身份

六组随机权重/路由的独立torch FP32参考及padding/inactive-tail哨兵全部通过，与pure逐bit一致：

- [N128/B1/padding0](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n128_b1_p0_r1.json#L1)、
  [N128/B33/padding128](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n128_b33_p128_r1.json#L1)覆盖单N/drain。
- [N256/padding32](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n256_b257_p32_r1.json#L1)、
  [N256/padding64](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n256_b257_p64_r1.json#L1)、
  [N2048/padding128](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n2048_b257_p128_r1.json#L1)，均B257/E8/TOPK4。
- [rolling禁用回退](../../../contrib/moe/results/multik_busy80/rolling_k640_v1_contract_n256_b257_p128_r0.json#L1)。

relative-L2范围0.003075--0.003365；[七K端到端回归](../../../contrib/moe/test_moe.py#L1359)为
**7 passed**。K320/K512生产ISA重新实编译，8465/10877条指令分别与已验证版本逐行相同。

| 资源 | pure | rolling |
| --- | ---: | ---: |
| 编译器next-free VGPR | 224 | **238** |
| next-free SGPR | 96 | 96 |
| LDS | 48KiB | 48KiB |
| private/scratch | 0 | **0** |
| 每wave MFMA / 静态barrier | 5120 / 325 | 5120 / 325 |
| 实际active waves/SIMD | 2 | 2 |

ATT口径为112 regular + 128 accum VGPR、112 SGPR；VGPR分配粒度与编译器next-free字段
不同，不能拿字段直接混算。新增rolling仍无spill、未改变驻留档位，但以后增加carry必须重验。
最终TOPK8性能版与debug ATT版的12647条指令完全一致。

- [源码快照](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py#L1)
  SHA256：`91eb56d2860390bc510b79870f4e6f36b38cc8c954a222b2fdc86fc7a8d57d7e`。
- 最终debug ISA SHA256：`a75f456e4d6bb3f3644e0a6a8db87ee424841e90cd6665b17ea7554e909f00c0`。
- [code表](../../../contrib/moe/results/multik_busy80/rolling_k640_v1/ui_output_agent_56200_dispatch_22/code.json#L1)
  SHA256：`2b81e8ff773c2256b670aaccf23b3539951655f2949ce0ef86a6d84aaeab5aa2`。

单次ATT dispatch22为1.481246 ms / 463.93有效TFLOPS，仅标识trace；正式性能采用未筛选ABBA24
中位数1.485366 ms / 462.64有效TFLOPS，不将union乘roof当成有效吞吐。

### K640复现入口

沿用K512节的`REPO/PY/TOOLS/RESULTS`及GPU环境变量。两份源码快照可直接作为同进程对照，
本次K640辅助函数在入口文件内，无额外K640源码依赖。ATT输出目录必须全新。

```bash
CONTROL640="$RESULTS/base_k640/ui_output_agent_49139_dispatch_22/source_2_gemm2_8x1.py"
CANDIDATE640="$RESULTS/rolling_k640_v1/ui_output_agent_56200_dispatch_22/source_2_gemm2_8x1.py"
export MOE_8X1_ROLLING_EPILOGUE=1

"$PY" "$TOOLS/run_moe_8x1.py" --mode check --k 640 --n 2048 \
  --batch 257 --experts 8 --topk 4 --random-routing \
  --source "$CANDIDATE640" --control "$CONTROL640" --output /tmp/moe8x1-k640-recheck.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode bench --k 640 --rounds 24 --managed \
  --source "$CANDIDATE640" --control "$CONTROL640" --output /tmp/moe8x1-k640-rebench.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode att --k 640 --managed \
  --source "$CANDIDATE640" --output /tmp/moe8x1-k640-recapture

PYTHONPATH=/usr/local/lib/python3.10/dist-packages "$PY" "$TOOLS/analyze_mfma_stall.py" \
  "$RESULTS/rolling_k640_v1/ui_output_agent_56200_dispatch_22" \
  --n-blocks 16 --cores-per-n 10 --mfma-per-core 32 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 4 \
  --json /tmp/moe8x1-k640-reanalysis.json
```

## K192基线：3×64 / pure epilogue（2026-09-05）

本节保留128+64实验之前的pure基线；源码SHA256与K512 pure基线相同，为`67cefe2a...`。
性能shape为B32768、TOPK8、E256、N2048、K192、BM256、BN128、padding128B。
GPU7使用1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off；每次结束
均恢复auto、PTL Disabled、NUMA on。

### 证据与复现参数

- [随机torch-reference](../../../contrib/moe/results/multik_busy80/current_k192_check.json#L1)：
  B257、E8、TOPK4、N2048，relative-L2为`0.0033148662`，finite；与原版control逐bit一致。
- [10-buffer ABBA24](../../../contrib/moe/results/multik_busy80/current_k192_abba24.json#L1)：
  当前中位数 **1.406506 ms / 146.57有效TFLOPS**，P25--P75为1.399176--1.409646 ms。
  control为1.406606 ms / 146.56有效TFLOPS；配对ratio 0.99939，IQR
  [0.99633, 1.00331]，14/24胜，属于基本持平而非优化收益。每版本48个样本。
- [七层机器账本](../../../contrib/moe/results/multik_busy80/current_k192/analysis.json#L1)；
  raw trace位于同目录`ui_output_agent_27246_dispatch_22`，416条完整active wave、
  16个physical SIMD、208个resident batch，采样中没有early-exit wave。
- [实际采集命令及硬件状态](../../../contrib/moe/results/multik_busy80/current_k192/hardware.json#L1)。

K192使用BK64：3个K块乘2个N半区，即每N **6个compute core，每core16条MFMA**。
分析器参数为`--n-blocks 16 --cores-per-n 6 --mfma-per-core 16`，每wave1536条MFMA；
不能套用K384的6×32。每条MFMA的执行窗仍是16 cycles，与每core的指令条数是两个概念。
窗口固定为`--first-n 2 --last-n-exclusive 14`，实际波形确认2 resident waves/SIMD。
静态参数为launch1280/active1024 WG、80 CU、8 waves/WG、4 SIMD/CU。

有效工作量$F=2\times32768\times8\times2048\times192=206,158,430,208$ FLOP，
有效TFLOPS由$F/(t_{ms}\times10^9)$求出，不使用ATT模型吞吐。

### 七层结果与结论

静态CU容量损失1.538%，CU内SIMD不均衡0%，均无零任务单元。生命周期为prologue
5.536%、steady92.020%、epilogue2.444%；inter-batch gap占horizon0.295%。内部N2--N13
窗口覆盖完整steady的75.937%。

| 类别 | cycles | 占内部steady |
| --- | ---: | ---: |
| MFMA busy | 7,614,464 | **28.005%** |
| VMEM issue | 880,412 | 3.238% |
| VMEM wait | 5,922,708 | 21.783% |
| LDS issue | 1,882,116 | 6.922% |
| LDS wait | 802,896 | 2.953% |
| VALU execution | 3,744,168 | 13.770% |
| barrier | 6,224,300 | 22.892% |
| other | 118,976 | 0.438% |

总窗口27,190,040 cycles，MFMA idle19,575,576 cycles，主账本闭合。
LDS issue包含693,268 cycles issue-stall和1,188,848 cycles正常服务。
最大joint见证为`VALU execution@tail + barrier@tail`（6.495%）和
`VALU execution@tail + barrier@core0->1`（5.953%），表明集中输出退休暴露明显。
此外B quarter预取路径的`vmcnt(0)`源码聚合项占内部steady13.132%，不应将所有损失归咎
于epilogue；需要另外检查BK64的条件加载、staging寄存器物化及预取距离。

资源为编译器171 VGPR、96 next-free SGPR、32KiB LDS、0 scratch；ATT口径为
44 regular + 132 accum VGPR、112 trace SGPR。目标dispatch22单次ATT时延为
1.347405 ms / 153.00有效TFLOPS，不替代ABBA24。**这份pure基线尚未达到80%。**

## K192测试：128+64，时延下降52.2%，busy未到80%（2026-09-05）

**128+64数值与性能测试通过，但physical MFMA busy为69.280%，尚未达到80%。**
保留独立的[gemm2_8x1_k192.py](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k192.py)，
测试当时仅显式`tile_k=128`选择它，默认仍是3×64 pure。随后按要求删除旧分支，
[当前K192默认与显式128都分流到混合实现](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L60-L72)。
以下保留该轮历史数据；默认切换不把“比基线快”改写成“busy达标”。
K320/K512/K640已验证实现保持不变，清理后的回归见[分支清理审计](#bk64-cleanup)。

### 真实分块与退休时序

固定B32768、TOPK8、E256、N2048、BM256、BN128、padding128B、512线程。
每N依次执行`K0L128 → K1L64 → K0H128 → K1H64`，MFMA数量为 **32/16/32/16**，
共96条/每N、1536条/每wave，与3×64基线相同；没有把K192补齐到256。

A只gather `[0,128)`和`[128,192)`；B使用原始dword staging，BK128每线程16B、BK64每线程8B，
全组参与，避免旧BK64条件路径的逐字节物化。两个N半区的LDS起点固定相距8192B，
尾段在各自半区内紧凑存放；单N prologue不发送不存在的下一N请求。

| compute位置 | 交织打包的已完成分片 |
| --- | --- |
| 当前N/core2，K0H128，packet0 | 当前SR0 |
| 当前N/core2，K0H128，packet1 | 当前SR1 |
| 下一N/core0，K0L128，packet0 | 上一N的SR2 |
| 下一N/core0，K0L128，packet1 | 上一N的SR3 |

40条VALU只与长BK128的16-MFMA packet交织，不放进BK64的8-MFMA packet。
四个memory stage各加载一个scale quarter、各写出上一N的一份`N64 × row-pair`。
上一N的SR2/SR3在新N/core0 compute完成打包，早于core2/3 memory读取；最后N单独drain。

### 正式ABBA24

[完整测量](../../../contrib/moe/results/multik_busy80/mixed_k192_final_abba24.json#L1)
使用10 rotating buffers、24轮ABBA/BAAB，每版本48个样本、24个paired ratio，全量保留。
control为[67cefe2a pure快照](../../../contrib/moe/results/multik_busy80/current_k192/ui_output_agent_27246_dispatch_22/source_2_gemm2_8x1.py#L1)。

| 版本 | 中位时延 / ms | P25--P75 / ms | 有效TFLOPS |
| --- | ---: | ---: | ---: |
| 3×64 pure | 1.405166 | 1.398586--1.410986 | 146.71 |
| **128+64 rolling实验** | **0.672143** | 0.664703--0.676482 | **306.72** |

paired ratio **0.477519**，IQR **[0.475182,0.479630]**，**24/24轮胜出**。
绝对中位时延下降 **52.166%**，有效吞吐提高109.058%。该收益包括混合K块、无条件原始B预取
和rolling退休，不全归因于少两个core。

$$
F=2\times32768\times8\times2048\times192=206,158,430,208\ \mathrm{FLOP},
\qquad \mathrm{TFLOPS}_{effective}=F/(t_{ms}\times10^9).
$$

GPU7固定1800MHz determinism、650W、PTL `Enabled / VECTOR,F8`、NUMA off；
ABBA前全机0% busy、managed全机1%，目标initial查询4%，均通过busy≤5%/VRAM≤20%门禁。
测试结束恢复auto、PTL Disabled/N/A、NUMA on；门禁拒绝的尝试不产生性能样本。

### 七层ATT与未达标原因

[机器账本](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/analysis.json#L1)及
[采集状态](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/hardware.json#L1)
对应fresh dispatch22，416条完整active wave、16个physical SIMD、208个resident batch；
采样无early-exit，各SIMD均26条active wave，实际最大驻留均为2。
所有采样SIMD的busy为 **69.261%--69.304%**，不是少数SIMD把平均值拉低。

**第1--2层**：静态分配不变，launch1280/active1024/early-exit256 WG；64个CU各13任务，
16个CU各12任务，$I_{CU}=1.53846\%$、critical inflation1.5625%。256个SIMD各26条wave，
64个各24条，分别13或12个resident batch；零任务与CU内wave/batch不均衡均为0。

**第3--5层**：active lifecycle为17,173,460 cycles，三段闭合。

| 阶段 | 3×64 pure cycles/batch | 128+64 cycles/batch | mixed lifecycle占比 |
| --- | ---: | ---: | ---: |
| prologue | 10,356.50 | 8,649.54 | 10.476% |
| steady | 172,143.62 | 70,478.62 | 85.362% |
| epilogue | 4,572.81 | 3,436.56 | 4.162% |

inter-batch gap为248.19 cycles/gap，占horizon0.277%；N2--N13覆盖完整steady的74.449%。
prologue/epilogue的占比因总时长下降而上升，绝对cycles/batch均下降。

**第6层**：内部窗口10,913,928 cycles = busy7,561,216 + idle3,352,712。

| 类别 | mixed cycles | 3×64 pure steady占比 | 128+64 steady占比 | 变化 / 百分点 |
| --- | ---: | ---: | ---: | ---: |
| MFMA busy | 7,561,216 | 28.005% | **69.280%** | **+41.276** |
| VMEM issue | 618,192 | 3.238% | 5.664% | +2.426 |
| VMEM wait | 58,232 | 21.783% | 0.534% | -21.249 |
| LDS issue | 964,452 | 6.922% | 8.837% | +1.915 |
| LDS wait | 177,588 | 2.953% | 1.627% | -1.326 |
| VALU execution | 495,284 | 13.770% | 4.538% | -9.232 |
| barrier | 951,880 | 22.892% | 8.722% | -14.170 |
| other | 87,084 | 0.438% | 0.798% | +0.360 |

两个版本均为208个batch，内部窗口/busy/idle的每batch周期从
130,721.35 / 36,608 / 94,113.35变为52,470.81 / 36,352 / 16,118.81。
不同core宽度改变两个resident wave的内部N窗口交集端点，故内部busy分子略变；
完整wave仍严格1536条MFMA，交集规则没有变化。

**第7层**：VMEM completion wait已大幅消除，但LDS issue、barrier和输出store仍暴露。
LDS issue由330,100 cycles issue-stall加634,352 cycles正常service构成，不能都叫bank conflict。
最大joint为`LDS issue@tail + barrier@tail`，占steady1.442%；其余主要见证在`core1->2`及tail。
短BK64 core内部可以连续计算，但128/64长度不对称与每N固定输出工作仍在阶段交界留下空洞。
这只是目前的归因证据，不把换统计口径或添加无效MFMA作为达标办法。

### 尾段样本与回退实验

[N2/core1尾段](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/analysis.json#L66)
位于SE0/CU1/SIMD0/slot0，ordinal224--239、PC index1169--1187，对应
[`mma()`历史调用](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/ui_output_agent_41716_dispatch_22/source_2_gemm2_8x1_k192.py#L294)。
窗口 **[89648,89904)** 恰好16条MFMA、256 cycles，physical busy256、idle0。
其raw MFMA stall合计168 cycles全部被union覆盖，不能累加成physical idle；
这个短stage的100%不替代完整N2--N13的69.280%。

另试两种退休安排，随机参考均通过，但短ABBA4不优于首版，已撤回：

| 候选 | LDS | 短测时延 / 有效TFLOPS | 处理 |
| --- | ---: | --- | --- |
| [首版四stage各一quarter](../../../contrib/moe/results/multik_busy80/mixed_k192_v1_abba4.json#L1) | 48KiB | 0.674543 ms / 305.63 | 保留，另做上面的ABBA24 |
| [独占scratch，读回跨compute延迟store](../../../contrib/moe/results/multik_busy80/readpipe_k192_v2_abba4.json#L1) | 64KiB | 0.708324 ms / 291.05 | 撤回，未采ATT |
| [仅BK64 memory阶段各退休两个row-pair](../../../contrib/moe/results/multik_busy80/tailretire_k192_v3_abba4.json#L1) | 48KiB | 0.708943 ms / 290.80 | 撤回，未采ATT |

三行是不同短测轮次，均对同一pure控制，不合并成正式相邻版本ABBA结论。当前独立源码
已恢复到首版，SHA256与首版ATT快照逐字节一致。

### 正确性、资源与身份

- 7组[输出契约检查](../../../contrib/moe/results/multik_busy80/mixed_k192_final_contract_n2048_b257_p128_r1.json#L1)：
  N128/B1/padding0、N128/B33/padding128、N256/B257/padding32/64、N2048/B257/padding128，
  加上N128/N256的rolling禁用；relative-L2为0.003304--0.003457，全部与3×64 pure逐bit一致，
  padding/inactive tail的NaN哨兵未被写入。
- 该轮默认七K加显式混合K192端到端用例为 **8 passed**；当时用缓存隔离确认真正传入128。
  清理后默认K192已覆盖混合实现，重复用例删除，改用[分流与旧参数拒绝测试](../../../contrib/moe/test_moe.py#L1380-L1420)；本轮结果见下一节。
  [分析器CPU测试](test_analyze_mfma_stall.py)为 **14 passed**，包含四stage的前缀边界。
- 编译器 **184 VGPR、96 next-free SGPR、48KiB LDS、0 private/scratch**；ATT为
  56 regular + 128 accum VGPR、112 SGPR，实际2 waves/SIMD。原3×64为171 VGPR、32KiB LDS。
- 每wave1536 MFMA不变，静态`s_barrier` **197→133**。生产TOPK8非debug与ATT的6751条
  指令逐行一致；清理前默认K192/K320/K512/K640重新编译后的8874/8465/10877/12647条
  指令分别与当时各自版本一致。其中8874对应旧3×64默认，不是当前混合默认的指令数。

该测试轮、清理前入口源码SHA256为`f38fa69a07c06b9697f2e0f4f694cf276ff4d9fa5bebe754326f58753da3a51e`；
[独立builder快照](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/ui_output_agent_41716_dispatch_22/source_2_gemm2_8x1_k192.py#L1)
为`b11e7cdaf3f69ded9e62a0472493a1bd7780577b1a98bb22d5c2b2026fa29263`；debug ISA为
`199bb25bdc3d654b549f8a1bb305d74dc3bb8750065e51a06be9096f910c6098`；
[code表](../../../contrib/moe/results/multik_busy80/mixed_k192_v1/ui_output_agent_41716_dispatch_22/code.json#L1)
为`b85fe1069a14921877525a1b524f5a358ef351d85d93bd5b823a6a861f6170db`。
运行器记录入口和独立builder两个哈希。

单次ATT dispatch22为0.599162 ms / 344.08有效TFLOPS，仅标识trace，不能替代正式ABBA24的
0.672143 ms / 306.72有效TFLOPS；不把union乘roof当成墙钟吞吐。

### K192实验复现

沿用K512节的`REPO/PY/TOOLS/RESULTS`和运行环境，显式候选`--tile-k 128`、控制`--control-tile-k 64`。
当前省略`--tile-k`同样选择128+64；3×64只能通过下面的历史控制快照复现。
重新运行前核对入口及依赖源码哈希；ATT输出目录必须全新。

```bash
CONTROL192="$RESULTS/current_k192/ui_output_agent_27246_dispatch_22/source_2_gemm2_8x1.py"
export MOE_8X1_ROLLING_EPILOGUE=1
"$PY" "$TOOLS/run_moe_8x1.py" --mode check --k 192 --tile-k 128 --control-tile-k 64 \
  --n 2048 --batch 257 --experts 8 --topk 4 --random-routing \
  --control "$CONTROL192" --output /tmp/moe8x1-k192-mixed-recheck.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode bench --k 192 --tile-k 128 --control-tile-k 64 \
  --rounds 24 --managed --control "$CONTROL192" --output /tmp/moe8x1-k192-mixed-rebench.json
"$PY" "$TOOLS/run_moe_8x1.py" --mode att --k 192 --tile-k 128 --managed \
  --output /tmp/moe8x1-k192-mixed-recapture

PYTHONPATH=/usr/local/lib/python3.10/dist-packages "$PY" "$TOOLS/analyze_mfma_stall.py" \
  "$RESULTS/mixed_k192_v1/ui_output_agent_41716_dispatch_22" \
  --n-blocks 16 --cores-per-n 4 --core-mfma-counts 32 16 32 16 \
  --first-n 2 --last-n-exclusive 14 \
  --launch-workgroups 1280 --active-workgroups 1024 \
  --cu-count 80 --waves-per-wg 8 --simds-per-cu 4 --resident-waves 2 \
  --stage-wave se0_sm0_sl0_wv0.json --stage-n 2 --stage-core 1 \
  --json /tmp/moe8x1-k192-mixed-reanalysis.json
```

<a id="bk64-cleanup"></a>

## 3×64/5×64与K128遗留分支清理验收（2026-09-05）

第一轮只删除3×64/5×64并做正确性与ISA验证，**该轮没有重新测性能或采ATT**。
[验证汇总](../../../contrib/moe/results/multik_busy80/bk64_cleanup_validation.json#L1)保存精确源码身份、
逐指令哈希、资源、测试和硬件状态。随后按要求清理K128死代码与恒真条件、补per-tensor并重测矩阵，
新一轮结果见[当前汇总](../../../contrib/moe/results/8x1_final_matrix/summary.json)。前面各节ATT仍归属于历史测量。

### 已删除与当前参数契约

- 删除K192的3×64和K320的5×64通用实现，包括`DEDICATED_K320`、BK64条件加载/提交、
  专属Uint32/64-bit staging、BK64 A/B索引、`pack_k320_rolling()`及单store流水状态。
  通用内核的`BLOCK_K`现在恒为128；**混合实现中真实64尾段的搬运与MFMA保留**，没有K-padding。
- [入口](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L60-L81)只接受`tile_k=None/128`，
  K192默认128+64、K320默认128+128+64；显式64抛出说明旧路径已移除的`AssertionError`，不静默换算法。
- [运行器](run_moe_8x1.py#L197-L217)默认也统一为128；CLI的64选项继续服务于历史`--source/--control`快照，
  不是保留当前kernel的旧分支。历史JSON、raw ATT和源码快照均未删除。
- 第一轮主builder由1815行降至1685行：删除197行、增加67行，**净删130行**。
  清理前SHA为`f38fa69a…`，清理后为`5f8f3dbca9252b3fff8b5c7630b0b85256195dfe967947bdeef91eef14b69750`。
  该统计以同SHA清理前源码为基准，不用包含其他历史改动的HEAD差异冒充本轮改动量。
  当时两份独立混合builder的SHA分别为`b11e7cda…`和`e17234c0…`；第二轮量化修改后身份见当前汇总。

### 第一轮清理回归（历史）

当时七K端到端7例，加上默认/显式128、rolling开/关的混合分流8例和旧64拒绝4例，
共 **19 passed，28.34s**；分析器CPU回归 **14 passed，0.20s**。当前[测试集](../../../contrib/moe/test_moe.py)
已扩展量化维度，不能拿历史19项覆盖面代替下面的新验收。

额外使用默认入口、N256/B257/E8/TOPK4/padding128B及随机路由做4组独立FP32参考：

| 默认路径 | rolling关 / 开的结果 | relative-L2 | 对历史BK64 |
| --- | --- | ---: | --- |
| K192，128+64 | [关](../../../contrib/moe/results/multik_busy80/bk64_cleanup_default_k192_r0_contract.json#L1) / [开](../../../contrib/moe/results/multik_busy80/bk64_cleanup_default_k192_r1_contract.json#L1) | 0.00330369 | 均逐bit相同 |
| K320，128+128+64 | [关](../../../contrib/moe/results/multik_busy80/bk64_cleanup_default_k320_r0_contract.json#L1) / [开](../../../contrib/moe/results/multik_busy80/bk64_cleanup_default_k320_r1_contract.json#L1) | 0.00329561 | 均逐bit相同 |

四组padding及inactive tail的NaN哨兵均保持未写。pure回退因此仍是可用路径，不属于死代码。

生产shape固定B32768/N2048/E256/TOPK8/padding128B，缓存关闭、无debug-info，重新JIT并执行：

| K | rolling开指令数 | rolling关指令数 | 对比基准与结果 |
| ---: | ---: | ---: | --- |
| 128 | 4858 | 5583 | 清理前后两条路径逐指令、资源相同 |
| 256 | 7437 | 7368 | 同上 |
| 384 | 9167 | 9104 | 同上 |
| 512 | 10877 | 10798 | 同上 |
| 640 | 12647 | 12537 | 同上 |
| 192 | 6751 | — | 新默认与原已验证128+64 rolling逐指令、资源相同 |
| 320 | 8465 | — | 新默认与原已验证128+128+64 rolling逐指令、资源相同 |

比较保留操作数和branch target，只规范空白；同时核对LDS、private、next-free VGPR/SGPR、
accum offset五个资源字段。K192/K320的rolling关闭路径用上面的随机contract实测，不将其误报为生产ISA对比。
第一轮未接管硬件；结束只读核验GPU7为auto、650W、PTL Disabled/N/A、NUMA balancing=1。

### 第二轮：已清理K128遗留与恒真条件

根因是[入口提前分流](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L82-L115)：
K128且rolling开启时已返回独立builder；关闭时通用`ROLLING_EPILOGUE=False`。
在一次构建期间环境配置固定的正常调用契约下，旧通用内核里的`K == 128 && ROLLING_EPILOGUE`永不成立。

| 已完成清理 | 保留行为 |
| --- | --- |
| 删除K128专属scale/pack、rolling输出、drain，以及`write_packed_super_record_k128()`和`read_packed_half_k128()` | K384/K256/K512/K640退休时序不变 |
| 移除8-wave CShuffle、joined-fragment、非空`scratch_half`、K128输出映射与4-request特例 | [当前CShuffle读回/存储](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L870-L950)只保留可达实现 |
| 简化各标志的恒真`BLOCK_K == 128`，`SPECIALIZED_ROLLING`只保留K384 | [当前专用标志](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L93-L114)不删除实际K特化 |
| 删除K192 q2预取恒真保护（target=1、NT≥1，所以`1 < 2*NT`） | [实际预取](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1_k192.py#L338-L344)保留；K320的单N边界仍有意义，未删 |

清理步骤单独重新编译12条生产路径，指令与资源全部不变；随后新增量化，再对这12条PTPC路径
重复验证，仍完全一致。通用K128 pure与独立K128继续保留；没有删除有效实现来凑代码简化。

### 第二轮：per-tensor量化与完整链矩阵

- 七K均支持PTPC/PTPC、per-tensor/per-tensor、per-tensor权重＋PTPC activation。
  per-tensor权重每expert一个scale，activation全局一个，前置与routing融合；
  不再每N加载channel scale，相应vmcnt额度归零。rolling pack的40-VALU变为24-VALU配额。
- [随机输出契约](test_moe_8x1_quant.py)56项通过；七K端到端/分流42项，API/统计/分析器41项，合计139项。
  scalar版本七K及Hy3真实生产shape均finite、0 scratch；详细资源与源码哈希见
  [当前汇总](../../../contrib/moe/results/8x1_final_matrix/summary.json)。
- 完整性能矩阵重测42点ABBA12并以14点ABBA48覆盖边界。最终8x1进入Qwen35 K512/K256的
  16K/32K及H3 8K--32K；Hy3、Xiaomi及8K端到端持平点按保守规则保留原路径。
  完整wall-time、有效TFLOPS、IQR与原始样本只维护在[最终性能报告](../../../contrib/moe/MAIN_MERGE_PERFORMANCE_REPORT.md)。
- 本轮未采per-tensor ATT，不更新前面的PTPC busy；managed性能轮次均记录并恢复PTL/DPM/NUMA。

### 未消费结果与不能误删的路径

- [两个quarter-load结果](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1346-L1376)只在
  [`ROLLING_EPILOGUE and block_n > 0`](../../../../src/contrib/flydsl/moe_gemm_2stage/gemm2_8x1.py#L1424-L1436)消费；
  pure/首N改用full-half fragment。已证明这些场景下结果未消费，**尚未证明会残留额外ISA访存**，
  不能跳过DCE/JIT核验就宣称性能收益，也不能整段删除rolling稳态需要的quarter load。
- pure开关、单N/跨N预取边界、K256/K384/K512/K640有效时序，以及混合64尾段都保留。
  K192曾测试的readpipe_v2/tailretire_v3已撤回，不是当前还要删除的分支；K128/K192未达80%也不等于无用。

## 什么是rolling epilogue与按K特化

这里的epilogue是矩阵累加完成后的输出处理：权重scale、activation/routing scale、
BF16打包、CShuffle重排、global store。这里的“退休”表示某个输出分片已完成上述处理，
不再需要其累加器；不是另一次GEMM，也不是split-K归约。

- **Pure epilogue（集中退休）**：算完一个N tile全部K贡献，再集中调用
  `retire_output()`处理整个tile。即使B预取已有流水，pack/scale/重排/store仍可能造成
  一段连续的非MFMA时间。
- **Rolling epilogue（分片滚动退休）**：某一输出分片的最后一个K贡献完成后，便把它的
  缩放/打包穿插到其他分片的独立MFMA中；CShuffle读写及store再分布到后续memory stage。
  “上一N的输出”与“当前N的计算”可以同时处于流水中，最后单独drain尚未退休的分片。

以K256为例，每N依次计算`K0L → K1L → K0H → K1H`（L/H是N的低/高半区）：

| 计算位置 | 累加工作 | 交织的输出打包 |
| --- | --- | --- |
| `N=n, K1L`后半packet | 完成低半区的最后K贡献 | 打包当前N的SR0 |
| `N=n, K0H`前半packet | 开始高半区累加 | 打包已经完成的低半区SR1 |
| `N=n, K1H`后半packet | 完成高半区的最后K贡献 | 打包当前N的SR2 |
| `N=n+1, K0L`前半packet | 计算下一N的低半区 | 打包上一N的SR3 |

SR是一个N32输出分片。下一N的四个memory stage逐次写出上一N的四个
`N64 × row-pair`输出块；这些store块与SR打包分片不是同一个划分。
两组4-wave反相，让memory工作有机会被peer的MFMA覆盖，但重叠必须由ATT union验证。

**“按K特化”是编译期选择一张满足依赖的时序表**：何时加载scale、何时一个分片已完成
所有K累加、何时打包/store、何时可覆盖C或LDS，以及prologue/drain如何补齐。它不改变
数学结果或输出布局，不增加无效MFMA，也不代表拆成多个GPU kernel。
K256是4×32、K384是6×32、K512是8×32，而当前K192是32/16/32/16（旧3×64为6×16）；
可隐藏输出处理的计算窗口长度不同，不能只把`ROLLING_EPILOGUE`改成True而照搬K256的硬编码退休点。
当前K256已有rolling，K384候选也已实现；K512专用八core rolling已通过上面的终验，
K320混合分块也已通过两轮ATT与正式ABBA24，K640十core rolling已通过上述终验。
K192默认128+64已通过数值和性能测试，但busy仅69.280%；旧3×64/5×64只保留历史快照。
K128独立实验尚未达到80%，本次未继续调整。
B的ping/pong与跨stage预取解决的是**输入供给**，rolling epilogue解决的是**输出退休**，
二者互补，不能混称为同一种优化，也不能保证任何K仅靠rolling就超过80% busy。

## 标准工作流

### 0. 固定问题和控制变量

在采集前记录：

- commit、kernel SHA256和候选patch；
- GPU、gfx、ROCm/FlyDSL版本、80 CU与power cap；
- shape：`B/TOPK/E/N/K/BM`、量化、padding和metadata排序单位；
- tile、threads/WG、waves/WG、TiledMMA结构；
- 初始GPU idle状态、1800MHz determinism、PTL `Enabled / VECTOR,F8`；
- control与candidate是否使用同一数学输出契约。

若PTL、外部负载或metadata不同，先修复实验，不分析绝对时间。

### 1. 正确性与ISA资源门禁

ATT前至少确认：

- valid physical rows、padding和inactive tail；
- reduced row-major输出、finite检查和`rel_l2`；
- MFMA/load/store/barrier数量符合算法；
- VGPR/AGPR、LDS、private/scratch、实际waves/SIMD；
- 没有意外spill或跨occupancy台阶。

若候选跨越VGPR/LDS occupancy门槛，后续stall分布已不是同资源实验，必须单独解释。

### 2. 先用clean ABBA确定是否值得解释

正式协议使用10 rotating buffers、24轮ABBA；每版本有48个绝对样本和24个paired ratio。报告：

```text
control ms -> candidate ms
candidate/control median
ratio Q1..Q3
wins/rounds
```

短ABBA只淘汰明显回退。外部任务占用时只可将同进程ratio标成stress证据，不能替代clean结果；idle gate拒绝运行是正确行为。

### 3. 采集或选择raw ATT

后处理的输入目录必须至少包含`code.json`和完整的
`se<SE>_sm<SIMD>_sl<SLOT>_wv<WAVE>.json`。每个候选使用新目录，禁止用旧trace解释新ISA。
先记录kernel名、dispatch、源码/ISA/code哈希、shape、GPU/PTL/DPM和外部负载，再分析。

仓库保留了K256及多K实验的raw trace，因此**从raw trace到报告可独立复现**。
同目录的[run_moe_8x1.py](run_moe_8x1.py)负责down-only launch、随机reference、ABBA与
fresh ATT；重新采集仍需要本机ROCm/FlyDSL/AITER和ATT decoder，不由后处理脚本安装。
采集时使用的ATT配置为：

```yaml
kernel_include_regex: "^moe_2stage_down_prefill_8x1_0$"
kernel_iteration_range: "[2]"
advanced_thread_trace: true
att_target_cu: 1
att_shader_engine_mask: "0xf"
att_simd_select: "0xf"
att_buffer_size: "0x60000000"
```

采集前必须通过全机idle gate，GPU7固定1800MHz determinism、PTL
`Enabled / VECTOR,F8`、650W、NUMA off；无论成功与否都恢复原状态。先用kernel trace
确认匹配序号，不能假设`[2]`对不同launch脚本仍指向同一dispatch。

### 4. 运行唯一分析器

使用上文的[统一复算命令](#唯一复算命令)。参数必须来自实际geometry，不能照抄K256：

- 均匀geometry的`n-blocks * cores-per-n * mfma-per-core`必须等于每条active wave的动态MFMA数；
- 非均匀geometry使用互斥参数`--core-mfma-counts`，按真实执行顺序给出每个core的数量；
  `n-blocks * sum(core-mfma-counts)`必须等于每条active wave的动态MFMA数，core边界由前缀和确定；
- `resident-waves`必须与资源和trace中的同时active slot数一致；
- `first-n..last-n-exclusive`应排除首尾非稳态块，并报告其steady覆盖率；
- control和candidate须使用同一真实工作量及steady窗口定义；各自geometry必须匹配其代码。
  混合K320的6×非均匀core与5×64的10×16不同，不能为了比较强行套成同一core划分。

脚本会强制检查active wave MFMA数、lifecycle闭合、union busy/idle闭合、七类owner闭合和
类别子项闭合。任何断言失败都意味着模型不适用或输入不完整，不能继续解读百分比。

### 5. 用joint phase定位可改代码

owner只回答“谁占了空槽”，joint state回答“哪些wave阶段同时发生”。对热点PC检查：

1. 它是请求发射、completion wait还是普通issue？
2. peer waves处于`core0/core1/core2/boundary/tail`哪一阶段？
3. producer到first consumer还有多少successful-issue距离？
4. 是否所有producer边都已被计数器顺序有效的wait覆盖？
5. 该PC能否移动而不增加指令、wait、寄存器或barrier？

只有在单条记录与union idle求交后，热点PC才可进入候选列表。先区分service与真正stall，
再结合peer phase判断是请求同相、completion latency、依赖链还是结构性tail。

### 6. 只做一个可证伪改动

候选应直接对应一个owner假设：

| 假设 | 最小候选 | 可证伪检查 |
| --- | --- | --- |
| 两slot VMEM同相 | slot/role priority或固定phase rotation | same-reason VMEM下降，wait不反弹 |
| DS read消费距离不足 | 在read与wait间放独立pack/write | lgkmcnt owner下降，指令数不增 |
| CShuffle tail过长 | 单slice分片退休probe | 下一N首批MFMA真实进入旧tail |
| occupancy是主因 | 仅跨一个VGPR/LDS门槛 | resident waves增加且墙钟改善 |

禁止一次同时改tile、priority、store policy和padding；否则ATT即使变好也无法归因。

### 7. 重新执行四层闭环

保留候选必须同时满足：

1. 正确性不退化，finite/tail/padding全部通过。
2. ISA工作量与资源变化已解释，没有意外spill。
3. clean ABBA24的ratio/IQR/wins稳定改善。
4. fresh ATT中目标owner、physical union busy和墙钟方向一致。

若owner下降但union/墙钟不变，结论是“暴露转移”，不是成功。若墙钟改善但ATT未解释，先检查trace dispatch、PTL、代码SHA和steady窗口，不补故事。

## 常见失误清单

- 用`first_attempt`画MFMA window。
- 用`code[pc_index - 1]`映射ISA。
- 把所有wave的stall cycles直接求和。
- 把`duration`或4-cycle trace issue cost当MFMA initiation interval。
- 把normal DS/VMEM issue当stall。
- 把VMEM issue stall与`vmcnt` completion wait合并。
- 只看steady busy，不看包含prologue/drain/tail的lifecycle busy。
- 把owner、same-reason witness和oracle上界重复相加。
- 目标owner下降后不报告转移到哪里。
- 用旧trace解释新ISA，或复用旧output目录。
- 采集时不记录PTL、DPM、外部负载和源码SHA。
- 在模型不闭合时继续解读百分比。

## 文件职责

- 本文：唯一方法、复现命令、K256案例、多K基线、K320/K512/K640终验及K192混合分块测试。
- [analyze_mfma_stall.py](analyze_mfma_stall.py)：唯一后处理实现。
- [test_analyze_mfma_stall.py](test_analyze_mfma_stall.py)：均匀/非均匀geometry的CPU回归测试。
- [run_moe_8x1.py](run_moe_8x1.py)：统一down-only reference/ABBA/ATT工作负载及硬件门禁与恢复。
- `ui_output_moe_8x1_k256_current_dispatch_16/`：raw trace及脚本生成的唯一JSON账本。
- [design_moe_gemm_8wave_down.md](../../../../design_moe_gemm_8wave_down.md)：kernel设计与性能演进；不再复制ATT账本。
