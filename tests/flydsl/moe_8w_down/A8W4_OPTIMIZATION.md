# A8W4 MoE down 优化记录

**范围：2026-09-14～15，gfx950 / AMD Instinct MI350X。**
本文按优化机制整合历次实验，关键数据、比较条件和失败边界均在正文中，不依赖外部结果文件。
表格不是不同轮次的拼接排名；除明确标注外，时间单位为 **μs**，`Down / Full`表示两项独立计时。

## 1. 先看结论与当前状态

1. **长K先解决寄存器溢出。** 提前打包已完成的N64，能显著缩短FP32结果生存期；K384收益大，K256原本不溢出时收益很小。
2. **连续写出不必改变输出格式。** DPP将同一行的单次连续覆盖从64B扩大到128B，仍直接写routed BF16，不需要inverse或restore。
3. **cache策略必须跟布局一起测。** 64B写出下NT+SC1经常拖慢Down；128B下多数中大batch受益，不能沿用旧布局结论。
4. **更多独立CTA有代价。** M128四wave可提高驻留和独立推进机会，但增加任务数及B请求；不是所有batch都胜过M256八wave。
5. **swizzle没有固定赢家。** 32K四wave从64B改为128B写出后，最快分片任务顺序由width8变为无转置。
6. **最终看Full。** 专用归约在4K～16K的warm带宽提升约11～13%，但优化内核的Full通常只改善约1%；32K warm归约甚至变慢，Full却仍受益。

**验证限制：K512曾未通过原基线的模型参考门槛；后续与原route精确一致不代表这个问题已解决。** 具体条件见第10节。

### 当前实现与默认测试

| 路径 | 核心组织 | 写出 / 归约 | 当前用途 |
|---|---|---|---|
| 原八wave | M256、4+4错相、全局队列 | 原routed写出 | 保留对照，不改模型dispatch |
| 独立四wave | M128、OC4、512 workers、paired publication | routed DPP 128B；aux0/18 | 简洁优化实现，允许BN64/128及分片 |
| 优化八wave | M256、OC4、256 workers、原4+4错相 | DPP 128B、aux18、compute-overlap | 单独实现，保留各因素开关 |
| routed专用归约 | 256线程、2048列/CTA、每lane8个BF16 | NT读、普通写，Torch兼容累加顺序 | 当前测试默认归约 |

[测试入口](test_a8w4.py)默认候选为`aiter`、`8wave`、`4wave_bn64`、`8wave_optimized`，默认归约为`moe`。
默认形状：N6144、K256、E384、TOPK8、tokens=4096/8192/16384/32768。
**测试默认候选不等于模型dispatch，也不等于独立工厂的默认参数。**

- 四wave工厂默认BN128/aux0/全局队列；`4wave_bn64`候选显式改BN64。
  `select_config`仅在已测N6144/TOPK8/K256或384系列中，tokens≥8192选择aux18；K256且tokens≥32768选择分片。
  其他形状保留aux0/全局队列；BN128超过半个160KiB LDS预算或不能形成配对时选BN64。
- 八wave优化工厂默认BN128/OC4/DPP/aux18/overlap；没有移植四wave的paired同步或分片队列。
- 原`8wave`候选保留旧测试的OC选择：小于16K用OC4，16K起用OC1；`8wave_oc4`固定OC4。
- Aiter用`return_per_slot=True`写共享routed缓冲，再接相同归约；这不是它默认分配中间结果并使用自身reducer的API耗时。
  Opus仍为显式K384、Full-only对照，使用自身归约。

## 2. 数据契约与测量规则

### 数值与布局

- A：FP8 `[tokens, TOPK, K]`；B：Aiter GUI预重排的packed FP4 `[E, N, K/2]`，每值4bit。
- A/B scale：每1×32一个E8M0。使用原生`mfma_scale_f32_16x16x128_f8f6f4`，保留每个accumulator的K累积顺序及scale opsel。
- 每route最后执行routing乘法、BF16 RNE，再归约；不能把未舍入FP32结果提前跨route合并。
- routed中间值为`[tokens, TOPK, N]`。早期packed实验改为`[M_block, global_N64, row_in_block, col64]`，必须配套inverse和gather归约。
- M128与M256必须分别原生sort128/sort256，并重新排列A scale；不能只改计算M而沿用旧排序解释收益。

### 公平比较

- 环境为Python3.10、FlyDSL0.3.2、Torch2.9.1 ROCm7.2，GPU0运行前检查空闲，**未锁频**。
- 同一对照使用相同A/B/scales、路由语义、中间与最终输出地址；量化、排序和参考计算在计时外。
- Down包含必要queue reset一次；Full直接测Down＋消费者，packed还包含inverse fill/build。输出预分配，不把组件之和当Full。
- 逐因素建立parent。重构后的控制实现先与旧实现比较；改变多个机制时只称为组合收益，不能相乘各轮加速比。
- 筛选一般5轮×20次，确认一般7轮×30次，部分9轮×40次；预热2次、copies=1，交替/反转顺序，主要结论用seed1234和2026复测。
- 模型参考：误差超过`1 + 0.05*abs(ref)`的元素比例≤5%；Opus FP8额外量化沿用10%。非有限值直接失败。
- 本地优化相对原BF16 route零容差。旧packed归约对显式slot顺序FP32和比较；当前routed归约对Torch顺序精确比较，两者不能混称相同语义。
- 普通计时、ISA资源、PMC/ATT分开解释。静态MFMA条数不等于动态工作量；warm reduce不代表紧接Down的缓存状态。

**时间线口径：第3～8节的Full均使用当轮Torch归约，早期packed条目除外；第9节单独比较并启用专用归约。**
历史结论保留当轮条件，不把后续接口或消费者变化套回旧数字；未重跑时不宣称旧测量对应最新源码。

## 3. 第一轮：单因素筛选

实现为[独立实验工厂](experiments/moe_8wave_down_a8w4_experiments.py)，非源码字符串替换或运行时patch。
基线M256/8wave/BN128/OC1已具备A寄存器复用、B直入LDS、四槽ring、4+4错相和BF16/permlane包装，不能将这些列为新增。

K256、16K主形状，5轮×20次；packed条目的Full含其inverse＋gather消费者：

| 实验 | 父对照 | Down | Full | 结论 |
|---|---|---:|---:|---|
| 原baseline | — | 515.363 | 929.584 | 原OC1 |
| 原OC2 | baseline | 491.662 | 911.813 | 小幅有效 |
| 原OC4 | baseline | 487.670 | 916.673 | Down略好于OC2，Full不一定 |
| control | baseline | 515.765 | 929.566 | 等价实现控制 |
| OC2 | control | 491.652 | 911.651 | 复现 |
| OC4 | control | 488.762 | 917.787 | 复现 |
| OC8 | control | 537.880 | 980.032 | 过度拆分退化 |
| compute-overlap | control | 517.051 | 930.212 | K256近乎持平 |
| fine-overlap | compute-overlap | 514.441 | 931.140 | 更细粒度pack无稳定增益 |
| task级scale cache | control | 514.321 | 928.778 | 很小收益 |
| 每task独立CTA | control | 602.232 | 1014.915 | 退化 |
| 独立CTA＋width8 | 独立CTA | 513.987 | 926.632 | 修复独立CTA退化，并非比persistent再快17% |
| N相位旋转 | control | 572.987 | 989.092 | 退化 |
| Memory priority3 | control | 518.577 | 929.894 | 无稳定增益 |
| M128、仍8wave | control | 624.933 | 1044.094 | padding减少但更慢 |
| M128、旧4wave错相 | M128 | 628.015 | 1050.272 | 只减wave数不奏效 |
| packed | control | 503.616 | 891.082 | 格式＋消费者整体有效 |
| packed＋DPP | packed | 497.342 | 888.216 | 此形状DPP增益较小 |

其他关键结果：

- **4K/K256**：原OC1为203.829/299.595，OC2为184.860/280.877，OC4为189.383/287.741。
  此处OC1是实验控制，旧测试的小batch本来就用OC4。
- **32K/K256**：原OC1为1094.194/1938.434；OC4为1121.033/1969.161，不能无条件增加OC。
  control为1094.206/1936.646，packed为1070.329/1930.708，packed＋DPP为890.218/1719.782。
  但packed reducer约838，慢于Torch约793；收益主要不能归给归约。
- **scale cache**：32K仅小收益；OC1/K256占116736B LDS，OC1/N6144/K384需198656B，超160KiB而拒绝。
  scale cache、priority3、细粒度pack都不作为主组合的必要项。

<a id="compute-overlap"></a>
## 4. 计算与打包交错：先消除spill

BN128原先对全部N块逐K计算，最后统一打包。`compute_overlap`改为：

1. 完成左N64的全部K。
2. 开始右N64，在其首个K步的MFMA pair之间打包左半已完成结果。
3. 完成右半剩余K，再打包右半。

这是独立N accumulator之间的调度，不改变任何输出的K求和顺序；B DMA/ring、store位置和barrier不变。
早打包让FP32值更早退休：每lane左半32个32位结果，打包后只占16个32位值，但实际寄存器数量仍由编译器决定。

### OC1/M256/8wave资源与时间

| K | VGPR off→on | AGPR off→on | private bytes off→on | VGPR spills off→on | 静态MFMA off/on |
|---:|---:|---:|---:|---:|---:|
| 128 | 152→138 | 24→10 | 0→0 | 0→0 | 64/64 |
| 256 | 232→201 | 104→73 | 0→0 | 0→0 | 128/128 |
| 384 | 256→256 | 128→128 | 108→0 | 34→0 | 192/192 |
| 512 | 256→256 | 128→128 | 248→104 | 93→25 | 256/256 |

16K确认轮：

| K | Down off→on | Full off→on |
|---:|---:|---:|
| 256 | 516.739→516.818 | 922.331→923.700 |
| 384 | 870.554→577.269 | 1300.294→996.636 |
| 512 | 1228.429→650.133 | 1642.145→1081.956 |

K384消除spill，K512只减少spill；LDS没有变，不能解释成多驻留CTA。无spill的K256降低寄存器却基本不提速。
81组BN128代表性形状覆盖K128～512、OC1/2/4/8、M128/256、4/8wave、routed/packed及短任务，精确route/Full和graph通过。
最终矩阵未见超过2%的退化，但以下两个Down小幅回退可复现：

- 4K/N7168/K128/E64/TOPK2/M256/OC1：104.815→105.974，慢1.11%。
- 4K/N3072/K384/E64/TOPK2/M256/OC8，最短三packet：40.331→40.843，慢1.27%。

因此优先在spill瓶颈使用，不宣称任何形状都受益；BN64没有左右两个N64，不能直接开启这个方案。
旧K512精确对照不等于后续所有K512模型参考都通过，见第10节。

<a id="four-wave"></a>
## 5. 独立四wave CTA：驻留与任务开销的权衡

阶段路径是M256/8wave → 原生M128/8wave → M128/旧4wave错相 → uniform四wave → 512 workers → paired → 分片/转置。
**这一阶段仍为64B/行写出，没有DPP或NT+SC1。**

uniform四wave不再在CTA内部2+2错相；不同CTA可独立推进，但不握手、不假设固定相位。
paired一次发布两包、减少重复publication barrier；仍等待DMA与LDS读完成，并在任务结束后退休资源。
A8W4每N行scale不同，各包保留自己的scale，不能复制blockscale的N128 scale复用。

### 驻留资格与同机器码消融

| K256配置 | threads | static LDS B | dynamic LDS B | HIP最大CTA/CU | wave/SIMD上限 |
|---|---:|---:|---:|---:|---:|
| M256/8wave | 512 | 71680 | 0 | 1 | 2 |
| M128/4wave BN128 | 256 | 70660 | 0 | 2 | 2 |
| 同上，动态LDS限驻留 | 256 | 70660 | 20480 | 1 | 1 |
| M128/4wave BN64 | 256 | 35844 | 0 | 4 | 4 |

HIP查询是资格上限，不是实测同时驻留。1个8wave CTA变2个4wave CTA，总wave数可以不变，关键是同步/任务生命周期独立。
动态增加20KiB LDS的对照GPU指令相同；静态90KiB对照改变寄存器分配，不能当纯驻留证据。

| Tokens | 限制1CTA的Down | 允许2CTA的Down |
|---:|---:|---:|
| 4096 | 182.225 | 189.541 |
| 8192 | 314.447 | 304.041 |
| 16384 | 574.081 | 528.619 |
| 32768 | 1200.023 | 1023.769 |

大batch受益、小batch反而退化；没有ATT证明差值全部来自prologue/epilogue与body重叠。

### 64B阶段性能

| Tokens | 原八wave OC4 D/F | uniform512 D/F | paired512 D/F | 分片＋width8 D/F |
|---:|---:|---:|---:|---:|
| 4096 | 189.133 / 287.982 | 185.717 / 286.897 | 189.541 / 287.088 | 191.407 / 285.788 |
| 8192 | 291.133 / 499.452 | 305.817 / 515.179 | 304.041 / 512.827 | 294.835 / 502.712 |
| 16384 | 487.138 / 911.225 | 531.209 / 953.022 | 528.619 / 951.696 | 515.183 / 940.672 |
| 32768 | 1119.186 / 1963.347 | 1072.927 / 1932.798 | 1023.769 / 1894.292 | 962.413 / 1819.811 |

- 16K原生M128的padded行196608→147968，但M块768→1156，按M块计的B请求增加50.5%；更少padding不保证更快。
- 保持M256只改四wave，16K为936.136/1356.633，寄存器压力使其明显退化。
- K384/BN128占107524B LDS，只允许1CTA；BN64占54276B，允许3CTA。
  此阶段16K四waveBN64为584.073/1009.964，仍慢于已有八waveoverlap的546.234/972.457。
- 分片用`h=worker%8, task=8*atomicAdd(head[h],1)+h`，覆盖正确性不依赖物理XCD归属。
  每shard终值为`max(0,ceil((tasks-h)/8))+workers/8`；未测placement时，不宣称已绑定XCD。

<a id="coalescing"></a>
## 6. 写出布局与NT/SC1必须一起评估

### 6.1 为什么64B和128B不同

固定一条wave store，原布局同一行由lane `r,r+16,r+32,r+48`各写16B，起点为0/32/16/48B，合计连续64B。
DPP交换行低位与BF16列的bit5后，同一目标行由8lane各写16B，覆盖128B。
全wave仍写1024B：由16行×64B变8行×128B；BN128每行总计256B是多条store的和。

令`lr=lane%16`、`lk=lane//16`、`swap=(lk&1)*2+(lk>>1)`，`pair`为每lane的8BF16片段编号：

- 旧行：`wave_row + mi*16 + lr`；列：`n_base + pair*32 + swap*8`。
- 新行：`wave_row + mi*16 + (lr//2)*2 + pair%2`；列：`n_base + (pair//2)*64 + swap*8 + (lr%2)*32`。

先完成route乘法和BF16舍入，再做DPP；目标route和有效性必须同步重排。
无效目标落到buffer真正OOB位置抑制写入，无需额外可写sentinel行。
**128B是逻辑连续覆盖，不是证明一个128B物理事务或HBM字节减半。**

### 6.2 先前64B布局下的cache实验

aux0=默认，aux2=NT，aux16=SC1，aux18=两者。原Tensor赋值与显式copy可能生成不同代码，
因此保留`original → aux0 → aux18`三路，纯cache收益只比较后两者。
对每个实际计时ELF，去掉C store的`sc1 nt`后完整指令一致，才把差异归为cache标志。

64B阶段共16版本：4K全部Down/Full退化；16K/32K全部Down退化。
8K `sharded512`的Full有最大稳定相对收益，但它并非绝对最快：

| 配置 / seed | Down aux0→18 | Full aux0→18 |
|---|---:|---:|
| 8K sharded512 / 1234 | 346.146→335.684 | 559.879→515.077 |
| 8K sharded512 / 2026 | 317.100→325.237 | 536.252→508.223 |
| 8K sharded_w4 / 1234 | 295.833→305.392 | 503.078→496.588 |
| 16K paired512 / 1234 | 531.677→548.979 | 958.102→941.202 |
| 32K sharded_w8 / 1234 | 958.480→1009.543 | 1828.504→1814.805 |

NT与SC1不是线性叠加。16K paired512的aux0/2/16/18，Down分别531.677/699.904/567.891/548.979，
Full分别958.102/1086.334/989.157/941.202。单独NT或SC1均更慢，两者组合仅Full获益。
K384同轮也无可取的整体赢家；这些都是**64B阶段**结论，不能否定后续128B组合。

### 6.3 四组对照：64/128B × aux0/18

以下为四wave确认轮seed1234，7轮×30次；同一行只改写出重排或cache标志，仍用Torch归约：

| Tokens / 配置 | 64B aux0 D/F | 128B aux0 D/F | 64B aux18 D/F | 128B aux18 D/F |
|---|---:|---:|---:|---:|
| 4K paired_dyn1cta | 182.397 / 280.321 | 178.500 / 275.339 | 188.291 / 282.186 | 176.882 / 271.957 |
| 8K paired512 | 303.383 / 510.844 | 283.152 / 484.456 | 313.059 / 504.135 | 278.805 / 470.478 |
| 16K paired512 | 529.730 / 954.452 | 473.285 / 881.548 | 547.232 / 937.898 | 464.795 / 854.177 |
| 32K sharded512 | 1014.492 / 1868.849 | 850.188 / 1677.348 | 1031.192 / 1857.334 | 846.704 / 1631.174 |

128B下，paired512的aux0→18使Full再加速约3%；第二seed复现主要收益。
此阶段较快配置的Full，seed1234/2026：4K动态限1CTA为271.957/272.458；8K BN64为470.301/469.351；
16K BN64为852.951/853.803；32K无转置分片为1631.174/1631.746。
8K/16K的paired、priority3和BN64差距很小，不把<1%差异当作稳定唯一赢家。

资源查询中，K256 paired/sharded的寄存器210→233，仍允许2CTA/CU、无scratch；BN64为114→116、仍允许4CTA。
K384 BN64的16K为：64B aux0 584.149/1012.563，128B aux0 528.927/942.623，128B aux18 525.768/915.845；第二seed aux18 Full915.592。

**必须保留的反例：**

- 4K paired512在128B下，Full aux0→18为276.121→277.197、275.066→277.451，两seed均退化。
- 8K无转置分片，aux0下DPP使Down341.536→348.369、315.025→326.186，慢约2.0%/3.5%；Full略好。
- 4K M256四wave，aux0下DPP使Down390.990→410.561、Full487.964→511.333，不能修复该配置的资源问题。
- 32K写出改128B后，width4/8的Full约1636～1645，慢于无转置分片约1631；旧任务顺序排名不能直接沿用。

## 7. 提炼独立实现与移植到八wave

### 7.1 Tensor搬运与独立四wave

[四wave实现](moe_4wave_down_a8w4.py)只保留paired、DPP、BN64/128、aux0/18、可选分片，
去掉实验用priority、LDS消融、任务swizzle和自复位。caller提供counter，每launch当前stream清零一次；并发需独立counter和输出。

`copy_async_lds`接收Tensor，完整16B用`BufferLoadAsyncLDS128b`，最后1～3个dword用`BufferLoadAsyncLDS32b`，统一由`fx.copy`发射。
子视图保留descriptor边界；LDS传wave起点，硬件加lane偏移，不可重复相加。async分组和等待仍由原流水控制。

与保留的raw-DMA实验parent同址交替7轮×30次：

| 配置 | raw parent D/F | Tensor copy D/F |
|---|---:|---:|
| 4K K256 BN128 aux0 | 182.812 / 275.843 | 182.760 / 275.773 |
| 16K K256 BN128 aux18 | 465.377 / 858.354 | 465.366 / 858.779 |
| 32K K256 BN128 aux18 分片 | 843.967 / 1640.458 | 845.771 / 1641.371 |
| 16K K384 BN64 aux18 | 526.966 / 921.360 | 526.695 / 921.068 |

差异均小于0.3%；无新增scratch，DMA/MFMA/store/wait静态数量保持，完整指令序列不相同。
独立提炼时也对同配置parent复测，结果基本一致。旧occupancy辅助不识别新kernel名时，没有冒用parent查询作为新版实测。

### 7.2 八wave组合

[八wave优化实现](moe_8wave_down_a8w4_optimized.py)复用Tensor搬运与DPP，加入已有BN128提前打包，
但保留M256、256 workers、全局队列和原4+4错相协议；不能直接删掉错相的一端或换成四wave paired。

候选控制链：`8wave_opt_control` → `8wave_opt_dpp` → `8wave_opt_ntsc1` → `8wave_optimized`。
另有`8wave_opt_dpp_overlap`隔离cache因素、`8wave_opt_oc1`隔离OC、`8wave_opt_bn64`对比BN64无overlap。
control包含Tensor DMA等实现迁移，**原始版本→control不是纯cache/DPP对照**。

K256确认轮seed1234，7轮×30次：

| Tokens | 原八wave OC4 D/F | DPP aux0 D/F | DPP aux18，无overlap D/F | 完整组合 D/F | 四wave BN64 D/F |
|---:|---:|---:|---:|---:|---:|
| 4096 | 188.777 / 287.393 | 186.809 / 287.550 | 187.235 / 283.171 | 187.769 / 283.306 | 182.501 / 273.620 |
| 8192 | 290.505 / 497.283 | 275.668 / 483.929 | 272.615 / 465.277 | 273.211 / 466.083 | 278.971 / 469.886 |
| 16384 | 488.484 / 914.559 | 470.339 / 886.784 | 462.458 / 853.006 | 462.826 / 853.999 | 465.939 / 856.108 |
| 32768 | 1117.865 / 1963.202 | 887.617 / 1719.911 | 868.355 / 1656.822 | 870.647 / 1659.690 | 855.876 / 1646.131 |

第二seed完整组合Full为283.765/465.859/850.798/1662.230。K256 overlap减少寄存器，但无稳定额外提速；
完整组合是简洁跨K候选，不是每个shape最优。4K仍不如四wave/Aiter，32K K256仍略慢于四wave。

K384第二seed确认轮：

| Tokens | 已有八waveoverlap D/F | DPP＋overlap aux0 D/F | 完整组合aux18 D/F | 完整组合OC1 D/F | 四waveBN64 D/F |
|---:|---:|---:|---:|---:|---:|
| 8192 | 329.381 / 536.966 | 323.307 / 529.467 | 319.694 / 510.510 | 332.273 / 522.295 | 328.088 / 517.956 |
| 16384 | 547.490 / 980.044 | 530.803 / 965.634 | 517.790 / 914.824 | 546.433 / 936.869 | 525.557 / 916.874 |
| 32768 | 1168.004 / 2006.966 | 995.318 / 1823.933 | 970.068 / 1769.580 | 956.218 / 1748.642 | 1029.866 / 1837.964 |

第一seed完整组合Full为510.330/915.437/1751.092；32K OC1稍优，两seed同方向，保留为显式候选。

| K | 新八wave配置 | VGPR | private bytes | VGPR spills |
|---:|---|---:|---:|---:|
| 256 | control | 209 | 0 | 0 |
| 256 | DPP，无overlap | 214 | 0 | 0 |
| 256 | DPP＋overlap | 176 | 0 | 0 |
| 384 | control | 256 | 128 | 31 |
| 384 | DPP，无overlap | 256 | 132 | 32 |
| 384 | DPP＋overlap | 256 | 0 | 0 |

K384组合消除scratch，但LDS仍108544B；不是驻留增加。K512组合只将private256→120B、spill111→37。
**反例：**K384/16K新control Down1023.546慢于旧OC4的929.211；DPP＋aux18无overlap仍991.213，完整组合518.778。
八waveBN64也不通用：K256/16K Full925.427，慢于BN128完整组合851.495。

<a id="roofline"></a>
## 8. Roofline：性能合理，但没有达到硬件上界

本节是**替换专用归约之前**的当前默认重测，seed1234、7轮×30次，消费者为Torch sum：

| Tokens | 四waveBN64 Down | 八wave优化 Down | 四wave Full | 八wave Full | warm sum约 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 182.524 | 187.460 | 274.223 | 283.634 | 99 |
| 8192 | 278.735 | 274.205 | 469.486 | 467.106 | 197 |
| 16384 | 465.813 | 462.011 | 854.350 | 852.303 | 394 |
| 32768 | 856.714 | 870.891 | 1646.589 | 1663.325 | 787 |

MI350X规格HBM为8TB/s，dense FP8/MXFP8为4.6PFLOP/s。以FP8侧峰值作乐观计算上界，
未独立测此混合scaled-MFMA的持续峰值；不能拿纯MXFP4或稀疏FP8的9.2PFLOP/s直接作为目标。
TB/GB为十进制，GiB/MiB为二进制。

设tokens为T、真实route数R=8T、sorted padded行数L。计算量为：

$$F=2RNK,\qquad F_{exec}=2LNK.$$

输入理想各读一次、BF16中间值写出再读回时：

$$Q_A=RK(1+1/32),\quad Q_B=ENK(1/2+1/32),\quad Q_C=2RN,\quad Q_Y=2TN.$$
$$Q_{Down}=Q_A+Q_B+Q_C,\qquad Q_{Full}=Q_A+Q_B+2Q_C+Q_Y.$$
$$t\ge\max(F_{exec}/P_{peak},Q/BW).$$

这是理想复用的算法流量模型，不是所有warm-cache条件下不可突破的HBM下界。
计算/带宽平衡点约575FLOP/B；即使输入免费，F/Q_C=K=256FLOP/B也低于它，Full还要再读C。
Q_B含scale固定306MiB，C随batch为0.375/0.75/1.5/3GiB。

| Tokens | Down理想强度 FLOP/B | Full理想强度 FLOP/B | Down理想8TB/s时间 | Full理想8TB/s时间 |
|---:|---:|---:|---:|---:|
| 4096 | 140.8 | 87.0 | 91.5 | 148.1 |
| 8192 | 180.3 | 100.6 | 142.9 | 256.2 |
| 16384 | 209.7 | 109.1 | 245.8 | 472.3 |
| 32768 | 228.3 | 114.0 | 451.4 | 904.4 |

Down约为该理想roofline的49～53%、Full约52～55%；这是模型距离，不是实测HBM利用率，也不承诺还能轻易提速一倍。
真实padding计算下限，四wave约34/67/101/200，八wave约67/67/134/202，均低于对应访存模型时间。

### Padding、B复用与实测流量

| Tokens | 四wave L / M块数 | 八wave L / M块数 | 四wave有效行占比 | 八wave有效行占比 |
|---:|---:|---:|---:|---:|
| 4096 | 49152 / 384 | 98304 / 384 | 66.7% | 33.3% |
| 8192 | 98304 / 768 | 98304 / 384 | 66.7% | 66.7% |
| 16384 | 147968 / 1156 | 196608 / 768 | 88.6% | 66.7% |
| 32768 | 292352 / 2284 | 294912 / 1152 | 89.7% | 88.9% |

16K B＋scale按每M块请求，四wave约0.966GB、八wave0.642GB；32K约1.908GB、0.963GB。
OC分不同B列，B总字节不能再乘OC；A被各OC重复读取。逻辑请求不能直接当HBM字节。

同机Torch大块fill/copy/sum约4.5～4.7TB/s，只是实用参照，不是设备极限。
另用PMC读取DRAM32B计数，乘32后除以**同dispatch**时间，最后23次profile launch取中位数：

| 路径 | Tokens | HBM读 GB | HBM写 GB | profile时间 | HBM TB/s |
|---|---:|---:|---:|---:|---:|
| 四waveBN64 | 16384 | 0.910 | 1.611 | 478.2 | 5.27 |
| 八wave优化 | 16384 | 0.664 | 1.611 | 495.9 | 4.57 |
| 四waveBN64 | 32768 | 0.932 | 3.221 | 914.0 | 4.54 |
| 八wave优化 | 32768 | 1.057 | 3.221 | 930.6 | 4.59 |

C写出已等于预期有效BF16字节，未见整倍写放大；读流量证明缓存复用确实存在。
聚合计数不能分离A/B/scale，不能推导B专属命中率；也不能将profile字节除以另一轮普通时间。
真实16K Down→Torch sum中，sum约390.2，读1.6107GB、写0.2013GB，4.64TB/s，确实完整读了一遍C。
这说明继续优化归约有价值，不能仅凭低于算力峰值就继续追MFMA。

<a id="routed-sum"></a>
## 9. 专用routed归约：带宽提升与Full收益分开看

[make_moe_sum](moe_multistage_reduce.py)增加`source_layout='routed'`，输入BF16 `[tokens,topk,N]`，无需inverse。
原packed接口保持不变。新模式256线程、2048列/CTA、每lane8BF16，NT读/普通写，Tensor＋`fx.copy`，按token使用64位基址。

为兼容当前Torch非连续维sum，routed模式使用四个FP32累加器，按route%4累加再依次合并。
TOPK8为`(((x0+x4)+(x1+x5))+(x2+x6))+(x3+x7)`；这与packed的slot顺序和不同。
支持N%512==0、TOPK1～16，empty不launch；Torch顺序不是跨版本API保证，升级需重测。

### 独立warm归约

seed2026、9轮×40次；取四waveBN64后的计时。逻辑带宽分子为输入＋最终输出总字节，不是只算读流量：

| Tokens | Torch时间 | 专用时间 | Torch TB/s | 专用TB/s | 结论 |
|---:|---:|---:|---:|---:|---|
| 4096 | 99.613 | 88.731 | 4.547 | 5.105 | 带宽+12.3% |
| 8192 | 195.676 | 175.975 | 4.630 | 5.148 | 带宽+11.2% |
| 16384 | 394.045 | 352.271 | 4.598 | 5.144 | 带宽+11.9% |
| 32768 | 786.141 | 812.674 | 4.610 | 4.459 | 延迟+3.4%，退化 |

seed1234、7轮×30次复现同方向，没有隐藏32K负结果。

### 同轮Full：当前集成结果

以下仍为seed2026，**仅替换归约，Down和输入/输出地址不变**：

| Tokens | 四waveBN64 Torch→专用Full | 延迟下降 | 八wave优化 Torch→专用Full | 延迟下降 |
|---:|---:|---:|---:|---:|
| 4096 | 273.826→252.350 | 7.84% | 283.452→280.604 | 1.00% |
| 8192 | 469.527→465.132 | 0.94% | 466.058→461.257 | 1.03% |
| 16384 | 853.149→845.414 | 0.91% | 851.974→839.871 | 1.42% |
| 32768 | 1656.572→1626.742 | 1.80% | 1659.879→1639.052 | 1.25% |

第一seed四wave下降8.34%/0.89%/1.13%/1.66%，八wave下降1.04%/0.77%/1.44%/1.49%。
不能把warm归约约10%的降时直接加到Full；也不能从32K warm退化断言Full退化。

### 真实Down→归约PMC

最后23个完整流程，32K Torch分为**两次dispatch**，必须合计；专用归约一次：

| Tokens | 归约 | HBM读 / 写 GB | profile归约时间 | HBM TB/s |
|---:|---|---:|---:|---:|
| 16384 | Torch | 1.610706 / 0.201328 | 389.374 | 4.654 |
| 16384 | 专用 | 1.610631 / 0.201327 | 357.933 | 5.062 |
| 32768 | Torch，两dispatch合计 | 3.221423 / 0.402656 | 795.389 | 4.556 |
| 32768 | 专用 | 3.221243 / 0.402653 | 740.267 | 4.895 |

流量基本相同，提升表现为带宽而非少读写。新流程中profile Down也变慢：16K约470→487，32K约859→890。
这说明跨dispatch状态影响组件，尚不能归因到某一级cache；最终收益以普通Full配对计时为准。

## 10. 验证边界与实现经验

### 已验证的范围

| 阶段 | 验证重点 |
|---|---|
| 首轮单因素 | 登记表每条边只改一项；K128/256/384全部候选；packed真实4GiB两侧写出 |
| compute-overlap | 81组代表性shape；精确route/Full/graph；短三packet与轻微回退复测 |
| 四wave驻留与队列 | 分片任务非8整除、empty/invalid→restore、counter终值；动态LDS同代码对照 |
| 四wave DPP/cache | 163个四格组、652格；326对cache标志ISA对照；奇偶混合有效行与guard |
| Tensor异步copy | 34项，32b尾部、128b部分wave、多轮、64/256/512线程、动态偏移、OOB零填充 |
| 八wave移植 | 24个shape、176种开关组合；88对cache标志ISA对照；OC1/2/4/8，K128～512 |
| routed归约 | TOPK1～16 × 五种N共80项，另测真实跨4GiB读；宽指数相消、graph、empty和guard |
| packed归约兼容 | 原8项回归通过，保留原接口和跨4GiB行为 |

各阶段的条目存在重叠，不将它们简单相加宣称一次统一测试规模。未穷举任意GPU、编译器、routing或shape。

**K512限制必须保留：** N3072、tokens65、E4、TOPK2的原八wave模型超差比例为5.3270%（seed1234）、
5.2724%（seed43），略高于5%，在执行新DPP候选前已失败。没有放宽门槛，也没有定位根因。
后续K512零容差/graph通过只说明与原route一致，不能改写成模型参考通过；独立八wave组合仍有scratch。

### 易错点

- `_tensor`的shape先按传入指针元素定义，再recast；FP4/E8M0入口是byte，转Int32后索引单位是dword。
- 当前Aiter `mxfp4_moe_sort_fwd`的scale分配按**32行**对齐，不按sort block；不能从个别M128样本推断通用容量公式。
  IDs容量可能不整除M，expert块容量向上取整；只处理完整valid前缀。`num_valid_ids`有两个int32字段，内核读取首字段。
- Tensor全标量索引返回数值，含`None`的切片才是可`.load()`的Tensor。BF16转换部分旧API返回legacy标量，raw边界需兼容解包。
- DPP必须重排目标route/mask；packed无效地址要选整个块外packet，不能只改row导致别名到下一packet。
- 新Down接口限制各buffer小于4GiB；packed实验和归约另有64位重基址支持，不能把归约高地址能力推广成Down已支持。
- 同一counter不能被并发launch共享。global→LDS每组必须闭合asyncmark；slot重用前等待所有读者，不能只等当前wave。
- HIP occupancy看GPU符号而不是host wrapper；查询失败需清除HIP错误状态。静态LDS改变可能改变寄存器，动态LDS消融更利于隔离驻留。
- 实际计时ELF的aux0/18应在去掉store标志后完整一致；改变布局/循环展开时不要求静态MFMA数量跨版本相同。
- Pylance导入/DSL类型警告与已验证运行环境分开记录，不能把编辑器警告当作GPU失败，也不能声称所有编辑器检查干净。

## 11. 下一步：保持简单，不叠加未经验证的组合

- **K256：**DPP＋cache是主要收益；overlap是跨K一致实现的选择，不是K256额外大收益。
- **长K：**优先减少spill，再看DPP/cache；BN64应按实际资源和Full复测，不因更高驻留就设全局默认。
- **小batch：**优先减少padding和任务开销；Aiter在此前Torch口径的4K对照仍更快，不宣称本地实现全形状最优。
- **大batch：**分片/转置随写出布局重新测；没有证据时不改物理归属假设，不直接移植同步协议。
- **归约：**目前尚未达到6～7TB/s，32K warm退化仍待研究。先做发射、地址和cache单因素对照，再量Full。
- **结构性空间：**16K的C写＋读为3GiB，32K为6GiB；packed本身不会删掉这些字节。
  跨expert片上归约才能改变该流量下界，但必须保留每route BF16及求和语义，原子FP32不是免费等价替代。

此前roofline估算“归约若达6～7TB/s，16K Full可省约11～16%”只是Down不变时的条件上限；
现在实测只有约5.1TB/s，且Down状态会变化，不能把该估算当作本次已兑现的收益。

## 12. 代码与重现入口

| 文件 | 用途 |
|---|---|
| [test_a8w4.py](test_a8w4.py) | 默认比较、候选选择、模型/route检查、Down/Full、归约带宽表 |
| [moe_4wave_down_a8w4.py](moe_4wave_down_a8w4.py) | 独立四wave、Tensor DMA、DPP |
| [moe_8wave_down_a8w4_optimized.py](moe_8wave_down_a8w4_optimized.py) | 八wave组合及单因素开关 |
| [moe_multistage_reduce.py](moe_multistage_reduce.py) | packed及routed归约 |
| [a8w4_test_utils.py](a8w4_test_utils.py) | 公共数据生成、原生sorting、零容差检查与报表 |
| [a8w4_test_isa.py](a8w4_test_isa.py) | 从当前ELF读取ISA，供主回归使用 |
| [experiments/bench.py](experiments/bench.py) | 单因素／OC4／64/128B × aux0/18，共用轻量性能流程 |
| [experiments/moe_8wave_down_a8w4_experiments.py](experiments/moe_8wave_down_a8w4_experiments.py) | 历史单因素工厂；保留负结果开关 |
| [experiments/moe_a8w4_4wave_oc4.py](experiments/moe_a8w4_4wave_oc4.py) | 历史四wave工厂，非主回归依赖 |
| [历史源码快照](experiments/archive/pre_cleanup_20260915.tar.gz) | 五套旧探索脚本、两工厂和临时探针；不自动收集 |
| [test_a8w4_8wave_optimized.py](test_a8w4_8wave_optimized.py) | 八wave组合、边界、ISA检查 |
| [test_async_copy_lds.py](test_async_copy_lds.py) | Tensor异步搬运边界 |
| [test_routed_moe_sum.py](test_routed_moe_sum.py) | routed归约精确顺序与高地址 |
| [test_blockscaled.py](test_blockscaled.py) | 独立FP8 blockscale内核族，保持完整回归 |
| [prof-hbm.py](prof-hbm.py) | DRAM读写计数器采集与同dispatch带宽 |

比较脚本支持`--tokens`、`--n`、`--k`、`--experts`、`--topk`、`--candidate`、`--seed`、`--rounds`、`--iters`。
`--mode down`只测Down；默认测Down和Full；`--reducer torch`保留旧消费者；`--compare-reduce`同址交替比较两种归约。
`--profile`要求单候选/单形状，直接launch23次，不把profile输出当普通性能。
运行使用项目Python3.10环境；现有Torch/Triton插件导入问题可通过先导入`triton.language`再执行pytest规避，不修改数值门槛。

### 主回归与探索分层（2026-09-15）

- 顶层保留五个常用测试入口。主四wave以原八wave的原生sort128输出作精确参考，不再依赖历史四wave工厂。
- 四wave主配置保留全部边界；八waveDPP主配置保留24个shape × aux0/18的完整graph、guard、invalid→restore及ISA对照。
  非默认DPP/overlap/cache消融仅在K128/256/384/512各做一次普通精确输出检查，不再逐组合重复九种边界。
- 五套历史性能探索合并为[显式入口](experiments/bench.py)：每个候选先做一次零容差route/Full检查，再用共享地址正反序测Down和直接Full。
  不自动运行graph鲁棒性、guard矩阵、ISA、occupancy或跨4GiB探针。少测场景不等于放宽精度；失败仍立即中止。
- 从本目录以项目Python的`-m experiments.bench`模块方式运行；`--suite single`复现单因素，`--suite oc4`比较四wave组织，
  `--suite stores`比较64/128B × aux0/18；用`--variants`选择候选，shape/seed/rounds/iters均可显式指定。
  single自动补齐登记的父对照，OC4/stores补原八waveOC4；跨M或队列的差异不是单因素收益。
- 历史Full继续使用routed Torch sum，packed则包括inverse与顺序FP32 gather sum；不能与主入口默认专用routed归约的Full混为一谈。
  已移除的旧overlap快捷候选不暗中映射成新内核；旧OC4 overlap及低价值驻留/priority/width矩阵可从源码快照恢复。
- [本地收集规则](conftest.py)排除实验目录，避免根pytest配置的`python_files=*.py`意外收集。原始脚本完整归档，不删除历史负结果。

此次仅整理测试/入口，四wave、八wave优化、原八wave、两种归约及blockscale相关运行时源码均逐字节未变；
六个公共测试函数的源码及AST保持一致。四个A8W4/DMA/归约入口172项通过，blockscale另73项通过，共245项；
最终按目录统一运行，连同旧buffer工具自带的一项测试，共246项通过；三个探索suite的普通精度/短计时冒烟均通过。
第10章的大矩阵数字属于此前验证记录，不再要求每次探索重跑。

清理前后：MI350X GPU0、未锁频、seed1234，同候选7轮×30次，默认专用routed归约。Full中位数（μs）：

| K / Tokens | 四wave BN64 前→后 | 八wave优化 前→后 |
|---|---:|---:|
| 256 / 4096 | 251.246→252.269 | 280.660→280.111 |
| 256 / 8192 | 469.961→466.450 | 467.577→462.361 |
| 256 / 16384 | 844.924→845.509 | 839.556→840.216 |
| 256 / 32768 | 1627.947→1632.129 | 1637.122→1637.504 |
| 384 / 16384 | 904.173→903.046 | 892.020→893.695 |

对应Down变化−0.68%～+0.29%，Full变化−1.12%～+0.41%，未观察到明显回退，亦不据此宣称清理提升GPU性能。
GPU0启动前空闲；后测时另一个GPU有负载，且未锁频，因此这些微小变化不能解释为严格隔离的因果收益。