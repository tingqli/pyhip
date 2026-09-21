---
name: grread-bf16-prefill
description: 'Use when optimizing or reviewing gfx942 FlyDSL GRRead BF16 prefill Down/Up kernels, small-M M16/M32 tiles and wave counts, H64 stream weight permutations, X128 cooperative loads, MFMA and sigmoid/FMA overlap, W2 prefetch, N-split dispatch, runtime rows, or exact-row buffer bounds.'
---

# GRRead BF16 prefill优化

目标：在不改变已接受的数值顺序、有效工作量和两kernel调用方式的前提下，优化布局、数据复用与流水。先读当前代码，不把旧实验的M128、FP32 P、W3或persistent队列当作现行基线。

## 1. 固定数值和调用合同

从[唯一测试入口](../../../tests/contrib/gr_read/test_gr_read.py)和[Down](../../../src/pyhip/ops/gr_read/flydsl/prefill_down.py)、[Up](../../../src/pyhip/ops/gr_read/flydsl/prefill_up.py)核对：

- 固定`C=4, H=2560, R=320, K=C*H=10240`；X为BF16 `[T,K]`，P为BF16 `[T,R]`，Y为BF16 `[T,H]`，P无行padding。
- Down完整K10240 FP32累加，GEMM结果先舍入BF16，再以FP32乘0.25、SiLU，最后写BF16 P。不要将这次中间舍入移到激活之后。
- Up完整R320归约，sigmoid消费原始FP32 logits；同一输出保持stream0→1→2→3的FP32 FMA顺序，最后乘0.25并使用既有整数BF16 helper。
- 整数位模式加`0x8000`与浮点FMA加`bitcast_f32(0x8000)`不等价；不要把替换打包称为无损，也不要未经证明把既有helper等同于任意BF16转换模式。
- 原参考和逐元素容差保持：P `rtol=0.015625, atol=2e-5`；Y `rtol=0.01, atol=0.005`。rel_l2只报告，不能替代逐元素检查；通过容差不代表与另一实现bitexact。
- 非空执行Down＋Up两次launch；0行不launch。准备、shuffle、分配、首次JIT不进入热路径计时。

## 2. 先证明权重顺序和lane映射

Up权重原索引为`W[stream,h,r]`。将H拆为：

$$
h=64t+32a+8g+4b+m,
\quad a,b\in[0,2),\quad g,m\in[0,4).
$$

当前预排把`[stream,t,a,g,b,m,r]`变为`[t,stream,a,b,g,m,r]`：

1. 同一H64的四stream聚在一起，8个H32 packet按`(s0,a0),(s0,a1),...,(s3,a1)`消费。
2. 物理H32位置`16b+4g+m`对应逻辑通道`8g+4b+m`；`g=lane//16`的lane组在两个H16片段中取得连续8个BF16，适配X恢复与Y的16B store。
3. 再调用[权重preshuffle](../../../src/pyhip/ops/gr_read/flydsl/common.py)，将N16/K32块变为连续的MFMA输入存储布局。这与上面的逻辑通道重排是两个不同步骤。

修改时先用CPU行标签覆盖全部10240个输出通道，证明排列不重不漏、R不改变；再验证实际X/输出寄存器对应。不能只改permute而沿用旧packet地址、totals索引和Y写出。

## 3. 按CTA轮数选择N分片

使用[当前选择模型](../../../src/pyhip/ops/gr_read/flydsl/common.py)，分别评估Down与Up：

$$
N^*=\arg\min_N\left\lceil B_MN/U\right\rceil c_N,
\quad B_D=\lceil T/64\rceil,\quad B_U=\lceil T/256\rceil.
$$

- 正式Down通过`select_down_config`返回`(block_m, num_waves, n_splits)`：80CU且1..1024行用M32/W4/N5，其余回退M64 N1/N2成本140/105模型；0行不launch。其它CU不套用此M32阈值。Up在80CU且1..4096行把N10/N20成本65/40加入原N2/N4/N8成本280/144/77模型，其余范围仍用原候选。成本和阈值是本机校准，不是跨硬件保证或每batch运行autotune。
- Up小batch按CTA轮数而非两段硬阈值选择：1..1024=N20、1025..2048=N10、2049..2560=N8、2561..3072=N20、3073..4096=N10；每个区间都已核选择函数，性能只测512/1024/2048/4096，中间行数是插值。不要把一次整轮／尾轮变化误判成选择器不应非单调。
- Down N2是两片N160，不是split-K。wave布局由4N×1M改为2N×2M，仍256线程；每wave每K64拍MFMA由80降到40，但CTA数翻倍。
- N2每wave仍有A2＋B10 VMEM请求，B在不同M wave独立读取，不能按输出列数机械推成B5。
- 大小batch分别验证：更多CTA可以填满小batch，也可能使大batch增加轮数和重复读取。Down grid用真实ceil(T/BM)×N，BM为所选32或64，不要恢复已删除的M256 padding模型；上述M64回退公式不描述M32驻留数。
- Up现行为M256／512线程，每CTA在N2/N4/N8下处理160/80/40个H32 packet；归约维始终是R320，不能把它误当packet总宽度。
- Up按40个完整H64组切分：N10/N20各4/2组；N20没有稳态循环，FIRST状态直接传LAST。N16不能等宽套用，因为会拆开四stream归约；N40还需单组首尾特化，未实现。切细重复P读取及首尾开销，不额外复制互不重叠的X/W/Y片段。
- Down N4/N5各BN80/64，wave布局1N×4M／4N×1M，每wave每K64拍分别20/16条MFMA、A2+B10／A2+B2请求。保留完整K归约和M64，重算真实请求/调度配额；更多CTA重复读完整X，160个K64步骤及同步不减少，不能只凭任务数推断提速。
- [第一轮1024/2048同址证据](../../../tests/contrib/gr_read/results/small_n_splits_20260920/timing/summary.json)支持Up N20/N10及Down N5作为下一步候选；[512/4096补测](../../../tests/contrib/gr_read/results/small_n_splits_20260920/extra_512_4096/timing/summary.json)中，512仍以Down N5＋Up N20较好，4096则需保留Down N1、Up候选N10。扩大batch必须加入该shape当前默认实现，不能只与旧小batch基线比较；Down N5相对4096的N1实际更慢。以上只覆盖已测候选和独立阶段，不代表其它batch或Total最优。新布局的VGPR减少不能替代实际occupancy证据，也不能套用旧路径1CTA/CU的假设。
- [Up正式接入证据](../../../tests/contrib/gr_read/results/up_splits_auto_20260920/audit.json)：40项基础正确性覆盖尾行与所有选型切换，默认17档不变；仅四档正式Down/Up/Total共120条样本、12份门禁。Down规则和计算未改、Up仅更新选择和说明；本轮与先前调优非同址前后对照，不用历史绝对时延相除声明收益。

### Down切小M与wave数

- 用户当前优先小M，不优先split-K。保持完整K10240累加、BF16边界、单次Down launch；工厂`make_down`接收`block_m`和`num_waves`，非M64只允许N5及M32/W4、M32/W2、M16/W2。仅M32/W4按用户要求进入正式小batch自动选择，两wave仍为显式实验候选。
- 固定BN64/BK64时，以`threads=64*num_waves`重建A copy线程布局。每wave的A VMEM请求数为`BM*BK/(threads*8)`，不是CTA所有wave的请求之和；B为`BN/N_WAVES/8`。M64/W4、M32/W4、M32/W2、M16/W2的A/B请求依次为2/2、1/2、2/4、1/4。
- A LDS读为`BM/M_WAVES/8`，MFMA为`BM/(16*M_WAVES)*BN/(16*N_WAVES)*(BK/16)`；A store条数随A请求数变化。调度配额应按这些真实粒度派生，不能沿用M64固定两条A store。双槽LDS字节数为`2*BM*BK*2`。
- 小M增加CTA及B权重重复请求，固定N时不额外翻倍有效X总请求；每CTA160个K64步骤和同步仍在。两wave不保证更快：同M32时每wave B和MFMA工作量加倍，必须测量。
- [小M同址证据](../../../tests/contrib/gr_read/results/down_small_m_20260920/timing/summary.json)：512以M16/W2较好，1024以M32/W4较好，2048仍M64/W4；均为N5阶段对照，不推断Total或未测batch。13行数×4配置逐位/guard通过，旧M64四分片ELF不变；新配置无spill，但不据此断言实际驻留数。
- [正式接入证据](../../../tests/contrib/gr_read/results/down_m32_auto_20260920/audit.json)：默认新增512，共17档；33项基础正确性通过，包含M32尾块及1024/1025切换。仅512/1024重测正式Down/Up/Total，60条raw和6份门禁保留；这不是与隔离调优同址配对，不用两场绝对时延计算接入收益。

## 4. 重叠独立算术，控制寄存器生存期

在Up的`run_group`中，MFMA算当前packet q，sigmoid、X展开、FMA和BF16打包处理q−1。先建立数据依赖表，再改变发射位置：

1. 保持当前MFMA结果与旧post数据独立；仍检查物理寄存器复用造成的WAR/WAW。
2. 把scale→exp→add→rcp→FMA的多条独立元素链交错，避免把整个后处理链挤在一起。
3. 当前源码每10条MFMA推进两个旧元素：exp/rcp各占独立间隔，其他间隔至多安排3条普通VALU。以实际ISA核验，不以源码顺序代替机器调度。
4. 控制中间值生命周期，不盲目pin所有临时值；检查VGPR/SGPR、scratch和spill。更少寄存器不保证更多驻留CTA，LDS也可能是限制项。
5. 分别验证FIRST、LOOP、LAST和独立drain；末包没有后续MFMA可遮盖，必须正确完成post与写出。

通用方法见[MFMA流水skill](../flydsl-mfma-pipeline/SKILL.md)。本核的具体等待表和资源预算不能直接套到FP8 MoE核上。

## 5. 联合设计X协作读取、W预取与同步

- 当前X128是同一行8个lane各读16B，合计128B；不是单lane128B指令。每wave四条load覆盖M32/H64，相邻两个H32 packet消费不同半区，不重复读取整个H64。
- LDS恢复MFMA的lane布局：每wave1KiB低半区分时复用，2KiB高半区保留至后处理消费。8wave共24KiB，配合两个20KiB W槽总计64KiB。
- W2在Memory(q)把W(q+1)写下一LDS槽，预读W(q+2)，寄存器只保留一个搬运包。W3→W2减少提前量和活跃数据，不减少总W读取量，也不保证所有布局都加速。
- 预取距离、load/store顺序改变时，重算FIRST/LOOP/LAST的VMEM完成队列；将W搬运、X转置、Y写出都计入。不要直接复用旧`vmcnt`表。
- 4＋4wave错相是闭合的同步协议；保留prologue和epilogue的配对barrier。CTA barrier不能替代VM/LDS完成等待；允许覆写LDS之前要确认所有消费者完成。
- X/P读取和Y写出当前使用NT，W使用default；它是此工作集的策略，不是所有张量都应NT的普遍规律。X128增加LDS工作，HBM字节减少也不自动代表时延减少。
- 任务映射现行为M-major、N相位0；不把历史观测的XCD归属当作硬件pinning保证。新swizzle需另行证明覆盖及实测归属。

## 6. 用局部descriptor消除大地址限制与行padding

1. 以64位计算`row_begin * stride`，在乘法前升宽，将X/P/Y指针移到当前CTA tile。
2. buffer内部只用局部偏移；descriptor范围只覆盖有效行，不传整个多GiB张量的extent。
3. Down N1输出extent为`valid_rows*R*2`；N2因行stride仍为R，extent是`((valid_rows-1)*R+BN)*2`，valid0用0，不是`valid_rows*BN*2`。
4. Host传二维X/P/Y，使动态shape各维可表示；展平后numel可能超过FlyDSL动态shape的i32范围。
5. rows为运行时Int64；Down工厂按N分片、M块与wave数缓存，Up按N分片缓存，不按batch生成kernel。不用Host窗口和多launch绕过地址问题。
6. 正常`flyc.compile`会执行launcher一次，必须传足量真实输入和输出，不能用单行占位给完整grid。测试内准备函数已经使用本次X。

## 7. 验收与证据

先跑当前唯一基础测试；针对本次布局或边界变更，用隔离验证检查N1/N2尾行guard、NaN覆盖、同址输出与原容差，不扩建永久专项测试。性能先用明确的少量shape、原timer和相同地址条件，确认后再按请求扩大；详见[GPU测量skill](../gpu-benchmark-validation/SKILL.md)。

可核验的方法来源（历史条件不冒充当前测量）：
- [X128合并请求与LDS恢复](../../../tests/contrib/gr_read/results/n4_x128_20260918/summary.json)。
- [W2与映射的独立2×2实验](../../../tests/contrib/gr_read/results/n4_prefetch_mapping_20260919/comparison.json)。
- [Down N1/N2的同址校准](../../../tests/contrib/gr_read/results/down_n2_20260919/summarize.json)。
- [无padding尾行写出的22组guard验证](../../../tests/contrib/gr_read/results/unpadded_rows_20260920/guards.json)。