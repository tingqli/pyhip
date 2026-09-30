# QSA 优化日志

## 2026-09-25：TP2 / TP4 三路精确分流

### 范围与预先声明的验收

- 主性能形状 M=12000；no_prefix P=0 / KV12000，chunk_prefill P=12000 / KV24000。
- TP2：Q12/KV1/D256；TP4：Q6/KV1/D256。TP8：Q3/KV1，仅正确性，不设性能门槛。
- BF16；512 个四-token 块和 0..3 个尾部，输出2051槽合同不变。
- 不改变逐-query Top-K、不扩大可见集合、不修改数值容差；有效 FLOPs 仍为有效token数×4×本地Qheads×256。
- shared 输入 selection_group 固定32，与 kernel BQ 解耦；同配置 seed17、索引哈希固定。
- 主目标仍约200有效TFLOPS；无法达到时明确失败，不用shared代替independent，不把无效并集MFMA计入分子。
- 正式原 cudaPerf、10独立buffers、2warmup、10samples；探索2buffers/6samples单独标记。
- 门禁 GPU use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8；不修改硬件配置、不等待或重采直到通过。

### 起点

- 上轮源码功能验收83项通过，TP1 independent约33.25/28.11T，shared181.97/187.75T。
- 原auto低重合仍调用Triton，原wave BF16单query/wave、BN16探索约26T，未采用。
- 本轮只修改本qsa目录；相邻MHA参考与冻结baseline保留。不存在旧qsa/opt.md，建立本日志后逐次追加。

### v01：实施顺序

1. 新增独立block-native FlyDSL direct，BN32/64候选，不依赖union建表；保留旧fallback和wave作对照。
2. dense_limit≤2051的前缀走真实dense causal：仅覆盖满足绝对可见长度条件的query。
3. union按实际计算tile对应query构造局部并集，先以BQ≤floor(128/G)确保每份并集只服务一个M128 tile。
4. 新DispatchPlan统一dense/direct/union，GPU gate保证互斥写入；direct模式跳过union。
5. 测试重点改TP2/4，TP8正确性；计时入口支持TP列表、固定selection_group、独立输出及完整raw/源码/ELF。

### v01-prep：静态准备

- 读取当前源码/技能/计时器；当前工作区无已跟踪修改，8张卡中GPU2保留约74%显存，其他卡初始空闲。
- dense采用原生linear参考；非BN64对齐上下文使用本地full-VOFFSET有界DMA，不以mask掩盖物理越界。
- dense静态检查及离线4形状编译通过；尚未据此声明GPU正确性或性能。

### v01-smoke：三路首版正确性

- direct新增BN32/64 block-native FlyDSL：每wave一个query和GQA heads，V占64KiB LDS，Q/K寄存器；BN32为4wave，BN64为2wave。
- auto不再调用Triton attention fallback；Triton仅用于动态union建表。direct模式无union分配、清零、scatter或compact。
- dense从每请求开头取 `min(M,max(0,2051-P))` 个query，保留原packed地址和绝对causal对齐。
- union effective BQ限制为 `min(requested_BQ, floor(128/G))`：TP2=10、TP4=21、TP8=32；每份局部并集仅供一个M128计算tile。
- GPU1入口PTL Enabled/VECTOR,F8，利用率0%；TP2/M37/P0或3000的direct/auto/union共6用例通过原FP32容差。
- [首版数值记录](results/tp24_v01_smoke.json)：最大绝对误差0.00749以内。首轮编译保留M0 warning；另一次仅从缓存执行并保存数值，不作为性能重采。
- 测试调整为TP2/4主性能与TP8功能，BN32/64、dense边界/guard和新DispatchPlan graph重放新增覆盖。

### v01-check：路径验证与首次测量门禁

- [路径测试](results/tp24_v01_routes.xml)：72 passed（147项未选），覆盖TP2/4/8 direct BN32/64、dense边界、graph和局部union。
- 首次direct-only BN32探索在TP2/no_prefix准备后被GPU1 use14%门禁拒绝，0个性能样本；[记录](results/tp24_v01_direct_bn32/no_prefix_tp2_independent_bq32/result.json)保留，不称性能成功。
- harness增加warmup之后对**实际使用的全部输入**重新验证causal/block/tail契约与索引哈希，补齐direct-only无union时的准备后审核；门禁阈值不变、无sleep。

### v02：block-native串行首版，未达到性能预期

- 两协议均2buffers/6samples、TP2/4、固定independent索引；GPU1门禁全部通过，原始结果分别在
	[BN32](results/tp24_v02_direct_bn32/no_prefix_tp2_independent_bq32/result.json)、
	[BN64](results/tp24_v02_direct_bn64/no_prefix_tp2_independent_bq32/result.json)同级四用例中。
- BN32 direct：TP2 no-prefix/chunk为22.12/21.86T，TP4为11.26/10.93T；同场baseline约29.2/28.5T和14.6/14.4T。
- BN64 direct：TP2为14.76/13.97T，TP4为7.35/7.00T；减少softmax迭代未抵消顺序load/wait、较少wave和LDS开销。
- 不采用BN64作为默认；未以较大的tile宣称优化成功。
- 下一候选v03：在16-token子片的QK MFMA前发射下一片K/V加载，V先写LDS，保留异步wait后的scheduler fence；检查寄存器/零spill与原数值门槛。

### v03：K/V预取无收益，继续保留失败证据

- [原始记录](results/tp24_v03_direct_prefetch/no_prefix_tp2_independent_bq32/result.json)同级4用例：TP2为21.48/20.57T，TP4为10.78/10.39T。
- 未优于v02，不能仅凭存在预取宣称隐藏延迟。数值和零spill通过，但额外活跃数据与LDS转置成本仍在。
- v04改为V按PV片段直接global向量加载，在寄存器进行2×2 BF16重排；不再将V写入LDS再读取。仅输出shuffle占32KiB LDS。
- 四个wave分别独立query，BN32/64都保持4wave；原始块/尾部VOFFSET边界不变，明确为不同内存路径的实验。

### v04：移除V LDS往返，采纳

- [BN32](results/tp24_v04_direct_globalv/no_prefix_tp2_independent_bq32/result.json)四用例：TP2 37.13/36.10T，TP4 18.62/18.16T，均优于同场baseline。
- [BN64](results/tp24_v04_direct_globalv_bn64/no_prefix_tp2_independent_bq32/result.json)：TP2 37.13/36.44T，TP4 18.62/18.27T，差别不大；BN32 VGPR214/SGPR55，BN64 VGPR234/SGPR71，scratch/spills均0。
- 保留BN32默认以留寄存器余量，BN64可选。以上2buffers/6samples仍是探索，非正式验收。
- v05：将下一16-token段V低半的加载放到当前PV高半前，检查能否进一步隐藏VMEM；不同variant均保留原始报告。

### v05/v06：完整分流与局部分组边界

- v05 PV局部预取与v04相近：TP2 37.15/36.22T，TP4 18.64/18.16T，不宣称微小差别为稳定加速。
- v06 auto/independent：TP2 41.23/35.80T，TP4 21.27/18.01T；dense前缀有效，但shared固定SG32时TP2/4仅40.46/38.57T与23.96/18.50T。
- 原因证据：[shared报告](results/tp24_v06_auto_shared/chunk_prefill_tp4_shared_bq32/result.json)表明TP4 BQ21有大量组跨越原SG32选择边界，仅215/572组走union；TP2 BQ10也跨界。未将改变selection_group视为修复。
- v07按硬件行容量取2幂BQ：TP2=8、TP4=16、TP8=32，且沿请求原始query边界对齐；dense切割后的首组允许短组，不把后续分组整体平移2051行。
- 输入和索引哈希仍固定SG32，本次只改执行分组；这是通用的tile边界策略，不更改模型选择集合。

- v07首跑数值通过但harness静态partition审核仍按旧2051起步等长分组，报 `Sparse-only metadata mismatch`，0性能样本；[失败记录](results/tp24_v07_aligned_shared/no_prefix_tp2_shared_bq32/result.json)保留。审核同步为请求原始边界对齐，未放宽验证。

### v07a：固定输入的对齐分组有效

- [shared测量](results/tp24_v07a_aligned_shared/no_prefix_tp2_shared_bq32/result.json)同级四用例：TP2 no-prefix/chunk 134.92/139.57T，TP4 122.52/128.71T；原SG32索引哈希逐字不变。
- 所有稀疏组都进入union，之前跨选择边界的退化消除。effective BQ8/16不是32，日志中明确记录。
- v08尝试每query排序512个块以改善低重合direct访问局部性；预分配排序buffer，Triton排序计入rebuild+run，原indices/block_indices不改写。

### v08/v09：排序与packet地址复用

- [v08 direct](results/tp24_v08_sorted_direct/no_prefix_tp2_independent_bq32/result.json)：TP2 37.62/37.70T、TP4 18.84/18.91T；计入排序后仍高于旧baseline，TP4/no-prefix收益较小但保留显式开关。
- v09将一个PV四-token packet内重复block读取/寻址合并；[完整auto](results/tp24_v09_packet_address/no_prefix_tp2_independent_bq32/result.json) TP2 43.16/37.04T、TP4 22.37/18.74T，含建表为41.35/35.54T和21.47/17.99T。
- v10减少无效预处理：先判断union膨胀gate，再只对active组计算前缀扫描/写compact表；block排序只处理实际走direct且非dense的行，graph重放时每次重新判断gate。

### v10：GPU建表按实际分流裁剪

- [探索报告](results/tp24_v10_gated_prepare/no_prefix_tp2_independent_bq32/result.json)四用例：TP2 run43.16/37.06T、plan+run41.72/35.80T；TP4 run22.38/18.73T、plan+run21.61/18.06T。
- 已优于当前同场baseline，但远未达到200；本轮不改变FLOPs定义，也不声称TP4只有6个heads就能维持TP2吞吐。
- 准备正式10buffer验收与完整正确性；测试中的旧BQ10/21和BN64两wave结构断言同步为采纳的8/16、统一4wave。
- audit默认审计冻结源码本身，`--check-current`才比较当前实现；历史报告不因后续优化而被篡改，TP8/参考baseline的target为空且不算达标失败。

### v11：完整测试通过后独立审查发现的边界修复

- [v10完整JUnit](results/tp24_v10_correctness.xml)：219 passed，包括TP2/4大形状及TP8功能；测试通过不等于不存在未覆盖边界。
- direct补齐wave查gate改为钳制到**当前direct tile的最后有效query**，不能读后续dense行对应的`query_tiles=-1`。
- 输出与Q/K/V重叠检查移到dense空调用早退之前，纯sparse模式也拒绝原地覆盖输入。
- sort_blocks=False时run读取当前inputs.block_indices而非prepare时保存的旧tensor；sort=True继续使用本轮重建的排序buffer。
- rebuild_plan显式切入inputs.q.device，避免当前设备不同导致Triton kernel和PyTorch zero在不同卡/stream执行。
- 增加auto gate翻转graph、无排序替换输入、纯sparse重叠拒绝等回归；上述修复不改变测试容差或选择语义。

- [v11回归](results/tp24_v11_boundary_fixes.xml)：19 passed，含auto gate正反翻转、sort=false替换tensor、跨当前GPU重建、输出alias拒绝。
- v12：direct以CTA-uniform gate检查四个实际query，全部由union处理时在进入计算/输出shuffle之前退出；禁止wave级提前退出破坏CTA barrier。

### v12：统一gate跳过无效direct CTA，最终功能验收

- [shared探索](results/tp24_v12_direct_ctagate/no_prefix_tp2_shared_bq32/result.json)四用例：TP2 139.34/144.78T、TP4 128.91/136.81T，计入建表后126.41/125.83T和110.26/109.97T；所有门禁通过，仍未达200。
- [完整JUnit](results/tp24_v12_final_correctness.xml)：238 passed、0 failed、0 skipped；包含TP2/TP4主矩阵、TP8功能、两种BN、dense阈值、guard、取消误差、graph gate变化、跨当前GPU和输出alias回归。
- Black88、Ruff F/I及Python语法通过；格式调整经AST对比不改变计算。开始冻结正式10buffers/2warmup/10samples的TP2/4结果；禁止用探索样本替换正式样本。

### 最终正式验收：TP2/TP4，10独立buffers × 10样本/scope

- GPU1 / PCI `0000:80:00.0` / MI308X gfx942；每场before/before_samples/after门禁均通过，PTL Enabled/VECTOR,F8，未写任何硬件策略。
- 固定M12000、seed17、SG32，effective BQ TP2=8 / TP4=16，dense_limit2051，direct BN32、sort=true。
- 三个scope分别使用独立output，所有10组输入完整校验、原indices哈希不变；240个正式计时样本全部保留。

| 正式报告 | candidate ms / T | plan+run ms / T | baseline ms | 完整调用加速 |
| --- | ---: | ---: | ---: | ---: |
| [TP2 independent/no-prefix](results/tp24_final_independent/no_prefix_tp2_independent_bq32/result.json) | 6.453 / 42.83 | 6.635 / 41.66 | 9.458 | 1.43x |
| [TP2 independent/chunk](results/tp24_final_independent/chunk_prefill_tp2_independent_bq32/result.json) | 8.142 / 37.12 | 8.448 / 35.77 | 10.568 | 1.25x |
| [TP4 independent/no-prefix](results/tp24_final_independent/no_prefix_tp4_independent_bq32/result.json) | 6.183 / 22.35 | 6.396 / 21.61 | 9.448 | 1.48x |
| [TP4 independent/chunk](results/tp24_final_independent/chunk_prefill_tp4_independent_bq32/result.json) | 8.110 / 18.63 | 8.392 / 18.01 | 10.573 | 1.26x |
| [TP2 shared/no-prefix](results/tp24_final_shared/no_prefix_tp2_shared_bq32/result.json) | 1.985 / 139.27 | 2.187 / 126.37 | 9.238 | 4.22x |
| [TP2 shared/chunk](results/tp24_final_shared/chunk_prefill_tp2_shared_bq32/result.json) | 2.089 / 144.66 | 2.402 / 125.82 | 10.223 | 4.26x |
| [TP4 shared/no-prefix](results/tp24_final_shared/no_prefix_tp4_shared_bq32/result.json) | 1.073 / 128.76 | 1.251 / 110.46 | 9.262 | 7.40x |
| [TP4 shared/chunk](results/tp24_final_shared/chunk_prefill_tp4_shared_bq32/result.json) | 1.106 / 136.63 | 1.372 / 110.12 | 10.187 | 7.42x |

- 主independent两用例均改善，**200有效TFLOPS目标仍未达到**；TP8只验正确性、target为空，未运行性能门槛。
- 不拿TP1历史181/188T替代本轮TP2/4结果；高重合shared不能代表实际模型Top-K。
- 最终union VGPR251/SGPR73/LDS64KiB，directBN32 VGPR214/SGPR67/LDS32KiB，dense bounded VGPR230/SGPR60；全部scratch/spills0。
- 八份报告经`audit --check-current`逐项复算raw中位、有效TFLOPS、240样本数、门禁、冻结当前源码和实际ELF SHA；全部通过。
- 正式结果之后只更新README/本日志，不改变被测计算实现。参考MHA、冻结baseline和历史结果未改写。

## 2026-09-25：迁移交接与清理状态

本节是清理后的继续工作入口；前面的逐版本日志保持历史含义。
本次**没有继续调性能、没有改动results目录**，只删除未使用路径、原样搬迁三个活跃helper、清理生成缓存并验证。

### 1. 清理决定与机械证明

- 删除当前源码中无调用的旧Triton fallback。
- 旧wave实验不能直接删除：direct仍使用它的`_load`、`_qk`、`_pv`。
	已将这三个函数逐字移入 [direct.py](direct.py)，移除旧模块import，将唯一helper绑定改成本地函数，再删除其余wave实验/launcher。
- 两个旧文件在 [正式报告冻结源码](results/tp24_final_independent/no_prefix_tp2_independent_bq32/source/direct.py) 同目录中保留；
	历史实现、IR/ELF、性能失败、原始样本均不改写、不重贴版本标签。
- 删除本目录生成的Python/pytest/Ruff缓存；不删除FlyDSL全局缓存、不清理其他实验目录。
- 其余文件均属于当前运行、测试、配置、来源或交接文档，不再为了文件数而合并。

迁移前后direct SHA-256：

| 对象 | SHA-256 |
| --- | --- |
| 清理前direct | `16f7de9611ffd829ad0832441d70f5794fa5a7025c343f8ac31b0272f6c35891` |
| 清理后direct | `81709d4cd738d0817b8e576156866b7c38c7e93e14fd79606e23b9f0efa18f94` |
| 搬迁的`_load`，含最终换行 | `e450995567f0b2ef0e7bdd5bce953f5f86d6b5c8c6624d14c93b7f14afd5563d` |
| 搬迁的`_qk`，含最终换行 | `42d4d3864b28ae4ba5c95b9274b1cbd6f24ce34d7f61dd8925bb27f5c46897ca` |
| 搬迁的`_pv`，含最终换行 | `2f53346097a59e40371af591857e38daf5ee9056dd55d3e7566407952b0f1dd9` |

已从冻结文件按以下三个唯一文本替换**完全复现清理后的direct字节**，不是只比较函数名或肉眼确认：
删除旧wave import；将三个helper的调用绑定去掉模块前缀；在DirectPlan前插入三个原函数。
函数体、dtype、偏移、MFMA顺序、wait/fence、接口和计时器不变。

清理验证分层：

| 层次 | 已验证事实 | 不等于 |
| --- | --- | --- |
| 源码 | 上述精确复现通过；无旧模块活跃import；其他计算文件未变 | 所有未来特化都机器码相同 |
| 静态 | Black88、Ruff F/I、Python语法、编辑器诊断通过 | GPU数值验证 |
| CPU | 64项契约测试通过 | 完整GPU矩阵重跑 |
| GPU | 18项direct/auto graph回归通过，覆盖TP2/4/8和BN32/64 | 新一轮性能测量 |
| 产物 | TP2/4的M12000/no_prefix/BN32 gated direct，与原正式ELF**整文件相同**，VGPR214/SGPR67、scratch/spill=0 | 未比较特化的逐位等价或跨机器时延不变 |

两个已比较的设备`.text` SHA-256：TP2 `d63a596657cd6b226095722c5131d398d379daba0ba3730665fd6af629e5553c`；
TP4 `e4fa92c7f10566ed09123a8d4be25b3c4d260b0cd4d65bcce01079baa9eff67b`。
未重新计时、未把本次结果写进历史results；18项回归的临时JUnit不是需要迁移的性能证据。

清理前results有2076个文件。按相对路径排序，将每项`relative_path + NUL + sha256 + newline`
串联再做SHA-256，聚合值为`74a9b2fe639a6d2ccf809d082ae45dc36189a29dfb10a3df476e5abe5ee25556`。
清理结束再次核验数量和聚合值，以证明历史证据没有被触碰。

### 2. 当前实现与约束（带到下一台机器）

- 主性能：TP2 Q12/KV1/D256、TP4 Q6/KV1/D256。TP8 Q3/KV1仅要求功能；不把TP1历史吞吐当成本轮结果。
- 两主用例M=12000表示**新增query数**。no_prefix P0/KV12000，chunk P12000/KV24000；BF16 Q/K/V，输出同Q。
- 四-token压缩、512个选中完整块、最多3个尾token；逻辑索引 `[M,2051]`，原块索引 `[M,512]`，有效前缀后-1。
- 每query保持独立选择，所有heads共享该query选择；最终attention读取原始K/V，不是压缩K。
- 默认seed17、benchmark SG32；SG与requested BQ分离，调BQ不改变输入哈希。independent/shared/recent都是合成负载，不是模型真实Top-K。
- `dense_limit=2051`：满足绝对可见长度≤2051的行走真实dense，原Q/O前t行、K/V前P+t行零复制；P12000没有dense行。
- `direct`：四wave/CTA，每wave一个query和GQA heads；BN32默认、64可选；V直接global→寄存器重排，只有输出shuffle占32KiB LDS。
- `union`：按原请求query边界对齐，effective BQ8/16/32；仅一个M128 tile使用该局部并集。共有完整块免mask，其余逐query mask；LDS64KiB。
- `auto`：dense、union active、direct inactive互斥写；不在CPU `.item()` 读gate。全union的direct CTA统一早退，不能wave提前退出CTA barrier。
- 默认attention计算全部FlyDSL；Triton负责union建表与可选block排序。direct模式不建union，但sort=true仍有GPU排序成本。
- `prepare(...)->DispatchPlan`持有dense/direct/union；`implementation.rebuild_plan`切入输入GPU重建gate和排序；`run(...,out=...)`不分配输出。
- 布局/长度/prefix/设备变化必须重新prepare；同布局选择变化同时更新indices和block_indices，再rebuild。
	graph需先warm，capture/replay地址固定；已验证gate正反翻转、排序刷新以及图外sort=false替换输入。
- 必须保留full-VOFFSET尾部边界、NaN guard、输出不重叠Q/K/V、padded wave只查有效query gate、异步wait后的scheduler fence。
- 精确指选择集合和因果语义不变；BF16浮点归约不要求跨实现逐位相同。保持原`rtol=atol=0.02`逐元素检查，relative L2仅补充。

### 3. 保留文件：运行与验收各有用途

除results外，当前保留19个文件；不是每个都参与热路径，但它们都用于继续开发、验证或交接。

| 分类 | 文件 | 保留理由 |
| --- | --- | --- |
| 核心入口 | [implementation.py](implementation.py)、[contract.py](contract.py) | 三路接口、输入/计划类型 |
| 计算与准备 | [dense.py](dense.py)、[direct.py](direct.py)、[kernel.py](kernel.py)、[plan.py](plan.py) | 当前执行的kernel与GPU建表；direct含三个搬迁helper |
| 输入与数学参考 | [inputs.py](inputs.py)、[reference.py](reference.py) | 固定seed用例、合法选择、独立FP32 oracle |
| 冻结对照 | [baseline.py](baseline.py)、[baseline_kernels.py](baseline_kernels.py) | 同场性能对照和数值交叉验证，不是无关fallback |
| 验收工具 | [test_qsa.py](test_qsa.py)、[bench.py](bench.py)、[audit.py](audit.py) | 正确性/边界/graph、原timer/raw/门禁、CPU证据审核 |
| 配置与来源 | [model_config.json](model_config.json)、[source_manifest.json](source_manifest.json) | Qwen固定revision字段、SGLang baseline哈希，不下载模型 |
| 包与缓存规则 | [__init__.py](__init__.py)、[.gitignore](.gitignore) | package-relative导入，忽略可重建缓存 |
| 文档 | [README.md](README.md)、[opt.md](opt.md) | 常用接口说明；本日志是当前状态、实验和下一步交接入口 |

### 4. 外部依赖与迁移边界

**推荐复制PyHIP checkout，而非只复制qsa目录，也不能只安装PyHIP wheel。** experiments/tests不包含在普通wheel里。
若做最小文件集合，必须保留相同相对目录和以下实际依赖：

- 三个MHA模块：[../mha/mha_pa_bf16_942.py](../mha/mha_pa_bf16_942.py)、
	[../mha/mha_pa_bf16_256_942.py](../mha/mha_pa_bf16_256_942.py)、
	[../mha/mha_pa_bf16_256_linear_942.py](../mha/mha_pa_bf16_256_linear_942.py)。它们是运行时helper/native dense依赖，不只是参考资料。
- 计时器：[../../../../src/pyhip/testing/misc.py](../../../../src/pyhip/testing/misc.py)，以及pyhip/testing和pyhip包初始化文件。
- 只读门禁：[../../../../tests/ops/gr_read/test_gr_read.py](../../../../tests/ops/gr_read/test_gr_read.py)，及tests/ops/gr_read的父包初始化文件；不调用GRRead模型或其kernel。
- package链：[../../../__init__.py](../../../__init__.py)、[../../__init__.py](../../__init__.py)、
	[../__init__.py](../__init__.py)、[../mha/__init__.py](../mha/__init__.py)。
- [../../../../pytest.ini](../../../../pytest.ini)、[../../../../conftest.py](../../../../conftest.py)、
	[../../../../pyproject.toml](../../../../pyproject.toml)、许可证/版权标记。
	SGLang快照的Apache-2.0来源不被PyHIP MIT覆盖，分发时一并携带
	[SGLang Apache-2.0 LICENSE](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/LICENSE)。

在目标机安装/选择已有兼容环境：Python≥3.10、ROCm PyTorch、配套Triton、FlyDSL、NumPy、msgspec、pytest。
内核仅支持gfx942；不假定其他AMD架构或NVIDIA可直接运行。ROCm下仍使用`torch.cuda`。
历史实测为Python3.10.12、Torch2.12.0+rocm7.2.4、Triton3.7.1；以新机实际import路径/版本为准，
不要用PyPI CUDA版Triton覆盖ROCm厂商发行版。

性能还需只读`rocm-smi`和支持PTL字段的AMD SMI；原机器临时bundle路径不属于迁移合同，
目标机可通过`--amd-smi`传入已有兼容工具。CPU审计不需要GPU工具；模块入口仍会加载NumPy/msgspec/Torch输入包。
纯审计也可直接运行 [audit.py](audit.py)，此脚本本身仅依赖标准库。

复制注意：

- 使用文件复制/rsync等保留实际工作树，确保新增或未跟踪文件也被带走；不要只依赖`git archive HEAD`遗漏本实验。
- 不带生成的Python、pytest、Ruff和本机JIT缓存；保留源码、config、许可证和需要的历史results。
- 当前results为证据而非运行依赖。可以不传全部探索产物，但建议至少保留tp24_final两组报告及其source/ELF/门禁、完整JUnit；
	部分复制后旧日志的其他链接可能不存在，不能据此补造或改写旧结果。
- 本次在原机器没有删减results。`bench`在新目录生成报告，绝不覆盖旧场次。

### 5. 新机器首先执行的检查

以下从PyHIP根目录执行；`python`应替换为选定的解释器。先运行CPU/功能，再进行计时。

```bash
PYTHONPATH=src python -c 'import torch, triton, flydsl; print(torch.__version__, triton.__version__, flydsl.__file__); print(torch.cuda.is_available())'
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -k cpu
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -k 'test_direct_matches_baseline_and_fp32 or test_auto_graph_gate_flips_rebuild_sorted_direct'
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -m 'perf or not perf'
python experiments/attention/flydsl/qsa/audit.py experiments/attention/flydsl/qsa/results/tp24_final_*/*/result.json
```

- 清理前完整矩阵为238项；本次64CPU+18GPU定向检查不能代替目标机完整验收。无GPU的skip不算功能通过。
- **对旧tp24_final报告不要使用`--check-current`期待通过**：helper搬迁让direct AST改变，严格检查应报告不一致。
	审计器未被放宽；旧结果只认证旧冻结源。下方复现证明可独立验证此次机械变化，目标机新测量生成新快照后再用`--check-current`。
- CPU复现清理（读取旧source快照，不写文件）：

```python
import ast
from pathlib import Path

root = Path("experiments/attention/flydsl/qsa")
saved = root / "results/tp24_final_independent/no_prefix_tp2_independent_bq32/source"
expected = (saved / "direct.py").read_text()
old_wave = (saved / "wave_kernel.py").read_text()
defs = {
		node.name: ast.get_source_segment(old_wave, node)
		for node in ast.parse(old_wave).body
		if isinstance(node, ast.FunctionDef)
}
helpers = "\n\n\n".join(defs[name] for name in ("_load", "_qk", "_pv"))
changes = (
		("from . import wave_kernel\n", ""),
		("    load, qk, pv = wave_kernel._load, wave_kernel._qk, wave_kernel._pv",
		 "    load, qk, pv = _load, _qk, _pv"),
		("class DirectPlan(", helpers + "\n\n\nclass DirectPlan("),
)
for old, new in changes:
		assert expected.count(old) == 1
		expected = expected.replace(old, new, 1)
assert expected == (root / "direct.py").read_text()
```

满足只读门禁之后，使用新的输出目录计时；物理GPU编号按目标机空闲设备设置，示例GPU0不是空闲保证：

```bash
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 0 --tp-list 2 4 --case all --algorithm auto --selection independent --selection-group 32 --output experiments/attention/flydsl/qsa/results/new_machine_independent
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 0 --tp-list 2 4 --case all --algorithm auto --selection shared --selection-group 32 --output experiments/attention/flydsl/qsa/results/new_machine_shared
```

保持原10buffers/2warmup/10samples、PTL Enabled/VECTOR,F8、GPU≤5%、VRAM≤20%。不修改PTL/功率/时钟，不循环等待门禁；
失败保留全部raw和错误。`candidate_run`与`plan_and_run`分开报告；初始化、JIT、cache gather、indexer和服务调度不在计时中。
有效FLOPs按原选择计数，不计并集padding或softmax额外算术。先核对SG32、seed17和indices哈希再比较优化前后。

### 6. 当前性能结论与下一轮任务

当前性能沿用本日志上一节八场正式报告，**没有新的清理后计时数据**：
TP2 independent run42.83/37.12T，TP4 independent22.35/18.63T；shared TP2 139.27/144.66T，TP4 128.76/136.63T。
完整调用较冻结baseline的主independent加速1.25–1.48x，但约200T目标仍未达到，不能用shared或TP1旧数据代替。

优先级与单变量实验：

1. **首段V低128维预取提前**：当前direct的首段V在softmax统计后发射，先只把低半提前，覆盖max/exp/sum窗口。
	 保持BN32、4wave、排序、gate不变；先测TP4/P12000，再测TP2。核对实际ISA、wait和214VGPR附近的live range，不盲目加双缓冲。
2. **四-token块V预排布**：转换为PV operand友好布局，跨query摊薄重复permutation/寻址。
	 转换与新V更新必须计入完整调用，不能只报预转换后的run；保留BF16位值、尾部和请求边界。
3. **TP特定cost model**：当前rho≤1.5未体现G、M/BN padding和common比例。direct固定M16，所以Q12→Q6有效FLOPs减半但指令不一定减半。
	 不可从TP2/TP4约8.1ms相同就断言HBM受限；区分逻辑字节与PMC实测流量。
4. **compact+mask构造融合**：active判定/选择性排序已经实现；进一步减少中间读写和launch，不能忽略跨CTA清零/atomic同步。
	 TP4 shared/chunk的run1.106ms、plan+run1.372ms差额约0.266ms只是两scope差，不是单独量到的plan耗时。
5. **次级探索**：M96/六计算wave或保持DMA职责时跳过无效32行；dense2048对齐主干+3行窄尾。
	 这些需重新验证barrier/布局，后者不改善P12000，不能同时引入多个变量掩盖归因。

已失败/已做的工作不要作为新方案重复：仅把BN变大、一般性K/V预取、V先写LDS再读、所有query共享无mask选择、
继续扩大并集、盲目增加四阶段展开或grid，都已有失败记录或语义风险。固定同一份输入和原容差，
每次在本日志追加假设、源码/ELF身份、正确性、raw、门禁、保留/拒绝理由。

### 7. 清理收尾检查

- 当前非results目录为19个文件、约260KiB；旧fallback/wave源码和本目录生成缓存已在磁盘上确认不存在。
- 2076个results文件数量与上述聚合SHA在清理前后完全一致；八份正式报告共240个raw样本按原口径审核通过。
- 文档本地链接和嵌入的机械迁移复现代码执行通过；严格`--check-current`对旧direct AST的拒绝行为保持不变。
- 未改参考MHA/计时器、未放宽数值/门禁/审计、未重跑性能；迁移后按本节步骤做新的完整验收，再继续优化。

## 2026-09-25：如何集成到 SGLang（方案，尚未实施）

本节基于本机SGLang revision `540d564c19436f28f2644e2247350da56c124452` 的实际调用链核对。
**本次只补充文档，没有修改SGLang生产代码、添加可用启动开关或完成服务端验收。**
当前238项实验测试、清理后回归和kernel性能不等于SGLang端到端集成已通过。

### 1. 接入位置与首版范围

推荐在现有 `QwenSparseAttnBackend.forward_extend()` 内增加默认关闭的实现分支，
不是替换全局attention backend，也不是把实验目录加入服务器的`sys.path`。
Qwen混合模型的full-attention侧会选择QSA backend；仅注册一个通用prefill backend名字，
不保证模型最终使用它。实际链路为：

```text
Qwen4ExpAttentionDecoderLayer.self_attention
		indexer: hidden_states -> logical topk_indices
		main attention: QKV projection -> Q/K norm + RoPE
		RadixAttention.forward
				hybrid full-attention backend
						QwenSparseAttnBackend.forward_extend
								cache write -> valid-row trim -> packed full-context K/V
								optional FlyDSL adapter -> [valid_rows, local_Q_heads, 256]
								_pad_extend_output -> flatten and restore DP padding
		existing sigmoid output gate -> O projection
```

首版只启用以下交集，其余在**发射新kernel之前**走原分支：

- 主模型的普通 `ForwardMode.EXTEND`、eager prefill，非draft/MTP runner。
	不仅判断`is_extend()`：该函数也包含MIXED、TARGET_VERIFY、SPLIT_PREFILL等模式；
	draft的某些prefill也可能使用普通EXTEND，因此还需检查runner角色。
- ROCm、输入设备架构恰为gfx942；`tensor.is_cuda`不能区分NVIDIA与ROCm。
- `qsa_profile.variant == QSA_VARIANT_COMPRESSED`，压缩比4、block_topk512、token预算2048。
	同一QSA backend也服务tokenwise DSA，不能对它应用四-token块恢复或dense阈值。
- 最终attention实际Q/K/V为BF16、D256、连续三维张量；首版以本地Q12/KV1、Q6/KV1为主，Q3/KV1做功能支持。
	从实际tensor与layer读取头数，不从全局`--tp`猜测；attention TP、DP和DCP可改变实际分片。
- Q/K/V同设备、16-byte对齐，当前实现要求各自byte span小于$2^{31}$；out连续且不与输入重叠。
  无法证明该ABI时不能通过强行reshape、忽略stride或修改descriptor extent来绕过检查。
- 首版不覆盖CP/DCP、交叉层共享导致K/V为空、量化/特殊KV布局、额外score修改、LSE返回等未验证合同。
	FP8权重模型仍可能产生BF16 Q/K/V；按实际输入检查，不按模型名或权重格式推断。
- 不接管decode、TARGET_VERIFY、DRAFT_EXTEND_V2、MTP索引复用、piecewise/breakable prefill graph。
	保留 `forward_extend()` 开头现有的speculative paged早退；空有效batch独立返回正确形状或保持原路径。

### 2. 需要查看或修改的生产边界

以下链接固定到已核对的SGLang版本；迁移到新revision后必须重新确认函数和合同。

| 生产位置 | 集成职责 |
| --- | --- |
| [QSA forward_extend](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1370-L1472) | 推荐首版唯一attention分流位置；保留cache、gather、padding语义 |
| [prefill block选择](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L453-L507) | 当前生成block_indices后展开并删除；第二阶段在这里保留原块索引 |
| [QSA profile](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/config.py) | 区分compressed/tokenwise，读取ratio、budget和index维度 |
| [模型indexer/attention衔接](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/models/qwen4_exp.py#L1483-L1577) | 第二阶段将原block选择显式传给attention，保持现有MTP和stream同步 |
| [RadixAttention及custom-op参数](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/radix_attention.py#L150-L365) | eager kwargs和graph schema是不同边界；新增block字段需显式贯穿，不能假定自动传递 |
| [QSA metadata](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/metadata.py) | 请求归属、序列长度、逻辑位置；压缩K仅供indexer，不是最终attention K/V |
| [主attention头分片](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/models/qwen3_5.py#L999-L1082) | 读取本地Q/KV heads、D、scale，处理attention TP与KV复制差异 |
| [backend组合](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py) | 确认最终full-attention侧为QSA；首版不改linear-attention和decode实现 |

### 3. 第一阶段：保持现有token索引接口，增加内部adapter

生产当前只把 `topk_indices[M,2051]` 传给attention；本实验需要额外的
`block_indices[M,512]`。首个可回滚集成可以在backend内部增加**device端恢复**，
不改模型返回值和RadixAttention签名。

在已验证的compressed/ordinary-extend合同下，令第i个query绝对逻辑位置为p：

$$c_i=\min\left(512,\left\lfloor\frac{p_i+1}{4}\right\rfloor\right),\qquad
B_{i,j}=\begin{cases}\lfloor I_{i,4j}/4\rfloor,&j<c_i\\-1,&j\ge c_i.\end{cases}$$

这里I是现有token索引。前`4*c_i`个有效项是完整块展开，之后才是0..3个尾token。
**不能不加有效完整块数判断就用 `topk_indices[:, :2048:4] // 4`**：短行的尾部紧跟有效块，
会被误认成完整块。恢复kernel还必须检查padding、每四项连续性、块起点对齐等前置合同；
测试对比恢复结果与生产indexer原始block输出，顺序也应保持。

这一方案仅用于当前固定四-token展开布局；上游若改变重排方式或变成tokenwise选择，应拒绝而非猜测。
把恢复成本计入adapter/服务端性能，不只测恢复后的run。它不改变Top-K计算，不省indexer本身。

从生产构造runtime输入的映射如下；不调用实验 `make_inputs()`、不读取实验JSON替代实际模型配置：

| 实验输入字段 | 生产来源/要求 |
| --- | --- |
| `q` | 主attention Q，裁剪到 `M=topk_indices.shape[0]`，reshape为`[M,layer.tp_q_head_num,D]` |
| `k`,`v` | 下一节两分支得到的原始packed K/V，**不是**compressed indexer K，也不是未转换的物理分页cache |
| `indices` | 当前layer/forward的logical `topk_indices`，int32连续，padding仍为-1 |
| `block_indices` | 第一阶段安全恢复，第二阶段由indexer直接传递；int32 `[M,512]` |
| `query_lens` | 当前forward每请求semantic extend长度，必须满足`sum(query_lens)==M` |
| `prefix_lens` | 当前总长度减extend长度，不是chunk编号或RoPE坐标 |
| `cu_q` | semantic extend长度的前缀和，显式输出连续int32 |
| `cu_k`,`kv_lens` | 各请求完整长度的前缀和与长度，显式连续int32 |
| `query_positions` | 每请求 `prefix+arange(extend_len)`，或已核对一致的indexer logical_positions |
| `query_sequence_ids` | packed query所属的请求行编号；不是 `req_pool_indices` 的全局槽号 |
| `max_seqlen_q/k` | host lengths求max，不在每层GPU tensor上`.max().item()` |
| `scale` | `layer.scaling`，当前D256为1/16；不能重复缩放 |
| 模型/profile字段 | 真实HF text config、QSAProfile与layer；不要硬编码TP2/4作为输入形状 |

实验的CaseSpec还携带seed/selection标签，生产adapter不需要它们；宜提取轻量runtime输入结构，
只传tensor、host lengths和已解析profile，不把整个ForwardBatch/ModelRunner交给kernel。
现有生产 `cumsum` 不保证输出int32，adapter必须显式转换，不能只把原tensor名称对接上。
CPU长度镜像缺失时应在本forward边界处理或保持原路径，不为此全局开启decode的CPU同步；
`needs_cpu_seq_lens=False` 不代表当前普通prefill完全不使用host lengths。

#### 无历史prefix分支

保留当前cache写入，直接传本次 `k[:M]`、`v[:M]` 的连续视图；
各请求Q与KV段长度相同，`cu_k=cu_q`，query位置从0开始。
FlyDSL dense分支仍只覆盖每请求前2051个可见token，其他query继续稀疏计算。

#### 有prefix / chunked prefill分支

保留现有layer pool getter和 `req_to_token[request,:seq_len]` 的gather逻辑；
先按照 `save_kv_cache` 约定写入本轮K/V，再收集prefix+当前chunk的完整K/V并拼接。
`cu_q`按新增长度，`cu_k`按完整长度；首个query的位置是P，不是0。
`save_kv_cache=False` 表示遵守调用方已有cache合同，不是可以遗漏新token的K/V。

首版沿用当前gather而不是同时改paged kernel，便于比较。将来直接访问paged cache应作为独立优化，
重新定义物理page表、对齐和buffer bounds，不能把logical token索引当成physical slot。

#### 后端执行伪代码（拟新增适配逻辑，不是已有API）

```text
forward_extend(..., topk_indices):
		preserve existing save_kv_cache handling
		trim semantic query rows; remember original padded row count
		preserve speculative paged and CPU bypasses
		compute real request lengths and packed Q/K/V as in current two branches

		if optional implementation is eligible before importing/launching it:
				recover original complete-block IDs with per-row complete-block counts
				build runtime inputs from this layer/forward, never from synthetic fixtures
				allocate or lease out[M, local_q_heads, D]
				prepare static layout when needed; rebuild current membership/sorted blocks
				execute FlyDSL dense/direct/local-union dispatch on current stream
		else:
				call the unchanged corresponding Triton prefill function

		return existing _pad_extend_output(out, original_query_row_count)
```

Q/K已经由模型做norm和RoPE，adapter不再做投影/归一化/旋转。
输出保持`[M,H,D]`、BF16，交给原 `_pad_extend_output()` 展平为`[padded_rows,H*D]`并补零。
已有sigmoid gate/O projection继续在模型层执行，不能在kernel中重复应用。

### 4. 第二阶段：从indexer传原始block，去掉恢复开销

正确性和回滚验证完成后，在 `select_prefill_tokens()` 的row-chunk循环内，
将每块 `block_indices` 写入本forward的 `[M,512]` 输出buffer，再执行原token展开。
无压缩K时写-1；不能只保留最后一块query的block结果。

推荐显式传递forward-local、layer-local的selection sidecar，例如
`token_indices + block_indices + logical_positions`，同时保留旧Tensor接口供其他backend使用。
需审查indexer、模型attention kwargs、RadixAttention及backend的所有调用者；
第一阶段仅eager，后续支持custom-op时必须修改其schema/fake输出/捕获路径。
当前piecewise边界显式包含token索引，**没有原始block参数**。

不要使用“模块全局last_blocks”或在共享metadata对象上放一个未分层的last-selection字段：
backend的 `get_indexer_metadata(layer_id,...)` 虽接收layer_id，当前可能返回同一forward metadata。
layer间选择会变化，overlap/多stream也会让无作用域的缓冲区被覆盖。
sidecar与输出在消费者完成前保持有效；如果indexer在alt stream上运行，
为新增block tensor保留与原token tensor同等的wait/record_stream生命周期约束。

第一阶段默认保留完整indexer计算和压缩cache更新，即使后续attention选dense。
进一步跳过短前缀评分/Top-K是第三个独立改动，必须保留pending K、RoPE位置和compressed cache状态，
并验证下一chunk及后续decode输出；不能为了dense fast path直接跳过整个indexer。

### 5. 可选依赖、启用开关和回退策略

- 把运行核心及三个MHA helper作为可安装、版本固定的模块或SGLang内vendor代码；
	不依赖运行机器上的PyHIP实验路径、开发容器绝对路径或仅含src的PyHIP wheel。
- 保留许可证与来源。只集成runtime核心、模型适配结构和编译依赖；
	不让生产import基准、pytest、AMD SMI、硬件空闲门禁、实验input生成器或结果审计。
- `implementation`顶层会导入FlyDSL，因此**eligibility/开关判断要在lazy import之前**；
	未启用、CPU/NVIDIA/其他ROCm架构或tokenwise模型不应因缺少FlyDSL而启动失败。
- 建议先加默认关闭的专家级选择开关；如命名 `SGLANG_USE_FLYDSL_QSA_PREFILL`，
	**这只是建议名，当前尚未实现，不能直接用它启动新路径**。
	新SGLang变量必须在Envs注册typed descriptor并通过`.get()`读取，测试用`.override()`。
	若做公开CLI，则在ServerArgs注册namespace metadata并从resolved config读取，不在业务代码直读raw seed。
- 实验中的`mode=auto/direct/union`是新实现内部算法选择，不能与“是否启用可选实现”的部署开关混为一谈。
- 默认关闭：旧路径完全不变。可选启用时，不支持的硬件/profile/ABI或明确缺少可选包可在launch前回退并记录原因。
	若用户要求强制使用新实现，应在初始化/预检时明确报错，不能悄悄使用baseline。
- 不允许`except Exception: old_attention(...)`包住编译和GPU执行：非法metadata、数值失败、device fault必须暴露，
	不能在状态已写入或设备异常后自动重算以掩盖错误。
- idle/PTL检查只属于benchmark验收。生产GPU本来就繁忙，服务运行时不得调用它、修改时钟功率或等待GPU空闲。

### 6. Serving计划与scratch生命周期

首版可每forward重建计划以保证正确，但不要把实验 `prepare()` 原样放到每层热路径后就称为低开销集成：
它包含host metadata构造、buffer分配，dense准备还核对device CU与host长度，可能发生同步。

生产建议拆分：

1. 每forward一次，从已知host lengths生成静态request布局、dense区间、local union/direct tile元数据。
2. 依据device、dtype、真实head数、请求长度/prefix布局、BQ/BN和容量获取独占scratch lease。
	 仅总M相同不足以复用布局，两个batch可有完全不同的请求边界和prefix。
3. 每层拿到本层block选择后，重建动态membership、gate、masks和direct排序，随后运行。
4. 顺序层可在消费者完成后复用scratch；并发forward/stream不得写同一组计划buffer。
	 不永久缓存按shape得到的active位图或排序结果，不把显存地址复用误认为内容没变。
5. 代码/JIT特化缓存与数据计划分开；只缓存验证过的代码身份，不缓存上一次输出。

遵循SGLang的runtime-context资源/stream/buffer生命周期，不修改只读ScheduleBatch来传临时值，
也不把请求级计划塞入永久model配置。每次launch使用输入设备当前stream，host metadata重建与device值更新明确分离。

### 7. CUDA graph / torch.compile 放到独立阶段

当前实验图测试只证明固定layout、固定地址下rebuild+run可重放，
**不证明SGLang的piecewise/breakable prefill已经能接入**。
已核对的SGLang版本特意未把Qwen4-Exp加入breakable prefill支持列表，理由就是host侧QSA metadata。
首版不要移除该限制；decode现有graph路径保持不变。

未来启用时必须同时完成：可捕获的metadata更新、预分配scratch、所有特化warmup、
block sidecar跨custom-op传递、稳定地址与生命周期、每次replay重新构造当前选择，
并测试shape bucket padding、graph fallback以及主/draft各自语义。
不能在capture中编译新特化、做`.tolist()`/`.item()`或动态host分配。

### 8. 验证清单和分阶段合入

| 阶段 | 必须证明的内容 |
| --- | --- |
| 包装与静态能力检查 | 未安装FlyDSL/非gfx942/不同QSA profile时旧路径正常；显式启用状态可观测；CPU import不拉起GPU依赖 |
| 原始选择桥接 | 从生产indexer获取真实token/block，短行恢复不吞尾、不引入重复，跨row chunk保留全部M行，未来token永不出现 |
| 后端数值 | 同一批实际Q/K/V与相同真实选择，对比旧prefill、FP32 oracle；TP2/4主验收，TP8功能；不放宽现有容差 |
| KV与布局 | P0/非0、分多chunk、radix prefix命中、不同请求长度、空query请求、非4/64对齐、DP padding、cache write顺序及save=false合同 |
| 生命周期 | 多层不同选择、连续batch相同M不同prefix、排序buffer刷新、输出alias、非当前GPU/stream、sidecar不串层、不串请求 |
| 未接管路径 | decode、TARGET_VERIFY、DRAFT_EXTEND_V2、draft普通EXTEND、tokenwise DSA均保持原路径和输出；首版禁用的graph/CP/DCP不被误入 |
| 服务精度与性能 | 单机kernel过关后再跑主模型请求、logits/生成质量和长上下文回归；测完整TTFT、throughput、后续decode行为，而非只报kernel TFLOPS |

测试按SGLang [测试指南](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/test/README.md)放置和注册。
建议先添加device桥接与backend单测，再做TP2/4服务测试；不要把实验目录的自定义argparse入口直接当CI测试。
新增或修改KL一致性测试时使用项目已有方法校准，不能借集成理由放宽精度阈值。

性能需拆开报告：indexer评分/Top-K、块恢复或sidecar、KV gather、动态plan/sort、attention、
全部forward以及服务端TTFT/吞吐。当前实验不含KV gather和indexer，不能把1.25–1.48x kernel链加速直接外推成服务加速。
加入分支命中/回退原因、dense/direct/union行数和临时显存统计，确认真实负载选择重合度；
不把SG32 synthetic shared成绩替代生产采样。

推荐合入顺序：**可选包与eligibility → 保持token接口的eager adapter → 真实请求/缓存验证 →
原block sidecar和scratch复用 → 服务端性能验收 → 最后考虑graph、paged直读和indexer短前缀跳算**。
每阶段默认关闭并能恢复旧路径；任何尚未完成的阶段在日志明确标记，不用实验图/合成测试代替生产验收。
