# TODO

## Attention JIT 后续

- [x] 用ATT按物理SIMD去重并按PC建立cycle账本：ATT内部442.196 cycle/tile + ATT外墙钟边界15.546 cycle/tile = 457.743 cycle/tile，闭合H=1/M=N=40960无尾批JIT 188.02T。
- [x] 将MFMA未隐藏与全流水线no-issue分成互斥矩阵：768逻辑shadow = 90.294重叠 + 255.435已隐藏 + 86.923 MFMA-only + 335.348 no-issue；shadow外另有399.370 no-issue。
- [x] 保持MFMA顺序扫描prepare/center/finish边界：串行归约阶段固定5/3/8，三路fanout后重调为4/3/9；shadow内VALU最终增至261.448 cycle/tile。
- [x] 用max/sum三路DS fanout替代串行双级归约：一次wait后合并，physical no-issue从679.221降至633.294 cycle/tile。
- [x] 验证V跨tile寄存器双缓冲：154 VGPR+96 AGPR仍为2 waves/SIMD且零spill，但40960回退到196.9--197.0T，未保留。
- [x] 不增加barrier数量，将两条next-K LDS写分别放到softmax1归约与`lgkmcnt(1)`之间；最终达到4415.4 us/194.54T。
- [x] 验证DPP `wave_rol`列向max/add：数学与精度正确，但需384条rotate+392个hazard NOP，40960仅114.6T，生产路径回退到DS。
- [x] 验证三次`wave_shr+max`再`readlane`：lane20实测104而列向期望304，readlane仅得到全wave标量315，当前布局不正确。
- [x] 将max/sum跨row归约改为xor16/xor32/xor48三路DS fanout，一次wait后合并；最终4/3/9达到4289.5 us/200.3T。
- [x] 验证lane内max/sum平衡树：max稳定回退，sum收益落在噪声区间，均未保留。
- [x] 将一个`n_block`的BF16概率打包移到sum DS wait前，另一个保留在running-sum FMA后；最终4225.9--4230.1 us/203.1--203.3T，physical no-issue降至611.249 cycle/tile。
- [x] 验证两个`n_block`全部前移：199.1--199.2T，因填窗过量延迟sum关键路径而回退。
- [x] 重新扫描半量打包后的相邻4/4/8、5/3/8、3/3/10配额；4/3/9仍为稳定最快路径。
- [x] 将scale/round/lazy常量改为VGPR以使用e32编码,并将pair循环改为字节offset+尾部条件分支+滚动offset；最终4180.9--4183.7 us/205.3--205.5T。
- [x] 将下一tile的K LDS读改成`MFMA -> DSRD -> MFMA`,ATT关键路径-3.625、physical idle -3.329、shadow no-issue -5.314 cycle/tile。
- [x] 将softmax1四条lane-local max放到`vmcnt(10)`和`vmcnt(2)`之间；40960达到4160.3--4161.5 us/206.4--206.5T。
- [x] 实测双MFMA混合bundle：`MFMA -> 3 ALU -> MFMA -> EXP`为36.053 cycle,相对0 ALU仅+0.013；只在center pair23保留一组,40960达到4124.7--4132.3 us/207.9--208.3T。
- [x] 验证全局6组3-ALU+EXP pair：精度/资源/静态指令不变但回退到200.7--201.0T,未保留。
- [x] 验证第二个局部pair（correction select/copy + p0 EXP）：单次206.3T,破坏probability EXP相位,未保留。
- [x] 用K写stage地址原地XOR替代写前地址ADD,在GEMM1 group1/3填两个空窗；再将V offset滚动移到group7,40960达到4114.9--4117.1 us/208.6--208.8T。
- [x] 验证GEMM1填窗候选：V load中置/lookahead、future-K提前、threshold/max预计算、全offset滚动和split-K双accumulator均中性或回退,未保留。
- [x] 验证center `J.emit`预算10→11：会多取完整第三条5-cycle FMA并回退到169.1T,不能用作细粒度填窗。
- [x] **空闲GPU复测最佳半wavefront并填写最终结果**：发现650W功耗上限导致production在4.13/6.20 ms间切换，阶段中位数无效；改用紧邻control配对。GPU 3上50轮：全体-0.23%，fast-state +0.76%（14轮），slow-state-0.29%；收益不稳定且明显低于最终setprio方案，未保留。结果已回填§22.21/§22.23和实验JSON。
- [x] **验证并迭代独立`s_setprio`双resident-wave流水线，超过production**：粗phase消融证明priority本身相对改善6.47%；扫描15块掩码、33个统一/邻域边界、16个per-block边界后，发现多次切换成本和高优先级窗口不连续是主因。最终每tile只切换一次：GEMM1 mt0第8条MFMA后`setprio(1)`，跨softmax0/GEMM1 mt1/softmax1，到GEMM2 mt0第16条MFMA后`setprio(0)`，GEMM2 mt1保持normal。8张冷态AUTO GPU共40/40有效配对：时间比0.90002，**+11.11%**，候选236.63T、局部control212.95T；标准GPU 3、10 buffers/50样本中位236.82T，清理后当前源码复测**3623.5 us / 237.1T**。随机40960与production逐元素相同，参考`rel_l2=0.00318646`；全1为0；156 VGPR+64 AGPR、2 waves、零spill。生产入口不变，保留独立`attn_gemm_jit_setprio_best`，最终汇编归档`archive/gemm/attn-gemm-jit-setprio-best-gfx942-m40960-n40960-237p1t.s`。
- [ ] 空闲GPU上夹心扫描softmax1 prepare预执行31/35/43 cycles（再前移DS fanout/threshold/K写）；小shape精度和资源已通过,当前未进入生产文件。
- [x] 将JIT优化移植到FlyDSL：第二条K写中置和按m_rep半量BF16转换,40960从185.1T提升到194.4T。
- [x] 修复FlyDSL移植对8192的shape回退：`N < 32768`恢复原scheduler+整片转换,8192从166.9--167.0T恢复到170.6--170.7T；40960保持194.0--194.1T。
- [x] **严格Fly最终ISA达到205.2T阶段门槛**：只处理当前`flyc.compile(build(...))`生成的ISA，删除8条identity max、把12条BF16 pack移入sum等待窗，并插入绝对priority事件`16:0,96:2`。`C-X-X-C`为12/12有效，约194.6T→205.2T（+5.42%），`rel_l2=0.00319`；ISA形态漂移会失败，不使用JIT归档机器码冒充Fly codegen。
- [x] **系统验证Fly源码级inline asm表达**：覆盖side-effect/pure max、概率pack anchor、原生`rocdl.s_setprio`、tied累加器、选择性调度屏障，以及2条+16条MFMA整块inline。所有40960候选均逐元素一致，但最佳精确`16:0,96:2`版本仍回退5.30%，最接近最终ISA的2+16块版本回退8.76%；源码探针已全部清理，正式205.2T最终ISA路径不变。
- [x] **扩大inline到完整JIT主流程**：Fly保留tensor ABI、block/head base offset和launch，单个asm块覆盖wave内offset、prologue、完整pair-loop、两次`kv_step`等价流程及epilogue/store。最终220V/34S/0A、零spill、128 MFMA/4 priority；与静态JIT oracle逐元素完全相同。严格配对高层Fly 194.5T→inline 236.4T（+21.61%），oracle 236.3T→inline 236.2T（-0.02%，噪声内）。该路径主要计算来自受SHA保护的JIT归档，不计为高层Fly原生codegen。
- [x] **验证sum归约后移的`v_pk`阻塞并用inline消除**：只后移shuffle生成2条`v_pk_fma`，inline scalar FMA将254V降到232V并提升4.16%（182.0T→189.5T）；完整后移生成额外13条`v_pk_mul`、5条`v_pk_add`且268V跨occupancy阈值，逐元素inline FMA将其降到246V并提升33.02%（143.6T→191.1T）。两者相对base仍分别回退2.54%和1.96%，默认保持base。
- [x] **测试两种inline sum组合延后`sm_scale_log2`**：逐元素inline `v_fma_f32(score,scale,-max)`删除16条`v_pk_mul`和32条SUB；只后移shuffle组合跨GPU稳定+0.29%--0.30%，完整后移sum组合稳定-0.83%--0.84%。前者相对base仍回退2.24%，不切换默认路径。
- [x] **用ATT闭合late-scale回退**：完整后移sum的late-scale物理wall增加14.049 cycles/tile（+0.85%），与严格性能-0.83%闭合；issue减少7.730但shadow外no-issue增加24.121。slot0快116.680，决定吞吐的slot1却慢28.097；phase32/96长softmax缩短后，等待转移到下一GEMM1的K-LDS progressive wait（慢slot phase34/42/106/108分别+86.427/+76.205/+124.341/+43.776）。根因是双resident-slot失衡和K LDS延迟暴露，不是scalar FMA吞吐。
- [x] 拆分验证raw/formal probability交织：raw完整组合相对205T回退5.18%；单步formal完整接管回退9.67%，两步对称接管收窄到6.91%，raw-domain FMA接管回退8.59%。对称接管ATT虽使shadow内no-issue减少50.340 cycles/tile，shadow外no-issue仍增加114.829，phase32双wave长团重叠从250.356升到445.208。priority第二边界80--112、M16/M17交换、回边GEMM2链旋转、max-only fanout和删6条冗余K wait均未转正，所有候选未保留。
- [ ] **FlyDSL codegen达到最新JIT约237T**：当前高层Fly约194.6T，严格来源于Fly的最终ISA后处理约205.2T，距离JIT ISA oracle约32T。下一步必须按完整pair重排两路softmax：p0与当前GEMM1交织，p1跨循环回边与下一轮GEMM2交织；禁止继续只搬p0局部FMA/EXP，因为这会把两resident wave锁得更同相。
- [ ] 在存在尾批的实际shape上评估persistent grid；H=1/M=40960理论示例已有320 WG整除80 CU，不需要尾批修正。
- [ ] 单独验证xor32地址复用；raw-domain FMA、probability寄存器复用、max fanout和K wait压缩已完成严格消融并回退。

## 当前任务

- [x] 扩展`archive/gemm/analyze-kernel-mfma-valu-coissue.py`，单次运行同时报告intra co-issue和8-wave inter co-issue，并更新实测报告。
- [x] 新增`archive/gemm/analyze-attn-att-cycle-ledger.py`，将最终ATT的VMEM/LDS/barrier/wait和依赖空洞归入BN32 tile并闭合墙钟。

> [提醒] JIT production为208.6--208.8T；最新独立`setprio_best`为236.5--237.1T，是Fly移植目标。FlyDSL高层长序列约194.6T，严格来源于Fly的最终ISA组合约205.2T；8192约170.6--170.7T。
