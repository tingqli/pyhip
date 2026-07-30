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
- [x] **Fly ABI机器后端达到最新JIT约237T**：将归档JIT的`156V+64A`机械重命名为`220V+0A`，12/12配对均为236.56T且输出逐元素相同；再转换为Fly 164-byte tensor ABI，实测3630.9us/236.6T、`rel_l2=0.00319`。入口为`ATTN_FLY_BACKEND=jit_all_vgpr`，严格配对Fly DSL 194.1T→机器后端236.1T（+21.59%）。机器后端目标已完成；FlyDSL codegen路径仍需对齐MFMA链、wait和resident-wave相位。
- [ ] 在存在尾批的实际shape上评估persistent grid；H=1/M=40960理论示例已有320 WG整除80 CU，不需要尾批修正。
- [ ] 拆分验证raw-domain FMA、probability寄存器复用、xor32地址复用各自的绝对性能贡献。

## 当前任务

- [x] 扩展`archive/gemm/analyze-kernel-mfma-valu-coissue.py`，单次运行同时报告intra co-issue和8-wave inter co-issue，并更新实测报告。
- [x] 新增`archive/gemm/analyze-attn-att-cycle-ledger.py`，将最终ATT的VMEM/LDS/barrier/wait和依赖空洞归入BN32 tile并闭合墙钟。

> [提醒] JIT production为208.6--208.8T；最新独立`setprio_best`为236.5--237.1T，是Fly移植目标。FlyDSL长序列基线约194.0--194.4T，纯Fly最终ISA短priority窗口约197.6--198.5T；8192约170.6--170.7T。
