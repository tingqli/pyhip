# gfx942 FP8 block-scale MoE GateUp（FlyDSL）优化记录

记录`moe_gateup_a8w8_blockscale_gfx942*`在MI300X上的测试方法、当前性能、已确认的问题和结论。

## 1. 范围

- 算子：MoE GEMM1，`out[token, slot] = BF16(SiLU(gate) * up)`，gate/up来自同一FP8 GEMM。A为FP8、每(token, K128)一个scale；B为FP8、每(128列, K128)一个scale；路由为aiter `moe_sorting`的256行block格式。
- 形状：MiMo EP8单rank，`I=2048`、`K(hidden)=4096`、32个本地expert、topk=8，FLOPs=`4*T*I*K`。
  - `ep_uniform`：每token一个本地route，每个expert恰好`T/32`行。
  - `ep_random`：每token的本地expert随机，expert负载不均（更接近真实）。
- Kernel（`src/contrib/flydsl/`）：
  - `moe_gateup_a8w8_blockscale_gfx942_8x1`：8x1 wave，A直接读进VGPR，B经LDS。
  - `moe_gateup_a8w8_blockscale_gfx942_lds`：2x4（默认）/4x2 wave，A、B都经LDS；选项`persistent`、`pair_blocks`。
- 测试：`tests/contrib/moe/test_gateup_a8w8_blockscale_gfx942.py`，13组配置×5个shape，共65例。

## 2. 测试方法

- **先确认GPU空闲**：`rocm-smi --showpids`、`amd-smi monitor -u -p`。其他容器占用时结果不可用：2026-11-02测32K时8张卡都被别的进程跑满（~740 W），median和best相差约2倍。
- **冷cache计时**：每次launch前顺序读一遍1 GB buffer，清掉L2和MALL；多个配置交错运行，正反顺序各3轮×16次，取中位数。同一配置两次运行之间约有±1%的波动。
- **时钟与效率**：在kernel内用`s_memtime`（shader时钟）和`s_memrealtime`（100 MHz）打点，得到真实时钟、CU忙碌比例和MFMA效率。amd-smi读到的频率偏高（1.05 GHz），不能直接用。
- **ATT**：8个SE各取1个CU，dispatch前做只读flush；按"上一个任务最后一条MFMA→下一个任务第一条MFMA"统计任务切换时间。

## 3. 当前性能（2x4，冷cache，ep_uniform）

| T | 时间 | 有效TFLOPS | 说明 |
|---|---:|---:|---|
| 4096 | 317 µs | 433 | 每个256行block有一半是padding |
| 8192 | 368–375 µs | 735–746 | |
| 16384 | 668–672 µs | 818–823 | `pair_blocks=True`：655 µs、839 TF |
| 32768 | 1221–1228 µs | 896–901 | 8x1、2x4、`pair_blocks`、`persistent`互相差别在噪声内 |

`ep_random`：8K为460 µs、598 TF（47个block，均分时为32个），16K为800 µs、686 TF（79个block，均分时为64个）。

32K时每个expert有4个block、每个CU约6.7个任务：冷cache的首轮影响被摊薄，2x4相对8x1不再有优势；同一expert的相邻block在原来的block顺序下本来就几乎同时执行，`pair_blocks`也没有额外收益。

## 4. 问题与结论

### 4.1 750 W功耗墙是第一限制

- 满载时kernel内实测shader时钟只有0.83–1.0 GHz（标称2.1 GHz），按实际时钟算的峰值约1.1 PF。
- 8K有效TFLOPS分解：2615 TF（标称峰值）×0.415（时钟）×0.855（CU忙碌比例）×0.76（忙碌时间内的MFMA效率）≈705 TF。
- 推论：消除空闲、隐藏延迟几乎不提速。省下的是低功耗的等待周期，平均功耗上升后频率随之下降。只有少算、少搬数据，降低每个任务的能耗，才能提速。

### 4.2 Wave排列

- 2x4主循环MFMA效率91–93%。
- 8x1在冷cache 8K只有80–86%：A按16行×64 B加载，相隔4 KB的行落在L1同一组，tag冲突造成访存发射背压。
- 4x2上限约86%：barrier等待加buffer_load发射卡顿。

### 4.3 输出store与TLB

- 98%的UTCL1 miss来自对`[T, topk, I]`的按token散写。每个WG写的每个不同行（页）都会miss，与单条store宽度无关。
- 128 B store（每条指令8行×128 B）：miss减少17%，时间改善≤1%。
- 先在LDS里重排、按256 B行写出：miss只减少3%，时间反而+1–1.4%（bank冲突和额外barrier），已删除。
- 按排序后行顺序写出（`sorted_rows`）能快3.2%，但输出布局不符合API，已撤回。
- 上限：用`num_records=0`丢弃全部store，只快0.6%/1.7%/2.8%（4K/8K/16K）。

### 4.4 Epilogue

- ATT（8K）：每个wave最后一条MFMA之后约10k cycle。
  - SiLU的VALU 5.1k，其中exp/rcp 3.0k；同一SIMD上两个wave争VALU又多1.8k。
  - store发射卡顿1.45k。
  - `s_endpgm`隐式等所有store写完1.7k。
- 上限：同时去掉SiLU和store，快3.7%/3.1%/5.5%。

### 4.5 常驻kernel（`persistent=True`）

- 结果：任务切换时间缩短（16K从20.3k降到16.9k cycle，8K从22.6k降到约18k），但冷cache和连续运行的墙钟时间都不变，原因见4.1。默认关闭。
- 实现中的坑：
  - 下一任务的读必须在任何输出store之前发出。TLB miss的store会堵住CU按序处理的访存队列，排在后面的读被卡约3.5k cycle。
  - store不能堆到全部SiLU之后（发射卡顿约9k cycle），要和SiLU交错。
  - 任务循环里要用不透明的thread id重建lane常量，否则LLVM把它们外提到循环外，K循环里多占寄存器（256→244 VGPR）。
  - 把最后一个任务剥到循环外，LLVM会把uniform的循环出口值放进VGPR，产生waterfall loop和spill。改为先剥出第一个任务的K循环（旋转循环）。

### 4.6 Padding

- `moe_sorting(256)`把每个expert补齐到256行。`ep_uniform`在8K/16K恰好没有padding，掩盖了这个问题。
- 真实负载下（按multinomial模拟），MFMA处理的行数是有效行数的2.00×/1.48×/1.25×（4K/8K/16K）；按64行跳过全padding的块，可降到1.24×/1.12×/1.06×。
- 实测`ep_random`比`ep_uniform`慢23%（8K）/20%（16K）。
- 4K跳过padding的上限只有−13.7%，不是2倍：padding行的A为零，这些MFMA本来就省电；而且512 MB的B读取占主导。

### 4.7 B权重流式读取

- 消融（所有block都用expert 0的权重）：8K快13.8%，16K快7.7%。
- PMC：8K每块B只从显存读一次，已是最少。16K写进L2约803 MB，最少只需512 MB，因为同一expert的两个block没有总是同时执行。
- `pair_blocks=True`：相邻且属于同一expert的两个block按N tile交错执行，在L2共享B。
  - 16K `ep_uniform`快2.2–2.6%（L2 miss从7.37M降到5.54M）；`ep_random` 16K快1.0%；8K不变。
  - 不检查expert的版本在`ep_random`下反而慢1–1.5%：成对的block跨expert，把原本相邻的同expert block拆开了。

### 4.8 反量化与GEMM核心

- 去掉每个K block的反量化FMA，上限−4.1%（8K）/−3.2%（16K）。
- B常驻cache时8K为853 TF；同尺寸hipBLASLt dense per-tensor FP8为889 TF（warm）。核心计算只差约4%，主要就是反量化。

## 5. 待做

1. 按64行跳过全padding的块（真实负载收益最大；4K上限−13.7%）。
2. 反量化改用`v_pk_fma_f32`（上限3–4%，预计1.5–2%）。
3. SiLU用packed指令（预计0.5–1%）。
4. 决定`pair_blocks`是否默认开启。
