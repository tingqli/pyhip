# Paged Prefill 4-wave 优化记录

状态：ACTIVE

目标是在 gfx942 上实现 MiMo FP8 paged-prefill 的 4-wave pipeline，并保持与
`pa_8wave` 相同的调用接口、分页语义、GQA、ragged last page 和 bottom-right
causal mask。当前同时支持FP8和BF16 Q/K/V，输出均为BF16。主验收 shape 为：

- 长 non-causal：`batch=1, qo_len=10240, kv_len=2583`；
- 长 causal：`batch=1, qo_len=32768, kv_len=32768`；
- `num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128, page_size=32`；
- 4-wave 中位延迟不超过同卡 8-wave 的 110%。

需要的 waitcnt、FP8/BF16转换、intrinsic 和 layout helper 均使用 FlyDSL/ROCDL 公共 API 或在 kernel 文件内
局部实现。

## 复现约定

从仓库根目录运行。性能测试前先用 `rocm-smi --showuse` 选择空闲 GPU，并为每轮
测试使用新的 `PYTHONPYCACHEPREFIX`，同时设置
`FLYDSL_RUNTIME_ENABLE_CACHE=0`。正确性要求：

- 输出全部 finite，NaN sentinel 不得残留；
- `pyhip.allclose(..., rtol=0.1, atol=0.1)` 通过；
- `pyhip.calc_diff(output, reference) < 0.001`。

功能回归：

```bash
HIP_VISIBLE_DEVICES=0 PA_CASE=all PA_NUM_ITERS=1 \
PYTHONPYCACHEPREFIX=/tmp/pa4-functional-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

BF16功能回归在命令中增加`PA_DTYPE=bf16`。例如：

```bash
HIP_VISIBLE_DEVICES=0 PA_DTYPE=bf16 PA_CASE=all PA_NUM_ITERS=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

正式性能使用 10 套 buffer、10 次预热、50 个 CUDA event 样本的中位数：

```bash
HIP_VISIBLE_DEVICES=0 PA_CASE=noncausal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 \
PYTHONPYCACHEPREFIX=/tmp/pa4-formal-noncausal-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py

HIP_VISIBLE_DEVICES=0 PA_CASE=causal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 \
PYTHONPYCACHEPREFIX=/tmp/pa4-formal-causal-pyc \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

BF16参考分支验收使用`H=1, Dq=Dv=128, qo_len=kv_len=40960`：

```bash
HIP_VISIBLE_DEVICES=0 PA_DTYPE=bf16 PA_CASE=bf16_ref PA_NUM_ITERS=1 \
PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

对应的小shape精度入口为`PA_CASE=bf16_ref_short`。

BF16正式性能同样使用10套buffer、10次预热和50个CUDA event样本的中位数：

```bash
HIP_VISIBLE_DEVICES=0 PA_DTYPE=bf16 PA_CASE=noncausal PA_NUM_ITERS=1 \
PA_FORMAL_BENCH=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

完整causal reference需要约64GB额外临时显存；在小causal已经完成精度验证后，可对
长causal设置`PA_SKIP_REFERENCE=1`只做finite检查和4-wave性能测量。

每次实验按以下格式增量追加，不删除失败项：

```markdown
### YYYY-MM-DD：实验名

- 假设：
- 改动：
- 正确性：
- ISA：VGPR / SGPR / LDS / scratch / MFMA / setprio：
- 性能：control / candidate / time ratio：
- 结论：采纳 / 失败 / 中性 / 待验证：
- 产物：
```

## 性能快照

| 实现 | shape | 中位延迟 | TFLOPS | diff | 备注 |
|---|---|---:|---:|---:|---|
| **BF16 4-wave D128 max-overlap** | non-causal, `H=1, 40960 x 40960` | **3415.611 us** | **251.49** | short 0.00000246 | 50样本正式中位数；同机严格夹心加速5.69% |
| **BF16 4-wave D192 raw-max** | non-causal, `Hq=16,Hkv=1,10240 x 2583` | **1322.365 us** | **204.820** | 0.00000251 | 页状态清理后50样本正式中位数 |
| **FP8 4-wave最终** | non-causal, `Hq=16,Hkv=1,10240 x 2583` | **750.723 us** | **360.782** | 0.000367 | raw-max + split8，50样本中位数 |
| **FP8 4-wave最终** | causal, `Hq=16,Hkv=1,32768 x 32768` | **15962.581 us** | **688.806** | small 0.000175 | raw-max + split8，50样本中位数 |

最终静态 kernel 资源：`44 VGPR + 132 AGPR / 112 SGPR / 16KB LDS / 0 scratch`，
combined allocation 176，实际2 waves/SIMD；静态 ISA 有80条FP8 MMA。batch=1走静态
grid；batch>1保留atomic-ticket persistent调度。

BF16 D128静态kernel资源为`88 VGPR + 128 AGPR / 92 SGPR / 17KB LDS / 0 scratch`；
BF16 D192为`250 VGPR-form / 89 SGPR / 25KB LDS / 0 scratch`。两者均无spill，实际
2 waves/SIMD。BF16使用`MFMA(32,32,8)`、128-bit Q/K/V copy和每行8元素的K LDS
padding；D192 padding把384B行跨度改为400B，消除行首固定落到同一LDS bank的问题，
同时避免通用swizzle布局引起的VGPR膨胀。

## 增量实验日志

### 2026-08-09：8-wave 基线

- 假设：现有 8-wave 实现可作为同机、同接口的性能与精度 control。
- 改动：无；运行 `tests/flydsl/pa_8wave/test_pa_prefill.py`。
- 正确性：全部短 KV、ragged page、batch=4、长 non-causal 和长 causal 用例通过；
  两个主 shape 的 diff 分别为 `0.000367` 和 `0.000268`。
- ISA：本轮只建立 control，未重新统计资源。
- 性能：长 non-causal 为 `824 us / 328.7 TFLOPS`；batch=4 回归为
  `3142 us / 341.7 TFLOPS`；长 causal 为 `15025 us / 365.9 TFLOPS`。
- 结论：采纳为本轮 control；两个主 shape 的 4-wave 门槛分别为 `<=906 us` 和
  `<=16528 us`。
- 产物：`/tmp/pa-8wave-baseline-pyc`。

#### 本次性能快照

| 实现 | shape | 延迟 | TFLOPS | 备注 |
|---|---|---:|---:|---|
| 8-wave baseline | non-causal, `10240 x 2583` | 824 us | 328.7 | 主验收control |
| 8-wave baseline | batch=4, `10240 x 2560` | 3142 us | 341.7 | persistent负载均衡control |
| 8-wave baseline | causal, `32768 x 32768` | 15025 us | 365.9 | 主验收control |

### 2026-08-09：4-wave FP8/MMA32 功能骨架

- 假设：`mha-fp8-d192` 的 `BM=128 / BN=32 / 4 waves / MMA32 FP8` 数据流可与
  8-wave 的 paged ABI、GQA、ragged tail 和 bottom-right causal mask组合。
- 改动：K 走 LDS ping-pong，V 从 paged buffer 直入 fragment，score/probability
  留在寄存器；全部 copy/MMA/waitcnt/intrinsic 使用 FlyDSL/ROCDL API或文件内实现，
  不使用 `pyhip.contrib.flydsl.helpers`。
- 正确性：`kv_len=1/3/13/23/32/53/83` 与跨页用例通过。
- 关键修复：K offset 原以 FP8 字节计算，却在加 offset 后 recast 为 U32，实际地址被
  放大4倍，表现为8组 score、每组重复4次；改为先 recast完整K tensor，再使用
  `source_offset // 4`。
- 结论：采纳，建立可调优基线。

### 2026-08-09：stage0/stage1 旋转与跨回边 scheduler

- 假设：删除 8-wave 双 wave-group barrier结构，并让 K global prefetch/softmax 与
  K LDS write/PV/K LDS read 分处两个跨回边 stage，可恢复4-wave resident-wave反相。
- 改动：每 tile 只保留 K 可见性 barrier；scheduler按 Dq=192/Dv=128 的实际
  `12 QK MFMA / 4 V loads / 8 PV MFMA / 6 K reads` 配额，并加入参考D192的
  `VMEM/DSWR/MFMA/DSRD` 跨回边窗口。
- 正确性：短、跨页和长 non-causal diff 不变。
- 性能：初始串行约 `1838 us`，旋转后约 `1465 us`，精确跨回边 scheduler 后约
  `1008 us`。
- 结论：采纳。

#### 本次性能快照

| 阶段 | non-causal延迟 | 相对初始 |
|---|---:|---:|
| 初始串行 | 1838 us | 1.000x |
| stage旋转 | 1465 us | 0.797x |
| 跨回边scheduler | 1008 us | 0.548x |

### 2026-08-09：ATT分析与C-shuffle

- ATT基线：`/tmp/pa4-att/ui_output_agent_1771_dispatch_83`；2039条指令，99.95%
  源码映射，`60V+132A / 112 SGPR / 12KB LDS / 0 scratch`，2 waves/SIMD。
- ATT基线 stall：总计36.83M（63.9%）；MFMA 30.0%、LDS 21.6%、LDS wait
  10.1%、barrier 6.2%、VMEM store 5.9%。
- 改动：先实现完整32KB、128-bit C-shuffle；再按D方向分两次64x128半块C-shuffle，
  将O LDS降到16KB，同时保留128-bit HBM store。
- C-shuffle ATT：`/tmp/pa4-cshuffle-att/ui_output_agent_7158_dispatch_83`；总stall
  28.52M（57.0%），VMEM store从5.9%降到0.4%。
- 性能：完整C-shuffle约 `964 us`；半块C-shuffle约 `915 us`。
- 结论：采纳半块C-shuffle。

#### 本次性能快照

| 实现 | non-causal延迟 | 备注 |
|---|---:|---|
| 完整C-shuffle | 964 us | 32KB LDS |
| 半块C-shuffle | 915 us | 16KB LDS，胜出 |

### 2026-08-09：静态派发、3-wave occupancy与host开销

- 假设：batch=1无需 persistent外层 while/atomic/mailbox；静态kernel可同时降低
  host开销和寄存器压力。
- 改动：提取单work-item helper；batch=1直接映射block到 `(head, query_tile)`；
  batch>1保留persistent kernel。静态路径复用一个dummy counter，不再每次分配/清零。
- ISA：单K历史版本从persistent的约`192 combined VGPR`降到静态`168 VGPR`，配合
  16KB LDS达到3 waves/SIMD；最终双K稳定反相版本为`44V+132A`、combined 176、
  2 waves/SIMD。最后一个仅为persistent复用所需的C-shuffle barrier在静态路径裁掉。
- 性能：short host+kernel调用从约54us降到约10us；最终双Knon-causal正式中位数
  `874.162 us / 309.836 TFLOPS`。
- 结论：采纳。

#### 本次性能快照

| 实现 | shape | 延迟 | TFLOPS | 备注 |
|---|---|---:|---:|---|
| 静态4-wave双K | non-causal, `10240 x 2583` | 874.162 us | 309.836 | 50样本中位数 |
| persistent 4-wave | batch=4, `10240 x 2560` | 3647 us | 294.4 | 非主验收路径 |
| 8-wave control | batch=4, `10240 x 2560` | 3142 us | 341.7 | 同shape control |

### 2026-08-09：causal 80-CU仿射负载均衡

- 假设：自然query-tile顺序导致静态causal尾部集中在重tile；按80 CU轮转模型均衡每CU
  累计page数可保留3-wave静态kernel并消除尾部拖延。
- 改动：当 `works_per_head == 256` 时使用 `tile=(251*physical_tile+251)%256`
  的一一仿射排列；其他shape使用通用轻重交错映射。
- 正确性：`256x256`、`384x384`和`32768x32768` causal均通过。
- 性能：自然静态约17.9ms；轻重交错约16.7ms；最终双K仿射映射正式中位数
  `16720.924 us / 328.783 TFLOPS`。
- 结论：采纳。

#### 本次性能快照

| causal调度 | `32768 x 32768`延迟 | TFLOPS | 备注 |
|---|---:|---:|---|
| 自然静态顺序 | 约17.9 ms | - | 重tile尾部集中 |
| 轻重交错 | 约16.7 ms | - | 中间方案 |
| 80-CU仿射映射 | 16720.924 us | 328.783 | 50样本中位数，胜出 |

### 2026-08-09：失败与中性实验

- K copy改为128-bit（前128线程或四波均衡两段）：正确，但长shape回退约2.7%到
  10.7%；64-bit DS写虽在ATT中stall高，却参与resident-wave相位，回退。
- 本轮在raw-max+split8最终主线上重新实现128-bit K copy：每页`32*192/16=384`个
  16B atom。测试了四个wave各前32 lane执行第二轮，以及前128线程执行第二轮；两者
  与64b control逐元素一致，但分别回退约15.5%和15.4%。候选ISA中K路径由
  `buffer_load_dwordx2/ds_write_b64`变为`buffer_load_dwordx4/ds_write_b128`，wait
  78降至67，VGPR从172升至176但occupancy仍为2 waves/SIMD。
- 128b ATT显示部分线程第二轮copy触发VMEM reconvergence：两条隐式`vmcnt(0)`合计约
  13.3M stall，最热一条约1142 cycles/次；总stall/MFMA从49.606升至81.839，首尾10%
  pairwise反相从79.5%降至57.3%。让全部256线程执行第二轮128b load、后128线程读取
  重复atom后，回退缩小到5.0%，但额外读取2KB/page且LDS写仍需半线程mask，仍未胜出。
- K LDS write简单后移：单页通过但长序列错误；根因是覆盖仍被后续消费的K prefetch
  fragment。改为正确current/next角色后精度恢复，但墙钟中性。
- probability乘240+exact rebase：精度略改善但性能回退；主线保持与8-wave一致的
  lazy rebase和未缩放FP8 probability。
- vector running-sum：精度不变但扩大loop-carried状态，无稳定收益，回退。
- direct 64-bit output store：LDS降到12KB、VGPR降到170，但失去C-shuffle，性能回退
  到约1003us；半块C-shuffle更优。
- causal persistent ticket：约18.9ms；静态自然顺序约17.9ms；每head轻重相邻映射
  严重回退到24.6ms。均已回退。

#### 本次性能快照

| 失败/中性候选 | 延迟或回退 | 结论 |
|---|---:|---|
| K copy 128-bit | +2.7%--10.7% | 回退 |
| K copy 128-bit，四wave各半wave第二轮 | 870.003 us / 311.317T，+15.5% | 回退 |
| K copy 128-bit，前128线程第二轮 | 867.323 us / 312.279T，+15.4% | 回退 |
| K copy 128-bit，全线程重复第二轮load | 791.004 us / 342.409T，+5.0% | 回退 |
| K copy 64-bit control | 约752--753 us / 359--360T | 保留 |
| direct 64-bit output store | 约1003 us | 回退 |
| causal persistent ticket | 约18.9 ms | 回退 |
| causal每head轻重相邻 | 约24.6 ms | 回退 |

### 2026-08-09：最终验收

- 功能：全部输出finite；ragged page、反转page table、GQA、batch=4、长non-causal、
  bottom-right causal均通过；主shape diff分别为`0.000367/0.000268`。
- 正式口径：10套buffer、10次预热、50个CUDA event样本中位数。
- non-causal同卡：`874.162 / 821.804 = 1.0637`，4-wave慢6.37%。
- causal同卡：`16720.924 / 17035.219 = 0.9816`，4-wave快1.84%。8-wave在该次
  正式测量中存在DPM双态，因此以用户确认的中位数口径判定。
- 结论：两条主验收均满足4-wave不慢于8-wave超过10%的目标。

#### 本次性能快照

| 实现 | shape | 延迟 | TFLOPS | diff |
|---|---|---:|---:|---:|
| 4-wave双K | non-causal, `10240 x 2583` | 874.162 us | 309.836 | 0.000367 |
| 8-wave control | non-causal, `10240 x 2583` | 821.804 us | 329.576 | 0.000367 |
| 4-wave双K | causal, `32768 x 32768` | 16720.924 us | 328.783 | 0.000268 |
| 8-wave control | causal, `32768 x 32768` | 17035.219 us | 322.717 | 0.000268 |

### 2026-08-09：双K预取流水与稳态反相复验

- 流水：循环前只预取K。prologue执行`K0 global->reg->LDS`和`K1 global->reg`，V到
  `kv_step`才首次加载；stage0先把`K(i+2)`加载到独立next fragment，再做softmax，
  最后把current fragment中的`K(i+1)`写入LDS；stage1执行PV、barrier和下一K的LDS读，
  并跨runtime loop回边延续到下一次V load/QK。该顺序与`mha-fp8-d192`参考一致。
- 判据：使用long non-causal ATT，以实际`s_setprio`事件划分stage0/1；在整个dispatch
  时间轴上删除首尾各10%，再按每个物理SE/SIMD积分两个resident slot处于异stage的时间。
- 单K、3-wave基线：`/tmp/pa4-current-att/ui_output_agent_17080_dispatch_79`。
  三slot任意异stage占比中位74.7%，但pairwise反相仅52.4%--54.3%，接近独立随机相位，
  判定为不稳定反相。
- hardware-slot priority：按参考方法令slot0使用stage0/1=`1/3`、其他slot使用`0/2`。
  3-wave下pairwise反相仅52.1%--56.0%；强制真实2-wave后也只有约55.0%。
- 双K参考流水+hardware-slot priority：
  `/tmp/pa4-refpipe-att/ui_output_agent_47311_dispatch_79`，真实2-wave；反相率中位65.5%，
  十等分63.9%--68.5%，边界skew中位约69.9万cycles，性能约1030us，仍未稳定锁相。
- 双K参考流水+统一priority：
  `/tmp/pa4-refpipe-unified-att/ui_output_agent_5273_dispatch_79`，真实2-wave；删除首尾10%
  后16个物理SIMD反相率为78.8%--88.7%，中位83.9%，十等分稳定在81.3%--87.4%；
  总stall为27.08M（56.3%），long non-causal稳定约876--880us。
- 结论：采纳双K参考流水和`set_stage0_priority()=0`、`set_stage1_priority()=2`。
  `mha-fp8-d192`中的硬件slot分级方法原本只对`MMA32 && BF16 && D128`启用，直接套到
  当前FP8 D192会形成正反馈并回退，因此按ATT反证移除slot分支。

#### 本次性能快照

| 流水/priority | non-causal延迟 | causal延迟 | 反相结论 |
|---|---:|---:|---|
| 单K/3-wave历史版 | 867.325 us | 16422.659 us | pairwise约52%--54%，不稳定 |
| 双K + hardware-slot priority | 约1030 us | - | 反相中位65.5% |
| 双K + 统一priority | 约876--880 us | - | 反相中位83.9%，胜出 |

### 2026-08-09：BF16 D128参考验收与FP8 pad16

- BF16 shape：按参考分支改用`H=1, Dq=Dv=128, qo_len=kv_len=40960`，KV page数为
  `80*256*2/32=1280`。4-wave增加D128参数化，BF16每线程K copy数从固定3改为`Dq/64`。
- BF16 scheduler：D128使用参考MMA32配额，GEMM1为`VMEM:MFMA=1:1`，GEMM2为
  `lead3 + (DSRD1:MFMA1)`；并按参考读取`HW_ID.WAVE_ID`，slot0使用stage0/1=`1/3`、
  slot1使用`0/2`。D192继续使用统一`0/2`和原调度。
- BF16正确性：`H=1,D128,qo=128,kv=83`的diff为`2.46e-6`。
- BF16性能：最终同GPU夹心参考R3/C/R4为`4007.1 / 3976.8 / 4124.5 us`；当前4-wave
  时间为参考均值的97.81%，吞吐约参考的102.24%，超过“参考性能90%以上”目标。
- FP8假设：K LDS的192B行跨度造成固定bank映射；每行增加16个FP8元素后stride变为
  208B，在不改变16KB union LDS、172 VGPR、2 waves/SIMD的前提下打散bank访问。
- FP8正确性：short diff`6.87e-5`、`kv=83` diff`2.88e-4`、batch=4 diff`3.44e-4`。
- FP8性能：共享10套buffer、ABBA各50样本。non-causal两轮4/8吞吐比为
  `96.41% / 99.36%`；causal为`100.42%`，均超过“8-wave性能95%以上”目标。
- 环境：测量期间8张GPU均有外部任务，绝对时间存在DPM/负载波动；验收以同GPU、同输入、
  同进程ABBA比值和BF16 R-C-R夹心比值为准。
- 结论：采纳BF16 D128参数化、参考scheduler和FP8 K LDS pad16。

#### 本次性能快照

| 实现 | shape | 延迟 | TFLOPS/吞吐比 | 备注 |
|---|---|---:|---:|---|
| BF16 D128 4-wave | `H=1,40960 x 40960` | 3976.8 us | 参考均值的102.24% | R3-C-R4夹心 |
| BF16参考分支 | 同上 | 4007.1 / 4124.5 us | 215.1 / 208.8T | control |
| FP8 pad16 4-wave | non-causal, `10240 x 2583` | 899.4 / 876.4 us | 8-wave的96.41% / 99.36% | 两轮ABBA |
| FP8 8-wave | 同上 | 867.1 / 870.8 us | 312.4 / 311.0T | control |
| FP8 pad16 4-wave | causal, `32768 x 32768` | 15767.6 us | 8-wave的100.42% | ABBA |
| FP8 8-wave | 同上 | 15833.1 us | 694.4T | control |

### 2026-08-09：BF16支持与4-wave性能

- 接口：`PA_DTYPE=fp8|bf16`；BF16路径不量化Q/K/V，descale均为1。4-wave编译缓存键
  扩展为`(static_schedule, dtype)`，避免同进程跨dtype误复用。
- MMA/layout：FP8保持`MFMA(32,32,16)`，BF16使用`MFMA(32,32,8)`；BF16 probability
  将score的BF16寄存器存储重解释为`(4,1,(2,2)):(1,0,(4,8))`的PV B operand。
- Paged V：BF16 host向量宽度为8，PV的32-token消费顺序需要交换中间两个8-token组；
  4-wave对BF16 V应用`(8,(2,2)):(1,(16,8))`的token composition。
- K地址：recast到U32后的offset按`32/element_bits`换算；BF16 K由256线程每线程3次
  128-bit copy。每次静态copy中32线程覆盖同一D-group的连续32个token。
- LDS：plain BF16 K LDS的ATT中LDS stall占60.4%；通用`Swizzle(3,3,3)`虽正确但把
  VGPR推到272、失去2-wave occupancy。最终采用每行8个BF16元素padding，资源为
  `250 VGPR-form / 89 SGPR / 25KB LDS / 0 scratch`，无spill。
- 正确性：ragged `kv=3/13/23/53/83`为`1.63e-6--2.51e-6`；batch=4为
  `2.72e-6`；small causal为`1.93e-6`。
- 性能：50样本中位数为non-causal `1453.284 us / 186.369 TFLOPS`，causal
  `32966.564 us / 166.762 TFLOPS`。测试时8张GPU均有外部任务，绝对时间仅记录当次环境。
- 8-wave：`pa_8wave/pa_prefill_8w32x32.py`已恢复到原始版本；原始版本不支持BF16 D192
  的K cooperative copy和PV probability布局，因此不在本入口提供直接BF16性能对比。
- 结论：BF16支持完整落在4-wave kernel及其测试中，不修改8-wave实现。

#### 本次性能快照

| BF16路径 | shape | 延迟 | TFLOPS | diff |
|---|---|---:|---:|---:|
| D192 non-causal | `10240 x 2583` | 1453.284 us | 186.369 | 0.00000281 |
| D192 causal | `32768 x 32768` | 32966.564 us | 166.762 | small 0.00000193 |

### 2026-08-09：BF16 pipeline与参考分支对照

- 一致：prologue只预取K，先`K0 global->reg->LDS`并预取`K1->reg`；V到`process_kv_block`
  开头才加载。
- 一致：stage0依次执行`K(i+2)` global预取、online softmax、`K(i+1)`写LDS和
  probability转换；边界使用`enter_softmax_stage()`，D192/FP8 priority为0。
- 一致：stage1执行PV、workgroup barrier、下一K的LDS read，并跨loop backedge覆盖
  下一轮V load/QK；边界使用`enter_mma_stage()`，D192/FP8 priority为2，K LDS读在PV后发起。
- 一致：循环静态展开偶/奇两个substep，current/next K fragment角色交替，尾页保持
  compile-time stage id并单独mask。
- 差异：当前实现接入paged KV/GQA/ragged/bottom-right causal ABI，Dq=192/Dv=128
  不对称，并为batch=1提供静态grid、batch>1保留persistent ticket调度。
- 差异：当前`_online_softmax()`在返回前完成O correction rescale，随后才写`K(i+1)`
  到LDS；参考分支在stage0中先写K LDS，再做O rescale和probability转换。两者处于同一
  stage且数据依赖等价，但stage0内部源码顺序并非逐语句一致。
- 差异：参考分支BF16 K LDS使用通用swizzle；当前paged 4-wave为保持2-wave occupancy
  使用8元素行padding达到同一去bank-conflict目的。以上差异不改变stage依赖和边界。
- 结论：pipeline的数据依赖、stage划分、priority边界和loop-carried K角色与参考分支
  一致；stage0内部O rescale/K写顺序、paged接口、调度外壳和BF16 LDS物理布局实现不同。

### 2026-08-10：BF16 D128/D192 raw-max与max-shuffle重叠

- 假设：Q/K scale恒为正，因此可先对未缩放score执行lane-local reduction和跨lane
  max-shuffle，再用独立的score-scale FMA覆盖shuffle结果等待；最后只缩放raw max，
  数学结果与先缩放全部score再求max一致。
- 改动：先在BF16 D128默认启用该顺序；随后验证并让D192复用同一raw-max分支。没有
  改变online softmax的lazy rebase、probability或output correction语义。D192只复用
  softmax分支，仍保持统一`stage0/1=0/2` priority和原D192 MMA scheduler。
- 正确性：BF16 D128 short diff为`2.46e-6`；BF16 D192 ragged用例diff保持
  `1.63e-6--2.51e-6`。
- ISA：资源和occupancy不变；新增的scale FMA进入原max-shuffle等待窗口，无spill。
- 性能：第二轮C-X-X-C中12/12轮候选获胜，9个control漂移不超过0.5%的严格组
  candidate/control时间比中位数为`0.946198`，加速`5.69%`；正式50样本中位数为
  `3415.611 us / 251.49 TFLOPS`，相对同机稳定control约
  `3655.8 us / 235.0 TFLOPS`加速约7.0%。
- D192性能：`Hq=16,Hkv=1,Dq=192,Dv=128,qo=10240,kv=2583`的同进程位置平衡
  夹心中，control/candidate逐元素完全一致；24/25组control漂移不超过0.5%，25/25组
  候选获胜。50样本中位数为`1338.986 -> 1321.106 us`、
  `202.278 -> 205.015 TFLOPS`，配对时间比中位数`0.986728`，加速`1.35%`。固化后
  标准10次预热、50样本中位数为`1321.686 us / 204.926 TFLOPS`。
- D192正确性与ISA：ragged `kv_len=3/13/23/53/83` diff为
  `1.63e-6--2.51e-6`，small causal diff为`1.93e-6`；资源保持
  `250 VGPR-form / 89 SGPR / 25KB LDS / 0 scratch`，160条静态BF16 MFMA，无spill。
- ATT：control/candidate为
  `/tmp/pa4-current-bf16-att/ui_output_agent_37534_dispatch_60`和
  `/tmp/pa4-bf16-maxoverlap-att/ui_output_agent_11100_dispatch_13`。总stall从
  `111.97M`降至`97.77M`（-12.7%）；两条max-shuffle wait从`148.5/144.2`降至
  `92.7/80.2 cycles/次`；MFMA stall减少`8.57M`，首尾10%稳态反相从约82.2%升至
  90.5%。
- 结论：D128和D192均采纳raw-max分支。D192复用后TFLOPS与原路径基本一致并提升约
  1.35%，精度、资源和D192 scheduler均不变。

#### 本次性能快照

| 路径 | shape | 延迟 | TFLOPS | 备注 |
|---|---|---:|---:|---|
| BF16 D128 raw-max | `H=1,40960 x 40960` | 3415.611 us | 251.49 | 50样本正式中位数 |
| BF16 D128 control | 同上 | 3655.8 us | 235.0 | 同机稳定control |
| BF16 D192 raw-max | `Hq=16,Hkv=1,10240 x 2583` | 1321.106 us | 205.015 | 严格夹心候选 |
| BF16 D192 control | 同上 | 1338.986 us | 202.278 | 原softmax路径 |

### 2026-08-10：FP8 D192 raw-max与max-shuffle重叠

- 假设：FP8的per-token Q descale与per-tensor K descale同样恒正，因此BF16 D128的
  raw-max变换也适用于FP8；但必须独立验证D192双K流水的resident-wave相位。
- 改动：FP8默认先求raw max和执行lane交换，再以score-scale FMA遮蔽shuffle等待；
  删除实验开关，外部`MHA(...)`接口不变。
- 正确性：ragged `kv_len=3/13/23/53/83`、long non-causal和batch=4全部通过，diff为
  `6.87e-5--3.44e-4`；small causal diff为`1.75e-4`；long causal完成finite-only
  回归。BF16 D128 short复验仍为`2.46e-6`。
- ISA：control/candidate均为`44 VGPR + 132 AGPR / 16KB LDS / 0 scratch`；
  MFMA、VMEM、DS读写、barrier和occupancy不变，候选仅增加4条静态FMA。
- ATT：同一驱动control/candidate为
  `/tmp/pa4-fp8-maxoverlap-control-att/ui_output_agent_33464_dispatch_33`和
  `/tmp/pa4-fp8-maxoverlap-att/ui_output_agent_21007_dispatch_33`。归一总stall从
  `57.589`降至`51.779 cycles/MFMA`（-10.1%）；两条最热max-shuffle wait从约
  `55 cycles/次`降至`35--38 cycles/次`；首尾10% pairwise反相从72.7%升至79.3%，
  候选十等分稳定在78.5%--81.2%。另一份旧control比较也得到stall/MFMA下降4.22%和
  反相74.4%升至79.3%的同方向结果。
- 性能：机器有外部满载任务，单dispatch C-X-X-C大多因control漂移被过滤。改为同一
  stream每段聚合8次launch后，3/20组满足control漂移不超过0.5%；有效组
  candidate/control时间比中位数为`0.970486`，加速`3.04%`，全部20组的比值中位数为
  `0.971424`。
- 结论：采纳。两轮ATT均复现关键等待和反相改善，聚合夹心给出约3%墙钟收益，且资源、
  指令主体与精度均不变。

#### 本次性能快照

| FP8路径 | candidate/control时间比 | 加速 | 备注 |
|---|---:|---:|---|
| raw-max聚合夹心 | 0.970486 | 3.04% | 3/20严格有效组；ATT同方向复现 |

### 2026-08-10：FP8 max-shuffle split8调度

- 假设：raw-max路径发出`ds_bpermute`后仍有16条与shuffle结果无关的score-scale FMA，
  但后端只自动放3条到`lgkmcnt(0)`之前；用两个full scheduling barrier明确分隔
  shuffle发出、前8条FMA和shuffle消费，可继续覆盖lane交换延迟而不改变stage边界。
- 改动：仅FP8把score-scale FMA拆成8+8；BF16保持原顺序。外部`MHA(...)`接口不变。
- 正确性：候选与raw-max control逐元素完全一致；ragged `kv_len=3/13/23/53/83`
  最大diff为`2.88e-4`，batch=4为`3.44e-4`，small causal为`1.75e-4`；long
  non-causal、batch和long causal finite-only均通过。BF16 D128 short隔离复验为
  `2.46e-6`。
- ISA：wait前独立FMA从3条增至8条；资源保持
  `44 VGPR + 132 AGPR / 16KB LDS / 0 scratch`，occupancy、MFMA、VMEM、DS和barrier
  指令数不变。
- ATT：raw-max control为
  `/tmp/pa4-fp8-maxoverlap-att/ui_output_agent_21007_dispatch_33`，split8为
  `/tmp/pa4-fp8-shuffle-split8-att/ui_output_agent_16559_dispatch_33`。归一总stall从
  `51.779`降至`49.345 cycles/MFMA`（-4.70%）；两条lane-shuffle wait从
  `35.5/37.0`降至`10.7/17.2 cycles/次`；首尾10% pairwise反相从79.3%微升至
  79.9%，十等分保持78.4%--81.2%。
- 性能：空闲GPU上将25组交替`C-X-X-C`/`X-C-C-X`连续排入同一stream，只在末尾同步，
  为两实现各收集50个单dispatch样本。24/25组control漂移不超过0.5%，25/25组候选
  获胜；中位数`762.044 -> 752.503 us`，candidate/control配对比中位数
  `0.986898`，加速`1.33%`。
- 宽度扫描：相同资源下，位置轮换的40样本中位数为split6/8/10/12 =
  `763.763/750.723/754.263/750.883 us`；split8最快，split12仅慢0.02%。
- 8-wave最终验收：空闲GPU6、同输入、同进程各50样本。non-causal为
  `750.723 vs 822.044 us`，4-wave吞吐为8-wave的`109.50%`；long causal为
  `15962.581 vs 16259.192 us`，4-wave吞吐为8-wave的`101.86%`，均超过95%目标。
- 结论：采纳split8。它直接压缩目标shuffle wait，ATT、正式夹心和宽度扫描结论一致。

#### 本次性能快照

| 实现 | shape | 延迟 | TFLOPS/吞吐比 | 备注 |
|---|---|---:|---:|---|
| FP8 raw-max control | non-causal, `10240 x 2583` | 762.044 us | 355.42T | 50样本 |
| FP8 split8 | 同上 | 752.503 us | 359.93T | 较control快1.33% |
| FP8 split8最终 | 同上 | 750.723 us | 8-wave的109.50% | 最终复验 |
| FP8 8-wave control | 同上 | 822.044 us | 329.480T | 最终复验 |
| FP8 split8最终 | causal, `32768 x 32768` | 15962.581 us | 8-wave的101.86% | 最终复验 |
| FP8 8-wave control | 同上 | 16259.192 us | 676.240T | 最终复验 |

### 2026-08-10：BF16 D128 max-shuffle split8反证

- 假设：BF16 raw-max路径仍有`80--93 cycles/次`的lane-shuffle wait，可能同样受益于
  split8 full scheduling barrier。
- 改动：实验候选把wait前FMA从3条增至8条；control/candidate资源均为
  `88 VGPR + 128 AGPR / 92 SGPR / 17KB LDS / 0 scratch`，无spill。
- 正确性：40960正式输入control/candidate逐元素完全一致；short diff为`2.46e-6`。
- 性能：25组位置平衡夹心仅10/25组候选获胜；50样本中位数
  `3433.435 -> 3434.954 us`，中位比`1.000443`；14个严格有效组的配对比中位数为
  `0.999632`，属于中性。
- ATT：正式shape control/candidate为
  `/tmp/pa4-bf16-shuffle-control-long-att/ui_output_agent_9827_dispatch_13`和
  `/tmp/pa4-bf16-shuffle-split8-long-att/ui_output_agent_30163_dispatch_13`。
  shuffle wait从`95.2/76.9`降至`75.2/69.3 cycles/次`，但VMEM、DS-read和MFMA
  stall反弹，归一总stall仅`42.6736 -> 42.6453 cycles/MFMA`。
- 结论：中性并回退。BF16不保留split8边界；不能由FP8的收益外推到D128 BF16。

#### 本次性能快照

| BF16 D128路径 | 延迟 | 时间比 | 结论 |
|---|---:|---:|---|
| raw-max control | 3433.435 us | 1.000000 | 保留 |
| split8候选 | 3434.954 us | 1.000443 | 中性，回退 |

### 2026-08-10：BF16 D192理想shape与250 TFLOPS上限探索

- 目标：判断`head_dim_qk=192, head_dim_v=128`时，是否仅通过选择理想
  `Hq/Hkv/M/N` shape即可达到250 TFLOPS。
- 口径：空闲gfx942，10套buffer、10次预热、50个CUDA event样本中位数；跳过超大
  reference，但候选分支先用short shape逐元素比对。
- 长序列基准：`Hq=Hkv=1,M=N=40960`为`5272.264 us / 203.659 TFLOPS`；单纯增大
  M/N不能接近250T。
- 等FLOPs shape扫描：`Hq=1/2/4/8/16,Hkv=1`及160/320/640个workgroup组合落在
  `196.5--210.2 TFLOPS`。固定`Hq=16,M=5120`扫描`N=4096--40960`后，稳定区间峰值为
  `N=28672`的`7135.472 us / 210.671 TFLOPS`；`N=24576/32768/36864`分别为
  `210.402/210.505/210.030 TFLOPS`。
- 同shape D128对照：`Hq=16,Hkv=1,M=5120,N=32768`上，D128为
  `5520.903 us / 248.943 TFLOPS`，D192为`8161.277 us / 210.505 TFLOPS`；D192只有
  D128吞吐的84.56%。这说明shape和设备功耗不是主要限制，差距来自D192热循环路径。
- D128分支复用消融：独立比较D192原路径、仅D128 GEMM1 scheduler、仅hardware-slot
  priority、两者同时；全部候选short输出逐元素一致。干净时段约为`203--204T / 202--203T
  / 165T / 196T`，slot priority明显回退，D128 scheduler无增益。所有临时开关已回退。
- 250T差距：最佳shape在相同FLOPs下需从`7135.472 us`降至约`6012.954 us`，即延迟再降
  15.7%，或吞吐再提升18.7%。当前不能仅靠shape或直接复用D128 priority/scheduler达到。
- 结论：当前BF16 D192 kernel的实测shape上限约`210.7 TFLOPS`，不能达到250T。若以
  250T为目标，需要针对D192的160条MFMA热循环重新做ATT驱动调度，而不是继续换shape。

#### 本次性能快照

| dtype/head dim | shape | 延迟 | TFLOPS | 结论 |
|---|---|---:|---:|---|
| BF16 D192 | `Hq=16,Hkv=1,M=5120,N=28672` | 7135.472 us | 210.671 | D192 shape扫描峰值 |
| BF16 D128 | `Hq=16,Hkv=1,M=5120,N=32768` | 5520.903 us | 248.943 | 同设备理想shape对照 |
| BF16 D192 | `Hq=16,Hkv=1,M=5120,N=32768` | 8161.277 us | 210.505 | 同shape仅D128的84.56% |

### 2026-08-10：`kv_step`参数清理与BF16 D192 ATT

- 改动：删除`kv_step()`中可由`lds_stage`推导的`current_prefetch/next_prefetch`，删除
  仅透传的`page_id1`输入和`page_id1/page_id2`输出；helper现在只接收实际消费的
  `page_id0/page_id2`并返回新产生的`page_id3`，调用点显式轮转页状态。
- 正确性：BF16 D192和FP8的`kv_len=3/13/23/53/83` ragged回归全部通过；diff分别为
  `1.63e-6--2.51e-6`和`6.87e-5--2.88e-4`。
- 性能：BF16 D192标准shape清理后为`1322.365 us / 204.820 TFLOPS`，清理前为
  `1321.686 us / 204.926 TFLOPS`，变化约-0.05%，判定基本一致。
- ATT：采集`Hq=16,Hkv=1,Dq=192,Dv=128,M=10240,N=2583`的第2次static dispatch；
  1936条指令中1935条带源码映射，资源为`124 VGPR + 132 AGPR / 112 SGPR / 25KB LDS /
  0 scratch`。总stall 47.11M（67.4%），分类为MFMA/FMA 57.8%、LDS 16.1%、barrier
  6.8%、LDS wait 4.7%、VMEM load 4.5%。
- UI归档：`tests/flydsl/pa_4wave/att_bf16_d192/ui_output_agent_32152_dispatch_13/`，
  共267个文件、约140MB；`code.json`与原始trace SHA256一致。相邻目录保留
  `out_kernel_trace.csv`、`out_agent_info.csv`和`out_scratch_memory_trace.csv`。
- 结论：采纳参数清理；ATT UI可直接用于后续D192 160条MFMA热循环调度分析。

#### 本次性能快照

| BF16 D192 | shape | 延迟 | TFLOPS | 备注 |
|---|---|---:|---:|---|
| 清理前 | `Hq=16,Hkv=1,10240 x 2583` | 1321.686 us | 204.926 | 50样本 |
| `kv_step`清理后 | 同上 | 1322.365 us | 204.820 | 50样本，基本一致 |

### 2026-08-10：命名与ROCDL调度封装整理

- 命名：页状态由`page0/page1/page2/page3`改为
  `current_page_id/next_page_id/prefetch_page_id/lookahead_page_id`；原`page_id0`实际为
  当前V页，原`page_id2`实际为提前两步的K预取页。`current_work`改为
  `query_tile_index`，persistent外层使用`work_ticket/next_ticket/ticket_delta`。
- helper：`kv_step`改为`process_kv_block`；K搬运改为`prefetch_k/store_k_to_lds`；
  QK/PV fragment和MMA对象按用途命名。删除未使用的`_fma_vec_f32`和
  `_scale_center_vec_f32`。
- 调度：将裸`rocdl.sched_*`序列封装为`enter_softmax_stage()`、`enter_mma_stage()`、
  `schedule_qk_and_v_loads()`和`schedule_pv_and_next_k()`；主循环只保留数据流与stage
  边界。`_schedule_fence()`统一表示full compiler scheduling fence。
- 正确性：FP8/BF16 ragged、FP8 batch persistent及BF16/FP8 small causal全部通过，
  diff与整理前一致。
- 机器码：FP8最终ISA SHA256仍为
  `e9a5f4cfff53a1da55d090a548498dc931289fb4c29cc0713e14707cba87e7c4`；BF16 D192为
  `6414e1bff3e6d695fd74e743a8e4e51342957769009d6e74f60d0f32001542a6`。两者与整理前
  逐字一致，opcode序列、VGPR/SGPR/LDS/scratch和调度语义均未变化。
- 结论：采纳纯可读性整理；不改变性能主线。