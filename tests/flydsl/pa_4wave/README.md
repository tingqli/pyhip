# Paged Prefill 4-wave 优化记录

状态：ACTIVE

目标是在 gfx942 上实现 MiMo FP8 paged-prefill 的 4-wave pipeline，并保持与
`pa_8wave` 相同的调用接口、分页语义、GQA、ragged last page 和 bottom-right
causal mask。主验收 shape 为：

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
| **4-wave 最终** | non-causal, `batch=1, 10240 x 2583` | **867.325 us** | **312.279** | 0.000367 | GPU0，50 样本中位数 |
| 8-wave control | non-causal, `batch=1, 10240 x 2583` | 821.804 us | 329.576 | 0.000367 | GPU0，同口径；4-wave 慢 5.54% |
| **4-wave 最终** | causal, `batch=1, 32768 x 32768` | **16422.659 us** | **334.754** | 0.000268 | GPU1，50 样本中位数 |
| 8-wave control | causal, `batch=1, 32768 x 32768` | 17035.219 us | 322.717 | 0.000268 | GPU1，同口径；4-wave 快 3.60% |
| 8-wave 基线 | non-causal, `batch=4, 10240 x 2560` | 3142 us | 341.7 | 0.000344 | 额外 persistent 负载均衡回归 |
| 4-wave persistent | non-causal, `batch=4, 10240 x 2560` | 3875 us | 277.1 | 0.000344 | 功能/调度回归，非主性能验收 |

最终静态 kernel 资源：`168 VGPR / 89 SGPR / 16KB LDS / 0 scratch`，达到
3 waves/SIMD；静态 ISA 有 80 条 FP8 MMA、9 条 `s_barrier`。batch=1 走静态
grid；batch>1 保留 atomic-ticket persistent 调度。

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

### 2026-08-09：静态派发、3-wave occupancy与host开销

- 假设：batch=1无需 persistent外层 while/atomic/mailbox；静态kernel可同时降低
  host开销和寄存器压力。
- 改动：提取单work-item helper；batch=1直接映射block到 `(head, query_tile)`；
  batch>1保留persistent kernel。静态路径复用一个dummy counter，不再每次分配/清零。
- ISA：从persistent的约`192 combined VGPR`降到静态`168 VGPR`；配合16KB LDS达到
  3 waves/SIMD；最后一个仅为persistent复用所需的C-shuffle barrier在静态路径裁掉。
- 性能：short host+kernel调用从约54us降到约10us；non-causal正式中位数
  `867.325 us / 312.279 TFLOPS`。
- 结论：采纳。

### 2026-08-09：causal 80-CU仿射负载均衡

- 假设：自然query-tile顺序导致静态causal尾部集中在重tile；按80 CU轮转模型均衡每CU
  累计page数可保留3-wave静态kernel并消除尾部拖延。
- 改动：当 `works_per_head == 256` 时使用 `tile=(251*physical_tile+251)%256`
  的一一仿射排列；其他shape使用通用轻重交错映射。
- 正确性：`256x256`、`384x384`和`32768x32768` causal均通过。
- 性能：自然静态约17.9ms；轻重交错约16.7ms；仿射映射正式中位数
  `16422.659 us / 334.754 TFLOPS`。
- 结论：采纳。

### 2026-08-09：失败与中性实验

- K copy改为128-bit（前128线程或四波均衡两段）：正确，但长shape回退约2.7%到
  10.7%；64-bit DS写虽在ATT中stall高，却参与resident-wave相位，回退。
- K LDS write简单后移：单页通过但长序列错误；根因是覆盖仍被后续消费的K prefetch
  fragment。改为正确current/next角色后精度恢复，但墙钟中性。
- probability乘240+exact rebase：精度略改善但性能回退；主线保持与8-wave一致的
  lazy rebase和未缩放FP8 probability。
- vector running-sum：精度不变但扩大loop-carried状态，无稳定收益，回退。
- direct 64-bit output store：LDS降到12KB、VGPR降到170，但失去C-shuffle，性能回退
  到约1003us；半块C-shuffle更优。
- causal persistent ticket：约18.9ms；静态自然顺序约17.9ms；每head轻重相邻映射
  严重回退到24.6ms。均已回退。

### 2026-08-09：最终验收

- 功能：全部输出finite；ragged page、反转page table、GQA、batch=4、长non-causal、
  bottom-right causal均通过；主shape diff分别为`0.000367/0.000268`。
- 正式口径：10套buffer、10次预热、50个CUDA event样本中位数。
- non-causal同卡：`867.325 / 821.804 = 1.0554`，4-wave慢5.54%。
- causal同卡：`16422.659 / 17035.219 = 0.9640`，4-wave快3.60%。8-wave在该次
  正式测量中存在DPM双态，因此以用户确认的中位数口径判定。
- 结论：两条主验收均满足4-wave不慢于8-wave超过10%的目标。