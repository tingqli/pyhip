# Paged Prefill 4-wave 优化记录

状态：ACTIVE

面向MI308X/gfx942的paged prefill attention，支持FP8/BF16 Q/K/V、BF16输出、GQA、
ragged page、变长batch和bottom-right causal mask。主配置为
`Hq=16, Hkv=1, Dq=192, Dv=128, page_size=32`；`batch=1`走static grid，
`batch>1`走atomic-ticket persistent grid。

版本：4-wave `ec42e3359e6a9685495220c52122874b931210946afb908a5afc394a220dfeb0`；
8-wave `71cfe876284d84872cbbe18f947e1dc4695fa911efe8d481fa36eea629fefc12`。

## 最新结果

更新时间：2026-08-11。最新8-wave参考使用per-tensor Q量化；4-wave接收同一份FP8 Q和
等值descale。两者在同一进程使用10套buffer、各10次预热和50个CUDA event样本，采用
位置平衡顺序；“相对8-wave”为25组配对时间比中位数的倒数。

| 场景 | 时钟 | 实现 | 调度 | 中位延迟 | Actual TFLOPS | 相对8-wave |
|---|---|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | **4-wave** | static | **671.343 us** | **403.441** | **1.224x** |
| 同上 | auto | 8-wave | persistent | 821.883 us | 329.544 | 1.000x |
| causal `Q=KV=32768` | auto | **4-wave** | static | **17918.507 us** | **306.809** | **1.059x** |
| 同上 | auto | 8-wave | persistent | 18872.409 us | 291.301 | 1.000x |
| causal `Q=KV=32768` | 1300MHz | **4-wave** | static | **18836.170 us** | **291.862** | **1.061x** |
| 同上 | 1300MHz | 8-wave | persistent | 19992.191 us | 274.985 | 1.000x |

non-causal 25/25组获胜；causal auto 24/25组获胜；causal 1300MHz 25/25组获胜。
causal按三角有效FLOPs计数；auto-DPM存在双态，因此同时保留1300MHz结果。

4-wave当前只支持page32，因此4/8-wave表使用共同支持的page32。

主shape的4-wave static/persistent同代码对照：

| 场景 | 时钟 | static | persistent | static收益 |
|---|---|---:|---:|---:|
| non-causal `Q10240,KV2583` | auto | 670.102 us / 404.188T | 839.263 us / 322.720T | **25.18%** |
| causal `Q=KV=32768` | 1300MHz | 18836.068 us / 291.863T | 19555.952 us / 281.119T | **3.85%** |

因此non-causal的400T主要依赖batch=1 static调度；persistent路径仍约323T。causal也受益于
static，但收益明显较小。两组static/persistent输出逐元素一致。8-wave始终使用persistent。

### 4-wave性能矩阵

除H3使用3次预热和10样本外，其余formal结果均为10套buffer、10次预热和50样本中位数。
causal括号内为当次快档min。

| dtype | Dq/Dv | 调度 | 场景 | shape | 中位延迟 | Actual TFLOPS |
|---|---:|---|---|---|---:|---:|
| FP8 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 672.883 us | 402.518 |
| FP8 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 3065.972 us | 350.213 |
| FP8 | 192/128 | static | causal | `H16,Q=KV=32768` | 17802.427 us (13973.054) | 308.809 |
| BF16 | 192/128 | static | non-causal | `H16,Q10240,KV2583` | 1323.445 us | 204.653 |
| BF16 | 192/128 | persistent | batch=4 | `B4,H16,Q10240,KV2560` | 7475.309 us | 143.638 |
| BF16 | 192/128 | static | causal | `H16,Q=KV=32768` | 35486.656 us (25454.100) | 154.919 |
| FP8 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 2500.491 us | 343.530 |
| FP8 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 208.721 us | 257.219 |
| FP8 | 128/128 | static | causal | `H1,Q=KV=32768` | 1137.484 us | 241.654 |
| BF16 | 128/128 | static | non-causal | `H1,Q=KV=40960` | 3422.933 us | 250.952 |
| BF16 | 128/128 | persistent | batch=4 | `B4,H1,Q10240,KV2560` | 268.201 us | 200.175 |
| BF16 | 128/128 | static | causal | `H1,Q=KV=32768` | 1650.006 us | 166.592 |
| FP8 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 86.369 ms | 331.755 |
| BF16 | 128/128 | persistent | H3 varlen | `(63225,7),H14` | 179.958 ms | 159.223 |

### 精度矩阵

`diff`为`pyhip.calc_diff`对PyTorch reference；全部通过`rtol=atol=0.1`和finite检查。

| dtype | Dq/Dv | ragged最大diff | batch=4 diff | small causal diff | 主shape/额外验证 |
|---|---:|---:|---:|---:|---|
| FP8 | 192/128 | `2.8836e-4` | `3.4356e-4` | `1.7518e-4` | 主non-causal `3.6652e-4` |
| BF16 | 192/128 | `2.5129e-6` | `2.7224e-6` | `1.9344e-6` | 主non-causal `2.8093e-6` |
| FP8 | 128/128 | `2.6076e-4` | `3.4029e-4` | `1.7112e-4` | H3 finite |
| BF16 | 128/128 | `2.4619e-6` | `2.7061e-6` | `1.8679e-6` | H3 finite |

ragged覆盖`KV=3/13/23/53/83`，small causal为`Q=KV=256`。4-wave/8-wave同输入的
non-causal与causal relative-L2分别为`1.17e-4`和`1.12e-4`。

## 当前实现

- block为`BM128 x BN32 x 256 threads`；每个workgroup 4个wave；
- K使用LDS ping-pong，V直接进入fragment，output使用两个半块C-shuffle；
- online softmax使用raw-max、lazy rebase和loop-carried max/sum；
- FP8使用QK `VMEM1 -> MFMA2`、score MUL split11和FP8-only fast-math；
- FP8 D192为168 combined VGPR、16KB LDS、0 scratch，自然达到3 waves/SIMD；
- BF16 D128使用专用scheduler/HW-slot priority，D192使用独立scheduler；
- BF16/FP8共享pipeline时序骨架，dtype专属K搬运、scheduler、probability写回、V布局、
  epilogue地址和compile hint均封装在独立helper。

refactor前后fresh执行ISA逐条一致：

| specialization | ISA资源 | MFMA |
|---|---|---:|
| FP8 D192 | 168 VGPR-form / 16KB / 0 scratch | 80 |
| FP8 D128 | 153 VGPR-form / 16KB / 0 scratch | 64 |
| BF16 D192 | 250 VGPR-form / 25KB / 0 scratch | 160 |
| BF16 D128 | 214 VGPR-form / 17KB / 0 scratch | 128 |

FP8 D192 dynamic persistent kernel的执行ISA同样逐条一致。

## 复现

```bash
cd /root/workspace/luocheng/pyhip
export HIP_VISIBLE_DEVICES=7
export FLYDSL_RUNTIME_ENABLE_CACHE=0

PA_CASE=tails PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=batch PA_NUM_ITERS=1 python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=noncausal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_CASE=causal PA_NUM_ITERS=1 PA_FORMAL_BENCH=1 PA_SKIP_REFERENCE=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
PA_DTYPE=bf16 PA_CASE=bf16_ref_short PA_NUM_ITERS=1 \
  python3 -B tests/flydsl/pa_4wave/test_pa_prefill.py
```

测试前用`rocm-smi --showuse`选择空闲GPU。完整causal reference需要约64GB临时显存；主shape
通常执行finite和4/8-wave同输入对比。定频诊断流程见`tests/flydsl/H3_ATTENTION_THROTTLE_PROFILE.md`。

## 主要修改

记录只保留改变当前实现或建立关键反证的里程碑，格式统一为“改动 / 验证 / 结果”。

### 2026-08-10：4-wave pipeline与C-shuffle

- **改动**：建立MMA32骨架；K走LDS ping-pong，V直读；接入paged ABI、GQA、ragged和causal；
  output改为两个64x128半块C-shuffle。
- **验证**：反转page table、跨页和ragged尺寸通过。
- **结果**：约`1838 -> 1465 -> 1008 -> 915 us`；保留双缓冲和半块C-shuffle。

### 2026-08-10：static dispatch与causal均衡

- **改动**：batch=1改用static grid；batch>1保留persistent；causal使用
  `(251 * tile + 251) % 256`映射。
- **验证**：non-causal、batch=4和long causal通过。
- **结果**：short调用约`54 -> 10 us`；causal约`17.9 -> 16.7 ms`；保留static/仿射路径。

### 2026-08-10：双K流水与priority

- **改动**：形成`K(i+2)`预取、softmax、`K(i+1)`写入、PV/barrier/K-read跨回边流水；
  FP8统一stage priority为`0/2`。
- **验证**：双K统一priority反相稳定；HW-slot priority回退。
- **结果**：主路径约876--880us；保留双K与统一priority。

### 2026-08-10：raw-max与softmax调度

- **改动**：先对raw score做max/shuffle，再用score scaling覆盖等待；FP8增加固定切分。
- **验证**：FP8/BF16数值不变；shuffle wait由约55降至10.7/17.2 cycles。
- **结果**：raw-max约提升3%，split8再提升1.33%；BF16不采用split8。

### 2026-08-10：BF16与H3

- **改动**：加入D128/D192 BF16 MMA、K/V layout、128-bit copy、LDS padding和D128 scheduler。
- **验证**：BF16 ragged/batch/causal及真实H3通过。
- **结果**：当前BF16 D128为250.952T，D192为204.653T，H3为159.223T。

### 2026-08-10：FP8自然3-wave

- **改动**：epilogue重建C-shuffle地址，将资源从176降至168 combined VGPR；固定gap2、
  score MUL split11和FP8-only fast-math。
- **验证**：16KB LDS、0 scratch、80 MFMA；最终ATT三槽`2+1`混合相90.36%。
- **结果**：当前non-causal为402.518T；最终4-wave相对8-wave为1.222x。

### 2026-08-11：最新8-wave参考与最终复测

- **改动**：使用最新8-wave per-tensor Q和多page-size实现，共享输入位置平衡复测；
  causal额外使用1300MHz固定频率。
- **验证**：page32/64/128 short reference通过；定频结束后恢复auto。
- **结果**：non-causal为1.224x；causal auto/1300MHz分别为1.059x/1.061x。

### 2026-08-11：代码refactor

- **改动**：保留共享pipeline，将BF16/FP8 K搬运、scheduler、probability、V布局、epilogue和
  compile hint封装为helper；删除恒真/恒假参数并统一命名。
- **验证**：4种static specialization和FP8 dynamic persistent执行ISA逐条一致；完整精度矩阵通过。
- **结果**：性能矩阵与重构前一致；共享时序骨架、独立dtype细节，不复制两份pipeline。

## 已否决方向

| 类别 | 关键证据 | 保留方案 |
|---|---|---|
| K copy 128-bit | 隐式`vmcnt(0)`，回退15.4%--15.5% | 64-bit K copy |
| 阶段/HW-slot | 循环slot回退2.76%；入口复制pipeline回退16.71% | 单pipeline、FP8统一priority |
| barrier/PV/K写 | 隔页barrier等待约124增至403 cycles；PV切分增scratch | 每页barrier、完整PV |
| page-table pair load | VGPR 168升至172，失去3-wave | 标量lookahead |
| 映射/priority | tile-major回退0.5%；反向priority约回退3% | head-major、`0/2` |
| 数值调度 | sum多链/非均匀gap无收益；显式rcp仅397.112T | 单链sum、gap2、fast-math |
| BF16实验 | split8中性；D192 shape峰值约210.7T | 原BF16 softmax、独立D192优化 |

## ATT

- FP8 D192：`tests/flydsl/pa_4wave/att_fp8_d192_3wave/ui_output_agent_28524_dispatch_66`；
- BF16 D192：`tests/flydsl/pa_4wave/att_bf16_d192/ui_output_agent_32152_dispatch_13`；
- FP8主要stall/MFMA：MFMA 36.674、VALU 12.619、barrier 7.413、VMEM-load 6.397、
  LDS-wait 5.900；两条barrier约128/145 cycles。
