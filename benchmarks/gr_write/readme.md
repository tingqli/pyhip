# GR write：接入、设计与性能

公共入口为 `gr_write(block_output, residual, normed_residual, packed_inject, packed_norm, *, eps=1e-6, out=None, normed=None)`。
它实现 Hyper-Connection 的 GR write，即 SGLang 的 `hc_combine`（[S2]），并把消费其输出的下一个 per-branch
Gemma RMSNorm 融合进来；只覆盖 prefill（大 T）。下文的符号、常量和数字都注明来源。

## 符号与来源

外部来源（2026-10-09 读取，行号对应以下版本）：

- **[S1]** [Qwen3.8-Flash-Next `config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
  （`main` 分支）中的 `text_config`。
- **[S2]** SGLang commit `c9ef753c67be67f4e6d284575f22483cfd9afe2c` 的
  [`hc_combine.py`](https://github.com/apinge/sglang/blob/c9ef753c67be67f4e6d284575f22483cfd9afe2c/python/sglang/kernels/ops/elementwise/hc_combine.py#L49-L93)，
  函数 `hc_combine`（L49–L93）。
- **[S3]** 同一 commit 的
  [`hyperconnection.py`](https://github.com/apinge/sglang/blob/c9ef753c67be67f4e6d284575f22483cfd9afe2c/python/sglang/srt/layers/hyperconnection.py)。
- **[S4]** 同一 commit 的
  [`qwen4_exp.py`](https://github.com/apinge/sglang/blob/c9ef753c67be67f4e6d284575f22483cfd9afe2c/python/sglang/srt/models/qwen4_exp.py)。

本仓库中的名字定义在 [common.py](../../src/pyhip/ops/gr_write/flydsl/common.py) 与
[host.py](../../src/pyhip/ops/gr_write/flydsl/host.py)。

| 符号 | 含义 | 本仓库名称 | 形状 / 取值 | 来源 |
|---|---|---|---|---|
| C | branch 数 | `C` | 4 | [S1] L15 `hc_count`；[S4] L1256、L1285 |
| H | 每个 branch 的宽度 | `H` | 2560 | [S1] L20 `hidden_size`；[S4] L1257、L1286 |
| K | residual 的行宽 | `K` | C·H = 10240 | [S2] L72 `[..., hc_count * hidden_size]` |
| eps | RMSNorm 的 epsilon | `EPS`、参数 `eps` | 1e-6 | [S1] L114 `rms_norm_eps`；[S4] L1289；[S3] L133–L135 |
| — | 数据类型 | | BF16 | [S1] L12 `dtype`；[S4] L1287 `params_dtype` |
| T | 行数（token 数） | `rows = residual.shape[0]` | 运行时参数 | [S2] L83–L85 把输入展平为 `[-1, ...]` |
| m, c, i | 行、branch、branch 内列的下标 | | 0≤m<T，0≤c<C，0≤i<H | [S2] L61–L62 |
| y | block 输出 | `block_output` | BF16 [T, H] | [S2] L71、L83 |
| r | combine 前的 residual（`mix` 的 `hyper_input`） | `residual` | BF16 [T, K] | [S2] L72、L84；[S3] L287、L290、L319–L326 |
| n | r 的 per-branch RMSNorm（`hyper_input_normed`） | `normed_residual` | BF16 [T, K] | [S2] L73、L85；[S3] L239–L240、L287 |
| W | gate 权重 `block_inject_weight.weight` | `inject_weight` | BF16 [C, K] | [S2] L74；[S3] L168–L175（`nn.Linear(K, C)`） |
| a | gate 系数 | 工作区前 T·C 个元素 | FP32 [T, C] | [S2] L61；[S3] L220–L222 |
| out | combine 结果，即新的 residual | `out` | BF16 [T, K] | [S2] L62；[S3] L223–L226 |
| w | 消费 out 的下一个 GatedResidual 的 `hc_norm.weight` | `norm_weight` | BF16 [K] | [S3] L30、L125–L135；[S4] 见下文 |
| rstd | 每行每个 branch 的 `rsqrt(mean(out²) + eps)` | | FP32 [T, C] | [S3] L69–L72 |
| normed | `hc_norm(out)` | `normed` | BF16 [T, K] | [S3] L73 |

对每行 m、每个 branch c、branch 内列 i：

```text
a[m, c]            = 2 * sigmoid(dot(n[m, :], W[c, :]) / C)          # [S2] L61；[S3] L220–L222
out[m, c*H + i]    = r[m, c*H + i] + a[m, c] * y[m, i]                # [S2] L62；[S3] L223–L226
rstd[m, c]         = rsqrt(mean_i(out[m, c*H + i]^2) + eps)           # [S3] L69–L72，每组 H 列（L130–L132）
normed[m, c*H + i] = out[m, c*H + i] * rstd[m, c] * (1 + w[c*H + i])  # [S3] L73
```

- **为什么融合这个 RMSNorm**：[S4] 中每次 combine 的输出都直接进入下一个 GatedResidual 的 `mix`。attention 的
  combine（L1344）之后是 MLP 的 `mix`（L1345）；MLP 的 combine（L1411）之后是下一层 attention 的 `mix`（L1333），
  最后一层之后是模型的 `hyper_connection_mixer.mix`（L1695）。`mix` 先计算 `hc_norm(hyper_input)`（[S3] L239–L240），
  用它做 GR read（`_mix_compute`，L193–L207），并把 `(hyper_input, hyper_input_normed)` 作为下一次 combine 的
  `(r, n)` 返回（L287、L290）。因此输出 `normed` 既是下一个 GR read 的输入，也是下一次 GR write 的 n。
- **per-branch**：[S1] 中没有 `hc_per_branch_norm` 字段；[S4] L1290、L1644 把它写死为 `True`，[S3] L18 的默认值
  `False` 不生效。于是 `hc_norm` 的宽度为 K、按 H 列分组（[S3] L125–L135），w 的形状为 [K]（L30）。
- **精度**：[S2] L65 说明 `hc_combine` 全部按 FP32 计算；RMSNorm 在 FP32 中算完再转 BF16（[S3] L58–L73；CUDA
  BF16 输入时上游改走 `grouped_gemma_rmsnorm` JIT kernel，L46–L57）。本实现的具体舍入点以
  [`pyhip.testing.gr_write.reference`](../../src/pyhip/testing/gr_write.py) 为准，误差标准见
  [tests/ops/gr_write](../../tests/ops/gr_write/readme.md)。

## 最小接入

```python
from pyhip.ops.gr_write import gr_write, prepare_weights

# 加载阶段每层做一次。inject_weight 即 W：BF16 [C, K] = [4, 10240]；
# norm_weight 即 w：BF16 [K] = [10240]（消费 out 的下一个 hc_norm 的权重）。
packed_inject, packed_norm = prepare_weights(inject_weight, norm_weight)

# y：BF16 [T, 2560]；r、n：BF16 [T, 10240]；均连续且 16B 对齐。
with torch.inference_mode():
    out, normed = gr_write(y, r, n, packed_inject, packed_norm)
    # 已有输出时直接写入并返回同一对象。
    out, normed = gr_write(y, r, n, packed_inject, packed_norm, out=out_buf, normed=normed_buf)
```

- `packed_inject`：BF16 [C·K] = [40960]，形状 `[slice 4][group 5][t 4][lane 64][8]`，
  `packed_inject[s][g][t][4·block + c][e] = W[c][s·2560 + g·512 + t·128 + block·8 + e]`（common.py 的
  `prepare_weights`）。s 为 gate 的 wave，g 为每次读入的 512 列，t 为其中 128 列一段，lane = 4·block + c 是
  MFMA 4x4x4 A 操作数的 lane 分配，e 为连续 8 列；每个 (s, g, t) 是连续 1 KiB（64 lane × 16 B）。
- `packed_norm`：FP32 [K] = [10240]，内容为 FP32 的 `1 + w`，按 apply 的 `[wave][chunk][half][lane][4]` 归属排列
  （同上）。
- 每次调用申请 FP32 工作区 `[T·C + 4]`（common.py 的 `workspace_words`）：前 T·C 个元素是 a，末尾 4 个元素存 apply
  的行计数器，由 gate kernel 的 workgroup 0 清零（gate.py）。工作区不跨调用保留，并发 stream 各自持有。
- 编译缓存的 key 为 `(device index, gcnArchName, eps)`，T 是运行时参数；缓存未命中时在 Graph capture 内调用会报错，
  所以 capture 前要先在该 device 上调用一次（host.py）。
- 输入、输出须连续且基址 16 B 对齐；输出不能与输入、权重或彼此重叠；T=0 不编译、不 launch；不支持梯度（host.py）。

## 设计

`gr_write`（host.py）通过一个编译后的 `launch` 在当前 stream 上依次提交两个 kernel，二者不重叠：

- **gate**：算上面公式的第 1 行 a = 2·sigmoid(n·Wᵀ / C)，对应 [S3] L220–L222 的 `block_inject_weight_out`；读 n 与
  `packed_inject`，把 a 写进工作区，并清零 apply 的行计数器。
- **apply**：算第 2–4 行，即 combine 的 out = r + a·y（[S3] L223–L226）和下一个 `hc_norm` 的 rstd、normed
  （[S3] L69–L73）；读 y、r、a 与 `packed_norm`，写 out 与 normed。

a[m, c] 要用到 n[m] 的全部 K 列，而 apply 处理一行时要用到 4 个 branch 的 a，所以先跑 gate 再跑 apply。两者读的
输入不重叠，拆分只多出 a 的一次写和一次读（每行 C × 4 B = 16 B）。下列常量除注明的文件外都定义在 common.py。

1. **Gate**（[gate.py](../../src/pyhip/ops/gr_write/flydsl/gate.py)）
   - 一个 workgroup 有 `GATE_SLICES = 4` 个 wave，处理一个 4 行 quad；每个 wave 负责
     `GATE_SLICE_COLUMNS = K / 4 = 2560` 列，分为 `GATE_GROUPS = 5` 个 `GATE_GROUP = 512` 列的 group。
   - 每行每个 group 一条连续 1 KiB 的 load（64 lane × 16 B），写入 wave 私有的 LDS tile，再按 MFMA 的 B 操作数
     布局读出，与寄存器中的 W 分片做 FP32 累加。MFMA 在代码中是 ROCDL/LLVM intrinsic
     `mfma_f32_4x4x4bf16_1k`（`llvm.amdgcn.mfma.f32.4x4x4bf16.1k`），gfx942 ISA 中为
     `v_mfma_f32_4x4x4_16b_bf16`（见 FlyDSL dump 的 `22_final_isa.s`）。LDS 行距
     `ROW_STRIDE = 512 × 2 + 32` B，多出的 32 B 让转置读取的 4 行落在不同 bank（gate.py）。
   - 每个 wave 的 16 个点积用 DPP `row_ror` 4/8、`ds_swizzle`（xor 16）和 `ds_bpermute`（xor 32）组成的转置
     butterfly 归约；4 个 slice 的部分和在 LDS 汇总，由 wave 0 写出 16 个 a。上一个 quad 的这一步放在当前 quad
     第 `FINISH_GROUP = 0` 个 group 之后（gate.py），与当前 quad 的读取重叠。
   - Persistent grid：`GATE_WORKGROUPS_PER_CU = 2`，grid = max(1, min(2·CU, ⌈T/4⌉))（`launch_grids`）；
     workgroup 按静态步长遍历 quad，预取 `GATE_PREFETCH = 3` 个 group。
2. **Apply**（[apply.py](../../src/pyhip/ops/gr_write/flydsl/apply.py)）
   - 一个 workgroup 有 `APPLY_WAVES = 4` 个 wave，处理一行，对应 CDNA3 每个 CU 的 4 个 SIMD。wave w 的 lane t
     负责 `APPLY_CHUNKS = 5` 个 8 列 chunk：chunk b < 4 为 branch b 的列 512w + 8t；chunk 4 为 branch w 的列
     `TAIL_COLUMN + 8t`（`TAIL_COLUMN = 4 × 512 = 2048`，apply.py）。前 4 个 chunk 覆盖每个 branch 的列 0–2047，
     chunk 4 覆盖列 2048–2559，合计 4·4·512 + 4·512 = 10240 = K。每条 load/store 都是连续 1 KiB，4 个 wave
     工作量相同。y、r 的读取用默认缓存策略（4 个 wave 都会读 y 的最后 512 列），out、normed 的写用 NT；读取策略的
     A/B 见下文。
   - out 由 packed FP32 FMA 算出后按最近偶数舍入到 BF16。每个 branch 的平方和先用 DPP xor 1/2 的转置 butterfly
     在 wave 内归约，再经 LDS 跨 wave 汇总，每行一次 barrier。gate 与 rstd 用 `readlane` 读进 SGPR：用 VGPR 广播时，
     寄存器分配可能把它和仍在飞行的 load 目的寄存器配对，迫使整段 `vmcnt(0)`（apply.py 模块说明）。
   - 行由原子计数器按序领取，使同时处理的行在内存中相邻；两套寄存器 ping-pong，处理第 k 行时预取第 k+1 行。
     领取在其伴随的预取之前发出、延后一行才使用，因为 vmcnt 按序计数，等原子返回也会等所有更早的访存（apply.py
     模块说明）。uniform 地址的原子会被 AMDGPU atomic optimizer 改写成等待 `vmcnt(0)` 的 wave scan，因此地址加一个
     来自 VGPR 的 0（helpers.py 的 `make_claim`）。
   - 每行 10 个 store（5 个 chunk × out/normed）在归一化之后集中发出。barrier 前 `s_waitcnt vmcnt(4)`，只让最后
     4 个预取 load 跨过 barrier（A/B 见下文）。Prologue 的 10 个越界 dummy store 让循环头的 vmcnt 历史与稳态一致；
     第二行无条件处理（越界时全部屏蔽），保证各路径的 vmcnt 历史相同（apply.py 注释）。
   - Persistent grid：`APPLY_WORKGROUPS_PER_CU = 3`，grid = max(1, min(3·CU, T))（`launch_grids`）。3 个
     workgroup × 4 wave 分到 4 个 SIMD，即每个 SIMD 3 个 wave。gfx90a/gfx942 每 lane 有 512 个 VGPR+AGPR、分配
     粒度为 8（LLVM `AMDGPU::IsaInfo::getTotalNumVGPRs`、`getVGPRAllocGranule`），⌊512/3⌋ = 170 向下取 8 的倍数
     得 168，所以 apply 不能超过 168 VGPR。

## 性能数据

**测量条件**：2026-10-09 07:51 UTC，物理 GPU2，AMD Instinct MI308X（gfx942:sramecc+:xnack-，80 CU）。测前
`rocm-smi --showuse` 为 0%；`amd-smi process` 列出的其他进程（本容器内不可见）VRAM 与 CU 占用均为 0；
`amd-smi static -g 2 --limit` 为 PTL Enabled / VECTOR,F8（GPU3 相同）。
ROCm 7.2.4（amdgpu 6.16.13），FlyDSL 0.3.4.1，PyTorch `2.12.0+rocm7.2.4.gitcf5ea6e.post2`。

**协议**（bench_gr_write.py 的默认参数 `--buffers 10 --warmup 2 --iters 10 --seed 0`）：eager cudaPerf，每次调用
轮换一组独立的输入/输出，丢掉 2 个预热样本，取 10 个样本的中位数。所有 T 共用一次按最大 T 分配的缓冲区（行前缀
视图），计时前用 `check_close` 做一次精度检查。

**列定义**：

- Total：`gr_write(..., out=, normed=)`，即 gate + apply。
- TB/s = `bytes_moved(T)` / Total，其中 `bytes_moved(T) = T × (H + 4K) × 2 B = T × 87040 B`：读 y（H×2 B）
  与 r、n，写 out、normed（各 K×2 B），不计权重与工作区（`pyhip.testing.gr_write.bytes_moved`）。
- Gate / Apply：用与 `gr_write` 相同的 grid 单独 launch 各自的 kernel；apply 每个样本前在计时区外清零行计数器。
- Torch eager：bench 中的 `torch_baseline`，未融合的 PyTorch 实现（matmul + sigmoid + FP32 逐元素 + RMSNorm），
  仅作数量级对照；Speedup = Torch eager / Total。

被测源码（`sha256sum`）：

```text
c8084b0ec17755ffffd42e45c240b6953b5db527ca2bf61ff595b2ea0afb8402  benchmarks/gr_write/bench_gr_write.py
c8cd764562d75df0350a24f43e886ab5776a9216e8a839204dcdfeadbf5e45ef  src/pyhip/ops/gr_write/__init__.py
d2929319e19cb92164573e92e4f6933f48b5f3deea9ccf518ba11a5f6bd005d3  src/pyhip/ops/gr_write/flydsl/__init__.py
ef7b28f14a7742643ae6b3d9f0804aae916983127aa9144e62cb86c68ce07f7f  src/pyhip/ops/gr_write/flydsl/apply.py
bc6c84aef08ab6251ef79dc28e2a1e7ccd790a72f5ad08dd992307a5accd8e1a  src/pyhip/ops/gr_write/flydsl/common.py
1295718296bd344c504e8b83e2944a7de5f00aef3db0ec79afabc4a1eabb0b00  src/pyhip/ops/gr_write/flydsl/gate.py
f6d29f1960f71ccb2daffa82efed69d1d8f589d1641c9dbaeb38aad8f322e339  src/pyhip/ops/gr_write/flydsl/helpers.py
d0ba16f986e01e621a178e9788cb798c8134c2aa31a9f6aa4e8159f27621d812  src/pyhip/ops/gr_write/flydsl/host.py
bbffbb6e8f1833ae564e254fc356bb259822b6301918cab2550a5bab7c7b08de  src/pyhip/testing/gr_write.py
```

```bash
python3 benchmarks/gr_write/bench_gr_write.py --gpu 2 --breakdown --baseline
```

| T | Total us | TB/s | Gate us | Apply us | Torch eager us | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 41.0 | 2.18 | 11.6 | 32.5 | 294.7 | 7.19x |
| 2048 | 64.3 | 2.77 | 15.6 | 51.0 | 530.6 | 8.25x |
| 4096 | 111.5 | 3.20 | 24.0 | 89.3 | 1085.0 | 9.73x |
| 8192 | 211.3 | 3.37 | 45.6 | 167.2 | 2143.0 | 10.14x |
| **12000** | **301.4** | 3.46 | 62.6 | 240.4 | 3253.5 | 10.79x |
| 16384 | 403.2 | 3.54 | 82.2 | 323.6 | 4492.6 | 11.14x |
| 32768 | 792.2 | 3.60 | 157.2 | 632.6 | 9086.6 | 11.47x |

### T=12000 重复测量

与上表同一源码，2026-10-09 07:51 UTC，每次为独立进程：

```bash
python3 benchmarks/gr_write/bench_gr_write.py --gpu 2 --rows 12000 --breakdown   # GPU3 改为 --gpu 3
```

| GPU | Total us | Gate us | Apply us |
|---|---|---|---|
| 2 | 296.5 / 297.6 / 297.8 | 62.3 / 62.2 / 62.5 | 235.8 / 235.4 / 235.5 |
| 3 | 296.6 / 297.3 / 297.6 | 62.6 / 62.8 / 62.3 | 236.4 / 236.5 / 238.4 |

这 6 次独立进程的 Total 都低于 300 us；上表的整表运行（按 T=32768 一次分配、取行前缀视图）中 T=12000 为
301.4 us，比这 6 次高 3.6–4.9 us。

### y、r 读取的缓存策略：与 commit 629da53 对比

commit `629da53`（add prefill gr_write）中 apply 的 y、r 读取带 NT，其 apply.py 的 `sha256sum` 为
`d866820b4ff3d455828157b7b2d0bf66f0b90a14ba4687cd8642c89ee30f7161`；当前版只去掉这两处 NT，其余 gr_write 文件相同。
两版分别用 `git archive 629da53 src/pyhip benchmarks/gr_write | tar -x -C <dir>` 和工作区副本导出到临时目录，在
GPU2 上交替运行上面的 T=12000 命令各 4 次（2026-10-09 07:52 UTC，`FLYDSL_RUNTIME_ENABLE_CACHE=0`；bench 从脚本
所在目录树的 `src` 导入）：

| 版本 | Total us | Gate us | Apply us |
|---|---|---|---|
| 629da53（读取 NT） | 300.9 / 299.0 / 300.3 / 302.8 | 62.4 / 62.9 / 62.5 / 63.1 | 237.2 / 255.7 / 240.0 / 259.5 |
| 当前（读取默认） | 298.8 / 298.6 / 296.8 / 296.5 | 62.9 / 63.0 / 62.5 / 62.5 | 236.6 / 290.0 / 239.8 / 238.9 |

逐次配对，当前版 Total 快 0.4–6.3 us（中位数 2.8 us）。Apply 单测中的 255.7、259.5、290.0 us 是离群值（同次 Total
正常），按原值保留。

### barrier 前 `s_waitcnt vmcnt(4)` 的 A/B

把当前源码复制两份到临时目录，其中一份删掉 apply.py 中 barrier 前的 `rocdl.s_waitcnt(vmcnt=4)`（保留两侧的
`sched_barrier`），与上一节的两版在同一轮中交替运行（2026-10-09 07:52 UTC，条件同上）：

| 版本 | Total us | Apply us |
|---|---|---|
| 当前（有等待） | 298.8 / 298.6 / 296.8 / 296.5 | 236.6 / 290.0 / 239.8 / 238.9 |
| 删掉等待 | 299.7 / 300.0 / 299.4 / 300.4 | 238.8 / 288.1 / 238.3 / 244.0 |

逐次配对，删掉等待后 Total 慢 0.9–3.9 us。两版第 2 次的 Apply 单测都在 288–290 us（同次 Total 正常）。

### 稳态斜率与固定开销（由上表推出）

取上表 T=16384、T=32768 两行做两点线性拟合：斜率 = (X(32768) − X(16384)) / 16384，截距 = X(16384) − 16384 × 斜率，
稳态带宽 = 每行字节数 / 斜率。

| 项 | 每行字节 | 斜率 us/千行 | 截距 us | 稳态 TB/s | T=12000 外推 / 实测 us |
|---|---:|---:|---:|---:|---:|
| Total | (H + 4K)·2 = 87040 | 23.74 | 14.2 | 3.67 | 299.1 / 301.4 |
| Gate：读 n | K·2 = 20480 | 4.58 | 7.2 | 4.47 | 62.1 / 62.6 |
| Apply：读 y、r，写 out、normed | (H + 3K)·2 = 66560 | 18.86 | 14.6 | 3.53 | 240.9 / 240.4 |

按此口径，T=12000 的 Total 中约 285 us 随 T 线性增长，约 14 us 不随 T 增长（launch、prologue、尾部等；这是拟合值，
未单独测量）。Gate、Apply 单测各含一次 launch 与事件开销，所以二者之和大于 Total。

### 测量注意事项

- benchmark 只分配一次缓冲区，所有 T 用行前缀视图；原因见 bench_gr_write.py 中的注释（重新分配会改变物理放置与带宽）。
- 单独计时的 Apply 不时出现 255.7–290.0 us 的离群值，同次 Total 正常（见上面两个 A/B 表）；比较时以 Total 为准。
- 数据只代表本次软硬件与分配条件，不是跨机器的性能保证。正确性回归见
  [tests/ops/gr_write](../../tests/ops/gr_write/readme.md)。
