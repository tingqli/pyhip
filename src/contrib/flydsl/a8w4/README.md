# a8w4：MiMo-V2.6-Flash 的 A8W4 MoE（gfx942，FlyDSL；TP / EP 一套接口）

fp8 激活（FNUZ）× mxfp4 权重的 MoE，面向 MI300X / MI325X（gfx942）。TP 和 EP 共用一个入口 `a8w4_moe`：每次调用按（shape，token 数）查调优表，从 5 类 kernel 里选出当前上下文（激活函数、是否在 CUDA graph capture 中、token 数）允许的最快配置。EP 的 all-to-all 接收缓冲区（例如 MORI dispatch：按容量分配，有效行数只在 device 上）用 `a8w4_moe_dyn`，不做 host 同步，可以放进 CUDA graph（2.4 节）。调优表覆盖 XiaomiMiMo/MiMo-V2.6-Flash(-RL) 的 TP8、EP8 每卡 shape（另有 TP4），1–32768 tokens，SiLU 与 SiTUv2 各一组（TP4 只有 SiLU）。

本目录是一个独立的 Python 包。kernel 只依赖 torch 和 FlyDSL 0.3.2，aiter 的 FlyDSL helper 已复制到 `_vendor/`。aiter 只在三处用到：TP 族 kernel 的路由排序（`moe_sorting`）、权重 preshuffle 辅助函数，以及测试和调优。

## 目录

1. [快速开始](#1-快速开始)
2. [接口与约定](#2-接口与约定)
3. [文件与依赖](#3-文件与依赖)
4. [kernel 一览与按 batch 的选择](#4-kernel-一览与按-batch-的选择)
5. [算法](#5-算法)
6. [调优机制](#6-调优机制)
7. [性能](#7-性能)
8. [正确性与测试](#8-正确性与测试)
9. [继续开发的注意点](#9-继续开发的注意点)
10. [踩过的坑](#10-踩过的坑)
11. [后续方向](#11-后续方向)
12. [来源](#12-来源)

---

## 1. 快速开始

```python
# 在 pyhip 里：from pyhip.contrib.flydsl.a8w4 import ...；单独复制出去时：
import sys
sys.path.insert(0, "/path/to/dir_that_contains_a8w4")
from a8w4 import a8w4_moe, a8w4_moe_dyn, prepare_weights, shuffle_mxfp4

# 加载权重时做一次
W = prepare_weights(w1, w2, w1_scale, w2_scale)            # 已是 aiter a16w4 preshuffle 格式的 mxfp4 权重
W = prepare_weights(*shuffle_mxfp4(w1q, w1s, w2q, w2s))    # 或者：未 shuffle 的 checkpoint 张量

# 前向
out = a8w4_moe(x, W, topk_weights, topk_ids)                        # TP：之后在 TP 组内 all-reduce
out = a8w4_moe(x, W, topk_weights, topk_ids,
               expert_offset=ep_rank * E_local, num_experts=256)   # EP：之后跨 EP 组求和
# EP + all-to-all（例如 MORI）：x 是按容量分配的接收缓冲区，有效行数 m_valid 是 int32 device 张量
out = a8w4_moe_dyn(x8, W, topk_weights, topk_ids, m_valid, x_scale=x8_scale,
                   expert_offset=ep_rank * E_local, num_experts=256)
```

在包含 `a8w4/` 的目录下运行：

```bash
pytest -q a8w4/tests/test_a8w4_moe.py                        # 正确性（gfx942）
python -m a8w4.tune show   --case mimo-v2.6-flash-tp8         # 查看调优表：每个 token 数选中的配置
python -m a8w4.tune verify --case mimo-v2.6-flash-ep8         # 按表实际调用：耗时、正确性、与未调优默认值对比
pytest -q a8w4/tests/test_a8w4_mori_ep8.py                   # 8 卡 EP8 整层：MORI dispatch + a8w4_moe_dyn + MORI combine 对参考实现
python a8w4/tests/test_a8w4_mori_ep8.py --act silu situv2     # 整层性能（总 token 64..4096），SiLU / SiTUv2 对比；更多用法见 8.1 节
pytest -q a8w4/tests/test_a8w4_mori_tp8.py                   # 8 卡 TP8 整层：a8w4_moe + all-reduce（aiter custom / RCCL）对参考实现
python a8w4/tests/test_a8w4_mori_tp8.py --act silu situv2     # TP8 整层性能（batch 64..4096），SiLU / SiTUv2 对比；更多用法见 8.2 节
```

## 2. 接口与约定

### 2.1 `a8w4_moe`

```
out[t] = Σ_{k : e = topk_ids[t,k] ∈ [e0, e0+E)}  topk_weights[t,k] · W2[e-e0] · act(W1[e-e0] · x[t])
```

E 是 `W` 中的本地专家数，e0 = `expert_offset`。

- **TP**：每卡持有全部专家，以及 inter_dim 的 1/tp。调用时 `expert_offset=0`，`num_experts=None`。
- **EP**：每卡持有 E 个完整专家。调用时 `expert_offset=rank*E`，`num_experts` 填全局专家数。

两种情况返回的都是本卡的部分和，跨卡归约（TP 的 all-reduce、EP 的 reduce-scatter / combine）由调用方完成。

| 参数 | 要求 |
|---|---|
| `x` | bf16 [M, H]，连续 |
| `topk_weights` | f32 [M, topk]。`routed_scaling_factor` 之类的系数要事先乘进去 |
| `topk_ids` | int32 [M, topk]，全局专家 id。EP kernel 只处理 [e0, e0+E) 内的 id，其他值（包括 -1）视为不属于本卡；TP 族 kernel 走 aiter `moe_sorting`，id 必须落在 [0, num_experts) 内 |
| `W` | `prepare_weights` 的输出 |
| `expert_offset` / `num_experts` | 见上；默认 0 / E，即 TP |
| `act`, `situ` | `"silu"`（默认）或 `"situv2"`；`situ=dict(situ_beta, situ_linear_beta, swiglu_limit)`。两族 kernel 都支持（5.8 节）；查表用该 act 的行（6.2 节） |
| `config` | 强制使用某个配置（dict 或名字，例如 `"ep3k l128"`），不查表 |
| `routing`, `csv_path` | 调优表的 key 和路径，默认 `"uniform"` 与 `configs/a8w4_moe_tuned.csv` |
| 返回 | 新分配的 bf16 [M, H] |

### 2.2 其他接口

| 函数 | 作用 |
|---|---|
| `prepare_weights(w1, w2, w1_scale, w2_scale)` | 把 preshuffle 好的 mxfp4 权重转成 fd2 格式的 dict（5.2 节）。只做一次，结果所有 kernel 共用 |
| `shuffle_mxfp4(w1q, w1s, w2q, w2s)` | 未 shuffle 的 checkpoint 张量 → `prepare_weights` 的输入（调用 `aiter.ops.shuffle`） |
| `run_config(cfg, x, W, tw, ti, ...)` | 跑指定配置，不查表 |
| `a8w4_moe_dyn(x, W, tw, ti, m_valid, x_scale=, ...)` | EP all-to-all 接收缓冲区（有效行数在 device 上），见 2.4 |
| `lookup_config(shape, tokens, act=, capturing=, routing=, csv_path=)` | 查表得到配置；`shape = (H, I, E, E_global, topk)` |
| `dyn_config(shape, tokens, act=, routing=, csv_path=)` | `a8w4_moe_dyn` 用的配置（6.2 节）；EP kernel 不支持该 shape 时抛 `ValueError` |
| `supported(cfg, shape, act=, capturing=, tokens=)` | 某个配置在该 shape / 上下文下能否运行（只检查结构限制） |
| `default_config(shape, tokens)` | 表里没有该 shape 时的回退配置 |
| `config_name` / `parse_config` | 配置 dict ↔ 名字 |
| `weight_shape(W)` | 返回 (E, H, I) |

### 2.3 使用约定

- **权重**：`prepare_weights` 返回的 dict 包含：
  - `w1`, `w2`：nibble 重排后的 fp4；
  - `w1_scale`：每个 32-K 组的 delta；
  - `w2_scale`：同样是 delta，后面再接每通道最大 code；
  - `w1_exp`, `w2_exp`：每通道最大 e8m0 code，其中 `w2_exp` 是 `w2_scale` 尾部的 view，不额外占显存；
  - `w*_fast_dec=2`。
  
  这套权重只有本包的 kernel 能读，aiter 的 CK / asm 路径不能用。
- **首次调用**：每个配置第一次被调用时会 JIT 编译，FlyDSL 缓存为空时约 5–13 s（tp1k 5.5 s、ep3k l128 8 s、ep1k 13 s）。编译结果落盘在 `~/.flydsl/cache`（可用 `FLYDSL_RUNTIME_CACHE_DIR` 改位置），之后的进程直接加载。
- **CUDA graph**：
  - `a8w4_moe` 会检查 `torch.cuda.is_current_stream_capturing()`，capture 期间只选 graph-safe 的配置（排除 `tp1kq`）。
  - capture 之前要用同样的 token 数先 eager 调用一次，完成编译并分配 workspace / 计数器。sglang 的 graph runner 在 capture 前本来就会先跑 warmup。
- **并发**：`ep_prep`、`ep1k`、`tp1kq` 的全局计数器和 workspace 是每张卡一份（由 kernel 内最后一个 WG 复位）。同一 stream 上的顺序调用没有问题；如果框架会在同一张卡的两个 stream 上并发执行 MoE（例如两个 micro-batch 各用一个 stream），要先改成每个 stream 一份 workspace。
- **输出**：结果由 bf16x2 atomic 累加得到，topk > 1 时多次运行的结果不逐位相同，cos 差约 1e-7（EP）到 1e-5（TP，token 很少时）。

### 2.4 `a8w4_moe_dyn`：EP all-to-all 接收缓冲区（MORI）

EP 用 all-to-all（例如 MORI）分发 token 时，本卡收到的是按容量分配的缓冲区：只有前 `m_valid[0]` 行有效，而这个数只在 device 上，host 要读就得同步一次。`a8w4_moe_dyn` 直接在这个缓冲区上计算：

```python
out = a8w4_moe_dyn(x, W, topk_weights, topk_ids, m_valid, x_scale=None,
                   expert_offset=0, num_experts=None, act="silu", situ=None,
                   est_tokens=None, out=None, config=None, routing="uniform",
                   csv_path=DEFAULT_CSV)
```

| 参数 | 要求 |
|---|---|
| `x` | [M, H]，M 是缓冲区容量，连续。两种格式：bf16（`x_scale=None`，由 prep 按行量化）；fp8 FNUZ 加 `x_scale`，`x_scale` 为 f32 [M, H/128]（每（token, 128）一个，即 MORI 的 blockwise fp8 dispatch）或 [M]（每 token 一个） |
| `topk_ids` / `topk_weights` | int32 / f32 [M, topk]，全局专家 id。只有前 m_valid 行有效 |
| `m_valid` | int32 device 张量。kernel 读 `m_valid[0]`，并截到 [0, M] |
| `est_tokens` | host 端对有效行数的估计，默认 M。只用来选配置和定 GEMM 的 grid；WG 会循环处理全部实际 tile，所以估计偏低也正确，只是会慢 |
| `out` | 可选，bf16 [M, H]，返回值就是它。第 m_valid 行及以后的内容未定义 |
| `config` | 强制使用某个配置，只能是 `ep3k` |
| `act`, `situ` | 同 `a8w4_moe` |

- **只走 EP kernel（ep3k）**：要求本地专家 ≤ 32、64 % topk == 0、M < 65536。更大的缓冲区要分块调用，sglang 每块 32768 行。
- **配置**：`dyn_config(shape, est_tokens)`，见 6.2 节。
- **没有 host 同步**：同一个 CUDA graph 可以服务任意 m_valid ≤ M。
  - capture 时 M 是容量；eager 时调用方可以先读出计数，只传有效的前几行（sglang 就是这样做的）；
  - 两者的 M 不同，可能用到不同的编译变体（ep_prep 按 M·topk 是否超过 2048 分为单 WG / 多 WG 排序）；
  - capture 期间第一次编译、加载 kernel 是可以的：FlyDSL 缓存为空时 capture 和 replay 都正常。
- **性能**见 7.7 节。bf16 输入且估计准确时与 `a8w4_moe` 相同；fp8 分组 scale 输入在 ≥512 tokens 时慢 3–8%；估计值等于容量时，小 batch 多 14–28 us。

## 3. 文件与依赖

```
a8w4/
  __init__.py                  公开接口
  moe.py                       a8w4_moe / a8w4_moe_dyn、5 类 runner、能力过滤 supported()、调优表查找（lookup_config / dyn_config）
  weights.py                   prepare_weights（fd2 权重）、shuffle_mxfp4、weight_shape
  tune.py                      调优 CLI：tune / merge / show / verify / candidates
  ref.py                       测试与调优用：随机权重和路由、模拟 fp8 量化的 fp32 参考、冷 cache 计时
  configs/a8w4_moe_tuned.csv   调优表（每个点保存全部候选的耗时）
  configs/README.md            调优表的内容与生成命令
  tests/test_a8w4_moe.py       pytest
  tests/test_a8w4_mori_ep8.py  8 卡 EP8 整层（MORI dispatch → a8w4_moe_dyn → MORI combine）：pytest 查正确性，命令行测性能（8.1 节）
  tests/test_a8w4_mori_tp8.py  8 卡 TP8 整层（a8w4_moe → all-reduce），EP8 脚本的 TP 对照：pytest 查正确性，命令行测性能（8.2 节）
  kernels/                     FlyDSL device 代码
    row_quant.py               x 按行 fp8 量化（行数是运行时参数）
    tp_gemm1.py                TP 族 gemm1（gate|up + 激活 + q8s 量化 epilogue）；fd2 权重准备函数也在这里
    tp_gemm2.py                TP 族 gemm2（down + 路由加权 atomic 写回；base / nt4）
    tp_fmoe.py                 tp1k：BM 64 单 kernel 融合 MoE
    tp_fmoe_q.py               tp1kq：tp1k 的 persistent 工作队列版
    ep_prep.py                 ep3k 的 prep：本地专家排序 + x 量化 + tile 调度 + 路数统计；m_valid 变体（MDYN）
    ep_gemm.py                 ep3k 的 gemm1 / gemm2、配置表 CFGS、入口 ep_moe 和 ep_moe_dyn（XG、DYN 变体）
    ep_small.py                ep1k：单 launch 的小 batch 融合
  _vendor/                     从 aiter/ops/flydsl/kernels（aiter 7cd70aca8）复制的 FlyDSL helper，import 已改为相对路径
```

| 依赖 | 版本 | 用途 |
|---|---|---|
| GPU | gfx942（MI300X / MI325X） | fp8 FNUZ MFMA；硬件假设见 9.2 |
| torch | ROCm 版，测试用 2.9.1+rocm7.2.0 | |
| FlyDSL | **0.3.2**（用到 `flydsl.expr.meta.dsl_loc_tracing` 等新 API，0.1.x 导入时就会失败） | 全部 kernel |
| aiter | 测试用 tq/mimo-moe 分支 | 运行时只有 TP 族用 `aiter.fused_moe.moe_sorting`；`shuffle_mxfp4` 用 `aiter.ops.shuffle`（纯 torch 重排）；`ref.py` 和测试用 `aiter.get_torch_quant` 生成权重 |

EP 族（`ep3k` / `ep1k`）和 `prepare_weights` 在运行时都不需要 aiter。

**复制到其他工程**：

- 整个目录复制过去即可，包内全部使用相对 import，也可以放进别的包里。只有 `tests/` 和 `tune.py` 的命令行入口按顶层包名 `a8w4` 导入，改了包名要同步修改。
- 调优表按 shape 存放，其他模型要先调优（第 6 节）。
- 换了机器（不同 SKU 或功耗墙）要先跑 `verify`，必要时重新调优。

## 4. kernel 一览与按 batch 的选择

| 类别 | 流程（launch 数） | 适用场景 | 结构限制（`supported()`） | graph-safe |
|---|---|---|---|---|
| `tp2k` | moe_sorting → row_quant → gemm1（q8s）→ gemm2（4） | TP 小 / 中 batch | I % 256、H % 512；`g2nt4` 只支持 I = 256 | 是 |
| `tp1k` | moe_sorting → row_quant → 融合 kernel（3） | TP 大 batch | I % 256；每个 CTA ≤ 512 列 inter | 是 |
| `tp1kq` | 同上，融合 kernel 换成 persistent 工作队列（3） | TP 4096 tokens 左右 | 只支持 I = 256 | **否** |
| `ep3k` | ep_prep → ep gemm1 → ep gemm2（3） | EP 全部 batch；`a8w4_moe_dyn` 只用它 | 本地专家 ≤ 32、64 % topk == 0、gemm2 的 K 循环 I/ARB ≥ 8、M < 65536 | 是 |
| `ep1k` | 单个 persistent launch（1） | EP 极小 batch（实测没有一个点最快） | 同 ep3k，另需 topk = 8、I % 2048、M ≤ 512 | 是 |

配置名的格式：

- `tp2k bm{32|64} g1bcm{1|2} g2{base|nt4}`：bm 是排序的 block_m；g1bcm 是 gemm1 读 W 的 cache 策略；g2 是 gemm2 的变体（5.3 节）。
- `tp1k bcm{1|2} ks{N}`：ks 是每个 64 行 block 分给几个 CTA。
- `tp1kq bcm1`。
- `ep3k {t16|s16|s32|m64|l128}`（见 5.6 节的 CFGS）。
- `ep1k m512`。

MiMo-V2.6-Flash 上的调优结果（完整表格见第 7 节，`tune show` 可列出全部 35 个点）：

- **TP8**（I 256）：
  - ≤1280 tokens 选 `tp2k`。多数是 bm32；从 3 tokens 起 gemm2 多数选 nt4；相邻几个配置只差 1–3%。
  - ≥1536 tokens 选 `tp1k ks1`。
  - 4096 tokens 选 `tp1kq`；graph capture 时退回 `tp1k`，慢 6%。
- **TP4**（I 512）：≤768 tokens 选 `tp2k bm32 g2base`；≥1024 tokens 选 `tp1k`，多数是 ks2。
- **EP8**（每卡 32 专家，I 2048）：
  - 按 token 数：≤16 选 `ep3k t16`（≤4 tokens 时比 s16 快 7–11%）；24–384 tokens t16 与 s16 交替胜出（相差 ≤3%）；448–768 选 `s32`；1024–1280 选 `m64`；≥1536 选 `l128`。
  - 同一 shape 上，TP 族 kernel 慢 10–46%，`ep1k`（≤512 tokens）慢 14–35%。

## 5. 算法

### 5.1 数据流与数值格式

```
x bf16 [M,H] ──行量化──> x8 fp8 [M,H] + xs f32 [M]             （每行 amax/240）
topk_ids/weights ─排序─> sorted_ids / sorted_weights / sorted_expert_ids / num_valid_ids
gemm1：按 sorted 行 gather x8，x8·W1ᵀ（gate|up，fp32 累加）× xs × 通道因子 → act(gate)·up
       → 每（行, 128 列）一个 scale 量化成 fp8：A2 [sorted, I] + s2 [sorted, I/128]
gemm2：A2·W2ᵀ（按 128-K 组的 scale 重标定）× 通道因子 × 路由权重 → bf16 → atomic 加到 out[token]
```

- **MFMA 与 fp8**：MFMA 用 16×16×32 fp8，fp32 累加。fp8 是 gfx942 的 FNUZ 格式，量化时饱和到 ±240、round-to-nearest-even、不产生 -0。
- **排序格式**：沿用 aiter `moe_sorting` 的约定。
  - `sorted_ids = token | slot<<24`，填充行为 `M | topk<<24`；
  - 每个专家的行数补齐到 block_m 的整数倍；
  - `sorted_expert_ids` 每个 block 存一个本地专家号；
  - `num_valid_ids[0]` 是补齐后的总行数。
  
  `ep_prep` 输出同样的格式，只多了两处：本卡单路 token 的 sorted id 置位 bit 31（`SID_SINGLE`），以及额外的 tile 调度和路数统计（5.6 节）。
- **输出写法**：`out` 先清零，再用 bf16x2 atomic（agent scope）累加。EP 的 single_store 例外：本卡只有 1 路的 token 直接 store，不做 atomic。

### 5.2 mxfp4 解码与 fd2 权重（`prepare_weights`）

**解码方式。** mxfp4 是 E2M1 nibble，每 1×32 个元素一个 E8M0 scale。kernel 里用 `v_perm` 查字节表，把 fp4 转成 fp8：

- 每个 32-K 组的 scale 先换算成相对于该输出通道最大 scale 的差值，这个差值编进查找表（fp8 指数平移）；
- 通道最大 scale 对应的因子 2^(max−132) 在 epilogue 里以 fp32 乘上；
- 比通道最大值小 12 个以上二进制量级的组，解码结果为 0（flush）。

**权重是静态的，所以把能挪的计算都挪到权重准备阶段。** 改写后的结果与原解码逐位一致：

| 级别 | 权重准备时做的改写 | kernel 里省掉的操作 |
|---|---|---|
| fd1 | 每个 dword 内把 nibble 顺序 k0..k7 改存为 (k_j \| k_{j+4}<<4)；-0 改成 +0 | 拼回 K 顺序的 2 条 `v_perm`；符号直接用位运算加上（又省 2 条 `v_perm` 和符号表） |
| fd2 | 再把 scale 字节改存为 `min(通道最大 code − 组 code, 11)×8`，会被 flush 的组在 W 里直接写 0 | 查找表的构建只剩 1 条 `v_perm` 加 sub / and / sub，不再需要 `v_min`、`v_cndmask` 和慢速的 `v_mul_lo` |

- **效果**：gemm1 K 循环每个 wave 每次迭代的 VALU 从 531 条降到 325 条（−39%），gemm2 整个 kernel 的 VALU −29%。端到端只快 7–11%，因为 750 W 功耗墙下瓶颈转成了功耗。
- **布局**：改写直接在标准 preshuffle 布局上做，不需要先 unshuffle。
  - W 字节按 `(N/16, K/128, klane 4, nlane 16, 16 B)` 排列；
  - scale 字节按 `(N/32, K/256, klane, nlane, kpack 2, npack 2)` 排列；
  - 准备函数会断言由此换算出的每通道最大值与 `channel_exponents` 一致，以此校验映射。
- **W2 的通道最大 code**：fd2 改写 scale 后，kernel 没法再从 scale 里算出通道最大值，所以把这些值接在 `w2_scale` 后面。TP kernel 从偏移 `WS_BYTES` 处读，EP kernel 读 `w2_exp`（同一段字节的 view）。

### 5.3 tp2k：gemm1（q8s）+ gemm2（base / nt4）

**gemm1**（`tp_gemm1.a8w4_gemm1`，`epilog="q8s"`，`fast_dec=2`）

- **tile 划分**：grid = sorted block 数 × (I/128)。每个 WG 有 4 个 wave，按列分 tile_n=128 列 inter（W 的 gate、up 各 128 行）。K-tile 为 256，A 按 token gather 后放进 XOR swizzle 的 LDS。
- **`pipeline`**（BM ≤ 32 时默认打开）：在计算当前 K-tile 时预取下一个 K-tile 的 A 和 W，A-LDS 双缓冲。BM ≥ 64 时 VGPR 不够（> 168），不开。
- **`dyn_m`**（bm64）：运行时检查本 block 第 32 行是否已是填充行，是的话只算前 32 行。两种行数变体都编译进同一个 kernel。
- **`xcd_swizzle=1`**：每个 XCD 拿到一段连续的 m-block，同一专家的 W 能在 XCD 的 L2 里复用。
- **`b_cache_mod`**（配置名里的 g1bcm）：读 W 的 cache 策略，2 表示 nt。
- **q8s epilogue**：累加结果 × xs × 通道因子 → act → bf16 暂存在 LDS → 每（行, 128 列）求 amax → fp8 + f32 scale，按 sorted 行写出。输出就是 gemm2 `a_group=128` 要读的格式，不再需要单独的量化 kernel。
- **行数是运行时参数**：输出行数 `rows` 作为 launch 参数传入，所以不会按 token 数重新编译。

**gemm2**（`tp_gemm2.a8w4_gemm2`，`a_group=128`，`fast_dec=2`）

- **`g2base`**：
  - 每个 WG 负责一个（m-block, 128 输出列），K = I，按 256 切 K-tile；
  - 每个 128-K 组的 A 有自己的累加器，dequant 时乘各自的 scale 再相加；
  - `dyn_rows`：跳过全是填充行的 16 行子块；
  - epilogue：乘路由权重后转 bf16，暂存在 LDS，每行以 128 B 连续的 bf16x2 atomic 写出。
- **`g2nt4`**（`n_tiles=4, epilog="direct64", row_vgpr=True`）：
  - 一个 CTA 负责同一 m-block 的 4 个相邻 n-tile（grid 缩小为 1/4）：A 只 stage 一次；计算 tile t 时预取 tile t+1 的 W；
  - 每行的静态数据（两组 A scale 的比值、s × 路由权重、token id）放在 VGPR 里；只用一组累加器，在组边界重标定；
  - epilogue 不经过 LDS，每行写 64 B 段。
  
  这种做法要求 A 只有一个 K-tile，所以只支持 I = 256。

### 5.4 tp1k：单 kernel 融合（`tp_fmoe.a8w4_fmoe_bm64`）

- **CTA 结构**：一个 CTA 处理一个 64 行 block（同一个专家），4 个 wave。
- **gemm1 阶段**：按 128 列一块处理 inter。K 循环遍历 H（K-tile 256），A 用单槽 LDS，下一个 K-tile 预取到寄存器。算完后 act → bf16 暂存 → 每（行, 128 列）量化成 fp8（与 q8s 数值相同）→ 写进 LDS 槽 → 每个 wave 把自己的 gemm2 A 片段读进 VGPR。中间结果 A2 不经过 HBM。
- **gemm2 阶段**：32 个 n-tile，每个 128 列，A 常驻 VGPR，W2 预取距离为 1。一组累加器在 K 组边界重标定。tile t−1 的 epilogue 穿插在 tile t 的 K 步之间；atomic 不经过 LDS，每行写 64 B 段。
- **占用率**：≤256 VGPR、24.5 KB LDS，每个 CU 可放 2 个 CTA，一个 CTA 的 atomic 写回能与另一个 CTA 的 gemm1 重叠。
- **`ksplit`**：把一个 block 的 I/128 个列块分给 ksplit 个 CTA。每个 CTA 只算自己那部分 gemm1，再做相应 K 段的 gemm2（split-K），部分和经同一套 atomic 相加。每个 CTA 负责 256 列时用 `waves_per_eu=2`；负责 384–512 列时 A2 要占更多 VGPR，只能用 1。
- **XCD 映射**：每个 XCD 处理一段连续的 m-block，同一专家的多个 block 共享 L2。
- **什么时候赢**：大 batch 时，省掉了中间结果写回 HBM 再读回，atomic 也能和计算重叠。小 batch 时 CTA 数只等于 block 数（tp2k 是 block 数 × n-tile 数），所以输。

### 5.5 tp1kq：persistent 工作队列（`tp_fmoe_q.a8w4_fmoe_bm64q`）

- **调度**：每个 CU 常驻 2 个 CTA。每个 XCD 一条队列，CTA 通过读 HW_REG_XCC_ID 确定自己所在的 XCD。队列顺序是 g1(0..D−1), g2(0), g1(D), g2(1), …，第一轮之后约一半 CTA 在做 gemm2（atomic），另一半在做 gemm1。
- **中间结果**：A2（fp8 64×256 + scale，每个 block 约 16.5 KB）经全局内存传递，生产者和消费者在同一个 XCD 上，所以只是一次 L2 往返。每个 block 的 ready flag 写入 launch epoch，队列计数器按 epoch 奇偶双缓冲。
- **限制**：只支持 I = 256，有约 10 个 VGPR spill 和 62 个 SGPR spill。只在 4096 tokens 附近比 tp1k 快（TP8 上快 6%）。
- **不是 graph-safe**：epoch 是 host 每次调用加 1 的 launch 参数，graph 回放时它不变，flag 会被误判为已就绪，计数器也不会复位。此外 workspace 会随 block 数增长而重新分配。

### 5.6 ep3k：prep + gemm1 + gemm2（`ep_prep.py`、`ep_gemm.py`）

**`ep_prep`**（1 次 launch，替代 `moe_sorting` + `quantize_rows`）

- **排序**：只对本卡专家排序。
  - M·topk ≤ 2048 时，单个 WG 只用 LDS atomic 完成；
  - 更大时用多个 WG：每个 WG 在 LDS 里统计自己的 2048 个条目，每个专家用一次全局 atomic 预留区间，再在排序 WG 之间做一次 grid barrier（这些 WG 必须同时驻留）；之后算偏移、scatter、写填充行；
  - 最后一个 WG 复位计数器，所以是 graph-safe 的；
  - 专家内部的行顺序取决于 atomic 的先后，不确定。
- **x 量化**：由另外的 WG 完成，只量化有本地专家的 token 行，结果与 `quantize_rows` 逐位一致。
- **tile 调度**（m64 / l128 使用）：
  - 把 m-block 按专家顺序、按估计代价（max(64, 行数变体)）切成 8 段连续区间，每个 XCD 一段，热点专家可以跨 XCD；
  - 段内重的 tile 排在前面；
  - GEMM 中 WG bx 处理第 bx % 8 段的第 bx // 8 个 tile，由硬件按 WG 顺序动态派发，相当于每个 XCD 内部做一次 LPT 调度。
- **路数统计**：统计每个 token 在本卡的路数，并在单路 token 的 sorted id 上置 `SID_SINGLE`。

**GEMM 主循环**（`_make_core`，gemm1 和 gemm2 共用）

- **tile 与 wave**：一个 WG 处理一个（m-block, n-block）tile，NW 个 wave 按列切分。每个 wave 负责自己那些列的全部行，所以一个解码出的 W 片段要喂 R/16 次 MFMA，同一 WG 内不会重复解码 W。
- **W 的读取**：W 从全局内存直接读进 VGPR（MFMA 操作数布局），提前 PF 个 K-tile 发出。运行时的 K 循环展开 2·PF 次，每个预取 buffer 在两组静态寄存器之间轮换，循环回边处不需要拷贝。
- **A 的读取**：A 的 K-tile 经双缓冲、XOR swizzle 的 LDS，A 片段提前一个 MFMA step 读出。
- **`sch=3`**：每个 K-tile 最后一步的 MFMA 推迟到下一个 K-tile 的 barrier 之后，填补 barrier 等待。
- **ROWV**：编译多个行数变体（如 32 / 64 / 128 行），每个 WG 选能覆盖本 block 有效行数的最小变体。
- **`_opaque_tx`**：每个 tile 的 body 开始前，用一条带副作用的 `v_mov` 复制一份 `tx`，阻止 LLVM 把按 lane 计算的地址提到循环外。这样所有配置都没有 spill。

**gemm1**

- 每个 WG 处理 NI=128 列 inter（gate + up）。a8w4 的输出与 q8s 格式相同：fp8 [sorted, I] + f32 [sorted, I/128]。
- **TR epilogue**（m64 / l128）：把 W 作为 MFMA 的 A 操作数，累加器因此是转置的：每个 lane 持有一行的 4 个连续列，act 和量化都在寄存器里完成。
- 计算过程中顺带把 gemm2 的 `out` 清零；开启 single_store 时只清零路数 ≠ 1 的行。

**gemm2**

- 每个 WG 处理 NO 个输出列。
- A 的分组 scale：在 LDS 里建一张比值表 s_{g−1}/s_g，只用一组累加器，在每组开头乘一次比值，epilogue 再乘最后一组的 scale。
- epilogue：乘路由权重后转 bf16，经 LDS 以 bf16x2 atomic 写出（每行 128 B）；single_store 的行直接 store。

**配置**（`ep_gemm.CFGS`；所有配置都是 sch 3、PF 2、W 读用 nt）

| 名字 | bm | ROWV | gemm1 | gemm2 | tile 顺序 |
|---|---:|---|---|---|---|
| `t16` | 16 | 16 | NI 128, NW 8 | NO 64, NW 4 | m-major |
| `s16` | 16 | 16 | NI 128, NW 8 | NO 256, NW 8 | m-major |
| `s32` | 32 | 32 | NI 128, NW 8 | NO 256, NW 8 | m-major |
| `m64` | 64 | 32 / 64 | NI 128, NW 8, ARB 128, TR | NO 256, NW 8, ARB 128 | prep 调度 |
| `l128` | 128 | 32 / 64 / 128 | NI 128, NW 8, ARB 128, TR | NO 256, NW 8, ARB 128 | prep 调度 |

ARB 是每个 K-tile 每行 A 的字节数，默认 256。`t16` 的 gemm2 用窄 n-tile（NO 64），让更多 CU 同时读 W，对 ≤8 tokens 有效。

**接收缓冲区变体**（`ep_moe_dyn`，即 `a8w4_moe_dyn` 的实现，来自 sglang PR #45 的 `ep8_moe_dyn`）

- **`ep_prep(m_valid=...)`（MDYN）**：
  - M 是缓冲区容量，决定填充 token id 和各 buffer 的布局；
  - 只对前 m_valid 行排序和量化（m_valid 在 device 上读，截到 [0, M]），`num_valid_ids[1]` 写成 m_valid；
  - 按 M 编译成多 WG 排序的 kernel，如果运行时发现有效条目放得进一个 WG，就改走单 WG 的 LDS 排序。
- **gemm1 `XG`**：x 的 scale 按（token, 128）分组时，处理方式与 gemm2 的 A 分组 scale 相同：只用一组累加器，每组开头乘 s_{g−1}/s_g，epilogue 再乘最后一组的 scale。
- **gemm1 / gemm2 `DYN`**：
  - grid 按 `est_tokens` 开：`nblk = min(块数上限, est·topk/bm + E)`，WG 以 grid 为步长循环处理全部实际 tile；
  - gemm1 不再按 M 清零整个 `out`，改为在 tile 之后，只清零有效行中本卡路数 ≠ 1 的行。单路的行由 gemm2 整行写入，无效行不会被读。

### 5.7 ep1k：单 launch 的小 batch 融合（`ep_small.ep_moe_small`）

- **路由**：每个 WG 自己算一遍路由，不做全局排序。每个 wave 读 64 个 token 的 topk id，用 ballot 为每个（专家, 64 token）建一个 64 位 mask。专家 e 的第 k 行就是它 mask 中第 k 个置位的 token，结果是确定的，所有 WG 算出来一致。
- **setup**：每个 WG 分担一部分工作：清零 `out`，并把 x 量化成 fp8。数据用写穿 store（sc0|sc1）加 `s_waitcnt` 释放；完成情况用两级计数器统计（32 个子计数器），因为所有 WG 打同一个地址会让 atomic 串行化，每次约 40 ns。
- **任务**：静态 round robin 分给各 WG，先排所有 gemm1 任务。gemm1 输出 bf16 的中间结果；gemm2 任务先等对应 m-block 的 gemm1 计数器到齐，再在 stage A 时按（行, 128）量化，最后 atomic 写出。每个任务在开始等待之前先发出 W 预取。
- **graph-safe**：最后一个 WG 复位全部计数器。
- **限制**：grid 固定为 304 × 2，要求所有 WG 能同时驻留；M ≤ 512（编译期容量）。
- **结果**：在 MiMo EP8 上没有一个点最快（1 token 时 42.5 us，ep3k t16 34.8 us）。保留它是因为它正确、graph-safe，供以后改进。

### 5.8 激活

- **silu**：`silu(g)·u`，所有 kernel 都支持。
- **situv2**（所有 kernel 都支持）：
  - 公式：`β·tanh(g/β)·sigmoid(g) · β_l·tanh(u/β_l)`，计算前先 clamp：g ≤ L，|u| ≤ L；
  - β、1/β、β_l、1/β_l、L 是 5 个运行时 f32 launch 参数（`tp_gemm1.situ_args` 生成，β ≤ 0 时报 `ValueError`）；kernel 按 act 分别编译，kernel 名加 `_situv2`；silu 也带这 5 个参数，但不使用；
  - 位置：TP 族在 gemm1 / 融合 kernel 的 epilogue；EP 族在 ep gemm1 的两种 epilogue（TR 和非 TR）以及 ep1k 的 gemm1 item。都调用同一个 `_vendor/act.gate_up_act`，gemm2 不变；
  - 实测（7.8 节）比 silu 慢：EP 族（ep3k）0.8–3.1%，`a8w4_moe_dyn` 0.6–4.4%，TP 族 1.5–4.7%。多出的只是 gemm1 epilogue 里的 VALU（tanh、sigmoid 各一组 exp2 + rcp），大 batch 时占比稍高。
- MiMo-V2.6-Flash 用的是 SiLU（sglang `mimo_v2.py` 要求 `hidden_act == "silu"`）。调优表里 SiLU 和 SiTUv2 是分开的两组行（6.1 节）。

## 6. 调优机制

### 6.1 调优表 `configs/a8w4_moe_tuned.csv`

列：`model_dim, inter_dim, experts, experts_global, topk, routing, act, num_tokens, config, us, cos`。

**每个点保存全部候选的耗时**，而不是只存最快的那个，这样运行时才能按上下文过滤（capture 时排除 tp1kq，超过 512 tokens 时排除 ep1k）。现有内容：

- 3 个 shape（TP8 4096/256/256/256/8、TP4 4096/512/256/256/8、EP8 4096/2048/32/256/8），routing 都是 uniform；
- act：3 个 shape 都有 silu 行，TP8、EP8 另有 situv2 行；
- 每组（shape, act）35 个点（1, 2, 3, 4, 6, 8, 12, … , 24576, 32768）；
- 共 2000 行（silu 1140、situv2 860）。

各组的来源和生成命令见 [configs/README.md](configs/README.md)。

### 6.2 查表规则（`lookup_config`）

1. key 为 (H, I, E, E_global, topk, routing, act)。该 shape 没有调用时那个 act 的行时，用同 shape 的 silu 行（例如 TP4 的 situv2）；没有 act 列的旧表全部按 silu 读。
2. 取第一个 `num_tokens ≥ M` 的点；M 超过最大点时取最大点。
3. 在该点的候选中，按耗时从小到大，返回第一个满足 `supported(cfg, shape, act, capturing, tokens=M)` 的配置。
4. 表里没有这个 shape 时用 `default_config`：EP kernel 能跑就按原来的 EP 阈值选（≤8 t16、≤480 s16、≤960 s32、≤2176 m64、其余 l128），否则用 `tp2k bm64 g1bcm1 g2base`。
5. `dyn_config`（`a8w4_moe_dyn` 用）：同样按 `est_tokens` 取点，但只考虑 ep3k 候选。表里没有该 shape 时按上面的 EP 阈值；阈值对应的配置不支持时，依次试其他 ep3k 配置；都不支持则抛 `ValueError`。

表按文件路径和 mtime 缓存，文件更新后自动重新读取。

### 6.3 tuner（`tune.py`）

- **候选**（`candidates(shape)`，再经 `supported` 过滤）：
  - tp2k：bm × g1bcm × g2，共 8 个；
  - tp1k：bcm × ksplit，ksplit 取每个 CTA 256 / 384 / 512 列对应的值；
  - tp1kq；
  - ep3k 的 5 个配置；
  - ep1k。
  
  过滤后 TP8 有 11 个候选，TP4 有 8 个，EP8 有 14 个。
- **流程**：每个 token 数跑 `--rounds`（默认 3）轮，每轮重新抽一次路由。小 batch 时本卡的工作量取决于抽到的路由，例如 EP8 1 token 时约 1/3 的抽样没有本地专家。每轮对每个候选：
  - 先与参考实现比 cos，≥ 5e-3 的判为 BAD 并剔除；
  - 再计时（冷 cache，`--iters` 次取中位数）；
  - 同一轮内各候选交错执行，时钟和功耗的漂移会均匀落到每个候选上。
  
  每个候选最终取各轮的中位数。每测完一个 token 数就写一次文件（同 key 的行覆盖）。
- **计时**（`ref.time_cold`）：
  - 每次迭代前读一遍 512 MB，用干净行把 L2 和 256 MB 的 MALL 全部挤掉。如果反复调用不刷 cache，小 batch 的权重会直接从 MALL 读到。用读而不是读改写，是为了不留下 dirty 行，否则写回会抢 HBM 带宽；
  - 再让 GPU 空转约 1 ms（`torch.cuda._sleep`），这段时间里 host 已经把本次调用的所有 launch 入队，因此 host 开销不计入，和 graph 回放的情况一致；
  - 用 event 包住调用，每次迭代都同步，最后取中位数。

```bash
cd <包含 a8w4/ 的目录>
python -m a8w4.tune candidates --case mimo-v2.6-flash-ep8
# 多卡并行：每张卡分一部分 token 数，最后合并
T="1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 320 384 448 512 640 768 1024 1280 1536 2048 2560 3072 4096 6144 8192 12288 16384 24576 32768"
for g in 0 1 2 3; do
  HIP_VISIBLE_DEVICES=$g python -m a8w4.tune tune --case mimo-v2.6-flash-ep8 \
      -t $(echo $T | tr ' ' '\n' | awk -v g=$g 'NR % 4 == g') --out /tmp/ep8_$g.csv &
done; wait
python -m a8w4.tune merge /tmp/ep8_*.csv          # 合并进 configs/a8w4_moe_tuned.csv
python -m a8w4.tune show   --case mimo-v2.6-flash-ep8
python -m a8w4.tune verify --case mimo-v2.6-flash-ep8 -t 1 100 3000 20000
# situv2：同样的命令加 --act situv2，结果写成 act=situv2 的行（完整命令见 configs/README.md）

# 新 shape：H,I,E_local,E_global,topk（也可以把它加进 tune.py 的 CASES）
python -m a8w4.tune tune --shape 6144,256,384,384,8 --out /tmp/new.csv
```

- **调优时机器要空闲**：调优时 8 张卡各跑一个进程，事后在空闲卡上单独复测，大 batch 的结果与表中数值相差 ±5–8%，方向不固定。
- **路由分布**：`--routing hot`（1/32 的专家分数 +2）会写成另一组行，调用时用 `a8w4_moe(..., routing="hot")` 选这组表。
- **situv2**：`--act situv2` 按 situv2 计时，结果写成单独的一组行，调用时按 act 查（6.2 节）。
  - 参数取 `ref.SITU`。β / clamp 是 launch 参数，不影响耗时，所以这组行对任意 SiTUv2 参数都适用。
  - TP8、EP8 已按 situv2 调优。同一配置、同一点上，situv2 / silu 的中位数是 1.022（TP8）和 1.012（EP8），p5–p95 分别为 0.992–1.055 和 0.975–1.045。
  - 35 个点里，TP8 有 8 个点（16、48、64、128、320、448、640、16384）、EP8 有 4 个点（96、192、320、1536）选中的配置与 silu 不同，都是候选之间本来就相差很小的点。按 situv2 实测，新的选择比沿用 silu 的选择只快 0–1.3%，在测量误差以内。
  - capture 时，TP8 4096 tokens 的选择从 `tp1k bcm1 ks1` 变成 `tp2k bm64 g1bcm1 g2nt4`（silu 下两者相差不到 0.1%）。

## 7. 性能

### 7.1 环境与读法

- **环境**：8× MI300X（gfx942，304 CU，192 GB，功耗上限 750 W），ROCm 7.2，torch 2.9.1，FlyDSL 0.3.2，aiter tq/mimo-moe（c9399033d）。
- **数据来源**：调优表（冷 cache，中位数；8 张卡同时调优），uniform 路由。耗时包含排序和量化，即整个 `run_config` 调用。
- **mem 下限**：（本卡活跃专家的权重 + 激活读写）/ 4.0 TB/s。
- **fp8 下限**：gemm1 和 gemm2 的有效 FLOPs 分别除以 hipBLASLt fp8 dense GEMM 在同形状上的实测吞吐（1080 / 900 TFLOPS）。
- 这两个上限都是在本机功耗墙下用 `torch.sum` 和 `torch._scaled_mm` 实测的。TFLOPS 一列按有效计算量 6·rows·I·H 除以实测时间得到。

### 7.2 TP8（H 4096，I 256，256 专家，topk 8）

| tokens | 选中配置 | us | 最快 tp2k | 最快 tp1k | tp1kq | mem 下限 | × mem | × (mem+fp8) | TFLOPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tp2k bm32 g1bcm2 g2base` | 43.6 | 43.6 | 96.3 | 110.3 | 3.3 | 13.03 | 12.84 | 1 |
| 2 | `tp2k bm32 g1bcm2 g2base` | 43.9 | 43.9 | 97.8 | 113.1 | 6.7 | 6.56 | 6.46 | 2 |
| 4 | `tp2k bm32 g1bcm2 g2nt4` | 47.8 | 47.8 | 101.1 | 114.5 | 13.1 | 3.65 | 3.59 | 4 |
| 8 | `tp2k bm32 g1bcm2 g2nt4` | 52.5 | 52.5 | 104.2 | 117.3 | 22.7 | 2.31 | 2.27 | 8 |
| 16 | `tp2k bm32 g1bcm1 g2base` | 70.6 | 70.6 | 113.5 | 126.5 | 40.3 | 1.75 | 1.72 | 11 |
| 32 | `tp2k bm32 g1bcm1 g2nt4` | 99.0 | 99.0 | 112.1 | 137.4 | 69.2 | 1.43 | 1.40 | 16 |
| 64 | `tp2k bm64 g1bcm1 g2nt4` | 118.5 | 118.5 | 141.7 | 177.3 | 94.3 | 1.26 | 1.22 | 27 |
| 128 | `tp2k bm32 g1bcm2 g2nt4` | 143.1 | 143.1 | 162.8 | 199.3 | 105.9 | 1.35 | 1.27 | 45 |
| 256 | `tp2k bm32 g1bcm2 g2nt4` | 151.3 | 151.3 | 170.5 | 205.6 | 108.0 | 1.40 | 1.25 | 85 |
| 512 | `tp2k bm32 g1bcm2 g2nt4` | 165.1 | 165.1 | 177.8 | 213.5 | 109.1 | 1.51 | 1.23 | 156 |
| 1024 | `tp2k bm64 g1bcm2 g2nt4` | 215.7 | 215.7 | 217.6 | 262.0 | 111.1 | 1.94 | 1.33 | 239 |
| 2048 | `tp1k bcm1 ks1` | 343.9 | 344.2 | 343.9 | 353.9 | 115.3 | 2.98 | 1.58 | 300 |
| 4096 | `tp1kq bcm1` | 516.9 | 550.2 | 550.1 | 516.9 | 123.7 | 4.18 | 1.58 | 399 |
| 8192 | `tp1k bcm1 ks1` | 906.5 | 981.0 | 906.5 | 932.6 | 140.5 | 6.45 | 1.65 | 455 |
| 16384 | `tp1k bcm1 ks1` | 1627.3 | 1750.5 | 1627.3 | 1635.9 | 174.1 | 9.35 | 1.65 | 507 |
| 32768 | `tp1k bcm1 ks1` | 3025.8 | 3296.3 | 3025.8 | 3140.4 | 241.2 | 12.55 | 1.62 | 545 |

### 7.3 EP8（H 4096，I 2048，每卡 32 / 256 专家，topk 8）

| tokens | 选中配置 | us | ep1k | 最快 TP 族 | TP 族 / 选中 | mem 下限 | × mem | × (mem+fp8) | TFLOPS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ep3k t16` | 34.8 | 42.5 | 50.8 | 1.46 | 2.2 | 15.59 | 15.37 | 1 |
| 2 | `ep3k t16` | 34.8 | 44.3 | 50.4 | 1.45 | 5.6 | 6.24 | 6.15 | 2 |
| 4 | `ep3k t16` | 36.5 | 48.4 | 51.2 | 1.40 | 14.5 | 2.52 | 2.48 | 6 |
| 8 | `ep3k t16` | 43.5 | 55.9 | 54.3 | 1.25 | 24.5 | 1.77 | 1.74 | 11 |
| 16 | `ep3k t16` | 56.2 | 76.0 | 69.8 | 1.24 | 39.1 | 1.44 | 1.41 | 16 |
| 32 | `ep3k t16` | 97.5 | 111.6 | 110.1 | 1.13 | 67.0 | 1.46 | 1.43 | 15 |
| 64 | `ep3k t16` | 115.8 | 138.4 | 127.7 | 1.10 | 92.7 | 1.25 | 1.21 | 25 |
| 128 | `ep3k t16` | 128.5 | 149.0 | 148.2 | 1.15 | 104.1 | 1.23 | 1.16 | 52 |
| 256 | `ep3k t16` | 137.4 | 168.8 | 161.3 | 1.17 | 107.9 | 1.27 | 1.14 | 92 |
| 512 | `ep3k s32` | 149.4 | 187.1 | 176.9 | 1.18 | 108.9 | 1.37 | 1.12 | 169 |
| 1024 | `ep3k m64` | 193.5 | — | 215.9 | 1.12 | 110.8 | 1.75 | 1.19 | 273 |
| 2048 | `ep3k l128` | 284.8 | — | 324.0 | 1.14 | 114.7 | 2.48 | 1.31 | 365 |
| 4096 | `ep3k l128` | 432.7 | — | 530.2 | 1.23 | 122.2 | 3.54 | 1.34 | 469 |
| 8192 | `ep3k l128` | 760.5 | — | 858.5 | 1.13 | 137.7 | 5.52 | 1.39 | 545 |
| 16384 | `ep3k l128` | 1414.8 | — | 1602.9 | 1.13 | 168.4 | 8.40 | 1.44 | 582 |
| 32768 | `ep3k l128` | 2582.6 | — | 2852.0 | 1.10 | 229.9 | 11.24 | 1.39 | 638 |

### 7.4 TP4（H 4096，I 512，256 专家，topk 8）

| tokens | 1 | 4 | 16 | 64 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 选中配置 | 2k bm32 | 2k bm32 | 2k bm32 | 2k bm32 | 2k bm32 | 2k bm32 | 1k ks2 | 1k ks2 | 1k ks2 | 1k ks2 | 1k ks2 | 1k ks2 |
| us | 43.9 | 53.9 | 114.5 | 222.3 | 263.7 | 290.9 | 382.4 | 614.5 | 958.3 | 1671.0 | 2858.6 | 5520.8 |
| × mem | 6.56 | 2.06 | 1.42 | 1.18 | 1.23 | 1.34 | 1.74 | 2.74 | 4.08 | 6.53 | 9.60 | 14.46 |
| × (mem+fp8) | 6.47 | 2.03 | 1.39 | 1.14 | 1.10 | 1.09 | 1.19 | 1.44 | 1.49 | 1.56 | 1.48 | 1.52 |

### 7.5 时间分解（调优选中的配置，冷 cache，torch profiler，单位 us）

| shape | tokens | 配置 | 排序 | x 量化 | gemm1 / 融合 kernel | gemm2 | kernel 合计 | 端到端 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| TP8 | 1 | tp2k bm32 g1bcm2 g2base | 8.6 | 4.2 | 24.4 | 5.2 | 42.4 | 43.6 |
| TP8 | 16 | tp2k bm32 g1bcm1 g2base | 13.7 | 4.9 | 32.3 | 18.5 | 69.4 | 70.2 |
| TP8 | 64 | tp2k bm64 g1bcm1 g2nt4 | 9.8 | 5.4 | 64.0 | 40.8 | 119.9 | 120.3 |
| TP8 | 512 | tp2k bm32 g1bcm2 g2nt4 | 12.4 | 5.5 | 85.7 | 56.2 | 159.9 | 163.3 |
| TP8 | 2048 | tp1k bcm1 ks1 | 27.0 | 11.5 | 295.6 | — | 334.2 | 336.7 |
| TP8 | 8192 | tp1k bcm1 ks1 | 44.7 | 30.2 | 680.8 | — | 755.7 | 821.2 |
| EP8 | 1 | ep3k t16 | 7.3（prep） | — | 18.7 | 6.7 | 32.7 | 34.6 |
| EP8 | 16 | ep3k t16 | 7.6（prep） | — | 33.8 | 18.6 | 60.0 | 61.6 |
| EP8 | 64 | ep3k t16 | 8.6（prep） | — | 65.4 | 36.7 | 110.7 | 111.9 |
| EP8 | 512 | ep3k s32 | 10.4（prep） | — | 86.0 | 51.4 | 147.8 | 150.5 |
| EP8 | 2048 | ep3k l128 | 15.6（prep） | — | 153.9 | 101.9 | 271.3 | 275.4 |
| EP8 | 8192 | ep3k l128 | 33.5（prep） | — | 376.1 | 251.5 | 661.1 | 709.0 |

profile 中的 kernel 名：

- 排序：`opus_moe_sorting_entry`（aiter）；
- x 量化：`kernel_0`（`row_quant`）；
- TP 族：`a8w4_gemm1_*`、`a8w4_gemm2_*`、`a8w4_fmoe_bm64_*`、`a8w4_fmoe_bm64q_*`；
- EP 族：`ep_prep_*`、`ep_gemm1_*`、`ep_gemm2_*`、`ep_small_*`。

### 7.6 观察

- **≤16 tokens：被固定开销主导。**
  - TP8 1 token 的 W 读取下限只有 3.3 us，实测 43.6 us。其中排序 8.6 us、量化 4.2 us、gemm2 5.2 us，gemm1 24.4 us。gemm1 慢的原因是 8 个活跃专家 × 2 个 n-tile，只有 16 个 WG 在读 W，每个 WG 要串行读 512 KB，受单个 WG 的延迟和带宽限制。
  - EP8 的 prep 合并了排序和量化（约 7 us），t16 用窄 n-tile 让更多 WG 读 W，所以 1 token 只要 34.8 us。
- **64–512 tokens：接近访存上限。** 实测是 mem 下限的 1.26–1.51 倍（TP8）、1.23–1.37 倍（EP8）。
- **≥2048 tokens：计算和 atomic 写回占主导。**
  - 实测是（mem + fp8）下限之和的 1.3–1.65 倍；
  - TP8 tp1k 在 32768 tokens 时为 545 TFLOPS，EP8 l128 为 638 TFLOPS，hipBLASLt dense fp8 约 900–1080；
  - 大 batch 时 gemm2 的 bf16x2 atomic（上限约 1.5 TB/s）和 W 解码的 VALU 是主要瓶颈。
- **EP shape 上 EP 族全面领先**：TP 族在每个点都慢 10–46%，ep1k 在 ≤512 tokens 时也慢 14–35%。
- **TP8 调优前后**：与未调优的默认配置（`tp2k bm64 g1bcm1 g2base`）相比，调优后在 1–20000 tokens 快 1.00–1.20 倍（TP4 为 1.05–1.13 倍）。EP8 原本的 AUTO 阈值就是在这个 shape 上 A/B 定出来的，调优后只在 2048 附近快约 9%（l128 对 m64）。
- **抖动**：
  - 大 batch 的绝对值有 ±5–8% 的抖动，来自路由抽样和功耗状态；
  - 小 batch 的 EP 结果取决于本卡抽到几个专家，例如在 tune verify 中，5 tokens 测得 44 us，6 tokens 的表值是 38 us。

### 7.7 `a8w4_moe_dyn`（EP8 shape，接收缓冲区）

缓冲区容量 8192 行，前 tokens 行有效，每行至少有一路落在本卡（rank 3）。其余条件同 7.1，单位 us。每行都有本地路由，而 7.3 是均匀路由（部分 token 没有本地专家），所以同样的 tokens 下比 7.3 的表值慢。

| tokens | 配置 | `a8w4_moe`（bf16，查表） | dyn，bf16 | dyn，fp8 + 每（token, 128）scale | dyn，fp8，`est_tokens` = 8192 |
|---:|---|---:|---:|---:|---:|
| 1 | ep3k t16 | 34.4 | 34.8 | 35.7 | 49.5 |
| 16 | ep3k t16 | 75.9 | 75.4 | 76.1 | 90.8 |
| 128 | ep3k t16 | 136.5 | 135.6 | 136.8 | 165.1 |
| 512 | ep3k s32 | 156.6 | 161.9 | 166.9 | 177.4 |
| 2048 | ep3k l128 | 324.4 | 328.4 | 350.7 | 351.8 |
| 8192 | ep3k l128 | 993.6 | 987.1 | 1047.5 | 1060.2 |

- **bf16 输入**：dyn 与 `a8w4_moe` 相差在 ±3% 以内。
- **fp8 分组 scale（XG）**：≤128 tokens 没有差别；512 tokens 起慢 3–8%，来自 gemm1 每组开头的比值缩放。fp8 dispatch 的好处在通信：all-to-all 的字节数减半（代价是发送端多一次量化），这部分不在表里。
- **估计值等于容量**：配置变成 l128，grid 按容量开，1–128 tokens 多 14–28 us。capture 时估计值应尽量接近实际行数，sglang 用的是 DP 的 token 数元数据。

### 7.8 SiTUv2 与 SiLU 的耗时

条件同 7.1；两种激活交替测 3 轮，取中位数，单位 us。这些数是在表里还没有 situv2 行时测的：两种激活都按 silu 行选配置，所以每列比较的是同一个配置。SiTUv2 的 β / clamp 是运行时参数，取值不影响耗时。按 situv2 调优之后的选择差别见 6.3 节。

**EP8（rank 3），`a8w4_moe`**

| tokens | 1 | 4 | 16 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 配置 | t16 | t16 | t16 | t16 | t16 | t16 | s32 | m64 | l128 | l128 | l128 | l128 | l128 |
| silu | 35.4 | 35.9 | 75.4 | 110.8 | 136.0 | 138.5 | 153.0 | 197.6 | 304.9 | 477.3 | 721.5 | 1385.9 | 2608.1 |
| situv2 | 36.0 | 36.6 | 76.2 | 111.9 | 137.3 | 139.5 | 154.9 | 202.2 | 313.7 | 492.1 | 740.9 | 1423.4 | 2664.0 |
| situv2 / silu | 1.016 | 1.018 | 1.010 | 1.010 | 1.010 | 1.008 | 1.013 | 1.024 | 1.029 | 1.031 | 1.027 | 1.027 | 1.021 |

**EP8，`a8w4_moe_dyn`**（容量 8192，fp8 输入加每（token, 128）scale，`est_tokens` = 有效行数，同 7.7）

| tokens | 1 | 16 | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| 配置 | t16 | t16 | t16 | s32 | l128 | l128 |
| silu | 35.5 | 77.3 | 139.0 | 169.3 | 351.8 | 1039.5 |
| situv2 | 36.4 | 77.9 | 139.8 | 172.0 | 362.8 | 1084.9 |
| situv2 / silu | 1.025 | 1.008 | 1.006 | 1.016 | 1.031 | 1.044 |

**TP8，`a8w4_moe`**

| tokens | 1 | 4 | 16 | 64 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 配置 | 2k bm32 | 2k bm32 nt4 | 2k bm32 | 2k bm64 nt4 | 2k bm32 nt4 | 2k bm32 nt4 | 2k bm64 nt4 | 1k ks1 | 1kq | 1k ks1 | 1k ks1 |
| silu | 43.5 | 45.7 | 70.7 | 117.0 | 148.9 | 165.9 | 213.7 | 338.3 | 508.2 | 837.8 | 2992.0 |
| situv2 | 44.8 | 46.9 | 72.0 | 119.8 | 151.1 | 168.5 | 219.0 | 345.6 | 517.4 | 877.5 | 3038.6 |
| situv2 / silu | 1.031 | 1.028 | 1.019 | 1.024 | 1.015 | 1.016 | 1.025 | 1.022 | 1.018 | 1.047 | 1.016 |

- **EP 族**：≤512 tokens 慢 1–2%；≥1024 tokens（m64 / l128，TR epilogue，每个 tile 的行更多）慢 2–3%。
- **`a8w4_moe_dyn`**：与上面相近，8192 tokens 时慢 4.4%。
- **TP 族**：慢 1.5–4.7%，与实验版在 MI325X 上的结果（0–5%）一致。
- **整层（MORI EP8 / TP8）**：dispatch / combine / all-reduce 与激活无关，SiTUv2 只影响 MoE 一段。命令见 8.1、8.2 节（`--act silu situv2`）。

## 8. 正确性与测试

- **参考实现**：`ref.ref_moe` 是 fp32 参考，按 kernel 的方式模拟 fp8 量化：x 每行量化一次，中间结果每（行, 128 列）量化一次。剩下的差别只来自累加顺序和 bf16 输出。阈值是 cos 差 < 5e-3，实测 1e-5 到 3.5e-4。
- **`tests/test_a8w4_moe.py`**：本机结果为 84 passed、1 skipped（ep1k 在 600 tokens 时不支持）。覆盖范围：
  - 每类 kernel 对参考实现，用 TP8 和 EP8 shape；EP 用 rank 3、全局专家 id，tokens 取 1 / 77 / 600；
  - `a8w4_moe` 查表调用，tokens 包括不在调优点上的值；
  - 调优表的 act key：`lookup_config` 和 `dyn_config` 分别查 silu、situv2 的行；没有 situv2 行的 shape，以及没有 act 列的旧表，退回 silu 行；
  - CUDA graph：capture 期间只选 graph-safe 配置，多次 replay 的结果与 eager 一致；
  - situv2：
    - 两个 shape 上的全部配置（tp2k / tp1k / tp1kq，ep3k 的 5 个配置，ep1k）对参考实现；
    - 查表用 situv2 的行（EP shape 选中 EP 族），`a8w4_moe` 端到端正确；β = 0 时抛 `ValueError`；
    - `a8w4_moe_dyn`：s16（非 TR）和 m64（TR）两种 epilogue，bf16 / fp8 输入；
  - `w2_exp` 确实是 `w2_scale` 尾部的 view；
  - `a8w4_moe_dyn` 对参考实现：bf16 输入和 fp8 分组 scale 输入，容量 512，有效行数 0 / 1 / 37 / 300，无效行填 NaN 和随机 id；
  - `a8w4_moe_dyn` 的 CUDA graph：估计值 128，一次 capture 后依次用有效行数 64 / 1 / 400 / 0 / 512 replay，结果与 eager 一致；
  - `dyn_config`：只返回支持的 ep3k 配置；TP8 shape（256 个本地专家）抛 `ValueError`。
- **`tests/test_a8w4_mori_ep8.py`**（8 张 gfx942，需要 mori）：按 sglang `--moe-a2a-backend mori`（normal 模式，IntraNode kernel）的步骤跑一整层 EP8 MoE：aiter 每（token, 128）fp8 量化 → MORI dispatch → `a8w4_moe_dyn` → MORI combine。
  - 参考值：每卡用 fp32 算本卡专家在所有卡 token 上的结果（x 用 dispatch 出去的 fp8 反量化值），再跨卡求和；
  - 每卡 token 数：每卡 64、不均匀（含空闲卡和只有 1 个 token 的卡）、每卡 4096（MORI 容量上限）；
  - MoE 的两种调用：eager（只算收到的行）和 CUDA graph（整个接收缓冲区，`est_tokens` = 总 token 数）；
  - 检查每卡收到的行数与路由一致、cos 差 < 5e-3。用法见 8.1。
- **`tests/test_a8w4_mori_tp8.py`**（8 张 gfx942，不需要 mori）：按 sglang `--tp 8` 的步骤跑一整层 TP8 MoE：每卡持有整个 batch 和全部专家的 1/8 inter_dim，`a8w4_moe` 算本卡的部分和 → all-reduce。
  - 参考值：每卡用 fp32 算本卡权重分片的结果，再跨卡求和；
  - tokens 取 1 / 77 / 600 / 4096，覆盖 tp2k base / nt4、eager 的 tp1kq 和 capture 下的 tp1k；
  - 两种前向（eager、CUDA graph）× 两种 all-reduce（aiter custom all-reduce、torch / RCCL）；
  - cos 差 < 5e-3（实测约 1e-4）。用法见 8.2。
- **sglang**：`test/registered/unit/layers/quantization/test_mxfp4_aiter_a16w4_moe.py` 覆盖了以下场景：
  - TP；
  - EP 全局 id：TP 族加 expert mask，以及 EP 族的 5 个配置；
  - MORI 接收缓冲区：fp8 / bf16、Normal / LL、分块；
  - MORI 路径的 CUDA graph：capture 时用 DP 元数据估计行数，replay 时的行数高于估计值；
  - situv2：TP、EP 全局 id、MORI 三条路径对参考实现，以及参数检查。
- **与实验版 kernel 一致**（`doc-mimo/kernels/` 下的原文件）：
  - `prepare_weights` 与原 TP、EP 两套权重准备逐字节相同；
  - 行量化、TP gemm1 的 q8s 输出逐位相同；
  - topk=1 时每个输出元素只被 atomic 加一次，tp2k / tp1k / tp1kq / ep3k 全部配置在 1 / 300 / 3000 tokens 下与原 kernel 逐位相同；
  - topk=8 时，新旧 kernel 之间的 cos 差与原 kernel 自身两次运行之间的差在同一量级（1e-7 – 1e-5）。
- **与 sglang PR #45 的 `ep8_moe_dyn` 一致**：topk=1 时 t16 / s32 / l128、bf16 / fp8 输入逐位相同；topk=8 时 cos 差约 3e-7，与 PR 版自身两次运行之间的差相同。

### 8.1 整层测试与性能（MORI EP8）

在包含 `a8w4/` 的目录下运行，需要 8 张 gfx942 和 mori：

```bash
# 正确性（约 40 s）
pytest -q a8w4/tests/test_a8w4_mori_ep8.py

# 性能：MiMo-V2.6-Flash EP8，总 token 64..4096（平均分到 8 卡），分解 quant / dispatch / moe / combine
python a8w4/tests/test_a8w4_mori_ep8.py

# 只测几个 token 数、多测几轮；MoE 按 sglang CUDA graph 下的方式调用
python a8w4/tests/test_a8w4_mori_ep8.py -t 256 1024 --iters 50 --graph

# SiLU 与 SiTUv2 对比，指定 shape
python a8w4/tests/test_a8w4_mori_ep8.py --act silu situv2 \
    --model-dim 4096 --inter-dim 2048 --experts 256 --topk 8 -t 512 4096 --iters 20
```

| 选项 | 含义 | 默认 |
|---|---|---|
| `-t` | 总 token 数（8 卡合计，平均分到每卡），可给多个；每卡最多 4096（MORI 容量） | 64 128 256 512 1024 2048 4096 |
| `--act` | `silu` / `situv2`，两个都给时并排对比 | `silu` |
| `--model-dim` / `--inter-dim` | hidden / moe_intermediate_size | 4096 / 2048 |
| `--experts` / `--topk` | 全局专家数（8 的倍数）/ top-k | 256 / 8 |
| `--graph` | MoE 按 CUDA graph 下的方式调用：整个接收缓冲区，`est_tokens` = 总 token 数。不加时按 eager 方式：只算收到的行，`est_tokens` = 收到的行数 | eager |
| `--iters` / `--warmup` | 每个点的计时轮数 / 预热次数 | 20 / 3 |

- **输出**：
  - 每个 token 数先对参考实现检查一次，再计时；
  - 每种激活一张表：每卡收到的行数、选中的 a8w4 配置、quant / dispatch / moe（各卡平均 / 最慢）/ combine 各自的耗时、四项之和、e2e；
  - 两种激活都给时，最后再打一张 moe 和 e2e 的 situv2 / silu 对比表。
- **计时方法**：
  - 每轮先清空 L2 和 MALL，权重是冷的；
  - host 屏障后 GPU 先空转，等 host 把整层的 launch 都排好，所以不计 host 开销，相当于 graph 回放；eager 方式下读接收计数的那次同步也不计入；
  - 再用 MORI 的 device barrier 让 8 卡同时开始；
  - 每轮跑两次：一次在每个阶段前加 barrier，单独计时各阶段（取最慢的卡）；一次四个阶段连着跑，最慢的卡的总时间就是 e2e。
- **shape 限制**：只走 EP kernel，限制见 9.1 节 ep3k 一行（每卡本地专家 ≤ 32、64 % topk == 0、H 是 512 的倍数等）。不支持的 shape 会报 `ValueError`。
- **SiTUv2 参数**：用 `ref.SITU`（β 4、β_l 25、clamp 16），取值不影响耗时。
- **其他**：路由是均匀随机的（softmax 后取 top-k 再归一化）。GPU 被别的进程占用时结果不可信，严重时会卡住：MORI 和 ep_prep 都要求相关 WG 同时驻留。

### 8.2 整层测试与性能（TP8）

EP8 脚本的 TP 对照，对应 sglang `--tp 8`（不开 EP，没有 all-to-all）。每卡持有整个 batch（TP attention 之后各卡的 hidden states 和路由相同），以及全部专家的 1/8 inter_dim。每卡按 sglang 的顺序算一层：

1. **moe**：`a8w4_moe(x, W_rank, topk_weights, topk_ids)` 得到本卡的部分和。查表选 TP 族 kernel，x 在 kernel 内按行量化。
2. **allreduce**：跨卡求和。默认用 aiter 的 custom all-reduce（sglang 在 ROCm 上的默认，out of place，超过大小上限时回退 RCCL）；`--allreduce torch` 用 `torch.distributed.all_reduce`（RCCL）。

在包含 `a8w4/` 的目录下运行，需要 8 张 gfx942，不需要 mori：

```bash
# 正确性（约 1 min；第一次运行还要 JIT 编译 aiter 的 all-reduce 模块）
pytest -q a8w4/tests/test_a8w4_mori_tp8.py

# 性能：MiMo-V2.6-Flash TP8，batch 64..4096，分解 moe / allreduce
python a8w4/tests/test_a8w4_mori_tp8.py

# CUDA graph 回放
python a8w4/tests/test_a8w4_mori_tp8.py --graph -t 64 4096 --iters 50

# RCCL all-reduce，SiLU 与 SiTUv2 对比，指定 shape
python a8w4/tests/test_a8w4_mori_tp8.py --allreduce aiter --act silu situv2 \
    --model-dim 4096 --inter-dim 2048 --experts 256 --topk 8 -t 512 4096 --iters 20
```

| 选项 | 含义 | 默认 |
|---|---|---|
| `-t` | batch 的 token 数，可给多个。每卡都持有全部 token，所以和 EP8 脚本的总 token 数直接对应 | 64 128 256 512 1024 2048 4096 |
| `--allreduce` | `aiter`：aiter custom all-reduce，超过大小上限时走 RCCL；`torch`：`torch.distributed.all_reduce`（RCCL） | `aiter` |
| `--graph` | 回放 CUDA graph，按 sglang capture decode 的方式：MoE 只选 graph-safe 配置（4096 tokens 时用 tp1k 而不是 tp1kq），aiter all-reduce 直接读注册过的输入。不加时 eager 调用 | eager |
| `--act` | `silu` / `situv2`，两个都给时并排对比 | `silu` |
| `--model-dim` / `--inter-dim` | hidden / 模型的 moe_intermediate_size（每卡 1/8） | 4096 / 2048 |
| `--experts` / `--topk` | 专家数 / top-k | 256 / 8 |
| `--iters` / `--warmup` | 每个点的计时轮数 / 预热次数 | 20 / 3 |

- **输出**：
  - 每个 token 数先对参考实现检查一次，再计时；
  - 每种激活一张表：选中的 a8w4 配置、moe（各卡平均 / 最慢）、allreduce、实际走的 all-reduce 路径（`aiter custom` / `rccl`）、两项之和、e2e；
  - 两种激活都给时，最后再打一张 moe 和 e2e 的 situv2 / silu 对比表。
- **计时方法**：与 8.1 相同（冷 L2 / MALL，GPU 空转等 host 排好 launch，device barrier 后各卡同时开始，每轮分阶段计时和连着跑各一次）。device barrier 改用 1 个元素的 RCCL all-reduce。
- **all-reduce 路径**：
  - aiter custom all-reduce 处理 ≤ 64 MiB 的输入（H 4096 时 T ≤ 8192），更大的回退 RCCL。窗口可用 `AITER_CUSTOM_AR_MAX_SIZE` / `AITER_CUSTOM_AR_MIN_SIZE` 调整；
  - eager 时 aiter 先把输入拷进自己注册过的缓冲区再归约；
  - graph 下 capture 结束时把 graph 里的输入地址注册给各卡，省掉这次拷贝。
- **shape 限制**：只走 TP 族 kernel。每卡的 inter_dim（`--inter-dim` / 8）须是 256 的倍数，H 须是 512 的倍数（9.1 节 tp2k / tp1k 各行）。调优表里只有 MiMo-V2.6-Flash TP8 / TP4 的 shape，其他 shape 用未调优的 `default_config`（tp2k bm64 base）。
- **其他**：
  - x 和路由在 rank 0 生成后广播，每卡的权重分片随机生成（seed = rank）；
  - SiTUv2 参数同 8.1；
  - 文件名沿用 `mori` 前缀，与 EP8 脚本成对，但 TP 路径不用 MORI。

## 9. 继续开发的注意点

### 9.1 形状约束

| 部分 | model_dim (H) | inter_dim (I) | 其他 |
|---|---|---|---|
| `prepare_weights`（`channel_exponents`、fd2） | 256 的倍数；H ≥ 4096 时须为 512 的倍数（4352 报 `unsupported scale group geometry`） | 256 的倍数 | |
| row_quant / ep_prep 量化 | row_quant：256 的倍数；ep_prep：512 的倍数 | — | |
| tp2k gemm1（q8s） | 256 的倍数 | 128 的倍数 | |
| tp2k gemm2 base | 512 的倍数（`CHUNK_N=512`） | 256 的倍数 | |
| tp2k gemm2 nt4 | 512 的倍数 | 只能是 256 | |
| tp1k | 256 的倍数 | 256 的倍数，(I/128) % ksplit == 0，每个 CTA ≤ 512 列 | |
| tp1kq | 256 的倍数 | 只能是 256 | 不是 graph-safe |
| ep3k | 512 的倍数 | I/ARB ≥ 8 且为 4 的倍数（ARB 128：I ≥ 1024；ARB 256：I ≥ 2048） | 本地专家 ≤ 32，64 % topk == 0，M < 65536（`a8w4_moe_dyn` 的 M 是缓冲区容量） |
| ep1k | 同上 | I % 2048 == 0 | 同上，另需 topk = 8、M ≤ 512 |
| 32 位偏移 | W1 字节数 E·I·H < 4 GiB；tokens·H < 2^31 | | sorted id 中 token 占 24 位 |

`supported()` 里写的是其中会影响选择的部分。其他组合在调用前就会被 kernel 里的 assert 拦下。

### 9.2 硬件假设

- **gfx942 专用**：kernel 用的是 fp8 **FNUZ** 的 MFMA 和 gfx942 的 cache modifier（sc0 / sc1 / nt）。gfx950 的 fp8 是 OCP 格式，指令和 LDS 容量也不同，不能直接运行。
- **写死了 304 CU / 8 XCD**：
  - `ep_prep.NUM_CU`、`ep_small.NUM_CU`（ep1k 的 grid = 304 × 2）；
  - tile 到 XCD 的映射里的 8（tp_gemm1 / tp_gemm2 的 `NXCD`、tp_fmoe、tp_fmoe_q、ep_gemm 的 `bx % 8` 与 prep 调度）。
  
  其他 CU 数的 SKU 要先改这些常量。
- **persistent kernel 和 grid barrier 要求相关 WG 同时驻留**：ep1k 的全部 WG、ep_prep 的排序 WG、tp1kq 的 CTA。如果 CU 更少或被别的进程占用，会死锁而不是变慢。
- **LDS 上限 64 KB**：超过时 launch 报 `hipErrorIllegalState`，编译期不报错。

### 9.3 加一个新 kernel 或新配置

1. 在 `kernels/` 加 device 代码，在 `moe.py` 加 runner，并注册到 `RUNNERS`。
2. 更新 `config_name` / `parse_config`（名字要能解析回 dict，CSV 里存的是名字）。
3. 在 `supported()` 里写清结构限制，包括 act、capture、tokens。
4. 在 `tune.candidates()` 里加入候选。
5. 在 `tests/` 里加对参考实现的用例，然后重新调优并 merge。

按某个激活调优：`tune --act <act>`，结果写成该 act 的一组行（CSV 的 act 列），命令见 [configs/README.md](configs/README.md)。

### 9.4 修改 kernel 的流程

- **先确认结果没变**：
  - 把 topk 设为 1，每个输出元素只被加一次，新旧 kernel 应逐位一致；
  - 中间结果（q8s、ep gemm1 输出）直接逐位比较；
  - 再看对参考实现的 cos。
- **再看资源**：`FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=/tmp/fdX`，检查 `*/21_final_isa.s` 里的 `.vgpr_count`、`.vgpr_spill_count`、`.group_segment_fixed_size`。kernel 命中磁盘缓存时不会重新 dump，需要另给一个空的 `FLYDSL_RUNTIME_CACHE_DIR`。
- **最后重新调优受影响的 shape**。tile 改动会改变各配置的相对快慢。

### 9.5 kernel 文件里保留的实验选项

为了与实验版逐位一致，kernel 文件保留了原来的全部编译选项，`a8w4_moe` 只用到其中一部分：

- **gemm1**：`q8push` / `q8pushs` 等 TP 通讯融合用的 epilogue，`w_layout="tile"`，`pipeline` 开关。
- **gemm2**：`AR` / `AR_SEP` 的 all-reduce 融合、`TILE_ORDER`、`PROBE` 等。其中 AR 路径依赖实验目录里的 `gemm2_ar_fused`，本包没有复制。
- **冻结成默认值的环境变量开关**：在实验版里可以通过环境变量调，在本包里固定为默认值：
  - tp_gemm1：`PUSH_VAR`；
  - tp_gemm2：`EPI_DRAIN`、`BF16_FAST`、`WPERM`、`PERSIST`、`AR_HOOK`；
  - tp_fmoe：`ABL`、`LDS_PAD`、`DELAY`、`G1_SPREAD`、`PROBE`；
  - tp_fmoe_q：`DBG`；
  - ep_gemm：`TL`（per-tile 时间线）。
  
  要做这类实验，改源文件里的常量即可；源码 hash 变了，FlyDSL 会重新编译。

### 9.6 集成到推理框架

- **加载权重时**：调用 `prepare_weights`。fd2 权重会替换原 mxfp4 权重；原权重如果还要给别的路径用，需要另外保留。
- **warmup**：capture 前先用同样的 M eager 调用一次。配置按 token 数选择，ep_prep 也按 M·topk 是否超过 2048 分成两种编译变体，TP 族的 expert mask 等缓存也在第一次调用时分配。capture 期间第一次编译、加载 kernel 本身可以工作（在 `a8w4_moe_dyn` 上验证过），但缓存张量会分配到 graph 的私有内存池里。
- **跨卡归约**：TP 用 all-reduce，EP 用 reduce-scatter / combine。
  - 如果 EP 是先 all-gather 再计算（每卡都拿到全部 token），`x` 就是全部 token，输出要做 reduce-scatter；
  - DeepEP 低延迟模式那种按专家打包的布局 [E_local, max_tokens, H] 不符合本接口的约定，需要先转换。
  - MORI 那种按 token 打包的接收缓冲区用 `a8w4_moe_dyn`。
- **sglang**：`python/sglang/srt/layers/quantization/mxfp4_aiter_a16w4_moe.py` 中的 `Mxfp4AiterA16W4MoEMethod` 用一个 method 覆盖三种部署。
  - **TP**：`a8w4_moe(x, W, tw, ti)`。
  - **EP，`--moe-a2a-backend none`**：
    - 每卡拿到全部 token 和全局 id，调用 `a8w4_moe(..., expert_offset=ep_rank*E_local, num_experts=E_global)`；
    - 由 MoE 的 all-reduce 求和；
    - 本地专家超过 32 时走 TP 族加 expert mask。
  - **EP，`--moe-a2a-backend mori`**：`a8w4_moe_dyn` 直接读 MORI 的接收缓冲区。
    - method 向 dispatcher 报告 `weight_dtype=float8_e4m3fnuz`，MORI 就按 blockwise fp8（每 128 列一个 f32 scale）分发；`SGLANG_MORI_DISPATCH_DTYPE=bf16` 也能用，fp4 分发不支持。
    - 输出按容量分配，MORI combine 只读前 total_recv 行。
    - eager 时读一次计数（每层一次同步），只算有效行。
    - capture 时算整个容量，`est_tokens` 取自 DP 的 token 数元数据。
    - 缓冲区超过 32768 行时分块调用。
  - **加载时检查形状**（`create_weights`）：TP 族要求 H % 512 == 0、I % 256 == 0；MORI 要求 EP kernel 能跑（本地专家 ≤ 32 等）。
  - **激活**：`activation="silu"`，或 `"situv2"`（`MoeRunnerConfig` 的 `situ_beta` / `situ_linear_beta` 必须设置且 > 0；clamp 上限取 `gemm1_clamp_limit` 或 `swiglu_limit`，未设置表示不 clamp）。
  - **不支持的配置直接报错**：SiLU 带 clamp、silu / situv2 以外的激活、fused shared experts、`apply_router_weight_on_input`，以及除 none / mori 之外的 a2a backend。

## 10. 踩过的坑

**FlyDSL 与编译器**

1. **FlyDSL 的磁盘缓存 key 只包含源文件 hash 和闭包中的标量。** 在 helper 里 trace 时读到的运行时值（例如调试 buffer 的地址）不在 key 里，新进程会拿到把旧值写死的二进制，访存出错，这种值要放进 kernel 的闭包标量。gemm2 epilogue 的消融实验曾因为复用了旧二进制，得出「写回方式不影响耗时」的错误结论。
2. **运行时控制流的写法有限制。**
   - 运行时的 `if` / `while` 必须写在 `@flyc.jit` 或 `@flyc.kernel` 函数体里（会做 AST 改写），`range()` 也只有在 jit 函数里才会生成 `scf.for`；
   - 「编译期 if 后接运行时 elif」不可靠，要写成嵌套的 if（见 `_dispatch_rows`）；
   - 嵌套的 `@flyc.jit` 不会因为 Python int 参数不同而重新特化，尺寸要从列表长度推出来。
3. **运行时 `while` 里不要写 lane 级发散的 `if`**（例如 `if tx == 0` 去取 ticket）。LLVM 会让整个循环体在部分 exec mask 下执行，结果错误，或者直接 hang。改成 wave 级分支，再按 lane 选择增量。
4. **改变循环结构后，LLVM 会把按 lane 计算的地址提到循环外，长期占住 VGPR。** a16w4 曾因此出现 73–83 个 spill。用 `_opaque_tx` 打断这种提升。
5. **LDS 超过 64 KB 是在 launch 时报 `hipErrorIllegalState`**，编译期不报错。gemm2 fp8 l128 刚好用满 64 KB，再多 16 B 就失败。
6. **BM ≥ 64 时打开预取，或 `dyn_m` 编进两种行数变体后，VGPR 会超过 168，每个 SIMD 少驻留 wave，大 batch 反而变慢。** 所以 pipeline 只在小 BM 时打开。

**内存模型与同步**

7. **gfx942 上的 release fence 会生成 `buffer_wbl2 sc0 sc1`，把整个 L2 写回，每次 7–14 us。** 数据改用写穿 store（sc0|sc1），加 `s_waitcnt vmcnt(0)` 后再写 flag 就够了。
8. **各 XCD 的 L2 之间不一致。** 读其他 WG（可能在别的 XCD 上）写的数据，要用 sc0|sc1 的一致性 load（`_gload_sys`）绕过本地 L2。
9. **所有 WG 对同一个地址做 atomic 会串行化，每次约 40 ns。** 计数器要拆成多级（ep1k 用 32 个子计数器），并且每个计数器单独占一条 cache line。
10. **用 host 端的 epoch 当 flag 不是 graph-safe 的**（tp1kq）。计数器要在 kernel 里由最后一个 WG 复位（ep_prep、ep1k）。

**性能测量**

11. **先在实际条件下测出上限。** 750 W 功耗墙下 gfx 时钟只有约 1.05–1.4 GHz，纯读带宽只有 3.6–4.0 TB/s，拿规格峰值来比会误判。fd2 减少了 39% 的 VALU，时间却只少约 10%，因为瓶颈从 VALU 发射变成了功耗。
12. **负载更均衡不一定更快。** EP 的 per-XCD atomic 工作队列让负载完全均衡，但把重 tile 集中在一起跑，功耗更高、时钟更低，≤1024 tokens 反而慢 12–18%。
13. **大 batch 时 gemm2 的 bf16x2 atomic 上限约 1.5 TB/s**，只有 dwordx4 store 的 1/3，这时 gemm2 由 atomic 吞吐决定。改用 store + 求和只在 512–1024 tokens 省约 8 us，≥8192 tokens 反而慢 50–98 us。
14. **ATT 每个 dispatch 的 `realtime.json` 换算出的时钟不可靠**：同一 kernel 两次 trace 的 cycle 数能差 2 倍，只能比较 cycle 数。
15. **计时要排除几种干扰。**
    - 刚 JIT 完的第一次计时偏慢；
    - 反复调用时小 batch 的权重会从 256 MB 的 MALL 里读到，要冷 cache；
    - host 的 launch 开销要排除（GPU 先空转，再记开始 event）；
    - 单次测量有 ±3–5% 的抖动，候选之间要交错测量；
    - 小 batch 的 EP 结果取决于路由抽样，要多抽几次。
16. **代价模型要与实际 tile 时间一致。** 小 tile 受 W 读取限制，32 行和 64 行的 tile 一样慢；按「固定开销 + 行数」估算代价会切出不均匀的区间。

**数值与格式**

17. **aiter 的 `quantize_rows` 和早期 gemm1 的 q8 epilogue 把 token 数当作编译期常量**，每个新 batch size 都会重新编译一次。本包的 `row_quant` 和 gemm1 都已改为运行时参数。
18. **EP gemm1 的 TR epilogue 与 TP gemm1 的 q8s 输出不逐位相同**（误差在容差内），两族的 gemm1 / gemm2 混用时要注意。不开 TR 时两者逐位相同。
19. **TP 族的 gemm2 原本在 kernel 内从 scale 推出每通道最大值**，fd2 改写 scale 之后就推不出来了，所以要把最大值接在 scale 后面。复制或切分权重时要连这段尾部一起带上，它就是 `w2_exp`。

**环境（本次打包时遇到的）**

20. **容器自带的 FlyDSL 是 0.1.5，本包需要 0.3.2。** 用 `pip install --target <目录>` 装到独立目录，再通过 `PYTHONPATH` 生效。
21. **aiter 的 JIT `.so` 与源码版本不一致**（`module_aiter_core` 缺 `MlaVersion`）。设置 `AITER_JIT_DIR=<新目录>`，让 aiter 在那里重新编译，不改动共享环境。
22. **本包现在放在 pyhip 里（`pyhip.contrib.flydsl.a8w4`）。**
    - 非 editable 安装时，`pyproject.toml` 要列出 `a8w4`、`a8w4._vendor`、`a8w4.kernels` 三个子包，并用 package-data 带上 `configs/*.csv`。漏了子包会 import 失败；漏了 CSV 不报错，查表会静默退回默认配置。
    - aiter 的 import 依赖 pyhip，要用源码里的 pyhip（例如 `pip install -e`）。早先把 pyhip stub 掉的做法不再适用。

**EP all-to-all（MORI）**

23. **接收缓冲区只有前 total_recv 行有效，后面是旧数据**（NaN、随机的 id 都可能出现）。kernel 必须从 device 读有效行数；如果按 M 排序，旧行的路由会混进来。
24. **capture 时 host 不知道有效行数。**
    - grid 只能按估计值开，所以 GEMM 改成 WG 循环处理 tile（DYN）；
    - `out` 的清零改为只清有效行，按容量清零是白做的工作；
    - 估计值偏大会选到大 batch 的配置，小 batch 慢 14–28 us（7.7 节）。
25. **ep_prep 的 sorted id 中 token 占 24 位，排序 WG 必须同时驻留。** 所以一次调用要求 M < 65536，更大的接收缓冲区要分块。分块时：
    - 第一块直接传计数张量，kernel 会截到块大小；
    - 后面的块传 clamp(total − 起点, 0, 块大小)，全程在 device 上算。

## 11. 后续方向

- **TP 小 batch（≤16 tokens）**：
  - 把 `ep_prep` 推广到 256 个专家（需要把专家维的扫描从 32 lane 扩展到多个 wave），把排序和量化合成一次 launch。目前这两步合计 12.8 us，EP 的 prep 只要约 7 us；
  - gemm1 在极少 token 时用 split-K 或窄 n-tile（做法同 EP 的 t16），让更多 WG 并行读 W（目前 1 token 时只有 16 个 WG，gemm1 要 24 us）。
- **EP 小 batch**：把 prep 合进 gemm1，或者与上一层计算重叠；ep1k 的固定开销还要再分析。
- **中等 batch**：tile 数只有 CU 数的 1.7–3 倍，轮次量化明显。a8w4 gemm1 的量化组是 128 列，NI 不能再小；可以考虑让相邻 WG 交换行 amax，把 tile 切得更细。
- **大 batch**：
  - 重写 gemm2：用更大的 M tile，解码后的 W 经 LDS 在多个 wave 之间共用，参考 pyhip 的 Down，16K tokens 有望快约 6%；
  - 提高 gemm1 的 MFMA 利用率；
  - TP 在 8192 tokens 时排序加量化还要 75 us，换成合并的 prep 能省一半左右。
- **按需要补**：TP4 的 situv2 行；hot 路由的调优表；其他模型（例如 MiMo-V2.6-Pro TP8：H 6144、384 个专家）的调优。

## 12. 来源

- **从哪里来**：`kernels/` 和 `_vendor/` 是从实验文件机械生成的。做法是：
  - 删掉测试代码（参考实现、bench、main）；
  - import 改为指向包内；
  - 环境变量开关冻结成默认值；
  - EP 入口的 `rank` 参数改成 `e0`；
  - 实验版的 TP 调优器（`a8w4_moe_tune.py`）和 EP 的 `AUTO` 阈值，合并成这里的 `moe.py` 和 `tune.py`。
- **现在的地位**：包内文件是 `a8w4_moe` 这条路径的正式版本。实验目录里的原文件保持不变，因为实验脚本和通讯融合版本仍在使用它们。

| 包内文件 | 实验版（aiter 仓库 `doc-mimo/kernels/`） |
|---|---|
| `kernels/row_quant.py` | `a8w4_row_quant.py` |
| `kernels/tp_gemm1.py` | `test_a8w4_gemm1.py` |
| `kernels/tp_gemm2.py` | `test_a8w4_gemm2.py`（不含 `prep_w2_fast`，其功能并入 `weights.py`） |
| `kernels/tp_fmoe.py` | `test_a8w4_fmoe_bm64.py`（另含 `test_a8w4_gemm2_bm128.py` 的 `_rsrc`） |
| `kernels/tp_fmoe_q.py` | `test_a8w4_fmoe_bm64q.py` |
| `kernels/ep_prep.py` | `ep8_prep.py` |
| `kernels/ep_gemm.py` | `ep8_moe.py` |
| `kernels/ep_small.py` | `ep8_small.py` |
| `kernels/ep_prep.py` 的 MDYN、`kernels/ep_gemm.py` 的 XG / DYN 和 `ep_moe_dyn` | sglang PR #45（sammysun0711/sglang）中 `flydsl_moe_ep_w4a8_gfx942/` 的 `ep8_prep.py`、`ep8_moe.py`（`ep8_moe_dyn` 的 `rank` 参数改为 `e0`） |
| `weights.py` | `test_flydsl_moe_a16wfp4.a8w4_prepare_weights` + `ep8_moe.prepare_weights` |
| `moe.py`、`tune.py` | `a8w4_moe_tune.py`（TP）+ `ep8_moe.CFGS / AUTO`（EP） |
| `ref.py` | `ep8_common.py`、`a8w4_moe_tune.torch_ref` 的参考实现和计时部分 |
| `_vendor/*` | `aiter/ops/flydsl/kernels/{buffer_ops, dpp_utils, layout_utils, kernels_common, tensor_shim, act, mxfp4_gemm_common}.py`，以及 `moe_2stage_a16wmix/{a8w4, utils}.py` 中用到的函数 |

设计过程、实验数据和放弃过的方案，见 aiter 仓库 tq/mimo-moe 分支的 `doc-mimo/README.md`（第 1–3、7、8、10 节）和 `doc-mimo/experiments/`。
