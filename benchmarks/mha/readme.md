# MHA：使用说明与性能

FlyDSL多头注意力kernel的集成方法、测试命令和当前性能。实现细节、流水线和优化历史见
[源码readme](../../src/pyhip/ops/mha/flydsl/readme.md)。已安装实现不依赖experiments、tests或benchmarks目录。

## 目录和依赖

| 内容 | 入口 |
|---|---|
| gfx942 BF16分页注意力（D128/192，D256自动分派）与D256线性varlen API | [mha_pa_bf16_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_942.py) |
| gfx942 BF16 D256分页kernel（SHUFFLE-5D，K/V DMA） | [mha_pa_bf16_256_paged_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_256_paged_942.py) |
| gfx942 BF16 D256线性kernel（linear/page1/page4，QSA复用） | [mha_pa_bf16_256_linear_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_256_linear_942.py) |
| gfx942 FP8分页注意力 | [mha_pa_fp8_942.py](../../src/pyhip/ops/mha/flydsl/mha_pa_fp8_942.py) |
| gfx950 BF16分页注意力（SWA、sink） | [mha_pa_bf16_950.py](../../src/pyhip/ops/mha/flydsl/mha_pa_bf16_950.py) |
| gfx942/gfx950单wave SWA | [mha_pa_swa_bf16.py](../../src/pyhip/ops/mha/flydsl/mha_pa_swa_bf16.py) |
| 单元测试（80项） | [test_paged.py](../../tests/ops/mha/test_paged.py)、[test_linear.py](../../tests/ops/mha/test_linear.py) |
| 共用输入、FP32参考与检查 | [_mha.py](../../tests/ops/mha/_mha.py) |
| 集成测试：精度+性能（本目录） | [test_mha.py](test_mha.py) |

kernel只依赖ROCm PyTorch、FlyDSL和PyHIP本身。集成测试另需Triton（gather对照）和AITER（对照，可用`--skip-aiter`关闭）。
当前验证环境：MI308X（gfx942，80 CU），Python 3.10.12，torch 2.12.0+rocm7.2.4，ROCm 7.2.4，FlyDSL 0.3.4.1，
Triton 3.7.1，AITER `4a1c1b823`。从仓库根目录运行：先`python -m pip install -e .`，或在命令前加`PYTHONPATH=src`
（仓库根目录下的`pyhip/`目录会遮蔽未安装的包）。

## 1. 集成

### 1.1 分页注意力（SHUFFLE-5D KV）

四个分页模块导出同一个工厂与调用约定，按架构/类型显式选择，没有隐式回退：

```python
import torch
from pyhip.ops.mha.flydsl import mha_pa_bf16_942 as mha   # gfx942 BF16
# 其他：mha_pa_fp8_942（gfx942 FP8）、mha_pa_bf16_950（gfx950 BF16）、mha_pa_swa_bf16（SWA）

attn = mha.PagedAttention(num_qo_heads=16, num_kv_heads=1, head_dim_qk=192, head_dim_v=128,
                          page_size=64, is_causal=True)            # functools.cache，按配置复用
out = attn(q, k_cache, v_cache, cu_seqlens_q, None, kv_indptr, kv_page_indices,
           max_seqlen_q, max_seqlen_k, True,                        # causal须与工厂一致
           q_descale, k_descale, v_descale, kv_last_page_lens,
           out=None, stream=None, return_lse=False, softmax_scale=None)
```

| 参数 | 形状/类型 |
|---|---|
| `q` | `[T, H, Dq]`，gfx942为连续张量；gfx950/SWA允许行、头stride（最后一维连续） |
| `k_cache` / `v_cache` | SHUFFLE-5D：K `[P, HK, Dq/v, page, v]`，V `[P, HK, page/v, Dv, v]`，v=16字节/元素（BF16为8，FP8为16），连续 |
| `cu_seqlens_q` | int32 `[B+1]`，Q前缀和 |
| `kv_indptr` / `kv_page_indices` | int32 `[B+1]` / `[页数]`：序列b的页是`kv_page_indices[kv_indptr[b]:kv_indptr[b+1]]` |
| `kv_last_page_lens` | int32 `[B]`，最后一页有效token数（1..page）；KV长度=`(页数-1)·page+last` |
| `q_descale` / `k_descale` / `v_descale` | FP32；Q为`[1]`或per-token `[T, H, 1]`，K/V为`[1]`；BF16通常传全1 |
| `max_seqlen_q` / `max_seqlen_k` | host整数上界 |
| `out` / `lse` | 可选预分配：O为BF16 `[T, H, Dv]`；`return_lse=True`时返回`(O, LSE)`，LSE为FP32 `[T, H]`自然对数 |

普通`[P, page, HK, D]`页可这样转成SHUFFLE-5D（与测试中的`vectorize_kv`相同）：

```python
def vectorize_kv(k_pages, v_pages):
    pages, page, heads, dq = k_pages.shape
    dv = v_pages.shape[-1]
    vec = 16 // k_pages.element_size()
    k = k_pages.reshape(pages, page, heads, dq // vec, vec).permute(0, 2, 3, 1, 4).contiguous()
    v = v_pages.reshape(pages, page // vec, vec, heads, dv).permute(0, 3, 1, 4, 2).contiguous()
    return k, v
```

| 模块 | 架构 | 类型 | Dq/Dv | page | 功能 | 调度 |
|---|---|---|---|---|---|---|
| `mha_pa_bf16_942` | gfx942 | BF16 | {128,192}×{128,192}，或256/256 | 32/64/128 | full/causal、LSE；活动序列KV非空，causal时KV≥Q | `persistent=None/True`常驻网格（默认），`False`普通网格 |
| `mha_pa_fp8_942` | gfx942 | FP8 e4m3fnuz | {128,192}/128 | 64 | full/causal、LSE；`quant_query_mode`为per-token或per-tensor | 普通网格 |
| `mha_pa_bf16_950` | gfx950 | BF16 | {128,192}/128 | 64 | full/causal、`window_left`（SWA，须causal）、`has_sink`+`sink_ptr`、LSE | 默认普通网格；`persistent=True`用原子ticket |
| `mha_pa_swa_bf16` | gfx942/gfx950 | BF16 | {128,192}/128 | 64 | 仅causal SWA（默认`window_left=128`）、sink、LSE；`query_tile`/`block_n`可调 | 单wave网格 |

### 1.2 D256线性varlen（`flash_attn_varlen_func`）

```python
from pyhip.ops.mha.flydsl.mha_pa_bf16_942 import flash_attn_varlen_func

out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             causal=True, softmax_scale=None)       # BF16 [T, H, 256]
out, lse = flash_attn_varlen_func(..., return_lse=True)             # LSE: FP32 [T, H]，自然对数
# KV分页：k/v为物理行[P*page, HK, 256]，block_table为int32 [B, max_pages]，页号可任意置换
out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, block_table=table, page_size=4)
```

- 仅gfx942、BF16、连续`[tokens, heads, 256]`，H为HK的正整数倍，`page_size`为1或4，causal为bottom-right且每个序列KV≥Q，
  指针16字节对齐；`softmax_scale`默认`1/16`。`cu_seqlens_k=None`表示等长自注意力。
- dropout、window/sink、alibi、bias、soft cap、`num_waves≠8`、autograd等抛`NotImplementedError`；输出与输入/metadata重叠、
  scale非正有限值、长度超过上界等抛`ValueError`。
- metadata在第一次使用（或原地修改后）时于host完整校验一次（含一次`.cpu()`同步），之后按tensor身份与版本号缓存。

### 1.3 编译、缓存、stream与graph

- 每个specialization首次调用时编译（数秒）。头数、维度、page、causal、scale、LSE、persistent、CU数，
  以及**批大小B、`max_seqlen_q`、物理页数（K/V第0维）**等都是编译期常量，取值变化会触发新的编译。
  服务集成时应固定KV页池大小，并对B/`max_seqlen_q`分桶后预热，避免在线编译。
- FlyDSL持久缓存默认开启。升级或修改PyHIP后若结果异常，先清空缓存或设`FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`；
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`完全关闭磁盘缓存。
- 热路径不把GPU metadata复制回host，可放进CUDA graph；capture前须在eager下预热同一specialization（线性API在capture中遇到
  未预热的metadata会报错）。`stream=`可显式指定；gfx942的persistent是无共享状态的grid-stride，可并发于多个stream；
  gfx950的`persistent=True`为每个(device, stream, grid)缓存8字节调度头，需在各自capture stream上预热。
- 不检查metadata内容的合法性（页号越界、前缀和不单调等）；调用方保证其描述当前输入。

## 2. 测试与性能复现

```bash
# 单元测试：80项（gfx942全部执行；冷编译约17分钟）
MHA_TEST_GPU=0 PYTHONPATH=src python3 -m pytest tests/ops/mha -q

# 集成测试：列出case（不初始化GPU）
PYTHONPATH=src python3 benchmarks/mha/test_mha.py --list
# 全部suite，默认协议；结果写入mytest/mydata/mha_all_<时间>_<pid>/
PYTHONPATH=src python3 benchmarks/mha/test_mha.py --gpu 0
# 指定suite或case
PYTHONPATH=src python3 benchmarks/mha/test_mha.py --suite linear --gpu 0
PYTHONPATH=src python3 benchmarks/mha/test_mha.py --suite bf16-mha --case bf16-full-d128 bf16-full-d192 --gpu 0
# 只做精度检查 / 只跑小shape
PYTHONPATH=src python3 benchmarks/mha/test_mha.py --check-only --gpu 0
PYTHONPATH=src python3 -m pytest benchmarks/mha/test_mha.py -s      # 各suite的smoke case，打印性能表
```

| 参数 | 含义 |
|---|---|
| `--suite all`/`bf16-mha`/`fp8-mha`/`swa`/`linear` | 默认`all`；同一进程、同一GPU依次运行 |
| `--case ID ...` | 只运行指定的case（ID见`--list`），不改变其shape和对照 |
| `--smoke` | 只运行小shape |
| `--gpu N` | 当前进程可见的设备编号 |
| `--buffers 10 --run-count 5 --warmup 10 --repeat 1` | 10组独立输入/输出轮换，5个完整轮次，每候选50个样本 |
| `--check-only` | 逐buffer精度检查，不计时 |
| `--skip-aiter` | 只测自有kernel |
| `--verbose-runs` | 打印每个样本 |
| `--output DIR` | 新的结果目录（默认`mytest/mydata/mha_<suite>_<时间>_<pid>`），写`result.json`（环境、源码SHA256、全部样本、dispatch名）与`summary.md` |

候选：`bf16-mha`每个shape测persistent与普通网格，非smoke shape加AITER公共router（prepared线性KV，不计转换）；
`fp8-mha`只测自有kernel；`swa`长shape加AITER CK（prepared）与“完整KV Triton gather + CK”（每次调用都gather）；
`linear`测persistent与普通网格，非分页shape加AITER公共router。声明的对照缺依赖或失败时直接报错，不静默跳过。
与QSA相同，benchmark不检查GPU利用率、显存或PTL，也不修改时钟/功率；性能测试前请自行确认设备空闲。

## 3. 当前性能（MI308X，2026-10-22）

原始数据：`mytest/mydata/mha_bench_all_20261022/result.json`（逐样本、逐buffer精度、dispatch名）。GPU 2，
cudaPerf事件计时（每次计时前GPU spin 2 ms），10 buffer×5轮=50样本的中位数；时间单位µs，TFLOPS只计可见QK/PV。
`persistent`/`grid`为同一kernel的两种调度，加速比为相对AITER。全部候选精度检查通过（BF16 acc约2–3e-6，FP8约3e-4）。

此后按aiter PR #6576评审意见修复极端logit（见[源码readme第12节](../../src/pyhip/ops/mha/flydsl/readme.md)）：
BF16 D128/D192分页、FP8与SWA的输出与改动前逐位一致，交替A/B耗时变化在±1.1%以内（FP8 causal D128快2.1%），
D256分页/linear在±0.3%以内，复用D256 helper的QSA union kernel快约2%；下表未重测。

### 3.1 BF16分页（`bf16-mha`，B1）

| case | 形状（Q/KV，H/HK，Dq/Dv，page） | persistent | grid | AITER | 加速比 |
|---|---|---:|---:|---:|---:|
| bf16-full-d128 | 10240/2583，16/1，128/128，64 | 920.4（235.4T） | 899.2（241.0T） | 1315.8（164.7T） | 1.43–1.46 |
| bf16-full-d192 | 10240/2583，16/1，192/128，64 | 1115.3（242.9T） | 1122.4（241.3T） | 1605.1（168.7T） | 1.43–1.44 |
| bf16-causal-d128 | 32768/32768，16/1，128/128，64，causal | 18061（243.5T） | 17951（245.0T） | 26125（168.4T） | 1.45–1.46 |
| bf16-causal-d192 | 32768/32768，16/1，192/128，64，causal | 21956（250.4T） | 21752（252.7T） | 31142（176.5T） | 1.42–1.43 |
| bf16-mha-long-p32 | 20480/20480，8/8，128/128，32 | 6546.6（262.4T） | 6549.7（262.3T） | 9962.8（172.4T） | 1.52 |
| bf16-mha-short-p32 | 10240/2560，8/8，128/128，32 | 443.8（241.9T） | 441.7（243.1T） | 653.1（164.4T） | 1.47–1.48 |
| bf16-mha-short-p64 | 10240/2560，8/8，128/128，64 | 448.1（239.6T） | 440.8（243.6T） | 651.8（164.7T） | 1.45–1.48 |
| bf16-mha-long-p32-d192 | 20480/20480，8/8，192/128，32 | 8180.2（262.5T） | 8123.4（264.4T） | 12099（177.5T） | 1.48–1.49 |
| bf16-mha-short-p32-d192 | 10240/2560，8/8，192/128，32 | 549.2（244.4T） | 543.6（246.9T） | 795.3（168.8T） | 1.45–1.46 |
| bf16-mha-short-p64-d192 | 10240/2560，8/8，192/128，64 | 541.6（247.8T） | 543.6（246.9T） | 795.4（168.8T） | 1.46–1.47 |
| bf16-gqa-full-d256 | 10240/2583，24/2，256/256，64 | 3046.7（213.4T） | 3048.4（213.2T） | 6435.1（101.0T） | 2.11 |
| bf16-gqa-causal-d256 | 32768/32768，24/2，256/256，64，causal | 58930（223.9T） | 58725（224.7T） | 117858（112.0T） | 2.00–2.01 |
| bf16-gqa-long-p32-d256 | 20480/20480，24/2，256/256，32 | 45246（227.8T） | 45234（227.9T） | 91710（112.4T） | 2.03 |
| bf16-gqa-short-p32-d256 | 10240/2560，24/2，256/256，32 | 3028.7（212.7T） | 3030.3（212.6T） | 6137.4（105.0T） | 2.03 |
| bf16-gqa-short-p64-d256 | 10240/2560，24/2，256/256，64 | 2954.4（218.1T） | 2955.5（218.0T） | 6139.3（104.9T） | 2.08 |

小shape（Q65/KV129）：D128 18.8、D192 21.2、D256 21.0–21.1µs，主要是launch与流水线填充。

### 3.2 FP8分页（`fp8-mha`，自有kernel）

2026-10-22 FP8 VALU重排后在GPU 7同址重测（GPU 0–5当时有外部负载），“改前”为同GPU、同协议的旧kernel；
输出与改前逐位一致。原始数据：`mytest/mydata/mha_shape_164800_h7_20261022/bench_fp8_{orig,v7}/result.json`。

| case | 形状（Q/KV，H/HK，Dq，Q scale） | 改前µs（TFLOPS） | 当前µs（TFLOPS） | 提升 |
|---|---|---:|---:|---:|
| fp8-full-token-d128 | 10240/2583，16/1，128，per-token | 649.5（333.6T） | 574.4（377.2T） | +13.1% |
| fp8-full-token-d192 | 10240/2583，16/1，192，per-token | 682.5（396.8T） | 660.6（410.0T） | +3.3% |
| fp8-causal-token-d128 | 32768/32768，16/1，128，per-token，causal | 16414（268.0T） | 14357（306.4T） | +14.3% |
| fp8-causal-token-d192 | 32768/32768，16/1，192，per-token，causal | 17450（315.1T） | 15342（358.3T） | +13.7% |
| fp8-full-tensor-d128 | 10240/2560，8/1，128，per-tensor | 313.6（342.4T） | 274.9（390.6T） | +14.1% |
| fp8-full-tensor-d192 | 10240/2560，16/1，192，per-tensor | 659.9（406.8T） | 635.6（422.4T） | +3.8% |
| fp8-causal-tensor-d192 | 32768/32768，16/1，192，per-tensor，causal | 17489（314.4T） | 15339（358.4T） | +14.0% |

长序列持续负载（[164800, 7, 128]全MHA，空闲5 s、预热3次后50次背靠背取平均，GPU 7）：per-token/per-tensor均为
382.7/381.2T→405.9T。此时功耗约630 W、时钟1.78→1.71 GHz，受功耗上限限制；同条件hipBLASLt FP8 GEMM 16384³为399.2T。

### 3.3 SWA（`swa`，causal，W128，sink，Q16384，H16/1）

| case | 自有`swa_bf16` | AITER CK（prepared） | gather+CK | 加速比（对CK） |
|---|---:|---:|---:|---:|
| swa-kv32768-d128 | 241.9（71.6T） | 508.7 | 593.8 | 2.10 |
| swa-kv32768-d192 | 277.2（78.1T） | 568.7 | 652.5 | 2.05 |
| swa-kv65536-d128 | 240.9（71.9T） | 509.5 | 673.9 | 2.12 |
| swa-kv65536-d192 | 276.1（78.4T） | 568.6 | 725.6 | 2.06 |
| swa-kv131072-d128 | 241.6（71.7T） | 509.3 | 818.5 | 2.11 |
| swa-kv131072-d192 | 277.2（78.1T） | 569.0 | 883.7 | 2.05 |

自有kernel只访问窗口内的tile，时延与KV总长无关；gather+CK每次搬运完整KV，随KV线性增长。W0小shape约6.1–6.3µs。

### 3.4 D256线性（`linear`，`flash_attn_varlen_func`，无LSE）

| case | 形状 | persistent | grid | AITER | 加速比 |
|---|---|---:|---:|---:|---:|
| linear-full | B1 Q10240/KV2583，H24/HK2 | 2938.7（221.2T） | 2955.6（219.9T） | 6434.6（101.0T） | 2.18–2.19 |
| linear-full-page1 | 同上，page1，逆序物理页 | 4731.1（137.4T）* | 4785.6（135.8T）* | — | — |
| linear-full-page4 | 同上，page4，逆序物理页 | 4703.4（138.2T）* | 4719.5（137.7T）* | — | — |
| linear-causal-2048 | B1 Q=KV2048，H12/HK1，causal | 162.2（159.0T） | 192.9（133.6T） | 336.9（76.5T） | 2.08 / 1.75 |
| linear-causal-8192 | B1 Q=KV8192，H12/HK1，causal | 1848.1（223.1T） | 2012.8（204.9T） | 3932.0（104.9T） | 2.13 / 1.95 |

\* **分页case的中位数受GPU降频影响。** 两个分页case在每个候选的前15个样本约3.05 ms（page1 3056.9、page4 3048.5µs，约213T），
之后两个候选同时降到约4.72 ms并保持到case结束；下一个case开始时恢复。在GPU 3重跑（`mha_bench_linear_recheck_20261022`）
完全复现。sysfs采样显示同期SCLK由约1.75 GHz降到约1.16 GHz（功率约350 W，远低于650 W PPT），PPT/温度违规计数基本不变；
非分页kernel在连续背靠背调用约80次（约0.25 s）后同样降速（2.94→4.55 ms），约170次后自行恢复，只是在本协议的2 ms间隔下没有触发。
因此这是持续负载下的GPU时钟管理现象，不是kernel或本次重构引入的问题（机器码与重构前逐字节相同）；
未降频时page1/page4比非分页慢约4%。表中保留按协议计算的中位数，不剔除样本。

causal的persistent路径使用“最长query块优先+蛇形分配”，比普通网格快8–16%。

## 4. 指标与协议

- `acc` = Σ(ref−out)²/Σ(ref²+out²)（与`pyhip.calc_diff`相同，越小越好）；同时要求逐元素`rtol=atol=0.02`（FP8为0.1），
  参考为独立FP32实现。每个buffer都检查，另做两次逐位相同的重复调用，并用torch profiler确认发生了原生dispatch。
- 时间为每次调用的cudaPerf事件间隔中位数（含launch间隙），不选最快样本、不删慢样本；各候选使用相同的buffer序列，
  每个样本交替候选顺序。JIT、FP32参考、量化、AITER的KV预转换都在计时外；gather+CK的gather在计时内。
- TFLOPS = 2·H·可见(query,key)对数·(Dq+Dv)/时间，不计mask/padding与sink。GB/s按逻辑Q/K/V/O字节计算，
  不是实测HBM流量；gather+CK另加完整KV的读写。
- 结果仅代表所列形状和本机条件（非独占、未固定时钟）；比较优先看同shape、同轮次的时延。
