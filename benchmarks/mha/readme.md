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
| 单元测试（129项） | [test_paged.py](../../tests/ops/mha/test_paged.py)、[test_linear.py](../../tests/ops/mha/test_linear.py) |
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
| `q` | `[T, H, Dq]`，最后一维连续；gfx942 BF16 Dq/Dv 128/192、gfx950与SWA允许行、头stride（如融合QKV投影输出的切片，gfx942要求行间距≥Dq）；gfx942 FP8与D256须连续 |
| `k_cache` / `v_cache` | SHUFFLE-5D：K `[P, HK, Dq/v, page, v]`，V `[P, HK, page/v, Dv, v]`，v=16字节/元素（BF16为8，FP8为16），连续；gfx942 BF16 page32的K与D256的K/V须 < 2 GiB（其余用64-bit页基址，可超过4 GiB） |
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

- 每个specialization首次调用时编译（数秒）。gfx942 BF16 Dq/Dv 128/192把头数、维度、page、causal、scale、LSE、
  per-token Q scale、persistent、CU数和Q步长编进kernel；批大小B、`max_seqlen_q`、物理页数（K/V第0维）是运行时参数，
  同一模型配置只编译一次，服务集成无需分桶或预热多个变体。D256与其他kernel仍把批大小、`max_seqlen_q`、页数等
  编进kernel，取值变化会触发新的编译：服务集成时应固定KV页池大小，并对B/`max_seqlen_q`分桶后预热。
- FlyDSL持久缓存默认开启。升级或修改PyHIP后若结果异常，先清空缓存或设`FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d)`；
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`完全关闭磁盘缓存。
- 热路径不把GPU metadata复制回host，可放进CUDA graph；capture前须在eager下预热同一specialization（线性API在capture中遇到
  未预热的metadata会报错）。`stream=`可显式指定；gfx942的persistent是无共享状态的grid-stride，可并发于多个stream；
  gfx950的`persistent=True`为每个(device, stream, grid)缓存8字节调度头，需在各自capture stream上预热。
- 不检查metadata内容的合法性（页号越界、前缀和不单调等）；调用方保证其描述当前输入。

## 2. 测试与性能复现

```bash
# 单元测试：129项（gfx942上7项gfx950专用或不适用而跳过；冷编译约17分钟）
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

2026-10-27（相对父提交0cff467）：BF16 D128的S3配平与stage取指对齐，FP8的S7取指对齐、行和配平与per-tensor scale入SGPR
（[源码readme第3.3、6节](../../src/pyhip/ops/mha/flydsl/readme.md)），以及D256与gfx950的inline asm hazard修复（第13节）。
424个配置（默认、高精度与RTA模式）和下文交替A/B的68个大shape的输出都与父提交逐位一致（源码readme第13节），
集成测试64个结果的逐buffer精度与父提交相同。本协议加`--skip-aiter`
在GPU 6、7上按相反顺序各跑一次A/B，原始数据与逐case对比见`mytest/mydata/mha_bench_ab_20261027/`（`compare.md`为协议中位数，
`fast_cluster.md`为全频样本）。同GPU耗时变化（两块GPU的范围；受降频影响的shape按全频样本，即不超过该case最小值1.1倍的样本）：

| kernel | persistent | grid |
|---|---:|---:|
| BF16 D128 full、short p32/p64 | −4.8%到−6.2% | −3.3%到−5.0% |
| BF16 D128 causal、long-p32 | −4.3%到−5.6% | −1.9%到−3.1% |
| FP8（7个shape，只有普通网格） | — | −0.8%到−1.3% |
| D256分页 full、short p32/p64 | −0.2%到+0.5% | +0.4%到+1.0% |
| D256分页 causal、long-p32 | 见下文交替A/B | 见下文交替A/B |
| D256 linear | −0.5%到+0.3% | −0.8%到+0.3% |

机器码未变的BF16 D192与SWA可作噪声参照：稳定shape在±0.6%以内，D192 causal的全频样本在±1.7%以内。
D256分页的causal与long-p32每次只有5–13个全频样本，两块GPU的结果不一致（−1.3%到+3.9%）。smoke shape（6–21µs）的变化在±2.2%以内，
与机器码未变的SWA smoke相当。长shape仍出现3.4节所述的降频，样本在全频与降频两档之间切换，协议中位数取决于两档样本的比例：
如linear-causal-8192 persistent中位数−3.8%到−4.1%，全频样本−0.3%到+0.1%；bf16-mha-long-p32 grid中位数+1.7%到+2.0%，
全频样本−3.0%到−3.1%。

为排除降频的影响，另用两个进程（父提交、本提交）在同一GPU上交替计时68个case：每轮各跑约40 ms的一组调用
（每次调用前GPU spin，与cudaPerf相同），组间空闲0.25 s，共10轮、先后顺序逐轮交换，取逐轮配对比值的中位数；
在GPU 6、7上各跑一遍，先手相反。机器码未变的BF16 D128带LSE与D192在−0.07%到+0.02%之间，可作噪声下限。

| kernel（case数） | 两块GPU的配对中位数 |
|---|---:|
| BF16 D128（18；含混合长度、decode、page32/128、普通网格） | −3.7%到−7.8%（中位−5.5%） |
| FP8（15） | −0.7%到−2.0%（中位−1.2%） |
| D256分页（17） | −0.6%到+0.6%（中位+0.2%） |
| D256 linear（13） | −1.3%到+0.7%（中位0.0%） |

D256分页慢得最多的是混合长度causal、普通网格、page32与16384²（+0.3%到+0.6%），decode形状与非causal LSE快0.2%到0.6%；
linear非causal带LSE慢约0.7%、16384² causal慢约0.5%，混合长度causal与普通网格causal快0.5%到1.3%，
causal page4在两块GPU上一快一慢（−0.5%、+0.4%）。
脚本与逐case结果：`mytest/mydata/mha_more_cases_20261027/`（`ab_perf.py`、`perf_summary.md`）。下表未重测。

2026-10-29：BF16 D128/D192的任务改为长query块优先、persistent蛇形分配，D192的compute stage按ATT配平，Q可带行/头stride
（[源码readme第2、3.1、3.3节](../../src/pyhip/ops/mha/flydsl/readme.md)），输出逐位不变。MI300X上MiMo prefill形状
（H16/HK1、D192/V128、causal）的交替A/B快4.8%–13.5%（源码readme第11节），本suite的MI300X实测见3.5节；
3.1–3.4节（MI308X）未重测。

2026-10-30：K改用64-bit页基址，批大小、query块数与页池大小改为运行时参数（源码readme第11节），输出逐位不变。
同进程交替A/B：无LSE的D192/D128在−0.3%到+0.35%之间（0.1 ms的B8短形状+1.0%），LSE在−0.9%到+0.6%之间；3.5节未重测。
MiMo prefill的组件与E2E性能、精度见3.6节。

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

### 3.5 MI300X（BF16 D128/D192，2026-10-29）

环境：MI300X（gfx942，304 CU，功耗上限750 W，未固定时钟），Python 3.12.3，torch 2.11.0+rocm7.2，ROCm 7.2.4，FlyDSL 0.3.2，
Triton 3.7.0，AITER `2b81433bb`。FlyDSL 0.3.2没有`flydsl.expr.generic_load`，D256分页/linear无法编译，未测。

3.1节中D128/D192的10个case与2个smoke shape；输入、逐buffer精度检查、cudaPerf计时和中位数（10 buffer×5轮=50样本）同
test_mha.py，另把改前（`b0a42a0`）kernel的副本作为两个候选放进同一进程，与当前kernel共用buffer、轮次和GPU。
与test_mha.py只有一处不同：每次计时前先在同一buffer上不计时地调用该候选一次。test_mha.py逐样本正反交替候选顺序，
排在两端的候选有一半样本紧跟在另一buffer的调用之后（cache中没有本buffer的数据），非causal shape的这部分样本慢0.3%–2.4%，
中间的候选不受影响；3.1节的persistent与AITER列就在两端。GPU 0、1同时各跑一遍：表中耗时为GPU 0，加速比（对AITER）
与相对改前的耗时变化为两块GPU的范围。全部候选精度检查通过（acc 2.2–2.9e-6）。脚本与逐样本数据（未入库）：
`/root/workspace/data/att_pyhip_mha/`的`bench_mi300x.py --prime`与`bench_primed_g{0,1}.json`。

| case | persistent | grid | AITER | 加速比 | 相对改前：persistent | 相对改前：grid |
|---|---:|---:|---:|---:|---:|---:|
| bf16-full-d128 | 538.1（402.7T） | 542.0（399.8T） | 576.4（375.9T） | 1.04–1.07 | −1.1%到+0.3% | −1.7%到+1.0% |
| bf16-full-d192 | 651.7（415.6T） | 650.1（416.6T） | 647.0（418.6T） | 0.99–1.06 | −1.0%到−4.2% | 0.0%到−4.2% |
| bf16-causal-d128 | 7672.6（573.2T） | 7737.5（568.4T） | 9820.7（447.9T） | 1.24–1.28 | −12.8%到−15.4% | −12.5%到−14.6% |
| bf16-causal-d192 | 9352.8（587.8T） | 9389.1（585.5T） | 10642（516.6T） | 1.11–1.14 | −16.7%到−18.6% | −16.6%到−17.7% |
| bf16-mha-long-p32 | 4103.4（418.7T） | 4112.4（417.8T） | 4487.0（382.9T） | 1.09 | −0.6%到+0.4% | −0.1%到+1.3% |
| bf16-mha-short-p32 | 316.5（339.3T） | 317.1（338.6T） | 342.6（313.5T） | 1.08 | −0.2%到0.0% | −0.1%到0.0% |
| bf16-mha-short-p64 | 317.1（338.6T） | 316.5（339.3T） | 341.7（314.2T） | 1.08 | +0.1%到+0.3% | −0.3%到−0.4% |
| bf16-mha-long-p32-d192 | 4945.6（434.2T） | 4952.4（433.6T） | 4777.0（449.5T） | 0.96–0.99 | −3.4%到−3.8% | −3.2%到−3.6% |
| bf16-mha-short-p32-d192 | 388.5（345.5T） | 388.4（345.6T） | 378.2（354.9T） | 0.97 | −2.2%到−2.3% | −2.2% |
| bf16-mha-short-p64-d192 | 386.7（347.1T） | 387.8（346.1T） | 378.9（354.2T） | 0.98 | −1.9%到−2.1% | −2.0% |

- causal（Q=KV 32768，H16/HK1）快12.5%–18.6%：最长的query块先开始（persistent蛇形分配，普通网格qb反序），
  D192另有compute stage配平；GPU 0上568–588T。
- D192非causal的H8/HK8 shape快1.9%–3.8%（compute stage配平），bf16-full-d192在GPU 0上快0.0%–1.0%、GPU 1上快4.2%；
  D128非causal与改前相差±1.7%以内，与两块GPU之间的差异相当。
- 对AITER：D128全部shape与D192 causal快1.04–1.28倍；D192非causal的H8/HK8 shape AITER快1%–4%，bf16-full-d192在GPU 0上
  持平（0.99–1.00）、GPU 1上快1.05–1.06倍。MI300X上两者的差距远小于MI308X（3.1节1.42–1.52倍）。
- 小shape（Q65/KV129）：D128 17.0–17.1µs（改前17.1–17.2），D192 18.6µs（改前19.3–19.4，−3.5%到−4.0%）。
- 按test_mha.py原协议（不预调用）在GPU 0、1上各跑一遍：相对改前causal −11.8%到−16.4%、D192非causal −1.2%到−3.1%；
  D128非causal的grid −0.7%到+0.3%，persistent却+0.3%到+1.5%，即上述位置偏差（persistent在两端，改前kernel在中间）。

### 3.6 MiMo-V2.6-Flash全注意力prefill（MI300X，2026-10-30）

MiMo-V2.6-Flash的48层中有9层全注意力（其余为W128 SWA）。按sglang的attention DP2×TP4部署，每个rank为H16/HK1、
D192/V128、page64、bottom-right causal，Q是融合QKV投影输出的strided切片（行间距3392个元素）。代码：PyHIP `f55f549`，
sglang `feat/mimo-ck-linear-prefill`分支`f53bcae0f8`（`SGLANG_USE_PYHIP_MIMO_PREFILL=1`）。对照为同一分支的FlyPA（默认路径）
与CK（`SGLANG_MIMO_CK_LINEAR_PREFILL=1`：每层先把SHUFFLE-5D KV gather成线性KV，再调AITER `mha_varlen_fwd`）。环境同3.5节。

**组件（单卡kernel）**。sglang的`evaluation/mimo_v26_flash/repro_flypa_ck_prefill/benchmark.py`（本地加了PyHIP候选，
并把CK拆成kernel与gather两个graph，未入库）：单序列，FlyPA用sglang的调优配置（8 wave、`manual_schedule=False`、
`cooperative_v=False`），各候选录成HIP graph后交替重放，每次计时前先空跑同一graph，CUDA event计时，取10个样本的中位数，
单位ms。Q=32768对应E2E里每个DP组的一个chunk（全局65536÷DP2），KV=已有前缀+本chunk。数据（未入库）：
`/root/workspace/data/att_pyhip_mha/component_20261030/`。

| Q | KV | FlyPA | CK kernel | CK gather | CK合计 | PyHIP（TFLOPS） | 对CK | 对FlyPA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 8192 | 1.346 | 0.784 | 0.023 | 0.791 | 0.742（463T） | 1.07 | 1.81 |
| 32768 | 32768 | 15.82 | 10.51 | 0.063 | 10.57 | 9.21（597T） | 1.15 | 1.72 |
| 32768 | 65536 | 43.09 | 33.00 | 0.110 | 33.46 | 27.26（605T） | 1.23 | 1.58 |
| 32768 | 131072 | 94.91 | 75.52 | 0.152 | 75.03 | 63.21（609T） | 1.19 | 1.50 |
| 32768 | 262144 | 195.6 | 157.1 | 0.301 | 157.4 | 134.4（614T） | 1.17 | 1.46 |
| 32768 | 524288 | 396.3 | 323.0 | 0.590 | 323.3 | 271.8（627T） | 1.19 | 1.46 |
| 32768 | 983040 | 777.5 | 672.3 | 1.085 | 674.8 | 540.2（600T） | 1.25 | 1.44 |
| 18432 | 378880 | 166.2 | 135.8 | 0.432 | 136.5 | 113.7（614T） | 1.20 | 1.46 |

- PyHIP比CK（含gather）快1.07–1.25倍、比FlyPA快1.44–1.81倍。
- CK的gather最多1.1 ms（不到CK合计的0.2%），差距来自CK的attention kernel本身。三个CK列是分别录的graph，
  合计与两项之和的差为噪声。
- 单元测试：`tests/ops/mha`的gfx942 BF16子集58项通过（含冷编译约12分钟），sglang
  `test/manual/test_mimo_pyhip_prefill.py` 6项通过（多序列不等长、末页NaN、padding、与FlyPA对齐、graph重放、多stream）。

**E2E（8×MI300X）**。dummy权重（`--load-format dummy`），attention DP2×TP4、EP1、BF16 KV，chunked prefill全局65536
（每组32768），max running 192。客户端为sglang的`evaluation/run_benchmark_mimo_pro_prefill.sh`（bench_serving random数据，
每请求输出1 token即只测prefill，4个预热请求不计），每行跑一次。CK、FlyPA与“PyHIP旧”取自同配置的前一轮
（“PyHIP旧”为`bf536ee`加旧集成：`max_seqlen_q`按2的幂分桶，首次调用时预热32个变体），本轮只重跑PyHIP。
吞吐单位为input tok/s；对CK、对FlyPA为PyHIP列的变化。

限容池（`--max-total-tokens 5242880`，SWA比例0.0658）：

| ISL/并发 | CK | FlyPA | PyHIP旧 | PyHIP | 对CK | 对FlyPA | 中位TTFT s（CK/FlyPA/PyHIP） |
|---|---:|---:|---:|---:|---:|---:|---|
| 8K/32 | 69,346 | 71,532 | 74,291 | 71,989；重跑74,293 | +3.8%；+7.1% | +0.6%；+3.9% | 3.56/3.60/3.60（重跑3.47） |
| 16K/32 | 72,746 | 68,406 | 72,721 | 72,693 | −0.1% | +6.3% | 7.12/7.61/7.16 |
| 32K/32 | 68,844 | 61,459 | 69,347 | 69,599 | +1.1% | +13.2% | 15.2/17.0/14.3 |
| 64K/32 | 61,426 | 48,270 | 62,621 | 62,867 | +2.3% | +30.2% | 33.6/41.5/32.9 |
| 128K/16 | 47,299 | 32,904 | 50,030 | 50,242 | +6.2% | +52.7% | 43.8/63.1/41.6 |
| 256K/8 | 33,757 | 21,475 | 37,088 | 37,267 | +10.4% | +73.5% | 61.5/96.8/55.8 |
| 512K/4 | 23,132 | 13,268 | 26,731 | 26,874 | +16.2% | +102.6% | 88.7/156.3/77.9 |
| 960K/4 | 14,639 | 8,351 | 17,235 | 17,507 | +19.6% | +109.6% | 267.2/470.7/223.3 |

默认池（不限容，SWA比例0.03；K池4.11 GiB/层，走K的64-bit页基址）：

| ISL/并发 | CK | FlyPA | PyHIP旧 | PyHIP | 对CK | 对FlyPA | 中位TTFT s（CK/FlyPA/PyHIP） |
|---|---:|---:|---:|---:|---:|---:|---|
| 8K/32 | 73,319 | 70,700 | 74,758 | 73,989；重跑74,258 | +0.9%；+1.3% | +4.7%；+5.0% | 3.51/3.66/3.48（重跑3.46） |
| 64K/32 | 60,087 | 47,058 | 62,671 | 62,917 | +4.7% | +33.7% | 34.2/41.4/32.7 |

- 与旧集成相比，16K–960K在−0.04%到+1.6%之间，与kernel级交替A/B一致。8K首轮为−3.1%（限容）与−1.0%，
  重跑为0.0%与−0.7%，属单次运行的波动（CK自己的8K限容比默认池低5.4%，那一次偏低）。
- 对CK：kernel快1.15–1.25倍，E2E收益随attention占比增大：16K持平，32K–64K +1%到+5%，128K起+6%到+20%。
- 启动：sglang的FlyDSL模块把`FLYDSL_RUNTIME_ENABLE_CACHE`默认设为0，服务不写FlyDSL磁盘缓存，每次启动都在
  首次prefill时编译。旧集成编译32个变体，客户端阶段比现在长237 s（默认池）与270 s（限容），计时部分相差
  不到20 s；现在每个rank只编一个kernel，落在预热请求里。
- 全部请求完成，服务日志无报错。FlyPA为本分支版本（不含另一分支的FlyPA tail mask修复`e8737be`）。
  原始数据（未入库）：`/root/workspace/data/mimo_e2e/`的`ab_20261030T021114Z`与`rerun_pyhip_20261030T144232Z`。

**精度**。`/root/workspace/data/att_pyhip_mha/accuracy_mimo.py`（未入库，结果在同目录`accuracy_20261030/`）按sglang的
调用方式（per-tensor单位scale、`softmax_scale=192^-0.5`、`cu_seqlens_k=None`、Q为融合QKV切片）调用PyHIP、FlyPA与CK，
把每个输出行与FP64参考比较（FP64下对同一组BF16输入计算）。KV页随机散布在18万页、预填NaN的池里（K 4.4 GB，
共10,120个K页位于4 GiB之后），不满的末页尾部为NaN，Q的活动行之后是NaN行，输出缓冲预填哨兵值。
“理想”是把FP64结果舍入到BF16的误差，即BF16输出的下限。共104个case、904个序列、30.8亿个输出元素。

| 组（case数） | 覆盖 |
|---|---|
| 形状（18） | B1，H16与H8（TP8），Q 1–32768，KV 1–983040（含Q1/KV983040、Q32768/KV983040），randn输入 |
| 分布（22） | Q=KV 8192与Q32768/KV262144各9种：normal、peaked（每行一个key高出约4√Dq）、offset（logit约724）、sinks（32个突出的key，其V×10）、ascending、flat、zero_q、v_outliers（1%的V×100）、self_similar；另有zero_q、sinks、offset、ascending在KV 983040 |
| 随机batch（64） | 864个序列：B 2–32，每批extend token不超过4096/16384/32768（16%的序列只有1 token），58%带前缀（KV最大261897），30个batch为非normal分布，H16与H8 |

相对RMS误差为‖O−R‖/‖R‖（中位/最大）；最差行为每个（token, head）输出向量‖o−r‖/‖r‖的最大值：

| 组 | PyHIP | FlyPA | CK | 理想 |
|---|---:|---:|---:|---:|
| 形状：相对RMS | 2.32e-3/2.40e-3 | 2.32e-3/2.40e-3 | 6.07e-3/6.16e-3 | 1.66e-3/1.66e-3 |
| 形状：最差行 | 3.9e-3 | 3.8e-3 | 9.4e-3 | 2.4e-3 |
| 分布：相对RMS | 2.07e-3/2.35e-3 | 2.07e-3/2.35e-3 | 5.76e-3/6.20e-3 | 1.64e-3/1.68e-3 |
| 分布：最差行 | 5.7e-3 | 5.5e-3 | 1.6e-2 | 3.4e-3 |
| 随机batch：相对RMS | 2.02e-3/2.31e-3 | 2.04e-3/2.31e-3 | 5.37e-3/6.11e-3 | 1.61e-3/1.68e-3 |
| 随机batch：最差行 | 6.0e-3 | 6.0e-3 | 1.2e-2 | 3.7e-3 |

- PyHIP与FlyPA的误差几乎相同（逐case相对RMS之比0.97–1.02），是理想下限的1.00–1.45倍；系统偏差（误差在参考上的投影，
  −4.5e-5到3.4e-4）与理想舍入的范围相同。peaked（输出就是一行V）与单key时与理想相同，flat、zero_q（KV 983040的均匀注意力）
  与理想几乎相同。
- CK的误差是PyHIP的1.9–4.0倍，104个case的输出全部系统偏小0.10%–0.57%（单key时也偏小0.1%，PyHIP与FlyPA此时与V逐位相同）。
- 逐元素`rtol=atol=0.02`只在V被放大的两种分布（sinks的V×10、v_outliers的V×100，共13个case、3.6亿个元素）中超限：
  PyHIP 3887个、FlyPA 3906个、CK 9667个。超限的是相消后远小于所在行量级的元素（sinks Q8192：|R|中位数为行RMS的2.8%，
  误差不超过行RMS的1.2%），所在行的相对误差不超过0.6%；其余分布与全部形状没有超限。
- PyHIP逐位一致：重复调用104/104，persistent与普通网格104/104，batch中每个序列单独调用64/64，把query拆成两个chunk
  （前一个chunk的KV相应变短，同chunked prefill）37/37。
- 三个后端都没有写padding行（104/104），NaN尾页与NaN行没有进入输出（非有限值0个）；CK的gather与原KV逐位相同。
- E2E的任务精度（如GSM8K）需要真实权重，本机只有dummy权重，未测。其他配置（Dq/Dv 128/192的四种组合、page32/128、LSE、
  GQA 1–16、per-token scale、普通网格等）由单元测试与随机sweep覆盖：sweep的240个配置与改动前逐位相同，O全部在FP32参考的
  `rtol=atol=0.02`以内、LSE误差不超过0.002。

## 4. 指标与协议

- `acc` = Σ(ref−out)²/Σ(ref²+out²)（与`pyhip.calc_diff`相同，越小越好）；同时要求逐元素`rtol=atol=0.02`（FP8为0.1），
  参考为独立FP32实现。每个buffer都检查，另做两次逐位相同的重复调用，并用torch profiler确认发生了原生dispatch。
- 时间为每次调用的cudaPerf事件间隔中位数（含launch间隙），不选最快样本、不删慢样本；各候选使用相同的buffer序列，
  每个样本交替候选顺序。JIT、FP32参考、量化、AITER的KV预转换都在计时外；gather+CK的gather在计时内。
- TFLOPS = 2·H·可见(query,key)对数·(Dq+Dv)/时间，不计mask/padding与sink。GB/s按逻辑Q/K/V/O字节计算，
  不是实测HBM流量；gather+CK另加完整KV的读写。
- 结果仅代表所列形状和本机条件（非独占、未固定时钟）；比较优先看同shape、同轮次的时延。
