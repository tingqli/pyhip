# 函数式完整 MoE driver

[moe_driver.py](moe_driver.py) 封装完整MoE管线，不是单独GEMM测试。没有driver继承层级：

- `MOEconfig`：不可变问题配置，使用本地 `inter_dim_tp`。
- 工厂返回 `(prepare, run)`；构造只做CPU检查，不初始化GPU、不导入优化kernel。
- `prepare(weight1, weight2)` 返回本kernel理解的权重字典。
- `run(hidden_states, weights, topk_ids, topk_weights, output)` 返回传入的 `output`。
- `registry` 是普通的“注册函数名→只接受config的工厂”字典；`@register` 添加候选。

旧对象API、`PreparedWeights`、`op_params`不再保留。模块是单机、串行、可信输入的实验工具，不添加所有权/并发/兼容框架。

## 接口

```python
import torch
import moe_driver

config = moe_driver.MOEconfig(
    model_dim=1024, inter_dim_tp=256, experts=4, topk=2,
    quant_scheme="bf16", activation="silu", preshuffle=True,
)
prepare, run = moe_driver.registry["jit_splitk_64_128_True"](config)
weights = prepare(weight1, weight2)

# x: BF16 [M,1024]; ids: Int32 [M,2]; scores: FP32 [M,2]
moe_driver.validate_routes(config, ids, scores)  # 可选，计时外GPU同步检查
output = torch.empty_like(x, dtype=config.output_dtype)
assert run(x, weights, ids, scores, output) is output
```

`MOEconfig`字段：`model_dim, inter_dim_tp, experts, topk, quant_scheme, activation, preshuffle, swiglu_limit=None, beta=1.0, linear_beta=1.0, output_dtype=torch.bfloat16`。

- `inter_dim_tp` 已经是TP shard宽度；不再接收TP，不做切片、通信或padding。
- W1总是自然BF16 `[E,2I,H]`，按 `[gate; up]` 排列；也可传两个连续 `[E,I,H]` 的 `(gate, up)`。W2是 `[E,H,I]`。
- 所有driver固定标准gated数学语义、**stage2路由加权＋设备原生FP8**。删除 `doweight_stage1`、`fp8_dtype`、`max_tokens` 与driver层SmoothQuant/bias/EP mask接口。
- `gate_mode` 只作为相关工厂的物理packing超参，值为 `separated` / `interleave`；不表示ungated或mock激活。源W1仍是自然布局。
- `preshuffle` 表示准备后是否weight shuffle，不是源Tensor是否已重排。已有量化/重排的源权重明确拒绝。
- 优化输出目前要求BF16。参考driver可用FP16/BF16/FP32，统一benchmark参考默认FP32。
- `output` 必传、同设备、连续且不得与输入重叠。运行时信任shape/dtype/路由，保留必要的batch与32-bit地址限制；空batch不启动kernel。

## 权重字典与生命周期

普通driver返回 `dict(w1=..., w2=..., w1s=..., w2s=...)`，不量化时scale为 `None`。
框架不解读后端布局；kernel家族的prepare自行执行scale展开、MXFP4交织、weight/scale shuffle。
公共函数只复用自然权重量化和明确的packing步骤，不建立第二套布局注册系统。

- 源权重不修改；每次prepare产生独立存储，准备另一组权重不会覆盖此前返回结果。
- callable不绑定权重、路由、workspace、output或stream。它只缓存编译launcher/必要的Aiter元数据。
- 两阶段launcher在首次run、当前device下编译；后续复用。底层编译缓存仍由PyHIP/FlyDSL管理。
- 每次run按当前stream执行；sorting/必要清零、动态A1/A2量化、GEMM/激活、任务表、reduce均在run内。
- 单阶段也可能需要sorting，例如 `jit_fused`；是否排序由kernel决定，不由阶段数决定。
- benchmark每driver只prepare一次，再按字节复制字典中的Tensor形成独立轮换buffer，保留 `is_shuffled`；不重复量化10次，也不共享权重地址。

### 注册一个候选

```python
@moe_driver.register
def my_prefill(config):
    return moe_driver.fly_prefill(
        config, sort_block_m=64, stage1_blockn=128, stage2_blockn=64,
        atomic=False, gate_mode="separated",
    )
```

该函数导入后自动进入比较候选；重复名字报错。在运行比较前导入注册模块即可，不自动扫描插件目录。
大量规则化tile组合用一个小循环生成config-only函数，避免重复维护超参字典。`ref`、官方 `aiter` baseline和 `tuned` 是工具入口，不参加候选注册；自然A4W4 Aiter变体仍作为独立候选。

## 工厂与支持范围

| 工厂 | 策略/主要限制 |
|---|---|
| `ref` | 除SmoothQuant外的参考quantizer；自然权重、标准gated、stage2路由加权 |
| `aiter` | BF16/no-quant、FP8 PTPC/per-tensor/blockscale、A16/A8/A4W4；public fused_moe＋实际dispatch检查 |
| `tuned` | 按GPU、本地完整config和实际tokens选择已测winner |
| `jit_splitk` | BF16、A16W8 row/tensor、A16W4、FP8 PTPC/per-tensor/per-token-per-tensor |
| `jit_batch1` / `jit_batch` | legacy N32 gateup；batch1仅M=1，batch使用sorting |
| `jit_fused` | 单kernel BF16；BM16/32，BN64/128，I<=384；仍需sorting |
| `jit_loopn` | FP8 row/tensor W，H%1024=0、I<=512；`atomic_write=False`显式TOPK归约 |
| `jit_mxfp4` | A4W4，BM64/128、BN128，H%1024=0、I%256=0、TOPK<=16 |
| `jit_blockscale` | 原生A8W8 8-wave、gfx950、unclamped SiLU；显式persistent/tiled down |
| `fly_splitk` / `fly_decode` | BF16、A16W8、A16W4及三种FP8；decode不排序，支持多M |
| `fly_prefill` | BF16或原生FP8；显式gate/down tile及down path；当前只支持 `atomic=False` |
| `prefill_bf16` / `prefill_fp8` | 限定量化策略的prefill工厂 |

本地split-K/decode请求FP8 A时仍显式quant→dequant→BF16，`run.activation_path="fp8_qdq_bf16"`；不是原生A8W8。prefill与blockscale使用原生FP8。
per-tensor动态量化固定复用Aiter Torch，其余受支持策略使用HIP；不修补上游零值/极小值/舍入行为。per-tensor全零产生NaN仍被严格比较报告为错误。

### 超参

- `jit_splitk(config, block_m=16, block_n=128, down_bn=None, gate_mode="separated")`。
- `jit_batch(config, down_bn=64)`；`jit_batch1(config)`固定M16/N32；`jit_fused(config, block_m=16, block_n=128)`。
- `jit_loopn(config, atomic_write=True)`固定loop-N1024/stages3；删除不能真正调整的参数。
- `fly_splitk` / `fly_decode`：`block_m, g1u1_block_n, down_bn, gate_mode`。A16W4默认interleave，可显式separated；down使用kernel指定的物理布局。
- `fly_prefill`：`sort_block_m, stage1_blockn, stage2_blockn, atomic, gate_mode`；可选 `stage1_blockm, stage1_tile_k, stage2_tile_k, down_path, down_output_padding_bytes`。
- `prefill_bf16` / `prefill_fp8`使用相同prefill超参。旧 `g1u1_*` / `down_bn` prefill参数不作别名。

| prefill `down_path` | Metadata/down tile | local I | 附加限制 |
|---|---|---|---|
| `default` | BN64，BM32/64/128/256 | 完整K microtile | BM*I*element_bytes%4096=0，A缓存<=64KiB，无row padding |
| `1x4_64x256` | BM64/BN256 | I%64=0 | 原生FP8，H%256=0，LDS检查 |
| `8x1` | BM256/BN128 | 192/256/320/384/512/640 | 原生FP8，gate BM可独立设置 |
| `8x1_compact` | BM64/BN128 | 同上 | E<=2048，每次重建M256满块/M64尾块任务表 |

专用down显式指定padding为0/32/64/128字节，不补模型维度；仅8x1接受stage2 tile K128（I192/320也可192）。

`jit_blockscale`保留 `block_m=128/256, block_n=256, down_path, down_bn, num_oc_splits, persistent_workers`：
- persistent：I256、BN64；每OC split完整N128且至少256列，每次新建并清零counter。
- tiled：BN256、H%256=0、I%128=0，不接受队列参数；覆盖I128/256/384/1536。
- persistent I128存在已知未初始化queue tail，仍拒绝；旧JIT blockscale首K组scale错误也仍拒绝。不因接口重构放宽限制。

## Aiter 与 tuned

```python
prepare, run = moe_driver.aiter(config, tuned_config=selected_csv_row)
weights = prepare(weight1, weight2)
run(x, weights, ids, scores, output)
metadata = run.metadata  # 首次实际batch核对后保留的stage信息
```

`tuned_config=None`调用标准heuristic；比较器baseline必须显式传官方winner。
A8W4自动使用interleave，其他标准组合separated；只改变packing，不改变源W1。SiTUv2调优行要求beta=4/linear_beta=25。
自然A4W4/SiLU允许 `preshuffle=False`，weight不shuffle但E8M0 scale仍shuffle；不能传shuffled调优行。
Aiter共享状态串行切换，首次实际M核对完整key/stage/A dtype，稳态不重复审计。不并发或混入绕过driver的Aiter调用。

### tuned：按实测 CSV 分派

```python
prepare, run = moe_driver.tuned(config, tuned_csv="./tuned_aiter/moe_tuned.csv", tokens=[8192, 16384])
weights = prepare(weight1, weight2)
run(x, weights, ids, scores, output)
```

构造读CSV一次，prepare按实际device选匹配项。`tokens=None`准备全部已测M，可给整数/列表减少显存。
返回不透明字典，包含每M的run函数和prepared字典；相同注册名/配置快照只prepare一次。运行仅按实际M查表，不读CSV、不重新量化/shuffle。
没有精确M或GPU/config匹配就报错，不插值、不fallback、不在线调优。

新CSV键是 `gfx,cu_num`＋全部 `MOEconfig` 字段＋实际tokens；TP/global I保留在比较报告，不再进入等价本地shard的调度键。
`driver`保存注册函数名，Aiter的 `params.tuned_config` 保存官方配置行。注册函数超参变动后必须重测。
**旧类接口的最佳driver CSV不自动迁移，换输出目录或移走旧表后重新比较。官方Aiter tuned历史仍可复用。**

## Driver 优化与对比工具

三个CLI仍接受 `TP,H,global_I,E,topk,quant,activation[,key=value...][,tokens...]`，在边界转换为local config。
省略tokens使用1至65536的17个2次幂；kwargs仅支持 `preshuffle, swiglu_limit, beta, linear_beta, output_dtype`。
未知/已删除的 `preshuffled, gate_mode, fp8_dtype, doweight_stage1, max_tokens` 明确报错。
dtype仅接受FP16/BF16/FP32名称，kwargs位于tokens之前，允许单个末尾逗号。

| 简称 | 完整config |
|---|---|
| `hy3` | `8,4096,1536,193,9,fp8_per_tensor,silu` |
| `hy3_pad` | `8,4096,2048,193,9,fp8_per_tensor,silu` |
| `qwen35_397B` | `8,4096,4096,512,10,fp8_ptpc,silu` |
| `qwen35_397B_k256` | `8,4096,2048,512,10,fp8_ptpc,silu` |
| `qwen35_35B` | `1,2048,512,256,8,fp8_ptpc,silu` |
| `qwen35_35B_k256` | `1,2048,256,256,8,fp8_ptpc,silu` |
| `xiaomi` | `8,6144,2048,384,8,fp8_ptpc,silu` |
| `h3` | `8,6144,3072,128,4,fp8_ptpc,silu` |

在上一级MoE目录运行：

```bash
python utils/moe_bench.py 1,512,256,4,2,bf16,silu,1,7 --driver 'jit_splitk_(16|32)_64_True'
python utils/moe_bench.py '1,512,256,4,2,bf16,swiglu,swiglu_limit=6,7' --driver fly_decode
python cross_compare.py hy3_pad,4096 --no-tune -v 0
```

`--driver`对registry名字及 `ref/aiter/tuned` 做full-match regex，按名字排序。每batch只生成一份数据/参考，每driver仅准备一份私有输入，由 `pyhip.run_perftest` 复制轮换buffer（包括静态权重）。Aiter/tuned的CSV取自 `--output-dir`。
直接CLI在主进程测量、不调优；精度FAIL也做诊断计时，以首个driver为性能基准，失败退出1。
比较器同样在主进程串行测量，严格精度检查；保留 `-v 0/1/>1`、日志与HTML/CSV，不再提供进程隔离、测试超时或崩溃恢复，详见[比较器](../cross_compare.md)。Aiter官方调优子进程保持不变。

计时协议为 `eager_moe_driver_v3`：准备、复制、参考、首次JIT和预热在计时外，完整run中的动态量化/sorting/GEMM/reduction在计时内。`pyhip.run_perftest` 逐调用event计时/同步，每轮返回平均微秒，再取多轮中位数；旧性能结果需重测，不能当成kernel本身加速。

## 注意事项与验证

- 不支持autograd、torch.compile、多线程、任意跨架构或CUDA Graph生命周期承诺；主进程单GPU串行使用。
- 优化路由IDs必须在 `[0,E)`，sorted路径每token专家互异；direct-route decode可重复，显式 `validate_routes(..., sorted_routes=False)`。
- 独立 [moe_ref.py](moe_ref.py) / [quantizer.py](quantizer.py) 的扩展参考功能不因这次driver API缩减而删除；新driver不暴露SmoothQuant/bias/mask等接口。
- [test_moe_driver.py](test_moe_driver.py) 已迁移原数值矩阵，覆盖各family、量化、activation、变化路由/输入、NaN输出覆盖、current stream、packing、Aiter切换及blockscale queue生命周期。未添加专用比较/调优测试文件。
- pytest建议用 `--import-mode=importlib` 并将本utils目录放入 `PYTHONPATH`，避免仓库tests/triton遮蔽真正Triton。GPU矩阵要求ROCm gfx950。