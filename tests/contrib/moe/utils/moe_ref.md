# MoE Torch 参考实现

[moe_ref.py](moe_ref.py)只依赖 Torch 和同目录的 [quantizer.py](quantizer.py)。
MoE 计算不调用 sorting、router、优化 kernel 或旧测试入口；量化和 MX 编解码通过量化模块懒加载 Aiter 的 Torch 参考函数。
这是**计时外的数值参考**，不是高性能实现，不是调优器，也不保证 CUDA graph capture。

## API

在本目录运行时：

```python
import torch
import moe_ref

ref_op = moe_ref.get(
    TP=8, model_dim=6144, inter_dim=384, experts=256, topk=8,
    quant_scheme_str="fp8_ptpc", activation="silu",
)
moe_out = ref_op(hidden_states, weight1, weight2, topk_weight, topk_ids)
```

从上一级 MoE 测试目录使用 `from utils import moe_ref`；无需依赖 [test_moe.py](../test_moe.py)。
`get()`只绑定编译期/数值参数并返回普通函数，不执行 JIT，也不初始化 GPU。

### 维度约定

`inter_dim`按[当前设计](../tune_moe.md)表示 **TP 切分前**的维度，必须能被 `TP` 整除。
令 `I = inter_dim // TP`，`H = model_dim`，`E = experts`，`M = tokens`：

| 运行期输入 | 自然布局 |
|---|---|
| `hidden_states` | `[M,H]`，FP16/BF16/FP32 |
| `weight1` | `[E,2I,H]`；无门控时 `[E,I,H]` |
| `weight2` | `[E,H,I]` |
| `topk_weight` | `[M,topk]`，FP16/BF16/FP32，计算时转 FP32 |
| `topk_ids` | `[M,topk]`，INT32/INT64 |
| `bias1` / `bias2`（可选） | `[E,W1_rows]` / `[E,H]` |
| `output`（可选） | `[M,H]`，连续、匹配 `output_dtype`，不能与输入共享存储 |

上述示例的本地 `I=48`，不是 384。若本地 `I=384`，应传 `inter_dim=8*384`。
输入权重已经是一个 TP rank 的分片；不自动切权重、补零、做 TP all-reduce 或 EP 通信。
该示例参数已在 GPU0 以 3 tokens、BF16 原始 A/W、FP8 PTPC 跑通，返回 FP32 `[3,6144]`；
对无权重压缩的 `calc_diff=0.001603048`，对 no_quant 为 `0.001925147`，实测峰值已分配显存约 576 MiB。
权重和输入必须有限；可在调用前使用 `quantizer.validate_input()`做额外值域检查。
量化策略自己的分组约束仍有效，例如 blockscale 需要相应 N/K 整除 128，MX 需要 K 整除 32。

### 路由

- 直接通过 `torch.where(topk_ids == expert)`提取 token/route，不构造排序结果、padding 或 one-hot 专家矩阵。
- `topk_weight`按给定数值使用，**不重新 softmax/归一化**。
- 重复专家 ID 是独立 route，分别计算并累加；`-1`表示不贡献结果，其他越界 ID 报错。
- 没有 token 或没有有效 route 时输出对应形状的零值。
- `expert_mask=[global_E]`可选择本地专家，必须包含恰好 `E`个 1；权重按被选全局专家 ID 的升序紧凑排列。非本地专家贡献 0，不重新归一化路由权重。
- 共享专家若与其他专家具有相同维度，可显式追加到专家权重和每个 token 的 route 中；不推断隐藏的共享专家。异构维度共享专家、远端 dispatch/combine 不在此接口内。

## 量化与权重入口

支持量化模块当前全部 **15** 个可执行策略：

`no_quant`、`bf16`、`fp8_ptpc`、`fp8_per_tensor`、`fp8_blockscale`、
`a16w8_per_channel`、`a16w8_per_tensor`、`a16w8_blockscale`、
`fp8_per_token_per_tensor`、`a16w4`、`a8w4`、`a4w4`、
`int8_ptpc`、`int8_smoothquant`、`fp8_int4_ptpc`。

没有第二套量化描述解释器；实际执行 `quant.dequant_a/w(*quant.apply_a/w(...))`。
`quant_scheme_str`允许与量化模块一致的短横线拼写。
数值边界随 Aiter 参考接口，包括 per-tensor 全零 activation 的 `scale=0、q=NaN`；不再用本地 scale 下限修补。空输入和全无效路由仍返回空/零输出。

### 原始权重

默认传入原始 FP16/BF16/FP32 权重，参考内部按专家分别 `apply_w()`后 `dequant_w()`。
每次调用都重新准备，不缓存权重/输入指针，不隐含性能预处理。

### 已量化权重

同一参考也能直接校验将要送入优化 kernel 的量化数据：

```python
import quantizer

quant = quantizer.fp8_ptpc
w1q, s1 = quant.apply_w(weight1)
w2q, s2 = quant.apply_w(weight2)
moe_out = ref_op(
    hidden_states, w1q, w2q, topk_weight, topk_ids,
    w1_scale=s1, w2_scale=s2,
)
```

提供 scale 时不再量化权重，直接解码。必须保留量化模块返回的自然 expert 维和 scale 布局；不接收 Aiter 专用 shuffle。
MXFP4 最后一维是逻辑 K 的一半；INT4 仍是未打包的 int8 容器，不是 MXFP4。
预量化输入由调用方提供同一quantizer的自然输出，反量化函数检查scale/group形状；不再运行空expert量化来推导元数据。
BF16/no-quant 的 scale 为 `None`，重复使用对应策略不会引入额外损失。

`fp8_dtype`可显式选择 E4M3FN/FNUZ；准备权重和参考必须一致。
未指定时由量化模块按 CPU/架构选择，不擅自把未知架构当成 gfx942。

### 激活量化位置

1. GEMM1 前：对完整 `[M,H]`调用 `apply_a()`并重建到 FP32。
2. GEMM1 + gate/up 激活后：对完整自然 `[M,topk,I]`调用同一策略的 `apply_a()`，然后执行 GEMM2。

特别是 `fp8_per_tensor`，每阶段各有一个**全张量** scale，不能随 expert 或 `token_chunk_size`重新计算。
GEMM1 scale 包括原始所有 tokens，即使某个 token 的 route 全无效；GEMM2 无效 route 填 0 后参与该张量的量化。

SmoothQuant 例外是按专家确定平滑系数，而量化本身仍为逐 row：

- `a1_smooth_scale`对应 H；`a2_smooth_scale`对应本地 I，两个都必须显式提供。
- 共享形状支持 `[K]`、`[1,K]`、`[1,1,K]`，逐专家支持 `[E,K]`、`[E,1,K]`。
- 根据自然 route 选择同一专家的系数，量化 `A*s`和 `W/s`；dequant 返回平滑域，不能再撤销一次平滑。
- 系数必须正且有限；传全 1 可禁用平滑。逐 row scale 不受 token 分块影响。
- `quantize_output=True`只用于此策略，在 route 加权后显式执行 N-group32 输出量化/反量化，再 TOPK 归约；默认关闭。

## 激活与门控

`gate_mode`：

- `separated`（默认）：`[gate_0,...,gate_I-1,up_0,...,up_I-1]`。
- `interleave`：`[gate_0,up_0,gate_1,up_1,...]`，bias1 同样交错。
- `none`：无门控/G1U0，W1 只有 I 行。
- `mock_gate_only`：数值等同 `separated`，只是现有 kernel 的访存实验名，**不是**无门控。
- 不支持未实现语义的 `gate_only`。不通过权重形状默默改门控模式。

| `activation` | 有门控的 FP32 公式 | 无门控 |
|---|---|---|
| `no`（别名 `identity` / `none`） | `gate * up` | 恒等 |
| `silu` | `silu(gate) * up` | `silu(x)` |
| `gelu` | `gelu(gate, approximate="none") * up` | 精确 GELU |
| `gelu_tanh`（别名 `GeluTanh`） | `gelu(gate, approximate="tanh") * up` | tanh GELU |
| `swiglu` | `gate * sigmoid(1.702*gate) * (up+1)` | 不允许 |
| `situv2`（别名 `situ`） | `beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)` | 不允许 |

`swiglu_limit=L`表示在激活前截断 `gate=min(gate,L)`、`up=clamp(up,-L,L)`，注意 gate **没有下界**。
只有 `swiglu`在参数省略时默认 L=7；其他激活默认不截断。显式 L=0 有效，负数/非有限值报错。
`beta`和 `linear_beta`默认均为 1，必须正且有限。

**不要把现有不同路径的默认参数视为相同契约：**

- 旧本地 SiTUv2 参考默认 `beta=linear_beta=1`且截断 L=7；复现它应显式传 `swiglu_limit=7`。
- Aiter 的 `apply_gate_up()`中 SiTUv2 不截断，默认 1/1；Aiter 的 `torch_moe_stage1()`参考默认 4/25，同样不截断。
- 复现后一种参考应传 `beta=4, linear_beta=25`，而不是依赖一个模糊的“SiTU 默认”。
- 本模块 `swiglu`专指仓库中的 GPT-OSS 公式，不是普通 `silu(gate)*up`的另一个名字。
- 量化权重的行顺序也决定 blockscale 的块边界；交错前后重新量化不保证相同字节/输出。

## FP32 与显式舍入

默认流程：

1. 重建 A1/W1 为 FP32，`projected = A1 @ W1.T`。
2. 默认不在此乘路由权重；若 `doweight_stage1=True`，**先**乘权重，**再**加 bias1、执行激活。
3. 可选 `intermediate_dtype`转换，发生在激活后、A2 量化前；默认 `None`。
4. 重建 A2/W2 为 FP32，`route = A2 @ W2.T + bias2`。
5. 默认 `route *= topk_weight`；若阶段1已加权，这里不再乘。
6. 可选输出量化，然后可选 `route_dtype`转换；默认不做额外 route 舍入。
7. 按自然 TOPK 槽位在 FP32 中求和；最后转为 `output_dtype`，默认 FP32。

`intermediate_dtype`、`route_dtype`可选 FP16/BF16/FP32；`output_dtype`同样可选三种。
例如 BF16 中间值＋加权 route 各自 BF16 舍入，可设两者为 BF16，但这**不是**任意 kernel 的逐位模拟器：
BF16 atomic 累加、split-K 顺序或“GEMM2 先舍入再乘权重”的路径可能仍有不同数值。
阶段1加权在非线性之前，因此不能随意挪到阶段2当作同一个问题。

算子内部关闭 autocast，并临时设 FP32 matmul precision 为 `highest`，退出（包括异常）时恢复原设置。
该设置是进程级状态；参考应在计时外串行执行，不与其他线程的性能测试并发修改它。
不同设备/GEMM 形状仍有合法 FP32 求和差异，不承诺 CPU/GPU 逐位相同。

## 内存与执行边界

- 按专家反量化 W1/W2，不同时展开全体专家的 FP32 权重或按 route 复制全部权重。
- `token_chunk_size=256`默认限制每次 GEMM 的行数，以及 `[chunk,topk,H]`的阶段2临时输出。
- 不分配完整 `[M,topk,H]`的 FP32 Down 输出，不用非确定原子 `index_add_`归约。
- 保留完整 `[M,topk,I]`激活张量以维持阶段2 per-tensor 量化定义；A1/最终输出为 `[M,H]`。这些张量及量化临时数据仍需足够内存，分块不意味着常量内存。
- 为控制内存，阶段2可能重复反量化同一专家权重；这是参考而非性能候选。可在调用前预量化 W 减少重复量化。
- `torch.where`和参数值域检查会同步 GPU，不保证 graph capture；不修改输入、不保留 autograd 图，输出 buffer 会被覆盖并原样返回。

## 精度测试与误差报告

[test_moe_ref.py](test_moe_ref.py)同时提供 pytest 和独立精度报告。以下从本目录执行：

```bash
# 独立精度表，默认全部策略 + SiLU；不是性能测试。
HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python test_moe_ref.py --device cuda

# 扩展激活表。
HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python test_moe_ref.py --device cuda --activation silu gelu gelu_tanh swiglu situv2 no

# 当前环境需在 pytest 收集前导入 triton.language，避免既有路径/插件导入冲突。
HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python -c "import triton.language; import pytest; raise SystemExit(pytest.main(['test_moe_ref.py', 'test_quantizer.py', '-q']))"
```

`calc_diff(x,y)`复用 PyHIP 的数学定义，采用 FP64 归约，返回 Python float：

$$
d(x,y)=1-\frac{2\sum xy}{\sum(x^2+y^2)}.
$$

全零/全零返回 0；非有限结果、形状不同或设备不同显式报错。
这是对称平方差指标，不是最大相对误差或错误元素百分比。

报告给出两个对照：

- **无权重压缩**：`get(..., quantize_weights=False)`，使用原始 W（SmoothQuant 使用 W/s），但保持相同 A 量化策略和激活/舍入配置。此模式拒绝已压缩权重/scale，不能把反量化后的权重当成原始基线。
- **完全不量化**：`get(..., quant_scheme_str="no_quant")`，同时移除 A/W 量化，保留同一激活公式。SmoothQuant 原始全精度参考无需平滑系数。

### 测试范围

- 每个量化策略配SiLU验证，其他激活单独用no-quant验证，再选少量量化/激活交叉路径；不做两套完整笛卡尔积。
- 独立dense oracle保留：精确W1投影的用例不读取实现中间值；少量密集数据用例分别验证投影和后续量化，避免FP32 ulp跨越量化档位的误报，不放宽容差。
- 保留TP分片、gate布局、SiTU常量/截断、route-weight位置、重复/无效/EP route、per-tensor全局scale域、显式舍入等数值边界。
- 不再维护大量构造参数错误、import拦截器或函数签名测试。

精度表由脚本按当前输入生成，不在文档维护每次运行的数字。量化损失不等于kernel实现误差，也不是模型质量结论。