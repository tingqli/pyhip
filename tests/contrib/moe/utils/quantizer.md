# MoE quantizer：参考与运行时量化

[quantizer.py](quantizer.py)提供统一的 `apply_a()` / `apply_w()`，没有额外的“字符串描述→量化解释器”。
参考/权重量化复用 Aiter Torch；运行时 per-tensor 激活量化固定使用 Torch，其余支持的量化策略使用 Aiter HIP。
本地仅适配分组、形状、后端选择和反量化；不依赖 [test_moe.py](../test_moe.py)，不修改优化kernel或sorting。

## 1. 使用方法

```python
import quantizer

quant = getattr(quantizer, "fp8_ptpc")
a_args = quant.apply_a(a)  # a: [tokens, K] 或 [tokens, topk, K]
w_args = quant.apply_w(w)  # w: [N, K] 或 [experts, N, K]
aq, a_scale = a_args
wq, w_scale = w_args

# 独立参考/精度检查使用，不放入性能计时。
# 反量化接收的是apply的返回值，而不是原始输入。
a_ref = quant.dequant_a(*a_args)
w_ref = quant.dequant_w(*w_args)
```

配置建议使用Python属性名，例如 `quantizer="fp8_ptpc"`。
`get_quantizer("fp8-ptpc")`也接受短横线拼写并仅解析公开策略；`no-quant`解析为 `no_quant`。
未知名字报错，不根据dtype猜策略。

`apply_w(x, *, fp8_dtype=None, smooth_scale=None)` 始终用于准备阶段的 Torch 参考量化。
`apply_a()` 另外接受 `backend="torch"` 和 `transpose_scale=False`，返回同样的普通二元组 `(q, scale)`。
`dequant_a(q, scale)` / `dequant_w(q, scale)`返回相同逻辑形状的FP32参考张量。

- 输入为FP16、BF16或FP32，秩至少2，K非空；保留所有前导维度，允许空tokens/空expert维。
- `no_quant`严格返回输入对象本身和 `None`，不转换dtype、stride或梯度状态。
- 其他策略不修改输入，输出连续且位于输入设备，主动断开autograd；`bf16`显式转为BF16，输入已为连续BF16时可能共享存储。
- `fp8_dtype`只控制产生FP8的一侧，可显式指定E4M3FN/FNUZ，另一侧仍按其策略执行。
- 默认CPU/CUDA使用E4M3FN；ROCm gfx942使用FNUZ，gfx950/gfx1250使用FN；其他架构要求显式指定，不把未知GPU当作gfx942。

### 运行时激活量化

- `backend="torch"`：默认参考路径，保留 CPU、FP32、非连续输入和显式跨 FP8 格式支持。
- `backend="hip"`：调用 Aiter HIP；支持 FP8 逐行/group128、MXFP4/MXFP8、INT8 和 fused SmoothQuant。要求 ROCm、连续 FP16/BF16、K 整除32和原生 FP8 格式；不静默转 dtype。group128 可设 `transpose_scale=True`，返回的是转置存储字节。
- `fp8_per_tensor` 只保留 Torch 实现；即使通用调用方传入 `backend="hip"`，也使用 Torch。不再提供 FlyDSL 后端、per-tensor HIP 分支、支持条件探测或自动择优。

```python
aq, scale = quantizer.fp8_ptpc.apply_a(a, backend="hip")
aq, scale = quantizer.fp8_per_tensor.apply_a(a)
```

本地 MoE driver 已按上述策略选后端；public Aiter driver 仍原样调用 `fused_moe`，不替换其内部量化。原优化 kernel 保留供其它调用方使用，此处不再依赖它。

## 2. 15种公开策略

下表只供阅读，运行时以方法代码为准，不需要维护另一份描述注册表。

| 对象 | 激活A | 权重W |
|---|---|---|
| `no_quant` | 原样返回，无scale | 原样返回，无scale |
| `bf16` | 显式BF16，无scale | 显式BF16，无scale |
| `fp8_ptpc` | FP8逐token/route scale | FP8逐expert、逐输出通道scale |
| `fp8_per_tensor` | FP8全activation tensor一个scale | FP8每个expert矩阵一个scale |
| `fp8_blockscale` | FP8 1×128，FP32 scale | FP8 128×128，FP32 scale |
| `a16w8_per_channel` | BF16，不量化 | FP8逐输出通道scale |
| `a16w8_per_tensor` | BF16，不量化 | FP8逐expert矩阵scale |
| `a16w8_blockscale` | BF16，不量化 | FP8 128×128 scale |
| `fp8_per_token_per_tensor` | FP8逐token/route scale | FP8逐expert矩阵scale |
| `a16w4` | BF16，不量化 | MXFP4，K方向32元素一组，E8M0 |
| `a8w4` | MXFP8，K方向32元素一组，E8M0 | MXFP4，K方向32元素一组，E8M0 |
| `a4w4` | MXFP4，K方向32元素一组，E8M0 | MXFP4，K方向32元素一组，E8M0 |
| `int8_ptpc` | INT8逐token/route scale | INT8逐输出通道scale |
| `int8_smoothquant` | 先乘smooth scale，再INT8逐row量化 | 先除smooth scale，再INT8逐输出通道量化 |
| `fp8_int4_ptpc` | FP8逐token/route scale | signed INT4逐输出通道scale，int8容器 |

同一套权重量化、不同激活量化是不同对象；不会随batch自动把BF16激活换成FP8。
这些是量化能力，不表示所有策略都有完整MoE kernel，也不表示Aiter公共API都能按该契约运行。

## 3. 布局与scale形状

令A为 `[..., K]`、W为 `[..., N, K]`，前导维度可以是tokens/TOPK或experts：

| 粒度 | scale形状 |
|---|---|
| 无量化 | `None` |
| A逐row / W逐输出通道 | `[..., 1]`（保留输入除K外的维度） |
| A per-tensor | `[1]` |
| W per-expert tensor | `[..., 1, 1]` |
| A 1×128 | `[..., K/128]` |
| W 128×128 | `[..., N/128, K/128]` |
| MX 1×32 | `[..., K/32]`（保留输入除K外的维度） |

FP8/INT8/INT4普通scale为FP32，重建是 `q * scale`，不是乘倒数。
MX scale为 `torch.float8_e8m0fnu`，解码后按32元素组广播。

- FP8/INT8输出形状与输入相同。
- MXFP4输出最后一维为K/2：偶数K元素在字节低4位，奇数在高4位。不能用普通 `.float()`代替FP4解包，应调用对应 `dequant_*()`。
- INT4权重保持原shape，dtype为int8，数值范围 `[-7,7]`。有意不做uint32打包或Aiter私有重排，adapter再调用匹配的pack/shuffle工具；不能把int8容器字节当作packed FP4。
- blockscale严格要求相应N/K整除128；MX要求K整除32；不自动补零、截尾或转置scale。
- 不做expert routing、scale sorting、权重shuffle或gate/up interleave。adapter负责kernel需要的布局转换。

## 4. Torch 参考接口与数值行为

| 用途 | 复用的 `aiter.ops.quant` 接口 |
|---|---|
| FP8 逐行、INT8 逐行 | `pertoken_quant`，传入目标 dtype |
| FP8 全 activation tensor | `per_tensor_quant` |
| FP8 每个 expert 矩阵 | 将 N/K 展平为一行后调用 `pertoken_quant`，与 Aiter MoE 权重准备相同 |
| FP8 A 1×128 / W 128×128 | 将每组/每块展平为一行后调用 `pertoken_quant`，再恢复自然布局 |
| INT4 逐行 | `pertoken_quant(quant_dtype=torch.int8, dtypeMax=7)` |
| SmoothQuant A | `pertoken_quant(..., x_scale=smooth_scale)`；W 先除以 smooth scale |
| INT8 输出 group32 | 按32元素分组后调用 `pertoken_quant` |
| MXFP4 | `per_1x32_f4_quant(shuffle=False)` |
| 原生格式 MXFP8 | `per_1x32_f8_scale_f8_quant(scale_type=fp8_e8m0, shuffle=False)` |

- 普通 FP8 不追加本地 clamp 或 FP32 最小 normal 下限。`pertoken_quant` 按 `amax / finfo(fp8).max` 求 scale，仅将计算结果为0的 scale 改为1。
- `per_tensor_quant` 不修正零 scale：**全零 activation 返回 `scale=0、q=NaN`**，本地参考和 per-tensor 运行时均保留该上游行为，完整 MoE 输出可能传播 NaN。per-expert W 使用逐行接口，因此全零矩阵仍返回 `q=0、scale=1`。
- INT8 改为 Aiter 的 **`/127` + 向零截断**；不再使用本地 `/128`、RNE 或 `1e-6` 下限。INT4 使用 `/7` + 向零截断，仍是未打包 int8 容器。
- MX 采用 Aiter 的默认 E8M0 舍入模式，当前为 **RoundUp**，不在本地冻结另一份默认值。上游 MXFP8 完整接口只接受本机原生 FP8 dtype；显式跨 FN/FNUZ 参考改用其 dtype-aware `f32_to_mx_e8m0_scale` / `e8m0_to_f32`，不自行实现 scale 公式。
- 空前导维保留：Aiter per-tensor 不能归约空输入，适配层传入占位 `scale=0`；MX 上游 reshape 不能处理空输入，直接返回相应空 q/scale。非空输入不增加修补。

默认参考对齐 **Aiter Torch 接口**。[moe_driver.py](moe_driver.py) 的 per-tensor 运行时也固定使用同一 Torch 接口，其余使用 Aiter HIP；Torch 与 HIP 本身的零值、极小值和舍入差异不在本地补齐。

## 5. SmoothQuant和Down输出量化

定义为 `A' = A * s`、`W' = W / s`，两侧必须使用对应的同一smooth系数。

```python
quant = quantizer.int8_smoothquant
aq, sa = quant.apply_a(a, smooth_scale=activation_smooth)
wq, sw = quant.apply_w(w, smooth_scale=weight_smooth)
```

- 共享系数可为 `[K]` / `[1,K]`；expert权重系数可为 `[E,1,K]`。
- 路由激活逐expert系数先由调用方按topk映射为 `[tokens,topk,K]`；模块不隐式扩展token或根据E猜routing。
- smooth系数须正且有限，能广播到输入但不得扩展输入形状。
- `dequant_*()`得到平滑后域的近似值，不自动撤销smooth变换。

另外提供 `int8_smoothquant.apply_output()` / `dequant_output()`，对已经乘好routing weight的Down输出按N方向32元素分组量化/反量化。
这是单独的可选压缩步骤，**不由A/W策略自动开启，不提前做TOPK归约**；是否采用由pipeline和数值验证决定。

## 6. 性能与检查边界

参考与权重准备依赖 Aiter Torch；per-tensor 运行时也使用 Torch，其余支持的量化策略复用 Aiter HIP。`no_quant` / `bf16` 仍只依赖 Torch。

- `apply_*()`只检查shape/dtype/device；值域可在计时外调用 `validate_input()`检查，该函数在GPU上会同步。
- 首次量化调用懒加载 Aiter，在图捕获/计时前预热。仅import模块或获取对象不加载 Aiter、旧测试或初始化GPU；量化器不再依赖 PyHIP/FlyDSL。
- 已验证当前环境预热后graph replay可随输入更新；恒等路径可能没有GPU指令，不强行制造工作。
- 静态权重量化放在E2E计时外。动态activation量化若用这些函数，就要将真实开销计入E2E，不能提前算好再称为完整流程。
- 大expert权重和FP32参考有较大临时内存，应按expert准备；A per-tensor不能任意切块后分别求scale而改变语义。
- 后续替换量化后端时，先验证同策略逐字节/数值一致性，不能只因同名就认定等价。

## 7. 验证

[test_quantizer.py](test_quantizer.py)可单独运行，不依赖主MoE测试：

```bash
HIP_VISIBLE_DEVICES=0 /opt/venv/bin/python -c "import triton.language; import pytest; raise SystemExit(pytest.main(['test_quantizer.py', '-q']))"
```

当前环境需要在pytest开始收集前导入 `triton.language`，避免仓库测试路径/插件与已安装Triton的导入冲突；不修改全局pytest配置。
测试覆盖全部策略、普通/非连续/空/零/极小输入、自然scale布局、两种E4M3格式、独立FP4打包样例、INT8/INT4截断、SmoothQuant方向、Aiter Torch/HIP 逐字节对照、per-tensor 始终使用 Torch 及graph输入更新。per-tensor 全零 NaN 是显式上游一致性检查，不作有限结果假设。
不逐策略重复测试普通参数类型错误；数值公式、packing和零/极小值边界优先。FNUZ格式测试不等于在gfx942硬件上的MoE验证。