# Linear Flash Attention 接口

- [flash_attn_varlen_8wave.py](flash_attn_varlen_8wave.py)：原 D128 adapter，行为未修改。
- [flash_attn_varlen_d256.py](flash_attn_varlen_d256.py)：新增 gfx942 BF16 D256 原生 linear adapter，独立导入，不替换原接口。

## D256 使用方式

```python
from experiments.attention.flydsl.flash_attn_api.flash_attn_varlen_d256 import (
    flash_attn_varlen_func,
)

# q: BF16 [total_q, H, 256]
# k/v: BF16 [total_k, HK, 256]
# cq/ck: CUDA/ROCm int32 [batch + 1]，普通PyTorch张量，有版本计数器
o = flash_attn_varlen_func(
    q, k, v, cq, ck, max_seqlen_q, max_seqlen_k,
    layout="linear", page_size=1,
)

# 可选：任意物理页映射。k/v仍为扁平token-major [physical_tokens, HK, 256]。
# block_table: int32 [batch, max_pages]，每项是物理页ID；这里一页4个token。
o, lse = flash_attn_varlen_func(
    q, k, v, cq, ck, max_seqlen_q, max_seqlen_k,
    page_size=4, block_table=block_table,
    causal=True, return_lse=True, out=out,
)
```

### 张量及分页合同

| 项目 | 支持范围 |
|---|---|
| GPU/数据类型 | gfx942，BF16，`DQ=DV=256` |
| Q/O | 连续 `[total_q,H,256]` |
| K/V | 连续、同shape `[physical_or_linear_tokens,HK,256]`；`H>0`、`HK>0`、`H % HK == 0` |
| KV page | `page_size=1` 或 `4`；无页表时无需padding，也不做KV转换 |
| 有页表 | `block_table[b,j]` 指向第 `j` 个逻辑页的物理页；物理token索引为 `block_table[b,t//page_size]*page_size+t%page_size` |
| 元数据 | CQ/CK同shape int32 `[B+1]`，首项0、非降序、长度不超过给定max；有页表时CK是逻辑KV累计长度 |
| 尾部 | KV物理存储仅在有页表时要求token数为page的倍数；不读取无效页表列，未使用列允许 `-1`；V尾token在DMA时归零 |
| GQA | 包含 `H=24/HK=2`；无需展开K/V heads |
| causal | bottom-right对齐；有query的序列要求 `KV>=Q`；无query段和全空Q支持 |
| LSE | `return_lse=True` 返回 `(O,LSE)`；LSE是FP32自然对数 `[total_q,H]` |
| 调度 | `persistent=None/True` 固定驻留CTA；`False` 普通grid |
| stream/graph | 当前stream或显式 `torch.cuda.Stream`；先在capture外预热相同metadata和编译特化 |

Q/K/V/O指针要求16-byte对齐，所有输入/输出字节跨度小于2GiB；`out`不允许与输入或metadata重叠。
`softmax_scale`支持有限正host scalar，默认 `1/16`；缺省CK仅支持无页表且Q/K总token数相同的self-attention。

首调用会把CQ/CK及有效页表内容复制到CPU做验证；之后用weakref、版本号、shape和pointer校验缓存，热调用无metadata D2H。
元数据需在 `torch.inference_mode()` 外创建；Q/K/V可以用于inference。
使用正常PyTorch操作修改metadata后需在capture外重新调用验证，重新捕获使用新metadata的graph。
**graph replay不会执行Python验证**，不能在未重新验证时重放任意修改过的页表/边界；不同stream上的输入准备需要调用方建立依赖。

保持原adapter的完整参数顺序，但首版不支持backward/autograd、dropout、soft-cap、bias、ALiBi、局部window/sink、padding边界、attention-prob返回或其他wave数；这些选项显式报错，不静默忽略。

## 实现与验证

原生实现见 [../mha/mha_pa_bf16_256_linear_942.py](../mha/mha_pa_bf16_256_linear_942.py)：
K/V直接DMA到LDS，V在 `ds_read_b128` 后用编译器可见 `v_perm_b32` 做BF16 2×2转置，再进入M16 MFMA。
**每次热调用只有一个attention kernel，没有KV转换/gather kernel，也没有调用外预转换要求。**

[test_flash_attn_varlen_d256.py](test_flash_attn_varlen_d256.py) **98项通过**，含causal/LSE、实际随机/逆序页表、GQA、尾部NaN、guard、stream/graph、metadata修改、拒绝条件；所有本次测试编译特化检查private及VGPR/SGPR spill为0。

2026-09-24正式Full验收：B1/Q10240/KV2583/H24/HK2/D256、noncausal/noLSE/persistent，10独立buffer×5轮、每候选50样本，原 `cudaPerf`。
对同场v98，随机page1/page4 **TFLOPS分别低8.90%/9.37%**；时延分别高 **9.77%/10.33%**。
因此吞吐差在10%以内，但预先设置的更严格 `linear_us <= 1.10*v98_us` 门槛，**page4仍未通过**。
相对当前默认v73，两者时延分别高8.73%/9.30%。不把该Full范围扩大为所有shape或grid的性能保证。
所有慢样本和失败试验保留，详见 [证据与完整性能表](../mha/results/d256_linear_20260924/README.md)。