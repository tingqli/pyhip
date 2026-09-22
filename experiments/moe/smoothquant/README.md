# fused MOE g1u0 gelu

## fused MOE 基本流程

 - hidden_states: [num_tokens, model_dims]
 - 逐点乘以 smooth_scale + per-token 量化到 INT8 [[1]](#共享smooth_scales)
 - moe_sorting(block_m) : 每个token的topk个需要infer的复本，根据专家进行分组排序，每组内token的个数padding到block_m整数倍 [[2]](#最大化block_m)
 - GEMM1 + fused_gelu: 输出数据tensor bf16：[num_tokens, topk, inter_dims] [[3]](#HIPKitten's 8wave pipeline) [[4]](#指令优化gelu)
 - 逐点乘以 per-expert smooth_scale + per-token 量化到 INT8
 - GEMM2: [num_tokens, topk, model_dims] [[6]](#中间结果动态量化) [[7]](#XCD Swizzle)
 - ReduceSum : [num_tokens, model_dims] [[5]](#避免ATOM访存带宽限制)


## 共享smooth_scales
根据 [SmoothQuant论文](https://arxiv.org/abs/2211.10438)，outlier主要出现在激活input的某些channel，权重相对则要均匀很多，作为整个MOE的原始输入，应该可以共享同一份smooth scale, 相较于每个专家独享一份smooth scale的方案，这可以把第一个量化kernel的数据访问量至少降低TOPK倍。

## 最大化block_m

GEMM问题中，一个workgroup (threadblock)处理的输出矩阵越大，越接近正方形，越能够降低访存计算比 `(M+N)*K/(MNK)`, 在LDS和寄存器资源允许的情况下，配合jit的手工寄存器分配，该值可以增加到256。（目前aiter代码中最大128）

## HIPKitten's 8wave pipeline

[HipKittens](https://arxiv.org/abs/2511.08083) 针对AMDGPU引入了8-wave 排流水的创新性pipeline：
 - 按照4wave为单位，分为两组，两组交替执行计算和加载，遮盖访存延迟
 - 可以大大简化流水线的排布，并且获得几乎不输于4wave的性能
 - 相对传统的`4wave`排流水，可以避免手工决定使用AccVGPR还是普通VGPR的问题，全部分配为普通VGPR

## 指令优化gelu

[Gelu](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) 激活函数的计算使用到了[tanh](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html)函数，HIP的C++ STL生成的汇编代码比较复杂并且可能出现thread-diverge（根据每个thread输入值的取值范围使用不同方式近似计算，以避免exp越界），可以更通过简单的方式保证exp的输入永远是负数来避免越界：
```python
    sign = np.sign(v)
    exp = np.exp(-2*sign*v)
    tanh = (sign - sign*exp)/(1 + exp)
```

## 避免ATOM访存带宽限制

目前Aiter中某些case下的moe gemm2的输出是使用 `global_atomic_pk_add_bf16` 指令直接累加到外存来避免写出巨量的中间结果和额外的ReduceSum kernel开销，但是实测表明这种 atomic 访存指令在 gfx942/gfx950 上的带宽只有普通写出访存指令的1/4，因此性能还不如`直接写出巨量数据，再使用ReduceSum读回做sum再存出最终结果`。

## 中间结果动态量化

gemm2写出巨量中间结果发生在gemm kernel的尾部，没有MFMA指令可以与其并行遮盖，因此拖累了性能，以1x32为单位的量化该中间结果到INT8再存出，可以显著降低写出消耗。

另外使用`sc1 nt`修饰符bypass cache，直接streaming到外存可以结果数据对L2-cache的污染，进一步提升gemm核心循环的性能


## XCD Swizzle

gfx950的256个CU分布在8个XCD上，每个XCD具有独立的L2-cache, 通过把访问相同专家权重的 gemm-block 计算任务分配到相同的 XCD上，当这些任务在gemm核心循环中，以一致的步调访问接近相同的权重和输入数据的某个K维度slice的时候，就可以从L2-cache受益，减少冗余的外存加载操作。
