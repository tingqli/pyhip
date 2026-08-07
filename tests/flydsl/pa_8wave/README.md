
# pa_prefill_8w32x32

基本并行原则： 每个8wave处理共享同一个`kv-head`的32行`query-tokens`, 共处理 `[256, head_dim_qk] @ [head_dim_qk, kv-len] ` 这么大的 Q@K, 8wave分为两组交织 MFMA 和 oneline-softmax.

使用 persistent kernel， 查询`torch.cuda.get_device_properties().multi_processor_count`确定启动的work-group个数，每个work-group使用 atomic_add 动态从`[变长的cu_seqlens_q列表, num_query_head]` 这么大的任务空间中分配任务直到所有任务都完成。跟静态分配（work-group-id += num-workgroups）相比对任务计算量不均等的 causal 情况帮助很大。

一些优化点：

  - 全部使用`global->lds->register`数据加载对LDS带宽压力较大，对v使用`global->register`加载不经过LDS
  - 32x32的MFMA指令有助于减少Online-softmax的row-max/row-sum的cross-lane操作计算量
  - 355引入的 v_permlane 指令使得 cross-lane 计算不经过LDS，不占用LDS带宽
  - 使用`内联汇编`/`-packed-fp32-ops`选项避免使用`v_pk_`VALU指令，因为这些指令跟MFMA有co-issue问题
  - 使用 U32 的虚假类型表达 fp8 的数据加载路径上的tensor类型，避免llvm生成冗余拆解fp8代码
  - 使用 B@A 模式的fx.gemm，这样Q@K的结果layout自然满足P@V的输入layout需要
  - 从外存加载k数据时 permute kv-length 维度，使之匹配 P@V 里面 V 的 reduction 维度的layout

