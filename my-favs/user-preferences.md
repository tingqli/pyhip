# 用户偏好（全局，跨会话持久）

## 语言
- **始终用中文回复**（用户 2026-07-21 明确要求，适用于所有会话，包括新会话）。
- 代码注释也优先用中文（与现有代码库风格一致）。
- GPU流水变量/函数按数据和搬运方向命名，避免 commit/issue/pending/retire 等模糊词；g=VMEM、r=寄存器、s=LDS，如 read_b_g2r、store_b_r2s、b_r2s_kb、c_bf16。


## 工作方式
- 用户在 GPU kernel 优化上非常 hands-on：喜欢"小步改 -> 重测 -> 更新文档"的迭代循环。
- 只有性能测试需要等待空闲 GPU（`rocm-smi --showuse` 挑 util 低的）；正确性测试无需等待空闲，可直接运行。
- 性能测试用 cudaPerf 多 buffer 轮换取中位数。
- 改完 kernel 先验证功能（rel_l2）再验证性能。
- 后续 GPU 性能 tune 优先以 PTL `Enabled / VECTOR,F8` 作为稳定参考环境；测试前显式核验并在结果中记录 PTL，测试结束后恢复机器原状态。
- 所有 GPU kernel 性能结果同时给出时延和有效 TFLOPS，并明确工作量公式；由 ATT union × roof 得到的数值必须标为“模型 TFLOPS”，不可与 wall-time 有效 TFLOPS 混用。
- 微小kernel调度改动先做两版本Down-only快速测试，优先复用经源码/ISA哈希核验的已编译产物；确认后再扩展compact/Full/全矩阵，避免反复冷JIT。

- 2026-09-08明确要求：不要随意生成新文档；优化统一记录到一份新的追加式文档，后续简单追加，不修改已有文档。性能验证优先最简Down-only，不自动扩大矩阵。
