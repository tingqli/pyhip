# gfx942 BF16 D256 QSA

## 一个接口

从本包导入 `qsa`，调用 `qsa(q, k, v, indices, *, query_lens=None, prefix_lens=None, softmax_scale=None, out=None)`。
不需要构造contract、prepare/rebuild对象或管理scratch。

- Q为BF16 `[M,H,256]`，K/V为BF16 `[N,HK,256]`，均连续且16-byte对齐；推理专用。
- `H % HK == 0` 且 `H/HK <= 16`。主要验证TP2/4/8对应Q12/6/3、KV1；本函数不进行TP通信。
- `indices`为连续device int32 `[M,2051]`，每query独立选择，local heads共享；token ID在所属请求内编号。
- query位置$p$：`min((p+1)//4,512)`个不重复完整4-token块，接0–3个causal tail token，再以`-1`填充。
  完整块可无序；精确保留原选择，不近似合并各query的可见集合。
- host `query_lens`/`prefix_lens`描述packed请求。单请求默认`(M,)`/`(N-M,)`；多请求省略prefix表示全0。
  空query段仍计入KV偏移。
- scale默认`1/16`；可指定有限正数。`out`可复用，不得与Q/K/V或indices重叠。
- 首调用分配/校验元数据并JIT；热调用使用private per-layout/per-stream缓存，但每次按当前indices重建选择。
  graph须在**同一stream、同一布局**预热；捕获用workspace在进程生命周期内保留，普通缓存最多8项。
  调用者负责跨stream输入依赖；共享workspace的graph不得并发重放，也不得与使用该workspace的eager调用并发。
  当前FlyDSL同一进程只支持一个GPU；TP服务每rank独立进程。

## 仅保留胜出路径

| 文件 | 职责 |
|---|---|
| [qsa.py](qsa.py) | 唯一公共函数、token/block校验、private缓存与分流 |
| [dense.py](dense.py) | 每请求前`min(M,max(0,2051-prefix))`行；原tensor视图，64对齐用native linear，否则bounded DMA |
| [union.py](union.py) | membership/compact/mask/gate和BM128/BN64 union kernel；common前缀免mask |
| [direct.py](direct.py) | sorted BN32 fallback；低重合度时逐query计算，未选中CTA提前退出 |
| [test_qsa.py](test_qsa.py) | 输入生成、FP32 reference、真实输入hash/选择审计、正常测试与性能测试 |

固定auto/rho4：$U r \le 4\sum_i s_i$时用union，否则direct。requested BQ32按M128容量限制，TP2/4/8为8/16/32；grid上限`CU×2`。
没有公开mode/rho/BN/BQ等调参接口。内部依赖只有[共用helper](../mha/_common.py)与[linear D256 kernel](../mha/mha_pa_bf16_256_linear_942.py)，不依赖SGLang。

## 必要测试

[test_qsa.py](test_qsa.py)是唯一QSA测试/回放入口：

- 普通pytest：11种正常形状（dense边界、TP2/4/8、ragged/空段、NaN尾部、alias、rescale、选择更新和graph），
  8份真实layer/rank/shape输入，以及1个真实SGLang backend流程，共20项。原`rtol=atol=.02`不变。
- 真实输入默认使用已有8份采集，可用`QSA_REAL_INPUT_DIR`覆盖。
  显式目录不存在/无数据时报错；无本地数据的环境仍可运行合成功能测试。
- 性能需显式`-m perf`并设置`QSA_REPLAY_OUTPUT`到新的mytest/mydata子目录；8项，每项原cudaPerf、10独立buffer、2warmup、10sample、AB/BA交错。
  `QSA_REPLAY_REQUIRE_HALF=1`要求完整QSA不超过base一半。
- 直接运行同文件CLI：`--output`必需；`--inputs`指定采集文件，`--gpu`选卡，`--check-only`仅正确性，`--require-half`启用半base门槛。

每case仅base/QSA两个scope、共20raw；QSA包括恢复/校验、rebuild和dispatch，预分配输出，不计首次JIT、indexer或KV gather。
10组Q/K/V/indices/O独立分配；private scratch由单接口在同一stream复用。
检查**实际计时输出**、重复输入bitexact、真实地址、use≤5%/VRAM≤20%及PTL Enabled/VECTOR,F8；失败保留全部raw、不轮询、不改硬件。
不是TTFT或端到端性能声明。历史50/70raw报告保持其原口径。

## 临时SGLang与历史证据

SGLang仅在[sglang/README.md](sglang/README.md)所述临时目录：冻结baseline合成一个文件，插件一个文件，模型profile/输入采集一个脚本。
原分散的回放、summary/audit、构建包装和插件测试入口已删除；一次性核验证据放mytest/mydata，不作为永久工具维护。

- [本次整理验证](../../../../mytest/mydata/qsa_consolidation_20260925_01/README.md)：72项正常回归、13个ELF逐位一致；新性能在采样前门禁失败，无有效计时，整模型未重跑。
- [历史真实输入正式性能](../../../../mytest/mydata/qsa_real_study_20260925/README.md)：旧接口auto4完整3.82–4.60ms、base9.71–9.82ms；不是本次重测结果。
- [历史模型profile](../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/README.md)：0.1.4、48calls/rank三段216.74/216.49ms；非端到端2×。
- [opt.md](opt.md)仅为历史记录；旧数据、trace、冻结包与收据不改写。
