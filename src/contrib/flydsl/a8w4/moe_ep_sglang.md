# moe_ep 接入 sglang：token 总数的来源与各 rank 一致

`moe_ep`（[moe_ep.py](moe_ep.py)）还没有接入 sglang。本文是接入前的分析（20261118），讨论两个问题：查表用的 token 总数从哪里来；为什么各 rank 用的这个值必须相同。

- **依据的代码**：pyhip `df0a553`；sglang 分支 `tq/a8w4-tpep-situ` 的 `061dc8b3a`。
- **sglang 代码位置**：路径相对 `python/sglang/srt/`，按函数名给出（行号会变）。
- **「未验证」**：读代码得出、还没有运行验证的结论。

## 1. 结论

- **TP**：每个 rank 拿到同一批 token，`x.shape[0]` 处处相同。各 rank 独立查表，选出的配置必然一致；估错只是慢。
- **EP（DP attention + EP MoE）**：`x.shape[0]` 只是本 DP rank 自己那一份 token。
  - 各 rank 的值不同：idle rank 是 0，有的还被 padding 过。
  - 表 [a8w4_moe_ep_tuned.csv](configs/a8w4_moe_ep_tuned.csv) 的分桶列 `~tokens` 是所有 rank 的 token 总数。
  - `moe_ep_dims` 直接用调用方传入的 `tokens`，不检查各 rank 是否相同。
- **各 rank 的值是否相同，比估得准不准更要紧**：`moe_ep` 的 solver 自己带通信，各 rank 选的 solver 族不同就会 hang（第 2 节）。
- **eager 下总数能在 host 上精确拿到，不需要 GPU 同步**：sglang 调度器每一步都 all_gather 了各 DP rank 的 token 数。难点有两处：这个列表在几个地方被丢掉或改写（第 4 节）；CUDA graph（第 5 节）。
- **host 上的 `tokens` 只决定 solver 族和 knob**：
  - `tokens` 是 dispatch 之前的总数，host 上可知；dispatch 之后本 rank 实际收到多少行，只在 device 上。
  - kernel 按 device 上的真实计数工作：mega 的 ep_prep 交换了全局每个专家的计数，mori 的 ep3k 读 MORI 的 total_recv。
  - 所以只要各 rank 一致，`tokens` 估得不准只影响速度，不影响结果。

## 2. 各 rank 选得不一致为什么会 hang

- **`mega` 和 `mori` / `mori_eager` 是两套协议**：
  - mega 的 ep_prep 等其他 rank 往 mega 自己的 symmetric buffer 里写计数 flag；
  - mori 等 MORI dispatch 的信号；
  - 有的 rank 选 mega、有的选 mori，两边互相等。
- **族内 knob 不一致**：
  - `mori` / `mori_eager`：共用 `EpComm` 里同一个 MORI op，`cfg` 只影响本卡的 ep3k，可以混用。
  - `mega`：接收端只等 `arrived[b]` 达到该块应收的路由数（[kernels/ep_mega.py](kernels/ep_mega.py) 文档），与发送端有几个 producer 无关。从协议看，producers / lag / bcm 可以混用。**未验证**。
- **mega 的第一次调用也是集合操作**：`EpComm.mega()` 在第一次调用时构造 `MegaMoeEpA8W4`，其中分配 symmetric buffer 并调用 `shmem_barrier_all`。所有 rank 要在同一次调用里第一次选中 mega。
- **运行中改表**：kernelkit 每个 rank 每秒最多读一次表的 mtime。服务运行时改了表，各 rank 换用新表的时刻不同，可能短暂不一致。上线时表要冻结。

## 3. 表里两个族的交界

[a8w4_moe_ep_tuned.csv](configs/a8w4_moe_ep_tuned.csv) 每个点上各 solver 最快配置的耗时。

- **测量条件**：EP8，4096 / 2048，256 个专家，topk 8，uniform 路由，token 均分到各 rank。
- **耗时**：单位 us，取各 rank 对齐后最慢的那个 rank。

| tokens（总数） | mega（silu） | mori（silu） | mori / mega（silu） | mori / mega（situv2） |
|---:|---:|---:|---:|---:|
| 64 | 176.1 | 169.0 | 0.960 | 0.965 |
| 128 | 192.9 | 183.5 | 0.951 | 0.944 |
| 256 | 200.5 | 189.1 | 0.943 | 0.926 |
| 512 | 211.4 | 218.5 | 1.034 | 1.037 |
| 1024 | 242.3 | 272.8 | 1.126 | 1.125 |
| 2048 | 365.8 | 537.1 | 1.468 | 1.382 |
| 4096 | 528.9 | 779.6 | 1.474 | 1.344 |
| 8192 | 898.8 | 1113.3 | 1.239 | 1.229 |
| 16384 | 1646.9 | 2013.6 | 1.223 | 1.240 |
| 32768 | 3159.9 | 3794.6 | 1.201 | 1.239 |

- **交界在 256 和 512 之间，正好是 decode 的常见范围**：每 rank 32–64 个请求 × 8 = 256–512。
- **估高比估低便宜，拿不准时取上界**：
  - 估高（真实 ≤ 256，却选了 mega）：慢 4–8%。
  - 估低（真实 ≥ 512，却选了 mori）：慢 3–47%，2048–4096 tokens 时最多。
- **`mori_eager` 不会被选中**：它的耗时是 mori 的 1.02–2.3 倍，在每个点都不是最快。

## 4. sglang 里各场景能拿到的 token 数

| 场景 | 本 rank 的 `x.shape[0]` | 层里能读到的全局信息 | 各 rank 一致？ |
|---|---|---|---|
| EP，没开 DP attention（attention TP + a2a） | 总数按 attn_tp_size 对齐后均分，各 rank 相同 | 不需要 | 是：`x.shape[0] × world`，与 TP 一样（**未验证**） |
| DP attention，`require_mlp_tp_gather=True`，eager | extend（SUM_LEN）：各 rank 不同；decode（MAX_LEN）：都 pad 到 max | `get_dp_global_num_tokens()`：SUM_LEN 时是真实值，MAX_LEN 时是 `[max] * dp` | 是（MAX_LEN 时是上界） |
| 同上，decode CUDA graph | 各 rank 都取 max 所在的 bucket | capture 时是 `[bs] * dp` | 是（上界） |
| DP attention，`require_mlp_tp_gather=False`，eager | 各 rank 不同 | 只有 `[本 rank 的数]` | **否** |
| 同上，decode CUDA graph | 各取本 rank 数所在的 bucket，各 rank 不同 | capture 时是 None（attn_tp_size = 1）或 `[bs]` | **否**：各 rank replay 不同 bs 的 graph |

- **`require_mlp_tp_gather` 何时为 False**（`utils/common.py`）：DP attention 打开、a2a 不是 none、`--enable-dp-lm-head`，并且 `--moe-dense-tp-size` ≤ tp / dp（例如 1）。这正是 DeepEP / MORI 常用的配置。
- **全局列表被丢掉**：
  - 调度器的 `prepare_mlp_sync_batch_raw` 总是 all_gather 全部 rank 的 num_tokens（`MLPSyncBatchInfo.all_gather`）；
  - 但 `_update_gather_batch`（`managers/scheduler_components/dp_attn.py`）在 `require_mlp_tp_gather=False` 时只把本 rank 的数写进 `batch.global_num_tokens`。
- **MAX_LEN 下列表被改写**：
  - `DpPaddingMode.get_dp_padding_mode`：有 extend 时用 SUM_LEN；decode 在 2·sum ≥ max·dp 时用 MAX_LEN，否则用 SUM_LEN。
  - MAX_LEN 下，`ForwardBatch.prepare_mlp_sync_batch` 把 `global_num_tokens_cpu` 改成 `[max] * dp`；两种模式下都先按 attn_tp_size 和 CP 对齐。
  - `get_dp_global_num_tokens()` 返回改写后的值；真实值在 `ForwardBatch.original_global_num_tokens_cpu`（`init_new` 里设置）。
  - MAX_LEN 下各 rank 的 batch 都 pad 到 max，padding 行的 topk_ids 是 -1。
- **`SGLANG_SCHEDULER_SKIP_ALL_GATHER=1` 时没有全局信息**：调度器根本不做 all_gather。
- **现有 MORI 路径的估法用在 `moe_ep` 上会 hang**：
  - `layers/quantization/mxfp4_aiter_a16w4_moe.py` 的 `_mori_token_bounds` 拿不到完整列表时，按 `dp_size × 本 rank 的数` 外推。
  - 这个值只决定 `a8w4_moe_dyn` 在本卡用哪个 ep3k 配置，各 rank 不一致只是慢；用在 `moe_ep` 上就会 hang。
- **现有 MORI 路径已有一个性能问题**：
  - 上表最后一行的 capture 拿到的是 None，`_mori_token_bounds` 就退回整个接收缓冲区的容量；
  - 结果 decode graph 里的 ep3k 都按最大 batch 的配置跑。README 2.4 节：估计值等于容量时，小 batch 多 14–28 us。
- **sglang 的先例：用全局一致的量选协议**：
  - MORI / DeepEP 在 normal 和 low-latency 两套协议之间选，用的是 `get_is_extend_in_batch()`（`layers/moe/token_dispatcher/moriep.py`）；
  - 这个标志是 all_gather 之后取 max 得到的，各 rank 一定相同；
  - `moe_ep` 选族是同一类问题，应该走同一条路：用 all_gather 得到的全局量，不用本地量。

## 5. CUDA graph

- **选择在 capture 时定死，`tokens` 只能取上界**：
  - solver 和 knob 都在 capture 时确定；
  - `tokens` 只能取这张 graph 的上界：graph bs × num_tokens_per_bs × dp_size；
  - replay 时的真实总数 ≤ 这个上界。按第 3 节，取上界的代价可以接受。
- **硬性前提**：所有 rank replay 同一 bs 的 graph，或者所有 graph 都用同一个族。
  - `require_mlp_tp_gather=True` 时满足：`DecodeCudaGraphRunner.can_run` / `replay_prepare` 按 `max(global_num_tokens_cpu)` 选 bucket，各 rank 相同。
  - `require_mlp_tp_gather=False` 时不满足：按本 rank 的 `batch_size` 选 bucket。
- **例子**：rank 0 有 64 个请求，其余 rank 各 1 个，真实总数 71。
  - `require_mlp_tp_gather=True`：所有 rank replay bs=64 的 graph，按 512 选中 mega。按真实的 71（落在 128 那个点）本该选 mori，所以慢 5–6%。
  - `require_mlp_tp_gather=False`：rank 0 replay bs=64 的 graph（512 → mega），其余 rank replay bs=1 的 graph（8 → 64 那个点 → mori），结果 hang。

## 6. 总数之外

- **表的测量条件与真实情况有两处偏离**：表是在 token 均分到各 rank（`moe_ep_problem`）、uniform 路由下测的，`us` 取最慢的 rank。下面两种偏离 host 上都看不到：
  - **DP 不均衡**：同样 512 个 token，`[512, 0, …, 0]` 和 `[64] * 8` 的发送端负载完全不同，交界可能移动（未测）。
  - **路由倾斜**：热点专家所在的 rank 收到的行数多，最慢的 rank 更慢（未测）。
- **mega 的接收容量会静默丢路由（正确性问题）**：
  - `s_cap` 默认 = 2·tcap·topk + 32·128 行，tcap 是 `max_tokens` 向上取 2 的幂；
  - 超出 `s_cap` 的路由被丢弃（[megamoe.py](megamoe.py) `MegaMoeEpA8W4` 的文档；`rows_needed()` 可以查实际需要的行数）；
  - 所有 rank 的 token 总数 ≤ 2·tcap 时不可能溢出；
  - 总数更大时，只要某个 rank 收到的路由超过 2·tcap·topk 就会溢出。每 rank 都满载 tcap 时，这相当于均匀路由均值的 2 倍；
  - 大的 prefill chunk 遇上热点专家时可能发生。
- **`EpComm.max_tokens` 不在表的 key 里，却会改变 kernel**：它决定 token_shift、`s_cap` 和 grid。
  - mega 要求 `s_cap < 2^18`：topk 8 时 tcap ≤ 8192。
  - p8 配置另要求 tcap·topk ≤ 8·8·1024，同样是 tcap ≤ 8192。
  - 表是在 tcap = 4096（`moe_ep.CAP`）下调的，和 sglang `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 的默认值一致。其他 tcap 下的耗时没测过。

## 7. 接入清单

1. **sglang 给出各 rank 一致的全局 token 数**：
   - 调度器 all_gather 的结果里已经有完整列表，不需要额外通信。
   - 在 `_update_gather_batch` 里把它另存一个字段。不要改 `global_num_tokens`：它决定 DP gather buffer 的大小和 padding。
   - 经 `ForwardBatch` 传到层里，用一个类似 `get_is_extend_in_batch()` 的全局 getter 读。
   - 取值：
     - eager：各 DP rank 真实 token 数之和；
     - CUDA graph capture：graph bs × num_tokens_per_bs × dp_size；
     - 没开 DP attention：`x.shape[0] × world`。
2. **decode graph：各 rank replay 同一 bs**：
   - 用了 `moe_ep` 时，`require_mlp_tp_gather=False` 下 `can_run` / `replay_prepare` 也按 all_gather 得到的 max 选 bucket。
   - 不想改 graph 的选择逻辑的话，所有 decode graph 固定用一个族：例如用 `op.tuned(problem)` 筛出 mega 的配置，通过 `config=` 传入。代价是 ≤ 256 tokens 时慢 4–8%。族内 knob 仍按 graph bs 选。
3. **拿不到全局值时报错或固定族**，不要按本地值外推。
4. **idle rank 也要调用 `moe_ep`**（0 行）：
   - `Mxfp4AiterA16W4MoEMethod.apply` 在 a2a=none 路径上有 `if x.shape[0] == 0: return`；那里通信在外面，所以没问题；
   - `moe_ep` 路径不能有这种提前返回。
5. **其他路径同样处理**：EAGLE draft、draft-extend 的 graph runner，以及 TBO 的两个子 batch，各自有自己的 `global_num_tokens`。
6. **symmetric buffer 在 capture 之前分配**：
   - `EpComm` 第一次用到 mega / mori 时才构造它们；mega 的构造含 `torch.cuda.synchronize` 和 `shmem_barrier_all`，不能放在 capture 里；
   - 初始化时让各 rank 一起调用 `comm.mega(W, topk)` / `comm.mori(W, topk)`，或者在 capture 之前 eager 跑一次。
7. **防 hang**：
   - warmup 之后（或在 debug 开关下）all_gather 一次各 rank 为每个 graph 选出的 solver 名，不一致就报错；
   - 上线时冻结调优表。
8. **`EpComm.max_tokens`**：与 MORI 的每 rank 上限保持一致，并且 ≤ 8192。每 rank 超过这个数的 extend 要分块。
9. **调优侧**：在 `tune.EP_TUNING` 里加 DP 不均衡、hot routing 的 problem，看交界是否移动。如果移动明显，key 改成（总数，每 rank 最大值）。

## 8. 待验证

- mega 的 producers / lag / bcm 在各 rank 上取不同值时能否正常运行（8 卡，每个 rank 用不同的 `config=`）。
- 没开 DP attention 的 EP 下，各 rank 的 MoE 输入行数是否相同。
- DP 不均衡、路由倾斜对 mega / mori 交界的影响。
- sglang 现在的 MORI 路径怎么处理每 rank 超过 4096 token 的 extend。
