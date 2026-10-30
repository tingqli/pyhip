# QSA实现与优化记录

本目录是gfx942 QSA attention与indexer的生产实现，使用方法、测试命令和当前性能见[benchmark说明](../../../../../benchmarks/qsa/readme.md)。第0–3节对应当前源码（2026-10-30，attention按AITER#6576的结构重排之后），说明每个kernel的作用、数据流和伪代码，行号链接指向函数定义；第4节是文件、接口、编译、已知限制和当前数据来源；第5节是优化历史。

- [benchmark说明](../../../../../benchmarks/qsa/readme.md)只记录当前使用说明、最终有效性能和验收状态。改动、评估、回退和未采用的方案只追加到第5节，每项写清改动、结果和证据；源码改动后同步第0–4节。
- 原始数据、trace和冻结源码保留在mytest/mydata，不改写。功能通过、单kernel性能、完整调用和服务结果分别表述。
- TP2/4/8指单卡H12/H6/H3（HK=1）的kernel形状，不是分布式服务。

## 0. 记号与术语

- **wave、lane、CTA**：一个wave是64个同步执行的线程（lane 0–63）；CTA（workgroup）由若干wave组成，共享LDS（片上共享内存）。barrier指CTA内所有wave同步；`lane&15`、`lane>>4`等是lane号的低4位、高2位。
- **MFMA M×N×K**：一条矩阵乘加指令C[M,N] += A[M,K]·B[K,N]（BF16输入，FP32累加），A/B/C分散在wave的64个lane的寄存器里，每个lane持有固定的几个元素（称“操作数布局”）。本文用16×16×16（A、B每lane 4个值，C每lane 4个）和32×32×8（C每lane 16个）。
- **buffer读写**：带范围检查的全局访存，偏移越界时读返回0、写被丢弃。文中“越界偏移”就是用它做无分支的屏蔽。**DMA**指`buffer_load … lds`：全局内存直接写入LDS，不经寄存器。
- **wave内通信**：`ds_bpermute(源lane, x)`取另一个lane的x；`shuffle_xor(d)`与lane^d互换；`readlane(x, i)`把lane i的x广播给整个wave；`ballot(p)`是各lane谓词p组成的64位掩码；`mbcnt(mask)`是mask中编号小于本lane的置位数，用来按lane顺序紧凑写出。
- **XOR swizzle**：LDS地址的一部分位与行号异或，使一条宽读写指令的各lane落在不同bank，避免bank冲突。
- **块与步**：块b = 请求内token 4b..4b+3；因果尾是最后不满4个的token。N64步 = 64个token（16个块），BN32步 = 32个token。
- **exp2域softmax**：分数先乘scale·log2(e)再用exp2，与以e为底等价。

## 1. Attention

### 1.1 调用链

Attention沿用AITER [#6576](https://github.com/ROCm/aiter/pull/6576)（`2544b2927`）的源码结构和kernel名（`sparse_prefill_*`，10-30同步，见第5节），在其上保留PyHIP的K2、slot表读K/V池和按行数打包。一次调用在当前stream上依次发出下列launch（编号沿用，K4已删除），全部是FlyDSL kernel。不建union计划时只有K1和direct（packed为K6、K7，raw为K7′）。

| # | Kernel | grid×block，LDS | 作用 |
|---|---|---|---|
| K1 | sparse_prefill_recover_scatter（名称沿用AITER，那里K1还把行位散射进dense位图） | ⌈M/4⌉×256，每行一个wave | 不校验选择；把每行的T个块号还原并升序排序，写block_indices |
| K2 | sparse_prefill_compact | tiles×256（BQ≤21）或×512（BQ32/42），动态LDS≤32KiB | 读tile各行的block_indices求并集：计数、逐tile提案、压缩块表与成员；program 0复位direct_flag |
| K3 | sparse_prefill_order_masks | (排序CTA+tiles)×256 | 全局路由（步数阈值）、union任务排序与active、精确mask；有slot表时另写池块号和尾行 |
| K5 | sparse_prefill_union_bf16_d256 | min(tasks, CU)×512，64KiB | tile内query共享并集块，M128×N64流水 |
| K6 | sparse_prefill_pack_kv_bf16_d256（gated时为…_pack_kv_gated_bf16_d256） | ⌈packed块·HK/8⌉×256 | 各请求KV按4-token块重排成MFMA操作数顺序；gated时先读direct_flag |
| K7 | sparse_prefill_direct_packed_bf16_d256 | gated为(BQ·HK, tiles)×64，否则rows·HK×64；8KiB | union未接手的行，每wave一个query×一个KV头 |
| K7′ | sparse_prefill_direct_bf16_d256（raw，不打包时，见1.2.2） | Σ⌈rows_r/4⌉·HK×256，32KiB | 同上；4个wave为同请求连续4个query，直接读原KV |

传入`kv_slots`时K/V是token池（见1.2.9）：launch序列不变，K3另为union tile写池块号和尾行，K5、K6、K7′按池行取址，K1、K2和packed K7不变；池的字节跨度超过`_MAX_SPAN`（4GiB−1MiB）时K7′换成64位global读的WIDE变体。

### 1.2 伪代码

#### 1.2.0 K1–K3整体流程

attention把query分成两路计算：union让一个tile的BQ行共享这些行所选块的并集，用M128×N64的MFMA一起算；direct让每行单独读自己选中的块。K1–K3在GPU上为每个tile决定走哪一路，并备好两路要读的表。tile是同一请求内连续的至多BQ行（各tile行数至多差1），BQ在H12/H6/H3时分别为10/16/32（单请求、无prefix、HK=1且行数至少21·CU（H6）或42·CU（H3）时，H6/H3为21/42），由host在建union计划时切好。总行数少于`max(384, 64·H/HK)`，或union scratch会超过4GiB（`UNION_SCRATCH_LIMIT`）时不建计划，只跑K1，所有行走direct。

1. **K1（每行一个wave）**：把该行选中的至多T个完整块（T = (indices宽度−3)/4，256/512/768/1024）还原成块号并升序排序，写入`block_indices`，K2和direct都读它。K1不检查选择布局（见1.2.3）。
2. **K2（每tile一个CTA）**：读本tile各行的`block_indices`和各行的因果尾块，求出`blocks`（tile内各行所选块的并集，升序）和`membership`（每块被哪些行选中，第i位是tile的第i行）。tile的因果范围放得进LDS时，每个块号一个成员字，各行把自己那一位原子或进LDS，再按块号顺序压缩非0的字；范围更长时每次处理65536个块的窗口：先置存在位图，对位图做前缀和得到每块在并集里的序号，再按序号在LDS里累积成员字。例如tile首行是第100行，第103行选了块5，块5的成员字就有第3位。代价比较：direct为各行BN32步数之和；union为并集块数按16对齐，一个N64步记为16个BN32步。union不超过direct的1.7倍时提案为1（union），否则为0（direct）；超过1.7倍就停止压缩。
3. **K3（一次launch，两类CTA并行）**：
   - 排序CTA汇总全部tile的提案和代价，用标定的延迟模型选一个步数阈值L：union只保留步数≤L的提案1 tile，更长的改走direct。只有union受最长任务限制时，才在8档L（全部保留、6档降级、全部不保留）里选模型最快的一档，而且须比全部保留快5%以上。然后写`active[tile]`和`direct_flag`，并把union任务（tile×KV头×128行切片）按N64步数降序、蛇形排进持久grid（`task_order`）。
   - mask CTA为提案1的tile生成`score_masks`：对每个N64步、每个query，标出该query选中且因果可见的token。mask与路由并行生成，被路由改走direct的tile，其mask不会被读取。
4. **之后**：K5 union按`task_order`只处理active的tile；`direct_flag`=1时，K6先把KV重排成direct的操作数顺序，K7再处理非active tile的行。不建计划时，K6/K7无条件执行，K7处理全部行。

三个launch之间的kernel边界就是全局同步点：K2要等tile内所有行写完`block_indices`，K3的路由要等所有tile算完代价。在kernel内部做全局同步更慢（见第5节“10-02 K1–K4的减少与融合”）。

| 缓冲区 | 写 | 读 | 内容 |
|---|---|---|---|
| `block_indices[M,T]` | K1 | K2、K7 | 每行升序块号，其余为-1 |
| `blocks`、`membership[tiles,CAP]` | K2 | K3 mask CTA；K5只读`blocks` | tile并集的块号（升序）和成员位 |
| `counts[tiles]`、`costs[tiles,3]` | K2 | K3；K5另读`counts` | 并集块数；(direct步数, 最长行步数, 提案) |
| `active[tiles]`、`direct_flag` | K3（`direct_flag`先由K2清0） | K5、K6、K7 | tile是否走union；是否有行走direct |
| `task_order` | K3 | K5 | union任务的执行顺序 |
| `score_masks[tiles,CAP/16,BQ,4]` | K3 | K5 | 每N64步、每query要保留的token位（int16） |

#### 1.2.1 公共入口（[attention](attention.py#L219)）

```python
attention(q[M,H,256], k/v[N,HK,256], indices[M,4T+3], query_lens?, prefix_lens?, softmax_scale?, out?, kv_slots?):
    # host检查，每次调用都做，不读GPU数据
    scale = 1/16 if softmax_scale is None else float(softmax_scale)   # scale·log2(e)须是有限、为正的FP32值
    kv_slots给出时：同设备连续int32 [keys]，K/V是池[P,HK,256]；否则keys = N
    query_lens默认(M,)；prefix_lens默认：单请求(keys-M,)，多请求全0；各长度为int且0 ≤ 长度 < 2^31，Σq==M、Σ(q+p)==keys
    Q/K/V：3D、连续BF16、D256、16B对齐；H%HK==0且H/HK≤16；gfx942
    字节跨度：Q和每个请求的K/V（有slot表时只查Q）≤ _MAX_SPAN = 4GiB−1MiB   # 无符号32位字节偏移，1MiB余量给越界哨兵
    indices：同设备连续int32 [M,4T+3]，T∈{256,512,768,1024}，字节数 < 2GiB
    out可省略，给出时须同形、连续、16B对齐，且不与Q/K/V/indices重叠
    if M == 0: return out
    with 全局锁, 切到q所在设备:
        每进程只允许一张GPU
        key = (device, stream, query_lens, prefix_lens, H, HK, scale, T, 有kv_slots时的池行数P)
        ws = LRU[key]                       # 未capture的最多保留8个（WORKSPACE_CACHE）；capture过的不淘汰
        if ws 不存在: capture中则报错；否则 ws = _Workspace(...)
        ws.captured |= 正在capture
        ws.attach()                         # 按arena当前的缓冲切好scratch视图（见1.2.2）
        if ws.hot存在:
            ws.hot.prepare_call(indices, kv_slots)       # K1 → K2 → K3：预建参数的FlyDSL launcher，K1只换indices，slotted的K3只换kv_slots
            ws.hot.compute_call(q, k, v, out, kv_slots)  # K5，再K6+K7（raw为K7′）；K/V按3D传入
        else:
            ws.hot = _Hot(ws, q, k, v, indices, out, kv_slots)  # 第一次：按常规路径（prepare.run、union.run、direct.run）完成本次调用，同时建好上述launcher
    return out
```

没有首次调用预编译：每个kernel变体在第一次用到时编译（见4.3）。

#### 1.2.2 工作区、scratch与计划（[_Arena](attention.py#L50)、[_Workspace](attention.py#L76)、[allocate_plan](sparse_prefill_prepare.py#L1275)、[direct.prepare](sparse_prefill_direct.py#L169)）

```python
class _Workspace:      # 每个新的(device, stream, layout)创建一次
    # R个请求；请求r有q_r个新query、p_r个prefix token，KV长度kv_len_r = q_r + p_r
    numpy生成query_positions[M] = p_r+i，经pinned内存non_blocking上传（K1、K2读）
    wide = 有kv_slots 且 池字节数 > _MAX_SPAN（4GiB−1MiB，只换raw direct的变体）
    if M ≥ max(384, 64·H/HK):
        plan = allocate_plan(query_tile=32, slotted)
        union = plan if plan的scratch ≤ 4GiB（UNION_SCRATCH_LIMIT）else None
    direct = direct.prepare(union)
    specs = block_indices[M,T] + union的scratch + direct的PK/PV
    arena = _arenas[(device, stream)]; arena.reserve(各视图按256B对齐后的总大小)

class _Arena:          # 每个(device, stream)一个；弱引用字典，有缓存或capture的工作区引用时存活
    buffers[0]放全部scratch；每次调用都重写自己要读的内容，不需要清零
    reserve(size): 不够 → 重新分配为max(需要, 1.25×原大小)（torch.empty）
                   → generation+1 → 所有未固定的工作区release()（丢掉视图和hot）
    # attach()：generation变了就按specs在buffer中切256B对齐的视图（bind()先调attach()）
    #          capture中则 pinned = buffers，此后不再随arena换缓冲

allocate_plan:
    G = H/HK;  full = (HK==1 and 单请求 and prefix==0)
    BQ = 21 if G==6 and full and M ≥ 21·CU            # 126/128个MFMA行（Platform.wide_tiles_per_cu = 1）
       = 42 if G==3 and full and M ≥ 42·CU            # 同上，成员位用64位
       = 否则 min(32, 10 if G==12 else 不超过128/G的最大2的幂)   # G=12/6/3时为10/16/32
    每请求等分为ceil(q_r/BQ)个tile，各tile行数至多差1（大的在前）
    meta[tile] = (q_row0, rows, k0, kv_len, position0)   # 首行在Q中的行号、行数、请求在K/V（或slot表）中的起始行、请求KV长度、首行在请求内的位置
    MAXB = ceil(max_kv/4)；CAP = 16·ceil(min(MAXB, BQ·(T+1))/16)
    slices = ceil(BQ·G/128)；tasks = tiles·HK·slices；grid = min(tasks, CU)   # union CTA用满一个CU的64KiB LDS
    route_costs = (HK·slices/grid, direct_cu_step·HK/CU, direct_fixed + pack_block·packed块数·HK)   # µs，Platform的MI308X标定：0.304、41、0.0041
    设备张量：metadata、counts[tiles]、costs[tiles,3]、active[tiles]、task_order、direct_flag、query_tiles[M]
    arena：blocks/membership[tiles,CAP]（成员BQ>32时为i64）、score_masks[tiles,CAP/16,BQ,4] i16
    slotted时另有pool_blocks[tiles,CAP]、tails[tiles]（K3写、K5读）

direct.prepare:
    blocks_r = ceil((q_r+p_r)/4)（q_r>0时）；bytes = Σblocks_r·4·HK·1KiB（PK+PV）
    packed = (bytes ≤ 64MiB or (4·Σblocks_r ≤ 32·M and bytes ≤ 1GiB))   # 上下文不超过行数的32倍时整段打包更快（第5节10-30）
             and (有slot表 or max(Σkv_len, 4·Σblocks_r)·HK·512 < 2^32)   # 无slot表时pack从整批K/V起点按无符号32位寻址
    waves = 1 if packed else 4；每请求每waves行一个direct tile：meta = (q_row0, rows, base, extent, position0)
        packed：base = 4·Σ_{i<r}blocks_i（PK内起点），extent = 4·blocks_r
        raw：base = k_start_r，extent = kv_len_r
        # slotted时raw的k_start/kv_len是slot表中的下标范围；packed时pack从池里读，PK/PV与原来相同
    packed时 pack_sources[b] = 所属请求的k_start + 4·(块在请求内的序号)；PK/PV [Σblocks·4, HK, 256]放arena
    gated = (union存在)：复用union的active、query_tiles、direct_flag、metadata、BQ；否则用无关张量占位
```

Platform（[sparse_prefill_common.py](sparse_prefill_common.py#L56)）按gfx型号和CU数取MI308X的标定（路由模型系数、union阈值1.7、建计划的最少行数、打包预算等）；环境变量`AITER_QSA_PLATFORM`可覆盖其中的项（如`union_step=3.0,max_packed_kv_bytes=0`），只用于实验。

#### 1.2.3 K1 恢复（[_recover_kernel](sparse_prefill_prepare.py#L245)、[_recover_row](sparse_prefill_prepare.py#L201)）

```python
# 每CTA 4个wave，每个wave一行；从最后一行发起：因果请求里越靠后的行可见块越多，先发最重的行
row = M-1-(4·blockIdx + wave)
vis = query_positions[row] + 1          # 该query在请求内的位置+1 = 因果可见的token数
C = min(vis//4, T)                      # 应选的完整块数
# lane持有第k·64+lane列（每lane T/64列；每次读、写都是64个相邻列），一列对应一个4-token块
blk[col] = indices[row, 4·col] >> 2 if col < C else INT_MAX      # 只读每组第一个token
if wave内有col < C使blk[col] ≠ col:   # 块号恰为0,1,2,…（完整因果前缀）时已经有序，免排序
    经本wave的LDS转置成每lane连续的列 → bitonic排序（补INT_MAX到2的幂）→ 转置回来
    # 调用方的块可以乱序，排序后direct按固定的升序累加，结果与输入顺序无关
block_indices[row, col] = blk[col] if col < C else -1   # K2（并集）和K7（direct）读的每行块表
```

K1不检查选择布局：组内4个token不连续、块重复、超出因果范围或缺少因果尾时结果错误，不报错；同一stream的工作区共用scratch，错误还可能影响该stream之后的调用。因果尾不由K1读，K2、K3和direct都按位置推出（最后不满4个的token）。名称里的scatter来自AITER，那里K1还把行位写进dense位图；本仓库的K2直接读`block_indices`求并集（10-29），K1不散射。

#### 1.2.4 K2 计数/公式/并集（[_tile](sparse_prefill_prepare.py#L622)、[_dense](sparse_prefill_prepare.py#L528)、[_window](sparse_prefill_prepare.py#L577)、[_blocks](sparse_prefill_prepare.py#L450)、[lds_layout](sparse_prefill_prepare.py#L824)）

```python
# 每tile一个CTA：BQ≤21时256线程，BQ为32/42时512线程；从最后一个tile发起（因果请求里最后的tile块最多）
# program 0顺便把direct_flag清0，K3发现有tile不走union时再置1
rows, pos0 = meta[tile]                         # tile的行数、首行位置
end = min(MAXB, ceil((pos0+rows)/4))            # 最后一行可见的块数：各行的块都在[0, end)内

# 代价口径：direct按BN32步（每步32个token）计；union每个N64步（16个块=64个token）记为16个BN32步，
# 所以“16·ceil(count/16) ≤ r·direct_tiles”表示union不超过direct的r倍（r = union_ratio = 1.7）
for j in 0..rows−1:                             # tile内第j行（lane j算，wave内归约）
    vis_j = pos0 + j + 1；tokens_j = min(vis_j//4, T)·4 + vis_j%4；steps_j = ceil(tokens_j/32)
direct_tiles = Σ_j steps_j；longest = max_j steps_j
budget = 1.7·direct_tiles                       # 超过它就走direct，不必再压缩

# tile的块 = 各行block_indices里的块（K1已升序，-1跳过）+ 各行的因果尾块vis_j//4（vis_j%4 ≠ 0时）
# _blocks按元素顺序遍历rows·T个元素：线程t一次发16个读（元素t + k·threads），同一wave的lane读连续元素
# lds_layout（host）：全批最长范围MAXB每块一个成员字（4B，BQ>32时8B）放得进32KiB（LDS_BUDGET，每CU两个CTA）时
#   LDS按它分配，dense_limit = MAXB；否则LDS = 32KiB，dense_limit = 8184块（64位成员时4092），更长的tile走窗口
if end ≤ dense_limit:                           # _dense：LDS中每块一个成员字
    word[0:end] = 0；barrier
    for 第j行的每个块b: atomic_or(word[b], 1 << j)            # LDS原子或
    barrier
    每个wave分到连续的一段字，每轮64个：先数ballot(字≠0)的位数 → 各wave的个数经LDS求前缀 → 并集大小U
    if 16·ceil(U/16) ≤ budget:
        再扫一遍：非0的字b写到 位置 = wave起点 + mbcnt(ballot)：blocks[tile, 位置] = b，membership[tile, 位置] = word[b]
    count = U
else:                                           # _window：每次65536个块，超出预算即停
    count = 0
    for w0 in 0, 65536, … < end:
        presence[0:窗口字数] = 0；barrier                                # 存在位图，最多8KiB
        for 窗口内的每个块b: atomic_or(presence[(b−w0)>>5], 1 << ((b−w0)&31))
        barrier
        每个线程一段连续的字：位数之和 → CTA前缀和 → prefix[字] = 窗口内在它之前的存在块数；present = 窗口内并集块数
        if 16·ceil((count+present)/16) > budget: 提案0，停止
        各线程按字顺序把存在的块写到blocks[tile, count + prefix[字] + 字内序号]
        for 并集序号每4088个一段（64位成员时2044个；LDS里位图和前缀之后的全部空间，16KiB−32B）[c0, c0+chunk):
            member[0:min(chunk, present−c0)] = 0；barrier
            for 第j行在窗口内的每个块b:                              # 再遍历一遍tile的块
                i = rank(b) − c0             # rank = prefix[字] + popcount(字内低于b的位)
                if 0 ≤ i < chunk: atomic_or(member[i], 1 << j)
            barrier；membership[tile, count + c0 + i] = member[i]
        count += present
# 超出预算的tile走direct，count不完整，之后只读提案为1的tile的count
counts[tile] = count
costs[tile] = (direct_tiles, longest, 1 if 未超出预算 else 0)
# 工作量随选中的块数增长；只有窗口路径每32个块另加一个位图字，1M时每tile最多4个窗口
```

#### 1.2.5 K3 路由/排序/mask（[_order_masks_kernel](sparse_prefill_prepare.py#L1077)、[_route](sparse_prefill_prepare.py#L879)、[_route_cost](sparse_prefill_prepare.py#L849)、[_order](sparse_prefill_prepare.py#L942)、[_masks](sparse_prefill_prepare.py#L1007)）

K3是一次launch里两类互不依赖、并行执行的CTA：

- 排序CTA（`task_order`长度不超过4096时只有1个）：
  1. 全局路由：算出一个步数阈值L，union只保留N64步数不超过L的提案1 tile，更长的改走direct。只有union受最长任务限制时L才会小于最长步数（降级最长的一批，最多全部改走direct），否则保留全部提案1 tile。
  2. 发布结果：`active[tile]`（1 = 走union）；有任一tile走direct时`direct_flag` = 1，K6/K7据此决定是否执行。
  3. 给union任务排序：任务 = (tile, KV头, 128行切片)，按N64步数从长到短排，再蛇形分给持久grid的各CTA，写入`task_order`。
- mask CTA（每tile一个，4个wave各管1/4的N64步）：为提案1的tile生成`score_masks`，即每个N64步、每个query在64个key列里要保留哪些token。mask只依赖K2的输出，不等路由结果；被路由改走direct的tile，其mask不会被K5读取。

输入：K2写的`counts`（并集块数U）、`costs`（direct步数和、最长行步数、提案）、`blocks`/`membership`和`meta`。host参数：TASKS = tiles·HK·slices、GRID、排序宽度SIZE、键移位SHIFT/WIDE、排序CTA数、QB（=BQ）、CAP，以及`allocate_plan`按layout算好的三个时延系数：TASK_SHARE = HK·slices/grid，DIRECT_STEP = 0.304·HK/CU，DIRECT_FIXED = 41 + 0.0041·packed块数·HK。下文的各系数（10、3.2、0.7、0.05、41、0.0041、0.304）都取自Platform（见1.2.2）。

**① 全局路由**

union与direct先后执行，总时间 = union阶段 + direct阶段。逐tile规则只比较单个tile两种走法的工作量，看不到两种整体效应：

- union阶段的时间不低于最长的那个union任务（一个任务由一个CTA顺序执行）。任务少、某个tile特别长时，CU再多也帮不上。
- direct阶段有固定开销（约41µs，另加按打包块数计的pack时间）。只要还有一个tile走direct，就要付这笔开销。

时延模型（µs，由MI308X分阶段计时拟合；某阶段没有工作时记0）：

| 阶段 | 时间 |
|---|---|
| union | 10 + 3.2·max(最长union tile的N64步数, 全部union步数·TASK_SHARE) |
| direct | DIRECT_FIXED + max(0.7·最长direct行的BN32步数, DIRECT_STEP·全部direct步数) |

union受最长任务限制，指提案1中最长tile的步数M大于全部union步数·TASK_SHARE（每个CU平均分到的步数）。这时大部分CU早早做完，整个union阶段都在等最长的任务。direct按行并行，每行最多65个BN32步（约45µs下限），把最长的union tile改走direct能缩短关键路径。就是此时，路由用一遍扫描算出8档L的模型时间，取最快的一档；第0档的时间乘0.95，即其它档须快5%以上；并列时取靠前的档：

| 档 | 阈值L | union保留 | 含义 |
|---|---|---|---|
| 0 | M | 全部提案1 tile | 保持K2的逐tile提案 |
| 1–6 | ⌊M·2^(−j/2)⌋：0.71M、0.5M、0.35M、0.25M、0.18M、0.125M | 步数≤L的提案1 tile | 降级最长的一批 |
| 7 | −1 | 无 | 全部走direct |

不受最长任务限制时L = M，全部保留；关闭路由（只用于测试和benchmark的强制union）时L = 2^31−1。吞吐受限时改走direct只是把工作量搬到另一边，模型在这种情况下误差较大（实测去掉这个条件时long_high M4096 TP2慢37.6%）。

比值略高于1.7的tile一律走direct；若它们是仅有的direct工作，也要为这几个tile单独跑一次pack和direct。

示例（按模型手算的示意，不是实测；CU=80、HK=1、slices=1；H12时grid = min(任务数, 80)）：

- 例1，全部保留：TP2合成M12000，1200个tile，grid=80，TASK_SHARE=1/80，DIRECT_FIXED=41+0.0041·3000≈53.3。设K2给出633个union tile（平均40步，最长60步）和567个direct tile（direct步数合计340200，最长行65步）。M=60 < 633·40/80=316.5，union不受最长任务限制，L = 60。按模型，保持为1022.8+1346.1=2368.9，全direct（union tile改走direct也约600步，共约720000步）为53.3+2736=2789.3。
- 例2，降级：80个union tile各4步，另有1个union tile 64步（若走direct为650步，最长行65步）；短tile若走direct各300步、最长行不超65步；grid=min(81, 80)=80，DIRECT_FIXED按50算。M=64 > 384/80=4.8，受最长任务限制，8档为64/45/32/22/16/11/8/−1：

  | L | union保留 | union | direct | 合计 |
  |---|---|---|---|---|
  | 64 | 全部81个 | 10+3.2·64=214.8 | 0 | 214.8（×0.95=204.1） |
  | 45–8 | 80个短tile | 10+3.2·max(4, 320/80)=22.8 | 50+max(0.7·65, 0.0038·650)=95.5 | 118.3 |
  | −1 | 无 | 0 | 50+max(45.5, 0.0038·24650=93.7)=143.7 | 143.7 |

  取L=45：80个短tile在union里一轮做完，长tile的10行改走direct，分散到各CU上。
- 例3，全部走direct：6个tile，grid=6，TASK_SHARE=1/6，DIRECT_FIXED按50算。提案1为T0、T1、T2，分别4、6、30个N64步（若走direct为60、80、400步，最长行7、9、50步）；提案0为T3、T4、T5（direct 75、100、90步，最长行8、12、11步）。M=30 > 40/6≈6.7，8档为30/21/15/10/7/5/3/−1：
  - L=30：union 10+3.2·30=106，direct 50+max(0.7·12, 0.0038·265)=58.4，合计164.4（×0.95=156.2）。
  - L=21–7：留T0、T1，union 29.2，direct 50+max(0.7·50, 2.5)=85，合计114.2；L=5：只留T0，22.8+85=107.8。
  - L=3和−1：一个不留，只有direct，805步合计85。最小值先出现在L=3，结果与−1相同：全部走direct。T2一个union任务就要约96µs，全部走direct时805步分摊到80个CU，下限只是最长行的35µs。

**② 发布路由与任务排序**

- tile的最终路由：active = 提案==1 且 N64步数 ≤ L。
- 排序用KV头优先的rank（rank = hkv·tiles·slices + tile·slices + slice），所以同步数的任务先排完KV头0的所有tile，再排KV头1。发布给K5的是tile优先的编号task = (tile·slices + slice)·HK + hkv，K5据此解出tile、KV头和128行切片。
- 键 = (步数 << SHIFT) + (TASKS−1−rank)，降序排序：步数长的在前，同步数按rank从小到大。不走union的任务步数记0，排在最后，K5读到后跳过。
- 蛇形分配：排序后第j名放到`order[r·GRID + (r为偶数 ? c : GRID−1−c)]`，(r, c) = divmod(j, GRID)。持久grid的CTA c依次执行`order[c]`、`order[c+GRID]`……所以第0轮CTA c拿第c长的任务，第1轮反过来，拿到最长任务的CTA接着拿第1轮里最短的。
- `task_order`长度（任务数向上取整到GRID的倍数）超过4096时有多个排序CTA，各自只对自己那4096个rank排序，结果占`order`里对应的那一段。常见layout都在这个范围内。

示例：6个union任务的步数依次为[5, 9, 3, 9, 7, 1]（rank 0–5），GRID=3。排序后rank依次为1(9)、3(9)、4(7)、0(5)、2(3)、5(1)；蛇形后`order` = [1, 3, 4, 5, 2, 0]。CTA0执行rank 1和5，共10步；CTA1执行rank 3和2，共12步；CTA2执行rank 4和0，共12步。若按rank直接轮流分配，三个CTA分别为14、16、4步，最慢的CTA要16步。

**③ mask**

- 每个mask CTA处理一个tile（从最后一个tile开始），4个wave分别处理part 0–3：part p负责的N64步nt满足`nt mod 64`∈[16p, 16p+16)。wave内lane//4选nt，lane%4为quarter（每个N64步的64列分为4组，每组16列）。
- 一个N64步对应并集里的16个slot（16nt..16nt+15）。quarter q负责slot 16nt+2q、+2q+1、+2q+8、+2q+9，正是K5的MFMA里该lane持有的16个分数列（每块4个token）。
- 对tile的每个query：可见token数vis = min(pos0+query+1, kv_len)。对每个slot：该query选了这个块，就取块内可见token的位`(1 << clamp(vis − 4·块号, 0, 4)) − 1`（完整块0xF，尾块只有前vis%4位），否则为0。4个slot的4位拼成16位，写入`score_masks[tile, nt, query, quarter]`。
- K5每步每lane只读一个16位字（[_read_mask_g2r](sparse_prefill_union.py#L146)，`buffer_load_ushort`），第i位为0的分数列置为−inf（[_apply_mask](sparse_prefill_union.py#L156)，按位选择，不用比较）。

示例：BQ=2，两行位置为161、162，vis分别为162、163：完整块都是0–39，因果尾都在块40，分别有2、3个token。某N64步quarter 0的4个slot依次为块5、9、33、40（并集升序，slot 2–7是9与33之间的其它块）；q0选了块5、33和尾块40，q1选了块9、33和尾块40。

| query | 块5 | 块9 | 块33 | 块40 | 16位字 |
|---|---|---|---|---|---|
| q0（vis 162） | 0xF | 0x0 | 0xF | 162−160=2 → 0x3 | 0x3F0F |
| q1（vis 163） | 0x0 | 0xF | 0xF | 163−160=3 → 0x7 | 0x7FF0 |

K5在这16列上：q0保留块5、33的全部token和块40的前2个，其余为−inf；q1保留块9、33的全部token和块40的前3个。

```python
# host（_launches）
slices = ceil(BQ·G/128)；TASKS = tiles·HK·slices
numel = len(task_order) = ceil(TASKS/GRID)·GRID
SIZE = min(numel, 4096)；order_ctas = ceil(numel/SIZE)
SHIFT = max(1, bit_length(TASKS−1))；WIDE = ((CAP/16) << SHIFT) + TASKS−1 ≥ 2^31      # 键是否要用int64
grid = order_ctas + tiles                         # 每个tile一个mask CTA；所有CTA都是4 warps
# SIZE、SHIFT、QB、CAP、TASKS都是运行时参数：每线程1/2/4/8/16个键（≤256–≤4096个键）的五档排序编译在同一kernel里，按SIZE选一档

_order_masks_kernel(pid):
    if pid < order_ctas:                          # 排序CTA
        L = _route() if 开启路由 else 2^31−1    # 关闭路由时保留全部提案1 tile
        _order(pid, L)
    else:
        _masks(pid − order_ctas)

_route():      # 每个排序CTA各算一遍（结果相同），省一次全局同步；一个wave的64个lane分段扫全部tile，最后归约
    for tile in 全部tile:                         # 第一遍只看提案1
        s = ceil(counts[tile]/16) if 提案==1 else 0      # 提案0的count不完整，不用
        S += s；M = max(M, s)
    if M ≤ S·TASK_SHARE: return M                 # 不受最长任务限制：全部保留
    L_k = [M, int(M·2^(−1/2)), …, int(M·2^(−6/2)), −1]，k = 0..7
    for tile in 全部tile:                         # 第二遍：8档一起累计
        s = ceil(counts[tile]/16)；d, r, p = costs[tile]       # direct步数和、最长行步数、提案
        对每档k：p==1且s ≤ L_k → k_sum_k += s，k_max_k = max(k_max_k, s)
                 否则 → d_sum_k += d，d_max_k = max(d_max_k, r)
    cost_k = [k_sum_k>0]·(10 + 3.2·max(k_max_k, k_sum_k·TASK_SHARE))
           + [d_sum_k>0]·(DIRECT_FIXED + max(0.7·d_max_k, DIRECT_STEP·d_sum_k))
    cost_0 ×= 0.95                                # 其它档须快5%以上
    return L_argmin(cost)                         # 并列取靠前的档

_order(pid, L):
    rank = pid·SIZE + [0, SIZE)                   # KV头优先：rank = hkv·(TASKS/HK) + tile·slices + slice
    tile = (rank mod (TASKS/HK)) // slices
    active = costs[tile].p == 1 and ceil(counts[tile]/16) ≤ L
    if rank < TASKS/HK and rank % slices == 0:    # 每个tile由KV头0的切片0发布
        active_out[tile] = active
        if not active: direct_flag = 1            # K2的program 0已清0
    cost = active ? ceil(counts[tile]/16) : 0     # 不走union的任务排到最后
    key = (cost << SHIFT) + (TASKS−1−rank)；rank ≥ TASKS的位置为−1；WIDE时用int64
    sorted = sort(key, 降序)                       # 每个排序CTA只排自己的SIZE个rank
    r_task = TASKS−1 − (sorted的低SHIFT位)          # 还原rank
    task = (r_task mod (TASKS/HK))·HK + r_task // (TASKS/HK)    # 换成tile优先编号，给K5；补位（键为−1）写−1
    第j名（j = pid·SIZE + i）写到 order[r·GRID + (r为偶数 ? c : GRID−1−c)]，(r, c) = divmod(j, GRID)

_masks(cta):                                      # 每CTA 4个wave，每个wave一个(tile, part)
    tile = tiles−1 − cta；part = wave             # 从最后一个tile开始
    if proposal(tile) == 0: return                # 不等路由；被改走direct的tile的mask不会被读
    count = counts[tile]；rows, pos0, kv_len = meta[tile]
    slotted时part 0的lane 0写tails[tile]，各lane为自己的slot写pool_blocks（见1.2.9）
    nt0 = part·16 + lane//4；quarter = lane % 4
    for nt in nt0, nt0+64, nt0+128, …（nt < ceil(count/16)）:   # 4个part交错覆盖全部N64步
        slot_g = 16·nt + 2·quarter + {0, 1, 8, 9}[g]           # 该lane在K5里持有的16列 = 4个块
        member_g, block_g = membership[tile, slot_g], blocks[tile, slot_g]   # slot ≥ count视为空，每个slot只读一次
        for q in 0..BQ−1:                         # 运行时循环
            vis = min(pos0 + q + 1, kv_len)
            nib_g = (q < rows 且 member_g的第q位) ? (1 << clamp(vis − 4·block_g, 0, 4)) − 1 : 0
            score_masks[tile, nt, q, quarter] = nib_0 | nib_1 << 4 | nib_2 << 8 | nib_3 << 12   # int16
```

#### 1.2.6 K5 union（[_kernel](sparse_prefill_union.py#L585)、[_body](sparse_prefill_union.py#L219)、[_read_packet_g2s](sparse_prefill_union.py#L80)）

K5是持久kernel：GRID个CTA循环领取`task_order`中的任务，一个任务算完一个(tile, KV头, 128行切片)的attention输出。8个wave合起来是一个128行×64列的分数块：128行是tile的BQ个query × G个头（BQ·G ≤ 128时slices = 1），64列是并集中的16个块（一个N64步）。流水来自一个dense BM128/BN64 D256 attention kernel，helper在sparse_prefill_common.py。

```python
# grid = GRID = min(TASKS, CU)，512线程 = 8 wave，LDS 64KiB（K区32KiB + V区32KiB），每CU一个CTA
for work = blockIdx; work < ceil(TASKS/GRID)·GRID; work += GRID:
    task = task_order[work]                         # −1为补位
    tile = task // (HK·slices)
    if task ≥ 0 and active[tile]:                   # 被K3改走direct的tile跳过
        body(task, STAGGER = (wave ≥ 4))

body(task):
    tile = task // (HK·slices)；hkv = task % HK；slice = (task // HK) % slices
    q0, qvalid, k0, kv_len = meta[tile]；count = counts[tile]；steps = ceil(count/16)   # 并集块数；N64步数
    # wave w的lane对应分数块的行 row = 128·slice + 16w + (lane&15)：query = row // G，head = row % G
    # query ≥ qvalid、query ≥ BQ或head ≥ G的行无效：读到0，最后不写出
    Q：每lane把(query, head)那一行256维中的64维读进寄存器，整个任务常驻（QK的B操作数）
    # 第t个N64步 = 并集第16t..16t+15个块 = 64个token
    DMA K(t)：raw_ptr_buffer_load_lds（buffer_load … lds，M0由编译器设置），每wave 16个包；wave按wave&3分担16个token、按wave>>2分担维度0–127/128–255，
             块号查blocks[tile, 16t ..]（lane 0–3各读一个，readlane进SGPR）；源地址按token做XOR swizzle，K在LDS里免bank冲突；
             slot ≥ count时读到块号0（其分数会被mask）
    QK(t)：S^T[64 token, 128行] = K(t)·Q^T，分lo/hi两次各32个token，每次是K=256维的16条MFMA 16×16×16
           # 结果每lane 16个分数：本lane那一行 × 4个块的16个token，正好对应score_masks的一个16位字
    mask(t)：bits = score_masks[tile, t, query, lane>>4]（buffer_load_ushort）；第i位为0的分数按位选成−inf
    行最大值 = lane内16个取max，再与同行另外3个lane（lane^16、^32、^48）取max
    # softmax（log2域，下文的scale已乘log2(e)）：S = QK·scale，P = exp2(S − m)；m取“行最大值 + 1”，之后只在新的行最大值超过m + 7时
    # 才抬到“新最大值 + 1”（惰性rescale），所以P ≤ 2^7，O和行和l不会溢出，也不必每步rescale

    # 序幕：读第0–2步的块号和mask(0)；DMA K(0) → 等待 → STAGGER组多过一个stage → 读K(0)的lo/hi两半 → S = mask(QK)·scale
    m = max(行最大值(S), −FLT_MAX) + 1          # 下限防止整行被mask时出现−inf − (−inf)
    l = 0；O = 0（每lane 64个FP32：本行256维中的64维）；DMA K(1)
    # 稳态：phase t算“上一步的PV”和“本步的QK”，分为4步，每步先访存stage、再计算stage（stage之间CTA barrier）；一次循环两个phase
    for t in 1 .. steps−1:
        步0 访存：读mask(t)、第t+2步的块号；DMA V(t−1)进V区，16个DMA包之间穿插从LDS读K(t)的前32个token
            计算：S(t)前32个token = QK；P = exp2(S(t−1) − m)
        步1 访存：读K(t)的后32个token；等V(t−1)落地
            计算：S(t)后32个token = QK，加mask(t)；P的lane内行和；P16 = BF16(P)（软件RNE）
        步2 访存：与同行另外3个lane交换行和；DMA K(t+1)进K区，DMA包之间穿插从LDS读V(t−1)的维度0–127
            计算：O[:, 0:128] += V^T·P16^T（A操作数用perm拼成：1维 × 4个token；4个16-token段 × 8条MFMA 16×16×16）；
                  l += 行和；cand = S(t)的lane内最大值
        步3 访存：与同行另外3个lane交换cand；等K(t+1)落地；读V(t−1)的维度128–255
            计算：O[:, 128:256] += V^T·P16^T；new = max(同行4个lane的cand)·scale；S(t) = S(t)·scale
                  m' = new + 1 if new > m + 7 else m
                  if wave内有行m' ≠ m: O ·= exp2(m − m')；l ·= exp2(m − m')   # 未变的行乘1
                  m = m'
    # 收尾
    DMA V(steps−1)（越过4·count的token读0：V若是NaN，P = 0也屏蔽不了）；P = exp2(S − m)；l += P的行和
    O += V^T·BF16(P)^T（前、后128维各一次）
    inv = 1/l if l > 0 else 0
    O·inv → BF16（软件RNE）→ 写进LDS（XOR swizzle）→ barrier → 按行优先读出 → 有效行写out[q0 + query, hkv·G + head, :]
# STAGGER：wave 4–7在序幕比wave 0–3多过一个stage、在收尾少过一个，两组错开一个stage：一组做MFMA时另一组在DMA/读LDS
# stage之间的barrier把DMA写LDS与对应的LDS读隔开：K区在K(t)读完后才被K(t+1)覆盖，V区同理
# KV长度不是4的倍数时不另编变体：块内第k个token的SOFFSET取min(块行偏移, 第kv_len−1−k行)，越过KV末端的部分尾块
# 改读请求最后一个token（K被mask屏蔽、V权重为0）；请求不足k+1个token时上限为0，由buffer范围检查返回0
```

#### 1.2.7 K6 pack（[_pack_body](sparse_prefill_direct_packed.py#L197)、[_pack_block](sparse_prefill_direct_packed.py#L110)）

```python
# grid = ceil(packed_blocks·HK/8)，256线程 = 4 wave；每CTA处理8个(块, KV头)对：
#   线程按tid>>7分两组，每组2个wave（half = (tid>>6)&1 分别处理维度0–127、128–255），每组做4对
# 作用：每次调用把各请求的K/V按4-token块重排进scratch PK/PV，排成K7的MFMA操作数顺序，
#       K7每个lane一次16B读就是操作数，不用再做K的lane间转置和V的字节重排
if gated and direct_flag == 0: return             # 全部tile走union时不打包
for (block, hkv) in 本组的4对（仅block < packed_blocks）:
    start = pack_sources[block]                   # 该块第一个token在原K/V中的行号（请求起点 + 4·块序号）
    token = lane>>4（块内第几个token）；dims = 128·half + 8·(lane&15) + [0, 8)   # 每lane 16B
    K：k = K[start + token, hkv, dims]
       写PK：每块按“64维段 → KV头 → 32维半 → 8维组 → token”顺序存放，每16B是一个token的8维
       # K7中一个lane要的“某token的某8维组”因此连续存放，读出即QK的A操作数
    V：v = V[start + token, hkv, dims]（4个32位字，每字是相邻2维）
       两轮ds_bpermute（与lane^16、lane^32互换字）：把token编号从lane号换到字下标，
       之后每个lane持有同一对维度的4个token
       按维度奇偶重组成[偶数维的token0..3][奇数维的token0..3]（16B）写PV：
       其中每8字节（1维 × 4个token）正是PV MFMA里V^T的一个A操作数
请求最后一块中超出请求长度的token照常拷贝（下一个请求的数据，或越界读到的0）；K7把它们的分数置−inf、V置0
```

#### 1.2.8 K7 packed direct与K7′ raw direct（[packed _kernel](sparse_prefill_direct_packed.py#L574)、[packed _body](sparse_prefill_direct_packed.py#L394)、[raw _kernel](sparse_prefill_direct.py#L585)、[raw _body](sparse_prefill_direct.py#L324)）

```python
# packed，gated（有union计划）：grid (BQ·HK, union tiles)，64线程 = 1 wave；tile = blockIdx.y，local = blockIdx.x // HK，hkv = blockIdx.x % HK
#     只算路由为direct的tile的行：if active[tile] == 0 and local < union_meta[tile].rows: body(union_meta[tile].q_row0 + local, hkv)
# packed，ungated（没有union计划）：grid rows·HK，body(blockIdx // HK, blockIdx % HK)
body(row, hkv)：                                  # 1个wave = 1个query × 1个KV头，同时算该KV头下的G个query头
    base, pos = direct_meta[row]的PK起点与query位置；vis = pos + 1
    C = min(vis//4, T)；count = 4·C + vis%4；steps = ceil(count/32)   # 本行要读的token数（完整块 + 因果尾）
    # 本行第n个token：n < 4C时是PK中第block_indices[row, n//4]块的第n%4个token，否则是因果尾的第n−4C个token
    q：lane的head = lane&15（只用前G个，其余读0），把该头256维中的64维读进寄存器（QK的B操作数）
    cached = block_indices[row, lane]             # 每lane缓存1个块号，64个lane共64块 = 8个BN32步
    K(0)：每步的token分两段各16个；lane按n = 段起点 + (lane&15)算PK地址（块号用ds_bpermute从lane n//4取），
          读该token的64维（lane>>4选8维组，跨4个64维段×2个32维半），即QK的A操作数
    m = −FLT_MAX；l = 0；O = 0（每lane 64个FP32）
    for t in 0 .. steps−1:                       # BN32步：32个token分两段，各16个
        V地址 = 每4个token所在块的PK偏移（用ds_bpermute从K的地址取，不再查块表）；发出段0的V读（前、后128维）
        if (t+1) % 8 == 0: 预取下一组64个块号（下一步起换用）
        等K(t)；S = QK：每段一条K=256维的MFMA 16×16×16链（16条）→ S^T[16 token, 16头]，每lane 4个值
        s = S·scale·log2(e)；n ≥ count的列置−inf
        m' = max(m, 本步32个token上的最大值)     # 同一头的4个lane交换（lane^16、^32、^48）
        α = exp2(m − m')；P = exp2(s − m')；l = l·α + P的和（同样跨4个lane）
        if wave内有lane的α ≠ 1: O ·= α
        算下一步两段K的PK地址；P16 = BF16(P)（RNE）；发出段1的V读
        O[0:128] += V^T·P16^T(段0)；发出K(t+1)段0的读；O[128:256] += …；发出K(t+1)段1的读
        O += V^T·P16^T(段1)（两半）；m = m'          # PV：每段每半8条MFMA 16×16×16（维度×头，K=16个token）
        # 最后一步：n ≥ count的token对应的V元素在MFMA前置0（块里后面的token可能是NaN，不能只靠P=0）
    O ·= (1/l if l > 0 else 0) → BF16（RNE）→ LDS转置（8KiB）→ 有效head（< G）写out[row, hkv·G + head, :]

# raw（不打包时，见1.2.2；直接读原K/V）：grid Σ_r ceil(q_r/4)·HK，256线程 = 4个wave = 同请求连续4个query（各一个wave）
#   gated：direct_flag ≠ 0，且4行中至少一行所在tile不走union才执行；每个wave再查自己那行，走union的行不算也不写
#   LDS 32KiB只用于输出转置
#   每个wave的计算与packed相同，区别在K/V的取法：
#   K：n = 段起点 + (lane>>2)；相邻4个lane读同一token的512B（lane&3选16B，每lane读8次），QK前用ds_bpermute换成A操作数布局
#   V：每lane直接从原V读4个token × 同8维（越过count的token读0），在PV里用perm_b32拼成A操作数（1维 × 4个token）
#   V的地址同样用ds_bpermute从K的地址取，无需查块表
#   softmax每步无条件 O ·= α（不做ballot判断）；块号同样每8步（64块）预取一组，在V读之后发出
```

#### 1.2.9 按slot表读K/V池（kv_slots）

SGLang的K/V在按页分配的token池里。传入`kv_slots`时，kernel直接按池地址读K/V，不再要求调用方先拷成连续K/V；输出与先按slot表拷出再调用逐bit相同。池中一个四token块（请求内token 4j..4j+3）占连续4行、起始行是4的倍数（请求最后不满4个token的块也占整块的4行，多出的行值任意，pack会读到它们），所以块只需查一次表：块首token的池行除以4就是池块号。

```python
# 地址：buffer指令的VOFFSET、SOFFSET与指令偏移相加按32位算（描述符基址48位）。union和pack每次读一个块，
#       给每个块单独建描述符（_buffer_at：64位地址、字节范围），池大小不受这个限制；raw direct一个wave的lane读不同的块，
#       池的字节数超过_MAX_SPAN（4GiB−1MiB）时改用64位global读（WIDE）。
# K3 mask CTA（_masks）：k0, kv_len = meta[tile]（请求在slot表中的起点和长度）
    tails[tile] = slots[k0 + (kv_len−1)//4·4] if kv_len%4 else −1   # 请求最后一个不满块的首行；part 0的lane 0写
    每个N64步lane持有的4个slot：pool_blocks[tile, slot] = slots[k0 + 4·blocks[tile, slot]] >> 2
# K5 union：DMA的块号改读pool_blocks；每条DMA用该块自己的描述符：
    基址 = 池起点 + 池块·4·行字节 + hkv·512B + half     # 64位；行字节 = HK·512B，half = (wave>>2)·256B（该wave的128维）
    NUM_RECORDS = (kv_len%4)·行字节 − half if 池块 == tails>>2 else 4·行字节 − half   # 请求最后一个不满块只含有效行
    SOFFSET = 0，块内第k个token的偏移为k·行字节；越过请求末端的token由范围检查读0（非slot表路径改读最后一个token）
# K6 pack：row = slots[pack_sources[block]]（块首token的池行，wave内一致）；
    K、V各建一个描述符：基址 = 池起点 + (row·HK + hkv)·512B，范围4·HK·512B；
    lane的偏移 = token·HK·512B + 维度·2B，无效的对取范围外（读0）；之后与原来相同。packed K7读PK/PV，不变
# K7′ raw direct：
    slots = 本tile请求的视图（基址slots + k0，范围kv_len·4字节）  # padding块号、越过请求末端的下标都读0
    序幕：读本行全部T个块号（T/64组，每组64个，lane一个）→ 查表、>>2得池块号 → 第0组留作cached，
          其余各组写进本wave的LDS（T·4字节；LDS在结尾输出转置前不用）
    tail_start = slots[visible & −4]                  # 因果尾首token的池行；visible是4的倍数时没有因果尾，读到的值不用（行在请求末端时越界读0）
    循环：(t+1)%8 == 0时从LDS读下一组池块号（非slot表路径在这里读block_indices），其余与非slot表路径相同
    池 ≤ _MAX_SPAN：K/V以池起点为buffer基址，字节偏移按无符号32位算，越过NUM_RECORDS的读返回0
              （范围检查把SOFFSET也算在内，不会回绕进范围，qsa_buffer_range_probe实测）
    池 > _MAX_SPAN（WIDE）：地址 = 池起点 + 池行·行字节（每lane 64位），global_load_dwordx4；越过选择（n ≥ count）的位置
              改读slots[0]（请求第一个token：K的分数被mask，V的权重为0），代替越界读0
# K/V保持3D传入：FlyDSL按int32传张量各维，3D时每维都小于2^31，池可以超过2^31个元素
```

## 2. Indexer prefill

### 2.1 调用链

SGLang在每个QSA层的extend forward中调用一次，各TP rank重复计算，不随TP变化。依次发出下列launch（编号沿用，I3、I6已删除）；每块logits不超过512MiB（行很长时每块至少1024行，1M上下文约1.1GiB），长上下文时I4、I5按块重复。

| # | Kernel（实现） | grid×block | 作用 |
|---|---|---|---|
| I1 | _indexer_q_prep（Triton） | ⌈T/8⌉×256 | q norm/RoPE（与SGLang逐bit一致）、ring写、stats复位 |
| I2 | _indexer_k_compress（Triton） | ⌈groups/16⌉×256 | 4成员组均值→norm/RoPE→压缩池 |
| I4 | qsa_indexer_logits（FlyDSL） | items×256 | MFMA logits（按token_slot_table直读压缩池），顺带每行min/max位 |
| I5 | qsa_indexer_topk（FlyDSL） | ⌈rows/4⌉×256，长行rows×256 | 每wave一行（宽度≥16384时4个wave共一行）：top-512与2051宽token输出；位置与host布局不符时`s_trap` |

### 2.2 伪代码

#### 2.2.0 整体流程

输入qk是`index_qk_proj`的输出：每个token 4个query头加1个key，各128维。key每4个token压成一个压缩key（即一个“块”），选择以块为单位。

1. **I1（每8个token一个program）**：q做RMSNorm与RoPE（与SGLang逐bit一致）；按SGLang给的state_slots写pending ring：每个请求末尾凑不满一组的token（seq_len%4个）写进本请求的ring槽（prefill结束后qk就释放了，decode补齐这一组时只能从ring取），其余token写到不会被读取的dump行；复位每行的logits统计位。
2. **I2（每16个组一个program）**：本次凑满4个token的组求均值后做norm/RoPE，写进压缩池。请求s的第j个压缩key在池中的槽号为`token_slot_table[s, 4j] // 4`。
3. **按块循环**（每块logits不超过512MiB，行很长时每块至少1024行）：
   - I4：每个item（≤128行 × ≤512个key）用MFMA算logits = Σ_头 relu(q_h·k)·scale，key按槽号直接从压缩池读；同时把每行logits的最小/最大位折叠进stats。logits按512个key一块存放（key块主序，行距544个float），一个item写一段连续区域。
   - I5：每行一个wave（宽度≥16384时一个CTA的4个wave共一行），选logits最大的512个块（同值取块号小的），按块号升序展开成每块4个token，再接因果尾token和-1，写成2051宽的一行。行位置与host布局不符时在这里trap。

可见压缩key不超过512个的行不需要logits，I5直接全选；整个tile都不超过512时不建item。

#### 2.2.1 入口（[prefill_indexer](indexer.py#L338)、[_Layout](indexer.py#L234)）

```python
prefill_indexer(qk[T,640], *, heads=4, positions, logical_positions, state_slots, key_state, rope_state,
                write_locs, member_rows, group_sequences, group_ends, rope_matrix, compressed,
                token_slot_table, cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps,
                seq_lens, extend_lens, q_out=None):
    host检查：4头、D128、rotary维度约束、每请求≤262144个压缩key（1M token）、各张量的dtype/形状/设备、压缩池<2GiB
    # group_sequences、group_ends只检查形状，kernel不用
    layout = _layout(seq_lens, extend_lens)      # 按长度LRU缓存8个；numpy规划，一次pinned上传，不同步stream
    q = q_out或empty[T,4,128]；stats = empty[T,2]  # SGLang数值校验传q_out取回q
    I1                                           # q norm/RoPE、ring写、stats复位
    if groups: I2                                # 本次凑满的4-token组写进压缩池
    out = empty[T,2051]
    logits = empty[最大块rows·⌈width/512⌉·544 + 512] FP32
    # key块主序：块内第r行的第j个key在 [(j//512)·rows + r]·544 + j%512（行距544：2KiB行距使12k logits慢22–26%）
    for chunk in layout.chunks:                  # 每块 ≤ 512MiB，或不足1024行时继续加tile（< 2GiB）
        if chunk.items or I4尚未编译: I4 → 本块logits，并折叠stats   # 第一次调用即编译；没有item时launcher不发kernel
        I5 → out[row0 : row0+rows]               # 宽度>65536用int32块号变体，≥16384时4个wave共一行；位置不符时trap
    return out

_Layout(seq_lens, extend_lens):                  # 只由host长度决定
    prefix_r = seq_r - extend_r；compressed_r = seq_r//4       # 请求r的压缩key数
    row_info[row] = (prefix_r + i, seq_r)        # 每行应有的位置与序列长度：I5据此校验位置并确定count
    每请求按128行切tile：bound = min((prefix_r + start + size)//4, compressed_r)   # tile最后一行可见的压缩key数
    分块：依次加入tile，直到本块logits超过512MiB且已有1024行，或达到2GiB（width = 块内最大bound按4取整）
    item：只为bound > 512的tile建（≤512时I5全选，不需要logits），按KC=512切key区间
          → (row0, local0, rows, request, start, end)             # local0为块内的行号
          按(key块, tile)排序：相邻CTA写同一key块里相邻的tile，写出是连续流
```

#### 2.2.2 I1–I2（[_indexer_q_prep](indexer.py#L86)、[_indexer_k_compress](indexer.py#L128)）

```python
# I1、I2、D0的行数、组数和position stride列入do_not_specialize：任何取值都复用同一份编译结果
I1: # grid = ceil(T/8)，256线程；每program 8 token × 4头 = 32行 × 128列
    每个token：stats = (-1, -1)                 # I4用原子umin折叠(min位, ~max位)，所以从全1开始
    x = qk[token, head, :]；partner = 旋转维度的另一半（再从全局读一次）
    q = norm_rope(x, partner)：
        # Gemma RMSNorm：rstd按SGLang DPP的求和顺序（下标位3,2,1,0,4,5,6）分级求和；
        # NeoX RoPE，3轴位置由axis_map选；每步按SGLang的顺序做BF16舍入 → 与SGLang逐bit一致
    key_state[slot] = qk[token, 4, :]；rope_state[slot] = 3轴位置   # 尾部token的slot = req·4 + pos%4，其余token为dump行0–3
I2: # grid = ceil(groups/16)，256线程；每program 16组
    loc = write_locs[g]；first = member_rows[g]  # 组的4个成员是本次qk中从first起的连续4行（prefix按4对齐）
    # loc==0是写计划的填充组：读第0行（与ROCm上的SGLang相同），写到惰性的slot 0
    mean = 4个成员key之和/4（BF16舍入）→ norm_rope（k_weight；位置取首成员的rope_matrix）
    compressed[loc] = key
```

#### 2.2.3 I4 logits（[indexer_logits._kernel](indexer_logits.py#L120)）

```python
# grid = items，256线程 = 4 wave，waves_per_eu = 2（每CU两个CTA）
# 每CTA处理一个item = (row0, local0, rows ≤ 128, request, start, end)：
#   本tile的rows行（在Q/stats中从row0起，在本块logits中从local0起）× 请求request的压缩key区间[start, end)（≤ 512个）
# wave w负责第32w .. 32w+31行
# LDS：2×8KiB（32个key的块，双缓冲）+ 2KiB（本item 512个key在压缩池中的字节偏移）
# 输出：item覆盖一个512-key块，写到本块logits第start/512个key块里的本tile各行（行距544个float），是一段连续区域
① 偏移表：线程t读token_slot_table[request, 4j]（j = start+t、start+t+256；j ≥ end时重复最后一个），
   压缩槽 = 物理槽 // 4，把“压缩槽·256B”写进LDS；barrier
② Q：lane取 column = lane&31（本wave第几行）、half = lane>>5，读本行4个头各64维（维度[64·half, 64·half+64)）进寄存器；
   超出rows的行读最后一行（其结果写出时被丢弃）
③ 第一块：线程t按偏移表把第t//8个key的第2(t%8)、2(t%8)+1个16B拷进LDS，地址做XOR swizzle：
   key c的第j个16B放在 c·256 + (j ^ (c&7))·16，ds_read_b128读写都无bank冲突；barrier
low = 0xFFFFFFFF；high = 0                        # 本lane写出的logits位模式的最小/最大值
for block in [start, end) step 32:
    若还有下一块：把下一块各线程负责的32B读进寄存器，并从偏移表读“下下块”的偏移（提前两块查表）
    从LDS读当前块：lane(column, half)取key column的维度[64·half, 64·half+64)
    每个头：16个k-step的MFMA 32×32×8，S^T[32 key, 32行] += K·Q^T（key为MFMA行、query行为列；
            第s步用维度4s..4s+3与64+4s..64+4s+3，D=128的求和只是换了结合顺序）
    relu（按位：负数和−0变+0）→ 4头相加 → ×scale
    # 每lane得到本行16个logit：key = block + 8j + 4·half + [0, 4)，j = 0..3（4组，每组4个连续key）
    每组写16B到该key块的第local0 + 32w + column行、第key − start列；组起点 ≥ end或行超出rows的写被丢弃（越界偏移）
    # 跨过end的最后一组会多写≤3列，这些列不会被任何行当作因果logit读取
    logits ≥ 0，无符号位序即数值序：用umin/umax把写出值折进low/high（NaN的位模式大于+inf）
    若还有下一块：把寄存器里的下一块写进另一个LDS缓冲；barrier
两半lane（half 0、1持有同一行）合并low/high；half 0的lane：atomic_umin(stats[row].low, low)、atomic_umin(stats[row].not_high, ~high)
# 同一行可能被多个item（不同key区间）写，原子umin把它们合起来；stats由I1初始化为全1
# tile内各行都算到tile的bound，比本行因果范围多出的key也写出；I5只读本行前count个key，
# stats是超集上的界，直方图分箱仍然单调，选择结果不变
```

#### 2.2.4 I5 top-k与展开（[_row](indexer_topk.py#L686)、[_write_row](indexer_topk.py#L655)、[_select](indexer_topk.py#L603)、[_radix](indexer_topk.py#L519)、[_emit_ascending](indexer_topk.py#L557)）

```python
# grid = ceil(rows/4)，256线程：1个wave = 1行，wave之间互不等待（只用wave内同步）
# 宽度≥16384（SPLIT变体）：grid = rows，CTA的4个wave共一行，像decode一样各取一段连续的组，结果与单wave逐位相同
# 每wave的LDS：257个int直方图（256个bin + 1个不参与统计的sink bin；排名阶段复用为64个候选的(key, id)，
#   归并阶段复用为输出缓冲）+ 512个块号chosen：uint16；宽度>65536时用int32的WIDE变体（scratch扩到2KiB）
expected, seq_len = row_info[row]
if logical_positions[row] ≠ expected: s_trap 2   # 位置与host布局不符，进程中止
vis = expected + 1；count = min(vis//4, seq_len//4)  # 本行可见的压缩key（块）数
# 记x[j]为本行块j的logit（在第j//512个key块里），j ∈ [0, count)
# 扫描方式：每512个连续块为一组，lane读 j = 512g + 64i + lane（i = 0..7，一次发8个合并读）；j ≥ count的值不参与
if count ≤ 512:
    out[0 : 4·count] = 0 .. 4·count−1              # 全选，不读logits
    blocks = count
else:
    low = stats[row].low；high = ~stats[row].not_high；nan = (high的位模式 > +inf)   # I4折叠的位模式
    split = select(x, low, high, nan)               # chosen[0:512] = 选中的块号：两段，各自升序，第二段从split开始
    merge-path归并两段：lane i负责第8i..8i+7个输出，先二分查出两段各取多少，再顺序合并写进LDS
    out[4i : 4i+4] = merged[i]·4 + [0, 1, 2, 3]，i = 0..511   # 每块4个token，16B写出
    blocks = 512
tail = vis//4·4                                    # 因果尾：最后不满4个的token
for c in [4·blocks, 2051):
    o = c − 4·blocks
    out[c] = tail + o if (o < vis%4 and o < 3 and tail + o < seq_len) else −1

select(x, low, high, nan):                          # 选x最大的512个块，同值取块号小的
    if 无NaN、low/high有限、high > low:            # 快速路径：一次直方图
        scale = 256/(high − low)；digit(x) = min(int((x − low)·scale), 255)   # x越大digit越大，所以bin按值划分
        直方图遍：hist[digit(x[j])] += 1（LDS原子加）
        前缀和：找第512大所在的bin，above = 更高bin里的个数，size = 该bin里的个数
        if size ≤ 64:
            收集遍（j升序）：digit > bin的写chosen[0 : above]；digit == bin的存进候选(key = 保序位(x), id = j)
                # 用ballot和mbcnt按lane顺序紧凑写，所以两处都保持j升序
            排名：候选i数出 #{m: key_m > key_i，或key_m == key_i且m < i}；名次 < 512−above的按id升序写chosen[above : 512]
            return above
    # 基数选择：对任意值（含NaN、全相等）都精确
    保序位key(x)：正数置符号位、负数取反，使无符号比较与数值大小一致（NaN排在最大）
    key范围遍：所有key的最大值与最小值异或，最高不同位记为top；高于top的位所有key相同，记为prefix
    need = 512；whole = False
    while top ≥ 0:
        width = min(8, top+1)；shift = top − width + 1
        直方图遍：高位等于prefix的key按 (key >> shift) & (2^width − 1) 计数
        找第need大所在的值bin：above、size；need −= above；prefix的这width位置为bin
        if size == need: whole = True；break          # 这个bin里的key正好全要
        top = shift − 1
    选择遍（j升序）：
        whole时：已确定的高位 ≥ prefix 的全选（恰好512个），按j升序写chosen（只有一段）
        否则：key > prefix 的写chosen[0 : 512−need]；key == prefix 的按j取前need个写chosen[512−need : 512]
    return 512 − need                             # whole时chosen整段已升序，从哪里分段都不影响归并
```

## 3. Indexer decode

### 3.1 调用链

PyHIP在decode只负责indexer，decode attention仍是SGLang原生实现。SGLang在CUDA graph decode和MTP的CUDA graph TARGET_VERIFY中每层调用一次`decode_indexer`（verify时`verify=True`；eager decode/verify走原生路径）；各TP rank重复计算，TP2/4/8相同。

| # | Kernel（实现） | grid×block，LDS | 作用 |
|---|---|---|---|
| D0 | _indexer_decode_prep（Triton） | B×256 | q norm/RoPE、ring写；decode时还按group_locs压缩并写池（与SGLang逐bit一致） |
| D0′ | _indexer_ring_compress（Triton） | B×256 | 仅verify：所有行写完ring后，再按group_locs压缩并写池 |
| D1 | qsa_indexer_decode_logits（FlyDSL） | (splits, B)×256，16KiB | 按页表只读本行的压缩K，MFMA算logits；长页表时顺带写每行min/max |
| D2a | qsa_indexer_decode_topk_hist（FlyDSL） | (parts, B)×512，8KiB | 仅页表宽度>32768个key：长行分成parts段，各段统计256-bin直方图 |
| D2b | qsa_indexer_decode_topk_gather（FlyDSL） | (parts, B)×512，约20KiB | 同上：找第512大所在的bin，各段写出更高bin的块和该bin的候选 |
| D2 | qsa_indexer_decode_topk（FlyDSL） | B×512，约9.7KiB | 每行一个CTA（8 wave）选top-512并展开token；长行只排名D2b的候选 |

### 3.2 伪代码

#### 3.2.0 整体流程

decode时每行是一个请求的当前token；MTP verify时一个请求占连续W（≤4）行，即bonus token和草稿token（位置L..L+W−1）。

1. **D0（每行一个program）**：q做norm/RoPE；本token的key和位置写进pending ring；按SGLang的固定形状，把group_locs指向的4个成员求均值并norm/RoPE，写到write_locs（与SGLang逐bit一致）。每步每行都压缩一次：只有本token补满一组（seq_len是4的倍数）时write_locs才是该组的压缩槽，否则是惰性的slot 0，结果丢弃。verify时D0只做q和ring写，压缩由D0′在本次所有行写完ring之后完成，与SGLang先写全部ring、再gather的顺序一致；SGLang的ring每个请求有2×ratio个槽，窗口行不会覆盖该组在窗口之前的成员。
2. **D1（(splits, B)个CTA）**：按页表只读本行的压缩key（每页16个），MFMA算logits；超出本行长度的位置写-inf。
3. **D2（每行一个8-wave CTA）**：与prefill共用选择核心，选512个块并展开成2051宽的token行。页表宽度超过32768个key时，先由D2a、D2b把实际也超过它的行分给多个CTA统计并收集，D2只排名候选、归并和展开。

长度、位置和页表都在设备端读取，grid只由页表宽度决定，所以graph重放时可以原地更新。

#### 3.2.1 入口（[decode_indexer](indexer.py#L417)）

```python
decode_indexer(qk[B,640], *, positions, logical_positions[B], state_slots, key_state, rope_state, write_locs[B],
               group_locs[B,4], compressed, page_table[B,P], lengths[B], cos_sin_cache, axis_map, q_weight,
               k_weight, q_eps, k_eps, seq_lens[B], verify=False, q_out=None, logits_out=None):
    # 每行一个query token，同名参数与prefill含义相同；compressed按页（每页16个压缩key）读，lengths是各行的压缩key数
    host检查：与prefill共用的检查，另要求B、P > 0和各[B]、[B,4]、[B,P]整型张量的dtype/形状/设备
    q = q_out或empty[B,4,128]
    D0                                          # q norm/RoPE、ring写；verify=False时还压缩一个组
    if verify: D0′                              # 所有行的ring写之后，压缩各行的组
    width = 16·P（≤262144，B·width·4 < 2GiB）；logits = logits_out或empty[B·width + 512] FP32；out = empty[B,2051]
    long = width > 32768；long时stats = empty[B,256,2] int32（D0把它置为全1）
    D1                                          # 按页表算本行logits；long时顺带把min/max折进stats
    if long: D2a、D2b                          # 长行分到多个CTA
    D2                                          # 选512个块并展开成token
    return out
# verify=False时各行必须属于不同请求（D0只看得到本行自己的ring写）；verify=True时一个请求可占多行
# 先eager预热再capture，graph重放可原地更新长度与位置；SGLang数值校验传q_out、logits_out取回Q和logits
```

#### 3.2.2 D0 prep与D0′ verify压缩（[_indexer_decode_prep](indexer.py#L154)、[_indexer_ring_compress](indexer.py#L205)）

```python
# grid = B，256线程；每行一个program，复现SGLang CUDA graph decode未融合的prep（BF16 cos/sin，逐bit一致）
q = norm_rope(qk[row, 0:4])                     # 4个query头
slot = state_slots[row]：key_state[slot] = qk[row, 4]；rope_state[slot] = 3轴位置   # 本token进pending ring
# 按SGLang的固定形状压缩group_locs指向的4个成员（最早的在前）。SGLang先写ring再gather，
# 所以成员恰是本行自己的槽时，直接用本token的key（寄存器里的值）
# group_locs[row] = 位置pos−3..pos在ring中的4个槽（位置小于0时按位置0；SGLang的槽号是req·8 + pos%8）；
# write_locs[row] = 本token补满一组时该组的压缩槽，否则是惰性slot 0；graph的padding行属于从不分配的request 0，读写都落在dump行
if verify: return                               # verify不在D0里压缩（COMPRESS是constexpr）
mean = Σ_{g∈group_locs[row]} (g==slot ? 本token的key : key_state[g]) / 4（BF16舍入）
k = norm_rope(mean, k_weight)，位置取首成员（首成员是自己时用本token位置）
compressed[write_locs[row]] = k
# 另外：页表宽度>32768时stat_words = 512，各行把stats[row]的256对置为全1（运行时参数，否则为0，不写）

# D0′：grid = B，256线程，仅verify。D0已把本次所有行的key和位置写进ring（同一请求的W行占W个槽），
# 与SGLang一样全部从ring读成员
mean = Σ_{g∈group_locs[row]} key_state[g] / 4（BF16舍入）
k = norm_rope(mean, k_weight)，位置取rope_state[group_locs[row, 0]]
compressed[write_locs[row]] = k
```

#### 3.2.3 D1 logits（[indexer_decode._kernel](indexer_decode.py#L70)）

```python
# grid = (splits, B)，256线程 = 4 wave，LDS 4×4KiB；splits = max(1, min(ceil(P/4), ceil(8·CU/B)))
# compressed按页看：每页16个压缩key（4KiB）；page_table[row, p]是本行第p页的物理页号
# 静态grid覆盖页表宽度内的任意长度，实际长度在设备端读取（graph安全，无host同步）
length = min(lengths[row], width)；pages = ceil(length/16)
CTA x的wave w处理页 p = 4x+w, 4x+w+step, …（step = 4·splits）；第一页就超出pages的wave直接退出
q：lane取 column = lane&15、group = lane>>4；column < 4时读头column的维度[32·group, 32·group+32)，否则读0
for 本wave的页，每64页一批:
    一次buffer load取这64页的页号（lane i持有第i页的），之后用readlane广播
    for 每页（下一页的4×1KiB合并读已预取在寄存器里）:
        把本页写进本wave的LDS：key c的第j个16B放在 c·256 + (j ^ (c&7))·16（XOR swizzle）
        按MFMA布局读回：lane(column, group)取key column的维度[32·group, 32·group+32)
        # 直接按MFMA布局读全局时，每个四分之一wave要碰16个cache line，带宽约减半，所以经LDS转置
        8次MFMA 16×16×16：S[16行, 16 key] += Q·K^T（4个头为行，其余12行读0；16个key为列）
        lane column（< 16）持有key column的4个头的分数 → relu → 4头相加 → ×scale
        key = 16p + column ≥ length时置−inf；lane < 16各写1个FP32到logits[row, 16p + column]
        FOLD变体（页表宽度>32768）：有效key的位模式并入本wave的(umin, umax)
# FOLD：每个wave结束时把(min位模式, ~max位模式)原子umin进stats[row, (split·4 + wave) % 256]；
#   1M时一行有几千个wave，都原子到同一地址要多约40µs，所以分到256个槽；有效logits ≥ 0，位模式的无符号序即数值序
#   （NaN或−0会被当作NaN，D2退回精确的基数选择）
# 行中[16·pages, width)不写，D2只读[0, length)
```

#### 3.2.4 D2 top-k与展开（[_decode_kernel](indexer_topk.py#L1025)、[_write_row](indexer_topk.py#L655)、[_span](indexer_topk.py#L426)、[_bases](indexer_topk.py#L451)、[_select_long](indexer_topk.py#L992)、[_part_hist](indexer_topk.py#L910)、[_part_gather](indexer_topk.py#L940)、[_parts](indexer_topk.py#L1074)）

```python
# grid = B，512线程 = 8 wave，一行由8个wave分担；LDS：8个直方图（各257个int）+ 每wave 4个归约字 + 64个候选 + 512个块号（uint16；宽度>65536时int32，约9.7/10.7KiB）
# 算法与I5相同（共用_write_row、select、基数选择和归并），结果与单wave逐位相同；差别如下
count = lengths[row]；vis = logical_positions[row] + 1；seq_len = seq_lens[row]   # 不做位置校验
分工：共ngroups = ceil(count/512)组，wave w负责连续的组[w·per, (w+1)·per) ∩ [0, ngroups)，per = ceil(ngroups/8)
low/high/NaN：页表宽度不超过32768时没有stats，先多扫一遍：各wave算自己那段的min/max/NaN，写进LDS归约字，CTA barrier后合并；更宽时读D1折叠的stats
直方图：每wave一个；找阈值时把8个直方图的对应bin相加
写出位置：收集遍和选择遍中，wave w从“编号更小的wave在同一类里的个数之和”开始写（各wave的计数经LDS交换）；
    因为wave按块号顺序分段，拼起来的chosen与单wave扫描的顺序一致
候选排名只由wave 0做；同步用CTA barrier（I5只用wave内同步）
归并：512个线程，每个线程负责1个输出位置；因果尾token与−1同I5

# 长行（页表宽度 > LONG_KEYS = 32768，编译为LONG变体）：
# parts = max(1, min(32, 组数/8, ceil(2·CU/B)))，按宽度和B在host算好（MI308X上B1–B32、65536–262144个key的扫描中最快或差约1µs）
# 一行走多CTA的条件：count > 32768，且stats给出的low/high有限、无NaN、high > low（256-bin快速路径）；否则D2独自选择
low, high = stats[row]的256对归约（umin位模式、umin ~位模式）；digit(x) = min(int((x − low)·256/(high − low)), 255)
D2a(part, row)：组[first, last)均分给8个wave，各自统计直方图；合并后写suffix[row, part, b] = digit ≥ b的个数
D2b(part, row)：
    wave 0：对全部part的suffix求和（每轮8个合并读）→ 第512大所在的bin及其大小size；
            本part之前各part在该bin之上、之内的个数 = kept_before、cand_before
    if size ≤ 64：各wave扫自己的组，digit > bin的块写kept[row, kept_before + …]，digit == bin的(key, id)写cand[row, cand_before + …]
    # 各part按块号顺序分段，kept拼起来升序，与单CTA的收集遍一致
D2（_select_long）：
    同样从suffix算出above、size；size ≤ 64时chosen[0:above] = kept[row]，候选读进LDS，按单CTA的规则排名
    size > 64（阈值bin里的值太多）时，D2b不写，D2独自走基数选择
    归并与展开不变；结果与单CTA逐位相同
```

## 4. 文件、接口与限制

### 4.1 文件

| 文件 | 内容 |
|---|---|
| [attention.py](attention.py) | 公共入口与输入检查、工作区和scratch arena、热路径重放（`_Hot`） |
| [sparse_prefill_common.py](sparse_prefill_common.py) | 平台标定（`Platform`，按gfx型号和CU数选取，`AITER_QSA_PLATFORM`可覆盖字段）与资源策略常量；buffer描述符（`_buffer_at`等）、寄存器固定和流水barrier；union的BM128/BN64 MFMA、softmax和LDS部件 |
| [sparse_prefill_prepare.py](sparse_prefill_prepare.py) | K1–K3：恢复、每tile在LDS中求并集与计数/逐tile提案（K2）、全局路由、任务排序、精确mask；host端`allocate_plan`、`launchers`，按K1 → K2 → K3发出 |
| [sparse_prefill_union.py](sparse_prefill_union.py) | K5 union：M128×N64，按任务顺序的持久grid，两组wave错拍流水 |
| [sparse_prefill_direct_packed.py](sparse_prefill_direct_packed.py) | K6 pack与K7 packed direct，以及packed/raw共用的direct helper |
| [sparse_prefill_direct.py](sparse_prefill_direct.py) | Direct计划（`DirectPlan`、`prepare`）与K7′ raw direct |
| [indexer.py](indexer.py) | Prefill/decode入口与布局，I1、I2、D0、D0′四个Triton kernel |
| [indexer_logits.py](indexer_logits.py) | I4 prefill logits（顺带归约每行min/max） |
| [indexer_topk.py](indexer_topk.py) | I5/D2：prefill与decode共用的top-512选择、升序归并和token展开；decode长行的D2a/D2b |
| [indexer_decode.py](indexer_decode.py) | D1 decode logits（按页表读压缩key，长页表时顺带折叠min/max） |

Attention的kernel文件名和kernel名沿用AITER #6576（`sparse_prefill_*`），公共入口仍是`attention.py`的`attention`（SGLang按此导入）。测试：[sparse_prefill_op_test.py](../../../../../tests/ops/qsa/sparse_prefill_op_test.py)是AITER `op_tests/test_flydsl_qsa_sparse_prefill.py`的副本，只改了import，以及graph用例中让arena增长的布局（改为`chunk_1024_ctx_31024_random`：原布局在PyHIP走raw direct、不打包，scratch不比被capture的布局大）；它依赖AITER（`aiter.test_common`）、pandas和tabulate，与SGLang原生Triton实现和torch参考比较精度与耗时。[test_attention.py](../../../../../tests/ops/qsa/test_attention.py)把其中的功能、常量和graph用例按H12/6/3包成pytest，另加slot表（含4GiB以上的池）、按行数打包和packed K/V超过2GiB等PyHIP用例。

### 4.2 接口

- Attention只接受3D BF16 Q/O `[M,H,256]`、K/V `[N,HK,256]`和int32 `[M,4T+3]`选择（T = 256/512/768/1024，由宽度决定）：行的因果窗口有v个token时，前`4·min(v//4, T)`列恰为min(v//4, T)个不同的完整四token块（块内4个token依次写出，块可乱序），接v%4个因果尾token，其余为−1。local heads共享选择，`H/HK ≤ 16`。只支持gfx942；5D布局已撤销。
- 选择不校验：块数不对、缺尾token或块重复时结果错误（同一stream之后的调用也可能受影响），不报错也不trap。K/V须为有限值：同一tile的行共享union的块，一行选中的非有限值也会影响没选它的行。
- 字节跨度：Q和每个请求的K（V同）都不超过4GiB−1MiB（`_MAX_SPAN`）。kernel的buffer字节偏移按无符号32位算，1MiB余量让越界哨兵加上lane偏移不回绕；K/V按请求起点寻址，整批packed K/V不受此限（HK1时一个请求约8M token）。
- `softmax_scale`是host标量（默认1/16），`scale·log2(e)`须是有限的正FP32（kernel用它乘被mask的−inf分数，为0时得NaN）；`query_lens`/`prefix_lens`是小于2^31的host整数；`out`不得与Q/K/V/indices重叠。
- `kv_slots`（同设备连续int32 `[N]`）给出时，K/V是token池`[P,HK,256]`，第i项是packed上下文第i个token的池行。每个请求的每个四token块（含请求末尾不满的块）须占池中连续4行、起始行是4的倍数（页大小为4的倍数的分页池满足），不满块多出的行值任意；slot值不校验。union和pack按块建buffer描述符，池大小不受限；池的字节数超过`_MAX_SPAN`时raw direct改用64位global读（WIDE，比buffer读慢约0–10%）。结果与按slot表拷出连续K/V再调用逐bit相同。
- Indexer为独立的D128投影、4个Q头和1个K头、ratio4/top512，最多262144个压缩key（1M token）。Prefill输出每行块号升序，可原样交给attention。`prefill_indexer`和`decode_indexer`除`qk`外都是显式关键字参数，同名参数含义相同；可选的`q_out`（decode另有`logits_out`）取回归一化后的index Q（和logits）。
- 不另设校验launch：attention和indexer的入口在launch前只在host检查张量的dtype、形状、设备、对齐和长度，非法时抛`ValueError`（`softmax_scale`传tensor时抛`TypeError`）；prefill top-k发现query位置与host布局不符时执行`s_trap 2`，GPU队列出错，进程以HSA异常（code 0x1016，日志含kernel名）中止。Decode不校验位置。

### 4.3 Graph、scratch与编译

- Graph capture前须在同一stream、同一layout上eager调用一次。工作区只保存layout元数据和计划；scratch来自每个(device, stream)一块按需增长的arena，同stream的各layout共用。Capture时固定当时的缓冲，arena之后增长不影响graph里的指针；同一stream上capture的graph不得彼此并发重放，也不得与该stream上的eager调用并发。
- 精确长度、任务数、grid和容量，K2的LDS大小和dense上限，K3的排序宽度（五档编译在同一kernel里）、移位量和tile行数都是运行时参数；零任务launcher不发GPU kernel。
- 各kernel在一组编译期参数第一次出现时编译，capture中遇到未编译的变体会报错：K1按T；K2按成员位宽、CTA档位和T（BQ≤21、BQ32、BQ42三种）；K3按HK、每tile的128行切片数、排序键是否超过31位、是否全局路由、平台系数、成员位宽和是否有slot表；union按H、HK、BQ、补齐后的组宽、scale和是否有slot表；packed与raw direct在同一组（H、HK、是否gated、scale、T、是否有slot表）第一次用到时一起编译，池超过`_MAX_SPAN`时另编WIDE。FlyDSL缓存为空、每种head数一个新进程时：第一次12k prefill（K1–K3、union、gated pack与packed/raw direct）约14–15秒；不建计划的调用约1.7秒；用到新tile行数的带prefix chunk（H6的16行、H3的32行union）约7–12秒（H12与12k prefill同为10行，不再编译）；第一次传slot表的调用约13秒；WIDE约0.8–0.9秒。持久缓存命中时第一次调用0.02–0.4秒（[数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/cold_compile)）。
- Indexer的I1、I2、D0对行数、组数和position stride不做整数特化，各只编译一份；prefill第一次调用即编译logits（即使该批没有行需要logits），decode的三个kernel在第一次decode调用时编译，verify另需的不压缩D0与D0′在第一次verify调用时编译（SGLang在graph capture前的eager预热中）。top-k在各入口第一次调用时编译全部变体（prefill：uint16/int32块号 × 每wave一行/4个wave共一行；decode：uint16/int32 × 是否长行，另加D2a/D2b和D1的FOLD变体），Triton和FlyDSL缓存都为空时第一次prefill/decode约5.7/6.5秒。
- FlyDSL持久缓存的键只追踪被编译函数的源码，只改嵌套`@flyc.jit`或helper时可能返回旧kernel。修改PyHIP后先清空缓存，或设置新的`FLYDSL_RUNTIME_CACHE_DIR`。以同一模块名导入的包副本（如几个实验变体）共用缓存时，也会互相拿到对方的kernel。

### 4.4 已知限制

- 路由时延模型由MI308X分阶段计时拟合（union中位误差4.4%、p90 13.2%；direct中位误差1.4%），不保证每个布局都选到最快的路径。Union不受最长任务限制时总是全部保留：模型低估raw direct的代价，去掉这一条件后long_high M4096慢37.6%。
- 删除dense后，完整因果前缀prompt 384–2051行比原dense慢0–27%（10-02测，12层合计不超过约0.33ms）；删除候选提升后，少数TP2高共享、P0 4k随机选择和TP4分块布局的attention慢8–21%（10-04测）。真实12k单请求和服务布局不受影响。
- 长行decode logits的耗时随buffer所在显存位置分成快慢两组，相差13–19%。长上下文prefill logits也随buffer位置波动（P245760约±30%）。
- 1M上下文仍很慢：prefill indexer是O(L²)，上下文末尾16384行的chunk每层约46ms（P245760约9.4ms），其中logits约25.6ms（约4.4 TFLOP，约170 TFLOP/s；写出约18GB FP32 logits）、top-k约20.8ms（直方图与收集两遍读约34GB，约1.6TB/s）（16块1024行的profiler合计，[数据](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/chunk_rows_1m/chunk_rows.json)）。线性外推整条1M prompt的indexer每rank约18秒，各TP rank重复计算；logits块最多约1.1GiB。Decode每层B1/B8约50/197µs，B8以logits（176µs）为主。Attention上下文末尾16384行的chunk每层H12全union约3.4ms、全direct约10.2–10.6ms。
- SGLang对带prefix的chunk传该层的K/V池和packed slot表（`qwen_sparse_attn_backend`经`PyHIPAttention.can_read_kv_slots`判断：int32、一维连续、同设备、K/V池连续），不满足时退回`index_select`拷贝。TP2/TP4服务的每层K池约2.49/3.30GiB；TP8在`MEM_FRACTION_STATIC=0.93`时为4.50GiB（同步前4.30GiB），raw direct用WIDE。Raw direct只在上下文超过行数的32倍（或PK+PV超过1GiB）时使用，16384行的chunk为512k以上；合成测试中1M的原地读比拷贝快1.3–1.8ms/层，4.5GiB池的1M分散选择（WIDE）快0.58–1.04ms/层（见benchmark说明第2节）。
- Raw direct的gated grid每4行一个CTA，按行号排列。union与direct的tile以16–32行交替时，raw direct的耗时接近全部行都走direct：TP4、M16384、上下文128k、一半行走direct时为5.02ms（全部走direct为6.00ms），交替窗口≥64行时为2.89–2.91ms；packed direct每行一个CTA，不受影响（2.67–2.73ms）。推测是CTA按编号轮流分给各shader engine，有计算的CTA集中到一部分上，未用计数器确认。按行数打包后16384行的chunk在512k以内不走raw；1M或行数少的chunk仍可能遇到。[32行交替与全direct](../../../../../mytest/mydata/qsa_wide_20261030_01/mixed_profile/tp4_128k.json)、[窗口扫描](../../../../../mytest/mydata/qsa_wide_20261030_01/mixed_profile/tp4_128k_windows.json)
- K2的LDS按每CU两个CTA的32KiB预算分配：tile的最长因果范围超过8184块（BQ42用64位成员时4092块，约32k/16k token）时走按65536块窗口的多遍路径。16k–32k上下文的K2因此比AITER的dense K2慢（热缓存32k H3 117 vs 27µs），K1不再散射dense位图而快（311→202µs），整次调用热缓存时持平或更快（32k H3/H6/H12 −0.6%/−0.5%/−0.03%）；AITER op test自己的计时协议下32k仍慢0.4–1.2%。预算加大到64KiB或成员窗口加到32KiB时每CU只能放一个CTA，256k/1M的K2慢50–80%，未采用。
- 选择不校验（原K1的逐行校验与trap随AITER结构删除）：上游选择出错时只得到错误结果，进程不会中止。
- 各变体第一次用到才编译（同AITER，不再有首次调用的整体预编译）：缓存为空时，服务中第一个建union计划的prefill（如12k）约12–15秒，第一个带prefix的chunk（slot表变体）约13秒，H6/H3第一次出现16/32行tile的chunk另需7–12秒，池超过4GiB时的WIDE约1秒，都落在触发它的请求上（10-30冷缓存服务中第一条11888 token请求TP2/TP4为17.1/12.8秒，缓存命中时约1.1秒）。SGLang自带的预热请求很短，只编译不建计划的变体；服务应保留FlyDSL持久缓存（命中时0.02–0.4秒），或在预热中覆盖实际会出现的布局（本页服务协议的11888/12000 token预热覆盖12k prefill，覆盖不到带prefix的chunk）。
- SGLang的`pyhip_backend._supports_decode`已改为按`MAX_COMPRESSED_KEYS`判断页表宽度（10-29），1M的graph decode走PyHIP。模型配置`max_position_embeddings=262144`；服务验收最长测到100000 token的请求（10-29），1M的服务级验收尚未做。
- 服务验收是数值验收（attention `.02/.02`、prep逐bit），不是模型质量或生成文本逐bit一致的验收。

### 4.5 当前数据来源

[benchmark说明](../../../../../benchmarks/qsa/readme.md)第2、3节的数据（attention部分在2026-10-30同步AITER #6576并重新应用之后重测，数据目录[qsa_aiter_sync_20261030_01](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01)）：

- Attention逐kernel与整体：[kernel_perf](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/kernel_perf/summary_ab.json)（两轮，新代码/同步提交/重排前同卡交替）；1M事件计时：[ab_1m](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/ab_1m/attention_ab.json)；按slot表原地读与拷贝的比较：[池为上下文2倍](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/slots/new/slots_bench.json)与[两轮复测](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/slots_abab/summary.json)、[4.5GiB池](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/slots/new_wide/slots_bench.json)与[两轮复测](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/slots_abab_wide/summary.json)。
- AITER op test的精度与耗时：[输出比较](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/outputs/compare.json)、[耗时](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/optest_perf/summary.json)、[热缓存逐kernel](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/case_kernels.json)。
- Prefill indexer与12k decode：[indexer_12k_flydsl040](../../../../../mytest/mydata/qsa_p1p2_20261029_01/indexer_12k_flydsl040/kernels.txt)；1M decode：[decode_1m_final](../../../../../mytest/mydata/qsa_p1p2_20261029_01/decode_1m_final/kernels.txt)；长上下文prefill事件计时：[prefill_long](../../../../../mytest/mydata/qsa_p1p2_20261029_01/prefill_long/config.json)。Indexer源码在同步中未改。
- 首次调用编译时间：[cold_compile](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/cold_compile)；host下发：[host](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/host/summary.json)。
- 服务性能与数值验收（TP2/TP4/TP8）：[qsa_aiter_sync_service_20261030_01](../../../../../mytest/mydata/qsa_aiter_sync_service_20261030_01/summary.json)（计时窗口内的编译缓存检查：[cache_writes.json](../../../../../mytest/mydata/qsa_aiter_sync_service_20261030_01/cache_writes.json)）。
- MTP（EAGLE 3/1/4）服务性能、数值验收与C1 profile：[qsa_mtp_ringfix_20261008_01](../../../../../mytest/mydata/qsa_mtp_ringfix_20261008_01/analysis.json)。

## 5. 优化历史

按时间记录改动、关键结果和证据。数值只代表当时的源码与协议，当前性能以benchmark说明为准。2026-10-04整理前的完整记录（逐例A/B表、探索过程、服务细节）只读保留在[readme快照](../../../../../mytest/mydata/qsa_readme_merge_20261004_01/readme_before.md.snapshot)和[伪代码快照](../../../../../mytest/mydata/qsa_readme_merge_20261004_01/todo_before.md.snapshot)；10-02的逐kernel耗时分析与AP/AS/IP/IS/DP/DS条目见[更早的伪代码快照](../../../../../mytest/mydata/qsa_todo_cleanup_20261004_01/todo_before.md.snapshot)，09-30之前实验目录的README和opt日志见[qsa_docs_layout_20260930_01](../../../../../mytest/mydata/qsa_docs_layout_20260930_01/archive.json)。

测量口径（各条不再重复）：

- A/B：冻结对照与当前源码在同一进程交替测量，原cudaPerf、10 buffers/2 warmup/128 samples，计时输出逐位复核。10-02起为138例矩阵（真实12k、合成、P30k共享、P0、长上下文、小批、ragged和服务布局，各TP2/4/8），分乱序和升序两种行序。真实TP4/8输入是TP2 capture的派生头切片。
- 不建计划的小批受host限制，两臂代码相同时也有约4%的顺序偏差（[对照](../../../../../mytest/mydata/qsa_runtime_k3_20261002_01/ab_control_small/result.json)），这类用例要按对照校正。
- 服务数值验收：TEST=1，TP2/TP4各35请求（含24k分块、32并发和短请求），10-29起另加一条100000 token请求，每个rank的12层都校验attention与prefill indexer。服务性能：原生与PyHIP依次测TP2/TP4×C1/C2/C4/C8，每档32请求、12000输入/350输出，计时时关闭校验（10-29起每个服务计时前先跑一轮不计时的8请求C8）。服务测量期间不得修改被哈希的源码（10-02有一轮因此作废）。
- Graph中前后依赖的空kernel，每个边界约1.86µs（grid不超过47）或2.91µs（grid 640）；rocprofv3给每个kernel多算约2.65µs（[floor_g*.json与trace](../../../../../mytest/mydata/qsa_decode_fusion_20261004_01)）。

### 2026-09-25～10-10：初版到直接接入

- **09-25 单接口**：一个attention调用在内部管理metadata和scratch，分dense因果前缀、跨query union和block-native direct三路（dense于10-02删除）。临时插件完成真实TP2接入和输入捕获。[模型profile](../../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/README.md)
- **09-25～26 Union**：tile 10行、按N64步数排序任务、蛇形分配、4+4 wave错拍流水，真实两层prepared union约4086/3844→2791/2700µs。[证据](../../../../../mytest/mydata/qsa_union_210t_20260925_01/README.md)
- **09-26 Direct**：raw direct让相邻4个lane合并读同一token、在消费端转置K、缓存64个块号、K/V流水，M2048低重合约1428→763µs；`llvm.target_features`改为类型化属性后才真正去掉packed FP32指令（快2.6–3.0%）；按四token预排KV（packed direct）并改为每wave一个query，真实prepared direct约3035→2500µs（含pack）。[流水](../../../../../mytest/mydata/qsa_direct_pipeline_20260926_01/README.md)、[属性](../../../../../mytest/mydata/qsa_direct_packed_20260926_01/README.md)、[pack](../../../../../mytest/mydata/qsa_direct_100t_20260926_01/README.md)
- **09-26～27 路由与pack预算**：packed时union的填充工作量不超过direct的1.7倍才走union（raw/ragged当时仍用rho4），8个真实输入快1.5–13.0%；PK+PV预算64MiB，超出走raw。[路由](../../../../../mytest/mydata/qsa_route_20260926_01/README.md)、[预算](../../../../../mytest/mydata/qsa_pack_limit_20260927_01)、[packed与raw](../../../../../mytest/mydata/qsa_pack_vs_raw_20260927_01/README.md)
- **09-27～28 5D研究（撤销）**：原生SHUFFLE-5D的V以8个token为内层，与四token选择不匹配，union/direct未全面达到3D的水平，按用户要求撤销，只保留3D。[3D恢复验收](../../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/analysis.json)
- **09-28 零spill与准备链**：修复compact和bool归约的private/spill；准备链8→4个launch（完整调用12→8），恢复约159→93µs，真实prep快40–45%、full快3.4–8.7%；实际TP2 TTFT中位数约1037→956ms。[零spill](../../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_analysis.json)、[准备链](../../../../../mytest/mydata/qsa_prepare_latency_20260928_01/analysis.json)、[服务](../../../../../mytest/mydata/qsa_system_20260928_01/final_analysis.json)
- **09-28～29 Prefill indexer**：融合q norm/RoPE、ring写、四token压缩、MFMA ReLU logits和每wave top-512，107个kernel加3次host同步缩为投影GEMM加5个kernel（当时含投影约6.3→0.50–0.78ms）；压缩key上限16384→65536；logits和top-k由HIP改写为FlyDSL，逐bit一致，12k快2–3.5%。[服务](../../../../../mytest/mydata/qsa_indexer_system_20260928_01)、[长上下文](../../../../../mytest/mydata/qsa_indexer_long_20260928_01)、[FlyDSL](../../../../../mytest/mydata/qsa_indexer_flydsl_20260929_01)
- **09-29 Decode indexer与H6 21行tile**：decode logits按实际压缩长度读分页K；一个Triton kernel完成decode prep（原43个kernel），与SGLang逐bit一致；含GEMM的B1/B32完整decode约221.7/2022.7→31.2/43.3µs，TP2 C1 ITL约13.40→10.85ms。H6大单请求tile取21行（MFMA行槽利用96→126），L3 M12000约2.285→1.877ms。[服务](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_03)、[开校验](../../../../../mytest/mydata/qsa_indexer_decode_system_20260929_04)、[TP矩阵](../../../../../mytest/mydata/qsa_tp_kernels_20260929_02/analysis.json)
- **09-29 精度参考与直接接入**：T13的差异来自原生BF16预缩放参考自身超出`.02`，改用独立FP32全元素参考，阈值不变；SGLang改为直接依赖安装的PyHIP，移除临时插件。[T13](../../../../../mytest/mydata/qsa_t13_20260929_02/delivery.json)、[接入](../../../../../mytest/mydata/qsa_native_integration_20260929_01/delivery.json)
- **09-30 JIT长尾**：精确长度、task/grid和mask容量改为运行时参数，用零任务launcher预编译有限的变体；TP2/TP4 C1吞吐35.1/39.8→74.3/81.4 token/s，p99 TTFT 16.7/14.1→0.90/0.70s。[证据](../../../../../mytest/mydata/qsa_jit_latency_20260929_01/delivery.json)
- **09-30 测量协议**：删除旧新逐kernel对照（对084fd8f的对照因GPU门禁失败没有完成，随后撤销）；两个正式benchmark默认输出整体和逐kernel耗时；GPU状态门禁先改为整轮两次、再改为只在入口，最后按用户要求全部删除，只保留正确性和源码哈希检查。服务矩阵扩到C1/C2/C4/C8。[084fd8f收据](../../../../../mytest/mydata/qsa_refactor_084fd8f_20260930_01/delivery.json)、[服务矩阵](../../../../../mytest/mydata/qsa_benchmark_matrix_20260930_01/analysis/summary.json)、[C8补测](../../../../../mytest/mydata/qsa_default_bench_20260930_01/c8_summary.json)
- **09-30 Decode去SGLang依赖**：删除decode对SGLang fast_topk和token展开的依赖，prefill与decode共享FlyDSL的top-512和展开。服务复测（`a04b9ba`）PyHIP相对原生吞吐TP2 +21.2%～+35.0%、TP4 +24.2%～+32.6%（C1–C8）。[indexer](../../../../../mytest/mydata/qsa_no_sglang_20260930T065525_3523362/audit.json)、[服务](../../../../../mytest/mydata/qsa_service_retest_20260930_01/summary.json)
- **10-02 删除dense、少行只走direct、等分tile**：删除attention_dense.py（766行），完整前缀行与稀疏行一样按tile选择union或direct；总行数少于`max(384, 64·H/HK)`时不建计划；每个请求等分tile，消除2–4行的尾tile（TP2完整前缀M1024 138.5→85.3µs）。Attention的6个文件3137→2374行。代价在完整前缀prompt 384–2051行（比dense慢0–27%），长prompt和长前缀小批持平或变快。[阈值标定](../../../../../mytest/mydata/qsa_simplify_20261001_01/calibrate_balanced/result.json)、[A/B](../../../../../mytest/mydata/qsa_simplify_20261001_01/ab/result.json)
- **10-02 packed覆盖多请求、单一公式与全局路由**：packed direct支持多请求和KV长度非4倍数（各请求按4-token块对齐依次打包）；逐tile只留`16·ceil(U/16) ≤ 1.7·Σceil(selected/32)`一个公式，删除rho4；在K3的排序CTA里加全局路由（时延模型由MI308X分阶段计时拟合，不新增launch）。服务多请求混合布局快8.9–25.3%，raw小批、ragged和非4倍数KV快16–21%；9行小调用慢10–13%（多一个pack launch）。[A/B](../../../../../mytest/mydata/qsa_cd_20261001_01/ab_v4/result.json)、[模型拟合](../../../../../mytest/mydata/qsa_cd_20261001_01/calibrate_phases/result.json)
- **10-02 O4–O13与indexer X1–X5**：
  - O4/X5：host元数据改用numpy一次生成、经pinned内存异步上传，新layout首次调用8.3–10.4→0.67–1.07ms，隐式同步归零。
  - O10：每个(device, stream)一个scratch arena。O12：空的gated分支提前退出。O9：K3每个wave为一个(tile, part)生成mask，mask改存int16。
  - O8：H3大单请求tile取42行（成员位扩为64位），TP8 12k快14–21%。O13：raw direct预取块号。
  - X1：decode top-k改为每行一个8-wave CTA，65536 key B1 194→39µs。X2：logits顺带写每行min/max，top-k省掉一遍扫描（长prefix每chunk省约40µs）。X3：top-k整行升序输出。
  - 合计auto QSA TP2/4/8为−1.1%/−1.2%/−23.2%。[A/B](../../../../../mytest/mydata/qsa_opt_20261001_01/ab_final/result.json)、[benchmark](../../../../../mytest/mydata/qsa_opt_20261001_01/official_attention/kernels.txt)
- **10-02 host下发、logits直读压缩池、统一top-k（AP1、IS2、IS1/DS1）**：热路径直接调用预建参数的Triton/FlyDSL launcher，attention的CPU下发约362→93µs（建计划）、220→69µs（不建计划）；logits按token_slot_table直读压缩池，删除prefix拷贝，prefill indexer快0.2–3.8%；prefill与decode共用一套选择核心。[host](../../../../../mytest/mydata/qsa_ap1_is_20261001_01/host_final.json)、[indexer A/B](../../../../../mytest/mydata/qsa_ap1_is_20261001_01/indexer_ab_is2f/indexer_kernels.json)
- **10-02 准备链简化**：删除升序提示（K1只对块号恰为0,1,2,…的行跳过排序）；行、tile和mask任务一律倒序；K2一遍扫描，累计块数超出预算即停止压缩（long_last TP2的K2 345→279µs）；union每步都加mask，删除公共块路径；同时把阈值改为1.8并删除候选提升。A/B几何平均−1.6%，但真实TP2 12k（几何平均）慢1.4–1.6%。[A/B](../../../../../mytest/mydata/qsa_simplify2_20261001_01/ab_all_v3/result.json)
- **10-02 恢复提升**：按用户要求恢复1.7阈值和候选提升，其余简化保留；路由与冻结对照全部相同，A/B几何平均−1.7%。[A/B](../../../../../mytest/mydata/qsa_simplify2p_20261002_01/ab_all/result.json)
- **10-02 K1–K4的减少与融合**：非法行在K1内`s_trap 2`，删除K3的汇总校验和K4的`torch._assert_async`；建计划时4→3个launch，不建计划时3→1个。A/B几何平均−2.9%/−2.7%，M≤256小批快约7.8%。把K2并入K1、mask并入K2的原型输出逐bit一致，但最坏慢187%/5.7%：kernel边界是最便宜的全局同步。[原型](../../../../../mytest/mydata/qsa_fuse_20261002_01)、[A/B](../../../../../mytest/mydata/qsa_fuse_20261002_01/ab_all/result.json)
- **10-02 删除indexer的assert_async**：prefill top-k发现位置不符即trap，每次调用少一个launch，整体294.5→291.5µs。[A/B](../../../../../mytest/mydata/qsa_indexer_trap_20261002_01/indexer_ab/indexer_kernels.json)
- **10-02 benchmark全部重测**：benchmark说明第2、3节改为同一版本的重测；发现长行decode logits随buffer分成快慢两组。冷缓存的服务中K3在测量窗口内编译3–8秒，促成下面两项。[单元](../../../../../mytest/mydata/qsa_perf_final_20261002_01)、[服务](../../../../../mytest/mydata/qsa_service_final_20261002_02/summary.json)
- **10-02 K2/K3参数改为运行时**：K3的排序宽度（128–4096六档编译在一个kernel里）、移位量和tile行数改为运行时值，warps固定为4；K2的扫描宽度在三档中运行时选择。K3变体19→2，输出逐位相同；TP2的K3 32.8→38.4µs（2048档由8 warps改为4），整体持平。[编译](../../../../../mytest/mydata/qsa_runtime_k3_20261002_01/variants_live2/result.json)、[A/B](../../../../../mytest/mydata/qsa_runtime_k3_20261002_01/ab2_all/result.json)
- **10-02 union单一变体、indexer去特化、首次调用预编译**：union在SOFFSET上钳位越过KV末端的部分尾块，删除TAIL_BOUNDS变体，KV长度非4倍数的请求快约3.4%；I1/I2/D0不做整数特化；每种head形状第一次调用attention时预编译全部变体（见4.3）。服务测量中没有编译，吞吐与上一轮相差不超过±0.4%。[编译清单](../../../../../mytest/mydata/qsa_fresh_compile_20261002_03)、[union A/B](../../../../../mytest/mydata/qsa_compile_once_20261002_01/union_ab_final.log)、[服务](../../../../../mytest/mydata/qsa_compile_once_service_20261002_01/summary.json)
- **10-04 删除候选提升**：比较两种删法后保留逐tile阈值1.7（np17）：真实12k和服务布局路由不变，少数TP2高共享、P0 4k随机选择和TP4分块布局attention慢8–21%（每次forward约多0.7–1.6ms）；改用阈值1.8（np18）会把代价移到真实TP2 12k上。[评估](../../../../../mytest/mydata/qsa_promote_eval_20261003_01/summary.json)、[改动](../../../../../mytest/mydata/qsa_promote_remove_20261003_01)
- **10-04 K3路由改为单一阈值**：全direct只在少数混合批中有约10%收益，且它就是“全部降级”这一档；路由改为一个步数阈值（8档），594个布局的路由和输出与原实现逐位相同，attention_prepare.py 513→489行。服务复测吞吐与10-02相差不超过±0.35%，TTFT/ITL中位数不超过±0.8%。[评估](../../../../../mytest/mydata/qsa_route_simplify_20261004_01)、[改动](../../../../../mytest/mydata/qsa_route_threshold_20261004_01)、[服务](../../../../../mytest/mydata/qsa_route_threshold_service_20261004_01/summary.json)
- **10-06～10-08 MTP的verify走PyHIP**：EAGLE 3/1/4时目标模型每步跑CUDA graph TARGET_VERIFY（每请求4行窗口），不再跑DECODE，PyHIP原先只接管prefill。`decode_forward(verify=True)`中D0只做q和ring写，新增的D0′在本次所有行写完ring后再压缩，按SGLang先写全部ring、再gather的顺序，q、ring和压缩key逐bit一致；SGLang把graph TARGET_VERIFY交给PyHIP。C1 profile中每个verify graph的indexer由约5.67ms降到0.36ms，verify graph在TP2由21.74ms降到16.22ms、TP4由19.65ms降到14.28ms。服务吞吐（MTP原生→MTP+PyHIP）TP2 C1–C8 +15.4%/+24.2%/+30.3%/+33.5%（原生取热缓存复测；首轮原生C8在测量中编译了12次`apply_interleaved_rope_kernel`），TP4 +18.6%/+30.3%/+34.7%/+41.3%；相对10-06只接prefill的PyHIP，TP2快13.4%–21.1%。TEST=1验收TP2/TP4每rank每层217/222次verify校验全部通过。这些数字测于下一条的ring修复之前。[服务](../../../../../mytest/mydata/qsa_mtp_verify_20261008_01/analysis.json)、[基线](../../../../../mytest/mydata/qsa_mtp_baseline_20261006_01/analysis.json)
- **10-08 修复原生verify的ring覆盖**：SGLang原生每个请求的pending ring只有ratio个槽，而verify先写本次所有行、再gather，窗口跨过压缩边界时，边界之后的草稿行会覆盖该组在窗口之前的成员：窗口起点L%4为1/2/3时，该组压缩key与逐token decode的余弦只有0.68/0.45/0.27，PyHIP为了逐bit一致也照样复现。SGLang把ring扩到每个请求2×ratio个槽（`qsa_ring_slots_per_request`，两个槽号builder、graph元数据kernel、pool和验收一起改），修复后4种对齐都与逐token decode逐bit一致；新增的回归测试在旧布局下对齐1–3失败。PyHIP不用改，测试夹具改用同一布局。修复后服务吞吐（MTP原生→MTP+PyHIP）TP2 C1–C8 +17.5%/+23.4%/+27.7%/+29.7%、TP4 +21.5%/+28.8%/+33.8%/+41.6%，16场测量都没有服务中编译；TEST=1验收TP2/TP4每rank每层234/247次verify校验全部通过。[修复前](../../../../../mytest/mydata/qsa_native_verify_ring_20261008_01/result.json)、[修复后](../../../../../mytest/mydata/qsa_native_verify_ring_20261008_02/result.json)、[服务](../../../../../mytest/mydata/qsa_mtp_ringfix_20261008_01/analysis.json)
- **10-08 查明MTP接受长度差**：用64个与服务基准同协议的prompt（ShareGPT首轮平铺到12000 token，前32个就是基准用的），C1贪心逐请求记录，原生、PyHIP、只用PyHIP indexer、只用PyHIP attention各跑两轮。两边输出相同时，接受的草稿逐步一致；同一段文本的teacher-forced对数似然也相同（自然续写上PyHIP每token高0.008，置信区间含0）。差距全部来自生成的文本不同：prefill的微小数值差（PyHIP attention与indexer各自都在容差内）在near-tie处翻转贪心选择，原生自己两轮之间也有45%的prompt输出不同。基准的32个prompt里有2个在PyHIP下稳定走到更难预测的分支（第25个在`<|im_start|>`之后续成user而不是assistant），占约75%的差距；另外32个prompt没有差距，64个prompt两轮合并差1.5%且置信区间含0，原生同配置两轮之间就差2.3%。不是verify或draft的问题，也不是数值退化。ring修复后轨迹又变了，服务基准中两边的接受长度相当（TP2差0.5%以内，TP4 PyHIP反而高0.5%–1.2%）。[研究](../../../../../mytest/mydata/qsa_accept_study_20261008_01/summary.json)
- **10-08 prefill_indexer改为显式参数**：签名从`(qk, **kwargs)`改为显式关键字参数（原`_run`并入，`_run`和`_prefill`删除）。新增可选`q_out`接收归一化后的q；SGLang数值校验靠它取回q，按请求取压缩key的逻辑移到校验方。SGLang用显式关键字绑定一次调用（`functools.partial`），校验经同一绑定运行PyHIP。
- **10-08 decode入口与prefill对齐**：`decode_forward(qk, **kwargs)`改为显式参数并更名`decode_indexer`，与prefill同义的参数同名（`query_positions`→`logical_positions`、`sequence_lengths`→`seq_lens`）；去掉`cache`，直接按16-key页读`compressed`（SGLang的cache本就是它的reshape）。SGLang只用这一个decode入口；只选块的旧`decode_indexer(q, cache, …)`、`_decode_select`、`_decode_forward`和benchmark的select模式（`--keys`）删除，数值校验改用`q_out`、`logits_out`取回Q和logits。原select测试覆盖的边界（511–513个key、非512倍数的528-key页表、65536个key上限）并入decode测试。删除前benchmark说明中的select数据：3000 key B1/B32整体18.720/25.441微秒；长行65536 key B1/B8/B32整体47.640/88.020/217.421微秒，其余见[长行数据](../../../../../mytest/mydata/qsa_perf_final_20261002_01/official_decode_long/kernels.txt)。两项接口改动后的TEST=1服务验收全部通过（GPU0/1被其他会话占用，用户启动脚本复制一份只把设备改为4–7）：普通decode的TP2/TP4每rank 372次attention校验，每层31次prefill、292次graph decode校验；MTP（EAGLE 3/1/4）TP2/TP4每rank每层30/31次prefill、216/248次verify校验，MTP草稿层另有30/31次prefill校验；日志无异常、无设备端断言。[验收](../../../../../mytest/mydata/qsa_api_service_20261008_01/summary.json)
- **10-10 decode入口的host检查**：按PR #57的Copilot review，`decode_indexer`在launch前做与`prefill_indexer`共用的检查（`_check_shared`：头数、D128、rotary维度、qk/positions/ring/压缩池/权重的dtype与形状、压缩池小于2GiB），另检查B、P > 0和各[B]、[B,4]、[B,P]整型张量，非法输入抛`ValueError`，不再走到kernel；`logical_positions`接受int32或int64（SGLang的verify测试传int64）。PyHIP indexer测试32项（新增7项非法输入）、SGLang `test_pyhip_qsa` 3项和benchmark说明里的decode示例均通过。[日志](../../../../../mytest/mydata/pr57_copilot_review_20261010_01)

### 2026-10-29：1M上下文

- **10-29 评估**：客户要求`MAX_COMPRESSED_KEYS`放大到262144（1M token）。HEAD的top-k把块号存成uint16，超过65536个key会截断；attention在1M可以运行，上下文末尾16384行每层H12/H6/H3约11.8/11.4/10.6ms（均匀分散选择，全部direct），K2随上下文线性增长。用“每32行共享448块+各自最近64块”的选择验证union：自动路由全部走union，每例192个抽样行与FP32参考误差≤0.0008，强制union（并集膨胀9.6–31倍）也全部通过；1M时union kernel与256k相同，K2从0.70增到2.55ms（H12）。旧布局下1M末尾chunk每层indexer prefill 176.9ms（logits 115.8、top-k 60.8），是256k的13.4倍：同工作量下logits行跨距1MiB比256KiB慢5.6倍，补256B–4KiB仍慢1.5–3倍；每块256行时top-k只有64个CTA。[indexer](../../../../../mytest/mydata/qsa_1m_eval_20261029_01/result.json)、[attention](../../../../../mytest/mydata/qsa_1m_eval_20261029_02/result.json)、[跨距](../../../../../mytest/mydata/qsa_1m_eval_20261029_04/stride.json)、[union](../../../../../mytest/mydata/qsa_1m_union_20261029_01/result.json)
- **10-29 支持1M**：
  - top-k在宽度超过65536时用int32块号的WIDE变体（chosen与归并缓冲32位，每wave LDS 4KiB），不超过65536仍用原uint16变体；`MAX_COMPRESSED_KEYS = 262144`；decode检查`B·width·4 < 2GiB`。
  - prefill logits改为key块主序：512个key一块，块内行距544个float，item按(key块, tile)排序，一个item写一段连续区域；行距512时12k logits慢22–26%。
  - prefill top-k在宽度≥16384时由一个CTA的4个wave共一行（与decode相同的分段，结果逐位相同）；短行仍每wave一行（12k全部共行时top-k慢62%）。
  - logits每块预算256→512MiB；行很长时每块至少1024行（1M约1.1GiB，小于2GiB）。
  - attention：整批packed K/V达到2GiB时按连续请求分组调用（SGLang只检查本块K/V，原先会抛错中止服务）；K1/K2的dense偏移改为64位。
  - 事件计时整次调用（每层，冻结HEAD→当前）：12k 423.7→423.8µs，3请求混合634.8→595.6µs，P61440 M4096 873.9→772.0µs，P122880 M8192 3105→2476µs，P245760 M16384 13.31→9.59ms；1M末尾chunk 176.9→47.0ms。Attention在256k/1M的12个case与HEAD逐bit一致，耗时持平。QSA测试103项通过；缓存为空时第一次prefill/decode编译3.3/1.6→5.7/2.4秒。
  - 官方12k基准重测在噪声内不变：attention TP2/4/8 2523.8/1894.4/1103.2µs，prefill indexer 293.1µs，decode B1/B32 22.2/31.6µs；1M decode B1/B8 142.8/302.6µs。Host下发同机同测：attention不变（约89/56µs），prefill M12000不变（202µs），3请求256.6→202.2µs（logits由2块变1块）。[官方attention](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/official_attention/kernels.txt)、[官方indexer](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/official_indexer/kernels.txt)、[1M decode](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/official_decode_1m/kernels.txt)、[host](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/host.json)
  - torch.profiler在一次调用发出很多chunk时会丢kernel记录：64MiB预算时按kernel求和显示P245760的logits只有0.69ms，而整次调用16.8ms。长上下文的比较一律用事件计时整次调用。[数据](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/config_02/config.json)、[attention](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/attention_ab_final/attention_ab.json)、[测试](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/final_tests.log)
  - 未做：decode top-k按行拆到多个CTA（1M B1的top-k 118µs）；K2改为与上下文无关的并集构造；SGLang`_supports_decode`里写死的65536（SGLang只读，未改）。

### 2026-10-29（续）：SGLang判断、decode长行top-k与K2

- **10-29 SGLang的decode判断**（用户授权改SGLang）：`pyhip_backend._supports_decode`改为实例方法，页表宽度按`self.runtime.MAX_COMPRESSED_KEYS`判断，并要求decode logits的FP32偏移`(rows·width + 512)·4 < 2^31`；新增回归测试`test_graph_decode_with_1m_context_page_table`（1M上下文的graph页表16384页、262144个key，必须走PyHIP，不能悄悄退回原生），原判断下失败、改后通过，`test_pyhip_qsa`共6项通过。测试时发现/opt/aiter已更新到`2544b2927`，而预编译的`module_hipbsolgemm.so`仍是09-23的，缺`_set_current_hip_stream`，SGLang所有`tgemm.mm`都会失败；测试经包装脚本让unquantized linear走Torch（PyHIP与原生校验用同一投影，比较不受影响）。重建AITER之前无法做服务验收。
- **10-29 decode长行top-k分到多个CTA**：页表宽度超过32768个key时编译LONG变体：D1顺带把每个wave的(min位模式, ~max位模式)原子折进每行256个槽（D0按运行时`stat_words`重置；只用1个槽时B1的几千个wave原子到同一地址，多约40µs）；D2a每个(part, row)统计一段的256-bin直方图并写suffix，D2b求第512大所在的bin、按前缀写出各段的块和候选，D2只排名候选、归并、展开，结果与单CTA逐位相同。parts = min(32, 组数/8, ceil(2·CU/B))：在65536–262144个key、B1–B32的扫描中都是最快或只差约1µs（扫描用连续200次图重放，与正式协议的绝对值不同）。官方1M B1/B8 142.8/302.6→50.4/197.3µs（B1的top-k 118.0→36.8µs，三个kernel 10.0+13.0+13.8），12k B1/B32 22.2/31.6→21.9/31.0µs（短页表不走FOLD/LONG路径，持平）；indexer测试51项通过。[1M](../../../../../mytest/mydata/qsa_p1p2_20261029_01/decode_1m_final/kernels.txt)、[12k](../../../../../mytest/mydata/qsa_p1p2_20261029_01/decode_12k_final/kernels.txt)、[parts扫描](../../../../../mytest/mydata/qsa_p1p2_20261029_01/parts_sweep_final.json)、[1M扫描](../../../../../mytest/mydata/qsa_p1p2_20261029_01/parts_sweep_final_1m.json)
- **10-29 K2改为FlyDSL、删除dense位图**：K1不再原子或进tile的dense位图，K2（新文件attention_compact.py）每tile一个CTA直接读各行的`block_indices`：因果范围放得进LDS时每块一个成员字（LDS原子或，再用wave ballot按块号顺序合并写出）；更长时按65536块的窗口做存在位图、前缀序号和成员块。dense scratch（1M时TP2/4/8为1.6/1.0/0.5GiB）和需要保持全0的第二块arena一起删除。12k迭代：首版只有窗口路径，K2为67.1/72.5/91.8µs（TP2/4/8，原Triton 27.5/19.9/17.4）；加LDS dense路径、每线程16个块号读在途、同wave的lane读连续元素、ballot合并写出，并按BQ分256/512线程两档后为27.7/22.1/23.2µs，K1因去掉散射100.9/99.6/101.7→93.5/92.9/92.5µs，K1+K2快7.2/4.5/3.5µs，整体auto QSA 2516.4/1899.5/1100.3µs与之前在噪声内。256k/1M A/B（12例与冻结版逐bit一致）：1M共享选择H12/H6/H3 5757→3421（−40.6%）、3506→1923（−45.1%）、2141→1166µs（−45.5%），均匀分散−14.5%/−11.0%/−6.8%；256k共享−17.5%/−21.5%/−24.7%、分散−7.4%/−5.9%/−4.6%；1M H12共享时K2 2562→433µs、K1 373→141µs。代价：attention建计划时的host下发多0–3.6µs（K2经FlyDSL launcher，原为Triton重放）；首次编译attention 36/39/68→33/37/59秒，decode 2.4→6.5秒（LONG/FOLD与D2a/D2b变体）。attention测试52项、QSA共103项、SGLang 6项通过。[12k](../../../../../mytest/mydata/qsa_p1p2_20261029_01/attention_12k_k2v8/kernels.txt)、[迭代](../../../../../mytest/mydata/qsa_p1p2_20261029_01)、[1M A/B](../../../../../mytest/mydata/qsa_p1p2_20261029_01/k2_ab_v2/attention_ab.json)、[CTA档位](../../../../../mytest/mydata/qsa_p1p2_20261029_01/k2_threads/threads.json)、[host](../../../../../mytest/mydata/qsa_p1p2_20261029_01/host.json)、[编译](../../../../../mytest/mydata/qsa_p1p2_20261029_01/cold_compile/cold_compile.json)
- **10-29 按slot表直读K/V（评估，未改）**：SGLang在chunked prefill时每层先用`get_key_buffer(layer).index_select(0, slots)`（V同样）把整段上下文拷成packed K/V再交给attention。按合成pool（64-token页乱序）测：128k/256k/512k/1M每层0.17/0.34/0.73/1.48ms（约1.5TB/s，连续拷贝1M为1.02ms），临时显存1M时1GiB；整条1M prompt每层64个chunk合计48.1ms，按每rank 12个QSA层约0.58秒。它相当于同chunk attention（1M H12共享3.4ms、分散10.2ms）的15–44%，但远小于1M的TTFT（单indexer外推就约18秒/rank）。改为直读需要SGLang的qwen_sparse_attn_backend传K/V池与slot表，union、direct、pack三个kernel按表取址（page 64下每个4-token块在池中连续），单层池超过2GiB时还要64位或分段寻址；需授权改SGLang的attention后端，本轮未做。[数据](../../../../../mytest/mydata/qsa_p1p2_20261029_01/kv_gather/kv_gather.json)

### 2026-10-29（续2）：按slot表直读K/V、AITER重建与服务环境

- **10-29 按slot表直读K/V**（用户授权改SGLang）：`attention`新增`kv_slots`，K/V可以是SGLang的token池（1.2.9）。池字节偏移按无符号32位计算，池须小于4GiB：`mytest/qsa_buffer_range_probe.py`在gfx942上确认buffer范围检查包含SOFFSET，越过NUM_RECORDS的读返回0且不会回绕（NUM_RECORDS约3.25GiB、SOFFSET ≥ 2^31时仍正确）；SGLang TP2/TP4的每层K池约2.5/3.3GiB。8种布局（多请求、KV长度非4倍数、NaN填充的池，另有6.5M行、页超过2^31字节的池）输出与先按slot表拷贝再调用逐bit相同（`mytest/qsa_slots_check.py`，`--big`）；attention测试新增slot表3项，QSA共121项通过。不传slot表时的CPU下发：建计划的attention每次多约2µs（与同进程冻结包的差平均+2.0µs，各例−1.4–+6.3µs），不建计划的和prefill_indexer持平（[改前](../../../../../mytest/mydata/qsa_p1p2_20261029_01/host.json)、[改后](../../../../../mytest/mydata/qsa_p1p2_20261029_01/host_final.json)）。首次调用的预编译对每种BQ各跑有、无slot表一次，K3、union、pack和raw direct各多一组slotted变体：Triton和FlyDSL缓存都为空时，第一次attention在H12/H6/H3由33/37/59秒增到65/72/124秒（新值在FlyDSL 0.4.0下测，旧值在0.3.2下测，未分开两者），prefill/decode的5.7/6.5秒不变（[数据](../../../../../mytest/mydata/qsa_p1p2_20261029_01/cold_compile_slots/cold_compile.json)）。
  - 首版raw direct取因果尾起始行`slots[k0 + (visible & −4)]`时，visible是4的倍数的末行会读到slot表之后一个元素；slot表的buffer资源没有设范围（NUM_RECORDS = 0xFFFFFFFF），表恰在分配末尾时GPU访问错误。合成256k、恒等slot表时稳定复现，rocgdb停在这条`buffer_load_dword`，资源字s30为0xFFFFFFFF、偏移为表长×4（[rocgdb](../../../../../mytest/mydata/qsa_p1p2_20261029_01/slots_debug/rocgdb_fault.txt)、[复现脚本](../../../../../mytest/mydata/qsa_p1p2_20261029_01/slots_debug/fault_repro.py)）。改为每tile只覆盖本请求的有界视图（基址slots+k0、范围kv_len·4字节），padding块号和末尾下标都读0。
  - 首版raw direct每8步先读块号、下一步再查表；查表后的`>>2`让编译器紧跟一个`s_waitcnt vmcnt(0)`，连同之前发出的V读一起等待（[原ISA](../../../../../mytest/mydata/qsa_p1p2_20261029_01/slots_debug/direct_slotted_before_isa.s)）。改为序幕一次查完512个块号，第1–7组存入本wave的LDS，循环里在plain读块号的位置读LDS：循环体与plain只差这一条读，VGPR 248与plain相同，plain的ISA逐字节不变（[现ISA](../../../../../mytest/mydata/qsa_p1p2_20261029_01/slots_debug/direct_slotted_lds_isa.s)）。
  - 末尾16384行、每层的ABBA事件计时（[汇总](../../../../../mytest/mydata/qsa_p1p2_20261029_01/slots_bench_v2/summary.json)，72组输出全部逐bit相同）：与SGLang原做法（两次`index_select`再调用）相比，union为主的共享选择在64k–1M每层快33–1426µs；64k以内direct先打包整段上下文，与拷贝相当；raw direct（>64k）在页序连续的池上比读连续K/V慢77–329µs，页序随机时128k/256k慢400–831µs，这时比拷贝还慢0.24–0.64ms，512k起都更快。同一个slotted kernel在页序连续的池上只慢约150µs，差额来自池页序；推测是读散在更多2MiB页上超出TLB覆盖，未用计数器确认。计时注意：同一进程里第一个被测的调用方式比后面的慢约200µs（交换顺序时跟着移动），单进程顺序测量会把它算到第一种方式上，所以只用ABBA的数。
  - SGLang：`qwen_sparse_attn_backend`对带prefix的chunk先用`PyHIPAttention.can_read_kv_slots`判断（int32、一维连续、同设备、K/V池连续且小于`MAX_POOL_BYTES`），满足时传该层的K/V池和`packed_kv_slots`，否则仍用`index_select`拷贝；TEST=1校验在参考侧自行按slot表取K/V。`test_pyhip_qsa`记录每次调用是否传了slot表（带prefix的调用必须传），并把`MAX_POOL_BYTES`置0验证拷贝回退与原地读逐bit相同；6项通过。该测试在导入前须同时设`SGLANG_USE_AITER=1`和`SGLANG_USE_PYHIP_QSA=1`（后者在模块导入时读取，测试内override不起作用），否则全部跳过或拿不到PyHIP后端。[日志](../../../../../mytest/mydata/qsa_p1p2_20261029_01/sglang_test_slots_final.log)
- **10-29 AITER重建**（用户授权）：/opt/aiter（`2544b2927`）的Python已更新，`aiter/jit`下130个模块仍是09-23/24构建的旧版（如`module_hipbsolgemm.so`缺`_set_current_hip_stream`，SGLang的GEMM失败）。旧模块整体移到[prebuilt_20260923](../../../../../mytest/mydata/aiter_rebuild_20261029/prebuilt_manifest.txt)，回滚即移回`aiter/jit`；按当前源码重建aiter_core、hipbsolgemm、custom、custom_all_reduce、quant、sample、moe_asm、moe_sorting_opus、gemm_a8w8_bpreshuffle（含cktile）、deepgemm_opus、activation、rmsnorm_quant、cache、top_k_per_row、mhc等（[记录](../../../../../mytest/mydata/aiter_rebuild_20261029/rebuild_standard.json)），CK MoE 2-stage、mha_varlen等由服务首次启动按需构建。`module_gemm_a8w8_bpreshuffle`首次构建因调优表重复项失败：AITER合并调优表时按`us`去重并写回源CSV，删掉了未跟踪文件`model_configs/qwen3_8_flash_next_tp2_decode_a8w8_bpreshuffle_tuned_gemm.csv`中M=4、N=2560、K=3072的cktile行（7.4758µs；保留`qwen38_flash_next_a8w8_bpreshuffle_tuned_gemm.csv`中7.3966µs的ck行），删除的行另存[这里](../../../../../mytest/mydata/aiter_rebuild_20261029/qwen3_8_flash_next_tp2_decode_a8w8_bpreshuffle_tuned_gemm.removed_by_aiter_dedup.csv)。
- **10-29 venv的FlyDSL 0.3.2 → 0.4.0**：AITER `2544b2927`固定`flydsl==0.4.0`，其FlyDSL MoE kernel调用0.3.2没有的`fx.atomic_add`，TP2服务启动即失败（[第一次尝试](../../../../../mytest/mydata/qsa_service_20261029_01/validate_tp2/server.log)）。按10-13恢复0.3.2时的做法，用系统site-packages的同一wheel装进venv，168个文件的哈希与系统安装一致（[记录](../../../../../mytest/mydata/flydsl_040_upgrade_20261029_01/after.txt)；回滚wheel在[flydsl_032_restore_20261013_01](../../../../../mytest/mydata/flydsl_032_restore_20261013_01/before.txt)）。0.4.0下PyHIP QSA测试121项通过（[日志](../../../../../mytest/mydata/qsa_p1p2_20261029_01/pytest_flydsl040.log)），12k attention/indexer的逐kernel与整体耗时和0.3.2相差±1.5%以内（[0.4.0](../../../../../mytest/mydata/qsa_p1p2_20261029_01/attention_12k_flydsl040/kernels.txt)、[0.3.2](../../../../../mytest/mydata/qsa_p1p2_20261029_01/attention_12k_slots/kernels.txt)、[indexer 0.4.0](../../../../../mytest/mydata/qsa_p1p2_20261029_01/indexer_12k_flydsl040/kernels.txt)、[indexer 0.3.2](../../../../../mytest/mydata/qsa_p1p2_20261029_01/indexer_12k_flydsl032/kernels.txt)）。`pip check`仍报amd-aiter的editable元数据（`4752a4086`时生成，固定0.3.2）与0.4.0不符，元数据未重新生成。
- **10-29 服务环境的两处绕行**：venv还登记着10-18以editable安装的`pyhip_monokernel` SGLang插件（benchmarks/monokernel/plugin），其模块已不存在，每次启动打印traceback（不致命，但会触发验收的日志检查，[第二次尝试](../../../../../mytest/mydata/qsa_service_20261029_02/validate_tp2/server.log)因此中止）；服务用`SGLANG_PLUGINS=none`跳过插件。AITER随源码更新的`model_configs/qwen38_flash_next_bf16_tuned_gemm.csv`（已跟踪，本机另有8行本地删除）给BF16 GEMM的hipBLASLt解（solidx 205436/205437等）在本机hipBLASLt里对应FP8+bias的kernel（`Cijk_Ailk_Bjlk_F8NSS_BH_Bias_…`）：原生与PyHIP组、PTPC-FP8与PLE-BF16模型都在启动预热中于QSA indexer的`index_qk_proj`（BF16，`aiter.tuned_gemm.mm` → `hipb_mm`）触发GPU内存访问错误（[服务日志](../../../../../mytest/mydata/qsa_service_20261029_03/validate_tp2/server.log)、[同步launch定位](../../../../../mytest/mydata/qsa_service_20261029_smoke/ptpc_native_blocking/server.log)）。服务改用此前各轮使用的09-25 BF16调优表快照（`AITER_CONFIG_GEMM_BF16`），其余AITER调优表用默认（09-25的FMoE快照格式与当前AITER不兼容）；这样配置的TP2服务能启动并完成12000与24000 token请求（[smoke](../../../../../mytest/mydata/qsa_service_20261029_smoke/ptpc_pyhip_bf16snapshot/result.json)）。
- **10-29 服务验收与复测**：数值验收（TEST=1）TP2/TP4各36请求，在原35请求上加一条100000 token请求（chunk 16384，最后一个chunk带98304 token的prefix），全部通过：每个rank的12层都有带prefix、经slot表读K/V的chunk，attention校验各444次，每层prefill indexer 37次、graph decode 293次，日志无异常。性能（TEST=0，C1–C8各32请求）PyHIP相对原生吞吐TP2 +35.3%/+35.2%/+38.7%/+40.9%、TP4 +39.2%/+42.5%/+46.9%/+53.0%（16场测量中服务日志没有Triton编译或AITER构建，FlyDSL/Triton缓存在计时窗口内没有新写入）。每个服务计时前先跑一轮不计时的8请求C8，这是本轮新加的，没有与旧协议对照。与10-04相比两组都快得多（TP2 C1原生61.1→84.9、PyHIP 75.0→114.9 token/s），这期间SGLang由`c3c4bc8`更新到`4475222`（launcher默认开启PyHIP GR read）、AITER更新并重建、调优表只固定BF16一类，差值不能归于QSA。[验收与性能](../../../../../mytest/mydata/qsa_service_20261029_04/summary.json)、[编译缓存检查](../../../../../mytest/mydata/qsa_service_20261029_04/cache_writes.json)
- **10-30 与HEAD（`edde17b`）逐例对比**：冻结的HEAD包与当前工作区在同一进程、同一输入上调用公开入口（不传slot表），调用前先排一个sleep kernel，CUDA事件只计本次调用的kernel（设备时间），ABBA共40次取中位数（`mytest/qsa_head_ab.py`，FlyDSL 0.4.0，GPU6）。Attention 138例（真实12k、合成12k、P30k四种共享度、P0、小批、ragged、服务布局、长上下文，各TP2/4/8）输出全部逐bit相同，几何平均−2.0%：12k真实与合成−0.2%～−0.6%，P30k的共享选择−2%～−12%，长上下文（131072/245760）−3%～−25%，小批与P0持平；变慢的11例都在+0.7%以内（最大p0 M1024 TP2，约0.5µs）。Prefill indexer 12k持平（291.0→290.6µs），3请求−7.1%，P61440/P122880/P245760为−13.6%/−25.1%/−26.8%；decode 12k B1/B32持平（18.0→18.0、26.5→26.7µs），262144 token的B1/B8为−34.8%/−22.2%；选择全部相同。设备时间之外变差的有三处：建计划的attention每次CPU下发比HEAD多1.5–6µs（不建计划的与prefill_indexer持平，3请求prefill少56µs）；缓存为空时首次调用编译变长（attention 36/39/68→65/72/124秒，prefill 3.3→5.7秒，decode 1.6→6.5秒；HEAD的数在FlyDSL 0.3.2下测）；SGLang改用slot表后，页序随机的128k/256k raw direct比原先拷贝慢0.24–0.64ms/层。[对比数据](../../../../../mytest/mydata/qsa_head_ab_20261030_01/head_ab.json)

### 2026-10-30：K/V改用64位global读的评估（未采用）

- **上限的来源**：buffer资源的基址是48位，但VOFFSET、SOFFSET与指令偏移相加按32位算，所以slot表的池须小于4GiB。structured buffer（idxen，16字节一条记录、32位记录号）也一样：4.5GiB的池里请求页全在4GiB以下时与按slot表拷贝后调用逐bit相同，有页在4GiB以上时256个query全部出错（读到池里的NaN行）。[回绕测试](../../../../../mytest/mydata/qsa_global64_20261030_01/wrap_idx.txt)
- **原型**（只改raw direct；packed direct、union、pack不变）：K/V地址 = 64位基址 + 行号·行字节，用`global_load_dwordx4`；越过选择的位置改读请求第一行（K被mask、V的权重为0），代替buffer越界返回0。H12 gated的ISA：VGPR 248→252、无spill，循环504→521条（VALU 285→292，10条`v_lshl_add_u64`），occupancy不变。FlyDSL的Tensor参数形状按int32传，≥2^31个元素的张量（BF16池4GiB）launch时报`struct.error`，原型只传首行的视图（kernel只用基址）。[源码](../../../../../mytest/mydata/qsa_global64_20261030_01/variant/qsa_g64/attention_direct.py)、[ISA](../../../../../mytest/mydata/qsa_global64_20261030_01/isa/isa_gated_h12.json)
- **结果**：与当前包同进程、同输入，设备时间ABBA 4轮×5次（`mytest/qsa_global64_eval.py`，GPU6），TP2/4/8共30例输出全部逐bit相同。行数≥4096（带宽受限）时raw direct慢4.8–13.0%（几何平均6.8%）：packed K/V的M4096–M16384为4.8–7.0%，slot表128k/256k为5.2–7.6%，1M页序随机6.3–7.1%、页序连续10.9–13.0%；M68小批（时延受限）快0.5–0.8%。4.5GiB的池（一半上下文页在4GiB以上）TP2/4/8都与按slot表拷贝后调用逐bit相同。与10-29同输入的拷贝回退（`index_select`后调用）相比，原地global读在128k/256k慢66–726µs/层、1M快548–1610µs/层。[汇总](../../../../../mytest/mydata/qsa_global64_20261030_01/summary.json)、[逐例](../../../../../mytest/mydata/qsa_global64_20261030_01/full_global/global64_eval.json)
- **代价在哪**（TP2，同协议）：越界位置改读真实行而仍用buffer：±0.3%；同样的32位偏移改用global的SGPR基址形式（只适用于4GiB以内）：慢3.1–4.7%；64位逐lane地址再加约2–3%；行号零扩展把循环VALU降到284，仍慢5.2–6.7%；structured buffer慢2.8–4.4%。代价来自访存指令的路径，不在地址计算的指令数。
- **其它kernel**：`MAX_POOL_BYTES`对整次调用检查，union和pack同样以池起点为buffer基址、块行偏移放在32位的SOFFSET/VOFFSET，也受这个上限。但它们的块基址在wave内一致（union的DMA行经readlane取得，pack每个wave只搬一个块），可以仍用buffer指令：用SALU把“池起点+块行”的64位和写进描述符基址，偏移只剩块内的小偏移；请求不足k+1个token时靠范围检查读0的用法要改为钗位到有效行。raw direct一个wave的lane读不同的块，没有统一基址，只能逐lane 64位地址。未实现；indexer与Q/O不受此限。
- **结论**：TP2/TP4当前的池（2.5/3.3GiB）都在4GiB以内，全部改用global只会让raw direct变慢约5–13%。要支持≥4GiB的池，应只为这种池编译global变体，并同时改union/pack和FlyDSL传参，按池大小选择（之后按此实现，见下一节）。
- **测量陷阱**：各变体都以`qsa_g64`导入、共用一个FlyDSL持久缓存时，缓存键不含被改的helper，之后的变体都拿到第一个变体的kernel：未修改的副本因此“慢5%”，用新缓存后与当前包相同。这几轮数据移到[invalid_shared_cache](../../../../../mytest/mydata/qsa_global64_20261030_01/invalid_shared_cache/README.txt)；有效数据都在`FLYDSL_RUNTIME_ENABLE_CACHE=0`下测，未修改副本作对照为−0.5%～+0.0%。

### 2026-10-30（续）：去掉4GiB池上限、按行数打包（用户授权改SGLang）

- **改动**（按上一节的方案）：
  - union（K5）的slotted DMA每条用该块自己的buffer描述符（1.2.9）：64位基址为“池起点 + 池块·4行 + KV头·512B + 半行偏移”，NUM_RECORDS为4行（请求最后一个不满块为有效行数）减去半行偏移，SOFFSET为0；越过请求末端的token由范围检查读0，代替原来每条DMA前的钳位（s_cmp_eq_u32、s_cselect_b32、s_min_u32）。plain路径不变。
  - pack（K6）的slotted版每个wave按块首池行建64位基址的K、V描述符，lane只用块内偏移。
  - raw direct（K7′）新增WIDE变体：池 ≥ 4GiB时K/V地址 = 64位池起点 + 池行·行字节，用`global_load_dwordx4`，越过选择的位置改读slots[0]（请求第一个token）；池 < 4GiB仍用原buffer变体。首次调用的预编译多跑一遍wide的slot表调用，每种head形状多编译gated/ungated两个WIDE变体。
  - FlyDSL按int32传Tensor尺寸，wide时K/V只传首行的视图（新增`_buffer_at`，按地址建描述符）。
  - 入口：`MAX_POOL_BYTES`（4GiB）换成`MAX_POOL_ROWS` = 2^31（slot表为int32）和`WIDE_POOL_BYTES` = 4GiB（选WIDE）。
  - Direct打包规则：PK+PV ≤ 64MiB，或上下文token数 ≤ 32·行数且PK+PV ≤ 1GiB时打包（`PACK_TOKENS_PER_ROW`、`MAX_ROW_PACKED_KV_BYTES`），其余走raw。依据上一节的预算评估：M16384在128k–512k打包更快，1M和小批更慢。
  - SGLang：`PyHIPAttention.can_read_kv_slots`的池上限改为行数 < `MAX_POOL_ROWS`；`test_pyhip_qsa`改为把`MAX_POOL_ROWS`置0来验证拷贝回退。
- **ISA**（3种head形状首次调用编译的全部FlyDSL变体，[改前](../../../../../mytest/mydata/qsa_wide_20261030_01/isa_baseline/isa.json)、[改后](../../../../../mytest/mydata/qsa_wide_20261030_01/isa_live/isa.json)）：K2 3种、plain的union 5种和pack 2种、全部packed/raw direct 18种（含slotted、非WIDE）与改前逐字节相同；slotted union 5种SGPR 106→101/102、SGPR spill 1→0、指令4976–4993→4753–4770（SALU 1555–1562→1339–1346，VGPR 242不变）；slotted pack 2种VGPR 28→26、SGPR 35→46、SALU 76/78→141/143；WIDE raw direct 6种VGPR 252、无spill。slotted union的asm改动后重跑M0审计（`mytest/m0_clobber_audit.py`；首次调用最后编译的是slotted变体，dump只留同名kernel的最后一个）：H12/H6/H3的slotted union各有64个自写M0的asm DMA和96个读asm输出M0的编译器DMA，0违例。[审计](../../../../../mytest/mydata/m0_clobber_audit_20261030_01/m0_audit.json)
- **正确性**：QSA测试127项通过（新增：4.5GiB池上union+packed、raw、gated raw（H3）、ragged四种布局与按slot表拷贝后调用逐bit相同；强制union时请求最后不满块的页在4GiB以上；按行数打包与raw逐bit相同）；SGLang `test_pyhip_qsa` 6项（17个子测试）通过。[QSA](../../../../../mytest/mydata/qsa_wide_20261030_01/pytest_qsa_all_2.txt)、[SGLang](../../../../../mytest/mydata/qsa_wide_20261030_01/sglang_test_pyhip_qsa.txt)
- **设备时间**（与改前冻结包同进程、同输入，调用前排sleep kernel，ABBA 4轮、每轮3次（138例为5次），FlyDSL缓存关闭，输出全部逐bit相同）：
  - 不传slot表的138例（与10-30 HEAD对比同一组）：几何平均+0.02%，各例−2.6%～+1.4%；只有long_last M16384 TP2（上下文262144，现在打包）快2.6%。[逐例](../../../../../mytest/mydata/qsa_wide_20261030_01/ab_head/head_ab.json)
  - slot表、池 < 4GiB（上下文末尾16384行，64k–1M，共享/分散/混合选择，页序随机/连续，TP2/4/8，90例）：共享选择快0.6–1.8%（union的DMA少了钳位）；分散选择在128k–512k因打包快2.1–11.1%（128k/256k页序随机8.3–11.1%）；混合选择（32行交替共享/分散）在128k–512k快23–46%；64k与1M的分散、混合为−2.3%～+0.4%。[逐例](../../../../../mytest/mydata/qsa_wide_20261030_01/ab_slots/wide_ab.json)
  - 池 ≥ 4GiB（4.5GiB池，上下文页散在整个池里，128k/256k/1M，27例）：改前只能退回`index_select`拷贝再调用，现在原地读快4.9–56.3%：共享选择5.6–56.3%（拷贝占比大），分散4.9–8.6%，混合128k/256k 24–42%、1M约8%。
  - 打包规则（强制raw与强制打包，连续K/V与slot表，M4096/16384，128k–512k，TP2/4/8，各36例逐bit相同）：全部走union时打包只多一个提前返回的pack launch，规则选中的组合慢0.0–1.3%；混合选择时打包快15–46%。混合选择时强制raw的整次调用与全部分散（全走direct）时相近（TP4 M16384 128k：6040/6093µs），强制打包为3762µs。[共享](../../../../../mytest/mydata/qsa_wide_20261030_01/pack_shared/pack_budget.json)、[混合](../../../../../mytest/mydata/qsa_wide_20261030_01/pack_mixed/pack_budget.json)
- **编译**：Triton和FlyDSL缓存都为空时，第一次attention在H12/H6/H3约67/77/122秒（加WIDE前65/72/124秒，差在测量波动内），prefill/decode 5.7/6.5秒不变。[数据](../../../../../mytest/mydata/qsa_wide_20261030_01/cold_compile/cold_compile.json)
- **原地读与拷贝**（`mytest/qsa_slots_bench.py`，协议同10-29，新增`--pool-rows`把上下文的页随机散在固定大小的池里）：池为上下文2倍时72组、4.5GiB池时18组输出全部逐bit相同。128k/256k的分散选择现在打包，原地读比拷贝快125–348µs（10-29页序随机时慢236–642µs）；512k快611–978µs，1M快1.30–1.77ms；共享选择在128k–1M快137–1443µs（H12 128k页序连续一组为−244µs，该组三种调用都比页序随机一组慢约40%），64k以内两边都打包，差值在±130µs内；4.5GiB池上共享/分散选择快137–1413µs，1M的WIDE raw direct比读连续K/V慢4.4–11.6%。同一上下文的连续K/V在页序连续与随机两组间最多差约48%（H12 128k共享），是buffer位置的影响，只比较同一组内的数。[池为上下文2倍](../../../../../mytest/mydata/qsa_wide_20261030_01/slots_bench/slots_bench.json)、[4.5GiB池](../../../../../mytest/mydata/qsa_wide_20261030_01/slots_bench_wide/slots_bench.json)
- **混合路由下raw direct的效率**（`mytest/qsa_mixed_raw_profile.py`，torch.profiler逐kernel，强制raw与强制打包，两边路由相同、输出逐bit相同）：TP4、M16384、128k，32行交替共享/分散时raw direct 5020µs、packed 2670µs，全部分散时6001/5679µs；交替窗口为16行时5026/2679µs，64–8192行时2888–2914/2676–2731µs；4行交替时tile都含两种行、全部走direct，5501/4803µs。即窗口小于64行（16个raw CTA）时，raw direct只算一半行却接近全部行的耗时。推测是CTA按编号轮流分给各shader engine，有计算的CTA集中到一部分上（packed每tile 16个以上CTA，不受影响）；未用计数器确认，也未改。可能的改法：raw grid按编号打散行组，或由K3写出direct行组列表、固定grid按列表取。[32行交替与全direct](../../../../../mytest/mydata/qsa_wide_20261030_01/mixed_profile/tp4_128k.json)、[窗口扫描](../../../../../mytest/mydata/qsa_wide_20261030_01/mixed_profile/tp4_128k_windows.json)
- **服务验收**（TEST=1，`mytest/qsa_service_20261030.py`，36请求含一条100000 token请求）：TP2/TP4默认配置（每层K池2.49/3.30GiB），TP8设`MEM_FRACTION_STATIC=0.93`（实际0.7905，9008768行、4.30GiB，原先这里退回拷贝），三组全部通过：每个rank 12层都有带prefix的chunk，attention校验各444次，每层prefill indexer 37次、graph decode 293次，日志无异常；源码哈希在各组前后不变。[TP2](../../../../../mytest/mydata/qsa_wide_service_20261030_01/validate_tp2/status.json)、[TP4](../../../../../mytest/mydata/qsa_wide_service_20261030_01/validate_tp4/status.json)、[TP8](../../../../../mytest/mydata/qsa_wide_service_20261030_01/validate_tp8w/status.json)
- **服务性能**（TEST=0，C1–C8各32请求，协议同10-29）：验收组之后原定的native_tp2因GPU0被其它进程占用（100%、6.7GB）未能启动（[记录](../../../../../mytest/mydata/qsa_wide_service_20261030_01/native_tp2/status.json)），性能组改用`mytest/qsa_service_20261030b.py`在GPU5、6（TP2）和GPU5、6、7、4（TP4）上另起（源码哈希与验收组相同，缓存复制自验收组结束时）。PyHIP相对原生吞吐TP2 +33.7%/+35.1%/+35.6%/+39.6%、TP4 +40.5%/+43.3%/+47.2%/+52.5%；与10-29同协议相比，PyHIP吞吐−1.6%～+0.2%、原生−2.5%～+1.1%，在服务测量的波动内：12k负载里带prefix的chunk上下文不超过12000，只用到slotted union和pack，不走raw。TP2 C4 PyHIP的TTFT中位数2483→2737ms，但均值2249→2319ms、p90 3060→3071ms，是分布成簇时中位数落点的变化。16场日志没有编译；退出时GPU4上有其它进程忙碌，不能排除TP4计时期间共用GPU4。[汇总](../../../../../mytest/mydata/qsa_wide_service_20261030_02/summary.json)
- **删去`MAX_POOL_ROWS`**（之后）：2^31行的池至少1TiB，不会出现；slot表是int32，kernel只读表中的行，算地址时块偏移和WIDE的行偏移都是64位，这个检查没有作用。入口不再查池行数；SGLang的`can_read_kv_slots`去掉行数条件和对应的import，`test_pyhip_qsa`改为让`can_read_kv_slots`返回False来验证拷贝回退与原地读逐bit相同；PyHIP的输入检查测试去掉行数一项。QSA测试127项、SGLang 6项（17个子测试）通过。[QSA](../../../../../mytest/mydata/qsa_wide_20261030_01/pytest_qsa_all_3.txt)、[SGLang](../../../../../mytest/mydata/qsa_wide_20261030_01/sglang_test_pyhip_qsa_2.txt)
- **attention_compact.py并入attention_prepare.py**（之后）：K2的FlyDSL代码逐字搬到attention_prepare.py（K1之后、K3之前），只把`launcher`改名为`compact_launcher`，原模块docstring改为该段的注释；kernel名仍为`attention_compact`。搬移块与原文件逐字节相同，其余只改了docstring、import和两处调用；全部FlyDSL变体（K2 3种、union 10种、pack 4种、direct 24种）的指令与合并前相同；QSA测试127项、SGLang 6项通过。[合并前副本、ISA与测试](../../../../../mytest/mydata/qsa_merge_compact_20261030_01/isa_after/isa.json)

### 2026-10-30（续2）：同步AITER #6576并重新应用

下文“同步提交”指`14592cb`，“重排前”指同步前的工作区（上一节的代码），“新代码”指同步提交之上重新应用后的代码（下一个提交）。数据都在[qsa_aiter_sync_20261030_01](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01)，脚本为`mytest/qsa_aiter_sync_*`。

- **同步提交**（用户要求）：在`edde17b`上合并ROCm/aiter#6576（AITER `2544b2927`）的attention，生成提交`14592cb`“sync with ROCm/aiter#6576 (2544b2927)”。attention_*.py换成AITER的sparse_prefill_{common,prepare,union,direct,direct_packed}.py（kernel名沿用AITER）；`attention.py`是AITER入口改名为`attention`（SGLang按此导入），`get_gfx_runtime`/`get_cu_num`在common里本地实现，不依赖AITER；indexer改从sparse_prefill_prepare导入`upload`。AITER结构与重排前的主要差别：K1–K3全部是FlyDSL（不用Triton），不校验选择、不trap；K2按dense位图建并集，用一块保持全0的arena；平台标定集中在`Platform`（`AITER_QSA_PLATFORM`可覆盖）；T可为256/512/768/1024（indices宽4T+3）；Q和每个请求的K/V各不超过4GiB−1MiB（无符号偏移，整批不再按2GiB分组）；没有首次调用的整体预编译，各变体第一次用到时编译；union的流水按AITER重写（不用inline asm，P用软件RNE转BF16，running max的下界为−FLT_MAX）。AITER的op test复制为tests/ops/qsa/sparse_prefill_op_test.py（改import），test_attention.py改为它的pytest封装；同步提交上QSA测试68项通过。op test用pandas的`to_markdown`，为此在venv里装了tabulate 0.10.0。
- **重新应用**（同步提交的下一个提交）：同步前的工作区相对`edde17b`的改动备份在[current_snapshot](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/current_snapshot)和[补丁](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/current_vs_edde17b.patch)，按AITER的结构逐项重做：K1不再散射dense位图；K2换成每tile在LDS求并集的窗口版（dense位图和全0 arena删除，arena只剩一块）；K3、union、pack、raw direct的slot表变体与WIDE（1.2.9）；按行数打包（`Platform.pack_tokens_per_row = 32`，`MAX_ROW_PACKED_KV_BYTES` = 1GiB）；入口的`kv_slots`、WIDE判定（池超过`_MAX_SPAN`）和工作区键中的池行数。测试加回slot表（含4GiB以上的池）、按行数打包、2GiB以上packed K/V等用例；indexer（1M支持等）原样保留。不再保留的：首次调用预编译`_warm`（改为AITER的第一次用到时编译）、K1的逐行校验与trap、`_kv_groups`（AITER按请求寻址，不用分组）、torch.profiler下的`record_function`分段（没有调用方）；`pyhip/ops/mha/flydsl/_common.py`恢复为`edde17b`（attention不再引用）。op test的graph用例把让arena增长的布局改为`chunk_1024_ctx_31024_random`：原布局在PyHIP走raw direct、不打包，scratch不比被capture的布局大。SGLang本轮未改。
- **K2的LDS预算**：第一轮验证中AITER op test的32k H3比同步提交慢2.2%：窗口路径要多遍扫描，K2为174µs，AITER的dense K2为27µs。比较的变体（`mytest/qsa_k2_variants.py`，[数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/k2_variants)，输出全部与AITER逐bit相同）：LDS预算加到64KiB，或成员窗口加到32KiB（共约49KiB）时每CU只能放一个CTA，256k/1M的K2慢75–85%（1M共享H12 450→805µs）；窗口偏移改为运行时值时1M共享H12的K2慢4%。采用：最长因果范围放得进32KiB时整段dense，否则LDS取32KiB，其中约16KiB作成员窗口（4088个32位或2044个64位成员），每个窗口只清零实际填入的成员。K2在16k H3 58→32µs，32k H3 174→119µs、H6 88→79µs，256k/1M相差不超过约10µs。
- **正确性**：QSA测试112项通过（attention 61项：op test的功能、常量和graph用例按H12/6/3展开，加PyHIP用例；重排前的127项中被op test覆盖的合成用例删去；indexer 51项不变）；SGLang `test_pyhip_qsa` 6项（17个子测试）通过。test_attention.py里的op test用例原先在当前卡（0号）上运行，`QSA_REPLAY_GPU`指向其它卡时之后的PyHIP用例换卡会被attention拒绝（一个进程只支持一张卡），已改为先按`QSA_REPLAY_GPU`选卡（全部8卡可见、`QSA_REPLAY_GPU=1`、FlyDSL缓存为空时112项通过）；该文件按同步提交的风格用black重排。op test的54组（case × H）输出与同步提交逐bit相同，与FP32参考的最大误差0.0104，`.02`下0个不符。重排前在AITER op test的`large_logits_prefill_768`（常量Q/K，union路径上logits约1e11）H12/6/3输出NaN，新代码与AITER逐bit相同、与参考的最大误差1.1e-5。[输出比较](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/outputs/compare.json)、[pytest](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/pytest_qsa.log)、[pytest（GPU1）](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/pytest_qsa_gpu1.log)、[SGLang](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/sglang_test.log)
- **12k基准**（`benchmarks/qsa/test_attention.py`默认，GPU2，新代码/同步提交/重排前依次各两轮，取两轮中位数的平均）：整体auto QSA TP2/4/8为2474.4/1849.6/1066.0µs（同步提交2474.1/1853.2/1073.7，重排前2518.7/1897.4/1100.2），2051行完整前缀199.2/136.9/134.0µs（同步提交198.5/137.5/133.9，重排前207.0/140.7/135.3）。逐kernel与同步提交相比：K1快5.0–15.2µs（不散射dense位图），K2慢5.3–7.2µs（K2自己读各行块号建并集，AITER由K1散射、K2读位图），K3、union、pack、direct持平；与重排前相比K1 93→76µs（AITER的K1不校验），union快1.2–2.1%。路由三者相同。[汇总](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/kernel_perf/summary_ab.json)
- **AITER op test的耗时**（21个prefill用例 = 7种长度 × H12/6/3，各跑两轮取较小值，GPU3）：新代码相对同步提交几何平均−0.60%（−2.92%～+1.50%），相对重排前−2.30%（−5.15%～+0.22%）；仍比同步提交慢的是32k H12/H6/H3 +0.37%/+0.65%/+1.16%和1024 H3 +1.5%（约1µs）。SGLang原生Triton慢2.9–11.1倍。热缓存逐kernel（`mytest/qsa_aiter_sync_case_kernels.py`，同进程交替）：32k H3 5404.6→5372.4µs（K2 26.9→117.2µs，K1 310.9→202.3µs），H6 6794→6763µs，H12 7969→7967µs，16k H3 2156→2107µs，12k H12 2643→2632µs，1024 H3 317.2→319.5µs。[耗时](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/optest_perf/summary.json)、[逐kernel](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/case_kernels.json)
- **256k/1M**（上下文末尾16384行，共享/分散选择，与冻结的同步提交同进程交替，事件计时整次调用，GPU1）：12例输出逐bit相同，新代码快5.7–26.3%，K1+K2由420–1730µs降到142–573µs。H12/H6/H3的1M共享选择4277→3458、2501→1893、1548→1141µs，分散选择11893→10559、11485→10363、10664→10055µs。[数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/ab_1m/attention_ab.json)
- **slot表**（`mytest/qsa_slots_bench.py`，新代码与重排前按head数各一个进程，GPU5，两轮交替）：池为上下文2倍的72组、4.5GiB池的18组输出全部逐bit相同。原地读新代码相对重排前几何平均−1.60%（4.5GiB池−2.02%，−4.02%～−0.04%），读连续K/V −1.43%（−1.58%），拷贝后调用−0.91%（−1.39%）。最大的+15.2%是H3 128k页序连续一组：新代码进程里这一组的原地读三轮依次约1000、1200、1480µs，拷贝后调用同样变慢，读连续K/V不变（约1035µs）；重排前三轮都约1040µs。同一序列的两个进程都复现，按head数合在一个进程里测（final/slots）时为1000µs，推测是buffer位置的影响，未进一步定位。[2倍池](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/slots_abab/summary.json)、[4.5GiB池](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/slots_abab_wide/summary.json)
- **重排前的一处故障**：重排前的slot表测试在同一进程先测H12、再测H6时，H6第一次调用（`_warm`预编译）在`attention_pack_kv`报GPU内存访问错误，可稳定复现；每种head数单独一个进程时不出现。推断是`_warm`用只有1行的slot池预热slotted pack，而pack按块读4行，越过了分配（未用rocgdb确认）。新代码没有`_warm`；接口要求每个四token块在池中占4行（4.2），文档已写明。[日志](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/slots_b_repro/full_rerun.log)
- **编译**（FlyDSL缓存为空，每种head数一个新进程）：重排前第一次调用由`_warm`编译全部变体，H12/H6/H3约67/77/122秒；新代码各变体第一次用到时编译：12k prefill 14.1–14.8秒，不建计划1.7秒，H6/H3第一个16/32行tile的带prefix chunk 7.4/11.6秒，第一次slot表调用13.2–13.4秒，WIDE 0.8–0.9秒；持久缓存命中时0.02–0.39秒。服务中要靠持久缓存或预热覆盖（4.4）。[数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/cold_compile)
- **host下发**（`mytest/qsa_aiter_sync_host.py`，GPU保持忙，60次中位数；新代码、重排前各一个进程，都以冻结的同步提交为同进程对照）：建计划时新代码95.4–99.1µs，比同步提交多0.5–3.8µs（平均1.9µs），比重排前多约3.4µs；不建计划时57.9–59.6µs，比同步提交少约1.0µs、比重排前多约1.2µs；prefill_indexer约202µs不变。[数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/host/summary.json)
- **服务**（`mytest/qsa_service_20261030c.py`，协议同10-29；TP2用GPU2、3，TP4用GPU2–5，TP8验收用8卡）：数值验收（TEST=1）TP2/TP4/TP8各36请求全部通过：每个rank的12层都有带prefix的chunk，attention校验各444次，每层prefill indexer 37次、graph decode 293次，日志无异常；TP8设`MEM_FRACTION_STATIC=0.93`，每层K池9437696行、4.50GiB（同步前4.30GiB），100000 token请求的最后一个chunk（1696行）raw direct用WIDE变体。FlyDSL缓存从空开始，验收服务的第一条11888 token预热请求在TP2/TP4用了17.1/12.8秒（同步前4.1秒，当时的预编译在启动时完成），缓存命中后约1.1秒。性能（TEST=0，C1–C8各32请求，复用验收服务编好的缓存）：PyHIP相对原生吞吐TP2 +34.2%/+31.9%/+36.7%/+40.3%、TP4 +39.8%/+43.7%/+49.7%/+52.0%；与同步前同协议相比，PyHIP −2.7%～+1.9%、原生−1.2%～+2.4%。差得最多的是TP2 C2的PyHIP（166.5→162.1 token/s）：TP2两组重测为+35.2%/+34.9%/+37.4%/+41.7%（C2 166.7 token/s）；该负载C2的两个prefill chunk（两请求12000+4384行；带4384 token prefix、经slot表读池的7616行）单卡逐例计时（`mytest/qsa_aiter_sync_c2_layouts.py`），新代码与同步提交逐bit相同、持平或快0–3.1%，比重排前快0.9–5.2%。6个性能服务（24场测量）的计时窗口内FlyDSL/Triton缓存没有新写入，日志没有编译。[汇总](../../../../../mytest/mydata/qsa_aiter_sync_service_20261030_01/summary.json)、[TP2重测](../../../../../mytest/mydata/qsa_aiter_sync_service_20261030_01/repeat_tp2.json)、[C2布局](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/final/c2_layouts/summary.json)、[缓存检查](../../../../../mytest/mydata/qsa_aiter_sync_service_20261030_01/cache_writes.json)

### 评估过但未采用

| 方案 | 结论 | 证据 |
|---|---|---|
| Direct 16×4免转置布局（09-26） | 正确且零spill，但慢79–89% | [16×4](../../../../../mytest/mydata/qsa_direct_16x4_pipeline_20260926_01/README.md) |
| Direct MFMA/VALU交织冲160T（09-27） | 收益被VMEM等待抵消，短M更慢 | [160T](../../../../../mytest/mydata/qsa_direct_160t_20260927_01/README.md) |
| 删除raw direct、放开pack预算（10-02） | 256k上下文M64/M512慢71–74%/33–42% | [预算探测](../../../../../mytest/mydata/qsa_cd_20261001_01/pack_probe/result.json) |
| 只走direct或只走union（10-02） | 只走direct在TP2/4/8慢12.0%/49.3%/154%；只走union在TP2慢39.7% | [benchmark](../../../../../mytest/mydata/qsa_simplify2_20261001_01/official_attention/kernels.txt) |
| 删除K3降级（10-02） | 78个混合批几何平均慢11.4%、最坏43.8%；10-04并入单一阈值 | [扫描与A/B](../../../../../mytest/mydata/qsa_demote_20261001_01) |
| union、pack、direct合并为一个launch（10-02） | 省4.6µs，但每个变体多约11.6秒编译，变体数翻倍 | [host](../../../../../mytest/mydata/qsa_ap1_is_20261001_01/host_final.json) |
| raw direct用ballot跳过rescale（10-02） | 慢1.3–2.7% | [A/B](../../../../../mytest/mydata/qsa_opt_20261001_01/ab_o13_rescale_vs_none_fresh/result.json) |
| K1检测严格递增以跳过排序（10-02） | 要多读一遍，TP8真实12k乱序行105.4→119.3µs | [K1变体](../../../../../mytest/mydata/qsa_simplify2_20261001_01/k1_variants_v2.json) |
| K2内用LDS构造并集（O3） | 未做；上限约为long_last TP2中K2/K1所占的4.7%/5.6% | [逐kernel](../../../../../mytest/mydata/qsa_opt_20261001_01/kernels_o13c_long/kernels.json) |
| q_prep调参（X4）、logits KC取1024/2048（X8） | q_prep最多快5%（M12000 36.1→34.3µs）；KC更大反而更慢 | 见整理前快照 |
| K3固定排序宽度、全部8 warps、按最大QB展开（10-02） | 小批排序约6→33µs；mask最多慢49%；TP4慢约16µs且编译31–60秒 | [探索数据](../../../../../mytest/mydata/qsa_runtime_k3_20261002_01) |
| I1只对pending尾部写ring（10-04） | 每次快0.3–0.7µs，与I2对照的臂间噪声同量级，12k prefill约省4–8µs；待定 | [A/B](../../../../../mytest/mydata/qsa_ring_tail_20261004_01/ab/result.json) |
| decode indexer融合D0–D2（10-04） | 每层最多省两个kernel边界（约3.7–5.8µs），约占ITL 0.3–0.6%；暂不建议 | [测量](../../../../../mytest/mydata/qsa_decode_fusion_20261004_01/summary.json) |
| K2按4096宽扫描、dense每tile补32/1056个字（10-29） | 1M时K2不变（H12共享选择2.55→2.63/2.56ms），K2不受单次迭代延迟或通道冲突限制 | [4096宽](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/attention_ab_k2_4096/attention_ab.json)、[补齐](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/attention_ab_pad32/attention_ab.json) |
| logits行距512/528/576/1024、item按tile排序（10-29） | 行距512时12k logits慢22–26%，1024慢50%；528在3请求的top-k慢3%；长上下文各行距随buffer位置波动约±30% | [位置实验](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/placement_02/placement.json) |
| logits块预算64/128MiB（10-29） | 整次调用更慢：P245760 16.8/11.9ms，256MiB为10.6ms | [事件计时](../../../../../mytest/mydata/qsa_1m_impl_20261029_01/chunk_budget_02/chunk_rows.json) |
| K2的CTA档位（10-29） | 只看K2重放，(小BQ, 大BQ)线程(256, 512)为20.4/15.4/16.9µs；(256, 256)、(256, 1024)在H3慢31%/53%（1024线程barrier更多），(128, 512)、(512, 512)在H12/H6慢21–47% | [扫描](../../../../../mytest/mydata/qsa_p1p2_20261029_01/k2_threads/threads.json) |
| K2的64位成员拆成两个32位LDS原子或（10-29） | 无收益，回退 | 迭代时控制台记录 |
| K/V改用64位global读，去掉4GiB池上限（10-30） | 全部改用时raw direct慢4.8–13.0%（几何平均6.8%）；structured buffer同样在4GiB回绕。之后只对≥4GiB的池采用，见10-30（续） | [汇总](../../../../../mytest/mydata/qsa_global64_20261030_01/summary.json) |
| 加大pack预算，整段打包代替raw direct（10-30） | 打包代价随整段上下文增长（均匀分散选择、全走direct，设备时间ABBA，TP2/4/8）：M16384在128k–512k快0.3–2.9%（连续K/V）、4.8–10.8%（slot表、页序随机），1M仍慢8.9–14.3%；M2048/4096在128k更快、256k互有快慢，512k起全更慢；M64–M512慢8.9–493%。120例输出与raw逐bit相同。固定加大预算会拖慢小批；按行数判断（约上下文 ≤ 32·行数时打包）可在128k–512k的大chunk上省几个百分点，之后按此采用，见10-30（续）；pack本身同样受4GiB池的限制（已改为按块建描述符） | [数据](../../../../../mytest/mydata/qsa_pack_budget_20261030_01/run/pack_budget.json) |
| K2的LDS预算64KiB、成员窗口32KiB、窗口偏移改为运行时值（10-30） | 前两种每CU只能放一个CTA：32k H3的K2最快（84/112µs，采用的32KiB为119µs），但256k/1M的K2慢75–85%（1M共享H12 450→805µs）；运行时偏移使1M共享H12的K2慢4%（449→469µs）。输出都与AITER逐bit相同 | [数据](../../../../../mytest/mydata/qsa_aiter_sync_20261030_01/k2_variants) |
