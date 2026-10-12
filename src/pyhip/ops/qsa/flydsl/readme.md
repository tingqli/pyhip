# QSA实现与优化记录

本目录是gfx942 QSA attention与indexer的生产实现，使用方法、测试命令和当前性能见[benchmark说明](../../../../../benchmarks/qsa/readme.md)。第0–3节对应当前源码（2026-10-04，K3路由改为单一步数阈值之后），说明每个kernel的作用、数据流和伪代码，行号链接指向函数定义；第4节是文件、接口、编译、已知限制和当前数据来源；第5节是优化历史。

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

一次调用在当前stream上依次发出下列launch（编号沿用，K4已删除）。不建union计划时只有K1和direct（packed为K6、K7，raw为K7′）。

| # | Kernel（实现） | grid×block，LDS | 作用 |
|---|---|---|---|
| K1 | attention_recover_scatter（Triton） | M×64 | 校验2051宽token ABI，非法行`s_trap`；恢复并排序≤512块写block_indices；有计划时把行位原子或进tile成员位图 |
| K2 | attention_compact（Triton） | tiles×256 | 扫描位图：计数、逐tile提案、压缩块表与成员、清零；program 0复位direct_flag |
| K3 | attention_order_masks（Triton） | (排序CTA+tiles)×256 | 全局路由（步数阈值）、union任务排序与active、精确mask |
| K5 | attention_union_bf16_d256（FlyDSL） | min(tasks, CU或2·CU)×512，64KiB | tile内query共享并集块，M128×N64流水 |
| K6 | attention_pack_kv_bf16_d256（FlyDSL） | ⌈packed块·HK/8⌉×256 | 各请求KV按4-token块重排成MFMA操作数顺序；gated时先读direct_flag |
| K7 | attention_direct_bf16_d256（FlyDSL，packed） | gated为(BQ·HK, tiles)×64，否则rows·HK×64；8KiB | union未接手的行，每wave一个query×一个KV头 |
| K7′ | 同名（raw，PK+PV>64MiB） | Σ⌈rows_r/4⌉·HK×256，32KiB | 同上；4个wave为同请求连续4个query，直接读原KV |

### 1.2 伪代码

#### 1.2.0 K1–K3整体流程

attention把query分成两路计算：union让一个tile的BQ行共享这些行所选块的并集，用M128×N64的MFMA一起算；direct让每行单独读自己选中的块。K1–K3在GPU上为每个tile决定走哪一路，并备好两路要读的表。tile是同一请求内连续的至多BQ行（各tile行数至多差1），BQ在H12/H6/H3时分别为10/16/32（单请求、无prefix、HK=1且行数够多时，H6/H3为21/42），由host在建union计划时切好。总行数少于`max(384, 64·H/HK)`时不建计划，只跑K1，所有行走direct。

1. **K1（每行一个program）**：检查该行的2051宽token选择，非法就`s_trap`，进程中止。合法时把它还原成≤512个块号并升序排序，写入`block_indices`供direct使用。有计划时，还要把本行登记到所属tile的位图`dense`：每个块号b对应一个字`dense[tile, b]`，字的第i位表示tile的第i行选了块b。本行对自己选中的每个块（含因果尾所在的块），把自己那一位原子或进对应的字。例如tile首行是第100行，第103行选了块5，就把`dense[tile, 5]`的第3位置1。
2. **K2（每tile一个program）**：按块号顺序扫描本tile的位图，把非0项压缩成`blocks`（tile内各行所选块的并集，升序）和`membership`（每块被哪些行选中），同时把位图清回0。再比较两种代价：direct为各行BN32步数之和；union为并集块数按16对齐，一个N64步记为16个BN32步。union不超过direct的1.7倍时提案为1（union），否则为0（direct）；累计超过1.7倍就停止压缩。
3. **K3（一次launch，两类CTA并行）**：
   - 排序CTA汇总全部tile的提案和代价，用标定的延迟模型选一个步数阈值L：union只保留步数≤L的提案1 tile，更长的改走direct。只有union受最长任务限制时，才在8档L（全部保留、6档降级、全部不保留）里选模型最快的一档，而且须比全部保留快5%以上。然后写`active[tile]`和`direct_flag`，并把union任务（tile×KV头×128行切片）按N64步数降序、蛇形排进持久grid（`task_order`）。
   - mask CTA为提案1的tile生成`score_masks`：对每个N64步、每个query，标出该query选中且因果可见的token。mask与路由并行生成，被路由改走direct的tile，其mask不会被读取。
4. **之后**：K5 union按`task_order`只处理active的tile；`direct_flag`=1时，K6先把KV重排成direct的操作数顺序，K7再处理非active tile的行。不建计划时，K6/K7无条件执行，K7处理全部行。

三个launch之间的kernel边界就是全局同步点：K2要等tile内所有行写完位图，K3的路由要等所有tile算完代价。在kernel内部做全局同步更慢（见第5节“10-02 K1–K4的减少与融合”）。

| 缓冲区 | 写 | 读 | 内容 |
|---|---|---|---|
| `block_indices[M,512]` | K1 | K7 | 每行升序块号，其余为-1 |
| `dense_membership[tiles,MAXB]` | K1（原子或） | K2（读后清0） | 块b被tile内哪些行选中；两次调用之间保持全0 |
| `blocks`、`membership[tiles,CAP]` | K2 | K3 mask CTA；K5只读`blocks` | tile并集的块号（升序）和成员位 |
| `counts[tiles]`、`costs[tiles,3]` | K2 | K3；K5另读`counts` | 并集块数；(direct步数, 最长行步数, 提案) |
| `active[tiles]`、`direct_flag` | K3（`direct_flag`先由K2清0） | K5、K6、K7 | tile是否走union；是否有行走direct |
| `task_order` | K3 | K5 | union任务的执行顺序 |
| `score_masks[tiles,CAP/16,BQ,4]` | K3 | K5 | 每N64步、每query要保留的token位（int16） |

#### 1.2.1 公共入口（[attention](attention.py#L179)）

```python
attention(q[M,H,256], k/v[N,HK,256], indices[M,2051], query_lens?, prefix_lens?, softmax_scale?, out?):
    # host检查，每次调用都做，不读GPU数据
    scale = 1/16 if softmax_scale is None else float(softmax_scale)      # 有限且>0
    query_lens默认(M,)；prefix_lens默认：单请求(N-M,)，多请求全0；要求Σq==M、Σ(q+p)==N
    Q/K/V：连续BF16、D256、16B对齐、字节跨度<2^31、无grad；H%HK==0且H/HK≤16；gfx942
    indices：同设备连续int32 [M,2051]；out可省略，给出时须同形、对齐、不与Q/K/V/indices重叠
    if M == 0: return out
    with 全局锁, 切到q所在设备:
        每进程只允许一张GPU
        if (device, H, HK, scale)第一次出现且不在capture中:
            _warm(...)                      # 用全因果dummy跑一遍不建计划和该形状可能用到的每种BQ，编译全部kernel变体
        key = (device, stream, query_lens, prefix_lens, H, HK, scale)
        ws = LRU[key]                       # 未capture的最多保留8个；capture过的不淘汰
        if ws 不存在: capture中则报错；否则 ws = _Workspace(...)
        ws.captured |= 正在capture
        ws.attach()                         # 按arena当前的缓冲切好scratch视图（见1.2.2）
        hot = ws.hot[indices是否按16B对齐]  # 该workspace录好的launch序列
        if hot存在 且 没有注册Triton launch hook:
            hot.prepare_call(indices)       # 重放K1 → K2 → K3：跳过Triton的JIT分派，只换K1的indices指针
            hot.compute_call(q, k, v, out)  # 用预先打包好参数的FlyDSL launcher发K5，再发K6+K7（raw为K7′）
        elif 有hook或开着profiler:
            x = ws.bind(q, k, v, indices)   # 常规路径，每个kernel都经过JIT分派，便于被hook/profiler看到
            prepare.run(x, ws.union)                                 # K1 → K2 → K3（不建计划时只有K1）
            if ws.union is not None: union.run(x, ws.union, out)     # K5
            direct.run(x, ws.direct, out)                             # packed：K6+K7；raw：K7′
        else:
            ws.hot[...] = _Hot(ws, q, k, v, indices, out)  # 第一次：按常规路径完成本次调用，同时录下上述launch
    return out
```

#### 1.2.2 工作区、scratch与计划（[_Arena](attention.py#L27)、[_Workspace](attention.py#L57)、[allocate_plan](attention_prepare.py#L368)、[direct.prepare](attention_direct.py#L149)）

```python
class _Workspace:      # 每个新的(device, stream, layout)创建一次
    # R个请求；请求r有q_r个新query、p_r个prefix token，KV长度kv_len_r = q_r + p_r
    numpy生成kv_lens[R]、query_positions[M] = p_r+i、query_sequence_ids[M]，经pinned内存non_blocking上传
    union = allocate_plan(query_tile=32, grid_multiplier=2) if M ≥ max(384, 64·H/HK) else None
    direct = direct.prepare(union)
    specs = block_indices[M,512] + union的scratch + direct的PK/PV
    arena = _arenas[(device, stream)]; arena.reserve(两块的大小)

class _Arena:          # 每个(device, stream)一个
    buffers[0]为通用scratch；buffers[1]为成员位图，调用之间保持全0
    reserve(sizes): 任一块不够 → 重新分配为max(需要, 1.25×原大小)（位图用zeros）
                    → generation+1 → 所有未固定的工作区release()
    # attach()：generation变了就按specs在两块buffer中切256B对齐的视图（bind()先调attach()）
    #          capture中则 pinned = (buffers)，此后不再随arena换缓冲

allocate_plan:
    G = H/HK;  full = (HK==1 and 单请求 and prefix==0)
    BQ = 21 if G==6 and full and M ≥ 21·CU·2           # 126/128个MFMA行
       = 42 if G==3 and full and M ≥ 42·CU·2           # 同上，成员位用64位
       = 否则 min(32, 10 if G==12 else 不超过128/G的最大2的幂)   # G=12/6/3时为10/16/32
    每请求等分为ceil(q_r/BQ)个tile，各tile行数至多差1（大的在前）
    meta[tile] = (q_row0, rows, k0, kv_len, position0)   # 首行在Q中的行号、行数、请求在K/V中的起始行、请求KV长度、首行在请求内的位置
    MAXB = ceil(max_kv/4)；CAP = 16·ceil(min(MAXB, 513·BQ)/16)
    slices = ceil(BQ·G/128)；tasks = tiles·HK·slices；grid = min(tasks, CU·(1 if G==12 else 2))
    route_costs = (HK·slices/min(grid,CU), 0.304·HK/CU, 41 + 0.0041·packed块数·HK)   # µs，MI308X标定
    设备张量：metadata、counts[tiles]、costs[tiles,3]、active[tiles]、task_order、direct_flag、query_tiles[M]
    arena：dense_membership[tiles,MAXB]（i32，BQ>32时i64，全0）、blocks/membership[tiles,CAP]、
           score_masks[tiles,CAP/16,BQ,4] i16

direct.prepare:
    blocks_r = ceil((q_r+p_r)/4)（q_r>0时）；packed = Σblocks_r·4·HK·1KiB ≤ 64MiB
    waves = 1 if packed else 4；每请求每waves行一个direct tile：meta = (q_row0, rows, base, extent, position0)
        packed：base = 4·Σ_{i<r}blocks_i（PK内起点），extent = 4·blocks_r
        raw：base = k_start_r，extent = kv_len_r
    packed时 pack_sources[b] = 所属请求的k_start + 4·(块在请求内的序号)；PK/PV [Σblocks·4, HK, 256]放arena
    gated = (union存在)：复用union的active、query_tiles、direct_flag、metadata、BQ；否则用无关张量占位
```

#### 1.2.3 K1 恢复/校验/散射（[attention_recover_scatter](attention_prepare.py#L79)）

```python
# 每行一个1-wave program（64 lane × 每lane 8列 = 512列，一列对应一个4-token块）
# 从最后一行发起：因果请求里越靠后的行可见块越多，先发最重的行
row = M-1-pid
pos = query_positions[row]              # 该query在请求内的位置（= prefix + i）
L = kv_lens[query_sequence_ids[row]]   # 所在请求的KV长度
vis = pos+1                             # 因果可见的token数
C = min(vis//4, 512)                    # 应选的完整块数：可见块不足512个时全选

# ① 校验2051宽token ABI：前2048列是512组、每组4个连续token（一个块）；
#    因果尾（不满4个token的最后一块）放在第C组，选满512块时放在最后3列
a,b,c,d = indices[row, 4col+0..3]       # 第col组的4个token（按列跨步读）
col < C：ok = a≥0, a%4==0, (b,c,d)==(a+1,a+2,a+3), d<vis, d<L        # 合法的完整块
col ≥ C：必须等于(col==C ? 因果尾的前vis%4个token : -1)              # 尾块放在第C组，其余全-1
error = pos<0 | vis>L | any(!ok) | indices[row,2048:2051] ≠ (C==512 ? 因果尾 : -1)   # 选满512块时尾在最后3列

# ② 还原块号并升序排序：调用方的块可以乱序，排序后direct按固定的升序累加，结果与输入顺序无关；
#    排序后相邻相等即重复块，也算非法
blk = col<C ? (ok ? a/4 : -1) : +inf
if 存在col<C使blk[col]≠col:             # 块号恰为0,1,2,…（完整因果前缀）时已经有序，免排序
    blk = bitonic_sort_512(blk); error |= 存在相邻相等

# ③ 非法行：trap让队列进入错误状态，进程中止（测试传入errors时改为写errors[row]，不trap）
if error: s_trap 2
block_indices[row] = col<C ? blk : -1   # direct（K7）读的每行块表

# ④ 有union计划时登记位图：dense[t, b]的第i位 = tile t的第i行选了块b（每个块号一个字，字宽≥BQ）
#    同一tile的各行在不同program里并发写同一个字，所以用原子或；K2扫非0的字即得并集块表和每块的选中行
if 有计划 and (t := query_tiles[row]) ≥ 0:
    bit = 1 << (row - meta[t].q_row0)                  # 行在tile内的序号；BQ>32时用64位
    for b in blk[:C]: atomic_or(dense[t, b], bit)      # 每行≤512次relaxed原子或
    if vis%4: atomic_or(dense[t, vis//4], bit)         # 不满4个token的尾块也算一个块
```

#### 1.2.4 K2 计数/公式/压缩（[attention_compact](attention_prepare.py#L216)、[_compact_scan](attention_prepare.py#L193)）

```python
# 每tile一个4-warp program，从最后一个tile发起（因果请求里最后的tile块最多）
# program 0顺便把direct_flag清0，K3发现有tile不走union时再置1
rows, pos0 = meta[tile]                         # tile的行数、首行位置
end = min(MAXB, ceil((pos0+rows)/4))            # 最后一行可见的块数：K1只会在[0, end)内置位

# 代价口径：direct按BN32步（每步32个token）计；union每个N64步（16个块=64个token）记为16个BN32步，
# 所以“16·ceil(count/16) ≤ r·direct_tiles”表示union不超过direct的r倍（r = union_ratio = 1.7）
for j in 0..rows−1:                             # tile内第j行
    vis_j = pos0 + j + 1；tokens_j = min(vis_j//4, 512)·4 + vis_j%4；steps_j = ceil(tokens_j/32)
direct_tiles = Σ_j steps_j                       # tile各行单独走direct的步数之和
budget = 1.7·direct_tiles                       # 超过它就走direct，不必再压缩

# 一遍扫描：每轮读W个连续块号的字，W在运行时按MAXB取256（≤256）、512（≤512）或1024，三档编译在同一kernel里；
# 非0的字 = 至少有一行选了该块
count = 0
while 未到end 且 16·ceil(count/16) ≤ budget:
    本轮W个字中非0者按块号顺序追加到blocks[tile, count..]（块号）和membership[tile, count..]（选中它的行位），
    写入位置 = count + 本轮前缀和；并把这些字清0；count += 本轮非0个数
# 超出预算：剩余范围只清0，保证位图在两次调用之间全0；
# 这样的tile走direct，count不完整，之后只读提案为1的tile的count
padded = 16·ceil(count/16)
# 提案：1 = union（扫完且不超过1.7倍）；0 = direct
counts[tile] = count
costs[tile] = (direct_tiles, max_j steps_j, 1 if 扫完 且 padded ≤ budget else 0)
```

#### 1.2.5 K3 路由/排序/mask（[attention_order_masks](attention_prepare.py#L328)、[_route](attention_prepare.py#L256)、[_route_cost](attention_prepare.py#L249)、[_order](attention_prepare.py#L301)、[_masks](attention_prepare.py#L153)）

K3是一次launch里两类互不依赖、并行执行的CTA：

- 排序CTA（`task_order`长度不超过4096时只有1个）：
  1. 全局路由：算出一个步数阈值L，union只保留N64步数不超过L的提案1 tile，更长的改走direct。只有union受最长任务限制时L才会小于最长步数（降级最长的一批，最多全部改走direct），否则保留全部提案1 tile。
  2. 发布结果：`active[tile]`（1 = 走union）；有任一tile走direct时`direct_flag` = 1，K6/K7据此决定是否执行。
  3. 给union任务排序：任务 = (tile, KV头, 128行切片)，按N64步数从长到短排，再蛇形分给持久grid的各CTA，写入`task_order`。
- mask CTA（每tile一个，4个wave各管1/4的N64步）：为提案1的tile生成`score_masks`，即每个N64步、每个query在64个key列里要保留哪些token。mask只依赖K2的输出，不等路由结果；被路由改走direct的tile，其mask不会被K5读取。

输入：K2写的`counts`（并集块数U）、`costs`（direct步数和、最长行步数、提案）、`blocks`/`membership`和`meta`。host参数：TASKS = tiles·HK·slices、GRID、排序宽度SIZE、键移位SHIFT/WIDE、排序CTA数、QB（=BQ）、CAP，以及`allocate_plan`按layout算好的三个时延系数：TASK_SHARE = HK·slices/min(grid, CU)，DIRECT_STEP = 0.304·HK/CU，DIRECT_FIXED = 41 + 0.0041·packed块数·HK。

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
- K5每步每lane只读一个16位字（[_prefetch_mask](attention_union.py#L132)，`buffer_load_ushort`），第i位为0的分数列置为−inf（[_apply_mask](attention_union.py#L149)）。

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
SIZE = clamp(next_pow2(numel), 128, 4096)；order_ctas = ceil(numel/SIZE)
SHIFT = max(1, bit_length(TASKS−1))；WIDE = ((CAP/16) << SHIFT) + TASKS−1 ≥ 2^31      # 键是否要用int64
grid = order_ctas + tiles                         # 每个tile一个mask CTA；所有CTA都是4 warps
# SIZE、SHIFT、QB、CAP、TASKS都是运行时参数：128–4096六档排序编译在同一kernel里，按SIZE选一档

attention_order_masks(pid):
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
    nt0 = part·16 + lane//4；quarter = lane % 4
    for nt in nt0, nt0+64, nt0+128, …（nt < ceil(count/16)）:   # 4个part交错覆盖全部N64步
        slot_g = 16·nt + 2·quarter + {0, 1, 8, 9}[g]           # 该lane在K5里持有的16列 = 4个块
        member_g, block_g = membership[tile, slot_g], blocks[tile, slot_g]   # slot ≥ count视为空，每个slot只读一次
        for q in 0..BQ−1:                         # 运行时循环
            vis = min(pos0 + q + 1, kv_len)
            nib_g = (q < rows 且 member_g的第q位) ? (1 << clamp(vis − 4·block_g, 0, 4)) − 1 : 0
            score_masks[tile, nt, q, quarter] = nib_0 | nib_1 << 4 | nib_2 << 8 | nib_3 << 12   # int16
```

#### 1.2.6 K5 union（[_kernel](attention_union.py#L556)、[_body](attention_union.py#L208)、[phase](attention_union.py#L362)）

K5是持久kernel：GRID个CTA循环领取`task_order`中的任务，一个任务算完一个(tile, KV头, 128行切片)的attention输出。8个wave合起来是一个128行×64列的分数块：128行是tile的BQ个query × G个头（BQ·G ≤ 128时slices = 1），64列是并集中的16个块（一个N64步）。

```python
# grid = GRID = min(TASKS, CU·(1 if G==12 else 2))，512线程 = 8 wave，LDS 64KiB（K区32KiB + V区32KiB），每CU 2个CTA
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
    DMA K(t)：wave按wave&3分担16个token、按wave>>2分担维度0–127/128–255，块号查blocks[tile, 16t ..]；
             源地址按token做XOR swizzle，K在LDS里免bank冲突；slot ≥ count时重复读最后一个块（其分数会被mask）
    QK(t)：S^T[64 token, 128行] = K(t)·Q^T，分lo/hi两次各32个token，每次是K=256维的16条MFMA 16×16×16
           # 结果每lane 16个分数：本lane那一行 × 4个块的16个token，正好对应score_masks的一个16位字
    mask(t)：bits = score_masks[tile, t, query, lane>>4]（buffer_load_ushort）；第i位为0的分数置−inf
    行最大值 = lane内16个取max，再与同行另外3个lane（lane^16、^32、^48）取max
    # softmax：基准m取“行最大值·scale + 1”，之后只在新的行最大值·scale超过m + 7时才抬高（惰性rescale），
    # 所以exp2(s·scale − m) ≤ 2^7，O和行和l不会溢出，也不必每步rescale

    # 序幕：t = 0
    DMA K(0) → 等待 → STAGGER组多过一个barrier → 读K(0)的lo/hi两半 → S = mask(QK)
    m = max(行最大值(S)·scale, −1e30) + 1        # −1e30防止整行被mask时出现−inf − (−inf)
    P = S·scale − m（下一阶段再取exp2）；l = 0；O = 0（每lane 64个FP32：本行256维中的64维）
    DMA K(1)
    # 稳态：每个phase算“上一步的PV”和“本步的QK”
    for t in 1 .. steps−1:
        预取bits(t)，查第t+2步的块号
        DMA V(t−1)进V区，同时从LDS读K(t)的lo半 → lo = QK
        P = exp2(P)                             # 上一步的未归一化概率
        读K(t)的hi半 → hi = QK；S = mask(lo‖hi, bits(t))
        P16 = BF16(P)                           # +0x8000后取高16位（四舍五入）
        DMA K(t+1)进K区，同时从LDS读V(t−1)的前128维，用perm_b32转成PV的A操作数（每lane：1维 × 4个token）
        O[:, 0:128] += V^T·P16^T                # 4个16-token段 × 8条MFMA 16×16×16
        l += P的行和（lane内16个求和，再加同行另外3个lane的）；cand = S的lane内最大值
        读V(t−1)的后128维 → O[:, 128:256] += V^T·P16^T
        new = max(同行4个lane的cand)·scale；m' = new + 1 if new > m + 7 else m
        if wave内有行m' ≠ m: O ·= exp2(m − m')；l ·= exp2(m − m')   # 未变的行乘1
        P = S·scale − m'；m = m'
    # 收尾
    DMA V(steps−1)（越过4·count的token读0：V若是NaN，P = 0也屏蔽不了）；P = exp2(P)；l += P的行和
    O += V^T·BF16(P)^T（前、后128维各一次）
    inv = 1/l if l > 0 else 0
    O·inv → BF16（RNE）→ 写进LDS（XOR swizzle）→ barrier → 按行优先读出 → 有效行写out[q0 + query, hkv·G + head, :]
# STAGGER：wave 4–7在开头比wave 0–3多过一个barrier、结尾少过一个，两组阶段错开一拍：一组做MFMA时另一组在DMA/读LDS
# phase内的barrier把DMA写LDS与对应的LDS读隔开：K区在K(t)读完后才被K(t+1)覆盖，V区同理
# KV长度不是4的倍数时不另编变体：块内第k个token的DMA行偏移取min(块行偏移, 第kv_len−1−k行)（s_min_u32，与DMA在同一段asm），
# 越过KV末端的部分尾块改读请求最后一个token（K被mask屏蔽、V权重为0）；请求不足k+1个token时上限为0，由buffer范围检查返回0
```

#### 1.2.7 K6 pack（[_pack_body](attention_direct_packed.py#L172)、[_pack_block](attention_direct_packed.py#L112)）

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

#### 1.2.8 K7 packed direct与K7′ raw direct（[packed _kernel](attention_direct_packed.py#L427)、[packed _body](attention_direct_packed.py#L319)、[raw _kernel](attention_direct.py#L433)、[raw _body](attention_direct.py#L227)）

```python
# packed，gated（有union计划）：grid (BQ·HK, union tiles)，64线程 = 1 wave；tile = blockIdx.y，local = blockIdx.x // HK，hkv = blockIdx.x % HK
#     只算路由为direct的tile的行：if active[tile] == 0 and local < union_meta[tile].rows: body(union_meta[tile].q_row0 + local, hkv)
# packed，ungated（没有union计划）：grid rows·HK，body(blockIdx // HK, blockIdx % HK)
body(row, hkv)：                                  # 1个wave = 1个query × 1个KV头，同时算该KV头下的G个query头
    base, pos = direct_meta[row]的PK起点与query位置；vis = pos + 1
    C = min(vis//4, 512)；count = 4·C + vis%4；steps = ceil(count/32)   # 本行要读的token数（完整块 + 因果尾）
    # 本行第n个token：n < 4C时是PK中第block_indices[row, n//4]块的第n%4个token，否则是因果尾的第n−4C个token
    q：lane的head = lane&15（只用前G个，其余读0），把该头256维中的64维读进寄存器（QK的B操作数）
    cached = block_indices[row, lane]             # 每lane缓存1个块号，64个lane共64块 = 8个BN32步
    K(0)：每步的token分两段各16个；lane按n = 段起点 + (lane&15)算PK地址（块号用ds_bpermute从lane n//4取），
          读该token的64维（lane>>4选8维组，跨4个64维段×2个32维半），即QK的A操作数
    m = −1e30；l = 0；O = 0（每lane 64个FP32）
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

# raw（PK+PV > 64MiB，不打包，直接读原K/V）：grid Σ_r ceil(q_r/4)·HK，256线程 = 4个wave = 同请求连续4个query（各一个wave）
#   gated：direct_flag ≠ 0，且4行中至少一行所在tile不走union才执行；每个wave再查自己那行，走union的行不算也不写
#   LDS 32KiB只用于输出转置
#   每个wave的计算与packed相同，区别在K/V的取法：
#   K：n = 段起点 + (lane>>2)；相邻4个lane读同一token的512B（lane&3选16B，每lane读8次），QK前用ds_bpermute换成A操作数布局
#   V：每lane直接从原V读4个token × 同8维（越过count的token读0），在PV里用perm_b32拼成A操作数（1维 × 4个token）
#   V的地址同样用ds_bpermute从K的地址取，无需查块表
#   softmax每步无条件 O ·= α（不做ballot判断）；块号同样每8步（64块）预取一组，在V读之后发出
```

## 2. Indexer prefill

### 2.1 调用链

SGLang在每个QSA层的extend forward中调用一次，各TP rank重复计算，不随TP变化。依次发出下列launch（编号沿用，I3、I6已删除）；每块logits不超过256MiB，长上下文时I4、I5按块重复。

| # | Kernel（实现） | grid×block | 作用 |
|---|---|---|---|
| I1 | _indexer_q_prep（Triton） | ⌈T/8⌉×256 | q norm/RoPE（与SGLang逐bit一致）、ring写、stats复位 |
| I2 | _indexer_k_compress（Triton） | ⌈groups/16⌉×256 | 4成员组均值→norm/RoPE→压缩池 |
| I4 | qsa_indexer_logits（FlyDSL） | items×256 | MFMA logits（按token_slot_table直读压缩池），顺带每行min/max位 |
| I5 | qsa_indexer_topk（FlyDSL） | ⌈rows/4⌉×256 | 每wave一行：top-512与2051宽token输出；位置与host布局不符时`s_trap` |

### 2.2 伪代码

#### 2.2.0 整体流程

输入qk是`index_qk_proj`的输出：每个token 4个query头加1个key，各128维。key每4个token压成一个压缩key（即一个“块”），选择以块为单位。

1. **I1（每8个token一个program）**：q做RMSNorm与RoPE（与SGLang逐bit一致）；按SGLang给的state_slots写pending ring：每个请求末尾凑不满一组的token（seq_len%4个）写进本请求的ring槽（prefill结束后qk就释放了，decode补齐这一组时只能从ring取），其余token写到不会被读取的dump行；复位每行的logits统计位。
2. **I2（每16个组一个program）**：本次凑满4个token的组求均值后做norm/RoPE，写进压缩池。请求s的第j个压缩key在池中的槽号为`token_slot_table[s, 4j] // 4`。
3. **按块循环**（每块logits不超过256MiB，长上下文分多块）：
   - I4：每个item（≤128行 × ≤512个key）用MFMA算logits = Σ_头 relu(q_h·k)·scale，key按槽号直接从压缩池读；同时把每行logits的最小/最大位折叠进stats。
   - I5：每行一个wave，选logits最大的512个块（同值取块号小的），按块号升序展开成每块4个token，再接因果尾token和-1，写成2051宽的一行。行位置与host布局不符时在这里trap。

可见压缩key不超过512个的行不需要logits，I5直接全选；整个tile都不超过512时不建item。

#### 2.2.1 入口（[prefill_indexer](indexer.py#L323)、[_Layout](indexer.py#L224)）

```python
prefill_indexer(qk[T,640], *, heads=4, positions, logical_positions, state_slots, key_state, rope_state,
                write_locs, member_rows, group_sequences, group_ends, rope_matrix, compressed,
                token_slot_table, cos_sin_cache, axis_map, q_weight, k_weight, q_eps, k_eps,
                seq_lens, extend_lens, q_out=None):
    host检查：4头、D128、rotary维度约束、每请求≤65536个压缩key、各张量的dtype/形状/设备、压缩池<2GiB
    # group_sequences、group_ends只检查形状，kernel不用
    layout = _layout(seq_lens, extend_lens)      # 按长度LRU缓存8个；numpy规划，一次pinned上传，不同步stream
    q = q_out或empty[T,4,128]；stats = empty[T,2]  # SGLang数值校验传q_out取回q
    I1                                           # q norm/RoPE、ring写、stats复位
    if groups: I2                                # 本次凑满的4-token组写进压缩池
    out = empty[T,2051]
    logits = empty[最大块rows·width + 512] FP32  # +512：top-k按512个值对齐读到行尾之后
    for chunk in layout.chunks:                  # 每块logits ≤ 256MiB
        if chunk.items or I4尚未编译: I4 → logits[rows, width]，并折叠stats   # 第一次调用即编译；没有item时launcher不发kernel
        I5 → out[row0 : row0+rows]               # 行位置与host布局不符时在这里trap
    return out

_Layout(seq_lens, extend_lens):                  # 只由host长度决定
    prefix_r = seq_r - extend_r；compressed_r = seq_r//4       # 请求r的压缩key数
    row_info[row] = (prefix_r + i, seq_r)        # 每行应有的位置与序列长度：I5据此校验位置并确定count
    每请求按128行切tile：bound = min((prefix_r + start + size)//4, compressed_r)   # tile最后一行可见的压缩key数
    分块：依次加入tile，直到rows·width·4 > 256MiB（width = 块内最大bound按4取整）
    item：只为bound > 512的tile建（≤512时I5全选，不需要logits），按KC=512切key区间
          → (row0, local0, rows, request, start, end)             # local0为块内的行号
```

#### 2.2.2 I1–I2（[_indexer_q_prep](indexer.py#L79)、[_indexer_k_compress](indexer.py#L121)）

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
    每组写16B到logits[local0 + 32w + column, key]；组起点 ≥ end或行超出rows的写被丢弃（越界偏移）
    # 跨过end的最后一组会多写≤3列，这些列不会被任何行当作因果logit读取
    logits ≥ 0，无符号位序即数值序：用umin/umax把写出值折进low/high（NaN的位模式大于+inf）
    若还有下一块：把寄存器里的下一块写进另一个LDS缓冲；barrier
两半lane（half 0、1持有同一行）合并low/high；half 0的lane：atomic_umin(stats[row].low, low)、atomic_umin(stats[row].not_high, ~high)
# 同一行可能被多个item（不同key区间）写，原子umin把它们合起来；stats由I1初始化为全1
# tile内各行都算到tile的bound，比本行因果范围多出的key也写出；I5只读本行前count个key，
# stats是超集上的界，直方图分箱仍然单调，选择结果不变
```

#### 2.2.4 I5 top-k与展开（[_row](indexer_topk.py#L622)、[_write_row](indexer_topk.py#L596)、[_select](indexer_topk.py#L555)、[_radix](indexer_topk.py#L473)、[_emit_ascending](indexer_topk.py#L511)）

```python
# grid = ceil(rows/4)，256线程：1个wave = 1行，wave之间互不等待（只用wave内同步）
# 每wave的LDS：257个int直方图（256个bin + 1个不参与统计的sink bin；排名阶段复用为64个候选的(key, id)，
#   归并阶段复用为输出缓冲）+ 512个uint16块号chosen
expected, seq_len = row_info[row]
if logical_positions[row] ≠ expected: s_trap 2   # 位置与host布局不符，进程中止
vis = expected + 1；count = min(vis//4, seq_len//4)  # 本行可见的压缩key（块）数
# 记x[j] = logits[row, j]（块j的logit），j ∈ [0, count)
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
| D1 | qsa_indexer_decode_logits（FlyDSL） | (splits, B)×256，16KiB | 按页表只读本行的压缩K，MFMA算logits |
| D2 | qsa_indexer_decode_topk（FlyDSL） | B×512，约9.7KiB | 每行一个CTA（8 wave）选top-512并展开token |

### 3.2 伪代码

#### 3.2.0 整体流程

decode时每行是一个请求的当前token；MTP verify时一个请求占连续W（≤4）行，即bonus token和草稿token（位置L..L+W−1）。

1. **D0（每行一个program）**：q做norm/RoPE；本token的key和位置写进pending ring；按SGLang的固定形状，把group_locs指向的4个成员求均值并norm/RoPE，写到write_locs（与SGLang逐bit一致）。每步每行都压缩一次：只有本token补满一组（seq_len是4的倍数）时write_locs才是该组的压缩槽，否则是惰性的slot 0，结果丢弃。verify时D0只做q和ring写，压缩由D0′在本次所有行写完ring之后完成，与SGLang先写全部ring、再gather的顺序一致；SGLang的ring每个请求有2×ratio个槽，窗口行不会覆盖该组在窗口之前的成员。
2. **D1（(splits, B)个CTA）**：按页表只读本行的压缩key（每页16个），MFMA算logits；超出本行长度的位置写-inf。
3. **D2（每行一个8-wave CTA）**：与prefill共用选择核心，选512个块并展开成2051宽的token行。

长度、位置和页表都在设备端读取，grid只由页表宽度决定，所以graph重放时可以原地更新。

#### 3.2.1 入口（[decode_indexer](indexer.py#L401)）

```python
decode_indexer(qk[B,640], *, positions, logical_positions[B], state_slots, key_state, rope_state, write_locs[B],
               group_locs[B,4], compressed, page_table[B,P], lengths[B], cos_sin_cache, axis_map, q_weight,
               k_weight, q_eps, k_eps, seq_lens[B], verify=False, q_out=None, logits_out=None):
    # 每行一个query token，同名参数与prefill含义相同；compressed按页（每页16个压缩key）读，lengths是各行的压缩key数
    host检查：与prefill共用的检查，另要求B、P > 0和各[B]、[B,4]、[B,P]整型张量的dtype/形状/设备
    q = q_out或empty[B,4,128]
    D0                                          # q norm/RoPE、ring写；verify=False时还压缩一个组
    if verify: D0′                              # 所有行的ring写之后，压缩各行的组
    width = 16·P（≤65536）；logits = logits_out或empty[B·width + 512] FP32；out = empty[B,2051]
    D1                                          # 按页表算本行logits
    D2                                          # 选512个块并展开成token
    return out
# verify=False时各行必须属于不同请求（D0只看得到本行自己的ring写）；verify=True时一个请求可占多行
# 先eager预热再capture，graph重放可原地更新长度与位置；SGLang数值校验传q_out、logits_out取回Q和logits
```

#### 3.2.2 D0 prep与D0′ verify压缩（[_indexer_decode_prep](indexer.py#L147)、[_indexer_ring_compress](indexer.py#L195)）

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

# D0′：grid = B，256线程，仅verify。D0已把本次所有行的key和位置写进ring（同一请求的W行占W个槽），
# 与SGLang一样全部从ring读成员
mean = Σ_{g∈group_locs[row]} key_state[g] / 4（BF16舍入）
k = norm_rope(mean, k_weight)，位置取rope_state[group_locs[row, 0]]
compressed[write_locs[row]] = k
```

#### 3.2.3 D1 logits（[indexer_decode._kernel](indexer_decode.py#L68)）

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
# 行中[16·pages, width)不写，D2只读[0, length)
```

#### 3.2.4 D2 top-k与展开（[_decode_kernel](indexer_topk.py#L699)、[_write_row](indexer_topk.py#L596)、[_span](indexer_topk.py#L380)、[_bases](indexer_topk.py#L405)）

```python
# grid = B，512线程 = 8 wave，一行由8个wave分担；LDS：8个直方图（各257个int）+ 每wave 4个归约字 + 64个候选 + 512个uint16（约9.7KiB）
# 算法与I5相同（共用_write_row、select、基数选择和归并），结果与单wave逐位相同；差别如下
count = lengths[row]；vis = logical_positions[row] + 1；seq_len = seq_lens[row]   # 不做位置校验
分工：共ngroups = ceil(count/512)组，wave w负责连续的组[w·per, (w+1)·per) ∩ [0, ngroups)，per = ceil(ngroups/8)
low/high/NaN：decode没有stats，先多扫一遍：各wave算自己那段的min/max/NaN，写进LDS归约字，CTA barrier后合并
直方图：每wave一个；找阈值时把8个直方图的对应bin相加
写出位置：收集遍和选择遍中，wave w从“编号更小的wave在同一类里的个数之和”开始写（各wave的计数经LDS交换）；
    因为wave按块号顺序分段，拼起来的chosen与单wave扫描的顺序一致
候选排名只由wave 0做；同步用CTA barrier（I5只用wave内同步）
归并：512个线程，每个线程负责1个输出位置；因果尾token与−1同I5
```

## 4. 文件、接口与限制

### 4.1 文件

| 文件 | 内容 |
|---|---|
| [attention.py](attention.py) | 公共入口与输入检查、工作区和scratch arena、热路径重放（`_Hot`）、首次调用预编译（`_warm`） |
| [attention_prepare.py](attention_prepare.py) | K1–K3：恢复/校验/散射、计数与压缩、全局路由、任务排序、精确mask；host端`allocate_plan` |
| [attention_union.py](attention_union.py) | K5 union：M128×N64，按任务顺序的持久grid，两组wave错拍流水 |
| [attention_direct_packed.py](attention_direct_packed.py) | K6 pack与K7 packed direct，以及packed/raw共用的direct helper |
| [attention_direct.py](attention_direct.py) | Direct计划（`DirectPlan`、`prepare`）与K7′ raw direct |
| [indexer.py](indexer.py) | Prefill/decode入口与布局，I1、I2、D0、D0′四个Triton kernel |
| [indexer_logits.py](indexer_logits.py) | I4 prefill logits（顺带归约每行min/max） |
| [indexer_topk.py](indexer_topk.py) | I5/D2：prefill与decode共用的top-512选择、升序归并和token展开 |
| [indexer_decode.py](indexer_decode.py) | D1 decode logits（按页表读压缩key） |

### 4.2 接口

- Attention只接受3D BF16 Q/O `[M,H,256]`、K/V `[N,HK,256]`和int32 `[M,2051]`选择（完整唯一的四token块、0–3个因果尾token、-1填充），local heads共享选择，`H/HK ≤ 16`。只支持gfx942；5D布局已撤销。
- Indexer为独立的D128投影、4个Q头和1个K头、ratio4/top512，最多65536个压缩key（262144 token）。Prefill输出每行块号升序，可原样交给attention。`prefill_indexer`和`decode_indexer`除`qk`外都是显式关键字参数，同名参数含义相同；可选的`q_out`（decode另有`logits_out`）取回归一化后的index Q（和logits）。
- 不另设校验launch：indexer两个入口在launch前只在host检查张量的dtype、形状、设备和几何，非法时抛`ValueError`；attention的K1发现非法行、prefill top-k发现query位置与host布局不符时执行`s_trap 2`，GPU队列出错，进程以HSA异常（code 0x1016，日志含kernel名）中止。Decode不校验位置。

### 4.3 Graph、scratch与编译

- Graph capture前须在同一stream、同一layout上eager调用一次。工作区只保存layout元数据和计划；scratch来自每个(device, stream)一块按需增长的arena，同stream的各layout共用。Capture时固定当时的缓冲，arena之后增长不影响graph里的指针；同一stream上capture的graph不得彼此并发重放，也不得与该stream上的eager调用并发。
- 精确长度、任务数、grid和容量，以及K2/K3的排序宽度、移位量、tile行数和扫描宽度都是运行时参数；零任务launcher不发GPU kernel。
- 每种(device, H, HK, scale)第一次调用attention（不在capture中）时，先用全因果dummy跑不建计划和每种可用的tile行数，编译全部变体：H12为K1两种、K2/K3各一种、union一种（10行）、packed与raw direct各两种；H6另有21行union；H3另有42行union及使用64位成员的K1–K3。Triton和FlyDSL缓存都为空时，这次调用在H12/H6/H3约需36/39/68秒，之后任何layout都不再编译。
- Indexer的I1、I2、D0对行数、组数和position stride不做整数特化，各只编译一份；prefill第一次调用即编译logits（即使该批没有行需要logits），decode的三个kernel在第一次decode调用时编译，verify另需的不压缩D0与D0′在第一次verify调用时编译（SGLang在graph capture前的eager预热中）。
- FlyDSL持久缓存的键只追踪被编译函数的源码，只改嵌套`@flyc.jit`或helper时可能返回旧kernel。修改PyHIP后先清空缓存，或设置新的`FLYDSL_RUNTIME_CACHE_DIR`。

### 4.4 已知限制

- 路由时延模型由MI308X分阶段计时拟合（union中位误差4.4%、p90 13.2%；direct中位误差1.4%），不保证每个布局都选到最快的路径。Union不受最长任务限制时总是全部保留：模型低估raw direct的代价，去掉这一条件后long_high M4096慢37.6%。
- 删除dense后，完整因果前缀prompt 384–2051行比原dense慢0–27%（10-02测，12层合计不超过约0.33ms）；删除候选提升后，少数TP2高共享、P0 4k随机选择和TP4分块布局的attention慢8–21%（10-04测）。真实12k单请求和服务布局不受影响。
- 长行decode logits的耗时随buffer所在显存位置分成快慢两组，相差13–19%。
- 服务验收是数值验收（attention `.02/.02`、prep逐bit），不是模型质量或生成文本逐bit一致的验收。

### 4.5 当前数据来源

[benchmark说明](../../../../../benchmarks/qsa/readme.md)第2、3节的数据：

- Attention逐kernel与整体：[qsa_route_threshold_20261004_01](../../../../../mytest/mydata/qsa_route_threshold_20261004_01/official_attention/kernels.txt)。
- Prefill与decode indexer：[qsa_compile_once_20261002_01](../../../../../mytest/mydata/qsa_compile_once_20261002_01/official_indexer_2/kernels.txt)。
- host下发：[host.json](../../../../../mytest/mydata/qsa_perf_final_20261002_01/host.json)。
- 服务性能与数值验收：[qsa_route_threshold_service_20261004_01](../../../../../mytest/mydata/qsa_route_threshold_service_20261004_01/summary.json)。
- MTP（EAGLE 3/1/4）服务性能、数值验收与C1 profile：[qsa_mtp_ringfix_20261008_01](../../../../../mytest/mydata/qsa_mtp_ringfix_20261008_01/analysis.json)。

## 5. 优化历史

按时间记录改动、关键结果和证据。数值只代表当时的源码与协议，当前性能以benchmark说明为准。2026-10-04整理前的完整记录（逐例A/B表、探索过程、服务细节）只读保留在[readme快照](../../../../../mytest/mydata/qsa_readme_merge_20261004_01/readme_before.md.snapshot)和[伪代码快照](../../../../../mytest/mydata/qsa_readme_merge_20261004_01/todo_before.md.snapshot)；10-02的逐kernel耗时分析与AP/AS/IP/IS/DP/DS条目见[更早的伪代码快照](../../../../../mytest/mydata/qsa_todo_cleanup_20261004_01/todo_before.md.snapshot)，09-30之前实验目录的README和opt日志见[qsa_docs_layout_20260930_01](../../../../../mytest/mydata/qsa_docs_layout_20260930_01/archive.json)。

测量口径（各条不再重复）：

- A/B：冻结对照与当前源码在同一进程交替测量，原cudaPerf、10 buffers/2 warmup/128 samples，计时输出逐位复核。10-02起为138例矩阵（真实12k、合成、P30k共享、P0、长上下文、小批、ragged和服务布局，各TP2/4/8），分乱序和升序两种行序。真实TP4/8输入是TP2 capture的派生头切片。
- 不建计划的小批受host限制，两臂代码相同时也有约4%的顺序偏差（[对照](../../../../../mytest/mydata/qsa_runtime_k3_20261002_01/ab_control_small/result.json)），这类用例要按对照校正。
- 服务数值验收：TEST=1，TP2/TP4各35请求（含24k分块、32并发和短请求），每个rank的12层都校验attention与prefill indexer。服务性能：原生与PyHIP依次测TP2/TP4×C1/C2/C4/C8，每档32请求、12000输入/350输出，计时时关闭校验。服务测量期间不得修改被哈希的源码（10-02有一轮因此作废）。
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
