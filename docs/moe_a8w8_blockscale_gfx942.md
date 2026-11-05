# gfx942 FP8 block-scale MoE（GateUp + Down，FlyDSL）与 aiter asm 对比

记录两段式MoE（[`moe_gateup_a8w8_blockscale_gfx942_lds`](../src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py) + [`moe_down_a8w8_blockscale_gfx942_lds`](../src/contrib/flydsl/moe_down_a8w8_blockscale_gfx942_lds.py)）的优化过程、与aiter 1-stage asm（`fmoe_fp8_blockscale_g1u1`）的对比结果，以及在另一台机器上复现的步骤。GateUp本身的分析见[moe_gateup_a8w8_blockscale_gfx942.md](moe_gateup_a8w8_blockscale_gfx942.md)。

## 1. 范围

- 形状：MiMo-V2.6-Flash的一个EP8 rank。hidden 4096，moe_inter 2048，256个expert取top-8，本卡32个expert。权重FP8、每128×128一个scale；激活FP8、每(token, 128)一个scale。T=8K/16K/32K，FLOPs=`6*routes*I*H`。
- 路由`ep_uniform`（专家均分）：每个token恰好1个本地路由，每个本地expert分到T/32行。
- 我们的流程：
  1. aiter `moe_sorting`，block 256，EP方式（全局expert id + `expert_mask`）。
  2. GateUp `2x4_pf1`，`out_scale_t=`（fp8_out）：epilogue里把SiLU(gate)·up按(row, 128)量化成FP8，按排序后的行写出A2 `[blocks*256, I]`。
  3. Down `4x2_pf1`，读排序后的A2，`reduce="atomic"`，有两种写法：
     - **atomic**：每一路用BF16 `global_atomic_pk_add_bf16`加进`out [T, H]`。`out`需要先清零，由aiter的sort顺带完成。
     - **B（`single_store=True`）**：本卡只有1路的token直接`global_store_dword`。`moe_route_counts`统计每个token的本地路数，`moe_zero_rows`只清零count≠1的行；sort不再清零`out`（传一个空的moe_buf）。
- 对手：aiter `fmoe_fp8_blockscale_g1u1`，gfx942上的1-stage asm，GateUp、SiLU、量化、Down在一个kernel里。输入和`aiter.fused_moe`在gfx942上喂给它的一样：预先量化好的FP8 a1加转置的scale。激活量化两边都不计时。
- "调优后的asm"：CSV里所有kernel实测最快的那个，tuner也是这样选的。

## 2. 结果

MI300X，`ep_uniform`，冷权重，queued计时（见5.4），中位数，单位µs。2026-11-03在本机GPU0上测得，测量期间GPU空闲检查全部通过，所有输出通过`check()`。

### 2.1 asm kernel扫描（不含sort）

| T | 最快 | 第二 | auto（未调优） |
|---|---|---|---|
| 8K | novs_32x256 866 | vs_64x256 872 | 882 |
| 16K | vs_64x256 1491 | vs_32x256 1598 | 1613 |
| 32K | vs_64x256 2804 | vs_psx_64x256 2999 | 3136 |

8K前三名相差不到1%，下表统一用vs_64x256。

### 2.2 与调优后的asm对比

含moe_sorting（括号内是相对vs_64x256的加速比）：

| T | asm vs_64x256 | asm auto | 我们 atomic | 我们 B |
|---|---:|---:|---:|---:|
| 8K | 919 | 937（0.98x） | 583（1.57x） | 555（1.66x） |
| 16K | 1585 | 1667（0.95x） | 1071（1.48x） | 984（1.61x） |
| 32K | 2975 | 3255（0.91x） | 1897（1.57x） | 1776（1.67x） |

不含moe_sorting（完整路径减去计时区间内的sort段）：

| T | asm vs_64x256 | asm auto | 我们 atomic | 我们 B |
|---|---:|---:|---:|---:|
| 8K | 871 | 887（0.98x） | 530（1.64x） | 512（1.70x） |
| 16K | 1511 | 1590（0.95x） | 986（1.53x） | 923（1.64x） |
| 32K | 2845 | 3124（0.91x） | 1748（1.63x） | 1673（1.70x） |

- B的时间已包含`moe_route_counts`（8–9 µs）和`moe_zero_rows`（6 µs）。
- 流水线内sort段：asm（block 64）48/74/129，atomic（block 256，清零`out`）54/84/149，B（不清零）44/61/103。sort本身两边一样：单独测block 64和256都是37/57/96。
- B如果也用会清零的sort：562/1011/1819。
- 分项：GateUp 345/626/1125；Down atomic 208/390/695；Down B 190/330/605。

### 2.3 精度

- 参考按同样的方式量化（GateUp输出按(row, 128)量化到FP8，再算Down）。calc_diff：我们1.41–1.43e-6（atomic和B的结果相同），asm 8.1–8.2e-6。换成相对RMS误差：我们1.7–2.3e-3，asm 4.0–5.1e-3（ep_uniform和random路由）。
- 原因：asm把Down的K（=I）按tile切成8片（x256 kernel）或16片（x128），每片分别用BF16 atomic累加（用仿真推断，误差和偏差都对得上），所以误差大，且每次运行结果都不一样。我们在FP32里累加完整的K，每一路只舍入一次，单路token的结果逐位可复现。
- 相对中间结果不量化的FP32参考，两边都是约2.3e-2（asm 2.31–2.33e-2，我们2.28e-2），主要来自两边共有的FP8中间量化。
- asm的`vs_32x256`/`novs_32x256`（非ps）在32K偶尔输出错误：约1/20次，600–9000个元素，空闲GPU上也复现过。调优选中的vs_64x256没有出现过。

## 3. 优化过程中的主要提升

Down和流水线的数字都是32K、µs；有两个值时是"ep_uniform / ep8（真实EP8路由）"。各行来自不同时间、不同GPU的测量，同一kernel在不同GPU上可差约10%，只在同一行内比较。

| 提交 | 改动 | 效果 |
|---|---|---|
| `cea2b4a` | GateUp：FlyDSL 8x1和2x4/4x2 LDS kernel（8x1后来删除） | 2x4冷cache 433/740/820/约900 TF（4K/8K/16K/32K），见GateUp文档 |
| `45afb76` | Down kernel：BM256×BN256，复用GateUp的K循环；每个wave先经LDS转置，一条atomic写2行×128 B（不转置时atomic请求约4倍） | atomic（清零+Down）847 / 887，比"写行+单独FP32 reduce"的889 / 947快5–6%；atomic只比普通store慢约5%（798 vs 762）；Down用4x2比2x4快6–8% |
| `45afb76` | MoE的BF16输出改为round-to-nearest-even | 原来从PyHIP抄来的打包是截断，整体偏小0.28% |
| `45afb76` | `sorted_a2`：A2按排序后的行连续读。原来按`[T, topk, I]` gather，L1 TLB miss是GateUp的8倍 | mainloop MFMA效率59%→91–92%；atomic总时间850→711 / 895→729 |
| `45afb76` | GateUp fp8_out：SiLU·up的FP8量化和排序写出并进GateUp的epilogue，省掉BF16中间结果和单独的量化kernel | ep_uniform：GateUp 1099→1085；Down 871→712（−18%）；GateUp+Down 1970→1811（旧路径还没算量化kernel） |
| `e74364b` | Down `single_store`（B） | Down（含统计路数、清零）726→608（−16%）/ 766→719（−6%）；整条流水线1845→1732 / 2074→2011；配合不清零的sort，8K/16K/32K的sort再省10/23/46 |

试过但没有采用：

- **清零放到另一个stream，和GateUp并行**：慢1–4%。GateUp占满了每个CU的寄存器，清零kernel只能在GateUp收尾时才上得去，还要分走功耗。
- **常驻kernel，把atomic推到下一个任务的K循环里**：gfx942上普通读和atomic共用`vmcnt`，K循环等数据时也要等atomic确认。第一版proxy测得更慢，但后来发现proxy本身有问题，没有在真实kernel上实现。
- **GateUp常驻kernel**：任务切换变短，但在750 W功耗墙下墙钟时间不变（2026-11-04复测−0.3%～+1.0%），已删除。`pair_blocks`保留、默认关闭：FP8输出在16K `ep_uniform`快3.5%，`ep8`最多快1%。见GateUp文档4.5、4.7。接入`aiter.fused_moe`后（a2a EP8均衡路由）整次调用16K快0.9%、32K快0.4–0.5%，见6.1节。

## 4. single_store的适用条件

- `out`必须是本rank私有的。普通store会覆盖同一行里其他生产者的结果，例如多卡往同一个buffer原子累加。实测这种用法会出错，所以默认关闭。
- `zero_out=False`时，单路的行是覆盖而不是累加，不能用来往已有内容上累加。
- 收益取决于单路token的比例：`ep_uniform`是100%（全部store，不用清零）；真实EP8路由约60%（32K Down −6%）；非EP（所有路由都在本卡）没有收益。
- 接入aiter/MORI的EP时，`moe_zero_rows`可以传`num_rows=num_local_tokens`，接收buffer里有效行之后的行不动；`moe_route_counts`只数有效的sorted行，本来就有界。见第6节。

## 5. 复现

### 5.1 测量环境

| 项 | 值 |
|---|---|
| GPU | MI300X（gfx942，304 CU，750 W） |
| ROCm / HIP | 7.2.4 / 7.2.26015 |
| PyTorch | 2.11.0+rocm7.2，Python 3.12 |
| FlyDSL | 0.4.0（`pip show flydsl`） |
| aiter | 第2节：容器自带的`/sgl-workspace/aiter`（ROCm/aiter `285579b62`，有本地改动），benchmark的JSON里记录为`v0.1.14-rc0-1708-g285579b62-dirty`。第6节：https://github.com/luocheng25/aiter.git 分支`tq/moe_gemm_blockscale`，commit `0ea121f3`加第6节的hook，源码目录直接import。asm kernel列表来自`hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv` |
| pyhip | 分支`luwei/942-fp8-blockwise` |

```bash
cd pyhip && pip install -e .     # 或者：mkdir -p /tmp/pyhip_path && ln -s $PWD/src /tmp/pyhip_path/pyhip && export PYTHONPATH=/tmp/pyhip_path:$PYTHONPATH
python -c "import aiter, flydsl"  # aiter checkout到上表的commit
python -c "import amdsmi"         # 空闲检查用；缺少时：pip install /opt/rocm/share/amd_smi
```

### 5.2 正确性测试

```bash
export HIP_VISIBLE_DEVICES=0
python -m pytest -q tests/contrib/moe/test_gateup_a8w8_blockscale_gfx942.py    # GateUp，69例
python -m pytest -q tests/contrib/moe/test_down_a8w8_blockscale_gfx942.py      # Down、single_store、GateUp→Down，69例
python -m pytest -q tests/contrib/moe/test_fmoe_a8w8_blockscale_asm_gfx942.py  # aiter asm，44例
python -m pytest -q tests/contrib/moe/test_aiter_fused_moe_flydsl_gfx942.py    # 经aiter.fused_moe走FlyDSL，5例（需要带第6节hook的aiter）
```

### 5.3 与aiter对比

脚本：[tests/contrib/moe/bench_moe_a8w8_blockscale_gfx942_vs_asm.py](../tests/contrib/moe/bench_moe_a8w8_blockscale_gfx942_vs_asm.py)

```bash
cd tests/contrib/moe
for t in 8192 16384 32768; do
  HIP_VISIBLE_DEVICES=0 python bench_moe_a8w8_blockscale_gfx942_vs_asm.py --tokens $t --json ep_uniform_$t.json | tee ep_uniform_$t.log
done
```

- 第一次运行要编译aiter和FlyDSL的kernel；编译过后每个T不到1分钟。
- 默认跑auto加CSV里全部10个asm kernel。`--asm vs_64x256`只跑这一个（auto总会保留）。
- 2.2节的表就是输出最后的`summary`两行：`with moe_sorting`和`excluding moe_sorting`。tuned列是本次运行里不含sort最快的asm kernel。
- 输出只有标着`gates PASS`时才有效。JSON里有全部样本、分段时间、空闲检查记录和环境（GPU、torch、HIP、FlyDSL版本，pyhip/aiter的git commit）。
- 其他选项：`--routing random`、`--timing cudaperf`（见5.4）、`--dry-run`（每个变体跑两遍并检查，不计时）、`--host-time`（各变体的host入队时间）。

变体名对照：

| 变体 | 内容（计时区间内） |
|---|---|
| `asm <kernel>` | asm kernel；sort预先做好，`out`在计时前清零 |
| `sort+asm <kernel>` | moe_sorting（按该kernel的block_m，清零`out`）+ asm |
| `GEMM1+GEMM2 atomic` | GateUp + Down atomic；`out`在计时前清零 |
| `counts+zero_rows+GEMM1+GEMM2 B` | 统计路数 + 选择性清零 + GateUp + Down B |
| `sort+GEMM1+GEMM2 atomic` | sort（清零）+ GateUp + Down atomic |
| `sortNZ+counts+zero_rows+GEMM1+GEMM2 B` | sort（不清零）+ 统计路数 + 选择性清零 + GateUp + Down B |
| `sort+counts+GEMM1+GEMM2 B` | sort（清零）+ 统计路数 + GateUp + Down B |
| `... [prep NaN+sort]` | 对照：计时前（不计时）做和完整路径一样的NaN填充+sort，只计kernel |
| `GEMM1`、`GEMM2 atomic`、`GEMM2 B`、`counts`、`zero_rows`、`sort<block>`、`sort256NZ` | 单个kernel |
| `stages <variant>: ...` | 同一次调用内用event分段：`sort`、`counts`、`GEMM1`、`rest` |

### 5.4 计时方法与注意事项

- **GPU必须空闲**：脚本在开始、每轮之间和结束时检查（activity≤5%、VRAM≤20%）。开始时不通过直接退出，中途不通过整次运行标为FAIL。本机常有别的容器占满全部8张卡，这时的数字不能用。
- **只指定一张卡**：`HIP_VISIBLE_DEVICES`给一个物理编号，脚本会核对amdsmi和HIP看到的PCI地址。同一kernel在不同GPU上可差约10%，只比较同一次运行里的数字。
- **冷权重**：a、scale、w1、w2共5份clone，每次读权重的调用换下一份。四次调用读过的权重远大于256 MB的MALL，轮回来时早已不在cache里。
- **queued计时（默认）**：每个变体一个block，先连续跑至少50 ms不计时的预热，再连续跑8次计时调用，每次一对event。block前的sleep让host先把整个block排进队列，所以GPU中间既不空闲也不等host。每个变体4个block、共32个样本，block顺序每轮反向。
- **默认不用逐次cudaPerf**：逐次计时每次都要sync，GPU在两次调用之间空闲，时钟会降，短kernel因此最多慢约30%。8K GateUp紧接着预热时是310 µs，逐次sync稳定在410 µs；asm一个长kernel只受约1%影响，所以这种测法对我们不公平。`--timing cudaperf`可以复现这种测法：8K/16K我们慢10–15%，B对调优asm的加速比降到约1.4–1.5x。
- **不要交错单次调用**：ABBA交错时，排在空闲或轻kernel之后的变体最多慢28%。所以改成每个变体一个block，先连续预热50 ms。
- **"不含sort"不要直接用没有sort的变体**：这些变体的`out`在计时前一刻被64–256 MB的填充写过，会让我们的GateUp+Down慢16/7/82 µs（8K/16K/32K），asm不受影响。summary用的是完整路径减去计时区间内的sort段。两个`[prep NaN+sort]`对照组验证过：计时前状态相同时，sort带来的增量两边一样，asm +51/+95/+167，我们+51/+93/+165。
- **Host开销**：asm每次调用入队约40 µs，我们200–430 µs（当时每个FlyDSL kernel都走`JitFunction`调用，60–90 µs）。queued模式下host始终跑在GPU前面，这部分不进计时。现在第一次调用之后走缓存的`CompiledFunction`（`helpers.run_compiled`），每个kernel 11–23 µs。

## 6. 接入aiter.fused_moe（`AITER_FMOE_FP8_BLOCKSCALE_FLYDSL=1`）

- **位置**：aiter（luocheng25/aiter `tq/moe_gemm_blockscale`）的`_fused_moe_impl`，在查tuned config和sort之前判断：能走就调用[`fused_moe_a8w8_blockscale_gfx942`](../src/contrib/flydsl/moe_a8w8_blockscale_gfx942.py)，不能就每种原因记一次warning，然后走原来的路径（gfx942上是1-stage asm）。必须在sort之前接管，因为我们要block 256的sort，而且sort不能清零`out`。SGLang的调用链`moe_runner/aiter.py` → `aiter.fused_moe` → `fused_moe_` → `_fused_moe_impl` → `fused_moe_1stage`正好经过这里。
- **接管条件**：
  - gfx942；128×128 block-scale的FP8权重和激活，g1u1 SiLU，routing weight在Down里乘，BF16输出，gate/up分开存放。
  - 没有bias、padding、swiglu_limit、a2_scale、stage2_scatter。
  - 权重和scale连续，布局和asm一样：shuffle_weight(16,16)，FP32 scale `[E, 2I/128, H/128]`、`[E, H/128, I/128]`。
  - `topk_ids.shape[0]` ≥ `AITER_FMOE_FP8_BLOCKSCALE_FLYDSL_MIN_TOKENS`，默认4096。
  - pyhip能import，并且排序后的FP8中间结果放得进一个2 GiB的buffer descriptor。
- **激活**：BF16输入在里面用aiter的per-1x128量化（`num_rows=num_local_tokens`）；FP8输入（MORI FP8 dispatch）要带token-major的`a1_scale [tokens, H/128]`。转置成GateUp要的`[H/128, tokens]`时，stride用buffer行数（`.t().contiguous()`）。aiter的`partial_transpose`用的是`num_local_tokens`，和GateUp不一致。
- **EP和all-to-all**：`expert_mask`、`num_local_tokens`、dispatch policy照常交给`moe_sorting`；`moe_zero_rows`只清`num_local_tokens`以内的行；有`output=`（例如MORI combine的staging buffer）时原地写，有效行之后不动。
- **测试**：[test_aiter_fused_moe_flydsl_gfx942.py](../tests/contrib/moe/test_aiter_fused_moe_flydsl_gfx942.py)覆盖EP + BF16输入、MORI式FP8输入 + `num_local_tokens` + `output`（有效行之后是垃圾数据和NaN）、非EP三种用法。

实测：`aiter.fused_moe`整次调用，含sort和激活处理，单位µs，2026-11-03 GPU0空闲。MiMo-V2.6-Flash一个EP8 rank，random top-8。buffer行数等于EP组这一步的token数，其中有本地路由的token排在前面（约66%）。aiter默认走未调优的1-stage asm（`vs_silu_1tg_ps_32x256`）。

| EP组token数 | 有效token | asm（默认） | FlyDSL | 加速比 |
|---:|---:|---:|---:|---:|
| 2048 | 1341 | 379 | 478 | 0.79x |
| 4096 | 2700 | 545 | 480 | 1.14x |
| 8192 | 5415 | 951 | 702 | 1.35x |
| 16384 | 10781 | 1646 | 1034 | 1.59x |
| 32768 | 21667 | 3220 | 1941 | 1.66x |

- 所有token都在本卡、BF16输入（`--mode ep`）的32768：asm 3403，FlyDSL 2102，1.62x。
- calc_diff：FlyDSL 2.5–2.6e-6，asm 1.3e-5。
- 2048以下是asm更快：GateUp在小M时被权重读取卡住，我们还多4次launch。
- 复现：`HIP_VISIBLE_DEVICES=0 python tests/contrib/moe/test_aiter_fused_moe_flydsl_gfx942.py --tokens 32768 [--mode ep]`。

还没解决的问题：

- **2 GiB上限**：排序后的FP8中间结果是`(ceil(tokens*8/256)+32)*256`行 × I。I=2048时buffer行数超过约13万就放不下，会自动回退到asm。MORI不裁剪接收buffer时（例如8×16384=131072行）就会碰到，需要SGLang按实际token数裁剪接收上限，或者给GateUp/Down加64位基址。
- **额外显存**：asm 1-stage没有中间buffer；我们在32768行时要约570 MB（FP8中间结果加scale），每次调用从torch caching allocator分配，SGLang的显存预留要留出这部分。
- **首次编译**：每个进程第一次调用要准备4个FlyDSL kernel，冷缓存时现场编译。服务启动时要预热，或者把FlyDSL的磁盘缓存带进镜像。
- **FlyDSL版本**：这些kernel用FlyDSL 0.4.0，aiter自己固定的是0.3.4.1，部署时要确认aiter自己的FlyDSL kernel在0.4.0下也正常。
- **只在MI300X验证过**（304 CU，8个XCD），其他gfx942型号没测。prefill以外（CUDA graph捕获）也没测。
- 本卡有多条路由的token仍然用BF16 atomic，累加顺序不固定，和asm一样。

### 6.1 a2a EP8均衡路由：pair_blocks、各kernel时间、ATT阶段占比（2026-11-04）

条件：`aiter.fused_moe`整次调用，MORI式FP8接收buffer加`num_local_tokens`，buffer行数等于EP组token数。EP8（256个全局expert，本卡32个），`balanced`路由：每个本地expert恰好T/32行（16K为512行、32K为1024行，没有padding），有本地路由的token占65.6%。`single_store=True`。权重和接收buffer各5份，每次调用换一份（冷cache）。asm、FlyDSL、FlyDSL+`pair_blocks`三条路径按全部6种顺序交错，共12轮，每段8次预热、16次计时。GPU3，测前测后都空闲。共测了两次，第二次用的是下面的脚本。脚本：[bench_fused_moe_a2a_gfx942.py](../tests/contrib/moe/bench_fused_moe_a2a_gfx942.py)。

整次调用，中位数，µs（两次运行）：

| T | 有效token | asm（默认） | FlyDSL | FlyDSL+pair | pair_blocks | 加速比（FlyDSL / +pair） |
|---:|---:|---:|---:|---:|---|---|
| 16384 | 10752 | 1658 / 1660 | 1020 / 1022 | 1011 / 1012 | −0.87% / −0.94%，23/24轮更快 | 1.63x / 1.64x |
| 32768 | 21504 | 3189 / 3197 | 1829 / 1829 | 1821 / 1819 | −0.43% / −0.54%，15/24轮更快 | 1.74–1.75x / 1.75–1.76x |

- `pair_blocks`只改GateUp的任务顺序。GateUp kernel本身（profiler，交错8轮×40次）：16K 596→588 µs（−1.5%，8/8轮），32K 1084→1075 µs（−0.8%，8/8轮）。那次运行里不受影响的Down也差了0.8%，第二次运行Down不变（630.3 / 630.7 µs），所以32K的收益也成立，但很小。
- 正确性：calc_diff asm 1.3e-5，FlyDSL 2.6–2.7e-6。本卡只有1路的token，开不开`pair_blocks`逐位相同。2路以上的token用BF16 atomic，累加顺序不固定：16K同一配置重跑一次，这些token有1.9–3.4%的元素不同，开`pair_blocks`时（2.2–3.0%）也在这个范围内；32K两次都没有出现不同。

各kernel时间（torch profiler，20次调用的中位数，µs；相加比整次调用略长）：

| kernel | 16K | 32K |
|---|---:|---:|
| a1_scale转置（`.t().contiguous()`） | 8.2 | 13.5 |
| moe_sorting（block 256，不清零，4个kernel） | 49 | 82 |
| `moe_route_counts`（清零 + 计数） | 11.7 | 12.1 |
| `moe_zero_rows` | 12.5 | 20.7 |
| GateUp | 612 | 1087 |
| Down | 351 | 630 |
| asm路径：moe_sorting（block 32，清零`out`）/ `partial_transpose` / asm kernel | 60 / 6.5 / 1602 | 99 / 7.1 / 3084 |

ATT阶段占比：rocprofv3 1.1.0 `--att`，16个SE各取CU 1的SIMD 0，buffer 1 GiB，采第7、8、9次调用的GateUp和Down（每个尺寸各3个dispatch）。

- 两个kernel都是250 VGPR加64 KiB LDS，每个CU同时只跑一个8-wave workgroup，被采的SIMD上正好是同一个workgroup的G0、G1各一个wave。
- 阶段按K循环（最常走的那条回跳）划分：prolog是workgroup开始到两个wave都进入K循环；mainloop是两个wave都在循环内；epilog是第一个wave出循环到workgroup结束。
- MFMA效率：两个wave的MFMA执行窗口（从成功发射起算16 cycle）取并集，除以阶段的cycle数。所有MFMA都在K循环内（每个wave GateUp 4096条、Down 2048条）。
- 下表每格是3个dispatch的范围（32K GateUp去掉下面说的例外，是2个），括号内是每个workgroup的平均cycle数。ATT期间的时钟是0.87–1.17 GHz。

| kernel | T | prolog | mainloop | epilog | mainloop MFMA效率 | 整个workgroup MFMA效率 | 每个workgroup |
|---|---:|---|---|---|---:|---:|---:|
| GateUp | 16K | 4.9–6.3%（7.8–10.2k） | 86.2–87.5%（140k） | 7.5–7.7%（12.2k） | 92.8–93.1% | 80.6–82.1% | 160–163k cycle |
| GateUp | 32K | 4.7–5.3%（7.5–8.5k） | 87.1–87.6%（140k） | 7.6–7.7%（12.2k） | 93.0–93.1% | 81.6–82.2% | 约160k cycle |
| Down | 16K | 12.7–13.2%（13.0–13.7k） | 64.6–67.3%（69k） | 19.9–22.7%（20.5–24.2k） | 93.4–93.6% | 61.3–63.8% | 103–107k cycle |
| Down | 32K | 10.1–10.9%（9.9–10.6k） | 71.3–71.4%（69.5k） | 17.8–18.5%（17.3–18.1k） | 92.7–93.1% | 67.1–67.4% | 约97k cycle |

- GateUp的K是4096，循环16次。prolog（读路由和scale、装第一个K block）约8k cycle，epilog（SiLU、FP8量化、按排序后的行写出）约12k cycle，两段合计约13%，这段时间几乎没有MFMA。
- Down的K只有2048，循环8次。prolog和epilog的绝对时间和GateUp同一量级，所以占比高：mainloop只有65–71%，整个workgroup的MFMA效率是61–67%。epilog的p95达到30–36k cycle（atomic和store）。16K的prolog和epilog cycle数比32K多，是因为那几次的时钟更高（1.16 vs 1.0 GHz），按µs算差别不大。
- 两个kernel的mainloop MFMA效率都在93%左右，和GateUp文档4.2一致。剩下的约7%（每个workgroup GateUp约10k、Down约4.5k cycle）是barrier、LDS和等待。
- 空workgroup：grid按最坏情况的block数分配（`max_sorted_blocks`：16K为544个block×16个N tile=8704个workgroup，实际干活的只有1024个；32K为16896个，干活的2048个）。多出来的workgroup读到有效block数就退出，每个约440 cycle，在被采的SIMD上占kernel时间的0–3.3%。
- 例外：32K第8次调用的GateUp，所有被采CU上的第3轮workgroup同时变慢，K循环从140k变成297k cycle。多出来的时间在`s_waitcnt vmcnt`（每个wave 61.6k，正常是2.6k cycle）和barrier上。第7、9次调用都没有出现，原因没有确认，上表不含这个dispatch。
- ATT会拖慢kernel：GateUp在ATT下16K/32K的跨度是621–685 / 1134–1325 µs，正常计时是596–612 / 1084–1087 µs。所以ATT只用来看占比和效率，不看绝对时间。

复现（ATT的解码库`librocprof-trace-decoder.so`的安装见[profile-gpu.md](profile-gpu.md)）：

```bash
cd tests/contrib/moe
HIP_VISIBLE_DEVICES=3 python bench_fused_moe_a2a_gfx942.py time --tokens 16384 32768 --json a2a.json
HIP_VISIBLE_DEVICES=3 rocprofv3 --att --att-library-path <解码库目录> --att-target-cu 1 --att-shader-engine-mask 0xffff \
    --att-simd-select 0x1 --att-buffer-size 0x40000000 --kernel-include-regex "gateup_kernel_0|down_kernel_0" \
    --kernel-iteration-range "[7, 9]" -d att16k -o att -- python bench_fused_moe_a2a_gfx942.py run --tokens 16384 --calls 10
python bench_fused_moe_a2a_gfx942.py phases att16k/ui_output_agent_*_dispatch_* --json phases16k.json
```
