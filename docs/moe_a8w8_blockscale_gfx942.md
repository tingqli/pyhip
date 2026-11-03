# gfx942 FP8 block-scale MoE（GateUp + Down，FlyDSL）与 aiter asm 对比

记录两段式MoE（[`moe_gateup_a8w8_blockscale_gfx942_lds`](../src/contrib/flydsl/moe_gateup_a8w8_blockscale_gfx942_lds.py) + [`moe_down_a8w8_blockscale_gfx942_lds`](../src/contrib/flydsl/moe_down_a8w8_blockscale_gfx942_lds.py)）的优化过程、与aiter 1-stage asm（`fmoe_fp8_blockscale_g1u1`）的对比结果，以及在另一台机器上复现的步骤。GateUp本身的分析见[moe_gateup_a8w8_blockscale_gfx942.md](moe_gateup_a8w8_blockscale_gfx942.md)。

## 1. 范围

- 形状：MiMo-V2.6-Flash的一个EP8 rank。hidden 4096，moe_inter 2048，256个expert取top-8，本卡32个expert。权重FP8、每128×128一个scale；激活FP8、每(token, 128)一个scale。T=8K/16K/32K，FLOPs=`6*routes*I*H`。
- 路由`ep_uniform`（专家均分）：每个token恰好1个本地路由，每个本地expert分到T/32行。
- 我们的流程：
  1. aiter `moe_sorting`，block 256，EP方式（全局expert id + `expert_mask`）。
  2. GateUp 2x4，`out_scale_t=`（fp8_out）：epilogue里把SiLU(gate)·up按(row, 128)量化成FP8，按排序后的行写出A2 `[blocks*256, I]`。
  3. Down 4x2，读排序后的A2，`reduce="atomic"`，有两种写法：
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
- **GateUp常驻kernel**：任务切换变短，但在750 W功耗墙下墙钟时间不变。`pair_blocks`只在16K快约2%，默认关闭。见GateUp文档。

## 4. single_store的适用条件

- `out`必须是本rank私有的。普通store会覆盖同一行里其他生产者的结果，例如多卡往同一个buffer原子累加。实测这种用法会出错，所以默认关闭。
- `zero_out=False`时，单路的行是覆盖而不是累加，不能用来往已有内容上累加。
- 收益取决于单路token的比例：`ep_uniform`是100%（全部store，不用清零）；真实EP8路由约60%（32K Down −6%）；非EP（所有路由都在本卡）没有收益。
- 接入aiter/MORI的EP时，`moe_route_counts`和`moe_zero_rows`的范围要限制在`num_local_tokens`以内，目前还没做。

## 5. 复现

### 5.1 测量环境

| 项 | 值 |
|---|---|
| GPU | MI300X（gfx942，304 CU，750 W） |
| ROCm / HIP | 7.2.4 / 7.2.26015 |
| PyTorch | 2.11.0+rocm7.2，Python 3.12 |
| FlyDSL | 0.4.0（`pip show flydsl`） |
| aiter | https://github.com/luocheng25/aiter.git 分支`tq/moe_gemm_blockscale`，commit `0ea121f3`，源码目录直接import。asm kernel列表来自`hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv` |
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
- **Host开销**：asm每次调用入队约40 µs，我们200–430 µs。queued模式下host始终跑在GPU前面，这部分不进计时。实际模型里GPU一直有活，或者用CUDA graph，也是这样。
