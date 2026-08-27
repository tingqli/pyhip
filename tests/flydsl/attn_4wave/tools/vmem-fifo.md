# gfx942 单wave VMEM issue/backpressure

## TODO

- [x] 在独占GPU测试窗口重新复测四种96-op序列，并用新数据替换“正式结果”。
- [x] 按[第二阶段带宽搜索](vmem-bandwidth.md)在受控PTL环境运行完整矩阵并补充结果。

## 正式结果

2026-08-20在physical GPU 4重新采集40份ATT：80 CU同时运行，每CU一个wave；四种96-op序列各重复10次。
地址均为`disjoint`，burst前`alignment_nops=0`。表中“位置N”表示第N与N+1条VMEM之间。

| 序列 | 首个`>=16` gap | 首个`>=100` gap | 每份长gap数 | 最大gap范围 |
|---|---|---|---:|---:|
| 96 load | 16（10/10） | 50（8/10），52（2/10） | 8-14 | 472-1368 |
| 96 store | 1（10/10） | 48（2/10），39/42/43/45/46/47/60/61（各1次） | 6-12 | 128-176 |
| 48 load + 48 store，1:1交错 | 17（10/10） | 48（6/10），33/35/52/58（各1次） | 4-9 | 320-948 |
| 64 load + 32 store，2:1交错 | 16（10/10） | 49（6/10），37（3/10），40（1/10） | 4-9 | 652-1236 |

四种序列每份的全部gap中位数均为16 cycles。主要差异在快速区和长尾：

- 纯store没有load的4/8-cycle快速区，从第一个相邻gap起就是16 cycles；它也会出现
   `>=100` gap，但最大值仅128-176 cycles，明显短于含load序列。
- 纯load的首个长背压集中在位置50（8/10）；1:1和2:1分别以位置48、49为主（均6/10）。
- 混合序列的长尾主要发生在下一条为load时：1:1的65个长gap中，39个为S->L；
   2:1的70个长gap全部为L->L或S->L，L->S为0。

因此，store有独立且较浅的issue背压；混合流中的数百至上千cycle长尾主要与load接纳相关。
ATT不能进一步确定限制位于SQ、TCP还是TCC。

## 实验配置

| 项目 | 配置 |
|---|---|
| GPU | AMD Instinct MI308X，gfx942，physical GPU 4 |
| 拓扑 | 80 workgroup x 64线程；64 KiB LDS/workgroup；host验证80/80唯一CU |
| ATT | 只分析SE0/CU1的一条wave；gfx9时间量化为4 shader cycles |
| GPU状态 | PTL `Enabled / VECTOR,F8`，1800 MHz performance determinism |
| 重复 | 四种序列各10次，随机执行顺序；复测40/40均首次采集成功 |
| 访问 | 每条x4为16 B/lane、1 KiB/wave；load/store使用独立buffer和递增地址 |

- load目标全部唯一：96/48/64 load分别使用36组VGPR加60/12/28组AGPR；无WAW复用。
- store复用一组只读VGPR源；burst内无`waitcnt`，host检查实际写回值。
- 每份复测前后均确认GPU 4无其他实际VRAM/CU占用；采集结束恢复`auto / PTL Disabled`。
- 复测40份均验证静态/动态VMEM顺序、目标不复用和单wave trace完整性；原始ATT不保留。

时间轴：

```text
issue_begin = trace_begin_timestamp + stall_cycles
issue_gap[N] = issue_begin[N] - issue_begin[N-1]
```

`>=16`和`>=100`只是本文的统计阈值，不是硬件定义。

## 既有100-load基线

该100-load基线本次未重测。此前100条独立load、same/disjoint x alignment 0/1 x 5的20份ATT得到：

- alignment 0/1的首个`>=16` gap分别稳定在位置16/15。
- disjoint的首个`>=100` gap稳定在位置50（10/10），最大gap为708-1560 cycles。
- load 1-36写VGPR、37-100写AGPR；load 37处20/20均为16-cycle gap，长gap不是寄存器类型切换造成。

新的96-load结果与该基线一致：快速区约16条，主要长背压约在位置50。

## 边界

本实验测量相邻VMEM成功issue起点的间隔，不测memory completion或load-to-use latency。
wave级x4指令也不等于单个TCP/TCC事务，因此不能把位置16或50直接解释为物理FIFO深度。

## 工具

[vmem-fifo.py](vmem-fifo.py)的`run`支持`load`、`store`、`load-store-1to1`和
`load-store-2to1`；`analyze`读取rocprofv3 UI并输出逐VMEM及分类统计。

```bash
HIP_VISIBLE_DEVICES=4 /usr/bin/python3 /opt/rocm-7.2.0/bin/rocprofv3 \
   -i <att.yaml> --att-library-path=/opt/rocm-7.2.0/lib -- \
   /usr/bin/python3 tests/flydsl/attn_4wave/tools/vmem-fifo.py run \
   --device 0 --grid-blocks 80 --ops 96 --access-pattern load-store-1to1 \
   --launches 2 --alignment-nops 0 --buffer-mib 1280 --launch-address-mode disjoint
```
