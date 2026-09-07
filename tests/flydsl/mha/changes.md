# 跨机器修改记录

后续涉及接口、测试口径、kernel或依赖的改动都在此追加：日期/起点SHA、修改范围、旧行为→新行为、
必须重测的场景及实际结果。不同机器保留各自结果，不覆盖历史JSON/ISA，也不将compile-only视为native通过。

## 2026-09-07：10-buffer测试、SWA gather+AITER、移除FP8 register

起点：`483cad8ebdf9ebeb7418871a3624b404f2790509`及本机已有未提交参考/空闲等待适配。
状态：**MI308完整重测完成**。不commit、不push。gfx950仍需在对应机器重测。

| 修改 | 旧行为 | 新行为 / 其他机器需关注 |
|---|---|---|
| `--buffers` | 单输入集 | 默认10，独立随机输入seed+i、独立Q/K/V/metadata/scale、每候选每buffer独立O；JSON记录地址与seed |
| `--run-count` | 5个单buffer event | 5轮，每轮遍历全部buffer；默认每候选50个event样本，最终取所有样本中位数 |
| `--repeat` | 同一buffer重复 | event内每次调用继续轮换buffer，按调用数归一化；不同候选使用相同buffer索引序列 |
| 正确性 | 每候选1个输出 | 每buffer独立FP32 O参考、有限值/容差检查及两次额外逐位重复；汇总acc取各buffer最大值 |
| SWA AITER参考 | 仅prepared linear CK | 保留`aiter`，增加`aiter_gather`：每次完整KV gather+CK，单event总时间；不裁掉SWA前缀 |
| gather workspace | 无 | 每buffer独立slot mapping与linear KV，预分配不计时，但每次缓存读取/gather均计时 |
| 带宽 | 逻辑Q/K/V/O | 保持已有定义；`aiter_gather`另加完整KV读+写，按其总event时间计算；不是实测HBM带宽 |
| FP8 register | 可选低性能分支 | 移除backend、全局K/V加载/输出及备用归约分支；仅LDS。`memory_mode='register'`明确拒绝，不静默fallback |
| 文档 | 单buffer数据及零散链接 | [README.308.md](README.308.md)各场景给复现命令；新数据全部10-buffer重测，旧单buffer报告只读保留 |

### 修改文件

- [test_mha_pa.py](test_mha_pa.py)：唯一入口、多buffer测量、同文件契约与native边界检查。
- [_references.py](_references.py)、[_gather.py](_gather.py)：测试专用完整gather+CK，不进入生产dispatch。
- [_testing.py](_testing.py)、[mha_pa_fp8_942.py](mha_pa_fp8_942.py)：移除FP8 register；LDS算术、同步和调度保持原顺序。
- BF16/gfx942、BF16/gfx950、单waveSWA三个生产kernel不修改；LSE仍不纳入本轮测试。

### 跨机必测

1. 主入口pytest：buffer轮换/索引覆盖/去别名、FP8两种scale与C/NC尾页、SWA gather乱序/空KV/live cache。
2. 原各组：BF16 basic六shape、FP8 basic四shape（仅LDS+BN32参考）、BF16 dense三shape、FP8 scalar三shape。
3. SWA Q16384/KV32768、65536、131072 × D128/192：direct、prepared AITER、每次gather+AITER均先验精度再测。
4. BF16 Q65/KV129、W0+sink SWA小shape及显式`--repeat 3`轮换验证。
5. gfx950另机跑static/persistent+SWA；本机MI308只将架构不匹配记skip，不宣称已验gfx950。

统一PTL策略以机器授权为准。本轮MI308保持Enabled/VECTOR,F8，无硬件setter/reset；
`--wait-idle`无截止时间等待无其他进程的稳定空闲窗口。不因10-buffer增加内存就静默缩小shape或buffer数。

### MI308实测结果

- 主入口pytest：16通过/2个gfx950跳过。其中2项CPU契约、14项native边界/gather；不是16项全GPU。
- 14份性能报告全部complete：68条候选、3400个event区间、3700次候选调用；10buffer全部独立、每轮覆盖0–9。
- 每候选10个buffer均通过FP32 O、有限值及两次额外逐位重复；repeat3原生验证了跨buffer9回绕到0/1。
- SWA新增参考每次执行完整KV gather+CK，实际2个dispatch；每buffer独立K/V workspace，字节分子额外计完整KV读写。
- FP8 LDS七项均快于指定BN32；BF16 dense三项仍慢6.30%–11.37%，不能把accuracy passed称为性能达标。
- 结果目录：[mi308_multibuffer_20260907T135348Z](results/mi308_multibuffer_20260907T135348Z)。
	各场景的复现命令和重测表见[README.308.md](README.308.md)第6章。
- 旧单buffer证据未修改；BF16/gfx942、BF16/gfx950、SWA三个kernel SHA未变。
	FP8删除register的LDS源级结构与旧LDS specialization核对一致，并通过本轮原生功能与性能重测。
- 本机约13GiB可用磁盘，显存约192GiB；其他机器需关注10组输入与参考workspace占用，不做自动降规模。

## 2026-09-07：删除CPU测试工具契约，澄清repeat3

- 按用户要求删除`test_multibuffer_rotation_contract`和`test_multibuffer_defaults_and_removed_register`，
	同步移除README中的CPU测试复现命令。主入口现仅收集16项GPU用例，MI308适用14项，gfx950 persistent2项跳过。
- 只删除测试，不删除生产/计时入口的参数校验、buffer独立性断言、逐buffer正确性或register拒绝逻辑。
- `--repeat 3`原生专项保留：一个event内连续调用同一候选3次并轮换buffer，按3次归一化；
	3条汇总结果对应direct、prepared AITER、gather+AITER，不是3次重复各出一条。
- 不改kernel或性能协议；旧JUnit、68条性能结果及原始样本保留，旧“16通过”不改写为新计数。
- 删除后的回归状态：待运行剩余GPU测试，独立保存JUnit，不重新计时未变的性能场景。