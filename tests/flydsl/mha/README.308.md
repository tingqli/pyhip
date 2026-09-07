# MI308X Attention：10-buffer测试与复现

更新：2026-09-07。**MI308X / gfx942 / 80 CU；PTL Enabled/VECTOR,F8；不测LSE。**
代码起点`483cad8ebdf9ebeb7418871a3624b404f2790509`，跨机关注点见[changes.md](changes.md)。
性能重测已完成：**68条候选、3400个event样本**。主入口现已删除2项CPU工具契约，
仅保留16个GPU测试用例（MI308适用14项，gfx950 persistent2项跳过）；旧JUnit的16通过包含已删除的2项CPU测试。
每条候选的10个buffer均通过FP32 O和重复逐位检查。FP8 register已删除，仅保留LDS。

## 1. 环境与结果

| 项目 | 本轮实际值 |
|---|---|
| GPU | MI308X / gfx942 / 80 CU / 约192 GiB |
| Python | 3.11.11，`$HOME/.venvs/pyhip-mha-mi308` |
| torch / HIP / 系统ROCm | 2.12.1+rocm7.2 / 7.2.53211 / 7.2.1 |
| FlyDSL / Triton | 0.3.1 / triton-rocm 3.7.1 |
| AITER / CK | `bde46043bcf08e41ac40395a18369ab6309153ca` / `15e12dd7f25ee583617c78f66cb502ff9916585f` |
| 结果目录 | [mi308_multibuffer_20260907T135348Z](results/mi308_multibuffer_20260907T135348Z) |

AITER源码/JIT位于`/tmp/cheluo-mi308-full-20260907`，editable路径被清理后需按固定commit重建。
原始单buffer及register报告保留，不修改历史数据。无PTL setter/reset/时钟/功率/NUMA修改，不commit/push。

## 2. 默认10-buffer协议

- `--buffers 10`为缺省值；独立随机输入`seed+i`、Q/K/V、metadata、scale，每候选每组独立O。
  dense/AITER转换和gather workspace也按buffer独立；JSON记录地址和seed，实测已核对无别名。
- `--warmup 10`依次预热10组；`--run-count 5`表示**5个完整buffer轮次**，每轮测完全部10组，
  默认每候选**50个event样本**。不是只分配10组却只计时5组。
- `--repeat 1`为默认；repeat>1在event内逐调用轮换，例如buffer9开始、repeat3使用`[9,0,1]`，
  时间除以repeat。各候选使用同一索引序列，候选先后顺序交替。
- 每buffer独立FP32 O检查、有限值检查、两次额外逐位重复；BF16容差0.02、FP8容差0.1不变。
  汇总acc为10组最大值，逐组值记录于`acc_per_buffer`。
- 总结取全部event样本中位数，不选最快值、不删慢样本；小shape的dispatch抖动也保留。
- JIT、量化、布局转换、workspace分配不计时；counter初始化、辅助launch和间隙在event内。
  `cudaPerf` GPU spin在起始event之前。多buffer不等于强制清cache，不宣称完全冷cache。

## 3. 参考与SWA路径

| 候选 | 含义 | 计时范围 |
|---|---|---|
| 当前8wave/LDS/单wave | 对应生产paged kernel | 当前完整调用 |
| 指定dense（BF16） | [LKG/V-global MHA](../test_attn_8wave_32x32_lkgv.py) | 同逻辑输入，dense转换不计时 |
| 指定BN32（FP8） | [paged prefill FP8分支](../pa_8wave/pa_prefill_8w32x32.py) | 同量化输入/descales/页表的完整调用 |
| `aiter` Full/Causal | public varlen，MI308实测ASM | prepared linear KV，不含转换 |
| `aiter` SWA | prepared CK varlen | 不含gather |
| `aiter_gather` SWA | **完整KV gather+CK** | 每次重新gather全部KV再执行CK，**一对event覆盖两段** |

SWA的`--aiter on`自动增加prepared和gather+CK两条参考；固定slot/workspace预分配，但每次读缓存/gather都计时。
不裁SWA前缀，不用两个独立均值相加，不把gather引入生产dispatch。乱序页、空KV、live-cache变更已验证。
指定dense仅支持B1/H=HK/Dq=Dv128/NC/Q%256=0/KV%32=0/unit descales；常规GQA/causal/SWA明确不适用。
AITER quick适配不支持FP8，auto明确N/A，on报错，不换dtype冒充。

## 4. 指标规则

时间越低越好；延迟变化相对同组“基准”，**负值更快，正值更慢**。`passed`仅指正确性。
TFLOPS只计可见QK/PV；GB/s是逻辑字节除对应event时间，**不是实测HBM流量/饱和率**。

- direct/prepared沿用MI350逻辑Q/K/V/O定义，SWA也计完整逻辑KV。
- gather+CK在此基础上加**完整KV读+完整workspace写**，除以总路径时间。
- 各路径字节分子不同，优劣优先比较同shape延迟，不能只看GB/s大小。
- FP8源为BF16随机数经scale量化为FP8，量化不计时；与旧native-cast/FlyDSL0.2.2/profiler gate不同。

## 5. 统一复现准备

第6章命令从Git根、同一个bash终端执行。每轮创建新OUT；各场景文件名不同，避免覆盖证据。
`--buffers`虽默认10，命令仍显式注明。实际每份JSON的config保留完整参数。

```bash
PY="$HOME/.venvs/pyhip-mha-mi308/bin/python"
MHA=tests/flydsl/mha/test_mha_pa.py
OUT=$(mktemp -d "$PWD/tests/flydsl/mha/results/mi308-10buf.XXXXXX")
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx942 MAX_JOBS=4 FLYDSL_RUNTIME_ENABLE_CACHE=0
export AITER_JIT_DIR=/tmp/cheluo-mi308-full-20260907/aiter-jit
export PATH="$(dirname "$PY"):$PATH"
unset FLYDSL_COMPILE_ONLY FLYDSL_COMPILE_ARCH CUDAPERF
bench() {
  local name="$1"; shift
  "$PY" "$MHA" --buffers 10 --check 1 --warmup 10 --run-count 5 --repeat 1 \
    --wait-idle "$@" --output "$OUT/$name.json"
}
```

`--wait-idle`无截止时间等待：连续3次、5秒间隔的gfx/UMC均0%，且无其他进程（包括驻留worker）才放行。
等待前不生成GPU输入，测试前后另检查进程；读取错误失败闭锁，不reset/杀他人进程。空闲检查不等于独占预约。

## 6. 10-buffer实测与逐场景复现

全部为修改后新数据。正文49条主结果；另6条basic SWA重复验证、10条小shape、3条repeat3，共68条。

### 6.1 BF16：Full/Causal与dense

**A. 常规场景，以AITER为基准。** B1/H16/HK1/Dv128/page64/per-token。

| 场景 / Dq | 实现 | 时间µs | TFLOPS | 逻辑GB/s | 延迟变化 |
|---|---|---:|---:|---:|---:|
| Full / 128 | 当前8wave | 1383.029 | 156.669 | 61.610 | +5.36% |
| Full / 128 | AITER ASM | 1312.689 | 165.064 | 64.911 | 基准 |
| Full / 192 | 当前8wave | 1768.452 | 153.155 | 60.228 | +10.45% |
| Full / 192 | AITER ASM | 1601.170 | 169.156 | 66.521 | 基准 |
| Causal / 128 | 当前8wave | 27022.519 | 162.760 | 10.555 | +3.48% |
| Causal / 128 | AITER ASM | 26114.232 | 168.421 | 10.922 | 基准 |
| Causal / 192 | 当前8wave | 34783.188 | 158.057 | 10.250 | +11.75% |
| Causal / 192 | AITER ASM | 31125.064 | 176.633 | 11.454 | 基准 |

复现每个场景，两种D同时展开：

```bash
bench bf16-full --backend bf16_942 --q 10240 --kv 2583 --dq 128 192 --aiter on
bench bf16-causal --backend bf16_942 --q 32768 --kv 32768 --dq 128 192 --causal --aiter on
# 本轮实际basic命令还包含SWA两shape，专项SWA表统一在6.3。
bench bf16-basic --preset basic --backend bf16_942 swa --dq 128 192 --aiter on --requested-reference auto
```

**B. Dense匹配场景，以指定dense为基准。** B1/H=HK8/Dq=Dv128/NC/unit descales。

| Q / KV / page | 实现 | 时间µs | TFLOPS | 逻辑GB/s | 延迟变化 |
|---|---|---:|---:|---:|---:|
| 20480 / 20480 / 32 | 当前8wave | 9714.742 | 176.843 | 17.270 | +6.30% |
| 20480 / 20480 / 32 | AITER ASM | 9976.223 | 172.208 | 16.817 | +9.16% |
| 20480 / 20480 / 32 | 指定dense | 9139.378 | 187.976 | 18.357 | 基准 |
| 10240 / 2560 / 32 | 当前8wave | 664.944 | 161.479 | 78.847 | +8.38% |
| 10240 / 2560 / 32 | AITER ASM | 651.204 | 164.886 | 80.511 | +6.14% |
| 10240 / 2560 / 32 | 指定dense | 613.544 | 175.006 | 85.452 | 基准 |
| 10240 / 2560 / 64 | 当前8wave | 683.144 | 157.177 | 76.746 | +11.37% |
| 10240 / 2560 / 64 | AITER ASM | 650.924 | 164.957 | 80.545 | +6.12% |
| 10240 / 2560 / 64 | 指定dense | 613.404 | 175.046 | 85.472 | 基准 |

复现三个具体形状：

```bash
bench bf16-dense-default --backend bf16_942 --heads 8 --kv-heads 8 --dq 128 \
  --q 20480 --kv 20480 --page 32 --aiter on --requested-reference on
bench bf16-dense-short-p32 --backend bf16_942 --heads 8 --kv-heads 8 --dq 128 \
  --q 10240 --kv 2560 --page 32 --aiter on --requested-reference on
bench bf16-dense-short-p64 --backend bf16_942 --heads 8 --kv-heads 8 --dq 128 \
  --q 10240 --kv 2560 --page 64 --aiter on --requested-reference on
```

BF16常规场景比AITER慢3.48%–11.75%；dense三形状慢6.30%–11.37%。若沿用先前暂定5%标准，仍均未达标。

### 6.2 FP8：仅LDS，对照指定BN32

统一B1/HK1/Dv128/page64。F1与F5的H、KV均不同，不应仅归因于scale模式。

| 组别 | 模式 | Q / KV | Dq | H | Q scale |
|---|---|---|---:|---:|---|
| F1 | Full | 10240 / 2583 | 128 | 16 | per-token |
| F2 | Full | 10240 / 2583 | 192 | 16 | per-token |
| F3 | Causal | 32768 / 32768 | 128 | 16 | per-token |
| F4 | Causal | 32768 / 32768 | 192 | 16 | per-token |
| F5 | Full | 10240 / 2560 | 128 | 8 | per-tensor |
| F6 | Full | 10240 / 2560 | 192 | 16 | per-tensor |
| F7 | Causal | 32768 / 32768 | 192 | 16 | per-tensor |

| 组别 | 实现 | 时间µs | TFLOPS | 逻辑GB/s | 延迟变化 |
|---|---|---:|---:|---:|---:|
| F1 | 当前LDS | 646.864 | 334.966 | 98.283 | −38.18% |
| F1 | 指定BN32 | 1046.407 | 207.068 | 60.756 | 基准 |
| F2 | 当前LDS | 677.684 | 399.666 | 109.530 | −40.88% |
| F2 | 指定BN32 | 1146.207 | 236.299 | 64.759 | 基准 |
| F3 | 当前LDS | 16420.243 | 267.851 | 12.772 | −11.66% |
| F3 | 指定BN32 | 18588.436 | 236.608 | 11.282 | 基准 |
| F4 | 当前LDS | 17432.468 | 315.373 | 14.075 | −14.65% |
| F4 | 指定BN32 | 20425.148 | 269.165 | 12.013 | 基准 |
| F5 | 当前LDS | 312.122 | 344.014 | 102.885 | −40.87% |
| F5 | 指定BN32 | 527.864 | 203.413 | 60.835 | 基准 |
| F6 | 当前LDS | 649.664 | 413.191 | 114.243 | −41.25% |
| F6 | 指定BN32 | 1105.727 | 242.768 | 67.123 | 基准 |
| F7 | 当前LDS | 17391.292 | 316.119 | 14.109 | −13.46% |
| F7 | 指定BN32 | 20096.349 | 273.568 | 12.210 | 基准 |

复现每个场景：

```bash
# F1/F2
bench fp8-full --backend fp8_942 --q 10240 --kv 2583 --dq 128 192 --aiter off --requested-reference on
# F3/F4
bench fp8-causal --backend fp8_942 --q 32768 --kv 32768 --dq 128 192 --causal --aiter off --requested-reference on
# 本轮F1–F4实际basic命令；AITER适配不支持FP8，auto仅报告N/A。
bench fp8-basic --preset basic --backend fp8_942 --dq 128 192 --aiter auto --requested-reference on
# F5
bench fp8-tensor-full-d128 --backend fp8_942 --q 10240 --kv 2560 --dq 128 --heads 8 \
  --scale-mode per-tensor --aiter off --requested-reference on
# F6
bench fp8-tensor-full-d192 --backend fp8_942 --q 10240 --kv 2560 --dq 192 --heads 16 \
  --scale-mode per-tensor --aiter off --requested-reference on
# F7
bench fp8-tensor-causal-main --backend fp8_942 --q 32768 --kv 32768 --dq 192 --heads 16 --causal \
  --scale-mode per-tensor --aiter off --requested-reference on
```

七项当前LDS均快于BN32。F6的413.191T属于本轮10-buffer/BF16源量化/per-tensor/event协议，
不等于旧native-cast/FlyDSL0.2.2/profiler的400T历史gate复现。

### 6.3 SWA：direct、prepared AITER与每次gather+AITER

统一B1/H16/HK1/Q16384/Dv128/page64/W128/sink/per-token，以**gather+AITER总路径**为基准。
prepared AITER不含gather，仅作分解对照。每组均同输入、10-buffer、单event总区间。

| Dq / KV | 实现 | 时间µs | TFLOPS | 逻辑GB/s | 延迟变化 |
|---|---|---:|---:|---:|---:|
| 128 / 32768 | 当前单wave | 240.762 | 71.914 | 627.154 | −59.50% |
| 128 / 32768 | AITER prepared | 510.424 | 33.921 | 295.823 | −14.13% |
| 128 / 32768 | gather+AITER | 594.404 | 29.129 | 310.478 | 基准 |
| 192 / 32768 | 当前单wave | 276.382 | 78.307 | 682.910 | −57.55% |
| 192 / 32768 | AITER prepared | 568.824 | 38.048 | 331.814 | −12.63% |
| 192 / 32768 | gather+AITER | 651.025 | 33.244 | 354.344 | 基准 |
| 128 / 65536 | 当前单wave | 244.181 | 70.907 | 687.080 | −64.22% |
| 128 / 65536 | AITER prepared | 509.623 | 33.974 | 329.208 | −25.33% |
| 128 / 65536 | gather+AITER | 682.544 | 25.367 | 344.126 | 基准 |
| 192 / 65536 | 当前单wave | 275.481 | 78.563 | 761.268 | −61.94% |
| 192 / 65536 | AITER prepared | 566.583 | 38.198 | 370.140 | −21.73% |
| 192 / 65536 | gather+AITER | 723.845 | 29.900 | 405.613 | 基准 |
| 128 / 131072 | 当前单wave | 243.561 | 71.087 | 826.594 | −70.42% |
| 128 / 131072 | AITER prepared | 509.783 | 33.964 | 394.926 | −38.09% |
| 128 / 131072 | gather+AITER | 823.486 | 21.025 | 407.468 | 基准 |
| 192 / 131072 | 当前单wave | 278.522 | 77.705 | 903.549 | −68.55% |
| 192 / 131072 | AITER prepared | 567.544 | 38.134 | 443.416 | −35.91% |
| 192 / 131072 | gather+AITER | 885.545 | 24.440 | 473.641 | 基准 |

复现全部6项，或把循环改成单个KV复现该场景：

```bash
for kv in 32768 65536 131072; do
  bench "swa-kv$kv" --backend swa --q 16384 --kv "$kv" --dq 128 192 --window 128 --sink --aiter on
done
```

每buffer workspace独立，完整gather+CK两个dispatch，非prepared数据替代总路径。
六项direct比总路径延迟低57.55%–70.42%。不同带宽分子不能替代同shape时间比较。

### 6.4 GPU功能与repeat验证：按类别复现

仅保留分类和复现方法，不再堆砌acc/JSON/硬件日志链接。按§5准备环境。

| 类别 | 本轮结果 | 覆盖 |
|---|---|---|
| 主入口pytest | 保留14项MI308适用测试、2项gfx950跳过 | BF16边界2项、SWA边界2项、FP8 LDS边界8项、live gather2项；CPU工具契约已删除 |
| BF16小shape | 4条候选通过 | D128/192，当前+AITER，10buffer |
| W0+sink SWA | 6条候选通过 | D128/192，direct/prepared/gather+AITER |
| `--repeat 3`专项 | 3种实现各1条汇总结果通过 | SWA direct、prepared AITER、gather+AITER；每个event分别连续调用同一实现3次 |

**repeat3不是3个测试case，也不是3个buffer的配置。** 它是复现命令中`--repeat 3`的简称，
仍使用10个buffer：例如同一实现依次处理buffer `[9,0,1]`，一对event包住这3次调用，
报告时间=该区间总时间÷3。“3条结果”来自上面3种不同实现，与repeat参数恰好同为3无关；
即使改为`--repeat 1`，同样会输出这3种实现的3条汇总结果。
每种实现仍是5轮×10个event=50个样本；repeat3共150个event、450次候选调用。

```bash
# 完整功能：先无限等待GPU空闲，不给等待设置总超时。
"$PY" -c 'import sys; sys.path.insert(0,"tests/flydsl/mha"); from _hardware import wait_until_idle; wait_until_idle()'
"$PY" -m pytest "$MHA" --import-mode=importlib -q --junitxml="$OUT/tests.xml"
# 各类入口和总路径验证。
bench smoke-bf16 --backend bf16_942 --q 65 --kv 129 --dq 128 192 --aiter on
bench smoke-swa-w0 --backend swa --q 65 --kv 129 --dq 128 192 --window 0 --sink --aiter on
bench repeat3 --backend swa --q 65 --kv 129 --dq 128 --window 0 --sink --aiter on --repeat 3
```

已核对每轮覆盖0–9、输入/输出地址独立、每候选50个样本、TFLOPS/带宽与时间一致；
gather额外核对20个K/V workspace地址独立、双dispatch和字节计数。
3400个event包含3700次候选调用（repeat3的150个event各3次）。

## 7. 跨机关注点

先读[changes.md](changes.md)：新旧buffer/轮次/带宽口径不同，不能沿用旧单buffer表。
MI308不验证gfx950；MI350需另跑static/persistent及SWA，不把skip视作pass。
FP8生产文件仅LDS，register请求报错，无静默fallback；三个BF16/SWA生产kernel未改。
10buffer增加显存消耗，OOM应报告，不能自动减少buffer或缩短shape。
