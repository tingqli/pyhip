# pyhip.kit：用一组普通函数描述一个算子

`pyhip.kit` 把一个算子（op）写成一组**同前缀的普通函数**：参考实现、测试输入、结果检查、维度提取，以及若干个实现。只要按命名约定注册，kit 就能从这些函数得到：

- **配置空间**：每个实现的旋钮网格、配置名的打印与解析；
- **pytest 测试**：每个测试用例 × 每个应当能跑的配置；
- **调优**：在 GPU 上跑完整个网格、冷 cache 计时，结果写进 CSV 调优表；
- **分派**：`op(...)` 按调用的维度查表，运行最快的配置；
- **实验工具**：按配置名做 A/B 计时（带噪声带和 verdict）、做正确性检查（对参考、对 base、重复调用、CUDA graph 回放）。

kit 不规定实现用什么写（FlyDSL、Triton、HIP、torch 都行），也不要求声明实现的适用范围：实现跑不了某个调用时自己抛异常（`assert` 就够了），调优时这个配置记为 n/a，不进表。

目录：

1. [模块一览](#1-模块一览)
2. [完整例子：axpy](#2-完整例子axpy)
3. [概念与约定](#3-概念与约定)
4. [Registry 的参数](#4-registry-的参数)
5. [Op 的方法](#5-op-的方法)
6. [分派：查调优表](#6-分派查调优表)
7. [测试：kit.testing](#7-测试kittesting)
8. [调优：kit.tune 与命令行](#8-调优kittune-与命令行)
9. [表文件：kit.table](#9-表文件kittable)
10. [计时：kit.timing](#10-计时kittiming)
11. [A/B 与正确性：kit.bench](#11-ab-与正确性kitbench)
12. [实例：a8w4](#12-实例a8w4)
13. [常见错误](#13-常见错误)

## 1. 模块一览

| 模块 | 依赖 | 内容 |
|---|---|---|
| `pyhip.kit` | 标准库 | `Registry`、`Op`，标记 `param`、`meta`、`no_capture`、`stack`；配置名、查表分派 |
| `pyhip.kit.testing` | pytest（默认检查用 torch） | `op_tests(ops)`：生成 pytest 测试；`assert_close` |
| `pyhip.kit.table` | 标准库 | 调优表 CSV 的 `read`、`write`、`merge`、`identity`、`compare` |
| `pyhip.kit.timing` | torch | 冷 cache 计时：`flush_cache`、`copies`、`time_cold`、`Timer`、`interleaved`、`noise_band`、`verdict` |
| `pyhip.kit.tune` | torch | 调优器和查表工具：`cases`、`tune`、`show`、`verify`、`merge`、`tune_all` |
| `pyhip.kit.bench` | torch | 按配置名做 A/B 计时（`ab`）和正确性检查（`check`），实验记录 |
| `python -m pyhip.kit` | | 命令行：`cases`、`candidates`、`tune`、`merge`、`show`、`verify`、`tune-all`、`compare`（`kit.tune`），`ab`、`check`（`kit.bench`） |

`pyhip.kit` 本身不 import torch：只在 torch 已加载、CUDA 已初始化时才查询当前 stream 是否在 capture。

## 2. 完整例子：axpy

下面的 op 计算 `out = alpha * x + y`，有两个实现：`torch`（一次 `torch.add`）和 `tiled`（按 `block` 个元素分块，最多 64 块）。本节的命令都在放这个文件的目录里运行（`python -m` 会把当前目录加进 `sys.path`）。

### 2.1 定义 op

`axpy_ops.py`：

```python
import os

import torch

from pyhip.kit import Registry, meta, param

ops = Registry("demo", table=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "axpy_tuned.csv"),
               bucket="n", points=(1000, 4096, 1 << 20), unkeyed=("seed",), rotate=("x", "y"))
op_register = ops.register

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MAX_TILES = 64


@op_register
def axpy_ref(x, y, *, alpha=1.0):
    """The interface (x, y, alpha) and the expected answer (fp32)."""
    return alpha * x.float() + y.float()


@op_register
@param("n", (1000, 65536), id="n{}")
@param("dtype", ("bf16", "fp32"), id="{}")
def axpy_test(n, dtype="bf16", seed=0):
    """The inputs of one problem; the parameter names (n, dtype, seed) are the vocabulary."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, device="cuda", generator=g).to(DTYPES[dtype])
    y = torch.randn(n, device="cuda", generator=g).to(DTYPES[dtype])
    return dict(x=x, y=y, alpha=2.0)


@op_register
def axpy_dims(x, y, *, alpha=1.0):
    """The vocabulary of a real call (and the place to check its arguments)."""
    assert x.shape == y.shape and x.dtype == y.dtype, (x.shape, x.dtype, y.shape, y.dtype)
    return dict(n=x.numel(), dtype={v: k for k, v in DTYPES.items()}[x.dtype])


@op_register
def axpy_check(out, expected, **inputs):
    """Raise when out is wrong; the returned metrics go into the table and the reports."""
    err = ((out.float() - expected).abs().max() / expected.abs().max()).item()
    assert out.dtype == inputs["x"].dtype and err < 1e-2, f"rel err {err:.1e}"
    return {"err": err}


@op_register
def axpy_torch(x, y, *, alpha=1.0):
    return torch.add(y, x, alpha=alpha)


@op_register
@param("block", (1024, 16384), id="b{}")
@meta(tiles=lambda n, block: -(-n // block), cached_loads=False)
def axpy_tiled(x, y, *, alpha=1.0, block=1024):
    """One add per tile of ``block`` elements; at most MAX_TILES tiles (a limit: raise)."""
    assert -(-x.numel() // block) <= MAX_TILES, f"n={x.numel()}: > {MAX_TILES} tiles of {block}"
    out = torch.empty_like(x)
    for o, a, b in zip(out.split(block), x.split(block), y.split(block)):
        torch.add(b, a, alpha=alpha, out=o)
    return out
```

要点：

- `axpy_ref` 定义 op `axpy`：参数就是接口（`x, y, alpha`），返回值是期望结果。
- `axpy_test` 的参数名 `n, dtype, seed` 是这个 op 的**词汇**（vocabulary），它们描述“解的是哪个问题”；`@param` 网格给出测试用例。
- `axpy_tiled` 在接口参数之后多了一个仅限关键字的**旋钮** `block`，网格由 `@param` 给出。旋钮描述“这个实现怎么配置”。
- `tiled` 的限制（最多 64 块）不另外声明：跑不了就 `assert`。
- `Registry(...)` 声明调优表、分桶维 `n`、调优点、不进表的词汇 `seed`、计时时轮换的输入 `x, y`（第 4 节）。

### 2.2 查询

```python
>>> from axpy_ops import ops
>>> op = ops.axpy                      # 也可以写 ops["axpy"]
>>> op, op.vocab
(Op('axpy', impls=['torch', 'tiled']), ('n', 'dtype', 'seed'))
>>> op.cases()                         # <op>_test 的网格：(id, 全部参数)
[('n1000-bf16', {'dtype': 'bf16', 'seed': 0, 'n': 1000}), ('n1000-fp32', ...), ...]
>>> [op.name_of(c) for c in op.configs(dict(n=4096, dtype="bf16", seed=0))]
['torch', 'tiled b1024', 'tiled b16384']
>>> op.parse("tiled b16384")
{'kind': 'tiled', 'block': 16384}
>>> op.meta(op.parse("tiled b1024"), "tiles", dict(n=4096))
4
```

命令行同样可以查：

```
$ python -m pyhip.kit cases axpy_ops:axpy          # 调优 case = 测试网格去掉分桶轴 n
bf16 {'dtype': 'bf16', 'seed': 0}
fp32 {'dtype': 'fp32', 'seed': 0}
$ python -m pyhip.kit candidates axpy_ops:axpy bf16
torch
tiled b1024
tiled b16384
```

### 2.3 调优

```
$ python -m pyhip.kit tune axpy_ops:axpy bf16          # 默认：Registry 的 points，写进 Registry 的表
[22:12:50] axpy bf16 {'dtype': 'bf16', 'seed': 0}: 3 candidates, 8 copies of ['x', 'y']
[22:12:51] n   1000: torch 6.0 | tiled b1024 6.0 | tiled b16384 6.0
[22:12:51] n   4096: torch 5.9 | tiled b16384 5.9 | tiled b1024 14.2
[22:12:51] n 1048576: torch 6.2 | tiled b16384 129.6  (1 n/a: AssertionError)
$ python -m pyhip.kit tune axpy_ops:axpy fp32 -t 4096 --out part.csv   # 写到另一个文件
$ python -m pyhip.kit merge axpy_ops:axpy part.csv                     # 合并进 Registry 的表
merged 3 rows from 1 files into /tmp/kitdemo/axpy_tuned.csv
```

`n = 1048576` 时 `tiled b1024` 需要 1024 块，`assert` 失败，记为 n/a。得到的表（列由词汇推出：key 列 `dtype`、分桶列 `n`、`config`、`us`、`axpy_check` 返回的 `err`）：

```
dtype,n,config,us,err
bf16,1000,torch,6.0,2.2e-03
bf16,1000,tiled b1024,6.0,2.2e-03
bf16,1000,tiled b16384,6.0,2.2e-03
bf16,4096,torch,5.9,2.0e-03
bf16,4096,tiled b16384,5.9,2.0e-03
bf16,4096,tiled b1024,14.2,2.0e-03
bf16,1048576,torch,6.2,3.0e-03
bf16,1048576,tiled b16384,129.6,3.0e-03
fp32,4096,torch,6.0,0.0e+00
...
```

查看和验证：

```
$ python -m pyhip.kit show axpy_ops:axpy bf16          # 每个调优点：最快的配置、每种实现最快的配置
| n | best | us | best torch | us | best tiled | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|
| 1000 | torch | 6.0 | torch | 6.0 | tiled b1024 | 6.0 |  |
| 4096 | torch | 5.9 | torch | 5.9 | tiled b16384 | 5.9 |  |
| 1048576 | torch | 6.2 | torch | 6.2 | tiled b16384 | 129.6 |  |
$ python -m pyhip.kit verify axpy_ops:axpy bf16 -t 10 1000 3000 2000000   # 经 op(...) 分派，也测调优点之间的值
[22:12:58] n     10 -> point   1000: torch          5.9 us (table      6.0) | err 3.1e-03
[22:12:58] n   3000 -> point   4096: torch          6.1 us (table      5.9) | err 3.4e-03
[22:12:58] n 2000000 -> point 1048576: torch        7.2 us (table      6.2) | err 2.6e-03
```

多张卡一起调所有 case，再和旧表比较：

```
$ python -m pyhip.kit tune-all axpy_ops:axpy --gpus 0-1 --per-job 1 --out /tmp/kitdemo/all
[22:14:59] bf16: GPUs [0], 3 points
[22:14:59] fp32: GPUs [1], 3 points
[22:15:05] all jobs done in 0.1 min
no BAD candidates
merged 16 rows from 2 files into /tmp/kitdemo/all/axpy_tuned.new.csv
next: python -m pyhip.kit compare /tmp/kitdemo/axpy_tuned.csv /tmp/kitdemo/all/axpy_tuned.new.csv
      cp /tmp/kitdemo/all/axpy_tuned.new.csv /tmp/kitdemo/axpy_tuned.csv
$ python -m pyhip.kit compare axpy_tuned.csv /tmp/kitdemo/all/axpy_tuned.new.csv
dtype=bf16 n=1000  old torch   6.0  new torch   5.9  old-cfg-now   5.9 +0.0%
dtype=fp32 n=1000  new torch   5.9  (no old row)
...
0 of 6 points changed winner (*); ...
```

### 2.4 分派

```python
import torch
from axpy_ops import ops

x = torch.randn(3000, device="cuda", dtype=torch.bfloat16)
y = torch.randn_like(x)
out = ops.axpy(x, y, alpha=2.0)                          # dims -> {n: 3000, dtype: bf16} -> 调优点 4096 最快的配置
out = ops.axpy(x, y, alpha=2.0, config="tiled b1024")    # 指定配置（名字或 dict），不查表
ops.axpy.pick(dict(n=3000, dtype="bf16", seed=0))        # {'kind': 'torch'}
[ops.axpy.name_of(c) for c in ops.axpy.tuned(dict(n=3000, dtype="bf16", seed=0))]
# ['torch', 'tiled b16384', 'tiled b1024']：pick 的候选，最快的在前
ops.axpy(x, y, table=None)
# ValueError: axpy: no tuned config for {'n': 3000, 'dtype': 'bf16'} (no table): tune it, or pass config=
```

### 2.5 测试

`test_axpy.py`：

```python
from axpy_ops import ops

from pyhip.kit.testing import op_tests

test_axpy = op_tests(ops)
```

```
$ python -m pytest -q test_axpy.py --co
test_axpy.py::test_axpy[axpy-n1000-bf16-dims]
test_axpy.py::test_axpy[axpy-n1000-bf16-torch]
test_axpy.py::test_axpy[axpy-n1000-bf16-tiled b1024]
test_axpy.py::test_axpy[axpy-n1000-bf16-tiled b16384]
...
test_axpy.py::test_axpy[axpy-n65536-bf16-torch]
test_axpy.py::test_axpy[axpy-n65536-bf16-tiled b16384]   # 用 1048576 点的行：tiled b1024 没进表，不测
...
$ python -m pytest -q test_axpy.py -k "bf16 and tiled"
```

有调优表时，每个用例测的是分派可能选到的配置（`op.tuned(case)`）；表还不存在时只有 `-dims` 项。

### 2.6 A/B 计时和正确性检查

```
$ python -m pyhip.kit ab axpy_ops:axpy bf16 -t 4096 1000000 -v @table "tiled b16384" --rounds 3 --out exp
#### 1000000 n (seed 1000007)

| variant | config | protocol | us | delta % | rounds min-max | verdict | err ref | err base |
|---|---|---|---:|---:|---:|---|---:|---:|
| @table | `torch` | coldrot | 6.2 | +0.0 | 6.2-6.3 | base | 2.9e-03 | 0 |
| @table#2 | `torch` | coldrot | 6.2 | +0.3 | 6.2-6.3 | noise floor | 2.9e-03 | 0 |
| tiled b16384 | `tiled b16384` | coldrot | 6.3 | +1.0 | 6.2-6.3 | slower | 2.9e-03 | 0 |
| *noise band +-1.0%* |  |  |  |  |  |  |  |  |

$ python -m pyhip.kit check axpy_ops:axpy fp32 -t 1000 1000000 -v @table "tiled b1024" "mine=py:mine.py:axpy" --graph --diff-tol 1e-6
| n | variant | config | err ref | err base | repeat | bitwise | graph | result |
|---:|---|---|---:|---:|---:|---|---|---|
| 1000 | @table | `torch` | 0 | 0 | 0 | - | 0 | OK |
| 1000 | tiled b1024 | `tiled b1024` | 0 | 0 | 0 | - | 0 | OK |
| 1000 | mine | `py:mine.py:axpy` | 0 | 0 | 0 | - | 0 | OK |
| 1000000 | tiled b1024 | `tiled b1024` | - | - | - | - | - | FAIL AssertionError: n=1000000: > 64 tiles of 1024 |
...
1 FAILED
```

`mine.py` 里的 `axpy(x, y, *, alpha=1.0)` 是任意一个接受接口参数的函数（例如实验中的 kernel 副本）。`check` 有失败项时退出码为 1。

## 3. 概念与约定

### 3.1 函数的角色

函数名决定角色。`<op>_ref` 定义 op `<op>`，其余 `<op>_<后缀>` 加入已注册的、前缀最长的 op（`moe_dyn_ep3k` 在 `moe_dyn_ref` 注册后属于 `moe_dyn`，不属于 `moe`）。

| 函数 | 角色 | 签名 | 用途 |
|---|---|---|---|
| `<op>_ref` | 接口 + 参考实现（必需，最先注册） | 只有具名参数（不能有 `/`、`*args`、`**kwargs`，不能叫 `config` / `table`） | 参数就是接口；返回期望结果 |
| `<op>_test` | 测试输入（在实现之前注册） | 只有具名参数；没有默认值的参数必须有 `@param` | 参数名是词汇；返回一个问题的接口参数 dict；`@param` 网格是测试用例 |
| `<op>_check` | 结果检查（可选） | `(out, expected, **inputs)`，前两个按位置传 | 结果错就抛异常；可返回误差指标 dict（第一项是工具报告的列）。默认 `torch.testing.assert_close`（先把 expected 转成 out 的 dtype） |
| `<op>_dims` | 调用的词汇（分派需要） | 接受 ref 的参数 | 返回 `{词汇名: 值}`；`op(...)` 每次调用都会调用它（包括传了 `config=` 时），所以这里也是检查参数的地方 |
| `<op>_<kind>` | 实现 `kind` | 先是和 ref **完全相同**的参数（名字、种类、默认值），然后是带默认值的仅限关键字旋钮，最后可选 `**over` | 计算结果 |

后缀 `ref`、`test`、`check`、`dims` 是保留的；实现名也不能以 `_test` / `_check` / `_dims` 结尾。以 `_` 开头的函数不注册。

### 3.2 词汇和旋钮

- **词汇**（vocabulary）：`<op>_test` 的参数名，描述一个具体问题（形状、dtype、路由方式、seed……）。`<op>_dims` 从真实调用里算出词汇值；调优表用词汇做 key；`@meta` 函数和 `@param` 的取值函数可以用词汇。
- **旋钮**（knob）：实现的仅限关键字参数，是这个实现的可调选项。每个旋钮正好有一个 `@param`；旋钮名不能和词汇或 `kind` 重名。

一个 `vocab` 参数（`configs`、`pick`、`tuned`、`meta` 等）是一个包含**全部**词汇名的 dict，未知的值写 None。

### 3.3 配置和配置名

配置是 dict：`{"kind": 实现名, 旋钮: 值, ...}`。配置名把 kind 和各个 `@param` 的 id 用空格连起来，例如 `"tiled b1024"`、`"tp2k bm32 g1bcm2 g2nt4"`；带 `**over` 的实现可以在名字后面接覆盖项 `":k=v,k=v"`（例如 `"ep3k m64:g2.pf=1"`）。

- `op.name_of(cfg)` 打印名字，`op.parse(name)` 解析回来（旋钮按默认值的类型转换；覆盖项转成 int、float 或 str）。
- 注册时会检查每个静态网格值打印后能读回原值。因此 id 里的取值不能含空格或 `:`。

### 3.4 限制：实现自己抛异常

kit 不提供“这个配置能不能跑”的声明。实现遇到跑不了的调用时抛异常（`assert` 写清原因即可）：

- 调优时这个配置记为 n/a，不写进表；
- 分派只会选表里的配置，所以**只对调优过的 key 保证能跑**；没有调优过的 key 抛 `ValueError`，提示去调优或传 `config=`；
- 测试只测表里的配置；`bench ab` 把抛异常的变体列在 dropped 里，`bench check` 记为 FAIL 并写出异常。

分桶维（第 6 节）上的取值落在调优点之间时，用的是 ≥ 调用值的第一个调优点的配置。调用值不会超过所用的调优点（超出最大调优点时除外），所以分桶维上的**上限**型限制（例子里的“最多 64 块”）是安全的；**下限**型限制（例如要求 `n >= block`）在两个调优点之间可能失败。这类限制要在实现内部处理，或者把对应的维度作为 key 而不是分桶维。

### 3.5 标记

标记只把自己记在 `f.kit_marks` 里并原样返回 `f`；`@op_register` 必须写在最外层（它注册时检查下面的标记，写在它上面的标记会报错）。

**`@param(names, values, id=None)`**：网格的一个轴。多个 `@param` 取笛卡尔积，写在最上面的是最外层。

```python
@param("bm", (32, 64), id="bm{}")                    # 一列值；id 是只有一个 {} 的格式串（默认 name + "{}"）
@param("act", ("silu", "gelu"), id="{}")
@param("fp8", {"bf16": False, "fp8": True})          # dict：key 就是 id
@param("tokens, valid", {"v0": (512, 0), "v37": (512, 37)})   # 多个名字：必须用 dict
@param("split", lambda dim: (1, 2, 4))               # 词汇的函数（只能用在实现的旋钮上）
```

**`@meta(**items)`**：给工具读的元数据，用 `op.meta(cfg, key, vocab)` 读。值可以是常量，也可以是词汇名和旋钮的函数：

```python
@meta(tile_m=lambda block_m: block_m, cached_loads=lambda g1_bcm: g1_bcm != 2)
```

带 `**over` 的 meta 函数还会收到配置里的覆盖项（见下面 `stack` 的例子）。kit 自己读的只有 `cached_loads`（`kit.bench` 用它判断计时协议是否会偏向这个配置，第 11 节）；其他 key 由使用方约定（例如 a8w4 的 `tile_m`）。

**`@no_capture(reason)`**：这个实现不能在 CUDA graph capture 中运行（例如把 host 计数器作为 launch 参数传入）。当前 stream 正在 capture 时，分派跳过它，选下一个最快的；`op.capturable(cfg)` 返回 False；`bench check --graph` 把它记为 n/a。

**`stack(*marks)`**：把几个标记合成一个，用于几个实现共用的配置空间：

```python
ep3k_space = stack(  # a8w4 moe.py：moe_ep3k 和 moe_dyn_ep3k 共用
    param("cfg", EP_CFGS, id="{}"),
    meta(tile_m=lambda cfg, **over: ep_preset(cfg, over)["bm"],
         cached_loads=lambda cfg, **over: _ep_cached(ep_preset(cfg, over))))

@op_register
@ep3k_space
def moe_ep3k(x, W, topk_weights, topk_ids, *, expert_offset=0, num_experts=None, act="silu",
             situ=None, cfg="t16", **over): ...
```

### 3.6 注册

```python
ops = Registry()             # 每个包一个，放在一个小模块里（例如 _ops.py），注册函数的模块都 import 它
op_register = ops.register
```

- `Registry.register(f)` 按函数名把 `f` 加进它的 op，检查约定，原样返回 `f`。违反约定时抛 `TypeError`，op 保持不变。
- 顺序：`<op>_ref` 最先；`<op>_test` 在所有实现之前（实现要按它的词汇检查）；嵌套的 op（`moe_dyn`）要在它的实现之前注册 `moe_dyn_ref`。
- 读取：`Registry` 是 `{op 名: Op}` 的 Mapping（按注册顺序），也可以用属性：`ops.moe is ops["moe"]`。
- 同一个包被以两个模块名导入时，会有两个互不相干的注册表。

## 4. Registry 的参数

```python
Registry(name=None, *, table=None, bucket=None, relax=None, points=None, unkeyed=(), rotate=())
```

| 参数 | 谁用 | 含义 |
|---|---|---|
| `name` | 报告 | 注册表的名字 |
| `table` | 分派、测试、tune、bench | 调优表 CSV 的路径；None 表示没有表（`op(...)` 必须传 `config=`，测试跑整个网格） |
| `bucket` | 分派、tune | 分桶的词汇名（例如 token 数）：表里按这个维度存若干调优点，查表时取 ≥ 调用值的第一个点 |
| `relax` | 分派 | `{词汇名: 值}`：一个 key 没有任何行时，用这些值替换后再查一次（a8w4：没有单独调过的激活函数用 SiLU 的行） |
| `points` | tune | 分桶维上要调优的点（`tune` / `tune-all` 的默认 `-t`） |
| `unkeyed` | tune | 不写进表的词汇名（不影响哪个配置最快，例如 seed、expert_offset） |
| `rotate` | tune、bench | 计时时轮换副本的接口参数（通常是权重）：每次调用读一份最近没读过的副本，见第 10 节 |

## 5. Op 的方法

| 方法 / 属性 | 说明 |
|---|---|
| `name`、`ref`、`test`、`check`、`dims` | 名字和各角色的函数（没有时为 None） |
| `impls` | `{kind: 实现函数}`（只读） |
| `vocab` | 词汇名（`<op>_test` 的参数名）的 tuple |
| `registry` | 所属的 `Registry` |
| `cases()` | `[(id, case)]`：`<op>_test` 的网格，`case` 含全部参数（默认值已填），id 用 `-` 连接各轴的 id |
| `configs(vocab)` | 所有实现的所有网格点，按定义顺序（`vocab` 提供 `@param` 取值函数用到的词汇）；能不能在 `vocab` 上跑，要跑了才知道 |
| `name_of(cfg)` / `parse(name)` | 配置 ↔ 配置名 |
| `run(cfg, *args, **kwargs)` | 用 `cfg` 的旋钮（和覆盖项）调用对应实现 |
| `meta(cfg, key, vocab=None)` | `@meta` 的 `key`；函数在 `vocab`、旋钮和覆盖项上求值 |
| `capturable(cfg)` | 实现有没有 `@no_capture` |
| `tuned(vocab, table=)` | 查表得到的候选配置，最快的在前；没有行时为 `[]` |
| `pick(vocab, table=, capturing=False)` | `op(...)` 会运行的配置：`tuned` 里第一个能在当前 capture 状态下运行的；没有时 `ValueError` |
| `points(vocab, table=)` | `[(分桶值, [(us, cfg), ...])]`：这个 key 的所有调优点（不用 relax） |
| `op(*args, config=None, table=..., **kwargs)` | 运行 op，见第 6 节 |

`table=` 参数：默认用注册表的表，也可以传另一个路径，或 None（不用表）。

## 6. 分派：查调优表

`op(*args, **kwargs)`：

1. 调用 `<op>_dims(*args, **kwargs)` 得到词汇值；关键字参数里不是接口参数、但是词汇名的，作为**提示**一起参与查表（a8w4：`ops.moe(x, W, tw, ti, routing="hot")`）。既不是接口参数也不是词汇的关键字参数会报 `TypeError`。
2. `pick` 选配置，然后运行。
3. 传 `config=`（dict 或名字）时不查表，但仍调用 `<op>_dims` 检查参数。没有 `<op>_dims` 时只能传 `config=`。

`pick(vocab)` 查表的规则：

- 表是 CSV。列名是词汇名的列是 key（分桶列除外）；`config` 列是配置名，`us` 列是时间；其余列忽略。配置的 kind 不是这个 op 的实现时，这一行跳过（几个 op 可以共用一张表）。
- `vocab` 中值为 None 的 key 按 `<op>_test` 的默认值匹配。
- 先找 `vocab` 这个 key 的行；没有时用 `relax` 替换后再找一次。
- 有分桶维时，取 ≥ `vocab[bucket]` 的第一个调优点；比所有点都大时取最大的点。这个点上的行按 `us` 从快到慢就是 `tuned(vocab)`。
- `pick` 返回第一个；当前 stream 在 capture 时跳过 `@no_capture` 的实现。都没有时抛 `ValueError`，消息里写明是没有调优行，还是所有候选都不能 capture。

缓存：

- `pick` 按（表、表的 mtime、词汇值、capture 状态）缓存；返回的 dict 是共享的，**不要修改**。
- `op(...)` 按（表、mtime、capture 状态、`<op>_dims` 的值和提示）缓存选中的实现。
- 每张表的 mtime 每秒最多读一次：改了表之后一秒内生效，不用重启进程。

## 7. 测试：kit.testing

```python
from pyhip.kit.testing import op_tests

test_ops = op_tests(ops)      # 测试模块里的一行；ops 是 Registry（或任何 {名字: Op} 的 mapping）
```

对每个有 `<op>_test` 的 op、它网格里的每个用例，生成：

- `<op>-<case>-dims`：`<op>_dims(**inputs)` 和用例一致（有 `<op>_dims` 时）；
- `<op>-<case>-<配置名>`：`op.run(cfg, **inputs)`，再用 `<op>_check` 和 `<op>_ref` 比较。有调优表时，配置是 `op.tuned(case)`（分派可能选到的那些；没有行就没有这些项）；没有表时是整个网格（都必须能跑）。

用例逐个运行，当前用例的输入和参考结果只算一次，换到下一个用例时释放。用 `-k` 选择，例如 `-k "tp8 and tiled"`。

## 8. 调优：kit.tune 与命令行

```
python -m pyhip.kit <命令> <模块>:<op> [参数...] [选项]
```

`<模块>:<op>`：一个可导入的模块，里面正好有一个 `Registry`，加上 op 名，例如 `a8w4:moe`、`axpy_ops:axpy`。

**调优 case**：`<op>_test` 的网格去掉分桶轴后的每个点，id 由其余各轴的 id 用 `-` 连接（例如 `tp8-silu`）。`--set name=value` 改这个 case 的词汇值（换形状、换路由），详见下面的“调优时的词汇从哪来”。

| 命令 | 用法 | 作用 |
|---|---|---|
| `cases` | `cases M:OP` | 列出调优 case |
| `candidates` | `candidates M:OP CASE [-t V]` | 列出网格配置（取值函数用第一个点或 `-t` 的值） |
| `tune` | `tune M:OP CASE [-t V ...] [--out T.csv]` | 在每个点上调优，合并进 `--out`（默认 `--table`，再默认注册表的表） |
| `merge` | `merge M:OP A.csv B.csv ...` | 把若干张表合并进 `--table`（默认注册表的表） |
| `show` | `show M:OP CASE [--table T]` | Markdown 表：每个调优点最快的配置、每种实现最快的配置、capture 时的选择（与最快的不同时） |
| `verify` | `verify M:OP CASE [-t V ...]` | 经 `op(...)` 分派运行（也测调优点之间的值），检查结果，计时并和表里的时间对比 |
| `tune-all` | `tune-all M:OP [CASE ...] --gpus 0-7 --per-job 4 --out DIR` | 多卡调所有（或指定的）case，合并成 `DIR/<表名>.new.csv` |
| `compare` | `compare OLD.csv NEW.csv` | 每个调优点：新旧表的最快配置，以及沿用旧配置在新表中的代价 |
| `ab`、`check` | 见第 11 节 | |

通用选项：`-t/--points`（分桶值）、`--set NAME=VALUE ...`、`--table`、`--out`、`--iters`（每次计时的调用数，默认 20）、`--rounds`（默认 3）、`--copies`（轮换副本数，默认 8）、`--gpus`、`--per-job`、`--force`。

**调优时的词汇从哪来**。一次调优的全部词汇值有三个来源，后面的覆盖前面的：

1. **tune case**：`<op>_test` 的 `@param` 网格去掉分桶轴后的一个点；不在网格里的词汇取 `<op>_test` 的默认值。`cases` 命令列出全部 case 和它们的值（分桶维为 None）。
2. **`--set name=value ...`**：覆盖 case 里任意一个词汇。名字必须是词汇；旋钮不能 set，旋钮由网格枚举。值按 `<op>_test` 默认值的类型转换；没有默认值时依次尝试 int、float，否则当作 str。
3. **分桶维**：取 `-t` 的每个值，不写 `-t` 时用 `Registry(points=)`；即使对分桶维用了 `--set` 也会被覆盖。

```
$ python -m pyhip.kit cases axpy_ops:axpy
bf16 {'dtype': 'bf16', 'seed': 0}
fp32 {'dtype': 'fp32', 'seed': 0}
$ python -m pyhip.kit tune axpy_ops:axpy bf16 --set dtype=fp32 -t 2048 --out set.csv
[23:01:38] axpy bf16 {'dtype': 'fp32', 'seed': 0}: 3 candidates, 8 copies of ['x', 'y']
[23:01:39] n   2048: torch 6.1 | tiled b16384 6.2 | tiled b1024 8.4
$ cat set.csv
dtype,n,config,us,err
fp32,2048,torch,6.1,0.0e+00
...
$ python -m pyhip.kit tune axpy_ops:axpy bf16 --set block=4
--set 'block=4': expected name=value, name in ['n', 'dtype', 'seed']
```

`candidates`、`show`、`verify`、`tune-all`（作用于每个 case）和 `ab`、`check` 也接受同样的 `--set`。a8w4 的写法相同，例如 `python -m a8w4.tune tune tp8-silu --set routing=hot -t 1 2 4 --out /tmp/p.csv`。在 Python 里，把 case dict 直接传给 `tune`：

```python
from pyhip.kit import tune as kt

c = kt.case(op, "bf16", ["dtype=fp32"])        # {'dtype': 'fp32', 'seed': 0, 'n': None}
kt.tune(op, c, [2048], "out.csv")              # 也可以手写 c，但要包含全部词汇名
```

注意：

- **只有部分词汇是 key**：表的 key 列是词汇去掉 `unkeyed` 和分桶维后剩下的。`--set` 一个 unkeyed 的名字（例如 a8w4 的 `expert_offset`）只改变输入，不改变 key，所以它必须确实不影响哪个配置最快。
- **`seed` 不能用 `--set` 改**：调优器会把它改写成 101 × 点 + 轮次。
- **调出来的行要能被分派选中**：调用时的 key 来自 `<op>_dims` 的返回值，加上调用时传的提示参数；`<op>_dims` 不返回的名字按 `<op>_test` 的默认值匹配（第 6 节）。所以用 `--set routing=hot` 调出来的行，只有 `ops.moe(..., routing="hot")` 这样的调用才会选中。
- **长期使用的值放进网格**：`--set` 适合临时、一次性的调优。一个 shape 要长期保留时，把它加进 `<op>_test` 的 `@param` 网格：它会成为新的 tune case，`tune-all` 会自动调它，生成的测试也会覆盖它。

**`tune` 的过程**（每个点）：

1. 跑 `--rounds` 轮，每轮用新的输入（`<op>_test`，seed = 101 × 点 + 轮次；`seed` 是词汇时才传）。
2. 每轮里每个网格配置运行一次：抛异常记为 n/a（这个点上不再尝试）；`<op>_check` 失败记为 BAD；通过的用 `kit.timing.time_cold` 冷 cache 计时，`rotate` 输入在 `--copies` 份副本间轮换。各配置交错计时。
3. 跑通并通过检查的配置都写进表：`us` 是各轮的中位数，指标列是各轮最差的值。
4. 日志每个点一行：最快的三个配置、BAD 的原因、n/a 的个数和异常类型。

**表头**由词汇推出：key 列（词汇顺序，去掉 `unkeyed` 和分桶维）、分桶列、`config`、`us`、`<op>_check` 返回的指标。

**`verify`**：输入 seed = 7 + 值；每个值打印 `值 -> 调优点: 配置 时间 (table 表中时间) | 指标`；默认值为 1、3、20、100、300、1000、3000、10000。

**`tune-all`**：

- 先检查 GPU 是否空闲（3 次 rocm-smi 采样，GFX 使用率 > 10% 就退出；`--force` 跳过），再拿每张卡的 `/tmp/gpu-<i>.lock`。
- 每 `--per-job` 张卡一组，每组调一个 case：组内第 g 张卡负责 `points[g::per_job]`。组空出来后接着做下一个 case。
- 每个子进程是一条 `python -m pyhip.kit tune ...`，输出 `DIR/<case>_<g>.csv` 和 `.log`。
- 全部完成后汇总所有日志里的 BAD 行，合并出 `DIR/<表名>.new.csv`，并打印下一步的 `compare` 和 `cp` 命令。

**Python API**：`kit.tune.resolve("m:op")`、`cases(op)`、`case(op, cid, sets)`、`fields(op, metrics)`、`tune(op, case, points, out, rounds=, iters=, ncopies=, log=)`、`show(op, case, path)`、`verify(op, case, points, path)`、`merge(parts, path)`、`tune_all(...)`、`gpu_list("0-3,6")`、`busy_gpus(gpus)`、`lock_gpus(gpus)`。

## 9. 表文件：kit.table

调优表是 CSV：key 列、分桶列、`config`，然后是测量值（`us` 和各项指标，即 `config` 之后的所有列）。`config` 及之前的列是一行的**身份**：写入同身份的行会替换旧行。行按身份（去掉 config，数字按数值比较）排序，同一个点内按 `us` 排序，即最快的在前。

```python
from pyhip.kit import table

fields, rows = table.read(path)           # 表头，行（{列: str}）
table.identity(fields)                    # config 及之前的列
table.write(path, rows, fields)           # 排序后原子替换
table.merge(path, rows, fields=None)      # 加入 / 替换行；文件不存在时创建
table.compare(old, new, label=None, order=None, out=print)   # 返回最快配置变了的点数
```

## 10. 计时：kit.timing

目标是测出算子**在模型里**的耗时：模型的每一层读不同的权重，两次调用同一层之间 cache 里不会留下它的数据。规则和依据见 skill [gpu-timing-protocol](../../.claude/skills/gpu-timing-protocol/SKILL.md)。

```python
from pyhip.kit.timing import copies, time_cold, Timer, interleaved, noise_band, verdict
import itertools

Ws = copies(W)                                     # NCOPIES = 8 份副本（tensor，或 dict 里的 tensor）
nxt = itertools.cycle(Ws).__next__
us = time_cold(lambda: fn(nxt()))                  # GPU 时间的中位数（us），不含 host 开销

timer = Timer(Ws)                                  # 按协议计时，fn 接受这次要读的副本
res = interleaved({"a": fa, "a#2": fa, "b": fb}, ["coldrot"], timer, rounds=5, iters=16)
band = noise_band(res["coldrot", "a"], res["coldrot", "a#2"])
verdict(delta_pct, band)                           # "~" / "faster" / "slower"
```

| 协议 | 做法 | 用途 |
|---|---|---|
| `coldrot` | 读 512 MB 刷 L2，GPU sleep，每次调用读下一份副本 | **唯一和模型一致的协议**（副本数 ≥ `MIN_COPIES` = 8） |
| `rotate` | 副本连续调用，不刷 | 受上一次调用留在 MALL 里的数据影响 |
| `cold` | 刷 L2 + sleep，每次读同一份输入 | MALL 保留输入，走 cache 读的实现会显得快 5–13% |
| `bigread` | 读 2 GB 刷 + sleep，同一份输入 | 偏差同 `cold` |
| `wflush` | 写 512 MB 再读，同一份输入 | 留下脏行，走 cache 读的实现显得慢 9–48% |
| `warm` | 不刷，同一份输入 | |

- `flush_cache()`：读 512 MB。这能用干净的行挤掉 L2，但挤不掉每次调用都会再读的、留在 256 MB MALL 里的数据，所以要靠副本轮换。
- `time_cold(fn, iters=20, warmup=3, flush=True)`：刷 cache 之后让 GPU sleep 约 1 ms，host 趁这段时间把 `fn` 的 launch 都排进队列，所以测到的时间不含 host 开销（和 CUDA graph 回放一样）。
- `interleaved`：每轮把每个（协议, 变体）各测一次，隔一轮把变体顺序反过来（ABBA），使时钟和功耗漂移对所有变体的影响相同。返回 `{(协议, 标签): [每轮的中位数]}`。
- `noise_band(base, dup)`：噪声带（%），取 dup 相对 base 的差、base 各轮极差的一半、`MIN_NOISE` = 0.3% 三者中的最大值。
- `verdict(delta, noise)`：差值在噪声带内为 `"~"`，否则为 `"faster"` / `"slower"`。

## 11. A/B 与正确性：kit.bench

```
python -m pyhip.kit ab    M:OP CASE -t V [V ...] -v SPEC [SPEC ...] [选项]
python -m pyhip.kit check M:OP CASE -t V [V ...] -v SPEC [SPEC ...] [选项]
```

case、`--set` 和输入都和 `kit.tune` 相同：每个 `-t` 值作为分桶值，输入由 `<op>_test` 生成（seed = `--seed` + 值，ab 的第 s 个样本再加 1000 × s）。参考结果来自 `<op>_ref`，比较用 `<op>_check`（它返回的第一个指标就是报告里的列，例如 `err`、`cos`）。

**变体写法**：

```
[label=]<base>[:<k>=<v>,...]      base：配置名，或 "@table"（每个值上表里最快的配置）
[label=]py:<module | file.py>:<func>
```

- 覆盖项 `k=v` 作为配置项传给实现（实现要有 `**over`，或 k 是它的旋钮）；`@meta` 也能看到这些覆盖项。v 的类型：int、float、`true`/`false`、`none`；含 `/` 的转成 tuple（`a/b/c`）；`-` 原样保留；其余作为字符串。
- `py:` 变体是任意函数 `func(**接口参数)`，用来跑实验中的副本。文件路径的模块以私有名字加载，不会遮住同名的标准库模块。
- `label` 是表格里的名字，默认就是整个写法。第一个变体是 base。

**`ab` 的过程**：

1. 先解析所有值上的所有变体：base 在某个值上没有配置时直接退出；不安全的计时协议也在这一步拒绝（见下）。
2. 对每个值和每个样本，每个变体先运行一次：和参考比较（`<op>_check` 失败记 BAD，抛异常记 FAIL，这两种都会被丢弃并列在 dropped 里），再和 base 比较（`--diff-tol` 之上标 DIFF）。
3. 然后做 `--rounds` 轮 ABBA，每轮每个（协议, 变体）调用 `--iters` 次。base 会以 `<base>#2` 的名字再测一份，它和 base 的差就是噪声下限（`--no-dup` 关闭）。
4. 表格列：`us`（各轮中位数的中位数）、`delta %`、各轮的最小–最大值、verdict（只在噪声带之外才判 faster / slower）、对参考和对 base 的指标、`--bitwise` 时是否逐位相同。
5. 多个协议时检查各协议的 verdict 是否一致，不一致标 PROTOCOL-SENSITIVE（以 coldrot 为准）。多个值或多个样本时最后输出汇总表。

**协议检查**：`@meta(cached_loads=True)` 的实现、没有这项 meta 的实现和 `py:` 变体都可能通过 cache 读轮换的输入；除非某个协议是 `coldrot` 且 `--copies` ≥ 8，否则拒绝计时（`--allow-biased` 只给警告，用于专门测量这个偏差）。只有 `cached_loads=False` 的实现不受限制。

**`check`**：对每个值和每个变体，检查对参考（失败：`ref`）、对 base（`base`）、`--repeats` 次重复调用中和第一次最差的差（`repeat`，能发现没复位的状态）、`--bitwise` 时与 base 是否逐位相同、`--graph` 时 CUDA graph capture 后回放与 eager 的差（`@no_capture` 记为 n/a）。`--diff-tol` 是对 base / repeat / graph 允许的最大指标值。`@table` 在没有调优行的值上记为 n/a，不算失败。有失败项时退出码为 1。

**选项**：`--seed`（默认 7）、`--diff-tol`、`--bitwise`；ab：`--protocol`（默认 coldrot，可给多个）、`--copies`、`--rounds`（5）、`--iters`（16）、`--samples`（1）、`--no-dup`、`--allow-biased`、`--no-ref`；check：`--repeats`（5）、`--graph`。

**输出**：屏幕上打印 Markdown；`--out DIR` 时追加到 `DIR/<name>.md` 和 `DIR/<name>.jsonl`（`--name`，默认是命令名）。jsonl 的第一行是运行元数据 `{"meta": ...}`（命令、git 版本、torch / FlyDSL 版本、GPU、功耗上限、参数），之后每个测量一行（`T`、`variant`、`config`、`us`、`rounds`、`delta_pct`、`noise_pct`、`verdict`、`<指标>_ref`、`<指标>_base`、`copies` 等）。

**Python API**（包装成某个包自己的工具时用）：

```python
from pyhip.kit import bench as kb

v = kb.parse_variant(op, "pf1=ep3k m64:g2.pf=1", alias=None)   # -> Variant
r = kb.resolve(op, v, vocab)            # -> Resolved(label, spec, config, run(inputs), cfg, cached)，或 str（没有调优配置）
kb.ab(op, case, points, specs, protocols=("coldrot",), rounds=5, iters=16, samples=1, seed=7,
      dup=True, allow_biased=False, ref=True, bitwise=False, diff_tol=None, ncopies=8, rec=None,
      inputs=None, annotate=None, alias=None, unit=None, describe=None)
bad = kb.check(op, case, points, specs, seed=7, repeats=5, graph=False, bitwise=False,
               diff_tol=None, rec=None, inputs=None, alias=None, unit=None, describe=None)
rec = kb.Recorder(out_dir, name, kb.run_meta(tool, args, prog="python -m mypkg.bench", path=pkg_dir))
```

几个钩子：

- `inputs(value, seed) -> 接口参数 dict`：替换 `<op>_test`，例如生成特殊的路由分布。
- `annotate(inputs, resolved) -> (标题后缀, 记录字段)`：给每个样本加说明，例如路由到本地 expert 的行数。
- `alias(base) -> 配置名`：base 的简写，例如 a8w4 允许直接写 ep3k 的预设名 `l128`。
- `unit`：标题里分桶值的单位名，例如 `tokens`。
- `describe`：开头打印的 case 说明。

## 12. 实例：a8w4

`src/contrib/flydsl/a8w4` 是完整的使用者：

- [`_ops.py`](../contrib/flydsl/a8w4/_ops.py)：`Registry("a8w4", table=..., bucket="num_tokens", relax={"act": "silu"}, points=TOKENS, unkeyed=("expert_offset", "seed"), rotate=("W",))`。
- [`ref.py`](../contrib/flydsl/a8w4/ref.py)：问题这一侧，包括 `moe_ref`、`moe_test`（形状 × 激活函数的网格）、`moe_check`（cos 误差）、`moe_dims`，以及 `moe_dyn_*`。
- [`moe.py`](../contrib/flydsl/a8w4/moe.py)：实现 `moe_tp2k`、`moe_tp1k`、`moe_tp1kq`（`@no_capture`）、`moe_ep3k`（共用的 `stack`，`**over` 覆盖项）、`moe_ep1k`；`@meta(tile_m=..., cached_loads=...)`。
- [`tune.py`](../contrib/flydsl/a8w4/tune.py)：`python -m a8w4.tune <cmd>` 等同于 `python -m pyhip.kit <cmd> a8w4:moe`，另加 a8w4 自己的报告。
- [`bench/core.py`](../contrib/flydsl/a8w4/bench/core.py)：用 `kit.bench` 的钩子包装出 `python -m a8w4.bench ab / check`。
- [`tests/test_a8w4_ops.py`](../contrib/flydsl/a8w4/tests/test_a8w4_ops.py)：`op_tests(ops)`。

kit 自身的测试：[`tests/kit/test_kit.py`](../../tests/kit/test_kit.py)（用纯 Python 的 toy op，不需要 GPU）。

## 13. 常见错误

| 报错 | 原因与处理 |
|---|---|
| `@op_register must be the outermost decorator` | `@op_register` 要写在所有 kit 标记的上面 |
| `no registered op is a prefix of the name; register its <op>_ref first` | 先注册 `<op>_ref`（模块里写在前面） |
| `<op>_test: register it before the implementations` | `<op>_test` 要在实现之前注册 |
| `the parameters must start with exactly those of <op>_ref` | 实现的前几个参数要和 ref 完全相同（名字、种类、默认值） |
| `each knob needs exactly one @param` | 每个旋钮正好一个 `@param`，`@param` 也不能多出不是旋钮的名字 |
| `knob(s) [...] clash with the vocabulary` | 旋钮名和词汇名不能相同 |
| `... prints as '...', which reads back as ...` | id 里的取值含空格或 `:`，或者 id 格式无法区分不同的值 |
| `no tuned config for {...} in <表>: tune it, or pass config=` | 这个 key 没调优过：先 `tune`，或传 `config=` |
| `no tuned config for {...} can run under CUDA-graph capture` | 表里这个点的候选都是 `@no_capture` 的：给能 capture 的实现调优 |
| `picking a config needs <op>_dims; pass config=` | 没有 `<op>_dims` 时 `op(...)` 不能查表 |
| `unknown argument(s) [...]` | `op(...)` 的关键字参数既不是接口参数也不是词汇名 |
| `<module>: expected one Registry, found N` | 命令行的 `<模块>` 里要正好有一个 `Registry` |
| `variants [...] read the rotated inputs with cached loads ...` | 计时协议会偏向走 cache 读的实现：用 `coldrot` 且 `--copies` ≥ 8，或者给实现加 `@meta(cached_loads=False)` |
| `GPUs busy (GFX use %)` / `GPU g is locked by another job` | `tune-all` 发现卡被占用：等一下、换 `--gpus`，或者 `--force` |
