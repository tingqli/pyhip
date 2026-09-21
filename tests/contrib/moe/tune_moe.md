# Cowork with AI

经验不足的时候，设计文档不利于实践，相反我们自底向上，逐步递进的方式推进比较可控最终的效果


AI有over-thinking/over-design的特点，对于未指定的情况，逻辑覆盖严密全面，但是容易无形中拉高生成代码的维护成本，尝试下列提示：
 - 不要主动添加测试用例，但是在文档或者代码注释中可以加入“建议增加测试用例”的描述
 - 遇到未指定的边界情况，可以简化需求（在文档或者代码注释中写明预设情况），而不是复杂化解决方案，例如
   - 如果某个输出目录参数未提供，直接报错而不是创建维护临时目录
 - 生成复杂逻辑时，添加合适的代码注释

让足够聪明的AI产出满意的结果，只需要更多的上下文，比如上面的需求可以改为：

 - pyhip是一个只有几个人维护的非常小规模的repo，专注kernel优化；
 - pyhip/tests/contrib/moe/utils/目录下的文件只是测试相关的辅助脚本；
 - 由此对目前的该目录的文件进行一个反思，着重在如何简化设计，降低代码维护代价上


保证AI生成的代码基本符合预期，不要不加审查的接受，凭空增加维护成本（哪怕是AI维护，复杂冗长的代码维护成本也更大）

# MoE 测试框架

## 量化
每种量化策略的具体含义其实只能由代码进行解释（也就是函数名+少量超参），不应该再引入额外一层数据结构，例如：
例如
```python
import quantizer
quant = getattr(quantizer, "fp8_ptpc")
quant.apply_a/w 的具体行为决定了
quant.dequant_a/w(*args) 输入args是 apply_a/w 的输入

# 不量化情况"no-quant"，直接返回输入和一个None的scale，保持API一致
```

## 参考实现
MoE kernel的参考实现， 基于 quantizer 反量化到fp32计算完成，支持各种量化和激活配置选项（目前repo中可见的）
参考实现内部不会调用 moe-sorting，按照最原始的原理，直接调用torch实现

参考实现的API形式，跟任何优化kernel的一样，大家都遵循标准的一致的：
 1. 编译期参数输入(TP, model_dim, inter_dim, experts, topk, quant_scheme_str, activation，... ) 其中activation是silu这样的gate-up组合方式
 2. 运行期参数输入(hidden_states, weight1, weight2，...)

```python
import moe_ref
ref_op = moe_ref.get(TP=8, model_dim=6144, inter_dim=384, experts=256, topk=8, quant_scheme_str="fp8_ptpc", activation="silu"， ... )
moe_out = ref_op(hidden_states, weight1, weight2, ...)
```
参考实现的精度测试可以对比无权重压缩的形式的参考实现给出的结果，给出calc_diff的误差

## Aiter Tune
实现 tune_aiter.py, 独立自动化完成 aiter MoE tunning 的辅助函数，完成以下任务
  - 参考[CK/ASM/FlyDSL MoE调优目录](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen)文档
  - 根据输入模型参数，自动化生成 untuned csv文件
  - 调用脚本完成tune
  - 提取tuned的结果（csv中的一行文本）
  - 用tuned csv中的设定，在当前device上测试标准aiter的fused_moe接口，测试 e2e latency 性能指标
    以及调用 moe_ref，用 calc_diff 测试 e2e 精度指标
  - 最后返回 tuned 的结果，和 e2e 实测性能， 精度指标
该脚本也支持单独工具式调用，此时可以通过命令行指定模型参数，调用上面的辅助函数，并输出其返回结果

## moe_driver
把已有MoE kernel与Aiter sorting/quant组合为完整算子。当前采用下文的函数式接口：

```python
import moe_driver

config = moe_driver.MOEconfig(model_dim, inter_dim // TP, experts, topk,
                              "fp8_blockscale", "silu", True)
prepare, run = moe_driver.registry["jit_blockscale_256_256_tiled_True"](config)
weights = prepare(weight1, weight2)
moe_out = run(hidden_states, weights, topk_ids, topk_weights, output)
```

## cross_compare

基于 `moe_driver`/`tune_aiter.py` 实现测试脚本 `cross_compare.py`：
 - 使用相同（或者类似） tune_aiter.py 脚本的输入参数
 - tune aiter对应最佳moe kernel(不强制，可以复用现有output_dir下的tuned.csv)
 - 遍历要求的tokens(batch-size)，生成测试数据，使用 "ref" driver得到参考答案，使用 "aiter" driver (保证调用到output_dir下tune好的kernel) 得到基准性能
 - 遍历 moe_driver 中非"ref"，非"aiter" 的 kernel配置，逐个测试相对精度/相对性能，最终形成类似markdown的表格输出打印到屏幕，每行一个kernel，展示相对精度，绝对/相对性能，kernel名称，（第一行展示aiter的相对精度，绝对性能，以及选中的kernel名）


比较器负责搜索和保存最佳driver CSV，`moe_driver`不提供另一层tune入口。
对应测试用例可以选择driver，单独测试各个driver的正确性（相对"ref" driver）以及加速比（相对"aiter" driver）


```
增加功能：cross_compare.py不传入任何参数时，遍历 MODEL_CONFIGS 中所有模型，每个模型遍历全部driver，保证aiter是tuned过的第一个driver，测试全部batch-size(tokens)，然后组织测试结果：
 - 生成一个交互式html（最好是单文件的）可视化展示各个模型，各个batch-size下，各个driver的性能对比
 - 仿照aiter tuned.csv，生成一个 moe_tuned.csv 表格，其中记录模型 config+batch-size 对应的最佳性能 driver 名称
 - 给moe_driver增加一个 tuned 版本，选中 tuned 版本的话，就会根据输入读取 moe_tuned.csv 表格，决定具体内部使用的 driver，保证性能最优选择
设计一下然后实现，如果遇到未说明的情况，优先以代码简单易维护为主，把该情况记录下来作为“注意事项”，而不是over-design增加逻辑，测试用例能省则省
```

### 实现方案

- 无config时在主进程串行调用比较流程，遍历 `MODEL_CONFIGS` 和默认tokens，无worker协议、进程隔离或崩溃恢复。Aiter始终第一，使用缓存调优行或先调用官方tuner；仅官方调优保留子进程机制。
- 单文件HTML内嵌数据/CSS/JS，按模型、tokens、driver筛选，展示曲线、排名、状态和参数详情；没有前端构建或在线依赖。
- 每个batch在精度通过的 `OK` 行中选最低实测E2E latency，按GPU＋完整模型参数＋实际tokens更新 `moe_tuned.csv`。记录注册名和参数快照，Aiterwinner携带具体调优行。
- `moe_driver.tuned` 读取CSV；`prepare()` 为各个winner提前准备不同权重布局，运行期只按tokens选择delegate。缺记录报错，不插值、不在线调优、不在热路径重新准备权重。

### 注意事项

- “最优”是当前GPU和测量协议下的已测候选最优，不是跨硬件/软件/路由的永久保证。
- Hy3等上游无有效Aiter组合时，保留失败并测本地候选；不伪造基准。普通异常/OOM记录失败，不自动修改测试规模；测试不设超时，GPU崩溃或卡死需人工重启。
- 默认全量可能很耗时/显存；用户可配置buffer-count。tuned需要多份prepared布局时也由用户承担显存，支持构造tuned时显式限制tokens。
- 精确tokens匹配；未测batch直接报错。CSV只支持单写者，重测同key覆盖，不做锁/事务/版本审计/自动续跑。
- 使用和详细边界见 [cross_compare.md](cross_compare.md) 与 [utils/moe_driver.md](utils/moe_driver.md)；只做少量集成验证，不增加独立测试文件。

## 简化driver的逻辑

已按下面的函数化设计替换driver类层级：工厂返回prepare/run、权重用字典、构造不碰GPU、run首次调用编译。下面是接口示意，实际可用超参受kernel限制。

（去掉smoothquant支持）
```python
# 编译期参数，根据这些参数可以很容易的构造权重，预处理权重
# 激活需要知道batch-size才能构造
#
class MOEconfig(NamedTuple):
    model_dim: int
    inter_dim_tp: int
    experts: int
    topk: int
    quant_scheme: str
    activation: str
    preshuffle: bool
    swiglu_limit: float | None = None
    beta: float = 1.0
    linear_beta: float = 1.0
    output_dtype: torch.dtype = torch.bfloat16
    # doweight_stage1/fp8_dtype取消，内部固定stage2路由加权＋设备原生FP8

def fly_prefill(config:MOEconfig, sort_block_m, stage1_blockn, stage2_blockn, atomic, gate_mode):
  # 构造期检查
  # gate_mode 作为超参而非config的一部分，因为gate/up的具体layout是kernel可控的

  def prepare(weight1, weight2):
      # 预处理权重形成kernel需要的形式，包括 量化/preshuffle/特殊layout变换（例如根据gate_mode） 等，返回本kernel能够理解的 weight_dict
      return dict(w1=new_weight1, w2=new_weight2, w1s=w1_scale, w2s=w2_scale)

  # 运行时接口，标准化
  def callable(hidden_states, weight_dict, topk_ids, topk_weights, output):
     # 运行期检查，例如 decode 阶段的kernel发现 batch-size 太大就直接在此抛出异常
     # 首次调用时编译（类似_run_compiled逻辑，把编译好的kernel记录在本函数对象属性中）
     #  2stage: 此处调用 stage1和stage2的kernel编译过程，得到两个callable
     #  1stage: 此处调用 1stage编译过程，得到一个callable

     # 运行
     # 2stage: 内部调用 sorting, 动态量化输入，依次调用stage1/stage2的callable，调用reduce
    # 1stage也可能需要sorting，按实际kernel处理
     # ...
      return output

  return prepare, callable

# config-only工厂自动成为可测试项，其他超参数已经固定
# register 修饰符将其注册到driver的全局可用kernel中，并把其函数名作为一种缩写
@register
def fly_prefill_bm64_bn128_64_reduce(config:MOEconfig):
  return fly_prefill(config, 64, 128, 64, False, "separated")
```

### 落地边界

- 只支持标准gated MoE；源W1总是自然 `[gate;up]`。gate_mode是后端packing超参，不是改变数学语义的配置。
- prepare只做静态权重量化/布局；run只缓存launcher/必要metadata，不绑定具体权重、路由、输出、workspace或stream。
- SmoothQuant/bias/mask等从driver接口移除，独立reference/quantizer模块不随此次重构删功能。
- benchmark、主进程比较器、Aiter调优helper与tuned消费端均使用新接口；既有数值测试同步迁移，不加旧API兼容层。
- 新最佳driver CSV以local config/注册函数名为准，旧表需重新生成；Aiter官方调优历史不变。


pyHIP已经不适合开发新kernel，应该重新建立工程，
新工程的主要目的是：
 - 提供标准接口（比如MOE/attention之类的），可以方便的开发/集成/评估 kernel的性能
 - 总结优化经验，