---
name: flydsl-codegen-validation
description: 'Use when refactoring or debugging FlyDSL kernels, buffer/LDS byte offsets and alignment, runtime versus offline compilation, tensor wrappers, JIT artifact reuse, multi-kernel ELF/ISA extraction, or source-to-machine-code equivalence validation.'
---

# FlyDSL内存接口与机器码验证

目标：用轻量且可核验的证据确认接口单位、运行环境及代码变更影响，不把Python静态检查或源码相似当作GPU正确性证明。

## 1. 先确定真正执行的环境

1. 核实选定Python解释器、实际import来源、FlyDSL/ROCm版本与目标架构；编辑器可能索引另一个checkout，不能用其stub推断实际安装库缺少某API。
2. 本仓库editable包与根目录namespace可能影响导入；优先核对`__file__`、现有PYTHONPATH和子模块导入，不立即安装依赖、注入兼容shim或修改全局分析设置。
3. 历史`fx.buffer_ops`接口与当前`fx.rocdl`不同；替换必须保留offset单位、mask、访问位宽和aux，不能只让import通过。
4. 修改Python API时用语义引用检查调用点；若语言服务返回已删除文件，与磁盘和实际imports核对后再判断。不要恢复过期封装来迎合stale索引。

## 2. 明确每层的地址单位

| 接口 | 调用者传入的单位/含义 |
|---|---|
| typed pointer加法 | 指针元素，不是固定字节 |
| `fx.copy(..., soffset=...)` | 当前FlyDSL copy lowering中的元素偏移，随后乘dtype字节宽度 |
| raw buffer load/store | `voffset`/`soffset`字节偏移 |
| `BufferTensor.load(voffset_bytes, soffset_bytes)` | descriptor根相对的完整字节偏移；显式模式替代而非再叠加slice位置 |
| `LdsTensor.load/store(address_bytes=...)` | 完整预计算LDS字节地址；固定offset另按接口处理 |

实施时：
- 对FP8/BF16/FP32分别核对一次单位，防止byte offset被二次乘2/4。
- 显式热段接口要求预先准备的32-bit SSA；类型截断和单位转换在调用前完成，不能偷偷在Memory生成VALU。
- `soffset`必须wave一致。只有已证明uniform的值才能标量化，不能用`readfirstlane`掩盖divergent地址。
- 动态byte pointer偏移后再recast可能丢失对齐。不要谎报alignment；可从已对齐base转换后加可证明的typed偏移，或保留原FP8 packet load后bitcast。
- 24B搬运明确分16B+8B，不能扩读32B或声称是一条VM请求。copy32不保证自动合并为copy128，请求数变化会影响wait账本。
- 全局行/expert大偏移必须在乘法前升64位；局部buffer extent与实际stride、尾行共同决定边界。

现行定义及CPU/MLIR合同：[helpers](../../../src/contrib/flydsl/helpers.py)、[Tensor接口回归](../../../tests/contrib/moe/test_tensor_memory.py)。

## 3. 区分正常编译、纯离线与缓存复用

- 正常`flyc.compile`先调用launcher，可能真实执行一次kernel；构造参数必须覆盖完整grid，不能用一个tile的占位张量给大shape。
- 纯离线编译需显式compile-only模式和目标架构；仅隐藏GPU不保证没有runtime初始化。若宣称CPU-only，应阻止/核对Torch lazy init、device runtime及ExecutionEngine创建。
- 首次调用已命中进程缓存时，后续设置dump未必重新生成ISA。应在首次编译前指定独立dump目录，或按编译器支持的方式导出已核产物；不要删除共享cache作为默认修复。
- 复用artifact必须匹配源码/依赖、编译选项、shape、dtype、量化、目标架构和ABI。不同TOPK或rank可改变地址代码，不能拿另一shape的ISA作对照。
- 重构可能改变源哈希但不改变设备代码；也可能改变调用绑定或JIT cache key。函数名不变不代表整个artifact身份不变。

## 4. 等价性分层，不能混为一句“无变化”

| 层次 | 必须检查的事实 |
|---|---|
| 源码/AST | 允许变更的节点、数值表达式、边界、loop状态、调用数 |
| 编译产物 | host launcher、kernarg/ABI、grid/block、实际device ISA、descriptor与资源 |
| 数值 | 原参考、原容差、真实输入、有效/无效区域；bitexact另行断言 |
| 性能 | 相同计时边界、设备、地址和采样协议下的实测 |

- AST等价只能说明所检查表达式等价，不能单独证明调度/资源不变。
- `.text`相同与整ELF相同分开报告；符号名、调试信息或metadata可以改变文件哈希。
- 一个ELF可有多个kernel和多个退出。使用`STT_FUNC`的起点/size和对应descriptor提取，不在第一个`s_endpgm`截断，也不让最后一个descriptor覆盖全部kernel。
- 对照完整opcode/操作数、分支回边、资源和真实异步load-use/WAW。编译器补的wait应解释，不能删掉它来满足源码期望。
- 反汇编文本有差异但机器字相同时，以实际二进制核对并明确差异。可交换整数加数的个别差异不能成为忽略任意ISA差异的理由。
- `vgpr_count`、`next_free_vgpr`、SGPR字段、AGPR、private/scratch与occupancy分别记录；任何单一资源数都不直接等于实际驻留数。

## 5. 验证最小充分范围

1. 先做Python语法/定向静态检查，再做目标离线编译。FlyDSL dynamic IR诊断不等于运行失败，但也不能用“常见误报”忽略真正缩进/签名错误。
2. 数值变更、边界变更或无法证明编译等价时，再做GPU验证；纯文档整理不重跑GPU，纯搬迁不能无证据声称性能无变化。
3. 先读[pytest配置](../../../pytest.ini)：当前`python_files=*.py`，MoE目录含历史执行脚本，必须定向file/node，不能盲目全目录收集。
4. 用既有框架验证，不为了一个重构扩建永久专项测试；保存必要的独立结果和失败记录。
5. 输出完整性从真实JSON/JUnit和产物判断，stdout被重定向不代表未运行；源文件已被用户修改时，不能覆盖回旧哈希。
6. 历史产物只读：审计以它的冻结源码为准，当前执行以当前源码为准。旧路径或模块删除不应通过篡改旧收据“修复”。

证据：[Tensor重构离线对照](../../../tests/contrib/moe/results/tensor_pipeline_20260910/offline_audited.json)、[实际安装API兼容验收](../../../tests/contrib/moe/results/system_python_20260911/gpu_v4.json)。性能验证另见[测量skill](../gpu-benchmark-validation/SKILL.md)。