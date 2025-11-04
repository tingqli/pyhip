 
 
It seems that `machine-scheduler` has some issue, it produces sub-optimal instruction sequence in carefully written code.

hip优化代码经常会把数据搬运指令连续排列，完全数据无关的计算指令也连续排列，
但是`machine-scheduler`应该将这两者交织排列而不应该各自独立排列，这可以构造一个非常简单的测试代码验证，源码在[test-mi-sched.cpp](test-mi-sched.cpp)，programmer在书写这份代码时会有如下的自然的假定：
 - `Aregs0`/`Aregs1`/`Bregs0`/`Bregs1`都会被映射为寄存器
 - 程序会按照书写次序执行，`compute0()`/`load1()`因为完全不存在数据相互依赖，因此应该被LLVM很好的并行issue
 - 同样道理，`compute1()`/`load0()`也应该被并行issue(也就是交织issue)

但是我们观察产生的汇编，如果定义宏`MI_SCHED_BARRIER`,则能够产生预期的代码，否则`MFMA`会被大量连续issue，`DS_READ`也会被连续issue.

```bash

# 没有 -DMI_SCHED_BARRIER 的话，生成的代码为了减轻寄存器压力，连续的issue MFMA指令，并未跟DS_READ交织起来
hipcc -x hip --offload-device-only --offload-arch=gfx942 -std=c++20 -I.  -Rpass-analysis=kernel-resource-usage -DMI_SCHED_BARRIER -O2 -S -o test-mi-sched.s test-mi-sched.cpp

# 生成 llvm IR
hipcc -x hip --offload-device-only --offload-arch=gfx942 -std=c++20 -I.  -Rpass-analysis=kernel-resource-usage -O2 -S -emit-llvm -o test-mi-sched.ll test-mi-sched.cpp

# 检查 LLVM passes
./build/bin/llc -march=amdgcn  -mcpu=gfx942 -print-after-all < test-mi-sched.ll > test-mi-sched.s 2> test-mi-sched.irlog

```

查看产生的`test-mi-sched.irlog`文件, LLVM编译的大致过程如下：

**原始指令次序**
 - V_MFMA_(virtual_vreg_group_1)
 - virtual_vreg_group_2 = DS_READ_
 - V_MFMA_(virtual_vreg_group_2)
 - virtual_vreg_group_1 = DS_READ_
 
**Machine Instruction Scheduler (machine-scheduler)**：如果通过 __builtin_amdgcn_sched_group_barrier 约束了指令组次序，这一步将会根据约束interleaving指令次序，但是如果没有约束指令次序，则这一步没有特殊的重排指令，大致仍然是上面的原始次序。

**Virtual Register Rewriter (virtregrewriter)**：这一步骤之前整个程序还是基于虚拟寄存器的类似SSA的形式虚拟寄存器被映射为物理寄存器，单纯不改变指令次序，尽可能的复用失效的SSA寄存器
因此：
   - 当带有指令interleaving约束时，寄存器会被分配两份；
   - 当没有指令interleaving约束时，被连续的`MFMA`指令使用过的AB寄存器在`DS_READ_`指令处已经释放并可以复用, 最终原始代码中的两组寄存器就被大量重叠的映射到少的的多的寄存器；

