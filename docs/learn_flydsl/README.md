
同一个功能可以使用MLIR的很多不同表达实现，根据软件工程，应该尽可能的书写通用代码：
 - 复用现有组件；
 - 同一套源码支持尽可能多的case:
   - 关于这一点，适度非常重要，避免为了支持更多case而去over-design一个组件，组件功能必须有边界，才能保证其行为的可预测性。

例如kernel中需要从输入tensor中搬运数据时，代码应该尽量使用一套逻辑处理：
 - 不同的输入数据layout (包括size，维度)
 - 不同的输入数据的dtype
 - 不同的搬运的Tile尺寸


# GPU kernel 指令生成的编程模式
需要多线程协同工作的时候，只有确定好了每个线程每条指令要做的事情，才能生成这个指令。根本目的还是要生成我们要的指令序列，完成一个imperaive的命令, 本质上我们还是在做一个imperative的命令，但是每条这样的imperative指令都需要综合考虑：
  - 所有的wave都会参与执行这条指令，因此需要先获取切分方案
tv-layout描述的就是全部线程都执行一条 atom load 指令时，从work-group角度，所有的线程究竟完成了一个什么操作。

答案是一个tile的数据的读入，并且读入的结果以什么方式分布在各个线程的寄存器中（由于有转置读取指令，不是线程给什么地址最终就得到什么数据）。换句话说，读入到fragment之后，这个fragment其实也有layout，所有线程的数据在寄存器中的分布，合起来才是完整的tile数据

一旦进入kernel之后，我们就不再关心外部tensor的layout了，我们只是假定外部的逻辑layout，例如MOE的专家权重矩阵完全可以preshuffle，如果要使用某个copy_atom, 我们就必须传入一个tensor切片表达源的layout, fx.copy本质上就是copy1d，因此可以说，layout系统的最大受益者就是数据搬运，外存，寄存器，LDS之间交换数据。copy指令本质上可以理解为：把src按照copy_atom的src切片，dst按照copy_atom的dst切片，然后执行copy_atom完成复制

# make_tile_copy

cute DSL中`make_tile_copy`的设计存在一些不完美：
  - 接口奇怪，含义不明
  - tv-layout跟copy-atom分开，导致复杂的layout计算逻辑切分，而这样的切分代码还被隐藏在MLIR内部
    难以理解，调试

tile级别的API应该表达以tile为单位的一些操作，至少是以tile为单位的加载，存储操作，这已经可以大大简化代码的设计了。
例如，指定tile的layout，可以把它加载到寄存器/LDS中。从内部来看这样的加载操作需要得到一个线程划分

cute DSL应该是做到了一定的类似torch那样的功能，就是可以slice/index来获取tile的引用，但是一旦读入寄存器了（arith.value）
就不能看作tile了，因为不像tensor，这样的arith.value没有layout.
