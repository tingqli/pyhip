
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

跟高级语言表达tile操作的简洁相比确实差很多，并且copy读入fragment之后，如何获取fragment的坐标还需要
使用coord-tensor，或者根据tv-layout推算。

tile级别的API应该表达以tile为单位的一些操作，至少是以tile为单位的加载，存储操作，这已经可以大大简化代码的设计了。
例如，指定tile的layout，可以把它加载到寄存器/LDS中。从内部来看这样的加载操作需要得到一个线程划分

cute DSL应该是做到了一定的类似torch那样的功能，就是可以slice/index来获取tile的引用，但是一旦读入寄存器了（arith.value）
就不能看作tile了，因为不像tensor，这样的arith.value没有layout.

# Fly的一些理念

Fly里面避免使用指针，因此数据的搬运都是通过tensor切片，加上copy_atom完成的，类似torch跟C++的区别。

但是需要注意的是，原始语义是tensor级别的操作，映射到work-group级别（也就是triton的级别）仍然可以保证是tensor级别语义
但是要真正生成代码，还需要使用tv-layout映射到线程级别，也就是每个线程要进一步切分任务到它所需要处理的那个小fragment，才能
正确生成代码完成任务，例如make_tiled_copy做的那样。

当tensor的thread信息丢失之后，虽然他还是一个tensor 切片，但是脱离 tv-layout的话，仍然会让人困惑

在我看来，加载数据的语句应该是用带着 thread-idx切片的layout描述，比如某个tensor的原始shape对于algorithm的实现描述
非常有用，因此应该是非常规则的，例如描述一个矩阵乘法时，从逻辑shape上就是 `[BM,BK] * [BN, BK]=>[BM, BN]`的形式，
但是细化下去，`[BM,BN]`的布局决定了A和B矩阵的布局，也就是我们可以指定很多种方式完成，而指定之后，对应的A/B矩阵的
布局也就确定了，形式化每个线程加载到一个fragment，但是结合背后的布局，其实是加载了整个`[BM, BK]`矩阵，而这种指定布局
加载整个矩阵的思路，才是更加容易理解的思路，就像gluon那样。但是也要注意，不像LDS/mem-tensor，fragment tensor 一旦
加载成功，其layout是不能无代价的改变的，并且也不能任意坐标访问，因为跨线程操作是高代价的。

另一个非常困惑的点是，fragment tensor没有任何全局布局信息，是一个纯局部的概念，另外其layout非常复杂，难以理解
按照cute layout的设计，layout的复杂性应该是屏蔽在嵌套层次内部的。不应该被感知。

比如一个tensor生成了tv-layout之后，只有用tid去索引，然后加载才会变成一个fragment，但是现在索引发生的比较隐秘。


