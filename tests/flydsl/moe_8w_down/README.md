# moe down kernel - BlockScaled(128x128)

FlyDSL语义复杂，需要使用COT逐步逼近目标。

Copilot帮我们实现了基本调度框架，但是实现高效的主Pipeline还是比较困难，因此我们需要逐个解决问题：

## workgroup 多线程协作加载scales

