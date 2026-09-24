"""MoE 测试和 benchmark 共用的模型配置，导入时不会初始化 GPU。

INTER_SIZE 是 TP 切分前的中间维度，kernel 使用 INTER_SIZE/TP。
E 是 expert 数量，TOPK 是每个 token 选择的 expert 数量。
这些配置沿用 test_moe 的预设，不代表模型的所有版本或 TP 设置。
tile 和 down_path 留给调用方或调优器选择，这里不包含后端专属的 padding。
"""

# 默认 FP8 量化方式由模型配置决定；BF16/MXFP4 通过命令行选择。
MOE_MODELS = {
    "hy3": dict(HIDDEN_SIZE=4096, INTER_SIZE=192 * 8, TP=8, E=193, TOPK=9,
                quant_type="per_tensor"),
    "qwen35_397B": dict(HIDDEN_SIZE=4096, INTER_SIZE=512 * 8, TP=8, E=512, TOPK=10,
                        quant_type="ptpc"),
    "qwen35_397B_k256": dict(HIDDEN_SIZE=4096, INTER_SIZE=256 * 8, TP=8, E=512, TOPK=10,
                             quant_type="ptpc"),
    "qwen35_35B": dict(HIDDEN_SIZE=2048, INTER_SIZE=512, TP=1, E=256, TOPK=8,
                       quant_type="ptpc"),
    "qwen35_35B_k256": dict(HIDDEN_SIZE=2048, INTER_SIZE=256, TP=1, E=256, TOPK=8,
                            quant_type="ptpc"),
    "mimo_ptpc": dict(HIDDEN_SIZE=6144, INTER_SIZE=256 * 8, TP=8, E=384, TOPK=8,
                   quant_type="ptpc"),
    "mimo_block": dict(HIDDEN_SIZE=6144, INTER_SIZE=256 * 8, TP=8, E=384, TOPK=8,
                   quant_type="block"),
    "h3": dict(HIDDEN_SIZE=6144, INTER_SIZE=384 * 8, TP=8, E=128, TOPK=4,
               quant_type="ptpc"),
}