import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def shuffle_weight(x: torch.Tensor, layout=(16, 16)):
    x_type = x.dtype
    IN, IK = layout
    BK = IK * 2
    K = 16 // 2  # bfloat16, K=8
    BN = IN
    orig_shape = x.shape
    
    # 执行洗牌逻辑
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K) # after (1, 2, 16, 10, 4, 8)
    x_ = x_.permute(0, 1, 3, 4, 2, 5) # after (1, 2, 10, 4, 16, 8)
    x_ = x_.contiguous()
    x_ = x_.view(*orig_shape)
    return x_

# 创建 32x320 矩阵，每一行一个颜色
height, width = 32, 320
row_colors = torch.arange(height).view(height, 1).repeat(1, width).float()
shuffled_colors = shuffle_weight(row_colors)

def plot_with_grid(ax, data, title, k=8):
    im = ax.imshow(data.cpu().numpy(), aspect='auto', cmap='tab20', interpolation='nearest')
    # 设置主刻度为 8 的倍数
    ax.xaxis.set_major_locator(ticker.MultipleLocator(k))
    # 显示网格，颜色设为白色或黑色以便区分
    ax.grid(which='major', axis='x', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(f"Columns (Grid line every {k} elements)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
plot_with_grid(ax1, row_colors, "Original Layout: Continuous Rows")
plot_with_grid(ax2, shuffled_colors, "Shuffled Layout: Interleaved K-groups (8 elements each)")

plt.tight_layout()
plt.savefig("shuffle_weight.png")