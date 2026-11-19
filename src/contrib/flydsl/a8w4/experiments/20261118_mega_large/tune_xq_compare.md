# EP table with the x_quant dim (20261119): comparisons

## per_token vs per_1x128（同一次调优）

「最快」两列是各 x_quant 下所有 solver 中最快的配置；后三列是各 solver 族自己最快的配置从 per_1x128 换成 per_token 的变化（负数 = per_token 更快）。

| act | tokens | per_1x128 最快 | us | per_token 最快 | us | 最快 | mega | mega_small | mega_large |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| silu | 64 | mega_small bm64 | 142.4 | mega_small bm64 | 139.8 | -1.8% | -3.5% | -1.8% | -3.7% |
| silu | 128 | mega_small bm64 | 153.9 | mega_small bm64 | 151.5 | -1.6% | -4.3% | -1.6% | -5.1% |
| silu | 256 | mega_small bm32 | 160.0 | mega_small bm32 | 156.4 | -2.2% | -4.6% | -2.2% | -4.4% |
| silu | 512 | mega_small bm64 | 174.0 | mega_small bm64 | 170.5 | -2.0% | -5.3% | -2.0% | -3.9% |
| silu | 1024 | mega_small bm64 | 229.3 | mega_small bm64 | 218.1 | -4.9% | -4.9% | -4.9% | -4.2% |
| silu | 2048 | mega_large cadd bcm22 | 353.4 | mega_large cadd bcm22 | 333.0 | -5.8% | -6.4% | - | -5.8% |
| silu | 4096 | mega_large cfp8 bcm01 | 519.0 | mega_large cfp8 bcm01 | 491.0 | -5.4% | -3.3% | - | -5.4% |
| silu | 8192 | mega_large cfp8 bcm01 | 877.7 | mega_large cadd bcm01 | 835.8 | -4.8% | -5.5% | - | -4.8% |
| silu | 16384 | mega_large cadd bcm01 | 1592.6 | mega_large cadd bcm01 | 1498.7 | -5.9% | -7.4% | - | -5.9% |
| silu | 32768 | mega_large cfp8 bcm01 | 2994.1 | mega p16 l2 bcm01 | 2833.2 | -5.4% | -12.1% | - | -1.9% |
| situv2 | 64 | mega_small bm32 | 145.4 | mega_small bm64 | 140.8 | -3.2% | -2.7% | -3.2% | -3.7% |
| situv2 | 128 | mega_small bm64 | 154.6 | mega_small bm32 | 152.8 | -1.2% | -4.6% | -1.2% | -5.4% |
| situv2 | 256 | mega_small bm64 | 160.7 | mega_small bm64 | 157.8 | -1.8% | -4.2% | -1.8% | -4.0% |
| situv2 | 512 | mega_small bm32 | 175.0 | mega_small bm64 | 171.1 | -2.2% | -4.8% | -2.2% | -3.6% |
| situv2 | 1024 | mega_small bm64 | 232.0 | mega_small bm64 | 221.3 | -4.6% | -4.9% | -4.6% | -3.8% |
| situv2 | 2048 | mega_large cadd bcm22 | 360.7 | mega_large cadd bcm22 | 341.4 | -5.4% | -6.6% | - | -5.4% |
| situv2 | 4096 | mega_large cfp8 bcm01 | 524.2 | mega_large cfp8 bcm01 | 497.7 | -5.1% | -4.8% | - | -5.1% |
| situv2 | 8192 | mega_large cfp8 bcm01 | 900.6 | mega_large cfp8 bcm01 | 845.0 | -6.2% | -6.1% | - | -6.2% |
| situv2 | 16384 | mega_large cfp8 bcm01 | 1643.1 | mega_large cadd bcm01 | 1523.5 | -7.3% | -7.9% | - | -7.3% |
| situv2 | 32768 | mega_large cfp8 bcm01 | 3081.2 | mega p16 l2 bcm01 | 2898.7 | -5.9% | -11.5% | - | -3.1% |

## 与上一版表（v2，`tune_compare.txt`）比较

v2 没有 x_quant 列：`mega_large` 名字带 `xrow` 的配置是逐行量化（记作 per_token），其余配置都是 per_1x128。每格是该 x_quant 下所有 solver 中最快的配置。

| act | x_quant | tokens | v2 最快 | us | 新表最快 | us | 变化 |
|---|---|---:|---|---:|---|---:|---:|
| silu | per_1x128 | 64 | mega_small bm32 | 143.0 | mega_small bm64 | 142.4 | -0.4% |
| silu | per_1x128 | 128 | mega_small bm64 | 153.5 | mega_small bm64 | 153.9 | +0.3% |
| silu | per_1x128 | 256 | mega_small bm64 | 159.4 | mega_small bm32 | 160.0 | +0.4% |
| silu | per_1x128 | 512 | mega_small bm64 | 174.4 | mega_small bm64 | 174.0 | -0.2% |
| silu | per_1x128 | 1024 | mega_small bm64 | 229.2 | mega_small bm64 | 229.3 | +0.0% |
| silu | per_1x128 | 2048 | mega_large cadd x128 bcm22 | 354.3 | mega_large cadd bcm22 | 353.4 | -0.3% |
| silu | per_1x128 | 4096 | mega_large cfp8 x128 bcm01 | 517.2 | mega_large cfp8 bcm01 | 519.0 | +0.3% |
| silu | per_1x128 | 8192 | mega_large cfp8 x128 bcm01 | 878.5 | mega_large cfp8 bcm01 | 877.7 | -0.1% |
| silu | per_1x128 | 16384 | mega_large cadd x128 bcm01 | 1586.7 | mega_large cadd bcm01 | 1592.6 | +0.4% |
| silu | per_1x128 | 32768 | mega_large cfp8 x128 bcm01 | 3009.1 | mega_large cfp8 bcm01 | 2994.1 | -0.5% |
| silu | per_token | 64 | mega_large cadd xrow bcm22 | 159.0 | mega_small bm64 | 139.8 | -12.1% |
| silu | per_token | 128 | mega_large cadd xrow bcm22 | 170.4 | mega_small bm64 | 151.5 | -11.1% |
| silu | per_token | 256 | mega_large cadd xrow bcm22 | 177.0 | mega_small bm32 | 156.4 | -11.6% |
| silu | per_token | 512 | mega_large cadd xrow bcm22 | 187.4 | mega_small bm64 | 170.5 | -9.0% |
| silu | per_token | 1024 | mega_large cadd xrow bcm22 | 221.7 | mega_small bm64 | 218.1 | -1.6% |
| silu | per_token | 2048 | mega_large cadd xrow bcm22 | 330.8 | mega_large cadd bcm22 | 333.0 | +0.7% |
| silu | per_token | 4096 | mega_large cfp8 xrow bcm01 | 489.3 | mega_large cfp8 bcm01 | 491.0 | +0.3% |
| silu | per_token | 8192 | mega_large cfp8 xrow bcm01 | 828.3 | mega_large cadd bcm01 | 835.8 | +0.9% |
| silu | per_token | 16384 | mega_large cadd xrow bcm01 | 1496.5 | mega_large cadd bcm01 | 1498.7 | +0.1% |
| silu | per_token | 32768 | mega_large cfp8 xrow bcm01 | 2934.4 | mega p16 l2 bcm01 | 2833.2 | -3.4% |
| situv2 | per_1x128 | 64 | mega_small bm64 | 144.7 | mega_small bm32 | 145.4 | +0.5% |
| situv2 | per_1x128 | 128 | mega_small bm64 | 155.4 | mega_small bm64 | 154.6 | -0.5% |
| situv2 | per_1x128 | 256 | mega_small bm64 | 160.5 | mega_small bm64 | 160.7 | +0.1% |
| situv2 | per_1x128 | 512 | mega_small bm64 | 175.4 | mega_small bm32 | 175.0 | -0.2% |
| situv2 | per_1x128 | 1024 | mega_small bm64 | 231.9 | mega_small bm64 | 232.0 | +0.0% |
| situv2 | per_1x128 | 2048 | mega_large cadd x128 bcm22 | 362.2 | mega_large cadd bcm22 | 360.7 | -0.4% |
| situv2 | per_1x128 | 4096 | mega_large cfp8 x128 bcm01 | 526.4 | mega_large cfp8 bcm01 | 524.2 | -0.4% |
| situv2 | per_1x128 | 8192 | mega_large cadd x128 bcm01 | 900.6 | mega_large cfp8 bcm01 | 900.6 | +0.0% |
| situv2 | per_1x128 | 16384 | mega_large cadd x128 bcm01 | 1645.7 | mega_large cfp8 bcm01 | 1643.1 | -0.2% |
| situv2 | per_1x128 | 32768 | mega_large cfp8 x128 bcm01 | 3201.4 | mega_large cfp8 bcm01 | 3081.2 | -3.8% |
| situv2 | per_token | 64 | - | - | mega_small bm64 | 140.8 | - |
| situv2 | per_token | 128 | - | - | mega_small bm32 | 152.8 | - |
| situv2 | per_token | 256 | - | - | mega_small bm64 | 157.8 | - |
| situv2 | per_token | 512 | - | - | mega_small bm64 | 171.1 | - |
| situv2 | per_token | 1024 | - | - | mega_small bm64 | 221.3 | - |
| situv2 | per_token | 2048 | - | - | mega_large cadd bcm22 | 341.4 | - |
| situv2 | per_token | 4096 | - | - | mega_large cfp8 bcm01 | 497.7 | - |
| situv2 | per_token | 8192 | - | - | mega_large cfp8 bcm01 | 845.0 | - |
| situv2 | per_token | 16384 | - | - | mega_large cadd bcm01 | 1523.5 | - |
| situv2 | per_token | 32768 | - | - | mega p16 l2 bcm01 | 2898.7 | - |
