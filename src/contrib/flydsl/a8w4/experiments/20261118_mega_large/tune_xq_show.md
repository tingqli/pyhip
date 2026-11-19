# EP table with the x_quant dim (20261119): best per solver

`python -m kernelkit show a8w4.tune:EP_TUNING mimo-flash2.6-ep8` on the table of
`python -m kernelkit tune-all a8w4.tune:EP_TUNING --gpus 0-7 --out /tmp/a8w4_ep_xq` (1160 rows;
silu per_1x128 at 128 / 512 / 1024 tokens re-tuned and merged: in the first pass some mori /
mori_eager configs took 1.3-4.6x their usual times there, `tune_xq_silu-per_1x128_retune.log`).
mori / mori_eager are n/a for per_token (they quantize per (row, 128 cols)).

#### mimo-flash2.6-ep8-silu-per_1x128

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | best mori | us | best mori_eager | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm64 | 142.4 | mega p8 l2 bcm22 | 177.0 | mega_small bm64 | 142.4 | mega_large cadd bcm22 | 164.8 | mori s16 | 170.0 | mori_eager s32 | 399.7 |  |
| 128 | mega_small bm64 | 153.9 | mega p32 l2 bcm22 | 193.9 | mega_small bm64 | 153.9 | mega_large cadd bcm22 | 178.0 | mori t16 | 182.5 | mori_eager s32 | 358.4 |  |
| 256 | mega_small bm32 | 160.0 | mega p32 l1 bcm22 | 200.8 | mega_small bm32 | 160.0 | mega_large cadd bcm22 | 185.1 | mori s16 | 187.6 | mori_eager s16 | 362.0 |  |
| 512 | mega_small bm64 | 174.0 | mega p32 l1 bcm22 | 211.4 | mega_small bm64 | 174.0 | mega_large cadd bcm22 | 194.4 | mori s32 | 219.0 | mori_eager s32 | 382.4 |  |
| 1024 | mega_small bm64 | 229.3 | mega p32 l4 bcm22 | 242.9 | mega_small bm64 | 229.3 | mega_large cadd bcm22 | 231.0 | mori m64 | 272.6 | mori_eager m64 | 435.8 |  |
| 2048 | mega_large cadd bcm22 | 353.4 | mega p32 l4 bcm22 | 366.2 | - | - | mega_large cadd bcm22 | 353.4 | mori m64 | 440.6 | mori_eager l128 | 622.0 |  |
| 4096 | mega_large cfp8 bcm01 | 519.0 | mega p16 l4 bcm01 | 531.4 | - | - | mega_large cfp8 bcm01 | 519.0 | mori l128 | 709.9 | mori_eager l128 | 858.5 |  |
| 8192 | mega_large cfp8 bcm01 | 877.7 | mega p16 l4 bcm01 | 904.4 | - | - | mega_large cfp8 bcm01 | 877.7 | mori l128c | 1116.0 | mori_eager l128c | 1254.0 |  |
| 16384 | mega_large cadd bcm01 | 1592.6 | mega p8 l4 bcm01 | 1654.0 | - | - | mega_large cadd bcm01 | 1592.6 | mori l128c | 2035.8 | mori_eager l128c | 2131.5 |  |
| 32768 | mega_large cfp8 bcm01 | 2994.1 | mega p32 l4 bcm01 | 3224.4 | - | - | mega_large cfp8 bcm01 | 2994.1 | mori l128c | 3877.0 | mori_eager l128c | 3914.5 |  |

#### mimo-flash2.6-ep8-silu-per_token

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm64 | 139.8 | mega p16 l1 bcm22 | 170.8 | mega_small bm64 | 139.8 | mega_large cadd bcm22 | 158.7 |  |
| 128 | mega_small bm64 | 151.5 | mega p16 l4 bcm22 | 185.6 | mega_small bm64 | 151.5 | mega_large cadd bcm22 | 168.9 |  |
| 256 | mega_small bm32 | 156.4 | mega p32 l2 bcm22 | 191.6 | mega_small bm32 | 156.4 | mega_large cadd bcm22 | 176.9 |  |
| 512 | mega_small bm64 | 170.5 | mega p32 l2 bcm22 | 200.1 | mega_small bm64 | 170.5 | mega_large cadd bcm22 | 186.8 |  |
| 1024 | mega_small bm64 | 218.1 | mega p32 l2 bcm22 | 231.0 | mega_small bm64 | 218.1 | mega_large cadd bcm22 | 221.3 |  |
| 2048 | mega_large cadd bcm22 | 333.0 | mega p16 l1 bcm22 | 342.9 | - | - | mega_large cadd bcm22 | 333.0 |  |
| 4096 | mega_large cfp8 bcm01 | 491.0 | mega p16 l4 bcm01 | 513.7 | - | - | mega_large cfp8 bcm01 | 491.0 |  |
| 8192 | mega_large cadd bcm01 | 835.8 | mega p16 l4 bcm01 | 855.0 | - | - | mega_large cadd bcm01 | 835.8 |  |
| 16384 | mega_large cadd bcm01 | 1498.7 | mega p16 l4 bcm01 | 1532.4 | - | - | mega_large cadd bcm01 | 1498.7 |  |
| 32768 | mega p16 l2 bcm01 | 2833.2 | mega p16 l2 bcm01 | 2833.2 | - | - | mega_large cfp8 bcm01 | 2937.6 |  |

#### mimo-flash2.6-ep8-situv2-per_1x128

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | best mori | us | best mori_eager | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm32 | 145.4 | mega p8 l4 bcm22 | 178.2 | mega_small bm32 | 145.4 | mega_large cadd bcm22 | 166.1 | mori s16 | 171.0 | mori_eager l128 | 421.1 |  |
| 128 | mega_small bm64 | 154.6 | mega p32 l2 bcm22 | 196.2 | mega_small bm64 | 154.6 | mega_large cadd bcm22 | 180.0 | mori s16 | 183.9 | mori_eager s32 | 409.0 |  |
| 256 | mega_small bm64 | 160.7 | mega p32 l4 bcm22 | 203.2 | mega_small bm64 | 160.7 | mega_large cadd bcm22 | 186.1 | mori s16 | 189.5 | mori_eager l128 | 412.9 |  |
| 512 | mega_small bm32 | 175.0 | mega p32 l4 bcm22 | 214.2 | mega_small bm32 | 175.0 | mega_large cadd bcm22 | 195.8 | mori s32 | 221.6 | mori_eager t16 | 432.9 |  |
| 1024 | mega_small bm64 | 232.0 | mega p32 l2 bcm22 | 246.6 | mega_small bm64 | 232.0 | mega_large cadd bcm22 | 233.0 | mori m64 | 277.3 | mori_eager l128 | 446.4 |  |
| 2048 | mega_large cadd bcm22 | 360.7 | mega p16 l1 bcm22 | 373.3 | - | - | mega_large cadd bcm22 | 360.7 | mori m64 | 445.1 | mori_eager m64 | 631.6 |  |
| 4096 | mega_large cfp8 bcm01 | 524.2 | mega p16 l4 bcm01 | 542.5 | - | - | mega_large cfp8 bcm01 | 524.2 | mori l128 | 727.5 | mori_eager l128c | 875.0 |  |
| 8192 | mega_large cfp8 bcm01 | 900.6 | mega p16 l4 bcm01 | 921.3 | - | - | mega_large cfp8 bcm01 | 900.6 | mori l128c | 1135.3 | mori_eager l128c | 1253.5 |  |
| 16384 | mega_large cfp8 bcm01 | 1643.1 | mega p8 l4 bcm01 | 1693.4 | - | - | mega_large cfp8 bcm01 | 1643.1 | mori l128c | 2059.8 | mori_eager l128c | 2201.1 |  |
| 32768 | mega_large cfp8 bcm01 | 3081.2 | mega p16 l4 bcm01 | 3274.9 | - | - | mega_large cfp8 bcm01 | 3081.2 | mori l128c | 3954.0 | mori_eager l128c | 3983.7 |  |

#### mimo-flash2.6-ep8-situv2-per_token

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm64 | 140.8 | mega p32 l1 bcm22 | 173.4 | mega_small bm64 | 140.8 | mega_large cadd bcm22 | 159.9 |  |
| 128 | mega_small bm32 | 152.8 | mega p32 l1 bcm22 | 187.1 | mega_small bm32 | 152.8 | mega_large cadd bcm22 | 170.2 |  |
| 256 | mega_small bm64 | 157.8 | mega p32 l1 bcm22 | 194.6 | mega_small bm64 | 157.8 | mega_large cadd bcm22 | 178.6 |  |
| 512 | mega_small bm64 | 171.1 | mega p32 l1 bcm22 | 203.9 | mega_small bm64 | 171.1 | mega_large cadd bcm22 | 188.8 |  |
| 1024 | mega_small bm64 | 221.3 | mega p32 l2 bcm22 | 234.6 | mega_small bm64 | 221.3 | mega_large cadd bcm22 | 224.2 |  |
| 2048 | mega_large cadd bcm22 | 341.4 | mega p16 l1 bcm22 | 348.8 | - | - | mega_large cadd bcm22 | 341.4 |  |
| 4096 | mega_large cfp8 bcm01 | 497.7 | mega p16 l4 bcm01 | 516.3 | - | - | mega_large cfp8 bcm01 | 497.7 |  |
| 8192 | mega_large cfp8 bcm01 | 845.0 | mega p16 l4 bcm01 | 864.7 | - | - | mega_large cfp8 bcm01 | 845.0 |  |
| 16384 | mega_large cadd bcm01 | 1523.5 | mega p16 l4 bcm01 | 1560.3 | - | - | mega_large cadd bcm01 | 1523.5 |  |
| 32768 | mega p16 l2 bcm01 | 2898.7 | mega p16 l2 bcm01 | 2898.7 | - | - | mega_large cfp8 bcm01 | 2985.9 |  |
