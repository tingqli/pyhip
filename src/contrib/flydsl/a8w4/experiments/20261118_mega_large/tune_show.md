
#### mimo-flash2.6-ep8-silu

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | best mori | us | best mori_eager | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm32 | 143.0 | mega p16 l1 bcm22 | 177.5 | mega_small bm32 | 143.0 | mega_large cadd xrow bcm22 | 159.0 | mori s16 | 170.2 | mori_eager t16 | 361.4 |  |
| 128 | mega_small bm64 | 153.5 | mega p16 l1 bcm22 | 193.8 | mega_small bm64 | 153.5 | mega_large cadd xrow bcm22 | 170.4 | mori s16 | 182.8 | mori_eager t16 | 355.7 |  |
| 256 | mega_small bm64 | 159.4 | mega p32 l2 bcm22 | 200.3 | mega_small bm64 | 159.4 | mega_large cadd xrow bcm22 | 177.0 | mori s16 | 186.9 | mori_eager m64 | 377.6 |  |
| 512 | mega_small bm64 | 174.4 | mega p16 l2 bcm22 | 211.5 | mega_small bm64 | 174.4 | mega_large cadd xrow bcm22 | 187.4 | mori s32 | 218.6 | mori_eager s32 | 400.6 |  |
| 1024 | mega_large cadd xrow bcm22 | 221.7 | mega p16 l2 bcm22 | 243.0 | mega_small bm64 | 229.2 | mega_large cadd xrow bcm22 | 221.7 | mori m64 | 272.9 | mori_eager m64 | 470.7 |  |
| 2048 | mega_large cadd xrow bcm22 | 330.8 | mega p32 l4 bcm22 | 366.1 | - | - | mega_large cadd xrow bcm22 | 330.8 | mori m64 | 510.3 | mori_eager l128 | 714.7 |  |
| 4096 | mega_large cfp8 xrow bcm01 | 489.3 | mega p16 l4 bcm01 | 530.5 | - | - | mega_large cfp8 xrow bcm01 | 489.3 | mori l128 | 709.5 | mori_eager l128 | 885.7 |  |
| 8192 | mega_large cfp8 xrow bcm01 | 828.3 | mega p16 l4 bcm01 | 902.5 | - | - | mega_large cfp8 xrow bcm01 | 828.3 | mori l128c | 1110.9 | mori_eager l128c | 1279.8 |  |
| 16384 | mega_large cadd xrow bcm01 | 1496.5 | mega p16 l4 bcm01 | 1619.1 | - | - | mega_large cadd xrow bcm01 | 1496.5 | mori l128c | 2024.7 | mori_eager l128c | 2159.5 |  |
| 32768 | mega_large cfp8 xrow bcm01 | 2934.4 | mega p16 l4 bcm01 | 3153.9 | - | - | mega_large cfp8 xrow bcm01 | 2934.4 | mori l128c | 3845.6 | mori_eager l128c | 4014.0 |  |

#### mimo-flash2.6-ep8-situv2

| tokens | best | us | best mega | us | best mega_small | us | best mega_large | us | best mori | us | best mori_eager | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|
| 64 | mega_small bm64 | 144.7 | mega p32 l4 bcm22 | 177.0 | mega_small bm64 | 144.7 | mega_large cadd x128 bcm22 | 165.9 | mori s16 | 171.2 | mori_eager t16 | 384.3 |  |
| 128 | mega_small bm64 | 155.4 | mega p32 l2 bcm22 | 196.8 | mega_small bm64 | 155.4 | mega_large cadd x128 bcm22 | 181.3 | mori s16 | 183.7 | mori_eager t16 | 371.0 |  |
| 256 | mega_small bm64 | 160.5 | mega p32 l2 bcm22 | 203.8 | mega_small bm64 | 160.5 | mega_large cadd x128 bcm22 | 186.9 | mori s16 | 188.2 | mori_eager s32 | 368.3 |  |
| 512 | mega_small bm64 | 175.4 | mega p32 l2 bcm22 | 214.6 | mega_small bm64 | 175.4 | mega_large cadd x128 bcm22 | 197.5 | mori s32 | 221.3 | mori_eager m64 | 392.9 |  |
| 1024 | mega_small bm64 | 231.9 | mega p32 l4 bcm22 | 246.8 | mega_small bm64 | 231.9 | mega_large cadd x128 bcm22 | 233.5 | mori m64 | 279.2 | mori_eager m64 | 444.1 |  |
| 2048 | mega_large cadd x128 bcm22 | 362.2 | mega p16 l4 bcm22 | 373.1 | - | - | mega_large cadd x128 bcm22 | 362.2 | mori m64 | 445.9 | mori_eager m64 | 597.0 |  |
| 4096 | mega_large cfp8 x128 bcm01 | 526.4 | mega p16 l4 bcm01 | 544.1 | - | - | mega_large cfp8 x128 bcm01 | 526.4 | mori l128 | 729.9 | mori_eager l128c | 868.5 |  |
| 8192 | mega_large cadd x128 bcm01 | 900.6 | mega p16 l4 bcm01 | 924.3 | - | - | mega_large cadd x128 bcm01 | 900.6 | mori l128c | 1138.0 | mori_eager l128c | 1270.7 |  |
| 16384 | mega_large cadd x128 bcm01 | 1645.7 | mega p32 l2 bcm01 | 1708.4 | - | - | mega_large cadd x128 bcm01 | 1645.7 | mori l128c | 2064.1 | mori_eager l128c | 2178.0 |  |
| 32768 | mega_large cfp8 x128 bcm01 | 3201.4 | mega p16 l4 bcm01 | 3214.1 | - | - | mega_large cfp8 x128 bcm01 | 3201.4 | mori l128c | 3914.0 | mori_eager l128c | 4085.2 |  |
