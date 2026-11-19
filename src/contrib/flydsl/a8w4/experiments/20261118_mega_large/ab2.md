
> 2026-11-18T23:40:24 | git 937370e+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 8192 32768 -v 'mega_large p16 bcm01' 'mega_large p32 bcm01' 'mega_large p8 bcm01' 'no128=mega_large p16 bcm01:no=128' 'pf1=mega_large p16 bcm01:pf=1' 'b00=mega_large p16 bcm01:bcm2=0' 'b11=mega_large p16 bcm01:bcm1=1,bcm2=1' --out a8w4/experiments/20261118_mega_large --name ab2`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.6 | +0.0 | 337.3-339.6 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 338.9 | +0.1 | 338.2-339.8 | noise floor | 1.8e-03 | 3.1e-06 |
| mega_large p32 bcm01 | `mega_large p32 bcm01` | 344.6 | +1.8 | 343.5-345.0 | slower | 1.8e-03 | 5.3e-06 |
| mega_large p8 bcm01 | `mega_large p8 bcm01` | 342.0 | +1.0 | 340.1-343.0 | slower | 1.8e-03 | 5.1e-06 |
| no128 | `mega_large p16 bcm01:no=128` | 347.2 | +2.5 | 347.1-347.8 | slower | 1.8e-03 | 5.9e-06 |
| pf1 | `mega_large p16 bcm01:pf=1` | 342.7 | +1.2 | 340.4-343.5 | slower | 1.8e-03 | 3.8e-06 |
| b00 | `mega_large p16 bcm01:bcm2=0` | 339.4 | +0.2 | 339.0-340.5 | ~ | 1.8e-03 | 4.3e-06 |
| b11 | `mega_large p16 bcm01:bcm1=1,bcm2=1` | 339.0 | +0.1 | 336.1-340.6 | ~ | 1.8e-03 | 5.3e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 818.1 | +0.0 | 814.6-819.3 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 818.8 | +0.1 | 816.9-822.2 | noise floor | 1.8e-03 | 1.7e-06 |
| mega_large p32 bcm01 | `mega_large p32 bcm01` | 867.4 | +6.0 | 863.5-874.7 | slower | 1.8e-03 | 3.2e-06 |
| mega_large p8 bcm01 | `mega_large p8 bcm01` | 840.4 | +2.7 | 834.6-841.3 | slower | 1.8e-03 | 3.0e-06 |
| no128 | `mega_large p16 bcm01:no=128` | 859.2 | +5.0 | 857.5-859.6 | slower | 1.8e-03 | 3.0e-06 |
| pf1 | `mega_large p16 bcm01:pf=1` | 825.1 | +0.9 | 823.2-826.3 | slower | 1.8e-03 | 2.0e-06 |
| b00 | `mega_large p16 bcm01:bcm2=0` | 817.1 | -0.1 | 815.9-819.3 | ~ | 1.8e-03 | 1.9e-06 |
| b11 | `mega_large p16 bcm01:bcm1=1,bcm2=1` | 819.7 | +0.2 | 816.5-820.5 | ~ | 1.8e-03 | 2.2e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2810.5 | +0.0 | 2778.5-2826.8 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 2871.6 | +2.2 | 2770.0-2881.4 | noise floor | 1.8e-03 | 3.1e-06 |
| mega_large p32 bcm01 | `mega_large p32 bcm01` | 2825.7 | +0.5 | 2788.5-2936.2 | ~ | 1.8e-03 | 4.1e-06 |
| mega_large p8 bcm01 | `mega_large p8 bcm01` | 2891.7 | +2.9 | 2790.9-2926.3 | slower | 1.8e-03 | 3.6e-06 |
| no128 | `mega_large p16 bcm01:no=128` | 3012.2 | +7.2 | 2872.1-3378.0 | slower | 1.8e-03 | 3.7e-06 |
| pf1 | `mega_large p16 bcm01:pf=1` | 2874.9 | +2.3 | 2805.1-3569.3 | slower | 1.8e-03 | 3.8e-06 |
| b00 | `mega_large p16 bcm01:bcm2=0` | 2873.9 | +2.3 | 2812.4-2948.3 | slower | 1.8e-03 | 3.8e-06 |
| b11 | `mega_large p16 bcm01:bcm1=1,bcm2=1` | 2889.5 | +2.8 | 2799.5-3514.0 | slower | 1.8e-03 | 4.9e-06 |
| *noise band +-2.2%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 8192 | 32768 |
|---|---|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.6 us | 818.1 us | 2810.5 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | +0.1%~ | +0.1%~ | +2.2%~ |
| mega_large p32 bcm01 | `mega_large p32 bcm01` | +1.8% | +6.0% | +0.5%~ |
| mega_large p8 bcm01 | `mega_large p8 bcm01` | +1.0% | +2.7% | +2.9% |
| no128 | `mega_large p16 bcm01:no=128` | +2.5% | +5.0% | +7.2% |
| pf1 | `mega_large p16 bcm01:pf=1` | +1.2% | +0.9% | +2.3% |
| b00 | `mega_large p16 bcm01:bcm2=0` | +0.2%~ | -0.1%~ | +2.3% |
| b11 | `mega_large p16 bcm01:bcm1=1,bcm2=1` | +0.1%~ | +0.2%~ | +2.8% |

