
> 2026-11-19T01:34:33 | git 5312964+dirty | FlyDSL 0.3.2 | torch 2.9.1+rocm7.2.0.git7e1940d4 | AMD Instinct MI300X (gfx942, 304 CU, cap 750 W) | HIP_VISIBLE_DEVICES None | 8 ranks
> `python -m kernelkit ab a8w4.tune:EP_TUNING mimo-flash2.6-ep8-silu -t 2048 8192 32768 -v 'mega_large p16 bcm01' 'lag4=mega_large p16 bcm01:lag=4' 'lag8=mega_large p16 bcm01:lag=8' 'lag12=mega_large p16 bcm01:lag=12' 'lag20=mega_large p16 bcm01:lag=20' 'lag32=mega_large p16 bcm01:lag=32' --out a8w4/experiments/20261118_mega_large --name ab4`

moe_ep problem mimo-flash2.6-ep8-silu: {'model_dim': 4096, 'inter_dim': 2048, 'experts_global': 256, 'world': 8, 'topk': 8, 'routing': 'uniform', 'act': 'silu'}

every call cold: L2 flush, the next of 8 copies of ['W']; 5 rounds x 16 calls; us = median of the round medians; verdict only beyond the noise band (max(|mega_large p16 bcm01#2 delta|, half the base's round spread, 0.3%))

#### 2048 tokens (seed 2055)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.1 | +0.0 | 337.0-338.8 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 337.4 | -0.2 | 335.8-338.4 | noise floor | 1.8e-03 | 4.8e-06 |
| lag4 | `mega_large p16 bcm01:lag=4` | 361.6 | +6.9 | 360.9-362.0 | slower | 1.8e-03 | 6.8e-06 |
| lag8 | `mega_large p16 bcm01:lag=8` | 366.0 | +8.2 | 365.5-367.3 | slower | 1.8e-03 | 6.9e-06 |
| lag12 | `mega_large p16 bcm01:lag=12` | 352.9 | +4.4 | 350.8-354.5 | slower | 1.8e-03 | 6.6e-06 |
| lag20 | `mega_large p16 bcm01:lag=20` | 337.6 | -0.2 | 336.8-338.8 | ~ | 1.8e-03 | 5.3e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 338.1 | +0.0 | 336.3-339.3 | ~ | 1.8e-03 | 4.2e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 8192 tokens (seed 8199)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 821.4 | +0.0 | 820.3-824.7 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 822.1 | +0.1 | 820.5-832.7 | noise floor | 1.8e-03 | 2.1e-06 |
| lag4 | `mega_large p16 bcm01:lag=4` | 980.1 | +19.3 | 973.1-984.0 | slower | 1.8e-03 | 5.3e-06 |
| lag8 | `mega_large p16 bcm01:lag=8` | 913.7 | +11.2 | 907.5-920.1 | slower | 1.8e-03 | 3.9e-06 |
| lag12 | `mega_large p16 bcm01:lag=12` | 850.5 | +3.5 | 849.0-853.7 | slower | 1.8e-03 | 4.0e-06 |
| lag20 | `mega_large p16 bcm01:lag=20` | 829.4 | +1.0 | 828.2-833.4 | slower | 1.8e-03 | 3.8e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 813.2 | -1.0 | 811.6-828.9 | faster | 1.8e-03 | 3.4e-06 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### 32768 tokens (seed 32775)

| variant | config | us | delta % | rounds min-max | verdict | cos ref | cos base |
|---|---|---:|---:|---:|---|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 2912.2 | +0.0 | 2773.5-2983.6 | base | 1.8e-03 | 0 |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | 2912.9 | +0.0 | 2868.2-2952.7 | noise floor | 1.8e-03 | 3.7e-06 |
| lag4 | `mega_large p16 bcm01:lag=4` | 3465.0 | +19.0 | 3440.7-3481.6 | slower | 1.8e-03 | 3.0e-06 |
| lag8 | `mega_large p16 bcm01:lag=8` | 3302.6 | +13.4 | 3275.0-3305.4 | slower | 1.8e-03 | 2.9e-06 |
| lag12 | `mega_large p16 bcm01:lag=12` | 3070.5 | +5.4 | 3026.7-3373.9 | slower | 1.8e-03 | 3.0e-06 |
| lag20 | `mega_large p16 bcm01:lag=20` | 2881.0 | -1.1 | 2723.0-3036.3 | ~ | 1.8e-03 | 2.9e-06 |
| lag32 | `mega_large p16 bcm01:lag=32` | 2728.3 | -6.3 | 2664.5-2791.0 | faster | 1.8e-03 | 2.9e-06 |
| *noise band +-3.6%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 2048 | 8192 | 32768 |
|---|---|---:|---:|---:|
| mega_large p16 bcm01 | `mega_large p16 bcm01` | 338.1 us | 821.4 us | 2912.2 us |
| mega_large p16 bcm01#2 | `mega_large p16 bcm01` | -0.2%~ | +0.1%~ | +0.0%~ |
| lag4 | `mega_large p16 bcm01:lag=4` | +6.9% | +19.3% | +19.0% |
| lag8 | `mega_large p16 bcm01:lag=8` | +8.2% | +11.2% | +13.4% |
| lag12 | `mega_large p16 bcm01:lag=12` | +4.4% | +3.5% | +5.4% |
| lag20 | `mega_large p16 bcm01:lag=20` | -0.2%~ | +1.0% | -1.1%~ |
| lag32 | `mega_large p16 bcm01:lag=32` | +0.0%~ | -1.0% | -6.3% |

